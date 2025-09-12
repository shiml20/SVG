import math
import torch 
from torch.nn.parallel import DistributedDataParallel as DDP 
from tqdm import tqdm
from torchdiffeq import odeint

import os

class RectifiedFlow(torch.nn.Module):
    def __init__(self, model, ln=False):
        super().__init__()
        self.model = model
        self.ln = ln
        self.stratified = False 
        if isinstance(model, DDP):
            self.learn_sigma = model.module.learn_sigma 
        else:
            self.learn_sigma = model.learn_sigma 

    def forward(self, x, cond):

        b = x.size(0)
        # import pdb; pdb.set_trace()
       
        z1 = x
        z0 = torch.randn_like(x)
        t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        ut = z1 - z0

        # 0 is noise 1 is data
        zt = (1 - texp) * z0 + texp * z1

        # make t, zt into same dtype as x
        zt, t = zt.to(x.dtype), t.to(x.dtype)

        model_output = self.model(zt, t, cond, ) 

        terms = {}
        terms["loss"] = 0

        if isinstance(model_output, tuple):
            print('here')
            # terms["total_aux_loss"] = sum(model_output[1])
            # terms["balance_loss"] = sum(model_output[2])
            # terms["router_z_loss"] = sum(model_output[3])

            loss_stratgy_name = model_output[1]
            # print(loss_stratgy_name)
            if loss_stratgy_name == "router_distill":
                affinity_list, index_list, sparse_loss_weight = model_output[2:]
                for i, (affinity, index) in enumerate(zip(affinity_list, index_list)):
                    mask = th.zeros_like(affinity)
                    mask.scatter_(-1, index, 1)
                    terms[f"sparse_loss_{i}"] = mean_flat(th.abs(affinity-mask)).mean()
                    terms["loss"] +=  mean_flat(th.abs(affinity-mask)) * sparse_loss_weight
            
            elif loss_stratgy_name == "Capacity_Pred":
                terms["cp_loss"] = 0
                layer_idx_list, ones_list, pred_c_list, CapacityPred_loss_weight = model_output[2:]
                for layer_idx, ones, pred_c in zip(layer_idx_list, ones_list, pred_c_list):
                    # ones = ones.view(-1,1)
                    # pred_c = pred_c.view(-1, 1)
                    terms[f"Capacity_Pred_loss_{layer_idx}"] = nn.BCEWithLogitsLoss()(pred_c, ones)
                    terms["loss"] +=  terms[f"Capacity_Pred_loss_{layer_idx}"]  * CapacityPred_loss_weight
                    terms["cp_loss"] += terms[f"Capacity_Pred_loss_{layer_idx}"]  * CapacityPred_loss_weight

            elif loss_stratgy_name == "dynamic_moe":
                layer_idx_list, balance_loss_list, dynamic_loss_list, balance_loss_weight, dynamic_loss_weight = model_output[2:]
                for layer_idx, balance_loss, dynamic_loss in zip(layer_idx_list, balance_loss_list, dynamic_loss_list):
                    terms[f"balance_loss_{layer_idx}"] = balance_loss 
                    terms["loss"] +=  terms[f"balance_loss_{layer_idx}"] * balance_loss_weight
                    terms[f"dynamic_loss_{layer_idx}"] = dynamic_loss 
                    terms["loss"] +=  terms[f"dynamic_loss_{layer_idx}"] * dynamic_loss_weight
            elif loss_stratgy_name == "MovingThreshold":
                layer_idx_list, balance_loss_list, balance_loss_weight, real_capacity_list = model_output[2:]
                # for layer_idx, balance_loss, real_capacity in zip(layer_idx_list, balance_loss_list, real_capacity_list):
                    # terms[f"balance_loss_{layer_idx}"] = balance_loss 
                    # terms["loss"] +=  terms[f"balance_loss_{layer_idx}"] * balance_loss_weight
                    # terms[f"real_capacity_{layer_idx}"] = real_capacity 
            elif loss_stratgy_name == "MovingThreshold_wLBL":
                layer_idx_list, balance_loss_list, balance_loss_weight, real_capacity_list = model_output[2:]
                for layer_idx, balance_loss, real_capacity in zip(layer_idx_list, balance_loss_list, real_capacity_list):
                    terms[f"balance_loss_{layer_idx}"] = balance_loss 
                    terms["loss"] +=  terms[f"balance_loss_{layer_idx}"] * balance_loss_weight
                    terms[f"real_capacity_{layer_idx}"] = real_capacity 
            else:
                raise Exception("not defined training loss")

            model_output = model_output[0]


        if self.learn_sigma == True: 
            model_output, _ = model_output.chunk(2, dim=1) 
        batchwise_mse = torch.mean(((model_output - ut) ** 2), dim=list(range(1, len(x.shape))))

        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]

        terms["mse"] = batchwise_mse

        if "vb" in terms:
            terms["loss"] += terms["mse"].mean() + terms["vb"].mean()
        else:
            terms["loss"] += terms["mse"].mean()

        # import pdb; pdb.set_trace()

        return terms, {"batchwise_loss": ttloss}


    @torch.no_grad()
    def sample_v1(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0, progress=False, mode='euler'):
        print(f'Using {mode} Sampler')
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        device = z.device
        images = [z]
        # import pdb; pdb.set_trace()
        
        # Use tqdm for progress bar if progress is True
        # loop_range = tqdm(range(sample_steps, 0, -1), desc="Sampling") if progress else range(sample_steps, 0, -1)
        loop_range = tqdm(range(0, sample_steps, 1), desc="Sampling") if progress else range(0, sample_steps, 1)

        def fn(z, t, cond):
            vc = self.model(z, t, cond)
            if isinstance(vc, tuple):
                vc = vc[0]
            if vc.shape[1] != z.shape[1]:
                vc, _ = vc.chunk(2, dim=1)
            return vc

        def fn_v(z, t):
            vc = fn(z, t, cond)
            if null_cond is not None:
                # print('here')
                vu = fn(z, t, null_cond)
                vc = vu + cfg * (vc - vu)
            return vc

        def _fn(t, z):
            # import pdb; pdb.set_trace()
            t = torch.tensor([t] * b).to(z.device)
            return fn_v(z, t)

        def euler_step(z, i):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)
            vc = fn_v(z, t)
            z = z + dt * vc
            return z

        def heun_step(z, i):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)
            t_plus_1 = (i+1) / sample_steps
            t_plus_1 = torch.tensor([t_plus_1] * b).to(z.device)
            vc = fn_v(z, t)
            z_tilde_plus_1 = z + dt * vc
            vc_plus_1 = fn_v(z_tilde_plus_1, t_plus_1)
            z = z + 1/2 * dt * (vc + vc_plus_1)
            return z


        if 'torchdiff' in mode:
            mode = mode.split('-')[-1]
            self.atol = 1e-6
            self.rtol = 1e-3
            atol = [self.atol] * len(z) if isinstance(z, tuple) else [self.atol]
            rtol = [self.rtol] * len(z) if isinstance(z, tuple) else [self.rtol]
            t = torch.linspace(0, 1, sample_steps).to(z.device)

            samples = odeint(
                _fn,
                z,
                t,
                method=mode,
                atol=atol,
                rtol=rtol
            )
            images.append(samples[-1])

        else:
            for i in loop_range:
                os.environ["cur_step"] = f"{i:003d}"
                # 主逻辑
                if 'euler' in mode:
                    z = euler_step(z, i)
                elif 'heun' in mode:
                    z = heun_step(z, i)
                else:
                    raise NotImplementedError(f"Mode '{mode}' is not implemented.")
            images.append(z)

        return images

    @torch.no_grad()
    def sample_with_xps(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0, progress=False):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]

        # Use tqdm for progress bar if progress is True
        # loop_range = tqdm(range(sample_steps, 0, -1), desc="Sampling with XPS") if progress else range(sample_steps, 0, -1)
        loop_range = tqdm(range(0, sample_steps, 1), desc="Sampling") if progress else range(0, sample_steps, 1)

        for i in loop_range:
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = self.model(z, t, cond)
            if self.learn_sigma == True:
                vc, _ = vc.chunk(2, dim=1)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                if self.learn_sigma == True:
                    vu, _ = vu.chunk(2, dim=1)
                vc = vu + cfg * (vc - vu)

            x = z + i * dt * vc
            z = z + dt * vc
            images.append(x)
        return images


    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0, progress=False, mode='euler', timestep_shift=1.0):
        print(f'Using {mode} Sampler')
        b = z.size(0)
        device = z.device
        images = [z]

        # --- timestep shift helper ---
        def compute_tm(t_n, shift):
            if shift <= 0:
                return t_n
            numerator = shift * t_n
            denominator = 1 + (shift - 1) * t_n
            return numerator / denominator

        def apply_shift(t):
            return compute_tm(t, timestep_shift)

        # --- prepare all timesteps ---
        # 用 sample_steps+1，这样保证 i 和 i+1 都有
        t_seq = torch.linspace(0, 1, sample_steps + 1, device=device)
        t_seq = torch.tensor([apply_shift(t.item()) for t in t_seq], device=device)
        print(t_seq)
        loop_range = tqdm(range(sample_steps), desc="Sampling") if progress else range(sample_steps)

        # 每个 step 的 Δt = t[i+1] - t[i]，这样支持 shift 后的非均匀步长
        dt_seq = t_seq[1:] - t_seq[:-1]
        dt_seq = dt_seq.view(sample_steps, *([1] * (z.ndim - 1)))  # broadcast 形状匹配 z

        # --- base functions ---
        def fn(z, t, cond):
            vc = self.model(z, t, cond)
            if isinstance(vc, tuple):
                vc = vc[0]
            if vc.shape[1] != z.shape[1]:
                vc, _ = vc.chunk(2, dim=1)
            if vc.shape[2] != z.shape[2]:
                vc, _ = vc.chunk(2, dim=2)
            return vc

        def fn_v(z, t):
            vc = fn(z, t, cond)
            if null_cond is not None:
                vu = fn(z, t, null_cond)
                # channels_num = 3
                # print(model_out.shape)
                # vu, rest = vu[:, :, :channels_num], vc[:, :, channels_num:]
                # channels_num = 3
                # print(model_out.shape)
                vc = vu + cfg * (vc - vu)
            return vc

        def _fn(t, z):
            t = torch.tensor([t] * b, device=device)
            return fn_v(z, t)

        # --- integrators ---
        def euler_step(z, i):
            t = torch.tensor([t_seq[i]] * b, device=device)
            vc = fn_v(z, t)
            # import ipdb; ipdb.set_trace()
            z = z + dt_seq[i].to(z.device) * vc
            return z

        def heun_step(z, i):
            t = torch.tensor([t_seq[i]] * b, device=device)
            t_plus_1 = torch.tensor([t_seq[i + 1]] * b, device=device)
            vc = fn_v(z, t)
            z_tilde_plus_1 = z + dt_seq[i].to(z.device) * vc
            vc_plus_1 = fn_v(z_tilde_plus_1, t_plus_1)
            z = z + 0.5 * dt_seq[i].to(z.device) * (vc + vc_plus_1)
            return z

        # --- main loop ---
        if 'torchdiff' in mode:
            mode = mode.split('-')[-1]
            self.atol = 1e-6
            self.rtol = 1e-3
            atol = [self.atol] * len(z) if isinstance(z, tuple) else [self.atol]
            rtol = [self.rtol] * len(z) if isinstance(z, tuple) else [self.rtol]

            samples = odeint(
                _fn,
                z,
                t_seq,
                method=mode,
                atol=atol,
                rtol=rtol
            )
            images.append(samples[-1])
        else:
            for i in loop_range:
                os.environ["cur_step"] = f"{i:003d}"
                if 'euler' in mode:
                    z = euler_step(z, i)
                elif 'heun' in mode:
                    z = heun_step(z, i)
                else:
                    raise NotImplementedError(f"Mode '{mode}' is not implemented.")
            images.append(z)

        return images
