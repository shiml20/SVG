import math
import torch 
from torch.nn.parallel import DistributedDataParallel as DDP 
from tqdm import tqdm
from torchdiffeq import odeint

import os

def compute_tm(t_n, shift):
    if shift <= 0:
        return t_n
    numerator = shift * t_n
    denominator = 1 + (shift - 1) * t_n
    return numerator / denominator

def apply_shift(t, timestep_shift=1.0):
    return compute_tm(t, timestep_shift)

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def expand_t_like_x(t, x):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t


def prepare_t_seq(sample_steps, device, timestep_shift=1.0, 
                  custom_t_seq: torch.Tensor = None):
    """
    Args:
        sample_steps (int): 采样步数
        device (torch.device)
        timestep_shift (float): shift 参数（仅在没有自定义序列时用）
        custom_t_seq (torch.Tensor): 直接传入自定义时间点 [0,1] 区间张量，长度必须是 sample_steps+1
    """
    if custom_t_seq is not None:
        # 确保在 device 上 & 长度正确
        t_seq = custom_t_seq.to(device)
        assert len(t_seq) == sample_steps + 1, \
            f"custom_t_seq 长度必须是 sample_steps+1 ({sample_steps+1}), 但现在是 {len(t_seq)}"
    else:
        # 默认: linear + shift
        base = torch.linspace(0, 1, sample_steps + 1, device=device)
        t_seq = torch.tensor(
            [apply_shift(t.item(), timestep_shift) for t in base],
            device=device
        )

    # Δt 序列
    dt_seq = t_seq[1:] - t_seq[:-1]
    return t_seq, dt_seq.view(sample_steps, *([1] * (1)))  # 按需 reshape


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


    def forward(self, x, cond, timestep_shift=0.1):
        b = x.size(0)

        z1 = x
        z0 = torch.randn_like(x)
        t = torch.rand((b,), device=x.device)

        # --- apply timestep shift ---
        t = apply_shift(t, timestep_shift)

        # t = torch.zeros_like(t)
        ratio_max = 0.0
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        # ratio depends on t (linearly)
        ratio = ratio_max * texp   # shape: [b, 1, 1, ...] broadcast with z0
        ut = z1 - z0 + ratio * torch.randn_like(z0)
        zt = (1 - texp) * z0 + texp * z1
        
        # import ipdb; ipdb.set_trace()
        
        zt, t = zt.to(x.dtype), t.to(x.dtype)
        model_output, repa_out = self.model(zt, t, cond)


        terms = {}
        terms["loss"] = 0

        # if self.learn_sigma == True: 
            # model_output, _ = model_output.chunk(2, dim=1) 

        # import ipdb; ipdb.set_trace()
        if model_output.shape[2] != x.shape[2]:
            # model_output, _ = model_output.chunk(2, dim=1)
            model_output, _ = model_output.chunk(2, dim=2)
            # model_output_offset, _ = model_output_offset.chunk(2, dim=1)

        batchwise_mse = mean_flat(((model_output - ut) ** 2))
        batchwise_repa = mean_flat(((repa_out - x) ** 2))

        # batchwise_mse_offset = mean_flat(((model_output_offset - (ut - model_output)) ** 2))
        # model_output_offset = mean_flat(((model_output_offset)))

        terms["mse"] = batchwise_mse
        # terms["offset"] = batchwise_mse_offset
        # terms["model_output_offset"] = model_output_offset
        terms["loss"] += batchwise_repa * 0.5



        if "vb" in terms:
            terms["loss"] += terms["mse"].mean() + terms["vb"].mean()
            # terms["loss"] += terms["offset"].mean()
        else:
            terms["loss"] += terms["mse"].mean()
            # terms["loss"] += terms["offset"].mean()

        return terms



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
        cfg_ori = cfg


        # # --- prepare all timesteps ---
        # # 用 sample_steps+1，这样保证 i 和 i+1 都有
        # t_seq = torch.linspace(0, 1, sample_steps + 1, device=device)
        # t_seq = torch.tensor([apply_shift(t.item(), timestep_shift) for t in t_seq], device=device)
        # print(t_seq)

        # # 每个 step 的 Δt = t[i+1] - t[i]，这样支持 shift 后的非均匀步长
        # dt_seq = t_seq[1:] - t_seq[:-1]
        # dt_seq = dt_seq.view(sample_steps, *([1] * (z.ndim - 1)))  # broadcast 形状匹配 z


        # 1. 默认 (linear+shift)
        t_seq, dt_seq = prepare_t_seq(sample_steps=sample_steps, device="cuda", timestep_shift=timestep_shift)

        

        # 2. 手动指定 t_seq（非均匀）
        # manual_t_seq = torch.tensor([0.0000, 0.0040, 0.0081, 0.0122, 0.0164, 0.0206, 0.0249, 0.0292, 0.0336,
        # 0.0381, 0.0426, 0.0471, 0.0517, 0.0564, 0.0611, 0.0659, 0.0708, 0.0757,
        # 0.0807, 0.0858, 0.0909, 0.0961, 1.0000])  # len=sample_steps+1
        # sample_steps = len(manual_t_seq) - 1
        # t_seq, dt_seq = prepare_t_seq(sample_steps=sample_steps, device="cuda", custom_t_seq=manual_t_seq)

        print("t_seq:", t_seq)
        print("dt_seq:", dt_seq.squeeze())
        loop_range = tqdm(range(sample_steps), desc="Sampling") if progress else range(sample_steps)



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
            # print(t)
            vc = fn(z, t, cond)
            # vc = 1.2 * vc
            vc = 1 * vc

            if t[0] < 0.0:
                pass
                cfg = 1
            else:
                cfg = cfg_ori
                
            if null_cond is not None:
                vu = fn(z, t, null_cond)
                # channels_num = 3
                # print(model_out.shape)
                # vu, rest = vu[:, :, :channels_num], vc[:, :, channels_num:]
                # channels_num = 3
                # print(model_out.shape)
                vc = vu + cfg * (vc - vu)

            # if t[0] < 1:
            # if t[0] > 0.02 and t[0] < 0.8:
                # vc = vu

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
                    # print(( 1 - (i + 1) / len(loop_range)))
                    # z = z + torch.randn_like(z) * 0.1 * ( 1 - (i + 1) / len(loop_range))
                elif 'heun' in mode:
                    z = heun_step(z, i)
                else:
                    raise NotImplementedError(f"Mode '{mode}' is not implemented.")
            images.append(z)

        return images
