import math
import torch 
from torch.nn.parallel import DistributedDataParallel as DDP 
import torch.nn as nn
from tqdm import tqdm
from torchdiffeq import odeint
import torch.distributed as dist

import os

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

class RectifiedFlow(torch.nn.Module):
    def __init__(self, model, ln=False):
        super().__init__()
        self.ln = ln
        self.model = model

    def forward(self, x, cond, is_discrete=False):

        b = x.size(0)
        z1 = x
        z0 = torch.randn_like(x)

        if is_discrete:
            # 生成离散时间步索引（0~999随机整数）
            indices = torch.randint(0, 1000, (b,), device=x.device)
            # 转换为0~1之间的离散值
            t = indices.float() / 999.0
            texp = expand_t_like_x(t, x)
        else:
            t = torch.rand((b,)).to(x)
            texp = expand_t_like_x(t, x)

        alpha_t = texp
        sigma_t = 1 - texp
        d_alpha_t = 1
        d_sigma_t = -1
        ut = d_alpha_t * z1 + d_sigma_t * z0
        zt = alpha_t * z1 + sigma_t * z0
        model_output = self.model(zt, t, cond) 

        terms = {}
        terms["loss"] = 0

        if isinstance(model_output, tuple):
            loss_stratgy_name = model_output[1]
            if loss_stratgy_name == "Capacity_Pred":
                layer_idx_list, ones_list, pred_c_list, CapacityPred_loss_weight = model_output[2:]

                if len(ones_list) == 0:
                    pass
                else:
                    terms["cp_loss"] = 0
                    for layer_idx, ones, pred_c in zip(layer_idx_list, ones_list, pred_c_list):
                        terms[f"Capacity_Pred_loss_{layer_idx}"] = nn.BCEWithLogitsLoss()(pred_c, ones)
                        terms["loss"] +=  terms[f"Capacity_Pred_loss_{layer_idx}"]  * CapacityPred_loss_weight
                        terms["cp_loss"] += terms[f"Capacity_Pred_loss_{layer_idx}"]  * CapacityPred_loss_weight
            elif loss_stratgy_name == "TC_Glbl":
                terms["loss"] = 0
                layer_idx_list, fi_list, Pi_list = model_output[2:]
                for layer_idx, fi, Pi in zip(layer_idx_list, fi_list, Pi_list):
                    # Step 1: 将所有卡上的 fi 和 Pi 收集到当前卡
                    # ---------------------------------------------------------
                    # 获取分布式进程信息
                    world_size = dist.get_world_size()
                    rank = dist.get_rank()
                    # print(fi.shape)
                    # print(Pi.shape)

                    # 创建容器来存储所有卡上的 fi 和 Pi
                    gathered_fi = [torch.zeros_like(fi) for _ in range(world_size)]
                    # gathered_Pi = [torch.zeros_like(Pi) for _ in range(world_size)]

                    # 执行 all_gather 操作
                    dist.all_gather(gathered_fi, fi)  # 同步操作，所有卡都会收集
                    # dist.all_gather(gathered_Pi, Pi)

                    # Step 2: 将收集后的张量拼接成一个全局张量（按批次维度）
                    # ---------------------------------------------------------
                    # 假设 fi 和 Pi 的维度为 [batch_size, ...]，按第0维拼接
                    global_fi = torch.stack(gathered_fi, dim=0).contiguous().float().mean(0)
                    # global_Pi = torch.cat(gathered_Pi, dim=0).detach().contiguous().float()
                    
                    # 如果只需要在主卡上收集，可以用 dist.gather
                    # 但需要主卡指定 rank=0
                    # if rank == 0:
                    #     dist.gather(fi, dst=0)
                    #     dist.gather(Pi, dst=0)
                    # else:
                    #     dist.gather(fi, dst=0)
                    #     dist.gather(Pi, dst=0)

                    # Step 3: 打印全局形状（例如仅在主卡打印）
                    # if rank == 0:
                        # print(f"Global fi shape: {global_fi.shape}")
                        # print(f"Global Pi shape: {global_Pi.shape}")

                    balance_loss = (global_fi * Pi.float()).sum()

                    terms[f"balance_loss_{layer_idx}"] = balance_loss * 0.0001
                    terms["loss"] +=  terms[f"balance_loss_{layer_idx}"]
            elif loss_stratgy_name == "RaceAll":
                terms["loss"] = 0
                # layer_idx_list, bias_list, token_freq_list, CapacityPred_loss_weight = model_output[2:]
                # terms["RaceAll"] = [bias_list, token_freq_list]
                layer_idx_list, fi_list, Pi_list, CapacityPred_loss_weight = model_output[2:]
                for layer_idx, fi, Pi in zip(layer_idx_list, fi_list, Pi_list):
                    # Step 1: 将所有卡上的 fi 和 Pi 收集到当前卡
                    # ---------------------------------------------------------
                    # 获取分布式进程信息
                    world_size = dist.get_world_size()
                    rank = dist.get_rank()
                    # print(fi.shape)
                    # print(Pi.shape)

                    # 创建容器来存储所有卡上的 fi 和 Pi
                    gathered_fi = [torch.zeros_like(fi) for _ in range(world_size)]
                    # gathered_Pi = [torch.zeros_like(Pi) for _ in range(world_size)]
                    # 执行 all_gather 操作
                    dist.all_gather(gathered_fi, fi)  # 同步操作，所有卡都会收集
                    # dist.all_gather(gathered_Pi, Pi)
                    # Step 2: 将收集后的张量拼接成一个全局张量（按批次维度）
                    # ---------------------------------------------------------
                    # 假设 fi 和 Pi 的维度为 [batch_size, ...]，按第0维拼接
                    global_fi = torch.stack(gathered_fi, dim=0).contiguous().float().mean(0)
                    # global_Pi = torch.cat(gathered_Pi, dim=0).detach().contiguous().float()

                    balance_loss = (global_fi * Pi.float()).sum()
                    terms[f"balance_loss_{layer_idx}"] = balance_loss * CapacityPred_loss_weight
                    terms["loss"] +=  terms[f"balance_loss_{layer_idx}"]

            elif loss_stratgy_name == "RaceAll_official":
                layer_idx_list, router_similarity_loss_list, plr_value_list = model_output[2:]
                terms["loss"] = 0

                for layer_idx, router_similarity_loss, plr_value in zip(layer_idx_list, router_similarity_loss_list, plr_value_list):
                    terms[f"rsl_loss_{layer_idx}"] = router_similarity_loss * 1e-4
                    terms["loss"] += terms[f"rsl_loss_{layer_idx}"]
                    if plr_value is not None:
                        terms[f"prl_loss_{layer_idx}"] =  mean_flat(((ut - plr_value) ** 2)) * 1e-2
                        terms["loss"] += terms[f"prl_loss_{layer_idx}"]
            else:
                raise Exception("not defined training loss")

            model_output = model_output[0]

        if model_output.shape[1] != x.shape[1]:
            model_output, _ = model_output.chunk(2, dim=1)
            # model_output_offset, _ = model_output_offset.chunk(2, dim=1)

        batchwise_mse = mean_flat(((model_output - ut) ** 2))

        # batchwise_mse_offset = mean_flat(((model_output_offset - (ut - model_output)) ** 2))
        # model_output_offset = mean_flat(((model_output_offset)))

        terms["mse"] = batchwise_mse
        # terms["offset"] = batchwise_mse_offset
        # terms["model_output_offset"] = model_output_offset

        if "vb" in terms:
            terms["loss"] += terms["mse"].mean() + terms["vb"].mean()
            # terms["loss"] += terms["offset"].mean()
        else:
            terms["loss"] += terms["mse"].mean()
            # terms["loss"] += terms["offset"].mean()

        return terms

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0, progress=False, mode='euler'):
        print(f'Using {mode} Sampler')
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        device = z.device
        images = [z]
        # Use tqdm for progress bar if progress is True
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
                vu = fn(z, t, null_cond)
                vc = vu + cfg * (vc - vu)
            return vc

        def _fn(t, z):
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
                print(os.environ["cur_step"])
                if 'euler' in mode:
                    z = euler_step(z, i)
                elif 'heun' in mode:
                    z = heun_step(z, i)
                else:
                    raise NotImplementedError(f"Mode '{mode}' is not implemented.")
            images.append(z)

        return images
