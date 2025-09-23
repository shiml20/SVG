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
        model_output = self.model(zt, t, cond)


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
    def sample(self, 
            z, cond, null_cond=None, 
            sample_steps=50, cfg=2.0, progress=False, 
            mode='euler', timestep_shift=1.0, 
            cfg_mode="constant", cfg_interval=2):
        """
        Sampling function with support for different CFG modes.

        Args:
            z: latent tensor
            cond: conditional input
            null_cond: unconditional input
            sample_steps: number of steps
            cfg: cfg scale
            progress: show tqdm
            mode: integrator mode ('euler' | 'heun')
            timestep_shift: shift for timestep schedule
            cfg_mode: 'constant' | 'interval' | 'late' | 'linear'
            cfg_interval: interval for 'interval' mode
        """
        print(f'Using {mode} Sampler (cfg_mode={cfg_mode})')
        b = z.size(0)
        device = z.device
        images = [z]
        cfg_ori = cfg

        # --- prepare timesteps ---
        t_seq, dt_seq = prepare_t_seq(sample_steps=sample_steps, device=device, timestep_shift=timestep_shift)
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

        def fn_v(z, t, step_i=None):
            vc = fn(z, t, cond)

            if null_cond is not None:
                vu = fn(z, t, null_cond)

                # --- choose cfg depending on mode ---
                if cfg_mode == "constant":
                    cur_cfg = cfg_ori

                elif cfg_mode == "interval":
                    cur_cfg = cfg_ori if (step_i % cfg_interval == 0) else 1.0

                elif cfg_mode == "late":
                    ratio = (step_i + 1) / sample_steps
                    cur_cfg = cfg_ori if ratio > 0.5 else 1.0

                elif cfg_mode == "early":
                    ratio = (step_i + 1) / sample_steps
                    cur_cfg = cfg_ori if ratio < 0.5 else 1.0

                elif cfg_mode == "linear":
                    ratio = (step_i + 1) / sample_steps
                    cur_cfg = 1.0 + (cfg_ori - 1.0) * ratio

                elif "cfg_star" in cfg_mode:
                    # import ipdb; ipdb.set_trace()
                    cur_cfg = cfg_ori
                    skip = int(cfg_mode.split('-')[1])

                    B = vc.shape[0]
                    def optimized_scale(positive_flat, negative_flat):
                        # Calculate dot production
                        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

                        # Squared norm of uncondition
                        squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8

                        # st_star = v_condˆT * v_uncond / ||v_uncond||ˆ2
                        st_star = dot_product / squared_norm
                        return st_star
                
                    s_star = optimized_scale(vc.reshape(B, -1), vu.reshape(B, -1))
                    s_star = s_star.view(B, *([1] * (len(vc.shape) - 1)))

                    # vc = (1 - cur_cfg) * s_star * vu + cur_cfg * vc
                    if step_i < skip:
                        # Perform zero init
                        vc = vu * 0.
                        # vc = (1 - cur_cfg) * s_star * vu + cur_cfg * vc
                    else:
                        # Perform optimized scale
                        # noise_pred = noise_pred_uncond * st_star + guidance_scale * (noise_pred_text - noise_pred_uncond * st_star)
                        # vc = (1 - cur_cfg) * s_star * vu + cur_cfg * vc
                        vc = (1 - cur_cfg) * vu + cur_cfg * vc
                    

                elif cfg_mode == "s^2":
                    import random
                    cur_cfg = cfg_ori
                    # import ipdb; ipdb.set_trace()
                    # 生成一个包含 3 个 0～27 之间随机数的列表
                    vss = []
                    for i in range(3):
                        random_numbers = random.sample(range(28), 3)
                        print(random_numbers)
                        vs = self.model(z, t, cond, skip=random_numbers)
                        if isinstance(vs, tuple):
                            vs = vs[0]
                        if vs.shape[1] != z.shape[1]:
                            vs, _ = vs.chunk(2, dim=1)
                        if vs.shape[2] != z.shape[2]:
                            vs, _ = vs.chunk(2, dim=2)
                        vss.append(vs)

                    vc = (1 - cur_cfg) * vu + cur_cfg * vc - 0.1 * (vss[0] + vss[1] + vss[2]) / 3

                    return vc

                else:
                    raise ValueError(f"Unknown cfg_mode: {cfg_mode}")

                vc = vu + cur_cfg * (vc - vu)

            return vc

        # --- integrators ---
        def euler_step(z, i):
            t = torch.tensor([t_seq[i]] * b, device=device)
            vc = fn_v(z, t, step_i=i)
            z = z + dt_seq[i].to(z.device) * vc
            return z

        def heun_step(z, i):
            t = torch.tensor([t_seq[i]] * b, device=device)
            t_plus_1 = torch.tensor([t_seq[i + 1]] * b, device=device)
            vc = fn_v(z, t, step_i=i)
            z_tilde_plus_1 = z + dt_seq[i].to(z.device) * vc
            vc_plus_1 = fn_v(z_tilde_plus_1, t_plus_1, step_i=i)
            z = z + 0.5 * dt_seq[i].to(z.device) * (vc + vc_plus_1)
            return z

        # --- main loop ---
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

