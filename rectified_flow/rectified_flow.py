import os
import math
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm


# ----------------------------
# Utility Functions
# ----------------------------

def compute_tm(t_n, shift):
    """Compute shifted timestep using rational transformation."""
    if shift <= 0:
        return t_n
    return (shift * t_n) / (1 + (shift - 1) * t_n)


def apply_shift(t, timestep_shift=1.0):
    """Apply timestep shift transformation."""
    return compute_tm(t, timestep_shift)


def mean_flat(x):
    """Compute mean over all non-batch dimensions."""
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def expand_t_like_x(t, x):
    """Expand a 1D time tensor to match x's dimensions for broadcasting."""
    dims = [1] * (len(x.size()) - 1)
    return t.view(t.size(0), *dims)


def prepare_t_seq(sample_steps, device, timestep_shift=1.0, custom_t_seq=None):
    """
    Prepare the timestep sequence and its corresponding deltas.
    
    Args:
        sample_steps (int): Number of sampling steps.
        device (torch.device): Target device.
        timestep_shift (float): Shift applied to timestep schedule.
        custom_t_seq (torch.Tensor): Optional custom time sequence in [0, 1],
                                     must have length (sample_steps + 1).
    Returns:
        t_seq: Tensor of timesteps.
        dt_seq: Tensor of timestep differences (Δt).
    """
    if custom_t_seq is not None:
        t_seq = custom_t_seq.to(device)
        assert len(t_seq) == sample_steps + 1, \
            f"custom_t_seq length must be {sample_steps + 1}, got {len(t_seq)}"
    else:
        base = torch.linspace(0, 1, sample_steps + 1, device=device)
        t_seq = torch.tensor([apply_shift(t.item(), timestep_shift) for t in base], device=device)

    dt_seq = t_seq[1:] - t_seq[:-1]
    return t_seq, dt_seq.view(sample_steps, 1)


# ----------------------------
# Rectified Flow Class
# ----------------------------

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
        """
        Training forward pass for rectified flow.
        Args:
            x: Input tensor.
            cond: Conditioning input.
            timestep_shift: Shift applied to timesteps.
        Returns:
            dict: Containing 'loss' and 'mse'.
        """
        b = x.size(0)
        z1 = x
        z0 = torch.randn_like(x)
        t = torch.rand((b,), device=x.device)
        t = apply_shift(t, timestep_shift)

        t_expanded = expand_t_like_x(t, x)
        ratio = torch.zeros_like(t_expanded)
        ut = z1 - z0 + ratio * torch.randn_like(z0)
        zt = (1 - t_expanded) * z0 + t_expanded * z1

        zt, t = zt.to(x.dtype), t.to(x.dtype)
        model_output = self.model(zt, t, cond)

        # Align shape if needed
        if model_output.shape[2] != x.shape[2]:
            model_output, _ = model_output.chunk(2, dim=2)

        mse = mean_flat((model_output - ut) ** 2)
        loss = mse.mean()

        return {"loss": loss, "mse": mse}

    @torch.no_grad()
    def sample(
        self,
        z,
        cond,
        null_cond=None,
        sample_steps=50,
        cfg=2.0,
        progress=False,
        mode="euler",
        timestep_shift=1.0,
        cfg_mode="constant",
        cfg_interval=2,
    ):
        """
        Sampling function with flexible CFG scheduling and integrators.
        
        Args:
            z: Initial latent tensor.
            cond: Conditional input.
            null_cond: Unconditional input (for CFG).
            sample_steps (int): Number of steps.
            cfg (float): CFG guidance scale.
            progress (bool): Whether to show a tqdm progress bar.
            mode (str): Integration mode ('euler' or 'heun').
            timestep_shift (float): Shift applied to timestep schedule.
            cfg_mode (str): CFG strategy ('constant', 'interval', 'late', 'linear', etc.).
            cfg_interval (int): Step interval used when cfg_mode='interval'.
        """
        print(f"Using {mode} sampler (cfg_mode={cfg_mode})")
        b = z.size(0)
        device = z.device
        cfg_ori = cfg

        t_seq, dt_seq = prepare_t_seq(sample_steps, device, timestep_shift)
        loop_range = tqdm(range(sample_steps), desc="Sampling") if progress else range(sample_steps)

        def fn(z, t, cond):
            """Forward through the model."""
            vc = self.model(z, t, cond)
            if isinstance(vc, tuple):
                vc = vc[0]
            if vc.shape[1] != z.shape[1]:
                vc, _ = vc.chunk(2, dim=1)
            if vc.shape[2] != z.shape[2]:
                vc, _ = vc.chunk(2, dim=2)
            return vc

        def fn_v(z, t, step_i=None):
            """Compute velocity field with classifier-free guidance."""
            vc = fn(z, t, cond)
            if null_cond is None:
                return vc

            vu = fn(z, t, null_cond)

            # Select CFG schedule
            if cfg_mode == "constant":
                cur_cfg = cfg_ori
            elif cfg_mode == "late":
                ratio = (step_i + 1) / sample_steps
                cur_cfg = cfg_ori if ratio > 0.5 else 1.0
            elif "early" in cfg_mode:
                ratio = float(cfg_mode.split('-')[1])
                cur_cfg = cfg_ori if float(t[0]) > ratio else 1.0
            elif cfg_mode == "linear":
                ratio = (step_i + 1) / sample_steps
                cur_cfg = 1.0 + (cfg_ori - 1.0) * ratio
            elif "cfg_star" in cfg_mode:
                cur_cfg = cfg_ori
                skip = int(cfg_mode.split('-')[1])
                ratio = float(cfg_mode.split('-')[2])
                if step_i < skip:
                    vc = -vu * ratio
                else:
                    vc = (1 - cur_cfg) * vu + cur_cfg * vc
                
                return vc
            else:
                raise ValueError(f"Unknown cfg_mode: {cfg_mode}")

            return vu + cur_cfg * (vc - vu)

        # Integrators
        def euler_step(z, i):
            t = torch.full((b,), t_seq[i], device=device)
            vc = fn_v(z, t, step_i=i)
            return z + dt_seq[i].to(z.device) * vc

        def heun_step(z, i):
            t = torch.full((b,), t_seq[i], device=device)
            t_next = torch.full((b,), t_seq[i + 1], device=device)
            vc = fn_v(z, t, step_i=i)
            z_tilde = z + dt_seq[i].to(z.device) * vc
            vc_next = fn_v(z_tilde, t_next, step_i=i)
            return z + 0.5 * dt_seq[i].to(z.device) * (vc + vc_next)

        # Sampling loop
        for i in loop_range:
            os.environ["cur_step"] = f"{i:03d}"
            if mode == "euler":
                z = euler_step(z, i)
            elif mode == "heun":
                z = heun_step(z, i)
            else:
                raise NotImplementedError(f"Unsupported mode: {mode}")

        return z
