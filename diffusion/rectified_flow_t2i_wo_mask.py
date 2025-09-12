import math
import torch 
from torch.nn.parallel import DistributedDataParallel as DDP 
import torch.nn as nn
from tqdm import tqdm
from torchdiffeq import odeint
import torch.distributed as dist
import os
from contextlib import contextmanager
from typing import Optional, Tuple

def mean_flat(x: torch.Tensor) -> torch.Tensor:
    """Take the mean over all non-batch dimensions."""
    return x.mean(dim=tuple(range(1, x.ndim)))

def expand_t_like_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape time vector to broadcastable dimensions."""
    return t.view(-1, *([1] * (x.ndim - 1)))

class RectifiedFlow(nn.Module):
    def __init__(self, model: nn.Module, ln: bool = False):
        super().__init__()
        self.model = model
        self.ln = ln
        self._setup_solver_defaults()

    def _setup_solver_defaults(self):
        """Initialize default solver parameters"""
        self.solver_config = {
            'atol': 1e-6,
            'rtol': 1e-3,
            'min_step': 0.1  # For adaptive solvers
        }

    def forward(self, 
                x: torch.Tensor, 
                y: torch.Tensor, 
                y_mask: Optional[torch.Tensor]) -> dict:
        """Main training forward pass with improved distributed handling"""
        b = x.size(0)
        device = x.device
        
        # Initialize latent variables
        z1, z0 = x, torch.randn_like(x)
        t = torch.rand(b, device=device)
        texp = expand_t_like_x(t, x)
        
        # Compute velocity components
        alpha_t, sigma_t = texp, 1 - texp
        d_alpha_t, d_sigma_t = 1.0, -1.0
        ut = d_alpha_t * z1 + d_sigma_t * z0
        zt = alpha_t * z1 + sigma_t * z0
        
        # Model forward pass
        model_output = self.model(zt, t, y, None)

        return self._compute_losses(model_output, ut, x, zt, t, y_mask)

    def _compute_losses(self,
                    model_output: Tuple[torch.Tensor, ...],
                    ut: torch.Tensor,
                    x: torch.Tensor,
                    zt: torch.Tensor,
                    t: torch.Tensor,
                    y_mask: Optional[torch.Tensor]) -> dict:
        """Unified loss computation with improved distributed handling"""
        terms = {"loss": 0.0}
        
        # Handle different output types
        if isinstance(model_output, tuple):
            loss_strategy = model_output[1]
            
            if loss_strategy == "Capacity_Pred":
                terms = self._capacity_prediction_loss(model_output, terms)

            elif loss_strategy == "TC_Glbl":
                terms = self._distributed_balance_loss(model_output, terms)
            else:
                raise ValueError(f"Unknown loss strategy: {loss_strategy}")

            model_output = model_output[0]
    
        # Base MSE calculation
        model_output = self._maybe_split_output(model_output, x)
        batch_mse = mean_flat((model_output - ut) ** 2)
        
        terms["mse"] = batch_mse
        terms["loss"] += batch_mse.mean()

        return terms

    def _maybe_split_output(self, 
                            output: torch.Tensor, 
                            x: torch.Tensor) -> torch.Tensor:
        """Handle channel-split outputs"""
        return output if output.shape[1] == x.shape[1] else output.chunk(2, dim=1)[0]

    def _capacity_prediction_loss(self, 
                                model_output: Tuple[torch.Tensor, ...],
                                terms) -> dict:
        """Capacity prediction loss computation"""
        # import ipdb; ipdb.set_trace()
        # print(len(model_output))
        layer_idx_list, ones_list, pred_c_list, weight = model_output[2:]
        
        for idx, (layer_idx, ones, pred_c) in enumerate(zip(layer_idx_list, ones_list, pred_c_list)):
            loss = nn.BCEWithLogitsLoss()(pred_c, ones) * weight
            terms[f"Capacity_Pred_loss_{layer_idx}"] = loss
            terms["loss"] += loss
            terms.setdefault("cp_loss", 0.0)
            terms["cp_loss"] += loss
            
        return terms

    def _distributed_balance_loss(self, 
                                model_output: Tuple[torch.Tensor, ...],
                                terms) -> dict:
        """Distributed balance loss computation with optimized communication"""
        layer_idx_list, fi_list, Pi_list = model_output[2:]
        
        for layer_idx, fi, Pi in zip(layer_idx_list, fi_list, Pi_list):
            # Optimized all_gather implementation
            gathered_fi = [torch.empty_like(fi) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_fi, fi)
            
            # Use stack instead of cat for better dimension management
            global_fi = torch.stack(gathered_fi).mean(dim=0)
            
            # Use in-place operations where possible
            balance_loss = (global_fi * Pi).sum() * 1e-4
            terms[f"balance_loss_{layer_idx}"] = balance_loss
            terms["loss"] += balance_loss
            
        return terms

    @torch.no_grad()
    def sample(self,
                z: torch.Tensor,
                y: torch.Tensor,
                y_mask: torch.Tensor,
                null_cond: Optional[torch.Tensor] = None,
                sample_steps: int = 50,
                cfg: float = 2.0,
                progress: bool = False,
                mode: str = 'euler') -> list:
        """Enhanced sampling method with improved numerical stability"""
        self._validate_inputs(z, y)
        b = z.size(0)
        device = z.device
        
        # Initialize sampling
        images = [z]
        # dt = torch.full((b,), 1.0/sample_steps, device=device).view_as(z)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])

        # Select sampling strategy
        if 'torchdiff' in mode:
            images.append(self._adaptive_sampling(z, mode, sample_steps, y, y_mask, null_cond, cfg))
        else:
            images.append(self._fixed_step_sampling(z, dt, sample_steps, mode, y, y_mask, null_cond, cfg, progress))
            
        return images

    def _validate_inputs(self, z: torch.Tensor,  y: torch.Tensor):
        """Validate input tensor dimensions"""
        assert y.size(0) == z.size(0), f"Batch size mismatch: y{y.size(0)} vs z{z.size(0)}"

    def _adaptive_sampling(self,
                         z: torch.Tensor,
                         mode: str,
                         steps: int,
                         y: torch.Tensor,
                         y_mask: torch.Tensor,
                         null_cond: Optional[torch.Tensor],
                         cfg: float) -> torch.Tensor:
        """Adaptive step-size sampling"""
        solver_type = mode.split('-', 1)[1]
        t_eval = torch.linspace(0, 1, steps+1, device=z.device)
        
        def ode_func(t, z):
            return self._velocity_function(z, t, y, y_mask, null_cond, cfg)
            
        return odeint(ode_func, z, t_eval,
                      method=solver_type,
                      atol=self.solver_config['atol'],
                      rtol=self.solver_config['rtol'])[-1]

    def _fixed_step_sampling(self,
                           z: torch.Tensor,
                           dt: torch.Tensor,
                           steps: int,
                           mode: str,
                           y: torch.Tensor,
                           y_mask: torch.Tensor,
                           null_cond: Optional[torch.Tensor],
                           cfg: float,
                           progress: bool) -> torch.Tensor:
        """Fixed step-size sampling with progress control"""
        step_fn = self._get_step_function(mode)
        loop = range(steps)
        
        with self._progress_context(loop, progress, desc=f"{mode} Sampling") as pbar:
            for i in pbar:
                z = step_fn(z, i/steps, dt, y, y_mask, null_cond, cfg)
                os.environ["cur_step"] = f"{i:03d}"
                
        return z

    @contextmanager
    def _progress_context(self, iterable, active, **kwargs):
        """Context manager for progress bars"""
        if active:
            with tqdm(iterable, **kwargs) as pbar:
                yield pbar
        else:
            yield iterable

    def _get_step_function(self, mode: str):
        """Get numerical integration step function"""
        return {
            'euler': self._euler_step,
            'heun': self._heun_step
        }.get(mode, self._default_step)

    def _default_step(self, *args):
        raise NotImplementedError("Unsupported integration method")

    def _velocity_function(self,
                         z: torch.Tensor,
                         t: float,
                         y: torch.Tensor,
                         y_mask: torch.Tensor,
                         null_cond: Optional[torch.Tensor],
                         cfg: float) -> torch.Tensor:
        """Compute velocity field with optional conditioning"""
        t_tensor = torch.full((z.size(0),), t, device=z.device)
        v_cond = self.model(z, t_tensor, y, y_mask)
        # v_cond = self.model(z, t_tensor, y)
        
        if v_cond.shape[1] != z.shape[1]:
            v_cond, _ = v_cond.chunk(2, dim=1)

        if null_cond is not None and cfg > 1.0:
            v_uncond = self.model(z, t_tensor, null_cond, y_mask)

            if v_uncond.shape[1] != z.shape[1]:
                v_uncond, _ = v_uncond.chunk(2, dim=1)

            return v_uncond + cfg * (v_cond - v_uncond)

        return v_cond

    def _euler_step(self,
                  z: torch.Tensor,
                  t: float,
                  dt: torch.Tensor,
                  y: torch.Tensor,
                  y_mask: torch.Tensor,
                  null_cond: Optional[torch.Tensor],
                  cfg: float) -> torch.Tensor:
        """Euler integration step"""
        return z + dt * self._velocity_function(z, t, y, y_mask, null_cond, cfg)

    def _heun_step(self,
                 z: torch.Tensor,
                 t: float,
                 dt: torch.Tensor,
                 y: torch.Tensor,
                 y_mask: torch.Tensor,
                 null_cond: Optional[torch.Tensor],
                 cfg: float) -> torch.Tensor:
        """Heun's method (2nd order Runge-Kutta)"""
        k1 = self._velocity_function(z, t, y, y_mask, null_cond, cfg)
        k2 = self._velocity_function(z + dt*k1, t + dt.item(), y, y_mask, null_cond, cfg)
        return z + 0.5 * dt * (k1 + k2)