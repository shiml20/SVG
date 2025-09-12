import numpy as np
import torch as th
import torch.nn as nn
from torchdiffeq import odeint
from functools import partial
from tqdm import tqdm
from .unipc_utils import multistep_unipc_update
from .unipc_diffusion_utils import unipc_diffusion

class sde:
    """SDE solver class"""
    def __init__(
        self, 
        drift,
        diffusion,
        *,
        t0,
        t1,
        num_steps,
        sampler_type,
    ):
        assert t0 < t1, "SDE sampler has to be in forward time"

        self.num_timesteps = num_steps
        self.t = th.linspace(t0, t1, num_steps)
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type

    def __Euler_Maruyama_step(self, x, mean_x, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        t = th.ones(x.size(0)).to(x) * t
        dw = w_cur * th.sqrt(self.dt)
        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        mean_x = x + drift * self.dt
        x = mean_x + th.sqrt(2 * diffusion) * dw
        return x, mean_x
    
    def __Heun_step(self, x, _, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        dw = w_cur * th.sqrt(self.dt)
        t_cur = th.ones(x.size(0)).to(x) * t
        diffusion = self.diffusion(x, t_cur)
        xhat = x + th.sqrt(2 * diffusion) * dw
        K1 = self.drift(xhat, t_cur, model, **model_kwargs)
        xp = xhat + self.dt * K1
        K2 = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
        return xhat + 0.5 * self.dt * (K1 + K2), xhat # at last time point we do not perform the heun step

    def __forward_fn(self):
        """TODO: generalize here by adding all private functions ending with steps to it"""
        sampler_dict = {
            "Euler": self.__Euler_Maruyama_step,
            "Heun": self.__Heun_step,
        }

        try:
            sampler = sampler_dict[self.sampler_type]
        except:
            raise NotImplementedError("Smapler type not implemented.")
    
        return sampler

    def sample(self, init, model, **model_kwargs):
        """forward loop of sde"""
        x = init
        mean_x = init 
        samples = []
        sampler = self.__forward_fn()
        for ti in self.t[:-1]:
            with th.no_grad():
                x, mean_x = sampler(x, mean_x, ti, model, **model_kwargs)
                samples.append(x)

        return samples

class ode:
    """ODE solver class"""
    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol,
        rtol,
        time_shifting_factor
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.drift = drift
        self.t = th.linspace(t0, t1, num_steps)
        print(time_shifting_factor)
        if time_shifting_factor:
            self.t = self.t / (self.t + time_shifting_factor - time_shifting_factor * self.t)
        
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type
        
    def sample_unipc(self, x, model, ratio, **model_kwargs):
        lower_order_final = True
        
        if "unipc_diffusion" in self.sampler_type:
            _, unipc_variant, unipc_order, mode, disable_corrector_all_steps = self.sampler_type.split("-")
            unipc_order = int(unipc_order)
            disable_corrector_all_steps = bool(int(disable_corrector_all_steps))
            self.t[0] = 1e-3
            self.t[-1] = 1 - 1e-3
            
        elif "unipc" in self.sampler_type:
            _, unipc_variant, unipc_order = self.sampler_type.split("-")
            unipc_order = int(unipc_order)
        else:
            unipc_order = None
            
        device = x[0].device if isinstance(x, tuple) else x.device
        self.NFE_COUNT = 0
        
        def _fn(t, x):
            
            # x = x.to(th.bfloat16)         
            t = th.ones(x[0].size(0)).to(device) * t if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t
            model_output = self.drift(x, t, model, **model_kwargs)
            self.NFE_COUNT += 1
            # model_output = model_output.to(th.float32)
            
            return model_output

        t = self.t.to(device)
        self.delta_t = t[1] - t[0]
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        # import pdb; pdb.set_trace()
        x_i = x
        samples = [x]
        v = []
        d_i_cache = None
        step = -1
        d_i_plus_1_cache = None
        # import json
        # self.dc_ratios = json.load(open(f"dc_solver/dc_ratios_NFE{len(t) - 1}.json"))

        def euler_predictor(i, xi):
            d_i = _fn(t[i], xi)
            xi = xi + self.delta_t * d_i
            return xi, d_i

        def heun_predictor(i, xi):
            d_i = _fn(t[i], xi)
            x_tilde_i_plus_1 = xi + self.delta_t * d_i
            d_i_plus_1 = _fn(t[i+1], x_tilde_i_plus_1)
            xi = xi + 1/2 * self.delta_t * (d_i + d_i_plus_1)
            return xi, d_i

        def unipc(x, t, t_prev_list, model_prev_list, disable_corrector=False, order=1):
            # NOTE: t_{i-1}: t[i] / t_prev_list[-1]
            # predictor
            assert unipc_order is not None
            x_t, model_t = multistep_unipc_update(x, model_prev_list, t_prev_list, t, order, \
                variant=unipc_variant, use_corrector=not disable_corrector, model_fn=lambda x, t: _fn(t, x))
            return x_t, model_t

        def unipc_diffusion_version(x, t, t_prev_list, model_prev_list, disable_corrector=False, order=1, mode='data_prediction', ratio=1):
            # NOTE: t_{i-1}: t[i] / t_prev_list[-1]
            # predictor
            assert unipc_order is not None
            x_t, model_t = unipc_diffusion(x, model_prev_list, t_prev_list, t, order, \
                variant=unipc_variant, use_corrector=not disable_corrector, model_fn=lambda x, t: _fn(t, x), 
                mode=mode, ratio=ratio)
            return x_t, model_t           



        model_prev_list = []
        t_prev_list = []
        for seq, i in enumerate(range(len(t) - 1)):
            self.delta_t = t[seq+1] - t[seq]
            
            if self.sampler_type == "euler":
                x_i, d_i = euler_predictor(i, x_i)
            elif self.sampler_type == "heun":
                if seq == len(t) - 2:
                    x_i, d_i = euler_predictor(i, x_i)
                    # print('1')
                else:
                    x_i, d_i = heun_predictor(i, x_i)
                    # print('2')

            elif "unipc_diffusion" in self.sampler_type:
                if disable_corrector_all_steps:
                    print('disable_corrector_all_steps', disable_corrector_all_steps)
                    disable_corrector = disable_corrector_all_steps
                else:
                    disable_corrector = seq == len(t) - 2
                    
                # prepare the list
                if len(t_prev_list) == 0:
                    t_prev_list.append(t[i])
                    if mode == 'data_prediction':
                        model_prev0_convert = x_i + (1 - t[i]) * _fn(t[i], x_i)
                    else:
                        model_prev0_convert = x_i - t[i] * _fn(t[i], x_i)
                    model_prev_list.append(model_prev0_convert)
                
                order = min(unipc_order, len(model_prev_list))
                
                if lower_order_final:
                    step_order = min(order, len(t) - 1 - seq)
                else:
                    step_order = order
                # print("this step order:", step_order)
                
                x_i, d_i_plus_1_cache = unipc_diffusion_version(x_i, t[i+1], t_prev_list, model_prev_list, disable_corrector=disable_corrector, order=step_order, mode=mode,
                                                                ratio=ratio)
                if d_i_plus_1_cache is None and seq < len(t) - 2:
                    if mode == 'data_prediction':
                        d_i_plus_1_cache = x_i + (1 - t[i+1]) * _fn(t[i+1], x_i)
                    else:
                        d_i_plus_1_cache = x_i - t[i+1] * _fn(t[i+1], x_i)

                model_prev_list.append(d_i_plus_1_cache)
                t_prev_list.append(t[i+1])
                
            elif "unipc" in self.sampler_type:
                disable_corrector = seq == len(t) - 2
                # prepare the list
                if len(t_prev_list) == 0:
                    t_prev_list.append(t[i])
                    model_prev_list.append(_fn(t[i], x_i))
                order = min(unipc_order, len(model_prev_list))
                
                if lower_order_final:
                    step_order = min(order, len(t) - 1 - seq)
                else:
                    step_order = order
                # print("this step order:", step_order)
                
                x_i, d_i_plus_1_cache = unipc(x_i, t[i+1], t_prev_list, model_prev_list, disable_corrector=disable_corrector, order=step_order)
                
                model_prev_list.append(d_i_plus_1_cache)
                t_prev_list.append(t[i+1])

            else:
                raise NotImplementedError()
            
            samples.append(x_i)
        print("NFE:", self.NFE_COUNT)
        return samples
    

    def sample(self, x, model, ratio=1, **model_kwargs):
        if self.sampler_type in ["euler", "heun"] or "unipc" in self.sampler_type:
            return self.sample_unipc(x, model, ratio, **model_kwargs)
        
        device = x[0].device if isinstance(x, tuple) else x.device
        def _fn(t, x):
            
            t = th.ones(x[0].size(0)).to(device) * t if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t
            model_output = self.drift(x, t, model, **model_kwargs)
            
            return model_output

        t = self.t.to(device)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        samples = odeint(
            _fn,
            x,
            t,
            method=self.sampler_type,
            atol=atol,
            rtol=rtol
        )
        return samples