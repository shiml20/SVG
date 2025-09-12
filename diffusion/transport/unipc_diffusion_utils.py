import torch
import numpy as np
from typing import List, Optional, Tuple, Union


def model_output_convert(x_t, v_t, t, mode='data_prediction'):
    if mode == 'data_prediction':
        return x_t + (1 - t) * v_t
    elif mode == 'eps_prediction':
        return x_t - t * v_t
    else:
        raise NotImplementedError()


# Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler._sigma_to_alpha_sigma_t
def _sigma_to_alpha_sigma_t(sigma, mode='flow'):
    if mode == 'flow':
        # alpha_t 0 -> 1 
        # sigma_t 1 -> 0
        # t 0 -> 1
        return 1 - sigma, sigma
    else:
        alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
        sigma_t = sigma * alpha_t
        return alpha_t, sigma_t


def unipc_diffusion(x, model_prev_list, t_prev_list, t, order, x_t=None, variant="bh1", use_corrector=True, model_fn=None, mode='data_prediction',
                    ratio=1):
    
    model_prev_0 = model_prev_list[-1]
    sigma_t, sigma_s0 = 1 - t, 1 - t_prev_list[-1]
    
    alpha_t, sigma_t = _sigma_to_alpha_sigma_t(sigma_t)
    alpha_s0, sigma_s0 = _sigma_to_alpha_sigma_t(sigma_s0)

    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

    h = lambda_t - lambda_s0
    device = x.device

    rks = []
    D1s = []

    for i in range(1, order):
        
        si =  1 - t_prev_list[-(i + 1)]
        model_prev_i = model_prev_list[-(i + 1)]
        alpha_si, sigma_si = _sigma_to_alpha_sigma_t(si)
        lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
        rk = (lambda_si - lambda_s0) / h
        rks.append(rk)
        D1s.append((model_prev_i - model_prev_0) / rk)

    rks.append(1.0)
    rks = torch.tensor(rks, device=device)

    R = []
    b = []

    hh = -h if mode == 'data_prediction' else h
    h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
    h_phi_k = h_phi_1 / hh - 1

    factorial_i = 1

    if variant == "bh1":
        B_h = ratio * hh
    elif variant == "bh2":
        B_h = ratio * torch.expm1(hh)
    else:
        raise NotImplementedError()

    for i in range(1, order + 1):
        R.append(torch.pow(rks, i - 1))
        b.append(h_phi_k * factorial_i / B_h)
        factorial_i *= i + 1
        h_phi_k = h_phi_k / hh - 1 / factorial_i

    R = torch.stack(R)
    b = torch.tensor(b, device=device)
    

    if len(D1s) > 0:
        D1s = torch.stack(D1s, dim=1)  # (B, K)
        # for order 2, we use a simplified version
        if order == 2:
            rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1]).to(device).to(x.dtype)
    else:
        D1s = None

    # for order 1, we use a simplified version
    if order == 1:
        rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
    else:
        rhos_c = torch.linalg.solve(R, b).to(device).to(x.dtype)

    model_t = None
    
    if mode == 'data_prediction':
        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * model_prev_0
        if D1s is not None:
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
        else:
            pred_res = 0
        x_t = x_t_ - alpha_t * B_h * pred_res
    else:
        x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * model_prev_0
        if D1s is not None:
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
        else:
            pred_res = 0
        x_t = x_t_ - sigma_t * B_h * pred_res


    if D1s is not None:
        pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
    else:
        pred_res = 0
    
    # print(x_t_.sum())
    # print((alpha_t * B_h * pred_res).sum())
    # x_t = x_t_ #- alpha_t * B_h * pred_res
    # print('aaa')
    x_t = x_t.to(x.dtype)


    if use_corrector:
        assert model_fn is not None
        model_t = model_fn(x_t, t)
        
        if mode == 'data_prediction':
            model_t = x_t + (1 - t) * model_t
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * model_prev_0
            if D1s is not None:
                corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = model_t - model_prev_0
            x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        else:
            model_t = x_t - t * model_t
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * model_prev_0
            if D1s is not None:
                corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = model_t - model_prev_0
            x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)
            
        x_t = x_t.to(x.dtype)   
    
    return x_t, model_t
