import torch

def multistep_unipc_update(x, model_prev_list, t_prev_list, t, order, x_t=None, variant="bh1", use_corrector=True, model_fn=None):
    t_prev_0 = t_prev_list[-1]
    model_prev_0 = model_prev_list[-1]
    h = t - t_prev_0
    rks = []
    D1s = []
    for i in range(1, order):
        t_prev_i = t_prev_list[-(i + 1)]
        model_prev_i = model_prev_list[-(i + 1)]
        rk = ((t_prev_i - t_prev_0) / h).item()
        rks.append(rk)
        D1s.append((model_prev_i - model_prev_0) / rk)
        
    rks.append(1)
    rks = torch.tensor(rks, device=x.device)
    
    R = []
    b = []
    
    if variant == "bh1":
        B_h = h
    elif variant == "bh2":
        B_h = torch.expm1(h)
    else:
        raise NotImplementedError()
    
    for i in range(1, order + 1):
        R.append(torch.pow(rks, i - 1))
        b.append(h / B_h * (1 / (i + 1)))
    
    R = torch.stack(R)
    b = torch.tensor(b, device=x.device)

    # now predictor
    use_predictor = len(D1s) > 0 and x_t is None
    if len(D1s) > 0:
        D1s = torch.stack(D1s, dim=1) # (B, K)
        if x_t is None:
            # for order 2, we use a simplified version
            if order == 2:
                rhos_p = torch.tensor([0.5], device=b.device)
            else:
                rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
    else:
        D1s = None
        
    if use_corrector:
        # print('using corrector')
        # for order 1, we use a simplified version
        if order == 1:
            rhos_c = torch.tensor([0.5], device=b.device)
        else:
            rhos_c = torch.linalg.solve(R, b)
            
    model_t = None
    x_t_euler = x + model_prev_0 * h
    if x_t is None:
        if use_predictor:
            pred_res = torch.einsum('k,bkchw->bchw', rhos_p, D1s)
        else:
            pred_res = 0
        x_t = x_t_euler + B_h * pred_res
    
    if use_corrector:
        assert model_fn is not None
        model_t = model_fn(x_t, t)
        if D1s is not None:
            corr_res = torch.einsum('k,bkchw->bchw', rhos_c[:-1], D1s)
        else:
            corr_res = 0
        D1_t = (model_t - model_prev_0)
        x_t = x_t_euler + B_h * (corr_res + rhos_c[-1] * D1_t)

    return x_t, model_t