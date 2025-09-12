import torch
from omegaconf import OmegaConf
# import torchprofile
from utils import update_ema, requires_grad, cleanup, setup_ddp, setup_exp_dir, setup_data, instantiate_from_config
from download import find_model

def main():
    device = torch.device("cuda")
    config_path = "/m2v_intern/shiminglei/DiT_MoE_Dynamic/config/exp_configs_sml/8005_Dense_DiT_XXXL_Flow_GPU4_bs256.yaml"
    config_path = "/m2v_intern/shiminglei/DiffMoE_research/config/004_DiffMoE_S_E16_Flow128_A1_swiglu_iniMoEmlp.yaml"
    # config_path = "/m2v_intern/shiminglei/DiffMoE_research/config/000_DiffMoE_S_E16_Flow.yaml"



    print(config_path)
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"DiT Parameters: {total_params:,}")
    print()

    count_attn = 0
    count_ffn = 0
    count_adaLN = 0
    for name, param in model.named_parameters():
        if "blocks" in name:
            if "attn" in name:
                count_attn += param.numel()
            elif "mlp" in name or "moe" in name:
                count_ffn += param.numel()
            elif "adaLN_modulation" in name:
                count_adaLN += param.numel()
    count_remain = total_params - count_attn - count_ffn - count_adaLN
    print("Attention Parameters: ", count_attn, "  ratio: ", round(count_attn/total_params, 4))
    print("FFN Parameters: ", count_ffn, "  ratio: ", round(count_ffn/total_params, 4))
    print("AdaLN Parameters: ", count_adaLN, "  ratio: ", round(count_adaLN/total_params, 4))
    print("Remaining Parameters: ", count_remain, "  ratio: ", round(count_remain/total_params, 4))

    # MT_MOE_PATH = "/m2v_intern/shiminglei/DiT_MoE_Dynamic/exps/0005-908_DiT-S-MovingThreshold-MoE-0-1-16_Interleave_Layer12_Dim384_Act32M_Total139M_bs256/checkpoints/0300000.pt"
    # TC_MOE_PATH = "/m2v_intern/shiminglei/DiT_MoE_Dynamic/exps/0039-1008_DiT-S_TCMoE82AS0_Layer12_Dim384_Act32M_Total132M_bs256/checkpoints/0300000.pt"
    # CP_MOE_PATH = "/m2v_intern/shiminglei/DiT_MoE_Dynamic/yzy_treasure/yzy_exp/0263-906_DiT_EC-MoE-0-1-16_BatchLevel_CapacityPred_TrainTogether_Layer12_Dim384_Act32M_Total139M_bs256/checkpoints/0300000.pt"    

    # state_dict = find_model(MT_MOE_PATH)
    # state_dict = find_model(TC_MOE_PATH)
    # state_dict = find_model(CP_MOE_PATH)
    # model.load_state_dict(state_dict)
    model.eval() # important!
    model = model.to(device)

    from fvcore.nn import FlopCountAnalysis

    def compute_flops(model, inputs):
        flops = FlopCountAnalysis(model, inputs)
        return flops.total()

    latents = torch.load("/m2v_intern/yuanziyang/data/train/n02966193/n02966193_15942.pt")
    # latents = torch.load("/m2v_intern/yuanziyang/data/train/n02966193/n02966193_15796.pt")
    print(latents.shape)
    latents = torch.zeros([1, 4, 32//2, 32//2])
    embedding = latents.to(device)
    bs = 2
    embedding = embedding.repeat(bs, 1, 1, 1)

    with torch.no_grad():
        for i in range(1):
            t = torch.randint(0, 1000, size=(), device=device)
            y = torch.randint(0, 1000, size=(), device=device)
            coeff = torch.randint(0, 1000, size=(), device=device)

            t = torch.tensor([int(t)]*bs, device=device)
            y = torch.tensor([int(y)]*bs, device=device)
            embedding =  embedding + coeff / 1000 * torch.rand_like(embedding).to(device)


            # 使用 script 而不是 trace

            # Inside your main function
            flops = compute_flops(model, (embedding, t, y))
            print(f"Total FLOPs: {flops / bs / 1e9:.3f} GFLOPs")


if __name__ == "__main__":
    main()
