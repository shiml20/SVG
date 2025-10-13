# %% -------------------------
# 遍历 1000 类，保存单张图像
# ----------------------------
import tqdm
# %% [markdown]
# # Scalable Diffusion Models with Transformer (DiT)
# 
# 优化后的 Notebook，支持完整采样、解码，以及单 token 可视化。

# %%
import os
import glob
import torch
from torchvision.utils import save_image, make_grid
from PIL import Image
from IPython.display import display
from omegaconf import OmegaConf

from diffusion.rectified_flow_ori import RectifiedFlow
from utils import instantiate_from_config
from download import find_model
from ldm.modules.diffusionmodules.model import Decoder
from ldm.models.dino_decoder import DinoDecoder

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

# %% -------------------------
# 1. 配置加载
# ----------------------------
def get_config(ckpt_path):
    exp_root = "/".join(ckpt_path.split("/")[:-2])
    config_path = glob.glob(os.path.join(exp_root, "*.yaml"))
    assert len(config_path) == 1, "Expected exactly one config file."
    config = OmegaConf.load(config_path[0])
    exp_name = exp_root.split("/")[-1]
    return exp_name, config

ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0141-E0006_Dense_B_Flow_Dinov3_vitb_BS256-GPU8/checkpoints/0300000.pt"
ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0139-E0004_Dense_XL_Flow_Dinov3_vitb_BS256-GPU8/checkpoints/0300000.pt"
ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt"
# ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0152-E0012_LightingDiT_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints/0050000.pt"
# ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0152-E0012_LightingDiT_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints/0100000.pt"
ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0100000.pt"
ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0182-E0020_XL_Flow_Dinov3sp_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0150000.pt"
ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0600000.pt"
ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints/1850000.pt"

exp_name, config = get_config(ckpt_path)
step = ckpt_path.split('/')[-1].split('.')[0]

print("Experiment:", exp_name)

# %% -------------------------
# 2. 模型与 Decoder 加载
# ----------------------------
# 主模型
model = instantiate_from_config(config.model)
state_dict = find_model(ckpt_path)
model.load_state_dict(state_dict, strict=False)
model = model.to(device).eval()
# from omegaconf import OmegaConf

encoder_config = OmegaConf.load(config.basic.encoder_config)

dinov3 = DinoDecoder(
    ddconfig=encoder_config.model.params.ddconfig,
    dinoconfig=encoder_config.model.params.dinoconfig,
    lossconfig=encoder_config.model.params.lossconfig,
    embed_dim=encoder_config.model.params.embed_dim,
    ckpt_path=encoder_config.ckpt_path,
    extra_vit_config=encoder_config.model.params.extra_vit_config,
).cuda().eval()

z_channels = encoder_config.model.params.ddconfig.z_channels


num_steps = 50
cfg_scale = 4
timestep_shift = 0.15
samples_per_class = 10   # 每类要多少个样本
seeds = [0, 1, 233, 1000]

out_dir = f"samples_{exp_name}_{step}"
os.makedirs(out_dir, exist_ok=True)

diffusion = RectifiedFlow(model)

dinov3_sp_stats = torch.load("dinov3_sp_stats.pt")
dinov3_sp_mean = dinov3_sp_stats["dinov3_sp_mean"].to(device)[:, :, :z_channels]
dinov3_sp_std = dinov3_sp_stats["dinov3_sp_std"].to(device)[:, :, :z_channels]

c2 = [33,88,89,207,270,250,279,291,387,388,928,933,972,973,975,980]
c2 = [668, 562, 207, 360, 387, 974, 88, 979, 417, 279]
c2 = [263, 324, 326, 397, 393]

# c2 = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 24, 30, 33, 37, 39, 40, 47, 48, 50, 51, 63, 64, 69, 71, 74, 76, 78, 81, 84, 88, 89, 90, 96, 97, 99, 100, 101, 105, 107, 108, 112, 115, 117, 118, 120, 121, 122, 124, 125, 134, 139, 148, 151, 153, 155, 156, 157, 158, 160, 162, 165, 167, 174, 175, 177, 178, 179, 184, 185, 186, 187, 188, 194, 195, 205, 207, 208, 215, 219, 222, 223, 228, 230, 232, 234, 235, 240, 244, 247, 248, 250, 253, 258, 259, 263, 264, 266, 269, 270, 272, 277, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 291, 292, 294, 296, 299, 301, 306, 307, 308, 310, 311, 313, 315, 316, 317, 319, 321, 322, 323, 324, 325, 326, 327, 330, 335, 338, 346, 348, 349, 355, 356, 358, 359, 360, 366, 367, 368, 371, 373, 378, 387, 388, 393, 402, 403, 406, 407, 414, 417, 425, 437, 438, 441, 442, 449, 451, 453, 455, 460, 465, 474, 475, 483, 492, 493, 496, 538, 551, 553, 562, 567, 572, 588, 591, 600, 605, 607, 608, 610, 629, 632, 635, 637, 643, 644, 654, 658, 664, 681, 695, 698, 712, 717, 719, 721, 738, 742, 782, 796, 798, 803, 804, 805, 809, 824, 827, 829, 831, 832, 846, 850, 851, 852, 854, 857, 894, 906, 907, 908, 923, 927, 928, 930, 931, 933, 934, 936, 937, 938, 940, 941, 943, 945, 946, 948, 949, 951, 952, 953, 955, 957, 958, 961, 962, 963, 968, 971, 972, 973, 974, 975, 976, 977, 978, 984, 985, 990, 991, 992, 997, 998]

for seed in seeds:
    torch.manual_seed(seed)

    for class_id in tqdm.tqdm(c2, desc=f"Seed {seed} - Sampling all classes"):
        # 一次性生成一批 latent
        z = torch.randn(samples_per_class, 256, z_channels, device=device)
        y = torch.full((samples_per_class,), class_id, device=device)
        y_null = torch.full((samples_per_class,), 1000, device=device)

        # 批量采样
        samples = diffusion.sample(
            z, y, y_null,
            sample_steps=num_steps,
            cfg=cfg_scale,
            mode="euler",
            timestep_shift=timestep_shift
        )[-1]

        if config.basic.get("feature_norm", False):
            samples = samples * dinov3_sp_std + dinov3_sp_mean

        # [B, T, D] -> [B, D, 16, 16]
        B, T, D = samples.shape
        samples_latent = samples.permute(0, 2, 1).reshape(B, D, 16, 16)

        with torch.no_grad():
            decoded_full = dinov3.decode(samples_latent)

        decoded_full = torch.clamp(decoded_full, -1, 1)

        # 单张图分别保存
        for i in range(B):
            save_path = os.path.join(
                out_dir,
                f"class{class_id:04d}_seed{seed}_idx{i}_steps{num_steps}_cfg{cfg_scale}_shift{timestep_shift}.png"
            )
            save_image(decoded_full[i], save_path, nrow=1, normalize=True, value_range=(-1, 1))
