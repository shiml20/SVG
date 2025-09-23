# %% [markdown]
# # Scalable Diffusion Models with Transformer (DiT)
# 
# 优化版 Notebook，支持完整采样、解码，以及 latent 插值可视化。

# %%
import os
import glob
import torch
from torchvision.utils import save_image, make_grid
from PIL import Image
from IPython.display import display
from omegaconf import OmegaConf

from diffusion.rectified_flow_ori_ablation_guidance import RectifiedFlow
from utils import instantiate_from_config
from download import find_model
from ldm.models.dino_decoder import DinoDecoder

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

# %% -------------------------
# 1. 配置加载
# ----------------------------
def get_config(ckpt_path):
    exp_root = os.path.dirname(os.path.dirname(ckpt_path))
    config_path = glob.glob(os.path.join(exp_root, "*.yaml"))
    assert len(config_path) == 1, f"Expected 1 config, found {len(config_path)}"
    config = OmegaConf.load(config_path[0])
    exp_name = os.path.basename(exp_root)
    return exp_name, config

# ✅ 修改这里选择 checkpoint
ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints/1150000.pt"

exp_name, config = get_config(ckpt_path)
step = os.path.splitext(os.path.basename(ckpt_path))[0]
print(f"Experiment: {exp_name} | Step: {step}")

# %% -------------------------
# 2. 模型与 Decoder 加载
# ----------------------------
# 主模型
model = instantiate_from_config(config.model)
state_dict = find_model(ckpt_path)
model.load_state_dict(state_dict, strict=False)
model = model.to(device).eval()


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


dinov3_sp_stats = torch.load("dinov3_sp_stats.pt")
dinov3_sp_mean = dinov3_sp_stats["dinov3_sp_mean"].to(device)[:,:,:z_channels]
dinov3_sp_std = dinov3_sp_stats["dinov3_sp_std"].to(device)[:,:,:z_channels]

# %% -------------------------
# 3. 插值采样函数
# ----------------------------
def interpolate_and_decode(
    model, decoder, 
    z1, z2, y, y_null,
    num_steps=20, cfg_scale=4, 
    mode="euler", timestep_shift=0.0, 
    num_interp=10,
    interp_mode=1
):
    """对两个 latent 向量进行插值，采样并解码。"""
    diffusion = RectifiedFlow(model)
    ratios = torch.linspace(0, 1, num_interp, device=z1.device)

    decoded_list = []
    for r in ratios:
        if interp_mode:
            z = (r * z1 + (1 - r) * z2) / torch.sqrt(r**2 + (1 - r) ** 2 + 1e-8)
        else:
            z = (r * z1 + (1 - r) * z2)

        samples = diffusion.sample(
            z, y, y_null, 
            sample_steps=num_steps, cfg=cfg_scale, 
            mode=mode, timestep_shift=timestep_shift,
            # cfg_mode="cfg_star-1-0"
        )[-1]

        # import ipdb; ipdb.set_trace()
        if config.basic.get("feature_norm", False):
            samples = samples * dinov3_sp_std + dinov3_sp_mean

        # [B, T, D] -> [B, D, 16, 16]
        B, T, D = samples.shape
        samples_latent = samples.permute(0, 2, 1).reshape(B, D, 16, 16)
        
        decoded = decoder.decode(samples_latent)
        decoded_list.append(decoded)

    return torch.cat(decoded_list, dim=0)   # [num_interp*B, 3, H, W]

# %% -------------------------
# 4. 运行插值可视化
# ----------------------------
# 参数设置
seed = 233
torch.manual_seed(seed)
num_steps = 25
cfg_scale = 6
# c2 = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 24, 30, 33, 37, 39, 40, 47, 48, 50, 51, 63, 64, 69, 71, 74, 76, 78, 81, 84, 88, 89, 90, 96, 97, 99, 100, 101, 105, 107, 108, 112, 115, 117, 118, 120, 121, 122, 124, 125, 134, 139, 148, 151, 153, 155, 156, 157, 158, 160, 162, 165, 167, 174, 175, 177, 178, 179, 184, 185, 186, 187, 188, 194, 195, 205, 207, 208, 215, 219, 222, 223, 228, 230, 232, 234, 235, 240, 244, 247, 248, 250, 253, 258, 259, 263, 264, 266, 269, 270, 272, 277, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 291, 292, 294, 296, 299, 301, 306, 307, 308, 310, 311, 313, 315, 316, 317, 319, 321, 322, 323, 324, 325, 326, 327, 330, 335, 338, 346, 348, 349, 355, 356, 358, 359, 360, 366, 367, 368, 371, 373, 378, 387, 388, 393, 402, 403, 406, 407, 414, 417, 425, 437, 438, 441, 442, 449, 451, 453, 455, 460, 465, 474, 475, 483, 492, 493, 496, 538, 551, 553, 562, 567, 572, 588, 591, 600, 605, 607, 608, 610, 629, 632, 635, 637, 643, 644, 654, 658, 664, 681, 695, 698, 712, 717, 719, 721, 738, 742, 782, 796, 798, 803, 804, 805, 809, 824, 827, 829, 831, 832, 846, 850, 851, 852, 854, 857, 894, 906, 907, 908, 923, 927, 928, 930, 931, 933, 934, 936, 937, 938, 940, 941, 943, 945, 946, 948, 949, 951, 952, 953, 955, 957, 958, 961, 962, 963, 968, 971, 972, 973, 974, 975, 976, 977, 978, 984, 985, 990, 991, 992, 997, 998]
c2 = [270, 14, 17, 20]
for class_labels in c2:
    class_labels = [class_labels]  # 🐱 类别 ID，可改
    y = torch.tensor(class_labels, device=device)
    y_null = torch.tensor([1000] * len(class_labels), device=device)

    # 随机 latent
    z1 = torch.randn(len(class_labels), 256, z_channels, device=device)
    z2 = torch.randn_like(z1)



    # 插值采样
    decoded_all = interpolate_and_decode(
        model, dinov3, 
        z1, z2, y, y_null,
        num_steps=num_steps, cfg_scale=cfg_scale, 
        mode="euler", timestep_shift=0.15, 
        num_interp=10,
        interp_mode=1
    )

    # 保存 & 显示
    grid = make_grid(decoded_all, nrow=10, normalize=True, value_range=(-1, 1))
    save_path = f"nonlinear_interp_{exp_name}_{step}_steps{num_steps}_cfg{cfg_scale}_{class_labels[0]}_seed{seed}.png"
    save_image(grid, save_path)
    display(Image.open(save_path))
    print("Saved to:", save_path)

    # 插值采样
    decoded_all = interpolate_and_decode(
        model, dinov3, 
        z1, z2, y, y_null,
        num_steps=num_steps, cfg_scale=cfg_scale, 
        mode="euler", timestep_shift=0.15, 
        num_interp=10,
        interp_mode=0
    )

    # 保存 & 显示
    grid = make_grid(decoded_all, nrow=10, normalize=True, value_range=(-1, 1))
    save_path = f"linear_interp_{exp_name}_{step}_steps{num_steps}_cfg{cfg_scale}_{class_labels[0]}_seed{seed}.png"
    save_image(grid, save_path)
    display(Image.open(save_path))
    print("Saved to:", save_path)

