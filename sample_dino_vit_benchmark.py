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

from diffusion.rectified_flow_ori_ablation_guidance import RectifiedFlow
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
ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0183-E0021_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0500000.pt"
ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0182-E0020_XL_Flow_Dinov3sp_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0500000.pt"
ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0500000.pt"
ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints/1100000.pt"
# ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0193-E0024_B_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift1_featureNorm_repa-GPU8/checkpoints/0050000.pt"
ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0195-G0000_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load1600K_512-GPU8/checkpoints/0500000.pt"



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

# %% -------------------------
# 3. 采样
# ----------------------------
seed = 0
torch.manual_seed(seed)
num_steps = 250
num_steps = 50
# cfg_scale = 1.25
cfg_scale = 4
class_labels = [130, 270, 284, 688, 250, 
                146, 980, 484, 207, 360, 
                387, 974, 88, 979, 417, 
                279]
# class_labels = [270, ]
# class_labels = [279] * 20
# class_labels = [207] * 20
samples_per_row = 4

# 0.022425 s / NFE

diffusion = RectifiedFlow(model)

n = len(class_labels)
image_size = 256 
image_size = 512
# z = torch.randn(n, 256, z_channels, device=device)
z = torch.randn(n, (image_size // 16) ** 2, z_channels, device=device)
# z = torch.randn(n, 256, 384, device=device)
# z1 = torch.load("z1.pt")
# z2 = torch.load("z2.pt")

dinov3_sp_stats = torch.load("dinov3_sp_stats.pt")
dinov3_sp_mean = dinov3_sp_stats["dinov3_sp_mean"].to(device)[:,:,:z_channels]
dinov3_sp_std = dinov3_sp_stats["dinov3_sp_std"].to(device)[:,:,:z_channels]

ratio = 0
timestep_shift = 0.15
# timestep_shift = 1.0
# z = ratio * z1 + (1 - ratio) * z2
# cfg_mode="cfg_star-1"
cfg_mode="constant"
# cfg_mode="s^2"

y = torch.tensor(class_labels, device=device)
y_null = torch.tensor([1000] * n, device=device)
mode = "euler"
# samples = diffusion.sample(z, y, y_null, sample_steps=num_steps, cfg=cfg_scale, mode=mode, timestep_shift=timestep_shift,
#                             cfg_mode=cfg_mode)[-1]

# # import ipdb; ipdb.set_trace()
# if config.basic.get("feature_norm", False):
#     samples = samples * dinov3_sp_std + dinov3_sp_mean

# # [B, T, D] -> [B, D, 16, 16]
# B, T, D = samples.shape
# samples_latent = samples.permute(0, 2, 1).reshape(B, D, image_size // 16, image_size // 16)
# # samples_latent = samples.permute(0, 2, 1).reshape(B, D, 32, 32)

# # %% -------------------------
# # 4. 解码完整图像
# # ----------------------------
# with torch.no_grad():
#     decoded_full = dinov3.decode(samples_latent)
#     # decoded_full = decoder(samples_latent)

# decoded_full = torch.clamp(decoded_full, -1, 1)  # 确保在指定范围内
# # import ipdb; ipdb.set_trace()

# save_path = f"{cfg_mode}_sample_{exp_name}_{step}_sample{num_steps}_{mode}_cfg{cfg_scale}_shift{timestep_shift}_{image_size}.png"
# save_image(decoded_full, save_path, nrow=samples_per_row, normalize=True, value_range=(-1, 1))
# display(Image.open(save_path))


import time
import torch
from torchvision.utils import save_image

def benchmark(diffusion, dinov3, z, y, y_null, 
              num_steps=50, cfg_scale=4, mode="euler", 
              timestep_shift=0.15, cfg_mode="constant", 
              image_size=512, samples_per_row=4, 
              dinov3_sp_mean=None, dinov3_sp_std=None, 
              feature_norm=False,
              n_runs=5, save_path=None):
    """
    Benchmark sampling + decoding time.
    """
    sampling_times, decoding_times = [], []

    for run in range(n_runs):
        torch.cuda.synchronize()
        start_time = time.time()

        # --- Sampling ---
        samples = diffusion.sample(
            z, y, y_null, 
            sample_steps=num_steps, 
            cfg=cfg_scale, 
            mode=mode, 
            timestep_shift=timestep_shift,
            cfg_mode=cfg_mode
        )[-1]

        torch.cuda.synchronize()
        sampling_time = time.time() - start_time
        sampling_times.append(sampling_time)

        if feature_norm:
            samples = samples * dinov3_sp_std + dinov3_sp_mean

        B, T, D = samples.shape
        samples_latent = samples.permute(0, 2, 1).reshape(B, D, image_size // 16, image_size // 16)

        # --- Decoding ---
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            decoded_full = dinov3.decode(samples_latent)

        torch.cuda.synchronize()
        decoding_time = time.time() - start_time
        decoding_times.append(decoding_time)

    # --- Summary ---
    avg_sampling = sum(sampling_times) / n_runs
    avg_decoding = sum(decoding_times) / n_runs
    print(f"[Benchmark over {n_runs} runs]")
    print(f"  Avg Sampling time: {avg_sampling:.2f} s")
    print(f"  Avg Decoding time: {avg_decoding:.2f} s")
    print(f"  Total Avg time   : {avg_sampling + avg_decoding:.2f} s")

    # 保存最后一次的结果
    decoded_full = torch.clamp(decoded_full, -1, 1)
    if save_path is not None:
        save_image(decoded_full, save_path, nrow=samples_per_row, normalize=True, value_range=(-1, 1))
        print(f"Saved result to {save_path}")

    return avg_sampling, avg_decoding


save_path = f"sample_{exp_name}_{step}_sample{num_steps}_{mode}_cfg{cfg_scale}_shift{timestep_shift}_{image_size}.png"

benchmark(
    diffusion, dinov3, z, y, y_null,
    num_steps=num_steps,
    cfg_scale=cfg_scale,
    mode=mode,
    timestep_shift=timestep_shift,
    cfg_mode=cfg_mode,
    image_size=image_size,
    samples_per_row=samples_per_row,
    dinov3_sp_mean=dinov3_sp_mean,
    dinov3_sp_std=dinov3_sp_std,
    feature_norm=config.basic.get("feature_norm", False),
    n_runs=5,  # 这里可以改成 10 次或者更多
    save_path=save_path
)
