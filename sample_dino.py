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

ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0160-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm_load500K-GPU8/checkpoints/1100000.pt"
ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0206-E0031_LightingDiT_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints/0050000.pt"


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

# Decoder 配置
encoder_type = "vitsp16"
config_dict = {
    "vitb16": {
        "z_channels": 768,
        "weight_path": "/ytech_m2v3_hdd/yuanziyang/sml/FVG/model_vitb16.pt",
    },
    "vitsp16": {
        "z_channels": 384,
        "weight_path": "/ytech_m2v3_hdd/yuanziyang/sml/Feature-Visual-Generation/vavae/logs/f16d32_ldm_dinov3_vitsp/checkpoints/epoch=000047.ckpt",
    },
}

z_channels = config_dict[encoder_type]["z_channels"]

ddconfig = {
    "double_z": True,
    "z_channels": z_channels,
    "resolution": 256,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": [1, 1, 2, 2, 4],
    "num_res_blocks": 2,
    "attn_resolutions": [16],
    "dropout": 0.0,
}

def load_decoder(weight_path):
    decoder = Decoder(**ddconfig).to(device)
    state = torch.load(weight_path, map_location=device)["state_dict"]
    rename_dict = {k[8:]: v for k, v in state.items() if "decoder" in k}
    decoder.load_state_dict(rename_dict, strict=False)
    decoder.eval()
    return decoder

decoder = load_decoder(config_dict[encoder_type]["weight_path"])

# %% -------------------------
# 3. 采样
# ----------------------------
seed = 0
torch.manual_seed(seed)
num_steps = 2
cfg_scale = 4
class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
class_labels = [130, 270, 284, 688, 250, 146, 980, 484, 207, 360, 387, 974, 88, 979, 417, 279]

samples_per_row = 4

diffusion = RectifiedFlow(model)

n = len(class_labels)
z1 = torch.randn(n, 256, z_channels, device=device)
z2 = torch.randn(n, 256, z_channels, device=device)


dinov3_sp_stats = torch.load("dinov3_sp_stats.pt")
dinov3_sp_mean = dinov3_sp_stats["dinov3_sp_mean"].to(device)[:,:,:z_channels]
dinov3_sp_std = dinov3_sp_stats["dinov3_sp_std"].to(device)[:,:,:z_channels]


ratio = 0
timestep_shift = 0.15
# timestep_shift = 1.0
z = ratio * z1 + (1 - ratio) * z2

y = torch.tensor(class_labels, device=device)
y_null = torch.tensor([1000] * n, device=device)
mode = "euler"
samples = diffusion.sample(z, y, y_null, sample_steps=num_steps, cfg=cfg_scale, mode=mode, timestep_shift=timestep_shift)[-1]

# import ipdb; ipdb.set_trace()
if config.basic.get("feature_norm", False):
    samples = samples * dinov3_sp_std + dinov3_sp_mean


# [B, T, D] -> [B, D, 16, 16]
B, T, D = samples.shape
samples_latent = samples.permute(0, 2, 1).reshape(B, D, 16, 16)

# %% -------------------------
# 4. 解码完整图像
# ----------------------------
with torch.no_grad():
    decoded_full = decoder(samples_latent)

save_path = f"sample_{exp_name}_{step}_sample{num_steps}_{mode}_cfg{cfg_scale}_shift{timestep_shift}.png"
save_image(decoded_full, save_path, nrow=samples_per_row, normalize=True, value_range=(-1, 1))
display(Image.open(save_path))

# %% -------------------------
# 5. 单 token 可视化
# ----------------------------
# def decode_single_tokens(latent, decoder, out_dir="token_decodes"):
#     B, D, H, W = latent.shape
#     N = H * W
#     os.makedirs(out_dir, exist_ok=True)

#     # 批量构造 masked latent
#     masked_latents = torch.zeros(N * B, D, H, W, device=device)
#     for i in range(H):
#         for j in range(W):
#             idx = i * W + j
#             masked_latents[idx*B:(idx+1)*B, :, i, j] = latent[:, :, i, j]

#     with torch.no_grad():
#         decoded_tokens = decoder(masked_latents)

#     # 保存单独 token 图像
#     for idx in range(N):
#         for b in range(B):
#             save_image(
#                 decoded_tokens[idx*B + b],
#                 os.path.join(out_dir, f"b{b}_token_{idx//W:02d}_{idx%W:02d}.png"),
#                 normalize=True, value_range=(-1, 1)
#             )

#     # 拼接成一个网格方便对比
#     grid = make_grid(decoded_tokens, nrow=W, normalize=True, value_range=(-1, 1))
#     save_image(grid, os.path.join(out_dir, "all_tokens_grid.png"))
#     return grid

# grid_img = decode_single_tokens(samples_latent, decoder)
# display(Image.open("token_decodes/all_tokens_grid.png"))
