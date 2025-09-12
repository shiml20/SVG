# %% [markdown]
# # Scalable Diffusion Models with Transformer (DiT, T2I)
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
    exp_root = os.path.dirname(os.path.dirname(ckpt_path))
    config_path = glob.glob(os.path.join(exp_root, "*.yaml"))
    assert len(config_path) == 1, f"Expected 1 config, found {len(config_path)}"
    config = OmegaConf.load(config_path[0])
    exp_name = os.path.basename(exp_root)
    return exp_name, config

# ✅ 选择 checkpoint
ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0150-Ablation-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints/0300000.pt"

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

# Decoder 配置
encoder_type = "vitsp16"
config_dict = {
    "vitb16": {
        "z_channels": 768,
        "weight_path": "/ytech_m2v3_hdd/yuanziyang/sml/FVG/model_vitb16.pt",
    },
    "vitsp16": {
        "z_channels": 384,
        "weight_path": "/ytech_m2v3_hdd/yuanziyang/sml/Feature-Visual-Generation/vavae/logs/f16d32_ldm_dinov3_vitsp/checkpoints/epoch=000029.ckpt",
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
# 3. 插值采样函数
# ----------------------------
def interpolate_and_decode(
    model, decoder,
    z1, z2, y, y_null,
    num_steps=20, cfg_scale=4,
    mode="euler", timestep_shift=0.4, 
    num_interp=10
):
    """对两个 latent 向量进行插值，采样并解码。"""
    diffusion = RectifiedFlow(model)
    ratios = torch.linspace(0, 1, num_interp, device=z1.device)

    decoded_list = []
    for r in ratios:
        # 插值并归一化
        z = (r * z1 + (1 - r) * z2) / torch.sqrt(r**2 + (1 - r) ** 2 + 1e-8)
        z = (r * z1 + (1 - r) * z2)

        samples = diffusion.sample(
            z, y, y_null,
            sample_steps=num_steps, cfg=cfg_scale, mode=mode, timestep_shift=timestep_shift
        )

        # [B, T, D] -> [B, D, 16, 16]
        B, T, D = samples[-1].shape
        samples_latent = samples[-1].permute(0, 2, 1).reshape(B, D, 16, 16)

        decoded = decoder(samples_latent)
        decoded_list.append(decoded)

    return torch.cat(decoded_list, dim=0)

# %% -------------------------
# 4. 运行插值可视化
# ----------------------------
# 参数设置
seed = 0
torch.manual_seed(seed)
num_steps = 20
cfg_scale = 4
class_labels = [279]   # 类别 ID，可改
class_labels = [207]   # 类别 ID，可改
y = torch.tensor(class_labels, device=device)
y_null = torch.tensor([1000] * len(class_labels), device=device)

# latent 初始化（可换成随机）
z1 = torch.load("z1.pt")[:1].to(device)
z2 = torch.load("z2.pt")[:1].to(device)

# 插值采样
decoded_all = interpolate_and_decode(
    model, decoder,
    z1, z2, y, y_null,
    num_steps=num_steps, cfg_scale=cfg_scale,
    mode="euler", num_interp=10
)

# 保存 & 显示
grid = make_grid(decoded_all, nrow=10, normalize=True, value_range=(-1, 1))
save_path = f"interp_{exp_name}_{step}_steps{num_steps}_cfg{cfg_scale}_{class_labels[0]}.png"
save_image(grid, save_path)
display(Image.open(save_path))
print("Saved to:", save_path)
