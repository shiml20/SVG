# %% [markdown]
# # Scalable Diffusion Models with Transformer (DiT, VA-VAE)
# 
# 优化版 Notebook，支持完整采样、解码，以及 latent 插值可视化。

# %%
import os
import sys
import glob
import torch
from torchvision.utils import save_image, make_grid
from PIL import Image
from IPython.display import display
from omegaconf import OmegaConf

from diffusion.rectified_flow_ori import RectifiedFlow
from utils import instantiate_from_config
from download import find_model

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("⚠️ GPU not found. Using CPU instead.")

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
ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0156-E0014_Dense_XL_Flow_vavae_BS256-GPU8/checkpoints/0300000.pt"

exp_name, config = get_config(ckpt_path)
step = os.path.splitext(os.path.basename(ckpt_path))[0]
print(f"Experiment: {exp_name} | Step: {step}")

# %% -------------------------
# 2. 模型与 VAE 加载
# ----------------------------
# 主模型
model = instantiate_from_config(config.model)
state_dict = find_model(ckpt_path)
model.load_state_dict(state_dict, strict=False)
model = model.to(device).eval()

# ✅ 加载 VA-VAE
tokenizer_path = "/ytech_m2v3_hdd/yuanziyang/sml/Feature-Visual-Generation"
if tokenizer_path not in sys.path:
    sys.path.append(tokenizer_path)
from tokenizer.vavae import VA_VAE

vae = VA_VAE(
    "/ytech_m2v3_hdd/yuanziyang/sml/Feature-Visual-Generation/vavae/configs/f16d32_vfdinov2.yaml"
).load().model.to(device).eval()

# %% -------------------------
# 3. 插值采样函数
# ----------------------------
def interpolate_and_decode(
    model, vae,
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
        decoded = vae.decode(samples[-1])  # [B,3,H,W]
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

# latent 初始化
latent_size = 256 // 16   # 因为图像 256x256，VAE downsample 16x
z1 = torch.randn(len(class_labels), 32, latent_size, latent_size, device=device)
z2 = torch.randn_like(z1)

# 插值采样
decoded_all = interpolate_and_decode(
    model, vae,
    z1, z2, y, y_null,
    num_steps=num_steps, cfg_scale=cfg_scale,
    mode="euler", num_interp=10
)

# 保存 & 显示
grid = make_grid(decoded_all, nrow=10, normalize=True, value_range=(-1, 1))
save_path = f"vae_interp_{exp_name}_{step}_steps{num_steps}_cfg{cfg_scale}_{class_labels[0]}.png"
save_image(grid, save_path)
display(Image.open(save_path))
print("Saved to:", save_path)
