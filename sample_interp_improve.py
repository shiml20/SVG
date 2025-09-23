# %% [markdown]
# # 1. Setup

import os
import glob
import torch
from torchvision.utils import make_grid, save_image
from diffusion.rectified_flow import RectifiedFlow
from download import find_model
from utils import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from IPython.display import display

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")

def get_config(ckpt_path):
    """Load configuration from the checkpoint path."""
    exp_root = ckpt_path.split("/")[:-2]
    exp_name = exp_root[-1]
    exp_root = "/".join(exp_root)
    config_path = glob.glob(os.path.join(exp_root, "*.yaml"))
    assert len(config_path) == 1, "Expected exactly one config file."
    config = OmegaConf.load(config_path[0])
    return exp_name, config

# -----------------------------
# 配置区：选择 VAE 类型
# -----------------------------
USE_LOCAL_VAE = False   # True: 用本地 VA_VAE, False: 用 HuggingFace AutoencoderKL
image_size = 256
ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints/2000000.pt"

exp_name, config = get_config(ckpt_path)
step = ckpt_path.split('/')[-1].split('.')[0]
print("exp_name", exp_name)

# %% [markdown]
# # 2. Load Model & VAE

# Load model
model = instantiate_from_config(config.model)
state_dict = find_model(ckpt_path)
model.load_state_dict(state_dict, strict=False)
model = model.to(device).eval()

# Load VAE
if USE_LOCAL_VAE:
    # --- 本地 VA_VAE ---
    tokenizer_path = "/ytech_m2v3_hdd/yuanziyang/sml/Feature-Visual-Generation"
    if tokenizer_path not in sys.path:
        sys.path.append(tokenizer_path)
    from tokenizer.vavae import VA_VAE
    vae = VA_VAE(
        "/ytech_m2v3_hdd/yuanziyang/sml/Feature-Visual-Generation/vavae/configs/f16d32_vfdinov2.yaml"
    ).load().model
    latent_channels = 32
    latent_size = image_size // 16
    vae_scale_factor = 1.0   # 不需要缩放

else:
    # --- HuggingFace AutoencoderKL ---
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    latent_channels = 4
    latent_size = image_size // 8
    vae_scale_factor = 0.18215   # decode 前要除以这个

# %% [markdown]
# # 3. Sampling

seed = 0
torch.manual_seed(seed)
num_steps = 50
cfg_scale = 3

c2 = [14, 17, 20, 270]
for class_labels in c2:

    class_labels = [class_labels]
    mode = "euler"

    diffusion = RectifiedFlow(model)

    n = len(class_labels)
    y = torch.tensor(class_labels, device=device)
    y_null = torch.tensor([1000] * n, device=device)

    # 两个 latent 噪声
    z1 = torch.randn(n, latent_channels, latent_size, latent_size, device=device)
    z2 = torch.randn(n, latent_channels, latent_size, latent_size, device=device)

    ratios = torch.linspace(0, 1, 10)  # 插值

    decoded_list = []
    with torch.no_grad():
        for r in ratios:
            z = (r * z1 + (1 - r) * z2) / torch.sqrt(r ** 2 + (1 - r) ** 2)
            # z = (r * z1 + (1 - r) * z2)
            samples = diffusion.sample(z, y, y_null, sample_steps=num_steps, cfg=cfg_scale, mode=mode)
            
            # Decode
            latent = samples[-1] / vae_scale_factor
            decoded = vae.decode(latent).sample
            decoded_list.append(decoded)

    # 拼接结果
    decoded_all = torch.cat(decoded_list, dim=0)
    grid = make_grid(decoded_all, nrow=len(ratios), normalize=True, value_range=(-1,1))

    save_path = f"nonlinear_vae_interp_{exp_name}_{step}_sample{num_steps}_{mode}_cfg{cfg_scale}_class{class_labels[0]}.png"
    save_image(grid, save_path)
    # display(Image.open(save_path))

    decoded_list = []
    with torch.no_grad():
        for r in ratios:
            # z = (r * z1 + (1 - r) * z2) / torch.sqrt(r ** 2 + (1 - r) ** 2)
            z = (r * z1 + (1 - r) * z2)
            samples = diffusion.sample(z, y, y_null, sample_steps=num_steps, cfg=cfg_scale, mode=mode)
            
            # Decode
            latent = samples[-1] / vae_scale_factor
            decoded = vae.decode(latent).sample
            decoded_list.append(decoded)

    # 拼接结果
    decoded_all = torch.cat(decoded_list, dim=0)
    grid = make_grid(decoded_all, nrow=len(ratios), normalize=True, value_range=(-1,1))

    save_path = f"linear_vae_interp_{exp_name}_{step}_sample{num_steps}_{mode}_cfg{cfg_scale}_class{class_labels[0]}.png"
    save_image(grid, save_path)
    # display(Image.open(save_path))
