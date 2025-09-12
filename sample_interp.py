# %% [markdown]
# # Scalable Diffusion Models with Transformer (DiT)
# 
# This notebook samples from pre-trained DiT models. DiTs are class-conditional latent diffusion models trained on ImageNet that use transformers in place of U-Nets as the DDPM backbone. DiT outperforms all prior diffusion models on the ImageNet benchmarks.
# 
# [Project Page](https://www.wpeebles.com/DiT) | [HuggingFace Space](https://huggingface.co/spaces/wpeebles/DiT) | [Paper](http://arxiv.org/abs/2212.09748) | [GitHub](github.com/facebookresearch/DiT)

# %% [markdown]
# # 1. Setup
# 
# We recommend using GPUs (Runtime > Change runtime type > Hardware accelerator > GPU). Run this cell to clone the DiT GitHub repo and setup PyTorch. You only have to run this once.

# %%
# DiT imports:
import os
import glob
import torch
from torchvision.utils import save_image
from diffusion.rectified_flow import RectifiedFlow
from diffusers.models import AutoencoderKL
from download import find_model
from PIL import Image
from IPython.display import display
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")
from utils import instantiate_from_config
from omegaconf import OmegaConf

def get_config(ckpt_path):
    """Load configuration from the checkpoint path."""
    exp_root = ckpt_path.split("/")[:-2]
    exp_name = exp_root[-1]
    exp_root = "/".join(exp_root)
    config_path = glob.glob(os.path.join(exp_root, "*.yaml"))
    assert len(config_path) == 1, "Expected exactly one config file."
    config = OmegaConf.load(config_path[0])
    return exp_name, config


# ckpt_path = "/ytech_m2v2_hdd/sml/DiffMoE_research_local/exps/7146-8001_RaceAll_S_AdaLN/checkpoints/0150000.pt"
ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints/3000000.pt"
# ckpt_path = "/m2v_intern/shiminglei/DiffMoE_research_local/exps/0010-1011_Dense_L_Flow256_MoBA_blockSize16_interleave_flash/checkpoints/0250000.pt"
# ckpt_path = "/m2v_intern/shiminglei/D/iffMoE_research_local/exps/0011-1010_Dense_L_Flow256/checkpoints/0250000.pt"

exp_name, config = get_config(ckpt_path)
step = ckpt_path.split('/')[-1].split('.')[0]

# config = OmegaConf.load("/m2v_intern/shiminglei/DiffMoE_research_local/config/1012_Dense_S_Flow256_MoBA_blockSize16_interleave_flash.yaml")

print("exp_name", exp_name)

# %% [markdown]
# # Download DiT-XL/2 Models
# You can choose between a 512x512 model and a 256x256 model. You can swap-out the LDM VAE, too.


# %%
image_size = 256 #@param [256, 512]
vae_model = "stabilityai/sd-vae-ft-ema" #@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
latent_size = int(image_size) // 8
# latent_size = 384
model = instantiate_from_config(config.model)
state_dict = find_model(ckpt_path)


# %%
model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval() # important!
vae = AutoencoderKL.from_pretrained(vae_model).to(device)

# %% [markdown]
# # 2. Sample from Pre-trained DiT Models
# 
# You can customize several sampling options. For the full list of ImageNet classes, [check out this](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).

# %%
# Set user inputs:
# %% -------------------------
# 3. 采样
# ----------------------------
seed = 0
torch.manual_seed(seed)
num_steps = 100
cfg_scale = 4
class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
class_labels = [279] * 1

diffusion = RectifiedFlow(model)

n = len(class_labels)
y = torch.tensor(class_labels, device=device)
y_null = torch.tensor([1000] * n, device=device)
mode = "euler"
# z = torch.randn(n, 4, latent_size, latent_size, device=device)
# torch.save(z, 'vae_z1.pt')
# z = torch.randn(n, 4, latent_size, latent_size, device=device)
# torch.save(z, 'vae_z2.pt')
z1 = torch.load("vae_z1.pt")[:1]
z2 = torch.load("vae_z2.pt")[:1]

# z = torch.randn(n, 256, z_channels, device=device)
import torch
from torchvision.utils import make_grid, save_image
from PIL import Image

# 生成比例 (左:0 → 右:1)
ratios = torch.linspace(0, 1, 10)  # 20 张

decoded_list = []
with torch.no_grad():
    for r in ratios:
        z = (r * z1 + (1 - r) * z2) / torch.sqrt(r ** 2 + (1 - r) ** 2)
        samples = diffusion.sample(z, y, y_null, sample_steps=num_steps, cfg=cfg_scale, mode=mode)
        
        # [B, T, D] -> [B, D, 16, 16]
        # B, T, D = samples[-1].shape
        # samples_latent = samples[-1].permute(0, 2, 1).reshape(B, D, 16, 16)
        decoded = vae.decode(samples[-1] / 0.18215).sample
        # decoded = decoder(samples_latent)  # [B,3,H,W]
        decoded_list.append(decoded)

# 拼接成一行
decoded_all = torch.cat(decoded_list, dim=0)   # [20*B, 3, H, W]
grid = make_grid(decoded_all, nrow=len(ratios), normalize=True, value_range=(-1,1))

# 保存
save_path = f"vae_interp_{exp_name}_{step}_sample{num_steps}_{mode}_cfg{cfg_scale}.png"
save_image(grid, save_path)
display(Image.open(save_path))
