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
import sys
import glob
import torch
from torchvision.utils import save_image
from diffusion.rectified_flow_ori import RectifiedFlow
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
ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0156-E0014_Dense_XL_Flow_vavae_BS256-GPU8/checkpoints/0200000.pt"
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
latent_size = int(image_size) // 16
# latent_size = 384
model = instantiate_from_config(config.model)
state_dict = find_model(ckpt_path)


# %%
model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval() # important!
# 假设 tokenizer 文件夹在 /ytech_m2v3_hdd/yuanziyang/sml/Feature-Visual-Generation 目录下
tokenizer_path = "/ytech_m2v3_hdd/yuanziyang/sml/Feature-Visual-Generation"

# 加到 sys.path
if tokenizer_path not in sys.path:
    sys.path.append(tokenizer_path)

# 再导入
from tokenizer.vavae import VA_VAE
# 使用
vae = VA_VAE("/ytech_m2v3_hdd/yuanziyang/sml/Feature-Visual-Generation/vavae/configs/f16d32_vfdinov2.yaml").load().model


# %% [markdown]
# # 2. Sample from Pre-trained DiT Models   
# You can customize several sampling options. For the full list of ImageNet classes, [check out this](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).

# %%
# Set user inputs:
seed = 0 #@param {type:"number"}
torch.manual_seed(seed)
num_sampling_steps = 10 #@param {type:"slider", min:0, max:1000, step:1}
cfg_scale = 4 #@param {type:"slider", min:1, max:10, step:0.1}
class_labels = 207, 360, 387, 974, 88, 979, 417, 279 #@param {type:"raw"}
# class_labels = 207, 360, 387, 974, 88, 979, 417, 279 #@param {type:"raw"}
class_labels = [279] * 20
samples_per_row = 4 #@param {type:"number"}

# Create diffusion object:
diffusion = RectifiedFlow(model)

# Create sampling noise:
n = len(class_labels)
z = torch.randn(n, 32, latent_size, latent_size, device=device)
y = torch.tensor(class_labels, device=device)

# Setup classifier-free guidance:
# z = torch.cat([z, z], 0)
y_null = torch.tensor([1000] * n, device=device)
# y = torch.cat([y, y_null], 0)
model_kwargs = dict(y=y, cfg_scale=cfg_scale)
timestep_shift = 0.4

# Sample images:
samples = diffusion.sample(
    z, y, y_null, sample_steps=num_sampling_steps, cfg=cfg_scale, timestep_shift=timestep_shift
    # model_kwargs=model_kwarg, progress=True, device=device
)

latents_stats = torch.load("/ytech_m2v3_hdd/yuanziyang/sml/Feature-Visual-Generation/configs/vavae_ckpt/latents_stats.pt")
latent_mean = latents_stats["mean"].to(device)
latent_std = latents_stats["std"].to(device)
latent_multiplier = 1.0

# samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
# samples = vae.decode(samples[-1] / 0.18215).sample
with torch.no_grad():
    # import ipdb; ipdb.set_trace()
    # samples = (samples[-1] * latent_std) / latent_multiplier + latent_mean
    samples = (samples[-1])
    samples = vae.decode(samples)

# Save and display images:
save_image(samples, f"sample_{exp_name}_{step}_sample{num_sampling_steps}_cfg_scale{cfg_scale}_shift{timestep_shift}.png", nrow=int(samples_per_row), 
           normalize=True, value_range=(-1, 1), )
# samples = Image.open(f"sample_{exp_name}.png")
# display(samples)
