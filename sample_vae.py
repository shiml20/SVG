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
ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-SiT-XL/checkpoints/0100000.pt"
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
seed = 0 #@param {type:"number"}
torch.manual_seed(seed)
num_sampling_steps = 100 #@param {type:"slider", min:0, max:1000, step:1}
cfg_scale = 1 #@param {type:"slider", min:1, max:10, step:0.1}
class_labels = 207, 360, 387, 974, 88, 979, 417, 279 #@param {type:"raw"}
class_labels = 798, 714, 443, 302, 418, 916, 499, 641 #@param {type:"raw"}
# class_labels = [279] * 20
samples_per_row = 4 #@param {type:"number"}

# Create diffusion object:
diffusion = RectifiedFlow(model)

# Create sampling noise:
n = len(class_labels)
z = torch.randn(n, 4, latent_size, latent_size, device=device)
y = torch.tensor(class_labels, device=device)

timestep_shift = 0.3

# Setup classifier-free guidance:
# z = torch.cat([z, z], 0)
y_null = torch.tensor([1000] * n, device=device)
# y = torch.cat([y, y_null], 0)
model_kwargs = dict(y=y, cfg_scale=cfg_scale)

# Sample images:
samples = diffusion.sample(
    z, y, y_null, sample_steps=num_sampling_steps, cfg=cfg_scale, timestep_shift=timestep_shift
    # model_kwargs=model_kwarg, progress=True, device=device
)

# samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
samples = vae.decode(samples[-1] / 0.18215).sample

# Save and display images:
save_image(samples, f"sample_{exp_name}_{step}_sample{num_sampling_steps}_cfg{cfg_scale}_shift{timestep_shift}.png", nrow=int(samples_per_row), 
           normalize=True, value_range=(-1, 1))
# samples = Image.open(f"sample_{exp_name}.png")
# display(samples)


# import numpy as np
# from pathlib import Path

# def calculate_average_utilization(file_path="expert_utilization.txt"):
#     # 检查文件是否存在
#     if not Path(file_path).exists():
#         print(f"Error: File {file_path} not found!")
#         return None
    
#     # 读取所有数据
#     with open(file_path, "r") as f:
#         lines = f.readlines()
    
#     # 解析数据
#     data = []
#     for line in lines:
#         try:
#             values = [float(x) for x in line.strip().split(",")]
#             data.append(values)
#         except ValueError:
#             continue
    
#     if not data:
#         print("No valid data found in the file!")
#         return None
    
#     # 转换为numpy数组
#     data_array = np.array(data)
    
#     # 计算统计量
#     averages = np.mean(data_array, axis=0)
#     std_devs = np.std(data_array, axis=0)
#     min_values = np.min(data_array, axis=0)
#     max_values = np.max(data_array, axis=0)
    
#     # 打印结果
#     print("\nExpert Utilization Statistics:")
#     print(f"{'Expert':<8} {'Average':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}")
#     for i, (avg, std, minv, maxv) in enumerate(zip(averages, std_devs, min_values, max_values)):
#         print(f"{i:<8} {avg:.6f}    {std:.6f}    {minv:.6f}    {maxv:.6f}")
    
#     # 返回所有数据供进一步分析
#     return {
#         "data": data_array,
#         "averages": averages,
#         "std_devs": std_devs,
#         "min_values": min_values,
#         "max_values": max_values
#     }

# results = calculate_average_utilization()
    
# # 可选：保存统计结果到新文件
# if results is not None:
#     with open("expert_utilization_stats.txt", "w") as f:
#         f.write("Expert,Average,StdDev,Min,Max\n")
#         for i, (avg, std, minv, maxv) in enumerate(zip(
#             results["averages"],
#             results["std_devs"],
#             results["min_values"],
#             results["max_values"]
#         )):
#             f.write(f"{i},{avg:.6f},{std:.6f},{minv:.6f},{maxv:.6f}\n")
#     print("\nStatistics saved to expert_utilization_stats.txt")

