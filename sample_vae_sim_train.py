"""
Scalable Diffusion Models with Transformer (DiT) Sampling Script
This script loads a pre-trained DiT model, encodes an image with VAE, adds noise to the latent,
and performs conditional sampling.
"""

import os
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.utils import save_image

from diffusers.models import AutoencoderKL
from utils import instantiate_from_config
from omegaconf import OmegaConf
from download import find_model

# -------------------------
# Device
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
if device == "cpu":
    print("GPU not found. Using CPU instead.")

# -------------------------
# 1. Load Config and Model
# -------------------------
def get_config(ckpt_path):
    """Load configuration from the checkpoint path."""
    exp_root = ckpt_path.split("/")[:-2]
    exp_name = exp_root[-1]
    exp_root = "/".join(exp_root)
    config_path = glob.glob(os.path.join(exp_root, "*.yaml"))
    assert len(config_path) == 1, "Expected exactly one config file."
    config = OmegaConf.load(config_path[0])
    return exp_name, config

ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints/2000000.pt"
exp_name, config = get_config(ckpt_path)
step = ckpt_path.split('/')[-1].split('.')[0]

# Instantiate model and load state
model = instantiate_from_config(config.model)
state_dict = find_model(ckpt_path)
model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()

# Load VAE
vae_model_name = "stabilityai/sd-vae-ft-ema"
vae = AutoencoderKL.from_pretrained(vae_model_name).to(device)

# -------------------------
# 2. Image Preprocessing
# -------------------------
def center_crop_arr(pil_image, image_size):
    """Center crop PIL image to given size."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def get_transform(image_size=256):
    """Return preprocessing transform matching DiT training."""
    return transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.RandomHorizontalFlip(p=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

image_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/test_sample_0182-E0020_XL_Flow_Dinov3sp_BS256_qknorm_shift1_featureNorm-GPU8_0500000_sample100_euler_cfg4_shift0.15.png"
image_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/selected/class0330_seed1_idx3_steps50_cfg4_shift0.15.png"
pil_image = Image.open(image_path).convert('RGB')
image_size = 256
transform = get_transform(image_size)
processed_image = transform(pil_image).unsqueeze(0).to(device)

# -------------------------
# 3. Encode image to latent
# -------------------------
with torch.no_grad():
    latent = vae.encode(processed_image).latent_dist.sample().mul_(0.18215)  # [1, C, H/8, W/8]

# -------------------------
# 4. Add noise
# -------------------------
def add_noise(latent, t=0.3, B=1):
    """Add noise to latent tensor with factor t."""
    noise = torch.randn_like(latent)
    t_tensor = torch.tensor([t]*B, device=latent.device)
    texp = t_tensor.view([B, *([1] * len(latent.shape[1:]))])
    noisy_latent = texp * latent + (1-texp) * noise
    return noisy_latent

B = latent.shape[0]

for t in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
    noisy_latent = add_noise(latent, t=t, B=B)
    print(f"Noisy latent shape: {noisy_latent.shape}")

    # -------------------------
    # 5. Conditional Sampling
    # -------------------------
    seed = 0
    torch.manual_seed(seed)

    num_steps = 25
    cfg_scale = 4
    class_labels = [330]  # ImageNet class
    cfg_mode = "constant"

    y = torch.tensor(class_labels, device=device)
    y_null = torch.tensor([1000]*len(class_labels), device=device)  # null class

    # Forward pass (single step)
    model_output = model(noisy_latent, torch.tensor([t]*B, device=device), y)
