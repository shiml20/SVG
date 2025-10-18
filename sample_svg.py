# %% [markdown]
# # Scalable Diffusion Models with Transformer (DiT)
# Optimized version — supports end-to-end sampling, decoding, and visualization.

# %%
import os
import glob
import torch
from torchvision.utils import save_image
from PIL import Image
from IPython.display import display
from omegaconf import OmegaConf

from diffusion.rectified_flow_ori_ablation_guidance import RectifiedFlow
from utils import instantiate_from_config
from utils import find_model
from ldm.models.dino_decoder import DinoDecoder

# -------------------------
# Global setup
# -------------------------
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# 1. Load config
# -------------------------
def get_config(ckpt_path):
    """Automatically locate the corresponding config for a checkpoint."""
    exp_root = "/".join(ckpt_path.split("/")[:-2])
    config_path = glob.glob(os.path.join(exp_root, "*.yaml"))
    assert len(config_path) == 1, f"Expected one config file under {exp_root}"
    config = OmegaConf.load(config_path[0])
    exp_name = os.path.basename(exp_root)
    return exp_name, config


# === User-defined path ===
ckpt_path = ""  # TODO: fill with checkpoint path

assert ckpt_path and os.path.exists(ckpt_path), "Please set a valid ckpt_path"
exp_name, config = get_config(ckpt_path)
step = os.path.splitext(os.path.basename(ckpt_path))[0]
print(f"Experiment: {exp_name}")

# -------------------------
# 2. Load main model & decoder
# -------------------------
# Load DiT backbone
model = instantiate_from_config(config.model)
state_dict = find_model(ckpt_path)
model.load_state_dict(state_dict, strict=False)
model = model.to(device).eval()

# Load DinoDecoder from encoder config
encoder_config = OmegaConf.load(config.basic.encoder_config)
dinov3 = DinoDecoder(
    ddconfig=encoder_config.model.params.ddconfig,
    dinoconfig=encoder_config.model.params.dinoconfig,
    lossconfig=encoder_config.model.params.lossconfig,
    embed_dim=encoder_config.model.params.embed_dim,
    ckpt_path=encoder_config.ckpt_path,
    extra_vit_config=encoder_config.model.params.extra_vit_config,
).to(device).eval()

z_channels = encoder_config.model.params.ddconfig.z_channels


# -------------------------
# 3. Sampling setup
# -------------------------
seed = 0
torch.manual_seed(seed)

num_steps = 25
cfg_scale = 12
image_size = 256
samples_per_row = 4

# Example class labels (ImageNet indices)
class_labels = [130, 270, 284, 688, 250, 146, 980, 484, 207, 360, 387, 974, 88, 979, 417, 279]
n = len(class_labels)

# Initialize latent z
z = torch.randn(n, (image_size // 16) ** 2, z_channels, device=device)

# Load feature normalization stats if applicable
stats_path = "dinov3_sp_stats.pt"
assert os.path.exists(stats_path), "Missing dinov3_sp_stats.pt"
stats = torch.load(stats_path)
sp_mean = stats["dinov3_sp_mean"].to(device)[:, :, :z_channels]
sp_std = stats["dinov3_sp_std"].to(device)[:, :, :z_channels]

# CFG and timestep settings
timestep_shift = 0.15
cfg_mode = "early-0.1"
mode = "euler"

y = torch.tensor(class_labels, device=device)
y_null = torch.full_like(y, 1000)

# Run sampling
diffusion = RectifiedFlow(model)
print(f"Sampling with cfg_mode={cfg_mode}, steps={num_steps}, cfg={cfg_scale}")

samples = diffusion.sample(
    z,
    cond=y,
    null_cond=y_null,
    sample_steps=num_steps,
    cfg=cfg_scale,
    mode=mode,
    timestep_shift=timestep_shift,
    cfg_mode=cfg_mode,
)[-1]

# Apply feature normalization if needed
if config.basic.get("feature_norm", False):
    samples = samples * sp_std + sp_mean

# Reshape [B, T, D] -> [B, D, H, W]
B, T, D = samples.shape
samples_latent = samples.permute(0, 2, 1).reshape(B, D, image_size // 16, image_size // 16)


# -------------------------
# 4. Decode to full-resolution images
# -------------------------
with torch.no_grad():
    decoded_full = dinov3.decode(samples_latent)

decoded_full = torch.clamp(decoded_full, -1, 1)

# Save and visualize
save_path = (
    f"{cfg_mode}_sample_{exp_name}_{step}_"
    f"steps{num_steps}_{mode}_cfg{cfg_scale}_shift{timestep_shift}_{image_size}.png"
)
save_image(decoded_full, save_path, nrow=samples_per_row, normalize=True, value_range=(-1, 1))
display(Image.open(save_path))

print(f"Saved results to {save_path}")
