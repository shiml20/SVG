# %% [markdown]
# # Scalable Diffusion Models with Transformer (DiT)
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
from ldm.models.dino_decoder import DinoDecoder

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

# ✅ 修改这里选择 checkpoint
ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0158-E0013_Dense_XL_Flow_Dinov3_vitsp_resNorm_BS256_cache_qknormF-GPU8/checkpoints/0300000.pt"

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

# Dino 解码器
encoder_config = OmegaConf.load(
    "/ytech_m2v3_hdd/yuanziyang/sml/FVG/config/f16d32_ldm_dinov3_vitsp_plus_mask09_norm.yaml"
)
dinov3 = DinoDecoder(
    ddconfig=encoder_config.model.params.ddconfig,
    dinoconfig=encoder_config.model.params.dinoconfig,
    lossconfig=encoder_config.model.params.lossconfig,
    embed_dim=encoder_config.model.params.embed_dim,
    ckpt_path=encoder_config.ckpt_path,
    extra_vit_config=encoder_config.model.params.extra_vit_config,
).to(device).eval()

# %% -------------------------
# 3. 插值采样函数
# ----------------------------
def interpolate_and_decode(
    model, decoder, 
    z1, z2, y, y_null,
    num_steps=20, cfg_scale=4, 
    mode="euler", timestep_shift=0.0, 
    num_interp=10
):
    """对两个 latent 向量进行插值，采样并解码。"""
    diffusion = RectifiedFlow(model)
    ratios = torch.linspace(0, 1, num_interp, device=z1.device)

    decoded_list = []
    for r in ratios:
        z = (r * z1 + (1 - r) * z2) / torch.sqrt(r**2 + (1 - r) ** 2 + 1e-8)
        samples = diffusion.sample(
            z, y, y_null, 
            sample_steps=num_steps, cfg=cfg_scale, 
            mode=mode, timestep_shift=timestep_shift
        )

        # [B, T, D] -> [B, D, 16, 16]
        B, T, D = samples[-1].shape
        samples_latent = samples[-1].permute(0, 2, 1).reshape(B, D, 16, 16)
        
        decoded = decoder.decode(samples_latent)
        decoded_list.append(decoded)

    return torch.cat(decoded_list, dim=0)   # [num_interp*B, 3, H, W]

# %% -------------------------
# 4. 运行插值可视化
# ----------------------------
# 参数设置
seed = 0
torch.manual_seed(seed)
num_steps = 20
cfg_scale = 8
class_labels = [279]  # 🐱 类别 ID，可改
y = torch.tensor(class_labels, device=device)
y_null = torch.tensor([1000] * len(class_labels), device=device)

# 随机 latent
z1 = torch.randn(len(class_labels), 256, 392, device=device)
z2 = torch.randn_like(z1)

# 插值采样
decoded_all = interpolate_and_decode(
    model, dinov3, 
    z1, z2, y, y_null,
    num_steps=num_steps, cfg_scale=cfg_scale, 
    mode="euler", timestep_shift=0.4, 
    num_interp=10
)

# 保存 & 显示
grid = make_grid(decoded_all, nrow=10, normalize=True, value_range=(-1, 1))
save_path = f"interp_{exp_name}_{step}_steps{num_steps}_cfg{cfg_scale}_{class_labels[0]}.png"
save_image(grid, save_path)
display(Image.open(save_path))
print("Saved to:", save_path)
