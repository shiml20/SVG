# %% [markdown]
# # Scalable Diffusion Models with Transformer (DiT)
# 
# 优化后的 Notebook，支持完整采样、解码，以及单 token 可视化。

# %%
import os
import glob
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
from IPython.display import display
from omegaconf import OmegaConf

from diffusion.rectified_flow_ori_ablation_guidance import RectifiedFlow
from utils import instantiate_from_config
from download import find_model
from ldm.modules.diffusionmodules.model import Decoder
from ldm.models.dino_decoder import DinoDecoder
import numpy as np

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

# %% -------------------------
# 图像预处理函数
# ----------------------------
def center_crop_arr(pil_image, image_size):
    """
    中心裁剪图像到指定尺寸
    """
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
    """
    获取与训练时相同的预处理变换
    """
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.RandomHorizontalFlip(p=0),  # 设置为0表示不进行随机翻转，保持确定性
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),  # ImageNet 均值
            std=(0.229, 0.224, 0.225),   # ImageNet 方差
        ),
    ])
    return transform

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

ckpt_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints/1100000.pt"
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

encoder_config = OmegaConf.load(config.basic.encoder_config)

dinov3 = DinoDecoder(
    ddconfig=encoder_config.model.params.ddconfig,
    dinoconfig=encoder_config.model.params.dinoconfig,
    lossconfig=encoder_config.model.params.lossconfig,
    embed_dim=encoder_config.model.params.embed_dim,
    ckpt_path=encoder_config.ckpt_path,
    extra_vit_config=encoder_config.model.params.extra_vit_config,
).cuda().eval()

z_channels = encoder_config.model.params.ddconfig.z_channels

# %% -------------------------
# 3. 加载并预处理图像
# ----------------------------
image_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/test_sample_0182-E0020_XL_Flow_Dinov3sp_BS256_qknorm_shift1_featureNorm-GPU8_0500000_sample100_euler_cfg4_shift0.15.png"  # 替换为你的图像路径
image_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/selected/class0330_seed1_idx3_steps50_cfg4_shift0.15.png"
image_size = 256

# 加载图像
pil_image = Image.open(image_path).convert('RGB')

# 应用预处理
transform = get_transform(image_size)
processed_image = transform(pil_image).unsqueeze(0).to(device)  # [1, 3, H, W]

print(f"Processed image shape: {processed_image.shape}")

# %% -------------------------
# 4. 编码图像到潜在空间
# ----------------------------
with torch.no_grad():
    # 使用DINOv3编码器将图像编码到潜在空间
    encoded = dinov3.encode(processed_image)
    import ipdb; ipdb.set_trace()
    # 根据你的模型结构，可能需要调整encoded的形状
    if isinstance(encoded, tuple):
        latent = encoded[0]  # 通常第一个元素是潜在表示
    else:
        latent = encoded
    
    # 重塑潜在表示到 [B, T, D] 格式
    B, C, H, W = latent.shape
    latent_flat = latent.view(B, C, H*W).permute(0, 2, 1)  # [B, T, D]
    
    print(f"Latent shape: {latent_flat.shape}")

# %% -------------------------
# 5. 对潜在表示加噪声
# ----------------------------
# 设置噪声水平

for t in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
    # 生成噪声
    noise = torch.randn_like(latent_flat)

    # 添加噪声
    t = torch.tensor([t]*B, device=latent_flat.device)
    texp = t.view([B, *([1] * len(latent_flat.shape[1:]))])
    noisy_latent = texp * latent_flat + (1-texp) *noise


    print(f"Noisy latent shape: {noisy_latent.shape}")

    # %% -------------------------
    # 6. 使用模型进行去噪生成
    # ----------------------------
    seed = 0
    torch.manual_seed(seed)
    num_steps = 25
    cfg_scale = 4
    class_labels = [330]  # 可以根据你的图像类别修改
    timestep_shift = 0.15
    cfg_mode = "constant"

    # 初始化扩散模型
    diffusion = RectifiedFlow(model)

    # 准备条件
    y = torch.tensor(class_labels, device=device)
    y_null = torch.tensor([1000] * len(class_labels), device=device)


    model_output = model(noisy_latent, t, y)

# # 使用加噪的潜在表示作为起点进行生成
# z_start = noisy_latent

# # 进行去噪生成
# mode = "euler"

# print(f"Generated samples shape: {generated_samples.shape}")

# # %% -------------------------
# # 7. 特征归一化（如果配置需要）
# # ----------------------------
# # if config.basic.get("feature_norm", False):
#     # dinov3_sp_stats = torch.load("dinov3_sp_stats.pt")
#     # dinov3_sp_mean = dinov3_sp_stats["dinov3_sp_mean"].to(device)[:,:,:z_channels]
#     # dinov3_sp_std = dinov3_sp_stats["dinov3_sp_std"].to(device)[:,:,:z_channels]
#     # generated_samples = generated_samples * dinov3_sp_std + dinov3_sp_mean

# # %% -------------------------
# # 8. 解码生成图像
# # ----------------------------
# # 重塑潜在表示回 [B, D, H, W] 格式
# B, T, D = generated_samples.shape
# H = W = int(T ** 0.5)
# generated_latent = generated_samples.permute(0, 2, 1).reshape(B, D, H, W)

# print(f"Reshaped latent shape: {generated_latent.shape}")

# # 解码生成图像
# with torch.no_grad():
#     decoded_images = dinov3.decode(generated_latent)
#     decoded_images = torch.clamp(decoded_images, -1, 1)

# print(f"Decoded images shape: {decoded_images.shape}")

# # %% -------------------------
# # 9. 保存和显示结果
# # ----------------------------
# # 保存生成的图像
# save_path = f"generated_from_image_{exp_name}_{step}.png"
# save_image(decoded_images, save_path, nrow=2, normalize=True, value_range=(-1, 1))

# # 显示原始图像、加噪图像和生成图像对比
# # 首先将处理后的图像反标准化以便显示
# def denormalize(tensor):
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
#     return tensor * std + mean

# # 反标准化处理后的图像
# denorm_original = denormalize(processed_image)
# denorm_generated = denormalize(decoded_images)

# # 创建对比图像网格
# comparison_grid = make_grid(
#     torch.cat([denorm_original, denorm_generated], dim=0), 
#     nrow=2, 
#     normalize=True,
#     value_range=(0, 1)  # 反标准化后值范围是[0,1]
# )

# # 保存对比图
# comparison_path = f"comparison_{exp_name}_{step}.png"
# save_image(comparison_grid, comparison_path)

# # 显示结果
# print("Original image:")
# display(Image.open(image_path).resize((256, 256)))

# print("Generated image:")
# display(Image.open(save_path))

# print("Comparison (Original vs Generated):")
# display(Image.open(comparison_path))

# # %% -------------------------
# # 10. 可选：逐步生成过程可视化
# # ----------------------------
# # 如果你想要查看逐步生成过程，可以修改采样代码保存中间结果
# if False:  # 设置为True来启用逐步可视化
#     all_steps = diffusion.sample(
#         z_start, y, y_null, 
#         sample_steps=num_steps, 
#         cfg=cfg_scale, 
#         mode=mode, 
#         timestep_shift=timestep_shift,
#         cfg_mode=cfg_mode,
#         return_all_steps=True
#     )
    
#     # 解码每个步骤并保存
#     step_images = []
#     for i, step_latent in enumerate(all_steps):
#         if i % 5 == 0:  # 每5步保存一次
#             B, T, D = step_latent.shape
#             step_latent_reshaped = step_latent.permute(0, 2, 1).reshape(B, D, H, W)
            
#             with torch.no_grad():
#                 step_decoded = dinov3.decode(step_latent_reshaped)
#                 step_decoded = torch.clamp(step_decoded, -1, 1)
#                 step_images.append(step_decoded)
    
#     # 创建逐步生成网格
#     if step_images:
#         steps_grid = make_grid(
#             torch.cat(step_images, dim=0), 
#             nrow=len(step_images),
#             normalize=True,
#             value_range=(-1, 1)
#         )
#         steps_path = f"generation_steps_{exp_name}_{step}.png"
#         save_image(steps_grid, steps_path)
#         display(Image.open(steps_path))