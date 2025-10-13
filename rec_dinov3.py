import os
import glob
import torch
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
from IPython.display import display
from omegaconf import OmegaConf
from tqdm import tqdm

from diffusion.rectified_flow_ori import RectifiedFlow
from utils import instantiate_from_config
from download import find_model
from ldm.modules.diffusionmodules.model import Decoder
from ldm.models.dino_decoder import DinoDecoder

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

# %% -------------------------
# 1. 配置加载
# ----------------------------

encoder_config = OmegaConf.load("/ytech_m2v3_hdd/yuanziyang/sml/FVG/config/f16d32_ldm_dinov3_vitsp.yaml")

dinov3 = DinoDecoder(
    ddconfig=encoder_config.model.params.ddconfig,
    dinoconfig=encoder_config.model.params.dinoconfig,
    lossconfig=encoder_config.model.params.lossconfig,
    embed_dim=encoder_config.model.params.embed_dim,
    ckpt_path=encoder_config.ckpt_path,
    extra_vit_config=encoder_config.model.params.extra_vit_config,
).to(device).eval()

z_channels = encoder_config.model.params.ddconfig.z_channels

# %% -------------------------
# 2. 图像加载与预处理
# ----------------------------

# 图像路径配置（请替换为您的图像文件夹路径）
image_dir = "/ytech_m2v3_hdd/yuanziyang/sml/Video-Decoder/src/exp/T000-mstep_wan_dinov3/version_3/val_samples/step_3400/rank_0"  # 包含1000类图像的文件夹
out_dir = "./output_single_images"   # 输出保存路径
os.makedirs(out_dir, exist_ok=True)

# 定义图像变换（使用您提供的配置）
transform = transforms.Compose([
    # 根据DINOv3预训练设置，使用518x518分辨率
    transforms.Resize(1024, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),  # ImageNet均值
        std=(0.229, 0.224, 0.225),   # ImageNet方差
        inplace=True
    ),
])

# 加载图像路径（假设图像按类别分文件夹存放，或直接在image_dir下）
image_paths = glob.glob(os.path.join(image_dir, "**", "*.png"), recursive=True) + \
              glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True)
print(f"找到{len(image_paths)}张图像")

# 批量处理图像
batch_size = 16  # 根据GPU内存调整
for i in tqdm(range(0, len(image_paths), batch_size), desc="处理图像"):
    batch_paths = image_paths[i:i+batch_size]
    
    # 加载并预处理批量图像
    batch_imgs = []
    for img_path in batch_paths:
        try:
            img = Image.open(img_path).convert("RGB")  # 确保为RGB格式
            img_tensor = transform(img).unsqueeze(0)   # 添加批次维度
            batch_imgs.append(img_tensor)
        except Exception as e:
            print(f"处理图像{img_path}失败: {e}")
            continue
    
    if not batch_imgs:
        continue
        
    # 拼接为批次张量并移动到设备
    imgs = torch.cat(batch_imgs, dim=0).to(device)
    B = imgs.shape[0]  # 实际批次大小（可能小于batch_size）
    import ipdb; ipdb.set_trace()
    
    # 编码与解码
    samples_latent = dinov3.encode(imgs)
    with torch.no_grad():
        decoded_full = dinov3.decode(samples_latent)
    decoded_full = torch.clamp(decoded_full, -1, 1)
    
    # 单张保存（这里假设class_id和seed等参数需要从路径中解析，您可根据实际情况调整）
    for j in range(B):
        # 解析图像对应的类别ID（示例逻辑，需根据您的文件结构修改）
        img_path = batch_paths[j]
        print(img_path)
        basename = img_path.split('/')[-1].split('.')[0]
        # class_id = int(os.path.basename(os.path.dirname(img_path)))  # 假设父文件夹名为类别ID
        # seed = 42  # 可根据需要修改或从文件名解析
        # num_steps = 1000  # 根据实际扩散步数修改
        # cfg_scale = 1.0   # 根据实际配置修改
        # timestep_shift = 0  # 根据实际配置修改
        
        save_path = os.path.join(
            out_dir,
            f"{basename}_rec.png"
        )
        save_image(decoded_full[j], save_path, nrow=1, normalize=True, value_range=(-1, 1))