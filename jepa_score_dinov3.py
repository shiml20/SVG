import os
import torch
import glob
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm
from ldm.models.dino_decoder import DinoDecoder
from torch.autograd.functional import jacobian

# --- 0. 环境设置 ---
torch.set_grad_enabled(False) # 全局禁用梯度，因为我们只做前向传播和Jacobian计算
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- 1. 配置和模型加载 ---
print("Loading DINOv3 model...")
encoder_config = OmegaConf.load("/ytech_m2v3_hdd/yuanziyang/sml/FVG/config/f16d32_ldm_dinov3_vitsp.yaml")
dinov3 = DinoDecoder(
    ddconfig=encoder_config.model.params.ddconfig,
    dinoconfig=encoder_config.model.params.dinoconfig,
    lossconfig=encoder_config.model.params.lossconfig,
    embed_dim=encoder_config.model.params.embed_dim,
    ckpt_path=encoder_config.ckpt_path,
    extra_vit_config=encoder_config.model.params.extra_vit_config,
).to(device).eval()
print("Model loaded.")

# --- 2. 图像路径和预处理 ---
# 请替换为您的图像文件夹路径
image_dir = "/ytech_m2v3_hdd/yuanziyang/sml/Video-Decoder/src/exp/T000-mstep_wan_dinov3_jdb/version_0/val_samples/step_0/rank_2/"
# 您可以调整图像尺寸，尺寸越小，计算越快
IMAGE_SIZE = 256

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), inplace=True),
])

# 查找所有支持的图像文件
image_paths = glob.glob(os.path.join(image_dir, "*.png")) + \
              glob.glob(os.path.join(image_dir, "*.jpg")) + \
              glob.glob(os.path.join(image_dir, "*.jpeg"))

if not image_paths:
    print(f"错误: 在目录 '{image_dir}' 中未找到任何图像。请检查路径。")
    exit()

print(f"Found {len(image_paths)} images to process.")

# --- 3. JEPA-SCORE 计算核心函数 ---
def calculate_jepa_score(model, x_tensor):
    """
    计算单张图像的 JEPA-SCORE。
    输入 x_tensor 的形状应为 [C, H, W]。
    """
    # 确保输入在正确的设备上
    x_tensor = x_tensor.to(device)

    # 使用自动混合精度以加速并减少内存使用
    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
        # 论文中的Jacobian计算。输入需要增加一个批次维度。
        # model.encode的输出需要展平 (flatten) 以便 .sum()
        J = jacobian(lambda t: model.encode(t).flatten(1).sum(0), inputs=x_tensor.unsqueeze(0))

    # J 的形状可能是 (embed_dim, 1, 3, H, W)，需要重塑为 2D 矩阵
    J = J.view(J.shape[0], -1)

    # SVD 计算
    eps = 1e-6
    svdvals = torch.linalg.svdvals(J)
    
    # 计算 JEPA-SCORE 标量值
    score = svdvals.clip_(eps).log().sum()
    
    # .item() 将PyTorch标量转换为Python数字
    return score.item()

# --- 4. 主处理循环 ---
if __name__ == "__main__":
    results = {}
    
    # 使用tqdm显示进度条
    for img_path in tqdm(image_paths, desc="Calculating JEPA-SCORES"):
        try:
            # 加载并预处理图像
            image = Image.open(img_path).convert("RGB")
            image_tensor = transform(image)
            
            # 计算分数
            score = calculate_jepa_score(dinov3, image_tensor)
            
            # 存储结果
            image_name = os.path.basename(img_path)
            results[image_name] = score
            
            # (可选) 实时打印结果
            # print(f"Image: {image_name}, JEPA-SCORE: {score:.4f}")

        except Exception as e:
            print(f"处理图像 '{img_path}' 时出错: {e}")
            continue
            
    # --- 5. 打印最终结果 ---
    print("\n--- JEPA-SCORE 计算完成 ---")
    if results:
        # 按分数排序（从高到低）
        sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
        
        print("图像分数排名 (从高到低):")
        for image_name, score in sorted_results:
            print(f"- {image_name}: {score:.4f}")
            
        # (可选) 将结果保存到文件
        # import json
        # with open('jepa_scores.json', 'w') as f:
        #     json.dump(dict(sorted_results), f, indent=4)
        # print("\n结果已保存到 jepa_scores.json")
    else:
        print("没有成功处理任何图像。")

