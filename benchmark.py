# DiT imports:
import torch
import numpy as np
import time

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import sys
sys.path.insert(0, "/ytech_m2v2_hdd/sml/DiffMoE_research_local")

from models.models_DiT import DiT_S_2, DiT_XL_2
from models.models_DiT_MoBA_customized_flash_0310 import DiT_S_2 as DiT_S_2_MoBA
import time
from PIL import Image
from IPython.display import display
from diffusion.rectified_flow import RectifiedFlow

torch.set_grad_enabled(False)

# Set user inputs:
seed = 1967 #@param {type:"number"}
torch.manual_seed(seed)


device = "cuda:1" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")

image_size = 4096 #@param [256, 512]
vae_model = "stabilityai/sd-vae-ft-mse" #@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
latent_size = int(image_size) // 8
# Load model:

model_MoBA = DiT_S_2_MoBA(input_size=latent_size, moba_spatial=True, moba_block_size=256, moba_topk=4).to(device)
# state_dict256_moba = find_model(f"/home/sml/ljz_workspace/DiT/results/*050-DiT-S-2-moba_attention_first3full_spatial_efficient-blocksize16-top4-expand1-batchsize192-3GPU/checkpoints/0040000.pt")
model_origin = DiT_S_2(input_size=latent_size).to(device)

model_MoBA = model_MoBA.to(device)
model_MoBA.eval() # important!

# state_dict = find_model(ckpt_path)
model_Dense = model_origin.to(device)
model_Dense.eval() # important!

# Create diffusion object:
diffusion_MoBA = RectifiedFlow(model_MoBA)
diffusion_Dense = RectifiedFlow(model_Dense)

def benchmark(diffusion, n_runs=2, warmup=1, bs=4, steps=20, cfg_scale=4):
    # 预热运行
    print(f"Warming up with {warmup} runs...")
    for _ in range(warmup):
        z = torch.randn(bs, 4, latent_size, latent_size, device=device)
        print("Tokens", z.shape[2] ** 2 // 4)
        y = torch.tensor([207]*bs, device=device)  # 示例类别
        y_null = torch.tensor([1000] * bs, device=device)

        with torch.no_grad():
            _ = diffusion.sample(z, y, y_null, sample_steps=steps, cfg=cfg_scale, progress=True)

    # 正式测速
    timings = []
    print(f"Benchmarking {n_runs} runs...")
    for _ in range(n_runs):
        # 生成新的随机输入
        z = torch.randn(bs, 4, latent_size, latent_size, device=device)
        y = torch.tensor([207]*bs, device=device)  # 固定类别或随机生成
        
        # CUDA同步确保时间测量准确
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            latent = diffusion.sample(z, y, y_null, sample_steps=steps, cfg=cfg_scale, progress=True)
            # samples = vae.decode(latent[-1] / 0.18215).sample
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        timings.append(end_time - start_time)
    
    # 打印结果
    print(f"Average time per run: {(np.mean(timings)/bs):.2f}s")
    print(f"Fastest run: {(np.min(timings)/bs):.2f}s")
    print(f"Slowest run: {(np.max(timings)/bs):.2f}s")
    print(f"Standard deviation: {(np.std(timings)/bs):.2f}s")
    
    return timings

# 使用示例（添加到代码末尾）
if __name__ == "__main__":
    # 设置测试参数
    n_test_runs = 3       # 测试次数
    test_cfg_scale = 4     # 使用和生成时相同的参数
    test_steps = 20
    
    # 运行测试
    benchmark_results = benchmark(diffusion_Dense, n_runs=n_test_runs, bs=2, steps=test_steps)
    benchmark_results = benchmark(diffusion_MoBA, n_runs=n_test_runs, bs=2, steps=test_steps)
