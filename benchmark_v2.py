# 添加到现有导入部分
import time
import numpy as np

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
device = "cuda:1" if torch.cuda.is_available() else "cpu"
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


exp_name = "1013_Dense_S_Flow512_MoBA_blockSize16_interleave_flash_0310"
config = OmegaConf.load("/ytech_m2v2_hdd/sml/DiffMoE_research_local/config/1013_Dense_S_Flow512_MoBA_blockSize16_interleave_flash_0310.yaml")
print("exp_name", exp_name)


# %% [markdown]
# # Download DiT-XL/2 Models
# 
# You can choose between a 512x512 model and a 256x256 model. You can swap-out the LDM VAE, too.

# %%
image_size = 1024 #@param [256, 512]
vae_model = "stabilityai/sd-vae-ft-ema" #@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
latent_size = int(image_size) // 8
model = instantiate_from_config(config.model)
# state_dict = find_model(ckpt_path)


# %%
# model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval() # important!
vae = AutoencoderKL.from_pretrained(vae_model).to(device)

seed = 0 #@param {type:"number"}
torch.manual_seed(seed)
num_sampling_steps = 250 #@param {type:"slider", min:0, max:1000, step:1}
cfg_scale = 4 #@param {type:"slider", min:1, max:10, step:0.1}
class_labels = 207, 360, 387, 974, 88, 979, 417, 279 #@param {type:"raw"}
samples_per_row = 4 #@param {type:"number"}

# Create diffusion object:
diffusion = RectifiedFlow(model)

def benchmark(n_runs=2, warmup=1, bs=4):
    # 预热运行
    print(f"Warming up with {warmup} runs...")
    for _ in range(warmup):
        z = torch.randn(bs, 4, latent_size, latent_size, device=device)
        y = torch.tensor([207]*bs, device=device)  # 示例类别
        y_null = torch.tensor([1000] * bs, device=device)

        with torch.no_grad():
            _ = diffusion.sample(z, y, y_null, sample_steps=50, cfg=cfg_scale)
    

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
            latent = diffusion.sample(z, y, y_null, sample_steps=50, cfg=cfg_scale)
            samples = vae.decode(latent[-1] / 0.18215).sample
        
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
    n_test_runs = 10       # 测试次数
    test_cfg_scale = 4     # 使用和生成时相同的参数
    test_steps = 50
    
    # 运行测试
    benchmark_results = benchmark(n_runs=n_test_runs)