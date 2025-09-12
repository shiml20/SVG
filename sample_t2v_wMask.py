import torch
import argparse
from omegaconf import OmegaConf
from datetime import datetime
from pathlib import Path
import PIL
from diffusers.utils import make_image_grid
from accelerate import Accelerator
from diffusers.models import AutoencoderKL
from transformers import T5EncoderModel, T5Tokenizer

from utils import instantiate_from_config, cleanup
from diffusion.utils.checkpoint import load_checkpoint
from diffusion.rectified_flow_t2i import RectifiedFlow


import os
from pathlib import Path
from omegaconf import OmegaConf
import glob

def get_config(ckpt_path: str) -> tuple[str, OmegaConf]:
    """
    自动从检查点路径推断配置文件位置
    
    参数:
        ckpt_path: 检查点文件路径，格式应为 
        ".../experiment_name/checkpoints/model.ckpt"
    
    返回:
        (实验名称, 配置对象)
    """
    ckpt_path = Path(ckpt_path).resolve()
    
    # 验证检查点路径
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if ckpt_path.suffix not in ['.pt', '.ckpt', '.pth']:
        raise ValueError(f"Invalid checkpoint file extension: {ckpt_path.suffix}")

    # 自动查找配置目录（检查点父目录的上级）
    config_dir = ckpt_path.parent.parent.parent
    if not config_dir.is_dir():
        raise NotADirectoryError(f"Invalid config directory: {config_dir}")
    
    # 获取实验名称
    exp_name = config_dir.name
    
    # 查找所有可能的配置文件
    config_files = list(config_dir.glob("*.yaml"))
    
    # 验证配置文件
    if len(config_files) == 0:
        raise FileNotFoundError(f"No YAML config found in {config_dir}")
    if len(config_files) > 1:
        raise RuntimeError(
            f"Multiple configs found: {[f.name for f in config_files]}\n"
            f"Expected exactly one config file in {config_dir}"
        )
    
    # 加载配置
    config = OmegaConf.load(config_files[0])
    
    # 添加自动推导的配置项
    config.basic.ckpt_path = str(ckpt_path)
    config.basic.exp_name = exp_name
    
    return exp_name, config



def main(config, args):
    # 初始化Accelerator
    accelerator = Accelerator(mixed_precision=args.precision)
    device = accelerator.device
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建基础模型
    model = instantiate_from_config(config.model).to(device)
    
    # 加载EMA模型
    if args.use_ema:
        ema = deepcopy(model).to(device)
    else:
        ema = model
    
    # 加载预训练权重
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(checkpoint.keys())
    ema.load_state_dict(checkpoint['state_dict_ema'], strict=False)

    ema.eval()
    
    # 初始化采样器
    sampler = RectifiedFlow(
        model=ema.to(device),
    )
    
    # 加载文本编码组件
    tokenizer = T5Tokenizer.from_pretrained(config.model.t5_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        config.model.t5_path, 
        subfolder="text_encoder",
        torch_dtype=torch.float16 if args.precision == "fp16" else torch.float32
    ).to(device)
    text_encoder.requires_grad_(False)
    
    # 加载VAE
    vae = AutoencoderKL.from_pretrained(config.basic.vae_path).to(device)
    
    # 生成参数
    height = args.height or config.data.image_size
    width = args.width or config.data.image_size
    latent_shape = (args.num_samples, 4, height//8, width//8)
    
    # 文本编码处理
    with torch.inference_mode():
        # 正向提示词编码
        tokenized = tokenizer(
            [args.prompt]*args.num_samples,
            max_length=config.model.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)
        cond_emb = text_encoder(
            tokenized.input_ids,
            attention_mask=tokenized.attention_mask
        ).last_hidden_state
        
        # 负向提示词编码
        if args.guidance_scale > 1.0:
            tokenized_uncond = tokenizer(
                [args.negative_prompt]*args.num_samples,
                max_length=config.model.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(device)
            uncond_emb = text_encoder(
                tokenized_uncond.input_ids,
                attention_mask=tokenized_uncond.attention_mask
            ).last_hidden_state
        else:
            uncond_emb = None

    # 生成初始噪声
    z = torch.randn(latent_shape, device=device)
    
    # 生成潜变量
    with accelerator.autocast():
        samples = sampler.sample(
            z=z,
            y=cond_emb,
            y_mask=tokenized.attention_mask,
            null_cond=uncond_emb,
            sample_steps=args.num_sampling_steps,
            cfg=args.guidance_scale,
            mode=args.solver_type,
            progress=accelerator.is_main_process,
        )[-1]
    
    # 解码潜变量为图像
    samples = samples / config.basic.scale_factor
    with torch.inference_mode():
        imgs = vae.decode(samples).sample
    
    # 保存结果
    if accelerator.is_main_process:
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        imgs = imgs.cpu().permute(0, 2, 3, 1).float().numpy()
        imgs = (imgs * 255).round().astype("uint8")
        
        # 保存单张图片
        for i, img in enumerate(imgs):
            pil_img = PIL.Image.fromarray(img)
            filename = output_dir / f"sample_{i}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
            pil_img.save(filename)
            print(f"Saved {filename}")
        
        # 创建并保存拼图
        grid = make_image_grid([PIL.Image.fromarray(x) for x in imgs], rows=1, cols=1)
        grid.save(output_dir / "grid.png")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from text prompts")
    # parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--num_sampling_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--solver_type", type=str, default="euler", 
                      choices=["euler", "heun", "torchdiffeq-dopri5", "torchdiffeq-adams"],
                      help="Sampling method: euler/heun for fixed-step, torchdiffeq-* for adaptive")
    
    args = parser.parse_args()
    exp_name, config = get_config(args.ckpt)
    print(f"Experiment: {exp_name}")
    print(OmegaConf.to_yaml(config))
    main(config, args)