import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType, set_seed
import torch.distributed as dist

from torch.utils.data import DataLoader
import numpy as np
from time import time
import argparse
import logging
import os
from datetime import datetime
from copy import deepcopy

from diffusion import create_diffusion
from diffusion.rectified_flow_t2i import RectifiedFlow
from diffusers.models import AutoencoderKL
from transformers import T5EncoderModel, T5Tokenizer

from utils import update_ema, requires_grad, cleanup, setup_exp_dir, instantiate_from_config
from omegaconf import OmegaConf
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint

# 新增数据相关导入
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.utils.data_sampler import AspectRatioBatchSampler
from torch.utils.data import RandomSampler


from datetime import timedelta



def main(config):
    config = OmegaConf.load(config)
    
    # 初始化Accelerator
    init_handler = InitProcessGroupKwargs()
    accelerator = Accelerator(
        mixed_precision=config.basic.mixed_precision,
        gradient_accumulation_steps=config.basic.gradient_accumulation_steps,
        kwargs_handlers=[init_handler]
    )
    
    # 日志初始化（确保setup_exp_dir正确处理多进程）
    logger, writer, checkpoint_dir = setup_exp_dir(accelerator.process_index, config)
    logger.info(f"Using accelerator: {accelerator.state}")
    
    # 设置随机种子（Accelerator自动处理多进程种子）
    if config.basic.global_seed is not None:
        set_seed(config.basic.global_seed)
    
    # 获取当前进程信息（可选，根据需求使用）
    rank = accelerator.process_index
    device = accelerator.device
    
    # 使用accelerator.print确保打印信息整洁
    accelerator.print(f"Starting rank={rank}, device={device}, world_size={accelerator.num_processes}.")
    

    # 构建模型
    model = instantiate_from_config(config.model)
    if config.model.get('ckpt'):
        state_dict = find_model(config.model.ckpt)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded model from {config.model.ckpt}")
    
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # EMA初始化
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    

    # 构建扩散模型
    if config.basic.rf:
        diffusion = RectifiedFlow(model)
    else:
        diffusion = create_diffusion(timestep_respacing="")
    
    # 优化器
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.base_learning_rate,
        weight_decay=config.optim.weight_decay,
        betas=config.optim.betas
    )
    
    # 加载文本编码组件 --------------------------------------------------
    load_t5_feat = config.data.get('load_t5_feat', False)
    tokenizer = text_encoder = None
    if not load_t5_feat:
        # 从HuggingFace加载T5模型
        t5_path = config.model.get('t5_path', 'PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers')
        tokenizer = T5Tokenizer.from_pretrained(t5_path, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(
            t5_path, subfolder="text_encoder", 
            torch_dtype=torch.float16 if config.basic.mixed_precision == 'fp16' else torch.float32
        ).to(accelerator.device)
        text_encoder.requires_grad_(False)
    
    # 构建数据集 --------------------------------------------------------
    set_data_root(config.data.root)
    dataset = build_dataset(
        OmegaConf.to_container(config.data, resolve=True),
        resolution=config.data.image_size,
        aspect_ratio_type=config.data.aspect_ratio_type,
        max_length=config.model.model_max_length,
        load_t5_feat=load_t5_feat  # 传递文本特征加载方式
    )
    
    # 构建采样器
    if config.data.multi_scale:
        batch_sampler = AspectRatioBatchSampler(
            sampler=RandomSampler(dataset),
            dataset=dataset,
            batch_size=config.basic.batch_size,
            aspect_ratios=dataset.aspect_ratio,
            drop_last=True,
            config=config
        )
        train_loader = build_dataloader(dataset, batch_sampler=batch_sampler)
    else:
        train_loader = build_dataloader(
            dataset,
            batch_size=config.basic.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers
        )
    
    # 准备VAE（如果使用在线编码）
    vae = None
    if not config.data.load_vae_feat:
        vae = AutoencoderKL.from_pretrained(config.basic.vae_path)
        vae = vae.to(accelerator.device)
    
    # 使用Accelerator准备组件
    model, opt, train_loader = accelerator.prepare(model, opt, train_loader)
    
    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    ema.eval()  # EMA model should always be in eval mode

    # 在训练循环开始前初始化
    data_time_all = 0.0
    last_tic = time()
    time_start = time()  # 记录训练开始时间
    total_steps = config.basic.epochs * len(train_loader)  # 总训练步数

    # Variables for monitoring/logging purposes:
    global_step = 0
    train_steps = 0
    log_steps = 0
    running_loss = {}  # 用于累加损失值
    accum_iter = config.basic.gradient_accumulation_steps
    log_every = config.basic.log_every
    ckpt_every = config.basic.ckpt_every
    time_start = time()


    # 训练循环 ----------------------------------------------------------
    for epoch in range(config.basic.epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                # 数据预处理
                # 处理图像特征
                if config.data.load_vae_feat:
                    x = batch[0]
                else:
                    with torch.no_grad():
                        x = vae.encode(batch[0]).latent_dist.sample()
                        x = x * config.basic.scale_factor
                # 处理文本特征
                if load_t5_feat:
                    # 直接加载预计算的文本嵌入
                    y = batch[1]
                    y_mask = batch[2]
                else:
                    # 动态生成文本嵌入
                    with torch.no_grad():
                        # 使用T5处理文本
                        tokenized = tokenizer(
                            batch[1], 
                            max_length=config.model.model_max_length, 
                            padding="max_length", 
                            truncation=True, 
                            return_tensors="pt"
                        ).to(accelerator.device)
                        
                        y = text_encoder(
                            tokenized.input_ids,
                            attention_mask=tokenized.attention_mask
                        ).last_hidden_state
                        y_mask = tokenized.attention_mask[:, None, None]
                
                # import ipdb; ipdb.set_trace()
                
                if config.basic.rf:
                    loss_dict = diffusion.forward(x, y, y_mask)
                    loss = loss_dict["loss"].mean()
                else:
                    if 'same_t_per_batch' not in config.basic:
                        config.basic.same_t_per_batch = False 
                    if config.basic.same_t_per_batch:
                        t = torch.randint(0, diffusion.num_timesteps, (1,), device=device)
                        t = t.expand(x.shape[0])
                    else:
                        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

                    model_kwargs = dict(y=y, mask=y_mask)
                    loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                    loss = loss_dict["loss"].mean()

                # 反向传播
                accelerator.backward(loss)
                if config.basic.clip_grad_norm:
                    accelerator.clip_grad_norm_(model.parameters(), config.basic.clip_grad_norm)
                
                opt.step()
                opt.zero_grad()
                update_ema(ema, model.module)

                # Log loss values:
                for k, v in loss_dict.items():
                    if k not in running_loss:
                        running_loss[k] = 0
                    running_loss[k] += loss_dict[k].mean().item() / accum_iter

                log_steps += 1
                global_step += 1

                # ==================== 日志记录部分 ====================
                if (global_step % log_every == 0) or (global_step == 1):
                    # 计算时间指标
                    t = (time() - last_tic) / config.log_interval
                    t_d = data_time_all / config.log_interval
                    avg_time = (time() - time_start) / (global_step + 1)
                    eta_total = timedelta(seconds=int(avg_time * (total_steps - global_step - 1)))
                    eta_epoch = timedelta(seconds=int(avg_time * (len(train_loader) - step - 1)))
                    
                    # 获取模型参数
                    model_for_log = accelerator.unwrap_model(model)
                    spatial_info = (model_for_log.h, model_for_log.w) if hasattr(model_for_log, 'h') else (None, None)
                    
                    # 结构化日志数据
                    log_data = {
                        'train/lr': opt.param_groups[0]['lr'] if opt.param_groups else 0.0,
                        'meta/epoch': epoch,
                        'time/step': t if t is not None else 0.0,
                        'time/data': t_d if t_d is not None else 0.0,
                        'spatial/height': spatial_info[0],
                        'spatial/width': spatial_info[1]
                    }
                    
                    # 计算平均损失并添加到日志数据
                    for k, v in running_loss.items():
                        avg_loss = torch.tensor(v / config.log_interval, device=device)
                        if accelerator.num_processes > 1:
                            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                            avg_loss = avg_loss.item() / accelerator.num_processes
                        else:
                            avg_loss = avg_loss.item()
                        log_data[f'train/{k}'] = avg_loss
                    
                    # 主进程记录日志
                    if accelerator.is_main_process:
                        # 控制台日志
                        info = (
                            f"Step/Epoch [{global_step}/{epoch}][{step + 1}/{len(train_loader)}]: "
                            f"Total ETA: {eta_total}, Epoch ETA: {eta_epoch}\n"
                            f"Time - Step:{t:.3f}s, Data:{t_d:.3f}s, Avg:{avg_time:.3f}s | "
                            f"LR: {log_data['train/lr']:.3e} | "
                            f"Spatial: ({log_data['spatial/height']}, {log_data['spatial/width']})"
                        )
                        for k, v in running_loss.items():
                            info += f", {k}: {log_data[f'train/{k}']:.4f}"
                        accelerator.print(info)
                        
                        # 写入 TensorBoard
                        for key, value in log_data.items():
                            if value is not None:
                                writer.add_scalar(key, value, global_step=global_step)
                            # else:
                                # logger.warning(f"Value for {key} is None. Skipping logging.")
                        
                        # # 可选：添加直方图记录权重分布
                        # if config.basic.log_weights:
                        #     for name, param in model.named_parameters():
                        #         writer.add_histogram(f"weights/{name}", param, global_step)
                        #         if param.grad is not None:
                        #             writer.add_histogram(f"grads/{name}", param.grad, global_step)
                    
                    # 统一记录到 Accelerator
                    accelerator.log(log_data, step=global_step)
                    
                    # 重置计时器和 running_loss
                    last_tic = time()
                    data_time_all = 0
                    for k in running_loss.keys():
                        running_loss[k] = 0


                # 保存检查点
                if global_step % ckpt_every == 0 and accelerator.is_main_process:
                    save_checkpoint(
                        os.path.join(checkpoint_dir, f'step_{global_step}'),
                        epoch=epoch,
                        model=accelerator.unwrap_model(model),
                        model_ema=ema,
                        optimizer=opt,
                        step=global_step
                    )

                # # 保存检查点（整合两个逻辑）
                # if global_step % config.basic.ckpt_every == 0:
                #     # 只在主进程保存
                #     if accelerator.is_main_process:
                #         # 使用 Accelerator 的解包方法获取原始模型
                #         raw_model = accelerator.unwrap_model(model)
                        
                #         # 构建检查点字典（包含所有必要状态）
                #         checkpoint = {
                #             "model": raw_model.state_dict(),  # 原始模型参数
                #             "ema": ema.state_dict(),          # EMA 模型参数
                #             "opt": opt.state_dict(),          # 优化器状态
                #             "step": global_step,              # 当前训练步数
                #             "args": args                      # 训练参数配置
                #         }
                        
                #         # 生成检查点路径
                #         checkpoint_path = os.path.join(
                #             checkpoint_dir, 
                #             f"step_{global_step:07d}.pt"  # 保持7位零填充命名规范
                #         )
                        
                #         # 保存检查点
                #         torch.save(checkpoint, checkpoint_path)
                        
                #         # 记录日志（使用 Accelerator 的打印方法避免多进程混乱）
                #         accelerator.print(f"[Checkpoint] Saved to {os.path.abspath(checkpoint_path)}")
                #         accelerator.print(f"[Status] Global step: {global_step}, Batch size per device: {y.size(0)}")
                        

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(os.path.join(checkpoint_dir, 'final'), model, ema, opt)
        writer.close()
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)