import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from copy import deepcopy
from time import time
import os
import argparse
from omegaconf import OmegaConf

from rectified_flow.rectified_flow import RectifiedFlow
from utils import (
    update_ema, requires_grad, cleanup, setup_ddp, setup_exp_dir, setup_data,
    instantiate_from_config, get_lr_scheduler_config
)
from utils import find_model


def main(config_path):
    config = OmegaConf.load(config_path)
    rank, device, seed = setup_ddp(config.basic)
    logger, writer, checkpoint_dir = setup_exp_dir(rank, config)

    # === Build Model ===
    model = instantiate_from_config(config.model)
    if config.model.ckpt is not None:
        state_dict = find_model(config.model.ckpt, is_train=True)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded model checkpoint from {config.model.ckpt}")

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # === EMA ===
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])

    # === Rectified Flow ===
    logger.info("Training with Rectified Flow")
    diffusion = RectifiedFlow(model)

    # === Optimizer ===
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.base_learning_rate,
        weight_decay=config.optim.weight_decay,
        betas=config.optim.betas,
    )
    max_grad_norm = config.basic.clip_grad_norm

    # === Data Loading ===
    dataset, sampler, loader = setup_data(rank, config.basic)
    logger.info(f"Dataset contains {len(dataset):,} images ({config.basic.data_path})")

    # === SVG(DINO+Res) Feature Encoder ===
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "autoencoder")))
    from ldm.models.dino_decoder import DinoDecoder
    encoder_config = OmegaConf.load(config.basic.encoder_config)
    svg_autoencoder = DinoDecoder(
        ddconfig=encoder_config.model.params.ddconfig,
        dinoconfig=encoder_config.model.params.dinoconfig,
        lossconfig=encoder_config.model.params.lossconfig,
        embed_dim=encoder_config.model.params.embed_dim,
        ckpt_path=encoder_config.ckpt_path,
        extra_vit_config=encoder_config.model.params.extra_vit_config,
        is_train=False
    ).cuda().eval()

    # === Initialize DINO Feature Statistics ===
    dinov3_sp_stats = torch.load("./dinov3_sp_stats.pt")
    dinov3_sp_mean = dinov3_sp_stats["dinov3_sp_mean"].to(device)[:, :, :encoder_config.model.params.ddconfig.z_channels]
    dinov3_sp_std = dinov3_sp_stats["dinov3_sp_std"].to(device)[:, :, :encoder_config.model.params.ddconfig.z_channels]

    # === Initialize Training States ===
    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    global_step = 0
    train_steps = 0
    running_loss = {}
    accum_iter = config.basic.accum_iter
    log_every = config.basic.log_every
    ckpt_every = config.basic.ckpt_every
    start_time = time()

    logger.info(f"Training for {config.basic.epochs} epochs...")
    for epoch in range(config.basic.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Starting epoch {epoch}...")
        for x, y in loader:

            x, y = x.to(device), y.to(device)

            # === DINO Feature Extraction ===
            with torch.no_grad():
                x = svg_autoencoder.encode(x)
                B, D, H, W = x.shape
                x = x.view(B, D, H * W).contiguous().permute(0, 2, 1)

            # === Feature Normalization ===
            if config.basic.feature_norm:
                x = (x - dinov3_sp_mean) / dinov3_sp_std

            # === RF Forward and Loss Computation ===
            loss_dict = diffusion.forward(x, y, config.basic.shift)
            loss = loss_dict["loss"].mean()
            loss.backward()

            # === Gradient Accumulation ===
            if (global_step + 1) % accum_iter == 0:
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                opt.zero_grad()
                update_ema(ema, model.module)
                train_steps += 1

            # === Logging ===
            for k, v in loss_dict.items():
                running_loss[k] = running_loss.get(k, 0) + v.mean().item() / accum_iter

            global_step += 1
            if global_step % (log_every * accum_iter) == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = (log_every * accum_iter) / (end_time - start_time)

                log_msg = f"[Step {global_step}] "
                for k, v in running_loss.items():
                    avg_loss = torch.tensor(v / log_every, device=device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                    log_msg += f"{k}: {avg_loss:.4f} "
                    if rank == 0:
                        writer.add_scalar(k, avg_loss, train_steps)

                log_msg += f"LR={opt.param_groups[0]['lr']:.6f}, Speed={steps_per_sec:.2f} it/s"
                logger.info(log_msg)

                running_loss = {}
                start_time = time()

            # === Save Checkpoint ===
            if global_step % (ckpt_every * accum_iter) == 0 and rank == 0:
                checkpoint = {"ema": ema.state_dict()}
                ckpt_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")

        logger.info(f"Epoch {epoch} done. Step={train_steps}")

    logger.info("Training finished.")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
