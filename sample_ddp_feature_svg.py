# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
DDP Sampling Script for SiT/DiT-based Models.
Generates samples, extracts Inception features, and saves a .npz file
for FID computation.
"""

import os
import glob
import math
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import Dict, Tuple

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from utils import instantiate_from_config, find_model
from rectified_flow.rectified_flow import RectifiedFlow
from evaluation.inception import InceptionV3

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def get_config(ckpt_path: str) -> Tuple[str, Dict]:
    """Given checkpoint path, locate and load its corresponding config file."""
    exp_root = "/".join(ckpt_path.split("/")[:-2])
    exp_name = exp_root.split("/")[-1]
    config_files = glob.glob(os.path.join(exp_root, "*.yaml"))
    if len(config_files) != 1:
        raise FileNotFoundError(f"Expected 1 YAML config under {exp_root}, got {config_files}")
    config = OmegaConf.load(config_files[0])
    return exp_name, config


def create_npz_from_folder(sample_dir: str, num: int, filenames: list, folder_name):
    """Aggregate all saved .npy Inception features and save to .npz."""
    activations = []
    for name in tqdm(filenames, desc=f"Building .npz from {sample_dir}"):
        feat = np.load(os.path.join(sample_dir, name))
        activations.append(feat)
    activations = np.concatenate(activations)
    assert activations.shape == (num, 2048), f"Invalid shape: {activations.shape}"

    npz_path = f"samples/{folder_name}.npz" # save both samples and statistics
    mu, sigma = np.mean(activations, axis=0), np.cov(activations, rowvar=False)
    np.savez(npz_path, activations=activations, mu=mu, sigma=sigma)
    print(f"[Rank 0] Saved FID .npz to {npz_path} [shape={activations.shape}]")
    return npz_path


def setup_ddp_and_seed(args):
    """Initialize distributed environment and set deterministic seed."""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    print(f"[DDP] Rank {rank}/{world_size} | Device {device} | Seed {seed}")
    return rank, world_size, device


def load_model_from_ckpt(ckpt_path: str, device: torch.device):
    """Load model(s) and configuration from checkpoint path."""
    if "{" not in ckpt_path:
        exp_name, config = get_config(args.ckpt)
        model = instantiate_from_config(config.model).to(device)
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict)
        # model.eval()  # important!
        print(f"Before Eval, model.training: {model.training}")
        model.eval()
        print(f"After Eval, model.training: {model.training}")
        model_string_name = exp_name
        ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    else:
        print("load model for different timestep")
        print(ckpt_path)
        ckpt_path = eval(ckpt_path)
        print(type(ckpt_path))
        model = {}

        exp_name = ckpt_path["infer_exp_name"]
        ckpt_step = ckpt_path["ckpt_step"]
        del ckpt_path["infer_exp_name"]
        del ckpt_path["ckpt_step"]

        for k, v in ckpt_path.items():
            k = k.split(",")
            k = [int(_) for _ in k]
            k[1]-= 1
            k = tuple(k)
            print("--------> loading model")
            print(f"set timestep from: {k[0]} to {k[1]}")
            print(f'using model: {v}')
            _, config = get_config(v)
            _model = instantiate_from_config(config.model).to(device)
            state_dict = find_model(v)
            _model.load_state_dict(state_dict)
            _model.eval()
            model[k] = _model
            print()
        model_string_name = exp_name
        ckpt_string_name = ckpt_step

    return model, config, exp_name, model_string_name, ckpt_string_name


# -----------------------------------------------------------------------------
# Main Sampling Logic
# -----------------------------------------------------------------------------

@torch.no_grad()
def sample_loop(model, config, encoder_config, inception, diffusion, dinov3, dinov3_stats, args, rank, world_size, device, model_string_name, ckpt_string_name):
    """Core sampling loop per GPU."""

    z_channels = encoder_config.model.params.ddconfig.z_channels
    dinov3_sp_stats = torch.load("dinov3_sp_stats.pt")
    dinov3_sp_mean = dinov3_sp_stats["dinov3_sp_mean"].to(device)[:,:,:z_channels]
    dinov3_sp_std = dinov3_sp_stats["dinov3_sp_std"].to(device)[:,:,:z_channels]

    # Setup directories
    vae_name = args.vae.split("-")[-1]
    exp_tag = f"{args.cfg_mode}_{args.tag}_{model_string_name}_{ckpt_string_name}_{args.image_size}_cfg{args.cfg_scale}"
    folder_name = f"{exp_tag}-seed{args.global_seed}-FID{int(args.num_fid_samples/1000)}K-bs{args.per_proc_batch_size}-steps{args.num_sampling_steps}-shift{args.shift}"
    sample_dir = os.path.join(args.sample_dir, "npy", folder_name)
    if rank == 0:
        os.makedirs(sample_dir, exist_ok=True)
        print(f"[Rank 0] Saving samples to {sample_dir}")
    dist.barrier()

    n = args.per_proc_batch_size
    global_batch = n * world_size
    total_needed = int(math.ceil(args.num_fid_samples / global_batch) * global_batch)
    per_gpu_samples = total_needed // world_size
    iterations = per_gpu_samples // n

    pbar = tqdm(range(iterations), desc=f"Rank {rank}") if rank == 0 else range(iterations)
    latent_size = args.image_size // 16
    total = 0

    for _ in pbar:
        # prepare input noise and labels
        z = torch.randn(n, latent_size**2, z_channels, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)
        y_null = torch.full_like(y, 1000)

        using_cfg = args.cfg_scale > 1.0
        if using_cfg:
            z_cat = torch.cat([z, z], 0)
            y_cat = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y_cat, cfg_scale=args.cfg_scale)
        else:
            z_cat, model_kwargs = z, dict(y=y)

        # choose sample function
        # sampling
        if config.basic.get("rf", False):
            samples = diffusion.sample(
                z, y, y_null, sample_steps=args.num_sampling_steps,
                cfg=args.cfg_scale, progress=False, mode=args.tag,
                timestep_shift=args.shift, cfg_mode=args.cfg_mode
            )
            if config.basic.get("feature_norm", False):
                samples = samples * dinov3_sp_std + dinov3_sp_mean
            B, T, D = samples.shape
            samples = dinov3.decode(samples.permute(0, 2, 1).reshape(B, D, latent_size, latent_size))
        else:
            pass

        samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
        features = inception(samples / 255.).cpu().numpy()
        np.save(f"{sample_dir}/{rank + total:06d}.npy", features)
        total += global_batch

    dist.barrier()
    if rank == 0:
        filenames = [f for f in os.listdir(sample_dir) if f.endswith(".npy")]
        create_npz_from_folder(sample_dir, total_needed, filenames, folder_name)
    dist.barrier()


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------

def main(args):
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    rank, world_size, device = setup_ddp_and_seed(args)

    # load model and diffusion
    model, config, exp_name, model_string_name, ckpt_string_name = load_model_from_ckpt(args.ckpt, device)
    if 'rf' not in config.basic:
        config.basic.rf = False
    diffusion = RectifiedFlow(model) if config.basic.rf else None
    inception = InceptionV3().to(device).eval()

    # load encoder (dinov3)
    encoder_cfg = OmegaConf.load(config.basic.encoder_config)
    dinov3 = instantiate_from_config(encoder_cfg.model).cuda().eval()
    dinov3_stats = torch.load("dinov3_sp_stats.pt")

    sample_loop(model, config, encoder_cfg, inception, diffusion, dinov3, dinov3_stats, args, rank, world_size, device, model_string_name, ckpt_string_name)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--sample-dir", type=str, default="./samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=50)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--shift", type=float, default=0.15)
    parser.add_argument("--num-sampling-steps", type=int, default=25)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--tag", type=str, default="euler")
    parser.add_argument("--cfg_mode", type=str, default="cfg_star-1-0")
    args = parser.parse_args()
    main(args)
