# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via evaluation/evaluator.py

"""

import os
import glob
from logging import config
import torch
import torch.distributed as dist
# from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from diffusion.rectified_flow_ori import RectifiedFlow
from diffusers.models import AutoencoderKL
from tqdm import tqdm

from PIL import Image
import numpy as np
import math
import argparse

from typing import Dict, Tuple
from omegaconf import OmegaConf
from utils import instantiate_from_config

from evaluation.inception import InceptionV3

def get_config(ckpt_path):
    exp_root = ckpt_path.split("/")[:-2]
    exp_name = exp_root[-1]
    exp_root = "/".join(exp_root)
    # 转义路径中的特殊字符
    # escaped_path = glob.escape(os.path.join(exp_root, "*.yaml"))
    # config_path = glob.glob(escaped_path)
    config_path = glob.glob(os.path.join(exp_root, "*.yaml"))

    try:
        print(config_path)
        assert len(config_path) == 1 
    except:
        print(config_path)
        raise AssertionError("len(config_path) != 1 ")
    config_path = config_path[0]
    config = OmegaConf.load(config_path)
    return exp_name, config

def create_npz_from_sample_folder(sample_dir, num=50_000, batch_size=4):
    """
    Builds a single .npz file from a folder of activations samples.
    """
    activations = []
        
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        feature = np.load(f"{sample_dir}/{i:06d}.npy")
        activations.append(feature)

    activations = np.concatenate(activations)
    assert activations.shape == (num, 2048)
    npz_path = f"{sample_dir}.npz" # save both samples and statistics
    mu = np.nanmean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    np.savez(npz_path, activations=activations, mu=mu, sigma=sigma)
    print(f"Saved .npz file to {npz_path} [shape={activations.shape}].")
    return npz_path


def create_npz_from_sample_folder_png(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)

    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path



@torch.no_grad()
def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load model:
    latent_size = args.image_size // 8

    ckpt_path = args.ckpt
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

    # 检查并添加 'rf' 键
    print(config)
    if 'rf' not in config.basic:
        config.basic.rf = False  # 或者你可以根据需要设置为其他默认值


    inception = InceptionV3().to(device).eval()
    if config.basic.rf:
        print("sample with rectified flow")
        diffusion = RectifiedFlow(model)
    else:
        diffusion = create_diffusion(str(args.num_sampling_steps))  # default: 1000 steps, linear noise schedule

    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0
    if not using_cfg:
        print('not use cfg')

    # Create folder to save samples:
    # model_string_name = args.model.replace("/", "-")

    vae_name = args.vae.split("-")[-1]
    folder_name = f"{args.tag}_{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{vae_name}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}-FID-{int(args.num_fid_samples/1000)}K-bs{args.per_proc_batch_size}-sampling_{args.num_sampling_steps}-{args.shift}ema"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    # num_samples = len([name for name in os.listdir(sample_folder_dir) if (os.path.isfile(os.path.join(sample_folder_dir, name)) and ".png" in name)])
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    # done_iterations = int( int(num_samples // dist.get_world_size()) // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)
        y_null = None
        # Setup classifier-free guidance:
        if using_cfg:
            z_cat = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y_cat = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y_cat, cfg_scale=args.cfg_scale)  
          
            if not isinstance(model, Dict):
                sample_fn = model.forward_with_cfg  
            else:
                sample_fn = {}
                for k, v in model.items():
                    sample_fn[k] = v.forward_with_cfg

        else:
            z_cat = z
            model_kwargs = dict(y=y)
            # sample_fn = model.forward
            if not isinstance(model, Dict):
                sample_fn = model.forward  
            else:
                sample_fn = {}
                for k, v in model.items():
                    sample_fn[k] = v.forward


        # Sample images:
        if config.basic.rf:
            # Sample images:
            samples = diffusion.sample(
                z, y, y_null, sample_steps=args.num_sampling_steps, cfg=args.cfg_scale, progress=False, mode=args.tag, timestep_shift=args.shift
                # model_kwargs=model_kwargs, progress=True, device=device
            )
            
            # samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            samples = vae.decode(samples[-1] / 0.18215).sample

        else:
            diffusion = create_diffusion(str(args.num_sampling_steps))  # default: 1000 steps, linear noise schedule
            samples = diffusion.p_sample_loop(
                sample_fn, z_cat.shape, z_cat, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples


            samples = vae.decode(samples / 0.18215).sample


        samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
        # now [0, 255]
        inception_feature = inception(samples / 255.).cpu().numpy()
        
        # Save samples to disk as individual .png files
        index = rank + total
        np.save(f"{sample_folder_dir}/{index:06d}.npy", inception_feature)
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        # Make sure all processes have finished saving their samples before attempting to convert to .npz
        def get_all_filenames_in_folder(folder_path):
            if not os.path.isdir(folder_path):
                print(f"Error: {folder_path} is an illegal path. ")
                return []
            filenames = os.listdir(folder_path, )
            return filenames
        sample_dir = sample_folder_dir + '/'
        filenames = get_all_filenames_in_folder(sample_dir)

        def create_npz_from_sample_folder(sample_dir, num=args.num_fid_samples, batch_size=50*4):
            """
            Builds a single .npz file from a folder of .png samples.
            """
            activations = []
            cnt = 0
            for name in tqdm(filenames):
                feature = np.load(sample_dir+name)
                activations.append(feature)
                # print(feature.shape)
                cnt += 1

            activations = np.concatenate(activations)
            print(activations.shape)
            assert activations.shape == (num, 2048)
            npz_path = f"sml_samples/{folder_name}.npz" # save both samples and statistics
            mu = np.mean(activations, axis=0)
            sigma = np.cov(activations, rowvar=False)
            np.savez(npz_path, activations=activations, mu=mu, sigma=sigma)
            print(f"Saved .npz file to {npz_path} [shape={activations.shape}].")
            return npz_path
        print(filenames)
        create_npz_from_sample_folder(sample_dir, num=total_samples)
        print("Done.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    # parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--vae",  type=str, default="stabilityai/sd-vae-ft-mse")
    
    parser.add_argument("--sample-dir", type=str, default="/ytech_m2v2_hdd/yuanziyang/projects/VGI-Interns/analysis_t2v/DiT-MoE-Project/yzy_samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--shift",  type=float, default=0.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    # parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
    #                     help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--tf32", action="store_false")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--tag",  type=str, default="")


    args = parser.parse_args()
    main(args)