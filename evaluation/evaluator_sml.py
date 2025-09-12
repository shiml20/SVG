import argparse
import io
import os
import random
import warnings
import zipfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import Iterable, Optional, Tuple

import numpy as np
import requests
from scipy import linalg
from tqdm.auto import tqdm

INCEPTION_V3_URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb"
INCEPTION_V3_PATH = "classify_image_graph_def.pb"

FID_POOL_NAME = "pool_3:0"
FID_SPATIAL_NAME = "mixed_6/conv:0"

def read_statistics(npz_path: str):
    obj = np.load(npz_path)

    if "mu" in list(obj.keys()):
        return  obj['activations']
    raise NotImplementedError()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_batch", default="VIRTUAL_imagenet256_labeled.npz")
    parser.add_argument("--ref_npz", default="VIRTUAL_imagenet256_labeled.npz")
    parser.add_argument("--sample_batch", default="/m2v_intern/shiminglei/DiT_MoE_Dynamic/sml_samples/_0354-5000_DiT-L-2_Flow-ECMoE_BatchLevel_CapacityPred_w_Threshold_E8_GPU8_SOTA_resume600K-3500000-size-256-vae-mse-cfg-1.1-seed-0-FID-50K-bs125-ema.npz")
    parser.add_argument("--sample_npz", default="/m2v_intern/shiminglei/DiT_MoE_Dynamic/sml_samples/_0354-5000_DiT-L-2_Flow-ECMoE_BatchLevel_CapacityPred_w_Threshold_E8_GPU8_SOTA_resume600K-3500000-size-256-vae-mse-cfg-1.1-seed-0-FID-50K-bs125-ema.npz")
    args = parser.parse_args()


    # ref_acts = read_statistics(args.ref_npz)
    sample_acts = read_statistics(args.sample_npz)
    print(sample_acts.shape)

main()
