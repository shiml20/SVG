from torchvision.datasets.utils import download_url
import torch
import os
import time

pretrained_models = {'DiT-XL-2-512x512.pt', 'DiT-XL-2-256x256.pt'}


def find_model(model_name, is_train=False):
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:  # Find/download our pre-trained DiT checkpoints
        return download_model(model_name)
    else:  # Load a custom DiT checkpoint:
        assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
        
        start_time = time.time()

        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage, mmap=True, weights_only=False)
        end_time = time.time()
        load_time = end_time - start_time
        print(f"Model loading time: {load_time:.2f} seconds")
        
        if "ema" in checkpoint:  # supports checkpoints from train.py
            checkpoint = checkpoint["ema"] 
            print("load ema ckpt")
        elif ("model" in checkpoint) and is_train: 
            checkpoint = checkpoint["model"]
            print("load non-ema ckpt")

        return checkpoint


def download_model(model_name):
    """
    Downloads a pre-trained DiT model from the web.
    """
    assert model_name in pretrained_models
    local_path = f'pretrained_models/{model_name}'
    if not os.path.isfile(local_path):
        os.makedirs('pretrained_models', exist_ok=True)
        web_path = f'https://dl.fbaipublicfiles.com/DiT/models/{model_name}'
        download_url(web_path, 'pretrained_models')
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model


if __name__ == "__main__":
    # Download all DiT checkpoints
    for model in pretrained_models:
        download_model(model)
    print('Done.')
