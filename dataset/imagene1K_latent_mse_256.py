import os
import json
import glob
import torch


class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, latent_dir, class_to_idx):
        self.latent_path_list = glob.glob(os.path.join(latent_dir, "*", "*.pt"))
        with open(class_to_idx, "r") as f:
            self.class_to_idx = json.load(f)
        self.use_latent = True

    def __getitem__(self, idx):
        latent_path = self.latent_path_list[idx]
        latent = torch.load(latent_path).squeeze(0).detach()
        latent.requires_grad = False
        label = latent_path.split("/")[-2]
        label = self.class_to_idx[label]
        return (latent, label)


    def __len__(self,):
        return len(self.latent_path_list)