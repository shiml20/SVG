"""
We use the file to instantiate the vision foundation models.
They serves as the auxiliary regularizer for the autoencoder.

by Jingfeng Yao
from HUST-VL
"""

import timm
import torch
import torch.nn as nn

def get_mae_encoder():
    """
    Load the MAE pretrained ViT-L encoder from the timm library.
    """
    model = timm.create_model("hf-hub:timm/vit_large_patch16_224.mae", pretrained=True, dynamic_img_size=True)
    model.requires_grad_(False)
    return model

def get_dinov2_encoder():
    """
    Load the DINOv2 pretrained ViT-L encoder from the timm library.
    """
    model = timm.create_model("hf-hub:timm/vit_large_patch14_dinov2.lvd142m", pretrained=True, dynamic_img_size=True)
    model.requires_grad_(False)
    return model

def create_foundation_model(
    type,
):
    assert type in ['mae', 'dinov2'], f"Unsupported foundation model type: {type}"

    if type == 'mae':
        return get_mae_encoder(), 1024
    elif type == 'dinov2':
        return get_dinov2_encoder(), 1024

class aux_foundation_model(nn.Module):
    """
    Load the foundation model and forward the input image to get 
    the feature maps.
    """
    def __init__(self, type):
        super().__init__()
        self.model, feature_dim = create_foundation_model(type)
        self.type = type
        self.feature_dim = feature_dim

    def forward_mae(self, x):
        b, c, h, w = x.shape
        return self.model.forward_features(x)[:, 1:].reshape(b, h//16, w//16, -1).permute(0, 3, 1, 2)
    
    def forward_dinov2(self, x):
        b, c, h, w = x.shape
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.model.forward_features(x)[:, 1:].reshape(b, h//16, w//16, -1).permute(0, 3, 1, 2)
        
    def forward(self, x):
        with torch.no_grad():
            if self.type == 'mae':
                return self.forward_mae(x)
            elif self.type == 'dinov2':
                return self.forward_dinov2(x)