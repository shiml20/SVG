import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import instantiate_from_config
from torchvision.models.vision_transformer import VisionTransformer
import torch.nn as nn


def create_small_vit_s(output_dim=8, patch_size=16, img_size=256):
    """
    Create a lightweight ViT-S model.

    Args:
        output_dim: Output feature dimension.
        patch_size: Patch size for input images.
        img_size: Input image size.
    """
    # Compute number of patches (e.g., 256/16 = 16 → 16x16 = 256 patches)
    num_patches = (img_size // patch_size) ** 2

    # Small ViT-S configuration
    vit_config = {
        'image_size': img_size,
        'patch_size': patch_size,
        'num_layers': 6,         # fewer layers for a lightweight model
        'num_heads': 8,          # fewer attention heads
        'hidden_dim': 384,       # smaller hidden dimension
        'mlp_dim': 1536,         # typically 4x hidden_dim
        'num_classes': output_dim,
        'dropout': 0.1,
        'attention_dropout': 0.1,
    }

    model = VisionTransformer(**vit_config)

    # Replace classification head with a linear projection to output_dim
    # The output shape will be (B, 8, 256)
    model.heads = nn.Sequential(
        nn.Linear(vit_config['hidden_dim'], vit_config['hidden_dim']),
        nn.GELU(),
        nn.Linear(vit_config['hidden_dim'], output_dim)
    )

    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Small ViT-S Total parameters: {total_params:,}")
    print(f"Small ViT-S Trainable parameters: {trainable_params:,}")

    # Define custom forward to return patch-level features
    def forward_custom(x):
        # Extract features via ViT
        x = model._process_input(x)
        n = x.shape[1]

        # Add class token
        batch_size = x.shape[0]
        cls_tokens = model.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Pass through Transformer encoder
        x = model.encoder(x)

        # Remove class token, keep patch tokens only
        x = x[:, 1:, :]  # shape: (B, 256, 384)

        # Apply head projection
        x = model.heads(x)  # shape: (B, 256, 8)

        # Transpose to (B, 8, 256)
        return x.transpose(1, 2)

    model.forward = forward_custom
    return model


def match_distribution(h, h_vit, eps=1e-6):
    """
    Match h_vit distribution to h distribution.

    Args:
        h: [B, D1, N]   (DINO features)
        h_vit: [B, D2, N] (ViT features)
    """
    # Compute global mean and std for DINO features
    mean_h = h.mean(dim=(0, 2), keepdim=True)
    std_h = h.std(dim=(0, 2), keepdim=True)

    mean_h_scalar = mean_h.mean().detach()
    std_h_scalar = std_h.mean().detach()

    # Compute mean and std for ViT features
    mean_vit = h_vit.mean(dim=(0, 2), keepdim=True)
    std_vit = h_vit.std(dim=(0, 2), keepdim=True)

    mean_vit_scalar = mean_vit.mean().detach()
    std_vit_scalar = std_vit.mean().detach()

    # Normalize and re-scale
    h_vit_normed = (h_vit - mean_vit_scalar) / (std_vit_scalar + eps)
    h_vit_aligned = h_vit_normed * std_h_scalar + mean_h_scalar

    return h_vit_aligned


class DinoDecoder(pl.LightningModule):
    def __init__(
        self,
        ddconfig,
        dinoconfig,
        lossconfig,
        embed_dim,
        extra_vit_config=None,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        use_vf=None,
        proj_fix=False,
        is_train=True
    ):
        super().__init__()
        self.image_key = image_key

        # Load DINOv3 encoder
        self.encoder = torch.hub.load(
            repo_or_dir=dinoconfig['dinov3_location'],
            model=dinoconfig['model_name'],
            source="local",
            weights=dinoconfig['weights'],
        ).eval()

        # Optionally include extra lightweight ViT
        self.use_extra_vit = False
        print("self.use_extra_vit", self.use_extra_vit)
        self.use_outnorm = False

        if extra_vit_config is not None:
            self.use_extra_vit = True
            self.extra_vit = create_small_vit_s(output_dim=extra_vit_config['output_dim'])

            self.mask_ratio = extra_vit_config.get('mask_ratio', 0.0)
            self.use_outnorm = extra_vit_config.get('use_outnorm', False)

            if self.mask_ratio > 0:
                self.mask_token = nn.Parameter(torch.zeros(1, extra_vit_config['output_dim'], 1))
                nn.init.normal_(self.mask_token, std=0.02)

            self.norm_vit = nn.LayerNorm(extra_vit_config['output_dim'] + embed_dim)

        self.decoder = Decoder(**ddconfig)
        
        if is_train:
            self.loss = instantiate_from_config(lossconfig)

            self.embed_dim = embed_dim
            if colorize_nlabels is not None:
                assert isinstance(colorize_nlabels, int)
                self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
            if monitor is not None:
                self.monitor = monitor
            if ckpt_path is not None:
                self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

            self.automatic_optimization = False
            self.proj_fix = proj_fix

    def init_from_ckpt(self, path, ignore_keys=list()):
        """Load checkpoint with optional key filtering."""
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in list(sd.keys()):
            if any(k.startswith(ik) for ik in ignore_keys):
                print(f"Deleting key {k} from state_dict.")
                del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        """Encode input images into latent features."""
        h = self.encoder.forward_features(x)['x_norm_patchtokens']  # [B, D, N]
        h = h.permute(0, 2, 1)  # Adjust to [B, N, D] or [B, D, N]

        if self.use_extra_vit:
            h_vit = self.extra_vit(x)

            if self.training and self.mask_ratio > 0:
                B, D, N = h_vit.shape
                mask_flags = (torch.rand(B, device=x.device) < self.mask_ratio).float().view(B, 1, 1)
                mask_token_exp = self.mask_token.expand(B, D, N)
                h_vit = h_vit * (1 - mask_flags) + mask_token_exp * mask_flags

            if self.use_outnorm:
                h_vit = match_distribution(h, h_vit)

            # Concatenate DINO and ViT features
            h = torch.cat([h, h_vit], dim=1)

        # Reshape to [B, D_total, H_patch, W_patch]
        h = h.view(h.shape[0], -1, int(x.shape[2] // 16), int(x.shape[3] // 16)).contiguous()
        return h

    def decode(self, z):
        """Decode latent feature maps into reconstructed images."""
        return self.decoder(z)

    def forward(self, input):
        """Full forward pass: encode then decode."""
        z = self.encode(input)
        return self.decode(z)

    def get_input(self, batch, k):
        """Prepare input batch and normalize for DINO."""
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).contiguous().float()

        # Normalize to match DINO preprocessing
        x_dino = (x + 1.0) / 2.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_dino = (x_dino - mean) / std
        return x, x_dino

    def training_step(self, batch, batch_idx):
        """One training step: autoencoder and discriminator updates."""
        inputs, inputs_dino = self.get_input(batch, self.image_key)
        reconstructions = self(inputs_dino)
        ae_opt, disc_opt = self.optimizers()

        # Autoencoder update
        aeloss, log_dict_ae = self.loss(
            inputs, reconstructions, 0, self.global_step,
            last_layer=self.get_last_layer(), split="train"
        )
        self.log("aeloss", aeloss, prog_bar=True)
        self.log_dict(log_dict_ae, prog_bar=False)

        ae_opt.zero_grad()
        self.manual_backward(aeloss)
        ae_opt.step()

        # Discriminator update
        discloss, log_dict_disc = self.loss(
            inputs, reconstructions, 1, self.global_step,
            last_layer=self.get_last_layer(), split="train"
        )
        self.log("discloss", discloss, prog_bar=True)
        self.log_dict(log_dict_disc, prog_bar=False)

        disc_opt.zero_grad()
        self.manual_backward(discloss)
        disc_opt.step()

    def validation_step(self, batch, batch_idx, dataloader_idx=0, data_type=None):
        """Validation step."""
        inputs, inputs_dino = self.get_input(batch, self.image_key)
        reconstructions = self(inputs_dino)

        aeloss, log_dict_ae = self.loss(
            inputs, reconstructions, 0, self.global_step,
            last_layer=self.get_last_layer(), split="val"
        )
        discloss, log_dict_disc = self.loss(
            inputs, reconstructions, 1, self.global_step,
            last_layer=self.get_last_layer(), split="val"
        )

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        """Configure autoencoder and discriminator optimizers."""
        lr = self.learning_rate

        if self.use_extra_vit:
            params = list(self.decoder.parameters()) + list(self.extra_vit.parameters())
            if self.mask_ratio > 0:
                params += [self.mask_token]
        else:
            params = list(self.decoder.parameters())

        opt_ae = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
        )
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        """Return the final decoder layer for logging losses."""
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        """Generate reconstruction logs for visualization."""
        log = {}
        x, x_dino = self.get_input(batch, self.image_key)
        x, x_dino = x.to(self.device), x_dino.to(self.device)
        if not only_inputs:
            xrec = self(x_dino)
            if x.shape[1] > 3:
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        """Project non-RGB feature maps to RGB for visualization."""
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x
