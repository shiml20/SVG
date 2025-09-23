import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager


from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config
from torchvision.models.vision_transformer import VisionTransformer
import torch.nn as nn

from transformers import AutoModel
from typing import Any, Callable, Optional, Union


def get_image_features_custom(
    self,
    pixel_values: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    interpolate_pos_encoding: bool = False,
) -> torch.FloatTensor:
    r"""
    Returns:
        image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
        applying the projection layer to the pooled output of [`SiglipVisionModel`].

    Examples:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, AutoModel
    >>> import torch

    >>> model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
    >>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> inputs = processor(images=image, return_tensors="pt")

    >>> with torch.no_grad():
    ...     image_features = model.get_image_features(**inputs)
    ```"""
    # Use SiglipModel's config for some fields (if specified) instead of those of vision & text components.
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    vision_outputs = self.vision_model(
        pixel_values=pixel_values,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        interpolate_pos_encoding=interpolate_pos_encoding,
    )

    last_hidden_state = vision_outputs.last_hidden_state

    return last_hidden_state

class SigLipv2Decoder(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 siglipconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 use_vf=None,
                 reverse_proj=False,
                 proj_fix=False
                 ):
        super().__init__()
        self.image_key = image_key

        self.encoder = AutoModel.from_pretrained(siglipconfig['siglip_location']).eval()
        # self.encoder.get_image_features = get_image_features_custom.__get__(self.encoder)
        
   
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)

        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        self.reverse_proj = reverse_proj
        self.automatic_optimization = False
        self.proj_fix = proj_fix

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

   
    
    
    def encode(self, x):
        # h = self.encoder.get_image_features(x)
        h = get_image_features_custom(self.encoder, x)

        
        
        h = h.permute(0, 2, 1)  # [B, N, D] 或 [B, D, N] 根据后续维度调整

        # print(f'h.shape before view: {h.shape}, x.shape: {x.shape}')
        h = h.view(h.shape[0], -1, int(x.shape[2]//16), int(x.shape[3]//16)).contiguous()
        return h
    
  
    
    
    def decode(self, z):
        # print(f'z.requires_grad: {z.requires_grad}')
        # z = self.post_quant_conv(z)
        # print(f'z.requires_grad after conv: {z.requires_grad}')
        dec = self.decoder(z)
        # print(f'dec.requires_grad: {dec.requires_grad}')
        return dec

    def forward(self, input):
        z = self.encode(input)
      
        dec = self.decode(z)

       
        return dec

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]

        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        # print(f'x.min(): {x.min()}, x.max(): {x.max()}')
        # import pdb; pdb.set_trace()
        ####
        x_dino = (x + 1.0) / 2.0  # scale to [0, 1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_dino = (x_dino - mean) / std
        
        # x_dino = F.interpolate(x_dino, size=(224, 224), mode='bicubic', align_corners=False)
        
        ####
        
        return x, x_dino

    def training_step(self, batch, batch_idx):

        # print(f'self.encoder.training: {self.encoder.training}')
        # print(f'self.decoder.training: {self.decoder.training}')
        # print(f'self.post_quant_conv.training: {self.post_quant_conv.training}')
        inputs, inputs_dino = self.get_input(batch, self.image_key)
        reconstructions = self(inputs_dino)
        ae_opt, disc_opt = self.optimizers()

        # if optimizer_idx == 0:
        # train encoder+decoder+logvar
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, 0, self.global_step,
                                       last_layer=self.get_last_layer(),  split="train",  
                                       )
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        # return aeloss

        ae_opt.zero_grad()
        self.manual_backward(aeloss)
        ae_opt.step()

        # if optimizer_idx == 1:
        # train the discriminator
        # print(f'inputs.max(): {inputs.max()}, inputs.min(): {inputs.min()}')
        # import pdb; pdb.set_trace()
        discloss, log_dict_disc = self.loss(inputs, reconstructions, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train", )

        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        # return discloss

        disc_opt.zero_grad()
        self.manual_backward(discloss)
        disc_opt.step()

    def validation_step(self, batch, batch_idx, dataloader_idx=0, data_type=None):
        inputs, inputs_dino = self.get_input(batch, self.image_key)
        reconstructions = self(inputs_dino)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, 0, self.global_step,
                                      last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions,  1, self.global_step,
                                         last_layer=self.get_last_layer(),   split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        
        

        params = (list(self.decoder.parameters()) 
                    )
        
               
                
      
        opt_ae = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.9))

        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x, x_dino = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        x_dino = x_dino.to(self.device)
        if not only_inputs:
            xrec = self(x_dino)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            # log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
