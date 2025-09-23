import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager


from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config


class DinoDecoder(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 dinoconfig,
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
        # self.encoder = Encoder(**ddconfig)
        self.encoder = torch.hub.load(
            repo_or_dir=dinoconfig['dinov3_location'],
            model=dinoconfig['model_name'],
            source="local",
            weights=dinoconfig['weights'],
        )
        print(f'dd_config = {ddconfig}')
        self.decoder = Decoder(**ddconfig)
        print(f'dd_config = {ddconfig}')
        print(f'self.decoder.conv_in.weight.shape = {self.decoder.conv_in.weight.shape}')
        # import pdb; pdb.set_trace()
        self.loss = instantiate_from_config(lossconfig)
        # self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

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
        with torch.no_grad():
            h = self.encoder.forward_features(x)['x_norm_patchtokens']
        h = h.permute(0, 2, 1).view(h.shape[0], -1, int(x.shape[2]//16), int(x.shape[3]//16)).contiguous()
        
        h = h.detach()
        # h.requires_grad_(True)
        
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
                #   + list(self.encoder.parameters())
                #   +list(self.post_quant_conv.parameters()) 
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
