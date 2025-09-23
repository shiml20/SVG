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
    创建小型ViT-S模型
    
    参数:
        output_dim: 输出特征维度
        patch_size: 图像分块大小
        img_size: 输入图像尺寸
    """
    # 计算patch数量 (256/16=16，所以16x16=256个patch)
    num_patches = (img_size // patch_size) ** 2  # 256
    
    # 配置小型ViT-S参数
    vit_config = {
        'image_size': img_size,
        'patch_size': patch_size,
        'num_layers': 6,  # 小型模型使用较少的层
        'num_heads': 8,   # 较少的注意力头
        'hidden_dim': 384, # 较小的隐藏层维度
        'mlp_dim': 1536,   # MLP维度通常是hidden_dim的4倍
        'num_classes': output_dim,  # 输出维度
        'dropout': 0.1,
        'attention_dropout': 0.1,
    }
    
    # 创建ViT模型
    model = VisionTransformer(** vit_config)
    

    # 替换分类头为适应输出维度的卷积层
    # 确保输出格式为 (b, 8, 256)
    model.heads = nn.Sequential(
        nn.Linear(vit_config['hidden_dim'], vit_config['hidden_dim']),
        nn.GELU(),
        nn.Linear(vit_config['hidden_dim'], output_dim)
    )
    
    # 计算并打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Small ViT-S Total parameters: {total_params:,}")
    print(f"Small ViT-S Trainable parameters: {trainable_params:,}")
    
    
    # 自定义前向传播以获取所需形状
    def forward_custom(x):
        # 原始ViT特征提取
        x = model._process_input(x)
        n = x.shape[1]
        
        # 添加类别嵌入
        batch_size = x.shape[0]
        cls_tokens = model.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer编码器
        x = model.encoder(x)
        
        # 仅使用patch特征（去除类别token）
        x = x[:, 1:, :]  # 形状: (b, 256, 384)
        
        # 通过自定义头获取输出
        x = model.heads(x)  # 形状: (b, 256, 8)
        
        # 转置为 (b, 8, 256)
        
        return x.transpose(1, 2)
    
    # 替换前向方法
    model.forward = forward_custom
    return model

def match_distribution(h, h_vit, eps=1e-6):
    """
    将 h_vit 的分布对齐到 h 的分布
    
    h: [B, D1, N]   (DINO features)
    h_vit: [B, D2, N] (small ViT features)
    """
    # 计算 DINO 全局均值和方差（跨 batch 和 patch，但保留通道）
    mean_h = h.mean(dim=(0, 2), keepdim=True)   # [1, D1, 1]
    std_h  = h.std(dim=(0, 2), keepdim=True)    # [1, D1, 1]
    
    # 再取全局均值/方差（因为 D1 ≠ D2，只能取标量来对齐）
    mean_h_scalar = mean_h.mean().detach()
    std_h_scalar  = std_h.mean().detach()

    # 计算 h_vit 的均值和方差
    mean_vit = h_vit.mean(dim=(0, 2), keepdim=True)  # [1, D2, 1]
    std_vit  = h_vit.std(dim=(0, 2), keepdim=True)   # [1, D2, 1]

    mean_vit_scalar = mean_vit.mean().detach()
    std_vit_scalar  = std_vit.mean().detach()

    # 标准化 + 重映射
    h_vit_normed = (h_vit - mean_vit_scalar) / (std_vit_scalar + eps)
    h_vit_aligned = h_vit_normed * std_h_scalar + mean_h_scalar

    return h_vit_aligned



class DinoDecoder(pl.LightningModule):
    def __init__(self,
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
                 reverse_proj=False,
                 proj_fix=False
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = torch.hub.load(
            repo_or_dir=dinoconfig['dinov3_location'],
            model=dinoconfig['model_name'],
            source="local",
            weights=dinoconfig['weights'],
        ).eval()
        
        self.use_extra_vit = False
        print("self.use_extra_vit", self.use_extra_vit)
        self.use_outnorm = False
        if extra_vit_config is not None:
            self.use_extra_vit = True
            self.extra_vit = create_small_vit_s(output_dim=extra_vit_config['output_dim'])
            
            self.mask_ratio = extra_vit_config.get('mask_ratio', 0.0)
            self.use_outnorm = extra_vit_config.get('use_outnorm', False)
            if self.mask_ratio > 0:
                self.mask_token = nn.Parameter(torch.zeros(1, extra_vit_config['output_dim'],1))
                nn.init.normal_(self.mask_token, std=0.02)
            
            self.norm_vit = nn.LayerNorm(extra_vit_config['output_dim'] + embed_dim)
                

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

    # def encode(self, x):
    #     # print(f'x.shape = {x.shape}')
    #     # import pdb; pdb.set_trace()
    #     # with torch.no_grad():
    #     # print(f'x.requires_grad: {x.requires_grad}')
    #     h = self.encoder.get_intermediate_layers(x, n=range(12), reshape=True, norm=True)[-1]
    #     # h = self.encoder.encode_trainable(x)
    #     # h = h.requires_grad_(True)
    #     return h
    
    def encode(self, x):
        h = self.encoder.forward_features(x)['x_norm_patchtokens']  # [B, D, N]
        h = h.permute(0, 2, 1)  # [B, N, D] 或 [B, D, N] 根据后续维度调整

        if self.use_extra_vit:
            h_vit = self.extra_vit(x)  # [B, D, N]
            # print('sss')
            # self.mask_ratio = -1
            # print(self.mask_ratio)
            if self.training:
                if self.mask_ratio > 0:
                    B, D, N = h_vit.shape

                    # 每个样本生成一个 mask 标志，True 表示该样本全部用 mask_token 替换
                    mask_flags = (torch.rand(B, device=x.device) < self.mask_ratio).float().view(B, 1, 1)  # [B,1,1]

                    # 扩展 mask_token
                    mask_token_exp = self.mask_token.expand(B, D, N)  # [B,D,N]

                    # 用 mask_flags 控制替换：如果 mask_flags[b]=1 -> 全部替换，否则保持原值
                    h_vit = h_vit * (1 - mask_flags) + mask_token_exp * mask_flags

                    # if self.use_outnorm:
                        # h_vit = self.norm_vit(h_vit.transpose(1, 2)).transpose(1, 2)

            else:
                # import ipdb; ipdb.set_trace()
                # pass
                self.mask_ratio = -1
                print(self.mask_ratio)
                if self.mask_ratio > 0:
                    B, D, N = h_vit.shape

                    # 每个样本生成一个 mask 标志，True 表示该样本全部用 mask_token 替换
                    mask_flags = (torch.rand(B, device=x.device) < self.mask_ratio).float().view(B, 1, 1)  # [B,1,1]

                    # 扩展 mask_token
                    mask_token_exp = self.mask_token.expand(B, D, N)  # [B,D,N]

                    # 用 mask_flags 控制替换：如果 mask_flags[b]=1 -> 全部替换，否则保持原值
                    h_vit = h_vit * (1 - mask_flags) + mask_token_exp * mask_flags

            if self.use_outnorm:
                h_vit = match_distribution(h, h_vit)  # 对齐到 DINO 的分布
            # 拼接原始 encoder 特征
            h = torch.cat([h, h_vit], dim=1)

                # h = self.norm_vit(h.transpose(1, 2)).transpose(1, 2)

        # reshape 到 [B, D_total, H_patch, W_patch]
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
        
        
        if self.use_extra_vit:
            params = (list(self.decoder.parameters()) 
                  +list(self.extra_vit.parameters())
                  )
            if self.mask_ratio > 0:
                params = params + [self.mask_token]
        else:
        
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