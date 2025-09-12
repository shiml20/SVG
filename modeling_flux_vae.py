import inspect
import json
import os
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from loguru import logger
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

torch._dynamo.config.optimize_ddp = False


@dataclass
class AutoEncoderParams:
    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: list = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    z_channels: int = 16
    scaling_factor: float = 0.3611
    shift_factor: float = 0.1159
    deterministic: bool = False
    encoder_norm: bool = False
    psz: int = None


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list,
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

        self.grad_checkpointing = False

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                block_fn = self.down[i_level].block[i_block]
                if self.grad_checkpointing:
                    h = checkpoint(block_fn, hs[-1])
                else:
                    h = block_fn(hs[-1])
                if len(self.down[i_level].attn) > 0:
                    attn_fn = self.down[i_level].attn[i_block]
                    if self.grad_checkpointing:
                        h = checkpoint(attn_fn, h)
                    else:
                        h = attn_fn(h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list,
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

        self.grad_checkpointing = False

    def forward(self, z: Tensor) -> Tensor:
        # get dtype for proper tracing
        upscale_dtype = next(self.up.parameters()).dtype

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # cast to proper dtype
        h = h.to(upscale_dtype)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                block_fn = self.up[i_level].block[i_block]
                if self.grad_checkpointing:
                    h = checkpoint(block_fn, h)
                else:
                    h = block_fn(h)
                if len(self.up[i_level].attn) > 0:
                    attn_fn = self.up[i_level].attn[i_block]
                    if self.grad_checkpointing:
                        h = checkpoint(attn_fn, h)
                    else:
                        h = attn_fn(h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


def layer_norm_2d(input: torch.Tensor, normalized_shape: torch.Size, eps: float = 1e-6) -> torch.Tensor:
    # input.shape = (bsz, c, h, w)
    _input = input.permute(0, 2, 3, 1)
    _input = F.layer_norm(_input, normalized_shape, None, None, eps)
    _input = _input.permute(0, 3, 1, 2)
    return _input


class AutoencoderKL(nn.Module):
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.config = params
        self.config = OmegaConf.create(asdict(self.config))
        self.config.latent_channels = params.z_channels
        self.config.block_out_channels = params.ch_mult

        self.params = params
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )

        self.encoder_norm = params.encoder_norm
        self.psz = params.psz

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.GroupNorm):
            if module.weight is not None:
                module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def gradient_checkpointing_enable(self):
        self.encoder.grad_checkpointing = True
        self.decoder.grad_checkpointing = True

    @property
    def dtype(self):
        return self.encoder.conv_in.weight.dtype

    @property
    def device(self):
        return self.encoder.conv_in.weight.device

    def patchify(self, img: torch.Tensor):
        """
        img: (bsz, C, H, W)
        x: (bsz, patch_size**2 * C, H / patch_size, W / patch_size)
        """
        bsz, c, h, w = img.shape
        p = self.psz
        h_, w_ = h // p, w // p

        img = img.reshape(bsz, c, h_, p, w_, p)
        img = torch.einsum("nchpwq->ncpqhw", img)
        x = img.reshape(bsz, c * p**2, h_, w_)
        return x

    def unpatchify(self, x: torch.Tensor):
        """
        x: (bsz, patch_size**2 * C, H / patch_size, W / patch_size)
        img: (bsz, C, H, W)
        """
        bsz = x.shape[0]
        p = self.psz
        c = self.config.latent_channels
        h_, w_ = x.shape[2], x.shape[3]

        x = x.reshape(bsz, c, p, p, h_, w_)
        x = torch.einsum("ncpqhw->nchpwq", x)
        img = x.reshape(bsz, c, h_ * p, w_ * p)
        return img

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        moments = self.encoder(x)

        mean, logvar = torch.chunk(moments, 2, dim=1)
        if self.psz is not None:
            mean = self.patchify(mean)

        if self.encoder_norm:
            mean = layer_norm_2d(mean, (mean.size()[1],))

        if self.psz is not None:
            mean = self.unpatchify(mean)

        moments = torch.cat([mean, logvar], dim=1).contiguous()

        posterior = DiagonalGaussianDistribution(moments, deterministic=self.params.deterministic)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(self, input, sample_posterior=True, noise_strength=0.0):
        posterior = self.encode(input).latent_dist
        z = posterior.sample() if sample_posterior else posterior.mode()
        if noise_strength > 0.0:
            p = torch.distributions.Uniform(0, noise_strength)
            z = z + p.sample((z.shape[0],)).reshape(-1, 1, 1, 1).to(z.device) * randn_tensor(
                z.shape, device=z.device, dtype=z.dtype
            )
        dec = self.decode(z).sample
        return dec, posterior

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        config_path = os.path.join(model_path, "config.json")
        ckpt_path = os.path.join(model_path, "checkpoint.pt")

        if not os.path.isdir(model_path) or not os.path.isfile(config_path) or not os.path.isfile(ckpt_path):
            raise ValueError(
                f"Invalid model path: {model_path}. The path should contain both config.json and checkpoint.pt files."
            )

        state_dict = torch.load(ckpt_path, map_location="cpu")

        with open(config_path, "r") as f:
            config: dict = json.load(f)
        config.update(kwargs)
        kwargs = config

        # Filter out kwargs that are not in AutoEncoderParams
        # This ensures we only pass parameters that the model can accept
        valid_kwargs = {}
        param_signature = inspect.signature(AutoEncoderParams.__init__).parameters
        for key, value in kwargs.items():
            if key in param_signature:
                valid_kwargs[key] = value
            else:
                logger.info(f"Ignoring parameter '{key}' as it's not defined in AutoEncoderParams")

        params = AutoEncoderParams(**valid_kwargs)
        model = cls(params)
        try:
            msg = model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded state_dict from {ckpt_path}")
            logger.info(f"Missing keys:\n{msg.missing_keys}")
            logger.info(f"Unexpected keys:\n{msg.unexpected_keys}")
        except Exception as e:
            logger.error(e)
            logger.warning(f"Failed to load state_dict from {ckpt_path}, using random initialization")
        return model
