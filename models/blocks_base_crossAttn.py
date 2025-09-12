from copyreg import dispatch_table
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
import torch.nn.functional as F
import torch.distributed as dist
from torch import vmap



class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x, c):
        x = self.linear(self.norm_final(x))
        return x


class CrossAttention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            head_dim = None,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        if head_dim is None:
            assert dim % num_heads == 0, 'dim should be divisible by num_heads'
            self.head_dim = dim // num_heads
        else:
            self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.q = nn.Linear(dim, self.head_dim * self.num_heads, bias=qkv_bias)
        self.kv = nn.Linear(dim, self.head_dim * self.num_heads * 2, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)



    def forward(self, x: torch.Tensor, cond, mask) -> torch.Tensor:
        # c = c.unsqueeze(1).repeat(1, x.shape[1], 1)
        B, N, _ = x.shape
        B, NC, _ = cond.shape
        # import ipdb; ipdb.set_trace()

        # q = self.q(x).reshape(B, N, 1, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # kv = self.kv(cond).reshape(B, NC, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # B, num_heads, N, head_dim

        q = self.q(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        # q = q.squeeze(0)
        # k, v = kv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)


        # attn_bias = None
        # if mask is not None:
        #     attn_bias = torch.zeros([B * self.num_heads, q.shape[1], k.shape[1]], dtype=q.dtype, device=q.device)
        #     attn_bias.masked_fill_(mask.squeeze(1).repeat(self.num_heads, 1, 1) == 0, float('-inf'))
        

        attn_bias = None
        if mask is not None:
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)

        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)

        # if self.fused_attn:
        #     x = F.scaled_dot_product_attention(
        #             q, k, v,
        #             attn_mask=attn_bias,  # 注意力掩码
        #             dropout_p=self.attn_drop.p if self.training else 0.,   # Dropout 概率
        #             is_causal=False,  # 是否使用因果掩码
        #         )
        # else:
        #     q = q * self.scale
        #     attn = q @ k.transpose(-2, -1)
        #     attn = attn + attn_bias
        #     attn = attn.softmax(dim=-1)
        #     attn = self.attn_drop(attn)
        #     x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x





