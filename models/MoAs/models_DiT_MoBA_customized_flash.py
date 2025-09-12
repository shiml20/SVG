import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
from .moba_efficient_sml_0228 import moba_attn_varlen
import torch.nn.functional as F

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MobaAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        head_dim: int = None,
        norm_layer: nn.Module = nn.LayerNorm,
        moba_chunk_size: int = 16,  # 新增MOBA参数
        moba_topk: int = 16,         # 新增MOBA参数
        moba_spatial: bool = False,
        expand: int = 1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.moba_chunk_size = moba_chunk_size
        self.moba_topk = moba_topk
        
        if head_dim is None:
            assert dim % num_heads == 0, 'dim should be divisible by num_heads'
            self.head_dim = expand * int(dim // num_heads)
        else:
            self.head_dim = head_dim
        
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False  # 强制使用MOBA实现
        self.moba_spatial = moba_spatial
        self.qkv = nn.Linear(dim, self.head_dim * self.num_heads * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def dense_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Dense Attention"""
        softmax_scale = self.scale
        qk = torch.einsum("bhsd,bhtd->bhst", q, k) * softmax_scale
        attn = F.softmax(qk, dim=-1)
        attn = self.attn_drop(attn)
        o = torch.einsum("bhst,bhtd->bshd", attn, v)

        return o

    def moba_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_indices=None,
    ) -> torch.Tensor:
        """MOBA核心实现"""
        softmax_scale = self.scale
        o = torch.zeros_like(q)
        batch_size, num_heads, seq_len, head_dim = k.shape  # [b, h, s, d] 
        # 提取当前batch的qkv
        q_ = q
        k_ = k
        v_ = v
        # 分块计算key gate
        num_blocks = math.ceil(seq_len / self.moba_chunk_size)
        chunks = k_.unfold(2, self.moba_chunk_size, self.moba_chunk_size)
        key_gate = chunks.mean(dim=-1)
        # unfold会忽略不完整的块，如果 seq_len 不是 moba_chunk_size 的整数倍，处理最后一个不完整的块
        if seq_len % self.moba_chunk_size != 0:
            last_chunk = k_[:, :, num_blocks * self.moba_chunk_size:, :]  # 取出最后一个不完整的块
            last_mean = last_chunk.mean(dim=2, keepdim=True)  # 计算均值
            key_gate = torch.cat([key_gate, last_mean], dim=2)  # 拼接结果
        
        gate = torch.einsum("bhsd,bhnd->bhsn", q_.float(), key_gate.float())
        
        # Top-k选择
        topk_val, topk_idx = torch.topk(gate, self.moba_topk, dim=-1)
        mask = torch.zeros_like(gate, dtype=torch.bool)
        mask.scatter_(-1, topk_idx, True)
        gate[~mask] = -float('inf')
        gate[mask] = 0

        # import ipdb; ipdb.set_trace()
        # 扩展块掩码到序列级
        gate = gate.repeat_interleave(self.moba_chunk_size, dim=-1)[:, :, :seq_len]
        qk = torch.einsum("bhsd,bhtd->bhst", q_, k_) * softmax_scale
        qk = qk + gate.to(q_.dtype)
        attn = F.softmax(qk, dim=-1)
        attn = self.attn_drop(attn)
        o = torch.einsum("bhst,bhtd->bshd", attn, v_)

        return o

    def get_spatial_indices(self, block_x=4, block_y=4, seq_x=16, seq_y=16):
        if seq_x % block_x != 0 or seq_y % block_y != 0:
            return None
        idx = torch.arange(seq_x * seq_y)
        idx = idx.view(seq_x, seq_y).unfold(0, block_x, block_x).unfold(1, block_y, block_y)
        block_xn = seq_x // block_x
        block_yn = seq_y // block_y
        idx = idx.contiguous().view(block_xn * block_yn, block_x * block_y)
        return idx
    def moba_attention_spatial(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_indices : torch.Tensor=None,
    ):
        softmax_scale = self.scale
        o = torch.zeros_like(q)
        batch_size, num_heads, seq_len, head_dim = k.shape  # [b, h, s, d] 
        # 提取当前batch的qkv
        q_ = q[:,:,block_indices.flatten(), :]
        k_ = k[:,:,block_indices.flatten(), :]
        v_ = v[:,:,block_indices.flatten(), :]
        # 分块计算key gate
        num_blocks = block_indices.shape[0]
        chunks = k_[:,:,block_indices.flatten(), :]
        
        chunks = k_.unfold(2, self.moba_chunk_size, self.moba_chunk_size)
        key_gate = chunks.mean(dim=-1)
        
        gate = torch.einsum("bhsd,bhnd->bhsn", q_.float(), key_gate.float())
        
        # Top-k选择
        topk_val, topk_idx = torch.topk(gate, self.moba_topk, dim=-1)
        # import ipdb; ipdb.set_trace()
        self_idx = torch.arange(0, seq_len, device=k.device) // self.moba_chunk_size
        self_idx = self_idx.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(batch_size, num_heads, seq_len, 1)
        topk_idx = torch.cat([topk_idx, self_idx], dim=-1)
        block_indices = block_indices.to(topk_idx.device)
        block_map = block_indices[topk_idx].view(batch_size, num_heads, seq_len, -1)  # bhs kB(idx)
        
        
        qk = torch.einsum("bhsd,bhtd->bhst", q_, k_) * softmax_scale  
        # 创建广播用的索引
        batch_indices = np.arange(batch_size)[:, None, None, None]
        h_indices = np.arange(num_heads)[None, :, None, None]
        s_indices = np.arange(seq_len)[None, None, :, None]
        # 初始化一个与qk相同形状的全为-inf的数组
        qk_ = -torch.inf * torch.ones_like(qk)
        # 将原始data中的有效值复制到ik_中
        qk_[batch_indices, h_indices, s_indices, block_map] = qk[batch_indices, h_indices, s_indices, block_map]

        attn = F.softmax(qk_, dim=-1)
        import ipdb; ipdb.set_trace()
        attn = self.attn_drop(attn)
        o = torch.einsum("bhst,bhtd->bshd", attn, v_)

        return o

    def moba_attention_flash(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_indices : torch.Tensor=None,
    ):
        batch_size, num_heads, seq_len, head_dim = k.shape
        cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=q.device)
        max_seqlen = seq_len
        return moba_attn_varlen(q.bfloat16(), k.bfloat16(), v.bfloat16(), cu_seqlens, max_seqlen, self.moba_chunk_size, self.moba_topk).view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).float()

    def moba_attention_spatial_flash(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_indices : torch.Tensor=None,
    ):
        batch_size, num_heads, seq_len, head_dim = k.shape
        # q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        # q = q.view(batch_size * seq_len, num_heads, head_dim).bfloat16()
        # k = k.view(batch_size * seq_len, num_heads, head_dim).bfloat16()
        # v = v.view(batch_size * seq_len, num_heads, head_dim).bfloat16()
        q = q[:,:,block_indices.flatten(), :]
        k = k[:,:,block_indices.flatten(), :]
        v = v[:,:,block_indices.flatten(), :]
        cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=q.device)
        max_seqlen = seq_len
        return moba_attn_varlen(q.bfloat16(), k.bfloat16(), v.bfloat16(), cu_seqlens, max_seqlen, self.moba_chunk_size, self.moba_topk).view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)[:,:,block_indices.flatten(), :].float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # B, self.num_heads, N, head_dim
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k) # bhsd
        # print('ss')

        block_xy = int(math.sqrt(self.moba_chunk_size))
        seq_xy = int(math.sqrt(k.shape[2]))

        # （暂时）只在完整分块的情况下使用空间block
        if self.moba_spatial:
            if seq_xy % block_xy != 0:
                x = self.moba_attention_flash(q, k, v)
            else:
                block_indices = self.get_spatial_indices(block_xy, block_xy, seq_xy, seq_xy)
                x = self.moba_attention_spatial_flash(q, k, v, block_indices)
        else:
            x = self.moba_attention_flash(q, k, v)

        # 恢复原始形状
        x = x.permute(0, 2, 1, 3).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):

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

        self.qkv = nn.Linear(dim, self.head_dim * self.num_heads * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x





class MoeMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, pretraining_tp=2):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.pretraining_tp = pretraining_tp

    def forward(self, x):
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0) 
            # print(self.up_proj.weight.size(), self.down_proj.weight.size())
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=-1)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, head_dim=None, mlp_ratio=4.0, use_moba=True, 
        use_swiglu=False, moba_chunk_size=16, moba_topk=4, headdim_scale=1, moba_spatial=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if use_moba==False:
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        else:
            self.attn = MobaAttention(hidden_size, num_heads=num_heads, head_dim=head_dim, qkv_bias=True, 
                moba_chunk_size=moba_chunk_size, moba_topk=moba_topk, moba_spatial=moba_spatial, expand=headdim_scale, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu==False:
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        else:
            self.mlp = MoeMLP(hidden_size=hidden_size, intermediate_size=mlp_hidden_dim, )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        moba_block_size=16,
        moba_topk=4,
        learn_sigma=True,
        head_dim=None,
        use_swiglu=False,
        use_moba_list=[],
        headdim_scale=1,
        moba_spatial=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)


        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio, 
            use_swiglu=use_swiglu, use_moba=_,
            moba_chunk_size=moba_block_size, moba_topk=moba_topk,
            headdim_scale=headdim_scale, moba_spatial=moba_spatial) 
            for _ in use_moba_list
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)


        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        # import pdb; pdb.set_trace()
        
        for idx, block in enumerate(self.blocks):
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)

        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        
        
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}