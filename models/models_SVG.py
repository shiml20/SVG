import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
import torch.nn.functional as F

def modulate(x, shift, scale):
    """
    Modulate the input tensor with shift and scale parameters.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, hidden_size)
        shift: Shift tensor of shape (batch_size, hidden_size)
        scale: Scale tensor of shape (batch_size, hidden_size)
    
    Returns:
        Modulated tensor of same shape as input
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    """
    Multi-head self-attention mechanism with optional normalization.
    """
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
        # print(f'Using fused attention implementation: {self.fused_attn}')

        self.qkv = nn.Linear(dim, self.head_dim * self.num_heads * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for attention module.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Output tensor of same shape as input
        """
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


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# Based on: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    Generate 2D sine-cosine positional embeddings.
    
    Args:
        embed_dim: Dimension of the embeddings
        grid_size: Size of the grid (height and width)
        cls_token: Whether to include CLS token
        extra_tokens: Number of extra tokens
        
    Returns:
        pos_embed: Positional embedding matrix of shape (grid_size*grid_size, embed_dim) 
                  or (1+grid_size*grid_size, embed_dim) if cls_token is True
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
    """
    Generate 2D sine-cosine positional embeddings from grid.
    
    Args:
        embed_dim: Dimension of the embeddings (must be even)
        grid: Grid coordinates of shape (2, H, W)
        
    Returns:
        emb: Positional embeddings of shape (H*W, embed_dim)
    """
    assert embed_dim % 2 == 0

    # Use half of dimensions to encode grid_h and half for grid_w
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Generate 1D sine-cosine positional embeddings from positions.
    
    Args:
        embed_dim: Output dimension for each position (must be even)
        pos: List of positions to be encoded of shape (M,)
        
    Returns:
        emb: Positional embeddings of shape (M, embed_dim)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def interpolate_pos_embed(pos_embed, new_shape):
    """
    Interpolate positional embeddings to new shape.
    
    Args:
        pos_embed: Original positional embeddings
        new_shape: Target shape (H, W)
        
    Returns:
        Interpolated positional embeddings
    """
    # Implementation would go here
    # This is a placeholder - actual implementation depends on specific requirements
    return pos_embed

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations using sinusoidal embeddings.
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
        
        Args:
            t: a 1-D Tensor of N indices, one per batch element. These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.
            
        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        # Implementation from: https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
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
        """
        Forward pass for timestep embedding.
        
        Args:
            t: Tensor of timesteps of shape (batch_size,)
            
        Returns:
            Embedded timesteps of shape (batch_size, hidden_size)
        """
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
        
        Args:
            labels: Input labels tensor
            force_drop_ids: Optional tensor forcing specific drops
            
        Returns:
            Modified labels with some replaced by special token
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        """
        Forward pass for label embedding.
        
        Args:
            labels: Input labels tensor of shape (batch_size,)
            train: Whether in training mode
            force_drop_ids: Optional tensor forcing specific drops
            
        Returns:
            Embedded labels of shape (batch_size, hidden_size)
        """
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
    def __init__(self, hidden_size, num_heads, head_dim=None, mlp_ratio=4.0, use_swiglu=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, head_dim=head_dim, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        
        if use_swiglu == False:
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        else:
            self.mlp = MoeMLP(hidden_size=hidden_size, intermediate_size=mlp_hidden_dim)
            
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        """
        Forward pass for DiT block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            c: Conditioning tensor of shape (batch_size, hidden_size)
            
        Returns:
            Output tensor of same shape as input
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT that projects to output space.
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
        """
        Forward pass for final layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            c: Conditioning tensor of shape (batch_size, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, patch_size^2 * out_channels)
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone (DiT: Diffusion Transformer).
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=16,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        head_dim=None,
        use_swiglu=False,
        qk_norm=False
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.num_patches = 256  # Fixed number of patches for 256x256 resolution
        
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio, 
                    use_swiglu=use_swiglu, qk_norm=qk_norm) for _ in range(depth)
        ])
        self.ln1 = nn.Linear(in_channels, hidden_size)
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize weights for transformer components.
        """
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

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
        Convert patchified sequence back to image format.
        
        Args:
            x: (N, T, patch_size**2 * C) tensor of patches
            
        Returns:
            imgs: (N, C, H, W) tensor of images
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, skip=[]):
        """
        Forward pass of DiT.
        
        Args:
            x: (N, T, D) tensor of input sequences
            t: (N,) tensor of diffusion timesteps
            y: (N,) tensor of class labels
            skip: List of layer indices to skip during forward pass
            
        Returns:
            Output tensor of shape (N, T, patch_size^2 * out_channels)
        """
        N, T, D = x.shape
        x = self.ln1(x)

        # Handle positional embeddings for different sequence lengths
        if T != 16 * 16:
            pos_embed = interpolate_pos_embed(self.pos_embed, (32, 32))
            x = x + pos_embed.to(x)
        else:
            x = x + self.pos_embed

        # Embed timesteps and class labels
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        
        # Apply transformer blocks
        for idx, block in enumerate(self.blocks):
            if idx in skip:
                continue
            x = block(x, c)                      # (N, T, D)
        
        # Final projection
        x = self.final_layer(x, c)               # (N, T, patch_size ** 2 * out_channels)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT with classifier-free guidance.
        Batches the unconditional forward pass for guidance.
        
        Args:
            x: Input tensor
            t: Timestep tensor  
            y: Class label tensor
            cfg_scale: Scale factor for classifier-free guidance
            
        Returns:
            Combined output with CFG applied
        """
        # Split batch for conditional and unconditional passes
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        
        # Apply classifier-free guidance
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

