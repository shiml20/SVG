# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

import torch
from scipy.optimize import linear_sum_assignment
from .dtr_taskrouting_switchDiT import *


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

@torch.jit.script
def compute_gating(k: int, zeros: torch.Tensor, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor):
    gates = zeros.scatter(1, top_k_indices, top_k_gates)
    top_k_gates = top_k_gates.flatten()
    top_k_experts = top_k_indices.flatten()
    nonzeros = top_k_gates.nonzero().squeeze(-1)
    top_k_experts_nonzero = top_k_experts[nonzeros]
    _, _index_sorted_experts = top_k_experts_nonzero.sort(0)
    expert_size = (gates > 0).long().sum(0)
    index_sorted_experts = nonzeros[_index_sorted_experts]
    batch_index = index_sorted_experts.div(k, rounding_mode='trunc')
    batch_gates = top_k_gates[index_sorted_experts]
    return batch_gates, batch_index, expert_size

@torch.no_grad()
def bipartite_matching(x1, x2):
    C = torch.abs(x1[:, :, None] - x2[:, None]).sum(0)
    indices = linear_sum_assignment(C.cpu())
    x2_index = torch.as_tensor(indices[1], dtype=torch.int64)
    return x2_index

#################################################################################
#               Routing function and Learnable Diffusion Priors                 #
#################################################################################

def _change_task(m):
    if hasattr(m, "active_task"):
        m.set_active_task(active_task)

def create_mask_tasks(unit_count, task_count, beta, alpha=4.0):
    _unit_mapping = torch.zeros((task_count, unit_count))
    num_filled_channel = int(unit_count * beta)
    last_index = unit_count - num_filled_channel
    index = ((np.linspace(0, 1, task_count) ** alpha) * last_index).round().astype(np.int64)
    for task in range(task_count):
        _unit_mapping[task, index[task]:index[task] + num_filled_channel] = torch.ones((num_filled_channel))

    return _unit_mapping

class DiffusionPrior(nn.Module):
    def __init__(self,  taskcount, depth, k, num_experts):
        super(DiffusionPrior, self).__init__()
        self.depth = depth
        self.register_buffer("_unit_mapping", create_mask_tasks(depth * num_experts, taskcount, beta=k/num_experts))
        self.active_task = 0  # need for "set_active_task"

    def set_active_task(self, active_task):
        self.active_task = active_task
        self.active_mask = torch.index_select(self._unit_mapping, 0, (torch.ones(len(active_task), device=active_task.device) * self.active_task).long())

    def compute_JSD_loss(self, x1, x2, epsilon=1e-6):
        M = (x1 + x2) / 2
        jsd = ((x1 * (torch.log(x1 + epsilon) - torch.log(M + epsilon))).sum(1) + (x2 * (torch.log(x2 + epsilon) - torch.log(M + epsilon))).sum(1))/2
        return jsd.mean()

    def forward(self, probs, probs_full_quantized):
        rand_gate_idx = torch.rand(probs.size(-1)).argsort()
        rand_prior_idx = torch.rand(probs.size(-1)).argsort()
        prior_idx = bipartite_matching(probs_full_quantized[:, rand_gate_idx], self._unit_mapping[:, rand_prior_idx])
        JSD_loss = self.compute_JSD_loss(probs[:, rand_gate_idx] / self.depth, self.active_mask[:, rand_prior_idx][:, prior_idx] / self.active_mask.sum(1, keepdim=True))
        return JSD_loss


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


from .blocks_base import *
from .blocks_base_adaLN import *
from .blocks_diffmoe import *


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, head_dim=None, mlp_ratio=4.0, 
                 use_swiglu=False, MoE_config=None, use_moe=False,
                 **block_kwargs):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        granularity = MoE_config.granularity
        mlp_hidden_dim = int(hidden_size * mlp_ratio / granularity)
        capacity = MoE_config.capacity
        num_experts = MoE_config.num_experts
        n_shared_experts = MoE_config.n_shared_experts
        
        self.use_moe = use_moe
        if use_moe:
            if use_swiglu==False:
                approx_gelu = lambda: nn.GELU(approximate="tanh")
                self.mlp = SparseMoEBlock(
                    experts=[Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0) 
                            for _ in range(num_experts)],
                    hidden_dim=hidden_size,
                    num_experts=num_experts,
                    capacity=capacity,
                    n_shared_experts=n_shared_experts,
                    mlp_ratio=4.0
                    )
            else:
                self.mlp = SparseMoEBlock(
                    experts=[MoeMLP(hidden_size=hidden_size, intermediate_size=mlp_hidden_dim, ) for _ in range(MoE_config.num_experts)],
                    hidden_dim=hidden_size,
                    num_experts=num_experts,
                    capacity=capacity,
                    n_shared_experts=n_shared_experts,
                    )
        else:
            if use_swiglu==False:
                approx_gelu = lambda: nn.GELU(approximate="tanh")
                self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
            else:
                self.mlp = MoeMLP(hidden_size=hidden_size, intermediate_size=mlp_hidden_dim, )




        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )




    def forward(self, x, c, mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        if mask is not None:  ## routing applied
            # print(x)
            routed_x = x * mask.float()
            routed_x = routed_x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(routed_x), shift_msa, scale_msa))
            # routed_x = routed_x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(routed_x), shift_mlp, scale_mlp))
        else:
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            # x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        
        if self.use_moe:
            if mask is not None:
                x_mlp, ones, pred_c = self.mlp(modulate(self.norm2(routed_x), shift_mlp, scale_mlp))
                routed_x = routed_x + gate_mlp.unsqueeze(1) * x_mlp
                x = routed_x + (1 - mask) * x
            else:
                x_mlp, ones, pred_c = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
                x = x + gate_mlp.unsqueeze(1) * x_mlp
            
            return x, ones, pred_c


        else:
            if mask is not None:
                routed_x = routed_x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(routed_x), shift_mlp, scale_mlp))
                x = routed_x + (1 - mask) * x
            else:
                x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            
            return x, None, None


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
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
        learn_sigma=True,
        router=TaskRouter,
        routing_config=None,
        head_dim=None,
        use_swiglu=False,

        MoE_config=None,
        init_MoeMLP=False,
        use_capacity_schedule=None,
        CapacityPred_loss_weight = 0.01,



    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        class_dropout_prob = 0.0 if num_classes == 1 else class_dropout_prob
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)




        self.MoE_config = MoE_config
        use_moe_flag = [True] * depth
        if self.MoE_config.interleave:
            use_moe_flag = [i%2==1 for i in range(depth)]
        print(use_moe_flag)
        self.CapacityPred_loss_weight = CapacityPred_loss_weight




        self.routing_method = "DTR"

        if self.routing_method == "DTR":
            self.routers = nn.ModuleList([router(hidden_size, taskcount=1000, alpha=4.0, beta=0.8) for _ in range(depth)])
            
            
            self.blocks = nn.ModuleList([
                DiTBlock(hidden_size, num_heads // (1 if use_moe_flag[_] else 1), head_dim=head_dim, mlp_ratio=mlp_ratio, 
                        use_swiglu=use_swiglu, MoE_config=MoE_config, use_moe=use_moe_flag[_]) for _ in range(depth)
            ])



        elif self.routing_method == "DMoE":
            self.w_JSD = routing_config.w_JSD
            self.register_buffer("full_time", torch.arange(0, routing_config.taskcount))
            self.prior = DiffusionPrior(routing_config.taskcount, depth, routing_config.param.k, routing_config.param.num_experts)
            self.routers = nn.ModuleList([router(hidden_size, **routing_config.param) for _ in range(depth)])
            self.blocks = nn.ModuleList(
                [
                    DiTBlock(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio
                    )
                    for _ in range(depth)
                ]
            )

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
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
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

        # Zero-out MoEs and initialize gating functions
        if self.routing_method == "DMoE":
            for router in self.routers:
                nn.init.constant_(router.experts.w, 0)
                nn.init.constant_(router.experts.b, 0)

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
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        global active_task
        active_task = t
        self.apply(_change_task)

        if self.routing_method == "DMoE":
            full_time = self.t_embedder(self.full_time)
            t = full_time[active_task]
            probs_list, probs_full_quantized_list = [], []
        else:
            t = self.t_embedder(t)  # (N, D)

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        y = self.y_embedder(y, self.training)  # (N, D)
        c = t + y  # (N, D)
        ones_list = []
        pred_c_list = []
        layer_idx_list = []
        for i, block in enumerate(self.blocks):
            if self.routing_method == "DMoE":
                mask, probs, probs_full_quantized = self.routers[i](x, full_time)
                probs_list.append(probs)
                probs_full_quantized_list.append(probs_full_quantized)
            elif self.routing_method == "DTR":
                mask = self.routers[i].active_mask
            else:
                mask = None

            x, ones, pred_c = block(x, c, mask)                      # (N, T, D)
            if ones is not None:
                ones_list.append(ones)
                pred_c_list.append(pred_c)
                layer_idx_list.append(i)

        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        if self.training:
            if self.routing_method == "DMoE":
                JSD_loss = self.prior(torch.cat(probs_list, dim=1), torch.cat(probs_full_quantized_list, dim=1))
                sum_aux_loss = self.w_JSD * JSD_loss
            else:
                sum_aux_loss = 0

        return x, "Capacity_Pred", layer_idx_list, ones_list, pred_c_list, self.CapacityPred_loss_weight


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
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
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

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

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
    "DiT-XL/2": DiT_XL_2,
    "DiT-XL/4": DiT_XL_4,
    "DiT-XL/8": DiT_XL_8,
    "DiT-L/2": DiT_L_2,
    "DiT-L/4": DiT_L_4,
    "DiT-L/8": DiT_L_8,
    "DiT-B/2": DiT_B_2,
    "DiT-B/4": DiT_B_4,
    "DiT-B/8": DiT_B_8,
    "DiT-S/2": DiT_S_2,
    "DiT-S/4": DiT_S_4,
    "DiT-S/8": DiT_S_8,
}