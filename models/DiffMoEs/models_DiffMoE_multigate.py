from copyreg import dispatch_table
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
import torch.nn.functional as F
import torch.distributed as dist
from torch import vmap

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

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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


class SparseMoEBlock(nn.Module):
    """
    改进的稀疏混合专家模块，包含以下增强特性：
    1. 多头门控机制
    2. 噪声增强的稀疏门控
    3. 层次化专家选择
    4. 动态容量调整
    5. 专家专业化正则支持
    """
    def __init__(self, experts, hidden_dim, num_experts, n_shared_experts=0, 
                capacity=2, mlp_ratio=4.0, num_gates=4, noise_factor=0.1):
        super().__init__()
        # 基础参数配置
        self.experts = nn.ModuleList(experts)
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.n_shared_experts = n_shared_experts
        self.capacity = capacity
        
        # 多门控系统
        self.num_gates = num_gates
        self.gates = nn.ModuleList([
            nn.Linear(hidden_dim, num_experts) for _ in range(num_gates)
        ])
        
        # 动态容量预测器
        self.capacity_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_experts, bias=True),
        )

        # 共享专家（如果启用）
        if n_shared_experts > 0:
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.shared_experts = Mlp(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )

        # EMA阈值跟踪
        ema_decay = 0.95
        expert_threshold = torch.tensor([0.0]*num_experts)
        self.register_buffer('expert_threshold', expert_threshold)
        self.register_buffer('ema_decay', torch.tensor(ema_decay))

    def update_threshold(self, capacity_pred):
        """动态更新专家激活阈值"""
        if self.training:
            S = capacity_pred.size(0)
            topk = int((S / self.num_experts) * self.capacity)
            threshold = self.expert_threshold
            ema_decay = self.ema_decay

            for i in range(self.num_experts):
                # 计算每个专家的动态分位数
                scores, _ = torch.topk(capacity_pred[:, i], k=topk, sorted=True)
                quantile = scores[-1].detach()
                # EMA更新阈值
                threshold[i] = threshold[i] * ema_decay + (1 - ema_decay) * quantile
            
            # 分布式训练同步
            if dist.is_initialized():
                dist.all_reduce(threshold, op=dist.ReduceOp.SUM)
                threshold /= dist.get_world_size()
            
            self.expert_threshold = threshold


    def forward_train(self, x):
        B, s, D = x.shape
        identity = x

        # Flatten the input for processing
        x = x.view(-1, D)  # (S, D), where S = B * s
        S = x.shape[0]

        # Predict capacity
        capacity_pred = self.capacity_predictor(x.detach())  # (S, num_experts)
        k = int((S/self.num_experts) * self.capacity)

        # # Compute gating logits and scores
        # logits = F.linear(x, self.gate_weight, None)  # (S, num_experts)
        # scores = logits.softmax(dim=-1).permute(1, 0)  # (num_experts, S)

        # 多头门控
        gate_outputs = [gate(x) for gate in self.gates]
        logits = torch.stack(gate_outputs).mean(dim=0)  # 平均多门控结果
        # 后续流程与基础版本相同...
        scores = logits.softmax(dim=-1).permute(1, 0)

        # # Get top-k gating values and indices for each expert
        gating, index = torch.topk(scores, k=k, dim=-1, sorted=False)  # gating: (num_experts, k), index: (num_experts, k)

        # Prepare a mask for selected indices
        mask = torch.zeros((self.num_experts, S), dtype=x.dtype, device=x.device)
        mask.scatter_(1, index, 1.0)  # Set selected indices to 1

        # Expand gating weights for broadcasting
        gating_expanded = gating.unsqueeze(-1)  # (num_experts, k, 1)

        # # Gather inputs for each expert
        expert_inputs = x[index]  # (num_experts, k, D)
        expert_outputs = torch.stack([expert(expert_inputs[i]) for i, expert in enumerate(self.experts)])  # (num_experts, k, D)
        # Apply gating to expert outputs
        gated_outputs = gating_expanded * expert_outputs  # (num_experts, k, D)
        # Scatter the gated outputs back to the full output tensor
        y = torch.zeros((S * self.num_experts, D), dtype=x.dtype, device=x.device)
        offset = torch.arange(0, self.num_experts).unsqueeze(1).to(device=x.device) * S  # (num_experts, 1)
        index = (index + offset.long()).view(-1)  # Flatten the index to shape [num_experts * k]

        # Ensure gated_outputs is correctly reshaped
        gated_outputs_flat = gated_outputs.view(-1, D)  # Reshape to [num_experts * k, D]

        # Use torch.scatter with correct shapes
        y = torch.scatter(
            y,  # Target tensor of shape [S * num_experts, D]
            0,  # Dimension to scatter along
            index.unsqueeze(1).expand(-1, D),  # Indices of shape [num_experts * k, D]
            gated_outputs_flat  # Source values of shape [num_experts * k, D]
        )

        # Sum the outputs from all experts
        y = y.view(self.num_experts, S, D).sum(dim=0, keepdim=False)  # (S, 1, D)

        # Update capacity prediction
        self.update_threshold(capacity_pred)

        # Reshape the output to match the input shape
        x_out = y.view(B, s, D)

        # Prepare the ones tensor
        ones = mask.permute(1, 0).view(B, s, self.num_experts)

        # Reshape capacity prediction
        capacity_pred = capacity_pred.view(B, s, self.num_experts)

        # Add shared expert outputs if applicable
        if self.n_shared_experts > 0:
            x_out = x_out + self.shared_experts(identity)

        return x_out, ones, capacity_pred




    def forward_eval(self, x):
        """推理阶段前向传播（示例实现，需根据需求完善）"""
        # 这里可以简化计算，例如使用固定阈值过滤专家
        B, s, D = x.shape
        x_flat = x.view(-1, D)
        
        # 计算门控分数
        logits = sum(gate(x_flat) for gate in self.gates) / self.num_gates
        scores = logits.softmax(dim=-1)
        
        # 基于训练阶段学习到的阈值过滤专家
        mask = scores > self.expert_threshold
        selected_experts = torch.where(mask.any(dim=1))[0]
        
        # 聚合选择结果
        y = torch.zeros_like(x_flat)
        for idx in selected_experts:
            expert_out = self.experts[idx](x_flat)
            y += scores[:, idx].unsqueeze(-1) * expert_out
        
        return y.view(B, s, D), None, None

    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_eval(x)

    def expert_diversity_loss(self, inputs):
        """专家多样性正则损失（需在训练循环中调用）"""
        # 计算专家输出的余弦相似度
        outputs = [expert(inputs).flatten(1) for expert in self.experts]
        similarity_loss = 0
        for i in range(len(self.experts)):
            for j in range(i+1, len(self.experts)):
                cos_sim = F.cosine_similarity(outputs[i], outputs[j], dim=1)
                similarity_loss += cos_sim.mean()
        return similarity_loss / (len(self.experts)*(len(self.experts)-1)/2)

# moe_output, mask, capacity = moe_layer(inputs)
# diversity_loss = moe_layer.expert_diversity_loss(inputs)
# total_loss = task_loss + 0.1 * diversity_loss


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
    def __init__(self, hidden_size, num_heads, head_dim=None, mlp_ratio=4.0, 
                 use_swiglu=False, MoE_config=None,
                 use_moe=False,
                 **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, head_dim=head_dim, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.use_moe = use_moe
        if use_moe:
            if use_swiglu==False:
                approx_gelu = lambda: nn.GELU(approximate="tanh")
                self.mlp = SparseMoEBlock(
                    experts=[Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0) for _ in range(MoE_config.num_experts)],
                    hidden_dim=hidden_size,
                    num_experts=MoE_config.num_experts,
                    capacity=MoE_config.capacity,
                    n_shared_experts=MoE_config.n_shared_experts,
                    mlp_ratio=4.0
                    )
            else:
                self.mlp = SparseMoEBlock(
                    experts=[MoeMLP(hidden_size=hidden_size, intermediate_size=mlp_hidden_dim, ) for _ in range(MoE_config.num_experts)],
                    hidden_dim=hidden_size,
                    num_experts=MoE_config.num_experts,
                    capacity=MoE_config.capacity,
                    n_shared_experts=MoE_config.n_shared_experts,
                    )
        else:
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
        if self.use_moe:
            x_mlp, ones, pred_c = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            x = x + gate_mlp.unsqueeze(1) * x_mlp
            return x, ones, pred_c
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
        learn_sigma=True,

        use_swiglu=False,
        MoE_config=None,
        init_MoeMLP=False,
        head_dim=None,
        use_capacity_schedule=None,
        CapacityPred_loss_weight = 0.01,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads


        self.MoE_config = MoE_config
        use_moe_flag = [True] * depth
        if self.MoE_config.interleave:
            use_moe_flag = [i%2==1 for i in range(depth)]
        print(use_moe_flag)
        self.CapacityPred_loss_weight = CapacityPred_loss_weight

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads // (1 if use_moe_flag[_] else 1), head_dim=head_dim, mlp_ratio=mlp_ratio, 
                     use_swiglu=use_swiglu, MoE_config=MoE_config, use_moe=use_moe_flag[_]) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.init_MoeMLP= MoE_config.init_MoeMLP
        self.initialize_weights()


        self.capacity_schedule = MoE_config.get("capacity_schedule", None)
        if self.capacity_schedule:
            self.training_iters = -1
        print(self.capacity_schedule)


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

        # new init 
        def init_MoeMLP(module, std=0.006):
            nn.init.normal_(module.gate_proj.weight, std=std)
            nn.init.normal_(module.up_proj.weight, std=std)
            nn.init.normal_(module.down_proj.weight, std=std)
        if self.init_MoeMLP:
            for block in self.blocks:
                for expert in block.mlp.experts:
                    init_MoeMLP(expert)
            print("init MoE related module with std 0.006 like DeepSeek-MoE")

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

        if self.training and self.capacity_schedule:
            num_experts = self.MoE_config.num_experts
            capacity = self.MoE_config.capacity
            capacity_schedule_stage_I_iters = self.MoE_config.capacity_schedule.capacity_schedule_stage_I_iters
            capacity_schedule_stage_II_iters = self.MoE_config.capacity_schedule.capacity_schedule_stage_II_iters

            if self.training_iters <= capacity_schedule_stage_I_iters:
                capacity = num_experts
            elif self.training_iters <= capacity_schedule_stage_II_iters:
                capacity = capacity + (num_experts - capacity) * \
                            (capacity_schedule_stage_II_iters - self.training_iters) / (capacity_schedule_stage_II_iters - capacity_schedule_stage_I_iters)
            else:
                pass
            for i, block in enumerate(self.blocks):
                block.mlp.capacity = capacity

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        ones_list = []
        pred_c_list = []
        layer_idx_list = []
        for layer_idx, block in enumerate(self.blocks):
            x, ones, pred_c = block(x, c)                      # (N, T, D)
            if ones is not None:
                ones_list.append(ones)
                pred_c_list.append(pred_c)
                layer_idx_list.append(layer_idx)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x, "Capacity_Pred", layer_idx_list, ones_list, pred_c_list, self.CapacityPred_loss_weight

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        model_out = model_out[0]
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