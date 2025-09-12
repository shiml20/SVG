from copyreg import dispatch_table
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
import torch.nn.functional as F
import torch.distributed as dist
from torch import vmap
import xformers.ops


def find_value_indices(tensor, max_val):
    result = []
    
    # 遍历所有可能的值（0到N）
    for val in range(max_val):
        # 创建布尔掩码标记目标值位置[6](@ref)
        mask = (tensor == val)
        # 获取所有满足条件的索引[6,7](@ref)
        indices = torch.nonzero(mask, as_tuple=True)[0]
        result.append(indices)
    
    return result

# 辅助函数优化（如果仍然需要）
def find_value_indices(tensor, max_val):
    # 使用向量化操作代替循环
    mask = tensor.unsqueeze(-1) == torch.arange(max_val, device=tensor.device)
    return [torch.where(mask[:, i])[0] for i in range(max_val)]




class SparseMoEBlock_vallina(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, experts, hidden_dim, num_experts, n_shared_experts=0, capacity=2, mlp_ratio=4.0):
        super().__init__()
        self.gate_weight = nn.Parameter(torch.empty((num_experts, hidden_dim)))
        nn.init.normal_(self.gate_weight, std=0.006)
        self.experts = nn.ModuleList(experts)
        self.capacity = capacity
        self.num_experts = num_experts

        self.n_shared_experts = n_shared_experts

        if self.n_shared_experts > 0:
            print("self.n_shared_experts", self.n_shared_experts)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            # print(mlp_hidden_dim)
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.shared_experts = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_train(x)


    def forward_train(self, x):
        B, s, D = x.shape
        identity = x

        # Flatten the input for processing
        x = x.view(-1, D)  # (S, D), where S = B * s
        S = x.shape[0]
        
        # Predict capacity
        k = int(S * self.capacity)

        # Compute gating logits and scores
        logits = F.linear(x, self.gate_weight, None)  # (S, num_experts)
        scores = logits.softmax(dim=-1).permute(1, 0)  # (num_experts, S)
        value, index = torch.topk(scores.flatten(), k=k, dim=-1, sorted=False)
        index_expert = index // S
        index_token = index % S

        index_list = find_value_indices(index_expert, self.num_experts)

        y = torch.zeros((self.num_experts, S, D), dtype=x.dtype, device=x.device)

        for idx, expert in enumerate(self.experts):
            idx_token = index_list[idx]

            idx_token = index_token[idx_token]
            x_in = x[idx_token]
            gating_value = scores[idx][idx_token]

            expert_out = expert(x_in)
            y[idx][idx_token] += gating_value.unsqueeze(-1) * expert_out

        # Sum the outputs from all experts
        y = y.view(self.num_experts, S, D).sum(dim=0, keepdim=False)  # (S, 1, D)

        # Reshape the output to match the input shape
        x_out = y.view(B, s, D)
        # import ipdb; ipdb.set_trace()

        return x_out, None, None

class SparseMoEBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, experts, hidden_dim, num_experts, n_shared_experts=0, capacity=2, mlp_ratio=4.0):
        super().__init__()
        self.gate_weight = nn.Parameter(torch.empty((num_experts, hidden_dim)))
        nn.init.normal_(self.gate_weight, std=0.006)
        self.experts = nn.ModuleList(experts)
        self.capacity = capacity
        self.num_experts = num_experts

        self.n_shared_experts = n_shared_experts

        if self.n_shared_experts > 0:
            print("self.n_shared_experts", self.n_shared_experts)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.shared_experts = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        # ema_decay = 0.95
        # expert_bias = torch.tensor([1/self.num_experts]*self.num_experts).unsqueeze(-1)
        # self.register_buffer('expert_bias', expert_bias)
        # ema_decay = torch.tensor([ema_decay])
        # self.register_buffer('ema_decay', ema_decay)


    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_train(x)

    def forward_train(self, x):
        B, s, D = x.shape
        identity = x

        # Flatten the input for processing
        x = x.view(-1, D)  # (S, D), where S = B * s
        S = x.shape[0]
        
        # Predict capacity
        k = int(S * self.capacity)

        # Compute gating logits and scores
        logits = F.linear(x, self.gate_weight, None).permute(1, 0)  # (num_experts, S)
        logits += self.expert_bias

        scores = logits.softmax(dim=0).permute(1, 0)  # (num_experts, S)
        value, index = torch.topk(scores.flatten(), k=k, dim=-1, sorted=False)
        index_expert = index // S
        index_token = index % S

        index_list = find_value_indices(index_expert, self.num_experts)
        token_each_expert = []

        print(token_each_expert)
        token_each_expert = [len(idx_expert) for idx_expert in index_list]
        # y = torch.zeros((self.num_experts, S, D), dtype=x.dtype, device=x.device)
        output = torch.zeros_like(x)


        for idx, expert in enumerate(self.experts):
            idx_token = index_list[idx]
            idx_token = index_token.index_select(0, idx_token)
            x_in = x.index_select(0, idx_token)
            gating_value = scores[idx].index_select(0, idx_token)
            expert_out = expert(x_in)
            output.index_add_(
                0, 
                idx_token, 
                expert_out * gating_value.unsqueeze(-1)
            )

        # Reshape the output to match the input shape
        x_out = output.view(B, s, D)
        # import ipdb; ipdb.set_trace()

        return x_out, None, None


class SparseMoEBlock_(nn.Module):
    """
    Optimized sparse mixture-of-experts with shared experts.
    Key improvements:
    1. Vectorized expert index processing
    2. Memory-efficient in-place operations
    3. Batched expert computation
    """
    def __init__(self, experts, hidden_dim, num_experts, n_shared_experts=0, capacity=2, mlp_ratio=4.0):
        super().__init__()
        self.gate_weight = nn.Parameter(torch.empty(num_experts, hidden_dim))
        nn.init.normal_(self.gate_weight, std=0.006)
        self.experts = nn.ModuleList(experts)
        self.capacity = capacity
        self.num_experts = num_experts
        self.n_shared_experts = n_shared_experts

        if n_shared_experts > 0:
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.shared_experts = nn.Sequential(
                nn.Linear(hidden_dim, mlp_hidden_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(mlp_hidden_dim, hidden_dim)
            )

    def forward(self, x):
        B, s, D = x.shape
        x_flat = x.view(-1, D)
        S = x_flat.size(0)
        
        # Gating logic with fused operations
        logits = F.linear(x_flat, self.gate_weight, None)  # (S, E)
        scores = logits.softmax(-1).permute(1, 0)  # (E, S)
        
        # Vectorized top-k selection
        k = int(S * self.capacity)
        topk_values, topk_indices = torch.topk(scores.flatten(), k, sorted=False)
        
        # Get expert and token indices
        expert_indices = topk_indices // S
        token_indices = topk_indices % S

        # Sort by expert for memory coalescing
        sort_order = torch.argsort(expert_indices)
        expert_indices = expert_indices[sort_order]
        token_indices = token_indices[sort_order]
        gating_values = topk_values[sort_order]

        # Vectorized expert processing
        output = torch.zeros_like(x_flat)
        expert_counts = torch.bincount(expert_indices, minlength=self.num_experts)
        
        # Batch expert computation using index_select
        unique_experts = torch.unique(expert_indices)
        for expert_id in unique_experts:
            mask = expert_indices == expert_id
            selected_tokens = token_indices[mask]
            
            expert_input = x_flat.index_select(0, selected_tokens)
            expert_output = self.experts[expert_id](expert_input)
            
            output.index_add_(
                0, 
                selected_tokens, 
                expert_output * gating_values[mask].unsqueeze(-1)
            )

        # Shared experts with residual connection
        if self.n_shared_experts > 0:
            output += self.shared_experts(x_flat)

        return output.view(B, s, D), None, None
