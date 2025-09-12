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
import torch.distributed as dist


def get_local_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0  # 如果没有初始化分布式环境，默认返回 0
    global_rank = dist.get_rank()
    local_rank = global_rank % torch.cuda.device_count()
    return local_rank

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

class SparseMoEBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, experts, hidden_dim, mlp_ratio=4.0, MoE_config=None):
        super().__init__()


        self.num_experts = MoE_config.num_experts
        self.capacity = MoE_config.capacity
        self.n_shared_experts = MoE_config.n_shared_experts
        self.router_mode = "bias_sigmoid"

        print(MoE_config)
        if "router_mode" in MoE_config.keys():
            self.router_mode =  MoE_config.router_mode
        self.gate_weight = nn.Parameter(torch.empty((self.num_experts, hidden_dim)))
        nn.init.normal_(self.gate_weight, std=0.006)
        self.experts = nn.ModuleList(experts)

        if self.n_shared_experts > 0:
            print("self.n_shared_experts", self.n_shared_experts)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.shared_experts = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        # ema_decay = 0.95
        expert_bias = torch.tensor([1.0]*self.num_experts).unsqueeze(-1)
        self.register_buffer('expert_bias', expert_bias)
        # ema_decay = torch.tensor([ema_decay])
        # self.register_buffer('ema_decay', ema_decay)
        self.global_expert_freq = True


    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_eval(x)


    def router(self, x):
        if self.router_mode == "bias_sigmoid":
            # Compute gating logits and scores
            logits = F.linear(x, self.gate_weight, None).permute(1, 0)  # (num_experts, S)
            logits += self.expert_bias
            scores = F.sigmoid(logits)  # (num_experts, S)
        elif self.router_mode == "sigmoid":
            # Compute gating logits and scores
            logits = F.linear(x, self.gate_weight, None).permute(1, 0)  # (num_experts, S)
            scores = F.sigmoid(logits)  # (num_experts, S)
        elif self.router_mode == "identity":
            logits = F.linear(x, self.gate_weight, None).permute(1, 0)  # (num_experts, S)
            scores = logits  # (num_experts, S)
        elif self.router_mode == "bias_softmax":
            logits += self.expert_bias  # 将 expert_bias 加到 logits 上
            scores = logits.softmax(dim=0)  # 对 logits 应用 softmax
        else:
            logits = F.linear(x, self.gate_weight, None).permute(1, 0)  # (num_experts, S)
            scores = logits.softmax(dim=0)  # 对 logits 应用 softmax

        return scores

    def update_bias(self, token_each_expert):

        if self.global_expert_freq and self.training:
            token_each_expert_global = [torch.empty_like(token_each_expert) for _ in range(dist.get_world_size())]
            dist.all_gather(token_each_expert_global, token_each_expert)
            token_each_expert = torch.stack(token_each_expert_global).mean(dim=0)

        if self.router_mode == "bias_sigmoid_gamma095":
            # print(token_each_expert)
            mask = token_each_expert > 0.1
            self.expert_bias[mask] *= 0.3
            mask = token_each_expert < 0.04
            self.expert_bias[mask] /= 0.95
            # print("self.expert_bias.max()", self.expert_bias.max())
            # print("self.expert_biasto.min()", self.expert_bias.min())

        else:
            self.expert_bias = F.softmax(token_each_expert.unsqueeze(-1) * (1 - self.expert_bias), dim=0)


    def forward_train(self, x):
        B, s, D = x.shape
        identity = x

        # Flatten the input for processing
        x = x.view(-1, D)  # (S, D), where S = B * s
        S = x.shape[0]
        # Predict capacity
        k = int(S * self.capacity)
        scores = self.router(x)


        value, index = torch.topk(scores.flatten(), k=k, dim=-1, sorted=False)
        print(index)
        import ipdb; ipdb.set_trace()

        index_expert = index // S
        index_token = index % S

        index_list = find_value_indices(index_expert, self.num_experts)
        token_each_expert = torch.tensor([len(idx_expert)/int(k) for idx_expert in index_list]).to(x.device)
        # print(token_each_expert.shape)
        # print(scores.shape)
        num_token_each_expert = [len(idx_expert) for idx_expert in index_list]


        output = torch.zeros_like(x)
        for expert_idx, expert in enumerate(self.experts):
            expert_indices = index_list[expert_idx]
            idx_token = index_token.index_select(0, expert_indices)
            x_in = x.index_select(0, idx_token)
            gating_value = scores[expert_idx].index_select(0, idx_token)
            expert_out = expert(x_in)
            output.index_add_(
                0, 
                idx_token, 
                expert_out * gating_value.unsqueeze(-1)
            )

        # self.update_bias(token_each_expert)

        # Reshape the output to match the input shape
        x_out = output.view(B, s, D)


        return x_out, token_each_expert, torch.ones_like(scores.mean(1))


