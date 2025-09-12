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

        ema_decay = 0.95
        expert_tau = torch.tensor([1.0])
        self.register_buffer('expert_tau', expert_tau)
        ema_decay = torch.tensor([ema_decay])
        self.register_buffer('ema_decay', ema_decay)
        self.global_expert_freq = True


    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_eval(x)


    def router(self, x):
        # B,L,D -> B,L,E
        logits = F.linear(x, self.gate_weight, None)

        return logits


    def forward_train(self, x):
        B, L, D = x.shape

        k = int(self.capacity)
        logits = self.router(x)
        score = logits.flatten()
        gates = nn.Identity()(logits)
        expert_k = B * L * k
    
        if self.training:
            kth_val, indices = torch.kthvalue(score, k=expert_k)
            mask = logits >= kth_val
            self.expert_tau = self.ema_decay * self.expert_tau + (1 - self.ema_decay) * kth_val
        else:
            mask = logits >= self.expert_tau


        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        gates = gates * mask
        output = torch.sum(gates.unsqueeze(-2) * expert_outputs, dim=-1)


        return output, None, None
