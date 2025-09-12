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




class SparseMoEBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, experts, hidden_dim, num_experts, n_shared_experts=0, capacity=2):
        super().__init__()
        self.gate_weight = nn.Parameter(torch.empty((num_experts, hidden_dim)))
        nn.init.normal_(self.gate_weight, std=0.006)
        self.experts = nn.ModuleList(experts)
        self.capacity = capacity
        self.num_experts = num_experts

        self.n_shared_experts = n_shared_experts
     
        if self.n_shared_experts > 0:
            # intermediate_size =  mlp_ratio * embed_dim * self.n_shared_experts
            intermediate_size = hidden_dim * self.n_shared_experts
            self.shared_experts = MoeMLP(hidden_size = hidden_dim, intermediate_size = intermediate_size, pretraining_tp=2)
    

    def forward(self, x):
        identity = x
        B, S, D = x.shape
        # 1. Compute token-expert affinity scores
        logits = F.linear(x, self.gate_weight, None)    # bs, seq_len, num_experts
        affinity = logits.softmax(dim=-1)
        affinity = torch.einsum('b s e->b e s', affinity)
        # 2. select the top-k tokens for each experts
        k = int( (S/self.num_experts) * self.capacity)
        # print(k, S, self.capacity, self.num_experts)
        gating, index = torch.topk(affinity, k=k, dim=-1, sorted=False)
        dispatch = F.one_hot(index, num_classes=S).to(device=x.device, dtype=x.dtype)
        # 3. Process the tokens by each expert and combine
        x_in = torch.einsum(" b e c s, b s d -> b e c d", dispatch, x)
        x_e = [self.experts[e](x_in[:,e]) for e in range(self.num_experts)]
        x_e = torch.stack(x_e, dim=1)
        x_out = torch.einsum('b e c s, b e c, b e c d -> b s d', dispatch, gating, x_e)
        if self.n_shared_experts >0:
            x_out = x_out + self.shared_experts(identity)
        return x_out
        

