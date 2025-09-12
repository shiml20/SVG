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
    A mixed expert module containing shared experts.
    """
    def __init__(self, experts, hidden_dim, num_experts, n_shared_experts=0, capacity=2, mlp_ratio=4.0, k_ratio=[]):
        super().__init__()
        self.gate_weight = nn.Parameter(torch.empty((num_experts, hidden_dim)))
        nn.init.normal_(self.gate_weight, std=0.006)
        self.experts = nn.ModuleList(experts)
        self.capacity = capacity
        self.num_experts = num_experts
        self.k_ratio = k_ratio

        self.n_shared_experts = n_shared_experts

        self.capacity_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_experts, bias=True),
        )

        if self.n_shared_experts > 0:
            print("self.n_shared_experts", self.n_shared_experts)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            # print(mlp_hidden_dim)
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.shared_experts = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        ema_decay = 0.95
        expert_threshold = torch.tensor([0.0]*num_experts)
        self.register_buffer('expert_threshold', expert_threshold)
        ema_decay = torch.tensor([ema_decay])
        self.register_buffer('ema_decay', ema_decay)



    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_eval(x)


    def update_threshold(self, capacity_pred):
        if self.training:
            capacity_pred = F.sigmoid(capacity_pred)
            S = capacity_pred.size(0)
            topk = int((S / self.num_experts) * self.capacity)
            threshold = self.expert_threshold
            ema_decay = self.ema_decay

            for i in range(self.num_experts):
                capacity_pred_scores, _ = torch.topk(capacity_pred[:, i], k=topk, dim=-1, sorted=True)
                quantile = capacity_pred_scores[-1].detach()
                threshold[i] = threshold[i] * ema_decay + (1-ema_decay) * quantile
        
            dist.all_reduce(threshold, op=dist.ReduceOp.SUM)
            threshold /= dist.get_world_size()
            self.expert_threshold = threshold

    def forward_train(self, x):
        B, s, D = x.shape
        identity = x
        x = x.view(-1, D)
        S = x.shape[0]

        capacity_pred = self.capacity_predictor(x.detach())
        k = int((S/self.num_experts) * self.capacity)
        logits = F.linear(x, self.gate_weight, None)    # bs*seq_len, num_experts
        scores = logits.softmax(dim=-1).permute(-1, -2)     #  num_experts, bs*seq_len

        y = torch.zeros_like(x, dtype=x.dtype)
        ones = torch.zeros((self.num_experts, S), dtype=x.dtype, device=x.device)

        for i, expert in enumerate(self.experts): 
            gating, index = torch.topk(scores[i], k=self.k_ratio[i], dim=-1, sorted=False)   # num_experts, k
            # y[index[i]] += gating[index[i]] * expert(x[index[i]])
            y[index, :] += gating.unsqueeze(-1) * expert(x[index, :])
            ones[i][index] = 1.


        self.update_threshold(capacity_pred)
        x_out = y.view(B, s, D)
        ones = ones.permute(1,0).view(B, s, self.num_experts)
        capacity_pred = capacity_pred.view(B, s, self.num_experts)


        if self.n_shared_experts >0:
            x_out = x_out + self.shared_experts(identity)


        return x_out, ones, capacity_pred




    def forward_eval(self, x):
        B, s, D = x.shape
        identity = x

        x = x.view(-1, D)
        S = x.shape[0]

        capacity_pred = self.capacity_predictor(x.detach())
        capacity_pred = F.sigmoid(capacity_pred)
        threshold = self.expert_threshold


        logits = F.linear(x, self.gate_weight, None)    # bs*seq_len, num_experts
        scores = logits.softmax(dim=-1).permute(-1, -2)     #  num_experts, bs*seq_len

        y = torch.zeros_like(x, dtype=x.dtype)
        processed_tokens = 0

        for i, expert in enumerate(self.experts): 
            k_fixed = torch.where(capacity_pred[:,i]>threshold[i], 1, 0).sum()
            processed_tokens += k_fixed
            gating, index = torch.topk(scores[i], k=k_fixed, dim=-1, sorted=False) 
            y[index, :] += gating.unsqueeze(-1) * expert(x[index, :])

        real_capacity = processed_tokens / S / self.num_experts
        x_out = y.view(B, s, D)

        if self.n_shared_experts > 0:
            x_out = x_out + self.shared_experts(identity)

        return x_out, None, None

