from copyreg import dispatch_table
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
import torch.nn.functional as F
import torch.distributed as dist
from torch import vmap
# import xformers.ops


class SparseMoEBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, hidden_size, mlp_ratio=4, MoE_config=None):
        super().__init__()

        self.capacity = MoE_config.capacity
        self.num_experts = MoE_config.num_experts
        self.n_shared_experts = MoE_config.n_shared_experts
        self.granularity = MoE_config.granularity


        self.gate_weight = nn.Parameter(torch.empty(( self.num_experts, hidden_size)))
        nn.init.normal_(self.gate_weight, std=0.006)


        approx_gelu = lambda: nn.GELU(approximate="tanh")
        mlp_hidden_size = int(hidden_size * mlp_ratio // self.granularity)
        experts = [Mlp(in_features=hidden_size, hidden_features=mlp_hidden_size, act_layer=approx_gelu, drop=0) for _ in range(MoE_config.num_experts)]
        self.experts = nn.ModuleList(experts)

        self.capacity_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, self.num_experts, bias=True),
        )




        if self.n_shared_experts > 0:
            print("self.n_shared_experts", self.n_shared_experts)
            mlp_hidden_size = int(hidden_size * mlp_ratio)
            # print(mlp_hidden_size)
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.shared_experts = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_size, act_layer=approx_gelu, drop=0)

        ema_decay = 0.95
        expert_threshold = torch.tensor([0.0]*self.num_experts)
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

        # Flatten the input for processing
        x = x.view(-1, D)  # (S, D), where S = B * s
        S = x.shape[0]

        # Predict capacity
        capacity_pred = self.capacity_predictor(x.detach())  # (S, num_experts)
        k = int((S/self.num_experts) * self.capacity)

        # Compute gating logits and scores
        logits = F.linear(x, self.gate_weight, None)  # (S, num_experts)
        scores = logits.softmax(dim=-1).permute(1, 0)  # (num_experts, S)

        # Get top-k gating values and indices for each expert
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

