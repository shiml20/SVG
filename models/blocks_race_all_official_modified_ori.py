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



class SparseMoEBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, experts, hidden_dim, mlp_ratio=4.0, MoE_config=None):
        super().__init__()
        print(MoE_config)
        self.num_experts = MoE_config.num_experts
        self.capacity = MoE_config.capacity
        self.n_shared_experts = MoE_config.n_shared_experts

        self.router_mode = MoE_config.get("router_mode", "bias_sigmoid")
        self.global_rsl = MoE_config.get("global_rsl", True)  # For router similarity loss
        self.enable_plr = MoE_config.get("enable_plr", True)

        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.gating_head = nn.Linear(hidden_dim, self.num_experts)
        # 初始化所有线性层
        nn.init.xavier_uniform_(self.router[0].weight)
        nn.init.xavier_uniform_(self.gating_head.weight)

        if self.enable_plr:
            print("Enable Per-Layer Regularization")
            self.target_head = nn.Linear(hidden_dim, 2*2*4)  # 2 is the patch size
            nn.init.xavier_uniform_(self.target_head.weight)
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


    def router_similarity_loss(self, logits, mask, is_global=False):
        """
        计算路由相似性损失。
        
        参数:
            logits: 路由器的logits, 形状为 (B*L, E)
            mask: 指示矩阵，形状为 (B*L, E)，其中 mask[i,j] = 1 表示专家j选择了第i个token
        返回:
            loss_sim: 路由相似性损失
        """
        if is_global:
            logits_list = [torch.empty_like(logits) for _ in range(dist.get_world_size())]
            dist.all_gather(logits_list, logits)
            logits = torch.cat(logits_list, dim=0)

            mask_list = [torch.empty_like(mask) for _ in range(dist.get_world_size())]
            dist.all_gather(mask_list, mask)
            mask = torch.cat(mask_list, dim=0)

        # 计算归一化概率 P
        B, L, D = logits.shape
        E = self.num_experts

        P = logits.reshape(B * L, E)


        P = torch.softmax(P, dim=-1)  # 沿专家维度进行softmax
        
        # 计算 P' = P^T P
        P_prime = torch.matmul(P.transpose(-1, -2), P)
        
        # 计算 M 矩阵
        M = (mask * 1.0).reshape(B * L, E)
        
        # 计算 M' = M^T M
        M_prime = torch.matmul(M.transpose(-1, -2), M)
        
        # 计算权重函数 W(i,j)
        W = torch.zeros_like(M_prime)
        # import ipdb; ipdb.set_trace()
        # 对角线部分
        diag_M_prime = torch.diagonal(M_prime)
        sum_diag = torch.sum(diag_M_prime)
        if sum_diag != 0:
            W_diag = (diag_M_prime / sum_diag) * E
            W = W + torch.diag_embed(W_diag)
            
        # 非对角线部分
        off_diag_M_prime = M_prime - torch.diag_embed(torch.diagonal(M_prime))
        sum_off_diag = torch.sum(off_diag_M_prime)
        if sum_off_diag != 0:
            W_off_diag = (off_diag_M_prime / sum_off_diag) * (E**2 - E)
            W = W + W_off_diag
        
        # 计算损失
        weighted_P_prime = W * P_prime
        loss_sim = torch.sum(weighted_P_prime) / (B * L)
        if is_global:
            loss_sim = loss_sim / dist.get_world_size()
        
        return loss_sim


    def forward(self, x):
        B, L, D = x.shape

        k = int(self.capacity)
        # B,L,D -> B,L,E
        router_value = self.router(x)
        logits = self.gating_head(router_value)
        plr_value = self.target_head(router_value) if self.enable_plr else None  # .reshape(B, 32, 32, 4)

        score = logits.flatten()
        gates = nn.Identity()(logits)
        expert_k = B * L * k * dist.get_world_size()
    
        if self.training:
            score_list = [torch.empty_like(score) for _ in range(dist.get_world_size())]
            dist.all_gather(score_list, score)
            gathered_score = torch.cat(score_list, dim=0)

            kth_val, indices = torch.kthvalue(-gathered_score, k=expert_k)
            mask = logits >= -kth_val
            self.expert_tau = self.ema_decay * self.expert_tau + (1 - self.ema_decay) * -kth_val
        else:
            mask = logits >= self.expert_tau

        router_similarity_loss = self.router_similarity_loss(logits, mask, is_global=self.global_rsl)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        gates = gates * mask
        output = torch.sum(gates.unsqueeze(-2) * expert_outputs, dim=-1)


        return output, router_similarity_loss, plr_value
