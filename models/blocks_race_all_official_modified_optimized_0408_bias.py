from copyreg import dispatch_table
import os
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


def method_vectorized(mask: torch.Tensor):
    """向量化split方法"""
    all_indices = mask.nonzero()
    if all_indices.size(0) == 0:
        return [torch.empty(0, dtype=torch.long, device=mask.device) for _ in range(mask.size(0))]
    return torch.split(all_indices[:,1], mask.sum(dim=1).tolist())

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


        self.capacity_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_experts, bias=True),
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

        expert_tau = torch.tensor([1.0])
        self.register_buffer('expert_tau', expert_tau)
        bias= torch.tensor([0.0]*self.num_experts)
        self.register_buffer('bias', bias)
        self.global_expert_freq = True
        self.alpha = MoE_config.get("alpha", 1e-4)
        ema_decay = torch.tensor([0.95])
        self.register_buffer('ema_decay', ema_decay)


        bias_update_freq = torch.tensor([1.0])
        bias_update_recorder = torch.tensor([0.0])
        self.register_buffer('bias_update_freq', bias_update_freq)
        self.register_buffer('bias_update_recorder', bias_update_recorder)



    def update_router_bias_vallina(self, mask):

        if self.training:
            gathered_mask = [torch.empty_like(mask) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_mask, mask)
            gathered_mask = torch.stack(gathered_mask, dim=0)

        Q = torch.tensor([1/self.num_experts]*self.num_experts).to(mask.device)
        # mask [GPU, bs, token length, experts]
        F = gathered_mask.sum(0).permute(2, 0, 1).reshape(self.num_experts, -1).sum(-1).float() / sum(gathered_mask.flatten())
        # import ipdb; ipdb.set_trace()

        self.bias = self.bias - self.alpha * torch.sign(F - Q)


    
    def update_router_bias(self, mask):
        if self.training:
            # 优化点1：使用非阻塞通信和缓冲区复用
            # gathered_mask = torch.empty(
                # dist.get_world_size(), *mask.shape, 
                # dtype=mask.dtype, 
                # device=mask.device
            # )
            # dist.all_gather_into_tensor(gathered_mask, mask)
            # import ipdb; ipdb.set_trace()
            # 优化点3：优化reduce操作
            F = (mask.sum(dim=(0, 1, 2)) / mask.sum()).float()  # [num_experts]
            # 优化点4：原地操作避免内存分配
            self.bias.sub_(self.alpha * torch.sign(F - 1/self.num_experts))




    def forward(self, x):
        B, L, D = x.shape

        k = int(self.capacity)
        # B,L,D -> B,L,E
        # 计算路由和门控
        router_value = self.router(x)
        logits = self.gating_head(router_value)
        plr_value = self.target_head(router_value) if self.enable_plr else None
        
        # 专家选择逻辑
        gates_value = nn.Identity()(logits)
        capacity_pred = self.capacity_predictor(x.detach())  # (S, num_experts)
        gates_score = F.sigmoid(logits) + self.bias[None, None, :]

        # import ipdb; ipdb.set_trace()
        if self.training:
            expert_k = B * L * k * dist.get_world_size()
        else:
            expert_k = B * L * k


        if self.training:
        # 分布式分数收集
            score_list = [torch.empty_like(gates_score) for _ in range(dist.get_world_size())]
            dist.all_gather(score_list, gates_score)
            gathered_score = torch.cat(score_list, dim=0)
            # 动态阈值计算
            kth_val, _ = torch.kthvalue(-gathered_score.flatten(), k=expert_k)
            mask = gates_score >= -kth_val

            self.expert_tau = self.ema_decay * self.expert_tau + (1 - self.ema_decay) * -kth_val

        else:
            # kth_val, _ = torch.kthvalue(-F.sigmoid(capacity_pred).flatten(), k=expert_k)
            kth_val, _ = torch.kthvalue(-gates_score.flatten(), k=expert_k)
            # import ipdb; ipdb.set_trace()
            mask = gates_score >= -kth_val
            
            # 计算专家利用率
            expert_utilization = mask.sum(dim=(0, 1, 2)) / expert_k
            print(mask.sum(dim=(0, 1)))
            print(self.bias)
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname("/ytech_m2v2_hdd/sml/DiffMoE_research_local/expert_utilization.txt"), exist_ok=True)
            
            # 写入文件（追加模式）
            with open("expert_utilization.txt", "a+") as f:  # a+模式可读可写，文件不存在则创建
                # 如果是新文件，写入标题行
                if f.tell() == 0:
                    f.write("# Expert utilization ratios (comma-separated)\n")
                f.write(",".join([f"{expert_utilization.cpu().numpy():.6f}"]) + "\n")

        if self.training:
            if int(self.bias_update_recorder) % int(self.bias_update_freq) == 0:
                self.update_router_bias(torch.stack(score_list, dim=0) > -kth_val)
                self.bias_update_recorder = torch.tensor([0]).to(x.device)
            else:
                self.bias_update_recorder += 1

        else:
            f = (mask.sum(dim=(0, 1)))  # [num_experts]
            # print(f)
            # print(self.bias)
            # print(1/self.num_experts)

        
        # # optimized 
        # # 稀疏专家计算
        output = torch.zeros(B*L, D, device=x.device, dtype=x.dtype)
        E = self.num_experts
        x = x.view(-1 , D)
        gates_value = gates_value.permute(2, 0, 1).view(E, -1)
        mask = mask.permute(2, 0, 1).view(E, -1)
        
        # import ipdb; ipdb.set_trace()
        # mask.permute(2, 0, 1) E, B, L
        expert_assignment = method_vectorized(mask)
        # 对每个专家进行稀疏计算

        for expert_idx, expert in enumerate(self.experts):
            # 获取当前专家对应的token掩码
            index = expert_assignment[expert_idx]
            # 收集被选中的token
            selected_x = x[index]  # [num_selected, D]
            # 专家计算
            expert_out = expert(selected_x)  # [num_selected, D]
            # import ipdb; ipdb.set_trace()
            # 加权输出并散射回结果张量
            output[index] += expert_out * gates_value[expert_idx][index].unsqueeze(-1)


        return output.reshape(B, L, -1), mask.float(), capacity_pred.reshape(self.num_experts, -1), plr_value
