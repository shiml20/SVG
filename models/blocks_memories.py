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

import torch
import torch.nn as nn

class MemoryBank(nn.Module):
    def __init__(self, dim, num_slots=64, mem_dim=512, temperature=0.1):
        super().__init__()
        self.num_slots = num_slots
        self.mem_dim = mem_dim
        self.temperature = temperature
        
        # 可学习记忆库
        self.memory = nn.Parameter(torch.randn(num_slots, mem_dim))
        # 记忆检索的键值映射
        self.key_proj = nn.Linear(dim, mem_dim, bias=False)
        self.value_proj = nn.Linear(mem_dim, dim, bias=False)
        
        # 记忆更新门控
        self.update_gate = nn.Sequential(
            nn.Linear(dim + dim, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        """
        x: [B, L, D] 输入特征
        """
        B, L, D = x.shape
        
        # 1. 记忆检索
        # 计算查询向量
        query = self.key_proj(x)  # [B, L, M]
        # 计算记忆相似度
        attn = torch.einsum('blm, nm -> bln', query, self.memory) / self.temperature
        attn = torch.softmax(attn, dim=-1)  # [B, L, N]
        # 加权聚合记忆值
        retrieved = torch.einsum('bln, nm -> blm', attn, self.memory)  # [B, L, M]
        retrieved = self.value_proj(retrieved)  # [B, L, D]
        
        # 2. 记忆融合
        # 门控融合原始输入与记忆
        gate_input = torch.cat([x, retrieved], dim=-1)
        gate = self.update_gate(gate_input)  # [B, L, 1]
        output = gate * x + (1 - gate) * retrieved
        
        # 3. 记忆更新（可选）
        # 动态更新策略（例如EMA或梯度更新）
        
        return output