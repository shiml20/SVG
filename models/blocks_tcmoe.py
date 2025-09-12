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


def get_local_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0  # 如果没有初始化分布式环境，默认返回 0
    global_rank = dist.get_rank()
    local_rank = global_rank % torch.cuda.device_count()
    return local_rank


import torch.nn.functional as F

try:
    import flash_attn
    if hasattr(flash_attn, '__version__') and int(flash_attn.__version__[0]) == 2:
        from flash_attn.flash_attn_interface import flash_attn_kvpacked_func
        from flash_attn.modules.mha import FlashSelfAttention 
    else:
        from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
        from flash_attn.modules.mha import FlashSelfAttention
except Exception as e:
    print(f'flash_attn import failed: {e}')


#################################################################################
#                                MoE Layer.                                     #
#################################################################################


class MoEGate(nn.Module):
    def __init__(self, embed_dim, num_experts=16, num_experts_per_tok=2, 
                aux_loss_alpha=1e-6):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts

        self.scoring_func = 'softmax'
        self.alpha = aux_loss_alpha
        self.seq_aux = False

        # topk selection algorithm
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape    
        # print(bsz, seq_len, h)    
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        # import pdb; pdb.set_trace()
        
        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
            else:

                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha

        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss, fi, Pi



class XMoEGate(nn.Module):
    def __init__(self, embed_dim, num_experts=16, num_experts_per_tok=1,
                 aux_loss_alpha=1e-6, use_sampling=False, apply_dim_reduction=True, dim_reduction_ratio=0.5):
        super().__init__()
        self.top_k = int(num_experts_per_tok)
        self.num_experts = num_experts
        self.alpha = aux_loss_alpha
        self.use_sampling = use_sampling

        self.embed_dim = embed_dim
        reduced_dim = int(embed_dim * dim_reduction_ratio) if apply_dim_reduction else embed_dim

        # Learnable expert embeddings
        self.expert_embeddings = nn.Parameter(torch.randn(num_experts, embed_dim))

        # Optional dimension reduction layer
        self.dim_reduce = nn.Linear(embed_dim, reduced_dim) if apply_dim_reduction else nn.Identity()

        # Learnable temperature
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)  # [B*T, H]
        x = F.normalize(hidden_states, p=2, dim=-1)  # L2 norm

        # Apply dimension reduction
        x = self.dim_reduce(x)  # [B*T, D]
        expert_embeds = self.dim_reduce(F.normalize(self.expert_embeddings, p=2, dim=-1))  # [E, D]

        # Compute routing scores via cosine similarity
        logits = torch.matmul(x, expert_embeds.T)  # [B*T, E]

        # Apply softmax with learnable temperature
        scores = F.softmax(logits / self.temperature, dim=-1)

        # Select top-k experts
        if self.use_sampling and self.training:
            gumbel_noise = -torch.empty_like(scores).exponential_().log()
            logits_gumbel = (logits + gumbel_noise) / self.temperature
            topk_idx = torch.topk(logits_gumbel, k=self.top_k, dim=-1).indices
        else:
            topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        topk_weight = torch.gather(scores, 1, topk_idx)

        # Normalize weights if k > 1
        if self.top_k > 1:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-6)

        # Auxiliary load balancing loss
        aux_loss, fi, Pi = None, None, None
        if self.training and self.alpha > 0.0:
            all_indices = topk_idx.view(-1)
            mask_ce = F.one_hot(all_indices, num_classes=self.num_experts).float()
            ce = mask_ce.mean(0)
            Pi = scores.mean(0)
            fi = ce * self.num_experts
            aux_loss = (Pi * fi).sum() * self.alpha

        return topk_idx, topk_weight, aux_loss, fi, Pi


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss, 
    which includes the gradient of the aux loss during backpropagation.
    """
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class SparseMoEBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, experts, hidden_dim, mlp_ratio=4,
                num_experts=16, num_experts_per_tok=2, pretraining_tp=2, n_shared_experts=2):
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.experts = nn.ModuleList(experts)


        self.gate = MoEGate(embed_dim=hidden_dim, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.n_shared_experts = n_shared_experts
        
        if self.n_shared_experts > 0:
            intermediate_size =  hidden_dim * self.n_shared_experts
            self.shared_experts = MoeMLP(hidden_size = hidden_dim, intermediate_size = intermediate_size, pretraining_tp=pretraining_tp)
    
    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss, fi, Pi = self.gate(hidden_states) 
        # print(topk_idx.tolist(), print(len(topk_idx.tolist()))) 
        # global selected_ids_list
        # selected_ids_list.append(topk_idx.tolist())
        # print(selected_ids_list)
        # import pdb; pdb.set_trace()

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states, dtype=hidden_states.dtype)
            for i, expert in enumerate(self.experts): 
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i]).float()
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)

            y =  y.view(*orig_shape)
            
            # y = AddAuxiliaryLoss.apply(y, aux_loss)

        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        if self.n_shared_experts > 0:
            y = y + self.shared_experts(identity)

        if self.training:
            return y, fi, Pi

        return y, None , None

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x) 
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok 
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]]) 
            
            # for fp16 and other dtype
            expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        return expert_cache
    