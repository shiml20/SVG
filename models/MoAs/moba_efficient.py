"""A clean version of efficient moba implementation with flash-attn"""

import torch
import math
from flash_attn import flash_attn_varlen_func
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)
from functools import lru_cache
from einops import rearrange

#处理变长序列。
@lru_cache(maxsize=16)
def calc_chunks(cu_seqlen, moba_chunk_size):
    """calc chunks that needs moba attention"""

    # batch_sizes[batch_idx] = batch size ( seqlen ) of batch idx
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]
    # batch_num_chunk[batch_idx] = how many chunk in batch idx
    batch_num_chunk = (batch_sizes + (moba_chunk_size - 1)) // moba_chunk_size
    # cu_num_chunk[batch_idx] = first chunk id of this batch
    cu_num_chunk = torch.ones(
        batch_num_chunk.numel() + 1,
        device=cu_seqlen.device,
        dtype=batch_num_chunk.dtype,
    )
    cu_num_chunk[1:] = batch_num_chunk.cumsum(dim=0)
    # total chunk ( for all batch )
    num_chunk = cu_num_chunk[-1]
    # chunk_sizes[chunk_idx] = chunk_size of chunk idx
    chunk_sizes = torch.full(
        (num_chunk + 1,), moba_chunk_size, dtype=torch.int32, device=cu_seqlen.device
    )
    chunk_sizes[0] = 0  # for calc cu chunk
    batch_last_chunk_size = batch_sizes - (batch_num_chunk - 1) * moba_chunk_size
    chunk_sizes[cu_num_chunk[1:]] = batch_last_chunk_size
    # cu_chunk[chunk_idx] = the start chunk offset of chunk idx
    cu_chunk = chunk_sizes.cumsum(dim=-1, dtype=torch.int32)
    # chunk_to_batch[chunk_idx] = batch idx of the chunk idx
    chunk_to_batch = torch.zeros(
        (num_chunk,), dtype=torch.int32, device=cu_seqlen.device
    )
    # 只有个别位置有值1，间隔为batch长度，cumsum后就是chunk到batch的映射。
    chunk_to_batch[cu_num_chunk[1:-1]] = 1
    chunk_to_batch = chunk_to_batch.cumsum(dim=0, dtype=torch.int32)

    """ filter chunks that need moba attn """

    # filter chunks ( remove last chunk of each batch )
    # filtered_chunk_indices: chunk index list that excludes the last chunk of each batch
    chunk_to_remove = cu_num_chunk[1:] - 1
    chunk_to_remain = torch.ones(
        (num_chunk,), dtype=torch.bool, device=cu_seqlen.device
    )
    chunk_to_remain[chunk_to_remove] = False
    filtered_chunk_indices = chunk_to_remain.nonzero(as_tuple=True)[0]
    num_filtered_chunk = len(filtered_chunk_indices)

    return (
        cu_chunk,
        filtered_chunk_indices,
        num_filtered_chunk,
        filtered_chunk_indices,
        chunk_to_batch,
    )

@lru_cache(maxsize=16)
def calc_chunks_DiT(cu_seqlen, moba_chunk_size):
    """calc chunks that needs moba attention"""

    # batch_sizes[batch_idx] = batch size ( seqlen ) of batch idx
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]
    # batch_num_chunk[batch_idx] = how many chunk in batch idx
    batch_num_chunk = (batch_sizes + (moba_chunk_size - 1)) // moba_chunk_size
    # cu_num_chunk[batch_idx] = first chunk id of this batch
    cu_num_chunk = torch.ones(
        batch_num_chunk.numel() + 1,
        device=cu_seqlen.device,
        dtype=batch_num_chunk.dtype,
    )
    cu_num_chunk[1:] = batch_num_chunk.cumsum(dim=0)
    # total chunk ( for all batch )
    num_chunk = cu_num_chunk[-1]
    # chunk_sizes[chunk_idx] = chunk_size of chunk idx
    chunk_sizes = torch.full(
        (num_chunk + 1,), moba_chunk_size, dtype=torch.int32, device=cu_seqlen.device
    )
    chunk_sizes[0] = 0  # for calc cu chunk
    batch_last_chunk_size = batch_sizes - (batch_num_chunk - 1) * moba_chunk_size
    chunk_sizes[cu_num_chunk[1:]] = batch_last_chunk_size
    # cu_chunk[chunk_idx] = the start chunk offset of chunk idx
    cu_chunk = chunk_sizes.cumsum(dim=-1, dtype=torch.int32)
    # chunk_to_batch[chunk_idx] = batch idx of the chunk idx
    chunk_to_batch = torch.zeros(
        (num_chunk,), dtype=torch.int32, device=cu_seqlen.device
    )
    chunk_to_batch[cu_num_chunk[1:-1]] = 1
    chunk_to_batch = chunk_to_batch.cumsum(dim=0, dtype=torch.int32)

    num_chunk = cu_seqlen[-1] // moba_chunk_size

    return (
        cu_chunk,
        num_chunk,
        chunk_to_batch,
    )

class MixedAttention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        self_attn_cu_seqlen,
        moba_q,
        moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        max_seqlen,
        moba_chunk_size,
        moba_q_sh_indices,
    ):
        ctx.max_seqlen = max_seqlen
        ctx.moba_chunk_size = moba_chunk_size
        ctx.softmax_scale = softmax_scale = q.shape[-1] ** (-0.5)

        # self attn
        _, _, _, _, self_attn_out_sh, self_attn_lse_hs, _, _ = (
            _flash_attn_varlen_forward(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=self_attn_cu_seqlen,
                cu_seqlens_k=self_attn_cu_seqlen,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=softmax_scale,
                causal=True,
                dropout_p=0.0,
            )
        )

        # moba attn。moba_cu_seqlen_q，moba_cu_seqlen_kv的形状相同，表示q和k batch 的划分。只在相同的batch内算attention。
        _, _, _, _, moba_attn_out, moba_attn_lse_hs, _, _ = _flash_attn_varlen_forward(
            q=moba_q,
            k=moba_kv[:, 0],
            v=moba_kv[:, 1],
            cu_seqlens_q=moba_cu_seqlen_q,
            cu_seqlens_k=moba_cu_seqlen_kv,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=moba_chunk_size,
            softmax_scale=softmax_scale,
            causal=False,
            dropout_p=0.0,
        )
        # import ipdb; ipdb.set_trace()

        # convert lse shape hs -> sh ( follow the legacy mix attn logic )
        self_attn_lse_sh = self_attn_lse_hs.t().contiguous()
        moba_attn_lse = moba_attn_lse_hs.t().contiguous()

        # output buffer [S, H, D], same shape as q
        output = torch.zeros(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )

        # flatten vS & H for index ops # B*S*H, D
        output_2d = output.view(-1, q.shape[2])

        # calc mixed_lse
        # minus max lse to avoid exp explosion
        max_lse_1d = self_attn_lse_sh.view(-1)
        max_lse_1d = max_lse_1d.index_reduce(
            0, moba_q_sh_indices, moba_attn_lse.view(-1), "amax"
        )
        self_attn_lse_sh = self_attn_lse_sh - max_lse_1d.view_as(self_attn_lse_sh) # 此时self_attn_lse_sh <= 0
        moba_attn_lse = (
            moba_attn_lse.view(-1)
            .sub(max_lse_1d.index_select(0, moba_q_sh_indices))
            .reshape_as(moba_attn_lse)
        )

        mixed_attn_se_sh = self_attn_lse_sh.exp()
        moba_attn_se = moba_attn_lse.exp()

        mixed_attn_se_sh.view(-1).index_add_(
            0, moba_q_sh_indices, moba_attn_se.view(-1)
        )
        mixed_attn_lse_sh = mixed_attn_se_sh.log()

        # add attn output
        factor = (self_attn_lse_sh - mixed_attn_lse_sh).exp()  # [ vS, H ]
        self_attn_out_sh = self_attn_out_sh * factor.unsqueeze(-1)
        output_2d += self_attn_out_sh.reshape_as(output_2d)

        # add moba output
        mixed_attn_lse = (
            mixed_attn_lse_sh.view(-1)
            .index_select(0, moba_q_sh_indices)
            .view_as(moba_attn_lse)
        )
        factor = (moba_attn_lse - mixed_attn_lse).exp()  # [ vS, H ]
        moba_attn_out = moba_attn_out * factor.unsqueeze(-1)
        raw_attn_out = moba_attn_out.view(-1, moba_attn_out.shape[-1])
        output_2d.index_add_(0, moba_q_sh_indices, raw_attn_out)
        output = output.to(q.dtype)
        # add back max lse
        mixed_attn_lse_sh = mixed_attn_lse_sh + max_lse_1d.view_as(mixed_attn_se_sh)
        ctx.save_for_backward(
            output,
            mixed_attn_lse_sh,
            q,
            k,
            v,
            self_attn_cu_seqlen,
            moba_q,
            moba_kv,
            moba_cu_seqlen_q,
            moba_cu_seqlen_kv,
            moba_q_sh_indices,
        )

        return output

    @staticmethod
    def backward(ctx, d_output):

        max_seqlen = ctx.max_seqlen
        moba_chunk_size = ctx.moba_chunk_size
        softmax_scale = ctx.softmax_scale

        (
            output,
            mixed_attn_vlse_sh,
            q,
            k,
            v,
            self_attn_cu_seqlen,
            moba_q,
            moba_kv,
            moba_cu_seqlen_q,
            moba_cu_seqlen_kv,
            moba_q_sh_indices,
        ) = ctx.saved_tensors

        d_output = d_output.contiguous()

        dq, dk, dv, _ = _flash_attn_varlen_backward(
            dout=d_output,
            q=q,
            k=k,
            v=v,
            out=output,
            softmax_lse=mixed_attn_vlse_sh.t().contiguous(),
            dq=None,
            dk=None,
            dv=None,
            cu_seqlens_q=self_attn_cu_seqlen,
            cu_seqlens_k=self_attn_cu_seqlen,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale,
            causal=True,
            dropout_p=0.0,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            deterministic=True,
        )

        headdim = q.shape[-1]
        d_moba_output = (
            d_output.view(-1, headdim).index_select(0, moba_q_sh_indices).unsqueeze(1)
        )
        moba_output = (
            output.view(-1, headdim).index_select(0, moba_q_sh_indices).unsqueeze(1)
        )

        mixed_attn_vlse = (
            mixed_attn_vlse_sh.view(-1).index_select(0, moba_q_sh_indices).view(1, -1)
        )

        dmq, dmk, dmv, _ = _flash_attn_varlen_backward(
            dout=d_moba_output,
            q=moba_q,
            k=moba_kv[:, 0],
            v=moba_kv[:, 1],
            out=moba_output,
            softmax_lse=mixed_attn_vlse,
            dq=None,
            dk=None,
            dv=None,
            cu_seqlens_q=moba_cu_seqlen_q,
            cu_seqlens_k=moba_cu_seqlen_kv,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=moba_chunk_size,
            softmax_scale=softmax_scale,
            causal=False,
            dropout_p=0.0,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            deterministic=True,
        )

        dmkv = torch.stack((dmk, dmv), dim=1)
        return dq, dk, dv, None, dmq, dmkv, None, None, None, None, None


def transform_tensor(A):
    Batch, Head, Sequence, Block = A.shape
    
    # 创建一个全零的张量 B
    B = torch.zeros(Batch * Block, Head, Batch * Sequence, device=A.device)
    
    # 生成 batch_indices, block_indices
    batch_indices = torch.arange(Batch).repeat_interleave(Block * Sequence)
    block_indices = torch.arange(Block).repeat(Batch * Sequence)
    
    # 计算 B 的主对角块的行索引和列索引
    row_indices = batch_indices * Block + block_indices
    col_indices = torch.arange(Sequence*Batch).repeat_interleave(Block)
    
    # 将 A 的值填充到 B 的主对角块
    for h in range(Head):
        # 将 A[:, h, :, :] 展平为一维张量
        A_flat = A[:, h, :, :].reshape(-1)
        # 将 A_flat 填充到 B 的主对角块
        B[row_indices, h, col_indices] = A_flat
    
    return B

def moba_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    moba_chunk_size: int,
    moba_topk: int,
) -> torch.Tensor:
    """An efficient version of moba implementation with triton kernels and flash-attn, the core logic:
    1. Calculate the chunks and the number of chunks, n = floor(data_size / chunk_size)
       - tokens in the tail chunk are reserved for self attn
       - tokens in other chunks will be processed in later steps
    2. K in each chunk will calculate mean value as the representative k, and Q will attend to these representative
    k to get the gate logit, which will be used to select topk chunks
    3. Select the topk chunks and get the dense q for each kv chunk pair and do the varlen attention
    4. Combine the varlen attn and self attn results via online softmax to get the final result

    Args:
        q (torch.Tensor): [seqlen, head, head_dim]
        k (torch.Tensor): [seqlen, head, head_dim]
        v (torch.Tensor): [seqlen, head, head_dim]
        cu_seqlens (torch.Tensor): the cumulative sequence length tensor, same definition in flash attn
        max_seqlen (int): the max sequence length of the batch, same definition in flash attn

    Returns:
        attn_output (torch.Tensor): [seqlen, head, head_dim]
    """
    print(k.shape)

    batch_size, num_heads, seq_len, head_dim = k.shape
    # import ipdb; ipdb.set_trace()
    
    q_ = q.permute(0, 2, 1, 3).reshape(batch_size*seq_len, num_heads, head_dim)
    k_ = k.permute(0, 2, 1, 3).reshape(batch_size*seq_len, num_heads, head_dim)
    v_ = v.permute(0, 2, 1, 3).reshape(batch_size*seq_len, num_heads, head_dim)
    
    kv = torch.stack((k_, v_), dim=1)

    """ some basic variables """
    # qkv shape = [ S, H, D ]
    seqlen, num_head, head_dim = q_.shape

    """ prepare chunk meta """
    (
        cu_chunk,
        num_chunk,
        chunk_to_batch,
    ) = calc_chunks_DiT(cu_seqlens, moba_chunk_size)
    chunk_indices = torch.arange(0, batch_size*seq_len + 1)
    
    # print("num_chunk: ", num_chunk)
    # print("topk: ", moba_topk)
    # we will adjust selective topk to moba_topk - 1, as the last chunk is always chosen
    # moba_topk = min(moba_topk - 1, num_filtered_chunk)
    need_moba_attn = moba_topk > 0

    # corner case: if no moba attn needed, just return self attn
    if not need_moba_attn:
        # 这里把qkv转化成 BS，H，D
        return flash_attn_varlen_func(
            q_, k_, v_, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True
        )

    self_attn_cu_seqlen = cu_chunk

        # 分块计算key gate
    num_blocks = math.ceil(seq_len / moba_chunk_size)
    chunks = k.unfold(2, moba_chunk_size, moba_chunk_size)
    key_gate = chunks.mean(dim=-1)
    # unfold会忽略不完整的块，如果 seq_len 不是 moba_chunk_size 的整数倍，处理最后一个不完整的块
    if seq_len % moba_chunk_size != 0:
        last_chunk = k[:, :, num_blocks * moba_chunk_size:, :]  # 取出最后一个不完整的块
        last_mean = last_chunk.mean(dim=2, keepdim=True)  # 计算均值
        key_gate = torch.cat([key_gate, last_mean], dim=2)  # 拼接结果
    
    gate = torch.einsum("bhsd,bhnd->bhsn", q.float(), key_gate.float())
    
    # Top-k选择
    topk_val, topk_idx = torch.topk(gate, moba_topk, dim=-1)
    mask = torch.zeros_like(gate, dtype=torch.bool, device=gate.device)
    mask.scatter_(-1, topk_idx, True)
    gate[~mask] = 0
    gate[mask] = 1
    
    gate_mask = gate.permute(0, 3, 1, 2).reshape(-1, num_heads * seq_len)
    
    moba_seqlen_q = gate_mask.reshape(-1, num_heads, seq_len).sum(dim=-1).flatten()
    
    q_idx = gate_mask.nonzero(as_tuple=True)[-1].reshape(batch_size, -1)
    HS = torch.arange(batch_size, device=q.device) * num_heads * seq_len
    HS = HS.unsqueeze(-1)
    q_idx = q_idx + HS
    q_idx = q_idx.reshape(-1)
    
    moba_q = rearrange(q, "b h s d -> ( b h s ) d").index_select(0, q_idx)
    moba_q = moba_q.unsqueeze(1)
    # 下面需要把 (b h s) 的索引转化成 (h b s) 的索引
    b_idx = q_idx // (num_heads * seq_len)
    h_idx = (q_idx % (num_heads * seq_len)) // seq_len
    s_idx = q_idx % seq_len
    hbs_idx = h_idx * (batch_size * seq_len) + b_idx * seq_len + s_idx
    moba_q_sh_indices = hbs_idx % seqlen * num_head + hbs_idx // seqlen
    # 此时q_[moba_q_sh_indices[k]]应该等于rearrange(q, "b h s d -> ( b h s ) d").index_select(0, q_idx[k])
    # 验证完成，应该正确。
    
    
    
    
    
    # # gate_mask = transform_tensor(gate)

    # # 至此，应该得到一个[chunknum*batchsize，head_num，batchsize*seqlen]的gatemask

    # # import ipdb; ipdb.set_trace()

    # # varlen trick: combining all q index that needs moba attn
    # # the result will be like [ C0H0 ][ C0H1 ][ C0H2 ][ ... ][ CnHm ]
    
    # # gate_mask.reshape(gate_mask.shape[0], -1).shape: [chunk_num, head_num * seqlens], seqlens: 全部batch的长度和
    # #          .nonzero(as_tuple=True)[-1].shape: 在不对chunk进行mask的情况下应该是 k*head_num * seqlens 的数组，表示hs索引
    # # print("gatemask..reshape(gate_mask.shape[0], -1).shape: ", gate_mask.reshape(gate_mask.shape[0], -1).shape)
    # moba_q_indices = gate_mask.reshape(gate_mask.shape[0], -1).nonzero(as_tuple=True)[
    #     -1
    # ]  # [ HS indices ] * N。  moba_q_indices.shape = [k * head_num * seqlens]
    # # moba_seqlen_q indicates that how many q chunks are selected for each kv chunk - head
    # # 每个取值相当于q的变长序列的长度
    # moba_seqlen_q = gate_mask.sum(dim=-1).flatten()   # .shape=[chunk_num * head_num]
    # # select all q that needs moba attn based on the moba_q_indices
    # moba_q = rearrange(q_, "s h d -> ( h s ) d").index_select(
    #     0, moba_q_indices
    # )  # [ selected_S, D ]
    # moba_q = moba_q.unsqueeze(1)                        # moba_q.shape[0] = (q，chunk) pairs 的数量，其中有重复的q，但是对应不同的chunk，平均每个q重复k次。
    # # moba_q_sh_indices represents the position in the origin q tensor of each q token inside moba_q
    # moba_q_sh_indices = moba_q_indices % seqlen * num_head + moba_q_indices // seqlen  # 从hs索引转化为sh索引。
    # --------------------------------------------------------------------------------------------------------------------------------------------------

    # import ipdb; ipdb.set_trace()
    # --------------------------------------------------------------------------------------------------------------------------------------------------

    """ prepare moba kv """
    # Since moba_q is organized as HS * N, we need to reorganize kv to adapt to q

    # 这里去掉的应该是没有任何q关注到的chunk
    # cut off zero experts
    q_zero_mask = moba_seqlen_q == 0
    valid_expert_mask = ~q_zero_mask
    zero_expert_count = q_zero_mask.sum()
    # only keep the kv that has q select > 0
    if zero_expert_count > 0:
        moba_seqlen_q = moba_seqlen_q[valid_expert_mask]
    # moba cu_seqlen for flash attn
    # 把每个chunk对应的q当做一个变长序列？moba_seqlen_q 是每个chunk对应的q的个数。
    moba_cu_seqlen_q = torch.cat(
        (
            torch.tensor([0], device=q.device, dtype=moba_seqlen_q.dtype),
            moba_seqlen_q.cumsum(dim=0),
        ),
        dim=0,
    ).to(torch.int32)
    # 对DiT来说，不需要过滤chunk，所以直接用kv
    moba_kv = rearrange(kv, "s x h d -> h s x d")
    moba_kv = moba_kv.split(moba_chunk_size, dim=1)
    moba_kv = torch.cat(moba_kv, dim=0)     # moba_kv.shape = [head_num * chunk_num, chunk_size, 2, dim]
    if zero_expert_count > 0:
        assert valid_expert_mask.sum() == moba_kv.shape[0] - zero_expert_count
        moba_kv = moba_kv[
            valid_expert_mask
        ]  # cut off zero Q expert from kv , or the grad may be nan
    moba_kv = moba_kv.flatten(start_dim=0, end_dim=1).unsqueeze(2)  # moba_kv.shape = [valid_chunk_num *chunksize, 2, 1, dim]
    moba_cu_seqlen_kv = (
        torch.arange(
            0,
            num_chunk * num_head + 1 - zero_expert_count,
            dtype=torch.int32,
            device=q.device,
        )
        * moba_chunk_size
    )
    # import ipdb; ipdb.set_trace()

    # Shape check
    assert (
        moba_cu_seqlen_kv.shape == moba_cu_seqlen_q.shape
    ), f"moba_cu_seqlen_kv.shape != moba_cu_seqlen_q.shape {moba_cu_seqlen_kv.shape} != {moba_cu_seqlen_q.shape}"


    # qkv，cu_chunk
    # moba_q，所有batch的 每个chunk对应的q 组成的sequence，由moba_cu_seqlen_q再次划分为不同的batch
    # moba_kv，所有batch的所有chunk里的kv tokens，由moba_cu_seqlen_kv划分为不同的batch，与moba_q的batch对应，在flash attention中只在同一batch内计算attention。如果实现spatial的话，
    # 应该先按照block_indices从kv中采样，然后再reshape成moba_kv。
    # moba_q_sh_indices，moba_q中的q的sh索引。
    
    # 先把spatial chunk分好，调整好顺序之后和普通的chunk划分是一样的。
    # 
    
    # Wrapping up the flash attn call and online softmax dlse inside MixedAttention class
    return MixedAttention.apply(
        q_,
        k_,
        v_,
        self_attn_cu_seqlen,
        moba_q,
        moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        max_seqlen,
        moba_chunk_size,
        moba_q_sh_indices,
    )
