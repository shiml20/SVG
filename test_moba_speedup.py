import torch
import random
import time
from flash_attn import flash_attn_varlen_func


import torch.nn.functional as F


import sys
import os

# 获取项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
# from moba.moba_efficient import moba_attn_varlen
from models.moba_efficient_ljz_0308 import moba_attn_varlen

def generate_data(batch, seqlen, num_q_head, num_kv_head, headdim, dtype):
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.cuda.current_device()

    # gen qkv
    q = torch.randn(
        (seqlen, num_q_head, headdim), dtype=dtype, device=device, requires_grad=True
    )
    k = torch.randn(
        (seqlen, num_kv_head, headdim), dtype=dtype, device=device, requires_grad=True
    )
    v = torch.randn(
        (seqlen, num_kv_head, headdim), dtype=dtype, device=device, requires_grad=True
    )

    # gen cu seqlen
    cu_seqlen = random.sample(range(1, seqlen - 1), batch - 1) if batch > 1 else []
    cu_seqlen.sort()
    cu_seqlen = [0] + cu_seqlen + [seqlen]
    cu_seqlen = torch.tensor(cu_seqlen, device=device, dtype=torch.int32)

    # max_seqlen
    max_seqlen = torch.amax(cu_seqlen[1:] - cu_seqlen[:-1])

    return q, k, v, cu_seqlen, max_seqlen.item()



def test_attn_varlen_moba_speed_ori(batch, head, seqlen, head_dim, moba_chunk_size, moba_topk, dtype=torch.bfloat16):
    """Speed test comparing v3 vs v4 moba attention"""
    # Get data
    q, k, v, cu_seqlen, max_seqlen = generate_data(batch, seqlen, head, head, head_dim, dtype)
    print("q.shape, k.shape, v.shape, cu_seqlen.shape, max_seqlen")
    print(q.shape, k.shape, v.shape, cu_seqlen.shape, max_seqlen)
    
    vo_grad = torch.randn_like(q)
    
    # Warmup
    warmup_iters = 3
    perf_test_iters = 10

    # Warmup
    for _ in range(warmup_iters):
        o = flash_attn_varlen_func(q, k, v, cu_seqlen, cu_seqlen, max_seqlen, max_seqlen, causal=True)
        torch.autograd.backward(o, vo_grad)
    
    torch.cuda.synchronize()
    start_flash = time.perf_counter()
    for _ in range(perf_test_iters):
        o = flash_attn_varlen_func(q, k, v, cu_seqlen, cu_seqlen, max_seqlen, max_seqlen, causal=True)
        torch.autograd.backward(o, vo_grad)
        
    torch.cuda.synchronize()
    time_flash = (time.perf_counter() - start_flash) / perf_test_iters * 1000


    # Warmup
    for _ in range(warmup_iters):
        om = moba_attn_varlen(q, k, v, cu_seqlen, max_seqlen, moba_chunk_size=moba_chunk_size, moba_topk=moba_topk)
        torch.autograd.backward(om, vo_grad)

        
    torch.cuda.synchronize()
    start_moba = time.perf_counter()
    for _ in range(perf_test_iters):
        om = moba_attn_varlen(q, k, v, cu_seqlen, max_seqlen, moba_chunk_size=moba_chunk_size, moba_topk=moba_topk)
        torch.autograd.backward(om, vo_grad)
    
    torch.cuda.synchronize()
    time_moba = (time.perf_counter() - start_moba) / perf_test_iters * 1000
    
    print(f"\nbatch:{batch} head:{head} seqlen:{seqlen} chunk:{moba_chunk_size} topk:{moba_topk}")
    print(f"Flash: {time_flash:.2f}ms, MoBA: {time_moba:.2f}ms")
    print(f"Speedup:  {time_flash / time_moba:.2f}x")


def test_attn_varlen_moba_speed(batch, head, seqlen, head_dim, moba_chunk_size, moba_topk, dtype=torch.bfloat16):
    """性能对比测试：传统注意力 vs Flash Attention vs MoBA"""
    # 数据生成
    q, k, v, cu_seqlen, max_seqlen = generate_data(batch, seqlen, head, head, head_dim, dtype)
    print("输入张量形状:")
    print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")
    
    vo_grad = torch.randn_like(q)
    warmup_iters = 5
    perf_test_iters = 20

    # 传统注意力实现 ---------------------------------------------------
    def classic_attention(q, k, v):
        q = q.unsqueeze(0).permute(0, 2, 1, 3)
        k = k.unsqueeze(0).permute(0, 2, 1, 3)
        v = v.unsqueeze(0).permute(0, 2, 1, 3)
        
        softmax_scale = 1.0 / (head_dim ** 0.5)
        qk = torch.einsum("bhsd,bhtd->bhst", q, k) * softmax_scale
        attn = F.softmax(qk, dim=-1)
        # attn = torch.nn.Dropout(0.1)(attn)
        return torch.einsum("bhst,bhtd->bhsd", attn, v)

    # 预热传统注意力
    for _ in range(warmup_iters):
        o_classic = classic_attention(q, k, v)
        classic_attention_vo_grad = torch.randn_like(o_classic)
        # torch.autograd.backward(o_classic, classic_attention_vo_grad)
    
    # 计时传统注意力
    torch.cuda.synchronize()
    start_classic = time.perf_counter()
    for _ in range(perf_test_iters):
        o_classic = classic_attention(q, k, v)
        classic_attention_vo_grad = torch.randn_like(o_classic)
        # torch.autograd.backward(o_classic, classic_attention_vo_grad)
    torch.cuda.synchronize()
    time_classic = (time.perf_counter() - start_classic) / perf_test_iters * 1000

    # Flash Attention ---------------------------------------------------
    for _ in range(warmup_iters):
        o_flash = flash_attn_varlen_func(q, k, v, cu_seqlen, cu_seqlen, 
                                      max_seqlen, max_seqlen)
        # torch.autograd.backward(o_flash, vo_grad)
    
    torch.cuda.synchronize()
    start_flash = time.perf_counter()
    for _ in range(perf_test_iters):
        o_flash = flash_attn_varlen_func(q, k, v, cu_seqlen, cu_seqlen,
                                      max_seqlen, max_seqlen)
        torch.autograd.backward(o_flash, vo_grad)
    torch.cuda.synchronize()
    time_flash = (time.perf_counter() - start_flash) / perf_test_iters * 1000

    # MoBA Attention ---------------------------------------------------
    q = q.unsqueeze(0).permute(0, 2, 1, 3)
    k = k.unsqueeze(0).permute(0, 2, 1, 3)
    v = v.unsqueeze(0).permute(0, 2, 1, 3)

    for _ in range(warmup_iters):


        o_moba = moba_attn_varlen(q, k, v, cu_seqlen, max_seqlen, 
                                moba_chunk_size=moba_chunk_size, moba_topk=moba_topk)
        # torch.autograd.backward(o_moba, vo_grad)
    
    torch.cuda.synchronize()
    start_moba = time.perf_counter()
    for _ in range(perf_test_iters):
        o_moba = moba_attn_varlen(q, k, v, cu_seqlen, max_seqlen,
                                moba_chunk_size=moba_chunk_size, moba_topk=moba_topk)
        # torch.autograd.backward(o_moba, vo_grad)
    torch.cuda.synchronize()
    time_moba = (time.perf_counter() - start_moba) / perf_test_iters * 1000

    # 结果输出 ---------------------------------------------------------
    print(f"\n配置参数: batch={batch}, head={head}, seqlen={seqlen}, dim={head_dim}")
    print(f"Chunk Size: {moba_chunk_size}, TopK: {moba_topk}")
    print(f"传统注意力: {time_classic:.2f}ms")
    print(f"Flash Attention: {time_flash:.2f}ms  ({time_classic/time_flash:.1f}x加速)")
    print(f"MoBA: {time_moba:.2f}ms  ({time_classic/time_moba:.1f}x加速)")
    print(f"MoBA vs Flash: {time_flash/time_moba:.1f}x加速")

# 测试用例保持不变

if __name__ == "__main__":
    test_attn_varlen_moba_speed(batch=1, head=16, seqlen=256, head_dim=64, moba_chunk_size=16, moba_topk=4)
    print("simple speed test finished")
    test_attn_varlen_moba_speed(batch=1, head=6, seqlen=16_384, head_dim=64, moba_chunk_size=16, moba_topk=4)
    print("simple speed test finished")
    test_attn_varlen_moba_speed(batch=1, head=6, seqlen=16_384, head_dim=64, moba_chunk_size=256, moba_topk=3)
    print("simple speed test finished")
    test_attn_varlen_moba_speed(batch=1, head=6, seqlen=16_384, head_dim=64, moba_chunk_size=256, moba_topk=3)
    print("simple speed test finished")
    test_attn_varlen_moba_speed(batch=1, head=1, seqlen=32_768, head_dim=128, moba_chunk_size=512, moba_topk=3)
    print("simple speed test finished")

