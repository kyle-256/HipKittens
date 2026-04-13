"""Test MLA backward kernel (D_QK=192, D_V=128): dK/dV correctness.
Currently non-causal only (causal blocked by gfx950 VGPR spill corruption)."""
import torch, math

import tk_kernel_mla_bkwd_simple

torch.manual_seed(42)

B, H, H_KV, N = 8, 16, 16, 2048
D_QK, D_V = 192, 128
causal = False
dtype = torch.bfloat16

num_warmup, num_iters = 100, 100

def attn_bwd_flops(b, s, h, d_qk, d_v, c):
    div = 2 if c else 1
    qk  = 2 * b * h * s * s * d_qk // div
    av  = 2 * b * h * s * s * d_v  // div
    return int(2.5 * (qk + av))

def tflops(flop, ms):
    return (flop / 1e12) / (ms / 1e3)

def bench(fn):
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()
    se = torch.cuda.Event(enable_timing=True)
    ee = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        se.record(); fn(); ee.record()
        torch.cuda.synchronize()
        ts.append(se.elapsed_time(ee))
    return sum(ts) / len(ts)

def ref_attention_bwd(Q, K, V, dO, causal=False):
    Q, K, V, dO = Q.float(), K.float(), V.float(), dO.float()
    Q.requires_grad_(True); K.requires_grad_(True); V.requires_grad_(True)
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = torch.einsum("bshd,bthd->bhst", Q, K) * scale
    if causal:
        s = scores.shape[-1]
        mask = torch.triu(torch.ones(s, s, device=scores.device, dtype=torch.bool), 1)
        scores.masked_fill_(mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    O = torch.einsum("bhst,bthd->bshd", attn, V)
    O.backward(dO)
    return O.bfloat16(), Q.grad.bfloat16(), K.grad.bfloat16(), V.grad.bfloat16()

Q  = torch.randn(B, N, H, D_QK, dtype=dtype, device='cuda') * 0.1
K  = torch.randn(B, N, H_KV, D_QK, dtype=dtype, device='cuda') * 0.1
V  = torch.randn(B, N, H_KV, D_V, dtype=dtype, device='cuda')
dO = torch.randn(B, N, H, D_V, dtype=dtype, device='cuda') * 0.1

O_ref, dQ_ref, dK_ref, dV_ref = ref_attention_bwd(Q, K, V, dO, causal)

# Compute L and delta matching the kernel's bf16 precision
scale = 1.0 / math.sqrt(D_QK)
scores = torch.einsum("bshd,bthd->bhst", Q.float(), K.float()) * scale
L = torch.logsumexp(scores, dim=-1).unsqueeze(2).contiguous()  # (B, H, 1, N)
delta = (O_ref.float() * dO.float()).sum(dim=-1)  # (B, N, H)
delta = delta.permute(0, 2, 1).unsqueeze(2).contiguous()  # (B, H, 1, N)

dK_hk = torch.empty_like(K)
dV_hk = torch.empty_like(V)
tk_kernel_mla_bkwd_simple.dispatch_bwd(
    Q.contiguous(), K.contiguous(), V.contiguous(), dO.contiguous(),
    dK_hk, dV_hk, L, delta
)

def check(name, ref, hk):
    cos  = torch.nn.functional.cosine_similarity(
        ref.float().flatten(), hk.float().flatten(), dim=0).item()
    maxd = (ref.float() - hk.float()).abs().max().item()
    nan  = hk.isnan().sum().item()
    ok   = 'PASS' if cos > 0.99 and nan == 0 else 'FAIL'
    print(f'{name:4s}: cos={cos:.8f}  maxdiff={maxd:.4e}  nan={nan}  {ok}')

print(f'=== MLA BWD Correctness (D_QK={D_QK}, D_V={D_V}, causal={causal}) ===')
check('dK', dK_ref, dK_hk)
check('dV', dV_ref, dV_hk)

flops_b = attn_bwd_flops(B, N, H, D_QK, D_V, causal)

def hk_bwd():
    tk_kernel_mla_bkwd_simple.dispatch_bwd(
        Q, K, V, dO, dK_hk, dV_hk, L, delta)

t_bwd = bench(hk_bwd)
print(f'\n=== MLA BWD Benchmark ===')
print(f'HK-MLA BWD: {t_bwd:.4f} ms  {tflops(flops_b, t_bwd):.2f} TFLOPS')
print(f'Config: B={B} H={H} H_KV={H_KV} N={N} D_QK={D_QK} D_V={D_V} causal={causal}')
