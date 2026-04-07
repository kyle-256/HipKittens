"""Test MLA forward kernel (D_QK=192, D_V=128) correctness and TFLOPS."""
import torch, math
import tk_kernel_mla_fwd

try:
    import aiter
    use_aiter = True
except ImportError:
    use_aiter = False

torch.manual_seed(42)

B, H, H_KV, N = 8, 16, 16, 2048
D_QK, D_V = 192, 128
causal = True
dtype = torch.bfloat16

num_warmup, num_iters = 200, 200

def attn_flops_fwd(b, s, h, d_qk, d_v, c):
    qk_flops = 2 * b * h * s * s * d_qk // (2 if c else 1)
    av_flops = 2 * b * h * s * s * d_v // (2 if c else 1)
    return qk_flops + av_flops

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

def ref_attention(Q, K, V, causal=True):
    """Reference MLA attention: Q/K are (B,S,H,D_QK), V is (B,S,H,D_V)."""
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = torch.einsum("bshd,bthd->bhst", Q.float(), K.float()) * scale
    if causal:
        s = scores.shape[-1]
        mask = torch.triu(torch.ones(s, s, device=scores.device, dtype=torch.bool), 1)
        scores.masked_fill_(mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhst,bthd->bshd", attn, V.float())
    return out.to(dtype)

Q = torch.randn(B, N, H, D_QK, dtype=dtype, device='cuda')
K = torch.randn(B, N, H_KV, D_QK, dtype=dtype, device='cuda')
V = torch.randn(B, N, H_KV, D_V, dtype=dtype, device='cuda')

flops_f = attn_flops_fwd(B, N, H, D_QK, D_V, causal)

# ── HipKittens MLA Forward ──────────────────────────
O_hk = torch.empty(B, N, H, D_V, dtype=dtype, device='cuda')
L_hk = torch.empty(B, H, 1, N, dtype=torch.float32, device='cuda')

def hk_mla_fwd():
    tk_kernel_mla_fwd.dispatch_fwd(Q, K, V, O_hk, L_hk)

hk_mla_fwd()  # warm up + get output

# ── Reference ────────────────────────────────────────
O_ref = ref_attention(Q, K, V, causal=causal)

# ── Correctness ──────────────────────────────────────
diff = (O_hk.float() - O_ref.float()).abs()
cos = torch.nn.functional.cosine_similarity(O_hk.float().flatten(), O_ref.float().flatten(), dim=0).item()
print(f"=== MLA FWD Correctness (D_QK={D_QK}, D_V={D_V}) ===")
print(f"Max abs diff: {diff.max().item():.6e}")
print(f"Mean abs diff: {diff.mean().item():.6e}")
print(f"Cosine sim:   {cos:.10f}")
print(f"Status:       {'PASS' if cos > 0.999 else 'FAIL'}")

# ── Benchmark ────────────────────────────────────────
t_hk = bench(hk_mla_fwd)
print(f"\n=== MLA FWD Benchmark ===")
print(f"HK-MLA FWD:  {t_hk:.4f} ms  {tflops(flops_f, t_hk):.2f} TFLOPS")

if use_aiter:
    def aiter_fwd():
        aiter.flash_attn_func(Q, K, V, causal=causal, return_lse=True, deterministic=False)
    try:
        aiter_fwd()
        t_aiter = bench(aiter_fwd)
        print(f"AITER FWD:   {t_aiter:.4f} ms  {tflops(flops_f, t_aiter):.2f} TFLOPS")
        print(f"Speedup:     {t_aiter/t_hk:.3f}x")
    except Exception as e:
        print(f"AITER failed: {e}")

print(f"\nConfig: B={B} H={H} H_KV={H_KV} N={N} D_QK={D_QK} D_V={D_V} causal={causal}")
