"""
SBHD vs AITER FLOPS benchmark.

Compares:
  1. HipKittens BSHD  (existing path)
  2. HipKittens SBHD  (native, zero-copy)
  3. AITER BSHD        (existing path)
  4. AITER SBHD→BSHD   (transpose + contiguous overhead)

Reports fwd, bwd, fwd+bwd TFLOPS for each.
"""
import torch, random, math, sys

import tk_kernel_fwd
import tk_kernel_bkwd
import tk_kernel_bkwd_prep

use_aiter = True
try:
    import aiter
except ImportError:
    use_aiter = False
    print("aiter not available, skipping aiter benchmarks")

torch.manual_seed(0)
random.seed(0)

# ── Config ──────────────────────────────────────────
B      = 8
D      = 128
H      = 16
H_KV   = 16
N      = 2048
causal = 1
dtype  = torch.bfloat16

num_warmup = 200
num_iters  = 100

# ── Helpers ─────────────────────────────────────────
def attn_flops(batch, seqlen, nheads, headdim, causal, mode="fwd"):
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return {"fwd": f, "bwd": 2.5 * f, "fwd_bwd": 3.5 * f}[mode]

def tflops(flop, ms):
    return (flop / 1e12) / (ms / 1e3)

def gen(shape):
    t = torch.randn(shape, dtype=dtype, device='cuda')
    mag = t.norm(dim=-1, keepdim=True)
    return (t * (torch.randn(mag.shape, dtype=dtype, device='cuda') * 0.1 + 10) / mag).contiguous()

def bench(fn, num_warmup=num_warmup, num_iters=num_iters):
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sum(times) / len(times)

# ── Generate inputs (BHND canonical, then derive layouts) ───
Q_bhnd = gen((B, H, N, D)).requires_grad_(True)
K_bhnd = gen((B, H_KV, N, D)).requires_grad_(True)
V_bhnd = gen((B, H_KV, N, D)).requires_grad_(True)
dO_bhnd = gen((B, H, N, D))

flops_fwd     = attn_flops(B, N, H, D, causal, "fwd")
flops_bwd     = attn_flops(B, N, H, D, causal, "bwd")
flops_fwd_bwd = attn_flops(B, N, H, D, causal, "fwd_bwd")

results = {}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1) HipKittens BSHD (existing)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 70)
print("HipKittens BSHD (existing path)")
print("=" * 70)

Q_bshd = Q_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)
K_bshd = K_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)
V_bshd = V_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)
dO_bshd = dO_bhnd.transpose(1, 2).contiguous()

O_hk_bshd = torch.empty_like(Q_bshd)
L_hk      = torch.empty((B, H, 1, N), device='cuda', dtype=torch.float32)

def hk_fwd_bshd():
    tk_kernel_fwd.dispatch_fwd(Q_bshd, K_bshd, V_bshd, O_hk_bshd, L_hk)

def hk_bwd_bshd():
    dQ_in = torch.zeros((B, H, N, D), dtype=dtype, device='cuda')
    dQ    = torch.empty_like(Q_bshd)
    dK    = torch.empty_like(K_bshd)
    dV    = torch.empty_like(V_bshd)
    delta = torch.empty((B, H, 1, N), device='cuda', dtype=torch.float32)
    tk_kernel_bkwd_prep.dispatch_prep(O_hk_bshd, dO_bshd, delta)
    tk_kernel_bkwd.dispatch_bwd_combined(Q_bshd, K_bshd, V_bshd, dO_bshd, dQ_in, dK, dV, L_hk, delta)
    tk_kernel_bkwd_prep.dispatch_dq_shuffle(dQ_in, dQ)

def hk_fwd_bwd_bshd():
    hk_fwd_bshd()
    hk_bwd_bshd()

t_fwd = bench(hk_fwd_bshd)
t_bwd = bench(hk_bwd_bshd)
t_fb  = bench(hk_fwd_bwd_bshd)
results["HK-BSHD"] = (t_fwd, t_bwd, t_fb)
print(f"  FWD:     {t_fwd:.3f} ms  ->  {tflops(flops_fwd, t_fwd):.2f} TFLOPS")
print(f"  BWD:     {t_bwd:.3f} ms  ->  {tflops(flops_bwd, t_bwd):.2f} TFLOPS")
print(f"  FWD+BWD: {t_fb:.3f} ms  ->  {tflops(flops_fwd_bwd, t_fb):.2f} TFLOPS")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2) HipKittens SBHD (native, zero-copy)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print()
print("=" * 70)
print("HipKittens SBHD (native, zero-copy)")
print("=" * 70)

Q_sbhd = Q_bshd.transpose(0, 1).contiguous().detach().requires_grad_(True)
K_sbhd = K_bshd.transpose(0, 1).contiguous().detach().requires_grad_(True)
V_sbhd = V_bshd.transpose(0, 1).contiguous().detach().requires_grad_(True)
dO_sbhd = dO_bshd.transpose(0, 1).contiguous()

O_hk_sbhd = torch.empty((N, B, H, D), dtype=dtype, device='cuda')
L_hk_s    = torch.empty((B, H, 1, N), device='cuda', dtype=torch.float32)

def hk_fwd_sbhd():
    tk_kernel_fwd.dispatch_fwd_sbhd(Q_sbhd, K_sbhd, V_sbhd, O_hk_sbhd, L_hk_s)

def hk_bwd_sbhd():
    dQ_in = torch.zeros((B, H, N, D), dtype=dtype, device='cuda')
    dQ    = torch.empty((N, B, H, D), dtype=dtype, device='cuda')
    dK    = torch.empty((N, B, H_KV, D), dtype=dtype, device='cuda')
    dV    = torch.empty((N, B, H_KV, D), dtype=dtype, device='cuda')
    delta = torch.empty((B, H, 1, N), device='cuda', dtype=torch.float32)
    tk_kernel_bkwd_prep.dispatch_prep_sbhd(O_hk_sbhd, dO_sbhd, delta)
    tk_kernel_bkwd.dispatch_bwd_combined_sbhd(Q_sbhd, K_sbhd, V_sbhd, dO_sbhd, dQ_in, dK, dV, L_hk_s, delta)
    tk_kernel_bkwd_prep.dispatch_dq_shuffle_sbhd(dQ_in, dQ)

def hk_fwd_bwd_sbhd():
    hk_fwd_sbhd()
    hk_bwd_sbhd()

t_fwd = bench(hk_fwd_sbhd)
t_bwd = bench(hk_bwd_sbhd)
t_fb  = bench(hk_fwd_bwd_sbhd)
results["HK-SBHD"] = (t_fwd, t_bwd, t_fb)
print(f"  FWD:     {t_fwd:.3f} ms  ->  {tflops(flops_fwd, t_fwd):.2f} TFLOPS")
print(f"  BWD:     {t_bwd:.3f} ms  ->  {tflops(flops_bwd, t_bwd):.2f} TFLOPS")
print(f"  FWD+BWD: {t_fb:.3f} ms  ->  {tflops(flops_fwd_bwd, t_fb):.2f} TFLOPS")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3) AITER BSHD (existing)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if use_aiter:
    print()
    print("=" * 70)
    print("AITER BSHD (existing path)")
    print("=" * 70)

    Q_a = Q_bshd.detach().requires_grad_(True)
    K_a = K_bshd.detach().requires_grad_(True)
    V_a = V_bshd.detach().requires_grad_(True)
    dO_a = dO_bshd.clone()

    def aiter_fwd_bshd():
        return aiter.flash_attn_func(Q_a, K_a, V_a, causal=bool(causal), return_lse=True, deterministic=False)

    def aiter_bwd_bshd():
        out, _ = aiter.flash_attn_func(Q_a, K_a, V_a, causal=bool(causal), return_lse=True, deterministic=False)
        out.backward(dO_a)

    def aiter_fwd_only():
        aiter.flash_attn_func(Q_a, K_a, V_a, causal=bool(causal), return_lse=True, deterministic=False)

    t_fwd = bench(aiter_fwd_only)
    t_bwd = bench(aiter_bwd_bshd)
    results["AITER-BSHD"] = (t_fwd, t_bwd, t_fwd + t_bwd)
    print(f"  FWD:     {t_fwd:.3f} ms  ->  {tflops(flops_fwd, t_fwd):.2f} TFLOPS")
    print(f"  BWD:     {t_bwd:.3f} ms  ->  {tflops(flops_bwd, t_bwd):.2f} TFLOPS")
    # Note: bwd includes fwd recompute in autograd, so fwd+bwd ≈ bwd timing
    print(f"  FWD+BWD: ~{t_bwd:.3f} ms  ->  {tflops(flops_fwd_bwd, t_bwd):.2f} TFLOPS")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4) AITER with SBHD input (transpose overhead)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if use_aiter:
    print()
    print("=" * 70)
    print("AITER SBHD->BSHD (transpose + contiguous overhead)")
    print("=" * 70)

    def aiter_fwd_sbhd_to_bshd():
        q = Q_sbhd.transpose(0, 1).contiguous()
        k = K_sbhd.transpose(0, 1).contiguous()
        v = V_sbhd.transpose(0, 1).contiguous()
        out, _ = aiter.flash_attn_func(q, k, v, causal=bool(causal), return_lse=True, deterministic=False)
        return out.transpose(0, 1)

    def aiter_fwd_bwd_sbhd_to_bshd():
        q = Q_sbhd.detach().requires_grad_(True)
        k = K_sbhd.detach().requires_grad_(True)
        v = V_sbhd.detach().requires_grad_(True)
        q_b = q.transpose(0, 1).contiguous()
        k_b = k.transpose(0, 1).contiguous()
        v_b = v.transpose(0, 1).contiguous()
        out, _ = aiter.flash_attn_func(q_b, k_b, v_b, causal=bool(causal), return_lse=True, deterministic=False)
        dO_b = dO_sbhd.transpose(0, 1).contiguous()
        out.backward(dO_b)

    t_fwd = bench(aiter_fwd_sbhd_to_bshd)
    t_bwd = bench(aiter_fwd_bwd_sbhd_to_bshd)
    results["AITER-SBHD"] = (t_fwd, t_bwd, t_fwd + t_bwd)
    print(f"  FWD:     {t_fwd:.3f} ms  ->  {tflops(flops_fwd, t_fwd):.2f} TFLOPS")
    print(f"  BWD:     {t_bwd:.3f} ms  ->  {tflops(flops_bwd, t_bwd):.2f} TFLOPS")
    print(f"  FWD+BWD: ~{t_bwd:.3f} ms  ->  {tflops(flops_fwd_bwd, t_bwd):.2f} TFLOPS")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Correctness: HK-SBHD vs HK-BSHD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print()
print("=" * 70)
print("Correctness: HK-SBHD output vs HK-BSHD output")
print("=" * 70)

hk_fwd_bshd()
hk_fwd_sbhd()

O_sbhd_as_bshd = O_hk_sbhd.transpose(0, 1)
diff = (O_hk_bshd.float() - O_sbhd_as_bshd.float()).abs()
max_diff = diff.max().item()
cos = torch.nn.functional.cosine_similarity(
    O_hk_bshd.float().flatten(), O_sbhd_as_bshd.float().flatten(), dim=0
).item()
print(f"  Max abs diff: {max_diff:.6f}")
print(f"  Cosine sim:   {cos:.8f}")
print(f"  Match:        {'PASS' if max_diff < 1e-3 else 'FAIL'}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Summary table
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print()
print("=" * 70)
print(f"Summary:  B={B}  H={H}  H_KV={H_KV}  N={N}  D={D}  causal={causal}")
print("=" * 70)
print(f"{'Method':<22} {'FWD ms':>8} {'FWD TF':>8} {'BWD ms':>8} {'BWD TF':>8} {'F+B ms':>8} {'F+B TF':>8}")
print("-" * 70)
for name, (tf, tb, tfb) in results.items():
    print(f"{name:<22} {tf:8.3f} {tflops(flops_fwd,tf):8.2f} {tb:8.3f} {tflops(flops_bwd,tb):8.2f} {tfb:8.3f} {tflops(flops_fwd_bwd,tfb):8.2f}")
