"""
SBHD vs AITER forward-only FLOPS benchmark.
Compares HipKittens BSHD / SBHD / AITER paths.
"""
import torch, random
import tk_kernel_fwd

use_aiter = True
try:
    import aiter
except ImportError:
    use_aiter = False
    print("aiter not available, skipping")

torch.manual_seed(0)
random.seed(0)

B, D, H, H_KV, N = 8, 128, 16, 16, 2048
causal = 1
dtype = torch.bfloat16
num_warmup, num_iters = 300, 200

def attn_flops_fwd(b, s, h, d, c):
    return 4 * b * s**2 * h * d // (2 if c else 1)

def tflops(flop, ms):
    return (flop / 1e12) / (ms / 1e3)

def gen(shape):
    t = torch.randn(shape, dtype=dtype, device='cuda')
    mag = t.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (t * (torch.randn_like(mag) * 0.1 + 10) / mag).contiguous()

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

flops_f = attn_flops_fwd(B, N, H, D, causal)

# Inputs
Q_bshd = gen((B, N, H, D))
K_bshd = gen((B, N, H_KV, D))
V_bshd = gen((B, N, H_KV, D))

Q_sbhd = gen((N, B, H, D))
K_sbhd = gen((N, B, H_KV, D))
V_sbhd = gen((N, B, H_KV, D))

results = {}

# ── HipKittens BSHD ─────────────────────────────────
O_bshd = torch.empty_like(Q_bshd)
L_bshd = torch.empty((B, H, 1, N), device='cuda', dtype=torch.float32)
def hk_fwd_bshd():
    tk_kernel_fwd.dispatch_fwd(Q_bshd, K_bshd, V_bshd, O_bshd, L_bshd)
t = bench(hk_fwd_bshd)
results["HK-BSHD"] = t
print(f"HK-BSHD  FWD: {t:.4f} ms  {tflops(flops_f, t):.2f} TFLOPS")

# ── HipKittens SBHD (native) ────────────────────────
O_sbhd = torch.empty_like(Q_sbhd)
L_sbhd = torch.empty((B, H, 1, N), device='cuda', dtype=torch.float32)
def hk_fwd_sbhd():
    tk_kernel_fwd.dispatch_fwd_sbhd(Q_sbhd, K_sbhd, V_sbhd, O_sbhd, L_sbhd)
t = bench(hk_fwd_sbhd)
results["HK-SBHD"] = t
print(f"HK-SBHD  FWD: {t:.4f} ms  {tflops(flops_f, t):.2f} TFLOPS")

# ── AITER BSHD ──────────────────────────────────────
if use_aiter:
    def aiter_fwd_bshd():
        aiter.flash_attn_func(Q_bshd, K_bshd, V_bshd, causal=bool(causal), return_lse=True, deterministic=False)
    t = bench(aiter_fwd_bshd)
    results["AITER-BSHD"] = t
    print(f"AITER-BSHD FWD: {t:.4f} ms  {tflops(flops_f, t):.2f} TFLOPS")

# ── AITER from SBHD (transpose overhead) ────────────
if use_aiter:
    def aiter_fwd_sbhd():
        q = Q_sbhd.transpose(0, 1).contiguous()
        k = K_sbhd.transpose(0, 1).contiguous()
        v = V_sbhd.transpose(0, 1).contiguous()
        aiter.flash_attn_func(q, k, v, causal=bool(causal), return_lse=True, deterministic=False)
    t = bench(aiter_fwd_sbhd)
    results["AITER-SBHD"] = t
    print(f"AITER-SBHD FWD: {t:.4f} ms  {tflops(flops_f, t):.2f} TFLOPS")

# ── Correctness (same data, different layout) ───────
print("\n--- Correctness: HK-SBHD vs HK-BSHD (same data) ---")
Q_check = gen((B, N, H, D))
K_check = gen((B, N, H_KV, D))
V_check = gen((B, N, H_KV, D))

O_b = torch.empty_like(Q_check)
L_b = torch.empty((B, H, 1, N), device='cuda', dtype=torch.float32)
tk_kernel_fwd.dispatch_fwd(Q_check, K_check, V_check, O_b, L_b)

Q_s = Q_check.transpose(0, 1).contiguous()
K_s = K_check.transpose(0, 1).contiguous()
V_s = V_check.transpose(0, 1).contiguous()
O_s = torch.empty((N, B, H, D), dtype=dtype, device='cuda')
L_s = torch.empty((B, H, 1, N), device='cuda', dtype=torch.float32)
tk_kernel_fwd.dispatch_fwd_sbhd(Q_s, K_s, V_s, O_s, L_s)

O_s_as_bshd = O_s.transpose(0, 1).contiguous()
diff_O = (O_b.float() - O_s_as_bshd.float()).abs()
cos_O = torch.nn.functional.cosine_similarity(O_b.float().flatten(), O_s_as_bshd.float().flatten(), dim=0).item()
print(f"O: max_diff={diff_O.max().item():.6e}  cos={cos_O:.10f}  {'PASS' if diff_O.max().item() < 1e-3 else 'FAIL'}")

diff_L = (L_b.float() - L_s.float()).abs()
cos_L = torch.nn.functional.cosine_similarity(L_b.float().flatten(), L_s.float().flatten(), dim=0).item()
print(f"L: max_diff={diff_L.max().item():.6e}  cos={cos_L:.10f}  {'PASS' if diff_L.max().item() < 1e-3 else 'FAIL'}")

print(f"\nO_bshd sample [0,0,0,:8]: {O_b[0,0,0,:8]}")
print(f"O_sbhd sample [0,0,0,:8]: {O_s_as_bshd[0,0,0,:8]}")
print(f"L_bshd sample [0,0,0,:8]: {L_b[0,0,0,:8]}")
print(f"L_sbhd sample [0,0,0,:8]: {L_s[0,0,0,:8]}")

# ── Summary ─────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Config: B={B} H={H} H_KV={H_KV} N={N} D={D} causal={causal}")
print(f"{'='*60}")
print(f"{'Method':<16} {'ms':>8} {'TFLOPS':>8} {'vs HK-BSHD':>12}")
print(f"{'-'*60}")
base = results["HK-BSHD"]
for name, t in results.items():
    speedup = base / t
    print(f"{name:<16} {t:8.4f} {tflops(flops_f, t):8.2f} {speedup:11.3f}x")
