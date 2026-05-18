"""FP8 blockwise GEMM test on MI300X.

Three sections share one launcher (selected via BW_SECTION env var):
  fwd   (RCR/NT) : C[M,N] = A[M,K] @ B[N,K]^T   — dispatch_micro
  dgrad (RRR/NN) : C[M,K] = A[M,N] @ B[N,K]     — dispatch_micro_dgrad
  wgrad (CRR/TN) : C[N,K] = g_T[N,M] @ a_T[K,M]^T — dispatch_micro_wgrad

dgrad routes through the fwd kernel via Python-side B^T pre-transpose.
wgrad routes through the fwd kernel via K-contig col-T inputs (matches
Triton's gemm_fp8_blockwise_wgrad_kc_triton_kernel) with per-K-element
b_scale dispatched to the polymorphic micro_tk<true> path.

Scale conventions follow DeepSeek-V3 1×128 / 128×128. The harness
pre-transposes scales so the reduction-axis block is the leading dim
(fwd→K, dgrad→N, wgrad→M).
"""
import math
import os
import subprocess
import sys
import sysconfig

import torch

_KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _KERNEL_DIR)
from tuned_configs import TUNED_REGISTRY  # noqa: E402

# Module selection precedence (first match wins):
#   1. BW_USE_TUNED=1 + BW_BLOCK_M/N/K/NUM_WARPS env vars — manual override.
#   2. TUNED_REGISTRY[(M, N, K, section)] in tuned_configs.py — per-shape best.
#   3. Default tk_kernel.so from `make` (256/128/128/8 baseline).
# Missing tuned .so files are built on demand via `make tuned`.


def _load_tuned_so(bm: int, bn: int, bk: int, nw: int,
                   reg_m: int = 64, raw: int = 0, pre: int = 0, chunk: int = 1):
    mod_name = f"tk_kernel_BM{bm}_BN{bn}_BK{bk}_W{nw}"
    if reg_m != 64: mod_name += f"_RM{reg_m}"
    if raw:        mod_name += f"_RAW{raw}"
    if pre:        mod_name += f"_PRE{pre}"
    if chunk != 1: mod_name += f"_CHK{chunk}"
    ext = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    so_path = os.path.join(_KERNEL_DIR, mod_name + ext)
    if not os.path.exists(so_path):
        r = subprocess.run(
            ["make", "tuned",
             f"BLOCK_M={bm}", f"BLOCK_N={bn}", f"BLOCK_K={bk}",
             f"NUM_WARPS={nw}", f"REG_M={reg_m}",
             f"BW_RAW_DRAIN={raw}", f"BW_PRESCALE_BS={pre}",
             f"BW_CHIPLET_CHUNK={chunk}"],
            cwd=_KERNEL_DIR, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"build of {mod_name} FAILED:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}",
                  file=sys.stderr)
            sys.exit(2)
    import importlib.util
    spec = importlib.util.spec_from_file_location(mod_name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_tk_kernel_module(M_: int, N_: int, K_: int, section_: str):
    # 1. Manual env override.
    if os.environ.get("BW_USE_TUNED", "0") == "1":
        return _load_tuned_so(int(os.environ["BW_BLOCK_M"]),
                              int(os.environ["BW_BLOCK_N"]),
                              int(os.environ["BW_BLOCK_K"]),
                              int(os.environ["BW_NUM_WARPS"]),
                              int(os.environ.get("BW_REG_M", "64")),
                              int(os.environ.get("BW_RAW_DRAIN", "0")),
                              int(os.environ.get("BW_PRESCALE_BS", "0")),
                              int(os.environ.get("BW_CHIPLET_CHUNK", "1")))
    # 2. Per-shape registry.
    cfg = TUNED_REGISTRY.get((M_, N_, K_, section_))
    if cfg is not None:
        return _load_tuned_so(*cfg)
    # 3. Baseline.
    import tk_kernel  # noqa: F401
    return tk_kernel


torch.manual_seed(0)

BK = 128
M = int(os.environ.get("BW_M", "8192"))
N = int(os.environ.get("BW_N", "8192"))
K = int(os.environ.get("BW_K", "8192"))
SECTION = os.environ.get("BW_SECTION", "fwd").lower()
if SECTION not in ("fwd", "dgrad", "wgrad"):
    print(f"BW_SECTION={SECTION!r} not recognized (expected fwd|dgrad|wgrad)")
    sys.exit(2)

tk_kernel = _load_tk_kernel_module(M, N, K, SECTION)
if SECTION == "wgrad":
    # M_fwd=N must be %256 (fwd kernel BLOCK_M); N_fwd=K and K_fwd=M must be
    # %128. All target shapes satisfy these (verified in
    # scripts/_shapes_target.py:hk_fp8_unsupported_reason).
    assert N % 256 == 0, f"wgrad: N={N} must be %256 (fwd BLOCK_M)"
    assert K % 128 == 0, f"wgrad: K={K} must be %128 (fwd BLOCK_N)"
    assert M % 128 == 0, f"wgrad: M={M} must be %128 (fwd BLOCK_K)"
else:
    assert M % 256 == 0, f"M={M} must be divisible by 256 (BLOCK_M)"
    assert N % 128 == 0, f"N={N} must be divisible by 128 (BLOCK_N)"
    assert K % 128 == 0, f"K={K} must be divisible by 128 (BLOCK_K)"
Kb = K // BK
Nb = N // BK

num_warmup = int(os.environ.get("BW_WARMUP", "100"))
num_iters = int(os.environ.get("BW_ITERS", "100"))
check = os.environ.get("BW_CHECK", "1") != "0"


def gen_fp8(rows, cols):
    return (torch.randn(rows, cols, device="cuda") * 0.1).to(torch.float8_e4m3fnuz)


def gen_scales(d0, d1):
    return torch.rand(d0, d1, device="cuda", dtype=torch.float32) * 0.5 + 0.5


def ref_rcr_blockwise(A, B, a_scale_nat, b_scale_nat):
    """fwd: C[M,N] = sum_ki (A[:, k0:k1] @ B[:, k0:k1].T) * a_scale[:, ki:ki+1] * b_scale[:, ki][N-block]"""
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    for ki in range(Kb):
        k0, k1 = ki * BK, (ki + 1) * BK
        partial = A[:, k0:k1].float() @ B[:, k0:k1].float().T
        a_s = a_scale_nat[:, ki : ki + 1]                       # [M, 1]
        b_s = b_scale_nat[:, ki].repeat_interleave(BK)[:N].unsqueeze(0)  # [1, N]
        C += partial * a_s * b_s
    return C


def ref_nn_blockwise(A, B, a_scale_nat, b_scale_nat):
    """dgrad: C[M,K] = sum_ni (A[:, n0:n1] @ B[n0:n1, :]) * a_scale[:, ni:ni+1] * b_scale[ni, :K-block]
    A=dY[M,N], B=W[N,K]; reduction axis = N in 128-chunks."""
    C = torch.zeros(M, K, dtype=torch.float32, device="cuda")
    for ni in range(Nb):
        n0, n1 = ni * BK, (ni + 1) * BK
        partial = A[:, n0:n1].float() @ B[n0:n1, :].float()
        a_s = a_scale_nat[:, ni : ni + 1]                       # [M, 1]
        b_s = b_scale_nat[ni, :].repeat_interleave(BK)[:K].unsqueeze(0)  # [1, K]
        C += partial * a_s * b_s
    return C


def ref_tn_blockwise_kc(g_T, a_T, g_scale_kc, a_scale_kc, M_red, N_out, K_out, Mb_red, Kb_out):
    """wgrad K-contig: C[N, K] = sum_mb (g_T[:, m0:m1] @ a_T[:, m0:m1].T)
    * g_scale_kc[mb, :] (per-N) * a_scale_kc[mb, :] (per-K).
    Matches Triton's gemm_fp8_blockwise_wgrad_kc_triton_kernel convention."""
    C = torch.zeros(N_out, K_out, dtype=torch.float32, device="cuda")
    for mb in range(Mb_red):
        m0, m1 = mb * BK, (mb + 1) * BK
        partial = g_T[:, m0:m1].float() @ a_T[:, m0:m1].float().T  # [N, K]
        g_s = g_scale_kc[mb, :].unsqueeze(1)                        # [N, 1]
        a_s = a_scale_kc[mb, :].unsqueeze(0)                        # [1, K]  per-K
        C += partial * g_s * a_s
    return C


def snr_db(test, ref):
    t, r = test.float(), ref.float()
    sig = (r * r).sum().item()
    noise = ((t - r) ** 2).sum().item()
    if noise == 0:
        return float("inf")
    return 10.0 * math.log10(sig / max(noise, 1e-45))


# ─────────────────────────────────────────────────────────────────────────────
# Per-section input + scale + dispatch wiring. Each branch builds the inputs
# in their natural orientation (caller convention) and pre-transposes scales
# to the reduction-axis-first layout the kernel expects.
# ─────────────────────────────────────────────────────────────────────────────
if SECTION == "fwd":
    label = "RCR/NT fwd"
    out_rows, out_cols = M, N
    A = gen_fp8(M, K)
    B = gen_fp8(N, K)
    C = torch.zeros(out_rows, out_cols, dtype=torch.bfloat16, device="cuda")
    a_scale_nat = gen_scales(M, Kb)        # [M,  Kb]
    b_scale_nat = gen_scales(Nb, Kb)       # [Nb, Kb]
    A_scale_t = a_scale_nat.T.contiguous() # [Kb, M]
    B_scale_t = b_scale_nat.T.contiguous() # [Kb, Nb]
    dispatch = lambda: tk_kernel.dispatch_micro(A, B, C, A_scale_t, B_scale_t)
    ref_fn   = lambda: ref_rcr_blockwise(A, B, a_scale_nat, b_scale_nat)

elif SECTION == "wgrad":
    label = "CRR/TN wgrad (K-contig col-T inputs, routed through fwd kernel)"
    out_rows, out_cols = N, K
    Mb = M // BK
    Kb_local = K // BK
    # Inputs in col-T format with M as the contiguous reduction axis. Per-
    # element scales (g_scale per-N, a_scale per-K) → micro_tk<true>.
    g_T = gen_fp8(N, M)                       # plays kernel A [M_fwd=N, K_fwd=M]
    a_T = gen_fp8(K, M)                       # plays kernel B [N_fwd=K, K_fwd=M]
    C = torch.zeros(out_rows, out_cols, dtype=torch.bfloat16, device="cuda")
    g_scale = gen_scales(Mb, N)
    a_scale = gen_scales(Mb, K)
    dispatch = lambda: tk_kernel.dispatch_micro_wgrad(g_T, a_T, C, g_scale, a_scale)
    ref_fn   = lambda: ref_tn_blockwise_kc(g_T, a_T, g_scale, a_scale,
                                           M, N, K, Mb, Kb_local)

elif SECTION == "dgrad":
    label = "RRR/NN dgrad"
    out_rows, out_cols = M, K
    # A = dY [M, N], B = W [N, K], C = dX [M, K]; reduction axis = N.
    A = gen_fp8(M, N)
    B = gen_fp8(N, K)
    C = torch.zeros(out_rows, out_cols, dtype=torch.bfloat16, device="cuda")
    a_scale_nat = gen_scales(M, Nb)        # [M,  Nb]   (dY scales, per-128 along N)
    b_scale_nat = gen_scales(Nb, Kb)       # [Nb, Kb]   (W  scales, per-128 along N and K)
    A_scale_t = a_scale_nat.T.contiguous() # [Nb, M]
    B_scale_t = b_scale_nat.contiguous()   # [Nb, Kb] already reduction-axis-first
    # Pre-transpose B so the kernel sees its native NT contract:
    #   dgrad C[M,K] = A[M,N] @ B[N,K]  ==  A[M, K_red=N] @ B'[N_out=K, K_red=N]^T
    Bp = B.T.contiguous()
    dispatch = lambda: tk_kernel.dispatch_micro_dgrad(A, Bp, C, A_scale_t, B_scale_t)
    ref_fn   = lambda: ref_nn_blockwise(A, B, a_scale_nat, b_scale_nat)

else:
    raise AssertionError(f"unreachable: SECTION={SECTION}")


print(f"=== FP8 Blockwise ({label}) MI300X — M={M}, N={N}, K={K}, Kb={Kb}, Nb={Nb} ===")
print(f"warmup={num_warmup}, iters={num_iters}, check={check}\n")

# warmup + correctness
dispatch()
torch.cuda.synchronize()

if check:
    C_ref = ref_fn()
    snr = snr_db(C, C_ref)
    diff = (C.float() - C_ref).abs()
    print(f"  SNR:  {snr:.2f} dB")
    print(f"  Max err: {diff.max().item():.4f}   Mean err: {diff.mean().item():.6f}")
    print(f"  Large errs (>0.5): {(diff > 0.5).sum().item()}")
    ok = snr > 48.0
    print(f"  Correctness: {'PASS' if ok else 'FAIL'}\n")
    if not ok:
        sys.exit(1)

start_ev = torch.cuda.Event(enable_timing=True)
end_ev = torch.cuda.Event(enable_timing=True)

for _ in range(num_warmup):
    dispatch()

times = []
for _ in range(num_iters):
    torch.cuda.synchronize()
    start_ev.record()
    dispatch()
    end_ev.record()
    torch.cuda.synchronize()
    times.append(start_ev.elapsed_time(end_ev))

avg = sum(times) / len(times)
flops = 2 * M * N * K
tflops = flops / (avg * 1e9)
print(f"Avg time: {avg:.4f} ms")
print(f"Performance: {tflops:.2f} TFLOPS  (target ~900, PR#52 unscaled: 1125)")
