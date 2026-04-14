#!/usr/bin/env python3
"""
Backward-pass test for HipKittens asymmetric attention (D_QK=192, D_V=128).

Kernels available for asymmetric D:
  - tk_kernel_fwd        (forward, D_QK=192 / D_V=128)
  - tk_kernel_bkwd_prep  (backward prep + dQ shuffle, D_QK=192 / D_V=128)
  - tk_kernel_bkwd       (main backward -- symmetric D=128 ONLY; NOT yet
                           implemented for asymmetric D)

This script tests the forward + backward prep kernels and compares against
AITER's flash_attn_func backward pass.  The main backward kernel is tested
only when it is available for D=128 symmetric mode.

Build:
    cd /shared_nfs/kyle/test/HipKittens/kernels/attn/gqa_causal_backwards
    make asymmetric ATTN_D_QK=192 ATTN_D_V=128 ATTN_N=<N>

Usage:
    python test_python_d192v128.py [B] [N] [H] [H_KV] [causal]
    python test_python_d192v128.py 16 1024 64 8 1
"""

import torch
import random
import math
import time
import sys
import os

torch.manual_seed(0)
random.seed(0)

torch.set_printoptions(
    precision=3,
    sci_mode=False,
    linewidth=220,
    threshold=float("inf")
)

# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------
B      = int(sys.argv[1]) if len(sys.argv) > 1 else 16
N      = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
H      = int(sys.argv[3]) if len(sys.argv) > 3 else 64
H_KV   = int(sys.argv[4]) if len(sys.argv) > 4 else 8
causal = int(sys.argv[5]) if len(sys.argv) > 5 else 1

# Asymmetric head dimensions
D_QK = 192   # Q/K dimension
D_V  = 128   # V/O dimension

dtype = torch.bfloat16
group_size = H // H_KV

print(f"Asymmetric Backward Test: {B=} {N=} {H=} {H_KV=} {D_QK=} {D_V=} {causal=}")
print(f"  group_size={group_size}")

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
num_warmup = 500
num_iters  = 100
start_event = torch.cuda.Event(enable_timing=True)
end_event   = torch.cuda.Event(enable_timing=True)


def flops(batch, seqlen, nheads, d_qk, d_v, causal, mode="bwd"):
    """FLOPs for asymmetric attention.
    Forward:  2 * B * N^2 * H * (D_QK + D_V)  [/2 if causal]
    Backward: ~2.5x forward
    """
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 2 * batch * seqlen**2 * nheads * (d_qk + d_v) // (2 if causal else 1)
    if mode == "fwd":
        return f
    elif mode == "bwd":
        return int(2.5 * f)
    else:
        return int(3.5 * f)


def efficiency(flop, time_ms):
    return (flop / 1e12) / (time_ms / 1e3)


def robustness_check(ref, pred, label=""):
    ref  = ref.float()
    pred = pred.float()
    diff = (ref - pred).abs()
    denom = ref.abs().clamp_min(1e-6)
    mask  = (diff > (0.001 + 0.05 * denom))
    error_count = mask.sum().item()
    numel = ref.numel()
    rel_error = error_count / numel
    l2_error  = (diff.pow(2).sum().sqrt() / ref.pow(2).sum().sqrt()).item()
    cos = torch.nn.functional.cosine_similarity(
        ref.flatten(), pred.flatten(), dim=0
    ).item()
    return diff, error_count, numel, rel_error, l2_error, cos, mask


# ---------------------------------------------------------------
# Data generation (BHND format, as in original test_python.py)
# ---------------------------------------------------------------
mean = 10
std  = 0.1

flops_bwd = flops(B, N, H, D_QK, D_V, causal, mode="bwd")
flops_fwd = flops(B, N, H, D_QK, D_V, causal, mode="fwd")


def generate_tensor(shape, mean, std, dtype, device):
    tensor = torch.randn(shape, dtype=dtype, device=device)
    magnitude = torch.norm(tensor, dim=-1, keepdim=True)
    scaled_tensor = tensor * (
        torch.randn(magnitude.shape, dtype=dtype, device=device) * std + mean
    ) / magnitude
    return scaled_tensor.contiguous()


def generate_inputs():
    """Generate in BHND format.
    Q:  [B, H,    N, D_QK]
    K:  [B, H_KV, N, D_QK]
    V:  [B, H_KV, N, D_V]
    dO: [B, H,    N, D_V]
    """
    Q  = generate_tensor((B, H,    N, D_QK), mean, std, dtype, 'cuda')
    K  = generate_tensor((B, H_KV, N, D_QK), mean, std, dtype, 'cuda')
    V  = generate_tensor((B, H_KV, N, D_V),  mean, std, dtype, 'cuda')
    dO = generate_tensor((B, H,    N, D_V),  mean, std, dtype, 'cuda')
    Q.requires_grad_(True)
    K.requires_grad_(True)
    V.requires_grad_(True)
    return Q, K, V, dO


Q_bhnd, K_bhnd, V_bhnd, dO_bhnd = generate_inputs()


# ---------------------------------------------------------------
# Reference: PyTorch manual implementation (float64)
# ---------------------------------------------------------------
def expand_kv_for_gqa(K, V, h_q, h_kv):
    group = h_q // h_kv
    K_exp = K.repeat_interleave(group, dim=1)
    V_exp = V.repeat_interleave(group, dim=1)
    return K_exp, V_exp


def reference_fwd_bwd(Q, K, V, dO, causal):
    """Full forward + backward in float64, BHND layout."""
    q_ = Q.detach().to(torch.float64).requires_grad_(True)
    k_ = K.detach().to(torch.float64).requires_grad_(True)
    v_ = V.detach().to(torch.float64).requires_grad_(True)

    k_exp, v_exp = expand_kv_for_gqa(k_, v_, H, H_KV)
    scale = 1.0 / math.sqrt(D_QK)
    S = torch.matmul(q_, k_exp.transpose(-2, -1)) * scale

    if causal:
        mask = torch.triu(
            torch.ones(N, N, device='cuda', dtype=torch.bool), diagonal=1
        )
        S = S.masked_fill(mask, float('-inf'))

    P = torch.softmax(S, dim=-1)
    O = torch.matmul(P, v_exp)

    # LSE for each row
    L = torch.logsumexp(S, dim=-1)

    # Backward
    dO_ = dO.detach().to(torch.float64)
    O.backward(dO_, retain_graph=True)

    return O, L, q_.grad, k_.grad, v_.grad


# ---------------------------------------------------------------
# AITER forward + backward (skip if env SKIP_AITER=1 or backward faults)
# ---------------------------------------------------------------
use_aiter = os.environ.get("SKIP_AITER", "1") != "1"
if use_aiter:
    try:
        import aiter
    except ImportError:
        print("WARNING: aiter not available, skipping AITER comparison")
        use_aiter = False

if use_aiter:
    timings = []
    print("\nRunning AITER backward ...")

    # AITER expects BNHD layout
    for _ in range(num_warmup):
        Q_a = Q_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)
        K_a = K_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)
        V_a = V_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)
        dO_a = dO_bhnd.transpose(1, 2).contiguous()
        out_a, lse_a = aiter.flash_attn_func(
            Q_a, K_a, V_a, causal=bool(causal),
            return_lse=True, deterministic=False
        )
        out_a.backward(dO_a)

    for _ in range(num_iters):
        Q_a = Q_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)
        K_a = K_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)
        V_a = V_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)
        dO_a = dO_bhnd.transpose(1, 2).contiguous()
        out_a, lse_a = aiter.flash_attn_func(
            Q_a, K_a, V_a, causal=bool(causal),
            return_lse=True, deterministic=False
        )
        torch.cuda.synchronize()
        start_event.record()
        out_a.backward(dO_a)
        end_event.record()
        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event))

    avg_aiter = sum(timings) / len(timings)
    eff_aiter = efficiency(flops_bwd, avg_aiter)
    print(f"AITER backward avg time: {avg_aiter:.4f} ms")
    print(f"AITER backward perf: {eff_aiter:.2f} TFLOPS  "
          f"({B=} {H=} {H_KV=} {N=} {D_QK=} {D_V=} {causal=})")

    dQ_aiter_bnhd = Q_a.grad
    dK_aiter_bnhd = K_a.grad
    dV_aiter_bnhd = V_a.grad
    O_aiter_bnhd  = out_a
    lse_aiter     = lse_a


# ---------------------------------------------------------------
# HipKittens forward (asymmetric D_QK=192, D_V=128)
# ---------------------------------------------------------------
print("\nRunning HipKittens forward ...")
import tk_kernel_fwd

# Forward expects BNHD: Q [B,N,H,D_QK], K [B,N,H_KV,D_QK], V [B,N,H_KV,D_V]
Q_tk  = Q_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)
K_tk  = K_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)
V_tk  = V_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)
dO_tk = dO_bhnd.transpose(1, 2).contiguous()

O_tk  = torch.zeros(B, N, H, D_V, dtype=dtype, device='cuda')
L_tk  = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda')

tk_kernel_fwd.dispatch_fwd(Q_tk, K_tk, V_tk, O_tk, L_tk)
torch.cuda.synchronize()

# Benchmark forward
timings_fwd = []
for _ in range(num_warmup):
    O_tmp = torch.zeros_like(O_tk)
    L_tmp = torch.zeros_like(L_tk)
    tk_kernel_fwd.dispatch_fwd(Q_tk, K_tk, V_tk, O_tmp, L_tmp)
for _ in range(num_iters):
    O_tmp = torch.zeros_like(O_tk)
    L_tmp = torch.zeros_like(L_tk)
    torch.cuda.synchronize()
    start_event.record()
    tk_kernel_fwd.dispatch_fwd(Q_tk, K_tk, V_tk, O_tmp, L_tmp)
    end_event.record()
    torch.cuda.synchronize()
    timings_fwd.append(start_event.elapsed_time(end_event))

avg_fwd = sum(timings_fwd) / len(timings_fwd)
eff_fwd = efficiency(flops_fwd, avg_fwd)
print(f"HK forward avg time: {avg_fwd:.4f} ms")
print(f"HK forward perf: {eff_fwd:.2f} TFLOPS")


# ---------------------------------------------------------------
# HipKittens backward prep (delta computation + dQ shuffle)
# ---------------------------------------------------------------
print("\nRunning HipKittens backward prep ...")
import tk_kernel_bkwd_prep

# dispatch_prep expects: O [B,N,H,D_V], dO [B,N,H,D_V], delta [B,H,1,N]
delta_tk = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda')
tk_kernel_bkwd_prep.dispatch_prep(O_tk, dO_tk, delta_tk)
torch.cuda.synchronize()


# ---------------------------------------------------------------
# Try loading the main backward kernel (only available for D=128)
# ---------------------------------------------------------------
has_bkwd = False
try:
    import tk_kernel_bkwd
    has_bkwd = True
    print("\nMain backward kernel (tk_kernel_bkwd) loaded (asymmetric build: D_QK=192, D_V=128).\n")
except ImportError:
    print("\nMain backward kernel (tk_kernel_bkwd) not available — build with "
          "`make asymmetric ATTN_D_QK=192`.\n")


# ---------------------------------------------------------------
# Correctness checks: HK forward vs PyTorch float64 reference
# ---------------------------------------------------------------
print("\nComputing PyTorch float64 reference ...")
O_ref, L_ref, dQ_ref, dK_ref, dV_ref = reference_fwd_bwd(
    Q_bhnd, K_bhnd, V_bhnd, dO_bhnd, causal
)

print("=" * 90)
print("Correctness: HK forward vs float64 reference")
print("=" * 90)
O_ref_bnhd = O_ref.transpose(1, 2).to(dtype)
o_diff, o_err, o_tot, o_rel, o_l2, o_cos, _ = robustness_check(O_tk, O_ref_bnhd, "O")
print(f"O:   max_abs={o_diff.max().item():.6f}  rel_err={o_rel:.4f}  "
      f"l2={o_l2:.6f}  cos={o_cos:.6f}  "
      f"errors={o_err}/{o_tot} ({100*o_err/o_tot:.4f}%)")


# ---------------------------------------------------------------
# Gradient comparison: HK backward vs float64 reference
# ---------------------------------------------------------------
print()
print("=" * 90)
print("Gradient comparison: HK backward vs float64 reference")
print("=" * 90)

avg_hk_bwd = None
eff_hk_bwd = None
if has_bkwd:
    # dQ atomic accumulation layout: BHND (axis=2 stores along N rows)
    dQ_tk_in = torch.zeros(B, H, N, D_QK, dtype=dtype, device='cuda')
    dQ_tk = torch.zeros(B, N, H, D_QK, dtype=dtype, device='cuda')
    dK_tk = torch.zeros(B, N, H_KV, D_QK, dtype=dtype, device='cuda')
    dV_tk = torch.zeros(B, N, H_KV, D_V, dtype=dtype, device='cuda')

    num_warmup_bwd = 5
    num_iters_bwd = 5
    dQ_throwaway = torch.zeros(B, H, N, D_QK, dtype=dtype, device='cuda')
    for _ in range(num_warmup_bwd):
        dQ_throwaway.zero_()
        dQ_tk_in.zero_()
        dK_tk.zero_()
        dV_tk.zero_()
        tk_kernel_bkwd.dispatch_bwd_combined(
            Q_tk, K_tk, V_tk, dO_tk,
            dQ_throwaway, dK_tk, dV_tk,
            L_tk, delta_tk,
        )
        tk_kernel_bkwd.dispatch_bwd_dq(
            Q_tk, K_tk, V_tk, dO_tk,
            dQ_tk_in, L_tk, delta_tk,
        )
        tk_kernel_bkwd_prep.dispatch_dq_shuffle(dQ_tk_in, dQ_tk)

    timings_bwd = []
    for _ in range(num_iters_bwd):
        dQ_throwaway.zero_()
        dQ_tk_in.zero_()
        dK_tk.zero_()
        dV_tk.zero_()
        torch.cuda.synchronize()
        start_event.record()
        tk_kernel_bkwd.dispatch_bwd_combined(
            Q_tk, K_tk, V_tk, dO_tk,
            dQ_throwaway, dK_tk, dV_tk,
            L_tk, delta_tk,
        )
        tk_kernel_bkwd.dispatch_bwd_dq(
            Q_tk, K_tk, V_tk, dO_tk,
            dQ_tk_in, L_tk, delta_tk,
        )
        tk_kernel_bkwd_prep.dispatch_dq_shuffle(dQ_tk_in, dQ_tk)
        end_event.record()
        torch.cuda.synchronize()
        timings_bwd.append(start_event.elapsed_time(end_event))

    avg_hk_bwd = sum(timings_bwd) / len(timings_bwd)
    eff_hk_bwd = efficiency(flops_bwd, avg_hk_bwd)
    print(f"HK backward avg time: {avg_hk_bwd:.4f} ms")
    print(f"HK backward perf: {eff_hk_bwd:.2f} TFLOPS\n")

    # Reference gradients in BNHD for comparison
    dQ_r = dQ_ref.transpose(1, 2).contiguous().to(dtype)   # BHND → BNHD
    dK_r = dK_ref.transpose(1, 2).contiguous().to(dtype)
    dV_r = dV_ref.transpose(1, 2).contiguous().to(dtype)

    dq_diff, dq_err, dq_tot, dq_rel, dq_l2, dq_cos, _ = robustness_check(dQ_r, dQ_tk, "dQ")
    dk_diff, dk_err, dk_tot, dk_rel, dk_l2, dk_cos, _ = robustness_check(dK_r, dK_tk, "dK")
    dv_diff, dv_err, dv_tot, dv_rel, dv_l2, dv_cos, _ = robustness_check(dV_r, dV_tk, "dV")

    print(f"dQ: max_abs={dq_diff.max().item():.6f}  rel_err={dq_rel:.4f}  "
          f"l2={dq_l2:.6f}  cos={dq_cos:.6f}  "
          f"errors={dq_err}/{dq_tot} ({100*dq_err/dq_tot:.4f}%)")
    print(f"dK: max_abs={dk_diff.max().item():.6f}  rel_err={dk_rel:.4f}  "
          f"l2={dk_l2:.6f}  cos={dk_cos:.6f}  "
          f"errors={dk_err}/{dk_tot} ({100*dk_err/dk_tot:.4f}%)")
    print(f"dV: max_abs={dv_diff.max().item():.6f}  rel_err={dv_rel:.4f}  "
          f"l2={dv_l2:.6f}  cos={dv_cos:.6f}  "
          f"errors={dv_err}/{dv_tot} ({100*dv_err/dv_tot:.4f}%)")
else:
    print("Build tk_kernel_bkwd with `make asymmetric ATTN_D_QK=192` to run gradient checks.\n")

# ---------------------------------------------------------------
# Performance summary
# ---------------------------------------------------------------
print()
print("=" * 90)
print("Performance Summary")
print("=" * 90)
print(f"HK  forward:  {avg_fwd:.4f} ms  ({eff_fwd:.2f} TFLOPS)")
if has_bkwd and avg_hk_bwd is not None:
    print(f"HK  backward: {avg_hk_bwd:.4f} ms  ({eff_hk_bwd:.2f} TFLOPS)")
if use_aiter:
    print(f"AITER backward: {avg_aiter:.4f} ms  ({eff_aiter:.2f} TFLOPS)")
print(f"  Config: {B=} {H=} {H_KV=} {N=} {D_QK=} {D_V=} {causal=}")

print("\nDone.")
