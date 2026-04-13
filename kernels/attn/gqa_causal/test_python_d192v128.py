import torch
import tk_kernel
import random
import time
import math
import sys
import os
from torch.nn.functional import scaled_dot_product_attention

torch.manual_seed(0)
random.seed(0)

torch.set_printoptions(
    precision=3,
    sci_mode=False,
    linewidth=220,
    threshold=float("inf")
)

# Inputs
B = int(sys.argv[1]) if len(sys.argv) > 1 else 16
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
H = int(sys.argv[3]) if len(sys.argv) > 3 else 64
H_KV = int(sys.argv[4]) if len(sys.argv) > 4 else 8
causal = int(sys.argv[5]) if len(sys.argv) > 5 else 1

# Asymmetric head dimensions: Q/K use D_QK, V/O use D_V
D_QK = int(sys.argv[6]) if len(sys.argv) > 6 else 192
D_V  = int(sys.argv[7]) if len(sys.argv) > 7 else 128

dtype = torch.bfloat16

print(f"Asymmetric Causal Attention: {B=} {N=} {H=} {H_KV=} {D_QK=} {D_V=} {causal=}")

# Q, K have D_QK columns; V has D_V columns
q = torch.randn(B, N, H, D_QK, dtype=dtype, device='cuda', requires_grad=True)
k = torch.randn(B, N, H_KV, D_QK, dtype=dtype, device='cuda', requires_grad=True)
v = torch.randn(B, N, H_KV, D_V, dtype=dtype, device='cuda', requires_grad=True)


def flops(batch, seqlen, nheads, d_qk, d_v, causal, mode="fwd"):
    """Calculate FLOPs for asymmetric attention.
    QK^T: 2 * B * N^2 * H * D_QK
    PV:   2 * B * N^2 * H * D_V
    Total: 2 * B * N^2 * H * (D_QK + D_V)
    """
    flop = 2 * batch * seqlen**2 * nheads * (d_qk + d_v) // (2 if causal else 1)
    return flop

def efficiency(flop, time):
    flop = flop / 1e12
    time = time / 1e3
    return flop / time

def robustness_check(ref, pred):
    ref = ref.float()
    pred = pred.float()
    diff = (ref - pred).abs()
    denom = ref.abs().clamp_min(1e-6)
    mask = (diff > (0.001 + 0.05 * denom))
    error_count = mask.sum().item()
    numel = ref.numel()
    rel_error = error_count / numel
    l2_error = (diff.pow(2).sum().sqrt() / ref.pow(2).sum().sqrt()).item()
    cos = torch.nn.functional.cosine_similarity(ref.flatten(), pred.flatten(), dim=0).item()
    return diff, error_count, numel, rel_error, l2_error, cos, mask


num_warmup = 500
num_iters = 100

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
flops_ref = flops(B, N, H, D_QK, D_V, causal)


# AITER baseline
try:
    import aiter
    print("AITER:")
    for _ in range(num_warmup):
        q = torch.randn(B, N, H, D_QK, dtype=dtype, device='cuda', requires_grad=True)
        k = torch.randn(B, N, H_KV, D_QK, dtype=dtype, device='cuda', requires_grad=True)
        v = torch.randn(B, N, H_KV, D_V, dtype=dtype, device='cuda', requires_grad=True)
        out_ref, lse_ref = aiter.flash_attn_func(q, k, v, causal=causal, return_lse=True, deterministic=True)
    timings_ref = []
    torch.manual_seed(0)
    random.seed(0)
    for _ in range(num_iters):
        q = torch.randn(B, N, H, D_QK, dtype=dtype, device='cuda', requires_grad=True)
        k = torch.randn(B, N, H_KV, D_QK, dtype=dtype, device='cuda', requires_grad=True)
        v = torch.randn(B, N, H_KV, D_V, dtype=dtype, device='cuda', requires_grad=True)
        torch.cuda.synchronize()
        start_event.record()
        out_ref, lse_ref = aiter.flash_attn_func(q, k, v, causal=causal, return_lse=True, deterministic=True)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        timings_ref.append(elapsed_time)
    print(f"{out_ref.dtype=}")
    avg_time_ref = sum(timings_ref) / len(timings_ref)
    eff_ref = efficiency(flops_ref, avg_time_ref)
    print(f"AITER average execution time: {avg_time_ref:.4f} ms")
    print(f"AITER performance: {eff_ref:.2f} TFLOPS for {B=} {H=} {N=} {D_QK=} {D_V=} {causal=}.\n")
    has_aiter = True
except Exception as e:
    print(f"AITER not available or failed: {e}")
    has_aiter = False
    # PyTorch reference
    print("PyTorch reference:")
    q_ref = torch.randn(B, N, H, D_QK, dtype=dtype, device='cuda')
    k_ref = torch.randn(B, N, H_KV, D_QK, dtype=dtype, device='cuda')
    v_ref = torch.randn(B, N, H_KV, D_V, dtype=dtype, device='cuda')
    scale = 1.0 / math.sqrt(D_QK)
    # Expand K/V for GQA
    k_exp = k_ref.repeat_interleave(H // H_KV, dim=2)
    v_exp = v_ref.repeat_interleave(H // H_KV, dim=2)
    # [B, H, N, D]
    q_t = q_ref.transpose(1, 2)
    k_t = k_exp.transpose(1, 2)
    v_t = v_exp.transpose(1, 2)
    S = torch.matmul(q_t.float(), k_t.float().transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(N, N, device='cuda', dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float('-inf'))
    P = torch.softmax(S, dim=-1)
    out_ref = torch.matmul(P, v_t.float()).to(dtype).transpose(1, 2)
    lse_ref = torch.logsumexp(S, dim=-1)


print("HipKittens:")
for _ in range(num_warmup):
    # Output has D_V columns
    out = torch.zeros(B, N, H, D_V, dtype=dtype, device='cuda', requires_grad=True)
    lse = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda', requires_grad=True)
    q = torch.randn(B, N, H, D_QK, dtype=dtype, device='cuda', requires_grad=True)
    k = torch.randn(B, N, H_KV, D_QK, dtype=dtype, device='cuda', requires_grad=True)
    v = torch.randn(B, N, H_KV, D_V, dtype=dtype, device='cuda', requires_grad=True)
    tk_kernel.dispatch_micro(q, k, v, out, lse)
timings = []
out = torch.zeros(B, N, H, D_V, dtype=dtype, device='cuda', requires_grad=True)
lse = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda', requires_grad=True)
torch.manual_seed(0)
random.seed(0)
for _ in range(num_iters):
    q = torch.randn(B, N, H, D_QK, dtype=dtype, device='cuda', requires_grad=True)
    k = torch.randn(B, N, H_KV, D_QK, dtype=dtype, device='cuda', requires_grad=True)
    v = torch.randn(B, N, H_KV, D_V, dtype=dtype, device='cuda', requires_grad=True)
    torch.cuda.synchronize()
    start_event.record()
    tk_kernel.dispatch_micro(q, k, v, out, lse)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings.append(elapsed_time)
print(f"{out.dtype=}")
avg_time = sum(timings) / len(timings)
eff = efficiency(flops_ref, avg_time)
print(f"Average execution time: {avg_time:.4f} ms")
print(f"Performance: {eff:.2f} TFLOPS for {B=} {H=} {N=} {D_QK=} {D_V=} {causal=}.\n")

# Compare against reference
num_print = 16
if has_aiter:
    print(f"\n TK vs AITER comparison:")
    print("\nO outputs:")
    print("TK: ", out[0, 0, :num_print, 0], "Max:", out.max().item())
    print("AITER: ", out_ref[0, 0, :num_print, 0], "Max:", out_ref.max().item())
    print("\nLSE outputs:")
    print("TK: ", lse[0, 0, 0, :num_print], "Max:", lse.max().item())
    print("AITER: ", lse_ref[0, 0, :num_print], "Max:", lse_ref.max().item())
    print("Robustness check:")
    o_diff, o_err_cnt, o_total, o_rel_error, o_l2_error, o_cos, o_mask = robustness_check(out, out_ref)
    print(f"O: max_abs={o_diff.max().item():.6f}, max_rel={o_rel_error:.4f}, "
        f"rel_l2={o_l2_error:.4f}, cos={o_cos:.6f}, "
        f"errors={o_err_cnt}/{o_total} ({100*o_err_cnt/o_total:.4f}%)")
    l_diff, l_err_cnt, l_total, l_rel_error, l_l2_error, l_cos, l_mask = robustness_check(lse, lse_ref.unsqueeze(-1).transpose(-1, -2))
    print(f"LSE: max_abs={l_diff.max().item():.6f}, max_rel={l_rel_error:.4f}, "
        f"rel_l2={l_l2_error:.4f}, cos={l_cos:.6f}, "
        f"errors={l_err_cnt}/{l_total} ({100*l_err_cnt/l_total:.4f}%)")
else:
    print(f"\n TK vs PyTorch comparison:")
    print("\nO outputs:")
    print("TK: ", out[0, 0, :num_print, 0], "Max:", out.max().item())
    print("Ref: ", out_ref[0, 0, :num_print, 0], "Max:", out_ref.max().item())
    print("Robustness check:")
    o_diff, o_err_cnt, o_total, o_rel_error, o_l2_error, o_cos, o_mask = robustness_check(out, out_ref)
    print(f"O: max_abs={o_diff.max().item():.6f}, max_rel={o_rel_error:.4f}, "
        f"rel_l2={o_l2_error:.4f}, cos={o_cos:.6f}, "
        f"errors={o_err_cnt}/{o_total} ({100*o_err_cnt/o_total:.4f}%)")
