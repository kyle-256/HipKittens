import torch
import random
import math
import tk_kernel_fwd_d64 as tk_kernel_fwd
import tk_kernel_bkwd_d64 as tk_kernel_bkwd
import tk_kernel_bkwd_prep_d64 as tk_kernel_bkwd_prep
import time
import sys
import os

use_aiter = True
if use_aiter:
    import aiter

torch.manual_seed(0)
random.seed(0)

torch.set_printoptions(precision=3, sci_mode=False, linewidth=220, threshold=float("inf"))

# Inputs (gpt-oss config)
B = int(sys.argv[1]) if len(sys.argv) > 1 else 4
D = 64
H = int(sys.argv[3]) if len(sys.argv) > 3 else 64
H_KV = int(sys.argv[4]) if len(sys.argv) > 4 else 8
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
causal = int(sys.argv[5]) if len(sys.argv) > 5 else 1
dtype = torch.bfloat16

group_size = H // H_KV

num_warmup = 50
num_iters = 50
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)


def flops(batch, seqlen, nheads, headdim, causal, mode="bwd"):
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


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


def expand_kv_for_gqa(K, V, h_q, h_kv):
    gs = h_q // h_kv
    return K.repeat_interleave(gs, dim=1), V.repeat_interleave(gs, dim=1)


# ---------- Generate inputs (BHND) ----------
mean = 10
std = 0.1
flops_ref = flops(B, N, H, D, causal, mode="bwd")


def generate_tensor(shape, mean, std, dtype, device):
    tensor = torch.randn(shape, dtype=dtype, device=device)
    magnitude = torch.norm(tensor, dim=-1, keepdim=True)
    scaled_tensor = tensor * (torch.randn(magnitude.shape, dtype=dtype, device=device) * std + mean) / magnitude
    return scaled_tensor.contiguous()


def generate_inputs():
    Q = generate_tensor((B, H, N, D), mean, std, torch.bfloat16, 'cuda')
    K = generate_tensor((B, H_KV, N, D), mean, std, torch.bfloat16, 'cuda')
    V = generate_tensor((B, H_KV, N, D), mean, std, torch.bfloat16, 'cuda')
    dO = generate_tensor((B, H, N, D), mean, std, torch.bfloat16, 'cuda')
    Q.requires_grad_(True); K.requires_grad_(True); V.requires_grad_(True)
    return Q, K, V, dO


Q_bhnd, K_bhnd, V_bhnd, dO_bhnd = generate_inputs()

# ---------- AITER reference (D=64) ----------
if use_aiter:
    timings = []
    print("\nRunning AITER (D=64)...")
    for _ in range(num_warmup):
        Q_aiter = Q_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)
        K_aiter = K_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)
        V_aiter = V_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)
        dO_aiter = dO_bhnd.transpose(1, 2).contiguous()
        out_aiter, softmax_lse = aiter.flash_attn_func(Q_aiter, K_aiter, V_aiter, causal=causal, return_lse=True, deterministic=False)
        out_aiter.backward(dO_aiter)

    for _ in range(num_iters):
        Q_aiter = Q_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)
        K_aiter = K_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)
        V_aiter = V_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)
        dO_aiter = dO_bhnd.transpose(1, 2).contiguous()
        out_aiter, softmax_lse = aiter.flash_attn_func(Q_aiter, K_aiter, V_aiter, causal=causal, return_lse=True, deterministic=False)
        torch.cuda.synchronize()
        start_event.record()
        out_aiter.backward(dO_aiter)
        end_event.record()
        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event))

    avg_time_aiter = sum(timings) / len(timings)
    eff_aiter = efficiency(flops_ref, avg_time_aiter)
    print(f"AITER bwd avg time: {avg_time_aiter:.4f} ms ({eff_aiter:.2f} TFLOPS) for B={B} H={H} H_KV={H_KV} N={N} D={D} Causal={causal}\n")

    q_grad_aiter_bnhd = Q_aiter.grad
    k_grad_aiter_bnhd = K_aiter.grad
    v_grad_aiter_bnhd = V_aiter.grad
    out_aiter_bnhd = out_aiter

# ---------- HipKittens (D=64) ----------
Q_tk = Q_bhnd.transpose(1, 2).bfloat16().clone().contiguous().detach().requires_grad_(True)
K_tk = K_bhnd.transpose(1, 2).bfloat16().clone().contiguous().detach().requires_grad_(True)
V_tk = V_bhnd.transpose(1, 2).bfloat16().clone().contiguous().detach().requires_grad_(True)
dO_tk = dO_bhnd.transpose(1, 2).bfloat16().clone().contiguous()

O_tk = torch.zeros_like(out_aiter_bnhd).bfloat16().clone().contiguous()
L_tk = torch.zeros((B, H, N, 1), device='cuda').float().transpose(-1, -2).contiguous()
tk_kernel_fwd.dispatch_fwd(Q_tk, K_tk, V_tk, O_tk, L_tk)

print("Running HipKittens D=64...")
timings = []
for _ in range(num_warmup):
    dQ_tk_in = torch.zeros_like(q_grad_aiter_bnhd).bfloat16().transpose(1, 2).contiguous()
    dQ_tk = torch.zeros_like(q_grad_aiter_bnhd).bfloat16().contiguous()
    dK_tk = torch.zeros_like(k_grad_aiter_bnhd).bfloat16().contiguous()
    dV_tk = torch.zeros_like(v_grad_aiter_bnhd).bfloat16().contiguous()
    delta_tk = torch.zeros((B, H, N, 1), device='cuda').float().transpose(-1, -2).contiguous()

    tk_kernel_bkwd_prep.dispatch_prep(O_tk, dO_tk, delta_tk)
    tk_kernel_bkwd.dispatch_bwd_combined(Q_tk, K_tk, V_tk, dO_tk, dQ_tk_in, dK_tk, dV_tk, L_tk, delta_tk)
    tk_kernel_bkwd_prep.dispatch_dq_shuffle(dQ_tk_in, dQ_tk)

for _ in range(num_iters):
    dQ_tk_in = torch.zeros_like(q_grad_aiter_bnhd).bfloat16().transpose(1, 2).contiguous()
    dQ_tk = torch.zeros_like(q_grad_aiter_bnhd).bfloat16().contiguous()
    dK_tk = torch.zeros_like(k_grad_aiter_bnhd).bfloat16().contiguous()
    dV_tk = torch.zeros_like(v_grad_aiter_bnhd).bfloat16().contiguous()
    delta_tk = torch.zeros((B, H, N, 1), device='cuda').float().transpose(-1, -2).contiguous()
    torch.cuda.synchronize()
    start_event.record()
    tk_kernel_bkwd_prep.dispatch_prep(O_tk, dO_tk, delta_tk)
    tk_kernel_bkwd.dispatch_bwd_combined(Q_tk, K_tk, V_tk, dO_tk, dQ_tk_in, dK_tk, dV_tk, L_tk, delta_tk)
    tk_kernel_bkwd_prep.dispatch_dq_shuffle(dQ_tk_in, dQ_tk)
    end_event.record()
    torch.cuda.synchronize()
    timings.append(start_event.elapsed_time(end_event))

avg_time_tk = sum(timings) / len(timings)
eff_tk = efficiency(flops_ref, avg_time_tk)
print(f"HK D=64 bwd avg time: {avg_time_tk:.4f} ms ({eff_tk:.2f} TFLOPS) for B={B} H={H} H_KV={H_KV} N={N} D={D} Causal={causal}\n")

# ---------- Compare ----------
print("Robustness (HK vs AITER):")
o_diff, _, _, o_rel, o_l2, o_cos, _ = robustness_check(O_tk, out_aiter_bnhd)
print(f"  O    : amax={o_diff.max():.6f}  rel_l2={o_l2:.4f}  cos={o_cos:.6f}")
q_diff, _, _, _, q_l2, q_cos, _ = robustness_check(q_grad_aiter_bnhd, dQ_tk)
k_diff, _, _, _, k_l2, k_cos, _ = robustness_check(k_grad_aiter_bnhd, dK_tk)
v_diff, _, _, _, v_l2, v_cos, _ = robustness_check(v_grad_aiter_bnhd, dV_tk)
print(f"  dQ   : amax={q_diff.max():.6f}  rel_l2={q_l2:.4f}  cos={q_cos:.6f}")
print(f"  dK   : amax={k_diff.max():.6f}  rel_l2={k_l2:.4f}  cos={k_cos:.6f}")
print(f"  dV   : amax={v_diff.max():.6f}  rel_l2={v_l2:.4f}  cos={v_cos:.6f}")

# Speedup
print(f"\nSpeedup HK vs AITER: {avg_time_aiter/avg_time_tk:.2f}x")
