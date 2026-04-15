#!/usr/bin/env python3
"""Test script for ART backward kernel (D_QK=192, D_V=128).
Tests dV, dK, and dQ against float64 reference.
"""
import torch, math, sys, os
torch.manual_seed(42)

B = int(sys.argv[1]) if len(sys.argv) > 1 else 1
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
H = int(sys.argv[3]) if len(sys.argv) > 3 else 8
H_KV = int(sys.argv[4]) if len(sys.argv) > 4 else 1
D_QK, D_V = 192, 128
dtype = torch.bfloat16
gs = H // H_KV

print(f"ART Backward Test: B={B} N={N} H={H} H_KV={H_KV} D_QK={D_QK} D_V={D_V} gs={gs}")

# --- Generate inputs ---
mean, std = 10, 0.1
def gen(shape):
    t = torch.randn(shape, dtype=dtype, device='cuda')
    mag = torch.norm(t, dim=-1, keepdim=True)
    scale = (torch.randn(mag.shape, dtype=dtype, device='cuda') * std + mean) / mag
    return (t * scale).contiguous()

# BHND format
Q_b = gen((B, H, N, D_QK))
K_b = gen((B, H_KV, N, D_QK))
V_b = gen((B, H_KV, N, D_V))
dO_b = gen((B, H, N, D_V))

# --- Float64 reference ---
q_ = Q_b.detach().to(torch.float64).requires_grad_(True)
k_ = K_b.detach().to(torch.float64).requires_grad_(True)
v_ = V_b.detach().to(torch.float64).requires_grad_(True)
ke = k_.repeat_interleave(gs, dim=1)
ve = v_.repeat_interleave(gs, dim=1)
S = torch.matmul(q_, ke.transpose(-2, -1)) / math.sqrt(D_QK)
mask = torch.triu(torch.ones(N, N, device='cuda', dtype=torch.bool), diagonal=1)
S = S.masked_fill(mask, float('-inf'))
P = torch.softmax(S, dim=-1)
O = torch.matmul(P, ve)
L = torch.logsumexp(S, dim=-1)
O.backward(dO_b.detach().to(torch.float64))
# Clone so HIP kernels cannot clobber autograd backing storage for reference tensors.
dQ_ref = q_.grad.to(dtype).clone()  # BHND
dK_ref = k_.grad.to(dtype).clone()  # BH_KV ND
dV_ref = v_.grad.to(dtype).clone()  # BH_KV ND
print(f"Reference computed. dQ range: [{dQ_ref.float().min():.4f}, {dQ_ref.float().max():.4f}]")
print(f"  dK range: [{dK_ref.float().min():.4f}, {dK_ref.float().max():.4f}]")
print(f"  dV range: [{dV_ref.float().min():.4f}, {dV_ref.float().max():.4f}]")

# --- Prepare kernel inputs (BNHD layout) ---
Q_tk = Q_b.transpose(1, 2).contiguous()   # B,N,H,D_QK
K_tk = K_b.transpose(1, 2).contiguous()   # B,N,H_KV,D_QK
V_tk = V_b.transpose(1, 2).contiguous()   # B,N,H_KV,D_V
dO_tk = dO_b.transpose(1, 2).contiguous() # B,N,H,D_V

# --- Forward pass for L ---
import tk_kernel_fwd
O_tk = torch.zeros(B, N, H, D_V, dtype=dtype, device='cuda')
L_tk = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda')
tk_kernel_fwd.dispatch_fwd(Q_tk, K_tk, V_tk, O_tk, L_tk)
torch.cuda.synchronize()
print("Forward done.")

# Check O
O_ref_bnhd = O.detach().transpose(1, 2).to(dtype)
o_cos = torch.nn.functional.cosine_similarity(O_tk.float().flatten(), O_ref_bnhd.float().flatten(), dim=0).item()
print(f"O cos: {o_cos:.6f}")

# --- Backward prep for delta ---
import tk_kernel_bkwd_prep
delta_tk = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda')
tk_kernel_bkwd_prep.dispatch_prep(O_tk, dO_tk, delta_tk)
torch.cuda.synchronize()
print("Prep done.")

# --- Backward kernel ---
import tk_kernel_bkwd
dQ_tk_bhnd = torch.zeros(B, H, N, D_QK, dtype=dtype, device='cuda')  # atomic accumulation layout
dK_tk = torch.zeros(B, N, H_KV, D_QK, dtype=dtype, device='cuda')
dV_tk = torch.zeros(B, N, H_KV, D_V, dtype=dtype, device='cuda')

# Warmup
# opt1 combined kernel computes dK/dV only; dQ comes from dispatch_bwd_dq (BHND atomic layout).
for _ in range(3):
    dQ_tk_bhnd.zero_()
    dK_tk.zero_()
    dV_tk.zero_()
    tk_kernel_bkwd.dispatch_bwd_combined(Q_tk, K_tk, V_tk, dO_tk, dQ_tk_bhnd, dK_tk, dV_tk, L_tk, delta_tk)
    tk_kernel_bkwd.dispatch_bwd_dq(Q_tk, K_tk, V_tk, dO_tk, dQ_tk_bhnd, L_tk, delta_tk)
    torch.cuda.synchronize()

# Timed run (split: combined vs standalone dQ — matches opt1 split design)
dQ_tk_bhnd.zero_()
dK_tk.zero_()
dV_tk.zero_()
start_c = torch.cuda.Event(enable_timing=True)
end_c = torch.cuda.Event(enable_timing=True)
start_q = torch.cuda.Event(enable_timing=True)
end_q = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
start_c.record()
tk_kernel_bkwd.dispatch_bwd_combined(Q_tk, K_tk, V_tk, dO_tk, dQ_tk_bhnd, dK_tk, dV_tk, L_tk, delta_tk)
end_c.record()
start_q.record()
tk_kernel_bkwd.dispatch_bwd_dq(Q_tk, K_tk, V_tk, dO_tk, dQ_tk_bhnd, L_tk, delta_tk)
end_q.record()
torch.cuda.synchronize()
ms_c = start_c.elapsed_time(end_c)
ms_q = start_q.elapsed_time(end_q)
print(f"Backward kernel (dK/dV combined): {ms_c:.3f} ms")
print(f"Backward kernel (dQ standalone): {ms_q:.3f} ms")

# Convert dQ from BHND to BNHD for comparison
dQ_tk = dQ_tk_bhnd.transpose(1, 2).contiguous()

# --- Compare ---
def check(name, ref_bhnd, pred, is_bnhd=True):
    if is_bnhd:
        ref = ref_bhnd.transpose(1, 2).contiguous().float()
    else:
        ref = ref_bhnd.float()
    pred = pred.float()
    cos = torch.nn.functional.cosine_similarity(ref.flatten(), pred.flatten(), dim=0).item()
    diff = (ref - pred).abs()
    l2 = (diff.pow(2).sum().sqrt() / ref.pow(2).sum().sqrt()).item()
    print(f"  {name}: cos={cos:.6f}  l2={l2:.6f}  max_abs_diff={diff.max().item():.6f}  "
          f"ref_range=[{ref.min().item():.4f}, {ref.max().item():.4f}]  "
          f"pred_range=[{pred.min().item():.4f}, {pred.max().item():.4f}]")
    return cos

print("\n=== Gradient Comparison vs float64 ===")
dv_cos = check("dV", dV_ref, dV_tk)
dk_cos = check("dK", dK_ref, dK_tk)
dq_cos = check("dQ", dQ_ref, dQ_tk)

print(f"\n=== SUMMARY ===")
print(f"dV cos={dv_cos:.6f}  {'PASS' if dv_cos > 0.99 else 'FAIL'}")
print(f"dK cos={dk_cos:.6f}  {'PASS' if dk_cos > 0.99 else 'FAIL'}")
print(f"dQ cos={dq_cos:.6f}  {'PASS' if dq_cos > 0.99 else 'FAIL'}")
all_pass = dv_cos > 0.99 and dk_cos > 0.99 and dq_cos > 0.99
print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAIL'}")
