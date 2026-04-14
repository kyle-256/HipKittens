#!/usr/bin/env python3
"""Simple BWD benchmark for D_QK=192, D_V=128 — measures just the dK/dV kernel."""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

B = int(sys.argv[1]) if len(sys.argv) > 1 else 16
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
H = int(sys.argv[3]) if len(sys.argv) > 3 else 64
H_KV = int(sys.argv[4]) if len(sys.argv) > 4 else 8
D_QK = 192
D_V = 128
causal = 1
dtype = torch.bfloat16

# Full BWD flops = 2.5 * FWD, where FWD = 2*B*N^2*H*(D_QK+D_V)/2
fwd_flops = 2 * B * N**2 * H * (D_QK + D_V) // 2
bwd_flops = int(2.5 * fwd_flops)
# Combined kernel (dK/dV only, no dQ) = 2 * FWD
combined_flops = 2 * fwd_flops

print(f"BWD Bench: B={B} N={N} H={H} H_KV={H_KV} D_QK={D_QK} D_V={D_V}")
print(f"  FWD FLOPs: {fwd_flops/1e12:.3f} T, Combined FLOPs: {combined_flops/1e12:.3f} T, Full BWD: {bwd_flops/1e12:.3f} T")

import tk_kernel_fwd
import tk_kernel_bkwd_prep
import tk_kernel_bkwd

num_warmup = 20
num_iters = 10
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

Q = torch.randn(B, N, H, D_QK, dtype=dtype, device='cuda')
K = torch.randn(B, N, H_KV, D_QK, dtype=dtype, device='cuda')
V = torch.randn(B, N, H_KV, D_V, dtype=dtype, device='cuda')
dO = torch.randn(B, N, H, D_V, dtype=dtype, device='cuda')
O = torch.zeros(B, N, H, D_V, dtype=dtype, device='cuda')
L = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda')

# Forward
tk_kernel_fwd.dispatch_fwd(Q, K, V, O, L)
torch.cuda.synchronize()

# Prep
delta = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda')
tk_kernel_bkwd_prep.dispatch_prep(O, dO, delta)
torch.cuda.synchronize()

# Benchmark combined (dK/dV) only — NO dQ
dQ_dummy = torch.zeros(B, H, N, D_QK, dtype=dtype, device='cuda')

for _ in range(num_warmup):
    dK = torch.zeros(B, N, H_KV, D_QK, dtype=dtype, device='cuda')
    dV = torch.zeros(B, N, H_KV, D_V, dtype=dtype, device='cuda')
    tk_kernel_bkwd.dispatch_bwd_combined(Q, K, V, dO, dQ_dummy, dK, dV, L, delta)
    torch.cuda.synchronize()

timings_combined = []
for _ in range(num_iters):
    dK = torch.zeros(B, N, H_KV, D_QK, dtype=dtype, device='cuda')
    dV = torch.zeros(B, N, H_KV, D_V, dtype=dtype, device='cuda')
    torch.cuda.synchronize()
    start_event.record()
    tk_kernel_bkwd.dispatch_bwd_combined(Q, K, V, dO, dQ_dummy, dK, dV, L, delta)
    end_event.record()
    torch.cuda.synchronize()
    timings_combined.append(start_event.elapsed_time(end_event))

avg_combined = sum(timings_combined) / len(timings_combined)
eff_combined = (combined_flops / 1e12) / (avg_combined / 1e3)
eff_bwd = (bwd_flops / 1e12) / (avg_combined / 1e3)
print(f"  dK/dV kernel: {avg_combined:.4f} ms")
print(f"    Combined TFLOPS (2x fwd):   {eff_combined:.2f}")
print(f"    Equiv BWD TFLOPS (2.5x fwd): {eff_bwd:.2f}")
