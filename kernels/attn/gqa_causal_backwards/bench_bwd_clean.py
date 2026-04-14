#!/usr/bin/env python3
"""BWD benchmark with guaranteed fresh module imports."""
import torch
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force fresh imports
for mod_name in list(sys.modules.keys()):
    if 'tk_kernel' in mod_name:
        del sys.modules[mod_name]

import importlib
tk_kernel_fwd = importlib.import_module('tk_kernel_fwd')
tk_kernel_bkwd_prep = importlib.import_module('tk_kernel_bkwd_prep')
tk_kernel_bkwd = importlib.import_module('tk_kernel_bkwd')

B = int(os.environ.get('BENCH_B', '16'))
N = int(os.environ.get('BENCH_N', '4096'))
H = int(os.environ.get('BENCH_H', '64'))
H_KV = int(os.environ.get('BENCH_H_KV', '8'))
D_QK, D_V = 192, 128
dtype = torch.bfloat16

fwd_flops = 2 * B * N**2 * H * (D_QK + D_V) // 2
bwd_flops = int(2.5 * fwd_flops)

print(f"BWD Bench: B={B} N={N} H={H} H_KV={H_KV}")

Q = torch.randn(B, N, H, D_QK, dtype=dtype, device='cuda')
K = torch.randn(B, N, H_KV, D_QK, dtype=dtype, device='cuda')
V = torch.randn(B, N, H_KV, D_V, dtype=dtype, device='cuda')
dO = torch.randn(B, N, H, D_V, dtype=dtype, device='cuda')
O = torch.zeros(B, N, H, D_V, dtype=dtype, device='cuda')
L = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda')
delta = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda')
dQ_dummy = torch.zeros(B, H, N, D_QK, dtype=dtype, device='cuda')

tk_kernel_fwd.dispatch_fwd(Q, K, V, O, L)
torch.cuda.synchronize()
tk_kernel_bkwd_prep.dispatch_prep(O, dO, delta)
torch.cuda.synchronize()

num_warmup = 10
num_iters = 5
s = torch.cuda.Event(enable_timing=True)
e = torch.cuda.Event(enable_timing=True)

for _ in range(num_warmup):
    dK = torch.zeros(B, N, H_KV, D_QK, dtype=dtype, device='cuda')
    dV = torch.zeros(B, N, H_KV, D_V, dtype=dtype, device='cuda')
    tk_kernel_bkwd.dispatch_bwd_combined(Q, K, V, dO, dQ_dummy, dK, dV, L, delta)
    torch.cuda.synchronize()

timings = []
for _ in range(num_iters):
    dK = torch.zeros(B, N, H_KV, D_QK, dtype=dtype, device='cuda')
    dV = torch.zeros(B, N, H_KV, D_V, dtype=dtype, device='cuda')
    torch.cuda.synchronize()
    s.record()
    tk_kernel_bkwd.dispatch_bwd_combined(Q, K, V, dO, dQ_dummy, dK, dV, L, delta)
    e.record()
    torch.cuda.synchronize()
    timings.append(s.elapsed_time(e))

avg = sum(timings) / len(timings)
eff = bwd_flops / 1e12 / (avg / 1e3)
pct = 100 * (dK != 0).sum().item() / dK.numel()
print(f"  avg={avg:.4f} ms, bwd_equiv={eff:.2f} TFLOPS")
print(f"  dK nonzero: {pct:.1f}%, dK max: {dK.abs().max().item():.4f}")
