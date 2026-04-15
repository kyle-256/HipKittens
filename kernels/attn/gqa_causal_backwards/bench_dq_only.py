#!/usr/bin/env python3
"""Benchmark ONLY the dQ kernel (attend_bwd_dq_d192v128_ker)."""
import torch
import sys
import os
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force fresh imports
for mod_name in list(sys.modules.keys()):
    if 'tk_kernel' in mod_name:
        del sys.modules[mod_name]

tk_kernel_fwd = importlib.import_module('tk_kernel_fwd')
tk_kernel_bkwd_prep = importlib.import_module('tk_kernel_bkwd_prep')
tk_kernel_bkwd = importlib.import_module('tk_kernel_bkwd')

B = int(os.environ.get('BENCH_B', '16'))
N = int(os.environ.get('BENCH_N', '4096'))
H = int(os.environ.get('BENCH_H', '64'))
H_KV = int(os.environ.get('BENCH_H_KV', '8'))
D_QK, D_V = 192, 128
dtype = torch.bfloat16

print(f"dQ-only Bench: B={B} N={N} H={H} H_KV={H_KV} D_QK={D_QK} D_V={D_V}")

Q = torch.randn(B, N, H, D_QK, dtype=dtype, device='cuda')
K = torch.randn(B, N, H_KV, D_QK, dtype=dtype, device='cuda')
V = torch.randn(B, N, H_KV, D_V, dtype=dtype, device='cuda')
dO = torch.randn(B, N, H, D_V, dtype=dtype, device='cuda')
O = torch.zeros(B, N, H, D_V, dtype=dtype, device='cuda')
L = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda')
delta = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda')

# Forward + prep
tk_kernel_fwd.dispatch_fwd(Q, K, V, O, L)
torch.cuda.synchronize()
tk_kernel_bkwd_prep.dispatch_prep(O, dO, delta)
torch.cuda.synchronize()

num_warmup = 20
num_iters = 20
s = torch.cuda.Event(enable_timing=True)
e = torch.cuda.Event(enable_timing=True)

# Warmup
for _ in range(num_warmup):
    dQ = torch.zeros(B, H, N, D_QK, dtype=dtype, device='cuda')
    tk_kernel_bkwd.dispatch_bwd_dq(Q, K, V, dO, dQ, L, delta)
    torch.cuda.synchronize()

# Benchmark
timings = []
for _ in range(num_iters):
    dQ = torch.zeros(B, H, N, D_QK, dtype=dtype, device='cuda')
    torch.cuda.synchronize()
    s.record()
    tk_kernel_bkwd.dispatch_bwd_dq(Q, K, V, dO, dQ, L, delta)
    e.record()
    torch.cuda.synchronize()
    timings.append(s.elapsed_time(e))

avg = sum(timings) / len(timings)
mn = min(timings)
mx = max(timings)
print(f"  dQ kernel: avg={avg:.4f} ms, min={mn:.4f} ms, max={mx:.4f} ms")
print(f"  dQ nonzero: {100*(dQ!=0).sum().item()/dQ.numel():.1f}%")
