#!/usr/bin/env python3
"""Quick sanity test for BWD dK/dV kernel."""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

B, N, H, H_KV = 16, 4096, 64, 8
D_QK, D_V = 192, 128
dtype = torch.bfloat16

print(f"Quick test: B={B} N={N} H={H} H_KV={H_KV}")

import tk_kernel_fwd, tk_kernel_bkwd_prep, tk_kernel_bkwd

Q = torch.randn(B, N, H, D_QK, dtype=dtype, device='cuda')
K = torch.randn(B, N, H_KV, D_QK, dtype=dtype, device='cuda')
V = torch.randn(B, N, H_KV, D_V, dtype=dtype, device='cuda')
dO = torch.randn(B, N, H, D_V, dtype=dtype, device='cuda')
O = torch.zeros(B, N, H, D_V, dtype=dtype, device='cuda')
L = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda')

tk_kernel_fwd.dispatch_fwd(Q, K, V, O, L)
torch.cuda.synchronize()

delta = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda')
tk_kernel_bkwd_prep.dispatch_prep(O, dO, delta)
torch.cuda.synchronize()

dQ_dummy = torch.zeros(B, H, N, D_QK, dtype=dtype, device='cuda')
dK = torch.zeros(B, N, H_KV, D_QK, dtype=dtype, device='cuda')
dV = torch.zeros(B, N, H_KV, D_V, dtype=dtype, device='cuda')

tk_kernel_bkwd.dispatch_bwd_combined(Q, K, V, dO, dQ_dummy, dK, dV, L, delta)
torch.cuda.synchronize()

print(f"dK abs max: {dK.abs().max().item():.6f}")
print(f"dV abs max: {dV.abs().max().item():.6f}")
print(f"dK nonzero: {(dK != 0).sum().item()}/{dK.numel()}")
print(f"dV nonzero: {(dV != 0).sum().item()}/{dV.numel()}")

if dK.abs().max().item() > 0 and dV.abs().max().item() > 0:
    print("PASS: dK and dV are non-zero")
else:
    print("FAIL: dK or dV is all zeros!")
