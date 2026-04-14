#!/usr/bin/env python3
"""Diagnostic: check if the dQ kernel causes memory faults."""
import torch
import math

torch.manual_seed(0)

B, N, H, H_KV = 2, 1024, 64, 8
D_QK, D_V = 192, 128
dtype = torch.bfloat16

print(f"Diagnostic: {B=} {N=} {H=} {H_KV=} {D_QK=} {D_V=}")

mean, std_ = 10, 0.1
def gen(shape):
    t = torch.randn(shape, dtype=dtype, device='cuda')
    mag = torch.norm(t, dim=-1, keepdim=True)
    return (t * (torch.randn(mag.shape, dtype=dtype, device='cuda') * std_ + mean) / mag).contiguous()

# BNHD layout
Q = gen((B, N, H, D_QK))
K = gen((B, N, H_KV, D_QK))
V = gen((B, N, H_KV, D_V))
dO = gen((B, N, H, D_V))

import tk_kernel_fwd, tk_kernel_bkwd_prep, tk_kernel_bkwd

# Forward
O = torch.zeros(B, N, H, D_V, dtype=dtype, device='cuda')
L = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda')
tk_kernel_fwd.dispatch_fwd(Q, K, V, O, L)
torch.cuda.synchronize()
print("Forward OK")

# Prep
delta = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda')
tk_kernel_bkwd_prep.dispatch_prep(O, dO, delta)
torch.cuda.synchronize()
print("Prep OK")

# Combined backward (dK, dV only)
dQ_throw = torch.zeros(B, H, N, D_QK, dtype=dtype, device='cuda')
dK = torch.zeros(B, N, H_KV, D_QK, dtype=dtype, device='cuda')
dV = torch.zeros(B, N, H_KV, D_V, dtype=dtype, device='cuda')
tk_kernel_bkwd.dispatch_bwd_combined(Q, K, V, dO, dQ_throw, dK, dV, L, delta)
torch.cuda.synchronize()
print("Combined backward OK (dK, dV)")

# Separate dQ kernel
dQ_in = torch.zeros(B, H, N, D_QK, dtype=dtype, device='cuda')
print(f"dQ_in device: {dQ_in.device}, contiguous: {dQ_in.is_contiguous()}")
print(f"dQ_in shape: {dQ_in.shape}, stride: {dQ_in.stride()}")
print(f"dQ_in data_ptr: {dQ_in.data_ptr()}")
print("Launching dQ kernel...")
tk_kernel_bkwd.dispatch_bwd_dq(Q, K, V, dO, dQ_in, L, delta)
torch.cuda.synchronize()
print("dQ kernel OK")

nan_count = torch.isnan(dQ_in.float()).sum().item()
inf_count = torch.isinf(dQ_in.float()).sum().item()
print(f"dQ NaN: {nan_count}, Inf: {inf_count}, Total: {dQ_in.numel()}")

# Look at dQ_in values
print(f"dQ_in[0,0,0,:10] = {dQ_in[0,0,0,:10]}")
print(f"dQ_in[0,0,1,:10] = {dQ_in[0,0,1,:10]}")
print(f"dQ_in[0,0,0,180:] = {dQ_in[0,0,0,180:]}")
