#!/usr/bin/env python3
"""Diagnostic: check dQ output for NaN, inf, and pattern analysis."""
import torch
import math
import sys

torch.manual_seed(0)

B, N, H, H_KV = 2, 1024, 64, 8
D_QK, D_V = 192, 128
causal = 1
dtype = torch.bfloat16
group_size = H // H_KV

print(f"Diagnostic: {B=} {N=} {H=} {H_KV=} {D_QK=} {D_V=} {causal=}")

# Generate simple inputs
Q = torch.randn(B, N, H, D_QK, dtype=dtype, device='cuda') * 0.1
K = torch.randn(B, N, H_KV, D_QK, dtype=dtype, device='cuda') * 0.1
V = torch.randn(B, N, H_KV, D_V, dtype=dtype, device='cuda') * 0.1
dO = torch.randn(B, N, H, D_V, dtype=dtype, device='cuda') * 0.1

import tk_kernel_fwd, tk_kernel_bkwd_prep, tk_kernel_bkwd

# Forward
O = torch.zeros(B, N, H, D_V, dtype=dtype, device='cuda')
L = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda')
tk_kernel_fwd.dispatch_fwd(Q, K, V, O, L)
torch.cuda.synchronize()

# Prep
delta = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda')
tk_kernel_bkwd_prep.dispatch_prep(O, dO, delta)
torch.cuda.synchronize()

# Backward - just dQ
dQ_in = torch.zeros(B, H, N, D_QK, dtype=dtype, device='cuda')
tk_kernel_bkwd.dispatch_bwd_dq(Q, K, V, dO, dQ_in, L, delta)
torch.cuda.synchronize()

print(f"\ndQ_in shape: {dQ_in.shape}")
print(f"dQ_in NaN count: {torch.isnan(dQ_in.float()).sum().item()}")
print(f"dQ_in Inf count: {torch.isinf(dQ_in.float()).sum().item()}")
print(f"dQ_in zero count: {(dQ_in == 0).sum().item()}")
print(f"dQ_in total elements: {dQ_in.numel()}")

finite = dQ_in.float()[~torch.isnan(dQ_in.float()) & ~torch.isinf(dQ_in.float())]
print(f"dQ_in finite count: {finite.numel()}")
if finite.numel() > 0:
    print(f"dQ_in finite min: {finite.min().item():.6f}")
    print(f"dQ_in finite max: {finite.max().item():.6f}")
    print(f"dQ_in finite mean: {finite.mean().item():.6f}")
    print(f"dQ_in finite std: {finite.std().item():.6f}")

# Check which positions have NaN
nan_mask = torch.isnan(dQ_in.float())
if nan_mask.any():
    # Find first NaN position
    nz = nan_mask.nonzero()
    print(f"\nFirst 10 NaN positions (b, h, n, d):")
    for i in range(min(10, nz.shape[0])):
        print(f"  {nz[i].tolist()}")

    # Check NaN distribution by dimension
    for dim_name, dim in [("batch", 0), ("head", 1), ("seq", 2), ("d", 3)]:
        nan_per = nan_mask.any(dim=tuple(d for d in range(4) if d != dim)).sum().item()
        total = dQ_in.shape[dim]
        print(f"  {dim_name}: {nan_per}/{total} dims have NaN")

# Now let's check: is the shuffle kernel the problem?
# Try writing known values to dQ_in and see what shuffle produces
dQ_test = torch.arange(D_QK, dtype=dtype, device='cuda').unsqueeze(0).unsqueeze(0).unsqueeze(0)
dQ_test = dQ_test.expand(B, H, N, D_QK).contiguous()
dQ_out = torch.zeros(B, N, H, D_QK, dtype=dtype, device='cuda')
tk_kernel_bkwd_prep.dispatch_dq_shuffle(dQ_test, dQ_out)
torch.cuda.synchronize()

# Check if shuffle is identity-like
print(f"\nShuffle test: dQ_test[0,0,0,:10] = {dQ_test[0,0,0,:10]}")
print(f"Shuffle test: dQ_out[0,0,0,:10] = {dQ_out[0,0,0,:10]}")

expected = torch.arange(D_QK, dtype=dtype, device='cuda')
match = (dQ_out[0, 0, 0, :] == expected).all().item()
print(f"Shuffle preserves values: {match}")

# Check the actual dQ through the full pipeline
dQ_in2 = torch.zeros(B, H, N, D_QK, dtype=dtype, device='cuda')
dQ_out2 = torch.zeros(B, N, H, D_QK, dtype=dtype, device='cuda')
tk_kernel_bkwd.dispatch_bwd_dq(Q, K, V, dO, dQ_in2, L, delta)
torch.cuda.synchronize()
tk_kernel_bkwd_prep.dispatch_dq_shuffle(dQ_in2, dQ_out2)
torch.cuda.synchronize()

print(f"\nFull pipeline dQ_out2 NaN count: {torch.isnan(dQ_out2.float()).sum().item()}")
print(f"Full pipeline dQ_out2 Inf count: {torch.isinf(dQ_out2.float()).sum().item()}")

# Check a small slice of the raw dQ_in (before shuffle)
print(f"\ndQ_in2[0,0,0,:10] = {dQ_in2[0,0,0,:10]}")
print(f"dQ_in2[0,0,0,180:192] = {dQ_in2[0,0,0,180:192]}")
print(f"dQ_in2[0,0,1,:10] = {dQ_in2[0,0,1,:10]}")
