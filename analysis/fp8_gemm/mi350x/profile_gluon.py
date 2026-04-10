#!/usr/bin/env python3
"""Minimal profiling script for rocprof — runs kernel once."""
import torch, sys, os
sys.path.insert(0, os.path.dirname(__file__))

M = N = K = 8192
A = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device='cuda')
B = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device='cuda')
A_scale = torch.ones(M // 32, K // 128, 4, dtype=torch.uint8, device='cuda').contiguous()
B_scale = torch.ones(N // 32, K // 128, 4, dtype=torch.uint8, device='cuda').contiguous()
C = torch.zeros(M, N, dtype=torch.float32, device='cuda')

which = os.environ.get("PROFILE_KERNEL", "gluon_cpp")
if which == "gluon_cpp":
    import tk_mxfp4_gluon_cpp as mod
    mod.gemm_rcr(A, B, A_scale, B_scale, C)
elif which == "asm":
    import tk_mxfp4_asm_inline as mod
    mod.gemm_rcr(A, B, A_scale, B_scale, C)
torch.cuda.synchronize()
