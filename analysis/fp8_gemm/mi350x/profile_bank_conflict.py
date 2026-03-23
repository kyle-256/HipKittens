import torch, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import tk_fp8_layouts as tk

M = N = K = 8192
A = torch.randn(M, K, device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
B = torch.randn(K, N, device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
C = torch.zeros(M, N, device='cuda', dtype=torch.bfloat16)

torch.cuda.synchronize()

# RCR: C = A @ B^T  (A row, B row for NT)
tk.gemm_rcr(A, B, C)
torch.cuda.synchronize()

# RRR: C = A @ B  (A row, B col)
tk.gemm_rrr(A, B, C)
torch.cuda.synchronize()

# CRR: C = A^T @ B  (A col, B col)
tk.gemm_crr(A, B, C)
torch.cuda.synchronize()
