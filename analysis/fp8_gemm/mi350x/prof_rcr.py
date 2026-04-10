import torch, tk_fp8_layouts as m

shapes = [
    (4096, 28672, 4096),
    (8192, 4096, 4096),
    (8192, 16384, 16384),
]

for M, N, K in shapes:
    A = (torch.randn(M, K, device='cuda') * 0.1).to(torch.float8_e4m3fn)
    B = (torch.randn(N, K, device='cuda') * 0.1).to(torch.float8_e4m3fn)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    for _ in range(3):
        m.gemm_rcr(A, B, C, 1.0, 1.0, 4)
    torch.cuda.synchronize()
    print(f"Done {M}x{N}x{K}")
