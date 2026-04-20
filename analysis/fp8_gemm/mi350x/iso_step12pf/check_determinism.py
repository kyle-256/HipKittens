#!/usr/bin/env python3
"""Check if base kernel produces deterministic output across two runs."""
import sys, math, torch, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "build"))
import tk_iso_base as mod

torch.manual_seed(0)
M, N, K = 16384, 4096, 2048
k_blocks = K // 32

def gen_fp4(rows, K):
    cols = K // 2
    lo = torch.randint(1, 4, (rows, cols), dtype=torch.uint8, device="cuda")
    hi = torch.randint(1, 4, (rows, cols), dtype=torch.uint8, device="cuda")
    return (hi << 4) | lo

def preshuffle_mfma16_merged(scale_exp):
    rows, kb = scale_exp.shape
    pr = math.ceil(rows / 64) * 64
    pk = math.ceil(kb / 8) * 8
    raw = torch.full((pr, pk), 0x7F, dtype=torch.uint8, device=scale_exp.device)
    raw[:rows, :kb] = (scale_exp.to(torch.int16) + 127).to(torch.uint8)
    sh = raw.view(pr // 32, 2, 16, pk // 8, 2, 4, 1)
    sh = sh.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    sh = sh.view(pr // 32, pk * 32)
    sh = sh.view(pr // 64, 2, pk * 32 // 4, 4)
    sh = sh.permute(0, 2, 1, 3).contiguous()
    return sh.view(pr // 64, pk * 64)

A = gen_fp4(M, K); B = gen_fp4(N, K)
sc_exp_a = torch.zeros(M, k_blocks, dtype=torch.int8, device="cuda")
sc_exp_b = torch.zeros(N, k_blocks, dtype=torch.int8, device="cuda")
A_sc = preshuffle_mfma16_merged(sc_exp_a)
B_sc = preshuffle_mfma16_merged(sc_exp_b)

C1 = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
mod.gemm_rcr(A, B, A_sc, B_sc, C1)
torch.cuda.synchronize()

C2 = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
mod.gemm_rcr(A, B, A_sc, B_sc, C2)
torch.cuda.synchronize()

eq = torch.equal(C1.view(torch.int16), C2.view(torch.int16))
diff = (C1.view(torch.int16) != C2.view(torch.int16)).sum().item()
print(f"Same kernel two runs equal: {eq}, diff_cells: {diff}/{C1.numel()}")
print(f"C1 inf: {torch.isinf(C1).sum().item()}  nan: {torch.isnan(C1).sum().item()}")
print(f"C2 inf: {torch.isinf(C2).sum().item()}  nan: {torch.isnan(C2).sum().item()}")
print(f"C1 sample [0,0..5]: {C1[0,:5].tolist()}")
print(f"C1 sample [100,100..105]: {C1[100,100:105].tolist()}")
print(f"C1 max finite: {C1[~torch.isinf(C1) & ~torch.isnan(C1)].float().abs().max().item()}")
print(f"C1 expected ~K*1.5*1.5 = {K*1.5*1.5}")

# Reference: torch matmul
A_f = torch.zeros(M, K, dtype=torch.float32, device="cuda")
B_f = torch.zeros(N, K, dtype=torch.float32, device="cuda")
# Decode fp4 to float (codes 1=0.5, 2=1.0, 3=1.5)
fp4_to_f = torch.tensor([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                          -0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                         dtype=torch.float32, device="cuda")
A_low  = A & 0xF
A_high = (A >> 4) & 0xF
A_f[:, 0::2] = fp4_to_f[A_low.long()]
A_f[:, 1::2] = fp4_to_f[A_high.long()]
B_low  = B & 0xF
B_high = (B >> 4) & 0xF
B_f[:, 0::2] = fp4_to_f[B_low.long()]
B_f[:, 1::2] = fp4_to_f[B_high.long()]
ref = (A_f @ B_f.T).to(torch.bfloat16)
print(f"Ref [0,0..5]: {ref[0,:5].tolist()}")
print(f"Ref [100,100..105]: {ref[100,100:105].tolist()}")
print(f"Ref max abs: {ref.float().abs().max().item()}")
