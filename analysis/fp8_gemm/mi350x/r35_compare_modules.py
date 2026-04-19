#!/usr/bin/env python3
"""R35: compare non-finite rate across multiple modules at M=N=4096, K=2048."""
import os, sys, math, importlib.util
import torch
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR  = os.path.join(SCRIPT_DIR, "build_all42")
sys.path.insert(0, BUILD_DIR)

def gen_fp4(rows, K):
    cols = K // 2
    lo = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device='cuda')
    hi = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device='cuda')
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

def load_module(mod_name):
    so_path = os.path.join(BUILD_DIR, f"{mod_name}.cpython-310-x86_64-linux-gnu.so")
    if not os.path.exists(so_path):
        return None
    spec = importlib.util.spec_from_file_location(mod_name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mods = [
    "tk_mxfp4_gluon_cpp_n4096_k2048",
    "tk_mxfp4_gluon_cpp_n4096_k2048_ext_br",
    "tk_mxfp4_gluon_cpp_n4096_k2048_f34",
    "tk_mxfp4_gluon_cpp_n4096_k2048_v12",
    "tk_mxfp4_gluon_cpp_n4096_k2048_v4",
    "tk_mxfp4_gluon_cpp_n4096_k2048_lgk2",
    "tk_mxfp4_gluon_cpp_n4096_k2048_swap",
    "tk_mxfp4_gluon_cpp_n4096_k2048_swap_gm8",
    "tk_mxfp4_gluon_cpp_n4096_k2048_no_embed",
]

M, N, K = 4096, 4096, 2048
torch.manual_seed(42)
A = gen_fp4(M, K); B = gen_fp4(N, K)
sa = torch.full((M, K // 32), -4, dtype=torch.int8, device='cuda')
sb = torch.full((N, K // 32), -4, dtype=torch.int8, device='cuda')
A_sc = preshuffle_mfma16_merged(sa)
B_sc = preshuffle_mfma16_merged(sb)

print(f"{'module':<55} | nonfinite% | det% | upper-left128 nf% | other 75% nf%")
for mod_name in mods:
    mod = load_module(mod_name)
    if mod is None:
        print(f"{mod_name:<55} | MISSING")
        continue
    runs = []
    for _ in range(3):
        C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
        mod.gemm_rcr(A, B, A_sc, B_sc, C)
        torch.cuda.synchronize()
        runs.append(C.view(torch.int16).clone())
    stack = torch.stack(runs, 0)
    agree = (stack == stack[0:1]).all(dim=0)
    det = agree.float().mean().item()*100
    C0 = runs[0].view(torch.bfloat16).float()
    nf = ~torch.isfinite(C0)
    nf_total = nf.float().mean().item()*100
    # Upper-left 128x128 of every 256x256 tile
    M_t = M//256; N_t = N//256
    nf_t = nf.view(M_t, 256, N_t, 256).permute(0,2,1,3)
    upper_left = nf_t[:,:,:128,:128].float().mean().item()*100
    other = (nf_t.float().sum() - nf_t[:,:,:128,:128].float().sum()) / ((256*256 - 128*128) * M_t * N_t) * 100
    print(f"{mod_name:<55} | {nf_total:8.2f}%  | {det:5.1f}% | {upper_left:8.2f}%        | {other.item():.4f}%")
