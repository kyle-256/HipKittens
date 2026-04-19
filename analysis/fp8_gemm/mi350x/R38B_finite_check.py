#!/usr/bin/env python3
"""Check finite-fraction (single-rep) and 50-rep stability for R37 vs R38B builds
on the test shape.
"""
import importlib.util, math, os, sys, sysconfig
import torch
torch.manual_seed(0)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"

M, N, K = 32768, 4096, 2048
PARENT_TAG = "ts_lgk2_gm6_v12_memc_pfoff4"

WHICH = sys.argv[1] if len(sys.argv) > 1 else "R38B"
BUILD = "build_R38B" if WHICH == "R38B" else "build_R37"
SUFFIX = "_R38B" if WHICH == "R38B" else "_R37"

module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}_{PARENT_TAG}{SUFFIX}"
so_path = os.path.join(SCRIPT_DIR, BUILD, f"{module_name}{EXT_SUFFIX}")
print(f"Loading {WHICH}: {so_path}")
assert os.path.exists(so_path)
spec = importlib.util.spec_from_file_location(module_name, so_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def gen_fp4(r, K):
    c = K // 2
    return (torch.randint(0, 16, (r, c), dtype=torch.uint8, device='cuda') << 4) | \
           torch.randint(0, 16, (r, c), dtype=torch.uint8, device='cuda')


def preshuffle(se):
    r, kb = se.shape
    pr = math.ceil(r / 64) * 64
    pk = math.ceil(kb / 8) * 8
    raw = torch.full((pr, pk), 0x7F, dtype=torch.uint8, device=se.device)
    raw[:r, :kb] = (se.to(torch.int16) + 127).to(torch.uint8)
    sh = raw.view(pr // 32, 2, 16, pk // 8, 2, 4, 1).permute(0, 3, 5, 2, 4, 1, 6).contiguous().view(pr // 32, pk * 32)
    sh = sh.view(pr // 64, 2, pk * 32 // 4, 4).permute(0, 2, 1, 3).contiguous()
    return sh.view(pr // 64, pk * 64)


A = gen_fp4(M, K)
B = gen_fp4(N, K)

# correctness check (all-128 scales)
sc_a_corr = torch.full((M, K // 32), -4, dtype=torch.int8, device='cuda')
sc_b_corr = torch.full((N, K // 32), -4, dtype=torch.int8, device='cuda')
A_sc_corr = preshuffle(sc_a_corr)
B_sc_corr = preshuffle(sc_b_corr)

C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

# Run 50 reps and report finite-frac per rep
for rep in range(10):
    C.zero_()
    mod.gemm_rcr(A, B, A_sc_corr, B_sc_corr, C)
    torch.cuda.synchronize()
    f = float(torch.isfinite(C.float()).sum().item()) / float(C.numel())
    print(f"  rep{rep:2d}  finite={f:.6f}")
