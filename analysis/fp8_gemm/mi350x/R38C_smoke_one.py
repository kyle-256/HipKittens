#!/usr/bin/env python3
"""R38C single-shape smoke test on the canonical R37 CRASH case:
    m32768_n4096_k2048 ts_lgk2_gm6_v12_memc_pfoff4
Goal: 50 reps with no GPU fault, finite >= 0.995.
"""
import importlib.util, math, os, sys, sysconfig, json
import torch
torch.manual_seed(0)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
M, N, K = 32768, 4096, 2048
TAG = "ts_lgk2_gm6_v12_memc_pfoff4"
MOD = f"tk_mxfp4_gluon_cpp_n{N}_k{K}_{TAG}_R38C"
SO = os.path.join(SCRIPT_DIR, "build_R38C", f"{MOD}{EXT_SUFFIX}")

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 50

def gen_fp4(r, K):
    c = K // 2
    return (torch.randint(0,16,(r,c),dtype=torch.uint8,device='cuda') << 4) | \
           torch.randint(0,16,(r,c),dtype=torch.uint8,device='cuda')

def preshuffle(se):
    r, kb = se.shape
    pr = math.ceil(r/64)*64
    pk = math.ceil(kb/8)*8
    raw = torch.full((pr,pk),0x7F,dtype=torch.uint8,device=se.device)
    raw[:r,:kb] = (se.to(torch.int16)+127).to(torch.uint8)
    sh = raw.view(pr//32,2,16,pk//8,2,4,1).permute(0,3,5,2,4,1,6).contiguous().view(pr//32,pk*32)
    sh = sh.view(pr//64,2,pk*32//4,4).permute(0,2,1,3).contiguous()
    return sh.view(pr//64,pk*64)

print(f"Loading {MOD}", flush=True)
spec = importlib.util.spec_from_file_location(MOD, SO)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

A = gen_fp4(M, K); B = gen_fp4(N, K)
sc_a = torch.full((M, K//32), -4, dtype=torch.int8, device='cuda')
sc_b = torch.full((N, K//32), -4, dtype=torch.int8, device='cuda')
A_sc = preshuffle(sc_a); B_sc = preshuffle(sc_b)
C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

print(f"Running {REPS} reps shape={M}x{N}x{K} variant={TAG}", flush=True)
for i in range(REPS):
    C.zero_()
    mod.gemm_rcr(A, B, A_sc, B_sc, C)
    torch.cuda.synchronize()
    fin = float(torch.isfinite(C.float()).sum().item()) / float(C.numel())
    if i < 5 or i % 10 == 0 or i == REPS-1:
        print(f"  rep {i:3d}  finite={fin:.6f}", flush=True)
    if fin < 0.995:
        print(f"  WARN: rep {i} finite below gate ({fin:.6f})", flush=True)
print("DONE", flush=True)
