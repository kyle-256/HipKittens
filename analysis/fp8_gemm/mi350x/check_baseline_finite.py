#!/usr/bin/env python3
"""Quick check: is the low finite_frac (0.3-0.7) seen in R13C candidates also seen
in the existing iterilp BASELINE wins? If yes, it's a benign property of large-K
random fp4 inputs (overflow to bf16 inf), not a candidate-specific issue."""
import json, math, os, subprocess, sys, sysconfig
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

CASES = [
    (14336, 4096, 32768, "_v16_wpe2_r10a_iterilp"),
    (4096, 32768, 28672, "_v20_memc_r11_iterilp"),
    (4096, 32768, 14336, "_ts_lgk2_memc_r11_iterilp"),
]

for (M, N, K, suf) in CASES:
    module = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suf}"
    so = os.path.join(BUILD_DIR, f"{module}{EXT_SUFFIX}")
    script = f"""
import math, torch, importlib.util
torch.manual_seed(0)
M, N, K = {M}, {N}, {K}
def gen_fp4(r, K):
    c = K // 2
    return (torch.randint(0,16,(r,c),dtype=torch.uint8,device='cuda')<<4)|torch.randint(0,16,(r,c),dtype=torch.uint8,device='cuda')
def preshuffle(se):
    r,kb=se.shape; pr=math.ceil(r/64)*64; pk=math.ceil(kb/8)*8
    raw=torch.full((pr,pk),0x7F,dtype=torch.uint8,device=se.device)
    raw[:r,:kb]=(se.to(torch.int16)+127).to(torch.uint8)
    sh=raw.view(pr//32,2,16,pk//8,2,4,1).permute(0,3,5,2,4,1,6).contiguous().view(pr//32,pk*32)
    sh=sh.view(pr//64,2,pk*32//4,4).permute(0,2,1,3).contiguous()
    return sh.view(pr//64,pk*64)
spec=importlib.util.spec_from_file_location('{module}','{so}')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
A=gen_fp4(M,K); B=gen_fp4(N,K)
sc_a=torch.randint(-2,3,(M,K//32),dtype=torch.int8,device='cuda')
sc_b=torch.randint(-2,3,(N,K//32),dtype=torch.int8,device='cuda')
A_sc=preshuffle(sc_a); B_sc=preshuffle(sc_b)
C=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
mod.gemm_rcr(A,B,A_sc,B_sc,C); torch.cuda.synchronize()
print('finite_frac', float(torch.isfinite(C.float()).float().mean().item()),
      'max_abs', float(C.float().abs().max().item()))
"""
    env = os.environ.copy(); env["HIP_VISIBLE_DEVICES"] = "7"
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)
    print(f"{M}x{N}x{K}  {suf}  ->  {r.stdout.strip()}", flush=True)
