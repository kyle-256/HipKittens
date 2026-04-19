"""R43 Opt B pre-flight: verify R35A V2_vmcnt15_n8 produces ANY correct output.
We test at M=4096 N=32768 K=4096 (matches build shape) with random scales (mxfp4 realistic case).
If bit_eq vs incumbent < 50% across multiple seeds, the VGPR-PF deposit path is broken
and the entire R43 Opt B axis is DEAD.
"""
import os, sys, math, importlib.util
import torch

SCRIPT_DIR = "/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
sys.path.insert(0, BUILD_DIR)

VARIANT = "tk_mxfp4_gluon_cpp_n32768_k4096_R35A_v12_memc_btw_all_vmcnt15_n8"
LEGACY  = "tk_mxfp4_gluon_cpp_n32768_k4096_R35A_v12_memc_btw_all_legacyfork"

def gen_fp4(rows, K):
    cols = K // 2
    lo = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device="cuda")
    hi = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device="cuda")
    return (hi << 4) | lo

def preshuffle(scale_exp):
    rows, kb = scale_exp.shape
    pr = math.ceil(rows/64)*64
    pk = math.ceil(kb/8)*8
    raw = torch.full((pr,pk),0x7F,dtype=torch.uint8,device=scale_exp.device)
    raw[:rows,:kb] = (scale_exp.to(torch.int16)+127).to(torch.uint8)
    sh = raw.view(pr//32,2,16,pk//8,2,4,1).permute(0,3,5,2,4,1,6).contiguous().view(pr//32,pk*32)
    sh = sh.view(pr//64,2,pk*32//4,4).permute(0,2,1,3).contiguous()
    return sh.view(pr//64,pk*64)

def load_mod(name):
    so = os.path.join(BUILD_DIR, f"{name}.cpython-310-x86_64-linux-gnu.so")
    if not os.path.exists(so):
        print(f"MISSING: {so}"); return None
    spec = importlib.util.spec_from_file_location(name, so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    M, N, K = 4096, 32768, 4096
    torch.manual_seed(42)
    A = gen_fp4(M, K); B = gen_fp4(N, K)
    sc_a = torch.randint(-2, 3, (M, K//32), dtype=torch.int8, device='cuda')
    sc_b = torch.randint(-2, 3, (N, K//32), dtype=torch.int8, device='cuda')
    A_sc = preshuffle(sc_a); B_sc = preshuffle(sc_b)

    inc = load_mod(LEGACY)
    var = load_mod(VARIANT)
    if inc is None or var is None:
        print("MISSING_SO"); sys.exit(2)

    C_inc = torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
    inc.gemm_rcr(A,B,A_sc,B_sc,C_inc); torch.cuda.synchronize()
    C_var = torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
    var.gemm_rcr(A,B,A_sc,B_sc,C_var); torch.cuda.synchronize()

    inc_f = torch.isfinite(C_inc.float())
    var_f = torch.isfinite(C_var.float())
    both = inc_f & var_f
    bit_eq = (C_inc.view(torch.int16)[both] == C_var.view(torch.int16)[both]).float().mean().item()
    print(f"M={M} N={N} K={K}")
    print(f"  legacy finite_frac = {inc_f.float().mean().item()*100:.4f}%")
    print(f"  vgprPF V2_n8 finite_frac = {var_f.float().mean().item()*100:.4f}%")
    print(f"  both_finite_frac = {both.float().mean().item()*100:.4f}%")
    print(f"  bit_eq (on both finite) = {bit_eq*100:.4f}%")
    if bit_eq < 0.50:
        print("VERDICT: VGPR-PF deposit path CONFIRMED BROKEN (R35 finding holds)")
        print("RECOMMENDATION: KILL R43 Opt B axis; document and exit")
    elif bit_eq > 0.95:
        print("VERDICT: VGPR-PF V2_n8 produces matching output; further investigation justified")
    else:
        print(f"VERDICT: partial match {bit_eq*100:.2f}%; ambiguous")

if __name__ == "__main__":
    main()
