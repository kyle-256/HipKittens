#!/usr/bin/env python3
"""Round 15 OptC correctness verification.

Cross-variant SNR check using SMALL scale range so bf16 doesn't overflow.
This is the SNR sanity gate — we compare each variant .so vs parent .so
on the SAME inputs, with scales in [-1, 1] to keep magnitudes finite.

A barrier reorder / prefetch reorder MUST yield bit-identical (or at most
floating-point-reduction-noise different) results vs the parent. SNR ≥ 50 dB
expected. SNR < 25 dB → REJECT (correctness regression).
"""
import os, sys, sysconfig, json, math, subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
PARENT_SUFFIX = "_ts_pf6_6_v12_memc"

SHAPES = {
    "DLA1": (4096, 32768, 128256),
    "DLA7": (28672, 32768, 4096),
}

VARIANTS = [
    ("_r15c_p1_lgkm0",     "DLA7"),
    ("_r15c_p1_lgkm2",     "DLA7"),
    ("_r15c_p1_lgkm4",     "DLA7"),
    ("_r15c_p2_s4v4",      "DLA1"),
    ("_r15c_p2_s4v8",      "DLA1"),
    ("_r15c_p2_s4v12",     "DLA1"),
    ("_r15c_p2_s4v16",     "DLA1"),
    ("_r15c_p2_s4v20",     "DLA1"),
    ("_r15c_p2_s4v4",      "DLA7"),
    ("_r15c_p2_s4v8",      "DLA7"),
    ("_r15c_p2_s4v12",     "DLA7"),
    ("_r15c_p2_s4v16",     "DLA7"),
    ("_r15c_p2_s4v20",     "DLA7"),
    ("_r15c_p3_pfg_neg1",  "DLA1"),
    ("_r15c_p3_pfg_pos1",  "DLA1"),
]


def script(M, N, K, suffix):
    parent_so = os.path.join(BUILD_DIR,
        f"tk_mxfp4_gluon_cpp_n{N}_k{K}{PARENT_SUFFIX}{EXT_SUFFIX}")
    variant_so = os.path.join(BUILD_DIR,
        f"tk_mxfp4_gluon_cpp_n{N}_k{K}{PARENT_SUFFIX}{suffix}{EXT_SUFFIX}")
    return f"""
import sys, math, torch, importlib.util, json
torch.manual_seed(42)
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
A=gen_fp4(M,K); B=gen_fp4(N,K)
# Use VERY tight scales (all 0 = 2^0 = 1.0) to avoid overflow at large K
sc_a=torch.zeros((M,K//32),dtype=torch.int8,device='cuda')
sc_b=torch.zeros((N,K//32),dtype=torch.int8,device='cuda')
# Also reduce fp4 input range: only use values 0..3 (low magnitudes)
A=(torch.randint(0,4,(M,K//2),dtype=torch.uint8,device='cuda')<<4)|torch.randint(0,4,(M,K//2),dtype=torch.uint8,device='cuda')
B=(torch.randint(0,4,(N,K//2),dtype=torch.uint8,device='cuda')<<4)|torch.randint(0,4,(N,K//2),dtype=torch.uint8,device='cuda')
A_sc=preshuffle(sc_a); B_sc=preshuffle(sc_b)
parent_module = 'tk_mxfp4_gluon_cpp_n{N}_k{K}{PARENT_SUFFIX}'
variant_module = 'tk_mxfp4_gluon_cpp_n{N}_k{K}{PARENT_SUFFIX}{suffix}'
spec_p=importlib.util.spec_from_file_location(parent_module,'{parent_so}')
mod_p=importlib.util.module_from_spec(spec_p); spec_p.loader.exec_module(mod_p)
spec_v=importlib.util.spec_from_file_location(variant_module,'{variant_so}')
mod_v=importlib.util.module_from_spec(spec_v); spec_v.loader.exec_module(mod_v)
C_p=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
C_v=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
mod_p.gemm_rcr(A,B,A_sc,B_sc,C_p); torch.cuda.synchronize()
mod_v.gemm_rcr(A,B,A_sc,B_sc,C_v); torch.cuda.synchronize()
fp = float(torch.isfinite(C_p.float()).float().mean().item())
fv = float(torch.isfinite(C_v.float()).float().mean().item())
# Compute SNR only over the region where BOTH are finite — robust to overflow saturation.
mask = torch.isfinite(C_p.float()) & torch.isfinite(C_v.float())
mask_frac = float(mask.float().mean().item())
import math as _m
if mask.sum().item() < 100:
    snr = float('nan')
else:
    cp = C_p.float()[mask]; cv = C_v.float()[mask]
    diff = cp - cv
    sig = (cp*cp).mean().item()
    err = (diff*diff).mean().item()
    if sig <= 0:
        snr = float('nan')
    elif err <= 0:
        snr = float('inf')
    else:
        snr = 10.0 * _m.log10(sig/err)
# also check that nan positions agree (control flow signal)
nan_match = float((torch.isfinite(C_p.float()) == torch.isfinite(C_v.float())).float().mean().item())
exact = bool(torch.equal(C_p, C_v))
print(json.dumps({{"snr_db": snr, "finite_p": fp, "finite_v": fv, "mask_frac": mask_frac, "nan_match": nan_match, "exact": exact}}))
"""


def run_one(M, N, K, suffix, gpu):
    s = script(M, N, K, suffix)
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu)
    try:
        r = subprocess.run([sys.executable, "-c", s], capture_output=True, text=True, timeout=400, env=env)
        if r.returncode != 0:
            return {"error": r.stderr[-300:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": str(e)}


def main():
    print("Round 15 OptC verify (parent vs variant SNR)")
    print("=" * 130)
    out = {"results": []}
    for i, (suffix, sh) in enumerate(VARIANTS):
        M, N, K = SHAPES[sh]
        gpu = 6 + (i % 2)
        r = run_one(M, N, K, suffix, gpu)
        tag = f"{sh} {suffix}"
        if "error" in r:
            print(f"  {tag:55s} ERROR  {r['error'][:100]}", flush=True)
        else:
            snr = r["snr_db"]
            ok = "PASS" if (isinstance(snr,(int,float)) and not math.isnan(snr) and snr >= 25.0) else "FAIL"
            ok2 = "EXACT" if r["exact"] else "diff"
            print(f"  {tag:55s} snr={snr:.2f}dB  fin_p={r['finite_p']:.3f} fin_v={r['finite_v']:.3f}  {ok2}  [{ok}]", flush=True)
        out["results"].append({"suffix": suffix, "shape": sh, **r})
    with open(os.path.join(SCRIPT_DIR, "bench_round15_optC_verify.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved bench_round15_optC_verify.json")


if __name__ == "__main__":
    main()
