#!/usr/bin/env python3
"""R13B single-shot bench: warmup=200, iters=500, trim=10% on GPU 6.
Bench original parent + the R13B baseline + each non-crashing R13B candidate.
"""
import json, math, os, subprocess, sys, sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10
GPU = 6
M, N, K = 28672, 4096, 16384
COMP = 5350.6  # aiter ASM competitor TFLOPS

# Each (parent_suffix, [variants])
GROUPS = [
    ("_ts_gm8", [
        "",  # original baseline (long-validated build)
        "_r13b_baseline",
        "_r13b_iterilp",
        # _r13b_maxilp crashes
        "_r13b_iterminreg",
        "_r13b_maxocc",
        "_r13b_iteroccexp",
        "_r13b_iterilp_stack_memc",
        "_r13b_memc_stack_iterilp",
    ]),
    ("_ts_lgk2_v12_memc", [
        "",  # original parent (long-built)
        "_r13b_baseline",
        "_r13b_iterilp",
        # _r13b_maxilp crashes
        "_r13b_iterminreg",
        "_r13b_maxocc",
        "_r13b_iteroccexp",
    ]),
]


def bench_one(full_suffix):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return {"error": "missing .so", "path": so_path}
    script = f"""
import sys, math, torch, importlib.util, json
torch.manual_seed(0)
WARMUP, ITERS, TRIM = {WARMUP}, {ITERS}, {TRIM}
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
spec=importlib.util.spec_from_file_location('{module_name}','{so_path}')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
A=gen_fp4(M,K); B=gen_fp4(N,K)
sc_a=torch.randint(-2,3,(M,K//32),dtype=torch.int8,device='cuda')
sc_b=torch.randint(-2,3,(N,K//32),dtype=torch.int8,device='cuda')
A_sc=preshuffle(sc_a); B_sc=preshuffle(sc_b)
C=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
run=lambda:mod.gemm_rcr(A,B,A_sc,B_sc,C)
for _ in range(WARMUP): run()
torch.cuda.synchronize()
times=[]
for _ in range(ITERS):
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record(); run(); e.record(); torch.cuda.synchronize()
    times.append(s.elapsed_time(e))
times.sort(); trim=int(len(times)*TRIM)
times=times[trim:-trim] if trim>0 else times
avg=sum(times)/len(times); t=2.0*M*N*K/(avg*1e-3)/1e12
print(json.dumps({{"tflops":round(t,2),"ms":round(avg,4)}}))
"""
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(GPU)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=900, env=env)
        if r.returncode != 0:
            return {"error": f"rc={r.returncode}", "stderr": r.stderr[-300:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": str(e)}


def main():
    print(f"R13B single-shot. shape={M}x{N}x{K} comp={COMP} GPU={GPU} warmup={WARMUP} iters={ITERS} trim={TRIM}")
    print("=" * 110)
    out = {"warmup": WARMUP, "iters": ITERS, "trim": TRIM, "gpu": GPU, "comp": COMP, "rows": []}
    for (psuf, variants) in GROUPS:
        print(f"\n--- Parent {psuf} ---")
        # Pick parent baseline TFLOPS for delta-pp computation: prefer _r13b_baseline (fresh same-build)
        baseline_tflops = None
        for vsuf in variants:
            full = psuf + vsuf
            r = bench_one(full)
            if "error" in r:
                print(f"  {full:50s}  ERROR: {r}", flush=True)
                out["rows"].append({"parent": psuf, "variant": vsuf, "full": full,
                                    "tflops": None, "ratio": None, "delta_pp": None,
                                    "error": str(r)})
                continue
            t = r["tflops"]
            ratio = t / COMP * 100
            if vsuf == "_r13b_baseline":
                baseline_tflops = t
            d_pp_str = ""
            if baseline_tflops is not None and vsuf.startswith("_r13b_") and vsuf != "_r13b_baseline":
                d_pp = (t - baseline_tflops) / COMP * 100
                d_pp_str = f"  delta={d_pp:+.2f}pp"
            print(f"  {full:50s}  {t:7.2f} TFLOPS  ({ratio:.2f}%){d_pp_str}", flush=True)
            out["rows"].append({"parent": psuf, "variant": vsuf, "full": full,
                                "tflops": t, "ratio": round(ratio, 2),
                                "delta_pp": round((t - baseline_tflops) / COMP * 100, 3)
                                              if baseline_tflops is not None and vsuf != "_r13b_baseline"
                                              else None})
    with open(os.path.join(SCRIPT_DIR, "bench_round13_optB_singleshot.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved bench_round13_optB_singleshot.json")


if __name__ == "__main__":
    main()
