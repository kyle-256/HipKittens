#!/usr/bin/env python3
"""R13B aperture smoke: low-cost run+SNR+finite check to weed out crashing/NaN variants.
Warmup=20, iters=50, 3 retries. GPU 6.
"""
import json, math, os, subprocess, sys, sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP, ITERS = 20, 50
GPU = 6
M, N, K = 28672, 4096, 16384

PARENTS = ["_ts_gm8", "_ts_lgk2_v12_memc"]
VARIANTS = [
    "_r13b_baseline",
    "_r13b_iterilp",
    "_r13b_maxilp",
    "_r13b_iterminreg",
    "_r13b_maxocc",
    "_r13b_iteroccexp",
]
STACK = ["_r13b_iterilp_stack_memc", "_r13b_memc_stack_iterilp"]


def smoke_one(full_suffix, retries=3):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return {"error": "missing .so"}
    script = f"""
import sys, math, torch, importlib.util, json
torch.manual_seed(0)
WARMUP, ITERS = {WARMUP}, {ITERS}
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
run(); torch.cuda.synchronize()
finite_frac = float(torch.isfinite(C.float()).float().mean().item())
for _ in range(WARMUP): run()
torch.cuda.synchronize()
times=[]
for _ in range(ITERS):
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record(); run(); e.record(); torch.cuda.synchronize()
    times.append(s.elapsed_time(e))
times.sort(); avg=sum(times)/len(times); t=2.0*M*N*K/(avg*1e-3)/1e12
print(json.dumps({{"tflops":round(t,2),"ms":round(avg,4),"finite_frac":finite_frac}}))
"""
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(GPU)
    last_err = None
    for attempt in range(retries):
        try:
            r = subprocess.run([sys.executable, "-c", script],
                               capture_output=True, text=True, timeout=300, env=env)
            if r.returncode != 0:
                last_err = {"error": f"rc={r.returncode}", "stderr": r.stderr[-300:]}
                continue
            line = r.stdout.strip().splitlines()[-1]
            return json.loads(line)
        except Exception as e:
            last_err = {"error": str(e)}
    return last_err or {"error": "unknown"}


def main():
    print(f"R13B smoke. GPU={GPU} warmup={WARMUP} iters={ITERS} (3 retries)")
    print("=" * 110)
    out = {"warmup": WARMUP, "iters": ITERS, "gpu": GPU, "rows": []}
    for psuf in PARENTS:
        print(f"\nParent: {psuf}")
        for vsuf in VARIANTS:
            full = psuf + vsuf
            r = smoke_one(full)
            tag = "OK" if "error" not in r else "FAIL"
            ff = r.get("finite_frac")
            ff_str = "" if ff is None else f"  finite={ff:.3f}"
            tflops = r.get("tflops")
            tf_str = "" if tflops is None else f"  {tflops:.2f} TFLOPS"
            err_str = "" if "error" not in r else f"  err={r}"
            print(f"  {full:50s} {tag:6s}{tf_str}{ff_str}{err_str}", flush=True)
            out["rows"].append({"parent": psuf, "variant": vsuf, "full": full,
                                "result": r, "status": tag})
        if psuf == "_ts_gm8":
            for vsuf in STACK:
                full = psuf + vsuf
                r = smoke_one(full)
                tag = "OK" if "error" not in r else "FAIL"
                ff = r.get("finite_frac")
                ff_str = "" if ff is None else f"  finite={ff:.3f}"
                tflops = r.get("tflops")
                tf_str = "" if tflops is None else f"  {tflops:.2f} TFLOPS"
                err_str = "" if "error" not in r else f"  err={r}"
                print(f"  {full:50s} {tag:6s}{tf_str}{ff_str}{err_str}", flush=True)
                out["rows"].append({"parent": psuf, "variant": vsuf, "full": full,
                                    "result": r, "status": tag})
    with open(os.path.join(SCRIPT_DIR, "bench_round13_optB_smoke.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved bench_round13_optB_smoke.json")


if __name__ == "__main__":
    main()
