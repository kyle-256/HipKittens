#!/usr/bin/env python3
"""R17C verify: 5-run replication for smoke-PASS candidates.
Single GPU per run to eliminate cross-GPU noise. Gate: mean_cand >= max(base_runs)
AND (mean_cand - mean_base)/comp*100 >= +1.0pp.
"""
import json, os, subprocess, sys, sysconfig, statistics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200; ITERS = 500; TRIM = 0.10
N_RUNS = 5
GPU = 5
NEW_PREFIX = "_r17c_"

SHAPE_INFO = {
    "DLA1": (4096,  32768, 128256, 5781.1, "_ts_pf6_6_v12_memc"),
    "DLA2": (128256, 32768,  4096, 4536.4, "_ts_gm2_v12_memc_dc"),
    "DLA7": (28672, 32768,  4096, 4466.6, "_ts_lgk2_v12_memc"),
    "P1":   (28672,  4096, 16384, 5273.0, "_ts_gm8"),
}

# Smoke-PASS candidates
CANDS = [
    ("DLA1", "sgpr96"),
    ("P1",   "fwgs256_256"),
    ("P1",   "sgpr80"),
]


def bench_one(M, N, K, full_suffix, gpu_id):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return None
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
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=1500, env=env)
        if r.returncode != 0:
            return {"error": r.stderr[-200:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": str(e)}


def main():
    out = {"runs": [], "summary": []}
    for shape, tag in CANDS:
        M, N, K, comp, ps = SHAPE_INFO[shape]
        cand_suffix = ps + NEW_PREFIX + tag
        base_runs = []; cand_runs = []
        for i in range(N_RUNS):
            b = bench_one(M, N, K, ps, GPU)
            c = bench_one(M, N, K, cand_suffix, GPU)
            print(f"  {shape:6s} {tag:14s} run{i}  base={b} cand={c}", flush=True)
            out["runs"].append({"shape": shape, "tag": tag, "i": i, "base": b, "cand": c})
            if b and "tflops" in b: base_runs.append(b["tflops"])
            if c and "tflops" in c: cand_runs.append(c["tflops"])
        if not base_runs or not cand_runs:
            verdict = "ERR"
            mean_b = mean_c = max_b = None; dpp = None
        else:
            mean_b = statistics.mean(base_runs); mean_c = statistics.mean(cand_runs)
            max_b = max(base_runs)
            dpp = (mean_c - mean_b) / comp * 100
            verdict = "WIN" if (mean_c >= max_b and dpp >= 1.0) else "fail"
        print(f"  -> {shape}/{tag}: mean_base={mean_b} mean_cand={mean_c} max_base={max_b} Δpp={dpp} {verdict}", flush=True)
        out["summary"].append({"shape": shape, "tag": tag, "mean_base": mean_b, "mean_cand": mean_c,
                                "max_base": max_b, "comp": comp, "delta_pp": dpp, "verdict": verdict,
                                "base_runs": base_runs, "cand_runs": cand_runs})
    json.dump(out, open(os.path.join(SCRIPT_DIR, "bench_round17_optC_verify.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
