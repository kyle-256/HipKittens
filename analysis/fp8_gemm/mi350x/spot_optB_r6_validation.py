#!/usr/bin/env python3
"""Round 6 Reviewer: validate Optimizer B's WIN claim on a SINGLE GPU.

- 5-run replication on 14336x4096x32768 (4 variants)
- WIN-sample regression spot-check (1 run on 3 shapes)
- Deep-LOSE neighbor spot-check (1 run on 4 shapes)
warmup=200, iters=500, trim=10%
"""
import os, sys, json, math, time, subprocess, sysconfig, statistics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP, ITERS, TRIM = 200, 500, 0.10
GPU = int(os.environ.get("BENCH_GPU", "5"))

VARIANTS = [
    "_v16_wpe2",                       # baseline
    "_optB_r6_u16_v16_wpe2",           # claimed +3.16pp winner
    "_optB_r6_u8_v16_wpe2_memc",       # claimed +3.04pp "best overall"
    "_optB_r6_u16_lgk2_dc_v16_wpe2",   # claimed +2.54pp
]

TARGET_SHAPE = (14336, 4096, 32768, 5245.4)
WIN_SAMPLE = [
    (16384, 4096, 2048, 2995.0),
    (4096, 128256, 32768, 3195.3),
    (4096, 4096, 8192, 3959.9),
]
DEEP_LOSE_NEIGHBORS = [
    (4096, 32768, 128256, 5781.1),
    (16384, 4096, 28672, 5525.3),
    (28672, 4096, 16384, 5350.6),
    (4096, 32768, 28672, 5568.2),
]


def bench(m, n, k, suffix, gpu_id):
    module_name = f"tk_mxfp4_gluon_cpp_n{n}_k{k}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return {"err": f"no_so: {so_path}"}
    script = f"""
import sys, math, torch, importlib.util, json
torch.manual_seed(0)
WARMUP, ITERS, TRIM = {WARMUP}, {ITERS}, {TRIM}
M, N, K = {m}, {n}, {k}
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
                           capture_output=True, text=True, timeout=600, env=env)
        if r.returncode != 0:
            return {"err": "run_failed", "stderr": r.stderr[-500:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"err": str(e)}


def main():
    print(f"=== Round 6 Reviewer validation (GPU={GPU}) ===")
    print(f"warmup={WARMUP} iters={ITERS} trim={TRIM}\n")
    results = {}

    # Phase 1: 5-run replication on TARGET_SHAPE
    M, N, K, COMP = TARGET_SHAPE
    shape_key = f"{M}x{N}x{K}"
    print(f"--- Phase 1: 5-run replication on {shape_key} ---")
    rep_results = {v: [] for v in VARIANTS}
    # Interleave runs so any drift hits all variants similarly
    for run_idx in range(5):
        print(f"  run {run_idx+1}/5:")
        for v in VARIANTS:
            r = bench(M, N, K, v, GPU)
            if "tflops" in r:
                rep_results[v].append(r["tflops"])
                print(f"    {v:38s} {r['tflops']:8.2f} TFLOPS  ({r['tflops']/COMP*100:.2f}%)")
            else:
                print(f"    {v:38s} FAIL: {r.get('err','?')[:200]}")
    results["replication"] = rep_results
    results["target_shape"] = list(TARGET_SHAPE)

    print(f"\n--- 5-run summary on {shape_key} (comp={COMP}) ---")
    print(f"{'variant':<40s} {'mean':>8s} {'min':>8s} {'max':>8s} {'std':>6s}  {'mean%':>6s}")
    summary = {}
    for v in VARIANTS:
        ts = rep_results[v]
        if not ts:
            print(f"  {v:<40s} EMPTY")
            continue
        mn = statistics.mean(ts); mx = max(ts); mi = min(ts)
        sd = statistics.stdev(ts) if len(ts) > 1 else 0.0
        summary[v] = {"mean": mn, "min": mi, "max": mx, "std": sd, "pct": mn/COMP*100}
        print(f"  {v:<40s} {mn:8.2f} {mi:8.2f} {mx:8.2f} {sd:6.2f}  {mn/COMP*100:6.2f}")
    results["summary"] = summary

    # Phase 2: WIN-sample regression
    print(f"\n--- Phase 2: WIN-sample regression (1 run each) ---")
    win_results = {}
    for (m, n, k, comp) in WIN_SAMPLE:
        sk = f"{m}x{n}x{k}"
        win_results[sk] = {"comp": comp}
        for v in VARIANTS:
            r = bench(m, n, k, v, GPU)
            if "tflops" in r:
                win_results[sk][v] = r["tflops"]
                print(f"  {sk:25s} {v:38s} {r['tflops']:8.2f} ({r['tflops']/comp*100:.2f}%)")
            else:
                win_results[sk][v] = None
                print(f"  {sk:25s} {v:38s} FAIL: {r.get('err','?')[:150]}")
    results["win_sample"] = win_results

    # Phase 3: Deep-LOSE neighbors
    print(f"\n--- Phase 3: Deep-LOSE neighbor regression (1 run each) ---")
    dl_results = {}
    for (m, n, k, comp) in DEEP_LOSE_NEIGHBORS:
        sk = f"{m}x{n}x{k}"
        dl_results[sk] = {"comp": comp}
        for v in VARIANTS:
            r = bench(m, n, k, v, GPU)
            if "tflops" in r:
                dl_results[sk][v] = r["tflops"]
                print(f"  {sk:25s} {v:38s} {r['tflops']:8.2f} ({r['tflops']/comp*100:.2f}%)")
            else:
                dl_results[sk][v] = None
                print(f"  {sk:25s} {v:38s} FAIL: {r.get('err','?')[:150]}")
    results["deep_lose_neighbors"] = dl_results

    # Decision logic
    print(f"\n--- Decision check ---")
    base_summary = summary.get("_v16_wpe2")
    if base_summary:
        base_mean = base_summary["mean"]; base_max = base_summary["max"]
        for v in VARIANTS[1:]:
            s = summary.get(v)
            if not s: continue
            mean_pp = (s["mean"] - base_mean) / COMP * 100
            min_vs_base_max_pp = (s["min"] - base_max) / COMP * 100
            mean_vs_base_max_pp = (s["mean"] - base_max) / COMP * 100
            print(f"  {v:<40s} mean_pp_over_base={mean_pp:+.2f}  mean-baseMax={mean_vs_base_max_pp:+.2f}  min-baseMax={min_vs_base_max_pp:+.2f}")

    out = os.path.join(SCRIPT_DIR, "spot_optB_r6_validation_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out}")


if __name__ == "__main__":
    main()
