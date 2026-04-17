#!/usr/bin/env python3
"""R19A smoke bench. Reads snr_probe_r19a.json, benches every OK pair.
Bench params: warmup=200 iters=500 trim=10% (per benchmark-rules).
Runs in parallel across GPUs 0-3 (R19B/C use 4-7)."""
import json, math, os, subprocess, sys, sysconfig, statistics, time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
WARMUP, ITERS, TRIM = 200, 500, 0.10
GPU_POOL = [0, 1, 2, 3]

# Same shape table as snr_probe_r19a.py
SHAPES = {
    "S1":  (14336,  4096, 32768, 5245.4, "_lgk2_dc"),
    "S2":  (16384,  4096, 28672, 5525.3, "_u32"),
    "S3":  ( 4096, 32768, 28672, 5568.2, "_v20_memc"),
    "S4":  ( 4096, 28672, 32768, 5649.9, "_u16"),
    "S5":  (32768,  4096, 14336, 5223.4, "_ts_gm8_v12"),
    "S6":  ( 4096, 32768, 14336, 5296.1, "_ts_lgk2_memc"),
    "S7":  (28672, 32768,  4096, 4466.6, "_ts_lgk2_v12_memc"),
    "S8":  ( 4096, 32768,  6144, 4548.6, "_ts_pf4_memc"),
    "S9":  (16384, 28672,  2048, 3482.3, "_ts_gm2_v12_memc_dc"),
    "S10": (16384, 28672,  4096, 4411.7, "_ts_gm2_v12_memc"),
    "S11": (28672,  4096,  8192, 4810.0, "_ts_lgk2_memc_dc"),
    "S12": (14336, 32768,  4096, 4462.6, "_ts_v12_tv0_memc"),
    "S13": ( 4096, 14336, 16384, 5013.0, "_ts_lgk2"),
    "S14": ( 6144,  4096, 16384, 4428.1, "_ts_lgk2"),
    "S15": (32768,  4096,  7168, 4666.8, "_ts_gm8_v12"),
}


def bench_script(M, N, K, suffix):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    return so_path, module_name, f"""
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


def bench_one(args):
    M, N, K, suffix, gpu = args
    sp, mn, script = bench_script(M, N, K, suffix)
    if not os.path.exists(sp):
        return suffix, gpu, {"err": f"missing {sp}"}
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu)
    try:
        r = subprocess.run([sys.executable, "-c", script],
            capture_output=True, text=True, timeout=900, env=env)
        if r.returncode != 0:
            return suffix, gpu, {"err": r.stderr[-300:]}
        return suffix, gpu, json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return suffix, gpu, {"err": str(e)}


def main():
    snr_path = os.path.join(SCRIPT_DIR, "snr_probe_r19a.json")
    if not os.path.exists(snr_path):
        print("No snr_probe_r19a.json yet — run that first")
        sys.exit(1)
    snr = json.load(open(snr_path))
    ok_pairs = snr["ok_pairs"]
    print(f"OK pairs from SNR probe: {len(ok_pairs)}")
    for p in ok_pairs:
        print(f"  {p}")

    # Bench parent + variant for each OK pair
    tasks = []  # (label, M, N, K, comp, parent_suffix, variant_suffix)
    for [lab, vs, verdict] in ok_pairs:
        if lab not in SHAPES:
            continue
        M, N, K, comp, ps = SHAPES[lab]
        tasks.append((lab, M, N, K, comp, ps, vs))

    # Build job list: parent (one per shape) + every variant
    parent_jobs = {}  # lab -> (M,N,K,ps)
    variant_jobs = []  # list of (lab, M, N, K, comp, ps, vs)
    for (lab, M, N, K, comp, ps, vs) in tasks:
        parent_jobs[lab] = (M, N, K, comp, ps)
        variant_jobs.append((lab, M, N, K, comp, ps, vs))

    # Run all jobs in parallel
    all_jobs = []  # (key, M, N, K, suffix, gpu)
    i = 0
    for lab, (M, N, K, comp, ps) in parent_jobs.items():
        gpu = GPU_POOL[i % len(GPU_POOL)]
        all_jobs.append((f"parent:{lab}", M, N, K, ps, gpu))
        i += 1
    for (lab, M, N, K, comp, ps, vs) in variant_jobs:
        gpu = GPU_POOL[i % len(GPU_POOL)]
        all_jobs.append((f"var:{lab}:{vs}", M, N, K, ps + vs, gpu))
        i += 1

    print(f"\nTotal bench jobs: {len(all_jobs)}")
    results = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(GPU_POOL)) as ex:
        futs = {ex.submit(bench_one, (M, N, K, suffix, gpu)): key
                for (key, M, N, K, suffix, gpu) in all_jobs}
        done = 0
        for fut in as_completed(futs):
            key = futs[fut]
            _, gpu, res = fut.result()
            results[key] = res
            done += 1
            print(f"  [{done}/{len(all_jobs)}] {key:50s} gpu={gpu}  -> {res}", flush=True)
    print(f"Bench done in {(time.time()-t0)/60:.1f} min\n")

    # Summary
    out = {"params": {"warmup": WARMUP, "iters": ITERS, "trim": TRIM}, "rows": []}
    print("=" * 110)
    print(f"{'Lab':5s} {'shape':25s} {'comp':>8s} {'parent':>10s} {'variant_suf':25s} {'variant':>10s} {'Δpp':>8s}")
    print("-" * 110)
    for (lab, M, N, K, comp, ps, vs) in variant_jobs:
        p_t = results.get(f"parent:{lab}", {}).get("tflops", 0)
        v_t = results.get(f"var:{lab}:{vs}", {}).get("tflops", 0)
        delta_pp = (v_t - p_t) / comp * 100 if (p_t > 0 and v_t > 0) else None
        d_str = f"{delta_pp:+.2f}" if delta_pp is not None else "n/a"
        print(f"{lab:5s} {f'{M}x{N}x{K}':25s} {comp:8.1f} {p_t:10.2f} {vs:25s} {v_t:10.2f} {d_str:>8s}")
        out["rows"].append({"lab": lab, "shape": [M, N, K], "comp": comp,
            "parent": ps, "variant": vs,
            "parent_tflops": p_t, "variant_tflops": v_t,
            "delta_pp": (None if delta_pp is None else round(delta_pp, 3))})
    with open(os.path.join(SCRIPT_DIR, "bench_round19_optA_smoke.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved bench_round19_optA_smoke.json")


if __name__ == "__main__":
    main()
