#!/usr/bin/env python3
"""R25B verify: 3-run × (warmup=200 iters=500 trim=10%) on candidate winners.

Each (shape, variant) is a separate process. Measure 3 times, report
mean+std to filter contention noise. SAFETY: skips known-crashing combos.
"""
import json, math, os, statistics, subprocess, sys, sysconfig, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
GPU = int(os.environ.get("R25B_GPU", "3"))
N_REPS = int(os.environ.get("R25B_REPS", "3"))

WARMUP = 200
ITERS = 500
TRIM = 0.10

# Shapes + parents matching build_round25_optB.py
SHAPES = [
    ("DLA1", 4096,   32768, 128256, "_ts_pf6_6_v12_memc"),
    ("DLA2", 128256, 32768,   4096, "_ts_v12_memc_dc"),
    ("DLA7", 28672,  32768,   4096, "_ts_lgk2_v12_memc"),
    ("MID1", 14336,  4096,   32768, "_ts_v12_memc_dc"),
    ("MID2", 4096,   28672,  32768, "_ts_v12"),
]

# Variants to verify; SKIP known-crashing combos.
# DLA1 gm12, gm16 crash with APERTURE_VIOLATION (M=4096 + STEP3_PF_N=6 + gm>=12).
# DLA2 gm16 crashed in original parallel run (rc=-6) — skip.
SKIP = {
    ("DLA1", "_r25b_gm12"),
    ("DLA1", "_r25b_gm16"),
    ("DLA2", "_r25b_gm16"),
}

VARIANT_SUFFIXES = [
    "_r25b_baseline",
    "_r25b_gm3",
    "_r25b_gm6",
    "_r25b_gm12",
    "_r25b_gm16",
]


def make_bench_script(M, N, K, suffix):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    return so_path, f"""
import sys, math, torch, importlib.util, json
WARMUP, ITERS, TRIM = {WARMUP}, {ITERS}, {TRIM}
M, N, K = {M}, {N}, {K}
torch.manual_seed(0)
def gen_fp4(r, K):
    c = K // 2
    return (torch.randint(0, 16, (r, c), dtype=torch.uint8, device='cuda') << 4) | torch.randint(0, 16, (r, c), dtype=torch.uint8, device='cuda')
def preshuffle(se):
    r, kb = se.shape; pr = math.ceil(r/64)*64; pk = math.ceil(kb/8)*8
    raw = torch.full((pr, pk), 0x7F, dtype=torch.uint8, device=se.device)
    raw[:r,:kb] = (se.to(torch.int16)+127).to(torch.uint8)
    sh = raw.view(pr//32,2,16,pk//8,2,4,1).permute(0,3,5,2,4,1,6).contiguous().view(pr//32,pk*32)
    sh = sh.view(pr//64,2,pk*32//4,4).permute(0,2,1,3).contiguous()
    return sh.view(pr//64,pk*64)
spec = importlib.util.spec_from_file_location('{module_name}', '{so_path}')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
A=gen_fp4(M,K); B=gen_fp4(N,K)
sc_a=torch.randint(-2,3,(M,K//32),dtype=torch.int8,device='cuda')
sc_b=torch.randint(-2,3,(N,K//32),dtype=torch.int8,device='cuda')
A_sc=preshuffle(sc_a); B_sc=preshuffle(sc_b)
C=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
def run(): mod.gemm_rcr(A,B,A_sc,B_sc,C)
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
print(json.dumps({{"tflops": round(t,2), "ms": round(avg,4)}}))
"""


def bench_one(M, N, K, suffix, gpu):
    so_path, script = make_bench_script(M, N, K, suffix)
    if not os.path.exists(so_path):
        return {"error": "missing .so"}
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=1800, env=env)
        if r.returncode != 0:
            return {"error": f"rc={r.returncode}", "stderr": r.stderr[-500:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": str(e)}


def main():
    print(f"R25B VERIFY: {N_REPS} reps each on GPU {GPU}")
    print(f"warmup={WARMUP} iters={ITERS} trim={TRIM}")
    print("=" * 120)
    out = {"warmup": WARMUP, "iters": ITERS, "trim": TRIM, "gpu": GPU,
           "n_reps": N_REPS, "shapes": {lab: {} for (lab, *_r) in SHAPES}}
    t0 = time.time()
    for (lab, M, N, K, ps) in SHAPES:
        print(f"\n--- {lab}  M={M} N={N} K={K} ---")
        for vs in VARIANT_SUFFIXES:
            if (lab, vs) in SKIP:
                print(f"  {vs:20s}  SKIPPED (known crash)")
                out["shapes"][lab][vs] = {"skipped": True}
                continue
            tflops_runs = []
            errs = []
            for rep in range(N_REPS):
                res = bench_one(M, N, K, ps + vs, GPU)
                if "error" in res:
                    errs.append(res["error"])
                else:
                    tflops_runs.append(res["tflops"])
            if not tflops_runs:
                print(f"  {vs:20s}  ALL ERR ({errs[0] if errs else '?'})")
                out["shapes"][lab][vs] = {"errors": errs}
                continue
            mn = max(tflops_runs)  # use best of N (warmer caches)
            md = statistics.median(tflops_runs)
            std = statistics.stdev(tflops_runs) if len(tflops_runs) > 1 else 0.0
            out["shapes"][lab][vs] = {"runs": tflops_runs, "max": mn,
                                      "median": md, "std": std, "errors": errs}
            print(f"  {vs:20s}  runs={tflops_runs}  max={mn:.1f}  med={md:.1f}  std={std:.2f}")

    print(f"\nElapsed: {time.time()-t0:.1f}s")
    out_path = os.path.join(SCRIPT_DIR, "bench_round25_optB_verify.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {out_path}")

    # Summary by max-TFLOPS comparison vs baseline
    print("\n" + "=" * 120)
    print("VERIFY SUMMARY (max-of-N TFLOPS, Δ vs baseline_max)")
    print("=" * 120)
    for (lab, *_r) in SHAPES:
        b = out["shapes"][lab].get("_r25b_baseline", {})
        bm = b.get("max")
        if bm is None:
            print(f"  {lab}: BASELINE FAILED")
            continue
        print(f"  {lab} baseline_max = {bm:.1f} TFLOPS (med={b.get('median'):.1f} std={b.get('std'):.2f})")
        for vs in VARIANT_SUFFIXES:
            if vs == "_r25b_baseline" or (lab, vs) in SKIP:
                continue
            v = out["shapes"][lab].get(vs, {})
            vm = v.get("max")
            if vm is None:
                print(f"    {vs:20s}  ERR")
                continue
            d_pp = (vm - bm) / bm * 100
            verdict = "WIN" if d_pp >= 1.5 else ("borderline" if d_pp >= 0 else "LOSE")
            print(f"    {vs:20s}  max={vm:.1f} med={v.get('median'):.1f} std={v.get('std'):.2f}  Δ={d_pp:+.2f}%  {verdict}")


if __name__ == "__main__":
    main()
