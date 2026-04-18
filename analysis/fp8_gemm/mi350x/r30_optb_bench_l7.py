#!/usr/bin/env python3
"""R30 OptB single-rep bench on L7 (32768x4096x14336).

L7 currently picks `ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all` = 6157.3 TFLOPS
(117.9% of comp 5223.4). Audit hypothesis: the L4 fix (tv0 plain w/o dc_gm7)
and L8 fix (u16 parent) might also win on L7 — same N=4096, K=14336 geometry,
larger M.

Tests two pre-built variants:
  1. ts_u16_gm7_pfoff52_kx14336_btw_all     (R28-C L8 fix)
  2. ts_v12_tv0_memc_btw_all_pfoff48_kx14336 (R29   L4 fix)

Per benchmark-rules.md: WARMUP=200 ITERS=500 TRIM=0.10. Single-rep first
then 5-rep verify only if any beats current 6157.3 by >1%.
GPU isolation via HIP_VISIBLE_DEVICES.
"""
import os, sys, json, math, subprocess, statistics, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

M, N, K = 32768, 4096, 14336
COMP = 5223.4
CURRENT_BEST = 6157.3  # ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all from R25_FINAL_v2

WARMUP = 200
ITERS = 500
TRIM = 0.10

GPU = os.environ.get("R30_GPU", "1")
REPS = int(os.environ.get("R30_REPS", "1"))
TIMEOUT = 1200

CANDIDATES = [
    {
        "tag": "u16_kx14336_R28C",
        "module": "tk_mxfp4_gluon_cpp_n4096_k14336_ts_u16_gm7_pfoff52_kx14336_btw_all",
    },
    {
        "tag": "tv0_btw_all_pfoff48_kx14336_R29",
        "module": "tk_mxfp4_gluon_cpp_n4096_k14336_ts_v12_tv0_memc_btw_all_pfoff48_kx14336",
    },
]


def bench_once(module_name, seed):
    so_path = os.path.join(BUILD_DIR, f"{module_name}.cpython-310-x86_64-linux-gnu.so")
    script = f"""
import sys, math, torch, importlib.util, json
torch.manual_seed({seed})
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
finite_frac = float(torch.isfinite(C).float().mean().item())
nz_frac = float((C != 0).float().mean().item())
print(json.dumps({{"tflops": round(t,2), "ms": round(avg,4), "finite_frac": finite_frac, "nz_frac": nz_frac}}))
"""
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = GPU
    try:
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=TIMEOUT, env=env)
        if r.returncode != 0:
            return {"err": "rc=%d" % r.returncode, "stderr": r.stderr[-400:]}
        return json.loads(r.stdout.strip())
    except subprocess.TimeoutExpired:
        return {"err": "timeout"}
    except Exception as e:
        return {"err": str(e)}


def bench_candidate(cand):
    tag = cand["tag"]
    module = cand["module"]
    so_path = os.path.join(BUILD_DIR, f"{module}.cpython-310-x86_64-linux-gnu.so")
    print(f"\n--- {tag} ---")
    print(f"  module: {module}")
    if not os.path.exists(so_path):
        print(f"  MISSING SO: {so_path}")
        return {"tag": tag, "verdict": "MISSING_SO"}
    results = []
    for i in range(REPS):
        t0 = time.time()
        r = bench_once(module, seed=i)
        dt = time.time() - t0
        if "err" in r:
            print(f"    rep{i}: ERR {r}  ({dt:.1f}s)")
            results.append({"err": r})
        else:
            print(f"    rep{i}: {r['tflops']:>8.2f} TFLOPS  ms={r['ms']}  finite={r['finite_frac']:.3f} nz={r['nz_frac']:.3f}  ({dt:.1f}s)")
            results.append(r)
    ok = [r for r in results if "err" not in r]
    if not ok:
        return {"tag": tag, "verdict": "DEAD", "reps": [], "mean": 0.0, "std": 0.0}
    tflops = [r["tflops"] for r in ok]
    mean = statistics.mean(tflops)
    std = statistics.stdev(tflops) if len(tflops) > 1 else 0.0
    delta_pct = (mean - CURRENT_BEST) / CURRENT_BEST * 100
    finite = min(r["finite_frac"] for r in ok)
    nz = min(r["nz_frac"] for r in ok)
    if mean < 1000.0 or finite < 0.5:
        verdict = "DEAD"
    elif mean >= CURRENT_BEST * 1.01:
        verdict = "WIN_GT_1PCT"
    elif mean >= CURRENT_BEST:
        verdict = "GAIN_LT_1PCT"
    else:
        verdict = "LOSE"
    print(f"    mean={mean:.2f} +/- {std:.2f}  vs current best {CURRENT_BEST}: {mean - CURRENT_BEST:+.2f} ({delta_pct:+.2f}%)  vs comp {COMP}: {mean/COMP*100:.2f}%  -> {verdict}")
    return {
        "tag": tag, "module": module, "verdict": verdict,
        "reps": tflops, "mean": mean, "std": std,
        "delta_pct_vs_current_best": delta_pct,
        "min_finite_frac": finite, "min_nz_frac": nz,
        "reps_ok": len(ok), "reps_total": REPS,
    }


def main():
    print(f"=== R30 OptB Bench L7: {M}x{N}x{K} ===")
    print(f"current best: {CURRENT_BEST} TFLOPS  (ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all)")
    print(f"competitor:   {COMP} TFLOPS")
    print(f"warmup={WARMUP} iters={ITERS} trim={TRIM} reps={REPS} GPU={GPU}")
    all_results = [bench_candidate(c) for c in CANDIDATES]
    summary = {
        "shape": [M, N, K], "comp": COMP, "current_best": CURRENT_BEST,
        "current_best_tag": "ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all",
        "warmup": WARMUP, "iters": ITERS, "trim": TRIM, "reps": REPS, "gpu": GPU,
        "candidates": all_results,
    }
    out = os.path.join(SCRIPT_DIR, f"R30_OPTB_BENCH_L7_reps{REPS}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out}")
    print("\n=== Summary ===")
    print(f"{'tag':40s}  {'mean':>9}  {'std':>7}  {'delta%':>7}  verdict")
    for r in all_results:
        m = r.get("mean", 0)
        s = r.get("std", 0)
        d = r.get("delta_pct_vs_current_best", 0)
        v = r.get("verdict", "?")
        print(f"{r['tag']:40s}  {m:>9.2f}  {s:>7.2f}  {d:>+6.2f}%  {v}")


if __name__ == "__main__":
    main()
