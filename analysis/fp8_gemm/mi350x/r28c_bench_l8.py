#!/usr/bin/env python3
"""R28-C bench: single-shape bench of new u16+kx14336 variant on 16384x4096x14336 (L8).
Runs 5 reps with WARMUP=200 ITERS=500 TRIM=0.10 per benchmark-rules.md.
GPU 3 only.
"""
import os, sys, json, math, subprocess, statistics, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
SO_NAME = "tk_mxfp4_gluon_cpp_n4096_k14336_ts_u16_gm7_pfoff52_kx14336_btw_all.cpython-310-x86_64-linux-gnu.so"
SO_PATH = os.path.join(BUILD_DIR, SO_NAME)

M, N, K = 16384, 4096, 14336
COMP = 5142.1
V1_BEST = 5032.7  # ts_u16
SUCCESS = 5300.0  # WIN threshold per R28_PLAN.md §2.C
WARMUP = 200
ITERS = 500
TRIM = 0.10
REPS = 5
TIMEOUT = 600


def bench_once(seed):
    module_name = "tk_mxfp4_gluon_cpp_n4096_k14336_ts_u16_gm7_pfoff52_kx14336_btw_all"
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
spec=importlib.util.spec_from_file_location('{module_name}','{SO_PATH}')
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
    env["HIP_VISIBLE_DEVICES"] = "3"
    try:
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=TIMEOUT, env=env
        )
        if r.returncode != 0:
            return {"err": "rc=%d" % r.returncode, "stderr": r.stderr[-400:]}
        return json.loads(r.stdout.strip())
    except subprocess.TimeoutExpired:
        return {"err": "timeout"}
    except Exception as e:
        return {"err": str(e)}


def main():
    if not os.path.exists(SO_PATH):
        print("MISSING SO: %s" % SO_PATH)
        sys.exit(2)
    print(f"=== R28-C Bench: {M}x{N}x{K} (L8) ===")
    print(f"SO: {SO_NAME}")
    print(f"v1 best: {V1_BEST} TFLOPS  competitor: {COMP} TFLOPS  success: {SUCCESS} TFLOPS")
    print(f"warmup={WARMUP} iters={ITERS} trim={TRIM} reps={REPS} GPU=3\n")
    results = []
    for i in range(REPS):
        t0 = time.time()
        r = bench_once(seed=i)
        dt = time.time() - t0
        if "err" in r:
            print(f"  rep{i}: ERR {r}  ({dt:.1f}s)")
            results.append(None)
        else:
            print(f"  rep{i}: {r['tflops']:>8.2f} TFLOPS  ms={r['ms']}  finite={r['finite_frac']:.3f} nz={r['nz_frac']:.3f}  ({dt:.1f}s)")
            results.append(r)
    ok = [r for r in results if r is not None]
    if not ok:
        print("\nALL REPS FAILED")
        verdict = "DEAD"
        summary = {
            "shape": [M, N, K], "so": SO_NAME, "verdict": verdict,
            "reps": [], "mean": 0.0, "std": 0.0,
            "v1_best": V1_BEST, "comp": COMP, "success": SUCCESS,
            "warmup": WARMUP, "iters": ITERS, "trim": TRIM, "reps_total": REPS, "reps_ok": 0,
            "errors": [r for r in results if r is None or "err" in (r or {})],
        }
        out = os.path.join(SCRIPT_DIR, "R28C_BENCH_L8.json")
        with open(out, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Wrote {out}")
        sys.exit(3)
    tflops = [r["tflops"] for r in ok]
    mean = statistics.mean(tflops)
    std = statistics.stdev(tflops) if len(tflops) > 1 else 0.0
    finite = min(r["finite_frac"] for r in ok)
    nz = min(r["nz_frac"] for r in ok)
    # Verdict logic
    if len(ok) < REPS:
        verdict = "DEAD"
    elif mean >= SUCCESS and std <= 50.0:
        verdict = "WIN"
    elif mean >= COMP:
        verdict = "GAP-CLOSE"
    else:
        verdict = "LOSE"
    print("\n--- Summary ---")
    print(f"reps_ok={len(ok)}/{REPS}")
    print(f"mean = {mean:.2f} TFLOPS  std = {std:.2f}")
    print(f"vs v1 best ({V1_BEST}): {mean - V1_BEST:+.2f} TFLOPS  ({(mean - V1_BEST)/V1_BEST*100:+.2f}%)")
    print(f"vs comp    ({COMP}): {mean - COMP:+.2f} TFLOPS  ({mean/COMP*100:.2f}%)")
    print(f"vs success ({SUCCESS}): {mean - SUCCESS:+.2f} TFLOPS")
    print(f"min finite_frac = {finite:.4f}, min nz_frac = {nz:.4f}")
    print(f"VERDICT: {verdict}")
    summary = {
        "shape": [M, N, K], "so": SO_NAME, "verdict": verdict,
        "reps": tflops, "mean": mean, "std": std,
        "v1_best": V1_BEST, "comp": COMP, "success": SUCCESS,
        "min_finite_frac": finite, "min_nz_frac": nz,
        "warmup": WARMUP, "iters": ITERS, "trim": TRIM, "reps_total": REPS, "reps_ok": len(ok),
    }
    out = os.path.join(SCRIPT_DIR, "R28C_BENCH_L8.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
