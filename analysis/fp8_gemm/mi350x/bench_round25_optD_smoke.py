#!/usr/bin/env python3
"""R25-D STACK bench (warmup=200 iters=500 trim=10%) — 3 reps per variant.

Tests whether GROUP_SIZE_M=6 (R25-B) and R25C_TAIL_PF_OFF_ITERS=4 (R25-C) STACK on DLA2/DLA7.

GPUs 6, 7 only (R25 reviewer using GPUs 0-4). Single-GPU per shape for noise control.
DLA2 -> GPU 6; DLA7 -> GPU 7. Each variant runs 3 times, sequential per GPU.
"""
import json, math, os, subprocess, sys, sysconfig, time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10
N_REPS = 3

# (label, M, N, K, gpu)
SHAPES = [
    ("DLA2", 128256, 32768, 4096, 6),
    ("DLA7",  28672, 32768, 4096, 7),
]

VARIANT_SUFFIXES = [
    "_r25d_baseline",
    "_r25d_gm6",
    "_r25d_pfoff4",
    "_r25d_gm6_pfoff4",
]

# Per-shape suffixes are e.g. "_r25d_baseline_dla2" — we append _<lab.lower()> at lookup time.


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
# Quick coverage check (NaN/Inf expected with this random scale data — matches r25c methodology).
# A 'failed' kernel typically writes 0 everywhere; we only flag if coverage is suspicious.
run(); torch.cuda.synchronize()
nz = (C != 0).float().mean().item()
nan_frac = torch.isnan(C).float().mean().item()
if nz < 0.5:
    print(json.dumps({{"error": "C-coverage too low", "nz": nz, "nan_frac": nan_frac}})); sys.exit(0)
times=[]
for _ in range(ITERS):
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record(); run(); e.record(); torch.cuda.synchronize()
    times.append(s.elapsed_time(e))
times.sort(); trim=int(len(times)*TRIM)
times=times[trim:-trim] if trim>0 else times
avg=sum(times)/len(times); t=2.0*M*N*K/(avg*1e-3)/1e12
print(json.dumps({{"tflops": round(t,2), "ms": round(avg,4), "n_iters_kept": len(times), "nz": nz}}))
"""


def bench_one(args):
    """Run a single (lab, vs, rep) bench."""
    lab, M, N, K, vs, gpu, rep = args
    full_suffix = f"{vs}_{lab.lower()}"
    so_path, script = make_bench_script(M, N, K, full_suffix)
    if not os.path.exists(so_path):
        return lab, vs, rep, gpu, {"error": "missing .so", "so": so_path}
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=1800, env=env)
        if r.returncode != 0:
            return lab, vs, rep, gpu, {"error": f"rc={r.returncode}", "stderr": r.stderr[-500:]}
        return lab, vs, rep, gpu, json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return lab, vs, rep, gpu, {"error": str(e)}


def run_shape_serial(shape):
    """Run all variants*reps for one shape, sequential on its dedicated GPU."""
    lab, M, N, K, gpu = shape
    print(f"\n[{lab}] starting on GPU {gpu}", flush=True)
    rows = []
    for vs in VARIANT_SUFFIXES:
        for rep in range(N_REPS):
            args = (lab, M, N, K, vs, gpu, rep)
            res = bench_one(args)
            _lab, _vs, _rep, _gpu, r = res
            tag = f"{lab} {vs} rep{rep} gpu{gpu}"
            if "error" in r:
                print(f"  {tag:55s}  ERR {r.get('error')}", flush=True)
            else:
                print(f"  {tag:55s}  {r['tflops']:>8.2f} TFLOPS (ms={r.get('ms')})", flush=True)
            rows.append((vs, rep, r))
    return lab, rows


def main():
    print(f"R25-D STACK bench: {len(SHAPES)} shapes x {len(VARIANT_SUFFIXES)} variants x {N_REPS} reps")
    print(f"warmup={WARMUP} iters={ITERS} trim={TRIM}")
    print(f"GPUs: {[s[4] for s in SHAPES]}")
    print("=" * 110)
    out = {"shapes": {}, "warmup": WARMUP, "iters": ITERS, "trim": TRIM, "n_reps": N_REPS}
    t0 = time.time()
    # Run shapes in parallel (different GPUs), variants/reps serial within each shape
    with ProcessPoolExecutor(max_workers=len(SHAPES)) as ex:
        futs = {ex.submit(run_shape_serial, s): s for s in SHAPES}
        for fut in as_completed(futs):
            lab, rows = fut.result()
            out["shapes"][lab] = {}
            for (vs, rep, r) in rows:
                out["shapes"][lab].setdefault(vs, []).append(r)
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    with open(os.path.join(SCRIPT_DIR, "bench_round25_optD_smoke.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Summary
    print("\n" + "=" * 110)
    print("SUMMARY (3-rep median, Δ vs _r25d_baseline)")
    print("=" * 110)
    summary_rows = []
    for (lab, *_rest) in SHAPES:
        bruns = [r.get("tflops") for r in out["shapes"].get(lab, {}).get("_r25d_baseline", []) if r.get("tflops") is not None]
        if not bruns:
            print(f"  {lab}: BASELINE FAILED")
            continue
        bruns_sorted = sorted(bruns)
        bmed = bruns_sorted[len(bruns_sorted)//2]
        bmax = max(bruns)
        print(f"\n  [{lab}] baseline  runs={[round(x,1) for x in bruns]}  med={bmed:.2f}  max={bmax:.2f}")
        for vs in VARIANT_SUFFIXES:
            if vs == "_r25d_baseline":
                continue
            vruns = [r.get("tflops") for r in out["shapes"][lab].get(vs, []) if r.get("tflops") is not None]
            if not vruns:
                err_runs = [r.get("error","?") for r in out["shapes"][lab].get(vs, [])]
                print(f"    {vs:25s}  ERR {err_runs}")
                continue
            vruns_sorted = sorted(vruns)
            vmed = vruns_sorted[len(vruns_sorted)//2]
            vmax = max(vruns)
            d_med = vmed - bmed
            d_med_pp = d_med / bmed * 100
            d_max = vmax - bmax
            d_max_pp = d_max / bmax * 100
            print(f"    {vs:25s}  runs={[round(x,1) for x in vruns]}  med={vmed:.2f}  max={vmax:.2f}  Δmed={d_med:+.2f}({d_med_pp:+.2f}%)  Δmax={d_max:+.2f}({d_max_pp:+.2f}%)")
            summary_rows.append((lab, vs, vmed, vmax, d_med_pp, d_max_pp))

    # Stack verdict
    print("\n" + "=" * 110)
    print("STACK VERDICT (does gm6_pfoff4 > max(gm6, pfoff4)?)")
    print("=" * 110)
    for (lab, *_rest) in SHAPES:
        def med(vs):
            xs = [r.get("tflops") for r in out["shapes"].get(lab, {}).get(vs, []) if r.get("tflops") is not None]
            if not xs: return None
            xs.sort()
            return xs[len(xs)//2]
        def maxof(vs):
            xs = [r.get("tflops") for r in out["shapes"].get(lab, {}).get(vs, []) if r.get("tflops") is not None]
            return max(xs) if xs else None
        b_med = med("_r25d_baseline")
        gm6_med = med("_r25d_gm6")
        pf_med = med("_r25d_pfoff4")
        st_med = med("_r25d_gm6_pfoff4")
        if None in (b_med, gm6_med, pf_med, st_med):
            print(f"  {lab}: MISSING DATA")
            continue
        best_singleton = max(gm6_med, pf_med)
        d_stack_vs_best = (st_med - best_singleton) / best_singleton * 100
        d_stack_vs_base = (st_med - b_med) / b_med * 100
        verdict = "STACK WINS" if d_stack_vs_best >= 0.5 else ("flat/anti-stack" if d_stack_vs_best > -1.0 else "REGRESSES vs best singleton")
        print(f"  {lab}: baseline={b_med:.1f}  gm6={gm6_med:.1f}  pfoff4={pf_med:.1f}  STACK={st_med:.1f}  ")
        print(f"        STACK Δ vs best_singleton({best_singleton:.1f}) = {d_stack_vs_best:+.2f}%   STACK Δ vs baseline = {d_stack_vs_base:+.2f}%   -> {verdict}")


if __name__ == "__main__":
    main()
