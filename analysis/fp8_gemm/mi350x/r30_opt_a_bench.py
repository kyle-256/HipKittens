#!/usr/bin/env python3
"""R30 OPT A: cross-shape transplant scout — single-rep bench of 5 missing variants.

Targets:
  DLA2 = (M=128256, N=32768, K=4096), current best 4943.3 TFLOPS, comp=4536.4
    - ts_lgk2_memc_btw_all
    - ts_lgk2_v12_memc_btw_all
    - v20_memc_btw_step3
  S5L = (M=32768, N=14336, K=2048), current best 3725.5 TFLOPS, comp=3351.4
    - ts_lgk2_memc_btw_all
    - v20_memc_btw_step3

Per .claude/rules/benchmark-rules.md: WARMUP=200 ITERS=500 TRIM=0.10
Each variant runs once (single-rep) on its own idle GPU.
"""
import os, sys, json, math, subprocess, time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10
TIMEOUT = 900

JOBS = [
    # (label, M, N, K, variant_suffix, comp, current_best)
    ("DLA2_lgk2_memc_btw_all",       128256, 32768, 4096, "ts_lgk2_memc_btw_all",       4536.4, 4943.3),
    ("DLA2_lgk2_v12_memc_btw_all",   128256, 32768, 4096, "ts_lgk2_v12_memc_btw_all",   4536.4, 4943.3),
    ("DLA2_v20_memc_btw_step3",      128256, 32768, 4096, "v20_memc_btw_step3",         4536.4, 4943.3),
    ("S5L_lgk2_memc_btw_all",         32768, 14336, 2048, "ts_lgk2_memc_btw_all",       3351.4, 3725.5),
    ("S5L_v20_memc_btw_step3",        32768, 14336, 2048, "v20_memc_btw_step3",         3351.4, 3725.5),
]

# Idle GPUs (verified via rocm-smi)
GPUS = [0, 1, 2, 3, 4]


def bench_one(args):
    label, M, N, K, variant, comp, cur_best, gpu = args
    so_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}_{variant}_r30oa_m{M}.cpython-310-x86_64-linux-gnu.so"
    so_path = os.path.join(BUILD_DIR, so_name)
    module_name = so_name.split(".cpython")[0]

    if not os.path.exists(so_path):
        return {"label": label, "err": "MISSING_SO", "so_path": so_path}

    script = f"""
import sys, math, torch, importlib.util, json
torch.manual_seed(0)
WARMUP, ITERS, TRIM = {WARMUP}, {ITERS}, {TRIM}
M, N, K = {M}, {N}, {K}

def gen_fp4(rows, K):
    cols = K // 2
    lo = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device='cuda')
    hi = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device='cuda')
    return (hi << 4) | lo

def preshuffle_mfma16_merged(scale_exp):
    rows, kb = scale_exp.shape
    pr = math.ceil(rows / 64) * 64
    pk = math.ceil(kb / 8) * 8
    raw = torch.full((pr, pk), 0x7F, dtype=torch.uint8, device=scale_exp.device)
    raw[:rows, :kb] = (scale_exp.to(torch.int16) + 127).to(torch.uint8)
    sh = raw.view(pr // 32, 2, 16, pk // 8, 2, 4, 1)
    sh = sh.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    sh = sh.view(pr // 32, pk * 32)
    sh = sh.view(pr // 64, 2, pk * 32 // 4, 4)
    sh = sh.permute(0, 2, 1, 3).contiguous()
    return sh.view(pr // 64, pk * 64)

spec = importlib.util.spec_from_file_location('{module_name}', '{so_path}')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

A = gen_fp4(M, K); B = gen_fp4(N, K)
k_blocks = K // 32
sc_a = torch.randint(-2, 3, (M, k_blocks), dtype=torch.int8, device='cuda')
sc_b = torch.randint(-2, 3, (N, k_blocks), dtype=torch.int8, device='cuda')
A_sc = preshuffle_mfma16_merged(sc_a); B_sc = preshuffle_mfma16_merged(sc_b)
C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

run = lambda: mod.gemm_rcr(A, B, A_sc, B_sc, C)
for _ in range(WARMUP): run()
torch.cuda.synchronize()

times = []
for _ in range(ITERS):
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record(); run(); e.record(); torch.cuda.synchronize()
    times.append(s.elapsed_time(e))
times.sort()
trim = int(len(times) * TRIM)
times = times[trim:-trim] if trim > 0 else times
avg = sum(times) / len(times)
t = 2.0 * M * N * K / (avg * 1e-3) / 1e12
finite_frac = float(torch.isfinite(C).float().mean().item())
nz_frac = float((C != 0).float().mean().item())
print(json.dumps({{"tflops": round(t,2), "ms": round(avg,4), "finite_frac": finite_frac, "nz_frac": nz_frac}}))
"""
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu)
    t0 = time.time()
    try:
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=TIMEOUT, env=env
        )
        dt = time.time() - t0
        if r.returncode != 0:
            return {"label": label, "gpu": gpu, "err": "rc=%d" % r.returncode,
                    "stderr": r.stderr[-500:], "elapsed_s": round(dt, 1)}
        out = json.loads(r.stdout.strip().splitlines()[-1])
        out["label"] = label
        out["gpu"] = gpu
        out["M"] = M; out["N"] = N; out["K"] = K
        out["variant"] = variant
        out["comp"] = comp
        out["current_best"] = cur_best
        out["elapsed_s"] = round(dt, 1)
        out["gain_vs_best_pct"] = round((out["tflops"] - cur_best) / cur_best * 100, 2)
        out["vs_comp_pct"] = round(out["tflops"] / comp * 100, 2)
        return out
    except subprocess.TimeoutExpired:
        return {"label": label, "gpu": gpu, "err": "timeout", "elapsed_s": round(time.time() - t0, 1)}
    except Exception as e:
        return {"label": label, "gpu": gpu, "err": str(e), "elapsed_s": round(time.time() - t0, 1)}


def main():
    print(f"=== R30 OPT A: 5 variants × single rep ===")
    print(f"warmup={WARMUP} iters={ITERS} trim={TRIM}")
    print(f"GPUs: {GPUS}")
    work = []
    for i, j in enumerate(JOBS):
        gpu = GPUS[i % len(GPUS)]
        work.append(j + (gpu,))
    results = []
    with ProcessPoolExecutor(max_workers=len(JOBS)) as ex:
        futs = {ex.submit(bench_one, w): w[0] for w in work}
        for fut in as_completed(futs):
            label = futs[fut]
            r = fut.result()
            results.append(r)
            if "err" in r:
                print(f"  {label:34s} GPU{r.get('gpu','?')} ERR: {r['err']}  ({r.get('elapsed_s',0)}s)")
            else:
                print(f"  {label:34s} GPU{r['gpu']} {r['tflops']:>8.2f} TFLOPS "
                      f"gain={r['gain_vs_best_pct']:+6.2f}% vs_comp={r['vs_comp_pct']:6.2f}% "
                      f"finite={r['finite_frac']:.3f} nz={r['nz_frac']:.3f}  ({r['elapsed_s']}s)")
    out_json = os.path.join(SCRIPT_DIR, "R30_OPT_A_BENCH.json")
    with open(out_json, "w") as f:
        json.dump({"warmup": WARMUP, "iters": ITERS, "trim": TRIM, "results": results}, f, indent=2)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
