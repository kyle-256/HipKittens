#!/usr/bin/env python3
"""R41 Integration: bench all 42 shapes using R41_INTEGRATION_MANIFEST.json.

The manifest dictates per-shape .so file (combining R41A / R41B / R40A / R40B).
This harness loads each .so directly and runs the R39B random-scale gate
(warmup=200, iters=500, trim=0.10, seed=42, gates: wcf<2%, snr_med>=10dB,
finite>=0.99).

Output: R41_INTEGRATION_5RUN.json with per-shape consensus across N runs:
  tflops_p50, wcf_max, wcf_mean, wcf_std, fin_min, n_OK_<runs>, verdict.
"""
import json
import math
import os
import subprocess
import sys
import sysconfig
import threading
import time
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

WARMUP = 200
ITERS = 500
TRIM = 0.10
SHAPE_TIMEOUT = 700
SNR_THRESHOLD_DB = 10.0
WRONG_CELL_GATE = 0.02
FINITE_GATE = 0.99
RANDOM_SEED = 42

# (M, N, K, competitor_TFLOPS) — same 42 shapes as R37/R40B
ALL_SHAPES = [
    (16384, 4096, 2048, 2995.0), (16384, 4096, 3072, 3492.3),
    (16384, 6144, 2048, 3047.6), (32768, 4096, 2048, 3131.8),
    (32768, 4096, 3072, 3630.6), (32768, 6144, 2048, 3239.9),
    (16384, 14336, 2048, 3301.3), (16384, 28672, 2048, 3482.3),
    (32768, 14336, 2048, 3351.4), (32768, 28672, 2048, 3353.4),
    (4096, 4096, 16384, 4642.1), (4096, 14336, 16384, 5013.0),
    (6144, 4096, 16384, 4428.1), (4096, 4096, 8192, 3959.9),
    (4096, 4096, 32768, 5152.8), (4096, 6144, 32768, 3784.2),
    (4096, 14336, 8192, 4345.8), (4096, 28672, 32768, 5649.9),
    (4096, 32768, 4096, 4166.5), (4096, 32768, 6144, 4548.6),
    (4096, 32768, 14336, 5296.1), (4096, 32768, 28672, 5568.2),
    (4096, 32768, 128256, 5781.1), (4096, 128256, 32768, 3195.3),
    (6144, 4096, 8192, 3822.0), (6144, 32768, 4096, 4291.0),
    (14336, 4096, 32768, 5245.4), (14336, 32768, 4096, 4462.6),
    (16384, 4096, 4096, 3951.8), (16384, 4096, 6144, 4259.9),
    (16384, 4096, 7168, 4443.2), (16384, 4096, 14336, 5142.1),
    (16384, 4096, 28672, 5525.3), (16384, 6144, 4096, 4042.5),
    (16384, 14336, 4096, 4255.8), (16384, 28672, 4096, 4411.7),
    (28672, 4096, 8192, 4810.0), (28672, 4096, 16384, 5350.6),
    (28672, 32768, 4096, 4466.6), (32768, 4096, 7168, 4666.8),
    (32768, 4096, 14336, 5223.4), (128256, 32768, 4096, 4536.4),
]


def make_runner_script(module_name, so_path, m, n, k, comp):
    return f"""\
import sys, math, json, importlib.util, torch
torch.manual_seed({RANDOM_SEED})
WARMUP, ITERS, TRIM = {WARMUP}, {ITERS}, {TRIM}
SNR_DB_GATE = {SNR_THRESHOLD_DB}
WRONG_CELL_GATE = {WRONG_CELL_GATE}
FINITE_GATE = {FINITE_GATE}
M, N, K = {m}, {n}, {k}
COMP = {comp}

FP4_TBL = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0,
], dtype=torch.float32, device='cuda')

def gen_fp4(r, K):
    c = K // 2
    return (torch.randint(0,16,(r,c),dtype=torch.uint8,device='cuda') << 4) | \\
           torch.randint(0,16,(r,c),dtype=torch.uint8,device='cuda')

def preshuffle(se):
    r, kb = se.shape
    pr = math.ceil(r/64)*64
    pk = math.ceil(kb/8)*8
    raw = torch.full((pr,pk),0x7F,dtype=torch.uint8,device=se.device)
    raw[:r,:kb] = (se.to(torch.int16)+127).to(torch.uint8)
    sh = raw.view(pr//32,2,16,pk//8,2,4,1).permute(0,3,5,2,4,1,6).contiguous().view(pr//32,pk*32)
    sh = sh.view(pr//64,2,pk*32//4,4).permute(0,2,1,3).contiguous()
    return sh.view(pr//64,pk*64)

def dequant_fp4_rows(packed_uint8, K, row_lo, row_hi):
    pk = packed_uint8[row_lo:row_hi]
    lo = (pk & 0x0F).to(torch.int64)
    hi = ((pk >> 4) & 0x0F).to(torch.int64)
    rows, cols = pk.shape
    out = torch.empty(rows, cols * 2, dtype=torch.float32, device=pk.device)
    out[:, 0::2] = FP4_TBL[lo]
    out[:, 1::2] = FP4_TBL[hi]
    return out[:, :K]

def apply_scales_rows(data_f32, scale_exp_i8, K_dim, row_lo, row_hi, block=32):
    sc = scale_exp_i8[row_lo:row_hi].to(torch.float32)
    scales = torch.pow(2.0, sc)
    rows, kb = scales.shape
    se = scales.unsqueeze(-1).expand(rows, kb, block).reshape(rows, -1)[:, :K_dim]
    return data_f32 * se

try:
    spec = importlib.util.spec_from_file_location({module_name!r}, {so_path!r})
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    A = gen_fp4(M, K); B = gen_fp4(N, K)
    sc_a = torch.randint(-2, 3, (M, K//32), dtype=torch.int8, device='cuda')
    sc_b = torch.randint(-2, 3, (N, K//32), dtype=torch.int8, device='cuda')
    A_sc = preshuffle(sc_a); B_sc = preshuffle(sc_b)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

    run = lambda: mod.gemm_rcr(A, B, A_sc, B_sc, C)
    C.zero_(); run(); torch.cuda.synchronize()
    C0 = C.clone()
    finite_frac = float(torch.isfinite(C0.float()).sum().item()) / float(C0.numel())
    finite_frac = round(finite_frac, 6)

    MAX_M_ROWS = 1024
    MAX_N_ROWS = 1024
    m_rows = min(M, MAX_M_ROWS)
    n_rows = min(N, MAX_N_ROWS)

    A_blk = dequant_fp4_rows(A, K, 0, m_rows)
    B_blk = dequant_fp4_rows(B, K, 0, n_rows)
    A_scaled = apply_scales_rows(A_blk, sc_a, K, 0, m_rows)
    B_scaled = apply_scales_rows(B_blk, sc_b, K, 0, n_rows)
    del A_blk, B_blk

    C_ref = torch.matmul(A_scaled, B_scaled.T)
    del A_scaled, B_scaled

    C_test = C0[:m_rows, :n_rows].float()
    base_mask = torch.isfinite(C_ref) & torch.isfinite(C_test)

    REL_TOL = 0.05
    ABS_TOL = 1e-2
    n_valid = int(base_mask.sum().item())

    if n_valid < 1024:
        snr_db = float('-inf')
        snr_med_db = float('-inf')
        wrong_cell_frac = 1.0
    else:
        ref_d = C_ref.double()
        test_d = C_test.double()
        abs_diff = (test_d - ref_d).abs()
        abs_ref = ref_d.abs()
        abs_test = test_d.abs()
        catastrophic = (abs_test > 100.0 * abs_ref + 1e-3) & (abs_diff > REL_TOL * abs_ref + ABS_TOL)
        catastrophic = catastrophic & base_mask
        wrong_cell_frac = float(catastrophic.sum().item()) / float(base_mask.sum().item())

        good_mask = base_mask & ~catastrophic
        n_good = int(good_mask.sum().item())
        if n_good < 1024:
            snr_db = float('-inf'); snr_med_db = float('-inf')
        else:
            rv = ref_d[good_mask]; tv = test_d[good_mask]
            sig = float((rv * rv).mean().item())
            err = float(((tv - rv) ** 2).mean().item())
            snr_db = 10.0 * math.log10(sig / err) if (sig > 0.0 and err > 0.0) else float('-inf')

            row_n = good_mask.sum(dim=1).clamp_min(1)
            row_sig = (ref_d ** 2 * good_mask).sum(dim=1) / row_n
            row_err = ((test_d - ref_d) ** 2 * good_mask).sum(dim=1) / row_n
            valid_rows = good_mask.any(dim=1) & (row_sig > 0) & (row_err > 0)
            if int(valid_rows.sum().item()) < 8:
                snr_med_db = float('-inf')
            else:
                row_snr = 10.0 * torch.log10(row_sig[valid_rows] / row_err[valid_rows])
                snr_med_db = float(row_snr.median().item())

    correct = ((snr_med_db >= SNR_DB_GATE)
               and (wrong_cell_frac < WRONG_CELL_GATE)
               and (finite_frac >= FINITE_GATE))

    if not correct:
        out = {{
            "M": M, "N": N, "K": K, "tflops": None, "avg_ms": None, "comp": COMP,
            "status": "WRONG_OUTPUT", "kernel_finite": finite_frac,
            "snr_db": snr_db if math.isfinite(snr_db) else None,
            "snr_med_db": snr_med_db if math.isfinite(snr_med_db) else None,
            "snr_n_valid": n_valid,
            "wrong_cell_frac": round(wrong_cell_frac, 6),
        }}
        print("BENCH_JSON_START"); print(json.dumps(out)); print("BENCH_JSON_END")
        sys.exit(0)

    for _ in range(WARMUP): run()
    torch.cuda.synchronize()

    times = []
    for _ in range(ITERS):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); run(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    times.sort()
    trim = int(len(times) * TRIM)
    if trim > 0: times = times[trim:-trim]
    avg = sum(times) / len(times)
    tflops = 2.0 * M * N * K / (avg * 1e-3) / 1e12

    out = {{
        "M": M, "N": N, "K": K, "tflops": round(tflops, 1),
        "avg_ms": round(avg, 4), "comp": COMP, "status": "OK",
        "kernel_finite": finite_frac,
        "snr_db": round(snr_db, 2) if math.isfinite(snr_db) else None,
        "snr_med_db": round(snr_med_db, 2) if math.isfinite(snr_med_db) else None,
        "snr_n_valid": n_valid,
        "wrong_cell_frac": round(wrong_cell_frac, 6),
    }}
    print("BENCH_JSON_START"); print(json.dumps(out)); print("BENCH_JSON_END")

except torch.cuda.OutOfMemoryError:
    out = {{"M": M, "N": N, "K": K, "tflops": None, "avg_ms": None,
           "comp": COMP, "status": "OOM", "kernel_finite": None}}
    print("BENCH_JSON_START"); print(json.dumps(out)); print("BENCH_JSON_END")
except Exception as e:
    out = {{"M": M, "N": N, "K": K, "tflops": None, "avg_ms": None,
           "comp": COMP, "status": f"ERR:{{e}}"[:120], "kernel_finite": None}}
    print("BENCH_JSON_START"); print(json.dumps(out)); print("BENCH_JSON_END")
"""


def bench_one_shape(m, n, k, comp, so_path, source, gpu_id):
    if not os.path.exists(so_path):
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None,
                "comp": comp, "status": "MISSING_SO", "kernel_finite": None,
                "source": source, "so_path": so_path}

    module_name = os.path.basename(so_path).split(".")[0]
    script = make_runner_script(module_name, so_path, m, n, k, comp)
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        r = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=SHAPE_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None,
                "comp": comp, "status": "TIMEOUT", "kernel_finite": None,
                "source": source}

    if r.returncode != 0:
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None,
                "comp": comp, "status": "CRASH", "kernel_finite": None,
                "source": source, "stderr_tail": r.stderr[-200:]}

    try:
        s = r.stdout.index("BENCH_JSON_START") + len("BENCH_JSON_START")
        e = r.stdout.index("BENCH_JSON_END")
        out = json.loads(r.stdout[s:e].strip())
        out["source"] = source
        return out
    except (ValueError, json.JSONDecodeError):
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None,
                "comp": comp, "status": "PARSE_FAIL", "kernel_finite": None,
                "source": source, "stdout_tail": r.stdout[-200:]}


def run_one_pass(gpus, run_label, jobs):
    """jobs: list of (m,n,k,comp,so_path,source) tuples."""
    print(f"R41 INTEG Bench [{run_label}] - GPUs: {gpus}, n_jobs={len(jobs)}")
    print("=" * 110)

    gpu_tasks = {g: [] for g in gpus}
    for i, item in enumerate(jobs):
        gpu = gpus[i % len(gpus)]
        gpu_tasks[gpu].append((i, item))

    results = [None] * len(jobs)
    lock = threading.Lock()
    t0 = time.time()
    done = [0]

    def run_gpu(gpu_id):
        for (i, (m, n, k, comp, so_path, source)) in gpu_tasks[gpu_id]:
            r = bench_one_shape(m, n, k, comp, so_path, source, gpu_id)
            with lock:
                results[i] = r
                done[0] += 1
                if r["status"] == "OK":
                    ratio = r["tflops"] / comp * 100
                    flag = "WIN" if r["tflops"] >= comp else "LOSE"
                    fin = r.get("kernel_finite")
                    fin_s = f"{fin:.4f}" if fin is not None else "N/A"
                    wcf = r.get("wrong_cell_frac")
                    wcf_s = f"{wcf:.4f}" if wcf is not None else "N/A"
                    print(f"  [{done[0]:>3}/{len(jobs)}] {m:>6}x{n:>6}x{k:>6} [{source}] "
                          f"{r['tflops']:>7.1f} vs {comp:>7.1f} ({ratio:>5.1f}%) "
                          f"fin={fin_s} wcf={wcf_s} {flag} GPU{gpu_id}",
                          flush=True)
                else:
                    fin = r.get("kernel_finite")
                    fin_s = f"{fin:.4f}" if fin is not None else "N/A"
                    wcf = r.get("wrong_cell_frac")
                    wcf_s = f"{wcf:.4f}" if wcf is not None else "N/A"
                    print(f"  [{done[0]:>3}/{len(jobs)}] {m:>6}x{n:>6}x{k:>6} [{source}] "
                          f"{r['status']} fin={fin_s} wcf={wcf_s} GPU{gpu_id}",
                          flush=True)

    threads = []
    for g in gpus:
        t = threading.Thread(target=run_gpu, args=(g,))
        t.start(); threads.append(t)
    for t in threads: t.join()

    elapsed = time.time() - t0
    return results, elapsed


def aggregate_consensus(per_run_results, n_runs, comp):
    """Compute 5-run consensus per shape."""
    oks = [r for r in per_run_results if r["status"] == "OK"]
    n_OK = len(oks)
    wcfs = [r.get("wrong_cell_frac") for r in per_run_results if r.get("wrong_cell_frac") is not None]
    fins = [r.get("kernel_finite") for r in per_run_results if r.get("kernel_finite") is not None]

    def safe_max(xs): return max(xs) if xs else None
    def safe_mean(xs): return sum(xs) / len(xs) if xs else None
    def safe_min(xs): return min(xs) if xs else None
    def safe_std(xs):
        if not xs or len(xs) < 2: return 0.0
        m = sum(xs) / len(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    if oks:
        tflops_list = sorted([r["tflops"] for r in oks])
        tflops_p50 = tflops_list[len(tflops_list) // 2]
    else:
        tflops_p50 = None

    wcf_max = safe_max(wcfs)
    wcf_mean = safe_mean(wcfs)
    wcf_std = safe_std(wcfs)
    fin_min = safe_min(fins)

    # Verdict
    # PASS_n/N if n_OK >= ceil(N/2) (majority);
    # Strict verified-correct = n_OK >= 3 AND wcf_max < 0.02 AND wcf_std < 0.01 AND fin_min >= 0.99
    verified_correct = (
        n_OK >= 3
        and wcf_max is not None and wcf_max < WRONG_CELL_GATE
        and wcf_std < 0.01
        and fin_min is not None and fin_min >= FINITE_GATE
    )

    if n_OK == n_runs:
        verdict = f"PASS_{n_OK}/{n_runs}"
    elif n_OK >= 3:
        verdict = f"PASS_{n_OK}/{n_runs}"
    elif n_OK >= 1:
        verdict = f"FLAKE_{n_OK}/{n_runs}"
    else:
        # Determine dominant fail mode
        wrong = sum(1 for r in per_run_results if r["status"] == "WRONG_OUTPUT")
        if wrong > 0:
            verdict = f"WRONG_{wrong}/{n_runs}"
        else:
            statuses = [r["status"] for r in per_run_results]
            verdict = f"FAIL_{statuses[0]}"

    return {
        "tflops_p50": tflops_p50,
        "wcf_max": wcf_max,
        "wcf_mean": wcf_mean,
        "wcf_std": wcf_std,
        "fin_min": fin_min,
        "n_OK_5": n_OK,
        "verdict": verdict,
        "verified_correct": verified_correct,
        "comp": comp,
        "pct_comp": (tflops_p50 / comp * 100) if (tflops_p50 is not None and comp > 0) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--manifest", default="R41_INTEGRATION_MANIFEST.json")
    ap.add_argument("--out", default="R41_INTEGRATION_5RUN.json")
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]

    manifest_path = os.path.join(SCRIPT_DIR, args.manifest)
    with open(manifest_path) as f:
        manifest = json.load(f)

    shapes_to_so = manifest["shapes_to_so_path"]
    shapes_to_source = manifest["shapes_to_source"]

    jobs = []
    for (m, n, k, comp) in ALL_SHAPES:
        key = f"{m}x{n}x{k}"
        if key not in shapes_to_so:
            print(f"WARNING: shape {key} missing from manifest", file=sys.stderr)
            continue
        jobs.append((m, n, k, comp, shapes_to_so[key], shapes_to_source[key]))

    print(f"R41 INTEGRATION bench: {len(jobs)} shapes, runs={args.runs}, GPUs={gpus}")
    print(f"  warmup={WARMUP}, iters={ITERS}, trim={TRIM}, seed={RANDOM_SEED}")
    print(f"  gates: snr_med>={SNR_THRESHOLD_DB}dB, wcf<{WRONG_CELL_GATE}, fin>={FINITE_GATE}")

    all_runs = []
    for ri in range(args.runs):
        results, elapsed = run_one_pass(gpus, f"run{ri+1}/{args.runs}", jobs)
        all_runs.append({"run": ri + 1, "elapsed_minutes": round(elapsed / 60, 1),
                         "results": results})
        print(f"\n[run{ri+1}/{args.runs}] elapsed={elapsed/60:.1f} min")

    # Per-shape consensus
    consensus = {}
    for i, (m, n, k, comp, so_path, source) in enumerate(jobs):
        key = f"{m}x{n}x{k}"
        per_shape_runs = [run["results"][i] for run in all_runs]
        agg = aggregate_consensus(per_shape_runs, args.runs, comp)
        agg["M"], agg["N"], agg["K"] = m, n, k
        agg["source"] = source
        agg["so_path"] = so_path
        agg["per_run"] = per_shape_runs
        consensus[key] = agg

    out_path = os.path.join(SCRIPT_DIR, args.out)
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "round": "R41_INTEGRATION",
            "warmup": WARMUP, "iters": ITERS, "trim_frac": TRIM,
            "snr_threshold_db": SNR_THRESHOLD_DB,
            "wrong_cell_gate": WRONG_CELL_GATE,
            "finite_gate": FINITE_GATE,
            "random_seed": RANDOM_SEED,
            "n_shapes": len(jobs),
            "n_runs": args.runs,
            "gpus": gpus,
            "manifest": args.manifest,
            "consensus": consensus,
            "per_run_summary": [
                {"run": r["run"], "elapsed_minutes": r["elapsed_minutes"]}
                for r in all_runs
            ],
        }, f, indent=2)

    n_verified = sum(1 for v in consensus.values() if v["verified_correct"])
    n_win = sum(1 for v in consensus.values()
                if v["pct_comp"] is not None and v["pct_comp"] >= 100.0)

    print(f"\n=== R41 INTEGRATION 5-RUN SUMMARY ===")
    print(f"Verified-correct: {n_verified}/{len(jobs)}")
    print(f"WIN (>=100% comp): {n_win}/{len(jobs)}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
