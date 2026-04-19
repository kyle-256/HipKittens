#!/usr/bin/env python3
"""R41 Opt B: bench cluster-B near-gate shapes against R41B per-variant builds.

Sweep harness: for each of 11 cluster-B shapes, bench every R41B variant build
(v0a, v0b, v0c, v0d, v1, v2, v3) that exists in build_R41B/. Per-variant per-shape
PASS/FAIL gate uses random-scale + SNR + wcf gate (same as bench_all_42_R40B.py).

Bench parameters: warmup=200, iters=500, trim=0.10. 8 GPUs. seed=42.

Output:
- For --mode sweep (default): bench every (shape, variant) pair, write JSON of all
  results. n_runs reps per (shape,variant) → consensus PASS/FAIL.
- For --mode bestonly: read R41B_BEST_VARIANTS.json (selected per-shape best),
  bench just those at n_runs reps → final consensus JSON.
"""
import json
import os
import re
import subprocess
import sys
import sysconfig
import threading
import time
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_R41B")
MOD_PREFIX = "tk_mxfp4_gluon_cpp"

WARMUP = 200
ITERS = 500
TRIM = 0.10
SHAPE_TIMEOUT = 700
SNR_THRESHOLD_DB = 10.0
WRONG_CELL_GATE = 0.02
FINITE_GATE = 0.99
RANDOM_SEED = 42

# 11 cluster-B near-gate shapes (M, N, K, competitor_TFLOPS, parent_variant_tag)
CLUSTER_B_SHAPES = [
    (16384, 28672, 2048,  3482.3, "ts_lgk2_gm6_v12_memc_pfoff4"),
    (32768, 28672, 2048,  3353.4, "ts_lgk2_gm6_v12_memc_pfoff4"),
    (4096,  32768, 6144,  4548.6, "ts_v12_gm7_memc_pfoff19_kx6144_btw_all"),
    (4096,  32768, 14336, 5296.1, "ts_v12_tv0_memc_btw_all"),
    (4096,  32768, 128256, 5781.1,"ts_lgk2_v12_memc_btw_all"),
    (14336, 32768, 4096,  4462.6, "ts_lgk2_gm7_v12_memc_pfoff14"),
    (16384, 14336, 4096,  4255.8, "ts_gm7_v12_memc_dc_pfoff14"),
    (16384, 4096, 14336,  5142.1, "ts_gm8_v12_btw_all"),
    (28672, 4096,  8192,  4810.0, "ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all"),
    (28672, 4096, 16384,  5350.6, "ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all"),
    (32768, 4096,  2048,  3131.8, "ts_lgk2_gm6_v12_memc_pfoff4"),
]

VARIANT_IDS = ["v0a", "v0b", "v0c", "v0d", "v1", "v2", "v3"]


def _safe_tag(parent_tag):
    out = re.sub(r"(?<![A-Za-z0-9])memc(?![A-Za-z0-9])", "", parent_tag)
    out = re.sub(r"_memc(?=_|$)", "", out)
    out = re.sub(r"__+", "_", out)
    return out.strip("_")


def module_name_for(n_dim, k_dim, parent_tag, variant_id):
    base = _safe_tag(parent_tag)
    return f"{MOD_PREFIX}_n{n_dim}_k{k_dim}_{base}_R41B_{variant_id}"


def find_so(m, n, k, parent_tag, variant_id):
    mod = module_name_for(n, k, parent_tag, variant_id)
    path = os.path.join(BUILD_DIR, f"{mod}{EXT_SUFFIX}")
    if os.path.exists(path):
        return mod, path
    return None, None


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
        snr_db = float('-inf'); snr_med_db = float('-inf'); wrong_cell_frac = 1.0
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


def bench_one(m, n, k, comp, parent_tag, variant_id, gpu_id):
    module_name, so_path = find_so(m, n, k, parent_tag, variant_id)
    if not so_path:
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None, "comp": comp,
                "status": "MISSING_SO", "kernel_finite": None,
                "variant_id": variant_id, "parent_tag": parent_tag}

    script = make_runner_script(module_name, so_path, m, n, k, comp)
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        r = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=SHAPE_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None, "comp": comp,
                "status": "TIMEOUT", "kernel_finite": None,
                "variant_id": variant_id, "parent_tag": parent_tag}

    if r.returncode != 0:
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None, "comp": comp,
                "status": "CRASH", "kernel_finite": None,
                "variant_id": variant_id, "parent_tag": parent_tag,
                "stderr_tail": r.stderr[-200:]}

    try:
        s = r.stdout.index("BENCH_JSON_START") + len("BENCH_JSON_START")
        e = r.stdout.index("BENCH_JSON_END")
        out = json.loads(r.stdout[s:e].strip())
        out["variant_id"] = variant_id
        out["parent_tag"] = parent_tag
        return out
    except (ValueError, json.JSONDecodeError):
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None, "comp": comp,
                "status": "PARSE_FAIL", "kernel_finite": None,
                "variant_id": variant_id, "parent_tag": parent_tag,
                "stdout_tail": r.stdout[-200:]}


def run_sweep_pass(tasks, gpus, run_label):
    """tasks: list of (m, n, k, comp, parent_tag, variant_id) tuples."""
    print(f"R41 Opt B Sweep [{run_label}] - {len(tasks)} (shape,variant) pairs - GPUs: {gpus}")
    print(f"warmup={WARMUP}, iters={ITERS}, trim={TRIM}, "
          f"gate=wcf<{WRONG_CELL_GATE} AND snr_med>={SNR_THRESHOLD_DB}dB AND finite>={FINITE_GATE}, "
          f"seed={RANDOM_SEED}")
    print("=" * 110)

    gpu_tasks = {g: [] for g in gpus}
    for i, t in enumerate(tasks):
        gpu = gpus[i % len(gpus)]
        gpu_tasks[gpu].append((i, t))

    results = [None] * len(tasks)
    lock = threading.Lock()
    t0 = time.time()

    def run_gpu(gpu_id):
        for (i, (m, n, k, comp, parent_tag, vid)) in gpu_tasks[gpu_id]:
            r = bench_one(m, n, k, comp, parent_tag, vid, gpu_id)
            with lock:
                results[i] = r
                if r["status"] == "OK":
                    ratio = r["tflops"] / comp * 100
                    flag = "WIN" if r["tflops"] >= comp else "LOSE"
                    snr_med = r.get("snr_med_db")
                    fin_v = r.get("kernel_finite")
                    fin_s = f"{fin_v:.4f}" if fin_v is not None else "N/A"
                    snr_s = f"{snr_med:5.1f}" if snr_med is not None else "  N/A"
                    wcf = r.get("wrong_cell_frac", -1)
                    print(f"  [{i+1:>3}/{len(tasks)}] {m:>6}x{n:>6}x{k:>6} {vid:>4} "
                          f"{r['tflops']:>7.1f}/{comp:>7.1f} ({ratio:>5.1f}%) "
                          f"snr={snr_s} wcf={wcf:.4f} fin={fin_s} {flag} G{gpu_id}", flush=True)
                else:
                    fin = r.get("kernel_finite")
                    fin_s = f"{fin:.4f}" if fin is not None else "N/A"
                    wcf = r.get("wrong_cell_frac", -1)
                    print(f"  [{i+1:>3}/{len(tasks)}] {m:>6}x{n:>6}x{k:>6} {vid:>4} "
                          f"{r['status']} wcf={wcf} fin={fin_s} G{gpu_id}", flush=True)

    threads = []
    for g in gpus:
        t = threading.Thread(target=run_gpu, args=(g,))
        t.start(); threads.append(t)
    for t in threads: t.join()

    elapsed = time.time() - t0
    return results, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default="R41_OPT_B_BENCH_FULL.json")
    ap.add_argument("--smoke", action="store_true",
                    help="restrict to 3 representative shapes for smoke testing (single-run)")
    ap.add_argument("--only-shapes", default=None,
                    help="comma-sep list of MxNxK to restrict to")
    ap.add_argument("--only-variants", default=None,
                    help="comma-sep list of variant_ids to restrict to")
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]

    smoke_set = {(16384, 28672, 2048),
                 (4096, 32768, 128256),
                 (28672, 4096, 8192)}

    only_shapes = None
    if args.only_shapes:
        only_shapes = set()
        for s in args.only_shapes.split(","):
            parts = s.strip().split("x")
            only_shapes.add(tuple(int(p) for p in parts))

    only_variants = None
    if args.only_variants:
        only_variants = set(v.strip() for v in args.only_variants.split(","))

    tasks = []
    for (m, n, k, comp, parent_tag) in CLUSTER_B_SHAPES:
        if args.smoke and (m, n, k) not in smoke_set:
            continue
        if only_shapes and (m, n, k) not in only_shapes:
            continue
        for vid in VARIANT_IDS:
            if only_variants and vid not in only_variants:
                continue
            # Only schedule task if SO file exists.
            mod = module_name_for(n, k, parent_tag, vid)
            so_path = os.path.join(BUILD_DIR, f"{mod}{EXT_SUFFIX}")
            if not os.path.exists(so_path):
                continue
            tasks.append((m, n, k, comp, parent_tag, vid))

    runs_to_do = 1 if args.smoke else args.runs
    print(f"R41B bench: {len(tasks)} (shape,variant) tasks x {runs_to_do} runs")

    all_runs = []
    for ri in range(runs_to_do):
        results, elapsed = run_sweep_pass(tasks, gpus, f"run{ri+1}/{runs_to_do}")
        all_runs.append({"run": ri + 1, "elapsed_minutes": round(elapsed/60, 1),
                         "results": results})
        print(f"\n[run{ri+1}] elapsed: {elapsed/60:.1f} min")

    # Per-(shape,variant) consensus
    n = len(tasks)
    consensus = []
    for i in range(n):
        m, n_, k_, comp, parent_tag, vid = tasks[i]
        per_run = [run["results"][i] for run in all_runs if i < len(run["results"]) and run["results"][i] is not None]
        oks = [r for r in per_run if r["status"] == "OK"]
        wrongs = [r for r in per_run if r["status"] == "WRONG_OUTPUT"]
        if len(oks) > runs_to_do // 2:
            oks_sorted = sorted(oks, key=lambda r: r["tflops"])
            chosen = dict(oks_sorted[len(oks_sorted) // 2])
            chosen["consensus"] = f"PASS_{len(oks)}/{runs_to_do}"
        elif len(wrongs) > runs_to_do // 2:
            chosen = dict(wrongs[0])
            chosen["consensus"] = f"WRONG_{len(wrongs)}/{runs_to_do}"
        else:
            chosen = dict(per_run[0]) if per_run else {"M": m, "N": n_, "K": k_, "comp": comp,
                                                        "status": "NO_DATA", "variant_id": vid,
                                                        "parent_tag": parent_tag}
            chosen["consensus"] = f"MIXED_oks={len(oks)}_wrongs={len(wrongs)}"
        # Also record wcf stats across runs
        wcfs = [r.get("wrong_cell_frac", None) for r in per_run if r.get("wrong_cell_frac") is not None]
        if wcfs:
            chosen["wcf_max_across_runs"] = max(wcfs)
            chosen["wcf_min_across_runs"] = min(wcfs)
        consensus.append(chosen)

    # Pick best variant per shape (PASS only, max tflops)
    by_shape = {}
    for c in consensus:
        key = (c.get("M"), c.get("N"), c.get("K"))
        if key not in by_shape:
            by_shape[key] = []
        by_shape[key].append(c)

    best_per_shape = {}
    pass_count = 0
    for shape_key, candidates in by_shape.items():
        passes = [c for c in candidates if c.get("consensus", "").startswith("PASS_")]
        if passes:
            best = max(passes, key=lambda c: c.get("tflops") or 0)
            best_per_shape[f"{shape_key[0]}x{shape_key[1]}x{shape_key[2]}"] = best
            pass_count += 1
        else:
            best_per_shape[f"{shape_key[0]}x{shape_key[1]}x{shape_key[2]}"] = {
                "status": "NO_PASS",
                "candidates": [{"variant_id": c.get("variant_id"), "consensus": c.get("consensus"),
                                "wcf_max_across_runs": c.get("wcf_max_across_runs")} for c in candidates],
            }

    print()
    print("=" * 110)
    print(f"Per-shape PASSes: {pass_count}/{len(by_shape)}")

    out_path = os.path.join(SCRIPT_DIR, args.out)
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "round": "R41B",
            "warmup": WARMUP, "iters": ITERS, "trim_frac": TRIM,
            "snr_threshold_db": SNR_THRESHOLD_DB,
            "wrong_cell_gate": WRONG_CELL_GATE,
            "finite_gate": FINITE_GATE,
            "random_seed": RANDOM_SEED,
            "n_tasks": len(tasks),
            "n_runs": runs_to_do,
            "gpus": gpus,
            "consensus_per_pair": consensus,
            "best_per_shape": best_per_shape,
            "per_run": all_runs,
        }, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
