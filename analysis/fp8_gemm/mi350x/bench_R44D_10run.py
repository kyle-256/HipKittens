#!/usr/bin/env python3
"""R44 Opt D — 10-run INDEPENDENT consensus probe (random scale, fresh seed).

For 3 FIN_BOUND target shapes:
- 32768x4096x2048
- 16384x14336x2048
- 16384x28672x2048

Each shape is run 10 times. Each run uses an INDEPENDENT seed (RANDOM_SEED + run_idx)
so that A, B, sc_a, sc_b are freshly drawn each run. Per run we record fin_frac and
wcf. Verdict computed at two gates (FINITE_GATE = 0.97 and 0.98).

Bench params (mandatory): warmup=200, iters=500, trim=0.10.

GPU isolation: HIP_VISIBLE_DEVICES=6,7 (round-robin run → GPU).

Output JSON: R44_OPT_D_10RUN.json
"""
import json
import math
import os
import subprocess
import sys
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
RANDOM_SEED_BASE = 42  # actual seed = RANDOM_SEED_BASE + run_idx

# (M, N, K, competitor_TFLOPS)
TARGET_SHAPES = [
    (32768, 4096, 2048, 3131.8),
    (16384, 14336, 2048, 3301.3),
    (16384, 28672, 2048, 3482.3),
]

SHAPES_TO_SO = {
    "32768x4096x2048": "/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R41B/tk_mxfp4_gluon_cpp_n4096_k2048_ts_lgk2_gm6_v12_pfoff4_R41B_v3.cpython-310-x86_64-linux-gnu.so",
    "16384x14336x2048": "/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R40B/tk_mxfp4_gluon_cpp_n14336_k2048_ts_v12_gm7_pfoff4_kx2048_btw_all_R40B_safe.cpython-310-x86_64-linux-gnu.so",
    "16384x28672x2048": "/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R40B/tk_mxfp4_gluon_cpp_n28672_k2048_ts_lgk2_gm6_v12_pfoff4_R40B_safe.cpython-310-x86_64-linux-gnu.so",
}


def make_runner_script(module_name, so_path, m, n, k, comp, seed):
    return f"""\
import sys, math, json, importlib.util, torch
torch.manual_seed({seed})
WARMUP, ITERS, TRIM = {WARMUP}, {ITERS}, {TRIM}
SNR_DB_GATE = {SNR_THRESHOLD_DB}
WRONG_CELL_GATE = {WRONG_CELL_GATE}
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

    # Per-run we report kernel_finite, wcf, snr regardless of any gate; the verdict
    # is computed downstream at gate=0.97 and gate=0.98.
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


def bench_one_run(m, n, k, comp, so_path, gpu_id, seed):
    if not os.path.exists(so_path):
        return {"M": m, "N": n, "K": k, "status": "MISSING_SO"}

    module_name = os.path.basename(so_path).split(".")[0]
    script = make_runner_script(module_name, so_path, m, n, k, comp, seed)
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        r = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=SHAPE_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return {"M": m, "N": n, "K": k, "status": "TIMEOUT"}

    if r.returncode != 0:
        return {"M": m, "N": n, "K": k, "status": "CRASH",
                "stderr_tail": r.stderr[-200:]}

    try:
        s = r.stdout.index("BENCH_JSON_START") + len("BENCH_JSON_START")
        e = r.stdout.index("BENCH_JSON_END")
        return json.loads(r.stdout[s:e].strip())
    except (ValueError, json.JSONDecodeError):
        return {"M": m, "N": n, "K": k, "status": "PARSE_FAIL",
                "stdout_tail": r.stdout[-300:]}


def aggregate_at_gate(per_run, comp, finite_gate, wrong_cell_gate=WRONG_CELL_GATE):
    """Apply a (wcf < gate AND fin >= gate) check per run and count n_OK."""
    fins = [r.get("kernel_finite") for r in per_run if r.get("kernel_finite") is not None]
    wcfs = [r.get("wrong_cell_frac") for r in per_run if r.get("wrong_cell_frac") is not None]

    n_OK = 0
    for r in per_run:
        if r.get("status") != "OK":
            continue
        f = r.get("kernel_finite")
        w = r.get("wrong_cell_frac")
        if f is None or w is None:
            continue
        if f >= finite_gate and w < wrong_cell_gate:
            n_OK += 1

    def safe_max(xs): return max(xs) if xs else None
    def safe_min(xs): return min(xs) if xs else None
    def safe_mean(xs): return sum(xs) / len(xs) if xs else None
    def safe_std(xs):
        if not xs or len(xs) < 2: return 0.0
        m = sum(xs) / len(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    return {
        "finite_gate": finite_gate,
        "n_OK": n_OK,
        "n_runs": len(per_run),
        "fin_min": safe_min(fins),
        "fin_max": safe_max(fins),
        "fin_mean": safe_mean(fins),
        "fin_std": safe_std(fins),
        "wcf_max": safe_max(wcfs),
        "wcf_mean": safe_mean(wcfs),
        "wcf_std": safe_std(wcfs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="6,7")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--shapes-json", default=None,
                    help="Optional JSON path with shapes_to_so override (for crossval)")
    ap.add_argument("--shapes-list", default=None,
                    help="Optional comma-sep MxNxK,comp; pairs (overrides default)")
    ap.add_argument("--out", default="R44_OPT_D_10RUN.json")
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]

    if args.shapes_json:
        with open(args.shapes_json) as f:
            sj = json.load(f)
        # Expect {"shapes": [{"shape": "MxNxK", "comp": float, "so_path": "..."}]}
        target = []
        shapes_to_so = {}
        for s in sj["shapes"]:
            m, n, k = [int(x) for x in s["shape"].split("x")]
            target.append((m, n, k, s["comp"]))
            shapes_to_so[s["shape"]] = s["so_path"]
    else:
        target = TARGET_SHAPES
        shapes_to_so = SHAPES_TO_SO

    n_runs = args.runs
    print(f"R44 Opt D 10-run: {len(target)} shapes x {n_runs} runs, GPUs={gpus}")
    print(f"  warmup={WARMUP}, iters={ITERS}, trim={TRIM}, seed_base={RANDOM_SEED_BASE}")

    # Build job list: (shape_idx, run_idx, m, n, k, comp, so_path, seed)
    jobs = []
    for si, (m, n, k, comp) in enumerate(target):
        key = f"{m}x{n}x{k}"
        if key not in shapes_to_so:
            print(f"WARNING: missing so for {key}", file=sys.stderr)
            continue
        for ri in range(n_runs):
            seed = RANDOM_SEED_BASE + ri  # INDEPENDENT seed per run
            jobs.append((si, ri, m, n, k, comp, shapes_to_so[key], seed))

    # Round-robin across GPUs
    gpu_tasks = {g: [] for g in gpus}
    for i, j in enumerate(jobs):
        gpu_tasks[gpus[i % len(gpus)]].append((i, j))

    results = [None] * len(jobs)
    lock = threading.Lock()
    t0 = time.time()
    done = [0]

    def run_gpu(gpu_id):
        for (i, (si, ri, m, n, k, comp, so_path, seed)) in gpu_tasks[gpu_id]:
            r = bench_one_run(m, n, k, comp, so_path, gpu_id, seed)
            r["seed"] = seed
            r["run_idx"] = ri
            r["shape_idx"] = si
            with lock:
                results[i] = r
                done[0] += 1
                fin = r.get("kernel_finite")
                wcf = r.get("wrong_cell_frac")
                tflops = r.get("tflops")
                fin_s = f"{fin:.4f}" if fin is not None else "N/A"
                wcf_s = f"{wcf:.5f}" if wcf is not None else "N/A"
                tfl_s = f"{tflops:>7.1f}" if tflops is not None else "  N/A"
                print(f"  [{done[0]:>3}/{len(jobs)}] {m:>6}x{n:>6}x{k:>6} run{ri} seed{seed} "
                      f"{r['status']:>6s} tflops={tfl_s} fin={fin_s} wcf={wcf_s} GPU{gpu_id}",
                      flush=True)

    threads = []
    for g in gpus:
        t = threading.Thread(target=run_gpu, args=(g,))
        t.start(); threads.append(t)
    for t in threads: t.join()

    elapsed = time.time() - t0

    # Aggregate per shape
    consensus = {}
    for si, (m, n, k, comp) in enumerate(target):
        key = f"{m}x{n}x{k}"
        per_run = [r for r in results if r and r.get("shape_idx") == si]
        per_run.sort(key=lambda r: r.get("run_idx", 0))
        agg_097 = aggregate_at_gate(per_run, comp, 0.97)
        agg_098 = aggregate_at_gate(per_run, comp, 0.98)

        oks = [r for r in per_run if r.get("status") == "OK" and r.get("tflops") is not None]
        tflops_list = sorted([r["tflops"] for r in oks])
        tflops_p50 = tflops_list[len(tflops_list) // 2] if tflops_list else None

        consensus[key] = {
            "M": m, "N": n, "K": k, "comp": comp,
            "tflops_p50": tflops_p50,
            "pct_comp": (tflops_p50 / comp * 100) if (tflops_p50 and comp > 0) else None,
            "agg_gate_0.97": agg_097,
            "agg_gate_0.98": agg_098,
            "per_run": per_run,
        }

    out_path = os.path.join(SCRIPT_DIR, args.out)
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "round": "R44_OPT_D_10RUN",
            "warmup": WARMUP, "iters": ITERS, "trim_frac": TRIM,
            "wrong_cell_gate": WRONG_CELL_GATE,
            "random_seed_base": RANDOM_SEED_BASE,
            "n_shapes": len(target),
            "n_runs": n_runs,
            "gpus": gpus,
            "consensus": consensus,
            "elapsed_minutes": round(elapsed / 60, 2),
        }, f, indent=2)

    print(f"\n=== R44 Opt D 10-run summary (elapsed {elapsed/60:.1f} min) ===")
    print(f"{'shape':22s} {'src_runs':>8s} {'n_OK_098':>9s} {'n_OK_097':>9s} "
          f"{'fin_min':>8s} {'fin_med':>8s} {'wcf_max':>8s} {'tflops_p50':>10s}")
    for key, c in consensus.items():
        a98 = c['agg_gate_0.98']; a97 = c['agg_gate_0.97']
        fmin = a97.get('fin_min')
        fins = [r.get('kernel_finite') for r in c['per_run'] if r.get('kernel_finite') is not None]
        fmed = sorted(fins)[len(fins)//2] if fins else None
        tp = c['tflops_p50']
        print(f"  {key:22s} {a97['n_runs']:>8d} {a98['n_OK']:>9d} {a97['n_OK']:>9d} "
              f"{(fmin or 0):>8.4f} {(fmed or 0):>8.4f} {(a97.get('wcf_max') or 0):>8.5f} "
              f"{(tp or 0):>10.1f}")
    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
