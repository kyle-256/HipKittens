#!/usr/bin/env python3
"""R40 Opt A: bench every shape's _R40A module with **random-scale + SNR** correctness gate.

Same harness as R39B, but points at build_R40A/ and `_R40A` module suffix. Uses
BEST_VARIANTS_V3 from R38_BEST_VARIANTS_v3 to determine the per-shape parent
tag and per-shape macro overrides — combined with R40A_PF_FENCE=1 layered on
every shape (matches what build_R40A.py emitted).

Bench parameters (mandatory): warmup=200, iters=500, trim=0.10
"""
import json
import os
import subprocess
import sys
import sysconfig
import threading
import time
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_R40A")

sys.path.insert(0, SCRIPT_DIR)
from R38_BEST_VARIANTS_v3 import BEST_VARIANTS_V3

WARMUP = 200
ITERS = 500
TRIM = 0.10
SHAPE_TIMEOUT = 700
SNR_THRESHOLD_DB = 10.0
WRONG_CELL_GATE  = 0.02
FINITE_GATE      = 0.99
RANDOM_SEED = 42

# (M, N, K, competitor_TFLOPS) — same 42 shapes
ALL_SHAPES = [
    (16384,  4096,  2048, 2995.0), (16384,  4096,  3072, 3492.3),
    (16384,  6144,  2048, 3047.6), (32768,  4096,  2048, 3131.8),
    (32768,  4096,  3072, 3630.6), (32768,  6144,  2048, 3239.9),
    (16384, 14336,  2048, 3301.3), (16384, 28672,  2048, 3482.3),
    (32768, 14336,  2048, 3351.4), (32768, 28672,  2048, 3353.4),
    (4096,   4096,  16384, 4642.1), (4096,  14336,  16384, 5013.0),
    (6144,   4096,  16384, 4428.1), (4096,   4096,   8192, 3959.9),
    (4096,   4096,  32768, 5152.8), (4096,   6144,  32768, 3784.2),
    (4096,  14336,   8192, 4345.8), (4096,  28672,  32768, 5649.9),
    (4096,  32768,  4096, 4166.5), (4096,  32768,  6144, 4548.6),
    (4096,  32768,  14336, 5296.1), (4096,  32768,  28672, 5568.2),
    (4096,  32768, 128256, 5781.1), (4096, 128256,  32768, 3195.3),
    (6144,   4096,   8192, 3822.0), (6144,  32768,   4096, 4291.0),
    (14336,  4096,  32768, 5245.4), (14336, 32768,   4096, 4462.6),
    (16384,  4096,  4096, 3951.8), (16384,  4096,  6144, 4259.9),
    (16384,  4096,  7168, 4443.2), (16384,  4096,  14336, 5142.1),
    (16384,  4096,  28672, 5525.3), (16384,  6144,   4096, 4042.5),
    (16384, 14336,   4096, 4255.8), (16384, 28672,   4096, 4411.7),
    (28672,  4096,   8192, 4810.0), (28672,  4096,  16384, 5350.6),
    (28672, 32768,   4096, 4466.6), (32768,  4096,   7168, 4666.8),
    (32768,  4096,  14336, 5223.4), (128256, 32768,  4096, 4536.4),
]


def macro_signature(macro_overrides):
    if not macro_overrides:
        return ""
    parts = [f"{k}{v}" for k, v in sorted(macro_overrides.items())]
    return "_" + "_".join(parts)


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


def find_so(m, n, k):
    """Locate the actual .so file for shape (m,n,k) in BUILD_DIR."""
    parent_tag, macro_overrides = BEST_VARIANTS_V3[(m, n, k)]
    # R40A always layers R40A_PF_FENCE=1 on top
    ovr = dict(macro_overrides)
    ovr["R40A_PF_FENCE"] = 1
    sig = macro_signature(ovr)
    mod = f"tk_mxfp4_gluon_cpp_n{n}_k{k}_{parent_tag}{sig}_R40A"
    path = os.path.join(BUILD_DIR, f"{mod}{EXT_SUFFIX}")
    if os.path.exists(path):
        return mod, path
    return None, None


def best_variant_label(m, n, k):
    parent_tag, macro_overrides = BEST_VARIANTS_V3[(m, n, k)]
    ovr = dict(macro_overrides)
    ovr["R40A_PF_FENCE"] = 1
    sig = macro_signature(ovr)
    return f"{parent_tag}{sig}_R40A"


def bench_one_shape(m, n, k, comp, gpu_id):
    label = best_variant_label(m, n, k)
    module_name, so_path = find_so(m, n, k)

    if not so_path:
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None,
                "comp": comp, "status": "MISSING_SO", "kernel_finite": None,
                "best_variant": label}

    script = make_runner_script(module_name, so_path, m, n, k, comp)
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        r = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=SHAPE_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None,
                "comp": comp, "status": "TIMEOUT", "kernel_finite": None,
                "best_variant": label}

    if r.returncode != 0:
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None,
                "comp": comp, "status": "CRASH", "kernel_finite": None,
                "best_variant": label,
                "stderr_tail": r.stderr[-200:]}

    try:
        s = r.stdout.index("BENCH_JSON_START") + len("BENCH_JSON_START")
        e = r.stdout.index("BENCH_JSON_END")
        out = json.loads(r.stdout[s:e].strip())
        out["best_variant"] = label
        return out
    except (ValueError, json.JSONDecodeError):
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None,
                "comp": comp, "status": "PARSE_FAIL", "kernel_finite": None,
                "best_variant": label,
                "stdout_tail": r.stdout[-200:]}


def run_one_pass(gpus, shapes, run_label):
    print(f"R40 Opt A Random-Scale + SNR Bench [{run_label}] - GPUs: {gpus}")
    print(f"Shapes: {len(shapes)}, warmup={WARMUP}, iters={ITERS}, trim={TRIM}, "
          f"SNR>={SNR_THRESHOLD_DB}dB AND wrong_cells<{WRONG_CELL_GATE} AND finite>={FINITE_GATE}, "
          f"seed={RANDOM_SEED}")
    print("=" * 110)

    gpu_tasks = {g: [] for g in gpus}
    for i, (m, n, k, comp) in enumerate(shapes):
        gpu = gpus[i % len(gpus)]
        gpu_tasks[gpu].append((i, m, n, k, comp))

    results = [None] * len(shapes)
    lock = threading.Lock()
    t0 = time.time()

    def run_gpu(gpu_id):
        for (i, m, n, k, comp) in gpu_tasks[gpu_id]:
            r = bench_one_shape(m, n, k, comp, gpu_id)
            with lock:
                results[i] = r
                if r["status"] == "OK":
                    ratio = r["tflops"] / comp * 100
                    flag = "WIN" if r["tflops"] >= comp else "LOSE"
                    snr = r.get("snr_db")
                    snr_med = r.get("snr_med_db")
                    if snr is not None and snr_med is not None:
                        snr_s = f"{snr:5.1f}/{snr_med:5.1f}"
                    else:
                        snr_s = "  N/A"
                    fin_v = r.get('kernel_finite')
                    fin_s = f"{fin_v:.4f}" if fin_v is not None else "N/A"
                    print(f"  [{i+1:>2}/{len(shapes)}] {m:>6}x{n:>6}x{k:>6}  "
                          f"{r['tflops']:>7.1f} vs {comp:>7.1f} ({ratio:>5.1f}%) "
                          f"snr={snr_s}dB fin={fin_s} {flag} GPU{gpu_id}", flush=True)
                else:
                    fin = r.get("kernel_finite")
                    fin_s = f"{fin:.4f}" if fin is not None else "N/A"
                    snr = r.get("snr_db"); snr_med = r.get("snr_med_db")
                    if snr is not None and snr_med is not None:
                        snr_s = f"{snr:.1f}/{snr_med:.1f}"
                    elif snr is not None:
                        snr_s = f"{snr:.1f}/?"
                    else:
                        snr_s = "?"
                    wcf = r.get("wrong_cell_frac")
                    wcf_s = f"wcf={wcf:.4f}" if wcf is not None else ""
                    print(f"  [{i+1:>2}/{len(shapes)}] {m:>6}x{n:>6}x{k:>6}  "
                          f"{r['status']} snr={snr_s} fin={fin_s} {wcf_s} GPU{gpu_id}", flush=True)

    threads = []
    for g in gpus:
        t = threading.Thread(target=run_gpu, args=(g,))
        t.start(); threads.append(t)
    for t in threads: t.join()

    elapsed = time.time() - t0
    return results, elapsed


def summarize(results):
    wins = losses = wrong = errs = 0
    for r in results:
        if r is None: errs += 1; continue
        if r["status"] == "OK":
            if r["tflops"] >= r["comp"]: wins += 1
            else: losses += 1
        elif r["status"] == "WRONG_OUTPUT": wrong += 1
        else: errs += 1
    return wins, losses, wrong, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default="bench_all42_results_R40_optA.json")
    ap.add_argument("--only", default="", help="comma list of M,N,K triples to filter; e.g. 16384,6144,2048;6144,32768,4096")
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]

    if args.only:
        only = set()
        for t in args.only.split(";"):
            m, n, k = [int(x) for x in t.split(",")]
            only.add((m, n, k))
        shapes = [s for s in ALL_SHAPES if (s[0], s[1], s[2]) in only]
    else:
        shapes = ALL_SHAPES

    all_runs = []
    for ri in range(args.runs):
        results, elapsed = run_one_pass(gpus, shapes, f"run{ri+1}/{args.runs}")
        wins, losses, wrong, errs = summarize(results)
        print()
        print("=" * 110)
        print(f"[run{ri+1}] WIN: {wins}/{len(shapes)}  LOSE: {losses}/{len(shapes)}  WRONG: {wrong}/{len(shapes)}  ERR: {errs}/{len(shapes)}  ({elapsed/60:.1f} min)")
        all_runs.append({"run": ri+1, "wins": wins, "losses": losses, "wrong_output": wrong,
                         "errors": errs, "elapsed_minutes": round(elapsed/60, 1),
                         "results": results})

    n = len(shapes)
    consensus = []
    placeholder = lambda m,n_,k_,c: {"M": m, "N": n_, "K": k_, "tflops": None,
                                     "avg_ms": None, "comp": c, "status": "NO_DATA",
                                     "kernel_finite": None}
    for i in range(n):
        m, n_, k_, comp = shapes[i]
        per_run = [(run["results"][i] if (i < len(run["results"]) and run["results"][i] is not None)
                    else placeholder(m, n_, k_, comp))
                   for run in all_runs]
        oks = [r for r in per_run if r["status"] == "OK"]
        wrongs = [r for r in per_run if r["status"] == "WRONG_OUTPUT"]
        if len(oks) > len(per_run) // 2:
            oks_sorted = sorted(oks, key=lambda r: r["tflops"])
            chosen = dict(oks_sorted[len(oks_sorted) // 2])
            chosen["consensus"] = f"PASS_{len(oks)}/{args.runs}"
            consensus.append(chosen)
        elif len(wrongs) > len(per_run) // 2:
            chosen = dict(wrongs[0]); chosen["consensus"] = f"WRONG_{len(wrongs)}/{args.runs}"
            consensus.append(chosen)
        else:
            chosen = dict(per_run[0]); chosen["consensus"] = f"MIXED_oks={len(oks)}_wrongs={len(wrongs)}"
            consensus.append(chosen)

    cwins = closses = cwrong = cerrs = 0
    for r in consensus:
        if r and r["status"] == "OK":
            if r["tflops"] >= r["comp"]: cwins += 1
            else: closses += 1
        elif r and r["status"] == "WRONG_OUTPUT": cwrong += 1
        else: cerrs += 1

    print()
    print("=" * 110)
    print(f"CONSENSUS  WIN: {cwins}/{n}  LOSE_CORRECT: {closses}/{n}  WRONG: {cwrong}/{n}  ERR: {cerrs}/{n}")
    print(f"           VERIFIED-CORRECT: {cwins + closses}/{n}")

    out_path = os.path.join(SCRIPT_DIR, args.out)
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "round": "R40_optA",
            "warmup": WARMUP, "iters": ITERS, "trim_frac": TRIM,
            "snr_threshold_db": SNR_THRESHOLD_DB,
            "finite_gate": FINITE_GATE,
            "random_seed": RANDOM_SEED,
            "n_shapes": n,
            "n_runs": args.runs,
            "consensus_wins": cwins,
            "consensus_losses_correct": closses,
            "consensus_wrong": cwrong,
            "consensus_errors": cerrs,
            "verified_correct": cwins + closses,
            "gpus": gpus,
            "consensus_results": consensus,
            "per_run": all_runs,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
