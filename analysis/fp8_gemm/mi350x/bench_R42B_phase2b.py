#!/usr/bin/env python3
"""R42 Opt B: bench the 2 K=28672 CRASH shapes against R42B cell variants.

Cells (per shape): fence0_po104, fence1_po104, fence0_po0, fence1_po0.

Bench rules: warmup=200, iters=500, trim=0.10 (per .claude/rules/benchmark-rules.md).
R39B random-scale gate: wcf < 2% AND finite >= 0.99 AND snr_med >= 10dB.
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
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_R42B_phase2b")


def _safe_tag(parent_tag):
    out = re.sub(r"(?<![A-Za-z0-9])memc(?![A-Za-z0-9])", "", parent_tag)
    out = re.sub(r"_memc(?=_|$)", "", out)
    out = re.sub(r"__+", "_", out)
    return out.strip("_")


WARMUP = 200
ITERS = 500
TRIM = 0.10
SHAPE_TIMEOUT = 700
SNR_THRESHOLD_DB = 10.0
WRONG_CELL_GATE  = 0.02
FINITE_GATE      = 0.99
RANDOM_SEED = 42

ALL_SHAPES = [
    (4096,  32768, 28672, 5568.2),
    (16384, 4096,  28672, 5525.3),
]

PARENT_VARIANT = "ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all"

ALL_CELLS = ["nf_R38B","nf_R38F2","nf_R38B_R38F2","nf_R38B_R39A","nf_R38B_R38F4"]


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


def find_so(m, n, k, cell):
    tag = _safe_tag(PARENT_VARIANT)
    mod = f"tk_mxfp4_gluon_cpp_n{n}_k{k}_{tag}_R42Bp2b_{cell}"
    path = os.path.join(BUILD_DIR, f"{mod}{EXT_SUFFIX}")
    if os.path.exists(path):
        return mod, path
    return None, None


def bench_one_shape(m, n, k, comp, cell, gpu_id):
    module_name, so_path = find_so(m, n, k, cell)

    if not so_path:
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None,
                "comp": comp, "status": "MISSING_SO", "kernel_finite": None,
                "best_variant": f"R42Bp2b_{cell}", "cell": cell}

    script = make_runner_script(module_name, so_path, m, n, k, comp)
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        r = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=SHAPE_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None,
                "comp": comp, "status": "TIMEOUT", "kernel_finite": None,
                "best_variant": f"R42Bp2b_{cell}", "cell": cell}

    if r.returncode != 0:
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None,
                "comp": comp, "status": "CRASH", "kernel_finite": None,
                "best_variant": f"R42Bp2b_{cell}", "cell": cell,
                "returncode": r.returncode,
                "stderr_tail": r.stderr[-300:]}

    try:
        s = r.stdout.index("BENCH_JSON_START") + len("BENCH_JSON_START")
        e = r.stdout.index("BENCH_JSON_END")
        out = json.loads(r.stdout[s:e].strip())
        out["best_variant"] = f"R42Bp2b_{cell}"
        out["cell"] = cell
        return out
    except (ValueError, json.JSONDecodeError):
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None,
                "comp": comp, "status": "PARSE_FAIL", "kernel_finite": None,
                "best_variant": f"R42Bp2b_{cell}", "cell": cell,
                "stdout_tail": r.stdout[-300:]}


def run_one_pass(gpus, run_label, shape_cells):
    print(f"R42B Bench [{run_label}] - GPUs: {gpus}, n_jobs={len(shape_cells)}")
    print(f"warmup={WARMUP}, iters={ITERS}, trim={TRIM}, "
          f"SNR>={SNR_THRESHOLD_DB}dB AND wrong_cells<{WRONG_CELL_GATE} AND "
          f"finite>={FINITE_GATE}, seed={RANDOM_SEED}")
    print("=" * 110)

    gpu_tasks = {g: [] for g in gpus}
    for i, item in enumerate(shape_cells):
        gpu = gpus[i % len(gpus)]
        gpu_tasks[gpu].append((i, item))

    results = [None] * len(shape_cells)
    lock = threading.Lock()
    t0 = time.time()

    def run_gpu(gpu_id):
        for (i, (m, n, k, comp, cell)) in gpu_tasks[gpu_id]:
            r = bench_one_shape(m, n, k, comp, cell, gpu_id)
            with lock:
                results[i] = r
                if r["status"] == "OK":
                    ratio = r["tflops"] / comp * 100
                    flag = "WIN" if r["tflops"] >= comp else "LOSE"
                    snr = r.get("snr_db"); snr_med = r.get("snr_med_db")
                    snr_s = f"{snr:5.1f}/{snr_med:5.1f}" if (snr is not None and snr_med is not None) else "  N/A"
                    fin_v = r.get('kernel_finite')
                    fin_s = f"{fin_v:.4f}" if fin_v is not None else "N/A"
                    wcf = r.get("wrong_cell_frac")
                    wcf_s = f"{wcf:.4f}" if wcf is not None else "N/A"
                    print(f"  [{i+1:>3}/{len(shape_cells)}] {m:>6}x{n:>6}x{k:>6}  "
                          f"{cell:<14s}  "
                          f"{r['tflops']:>7.1f} vs {comp:>7.1f} ({ratio:>5.1f}%) "
                          f"snr={snr_s}dB fin={fin_s} wcf={wcf_s} {flag} GPU{gpu_id}",
                          flush=True)
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
                    wcf_s = f"{wcf:.4f}" if wcf is not None else "N/A"
                    rc = r.get("returncode", "")
                    print(f"  [{i+1:>3}/{len(shape_cells)}] {m:>6}x{n:>6}x{k:>6}  "
                          f"{cell:<14s}  "
                          f"{r['status']} rc={rc} snr={snr_s} fin={fin_s} wcf={wcf_s} GPU{gpu_id}",
                          flush=True)

    threads = []
    for g in gpus:
        t = threading.Thread(target=run_gpu, args=(g,))
        t.start(); threads.append(t)
    for t in threads: t.join()

    elapsed = time.time() - t0
    return results, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="2,3")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default="R42_OPT_B_PHASE1_FENCE.json")
    ap.add_argument("--cells", default="all")
    ap.add_argument("--shape", default="all")
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]

    if args.cells == "all":
        cells = list(ALL_CELLS)
    else:
        cells = [c.strip() for c in args.cells.split(",")]

    if args.shape == "all":
        shapes = list(ALL_SHAPES)
    else:
        wanted = set()
        for s in args.shape.split(","):
            mnk = s.strip().split("x")
            wanted.add((int(mnk[0]), int(mnk[1]), int(mnk[2])))
        shapes = [s for s in ALL_SHAPES if (s[0], s[1], s[2]) in wanted]

    shape_cells = []
    for (m, n, k, comp) in shapes:
        for cell in cells:
            shape_cells.append((m, n, k, comp, cell))

    print(f"R42B bench: {len(shapes)} shapes x {len(cells)} cells = "
          f"{len(shape_cells)} jobs, runs={args.runs}")
    print(f"  shapes: {[(s[0],s[1],s[2]) for s in shapes]}")
    print(f"  cells:  {cells}")
    print(f"  gpus:   {gpus}")

    all_runs = []
    for ri in range(args.runs):
        results, elapsed = run_one_pass(gpus, f"run{ri+1}/{args.runs}", shape_cells)
        all_runs.append({"run": ri+1, "elapsed_minutes": round(elapsed/60, 1),
                         "results": results})

    consensus = []
    n_jobs = len(shape_cells)
    for i in range(n_jobs):
        per_run = [run["results"][i] for run in all_runs]
        oks = [r for r in per_run if r["status"] == "OK"]
        crashes = [r for r in per_run if r["status"] == "CRASH"]
        wrongs = [r for r in per_run if r["status"] == "WRONG_OUTPUT"]
        m, n_, k_, comp, cell = shape_cells[i]
        if len(oks) > len(per_run) // 2:
            oks_sorted = sorted(oks, key=lambda r: r["tflops"])
            chosen = dict(oks_sorted[len(oks_sorted) // 2])
            chosen["consensus"] = f"PASS_{len(oks)}/{args.runs}"
        elif len(crashes) > len(per_run) // 2:
            chosen = dict(crashes[0]); chosen["consensus"] = f"CRASH_{len(crashes)}/{args.runs}"
        elif len(wrongs) > len(per_run) // 2:
            chosen = dict(wrongs[0]); chosen["consensus"] = f"WRONG_{len(wrongs)}/{args.runs}"
        else:
            chosen = dict(per_run[0])
            chosen["consensus"] = (f"MIXED_oks={len(oks)}_wrongs={len(wrongs)}_crashes={len(crashes)}")
        wcfs = [r.get("wrong_cell_frac") for r in per_run]
        chosen["wcf_per_run"] = wcfs
        finites = [r.get("kernel_finite") for r in per_run]
        chosen["finite_per_run"] = finites
        statuses = [r.get("status") for r in per_run]
        chosen["status_per_run"] = statuses
        if any(r.get("status") == "OK" for r in per_run):
            tflops_per_run = [r.get("tflops") for r in per_run]
            chosen["tflops_per_run"] = tflops_per_run
        consensus.append(chosen)

    out_path = os.path.join(SCRIPT_DIR, args.out)
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "round": "R42_optB",
            "warmup": WARMUP, "iters": ITERS, "trim_frac": TRIM,
            "snr_threshold_db": SNR_THRESHOLD_DB,
            "wrong_cell_gate": WRONG_CELL_GATE,
            "finite_gate": FINITE_GATE,
            "random_seed": RANDOM_SEED,
            "n_shapes": len(shapes),
            "n_cells": len(cells),
            "n_runs": args.runs,
            "gpus": gpus,
            "shape_cells": [{"m": m, "n": n, "k": k, "comp": c, "cell": cell}
                            for (m, n, k, c, cell) in shape_cells],
            "consensus": consensus,
            "per_run": all_runs,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
