#!/usr/bin/env python3
"""R55 Opt E3_3 — bench harness for the aiter `.co` dlopen shim on (16384, 28672, 2048).

Reuses R50D shim AS-IS (256x256 tile, bdx=256). Only shape constants change.
Cohort-race rescue candidate: R54 cohort-race rescue candidate (HK FAIL on R40B); see R55_DECIDER_PLAN.md E-3.

Modes:
  --mode smoke    1 seed, full correctness gate + perf measurement
  --mode 10run    10 INDEPENDENT seeds, n_OK_5 / wcf_max / wcf_std / fin_min stats

Bench MANDATORY: warmup=200, iters=500, trim=0.10.
Verified-correct gate: kernel_finite >= 0.97, wcf_max < 0.02, snr_med_db >= 10.
10-run @ 80% gate: n_OK_5 >= 8/10 + wcf_max < 0.02 + wcf_std < 0.01 + fin_min >= 0.97.

Per-shape grid (computed inside shim from M,N,K and tile sizes):
  gdx = ceil(N/256) = ceil(28672/256) = 112
  gdy = ceil(M/256) = ceil(16384/256) = 64
  gdz = 1, bdx = 256
"""
import argparse
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHIM_SO = os.path.join(SCRIPT_DIR, "build_R50D",
                       "R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so")
AITER_CO = ("/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/"
            "f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co")
KERNEL_NAME = "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E"

WARMUP = 200
ITERS = 500
TRIM_FRAC = 0.10
SNR_DB_GATE = 10.0
WCF_GATE = 0.02
FINITE_GATE = 0.97
M, N, K = 16384, 28672, 2048
COMPETITOR_TFLOPS = 3482.3  # from R54_INTEGRATION_10RUN.json consensus for (16384, 28672, 2048)


def make_runner_script(seed: int, do_perf: bool) -> str:
    return f"""\
import sys, os, math, json, importlib.util
import torch
torch.manual_seed({seed})

# Load aiter (for quant + shuffle utilities)
import aiter
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import dtypes, fp4_utils

SHIM_PATH = {SHIM_SO!r}
AITER_CO  = {AITER_CO!r}
KERNEL_NAME = {KERNEL_NAME!r}
M, N, K = {M}, {N}, {K}
WARMUP, ITERS, TRIM_FRAC = {WARMUP}, {ITERS}, {TRIM_FRAC}
SNR_DB_GATE = {SNR_DB_GATE}
WCF_GATE = {WCF_GATE}
FINITE_GATE = {FINITE_GATE}
DO_PERF = {int(do_perf)}
SCALE_GROUP_SIZE = 32

def emit(d):
    print("BENCH_JSON_START")
    print(json.dumps(d))
    print("BENCH_JSON_END")

try:
    spec = importlib.util.spec_from_file_location("R50D_aiter_shim", SHIM_PATH)
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)

    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)

    # -------------------------------------------------------------------------
    # Random data, identical generator pattern to op_tests/test_gemm_a4w4.py
    # -------------------------------------------------------------------------
    dtype = dtypes.bf16
    x = torch.randn((M, K), dtype=dtype, device='cuda')
    w = torch.randn((N, K), dtype=dtype, device='cuda')

    # quant: shuffle=True returns aiter-preshuffled scales for B (and A)
    x_packed, x_scales_shuffle = quant_func(x, shuffle=True)
    w_packed, w_scales_shuffle = quant_func(w, shuffle=True)

    # B preshuffle (weight tile reorder) — required by `BpreShuffle_256x256.co`
    wshuffle = shuffle_weight(w_packed, layout=(16, 16))

    # Output: pad M to multiples of 32 (aiter convention)
    m_pad32 = ((M + 31) // 32) * 32
    C = torch.zeros((m_pad32, N), dtype=dtype, device='cuda')

    # cast scale views to uint8 for the e8m0 ABI
    x_scales_u8 = x_scales_shuffle.view(torch.uint8)
    w_scales_u8 = w_scales_shuffle.view(torch.uint8)
    if x_packed.dtype != torch.uint8:
        x_packed_u8 = x_packed.view(torch.uint8)
    else:
        x_packed_u8 = x_packed
    if wshuffle.dtype != torch.uint8:
        w_packed_u8 = wshuffle.view(torch.uint8)
    else:
        w_packed_u8 = wshuffle

    stream_handle = int(torch.cuda.current_stream().cuda_stream)

    def run():
        shim.launch(x_packed_u8, w_packed_u8, x_scales_u8, w_scales_u8,
                    C, M, N, K,
                    256, 256,
                    AITER_CO, KERNEL_NAME,
                    1.0, 0.0,
                    stream_handle)

    # warm + smoke
    C.zero_()
    run()
    torch.cuda.synchronize()
    C0 = C[:M].clone()
    finite_frac = float(torch.isfinite(C0.float()).sum().item()) / float(C0.numel())
    finite_frac = round(finite_frac, 6)

    # -------------------------------------------------------------------------
    # Torch reference (subset rows only, to stay within memory budget)
    # -------------------------------------------------------------------------
    MAX_ROWS = 1024
    m_rows = min(M, MAX_ROWS)
    n_rows = min(N, MAX_ROWS)

    # We need UN-shuffled scales for the reference; quant_func with shuffle=False
    # returns the canonical per-1x32 e8m0 layout suitable for repeat_interleave.
    # Re-quantize from the same x/w (deterministic at this seed).
    _, x_scales_canon = quant_func(x, shuffle=False)
    _, w_scales_canon = quant_func(w, shuffle=False)

    # Reference matmul mirrors op_tests/test_gemm_a4w4.py:run_torch
    x_f32 = fp4_utils.mxfp4_to_f32(x_packed)[:m_rows].float()
    w_f32 = fp4_utils.mxfp4_to_f32(w_packed)[:n_rows].float()

    x_sc_canon = x_scales_canon[:m_rows].view(torch.uint8)
    x_sc_canon = x_sc_canon.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    x_sc_f32 = fp4_utils.e8m0_to_f32(x_sc_canon)[:, :K]
    x_f32 = x_f32 * x_sc_f32

    w_sc_canon = w_scales_canon[:n_rows].view(torch.uint8)
    w_sc_canon = w_sc_canon.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    w_sc_f32 = fp4_utils.e8m0_to_f32(w_sc_canon)[:, :K]
    w_f32 = w_f32 * w_sc_f32

    C_ref = torch.mm(x_f32, w_f32.T).double()
    C_test = C0[:m_rows, :n_rows].float().double()
    base_mask = torch.isfinite(C_ref) & torch.isfinite(C_test)

    REL_TOL = 0.05
    ABS_TOL = 1e-2
    n_valid = int(base_mask.sum().item())

    if n_valid < 1024:
        snr_db = float('-inf'); snr_med_db = float('-inf'); wrong_cell_frac = 1.0
    else:
        abs_diff = (C_test - C_ref).abs()
        abs_ref = C_ref.abs(); abs_test = C_test.abs()
        catastrophic = (abs_test > 100.0 * abs_ref + 1e-3) & (abs_diff > REL_TOL * abs_ref + ABS_TOL)
        catastrophic = catastrophic & base_mask
        wrong_cell_frac = float(catastrophic.sum().item()) / float(base_mask.sum().item())

        good_mask = base_mask & ~catastrophic
        n_good = int(good_mask.sum().item())
        if n_good < 1024:
            snr_db = float('-inf'); snr_med_db = float('-inf')
        else:
            rv = C_ref[good_mask]; tv = C_test[good_mask]
            sig = float((rv * rv).mean().item())
            err = float(((tv - rv) ** 2).mean().item())
            snr_db = 10.0 * math.log10(sig / err) if (sig > 0 and err > 0) else float('-inf')

            row_n = good_mask.sum(dim=1).clamp_min(1)
            row_sig = (C_ref ** 2 * good_mask).sum(dim=1) / row_n
            row_err = ((C_test - C_ref) ** 2 * good_mask).sum(dim=1) / row_n
            valid_rows = good_mask.any(dim=1) & (row_sig > 0) & (row_err > 0)
            if int(valid_rows.sum().item()) < 8:
                snr_med_db = float('-inf')
            else:
                row_snr = 10.0 * torch.log10(row_sig[valid_rows] / row_err[valid_rows])
                snr_med_db = float(row_snr.median().item())

    correct = (snr_med_db >= SNR_DB_GATE
               and wrong_cell_frac < WCF_GATE
               and finite_frac >= FINITE_GATE)

    if not correct or not DO_PERF:
        out = {{
            "M": M, "N": N, "K": K,
            "tflops": None, "avg_ms": None,
            "comp": {COMPETITOR_TFLOPS},
            "status": "WRONG_OUTPUT" if not correct else "CORRECT_NO_PERF",
            "kernel_finite": finite_frac,
            "snr_db": snr_db if math.isfinite(snr_db) else None,
            "snr_med_db": snr_med_db if math.isfinite(snr_med_db) else None,
            "snr_n_valid": n_valid,
            "wrong_cell_frac": round(wrong_cell_frac, 6),
            "seed": {seed},
        }}
        emit(out); sys.exit(0)

    # ---------------- timing ----------------
    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize()

    times = []
    for _ in range(ITERS):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); run(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    times.sort()
    trim = int(len(times) * TRIM_FRAC)
    if trim > 0: times = times[trim:-trim]
    avg = sum(times) / len(times)
    tflops = 2.0 * M * N * K / (avg * 1e-3) / 1e12

    out = {{
        "M": M, "N": N, "K": K,
        "tflops": round(tflops, 1), "avg_ms": round(avg, 4),
        "comp": {COMPETITOR_TFLOPS},
        "status": "OK",
        "kernel_finite": finite_frac,
        "snr_db": round(snr_db, 2) if math.isfinite(snr_db) else None,
        "snr_med_db": round(snr_med_db, 2) if math.isfinite(snr_med_db) else None,
        "snr_n_valid": n_valid,
        "wrong_cell_frac": round(wrong_cell_frac, 6),
        "seed": {seed},
    }}
    emit(out)
except Exception as e:
    import traceback
    tb = traceback.format_exc()
    out = {{
        "M": M, "N": N, "K": K,
        "tflops": None, "avg_ms": None,
        "comp": {COMPETITOR_TFLOPS},
        "status": f"ERR:{{e}}"[:200],
        "kernel_finite": None,
        "seed": {seed},
        "tb": tb[-800:],
    }}
    emit(out)
"""


def run_one_seed(seed: int, gpu_id, do_perf: bool, timeout: int = 900):
    script = make_runner_script(seed, do_perf)
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"M": M, "N": N, "K": K, "status": "TIMEOUT", "seed": seed,
                "comp": COMPETITOR_TFLOPS, "kernel_finite": None,
                "tflops": None, "avg_ms": None}
    if r.returncode != 0:
        return {"M": M, "N": N, "K": K, "status": "CRASH", "seed": seed,
                "comp": COMPETITOR_TFLOPS, "kernel_finite": None,
                "tflops": None, "avg_ms": None,
                "stderr_tail": r.stderr[-400:],
                "stdout_tail": r.stdout[-400:]}
    try:
        s = r.stdout.index("BENCH_JSON_START") + len("BENCH_JSON_START")
        e = r.stdout.index("BENCH_JSON_END")
        return json.loads(r.stdout[s:e].strip())
    except (ValueError, json.JSONDecodeError):
        return {"M": M, "N": N, "K": K, "status": "PARSE_FAIL", "seed": seed,
                "comp": COMPETITOR_TFLOPS, "kernel_finite": None,
                "tflops": None, "avg_ms": None,
                "stdout_tail": r.stdout[-400:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "10run"], default="smoke")
    ap.add_argument("--gpu", default="0",
                    help="Single GPU index (e.g. '0') or comma-list to round-robin (e.g. '0,1').")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gpu_list = [int(g) for g in str(args.gpu).split(",") if g.strip() != ""]
    if not gpu_list:
        gpu_list = [0]

    if args.mode == "smoke":
        out_path = args.out or os.path.join(SCRIPT_DIR, "R55_OPT_E3_3_SMOKE.json")
        log_path = os.path.join(SCRIPT_DIR, "R55_OPT_E3_3_SMOKE.log")
        seeds = [101]
        do_perf = True
    else:
        out_path = args.out or os.path.join(SCRIPT_DIR, "R55_OPT_E3_3_10RUN.json")
        log_path = os.path.join(SCRIPT_DIR, "R55_OPT_E3_3_10RUN.log")
        seeds = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
        do_perf = False  # correctness only (first seed gets perf below)

    results = []
    log_lines = []
    log_lines.append(f"R55 Opt E3_3 bench — mode={args.mode}, gpu={gpu_list}, "
                     f"shape=({M},{N},{K})")
    log_lines.append("=" * 90)
    print(log_lines[-2]); print(log_lines[-1])

    for i, seed in enumerate(seeds):
        gpu_id = gpu_list[i % len(gpu_list)]
        t0 = time.time()
        # First seed in 10run does perf; rest do correctness only.
        this_perf = (do_perf or (args.mode == "10run" and seed == seeds[0]))
        r = run_one_seed(seed, gpu_id, this_perf)
        dt = time.time() - t0
        r["wall_s"] = round(dt, 1)
        r["gpu_id"] = gpu_id
        results.append(r)
        line = (f"  seed={seed:>5} gpu={gpu_id} status={r.get('status'):<22} "
                f"fin={r.get('kernel_finite')!s:<8} "
                f"wcf={r.get('wrong_cell_frac')!s:<10} "
                f"snr_med={r.get('snr_med_db')!s:<8} "
                f"tflops={r.get('tflops')!s:<10} ({dt:.1f}s)")
        print(line, flush=True)
        log_lines.append(line)
        if r.get("status") in ("CRASH", "TIMEOUT", "PARSE_FAIL"):
            log_lines.append(f"    stderr: {r.get('stderr_tail','')[:300]}")
            log_lines.append(f"    stdout: {r.get('stdout_tail','')[:300]}")

    # ----------------------------------------------------------------
    # Aggregate
    # ----------------------------------------------------------------
    fins = [r.get("kernel_finite") for r in results if r.get("kernel_finite") is not None]
    wcfs = [r.get("wrong_cell_frac") for r in results if r.get("wrong_cell_frac") is not None]
    snrs = [r.get("snr_med_db") for r in results if r.get("snr_med_db") is not None]
    n_ok = sum(1 for r in results
               if r.get("status") in ("OK", "CORRECT_NO_PERF")
               and (r.get("kernel_finite") is not None and r["kernel_finite"] >= FINITE_GATE)
               and (r.get("wrong_cell_frac") is not None and r["wrong_cell_frac"] < WCF_GATE)
               and (r.get("snr_med_db") is not None and r["snr_med_db"] >= SNR_DB_GATE))
    tflops_vals = [r.get("tflops") for r in results if r.get("tflops") is not None]

    summary = {
        "mode": args.mode,
        "shape": [M, N, K],
        "competitor_tflops": COMPETITOR_TFLOPS,
        "n_runs": len(results),
        "n_OK": n_ok,
        "n_OK_5": min(n_ok, len(seeds)),
        "fin_min": min(fins) if fins else None,
        "fin_max": max(fins) if fins else None,
        "wcf_min": min(wcfs) if wcfs else None,
        "wcf_max": max(wcfs) if wcfs else None,
        "wcf_std": (statistics.pstdev(wcfs) if len(wcfs) >= 2 else 0.0),
        "snr_min": min(snrs) if snrs else None,
        "snr_max": max(snrs) if snrs else None,
        "snr_med": (statistics.median(snrs) if snrs else None),
        "tflops_first": tflops_vals[0] if tflops_vals else None,
        "tflops_max": max(tflops_vals) if tflops_vals else None,
        "pct_comp_first": (round(tflops_vals[0] / COMPETITOR_TFLOPS * 100.0, 2)
                           if tflops_vals else None),
        "passes_10run_gate": (
            len(seeds) == 10 and n_ok >= 8
            and (max(wcfs) if wcfs else 1.0) < WCF_GATE
            and (statistics.pstdev(wcfs) if len(wcfs) >= 2 else 1.0) < 0.01
            and (min(fins) if fins else 0.0) >= FINITE_GATE
        ),
    }

    log_lines.append("-" * 90)
    log_lines.append("SUMMARY: " + json.dumps(summary))
    print(log_lines[-1])

    full = {"summary": summary, "results": results}
    with open(out_path, "w") as f:
        json.dump(full, f, indent=2)
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nWrote {out_path}")
    print(f"Wrote {log_path}")


if __name__ == "__main__":
    main()
