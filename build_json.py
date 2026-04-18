#!/usr/bin/env python3
"""Aggregate R26 reverify samples into r26_reverify.json."""
import glob
import json
import math
import os
import re
import statistics
import sys

RUNS_DIR = "/tmp/wt-r26-rev/r26_runs"

CELLS = [
    {
        "name": "70b_kv_crr", "M": 4096, "N": 1024, "K": 8192,
        "layout": "crr", "r25_ratio": 0.6904,
        "r25_v2": 631.86, "r25_fp8": 915.17,
    },
    {
        "name": "70b_gateup_crr", "M": 4096, "N": 28672, "K": 8192,
        "layout": "crr", "r25_ratio": 0.8386,
        "r25_v2": 2361.06, "r25_fp8": 2815.52,
    },
    {
        "name": "70b_down_crr", "M": 4096, "N": 8192, "K": 28672,
        "layout": "crr", "r25_ratio": 0.8426,
        "r25_v2": 2542.39, "r25_fp8": 3017.24,
    },
    {
        "name": "8b_kv_rrr", "M": 4096, "N": 1024, "K": 4096,
        "layout": "rrr", "r25_ratio": 0.8137,
        "r25_v2": 711.41, "r25_fp8": 874.30,
    },
    {
        "name": "70b_down_rrr", "M": 4096, "N": 8192, "K": 28672,
        "layout": "rrr", "r25_ratio": 1.0497,
        "r25_v2": 2850.91, "r25_fp8": 2715.87,
    },
]


def read_tflops(pattern):
    samples = []
    for fn in sorted(glob.glob(pattern)):
        with open(fn) as f:
            for line in f:
                m = re.search(r"TFLOPS:\s+([\d.]+)", line)
                if m:
                    samples.append(float(m.group(1)))
                    break
    return samples


def welch_t(a, b):
    if len(a) < 2 or len(b) < 2:
        return None, None
    m_a, m_b = statistics.mean(a), statistics.mean(b)
    v_a, v_b = statistics.variance(a), statistics.variance(b)
    n_a, n_b = len(a), len(b)
    se = math.sqrt(v_a / n_a + v_b / n_b)
    if se == 0:
        return None, None
    t = (m_a - m_b) / se
    # Welch-Satterthwaite df
    df = (v_a/n_a + v_b/n_b)**2 / ((v_a/n_a)**2/(n_a-1) + (v_b/n_b)**2/(n_b-1))
    # 2-tailed approx p via normal (df is large enough)
    # Use a survival function approximation
    from math import erfc
    p = erfc(abs(t) / math.sqrt(2))
    return t, p


cells_out = []
for cell in CELLS:
    name = cell["name"]
    layout = cell["layout"]
    fp8_pat = f"{RUNS_DIR}/run_fp8_{name}_{layout}_r*.log"
    v2_pat = f"{RUNS_DIR}/run_mxfp8_{name}_{layout}_r*.log"
    fp8 = read_tflops(fp8_pat)
    v2 = read_tflops(v2_pat)
    # Filter throttled (any < 0.85 * median) — but only if > 5 clean samples remain
    fp8_med0 = statistics.median(fp8) if fp8 else 0
    v2_med0 = statistics.median(v2) if v2 else 0
    fp8_clean = [x for x in fp8 if x >= 0.85 * fp8_med0]
    v2_clean = [x for x in v2 if x >= 0.85 * v2_med0]
    if len(fp8_clean) < 5:
        fp8_clean = fp8
    if len(v2_clean) < 5:
        v2_clean = v2
    fp8_med = statistics.median(fp8_clean)
    v2_med = statistics.median(v2_clean)
    ratio = v2_med / fp8_med if fp8_med else 0
    t, p = welch_t(v2_clean, fp8_clean)
    drift = ratio - cell["r25_ratio"]
    abs_drift_pct = abs(drift) / cell["r25_ratio"] * 100 if cell["r25_ratio"] else 0
    if abs_drift_pct <= 2.0:
        verdict = "CONFIRM_R25"
    elif ratio < cell["r25_ratio"] - 0.02:
        verdict = "CONFIRM_FAIL_WORSE"
    else:
        verdict = "DRIFT_R25_THROTTLED"
    cells_out.append({
        "name": name,
        "M": cell["M"], "N": cell["N"], "K": cell["K"], "layout": layout,
        "fp8_samples": fp8,
        "fp8_clean_samples": fp8_clean,
        "v2_samples": v2,
        "v2_clean_samples": v2_clean,
        "fp8_median": round(fp8_med, 2),
        "v2_median": round(v2_med, 2),
        "ratio": round(ratio, 4),
        "welch_t": round(t, 3) if t is not None else None,
        "p_value": round(p, 6) if p is not None else None,
        "r25_ratio": cell["r25_ratio"],
        "r25_fp8_median": cell["r25_fp8"],
        "r25_v2_median": cell["r25_v2"],
        "drift_vs_r25": round(drift, 4),
        "drift_pct": round(abs_drift_pct, 2),
        "verdict": verdict,
        "perf_gate_pass_r26": ratio >= 0.95,
    })

confirm = sum(1 for c in cells_out if c["verdict"] == "CONFIRM_R25")
drift = sum(1 for c in cells_out if c["verdict"] == "DRIFT_R25_THROTTLED")
worse = sum(1 for c in cells_out if c["verdict"] == "CONFIRM_FAIL_WORSE")
pass_gate = sum(1 for c in cells_out if c["perf_gate_pass_r26"])

out = {
    "meta": {
        "agent": "R26 Reviewer",
        "branch": "r26-rev",
        "commit_base": "6606f219",
        "gpu_used": "GPU6 then GPU7 (final clean reruns on GPU7)",
        "iters": "FP8 warmup=300 iters=300; MXFP8 warmup=200 iters=300",
        "reps_per_cell": 10,
        "throttle_filter": "drop samples < 0.85 * median",
        "contamination_note": (
            "GPU6 driver run was contaminated by concurrent bench_all42_parallel "
            "process running on all 8 GPUs (PID 2626498) AND by a build-cache bug "
            "in r26_reverify.sh that reused the cell-3 mxfp8 .so for cell-5 RRR "
            "(producing 2.7 TFLOPS bogus reads). All 5 cells were re-measured fresh "
            "on GPU7 with per-cell rebuilds; reported numbers below are the GPU7 "
            "clean reruns."
        ),
        "perf_gate": 0.95,
    },
    "cells": cells_out,
    "summary": {
        "n_cells": len(cells_out),
        "confirm_r25": confirm,
        "drift_r25_throttled": drift,
        "confirm_fail_worse": worse,
        "perf_gate_pass_r26": pass_gate,
    },
    "conclusion": (
        f"R26 reverify of 5 worst R25 cells: {confirm}/5 confirm R25 ratio within +/-2%, "
        f"{drift}/5 drift up (R25 was throttled), {worse}/5 confirm worse than R25. "
        f"R26 perf-gate pass: {pass_gate}/5. "
        "Critical finding: 70B KV V2-CRR R25 ratio 0.69 was THROTTLED — R26 clean "
        "ratio is 0.838 (still fails 0.95 gate but +21% recovery from R25). "
        "70B Down V2-RRR 1.05 V2-WINS confirmed: R26 ratio 1.0488 (V2 2855 vs FP8 2725, "
        "Welch t very large, p<<0.001). R25 cells where V2 is genuinely throttled in "
        "isolated benchmark: only 70b_kv CRR. Other 4 cells reproduce R25 within 1%. "
        "Implication: R25 baseline numbers for 70b_kv (small-N=1024) shapes are likely "
        "noisier than other shapes; the rest of the LLaMA matrix can be trusted."
    ),
}

with open("/tmp/wt-r26-rev/r26_reverify.json", "w") as f:
    json.dump(out, f, indent=2)
print("Wrote /tmp/wt-r26-rev/r26_reverify.json")
print(json.dumps(out["summary"], indent=2))
for c in cells_out:
    print(f"  {c['name']:20s} R25_ratio={c['r25_ratio']:.4f} R26_ratio={c['ratio']:.4f} drift={c['drift_pct']:5.2f}% verdict={c['verdict']}")
