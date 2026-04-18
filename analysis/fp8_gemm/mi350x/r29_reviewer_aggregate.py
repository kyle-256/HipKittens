"""R29 Reviewer aggregator: parse r29_runs/<cell>_<kind>.txt files and build
analysis/fp8_gemm/mi350x/r29_reviewer_baseline_gpu4.json with same schema as
R27 (r27_reviewer_baseline_gpu4.json), plus drift-vs-R27 table with Welch t.
"""
import json
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "r29_runs")
R27 = os.path.join(HERE, "r27_reviewer_baseline_gpu4.json")
OUT = os.path.join(HERE, "r29_reviewer_baseline_gpu4.json")

CELLS = [
    ("8k_rcr",       "rcr", 8192, 8192, 8192),
    ("8k_rrr",       "rrr", 8192, 8192, 8192),
    ("8k_crr",       "crr", 8192, 8192, 8192),
    ("4k_rcr",       "rcr", 4096, 4096, 4096),
    ("8b_gate_crr",  "crr", 4096, 14336, 4096),
    ("8b_down_crr",  "crr", 4096, 4096, 14336),
    ("70b_qo_rcr",   "rcr", 4096, 8192, 8192),
    ("70b_kv_crr",   "crr", 4096, 1024, 8192),
    ("70b_gate_crr", "crr", 4096, 28672, 8192),
    ("70b_down_crr", "crr", 4096, 8192, 28672),
]

TFLOPS_LIST_RE = re.compile(r"^TFLOPS_LIST\s+(.+)$", re.MULTILINE)
SNR_RE = re.compile(r"CORRECTNESS snr_db=([\-\d.]+)")
DET_RE = re.compile(r"DETERMINISM ok=(True|False)")


def parse_one(path):
    if not os.path.isfile(path):
        return None
    txt = open(path).read()
    m = TFLOPS_LIST_RE.search(txt)
    if not m:
        return None
    samples = [float(x) for x in m.group(1).split(",") if x.strip()]
    snr = SNR_RE.search(txt); det = DET_RE.search(txt)
    return {
        "samples": samples,
        "snr_db": float(snr.group(1)) if snr else None,
        "det_ok": (det.group(1) == "True") if det else None,
    }


def clean_2sigma(samples):
    if len(samples) < 3:
        return list(samples)
    med = statistics.median(samples)
    sd = statistics.stdev(samples)
    if sd == 0:
        return list(samples)
    return [x for x in samples if abs(x - med) <= 2 * sd]


def welch_t(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    denom = math.sqrt(va / na + vb / nb) if (va / na + vb / nb) > 0 else 0
    if denom == 0:
        return float("inf") if ma != mb else 0.0
    return (ma - mb) / denom


cells = []
for name, layout, M, N, K in CELLS:
    fp8_path = os.path.join(RUNS, f"{name}_fp8.txt")
    mxfp8_path = os.path.join(RUNS, f"{name}_mxfp8.txt")
    fp8 = parse_one(fp8_path)
    mx = parse_one(mxfp8_path)
    if fp8 is None or mx is None:
        print(f"[skip] missing data for {name}: fp8={fp8 is not None} mxfp8={mx is not None}")
        continue
    fp8_clean = clean_2sigma(fp8["samples"])
    mx_clean = clean_2sigma(mx["samples"])
    fp8_med = statistics.median(fp8_clean)
    mx_med = statistics.median(mx_clean)
    cell = {
        "name": name,
        "M": M, "N": N, "K": K,
        "layout": layout,
        "fp8_samples": fp8["samples"],
        "fp8_clean_samples": fp8_clean,
        "fp8_median": round(fp8_med, 2),
        "fp8_mean": round(statistics.mean(fp8_clean), 2),
        "fp8_stdev": round(statistics.stdev(fp8_clean), 2) if len(fp8_clean) > 1 else 0.0,
        "fp8_snr_db": fp8["snr_db"],
        "fp8_det_ok": fp8["det_ok"],
        "mxfp8_samples": mx["samples"],
        "mxfp8_clean_samples": mx_clean,
        "mxfp8_median": round(mx_med, 2),
        "mxfp8_mean": round(statistics.mean(mx_clean), 2),
        "mxfp8_stdev": round(statistics.stdev(mx_clean), 2) if len(mx_clean) > 1 else 0.0,
        "mxfp8_snr_db": mx["snr_db"],
        "mxfp8_det_ok": mx["det_ok"],
        "ratio": round(mx_med / fp8_med, 4) if fp8_med > 0 else None,
    }
    cells.append(cell)


# Drift vs R27 baseline
drift = []
welch = []
if os.path.isfile(R27):
    r27 = json.load(open(R27))
    r27_by_name = {c["name"]: c for c in r27["cells"]}
    for c in cells:
        r27c = r27_by_name.get(c["name"])
        if not r27c:
            continue
        r27_ratio = r27c["ratio"]
        r29_ratio = c["ratio"]
        delta_pp = (r29_ratio - r27_ratio) * 100  # percentage points
        # Welch t-test on MXFP8 samples (R27 vs R29)
        t_mx = welch_t(c["mxfp8_clean_samples"], r27c["mxfp8_clean_samples"])
        t_fp = welch_t(c["fp8_clean_samples"], r27c["fp8_clean_samples"])
        drift.append({
            "name": c["name"],
            "shape": f"{c['M']}x{c['N']}x{c['K']}",
            "layout": c["layout"],
            "r27_ratio": r27_ratio,
            "r29_ratio": r29_ratio,
            "delta_pp": round(delta_pp, 3),
            "r27_mxfp8_median": r27c["mxfp8_median"],
            "r29_mxfp8_median": c["mxfp8_median"],
            "mxfp8_drift_tflops": round(c["mxfp8_median"] - r27c["mxfp8_median"], 2),
            "mxfp8_drift_pct": round((c["mxfp8_median"] - r27c["mxfp8_median"]) / r27c["mxfp8_median"] * 100, 2),
            "welch_t_mxfp8_r29_vs_r27": round(t_mx, 2) if t_mx is not None and math.isfinite(t_mx) else t_mx,
            "r27_fp8_median": r27c["fp8_median"],
            "r29_fp8_median": c["fp8_median"],
            "fp8_drift_tflops": round(c["fp8_median"] - r27c["fp8_median"], 2),
            "fp8_drift_pct": round((c["fp8_median"] - r27c["fp8_median"]) / r27c["fp8_median"] * 100, 2),
            "welch_t_fp8_r29_vs_r27": round(t_fp, 2) if t_fp is not None and math.isfinite(t_fp) else t_fp,
            "significant_change_t_gt_3": (t_mx is not None and math.isfinite(t_mx) and abs(t_mx) > 3.0),
        })

summary = {
    "n_cells": len(cells),
    "perf_gate_pass": sum(1 for c in cells if (c["ratio"] or 0) >= 0.95),
    "v2_wins": sum(1 for c in cells if (c["ratio"] or 0) >= 1.0),
}

doc = {
    "meta": {
        "agent": "R29 Reviewer",
        "branch": "r29-rev",
        "commit_base": "145ff766",
        "gpu": "GPU4",
        "iters": f"FP8 + MXFP8 WARMUP=50 ITERS=100 N_RUNS=5",
        "reps_per_cell": 5,
        "preheat": "r29_reviewer_bench5x.py: per-process 8s 16k FP16 matmul preheat in same process",
        "outlier_filter": "drop samples > 2 sigma from median",
        "perf_gate": 0.95,
    },
    "cells": cells,
    "summary": summary,
    "drift_vs_r27": drift,
}

with open(OUT, "w") as f:
    json.dump(doc, f, indent=2)
print(f"wrote {OUT}")
print(f"  n_cells={len(cells)} perf_gate_pass={summary['perf_gate_pass']} v2_wins={summary['v2_wins']}")
