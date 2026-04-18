"""R33 Reviewer Phase 1: Aggregate 4-GPU triangulation results into JSON.
Reads r33_reviewer_4gpu_runs/70b_kv_crr_mxfp8_gpu{0,4,5,6}.txt and emits
r33_reviewer_kv_4gpu.json with summary statistics.

R33 difference vs R32 aggregator:
  - Uses R33 paths (r33_reviewer_4gpu_runs / r33_reviewer_kv_4gpu.json)
  - Compares per-GPU drift vs both R31 and R32 (3rd cross-cycle data point)
  - Tests "high regime rotates per cycle" hypothesis (R31 GPU0 high, R32 GPU5 high, R33 ???)
"""
import json
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(HERE, "r33_reviewer_4gpu_runs")
OUT_JSON = os.path.join(HERE, "r33_reviewer_kv_4gpu.json")

GPUS = [0, 4, 5, 6]

results = {}
for g in GPUS:
    path = os.path.join(RUN_DIR, f"70b_kv_crr_mxfp8_gpu{g}.txt")
    if not os.path.isfile(path):
        print(f"MISSING: {path}", file=sys.stderr)
        continue
    txt = open(path).read()
    snr = None
    det = None
    md5 = None
    sclk_pre = None
    sclk_post = None
    sclk_pre_bench = None
    sclk_post_bench = None
    for line in txt.splitlines():
        m = re.search(r"snr_db=([\d.]+)", line)
        if m:
            snr = float(m.group(1))
        if "DETERMINISM ok=" in line:
            det = "True" in line
        m = re.search(r"md5=([0-9a-f]+)", line)
        if m:
            md5 = m.group(1)
        m = re.search(r"\((\d+Mhz)\)", line)
        freq = m.group(1) if m else None
        if "[sclk-pre-preheat]" in line:
            sclk_pre = freq
        elif "[sclk-post-preheat]" in line:
            sclk_post = freq
        elif "[sclk-pre-bench]" in line:
            sclk_pre_bench = freq
        elif "[sclk-post-bench]" in line:
            sclk_post_bench = freq
    m = re.search(r"TFLOPS_LIST ([0-9.,]+)", txt)
    tlist = [float(x) for x in m.group(1).split(",")] if m else []
    if not tlist:
        print(f"NO TFLOPS in {path}", file=sys.stderr)
        continue
    med = statistics.median(tlist)
    mean = statistics.mean(tlist)
    sd = statistics.stdev(tlist) if len(tlist) > 1 else 0.0
    snr_db_bench = 20 * math.log10(mean / sd) if sd > 0 else float("inf")
    results[f"GPU{g}"] = {
        "tflops_median": round(med, 2),
        "tflops_mean": round(mean, 2),
        "tflops_stdev": round(sd, 2),
        "tflops_list": [round(x, 4) for x in tlist],
        "bench_snr_db": round(snr_db_bench, 2),
        "correctness_snr_db": snr,
        "det_pass": det,
        "md5": md5,
        "sclk_pre_preheat": sclk_pre,
        "sclk_post_preheat": sclk_post,
        "sclk_pre_bench": sclk_pre_bench,
        "sclk_post_bench": sclk_post_bench,
    }

medians = [r["tflops_median"] for r in results.values()]
means = [r["tflops_mean"] for r in results.values()]
overall_median = statistics.median(medians)
overall_mean = statistics.mean(medians)
mx = max(medians)
mn = min(medians)
spread_pct = (mx - mn) / mn * 100.0

verdict_parts = []
if spread_pct > 2.0:
    verdict_parts.append(
        f"cross-GPU spread {spread_pct:.2f}% exceeds bench-precision (~0.3-0.5%)"
    )
else:
    verdict_parts.append(
        f"cross-GPU spread {spread_pct:.2f}% within bench-precision"
    )

# Identify which GPU is the high outlier this cycle
sorted_by_med = sorted(results.items(), key=lambda kv: -kv[1]["tflops_median"])
high_gpu = sorted_by_med[0][0] if sorted_by_med else None
high_med = sorted_by_med[0][1]["tflops_median"] if sorted_by_med else None
others_for_high = [v["tflops_median"] for k, v in results.items() if k != high_gpu]
high_excess_pct = None
if high_gpu and others_for_high:
    others_med = statistics.median(others_for_high)
    high_excess_pct = (high_med - others_med) / others_med * 100.0
    verdict_parts.append(
        f"high outlier GPU={high_gpu} ({high_med:.2f} TF), "
        f"+{high_excess_pct:+.2f}% above other-3 median {others_med:.2f}"
    )

sclk_summary = {
    f"GPU{g}": (
        results.get(f"GPU{g}", {}).get("sclk_post_preheat"),
        results.get(f"GPU{g}", {}).get("sclk_pre_bench"),
    )
    for g in GPUS
}

# Per-cycle comparison vs R31 and R32 Reviewer baselines
r31_per_gpu = {"GPU0": 786.38, "GPU4": 767.31, "GPU5": 764.28, "GPU6": 766.14}
r31_median_of_4 = 766.72
r32_per_gpu = {"GPU0": 768.43, "GPU4": 768.37, "GPU5": 786.59, "GPU6": 776.60}
r32_median_of_4 = 772.51

per_gpu_drift_vs_r32 = {}
per_gpu_drift_vs_r31 = {}
for k, v in results.items():
    if k in r32_per_gpu:
        d_abs = v["tflops_median"] - r32_per_gpu[k]
        d_pct = d_abs / r32_per_gpu[k] * 100.0
        per_gpu_drift_vs_r32[k] = {
            "r32_median": r32_per_gpu[k],
            "r33_median": v["tflops_median"],
            "drift_tflops": round(d_abs, 2),
            "drift_pct": round(d_pct, 3),
        }
    if k in r31_per_gpu:
        d_abs = v["tflops_median"] - r31_per_gpu[k]
        d_pct = d_abs / r31_per_gpu[k] * 100.0
        per_gpu_drift_vs_r31[k] = {
            "r31_median": r31_per_gpu[k],
            "r33_median": v["tflops_median"],
            "drift_tflops": round(d_abs, 2),
            "drift_pct": round(d_pct, 3),
        }

# Track the rotating high-outlier hypothesis
rotation_data_points = {
    "R31": {"high_gpu": "GPU0", "high_median": 786.38, "low3_median": 766.14},
    "R32": {"high_gpu": "GPU5", "high_median": 786.59, "low3_median": 768.43},
    "R33": {
        "high_gpu": high_gpu,
        "high_median": high_med,
        "other3_median": (statistics.median(others_for_high) if others_for_high else None),
    },
}

summary = {
    "median_of_4_gpus": round(overall_median, 2),
    "mean_of_4_gpu_medians": round(overall_mean, 2),
    "max": round(mx, 2),
    "min": round(mn, 2),
    "spread_pct": round(spread_pct, 3),
    "spread_abs_tflops": round(mx - mn, 2),
    "verdict": "; ".join(verdict_parts),
    "high_outlier_gpu_this_cycle": high_gpu,
    "high_outlier_excess_pct": (round(high_excess_pct, 3) if high_excess_pct is not None else None),
    "sclk_post_preheat_per_gpu": {k: v[0] for k, v in sclk_summary.items()},
    "sclk_pre_bench_per_gpu": {k: v[1] for k, v in sclk_summary.items()},
    "per_gpu_drift_vs_r32": per_gpu_drift_vs_r32,
    "per_gpu_drift_vs_r31": per_gpu_drift_vs_r31,
    "median_of_4_drift_vs_r32": {
        "r32_median_of_4": r32_median_of_4,
        "r33_median_of_4": round(overall_median, 2),
        "drift_tflops": round(overall_median - r32_median_of_4, 2),
        "drift_pct": round((overall_median - r32_median_of_4) / r32_median_of_4 * 100, 3),
    },
    "median_of_4_drift_vs_r31": {
        "r31_median_of_4": r31_median_of_4,
        "r33_median_of_4": round(overall_median, 2),
        "drift_tflops": round(overall_median - r31_median_of_4, 2),
        "drift_pct": round((overall_median - r31_median_of_4) / r31_median_of_4 * 100, 3),
    },
    "rotating_high_outlier_3cycle_hypothesis": rotation_data_points,
}

doc = {
    "shape": {"M": 4096, "N": 1024, "K": 8192, "layout": "CRR"},
    "method": (
        "5x preheat-then-bench (8s 16k FP16 preheat, warmup=50, iters=100); "
        "rocm-smi -d $PHYS_GPU (R31 Reviewer fix applied; carried forward in R33)"
    ),
    "kind": "MXFP8 V2 (gemm_crr_pq_v2)",
    "build_md5": list({r["md5"] for r in results.values() if r["md5"]}),
    "results": results,
    "summary": summary,
    "historical_baselines_for_comparison": {
        "R27_GPU4": 787.96,
        "R29_GPU4": 766.86,
        "R30_GPU0_DevA": 791.25,
        "R30_GPU5": 767.23,
        "R31_GPU0": 786.38,
        "R31_GPU4": 767.31,
        "R31_GPU5": 764.28,
        "R31_GPU6": 766.14,
        "R31_median_of_4": 766.72,
        "R32_GPU0": 768.43,
        "R32_GPU4": 768.37,
        "R32_GPU5": 786.59,
        "R32_GPU6": 776.60,
        "R32_median_of_4": 772.51,
    },
}

with open(OUT_JSON, "w") as f:
    json.dump(doc, f, indent=2)

print(f"WROTE {OUT_JSON}")
print(json.dumps(summary, indent=2))
