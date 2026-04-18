"""R31 Reviewer Phase 1: Aggregate 4-GPU triangulation results into JSON.
Reads r31_4gpu_runs/70b_kv_crr_mxfp8_gpu{0,4,5,6}.txt and emits
r31_reviewer_kv_4gpu.json with summary statistics."""
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(HERE, "r31_4gpu_runs")
OUT_JSON = os.path.join(HERE, "r31_reviewer_kv_4gpu.json")

GPUS = [0, 4, 5, 6]

results = {}
for g in GPUS:
    path = os.path.join(RUN_DIR, f"70b_kv_crr_mxfp8_gpu{g}.txt")
    if not os.path.isfile(path):
        print(f"MISSING: {path}", file=sys.stderr)
        continue
    txt = open(path).read()
    snr = None; det = None
    md5 = None
    sclk_pre = None; sclk_post = None
    sclk_pre_bench = None; sclk_post_bench = None
    for line in txt.splitlines():
        m = re.search(r"snr_db=([\d.]+)", line)
        if m: snr = float(m.group(1))
        if "DETERMINISM ok=" in line:
            det = "True" in line
        m = re.search(r"md5=([0-9a-f]+)", line)
        if m: md5 = m.group(1)
        m = re.search(r"\((\d+Mhz)\)", line)
        freq = m.group(1) if m else None
        if "[sclk-pre-preheat]" in line: sclk_pre = freq
        elif "[sclk-post-preheat]" in line: sclk_post = freq
        elif "[sclk-pre-bench]" in line: sclk_pre_bench = freq
        elif "[sclk-post-bench]" in line: sclk_post_bench = freq
    m = re.search(r"TFLOPS_LIST ([0-9.,]+)", txt)
    tlist = [float(x) for x in m.group(1).split(",")] if m else []
    if not tlist:
        print(f"NO TFLOPS in {path}", file=sys.stderr)
        continue
    med = statistics.median(tlist)
    mean = statistics.mean(tlist)
    sd = statistics.stdev(tlist) if len(tlist) > 1 else 0.0
    # SNR(dB) signal-to-noise of bench: 20*log10(mean/stdev)
    import math
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
mx = max(medians); mn = min(medians)
spread_pct = (mx - mn) / mn * 100.0

# Verdict
verdict_parts = []
if spread_pct > 2.0:
    verdict_parts.append(f"cross-GPU spread {spread_pct:.2f}% exceeds bench-precision (~0.3-0.5%)")
else:
    verdict_parts.append(f"cross-GPU spread {spread_pct:.2f}% within bench-precision")
# GPU0 vs rest
g0 = results.get("GPU0", {}).get("tflops_median")
others = [results[k]["tflops_median"] for k in results if k != "GPU0"]
if g0 and others:
    others_med = statistics.median(others)
    delta = (g0 - others_med) / others_med * 100.0
    verdict_parts.append(f"GPU0 median {g0:.2f} differs from GPU4/5/6 median {others_med:.2f} by {delta:+.2f}%")

# DPM analysis
sclk_summary = {f"GPU{g}": (results.get(f"GPU{g}", {}).get("sclk_post_preheat"),
                            results.get(f"GPU{g}", {}).get("sclk_pre_bench"))
                for g in GPUS}

summary = {
    "median_of_4_gpus": round(overall_median, 2),
    "mean_of_4_gpu_medians": round(overall_mean, 2),
    "max": round(mx, 2),
    "min": round(mn, 2),
    "spread_pct": round(spread_pct, 3),
    "spread_abs_tflops": round(mx - mn, 2),
    "verdict": "; ".join(verdict_parts),
    "sclk_post_preheat_per_gpu": {k: v[0] for k, v in sclk_summary.items()},
    "sclk_pre_bench_per_gpu": {k: v[1] for k, v in sclk_summary.items()},
}

doc = {
    "shape": {"M": 4096, "N": 1024, "K": 8192, "layout": "CRR"},
    "method": "5x preheat-then-bench (8s 16k FP16 preheat, warmup=50, iters=100)",
    "kind": "MXFP8 V2 (gemm_crr_pq_v2)",
    "build_md5": list({r["md5"] for r in results.values() if r["md5"]}),
    "results": results,
    "summary": summary,
    "historical_baselines_for_comparison": {
        "R27_GPU4": 787.96,
        "R29_GPU4": 766.86,
        "R30_GPU0_DevA": 791.25,
        "R30_GPU5": 767.23,
    },
}

with open(OUT_JSON, "w") as f:
    json.dump(doc, f, indent=2)

print(f"WROTE {OUT_JSON}")
print(json.dumps(summary, indent=2))
