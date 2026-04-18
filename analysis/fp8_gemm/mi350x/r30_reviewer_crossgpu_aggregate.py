"""R30 Reviewer cross-GPU aggregator.

Parses analysis/fp8_gemm/mi350x/r30_crossgpu_runs/<cell>_<kind>.txt files for
the 3 R29-flagged cells (8b_gate_crr, 8b_down_crr, 70b_kv_crr) and writes
analysis/fp8_gemm/mi350x/r30_reviewer_crossgpu_reverify.json.

For each cell, computes:
  - R30 cross-GPU median + samples (this run)
  - R29 GPU4 median + samples (from r29_reviewer_baseline_gpu4.json)
  - R27 GPU4 median + samples (from r27_reviewer_baseline_gpu4.json)
  - Welch t (R30 cross-GPU vs R29 GPU4)  on MXFP8
  - Welch t (R30 cross-GPU vs R27 GPU4)  on MXFP8
  - Verdict: real_regression / gpu_state_artifact / inconclusive
"""
import json
import math
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "r30_crossgpu_runs")
R27 = os.path.join(HERE, "r27_reviewer_baseline_gpu4.json")
R29 = os.path.join(HERE, "r29_reviewer_baseline_gpu4.json")
OUT = os.path.join(HERE, "r30_reviewer_crossgpu_reverify.json")

CELLS = [
    ("8b_gate_crr",  "crr", 4096, 14336, 4096),
    ("8b_down_crr",  "crr", 4096, 4096, 14336),
    ("70b_kv_crr",   "crr", 4096, 1024, 8192),
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
    if not a or not b or len(a) < 2 or len(b) < 2:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    denom_sq = va / na + vb / nb
    if denom_sq <= 0:
        return None
    return (ma - mb) / math.sqrt(denom_sq)


def round_or_none(x, n=2):
    if x is None or not math.isfinite(x):
        return x
    return round(x, n)


# Load R27 + R29 baselines (GPU4)
r27_doc = json.load(open(R27)) if os.path.isfile(R27) else {"cells": []}
r29_doc = json.load(open(R29)) if os.path.isfile(R29) else {"cells": []}
r27_by_name = {c["name"]: c for c in r27_doc["cells"]}
r29_by_name = {c["name"]: c for c in r29_doc["cells"]}

results = []

for name, layout, M, N, K in CELLS:
    fp8_path = os.path.join(RUNS, f"{name}_fp8.txt")
    mx_path = os.path.join(RUNS, f"{name}_mxfp8.txt")
    fp8 = parse_one(fp8_path)
    mx = parse_one(mx_path)
    if fp8 is None or mx is None:
        print(f"[skip] missing {name}: fp8={fp8 is not None} mxfp8={mx is not None}")
        continue
    fp8_clean = clean_2sigma(fp8["samples"])
    mx_clean = clean_2sigma(mx["samples"])
    fp8_med = statistics.median(fp8_clean)
    mx_med = statistics.median(mx_clean)
    r27c = r27_by_name.get(name, {})
    r29c = r29_by_name.get(name, {})
    r27_mx_samples = r27c.get("mxfp8_clean_samples", [])
    r29_mx_samples = r29c.get("mxfp8_clean_samples", [])
    r27_fp8_samples = r27c.get("fp8_clean_samples", [])
    r29_fp8_samples = r29c.get("fp8_clean_samples", [])

    t_mx_vs_r29 = welch_t(mx_clean, r29_mx_samples)
    t_mx_vs_r27 = welch_t(mx_clean, r27_mx_samples)
    t_fp_vs_r29 = welch_t(fp8_clean, r29_fp8_samples)
    t_fp_vs_r27 = welch_t(fp8_clean, r27_fp8_samples)

    r27_mx_med = r27c.get("mxfp8_median")
    r29_mx_med = r29c.get("mxfp8_median")

    delta_mx_vs_r27_pct = ((mx_med - r27_mx_med) / r27_mx_med * 100) if r27_mx_med else None
    delta_mx_vs_r29_pct = ((mx_med - r29_mx_med) / r29_mx_med * 100) if r29_mx_med else None

    # Verdict logic:
    # - If R30 cross-GPU MXFP8 median is within ~1.5% of R27 GPU4 median AND
    #   |t vs R27| < 3, then the R29 GPU4 negative-t is a GPU-state artifact.
    # - If R30 cross-GPU MXFP8 median is within bench-to-bench noise of R29 GPU4
    #   (i.e. |t vs R29| < 3 AND |delta vs R29| < 1%), then both are consistent
    #   and the regression appears real (or persistent across GPUs).
    # - Otherwise inconclusive.
    verdict = "inconclusive"
    rationale = ""
    if delta_mx_vs_r27_pct is not None and t_mx_vs_r27 is not None:
        if abs(delta_mx_vs_r27_pct) <= 1.5 and abs(t_mx_vs_r27) < 3.0:
            verdict = "gpu_state_artifact"
            rationale = (
                f"R30 cross-GPU MXFP8 median {mx_med:.2f} matches R27 GPU4 "
                f"{r27_mx_med:.2f} within 1.5% (delta={delta_mx_vs_r27_pct:+.2f}%, "
                f"Welch t vs R27 = {t_mx_vs_r27:+.2f}). R29 GPU4 drop attributable to "
                f"GPU4-state/DPM, not source-driven regression."
            )
        elif (delta_mx_vs_r29_pct is not None
              and abs(delta_mx_vs_r29_pct) <= 1.0
              and t_mx_vs_r29 is not None and abs(t_mx_vs_r29) < 3.0
              and (t_mx_vs_r27 is not None and t_mx_vs_r27 < -3.0)):
            verdict = "real_regression_or_persistent"
            rationale = (
                f"R30 cross-GPU MXFP8 median {mx_med:.2f} matches R29 GPU4 "
                f"{r29_mx_med:.2f} (delta={delta_mx_vs_r29_pct:+.2f}%, "
                f"t vs R29 = {t_mx_vs_r29:+.2f}); both differ from R27 GPU4 "
                f"({r27_mx_med:.2f}, t vs R27 = {t_mx_vs_r27:+.2f}). "
                f"Consistent across GPUs ⇒ either real regression since R27 or "
                f"shared cross-cycle environment shift."
            )
        else:
            verdict = "inconclusive"
            t29s = f"{t_mx_vs_r29:+.2f}" if t_mx_vs_r29 is not None else "na"
            t27s = f"{t_mx_vs_r27:+.2f}" if t_mx_vs_r27 is not None else "na"
            d29s = f"{delta_mx_vs_r29_pct:+.2f}" if delta_mx_vs_r29_pct is not None else "na"
            d27s = f"{delta_mx_vs_r27_pct:+.2f}" if delta_mx_vs_r27_pct is not None else "na"
            r29ms = f"{r29_mx_med:.2f}" if r29_mx_med is not None else "na"
            r27ms = f"{r27_mx_med:.2f}" if r27_mx_med is not None else "na"
            rationale = (
                f"R30 cross-GPU MXFP8 median {mx_med:.2f} differs from both "
                f"R29 ({r29ms}, delta={d29s}%, t={t29s}) "
                f"and R27 ({r27ms}, delta={d27s}%, t={t27s})."
            )

    results.append({
        "name": name,
        "shape": f"{M}x{N}x{K}",
        "layout": layout,
        "r30_crossgpu": {
            "fp8_samples": fp8["samples"],
            "fp8_clean_samples": fp8_clean,
            "fp8_median": round(fp8_med, 2),
            "fp8_stdev": round(statistics.stdev(fp8_clean), 2) if len(fp8_clean) > 1 else 0.0,
            "fp8_snr_db": fp8["snr_db"],
            "fp8_det_ok": fp8["det_ok"],
            "mxfp8_samples": mx["samples"],
            "mxfp8_clean_samples": mx_clean,
            "mxfp8_median": round(mx_med, 2),
            "mxfp8_stdev": round(statistics.stdev(mx_clean), 2) if len(mx_clean) > 1 else 0.0,
            "mxfp8_snr_db": mx["snr_db"],
            "mxfp8_det_ok": mx["det_ok"],
            "ratio": round(mx_med / fp8_med, 4) if fp8_med > 0 else None,
        },
        "r29_gpu4": {
            "fp8_median": r29c.get("fp8_median"),
            "mxfp8_median": r29_mx_med,
            "ratio": r29c.get("ratio"),
        },
        "r27_gpu4": {
            "fp8_median": r27c.get("fp8_median"),
            "mxfp8_median": r27_mx_med,
            "ratio": r27c.get("ratio"),
        },
        "delta_mxfp8_pct_r30_vs_r27": round_or_none(delta_mx_vs_r27_pct, 2),
        "delta_mxfp8_pct_r30_vs_r29": round_or_none(delta_mx_vs_r29_pct, 2),
        "welch_t_mxfp8_r30_vs_r27": round_or_none(t_mx_vs_r27, 2),
        "welch_t_mxfp8_r30_vs_r29": round_or_none(t_mx_vs_r29, 2),
        "welch_t_fp8_r30_vs_r27": round_or_none(t_fp_vs_r27, 2),
        "welch_t_fp8_r30_vs_r29": round_or_none(t_fp_vs_r29, 2),
        "verdict": verdict,
        "rationale": rationale,
    })


doc = {
    "meta": {
        "agent": "R30 Reviewer",
        "branch": "r30-rev",
        "commit_base": "cd630fd8",
        "purpose": "cross-GPU reverify of 3 R29-flagged negative-t MXFP8 cells",
        "phys_gpu": os.environ.get("PHYS_GPU", "5_or_6"),
        "iters": "FP8 + MXFP8 WARMUP=50 ITERS=100 N_RUNS=5",
        "preheat": "per-process 8s 16k FP16 matmul preheat (same as R29)",
        "outlier_filter": "drop samples > 2 sigma from median",
        "build_cache_hygiene": "rm -f tk_*.so before each build; per-build md5 logged in build_md5.log",
        "note": "Source unchanged from R29 base (cd630fd8 = R29 cycle wrap); kernel binary equivalence enforced by verifying md5 of .so matches R27/R29 builds for same shape.",
    },
    "cells": results,
}

with open(OUT, "w") as f:
    json.dump(doc, f, indent=2)
print(f"wrote {OUT}")
for r in results:
    print(f"  {r['name']:14s} verdict={r['verdict']:30s} t_vs_r27={r['welch_t_mxfp8_r30_vs_r27']} t_vs_r29={r['welch_t_mxfp8_r30_vs_r29']} d_vs_r27={r['delta_mxfp8_pct_r30_vs_r27']}% d_vs_r29={r['delta_mxfp8_pct_r30_vs_r29']}%")
