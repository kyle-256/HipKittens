"""R27 Reviewer aggregator: parse run logs into baseline JSON.

Reads $WT/r27_baseline_runs/run_{kind}_{name}_{layout}_r{i}.log
Writes r27_baseline_gpu4.json
"""
import glob
import json
import os
import re
import statistics

WT = "/tmp/wt-r27-rev"
RUNDIR = os.path.join(WT, "r27_baseline_runs")

CELLS = [
    ("8k_rcr",        8192, 8192,  8192,  "rcr"),
    ("8k_rrr",        8192, 8192,  8192,  "rrr"),
    ("8k_crr",        8192, 8192,  8192,  "crr"),
    ("4k_rcr",        4096, 4096,  4096,  "rcr"),
    ("8b_gate_crr",   4096, 14336, 4096,  "crr"),
    ("8b_down_crr",   4096, 4096,  14336, "crr"),
    ("70b_qo_rcr",    4096, 8192,  8192,  "rcr"),
    ("70b_kv_crr",    4096, 1024,  8192,  "crr"),
    ("70b_gate_crr",  4096, 28672, 8192,  "crr"),
    ("70b_down_crr",  4096, 8192,  28672, "crr"),
]

TFLOPS_RE = re.compile(r"TFLOPS:\s*([\d.]+)")


def parse_log(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            m = TFLOPS_RE.search(line)
            if m:
                return float(m.group(1))
    return None


def filter_outliers(samples, sigma=2.0):
    """Drop samples >2sigma from median."""
    if len(samples) < 3:
        return samples
    med = statistics.median(samples)
    sd = statistics.pstdev(samples)
    if sd == 0:
        return samples
    return [s for s in samples if abs(s - med) <= sigma * sd]


def main():
    out = {
        "meta": {
            "agent": "R27 Reviewer",
            "branch": "r27-rev",
            "commit_base": "7790b606",
            "gpu": "GPU4",
            "iters": "FP8 warmup=300 iters=300; MXFP8 warmup=200 iters=300",
            "reps_per_cell": 5,
            "preheat": "preheat_then_bench.py / preheat_fp8_bench.py (8s/6s sustained matmul)",
            "outlier_filter": "drop samples > 2 sigma from median",
            "perf_gate": 0.95,
        },
        "cells": [],
    }
    for name, M, N, K, layout in CELLS:
        cell = {"name": name, "M": M, "N": N, "K": K, "layout": layout}
        for kind in ("fp8", "mxfp8"):
            samples = []
            for i in range(1, 6):
                path = os.path.join(
                    RUNDIR, f"run_{kind}_{name}_{layout}_r{i}.log"
                )
                v = parse_log(path)
                if v is not None:
                    samples.append(v)
            cell[f"{kind}_samples"] = samples
            clean = filter_outliers(samples)
            cell[f"{kind}_clean_samples"] = clean
            cell[f"{kind}_median"] = round(statistics.median(clean), 2) if clean else None
            cell[f"{kind}_mean"] = round(statistics.mean(clean), 2) if clean else None
            cell[f"{kind}_stdev"] = round(statistics.pstdev(clean), 2) if len(clean) > 1 else 0.0
        if cell["fp8_median"] and cell["mxfp8_median"]:
            cell["ratio"] = round(cell["mxfp8_median"] / cell["fp8_median"], 4)
        else:
            cell["ratio"] = None
        out["cells"].append(cell)
    out["summary"] = {
        "n_cells": len(out["cells"]),
        "perf_gate_pass": sum(1 for c in out["cells"] if c["ratio"] and c["ratio"] >= 0.95),
        "v2_wins": sum(1 for c in out["cells"] if c["ratio"] and c["ratio"] >= 1.0),
    }
    # Drift vs R26 reverify for 5 overlapping cells
    r26_path = os.path.join(WT, "r26_reverify.json")
    if os.path.exists(r26_path):
        with open(r26_path) as f:
            r26 = json.load(f)
        r26_by_shape = {(c["M"], c["N"], c["K"], c["layout"]): c for c in r26["cells"]}
        drift = []
        for c in out["cells"]:
            key = (c["M"], c["N"], c["K"], c["layout"])
            if key in r26_by_shape and c["mxfp8_median"]:
                r26c = r26_by_shape[key]
                v2_drift = c["mxfp8_median"] - r26c["v2_median"]
                fp8_drift = c["fp8_median"] - r26c["fp8_median"]
                drift.append({
                    "name": c["name"],
                    "shape": f"{c['M']}x{c['N']}x{c['K']}",
                    "layout": c["layout"],
                    "r26_v2_median": r26c["v2_median"],
                    "r27_v2_median": c["mxfp8_median"],
                    "v2_drift_tflops": round(v2_drift, 2),
                    "v2_drift_pct": round(100 * v2_drift / r26c["v2_median"], 2),
                    "r26_fp8_median": r26c["fp8_median"],
                    "r27_fp8_median": c["fp8_median"],
                    "fp8_drift_tflops": round(fp8_drift, 2),
                    "fp8_drift_pct": round(100 * fp8_drift / r26c["fp8_median"], 2),
                })
        out["drift_vs_r26_reverify"] = drift
    out_path = os.path.join(WT, "r27_baseline_gpu4.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
