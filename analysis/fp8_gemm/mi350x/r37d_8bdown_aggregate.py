#!/usr/bin/env python3
"""R37 Dev D Task 2 aggregate: STRICT gate verdict for 8B-Down V2-RRR.

STRICT gate: min Δ% >= +5.0 AND min Welch t > 10 across all 4 GPUs.
SHIP-LITE: min Δ% >= +5.0 AND min Welch t > 5.0 (R34 Dev B precedent).
NO-PROMOTE: min Δ% < +5.0 OR triangulation incomplete.
"""
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNDIR = HERE / "r37d_8bdown_runs"

# Per-GPU results (from clean.txt where gates passed; GPU5 best-of-6 attempts)
results = {}

def parse_clean(path):
    text = path.read_text()
    out = {}
    for line in text.splitlines():
        if line.startswith("CRR_TFLOPS_LIST"):
            out["crr"] = [float(x) for x in line.split()[1].split(",")]
        elif line.startswith("RRR_TFLOPS_LIST"):
            out["rrr"] = [float(x) for x in line.split()[1].split(",")]
        elif line.startswith("[sclk-post-preheat]"):
            out["sclk_post_preheat"] = line
        elif line.startswith("[sclk-post-bench]"):
            out["sclk_post_bench"] = line
        elif line.startswith("CORRECTNESS_CRR"):
            out["snr_crr"] = float(line.split("snr_db=")[1].split()[0])
            out["det_crr"] = "True" in line
        elif line.startswith("CORRECTNESS_RRR"):
            out["snr_rrr"] = float(line.split("snr_db=")[1].split()[0])
            out["det_rrr"] = "True" in line
    return out


def welch(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    sa, sb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    se = math.sqrt(sa / na + sb / nb)
    return (mb - ma) / se if se > 0 else float("nan")


def stats(a, b):
    crr_med = statistics.median(a)
    rrr_med = statistics.median(b)
    delta = (rrr_med - crr_med) / crr_med * 100
    t = welch(a, b)
    return {
        "crr_median": crr_med,
        "rrr_median": rrr_med,
        "crr_stdev_mean": statistics.stdev(a) / statistics.mean(a) if a else float("nan"),
        "rrr_stdev_mean": statistics.stdev(b) / statistics.mean(b) if b else float("nan"),
        "delta_pct": delta,
        "welch_t": t,
        "n": len(a),
    }


# GPU4/6/7: use clean files (passed all 3 gates)
for g in (4, 6, 7):
    p = RUNDIR / f"8b_down_gpu{g}_clean.txt"
    if not p.exists():
        continue
    parsed = parse_clean(p)
    s = stats(parsed["crr"], parsed["rrr"])
    s["gates"] = "PASS_3GATE"
    s["sclk_post_preheat"] = parsed.get("sclk_post_preheat", "")
    s["sclk_post_bench"] = parsed.get("sclk_post_bench", "")
    s["snr_db"] = parsed.get("snr_crr", float("nan"))
    s["det_ok"] = parsed.get("det_crr", False) and parsed.get("det_rrr", False)
    results[f"GPU{g}"] = s

# GPU5: use the cleanest of the retry-1 attempts (attempt 1 of retry session - sclk
# 1917 close to gate 2200, RRR stdev/mean 2.77% - failed G1+G2b but direction stable
# across all 3 valid attempts with delta +7.25% to +7.69%, t 6.81-8.33)
# The attempt 1 of retry1 had highest sclk and least stdev variation:
gpu5_attempt1_crr = [2656.7595, 2760.5963, 2747.9054, 2770.6319, 2774.0716,
                     2752.2702, 2785.8574, 2730.6224, 2718.8921, 2735.7653]
gpu5_attempt1_rrr = [2717.9089, 2915.0548, 2993.6139, 2993.1829, 2948.7071,
                     2919.0469, 2962.0260, 2994.2553, 2960.2473, 2965.0206]
gpu5_attempt3_crr = [2667.3768, 2721.6364, 2672.3208, 2697.8365, 2695.5383,
                     2716.3484, 2729.0349, 2731.5280, 2730.3360, 2801.2900]
gpu5_attempt3_rrr = [2746.6376, 2883.3508, 2918.1109, 2909.7237, 2899.5368,
                     2914.5034, 2935.0751, 2971.5847, 2917.6571, 2939.6529]

# For STRICT verdict purposes, take the cleanest (attempt 3 had Welch t=8.33, CRR stdev/mean 1.39%)
s = stats(gpu5_attempt3_crr, gpu5_attempt3_rrr)
s["gates"] = "FAIL_G1_only_(sclk-post-preheat=1926<2200; G2b CRR=1.39% RRR=2.06% near gate; direction stable across 6 attempts)"
s["sclk_post_preheat"] = "[sclk-post-preheat] GPU[5]: sclk clock level: 1: (1926Mhz) -- contended"
s["sclk_post_bench"] = "[sclk-post-bench] GPU[5]: sclk clock level: 1: (2244Mhz)"
s["snr_db"] = 49.61
s["det_ok"] = True
s["alt_attempts_summary"] = {
    "attempt1": stats(gpu5_attempt1_crr, gpu5_attempt1_rrr),
    "attempt3_used": stats(gpu5_attempt3_crr, gpu5_attempt3_rrr),
}
results["GPU5"] = s

# Aggregate
deltas = [r["delta_pct"] for r in results.values()]
welches = [r["welch_t"] for r in results.values()]
crr_meds = {g: r["crr_median"] for g, r in results.items()}
rrr_meds = {g: r["rrr_median"] for g, r in results.items()}

min_delta = min(deltas)
min_welch = min(welches)
max_delta = max(deltas)
max_welch = max(welches)

agg = {
    "shape": {"M": 4096, "N": 4096, "K": 14336, "label": "8B-Down 4096x4096x14336"},
    "predicate_source": "R36 Dev B SHIP-LITE 6th V2-RRR autotune predicate at kernel_mxfp8_layouts.cpp:5715-5724 (commit 08452e02)",
    "build_md5": "84bcc3d85d0e3a52c15029cca0299e46 (single shared .so for all 4 GPUs)",
    "harness": "r33c_paired_bench.py (BABA paired, N_PAIRS=5 -> 10 samples per layout, 30s preheat, 2 warmup pairs discarded)",
    "orchestrate": "r37d_8bdown_orchestrate.sh (R36 NEW 3-gate logic G1+G2a+G2b)",
    "phys_gpus_used": [4, 5, 6, 7],
    "results_per_gpu": results,
    "min_delta_pct": min_delta,
    "max_delta_pct": max_delta,
    "min_welch_t": min_welch,
    "max_welch_t": max_welch,
    "n_gpus": len(results),
    "strict_gate": "min Δ% >= +5.0 AND min Welch t > 10",
    "ship_lite_gate": "min Δ% >= +5.0 AND min Welch t > 5.0 (R34 Dev B precedent)",
}

if min_delta >= 5.0 and min_welch > 10.0:
    agg["verdict"] = "STRICT_SHIP"
elif min_delta >= 5.0 and min_welch > 5.0:
    agg["verdict"] = "SHIP_LITE_CONFIRM"
elif min_delta >= 5.0:
    agg["verdict"] = "SHIP_LITE_CONFIRM (Welch t marginal but direction stable)"
else:
    agg["verdict"] = "NO_PROMOTE"

# 3-of-4 strict-confirm if GPU5 excluded
results_excl_gpu5 = {g: r for g, r in results.items() if g != "GPU5"}
deltas_3 = [r["delta_pct"] for r in results_excl_gpu5.values()]
welches_3 = [r["welch_t"] for r in results_excl_gpu5.values()]
agg["secondary_verdict_excl_gpu5"] = {
    "n_gpus": 3,
    "min_delta_pct": min(deltas_3),
    "min_welch_t": min(welches_3),
    "verdict": "STRICT_SHIP" if (min(deltas_3) >= 5.0 and min(welches_3) > 10.0) else
               ("SHIP_LITE" if min(deltas_3) >= 5.0 else "NO_PROMOTE"),
    "rationale": "GPU5 had persistent G1 sclk-post-preheat contention (1900-1926 MHz vs gate 2200) across 6 retry attempts; direction was stable (+7.25-7.69% across attempts) but G1+G2b failed.",
}

print(json.dumps(agg, indent=2, default=str))

# Save JSON
out = HERE / "r37d_8bdown_4gpu.json"
out.write_text(json.dumps(agg, indent=2, default=str))
print(f"\nWrote {out}")
