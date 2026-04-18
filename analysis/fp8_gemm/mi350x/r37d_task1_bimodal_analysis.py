#!/usr/bin/env python3
"""R37 Dev D Task 1: Apply Dev D's R36 median-of-medians rule retroactively to R31-R36.

R36 Dev D finding: GPU6 R36 was bimodal {763, 783} (true mean ~771); the per-cycle
single 5-iter median had been picking up the high mode in R34/R35 streaks.

This script:
1. Loads the per-(GPU x cycle) tflops_list from R31-R36 baseline JSONs.
2. For each list, detects bimodality (gap-based: max gap between sorted samples
   > GAP_THRESH * intra-cluster spread).
3. Where bimodal, reports the per-cluster medians and the "median-of-modes"
   robust estimate (mean of the two cluster medians).
4. Computes "robust" per-cycle median-of-4 using the mode-aware per-GPU medians.
5. Flags any baseline that meaningfully shifts (>1%) under the new methodology.

Note: the true R36 methodology requires N>=3 BABA replicates per GPU per cycle.
We do NOT have that data for R31-R35 (only 1 replicate of 5 iters per GPU).
This is the best post-hoc reconstruction available without re-running.
"""
import json
import os
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
GAP_THRESH = 0.6  # cluster gap > 0.6 * intra-cluster spread => bimodal


def detect_bimodal(lst):
    """Returns (is_bimodal, lo_cluster, hi_cluster, gap_ratio)."""
    s = sorted(lst)
    if len(s) < 4:
        return False, s, [], 0.0
    # Find the largest gap in sorted samples
    gaps = [(s[i+1] - s[i], i) for i in range(len(s)-1)]
    gap, idx = max(gaps)
    lo = s[:idx+1]
    hi = s[idx+1:]
    if not lo or not hi:
        return False, s, [], 0.0
    # Intra-cluster spread (use range-of-largest cluster as scale)
    lo_spread = lo[-1] - lo[0] if len(lo) > 1 else 0.0
    hi_spread = hi[-1] - hi[0] if len(hi) > 1 else 0.0
    intra = max(lo_spread, hi_spread, 0.5)  # tiny floor to avoid div-by-zero
    ratio = gap / intra
    is_bimodal = ratio > 1.0 / GAP_THRESH and len(lo) >= 1 and len(hi) >= 1
    # Tighter: require gap > GAP_THRESH * mean(intra) AND gap > 1% of median
    med_all = statistics.median(s)
    rel_gap = gap / med_all if med_all > 0 else 0.0
    is_bimodal = ratio > 1.5 and rel_gap > 0.005
    return is_bimodal, lo, hi, ratio


def robust_median(lst):
    """Median-of-modes: if bimodal, return mean of cluster medians;
    else return plain median."""
    bm, lo, hi, _ = detect_bimodal(lst)
    if bm and len(lo) >= 1 and len(hi) >= 1:
        return (statistics.median(lo) + statistics.median(hi)) / 2.0
    return statistics.median(lst)


def load_cycle(cycle):
    """Load per-GPU tflops_list for the 70B-KV V2-CRR baseline of given cycle."""
    fname = f"{cycle.lower()}_reviewer_kv_4gpu.json"
    path = HERE / fname
    if not path.exists():
        return None
    j = json.loads(path.read_text())
    out = {}
    # Schema varies between cycles
    if "results" in j and isinstance(j["results"], list):
        for r in j["results"]:
            gpu = f"GPU{r['phys_gpu']}"
            out[gpu] = r["tflops_list"]
    elif "results" in j and isinstance(j["results"], dict):
        for gpu, r in j["results"].items():
            out[gpu] = r["tflops_list"]
    elif "phase1_runs" in j:
        for gpu, r in j["phase1_runs"].items():
            out[gpu] = r["tflops_list"]
    return out


def main():
    cycles = ["R31", "R32", "R33", "R34", "R35", "R36"]
    print(f"{'Cycle':<5} {'GPU':<5} {'samples':<55} {'old_med':>9} {'new_med':>9} {'shift%':>7} {'flag'}")
    print("-" * 110)
    cycle_summary = {}
    bimodal_events = []
    for cyc in cycles:
        data = load_cycle(cyc)
        if not data:
            continue
        old_meds, new_meds = {}, {}
        for gpu in sorted(data.keys()):
            lst = data[gpu]
            old_med = statistics.median(lst)
            new_med = robust_median(lst)
            shift_pct = (new_med - old_med) / old_med * 100 if old_med else 0.0
            bm, lo, hi, ratio = detect_bimodal(lst)
            flag = ""
            if bm:
                flag = f"BIMODAL (lo={statistics.median(lo):.1f} hi={statistics.median(hi):.1f} gap_ratio={ratio:.1f})"
                bimodal_events.append((cyc, gpu, lo, hi, old_med, new_med, shift_pct))
            elif abs(shift_pct) > 1.0:
                flag = "SHIFT>1%"
            samples_str = "[" + ", ".join(f"{x:.2f}" for x in sorted(lst)) + "]"
            print(f"{cyc:<5} {gpu:<5} {samples_str:<55} {old_med:>9.2f} {new_med:>9.2f} {shift_pct:>+7.2f} {flag}")
            old_meds[gpu] = old_med
            new_meds[gpu] = new_med
        old_med4 = statistics.median(old_meds.values())
        new_med4 = statistics.median(new_meds.values())
        shift4 = (new_med4 - old_med4) / old_med4 * 100 if old_med4 else 0.0
        cycle_summary[cyc] = (old_med4, new_med4, shift4)
        flag4 = "SHIFT>1%" if abs(shift4) > 1.0 else ""
        print(f"{cyc:<5} {'med4':<5} {'(per-GPU summary)':<55} {old_med4:>9.2f} {new_med4:>9.2f} {shift4:>+7.2f} {flag4}")
        print()

    print("=" * 110)
    print("CYCLE SUMMARY (median-of-4):")
    print(f"{'Cycle':<5} {'old_med4':>9} {'new_med4':>9} {'shift%':>7} {'flag'}")
    for cyc, (om, nm, sh) in cycle_summary.items():
        flag = "SHIFT>1%" if abs(sh) > 1.0 else ""
        print(f"{cyc:<5} {om:>9.2f} {nm:>9.2f} {sh:>+7.2f} {flag}")

    print()
    print(f"BIMODAL DETECTIONS: {len(bimodal_events)} (GAP_THRESH={GAP_THRESH}, requires gap>1.5x intra-spread AND >0.5% of median)")
    for cyc, gpu, lo, hi, om, nm, sh in bimodal_events:
        print(f"  {cyc} {gpu}: lo={lo}  hi={hi}  old_med={om:.2f}  new_med={nm:.2f}  shift={sh:+.2f}%")


if __name__ == "__main__":
    main()
