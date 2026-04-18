#!/usr/bin/env python3
"""Aggregate R44 Phase 2 R43 Dev B M=1 RRR/CRR reconfirm across 4 GPUs.
Computes per-cell median MXFP8/FP8 ratio + V1-PQ-FALLBACK speedup.
Reports cross-GPU spread (R43 NEW rule 3 Δ%-reproducibility).
"""
import re
import os
import statistics
from collections import defaultdict

GPUS = [2, 3, 6, 7]
HERE = os.path.dirname(os.path.abspath(__file__))
# GPU2 + GPU7 from R44; GPU3 + GPU6 from R43 sweep dir
SOURCES = {
    2: f"{HERE}/sweep_gpu2.log",
    3: f"{HERE}/../r43b_runs/sweep_gpu3.log",
    6: f"{HERE}/../r43b_runs/sweep_gpu6.log",
    7: f"{HERE}/sweep_gpu7.log",
}

# Parse R43B_RESULT lines
PAT = re.compile(
    r"R43B_RESULT module=(\S+) kind=(\S+) impl=(\S+) layout=(\S+) shape=(\S+) avg_ms=([0-9.]+) tflops=([0-9.]+) snr_db=([0-9.]+)"
)

# data[gpu][(impl, layout, shape)] = tflops
data = defaultdict(dict)

for gpu, path in SOURCES.items():
    if not os.path.exists(path):
        print(f"WARN missing {path}")
        continue
    with open(path) as f:
        for line in f:
            m = PAT.search(line)
            if not m:
                continue
            module, kind, impl, layout, shape, ams, tf, snr = m.groups()
            key = (impl, layout, shape)
            data[gpu][key] = float(tf)

# Cells: 4 shapes × 2 layouts (RRR, CRR) = 8 cells
SHAPES = ["1x4096x4096", "1x8192x8192", "1x14336x4096", "1x4096x14336"]
LAYOUTS = ["rrr", "crr"]

# Per cell, compute median across GPUs of: FAST/V1 speedup, FAST TF / FP8 TF ratio, FAST TF abs
print("R44 Phase 2 — R43 Dev B M=1 RRR/CRR reconfirm aggregate (4-GPU)")
print(f"GPUs: {GPUS}")
print()
header = f"{'shape':<14} {'lay':<4} | {'fast_med':>8} {'fast_min':>8} {'fast_max':>8} | {'spd_v1':>8} {'mxfp8/fp8':>10} | {'spread%':>8}"
print(header)
print("-" * len(header))
all_pass = True
results = []
for shape in SHAPES:
    for layout in LAYOUTS:
        fast_tfs = [data[g].get(("decode_m1_rrr_crr", layout, shape)) for g in GPUS]
        v1_tfs = [data[g].get(("legacy_v1pq", layout, shape)) for g in GPUS]
        fp8_tfs = [data[g].get(("fp8_pertensor", layout, shape)) for g in GPUS]
        fast_tfs = [t for t in fast_tfs if t is not None]
        v1_tfs = [t for t in v1_tfs if t is not None]
        fp8_tfs = [t for t in fp8_tfs if t is not None]
        if not fast_tfs or not v1_tfs or not fp8_tfs:
            print(f"{shape:<14} {layout:<4} | INCOMPLETE")
            continue
        fast_med = statistics.median(fast_tfs)
        fast_min = min(fast_tfs)
        fast_max = max(fast_tfs)
        v1_med = statistics.median(v1_tfs)
        fp8_med = statistics.median(fp8_tfs)
        spd_v1 = fast_med / v1_med
        ratio = fast_med / fp8_med * 100
        spread_pct = (fast_max - fast_min) / fast_med * 100
        # PASS gate per task: MXFP8/FP8 >= 95% (R43 reported 954-1097%)
        gate = "PASS" if ratio >= 95 else "FAIL"
        if ratio < 95:
            all_pass = False
        print(f"{shape:<14} {layout:<4} | {fast_med:>8.4f} {fast_min:>8.4f} {fast_max:>8.4f} | {spd_v1:>7.2f}x {ratio:>9.1f}% | {spread_pct:>7.2f}% {gate}")
        results.append((shape, layout, fast_med, spd_v1, ratio, spread_pct))

print()
print(f"Overall: {'PASS' if all_pass else 'FAIL'} (8/8 cells MXFP8/FP8 >= 95% gate)")

# R43 reference numbers from TODO: ratios 954-1097% on RRR/CRR cells; min 688% over 8 cells
print()
print("R43 reported ratios: 954-1097% on RRR/CRR cells; min 688% over 8 shape×layout cells")
print()
# Δ%-reproducibility check (NEW rule 3): per-cell spread<= 0.6pp
# Note: this rule was for predicate Δ%, not absolute TF; here we measure abs spread
print("R43 NEW rule 3 (Δ%-reproducibility): per-cell fast_TF cross-GPU spread (informative; rule applies to Δ%)")
worst_spread = max(r[5] for r in results)
print(f"worst spread (4-GPU range over median): {worst_spread:.2f}%")
