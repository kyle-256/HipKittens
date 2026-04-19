#!/usr/bin/env python3
"""R52 Dev O: parse strict-SCLK A/B logs into a TL;DR table."""
import json
import os
import re
import statistics

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "r52o_results")
CELLS = [
    ("8B_GateUp",  4096, 14336, 4096),
    ("8B_QO",      4096,  4096, 4096),
    ("8B_Down",    4096,  4096, 14336),
    ("70B_GateUp", 4096, 28672, 8192),
    ("70B_QO",     4096,  8192, 8192),
    ("70B_Down",   4096,  8192, 28672),
    ("Cube_8192",  8192,  8192, 8192),
]

TFLOPS_RE = re.compile(r"TFLOPS:\s*([\d.]+)")
SNR_RE    = re.compile(r"SNR:\s*([-\d.]+)\s*dB")
DET_RE    = re.compile(r"Determinism\s*\(3 runs\):\s*(\w+)")

def parse_run(path):
    if not os.path.exists(path): return None
    text = open(path).read()
    m = TFLOPS_RE.search(text)
    return float(m.group(1)) if m else None

def parse_check(path):
    if not os.path.exists(path): return None, None
    text = open(path).read()
    s = SNR_RE.search(text); d = DET_RE.search(text)
    return (float(s.group(1)) if s else None,
            d.group(1) if d else None)

def stats(vals):
    if not vals: return None, None
    med = statistics.median(vals)
    spread = ((max(vals) - min(vals)) / med * 100) if med else None
    return med, spread

def fmt(v, w, prec=2, suffix=""):
    return ("--" + " " * (w - 2)) if v is None else f"{v:>{w}.{prec}f}{suffix}"

def fmt_str(v, w):
    return ("--" + " " * (w - 2)) if v is None else f"{v:>{w}}"

header = (f"{'Cell':<14} {'V1 med':>9} {'V2 med':>9} {'delta':>9} "
          f"{'delta%':>8} {'V1 sp':>7} {'V2 sp':>7} "
          f"{'SNR v1':>7} {'SNR v2':>7} {'det v1':>7} {'det v2':>7} {'verdict':>10}")
print(header)
print("-" * len(header))

rows = []
for name, M, N, K in CELLS:
    v1_runs = [v for r in range(1, 6)
               if (v := parse_run(os.path.join(OUTDIR, f"{name}_v1_run{r}.log"))) is not None]
    v2_runs = [v for r in range(1, 6)
               if (v := parse_run(os.path.join(OUTDIR, f"{name}_v2_run{r}.log"))) is not None]
    v1_med, v1_spread = stats(v1_runs)
    v2_med, v2_spread = stats(v2_runs)
    snr_v1, det_v1 = parse_check(os.path.join(OUTDIR, f"{name}_v1_check.log"))
    snr_v2, det_v2 = parse_check(os.path.join(OUTDIR, f"{name}_v2_check.log"))
    delta = (v2_med - v1_med) if (v1_med and v2_med) else None
    delta_pct = (delta / v1_med * 100) if (delta is not None and v1_med) else None

    if v2_med is None:
        verdict = "PENDING"
    elif snr_v2 is None or snr_v2 < 45 or det_v2 != "PASS":
        verdict = "REFUTED"
    elif delta_pct is None:
        verdict = "PENDING"
    elif delta_pct >= 1.0:
        verdict = "V2-WIN"
    elif delta_pct <= -1.0:
        verdict = "V1-WIN"
    else:
        verdict = "TIE"

    print(f"{name:<14} "
          f"{fmt(v1_med, 9, 1)} {fmt(v2_med, 9, 1)} "
          f"{fmt(delta, 9, 1)} {fmt(delta_pct, 7, 2, '%')} "
          f"{fmt(v1_spread, 6, 2, '%')} {fmt(v2_spread, 6, 2, '%')} "
          f"{fmt(snr_v1, 7, 2)} {fmt(snr_v2, 7, 2)} "
          f"{fmt_str(det_v1, 7)} {fmt_str(det_v2, 7)} "
          f"{verdict:>10}")

    rows.append(dict(cell=name, M=M, N=N, K=K,
                     v1_runs=v1_runs, v2_runs=v2_runs,
                     v1_med=v1_med, v2_med=v2_med,
                     v1_spread=v1_spread, v2_spread=v2_spread,
                     delta_tflops=delta, delta_pct=delta_pct,
                     snr_v1=snr_v1, snr_v2=snr_v2,
                     det_v1=det_v1, det_v2=det_v2,
                     verdict=verdict))

os.makedirs(OUTDIR, exist_ok=True)
with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
    json.dump(rows, f, indent=2)
print(f"\nSummary JSON: {os.path.join(OUTDIR, 'summary.json')}")
