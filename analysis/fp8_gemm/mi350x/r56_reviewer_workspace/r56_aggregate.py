#!/usr/bin/env python3
"""Aggregate R56 reviewer baseline_9cell results.

For each (shape, layout):
  - Parse 5 fp8 run logs and 5 mxfp8 run logs
  - Take median TFLOPS
  - Compute MX/FP8 ratio
  - Verdict: PASS if >= 95.00%, else HEADROOM
  - Read SNR/det from check log
"""
import os, re, statistics, json, sys

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "r56_reviewer_results", "baseline_9cell")
OUTDIR = os.path.normpath(OUTDIR)

SHAPES = [
    ("8B_GateUp",  4096, 14336, 4096),
    ("70B_GateUp", 4096, 28672, 8192),
    ("70B_QO",     4096, 8192,  8192),
]
LAYOUTS = ["rcr", "rrr", "crr"]
RUNS = 5

def parse_tflops(log):
    """Extract the per-layout 'Avg time: ... TFLOPS: X' value from a single-layout run log."""
    if not os.path.exists(log):
        return None
    with open(log) as f:
        for line in f:
            m = re.search(r"TFLOPS:\s+([0-9.]+)", line)
            if m:
                return float(m.group(1))
    return None

def parse_check(log):
    if not os.path.exists(log):
        return (None, None, None)
    snr, passrate, det = None, None, None
    with open(log) as f:
        text = f.read()
    m = re.search(r"SNR:\s+([0-9.]+)\s+dB", text)
    if m: snr = float(m.group(1))
    m = re.search(r"Pass rate:\s+\d+/\d+\s+\(([0-9.]+)%\)", text)
    if m: passrate = float(m.group(1))
    m = re.search(r"Determinism\s+\(\d+ runs\):\s+(\w+)", text)
    if m: det = m.group(1)
    return (snr, passrate, det)

results = []
for label, M, N, K in SHAPES:
    for lay in LAYOUTS:
        fp8_vals = []
        mx_vals  = []
        for r in range(1, RUNS+1):
            fp8 = parse_tflops(os.path.join(OUTDIR, f"{label}_fp8_{lay}_run{r}.log"))
            mx  = parse_tflops(os.path.join(OUTDIR, f"{label}_mxfp8_{lay}_run{r}.log"))
            if fp8 is not None: fp8_vals.append(fp8)
            if mx  is not None: mx_vals.append(mx)
        snr, pr, det = parse_check(os.path.join(OUTDIR, f"{label}_check_{lay}.log"))
        fp8_med = statistics.median(fp8_vals) if fp8_vals else float('nan')
        mx_med  = statistics.median(mx_vals)  if mx_vals  else float('nan')
        ratio   = (mx_med / fp8_med * 100.0) if (fp8_med and not (mx_med != mx_med)) else float('nan')
        verdict = "PASS" if ratio >= 95.0 else ("HEADROOM" if ratio == ratio else "MISSING")
        results.append({
            "shape": label, "M":M, "N":N, "K":K, "layout":lay,
            "fp8_runs":fp8_vals, "mx_runs":mx_vals,
            "fp8_med":fp8_med, "mx_med":mx_med,
            "ratio":ratio, "snr":snr, "passrate":pr, "det":det,
            "verdict":verdict,
        })

# Print table
print(f"{'Shape':<12} {'Layout':<6} {'FP8(median)':>12} {'MX(median)':>12} {'MX/FP8 %':>9} {'pp from 95':>10} {'SNR':>7} {'det':>5} {'Verdict':<9}")
print("-"*100)
n_pass = 0
for r in results:
    pp = r['ratio'] - 95.0 if r['ratio']==r['ratio'] else float('nan')
    pp_s = f"{pp:+.2f}" if pp == pp else "n/a"
    snr_s = f"{r['snr']:.2f}" if r['snr'] is not None else "n/a"
    det_s = r['det'] if r['det'] else "n/a"
    print(f"{r['shape']:<12} {r['layout']:<6} {r['fp8_med']:>12.2f} {r['mx_med']:>12.2f} {r['ratio']:>8.2f}% {pp_s:>10} {snr_s:>7} {det_s:>5} {r['verdict']:<9}")
    if r['verdict'] == 'PASS':
        n_pass += 1

print(f"\nPASS count: {n_pass}/9")

# Save JSON
out_json = os.path.join(OUTDIR, "..", "r56_aggregate.json")
with open(out_json, "w") as f:
    json.dump({"results":results, "n_pass":n_pass}, f, indent=2)
print(f"Saved: {out_json}")
