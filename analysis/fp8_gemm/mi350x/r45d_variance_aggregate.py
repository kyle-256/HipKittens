#!/usr/bin/env python3
"""R45 Dev D — aggregate within-GPU variance across N reps per cell.

Reads r45d_variance_runs/${LABEL}_gpu${GPU}_rep{i}_clean_bench.txt for each
of 4 cells and reports mean / stdev / min / max of DELTA_MEDIAN_PCT plus the
implied ±2σ envelope.  Compares R44 Reviewer Phase 3 reported Δ% against the
envelope.
"""
import os, re, glob, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNDIR = os.path.join(HERE, "r45d_variance_runs")

CELLS = [
    # (label, gpu, R44 reviewer reported Delta%, R44 reviewer abs TF)
    ("p31_70bkv_b1",  4, 29.254, 1022.20),
    ("p32_8bkv_b1",   5, 25.202,  883.42),
    ("p33_8bdown",    5,  8.798, 2961.42),
    ("p34_8bgateup",  4,  5.369, 2523.41),
]

DELTA_RE = re.compile(r"DELTA_MEDIAN_PCT[^=]*=\s*([+-]?[\d.]+)%")
ABS_TF_RE = re.compile(r"^([A-Z][A-Z0-9_]+)\s+median=([\d.]+)", re.MULTILINE)

print(f"{'cell':<18} {'gpu':>3} {'N':>2}   reps (Δ%)              "
      f"mean±sd      min     max    range  envelope (±2σ)        "
      f"R44 Δ%   verdict")
print("-" * 145)

for label, gpu, r44_delta, r44_abs in CELLS:
    pattern = os.path.join(RUNDIR, f"{label}_gpu{gpu}_rep*_clean_bench.txt")
    files = sorted(glob.glob(pattern))
    deltas = []
    abs_b_list = []  # abs TFLOPS of "fast" / "B" arm
    for f in files:
        txt = open(f).read()
        m = DELTA_RE.search(txt)
        if not m: continue
        deltas.append(float(m.group(1)))
        # Capture the abs TF of the "winning" arm — second median line by R44
        # convention (r37: CRR_HBSHRINK; r33c: RRR for one_so_layout).
        for mm in ABS_TF_RE.finditer(txt):
            name, val = mm.group(1), float(mm.group(2))
            if "HBSHRINK" in name or name == "RRR":
                abs_b_list.append(val)
                break
    if not deltas:
        print(f"{label:<18} {gpu:>3}  NO DATA")
        continue
    n = len(deltas)
    mean = statistics.mean(deltas)
    sd = statistics.stdev(deltas) if n > 1 else 0.0
    mn, mx = min(deltas), max(deltas)
    rng = mx - mn
    lo, hi = mean - 2 * sd, mean + 2 * sd
    in_env = "INSIDE" if (lo - 1e-9) <= r44_delta <= (hi + 1e-9) else "OUTSIDE"
    reps_str = ",".join(f"{d:+.2f}" for d in deltas)
    abs_b_med = statistics.median(abs_b_list) if abs_b_list else float("nan")
    print(f"{label:<18} {gpu:>3} {n:>2}   {reps_str:<22}  "
          f"{mean:+6.3f}±{sd:.3f}  {mn:+5.2f}  {mx:+5.2f}  {rng:5.2f}  "
          f"[{lo:+6.2f}, {hi:+6.2f}]  {r44_delta:+6.3f}  {in_env} (R44 abs={r44_abs:.1f}, R45D abs_med={abs_b_med:.1f})")

print()
print("Verdict legend:")
print("  INSIDE  = R44 Reviewer's Δ% is within mean±2σ of within-GPU variance")
print("           → R44 measurement is consistent with single-GPU noise floor")
print("           → no kernel attention warranted (cell is at noise-limited stable margin)")
print("  OUTSIDE = R44 Reviewer's Δ% lies outside mean±2σ envelope")
print("           → R44 measurement is statistically distinct from within-GPU population")
print("           → escalate (cross-GPU silicon-bin or true cycle-drift candidate)")
