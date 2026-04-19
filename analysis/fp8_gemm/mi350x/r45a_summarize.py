"""R45 Dev A — summarize paired BABA bench results.

Walks r45a_runs/*_clean.txt for paired bench output and
r45a_runs/fp8_*.log for FP8 reference.

Computes per-cell:
  - DECODE / BASELINE / FP8 TFLOPS per GPU
  - DECODE Δ% over BASELINE per GPU + Welch t
  - MXFP8/FP8 ratio per GPU
  - Cross-GPU spread (max - min) of Δ%
  - Min Δ% / min Welch t / min MXFP8/FP8 ratio across GPUs
  - SHIP gate verdicts: ≥95% MXFP8/FP8, min Δ% ≥ +5%, Welch t > 10, spread ≤ 0.6pp
"""
import glob, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(HERE, "r45a_runs")

CELLS = ["4x4kx4k", "4x8kx8k", "8x8kx8k", "16x8kx8k"]
LAYOUTS = ["rcr", "rrr", "crr"]
GPUS = [2, 3, 6, 7]


def parse_paired(fname):
    """Parse one paired bench clean output → dict with decode_tf/base_tf/welch/dpct."""
    if not os.path.exists(fname): return None
    txt = open(fname).read()
    out = {"file": fname}
    m = re.search(r"^DECODE\s+median=([\d.]+)\s+mean=([\d.]+)\s+stdev=([\d.]+)\s+n=(\d+)", txt, re.M)
    if m:
        out["decode_med"] = float(m.group(1))
        out["decode_mean"] = float(m.group(2))
        out["decode_stdev"] = float(m.group(3))
    m = re.search(r"^BASELINE\s+median=([\d.]+)\s+mean=([\d.]+)\s+stdev=([\d.]+)\s+n=(\d+)", txt, re.M)
    if m:
        out["base_med"] = float(m.group(1))
        out["base_mean"] = float(m.group(2))
        out["base_stdev"] = float(m.group(3))
    m = re.search(r"^Welch t \(DECODE vs BASELINE\) = (-?[\d.]+)", txt, re.M)
    if m: out["welch_t"] = float(m.group(1))
    m = re.search(r"^DELTA_MEDIAN_PCT DECODE_vs_BASELINE = ([+\-\d.]+)%", txt, re.M)
    if m: out["delta_pct"] = float(m.group(1))
    m = re.search(r"sclk-post-preheat\].*?\((\d+)Mhz\)", txt)
    if m: out["sclk_pre"] = int(m.group(1))
    m = re.search(r"sclk-post-bench\].*?\((\d+)Mhz\)", txt)
    if m: out["sclk_post"] = int(m.group(1))
    m = re.search(r"CORRECTNESS_DECODE snr_db=([\-\d.]+)", txt)
    if m: out["snr_decode"] = float(m.group(1))
    return out


def parse_fp8(fname):
    if not os.path.exists(fname): return None
    txt = open(fname).read()
    m = re.search(r"^FP8_REF.*tflops=([\d.]+)", txt, re.M)
    if m: return float(m.group(1))
    return None


def cell_summary(cell, layout):
    rows = []
    for gpu in GPUS:
        paired_f = os.path.join(RUN_DIR, f"{cell}_{layout}_gpu{gpu}_clean.txt")
        fp8_f = os.path.join(RUN_DIR, f"fp8_{cell}_{layout}_gpu{gpu}.log")
        p = parse_paired(paired_f)
        fp8_tf = parse_fp8(fp8_f)
        if p is None: continue
        row = {"gpu": gpu, **p, "fp8_tf": fp8_tf}
        if fp8_tf and "decode_med" in p:
            row["mxfp8_over_fp8"] = p["decode_med"] / fp8_tf * 100.0
        rows.append(row)
    return rows


def fmt_pct(v): return "—" if v is None else f"{v:+.2f}%"
def fmt_tf(v): return "—" if v is None else f"{v:.4f}"

print("=" * 100)
print("R45 Dev A — M=2..16 fastpath SHIP triangulation results")
print("=" * 100)

ship_verdicts = {}
for cell in CELLS:
    for layout in LAYOUTS:
        rows = cell_summary(cell, layout)
        if not rows:
            print(f"\n--- {cell} {layout.upper()}: NO DATA ---")
            continue
        print(f"\n--- {cell} {layout.upper()} ({len(rows)} GPUs) ---")
        print(f"  {'GPU':>4} {'DECODE_TF':>11} {'BASE_TF':>10} {'FP8_TF':>9} {'Δ%':>9} "
              f"{'Welch t':>8} {'MXFP8/FP8':>10} {'sclk_pre/post':>14} {'snr':>7}")
        for r in rows:
            mxfp8_pct = r.get("mxfp8_over_fp8")
            print(f"  {r['gpu']:>4} {fmt_tf(r.get('decode_med')):>11} {fmt_tf(r.get('base_med')):>10} "
                  f"{fmt_tf(r.get('fp8_tf')):>9} {fmt_pct(r.get('delta_pct')):>9} "
                  f"{r.get('welch_t', 0):>8.2f} "
                  f"{(f'{mxfp8_pct:.1f}%' if mxfp8_pct else '—'):>10} "
                  f"{r.get('sclk_pre', 0):>5}/{r.get('sclk_post', 0):<6}  "
                  f"{r.get('snr_decode', 0):>5.1f}")
        deltas = [r["delta_pct"] for r in rows if "delta_pct" in r]
        welches = [r["welch_t"] for r in rows if "welch_t" in r]
        mxfp8s = [r["mxfp8_over_fp8"] for r in rows if r.get("mxfp8_over_fp8") is not None]
        decs = [r["decode_med"] for r in rows if "decode_med" in r]
        if deltas:
            spread = max(deltas) - min(deltas)
            min_d = min(deltas); min_t = min(welches) if welches else 0
            min_mxfp8 = min(mxfp8s) if mxfp8s else None
            min_dec = min(decs) if decs else None
            ship_pass = []
            if min_mxfp8 is not None and min_mxfp8 >= 95.0: ship_pass.append("MXFP8/FP8>=95")
            else: ship_pass.append(f"MXFP8/FP8<95 ({min_mxfp8:.1f}%)" if min_mxfp8 is not None else "MXFP8/FP8 NA")
            if min_d >= 5.0: ship_pass.append("Δ%>=+5")
            else: ship_pass.append(f"Δ%<+5 ({min_d:+.2f}%)")
            if min_t > 10: ship_pass.append("Welch>10")
            else: ship_pass.append(f"Welch<=10 ({min_t:.2f})")
            if spread <= 0.6: ship_pass.append("spread<=0.6pp")
            else: ship_pass.append(f"spread>0.6pp ({spread:.2f}pp)")
            n_gpus = len(rows)
            if n_gpus >= 3: ship_pass.append(f"≥3-GPU ({n_gpus})")
            else: ship_pass.append(f"<3-GPU ({n_gpus})")
            verdict = "PASS" if all(s.startswith(("MXFP8/FP8>=", "Δ%>=", "Welch>", "spread<=", "≥3")) for s in ship_pass) else "FAIL"
            print(f"  >>> SUMMARY n_gpus={n_gpus} min_d={min_d:+.2f}% min_t={min_t:.2f} "
                  f"min_mxfp8/fp8={min_mxfp8:.1f}% spread={spread:.2f}pp min_dec={min_dec:.4f}TF")
            print(f"  >>> SHIP gates [{verdict}]: {' | '.join(ship_pass)}")
            ship_verdicts[(cell, layout)] = (verdict, min_d, min_t, min_mxfp8, spread, n_gpus)

print("\n" + "=" * 100)
print("OVERALL SHIP TABLE")
print("=" * 100)
print(f"{'cell':>10} {'layout':>6} {'verdict':>8} {'min_Δ%':>8} {'min_t':>8} {'min_MXFP8/FP8':>14} {'spread':>8} {'#GPUs':>6}")
for (cell, layout), (v, md, mt, mp, sp, ng) in sorted(ship_verdicts.items()):
    print(f"{cell:>10} {layout:>6} {v:>8} {md:>+8.2f} {mt:>8.2f} {mp:>13.1f}% {sp:>7.2f}pp {ng:>6}")

# within-gpu rep summary if available
print("\n" + "=" * 100)
print("WITHIN-GPU N≥3 reps (R45 NEW rule 1) — same .so cross-rep variance on GPU2")
print("=" * 100)
for cell in CELLS:
    deltas = []
    for rep in (1, 2, 3):
        f = os.path.join(RUN_DIR, f"rep{rep}_{cell}_rcr_gpu2_clean.txt")
        p = parse_paired(f)
        if p and "delta_pct" in p:
            deltas.append((rep, p["delta_pct"], p.get("decode_med"), p.get("welch_t")))
    if deltas:
        ds = [d[1] for d in deltas]
        mn, mx = min(ds), max(ds)
        sp = mx - mn
        print(f"  {cell} RCR: reps={[(r, f'{d:+.2f}%') for r,d,_,_ in deltas]} cross-rep spread={sp:.2f}pp")
