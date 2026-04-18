"""Parse rocprofv3 CSV output for P23 Step 7 PMC validation.

Aggregates SQ_LDS_BANK_CONFLICT / SQ_INSTS_VALU_MFMA_BF16 ratio per
(layout, shape) and emits PASS/FAIL verdict for the Step 4 AFTER run.

Reused from /tmp/dev_f_parse.py + /tmp/p21_dev_c/parse.

Verdict (per design memo Step 7):
   PASS iff (BANK_CONFLICT/MFMA <= 0.05) AND (BANK_CONFLICT <= 1e6)
   on the RCR shape(s).
   CRR is reported but not gated (Step 4 only changes RCR).
"""
import sys, os, csv, glob, json, re, argparse, collections

THRESHOLD_RATIO = 0.05  # BANK_CONFLICT / MFMA  (~5%)
THRESHOLD_ABS = 1_000_000  # 1M absolute conflicts


def find_csv(d):
    files = sorted(glob.glob(os.path.join(d, "pmc_1", "*", "*_counter_collection.csv")))
    return files[0] if files else None


def parse_csv(csv_path):
    """Aggregate counters across all kernels, then pick the GEMM kernel
    (highest MFMA count)."""
    by_kernel = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            kshort = re.sub(r"<[^>]*>", "", r["Kernel_Name"]).strip()
            kshort = re.sub(r"\s+", " ", kshort)[:80]
            by_kernel[kshort][r["Counter_Name"]].append(float(r["Counter_Value"]))
    # Filter to GEMM-shaped kernels (high MFMA)
    out = {}
    for k, vs in by_kernel.items():
        mfma = sum(vs.get("SQ_INSTS_VALU_MFMA_BF16", [0]))
        if mfma < 1e5:
            continue
        agg = {cn: sum(vals) / len(vals) for cn, vals in vs.items()}
        out[k] = agg
    if not out:
        return None
    # pick highest-MFMA kernel
    best_k = max(out, key=lambda k: out[k].get("SQ_INSTS_VALU_MFMA_BF16", 0))
    return best_k, out[best_k]


def collect(run_dir):
    results = {}
    for sub in sorted(os.listdir(run_dir)):
        full = os.path.join(run_dir, sub)
        if not os.path.isdir(full):
            continue
        m = re.match(r"^(rcr|crr|rrr)_(.+)$", sub)
        if not m:
            continue
        layout, shape = m.group(1), m.group(2)
        csv_path = find_csv(full)
        if not csv_path:
            print(f"  NO CSV for {sub}", file=sys.stderr)
            continue
        parsed = parse_csv(csv_path)
        if parsed is None:
            print(f"  NO GEMM kernel in {sub}", file=sys.stderr)
            continue
        kname, counters = parsed
        results[(layout, shape)] = {
            "kernel": kname,
            "MFMA_BF16": counters.get("SQ_INSTS_VALU_MFMA_BF16", 0),
            "GRBM_GUI_ACTIVE": counters.get("GRBM_GUI_ACTIVE", 0),
            "INSTS_LDS": counters.get("SQ_INSTS_LDS", 0),
            "BANK_CONFLICT": counters.get("SQ_LDS_BANK_CONFLICT", 0),
            "WAIT_INST_LDS": counters.get("SQ_WAIT_INST_LDS", 0),
        }
    return results


def fmt_results(results, tag):
    lines = []
    lines.append(f"\n=== PMC results: {tag} ===")
    lines.append(
        f"{'layout':<6} {'shape':<20} {'MFMA':>14} {'GRBM':>14} {'LDS':>14} "
        f"{'BANK_CONFL':>14} {'WAIT_LDS':>14} {'BC/MFMA':>10}"
    )
    lines.append("-" * 110)
    for (layout, shape), v in sorted(results.items()):
        bc_per = v["BANK_CONFLICT"] / v["MFMA_BF16"] if v["MFMA_BF16"] else 0
        lines.append(
            f"{layout:<6} {shape:<20} "
            f"{v['MFMA_BF16']:>14,.0f} {v['GRBM_GUI_ACTIVE']:>14,.0f} "
            f"{v['INSTS_LDS']:>14,.0f} {v['BANK_CONFLICT']:>14,.0f} "
            f"{v['WAIT_INST_LDS']:>14,.0f} {bc_per:>10.4f}"
        )
    return "\n".join(lines)


def fmt_compare(before, after):
    lines = []
    lines.append("\n=== COMPARE BEFORE vs AFTER ===")
    lines.append(
        f"{'layout':<6} {'shape':<20} "
        f"{'BANK_BEFORE':>14} {'BANK_AFTER':>14} {'delta':>10} "
        f"{'MFMA_BEFORE':>14} {'MFMA_AFTER':>14} "
        f"{'BC/MFMA_BEFORE':>16} {'BC/MFMA_AFTER':>16}"
    )
    lines.append("-" * 140)
    for key in sorted(set(before) | set(after)):
        b = before.get(key, {})
        a = after.get(key, {})
        bc_b = b.get("BANK_CONFLICT", 0)
        bc_a = a.get("BANK_CONFLICT", 0)
        mf_b = b.get("MFMA_BF16", 0)
        mf_a = a.get("MFMA_BF16", 0)
        delta = (bc_a - bc_b) / bc_b if bc_b else (1.0 if bc_a else 0.0)
        rb = bc_b / mf_b if mf_b else 0
        ra = bc_a / mf_a if mf_a else 0
        lines.append(
            f"{key[0]:<6} {key[1]:<20} "
            f"{bc_b:>14,.0f} {bc_a:>14,.0f} {delta * 100:>+9.1f}% "
            f"{mf_b:>14,.0f} {mf_a:>14,.0f} "
            f"{rb:>16.4f} {ra:>16.4f}"
        )
    return "\n".join(lines)


def verdict(results, compare_results=None):
    """RCR-only PASS gate: (BC/MFMA <= 0.05) AND (BC <= 1e6)."""
    lines = []
    lines.append("\n=== VERDICT (RCR-only, gated on Step 4 AFTER) ===")
    rcr_keys = [k for k in results if k[0] == "rcr"]
    overall = True
    for k in sorted(rcr_keys):
        v = results[k]
        bc = v["BANK_CONFLICT"]
        mfma = v["MFMA_BF16"]
        ratio = bc / mfma if mfma else 0
        ok = (ratio <= THRESHOLD_RATIO) and (bc <= THRESHOLD_ABS)
        verdict_str = "PASS" if ok else "FAIL"
        if not ok:
            overall = False
        lines.append(
            f"  {k[0]} {k[1]}: BANK={bc:,.0f}  MFMA={mfma:,.0f}  "
            f"ratio={ratio:.4f}  -> {verdict_str}  "
            f"(threshold: BC<={THRESHOLD_ABS:,.0f} AND ratio<={THRESHOLD_RATIO})"
        )
    if compare_results:
        lines.append("\n=== REGRESSION CHECK (CRR should be unchanged ±5%) ===")
        for k in sorted(results):
            if k[0] != "crr":
                continue
            if k not in compare_results:
                lines.append(f"  {k[0]} {k[1]}: no BEFORE data")
                continue
            bc_a = results[k]["BANK_CONFLICT"]
            bc_b = compare_results[k]["BANK_CONFLICT"]
            if bc_b == 0:
                drift = 0 if bc_a == 0 else float("inf")
            else:
                drift = abs(bc_a - bc_b) / bc_b
            tag = "OK" if drift <= 0.05 else "REGRESSION"
            lines.append(
                f"  {k[0]} {k[1]}: BEFORE={bc_b:,.0f} AFTER={bc_a:,.0f} "
                f"drift={drift * 100:+.2f}% -> {tag}"
            )
    lines.append(f"\nOVERALL (RCR PASS gate): {'PASS' if overall else 'FAIL'}")
    return "\n".join(lines), overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("tag")
    ap.add_argument("--compare", default=None,
                    help="JSON from prior run to diff against")
    args = ap.parse_args()

    results = collect(args.run_dir)
    print(fmt_results(results, args.tag))

    compare_results = None
    if args.compare and os.path.exists(args.compare):
        with open(args.compare) as f:
            raw = json.load(f)
        compare_results = {tuple(k.split("|")): v for k, v in raw["results"].items()}
        print(fmt_compare(compare_results, results))

    v_text, v_pass = verdict(results, compare_results)
    print(v_text)

    # write JSON for future comparison
    out_json = os.path.join(args.run_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "tag": args.tag,
                "threshold_ratio": THRESHOLD_RATIO,
                "threshold_abs": THRESHOLD_ABS,
                "verdict_pass": v_pass,
                "results": {f"{k[0]}|{k[1]}": v for k, v in results.items()},
            },
            f,
            indent=2,
        )
    print(f"\nResults JSON -> {out_json}")
    sys.exit(0 if v_pass else 1)


if __name__ == "__main__":
    main()
