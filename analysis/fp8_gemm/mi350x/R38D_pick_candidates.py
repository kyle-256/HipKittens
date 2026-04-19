#!/usr/bin/env python3
"""R38 Opt D: For each WRONG_OUTPUT shape from R37, extract top candidate variants
from the R25 sweep that DO NOT contain the suspected aggressive-scheduler flags.

We exclude variants that contain `memc`, `dc`, or `tv0` in their name (the
R37-verdict-named "scheduler-aggressive" axes). We keep `_v12` for now since
many shapes already use it. If no clean winners, we relax the gate further.
"""
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 19 WRONG_OUTPUT shapes from R37 leaderboard (ordered as in leaderboard).
WRONG_OUTPUT_SHAPES = [
    (4096,   4096, 32768),
    (4096,   6144, 32768),
    (4096,  28672, 32768),
    (4096,  32768,  4096),
    (4096,  32768, 14336),
    (4096,  32768, 28672),
    (4096,  32768,128256),
    (4096, 128256, 32768),
    (6144,  32768,  4096),
    (14336,  4096, 32768),
    (16384,  4096, 14336),
    (16384,  4096, 28672),
    (16384, 14336,  2048),
    (16384, 14336,  4096),
    (16384, 28672,  4096),
    (32768,  4096,  3072),
    (32768,  4096, 14336),
    (32768, 14336,  2048),
]

# Bug in instructions: the leaderboard says 19 WRONG_OUTPUT, but list above is 18.
# Cross-reference the leaderboard for the missing entry.
# Re-counting leaderboard: lines 21,22,25,26,28,29,30,31,34,35,42,43,46,47,49,
# 54,56,58 — that's 18. Actually line 37 (16384,4096,2048 finite=0.6372) too — 19.
WRONG_OUTPUT_SHAPES.append((16384,  4096,  2048))

# Filter rules: skip variants whose name contains any of these substrings.
EXCLUDE_SUBSTRS = ["memc", "dc", "tv0"]

# Keep top N candidates per shape.
TOP_N = 10


def load_r25():
    with open(os.path.join(SCRIPT_DIR, "bench_all42_results_R25_FINAL.json")) as f:
        return json.load(f)


def parse_variants_from_bench():
    import re
    bench_path = os.path.join(SCRIPT_DIR, "bench_all_42.py")
    with open(bench_path) as f:
        src = f.read()
    pat = re.compile(r"variants\s*=\s*\[(.*?)^\s*\]", re.DOTALL | re.MULTILINE)
    m = pat.search(src)
    if not m:
        raise RuntimeError("could not parse variants from bench_all_42.py")
    src_list = "variants = [\n" + m.group(1) + "\n]"
    ns = {}
    exec(src_list, ns)
    return ns["variants"]


def is_clean(variant_name):
    """Reject variants with aggressive-scheduler flags."""
    return not any(s in variant_name for s in EXCLUDE_SUBSTRS)


def pick_for_shape(shape_result, all_variants_set, top_n=TOP_N):
    """Return a sorted list of (variant_name, tflops) candidates for this shape."""
    pv = shape_result.get("per_variant", {})
    # Filter to known clean variants we have flags for
    cands = []
    for vname, tflops in pv.items():
        if not isinstance(tflops, (int, float)) or tflops <= 0:
            continue
        # The R25 keys are missing the leading underscore — we'll match later.
        suffix_form = "_" + vname if vname != "default" else ""
        # Check the suffix form exists in the variant defs
        if suffix_form not in all_variants_set:
            continue
        if not is_clean(vname):
            continue
        cands.append((vname, tflops))
    cands.sort(key=lambda x: -x[1])
    return cands[:top_n]


def main():
    r25 = load_r25()
    results = r25["results"]

    # Build (M,N,K) -> result dict
    by_shape = {(r["M"], r["N"], r["K"]): r for r in results}

    variants = parse_variants_from_bench()
    all_variants_set = {sfx for sfx, _ in variants}

    wrong_set = set(WRONG_OUTPUT_SHAPES)
    print(f"Total WRONG_OUTPUT shapes: {len(WRONG_OUTPUT_SHAPES)}")

    # The 19th shape may or may not be in R25 sweep. Let's see.
    out = {}
    missing_in_r25 = []
    no_clean_winner = []

    for shape in WRONG_OUTPUT_SHAPES:
        m, n, k = shape
        if shape not in by_shape:
            missing_in_r25.append(shape)
            print(f"  MISSING {m}x{n}x{k}: shape not in R25 sweep")
            out[f"{m}x{n}x{k}"] = {"status": "missing_in_r25", "candidates": []}
            continue

        r = by_shape[shape]
        comp = r["comp"]
        # Gather clean candidates
        cands = pick_for_shape(r, all_variants_set, top_n=TOP_N)
        if not cands:
            no_clean_winner.append(shape)
            print(f"  NO CLEAN {m}x{n}x{k}: no non-(memc/dc/tv0) variant in R25")
            out[f"{m}x{n}x{k}"] = {"status": "no_clean_winner", "candidates": []}
            continue
        # Status
        comp_str = f"comp={comp:.0f}"
        top1 = cands[0]
        ratio = top1[1] / comp
        marker = "WIN" if ratio >= 1.0 else f"loss={ratio*100:.1f}%"
        print(f"  {m}x{n}x{k:>6}: top1={top1[0]} {top1[1]:.0f} ({marker}, {comp_str})")
        out[f"{m}x{n}x{k}"] = {
            "status": "have_candidates",
            "M": m, "N": n, "K": k, "comp": comp,
            "r25_best": r["best_variant"],
            "r25_best_tflops": r["tflops"],
            "candidates": [{"variant": v, "tflops": t, "ratio_vs_comp": round(t/comp*100, 1)} for v, t in cands],
        }

    out_path = os.path.join(SCRIPT_DIR, "R38D_candidates.json")
    with open(out_path, "w") as f:
        json.dump({
            "filter_rules": {
                "exclude_substrs": EXCLUDE_SUBSTRS,
                "top_n": TOP_N,
            },
            "missing_in_r25": [f"{m}x{n}x{k}" for (m, n, k) in missing_in_r25],
            "no_clean_winner": [f"{m}x{n}x{k}" for (m, n, k) in no_clean_winner],
            "shapes": out,
        }, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"  shapes_with_candidates: {sum(1 for v in out.values() if v['status']=='have_candidates')}")
    print(f"  shapes_missing_in_r25: {len(missing_in_r25)}")
    print(f"  shapes_no_clean_winner: {len(no_clean_winner)}")


if __name__ == "__main__":
    main()
