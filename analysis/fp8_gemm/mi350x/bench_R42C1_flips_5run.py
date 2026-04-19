#!/usr/bin/env python3
"""5-run reviewer for R42 Opt C1 — only the shapes that flipped verdict
relative to R41 integration baseline (FLIP_TO_PASS or FLIP_TO_FAIL).

Reads R42_OPT_C_C1_SMOKE.json and R41_INTEGRATION_5RUN.json, finds the diffs,
and reruns those shapes 5x using the same harness as bench_all_42_R42C1.py.
"""
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Borrow harness internals from bench_all_42_R42C1.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "bench_all_42_R42C1",
    os.path.join(SCRIPT_DIR, "bench_all_42_R42C1.py"))
bench_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench_mod)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="4,5")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", default="R42_OPT_C_C1_5RUN.json")
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]

    baseline = json.load(open(os.path.join(SCRIPT_DIR, "R41_INTEGRATION_5RUN.json")))["consensus"]
    smoke = json.load(open(os.path.join(SCRIPT_DIR, "R42_OPT_C_C1_SMOKE.json")))["consensus"]
    manifest = json.load(open(os.path.join(SCRIPT_DIR, "R42C_C1_BUILD_MANIFEST.json")))
    shapes_to_so = manifest["shapes_to_so_path"]
    shapes_to_source = manifest["shapes_to_source"]

    # Build comp lookup
    comp_map = {f"{m}x{n}x{k}": comp for (m, n, k, comp) in bench_mod.ALL_SHAPES}

    flips = []
    for shape, b in baseline.items():
        c = smoke.get(shape)
        if c is None:
            continue
        bvc = b.get("verified_correct", False)
        cvc = c.get("verified_correct", False)
        if bvc != cvc:
            flips.append(shape)

    print(f"Flipped shapes: {len(flips)}")
    for s in flips: print(f"  {s}")

    jobs = []
    for shape in flips:
        m, n, k = (int(x) for x in shape.split("x"))
        jobs.append((m, n, k, comp_map[shape], shapes_to_so[shape], shapes_to_source[shape]))

    print(f"\nR42C1 5-run reviewer: {len(jobs)} flipped shapes, GPUs={gpus}")
    all_runs = []
    for ri in range(args.runs):
        results, elapsed = bench_mod.run_one_pass(gpus, f"run{ri+1}/{args.runs}", jobs)
        all_runs.append({"run": ri + 1, "elapsed_minutes": round(elapsed / 60, 1),
                         "results": results})
        print(f"\n[run{ri+1}/{args.runs}] elapsed={elapsed/60:.1f} min")

    consensus = {}
    for i, (m, n, k, comp, so_path, source) in enumerate(jobs):
        key = f"{m}x{n}x{k}"
        per_shape_runs = [run["results"][i] for run in all_runs]
        agg = bench_mod.aggregate_consensus(per_shape_runs, args.runs, comp)
        agg["M"], agg["N"], agg["K"] = m, n, k
        agg["source"] = source
        agg["so_path"] = so_path
        agg["per_run"] = per_shape_runs
        consensus[key] = agg

    out_path = os.path.join(SCRIPT_DIR, args.out)
    with open(out_path, "w") as f:
        json.dump({
            "round": "R42C_C1_5RUN_FLIPS",
            "n_flipped": len(flips),
            "n_runs": args.runs,
            "consensus": consensus,
        }, f, indent=2)
    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
