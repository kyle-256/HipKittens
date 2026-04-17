#!/usr/bin/env python3
"""R13B ASM-diff gate: hash the device ELF inside each .so and compare
each candidate against its corresponding _r13b_baseline (sched-strategy=default).
"""
import hashlib, os, sys, sysconfig, json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

N, K = 4096, 16384

# (parent_suffix) -> the baseline within that parent
PARENTS = ["_ts_gm8", "_ts_lgk2_v12_memc"]

VARIANTS = [
    "_r13b_baseline",
    "_r13b_iterilp",
    "_r13b_maxilp",
    "_r13b_iterminreg",
    "_r13b_maxocc",
    "_r13b_iteroccexp",
]
STACK = ["_r13b_iterilp_stack_memc", "_r13b_memc_stack_iterilp"]


def extract_co_hash(so_path):
    if not os.path.exists(so_path):
        return None, "missing"
    with open(so_path, "rb") as f:
        data = f.read()
    elf_magic = b"\x7fELF"
    offsets = []
    pos = 0
    while True:
        i = data.find(elf_magic, pos)
        if i < 0:
            break
        offsets.append(i)
        pos = i + 4
    if len(offsets) < 2:
        return None, f"only {len(offsets)} ELF in file"
    last = offsets[-1]
    chunk = data[last:]
    return hashlib.sha256(chunk).hexdigest()[:16], None


def main():
    print(f"{'Parent':24s} {'Variant':30s}  {'hash':18s} {'baseline_hash':18s}  verdict")
    print("-" * 110)
    out = {"parents": {}}
    for psuf in PARENTS:
        # Baseline for this parent = parent + _r13b_baseline
        bsuf = psuf + "_r13b_baseline"
        bmod = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{bsuf}"
        bpath = os.path.join(BUILD_DIR, f"{bmod}{EXT_SUFFIX}")
        bh, berr = extract_co_hash(bpath)
        out["parents"][psuf] = {"baseline": bh, "variants": {}}

        for vsuf in VARIANTS:
            full = psuf + vsuf
            mod = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full}"
            cpath = os.path.join(BUILD_DIR, f"{mod}{EXT_SUFFIX}")
            ch, cerr = extract_co_hash(cpath)
            if ch is None:
                verdict = "MISSING"
            elif bh is None:
                verdict = "NO_BASELINE"
            elif ch == bh:
                verdict = "NOOP" if vsuf != "_r13b_baseline" else "BASELINE"
            else:
                verdict = "DIFF"
            print(f"{psuf:24s} {vsuf:30s}  {str(ch):18s} {str(bh):18s}  {verdict}")
            out["parents"][psuf]["variants"][vsuf] = {
                "hash": ch, "verdict": verdict, "err": cerr,
            }

        # Stacking variants only on _ts_gm8
        if psuf == "_ts_gm8":
            for vsuf in STACK:
                full = psuf + vsuf
                mod = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full}"
                cpath = os.path.join(BUILD_DIR, f"{mod}{EXT_SUFFIX}")
                ch, cerr = extract_co_hash(cpath)
                verdict = ("NOOP" if (ch and bh and ch == bh) else
                           ("MISSING" if ch is None else "DIFF"))
                print(f"{psuf:24s} {vsuf:30s}  {str(ch):18s} {str(bh):18s}  {verdict}")
                out["parents"][psuf]["variants"][vsuf] = {
                    "hash": ch, "verdict": verdict, "err": cerr,
                }

    # Also: cross-check baselines vs the original parent build (the one without _r13b_baseline)
    print("\n=== Cross-check: _r13b_baseline vs ORIGINAL parent .so (sanity) ===")
    for psuf in PARENTS:
        omod = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{psuf}"
        opath = os.path.join(BUILD_DIR, f"{omod}{EXT_SUFFIX}")
        oh, _ = extract_co_hash(opath)
        bh = out["parents"][psuf]["baseline"]
        verdict = "MATCH" if (oh and bh and oh == bh) else "DIFFER"
        print(f"  {psuf:24s} orig={oh}  r13b_baseline={bh}  {verdict}")
        out["parents"][psuf]["original_hash"] = oh
        out["parents"][psuf]["baseline_matches_original"] = (oh == bh)

    with open(os.path.join(SCRIPT_DIR, "asm_diff_probe_r13b_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved asm_diff_probe_r13b_results.json")


if __name__ == "__main__":
    main()
