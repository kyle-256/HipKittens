#!/usr/bin/env python3
"""Build R41_INTEGRATION_MANIFEST.json by combining per-shape best:
  - R41A for 5 catastrophic K=32768 cluster-C shapes
  - R41B PROMOTE for (16384,4096,14336) and (32768,4096,2048)
  - R40A for (4096,32768,128256)
  - R40B base for the remaining 34 shapes
"""
import json, os, sys, time

SCRIPT_DIR = "/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x"

# Per-shape source overrides (R41A / R41B / R40A)
R41A_PICKS = {
    "4096x4096x32768":   ("R41A", "po32_ef1", "build_R41A/tk_mxfp4_gluon_cpp_n4096_k32768_ts_v12_tv0_dc_gm7_pfoff120_kx32768_btw_all_R41A_po32_ef1.cpython-310-x86_64-linux-gnu.so"),
    "4096x6144x32768":   ("R41A", "po8_ef1",  "build_R41A/tk_mxfp4_gluon_cpp_n6144_k32768_ts_v12_tv0_dc_gm7_pfoff120_kx32768_btw_all_R41A_po8_ef1.cpython-310-x86_64-linux-gnu.so"),
    "4096x28672x32768":  ("R41A", "po32_ef1", "build_R41A/tk_mxfp4_gluon_cpp_n28672_k32768_ts_v12_tv0_dc_gm7_pfoff120_kx32768_btw_all_R41A_po32_ef1.cpython-310-x86_64-linux-gnu.so"),
    "4096x128256x32768": ("R41A", "po8_ef1",  "build_R41A/tk_mxfp4_gluon_cpp_n128256_k32768_ts_v12_tv0_dc_gm7_pfoff120_kx32768_btw_all_R41A_po8_ef1.cpython-310-x86_64-linux-gnu.so"),
    "14336x4096x32768":  ("R41A", "po0_ef1",  "build_R41A/tk_mxfp4_gluon_cpp_n4096_k32768_ts_v12_tv0_dc_gm7_pfoff120_kx32768_btw_all_R41A_po0_ef1.cpython-310-x86_64-linux-gnu.so"),
}

R41B_PICKS = {
    "16384x4096x14336":  ("R41B", "v0b", "build_R41B/tk_mxfp4_gluon_cpp_n4096_k14336_ts_gm8_v12_btw_all_R41B_v0b.cpython-310-x86_64-linux-gnu.so"),
    "32768x4096x2048":   ("R41B", "v3",  "build_R41B/tk_mxfp4_gluon_cpp_n4096_k2048_ts_lgk2_gm6_v12_pfoff4_R41B_v3.cpython-310-x86_64-linux-gnu.so"),
}

R40A_PICKS = {
    "4096x32768x128256": ("R40A", "PF_FENCE1", "build_R40A/tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_v12_memc_btw_all_R40A_PF_FENCE1_R40A.cpython-310-x86_64-linux-gnu.so"),
}

NOTES = {
    "4096x4096x32768":   "R41A po32 ef1 — extract_tile vmcnt fence; po=32 best 5-run pick",
    "4096x6144x32768":   "R41A po8 ef1 — extract_tile vmcnt fence; po=8 best 5-run pick",
    "4096x28672x32768":  "R41A po32 ef1 — extract_tile vmcnt fence; po=32 best stable cell",
    "4096x128256x32768": "R41A po8 ef1 — extract_tile vmcnt fence; po=8 (po=32 had perf cliff)",
    "14336x4096x32768":  "R41A po0 ef1 — extract_tile vmcnt fence; po=0 keeps parent pfoff=120",
    "16384x4096x14336":  "R41B v0b — pfoff=k_iters/8=7 retune on R37 base; supersedes R40C",
    "32768x4096x2048":   "R41B v3 — STEP12_BR_LGKMCNT swap 2->1 on R37 parent (recovers + WIN)",
    "4096x32768x128256": "R40A PF_FENCE1 — promoted by R41D 5-run audit (5/5 PASS)",
}

def main():
    R40B = json.load(open(os.path.join(SCRIPT_DIR, "R40B_BUILD_MANIFEST.json")))
    r40b_modules = R40B["shapes_to_module"]

    shapes_to_so_path = {}
    shapes_to_source = {}
    notes = {}

    overrides = {}
    overrides.update(R41A_PICKS)
    overrides.update(R41B_PICKS)
    overrides.update(R40A_PICKS)

    for shape in r40b_modules:
        if shape in overrides:
            src, variant, rel_path = overrides[shape]
            abs_path = os.path.join(SCRIPT_DIR, rel_path)
            if not os.path.exists(abs_path):
                print(f"MISSING: {shape} -> {abs_path}", file=sys.stderr)
                sys.exit(1)
            shapes_to_so_path[shape] = abs_path
            shapes_to_source[shape] = src
            notes[shape] = NOTES.get(shape, "")
        else:
            mod = r40b_modules[shape]
            abs_path = os.path.join(SCRIPT_DIR, "build_R40B", f"{mod}.cpython-310-x86_64-linux-gnu.so")
            if not os.path.exists(abs_path):
                print(f"MISSING: {shape} -> {abs_path}", file=sys.stderr)
                sys.exit(1)
            shapes_to_so_path[shape] = abs_path
            shapes_to_source[shape] = "R40B"
            notes[shape] = f"R40B base — {mod}"

    out = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "round": "R41_INTEGRATION",
        "total_shapes": len(shapes_to_so_path),
        "shapes_to_so_path": shapes_to_so_path,
        "shapes_to_source": shapes_to_source,
        "notes": notes,
        "source_counts": {
            "R41A": sum(1 for v in shapes_to_source.values() if v == "R41A"),
            "R41B": sum(1 for v in shapes_to_source.values() if v == "R41B"),
            "R40A": sum(1 for v in shapes_to_source.values() if v == "R40A"),
            "R40B": sum(1 for v in shapes_to_source.values() if v == "R40B"),
        },
    }
    out_path = os.path.join(SCRIPT_DIR, "R41_INTEGRATION_MANIFEST.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"Source counts: {out['source_counts']}")
    print(f"Total shapes:  {out['total_shapes']}")

if __name__ == "__main__":
    main()
