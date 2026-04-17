#!/usr/bin/env python3
"""R17C ASM diff probe: hash .text of each candidate .so vs parent .so.
NOOP candidates -> skip bench."""
import os, sys, json, hashlib, subprocess, sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

SHAPES = [
    ("DLA1", 4096,  32768, 128256, "_ts_pf6_6_v12_memc"),
    ("DLA2", 128256, 32768,  4096, "_ts_gm2_v12_memc_dc"),
    ("DLA7", 28672, 32768,  4096, "_ts_lgk2_v12_memc"),
    ("P1",   28672,  4096, 16384, "_ts_gm8"),
]

ATTRS = ["fwgs64_256", "fwgs256_256",
         "vgpr256", "vgpr224", "vgpr192",
         "sgpr96", "sgpr80", "mnwg8"]

NEW_PREFIX = "_r17c_"


def text_hash(so_path):
    if not os.path.exists(so_path):
        return None
    out = subprocess.run(["objdump", "-d", "-j.text", so_path],
                          capture_output=True, text=True).stdout
    # Strip first lines (filename, format) by hashing only mnemonic bytes
    lines = [l for l in out.splitlines() if "\t" in l]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]


def main():
    out = {"shapes": {}}
    for (lab, M, N, K, ps) in SHAPES:
        parent_so = os.path.join(BUILD_DIR, f"tk_mxfp4_gluon_cpp_n{N}_k{K}{ps}{EXT_SUFFIX}")
        ph = text_hash(parent_so)
        out["shapes"][lab] = {"parent_hash": ph, "attrs": {}}
        for tag in ATTRS:
            cand_so = os.path.join(BUILD_DIR, f"tk_mxfp4_gluon_cpp_n{N}_k{K}{ps}{NEW_PREFIX}{tag}{EXT_SUFFIX}")
            ch = text_hash(cand_so)
            verdict = "MISSING" if ch is None else ("NOOP" if ch == ph else "DIFF")
            out["shapes"][lab]["attrs"][tag] = {"hash": ch, "verdict": verdict}
            print(f"  {lab:6s} {tag:14s} parent={ph} cand={ch} {verdict}")
    json.dump(out, open(os.path.join(SCRIPT_DIR, "asm_diff_probe_r17c.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
