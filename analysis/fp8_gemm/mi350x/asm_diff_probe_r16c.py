#!/usr/bin/env python3
"""R16C ASM-diff probe: hash device ELF inside each compound .so vs parent .so.

Uses the .text/last-ELF hash trick (extract last ELF object, sha256 first 16 hex).
"""
import hashlib, os, sys, sysconfig, json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

# Mirror SHAPES + COMPOUNDS from build_round16_optC_stuck_compounds.py
SHAPES = [
    ("DLA1", 4096,  32768, 128256, "_ts_pf6_6_v12_memc"),
    ("DLA2", 128256, 32768,  4096, "_ts_gm2_v12_memc_dc"),
    ("DLA7", 28672, 32768,  4096, "_ts_lgk2_v12_memc"),
    ("P1",   28672,  4096, 16384, "_ts_gm8"),
]

COMPOUNDS = [
    ("iterilp", "regclassglob"),
    ("iterilp", "sinkavoidspill"),
    ("iterilp", "nolicm"),
    ("iterilp", "noemxpre"),
    ("iterilp", "largeivf2"),
    ("regclassglob", "sinkavoidspill"),
    ("regclassglob", "nolicm"),
    ("regclassglob", "noemxpre"),
    ("sinkavoidspill", "nolicm"),
    ("regclassglob", "nolicm", "sinkavoidspill"),
    ("iterilp", "regclassglob", "nolicm"),
    ("iterilp", "regclassglob", "sinkavoidspill"),
]

NEW_PREFIX = "_r16c_"


def extract_co_hash(so_path):
    if not os.path.exists(so_path):
        return None, "missing"
    with open(so_path, "rb") as f:
        data = f.read()
    elf_magic = b"\x7fELF"
    offsets, pos = [], 0
    while True:
        i = data.find(elf_magic, pos)
        if i < 0:
            break
        offsets.append(i)
        pos = i + 4
    if len(offsets) < 2:
        return None, f"only {len(offsets)} ELF in file"
    chunk = data[offsets[-1]:]
    return hashlib.sha256(chunk).hexdigest()[:16], None


def main():
    print(f"{'Shape':6s} {'Compound':50s}  {'hash':18s} {'parent_hash':18s}  verdict")
    print("-" * 110)
    out = {"shapes": {}, "diff_pairs": [], "summary": {"DIFF": 0, "NOOP": 0, "MISSING": 0}}
    for (lab, M, N, K, ps) in SHAPES:
        pmod = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{ps}"
        ppath = os.path.join(BUILD_DIR, f"{pmod}{EXT_SUFFIX}")
        ph, _ = extract_co_hash(ppath)
        out["shapes"][lab] = {"parent_suffix": ps, "parent_hash": ph, "compounds": {}}
        for tags in COMPOUNDS:
            csuf = NEW_PREFIX + "X".join(tags)
            tag_label = "+".join(tags)
            full = ps + csuf
            mod = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full}"
            cpath = os.path.join(BUILD_DIR, f"{mod}{EXT_SUFFIX}")
            ch, cerr = extract_co_hash(cpath)
            if ch is None:
                verdict = "MISSING"
                out["summary"]["MISSING"] += 1
            elif ph is None:
                verdict = "NO_PARENT"
            elif ch == ph:
                verdict = "NOOP"
                out["summary"]["NOOP"] += 1
            else:
                verdict = "DIFF"
                out["summary"]["DIFF"] += 1
                out["diff_pairs"].append([lab, tag_label])
            print(f"{lab:6s} {tag_label:50s}  {str(ch):18s} {str(ph):18s}  {verdict}")
            out["shapes"][lab]["compounds"][tag_label] = {
                "hash": ch, "verdict": verdict, "err": cerr, "csuf": csuf,
            }
    print(f"\nSummary: {out['summary']}")
    with open(os.path.join(SCRIPT_DIR, "asm_diff_probe_r16c.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved asm_diff_probe_r16c.json   diff_pairs={len(out['diff_pairs'])}")


if __name__ == "__main__":
    main()
