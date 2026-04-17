#!/usr/bin/env python3
"""R17B ASM-diff probe: hash device ELF inside each compound .so vs:
  - P1 parent _ts_gm8 (baseline)
  - R14A bare regclassglob: _ts_gm8_r14a_regclassglob
  - R15A regclassglob+tv16: _ts_gm8_r15a_regclassglob_tv16
  - R16C regclassglob+noemxpre: _ts_gm8_r16c_regclassglobXnoemxpre

Skip-flag any candidate that hashes identical to parent (NOOP) or to one of
the existing 2-stack winners (would be redundant).
"""
import hashlib, os, sys, sysconfig, json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

SHAPE = ("P1", 28672, 4096, 16384, "_ts_gm8")

# Mirror COMPOUNDS from build_round17_optB_p1_triplestack.py
COMPOUNDS = [
    ("rcg", "noemxpre", "tv16"),
    ("rcg", "noemxpre", "v20"),
    ("rcg", "noemxpre", "lgk2"),
    ("rcg", "noemxpre", "extbr"),
    ("rcg", "tv16", "v20"),
    ("rcg", "tv16", "lgk2"),
    ("rcg", "tv16", "extbr"),
    ("rcg", "noemxpre", "tv16", "v20"),
    ("rcg", "noemxpre", "tv16", "lgk2"),
    ("rcg", "noemxpre", "tv16", "extbr"),
]

NEW_PREFIX = "_r17b_"

REFERENCE_SUFFIXES = {
    "parent":             "_ts_gm8",
    "r14a_regclassglob":  "_ts_gm8_r14a_regclassglob",
    "r15a_rcg_tv16":      "_ts_gm8_r15a_regclassglob_tv16",
    "r16c_rcg_noemxpre":  "_ts_gm8_r16c_regclassglobXnoemxpre",
}


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
    lab, M, N, K, ps = SHAPE
    # Hash references
    ref_hashes = {}
    for name, suf in REFERENCE_SUFFIXES.items():
        mod = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suf}"
        path = os.path.join(BUILD_DIR, f"{mod}{EXT_SUFFIX}")
        h, err = extract_co_hash(path)
        ref_hashes[name] = h
        print(f"REF {name:24s} {suf:50s}  hash={h}  err={err}")
    print("-" * 110)

    out = {"shape": lab, "ref_hashes": ref_hashes, "compounds": [], "summary": {"DIFF_FROM_PARENT_AND_REFS": 0, "MATCH_PARENT": 0, "MATCH_REF": 0, "MISSING": 0}}

    print(f"{'Compound':40s}  {'hash':18s}  vs parent / refs")
    for tags in COMPOUNDS:
        csuf = NEW_PREFIX + "X".join(tags)
        full = ps + csuf
        mod = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full}"
        path = os.path.join(BUILD_DIR, f"{mod}{EXT_SUFFIX}")
        ch, cerr = extract_co_hash(path)
        tag_label = "+".join(tags)
        verdict = ""
        match_ref = None
        if ch is None:
            verdict = "MISSING"
            out["summary"]["MISSING"] += 1
        elif ch == ref_hashes.get("parent"):
            verdict = "MATCH_PARENT"
            out["summary"]["MATCH_PARENT"] += 1
        else:
            for name, h in ref_hashes.items():
                if name == "parent":
                    continue
                if ch == h:
                    verdict = f"MATCH_{name}"
                    match_ref = name
                    out["summary"]["MATCH_REF"] += 1
                    break
            if not verdict:
                verdict = "DIFF_FROM_PARENT_AND_REFS"
                out["summary"]["DIFF_FROM_PARENT_AND_REFS"] += 1
        print(f"{tag_label:40s}  {str(ch):18s}  {verdict}")
        out["compounds"].append({
            "tag": tag_label, "csuf": csuf, "hash": ch,
            "verdict": verdict, "match_ref": match_ref, "err": cerr,
        })
    print(f"\nSummary: {out['summary']}")
    with open(os.path.join(SCRIPT_DIR, "asm_diff_probe_r17b.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved asm_diff_probe_r17b.json")


if __name__ == "__main__":
    main()
