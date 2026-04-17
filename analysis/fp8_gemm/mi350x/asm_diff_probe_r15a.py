#!/usr/bin/env python3
"""Fresh ASM-diff probe for R15 OptA regclassglob variants.

Hashes the gfx950 .text bytes of each candidate vs its parent.
Confirms the per-shape DIFF/NOOP verdict for `regclassglob` and the
7 P1 compound combos. R14A claimed regclassglob produced DIFF on all
4 shapes; we re-verify here with a clean rebuild.
"""
import hashlib, os, subprocess, sys, sysconfig, json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
TMP = "/tmp/r15a_probe"
os.makedirs(TMP, exist_ok=True)

OBJCOPY = "/opt/rocm/llvm/bin/llvm-objcopy"
BUNDLER = "/opt/rocm/llvm/bin/clang-offload-bundler"

# Phase A: parent suffix -> regclassglob suffix per shape
PHASE_A = [
    ("DLA1", 4096,  32768, 128256,
     "_ts_pf6_6_v12_memc",
     "_ts_pf6_6_v12_memc_r15a_regclassglob"),
    ("DLA2", 128256, 32768,  4096,
     "_ts_gm2_v12_memc_dc",
     "_ts_gm2_v12_memc_dc_r15a_regclassglob"),
    ("DLA7", 28672, 32768,  4096,
     "_ts_lgk2_v12_memc",
     "_ts_lgk2_v12_memc_r15a_regclassglob"),
    ("P1",   28672,  4096, 16384,
     "_ts_gm8",
     "_ts_gm8_r15a_regclassglob"),
]

# Phase B: 7 P1 compounds (parent for each compound is the corresponding non-rcg macro variant)
# But for "DIFF" verdict it's most useful to compare against the bare regclassglob (P1 baseline)
# AND against the bare gm8 parent.  We'll record both deltas.
PHASE_B_COMPOUND = [
    ("P1", 28672, 4096, 16384, "_ts_gm8_r15a_regclassglob_v20"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_r15a_regclassglob_v24"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_r15a_regclassglob_lgk2"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_r15a_regclassglob_extbr"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_r15a_regclassglob_noembed"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_r15a_regclassglob_tv0"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_r15a_regclassglob_tv16"),
]


def text_hash(so_path, tag):
    if not os.path.exists(so_path):
        return None
    fatbin = os.path.join(TMP, f"{tag}.fatbin")
    co     = os.path.join(TMP, f"{tag}.gfx950.co")
    text   = os.path.join(TMP, f"{tag}.text.bin")
    for f in (fatbin, co, text):
        if os.path.exists(f):
            os.remove(f)
    r = subprocess.run([OBJCOPY, f"--dump-section=.hip_fatbin={fatbin}", so_path],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(fatbin):
        return f"DUMP_FAIL"
    r = subprocess.run([BUNDLER, "--type=o", "--unbundle",
                        f"--input={fatbin}", f"--output={co}",
                        "--targets=hipv4-amdgcn-amd-amdhsa--gfx950"],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(co):
        return f"UNBUNDLE_FAIL"
    r = subprocess.run([OBJCOPY, f"--dump-section=.text={text}", co],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(text):
        return f"TEXT_DUMP_FAIL"
    with open(text, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def hash_for(N, K, suffix, tag):
    mod = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix}"
    p = os.path.join(BUILD_DIR, f"{mod}{EXT_SUFFIX}")
    if not os.path.exists(p):
        return None
    return text_hash(p, tag)


def main():
    out = {"phase_a": {}, "phase_b": {}}
    print(f"{'Shape':6s} {'Variant suffix':55s}  {'cand_hash':18s}  {'parent_hash':18s}  verdict")
    print("-" * 130)
    summary = {"DIFF": 0, "NOOP": 0, "MISSING": 0}

    # Phase A
    for (lab, M, N, K, parent_suf, cand_suf) in PHASE_A:
        ph = hash_for(N, K, parent_suf, f"{lab}_parent")
        ch = hash_for(N, K, cand_suf, f"{lab}_cand")
        if ch is None:
            verdict = "MISSING"
        elif ph is not None and ch == ph:
            verdict = "NOOP"
        else:
            verdict = "DIFF"
        summary[verdict] = summary.get(verdict, 0) + 1
        print(f"{lab:6s} {cand_suf:55s}  {str(ch):18s}  {str(ph):18s}  {verdict}")
        out["phase_a"][lab] = {
            "M": M, "N": N, "K": K,
            "parent_suffix": parent_suf, "parent_hash": ph,
            "cand_suffix": cand_suf, "cand_hash": ch,
            "verdict": verdict,
        }

    # Phase B: compare each compound to BOTH bare regclassglob (P1) AND bare gm8 (P1 parent)
    p1_parent_hash = out["phase_a"]["P1"]["parent_hash"]
    p1_rcg_hash    = out["phase_a"]["P1"]["cand_hash"]
    print()
    print(f"P1 parent (_ts_gm8) hash:           {p1_parent_hash}")
    print(f"P1 bare-regclassglob hash:          {p1_rcg_hash}")
    print()
    print(f"{'Shape':6s} {'Variant suffix':55s}  {'cand_hash':18s}  {'vs_parent':10s}  {'vs_rcg':10s}")
    print("-" * 130)
    for (lab, M, N, K, suf) in PHASE_B_COMPOUND:
        ch = hash_for(N, K, suf, f"{lab}_compound_{suf}")
        if ch is None:
            v_parent = v_rcg = "MISSING"
        else:
            v_parent = "NOOP" if ch == p1_parent_hash else "DIFF"
            v_rcg    = "NOOP" if ch == p1_rcg_hash    else "DIFF"
        # primary verdict: did the compound change anything vs the parent (gm8)?
        primary = v_parent
        summary[primary] = summary.get(primary, 0) + 1
        print(f"{lab:6s} {suf:55s}  {str(ch):18s}  {v_parent:10s}  {v_rcg:10s}")
        out["phase_b"][suf] = {
            "M": M, "N": N, "K": K,
            "cand_hash": ch,
            "vs_parent": v_parent, "vs_regclassglob": v_rcg,
        }

    with open(os.path.join(SCRIPT_DIR, "asm_diff_probe_r15a.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + "-" * 130)
    print(f"Summary: {summary}")
    print("Saved asm_diff_probe_r15a.json")


if __name__ == "__main__":
    main()
