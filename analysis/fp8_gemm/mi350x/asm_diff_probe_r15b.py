#!/usr/bin/env python3
"""ASM-diff probe for R15 OptB untested LLVM flag families.

Identifies which R15B candidate flags actually mutate the gfx950 .text
relative to their parent. Verdicts: NOOP / DIFF / MISSING.
"""
import hashlib, os, subprocess, sys, sysconfig, json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
TMP = "/tmp/r15b_probe"
os.makedirs(TMP, exist_ok=True)

OBJCOPY = "/opt/rocm/llvm/bin/llvm-objcopy"
BUNDLER = "/opt/rocm/llvm/bin/clang-offload-bundler"

SHAPES = [
    ("DLA1", 4096,  32768, 128256, "_ts_pf6_6_v12_memc"),
    ("DLA2", 128256, 32768,  4096, "_ts_gm2_v12_memc_dc"),
    ("DLA7", 28672, 32768,  4096, "_ts_lgk2_v12_memc"),
    ("P1",   28672,  4096, 16384, "_ts_gm8"),
]

# Mirror build_round15_optB_newllvm.FLAGS tags
FLAG_TAGS = [
    "nojglc", "nojli", "nojse", "notarsch",
    "largeivf2", "largeivs8", "lateremat0", "revlocal",
    "defspill", "nospillfu", "sinkavoidspill", "preallocs", "noliveopt",
    "nosink", "nolicm", "hoistcheap", "noavoidspec", "sinkbfi", "sinkcycle100",
    "postmi", "enpostmi", "nopostra", "postraN", "breakcrit", "breakall",
    "iglpcut0", "iglpexact",
    "nolal", "looprmul", "loopdist",
    "aaaa", "noaa", "rerag", "eppra", "noeppra", "noemxpre",
    "nodalu", "vopd", "eifcvt",
    "nosgpcb", "nosgpmwc", "nosgpwa",
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
        return "DUMP_FAIL"
    r = subprocess.run([BUNDLER, "--type=o", "--unbundle",
                        f"--input={fatbin}", f"--output={co}",
                        "--targets=hipv4-amdgcn-amd-amdhsa--gfx950"],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(co):
        return "UNBUNDLE_FAIL"
    r = subprocess.run([OBJCOPY, f"--dump-section=.text={text}", co],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(text):
        return "TEXT_DUMP_FAIL"
    with open(text, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    out = {"shapes": {}}
    print(f"{'Shape':6s} {'Tag':16s}  {'text_hash':18s}  {'parent_hash':18s}  verdict")
    print("-" * 95)
    summary_counts = {"DIFF": 0, "NOOP": 0, "MISSING": 0, "OTHER": 0}
    diff_pairs = []
    for (lab, M, N, K, ps) in SHAPES:
        pmod = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{ps}"
        ppath = os.path.join(BUILD_DIR, f"{pmod}{EXT_SUFFIX}")
        ph = text_hash(ppath, f"{lab}_parent")
        out["shapes"][lab] = {"parent": ph, "flags": {}}
        for tag in FLAG_TAGS:
            cs = ps + "_r15b_" + tag
            cmod = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{cs}"
            cpath = os.path.join(BUILD_DIR, f"{cmod}{EXT_SUFFIX}")
            if not os.path.exists(cpath):
                verdict = "MISSING"
                ch = None
            else:
                ch = text_hash(cpath, f"{lab}_{tag}")
                if ch is None:
                    verdict = "MISSING"
                elif ph is not None and isinstance(ph, str) and len(ph) == 16 and \
                     isinstance(ch, str) and len(ch) == 16 and ph == ch:
                    verdict = "NOOP"
                elif isinstance(ch, str) and len(ch) == 16:
                    verdict = "DIFF"
                    diff_pairs.append((lab, tag))
                else:
                    verdict = "OTHER"
            summary_counts[verdict if verdict in summary_counts else "OTHER"] += 1
            print(f"{lab:6s} {tag:16s}  {str(ch):18s}  {str(ph):18s}  {verdict}")
            out["shapes"][lab]["flags"][tag] = {"text_hash": ch, "verdict": verdict}
    out["diff_pairs"] = diff_pairs
    out["summary"] = summary_counts
    with open(os.path.join(SCRIPT_DIR, "asm_diff_probe_r15b.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + "-" * 95)
    print(f"Summary: {summary_counts}")
    print(f"DIFF pairs ({len(diff_pairs)}): {diff_pairs}")
    print("Saved asm_diff_probe_r15b.json")


if __name__ == "__main__":
    main()
