#!/usr/bin/env python3
"""ASM-diff probe for R14 OptA non-scheduler flag variants.

For each (shape, flag) combo, extract the gfx950 .text bytes via
llvm-objcopy + clang-offload-bundler and SHA256 hash. Compare to
the PARENT (no R14 flag added) hash; identical .text => silent no-op.

Identifies BROKEN/MISSING variants and NOOP variants up front so we
don't waste bench cycles on either category.
"""
import hashlib, os, subprocess, sys, sysconfig, json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
TMP = "/tmp/r14a_probe"
os.makedirs(TMP, exist_ok=True)

OBJCOPY = "/opt/rocm/llvm/bin/llvm-objcopy"
BUNDLER = "/opt/rocm/llvm/bin/clang-offload-bundler"

SHAPES = [
    ("DLA1", 4096,  32768, 128256, "_ts_pf6_6_v12_memc"),
    ("DLA2", 128256, 32768,  4096, "_ts_gm2_v12_memc_dc"),
    ("DLA7", 28672, 32768,  4096, "_ts_lgk2_v12_memc"),
    ("P1",   28672,  4096, 16384, "_ts_gm8"),
]

FLAG_TAGS = [
    "mb0", "mb50", "mb100", "mb200",
    "relaxocc", "mb0_relaxocc",
    "pav16", "pav32", "pav64", "nopav",  # all known broken
    "nodiv", "nomergem0", "nomisched",
    "dppc", "nodppc", "vgprix",
    "regclassglob",
    "nolowoccr", "nohighrpr",
    "lst16", "lst256",
    "sghazard0", "sghazard64",
    "wpv0", "wpv500",
    "lwt50", "lwt100",
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


def main():
    out = {"shapes": {}}
    print(f"{'Shape':6s} {'Tag':14s}  {'text_hash':18s}  {'parent_hash':18s}  verdict")
    print("-" * 90)
    summary_counts = {"DIFF": 0, "NOOP": 0, "MISSING": 0, "OTHER": 0}
    for (lab, M, N, K, ps) in SHAPES:
        pmod = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{ps}"
        ppath = os.path.join(BUILD_DIR, f"{pmod}{EXT_SUFFIX}")
        ph = text_hash(ppath, f"{lab}_parent")
        out["shapes"][lab] = {"parent": ph, "flags": {}}
        for tag in FLAG_TAGS:
            cs = ps + "_r14a_" + tag
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
                else:
                    verdict = "OTHER"
            summary_counts[verdict if verdict in summary_counts else "OTHER"] += 1
            print(f"{lab:6s} {tag:14s}  {str(ch):18s}  {str(ph):18s}  {verdict}")
            out["shapes"][lab]["flags"][tag] = {"text_hash": ch, "verdict": verdict}
    with open(os.path.join(SCRIPT_DIR, "asm_diff_probe_r14a.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + "-" * 90)
    print(f"Summary: {summary_counts}")
    print("Saved asm_diff_probe_r14a.json")


if __name__ == "__main__":
    main()
