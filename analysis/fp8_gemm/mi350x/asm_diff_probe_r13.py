#!/usr/bin/env python3
"""For each (shape, strategy) variant built in R13A, extract the gfx950 .co
from the .so via llvm-objcopy + clang-offload-bundler. Then strip
the .comment, .note, .symtab, .strtab, .shstrtab, .debug_* sections (which
may contain build-time strings) and hash only the .text segment(s).
Compare to parent + against a freshly-built no-r13 baseline (same parent
flags) for unambiguous diff.
"""
import hashlib, os, subprocess, sys, sysconfig, json, tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
TMP = "/tmp/r13probe"
os.makedirs(TMP, exist_ok=True)

OBJCOPY = "/opt/rocm/llvm/bin/llvm-objcopy"
BUNDLER = "/opt/rocm/llvm/bin/clang-offload-bundler"

SHAPES = [
    ("DLA1", 4096,  32768, 128256, "_ts_pf6_6_v12_memc"),
    ("DLA2", 128256, 32768,  4096, "_ts_gm2_v12_memc_dc"),
    ("DLA7", 28672, 32768,  4096, "_ts_lgk2_v12_memc"),
    ("WIN2", 32768,  6144,  2048, "_ts_gm8_v12"),
]
STRATS = ["iterminreg", "itermaxocc", "maxilp", "maxocc", "itermaxoccx"]


def text_hash(so_path, tag):
    """Return SHA256 of just the gfx950 .text section bytes."""
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
        return f"DUMP_FAIL:{r.stderr[:80]}"
    r = subprocess.run([BUNDLER, "--type=o", "--unbundle",
                        f"--input={fatbin}", f"--output={co}",
                        "--targets=hipv4-amdgcn-amd-amdhsa--gfx950"],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(co):
        return f"UNBUNDLE_FAIL:{r.stderr[:80]}"
    # Dump only the .text section bytes to strip metadata
    r = subprocess.run([OBJCOPY, f"--dump-section=.text={text}", co],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(text):
        return f"TEXT_DUMP_FAIL:{r.stderr[:80]}"
    with open(text, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    out = {"shapes": {}}
    print(f"{'Shape':6s} {'Strat':14s}  {'text_hash':18s}  parent_hash       diff")
    print("-" * 80)
    for (lab, M, N, K, ps) in SHAPES:
        pmod = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{ps}"
        ppath = os.path.join(BUILD_DIR, f"{pmod}{EXT_SUFFIX}")
        ph = text_hash(ppath, f"{lab}_parent")
        out["shapes"][lab] = {"parent": ph, "strats": {}}
        for tag in STRATS:
            cs = ps + f"_r13_{tag}"
            cmod = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{cs}"
            cpath = os.path.join(BUILD_DIR, f"{cmod}{EXT_SUFFIX}")
            ch = text_hash(cpath, f"{lab}_{tag}")
            same = (ph is not None and ch is not None and ph == ch)
            verdict = "NOOP" if same else ("MISSING" if ch is None else "DIFF")
            print(f"{lab:6s} {tag:14s}  {str(ch):18s}  {str(ph):18s}  {verdict}")
            out["shapes"][lab]["strats"][tag] = {
                "text_hash": ch, "verdict": verdict,
            }
    with open(os.path.join(SCRIPT_DIR, "asm_diff_probe_r13_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved asm_diff_probe_r13_results.json")
    # Also detect strats that share hashes with each other (which means
    # those strategies produced byte-identical .text => silent noop pair)
    print("\nCross-strategy duplicate detection:")
    for lab, sd in out["shapes"].items():
        seen = {}
        for tag, info in sd["strats"].items():
            h = info["text_hash"]
            seen.setdefault(h, []).append(tag)
        for h, tags in seen.items():
            if len(tags) > 1:
                print(f"  {lab}: SAME hash {h}: {tags}")


if __name__ == "__main__":
    main()
