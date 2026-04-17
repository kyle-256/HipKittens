#!/usr/bin/env python3
"""ASM-diff probe for R16A iterilp+regclassglob compound variants.

For each of the 5 winning shapes, hash the gfx950 .text bytes of:
  (a) original parent  (no iterilp, no regclassglob)
  (b) existing iterilp-only winner (already-shipped)
  (c) NEW R16A compound (iterilp + regclassglob)

Verdicts:
  vs_parent  : DIFF iff (c) differs from (a)
  vs_iterilp : DIFF iff (c) differs from (b) — this is the meaningful one
               (NOOP here = regclassglob added nothing on top of iterilp)
"""
import hashlib, os, subprocess, sys, sysconfig, json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
TMP = "/tmp/r16a_probe"
os.makedirs(TMP, exist_ok=True)

OBJCOPY = "/opt/rocm/llvm/bin/llvm-objcopy"
BUNDLER = "/opt/rocm/llvm/bin/clang-offload-bundler"

# (tag, M, N, K, parent_suffix, iterilp_winner_suffix, compound_suffix)
JOBS = [
    ("S1", 14336,  4096, 32768,
     "_lgk2_dc",
     "_v16_wpe2_r10a_iterilp",
     "_lgk2_dc_r16a_iterilp_regclassglob"),
    ("S2", 16384,  4096, 28672,
     "_u32",
     "_u8_r10a_iterilp",
     "_u32_r16a_iterilp_regclassglob"),
    ("S3",  4096, 32768, 28672,
     "_v20_memc",
     "_v20_memc_r11_iterilp",
     "_v20_memc_r16a_iterilp_regclassglob"),
    ("S4",  4096, 28672, 32768,
     "_u16",
     "_u16_r11_iterilp",
     "_u16_r16a_iterilp_regclassglob"),
    ("S5",  4096, 32768, 14336,
     "_ts_lgk2_memc",
     "_ts_lgk2_memc_r11_iterilp",
     "_ts_lgk2_memc_r16a_iterilp_regclassglob"),
]


def text_hash(so_path, tag):
    if not os.path.exists(so_path):
        return None
    fatbin = os.path.join(TMP, f"{tag}.fatbin")
    co = os.path.join(TMP, f"{tag}.gfx950.co")
    text = os.path.join(TMP, f"{tag}.text.bin")
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


def hash_for(N, K, suffix, tag):
    mod = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix}"
    p = os.path.join(BUILD_DIR, f"{mod}{EXT_SUFFIX}")
    if not os.path.exists(p):
        return None
    return text_hash(p, tag)


def main():
    out = {}
    print(f"{'Tag':4s} {'parent_h':18s} {'iterilp_h':18s} {'compound_h':18s}  vs_parent  vs_iterilp")
    print("-" * 110)
    summary = {"DIFF_iterilp": 0, "NOOP_iterilp": 0, "MISSING": 0}
    for (tag, M, N, K, parent_suf, iterilp_suf, cmp_suf) in JOBS:
        ph = hash_for(N, K, parent_suf, f"{tag}_parent")
        ih = hash_for(N, K, iterilp_suf, f"{tag}_iterilp")
        ch = hash_for(N, K, cmp_suf, f"{tag}_compound")
        if ch is None:
            v_par = v_it = "MISSING"
            summary["MISSING"] += 1
        else:
            v_par = "MISSING" if ph is None else ("NOOP" if ch == ph else "DIFF")
            v_it = "MISSING" if ih is None else ("NOOP" if ch == ih else "DIFF")
            key = "NOOP_iterilp" if v_it == "NOOP" else "DIFF_iterilp"
            summary[key] = summary.get(key, 0) + 1
        print(f"{tag:4s} {str(ph):18s} {str(ih):18s} {str(ch):18s}  {v_par:9s}  {v_it}")
        out[tag] = {
            "M": M, "N": N, "K": K,
            "parent_suffix": parent_suf, "parent_hash": ph,
            "iterilp_suffix": iterilp_suf, "iterilp_hash": ih,
            "compound_suffix": cmp_suf, "compound_hash": ch,
            "vs_parent": v_par, "vs_iterilp": v_it,
        }
    with open(os.path.join(SCRIPT_DIR, "asm_diff_probe_r16a.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + "-" * 110)
    print(f"Summary: {summary}")
    print("Saved asm_diff_probe_r16a.json")


if __name__ == "__main__":
    main()
