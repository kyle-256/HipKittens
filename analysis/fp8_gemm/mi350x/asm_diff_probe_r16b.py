#!/usr/bin/env python3
"""ASM-diff probe for R16B compound variants.

For each (S1-S5, R15B safe-DIFF flag) compound, hash-check .text vs:
  - parent baseline (no iterilp, no R15B flag)
  - iterilp-only baseline (iterilp, no R15B flag)

Verdict per compound:
  - CRASH/MISSING: build artifact problem
  - NOOP_VS_BASE: text == parent baseline (no compile effect at all)
  - NOOP_VS_ILP:  text == iterilp baseline (R15B flag had no effect on top of iterilp)
  - DIFF:         text differs from BOTH (real compound mutation, candidate for bench)
"""
import hashlib, json, os, subprocess, sys, sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
TMP = "/tmp/r16b_probe"
os.makedirs(TMP, exist_ok=True)

OBJCOPY = "/opt/rocm/llvm/bin/llvm-objcopy"
BUNDLER = "/opt/rocm/llvm/bin/clang-offload-bundler"

# (lab, M, N, K, parent_suffix, iterilp_suffix)
SHAPES = [
    ("S1", 14336, 4096, 32768, "_lgk2_dc",        "_lgk2_dc_r10_iterilp"),
    ("S2", 16384, 4096, 28672, "_u32",            "_u32_r10_iterilp"),
    ("S3",  4096, 32768, 28672, "_v20_memc",      "_v20_memc_r11_iterilp"),
    ("S4",  4096, 28672, 32768, "_u16",           "_u16_r11_iterilp"),
    ("S5",  4096, 32768, 14336, "_ts_lgk2_memc",  "_ts_lgk2_memc_r11_iterilp"),
]

R16B_INFIX = "_r16b_iterilp_"

FLAG_TAGS = ["noemxpre", "largeivf2", "nolicm", "sinkavoidspill"]


def text_hash(N, K, suffix, key):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return None
    fatbin = os.path.join(TMP, f"{key}.fatbin")
    co = os.path.join(TMP, f"{key}.gfx950.co")
    text = os.path.join(TMP, f"{key}.text.bin")
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
    print(f"{'Shape':6s} {'Tag':16s}  {'cand_hash':18s}  {'iterilp_hash':18s}  {'parent_hash':18s}  verdict")
    print("-" * 110)
    counts = {"DIFF": 0, "NOOP_VS_ILP": 0, "NOOP_VS_BASE": 0, "MISSING": 0, "OTHER": 0}
    diff_pairs = []
    for (lab, M, N, K, parent_suf, ilp_suf) in SHAPES:
        ph = text_hash(N, K, parent_suf, f"{lab}_parent")
        ih = text_hash(N, K, ilp_suf, f"{lab}_iterilp")
        out["shapes"][lab] = {"parent_hash": ph, "iterilp_hash": ih, "flags": {}}
        for tag in FLAG_TAGS:
            cand_suf = parent_suf + R16B_INFIX + tag
            ch = text_hash(N, K, cand_suf, f"{lab}_{tag}")
            if ch is None or not isinstance(ch, str) or len(ch) != 16:
                verdict = "MISSING" if ch is None else "OTHER"
            elif isinstance(ih, str) and len(ih) == 16 and ch == ih:
                verdict = "NOOP_VS_ILP"
            elif isinstance(ph, str) and len(ph) == 16 and ch == ph:
                verdict = "NOOP_VS_BASE"
            else:
                verdict = "DIFF"
                diff_pairs.append((lab, tag))
            counts[verdict] += 1
            print(f"{lab:6s} {tag:16s}  {str(ch):18s}  {str(ih):18s}  {str(ph):18s}  {verdict}")
            out["shapes"][lab]["flags"][tag] = {
                "cand_hash": ch, "verdict": verdict
            }
    out["diff_pairs"] = diff_pairs
    out["summary"] = counts
    with open(os.path.join(SCRIPT_DIR, "asm_diff_probe_r16b.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + "-" * 110)
    print(f"Summary: {counts}")
    print(f"DIFF pairs ({len(diff_pairs)}): {diff_pairs}")
    print("Saved asm_diff_probe_r16b.json")


if __name__ == "__main__":
    main()
