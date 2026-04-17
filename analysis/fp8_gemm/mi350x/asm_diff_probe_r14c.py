#!/usr/bin/env python3
"""Round 14 OptC ASM-diff probe for DLA1.

Hash only the .text segment of the gfx950 .co bundled inside each .so
to detect silent no-op flags (text-identical to parent).

If a variant's text_hash == parent's text_hash, that flag/define has no
codegen effect and is a smoke-test waste. Mark NOOP.
"""
import hashlib, json, os, subprocess, sys, sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
TMP = "/tmp/r14c_probe"
os.makedirs(TMP, exist_ok=True)

OBJCOPY = "/opt/rocm/llvm/bin/llvm-objcopy"
BUNDLER = "/opt/rocm/llvm/bin/clang-offload-bundler"

N, K = 32768, 128256
PARENT_SUFFIX = "_ts_pf6_6_v12_memc"

VARIANT_SUFFIXES = [
    "_r14c_v1_membound50",
    "_r14c_v1_membound200",
    "_r14c_v1_regalloc_bias",
    "_r14c_v1_relaxedocc",
    "_r14c_v1_dceinra",
    "_r14c_v1_dpp",
    "_r14c_v1_loopprefetch",
    "_r14c_v1_aaincg",
    "_r14c_v1_membound200_relaxedocc",
    "_r14c_v2_v4",
    "_r14c_v2_v8",
    "_r14c_v2_v16",
    "_r14c_v2_v20",
    "_r14c_v2_v24",
    "_r14c_v3_brlgk0",
    "_r14c_v3_brlgk4",
    "_r14c_v4_tbv0",
    "_r14c_v4_tbv4",
    "_r14c_v4_tbv16",
    "_r14c_v5_gm1",
    "_r14c_v5_gm2",
    "_r14c_v5_gm8",
    "_r14c_v6_noembed",
    "_r14c_v7_extbr",
    "_r14c_v8_wpe2",
    "_r14c_v9_persistent_b16",
    "_r14c_v9_persistent_b32",
    "_r14c_v10_static_xcd",
    "_r14c_cb_tbv16_brlgk4",
    "_r14c_cb_gm8_tbv16",
    "_r14c_cb_membound200_dpp",
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
    pmod = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{PARENT_SUFFIX}"
    ppath = os.path.join(BUILD_DIR, f"{pmod}{EXT_SUFFIX}")
    ph_old = text_hash(ppath, "parent_old")

    # Use a fresh parent rebuild as the canonical baseline (same flags, current src).
    rb_suffix = "_ts_pf6_6_v12_memc_r14c_PARENT_REBUILD"
    rmod = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{rb_suffix}"
    rpath = os.path.join(BUILD_DIR, f"{rmod}{EXT_SUFFIX}")
    ph = text_hash(rpath, "parent_rebuild")
    print(f"PARENT old-cached  {PARENT_SUFFIX}: {ph_old}")
    print(f"PARENT rebuild     {rb_suffix}:    {ph}")
    print("(old-cached and rebuild differ → kernel source edits since parent was built;")
    print(" comparing each variant against rebuild for true ASM-diff verdict.)")
    print()
    print(f"{'Variant':50s}  {'text_hash':18s}  verdict")
    print("-" * 90)
    out = {"parent_suffix": PARENT_SUFFIX, "parent_hash": ph, "variants": {}}
    for s in VARIANT_SUFFIXES:
        full = PARENT_SUFFIX + s
        cmod = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full}"
        cpath = os.path.join(BUILD_DIR, f"{cmod}{EXT_SUFFIX}")
        ch = text_hash(cpath, s.lstrip("_"))
        if ch is None:
            verdict = "MISSING"
        elif ch == ph:
            verdict = "NOOP"
        elif ch.startswith(("DUMP", "UNBUNDLE", "TEXT")):
            verdict = ch
        else:
            verdict = "DIFF"
        print(f"{full:50s}  {str(ch):18s}  {verdict}")
        out["variants"][full] = {"hash": ch, "verdict": verdict}

    # Cross-detection of duplicate hashes
    print()
    print("Cross-variant duplicate hashes:")
    seen = {}
    for k, v in out["variants"].items():
        h = v["hash"]
        seen.setdefault(h, []).append(k)
    for h, ks in seen.items():
        if len(ks) > 1:
            print(f"  {h}: {ks}")

    with open(os.path.join(SCRIPT_DIR, "asm_diff_probe_r14c.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved asm_diff_probe_r14c.json")


if __name__ == "__main__":
    main()
