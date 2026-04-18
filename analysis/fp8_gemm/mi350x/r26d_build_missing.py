#!/usr/bin/env python3
"""R26-D: Build all missing K_EXACT variants for matching (N, K) combinations.

The R25_FINAL parallel script lookups variants by .so filename. The R25-G/H
K_EXACT variants need to be pre-built for every (N, K) shape that uses K matching.
"""
import os, subprocess, sys, sysconfig

SCRIPT_DIR = "/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x"
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")
TK_ROOT = "/shared_nfs/kyle/test/HipKittens"

# Map K -> (suffix, cppflags)
K_EXACT_VARIANTS = {
    32768: [
        ("_ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all",
         "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=7 -DSTEP3_BARRIER_VMCNT=12 -DTAIL_BARRIER_VMCNT=0 "
         "-DR25C_TAIL_PF_OFF_ITERS=120 -DR25C_K_LIMIT=32768 -DR25C_K_EXACT=32768 "
         "-mllvm -amdgpu-sched-strategy=max-memory-clause "
         "-mllvm -amdgpu-disable-clustered-low-occupancy-reschedule "
         "-DBARRIER_TO_WAITCNT_ALL=1"),
        ("_ts_lgk2_gm7_memc_pfoff124_kx32768_btw_all",
         "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DGROUP_SIZE_M=7 "
         "-DR25C_TAIL_PF_OFF_ITERS=124 -DR25C_K_LIMIT=32768 -DR25C_K_EXACT=32768 "
         "-mllvm -amdgpu-sched-strategy=max-memory-clause "
         "-DBARRIER_TO_WAITCNT_ALL=1"),
    ],
    14336: [
        ("_ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all",
         "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=7 -DSTEP3_BARRIER_VMCNT=12 -DTAIL_BARRIER_VMCNT=0 "
         "-DR25C_TAIL_PF_OFF_ITERS=54 -DR25C_K_LIMIT=32768 -DR25C_K_EXACT=14336 "
         "-mllvm -amdgpu-sched-strategy=max-memory-clause "
         "-mllvm -amdgpu-disable-clustered-low-occupancy-reschedule "
         "-DBARRIER_TO_WAITCNT_ALL=1"),
    ],
    28672: [
        ("_ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all",
         "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DGROUP_SIZE_M=7 "
         "-DR25C_TAIL_PF_OFF_ITERS=104 -DR25C_K_LIMIT=32768 -DR25C_K_EXACT=28672 "
         "-mllvm -amdgpu-sched-strategy=max-memory-clause "
         "-DBARRIER_TO_WAITCNT_ALL=1"),
    ],
    16384: [
        ("_ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all",
         "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DGROUP_SIZE_M=7 "
         "-DR25C_TAIL_PF_OFF_ITERS=56 -DR25C_K_LIMIT=32768 -DR25C_K_EXACT=16384 "
         "-mllvm -amdgpu-sched-strategy=max-memory-clause "
         "-DBARRIER_TO_WAITCNT_ALL=1"),
    ],
    # K=14336 doesn't match the kx14336 variant (K_EXACT=14336)? Both shapes 21 and 32 have K=14336.
    # Actually shape 32 (16384x4096x14336) → N=4096, K=14336 — needs build.
}

# For each shape, the matching K-based variants must be built for that (N, K).
# Iterate the 42 shapes.
ALL_SHAPES = [
    (16384,  4096,  2048), (16384,  4096,  3072), (16384,  6144,  2048),
    (32768,  4096,  2048), (32768,  4096,  3072), (32768,  6144,  2048),
    (16384, 14336,  2048), (16384, 28672,  2048), (32768, 14336,  2048),
    (32768, 28672,  2048), (4096,   4096,  16384), (4096,  14336,  16384),
    (6144,   4096,  16384), (4096,   4096,   8192), (4096,   4096,  32768),
    (4096,   6144,  32768), (4096,  14336,   8192), (4096,  28672,  32768),
    (4096,  32768,   4096), (4096,  32768,   6144), (4096,  32768,  14336),
    (4096,  32768,  28672), (4096,  32768, 128256), (4096, 128256,  32768),
    (6144,   4096,   8192), (6144,  32768,   4096), (14336,  4096,  32768),
    (14336, 32768,   4096), (16384,  4096,   4096), (16384,  4096,   6144),
    (16384,  4096,   7168), (16384,  4096,  14336), (16384,  4096,  28672),
    (16384,  6144,   4096), (16384, 14336,   4096), (16384, 28672,   4096),
    (28672,  4096,   8192), (28672,  4096,  16384), (28672, 32768,   4096),
    (32768,  4096,   7168), (32768,  4096,  14336), (128256, 32768,  4096),
]

def build(n, k, suffix, flags):
    module_name = f"tk_mxfp4_gluon_cpp_n{n}_k{k}{suffix}"
    out_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if os.path.exists(out_path):
        return ("cached", module_name)
    with open(KERNEL_SRC) as f: src = f.read()
    patched = src.replace("PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
                          f"PYBIND11_MODULE({module_name},")
    wrapper_src = os.path.join(BUILD_DIR, f"wrap_n{n}_k{k}{suffix}.cpp")
    with open(wrapper_src, "w") as f: f.write(patched)
    env = os.environ.copy(); env["THUNDERKITTENS_ROOT"] = TK_ROOT
    cmd = (f'make -C {SCRIPT_DIR} TARGET={os.path.join(BUILD_DIR, module_name)} '
           f'SRC={wrapper_src} '
           f'CPPFLAGS="-DK_DIM={k} -DN_DIM={n} {flags}"')
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if r.returncode != 0 or not os.path.exists(out_path):
        return ("FAIL", module_name)
    return ("OK", module_name)


def main():
    needed = set()
    for m, n, k in ALL_SHAPES:
        if k in K_EXACT_VARIANTS:
            for suffix, flags in K_EXACT_VARIANTS[k]:
                needed.add((n, k, suffix, flags))
    # Dedupe by (n, k, suffix)
    by_key = {}
    for n, k, s, f in needed:
        by_key[(n, k, s)] = f
    print(f"Need to ensure {len(by_key)} (N, K, suffix) combinations exist.")
    built = 0
    cached = 0
    failed = 0
    for (n, k, suffix), flags in sorted(by_key.items()):
        status, mod = build(n, k, suffix, flags)
        if status == "cached":
            cached += 1
        elif status == "OK":
            built += 1
            print(f"  BUILT: {mod}")
        else:
            failed += 1
            print(f"  FAIL:  {mod}")
    print(f"\nResult: built={built}  cached={cached}  failed={failed}")


if __name__ == "__main__":
    main()
