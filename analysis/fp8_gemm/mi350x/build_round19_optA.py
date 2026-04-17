#!/usr/bin/env python3
"""Round 19 OptA: build BARRIER_TO_WAITCNT_{STEP3,STEP12,ALL} variants for 15
candidate shapes (the 18 LOSE shapes minus the 3 known-broken DLA-like K=128256
or P1-already-won shapes).

Builds happen sequentially with `make` (the kernel build is itself parallelized,
and we run on the host — no GPU needed for compile). 45 builds total.
"""
import os, subprocess, sys, sysconfig, time, json
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")
TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT",
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..")))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

# Map parent variant suffix to its CPPFLAGS (lifted from bench_all_42.py).
PARENT_FLAGS = {
    "_ts_gm8":            "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8",
    "_lgk2_dc":           "-DSTEP12_BR_LGKMCNT=2 -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule",
    "_u32":               "-DUNROLL_K=32",
    "_v20_memc":          "-DSTEP3_BARRIER_VMCNT=20 -mllvm -amdgpu-sched-strategy=max-memory-clause",
    "_u16":               "-DUNROLL_K=16",
    "_ts_gm8_v12":        "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DSTEP3_BARRIER_VMCNT=12",
    "_ts_lgk2_memc":      "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -mllvm -amdgpu-sched-strategy=max-memory-clause",
    "_ts_lgk2_v12_memc":  "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause",
    "_ts_pf4_memc":       "-DTAIL_SPLIT=1 -DSTEP3_PF_N=4 -DSTEP4_PF_N=4 -mllvm -amdgpu-sched-strategy=max-memory-clause",
    "_ts_gm2_v12_memc_dc":"-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule",
    "_ts_gm2_v12_memc":   "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause",
    "_ts_lgk2_memc_dc":   "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -mllvm -amdgpu-sched-strategy=max-memory-clause -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule",
    "_ts_v12_tv0_memc":   "-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=12 -DTAIL_BARRIER_VMCNT=0 -mllvm -amdgpu-sched-strategy=max-memory-clause",
    "_ts_lgk2":           "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2",
}

# 15 candidates: (label, M, N, K, comp, parent_suffix)
SHAPES = [
    ("S1",  14336,  4096, 32768, 5245.4, "_lgk2_dc"),
    ("S2",  16384,  4096, 28672, 5525.3, "_u32"),
    ("S3",   4096, 32768, 28672, 5568.2, "_v20_memc"),
    ("S4",   4096, 28672, 32768, 5649.9, "_u16"),
    ("S5",  32768,  4096, 14336, 5223.4, "_ts_gm8_v12"),
    ("S6",   4096, 32768, 14336, 5296.1, "_ts_lgk2_memc"),
    ("S7",  28672, 32768,  4096, 4466.6, "_ts_lgk2_v12_memc"),
    ("S8",   4096, 32768,  6144, 4548.6, "_ts_pf4_memc"),
    ("S9",  16384, 28672,  2048, 3482.3, "_ts_gm2_v12_memc_dc"),
    ("S10", 16384, 28672,  4096, 4411.7, "_ts_gm2_v12_memc"),
    ("S11", 28672,  4096,  8192, 4810.0, "_ts_lgk2_memc_dc"),
    ("S12", 14336, 32768,  4096, 4462.6, "_ts_v12_tv0_memc"),
    ("S13",  4096, 14336, 16384, 5013.0, "_ts_lgk2"),
    ("S14",  6144,  4096, 16384, 4428.1, "_ts_lgk2"),
    ("S15", 32768,  4096,  7168, 4666.8, "_ts_gm8_v12"),
]
VARIANTS = ["_r19a_step3", "_r19a_step12", "_r19a_all"]
VARIANT_FLAGS = {
    "_r19a_step3":  "-DBARRIER_TO_WAITCNT_STEP3=1",
    "_r19a_step12": "-DBARRIER_TO_WAITCNT_STEP12=1",
    "_r19a_all":    "-DBARRIER_TO_WAITCNT_ALL=1",
}


def build_one(n, k, parent_suffix, parent_flags, var_suffix, var_flags):
    suffix = parent_suffix + var_suffix
    module_name = f"tk_mxfp4_gluon_cpp_n{n}_k{k}{suffix}"
    out_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if os.path.exists(out_path):
        return module_name, out_path, "cached"

    with open(KERNEL_SRC, "r") as f:
        src = f.read()
    patched = src.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},",
    )
    wrapper_src = os.path.join(BUILD_DIR, f"wrap_n{n}_k{k}{suffix}.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)

    extra = f"-DK_DIM={k} -DN_DIM={n} {parent_flags} {var_flags}"
    cmd = (
        f'make -C {SCRIPT_DIR} TARGET={os.path.join(BUILD_DIR, module_name)} '
        f'SRC={wrapper_src} CPPFLAGS="{extra}"'
    )
    env = os.environ.copy()
    env["THUNDERKITTENS_ROOT"] = TK_ROOT
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    el = time.time() - t0
    if r.returncode != 0 or not os.path.exists(out_path):
        return module_name, out_path, f"FAIL ({el:.1f}s): {r.stderr[-300:]}"
    return module_name, out_path, f"OK ({el:.1f}s)"


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    tasks = []
    for (lab, M, N, K, comp, ps) in SHAPES:
        if ps not in PARENT_FLAGS:
            print(f"!! {lab}: unknown parent {ps}; skip")
            continue
        for vs in VARIANTS:
            tasks.append((lab, M, N, K, ps, vs))
    print(f"Total builds: {len(tasks)}")
    t0 = time.time()
    # 8 parallel make jobs (CPU build, no GPU)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = []
        for (lab, M, N, K, ps, vs) in tasks:
            futs.append((lab, M, N, K, ps, vs,
                ex.submit(build_one, N, K, ps, PARENT_FLAGS[ps], vs, VARIANT_FLAGS[vs])))
        for (lab, M, N, K, ps, vs, fut) in futs:
            mn, op, status = fut.result()
            print(f"  [{lab}] {ps}{vs}  -> {status}", flush=True)
    print(f"Done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
