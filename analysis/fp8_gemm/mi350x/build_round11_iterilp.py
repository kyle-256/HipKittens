#!/usr/bin/env python3
"""Round 11: Apply -mllvm -amdgpu-sched-strategy=iterative-ilp to
all 7 remaining deep-LOSE shapes + 3 WIN-regression-check spot-shapes.

Each (shape, parent_best) gets a `_r11_iterilp` variant compiled.
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

# (label, M, N, K, parent_suffix, parent_flags)  -- aiter competitor in bench_round11
SHAPES = [
    # Priority A: 7 remaining deep-LOSE shapes
    ("DLA1_4096x32768x128256",   4096,  32768, 128256, "_ts_pf6_6_v12_memc",
     "-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("DLA2_128256x32768x4096", 128256,  32768,   4096, "_ts_gm2_v12_memc_dc",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    ("DLA3_4096x32768x28672",    4096,  32768,  28672, "_v20_memc",
     "-DSTEP3_BARRIER_VMCNT=20 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("DLA4_4096x28672x32768",    4096,  28672,  32768, "_u16",
     "-DUNROLL_K=16"),
    ("DLA5_32768x4096x14336",   32768,   4096,  14336, "_ts_gm8_v12",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DSTEP3_BARRIER_VMCNT=12"),
    ("DLA6_4096x32768x14336",    4096,  32768,  14336, "_ts_lgk2_memc",
     "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("DLA7_28672x32768x4096",   28672,  32768,   4096, "_ts_lgk2_v12_memc",
     "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    # Priority B: 3 WIN-regression-check shapes
    ("WIN1_16384x4096x7168",    16384,   4096,   7168, "_ts_lgk2_v20_memc",
     "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=20 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("WIN2_32768x6144x2048",    32768,   6144,   2048, "_ts_gm8_v12",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DSTEP3_BARRIER_VMCNT=12"),
    ("WIN3_4096x128256x32768",   4096, 128256,  32768, "_memc",
     "-mllvm -amdgpu-sched-strategy=max-memory-clause"),
]

NEW_SUFFIX = "_r11_iterilp"
SCHED_FLAG = "-mllvm -amdgpu-sched-strategy=iterative-ilp"

BASE = (
    f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    f"-I/opt/rocm/include/hip "
    f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    f"-shared -fPIC -std=c++20 -w "
    f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm"
)


def build_one(label, N, K, parent_flags, parent_suffix):
    full_suffix = parent_suffix + NEW_SUFFIX
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if os.path.exists(so_path):
        return (label, full_suffix, "cached", 0.0, "")
    with open(KERNEL_SRC, "r") as f:
        src = f.read()
    patched = src.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},"
    )
    wrapper_src = os.path.join(BUILD_DIR, f"wrap_n{N}_k{K}{full_suffix}.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)
    cmd = (
        f"/opt/rocm/bin/hipcc {wrapper_src} {BASE} "
        f"-DK_DIM={K} -DN_DIM={N} {parent_flags} {SCHED_FLAG} -o {so_path}"
    )
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(so_path):
        return (label, full_suffix, "FAIL", dt, r.stderr[-400:])
    return (label, full_suffix, "OK", dt, "")


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    workers = int(os.environ.get("BUILD_WORKERS", "10"))
    print(f"Total builds: {len(SHAPES)}  (workers={workers})")
    t0 = time.time()
    fail_lines = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_one, lab, N, K, pf, ps): lab
                for (lab, M, N, K, ps, pf) in SHAPES}
        for fut in as_completed(futs):
            label, full_suffix, status, dt, err = fut.result()
            print(f"  {label:32s} {full_suffix:36s} {status:8s} ({dt:.1f}s)", flush=True)
            if status == "FAIL":
                fail_lines.append(f"--- {label} {full_suffix} ---\n{err}\n")
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    if fail_lines:
        print("\nFAILURES:\n" + "\n".join(fail_lines))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
