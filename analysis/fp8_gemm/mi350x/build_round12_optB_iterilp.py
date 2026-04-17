#!/usr/bin/env python3
"""Round 12 Optimizer B: Apply -mllvm -amdgpu-sched-strategy=iterative-ilp to
8 untested NEAR-THRESHOLD and MID-LOSE shapes that may flip to WIN.

Each (shape, parent_best) gets a `_r12_iterilp` variant compiled.
Targets are NOT in the Round 11 set and are NOT in the 5 aperture-violation
shapes (4096x32768x128256, 128256x32768x4096, 28672x32768x4096,
16384x4096x7168, 32768x6144x2048).
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

# (label, M, N, K, parent_suffix, parent_flags)
SHAPES = [
    # NEAR-THRESHOLD LOSE (98-99%) -- could flip to WIN
    ("NT1_32768x4096x7168",     32768,   4096,   7168, "_ts_gm8_v12",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DSTEP3_BARRIER_VMCNT=12"),
    ("NT2_4096x14336x16384",     4096,  14336,  16384, "_ts_lgk2",
     "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2"),
    ("NT3_6144x4096x16384",      6144,   4096,  16384, "_ts_lgk2",
     "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2"),
    # MID-LOSE (95-97%)
    ("ML1_16384x28672x2048",    16384,  28672,   2048, "_ts_gm2_v12_memc_dc",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    ("ML2_4096x32768x6144",      4096,  32768,   6144, "_ts_pf4_memc",
     "-DTAIL_SPLIT=1 -DSTEP3_PF_N=4 -DSTEP4_PF_N=4 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("ML3_16384x28672x4096",    16384,  28672,   4096, "_ts_gm2_v12_memc",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("ML4_28672x4096x8192",     28672,   4096,   8192, "_ts_lgk2_memc_dc",
     "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -mllvm -amdgpu-sched-strategy=max-memory-clause -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    ("ML5_14336x32768x4096",    14336,  32768,   4096, "_ts_v12_tv0_memc",
     "-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=12 -DTAIL_BARRIER_VMCNT=0 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
]

NEW_SUFFIX = "_r12_iterilp"
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
    workers = int(os.environ.get("BUILD_WORKERS", "8"))
    print(f"Total builds: {len(SHAPES)}  (workers={workers})")
    t0 = time.time()
    fail_lines = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_one, lab, N, K, pf, ps): lab
                for (lab, M, N, K, ps, pf) in SHAPES}
        for fut in as_completed(futs):
            label, full_suffix, status, dt, err = fut.result()
            print(f"  {label:32s} {full_suffix:40s} {status:8s} ({dt:.1f}s)", flush=True)
            if status == "FAIL":
                fail_lines.append(f"--- {label} {full_suffix} ---\n{err}\n")
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    if fail_lines:
        print("\nFAILURES:\n" + "\n".join(fail_lines))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
