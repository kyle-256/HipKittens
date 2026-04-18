#!/usr/bin/env python3
"""R25-F EXTEND2: pfoff=14 still wins → probe pfoff ∈ {15, 16} for gm ∈ {6, 7}.
For K=4096, k_byte_iters=16; pfoff=15 means only 1 iter prefetches; pfoff=16
means NO prefetch in the steady-state main loop (the entire main loop runs
PF_N=0). The R25C macro source line 98 is gated `R25C_TAIL_PF_OFF_ITERS > 0`
so pfoff=16 is mechanically valid.
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

MEMC = "-mllvm -amdgpu-sched-strategy=max-memory-clause"

SHAPES = [
    ("DLA2", 128256, 32768, 4096,
     f"-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=12 {MEMC} -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    ("DLA7", 28672, 32768, 4096,
     f"-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 {MEMC}"),
]

GM_VALUES = [6, 7]
PFOFF_VALUES = [15, 16]

BASE = (
    f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    f"-I/opt/rocm/include/hip "
    f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    f"-shared -fPIC -std=c++20 -w "
    f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm"
)


def build_one(N, K, full_suffix, parent_flags, extra_flags):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if os.path.exists(so_path):
        return (full_suffix, "cached", 0.0, "")
    with open(KERNEL_SRC, "r") as f:
        src = f.read()
    patched = src.replace("PYBIND11_MODULE(tk_mxfp4_gluon_cpp,", f"PYBIND11_MODULE({module_name},")
    wrapper_src = os.path.join(BUILD_DIR, f"wrap_n{N}_k{K}{full_suffix}.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)
    cmd = (f"/opt/rocm/bin/hipcc {wrapper_src} {BASE} "
           f"-DK_DIM={K} -DN_DIM={N} {parent_flags} {extra_flags} -o {so_path}")
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900)
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(so_path):
        return (full_suffix, "FAIL", dt, r.stderr[-1500:])
    return (full_suffix, "OK", dt, "")


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    jobs = []
    for (lab, M, N, K, parent) in SHAPES:
        for gm in GM_VALUES:
            for pfoff in PFOFF_VALUES:
                gm_flag = f"-DGROUP_SIZE_M={gm}"
                pf_flag = f"-DR25C_TAIL_PF_OFF_ITERS={pfoff} -DR25C_K_LIMIT=32768"
                full = f"_r25f_gm{gm}_pfoff{pfoff}_{lab.lower()}"
                jobs.append((lab, N, K, full, parent, f"{gm_flag} {pf_flag}"))
    print(f"R25-F EXTEND2 builds: {len(jobs)}")
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(build_one, N, K, full, pf, ef): (lab, full)
                for (lab, N, K, full, pf, ef) in jobs}
        for fut in as_completed(futs):
            lab, full = futs[fut]
            full_suffix, status, dt, err = fut.result()
            print(f"  {lab:5s} {full_suffix:40s} {status:8s} ({dt:.1f}s)", flush=True)
            if status == "FAIL":
                print(f"    {err[-500:]}")


if __name__ == "__main__":
    main()
