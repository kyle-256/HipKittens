#!/usr/bin/env python3
"""Round 18 Optimizer A — R17A-P3: replace inner-loop s_barrier with s_waitcnt lgkmcnt(0).

Three orthogonal variants per shape:
  STEP3   — drop s_barrier from STEP3 producer→consumer sync (hot path, executes
            every K-iteration). Replaces s_barrier with s_waitcnt lgkmcnt(0).
  STEP12  — drop s_barrier from TAIL_BARRIER (last-iter sync). One-shot, but
            included as orthogonal axis to attribute deltas.
  ALL     — both.

For each of the 4 deep-LOSE shapes we build the 3 variants on top of the parent
flag set (the same flag set used by `bench_all_42.py` for that shape's best variant).
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

# Shape, M_native, N, K, parent_suffix, parent_flags
SHAPES = [
    ("DLA1", 4096,  32768, 128256, "_ts_pf6_6_v12_memc",
     "-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("DLA2", 128256, 32768,  4096, "_ts_gm2_v12_memc_dc",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    ("DLA7", 28672, 32768,  4096, "_ts_lgk2_v12_memc",
     "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("P1",   28672,  4096, 16384, "_ts_gm8",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8"),
]

# (suffix, extra_flags)
VARIANTS = [
    ("_r18a_p3_step3",  "-DBARRIER_TO_WAITCNT_STEP3=1"),
    ("_r18a_p3_step12", "-DBARRIER_TO_WAITCNT_STEP12=1"),
    ("_r18a_p3_all",    "-DBARRIER_TO_WAITCNT_ALL=1"),
]

BASE = (
    f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    f"-I/opt/rocm/include/hip "
    f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    f"-shared -fPIC -std=c++20 -w "
    f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm"
)


def build_one(N, K, full_suffix, parent_flags, extra_flags, force=False):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if force and os.path.exists(so_path):
        os.remove(so_path)
    if os.path.exists(so_path):
        return (full_suffix, "cached", 0.0, "")
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
        f"-DK_DIM={K} -DN_DIM={N} {parent_flags} {extra_flags} -o {so_path}"
    )
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900)
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(so_path):
        return (full_suffix, "FAIL", dt, r.stderr[-500:])
    return (full_suffix, "OK", dt, "")


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    force = "--force" in sys.argv
    jobs = []
    for (lab, M, N, K, ps, pflags) in SHAPES:
        for (vs, ef) in VARIANTS:
            full = ps + vs
            jobs.append((lab, N, K, full, pflags, ef))

    print(f"Round 18 OptA P3 builds: {len(jobs)} (4 shapes x 3 variants)")
    print("=" * 100)
    workers = int(os.environ.get("BUILD_WORKERS", "8"))
    t0 = time.time()
    fail_lines = []
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_one, N, K, full, pflags, ef, force): (lab, full)
                for (lab, N, K, full, pflags, ef) in jobs}
        for fut in as_completed(futs):
            lab, full = futs[fut]
            full_suffix, status, dt, err = fut.result()
            results.append((lab, full_suffix, status, dt, err))
            print(f"  {lab:5s} {full_suffix:55s} {status:8s} ({dt:.1f}s)", flush=True)
            if status == "FAIL":
                fail_lines.append(f"--- {lab} {full_suffix} ---\n{err}\n")
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    if fail_lines:
        print("\nFAILURES:\n" + "\n".join(fail_lines))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
