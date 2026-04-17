#!/usr/bin/env python3
"""Round 19 Optimizer B — fine-grained per-site barrier removal.

R18A discovered that BARRIER_TO_WAITCNT_ALL=1 gives +4.16pp on P1 but breaks SNR
on DLA1/DLA2/DLA7 (because the macro replaces ALL hot STEP3 + TAIL sites at once).
This explores whether SOME subset of those sites is safe to drop on the broken
shapes — finer granularity may give a smaller but real win.

For each of DLA1/DLA2/DLA7 we build:
  - parent (sanity rebuild)
  - 4 single-live-site flips: S2, S3, S4 (STEP3 templates), T1 (STEP12)
  - 2 vmcnt-sweep:           relax_vmcnt(0)  — strictest (no in-flight VMEM allowed past)
                             relax_vmcnt15  — most relaxed
  - 1 reference: r18a_p3_all (R18A's full ALL — verifies broken-on-DLA1 reproduces)

Sub-macro defaults preserve R18A behavior: each per-site macro defaults to its
aggregate (STEP3 or STEP12), so existing builds are bit-identical.
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
    ("DLA1", 4096,   32768, 128256, "_ts_pf6_6_v12_memc",
     "-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("DLA2", 128256, 32768,   4096, "_ts_gm2_v12_memc_dc",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    ("DLA7", 28672,  32768,   4096, "_ts_lgk2_v12_memc",
     "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
]

# Per-site flips (single live sites only)
SINGLE_SITE_VARIANTS = [
    ("_r19b_s2",  "-DBARRIER_TO_WAITCNT_STEP3_S2=1"),
    ("_r19b_s3",  "-DBARRIER_TO_WAITCNT_STEP3_S3=1"),
    ("_r19b_s4",  "-DBARRIER_TO_WAITCNT_STEP3_S4=1"),
    ("_r19b_t1",  "-DBARRIER_TO_WAITCNT_STEP12_S1=1"),
    # 2-site combos for next bisect step (cheap to add upfront)
    ("_r19b_s23",  "-DBARRIER_TO_WAITCNT_STEP3_S2=1 -DBARRIER_TO_WAITCNT_STEP3_S3=1"),
    ("_r19b_s24",  "-DBARRIER_TO_WAITCNT_STEP3_S2=1 -DBARRIER_TO_WAITCNT_STEP3_S4=1"),
    ("_r19b_s34",  "-DBARRIER_TO_WAITCNT_STEP3_S3=1 -DBARRIER_TO_WAITCNT_STEP3_S4=1"),
    ("_r19b_s234", "-DBARRIER_TO_WAITCNT_STEP3_S2=1 -DBARRIER_TO_WAITCNT_STEP3_S3=1 -DBARRIER_TO_WAITCNT_STEP3_S4=1"),
    # vmcnt sweep — keep barriers in place, just retune vmcnt
    ("_r19b_vmcnt0",  "-DBARRIER_TO_WAITCNT_RELAXED_VMCNT=1"),  # use 1 (cannot use 0; 0 would still replace, and 0 means "no in-flight" — strictest possible)
    ("_r19b_vmcnt4",  "-DBARRIER_TO_WAITCNT_RELAXED_VMCNT=4"),
    ("_r19b_vmcnt15", "-DBARRIER_TO_WAITCNT_RELAXED_VMCNT=15"),
    # ref — should reproduce R18A broken-on-DLA1 result
    ("_r19b_ref_all", "-DBARRIER_TO_WAITCNT_ALL=1"),
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
        return (full_suffix, "FAIL", dt, r.stderr[-800:])
    return (full_suffix, "OK", dt, "")


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    force = "--force" in sys.argv
    jobs = []
    # Always rebuild parents (force) to confirm regression-free
    for (lab, M, N, K, ps, pflags) in SHAPES:
        jobs.append((lab, N, K, ps + "_r19b_parent_rebuild", pflags, ""))
        for (vs, ef) in SINGLE_SITE_VARIANTS:
            full = ps + vs
            jobs.append((lab, N, K, full, pflags, ef))

    print(f"R19B builds: {len(jobs)} ({len(SHAPES)} shapes x {len(SINGLE_SITE_VARIANTS)+1} variants)")
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
            print(f"  {lab:5s} {full_suffix:60s} {status:8s} ({dt:.1f}s)", flush=True)
            if status == "FAIL":
                fail_lines.append(f"--- {lab} {full_suffix} ---\n{err}\n")
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    if fail_lines:
        print("\nFAILURES:\n" + "\n".join(fail_lines))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
