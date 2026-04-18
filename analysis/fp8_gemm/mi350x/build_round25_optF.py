#!/usr/bin/env python3
"""Round 25 Optimizer F — EXTENDED SWEEP of (gm × pfoff) on top of R25-D winning axis.

R25-D won with gm6 + pfoff4 (super-additive on DLA2 +6.69%, DLA7 +6.92%).
This sweep extends both axes to find a stronger stack:
  - GROUP_SIZE_M ∈ {5, 6, 7, 8}   (6 was best in R25-D)
  - R25C_TAIL_PF_OFF_ITERS ∈ {3,4,5,6,7,8}   (4 was best in R25-D)
  -> 24 builds per shape × 2 shapes (DLA2, DLA7) = 48 builds.

Suffix scheme: _r25f_gm{X}_pfoff{Y}_dla{2|7}  (shape label is required because
DLA2 and DLA7 share (N=32768, K=4096) — without it, the second build cache-hits
the first's parent flags, mixing DLA2/DLA7 binaries.)

R25-D excluded gm12/gm16 because they APERTURE_VIOLATE on small-M shapes; the
gm5/gm7 values are new (untested) but conform to the same parent stack
(STEP3_BARRIER_VMCNT=12, no STEP3_PF_N) so should not crash.
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

MEMC = "-mllvm -amdgpu-sched-strategy=max-memory-clause"

# (label, M_native, N, K, parent_flags_no_gm)
# parent_flags_no_gm omits any -DGROUP_SIZE_M= so we can inject per-variant.
SHAPES = [
    ("DLA2", 128256, 32768, 4096,
     f"-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=12 {MEMC} -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    ("DLA7", 28672, 32768, 4096,
     f"-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 {MEMC}"),
]

GM_VALUES = [5, 6, 7, 8]
PFOFF_VALUES = [3, 4, 5, 6, 7, 8]

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
        return (full_suffix, "FAIL", dt, r.stderr[-1500:])
    return (full_suffix, "OK", dt, "")


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    force = "--force" in sys.argv
    jobs = []
    for (lab, M, N, K, parent) in SHAPES:
        for gm in GM_VALUES:
            for pfoff in PFOFF_VALUES:
                gm_flag = f"-DGROUP_SIZE_M={gm}"
                pf_flag = f"-DR25C_TAIL_PF_OFF_ITERS={pfoff} -DR25C_K_LIMIT=32768"
                extra = f"{gm_flag} {pf_flag}"
                # Suffix encodes both axis values AND shape label for cache-key
                # disambiguation (DLA2 and DLA7 share N=32768,K=4096).
                full = f"_r25f_gm{gm}_pfoff{pfoff}_{lab.lower()}"
                jobs.append((lab, N, K, full, parent, extra))

    print(f"Round 25 OptF EXTENDED-SWEEP builds: {len(jobs)} "
          f"({len(SHAPES)} shapes × {len(GM_VALUES)} gm × {len(PFOFF_VALUES)} pfoff)")
    print("=" * 110)
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
            print(f"  {lab:5s} {full_suffix:40s} {status:8s} ({dt:.1f}s)", flush=True)
            if status == "FAIL":
                fail_lines.append(f"--- {lab} {full_suffix} ---\n{err}\n")
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    if fail_lines:
        print("\nFAILURES:\n" + "\n".join(fail_lines))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
