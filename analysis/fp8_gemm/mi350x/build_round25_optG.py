#!/usr/bin/env python3
"""Round 25 Optimizer G — extend R25-F (gm7 + first-2-iters-only prefetch) from
K=4096 to MID-GAP LARGER-K LOSE shapes.

R25-F won on K=4096 (k_byte_iters=16) with R25C_TAIL_PF_OFF_ITERS=14 — i.e.
ONLY the first 2 K-iters issue global prefetches. R25-G tests whether the
analogous "only first 2 prefetch" point wins on the larger K shapes that
R25-F's pfoff-grid (max 16) could not address:
    K=14336 → k_iters=56  → pfoff ∈ {48, 52, 54, 55}
    K=16384 → k_iters=64  → pfoff ∈ {56, 60, 62, 63}
    K=28672 → k_iters=112 → pfoff ∈ {104, 108, 110, 111}
    K=32768 → k_iters=128 → pfoff ∈ {120, 124, 126, 127}

Per-shape parent stack is preserved (each shape's current best variant flags
minus any -DGROUP_SIZE_M=); we then inject -DGROUP_SIZE_M=7 + the pfoff sweep.

Suffix scheme: _r25g_gm7_pfoff{Y}_K{K}_M{M}_N{N}  (fully shape-disambiguated
because K=32768 is shared between SA and SC, etc.)
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

MEMC = "-mllvm -amdgpu-sched-strategy=max-memory-clause"
DC   = "-mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"

# Per-shape: (label, M, N, K, parent_flags_no_gm, current_best_variant_name).
# parent_flags_no_gm is the variant's full -D and -mllvm flags MINUS any
# -DGROUP_SIZE_M= (we inject gm7).
# Reference best ratios from bench_all42_results_r22.json.
SHAPES = [
    # SA: 4096x28672x32768 — best=ts_v12_tv0_memc_btw_all (96.2%)
    ("SA", 4096, 28672, 32768,
     f"-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=12 -DTAIL_BARRIER_VMCNT=0 {MEMC} -DBARRIER_TO_WAITCNT_ALL=1",
     "ts_v12_tv0_memc_btw_all"),
    # SB: 4096x32768x14336 — best=ts_v12_tv0_memc_btw_all (96.8%)
    ("SB", 4096, 32768, 14336,
     f"-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=12 -DTAIL_BARRIER_VMCNT=0 {MEMC} -DBARRIER_TO_WAITCNT_ALL=1",
     "ts_v12_tv0_memc_btw_all"),
    # SC: 14336x4096x32768 — best=ts_lgk2_memc_btw_all (96.4%)
    ("SC", 14336, 4096, 32768,
     f"-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 {MEMC} -DBARRIER_TO_WAITCNT_ALL=1",
     "ts_lgk2_memc_btw_all"),
    # SD: 16384x4096x28672 — best=ts_lgk2_memc_btw_all (95.0%)
    ("SD", 16384, 4096, 28672,
     f"-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 {MEMC} -DBARRIER_TO_WAITCNT_ALL=1",
     "ts_lgk2_memc_btw_all"),
    # SE: 28672x4096x16384 — best=ts_lgk2_memc_btw_all (96.8%)
    ("SE", 28672, 4096, 16384,
     f"-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 {MEMC} -DBARRIER_TO_WAITCNT_ALL=1",
     "ts_lgk2_memc_btw_all"),
    # SF: 4096x14336x16384 — best=ts_lgk2_memc_btw_all (99.0%)
    ("SF", 4096, 14336, 16384,
     f"-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 {MEMC} -DBARRIER_TO_WAITCNT_ALL=1",
     "ts_lgk2_memc_btw_all"),
]

# pfoff sweep around (K_iters - 2) per K
PFOFF_BY_K = {
    14336: [48, 52, 54, 55],
    16384: [56, 60, 62, 63],
    28672: [104, 108, 110, 111],
    32768: [120, 124, 126, 127],
}

GM = 7  # R25-F winner

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
    for (lab, M, N, K, parent, parent_name) in SHAPES:
        for pfoff in PFOFF_BY_K[K]:
            gm_flag = f"-DGROUP_SIZE_M={GM}"
            pf_flag = f"-DR25C_TAIL_PF_OFF_ITERS={pfoff} -DR25C_K_LIMIT=32768"
            extra = f"{gm_flag} {pf_flag}"
            full = f"_r25g_gm{GM}_pfoff{pfoff}_K{K}_M{M}_N{N}"
            jobs.append((lab, M, N, K, full, parent, extra))

    # Also build per-shape REF (parent + gm7 only, pfoff=0/disabled) so we can
    # check whether gm7 alone explains any delta vs the parent-without-gm7. The
    # actual production REF is the parent variant (already in cache). We add
    # explicit gm7+pfoff0 just for diagnostics.
    for (lab, M, N, K, parent, parent_name) in SHAPES:
        gm_flag = f"-DGROUP_SIZE_M={GM}"
        # pfoff=0 means R25C disabled
        full = f"_r25g_gm{GM}_pfoff0_K{K}_M{M}_N{N}"
        extra = f"{gm_flag} -DR25C_TAIL_PF_OFF_ITERS=0 -DR25C_K_LIMIT=32768"
        jobs.append((lab, M, N, K, full, parent, extra))

    print(f"Round 25 OptG builds: {len(jobs)}")
    print("=" * 110)
    workers = int(os.environ.get("BUILD_WORKERS", "8"))
    t0 = time.time()
    fail_lines = []
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_one, N, K, full, pflags, ef, force): (lab, full)
                for (lab, M, N, K, full, pflags, ef) in jobs}
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
