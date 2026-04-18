#!/usr/bin/env python3
"""Round 25 Optimizer B — GROUP_SIZE_M fine-grained sweep {3, 6, 12, 16}.

Hypothesis: GROUP_SIZE_M values {3, 6, 12} are UNTESTED (prior coverage:
{1, 2, 4, 8, 16, 32, 64}). gm in the gap {3, 6, 12} sits between "too
small for B-tile reuse" (gm<=2) and "too big for L2 working-set" (gm>=32).
Larger GROUP_SIZE_M means more consecutive M-blocks share each B-tile
L2-fetch, attacking the HBM REUSE axis (orthogonal to closed POLICY/BANDWIDTH).

For each deep-LOSE shape, sweep gm in {3, 6, 12, 16} stacked on the shape's
current pre-btw parent. Plus a baseline rebuild for fair comparison.

Shapes (current 13 LOSE per R22; priority on deep-LOSE):
  DLA1 = 4096   x 32768  x 128256  (gap -8.1pp, R22 best ts_lgk2_btw_step3)
  DLA2 = 128256 x 32768  x 4096    (gap -3.7pp)
  DLA7 = 28672  x 32768  x 4096    (gap -3.1pp)
  MID1 = 14336  x 4096   x 32768   (gap -4.3pp, mid-gap LOSE)
  MID2 = 4096   x 28672  x 32768   (gap -4.0pp)

Each shape gets 5 builds: baseline + gm{3,6,12,16}.
Total: 5 shapes x 5 = 25 .so files.
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

MEMC = "-mllvm -amdgpu-sched-strategy=max-memory-clause"

# Per-shape parent flags (the macro stack, MINUS any GROUP_SIZE_M).
# Parents are selected to match each shape's R22 best non-btw variant family,
# so we isolate the gm axis. (btw is correctness-risky; we skip btw here.)
#
# (label, M, N, K, parent_suffix, parent_flags_no_gm)
SHAPES = [
    # DLA1: best_btw=ts_lgk2_btw_step3; non-btw family = ts_lgk2 / ts_pf6_6_v12_memc.
    # Use the R24B-DLA1 parent (ts_pf6_6_v12_memc) as base.
    ("DLA1", 4096, 32768, 128256, "_ts_pf6_6_v12_memc",
     f"-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DSTEP3_BARRIER_VMCNT=12 {MEMC}"),
    # DLA2: best_btw=ts_v12_tv0_memc_btw_all; non-btw = ts_gm2_v12_memc_dc.
    # Strip the gm2 -> base = ts_v12_memc_dc.
    ("DLA2", 128256, 32768, 4096, "_ts_v12_memc_dc",
     f"-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=12 {MEMC} -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    # DLA7: 28672x32768x4096; best_btw=ts_lgk2_v12_memc_btw_all.
    # Use ts_lgk2_v12_memc as parent (matches R24B-DLA7).
    ("DLA7", 28672, 32768, 4096, "_ts_lgk2_v12_memc",
     f"-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 {MEMC}"),
    # MID1: 14336x4096x32768; R22 top gm = ts_gm2_v12_memc_dc_btw_all (4973).
    # Parent = ts_v12_memc_dc.
    ("MID1", 14336, 4096, 32768, "_ts_v12_memc_dc",
     f"-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=12 {MEMC} -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    # MID2: 4096x28672x32768; R22 top gm = ts_gm8_v12_btw_step3 (5388).
    # Parent = ts_v12.
    ("MID2", 4096, 28672, 32768, "_ts_v12",
     f"-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=12"),
]

# Variants: baseline (parent only, no gm override = kernel default GROUP_SIZE_M=4)
# plus gm in the UNTESTED gap {3, 6, 12, 16}. gm16 is included because some
# parents in this set never had it tested.
VARIANTS = [
    ("_r25b_baseline", ""),
    ("_r25b_gm3",  "-DGROUP_SIZE_M=3"),
    ("_r25b_gm6",  "-DGROUP_SIZE_M=6"),
    ("_r25b_gm12", "-DGROUP_SIZE_M=12"),
    ("_r25b_gm16", "-DGROUP_SIZE_M=16"),
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
        return (full_suffix, "FAIL", dt, r.stderr[-1500:])
    return (full_suffix, "OK", dt, "")


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    force = "--force" in sys.argv
    jobs = []
    for (lab, M, N, K, ps, pflags) in SHAPES:
        for (vs, ef) in VARIANTS:
            full = ps + vs
            jobs.append((lab, N, K, full, pflags, ef))

    print(f"Round 25 OptB builds: {len(jobs)} ({len(SHAPES)} shapes x {len(VARIANTS)} variants)")
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
            print(f"  {lab:5s} {full_suffix:65s} {status:8s} ({dt:.1f}s)", flush=True)
            if status == "FAIL":
                fail_lines.append(f"--- {lab} {full_suffix} ---\n{err}\n")
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    if fail_lines:
        print("\nFAILURES:\n" + "\n".join(fail_lines))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
