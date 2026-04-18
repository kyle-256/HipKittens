#!/usr/bin/env python3
"""Round 25 Optimizer D — STACK TEST: gm6 (R25-B) x R25C_TAIL_PF_OFF_ITERS=4 (R25-C).

Question: do GROUP_SIZE_M=6 and R25C_TAIL_PF_OFF_ITERS=4 stack on DLA2/DLA7?

Per-shape variants (4 each, 2 shapes = 8 total):
  _r25d_baseline   — gm2 (DLA2) / no-gm (DLA7), R25C off
  _r25d_gm6        — gm6, R25C off
  _r25d_pfoff4     — gm2 (DLA2) / no-gm (DLA7), R25C TAIL_PF_OFF_ITERS=4
  _r25d_gm6_pfoff4 — gm6 + R25C TAIL_PF_OFF_ITERS=4 (the STACK)

Parent flag stacks (from r25b/r25c analyses):
  DLA2: TAIL_SPLIT=1 + STEP3_BARRIER_VMCNT=12 + memc + disable-clustered
  DLA7: TAIL_SPLIT=1 + STEP12_BR_LGKMCNT=2 + STEP3_BARRIER_VMCNT=12 + memc

Notes:
  - R25C macros default: R25C_K_LIMIT=32768; both shapes have K=4096 < limit so R25C is active.
  - DLA2 baseline includes -DGROUP_SIZE_M=2 explicitly (per r25c parent suffix _ts_gm2_v12_memc_dc).
  - DLA7 baseline does NOT pass -DGROUP_SIZE_M (kernel default is 4, per r25c parent _ts_lgk2_v12_memc).
  - For 'gm6' variants we replace/add -DGROUP_SIZE_M=6.
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

MEMC = "-mllvm -amdgpu-sched-strategy=max-memory-clause"

# (label, M_native, N, K, baseline_gm_flag, parent_flags_no_gm)
# parent_flags_no_gm omits any -DGROUP_SIZE_M= so we can inject per-variant.
SHAPES = [
    ("DLA2", 128256, 32768, 4096,
     "-DGROUP_SIZE_M=2",
     f"-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=12 {MEMC} -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    ("DLA7", 28672, 32768, 4096,
     "",  # DLA7 baseline uses kernel-default GROUP_SIZE_M=4
     f"-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 {MEMC}"),
]

# variant_label, gm_flag_override (None=use baseline), pfoff_flag
VARIANTS = [
    ("_r25d_baseline",   None,                 ""),
    ("_r25d_gm6",        "-DGROUP_SIZE_M=6",   ""),
    ("_r25d_pfoff4",     None,                 "-DR25C_TAIL_PF_OFF_ITERS=4 -DR25C_K_LIMIT=32768"),
    ("_r25d_gm6_pfoff4", "-DGROUP_SIZE_M=6",   "-DR25C_TAIL_PF_OFF_ITERS=4 -DR25C_K_LIMIT=32768"),
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
    for (lab, M, N, K, base_gm, parent) in SHAPES:
        for (vs, gm_override, pfoff) in VARIANTS:
            gm = gm_override if gm_override is not None else base_gm
            extra = f"{gm} {pfoff}".strip()
            # NB: DLA2 and DLA7 share (N=32768, K=4096) but have DIFFERENT parent flags.
            # We must include the shape label in the suffix to avoid binary collision.
            full = f"{vs}_{lab.lower()}"
            jobs.append((lab, N, K, full, parent, extra))

    print(f"Round 25 OptD STACK builds: {len(jobs)} ({len(SHAPES)} shapes x {len(VARIANTS)} variants)")
    print("=" * 110)
    workers = int(os.environ.get("BUILD_WORKERS", "4"))
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
            print(f"  {lab:5s} {full_suffix:30s} {status:8s} ({dt:.1f}s)", flush=True)
            if status == "FAIL":
                fail_lines.append(f"--- {lab} {full_suffix} ---\n{err}\n")
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    if fail_lines:
        print("\nFAILURES:\n" + "\n".join(fail_lines))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
