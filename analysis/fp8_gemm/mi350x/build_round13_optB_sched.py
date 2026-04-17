#!/usr/bin/env python3
"""Round 13 Optimizer B: Sched-strategy sweep on 28672x4096x16384.

Target: P1 deep-LOSE shape, current best _ts_gm8 at 93.7% aiter ratio.
R10A already tested 3 strategies on the _ts_gm8 parent; iterilp single-shot
showed +0.45pp but failed 5-run verify by a hair (mean +0.81 TFLOPS only).

R13B expands the search:
  - 5 sched-strategies (incl. max-occupancy variants not in R10A)
  - 2 parents: _ts_gm8 (original) + _ts_lgk2_v12_memc (alternative lineage)
  - 1 stacking experiment (iterilp + memc, last-flag-wins check)

All variants have suffix ending with _r13b_*.
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

SHAPE_M, SHAPE_N, SHAPE_K = 28672, 4096, 16384

# (parent_suffix, parent_flags) — two parent lineages
PARENTS = [
    ("_ts_gm8",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8"),
    ("_ts_lgk2_v12_memc",
     "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 "
     "-mllvm -amdgpu-sched-strategy=max-memory-clause"),
]

# (suffix, extra_flag) — five sched-strategies + a stacking probe
SCHED_VARIANTS = [
    ("_r13b_baseline",     ""),  # pristine rebuild for sanity
    ("_r13b_iterilp",      "-mllvm -amdgpu-sched-strategy=iterative-ilp"),
    ("_r13b_maxilp",       "-mllvm -amdgpu-sched-strategy=max-ilp"),
    ("_r13b_iterminreg",   "-mllvm -amdgpu-sched-strategy=iterative-minreg"),
    ("_r13b_maxocc",       "-mllvm -amdgpu-sched-strategy=max-occupancy"),
    ("_r13b_iteroccexp",   "-mllvm -amdgpu-sched-strategy=iterative-max-occupancy-experimental"),
]

# Extra: stacking experiments (last -amdgpu-sched-strategy= wins per LLVM cl::opt; but
# both are present so the kernel ASM may differ from a single-flag version due to
# other indirect side-effects).  We add these only on _ts_gm8 parent.
STACK_VARIANTS = [
    ("_r13b_iterilp_stack_memc",
     "-mllvm -amdgpu-sched-strategy=iterative-ilp "
     "-mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("_r13b_memc_stack_iterilp",
     "-mllvm -amdgpu-sched-strategy=max-memory-clause "
     "-mllvm -amdgpu-sched-strategy=iterative-ilp"),
]

BASE = (
    f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    f"-I/opt/rocm/include/hip "
    f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    f"-shared -fPIC -std=c++20 -w "
    f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm"
)


def build_one(N, K, parent_suffix, parent_flags, suffix, extra):
    full_suffix = parent_suffix + suffix
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if os.path.exists(so_path):
        return (parent_suffix, suffix, full_suffix, "cached", 0.0, "")
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
        f"-DK_DIM={K} -DN_DIM={N} {parent_flags} {extra} -o {so_path}"
    )
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(so_path):
        return (parent_suffix, suffix, full_suffix, "FAIL", dt, r.stderr[-400:])
    return (parent_suffix, suffix, full_suffix, "OK", dt, "")


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    tasks = []
    for (psuf, pflags) in PARENTS:
        for (suf, extra) in SCHED_VARIANTS:
            tasks.append((SHAPE_N, SHAPE_K, psuf, pflags, suf, extra))
    # Stacking only on _ts_gm8
    for (suf, extra) in STACK_VARIANTS:
        tasks.append((SHAPE_N, SHAPE_K, "_ts_gm8", PARENTS[0][1], suf, extra))

    workers = int(os.environ.get("BUILD_WORKERS", "8"))
    print(f"R13 OptB build: {len(tasks)} variants on {SHAPE_M}x{SHAPE_N}x{SHAPE_K}  (workers={workers})")
    print("=" * 110)
    t0 = time.time()
    fail_lines = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_one, *t): t for t in tasks}
        for fut in as_completed(futs):
            psuf, suf, full, status, dt, err = fut.result()
            print(f"  parent={psuf:24s}  variant={suf:30s}  {status:8s} ({dt:.1f}s)", flush=True)
            if status == "FAIL":
                fail_lines.append(f"--- {full} ---\n{err}\n")
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    if fail_lines:
        print("\nFAILURES:\n" + "\n".join(fail_lines))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
