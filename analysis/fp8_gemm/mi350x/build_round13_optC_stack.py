#!/usr/bin/env python3
"""Round 13 OptC: stack additional flags on top of the 5 verified iterative-ilp
WIN baselines from R10/R11.

ASM-diff verified up-front (asm_round13_optC/FINDINGS.md):
  - last `-mllvm -amdgpu-sched-strategy=` wins; cannot stack memc+iterilp
  - amdgpu-mfma-padding-ratio is a no-op on top of iterilp (drop)
  - iterminreg, iter-max-occupancy-experimental, brlgk, VMCNT all give real ASM diffs

For each of the 5 R10/R11 WIN shapes, build 4 candidates:
  _r13c_iterminreg     -- replace iterilp w/ iterminreg
  _r13c_itermaxocc     -- replace iterilp w/ iterative-max-occupancy-experimental
  _r13c_iterilp_brlgk4 -- iterilp + STEP12_BR_LGKMCNT=4
  _r13c_iterilp_v24    -- iterilp + STEP3_BARRIER_VMCNT=24 (override parent if any)
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

# (label, M, N, K, parent_suffix, parent_base_flags)
# parent_base_flags = the parent's flags WITHOUT any sched-strategy
# (sched-strategy is added per-variant below).
SHAPES = [
    # 14336x4096x32768  R10 WIN: _v16_wpe2_r10a_iterilp
    ("S1_14336x4096x32768", 14336, 4096, 32768,
     "_v16_wpe2",
     "-DSTEP3_BARRIER_VMCNT=16 -DWAVES_PER_EU_2=1"),
    # 16384x4096x28672  R10 WIN: _u8_r10a_iterilp
    ("S2_16384x4096x28672", 16384, 4096, 28672,
     "_u8",
     "-DUNROLL_K=8"),
    # 4096x32768x28672  R11 WIN: _v20_memc_r11_iterilp
    #   parent had max-memory-clause but R11 build appended iterilp last → overridden
    #   so true effective parent flags = -DSTEP3_BARRIER_VMCNT=20
    ("S3_4096x32768x28672", 4096, 32768, 28672,
     "_v20_memc",
     "-DSTEP3_BARRIER_VMCNT=20"),
    # 4096x28672x32768  R11 WIN: _u16_r11_iterilp
    ("S4_4096x28672x32768", 4096, 28672, 32768,
     "_u16",
     "-DUNROLL_K=16"),
    # 4096x32768x14336  R11 WIN: _ts_lgk2_memc_r11_iterilp
    #   memc overridden; effective: -DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2
    ("S5_4096x32768x14336", 4096, 32768, 14336,
     "_ts_lgk2_memc",
     "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2"),
]

# (suffix, extra flags appended after parent_base_flags)
STACK_VARIANTS = [
    ("_r13c_iterminreg",
     "-mllvm -amdgpu-sched-strategy=iterative-minreg"),
    ("_r13c_itermaxocc",
     "-mllvm -amdgpu-sched-strategy=iterative-max-occupancy-experimental"),
    ("_r13c_iterilp_brlgk4",
     "-DSTEP12_BR_LGKMCNT=4 -mllvm -amdgpu-sched-strategy=iterative-ilp"),
    ("_r13c_iterilp_v24",
     "-DSTEP3_BARRIER_VMCNT=24 -mllvm -amdgpu-sched-strategy=iterative-ilp"),
]

BASE = (
    f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    f"-I/opt/rocm/include/hip "
    f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    f"-shared -fPIC -std=c++20 -w "
    f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm"
)


def build_one(label, N, K, parent_base_flags, parent_suffix, suffix, extra):
    full_suffix = parent_suffix + suffix
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
        f"-DK_DIM={K} -DN_DIM={N} {parent_base_flags} {extra} -o {so_path}"
    )
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(so_path):
        return (label, full_suffix, "FAIL", dt, r.stderr[-400:])
    return (label, full_suffix, "OK", dt, "")


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    tasks = []
    for (label, M, N, K, parent_suf, parent_base) in SHAPES:
        for (suffix, extra) in STACK_VARIANTS:
            tasks.append((label, N, K, parent_base, parent_suf, suffix, extra))
    workers = int(os.environ.get("BUILD_WORKERS", "10"))
    print(f"Total builds: {len(tasks)}  (workers={workers})")
    t0 = time.time()
    fail_lines = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_one, *t): t for t in tasks}
        for fut in as_completed(futs):
            label, full_suffix, status, dt, err = fut.result()
            print(f"  {label:24s} {full_suffix:42s} {status:8s} ({dt:.1f}s)", flush=True)
            if status == "FAIL":
                fail_lines.append(f"--- {label} {full_suffix} ---\n{err}\n")
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    if fail_lines:
        print("\nFAILURES:\n" + "\n".join(fail_lines))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
