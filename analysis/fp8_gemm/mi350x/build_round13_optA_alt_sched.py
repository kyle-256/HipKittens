#!/usr/bin/env python3
"""Round 13 Optimizer A: Alternative LLVM sched-strategies on the 4 shapes
where un-prefixed iterative-ilp triggered the SGPR-clobber compiler bug.

Targets (4 shapes, parent best variants):
  DLA1  4096x32768x128256  _ts_pf6_6_v12_memc      (88.3%)
  DLA2  128256x32768x4096  _ts_gm2_v12_memc_dc     (92.9%)
  DLA7  28672x32768x4096   _ts_lgk2_v12_memc       (94.5%)
  WIN2  32768x6144x2048    _ts_gm8_v12             (103.9%)

Alternative un-prefixed strategies (verified valid enum names by string-dump
of libLLVMAMDGPUCodeGen.a):
  - iterative-minreg
  - iterative-maxocc
  - max-ilp
(Also tries `max-occupancy` and `iterative-max-occupancy-experimental`
even though they are likely silent-noop, so we can prove via ASM-diff.)

Each (shape, strategy) gets a `_r13_<strat>` variant. Build only — bench
in a separate script. Uses BUILD_DIR = build_all42 to share with other rounds.
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
    ("DLA1_4096x32768x128256",   4096,  32768, 128256, "_ts_pf6_6_v12_memc",
     "-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("DLA2_128256x32768x4096", 128256,  32768,   4096, "_ts_gm2_v12_memc_dc",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    ("DLA7_28672x32768x4096",   28672,  32768,   4096, "_ts_lgk2_v12_memc",
     "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("WIN2_32768x6144x2048",    32768,   6144,   2048, "_ts_gm8_v12",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DSTEP3_BARRIER_VMCNT=12"),
]

# Strategy candidates. We override the sched-strategy by appending after
# parent flags. Note: when parent_flags already contains
# `-amdgpu-sched-strategy=max-memory-clause`, the LLVM cl::opt is the LAST
# value seen. So adding our flag after it overrides cleanly.
STRATEGIES = [
    ("iterminreg",  "iterative-minreg"),
    ("itermaxocc",  "iterative-maxocc"),
    ("maxilp",      "max-ilp"),
    # Probes for silent-noop detection (expected to be no-op):
    ("maxocc",      "max-occupancy"),
    ("itermaxoccx", "iterative-max-occupancy-experimental"),
]

BASE = (
    f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    f"-I/opt/rocm/include/hip "
    f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    f"-shared -fPIC -std=c++20 -w "
    f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm"
)


def build_one(label, N, K, parent_flags, parent_suffix, strat_tag, strat_name):
    full_suffix = parent_suffix + f"_r13_{strat_tag}"
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
    sched_flag = f"-mllvm -amdgpu-sched-strategy={strat_name}"
    cmd = (
        f"/opt/rocm/bin/hipcc {wrapper_src} {BASE} "
        f"-DK_DIM={K} -DN_DIM={N} {parent_flags} {sched_flag} -o {so_path}"
    )
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900)
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(so_path):
        return (label, full_suffix, "FAIL", dt, r.stderr[-400:])
    return (label, full_suffix, "OK", dt, "")


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    workers = int(os.environ.get("BUILD_WORKERS", "8"))
    builds = []
    for (lab, M, N, K, ps, pf) in SHAPES:
        for (tag, sname) in STRATEGIES:
            builds.append((lab, M, N, K, ps, pf, tag, sname))
    print(f"Total builds: {len(builds)}  (workers={workers})")
    t0 = time.time()
    fail_lines = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_one, lab, N, K, pf, ps, tag, sname):
                f"{lab}/{tag}"
                for (lab, M, N, K, ps, pf, tag, sname) in builds}
        for fut in as_completed(futs):
            label, full_suffix, status, dt, err = fut.result()
            print(f"  {label:32s} {full_suffix:48s} {status:8s} ({dt:.1f}s)", flush=True)
            if status == "FAIL":
                fail_lines.append(f"--- {label} {full_suffix} ---\n{err}\n")
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    if fail_lines:
        print("\nFAILURES:\n" + "\n".join(fail_lines))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
