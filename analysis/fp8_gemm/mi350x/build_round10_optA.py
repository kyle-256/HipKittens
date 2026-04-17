#!/usr/bin/env python3
"""Round 10 Optimizer A: Test the THREE un-prefixed sched-strategy values
(max-ilp, iterative-ilp, iterative-minreg) on the 3 deep-LOSE shapes.

ASM-verification (Round 10) confirmed all 3 produce DISTINCT ASM vs default
(unlike the gcn-* prefixed variants used in R6 OptA / R8 OptA, which were no-ops).

Targets (each on top of its current-best flag combination):
- 14336x4096x32768  best=_v16_wpe2     base="-DSTEP3_BARRIER_VMCNT=16 -DWAVES_PER_EU_2=1"
- 16384x4096x28672  best=_u8           base="-DUNROLL_K=8"
- 28672x4096x16384  best=_ts_gm8       base="-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8"
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

# Per-shape configurations: (label, M, N, K, base_flags, parent_suffix)
SHAPES = [
    ("S1_14336x4096x32768", 14336, 4096, 32768,
     "-DSTEP3_BARRIER_VMCNT=16 -DWAVES_PER_EU_2=1",
     "_v16_wpe2"),
    ("S2_16384x4096x28672", 16384, 4096, 28672,
     "-DUNROLL_K=8",
     "_u8"),
    ("S3_28672x4096x16384", 28672, 4096, 16384,
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8",
     "_ts_gm8"),
]

# Round 10 variants: 3 sched-strategies + 1 fresh baseline rebuild per shape
SCHED_VARIANTS = [
    ("_r10a_baseline",  ""),  # fresh-build sanity baseline
    ("_r10a_maxilp",    "-mllvm -amdgpu-sched-strategy=max-ilp"),
    ("_r10a_iterilp",   "-mllvm -amdgpu-sched-strategy=iterative-ilp"),
    ("_r10a_iterminreg","-mllvm -amdgpu-sched-strategy=iterative-minreg"),
]

BASE = (
    f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    f"-I/opt/rocm/include/hip "
    f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    f"-shared -fPIC -std=c++20 -w "
    f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm"
)


def build_one(label, N, K, base_flags, parent, suffix, extra):
    """Build kernel for shape (N,K) with parent's base_flags + extra (sched flag).
    Variant suffix is parent + new round-10 suffix to encode lineage.
    """
    # full suffix: parent+round10 marker, e.g. _v16_wpe2_r10a_maxilp
    full_suffix = parent + suffix
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
        f"-DK_DIM={K} -DN_DIM={N} {base_flags} {extra} -o {so_path}"
    )
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=400)
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(so_path):
        return (label, full_suffix, "FAIL", dt, r.stderr[-400:])
    return (label, full_suffix, "OK", dt, "")


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    tasks = []
    for (label, M, N, K, base_flags, parent) in SHAPES:
        for (suffix, extra) in SCHED_VARIANTS:
            tasks.append((label, N, K, base_flags, parent, suffix, extra))
    workers = int(os.environ.get("BUILD_WORKERS", "8"))
    print(f"Total builds: {len(tasks)}  (workers={workers})")
    t0 = time.time()
    fail_lines = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_one, *t): t[5] for t in tasks}
        for fut in as_completed(futs):
            label, full_suffix, status, dt, err = fut.result()
            print(f"  {label:24s} {full_suffix:34s} {status:8s} ({dt:.1f}s)", flush=True)
            if status == "FAIL":
                fail_lines.append(f"--- {label} {full_suffix} ---\n{err}\n")
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    if fail_lines:
        print("\nFAILURES:\n" + "\n".join(fail_lines))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
