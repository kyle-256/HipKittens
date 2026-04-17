#!/usr/bin/env python3
"""Round 16 OptA: compound stack iterilp + regclassglob on R10/R11 winners.

R14A discovered `-mllvm -greedy-regclass-priority-trumps-globalness=true`
adds +0.236pp on P1 (sub-threshold). regclassglob is a regalloc flag and
does NOT collide with the `-amdgpu-sched-strategy=iterative-ilp` scheduler
flag (different LLVM option families).

This round builds 5 NEW variants (one per R10/R11 winning shape) that
combine the parent macros + iterilp + regclassglob.
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

ITERILP = "-mllvm -amdgpu-sched-strategy=iterative-ilp"
RCG = "-mllvm -greedy-regclass-priority-trumps-globalness=true"

# (tag, M, N, K, suffix, parent_macros)
# suffix encodes new compound variant naming
JOBS = [
    ("S1", 14336,  4096, 32768,
     "_lgk2_dc_r16a_iterilp_regclassglob",
     "-DSTEP12_BR_LGKMCNT=2 -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    ("S2", 16384,  4096, 28672,
     "_u32_r16a_iterilp_regclassglob",
     "-DUNROLL_K=32"),
    ("S3",  4096, 32768, 28672,
     "_v20_memc_r16a_iterilp_regclassglob",
     "-DSTEP3_BARRIER_VMCNT=20 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("S4",  4096, 28672, 32768,
     "_u16_r16a_iterilp_regclassglob",
     "-DUNROLL_K=16"),
    ("S5",  4096, 32768, 14336,
     "_ts_lgk2_memc_r16a_iterilp_regclassglob",
     "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
]

BASE = (
    f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    f"-I/opt/rocm/include/hip "
    f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    f"-shared -fPIC -std=c++20 -w "
    f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm"
)


def build_one(tag, M, N, K, suffix, parent_macros):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if os.path.exists(so_path):
        return (tag, suffix, "cached", 0.0, "")
    with open(KERNEL_SRC, "r") as f:
        src = f.read()
    patched = src.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},"
    )
    wrapper_src = os.path.join(BUILD_DIR, f"wrap_n{N}_k{K}{suffix}.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)
    # IMPORTANT: regclassglob first, iterilp LAST so it wins LAST-SPEC-WINS
    # for the sched-strategy family (parent may already contain a sched flag).
    flags = f"{parent_macros} {RCG} {ITERILP}"
    cmd = (
        f"/opt/rocm/bin/hipcc {wrapper_src} {BASE} "
        f"-DK_DIM={K} -DN_DIM={N} {flags} -o {so_path}"
    )
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900)
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(so_path):
        return (tag, suffix, "FAIL", dt, r.stderr[-600:])
    return (tag, suffix, "OK", dt, "")


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    workers = int(os.environ.get("BUILD_WORKERS", "5"))
    print(f"Total builds: {len(JOBS)}  (workers={workers})")
    print(f"Compound: parent_macros + regclassglob + iterilp (last)")
    t0 = time.time()
    fail_lines = []
    n_ok = n_cached = n_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_one, *j): j[0] for j in JOBS}
        for fut in as_completed(futs):
            tag, suffix, status, dt, err = fut.result()
            print(f"  {tag:3s} {suffix:60s} {status:8s} ({dt:.1f}s)", flush=True)
            if status == "FAIL":
                n_fail += 1
                fail_lines.append(f"--- {tag} {suffix} ---\n{err}\n")
            elif status == "cached":
                n_cached += 1
            else:
                n_ok += 1
    print(f"\nElapsed: {time.time()-t0:.1f}s   ok={n_ok} cached={n_cached} fail={n_fail}")
    if fail_lines:
        print("\nFAILURES:\n" + "\n".join(fail_lines))
    return 0 if not fail_lines else 1


if __name__ == "__main__":
    sys.exit(main())
