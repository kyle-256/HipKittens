#!/usr/bin/env python3
"""Round 24 Optimizer C — outer-K pull-forward L2 prefetch.

Hypothesis: Existing prefetch loads K+2 into LDS while iter K+0 runs MFMAs.
On HBM-bandwidth-bound DLA shapes, K+2 lands just-in-time. By issuing extra
buffer_load_dwordx4 for K+3 (discarded result, lands in L2/L1) at iter K+0,
we extend in-flight load count and hide more HBM latency. Cost: extra VMEM
pressure but no LDS budget impact.

Variants (per DLA shape):
  _r24c_baseline (OUTER_K_PF_DEPTH=1)         — fresh build, current behavior
  _r24c_okpf_l2  (OUTER_K_PF_DEPTH=2, MODE=1) — L2-only pull-forward depth 2
  _r24c_okpf_l3  (OUTER_K_PF_DEPTH=3, MODE=1) — L2-only pull-forward depth 3 (probe)

LDS-doublebuffer mode skipped: requires 3rd LDS slot, blows current 80KB+
double-buffered allocation past 160KB cap.
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

MEMC = "-mllvm -amdgpu-sched-strategy=max-memory-clause"

# (label, M_native, N, K, parent_suffix, parent_flags)
SHAPES = [
    ("DLA1", 4096,  32768, 128256, "_ts_pf6_6_v12_memc",
     f"-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DSTEP3_BARRIER_VMCNT=12 {MEMC}"),
    ("DLA2", 128256, 32768,   4096, "_ts_gm2_v12_memc_dc",
     f"-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP3_BARRIER_VMCNT=12 {MEMC} -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    ("DLA7", 28672,  32768,   4096, "_ts_lgk2_v12_memc",
     f"-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 {MEMC}"),
]

VARIANTS = [
    ("_r24c_baseline",    "-DOUTER_K_PF_DEPTH=1 -DOUTER_K_PF_MODE=0"),
    ("_r24c_okpf_l2",     "-DOUTER_K_PF_DEPTH=2 -DOUTER_K_PF_MODE=1"),
    ("_r24c_okpf_l3",     "-DOUTER_K_PF_DEPTH=3 -DOUTER_K_PF_MODE=1"),
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
        return (full_suffix, "FAIL", dt, r.stderr[-1000:])
    return (full_suffix, "OK", dt, "")


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    force = "--force" in sys.argv
    jobs = []
    for (lab, M, N, K, ps, pflags) in SHAPES:
        for (vs, ef) in VARIANTS:
            full = ps + vs
            jobs.append((lab, N, K, full, pflags, ef))

    print(f"Round 24 OptC builds: {len(jobs)} ({len(SHAPES)} shapes x {len(VARIANTS)} variants)")
    print("=" * 100)
    workers = int(os.environ.get("BUILD_WORKERS", "6"))
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
