#!/usr/bin/env python3
"""Round 12 Optimizer A: BISECT iterative-ilp.

For each of the 5 failing shapes, build a "control" variant that
applies the SAME parent flags as Round 11 but WITHOUT the
`-mllvm -amdgpu-sched-strategy=iterative-ilp` flag. If the control
runs successfully, the iterative-ilp flag is the trigger.

Note: the parent variants ALREADY exist in build_all42 with their
parent flags. The ONLY difference between parent and r11_iterilp is
the extra SCHED_FLAG. So bisect = re-bench the parent (which we do).

But we ALSO build a "_r12_noilp" variant which is parent_flags + a
NO-OP marker (e.g. extra `-DROUND12_NOILP=1`) so we know the build
goes through a fresh hipcc invocation, identical to the failing case
EXCEPT for the SCHED_FLAG. This rules out:
  - cached binary effects
  - compiler nondeterminism

Then a *second* control: parent_flags + `-mllvm
-amdgpu-sched-strategy=iterative-ilp` only (i.e. exactly what Round
11 did) -> rebuild fresh to ensure not a stale .so. We delete the
existing .so and rebuild.
"""
import os, sys, sysconfig, subprocess, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

# Same 5 failing shapes (label, M, N, K, parent_suffix, parent_flags)
SHAPES = [
    ("DLA1_4096x32768x128256",   4096,  32768, 128256, "_ts_pf6_6_v12_memc",
     "-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("DLA2_128256x32768x4096", 128256,  32768,   4096, "_ts_gm2_v12_memc_dc",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    ("DLA7_28672x32768x4096",   28672,  32768,   4096, "_ts_lgk2_v12_memc",
     "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("WIN1_16384x4096x7168",    16384,   4096,   7168, "_ts_lgk2_v20_memc",
     "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=20 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("WIN2_32768x6144x2048",    32768,   6144,   2048, "_ts_gm8_v12",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DSTEP3_BARRIER_VMCNT=12"),
]

BASE = (
    f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    f"-I/opt/rocm/include/hip "
    f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    f"-shared -fPIC -std=c++20 -w "
    f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm"
)


def build_one(label, N, K, parent_flags, parent_suffix, new_suffix, extra_flags, force=True):
    full_suffix = parent_suffix + new_suffix
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if force and os.path.exists(so_path):
        os.remove(so_path)
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
        f"-DK_DIM={K} -DN_DIM={N} {parent_flags} {extra_flags} -o {so_path}"
    )
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(so_path):
        return (label, full_suffix, "FAIL", dt, r.stderr[-400:])
    return (label, full_suffix, "OK", dt, "")


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    print("Round 12 OptA bisect builds")
    print("=" * 80)

    # Variant A: NOILP control (parent_flags only, fresh build)
    # Variant B: IPL rebuild (parent_flags + iterative-ilp, fresh build)
    work = []
    for (lab, M, N, K, ps, pf) in SHAPES:
        work.append((lab, N, K, pf, ps, "_r12_noilp", "-DROUND12_NOILP=1"))
        work.append((lab, N, K, pf, ps, "_r12_ilp_rebuild", "-mllvm -amdgpu-sched-strategy=iterative-ilp -DROUND12_ILP_REBUILD=1"))

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(build_one, lab, N, K, pf, ps, ns, ef, True): (lab, ns)
                for (lab, N, K, pf, ps, ns, ef) in work}
        for fut in as_completed(futs):
            label, full_suffix, status, dt, err = fut.result()
            print(f"  {label:32s} {full_suffix:44s} {status:8s} ({dt:.1f}s)", flush=True)
            if status == "FAIL":
                print(f"    err: {err[:300]}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
