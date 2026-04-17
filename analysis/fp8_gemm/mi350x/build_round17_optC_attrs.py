#!/usr/bin/env python3
"""Round 17 Optimizer C: untested __attribute__'s on the 4 stuck deep-LOSE shapes.

Per-attribute compile gate (run separately, see attr_compile_test_r17c.log):
  RECOGNIZED: amdgpu_flat_work_group_size, amdgpu_num_vgpr,
              amdgpu_num_sgpr, amdgpu_max_num_work_groups
  REJECTED  : amdgpu_no_agpr (unknown attribute, ignored)

For each (shape, attribute) we patch kernel_mxfp4_gluon_cpp.cpp in-memory by
inserting a single __attribute__(()) line directly above the existing
`__global__ __launch_bounds__(...)` line. We KEEP the parent's flags
(parent_flags) so we are isolating the attribute as the only delta.
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

# Same 4 broken deep-LOSE shapes / parents as R16C.
SHAPES = [
    ("DLA1", 4096,  32768, 128256, "_ts_pf6_6_v12_memc",
     "-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("DLA2", 128256, 32768,  4096, "_ts_gm2_v12_memc_dc",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    ("DLA7", 28672, 32768,  4096, "_ts_lgk2_v12_memc",
     "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("P1",   28672,  4096, 16384, "_ts_gm8",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8"),
]

# Tag -> __attribute__(( ... )) body. (no_agpr was rejected as unknown.)
ATTRS = {
    "fwgs64_256":  "amdgpu_flat_work_group_size(64,256)",
    "fwgs256_256": "amdgpu_flat_work_group_size(256,256)",
    "fwgs128_512": "amdgpu_flat_work_group_size(128,512)",
    "vgpr256":     "amdgpu_num_vgpr(256)",
    "vgpr224":     "amdgpu_num_vgpr(224)",
    "vgpr192":     "amdgpu_num_vgpr(192)",
    "sgpr96":      "amdgpu_num_sgpr(96)",
    "sgpr80":      "amdgpu_num_sgpr(80)",
    "mnwg8":       "amdgpu_max_num_work_groups(8,1,1)",
}

NEW_PREFIX = "_r17c_"

# Anchor we insert ABOVE.
ANCHOR = "__global__ __launch_bounds__(_NUM_THREADS, 1)\n"

BASE = (
    f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    f"-I/opt/rocm/include/hip "
    f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    f"-shared -fPIC -std=c++20 -w "
    f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm"
)


def build_one(label, N, K, parent_flags, parent_suffix, tag, attr_body):
    csuf = NEW_PREFIX + tag
    full_suffix = parent_suffix + csuf
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if os.path.exists(so_path):
        return (label, tag, full_suffix, "cached", 0.0, "")
    with open(KERNEL_SRC, "r") as f:
        src = f.read()
    if src.count(ANCHOR) != 1:
        return (label, tag, full_suffix, "FAIL", 0.0, "anchor not unique")
    inject = f"__attribute__(({attr_body}))\n" + ANCHOR
    patched = src.replace(ANCHOR, inject, 1)
    patched = patched.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},",
    )
    wrapper_src = os.path.join(BUILD_DIR, f"wrap_n{N}_k{K}{full_suffix}.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)
    cmd = (
        f"/opt/rocm/bin/hipcc {wrapper_src} {BASE} "
        f"-DK_DIM={K} -DN_DIM={N} {parent_flags} -o {so_path}"
    )
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(so_path):
        return (label, tag, full_suffix, "FAIL", dt, r.stderr[-400:])
    return (label, tag, full_suffix, "OK", dt, "")


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    workers = int(os.environ.get("BUILD_WORKERS", "8"))
    jobs = []
    for (lab, M, N, K, ps, pf) in SHAPES:
        for tag, body in ATTRS.items():
            jobs.append((lab, M, N, K, ps, pf, tag, body))
    print(f"Total builds: {len(jobs)} ({len(SHAPES)} shapes x {len(ATTRS)} attrs) workers={workers}")
    t0 = time.time()
    fail_lines = []
    n_ok = n_cached = n_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_one, lab, N, K, pf, ps, tag, body): (lab, tag)
                for (lab, M, N, K, ps, pf, tag, body) in jobs}
        for fut in as_completed(futs):
            label, tag, full_suffix, status, dt, err = fut.result()
            print(f"  {label:6s} {tag:14s}  {status:8s}  ({dt:.1f}s)", flush=True)
            if status == "FAIL":
                n_fail += 1
                fail_lines.append(f"--- {label} {tag} ---\n{err}\n")
            elif status == "cached":
                n_cached += 1
            else:
                n_ok += 1
    print(f"\nElapsed: {time.time()-t0:.1f}s ok={n_ok} cached={n_cached} fail={n_fail}")
    if fail_lines:
        print("\nFAILURES:\n" + "\n".join(fail_lines))
    return 0 if n_fail < len(jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
