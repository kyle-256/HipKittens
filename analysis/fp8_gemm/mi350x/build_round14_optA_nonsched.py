#!/usr/bin/env python3
"""Round 14 Optimizer A: sweep NON-scheduler LLVM AMDGPU flags on the 4
deep-LOSE shapes that the iterative-ilp/sched-strategy axis has now
exhausted (R12 + R13 findings).

Targets (per AGENT_PROMPT R13 frontier):
  DLA1: 4096x32768x128256, parent _ts_pf6_6_v12_memc, 88.3% (mega-K)
  DLA2: 128256x32768x4096, parent _ts_gm2_v12_memc_dc, 92.9% (mega-M)
  DLA7: 28672x32768x4096,  parent _ts_lgk2_v12_memc,  94.5% (large M+N)
  P1:   28672x4096x16384,  parent _ts_gm8,            93.7% (large K+M)

Each (flag, shape) is built into a `_r14a_<flag>` variant.

All candidate flags verified to exist in this LLVM (llc -help-hidden).
Discarded from the original prompt list because they don't exist:
    amdgpu-spill-threshold, amdgpu-disable-power-sched,
    amdgpu-late-structurize, amdgpu-aggressive-dpp-combine,
    amdgpu-codegenprepare-disable-promote-alloca-to-vector,
    amdgpu-vgpr-index-mode-spill-threshold,
    misched-cluster-mem-load
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
    ("DLA1", 4096,  32768, 128256, "_ts_pf6_6_v12_memc",
     "-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("DLA2", 128256, 32768,  4096, "_ts_gm2_v12_memc_dc",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    ("DLA7", 28672, 32768,  4096, "_ts_lgk2_v12_memc",
     "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    # 28672x4096x16384 parent ts_gm8 (no scheduler flags in parent)
    ("P1",   28672,  4096, 16384, "_ts_gm8",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8"),
]

# (tag, extra_flags)  -- tag goes into the variant suffix
FLAGS = [
    # 1. amdgpu-membound-threshold
    ("mb0",     "-mllvm -amdgpu-membound-threshold=0"),
    ("mb50",    "-mllvm -amdgpu-membound-threshold=50"),
    ("mb100",   "-mllvm -amdgpu-membound-threshold=100"),
    ("mb200",   "-mllvm -amdgpu-membound-threshold=200"),
    # 1b. relaxed-occupancy schedule (interacts with membound-threshold)
    ("relaxocc", "-mllvm -amdgpu-schedule-relaxed-occupancy"),
    ("mb0_relaxocc", "-mllvm -amdgpu-membound-threshold=0 -mllvm -amdgpu-schedule-relaxed-occupancy"),

    # 2. amdgpu-promote-alloca-to-vector-limit
    ("pav16",   "-mllvm -amdgpu-promote-alloca-to-vector-limit=16"),
    ("pav32",   "-mllvm -amdgpu-promote-alloca-to-vector-limit=32"),
    ("pav64",   "-mllvm -amdgpu-promote-alloca-to-vector-limit=64"),
    ("nopav",   "-mllvm -disable-promote-alloca-to-vector"),

    # 3-5. simple boolean toggles (default values vary by flag)
    ("nodiv",   "-mllvm -amdgpu-bypass-slow-div=false"),
    ("nomergem0", "-mllvm -amdgpu-enable-merge-m0=false"),
    ("nomisched", "-mllvm -misched-cluster=false"),
    ("dppc",    "-mllvm -amdgpu-dpp-combine=true"),
    ("nodppc",  "-mllvm -amdgpu-dpp-combine=false"),
    ("vgprix",  "-mllvm -amdgpu-vgpr-index-mode=true"),

    # 6. regalloc bias
    ("regclassglob", "-mllvm -greedy-regclass-priority-trumps-globalness=true"),

    # 7. reschedule disabling
    ("nolowoccr", "-mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    ("nohighrpr", "-mllvm -amdgpu-disable-unclustered-high-rp-reschedule"),

    # 8. large-stride-threshold (default 64)
    ("lst16",   "-mllvm -amdgpu-large-stride-threshold=16"),
    ("lst256",  "-mllvm -amdgpu-large-stride-threshold=256"),

    # 9. sgpr hazard mem wait cull threshold
    ("sghazard0",  "-mllvm -amdgpu-sgpr-hazard-mem-wait-cull-threshold=0"),
    ("sghazard64", "-mllvm -amdgpu-sgpr-hazard-mem-wait-cull-threshold=64"),

    # 10. wave priority valu insts threshold (default 100)
    ("wpv0",    "-mllvm -amdgpu-set-wave-priority-valu-insts-threshold=0"),
    ("wpv500",  "-mllvm -amdgpu-set-wave-priority-valu-insts-threshold=500"),

    # 11. limit-wave-threshold (default 0)
    ("lwt50",   "-mllvm -amdgpu-limit-wave-threshold=50"),
    ("lwt100",  "-mllvm -amdgpu-limit-wave-threshold=100"),
]

NEW_PREFIX = "_r14a_"

BASE = (
    f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    f"-I/opt/rocm/include/hip "
    f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    f"-shared -fPIC -std=c++20 -w "
    f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm"
)


def build_one(label, N, K, parent_flags, parent_suffix, tag, extra_flags):
    full_suffix = parent_suffix + NEW_PREFIX + tag
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if os.path.exists(so_path):
        return (label, tag, full_suffix, "cached", 0.0, "")
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
        return (label, tag, full_suffix, "FAIL", dt, r.stderr[-400:])
    return (label, tag, full_suffix, "OK", dt, "")


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    workers = int(os.environ.get("BUILD_WORKERS", "8"))
    jobs = []
    for (lab, M, N, K, ps, pf) in SHAPES:
        for (tag, ef) in FLAGS:
            jobs.append((lab, M, N, K, ps, pf, tag, ef))
    print(f"Total builds: {len(jobs)}  (workers={workers}, {len(SHAPES)} shapes x {len(FLAGS)} flags)")
    t0 = time.time()
    fail_lines = []
    n_ok = n_cached = n_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_one, lab, N, K, pf, ps, tag, ef): (lab, tag)
                for (lab, M, N, K, ps, pf, tag, ef) in jobs}
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
    print(f"\nElapsed: {time.time()-t0:.1f}s   ok={n_ok} cached={n_cached} fail={n_fail}")
    if fail_lines:
        print("\nFAILURES:\n" + "\n".join(fail_lines))
    return 0 if not fail_lines else 1


if __name__ == "__main__":
    sys.exit(main())
