#!/usr/bin/env python3
"""Round 15 OptA: cross-shape regclassglob probe + P1 compound builds.

R14A discovered a sub-threshold +0.24pp signal on P1 with the flag
  -mllvm -greedy-regclass-priority-trumps-globalness=true
The R14A asm-diff probe also showed the flag produces DIFF text on
all 4 deep-LOSE shapes (DLA1/DLA2/DLA7/P1), but only P1 appeared in
the verify run. R15A re-applies the flag systematically:

  Phase A: rebuild `_r15a_regclassglob` for the canonical parent of
           each of the 4 deep-LOSE shapes.
  Phase B: build 7 compound variants on P1 (the only confirmed >+0pp
           shape in R14A), pairing regclassglob with each of:
             v20, v24, lgk2, extbr, noembed, tv0, tv16

We re-run the asm-diff probe afterwards (asm_diff_probe_r15a.py) to
freshly hash the .text and confirm DIFF/NOOP per (shape,variant).

NOTE: P1 parent has no STEP3_BARRIER_VMCNT/STEP12_BR_LGKMCNT/etc set.
We add the macro on top of `-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8` per parent.
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

MEMC = "-mllvm -amdgpu-sched-strategy=max-memory-clause"
DC   = "-mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"
RCG  = "-mllvm -greedy-regclass-priority-trumps-globalness=true"

# ============================================================
# Phase A: regclassglob on the 4 deep-LOSE parent kernels
# ============================================================
PHASE_A = [
    # (label, M, N, K, suffix, parent_flags)
    ("DLA1", 4096, 32768, 128256,
     "_ts_pf6_6_v12_memc_r15a_regclassglob",
     f"-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DSTEP3_BARRIER_VMCNT=12 {MEMC}"),
    ("DLA2", 128256, 32768, 4096,
     "_ts_gm2_v12_memc_dc_r15a_regclassglob",
     f"-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP3_BARRIER_VMCNT=12 {MEMC} {DC}"),
    ("DLA7", 28672, 32768, 4096,
     "_ts_lgk2_v12_memc_r15a_regclassglob",
     f"-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 {MEMC}"),
    ("P1", 28672, 4096, 16384,
     "_ts_gm8_r15a_regclassglob",
     f"-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8"),
]

# ============================================================
# Phase B: P1 regclassglob + macro tweak compounds (7 candidates)
# ============================================================
PHASE_B = [
    # (label, M, N, K, suffix, base_macros)
    ("P1", 28672, 4096, 16384, "_ts_gm8_r15a_regclassglob_v20",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DSTEP3_BARRIER_VMCNT=20"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_r15a_regclassglob_v24",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DSTEP3_BARRIER_VMCNT=24"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_r15a_regclassglob_lgk2",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DSTEP12_BR_LGKMCNT=2"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_r15a_regclassglob_extbr",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DSTEP4_EXTERNAL_BR_PREFETCH=1"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_r15a_regclassglob_noembed",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DSTEP3_EMBED_BARRIER=0"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_r15a_regclassglob_tv0",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DTAIL_BARRIER_VMCNT=0"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_r15a_regclassglob_tv16",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DTAIL_BARRIER_VMCNT=16"),
]

ALL_JOBS = [(lab, M, N, K, suf, macros + " " + RCG) for (lab, M, N, K, suf, macros) in (PHASE_A + PHASE_B)]

BASE = (
    f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    f"-I/opt/rocm/include/hip "
    f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    f"-shared -fPIC -std=c++20 -w "
    f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm"
)


def build_one(label, N, K, suffix, flags):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if os.path.exists(so_path):
        return (label, suffix, "cached", 0.0, "")
    with open(KERNEL_SRC, "r") as f:
        src = f.read()
    patched = src.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},"
    )
    wrapper_src = os.path.join(BUILD_DIR, f"wrap_n{N}_k{K}{suffix}.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)
    cmd = (
        f"/opt/rocm/bin/hipcc {wrapper_src} {BASE} "
        f"-DK_DIM={K} -DN_DIM={N} {flags} -o {so_path}"
    )
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(so_path):
        return (label, suffix, "FAIL", dt, r.stderr[-400:])
    return (label, suffix, "OK", dt, "")


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    workers = int(os.environ.get("BUILD_WORKERS", "8"))
    print(f"Total builds: {len(ALL_JOBS)}  (workers={workers})")
    print(f"Phase A (4 shapes): regclassglob alone")
    print(f"Phase B (7 P1 compounds): regclassglob + macro tweak")
    t0 = time.time()
    fail_lines = []
    n_ok = n_cached = n_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_one, lab, N, K, suf, fl): (lab, suf)
                for (lab, M, N, K, suf, fl) in ALL_JOBS}
        for fut in as_completed(futs):
            label, suffix, status, dt, err = fut.result()
            print(f"  {label:5s} {suffix:60s} {status:8s} ({dt:.1f}s)", flush=True)
            if status == "FAIL":
                n_fail += 1
                fail_lines.append(f"--- {label} {suffix} ---\n{err}\n")
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
