#!/usr/bin/env python3
"""Round 16 Optimizer B: compound R15B SAFE DIFF flags ON TOP OF iterilp wins.

R10/R11 verified 5 deep-LOSE iterilp wins (S1-S5). R15B identified 4 SAFE
LLVM flags that produce non-trivial .text mutations without aperture-crash
or major regressions:

  - noemxpre       (-mllvm -amdgpu-opt-exec-mask-pre-ra=false)
  - largeivf2      (-mllvm -large-interval-freq-threshold=2)
  - nolicm         (-mllvm -disable-machine-licm)
  - sinkavoidspill (-mllvm -sink-insts-to-avoid-spills)

Compound `iterilp + each_safe_R15B_flag` has NEVER been tested on S1-S5.

This script:
  1. Builds iterilp baselines for S1 (_lgk2_dc) + S2 (_u32) if missing — these
     are NEW iterilp variants per AGENT_PROMPT current-best rebase.
  2. Builds 5 shapes x 4 flags = 20 R16B compound variants.

Variant naming: `<parent>_r16b_iterilp_<flag_tag>`, so iterilp + the new flag
are both visible in suffix. Source flags include both `iterative-ilp`
sched-strategy and the R15B flag.
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

ITERILP = "-mllvm -amdgpu-sched-strategy=iterative-ilp"

# (label, M, N, K, parent_suffix, parent_cppflags)
# Parent_cppflags include all macros AND any -mllvm flag in the existing parent suffix.
# The iterilp baseline = parent + ITERILP, suffix = parent + "_r10_iterilp" (S1/S2)
#                                                  or parent + "_r11_iterilp" (S3/S4/S5).
SHAPES = [
    # S1: 14336x4096x32768  parent _lgk2_dc; iterilp tag _r10_iterilp (per AGENT_PROMPT)
    ("S1", 14336, 4096, 32768, "_lgk2_dc",
     "-DSTEP12_BR_LGKMCNT=2 -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule",
     "_r10_iterilp"),
    # S2: 16384x4096x28672  parent _u32; iterilp tag _r10_iterilp
    ("S2", 16384, 4096, 28672, "_u32",
     "-DUNROLL_K=32",
     "_r10_iterilp"),
    # S3: 4096x32768x28672  parent _v20_memc; iterilp tag _r11_iterilp (already built)
    ("S3", 4096, 32768, 28672, "_v20_memc",
     "-DSTEP3_BARRIER_VMCNT=20 -mllvm -amdgpu-sched-strategy=max-memory-clause",
     "_r11_iterilp"),
    # S4: 4096x28672x32768  parent _u16; iterilp tag _r11_iterilp (already built)
    ("S4", 4096, 28672, 32768, "_u16",
     "-DUNROLL_K=16",
     "_r11_iterilp"),
    # S5: 4096x32768x14336  parent _ts_lgk2_memc; iterilp tag _r11_iterilp (already built)
    ("S5", 4096, 32768, 14336, "_ts_lgk2_memc",
     "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -mllvm -amdgpu-sched-strategy=max-memory-clause",
     "_r11_iterilp"),
]

# 4 R15B SAFE-DIFF flags. sinkbfi excluded (NOOP on all R15B shapes).
SAFE_FLAGS = [
    ("noemxpre",       "-mllvm -amdgpu-opt-exec-mask-pre-ra=false"),
    ("largeivf2",      "-mllvm -large-interval-freq-threshold=2"),
    ("nolicm",         "-mllvm -disable-machine-licm"),
    ("sinkavoidspill", "-mllvm -sink-insts-to-avoid-spills"),
]

R16B_INFIX = "_r16b_iterilp_"

BASE = (
    f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    f"-I/opt/rocm/include/hip "
    f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    f"-shared -fPIC -std=c++20 -w "
    f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm"
)


def build_so(N, K, full_suffix, full_flags):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if os.path.exists(so_path):
        return ("cached", 0.0, "", so_path)
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
        f"-DK_DIM={K} -DN_DIM={N} {full_flags} -o {so_path}"
    )
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900)
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(so_path):
        return ("FAIL", dt, r.stderr[-500:], so_path)
    return ("OK", dt, "", so_path)


def build_one_compound(lab, M, N, K, parent_suf, parent_flags, ilp_suf, tag, ef):
    full_suffix = parent_suf + R16B_INFIX + tag
    full_flags = f"{parent_flags} {ITERILP} {ef}"
    status, dt, err, so_path = build_so(N, K, full_suffix, full_flags)
    return (lab, "compound", tag, full_suffix, status, dt, err)


def build_one_iterilp_baseline(lab, M, N, K, parent_suf, parent_flags, ilp_suf):
    full_suffix = parent_suf + ilp_suf
    full_flags = f"{parent_flags} {ITERILP}"
    status, dt, err, so_path = build_so(N, K, full_suffix, full_flags)
    return (lab, "iterilp_base", ilp_suf, full_suffix, status, dt, err)


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    workers = int(os.environ.get("BUILD_WORKERS", "8"))
    jobs = []
    # Iterilp baselines (only S1/S2 missing; S3/S4/S5 cached)
    for (lab, M, N, K, ps, pf, ilp_suf) in SHAPES:
        jobs.append(("base", lab, M, N, K, ps, pf, ilp_suf, None, None))
    # Compound variants
    for (lab, M, N, K, ps, pf, ilp_suf) in SHAPES:
        for (tag, ef) in SAFE_FLAGS:
            jobs.append(("comp", lab, M, N, K, ps, pf, ilp_suf, tag, ef))
    print(f"Total builds: {len(jobs)}  (workers={workers})  "
          f"({len(SHAPES)} iterilp bases + {len(SHAPES) * len(SAFE_FLAGS)} compounds)")
    t0 = time.time()
    fail_lines = []
    n_ok = n_cached = n_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for j in jobs:
            kind, lab, M, N, K, ps, pf, ilp_suf, tag, ef = j
            if kind == "base":
                fut = ex.submit(build_one_iterilp_baseline, lab, M, N, K, ps, pf, ilp_suf)
            else:
                fut = ex.submit(build_one_compound, lab, M, N, K, ps, pf, ilp_suf, tag, ef)
            futs[fut] = (kind, lab, tag)
        for fut in as_completed(futs):
            lab, kind2, tag, full_suffix, status, dt, err = fut.result()
            label = f"{lab}/{kind2}{('/' + str(tag)) if tag else ''}"
            print(f"  {label:35s}  {status:8s}  ({dt:.1f}s)  {full_suffix}",
                  flush=True)
            if status == "FAIL":
                n_fail += 1
                fail_lines.append(f"--- {label} ---\n{err}\n")
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
