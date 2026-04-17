#!/usr/bin/env python3
"""Round 16 Optimizer C: cross-axis compounds on the 4 stuck deep-LOSE shapes.

Tests pair-wise / triple compounds of:
  - iterilp     (-mllvm -amdgpu-sched-strategy=iterative-ilp)        # known broken on DLA1/DLA2/DLA7
  - regclassglob (-mllvm -greedy-regclass-priority-trumps-globalness=true)
  - sinkavoidspill (-mllvm -sink-insts-to-avoid-spills)
  - nolicm       (-mllvm -disable-machine-licm)
  - noemxpre     (-mllvm -amdgpu-opt-exec-mask-pre-ra=false)
  - largeivf2    (-mllvm -large-interval-freq-threshold=2)

KEY HYPOTHESIS: regalloc-changing flags (regclassglob, sinkavoidspill, nolicm)
might restructure register allocation enough to AVOID the SGPR-clobber bug
that breaks iterilp on DLA1/DLA2/DLA7.

LAST-SPEC-WINS for sched-strategy: iterilp comes from -mllvm sched-strategy
override. Each parent already has -mllvm -amdgpu-sched-strategy=max-memory-clause
(except P1). Appending iterilp will make iterilp win (last-spec-wins). For
non-iterilp compounds we keep the parent's memc unchanged (no override).

Compound naming convention:
  parent_suffix + "_r16c_" + tag1[+tag2[+tag3]]
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

# Same 4 broken deep-LOSE shapes / parents as R15B.
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

# Single-flag tag -> -mllvm flag string
TAG_FLAG = {
    "iterilp":         "-mllvm -amdgpu-sched-strategy=iterative-ilp",
    "regclassglob":    "-mllvm -greedy-regclass-priority-trumps-globalness=true",
    "sinkavoidspill":  "-mllvm -sink-insts-to-avoid-spills",
    "nolicm":          "-mllvm -disable-machine-licm",
    "noemxpre":        "-mllvm -amdgpu-opt-exec-mask-pre-ra=false",
    "largeivf2":       "-mllvm -large-interval-freq-threshold=2",
}

# Compound tag list: tuple of single-tags (in canonical order so suffix is unique).
COMPOUNDS = [
    # iterilp pair-wise (4 shapes each → tests SGPR-bug avoidance hypothesis)
    ("iterilp", "regclassglob"),
    ("iterilp", "sinkavoidspill"),
    ("iterilp", "nolicm"),
    ("iterilp", "noemxpre"),
    ("iterilp", "largeivf2"),
    # non-iterilp pair-wise (safer; should avoid bug entirely)
    ("regclassglob", "sinkavoidspill"),
    ("regclassglob", "nolicm"),
    ("regclassglob", "noemxpre"),
    ("sinkavoidspill", "nolicm"),
    # triples
    ("regclassglob", "nolicm", "sinkavoidspill"),
    ("iterilp", "regclassglob", "nolicm"),
    ("iterilp", "regclassglob", "sinkavoidspill"),
]

NEW_PREFIX = "_r16c_"

BASE = (
    f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    f"-I/opt/rocm/include/hip "
    f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    f"-shared -fPIC -std=c++20 -w "
    f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm"
)


def compound_suffix(tags):
    # Use 'X' as join char (valid C identifier) instead of '+'.
    return NEW_PREFIX + "X".join(tags)


def compound_flags(tags):
    return " ".join(TAG_FLAG[t] for t in tags)


def build_one(label, N, K, parent_flags, parent_suffix, tags):
    csuf = compound_suffix(tags)
    full_suffix = parent_suffix + csuf
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if os.path.exists(so_path):
        return (label, csuf, full_suffix, "cached", 0.0, "")
    with open(KERNEL_SRC, "r") as f:
        src = f.read()
    patched = src.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},"
    )
    wrapper_src = os.path.join(BUILD_DIR, f"wrap_n{N}_k{K}{full_suffix}.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)
    extra = compound_flags(tags)
    cmd = (
        f"/opt/rocm/bin/hipcc {wrapper_src} {BASE} "
        f"-DK_DIM={K} -DN_DIM={N} {parent_flags} {extra} -o {so_path}"
    )
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(so_path):
        return (label, csuf, full_suffix, "FAIL", dt, r.stderr[-400:])
    return (label, csuf, full_suffix, "OK", dt, "")


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    workers = int(os.environ.get("BUILD_WORKERS", "8"))
    jobs = []
    for (lab, M, N, K, ps, pf) in SHAPES:
        for tags in COMPOUNDS:
            jobs.append((lab, M, N, K, ps, pf, tags))
    print(f"Total builds: {len(jobs)}  (workers={workers}, {len(SHAPES)} shapes x {len(COMPOUNDS)} compounds)")
    t0 = time.time()
    fail_lines = []
    n_ok = n_cached = n_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_one, lab, N, K, pf, ps, tags): (lab, tags)
                for (lab, M, N, K, ps, pf, tags) in jobs}
        for fut in as_completed(futs):
            label, csuf, full_suffix, status, dt, err = fut.result()
            print(f"  {label:6s} {csuf:60s}  {status:8s}  ({dt:.1f}s)", flush=True)
            if status == "FAIL":
                n_fail += 1
                fail_lines.append(f"--- {label} {csuf} ---\n{err}\n")
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
