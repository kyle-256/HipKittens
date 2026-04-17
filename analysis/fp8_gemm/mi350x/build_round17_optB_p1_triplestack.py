#!/usr/bin/env python3
"""Round 17 Optimizer B: P1 triple/quad-stack compounds (NO iterilp).

Hypothesis: stacking 3 positive sub-threshold flags might compound to +1pp on P1
(28672x4096x16384, parent _ts_gm8). Known sub-threshold winners (R14A/R15A/R16C):
  - regclassglob               -> +0.236pp 5-run (R14A/R15A)
  - regclassglob + tv16        -> +0.225pp 5-run (R15A)
  - regclassglob + noemxpre    -> +0.30pp  5-run (R16C re-verify)

Tests 10 NO-iterilp triple/quad combinations on P1 _ts_gm8.

LAST-SPEC-WINS for sched-strategy: P1 parent has NO -mllvm sched flag, so safe.
regclassglob and noemxpre are NOT sched-strategy flags so safe to stack.
Macros (-DTAIL_BARRIER_VMCNT, -DSTEP3_BARRIER_VMCNT, -DSTEP12_BR_LGKMCNT,
-DSTEP4_EXTERNAL_BR_PREFETCH) are -D defines so always safe.
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

# P1 only.
SHAPE = ("P1", 28672, 4096, 16384, "_ts_gm8", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8")

# Single-flag tag -> string (-mllvm or -D)
TAG_FLAG = {
    "rcg":      "-mllvm -greedy-regclass-priority-trumps-globalness=true",
    "noemxpre": "-mllvm -amdgpu-opt-exec-mask-pre-ra=false",
    "tv16":     "-DTAIL_BARRIER_VMCNT=16",
    "v20":      "-DSTEP3_BARRIER_VMCNT=20",
    "lgk2":     "-DSTEP12_BR_LGKMCNT=2",
    "extbr":    "-DSTEP4_EXTERNAL_BR_PREFETCH=1",
}

# 10 compound variants (NO iterilp)
COMPOUNDS = [
    ("rcg", "noemxpre", "tv16"),                # 3-stack of all 3 known sub-thr winners
    ("rcg", "noemxpre", "v20"),
    ("rcg", "noemxpre", "lgk2"),
    ("rcg", "noemxpre", "extbr"),
    ("rcg", "tv16", "v20"),
    ("rcg", "tv16", "lgk2"),
    ("rcg", "tv16", "extbr"),
    ("rcg", "noemxpre", "tv16", "v20"),         # 4-stack
    ("rcg", "noemxpre", "tv16", "lgk2"),        # 4-stack
    ("rcg", "noemxpre", "tv16", "extbr"),       # 4-stack
]

NEW_PREFIX = "_r17b_"

BASE = (
    f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    f"-I/opt/rocm/include/hip "
    f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    f"-shared -fPIC -std=c++20 -w "
    f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm"
)


def compound_suffix(tags):
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
    lab, M, N, K, ps, pf = SHAPE
    jobs = [(lab, M, N, K, ps, pf, tags) for tags in COMPOUNDS]
    print(f"Total builds: {len(jobs)}  (workers={workers})  shape={lab}")
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
