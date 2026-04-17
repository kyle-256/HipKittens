#!/usr/bin/env python3
"""Round 15 OptC source-level micro-probes.

3 NEW MACROS added to kernel_mxfp4_gluon_cpp.cpp:
  - TAIL_BARRIER_LGKMCNT (default -1 = off): adds extra lgkmcnt+barrier in TAIL_SPLIT tail.
  - STEP4_BARRIER_VMCNT  (default -1 = off): adds vmcnt+barrier between Step3 emit_pf_tail and Step4.
  - PF_GROUP_OFFSET      (default 0): rotates pf1 emission order by +1 / -1.

Parent: _ts_pf6_6_v12_memc with flags
  -DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DSTEP12_BR_LGKMCNT=2
  -DSTEP3_BARRIER_VMCNT=12
  -mllvm -amdgpu-sched-strategy=max-memory-clause

Probe matrix:
  P1 TAIL_BARRIER_LGKMCNT in {0,2,4}        — DLA7 (K=4096 small-K, only shape where tail is material)
  P2 STEP4_BARRIER_VMCNT  in {4,8,12,16,20} — DLA1 (K=128256) + DLA7 (K=4096)
  P3 PF_GROUP_OFFSET      in {-1, +1}       — DLA1 only

We re-build the parent fresh too as a sanity check (silent no-default change should yield asm-identical .so).
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

PARENT_SUFFIX = "_ts_pf6_6_v12_memc"
PARENT_NONSCHED = (
    "-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 "
    "-DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12"
)
PARENT_SCHED = "-mllvm -amdgpu-sched-strategy=max-memory-clause"

# DLA1: 4096x32768x128256, DLA7: 28672x32768x4096
SHAPES = {
    "DLA1": (4096, 32768, 128256),
    "DLA7": (28672, 32768, 4096),
}

# (suffix, extra_defines, shape_keys)
VARIANTS = [
    # P1: TAIL_BARRIER_LGKMCNT — DLA7 only
    ("_r15c_p1_lgkm0",  "-DTAIL_BARRIER_LGKMCNT=0", ["DLA7"]),
    ("_r15c_p1_lgkm2",  "-DTAIL_BARRIER_LGKMCNT=2", ["DLA7"]),
    ("_r15c_p1_lgkm4",  "-DTAIL_BARRIER_LGKMCNT=4", ["DLA7"]),
    # P2: STEP4_BARRIER_VMCNT — DLA1 + DLA7
    ("_r15c_p2_s4v4",   "-DSTEP4_BARRIER_VMCNT=4",  ["DLA1", "DLA7"]),
    ("_r15c_p2_s4v8",   "-DSTEP4_BARRIER_VMCNT=8",  ["DLA1", "DLA7"]),
    ("_r15c_p2_s4v12",  "-DSTEP4_BARRIER_VMCNT=12", ["DLA1", "DLA7"]),
    ("_r15c_p2_s4v16",  "-DSTEP4_BARRIER_VMCNT=16", ["DLA1", "DLA7"]),
    ("_r15c_p2_s4v20",  "-DSTEP4_BARRIER_VMCNT=20", ["DLA1", "DLA7"]),
    # P3: PF_GROUP_OFFSET — DLA1 only
    ("_r15c_p3_pfg_neg1", "-DPF_GROUP_OFFSET=-1", ["DLA1"]),
    ("_r15c_p3_pfg_pos1", "-DPF_GROUP_OFFSET=1",  ["DLA1"]),
]

# Add parent rebuild for both shapes (sanity)
SANITY = [
    ("_r15c_parent_rebuild_DLA1", "", ["DLA1"]),
    ("_r15c_parent_rebuild_DLA7", "", ["DLA7"]),
]

BASE = (
    f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    f"-I/opt/rocm/include/hip "
    f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    f"-shared -fPIC -std=c++20 -w "
    f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm"
)


def build_one(M, N, K, suffix, extra_defs, force=True):
    full_suffix = PARENT_SUFFIX + suffix
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if force and os.path.exists(so_path):
        os.remove(so_path)
    if os.path.exists(so_path):
        return (full_suffix, M, N, K, "cached", 0.0, "")
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
        f"-DK_DIM={K} -DN_DIM={N} {PARENT_NONSCHED} {extra_defs} "
        f"{PARENT_SCHED} -o {so_path}"
    )
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900)
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(so_path):
        return (full_suffix, M, N, K, "FAIL", dt, r.stderr[-500:])
    return (full_suffix, M, N, K, "OK", dt, "")


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    jobs = []
    # Parent rebuilds (sanity check) — one for DLA1, one for DLA7 (different N,K)
    for suffix, extra_defs, shapes in SANITY:
        for sk in shapes:
            M, N, K = SHAPES[sk]
            # Use the canonical parent suffix (no extra) — but we need to ensure
            # it exists. Bench will use PARENT_SUFFIX directly; just touch it.
            full_suffix = PARENT_SUFFIX  # canonical
            module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
            so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
            if os.path.exists(so_path):
                continue
            jobs.append((M, N, K, "", "", True))
    for suffix, extra_defs, shapes in VARIANTS:
        for sk in shapes:
            M, N, K = SHAPES[sk]
            jobs.append((M, N, K, suffix, extra_defs, True))

    print(f"Round 15 OptC source-level builds: {len(jobs)} jobs")
    print("=" * 110)
    workers = int(os.environ.get("BUILD_WORKERS", "12"))
    t0 = time.time()
    fail_lines = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_one, M, N, K, s, ed, fc): (M, N, K, s)
                for (M, N, K, s, ed, fc) in jobs}
        for fut in as_completed(futs):
            full_suffix, M, N, K, status, dt, err = fut.result()
            print(f"  M={M:>6} N={N:>6} K={K:>6}  {full_suffix:50s} {status:8s} ({dt:.1f}s)", flush=True)
            if status == "FAIL":
                fail_lines.append(f"--- M={M} N={N} K={K} {full_suffix} ---\n{err}\n")
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    if fail_lines:
        print("\nFAILURES:\n" + "\n".join(fail_lines))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
