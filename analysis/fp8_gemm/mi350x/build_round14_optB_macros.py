#!/usr/bin/env python3
"""Round 14 OptB: kernel-macro sweep on 4 deep-LOSE shapes (DLA1/DLA2/DLA7/P1).

Targets macro combinations NOT yet tested on these shapes per
bench_deep_lose_results.json + bench_all42_results.json (R10/R11 focused on
sched-strategy axis and skipped many macro crosses on these specific shapes).

Each variant labeled with shape prefix so it is unambiguous in the build dir.
We DO NOT add an iterilp/iterminreg/etc sched-strategy here — pure macro space.
For DLA1/DLA2/DLA7 we keep the parent's `max-memory-clause` flag where the
parent had it, since memc is part of the current best and OK on these shapes.
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

MEMC = "-mllvm -amdgpu-sched-strategy=max-memory-clause"

# (label, M, N, K, suffix, full_compile_flags)
# Suffix encodes macro choices; new suffixes use _r14b prefix per-shape group.
# parent reference for context (not built here):
#   DLA1 parent _ts_pf6_6_v12_memc  -> -DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DSTEP3_BARRIER_VMCNT=12 + MEMC
#   DLA2 parent _ts_gm2_v12_memc_dc -> -DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP3_BARRIER_VMCNT=12 + MEMC + dc
#   DLA7 parent _ts_lgk2_v12_memc   -> -DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 + MEMC
#   P1   parent _ts_gm8             -> -DTAIL_SPLIT=1 -DGROUP_SIZE_M=8

VARIANTS = [
    # ===== DLA1: 4096 x 32768 x 128256 (mega-K + large-N) =====
    # Untested on this shape: pf6_6 + (ext_br | no_embed | tv0 | tv16 | v24 | lgk4 | gm1)
    ("DLA1", 4096, 32768, 128256, "_ts_pf6_6_v12_memc_r14b_extbr",
     f"-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DSTEP3_BARRIER_VMCNT=12 -DSTEP4_EXTERNAL_BR_PREFETCH=1 {MEMC}"),
    ("DLA1", 4096, 32768, 128256, "_ts_pf6_6_v12_memc_r14b_noembed",
     f"-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DSTEP3_BARRIER_VMCNT=12 -DSTEP3_EMBED_BARRIER=0 {MEMC}"),
    ("DLA1", 4096, 32768, 128256, "_ts_pf6_6_v20_memc",
     f"-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DSTEP3_BARRIER_VMCNT=20 {MEMC}"),
    ("DLA1", 4096, 32768, 128256, "_ts_pf6_6_v24_memc",
     f"-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DSTEP3_BARRIER_VMCNT=24 {MEMC}"),
    ("DLA1", 4096, 32768, 128256, "_ts_pf6_6_lgk4_v12_memc",
     f"-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DSTEP12_BR_LGKMCNT=4 -DSTEP3_BARRIER_VMCNT=12 {MEMC}"),
    ("DLA1", 4096, 32768, 128256, "_ts_pf6_6_v12_memc_tv0",
     f"-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DSTEP3_BARRIER_VMCNT=12 -DTAIL_BARRIER_VMCNT=0 {MEMC}"),
    ("DLA1", 4096, 32768, 128256, "_ts_pf6_6_v12_memc_tv16",
     f"-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DSTEP3_BARRIER_VMCNT=12 -DTAIL_BARRIER_VMCNT=16 {MEMC}"),

    # ===== DLA2: 128256 x 32768 x 4096 (mega-M, A-bound) =====
    # Idea: GROUP_M=1 (linear walk -> A-friendly across XCDs), keep memc+dc
    # Also try GM=1 with various macro stacks
    ("DLA2", 128256, 32768, 4096, "_ts_gm1_v12_memc_dc",
     f"-DTAIL_SPLIT=1 -DGROUP_SIZE_M=1 -DSTEP3_BARRIER_VMCNT=12 {MEMC} -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    ("DLA2", 128256, 32768, 4096, "_ts_gm1_lgk2_v12_memc_dc",
     f"-DTAIL_SPLIT=1 -DGROUP_SIZE_M=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 {MEMC} -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    ("DLA2", 128256, 32768, 4096, "_ts_gm1_v20_memc_dc",
     f"-DTAIL_SPLIT=1 -DGROUP_SIZE_M=1 -DSTEP3_BARRIER_VMCNT=20 {MEMC} -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    ("DLA2", 128256, 32768, 4096, "_ts_gm1_v12_memc_dc_extbr",
     f"-DTAIL_SPLIT=1 -DGROUP_SIZE_M=1 -DSTEP3_BARRIER_VMCNT=12 -DSTEP4_EXTERNAL_BR_PREFETCH=1 {MEMC} -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    ("DLA2", 128256, 32768, 4096, "_ts_gm1_v12_memc_dc_tv0",
     f"-DTAIL_SPLIT=1 -DGROUP_SIZE_M=1 -DSTEP3_BARRIER_VMCNT=12 -DTAIL_BARRIER_VMCNT=0 {MEMC} -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    # noembed + parent gm2 stack
    ("DLA2", 128256, 32768, 4096, "_ts_gm2_v12_memc_dc_noembed",
     f"-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP3_BARRIER_VMCNT=12 -DSTEP3_EMBED_BARRIER=0 {MEMC} -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    # gm2 + extbr + memc_dc untested
    ("DLA2", 128256, 32768, 4096, "_ts_gm2_v12_memc_dc_extbr",
     f"-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP3_BARRIER_VMCNT=12 -DSTEP4_EXTERNAL_BR_PREFETCH=1 {MEMC} -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),

    # ===== DLA7: 28672 x 32768 x 4096 (large M+N, small K) =====
    # tail-related (K=4096 -> tail iters MATTER here, unlike DLA1 mega-K).
    # parent uses lgk2+v12+memc, no tail tuning yet on this shape with memc combo.
    ("DLA7", 28672, 32768, 4096, "_ts_lgk2_v12_memc_tv0",
     f"-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -DTAIL_BARRIER_VMCNT=0 {MEMC}"),
    ("DLA7", 28672, 32768, 4096, "_ts_lgk2_v12_memc_tv16",
     f"-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -DTAIL_BARRIER_VMCNT=16 {MEMC}"),
    ("DLA7", 28672, 32768, 4096, "_ts_lgk2_v12_memc_extbr",
     f"-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -DSTEP4_EXTERNAL_BR_PREFETCH=1 {MEMC}"),
    ("DLA7", 28672, 32768, 4096, "_ts_lgk2_v12_memc_noembed",
     f"-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -DSTEP3_EMBED_BARRIER=0 {MEMC}"),
    ("DLA7", 28672, 32768, 4096, "_ts_lgk4_v12_memc",
     f"-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=4 -DSTEP3_BARRIER_VMCNT=12 {MEMC}"),
    ("DLA7", 28672, 32768, 4096, "_ts_lgk2_v20_memc",
     f"-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=20 {MEMC}"),

    # ===== P1: 28672 x 4096 x 16384 (large K + large M) =====
    # parent _ts_gm8 (no memc, no lgk, no v-cnt change). HUGE untested space.
    # Untested on this shape: ts_gm8 + memc, ts_gm8 + lgk2_memc, ts_gm8 + extbr,
    # ts_gm8 + tv0, ts_gm8 + v12_memc + extbr, ts_gm8 + noembed
    ("P1", 28672, 4096, 16384, "_ts_gm8_memc",
     f"-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 {MEMC}"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_lgk2_memc",
     f"-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DSTEP12_BR_LGKMCNT=2 {MEMC}"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_v12_memc",
     f"-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DSTEP3_BARRIER_VMCNT=12 {MEMC}"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_extbr",
     f"-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DSTEP4_EXTERNAL_BR_PREFETCH=1"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_extbr_memc",
     f"-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DSTEP4_EXTERNAL_BR_PREFETCH=1 {MEMC}"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_noembed",
     f"-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DSTEP3_EMBED_BARRIER=0"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_noembed_memc",
     f"-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DSTEP3_EMBED_BARRIER=0 {MEMC}"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_v20_memc",
     f"-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DSTEP3_BARRIER_VMCNT=20 {MEMC}"),
    # tail tuning is irrelevant on K=16384 large-K, skip TV cross
]

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
    workers = int(os.environ.get("BUILD_WORKERS", "10"))
    print(f"Total builds: {len(VARIANTS)}  (workers={workers})")
    t0 = time.time()
    fail_lines = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_one, lab, N, K, suf, fl): (lab, suf)
                for (lab, M, N, K, suf, fl) in VARIANTS}
        for fut in as_completed(futs):
            label, suffix, status, dt, err = fut.result()
            print(f"  {label:5s} {suffix:50s} {status:8s} ({dt:.1f}s)", flush=True)
            if status == "FAIL":
                fail_lines.append(f"--- {label} {suffix} ---\n{err}\n")
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    if fail_lines:
        print("\nFAILURES:\n" + "\n".join(fail_lines))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
