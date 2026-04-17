#!/usr/bin/env python3
"""Round 14 Optimizer C: DEEP DIVE on DLA1 (4096x32768x128256, mega-K + large-N).

Parent: _ts_pf6_6_v12_memc with flags
  -DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DSTEP12_BR_LGKMCNT=2
  -DSTEP3_BARRIER_VMCNT=12
  -mllvm -amdgpu-sched-strategy=max-memory-clause

Already-tested in bench_deep_lose_results.json (REJECTED — DO NOT REBUILD):
  ts_pf6_6_v12_memc (best 5213.6), ts_pf6_6_u8 (5210.6), pf6_6_v12_memc (worse),
  ts_gm8 (5208.3), ts_gm4 (5206.6), ts_v16_u8/u16, wpe1/wpe2 family, agpr192,
  unrolled u8/u16, u32/v20 family, default sched.

Round 13 sched-strategy DEAD-END for DLA1 (iterilp/iterminreg/itermaxocc/maxilp
all SGPR-clobber bug). max-memory-clause already deployed in parent.

This round: try VECTORS that are NOT scheduler swaps and NOT already in
deep_lose_results. Focus on:

  V1 non-scheduler LLVM flags (membound-threshold, regalloc bias, spill threshold,
     late-structurize, aggressive-dpp-combine, assume-uniform-workgroup-size)
  V2 PF_N=6 + STEP3_BARRIER_VMCNT crosses untested ({4,8,16,20,24})
  V3 PF_N=6 + STEP12_BR_LGKMCNT untested ({0,4})
  V4 PF_N=6 + TAIL_BARRIER_VMCNT untested ({0,4,16}) since TAIL_SPLIT=1 active
  V5 PF_N=6 + GROUP_SIZE_M untested ({1,2,8}) — large-N B-tile reuse
  V6 PF_N=6 + STEP3_EMBED_BARRIER=0 (no_embed) untested for pf6_6
  V7 PF_N=6 + STEP4_EXTERNAL_BR_PREFETCH=1 untested
  V8 PF_N=6 + WAVES_PER_EU_2 (forces 2 waves/SIMD by spilling)
  V9 PERSISTENT_XCD with high BATCH (16, 32) — never tested on DLA1
  V10 STATIC_XCD_REMAP with GROUP_M=4 — DLA1 is B-bound (N=32768)

We REPLACE max-memory-clause flag in parent only when stacking another scheduler
(LAST-WINS rule per R13C). For non-scheduler flags we APPEND only.
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

M, N, K = 4096, 32768, 128256
PARENT_SUFFIX = "_ts_pf6_6_v12_memc"

# Parent's effective flags (from build_round12_optA_bisect.py).
# We split out the sched-strategy so we can preserve it (when not overridden)
# vs. drop it (when adding a new one).
PARENT_NONSCHED = (
    "-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 "
    "-DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12"
)
PARENT_SCHED = "-mllvm -amdgpu-sched-strategy=max-memory-clause"

# (suffix, parent_overrides, extra_flags, keep_parent_sched)
# parent_overrides can replace existing -D defines (we just add new -D AFTER which wins),
# but to override an existing -D you must pass -U first or rely on hipcc -D last-wins.
# hipcc/clang -D last-wins, so appending overrides earlier defines.
VARIANTS = [
    # ─── V1: non-scheduler LLVM flags stacked on parent ────────────────────
    ("_r14c_v1_membound50",
     "", "-mllvm -amdgpu-membound-threshold=50", True),
    ("_r14c_v1_membound200",
     "", "-mllvm -amdgpu-membound-threshold=200", True),
    ("_r14c_v1_regalloc_bias",
     "", "-mllvm -amdgpu-spill-vgpr-to-agpr=true", True),
    ("_r14c_v1_relaxedocc",
     "", "-mllvm -amdgpu-schedule-relaxed-occupancy", True),
    ("_r14c_v1_dceinra",
     "", "-mllvm -amdgpu-dce-in-ra", True),
    ("_r14c_v1_dpp",
     "", "-mllvm -amdgpu-dpp-combine=true", True),
    ("_r14c_v1_loopprefetch",
     "", "-mllvm -amdgpu-loop-prefetch", True),
    ("_r14c_v1_aaincg",
     "", "-mllvm -amdgpu-use-aa-in-codegen=true", True),
    ("_r14c_v1_membound200_relaxedocc",
     "", "-mllvm -amdgpu-membound-threshold=200 -mllvm -amdgpu-schedule-relaxed-occupancy", True),

    # ─── V2: PF6_6 + STEP3_BARRIER_VMCNT untested values ───────────────────
    # parent has v12; try v4, v8, v16, v20, v24
    ("_r14c_v2_v4",
     "-DSTEP3_BARRIER_VMCNT=4", "", True),
    ("_r14c_v2_v8",
     "-DSTEP3_BARRIER_VMCNT=8", "", True),
    ("_r14c_v2_v16",
     "-DSTEP3_BARRIER_VMCNT=16", "", True),
    ("_r14c_v2_v20",
     "-DSTEP3_BARRIER_VMCNT=20", "", True),
    ("_r14c_v2_v24",
     "-DSTEP3_BARRIER_VMCNT=24", "", True),

    # ─── V3: PF6_6 + STEP12_BR_LGKMCNT untested ────────────────────────────
    # parent has 2; try 0 and 4
    ("_r14c_v3_brlgk0",
     "-DSTEP12_BR_LGKMCNT=0", "", True),
    ("_r14c_v3_brlgk4",
     "-DSTEP12_BR_LGKMCNT=4", "", True),

    # ─── V4: PF6_6 + TAIL_BARRIER_VMCNT untested (TAIL_SPLIT=1 active) ─────
    # parent has TBV defaulting to STEP3_BARRIER_VMCNT=12
    ("_r14c_v4_tbv0",
     "-DTAIL_BARRIER_VMCNT=0", "", True),
    ("_r14c_v4_tbv4",
     "-DTAIL_BARRIER_VMCNT=4", "", True),
    ("_r14c_v4_tbv16",
     "-DTAIL_BARRIER_VMCNT=16", "", True),

    # ─── V5: PF6_6 + GROUP_SIZE_M untested (default 4) ─────────────────────
    ("_r14c_v5_gm1",
     "-DGROUP_SIZE_M=1", "", True),
    ("_r14c_v5_gm2",
     "-DGROUP_SIZE_M=2", "", True),
    ("_r14c_v5_gm8",
     "-DGROUP_SIZE_M=8", "", True),

    # ─── V6: PF6_6 + STEP3_EMBED_BARRIER=0 (no_embed) ──────────────────────
    ("_r14c_v6_noembed",
     "-DSTEP3_EMBED_BARRIER=0", "", True),

    # ─── V7: PF6_6 + STEP4_EXTERNAL_BR_PREFETCH=1 ──────────────────────────
    ("_r14c_v7_extbr",
     "-DSTEP4_EXTERNAL_BR_PREFETCH=1", "", True),

    # ─── V8: PF6_6 + WAVES_PER_EU_2 force 2 waves/SIMD ─────────────────────
    ("_r14c_v8_wpe2",
     "-DWAVES_PER_EU_2=1", "", True),

    # ─── V9: PERSISTENT_XCD with high BATCH ────────────────────────────────
    ("_r14c_v9_persistent_b16",
     "-DPERSISTENT_XCD=1 -DPERSISTENT_BATCH=16", "", True),
    ("_r14c_v9_persistent_b32",
     "-DPERSISTENT_XCD=1 -DPERSISTENT_BATCH=32", "", True),

    # ─── V10: STATIC_XCD_REMAP for B-bound large-N ─────────────────────────
    ("_r14c_v10_static_xcd",
     "-DSTATIC_XCD_REMAP=1", "", True),

    # ─── Combo: best-bet stacking ──────────────────────────────────────────
    ("_r14c_combo_v8_membound50",
     "", "-mllvm -amdgpu-membound-threshold=50", True),  # placeholder, see below
]

# Replace the misnamed combo with two real combos
VARIANTS = [v for v in VARIANTS if not v[0].startswith("_r14c_combo")]
VARIANTS += [
    # combo: tbv16 + brlgk4 (epilogue + br looseness for long-K shape)
    ("_r14c_cb_tbv16_brlgk4",
     "-DTAIL_BARRIER_VMCNT=16 -DSTEP12_BR_LGKMCNT=4", "", True),
    # combo: gm8 + tbv16 (B-tile reuse + epilogue tweak)
    ("_r14c_cb_gm8_tbv16",
     "-DGROUP_SIZE_M=8 -DTAIL_BARRIER_VMCNT=16", "", True),
    # combo: membound200 + dpp
    ("_r14c_cb_membound200_dpp",
     "", "-mllvm -amdgpu-membound-threshold=200 -mllvm -amdgpu-dpp-combine=true", True),
]

BASE = (
    f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    f"-I/opt/rocm/include/hip "
    f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    f"-shared -fPIC -std=c++20 -w "
    f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm"
)


def build_one(suffix, parent_overrides, extra_flags, keep_sched, force=False):
    full_suffix = PARENT_SUFFIX + suffix
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
    sched_part = PARENT_SCHED if keep_sched else ""
    cmd = (
        f"/opt/rocm/bin/hipcc {wrapper_src} {BASE} "
        f"-DK_DIM={K} -DN_DIM={N} {PARENT_NONSCHED} {parent_overrides} "
        f"{sched_part} {extra_flags} -o {so_path}"
    )
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900)
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(so_path):
        return (full_suffix, "FAIL", dt, r.stderr[-500:])
    return (full_suffix, "OK", dt, "")


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    print(f"Round 14 OptC DLA1 builds: {len(VARIANTS)} variants")
    print("=" * 100)
    workers = int(os.environ.get("BUILD_WORKERS", "8"))
    t0 = time.time()
    fail_lines = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_one, s, po, ef, kp): s for (s, po, ef, kp) in VARIANTS}
        for fut in as_completed(futs):
            full_suffix, status, dt, err = fut.result()
            print(f"  {full_suffix:50s} {status:8s} ({dt:.1f}s)", flush=True)
            if status == "FAIL":
                fail_lines.append(f"--- {full_suffix} ---\n{err}\n")
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    if fail_lines:
        print("\nFAILURES:\n" + "\n".join(fail_lines))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
