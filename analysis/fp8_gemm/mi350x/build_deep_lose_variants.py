#!/usr/bin/env python3
"""Build targeted variants for the 10 deep-LOSE shapes (large-K and large-tile).

Strategy:
- Cross-product targeting:
  * Mega-K (≥14336): waves_per_eu (1/2), UNROLL_K combos, deeper PF interactions
  * N=32768/28672 (B-bound): bigger gm × pf × ext_br combos
  * Mega-M (128256): gm1/gm4 × waves_per_eu

Builds ONLY for the 9 unique (N,K) pairs of the deep-LOSE shapes.
Idempotent: skips already-built .so files.
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

# Only the (N, K) pairs of the 10 deep-LOSE shapes
NK_PAIRS = [
    (32768, 128256),  # 4096×32768×128256
    (4096, 32768),    # 14336×4096×32768
    (4096, 28672),    # 16384×4096×28672
    (32768, 4096),    # 128256×32768×4096, 28672×32768×4096
    (32768, 28672),   # 4096×32768×28672
    (4096, 16384),    # 28672×4096×16384
    (28672, 32768),   # 4096×28672×32768
    (4096, 14336),    # 32768×4096×14336
    (32768, 14336),   # 4096×32768×14336
]

# Targeted NEW variants for deep-LOSE shapes
DEEP_LOSE_VARIANTS = [
    # ── Group A: waves_per_eu (occupancy/ILP tradeoffs) ────────────────────────
    # Mega-K shapes might benefit from higher occupancy or more ILP per wave
    ("_wpe1",                    "-DWAVES_PER_EU_1=1"),
    ("_wpe2",                    "-DWAVES_PER_EU_2=1"),
    ("_ts_lgk2_v12_wpe1_memc",   "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -DWAVES_PER_EU_1=1 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("_ts_lgk2_v12_wpe2_memc",   "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -DWAVES_PER_EU_2=1 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("_v16_wpe1",                "-DSTEP3_BARRIER_VMCNT=16 -DWAVES_PER_EU_1=1"),
    ("_v16_wpe1_memc",           "-DSTEP3_BARRIER_VMCNT=16 -DWAVES_PER_EU_1=1 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("_ts_v12_wpe1",             "-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=12 -DWAVES_PER_EU_1=1"),
    ("_ts_gm8_wpe1",             "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DWAVES_PER_EU_1=1"),
    ("_ts_gm8_wpe1_memc",        "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DWAVES_PER_EU_1=1 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("_u8_wpe1",                 "-DUNROLL_K=8 -DWAVES_PER_EU_1=1"),
    ("_u16_wpe1",                "-DUNROLL_K=16 -DWAVES_PER_EU_1=1"),
    ("_v16_wpe2",                "-DSTEP3_BARRIER_VMCNT=16 -DWAVES_PER_EU_2=1"),

    # ── Group B: large-K UNROLL × LGK × VMCNT crosses (NEW) ───────────────────
    # UNROLL_K=8/16 are existing winners; cross with LGK=2 + larger VMCNT
    ("_u8_lgk2",                 "-DUNROLL_K=8 -DSTEP12_BR_LGKMCNT=2"),
    ("_u8_lgk2_v12",             "-DUNROLL_K=8 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12"),
    ("_u8_lgk2_memc",            "-DUNROLL_K=8 -DSTEP12_BR_LGKMCNT=2 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("_u8_v12_memc",             "-DUNROLL_K=8 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("_u16_lgk2",                "-DUNROLL_K=16 -DSTEP12_BR_LGKMCNT=2"),
    ("_u16_lgk2_memc",           "-DUNROLL_K=16 -DSTEP12_BR_LGKMCNT=2 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("_u16_v12",                 "-DUNROLL_K=16 -DSTEP3_BARRIER_VMCNT=12"),
    ("_u16_v12_memc",            "-DUNROLL_K=16 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("_ts_u16_lgk2",             "-DTAIL_SPLIT=1 -DUNROLL_K=16 -DSTEP12_BR_LGKMCNT=2"),
    ("_ts_u16_v12_memc",         "-DTAIL_SPLIT=1 -DUNROLL_K=16 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("_ts_u8_lgk2_v12_memc",     "-DTAIL_SPLIT=1 -DUNROLL_K=8 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),

    # ── Group C: large-N (B-bound) gm × pf × ext_br crosses (NEW) ──────────────
    ("_gm8_lgk2_v12_memc",       "-DGROUP_SIZE_M=8 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("_ts_gm8_lgk2_v12_memc",    "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("_gm8_ext_br_lgk2",         "-DGROUP_SIZE_M=8 -DSTEP4_EXTERNAL_BR_PREFETCH=1 -DSTEP12_BR_LGKMCNT=2"),
    ("_gm8_ext_br_lgk2_memc",    "-DGROUP_SIZE_M=8 -DSTEP4_EXTERNAL_BR_PREFETCH=1 -DSTEP12_BR_LGKMCNT=2 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("_gm8_v12_lgk2_ext_br",     "-DGROUP_SIZE_M=8 -DSTEP3_BARRIER_VMCNT=12 -DSTEP12_BR_LGKMCNT=2 -DSTEP4_EXTERNAL_BR_PREFETCH=1"),
    ("_ts_gm4",                  "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=4"),
    ("_ts_gm4_lgk2_v12_memc",    "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=4 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("_gm4_v12",                 "-DGROUP_SIZE_M=4 -DSTEP3_BARRIER_VMCNT=12"),
    # pf6_6 was a R1 winner; cross with new things
    ("_pf6_6_lgk2_v12",          "-DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12"),
    ("_pf6_6_v12_memc",          "-DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("_ts_pf6_6_v12_memc",       "-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("_ts_pf6_6_u8",             "-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DUNROLL_K=8"),
    ("_ts_pf6_6_u16",            "-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DUNROLL_K=16"),

    # ── Group D: AGPR hint + waves_per_eu for register pressure (NEW) ──────────
    ("_agpr192_wpe2",            "-DAGPR_REGS_HINT_192=1 -DWAVES_PER_EU_2=1"),
    ("_agpr192",                 "-DAGPR_REGS_HINT_192=1"),
    ("_ts_lgk2_agpr192",         "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DAGPR_REGS_HINT_192=1"),

    # ── Group E: VMCNT=20/24 × UNROLL × LGK (R1 left some untested) ────────────
    ("_u8_v20",                  "-DUNROLL_K=8 -DSTEP3_BARRIER_VMCNT=20"),
    ("_u16_v20",                 "-DUNROLL_K=16 -DSTEP3_BARRIER_VMCNT=20"),
    ("_ts_lgk2_v20_u8",          "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=20 -DUNROLL_K=8"),
    ("_ts_v16_u8",               "-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=16 -DUNROLL_K=8"),
    ("_ts_v16_u16",              "-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=16 -DUNROLL_K=16"),
]

BASE = (
    f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    f"-I/opt/rocm/include/hip "
    f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    f"-shared -fPIC -std=c++20 -w "
    f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm"
)


def module_name_for_nk(n, k):
    return f"tk_mxfp4_gluon_cpp_n{n}_k{k}"


def build_one(n, k, suffix, flags):
    module_name = module_name_for_nk(n, k) + suffix
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if os.path.exists(so_path):
        return (n, k, suffix, "cached", 0.0)
    with open(KERNEL_SRC, "r") as f:
        src = f.read()
    patched = src.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},"
    )
    wrapper_src = os.path.join(BUILD_DIR, f"wrap_n{n}_k{k}{suffix}.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)
    cmd = (
        f"/opt/rocm/bin/hipcc {wrapper_src} {BASE} "
        f"-DK_DIM={k} -DN_DIM={n} {flags} -o {so_path}"
    )
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(so_path):
        return (n, k, suffix, f"FAIL: {r.stderr[-300:]}", dt)
    return (n, k, suffix, "OK", dt)


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    tasks = [(n, k, s, f) for (n, k) in NK_PAIRS for (s, f) in DEEP_LOSE_VARIANTS]
    print(f"Total builds: {len(tasks)}  ({len(DEEP_LOSE_VARIANTS)} variants × {len(NK_PAIRS)} nk-pairs)")
    workers = int(os.environ.get("BUILD_WORKERS", "16"))
    print(f"Workers: {workers}")
    t0 = time.time()
    done = 0; fail = 0; cached = 0; ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(build_one, n, k, s, f): (n, k, s) for (n, k, s, f) in tasks}
        for fut in as_completed(futures):
            n, k, suffix, status, dt = fut.result()
            done += 1
            if status == "cached":
                cached += 1
            elif status.startswith("FAIL"):
                fail += 1
                print(f"  [{done}/{len(tasks)}] FAIL n={n} k={k}{suffix}: {status[:200]}", flush=True)
            else:
                ok += 1
                if done % 30 == 0:
                    print(f"  [{done}/{len(tasks)}] OK   n={n} k={k}{suffix} ({dt:.1f}s)", flush=True)
    print(f"\nDone: {done} total, {ok} built, {cached} cached, {fail} failed in {time.time()-t0:.1f}s")
    return 1 if fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
