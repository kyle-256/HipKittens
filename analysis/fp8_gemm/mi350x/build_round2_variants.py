#!/usr/bin/env python3
"""Round-2 parallel build: Optimizer A WIN discoveries + memclause crosses.

These are confirmed WINs from spot_optA on near-threshold shapes that were
NOT in the 87-variant Round-1 bench. Build for build_all42/ shared cache.

Idempotent: skips already-built .so files.
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

NK_PAIRS = [
    (4096, 2048), (4096, 3072), (4096, 4096), (4096, 6144), (4096, 7168),
    (4096, 8192), (4096, 14336), (4096, 16384), (4096, 28672), (4096, 32768),
    (6144, 2048), (6144, 4096), (6144, 32768),
    (14336, 2048), (14336, 4096), (14336, 8192), (14336, 16384),
    (28672, 2048), (28672, 4096), (28672, 32768),
    (32768, 4096), (32768, 6144), (32768, 14336), (32768, 28672), (32768, 128256),
    (128256, 32768),
]

# Optimizer A WIN discoveries + memclause crosses
ROUND2_VARIANTS = [
    # Single-axis VMCNT=20 (new ceiling)
    ("_v20", "-DSTEP3_BARRIER_VMCNT=20"),
    ("_v24", "-DSTEP3_BARRIER_VMCNT=24"),
    ("_v32", "-DSTEP3_BARRIER_VMCNT=32"),
    ("_ts_v20", "-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=20"),
    ("_ts_v24", "-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=24"),
    # VMCNT × LGK crosses with v20 ceiling
    ("_ts_lgk2_v20", "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=20"),
    ("_ts_lgk2_v24", "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=24"),
    ("_ts_gm2_v20", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP3_BARRIER_VMCNT=20"),
    # TAIL_BARRIER_VMCNT × VMCNT crosses (the big winner family)
    ("_ts_no_embed_tv16", "-DTAIL_SPLIT=1 -DSTEP3_EMBED_BARRIER=0 -DTAIL_BARRIER_VMCNT=16"),
    ("_ts_no_embed_tv0", "-DTAIL_SPLIT=1 -DSTEP3_EMBED_BARRIER=0 -DTAIL_BARRIER_VMCNT=0"),
    ("_ts_lgk2_tv16", "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DTAIL_BARRIER_VMCNT=16"),
    ("_ts_lgk2_tv0", "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DTAIL_BARRIER_VMCNT=0"),
    ("_ts_v12_tv16", "-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=12 -DTAIL_BARRIER_VMCNT=16"),
    ("_ts_v12_tv0", "-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=12 -DTAIL_BARRIER_VMCNT=0"),
    ("_ts_v4_tv16", "-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=4 -DTAIL_BARRIER_VMCNT=16"),
    ("_ts_v4_tv0", "-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=4 -DTAIL_BARRIER_VMCNT=0"),
    ("_ts_pf4_tv16", "-DTAIL_SPLIT=1 -DSTEP3_PF_N=4 -DSTEP4_PF_N=4 -DTAIL_BARRIER_VMCNT=16"),
    # PF asymmetric (Optimizer A finding)
    ("_pf4_6", "-DSTEP3_PF_N=4 -DSTEP4_PF_N=6"),
    ("_pf6_4", "-DSTEP3_PF_N=6 -DSTEP4_PF_N=4"),
    ("_pf3_8", "-DSTEP3_PF_N=8 -DSTEP4_PF_N=2"),
    # no_embed × VMCNT crosses
    ("_no_embed_v4", "-DSTEP3_EMBED_BARRIER=0 -DSTEP3_BARRIER_VMCNT=4"),
    ("_no_embed_v16", "-DSTEP3_EMBED_BARRIER=0 -DSTEP3_BARRIER_VMCNT=16"),
    ("_no_embed_v20", "-DSTEP3_EMBED_BARRIER=0 -DSTEP3_BARRIER_VMCNT=20"),
    # MEMCLAUSE crossed with the new winners
    ("_ts_no_embed_tv16_memc", "-DTAIL_SPLIT=1 -DSTEP3_EMBED_BARRIER=0 -DTAIL_BARRIER_VMCNT=16 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("_ts_lgk2_v20_memc", "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=20 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("_ts_v12_tv0_memc", "-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=12 -DTAIL_BARRIER_VMCNT=0 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("_v20_memc", "-DSTEP3_BARRIER_VMCNT=20 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("_ts_v4_tv0_memc", "-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=4 -DTAIL_BARRIER_VMCNT=0 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
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
    tasks = [(n, k, s, f) for (n, k) in NK_PAIRS for (s, f) in ROUND2_VARIANTS]
    print(f"Total builds: {len(tasks)}  ({len(ROUND2_VARIANTS)} variants × {len(NK_PAIRS)} nk-pairs)")
    workers = int(os.environ.get("BUILD_WORKERS", "12"))
    print(f"Workers: {workers}")
    t0 = time.time()
    done = 0; fail = 0; cached = 0
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
                print(f"  [{done}/{len(tasks)}] OK   n={n} k={k}{suffix} ({dt:.1f}s)", flush=True)
    print(f"\nDone: {done} total, {cached} cached, {fail} failed in {time.time()-t0:.1f}s")
    return 1 if fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
