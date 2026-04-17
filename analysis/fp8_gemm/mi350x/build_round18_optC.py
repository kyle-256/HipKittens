#!/usr/bin/env python3
"""Round 18 Optimizer C — R17A-P2 attempt for DLA1 (4096x32768x128256).

Original P2 proposal: M-tile 128 → 256 (effective per-CTA tile BLK 256 → BLK_M 512)
to halve CTA count and amortize per-CTA sync overhead.

Analysis (see round18_optC_verdict.md):
  - Cleanly implementing BLOCK_M_512 requires duplicating all A0/A1 code paths
    (prologue/main loop STEPS 1-4 expand to STEPS 1-8) plus 4× accumulator state,
    8× scale loads. Multi-day refactor of 2647-line kernel.
  - LDS budget: doubling A-tiles pushes total LDS to ~192 KB (8 tiles × 16 KB →
    16 tiles × 16 KB) which exceeds 160 KB / CU on MI355X. Mitigation requires
    dropping double-buffering on A which negates the latency-hide gain.

Pragmatic substitute: probe the unexplored PERSISTENT_BATCH values (2, 4, 8).
R14C tested batch={16, 32} which BROKE (memory access fault); batch={1, 2, 4, 8}
have not been benched on DLA1 specifically.

PERSISTENT_BATCH groups N sequential tiles per CTA-claim; lower per-WG launch
overhead and slightly better L2 reuse for adjacent tiles. Doesn't increase per-CTA
M-tile but does reduce total kernel launches → partial proxy for "halve CTA count"
goal of P2.
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

PARENT_NONSCHED = (
    "-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 "
    "-DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12"
)
PARENT_SCHED = "-mllvm -amdgpu-sched-strategy=max-memory-clause"

# (suffix, extra_flags)
VARIANTS = [
    # PERSISTENT_BATCH unexplored values (R14C only tried 16, 32 which BROKE)
    ("_r18c_pb1",  "-DPERSISTENT_XCD=1 -DPERSISTENT_BATCH=1"),
    ("_r18c_pb2",  "-DPERSISTENT_XCD=1 -DPERSISTENT_BATCH=2"),
    ("_r18c_pb4",  "-DPERSISTENT_XCD=1 -DPERSISTENT_BATCH=4"),
    ("_r18c_pb8",  "-DPERSISTENT_XCD=1 -DPERSISTENT_BATCH=8"),
    # Combo with larger PERSISTENT_GRID (1216 = 2× default)
    ("_r18c_pb1_g1216",  "-DPERSISTENT_XCD=1 -DPERSISTENT_BATCH=1 -DPERSISTENT_GRID=1216"),
    ("_r18c_pb2_g1216",  "-DPERSISTENT_XCD=1 -DPERSISTENT_BATCH=2 -DPERSISTENT_GRID=1216"),
    # Persistent + STATIC remap
    ("_r18c_pb1_static",  "-DPERSISTENT_XCD=1 -DPERSISTENT_BATCH=1 -DSTATIC_XCD_REMAP=1"),
]

BASE = (
    f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    f"-I/opt/rocm/include/hip "
    f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    f"-shared -fPIC -std=c++20 -w "
    f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm"
)


def build_one(suffix, extra_flags, n=N, k=K, force=False):
    full_suffix = PARENT_SUFFIX + suffix
    module_name = f"tk_mxfp4_gluon_cpp_n{n}_k{k}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if force and os.path.exists(so_path):
        os.remove(so_path)
    if os.path.exists(so_path):
        return (full_suffix, n, k, "cached", 0.0, "")
    with open(KERNEL_SRC, "r") as f:
        src = f.read()
    patched = src.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},"
    )
    wrapper_src = os.path.join(BUILD_DIR, f"wrap_n{n}_k{k}{full_suffix}.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)
    cmd = (
        f"/opt/rocm/bin/hipcc {wrapper_src} {BASE} "
        f"-DK_DIM={k} -DN_DIM={n} {PARENT_NONSCHED} {PARENT_SCHED} {extra_flags} -o {so_path}"
    )
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(so_path):
        return (full_suffix, n, k, "FAIL", dt, r.stderr[-400:])
    return (full_suffix, n, k, "OK", dt, "")


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    # build for DLA1 (target) + DLA2 (deep-LOSE) + correctness shape (1024,4096)
    nk_pairs = [
        (32768, 128256),  # DLA1 target
        (1024, 4096),     # SNR check small
    ]
    tasks = [(s, f, n, k) for (s, f) in VARIANTS for (n, k) in nk_pairs]
    print(f"Total builds: {len(tasks)}")
    workers = int(os.environ.get("BUILD_WORKERS", "8"))
    print(f"Workers: {workers}")

    t0 = time.time()
    done = 0; fail = 0; cached = 0; ok = 0
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(build_one, s, f, n, k) for (s, f, n, k) in tasks]
        for fut in as_completed(futs):
            sfx, n, k, status, dt, err = fut.result()
            done += 1
            if status == "OK":
                ok += 1
                tag = "OK"
            elif status == "cached":
                cached += 1
                tag = "cached"
            else:
                fail += 1
                tag = "FAIL"
                failures.append((sfx, n, k, err))
            print(f"[{done}/{len(tasks)}] {tag:6s} n={n:6d} k={k:6d} {sfx}  ({dt:.1f}s)")
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s: ok={ok} cached={cached} fail={fail}")
    if failures:
        print("\n=== FAILURES ===")
        for sfx, n, k, err in failures:
            print(f"\n[{sfx}] n={n} k={k}\n{err}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
