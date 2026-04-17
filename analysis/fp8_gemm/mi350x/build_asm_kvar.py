#!/usr/bin/env python3
"""Build K-specialized variants of the gluon-derived ASM inline kernel.

Patches `s_cmp_lt_u32 s68, 28` with the appropriate K-loop terminator
for each target K, then compiles a separate .so per K.

K_per_iter = 512 elements (empirically: 16x16x128 MFMA × 4 op_sel positions).
trips = K/512 - 1     (since 1 prologue + 1 epilogue = 2 segments outside the loop)
T     = -2 + trips*2  (the constant in s_cmp_lt_u32)
"""
import os, sys, sysconfig, subprocess, time, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_asm_kvar")
HEADER_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_asm_inline.h")
KVAR_CPP = os.path.join(SCRIPT_DIR, "kernel_mxfp4_asm_inline_kvar.cpp")

# K → (terminator, trips) mapping
def kspec(K):
    """Returns (terminator_T, n_trips) for the K-loop. K must be divisible by 512."""
    assert K % 512 == 0, f"K={K} not divisible by 512 (K_per_iter)"
    segs = K // 512
    trips = segs - 1   # 1 prologue + 1 epilogue ≈ 1 outside trip
    T = -2 + trips * 2
    return T, trips

# Target K values from deep-LOSE shapes (only those divisible by 512)
TARGET_K = [
    8192,   # baseline (sanity)
    14336,  # T=52
    16384,  # T=60
    28672,  # T=108
    32768,  # T=124
    # 128256: NOT divisible by 512, skip
]

BASE = (
    f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    f"-I/opt/rocm/include/hip "
    f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    f"-shared -fPIC -std=c++20 -w "
    f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm"
)


def patch_header(K, T):
    """Read original header, patch s_cmp_lt_u32 s68, 28 → s68, T, write per-K copy."""
    with open(HEADER_SRC, "r") as f:
        src = f.read()
    needle = '"s_cmp_lt_u32 s68, 28\\n"'
    if needle not in src:
        raise RuntimeError(f"Original loop terminator not found in {HEADER_SRC}")
    new_src = src.replace(needle, f'"s_cmp_lt_u32 s68, {T}\\n"')
    out_h = os.path.join(BUILD_DIR, f"kernel_mxfp4_asm_inline_k{K}.h")
    with open(out_h, "w") as f:
        f.write(new_src)
    return out_h


def patch_cpp(K, header_path):
    """Write a .cpp that #include's the K-specific header and uses unique module name."""
    with open(KVAR_CPP, "r") as f:
        src = f.read()
    # Replace the include
    src = src.replace(
        '#include "kernel_mxfp4_asm_inline.h"',
        f'#include "{header_path}"'
    )
    out_cpp = os.path.join(BUILD_DIR, f"wrap_k{K}.cpp")
    with open(out_cpp, "w") as f:
        f.write(src)
    return out_cpp


def build_one(K):
    T, trips = kspec(K)
    h = patch_header(K, T)
    cpp = patch_cpp(K, h)
    mod_name = f"tk_mxfp4_asm_inline_k{K}"
    so_path = os.path.join(BUILD_DIR, f"{mod_name}{EXT_SUFFIX}")
    if os.path.exists(so_path):
        return (K, T, trips, "cached", 0.0, so_path)
    cmd = (
        f"/opt/rocm/bin/hipcc {cpp} {BASE} "
        f"-DK_TRIPS={T} -DMOD_NAME={mod_name} -o {so_path}"
    )
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(so_path):
        return (K, T, trips, f"FAIL: {r.stderr[-500:]}", dt, None)
    return (K, T, trips, "OK", dt, so_path)


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    print(f"Building {len(TARGET_K)} K-specialized variants...")
    for K in TARGET_K:
        T, trips = kspec(K)
        print(f"  K={K:>6}  trips={trips:>3}  T={T:>4}")
    workers = int(os.environ.get("BUILD_WORKERS", "5"))
    t0 = time.time()
    fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(build_one, K): K for K in TARGET_K}
        for fut in as_completed(futures):
            K, T, trips, status, dt, so = fut.result()
            if status.startswith("FAIL"):
                fail += 1
                print(f"  K={K} T={T}: {status[:300]}", flush=True)
            else:
                print(f"  K={K} T={T} trips={trips}: {status} ({dt:.1f}s)  {so}", flush=True)
    print(f"\nDone in {time.time()-t0:.1f}s ; fails={fail}")
    return 1 if fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
