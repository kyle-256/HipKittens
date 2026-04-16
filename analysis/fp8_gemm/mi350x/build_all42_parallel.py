#!/usr/bin/env python3
"""Parallel builder for all MXFP4 GEMM .so files.

Builds all 26 unique (N,K) pairs x 41 variants = 1066 .so files
using ThreadPoolExecutor with 16 workers.

Skips already-built .so files.
"""

import os
import subprocess
import sys
import sysconfig
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")
TK_ROOT = os.environ.get(
    "THUNDERKITTENS_ROOT",
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
)
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"

# All 42 shapes: (M, N, K, competitor_tflops)
ALL_SHAPES = [
    (16384,  4096,  2048, 2995.0),
    (16384,  4096,  3072, 3492.3),
    (16384,  6144,  2048, 3047.6),
    (32768,  4096,  2048, 3131.8),
    (32768,  4096,  3072, 3630.6),
    (32768,  6144,  2048, 3239.9),
    (16384, 14336,  2048, 3301.3),
    (16384, 28672,  2048, 3482.3),
    (32768, 14336,  2048, 3351.4),
    (32768, 28672,  2048, 3353.4),
    (4096,   4096,  16384, 4642.1),
    (4096,  14336,  16384, 5013.0),
    (6144,   4096,  16384, 4428.1),
    (4096,   4096,   8192, 3959.9),
    (4096,   4096,  32768, 5152.8),
    (4096,   6144,  32768, 3784.2),
    (4096,  14336,   8192, 4345.8),
    (4096,  28672,  32768, 5649.9),
    (4096,  32768,   4096, 4166.5),
    (4096,  32768,   6144, 4548.6),
    (4096,  32768,  14336, 5296.1),
    (4096,  32768,  28672, 5568.2),
    (4096,  32768, 128256, 5781.1),
    (4096, 128256,  32768, 3195.3),
    (6144,   4096,   8192, 3822.0),
    (6144,  32768,   4096, 4291.0),
    (14336,  4096,  32768, 5245.4),
    (14336, 32768,   4096, 4462.6),
    (16384,  4096,   4096, 3951.8),
    (16384,  4096,   6144, 4259.9),
    (16384,  4096,   7168, 4443.2),
    (16384,  4096,  14336, 5142.1),
    (16384,  4096,  28672, 5525.3),
    (16384,  6144,   4096, 4042.5),
    (16384, 14336,   4096, 4255.8),
    (16384, 28672,   4096, 4411.7),
    (28672,  4096,   8192, 4810.0),
    (28672,  4096,  16384, 5350.6),
    (28672, 32768,   4096, 4466.6),
    (32768,  4096,   7168, 4666.8),
    (32768,  4096,  14336, 5223.4),
    (128256, 32768,  4096, 4536.4),
]

VARIANTS = [
    ("", ""),
    ("_gm2", "-DGROUP_SIZE_M=2"),
    ("_u16", "-DUNROLL_K=16"),
    ("_gm2u16", "-DGROUP_SIZE_M=2 -DUNROLL_K=16"),
    ("_gm1", "-DGROUP_SIZE_M=1"),
    ("_gm8", "-DGROUP_SIZE_M=8"),
    ("_u8", "-DUNROLL_K=8"),
    ("_u32", "-DUNROLL_K=32"),
    ("_gm8u16", "-DGROUP_SIZE_M=8 -DUNROLL_K=16"),
    ("_gm16", "-DGROUP_SIZE_M=16"),
    ("_gm8u8", "-DGROUP_SIZE_M=8 -DUNROLL_K=8"),
    ("_gm16u16", "-DGROUP_SIZE_M=16 -DUNROLL_K=16"),
    ("_swap", "-DSWAP_STEP34_MAIN=1 -DSWAP_STEP12_MAIN=1"),
    ("_swap_gm8", "-DSWAP_STEP34_MAIN=1 -DSWAP_STEP12_MAIN=1 -DGROUP_SIZE_M=8"),
    ("_ts", "-DTAIL_SPLIT=1"),
    ("_ts_gm8", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8"),
    ("_ts_u16", "-DTAIL_SPLIT=1 -DUNROLL_K=16"),
    ("_gm2u8", "-DGROUP_SIZE_M=2 -DUNROLL_K=8"),
    ("_gm16u8", "-DGROUP_SIZE_M=16 -DUNROLL_K=8"),
    ("_gm1u16", "-DGROUP_SIZE_M=1 -DUNROLL_K=16"),
    ("_ts_v12", "-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=12"),
    ("_ts_gm8_v12", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DSTEP3_BARRIER_VMCNT=12"),
    ("_ts_gm2", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2"),
    ("_ts_gm2u8", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DUNROLL_K=8"),
    ("_spread_gm2u8", "-DSPREAD_LDS=1 -DGROUP_SIZE_M=2 -DUNROLL_K=8"),
    ("_spread_gm2u8_v12", "-DSPREAD_LDS=1 -DGROUP_SIZE_M=2 -DUNROLL_K=8 -DSTEP3_BARRIER_VMCNT=12"),
    ("_spread_gm8", "-DSPREAD_LDS=1 -DGROUP_SIZE_M=8"),
    ("_v12", "-DSTEP3_BARRIER_VMCNT=12"),
    ("_gm8_v12", "-DGROUP_SIZE_M=8 -DSTEP3_BARRIER_VMCNT=12"),
    ("_no_nvs", "-DNONVOLATILE_SCALE_X2_POC=0"),
    ("_pf4", "-DSTEP3_PF_N=4 -DSTEP4_PF_N=4"),
    ("_ts_pf4", "-DTAIL_SPLIT=1 -DSTEP3_PF_N=4 -DSTEP4_PF_N=4"),
    ("_gm2_v12", "-DGROUP_SIZE_M=2 -DSTEP3_BARRIER_VMCNT=12"),
    ("_ext_br", "-DSTEP4_EXTERNAL_BR_PREFETCH=1"),
    ("_ts_ext_br", "-DTAIL_SPLIT=1 -DSTEP4_EXTERNAL_BR_PREFETCH=1"),
    ("_gm8_ext_br", "-DGROUP_SIZE_M=8 -DSTEP4_EXTERNAL_BR_PREFETCH=1"),
    ("_ts_gm8_ext_br", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DSTEP4_EXTERNAL_BR_PREFETCH=1"),
    ("_gm32", "-DGROUP_SIZE_M=32"),
    ("_gm64", "-DGROUP_SIZE_M=64"),
    ("_ts_gm16", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=16"),
    ("_ts_gm32", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=32"),
    # New variants: EXT_BR×V12, GM2×EXT_BR, lgkmcnt, VMCNT tuning
    ("_ext_br_v12", "-DSTEP4_EXTERNAL_BR_PREFETCH=1 -DSTEP3_BARRIER_VMCNT=12"),
    ("_ts_ext_br_v12", "-DTAIL_SPLIT=1 -DSTEP4_EXTERNAL_BR_PREFETCH=1 -DSTEP3_BARRIER_VMCNT=12"),
    ("_gm2_ext_br", "-DGROUP_SIZE_M=2 -DSTEP4_EXTERNAL_BR_PREFETCH=1"),
    ("_ts_gm2_ext_br", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP4_EXTERNAL_BR_PREFETCH=1"),
    ("_gm16_v12", "-DGROUP_SIZE_M=16 -DSTEP3_BARRIER_VMCNT=12"),
    ("_gm1_v12", "-DGROUP_SIZE_M=1 -DSTEP3_BARRIER_VMCNT=12"),
    ("_ts_gm2_v12", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP3_BARRIER_VMCNT=12"),
    ("_gm2_ext_br_v12", "-DGROUP_SIZE_M=2 -DSTEP4_EXTERNAL_BR_PREFETCH=1 -DSTEP3_BARRIER_VMCNT=12"),
    ("_ts_gm2_ext_br_v12", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP4_EXTERNAL_BR_PREFETCH=1 -DSTEP3_BARRIER_VMCNT=12"),
    ("_lgk2", "-DSTEP12_BR_LGKMCNT=2"),
    ("_lgk4", "-DSTEP12_BR_LGKMCNT=4"),
    ("_ts_lgk2", "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2"),
    ("_ts_lgk4", "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=4"),
    ("_ts_lgk2_v12", "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12"),
    # LGKMCNT × VMCNT cross-products
    ("_lgk2_v4", "-DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=4"),
    ("_lgk2_v12", "-DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12"),
    ("_lgk2_v16", "-DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=16"),
    ("_lgk4_v4", "-DSTEP12_BR_LGKMCNT=4 -DSTEP3_BARRIER_VMCNT=4"),
    ("_lgk4_v12", "-DSTEP12_BR_LGKMCNT=4 -DSTEP3_BARRIER_VMCNT=12"),
    ("_lgk4_v16", "-DSTEP12_BR_LGKMCNT=4 -DSTEP3_BARRIER_VMCNT=16"),
    ("_ts_lgk2_v4", "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=4"),
    ("_ts_lgk2_v16", "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=16"),
    ("_ts_lgk4_v4", "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=4 -DSTEP3_BARRIER_VMCNT=4"),
    ("_ts_lgk4_v12", "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=4 -DSTEP3_BARRIER_VMCNT=12"),
    ("_ts_lgk4_v16", "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=4 -DSTEP3_BARRIER_VMCNT=16"),
    # LGKMCNT × NO_EMBED crosses
    ("_lgk2_no_embed", "-DSTEP12_BR_LGKMCNT=2 -DSTEP3_EMBED_BARRIER=0"),
    ("_ts_lgk2_no_embed", "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_EMBED_BARRIER=0"),
    ("_ts_lgk2_no_embed_v12", "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_EMBED_BARRIER=0 -DSTEP3_BARRIER_VMCNT=12"),
    ("_v4", "-DSTEP3_BARRIER_VMCNT=4"),
    ("_ts_v4", "-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=4"),
    ("_v16", "-DSTEP3_BARRIER_VMCNT=16"),
    ("_ts_v16", "-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=16"),
    ("_no_embed", "-DSTEP3_EMBED_BARRIER=0"),
    ("_ts_no_embed", "-DTAIL_SPLIT=1 -DSTEP3_EMBED_BARRIER=0"),
    ("_ts_no_embed_v12", "-DTAIL_SPLIT=1 -DSTEP3_EMBED_BARRIER=0 -DSTEP3_BARRIER_VMCNT=12"),
    # TAIL_BARRIER_VMCNT variants (separate tail iteration barrier VMCNT)
    ("_ts_tv16", "-DTAIL_SPLIT=1 -DTAIL_BARRIER_VMCNT=16"),
    ("_ts_tv0", "-DTAIL_SPLIT=1 -DTAIL_BARRIER_VMCNT=0"),
    # GM × LGK cross-products (tile scheduling × LDS wait)
    ("_gm8_lgk2", "-DGROUP_SIZE_M=8 -DSTEP12_BR_LGKMCNT=2"),
    ("_ts_gm2_lgk2", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP12_BR_LGKMCNT=2"),
    ("_ts_gm2_lgk2_v12", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12"),
    # FUSED_STEP34 variants
    ("_f34", "-DFUSED_STEP34=1"),
    ("_f34_ts", "-DFUSED_STEP34=1 -DTAIL_SPLIT=1"),
    ("_f34_lgk2", "-DFUSED_STEP34=1 -DSTEP12_BR_LGKMCNT=2"),
    ("_f34_ts_lgk2", "-DFUSED_STEP34=1 -DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2"),
    ("_f34_v12", "-DFUSED_STEP34=1 -DSTEP3_BARRIER_VMCNT=12"),
    ("_f34_v4", "-DFUSED_STEP34=1 -DSTEP3_BARRIER_VMCNT=4"),
    ("_f34_ts_v12", "-DFUSED_STEP34=1 -DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=12"),
    ("_f34_gm2", "-DFUSED_STEP34=1 -DGROUP_SIZE_M=2"),
    ("_f34_gm8", "-DFUSED_STEP34=1 -DGROUP_SIZE_M=8"),
    ("_f34_ts_gm2_v12", "-DFUSED_STEP34=1 -DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP3_BARRIER_VMCNT=12"),
    ("_f34_ts_lgk2_v12", "-DFUSED_STEP34=1 -DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12"),
    ("_f34_u8", "-DFUSED_STEP34=1 -DUNROLL_K=8"),
    ("_f34_u16", "-DFUSED_STEP34=1 -DUNROLL_K=16"),
]


def get_unique_nk_pairs():
    return sorted(set((n, k) for _, n, k, _ in ALL_SHAPES))


def build_one(n_dim, k_dim, suffix, extra_cppflags, build_dir, kernel_src_text):
    """Compile a single (N, K, variant) combination. Returns (key, success, elapsed)."""
    module_name = f"tk_mxfp4_gluon_cpp_n{n_dim}_k{k_dim}{suffix}"
    out_file = f"{module_name}{EXT_SUFFIX}"
    out_path = os.path.join(build_dir, out_file)
    key = f"N={n_dim},K={k_dim}{suffix}"

    # Skip if already built
    if os.path.exists(out_path):
        return (key, "cached", 0.0)

    # Create patched wrapper source
    patched = kernel_src_text.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},"
    )
    wrapper_src = os.path.join(build_dir, f"wrap_n{n_dim}_k{k_dim}{suffix}.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)

    t0 = time.time()

    env = os.environ.copy()
    env["THUNDERKITTENS_ROOT"] = TK_ROOT

    cmd = (
        f'make -C {SCRIPT_DIR} TARGET={os.path.join(build_dir, module_name)} '
        f'SRC={wrapper_src} '
        f'CPPFLAGS="-DK_DIM={k_dim} -DN_DIM={n_dim} {extra_cppflags}"'
    )
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, env=env
    )
    elapsed = time.time() - t0

    if result.returncode != 0 or not os.path.exists(out_path):
        return (key, "FAILED", elapsed, result.stderr[-500:])

    return (key, "ok", elapsed)


def main():
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16

    build_dir = os.path.join(SCRIPT_DIR, "build_all42")
    os.makedirs(build_dir, exist_ok=True)

    nk_pairs = get_unique_nk_pairs()
    total = len(nk_pairs) * len(VARIANTS)

    print(f"MXFP4 Parallel Builder")
    print(f"  Unique (N,K) pairs: {len(nk_pairs)}")
    print(f"  Variants: {len(VARIANTS)}")
    print(f"  Total compilations: {total}")
    print(f"  Max workers: {max_workers}")
    print(f"  Build dir: {build_dir}")
    print(f"  TK_ROOT: {TK_ROOT}")
    print()

    # Read kernel source once
    with open(KERNEL_SRC, "r") as f:
        kernel_src_text = f.read()

    # Build task list
    tasks = []
    for n_val, k_val in nk_pairs:
        for suffix, cppflags in VARIANTS:
            tasks.append((n_val, k_val, suffix, cppflags))

    t_start = time.time()
    completed = 0
    cached = 0
    failed = 0
    built = 0
    failures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for n_val, k_val, suffix, cppflags in tasks:
            fut = executor.submit(
                build_one, n_val, k_val, suffix, cppflags,
                build_dir, kernel_src_text
            )
            futures[fut] = (n_val, k_val, suffix)

        for fut in as_completed(futures):
            result = fut.result()
            completed += 1
            key = result[0]
            status = result[1]

            if status == "cached":
                cached += 1
            elif status == "ok":
                built += 1
                elapsed = result[2]
                if completed % 50 == 0 or completed == total:
                    elapsed_total = time.time() - t_start
                    rate = completed / elapsed_total
                    eta = (total - completed) / rate if rate > 0 else 0
                    print(f"  [{completed:>4}/{total}] built={built} cached={cached} "
                          f"failed={failed} | {elapsed_total:.0f}s elapsed, "
                          f"ETA {eta:.0f}s | last: {key} ({elapsed:.1f}s)")
            else:
                failed += 1
                err_msg = result[3] if len(result) > 3 else ""
                failures.append((key, err_msg))
                if completed % 50 == 0 or failed <= 5:
                    print(f"  [{completed:>4}/{total}] FAILED: {key}")

    elapsed_total = time.time() - t_start
    print()
    print(f"=" * 60)
    print(f"Build complete in {elapsed_total:.1f}s")
    print(f"  Total:  {total}")
    print(f"  Built:  {built}")
    print(f"  Cached: {cached}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {(built + cached) / total * 100:.1f}%")

    if failures:
        print(f"\nFailed compilations ({len(failures)}):")
        for key, err in failures[:20]:
            print(f"  {key}")
            if err:
                # Show last 2 lines of error
                lines = err.strip().split("\n")
                for line in lines[-2:]:
                    print(f"    {line}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")

    # Count actual .so files in build dir
    so_count = sum(1 for f in os.listdir(build_dir) if f.endswith(EXT_SUFFIX))
    print(f"\nTotal .so files in {build_dir}: {so_count}")
    print(f"=" * 60)


if __name__ == "__main__":
    main()
