#!/usr/bin/env python3
"""Build only the 4 R25-H wired variants for their target (N,K) pairs.

The variants were already verified in build_round25_optH.py with shape-specific
suffixes (_r25h_gm7_pfoffN_KX_MX_NX). Now we build them under the names the
auto-tuner in bench_all_42.py expects (_ts_..._kxK_btw_all).

Target builds (4 .so files):
  (N=28672, K=2048)  → _ts_v12_gm7_memc_pfoff4_kx2048_btw_all
  (N=32768, K=6144)  → _ts_v12_gm7_memc_pfoff19_kx6144_btw_all
  (N=4096,  K=7168)  → _ts_v12_tv0_gm7_memc_pfoff24_kx7168_btw_all
  (N=14336, K=8192)  → _ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from bench_all_42 import build_for_nk

BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
os.makedirs(BUILD_DIR, exist_ok=True)

JOBS = [
    # (N, K, suffix, cppflags)
    (28672, 2048,
     "_ts_v12_gm7_memc_pfoff4_kx2048_btw_all",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=7 -DSTEP3_BARRIER_VMCNT=12 "
     "-DR25C_TAIL_PF_OFF_ITERS=4 -DR25C_K_LIMIT=32768 -DR25C_K_EXACT=2048 "
     "-mllvm -amdgpu-sched-strategy=max-memory-clause "
     "-DBARRIER_TO_WAITCNT_ALL=1"),
    (32768, 6144,
     "_ts_v12_gm7_memc_pfoff19_kx6144_btw_all",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=7 -DSTEP3_BARRIER_VMCNT=12 "
     "-DR25C_TAIL_PF_OFF_ITERS=19 -DR25C_K_LIMIT=32768 -DR25C_K_EXACT=6144 "
     "-mllvm -amdgpu-sched-strategy=max-memory-clause "
     "-DBARRIER_TO_WAITCNT_ALL=1"),
    (4096, 7168,
     "_ts_v12_tv0_gm7_memc_pfoff24_kx7168_btw_all",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=7 -DSTEP3_BARRIER_VMCNT=12 -DTAIL_BARRIER_VMCNT=0 "
     "-DR25C_TAIL_PF_OFF_ITERS=24 -DR25C_K_LIMIT=32768 -DR25C_K_EXACT=7168 "
     "-mllvm -amdgpu-sched-strategy=max-memory-clause "
     "-DBARRIER_TO_WAITCNT_ALL=1"),
    (14336, 8192,
     "_ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=7 -DSTEP3_BARRIER_VMCNT=12 -DTAIL_BARRIER_VMCNT=0 "
     "-DR25C_TAIL_PF_OFF_ITERS=28 -DR25C_K_LIMIT=32768 -DR25C_K_EXACT=8192 "
     "-mllvm -amdgpu-sched-strategy=max-memory-clause "
     "-DBARRIER_TO_WAITCNT_ALL=1"),
]


def main():
    print(f"R25-H wired-variant builds: {len(JOBS)}")
    print("=" * 100)
    for n, k, suffix, flags in JOBS:
        so = build_for_nk(n, k, BUILD_DIR, extra_cppflags=flags, suffix=suffix)
        if so is None:
            print(f"FATAL: build failed for N={n} K={k} suffix={suffix}")
            return 1
    print("\nAll R25-H wired builds OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
