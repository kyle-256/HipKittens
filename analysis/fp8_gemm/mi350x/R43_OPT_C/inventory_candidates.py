#!/usr/bin/env python3
"""Inventory existing .so candidates for the 14 VC_CLAWBACK shapes.

Kernel is keyed by N,K only (M dimension doesn't change the .so).
For each target shape, find all .so files in build_R40A/R40B/R41A/R41B
that match n{N}_k{K}_*.
"""
import os, json, re, glob

SCRIPT_DIR = "/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x"
BUILD_DIRS = ["build_R40A", "build_R40B", "build_R41A", "build_R41B"]

VC_CLAWBACK_BASELINE = {
    # shape: (current so_path, current p50, comp, pct)
    "14336x4096x32768": ("build_R41A/tk_mxfp4_gluon_cpp_n4096_k32768_ts_v12_tv0_dc_gm7_pfoff120_kx32768_btw_all_R41A_po0_ef1.cpython-310-x86_64-linux-gnu.so", 3173.0, 5245.4, 60.49),
    "4096x28672x32768": ("build_R41A/tk_mxfp4_gluon_cpp_n28672_k32768_ts_v12_tv0_dc_gm7_pfoff120_kx32768_btw_all_R41A_po32_ef1.cpython-310-x86_64-linux-gnu.so", 3498.1, 5649.9, 61.91),
    "4096x32768x128256": ("build_R40A/tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_v12_memc_btw_all_R40A_PF_FENCE1_R40A.cpython-310-x86_64-linux-gnu.so", 4247.2, 5781.1, 73.47),
    "4096x4096x32768": ("build_R41A/tk_mxfp4_gluon_cpp_n4096_k32768_ts_v12_tv0_dc_gm7_pfoff120_kx32768_btw_all_R41A_po32_ef1.cpython-310-x86_64-linux-gnu.so", 3988.6, 5152.8, 77.41),
    "4096x6144x32768": ("build_R41A/tk_mxfp4_gluon_cpp_n6144_k32768_ts_v12_tv0_dc_gm7_pfoff120_kx32768_btw_all_R41A_po8_ef1.cpython-310-x86_64-linux-gnu.so", 3116.5, 3784.2, 82.36),
    "6144x4096x16384": ("build_R40B/tk_mxfp4_gluon_cpp_n4096_k16384_ts_lgk2_gm7_pfoff56_kx16384_btw_all_R40B_safe.cpython-310-x86_64-linux-gnu.so", 3718.0, 4428.1, 83.96),
    "4096x14336x16384": ("build_R40B/tk_mxfp4_gluon_cpp_n14336_k16384_ts_lgk2_gm7_pfoff56_kx16384_btw_all_R40B_safe.cpython-310-x86_64-linux-gnu.so", 4231.7, 5013.0, 84.41),
    "14336x32768x4096": ("build_R40B/tk_mxfp4_gluon_cpp_n32768_k4096_ts_lgk2_gm7_v12_pfoff14_R40B_safe.cpython-310-x86_64-linux-gnu.so", 3850.2, 4462.6, 86.28),
    "16384x4096x14336": ("build_R41B/tk_mxfp4_gluon_cpp_n4096_k14336_ts_gm8_v12_btw_all_R41B_v0b.cpython-310-x86_64-linux-gnu.so", 4444.7, 5142.1, 86.44),
    "28672x32768x4096": ("build_R40B/tk_mxfp4_gluon_cpp_n32768_k4096_ts_gm7_v12_dc_pfoff14_R40B_safe.cpython-310-x86_64-linux-gnu.so", 3972.9, 4466.6, 88.95),
    "6144x4096x8192": ("build_R40B/tk_mxfp4_gluon_cpp_n4096_k8192_ts_v12_tv0_gm7_pfoff28_kx8192_btw_all_R40B_safe.cpython-310-x86_64-linux-gnu.so", 3405.0, 3822.0, 89.09),
    "4096x14336x8192": ("build_R40B/tk_mxfp4_gluon_cpp_n14336_k8192_ts_v12_tv0_gm7_pfoff28_kx8192_btw_all_R40B_safe.cpython-310-x86_64-linux-gnu.so", 3993.0, 4345.8, 91.88),
    "4096x32768x4096": ("build_R40B/tk_mxfp4_gluon_cpp_n32768_k4096_lgk2_v16_R40B_safe.cpython-310-x86_64-linux-gnu.so", 3821.9, 4166.5, 91.73),
    "6144x32768x4096": ("build_R40B/tk_mxfp4_gluon_cpp_n32768_k4096_ts_lgk2_v24_R40B_safe.cpython-310-x86_64-linux-gnu.so", 4059.3, 4291.0, 94.60),
}

def find_candidates(N, K):
    """Find all .so files matching n{N}_k{K}_*"""
    candidates = []
    for bd in BUILD_DIRS:
        pattern = os.path.join(SCRIPT_DIR, bd, f"tk_mxfp4_gluon_cpp_n{N}_k{K}_*.so")
        for so in glob.glob(pattern):
            candidates.append(so)
    return sorted(candidates)

def main():
    inventory = {}
    for shape, (cur_so, cur_tflops, comp, pct) in VC_CLAWBACK_BASELINE.items():
        m, n, k = shape.split("x")
        N, K = int(n), int(k)
        cands = find_candidates(N, K)
        cur_full = os.path.join(SCRIPT_DIR, cur_so)
        inventory[shape] = {
            "M": int(m), "N": N, "K": K,
            "current_so": cur_full,
            "current_tflops_p50": cur_tflops,
            "comp_tflops": comp,
            "current_pct_comp": pct,
            "n_candidates": len(cands),
            "candidates": cands,
        }
    out = os.path.join(SCRIPT_DIR, "R43_OPT_C", "candidate_inventory.json")
    with open(out, "w") as f:
        json.dump(inventory, f, indent=2)
    print(f"Wrote {out}")
    for shape, info in inventory.items():
        print(f"{shape}: {info['n_candidates']} candidates")

if __name__ == "__main__":
    main()
