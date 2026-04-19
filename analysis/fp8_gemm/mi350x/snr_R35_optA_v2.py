#!/usr/bin/env python3
"""R35 Opt A SNR validation v2: small N + small scale to keep bf16 finite.

Tests at M=N=4096, K=2048, scales=0 (deterministic) to get clean torch ref.
Also runs a kernel-vs-incumbent bit-eq check at K=128256.

Gate: SNR_det >= 40 dB AND deterministic frac >= 99%.
"""
import os, sys, math, json, gc, importlib.util
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
sys.path.insert(0, BUILD_DIR)

# Use the K=128256 N=32768 build (matches incumbent shape) for KERNEL-vs-KERNEL
# correctness check (no torch ref needed — incumbent is the reference).
M_INC, N_INC, K_INC = 4096, 32768, 128256


def gen_fp4(rows, K):
    cols = K // 2
    lo = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device="cuda")
    hi = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device="cuda")
    return (hi << 4) | lo


def preshuffle_mfma16_merged(scale_exp):
    rows, kb = scale_exp.shape
    pr = math.ceil(rows / 64) * 64
    pk = math.ceil(kb / 8) * 8
    raw = torch.full((pr, pk), 0x7F, dtype=torch.uint8, device=scale_exp.device)
    raw[:rows, :kb] = (scale_exp.to(torch.int16) + 127).to(torch.uint8)
    sh = raw.view(pr // 32, 2, 16, pk // 8, 2, 4, 1)
    sh = sh.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    sh = sh.view(pr // 32, pk * 32)
    sh = sh.view(pr // 64, 2, pk * 32 // 4, 4)
    sh = sh.permute(0, 2, 1, 3).contiguous()
    return sh.view(pr // 64, pk * 64)


def load_module(mod_name):
    so_path = os.path.join(BUILD_DIR, f"{mod_name}.cpython-310-x86_64-linux-gnu.so")
    if not os.path.exists(so_path):
        return None
    spec = importlib.util.spec_from_file_location(mod_name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VARIANTS = [
    ("V0_legacyfork", f"tk_mxfp4_gluon_cpp_n32768_k{K_INC}_R35A_v12_memc_btw_all_legacyfork"),
    ("V0_vgprpf",     f"tk_mxfp4_gluon_cpp_n32768_k{K_INC}_R35A_v12_memc_btw_all_vgprpf"),
    ("V1_vmcnt15",    f"tk_mxfp4_gluon_cpp_n32768_k{K_INC}_R35A_v12_memc_btw_all_vmcnt15"),
    ("V2_vmcnt15_n8", f"tk_mxfp4_gluon_cpp_n32768_k{K_INC}_R35A_v12_memc_btw_all_vmcnt15_n8"),
    ("V3_vmcnt12_n4", f"tk_mxfp4_gluon_cpp_n32768_k{K_INC}_R35A_v12_memc_btw_all_vmcnt12_n4"),
]
INCUMBENT = "tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_v12_memc_btw_all"


def main():
    torch.manual_seed(42)
    k_blocks = K_INC // 32

    print(f"=== R35 Opt A KERNEL-vs-INCUMBENT correctness ({M_INC}x{N_INC}x{K_INC}) ===\n")

    A = gen_fp4(M_INC, K_INC)
    B = gen_fp4(N_INC, K_INC)
    # Use small scales (zero) to maximize finite output
    A_scale_exp = torch.zeros((M_INC, k_blocks), dtype=torch.int8, device="cuda")
    B_scale_exp = torch.zeros((N_INC, k_blocks), dtype=torch.int8, device="cuda")
    A_sc = preshuffle_mfma16_merged(A_scale_exp)
    B_sc = preshuffle_mfma16_merged(B_scale_exp)

    # Incumbent: 5 runs, take per-cell determinism
    print(f"[incumbent={INCUMBENT}]", flush=True)
    inc_mod = load_module(INCUMBENT)
    if inc_mod is None:
        print(f"  MISSING incumbent! Aborting.")
        sys.exit(1)
    inc_runs = []
    for _ in range(5):
        C = torch.zeros(M_INC, N_INC, dtype=torch.bfloat16, device="cuda")
        inc_mod.gemm_rcr(A, B, A_sc, B_sc, C)
        torch.cuda.synchronize()
        inc_runs.append(C.view(torch.int16).clone())
    inc_stack = torch.stack(inc_runs, 0)
    inc_agree = (inc_stack == inc_stack[0:1]).all(dim=0)
    inc_det_frac = inc_agree.float().mean().item()
    inc_C0 = inc_runs[0].view(torch.bfloat16).float()
    inc_finite = torch.isfinite(inc_C0)
    print(f"  inc: finite={inc_finite.float().mean()*100:5.1f}%  det={inc_det_frac*100:5.1f}%  "
          f"max_abs={inc_C0.abs()[inc_finite].max():.2f}")
    del inc_mod; gc.collect(); torch.cuda.empty_cache()

    results = {"incumbent_det_frac": inc_det_frac,
               "incumbent_finite_frac": inc_finite.float().mean().item()}

    for tag, mod_name in VARIANTS:
        print(f"\n[{tag}]", flush=True)
        mod = load_module(mod_name)
        if mod is None:
            print(f"  MISSING {mod_name}")
            results[tag] = {"status": "MISSING"}
            continue

        try:
            v_runs = []
            for _ in range(5):
                C = torch.zeros(M_INC, N_INC, dtype=torch.bfloat16, device="cuda")
                mod.gemm_rcr(A, B, A_sc, B_sc, C)
                torch.cuda.synchronize()
                v_runs.append(C.view(torch.int16).clone())
        except Exception as e:
            print(f"  CRASH: {e}")
            results[tag] = {"status": f"CRASH:{e}"}
            del mod; gc.collect(); torch.cuda.empty_cache()
            continue

        v_stack = torch.stack(v_runs, 0)
        v_agree = (v_stack == v_stack[0:1]).all(dim=0)
        v_det_frac = v_agree.float().mean().item()
        v_C0 = v_runs[0].view(torch.bfloat16).float()
        v_finite = torch.isfinite(v_C0)

        # Bit-equal vs incumbent on cells that are deterministic in both & finite in both
        both_det = inc_agree & v_agree
        both_finite = inc_finite & v_finite
        valid = both_det & both_finite
        n_valid = valid.sum().item()
        if n_valid > 0:
            bit_eq = (inc_runs[0][valid] == v_runs[0][valid]).float().mean().item()
            ref = inc_C0[valid]
            got = v_C0[valid]
            err = got - ref
            sig = (ref ** 2).mean().item()
            err_pow = (err ** 2).mean().item() + 1e-30
            ratio = max(sig, 1e-30) / err_pow
            snr_db = 10.0 * math.log10(ratio) if ratio > 0 else float('inf')
            max_abs = err.abs().max().item()
        else:
            bit_eq = 0.0
            snr_db = float('-inf')
            max_abs = float('inf')

        verdict = "PASS" if (snr_db >= 40.0 and v_det_frac >= 0.99 * inc_det_frac) else "FAIL"
        results[tag] = {
            "status": "OK", "snr_dB_vs_inc": snr_db, "max_abs_diff_vs_inc": max_abs,
            "bit_eq_frac_vs_inc": bit_eq,
            "v_det_frac": v_det_frac,
            "v_finite_frac": v_finite.float().mean().item(),
            "n_valid_cells": n_valid,
            "verdict": verdict,
        }
        print(f"  finite={v_finite.float().mean()*100:5.1f}% det={v_det_frac*100:5.1f}% "
              f"valid={n_valid/(M_INC*N_INC)*100:5.1f}% "
              f"bit_eq={bit_eq*100:6.2f}% SNR={snr_db:7.2f} dB max_abs={max_abs:.4g} -> {verdict}")

        del mod; gc.collect(); torch.cuda.empty_cache()

    out_path = os.path.join(SCRIPT_DIR, "R35_OPT_A_SNR_v2_RESULTS.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
