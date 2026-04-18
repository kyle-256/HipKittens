#!/usr/bin/env python3
"""R32-A SNR check v3: compare each variant's output to the INCUMBENT's output
on the FULL M*N matrix, restricted to entries that are finite in both.

This is the right kernel-vs-kernel correctness gate (incumbent is known-good
from prior R28-C verifications); the torch-FP32 ref blows bf16 dynamic range
with random [-2,3] exponent scales on K=128256.

Usage:
  HIP_VISIBLE_DEVICES=N python3 snr_R32_optA_v3.py
"""
import os, sys, math, json, gc
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
sys.path.insert(0, BUILD_DIR)

M, N, K = 4096, 32768, 128256
k_blocks = K // 32

VARIANTS = [
    ("V1_kls2",     "tk_mxfp4_gluon_cpp_n32768_k128256_R32A_ts_lgk2_v12_memc_btw_all_kls2"),
    ("V2_nt",       "tk_mxfp4_gluon_cpp_n32768_k128256_R32A_ts_lgk2_v12_memc_btw_all_nt"),
    ("V3_kls2_nt",  "tk_mxfp4_gluon_cpp_n32768_k128256_R32A_ts_lgk2_v12_memc_btw_all_kls2_nt"),
]
INCUMBENT = ("incumbent", "tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_v12_memc_btw_all")


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


def main():
    torch.manual_seed(0)
    print("Generating shared inputs...")
    A = gen_fp4(M, K)
    B = gen_fp4(N, K)
    sc_exp_a = torch.randint(-2, 3, (M, k_blocks), dtype=torch.int8, device="cuda")
    sc_exp_b = torch.randint(-2, 3, (N, k_blocks), dtype=torch.int8, device="cuda")
    A_sc = preshuffle_mfma16_merged(sc_exp_a)
    B_sc = preshuffle_mfma16_merged(sc_exp_b)

    # Run incumbent first (reference)
    print(f"\n[incumbent] running...")
    import importlib
    INC_MOD = importlib.import_module(INCUMBENT[1])
    C_inc = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    INC_MOD.gemm_rcr(A, B, A_sc, B_sc, C_inc)
    torch.cuda.synchronize()
    inc_finite_mask = torch.isfinite(C_inc)
    inc_finite_frac = inc_finite_mask.float().mean().item()
    print(f"  incumbent finite_frac={inc_finite_frac:.4f}")

    results = {"incumbent_finite_frac": inc_finite_frac, "variants": {}}

    for tag, mod_name in VARIANTS:
        print(f"\n[{tag}] running...")
        try:
            VMOD = importlib.import_module(mod_name)
        except Exception as e:
            print(f"  IMPORT FAIL: {e}")
            results["variants"][tag] = {"status": f"IMPORT_FAIL:{e}"}
            continue

        try:
            C_v = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
            VMOD.gemm_rcr(A, B, A_sc, B_sc, C_v)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  RUN EXCEPTION: {e}")
            results["variants"][tag] = {"status": f"RUN_FAIL:{e}"}
            continue

        v_finite_mask = torch.isfinite(C_v)
        v_finite_frac = v_finite_mask.float().mean().item()

        # Both-finite mask
        both_finite = inc_finite_mask & v_finite_mask
        both_finite_frac = both_finite.float().mean().item()

        # Compute SNR over both-finite entries: ref = incumbent, signal = (ref) power, error = (got - ref) power
        if both_finite.sum().item() == 0:
            snr_db = float("nan")
            sig_pow = err_pow = float("nan")
        else:
            ref_f = C_inc.float()[both_finite]
            got_f = C_v.float()[both_finite]
            err = got_f - ref_f
            sig_pow = (ref_f ** 2).mean().item()
            err_pow = (err ** 2).mean().item() + 1e-30
            snr_db = 10.0 * math.log10(sig_pow / err_pow) if sig_pow > 0 else float("nan")

        # Bit-exact match check on both-finite
        n_diff = 0; max_abs_diff = 0.0
        if both_finite.sum().item() > 0:
            diff_mask = (C_inc != C_v) & both_finite
            n_diff = diff_mask.sum().item()
            if n_diff > 0:
                max_abs_diff = (C_inc[both_finite].float() - C_v[both_finite].float()).abs().max().item()

        print(f"  finite_frac={v_finite_frac:.4f} both_finite_frac={both_finite_frac:.4f} snr_vs_inc={snr_db:.2f} dB n_diff={n_diff} max_abs_diff={max_abs_diff:.4f}")

        results["variants"][tag] = {
            "status": "OK",
            "module": mod_name,
            "finite_frac": v_finite_frac,
            "both_finite_frac": both_finite_frac,
            "snr_db_vs_incumbent": snr_db,
            "sig_pow": sig_pow,
            "err_pow": err_pow,
            "n_diff_finite": n_diff,
            "max_abs_diff_finite": max_abs_diff,
        }

        del C_v, VMOD
        gc.collect()
        torch.cuda.empty_cache()

    out_path = os.path.join(SCRIPT_DIR, "R32_OPT_A_SNR_v3_RESULTS.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {out_path}")

    print("\n=== VERDICT (kernel-vs-incumbent SNR ≥ 25 dB acceptance gate) ===")
    for tag, r in results["variants"].items():
        if r.get("status") != "OK":
            print(f"  {tag:14s} STATUS={r.get('status','?')} → DEAD-BY-CRASH")
        else:
            snr = r["snr_db_vs_incumbent"]
            verdict = "PASS-SNR" if snr >= 25 else "FAIL-SNR"
            print(f"  {tag:14s} SNR={snr:.2f} dB → {verdict}")


if __name__ == "__main__":
    main()
