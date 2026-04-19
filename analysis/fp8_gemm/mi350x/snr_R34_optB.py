#!/usr/bin/env python3
"""R34-OptB SNR check: kernel-vs-incumbent at K=128256 (R33-D _v2 methodology).

Acceptance per R34 mission: SNR >= 25 dB on small-entry subset OR diff_frac
within 5pp of V0_legacyfork (the K=128256 bf16 noise floor baseline).
"""
import os, sys, math, json, gc
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
sys.path.insert(0, BUILD_DIR)

M, N, K = 4096, 32768, 128256
k_blocks = K // 32
SMALL_THRESH = 1e6

VARIANTS = [
    ("V0_legacyfork",      "tk_mxfp4_gluon_cpp_n32768_k128256_R34B_v12_memc_btw_all_legacyfork"),
    ("V0_vgprpf",          "tk_mxfp4_gluon_cpp_n32768_k128256_R34B_v12_memc_btw_all_vgprpf"),
    ("V1_vmcnt15",         "tk_mxfp4_gluon_cpp_n32768_k128256_R34B_v12_memc_btw_all_vmcnt15"),
    ("V2_vmcnt15_snop",    "tk_mxfp4_gluon_cpp_n32768_k128256_R34B_v12_memc_btw_all_vmcnt15_snop"),
    ("V3_vmcnt20",         "tk_mxfp4_gluon_cpp_n32768_k128256_R34B_v12_memc_btw_all_vmcnt20"),
    ("V4_vgprpf_n0",       "tk_mxfp4_gluon_cpp_n32768_k128256_R34B_v12_memc_btw_all_vgprpf_n0"),
    ("V5_vgprpf_n1",       "tk_mxfp4_gluon_cpp_n32768_k128256_R34B_v12_memc_btw_all_vgprpf_n1"),
    ("V6_discard_vgpr",    "tk_mxfp4_gluon_cpp_n32768_k128256_R34B_v12_memc_btw_all_discard_vgpr"),
]
INCUMBENT_NAME = "tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_v12_memc_btw_all"


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
    print("Generating shared inputs (DETERMINISTIC scales=0)...")
    A = gen_fp4(M, K)
    B = gen_fp4(N, K)
    sc_exp_a = torch.zeros((M, k_blocks), dtype=torch.int8, device="cuda")
    sc_exp_b = torch.zeros((N, k_blocks), dtype=torch.int8, device="cuda")
    A_sc = preshuffle_mfma16_merged(sc_exp_a)
    B_sc = preshuffle_mfma16_merged(sc_exp_b)

    print(f"\n[incumbent] running...")
    import importlib
    INC_MOD = importlib.import_module(INCUMBENT_NAME)
    C_inc = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    INC_MOD.gemm_rcr(A, B, A_sc, B_sc, C_inc)
    torch.cuda.synchronize()
    inc_f = C_inc.float()
    inc_finite = torch.isfinite(inc_f)
    inc_small = inc_finite & (inc_f.abs() < SMALL_THRESH)
    print(f"  incumbent: finite_frac={inc_finite.float().mean():.4f}  "
          f"small_frac={inc_small.float().mean():.4f}  "
          f"max|small|={inc_f[inc_small].abs().max().item() if inc_small.any() else 0:.2f}")

    results = {"incumbent_small_frac": inc_small.float().mean().item(),
               "small_thresh": SMALL_THRESH,
               "variants": {}}

    v0_diff_frac = None
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

        v_f = C_v.float()
        v_finite = torch.isfinite(v_f)
        v_small = v_finite & (v_f.abs() < SMALL_THRESH)

        both_small = inc_small & v_small
        n_both = both_small.sum().item()
        n_inc_small = inc_small.sum().item()
        coverage = n_both / max(1, n_inc_small)

        if n_both == 0:
            snr_db = float("nan"); n_diff = 0; max_abs_diff = 0; mean_abs_diff = 0
        else:
            ref = inc_f[both_small]
            got = v_f[both_small]
            err = got - ref
            sig_pow = (ref ** 2).mean().item()
            err_pow = (err ** 2).mean().item()
            if err_pow == 0:
                snr_db = 999.0
            elif sig_pow > 0:
                snr_db = 10.0 * math.log10(sig_pow / err_pow)
            else:
                snr_db = float("nan")
            n_diff = int((ref != got).sum().item())
            max_abs_diff = err.abs().max().item()
            mean_abs_diff = err.abs().mean().item()

        diff_frac = n_diff / max(1, n_both)
        if tag == "V0_legacyfork":
            v0_diff_frac = diff_frac
        print(f"  small_frac={v_small.float().mean():.4f} both_small={n_both/M/N:.4f} cov={coverage:.4f} "
              f"snr={snr_db:.2f} dB n_diff={n_diff} ({diff_frac*100:.3f}% of both) "
              f"max_abs={max_abs_diff:.4f} mean_abs={mean_abs_diff:.6f}")

        results["variants"][tag] = {
            "status": "OK",
            "module": mod_name,
            "small_frac_self": v_small.float().mean().item(),
            "both_small_frac": n_both / (M * N),
            "coverage_of_inc_small": coverage,
            "snr_db_vs_incumbent": snr_db,
            "n_diff_small": n_diff,
            "diff_frac_of_both": diff_frac,
            "max_abs_diff_small": max_abs_diff,
            "mean_abs_diff_small": mean_abs_diff,
        }

        del C_v, VMOD
        gc.collect()
        torch.cuda.empty_cache()

    out_path = os.path.join(SCRIPT_DIR, "R34_OPT_B_SNR_RESULTS.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {out_path}")

    print("\n=== VERDICT (diff_frac <= V0_legacyfork + 5pp AND no-crash) ===")
    base_df = v0_diff_frac if v0_diff_frac is not None else 0.0
    for tag, r in results["variants"].items():
        if r.get("status") != "OK":
            print(f"  {tag:20s} STATUS={r.get('status','?')[:60]} -> CRASH-OR-DEAD")
        else:
            df = r["diff_frac_of_both"]
            ok = df <= base_df + 0.05
            verdict = "PASS" if ok else "FAIL"
            print(f"  {tag:20s} diff_frac={df:.4f} (base+0.05={base_df+0.05:.4f}) -> {verdict}")


if __name__ == "__main__":
    main()
