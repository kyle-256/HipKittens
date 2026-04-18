"""R42 Dev B — correctness check for SMALLM-B32-TAIL.

Compares smallm tail kernel output vs baseline (V1-LEGACY-FALLBACK tail)
output for the same MXFP8 quantized inputs. SNR computed in dB.
PASS gate: SNR >= 48 dB (per R42 Dev B SHIP gate).

Usage: python3 r42b_smallm_correctness.py <baseline_mod> <smallm_mod> M N K
"""
import math
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
import r41c_decode_bench as bench  # reuse helpers

torch.manual_seed(42)


def snr_db(ref: torch.Tensor, test: torch.Tensor) -> float:
    err = (ref.float() - test.float())
    sig_pow = (ref.float() ** 2).mean().item()
    err_pow = (err ** 2).mean().item()
    if err_pow <= 0:
        return float("inf")
    return 10.0 * math.log10(sig_pow / max(err_pow, 1e-30))


def main():
    base_mod_name = sys.argv[1]
    test_mod_name = sys.argv[2]
    M = int(sys.argv[3]); N = int(sys.argv[4]); K = int(sys.argv[5])

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
    base_mod = __import__(base_mod_name)
    test_mod = __import__(test_mod_name)

    k_blocks = (K + 31) // 32
    A = bench.gen_fp8(M, K)
    B = bench.gen_fp8(N, K)  # RCR: B is (N, K)
    A_se = bench.gen_scale_exp(M, k_blocks)
    B_se = bench.gen_scale_exp(N, k_blocks)
    # Use V1 preshuffle (decode path)
    A_s = bench.preshuffle_v1(A_se)
    B_s = bench.preshuffle_v1(B_se)

    C_ref = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    C_test = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    base_mod.gemm_rcr_pq(A, B, A_s, B_s, C_ref)
    torch.cuda.synchronize()
    test_mod.gemm_rcr_pq(A, B, A_s, B_s, C_test)
    torch.cuda.synchronize()

    snr = snr_db(C_ref, C_test)
    abs_max = (C_ref.float() - C_test.float()).abs().max().item()
    rel_max = abs_max / max(C_ref.float().abs().max().item(), 1e-30)
    print(f"R42B_SNR M={M} N={N} K={K} snr_db={snr:.2f} abs_max_err={abs_max:.4e} rel_max={rel_max:.4e}")
    if snr >= 48.0:
        print(f"R42B_VERDICT PASS (snr {snr:.2f} dB >= 48 dB)")
        sys.exit(0)
    else:
        print(f"R42B_VERDICT FAIL (snr {snr:.2f} dB < 48 dB)")
        sys.exit(1)


if __name__ == "__main__":
    main()
