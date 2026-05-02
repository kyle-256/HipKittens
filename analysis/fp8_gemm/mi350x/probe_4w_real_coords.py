"""Probe correctness of the R59 step-2B-2 4w-style grouped FP8 kernel.

Acceptance gate (per round-58-dm note):
  * max_abs ≤ 0.5 (BF16 quantization noise dominates fp8 mma noise)
  * SNR ≥ 22 dB (= ~16-bit equivalent precision)

Probe shapes
------------
  M_total=512, N=256, K=256, B=1 (single group)
  M_total=768, N=256, K=128, B=1
  M_total=512, N=512, K=384, B=1

All shapes satisfy M%256==0, N%256==0, K%128==0 (the R59 kernel's
caller contract; group binary search / K-tail / N-mask are deferred
to R60+).
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tk_fp8_layouts


def to_fp8(x):
    return x.to(torch.float8_e4m3fn)


def torch_ref(a_fp8, b_fp8, scale_a, scale_b):
    """Reference: A @ B^T (RCR layout) in fp32, dequantized + scaled."""
    a = a_fp8.float()
    b = b_fp8.float()
    return (a @ b.T) * (scale_a * scale_b)


def snr_db(c_test, c_ref):
    diff = c_test.float() - c_ref.float()
    sig = (c_ref.float() ** 2).sum().item()
    nse = (diff ** 2).sum().item()
    if nse == 0.0:
        return float("inf")
    if sig == 0.0:
        return float("-inf")
    return 10.0 * math.log10(sig / nse)


def probe_one(M_total, N, K, scale_a=1.0, scale_b=1.0, dump=False):
    torch.manual_seed(42)
    a = (torch.randn(M_total, K, dtype=torch.float32, device="cuda") * 0.1)
    a_fp8 = to_fp8(a)
    b = (torch.randn(N, K, dtype=torch.float32, device="cuda") * 0.1)
    b_fp8 = to_fp8(b)
    b_fp8_grouped = b_fp8.unsqueeze(0).contiguous()
    c = torch.zeros(M_total, N, dtype=torch.bfloat16, device="cuda")
    group_offs = torch.tensor([0, M_total], dtype=torch.int64, device="cuda")

    tk_fp8_layouts.test_4w_real_coords(
        a_fp8, b_fp8_grouped, c, scale_a, scale_b, group_offs
    )
    torch.cuda.synchronize()

    c_ref = torch_ref(a_fp8, b_fp8, scale_a, scale_b)
    c_test = c.float()

    diff = (c_test - c_ref).abs()
    max_abs = diff.max().item()
    snr = snr_db(c_test, c_ref)

    if dump:
        print(f"\n[DUMP] shape ({M_total}, {N}, {K})")
        print(f"  c_ref[0:4, 0:4]:\n{c_ref[0:4, 0:4]}")
        print(f"  c_test[0:4, 0:4]:\n{c_test[0:4, 0:4]}")
        print(f"  c_ref[0, :8]:    {c_ref[0, :8]}")
        print(f"  c_test[0, :8]:   {c_test[0, :8]}")
        print(f"  c_ref[64, :8]:   {c_ref[64, :8]}")
        print(f"  c_test[64, :8]:  {c_test[64, :8]}")
        print(f"  c_ref[128, :8]:  {c_ref[128, :8]}")
        print(f"  c_test[128, :8]: {c_test[128, :8]}")
        print(f"  c_ref[192, :8]:  {c_ref[192, :8]}")
        print(f"  c_test[192, :8]: {c_test[192, :8]}")
        # zero blocks?
        for r in [0, 64, 128, 192]:
            for col in [0, 64, 128, 192]:
                if r < M_total and col < N:
                    chunk = c_test[r:r+64, col:col+64]
                    ref_chunk = c_ref[r:r+64, col:col+64]
                    diff_chunk = (chunk - ref_chunk).abs()
                    diff_max = diff_chunk.max().item()
                    chunk_max = chunk.abs().max().item()
                    if diff_max > 0.5:
                        # Find offending rows
                        bad_rows = (diff_chunk.max(dim=1).values > 0.5).nonzero().flatten().tolist()
                        bad_cols = (diff_chunk.max(dim=0).values > 0.5).nonzero().flatten().tolist()
                        print(f"  [m={r:4d}, n={col:4d}] cAB cell: chunk_max={chunk_max:.3f} diff_max={diff_max:.3f}")
                        print(f"    BAD rows_in_cell ({len(bad_rows)} total)={bad_rows}")
                        print(f"    BAD cols_in_cell ({len(bad_cols)} total)={bad_cols}")
                    else:
                        print(f"  [m={r:4d}, n={col:4d}] cAB cell: chunk_max={chunk_max:.3f} diff_max={diff_max:.3f} OK")
    print(f"  shape M={M_total} N={N} K={K}: max_abs={max_abs:.4f} SNR={snr:.2f} dB")
    return max_abs, snr


def main():
    print("R59 step-2B-2 4w-style grouped FP8 correctness probe")
    print("-" * 60)
    results = []
    for i, (M, N, K) in enumerate([
        (512, 256, 256),
        (768, 256, 128),
        (512, 512, 384),
        (1024, 256, 512),
    ]):
        results.append(probe_one(M, N, K, dump=(i == 0)))
    print("-" * 60)
    pass_count = 0
    fail_count = 0
    for (max_abs, snr), (M, N, K) in zip(results, [
        (512, 256, 256), (768, 256, 128), (512, 512, 384), (1024, 256, 512),
    ]):
        ok_max = max_abs <= 0.5
        ok_snr = snr >= 22.0
        verdict = "PASS" if (ok_max and ok_snr) else "FAIL"
        if verdict == "PASS":
            pass_count += 1
        else:
            fail_count += 1
        print(f"  M={M} N={N} K={K}: max_abs={max_abs:.4f} (gate≤0.5: {'OK' if ok_max else 'FAIL'}) "
              f"SNR={snr:.2f} dB (gate≥22: {'OK' if ok_snr else 'FAIL'}) → {verdict}")
    print(f"\nSummary: {pass_count}/{pass_count+fail_count} probe shapes PASS")
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
