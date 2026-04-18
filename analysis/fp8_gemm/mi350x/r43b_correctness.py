"""R43 Dev B — correctness check: max|delta| between R43B fastpath and V1 baseline.

For each shape × layout, runs both fastpath and V1-legacy, compares output
element-wise. Expect max|delta| = 0.0 (kernel mathematically equivalent to
V1 — same scale-block FMA order — modulo IEEE non-associative reordering
of the inner 32-element dot which can introduce ULP-level rounding diff).

Usage:
    python3 r43b_correctness.py <fast_mod> <v1_mod> <layout> M N K
"""
import math, os, sys, random
import torch

torch.manual_seed(0); random.seed(0)

def gen_fp8(rows, cols, scale=0.05):
    x = torch.randn(rows, cols, dtype=torch.float32, device="cuda") * scale
    return x.to(torch.float8_e4m3fn)


def gen_scale_exp(rows, k_blocks):
    return torch.randint(low=-2, high=3, size=(rows, k_blocks), dtype=torch.int8, device="cuda")


def encode_scale_raw(scale_exp):
    raw = (scale_exp.to(torch.int16) + 127).to(torch.uint8)
    return torch.where(scale_exp == -128, torch.full_like(raw, 0xFF, dtype=torch.uint8), raw)


def preshuffle_v1(scale_exp):
    rows, k_blocks_local = scale_exp.shape
    padded_rows = math.ceil(rows / 32) * 32
    padded_k_blocks = math.ceil(k_blocks_local / 8) * 8
    raw = torch.full((padded_rows, padded_k_blocks), 0x7F, dtype=torch.uint8, device=scale_exp.device)
    raw[:rows, :k_blocks_local] = encode_scale_raw(scale_exp)
    shuffled = raw.view(padded_rows // 32, 2, 16, padded_k_blocks // 8, 2, 4, 1)
    shuffled = shuffled.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    return shuffled.view(padded_rows // 32, padded_k_blocks * 32)


def main():
    fast_mod_n, v1_mod_n, layout, M, N, K = sys.argv[1], sys.argv[2], sys.argv[3].lower(), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
    fast_mod = __import__(fast_mod_n)
    v1_mod = __import__(v1_mod_n)
    k_blocks = (K + 31) // 32

    if layout == "rrr":
        A = gen_fp8(M, K); B = gen_fp8(K, N)
        fn_name = "gemm_rrr_pq"
    elif layout == "crr":
        A = gen_fp8(K, M); B = gen_fp8(K, N)
        fn_name = "gemm_crr_pq"
    else:
        sys.exit("rcr not supported here (use Dev A's RCR file)")

    A_se = gen_scale_exp(M, k_blocks); B_se = gen_scale_exp(N, k_blocks)
    A_s = preshuffle_v1(A_se); B_s = preshuffle_v1(B_se)
    C_fast = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    C_v1   = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    getattr(fast_mod, fn_name)(A, B, A_s, B_s, C_fast); torch.cuda.synchronize()
    getattr(v1_mod, fn_name)(A, B, A_s, B_s, C_v1);   torch.cuda.synchronize()

    diff = (C_fast.to(torch.float32) - C_v1.to(torch.float32)).abs()
    print(f"R43B_PARITY layout={layout} shape={M}x{N}x{K} max_abs_diff={diff.max().item():.6e} "
          f"mean_abs_diff={diff.mean().item():.6e} fast_norm={C_fast.to(torch.float32).abs().mean().item():.4e} "
          f"v1_norm={C_v1.to(torch.float32).abs().mean().item():.4e}")


if __name__ == "__main__":
    main()
