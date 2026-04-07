"""
MXFP4 GEMM launcher using the Gluon a4w4 kernel (compiled .hsaco binary).

Achieves ~5300-5400 TFLOPS on MI355X for 8192^3 MXFP4 GEMM.
Requires: triton built from ROCm/triton matmul_4waves branch.

Data format (no preshuffle required):
  A:        (M, K//2) uint8, row-major K-contiguous
  B:        (N, K//2) uint8, row-major K-contiguous
  a_scales: (M, K//32) uint8, e8m0
  b_scales: (N, K//32) uint8, e8m0
  C:        (M, N) bfloat16
"""
import os
import sys

os.environ["TRITON_ENABLE_LLIR_SCHED"] = "1"
os.environ["TRITON_ENABLE_AMDGCN_AS"] = "1"

import torch

_GLUON_TUTORIALS_PATH = "/shared_nfs/kyle/gfx9-gluon-tutorials/kernels/gemm/a4w4"
sys.path.insert(0, _GLUON_TUTORIALS_PATH)
from matmul_kernel import matmul as _gluon_matmul


def mxfp4_gemm_gluon(a_fp4, b_fp4, a_scales, b_scales):
    """
    MXFP4 GEMM: C = scale(A) @ scale(B)^T, output bf16.

    Args:
        a_fp4:    (M, K//2) uint8  — packed FP4 activations
        b_fp4:    (N, K//2) uint8  — packed FP4 weights (N-major, K-contiguous)
        a_scales: (M, K//32) uint8 — e8m0 scales for A
        b_scales: (N, K//32) uint8 — e8m0 scales for B

    Returns:
        c: (M, N) bfloat16
    """
    assert a_fp4.dtype == torch.uint8 and b_fp4.dtype == torch.uint8
    assert a_scales.dtype == torch.uint8 and b_scales.dtype == torch.uint8
    assert a_fp4.is_contiguous() and b_fp4.is_contiguous()
    assert a_scales.stride(1) > 0 and b_scales.stride(1) > 0
    M, K_half = a_fp4.shape
    N = b_fp4.shape[0]
    assert b_fp4.shape[1] == K_half
    assert a_scales.shape == (M, K_half * 2 // 32), f"a_scales shape {a_scales.shape} != ({M}, {K_half * 2 // 32})"
    assert b_scales.shape == (N, K_half * 2 // 32), f"b_scales shape {b_scales.shape} != ({N}, {K_half * 2 // 32})"
    return _gluon_matmul(a_fp4, b_fp4, a_scales, b_scales)


if __name__ == "__main__":
    import time, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int, nargs="?", default=8192)
    parser.add_argument("N", type=int, nargs="?", default=8192)
    parser.add_argument("K", type=int, nargs="?", default=8192)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--check", action="store_true", default=True)
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    DEVICE = "cuda"
    SCALE_GROUP_SIZE = 32

    torch.manual_seed(42)
    a_low = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8)
    a_high = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8)
    a_fp4 = (a_high << 4 | a_low).to(device=DEVICE)
    b_low = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device=DEVICE)
    b_high = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device=DEVICE)
    b_fp4 = b_low | b_high << 4
    M_pad = (M + 255) // 256 * 256
    a_scales = torch.randint(124, 128, (K // SCALE_GROUP_SIZE, M_pad),
                             dtype=torch.uint8, device=DEVICE).T[:M]
    b_scales = torch.randint(124, 128, (K // SCALE_GROUP_SIZE, N),
                             dtype=torch.uint8, device=DEVICE).T

    if args.check:
        def mxfp4_to_f32(x):
            x = x.contiguous().repeat_interleave(2, dim=1)
            x[:, ::2] = x[:, ::2] & 0xF
            x[:, 1::2] = x[:, 1::2] >> 4
            lut = torch.tensor([0,0.5,1,1.5,2,3,4,6,-0,-0.5,-1,-1.5,-2,-3,-4,-6],
                               dtype=torch.float32, device=x.device)
            return lut[x.long()]

        def e8m0_to_f32(x):
            return 2 ** ((x.contiguous() - 127).to(torch.float32))

        a_f32 = mxfp4_to_f32(a_fp4) * e8m0_to_f32(a_scales.contiguous().repeat_interleave(SCALE_GROUP_SIZE, dim=1))
        b_f32 = mxfp4_to_f32(b_fp4) * e8m0_to_f32(b_scales.contiguous().repeat_interleave(SCALE_GROUP_SIZE, dim=1))
        ref = torch.mm(a_f32, b_f32.T).to(torch.bfloat16)

    c = mxfp4_gemm_gluon(a_fp4, b_fp4, a_scales, b_scales)
    torch.cuda.synchronize()

    if args.check:
        diff = (c.float() - ref.float()).abs()
        snr = 10 * torch.log10((ref.float()**2).mean() / ((c.float()-ref.float())**2).mean())
        print(f"Correctness: max_err={diff.max().item():.2f}, SNR={snr.item():.2f} dB, "
              f"pass_rate={(diff < 3.0).float().mean().item()*100:.2f}%")

    for _ in range(args.warmup):
        c = mxfp4_gemm_gluon(a_fp4, b_fp4, a_scales, b_scales)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.iters):
        c = mxfp4_gemm_gluon(a_fp4, b_fp4, a_scales, b_scales)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    ms = elapsed / args.iters * 1000
    tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
    print(f"MXFP4 GEMM (gluon): {M}x{N}x{K}, {ms:.3f} ms, {tflops:.0f} TFLOPS "
          f"(warmup={args.warmup}, iters={args.iters})")
