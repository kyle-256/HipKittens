"""FP8 blockwise GEMM test: correctness + benchmark vs per-tensor."""

import math
import os
import sys

import torch

torch.manual_seed(42)

import tk_fp8_blockwise_layouts as bw

BK = 128


def parse_size(argv):
    if len(argv) == 2:
        n = int(argv[1])
        return n, n, n
    if len(argv) == 4:
        return int(argv[1]), int(argv[2]), int(argv[3])
    return 8192, 8192, 8192


M, N, K = parse_size(sys.argv)
Kb = K // BK
Nb = (N + BK - 1) // BK

num_warmup = int(os.environ.get("BW_WARMUP", "50"))
num_iters = int(os.environ.get("BW_ITERS", "200"))
check = os.environ.get("BW_CHECK", "1") != "0"
layouts = {s.strip().lower() for s in os.environ.get("BW_LAYOUTS", "rcr").split(",") if s.strip()}

start_ev = torch.cuda.Event(enable_timing=True)
end_ev = torch.cuda.Event(enable_timing=True)
flops = 2 * M * N * K


def gen_fp8(rows, cols):
    return (torch.randn(rows, cols, device="cuda") * 0.1).to(torch.float8_e4m3fn)


def gen_scales(dim0, dim1):
    return (torch.rand(dim0, dim1, device="cuda", dtype=torch.float32) * 0.5 + 0.5)


def ref_rcr_blockwise(A, B, a_scale_nat, b_scale_nat, M, N, K, Kb):
    """Torch reference: C = Σ_ki (A[:,ki*128:(ki+1)*128] @ B[:,ki*128:(ki+1)*128]^T) * a_s * b_s."""
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    for ki in range(Kb):
        k0, k1 = ki * BK, min((ki + 1) * BK, K)
        partial = A[:, k0:k1].float() @ B[:, k0:k1].float().T
        a_s = a_scale_nat[:, ki : ki + 1]
        b_s = b_scale_nat[:, ki]
        b_s_exp = b_s.repeat_interleave(BK)[:N].unsqueeze(0)
        C += partial * a_s * b_s_exp
    return C


def benchmark(fn, out, warmup=num_warmup, iters=num_iters):
    for _ in range(warmup):
        out.zero_()
        fn()
    times = []
    for _ in range(iters):
        out.zero_()
        torch.cuda.synchronize()
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize()
        times.append(start_ev.elapsed_time(end_ev))
    return times


def snr_db(test, ref):
    t, r = test.float(), ref.float()
    sig = (r * r).sum().item()
    noise = ((t - r) ** 2).sum().item()
    if noise == 0:
        return float("inf")
    return 10.0 * math.log10(sig / max(noise, 1e-45))


print(f"=== FP8 Blockwise GEMM Test: M={M}, N={N}, K={K}, Kb={Kb}, Nb={Nb} ===")
print(f"Warmup={num_warmup}, iters={num_iters}, check={check}, layouts={','.join(sorted(layouts))}\n")

if "rcr" in layouts:
    print("--- RCR blockwise (MFMA kernel) ---")
    A = gen_fp8(M, K)
    B = gen_fp8(N, K)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    a_scale_nat = gen_scales(M, Kb)
    b_scale_nat = gen_scales(Nb, Kb)

    a_scale = a_scale_nat.T.contiguous()
    b_scale = b_scale_nat.T.contiguous()

    run = lambda: bw.gemm_rcr_blockwise(A, B, C, a_scale, b_scale)
    run()
    torch.cuda.synchronize()

    if check:
        C_ref = ref_rcr_blockwise(A, B, a_scale_nat, b_scale_nat, M, N, K, Kb)
        snr = snr_db(C[:M, :N], C_ref)
        max_err = (C[:M, :N].float() - C_ref).abs().max().item()
        mean_err = (C[:M, :N].float() - C_ref).abs().mean().item()
        print(f"  SNR:  {snr:.2f} dB")
        print(f"  Max err:  {max_err:.4f}   Mean err: {mean_err:.6f}")
        ok = snr > 48.0
        print(f"  Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    times = benchmark(run, C)
    avg = sum(times) / len(times)
    tflops = flops / (avg * 1e9)
    print(f"  Avg time: {avg:.4f} ms   TFLOPS: {tflops:.2f}")

    try:
        import tk_fp8_layouts

        C_pt = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        run_pt = lambda: tk_fp8_layouts.gemm_rcr(A, B, C_pt, 1.0)
        times_pt = benchmark(run_pt, C_pt)
        avg_pt = sum(times_pt) / len(times_pt)
        tflops_pt = flops / (avg_pt * 1e9)
        pct = tflops / tflops_pt * 100
        print(f"  Per-tensor: {avg_pt:.4f} ms   TFLOPS: {tflops_pt:.2f}")
        print(f"  Blockwise / Per-tensor: {pct:.1f}%")
    except ImportError:
        print("  (tk_fp8_layouts not available for comparison)")

    print()

if "rrr" in layouts:
    print("--- RRR blockwise (MFMA kernel) ---")
    A = gen_fp8(M, K)
    B = gen_fp8(K, N)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    a_scale_nat = gen_scales(M, Kb)
    b_scale_nat = gen_scales(Nb, Kb)
    a_scale = a_scale_nat.T.contiguous()
    b_scale = b_scale_nat.T.contiguous()
    run = lambda: bw.gemm_rrr_blockwise(A, B, C, a_scale, b_scale)
    run()
    torch.cuda.synchronize()
    if check:
        C_ref = torch.zeros(M, N, dtype=torch.float32, device="cuda")
        for ki in range(Kb):
            k0, k1 = ki * BK, min((ki + 1) * BK, K)
            partial = A[:, k0:k1].float() @ B[k0:k1, :].float()
            a_s = a_scale_nat[:, ki : ki + 1]
            b_s = b_scale_nat[:, ki]
            b_s_exp = b_s.repeat_interleave(BK)[:N].unsqueeze(0)
            C_ref += partial * a_s * b_s_exp
        snr = snr_db(C[:M, :N], C_ref)
        print(f"  SNR: {snr:.2f} dB  {'PASS' if snr > 48 else 'FAIL'}")
        if snr <= 48:
            sys.exit(1)
    times = benchmark(run, C)
    avg = sum(times) / len(times)
    tflops = flops / (avg * 1e9)
    print(f"  Avg time: {avg:.4f} ms   TFLOPS: {tflops:.2f}")
    print()

if "crr" in layouts:
    print("--- CRR blockwise (MFMA kernel) ---")
    At = gen_fp8(K, M)
    B = gen_fp8(K, N)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    a_scale = gen_scales(Kb, M)
    b_scale = gen_scales(Kb, N)
    run = lambda: bw.gemm_crr_blockwise(At, B, C, a_scale, b_scale)
    run()
    torch.cuda.synchronize()
    if check:
        C_ref = torch.zeros(M, N, dtype=torch.float32, device="cuda")
        for ki in range(Kb):
            k0, k1 = ki * BK, min((ki + 1) * BK, K)
            partial = At[k0:k1, :].float().T @ B[k0:k1, :].float()
            a_s = a_scale[ki, :].unsqueeze(1)
            b_s = b_scale[ki, :].unsqueeze(0)
            C_ref += partial * a_s * b_s
        snr = snr_db(C[:M, :N], C_ref)
        print(f"  SNR: {snr:.2f} dB  {'PASS' if snr > 48 else 'FAIL'}")
        if snr <= 48:
            sys.exit(1)
    times = benchmark(run, C)
    avg = sum(times) / len(times)
    tflops = flops / (avg * 1e9)
    print(f"  Avg time: {avg:.4f} ms   TFLOPS: {tflops:.2f}")
    print()

print("Done.")
