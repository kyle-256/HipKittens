import json
import math
import os
import random
import sys

import torch

torch.manual_seed(0)
random.seed(0)

import tk_mxfp8_layouts


def parse_problem_size(argv):
    if len(argv) == 2:
        n = int(argv[1])
        return n, n, n
    if len(argv) == 4:
        return int(argv[1]), int(argv[2]), int(argv[3])
    return 8192, 8192, 8192


M, N, K = parse_problem_size(sys.argv)
build_M = int(os.environ.get("MXFP8_BUILD_M", str(M)))
build_N = int(os.environ.get("MXFP8_BUILD_N", str(N)))
build_K = int(os.environ.get("MXFP8_BUILD_K", str(K)))

if build_M < M or build_N < N or build_K < K:
    raise ValueError(
        f"Build shape ({build_M}, {build_N}, {build_K}) must cover "
        f"problem shape ({M}, {N}, {K})"
    )

num_warmup = int(os.environ.get("MXFP8_WARMUP", "10"))
num_iters = int(os.environ.get("MXFP8_ITERS", "20"))
determinism_runs = max(1, int(os.environ.get("MXFP8_DETERMINISM_RUNS", "1")))
snr_threshold_db = float(os.environ.get("MXFP8_SNR_THRESHOLD_DB", "48.0"))
use_preshuffle_quant = os.environ.get("MXFP8_PRESHUFFLE_QUANT", "0") != "0"
requested_layouts = {
    layout.strip().lower()
    for layout in os.environ.get("MXFP8_LAYOUTS", "rcr,rrr,crr").split(",")
    if layout.strip()
}
check_results = os.environ.get("MXFP8_CHECK", "1") != "0"

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
flops_ref = 2 * M * N * K
k_blocks = (build_K + 31) // 32


def default_output_path():
    suffix = "_pq" if use_preshuffle_quant else ""
    if M == N == K:
        return f"mxfp8_layout_results_{M}{suffix}.json"
    return f"mxfp8_layout_results_{M}x{N}x{K}{suffix}.json"


def generate_fp8_matrix(total_rows, total_cols, valid_rows, valid_cols):
    x = torch.zeros(total_rows, total_cols, dtype=torch.float32, device="cuda")
    x[:valid_rows, :valid_cols] = (
        torch.randn(valid_rows, valid_cols, dtype=torch.float32, device="cuda") * 0.05
    )
    return x.to(torch.float8_e4m3fn)


def generate_scale_matrix(total_rows, total_k_blocks, valid_rows):
    # MXFP8 E8M0 is modeled as an int8 power-of-two exponent.
    scales = torch.zeros(total_rows, total_k_blocks, dtype=torch.int8, device="cuda")
    scales[:valid_rows, :] = torch.randint(
        low=-2,
        high=3,
        size=(valid_rows, total_k_blocks),
        dtype=torch.int8,
        device="cuda",
    )
    return scales


def encode_scale_matrix_raw(scale_exp):
    raw = (scale_exp.to(torch.int16) + 127).to(torch.uint8)
    return torch.where(
        scale_exp == -128,
        torch.full_like(raw, 0xFF, dtype=torch.uint8),
        raw,
    )


def preshuffle_scale_matrix_mfma16(scale_exp):
    rows, k_blocks_local = scale_exp.shape
    padded_rows = math.ceil(rows / 32) * 32
    padded_k_blocks = math.ceil(k_blocks_local / 8) * 8
    raw = torch.full(
        (padded_rows, padded_k_blocks),
        0x7F,
        dtype=torch.uint8,
        device=scale_exp.device,
    )
    raw[:rows, :k_blocks_local] = encode_scale_matrix_raw(scale_exp)
    shuffled = raw.view(padded_rows // 32, 2, 16, padded_k_blocks // 8, 2, 4, 1)
    shuffled = shuffled.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    return shuffled.view(padded_rows // 32, padded_k_blocks * 32)


def expand_row_scales(scale_exp, cols):
    return torch.pow(2.0, scale_exp.float()).repeat_interleave(32, dim=1)[:, :cols]


def expand_col_scales(scale_exp, rows):
    return torch.pow(2.0, scale_exp.float()).transpose(0, 1).repeat_interleave(32, dim=0)[:rows, :]


def benchmark_kernel(fn, output, warmup=num_warmup, iters=num_iters):
    for _ in range(warmup):
        output.zero_()
        fn()
    timings = []
    for _ in range(iters):
        output.zero_()
        torch.cuda.synchronize()
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event))
    return timings


def compute_snr_db(C_test, C_ref):
    C_test_f32 = C_test.float()
    C_ref_f32 = C_ref.float()
    noise = C_test_f32 - C_ref_f32
    signal_power = torch.sum(C_ref_f32 * C_ref_f32).item()
    noise_power = torch.sum(noise * noise).item()
    if noise_power == 0.0:
        return float("inf")
    if signal_power == 0.0:
        return float("-inf")
    return 10.0 * math.log10(signal_power / noise_power)


def check_determinism(fn, output, label):
    if determinism_runs <= 1:
        return True, 0.0

    output.zero_()
    fn()
    reference = output[:M, :N].clone()
    max_abs_diff = 0.0
    deterministic = True
    for _ in range(determinism_runs - 1):
        output.zero_()
        fn()
        candidate = output[:M, :N]
        if not torch.equal(candidate, reference):
            deterministic = False
            max_abs_diff = max(
                max_abs_diff,
                (candidate.float() - reference.float()).abs().max().item(),
            )

    status = "PASS" if deterministic else "FAIL"
    print(f"  [{label}] Determinism ({determinism_runs} runs): {status}")
    if not deterministic:
        print(f"    Max abs diff across runs: {max_abs_diff:.6f}")
    return deterministic, max_abs_diff


def check_correctness(C_test, C_ref, label):
    C_test_f32 = C_test.float()
    C_ref_f32 = C_ref.float()
    diff = (C_test_f32 - C_ref_f32).abs()
    scale = C_ref_f32.abs().clamp(min=1.0)
    rel_diff = diff / scale
    atol = 3.0
    rtol = 0.10
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = rel_diff.max().item()
    mean_rel = rel_diff.mean().item()
    snr_db = compute_snr_db(C_test, C_ref)
    snr_ok = snr_db > snr_threshold_db
    pass_count = ((diff <= atol) | (rel_diff <= rtol)).sum().item()
    total = C_test.numel()
    pass_rate = pass_count / total * 100
    print(f"  [{label}] Correctness (rtol={rtol:.4f}, atol={atol:.1f}):")
    print(f"    Max  abs error: {max_abs:.4f}, Mean abs error: {mean_abs:.4f}")
    print(f"    Max  rel error: {max_rel:.4f}, Mean rel error: {mean_rel:.4f}")
    print(f"    SNR: {snr_db:.2f} dB (threshold {snr_threshold_db:.1f} dB)")
    print(f"    Pass rate: {pass_count}/{total} ({pass_rate:.2f}%)")
    ok = pass_rate >= 99.0 and snr_ok
    print(f"    Result: {'PASS' if ok else 'FAIL'}")
    return {"snr_db": snr_db, "ok": ok}


print(
    f"=== MXFP8 GEMM Layout Benchmark ({'preshuffle-quant' if use_preshuffle_quant else 'reference scale layout'}): "
    f"M={M}, N={N}, K={K} ==="
)
if (build_M, build_N, build_K) != (M, N, K):
    print(f"Build shape={build_M}x{build_N}x{build_K} (zero-padded)")
print(f"Warmup={num_warmup}, iters={num_iters}, correctness={'on' if check_results else 'off'}")
if check_results:
    print(f"SNR threshold={snr_threshold_db:.1f} dB, determinism_runs={determinism_runs}")
print(f"Layouts={','.join(sorted(requested_layouts))}\n")

results = {f"{M}x{N}x{K}": {}}
result_key = f"{M}x{N}x{K}"


def record_result(layout, timings, output, reference, runner):
    avg_ms = sum(timings) / len(timings)
    tflops = flops_ref / (avg_ms * 1e9)
    print(f"  Avg time: {avg_ms:.4f} ms, TFLOPS: {tflops:.2f}")
    results[result_key][layout] = {"avg_ms": avg_ms, "tflops": tflops}
    if check_results:
        quality = check_correctness(output[:M, :N], reference, layout.upper())
        det_ok, det_max_abs = check_determinism(runner, output, layout.upper())
        results[result_key][layout].update(
            {
                "snr_db": quality["snr_db"],
                "deterministic": det_ok,
                "determinism_max_abs_diff": det_max_abs,
                "quality_ok": quality["ok"] and det_ok,
            }
        )
    print()


if "rcr" in requested_layouts:
    print("--- RCR Layout: C = A @ B^T ---")
    A = generate_fp8_matrix(build_M, build_K, M, K)
    B = generate_fp8_matrix(build_N, build_K, N, K)
    A_scale_exp = generate_scale_matrix(build_M, k_blocks, M)
    B_scale_exp = generate_scale_matrix(build_N, k_blocks, N)
    if use_preshuffle_quant:
        A_scale = preshuffle_scale_matrix_mfma16(A_scale_exp)
        B_scale = preshuffle_scale_matrix_mfma16(B_scale_exp)
        run = lambda: tk_mxfp8_layouts.gemm_rcr_pq(A, B, A_scale, B_scale, C)
    else:
        A_scale = A_scale_exp
        B_scale = B_scale_exp
        run = lambda: tk_mxfp8_layouts.gemm_rcr(A, B, A_scale, B_scale, C)
    C = torch.zeros(build_M, build_N, dtype=torch.bfloat16, device="cuda")
    timings = benchmark_kernel(run, C)
    A_ref = A[:M, :K].float() * expand_row_scales(A_scale_exp[:M, :k_blocks], K)
    B_ref = B[:N, :K].float() * expand_row_scales(B_scale_exp[:N, :k_blocks], K)
    C_ref = A_ref @ B_ref.T
    record_result("rcr", timings, C, C_ref, run)

if "rrr" in requested_layouts:
    print("--- RRR Layout: C = A @ B ---")
    A = generate_fp8_matrix(build_M, build_K, M, K)
    B = generate_fp8_matrix(build_K, build_N, K, N)
    A_scale_exp = generate_scale_matrix(build_M, k_blocks, M)
    B_scale_exp = generate_scale_matrix(build_N, k_blocks, N)
    if use_preshuffle_quant:
        A_scale = preshuffle_scale_matrix_mfma16(A_scale_exp)
        B_scale = preshuffle_scale_matrix_mfma16(B_scale_exp)
        run = lambda: tk_mxfp8_layouts.gemm_rrr_pq(A, B, A_scale, B_scale, C)
    else:
        A_scale = A_scale_exp
        B_scale = B_scale_exp
        run = lambda: tk_mxfp8_layouts.gemm_rrr(A, B, A_scale, B_scale, C)
    C = torch.zeros(build_M, build_N, dtype=torch.bfloat16, device="cuda")
    timings = benchmark_kernel(run, C)
    A_ref = A[:M, :K].float() * expand_row_scales(A_scale_exp[:M, :k_blocks], K)
    B_ref = B[:K, :N].float() * expand_col_scales(B_scale_exp[:N, :k_blocks], K)
    C_ref = A_ref @ B_ref
    record_result("rrr", timings, C, C_ref, run)

if "crr" in requested_layouts:
    print("--- CRR Layout: C = A^T @ B ---")
    A = generate_fp8_matrix(build_K, build_M, K, M)
    B = generate_fp8_matrix(build_K, build_N, K, N)
    A_scale_exp = generate_scale_matrix(build_M, k_blocks, M)
    B_scale_exp = generate_scale_matrix(build_N, k_blocks, N)
    if use_preshuffle_quant:
        A_scale = preshuffle_scale_matrix_mfma16(A_scale_exp)
        B_scale = preshuffle_scale_matrix_mfma16(B_scale_exp)
        run = lambda: tk_mxfp8_layouts.gemm_crr_pq(A, B, A_scale, B_scale, C)
    else:
        A_scale = A_scale_exp
        B_scale = B_scale_exp
        run = lambda: tk_mxfp8_layouts.gemm_crr(A, B, A_scale, B_scale, C)
    C = torch.zeros(build_M, build_N, dtype=torch.bfloat16, device="cuda")
    timings = benchmark_kernel(run, C)
    A_ref = A[:K, :M].float() * expand_col_scales(A_scale_exp[:M, :k_blocks], K)
    B_ref = B[:K, :N].float() * expand_col_scales(B_scale_exp[:N, :k_blocks], K)
    C_ref = A_ref.T @ B_ref
    record_result("crr", timings, C, C_ref, run)


outfile = os.environ.get("MXFP8_OUTPUT", default_output_path())
with open(outfile, "w") as f:
    json.dump(results, f, indent=4)
print(f"Results saved to {outfile}")

quality_checks = [
    results[result_key][layout].get("quality_ok", False)
    for layout in ("rcr", "rrr", "crr")
    if layout in results[result_key] and check_results
]
if quality_checks and not all(quality_checks):
    sys.exit(1)
