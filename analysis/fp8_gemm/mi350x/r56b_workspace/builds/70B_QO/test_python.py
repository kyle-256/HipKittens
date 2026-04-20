import json
import math
import os
import random
import sys

import torch

torch.manual_seed(0)
random.seed(0)

import tk_fp8_layouts


def parse_problem_size(argv):
    if len(argv) == 2:
        n = int(argv[1])
        return n, n, n
    if len(argv) == 4:
        return int(argv[1]), int(argv[2]), int(argv[3])
    return 8192, 8192, 8192


M, N, K = parse_problem_size(sys.argv)
build_M = int(os.environ.get("FP8_BUILD_M", str(M)))
build_N = int(os.environ.get("FP8_BUILD_N", str(N)))
build_K = int(os.environ.get("FP8_BUILD_K", str(K)))

if build_M < M or build_N < N or build_K < K:
    raise ValueError(
        f"Build shape ({build_M}, {build_N}, {build_K}) must cover "
        f"problem shape ({M}, {N}, {K})"
    )

num_warmup = int(os.environ.get("FP8_WARMUP", "500"))
num_iters = int(os.environ.get("FP8_ITERS", "100"))
target_pct = float(os.environ.get("FP8_TARGET_PCT", "95.0"))
determinism_runs = max(1, int(os.environ.get("FP8_DETERMINISM_RUNS", "1")))
snr_threshold_db = float(os.environ.get("FP8_SNR_THRESHOLD_DB", "48.0"))
requested_layouts = {
    layout.strip().lower()
    for layout in os.environ.get("FP8_LAYOUTS", "rcr,rrr,crr").split(",")
    if layout.strip()
}
check_results = os.environ.get("FP8_CHECK", "1") != "0"

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
flops_ref = 2 * M * N * K


def default_output_path():
    if M == N == K:
        return f"fp8_layout_results_{M}.json"
    return f"fp8_layout_results_{M}x{N}x{K}.json"


def generate_fp8_matrix(total_rows, total_cols, valid_rows, valid_cols):
    x = torch.zeros(total_rows, total_cols, dtype=torch.float32, device="cuda")
    x[:valid_rows, :valid_cols] = (
        torch.randn(valid_rows, valid_cols, dtype=torch.float32, device="cuda") * 0.1
    )
    return x.to(torch.float8_e4m3fn)

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
    eps_fp8 = 2 ** -3
    rtol = 4.0 * math.sqrt(max(M, N, K)) * eps_fp8
    atol = 2.0
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
    result_str = 'PASS' if ok else 'FAIL'
    print(f"    Result: {result_str}")
    return {
        "correctness_ok": pass_rate >= 99.0,
        "snr_db": snr_db,
        "snr_ok": snr_ok,
        "ok": ok,
    }

print(
    f"=== FP8 GEMM Layout Benchmark (native layouts, no preshuffle): "
    f"M={M}, N={N}, K={K} ==="
)
if (build_M, build_N, build_K) != (M, N, K):
    print(f"Build shape={build_M}x{build_N}x{build_K} (zero-padded)")
print(f"Warmup={num_warmup}, iters={num_iters}, correctness={'on' if check_results else 'off'}")
if check_results:
    print(f"SNR threshold={snr_threshold_db:.1f} dB, determinism_runs={determinism_runs}")
print(f"Layouts={','.join(sorted(requested_layouts))}\n")

result_key = f"{M}x{N}x{K}"
results = {
    result_key: {
        "build_M": build_M,
        "build_N": build_N,
        "build_K": build_K,
    }
}

if "rcr" in requested_layouts:
    print("--- RCR Layout: C = A @ B^T ---")
    A_rcr = generate_fp8_matrix(build_M, build_K, M, K)
    Bt_rcr = generate_fp8_matrix(build_N, build_K, N, K)
    C_rcr = torch.zeros(build_M, build_N, dtype=torch.bfloat16, device="cuda")

    run_rcr = lambda: tk_fp8_layouts.gemm_rcr(A_rcr, Bt_rcr, C_rcr, 1.0)
    timings_rcr = benchmark_kernel(run_rcr, C_rcr)
    avg_rcr = sum(timings_rcr) / len(timings_rcr)
    tflops_rcr = flops_ref / (avg_rcr * 1e9)
    print(f"  Avg time: {avg_rcr:.4f} ms, TFLOPS: {tflops_rcr:.2f}")

    if check_results:
        C_ref_rcr = A_rcr[:M, :K].float() @ Bt_rcr[:N, :K].float().T
        quality_rcr = check_correctness(C_rcr[:M, :N], C_ref_rcr, "RCR")
        det_ok_rcr, det_max_abs_rcr = check_determinism(run_rcr, C_rcr, "RCR")
    print()
    results[result_key]["rcr"] = {"avg_ms": avg_rcr, "tflops": tflops_rcr}
    if check_results:
        results[result_key]["rcr"].update(
            {
                "snr_db": quality_rcr["snr_db"],
                "deterministic": det_ok_rcr,
                "determinism_max_abs_diff": det_max_abs_rcr,
                "quality_ok": quality_rcr["ok"] and det_ok_rcr,
            }
        )
else:
    avg_rcr = None
    tflops_rcr = None

if "rrr" in requested_layouts:
    print("--- RRR Layout: C = A @ B ---")
    A_rrr = generate_fp8_matrix(build_M, build_K, M, K)
    B_rrr = generate_fp8_matrix(build_K, build_N, K, N)
    C_rrr = torch.zeros(build_M, build_N, dtype=torch.bfloat16, device="cuda")

    run_rrr = lambda: tk_fp8_layouts.gemm_rrr(A_rrr, B_rrr, C_rrr, 1.0)
    timings_rrr = benchmark_kernel(run_rrr, C_rrr)
    avg_rrr = sum(timings_rrr) / len(timings_rrr)
    tflops_rrr = flops_ref / (avg_rrr * 1e9)
    print(f"  Avg time: {avg_rrr:.4f} ms, TFLOPS: {tflops_rrr:.2f}")

    if check_results:
        C_ref_rrr = A_rrr[:M, :K].float() @ B_rrr[:K, :N].float()
        quality_rrr = check_correctness(C_rrr[:M, :N], C_ref_rrr, "RRR")
        det_ok_rrr, det_max_abs_rrr = check_determinism(run_rrr, C_rrr, "RRR")
    print()
    results[result_key]["rrr"] = {"avg_ms": avg_rrr, "tflops": tflops_rrr}
    if check_results:
        results[result_key]["rrr"].update(
            {
                "snr_db": quality_rrr["snr_db"],
                "deterministic": det_ok_rrr,
                "determinism_max_abs_diff": det_max_abs_rrr,
                "quality_ok": quality_rrr["ok"] and det_ok_rrr,
            }
        )
else:
    avg_rrr = None
    tflops_rrr = None

if "crr" in requested_layouts:
    print("--- CRR Layout: C = A^T @ B ---")
    At_crr = generate_fp8_matrix(build_K, build_M, K, M)
    B_crr = generate_fp8_matrix(build_K, build_N, K, N)
    C_crr = torch.zeros(build_M, build_N, dtype=torch.bfloat16, device="cuda")

    run_crr = lambda: tk_fp8_layouts.gemm_crr(At_crr, B_crr, C_crr, 1.0)
    timings_crr = benchmark_kernel(run_crr, C_crr)
    avg_crr = sum(timings_crr) / len(timings_crr)
    tflops_crr = flops_ref / (avg_crr * 1e9)
    print(f"  Avg time: {avg_crr:.4f} ms, TFLOPS: {tflops_crr:.2f}")

    if check_results:
        C_ref_crr = At_crr[:K, :M].float().T @ B_crr[:K, :N].float()
        quality_crr = check_correctness(C_crr[:M, :N], C_ref_crr, "CRR")
        det_ok_crr, det_max_abs_crr = check_determinism(run_crr, C_crr, "CRR")
    print()
    results[result_key]["crr"] = {"avg_ms": avg_crr, "tflops": tflops_crr}
    if check_results:
        results[result_key]["crr"].update(
            {
                "snr_db": quality_crr["snr_db"],
                "deterministic": det_ok_crr,
                "determinism_max_abs_diff": det_max_abs_crr,
                "quality_ok": quality_crr["ok"] and det_ok_crr,
            }
        )
else:
    avg_crr = None
    tflops_crr = None

if requested_layouts:
    print("=" * 60)
    print(f"{'Layout':<10} {'Time (ms)':<12} {'TFLOPS':<10} {'% of RCR':<10}")
    print("-" * 60)
    if avg_rcr is not None:
        print(f"{'RCR':<10} {avg_rcr:<12.4f} {tflops_rcr:<10.2f} {'100.00%':<10}")
    if avg_rrr is not None:
        pct_rrr = tflops_rrr / tflops_rcr * 100 if tflops_rcr else 0
        print(f"{'RRR':<10} {avg_rrr:<12.4f} {tflops_rrr:<10.2f} {pct_rrr:.2f}%")
        results[result_key]["rrr"]["pct_of_rcr"] = pct_rrr
    else:
        pct_rrr = None
    if avg_crr is not None:
        pct_crr = tflops_crr / tflops_rcr * 100 if tflops_rcr else 0
        print(f"{'CRR':<10} {avg_crr:<12.4f} {tflops_crr:<10.2f} {pct_crr:.2f}%")
        results[result_key]["crr"]["pct_of_rcr"] = pct_crr
    else:
        pct_crr = None
    print("=" * 60)

    target_values = []
    if pct_rrr is not None:
        target_values.append(pct_rrr)
    if pct_crr is not None:
        target_values.append(pct_crr)
    target_met = bool(target_values) and all(pct >= target_pct for pct in target_values)
    if pct_rrr is not None or pct_crr is not None:
        print(f"\n{target_pct:.0f}% target: {'MET' if target_met else 'NOT MET'}")
        if pct_rrr is not None:
            print(f"  RRR: {pct_rrr:.2f}% of RCR")
        if pct_crr is not None:
            print(f"  CRR: {pct_crr:.2f}% of RCR")
    results[result_key]["target_met"] = target_met

quality_checks = []
for layout in ("rcr", "rrr", "crr"):
    if layout in results[result_key] and check_results:
        quality_checks.append(results[result_key][layout].get("quality_ok", False))

success_gate = all(quality_checks) if quality_checks else True
results[result_key]["success_gate"] = success_gate

outfile = os.environ.get("FP8_OUTPUT", default_output_path())
with open(outfile, "w") as f:
    json.dump(results, f, indent=4)
print(f"\nResults saved to {outfile}")

if not success_gate:
    sys.exit(1)
