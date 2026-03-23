import torch
import random
import time
import sys
import os
import json
import math

torch.manual_seed(0)
random.seed(0)

import tk_fp8_layouts

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8192

num_warmup = int(os.environ.get("FP8_WARMUP", "500"))
num_iters = int(os.environ.get("FP8_ITERS", "100"))
target_pct = float(os.environ.get("FP8_TARGET_PCT", "98.0"))
requested_layouts = {
    layout.strip().lower()
    for layout in os.environ.get("FP8_LAYOUTS", "rcr,rrr,crr").split(",")
    if layout.strip()
}
check_results = os.environ.get("FP8_CHECK", "1") != "0"

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
flops_ref = 2 * N**3

def generate_fp8_matrix(rows, cols):
    x = torch.randn(rows, cols, dtype=torch.float32, device="cuda") * 0.1
    return x.to(torch.float8_e4m3fn)

def benchmark_kernel(fn, warmup=num_warmup, iters=num_iters):
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event))
    return timings

def check_correctness(C_test, C_ref, label):
    C_test_f32 = C_test.float()
    C_ref_f32 = C_ref.float()
    diff = (C_test_f32 - C_ref_f32).abs()
    scale = C_ref_f32.abs().clamp(min=1.0)
    rel_diff = diff / scale
    eps_fp8 = 2**-3
    rtol = 4.0 * math.sqrt(N) * eps_fp8
    atol = 2.0
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = rel_diff.max().item()
    mean_rel = rel_diff.mean().item()
    pass_count = ((diff <= atol) | (rel_diff <= rtol)).sum().item()
    total = C_test.numel()
    pass_rate = pass_count / total * 100
    print(f"  [{label}] Correctness (rtol={rtol:.4f}, atol={atol:.1f}):")
    print(f"    Max  abs error: {max_abs:.4f}, Mean abs error: {mean_abs:.4f}")
    print(f"    Max  rel error: {max_rel:.4f}, Mean rel error: {mean_rel:.4f}")
    print(f"    Pass rate: {pass_count}/{total} ({pass_rate:.2f}%)")
    ok = pass_rate >= 99.0
    result_str = 'PASS' if ok else 'FAIL'
    print(f"    Result: {result_str}")
    return ok

print(f"=== FP8 GEMM Layout Benchmark (native layouts, no preshuffle): M=N=K={N} ===")
print(f"Warmup={num_warmup}, iters={num_iters}, correctness={'on' if check_results else 'off'}")
print(f"Layouts={','.join(sorted(requested_layouts))}\n")

results = {str(N): {}}

if "rcr" in requested_layouts:
    print("--- RCR Layout: C = A @ B^T ---")
    A_rcr = generate_fp8_matrix(N, N)
    Bt_rcr = generate_fp8_matrix(N, N)
    C_rcr = torch.zeros(N, N, dtype=torch.bfloat16, device="cuda")

    timings_rcr = benchmark_kernel(lambda: tk_fp8_layouts.gemm_rcr(A_rcr, Bt_rcr, C_rcr))
    avg_rcr = sum(timings_rcr) / len(timings_rcr)
    tflops_rcr = flops_ref / (avg_rcr * 1e9)
    print(f"  Avg time: {avg_rcr:.4f} ms, TFLOPS: {tflops_rcr:.2f}")

    if check_results:
        C_ref_rcr = A_rcr.float() @ Bt_rcr.float().T
        check_correctness(C_rcr, C_ref_rcr, "RCR")
    print()
    results[str(N)]["rcr"] = {"avg_ms": avg_rcr, "tflops": tflops_rcr}
else:
    avg_rcr = None
    tflops_rcr = None

if "rrr" in requested_layouts:
    print("--- RRR Layout: C = A @ B ---")
    A_rrr = generate_fp8_matrix(N, N)
    B_rrr = generate_fp8_matrix(N, N)
    C_rrr = torch.zeros(N, N, dtype=torch.bfloat16, device="cuda")

    timings_rrr = benchmark_kernel(lambda: tk_fp8_layouts.gemm_rrr(A_rrr, B_rrr, C_rrr))
    avg_rrr = sum(timings_rrr) / len(timings_rrr)
    tflops_rrr = flops_ref / (avg_rrr * 1e9)
    print(f"  Avg time: {avg_rrr:.4f} ms, TFLOPS: {tflops_rrr:.2f}")

    if check_results:
        C_ref_rrr = A_rrr.float() @ B_rrr.float()
        check_correctness(C_rrr, C_ref_rrr, "RRR")
    print()
    results[str(N)]["rrr"] = {"avg_ms": avg_rrr, "tflops": tflops_rrr}
else:
    avg_rrr = None
    tflops_rrr = None

if "crr" in requested_layouts:
    print("--- CRR Layout: C = A^T @ B ---")
    At_crr = generate_fp8_matrix(N, N)
    B_crr  = generate_fp8_matrix(N, N)
    C_crr  = torch.zeros(N, N, dtype=torch.bfloat16, device="cuda")

    timings_crr = benchmark_kernel(lambda: tk_fp8_layouts.gemm_crr(At_crr, B_crr, C_crr))
    avg_crr = sum(timings_crr) / len(timings_crr)
    tflops_crr = flops_ref / (avg_crr * 1e9)
    print(f"  Avg time: {avg_crr:.4f} ms, TFLOPS: {tflops_crr:.2f}")

    if check_results:
        C_ref_crr = At_crr.float().T @ B_crr.float()
        check_correctness(C_crr, C_ref_crr, "CRR")
    print()
    results[str(N)]["crr"] = {"avg_ms": avg_crr, "tflops": tflops_crr}
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
        results[str(N)]["rrr"]["pct_of_rcr"] = pct_rrr
    else:
        pct_rrr = None
    if avg_crr is not None:
        pct_crr = tflops_crr / tflops_rcr * 100 if tflops_rcr else 0
        print(f"{'CRR':<10} {avg_crr:<12.4f} {tflops_crr:<10.2f} {pct_crr:.2f}%")
        results[str(N)]["crr"]["pct_of_rcr"] = pct_crr
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
    results[str(N)]["target_met"] = target_met

outfile = f"fp8_layout_results_{N}.json"
with open(outfile, "w") as f:
    json.dump(results, f, indent=4)
print(f"\nResults saved to {outfile}")
