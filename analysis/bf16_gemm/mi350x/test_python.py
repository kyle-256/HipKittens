import torch
import random
import time
import sys
import subprocess
import os
import tk_kernel
from aiter.tuned_gemm import tgemm

torch.manual_seed(0)
random.seed(0)

# Inputs
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
scale = 1.0
A = torch.randn(N, N, dtype=torch.bfloat16, device='cuda')  / scale
B = torch.randn(N, N, dtype=torch.bfloat16, device='cuda')  / scale
Bt = B.t().contiguous()  # Transpose B for the kernel

filename = sys.argv[2]

num_warmup = 500
num_iters = 100

start_event = torch.cuda.Event(enable_timing=True) # in milliseconds
end_event = torch.cuda.Event(enable_timing=True)
flops_ref = (2 * N**3)  # FLOPs for reference

# Reference matmul using PyTorch

for _ in range(num_warmup):
    C_pytorch = torch.matmul(A, Bt)
timings_pytorch = []
torch.random.manual_seed(0)
random.seed(0)
for _ in range(num_iters):
    A = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / scale
    B = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / scale
    Bt = B.t().contiguous()  # Transpose B for the kernel
    torch.cuda.synchronize()
    start_event.record()
    C_pytorch = torch.matmul(A, Bt)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings_pytorch.append(elapsed_time)
print(f"{C_pytorch.dtype=}")
avg_time_pytorch = sum(timings_pytorch) / len(timings_pytorch)
tflops_pytorch = flops_ref / (avg_time_pytorch * 1e9) 
print(f"PyTorch reference average execution time: {avg_time_pytorch:.4f} ms")
print(f"PyTorch reference performance: {tflops_pytorch:.2f} TFLOPS for {N}x{N} matrix multiplication.\n")


# Reference matmul using AITER (AMD)
for _ in range(num_warmup):
    C_aiter = tgemm.mm(A, B, None, None, None)
timings_aiter = []
torch.random.manual_seed(0)
random.seed(0)
for _ in range(num_iters):
    A = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / scale
    B = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / scale
    Bt = B.t().contiguous()  # Transpose B for the kernel
    torch.cuda.synchronize()
    start_event.record()
    C_aiter = tgemm.mm(A, B, None, None, None)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings_aiter.append(elapsed_time)
print(f"{C_aiter.dtype=}")
avg_time_aiter = sum(timings_aiter) / len(timings_aiter)
tflops_aiter = flops_ref / (avg_time_aiter * 1e9) 
print(f"AITER (AMD) reference average execution time: {avg_time_aiter:.4f} ms")
print(f"AITER (AMD) reference performance: {tflops_aiter:.2f} TFLOPS for {N}x{N} matrix multiplication.\n")


# Kernel matmul
C = torch.zeros(N, N, dtype=torch.bfloat16, device='cuda')
for _ in range(num_warmup):
    tk_kernel.dispatch_micro(A, B, C)
timings = []
torch.random.manual_seed(0)
random.seed(0)
for _ in range(num_iters):
    A = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / scale
    B = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / scale
    Bt = B.t().contiguous()  # Transpose B for the kernel
    torch.cuda.synchronize()
    start_event.record()
    tk_kernel.dispatch_micro(A, B, C)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings.append(elapsed_time)
print(f"{C.dtype=}")
avg_time = sum(timings) / len(timings)
tflops = flops_ref / (avg_time * 1e9) 
print(f"Average execution time: {avg_time:.4f} ms")
print(f"Performance: {tflops:.2f} TFLOPS for {N}x{N} matrix multiplication.\n")


# Compare against reference
# BF16 GEMM tolerance: accumulation order differs between implementations,
# expected relative error ~ sqrt(K) * eps_bf16 ≈ sqrt(K) * 2^-8
import math
eps_bf16 = 2**-8
rtol = 2.0 * math.sqrt(N) * eps_bf16  # ~0.7 for N=8192
atol = 1.0  # absolute tolerance for values near zero

C_float = C.float()
C_pytorch_float = C_pytorch.float()
diff = (C_float - C_pytorch_float).abs()
scale = C_pytorch_float.abs().clamp(min=atol)
rel_diff = diff / scale
max_abs_error = diff.max().item()
mean_abs_error = diff.mean().item()
max_rel_error = rel_diff.max().item()
mean_rel_error = rel_diff.mean().item()
pass_count = ((diff <= atol) | (rel_diff <= rtol)).sum().item()
total = N * N
pass_rate = pass_count / total * 100

print(f"Correctness check (rtol={rtol:.4f}, atol={atol:.1f}):")
print(f"  Max  absolute error: {max_abs_error:.4f}")
print(f"  Mean absolute error: {mean_abs_error:.4f}")
print(f"  Max  relative error: {max_rel_error:.4f}")
print(f"  Mean relative error: {mean_rel_error:.4f}")
print(f"  Pass rate: {pass_count}/{total} ({pass_rate:.2f}%)")
print(f"  Result: {'PASS' if pass_rate >= 99.0 else 'FAIL'}\n")

############### LOGGING OUTPUTS ####################

data_to_log = {
    "avg_time_pytorch": avg_time_pytorch,
    "tflops_pytorch": tflops_pytorch,
    "avg_time_aiter": avg_time_aiter,
    "tflops_aiter": tflops_aiter,
    "avg_time": avg_time,
    "tflops": tflops,
}
import json
if not os.path.exists(filename):
    with open(filename, "w") as f:
        json.dump({}, f, indent=4)
with open(filename, "r") as f:
    data = json.load(f)
    data[str(N)] = data_to_log
with open(filename, "w") as f:
    json.dump(data, f, indent=4)
print(f"Results saved to {filename}")

############### END LOGGING OUTPUTS ###############

    