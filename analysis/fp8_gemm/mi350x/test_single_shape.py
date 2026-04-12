"""Test a single shape with high precision timing."""
import math, os, sys, torch
torch.manual_seed(0)

# Parse args
M, N, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
comp = float(sys.argv[4]) if len(sys.argv) > 4 else 0

# Import correct module (compiled for this N,K)
module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}"
sys.path.insert(0, "build_all42")
mod = __import__(module_name)

k_blocks = K // 32
WARMUP = 200
ITERS = 500

def gen_fp4(rows, K):
    cols = K // 2
    lo = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device="cuda")
    hi = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device="cuda")
    return (hi << 4) | lo

def preshuffle_mfma16(scale_exp):
    rows, kb = scale_exp.shape
    pr = math.ceil(rows / 32) * 32
    pk = math.ceil(kb / 8) * 8
    raw = torch.full((pr, pk), 0x7F, dtype=torch.uint8, device=scale_exp.device)
    raw[:rows, :kb] = (scale_exp.to(torch.int16) + 127).to(torch.uint8)
    sh = raw.view(pr // 32, 2, 16, pk // 8, 2, 4, 1)
    sh = sh.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    return sh.view(pr // 32, pk * 32)

A = gen_fp4(M, K)
B = gen_fp4(N, K)
sc_exp_a = torch.randint(-2, 3, (M, k_blocks), dtype=torch.int8, device="cuda")
sc_exp_b = torch.randint(-2, 3, (N, k_blocks), dtype=torch.int8, device="cuda")
A_sc = preshuffle_mfma16(sc_exp_a)
B_sc = preshuffle_mfma16(sc_exp_b)
C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

run = lambda: mod.gemm_rcr(A, B, A_sc, B_sc, C)

for _ in range(WARMUP): run()
torch.cuda.synchronize()

times = []
for _ in range(ITERS):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record(); run(); e.record()
    torch.cuda.synchronize()
    times.append(s.elapsed_time(e))

times.sort()
trim = int(len(times) * 0.10)
trimmed = times[trim:-trim] if trim > 0 else times
avg = sum(trimmed) / len(trimmed)
tflops = 2.0 * M * N * K / (avg * 1e-3) / 1e12
ratio = tflops / comp * 100 if comp > 0 else 0
tag = "WIN" if tflops >= comp else "LOSE"
print(f"{M}x{N}x{K}: {tflops:.1f} vs {comp:.1f} ({ratio:.1f}%) {tag}  [avg={avg:.4f}ms, {ITERS} iters]")
