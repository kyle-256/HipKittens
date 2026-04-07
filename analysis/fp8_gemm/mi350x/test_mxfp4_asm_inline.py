"""Benchmark: MXFP4 inline ASM kernel (gluon-derived, C++ compiled)."""
import math, os, sys, torch
torch.manual_seed(0)

import tk_mxfp4_asm_inline

FP4_LUT = torch.tensor(
    [0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0],
    dtype=torch.float32,
)

def parse_size(argv):
    if len(argv) == 2: n = int(argv[1]); return n, n, n
    if len(argv) == 4: return int(argv[1]), int(argv[2]), int(argv[3])
    return 8192, 8192, 8192

M, N, K = parse_size(sys.argv)
num_warmup = int(os.environ.get("MXFP4_WARMUP", "100"))
num_iters  = int(os.environ.get("MXFP4_ITERS", "200"))
check      = os.environ.get("MXFP4_CHECK", "1") != "0"

print(f"=== MXFP4 inline ASM (C++ compiled): M={M}, N={N}, K={K} ===")
print(f"Warmup={num_warmup}, iters={num_iters}, check={check}")

k_blocks = K // 32
A = ((torch.randint(0,16,(M,K//2),dtype=torch.uint8,device="cuda") << 4) |
      torch.randint(0,16,(M,K//2),dtype=torch.uint8,device="cuda"))
B = ((torch.randint(0,16,(N,K//2),dtype=torch.uint8,device="cuda") << 4) |
      torch.randint(0,16,(N,K//2),dtype=torch.uint8,device="cuda"))

sc_exp_a = torch.randint(-2, 3, (M, k_blocks), dtype=torch.int8, device="cuda")
sc_exp_b = torch.randint(-2, 3, (N, k_blocks), dtype=torch.int8, device="cuda")
raw_a = (sc_exp_a.to(torch.int16) + 127).to(torch.uint8)
raw_b = (sc_exp_b.to(torch.int16) + 127).to(torch.uint8)

# Gluon expects column-major: store as (k_blocks, rows) contiguous
A_sc = raw_a.T.contiguous()   # (k_blocks, M) contiguous
B_sc = raw_b.T.contiguous()   # (k_blocks, N) contiguous
C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

run = lambda: tk_mxfp4_asm_inline.gemm_rcr(A, B, A_sc, B_sc, C)
run(); torch.cuda.synchronize()

for _ in range(num_warmup): run()
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(num_iters): run()
end.record(); torch.cuda.synchronize()

avg_ms = start.elapsed_time(end) / num_iters
tflops = 2 * M * N * K / (avg_ms * 1e9)
print(f"Avg: {avg_ms:.4f} ms, TFLOPS: {tflops:.2f}")

if check:
    def unpack(p, K):
        lo = (p & 0x0F).to(torch.int64); hi = ((p >> 4) & 0x0F).to(torch.int64)
        lut = FP4_LUT.cuda()
        out = torch.empty(p.shape[0], K, dtype=torch.float32, device="cuda")
        out[:, 0::2] = lut[lo]; out[:, 1::2] = lut[hi]
        return out
    def expand(exp, K):
        return torch.pow(2.0, exp.float()).repeat_interleave(32, dim=1)[:, :K]
    C.zero_(); run(); torch.cuda.synchronize()
    Af = unpack(A, K) * expand(sc_exp_a, K)
    Bf = unpack(B, K) * expand(sc_exp_b, K)
    Cref = Af @ Bf.T
    noise = C[:M,:N].float() - Cref.float()
    sig = (Cref.float()**2).sum().item()
    noi = (noise**2).sum().item()
    snr = 10 * math.log10(sig / noi) if noi > 0 else float("inf")
    print(f"SNR: {snr:.2f} dB, Max err: {noise.abs().max().item():.4f}")
    print(f"Result: {'PASS' if snr > 48 else 'FAIL'}")
