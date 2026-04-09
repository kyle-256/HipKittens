"""Benchmark: MXFP4 column-first KPAIR kernel."""
import json, math, os, random, sys
import torch

torch.manual_seed(0)
random.seed(0)

import tk_mxfp4_colwise

FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def parse_size(argv):
    if len(argv) == 2:
        n = int(argv[1]); return n, n, n
    if len(argv) == 4:
        return int(argv[1]), int(argv[2]), int(argv[3])
    return 8192, 8192, 8192


M, N, K = parse_size(sys.argv)
build_M = int(os.environ.get("MXFP4_BUILD_M", str(M)))
build_N = int(os.environ.get("MXFP4_BUILD_N", str(N)))
build_K = int(os.environ.get("MXFP4_BUILD_K", str(K)))
if build_M < M or build_N < N or build_K < K:
    raise ValueError(f"Build shape ({build_M},{build_N},{build_K}) < problem ({M},{N},{K})")
if K % 2 != 0:
    raise ValueError(f"K must be even (got {K})")

num_warmup = int(os.environ.get("MXFP4_WARMUP", "100"))
num_iters = int(os.environ.get("MXFP4_ITERS", "200"))
determinism_runs = max(1, int(os.environ.get("MXFP4_DETERMINISM_RUNS", "1")))
snr_threshold_db = float(os.environ.get("MXFP4_SNR_THRESHOLD_DB", "48.0"))
check = os.environ.get("MXFP4_CHECK", "1") != "0"
use_pq = os.environ.get("MXFP4_PRESHUFFLE_QUANT", "1") != "0"

k_blocks = (build_K + 31) // 32
flops_ref = 2 * M * N * K

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

print(f"=== MXFP4 Colwise KPAIR Kernel: M={M}, N={N}, K={K} ===")
if (build_M, build_N, build_K) != (M, N, K):
    print(f"Build shape={build_M}x{build_N}x{build_K}")
print(f"Warmup={num_warmup}, iters={num_iters}, check={check}, pq={use_pq}")
if check:
    print(f"SNR threshold={snr_threshold_db:.1f} dB, determinism_runs={determinism_runs}")


def generate_fp4_matrix(rows, cols_fp4):
    assert cols_fp4 % 2 == 0
    cols_bytes = cols_fp4 // 2
    lo = torch.randint(0, 16, (rows, cols_bytes), dtype=torch.uint8, device="cuda")
    hi = torch.randint(0, 16, (rows, cols_bytes), dtype=torch.uint8, device="cuda")
    return (hi << 4) | lo


def unpack_fp4(packed, K):
    lo = (packed & 0x0F).to(torch.int64)
    hi = ((packed >> 4) & 0x0F).to(torch.int64)
    lut = FP4_LUT.to(packed.device)
    out = torch.empty(packed.shape[0], K, dtype=torch.float32, device=packed.device)
    out[:, 0::2] = lut[lo]
    out[:, 1::2] = lut[hi]
    return out


def generate_scale_matrix(total_rows, total_k_blocks, valid_rows):
    s = torch.zeros(total_rows, total_k_blocks, dtype=torch.int8, device="cuda")
    s[:valid_rows, :] = torch.randint(-2, 3, (valid_rows, total_k_blocks),
                                       dtype=torch.int8, device="cuda")
    return s


def encode_scale_raw(scale_exp):
    raw = (scale_exp.to(torch.int16) + 127).to(torch.uint8)
    return torch.where(scale_exp == -128, torch.full_like(raw, 0xFF), raw)


def preshuffle_mfma16(scale_exp):
    rows, kb = scale_exp.shape
    pr = math.ceil(rows / 32) * 32
    pk = math.ceil(kb / 8) * 8
    raw = torch.full((pr, pk), 0x7F, dtype=torch.uint8, device=scale_exp.device)
    raw[:rows, :kb] = encode_scale_raw(scale_exp)
    sh = raw.view(pr // 32, 2, 16, pk // 8, 2, 4, 1)
    sh = sh.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    return sh.view(pr // 32, pk * 32)


def expand_scales(scale_exp, cols):
    return torch.pow(2.0, scale_exp.float()).repeat_interleave(32, dim=1)[:, :cols]


def compute_snr(C_test, C_ref):
    t, r = C_test.float(), C_ref.float()
    sig = (r * r).sum().item()
    noi = ((t - r) ** 2).sum().item()
    if noi == 0:
        return float("inf")
    if sig == 0:
        return float("-inf")
    return 10.0 * math.log10(sig / noi)


A = generate_fp4_matrix(build_M, build_K)
B = generate_fp4_matrix(build_N, build_K)
A_scale_exp = generate_scale_matrix(build_M, k_blocks, M)
B_scale_exp = generate_scale_matrix(build_N, k_blocks, N)
C = torch.zeros(build_M, build_N, dtype=torch.bfloat16, device="cuda")

if use_pq:
    A_scale = preshuffle_mfma16(A_scale_exp)
    B_scale = preshuffle_mfma16(B_scale_exp)
else:
    A_scale = A_scale_exp
    B_scale = B_scale_exp

run = lambda: tk_mxfp4_colwise.gemm_rcr(A, B, A_scale, B_scale, C)

for _ in range(num_warmup):
    C.zero_()
    run()
torch.cuda.synchronize()

start_event.record()
for _ in range(num_iters):
    C.zero_()
    run()
end_event.record()
torch.cuda.synchronize()

total_ms = start_event.elapsed_time(end_event)
avg_ms = total_ms / num_iters
tflops = flops_ref / (avg_ms * 1e9)
print(f"\nAvg: {avg_ms:.4f} ms, TFLOPS: {tflops:.2f}")

results = {"avg_ms": avg_ms, "tflops": tflops}

if check:
    C.zero_()
    run()
    torch.cuda.synchronize()

    A_ref = unpack_fp4(A[:M, :K // 2], K) * expand_scales(A_scale_exp[:M], K)
    B_ref = unpack_fp4(B[:N, :K // 2], K) * expand_scales(B_scale_exp[:N], K)
    C_ref = A_ref @ B_ref.T

    snr = compute_snr(C[:M, :N], C_ref)
    diff = (C[:M, :N].float() - C_ref.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    snr_ok = snr > snr_threshold_db

    print(f"SNR: {snr:.2f} dB (threshold {snr_threshold_db:.1f})")
    print(f"Max err: {max_err:.4f}, Mean err: {mean_err:.4f}")

    det_ok = True
    if determinism_runs > 1:
        C.zero_(); run(); torch.cuda.synchronize()
        ref_out = C[:M, :N].clone()
        for _ in range(determinism_runs - 1):
            C.zero_(); run(); torch.cuda.synchronize()
            if not torch.equal(C[:M, :N], ref_out):
                det_ok = False
                break
        print(f"Determinism ({determinism_runs} runs): {'PASS' if det_ok else 'FAIL'}")

    ok = snr_ok and det_ok
    print(f"Result: {'PASS' if ok else 'FAIL'}")
    results.update({"snr_db": snr, "deterministic": det_ok, "ok": ok})

suffix = "_pq" if use_pq else ""
outfile = os.environ.get("MXFP4_OUTPUT",
    f"mxfp4_colwise_results_{M}x{N}x{K}{suffix}.json")
with open(outfile, "w") as f:
    json.dump(results, f, indent=4)
print(f"Results saved to {outfile}")

if check and not results.get("ok", True):
    sys.exit(1)
