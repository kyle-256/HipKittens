"""Test: MXFP4 ART kernel — correctness check against reference."""
import math, os, sys, torch
torch.manual_seed(0)

import tk_mxfp4_art

FP4_LUT = torch.tensor(
    [0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0],
    dtype=torch.float32,
)

def parse_size(argv):
    if len(argv) == 2: n = int(argv[1]); return n, n, n
    if len(argv) == 4: return int(argv[1]), int(argv[2]), int(argv[3])
    return 8192, 8192, 8192

M, N, K = parse_size(sys.argv)
check = True

k_blocks = K // 32

print(f"=== MXFP4 ART kernel: M={M}, N={N}, K={K} ===")

def gen_fp4(rows, K):
    cols = K // 2
    lo = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device="cuda")
    hi = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device="cuda")
    return (hi << 4) | lo

def preshuffle_mfma16_merged(scale_exp):
    rows, kb = scale_exp.shape
    pr = math.ceil(rows / 64) * 64
    pk = math.ceil(kb / 8) * 8
    raw = torch.full((pr, pk), 0x7F, dtype=torch.uint8, device=scale_exp.device)
    raw[:rows, :kb] = (scale_exp.to(torch.int16) + 127).to(torch.uint8)
    sh = raw.view(pr // 32, 2, 16, pk // 8, 2, 4, 1)
    sh = sh.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    sh = sh.view(pr // 32, pk * 32)
    sh = sh.view(pr // 64, 2, pk * 32 // 4, 4)
    sh = sh.permute(0, 2, 1, 3).contiguous()
    return sh.view(pr // 64, pk * 64)

A = gen_fp4(M, K)
B = gen_fp4(N, K)

sc_exp_a = torch.randint(-2, 3, (M, k_blocks), dtype=torch.int8, device="cuda")
sc_exp_b = torch.randint(-2, 3, (N, k_blocks), dtype=torch.int8, device="cuda")
A_sc = preshuffle_mfma16_merged(sc_exp_a)
B_sc = preshuffle_mfma16_merged(sc_exp_b)
C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

run = lambda: tk_mxfp4_art.gemm_rcr(A, B, A_sc, B_sc, C)

# Run once
C.zero_(); run(); torch.cuda.synchronize()

if check:
    def unpack(p, K):
        lo = (p & 0x0F).to(torch.int64); hi = ((p >> 4) & 0x0F).to(torch.int64)
        lut = FP4_LUT.cuda()
        out = torch.empty(p.shape[0], K, dtype=torch.float32, device="cuda")
        out[:, 0::2] = lut[lo]; out[:, 1::2] = lut[hi]
        return out
    def expand(exp, K):
        return torch.pow(2.0, exp.float()).repeat_interleave(32, dim=1)[:, :K]

    Af = unpack(A, K) * expand(sc_exp_a, K)
    Bf = unpack(B, K) * expand(sc_exp_b, K)
    Cref = Af @ Bf.T
    noise = C.float() - Cref.float()
    sig = (Cref.float()**2).sum().item()
    noi = (noise**2).sum().item()
    snr = 10*math.log10(sig/noi) if noi > 0 else float("inf")
    max_err = noise.abs().max().item()
    print(f"SNR: {snr:.2f} dB, Max err: {max_err:.4f}")

    # Check if output is all zeros (common failure mode)
    nonzero_frac = (C != 0).float().mean().item()
    print(f"Non-zero fraction: {nonzero_frac:.4f}")

    # Determinism check
    C.zero_(); run(); torch.cuda.synchronize()
    ref = C.clone()
    C.zero_(); run(); torch.cuda.synchronize()
    det_ok = torch.equal(C, ref)
    print(f"Determinism: {'PASS' if det_ok else 'FAIL'}")

    ok = snr > 48 and det_ok
    print(f"Result: {'PASS' if ok else 'FAIL'}")
    if not ok: sys.exit(1)
