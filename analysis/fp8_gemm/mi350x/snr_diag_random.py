#!/usr/bin/env python3
"""SNR diagnostic with RANDOM scales (the methodology that earlier got 47.37 dB).

Tests one shape (16384x4096x2048) with both constant and random scales
to determine which methodology is correct.
"""
import os, sys, math, importlib, importlib.util, time, gc
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR  = os.path.join(SCRIPT_DIR, "build_all42")
sys.path.insert(0, BUILD_DIR)

FP4_E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0,
], dtype=torch.float32)

def dequant_fp4(packed_uint8, K):
    lo = (packed_uint8 & 0x0F).to(torch.int64)
    hi = ((packed_uint8 >> 4) & 0x0F).to(torch.int64)
    table = FP4_E2M1_TABLE.to(packed_uint8.device)
    rows, cols = packed_uint8.shape
    out = torch.empty(rows, cols * 2, dtype=torch.float32, device=packed_uint8.device)
    out[:, 0::2] = table[lo]
    out[:, 1::2] = table[hi]
    return out[:, :K]

def apply_block_scales(data_f32, scale_exp_i8, block_size=32):
    rows, K_dim = data_f32.shape
    k_blocks = K_dim // block_size
    scales = (2.0 ** scale_exp_i8.to(torch.float32))
    scales_expanded = scales.unsqueeze(-1).expand(rows, k_blocks, block_size)
    return scales_expanded.reshape(rows, -1)[:, :K_dim] * data_f32

def gen_fp4(rows, K):
    cols = K // 2
    lo = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device='cuda')
    hi = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device='cuda')
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

def load_module(mod_name):
    so_path = os.path.join(BUILD_DIR, f"{mod_name}.cpython-310-x86_64-linux-gnu.so")
    if not os.path.exists(so_path):
        return None
    spec = importlib.util.spec_from_file_location(mod_name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def torch_ref(A, B, sa, sb, K):
    A_f = dequant_fp4(A, K)
    B_f = dequant_fp4(B, K)
    A_s = apply_block_scales(A_f, sa)
    B_s = apply_block_scales(B_f, sb)
    return torch.matmul(A_s, B_s.T).to(torch.bfloat16)

def snr_double(ref, test, mask=None):
    rd = ref.double(); td = test.double()
    if mask is None:
        mask = torch.isfinite(rd) & torch.isfinite(td)
    else:
        mask = mask & torch.isfinite(rd) & torch.isfinite(td)
    n = mask.sum().item()
    if n == 0: return float('-inf'), float('inf'), 0
    rv = rd[mask]; tv = td[mask]; diff = tv - rv
    sig = (rv ** 2).mean().item()
    err = (diff ** 2).mean().item()
    snr = 10*math.log10(sig/err) if (sig > 0 and err > 0) else float('-inf')
    return snr, diff.abs().max().item(), n

def test_shape(mod, M, N, K, scale_mode, n_runs=5):
    """scale_mode: 'const_-4', 'rand_-2_3', 'rand_-1_2', 'zero'"""
    k_blocks = K // 32
    torch.manual_seed(42)
    A = gen_fp4(M, K); B = gen_fp4(N, K)
    if scale_mode == 'const_-4':
        sa = torch.full((M, k_blocks), -4, dtype=torch.int8, device='cuda')
        sb = torch.full((N, k_blocks), -4, dtype=torch.int8, device='cuda')
    elif scale_mode == 'rand_-2_3':
        sa = torch.randint(-2, 3, (M, k_blocks), dtype=torch.int8, device='cuda')
        sb = torch.randint(-2, 3, (N, k_blocks), dtype=torch.int8, device='cuda')
    elif scale_mode == 'rand_-1_2':
        sa = torch.randint(-1, 2, (M, k_blocks), dtype=torch.int8, device='cuda')
        sb = torch.randint(-1, 2, (N, k_blocks), dtype=torch.int8, device='cuda')
    elif scale_mode == 'zero':
        sa = torch.zeros((M, k_blocks), dtype=torch.int8, device='cuda')
        sb = torch.zeros((N, k_blocks), dtype=torch.int8, device='cuda')
    A_sc = preshuffle_mfma16_merged(sa)
    B_sc = preshuffle_mfma16_merged(sb)

    C_ref = torch_ref(A, B, sa, sb, K)
    ref_finite = torch.isfinite(C_ref.float()).float().mean().item()

    runs = []
    for _ in range(n_runs):
        C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
        mod.gemm_rcr(A, B, A_sc, B_sc, C)
        torch.cuda.synchronize()
        runs.append(C.view(torch.int16).clone())
    stack = torch.stack(runs, 0)
    agree = (stack == stack[0:1]).all(dim=0)
    det_frac = agree.float().mean().item()
    C0 = runs[0].view(torch.bfloat16)
    finite_frac = torch.isfinite(C0.float()).float().mean().item()

    snr_all,   md_all,   n_all   = snr_double(C_ref, C0)
    snr_det,   md_det,   n_det   = snr_double(C_ref, C0, mask=agree)

    print(f"  scale={scale_mode:10s}  ref_finite={ref_finite*100:5.1f}%  "
          f"kernel_finite={finite_frac*100:5.1f}%  det={det_frac*100:5.1f}%  "
          f"SNR_all={snr_all:8.2f} dB ({n_all:>10d})  "
          f"SNR_det={snr_det:8.2f} dB ({n_det:>10d})  "
          f"max|diff|_det={md_det:.4g}")
    return snr_det

# Test with multiple module/shape combos to find one that loads and works
candidates = [
    ("tk_mxfp4_gluon_cpp_n4096_k2048_ext_br", 4096, 4096, 2048),
    ("tk_mxfp4_gluon_cpp_n4096_k2048_f34", 4096, 4096, 2048),
]
for mod_name, M, N, K in candidates:
    print(f"\n=== {mod_name} on {M}x{N}x{K} ===")
    mod = load_module(mod_name)
    if mod is None:
        print(f"  MISSING")
        continue
    for mode in ['zero', 'const_-4', 'rand_-1_2', 'rand_-2_3']:
        try:
            test_shape(mod, M, N, K, mode)
        except Exception as e:
            print(f"  scale={mode}: ERROR: {e}")
