"""R33 Dev A — Task 2 Stage A2c correctness test for Path 1 rect kernel.

Builds a CRR V2 input + scales (square preshuffle, unchanged), runs both the
SQUARE V2-CRR fastpath and the PATH-1 RECT kernel, compares to reference.

Env: M, N, K, SO_SQUARE, SO_RECT, PHYS_GPU
"""
import math, os, importlib.util, sys
import torch

torch.manual_seed(0)

M = int(os.environ.get('M', '4096'))
N = int(os.environ.get('N', '1024'))
K = int(os.environ.get('K', '8192'))
phys_gpu = os.environ.get('PHYS_GPU', '0')
so_square = os.environ['SO_SQUARE']
so_rect = os.environ['SO_RECT']

def load(modname, sopath):
    spec = importlib.util.spec_from_file_location(modname, sopath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod_square = load('tk_mxfp8_r33a_task2_square', so_square)
mod_rect = load('tk_mxfp8_r33a_task2_rect', so_rect)

def gen_fp8(r, c):
    x = torch.randn(r, c, dtype=torch.float32, device='cuda') * 0.05
    return x.to(torch.float8_e4m3fn)

def gen_scale(r, kb):
    return torch.randint(-2, 3, (r, kb), dtype=torch.int8, device='cuda')

def encode_raw(s):
    raw = (s.to(torch.int16) + 127).to(torch.uint8)
    return torch.where(s == -128, torch.full_like(raw, 0xFF), raw)

def preshuffle_v2_a(scale_exp, blk=256, hb=128, rbm=64, warps_m=2):
    pack_count = 4
    rows, kb = scale_exp.shape
    padded_rows = math.ceil(rows / blk) * blk
    padded_kb = math.ceil(kb / 8) * 8
    num_ctiles = padded_rows // blk
    num_slabs = num_ctiles * warps_m
    rgs_per_ctile = blk // 32
    pack_a = rbm // 32
    raw = torch.full((padded_rows, padded_kb), 0x7F, dtype=torch.uint8, device=scale_exp.device)
    raw[:rows, :kb] = encode_raw(scale_exp)
    perm = torch.empty_like(raw)
    rg_view = raw.view(num_ctiles, rgs_per_ctile, 32, padded_kb)
    perm_view = perm.view(num_ctiles, num_slabs // num_ctiles, pack_count, 32, padded_kb)
    for wm in range(warps_m):
        rg_base = wm * (rbm // 32)
        rg_hi = (hb // 32) + rg_base
        for pidx in range(pack_a):
            perm_view[:, wm, 2*pidx, :, :] = rg_view[:, rg_base + pidx, :, :]
            perm_view[:, wm, 2*pidx + 1, :, :] = rg_view[:, rg_hi + pidx, :, :]
    kp_count = padded_kb // 8
    rv = perm.view(num_slabs, pack_count, 2, 16, kp_count, 2, 4)
    sh = rv.permute(0, 4, 6, 3, 1, 5, 2).contiguous()
    return sh.view(num_slabs, pack_count * 32 * padded_kb)

def preshuffle_v2_b(scale_exp, blk=256, hb=128, rbn=32, warps_n=4):
    pack_count = 2
    rows, kb = scale_exp.shape
    padded_rows = math.ceil(rows / blk) * blk
    padded_kb = math.ceil(kb / 8) * 8
    num_ctiles = padded_rows // blk
    num_slabs = num_ctiles * warps_n
    rgs_per_ctile = blk // 32
    raw = torch.full((padded_rows, padded_kb), 0x7F, dtype=torch.uint8, device=scale_exp.device)
    raw[:rows, :kb] = encode_raw(scale_exp)
    perm = torch.empty_like(raw)
    rg_view = raw.view(num_ctiles, rgs_per_ctile, 32, padded_kb)
    perm_view = perm.view(num_ctiles, warps_n, pack_count, 32, padded_kb)
    rbn_rg = rbn // 32
    rg_hi_off = hb // 32
    for wn in range(warps_n):
        rg_base = wn * rbn_rg
        perm_view[:, wn, 0, :, :] = rg_view[:, rg_base, :, :]
        perm_view[:, wn, 1, :, :] = rg_view[:, rg_base + rg_hi_off, :, :]
    kp_count = padded_kb // 8
    rv = perm.view(num_slabs, pack_count, 2, 16, kp_count, 2, 4)
    sh = rv.permute(0, 4, 6, 3, 1, 5, 2).contiguous()
    return sh.view(num_slabs, pack_count * 32 * padded_kb)

def expand_col(s, rows):
    return torch.pow(2.0, s.float()).transpose(0, 1).repeat_interleave(32, dim=0)[:rows, :]

k_blocks = (K + 31) // 32
# CRR layout: A=(K,M), B=(K,N)
A = gen_fp8(K, M); B = gen_fp8(K, N)
Ase = gen_scale(M, k_blocks); Bse = gen_scale(N, k_blocks)
As = preshuffle_v2_a(Ase); Bs = preshuffle_v2_b(Bse)
A_ref = A.float() * expand_col(Ase, K)
B_ref = B.float() * expand_col(Bse, K)
C_ref = A_ref.T @ B_ref

C_sq = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
C_rt = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

def run(mod, C):
    mod.gemm_crr_pq_v2(A, B, As, Bs, C)

# Square baseline
run(mod_square, C_sq); torch.cuda.synchronize()
diff_sq = (C_sq[:M, :N].float() - C_ref).abs()
sig_p = (C_ref * C_ref).sum().item()
noise_p_sq = (diff_sq * diff_sq).sum().item()
snr_sq = 10 * math.log10(sig_p / noise_p_sq) if noise_p_sq > 0 else float('inf')
pass_sq = ((diff_sq <= 3.0) | (diff_sq / C_ref.abs().clamp(min=1.0) <= 0.10)).float().mean().item() * 100
print(f'CORRECTNESS_SQUARE snr_db={snr_sq:.2f} pass_rate_pct={pass_sq:.2f}')
print(f'C_sq[0,:8]={C_sq[0,:8].tolist()}')
print(f'C_ref[0,:8]={C_ref[0,:8].tolist()}')

# Rect Path 1
run(mod_rect, C_rt); torch.cuda.synchronize()
diff_rt = (C_rt[:M, :N].float() - C_ref).abs()
noise_p_rt = (diff_rt * diff_rt).sum().item()
snr_rt = 10 * math.log10(sig_p / noise_p_rt) if noise_p_rt > 0 else float('inf')
pass_rt = ((diff_rt <= 3.0) | (diff_rt / C_ref.abs().clamp(min=1.0) <= 0.10)).float().mean().item() * 100
print(f'CORRECTNESS_RECT_PATH1 snr_db={snr_rt:.2f} pass_rate_pct={pass_rt:.2f}')
print(f'C_rt[0,:8]={C_rt[0,:8].tolist()}')

# Determinism check on rect
C_rt0 = C_rt.clone()
det_ok = True
for _ in range(2):
    C_rt.zero_()
    run(mod_rect, C_rt); torch.cuda.synchronize()
    if not torch.equal(C_rt[:M, :N], C_rt0[:M, :N]):
        det_ok = False; break
print(f'DETERMINISM_RECT_PATH1 ok={det_ok}')

# Verdict
gate_pass = (snr_rt >= 48.0) and (pass_rt >= 99.0) and det_ok
print(f'STAGE_A2c_VERDICT {"PASS" if gate_pass else "FAIL"}')
