"""R36 Dev A bit-compare: HB shrink Stage B1/B2/B3 vs default V2-CRR on C[0,:8].

Env: SO_A (default V2-CRR .so), SO_B (HB shrink variant .so),
     MOD_A (default 'tk_mxfp8_kv_default'), MOD_B (e.g. 'tk_mxfp8_kv_hbshrink_b1'),
     M, N, K (default 4096, 1024, 8192).
"""
import math, os, sys, importlib.util
import torch
torch.manual_seed(0)

def load(modname, sopath):
    spec = importlib.util.spec_from_file_location(modname, sopath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod_a_name = os.environ.get('MOD_A', 'tk_mxfp8_kv_default')
mod_b_name = os.environ.get('MOD_B', 'tk_mxfp8_kv_hbshrink_b1')
mod_def = load(mod_a_name, os.environ['SO_A'])
mod_hbs = load(mod_b_name, os.environ['SO_B'])

M = int(os.environ.get('M', '4096'))
N = int(os.environ.get('N', '1024'))
K = int(os.environ.get('K', '8192'))
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
            perm_view[:, wm, 2 * pidx, :, :] = rg_view[:, rg_base + pidx, :, :]
            perm_view[:, wm, 2 * pidx + 1, :, :] = rg_view[:, rg_hi + pidx, :, :]
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

k_blocks = (K + 31) // 32
A = gen_fp8(K, M); B = gen_fp8(K, N)
Ase = gen_scale(M, k_blocks); Bse = gen_scale(N, k_blocks)
As = preshuffle_v2_a(Ase); Bs = preshuffle_v2_b(Bse)

C_def = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
C_hbs = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

mod_def.gemm_crr_pq_v2(A, B, As, Bs, C_def); torch.cuda.synchronize()
mod_hbs.gemm_crr_pq_v2(A, B, As, Bs, C_hbs); torch.cuda.synchronize()

print(f"M={M} N={N} K={K}")
print(f"C_default[0,:8]   = {C_def[0,:8].cpu().tolist()}")
print(f"C_hbshrink[0,:8]  = {C_hbs[0,:8].cpu().tolist()}")
print(f"bit_equal_C[0,:8] = {torch.equal(C_def[0,:8], C_hbs[0,:8])}")
print(f"bit_equal_full    = {torch.equal(C_def, C_hbs)}")
print(f"max_abs_diff      = {(C_def.float() - C_hbs.float()).abs().max().item()}")
print(f"mean_abs_diff     = {(C_def.float() - C_hbs.float()).abs().mean().item()}")
