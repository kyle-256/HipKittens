"""R32 Dev C bonus: BABA-paired V2-RRR vs V2-RCR for 8B Down (K-large).

RCR uses A=(M,K), B=(N,K); entry gemm_rcr_pq_v2.
RRR uses A=(M,K), B=(K,N); entry gemm_rrr_pq_v2.
Both produce C=(M,N) bf16. Same A/B preshuffles for scales.

Env: M, N, K, SO_RCR, SO_RRR, MOD_RCR, MOD_RRR, PHYS_GPU, N_PAIRS, MXFP8_WARMUP, MXFP8_ITERS
"""
import math, os, statistics, subprocess, sys, time, importlib.util

import torch
torch.manual_seed(0)

M = int(os.environ['M'])
N = int(os.environ['N'])
K = int(os.environ['K'])
n_pairs = int(os.environ.get('N_PAIRS', '5'))
warmup = int(os.environ.get('MXFP8_WARMUP', '50'))
iters = int(os.environ.get('MXFP8_ITERS', '100'))
phys_gpu = os.environ.get('PHYS_GPU', '2')
so_rcr = os.environ['SO_RCR']
so_rrr = os.environ['SO_RRR']
mod_rcr_name = os.environ.get('MOD_RCR', 'tk_mxfp8_layouts_rcr')
mod_rrr_name = os.environ.get('MOD_RRR', 'tk_mxfp8_layouts_rrr')

def load(modname, sopath):
    spec = importlib.util.spec_from_file_location(modname, sopath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def read_sclk():
    try:
        out = subprocess.check_output(['rocm-smi', '--showclocks', '-d', phys_gpu],
                                      stderr=subprocess.DEVNULL).decode()
        for line in out.splitlines():
            if 'sclk clock level' in line:
                return line.strip()
    except Exception as e:
        return f'sclk-err: {e}'
    return 'unk'

mod_rcr = load(mod_rcr_name, so_rcr)
mod_rrr = load(mod_rrr_name, so_rrr)

preheat_s = float(os.environ.get('PREHEAT_S', '60'))
warmup_pairs = int(os.environ.get('WARMUP_PAIRS', '2'))
print(f'[sclk-pre-preheat] {read_sclk()}', flush=True)
A_h = torch.randn(16384, 16384, device='cuda', dtype=torch.float16)
B_h = torch.randn(16384, 16384, device='cuda', dtype=torch.float16)
t0 = time.time(); i = 0
while time.time() - t0 < preheat_s:
    C_h = A_h @ B_h
    i += 1
torch.cuda.synchronize()
print(f'[preheat {preheat_s}s] done {i} iters', flush=True)
del A_h, B_h, C_h
torch.cuda.empty_cache()
print(f'[sclk-post-preheat] {read_sclk()}', flush=True)

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

def expand_row(s, cols):
    return torch.pow(2.0, s.float()).repeat_interleave(32, dim=1)[:, :cols]
def expand_col(s, rows):
    return torch.pow(2.0, s.float()).transpose(0, 1).repeat_interleave(32, dim=0)[:rows, :]

k_blocks = (K + 31) // 32
# RCR layout: A=(M,K), B=(N,K)
A_rcr = gen_fp8(M, K); B_rcr = gen_fp8(N, K)
Ase_c = gen_scale(M, k_blocks); Bse_c = gen_scale(N, k_blocks)
As_c = preshuffle_v2_a(Ase_c); Bs_c = preshuffle_v2_b(Bse_c)
C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
A_rcr_ref = A_rcr.float() * expand_row(Ase_c, K)
B_rcr_ref = B_rcr.float() * expand_row(Bse_c, K)
C_rcr_ref = A_rcr_ref @ B_rcr_ref.T

# RRR layout: A=(M,K), B=(K,N)
A_rrr = gen_fp8(M, K); B_rrr = gen_fp8(K, N)
Ase_r = gen_scale(M, k_blocks); Bse_r = gen_scale(N, k_blocks)
As_r = preshuffle_v2_a(Ase_r); Bs_r = preshuffle_v2_b(Bse_r)
A_rrr_ref = A_rrr.float() * expand_row(Ase_r, K)
B_rrr_ref = B_rrr.float() * expand_col(Bse_r, K)
C_rrr_ref = A_rrr_ref @ B_rrr_ref

def fn_rcr(): mod_rcr.gemm_rcr_pq_v2(A_rcr, B_rcr, As_c, Bs_c, C)
def fn_rrr(): mod_rrr.gemm_rrr_pq_v2(A_rrr, B_rrr, As_r, Bs_r, C)

# Correctness
for fn, name, ref in [(fn_rcr, 'RCR', C_rcr_ref), (fn_rrr, 'RRR', C_rrr_ref)]:
    C.zero_(); fn(); torch.cuda.synchronize()
    diff = (C[:M, :N].float() - ref).abs()
    sig_p = (ref * ref).sum().item()
    noise_p = (diff * diff).sum().item()
    snr = 10 * math.log10(sig_p / noise_p) if noise_p > 0 else float('inf')
    pass_rate = ((diff <= 3.0) | (diff / ref.abs().clamp(min=1.0) <= 0.10)).float().mean().item() * 100
    C.zero_(); fn(); ref0 = C[:M, :N].clone()
    det_ok = True
    for _ in range(2):
        C.zero_(); fn()
        if not torch.equal(C[:M, :N], ref0):
            det_ok = False; break
    print(f'CORRECTNESS_{name} snr_db={snr:.2f} pass_rate_pct={pass_rate:.2f} det_ok={det_ok}', flush=True)

def bench_fn(fn):
    se_start = torch.cuda.Event(enable_timing=True); se_end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        C.zero_(); fn()
    timings = []
    for _ in range(iters):
        C.zero_(); torch.cuda.synchronize()
        se_start.record(); fn(); se_end.record(); torch.cuda.synchronize()
        timings.append(se_start.elapsed_time(se_end))
    return sum(timings) / len(timings)

flops = 2.0 * M * N * K
rcr_tf, rrr_tf = [], []
print(f'[sclk-pre-bench] {read_sclk()}', flush=True)
for w in range(warmup_pairs):
    avg_c = bench_fn(fn_rcr); tc = flops / (avg_c * 1e9)
    avg_r = bench_fn(fn_rrr); tr = flops / (avg_r * 1e9)
    print(f'WARMUP {w} RCR={tc:.2f} RRR={tr:.2f} sclk={read_sclk()}', flush=True)
for r in range(n_pairs):
    avg_c = bench_fn(fn_rcr); tc = flops / (avg_c * 1e9)
    avg_r = bench_fn(fn_rrr); tr = flops / (avg_r * 1e9)
    avg_c2 = bench_fn(fn_rcr); tc2 = flops / (avg_c2 * 1e9)
    avg_r2 = bench_fn(fn_rrr); tr2 = flops / (avg_r2 * 1e9)
    rcr_tf.append(tc); rcr_tf.append(tc2); rrr_tf.append(tr); rrr_tf.append(tr2)
    print(f'PAIR {r} RCR_1={tc:.2f} RRR_1={tr:.2f} RCR_2={tc2:.2f} RRR_2={tr2:.2f}', flush=True)
print(f'[sclk-post-bench] {read_sclk()}', flush=True)

def welch(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    sa, sb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    se = math.sqrt(sa/na + sb/nb)
    return (mb - ma) / se if se > 0 else float('nan')

print(f'RCR median={statistics.median(rcr_tf):.2f} mean={statistics.mean(rcr_tf):.2f} stdev={statistics.stdev(rcr_tf):.2f} n={len(rcr_tf)}')
print(f'RRR median={statistics.median(rrr_tf):.2f} mean={statistics.mean(rrr_tf):.2f} stdev={statistics.stdev(rrr_tf):.2f} n={len(rrr_tf)}')
t = welch(rcr_tf, rrr_tf)
print(f'Welch t (RCR vs RRR) = {t:.3f}  (positive = RRR faster)')
delta_pct = (statistics.median(rrr_tf) - statistics.median(rcr_tf)) / statistics.median(rcr_tf) * 100
print(f'DELTA_MEDIAN_PCT RRR_vs_RCR = {delta_pct:+.3f}%')
print(f'RCR_TFLOPS_LIST {",".join(f"{x:.4f}" for x in rcr_tf)}')
print(f'RRR_TFLOPS_LIST {",".join(f"{x:.4f}" for x in rrr_tf)}')
