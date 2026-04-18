"""R33 Dev C: paired BABA bench, single .so, 2 layouts (RRR vs CRR or RRR vs RCR).

Loads ONE .so containing all of gemm_crr_pq_v2 / gemm_rcr_pq_v2 / gemm_rrr_pq_v2.
30s+ preheat, BABA pattern over N_PAIRS pairs.

Env vars:
  M, N, K        — shape
  SO             — path to .so
  MOD            — module name (default tk_mxfp8_r33c)
  LAYOUT_A       — baseline layout (rcr/crr)
  LAYOUT_B       — candidate layout (default rrr)
  PHYS_GPU       — physical GPU index for rocm-smi (default 2)
  N_PAIRS        — number of paired BABA reps (default 5)
  MXFP8_WARMUP   — warmup iters per bench (default 30)
  MXFP8_ITERS    — measured iters per bench (default 50)
  PREHEAT_S      — preheat seconds (default 60)
  WARMUP_PAIRS   — discarded warmup BABA pairs (default 2)
"""
import math, os, statistics, subprocess, sys, time, importlib.util

import torch
torch.manual_seed(0)

M = int(os.environ['M'])
N = int(os.environ['N'])
K = int(os.environ['K'])
n_pairs = int(os.environ.get('N_PAIRS', '5'))
warmup = int(os.environ.get('MXFP8_WARMUP', '30'))
iters = int(os.environ.get('MXFP8_ITERS', '50'))
phys_gpu = os.environ.get('PHYS_GPU', '2')
so_path = os.environ['SO']
mod_name = os.environ.get('MOD', 'tk_mxfp8_r33c')
layout_a = os.environ.get('LAYOUT_A', 'crr').lower()
layout_b = os.environ.get('LAYOUT_B', 'rrr').lower()
preheat_s = float(os.environ.get('PREHEAT_S', '60'))
warmup_pairs = int(os.environ.get('WARMUP_PAIRS', '2'))

if not os.path.exists(so_path):
    print(f'ERROR: {so_path} missing', file=sys.stderr); sys.exit(1)

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

mod = load(mod_name, so_path)
fn_entry = {
    'crr': mod.gemm_crr_pq_v2,
    'rcr': mod.gemm_rcr_pq_v2,
    'rrr': mod.gemm_rrr_pq_v2,
}

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
C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

def build_for_layout(lay):
    if lay == 'rcr':
        A = gen_fp8(M, K); B = gen_fp8(N, K)
        Ase = gen_scale(M, k_blocks); Bse = gen_scale(N, k_blocks)
        As = preshuffle_v2_a(Ase); Bs = preshuffle_v2_b(Bse)
        A_ref = A.float() * expand_row(Ase, K)
        B_ref = B.float() * expand_row(Bse, K)
        C_ref = A_ref @ B_ref.T
    elif lay == 'rrr':
        A = gen_fp8(M, K); B = gen_fp8(K, N)
        Ase = gen_scale(M, k_blocks); Bse = gen_scale(N, k_blocks)
        As = preshuffle_v2_a(Ase); Bs = preshuffle_v2_b(Bse)
        A_ref = A.float() * expand_row(Ase, K)
        B_ref = B.float() * expand_col(Bse, K)
        C_ref = A_ref @ B_ref
    elif lay == 'crr':
        A = gen_fp8(K, M); B = gen_fp8(K, N)
        Ase = gen_scale(M, k_blocks); Bse = gen_scale(N, k_blocks)
        As = preshuffle_v2_a(Ase); Bs = preshuffle_v2_b(Bse)
        A_ref = A.float() * expand_col(Ase, K)
        B_ref = B.float() * expand_col(Bse, K)
        C_ref = A_ref.T @ B_ref
    else:
        raise SystemExit(f'unknown layout {lay}')
    fn_call = fn_entry[lay]
    return A, B, As, Bs, C_ref, (lambda: fn_call(A, B, As, Bs, C))

A_a, B_a, As_a, Bs_a, Cref_a, fn_a = build_for_layout(layout_a)
A_b, B_b, As_b, Bs_b, Cref_b, fn_b = build_for_layout(layout_b)

label_a = layout_a.upper()
label_b = layout_b.upper()

# Correctness
for fn, name, ref in [(fn_a, label_a, Cref_a), (fn_b, label_b, Cref_b)]:
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
a_tf, b_tf = [], []
print(f'[sclk-pre-bench] {read_sclk()}', flush=True)
for w in range(warmup_pairs):
    avg_b = bench_fn(fn_b); tb = flops / (avg_b * 1e9)
    avg_a = bench_fn(fn_a); ta = flops / (avg_a * 1e9)
    print(f'WARMUP {w} {label_b}={tb:.2f} {label_a}={ta:.2f} sclk={read_sclk()}', flush=True)
for r in range(n_pairs):
    avg_b = bench_fn(fn_b); tb = flops / (avg_b * 1e9)
    avg_a = bench_fn(fn_a); ta = flops / (avg_a * 1e9)
    avg_b2 = bench_fn(fn_b); tb2 = flops / (avg_b2 * 1e9)
    avg_a2 = bench_fn(fn_a); ta2 = flops / (avg_a2 * 1e9)
    a_tf.append(ta); a_tf.append(ta2); b_tf.append(tb); b_tf.append(tb2)
    print(f'PAIR {r} {label_b}_1={tb:.2f} {label_a}_1={ta:.2f} {label_b}_2={tb2:.2f} {label_a}_2={ta2:.2f}', flush=True)
print(f'[sclk-post-bench] {read_sclk()}', flush=True)

def welch(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    sa, sb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    se = math.sqrt(sa/na + sb/nb)
    return (mb - ma) / se if se > 0 else float('nan')

print(f'{label_a}  median={statistics.median(a_tf):.2f} mean={statistics.mean(a_tf):.2f} stdev={statistics.stdev(a_tf):.2f} n={len(a_tf)}')
print(f'{label_b}  median={statistics.median(b_tf):.2f} mean={statistics.mean(b_tf):.2f} stdev={statistics.stdev(b_tf):.2f} n={len(b_tf)}')
t = welch(a_tf, b_tf)
print(f'Welch t ({label_a} vs {label_b}) = {t:.3f}  (positive = {label_b} faster)')
delta_pct = (statistics.median(b_tf) - statistics.median(a_tf)) / statistics.median(a_tf) * 100
print(f'DELTA_MEDIAN_PCT {label_b}_vs_{label_a} = {delta_pct:+.3f}%')
print(f'{label_a}_TFLOPS_LIST {",".join(f"{x:.4f}" for x in a_tf)}')
print(f'{label_b}_TFLOPS_LIST {",".join(f"{x:.4f}" for x in b_tf)}')
