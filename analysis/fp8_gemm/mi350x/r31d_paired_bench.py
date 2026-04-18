"""R31 Dev D: paired A/B bench with EXTRA long preheat to stabilize sclk.

Builds two versions of the .so (base and persist304), loads them into
two separate module names, runs interleaved A/B/A/B... over many reps.
This avoids re-spawning a Python process between A/B and amortizes the
preheat cost across both cells.

Note: due to pybind11 module-name conflict, we copy the .so to
two distinct names and import each.
"""
import math, os, statistics, subprocess, sys, time, importlib.util, shutil

import torch
torch.manual_seed(0)

M = int(os.environ.get('M', '4096'))
N = int(os.environ.get('N', '4096'))
K = int(os.environ.get('K', '4096'))
n_pairs = int(os.environ.get('N_PAIRS', '5'))
warmup = int(os.environ.get('MXFP8_WARMUP', '50'))
iters = int(os.environ.get('MXFP8_ITERS', '100'))
phys_gpu = os.environ.get('PHYS_GPU', '3')

so_base = 'tk_mxfp8_layouts_base.cpython-310-x86_64-linux-gnu.so'
so_p304 = 'tk_mxfp8_layouts_p304.cpython-310-x86_64-linux-gnu.so'

if not os.path.exists(so_base) or not os.path.exists(so_p304):
    print(f'ERROR: {so_base} or {so_p304} missing — run r31d_paired_build.sh first', file=sys.stderr)
    sys.exit(1)

def load(modname, sopath):
    spec = importlib.util.spec_from_file_location(modname, sopath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def read_sclk():
    try:
        out = subprocess.check_output(['rocm-smi', '--showclocks', '-d', phys_gpu], stderr=subprocess.DEVNULL).decode()
        for line in out.splitlines():
            if 'sclk clock level' in line:
                return line.strip()
    except Exception as e:
        return f'sclk-err: {e}'
    return 'unk'

mod_base = load('tk_mxfp8_layouts_base', so_base)
mod_p304 = load('tk_mxfp8_layouts_p304', so_p304)

print(f'[sclk-pre-preheat] {read_sclk()}', flush=True)
A_h = torch.randn(16384, 16384, device='cuda', dtype=torch.float16)
B_h = torch.randn(16384, 16384, device='cuda', dtype=torch.float16)
t0 = time.time()
i = 0
# Extra long preheat: 30 seconds
while time.time() - t0 < 30.0:
    C_h = A_h @ B_h
    i += 1
torch.cuda.synchronize()
print(f'[preheat 30s] done {i} iters', flush=True)
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

k_blocks = (K + 31) // 32
A = gen_fp8(M, K); B = gen_fp8(N, K)
Ase = gen_scale(M, k_blocks); Bse = gen_scale(N, k_blocks)
As = preshuffle_v2_a(Ase); Bs = preshuffle_v2_b(Bse)
C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

A_ref = A.float() * expand_row(Ase, K)
B_ref = B.float() * expand_row(Bse, K)
C_ref = A_ref @ B_ref.T

def fn_base(): mod_base.gemm_rcr_pq_v2(A, B, As, Bs, C)
def fn_p304(): mod_p304.gemm_rcr_pq_v2(A, B, As, Bs, C)

# Correctness
for fn, name in [(fn_base, 'base'), (fn_p304, 'p304')]:
    C.zero_(); fn(); torch.cuda.synchronize()
    diff = (C[:M, :N].float() - C_ref).abs()
    sig_p = (C_ref * C_ref).sum().item()
    noise_p = (diff * diff).sum().item()
    snr = 10 * math.log10(sig_p / noise_p) if noise_p > 0 else float('inf')
    pass_rate = ((diff <= 3.0) | (diff / C_ref.abs().clamp(min=1.0) <= 0.10)).float().mean().item() * 100
    print(f'CORRECTNESS_{name} snr_db={snr:.2f} pass_rate_pct={pass_rate:.2f}', flush=True)

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
base_tf = []
p304_tf = []
print(f'[sclk-pre-bench] {read_sclk()}', flush=True)
for r in range(n_pairs):
    # Interleave: base, p304, base, p304... (a BABA pattern is more diff-robust)
    avg_b = bench_fn(fn_base); tb = flops / (avg_b * 1e9)
    avg_p = bench_fn(fn_p304); tp = flops / (avg_p * 1e9)
    base_tf.append(tb); p304_tf.append(tp)
    print(f'PAIR {r} base={tb:.2f} p304={tp:.2f} delta={tp-tb:+.2f} ({(tp-tb)/tb*100:+.2f}%)', flush=True)
print(f'[sclk-post-bench] {read_sclk()}', flush=True)

def welch(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    sa, sb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    se = math.sqrt(sa/na + sb/nb)
    return (mb - ma) / se if se > 0 else float('nan')

print(f'BASE  median={statistics.median(base_tf):.2f} mean={statistics.mean(base_tf):.2f} stdev={statistics.stdev(base_tf):.2f}')
print(f'P304  median={statistics.median(p304_tf):.2f} mean={statistics.mean(p304_tf):.2f} stdev={statistics.stdev(p304_tf):.2f}')
print(f'Welch t (base vs p304) = {welch(base_tf, p304_tf):.3f}')
print(f'BASE_TFLOPS_LIST {",".join(f"{x:.4f}" for x in base_tf)}')
print(f'P304_TFLOPS_LIST {",".join(f"{x:.4f}" for x in p304_tf)}')
