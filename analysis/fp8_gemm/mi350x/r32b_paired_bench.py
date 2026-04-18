"""R32 Dev B: paired BABA in-process bench for V2-CRR SB pipelining variants.

Loads two .so versions (baseline DB and a candidate SB-PIPE) under distinct
pybind11 module names. 30s preheat, BABA pattern, n_pairs reps.

Required env:
  M, N, K     : matmul shape
  N_PAIRS     : number of BABA pairs (default 5)
  CAND_TAG    : candidate tag (e.g. pipe1, pipe2, pipe3)
  PHYS_GPU    : physical GPU id (for rocm-smi reporting)

Inputs:
  tk_mxfp8_layouts_base.cpython-310-x86_64-linux-gnu.so
  tk_mxfp8_layouts_${CAND_TAG}.cpython-310-x86_64-linux-gnu.so
"""
import math, os, statistics, subprocess, sys, time, importlib.util
import torch
torch.manual_seed(0)

M = int(os.environ.get('M', '8192'))
N = int(os.environ.get('N', '8192'))
K = int(os.environ.get('K', '8192'))
n_pairs = int(os.environ.get('N_PAIRS', '5'))
warmup = int(os.environ.get('MXFP8_WARMUP', '50'))
iters = int(os.environ.get('MXFP8_ITERS', '100'))
phys_gpu = os.environ.get('PHYS_GPU', '1')
cand_tag = os.environ.get('CAND_TAG', 'pipe3')

so_base = f'tk_mxfp8_layouts_base.cpython-310-x86_64-linux-gnu.so'
so_cand = f'tk_mxfp8_layouts_{cand_tag}.cpython-310-x86_64-linux-gnu.so'

if not os.path.exists(so_base) or not os.path.exists(so_cand):
    print(f'ERROR: {so_base} or {so_cand} missing — build them first', file=sys.stderr)
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
mod_cand = load(f'tk_mxfp8_layouts_{cand_tag}', so_cand)

print(f'[md5-base] {subprocess.check_output(["md5sum", so_base]).decode().strip()}', flush=True)
print(f'[md5-{cand_tag}] {subprocess.check_output(["md5sum", so_cand]).decode().strip()}', flush=True)
print(f'[sclk-pre-preheat] {read_sclk()}', flush=True)

A_h = torch.randn(16384, 16384, device='cuda', dtype=torch.float16)
B_h = torch.randn(16384, 16384, device='cuda', dtype=torch.float16)
t0 = time.time()
i = 0
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

def expand_col(s, rows):
    return torch.pow(2.0, s.float()).transpose(0, 1).repeat_interleave(32, dim=0)[:rows, :]

k_blocks = (K + 31) // 32
A = gen_fp8(K, M); B = gen_fp8(K, N)  # CRR layout: A is K-major (KxM), B is K-major (KxN)
Ase = gen_scale(M, k_blocks); Bse = gen_scale(N, k_blocks)
As = preshuffle_v2_a(Ase); Bs = preshuffle_v2_b(Bse)

A_ref = A.float() * expand_col(Ase, K)
B_ref = B.float() * expand_col(Bse, K)
C_ref = A_ref.T @ B_ref

C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

def fn_base(): mod_base.gemm_crr_pq_v2(A, B, As, Bs, C)
def fn_cand(): mod_cand.gemm_crr_pq_v2(A, B, As, Bs, C)

# Correctness for both
for fn, name in [(fn_base, 'base'), (fn_cand, cand_tag)]:
    C.zero_(); fn(); torch.cuda.synchronize()
    diff = (C[:M, :N].float() - C_ref).abs()
    sig_p = (C_ref * C_ref).sum().item()
    noise_p = (diff * diff).sum().item()
    snr = 10 * math.log10(sig_p / noise_p) if noise_p > 0 else float('inf')
    pass_rate = ((diff <= 3.0) | (diff / C_ref.abs().clamp(min=1.0) <= 0.10)).float().mean().item() * 100
    print(f'CORRECTNESS_{name} snr_db={snr:.2f} pass_rate_pct={pass_rate:.2f}', flush=True)

# Determinism for candidate
C.zero_(); fn_cand(); ref0 = C[:M, :N].clone()
det_ok = True
for _ in range(2):
    C.zero_(); fn_cand()
    if not torch.equal(C[:M, :N], ref0):
        det_ok = False; break
print(f'DETERMINISM_{cand_tag} ok={det_ok}', flush=True)

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
base_tf = []; cand_tf = []
print(f'[sclk-pre-bench] {read_sclk()}', flush=True)
for r in range(n_pairs):
    # BABA pattern.
    avg_b = bench_fn(fn_base); tb = flops / (avg_b * 1e9)
    avg_c = bench_fn(fn_cand); tc = flops / (avg_c * 1e9)
    avg_b2 = bench_fn(fn_base); tb2 = flops / (avg_b2 * 1e9)
    avg_c2 = bench_fn(fn_cand); tc2 = flops / (avg_c2 * 1e9)
    base_tf.append(tb); base_tf.append(tb2)
    cand_tf.append(tc); cand_tf.append(tc2)
    print(f'PAIR {r} base={tb:.2f},{tb2:.2f} cand={tc:.2f},{tc2:.2f} delta={(tc+tc2)/2-(tb+tb2)/2:+.2f}', flush=True)
print(f'[sclk-post-bench] {read_sclk()}', flush=True)

def welch(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    sa, sb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    se = math.sqrt(sa/na + sb/nb)
    return (mb - ma) / se if se > 0 else float('nan')

print(f'BASE  median={statistics.median(base_tf):.2f} mean={statistics.mean(base_tf):.2f} stdev={statistics.stdev(base_tf):.2f} n={len(base_tf)}')
print(f'CAND  median={statistics.median(cand_tf):.2f} mean={statistics.mean(cand_tf):.2f} stdev={statistics.stdev(cand_tf):.2f} n={len(cand_tf)}')
print(f'Welch t (base vs cand) = {welch(base_tf, cand_tf):.3f}')
print(f'BASE_TFLOPS_LIST {",".join(f"{x:.4f}" for x in base_tf)}')
print(f'CAND_TFLOPS_LIST {",".join(f"{x:.4f}" for x in cand_tf)}')
