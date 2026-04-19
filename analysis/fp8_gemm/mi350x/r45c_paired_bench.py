"""R45 Dev C — paired BABA bench for SMALLM-B32-TAIL-BVEC vs R42B baseline.

Mirrors r37_paired_bench_2so.py pattern (paired BABA reps, sclk gates,
preheat, warmup pairs) but uses gemm_rcr_pq entry point (the smallm tail
kernel route).

Env vars:
  M, N, K        - shape
  SO_A, MOD_A    - R42B baseline .so + module name
  SO_B, MOD_B    - R45C bvec candidate .so + module name
  SO_FP, MOD_FP  - FP8 reference .so + module name (optional, for ratio)
  PHYS_GPU       - rocm-smi device index (HIP_VISIBLE_DEVICES is the runtime sel)
  N_PAIRS        - number of paired BABA reps (default 5)
  MXFP8_WARMUP   - warmup iters per bench (default 30)
  MXFP8_ITERS    - measured iters per bench (default 50)
  PREHEAT_S      - preheat seconds (default 60)
  WARMUP_PAIRS   - discarded warmup BABA pairs (default 2)
"""
import math, os, statistics, subprocess, sys, time, importlib.util
import torch
torch.manual_seed(0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
import r41c_decode_bench as bench

M = int(os.environ['M']); N = int(os.environ['N']); K = int(os.environ['K'])
n_pairs = int(os.environ.get('N_PAIRS', '5'))
warmup = int(os.environ.get('MXFP8_WARMUP', '30'))
iters = int(os.environ.get('MXFP8_ITERS', '50'))
phys_gpu = os.environ.get('PHYS_GPU', '0')
so_a = os.environ['SO_A']; mod_a = os.environ['MOD_A']
so_b = os.environ['SO_B']; mod_b = os.environ['MOD_B']
so_fp = os.environ.get('SO_FP', ''); mod_fp = os.environ.get('MOD_FP', '')
preheat_s = float(os.environ.get('PREHEAT_S', '60'))
warmup_pairs = int(os.environ.get('WARMUP_PAIRS', '2'))


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


mod_A = load(mod_a, so_a)
mod_B = load(mod_b, so_b)
mod_FP = load(mod_fp, so_fp) if so_fp and mod_fp else None
assert mod_A is not mod_B, (
    f'PY_MODULE_NAME collision: MOD_A={mod_a!r} MOD_B={mod_b!r} resolved to the same '
    f'in-memory module — both .so must be built with distinct -DPY_MODULE_NAME')
assert so_a != so_b, f'SO_A == SO_B == {so_a!r}'

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

# RCR layout: A is (M,K), B is (N,K)
k_blocks = (K + 31) // 32
A = bench.gen_fp8(M, K)
B = bench.gen_fp8(N, K)
A_se = bench.gen_scale_exp(M, k_blocks)
B_se = bench.gen_scale_exp(N, k_blocks)
A_s = bench.preshuffle_v1(A_se)
B_s = bench.preshuffle_v1(B_se)
C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

A_fp8 = A
B_fp8 = B

fn_A = lambda: mod_A.gemm_rcr_pq(A, B, A_s, B_s, C)  # R42B baseline
fn_B = lambda: mod_B.gemm_rcr_pq(A, B, A_s, B_s, C)  # R45C BVEC candidate
fn_FP = (lambda: mod_FP.gemm_rcr(A_fp8, B_fp8, C, 1.0)) if mod_FP else None

# Correctness check: compare BVEC (B) outputs vs R42B (A) outputs.
C.zero_(); fn_A(); torch.cuda.synchronize(); C_ref = C.clone()
C.zero_(); fn_B(); torch.cuda.synchronize(); C_test = C.clone()
diff = (C_ref.float() - C_test.float()).abs()
abs_max = diff.max().item()
sig_p = (C_ref.float() ** 2).sum().item()
err_p = (diff ** 2).sum().item()
snr = 10 * math.log10(sig_p / err_p) if err_p > 0 else float('inf')
print(f'CORRECTNESS_BVEC_vs_R42B abs_max_err={abs_max:.4e} snr_db={snr:.2f}', flush=True)


def bench_fn(fn):
    se_start = torch.cuda.Event(enable_timing=True); se_end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        C.zero_(); fn()
    timings = []
    for _ in range(iters):
        C.zero_(); torch.cuda.synchronize()
        se_start.record(); fn(); se_end.record(); torch.cuda.synchronize()
        timings.append(se_start.elapsed_time(se_end))
    return statistics.median(timings)


flops = 2.0 * M * N * K
a_tf, b_tf, fp_tf = [], [], []
print(f'[sclk-pre-bench] {read_sclk()}', flush=True)

# Warmup BABA pairs (discarded)
for w in range(warmup_pairs):
    avg_b = bench_fn(fn_B); tb = flops / (avg_b * 1e9)
    avg_a = bench_fn(fn_A); ta = flops / (avg_a * 1e9)
    print(f'WARMUP {w} B_bvec={tb:.4f} A_r42b={ta:.4f}', flush=True)

# Measured BABA reps. Per-pair: B,A,B,A
for r in range(n_pairs):
    avg_b = bench_fn(fn_B); tb = flops / (avg_b * 1e9)
    avg_a = bench_fn(fn_A); ta = flops / (avg_a * 1e9)
    avg_b2 = bench_fn(fn_B); tb2 = flops / (avg_b2 * 1e9)
    avg_a2 = bench_fn(fn_A); ta2 = flops / (avg_a2 * 1e9)
    if fn_FP is not None:
        avg_fp = bench_fn(fn_FP); tfp = flops / (avg_fp * 1e9)
        fp_tf.append(tfp)
    a_tf.extend([ta, ta2]); b_tf.extend([tb, tb2])
    print(f'PAIR {r} B_bvec_1={tb:.4f} A_r42b_1={ta:.4f} B_bvec_2={tb2:.4f} A_r42b_2={ta2:.4f}', flush=True)
print(f'[sclk-post-bench] {read_sclk()}', flush=True)


def welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return float('nan')
    ma, mb = statistics.mean(a), statistics.mean(b)
    sa, sb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    se = math.sqrt(sa/na + sb/nb)
    return (mb - ma) / se if se > 0 else float('nan')


a_med = statistics.median(a_tf); b_med = statistics.median(b_tf)
print(f'R42B_BASELINE  median={a_med:.4f} mean={statistics.mean(a_tf):.4f} stdev={statistics.stdev(a_tf):.4f} n={len(a_tf)}')
print(f'BVEC_CANDIDATE median={b_med:.4f} mean={statistics.mean(b_tf):.4f} stdev={statistics.stdev(b_tf):.4f} n={len(b_tf)}')
t = welch(a_tf, b_tf)
print(f'Welch t (BVEC vs R42B) = {t:+.3f}  (positive = BVEC faster)')
delta_pct = (b_med - a_med) / a_med * 100 if a_med > 0 else float('nan')
print(f'DELTA_MEDIAN_PCT BVEC_vs_R42B = {delta_pct:+.3f}%')
if fp_tf:
    fp_med = statistics.median(fp_tf)
    print(f'FP8_REFERENCE  median={fp_med:.4f} mean={statistics.mean(fp_tf):.4f} stdev={statistics.stdev(fp_tf):.4f} n={len(fp_tf)}')
    print(f'BVEC/FP8 ratio = {b_med / fp_med * 100:.2f}%')
    print(f'R42B/FP8 ratio = {a_med / fp_med * 100:.2f}%')
print(f'R42B_TFLOPS_LIST {",".join(f"{x:.4f}" for x in a_tf)}')
print(f'BVEC_TFLOPS_LIST {",".join(f"{x:.4f}" for x in b_tf)}')
if fp_tf:
    print(f'FP8_TFLOPS_LIST {",".join(f"{x:.4f}" for x in fp_tf)}')
