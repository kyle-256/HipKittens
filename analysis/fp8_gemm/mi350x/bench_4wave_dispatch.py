"""Benchmark: 4-wave K-specialized vs 8-wave dynamic vs hipBLASLt."""
import subprocess, os, json, math, sys

GPU = os.environ.get("HIP_VISIBLE_DEVICES", "4")
DIR = os.path.dirname(os.path.abspath(__file__))
WARMUP, ITERS = 30, 50

bench_tpl = '''import torch,tk_fp8_layouts as m
M,N,K={M},{N},{K}
A=(torch.randn(M,K,device='cuda')*0.1).to(torch.float8_e4m3fn)
B=(torch.randn(N,K,device='cuda')*0.1).to(torch.float8_e4m3fn)
C=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
fn=lambda:m.gemm_rcr(A,B,C,1.0,1.0,{gm})
for _ in range({w}):C.zero_();fn()
se=torch.cuda.Event(enable_timing=True);ee=torch.cuda.Event(enable_timing=True);ts=[]
for _ in range({it}):(torch.cuda.synchronize(),se.record(),fn(),ee.record(),torch.cuda.synchronize(),ts.append(se.elapsed_time(ee)))
print(f"{{2.0*M*N*K/(sum(ts)/len(ts)*1e9):.1f}}")'''

K_DIRS = {k: f"/tmp/4wave_k{k}" for k in [3584, 4096, 8192, 11008, 14336, 16384]}
DYN_DIR = os.path.join(DIR, ".jit_cache", "rcr_shared")
if not os.path.exists(DYN_DIR):
    DYN_DIR = DIR

cache = {}
cache_file = os.path.join(DIR, ".autotune_cache.json")
if os.path.exists(cache_file):
    with open(cache_file) as f:
        cache = json.load(f)

bl = {}
bl_file = os.path.join(DIR, "bench_full_all_layouts.json")
if os.path.exists(bl_file):
    with open(bl_file) as f:
        for r in json.load(f)["results"]:
            if r["layout"] == "rcr":
                bl[f'{r["M"]}_{r["N"]}_{r["K"]}'] = r["bl_tflops"]


def bench(M, N, K, so_dir, gm=4):
    script = bench_tpl.format(M=M, N=N, K=K, gm=gm, w=WARMUP, it=ITERS)
    env = {**os.environ, "HIP_VISIBLE_DEVICES": GPU, "PYTHONPATH": so_dir}
    try:
        r = subprocess.run(["python3", "-c", script], capture_output=True, text=True,
                           env=env, timeout=120)
        if r.returncode == 0:
            return float(r.stdout.strip())
    except Exception:
        pass
    return 0.0


from bench_vs_hipblaslt import DenseModelConfigs, gen_gemm_test_cases

all_shapes = set()
for config in DenseModelConfigs.values():
    for mbs in [1, 2]:
        for _, seq, n, k in gen_gemm_test_cases(config):
            M = seq * mbs
            if M % 256 == 0 and n % 256 == 0 and k % 128 == 0:
                all_shapes.add((M, n, k))
shapes = sorted(all_shapes)

print(f"{'Shape':>22} {'K':>6} {'8wave':>8} {'4wave':>8} {'best':>8} {'hipBL':>8} {'8w/BL':>7} {'4w/BL':>7} {'best/BL':>7}")
print("-" * 105)

ratios_8w = []
ratios_4w = []
ratios_best = []

for M, N, K in shapes:
    key = f"rcr_{M}_{N}_{K}"
    gm = cache.get(key, {}).get("group_m", 4)
    bl_key = f"{M}_{N}_{K}"
    bl_val = bl.get(bl_key, 0)

    tf_8w = bench(M, N, K, DIR, gm)

    tf_4w = 0.0
    if K in K_DIRS and os.path.exists(K_DIRS[K]):
        tf_4w = bench(M, N, K, K_DIRS[K], gm)

    best = max(tf_8w, tf_4w)
    which = "4w" if tf_4w > tf_8w else "8w"

    r8 = tf_8w / bl_val if bl_val > 0 else 0
    r4 = tf_4w / bl_val if bl_val > 0 else 0
    rb = best / bl_val if bl_val > 0 else 0

    if r8 > 0: ratios_8w.append(r8)
    if r4 > 0: ratios_4w.append(r4)
    if rb > 0: ratios_best.append(rb)

    shape_str = f"{M}x{N}x{K}"
    win = "*" if rb >= 1.0 else " "
    print(f"{shape_str:>22} {K:>6} {tf_8w:>8.1f} {tf_4w:>8.1f} {best:>8.1f} {bl_val:>8.1f} {r8:>6.4f}x {r4:>6.4f}x {rb:>6.4f}x{win} [{which}]")

print()
if ratios_8w:
    g8 = math.exp(sum(math.log(v) for v in ratios_8w) / len(ratios_8w))
    print(f"8-wave dynamic geo-mean:    {g8:.4f}x ({sum(1 for v in ratios_8w if v >= 1.0)}/{len(ratios_8w)} wins)")
if ratios_4w:
    g4 = math.exp(sum(math.log(v) for v in ratios_4w) / len(ratios_4w))
    print(f"4-wave K-spec geo-mean:     {g4:.4f}x ({sum(1 for v in ratios_4w if v >= 1.0)}/{len(ratios_4w)} wins)")
if ratios_best:
    gb = math.exp(sum(math.log(v) for v in ratios_best) / len(ratios_best))
    print(f"Best-of-both geo-mean:      {gb:.4f}x ({sum(1 for v in ratios_best if v >= 1.0)}/{len(ratios_best)} wins)")
