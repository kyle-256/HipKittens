"""A/B test: with vs without twotile (VGPR impact: 212 vs 244)."""
import subprocess, os, sys, json, math

GPU = os.environ.get("HIP_VISIBLE_DEVICES", "4")
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

cache = {}
cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".autotune_cache.json")
if os.path.exists(cache_file):
    with open(cache_file) as f:
        cache = json.load(f)

bl = {}
bl_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_full_all_layouts.json")
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


NO_TT = "/tmp/rcr_no_twotile"
WITH_TT = "/tmp/rcr_with_twotile"

test_shapes = [
    # Small K (should benefit from lower VGPRs)
    (4096, 28672, 4096),
    (8192, 4096, 4096),
    (8192, 22016, 4096),
    (8192, 28672, 4096),
    (8192, 6144, 4096),
    (4096, 4096, 4096),
    (16384, 28672, 4096),
    # Medium K
    (8192, 8192, 8192),
    (8192, 4096, 14336),
    # Large K (twotile helps here)
    (8192, 16384, 16384),
    (8192, 18432, 16384),
    (16384, 106496, 16384),
    (8192, 16384, 53248),
]

print(f"{'Shape':>22} {'gm':>3} {'ki':>4} {'NoTT(212)':>10} {'TT(244)':>10} {'hipBL':>9} {'NoTT/BL':>8} {'TT/BL':>8} {'Δ':>6}")
print("-" * 100)

all_no = []
all_tt = []

for M, N, K in test_shapes:
    key = f"rcr_{M}_{N}_{K}"
    gm = cache.get(key, {}).get("group_m", 4)
    bl_key = f"{M}_{N}_{K}"
    bl_val = bl.get(bl_key, 0)
    ki = K // 128

    no_tf = bench(M, N, K, NO_TT, gm)
    tt_tf = bench(M, N, K, WITH_TT, gm)

    no_ratio = no_tf / bl_val if bl_val > 0 else 0
    tt_ratio = tt_tf / bl_val if bl_val > 0 else 0
    if no_ratio > 0:
        all_no.append(no_ratio)
    if tt_ratio > 0:
        all_tt.append(tt_ratio)
    delta = (no_tf - tt_tf) / tt_tf * 100 if tt_tf > 0 else 0

    shape_str = f"{M}x{N}x{K}"
    marker = "<<" if abs(delta) > 1.0 else ""
    print(f"{shape_str:>22} {gm:>3} {ki:>4} {no_tf:>10.1f} {tt_tf:>10.1f} {bl_val:>9.1f} {no_ratio:>7.4f}x {tt_ratio:>7.4f}x {delta:>+5.1f}% {marker}")

print()
if all_no and all_tt:
    geo_no = math.exp(sum(math.log(v) for v in all_no) / len(all_no))
    geo_tt = math.exp(sum(math.log(v) for v in all_tt) / len(all_tt))
    print(f"Geo-mean: NoTT(212)={geo_no:.4f}x  TT(244)={geo_tt:.4f}x  Δ={(geo_no/geo_tt-1)*100:+.2f}%")
