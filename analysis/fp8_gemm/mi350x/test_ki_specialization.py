"""Quick A/B test: dynamic vs K-specialized RCR kernels."""
import subprocess, os, sys, json

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


def bench(M, N, K, so_dir, gm=4):
    script = bench_tpl.format(M=M, N=N, K=K, gm=gm, w=WARMUP, it=ITERS)
    env = {**os.environ, "HIP_VISIBLE_DEVICES": GPU, "PYTHONPATH": so_dir}
    r = subprocess.run(["python3", "-c", script], capture_output=True, text=True,
                       env=env, timeout=120)
    if r.returncode == 0:
        return float(r.stdout.strip())
    print(f"  ERROR: {r.stderr[-200:]}", file=sys.stderr)
    return 0.0


# Load autotune cache for optimal group_m
cache = {}
cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".autotune_cache.json")
if os.path.exists(cache_file):
    with open(cache_file) as f:
        cache = json.load(f)

# Load hipBLASLt baseline
bl = {}
bl_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_full_all_layouts.json")
if os.path.exists(bl_file):
    with open(bl_file) as f:
        for r in json.load(f)["results"]:
            if r["layout"] == "rcr":
                bl[f'{r["M"]}_{r["N"]}_{r["K"]}'] = r["bl_tflops"]

DYN_DIR = "/tmp/rcr_test_dyn"
K4096_DIR = "/tmp/rcr_test_k4096"
K16384_DIR = "/tmp/rcr_test_k16384"

# Test shapes: worst K=4096 shapes + best K=16384 shapes
test_shapes = [
    # K=4096 (should benefit from specialization)
    (4096, 28672, 4096),
    (8192, 4096, 4096),
    (8192, 22016, 4096),
    (8192, 28672, 4096),
    (8192, 6144, 4096),
    (4096, 4096, 4096),
    (16384, 28672, 4096),
    # K=16384 (control group)
    (8192, 16384, 16384),
    (8192, 18432, 16384),
    (16384, 16384, 16384),
]

print(f"{'Shape':>22} {'gm':>3} {'Dynamic':>9} {'K-spec':>9} {'hipBL':>9} {'Dyn/BL':>8} {'Spec/BL':>8} {'Δ':>6}")
print("-" * 90)

for M, N, K in test_shapes:
    key = f"rcr_{M}_{N}_{K}"
    gm = cache.get(key, {}).get("group_m", 4)
    bl_key = f"{M}_{N}_{K}"
    bl_val = bl.get(bl_key, 0)

    ki = K // 128
    spec_dir = K4096_DIR if K == 4096 else K16384_DIR if K == 16384 else DYN_DIR

    dyn_tf = bench(M, N, K, DYN_DIR, gm)
    spec_tf = bench(M, N, K, spec_dir, gm)

    dyn_ratio = dyn_tf / bl_val if bl_val > 0 else 0
    spec_ratio = spec_tf / bl_val if bl_val > 0 else 0
    delta = (spec_tf - dyn_tf) / dyn_tf * 100 if dyn_tf > 0 else 0

    shape_str = f"{M}x{N}x{K}"
    print(f"{shape_str:>22} {gm:>3} {dyn_tf:>9.1f} {spec_tf:>9.1f} {bl_val:>9.1f} {dyn_ratio:>7.4f}x {spec_ratio:>7.4f}x {delta:>+5.1f}%")
