"""4-way A/B test: LGKM × VMCNT combinations for RCR."""
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


configs = {
    "L4V8": "/tmp/test_lgkm4_vm8",
    "L8V8": "/tmp/test_lgkm8_vm8",
    "L4V6": "/tmp/test_lgkm4_vm6",
    "L8V6": "/tmp/test_lgkm8_vm6",
}

test_shapes = [
    (4096, 28672, 4096),
    (8192, 4096, 4096),
    (8192, 22016, 4096),
    (8192, 28672, 4096),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (8192, 16384, 16384),
    (16384, 106496, 16384),
]

header = f"{'Shape':>22} {'gm':>3}"
for name in configs:
    header += f" {name:>9}"
header += f" {'hipBL':>9}  {'best':>4}"
print(header)
print("-" * (22 + 3 + 10 * 4 + 9 + 10))

all_ratios = {name: [] for name in configs}

for M, N, K in test_shapes:
    key = f"rcr_{M}_{N}_{K}"
    gm = cache.get(key, {}).get("group_m", 4)
    bl_key = f"{M}_{N}_{K}"
    bl_val = bl.get(bl_key, 0)

    results = {}
    for name, so_dir in configs.items():
        tf = bench(M, N, K, so_dir, gm)
        results[name] = tf
        if tf > 0 and bl_val > 0:
            all_ratios[name].append(tf / bl_val)

    best_name = max(results, key=results.get)
    shape_str = f"{M}x{N}x{K}"
    row = f"{shape_str:>22} {gm:>3}"
    for name in configs:
        tf = results[name]
        row += f" {tf:>9.1f}"
    row += f" {bl_val:>9.1f}  {best_name}"
    print(row)

print()
print("Geo-mean vs hipBLASLt:")
for name in configs:
    vals = all_ratios[name]
    if vals:
        geo = math.exp(sum(math.log(v) for v in vals) / len(vals))
        print(f"  {name}: {geo:.4f}x")
