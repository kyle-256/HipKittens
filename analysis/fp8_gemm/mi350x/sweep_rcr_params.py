"""Sweep RCR compile-time parameters to find optimal settings."""
import subprocess, os, sys, time, json
from concurrent.futures import ThreadPoolExecutor

DIR = os.path.dirname(os.path.abspath(__file__))
TK = os.path.abspath(os.path.join(DIR, "..", "..", ".."))
GPU = os.environ.get("HIP_VISIBLE_DEVICES", "4")
SRC = os.path.join(DIR, "kernel_fp8_layouts.cpp")
EXT = subprocess.check_output(["python3-config", "--extension-suffix"], text=True).strip()
PY_INC = subprocess.check_output(["python3", "-m", "pybind11", "--includes"], text=True).strip()
PY_LD = subprocess.check_output(["python3-config", "--ldflags"], text=True).strip().replace("-lcrypt", "")

BASE = ["/opt/rocm/bin/hipcc", SRC,
        "-DKITTENS_CDNA4", "--offload-arch=gfx950",
        "-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-ffast-math",
        "-I/opt/rocm/include/rocrand",
        "-DGEMM_BLOCK_SWIZZLE=1", "-DGEMM_BLOCK_SWIZZLE_NUM_XCDS=8",
        "-std=c++20", "-w", "-shared", "-fPIC",
        f"-I{TK}/include", f"-I{TK}/prototype", "-I/opt/rocm/include/hip",
        *PY_INC.split(), *PY_LD.split()]

SHAPES = [
    (4096, 28672, 4096),   # 0.896x worst
    (8192, 4096, 4096),    # 0.914x small
    (8192, 28672, 4096),   # 0.934x medium
    (8192, 8192, 8192),    # 0.962x medium-K
    (8192, 16384, 16384),  # 1.005x good
    (16384, 106496, 16384),# 0.992x large
]

bench_tpl = '''import torch,tk_fp8_layouts as m
M,N,K={M},{N},{K}
A=(torch.randn(M,K,device='cuda')*0.1).to(torch.float8_e4m3fn)
B=(torch.randn(N,K,device='cuda')*0.1).to(torch.float8_e4m3fn)
C=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
fn=lambda:m.gemm_rcr(A,B,C,1.0,1.0,4)
for _ in range(30):C.zero_();fn()
se=torch.cuda.Event(enable_timing=True);ee=torch.cuda.Event(enable_timing=True);ts=[]
for _ in range(50):(torch.cuda.synchronize(),se.record(),fn(),ee.record(),torch.cuda.synchronize(),ts.append(se.elapsed_time(ee)))
print(f"{{2.0*M*N*K/(sum(ts)/len(ts)*1e9):.1f}}")'''

CONFIGS = [
    {"name": "baseline",       "flags": []},
    {"name": "vmcnt4",         "flags": ["-DRCR_STEADY_VMCNT=4"]},
    {"name": "vmcnt6",         "flags": ["-DRCR_STEADY_VMCNT=6"]},
    {"name": "vmcnt10",        "flags": ["-DRCR_STEADY_VMCNT=10"]},
    {"name": "unroll4",        "flags": ["-DRCR_MAIN_UNROLL=4"]},
    {"name": "unroll4_vm6",    "flags": ["-DRCR_MAIN_UNROLL=4", "-DRCR_STEADY_VMCNT=6"]},
    {"name": "unroll4_vm4",    "flags": ["-DRCR_MAIN_UNROLL=4", "-DRCR_STEADY_VMCNT=4"]},
    {"name": "twotile",        "flags": ["-DRCR_TWO_TILE_SCHEDULE=1"]},
    {"name": "batched_reads",  "flags": ["-DRCR_BATCHED_READS=1"]},
    {"name": "batched_epi",    "flags": ["-DRCR_BATCHED_EPILOGUE_MMA=1"]},
    {"name": "unroll4_epi",    "flags": ["-DRCR_MAIN_UNROLL=4", "-DRCR_BATCHED_EPILOGUE_MMA=1"]},
]

def compile_config(cfg):
    name = cfg["name"]
    outdir = os.path.join(DIR, ".sweep_cache", name)
    so = os.path.join(outdir, f"tk_fp8_layouts{EXT}")
    os.makedirs(outdir, exist_ok=True)
    cmd = BASE + cfg["flags"] + ["-o", so]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"  FAIL {name}: {r.stderr[-200:]}")
        return None
    print(f"  Compiled {name} in {dt:.1f}s")
    return outdir

def bench_one(outdir, M, N, K):
    script = bench_tpl.format(M=M, N=N, K=K)
    env = {**os.environ, "HIP_VISIBLE_DEVICES": GPU, "PYTHONPATH": outdir}
    try:
        r = subprocess.run(["python3", "-c", script], capture_output=True, text=True,
                           cwd=outdir, env=env, timeout=120)
        if r.returncode == 0:
            return float(r.stdout.strip())
    except Exception:
        pass
    return 0.0

print(f"[1] Compiling {len(CONFIGS)} configs...")
dirs = {}
for cfg in CONFIGS:
    d = compile_config(cfg)
    if d:
        dirs[cfg["name"]] = d

print(f"\n[2] Benchmarking {len(SHAPES)} shapes × {len(dirs)} configs on GPU{GPU}...")
results = {}
for name, outdir in dirs.items():
    results[name] = {}
    for M, N, K in SHAPES:
        tf = bench_one(outdir, M, N, K)
        results[name][f"{M}_{N}_{K}"] = tf
        print(f"  {name:>18s}  {M:>5d}x{N:>6d}x{K:>5d} = {tf:>7.1f}")

# Load hipBLASLt baseline
with open(os.path.join(DIR, "bench_full_all_layouts.json")) as f:
    bl = {}
    for r in json.load(f)["results"]:
        if r["layout"] == "rcr":
            bl[f'{r["M"]}_{r["N"]}_{r["K"]}'] = r["bl_tflops"]

print(f"\n{'Config':>18s}", end="")
for M, N, K in SHAPES:
    print(f"  {M//1000}k×{N//1000}k×{K//1000}k", end="")
print(f"  {'geomean':>8s}")
print("-" * (18 + len(SHAPES) * 12 + 10))

for name in dirs:
    row = f"{name:>18s}"
    ratios = []
    for M, N, K in SHAPES:
        key = f"{M}_{N}_{K}"
        tk = results[name].get(key, 0)
        b = bl.get(key, 0)
        if tk > 0 and b > 0:
            r = tk / b
            ratios.append(r)
            w = "*" if r >= 1.0 else " "
            row += f"  {r:>5.3f}x{w}"
        else:
            row += f"  {'':>7s}"
    import math
    geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if ratios else 0
    row += f"  {geo:>7.4f}x"
    print(row)

with open(os.path.join(DIR, "sweep_rcr_results.json"), "w") as f:
    json.dump({"results": results, "shapes": [[M,N,K] for M,N,K in SHAPES]}, f, indent=2)
print(f"\nSaved to sweep_rcr_results.json")
