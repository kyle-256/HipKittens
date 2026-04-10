"""Sweep round 2: more radical RCR configs."""
import subprocess, os, sys, time, json, math
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
    (4096, 28672, 4096),   # worst
    (8192, 4096, 4096),    # worst
    (8192, 28672, 4096),   # worst
    (8192, 8192, 8192),    # medium
    (8192, 16384, 16384),  # good
    (16384, 106496, 16384),# large
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
    {"name": "baseline",            "flags": []},
    {"name": "single_stg",          "flags": ["-DRCR_SINGLE_STAGE=1"]},
    {"name": "single_stg_vm0",      "flags": ["-DRCR_SINGLE_STAGE=1", "-DRCR_SINGLE_STAGE_INIT_VMCNT=0", "-DRCR_SINGLE_STAGE_STEADY_VMCNT=0"]},
    {"name": "single_stg_vm4",      "flags": ["-DRCR_SINGLE_STAGE=1", "-DRCR_SINGLE_STAGE_INIT_VMCNT=2", "-DRCR_SINGLE_STAGE_STEADY_VMCNT=4"]},
    {"name": "reduced_bar",         "flags": ["-DRCR_REDUCED_BARRIERS=1"]},
    {"name": "occ1",                "flags": ["-DGEMM_MIN_BLOCKS_PER_CU=1"]},
    {"name": "occ3",                "flags": ["-DGEMM_MIN_BLOCKS_PER_CU=3"]},
    {"name": "twotile_vm4",         "flags": ["-DRCR_TWO_TILE_SCHEDULE=1", "-DRCR_STEADY_VMCNT=4"]},
    {"name": "group_m8",            "flags": []},  # test with group_m=8 at runtime
    {"name": "no_swizzle",          "flags": ["-DGEMM_BLOCK_SWIZZLE=0"]},
    {"name": "v2a_shared",          "flags": ["-DRCR_USE_V2A_SHARED=1"]},
]

def compile_config(cfg):
    name = cfg["name"]
    outdir = os.path.join(DIR, ".sweep_cache2", name)
    so = os.path.join(outdir, f"tk_fp8_layouts{EXT}")
    os.makedirs(outdir, exist_ok=True)
    if os.path.exists(so):
        print(f"  Cached {name}")
        return outdir
    cmd = BASE + cfg["flags"] + ["-o", so]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"  FAIL {name}: {r.stderr[-200:]}")
        return None
    print(f"  Compiled {name} in {dt:.1f}s")
    return outdir

def bench_one(outdir, M, N, K, group_m=4):
    script = bench_tpl.format(M=M, N=N, K=K).replace("1.0,4)", f"1.0,{group_m})")
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

with open(os.path.join(DIR, "bench_full_all_layouts.json")) as f:
    bl = {}
    for r in json.load(f)["results"]:
        if r["layout"] == "rcr":
            bl[f'{r["M"]}_{r["N"]}_{r["K"]}'] = r["bl_tflops"]

print(f"\n[2] Benchmarking {len(SHAPES)} shapes × {len(dirs)} configs...")
results = {}
for name, outdir in dirs.items():
    results[name] = {}
    gm = 8 if name == "group_m8" else 4
    for M, N, K in SHAPES:
        tf = bench_one(outdir, M, N, K, group_m=gm)
        results[name][f"{M}_{N}_{K}"] = tf
        b = bl.get(f"{M}_{N}_{K}", 1)
        print(f"  {name:>18s}  {M:>5d}x{N:>6d}x{K:>5d} = {tf:>7.1f} ({tf/b:.3f}x)")

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
    geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if ratios else 0
    row += f"  {geo:>7.4f}x"
    print(row)

with open(os.path.join(DIR, "sweep_rcr_results2.json"), "w") as f:
    json.dump({"results": results}, f, indent=2)
print(f"\nSaved to sweep_rcr_results2.json")
