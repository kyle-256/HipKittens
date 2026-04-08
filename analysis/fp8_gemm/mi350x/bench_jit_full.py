"""Full JIT benchmark: all shapes × all layouts × 4wave/8wave, compared with hipBLASLt."""
import subprocess, json, os, math, sys, time
from concurrent.futures import ThreadPoolExecutor

DIR = os.path.dirname(os.path.abspath(__file__))
TK = os.environ.get("THUNDERKITTENS_ROOT", os.path.abspath(os.path.join(DIR, "..", "..", "..")))
GPU = os.environ.get("HIP_VISIBLE_DEVICES", "4")
PY_INC = subprocess.check_output(["python3", "-m", "pybind11", "--includes"], text=True).strip()
PY_LD = subprocess.check_output(["python3-config", "--ldflags"], text=True).strip().replace("-lcrypt", "")
EXT = subprocess.check_output(["python3-config", "--extension-suffix"], text=True).strip()

LAYOUT_IDS = {"rcr": 1, "rrr": 2, "crr": 3}
WARMUP, ITERS = 30, 50

bench_tpl = '''import torch,tk_fp8_layouts as m
M,N,K={M},{N},{K}
{make_ab}
C=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
fn=lambda:m.gemm_{lay}(A,B,C,1.0,1.0,4)
for _ in range({w}):C.zero_();fn()
se=torch.cuda.Event(enable_timing=True);ee=torch.cuda.Event(enable_timing=True);ts=[]
for _ in range({it}):(torch.cuda.synchronize(),se.record(),fn(),ee.record(),torch.cuda.synchronize(),ts.append(se.elapsed_time(ee)))
print(f"{{2.0*M*N*K/(sum(ts)/len(ts)*1e9):.1f}}")'''

MAKE_AB = {
    "rcr": "A=(torch.randn(M,K,device='cuda')*0.1).to(torch.float8_e4m3fn);B=(torch.randn(N,K,device='cuda')*0.1).to(torch.float8_e4m3fn)",
    "rrr": "A=(torch.randn(M,K,device='cuda')*0.1).to(torch.float8_e4m3fn);B=(torch.randn(K,N,device='cuda')*0.1).to(torch.float8_e4m3fn)",
    "crr": "A=(torch.randn(K,M,device='cuda')*0.1).to(torch.float8_e4m3fn);B=(torch.randn(K,N,device='cuda')*0.1).to(torch.float8_e4m3fn)",
}

# All shapes from the benchmark
from bench_vs_hipblaslt import DenseModelConfigs, gen_gemm_test_cases
all_shapes = set()
for model_name, config in DenseModelConfigs.items():
    cases = gen_gemm_test_cases(config)
    for mbs in [1, 2]:
        for op_name, seq, n, k in cases:
            M = seq * mbs
            if M % 256 == 0 and n % 256 == 0 and k % 128 == 0:
                all_shapes.add((M, n, k))
shapes = sorted(all_shapes)
layouts = ["rcr", "rrr", "crr"]

# --- Phase 1: Compile all kernels ---
print(f"[Phase 1] Compiling {len(shapes)} shapes × {len(layouts)} layouts (8-wave)...")
t0 = time.time()

cache_dir = os.path.join(DIR, ".jit_cache")
os.makedirs(cache_dir, exist_ok=True)

def compile_one(args):
    M, N, K, lay = args
    outdir = os.path.join(cache_dir, f"bench_{lay}_{M}x{N}x{K}")
    so = os.path.join(outdir, f"tk_fp8_layouts{EXT}")
    if os.path.exists(so):
        return True
    os.makedirs(outdir, exist_ok=True)
    lid = LAYOUT_IDS[lay]
    extra = ["-DRRR_MAIN_UNROLL=1"] if lay == "rrr" else []
    cmd = ["/opt/rocm/bin/hipcc", os.path.join(DIR, "kernel_jit_all.cpp"),
           "-DKITTENS_CDNA4", "--offload-arch=gfx950", "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
           "-ffast-math", "-I/opt/rocm/include/rocrand",
           f"-DM_DIM={M}", f"-DN_DIM={N}", f"-DK_DIM={K}",
           f"-DJIT_LAYOUT={lid}", "-DRCR_STEADY_VMCNT=8",
           *extra, "-std=c++20", "-w", "-shared", "-fPIC",
           f"-I{TK}/include", f"-I{TK}/prototype", "-I/opt/rocm/include/hip",
           *PY_INC.split(), *PY_LD.split(), "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return r.returncode == 0

tasks = [(M, N, K, lay) for M, N, K in shapes for lay in layouts]
with ThreadPoolExecutor(max_workers=8) as pool:
    results_compile = list(pool.map(compile_one, tasks))
failed = sum(1 for r in results_compile if not r)
print(f"[Phase 1] Done in {time.time()-t0:.1f}s ({len(tasks)-failed}/{len(tasks)} ok)")

# --- Phase 2: Benchmark ---
print(f"\n[Phase 2] Benchmarking {len(shapes)} shapes × {len(layouts)} layouts on GPU{GPU}...")

results = {}
for lay in layouts:
    results[lay] = {}
    for M, N, K in shapes:
        outdir = os.path.join(cache_dir, f"bench_{lay}_{M}x{N}x{K}")
        so = os.path.join(outdir, f"tk_fp8_layouts{EXT}")
        if not os.path.exists(so):
            continue
        script = bench_tpl.format(M=M, N=N, K=K, lay=lay, make_ab=MAKE_AB[lay], w=WARMUP, it=ITERS)
        env = {**os.environ, "HIP_VISIBLE_DEVICES": GPU, "PYTHONPATH": outdir}
        r = subprocess.run(["python3", "-c", script], capture_output=True, text=True,
                          cwd=outdir, env=env, timeout=120)
        if r.returncode == 0:
            tf = float(r.stdout.strip())
            results[lay][f"{M}_{N}_{K}"] = tf

# --- Print summary ---
print(f"\n{'='*120}")
print(f"{'Model':>18} {'Op':>14} {'MBS':>3} {'M':>6} {'N':>6} {'K':>6}  {'RCR':>8} {'RRR':>8} {'CRR':>8}")
print("-" * 120)

for model_name in DenseModelConfigs:
    config = DenseModelConfigs[model_name]
    cases = gen_gemm_test_cases(config)
    for mbs in [1, 2]:
        for op_name, seq, n, k in cases:
            M = seq * mbs
            key = f"{M}_{n}_{k}"
            rcr = results.get("rcr", {}).get(key, 0)
            rrr = results.get("rrr", {}).get(key, 0)
            crr = results.get("crr", {}).get(key, 0)
            if rcr > 0 or rrr > 0 or crr > 0:
                print(f"{model_name:>18} {op_name:>14} {mbs:>3} {M:>6} {n:>6} {k:>6}  {rcr:>7.1f}  {rrr:>7.1f}  {crr:>7.1f}")

# Averages
print("=" * 120)
for lay in layouts:
    vals = [v for v in results[lay].values() if v > 0]
    if vals:
        avg = sum(vals) / len(vals)
        geo = math.exp(sum(math.log(v) for v in vals) / len(vals))
        print(f"{lay.upper()}: avg={avg:.1f}  geo-mean={geo:.1f} TFLOPS  ({len(vals)} shapes)")

# Save
with open(os.path.join(DIR, "bench_jit_full_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to bench_jit_full_results.json")
