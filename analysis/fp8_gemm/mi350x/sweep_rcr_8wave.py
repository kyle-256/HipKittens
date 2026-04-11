"""
Test RCR 8-wave JIT vs current (4-wave JIT or shared SO) for worst RCR-vs-hipBLASLt shapes.

RCR_8WAVE_JIT_SHAPES currently = {K=11008 x2, K=18944 for M=8192, K=29568 x2}.
This script tests shapes with RCR speedup < 0.95x vs hipBLASLt that are NOT already
using 8-wave JIT. Compiles 8-wave JIT via kernel_jit_rcr.cpp with:
  -DRCR_USE_EXACT_4WAVE_FASTPATH=0 -DRCR_USE_EXACT_8WAVE_FASTPATH=1

Full bench parameters (30w/50i).
"""
import subprocess, os, sys, json, time
from concurrent.futures import ThreadPoolExecutor

DIR = os.path.dirname(os.path.abspath(__file__))
TK = os.environ.get("THUNDERKITTENS_ROOT", os.path.abspath(os.path.join(DIR, "..", "..", "..")))
GPU = os.environ.get("HIP_VISIBLE_DEVICES", "4")
PY_INC = subprocess.check_output(["python3", "-m", "pybind11", "--includes"], text=True).strip()
PY_LD = subprocess.check_output(["python3-config", "--ldflags"], text=True).strip().replace("-lcrypt", "")
EXT = subprocess.check_output(["python3-config", "--extension-suffix"], text=True).strip()

HIPCXX = "/opt/rocm/bin/hipcc"
BASE_FLAGS = [
    "-DKITTENS_CDNA4", "--offload-arch=gfx950",
    "-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-ffast-math",
    "-I/opt/rocm/include/rocrand",
    "-DRCR_STEADY_VMCNT=8",
    "-DGEMM_BLOCK_SWIZZLE=1", "-DGEMM_BLOCK_SWIZZLE_NUM_XCDS=8",
    "-std=c++20", "-w", "-shared", "-fPIC",
    f"-I{TK}/include", f"-I{TK}/prototype", "-I/opt/rocm/include/hip",
]

CACHE = os.path.join(DIR, ".jit_cache")
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

# Current RCR_GROUP_M from bench_jit_full.py
RCR_GROUP_M = {
    (4096,  4096,  4096): 16,
    (4096,  4096, 11008):  4,
    (4096, 22016,  4096): 16,
    (4096,  8192,  8192): 16,
    (4096, 10240,  8192): 16,
    (8192,  3584,  3584): 16,
    (8192,  4096,  4096):  8,
    (8192,  4608,  3584):  8,
    (8192,  6144,  4096): 16,
    (8192,  8192,  8192):  8,
    (8192, 10240,  8192):  4,
    (8192, 12288,  4096):  8,
    (8192, 16384, 16384):  8,
    (8192, 16384, 53248): 16,
    (8192, 18432, 16384): 16,
    (8192, 22016,  4096):  8,
    (8192, 28672,  4096):  8,
    (8192, 37888,  3584): 16,
    (8192, 57344,  8192):  8,
    (8192,  8192, 28672): 16,
    (8192, 106496,16384): 16,
    (16384, 3584,  3584): 16,
    (16384, 3584, 18944): 16,
    (16384, 4096,  4096): 16,
    (16384, 4608,  3584):  4,
    (16384, 8192,  8192):  8,
    (16384,10240,  8192):  8,
    (16384,16384, 16384):  8,
    (16384,16384, 53248): 16,
    (16384,18432, 16384):  8,
    (16384,28672,  4096):  4,
    (16384,37888,  3584):  4,
}

# Already using 8-wave JIT (skip these)
RCR_8WAVE_JIT_SHAPES = {
    (4096, 4096, 11008), (8192, 4096, 11008), (8192, 3584, 18944),
    (8192, 8192, 29568), (16384, 8192, 29568),
}

# Worst RCR vs hipBLASLt shapes NOT already using 8-wave JIT.
# Ordered by speedup (ascending = worst first), all speedup < 0.95x.
# Grid size included for reference: shapes with grid < 640 use shared SO currently.
test_shapes = [
    # speedup < 0.90x (very bad)
    (16384,  6144,  4096),  # 0.839x  grid=1536 (4wave)
    ( 4096, 28672,  4096),  # 0.893x  grid=1792 (4wave)
    (16384, 37888,  3584),  # 0.894x  grid=9472 (4wave)
    ( 8192, 12288,  4096),  # 0.895x  grid=1536 (4wave)
    (16384,  3584, 18944),  # 0.897x  grid=896  (4wave), K=18944 (like 8192x3584x18944 which IS 8wave)
    (16384,  8192,  8192),  # 0.898x  grid=2048 (4wave)
    ( 8192, 22016,  4096),  # 0.900x  grid=2752 (4wave)
    # speedup 0.90-0.95x (bad)
    ( 8192, 28672,  4096),  # 0.908x  grid=3584 (4wave)
    ( 8192, 37888,  3584),  # 0.909x  grid=4736 (4wave)
    (16384, 28672,  4096),  # 0.903x  grid=7168 (4wave)
    (16384,  4096, 14336),  # 0.913x  grid=1024 (4wave)
    (16384, 10240,  8192),  # 0.913x  grid=2560 (4wave)
    (16384, 106496,16384),  # 0.922x  grid=26624 (4wave)
    ( 8192, 10240,  8192),  # 0.923x  grid=1280 (4wave)
    ( 8192, 106496,16384),  # 0.939x  grid=13312 (4wave)
    (16384, 16384, 53248),  # 0.933x  grid=4096 (4wave)
    ( 8192, 16384, 53248),  # 0.943x  grid=2048 (4wave)
    ( 8192,  6144,  4096),  # 0.943x  grid=768  (4wave)
    ( 8192,  4096, 14336),  # 0.938x  grid=512  (SHARED SO currently)
]


def compile_rcr_8wave(M, N, K):
    """Compile RCR 8-wave JIT via kernel_jit_rcr.cpp."""
    outdir = os.path.join(CACHE, f"rcr_{M}x{N}x{K}_8wave")
    so = os.path.join(outdir, f"tk_fp8_layouts{EXT}")
    if os.path.exists(so):
        return outdir
    os.makedirs(outdir, exist_ok=True)
    cmd = [HIPCXX, os.path.join(DIR, "kernel_jit_rcr.cpp"),
           *BASE_FLAGS,
           f"-DM_DIM={M}", f"-DN_DIM={N}", f"-DK_DIM={K}",
           "-DRCR_USE_EXACT_4WAVE_FASTPATH=0",
           "-DRCR_USE_EXACT_8WAVE_FASTPATH=1",
           *PY_INC.split(), *PY_LD.split(), "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"RCR 8wave JIT failed {M}x{N}x{K}:\n{r.stderr[-400:]}")
    return outdir


def get_current_so(M, N, K):
    """Current best SO: 4wave JIT > shared."""
    jit_4wave = os.path.join(CACHE, f"rcr_{M}x{N}x{K}_4wave")
    if os.path.exists(os.path.join(jit_4wave, f"tk_fp8_layouts{EXT}")):
        return jit_4wave, "4wave"
    return os.path.join(CACHE, "rcr_shared"), "shared"


def bench(M, N, K, so_dir, gm):
    script = bench_tpl.format(M=M, N=N, K=K, gm=gm, w=WARMUP, it=ITERS)
    env = {**os.environ, "HIP_VISIBLE_DEVICES": GPU, "PYTHONPATH": so_dir}
    r = subprocess.run(["python3", "-c", script], capture_output=True, text=True,
                       cwd=so_dir, env=env, timeout=120)
    if r.returncode == 0:
        try:
            return float(r.stdout.strip())
        except ValueError:
            pass
    return 0.0


# Filter to shapes in our benchmark set
sys.path.insert(0, DIR)
from bench_vs_hipblaslt import DenseModelConfigs, gen_gemm_test_cases
all_shapes = set()
for config in DenseModelConfigs.values():
    for mbs in [1, 2]:
        for _, seq, n, k in gen_gemm_test_cases(config):
            M = seq * mbs
            if M % 256 == 0 and n % 256 == 0 and k % 128 == 0:
                all_shapes.add((M, n, k))

test_shapes = [(M, N, K) for M, N, K in test_shapes
               if (M, N, K) in all_shapes and (M, N, K) not in RCR_8WAVE_JIT_SHAPES]

# Load hipBLASLt baselines
bl_file = os.path.join(DIR, "bench_full_all_layouts.json")
bl_tflops = {}
if os.path.exists(bl_file):
    with open(bl_file) as f:
        for entry in json.load(f)["results"]:
            if entry["layout"] == "rcr":
                key = (entry["M"], entry["N"], entry["K"])
                bl_tflops[key] = max(bl_tflops.get(key, 0), entry["bl_tflops"])

# Load current RCR results for reference
results_file = os.path.join(DIR, "bench_jit_full_results.json")
ref_rcr = {}
if os.path.exists(results_file):
    with open(results_file) as f:
        ref_rcr = json.load(f).get("rcr", {})

print(f"Compiling RCR 8-wave JIT for {len(test_shapes)} shapes...")
t0 = time.time()
with ThreadPoolExecutor(max_workers=6) as pool:
    futs = {pool.submit(compile_rcr_8wave, M, N, K): (M, N, K) for M, N, K in test_shapes}
    failed = []
    for fut in futs:
        M, N, K = futs[fut]
        try:
            fut.result()
            print(f"  compiled {M}x{N}x{K}")
        except Exception as e:
            print(f"  FAILED {M}x{N}x{K}: {e}")
            failed.append((M, N, K))
print(f"Compilation done in {time.time()-t0:.1f}s ({len(failed)} failures)")
print()

print(f"Benchmarking: current (4wave/shared) vs 8-wave JIT...")
print(f"{'Shape':<25} {'cur_type':>8} {'gm':>3} {'current':>9} {'8wave':>9} {'delta':>8} {'bl':>9} {'ratio8w':>8}")
print("-" * 90)

gains = {}
for M, N, K in test_shapes:
    if (M, N, K) in failed:
        continue
    gm = RCR_GROUP_M.get((M, N, K), 4)
    so_cur, cur_type = get_current_so(M, N, K)
    so_8w = os.path.join(CACHE, f"rcr_{M}x{N}x{K}_8wave")

    r_cur = bench(M, N, K, so_cur, gm)
    r_8w = bench(M, N, K, so_8w, gm) if os.path.exists(os.path.join(so_8w, f"tk_fp8_layouts{EXT}")) else 0.0

    delta = r_8w - r_cur
    bl = bl_tflops.get((M, N, K), 0)
    ratio_8w = r_8w / bl if bl > 0 else 0

    key = f"{M}_{N}_{K}"
    ref = ref_rcr.get(key, 0)

    if delta > 15:
        gains[(M, N, K)] = (delta, cur_type, r_cur, r_8w, bl, ratio_8w)
        flag = " ← GAIN"
    elif delta < -15:
        flag = " ← REGRESS"
    else:
        flag = ""

    label = f"{M}x{N}x{K}"
    print(f"{label:<25} {cur_type:>8} {gm:>3} {r_cur:>9.1f} {r_8w:>9.1f} {delta:>+7.1f} {bl:>9.1f} {ratio_8w:>8.3f}{flag}")
    sys.stdout.flush()

print()
print(f"Shapes where 8-wave wins (>15 TFLOPS):")
for (M, N, K), (d, cur_type, r_cur, r_8w, bl, ratio_8w) in sorted(gains.items(), key=lambda x: -x[1][0]):
    print(f"  {M}x{N}x{K}: +{d:.1f} TFLOPS  ({cur_type}: {r_cur:.1f} → 8wave: {r_8w:.1f}, bl={bl:.1f}, ratio={ratio_8w:.3f}x)")
if not gains:
    print("  None found")

print()
print("Candidate additions to RCR_8WAVE_JIT_SHAPES:")
for (M, N, K), (d, _, _, _, _, _) in sorted(gains.items(), key=lambda x: -x[1][0]):
    print(f"  ({M:>5}, {N:>6}, {K:>5}),  # +{d:.1f} TFLOPS vs {gains[(M,N,K)][1]}")
