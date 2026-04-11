"""
Test CRR_MAIN_UNROLL=1 (current) vs UNROLL=2 for selected shapes.
Compiles new JIT SOs with UNROLL=2 and benchmarks them.
Uses full bench parameters (30w/50i).
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
A=(torch.randn(K,M,device='cuda')*0.1).to(torch.float8_e4m3fn)
B=(torch.randn(K,N,device='cuda')*0.1).to(torch.float8_e4m3fn)
C=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
fn=lambda:m.gemm_crr(A,B,C,1.0,1.0,{gm})
for _ in range({w}):C.zero_();fn()
se=torch.cuda.Event(enable_timing=True);ee=torch.cuda.Event(enable_timing=True);ts=[]
for _ in range({it}):(torch.cuda.synchronize(),se.record(),fn(),ee.record(),torch.cuda.synchronize(),ts.append(se.elapsed_time(ee)))
print(f"{{2.0*M*N*K/(sum(ts)/len(ts)*1e9):.1f}}")'''

# Current CRR_GROUP_M (for gm lookup)
CRR_GROUP_M = {
    ( 4096,   4096, 14336):  8,
    ( 4096,   6144,  4096):  8,
    ( 4096,   8192,  8192):  8,
    ( 4096,  10240,  8192):  8,
    ( 4096,  22016,  4096):  8,
    ( 4096,  28672,  4096):  8,
    ( 8192,   3584, 18944):  8,
    ( 8192,   8192,  8192):  8,
    ( 8192,   8192, 29568):  8,
    ( 8192,  10240,  8192):  8,
    ( 8192,  12288,  4096):  8,
    ( 8192,  37888,  3584):  8,
    (16384,   3584,  3584):  8,
    (16384,   4096,  4096):  8,
    (16384,   4608,  3584):  8,
    (16384,   6144,  4096):  8,
    (16384,  10240,  8192):  8,
    (16384,  28672,  4096):  8,
    (16384,  37888,  3584):  8,
    (16384,  59136,  8192):  8,
}

# Per-shape CRR LGKM/INSERT/VMCNT params (for reuse in new JIT SO)
CRR_SHAPE_PARAMS = {
    ( 4096,   4096,  4096): (4, 5, 5),
    ( 4096,   4096, 14336): (2, 3, 5),
    ( 4096,   8192,  8192): (4, 3, 4),
    ( 4096,  10240,  8192): (4, 3, 4),
    ( 4096,  12288,  4096): (2, 3, 5),
    ( 4096,  22016,  4096): (4, 5, 5),
    ( 4096,  28672,  4096): (4, 3, 4),
    ( 8192,   3584, 18944): (4, 3, 4),
    ( 8192,   4096,  4096): (2, 3, 5),
    ( 8192,   4096, 11008): (3, 5, 6),
    ( 8192,   4608,  3584): (3, 4, 5),
    ( 8192,   8192,  8192): (4, 5, 5),
    ( 8192,   8192, 28672): (3, 5, 6),
    ( 8192,   8192, 29568): (3, 5, 6),
    ( 8192,  10240,  8192): (4, 3, 4),
    ( 8192,  12288,  4096): (3, 4, 5),
    ( 8192,  16384, 16384): (2, 3, 5),
    ( 8192,  16384, 53248): (2, 3, 5),
    ( 8192,  22016,  4096): (3, 5, 6),
    ( 8192,  37888,  3584): (4, 5, 5),
    ( 8192, 106496, 16384): (4, 3, 4),
    (16384,   3584,  3584): (4, 3, 4),
    (16384,   3584, 18944): (3, 5, 6),
    (16384,   4096, 14336): (3, 5, 6),
    (16384,   4608,  3584): (4, 3, 4),
    (16384,   6144,  4096): (3, 5, 6),
    (16384,   8192, 29568): (2, 3, 5),
    (16384,  10240,  8192): (4, 3, 4),
    (16384,  16384, 53248): (3, 4, 5),
    (16384, 106496, 16384): (2, 3, 5),
}


def compile_crr_unroll2(M, N, K):
    """Compile CRR with UNROLL=2. Uses best available shape params."""
    outdir = os.path.join(CACHE, f"crr_{M}x{N}x{K}_8wave_unroll2")
    so = os.path.join(outdir, f"tk_fp8_layouts{EXT}")
    if os.path.exists(so):
        return outdir
    os.makedirs(outdir, exist_ok=True)

    params = CRR_SHAPE_PARAMS.get((M, N, K), None)
    extra = []
    if params:
        lgkm, insert, vmcnt = params
        extra = [
            f"-DCRR_EXACT_PREFETCH_LGKM={lgkm}",
            f"-DCRR_EXACT_B1_LDS_INSERT_AFTER={insert}",
            f"-DCRR_EXACT_STEADY_VMCNT={vmcnt}",
        ]
    else:
        extra = ["-DCRR_EXACT_PREFETCH_LGKM=4"]

    cmd = [HIPCXX, os.path.join(DIR, "kernel_jit_all.cpp"),
           *BASE_FLAGS,
           f"-DM_DIM={M}", f"-DN_DIM={N}", f"-DK_DIM={K}",
           "-DJIT_LAYOUT=3",
           "-DCRR_MAIN_UNROLL=2",
           *extra,
           *PY_INC.split(), *PY_LD.split(), "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"CRR UNROLL=2 JIT failed {M}x{N}x{K}:\n{r.stderr[-300:]}")
    return outdir


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


def get_current_so(M, N, K):
    """Get the current best SO (opt > 8wave > shared)."""
    opt = os.path.join(CACHE, f"crr_{M}x{N}x{K}_8wave_opt")
    if os.path.exists(os.path.join(opt, f"tk_fp8_layouts{EXT}")):
        return opt
    jit = os.path.join(CACHE, f"crr_{M}x{N}x{K}_8wave")
    if os.path.exists(os.path.join(jit, f"tk_fp8_layouts{EXT}")):
        return jit
    return os.path.join(CACHE, "crr_shared")


# Test shapes: focus on k_iters where unrolling is likely to help
# Avoid very large K (k_iters >= 200) where I-cache overflow is likely
test_shapes = [
    # k_iters=28 (K=3584)
    ( 8192,   3584,  3584),
    (16384,   3584,  3584),
    (16384,   4608,  3584),
    # k_iters=32 (K=4096)
    ( 4096,   4096,  4096),
    ( 8192,   4096,  4096),
    (16384,   4096,  4096),
    # k_iters=64 (K=8192)
    ( 8192,   8192,  8192),
    (16384,   8192,  8192),
    ( 8192,  10240,  8192),
    (16384,  10240,  8192),
    # k_iters=86 (K=11008)
    ( 8192,   4096, 11008),
    # k_iters=112 (K=14336)
    (16384,   4096, 14336),
    ( 4096,   4096, 14336),
    # k_iters=128 (K=16384) - borderline, test
    ( 8192,  16384, 16384),
    (16384,  16384, 16384),
    (16384,  18432, 16384),
    ( 8192, 106496, 16384),
    # k_iters=148 (K=18944)
    ( 8192,   3584, 18944),
    (16384,   3584, 18944),
]

sys.path.insert(0, DIR)
from bench_vs_hipblaslt import DenseModelConfigs, gen_gemm_test_cases
all_shapes = set()
for config in DenseModelConfigs.values():
    for mbs in [1, 2]:
        for _, seq, n, k in gen_gemm_test_cases(config):
            M = seq * mbs
            if M % 256 == 0 and n % 256 == 0 and k % 128 == 0:
                all_shapes.add((M, n, k))

test_shapes = [(M, N, K) for M, N, K in test_shapes if (M, N, K) in all_shapes]

# Load reference results
results_file = os.path.join(DIR, "bench_jit_full_results.json")
ref_crr = {}
if os.path.exists(results_file):
    with open(results_file) as f:
        ref_crr = json.load(f).get("crr", {})

print(f"Compiling UNROLL=2 JIT for {len(test_shapes)} shapes...")
t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as pool:
    futs = {pool.submit(compile_crr_unroll2, M, N, K): (M, N, K) for M, N, K in test_shapes}
    for fut in futs:
        M, N, K = futs[fut]
        try:
            fut.result()
            print(f"  compiled {M}x{N}x{K}")
        except Exception as e:
            print(f"  FAILED {M}x{N}x{K}: {e}")
print(f"Compilation done in {time.time()-t0:.1f}s")
print()

print(f"Benchmarking UNROLL=1 (current) vs UNROLL=2...")
print(f"{'Shape':<30} {'k_iters':>8} {'unroll1':>9} {'unroll2':>9} {'delta':>8} {'ref_crr':>9}")
print("-" * 90)

gains = {}
for M, N, K in test_shapes:
    gm = CRR_GROUP_M.get((M, N, K), 4)
    so_cur = get_current_so(M, N, K)
    so_u2 = os.path.join(CACHE, f"crr_{M}x{N}x{K}_8wave_unroll2")

    r1 = bench(M, N, K, so_cur, gm)
    r2 = bench(M, N, K, so_u2, gm) if os.path.exists(os.path.join(so_u2, f"tk_fp8_layouts{EXT}")) else 0.0

    delta = r2 - r1
    key = f"{M}_{N}_{K}"
    ref = ref_crr.get(key, 0)

    if delta > 15:
        gains[(M, N, K)] = delta
        flag = " ← GAIN"
    elif delta < -15:
        flag = " ← REGRESS"
    else:
        flag = ""

    label = f"{M}x{N}x{K}"
    print(f"{label:<30} {K//128:>8} {r1:>9.1f} {r2:>9.1f} {delta:>+7.1f} {ref:>9.1f}{flag}")
    sys.stdout.flush()

print()
print(f"Shapes where UNROLL=2 helps (>{15} TFLOPS):")
for (M, N, K), delta in sorted(gains.items(), key=lambda x: -x[1]):
    print(f"  {M}x{N}x{K}: +{delta:.1f} TFLOPS")
if not gains:
    print("  None found")
