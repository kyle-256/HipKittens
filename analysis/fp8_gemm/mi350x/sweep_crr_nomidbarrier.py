"""
Test CRR_ENABLE_STEADY_MID_BARRIER=0 vs current=1 for all shapes.
The mid-barrier between cA+cB and cC+cD sections is potentially unnecessary
(no LDS conflict at that point). Removing it may allow better warp scheduling.

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
A=(torch.randn(K,M,device='cuda')*0.1).to(torch.float8_e4m3fn)
B=(torch.randn(K,N,device='cuda')*0.1).to(torch.float8_e4m3fn)
C=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
fn=lambda:m.gemm_crr(A,B,C,1.0,1.0,{gm})
for _ in range({w}):C.zero_();fn()
se=torch.cuda.Event(enable_timing=True);ee=torch.cuda.Event(enable_timing=True);ts=[]
for _ in range({it}):(torch.cuda.synchronize(),se.record(),fn(),ee.record(),torch.cuda.synchronize(),ts.append(se.elapsed_time(ee)))
print(f"{{2.0*M*N*K/(sum(ts)/len(ts)*1e9):.1f}}")'''

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


def compile_crr_nomidbarrier(M, N, K):
    """Compile CRR with CRR_ENABLE_STEADY_MID_BARRIER=0."""
    outdir = os.path.join(CACHE, f"crr_{M}x{N}x{K}_8wave_nomid")
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
           "-DCRR_ENABLE_STEADY_MID_BARRIER=0",
           *extra,
           *PY_INC.split(), *PY_LD.split(), "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"CRR nomid JIT failed {M}x{N}x{K}:\n{r.stderr[-300:]}")
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
    opt = os.path.join(CACHE, f"crr_{M}x{N}x{K}_8wave_opt")
    if os.path.exists(os.path.join(opt, f"tk_fp8_layouts{EXT}")):
        return opt
    jit = os.path.join(CACHE, f"crr_{M}x{N}x{K}_8wave")
    if os.path.exists(os.path.join(jit, f"tk_fp8_layouts{EXT}")):
        return jit
    return os.path.join(CACHE, "crr_shared")


sys.path.insert(0, DIR)
from bench_vs_hipblaslt import DenseModelConfigs, gen_gemm_test_cases
all_shapes_set = set()
for config in DenseModelConfigs.values():
    for mbs in [1, 2]:
        for _, seq, n, k in gen_gemm_test_cases(config):
            M = seq * mbs
            if M % 256 == 0 and n % 256 == 0 and k % 128 == 0:
                all_shapes_set.add((M, n, k))
shapes = sorted(all_shapes_set)

# Load reference results
results_file = os.path.join(DIR, "bench_jit_full_results.json")
ref_crr = {}
if os.path.exists(results_file):
    with open(results_file) as f:
        ref_crr = json.load(f).get("crr", {})

print(f"Compiling CRR with STEADY_MID_BARRIER=0 for all {len(shapes)} shapes...")
t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as pool:
    futs = {pool.submit(compile_crr_nomidbarrier, M, N, K): (M, N, K) for M, N, K in shapes}
    failed = []
    for fut in futs:
        M, N, K = futs[fut]
        try:
            fut.result()
        except Exception as e:
            print(f"  FAILED {M}x{N}x{K}: {e}")
            failed.append((M, N, K))
print(f"Compilation done in {time.time()-t0:.1f}s ({len(failed)} failures)")
print()

print(f"Benchmarking: current (mid barrier) vs no-mid-barrier...")
print(f"{'Shape':<25} {'gm':>3} {'current':>9} {'no_mid':>9} {'delta':>8} {'ref':>9}")
print("-" * 80)

total_delta = 0
gains = {}
for M, N, K in shapes:
    gm = CRR_GROUP_M.get((M, N, K), 4)
    so_cur = get_current_so(M, N, K)
    so_nom = os.path.join(CACHE, f"crr_{M}x{N}x{K}_8wave_nomid")

    r_cur = bench(M, N, K, so_cur, gm)
    r_nom = bench(M, N, K, so_nom, gm) if os.path.exists(os.path.join(so_nom, f"tk_fp8_layouts{EXT}")) else 0.0

    delta = r_nom - r_cur
    total_delta += delta
    key = f"{M}_{N}_{K}"
    ref = ref_crr.get(key, 0)

    if abs(delta) > 15:
        gains[(M, N, K)] = delta
        flag = " ← GAIN" if delta > 0 else " ← REGRESS"
    else:
        flag = ""

    label = f"{M}x{N}x{K}"
    print(f"{label:<25} {gm:>3} {r_cur:>9.1f} {r_nom:>9.1f} {delta:>+7.1f} {ref:>9.1f}{flag}")
    sys.stdout.flush()

print()
print(f"Total delta: {total_delta:+.1f} TFLOPS avg delta: {total_delta/len(shapes):+.2f}")
print()
print("Shapes with |delta| > 15 TFLOPS:")
for (M, N, K), d in sorted(gains.items(), key=lambda x: -x[1]):
    cur_so_tag = "opt" if "opt" in get_current_so(M, N, K) else "jit"
    print(f"  {M}x{N}x{K}: {d:+.1f} TFLOPS (so={cur_so_tag})")
