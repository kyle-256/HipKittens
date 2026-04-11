"""
Sweep CRR exact-8wave fastpath compile-time parameters:
  CRR_EXACT_PREFETCH_LGKM (default=3)
  CRR_EXACT_B1_LDS_INSERT_AFTER (default=4)
  CRR_EXACT_STEADY_VMCNT (default=CRR_STEADY_VMCNT=4)

Test on representative shapes, pick best combo, apply globally.
"""
import subprocess, os, sys, itertools
from concurrent.futures import ThreadPoolExecutor

DIR = os.path.dirname(os.path.abspath(__file__))
TK = os.environ.get("THUNDERKITTENS_ROOT", os.path.abspath(os.path.join(DIR, "..", "..", "..")))
GPU = os.environ.get("HIP_VISIBLE_DEVICES", "4")
PY_INC = subprocess.check_output(["python3", "-m", "pybind11", "--includes"], text=True).strip()
PY_LD = subprocess.check_output(["python3-config", "--ldflags"], text=True).strip().replace("-lcrypt", "")
EXT = subprocess.check_output(["python3-config", "--extension-suffix"], text=True).strip()
HIPCXX = "/opt/rocm/bin/hipcc"
TMPDIR = "/tmp/crr_params_test"
os.makedirs(TMPDIR, exist_ok=True)

BASE_FLAGS = [
    "-DKITTENS_CDNA4", "--offload-arch=gfx950",
    "-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-ffast-math",
    "-I/opt/rocm/include/rocrand",
    "-DRCR_STEADY_VMCNT=8",
    "-DGEMM_BLOCK_SWIZZLE=1", "-DGEMM_BLOCK_SWIZZLE_NUM_XCDS=8",
    "-std=c++20", "-w", "-shared", "-fPIC",
    f"-I{TK}/include", f"-I{TK}/prototype", "-I/opt/rocm/include/hip",
]

WARMUP, ITERS = 20, 40

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

# Representative shapes (various M, N, K regimes)
test_shapes = [
    (4096,  4096,  4096, 16),  # small, gm=16
    (8192,  4096,  4096,  8),  # medium-small
    (8192,  8192,  8192,  8),  # medium
    (8192,  16384, 16384,  4), # large
    (16384, 16384, 16384,  4), # very large
    (16384, 59136,  8192,  8), # large N
    (8192,  16384, 53248,  8), # very large K
]

# Parameter combos to test
LGKM_vals       = [2, 3, 4]      # default=3
INSERT_vals     = [2, 3, 4, 5, 6] # default=4
VMCNT_vals      = [4, 5, 6]      # default=4 (vmcnt=3 known-bad)

combos = list(itertools.product(LGKM_vals, INSERT_vals, VMCNT_vals))
print(f"Testing {len(combos)} param combos on {len(test_shapes)} shapes")
print(f"(LGKM={LGKM_vals}, INSERT={INSERT_vals}, VMCNT={VMCNT_vals})")


def compile_crr_params(M, N, K, lgkm, insert, vmcnt):
    tag = f"crr_{M}x{N}x{K}_L{lgkm}_I{insert}_V{vmcnt}"
    outdir = os.path.join(TMPDIR, tag)
    so = os.path.join(outdir, f"tk_fp8_layouts{EXT}")
    if os.path.exists(so):
        return outdir
    os.makedirs(outdir, exist_ok=True)
    cmd = [HIPCXX, os.path.join(DIR, "kernel_jit_all.cpp"),
           *BASE_FLAGS,
           f"-DM_DIM={M}", f"-DN_DIM={N}", f"-DK_DIM={K}",
           "-DJIT_LAYOUT=3",
           f"-DCRR_EXACT_PREFETCH_LGKM={lgkm}",
           f"-DCRR_EXACT_B1_LDS_INSERT_AFTER={insert}",
           f"-DCRR_EXACT_STEADY_VMCNT={vmcnt}",
           *PY_INC.split(), *PY_LD.split(), "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"FAILED {tag}:\n{r.stderr[-300:]}")
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


# Step 1: Compile all combos for each shape in parallel
print("\n[Phase 1] Compiling...")
compile_jobs = []
for M, N, K, gm in test_shapes:
    for lgkm, insert, vmcnt in combos:
        compile_jobs.append((M, N, K, lgkm, insert, vmcnt))

with ThreadPoolExecutor(max_workers=12) as pool:
    futs = {pool.submit(compile_crr_params, M, N, K, l, i, v): (M, N, K, l, i, v)
            for M, N, K, l, i, v in compile_jobs}
    done = 0
    for fut in futs:
        try:
            fut.result()
        except Exception as e:
            M, N, K, l, i, v = futs[fut]
            print(f"  FAILED {M}x{N}x{K} L={l} I={i} V={v}: {e}")
        done += 1
        if done % 20 == 0:
            print(f"  {done}/{len(compile_jobs)} compiled")
print(f"  All {len(compile_jobs)} compiled.")

# Step 2: Benchmark default (L=3,I=4,V=4) for each shape first (baseline)
print("\n[Phase 2] Benchmarking...")
baselines = {}
for M, N, K, gm in test_shapes:
    d = os.path.join(TMPDIR, f"crr_{M}x{N}x{K}_L3_I4_V4")
    baselines[(M, N, K)] = bench(M, N, K, d, gm)
    print(f"  baseline {M}x{N}x{K}: {baselines[(M,N,K)]:.1f} TFLOPS")

# Step 3: Benchmark all combos per shape
print("\n[Phase 3] Measuring all combos (sequential to avoid GPU contention)...")
results = {}  # (M,N,K,lgkm,insert,vmcnt) -> TFLOPS

for M, N, K, gm in test_shapes:
    best_tf = 0.0
    best_combo = None
    for lgkm, insert, vmcnt in combos:
        d = os.path.join(TMPDIR, f"crr_{M}x{N}x{K}_L{lgkm}_I{insert}_V{vmcnt}")
        tf = bench(M, N, K, d, gm)
        results[(M, N, K, lgkm, insert, vmcnt)] = tf
        if tf > best_tf:
            best_tf = tf
            best_combo = (lgkm, insert, vmcnt)
    base = baselines[(M, N, K)]
    pct = (best_tf / base - 1) * 100 if base > 0 else 0
    print(f"  {M}x{N}x{K}: best={best_tf:.1f}(L={best_combo[0]},I={best_combo[1]},V={best_combo[2]}) "
          f"vs default={base:.1f} ({pct:+.2f}%)")
    sys.stdout.flush()

# Step 4: Find best combo averaged across shapes
print("\n[Phase 4] Ranking combos by avg TFLOPS...")
combo_scores = {}
for lgkm, insert, vmcnt in combos:
    total = 0.0
    count = 0
    for M, N, K, gm in test_shapes:
        tf = results.get((M, N, K, lgkm, insert, vmcnt), 0)
        if tf > 0:
            total += tf
            count += 1
    combo_scores[(lgkm, insert, vmcnt)] = total / count if count > 0 else 0

# Print top 10
ranked = sorted(combo_scores.items(), key=lambda x: x[1], reverse=True)
print(f"\nTop 10 combos (avg TFLOPS over {len(test_shapes)} shapes):")
default_score = combo_scores.get((3, 4, 4), 0)
for (lgkm, insert, vmcnt), score in ranked[:10]:
    pct = (score / default_score - 1) * 100 if default_score > 0 else 0
    flag = " <-- DEFAULT" if (lgkm, insert, vmcnt) == (3, 4, 4) else ""
    print(f"  L={lgkm} I={insert} V={vmcnt}: {score:.1f} TFLOPS ({pct:+.2f}%){flag}")
