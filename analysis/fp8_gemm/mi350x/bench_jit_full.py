"""
Full FP8 GEMM benchmark: JIT-compiled kernels vs hipBLASLt.

RCR: per-shape fastpath via kernel_jit_rcr.cpp (v2 swizzle, 8-wave)
RRR/CRR: full dynamic kernel via kernel_fp8_layouts.cpp (one compile per layout)

Reads hipBLASLt baseline from bench_full_all_layouts.json.
"""
import subprocess, json, os, math, sys, time
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

WARMUP, ITERS = 30, 50
CACHE = os.path.join(DIR, ".jit_cache")

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

from bench_vs_hipblaslt import DenseModelConfigs, gen_gemm_test_cases

all_shapes = set()
for config in DenseModelConfigs.values():
    for mbs in [1, 2]:
        for _, seq, n, k in gen_gemm_test_cases(config):
            M = seq * mbs
            if M % 256 == 0 and n % 256 == 0 and k % 128 == 0:
                all_shapes.add((M, n, k))
shapes = sorted(all_shapes)
layouts = ["rcr", "rrr", "crr"]


def _grid_size(M, N):
    return (M // 256) * (N // 256)


def _can_use_4wave_jit(M, N, K):
    """True iff shape meets 4-wave JIT criteria (alignment + grid threshold)."""
    return (M % 256 == 0 and N % 256 == 0 and K % 128 == 0
            and K >= 256 and _grid_size(M, N) >= 640)


def get_rcr_dir(M, N, K):
    """RCR: per-shape 4-wave JIT if available and eligible, else 8-wave shared."""
    if _can_use_4wave_jit(M, N, K):
        jit_dir = os.path.join(CACHE, f"rcr_{M}x{N}x{K}_4wave")
        so = os.path.join(jit_dir, f"tk_fp8_layouts{EXT}")
        if os.path.exists(so):
            return jit_dir
    return compile_shared("rcr")


def get_crr_dir(M, N, K):
    """CRR: per-shape 4-wave JIT if eligible and cached, else 8-wave shared."""
    if _can_use_4wave_jit(M, N, K):
        jit_dir = os.path.join(CACHE, f"crr_{M}x{N}x{K}_4wave")
        so = os.path.join(jit_dir, f"tk_fp8_layouts{EXT}")
        if os.path.exists(so):
            return jit_dir
    return compile_shared("crr")


def compile_crr_jit(M, N, K):
    """Compile per-shape 4-wave CRR JIT kernel."""
    outdir = os.path.join(CACHE, f"crr_{M}x{N}x{K}_4wave")
    so = os.path.join(outdir, f"tk_fp8_layouts{EXT}")
    if os.path.exists(so):
        return outdir
    os.makedirs(outdir, exist_ok=True)
    cmd = [HIPCXX, os.path.join(DIR, "kernel_jit_crr.cpp"),
           *BASE_FLAGS,
           f"-DM_DIM={M}", f"-DN_DIM={N}", f"-DK_DIM={K}",
           "-DCRR_USE_EXACT_4WAVE_FASTPATH=1",
           *PY_INC.split(), *PY_LD.split(), "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"CRR JIT compile failed {M}x{N}x{K}:\n{r.stderr[-300:]}")
    return outdir


def compile_shared(lay):
    """RRR (and CRR fallback): single compile of full kernel_fp8_layouts.cpp."""
    outdir = os.path.join(CACHE, f"{lay}_shared")
    so = os.path.join(outdir, f"tk_fp8_layouts{EXT}")
    if os.path.exists(so):
        return outdir
    os.makedirs(outdir, exist_ok=True)
    cmd = [HIPCXX, os.path.join(DIR, "kernel_fp8_layouts.cpp"),
           *BASE_FLAGS,
           *PY_INC.split(), *PY_LD.split(), "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"{lay.upper()} compile failed:\n{r.stderr[-300:]}")
    return outdir


def bench_one(M, N, K, lay, so_dir):
    """Run one benchmark point in a subprocess. Returns TFLOPS or 0."""
    script = bench_tpl.format(M=M, N=N, K=K, lay=lay, make_ab=MAKE_AB[lay],
                              w=WARMUP, it=ITERS)
    env = {**os.environ, "HIP_VISIBLE_DEVICES": GPU, "PYTHONPATH": so_dir}
    try:
        r = subprocess.run(["python3", "-c", script], capture_output=True, text=True,
                           cwd=so_dir, env=env, timeout=120)
        if r.returncode == 0:
            return float(r.stdout.strip())
    except Exception:
        pass
    return 0.0


# ---- Phase 1: Compile ----
print(f"[Phase 1] Compiling RCR × {len(shapes)} shapes + RRR/CRR shared...")
t0 = time.time()

# RRR and CRR: one compile each (8-wave shared)
shared_dirs = {}
for lay in ["rrr", "crr"]:
    shared_dirs[lay] = compile_shared(lay)
    print(f"  {lay.upper()} shared kernel ready")

print(f"[Phase 1] Done in {time.time()-t0:.1f}s (RRR/CRR shared ready, RCR uses per-shape JIT cache)")

# ---- Phase 2: Benchmark ----
print(f"\n[Phase 2] Benchmarking {len(shapes)} shapes × {len(layouts)} layouts on GPU{GPU}...")
results = {lay: {} for lay in layouts}

for lay in layouts:
    for i, (M, N, K) in enumerate(shapes):
        key = f"{M}_{N}_{K}"
        if lay == "rcr":
            so_dir = get_rcr_dir(M, N, K)
        else:
            so_dir = shared_dirs.get(lay)
        if not so_dir:
            continue
        tf = bench_one(M, N, K, lay, so_dir)
        results[lay][key] = tf
        print(f"  [{lay.upper()}] {i+1}/{len(shapes)} {M:>5}x{N:>6}x{K:>5} = {tf:>7.1f} TFLOPS")

# ---- Phase 3: Load hipBLASLt baseline ----
bl_file = os.path.join(DIR, "bench_full_all_layouts.json")
bl = {}
if os.path.exists(bl_file):
    with open(bl_file) as f:
        for r in json.load(f)["results"]:
            k = f'{r["M"]}_{r["N"]}_{r["K"]}'
            bl.setdefault(r["layout"], {})[k] = r["bl_tflops"]

# ---- Print summary ----
print(f"\n{'='*160}")
print(f"{'Model':>16} {'Op':>12} {'MBS':>3} {'M':>5} {'N':>6} {'K':>5}", end="")
for lay in ["RCR", "RRR", "CRR"]:
    print(f"  | {'TK':>7} {'BL':>7} {'ratio':>6}", end="")
print()
print("-" * 160)

stats = {l: [] for l in layouts}
for model_name in DenseModelConfigs:
    config = DenseModelConfigs[model_name]
    cases = gen_gemm_test_cases(config)
    for mbs in [1, 2]:
        for op_name, seq, n, k in cases:
            M = seq * mbs
            key = f"{M}_{n}_{k}"
            row = f"{model_name:>16} {op_name:>12} {mbs:>3} {M:>5} {n:>6} {k:>5}"
            for lay in layouts:
                tk = results[lay].get(key, 0)
                b = bl.get(lay, {}).get(key, 0)
                ratio = tk / b if tk > 0 and b > 0 else 0
                if ratio > 0:
                    stats[lay].append(ratio)
                win = "*" if ratio >= 1.0 else " "
                if tk > 0:
                    row += f"  | {tk:>7.1f} {b:>7.1f} {ratio:>.3f}x{win}"
                else:
                    row += f"  | {'':>7} {b:>7.1f} {'':>7}"
            print(row)

print("=" * 160)
print()
print("Summary (geo-mean vs hipBLASLt):")
print(f"{'Layout':>8} {'geo-mean':>10} {'wins':>12} {'avg TFLOPS':>12}")
print("-" * 50)
for lay in layouts:
    vals = stats[lay]
    if vals:
        geo = math.exp(sum(math.log(v) for v in vals) / len(vals))
        wins = sum(1 for v in vals if v >= 1.0)
        tflops_vals = [v for v in results[lay].values() if v > 0]
        avg = sum(tflops_vals) / len(tflops_vals) if tflops_vals else 0
        print(f"{lay.upper():>8} {geo:>9.4f}x  {wins:>4}/{len(vals):<4}  {avg:>10.1f}")

with open(os.path.join(DIR, "bench_jit_full_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to bench_jit_full_results.json")
