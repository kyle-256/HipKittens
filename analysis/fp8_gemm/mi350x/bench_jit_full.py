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
fn=lambda:m.gemm_{lay}(A,B,C,1.0,1.0,{group_m})
for _ in range({w}):C.zero_();fn()
se=torch.cuda.Event(enable_timing=True);ee=torch.cuda.Event(enable_timing=True);ts=[]
for _ in range({it}):(torch.cuda.synchronize(),se.record(),fn(),ee.record(),torch.cuda.synchronize(),ts.append(se.elapsed_time(ee)))
print(f"{{2.0*M*N*K/(sum(ts)/len(ts)*1e9):.1f}}")'''

# Per-shape tuned group_m for RCR (from sweep: best of gm=4,8,16)
# Default is 4 for shapes not listed.
RCR_GROUP_M = {
    # shapes where gm > 4 helps
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


def compile_rcr_8wave_jit(M, N, K):
    """RCR: per-shape 8-wave JIT via kernel_jit_rcr.cpp (compile-time k_iters)."""
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
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"RCR 8wave JIT failed {M}x{N}x{K}:\n{r.stderr[-300:]}")
    return outdir


# Shapes where 8-wave JIT outperforms 4-wave JIT (benchmarked, micro-bench verified).
# Pattern: large K with non-standard iteration counts where compile-time K unrolling
# and ping-pong scheduling yield better memory/compute overlap than 4-wave interleave.
# K=11008/18944: scheduling benefit for odd K iteration counts in small-grid shapes.
# K=29568: +3.5-3.8% vs 4-wave; large K with 231 iterations benefits from ping-pong.
RCR_8WAVE_JIT_SHAPES = {
    (4096, 4096, 11008), (8192, 4096, 11008), (8192, 3584, 18944),
    (8192, 8192, 29568), (16384, 8192, 29568),
}

# K values where RRR JIT (UNROLL=4, compile-time k_iters) beats shared kernel.
# Empirically: K in {3584,4096,8192,11008,18944} give +2-6% gains.
# K >= 14336 regresses (-5 to -50%) due to I-cache overflow from UNROLL=4.
RRR_JIT_K_GOOD = {3584, 4096, 8192, 11008, 18944}


def get_rcr_dir(M, N, K):
    """RCR: 8-wave JIT (select small-grid) → 4-wave JIT (grid≥640) → 8-wave shared."""
    # 8-wave JIT checked FIRST for select small-grid shapes where it outperforms 4-wave
    if (M, N, K) in RCR_8WAVE_JIT_SHAPES:
        jit_8wave = os.path.join(CACHE, f"rcr_{M}x{N}x{K}_8wave")
        if os.path.exists(os.path.join(jit_8wave, f"tk_fp8_layouts{EXT}")):
            return jit_8wave
    jit_4wave = os.path.join(CACHE, f"rcr_{M}x{N}x{K}_4wave")
    if os.path.exists(os.path.join(jit_4wave, f"tk_fp8_layouts{EXT}")):
        return jit_4wave
    return compile_shared("rcr")


# Per-shape CRR fastpath params (LGKM, INSERT_AFTER, VMCNT).
# Global default: L=4, I=4, V=4 (from sweep: +0.89% avg vs default L=3,I=4,V=4).
# Only list shapes where per-shape optimal beats L=4,I=4,V=4 by >0.2%.
CRR_SHAPE_PARAMS = {
    ( 4096,   4096,  4096): (4, 5, 5),  # +2.4%
    ( 4096,   4096, 14336): (2, 3, 5),  # +1.3%
    ( 4096,   8192,  8192): (4, 3, 4),  # +1.2%
    ( 4096,  10240,  8192): (4, 3, 4),  # +0.9%
    ( 4096,  12288,  4096): (2, 3, 5),  # +0.9%
    ( 4096,  22016,  4096): (4, 5, 5),  # +0.3%
    ( 4096,  28672,  4096): (4, 3, 4),  # +0.3%
    ( 8192,   3584, 18944): (4, 3, 4),  # +0.6%
    ( 8192,   4096,  4096): (2, 3, 5),  # +0.2%
    ( 8192,   4096, 11008): (3, 5, 6),  # +0.4%
    ( 8192,   4608,  3584): (3, 4, 5),  # +2.4%
    ( 8192,   8192,  8192): (4, 5, 5),  # +0.7%
    ( 8192,   8192, 28672): (3, 5, 6),  # +0.4%
    ( 8192,   8192, 29568): (3, 5, 6),  # +0.4%
    ( 8192,  10240,  8192): (4, 3, 4),  # +0.4%
    ( 8192,  12288,  4096): (3, 4, 5),  # +0.3%
    ( 8192,  16384, 16384): (2, 3, 5),  # +0.3%
    ( 8192,  16384, 53248): (2, 3, 5),  # +0.6%
    ( 8192,  22016,  4096): (3, 5, 6),  # +0.4%
    ( 8192,  37888,  3584): (4, 5, 5),  # +0.5%
    ( 8192, 106496, 16384): (4, 3, 4),  # +1.1%
    (16384,   3584,  3584): (4, 3, 4),  # +0.3%
    (16384,   3584, 18944): (3, 5, 6),  # +0.9%
    (16384,   4096, 14336): (3, 5, 6),  # +0.3%
    (16384,   4608,  3584): (4, 3, 4),  # +0.5%
    (16384,   6144,  4096): (3, 5, 6),  # +0.8%
    (16384,   8192, 29568): (2, 3, 5),  # +0.4%
    (16384,  10240,  8192): (4, 3, 4),  # +0.3%
    (16384,  16384, 53248): (3, 4, 5),  # +1.7%
    (16384, 106496, 16384): (2, 3, 5),  # +1.5%
    # Extended sweep (INSERT=5-8, VMCNT=5-8, LGKM=1-4): new optima found 2026-04-10
    ( 4096,  57344,  8192): (2, 5, 8),  # +0.44%
    (16384,  16384, 16384): (1, 8, 6),  # +0.33%
    (16384,  18432, 16384): (2, 8, 7),  # +0.23%
    ( 8192,  59136,  8192): (4, 6, 8),  # +0.18%
    ( 8192,  57344,  8192): (4, 7, 5),  # +0.17%
    ( 8192,  18432, 16384): (3, 8, 5),  # +0.11%
}


def get_crr_dir(M, N, K):
    """CRR: per-shape opt JIT → L=4 8wave JIT → shared."""
    # Shape-specific optimal params
    opt_dir = os.path.join(CACHE, f"crr_{M}x{N}x{K}_8wave_opt")
    if os.path.exists(os.path.join(opt_dir, f"tk_fp8_layouts{EXT}")):
        return opt_dir
    # L=4 global default (CRR_EXACT_PREFETCH_LGKM=4)
    jit_dir = os.path.join(CACHE, f"crr_{M}x{N}x{K}_8wave")
    if os.path.exists(os.path.join(jit_dir, f"tk_fp8_layouts{EXT}")):
        return jit_dir
    return compile_shared("crr")


def compile_crr_8wave_jit(M, N, K):
    """CRR: per-shape 8-wave JIT, global default L=4,I=4,V=4."""
    outdir = os.path.join(CACHE, f"crr_{M}x{N}x{K}_8wave")
    so = os.path.join(outdir, f"tk_fp8_layouts{EXT}")
    if os.path.exists(so):
        return outdir
    os.makedirs(outdir, exist_ok=True)
    cmd = [HIPCXX, os.path.join(DIR, "kernel_jit_all.cpp"),
           *BASE_FLAGS,
           f"-DM_DIM={M}", f"-DN_DIM={N}", f"-DK_DIM={K}",
           "-DJIT_LAYOUT=3",
           "-DCRR_EXACT_PREFETCH_LGKM=4",   # global best from sweep (+0.89% avg)
           *PY_INC.split(), *PY_LD.split(), "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"CRR 8wave JIT failed {M}x{N}x{K}:\n{r.stderr[-300:]}")
    return outdir


def compile_crr_opt_jit(M, N, K, lgkm, insert, vmcnt):
    """CRR: per-shape optimal JIT with tuned LGKM/INSERT/VMCNT."""
    outdir = os.path.join(CACHE, f"crr_{M}x{N}x{K}_8wave_opt")
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
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"CRR opt JIT failed {M}x{N}x{K}:\n{r.stderr[-300:]}")
    return outdir


def compile_rrr_8wave_jit(M, N, K):
    """RRR: per-shape 8-wave JIT via kernel_jit_all.cpp (compile-time k_iters + XCD swizzle)."""
    outdir = os.path.join(CACHE, f"rrr_{M}x{N}x{K}_8wave")
    so = os.path.join(outdir, f"tk_fp8_layouts{EXT}")
    if os.path.exists(so):
        return outdir
    os.makedirs(outdir, exist_ok=True)
    cmd = [HIPCXX, os.path.join(DIR, "kernel_jit_all.cpp"),
           *BASE_FLAGS,
           f"-DM_DIM={M}", f"-DN_DIM={N}", f"-DK_DIM={K}",
           "-DJIT_LAYOUT=2",
           *PY_INC.split(), *PY_LD.split(), "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"RRR 8wave JIT failed {M}x{N}x{K}:\n{r.stderr[-300:]}")
    return outdir


def get_rrr_dir(M, N, K):
    """RRR: per-shape 8-wave JIT for small K (UNROLL=4 helps), else 8-wave shared."""
    if K in RRR_JIT_K_GOOD:
        jit_dir = os.path.join(CACHE, f"rrr_{M}x{N}x{K}_8wave")
        so = os.path.join(jit_dir, f"tk_fp8_layouts{EXT}")
        if os.path.exists(so):
            return jit_dir
    return compile_shared("rrr")


def compile_shared(lay):
    """Shared fallback: single compile of full kernel_fp8_layouts.cpp."""
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


# Per-shape tuned group_m for CRR.
# Cross-validated with both sweep (25w/40i) and full bench (30w/50i).
# Shapes not listed use gm=4 (default). Confirmed changes vs original:
#   gm=4: (4096,4096,4096)+39.6, (8192,3584,3584)+30.7, (8192,4096,4096)+11.0,
#          (8192,8192,28672)+1.9, (8192,16384,53248)+3.9
#   gm=8: (16384,4608,3584)+17.6, (16384,6144,4096)+11.3, (4096,28672,4096)+8.5,
#          (8192,10240,8192)+3.4, (8192,4096,14336)+11.8
#   gm=16: (16384,59136,8192) tested but regressed in full bench (-57T vs micro-bench +52T)
CRR_GROUP_M = {
    ( 4096,   4096, 14336):  8,
    ( 4096,   6144,  4096):  8,
    ( 4096,   8192,  8192):  8,
    ( 4096,  10240,  8192):  8,
    ( 4096,  22016,  4096):  8,
    ( 4096,  28672,  4096):  8,
    ( 8192,   3584, 18944):  8,
    ( 8192,   4096, 14336):  8,
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


def bench_one(M, N, K, lay, so_dir):
    """Run one benchmark point in a subprocess. Returns TFLOPS or 0."""
    if lay == "rcr":
        gm = RCR_GROUP_M.get((M, N, K), 4)
    elif lay == "crr":
        gm = CRR_GROUP_M.get((M, N, K), 4)
    else:
        gm = 4
    script = bench_tpl.format(M=M, N=N, K=K, lay=lay, make_ab=MAKE_AB[lay],
                              group_m=gm, w=WARMUP, it=ITERS)
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
print(f"[Phase 1] Compiling: RCR JIT + RRR {len(shapes)} 8wave JIT + CRR {len(shapes)} 8wave JIT...")
t0 = time.time()

# Shared fallbacks (for CRR and RRR if JIT fails)
shared_dirs = {}
shared_dirs["rrr"] = compile_shared("rrr")
print(f"  RRR shared fallback ready")
shared_dirs["crr"] = compile_shared("crr")
print(f"  CRR shared fallback ready")

# RCR 8-wave JIT for select small-grid shapes (outperforms shared for K=11008, K=18944)
rcr_8wave_needed = [(M, N, K) for M, N, K in RCR_8WAVE_JIT_SHAPES if (M, N, K) in set(shapes)]
if rcr_8wave_needed:
    print(f"  Compiling RCR 8-wave JIT for {len(rcr_8wave_needed)} select shapes...")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(compile_rcr_8wave_jit, M, N, K): (M, N, K) for M, N, K in rcr_8wave_needed}
        for fut in futs:
            try: fut.result()
            except Exception as e:
                M, N, K = futs[fut]
                print(f"  RCR 8wave JIT FAILED {M}x{N}x{K}: {e}")
    print(f"  RCR 8-wave JIT done")

# Compile RRR 8-wave JIT for shapes with small K only (UNROLL=4 helps for K in RRR_JIT_K_GOOD)
rrr_jit_shapes = [(M, N, K) for M, N, K in shapes if K in RRR_JIT_K_GOOD]
print(f"  Compiling RRR 8-wave JIT for {len(rrr_jit_shapes)}/{len(shapes)} shapes (K in {{3584,4096,8192,11008,18944}})...")
with ThreadPoolExecutor(max_workers=8) as pool:
    futs = {pool.submit(compile_rrr_8wave_jit, M, N, K): (M, N, K) for M, N, K in rrr_jit_shapes}
    for fut in futs:
        try:
            fut.result()
        except Exception as e:
            M, N, K = futs[fut]
            print(f"  RRR 8wave JIT FAILED {M}x{N}x{K}: {e}")
print(f"  RRR 8wave JIT compilation done")

# Compile CRR 8-wave JIT for all shapes (global default L=4,I=4,V=4)
print(f"  Compiling CRR 8-wave JIT for {len(shapes)} shapes (LGKM=4 global default)...")
with ThreadPoolExecutor(max_workers=8) as pool:
    futs = {pool.submit(compile_crr_8wave_jit, M, N, K): (M, N, K) for M, N, K in shapes}
    for fut in futs:
        try:
            fut.result()
        except Exception as e:
            M, N, K = futs[fut]
            print(f"  CRR 8wave JIT FAILED {M}x{N}x{K}: {e}")
print(f"  CRR 8wave JIT compilation done")

# Compile per-shape optimal CRR JIT for shapes with tuned params
crr_opt_shapes = [(M, N, K) for M, N, K in shapes if (M, N, K) in CRR_SHAPE_PARAMS]
if crr_opt_shapes:
    print(f"  Compiling CRR opt JIT for {len(crr_opt_shapes)} shapes with per-shape params...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(compile_crr_opt_jit, M, N, K, *CRR_SHAPE_PARAMS[(M, N, K)]): (M, N, K)
                for M, N, K in crr_opt_shapes}
        for fut in futs:
            try:
                fut.result()
            except Exception as e:
                M, N, K = futs[fut]
                print(f"  CRR opt JIT FAILED {M}x{N}x{K}: {e}")
    print(f"  CRR opt JIT compilation done")

print(f"[Phase 1] Done in {time.time()-t0:.1f}s")

# ---- Phase 2: Benchmark ----
print(f"\n[Phase 2] Benchmarking {len(shapes)} shapes × {len(layouts)} layouts on GPU{GPU}...")
results = {lay: {} for lay in layouts}

for lay in layouts:
    for i, (M, N, K) in enumerate(shapes):
        key = f"{M}_{N}_{K}"
        if lay == "rcr":
            so_dir = get_rcr_dir(M, N, K)
        elif lay == "crr":
            so_dir = get_crr_dir(M, N, K)
        elif lay == "rrr":
            so_dir = get_rrr_dir(M, N, K)
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
