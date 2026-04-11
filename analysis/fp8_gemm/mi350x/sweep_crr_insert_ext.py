"""
Extended CRR INSERT_AFTER sweep: test INSERT=6,7,8 for worst CRR/RCR shapes.

Initial sweep (sweep_crr_params.py) tested INSERT=[2,3,4,5,6], VMCNT=[4,5,6].
Per-shape sweep (sweep_crr_all_shapes.py) tested only 6 top combos from initial sweep.
INSERT=6,7,8 and VMCNT=7,8 have NEVER been tested for any shape.
The kernel supports INSERT_AFTER in [0,8] (static_assert confirmed).

This script tests:
  LGKM = [2, 3, 4]
  INSERT = [5, 6, 7, 8]   ← 6,7,8 are NEW
  VMCNT = [5, 6, 7, 8]    ← 7,8 are NEW

For the 10 worst CRR/RCR shapes. Full bench params (25w/40i).
"""
import subprocess, os, sys, json, time, itertools
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
TMPDIR = os.path.join(DIR, ".crr_ext_sweep")
os.makedirs(TMPDIR, exist_ok=True)
WARMUP, ITERS = 25, 40

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

# Current CRR_GROUP_M from bench_jit_full.py
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

# Current CRR_SHAPE_PARAMS from bench_jit_full.py
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

# Target: worst CRR/RCR shapes (by absolute gap in TFLOPS, from bench_jit_full_results.json)
# Includes both shapes NOT in CRR_SHAPE_PARAMS (only 6 combos tested) and
# shapes IN CRR_SHAPE_PARAMS that are still far below target.
test_shapes = [
    # NOT in CRR_SHAPE_PARAMS — only 6 combos from initial sweep tested (4,4,4 was best):
    (16384, 16384, 16384),  # CRR=2980.5, RCR=3282.6, gap=302.1 TFLOPS
    ( 8192, 18432, 16384),  # CRR=3009.5, RCR=3306.9, gap=297.4 TFLOPS
    (16384, 18432, 16384),  # CRR=2970.5, RCR=3281.4, gap=310.9 TFLOPS
    ( 8192, 57344,  8192),  # CRR=2877.6, RCR=3171.2, gap=293.6 TFLOPS
    ( 8192, 59136,  8192),  # CRR=2873.8, RCR=3163.8, gap=290.0 TFLOPS
    ( 4096, 57344,  8192),  # CRR=2840.2, RCR=3127.5, gap=287.3 TFLOPS
    # IN CRR_SHAPE_PARAMS but large absolute gaps:
    (16384, 16384, 53248),  # CRR=2856.5, RCR=3231.3, gap=374.8 TFLOPS (WORST)
    ( 8192, 16384, 16384),  # CRR=2998.1, RCR=3300.8, gap=302.7 TFLOPS
    ( 8192, 16384, 53248),  # CRR=2973.0, RCR=3273.7, gap=300.7 TFLOPS
    ( 8192, 106496, 16384), # CRR=2898.5, RCR=3250.5, gap=352.0 TFLOPS
]

# Extended combos: INSERT=5-8 (6,7,8 are NEW), VMCNT=5-8 (7,8 are NEW)
# LGKM=1-4 (LGKM=1 is new for most shapes)
LGKM_vals  = [1, 2, 3, 4]
INSERT_vals = [5, 6, 7, 8]   # 6,7,8 never tested before
VMCNT_vals  = [5, 6, 7, 8]   # 7,8 never tested before

ALL_COMBOS = list(itertools.product(LGKM_vals, INSERT_vals, VMCNT_vals))
print(f"Testing {len(ALL_COMBOS)} extended combos per shape ({len(test_shapes)} shapes = {len(ALL_COMBOS)*len(test_shapes)} total compilations)")
print(f"LGKM={LGKM_vals}, INSERT={INSERT_vals}, VMCNT={VMCNT_vals}")
print()


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
        raise RuntimeError(f"FAILED {tag}:\n{r.stderr[-200:]}")
    return outdir


def get_current_so(M, N, K):
    """Get current best CRR SO (opt > 8wave > shared)."""
    opt = os.path.join(CACHE, f"crr_{M}x{N}x{K}_8wave_opt")
    if os.path.exists(os.path.join(opt, f"tk_fp8_layouts{EXT}")):
        return opt, "opt"
    jit = os.path.join(CACHE, f"crr_{M}x{N}x{K}_8wave")
    if os.path.exists(os.path.join(jit, f"tk_fp8_layouts{EXT}")):
        return jit, "8wave"
    return os.path.join(CACHE, "crr_shared"), "shared"


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


# Step 1: Compile all combos in parallel
print("[Phase 1] Compiling all combos in parallel...")
t0 = time.time()
compile_jobs = [(M, N, K, l, i, v) for M, N, K in test_shapes for l, i, v in ALL_COMBOS]
failed = []
with ThreadPoolExecutor(max_workers=8) as pool:
    futs = {pool.submit(compile_crr_params, M, N, K, l, i, v): (M, N, K, l, i, v)
            for M, N, K, l, i, v in compile_jobs}
    done = 0
    for fut in futs:
        M, N, K, l, i, v = futs[fut]
        try:
            fut.result()
        except Exception as e:
            print(f"  FAILED {M}x{N}x{K} L={l} I={i} V={v}: {e}")
            failed.append((M, N, K, l, i, v))
        done += 1
        if done % 50 == 0:
            print(f"  {done}/{len(compile_jobs)} compiled...", flush=True)
print(f"[Phase 1] Done in {time.time()-t0:.1f}s ({len(failed)} failures)")
print()

# Load current results for reference
results_file = os.path.join(DIR, "bench_jit_full_results.json")
ref_crr = {}
ref_rcr = {}
if os.path.exists(results_file):
    with open(results_file) as f:
        data = json.load(f)
        ref_crr = data.get("crr", {})
        ref_rcr = data.get("rcr", {})

# Step 2: Benchmark — per shape, get current baseline then test all combos
print("[Phase 2] Benchmarking (sequential per shape)...")
gains = {}  # (M,N,K) -> (best_lgkm, best_insert, best_vmcnt, best_tf, cur_tf)

for M, N, K in test_shapes:
    gm = CRR_GROUP_M.get((M, N, K), 4)
    cur_so, cur_type = get_current_so(M, N, K)
    cur_params = CRR_SHAPE_PARAMS.get((M, N, K), None)
    key = f"{M}_{N}_{K}"
    ref = ref_crr.get(key, 0)
    rcr = ref_rcr.get(key, 0)

    print(f"\n--- {M}x{N}x{K} (gm={gm}, cur_so={cur_type}, ref_crr={ref:.1f}, ref_rcr={rcr:.1f}) ---")
    if cur_params:
        print(f"  Current params: L={cur_params[0]} I={cur_params[1]} V={cur_params[2]}")
    else:
        print(f"  Current params: (4,4,4) default")

    # Benchmark current best
    r_cur = bench(M, N, K, cur_so, gm)
    print(f"  Current SO: {r_cur:.1f} TFLOPS")

    best_tf = r_cur
    best_combo = None

    for lgkm, insert, vmcnt in ALL_COMBOS:
        if (M, N, K, lgkm, insert, vmcnt) in [(m,n,k,l,i,v) for m,n,k,l,i,v in failed]:
            continue
        so_dir = os.path.join(TMPDIR, f"crr_{M}x{N}x{K}_L{lgkm}_I{insert}_V{vmcnt}")
        so_file = os.path.join(so_dir, f"tk_fp8_layouts{EXT}")
        if not os.path.exists(so_file):
            continue
        tf = bench(M, N, K, so_dir, gm)
        delta = tf - r_cur
        flag = " ← GAIN" if delta > 15 else (" ← REGRESS" if delta < -15 else "")
        print(f"  L={lgkm} I={insert} V={vmcnt}: {tf:.1f} ({delta:+.1f}){flag}", flush=True)
        if tf > best_tf:
            best_tf = tf
            best_combo = (lgkm, insert, vmcnt)

    if best_combo:
        delta = best_tf - r_cur
        print(f"  BEST: L={best_combo[0]} I={best_combo[1]} V={best_combo[2]} → +{delta:.1f} TFLOPS over current")
        gains[(M, N, K)] = (best_combo[0], best_combo[1], best_combo[2], best_tf, r_cur, delta)
    else:
        print(f"  No improvement found over current ({r_cur:.1f} TFLOPS)")
    sys.stdout.flush()

print("\n" + "=" * 60)
print("SUMMARY: Shapes where extended params improve >15 TFLOPS:")
for (M, N, K), (l, i, v, best, cur, delta) in sorted(gains.items(), key=lambda x: -x[1][5]):
    print(f"  {M}x{N}x{K}: L={l} I={i} V={v} → {cur:.1f} + {delta:.1f} = {best:.1f} TFLOPS")
    print(f"    Add to CRR_SHAPE_PARAMS: ({M:>5}, {N:>6}, {K:>5}): ({l}, {i}, {v}),  # +{delta/cur*100:.1f}%")
if not gains:
    print("  None found — INSERT=6,7,8 and VMCNT=7,8 do not help beyond current params")
