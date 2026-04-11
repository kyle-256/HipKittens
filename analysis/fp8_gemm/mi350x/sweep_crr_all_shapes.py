"""
Per-shape CRR param tuning: sweep top 4 combos across all 48 shapes.
Top combos from initial sweep:
  L=4,I=4,V=4: best global avg (+0.89%)
  L=4,I=5,V=5: good for large-N shapes
  L=3,I=4,V=5: good for medium shapes
  L=2,I=3,V=5: good for large shapes
Output: per-shape best params for bench_jit_full.py CRR_SHAPE_PARAMS dict.
"""
import subprocess, json, os, sys
from concurrent.futures import ThreadPoolExecutor

DIR = os.path.dirname(os.path.abspath(__file__))
TK = os.environ.get("THUNDERKITTENS_ROOT", os.path.abspath(os.path.join(DIR, "..", "..", "..")))
GPU = os.environ.get("HIP_VISIBLE_DEVICES", "4")
PY_INC = subprocess.check_output(["python3", "-m", "pybind11", "--includes"], text=True).strip()
PY_LD = subprocess.check_output(["python3-config", "--ldflags"], text=True).strip().replace("-lcrypt", "")
EXT = subprocess.check_output(["python3-config", "--extension-suffix"], text=True).strip()
HIPCXX = "/opt/rocm/bin/hipcc"
TMPDIR = "/tmp/crr_all_shapes"
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

# CRR_GROUP_M from sweep
CRR_GROUP_M = {
    ( 4096,   4096,  4096): 16,
    ( 4096,   4096, 14336):  8,
    ( 4096,   8192,  8192):  8,
    ( 4096,  10240,  8192):  8,
    ( 4096,  22016,  4096):  8,
    ( 8192,   3584,  3584): 16,
    ( 8192,   3584, 18944):  8,
    ( 8192,   4096,  4096):  8,
    ( 8192,   8192,  8192):  8,
    ( 8192,   8192, 28672):  8,
    ( 8192,   8192, 29568):  8,
    ( 8192,  12288,  4096):  8,
    ( 8192,  16384, 53248):  8,
    ( 8192,  37888,  3584):  8,
    (16384,   3584,  3584):  8,
    (16384,   4096,  4096):  8,
    (16384,  10240,  8192):  8,
    (16384,  28672,  4096):  8,
    (16384,  37888,  3584):  8,
    (16384,  59136,  8192):  8,
}

# Top combos from initial sweep (LGKM, INSERT_AFTER, VMCNT)
COMBOS = [
    (4, 4, 4),  # global best
    (4, 5, 5),  # large-N
    (3, 4, 5),  # medium
    (2, 3, 5),  # large
    (4, 3, 4),  # small variant
    (3, 5, 6),  # alternative
]


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
print(f"Total shapes: {len(shapes)}")
print(f"Combos to test: {len(COMBOS)}")
print(f"Total compilations: {len(shapes) * len(COMBOS)}")

# Phase 1: Compile all combos for all shapes in parallel
print("\n[Phase 1] Compiling...")
compile_jobs = [(M, N, K, l, i, v) for M, N, K in shapes for l, i, v in COMBOS]
with ThreadPoolExecutor(max_workers=12) as pool:
    futs = {pool.submit(compile_crr_params, M, N, K, l, i, v): (M, N, K, l, i, v)
            for M, N, K, l, i, v in compile_jobs}
    done = 0
    for fut in futs:
        try:
            fut.result()
        except Exception as e:
            M, N, K, l, i, v = futs[fut]
            print(f"  FAILED {M}x{N}x{K} L={l}I={i}V={v}: {e}")
        done += 1
        if done % 50 == 0:
            print(f"  {done}/{len(compile_jobs)} compiled")
print(f"  All {len(compile_jobs)} compiled.")

# Phase 2: Benchmark all combos per shape sequentially
print("\n[Phase 2] Benchmarking per shape...")
shape_best = {}  # (M,N,K) -> (lgkm, insert, vmcnt, tf)
results = {}  # (M,N,K,l,i,v) -> tf

print(f"{'Shape':<30} " + " ".join(f"L{l}I{i}V{v}" for l,i,v in COMBOS) + " best")
print("-" * 120)

for M, N, K in shapes:
    gm = CRR_GROUP_M.get((M, N, K), 4)
    row = {}
    for lgkm, insert, vmcnt in COMBOS:
        d = os.path.join(TMPDIR, f"crr_{M}x{N}x{K}_L{lgkm}_I{insert}_V{vmcnt}")
        tf = bench(M, N, K, d, gm)
        row[(lgkm, insert, vmcnt)] = tf
        results[(M, N, K, lgkm, insert, vmcnt)] = tf
    best_combo = max(row, key=lambda c: row[c])
    best_tf = row[best_combo]
    global_tf = row.get((4, 4, 4), 0)
    shape_best[(M, N, K)] = (best_combo, best_tf)
    label = f"{M}x{N}x{K}"
    vals = " ".join(f"{row[c]:>8.1f}" for c in COMBOS)
    delta = (best_tf / global_tf - 1) * 100 if global_tf > 0 else 0
    print(f"{label:<30} {vals}  L{best_combo[0]}I{best_combo[1]}V{best_combo[2]}({delta:+.1f}%vs_L4)")
    sys.stdout.flush()

# Phase 3: Print CRR_SHAPE_PARAMS dict
print("\n" + "=" * 80)
print("\n# Per-shape CRR fastpath params (LGKM, INSERT_AFTER, VMCNT)")
print("# Global default: L=4, I=4, V=4 (best average). Only list deviations.")
print("CRR_SHAPE_PARAMS = {")
for M, N, K in shapes:
    (lgkm, insert, vmcnt), best_tf = shape_best[(M, N, K)]
    global_tf = results.get((M, N, K, 4, 4, 4), 0)
    pct = (best_tf / global_tf - 1) * 100 if global_tf > 0 else 0
    if (lgkm, insert, vmcnt) != (4, 4, 4) and pct > 0.2:
        print(f"    ({M:>5}, {N:>6}, {K:>5}): ({lgkm}, {insert}, {vmcnt}),  # +{pct:.1f}%")
print("}")

# Phase 4: Estimate total gain
global_total = sum(results.get((M, N, K, 4, 4, 4), 0) for M, N, K in shapes)
best_total = sum(shape_best[(M, N, K)][1] for M, N, K in shapes)
print(f"\nGlobal-L4 avg:  {global_total/len(shapes):.1f} TFLOPS")
print(f"Per-shape best: {best_total/len(shapes):.1f} TFLOPS")
print(f"Improvement: +{(best_total-global_total)/len(shapes):.1f} TFLOPS/shape ({(best_total/global_total-1)*100:.2f}%)")

with open(os.path.join(DIR, "crr_all_shapes_params.json"), "w") as f:
    json.dump({f"{M}_{N}_{K}": {f"L{l}I{i}V{v}": results.get((M,N,K,l,i,v),0) for l,i,v in COMBOS}
               for M,N,K in shapes}, f, indent=2)
print("Raw results saved to crr_all_shapes_params.json")
