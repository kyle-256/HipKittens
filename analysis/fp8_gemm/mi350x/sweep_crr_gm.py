"""
CRR group_m sweep: test gm=4, 8, 16 for all shapes using existing CRR JIT SOs.
No recompilation needed — group_m is a runtime argument.
"""
import subprocess, json, os, sys
from concurrent.futures import ThreadPoolExecutor

DIR = os.path.dirname(os.path.abspath(__file__))
EXT = subprocess.check_output(["python3-config", "--extension-suffix"], text=True).strip()
GPU = os.environ.get("HIP_VISIBLE_DEVICES", "4")
CACHE = os.path.join(DIR, ".jit_cache")

WARMUP, ITERS = 15, 30

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


def bench_crr_gm(M, N, K, gm):
    so_dir = os.path.join(CACHE, f"crr_{M}x{N}x{K}_8wave")
    so = os.path.join(so_dir, f"tk_fp8_layouts{EXT}")
    if not os.path.exists(so):
        return 0.0
    script = bench_tpl.format(M=M, N=N, K=K, gm=gm, w=WARMUP, it=ITERS)
    env = {**os.environ, "HIP_VISIBLE_DEVICES": GPU, "PYTHONPATH": so_dir}
    try:
        r = subprocess.run(["python3", "-c", script], capture_output=True, text=True,
                           cwd=so_dir, env=env, timeout=90)
        if r.returncode == 0:
            return float(r.stdout.strip())
    except Exception:
        pass
    return 0.0


sys.path.insert(0, DIR)
from bench_vs_hipblaslt import DenseModelConfigs, gen_gemm_test_cases

all_shapes = set()
for config in DenseModelConfigs.values():
    for mbs in [1, 2]:
        for _, seq, n, k in gen_gemm_test_cases(config):
            M = seq * mbs
            if M % 256 == 0 and n % 256 == 0 and k % 128 == 0:
                all_shapes.add((M, n, k))
shapes = sorted(all_shapes)

# Only test shapes where CRR JIT SO exists
shapes = [(M, N, K) for M, N, K in shapes
          if os.path.exists(os.path.join(CACHE, f"crr_{M}x{N}x{K}_8wave", f"tk_fp8_layouts{EXT}"))]

print(f"Sweeping CRR group_m {{4,8,16}} for {len(shapes)} shapes (sequential to avoid GPU contention)")
print(f"{'Shape':<30} {'gm=4':>8} {'gm=8':>8} {'gm=16':>8} {'best':>14}")
print("-" * 75)

best_gm = {}
results = {}

for M, N, K in shapes:
    row = {}
    for gm in [4, 8, 16]:
        tf = bench_crr_gm(M, N, K, gm)
        row[gm] = tf
    best = max(row, key=lambda g: row[g])
    best_gm[(M, N, K)] = best
    results[(M, N, K)] = row
    label = f"{M}x{N}x{K}"
    print(f"{label:<30} {row[4]:>8.1f} {row[8]:>8.1f} {row[16]:>8.1f}   {row[best]:>7.1f}(gm={best})")
    sys.stdout.flush()

print("\n" + "=" * 75)
print("\nCRR_GROUP_M = {")
for (M, N, K), gm in sorted(best_gm.items()):
    if gm != 4:
        print(f"    ({M:>5}, {N:>6}, {K:>5}): {gm},")
print("}")

# Estimate improvement
gm4_total = sum(results[s][4] for s in shapes)
best_total = sum(results[s][best_gm[s]] for s in shapes)
print(f"\ngm=4 avg: {gm4_total/len(shapes):.1f} TFLOPS")
print(f"best avg: {best_total/len(shapes):.1f} TFLOPS")
print(f"improvement: +{(best_total-gm4_total)/len(shapes):.1f} TFLOPS/shape ({(best_total/gm4_total-1)*100:.2f}%)")

with open(os.path.join(DIR, "crr_gm_sweep.json"), "w") as f:
    json.dump({f"{M}_{N}_{K}": results[(M,N,K)] for M,N,K in shapes}, f, indent=2)
print("\nRaw results saved to crr_gm_sweep.json")
