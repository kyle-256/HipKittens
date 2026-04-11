"""
Re-sweep CRR group_m (gm=4,8,16) using CURRENT JIT SOs (L=4 exact fastpath).
Previous sweep used shared SO (dynamic kernel). Now JIT SOs might prefer different gm.
Output: updated CRR_GROUP_M dict for bench_jit_full.py.
"""
import subprocess, os, sys

DIR = os.path.dirname(os.path.abspath(__file__))
GPU = os.environ.get("HIP_VISIBLE_DEVICES", "4")
EXT = subprocess.check_output(["python3-config", "--extension-suffix"], text=True).strip()
CACHE = os.path.join(DIR, ".jit_cache")

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


def get_best_so(M, N, K):
    """Return the best available SO dir: opt > 8wave > shared."""
    opt = os.path.join(CACHE, f"crr_{M}x{N}x{K}_8wave_opt")
    if os.path.exists(os.path.join(opt, f"tk_fp8_layouts{EXT}")):
        return opt, "opt"
    jit = os.path.join(CACHE, f"crr_{M}x{N}x{K}_8wave")
    if os.path.exists(os.path.join(jit, f"tk_fp8_layouts{EXT}")):
        return jit, "jit"
    return os.path.join(CACHE, "crr_shared"), "shared"


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
print(f"Sweeping gm=4,8,16 for {len(shapes)} shapes using current best JIT SOs")
print()

# Current CRR_GROUP_M (before this sweep)
CURRENT_GM = {
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

print(f"{'Shape':<30} {'SO':>6} | {'gm=4':>8} {'gm=8':>8} {'gm=16':>8} | {'best':>5} {'cur':>5} {'delta':>8}")
print("-" * 100)

new_gm = {}
total_gain = 0.0

for M, N, K in shapes:
    so_dir, so_tag = get_best_so(M, N, K)
    cur_gm = CURRENT_GM.get((M, N, K), 4)

    r4  = bench(M, N, K, so_dir, 4)
    r8  = bench(M, N, K, so_dir, 8)
    r16 = bench(M, N, K, so_dir, 16)

    best_val = max(r4, r8, r16)
    best_gm = [4, 8, 16][[r4, r8, r16].index(best_val)]
    cur_val = [r4, r8, r16][[4, 8, 16].index(cur_gm)]

    delta = best_val - cur_val
    total_gain += delta
    if best_gm != 4:
        new_gm[(M, N, K)] = best_gm

    flag = " ← CHANGE" if best_gm != cur_gm and abs(delta) > 5 else ""
    label = f"{M}x{N}x{K}"
    print(f"{label:<30} {so_tag:>6} | {r4:>8.1f} {r8:>8.1f} {r16:>8.1f} | {best_gm:>5} {cur_gm:>5} {delta:>+7.1f}{flag}")
    sys.stdout.flush()

print()
print(f"Total gain from gm re-tuning: {total_gain:+.1f} TFLOPS ({total_gain/len(shapes):+.2f} avg)")
print()

# Print updated CRR_GROUP_M
print("# Updated CRR_GROUP_M (only non-4 gm values):")
print("CRR_GROUP_M = {")
for M, N, K in shapes:
    gm = new_gm.get((M, N, K), 4)
    if gm != 4:
        print(f"    ({M:>5}, {N:>6}, {K:>5}): {gm:>2},")
print("}")
