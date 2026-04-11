"""
Targeted CRR group_m sweep using full-bench parameters (30w/50i) for worst CRR/RCR shapes.
Only tests shapes not already in CRR_GROUP_M or where we want to verify the current setting.
Threshold: only flag changes with full-bench delta > 20 TFLOPS (to filter noise).
"""
import subprocess, os, sys, json

DIR = os.path.dirname(os.path.abspath(__file__))
GPU = os.environ.get("HIP_VISIBLE_DEVICES", "4")
EXT = subprocess.check_output(["python3-config", "--extension-suffix"], text=True).strip()
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
    opt = os.path.join(CACHE, f"crr_{M}x{N}x{K}_8wave_opt")
    if os.path.exists(os.path.join(opt, f"tk_fp8_layouts{EXT}")):
        return opt, "opt"
    jit = os.path.join(CACHE, f"crr_{M}x{N}x{K}_8wave")
    if os.path.exists(os.path.join(jit, f"tk_fp8_layouts{EXT}")):
        return jit, "jit"
    return os.path.join(CACHE, "crr_shared"), "shared"


# Current CRR_GROUP_M from bench_jit_full.py
CURRENT_CRR_GM = {
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

# Shapes to test: worst CRR/RCR ratio shapes + shapes with large N not yet in CRR_GROUP_M
# Focus on shapes where gm change might give significant gain
test_shapes = [
    # Worst ratio shapes (k_iters=128, K=16384)
    ( 8192, 106496, 16384),
    (16384, 106496, 16384),
    ( 8192,  16384, 16384),
    (16384,  16384, 16384),
    (16384,  18432, 16384),
    ( 8192,  18432, 16384),
    # k_iters=64 (K=8192), moderate gap
    ( 8192,  57344,  8192),
    (16384,  59136,  8192),
    ( 8192,  59136,  8192),
    ( 8192,  10240,  8192),  # in CRR_GROUP_M with gm=8, verify
    # k_iters=416 (K=53248), worst gap
    (16384,  16384, 53248),
    ( 8192,  16384, 53248),
    # Small shapes
    ( 4096,   4096,  4096),  # currently gm=4, verify
    ( 4096,  57344,  8192),
]

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

test_shapes = [(M, N, K) for M, N, K in test_shapes if (M, N, K) in all_shapes]
print(f"Testing {len(test_shapes)} shapes with gm=4,8,16 (full bench params 30w/50i)")
print()

# Load existing CRR results for reference
results_file = os.path.join(DIR, "bench_jit_full_results.json")
ref_crr = {}
ref_rcr = {}
if os.path.exists(results_file):
    with open(results_file) as f:
        d = json.load(f)
    ref_crr = d.get("crr", {})
    ref_rcr = d.get("rcr", {})

THRESHOLD = 15  # TFLOPS: only flag changes > 15 TFLOPS

print(f"{'Shape':<30} {'SO':>6} {'gm4':>8} {'gm8':>8} {'gm16':>9} | {'best':>5} {'cur':>5} {'delta':>8} {'ref_crr':>8} {'ref_rcr':>8}")
print("-" * 110)

updates = {}
for M, N, K in test_shapes:
    so_dir, so_tag = get_best_so(M, N, K)
    cur_gm = CURRENT_CRR_GM.get((M, N, K), 4)

    r4  = bench(M, N, K, so_dir, 4)
    r8  = bench(M, N, K, so_dir, 8)
    r16 = bench(M, N, K, so_dir, 16)

    best_val = max(r4, r8, r16)
    best_gm = [4, 8, 16][[r4, r8, r16].index(best_val)]
    cur_val = {4: r4, 8: r8, 16: r16}[cur_gm]
    delta = best_val - cur_val

    key = f"{M}_{N}_{K}"
    ref_c = ref_crr.get(key, 0)
    ref_r = ref_rcr.get(key, 0)

    if best_gm != cur_gm and delta > THRESHOLD:
        updates[(M, N, K)] = best_gm
        flag = " ← UPDATE"
    elif delta < -THRESHOLD:
        flag = " ← REGRESS?"
    else:
        flag = ""

    label = f"{M}x{N}x{K}"
    print(f"{label:<30} {so_tag:>6} {r4:>8.1f} {r8:>8.1f} {r16:>9.1f} | {best_gm:>5} {cur_gm:>5} {delta:>+7.1f} {ref_c:>8.1f} {ref_r:>8.1f}{flag}")
    sys.stdout.flush()

print()
print(f"Threshold for update: >{THRESHOLD} TFLOPS improvement in full bench")
print(f"Proposed updates ({len(updates)} shapes):")
for (M, N, K), gm in sorted(updates.items()):
    cur = CURRENT_CRR_GM.get((M, N, K), 4)
    print(f"  ({M:>5}, {N:>6}, {K:>5}): {cur} → {gm}")
