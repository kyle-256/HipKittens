"""
BF16 GEMM: JIT K-specialized vs torch.mm (hipBLASLt) on LLM shapes.
Uses subprocess per shape to avoid .so conflicts.
"""
import subprocess, json, os, math, sys, time

DIR = os.path.dirname(os.path.abspath(__file__))
TK = os.environ.get("THUNDERKITTENS_ROOT", os.path.abspath(os.path.join(DIR, "..", "..", "..")))
GPU = os.environ.get("HIP_VISIBLE_DEVICES", "4")
EXT = subprocess.check_output(["python3-config", "--extension-suffix"], text=True).strip()

WARMUP, ITERS = 25, 40
BLK, KS = 256, 64

sys.path.insert(0, os.path.join(DIR, "../../fp8_gemm/mi350x"))
from bench_vs_hipblaslt import DenseModelConfigs, gen_gemm_test_cases

all_shapes = set()
for config in DenseModelConfigs.values():
    for mbs in [1, 2]:
        for _, seq, n, k in gen_gemm_test_cases(config):
            M = seq * mbs
            if M % BLK == 0 and n % BLK == 0 and k % KS == 0:
                all_shapes.add((M, n, k))
shapes = sorted(all_shapes)

k_values = sorted({k for _, _, k in shapes})

bench_jit = '''import torch,sys,os,math
sys.path.insert(0,'{sodir}')
import tk_bf16_layouts as m
M,N,K={M},{N},{K}
bM,bN={bM},{bN}
A=torch.randn({a_shape},dtype=torch.bfloat16,device='cuda')
B_{lay}=torch.randn({b_shape},dtype=torch.bfloat16,device='cuda')
C=torch.zeros(bM,bN,dtype=torch.bfloat16,device='cuda')
best_gm,best_tf=4,0
for gm in [1,2,4,8,16]:
    fn=lambda:m.gemm_{lay}(A,B_{lay},C,gm)
    for _ in range(8):fn()
    se=torch.cuda.Event(enable_timing=True);ee=torch.cuda.Event(enable_timing=True);ts=[]
    for _ in range(12):(torch.cuda.synchronize(),se.record(),fn(),ee.record(),torch.cuda.synchronize(),ts.append(se.elapsed_time(ee)))
    tf=2.0*M*N*K/(sum(ts)/len(ts)*1e9)
    if tf>best_tf:best_tf,best_gm=tf,gm
fn=lambda:m.gemm_{lay}(A,B_{lay},C,best_gm)
for _ in range({w}):fn()
se=torch.cuda.Event(enable_timing=True);ee=torch.cuda.Event(enable_timing=True);ts=[]
for _ in range({it}):(torch.cuda.synchronize(),se.record(),fn(),ee.record(),torch.cuda.synchronize(),ts.append(se.elapsed_time(ee)))
tf=2.0*M*N*K/(sum(ts)/len(ts)*1e9)
print(f"{{tf:.1f}} {{best_gm}}")
'''

bench_torch_rcr = '''import torch
M,N,K={M},{N},{K}
A=torch.randn(M,K,dtype=torch.bfloat16,device='cuda')
B=torch.randn(N,K,dtype=torch.bfloat16,device='cuda')
fn=lambda:torch.mm(A,B.T)
for _ in range({w}):fn()
se=torch.cuda.Event(enable_timing=True);ee=torch.cuda.Event(enable_timing=True);ts=[]
for _ in range({it}):(torch.cuda.synchronize(),se.record(),fn(),ee.record(),torch.cuda.synchronize(),ts.append(se.elapsed_time(ee)))
print(f"{{2.0*M*N*K/(sum(ts)/len(ts)*1e9):.1f}}")'''

bench_torch_rrr = '''import torch
M,N,K={M},{N},{K}
A=torch.randn(M,K,dtype=torch.bfloat16,device='cuda');B=torch.randn(K,N,dtype=torch.bfloat16,device='cuda')
fn=lambda:torch.mm(A,B)
for _ in range({w}):fn()
se=torch.cuda.Event(enable_timing=True);ee=torch.cuda.Event(enable_timing=True);ts=[]
for _ in range({it}):(torch.cuda.synchronize(),se.record(),fn(),ee.record(),torch.cuda.synchronize(),ts.append(se.elapsed_time(ee)))
print(f"{{2.0*M*N*K/(sum(ts)/len(ts)*1e9):.1f}}")'''

bench_torch_crr = '''import torch
M,N,K={M},{N},{K}
A=torch.randn(K,M,dtype=torch.bfloat16,device='cuda');B=torch.randn(K,N,dtype=torch.bfloat16,device='cuda')
fn=lambda:torch.mm(A.T,B)
for _ in range({w}):fn()
se=torch.cuda.Event(enable_timing=True);ee=torch.cuda.Event(enable_timing=True);ts=[]
for _ in range({it}):(torch.cuda.synchronize(),se.record(),fn(),ee.record(),torch.cuda.synchronize(),ts.append(se.elapsed_time(ee)))
print(f"{{2.0*M*N*K/(sum(ts)/len(ts)*1e9):.1f}}")'''


def run_script(script, cwd=DIR):
    env = {**os.environ, "HIP_VISIBLE_DEVICES": GPU}
    try:
        r = subprocess.run(["python3", "-c", script], capture_output=True, text=True,
                           env=env, cwd=cwd, timeout=180)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return None

TORCH_TPL = {"rcr": bench_torch_rcr, "rrr": bench_torch_rrr, "crr": bench_torch_crr}

# Phase 1: compile
print(f"[Phase 1] Pre-compiling {len(shapes)} exact-dim kernels...")
sys.path.insert(0, DIR)
from jit_bf16_gemm import warmup_shapes, _exact_cache_dir, _MODULE_NAME, _can_jit
warmup_shapes(shapes, verbose=True)

# Also compile swap(M↔N) shapes for N > M cases
swap_shapes = [(N, M, K) for M, N, K in shapes if N > M and _can_jit(N, M, K)]
if swap_shapes:
    print(f"[Phase 1] Pre-compiling {len(swap_shapes)} swap(N↔M) kernels...")
    warmup_shapes(swap_shapes, verbose=True)

# Phase 2: benchmark
layouts = ["rcr", "rrr", "crr"]
print(f"\n[Phase 2] Benchmarking {len(shapes)} shapes x {len(layouts)} layouts on GPU{GPU}")
print(f"{'M':>5} {'N':>6} {'K':>5}", end="")
for lay in layouts:
    print(f" | {'TK':>7} {'gm':>2} {'torch':>7} {'ratio':>6}", end="")
print()
print("-" * 110)

results = {lay: [] for lay in layouts}
all_data = []

for M, N, K in shapes:
    row_data = {"M": M, "N": N, "K": K}
    line = f"{M:>5} {N:>6} {K:>5}"

    for lay in layouts:
        # Swap M↔N when N > M: compute C^T = B^T@A^T instead of C = A@B^T
        # This avoids tall-N grid which is inefficient for our 256×256 tile.
        use_swap = N > M and _can_jit(N, M, K)
        bM, bN, bK = (N, M, K) if use_swap else (M, N, K)
        sodir = _exact_cache_dir(bM, bN, bK)

        if lay == "rcr":
            a_shape = f"{bM},{bK}"  # A=(bM,K)
            b_shape = f"{bN},{bK}"  # B=(bN,K) [transposed at MMA]
        elif lay == "rrr":
            a_shape = f"{bM},{bK}"  # A=(bM,K)
            b_shape = f"{bK},{bN}"  # B=(K,bN)
        else:  # crr
            a_shape = f"{bK},{bM}"  # A=(K,bM) stored transposed
            b_shape = f"{bK},{bN}"  # B=(K,bN)

        script = bench_jit.format(sodir=sodir, M=M, N=N, K=K, bM=bM, bN=bN, lay=lay,
                                  a_shape=a_shape, b_shape=b_shape, w=WARMUP, it=ITERS)
        out = run_script(script)
        if out:
            parts = out.split()
            tk_tf = float(parts[0])
            gm = int(parts[1])
        else:
            tk_tf, gm = 0, 4

        torch_tpl = TORCH_TPL[lay]
        tout = run_script(torch_tpl.format(M=M, N=N, K=K, w=WARMUP, it=ITERS))
        t_tf = float(tout) if tout else 0

        ratio = tk_tf / t_tf if tk_tf > 0 and t_tf > 0 else 0
        w = "*" if ratio >= 1.0 else " "
        results[lay].append(ratio)
        row_data[f"{lay}_tk"] = tk_tf
        row_data[f"{lay}_torch"] = t_tf
        row_data[f"{lay}_ratio"] = ratio
        row_data[f"{lay}_gm"] = gm
        line += f" | {tk_tf:>7.1f} {gm:>2} {t_tf:>7.1f} {ratio:>.3f}x{w}"

    all_data.append(row_data)
    print(line)

print("=" * 110)
for lay in layouts:
    vals = [v for v in results[lay] if v > 0]
    if vals:
        geo = math.exp(sum(math.log(v) for v in vals) / len(vals))
        wins = sum(1 for v in vals if v >= 1.0)
        print(f"{lay.upper()}: geo-mean={geo:.4f}x  wins={wins}/{len(vals)}")

with open(os.path.join(DIR, "bench_bf16_jit_results.json"), "w") as f:
    json.dump(all_data, f, indent=2)
print(f"\nResults saved to bench_bf16_jit_results.json")
