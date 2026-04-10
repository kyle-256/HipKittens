"""BF16 best-of benchmark: shape-specialized(u2) vs dynamic vs torch."""
import subprocess, os, sys, math, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../fp8_gemm/mi350x"))
from bench_vs_hipblaslt import DenseModelConfigs, gen_gemm_test_cases

GPU = os.environ.get("HIP_VISIBLE_DEVICES", "4")
WARMUP, ITERS = 25, 40
BLK, KSTEP = 256, 64
EXT = subprocess.check_output(["python3-config", "--extension-suffix"], text=True).strip()

bench_spec = '''import torch,tk_kernel as m
M,N,K={M},{N},{K}
A=torch.randn(M,K,dtype=torch.bfloat16,device='cuda');B=torch.randn(N,K,dtype=torch.bfloat16,device='cuda')
C=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
fn=lambda:m.dispatch_micro(A,B,C)
for _ in range({w}):fn()
se=torch.cuda.Event(enable_timing=True);ee=torch.cuda.Event(enable_timing=True);ts=[]
for _ in range({it}):(torch.cuda.synchronize(),se.record(),fn(),ee.record(),torch.cuda.synchronize(),ts.append(se.elapsed_time(ee)))
print(f"{{2.0*M*N*K/(sum(ts)/len(ts)*1e9):.1f}}")'''

bench_dyn = '''import torch,tk_bf16_layouts as m
M,N,K={M},{N},{K}
A=torch.randn(M,K,dtype=torch.bfloat16,device='cuda');B=torch.randn(N,K,dtype=torch.bfloat16,device='cuda')
C=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
fn=lambda:m.gemm_rcr(A,B,C,{gm})
for _ in range({w}):fn()
se=torch.cuda.Event(enable_timing=True);ee=torch.cuda.Event(enable_timing=True);ts=[]
for _ in range({it}):(torch.cuda.synchronize(),se.record(),fn(),ee.record(),torch.cuda.synchronize(),ts.append(se.elapsed_time(ee)))
print(f"{{2.0*M*N*K/(sum(ts)/len(ts)*1e9):.1f}}")'''

bench_torch = '''import torch
M,N,K={M},{N},{K}
A=torch.randn(M,K,dtype=torch.bfloat16,device='cuda');B=torch.randn(N,K,dtype=torch.bfloat16,device='cuda')
fn=lambda:torch.mm(A,B.T)
for _ in range({w}):fn()
se=torch.cuda.Event(enable_timing=True);ee=torch.cuda.Event(enable_timing=True);ts=[]
for _ in range({it}):(torch.cuda.synchronize(),se.record(),fn(),ee.record(),torch.cuda.synchronize(),ts.append(se.elapsed_time(ee)))
print(f"{{2.0*M*N*K/(sum(ts)/len(ts)*1e9):.1f}}")'''

def run(script, pypath="."):
    env = {**os.environ, "HIP_VISIBLE_DEVICES": GPU, "PYTHONPATH": pypath}
    r = subprocess.run(["python3", "-c", script], capture_output=True, text=True, env=env, timeout=120)
    return float(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else 0

gm_cache = {}
try:
    with open(os.path.join(os.path.dirname(__file__), ".bf16_gm_cache.json")) as f:
        gm_cache = json.load(f)
except: pass

print(f"{'Model':>16} {'Op':>12} {'MBS':>3} {'M':>5} {'N':>6} {'K':>5} {'spec':>7} {'dyn':>7} {'torch':>7} {'best/t':>7} {'src':>4}")
print("-" * 95)

stats_dyn, stats_best = [], []
for model in DenseModelConfigs:
    config = DenseModelConfigs[model]
    for mbs in [1, 2]:
        for op, seq, n, k in gen_gemm_test_cases(config):
            M = seq * mbs
            if M % BLK != 0 or n % BLK != 0 or k % KSTEP != 0: continue

            key = f"{M}_{n}_{k}"
            spec_dir = f"/tmp/bf16u2_{M}_{n}_{k}"

            # Dynamic with autotune gm
            if key not in gm_cache:
                best_gm, best_tf = 4, 0
                for gm in [1, 2, 4, 8, 16]:
                    tf = run(bench_dyn.format(M=M, N=n, K=k, gm=gm, w=10, it=12))
                    if tf > best_tf: best_tf, best_gm = tf, gm
                gm_cache[key] = best_gm
            gm = gm_cache[key]

            sp = run(bench_spec.format(M=M, N=n, K=k, w=WARMUP, it=ITERS), spec_dir)
            dy = run(bench_dyn.format(M=M, N=n, K=k, gm=gm, w=WARMUP, it=ITERS))
            tt = run(bench_torch.format(M=M, N=n, K=k, w=WARMUP, it=ITERS))

            best = max(sp, dy)
            src = "S" if sp > dy else "D"
            r_dyn = dy / tt if tt > 0 else 0
            r_best = best / tt if tt > 0 else 0
            stats_dyn.append(r_dyn)
            stats_best.append(r_best)
            w = "*" if r_best >= 1.0 else " "
            print(f"{model:>16} {op:>12} {mbs:>3} {M:>5} {n:>6} {k:>5} {sp:>7.1f} {dy:>7.1f} {tt:>7.1f} {r_best:>6.3f}x{w} [{src}]")

print("=" * 95)
if stats_dyn:
    gd = math.exp(sum(math.log(v) for v in stats_dyn) / len(stats_dyn))
    wd = sum(1 for v in stats_dyn if v >= 1.0)
    gb = math.exp(sum(math.log(v) for v in stats_best) / len(stats_best))
    wb = sum(1 for v in stats_best if v >= 1.0)
    print(f"Dynamic only:  {gd:.4f}x ({wd}/{len(stats_dyn)} wins)")
    print(f"Best-of:       {gb:.4f}x ({wb}/{len(stats_best)} wins)")

with open(os.path.join(os.path.dirname(__file__), ".bf16_gm_cache.json"), "w") as f:
    json.dump(gm_cache, f)
