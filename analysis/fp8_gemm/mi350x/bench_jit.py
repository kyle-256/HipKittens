"""Benchmark JIT 4-wave vs hipBLASLt, each shape in a subprocess to avoid module conflicts."""
import subprocess, json, os, math, time

DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", os.path.abspath(os.path.join(DIR, "..", "..", "..")))
ROCM = os.environ.get("ROCM_PATH", "/opt/rocm")
GPU = os.environ.get("HIP_VISIBLE_DEVICES", "4")

shapes = [
    (4096,28672,4096),(8192,28672,4096),(8192,16384,16384),(16384,16384,16384),
    (8192,57344,8192),(4096,57344,8192),(16384,106496,16384),(8192,16384,53248),
    (4096,4096,4096),(8192,8192,8192),
]

with open(os.path.join(DIR, "bench_vs_hipblaslt_clean_gpu4.json")) as f:
    bl_raw = json.load(f)["results"]
bl_data = {}
for r in bl_raw:
    if r["layout"] == "rcr":
        bl_data[f'{r["M"]}_{r["N"]}_{r["K"]}'] = r["bl_tflops"]

bench_py = '''
import torch, tk_fp8_layouts
M,N,K = {M},{N},{K}
A = (torch.randn(M,K,device='cuda')*0.1).to(torch.float8_e4m3fn)
B = (torch.randn(N,K,device='cuda')*0.1).to(torch.float8_e4m3fn)
C = torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
fn = lambda: tk_fp8_layouts.gemm_rcr(A,B,C,1.0,1.0,4)
for _ in range(50): C.zero_(); fn()
se = torch.cuda.Event(enable_timing=True); ee = torch.cuda.Event(enable_timing=True)
ts = []
for _ in range(100):
    C.zero_(); torch.cuda.synchronize(); se.record(); fn(); ee.record(); torch.cuda.synchronize()
    ts.append(se.elapsed_time(ee))
print(f"{{2.0*M*N*K/(sum(ts)/len(ts)*1e9):.1f}}")
'''

print(f"{'Shape':>30} {'JIT-4w':>8} {'hipBLASLt':>10} {'Ratio':>7}")
print("-" * 60)
ratios = []

for M, N, K in shapes:
    env = {**os.environ, "THUNDERKITTENS_ROOT": TK_ROOT, "ROCM_PATH": ROCM,
           "HIPFLAGS": f"-DM_DIM={M} -DN_DIM={N} -DK_DIM={K} -DRCR_USE_EXACT_4WAVE_FASTPATH=1 -DRCR_USE_EXACT_8WAVE_FASTPATH=0 -DRCR_STEADY_VMCNT=8"}
    subprocess.run(["make", "clean"], capture_output=True, cwd=DIR, env=env)
    r = subprocess.run(["make", "-j4"], capture_output=True, text=True, cwd=DIR, env=env, timeout=120)
    if r.returncode != 0:
        print(f"({M:>5},{N:>6},{K:>5}) COMPILE FAIL")
        continue

    run_env = {**os.environ, "HIP_VISIBLE_DEVICES": GPU}
    r = subprocess.run(["python3", "-c", bench_py.format(M=M, N=N, K=K)],
                       capture_output=True, text=True, cwd=DIR, env=run_env, timeout=60)
    if r.returncode != 0:
        print(f"({M:>5},{N:>6},{K:>5}) RUN FAIL: {r.stderr[-100:]}")
        continue

    tf = float(r.stdout.strip())
    bl = bl_data.get(f"{M}_{N}_{K}", 0)
    ratio = tf / bl if bl > 0 else 0
    ratios.append(ratio)
    w = "**" if ratio >= 1.0 else "  "
    print(f"({M:>5},{N:>6},{K:>5}) {tf:>7.1f} {bl:>9.1f} {ratio:>6.3f}x {w}")

if ratios:
    geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    wins = sum(1 for r in ratios if r >= 1.0)
    print(f"\nGeo-mean: {geo:.4f}x  Wins: {wins}/{len(ratios)}")
