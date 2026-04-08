"""Direct A/B comparison: JIT 8-wave vs Dynamic, each shape in own subprocess."""
import subprocess, json, os, sys, importlib.machinery

DIR = os.path.dirname(os.path.abspath(__file__))
GPU = os.environ.get("HIP_VISIBLE_DEVICES", "4")
EXT = importlib.machinery.EXTENSION_SUFFIXES[0]

shapes = [(4096,28672,4096),(8192,16384,16384),(8192,57344,8192),(16384,106496,16384),(8192,16384,53248)]
layouts = ["rcr","rrr","crr"]
MAKE_AB = {
    "rcr": "A=(torch.randn(M,K,device='cuda')*0.1).to(torch.float8_e4m3fn);B=(torch.randn(N,K,device='cuda')*0.1).to(torch.float8_e4m3fn)",
    "rrr": "A=(torch.randn(M,K,device='cuda')*0.1).to(torch.float8_e4m3fn);B=(torch.randn(K,N,device='cuda')*0.1).to(torch.float8_e4m3fn)",
    "crr": "A=(torch.randn(K,M,device='cuda')*0.1).to(torch.float8_e4m3fn);B=(torch.randn(K,N,device='cuda')*0.1).to(torch.float8_e4m3fn)",
}
bench = 'import torch,tk_fp8_layouts as m;M,N,K={M},{N},{K};{ab};C=torch.zeros(M,N,dtype=torch.bfloat16,device="cuda");fn=lambda:m.gemm_{lay}(A,B,C,1.0,1.0,4);[fn()for _ in range(40)];se=torch.cuda.Event(enable_timing=True);ee=torch.cuda.Event(enable_timing=True);ts=[];[(torch.cuda.synchronize(),se.record(),fn(),ee.record(),torch.cuda.synchronize(),ts.append(se.elapsed_time(ee)))for _ in range(80)];print(f"{{2.0*M*N*K/(sum(ts)/len(ts)*1e9):.1f}}")'

def run_bench(so_dir, M, N, K, lay):
    env = {**os.environ, "HIP_VISIBLE_DEVICES": GPU, "PYTHONPATH": so_dir}
    s = bench.format(M=M,N=N,K=K,lay=lay,ab=MAKE_AB[lay])
    r = subprocess.run(["python3","-c",s], capture_output=True, text=True, cwd=so_dir, env=env, timeout=120)
    return float(r.stdout.strip()) if r.returncode == 0 else 0

print(f"{'Shape':>25} {'Layout':>5} | {'JIT-8w':>8} {'Dynamic':>8} {'Diff':>7}")
print("-"*60)
for M,N,K in shapes:
    for lay in layouts:
        jit_dir = os.path.join(DIR, ".jit_cache", f"{lay}_{M}x{N}x{K}_8wave")
        dyn_dir = DIR
        jit_tf = run_bench(jit_dir, M, N, K, lay)
        dyn_tf = run_bench(dyn_dir, M, N, K, lay)
        diff = (jit_tf/dyn_tf - 1)*100 if dyn_tf > 0 else 0
        marker = "+" if diff >= 0 else ""
        print(f"({M:>5},{N:>6},{K:>5}) {lay.upper():>5} | {jit_tf:>7.1f}  {dyn_tf:>7.1f}  {marker}{diff:>5.1f}%")
