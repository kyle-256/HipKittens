import subprocess, json, os, sys, time

configs = [
    {"name": "baseline",  "flags": ""},
    {"name": "vm2",       "flags": "-DRCR_STEADY_VMCNT=2"},
    {"name": "vm6",       "flags": "-DRCR_STEADY_VMCNT=6"},
    {"name": "vm8",       "flags": "-DRCR_STEADY_VMCNT=8"},
    {"name": "lgkm2",     "flags": "-DRCR_PREFETCH_LGKM=2"},
    {"name": "lgkm6",     "flags": "-DRCR_PREFETCH_LGKM=6"},
    {"name": "lgkm8",     "flags": "-DRCR_PREFETCH_LGKM=8"},
    {"name": "vm2_lgkm2", "flags": "-DRCR_STEADY_VMCNT=2 -DRCR_PREFETCH_LGKM=2"},
    {"name": "vm6_lgkm6", "flags": "-DRCR_STEADY_VMCNT=6 -DRCR_PREFETCH_LGKM=6"},
]

TK_ROOT = "/shared_nfs/kyle/HipKittens2"
DIR = f"{TK_ROOT}/analysis/fp8_gemm/mi350x"
shapes = [(4096,28672,4096),(8192,28672,4096),(8192,16384,16384),(16384,106496,16384),(8192,16384,53248)]

test_script = '''
import torch, tk_fp8_layouts, time
shapes = {shapes}
warmup, iters = 20, 40
results = {{}}
for M,N,K in shapes:
    A = (torch.randn(M,K,device='cuda')*0.1).to(torch.float8_e4m3fn)
    B = (torch.randn(N,K,device='cuda')*0.1).to(torch.float8_e4m3fn)
    C = torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
    fn = lambda: tk_fp8_layouts.gemm_rcr(A,B,C,1.0,1.0,4)
    for _ in range(warmup): C.zero_(); fn()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        C.zero_(); torch.cuda.synchronize(); start.record(); fn(); end.record(); torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    avg = sum(times)/len(times)
    flops = 2.0*M*N*K
    tflops = flops/(avg*1e9)
    results[f"{{M}}_{{N}}_{{K}}"] = round(tflops, 1)
    del A,B,C; torch.cuda.empty_cache()
import json
print("RESULTS:" + json.dumps(results))
'''.format(shapes=shapes)

all_results = {}
for cfg in configs:
    name = cfg["name"]
    flags = cfg["flags"]
    print(f"=== {name}: {flags or '(default)'} ===", flush=True)
    
    env = os.environ.copy()
    env["THUNDERKITTENS_ROOT"] = TK_ROOT
    env["ROCM_PATH"] = "/opt/rocm"
    if flags:
        env["HIPFLAGS"] = flags
    
    r = subprocess.run(["make", "clean"], capture_output=True, cwd=DIR, env=env)
    r = subprocess.run(["make", "-j4"], capture_output=True, text=True, cwd=DIR, env=env)
    if r.returncode != 0:
        print(f"  COMPILE FAILED: {r.stderr[-200:]}")
        continue
    
    env2 = os.environ.copy()
    env2["HIP_VISIBLE_DEVICES"] = "4"
    r = subprocess.run(["python3", "-c", test_script], capture_output=True, text=True, cwd=DIR, env=env2, timeout=120)
    if r.returncode != 0:
        print(f"  RUN FAILED: {r.stderr[-200:]}")
        continue
    
    for line in r.stdout.split('\n'):
        if line.startswith("RESULTS:"):
            data = json.loads(line[8:])
            all_results[name] = data
            for k, v in sorted(data.items()):
                print(f"  {k}: {v} TFLOPS")
            break
    print()

print("\n=== COMPARISON TABLE ===")
header = f"{'Config':<15}"
for s in shapes:
    header += f" ({s[0]},{s[1]},{s[2]})"[-18:].rjust(18)
print(header)
print("-" * len(header))
for name, data in all_results.items():
    row = f"{name:<15}"
    for s in shapes:
        key = f"{s[0]}_{s[1]}_{s[2]}"
        row += f" {data.get(key, 0):>17.1f}"
    print(row)
