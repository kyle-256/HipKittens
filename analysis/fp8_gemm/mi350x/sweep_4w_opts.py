import subprocess, json, os, sys, math

TK_ROOT = "/shared_nfs/kyle/HipKittens2"
DIR = f"{TK_ROOT}/analysis/fp8_gemm/mi350x"

configs = [
    {"name": "8w-baseline",   "flags": ""},
    {"name": "8w-occ1",       "flags": "-DGEMM_MIN_BLOCKS_PER_CU=1"},
    {"name": "8w-vm8",        "flags": "-DRCR_STEADY_VMCNT=8"},
    {"name": "8w-occ1-vm8",   "flags": "-DGEMM_MIN_BLOCKS_PER_CU=1 -DRCR_STEADY_VMCNT=8"},
    {"name": "8w-unroll4",    "flags": "-DRCR_MAIN_UNROLL=4"},
    {"name": "4w-dyn",        "flags": "-DRCR_USE_4WAVE_DYNAMIC=1"},
    {"name": "4w-dyn-vm8",    "flags": "-DRCR_USE_4WAVE_DYNAMIC=1 -DRCR_STEADY_VMCNT=8"},
]

shapes = [(8192,16384,16384),(16384,16384,16384),(8192,57344,8192),(4096,28672,4096),(8192,28672,4096),(16384,106496,16384),(8192,16384,53248)]
WARMUP, ITERS = 20, 40

bench_script = '''
import torch, tk_fp8_layouts
shapes = {shapes}
for M,N,K in shapes:
    A = (torch.randn(M,K,device='cuda')*0.1).to(torch.float8_e4m3fn)
    B = (torch.randn(N,K,device='cuda')*0.1).to(torch.float8_e4m3fn)
    C = torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
    fn = lambda: tk_fp8_layouts.gemm_rcr(A,B,C,1.0,1.0,4)
    for _ in range({w}): C.zero_(); fn()
    se = torch.cuda.Event(enable_timing=True); ee = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range({it}):
        C.zero_(); torch.cuda.synchronize(); se.record(); fn(); ee.record(); torch.cuda.synchronize()
        ts.append(se.elapsed_time(ee))
    avg = sum(ts)/len(ts)
    tf = 2.0*M*N*K/(avg*1e9)
    print(f"R:{{M}}_{{N}}_{{K}}:{{tf:.1f}}")
    del A,B,C; torch.cuda.empty_cache()
'''.format(shapes=shapes, w=WARMUP, it=ITERS)

all_results = {}
for cfg in configs:
    name = cfg["name"]
    flags = cfg["flags"]
    print(f"=== {name} ===", flush=True)
    
    env = {**os.environ, "THUNDERKITTENS_ROOT": TK_ROOT, "ROCM_PATH": "/opt/rocm"}
    if flags:
        env["HIPFLAGS"] = flags
    
    subprocess.run(["make", "clean"], capture_output=True, cwd=DIR, env=env)
    r = subprocess.run(["make", "-j4"], capture_output=True, text=True, cwd=DIR, env=env, timeout=120)
    if r.returncode != 0:
        print(f"  COMPILE FAIL")
        continue
    
    renv = {**os.environ, "HIP_VISIBLE_DEVICES": "4"}
    r = subprocess.run(["python3", "-c", bench_script], capture_output=True, text=True, cwd=DIR, env=renv, timeout=180)
    if r.returncode != 0:
        print(f"  RUN FAIL: {r.stderr[-100:]}")
        continue
    
    data = {}
    for line in r.stdout.strip().split('\n'):
        if line.startswith("R:"):
            parts = line[2:].split(":")
            data[parts[0]] = float(parts[1])
            print(f"  {parts[0]}: {parts[1]} TFLOPS")
    all_results[name] = data

print(f"\n{'Config':<18}", end="")
for s in shapes:
    label = f"{s[0]/1000:.0f}k,{s[1]/1000:.0f}k,{s[2]/1000:.0f}k"
    print(f" {label:>14}", end="")
print(" geo-mean")
print("-" * (18 + 15 * len(shapes) + 10))
for name, data in all_results.items():
    print(f"{name:<18}", end="")
    vals = []
    for s in shapes:
        k = f"{s[0]}_{s[1]}_{s[2]}"
        v = data.get(k, 0)
        vals.append(v)
        print(f" {v:>13.1f}", end="")
    geo = math.exp(sum(math.log(v) for v in vals if v > 0) / len([v for v in vals if v > 0])) if any(v > 0 for v in vals) else 0
    print(f" {geo:>9.1f}")
