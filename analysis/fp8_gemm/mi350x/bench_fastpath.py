import subprocess, json, os, sys

TK_ROOT = "/shared_nfs/kyle/HipKittens2"
DIR = f"{TK_ROOT}/analysis/fp8_gemm/mi350x"

# Shapes to test — we compile a fastpath for each
test_shapes = [
    (8192, 8192, 8192),
    (8192, 28672, 4096),
    (4096, 28672, 4096),
    (8192, 16384, 16384),
    (16384, 16384, 16384),
    (8192, 57344, 8192),
    (4096, 57344, 8192),
    (16384, 106496, 16384),
    (8192, 16384, 53248),
]

configs = [
    {"name": "dynamic",   "extra": ""},
    {"name": "8wave",     "extra": "-DRCR_USE_EXACT_8WAVE_FASTPATH=1"},
    {"name": "4wave",     "extra": "-DRCR_USE_EXACT_4WAVE_FASTPATH=1 -DRCR_USE_EXACT_8WAVE_FASTPATH=0"},
]

WARMUP = 20
ITERS = 40

bench_template = '''
import torch, tk_fp8_layouts, math
M, N, K = {M}, {N}, {K}
A = (torch.randn(M,K,device='cuda')*0.1).to(torch.float8_e4m3fn)
B = (torch.randn(N,K,device='cuda')*0.1).to(torch.float8_e4m3fn)
C = torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
ref = A.float()@B.float().T
tk_fp8_layouts.gemm_rcr(A,B,C,1.0,1.0,4)
snr = 10*math.log10((ref**2).sum().item()/((C.float()-ref)**2).sum().item())
ok = "PASS" if snr > 48 else "FAIL"
fn = lambda: tk_fp8_layouts.gemm_rcr(A,B,C,1.0,1.0,4)
for _ in range({warmup}): C.zero_(); fn()
start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
times = []
for _ in range({iters}):
    C.zero_(); torch.cuda.synchronize(); start.record(); fn(); end.record(); torch.cuda.synchronize()
    times.append(start.elapsed_time(end))
avg = sum(times)/len(times)
flops = 2.0*M*N*K
tflops = flops/(avg*1e9)
print(f"RESULT:{{tflops:.1f}}:{{snr:.2f}}:{{ok}}")
'''

results = {}
for cfg in configs:
    cname = cfg["name"]
    results[cname] = {}
    
    for M, N, K in test_shapes:
        shape_key = f"{M}x{N}x{K}"
        flags = f"-DM_DIM={M} -DN_DIM={N} -DK_DIM={K} {cfg['extra']}"
        
        env = os.environ.copy()
        env["THUNDERKITTENS_ROOT"] = TK_ROOT
        env["ROCM_PATH"] = "/opt/rocm"
        env["HIPFLAGS"] = flags
        
        r = subprocess.run(["make", "clean"], capture_output=True, cwd=DIR, env=env)
        r = subprocess.run(["make", "-j4"], capture_output=True, text=True, cwd=DIR, env=env, timeout=120)
        if r.returncode != 0:
            print(f"[{cname}] {shape_key}: COMPILE FAIL", flush=True)
            results[cname][shape_key] = {"tflops": 0, "snr": 0, "ok": "COMPILE_FAIL"}
            continue
        
        script = bench_template.format(M=M, N=N, K=K, warmup=WARMUP, iters=ITERS)
        env2 = os.environ.copy()
        env2["HIP_VISIBLE_DEVICES"] = "4"
        run_env = {**os.environ, "HIP_VISIBLE_DEVICES": "4"}
        r = subprocess.run(["python3", "-c", script], capture_output=True, text=True, cwd=DIR, env=run_env, timeout=120)
        
        if r.returncode != 0:
            print(f"[{cname}] {shape_key}: RUN FAIL: {r.stderr[-150:]}", flush=True)
            results[cname][shape_key] = {"tflops": 0, "snr": 0, "ok": "RUN_FAIL"}
            continue
        
        for line in r.stdout.strip().split('\n'):
            if line.startswith("RESULT:"):
                parts = line[7:].split(":")
                tflops = float(parts[0])
                snr = float(parts[1])
                ok = parts[2]
                results[cname][shape_key] = {"tflops": tflops, "snr": snr, "ok": ok}
                print(f"[{cname:>8}] {shape_key:>25}: {tflops:>7.1f} TFLOPS  SNR={snr:.1f}dB  {ok}", flush=True)
                break

# Summary table
print(f"\n{'Shape':>25}", end="")
for cfg in configs:
    print(f" {cfg['name']:>12}", end="")
print()
print("-" * (25 + 13 * len(configs)))
for M, N, K in test_shapes:
    sk = f"{M}x{N}x{K}"
    print(f"{sk:>25}", end="")
    for cfg in configs:
        r = results.get(cfg["name"], {}).get(sk, {})
        t = r.get("tflops", 0)
        ok = r.get("ok", "?")
        marker = " *" if ok == "FAIL" else ""
        print(f" {t:>10.1f}{marker}", end="")
    print()
