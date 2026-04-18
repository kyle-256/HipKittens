#!/usr/bin/env python3
"""DLA7-only serial bench on GPU 7 — to control for noise seen in parallel run."""
import json, math, os, subprocess, sys, sysconfig, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP, ITERS, TRIM, N_REPS = 200, 500, 0.10, 3
M, N, K = 28672, 32768, 4096
GPU = 7
LAB = "DLA7"

VARIANT_SUFFIXES = ["_r25d_baseline", "_r25d_gm6", "_r25d_pfoff4", "_r25d_gm6_pfoff4"]


def run_one(suffix):
    full_suffix = f"{suffix}_{LAB.lower()}"
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    script = f"""
import sys, math, torch, importlib.util, json
M, N, K = {M}, {N}, {K}
torch.manual_seed(0)
def gen_fp4(r, K):
    c = K // 2
    return (torch.randint(0, 16, (r, c), dtype=torch.uint8, device='cuda') << 4) | torch.randint(0, 16, (r, c), dtype=torch.uint8, device='cuda')
def preshuffle(se):
    r, kb = se.shape; pr = math.ceil(r/64)*64; pk = math.ceil(kb/8)*8
    raw = torch.full((pr, pk), 0x7F, dtype=torch.uint8, device=se.device)
    raw[:r,:kb] = (se.to(torch.int16)+127).to(torch.uint8)
    sh = raw.view(pr//32,2,16,pk//8,2,4,1).permute(0,3,5,2,4,1,6).contiguous().view(pr//32,pk*32)
    sh = sh.view(pr//64,2,pk*32//4,4).permute(0,2,1,3).contiguous()
    return sh.view(pr//64,pk*64)
spec = importlib.util.spec_from_file_location('{module_name}', '{so_path}')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
A=gen_fp4(M,K); B=gen_fp4(N,K)
sc_a=torch.randint(-2,3,(M,K//32),dtype=torch.int8,device='cuda')
sc_b=torch.randint(-2,3,(N,K//32),dtype=torch.int8,device='cuda')
A_sc=preshuffle(sc_a); B_sc=preshuffle(sc_b)
C=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
def run(): mod.gemm_rcr(A,B,A_sc,B_sc,C)
for _ in range({WARMUP}): run()
torch.cuda.synchronize()
nz=(C!=0).float().mean().item()
if nz < 0.5:
    print(json.dumps({{"error": "low coverage", "nz": nz}})); sys.exit(0)
times=[]
for _ in range({ITERS}):
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record(); run(); e.record(); torch.cuda.synchronize()
    times.append(s.elapsed_time(e))
times.sort(); trim=int(len(times)*{TRIM}); times=times[trim:-trim] if trim>0 else times
avg=sum(times)/len(times); t=2.0*M*N*K/(avg*1e-3)/1e12
print(json.dumps({{"tflops": round(t,2), "ms": round(avg,4)}}))
"""
    if not os.path.exists(so_path):
        return {"error": "missing .so"}
    env = os.environ.copy(); env["HIP_VISIBLE_DEVICES"] = str(GPU)
    try:
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=900, env=env)
        if r.returncode != 0:
            return {"error": f"rc={r.returncode}", "stderr": r.stderr[-300:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": str(e)}


def main():
    print(f"DLA7 serial bench, GPU {GPU}, warmup={WARMUP} iters={ITERS} trim={TRIM} reps={N_REPS}")
    out = {}
    for vs in VARIANT_SUFFIXES:
        runs = []
        for rep in range(N_REPS):
            r = run_one(vs)
            tag = f"{vs} rep{rep}"
            if "error" in r:
                print(f"  {tag:35s}  ERR {r.get('error')}", flush=True)
            else:
                print(f"  {tag:35s}  {r['tflops']:>8.2f} TFLOPS (ms={r.get('ms')})", flush=True)
            runs.append(r)
        out[vs] = runs
    with open(os.path.join(SCRIPT_DIR, "bench_round25_optD_dla7_serial.json"), "w") as f:
        json.dump(out, f, indent=2)
    # summary
    print("\nSUMMARY (median, Δ vs baseline):")
    def med(vs):
        xs=[r.get("tflops") for r in out.get(vs,[]) if r.get("tflops") is not None]
        if not xs: return None
        xs.sort(); return xs[len(xs)//2]
    bm = med("_r25d_baseline")
    if bm is None:
        print("baseline failed"); return
    for vs in VARIANT_SUFFIXES:
        m = med(vs)
        if m is None: print(f"  {vs:30s} ERR"); continue
        d = (m-bm)/bm*100 if vs!="_r25d_baseline" else 0
        print(f"  {vs:30s}  med={m:.2f}  Δ={d:+.2f}%")


if __name__ == "__main__":
    main()
