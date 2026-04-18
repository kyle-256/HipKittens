#!/usr/bin/env python3
"""R25-G VERIFY3: SD only, GPU 0, isolated, 5 reps to characterize stability."""
import json, os, subprocess, sys, sysconfig, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP, ITERS, TRIM, N_REPS = 200, 500, 0.10, 5
M, N, K = 16384, 4096, 28672
GPU = 1


def make_bench_script(suffix):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    return so_path, f"""
import sys, math, torch, importlib.util, json
WARMUP, ITERS, TRIM = {WARMUP}, {ITERS}, {TRIM}
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
for _ in range(WARMUP): run()
torch.cuda.synchronize()
run(); torch.cuda.synchronize()
nz = (C != 0).float().mean().item()
if nz < 0.5:
    print(json.dumps({{"error": "C-coverage too low", "nz": nz}})); sys.exit(0)
times=[]
for _ in range(ITERS):
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record(); run(); e.record(); torch.cuda.synchronize()
    times.append(s.elapsed_time(e))
times.sort(); trim=int(len(times)*TRIM)
times=times[trim:-trim] if trim>0 else times
avg=sum(times)/len(times); t=2.0*M*N*K/(avg*1e-3)/1e12
print(json.dumps({{"tflops": round(t,2), "ms": round(avg,4), "n_iters_kept": len(times), "nz": nz}}))
"""


def bench_one(suffix):
    so_path, script = make_bench_script(suffix)
    if not os.path.exists(so_path):
        return {"error": "missing .so"}
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(GPU)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=1800, env=env)
        if r.returncode != 0:
            return {"error": f"rc={r.returncode}", "stderr": r.stderr[-300:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": str(e)}


def main():
    print(f"R25-G VERIFY3 SD: {M}x{N}x{K} on GPU {GPU}, {N_REPS} reps each")
    out = {"runs": {}}
    for vlabel, vsuffix in [
        ("PARENT_REF", "_ts_lgk2_memc_btw_all"),
        ("gm7_pfoff104", f"_r25g_gm7_pfoff104_K{K}_M{M}_N{N}"),
    ]:
        out["runs"][vlabel] = []
        for rep in range(N_REPS):
            r = bench_one(vsuffix)
            if "error" in r:
                print(f"  {vlabel} rep{rep}  ERR {r.get('error')}")
            else:
                print(f"  {vlabel} rep{rep}  {r['tflops']:>8.2f} TFLOPS")
            out["runs"][vlabel].append(r)
    with open(os.path.join(SCRIPT_DIR, "bench_round25_optG_verify3_SD.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSummary:")
    for vlabel, runs in out["runs"].items():
        xs = [r.get("tflops") for r in runs if r.get("tflops")]
        if xs:
            m = sum(xs)/len(xs); s = (sum((x-m)**2 for x in xs)/len(xs))**0.5
            print(f"  {vlabel:20s} mean={m:7.2f} std={s:6.2f}  runs={xs}")


if __name__ == "__main__":
    main()
