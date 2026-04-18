#!/usr/bin/env python3
"""R25-G VERIFY2: clean re-bench of SC and SD parent + winner on isolated GPUs.

In verify1, GPU 4 had multiple `rc=-6` (aperture violations) suggesting GPU
contention during the run. Re-bench with each shape on its own GPU and one
shape at a time (no parallelism) to get a clean PARENT_REF and winner mean.
"""
import json, os, subprocess, sys, sysconfig, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10
N_REPS = 3

# (label, M, N, K, parent_variant, candidate_pfoffs, gpu)
SHAPES = [
    ("SC", 14336, 4096, 32768, "ts_lgk2_memc_btw_all", [124], 3),
    ("SD", 16384, 4096, 28672, "ts_lgk2_memc_btw_all", [110], 6),
]

GM = 7


def variants_for(parent, pfoffs, K, M, N):
    out = [("PARENT_REF", f"_{parent}")]
    for pf in pfoffs:
        out.append((f"gm7_pfoff{pf}", f"_r25g_gm{GM}_pfoff{pf}_K{K}_M{M}_N{N}"))
    return out


def make_bench_script(M, N, K, suffix):
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


def bench_one(M, N, K, vsuffix, gpu):
    so_path, script = make_bench_script(M, N, K, vsuffix)
    if not os.path.exists(so_path):
        return {"error": "missing .so"}
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=1800, env=env)
        if r.returncode != 0:
            return {"error": f"rc={r.returncode}", "stderr": r.stderr[-300:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": str(e)}


def main():
    out = {"shapes": {}, "warmup": WARMUP, "iters": ITERS, "trim": TRIM, "n_reps": N_REPS}
    print(f"R25-G VERIFY2: SC, SD only, isolated GPUs, serial")
    print("=" * 80)
    t0 = time.time()
    for shape in SHAPES:
        lab, M, N, K, parent, pfoffs, gpu = shape
        print(f"\n[{lab}] {M}x{N}x{K} GPU {gpu}", flush=True)
        out["shapes"][lab] = {}
        for (vlabel, vsuffix) in variants_for(parent, pfoffs, K, M, N):
            out["shapes"][lab][vlabel] = {"suffix": vsuffix, "runs": []}
            for rep in range(N_REPS):
                r = bench_one(M, N, K, vsuffix, gpu)
                if "error" in r:
                    print(f"  {lab} {vlabel} rep{rep}  ERR {r.get('error')}", flush=True)
                else:
                    print(f"  {lab} {vlabel} rep{rep}  {r['tflops']:>8.2f} TFLOPS", flush=True)
                out["shapes"][lab][vlabel]["runs"].append(r)
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    with open(os.path.join(SCRIPT_DIR, "bench_round25_optG_verify2.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 80)
    print(f"{'Shape':5s} {'Variant':22s} {'Mean':>10s} {'Std':>8s} {'Δ%':>10s}")
    print("=" * 80)
    for shape in SHAPES:
        lab, M, N, K, parent, pfoffs, gpu = shape
        sd = out["shapes"][lab]
        ref_runs = [r.get("tflops") for r in sd["PARENT_REF"]["runs"] if r.get("tflops")]
        ref_mean = sum(ref_runs)/len(ref_runs) if ref_runs else None
        ref_std = (sum((x-ref_mean)**2 for x in ref_runs)/len(ref_runs))**0.5 if ref_runs else 0
        print(f"{lab:5s} {'PARENT_REF':22s} {ref_mean:>10.2f} {ref_std:>8.2f} {'—':>10s}")
        for pf in pfoffs:
            v = sd[f"gm7_pfoff{pf}"]
            xs = [r.get("tflops") for r in v["runs"] if r.get("tflops")]
            if xs:
                m = sum(xs)/len(xs)
                s = (sum((x-m)**2 for x in xs)/len(xs))**0.5
                d = (m-ref_mean)/ref_mean*100 if ref_mean else 0
                tag = "FAIL_STD" if s > 30 else "OK"
                print(f"{lab:5s} {'gm7_pfoff'+str(pf):22s} {m:>10.2f} {s:>8.2f} {d:>9.2f}%  [{tag}]")


if __name__ == "__main__":
    main()
