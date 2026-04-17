#!/usr/bin/env python3
"""Round 11 5-run replication for iterative-ilp candidates with delta >= +0.8pp.
Round 6 methodology: candidate mean must >= baseline max.
GPU 6, warmup=200, iters=500, trim=10%.
"""
import json, math, os, subprocess, sys, time, sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10
GPU = 6
N_RUNS = 5
NEW_SUFFIX = "_r11_iterilp"

# +0.8pp candidates from single-shot
SHAPES = [
    ("DLA3_4096x32768x28672",   4096, 32768, 28672, 5568.2, "_v20_memc"),
    ("DLA4_4096x28672x32768",   4096, 28672, 32768, 5649.9, "_u16"),
    ("DLA6_4096x32768x14336",   4096, 32768, 14336, 5296.1, "_ts_lgk2_memc"),
    ("WIN3_4096x128256x32768",  4096,128256, 32768, 3195.3, "_memc"),
]

def bench_one(M, N, K, full_suffix, gpu_id):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return {"error": "missing .so", "path": so_path}
    script = f"""
import sys, math, torch, importlib.util, json
torch.manual_seed(0)
WARMUP, ITERS, TRIM = {WARMUP}, {ITERS}, {TRIM}
M, N, K = {M}, {N}, {K}
def gen_fp4(r, K):
    c = K // 2
    return (torch.randint(0,16,(r,c),dtype=torch.uint8,device='cuda')<<4)|torch.randint(0,16,(r,c),dtype=torch.uint8,device='cuda')
def preshuffle(se):
    r,kb=se.shape; pr=math.ceil(r/64)*64; pk=math.ceil(kb/8)*8
    raw=torch.full((pr,pk),0x7F,dtype=torch.uint8,device=se.device)
    raw[:r,:kb]=(se.to(torch.int16)+127).to(torch.uint8)
    sh=raw.view(pr//32,2,16,pk//8,2,4,1).permute(0,3,5,2,4,1,6).contiguous().view(pr//32,pk*32)
    sh=sh.view(pr//64,2,pk*32//4,4).permute(0,2,1,3).contiguous()
    return sh.view(pr//64,pk*64)
spec=importlib.util.spec_from_file_location('{module_name}','{so_path}')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
A=gen_fp4(M,K); B=gen_fp4(N,K)
sc_a=torch.randint(-2,3,(M,K//32),dtype=torch.int8,device='cuda')
sc_b=torch.randint(-2,3,(N,K//32),dtype=torch.int8,device='cuda')
A_sc=preshuffle(sc_a); B_sc=preshuffle(sc_b)
C=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
run=lambda:mod.gemm_rcr(A,B,A_sc,B_sc,C)
for _ in range(WARMUP): run()
torch.cuda.synchronize()
times=[]
for _ in range(ITERS):
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record(); run(); e.record(); torch.cuda.synchronize()
    times.append(s.elapsed_time(e))
times.sort(); trim=int(len(times)*TRIM)
times=times[trim:-trim] if trim>0 else times
avg=sum(times)/len(times); t=2.0*M*N*K/(avg*1e-3)/1e12
print(json.dumps({{"tflops":round(t,2),"ms":round(avg,4)}}))
"""
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=900, env=env)
        if r.returncode != 0:
            return {"error": f"rc={r.returncode}", "stderr": r.stderr[-300:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": str(e)}


def stats(vals):
    return {"runs": vals, "mean": round(sum(vals)/len(vals), 2),
            "min": min(vals), "max": max(vals), "n": len(vals)}


def main():
    print(f"Round 11 verify (5-run). GPU={GPU} warmup={WARMUP} iters={ITERS}")
    print("=" * 110)
    out = {"warmup": WARMUP, "iters": ITERS, "trim": TRIM, "gpu": GPU,
           "n_runs": N_RUNS, "shapes": []}
    for (label, M, N, K, comp, parent) in SHAPES:
        cand = parent + NEW_SUFFIX
        print(f"\n{label}  shape={M}x{N}x{K}  comp={comp}  parent={parent}")
        b_vals = []
        for i in range(N_RUNS):
            r = bench_one(M, N, K, parent, GPU)
            if "error" in r:
                print(f"  baseline run{i+1} ERROR: {r}", flush=True)
                continue
            b_vals.append(r["tflops"])
            print(f"  baseline  run{i+1}: {r['tflops']:.2f} TFLOPS", flush=True)
        c_vals = []
        for i in range(N_RUNS):
            r = bench_one(M, N, K, cand, GPU)
            if "error" in r:
                print(f"  candidate run{i+1} ERROR: {r}", flush=True)
                continue
            c_vals.append(r["tflops"])
            print(f"  candidate run{i+1}: {r['tflops']:.2f} TFLOPS", flush=True)
        b_st = stats(b_vals) if b_vals else None
        c_st = stats(c_vals) if c_vals else None
        if b_st and c_st:
            mean_delta = c_st["mean"] - b_st["mean"]
            mean_pp = mean_delta / comp * 100
            gate = c_st["mean"] >= b_st["max"]
            print(f"  baseline  mean={b_st['mean']:.2f} max={b_st['max']:.2f} min={b_st['min']:.2f}")
            print(f"  candidate mean={c_st['mean']:.2f} max={c_st['max']:.2f} min={c_st['min']:.2f}")
            print(f"  delta_mean={mean_delta:+.2f} TFLOPS  ({mean_pp:+.2f}pp)  "
                  f"gate(mean>=base.max)={'PASS' if gate else 'FAIL'}", flush=True)
        out["shapes"].append({
            "label": label, "M": M, "N": N, "K": K, "comp": comp,
            "parent": parent, "candidate": cand,
            "baseline": b_st, "candidate_stats": c_st,
        })
    with open(os.path.join(SCRIPT_DIR, "bench_round11_verify_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved bench_round11_verify_results.json")


if __name__ == "__main__":
    main()
