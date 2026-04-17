#!/usr/bin/env python3
"""Cross-check verify: parent and variant on the SAME gpu (sequential).
Eliminates the cross-GPU bias risk identified in R12B."""
import json, math, os, subprocess, sys, time, sysconfig, statistics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
WARMUP, ITERS, TRIM = 200, 500, 0.10
N_RUNS = 5
GPU = 0

PAIRS = [
    ("P1", 28672, 4096, 16384, 5350.6, "_ts_gm8", "_r18a_p3_all"),
]

def bench_one(M, N, K, suffix, gpu_id):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return {"err": f"missing {so_path}"}
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
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=900, env=env)
        if r.returncode != 0: return {"err": r.stderr[-300:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e: return {"err": str(e)}


def main():
    t0 = time.time()
    print(f"R18 OptA same-gpu verify (gpu={GPU}). warmup={WARMUP} iters={ITERS} trim={TRIM} N_RUNS={N_RUNS}.")
    print("=" * 100)
    out = {"gpu": GPU, "pairs": []}
    for (lab, M, N, K, comp, ps, vs) in PAIRS:
        p_runs, v_runs = [], []
        # Interleaved to share cooldown
        for i in range(N_RUNS):
            r = bench_one(M, N, K, ps, GPU)
            t = r.get("tflops", 0); p_runs.append(t)
            print(f"  {lab} parent  run {i}: {t}", flush=True)
            r = bench_one(M, N, K, ps + vs, GPU)
            t = r.get("tflops", 0); v_runs.append(t)
            print(f"  {lab} variant run {i}: {t}", flush=True)
        p_max = max(p_runs); p_mean = statistics.mean(p_runs)
        v_max = max(v_runs); v_mean = statistics.mean(v_runs); v_min = min(v_runs)
        delta_pp = (v_mean - p_mean) / comp * 100
        gate1 = v_mean >= p_max
        gate2 = delta_pp >= 1.0
        verdict = "WIN" if (gate1 and gate2) else "FAIL"
        print(f"\n  {lab} {vs}  (comp={comp})")
        print(f"    parent  runs: {[round(x,1) for x in p_runs]}  max={p_max:.2f}  mean={p_mean:.2f}")
        print(f"    variant runs: {[round(x,1) for x in v_runs]}  max={v_max:.2f}  mean={v_mean:.2f}  min={v_min:.2f}")
        print(f"    Δmean={delta_pp:+.2f}pp  gate1={gate1}  gate2={gate2}  -> {verdict}")
        out["pairs"].append({
            "lab": lab, "shape": [M, N, K], "comp": comp,
            "parent": ps, "variant": vs,
            "parent_runs": p_runs, "variant_runs": v_runs,
            "parent_max": p_max, "parent_mean": p_mean,
            "variant_max": v_max, "variant_mean": v_mean, "variant_min": v_min,
            "delta_pp": round(delta_pp, 3),
            "gate1": gate1, "gate2": gate2, "verdict": verdict,
        })
    with open(os.path.join(SCRIPT_DIR, "bench_round18_optA_verify_samegpu.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nElapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
