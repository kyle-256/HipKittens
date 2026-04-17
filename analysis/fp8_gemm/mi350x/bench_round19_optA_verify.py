#!/usr/bin/env python3
"""R19A 5-run verify: same-GPU interleaved parent vs variant.
Bench params: warmup=200 iters=500 trim=10%."""
import json, math, os, subprocess, sys, sysconfig, statistics, time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
WARMUP, ITERS, TRIM = 200, 500, 0.10
N_RUNS = 5

# Each pair gets its own GPU. (label, M, N, K, comp, parent_suffix, variant_suffix, gpu)
PAIRS = [
    ("S1_step3",  14336,  4096, 32768, 5245.4, "_lgk2_dc",     "_r19a_step3",  0),
    ("S1_all",    14336,  4096, 32768, 5245.4, "_lgk2_dc",     "_r19a_all",    1),
    ("S14_step12", 6144,  4096, 16384, 4428.1, "_ts_lgk2",     "_r19a_step12", 2),
    ("S4_step12",  4096, 28672, 32768, 5649.9, "_u16",         "_r19a_step12", 3),
]


def bench_script(M, N, K, suffix):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    return so_path, module_name, f"""
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


def bench_one(M, N, K, suffix, gpu):
    sp, mn, script = bench_script(M, N, K, suffix)
    if not os.path.exists(sp):
        return {"err": f"missing {sp}"}
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu)
    try:
        r = subprocess.run([sys.executable, "-c", script],
            capture_output=True, text=True, timeout=900, env=env)
        if r.returncode != 0:
            return {"err": r.stderr[-300:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"err": str(e)}


def verify_pair(args):
    lab, M, N, K, comp, ps, vs, gpu = args
    p_runs, v_runs = [], []
    for i in range(N_RUNS):
        r = bench_one(M, N, K, ps, gpu)
        p_runs.append(r.get("tflops", 0))
        r = bench_one(M, N, K, ps + vs, gpu)
        v_runs.append(r.get("tflops", 0))
    return lab, p_runs, v_runs, comp


def main():
    print(f"R19 OptA 5-run verify. warmup={WARMUP} iters={ITERS} trim={TRIM} N_RUNS={N_RUNS}")
    print(f"Pairs: {len(PAIRS)}")
    print("=" * 110)
    out = {"params": {"warmup": WARMUP, "iters": ITERS, "trim": TRIM, "n_runs": N_RUNS},
           "pairs": []}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(PAIRS)) as ex:
        futs = {ex.submit(verify_pair, p): p for p in PAIRS}
        for fut in as_completed(futs):
            lab, p_runs, v_runs, comp = fut.result()
            p_max = max(p_runs); p_mean = statistics.mean(p_runs)
            v_max = max(v_runs); v_mean = statistics.mean(v_runs); v_min = min(v_runs)
            delta_pp = (v_mean - p_mean) / comp * 100
            gate1 = v_mean >= p_max
            gate2 = delta_pp >= 1.0
            verdict = "WIN" if (gate1 and gate2) else "FAIL"
            print(f"\n  [{lab}]  comp={comp}")
            print(f"    parent  runs: {[round(x,1) for x in p_runs]}  max={p_max:.2f}  mean={p_mean:.2f}")
            print(f"    variant runs: {[round(x,1) for x in v_runs]}  max={v_max:.2f}  mean={v_mean:.2f}  min={v_min:.2f}")
            print(f"    Δmean={delta_pp:+.2f}pp  gate1={gate1}  gate2={gate2}  -> {verdict}", flush=True)
            out["pairs"].append({
                "lab": lab, "comp": comp,
                "parent_runs": p_runs, "variant_runs": v_runs,
                "parent_max": p_max, "parent_mean": p_mean,
                "variant_max": v_max, "variant_mean": v_mean, "variant_min": v_min,
                "delta_pp": round(delta_pp, 3),
                "gate1": gate1, "gate2": gate2, "verdict": verdict,
            })
    print(f"\nElapsed: {(time.time()-t0)/60:.1f} min")
    with open(os.path.join(SCRIPT_DIR, "bench_round19_optA_verify.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved bench_round19_optA_verify.json")


if __name__ == "__main__":
    main()
