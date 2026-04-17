#!/usr/bin/env python3
"""R19B 5-run same-GPU verify on DLA1/_r19b_t1.

Same-GPU interleaved: parent and variant run alternately on same GPU 5 to eliminate
cross-GPU bias. warmup=200 iters=500 trim=10%.

Gates:
  gate1: variant_mean >= parent_max
  gate2: variant_mean - parent_mean >= +1.0pp of competitor (5781.1 for DLA1)
"""
import json, math, os, subprocess, sys, time, sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10
N_RUNS = 5
GPU = 5

# DLA1
M, N, K = 4096, 32768, 128256
COMP = 5781.1
PARENT_SUFFIX = "_ts_pf6_6_v12_memc"
VARIANT_SUFFIX = "_ts_pf6_6_v12_memc_r19b_t1"


def bench_one(suffix, gpu_id):
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
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=900, env=env)
        if r.returncode != 0:
            return {"err": r.stderr[-300:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"err": str(e)}


def main():
    print(f"R19B 5-run same-GPU verify: DLA1 (M={M}, N={N}, K={K}), comp={COMP}")
    print(f"GPU={GPU}, warmup={WARMUP}, iters={ITERS}, trim={TRIM}, N_RUNS={N_RUNS}")
    print(f"Parent:  {PARENT_SUFFIX}")
    print(f"Variant: {VARIANT_SUFFIX}")
    print("=" * 100)

    parent_runs = []
    variant_runs = []
    # Interleaved: P V P V P V P V P V
    for i in range(N_RUNS):
        rp = bench_one(PARENT_SUFFIX, GPU)
        print(f"  parent  run {i+1}: {rp}", flush=True)
        if "err" in rp:
            print(f"FATAL: parent failed: {rp}"); return 1
        parent_runs.append(rp["tflops"])

        rv = bench_one(VARIANT_SUFFIX, GPU)
        print(f"  variant run {i+1}: {rv}", flush=True)
        if "err" in rv:
            print(f"FATAL: variant failed: {rv}"); return 1
        variant_runs.append(rv["tflops"])

    print("=" * 100)
    p_mean = sum(parent_runs) / len(parent_runs)
    p_max = max(parent_runs)
    v_mean = sum(variant_runs) / len(variant_runs)
    v_min = min(variant_runs)
    v_max = max(variant_runs)
    delta_pp = (v_mean - p_mean) / COMP * 100

    gate1 = v_mean >= p_max
    gate2 = delta_pp >= 1.0
    verdict = "WIN" if (gate1 and gate2) else "LOSE"

    print(f"\nparent  runs: {parent_runs}  max={p_max:.2f}  mean={p_mean:.2f}")
    print(f"variant runs: {variant_runs}  max={v_max:.2f}  mean={v_mean:.2f}  min={v_min:.2f}")
    print(f"Δmean = {delta_pp:+.2f} pp (vs comp {COMP})")
    print(f"gate1 (v_mean >= p_max): {gate1}   gate2 (Δ >= +1pp): {gate2}   ⇒ {verdict}")

    out = {
        "shape": "DLA1", "M": M, "N": N, "K": K, "comp": COMP,
        "gpu": GPU, "warmup": WARMUP, "iters": ITERS, "trim": TRIM, "n_runs": N_RUNS,
        "parent_suffix": PARENT_SUFFIX, "variant_suffix": VARIANT_SUFFIX,
        "parent_runs": parent_runs, "variant_runs": variant_runs,
        "parent_mean": round(p_mean, 2), "parent_max": round(p_max, 2),
        "variant_mean": round(v_mean, 2), "variant_max": round(v_max, 2), "variant_min": round(v_min, 2),
        "delta_pp": round(delta_pp, 2),
        "gate1_v_mean_ge_p_max": gate1, "gate2_delta_ge_1pp": gate2,
        "verdict": verdict,
    }
    with open(os.path.join(SCRIPT_DIR, "bench_round19_optB_verify.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved bench_round19_optB_verify.json")


if __name__ == "__main__":
    sys.exit(main() or 0)
