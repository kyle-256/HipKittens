#!/usr/bin/env python3
"""R19C 5-run verify of S5/all on same GPU.

Smoke showed +2.69pp on S5 (4096x32768x14336) for
  parent  = _ts_lgk2_memc_r11_iterilp
  cand    = _ts_lgk2_memc_r19c_iterilp_btw_all

5-run interleaved on the same GPU to eliminate cross-GPU bias (per R18A method).

Gate (per task spec):
  - mean(cand) >= max(parent) over all 5 parent runs (gate1)
  - mean(cand) - mean(parent) >= +1.0pp (gate2)
"""
import json, math, os, subprocess, sys, sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10
GPU = 6
N_RUNS = 5

# (label, M, N, K, comp_tflops, parent, cand)
JOB = ("S5_4096x32768x14336", 4096, 32768, 14336, 5296.1,
       "_ts_lgk2_memc_r11_iterilp",
       "_ts_lgk2_memc_r19c_iterilp_btw_all")


def bench_one(M, N, K, suffix, gpu_id):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return {"error": "missing .so"}
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


def main():
    label, M, N, K, comp, parent, cand = JOB
    print(f"R19C 5-run verify   shape={label}  GPU={GPU}  warmup={WARMUP} iters={ITERS} trim={TRIM}")
    print(f"  parent: {parent}")
    print(f"  cand  : {cand}")
    parent_t = []
    cand_t = []
    interleave = []
    for i in range(N_RUNS):
        rp = bench_one(M, N, K, parent, GPU)
        if "error" in rp:
            print(f"  run {i+1} parent ERROR: {rp}", flush=True)
            return
        parent_t.append(rp["tflops"])
        rc = bench_one(M, N, K, cand, GPU)
        if "error" in rc:
            print(f"  run {i+1} cand   ERROR: {rc}", flush=True)
            return
        cand_t.append(rc["tflops"])
        print(f"  run {i+1}  parent={rp['tflops']:7.2f}  cand={rc['tflops']:7.2f}  Δ={rc['tflops']-rp['tflops']:+7.2f} TFLOPS", flush=True)
        interleave.append({"i": i+1, "parent": rp["tflops"], "cand": rc["tflops"]})
    p_mean = sum(parent_t)/len(parent_t); p_max = max(parent_t); p_min = min(parent_t)
    c_mean = sum(cand_t)/len(cand_t); c_max = max(cand_t); c_min = min(cand_t)
    d_mean_pp = (c_mean - p_mean) / comp * 100
    gate1 = c_mean >= p_max
    gate2 = d_mean_pp >= 1.0
    win = gate1 and gate2
    out = {
        "label": label, "M": M, "N": N, "K": K, "comp": comp, "gpu": GPU,
        "parent_suffix": parent, "cand_suffix": cand,
        "parent_runs": parent_t, "cand_runs": cand_t,
        "parent_mean": p_mean, "parent_max": p_max, "parent_min": p_min,
        "cand_mean": c_mean, "cand_max": c_max, "cand_min": c_min,
        "delta_mean_pp": d_mean_pp,
        "gate1_cand_mean_ge_parent_max": gate1,
        "gate2_delta_ge_1pp": gate2,
        "WIN": win,
        "interleave": interleave,
    }
    print()
    print(f"  parent: mean={p_mean:7.2f}  max={p_max:7.2f}  min={p_min:7.2f}  ({p_mean/comp*100:.2f}% mean)")
    print(f"  cand  : mean={c_mean:7.2f}  max={c_max:7.2f}  min={c_min:7.2f}  ({c_mean/comp*100:.2f}% mean)")
    print(f"  Δmean = {d_mean_pp:+.2f}pp")
    print(f"  gate1 (cand_mean>=parent_max): {gate1}")
    print(f"  gate2 (Δ>=+1.0pp):              {gate2}")
    print(f"  ⇒ {'WIN' if win else 'NO-WIN'}")
    with open(os.path.join(SCRIPT_DIR, "bench_round19_optC_verify.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved bench_round19_optC_verify.json")


if __name__ == "__main__":
    main()
