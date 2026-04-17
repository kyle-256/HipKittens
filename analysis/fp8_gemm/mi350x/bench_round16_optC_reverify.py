#!/usr/bin/env python3
"""R16C re-verify: P1 regclassglob+noemxpre with a 1-run discard + 6 reps.
Goal: rule out the cold-start outlier from the first verify pass.
"""
import json, os, statistics, subprocess, sys, sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP, ITERS, TRIM, N_REPS, GPU = 200, 500, 0.10, 6, 5
M, N, K = 28672, 4096, 16384
COMP = 5273.0
PARENT = "_ts_gm8"
CSUF = "_r16c_regclassglobXnoemxpre"


def bench_one(suffix):
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
    env["HIP_VISIBLE_DEVICES"] = str(GPU)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=1500, env=env)
        if r.returncode != 0:
            return {"error": f"rc={r.returncode}", "stderr": r.stderr[-300:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": str(e)}


def main():
    cand_suffix = PARENT + CSUF
    print(f"R16C reverify: P1 regclassglob+noemxpre, GPU={GPU}, reps={N_REPS} (+1 discard each)")
    # Discard first base + first cand (cold-start filter)
    print("Discard runs:")
    db = bench_one(PARENT); print(f"  base discard: {db}")
    dc = bench_one(cand_suffix); print(f"  cand discard: {dc}")
    base_runs = [bench_one(PARENT) for _ in range(N_REPS)]
    cand_runs = [bench_one(cand_suffix) for _ in range(N_REPS)]
    base_t = [r["tflops"] for r in base_runs if "tflops" in r]
    cand_t = [r["tflops"] for r in cand_runs if "tflops" in r]
    print(f"\nbase_runs: {base_t}")
    print(f"cand_runs: {cand_t}")
    bm = statistics.mean(base_t); bx = max(base_t)
    cm = statistics.mean(cand_t); cx = max(cand_t)
    d = cm - bm; pp = d / COMP * 100
    gate_max = cm >= bx; gate_pp = pp >= 1.0
    verdict = "WIN" if (gate_max and gate_pp) else "FAIL"
    print(f"\nbase.mean={bm:.2f} (max={bx:.2f})  cand.mean={cm:.2f} (max={cx:.2f})  Δ={d:+.2f} ({pp:+.3f}pp)")
    print(f"gate_max={gate_max}  gate_pp={gate_pp}  -> {verdict}")
    out = {"warmup": WARMUP, "iters": ITERS, "trim": TRIM, "reps": N_REPS,
           "gpu": GPU, "comp": COMP, "parent_suffix": PARENT, "csuf": CSUF,
           "base_discard": db, "cand_discard": dc,
           "base_runs": base_runs, "cand_runs": cand_runs,
           "base_mean": round(bm,2), "base_max": round(bx,2),
           "cand_mean": round(cm,2), "cand_max": round(cx,2),
           "delta_tflops": round(d,2), "delta_pp": round(pp,3),
           "gate_max": gate_max, "gate_pp": gate_pp, "verdict": verdict}
    with open(os.path.join(SCRIPT_DIR, "bench_round16_optC_reverify.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved bench_round16_optC_reverify.json")


if __name__ == "__main__":
    main()
