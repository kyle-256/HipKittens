#!/usr/bin/env python3
"""R17B FINAL focused re-run: best candidate rcg+noemxpre+tv16 only,
on GPU 2 (no contamination, single GPU). 8 reps + discard 1.
Goal: pin down the rcg+noemxpre+tv16 signal cleanly.
"""
import json, os, subprocess, sys, sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10
N_REPS = 8
DISCARD = 1

SHAPE = ("P1", 28672, 4096, 16384, 5273.0, "_ts_gm8")
GPU = 2
NEW_PREFIX = "_r17b_"

CANDIDATES = [
    ("rcg", "noemxpre", "tv16"),
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
                           capture_output=True, text=True, timeout=1500, env=env)
        if r.returncode != 0:
            return {"error": f"rc={r.returncode}", "stderr": r.stderr[-300:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": str(e)}


def stat(runs, discard=0):
    runs = runs[discard:]
    ts = [r["tflops"] for r in runs if "tflops" in r]
    if not ts:
        return None
    return {"runs": ts, "mean": sum(ts)/len(ts), "max": max(ts), "min": min(ts), "n": len(ts)}


def main():
    out = {"warmup": WARMUP, "iters": ITERS, "trim": TRIM, "n_reps": N_REPS,
           "discard": DISCARD, "gpu": GPU, "shape": SHAPE[0],
           "M_N_K": [SHAPE[1], SHAPE[2], SHAPE[3]],
           "comp_baseline": SHAPE[4], "parent_suffix": SHAPE[5],
           "candidates": []}
    lab, M, N, K, comp, ps = SHAPE
    print(f"R17B FINAL  warmup={WARMUP} iters={ITERS} reps={N_REPS} discard={DISCARD}  GPU={GPU}")
    print(f"P1: {M}x{N}x{K}  parent={ps}  comp={comp} TFLOPS")
    print("=" * 130)
    for tags in CANDIDATES:
        csuf = NEW_PREFIX + "X".join(tags)
        tag_label = "+".join(tags)
        base_runs, cand_runs = [], []
        for i in range(N_REPS):
            b = bench_one(M, N, K, ps, GPU)
            base_runs.append(b)
            c = bench_one(M, N, K, ps + csuf, GPU)
            cand_runs.append(c)
            print(f"  rep {i}: base={b.get('tflops')} cand={c.get('tflops')}", flush=True)
        bs = stat(base_runs, discard=DISCARD); cs = stat(cand_runs, discard=DISCARD)
        d_mean = cs["mean"] - bs["mean"]
        pp_mean = d_mean / comp * 100
        gate_max = cs["mean"] >= bs["max"]
        gate_pp1 = pp_mean >= 1.0
        gate_pp05 = pp_mean >= 0.5
        verdict = "WIN" if (gate_max and gate_pp1) else ("RELAXED-PASS" if (gate_max and gate_pp05) else "FAIL")
        print(f"\n  {tag_label}")
        print(f"  base: mean={bs['mean']:.2f} max={bs['max']:.2f} runs={bs['runs']}")
        print(f"  cand: mean={cs['mean']:.2f} max={cs['max']:.2f} runs={cs['runs']}")
        print(f"  delta_mean={d_mean:+.2f} pp={pp_mean:+.3f}  max_gate={gate_max} +1pp={gate_pp1}  +0.5pp={gate_pp05}  -> {verdict}")
        out["candidates"].append({"tag":tag_label,"csuf":csuf,"gpu":GPU,"comp":comp,
                                   "base_stats":bs,"cand_stats":cs,
                                   "delta_mean_tflops":round(d_mean,2),
                                   "delta_mean_pp":round(pp_mean,3),
                                   "gate_mean_ge_basemax":gate_max,
                                   "gate_pp1":gate_pp1, "gate_pp05":gate_pp05,
                                   "verdict":verdict})
    with open(os.path.join(SCRIPT_DIR, "bench_round17_optB_final.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved bench_round17_optB_final.json")


if __name__ == "__main__":
    main()
