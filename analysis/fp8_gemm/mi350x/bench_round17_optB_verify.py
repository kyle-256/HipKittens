#!/usr/bin/env python3
"""R17B 5-run verify on smoke-passing P1 candidates.

Single GPU per candidate (sequential). 5 reps base + 5 reps cand interleaved
to control for drift. Gate: mean(cand) >= max(base) AND mean delta_pp >= +0.5pp
(relaxed gate per agent prompt; commit gate +1.0pp).
"""
import json, os, subprocess, sys, sysconfig
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10
N_REPS = 5

SHAPE = ("P1", 28672, 4096, 16384, 5273.0, "_ts_gm8")
GPU_POOL = [2, 3, 4]
NEW_PREFIX = "_r17b_"

# Smoke-passing + boundary-case candidates
CANDIDATES = [
    ("rcg", "noemxpre", "tv16"),         # smoke +0.50pp (boundary)
    ("rcg", "noemxpre", "lgk2"),         # smoke +3.87pp (likely base outlier)
    ("rcg", "tv16", "lgk2"),             # smoke +3.67pp (likely base outlier)
    ("rcg", "noemxpre", "tv16", "lgk2"), # smoke +0.67pp (4-stack)
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


def reps_one(M, N, K, suffix, gpu, n):
    runs = []
    for i in range(n):
        r = bench_one(M, N, K, suffix, gpu)
        runs.append(r)
    return runs


def task(args):
    tags, csuf, gpu = args
    lab, M, N, K, comp, ps = SHAPE
    # Interleave reps: B C B C B C B C B C
    base_runs, cand_runs = [], []
    for i in range(N_REPS):
        b = bench_one(M, N, K, ps, gpu)
        base_runs.append(b)
        c = bench_one(M, N, K, ps + csuf, gpu)
        cand_runs.append(c)
    return tags, csuf, gpu, comp, base_runs, cand_runs


def stat(runs):
    ts = [r["tflops"] for r in runs if "tflops" in r]
    if not ts:
        return None
    return {"runs": ts, "mean": sum(ts)/len(ts), "max": max(ts), "min": min(ts), "n": len(ts)}


def main():
    pairs = []
    for i, tags in enumerate(CANDIDATES):
        csuf = NEW_PREFIX + "X".join(tags)
        gpu = GPU_POOL[i % len(GPU_POOL)]
        pairs.append((tags, csuf, gpu))

    print(f"R17B VERIFY  warmup={WARMUP} iters={ITERS} trim={TRIM} reps={N_REPS}  GPUs={GPU_POOL}")
    print(f"P1: 28672x4096x16384  parent=_ts_gm8  comp_baseline=5273.0 TFLOPS")
    print(f"Total candidates: {len(pairs)}")
    print("=" * 130)

    out = {"warmup": WARMUP, "iters": ITERS, "trim": TRIM, "n_reps": N_REPS,
           "gpus": GPU_POOL, "shape": SHAPE[0],
           "M_N_K": [SHAPE[1], SHAPE[2], SHAPE[3]],
           "comp_baseline": SHAPE[4], "parent_suffix": SHAPE[5],
           "candidates": []}
    wins = []
    relaxed = []
    with ProcessPoolExecutor(max_workers=len(GPU_POOL)) as ex:
        futs = {ex.submit(task, a): a for a in pairs}
        for fut in as_completed(futs):
            tags, csuf, gpu, comp, base_runs, cand_runs = fut.result()
            tag_label = "+".join(tags)
            bs = stat(base_runs); cs = stat(cand_runs)
            if not bs or not cs:
                print(f"  {tag_label:48s} gpu={gpu}  ERROR base={base_runs} cand={cand_runs}")
                out["candidates"].append({"tag":tag_label,"csuf":csuf,"gpu":gpu,
                                          "base_runs":base_runs,"cand_runs":cand_runs})
                continue
            d_mean = cs["mean"] - bs["mean"]
            pp_mean = d_mean / comp * 100
            gate_max = cs["mean"] >= bs["max"]
            gate_pp1 = pp_mean >= 1.0
            gate_pp05 = pp_mean >= 0.5
            verdict = "FAIL"
            if gate_max and gate_pp1:
                verdict = "WIN"
                wins.append((tag_label, csuf, pp_mean))
            elif gate_max and gate_pp05:
                verdict = "RELAXED-PASS"
                relaxed.append((tag_label, csuf, pp_mean))
            print(f"  {tag_label:42s} gpu={gpu}  base mean={bs['mean']:7.1f} max={bs['max']:7.1f}  cand mean={cs['mean']:7.1f} max={cs['max']:7.1f}  d={d_mean:+6.1f}  ({pp_mean:+.3f}pp)  max_gate={'P' if gate_max else 'F'} +1pp={'P' if gate_pp1 else 'F'}  {verdict}", flush=True)
            print(f"      base runs: {bs['runs']}")
            print(f"      cand runs: {cs['runs']}")
            out["candidates"].append({"tag":tag_label,"csuf":csuf,"gpu":gpu,"comp":comp,
                                      "base_stats":bs,"cand_stats":cs,
                                      "delta_mean_tflops":round(d_mean,2),
                                      "delta_mean_pp":round(pp_mean,3),
                                      "gate_mean_ge_basemax":gate_max,
                                      "gate_pp1":gate_pp1,
                                      "gate_pp05":gate_pp05,
                                      "verdict":verdict})

    out["wins_+1pp"] = [{"tag":t,"csuf":c,"delta_pp":round(p,3)} for t,c,p in wins]
    out["relaxed_+0.5pp"] = [{"tag":t,"csuf":c,"delta_pp":round(p,3)} for t,c,p in relaxed]
    with open(os.path.join(SCRIPT_DIR, "bench_round17_optB_verify.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + "=" * 130)
    print(f"WIN (+1pp gate): {len(wins)}")
    for t,c,p in wins:
        print(f"  {t}  +{p:.3f}pp")
    print(f"RELAXED (+0.5pp gate): {len(relaxed)}")
    for t,c,p in relaxed:
        print(f"  {t}  +{p:.3f}pp")
    print("Saved bench_round17_optB_verify.json")


if __name__ == "__main__":
    main()
