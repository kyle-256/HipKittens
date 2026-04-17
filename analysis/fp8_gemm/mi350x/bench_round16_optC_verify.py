#!/usr/bin/env python3
"""R16C 5-run verify: for each smoke-pass candidate, run base & candidate
each 5 times back-to-back on a single GPU. Gate: mean ≥ base.max AND
mean Δ ≥ +1.0pp.
"""
import json, os, statistics, subprocess, sys, sysconfig
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10
N_REPS = 5

SHAPE_INFO = {
    "DLA1": (4096,  32768, 128256, 5781.1, "_ts_pf6_6_v12_memc"),
    "DLA2": (128256, 32768,  4096, 4536.4, "_ts_gm2_v12_memc_dc"),
    "DLA7": (28672, 32768,  4096, 4466.6, "_ts_lgk2_v12_memc"),
    "P1":   (28672,  4096, 16384, 5273.0, "_ts_gm8"),
}

GPU_POOL = [5, 6, 7]


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


def task(args):
    shape, tag_label, csuf, gpu = args
    M, N, K, comp, ps = SHAPE_INFO[shape]
    parent_suffix = ps
    cand_suffix = ps + csuf
    base_runs, cand_runs = [], []
    for _ in range(N_REPS):
        base_runs.append(bench_one(M, N, K, parent_suffix, gpu))
    for _ in range(N_REPS):
        cand_runs.append(bench_one(M, N, K, cand_suffix, gpu))
    return shape, tag_label, csuf, gpu, comp, base_runs, cand_runs


def main():
    smoke = json.load(open(os.path.join(SCRIPT_DIR, "bench_round16_optC_smoke.json")))
    candidates = [(p["shape"], p["tag"], p["csuf"]) for p in smoke["smoke_passes"]]
    print(f"R16C VERIFY  warmup={WARMUP} iters={ITERS} trim={TRIM} reps={N_REPS}")
    print(f"Candidates: {len(candidates)}")
    print("=" * 110)
    args_list = [(s, t, c, GPU_POOL[i % len(GPU_POOL)])
                 for i, (s, t, c) in enumerate(candidates)]

    out = {"warmup": WARMUP, "iters": ITERS, "trim": TRIM, "reps": N_REPS,
           "candidates": []}
    wins = []
    with ProcessPoolExecutor(max_workers=len(GPU_POOL)) as ex:
        futs = {ex.submit(task, a): a for a in args_list}
        for fut in as_completed(futs):
            shape, tag_label, csuf, gpu, comp, base_runs, cand_runs = fut.result()
            base_t = [r["tflops"] for r in base_runs if "tflops" in r]
            cand_t = [r["tflops"] for r in cand_runs if "tflops" in r]
            entry = {"shape": shape, "tag": tag_label, "csuf": csuf, "gpu": gpu,
                     "comp": comp,
                     "base_runs": base_runs, "cand_runs": cand_runs,
                     "base_tflops": base_t, "cand_tflops": cand_t}
            if not base_t or not cand_t:
                entry["verdict"] = "ERROR"
                print(f"  {shape:6s} {tag_label:48s} gpu={gpu}  ERROR (incomplete runs)", flush=True)
                out["candidates"].append(entry)
                continue
            bm = statistics.mean(base_t); bx = max(base_t)
            cm = statistics.mean(cand_t); cx = max(cand_t)
            d = cm - bm
            pp = d / comp * 100
            gate_max = cm >= bx
            gate_pp = pp >= 1.0
            entry["base_mean"] = round(bm, 2); entry["base_max"] = round(bx, 2)
            entry["cand_mean"] = round(cm, 2); entry["cand_max"] = round(cx, 2)
            entry["delta_tflops"] = round(d, 2); entry["delta_pp"] = round(pp, 3)
            entry["gate_mean_ge_max"] = gate_max
            entry["gate_pp_ge_1"] = gate_pp
            entry["verdict"] = "WIN" if (gate_max and gate_pp) else "FAIL"
            print(f"  {shape:6s} {tag_label:48s} gpu={gpu}  base.mean={bm:7.2f} (max={bx:7.2f})  cand.mean={cm:7.2f} (max={cx:7.2f})  Δ={d:+6.2f} ({pp:+.3f}pp)  gate_max={gate_max}  gate_pp={gate_pp}  {entry['verdict']}", flush=True)
            out["candidates"].append(entry)
            if entry["verdict"] == "WIN":
                wins.append(entry)
    out["wins"] = wins
    with open(os.path.join(SCRIPT_DIR, "bench_round16_optC_verify.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + "=" * 110)
    print(f"WINS: {len(wins)}")
    for w in wins:
        print(f"  {w['shape']} {w['tag']}  +{w['delta_pp']}pp  ({w['cand_mean']} vs base.max {w['base_max']})")
    print("Saved bench_round16_optC_verify.json")


if __name__ == "__main__":
    main()
