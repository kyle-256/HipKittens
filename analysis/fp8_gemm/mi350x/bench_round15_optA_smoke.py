#!/usr/bin/env python3
"""R15 OptA single-shot smoke bench.

For each (shape, candidate) pair that produced a DIFF asm, run a single
warmup=200/iters=500 trim=10% bench against the corresponding parent on
the SAME GPU (to avoid cross-GPU bias).

Pass criterion: candidate.tflops >= parent.tflops + 0.5pp of competitor.
Anything passing goes to bench_round15_optA_verify.py for 5-run replication.

GPU split: distribute across 3 idle GPUs (0,1,2) per shape — but we run
parent and candidate back-to-back on the SAME GPU within a thread.
"""
import json, math, os, subprocess, sys, time, sysconfig
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10
TIMEOUT = 1200

# (label, M, N, K, comp, parent_suffix, cand_suffix)
TARGETS = [
    # Phase A: regclassglob alone
    ("DLA1", 4096, 32768, 128256, 5781.1,
     "_ts_pf6_6_v12_memc",  "_ts_pf6_6_v12_memc_r15a_regclassglob"),
    ("DLA2", 128256, 32768, 4096, 4536.4,
     "_ts_gm2_v12_memc_dc", "_ts_gm2_v12_memc_dc_r15a_regclassglob"),
    ("DLA7", 28672, 32768, 4096, 4466.6,
     "_ts_lgk2_v12_memc",   "_ts_lgk2_v12_memc_r15a_regclassglob"),
    ("P1",   28672, 4096, 16384, 5273.0,
     "_ts_gm8",             "_ts_gm8_r15a_regclassglob"),
    # Phase B: P1 compounds (parent is bare _ts_gm8 throughout)
    ("P1c", 28672, 4096, 16384, 5273.0, "_ts_gm8", "_ts_gm8_r15a_regclassglob_v20"),
    ("P1c", 28672, 4096, 16384, 5273.0, "_ts_gm8", "_ts_gm8_r15a_regclassglob_v24"),
    ("P1c", 28672, 4096, 16384, 5273.0, "_ts_gm8", "_ts_gm8_r15a_regclassglob_lgk2"),
    ("P1c", 28672, 4096, 16384, 5273.0, "_ts_gm8", "_ts_gm8_r15a_regclassglob_extbr"),
    # _ts_gm8_r15a_regclassglob_noembed is NOOP vs bare rcg per asm probe; skip
    ("P1c", 28672, 4096, 16384, 5273.0, "_ts_gm8", "_ts_gm8_r15a_regclassglob_tv0"),
    ("P1c", 28672, 4096, 16384, 5273.0, "_ts_gm8", "_ts_gm8_r15a_regclassglob_tv16"),
]

GPUS = [0, 1, 2]
GATE_PP = 0.5  # pp of comp


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
                           capture_output=True, text=True, timeout=TIMEOUT, env=env)
        if r.returncode != 0:
            return {"error": f"rc={r.returncode}", "stderr": r.stderr[-200:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": str(e)}


def run_target(idx, lab, M, N, K, comp, parent_suf, cand_suf):
    gpu = GPUS[idx % len(GPUS)]
    print(f">> [GPU{gpu}] {lab}: {cand_suf} vs {parent_suf}", flush=True)
    pr = bench_one(M, N, K, parent_suf, gpu)
    cr = bench_one(M, N, K, cand_suf, gpu)
    if "error" in pr or "error" in cr:
        print(f"<< [GPU{gpu}] {lab} {cand_suf}: ERR parent={pr} cand={cr}", flush=True)
        return {"label": lab, "gpu": gpu, "parent": parent_suf, "candidate": cand_suf,
                "error": True, "parent_result": pr, "cand_result": cr}
    delta = cr["tflops"] - pr["tflops"]
    pp = delta / comp * 100
    gate_pass = pp >= GATE_PP
    print(f"<< [GPU{gpu}] {lab} {cand_suf}: parent={pr['tflops']:.2f} cand={cr['tflops']:.2f} "
          f"delta={delta:+.2f} ({pp:+.2f}pp) gate>={GATE_PP}pp: {'PASS' if gate_pass else 'FAIL'}",
          flush=True)
    return {"label": lab, "gpu": gpu, "comp": comp,
            "parent": parent_suf, "candidate": cand_suf,
            "parent_tflops": pr["tflops"], "cand_tflops": cr["tflops"],
            "delta_tflops": round(delta, 2), "delta_pp": round(pp, 3),
            "gate_pass": gate_pass}


def main():
    print(f"R15A smoke. GPUS={GPUS}  warmup={WARMUP} iters={ITERS} trim={TRIM}  gate=+{GATE_PP}pp")
    print("=" * 110)
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=len(GPUS)) as ex:
        futs = [ex.submit(run_target, i, *t) for i, t in enumerate(TARGETS)]
        for f in as_completed(futs):
            results.append(f.result())
    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed/60:.1f} min")
    out = {"warmup": WARMUP, "iters": ITERS, "trim": TRIM, "gate_pp": GATE_PP,
           "gpus": GPUS, "elapsed_minutes": round(elapsed/60, 2),
           "results": results}
    with open(os.path.join(SCRIPT_DIR, "bench_round15_optA_smoke.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + "=" * 110)
    print(f"{'Label':6s} {'Candidate':55s} {'parent':>9s} {'cand':>9s} {'delta':>9s} {'pp':>8s}  gate")
    print("-" * 110)
    for r in sorted(results, key=lambda x: (x["label"], x.get("candidate",""))):
        if r.get("error"):
            print(f"{r['label']:6s} {r['candidate']:55s}  ERROR")
            continue
        print(f"{r['label']:6s} {r['candidate']:55s} {r['parent_tflops']:9.2f} {r['cand_tflops']:9.2f} "
              f"{r['delta_tflops']:+9.2f} {r['delta_pp']:+7.3f}pp  {'PASS' if r['gate_pass'] else 'fail'}")
    n_pass = sum(1 for r in results if r.get("gate_pass"))
    print(f"\nGate-pass count: {n_pass}/{len(results)}")
    print("Saved bench_round15_optA_smoke.json")


if __name__ == "__main__":
    main()
