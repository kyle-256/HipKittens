#!/usr/bin/env python3
"""R16B verify: 5-run replication for smoke-PASS / borderline R16B candidates.

Promotes only candidates that beat the iterilp-only baseline by ≥+0.5pp on
mean (relaxed gate per AGENT_PROMPT) AND mean ≥ baseline.max.

Targets: smoke-PASS (S2/largeivf2 at +0.51pp) plus 3 borderline noemxpre
(S1/S2/S3 at +0.24..+0.30pp single-shot — may pass with replication).

5-run, single GPU per shape, warmup=200, iters=500, trim=10%.
"""
import json, math, os, subprocess, sys, sysconfig, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10
GPU = 2  # single idle GPU
N_RUNS = 5

# (label, tag, M, N, K, comp, parent_suffix, iterilp_suffix, cand_suffix)
TARGETS = [
    # Smoke-PASS
    ("S2", "largeivf2", 16384, 4096, 28672, 5525.3,
     "_u32", "_u32_r10_iterilp", "_u32_r16b_iterilp_largeivf2"),
    # Borderline noemxpre (single-shot +0.24..+0.30pp positive)
    ("S1", "noemxpre", 14336, 4096, 32768, 5245.4,
     "_lgk2_dc", "_lgk2_dc_r10_iterilp", "_lgk2_dc_r16b_iterilp_noemxpre"),
    ("S2", "noemxpre", 16384, 4096, 28672, 5525.3,
     "_u32", "_u32_r10_iterilp", "_u32_r16b_iterilp_noemxpre"),
    ("S3", "noemxpre",  4096, 32768, 28672, 5568.2,
     "_v20_memc", "_v20_memc_r11_iterilp", "_v20_memc_r16b_iterilp_noemxpre"),
]


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
                           capture_output=True, text=True, timeout=1800, env=env)
        if r.returncode != 0:
            return {"error": f"rc={r.returncode}", "stderr": r.stderr[-300:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": str(e)}


def stats(vals):
    return {"runs": vals, "mean": round(sum(vals) / len(vals), 2),
            "min": min(vals), "max": max(vals), "n": len(vals)}


def main():
    print(f"R16B verify (5-run). GPU={GPU} warmup={WARMUP} iters={ITERS} trim={TRIM}")
    print("=" * 110)
    out = {"warmup": WARMUP, "iters": ITERS, "trim": TRIM, "gpu": GPU,
           "n_runs": N_RUNS, "results": []}
    t0 = time.time()
    for (lab, tag, M, N, K, comp, parent_suf, ilp_suf, cand_suf) in TARGETS:
        print(f"\n{lab}/{tag}  parent={parent_suf}  ilp={ilp_suf}  cand={cand_suf}  comp={comp}",
              flush=True)
        # Run iterilp baseline 5x
        b_vals = []
        for i in range(N_RUNS):
            r = bench_one(M, N, K, ilp_suf, GPU)
            if "error" in r:
                print(f"  iterilp run{i+1} ERROR: {r}", flush=True)
                continue
            b_vals.append(r["tflops"])
            print(f"  iterilp   run{i+1}: {r['tflops']:.2f} TFLOPS", flush=True)
        c_vals = []
        for i in range(N_RUNS):
            r = bench_one(M, N, K, cand_suf, GPU)
            if "error" in r:
                print(f"  candidate run{i+1} ERROR: {r}", flush=True)
                continue
            c_vals.append(r["tflops"])
            print(f"  candidate run{i+1}: {r['tflops']:.2f} TFLOPS", flush=True)
        b_st = stats(b_vals) if b_vals else None
        c_st = stats(c_vals) if c_vals else None
        gate_max = gate_pp = False
        delta_mean = delta_pp = None
        if b_st and c_st:
            delta_mean = c_st["mean"] - b_st["mean"]
            delta_pp = delta_mean / comp * 100
            gate_max = c_st["mean"] >= b_st["max"]
            gate_pp = delta_pp >= 0.5
            print(f"  iterilp    mean={b_st['mean']:.2f} max={b_st['max']:.2f}", flush=True)
            print(f"  candidate  mean={c_st['mean']:.2f} max={c_st['max']:.2f}", flush=True)
            print(f"  delta_mean={delta_mean:+.2f}  Δpp={delta_pp:+.2f}  "
                  f"gate(mean>=ilp.max)={'PASS' if gate_max else 'FAIL'}  "
                  f"gate(Δpp>=0.5)={'PASS' if gate_pp else 'FAIL'}", flush=True)
        out["results"].append({
            "label": lab, "tag": tag, "M": M, "N": N, "K": K, "comp": comp,
            "parent": parent_suf, "iterilp": ilp_suf, "candidate": cand_suf,
            "iterilp_stats": b_st, "candidate_stats": c_st,
            "delta_mean": delta_mean, "delta_pp": delta_pp,
            "gate_max_pass": gate_max, "gate_pp_pass": gate_pp,
            "gate_combined": gate_max and gate_pp,
        })
    elapsed = time.time() - t0
    out["elapsed_minutes"] = round(elapsed / 60, 2)
    print(f"\nElapsed: {elapsed/60:.1f} min")
    with open(os.path.join(SCRIPT_DIR, "bench_round16_optB_verify.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + "=" * 110)
    print(f"{'Label':6s} {'Tag':16s} {'ilp.mean':>9s} {'cand.mean':>9s} "
          f"{'Δmean':>8s} {'Δpp':>7s}  gate_max  gate_pp  combined")
    print("-" * 110)
    for r in out["results"]:
        if r["delta_mean"] is None:
            print(f"{r['label']:6s} {r['tag']:16s}  ERROR")
            continue
        print(f"{r['label']:6s} {r['tag']:16s} "
              f"{r['iterilp_stats']['mean']:9.2f} {r['candidate_stats']['mean']:9.2f} "
              f"{r['delta_mean']:+8.2f} {r['delta_pp']:+6.2f}pp  "
              f"{'PASS' if r['gate_max_pass'] else 'fail':>8s}  "
              f"{'PASS' if r['gate_pp_pass'] else 'fail':>7s}  "
              f"{'PASS' if r['gate_combined'] else 'fail':>8s}")
    n_pass = sum(1 for r in out["results"] if r.get("gate_combined"))
    print(f"\nGate-combined (max+pp) PASS: {n_pass}/{len(out['results'])}")
    print("Saved bench_round16_optB_verify.json")


if __name__ == "__main__":
    main()
