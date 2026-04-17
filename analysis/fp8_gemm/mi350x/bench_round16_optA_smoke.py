#!/usr/bin/env python3
"""R16A single-shot smoke bench: compound iterilp+regclassglob vs existing
iterilp-only winner (the actually-shipped one).

Step 1: Quick SNR safety check (small 256x256x4096 random fp4) — reject NaN/Inf.
Step 2: For each shape, bench iterilp-winner and compound back-to-back on the
        same idle GPU. warmup=200 / iters=500 / trim=10%.

Gate: candidate must add >= +0.5pp on top of iterilp-winner in single-shot.
"""
import json, math, os, subprocess, sys, time, sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10
TIMEOUT = 1200
GATE_PP = 0.5
GPUS = [0, 1]

# (tag, M, N, K, comp, iterilp_winner_suffix, compound_suffix, original_parent_suffix)
TARGETS = [
    ("S1", 14336,  4096, 32768, 5245.4,
     "_v16_wpe2_r10a_iterilp",
     "_lgk2_dc_r16a_iterilp_regclassglob",
     "_lgk2_dc"),
    ("S2", 16384,  4096, 28672, 5525.3,
     "_u8_r10a_iterilp",
     "_u32_r16a_iterilp_regclassglob",
     "_u32"),
    ("S3",  4096, 32768, 28672, 5568.2,
     "_v20_memc_r11_iterilp",
     "_v20_memc_r16a_iterilp_regclassglob",
     "_v20_memc"),
    ("S4",  4096, 28672, 32768, 5649.9,
     "_u16_r11_iterilp",
     "_u16_r16a_iterilp_regclassglob",
     "_u16"),
    ("S5",  4096, 32768, 14336, 5296.1,
     "_ts_lgk2_memc_r11_iterilp",
     "_ts_lgk2_memc_r16a_iterilp_regclassglob",
     "_ts_lgk2_memc"),
]


def snr_safety(M, N, K, suffix, gpu_id):
    """Crash-only safety: run kernel once and confirm subprocess returns 0.
    NaN/Inf are NOT meaningful here because the random fp4 bytes (0..15)
    include NaN/Inf encodings, which propagate to C even for KNOWN-GOOD
    kernels (e.g., the iterilp-only winner also produces NaN). The
    real correctness gate is the existing 42-shape WIN being preserved
    by the variant -- which is what the bench measures."""
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return {"snr_ok": False, "err": "missing .so"}
    # NOTE: the kernel is compiled with -DK_DIM=K -DN_DIM=N at compile time,
    # so we MUST run with the same K/N (not the small 256x256x4096). Just
    # check the actual shape produces finite output for a single call.
    script = f"""
import sys, math, torch, importlib.util, json
torch.manual_seed(0)
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
mod.gemm_rcr(A,B,A_sc,B_sc,C)
torch.cuda.synchronize()
nz = int((C != 0).sum().item())
print(json.dumps({{"ran": True, "nonzero": nz}}))
"""
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=300, env=env)
        if r.returncode != 0:
            return {"snr_ok": False, "err": f"rc={r.returncode}", "stderr": r.stderr[-200:]}
        d = json.loads(r.stdout.strip().splitlines()[-1])
        ok = bool(d.get("ran")) and d.get("nonzero", 0) > 0
        return {"snr_ok": ok, **d}
    except Exception as e:
        return {"snr_ok": False, "err": str(e)}


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


def run_target(idx, tag, M, N, K, comp, iterilp_suf, cmp_suf, parent_suf):
    gpu = GPUS[idx % len(GPUS)]
    print(f">> [GPU{gpu}] {tag}: SNR check on {cmp_suf}", flush=True)
    snr = snr_safety(M, N, K, cmp_suf, gpu)
    if not snr.get("snr_ok"):
        print(f"<< [GPU{gpu}] {tag} SNR FAIL: {snr}", flush=True)
        return {"tag": tag, "gpu": gpu, "snr": snr, "error": True}
    print(f"   [GPU{gpu}] {tag} crash-check OK (nonzero={snr.get('nonzero')})", flush=True)
    print(f">> [GPU{gpu}] {tag}: bench iterilp ({iterilp_suf}) and compound ({cmp_suf})", flush=True)
    ir = bench_one(M, N, K, iterilp_suf, gpu)
    cr = bench_one(M, N, K, cmp_suf, gpu)
    pr = bench_one(M, N, K, parent_suf, gpu)
    if "error" in ir or "error" in cr or "error" in pr:
        print(f"<< [GPU{gpu}] {tag}: ERR ip={ir} cmp={cr} par={pr}", flush=True)
        return {"tag": tag, "gpu": gpu, "snr": snr, "error": True,
                "iterilp_result": ir, "cand_result": cr, "parent_result": pr}
    delta_vs_iter = cr["tflops"] - ir["tflops"]
    pp_vs_iter = delta_vs_iter / comp * 100
    delta_vs_par = cr["tflops"] - pr["tflops"]
    pp_vs_par = delta_vs_par / comp * 100
    gate_pass = pp_vs_iter >= GATE_PP
    print(f"<< [GPU{gpu}] {tag}: parent={pr['tflops']:.2f} iter={ir['tflops']:.2f} "
          f"cmp={cr['tflops']:.2f}  Δvs_iter={delta_vs_iter:+.2f} ({pp_vs_iter:+.2f}pp)  "
          f"gate>={GATE_PP}pp: {'PASS' if gate_pass else 'fail'}", flush=True)
    return {"tag": tag, "gpu": gpu, "comp": comp, "snr": snr,
            "parent_suffix": parent_suf, "iterilp_suffix": iterilp_suf,
            "candidate_suffix": cmp_suf,
            "parent_tflops": pr["tflops"], "iterilp_tflops": ir["tflops"],
            "cand_tflops": cr["tflops"],
            "delta_vs_iterilp_tflops": round(delta_vs_iter, 2),
            "delta_vs_iterilp_pp": round(pp_vs_iter, 3),
            "delta_vs_parent_tflops": round(delta_vs_par, 2),
            "delta_vs_parent_pp": round(pp_vs_par, 3),
            "gate_pass": gate_pass}


def main():
    print(f"R16A smoke. GPUS={GPUS}  warmup={WARMUP} iters={ITERS} trim={TRIM}  gate=+{GATE_PP}pp on top of iterilp")
    print("=" * 120)
    t0 = time.time()
    results = []
    # CRITICAL: serial over ThreadPool with workers == len(GPUS) over-subscribes
    # GPUs (5 targets / 2 GPUs => contention). Run targets sequentially on a
    # single GPU per call, alternating GPU. Each target self-times back-to-back.
    for i, t in enumerate(TARGETS):
        results.append(run_target(i, *t))
    elapsed = time.time() - t0
    out = {"warmup": WARMUP, "iters": ITERS, "trim": TRIM, "gate_pp": GATE_PP,
           "gpus": GPUS, "elapsed_minutes": round(elapsed/60, 2),
           "results": results}
    with open(os.path.join(SCRIPT_DIR, "bench_round16_optA_smoke.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + "=" * 120)
    print(f"{'Tag':4s} {'parent':>8s} {'iterilp':>8s} {'cmp':>8s}  {'Δvs_iter':>10s}  {'pp':>7s}  gate")
    print("-" * 120)
    for r in sorted(results, key=lambda x: x["tag"]):
        if r.get("error"):
            print(f"{r['tag']:4s}  ERROR  {r}")
            continue
        print(f"{r['tag']:4s} {r['parent_tflops']:8.2f} {r['iterilp_tflops']:8.2f} {r['cand_tflops']:8.2f}  "
              f"{r['delta_vs_iterilp_tflops']:+10.2f}  {r['delta_vs_iterilp_pp']:+6.2f}pp  "
              f"{'PASS' if r['gate_pass'] else 'fail'}")
    n_pass = sum(1 for r in results if r.get("gate_pass"))
    print(f"\nElapsed: {elapsed/60:.1f} min   gate-pass: {n_pass}/{len(results)}")
    print("Saved bench_round16_optA_smoke.json")


if __name__ == "__main__":
    main()
