#!/usr/bin/env python3
"""R22A smoke bench (warmup=200 iters=500 trim=10%) for LDS_RD_STAGGER_NOP probes.

Per benchmark-rules.md MANDATORY: warmup=200, iters=500, 10% trim.
Subprocess-isolated to avoid symbol conflicts. Runs across GPUs 6-7
(per AGENT_PROMPT: 0-5 are reserved for the R22-rebench full 42-shape job).
"""
import json, math, os, subprocess, sys, sysconfig, time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10

# (label, M_native, N, K, parent_suffix)
SHAPES = [
    ("DLA2", 128256, 32768,  4096, "_ts_gm2_v12_memc_dc"),
    ("DLA7",  28672, 32768,  4096, "_ts_lgk2_v12_memc_btw_all"),
    ("DLA1",   4096, 32768, 128256, "_ts_pf6_6_v12_memc"),
]

VARIANT_SUFFIXES = [
    "_r22a_baseline",
    "_r22a_nop1",
    "_r22a_nop2",
]

GPU_POOL = [6, 7]


def make_bench_script(M, N, K, suffix):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    return so_path, f"""
import sys, math, torch, importlib.util, json
WARMUP, ITERS, TRIM = {WARMUP}, {ITERS}, {TRIM}
M, N, K = {M}, {N}, {K}
torch.manual_seed(0)
def gen_fp4(r, K):
    c = K // 2
    return (torch.randint(0, 16, (r, c), dtype=torch.uint8, device='cuda') << 4) | torch.randint(0, 16, (r, c), dtype=torch.uint8, device='cuda')
def preshuffle(se):
    r, kb = se.shape; pr = math.ceil(r/64)*64; pk = math.ceil(kb/8)*8
    raw = torch.full((pr, pk), 0x7F, dtype=torch.uint8, device=se.device)
    raw[:r,:kb] = (se.to(torch.int16)+127).to(torch.uint8)
    sh = raw.view(pr//32,2,16,pk//8,2,4,1).permute(0,3,5,2,4,1,6).contiguous().view(pr//32,pk*32)
    sh = sh.view(pr//64,2,pk*32//4,4).permute(0,2,1,3).contiguous()
    return sh.view(pr//64,pk*64)
spec = importlib.util.spec_from_file_location('{module_name}', '{so_path}')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
A=gen_fp4(M,K); B=gen_fp4(N,K)
sc_a=torch.randint(-2,3,(M,K//32),dtype=torch.int8,device='cuda')
sc_b=torch.randint(-2,3,(N,K//32),dtype=torch.int8,device='cuda')
A_sc=preshuffle(sc_a); B_sc=preshuffle(sc_b)
C=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
def run(): mod.gemm_rcr(A,B,A_sc,B_sc,C)
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
print(json.dumps({{"tflops": round(t,2), "ms": round(avg,4), "n_iters_kept": len(times)}}))
"""


def bench_one(args):
    lab, M, N, K, ps, vs, gpu = args
    suffix = ps + vs
    so_path, script = make_bench_script(M, N, K, suffix)
    if not os.path.exists(so_path):
        return lab, vs, gpu, {"error": "missing .so"}
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=900, env=env)
        if r.returncode != 0:
            return lab, vs, gpu, {"error": f"rc={r.returncode}", "stderr": r.stderr[-500:]}
        return lab, vs, gpu, json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return lab, vs, gpu, {"error": str(e)}


def main():
    args_list = []
    i = 0
    for (lab, M, N, K, ps) in SHAPES:
        for vs in VARIANT_SUFFIXES:
            gpu = GPU_POOL[i % len(GPU_POOL)]
            args_list.append((lab, M, N, K, ps, vs, gpu))
            i += 1

    print(f"R22A smoke: {len(args_list)} bench jobs ({len(SHAPES)} shapes x {len(VARIANT_SUFFIXES)} variants), GPUs {GPU_POOL}")
    print(f"warmup={WARMUP} iters={ITERS} trim={TRIM}")
    print("=" * 110)
    out = {"shapes": {lab: {} for (lab, *_rest) in SHAPES},
           "warmup": WARMUP, "iters": ITERS, "trim": TRIM}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(GPU_POOL)) as ex:
        futs = {ex.submit(bench_one, a): a for a in args_list}
        for fut in as_completed(futs):
            lab, vs, gpu, res = fut.result()
            out["shapes"][lab][vs] = res
            if "error" in res:
                print(f"  {lab:6s} {vs:30s} gpu={gpu}  ERR {res.get('error')}", flush=True)
            else:
                print(f"  {lab:6s} {vs:30s} gpu={gpu}  {res['tflops']:>8.2f} TFLOPS", flush=True)

    print(f"\nElapsed: {time.time()-t0:.1f}s")
    with open(os.path.join(SCRIPT_DIR, "bench_round22_optA_smoke.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Summary: deltas vs each shape's baseline
    print("\n" + "=" * 110)
    print("SUMMARY (Δ vs _r22a_baseline)")
    print("=" * 110)
    for (lab, *_rest) in SHAPES:
        b = out["shapes"][lab].get("_r22a_baseline", {})
        bt = b.get("tflops")
        if bt is None:
            print(f"  {lab}: BASELINE FAILED  err={b.get('error')}")
            continue
        print(f"  {lab} baseline = {bt:.2f} TFLOPS")
        for vs in VARIANT_SUFFIXES:
            if vs == "_r22a_baseline":
                continue
            v = out["shapes"][lab].get(vs, {})
            vt = v.get("tflops")
            if vt is None:
                print(f"    {vs:30s}  ERR {v.get('error')}")
                continue
            d = vt - bt
            d_pp = d / bt * 100
            print(f"    {vs:30s}  {vt:>8.2f}  Δ={d:+.2f} ({d_pp:+.2f}%)")


if __name__ == "__main__":
    main()
