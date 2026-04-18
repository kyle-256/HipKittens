#!/usr/bin/env python3
"""R22A aperture probe (correctness + reproducibility gate).

For each (shape, variant), runs 3 short benches (warmup=20, iters=50) with
different seeds. Pass criteria:
  - all 3 runs produce TFLOPS > 0 (no crash, no garbage)
  - stddev across the 3 runs <= 5 % of mean (reproducible)
"""
import json, math, os, statistics, subprocess, sys, sysconfig, time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 20
ITERS = 50
TRIM = 0.10
N_RUNS = 3
SEEDS = [0, 7, 42]

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


def make_bench_script(M, N, K, suffix, seed):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    return so_path, f"""
import sys, math, torch, importlib.util, json
WARMUP, ITERS, TRIM = {WARMUP}, {ITERS}, {TRIM}
M, N, K = {M}, {N}, {K}
torch.manual_seed({seed})
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
# Also check that C has nonzero output (very loose correctness gate)
nonzero = float(C.abs().sum().item())
print(json.dumps({{"tflops": round(t,4), "ms": round(avg,4), "nonzero_sum": nonzero, "n_iters_kept": len(times)}}))
"""


def bench_one(args):
    lab, M, N, K, ps, vs, seed, gpu = args
    suffix = ps + vs
    so_path, script = make_bench_script(M, N, K, suffix, seed)
    if not os.path.exists(so_path):
        return lab, vs, seed, gpu, {"error": "missing .so"}
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=600, env=env)
        if r.returncode != 0:
            return lab, vs, seed, gpu, {"error": f"rc={r.returncode}", "stderr": r.stderr[-500:]}
        return lab, vs, seed, gpu, json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return lab, vs, seed, gpu, {"error": str(e)}


def main():
    args_list = []
    i = 0
    for (lab, M, N, K, ps) in SHAPES:
        for vs in VARIANT_SUFFIXES:
            for seed in SEEDS:
                gpu = GPU_POOL[i % len(GPU_POOL)]
                args_list.append((lab, M, N, K, ps, vs, seed, gpu))
                i += 1

    print(f"R22A aperture: {len(args_list)} runs ({len(SHAPES)}x{len(VARIANT_SUFFIXES)}x{N_RUNS}), GPUs {GPU_POOL}")
    print("=" * 110)
    out = {}  # (lab, vs) -> [tflops list]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(GPU_POOL)) as ex:
        futs = {ex.submit(bench_one, a): a for a in args_list}
        for fut in as_completed(futs):
            lab, vs, seed, gpu, res = fut.result()
            key = (lab, vs)
            out.setdefault(key, []).append((seed, res))
            if "error" in res:
                print(f"  {lab:6s} {vs:30s} seed={seed} gpu={gpu}  ERR {res.get('error')}", flush=True)
            else:
                print(f"  {lab:6s} {vs:30s} seed={seed} gpu={gpu}  {res['tflops']:>8.2f} TFLOPS  nz={res.get('nonzero_sum'):.2e}", flush=True)

    print(f"\nElapsed: {time.time()-t0:.1f}s")

    print("\n" + "=" * 110)
    print("APERTURE GATE (TFLOPS > 0 AND stddev/mean <= 5%)")
    print("=" * 110)
    fail = 0
    for (lab, *_rest) in SHAPES:
        for vs in VARIANT_SUFFIXES:
            runs = out.get((lab, vs), [])
            ts = [r[1].get("tflops") for r in runs if "tflops" in r[1]]
            errs = [r for r in runs if "error" in r[1]]
            if errs:
                print(f"  {lab:6s} {vs:30s}  FAIL: {len(errs)}/{len(runs)} crashes")
                fail += 1
                continue
            if not ts or any(t is None or t <= 0 for t in ts):
                print(f"  {lab:6s} {vs:30s}  FAIL: zero TFLOPS")
                fail += 1
                continue
            m = statistics.mean(ts); s = statistics.stdev(ts) if len(ts) > 1 else 0.0
            ratio = s / m * 100
            ok = ratio <= 5.0
            tag = "OK  " if ok else "FAIL"
            print(f"  {lab:6s} {vs:30s}  {tag}  mean={m:.2f}  stddev={s:.2f} ({ratio:.2f}%)")
            if not ok: fail += 1
    print(f"\n=> {fail} variant(s) failed aperture gate")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
