#!/usr/bin/env python3
"""R23B smoke bench (warmup=200 iters=500 trim=10%) of 3 PERSISTENT_XCD variants.

Per benchmark-rules.md MANDATORY: warmup=200, iters=500, 10% trim.
Subprocess-isolated. Runs across GPUs 0-2 (one shape per GPU).
"""
import json, math, os, subprocess, sys, sysconfig, time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10

# (label, M_native, N, K, parent_suffix, gpu)
SHAPES = [
    ("DLA1", 4096,   32768, 128256, "_ts_pf6_6_v12_memc",      0),
    ("DLA2", 128256, 32768,   4096, "_ts_gm2_v12_memc_dc",     1),
    ("DLA7", 28672,  32768,   4096, "_ts_lgk2_v12_memc",       2),
]

VARIANT_SUFFIXES = [
    "_pxcd_baseline",
    "_pxcd_b1",
    "_pxcd_b4",
]


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
# Also dump C row sums to detect under-launched grids (raw correctness sniff).
nonzero = int((C != 0).sum().item())
print(json.dumps({{"tflops": round(t,2), "ms": round(avg,4), "n_iters_kept": len(times), "nonzero_C": nonzero, "C_total": M*N}}))
"""


def bench_one(args):
    lab, M, N, K, ps, vs, gpu = args
    suffix = ps + vs
    so_path, script = make_bench_script(M, N, K, suffix)
    if not os.path.exists(so_path):
        return lab, vs, gpu, {"error": "missing .so", "so_path": so_path}
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=1800, env=env)
        if r.returncode != 0:
            return lab, vs, gpu, {"error": f"rc={r.returncode}", "stderr": r.stderr[-500:]}
        return lab, vs, gpu, json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return lab, vs, gpu, {"error": str(e)}


def main():
    args_list = []
    for (lab, M, N, K, ps, gpu) in SHAPES:
        for vs in VARIANT_SUFFIXES:
            args_list.append((lab, M, N, K, ps, vs, gpu))

    print(f"R23B smoke: {len(args_list)} bench jobs (3 shapes x 3 variants), 1 GPU per shape")
    print(f"warmup={WARMUP} iters={ITERS} trim={TRIM}")
    print("=" * 110)
    out = {"shapes": {lab: {} for (lab, *_rest) in SHAPES},
           "warmup": WARMUP, "iters": ITERS, "trim": TRIM}
    t0 = time.time()
    # Use 3 workers (one per GPU) but serialize per-GPU by submitting in pairs
    # Actually just run all 9 with 3 workers; each shape's 3 variants will land on its assigned GPU
    # Parallelism: 3 (one shape per GPU concurrently) — but variants on same GPU serialize via lock
    # Simpler: run all sequentially on assigned GPU — 9 jobs * ~1-3 min each.
    # Use ProcessPoolExecutor=3, but the GPU mapping ensures no conflict per shape.
    # Each shape's 3 variants run on the same GPU — they'll be queued by Python.
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(bench_one, a): a for a in args_list}
        for fut in as_completed(futs):
            lab, vs, gpu, res = fut.result()
            out["shapes"][lab][vs] = res
            if "error" in res:
                print(f"  {lab:6s} {vs:30s} gpu={gpu}  ERR {res.get('error')}", flush=True)
            else:
                cov = res.get("nonzero_C", -1) / max(1, res.get("C_total", 1)) * 100
                print(f"  {lab:6s} {vs:30s} gpu={gpu}  {res['tflops']:>8.2f} TFLOPS  C-coverage={cov:.1f}%", flush=True)

    print(f"\nElapsed: {time.time()-t0:.1f}s")
    with open(os.path.join(SCRIPT_DIR, "bench_round23_optB.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Summary: deltas vs each shape's _pxcd_baseline
    print("\n" + "=" * 110)
    print("SUMMARY (delta vs _pxcd_baseline)")
    print("=" * 110)
    for (lab, *_rest) in SHAPES:
        b = out["shapes"][lab].get("_pxcd_baseline", {})
        bt = b.get("tflops")
        if bt is None:
            print(f"  {lab}: BASELINE FAILED")
            continue
        print(f"  {lab} baseline = {bt:.2f} TFLOPS")
        for vs in VARIANT_SUFFIXES:
            if vs == "_pxcd_baseline":
                continue
            v = out["shapes"][lab].get(vs, {})
            vt = v.get("tflops")
            if vt is None:
                print(f"    {vs:30s}  ERR")
                continue
            d = vt - bt
            d_pp = d / bt * 100
            print(f"    {vs:30s}  {vt:>8.2f}  delta={d:+.2f} ({d_pp:+.2f}%)")


if __name__ == "__main__":
    main()
