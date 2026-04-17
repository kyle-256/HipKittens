#!/usr/bin/env python3
"""R19B smoke bench. Same protocol as R18A:
warmup=200 iters=500 trim=10%, GPUs 4 and 5 only.

Reads SNR probe results (snr_probe_r19b.json), benches every variant marked
OK-FOR-BENCH plus the parent baseline for comparison.
"""
import json, math, os, subprocess, sys, time, sysconfig
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10

# (lab, M, N, K, comp, parent_suffix)
SHAPE_INFO = {
    "DLA1": (4096,   32768, 128256, 5781.1, "_ts_pf6_6_v12_memc"),
    "DLA2": (128256, 32768,   4096, 4400.6, "_ts_gm2_v12_memc_dc"),  # TODO: confirm comp from bench_all_42
    "DLA7": (28672,  32768,   4096, 5247.5, "_ts_lgk2_v12_memc"),    # TODO: confirm
}

GPU_POOL = [4, 5]


def bench_one(M, N, K, suffix, gpu_id):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return {"err": f"missing {so_path}"}
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
                           capture_output=True, text=True, timeout=900, env=env)
        if r.returncode != 0:
            return {"err": r.stderr[-300:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"err": str(e)}


def task(args):
    lab, M, N, K, suf, gpu, label = args
    r = bench_one(M, N, K, suf, gpu)
    return label, lab, suf, gpu, r


def main():
    t0 = time.time()
    print(f"R19 OptB smoke bench. warmup={WARMUP} iters={ITERS} trim={TRIM}.")
    print("=" * 100)

    # Read SNR probe results
    snr_path = os.path.join(SCRIPT_DIR, "snr_probe_r19b.json")
    if not os.path.exists(snr_path):
        print(f"FATAL: {snr_path} not found"); return 1
    snr = json.load(open(snr_path))
    ok = snr.get("ok_pairs_aperture", [])
    if not ok:
        print("No SNR-OK + aperture-OK variants. Nothing to bench. Done.")
        return 0
    print(f"SNR/aperture OK pairs to bench: {len(ok)}")
    for p in ok: print(f"  {p}")

    # Build job list: parent (one per shape) + each ok variant
    jobs = []
    i = 0
    shapes_seen = set()
    for (lab, var_suffix) in ok:
        if lab not in shapes_seen:
            M, N, K, comp, ps = SHAPE_INFO[lab]
            gpu = GPU_POOL[i % len(GPU_POOL)]
            jobs.append((lab, M, N, K, ps, gpu, "parent"))
            i += 1
            shapes_seen.add(lab)
        M, N, K, comp, ps = SHAPE_INFO[lab]
        gpu = GPU_POOL[i % len(GPU_POOL)]
        jobs.append((lab, M, N, K, ps + var_suffix, gpu, var_suffix))
        i += 1

    print(f"\nTotal bench jobs: {len(jobs)}")
    results = {}
    with ProcessPoolExecutor(max_workers=len(GPU_POOL)) as ex:
        futs = {ex.submit(task, j): j for j in jobs}
        for fut in as_completed(futs):
            label, lab, suf, gpu, r = fut.result()
            results.setdefault(lab, {})[label] = {"suffix": suf, "gpu": gpu, "result": r}
            print(f"  {lab:6s} {label:25s} gpu={gpu} -> {r}", flush=True)

    print("\n" + "=" * 100)
    print("Summary (TFLOPS / Δpp vs parent):")
    out = {"warmup": WARMUP, "iters": ITERS, "trim": TRIM, "shapes": {}}
    for lab in sorted(results.keys()):
        sd = results[lab]
        M, N, K, comp, ps = SHAPE_INFO[lab]
        parent = sd.get("parent", {}).get("result", {})
        p_t = parent.get("tflops", 0.0)
        print(f"\n  {lab} (comp~{comp}):")
        print(f"    parent              {p_t:7.2f}  ({p_t/comp*100:5.2f}%)  baseline")
        out["shapes"][lab] = {"comp": comp, "parent_tflops": p_t, "variants": {}}
        for label, v in sd.items():
            if label == "parent": continue
            t = v["result"].get("tflops", 0.0)
            delta_pp = (t - p_t) / comp * 100 if p_t else 0
            ratio = t / comp * 100
            mark = ""
            if delta_pp >= 1.0: mark = " <-- VERIFY (>=+1pp)"
            elif delta_pp >= 0.5: mark = " <-- VERIFY (>=+0.5pp)"
            print(f"    {label:25s}{t:7.2f}  ({ratio:5.2f}%)  Δ{delta_pp:+.2f}pp{mark}")
            out["shapes"][lab]["variants"][label] = {"tflops": t, "delta_pp": round(delta_pp, 2)}
    with open(os.path.join(SCRIPT_DIR, "bench_round19_optB_smoke.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nElapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    sys.exit(main() or 0)
