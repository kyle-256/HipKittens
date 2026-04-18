#!/usr/bin/env python3
"""R25-F EXTEND2 verify: pfoff ∈ {14, 15, 16} for gm ∈ {6, 7} — 3 reps each.
Plus REF.  Find absolute peak.
"""
import json, math, os, subprocess, sys, sysconfig, time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP, ITERS, TRIM, N_REPS = 200, 500, 0.10, 3

SHAPES = [
    ("DLA2", 128256, 32768, 4096, 1),
    ("DLA7",  28672, 32768, 4096, 2),
]

GM_VALUES = [6, 7]
PFOFF_VALUES = [14, 15, 16]


def variants(lab):
    out = [("R25D_gm6_pfoff4_REF", f"_r25d_gm6_pfoff4_{lab.lower()}")]
    for gm in GM_VALUES:
        for pf in PFOFF_VALUES:
            out.append((f"gm{gm}_pfoff{pf}", f"_r25f_gm{gm}_pfoff{pf}_{lab.lower()}"))
    return out


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
run(); torch.cuda.synchronize()
nz = (C != 0).float().mean().item()
if nz < 0.5:
    print(json.dumps({{"error": "C-coverage too low", "nz": nz}})); sys.exit(0)
times=[]
for _ in range(ITERS):
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record(); run(); e.record(); torch.cuda.synchronize()
    times.append(s.elapsed_time(e))
times.sort(); trim=int(len(times)*TRIM)
times=times[trim:-trim] if trim>0 else times
avg=sum(times)/len(times); t=2.0*M*N*K/(avg*1e-3)/1e12
print(json.dumps({{"tflops": round(t,2), "ms": round(avg,4), "n_iters_kept": len(times), "nz": nz}}))
"""


def bench_one(args):
    lab, M, N, K, vlabel, vsuffix, gpu, rep = args
    so_path, script = make_bench_script(M, N, K, vsuffix)
    if not os.path.exists(so_path):
        return lab, vlabel, rep, gpu, {"error": "missing .so", "so": so_path}
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=1800, env=env)
        if r.returncode != 0:
            return lab, vlabel, rep, gpu, {"error": f"rc={r.returncode}", "stderr": r.stderr[-500:]}
        return lab, vlabel, rep, gpu, json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return lab, vlabel, rep, gpu, {"error": str(e)}


def run_shape_serial(shape):
    lab, M, N, K, gpu = shape
    print(f"\n[{lab}] starting on GPU {gpu}", flush=True)
    rows = []
    for (vlabel, vsuffix) in variants(lab):
        for rep in range(N_REPS):
            res = bench_one((lab, M, N, K, vlabel, vsuffix, gpu, rep))
            _l, _v, _r, _g, r = res
            tag = f"{lab} {vlabel} rep{rep} gpu{gpu}"
            if "error" in r:
                print(f"  {tag:55s}  ERR {r.get('error')}", flush=True)
            else:
                print(f"  {tag:55s}  {r['tflops']:>8.2f} TFLOPS", flush=True)
            rows.append((vlabel, vsuffix, rep, r))
    return lab, rows


def stats(xs):
    if not xs: return None, None, None
    n = len(xs); mn = sum(xs)/n
    var = sum((x-mn)**2 for x in xs)/n
    return mn, sorted(xs)[n//2], var**0.5


def main():
    print(f"R25-F EXTEND2 VERIFY: 7 variants × {N_REPS} reps × {len(SHAPES)} shapes")
    print(f"warmup={WARMUP} iters={ITERS} trim={TRIM}")
    out = {"shapes": {}, "warmup": WARMUP, "iters": ITERS, "trim": TRIM, "n_reps": N_REPS}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(SHAPES)) as ex:
        futs = {ex.submit(run_shape_serial, s): s for s in SHAPES}
        for fut in as_completed(futs):
            lab, rows = fut.result()
            out["shapes"][lab] = {}
            for (vl, vs, rep, r) in rows:
                out["shapes"][lab].setdefault(vl, {"suffix": vs, "runs": []})
                out["shapes"][lab][vl]["runs"].append(r)
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    with open(os.path.join(SCRIPT_DIR, "bench_round25_optF_extend2.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Summary
    print("\n" + "=" * 110)
    print(f"FINAL VERIFY SUMMARY (n={N_REPS}, std≤30 STABLE gate, Δ vs R25-D REF)")
    print("=" * 110)
    for lab, sd in out["shapes"].items():
        ref_runs = [r.get("tflops") for r in sd.get("R25D_gm6_pfoff4_REF", {}).get("runs", []) if r.get("tflops")]
        ref_mean, ref_med, ref_std = stats(ref_runs)
        print(f"\n[{lab}] REF runs={[round(x,1) for x in ref_runs]}  mean={ref_mean:.2f}  std={ref_std:.2f}")
        rows = []
        for vl, vd in sd.items():
            if vl == "R25D_gm6_pfoff4_REF": continue
            runs = [r.get("tflops") for r in vd["runs"] if r.get("tflops")]
            errs = [r.get("error") for r in vd["runs"] if r.get("error")]
            m, med, sd_ = stats(runs)
            if m is None:
                print(f"  {vl:25s} FAIL errs={errs}")
                continue
            d_pct = (m - ref_mean)/ref_mean*100
            gate = "STABLE" if sd_ <= 30 else "UNSTABLE"
            print(f"  {vl:25s} runs={[round(x,1) for x in runs]}  mean={m:.2f}  std={sd_:.2f}  Δ={d_pct:+.2f}%  [{gate}]")
            rows.append((d_pct, vl, m, sd_, gate, errs))
        rows.sort(reverse=True)
        for (d, vk, m, sd_, g, e) in rows:
            if g == "STABLE":
                print(f"  >>> BEST STABLE [{lab}]: {vk}   mean={m:.2f}  std={sd_:.2f}  Δ={d:+.2f}%")
                break


if __name__ == "__main__":
    main()
