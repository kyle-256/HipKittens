#!/usr/bin/env python3
"""R25-G VERIFY: 3-rep tight bench on top candidates per shape + parent REFs.

Top candidates from smoke (Δ% vs parent):
  SA: gm7_pfoff120 (+19.7%), gm7_pfoff126 (+18.6%)        [K_iters=128]
  SB: gm7_pfoff48 (+17.4%), gm7_pfoff54 (+17.3%)          [K_iters=56]
  SC: gm7_pfoff120, gm7_pfoff124                          [K_iters=128] (REF rc=-6 in smoke; re-bench)
  SD: gm7_pfoff104, gm7_pfoff110                          [K_iters=112] (REF was cold 3631; re-bench)
  SE: gm7_pfoff56 (+17.8%), gm7_pfoff62 (+17.1%)          [K_iters=64]
  SF: gm7_pfoff56 (+13.8%), gm7_pfoff60 (+6.9%)           [K_iters=64]

Always include PARENT_REF so we can compare in the same run.
"""
import json, os, subprocess, sys, sysconfig, time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10
N_REPS = 3

# (label, M, N, K, parent_variant, candidate_pfoffs, gpu)
SHAPES = [
    ("SA", 4096, 28672, 32768, "ts_v12_tv0_memc_btw_all", [120, 126],     3),
    ("SB", 4096, 32768, 14336, "ts_v12_tv0_memc_btw_all", [48, 54],       3),
    ("SC", 14336, 4096, 32768, "ts_lgk2_memc_btw_all",    [120, 124],     4),
    ("SD", 16384, 4096, 28672, "ts_lgk2_memc_btw_all",    [104, 110],     4),
    ("SE", 28672, 4096, 16384, "ts_lgk2_memc_btw_all",    [56, 62],       7),
    ("SF",  4096, 14336, 16384, "ts_lgk2_memc_btw_all",   [56, 60],       7),
]

GM = 7


def variants_for(lab, parent, pfoffs, K, M, N):
    out = [("PARENT_REF", f"_{parent}")]
    for pf in pfoffs:
        out.append((f"gm7_pfoff{pf}", f"_r25g_gm{GM}_pfoff{pf}_K{K}_M{M}_N{N}"))
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
    lab, M, N, K, parent, pfoffs, gpu = shape
    print(f"\n[{lab}] starting on GPU {gpu} (M={M},N={N},K={K})", flush=True)
    rows = []
    for (vlabel, vsuffix) in variants_for(lab, parent, pfoffs, K, M, N):
        for rep in range(N_REPS):
            args = (lab, M, N, K, vlabel, vsuffix, gpu, rep)
            res = bench_one(args)
            _lab, _vl, _rep, _gpu, r = res
            tag = f"{lab} {vlabel} rep{rep} gpu{gpu}"
            if "error" in r:
                print(f"  {tag:55s}  ERR {r.get('error')}", flush=True)
            else:
                print(f"  {tag:55s}  {r['tflops']:>8.2f} TFLOPS (ms={r.get('ms')})", flush=True)
            rows.append((vlabel, vsuffix, rep, r))
    return lab, rows


def run_gpu_group(shapes_on_gpu):
    results = []
    for s in shapes_on_gpu:
        results.append(run_shape_serial(s))
    return results


def main():
    print(f"R25-G VERIFY: 6 shapes × 3 variants × {N_REPS} reps")
    print(f"warmup={WARMUP} iters={ITERS} trim={TRIM}")
    print("=" * 110)
    out = {"shapes": {}, "warmup": WARMUP, "iters": ITERS, "trim": TRIM, "n_reps": N_REPS}
    t0 = time.time()
    by_gpu = {}
    for s in SHAPES:
        by_gpu.setdefault(s[6], []).append(s)
    print(f"  GPU groups: {[(g, [s[0] for s in lst]) for g, lst in by_gpu.items()]}")

    with ProcessPoolExecutor(max_workers=len(by_gpu)) as ex:
        futs = {ex.submit(run_gpu_group, lst): g for g, lst in by_gpu.items()}
        for fut in as_completed(futs):
            for (lab, rows) in fut.result():
                out["shapes"][lab] = {}
                for (vlabel, vsuffix, rep, r) in rows:
                    out["shapes"][lab].setdefault(vlabel, {"suffix": vsuffix, "runs": []})
                    out["shapes"][lab][vlabel]["runs"].append(r)
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    with open(os.path.join(SCRIPT_DIR, "bench_round25_optG_verify.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Per-shape summary with mean/std/Δ%
    print("\n" + "=" * 110)
    print(f"{'Shape':5s} {'Variant':22s} {'Mean':>10s} {'Std':>8s} {'Δ% vs PARENT':>14s}")
    print("=" * 110)
    for shape in SHAPES:
        lab, M, N, K, parent, pfoffs, gpu = shape
        sd = out["shapes"].get(lab, {})
        ref = sd.get("PARENT_REF", {})
        ref_runs = [r.get("tflops") for r in ref.get("runs", []) if r.get("tflops")]
        ref_mean = sum(ref_runs)/len(ref_runs) if ref_runs else None
        ref_std = (sum((x-ref_mean)**2 for x in ref_runs)/len(ref_runs))**0.5 if ref_runs and ref_mean else 0
        if ref_mean:
            print(f"{lab:5s} {'PARENT_REF':22s} {ref_mean:>10.2f} {ref_std:>8.2f} {'—':>14s}")
        else:
            print(f"{lab:5s} {'PARENT_REF':22s} {'NONE':>10s} {'—':>8s} {'—':>14s}")
        for pf in pfoffs:
            v = sd.get(f"gm7_pfoff{pf}", {})
            xs = [r.get("tflops") for r in v.get("runs", []) if r.get("tflops")]
            if xs:
                m = sum(xs)/len(xs)
                s = (sum((x-m)**2 for x in xs)/len(xs))**0.5
                d = (m-ref_mean)/ref_mean*100 if ref_mean else float('nan')
                tag = "FAIL_STD" if s > 30 else "OK"
                print(f"{lab:5s} {'gm7_pfoff'+str(pf):22s} {m:>10.2f} {s:>8.2f} {d:>13.2f}%  [{tag}]")
            else:
                print(f"{lab:5s} {'gm7_pfoff'+str(pf):22s} {'ERR':>10s}")
        print()


if __name__ == "__main__":
    main()
