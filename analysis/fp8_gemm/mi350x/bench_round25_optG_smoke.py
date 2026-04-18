#!/usr/bin/env python3
"""R25-G SMOKE bench: 1 rep per (shape × pfoff) variant + per-shape parent REF
+ gm7_pfoff0 diagnostic. warmup=200 iters=500 trim=10%.

Six shapes, 4 pfoff values + 1 gm7_pfoff0 diag + 1 parent REF = 6 variants per
shape × 6 shapes = 36 benches. Distributed across 3 GPUs (3,4,7) for ~3x
parallelism.
"""
import json, os, subprocess, sys, sysconfig, time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10
N_REPS = 1

# (label, M, N, K, parent_variant, pfoff_list, gpu)
SHAPES = [
    ("SA", 4096, 28672, 32768, "ts_v12_tv0_memc_btw_all", [120, 124, 126, 127], 3),
    ("SB", 4096, 32768, 14336, "ts_v12_tv0_memc_btw_all", [48, 52, 54, 55],     3),
    ("SC", 14336, 4096, 32768, "ts_lgk2_memc_btw_all",    [120, 124, 126, 127], 4),
    ("SD", 16384, 4096, 28672, "ts_lgk2_memc_btw_all",    [104, 108, 110, 111], 4),
    ("SE", 28672, 4096, 16384, "ts_lgk2_memc_btw_all",    [56, 60, 62, 63],     7),
    ("SF",  4096, 14336, 16384, "ts_lgk2_memc_btw_all",   [56, 60, 62, 63],     7),
]

GM = 7


def variants_for(lab, parent, pfoffs, K, M, N):
    """Return list of (variant_label, suffix). Includes parent REF + diag + pfoffs."""
    out = [("PARENT_REF", f"_{parent}")]
    out.append(("gm7_pfoff0_DIAG", f"_r25g_gm{GM}_pfoff0_K{K}_M{M}_N{N}"))
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
    print(f"R25-G SMOKE: 6 shapes × ~6 variants × {N_REPS} reps")
    print(f"warmup={WARMUP} iters={ITERS} trim={TRIM}")
    print(f"GPU layout: {[(s[0], s[6]) for s in SHAPES]}")
    print("=" * 110)
    out = {"shapes": {}, "warmup": WARMUP, "iters": ITERS, "trim": TRIM, "n_reps": N_REPS}
    t0 = time.time()
    # Group shapes by GPU and run sequentially per GPU; 3 GPU groups in parallel.
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
    with open(os.path.join(SCRIPT_DIR, "bench_round25_optG_smoke.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Per-shape ranking
    for shape in SHAPES:
        lab, M, N, K, parent, pfoffs, gpu = shape
        print(f"\n{'='*110}\n[{lab}] {M}x{N}x{K} parent={parent}\n{'='*110}")
        ref = out["shapes"].get(lab, {}).get("PARENT_REF", {})
        ref_runs = [r.get("tflops") for r in ref.get("runs", []) if r.get("tflops")]
        ref_tf = ref_runs[0] if ref_runs else None
        print(f"  PARENT_REF: {ref_tf} TFLOPS")
        diag = out["shapes"][lab].get("gm7_pfoff0_DIAG", {})
        diag_runs = [r.get("tflops") for r in diag.get("runs", []) if r.get("tflops")]
        diag_tf = diag_runs[0] if diag_runs else None
        if diag_tf and ref_tf:
            print(f"  gm7_pfoff0 (gm7 only, no R25C): {diag_tf} TFLOPS  ({(diag_tf-ref_tf)/ref_tf*100:+.2f}%)")
        cands = []
        for pf in pfoffs:
            v = out["shapes"][lab].get(f"gm7_pfoff{pf}", {})
            xs = [r.get("tflops") for r in v.get("runs", []) if r.get("tflops")]
            if xs and ref_tf:
                d = (xs[0] - ref_tf) / ref_tf * 100
                cands.append((d, pf, xs[0]))
        cands.sort(reverse=True)
        for (d, pf, tf) in cands:
            marker = " *" if d >= 0.5 else ""
            print(f"  gm7_pfoff{pf}: {tf:.1f} TFLOPS  ({d:+.2f}%){marker}")


if __name__ == "__main__":
    main()
