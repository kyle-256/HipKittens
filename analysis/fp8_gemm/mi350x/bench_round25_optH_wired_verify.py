#!/usr/bin/env python3
"""R25-H WIRED VERIFY: 5-rep verify of the 4 wired R25-H variants vs parent.
warmup=200 iters=500 trim=10%. Each .so is the same bits as the wired bench_all_42
variant (verified by identical compile flags).

GPU layout (idle: 2, 3, 7):
  GPU 2: SH1 (16384x28672x2048)
  GPU 3: SH2 (4096x32768x6144)
  GPU 7: SH4 (32768x4096x7168) then SH6 (4096x14336x8192) — serialized
"""
import json, math, os, subprocess, statistics, sys, sysconfig, time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10
N_REPS = 5

# (label, M, N, K, parent_suffix, child_suffix, gpu)
SHAPES = [
    ("SH1", 16384, 28672, 2048,
     "_ts_gm2_v12_memc_btw_all",
     "_ts_v12_gm7_memc_pfoff4_kx2048_btw_all", 2),
    ("SH2",  4096, 32768, 6144,
     "_ts_gm2_v12_memc_btw_all",
     "_ts_v12_gm7_memc_pfoff19_kx6144_btw_all", 3),
    ("SH4", 32768,  4096, 7168,
     "_ts_v12_tv0_memc_btw_all",
     "_ts_v12_tv0_gm7_memc_pfoff24_kx7168_btw_all", 7),
    ("SH6",  4096, 14336, 8192,
     "_ts_v12_tv0_memc_btw_all",
     "_ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all", 7),
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
    lab, M, N, K, parent, child, gpu = shape
    print(f"\n[{lab}] starting on GPU {gpu} (M={M},N={N},K={K})", flush=True)
    rows = []
    variants = [("PARENT", parent), ("CHILD_R25H", child)]
    for (vlabel, vsuffix) in variants:
        for rep in range(N_REPS):
            args = (lab, M, N, K, vlabel, vsuffix, gpu, rep)
            res = bench_one(args)
            _l, _v, _r, _g, r = res
            tag = f"{lab} {vlabel} rep{rep} gpu{gpu}"
            if "error" in r:
                print(f"  {tag:55s}  ERR {r.get('error')}", flush=True)
            else:
                print(f"  {tag:55s}  {r['tflops']:>8.2f} TFLOPS (ms={r.get('ms')})", flush=True)
            rows.append((vlabel, vsuffix, rep, r))
    return lab, rows


def run_gpu_serial(shapes_for_gpu):
    """Serialize multiple shapes that share a GPU."""
    out = []
    for shape in shapes_for_gpu:
        out.append(run_shape_serial(shape))
    return out


def main():
    print(f"R25-H WIRED VERIFY: 4 shapes × 2 variants × {N_REPS} reps")
    print(f"warmup={WARMUP} iters={ITERS} trim={TRIM}")
    print("=" * 110)

    # Group shapes by GPU
    by_gpu = {}
    for s in SHAPES:
        by_gpu.setdefault(s[6], []).append(s)
    groups = list(by_gpu.values())

    out = {"shapes": {}, "warmup": WARMUP, "iters": ITERS, "trim": TRIM, "n_reps": N_REPS}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(groups)) as ex:
        futs = [ex.submit(run_gpu_serial, g) for g in groups]
        for fut in as_completed(futs):
            for (lab, rows) in fut.result():
                out["shapes"][lab] = {}
                for (vlabel, vsuffix, rep, r) in rows:
                    out["shapes"][lab].setdefault(vlabel, {"suffix": vsuffix, "runs": []})
                    out["shapes"][lab][vlabel]["runs"].append(r)
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    with open(os.path.join(SCRIPT_DIR, "bench_round25_optH_wired_verify.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 110)
    print("PER-SHAPE 5-REP VERIFY (warmup=200 iters=500 trim=10%):")
    print("=" * 110)
    all_pass = True
    for shape in SHAPES:
        lab, M, N, K, parent, child, gpu = shape
        ref_xs = [r.get("tflops") for r in out["shapes"][lab].get("PARENT", {}).get("runs", []) if r.get("tflops")]
        ch_xs = [r.get("tflops") for r in out["shapes"][lab].get("CHILD_R25H", {}).get("runs", []) if r.get("tflops")]
        if not ref_xs or not ch_xs:
            print(f"\n{lab}: MISSING DATA  parent={ref_xs}  child={ch_xs}")
            all_pass = False
            continue
        ref_m = statistics.mean(ref_xs); ref_s = statistics.stdev(ref_xs) if len(ref_xs) >= 2 else 0
        ch_m = statistics.mean(ch_xs); ch_s = statistics.stdev(ch_xs) if len(ch_xs) >= 2 else 0
        d = (ch_m - ref_m) / ref_m * 100
        ok = "PASS" if d >= 5.0 else "FAIL"
        if d < 5.0:
            all_pass = False
        print(f"\n{lab} {M}x{N}x{K} (K_iters={K//256})  GPU{gpu}")
        print(f"  PARENT  : mean={ref_m:7.2f} std={ref_s:5.2f} ({len(ref_xs)}/{N_REPS}) suffix={parent}")
        print(f"  CHILD   : mean={ch_m:7.2f} std={ch_s:5.2f} ({len(ch_xs)}/{N_REPS}) suffix={child}")
        print(f"  Δ       : {d:+.2f}%   GATE(>=+5.0%) = {ok}")
    print("\n" + "=" * 110)
    print(f"OVERALL: {'ALL 4 PASS' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
