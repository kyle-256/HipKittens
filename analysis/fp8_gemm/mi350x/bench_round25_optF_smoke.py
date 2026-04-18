#!/usr/bin/env python3
"""R25-F SMOKE bench: 1 rep per (gm × pfoff) variant, plus R25-D gm6_pfoff4 baseline.

Goal: identify top 3-5 candidates per shape that beat R25-D baseline by ≥+0.5%.

GPUs 1 (DLA2), 2 (DLA7) — coexist with R25-D verify (GPUs 0,4,5,6,7) and R25-E (3).
warmup=200, iters=500, trim=10% per project rules.

Each shape runs serially on its dedicated GPU. The two shapes run in parallel
(2 worker processes, 1 per GPU).
"""
import json, math, os, subprocess, sys, sysconfig, time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10
N_REPS = 1  # smoke

# (label, M, N, K, gpu)
SHAPES = [
    ("DLA2", 128256, 32768, 4096, 1),
    ("DLA7",  28672, 32768, 4096, 2),
]

GM_VALUES = [5, 6, 7, 8]
PFOFF_VALUES = [3, 4, 5, 6, 7, 8]

# Build (variant_label, suffix) list per shape; include the R25-D gm6_pfoff4 reference.
def variants_for(lab):
    out = [("R25D_gm6_pfoff4_REF", f"_r25d_gm6_pfoff4_{lab.lower()}")]
    for gm in GM_VALUES:
        for pf in PFOFF_VALUES:
            label = f"gm{gm}_pfoff{pf}"
            suffix = f"_r25f_gm{gm}_pfoff{pf}_{lab.lower()}"
            out.append((label, suffix))
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
    for (vlabel, vsuffix) in variants_for(lab):
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


def main():
    print(f"R25-F SMOKE: 24 (gm × pfoff) variants + 1 ref per shape × {len(SHAPES)} shapes × {N_REPS} reps")
    print(f"warmup={WARMUP} iters={ITERS} trim={TRIM}  GPUs={[s[4] for s in SHAPES]}")
    print("=" * 110)
    out = {"shapes": {}, "warmup": WARMUP, "iters": ITERS, "trim": TRIM, "n_reps": N_REPS}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(SHAPES)) as ex:
        futs = {ex.submit(run_shape_serial, s): s for s in SHAPES}
        for fut in as_completed(futs):
            lab, rows = fut.result()
            out["shapes"][lab] = {}
            for (vlabel, vsuffix, rep, r) in rows:
                out["shapes"][lab].setdefault(vlabel, {"suffix": vsuffix, "runs": []})
                out["shapes"][lab][vlabel]["runs"].append(r)
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    with open(os.path.join(SCRIPT_DIR, "bench_round25_optF_smoke.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Sweep grid + ranking per shape
    for (lab, *_) in SHAPES:
        print(f"\n{'='*110}\n[{lab}] sweep grid (TFLOPS, 1 rep each)\n{'='*110}")
        ref = out["shapes"].get(lab, {}).get("R25D_gm6_pfoff4_REF", {})
        ref_runs = [r.get("tflops") for r in ref.get("runs", []) if r.get("tflops")]
        ref_tf = ref_runs[0] if ref_runs else None
        print(f"  R25D_gm6_pfoff4 REF: {ref_tf}")
        # grid: rows=gm, cols=pfoff
        header = "  gm\\pf  " + "  ".join(f"pf{pf:>2d}" for pf in PFOFF_VALUES)
        print(header)
        for gm in GM_VALUES:
            row_cells = [f"  gm{gm}    "]
            for pf in PFOFF_VALUES:
                v = out["shapes"][lab].get(f"gm{gm}_pfoff{pf}", {})
                xs = [r.get("tflops") for r in v.get("runs", []) if r.get("tflops")]
                if xs:
                    row_cells.append(f"{xs[0]:>6.1f}")
                else:
                    row_cells.append("  ERR ")
            print(" ".join(row_cells))

        # Δ% vs REF table
        if ref_tf:
            print(f"\n[{lab}] Δ% vs R25D_gm6_pfoff4 REF ({ref_tf:.1f})")
            print(header)
            for gm in GM_VALUES:
                row_cells = [f"  gm{gm}    "]
                for pf in PFOFF_VALUES:
                    v = out["shapes"][lab].get(f"gm{gm}_pfoff{pf}", {})
                    xs = [r.get("tflops") for r in v.get("runs", []) if r.get("tflops")]
                    if xs:
                        d = (xs[0] - ref_tf) / ref_tf * 100
                        row_cells.append(f"{d:+6.2f}")
                    else:
                        row_cells.append("  ERR ")
                print(" ".join(row_cells))

            # Top 5 candidates beating REF by ≥+0.5%
            cands = []
            for gm in GM_VALUES:
                for pf in PFOFF_VALUES:
                    v = out["shapes"][lab].get(f"gm{gm}_pfoff{pf}", {})
                    xs = [r.get("tflops") for r in v.get("runs", []) if r.get("tflops")]
                    if xs:
                        d = (xs[0] - ref_tf) / ref_tf * 100
                        cands.append((d, gm, pf, xs[0]))
            cands.sort(reverse=True)
            print(f"\n[{lab}] Top 5 candidates (sorted by Δ vs REF):")
            for (d, gm, pf, tf) in cands[:5]:
                marker = " ★" if d >= 0.5 else ""
                print(f"  gm{gm}_pfoff{pf}: {tf:.1f} TFLOPS  ({d:+.2f}%){marker}")


if __name__ == "__main__":
    main()
