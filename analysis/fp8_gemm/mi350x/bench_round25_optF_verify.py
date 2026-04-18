#!/usr/bin/env python3
"""R25-F VERIFY: extended pfoff sweep smoke (1 rep) + 3-rep verify on top candidates.

Phase 1 (smoke): bench pfoff ∈ {9,10,12,14} × gm ∈ {5,6,7} (24 variants),
  combined with R25-D REF baseline.
Phase 2 (verify): 3-rep tight bench on top 5 candidates per shape (drawn from
  initial smoke + extended smoke), reject any with std > 30 TFLOPS or rc=-6.

GPUs 1 (DLA2), 2 (DLA7).
warmup=200, iters=500, trim=10%.
"""
import json, math, os, subprocess, sys, sysconfig, time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10

# (label, M, N, K, gpu)
SHAPES = [
    ("DLA2", 128256, 32768, 4096, 1),
    ("DLA7",  28672, 32768, 4096, 2),
]

# Phase 1 smoke list: REF + (gm, pfoff) extension
EXT_GM = [5, 6, 7]
EXT_PFOFF = [9, 10, 12, 14]

def smoke_variants(lab):
    out = [("R25D_gm6_pfoff4_REF", f"_r25d_gm6_pfoff4_{lab.lower()}")]
    for gm in EXT_GM:
        for pf in EXT_PFOFF:
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


def run_shape_serial(args):
    """args = (shape, var_list, n_reps, phase_label)"""
    shape, var_list, n_reps, phase = args
    lab, M, N, K, gpu = shape
    print(f"\n[{phase} {lab}] starting on GPU {gpu}, {len(var_list)} variants × {n_reps} reps", flush=True)
    rows = []
    for (vlabel, vsuffix) in var_list:
        for rep in range(n_reps):
            res = bench_one((lab, M, N, K, vlabel, vsuffix, gpu, rep))
            _lab, _vl, _rep, _gpu, r = res
            tag = f"{phase} {lab} {vlabel} rep{rep} gpu{gpu}"
            if "error" in r:
                print(f"  {tag:65s}  ERR {r.get('error')}", flush=True)
            else:
                print(f"  {tag:65s}  {r['tflops']:>8.2f} TFLOPS", flush=True)
            rows.append((vlabel, vsuffix, rep, r))
    return phase, lab, rows


def stats(xs):
    if not xs: return None, None, None
    xs = sorted(xs)
    n = len(xs)
    med = xs[n//2]
    mn = sum(xs)/n
    var = sum((x-mn)**2 for x in xs)/n
    std = var**0.5
    return mn, med, std


def main():
    out = {"shapes": {}, "warmup": WARMUP, "iters": ITERS, "trim": TRIM}
    t0 = time.time()

    # PHASE 1: smoke extended sweep (1 rep each, parallel by shape)
    print("=" * 110)
    print("PHASE 1 — SMOKE EXTENDED SWEEP (1 rep each)")
    print("=" * 110)
    phase1_args = [(s, smoke_variants(s[0]), 1, "P1-SMOKE") for s in SHAPES]
    with ProcessPoolExecutor(max_workers=len(SHAPES)) as ex:
        futs = {ex.submit(run_shape_serial, a): a for a in phase1_args}
        for fut in as_completed(futs):
            phase, lab, rows = fut.result()
            out["shapes"].setdefault(lab, {"phase1": {}, "phase2": {}})
            for (vlabel, vsuffix, rep, r) in rows:
                out["shapes"][lab]["phase1"].setdefault(vlabel, {"suffix": vsuffix, "runs": []})
                out["shapes"][lab]["phase1"][vlabel]["runs"].append(r)

    # Print extended grid
    for (lab, *_) in SHAPES:
        print(f"\n[{lab}] EXTENDED SWEEP grid (TFLOPS, 1 rep)")
        ref_xs = [r.get("tflops") for r in out["shapes"][lab]["phase1"].get("R25D_gm6_pfoff4_REF", {}).get("runs", []) if r.get("tflops")]
        ref_tf = ref_xs[0] if ref_xs else None
        print(f"  REF (R25D gm6_pfoff4) = {ref_tf}")
        header = "  gm\\pf  " + "  ".join(f"pf{pf:>2d}" for pf in EXT_PFOFF)
        print(header)
        for gm in EXT_GM:
            cells = [f"  gm{gm}    "]
            for pf in EXT_PFOFF:
                v = out["shapes"][lab]["phase1"].get(f"gm{gm}_pfoff{pf}", {})
                xs = [r.get("tflops") for r in v.get("runs", []) if r.get("tflops")]
                cells.append(f"{xs[0]:>6.1f}" if xs else "  ERR ")
            print(" ".join(cells))

    # PHASE 2: verify — pick top 5 candidates per shape across BOTH the original
    # smoke + the extended smoke. Original smoke values are read from the
    # bench_round25_optF_smoke.json file.
    orig_smoke = {}
    orig_path = os.path.join(SCRIPT_DIR, "bench_round25_optF_smoke.json")
    if os.path.exists(orig_path):
        with open(orig_path) as f:
            orig_smoke = json.load(f).get("shapes", {})

    top_per_shape = {}
    for (lab, *_) in SHAPES:
        cands = {}
        # From original smoke (gm5-8, pfoff3-8)
        for vk, vd in orig_smoke.get(lab, {}).items():
            if vk == "R25D_gm6_pfoff4_REF":
                continue
            xs = [r.get("tflops") for r in vd.get("runs", []) if r.get("tflops")]
            if xs:
                # Reconstruct suffix
                suffix = vd.get("suffix") or f"_r25f_{vk}_{lab.lower()}"
                cands[vk] = (xs[0], suffix)
        # From extended smoke
        for vk, vd in out["shapes"][lab]["phase1"].items():
            if vk == "R25D_gm6_pfoff4_REF":
                continue
            xs = [r.get("tflops") for r in vd.get("runs", []) if r.get("tflops")]
            if xs:
                cands[vk] = (xs[0], vd["suffix"])
        # Top 5 by smoke TFLOPS
        ranked = sorted(cands.items(), key=lambda kv: -kv[1][0])[:5]
        top_per_shape[lab] = ranked
        print(f"\n[{lab}] TOP 5 CANDIDATES for verify:")
        for k, (tf, sfx) in ranked:
            print(f"  {k}: {tf:.1f} TFLOPS  (suffix={sfx})")

    # Run verify (3 reps each, parallel by shape)
    print("\n" + "=" * 110)
    print("PHASE 2 — VERIFY (3 reps on top 5 candidates + REF)")
    print("=" * 110)
    phase2_args = []
    for s in SHAPES:
        lab = s[0]
        var_list = [("R25D_gm6_pfoff4_REF", f"_r25d_gm6_pfoff4_{lab.lower()}")]
        for k, (tf, sfx) in top_per_shape[lab]:
            var_list.append((k, sfx))
        phase2_args.append((s, var_list, 3, "P2-VERIFY"))
    with ProcessPoolExecutor(max_workers=len(SHAPES)) as ex:
        futs = {ex.submit(run_shape_serial, a): a for a in phase2_args}
        for fut in as_completed(futs):
            phase, lab, rows = fut.result()
            for (vlabel, vsuffix, rep, r) in rows:
                out["shapes"][lab]["phase2"].setdefault(vlabel, {"suffix": vsuffix, "runs": []})
                out["shapes"][lab]["phase2"][vlabel]["runs"].append(r)

    print(f"\nTotal Elapsed: {time.time()-t0:.1f}s")
    with open(os.path.join(SCRIPT_DIR, "bench_round25_optF_verify.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Final verify summary with stability gate
    print("\n" + "=" * 110)
    print("FINAL VERIFY SUMMARY (3-rep mean, std, Δ vs R25-D REF; stability gate std≤30 TFLOPS)")
    print("=" * 110)
    final = {}
    for (lab, *_) in SHAPES:
        ref_runs = [r.get("tflops") for r in out["shapes"][lab]["phase2"].get("R25D_gm6_pfoff4_REF", {}).get("runs", []) if r.get("tflops")]
        ref_mean, ref_med, ref_std = stats(ref_runs)
        print(f"\n[{lab}] REF (R25D gm6_pfoff4): runs={ref_runs}  mean={ref_mean}  std={ref_std:.1f}" if ref_mean else f"\n[{lab}] REF FAILED")
        final[lab] = {"ref": (ref_mean, ref_std), "candidates": []}
        for (vlabel, vsuffix) in [(k, t[1]) for k, t in top_per_shape[lab]]:
            xs_all = out["shapes"][lab]["phase2"].get(vlabel, {}).get("runs", [])
            xs = [r.get("tflops") for r in xs_all if r.get("tflops")]
            errs = [r.get("error") for r in xs_all if r.get("error")]
            mean, med, std = stats(xs)
            stab = "STABLE" if (std is not None and std <= 30) else ("UNSTABLE" if std is not None else "FAIL")
            err_note = f" errs={errs}" if errs else ""
            d_pct = (mean - ref_mean)/ref_mean*100 if (mean and ref_mean) else None
            d_str = f"{d_pct:+.2f}%" if d_pct is not None else "N/A"
            print(f"  {vlabel:25s} runs={[round(x,1) for x in xs] if xs else '-':40s}  mean={mean if mean else 'N/A':>8}  std={std:.1f if std is not None else '-'}  Δ={d_str}  [{stab}]{err_note}" if std is not None else f"  {vlabel:25s} FAILED{err_note}")
            final[lab]["candidates"].append((vlabel, mean, std, d_pct, stab, errs))

    # Best per-shape that passes stability gate
    print("\n" + "=" * 110)
    print("BEST STABLE PER-SHAPE")
    print("=" * 110)
    for lab, fd in final.items():
        ref_mean, ref_std = fd["ref"]
        passers = [(c[3], c[0], c[1], c[2]) for c in fd["candidates"] if c[4] == "STABLE" and c[3] is not None]
        passers.sort(reverse=True)
        if not passers:
            print(f"  [{lab}] no stable candidate beats REF")
            continue
        d, vk, mean, std = passers[0]
        verdict = "WIN" if d >= 0.5 else "TIE"
        print(f"  [{lab}] BEST = {vk}   mean={mean:.1f}   std={std:.1f}   Δ vs REF={d:+.2f}%   [{verdict}]")


if __name__ == "__main__":
    main()
