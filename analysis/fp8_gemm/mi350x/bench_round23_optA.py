#!/usr/bin/env python3
"""R23A perf smoke (warmup=200 iters=500 trim=10%) — STATIC_XCD_REMAP probe
on 3 DLA shapes (DLA1/DLA2/DLA7) x 4 variants (baseline + remap + remap_g4 + remap_g8).

Per benchmark-rules.md MANDATORY: warmup=200, iters=500, 10% trim.
Subprocess-isolated. Runs across GPUs 0,1,2 (one shape per GPU, no concurrent same-GPU).
Gate: Δ% >= +1.5% (best variant vs _xcd_baseline) -> WIN -> recommend wiring.
"""
import json, math, os, subprocess, sys, sysconfig, time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10

# (label, M_native, N, K, parent_suffix, comp_tflops)
SHAPES = [
    ("DLA1",   4096, 32768, 128256, "_ts_pf6_6_v12_memc",         5781.1),
    ("DLA2", 128256, 32768,   4096, "_ts_gm2_v12_memc_dc",        4536.4),
    ("DLA7",  28672, 32768,   4096, "_ts_lgk2_v12_memc_btw_all",  4466.6),
]

VARIANT_SUFFIXES = [
    "_xcd_baseline",
    "_xcd_remap",
    "_xcd_remap_g4",
    "_xcd_remap_g8",
]

# One GPU per shape; intra-shape variants serialized on that GPU.
SHAPE_GPU = {"DLA1": 0, "DLA2": 1, "DLA7": 2}


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


def bench_shape(args):
    """Run all 4 variants for a single shape serially on its assigned GPU."""
    lab, M, N, K, ps, gpu = args
    results = {}
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu)
    for vs in VARIANT_SUFFIXES:
        suffix = ps + vs
        so_path, script = make_bench_script(M, N, K, suffix)
        if not os.path.exists(so_path):
            results[vs] = {"error": "missing .so"}
            continue
        try:
            r = subprocess.run([sys.executable, "-c", script],
                               capture_output=True, text=True, timeout=1800, env=env)
            if r.returncode != 0:
                results[vs] = {"error": f"rc={r.returncode}", "stderr": r.stderr[-500:]}
            else:
                results[vs] = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception as e:
            results[vs] = {"error": str(e)}
        if "tflops" in results[vs]:
            print(f"  {lab:6s} {vs:18s} gpu={gpu}  {results[vs]['tflops']:>8.2f} TFLOPS", flush=True)
        else:
            print(f"  {lab:6s} {vs:18s} gpu={gpu}  ERR {results[vs].get('error')}", flush=True)
    return lab, results


def main():
    args_list = [
        (lab, M, N, K, ps, SHAPE_GPU[lab])
        for (lab, M, N, K, ps, _comp) in SHAPES
    ]
    print(f"R23A smoke: {len(args_list)} shapes x {len(VARIANT_SUFFIXES)} variants, GPUs={list(SHAPE_GPU.values())}")
    print(f"warmup={WARMUP} iters={ITERS} trim={TRIM}")
    print("=" * 110)
    out = {"shapes": {lab: {} for (lab, *_rest) in SHAPES},
           "warmup": WARMUP, "iters": ITERS, "trim": TRIM}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(args_list)) as ex:
        futs = {ex.submit(bench_shape, a): a for a in args_list}
        for fut in as_completed(futs):
            lab, results = fut.result()
            out["shapes"][lab] = results
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    with open(os.path.join(SCRIPT_DIR, "bench_round23_optA.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 110)
    print("SUMMARY (Δ vs _xcd_baseline)")
    print("=" * 110)
    summary = {}
    for (lab, _M, _N, _K, _ps, comp) in SHAPES:
        b = out["shapes"][lab].get("_xcd_baseline", {})
        bt = b.get("tflops")
        if bt is None:
            print(f"  {lab}: BASELINE FAILED -> {b}")
            continue
        print(f"  {lab} _xcd_baseline = {bt:.2f} TFLOPS  (comp={comp})")
        best_d_pct = -1e9
        best_v = None
        for vs in VARIANT_SUFFIXES:
            if vs == "_xcd_baseline":
                continue
            v = out["shapes"][lab].get(vs, {})
            vt = v.get("tflops")
            if vt is None:
                print(f"    {vs:18s}  ERR {v.get('error')}")
                continue
            d = vt - bt
            d_rel = d / bt * 100  # delta % vs baseline
            d_pp = d / comp * 100  # delta in pp of competitor
            mark = " <-- WIN >=1.5%" if d_rel >= 1.5 else ""
            print(f"    {vs:18s}  {vt:>8.2f}  Δ={d:+.2f} ({d_rel:+.2f}%, {d_pp:+.2f}pp/comp){mark}")
            if d_rel > best_d_pct:
                best_d_pct = d_rel
                best_v = vs
        summary[lab] = {"baseline_tflops": bt, "best_variant": best_v, "best_delta_pct": best_d_pct}

    out["summary"] = summary
    with open(os.path.join(SCRIPT_DIR, "bench_round23_optA.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 110)
    print("VERDICT")
    print("=" * 110)
    any_win = False
    for lab, s in summary.items():
        bv = s.get("best_variant")
        bd = s.get("best_delta_pct")
        if bv is None:
            print(f"  {lab}: no valid variants")
        elif bd >= 1.5:
            print(f"  {lab}: WIN best={bv} Δ={bd:+.2f}% (>=1.5% gate)")
            any_win = True
        else:
            print(f"  {lab}: DEAD END best={bv} Δ={bd:+.2f}% (<1.5% gate)")
    if any_win:
        print("\n=> R23A: WIN on at least one DLA shape; recommend wiring STATIC_XCD_REMAP=1 into bench_all_42.py for winning shapes.")
    else:
        print("\n=> R23A: DEAD END across all 3 DLA shapes; STATIC_XCD_REMAP does not close the producer-side TCP_TA_DATA_STALL gap.")


if __name__ == "__main__":
    main()
