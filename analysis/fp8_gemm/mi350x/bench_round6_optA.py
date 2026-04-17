#!/usr/bin/env python3
"""Round 6 Optimizer A bench: scheduling-strategy sweep on 4096x32768x128256
plus regression check on 3 WIN-shape sample + 3 deep-LOSE neighbors.

warmup=200, iters=500, trim=10%. Multi-GPU parallel (uses GPUs 0,1).
Per-variant correctness check via cross-comparison vs the _r6a_sched_memc
variant (which IS the current best variant — identical source, only sched
strategy changes, so byte-equal output is sufficient correctness signal).
"""
import os, sys, json, math, time, subprocess, sysconfig
from concurrent.futures import ThreadPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10

# Variants to test (must match build_round6_optA.py)
VARIANTS = [
    "_r6a_sched_maxocc",
    "_r6a_sched_maxilp",
    "_r6a_sched_memc",          # baseline-equivalent (sanity)
    "_r6a_sched_iterilp",
    "_r6a_sched_iterminreg",
    "_r6a_sched_itermaxocc",
]

# Plus the actual existing baseline so we can compare directly
BASELINE_VARIANT = "_ts_pf6_6_v12_memc"

# Shapes: (M, N, K, comp_tflops, label)
TARGET_SHAPE = (4096, 32768, 128256, 5781.1, "TARGET")
WIN_SHAPES = [
    (16384, 4096, 2048, 2995.0, "WIN1"),
    (4096, 128256, 32768, 3195.3, "WIN2"),
    (4096, 4096, 8192, 3959.9, "WIN3"),
]
NEIGHBOR_SHAPES = [
    (14336, 4096, 32768, 5245.4, "NEIGHBOR_LOSE1"),
    (16384, 4096, 28672, 5525.3, "NEIGHBOR_LOSE2"),
    (4096, 32768, 28672, 5568.2, "NEIGHBOR_LOSE3"),
]

ALL_SHAPES = [TARGET_SHAPE] + WIN_SHAPES + NEIGHBOR_SHAPES


CHILD_SCRIPT = '''
import sys, math, torch, importlib.util, json
torch.manual_seed(0)
WARMUP, ITERS, TRIM = {WARMUP}, {ITERS}, {TRIM}
M, N, K = {M}, {N}, {K}
mode = "{MODE}"

def gen_fp4(r, K):
    c = K // 2
    return (torch.randint(0,16,(r,c),dtype=torch.uint8,device="cuda")<<4)|torch.randint(0,16,(r,c),dtype=torch.uint8,device="cuda")
def preshuffle(se):
    r,kb=se.shape; pr=math.ceil(r/64)*64; pk=math.ceil(kb/8)*8
    raw=torch.full((pr,pk),0x7F,dtype=torch.uint8,device=se.device)
    raw[:r,:kb]=(se.to(torch.int16)+127).to(torch.uint8)
    sh=raw.view(pr//32,2,16,pk//8,2,4,1).permute(0,3,5,2,4,1,6).contiguous().view(pr//32,pk*32)
    sh=sh.view(pr//64,2,pk*32//4,4).permute(0,2,1,3).contiguous()
    return sh.view(pr//64,pk*64)

spec=importlib.util.spec_from_file_location("{MOD_NAME}","{SO}")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

A=gen_fp4(M,K); B=gen_fp4(N,K)
sc_a=torch.randint(-2,3,(M,K//32),dtype=torch.int8,device="cuda")
sc_b=torch.randint(-2,3,(N,K//32),dtype=torch.int8,device="cuda")
A_sc=preshuffle(sc_a); B_sc=preshuffle(sc_b)
C=torch.zeros(M,N,dtype=torch.bfloat16,device="cuda")
run=lambda:mod.gemm_rcr(A,B,A_sc,B_sc,C)

if mode == "bench":
    for _ in range(WARMUP): run()
    torch.cuda.synchronize()
    times=[]
    for _ in range(ITERS):
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record(); run(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    tn=int(len(times)*TRIM)
    times=times[tn:-tn] if tn>0 else times
    avg=sum(times)/len(times); t=2.0*M*N*K/(avg*1e-3)/1e12
    print(json.dumps({{"tflops":round(t,2),"ms":round(avg,4)}}))
elif mode == "checksum":
    # Compute checksum + SNR vs torch reference
    C.zero_()
    run()
    torch.cuda.synchronize()
    cs = float(C.float().sum().item())
    cmax = float(C.float().abs().max().item())
    # Build torch ref
    FP4_LUT = torch.tensor(
        [0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0],
        dtype=torch.float32, device="cuda"
    )
    def unpack(p, K):
        lo=(p&0x0F).to(torch.int64); hi=((p>>4)&0x0F).to(torch.int64)
        out=torch.empty(p.shape[0],K,dtype=torch.float32,device="cuda")
        out[:,0::2]=FP4_LUT[lo]; out[:,1::2]=FP4_LUT[hi]
        return out
    def expand(exp, K):
        return torch.pow(2.0, exp.float()).repeat_interleave(32, dim=1)[:, :K]
    Af=unpack(A,K)*expand(sc_a,K)
    Bf=unpack(B,K)*expand(sc_b,K)
    Cref=Af@Bf.T
    Cf=C.float(); Crf=Cref.float()
    # NaN-safe SNR: a NaN-laden output must NOT silently pass via noi==NaN→False→snr=+Inf.
    got_nan = (~torch.isfinite(Cf)).any().item()
    ref_nan = (~torch.isfinite(Crf)).any().item()
    if got_nan or ref_nan:
        snr_out = None
    else:
        noise=Cf-Crf
        sig=(Crf**2).sum().item()
        noi=(noise**2).sum().item()
        if not (sig > 0):
            snr_out = None
        elif noi <= 0:
            snr_out = float("inf")
        else:
            snr_out = round(10*math.log10(sig/noi), 2)
    print(json.dumps({{"checksum": cs, "Cmax": cmax, "snr_db": snr_out, "got_nan": got_nan, "ref_nan": ref_nan}}))
'''


def run_child(gpu_id, mode, mod_name, so_path, M, N, K, timeout=300):
    if not os.path.exists(so_path):
        return {"error": f"missing {so_path}"}
    script = CHILD_SCRIPT.format(
        WARMUP=WARMUP, ITERS=ITERS, TRIM=TRIM, M=M, N=N, K=K,
        MODE=mode, MOD_NAME=mod_name, SO=so_path,
    )
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=timeout, env=env)
        if r.returncode != 0:
            return {"error": f"rc={r.returncode}", "stderr": r.stderr[-300:]}
        return json.loads(r.stdout.strip().split("\n")[-1])
    except Exception as e:
        return {"error": str(e)}


def bench_variant_on_shape(gpu_id, suffix, M, N, K):
    mod = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix}"
    so = os.path.join(BUILD_DIR, f"{mod}{EXT_SUFFIX}")
    return run_child(gpu_id, "bench", mod, so, M, N, K)


def checksum_variant_on_shape(gpu_id, suffix, M, N, K):
    mod = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix}"
    so = os.path.join(BUILD_DIR, f"{mod}{EXT_SUFFIX}")
    return run_child(gpu_id, "checksum", mod, so, M, N, K, timeout=120)


def main():
    gpus = [0, 1]
    if len(sys.argv) > 1:
        gpus = [int(g) for g in sys.argv[1].split(",")]
    print(f"Round 6 OptA bench  GPUs={gpus}  warmup={WARMUP} iters={ITERS} trim={TRIM}")
    print(f"Variants: {len(VARIANTS)}  Shapes: {len(ALL_SHAPES)}")
    print("=" * 100)

    # Step 1: correctness via checksum cross-comparison on small shape
    # 1024x1024x4096 (matches AGENT_PROMPT correctness recipe)
    print("\n--- Step 1: Correctness SNR check (M=1024 N=1024 K=4096) ---")
    M, N, K = 1024, 1024, 4096
    checksum_results = {}
    # Also include the established baseline _ts_pf6_6_v12_memc
    all_for_checksum = [BASELINE_VARIANT] + VARIANTS
    for i, suf in enumerate(all_for_checksum):
        gpu = gpus[i % len(gpus)]
        r = checksum_variant_on_shape(gpu, suf, M, N, K)
        checksum_results[suf] = r
        if "error" in r:
            print(f"  {suf:30s} ERROR: {r}")
        else:
            print(f"  {suf:30s} checksum={r['checksum']:.6e}  Cmax={r['Cmax']:.4e}")

    # Verify all checksums match the baseline
    # NaN-safe correctness gate (fixes documented Round 6 false-OK SNR bug).
    sys.path.insert(0, SCRIPT_DIR)
    from snr_check import is_snr_ok, classify_snr
    correctness_ok = {}
    for suf in VARIANTS:
        rec = checksum_results.get(suf, {})
        snr = rec.get("snr_db")  # may be None / NaN / +Inf / float
        ok = is_snr_ok(snr)
        cls, reason = classify_snr(snr)
        correctness_ok[suf] = ok
        mark = "OK" if ok else "BAD"
        extra = ""
        if rec.get("got_nan") or rec.get("ref_nan"):
            extra = f" [got_nan={rec.get('got_nan')} ref_nan={rec.get('ref_nan')}]"
        print(f"  {suf:30s} SNR={snr} dB {mark}{extra}  ({reason})")

    # Step 2: bench all variants on all shapes (parallel by GPU)
    print("\n--- Step 2: Bench all (variant x shape) ---")
    tasks = []
    for shape in ALL_SHAPES:
        M, N, K, comp, label = shape
        for suf in VARIANTS:
            if not correctness_ok.get(suf, False):
                continue
            tasks.append((suf, M, N, K, comp, label))
        # Also bench the established baseline for direct comparison
        tasks.append((BASELINE_VARIANT, M, N, K, comp, label))

    print(f"Tasks: {len(tasks)}")
    results = {}
    t0 = time.time()

    def worker(gpu_id, my_tasks):
        for (suf, M, N, K, comp, label) in my_tasks:
            r = bench_variant_on_shape(gpu_id, suf, M, N, K)
            key = (suf, M, N, K)
            results[key] = r
            if "error" in r:
                print(f"  [GPU{gpu_id}] {label} {M}x{N}x{K} {suf:30s} ERROR: {r.get('error')}", flush=True)
            else:
                ratio = r["tflops"] / comp * 100
                print(f"  [GPU{gpu_id}] {label} {M}x{N}x{K} {suf:30s} {r['tflops']:>7.1f} ({ratio:.2f}% of {comp})", flush=True)

    # Distribute tasks round-robin across GPUs
    per_gpu = {g: [] for g in gpus}
    for i, t in enumerate(tasks):
        per_gpu[gpus[i % len(gpus)]].append(t)

    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futs = [ex.submit(worker, g, per_gpu[g]) for g in gpus]
        for f in futs:
            f.result()

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed/60:.1f} min")

    # Step 3: Summarize
    print("\n--- Step 3: Summary ---")
    print(f"\n{'Shape':<25} {'Variant':<32} {'TFLOPS':>9} {'%comp':>8}  {'delta_vs_baseline':>20}")
    print("-" * 100)
    summary = []
    for shape in ALL_SHAPES:
        M, N, K, comp, label = shape
        base_t = results.get((BASELINE_VARIANT, M, N, K), {}).get("tflops", 0)
        for suf in [BASELINE_VARIANT] + VARIANTS:
            r = results.get((suf, M, N, K), {})
            if "tflops" not in r:
                continue
            t = r["tflops"]
            ratio = t / comp * 100
            delta = t - base_t
            delta_pp = (t - base_t) / comp * 100  # percentage points of comp
            shape_label = f"{label}({M}x{N}x{K})"
            print(f"{shape_label:<25} {suf:<32} {t:>9.1f} {ratio:>7.2f}% {delta:>+10.1f}  ({delta_pp:+.2f}pp)")
            summary.append({
                "shape": (M, N, K),
                "label": label,
                "variant": suf,
                "tflops": t,
                "comp": comp,
                "ratio_pct": round(ratio, 2),
                "vs_baseline_tflops": round(delta, 2),
                "vs_baseline_pp": round(delta_pp, 3),
            })
        print()

    out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "warmup": WARMUP, "iters": ITERS, "trim_frac": TRIM,
        "gpus": gpus,
        "elapsed_minutes": round(elapsed/60, 1),
        "checksum_results": {k: v for k, v in checksum_results.items()},
        "correctness_ok": correctness_ok,
        "summary": summary,
        "raw_results": {f"{suf}|{M}x{N}x{K}": v for (suf, M, N, K), v in results.items()},
    }
    out_path = os.path.join(SCRIPT_DIR, "bench_round6_optA_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
