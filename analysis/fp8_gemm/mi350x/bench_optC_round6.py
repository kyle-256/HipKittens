#!/usr/bin/env python3
"""Bench Round 6 Optimizer C variants on 16384x4096x28672.

Includes correctness check (SNR) on 1024x1024x4096 first.
warmup=200 iters=500 trim=10%
"""
import json, math, os, subprocess, sys, time, sysconfig
from concurrent.futures import ThreadPoolExecutor

# NaN-safe SNR gate (fixes documented Round 6 false-OK bug; see snr_check.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from snr_check import is_snr_ok, classify_snr  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10
VARIANT_TIMEOUT = 600

# Target shape
TARGET_M, TARGET_N, TARGET_K = 16384, 4096, 28672
TARGET_COMP = 5525.3
BASELINE_TFLOPS = 4997.5  # _u8 baseline

# Variants under test (suffix only; module = tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix})
VARIANTS = [
    "_u8",            # baseline (already built, TAIL_SPLIT=0, U=8)
    "_ts_u8",         # TAIL_SPLIT=1, U=8, default TAIL_BARRIER_VMCNT=8
    "_ts_u8_tv0",
    "_ts_u8_tv4",
    "_ts_u8_tv12",
    "_ts_u8_tv16",
]

def bench_variant(gpu_id, m, n, k, suffix):
    module_name = f"tk_mxfp4_gluon_cpp_n{n}_k{k}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return None, "missing .so"
    script = f"""
import sys, math, torch, importlib.util, json
torch.manual_seed(0)
WARMUP, ITERS, TRIM = {WARMUP}, {ITERS}, {TRIM}
M, N, K = {m}, {n}, {k}
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
                           capture_output=True, text=True, timeout=VARIANT_TIMEOUT, env=env)
        if r.returncode != 0:
            return None, r.stderr[-300:]
        return json.loads(r.stdout.strip()), None
    except Exception as e:
        return None, str(e)


def correctness_check(gpu_id, suffix):
    """Compare variant output against baseline _u8 at smallish shape. SNR_dB.

    We use the target N=4096 K=28672 with M reduced to 1024 to keep it cheap but realistic.
    """
    # Use small K to avoid bf16 overflow in C; same N to reuse target build
    M, N, K = 1024, 4096, 4096
    n, k = N, K
    module_name = f"tk_mxfp4_gluon_cpp_n{n}_k{k}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return None, f"missing .so {module_name}"
    ref_module = f"tk_mxfp4_gluon_cpp_n{n}_k{k}_u8"
    ref_path = os.path.join(BUILD_DIR, f"{ref_module}{EXT_SUFFIX}")
    if not os.path.exists(ref_path):
        return None, f"missing ref .so {ref_module}"

    script = f"""
import sys, math, torch, importlib.util, json
torch.manual_seed(42)
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
spec_v=importlib.util.spec_from_file_location('{module_name}','{so_path}')
mod_v=importlib.util.module_from_spec(spec_v); spec_v.loader.exec_module(mod_v)
spec_r=importlib.util.spec_from_file_location('{ref_module}','{ref_path}')
mod_r=importlib.util.module_from_spec(spec_r); spec_r.loader.exec_module(mod_r)
A=gen_fp4(M,K); B=gen_fp4(N,K)
sc_a=torch.randint(-2,3,(M,K//32),dtype=torch.int8,device='cuda')
sc_b=torch.randint(-2,3,(N,K//32),dtype=torch.int8,device='cuda')
A_sc=preshuffle(sc_a); B_sc=preshuffle(sc_b)
Cv=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
Cr=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
mod_v.gemm_rcr(A,B,A_sc,B_sc,Cv)
mod_r.gemm_rcr(A,B,A_sc,B_sc,Cr)
torch.cuda.synchronize()
ref = Cr.float()
got = Cv.float()
ref_nan = (~torch.isfinite(ref)).any().item()
got_nan = (~torch.isfinite(got)).any().item()
diff = ref - got
sig_pwr = (ref**2).mean().item()
err_pwr = (diff**2).mean().item()
# NaN-safe SNR: flag NaN/Inf inputs so the parent can mark FAIL (not silently OK).
if ref_nan or got_nan or not (sig_pwr > 0):
    snr_out = None
elif err_pwr <= 0:
    snr_out = float("inf")
else:
    snr_out = round(10*math.log10(sig_pwr/err_pwr), 2)
max_abs = float("nan") if got_nan or ref_nan else round(diff.abs().max().item(),4)
print(json.dumps({{"snr_db":snr_out,"max_abs":max_abs,"ref_nan":ref_nan,"got_nan":got_nan}}))
"""
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=180, env=env)
        if r.returncode != 0:
            return None, r.stderr[-300:]
        return json.loads(r.stdout.strip()), None
    except Exception as e:
        return None, str(e)


def main():
    gpus = [4, 5]
    if len(sys.argv) > 1:
        gpus = [int(g) for g in sys.argv[1].split(",")]
    print(f"Round 6 Optimizer C - GPUs: {gpus}")
    print(f"Target: {TARGET_M}x{TARGET_N}x{TARGET_K}, baseline _u8 = {BASELINE_TFLOPS}")
    print(f"warmup={WARMUP} iters={ITERS} trim={TRIM}")
    print("=" * 80)

    # Distribute variants across GPUs
    bench_results = {}
    snr_results = {}

    def run_variant(gpu_id, suffix):
        # First correctness check (only for non-baseline _ts variants)
        if suffix.startswith("_ts_u8"):
            snr, snr_err = correctness_check(gpu_id, suffix)
            if snr is None:
                snr_results[suffix] = {"snr_db": None, "error": snr_err}
                print(f"   [GPU{gpu_id}] {suffix:20s} CORRECTNESS FAIL: {snr_err}", flush=True)
                # Skip bench if correctness failed
                bench_results[suffix] = None
                return
            snr_results[suffix] = snr
            # NaN-safe gate: NaN/None/-Inf snr_db now correctly classified as FAIL
            # (was: `if snr_db < 25` mis-classified NaN as OK because NaN<25 is False).
            cls, reason = classify_snr(snr.get('snr_db'))
            print(f"   [GPU{gpu_id}] {suffix:20s} SNR={snr.get('snr_db')} dB  ref_nan={snr.get('ref_nan')} got_nan={snr.get('got_nan')}", flush=True)
            if not is_snr_ok(snr.get('snr_db')):
                print(f"   [GPU{gpu_id}] {suffix:20s} REJECTED ({reason})", flush=True)
                bench_results[suffix] = None
                return

        b, b_err = bench_variant(gpu_id, TARGET_M, TARGET_N, TARGET_K, suffix)
        if b is None:
            bench_results[suffix] = None
            print(f"   [GPU{gpu_id}] {suffix:20s} BENCH FAIL: {b_err}", flush=True)
        else:
            bench_results[suffix] = b
            pct = b['tflops'] / TARGET_COMP * 100
            delta = b['tflops'] - BASELINE_TFLOPS
            print(f"   [GPU{gpu_id}] {suffix:20s} {b['tflops']:7.2f} TFLOPS ({pct:.2f}% comp, delta={delta:+.1f})", flush=True)

    # Distribute round-robin
    assignments = {g: [] for g in gpus}
    for i, suffix in enumerate(VARIANTS):
        assignments[gpus[i % len(gpus)]].append(suffix)

    def run_gpu(gpu_id, suffixes):
        for s in suffixes:
            run_variant(gpu_id, s)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futs = [ex.submit(run_gpu, g, assignments[g]) for g in gpus]
        for f in futs:
            f.result()
    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed/60:.2f} min")
    print("=" * 80)

    # Print summary
    print("\nSummary table (target: 16384x4096x28672):")
    print(f"  {'Variant':<20s} {'SNR':>8s} {'TFLOPS':>10s} {'%comp':>8s} {'delta':>10s}")
    summary = []
    for suffix in VARIANTS:
        b = bench_results.get(suffix)
        snr = snr_results.get(suffix, {}).get("snr_db", "-") if suffix.startswith("_ts_u8") else "-"
        if b is None:
            print(f"  {suffix:<20s} {str(snr):>8} {'FAIL':>10}")
        else:
            pct = b['tflops'] / TARGET_COMP * 100
            delta = b['tflops'] - BASELINE_TFLOPS
            print(f"  {suffix:<20s} {str(snr):>8} {b['tflops']:>10.2f} {pct:>7.2f}% {delta:>+9.1f}")
            summary.append({"suffix": suffix, "snr_db": snr, "tflops": b['tflops'], "pct": pct, "delta": delta})

    # Save
    out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "warmup": WARMUP, "iters": ITERS, "trim_frac": TRIM,
        "gpus": gpus,
        "target": {"M": TARGET_M, "N": TARGET_N, "K": TARGET_K, "comp": TARGET_COMP, "baseline": BASELINE_TFLOPS},
        "snr": snr_results,
        "bench": {k: v for k, v in bench_results.items() if v is not None},
        "summary": summary,
        "elapsed_min": round(elapsed/60, 2),
    }
    out_path = os.path.join(SCRIPT_DIR, "bench_optC_round6_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
