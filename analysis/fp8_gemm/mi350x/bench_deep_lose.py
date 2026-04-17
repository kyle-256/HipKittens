#!/usr/bin/env python3
"""Benchmark deep-LOSE shapes against new targeted variants + R1 best.

Spot test only the 10 deep-LOSE shapes from Round 1.
Multi-GPU parallel via threads (one GPU per thread).

warmup=200, iters=500, trim=10% — MANDATORY benchmark rules.
"""
import json, math, os, subprocess, sys, time, sysconfig
from concurrent.futures import ThreadPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10
VARIANT_TIMEOUT = 240

# 10 deep-LOSE shapes from Round 1 (M, N, K, comp, r1_best_variant_suffix)
DEEP_LOSE_SHAPES = [
    (4096, 32768, 128256, 5781.1, "_ts_gm8"),
    (14336, 4096, 32768, 5245.4, "_v16"),
    (16384, 4096, 28672, 5525.3, "_ts_v12"),
    (128256, 32768, 4096, 4536.4, "_ts_gm2_v12_memc_dc"),
    (4096, 32768, 28672, 5568.2, "_u8"),
    (28672, 4096, 16384, 5350.6, "_ts_u16"),
    (4096, 28672, 32768, 5649.9, "_u16"),
    (28672, 32768, 4096, 4466.6, "_ts_lgk2_v12_memc"),
    (32768, 4096, 14336, 5223.4, "_ts_gm8"),
    (4096, 32768, 14336, 5296.1, "_ts_lgk2_v12_memc"),
]

# New targeted variants from build_deep_lose_variants.py
NEW_VARIANTS = [
    "_wpe1", "_wpe2",
    "_ts_lgk2_v12_wpe1_memc", "_ts_lgk2_v12_wpe2_memc",
    "_v16_wpe1", "_v16_wpe1_memc",
    "_ts_v12_wpe1", "_ts_gm8_wpe1", "_ts_gm8_wpe1_memc",
    "_u8_wpe1", "_u16_wpe1", "_v16_wpe2",
    "_u8_lgk2", "_u8_lgk2_v12", "_u8_lgk2_memc", "_u8_v12_memc",
    "_u16_lgk2", "_u16_lgk2_memc", "_u16_v12", "_u16_v12_memc",
    "_ts_u16_lgk2", "_ts_u16_v12_memc", "_ts_u8_lgk2_v12_memc",
    "_gm8_lgk2_v12_memc", "_ts_gm8_lgk2_v12_memc",
    "_gm8_ext_br_lgk2", "_gm8_ext_br_lgk2_memc",
    "_gm8_v12_lgk2_ext_br",
    "_ts_gm4", "_ts_gm4_lgk2_v12_memc", "_gm4_v12",
    "_pf6_6_lgk2_v12", "_pf6_6_v12_memc", "_ts_pf6_6_v12_memc",
    "_ts_pf6_6_u8", "_ts_pf6_6_u16",
    "_agpr192_wpe2", "_agpr192", "_ts_lgk2_agpr192",
    "_u8_v20", "_u16_v20", "_ts_lgk2_v20_u8",
    "_ts_v16_u8", "_ts_v16_u16",
]

# Existing R1 best variants for comparison (also tested)
R1_BEST_VARIANTS = [
    "_ts_gm8", "_v16", "_ts_v12", "_ts_gm2_v12_memc_dc", "_u8",
    "_ts_u16", "_u16", "_ts_lgk2_v12_memc",
]

ALL_VARIANTS = list(set(NEW_VARIANTS + R1_BEST_VARIANTS))


def bench_variant(gpu_id, m, n, k, suffix):
    module_name = f"tk_mxfp4_gluon_cpp_n{n}_k{k}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return None
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
print(json.dumps({{"tflops":round(t,1),"ms":round(avg,4)}}))
"""
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=VARIANT_TIMEOUT, env=env)
        if r.returncode != 0:
            return None
        return json.loads(r.stdout.strip())
    except Exception:
        return None


def bench_shape(gpu_id, m, n, k, comp, r1_best):
    results = {}
    for suffix in ALL_VARIANTS:
        r = bench_variant(gpu_id, m, n, k, suffix)
        if r:
            results[suffix] = r["tflops"]
            print(f"   [GPU{gpu_id}] {m}x{n}x{k} {suffix:35s} {r['tflops']:7.1f} ({r['tflops']/comp*100:.1f}%)", flush=True)
    if not results:
        return None
    best_suffix = max(results, key=results.get)
    best_t = results[best_suffix]
    return {
        "M": m, "N": n, "K": k, "comp": comp,
        "r1_best": r1_best,
        "all_results": results,
        "new_best_variant": best_suffix,
        "new_best_tflops": best_t,
        "new_ratio": round(best_t / comp * 100, 1),
    }


def main():
    gpus = [0, 1, 6, 7]
    if len(sys.argv) > 1:
        gpus = [int(g) for g in sys.argv[1].split(",")]
    print(f"Deep-LOSE bench — GPUs: {gpus}")
    print(f"Variants tested per shape: {len(ALL_VARIANTS)}")
    print(f"warmup={WARMUP} iters={ITERS} trim={TRIM}")
    print("=" * 80)

    # Distribute shapes across GPUs
    shape_assignments = {g: [] for g in gpus}
    for i, shape in enumerate(DEEP_LOSE_SHAPES):
        shape_assignments[gpus[i % len(gpus)]].append(shape)

    results_by_shape = {}
    t0 = time.time()

    def run_gpu(gpu_id, shapes):
        for (m, n, k, comp, r1_best) in shapes:
            print(f">> [GPU{gpu_id}] starting {m}x{n}x{k}", flush=True)
            res = bench_shape(gpu_id, m, n, k, comp, r1_best)
            if res:
                results_by_shape[(m, n, k)] = res
                print(f"<< [GPU{gpu_id}] {m}x{n}x{k} done. NEW best: {res['new_best_variant']} = {res['new_best_tflops']:.1f} ({res['new_ratio']}%)", flush=True)

    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futs = [ex.submit(run_gpu, g, shape_assignments[g]) for g in gpus]
        for f in futs:
            f.result()

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed/60:.1f} min")
    print("=" * 80)
    print(f"\n{'M':>7} {'N':>7} {'K':>7} {'R1_best':>20} {'R1_T':>8} {'NEW_best':>30} {'NEW_T':>8} {'R1%':>6} {'NEW%':>6} {'flip?':>8}")
    print("-" * 130)

    summary = []
    for (m, n, k, comp, r1_best) in DEEP_LOSE_SHAPES:
        res = results_by_shape.get((m, n, k))
        if not res:
            print(f"{m:>7} {n:>7} {k:>7}  NO RESULTS")
            continue
        r1_t = res["all_results"].get(r1_best, 0)
        new_t = res["new_best_tflops"]
        r1_pct = r1_t / comp * 100 if r1_t else 0
        new_pct = new_t / comp * 100
        flipped = "WIN!" if new_t >= comp else ("better" if new_pct > r1_pct + 0.5 else "")
        print(f"{m:>7} {n:>7} {k:>7} {r1_best:>20} {r1_t:>8.1f} {res['new_best_variant']:>30} {new_t:>8.1f} {r1_pct:>5.1f}% {new_pct:>5.1f}% {flipped:>8}")
        summary.append({
            "M": m, "N": n, "K": k, "comp": comp,
            "r1_best": r1_best, "r1_tflops": r1_t, "r1_ratio": round(r1_pct, 1),
            "new_best": res["new_best_variant"], "new_tflops": new_t, "new_ratio": round(new_pct, 1),
            "flipped_to_win": new_t >= comp,
        })

    out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "warmup": WARMUP, "iters": ITERS, "trim_frac": TRIM,
        "gpus": gpus,
        "elapsed_minutes": round(elapsed/60, 1),
        "summary": summary,
        "all_results": {f"{m}x{n}x{k}": v for (m,n,k), v in results_by_shape.items()},
    }
    with open(os.path.join(SCRIPT_DIR, "bench_deep_lose_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved.")


if __name__ == "__main__":
    main()
