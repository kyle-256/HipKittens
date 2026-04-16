#!/usr/bin/env python3
"""Parallel benchmark of all 42 shapes across multiple GPUs.

Uses pre-compiled .so files from build_all42/.
Splits shapes across GPUs for parallel execution.
Per-variant subprocess timeout to prevent hangs.

Usage:
    python3 bench_all42_parallel.py [gpu_list]
    # Default: GPUs 1,2,3,4
    # Example: python3 bench_all42_parallel.py 1,2
"""

import json, math, os, subprocess, sys, time, importlib.util
from multiprocessing import Pool, Manager
import sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10
VARIANT_TIMEOUT = 180  # 3 min per variant (prevents hangs)

ALL_SHAPES = [
    (16384,  4096,  2048, 2995.0), (16384,  4096,  3072, 3492.3),
    (16384,  6144,  2048, 3047.6), (32768,  4096,  2048, 3131.8),
    (32768,  4096,  3072, 3630.6), (32768,  6144,  2048, 3239.9),
    (16384, 14336,  2048, 3301.3), (16384, 28672,  2048, 3482.3),
    (32768, 14336,  2048, 3351.4), (32768, 28672,  2048, 3353.4),
    (4096,   4096,  16384, 4642.1), (4096,  14336,  16384, 5013.0),
    (6144,   4096,  16384, 4428.1), (4096,   4096,   8192, 3959.9),
    (4096,   4096,  32768, 5152.8), (4096,   6144,  32768, 3784.2),
    (4096,  14336,   8192, 4345.8), (4096,  28672,  32768, 5649.9),
    (4096,  32768,   4096, 4166.5), (4096,  32768,   6144, 4548.6),
    (4096,  32768,  14336, 5296.1), (4096,  32768,  28672, 5568.2),
    (4096,  32768, 128256, 5781.1), (4096, 128256,  32768, 3195.3),
    (6144,   4096,   8192, 3822.0), (6144,  32768,   4096, 4291.0),
    (14336,  4096,  32768, 5245.4), (14336, 32768,   4096, 4462.6),
    (16384,  4096,   4096, 3951.8), (16384,  4096,   6144, 4259.9),
    (16384,  4096,   7168, 4443.2), (16384,  4096,  14336, 5142.1),
    (16384,  4096,  28672, 5525.3), (16384,  6144,   4096, 4042.5),
    (16384, 14336,   4096, 4255.8), (16384, 28672,   4096, 4411.7),
    (28672,  4096,   8192, 4810.0), (28672,  4096,  16384, 5350.6),
    (28672, 32768,   4096, 4466.6), (32768,  4096,   7168, 4666.8),
    (32768,  4096,  14336, 5223.4), (128256, 32768,  4096, 4536.4),
]

VARIANTS = [
    ("", "default"), ("_gm2", "gm2"), ("_u16", "u16"),
    ("_gm2u16", "gm2u16"), ("_gm1", "gm1"), ("_gm8", "gm8"),
    ("_u8", "u8"), ("_u32", "u32"), ("_gm8u16", "gm8u16"),
    ("_gm16", "gm16"), ("_gm8u8", "gm8u8"), ("_gm16u16", "gm16u16"),
    ("_swap", "swap"), ("_swap_gm8", "swap_gm8"),
    ("_ts", "ts"), ("_ts_gm8", "ts_gm8"), ("_ts_u16", "ts_u16"),
    ("_gm2u8", "gm2u8"), ("_gm16u8", "gm16u8"), ("_gm1u16", "gm1u16"),
    ("_ts_v12", "ts_v12"), ("_ts_gm8_v12", "ts_gm8_v12"),
    ("_ts_gm2", "ts_gm2"), ("_ts_gm2u8", "ts_gm2u8"),
    ("_spread_gm2u8", "spread_gm2u8"), ("_spread_gm2u8_v12", "spread_gm2u8_v12"),
    ("_spread_gm8", "spread_gm8"), ("_v12", "v12"), ("_gm8_v12", "gm8_v12"),
    ("_no_nvs", "no_nvs"), ("_pf4", "pf4"), ("_ts_pf4", "ts_pf4"),
    ("_gm2_v12", "gm2_v12"),
    ("_ext_br", "ext_br"), ("_ts_ext_br", "ts_ext_br"),
    ("_gm8_ext_br", "gm8_ext_br"), ("_ts_gm8_ext_br", "ts_gm8_ext_br"),
    ("_gm32", "gm32"), ("_gm64", "gm64"),
    ("_ts_gm16", "ts_gm16"), ("_ts_gm32", "ts_gm32"),
    ("_ext_br_v12", "ext_br_v12"), ("_ts_ext_br_v12", "ts_ext_br_v12"),
    ("_gm2_ext_br", "gm2_ext_br"), ("_ts_gm2_ext_br", "ts_gm2_ext_br"),
    ("_gm16_v12", "gm16_v12"), ("_gm1_v12", "gm1_v12"),
    ("_ts_gm2_v12", "ts_gm2_v12"),
    ("_gm2_ext_br_v12", "gm2_ext_br_v12"), ("_ts_gm2_ext_br_v12", "ts_gm2_ext_br_v12"),
    ("_lgk2", "lgk2"), ("_lgk4", "lgk4"),
    ("_ts_lgk2", "ts_lgk2"), ("_ts_lgk4", "ts_lgk4"),
    ("_ts_lgk2_v12", "ts_lgk2_v12"),
    ("_v4", "v4"), ("_ts_v4", "ts_v4"),
    ("_v16", "v16"), ("_ts_v16", "ts_v16"),
    ("_no_embed", "no_embed"), ("_ts_no_embed", "ts_no_embed"),
    ("_ts_no_embed_v12", "ts_no_embed_v12"),
]


def bench_variant(gpu_id, m, n, k, module_suffix):
    """Run a single variant benchmark in a subprocess on specified GPU."""
    module_name = f"tk_mxfp4_gluon_cpp_n{n}_k{k}{module_suffix}"
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
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=VARIANT_TIMEOUT, env=env
        )
        if r.returncode != 0:
            return None
        return json.loads(r.stdout.strip())
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return None


def bench_shape(args):
    """Benchmark one shape across all variants. Returns (shape_idx, best_result)."""
    shape_idx, m, n, k, comp, gpu_id = args
    best_t, best_tag, best_ms = 0, "default", None

    for suffix, tag in VARIANTS:
        r = bench_variant(gpu_id, m, n, k, suffix)
        if r and r["tflops"] > best_t:
            best_t = r["tflops"]
            best_tag = tag
            best_ms = r["ms"]

    ratio = best_t / comp * 100 if comp > 0 and best_t > 0 else 0
    status = "WIN" if best_t >= comp else "LOSE"
    result = {
        "M": m, "N": n, "K": k,
        "tflops": round(best_t, 1), "avg_ms": best_ms,
        "comp": comp, "status": "OK" if best_t > 0 else "FAIL",
        "best_variant": best_tag, "ratio": round(ratio, 1),
    }
    print(f"[{shape_idx+1:>2}/42] {m:>6}x{n:>6}x{k:>6}  "
          f"{best_t:>7.1f} vs {comp:>7.1f}  ({ratio:>5.1f}%)  {status}  "
          f"[{best_tag}]  GPU{gpu_id}", flush=True)
    return (shape_idx, result)


def main():
    gpus = [1, 2, 3, 4]
    if len(sys.argv) > 1:
        gpus = [int(g) for g in sys.argv[1].split(",")]

    print(f"MXFP4 Parallel Benchmark - GPUs: {gpus}")
    print(f"Variants: {len(VARIANTS)}, Shapes: {len(ALL_SHAPES)}")
    print(f"warmup={WARMUP}, iters={ITERS}, trim={TRIM}")
    print(f"Per-variant timeout: {VARIANT_TIMEOUT}s")
    print("=" * 80)

    # Assign shapes to GPUs round-robin
    tasks = []
    for i, (m, n, k, comp) in enumerate(ALL_SHAPES):
        gpu = gpus[i % len(gpus)]
        tasks.append((i, m, n, k, comp, gpu))

    t0 = time.time()

    # Run with process pool — one process per GPU to avoid GPU contention
    # Actually, run sequentially per GPU but parallelize across GPUs
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    # Group tasks by GPU
    gpu_tasks = {g: [] for g in gpus}
    for task in tasks:
        gpu_tasks[task[-1]].append(task)

    results = [None] * len(ALL_SHAPES)
    lock = threading.Lock()

    def run_gpu_tasks(gpu_id, task_list):
        for task in task_list:
            idx, result = bench_shape(task)
            with lock:
                results[idx] = result

    threads = []
    for gpu_id in gpus:
        t = threading.Thread(target=run_gpu_tasks, args=(gpu_id, gpu_tasks[gpu_id]))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed/60:.1f} minutes")

    # Summary
    print("\n" + "=" * 80)
    print(f"{'M':>7} {'N':>7} {'K':>7} {'Ours':>8} {'Comp':>8} {'Ratio':>7} {'Tag':>15} Result")
    print("-" * 80)

    wins, losses, errors = 0, 0, 0
    for r in results:
        if r is None:
            errors += 1
            continue
        m, n, k = r["M"], r["N"], r["K"]
        if r["status"] == "OK":
            ours = r["tflops"]
            ratio = r["ratio"]
            tag = "WIN" if ours >= r["comp"] else "LOSE"
            if tag == "WIN": wins += 1
            else: losses += 1
            print(f"{m:>7} {n:>7} {k:>7} {ours:>8.1f} {r['comp']:>8.1f} "
                  f"{ratio:>6.1f}% {r['best_variant']:>15} {tag}")
        else:
            errors += 1
            print(f"{m:>7} {n:>7} {k:>7} {'---':>8} {r['comp']:>8.1f} "
                  f"{'---':>7} {'---':>15} {r['status']}")

    print("-" * 80)
    total = len(ALL_SHAPES)
    total_valid = wins + losses
    win_rate = (wins / total_valid * 100.0) if total_valid > 0 else 0
    print(f"WIN: {wins}/{total}  LOSE: {losses}/{total}  "
          f"ERR: {errors}/{total}  Win rate: {win_rate:.0f}%")

    if total_valid > 0:
        valid = [r for r in results if r and r["status"] == "OK"]
        avg_ratio = sum(r["ratio"] for r in valid) / len(valid)
        print(f"Avg ratio: {avg_ratio:.1f}%")

    # Save results
    results_file = os.path.join(SCRIPT_DIR, "bench_all42_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "warmup": WARMUP, "iters": ITERS, "trim_frac": TRIM,
            "total_shapes": total, "wins": wins, "losses": losses,
            "errors": errors, "win_rate": round(win_rate, 1),
            "elapsed_minutes": round(elapsed / 60, 1),
            "gpus": gpus,
            "results": [r for r in results if r],
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
