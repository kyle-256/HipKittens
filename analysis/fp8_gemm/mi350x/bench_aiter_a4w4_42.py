#!/usr/bin/env python3
"""Benchmark aiter a4w4 CK kernel across all 42 shapes.

Uses aiter's gemm_a4w4 (CK backend: f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256).
hipEvent timing: warmup=200, iters=500, trimmed mean (10% trim).
Each shape runs in its own subprocess for crash isolation.

Usage:
    PYTHONPATH=/shared_nfs/kyle/test/aiter:$PYTHONPATH python3 bench_aiter_a4w4_42.py
"""

import json
import os
import subprocess
import sys
import textwrap
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AITER_DIR = "/shared_nfs/kyle/test/aiter"

# All 42 shapes: (M, N, K)
ALL_SHAPES = [
    (16384,  4096,  2048),
    (16384,  4096,  3072),
    (16384,  6144,  2048),
    (32768,  4096,  2048),
    (32768,  4096,  3072),
    (32768,  6144,  2048),
    (16384, 14336,  2048),
    (16384, 28672,  2048),
    (32768, 14336,  2048),
    (32768, 28672,  2048),
    ( 4096,  4096, 16384),
    ( 4096, 14336, 16384),
    ( 6144,  4096, 16384),
    ( 4096,  4096,  8192),
    ( 4096,  4096, 32768),
    ( 4096,  6144, 32768),
    ( 4096, 14336,  8192),
    ( 4096, 28672, 32768),
    ( 4096, 32768,  4096),
    ( 4096, 32768,  6144),
    ( 4096, 32768, 14336),
    ( 4096, 32768, 28672),
    ( 4096, 32768, 128256),
    ( 4096, 128256, 32768),
    ( 6144,  4096,  8192),
    ( 6144, 32768,  4096),
    (14336,  4096, 32768),
    (14336, 32768,  4096),
    (16384,  4096,  4096),
    (16384,  4096,  6144),
    (16384,  4096,  7168),
    (16384,  4096, 14336),
    (16384,  4096, 28672),
    (16384,  6144,  4096),
    (16384, 14336,  4096),
    (16384, 28672,  4096),
    (28672,  4096,  8192),
    (28672,  4096, 16384),
    (28672, 32768,  4096),
    (32768,  4096,  7168),
    (32768,  4096, 14336),
    (128256, 32768,  4096),
]


def make_runner_script(m, n, k):
    script = textwrap.dedent(f'''\
        #!/usr/bin/env python3
        import gc, json, sys, torch
        sys.path.insert(0, {AITER_DIR!r})

        from aiter.ops.gemm_op_a4w4 import gemm_a4w4

        M, N, K = {m}, {n}, {k}
        WARMUP = 200
        ITERS = 500
        TRIM_FRAC = 0.10

        torch.manual_seed(42)

        try:
            A = torch.randint(0, 256, (M, K//2), dtype=torch.uint8, device='cuda')
            B = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device='cuda')
            A_sc = torch.randint(124, 128, (M, K//32), dtype=torch.uint8, device='cuda')
            B_sc = torch.randint(124, 128, (N, K//32), dtype=torch.uint8, device='cuda')

            run = lambda: gemm_a4w4(A, B, A_sc, B_sc)

            for _ in range(WARMUP):
                run()
            torch.cuda.synchronize()

            times_ms = []
            for _ in range(ITERS):
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                run()
                end_evt.record()
                torch.cuda.synchronize()
                times_ms.append(start_evt.elapsed_time(end_evt))

            times_ms.sort()
            trim_count = int(len(times_ms) * TRIM_FRAC)
            if trim_count > 0:
                trimmed = times_ms[trim_count:-trim_count]
            else:
                trimmed = times_ms
            avg_ms = sum(trimmed) / len(trimmed)

            tflops = 2.0 * M * N * K / (avg_ms * 1e-3) / 1e12
            result = {{
                "M": M, "N": N, "K": K,
                "avg_ms": round(avg_ms, 4),
                "tflops": round(tflops, 1),
                "status": "OK",
            }}

        except torch.cuda.OutOfMemoryError:
            result = {{"M": M, "N": N, "K": K, "avg_ms": None, "tflops": None, "status": "OOM"}}
        except Exception as e:
            result = {{"M": M, "N": N, "K": K, "avg_ms": None, "tflops": None, "status": f"ERR:{{e}}"}}

        print("BENCH_JSON_START")
        print(json.dumps(result))
        print("BENCH_JSON_END")
    ''')
    return script


def run_single_shape(m, n, k, work_dir):
    script_content = make_runner_script(m, n, k)
    runner_path = os.path.join(work_dir, f"_aiter_run_{m}_{n}_{k}.py")
    with open(runner_path, "w") as f:
        f.write(script_content)

    env = os.environ.copy()
    env["PYTHONPATH"] = AITER_DIR + ":" + env.get("PYTHONPATH", "")

    try:
        result = subprocess.run(
            [sys.executable, runner_path],
            capture_output=True, text=True,
            timeout=600, env=env,
        )
    except subprocess.TimeoutExpired:
        return {"M": m, "N": n, "K": k, "avg_ms": None, "tflops": None, "status": "TIMEOUT"}

    if result.returncode != 0:
        err_lines = result.stderr.strip().split("\n")
        short_err = err_lines[-1][:120] if err_lines else "unknown"
        return {"M": m, "N": n, "K": k, "avg_ms": None, "tflops": None, "status": f"CRASH:{short_err}"}

    stdout = result.stdout
    try:
        start_idx = stdout.index("BENCH_JSON_START") + len("BENCH_JSON_START")
        end_idx = stdout.index("BENCH_JSON_END")
        return json.loads(stdout[start_idx:end_idx].strip())
    except (ValueError, json.JSONDecodeError):
        return {"M": m, "N": n, "K": k, "avg_ms": None, "tflops": None, "status": "PARSE_FAIL"}


def main():
    print("=" * 70)
    print("aiter afp4wfp4 (preshuffled) - All 42 Shapes")
    print("=" * 70)
    print(f"Total shapes: {len(ALL_SHAPES)}")
    print(f"aiter dir: {AITER_DIR}")
    print(f"Timing: hipEvent, warmup=200, iters=500, trimmed mean (10%)")
    print()

    work_dir = os.path.join(SCRIPT_DIR, "work_aiter_42")
    os.makedirs(work_dir, exist_ok=True)

    all_results = []

    for idx, (m, n, k) in enumerate(ALL_SHAPES):
        print(f"[{idx+1:>2}/{len(ALL_SHAPES)}] "
              f"{m:>6} x {n:>6} x {k:>6}  ", end="", flush=True)
        t0 = time.time()

        r = run_single_shape(m, n, k, work_dir)
        all_results.append(r)
        elapsed = time.time() - t0

        if r["status"] == "OK":
            print(f"{r['tflops']:7.1f} TFLOPS  ({r['avg_ms']:.4f} ms)  [{elapsed:.1f}s]")
        else:
            print(f"{r['status'][:60]}  [{elapsed:.1f}s]")

    print()
    print("=" * 70)
    print(f"{'M':>7}  {'N':>7}  {'K':>7}  {'TFLOPS':>8}  {'ms':>8}  Status")
    print("-" * 70)

    ok_count = 0
    for r in all_results:
        if r["status"] == "OK":
            ok_count += 1
            print(f"{r['M']:>7}  {r['N']:>7}  {r['K']:>7}  {r['tflops']:>8.1f}  {r['avg_ms']:>8.4f}  OK")
        else:
            print(f"{r['M']:>7}  {r['N']:>7}  {r['K']:>7}  {'---':>8}  {'---':>8}  {r['status'][:40]}")

    print("-" * 70)
    valid = [r for r in all_results if r["status"] == "OK"]
    if valid:
        avg_tflops = sum(r["tflops"] for r in valid) / len(valid)
        print(f"OK: {ok_count}/{len(all_results)}  Avg TFLOPS: {avg_tflops:.1f}")
    print("=" * 70)

    results_file = os.path.join(SCRIPT_DIR, "bench_aiter_a4w4_42_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "aiter_dir": AITER_DIR,
            "warmup": 200, "iters": 500, "trim_frac": 0.10,
            "total_shapes": len(all_results),
            "ok_count": ok_count,
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
