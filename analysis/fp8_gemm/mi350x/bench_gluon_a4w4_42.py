#!/usr/bin/env python3
"""Benchmark Gluon a4w4 kernel (LLIR+amdgcnas) across all 42 shapes.

Uses raw e8m0 scale format (not preshuffle_mfma16).
hipEvent timing: warmup=30, iters=100, trimmed mean (10% trim).
Each shape runs in its own subprocess for crash isolation.

Environment variables (MUST be set before import):
    TRITON_ENABLE_LLIR_SCHED=1
    TRITON_ENABLE_AMDGCN_AS=1

Usage:
    TRITON_ENABLE_LLIR_SCHED=1 TRITON_ENABLE_AMDGCN_AS=1 python3 bench_gluon_a4w4_42.py
"""

import json
import os
import subprocess
import sys
import textwrap
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GLUON_DIR = "/shared_nfs/kyle/test/gfx9-gluon-tutorials/kernels/gemm/a4w4"

# All 42 shapes: (M, N, K, cpp_tflops)
# cpp_tflops from bench_all_42.py ALL_SHAPES (competitor reference)
ALL_SHAPES = [
    # K < 4096 (10 shapes)
    (16384,  4096,  2048, 2995.0),
    (16384,  4096,  3072, 3492.3),
    (16384,  6144,  2048, 3047.6),
    (32768,  4096,  2048, 3131.8),
    (32768,  4096,  3072, 3630.6),
    (32768,  6144,  2048, 3239.9),
    (16384, 14336,  2048, 3301.3),
    (16384, 28672,  2048, 3482.3),
    (32768, 14336,  2048, 3351.4),
    (32768, 28672,  2048, 3353.4),
    # K >= 4096 (32 shapes)
    ( 4096,  4096, 16384, 4642.1),
    ( 4096, 14336, 16384, 5013.0),
    ( 6144,  4096, 16384, 4428.1),
    ( 4096,  4096,  8192, 3959.9),
    ( 4096,  4096, 32768, 5152.8),
    ( 4096,  6144, 32768, 3784.2),
    ( 4096, 14336,  8192, 4345.8),
    ( 4096, 28672, 32768, 5649.9),
    ( 4096, 32768,  4096, 4166.5),
    ( 4096, 32768,  6144, 4548.6),
    ( 4096, 32768, 14336, 5296.1),
    ( 4096, 32768, 28672, 5568.2),
    ( 4096, 32768,128256, 5781.1),
    ( 4096,128256, 32768, 3195.3),
    ( 6144,  4096,  8192, 3822.0),
    ( 6144, 32768,  4096, 4291.0),
    (14336,  4096, 32768, 5245.4),
    (14336, 32768,  4096, 4462.6),
    (16384,  4096,  4096, 3951.8),
    (16384,  4096,  6144, 4259.9),
    (16384,  4096,  7168, 4443.2),
    (16384,  4096, 14336, 5142.1),
    (16384,  4096, 28672, 5525.3),
    (16384,  6144,  4096, 4042.5),
    (16384, 14336,  4096, 4255.8),
    (16384, 28672,  4096, 4411.7),
    (28672,  4096,  8192, 4810.0),
    (28672,  4096, 16384, 5350.6),
    (28672, 32768,  4096, 4466.6),
    (32768,  4096,  7168, 4666.8),
    (32768,  4096, 14336, 5223.4),
    (128256, 32768,  4096, 4536.4),
]


def make_runner_script(m, n, k):
    """Generate a Python script that benchmarks a single (M,N,K) shape
    using the Gluon a4w4 matmul with raw e8m0 scales."""
    script = textwrap.dedent(f'''\
        #!/usr/bin/env python3
        """Auto-generated Gluon a4w4 runner for M={m}, N={n}, K={k}."""
        import gc, json, sys, torch

        sys.path.insert(0, {GLUON_DIR!r})
        from matmul_kernel import matmul

        M, N, K = {m}, {n}, {k}
        WARMUP = 30
        ITERS = 100
        TRIM_FRAC = 0.10

        DEVICE = "cuda"
        SCALE_GROUP_SIZE = 32

        torch.manual_seed(42)

        try:
            # Generate random MXFP4 packed tensors
            a_low = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8)
            a_high = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8)
            a_fp4 = (a_high << 4 | a_low).to(device=DEVICE)

            b_low = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device=DEVICE)
            b_high = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device=DEVICE)
            b_fp4 = b_low | b_high << 4

            # Raw e8m0 scales (NOT preshuffle_mfma16)
            import math
            M_pad = (M + 255) // 256 * 256
            a_scales = torch.randint(
                124, 128, (K // SCALE_GROUP_SIZE, M_pad), dtype=torch.uint8, device=DEVICE
            ).T[:M]
            b_scales = torch.randint(
                124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device=DEVICE
            ).T

            # Warmup (also triggers JIT compilation)
            for _ in range(WARMUP):
                matmul(a_fp4, b_fp4, a_scales, b_scales)
            torch.cuda.synchronize()

            # Timed iterations using hipEvent
            times_ms = []
            for _ in range(ITERS):
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                matmul(a_fp4, b_fp4, a_scales, b_scales)
                end_evt.record()
                torch.cuda.synchronize()
                times_ms.append(start_evt.elapsed_time(end_evt))

            # Trimmed mean
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
            result = {{
                "M": M, "N": N, "K": K,
                "avg_ms": None, "tflops": None,
                "status": "OOM",
            }}
        except Exception as e:
            result = {{
                "M": M, "N": N, "K": K,
                "avg_ms": None, "tflops": None,
                "status": f"ERR:{{e}}",
            }}

        print("BENCH_JSON_START")
        print(json.dumps(result))
        print("BENCH_JSON_END")
    ''')
    return script


def run_single_shape(m, n, k, cpp_tflops, work_dir):
    """Run benchmark for a single shape in its own subprocess."""
    script_content = make_runner_script(m, n, k)
    runner_path = os.path.join(work_dir, f"_gluon_run_{m}_{n}_{k}.py")
    with open(runner_path, "w") as f:
        f.write(script_content)

    env = os.environ.copy()
    env["TRITON_ENABLE_LLIR_SCHED"] = "1"
    env["TRITON_ENABLE_AMDGCN_AS"] = "1"

    try:
        result = subprocess.run(
            [sys.executable, runner_path],
            capture_output=True, text=True,
            timeout=600,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {
            "M": m, "N": n, "K": k, "avg_ms": None, "tflops": None,
            "cpp_tflops": cpp_tflops, "status": "TIMEOUT",
        }

    if result.returncode != 0:
        err_lines = result.stderr.strip().split("\n")
        short_err = err_lines[-1][:120] if err_lines else "unknown"
        return {
            "M": m, "N": n, "K": k, "avg_ms": None, "tflops": None,
            "cpp_tflops": cpp_tflops, "status": f"CRASH:{short_err}",
        }

    stdout = result.stdout
    try:
        start_idx = stdout.index("BENCH_JSON_START") + len("BENCH_JSON_START")
        end_idx = stdout.index("BENCH_JSON_END")
        json_str = stdout[start_idx:end_idx].strip()
        r = json.loads(json_str)
        r["cpp_tflops"] = cpp_tflops
        return r
    except (ValueError, json.JSONDecodeError):
        return {
            "M": m, "N": n, "K": k, "avg_ms": None, "tflops": None,
            "cpp_tflops": cpp_tflops, "status": "PARSE_FAIL",
        }


def main():
    print("=" * 80)
    print("Gluon a4w4 Kernel (LLIR+amdgcnas) vs C++ Kernel - All 42 Shapes")
    print("=" * 80)
    print(f"Total shapes: {len(ALL_SHAPES)}")
    print(f"Gluon dir: {GLUON_DIR}")
    print(f"Scale format: raw e8m0 (Gluon) vs preshuffle_mfma16 (C++)")
    print(f"Timing: hipEvent, warmup=30, iters=100, trimmed mean (10%)")
    print(f"Env: TRITON_ENABLE_LLIR_SCHED=1, TRITON_ENABLE_AMDGCN_AS=1")
    print()

    work_dir = os.path.join(SCRIPT_DIR, "work_gluon_42")
    os.makedirs(work_dir, exist_ok=True)

    all_results = []

    for idx, (m, n, k, cpp) in enumerate(ALL_SHAPES):
        print(f"[{idx+1:>2}/{len(ALL_SHAPES)}] "
              f"{m:>6} x {n:>6} x {k:>6}  ", end="", flush=True)
        t0 = time.time()

        r = run_single_shape(m, n, k, cpp, work_dir)
        all_results.append(r)
        elapsed = time.time() - t0

        if r["status"] == "OK":
            ratio = r["tflops"] / r["cpp_tflops"] * 100.0
            tag = "GLUON" if r["tflops"] >= r["cpp_tflops"] else "C++"
            print(f"Gluon={r['tflops']:7.1f}T  C++={r['cpp_tflops']:7.1f}T  "
                  f"({ratio:5.1f}%)  {tag:>5}  [{elapsed:.1f}s]")
        else:
            print(f"{r['status']}  [{elapsed:.1f}s]")

    print()

    # --- Summary table ---
    print("=" * 80)
    print(f"{'Shape':<25} {'Gluon(LLIR)':>12} {'C++':>8} {'Gluon/C++':>10} {'Winner':>8}")
    print("-" * 80)

    wins_gluon = 0
    wins_cpp = 0
    errors = 0
    for r in all_results:
        M, N, K = r["M"], r["N"], r["K"]
        shape_str = f"{M}x{N}x{K}"
        cpp = r["cpp_tflops"]
        if r["status"] == "OK":
            gluon = r["tflops"]
            ratio = gluon / cpp * 100.0
            if gluon >= cpp:
                winner = "GLUON"
                wins_gluon += 1
            else:
                winner = "C++"
                wins_cpp += 1
            print(f"{shape_str:<25} {gluon:>10.1f}T {cpp:>7.1f}T {ratio:>9.1f}% {winner:>8}")
        else:
            errors += 1
            print(f"{shape_str:<25} {'FAIL':>12} {cpp:>7.1f}T {'---':>10} {'ERR':>8}  ({r['status'][:40]})")

    print("-" * 80)
    total = len(all_results)
    total_valid = wins_gluon + wins_cpp
    print(f"\nGluon WINS: {wins_gluon}/{total}  |  C++ WINS: {wins_cpp}/{total}  |  Errors: {errors}/{total}")

    if total_valid > 0:
        valid = [r for r in all_results if r["status"] == "OK"]
        avg_gluon = sum(r["tflops"] for r in valid) / len(valid)
        avg_cpp = sum(r["cpp_tflops"] for r in valid) / len(valid)
        avg_ratio = sum(r["tflops"]/r["cpp_tflops"] for r in valid) / len(valid) * 100
        geomean_ratio = 1.0
        for r in valid:
            geomean_ratio *= (r["tflops"] / r["cpp_tflops"])
        geomean_ratio = geomean_ratio ** (1.0/len(valid)) * 100
        print(f"Avg TFLOPS: Gluon={avg_gluon:.1f}  C++={avg_cpp:.1f}")
        print(f"Avg Ratio: {avg_ratio:.1f}%  Geomean Ratio: {geomean_ratio:.1f}%")

    print("=" * 80)

    # Save results to JSON
    results_file = os.path.join(SCRIPT_DIR, "bench_gluon_a4w4_42_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "env": {
                "TRITON_ENABLE_LLIR_SCHED": "1",
                "TRITON_ENABLE_AMDGCN_AS": "1",
            },
            "scale_format": "raw_e8m0",
            "warmup": 30,
            "iters": 100,
            "trim_frac": 0.10,
            "total_shapes": total,
            "wins_gluon": wins_gluon,
            "wins_cpp": wins_cpp,
            "errors": errors,
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
