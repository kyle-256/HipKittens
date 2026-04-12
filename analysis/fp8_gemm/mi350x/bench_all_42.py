#!/usr/bin/env python3
"""Benchmark MXFP4 Gluon C++ kernel across all 42 competitor shapes.

Uses preshuffle_mfma16 scale format.
hipEvent timing: warmup 50, timing 100, trimmed mean (10% trim).
Each shape runs in its own subprocess for crash isolation.

The kernel uses compile-time N_DIM for block decomposition (bpc = N_DIM / BLK),
so we must compile one .so per unique (N, K) pair.

Usage:
    python3 bench_all_42.py
"""

import json
import math
import os
import subprocess
import sys
import sysconfig
import textwrap
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")
TK_ROOT = os.environ.get(
    "THUNDERKITTENS_ROOT",
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
)
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"

# All 42 shapes: (M, N, K, competitor_tflops)
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
    (4096,   4096,  16384, 4642.1),
    (4096,  14336,  16384, 5013.0),
    (6144,   4096,  16384, 4428.1),
    (4096,   4096,   8192, 3959.9),
    (4096,   4096,  32768, 5152.8),
    (4096,   6144,  32768, 3784.2),
    (4096,  14336,   8192, 4345.8),
    (4096,  28672,  32768, 5649.9),
    (4096,  32768,   4096, 4166.5),
    (4096,  32768,   6144, 4548.6),
    (4096,  32768,  14336, 5296.1),
    (4096,  32768,  28672, 5568.2),
    (4096,  32768, 128256, 5781.1),
    (4096, 128256,  32768, 3195.3),
    (6144,   4096,   8192, 3822.0),
    (6144,  32768,   4096, 4291.0),
    (14336,  4096,  32768, 5245.4),
    (14336, 32768,   4096, 4462.6),
    (16384,  4096,   4096, 3951.8),
    (16384,  4096,   6144, 4259.9),
    (16384,  4096,   7168, 4443.2),
    (16384,  4096,  14336, 5142.1),
    (16384,  4096,  28672, 5525.3),
    (16384,  6144,   4096, 4042.5),
    (16384, 14336,   4096, 4255.8),
    (16384, 28672,   4096, 4411.7),
    (28672,  4096,   8192, 4810.0),
    (28672,  4096,  16384, 5350.6),
    (28672, 32768,   4096, 4466.6),
    (32768,  4096,   7168, 4666.8),
    (32768,  4096,  14336, 5223.4),
    (128256, 32768,  4096, 4536.4),
]


def get_unique_nk_pairs(shapes):
    """Get unique (N, K) pairs since the kernel needs both at compile time."""
    return sorted(set((n, k) for _, n, k, _ in shapes))


def module_name_for_nk(n, k):
    return f"tk_mxfp4_gluon_cpp_n{n}_k{k}"


def build_for_nk(n_dim, k_dim, build_dir):
    """Compile kernel with -DK_DIM=k_dim -DN_DIM=n_dim using make."""
    module_name = module_name_for_nk(n_dim, k_dim)
    out_file = f"{module_name}{EXT_SUFFIX}"
    out_path = os.path.join(build_dir, out_file)

    if os.path.exists(out_path):
        print(f"  [cached] N={n_dim}, K={k_dim}")
        return out_path

    # Patch PYBIND11_MODULE name so multiple variants can coexist
    with open(KERNEL_SRC, "r") as f:
        src = f.read()
    patched = src.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},"
    )
    wrapper_src = os.path.join(build_dir, f"wrap_n{n_dim}_k{k_dim}.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)

    print(f"  Compiling N={n_dim}, K={k_dim} ...", end=" ", flush=True)
    t0 = time.time()

    env = os.environ.copy()
    env["THUNDERKITTENS_ROOT"] = TK_ROOT

    cmd = (
        f'make -C {SCRIPT_DIR} TARGET={os.path.join(build_dir, module_name)} '
        f'SRC={wrapper_src} '
        f'CPPFLAGS="-DK_DIM={k_dim} -DN_DIM={n_dim}"'
    )
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, env=env
    )
    elapsed = time.time() - t0

    if result.returncode != 0 or not os.path.exists(out_path):
        print(f"FAILED ({elapsed:.1f}s)")
        print(result.stderr[-2000:])
        return None

    print(f"OK ({elapsed:.1f}s)")
    return out_path


def make_single_shape_script(module_name, so_dir, m, n, k, comp_tflops):
    """Generate a Python script that benchmarks a single (M,N,K) shape.

    Uses preshuffle_mfma16 scale format.
    hipEvent timing with trimmed mean.
    Outputs JSON result to stdout.
    """
    script = textwrap.dedent(f'''\
        #!/usr/bin/env python3
        """Auto-generated runner for M={m}, N={n}, K={k}."""
        import gc, json, math, sys, torch
        torch.manual_seed(0)

        sys.path.insert(0, {so_dir!r})
        import {module_name}

        M, N, K = {m}, {n}, {k}
        WARMUP = 100
        ITERS = 300
        TRIM_FRAC = 0.10  # trim 10% from each end

        k_blocks = K // 32

        def gen_fp4(rows, K):
            cols = K // 2
            lo = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device="cuda")
            hi = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device="cuda")
            return (hi << 4) | lo

        def preshuffle_mfma16(scale_exp):
            """Preshuffled format: [M/32, pk*32]."""
            rows, kb = scale_exp.shape
            pr = math.ceil(rows / 32) * 32
            pk = math.ceil(kb / 8) * 8
            raw = torch.full((pr, pk), 0x7F, dtype=torch.uint8, device=scale_exp.device)
            raw[:rows, :kb] = (scale_exp.to(torch.int16) + 127).to(torch.uint8)
            sh = raw.view(pr // 32, 2, 16, pk // 8, 2, 4, 1)
            sh = sh.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
            return sh.view(pr // 32, pk * 32)

        try:
            A = gen_fp4(M, K)
            B = gen_fp4(N, K)
            sc_exp_a = torch.randint(-2, 3, (M, k_blocks), dtype=torch.int8, device="cuda")
            sc_exp_b = torch.randint(-2, 3, (N, k_blocks), dtype=torch.int8, device="cuda")
            A_sc = preshuffle_mfma16(sc_exp_a)
            B_sc = preshuffle_mfma16(sc_exp_b)
            C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

            run = lambda: {module_name}.gemm_rcr(A, B, A_sc, B_sc, C)

            # Warmup
            for _ in range(WARMUP):
                run()
            torch.cuda.synchronize()

            # Timed iterations using hipEvent
            times_ms = []
            for _ in range(ITERS):
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                run()
                end_evt.record()
                torch.cuda.synchronize()
                times_ms.append(start_evt.elapsed_time(end_evt))

            # Trimmed mean: remove 10% fastest and 10% slowest
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
                "comp": {comp_tflops},
                "status": "OK",
            }}

        except torch.cuda.OutOfMemoryError:
            result = {{
                "M": M, "N": N, "K": K,
                "avg_ms": None, "tflops": None,
                "comp": {comp_tflops}, "status": "OOM",
            }}
        except Exception as e:
            result = {{
                "M": M, "N": N, "K": K,
                "avg_ms": None, "tflops": None,
                "comp": {comp_tflops}, "status": f"ERR:{{e}}",
            }}

        print("BENCH_JSON_START")
        print(json.dumps(result))
        print("BENCH_JSON_END")
    ''')
    return script


def run_single_shape(m, n, k, comp_tflops, build_dir, work_dir):
    """Run benchmark for a single shape in its own subprocess. Returns result dict."""
    module_name = module_name_for_nk(n, k)
    script_content = make_single_shape_script(module_name, build_dir, m, n, k, comp_tflops)

    runner_path = os.path.join(work_dir, f"_run_{m}_{n}_{k}.py")
    with open(runner_path, "w") as f:
        f.write(script_content)

    try:
        result = subprocess.run(
            [sys.executable, runner_path],
            capture_output=True, text=True,
            timeout=600,  # 10 min max per shape
        )
    except subprocess.TimeoutExpired:
        return {
            "M": m, "N": n, "K": k, "avg_ms": None, "tflops": None,
            "comp": comp_tflops, "status": "TIMEOUT",
        }

    if result.returncode != 0:
        # Extract short error from stderr
        err_lines = result.stderr.strip().split("\n")
        short_err = err_lines[-1][:80] if err_lines else "unknown"
        return {
            "M": m, "N": n, "K": k, "avg_ms": None, "tflops": None,
            "comp": comp_tflops, "status": "CRASH",
        }

    # Parse JSON from stdout
    stdout = result.stdout
    try:
        start_idx = stdout.index("BENCH_JSON_START") + len("BENCH_JSON_START")
        end_idx = stdout.index("BENCH_JSON_END")
        json_str = stdout[start_idx:end_idx].strip()
        return json.loads(json_str)
    except (ValueError, json.JSONDecodeError):
        return {
            "M": m, "N": n, "K": k, "avg_ms": None, "tflops": None,
            "comp": comp_tflops, "status": "PARSE_FAIL",
        }


def main():
    print("=" * 70)
    print("MXFP4 Gluon C++ - All 42 Shapes (preshuffle_mfma16 scales)")
    print("=" * 70)
    print(f"Total shapes: {len(ALL_SHAPES)}")
    print(f"ThunderKittens: {TK_ROOT}")
    print(f"Scale format: preshuffle_mfma16")
    print(f"Warmup: 50, Iters: 100, Trimmed mean (10% each end)")
    print(f"Each shape runs in its own subprocess for isolation")
    print()

    build_dir = os.path.join(SCRIPT_DIR, "build_all42")
    work_dir = os.path.join(SCRIPT_DIR, "work_all42")
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    nk_pairs = get_unique_nk_pairs(ALL_SHAPES)
    print(f"Unique (N,K) pairs: {len(nk_pairs)}")
    print()

    # --- Build phase ---
    print("--- Build Phase ---")
    for n_val, k_val in nk_pairs:
        so_path = build_for_nk(n_val, k_val, build_dir)
        if so_path is None:
            print(f"FATAL: compile failed for N={n_val}, K={k_val}. Aborting.")
            sys.exit(1)
    print()

    # --- Benchmark phase ---
    print("--- Benchmark Phase ---")
    all_results = []

    for idx, (m, n, k, comp) in enumerate(ALL_SHAPES):
        print(f"[{idx+1:>2}/{len(ALL_SHAPES)}] "
              f"{m:>6} x {n:>6} x {k:>6}  ", end="", flush=True)

        r = run_single_shape(m, n, k, comp, build_dir, work_dir)
        all_results.append(r)

        if r["status"] == "OK":
            ratio = r["tflops"] / r["comp"] * 100.0
            tag = "WIN" if r["tflops"] >= r["comp"] else "LOSE"
            print(f"{r['tflops']:7.1f} vs {r['comp']:7.1f}  "
                  f"({ratio:5.1f}%)  {tag}")
        else:
            print(f"{r['status']}")

    print()

    # --- Summary table ---
    print("=" * 70)
    print(f"{'M':>7}  {'N':>7}  {'K':>7}  {'Ours':>8}  {'Comp':>8}  "
          f"{'Ratio':>7}  Result")
    print("-" * 70)

    wins = 0
    losses = 0
    errors = 0
    for r in all_results:
        M, N, K = r["M"], r["N"], r["K"]
        comp = r["comp"]
        if r["status"] == "OK":
            ours = r["tflops"]
            ratio = ours / comp * 100.0
            tag = "WIN" if ours >= comp else "LOSE"
            if tag == "WIN":
                wins += 1
            else:
                losses += 1
            print(f"{M:>7}  {N:>7}  {K:>7}  {ours:>8.1f}  {comp:>8.1f}  "
                  f"{ratio:>6.1f}%  {tag}")
        else:
            errors += 1
            print(f"{M:>7}  {N:>7}  {K:>7}  {'---':>8}  {comp:>8.1f}  "
                  f"{'---':>7}  {r['status']}")

    print("-" * 70)
    total_valid = wins + losses
    total = len(all_results)
    win_rate = (wins / total_valid * 100.0) if total_valid > 0 else 0.0
    print(f"Total: {wins}/{total} WIN, {win_rate:.0f}% win rate")

    if total_valid > 0:
        valid = [r for r in all_results if r["status"] == "OK"]
        avg_ours = sum(r["tflops"] for r in valid) / len(valid)
        avg_comp = sum(r["comp"] for r in valid) / len(valid)
        avg_ratio = sum(r["tflops"]/r["comp"] for r in valid) / len(valid) * 100
        print(f"Avg TFLOPS: Ours={avg_ours:.1f}  Comp={avg_comp:.1f}  "
              f"Avg Ratio={avg_ratio:.1f}%")
    if errors > 0:
        print(f"Errors: {errors}/{total}")

    print("=" * 70)

    # Save results to JSON
    results_file = os.path.join(SCRIPT_DIR, "bench_all42_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scale_format": "preshuffle_mfma16",
            "warmup": 50,
            "iters": 100,
            "trim_frac": 0.10,
            "total_shapes": total,
            "wins": wins,
            "losses": losses,
            "errors": errors,
            "win_rate": round(win_rate, 1),
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
