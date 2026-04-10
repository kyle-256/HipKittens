#!/usr/bin/env python3
"""Sweep #pragma unroll factors for the RCR 4-wave dynamic kernel main loop."""

import subprocess
import sys
import os
import json
import time
import re

WORKDIR = os.path.dirname(os.path.abspath(__file__))
INC_FILE = os.path.join(WORKDIR, "rcr_4wave_dynamic.inc")

SHAPES = [
    (8192, 16384, 16384),
    (8192, 57344, 8192),
    (16384, 16384, 16384),
    (4096, 28672, 4096),
    (8192, 28672, 4096),
]

UNROLL_FACTORS = [1, 2, 4, "none"]
WARMUP = 20
ITERS = 40
GPU_ID = "4"

ENV = {
    **os.environ,
    "THUNDERKITTENS_ROOT": "/shared_nfs/kyle/HipKittens2",
    "ROCM_PATH": "/opt/rocm",
    "HIPFLAGS": "-DRCR_USE_4WAVE_DYNAMIC=1",
}

ORIGINAL_PRAGMA = "    #pragma unroll 1\n"
PRAGMA_LINE_PATTERN = re.compile(r"^(\s*)#pragma unroll.*$")
FOR_LINE_PATTERN = re.compile(r"^\s*for\s*\(int k = 0; k < ki - 2;")


def read_file():
    with open(INC_FILE, "r") as f:
        return f.readlines()


def write_file(lines):
    with open(INC_FILE, "w") as f:
        f.writelines(lines)


def find_pragma_location(lines):
    """Find the pragma unroll line before the main k-loop."""
    for i, line in enumerate(lines):
        if PRAGMA_LINE_PATTERN.match(line):
            if i + 1 < len(lines) and FOR_LINE_PATTERN.match(lines[i + 1]):
                return i, True
        if FOR_LINE_PATTERN.match(line):
            if i > 0 and not PRAGMA_LINE_PATTERN.match(lines[i - 1]):
                return i, False
    raise RuntimeError("Cannot find the main k-loop in " + INC_FILE)


def set_unroll(factor):
    """Set the unroll pragma. factor=int means #pragma unroll N, factor='none' removes it."""
    lines = read_file()
    idx, has_pragma = find_pragma_location(lines)

    if factor == "none":
        if has_pragma:
            del lines[idx]
    else:
        new_pragma = f"    #pragma unroll {factor}\n"
        if has_pragma:
            lines[idx] = new_pragma
        else:
            lines.insert(idx, new_pragma)

    write_file(lines)


def compile_kernel():
    cmd = (
        "THUNDERKITTENS_ROOT=/shared_nfs/kyle/HipKittens2 "
        "ROCM_PATH=/opt/rocm "
        'HIPFLAGS="-DRCR_USE_4WAVE_DYNAMIC=1" '
        "make clean && "
        "THUNDERKITTENS_ROOT=/shared_nfs/kyle/HipKittens2 "
        "ROCM_PATH=/opt/rocm "
        'HIPFLAGS="-DRCR_USE_4WAVE_DYNAMIC=1" '
        "make -j4"
    )
    print(f"  Compiling...", flush=True)
    t0 = time.time()
    result = subprocess.run(
        cmd, shell=True, cwd=WORKDIR,
        capture_output=True, text=True, timeout=600,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  COMPILE FAILED (exit {result.returncode}, {elapsed:.0f}s)")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        return False
    print(f"  Compiled OK ({elapsed:.0f}s)", flush=True)
    return True


def benchmark_shape(M, N, K):
    bench_code = f"""
import torch, tk_fp8_layouts, json
M, N, K = {M}, {N}, {K}
warmup, iters = {WARMUP}, {ITERS}
A = (torch.randn(M, K, device='cuda') * 0.1).to(torch.float8_e4m3fn)
B = (torch.randn(N, K, device='cuda') * 0.1).to(torch.float8_e4m3fn)
C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
fn = lambda: tk_fp8_layouts.gemm_rcr(A, B, C, 1.0, 1.0, 4)
for _ in range(warmup):
    C.zero_(); fn()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
times = []
for _ in range(iters):
    C.zero_(); torch.cuda.synchronize()
    start.record(); fn(); end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))
avg = sum(times) / len(times)
tflops = 2.0 * M * N * K / (avg * 1e9)
print(json.dumps({{"avg_ms": avg, "tflops": tflops}}))
"""
    env = {**os.environ, "HIP_VISIBLE_DEVICES": GPU_ID}
    result = subprocess.run(
        [sys.executable, "-c", bench_code],
        cwd=WORKDIR, capture_output=True, text=True,
        env=env, timeout=300,
    )
    if result.returncode != 0:
        print(f"    BENCH FAILED: {result.stderr[-500:]}")
        return None
    for line in result.stdout.strip().splitlines():
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    print(f"    Could not parse output: {result.stdout[-500:]}")
    return None


def main():
    original_lines = read_file()

    results = {}  # factor -> {shape_str: tflops}

    for factor in UNROLL_FACTORS:
        label = f"unroll={factor}" if factor != "none" else "no-pragma"
        print(f"\n{'='*60}")
        print(f"Testing: {label}")
        print(f"{'='*60}")

        set_unroll(factor)

        if not compile_kernel():
            results[label] = {f"{M}x{N}x{K}": None for M, N, K in SHAPES}
            continue

        results[label] = {}
        for M, N, K in SHAPES:
            shape_str = f"{M}x{N}x{K}"
            print(f"  Benchmarking {shape_str}...", end=" ", flush=True)
            data = benchmark_shape(M, N, K)
            if data:
                tflops = data["tflops"]
                avg_ms = data["avg_ms"]
                print(f"{tflops:.1f} TFLOPS ({avg_ms:.3f} ms)")
                results[label][shape_str] = tflops
            else:
                print("FAILED")
                results[label][shape_str] = None

    # Restore original file
    write_file(original_lines)
    print("\nRestored original file (unroll 1).")

    # Print comparison table
    shape_strs = [f"{M}x{N}x{K}" for M, N, K in SHAPES]
    labels = [f"unroll={f}" if f != "none" else "no-pragma" for f in UNROLL_FACTORS]

    col_w = 14
    header = f"{'Shape':>24s}" + "".join(f"{l:>{col_w}s}" for l in labels) + f"{'Best':>{col_w}s}"
    print(f"\n{'='*len(header)}")
    print("UNROLL FACTOR SWEEP RESULTS (TFLOPS)")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))

    for shape_str in shape_strs:
        row = f"{shape_str:>24s}"
        vals = []
        for label in labels:
            v = results.get(label, {}).get(shape_str)
            vals.append(v)
            row += f"{v:>{col_w}.1f}" if v else f"{'N/A':>{col_w}s}"
        valid = [(v, l) for v, l in zip(vals, labels) if v is not None]
        if valid:
            best_v, best_l = max(valid, key=lambda x: x[0])
            row += f"{best_l:>{col_w}s}"
        else:
            row += f"{'N/A':>{col_w}s}"
        print(row)

    print("-" * len(header))

    # Geo-mean per factor
    import math
    print(f"\n{'Geo-mean TFLOPS':>24s}", end="")
    for label in labels:
        vals = [results[label][s] for s in shape_strs if results[label].get(s) is not None]
        if vals:
            geo = math.exp(sum(math.log(v) for v in vals) / len(vals))
            print(f"{geo:>{col_w}.1f}", end="")
        else:
            print(f"{'N/A':>{col_w}s}", end="")
    print()

    # Save results
    out_path = os.path.join(WORKDIR, "sweep_unroll_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
