#!/usr/bin/env python3
"""
Multi-N benchmark for HipKittens asymmetric attention (D_QK=192, D_V=128).

For each sequence length N, this script:
  1. Recompiles the kernel with the correct ATTN_N via `make`
  2. Runs a fresh Python subprocess that loads the .so and benchmarks
  3. Collects AITER and HK timings + correctness
  4. Prints a summary table

Usage:
    python bench_d192v128.py                           # default N list
    python bench_d192v128.py 1024 2048 4096 8192       # custom N list
    python bench_d192v128.py --B 8 --H 64 --H_KV 8    # override batch/heads
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# The inner test script that gets written to a temp file and run per-N.
# It loads the freshly-compiled .so, benchmarks, and dumps JSON to stdout.
# ---------------------------------------------------------------------------
INNER_SCRIPT = textwrap.dedent(r'''
import torch
import random
import sys
import os
import json
import math
import importlib

# ---- params from env ----
B     = int(os.environ["BENCH_B"])
N     = int(os.environ["BENCH_N"])
H     = int(os.environ["BENCH_H"])
H_KV  = int(os.environ["BENCH_H_KV"])
D_QK  = int(os.environ.get("BENCH_D_QK", "192"))
D_V   = int(os.environ.get("BENCH_D_V", "128"))
causal = int(os.environ.get("BENCH_CAUSAL", "1"))
num_warmup = int(os.environ.get("BENCH_WARMUP", "200"))
num_iters  = int(os.environ.get("BENCH_ITERS", "100"))

dtype = torch.bfloat16

# ---- helpers ----
def calc_flops(batch, seqlen, nheads, d_qk, d_v, causal):
    """FLOPs for asymmetric attention forward: 2*B*N^2*H*(D_QK+D_V), /2 if causal."""
    flop = 2 * batch * seqlen**2 * nheads * (d_qk + d_v)
    if causal:
        flop //= 2
    return flop

def efficiency(flop, time_ms):
    return (flop / 1e12) / (time_ms / 1e3)

def robustness_check(ref, pred):
    ref = ref.float()
    pred = pred.float()
    diff = (ref - pred).abs()
    denom = ref.abs().clamp_min(1e-6)
    mask = (diff > (0.001 + 0.05 * denom))
    error_count = mask.sum().item()
    numel = ref.numel()
    rel_error = error_count / numel
    l2_error = (diff.pow(2).sum().sqrt() / ref.pow(2).sum().sqrt()).item()
    cos = torch.nn.functional.cosine_similarity(ref.flatten(), pred.flatten(), dim=0).item()
    return {
        "max_abs": diff.max().item(),
        "error_frac": rel_error,
        "l2_error": l2_error,
        "cosine": cos,
        "errors": error_count,
        "total": numel,
    }

total_flop = calc_flops(B, N, H, D_QK, D_V, causal)

start_event = torch.cuda.Event(enable_timing=True)
end_event   = torch.cuda.Event(enable_timing=True)

result = {"N": N, "B": B, "H": H, "H_KV": H_KV, "D_QK": D_QK, "D_V": D_V, "causal": causal}

# ---- AITER benchmark ----
try:
    import aiter
    for _ in range(num_warmup):
        q = torch.randn(B, N, H, D_QK, dtype=dtype, device='cuda')
        k = torch.randn(B, N, H_KV, D_QK, dtype=dtype, device='cuda')
        v = torch.randn(B, N, H_KV, D_V, dtype=dtype, device='cuda')
        out_ref, lse_ref = aiter.flash_attn_func(q, k, v, causal=bool(causal), return_lse=True, deterministic=True)
    timings = []
    torch.manual_seed(42)
    for _ in range(num_iters):
        q = torch.randn(B, N, H, D_QK, dtype=dtype, device='cuda')
        k = torch.randn(B, N, H_KV, D_QK, dtype=dtype, device='cuda')
        v = torch.randn(B, N, H_KV, D_V, dtype=dtype, device='cuda')
        torch.cuda.synchronize()
        start_event.record()
        out_ref, lse_ref = aiter.flash_attn_func(q, k, v, causal=bool(causal), return_lse=True, deterministic=True)
        end_event.record()
        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event))
    avg_ms = sum(timings) / len(timings)
    result["aiter_ms"]    = round(avg_ms, 4)
    result["aiter_tflops"] = round(efficiency(total_flop, avg_ms), 2)
    has_aiter = True
except Exception as e:
    result["aiter_error"] = str(e)
    has_aiter = False

# ---- HK benchmark ----
try:
    # Force reimport of the freshly compiled .so
    sys.path.insert(0, os.environ["BENCH_KERNEL_DIR"])
    if "tk_kernel" in sys.modules:
        del sys.modules["tk_kernel"]
    import tk_kernel

    for _ in range(num_warmup):
        out = torch.zeros(B, N, H, D_V, dtype=dtype, device='cuda')
        lse = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda')
        q = torch.randn(B, N, H, D_QK, dtype=dtype, device='cuda')
        k = torch.randn(B, N, H_KV, D_QK, dtype=dtype, device='cuda')
        v = torch.randn(B, N, H_KV, D_V, dtype=dtype, device='cuda')
        tk_kernel.dispatch_micro(q, k, v, out, lse)

    timings = []
    torch.manual_seed(42)
    for _ in range(num_iters):
        out = torch.zeros(B, N, H, D_V, dtype=dtype, device='cuda')
        lse = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda')
        q = torch.randn(B, N, H, D_QK, dtype=dtype, device='cuda')
        k = torch.randn(B, N, H_KV, D_QK, dtype=dtype, device='cuda')
        v = torch.randn(B, N, H_KV, D_V, dtype=dtype, device='cuda')
        torch.cuda.synchronize()
        start_event.record()
        tk_kernel.dispatch_micro(q, k, v, out, lse)
        end_event.record()
        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event))
    avg_ms = sum(timings) / len(timings)
    result["hk_ms"]    = round(avg_ms, 4)
    result["hk_tflops"] = round(efficiency(total_flop, avg_ms), 2)

    # Correctness check (last iteration tensors, same seed)
    if has_aiter:
        o_check = robustness_check(out_ref, out)
        result["correctness"] = {
            "cosine": round(o_check["cosine"], 6),
            "l2_error": round(o_check["l2_error"], 6),
            "error_pct": round(100 * o_check["error_frac"], 4),
        }
except Exception as e:
    import traceback
    result["hk_error"] = traceback.format_exc()

# ---- Emit JSON on a marker line so we can parse it reliably ----
print("@@BENCH_RESULT@@" + json.dumps(result))
''')


def compile_kernel(n: int, b: int, h: int, h_kv: int,
                   d_qk: int = 192, d_v: int = 128) -> bool:
    """Recompile the forward kernel with the given ATTN_N."""
    cmd = [
        "make", "-C", SCRIPT_DIR, "clean",
    ]
    subprocess.run(cmd, capture_output=True)

    cmd = [
        "make", "-j", "-C", SCRIPT_DIR,
        f"ATTN_N={n}",
        f"ATTN_B={b}",
        f"ATTN_H={h}",
        f"ATTN_H_KV={h_kv}",
        f"ATTN_D_QK={d_qk}",
        f"ATTN_D_V={d_v}",
    ]
    print(f"  Compiling: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        print(f"  COMPILE FAILED (rc={proc.returncode}):")
        print(proc.stderr[-2000:] if proc.stderr else proc.stdout[-2000:])
        return False
    return True


def run_benchmark(n: int, b: int, h: int, h_kv: int,
                  d_qk: int = 192, d_v: int = 128,
                  causal: int = 1,
                  warmup: int = 200, iters: int = 100) -> dict:
    """Run the inner benchmark script in a subprocess."""
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_bench_inner.py", delete=False
    ) as f:
        f.write(INNER_SCRIPT)
        script_path = f.name

    env = os.environ.copy()
    env.update({
        "BENCH_B": str(b),
        "BENCH_N": str(n),
        "BENCH_H": str(h),
        "BENCH_H_KV": str(h_kv),
        "BENCH_D_QK": str(d_qk),
        "BENCH_D_V": str(d_v),
        "BENCH_CAUSAL": str(causal),
        "BENCH_WARMUP": str(warmup),
        "BENCH_ITERS": str(iters),
        "BENCH_KERNEL_DIR": SCRIPT_DIR,
    })

    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=1200, env=env,
        )
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass

    # Parse result
    for line in proc.stdout.splitlines():
        if line.startswith("@@BENCH_RESULT@@"):
            return json.loads(line[len("@@BENCH_RESULT@@"):])

    # If we couldn't parse, return error info
    return {
        "N": n,
        "error": "Could not parse benchmark output",
        "stdout_tail": proc.stdout[-1500:],
        "stderr_tail": proc.stderr[-1500:],
    }


def print_table(results: list):
    """Print a nicely formatted results table."""
    sep = "-" * 110
    print("\n" + sep)
    print(f"{'N':>6}  |  {'AITER ms':>10}  {'AITER TFLOPS':>13}  |  "
          f"{'HK ms':>10}  {'HK TFLOPS':>10}  |  "
          f"{'Speedup':>8}  |  {'Cosine':>8}  {'L2 err':>8}  {'Err%':>7}")
    print(sep)
    for r in results:
        n = r.get("N", "?")
        a_ms = r.get("aiter_ms", "-")
        a_tf = r.get("aiter_tflops", "-")
        h_ms = r.get("hk_ms", "-")
        h_tf = r.get("hk_tflops", "-")

        if isinstance(a_ms, (int, float)) and isinstance(h_ms, (int, float)) and a_ms > 0:
            speedup = f"{a_ms / h_ms:.2f}x"
        else:
            speedup = "-"

        corr = r.get("correctness", {})
        cos_val  = corr.get("cosine", "-")
        l2_val   = corr.get("l2_error", "-")
        err_val  = corr.get("error_pct", "-")

        # Format numeric values
        a_ms_s  = f"{a_ms:.4f}" if isinstance(a_ms, (int, float)) else str(a_ms)
        a_tf_s  = f"{a_tf:.2f}" if isinstance(a_tf, (int, float)) else str(a_tf)
        h_ms_s  = f"{h_ms:.4f}" if isinstance(h_ms, (int, float)) else str(h_ms)
        h_tf_s  = f"{h_tf:.2f}" if isinstance(h_tf, (int, float)) else str(h_tf)
        cos_s   = f"{cos_val:.6f}" if isinstance(cos_val, (int, float)) else str(cos_val)
        l2_s    = f"{l2_val:.6f}" if isinstance(l2_val, (int, float)) else str(l2_val)
        err_s   = f"{err_val:.4f}" if isinstance(err_val, (int, float)) else str(err_val)

        print(f"{n:>6}  |  {a_ms_s:>10}  {a_tf_s:>13}  |  "
              f"{h_ms_s:>10}  {h_tf_s:>10}  |  "
              f"{speedup:>8}  |  {cos_s:>8}  {l2_s:>8}  {err_s:>7}")

        if "hk_error" in r:
            print(f"         HK ERROR: {r['hk_error'][:200]}")
        if "aiter_error" in r:
            print(f"         AITER ERROR: {r['aiter_error'][:200]}")

    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-N benchmark for HK asymmetric attention (D_QK=192, D_V=128)")
    parser.add_argument("seq_lengths", nargs="*", type=int,
                        default=[1024, 2048, 4096, 8192],
                        help="Sequence lengths to benchmark (default: 1024 2048 4096 8192)")
    parser.add_argument("--B", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--H", type=int, default=64, help="Query heads (default: 64)")
    parser.add_argument("--H_KV", type=int, default=8, help="KV heads (default: 8)")
    parser.add_argument("--D_QK", type=int, default=192, help="Q/K head dim (default: 192)")
    parser.add_argument("--D_V", type=int, default=128, help="V head dim (default: 128)")
    parser.add_argument("--causal", type=int, default=1, help="Causal mask (default: 1)")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Timing iterations")
    args = parser.parse_args()

    print("=" * 110)
    print(f"HipKittens Asymmetric Attention Benchmark  (D_QK={args.D_QK}, D_V={args.D_V})")
    print(f"B={args.B}  H={args.H}  H_KV={args.H_KV}  causal={args.causal}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print(f"Warmup={args.warmup}  Iters={args.iters}")
    print("=" * 110)

    results = []
    for n in args.seq_lengths:
        print(f"\n>>> N = {n}")
        print(f"  Step 1: Recompiling kernel with ATTN_N={n} ...")
        ok = compile_kernel(n, args.B, args.H, args.H_KV, args.D_QK, args.D_V)
        if not ok:
            results.append({"N": n, "hk_error": "compilation failed"})
            continue

        print(f"  Step 2: Running benchmark ...")
        r = run_benchmark(
            n, args.B, args.H, args.H_KV,
            args.D_QK, args.D_V, args.causal,
            args.warmup, args.iters,
        )
        results.append(r)
        # Print inline summary
        h_tf = r.get("hk_tflops", "ERR")
        a_tf = r.get("aiter_tflops", "ERR")
        print(f"  Result: HK={h_tf} TFLOPS, AITER={a_tf} TFLOPS")

    print_table(results)

    # Cleanup temp script
    tmp_path = os.path.join(SCRIPT_DIR, "_bench_inner.py")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


if __name__ == "__main__":
    main()
