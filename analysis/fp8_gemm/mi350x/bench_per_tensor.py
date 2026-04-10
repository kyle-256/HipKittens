"""
FP8 Per-Tensor GEMM Benchmark — covers all DenseModel shapes from Primus-Turbo.

Usage:
    # Quick smoke test (1 shape per model, RCR only)
    HIP_VISIBLE_DEVICES=7 python3 bench_per_tensor.py --mode smoke

    # Full benchmark (all shapes × all layouts × all batch sizes)
    HIP_VISIBLE_DEVICES=7 python3 bench_per_tensor.py --mode full

    # Specific model / layout / batch size
    HIP_VISIBLE_DEVICES=7 python3 bench_per_tensor.py --models Llama-3.1-8B --layouts rcr,rrr,crr --mbs 1

    # Custom shape
    HIP_VISIBLE_DEVICES=7 python3 bench_per_tensor.py --shapes 4096,4096,4096
"""

import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime

import torch

torch.manual_seed(42)

import tk_fp8_layouts

DenseModelConfigs = {
    "Llama-2-7B": {
        "seqlen": 4096, "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32, "num_key_value_heads": 32,
        "head_dim": 128,
    },
    "Llama-2-70B": {
        "seqlen": 4096, "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_attention_heads": 64, "num_key_value_heads": 8,
        "head_dim": 128,
    },
    "Llama-3.1-8B": {
        "seqlen": 8192, "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32, "num_key_value_heads": 8,
        "head_dim": 128,
    },
    "Llama-3.1-405B": {
        "seqlen": 8192, "hidden_size": 16384,
        "intermediate_size": 53248,
        "num_attention_heads": 128, "num_key_value_heads": 8,
        "head_dim": 128,
    },
    "Qwen2.5-7B": {
        "seqlen": 8192, "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_attention_heads": 28, "num_key_value_heads": 4,
        "head_dim": 128,
    },
    "Qwen2.5-72B": {
        "seqlen": 8192, "hidden_size": 8192,
        "intermediate_size": 29568,
        "num_attention_heads": 64, "num_key_value_heads": 8,
        "head_dim": 128,
    },
    "Mistral-7B": {
        "seqlen": 4096, "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32, "num_key_value_heads": 8,
        "head_dim": 128,
    },
}

BATCH_SIZE_LIST = [1, 2, 4]


def gen_gemm_test_cases(config):
    seq = config["seqlen"]
    hs = config["hidden_size"]
    inter = config["intermediate_size"]
    nah = config["num_attention_heads"]
    nkv = config["num_key_value_heads"]
    hd = config["head_dim"]
    return [
        ("attn_qkv",  seq, int((nah + 2 * nkv) * hd), hs),
        ("attn_out",   seq, hs, hs),
        ("mlp_gate_up", seq, int(2 * inter), hs),
        ("mlp_down",   seq, hs, inter),
    ]


LAYOUT_FNS = {
    "rcr": tk_fp8_layouts.gemm_rcr,
    "rrr": tk_fp8_layouts.gemm_rrr,
    "crr": tk_fp8_layouts.gemm_crr,
}


def generate_fp8_matrix(rows, cols):
    x = torch.randn(rows, cols, dtype=torch.float32, device="cuda") * 0.1
    return x.to(torch.float8_e4m3fn)


def compute_snr_db(test, ref):
    t = test.float()
    r = ref.float()
    sig = (r * r).sum().item()
    noise = ((t - r) * (t - r)).sum().item()
    if noise == 0.0:
        return float("inf")
    if sig == 0.0:
        return float("-inf")
    return 10.0 * math.log10(sig / noise)


def benchmark_one(M, N, K, layout, warmup, iters, check, det_runs):
    gemm_fn = LAYOUT_FNS[layout]
    scale_a, scale_b = 1.0, 1.0

    if layout == "rcr":
        A = generate_fp8_matrix(M, K)
        B = generate_fp8_matrix(N, K)
        C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        ref_fn = lambda: A[:M, :K].float() @ B[:N, :K].float().T
    elif layout == "rrr":
        A = generate_fp8_matrix(M, K)
        B = generate_fp8_matrix(K, N)
        C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        ref_fn = lambda: A[:M, :K].float() @ B[:K, :N].float()
    else:  # crr
        A = generate_fp8_matrix(K, M)
        B = generate_fp8_matrix(K, N)
        C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        ref_fn = lambda: A[:K, :M].float().T @ B[:K, :N].float()

    run = lambda: gemm_fn(A, B, C, scale_a, scale_b)

    for _ in range(warmup):
        C.zero_()
        run()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(iters):
        C.zero_()
        torch.cuda.synchronize()
        start_ev.record()
        run()
        end_ev.record()
        torch.cuda.synchronize()
        timings.append(start_ev.elapsed_time(end_ev))

    avg_ms = sum(timings) / len(timings)
    flops = 2.0 * M * N * K
    tflops = flops / (avg_ms * 1e9)

    result = {
        "M": M, "N": N, "K": K, "layout": layout,
        "avg_ms": round(avg_ms, 4),
        "tflops": round(tflops, 2),
    }

    if check:
        C.zero_()
        run()
        C_ref = ref_fn()
        snr = compute_snr_db(C[:M, :N], C_ref)
        result["snr_db"] = round(snr, 2)
        result["snr_ok"] = snr > 48.0

        det_ok = True
        if det_runs > 1:
            C.zero_()
            run()
            ref_out = C[:M, :N].clone()
            for _ in range(det_runs - 1):
                C.zero_()
                run()
                if not torch.equal(C[:M, :N], ref_out):
                    det_ok = False
                    break
        result["deterministic"] = det_ok
        result["pass"] = result["snr_ok"] and det_ok
    else:
        result["pass"] = True

    return result


def print_results_table(results):
    if not results:
        return

    header = (
        f"{'Model':<18} {'Op':<14} {'MBS':>4} "
        f"{'M':>6} {'N':>6} {'K':>6} "
        f"{'Layout':>6} {'ms':>8} {'TFLOPS':>8} "
        f"{'SNR':>8} {'Det':>4} {'Pass':>5}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        snr_str = f"{r.get('snr_db', '-'):>8}" if "snr_db" in r else f"{'--':>8}"
        det_str = "Y" if r.get("deterministic", True) else "N"
        pass_str = "PASS" if r.get("pass", True) else "FAIL"
        print(
            f"{r.get('model', ''):.<18} {r.get('op', ''):.<14} {r.get('mbs', ''):>4} "
            f"{r['M']:>6} {r['N']:>6} {r['K']:>6} "
            f"{r['layout']:>6} {r['avg_ms']:>8.3f} {r['tflops']:>8.2f} "
            f"{snr_str} {det_str:>4} {pass_str:>5}"
        )

    print("=" * len(header))

    pass_count = sum(1 for r in results if r.get("pass", True))
    fail_count = len(results) - pass_count
    avg_tflops = sum(r["tflops"] for r in results) / len(results) if results else 0
    print(f"\nTotal: {len(results)} tests | PASS: {pass_count} | FAIL: {fail_count}")
    print(f"Average TFLOPS: {avg_tflops:.2f}")

    for layout in ("rcr", "rrr", "crr"):
        lr = [r for r in results if r["layout"] == layout]
        if lr:
            avg = sum(r["tflops"] for r in lr) / len(lr)
            print(f"  {layout.upper()} avg: {avg:.2f} TFLOPS ({len(lr)} shapes)")


def main():
    parser = argparse.ArgumentParser(description="FP8 Per-Tensor GEMM Benchmark")
    parser.add_argument("--mode", choices=["smoke", "full", "custom"],
                        default="smoke")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (default: all)")
    parser.add_argument("--layouts", type=str, default=None,
                        help="Comma-separated layouts: rcr,rrr,crr (default: all)")
    parser.add_argument("--mbs", type=str, default=None,
                        help="Comma-separated batch sizes (default: 1,2,4)")
    parser.add_argument("--shapes", type=str, default=None,
                        help="Custom M,N,K shapes (e.g. '4096,4096,4096;8192,8192,8192')")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--check", action="store_true", default=None)
    parser.add_argument("--no-check", dest="check", action="store_false")
    parser.add_argument("--det-runs", type=int, default=None)
    parser.add_argument("--output", "-o", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "smoke":
        warmup = args.warmup or 5
        iters = args.iters or 10
        check = True if args.check is None else args.check
        det_runs = args.det_runs or 2
        layouts = (args.layouts or "rcr").split(",")
        mbs_list = [int(x) for x in (args.mbs or "1").split(",")]
    elif args.mode == "full":
        warmup = args.warmup or 50
        iters = args.iters or 100
        check = True if args.check is None else args.check
        det_runs = args.det_runs or 3
        layouts = (args.layouts or "rcr,rrr,crr").split(",")
        mbs_list = [int(x) for x in (args.mbs or "1,2,4").split(",")]
    else:
        warmup = args.warmup or 20
        iters = args.iters or 50
        check = True if args.check is None else args.check
        det_runs = args.det_runs or 2
        layouts = (args.layouts or "rcr").split(",")
        mbs_list = [1]

    models = (args.models or ",".join(DenseModelConfigs.keys())).split(",")
    models = [m.strip() for m in models if m.strip() in DenseModelConfigs]

    test_shapes = []

    if args.shapes:
        for s in args.shapes.split(";"):
            parts = [int(x) for x in s.split(",")]
            M, N, K = parts[0], parts[1], parts[2]
            test_shapes.append(("custom", "custom", 1, M, N, K))
    else:
        for model_name in models:
            config = DenseModelConfigs[model_name]
            cases = gen_gemm_test_cases(config)
            for mbs in mbs_list:
                for op_name, seq, n, k in cases:
                    M = seq * mbs
                    test_shapes.append((model_name, op_name, mbs, M, n, k))

    total = len(test_shapes) * len(layouts)
    print(f"FP8 Per-Tensor GEMM Benchmark ({args.mode} mode)")
    print(f"  Models: {', '.join(models)}")
    print(f"  Layouts: {', '.join(layouts)}")
    print(f"  Batch sizes: {mbs_list}")
    print(f"  Warmup: {warmup}, Iters: {iters}, Check: {check}, Det runs: {det_runs}")
    print(f"  Total test configurations: {total}")
    print()

    results = []
    test_id = 0

    for model_name, op_name, mbs, M, N, K in test_shapes:
        for layout in layouts:
            test_id += 1
            tag = f"[{test_id}/{total}] {model_name}/{op_name} MBS={mbs} ({M},{N},{K}) {layout.upper()}"
            print(f"{tag} ...", end=" ", flush=True)

            try:
                r = benchmark_one(M, N, K, layout, warmup, iters, check, det_runs)
                r["model"] = model_name
                r["op"] = op_name
                r["mbs"] = mbs
                status = "PASS" if r.get("pass", True) else "FAIL"
                snr_str = f"SNR={r.get('snr_db', '--')}" if check else ""
                print(f"{r['tflops']:.2f} TFLOPS  {r['avg_ms']:.3f} ms  {snr_str}  {status}")
                results.append(r)
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "model": model_name, "op": op_name, "mbs": mbs,
                    "M": M, "N": N, "K": K, "layout": layout,
                    "avg_ms": 0, "tflops": 0, "pass": False,
                    "error": str(e),
                })

    print_results_table(results)

    outfile = args.output
    if outfile is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outfile = f"bench_per_tensor_{args.mode}_{ts}.json"
    with open(outfile, "w") as f:
        json.dump({"mode": args.mode, "results": results}, f, indent=2)
    print(f"\nResults saved to {outfile}")

    fails = [r for r in results if not r.get("pass", True)]
    if fails:
        print(f"\n{len(fails)} FAILURES:")
        for r in fails:
            err = r.get("error")
            if err is None:
                snr_val = r.get("snr_db", "?")
                det_val = r.get("deterministic", "?")
                err = f"SNR={snr_val} det={det_val}"
            print(f"  ({r['M']},{r['N']},{r['K']}) {r['layout'].upper()}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
