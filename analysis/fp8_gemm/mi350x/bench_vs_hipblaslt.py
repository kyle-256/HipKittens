"""
FP8 Per-Tensor GEMM: HipKittens vs hipBLASLt comparison.

Usage:
    HIP_VISIBLE_DEVICES=7 python3 bench_vs_hipblaslt.py --mode smoke
    HIP_VISIBLE_DEVICES=7 python3 bench_vs_hipblaslt.py --mode full
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime

import torch

torch.manual_seed(42)

sys.path.insert(0, os.path.dirname(__file__))
import tk_fp8_layouts

from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import (
    GEMMFP8HipBLASLtBackend,
    ScalingGranularity,
)

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

LAYOUT_MAP = {
    "rcr": {"trans_a": False, "trans_b": True},
    "rrr": {"trans_a": False, "trans_b": False},
    "crr": {"trans_a": True,  "trans_b": False},
}


def gen_gemm_test_cases(config):
    seq = config["seqlen"]
    hs = config["hidden_size"]
    inter = config["intermediate_size"]
    nah = config["num_attention_heads"]
    nkv = config["num_key_value_heads"]
    hd = config["head_dim"]
    return [
        ("attn_qkv",    seq, int((nah + 2 * nkv) * hd), hs),
        ("attn_out",    seq, hs, hs),
        ("mlp_gate_up", seq, int(2 * inter), hs),
        ("mlp_down",    seq, hs, inter),
    ]


TK_FNS = {
    "rcr": tk_fp8_layouts.gemm_rcr,
    "rrr": tk_fp8_layouts.gemm_rrr,
    "crr": tk_fp8_layouts.gemm_crr,
}

from autotune import AutotunedGEMM
_autotuner = AutotunedGEMM(verbose=True)


def make_tensors(M, N, K, layout):
    if layout == "rcr":
        A = (torch.randn(M, K, device="cuda") * 0.1).to(torch.float8_e4m3fn)
        B = (torch.randn(N, K, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    elif layout == "rrr":
        A = (torch.randn(M, K, device="cuda") * 0.1).to(torch.float8_e4m3fn)
        B = (torch.randn(K, N, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    else:
        A = (torch.randn(K, M, device="cuda") * 0.1).to(torch.float8_e4m3fn)
        B = (torch.randn(K, N, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    return A, B


def bench_hipkittens(A, B, M, N, K, layout, warmup, iters):
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    fn = TK_FNS[layout]
    gm = _autotuner.get_group_m(M, N, K, layout, A, B)
    run = lambda: fn(A, B, C, 1.0, 1.0, gm)

    for _ in range(warmup):
        C.zero_()
        run()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        C.zero_()
        torch.cuda.synchronize()
        start_ev.record()
        run()
        end_ev.record()
        torch.cuda.synchronize()
        times.append(start_ev.elapsed_time(end_ev))

    avg_ms = sum(times) / len(times)
    return avg_ms


def bench_hipblaslt(A, B, M, N, K, layout, warmup, iters):
    sa = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    sb = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    trans_a = LAYOUT_MAP[layout]["trans_a"]
    trans_b = LAYOUT_MAP[layout]["trans_b"]

    hipblaslt_fn = torch.ops.primus_turbo_cpp_extension.hipblaslt_gemm_fp8
    run = lambda: hipblaslt_fn(A, sa, B, sb, torch.bfloat16, trans_a, trans_b, False, "TENSORWISE")

    for _ in range(warmup):
        run()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start_ev.record()
        run()
        end_ev.record()
        torch.cuda.synchronize()
        times.append(start_ev.elapsed_time(end_ev))

    avg_ms = sum(times) / len(times)
    return avg_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--layouts", type=str, default=None)
    parser.add_argument("--mbs", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--output", "-o", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "smoke":
        warmup = args.warmup or 10
        iters = args.iters or 20
        layouts = (args.layouts or "rcr").split(",")
        mbs_list = [int(x) for x in (args.mbs or "1").split(",")]
    else:
        warmup = args.warmup or 20
        iters = args.iters or 50
        layouts = (args.layouts or "rcr,rrr,crr").split(",")
        mbs_list = [int(x) for x in (args.mbs or "1,2").split(",")]

    models = (args.models or ",".join(DenseModelConfigs.keys())).split(",")
    models = [m.strip() for m in models if m.strip() in DenseModelConfigs]

    test_shapes = []
    for model_name in models:
        config = DenseModelConfigs[model_name]
        cases = gen_gemm_test_cases(config)
        for mbs in mbs_list:
            for op_name, seq, n, k in cases:
                M = seq * mbs
                test_shapes.append((model_name, op_name, mbs, M, n, k))

    total = len(test_shapes) * len(layouts)
    print(f"FP8 Per-Tensor GEMM: HipKittens vs hipBLASLt ({args.mode})")
    print(f"  Models: {', '.join(models)}")
    print(f"  Layouts: {', '.join(layouts)}, MBS: {mbs_list}")
    print(f"  Warmup: {warmup}, Iters: {iters}")
    print(f"  Total: {total} configurations\n")

    results = []
    idx = 0

    for model_name, op_name, mbs, M, N, K in test_shapes:
        for layout in layouts:
            idx += 1
            flops = 2.0 * M * N * K
            tag = f"[{idx}/{total}] {model_name}/{op_name} MBS={mbs} ({M},{N},{K}) {layout.upper()}"

            A, B = make_tensors(M, N, K, layout)

            try:
                tk_ms = bench_hipkittens(A, B, M, N, K, layout, warmup, iters)
                tk_tflops = flops / (tk_ms * 1e9)
            except Exception as e:
                print(f"{tag} TK ERROR: {e}")
                tk_ms, tk_tflops = float("inf"), 0

            try:
                bl_ms = bench_hipblaslt(A, B, M, N, K, layout, warmup, iters)
                bl_tflops = flops / (bl_ms * 1e9)
            except Exception as e:
                print(f"{tag} BL ERROR: {e}")
                bl_ms, bl_tflops = float("inf"), 0

            speedup = bl_ms / tk_ms if tk_ms > 0 and bl_ms < float("inf") else 0
            winner = "TK" if tk_ms < bl_ms else "BL"

            print(
                f"{tag}  "
                f"TK={tk_tflops:7.1f}  BL={bl_tflops:7.1f}  "
                f"ratio={speedup:.3f}x  [{winner}]"
            )

            results.append({
                "model": model_name, "op": op_name, "mbs": mbs,
                "M": M, "N": N, "K": K, "layout": layout,
                "tk_ms": round(tk_ms, 4), "tk_tflops": round(tk_tflops, 2),
                "bl_ms": round(bl_ms, 4), "bl_tflops": round(bl_tflops, 2),
                "speedup": round(speedup, 4),
            })

            del A, B
            torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'=' * 110}")
    print(
        f"{'Model':<18} {'Op':<14} {'MBS':>3} "
        f"{'M':>6} {'N':>6} {'K':>6} "
        f"{'Lay':>4} {'TK TFLOPS':>10} {'BL TFLOPS':>10} {'TK/BL':>7}"
    )
    print("-" * 110)
    for r in results:
        ratio = r["speedup"]
        marker = " **" if ratio > 1.05 else (" !!" if ratio < 0.95 else "")
        print(
            f"{r['model']:<18} {r['op']:<14} {r['mbs']:>3} "
            f"{r['M']:>6} {r['N']:>6} {r['K']:>6} "
            f"{r['layout']:>4} {r['tk_tflops']:>10.2f} {r['bl_tflops']:>10.2f} "
            f"{ratio:>6.3f}x{marker}"
        )
    print("=" * 110)

    # Aggregate stats
    for layout in layouts:
        lr = [r for r in results if r["layout"] == layout]
        if not lr:
            continue
        tk_avg = sum(r["tk_tflops"] for r in lr) / len(lr)
        bl_avg = sum(r["bl_tflops"] for r in lr) / len(lr)
        wins = sum(1 for r in lr if r["speedup"] > 1.0)
        geo_speedup = math.exp(sum(math.log(r["speedup"]) for r in lr if r["speedup"] > 0) / len(lr))
        print(
            f"\n{layout.upper()}: TK avg={tk_avg:.1f}  BL avg={bl_avg:.1f}  "
            f"geo-mean speedup={geo_speedup:.3f}x  TK wins {wins}/{len(lr)}"
        )

    outfile = args.output
    if outfile is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outfile = f"bench_vs_hipblaslt_{args.mode}_{ts}.json"
    with open(outfile, "w") as f:
        json.dump({"mode": args.mode, "results": results}, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
