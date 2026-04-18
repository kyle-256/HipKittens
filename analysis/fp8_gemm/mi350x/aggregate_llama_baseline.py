#!/usr/bin/env python3
"""R25 LLaMA baseline aggregator.

Reads per-run logs from llama_runs/ and produces llama_baseline_r25.json.
For each shape × kind × layout: collects 5 TFLOPS samples, computes
mean/std/median/min/max, parses SNR_dB and PASS/FAIL from log text.
Then computes perf_ratio (mxfp8_v2_median / fp8_median) per layout.
"""
from __future__ import annotations

import json
import os
import re
import statistics
from pathlib import Path

ROOT = Path("/tmp/wt-r25-llama/analysis/fp8_gemm/mi350x")
RUNS = ROOT / "llama_runs"
OUT = Path("/tmp/wt-r25-llama/llama_baseline_r25.json")

# (logical name, M, N, K) — note: 8b_qo == 8b_o, 8b_gateup == 8b_up; 70b same.
SHAPE_DEFS = [
    ("llama8b_qo",     4096, 4096, 4096),  # also covers 8b_o-attn
    ("llama8b_kv",     4096, 1024, 4096),
    ("llama8b_gateup", 4096, 14336, 4096), # also covers 8b_up-mlp
    ("llama8b_down",   4096, 4096, 14336),
    ("llama70b_qo",    4096, 8192, 8192),  # also covers 70b_o-attn
    ("llama70b_kv",    4096, 1024, 8192),
    ("llama70b_gateup",4096, 28672, 8192), # also covers 70b_up-mlp
    ("llama70b_down",  4096, 8192, 28672),
]

LAYOUTS = ["rcr", "rrr", "crr"]
KINDS = ["fp8", "mxfp8"]

# Regex helpers — capture each layout block so we can pull TFLOPS + SNR + PASS/FAIL
# Layout block markers: "--- RCR Layout: " ... "--- RRR Layout: " ... etc.
LAYOUT_HEADER_RE = re.compile(r"^--- (RCR|RRR|CRR) Layout: ", re.MULTILINE)


def parse_run_log(path: Path) -> dict:
    """Return {layout: {tflops, snr_db, ok}} from a single run log."""
    if not path.exists():
        return {}
    text = path.read_text(errors="replace")
    out = {}
    # Split by layout headers
    headers = list(LAYOUT_HEADER_RE.finditer(text))
    for i, m in enumerate(headers):
        lay = m.group(1).lower()
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block = text[start:end]
        tf = re.search(r"TFLOPS:\s*([0-9.]+)", block)
        snr = re.search(r"SNR:\s*([\-0-9.]+)\s*dB", block)
        result = re.search(r"Result:\s*(PASS|FAIL)", block)
        out[lay] = {
            "tflops": float(tf.group(1)) if tf else None,
            "snr_db": float(snr.group(1)) if snr else None,
            "ok": (result.group(1) == "PASS") if result else None,
        }
    return out


def parse_build_log(M, N, K, kind) -> str | None:
    """If build failed, return first 50 lines of log; else None."""
    p = RUNS / f"build_{kind}_{M}x{N}x{K}.log"
    if not p.exists():
        return f"build log missing: {p}"
    text = p.read_text(errors="replace")
    # Successful builds end with hipcc remark output and produce the .so —
    # check artifact existence by name pattern in mi350x dir.
    lines = text.splitlines()
    # Heuristic: if "error:" appears, treat as failed.
    if any("error:" in ln.lower() and "kernel-resource" not in ln for ln in lines):
        return "\n".join(lines[:50])
    return None


def aggregate_layout(samples: list[dict], layout: str) -> dict:
    tfs = [s.get(layout, {}).get("tflops") for s in samples if s.get(layout, {}).get("tflops") is not None]
    snrs = [s.get(layout, {}).get("snr_db") for s in samples if s.get(layout, {}).get("snr_db") is not None]
    oks = [s.get(layout, {}).get("ok") for s in samples if s.get(layout, {}).get("ok") is not None]
    if not tfs:
        return {"n": 0}
    out = {
        "n": len(tfs),
        "tflops_samples": tfs,
        "tflops_median": statistics.median(tfs),
        "tflops_mean": statistics.mean(tfs),
        "tflops_std": statistics.stdev(tfs) if len(tfs) > 1 else 0.0,
        "tflops_min": min(tfs),
        "tflops_max": max(tfs),
    }
    if snrs:
        out["snr_db_median"] = statistics.median(snrs)
        out["snr_db_min"] = min(snrs)
    if oks:
        out["pass_count"] = sum(1 for x in oks if x)
        out["pass_total"] = len(oks)
        out["correctness_pass"] = all(oks)
    return out


def main():
    shapes_out = {}
    perf_gate = []

    # Map shape -> set of "logical names" it covers (for reporting)
    LOGICAL_ALIAS = {
        "llama8b_qo":     ["llama8b_q-attn", "llama8b_o-attn"],
        "llama8b_kv":     ["llama8b_kv-attn"],
        "llama8b_gateup": ["llama8b_gate-mlp", "llama8b_up-mlp"],
        "llama8b_down":   ["llama8b_down-mlp"],
        "llama70b_qo":    ["llama70b_q-attn", "llama70b_o-attn"],
        "llama70b_kv":    ["llama70b_kv-attn"],
        "llama70b_gateup":["llama70b_gate-mlp", "llama70b_up-mlp"],
        "llama70b_down":  ["llama70b_down-mlp"],
    }

    for name, M, N, K in SHAPE_DEFS:
        shape_entry = {
            "M": M, "N": N, "K": K,
            "covers_logical": LOGICAL_ALIAS.get(name, [name]),
        }
        for kind in KINDS:
            be = parse_build_log(M, N, K, kind)
            if be:
                shape_entry[kind] = {"build_error": be}
                continue
            samples = []
            for r in (1, 2, 3, 4, 5):
                fn = RUNS / f"run_{kind}_{name}_r{r}.log"
                samples.append(parse_run_log(fn))
            kind_out = {}
            for lay in LAYOUTS:
                kind_out[lay] = aggregate_layout(samples, lay)
            kind_key = "fp8" if kind == "fp8" else "mxfp8_v2"
            shape_entry[kind_key] = kind_out
        shapes_out[name] = shape_entry

        # perf gate: mxfp8_v2 median vs fp8 median per layout
        for lay in LAYOUTS:
            fp8_block = shape_entry.get("fp8", {}).get(lay, {})
            v2_block = shape_entry.get("mxfp8_v2", {}).get(lay, {})
            fp8_med = fp8_block.get("tflops_median")
            v2_med = v2_block.get("tflops_median")
            if fp8_med and v2_med:
                ratio = v2_med / fp8_med
                gate_pass = ratio >= 0.95
                # also require correctness PASS on V2
                v2_ok = v2_block.get("correctness_pass", False)
                perf_gate.append({
                    "shape": name,
                    "kernel": lay,
                    "mxfp8_v2_median_tflops": round(v2_med, 2),
                    "fp8_median_tflops": round(fp8_med, 2),
                    "ratio": round(ratio, 4),
                    "perf_gate_pass": gate_pass,
                    "v2_correctness_pass": v2_ok,
                    "overall_pass": gate_pass and v2_ok,
                })

    out = {
        "meta": {
            "gpu": "GPU5 (HIP_VISIBLE_DEVICES=5)",
            "branch": "r25-llama-baseline",
            "warmup_fp8": 200, "iters_fp8": 100,
            "warmup_mxfp8": 100, "iters_mxfp8": 100,
            "samples_per_layout": 5,
            "perf_gate_threshold": 0.95,
            "notes": (
                "8 unique build shapes cover 10 LLaMA logical shapes "
                "(Q==O attn; gate==up MLP). MXFP8 uses preshuffle-quant V2 path "
                "(MXFP8_PRESHUFFLE_QUANT=1, V2 RCR/RRR/CRR runtime gates default ON)."
            ),
        },
        "shapes": shapes_out,
        "perf_gate_summary": perf_gate,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")

    # quick textual summary
    print("\n=== PER-SHAPE SUMMARY (median TFLOPS) ===")
    print(f"{'shape':<22} {'kernel':<6} {'fp8':>8} {'v2':>8} {'ratio':>7} {'gate':<6} {'corr':<6}")
    for row in perf_gate:
        print(f"{row['shape']:<22} {row['kernel']:<6} "
              f"{row['fp8_median_tflops']:>8.1f} "
              f"{row['mxfp8_v2_median_tflops']:>8.1f} "
              f"{row['ratio']:>7.4f} "
              f"{'PASS' if row['perf_gate_pass'] else 'FAIL':<6} "
              f"{'PASS' if row['v2_correctness_pass'] else 'FAIL':<6}")


if __name__ == "__main__":
    main()
