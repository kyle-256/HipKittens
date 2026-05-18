#!/usr/bin/env python3
"""Loser-shape-only score for HK blockwise FP8 GEMM (MI300X / gfx942).

Subset of `_metric_blockwise_fp8_target_shapes.py` that only scores the
14 (shape, section) pairs where HK currently runs *below* THIS-machine
Triton baseline (per round-63 PROBE comparison at score 906).

Use case: focus the auto-optimize daemon on the hard cases instead of
the full 54-pair metric (where many shapes are already capped at 1.0).
A single +30 T win on a loser shape moves THIS metric ~7% (1/14) instead
of ~0.6% (1/54 × 1/3).

Score formula:
  per-pair progress[i] = min(hk_tflops / (triton_tflops × 1.25), 1.0)
  score                = round(mean(progress) × 1000)

Stdout contract: last non-empty line is the integer score.

Env vars:
  METRIC_TRIALS  default 5  : per-shape independent perf runs (min wins)
  METRIC_BUILD   default 1  : if 0, skip make
  METRIC_VERBOSE default 1  : if 0, suppress per-shape table
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _shapes_target import (
    SHAPES, SECTIONS, LAYOUT_BY_SECTION, TARGET_MULTIPLIER,
    hk_fp8_unsupported_reason,
)

KERNEL_DIR = os.path.abspath(os.path.join(
    HERE, "..", "kernels", "gemm", "fp8fp32", "mi300x", "blockwise_8192"))

NUM_TRIALS = int(os.environ.get("METRIC_TRIALS", "5"))
DO_BUILD   = int(os.environ.get("METRIC_BUILD",   "1"))
VERBOSE    = int(os.environ.get("METRIC_VERBOSE", "1"))

HARD_SNR = 49.0    # current build at 49.59 dB; <0.6 dB margin trips this gate early
COMPILE_FAIL_PENALTY = 5000
CORRECTNESS_PENALTY  = 2000

# Loser pairs where HK runs *below* THIS-machine Triton.
#
# Round-100 PRUNE: the 5 wgrad pairs that consistently land 103-112% Triton
# across recent metric runs (LFM2.Down.M16384, Qwen3.GateUP.M16384,
# Qwen3.GateUP.M32768, DeepSeek.Down.M16384, DeepSeek.Down.M32768) have been
# DROPPED — they're already winning, so each daemon round spent re-measuring
# them is wasted budget. The 9 pairs below are the actual hard cases. If a
# future round lifts one of these above ~105% Triton consistently, drop it
# from this set so the daemon's focus stays narrow.
#
# Original 14-pair set is preserved in git history (round-63 to round-99).
LOSER_PAIRS = {
    # Round-100 PRUNE (revised): all (shape, section) pairs across rcr/rrr/crr
    # = fwd/dgrad/wgrad where HK/Triton < 105% per the most-recent full-target
    # metric run. ≥105% pairs are excluded — they're already winning by the
    # 5%+ margin we'd want from a "loser" definition. Refresh this set with
    # `python3 scripts/_metric_blockwise_fp8_target_shapes.py` and pick rows
    # with HK/Triton < 1.05.
    #
    # 9 fwd losers (95-101% Triton, concentrated in big-K + big-M)
    ("LFM2-8B-A1B.GateUP.M32768.N3584.K2048",     "fwd"),    # 98%
    ("Qwen3-235B-A22B.GateUP.M16384.N3072.K4096", "fwd"),    # 100%
    ("Qwen3-235B-A22B.GateUP.M32768.N3072.K4096", "fwd"),    # 96%
    ("Qwen3-235B-A22B.Down.M32768.N4096.K1536",   "fwd"),    # 99%
    ("DeepSeek-V3.GateUP.M8192.N4096.K7168",      "fwd"),    # 95%
    ("DeepSeek-V3.GateUP.M16384.N4096.K7168",     "fwd"),    # 96%
    ("DeepSeek-V3.GateUP.M32768.N4096.K7168",     "fwd"),    # 95%
    ("DeepSeek-V3.Down.M16384.N7168.K2048",       "fwd"),    # 101%
    ("DeepSeek-V3.Down.M32768.N7168.K2048",       "fwd"),    # 98%
    # 3 wgrad losers (round-124 dropped DS K=7168 prematurely — recent runs vary 88-110%, restore)
    ("LFM2-8B-A1B.Down.M32768.N2048.K1792",       "wgrad"),  # 97%
    ("Qwen3-235B-A22B.GateUP.M32768.N3072.K4096", "wgrad"),  # 103-104%
    ("DeepSeek-V3.GateUP.M32768.N4096.K7168",     "wgrad"),  # 88-110% (variable)
}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run_make() -> tuple[bool, int, int]:
    r = subprocess.run(
        ["make"], cwd=KERNEL_DIR, capture_output=True, text=True, timeout=300
    )
    output = r.stdout + r.stderr
    if r.returncode != 0:
        log("[METRIC] BUILD FAILED:")
        log(output[-2000:])
        return False, 0, 0
    vgpr = spill = 0
    for line in output.splitlines():
        m = re.search(r"VGPRs:\s+(\d+)", line)
        if m: vgpr = max(vgpr, int(m.group(1)))
        m = re.search(r"VGPRs Spill:\s+(\d+)", line)
        if m: spill = max(spill, int(m.group(1)))
    log(f"[METRIC] build OK  vgpr={vgpr}  spill={spill}")
    return True, vgpr, spill


def run_correctness_check() -> tuple[bool, float]:
    """Single-shape SNR gate via the kernel's bundled test_python.py."""
    env = dict(os.environ)
    env["BW_CHECK"] = "1"
    env["BW_WARMUP"] = "5"
    env["BW_ITERS"] = "5"
    r = subprocess.run(
        ["python3", "test_python.py"],
        cwd=KERNEL_DIR, env=env, capture_output=True, text=True, timeout=180,
    )
    snr = -1.0
    for line in (r.stdout + r.stderr).splitlines():
        m = re.search(r"SNR:\s+([\-\d.nainf]+)", line)
        if m:
            try: snr = float(m.group(1))
            except ValueError: snr = -1.0
            break
    ok = snr >= HARD_SNR
    log(f"[METRIC] correctness gate: SNR={snr:.2f} dB  {'PASS' if ok else 'FAIL'}")
    return ok, snr


def measure_hk_one(shape, section: str) -> tuple[float, str]:
    reason = hk_fp8_unsupported_reason(shape, section)
    if reason is not None:
        return 0.0, f"SKIP: {reason}"
    env = dict(os.environ)
    env["BW_CHECK"] = "0"
    env["BW_WARMUP"] = "50"
    env["BW_ITERS"] = "50"
    env["BW_M"] = str(shape.M)
    env["BW_N"] = str(shape.N)
    env["BW_K"] = str(shape.K)
    env["BW_SECTION"] = section
    trials: list[float] = []
    for _ in range(NUM_TRIALS):
        r = subprocess.run(
            ["python3", "test_python.py"],
            cwd=KERNEL_DIR, env=env, capture_output=True, text=True, timeout=180,
        )
        for line in (r.stdout + r.stderr).splitlines():
            m = re.search(r"Performance:\s+([\d.]+)\s+TFLOPS", line)
            if m:
                trials.append(float(m.group(1)))
                break
    if not trials:
        return 0.0, "MEASURE_FAIL"
    return min(trials), f"OK ({len(trials)}/{NUM_TRIALS} trials)"


def main() -> None:
    if DO_BUILD:
        ok, _, _ = run_make()
        if not ok:
            print(-COMPILE_FAIL_PENALTY)
            return

    snr_ok, _ = run_correctness_check()
    if not snr_ok:
        log(f"[METRIC] correctness gate failed → returning {-CORRECTNESS_PENALTY}")
        print(-CORRECTNESS_PENALTY)
        return

    # Walk in stable order (model.name then section) for reproducible output.
    rows: list[dict[str, Any]] = []
    for shape in SHAPES:
        for section in SECTIONS:
            if (shape.name, section) not in LOSER_PAIRS:
                continue
            tflops, status = measure_hk_one(shape, section)
            target = shape.target(section)
            progress = min(tflops / target, 1.0) if target > 0 else 0.0
            rows.append({
                "name": shape.name, "section": section,
                "listed": shape.listed(section),
                "triton": shape.triton(section),
                "target": target,
                "tflops": tflops, "progress": progress, "status": status,
            })

    if not rows:
        log("[METRIC] no loser pairs matched (LOSER_PAIRS empty?)")
        print(0)
        return

    overall = sum(r["progress"] for r in rows) / len(rows)
    score = round(overall * 1000)

    if VERBOSE:
        print_table(rows, overall, score)

    print(score)


def print_table(rows, overall, score) -> None:
    log("")
    log(f"# Per-shape × per-section results (LOSER SUBSET — {len(LOSER_PAIRS)} pairs)")
    log("")
    log(f"  Target = {TARGET_MULTIPLIER}× THIS-machine Triton baseline.")
    log(f"  Score = 1000 iff every loser pair hits its 1.25× target.")
    log("")
    log(f"  {'Shape':<46} {'Sec':<6} {'Triton':>7} {'Target':>7} {'HK':>7} {'%Tri':>5} {'Prog':>5}  Status")
    log("  " + "-" * 130)
    for r in rows:
        pct_tri = r["tflops"] / r["triton"] * 100 if r["triton"] > 0 else 0
        log(f"  {r['name']:<46} {r['section']:<6} "
            f"{r['triton']:>7.1f} {r['target']:>7.1f} {r['tflops']:>7.1f} "
            f"{pct_tri:>4.0f}% {r['progress']*100:>4.0f}%  {r['status']}")
    log("")
    log(f"  overall   {overall*100:>5.1f}%   →  score = {score}")


if __name__ == "__main__":
    main()
