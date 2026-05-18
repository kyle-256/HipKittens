#!/usr/bin/env python3
"""Multi-shape, multi-section score for HK blockwise FP8 GEMM (MI300X / gfx942).

Drives the outer optimization loop (auto_optimize.py / fleet driver) toward
the production target list of 18 shapes × 3 layouts (RCR / RRR / CRR), one
single integer score per Primus-Turbo's `_metric_gpt_oss_fp8_kernel.py`
contract.

Kernel under test:
  kernels/gemm/fp8fp32/mi300x/blockwise_8192/blockwise.cpp

Score formula (linear per-section, mean over sections):

  per-shape progress[s][i] = min(hk_tflops / (listed_tflops × 1.25), 1.0)
  section_progress[s]      = mean(progress[s] over shapes in section)
  overall_progress         = mean(section_progress for s in fwd, dgrad, wgrad)
  score                    = round(overall_progress × 1000)

  * Score = 1000 iff every (shape, section) hits 1.25× listed.
  * Unsupported (shape, section) → progress 0 (RRR/CRR not implemented;
    M/N/K ≠ kernel's hardcoded 8192³). Reasons logged.
  * Build / correctness fail → integer score is heavily negative.

Stdout contract: last non-empty line is the integer score.
All diagnostics + per-shape table go to stderr.

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
    """Try to measure HK FP8 kernel TFLOPS for (shape, section).
    Returns (tflops, status). Currently ONLY the kernel's hardcoded shape
    (8192³, RCR) is measurable; everything else returns 0.0 + SKIP reason."""
    reason = hk_fp8_unsupported_reason(shape, section)
    if reason is not None:
        return 0.0, f"SKIP: {reason}"

    # Run the kernel's bundled test_python.py N times. Pass shape via
    # BW_M/BW_N/BW_K env vars; test_python.py reads them (round-1 wired).
    # BW_SECTION selects fwd/dgrad/wgrad branch (round-8 wired dgrad through
    # the fwd kernel via Python-side B-transpose).
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
        ok, vgpr, spill = run_make()
        if not ok:
            print(-COMPILE_FAIL_PENALTY)
            return

    snr_ok, snr = run_correctness_check()
    if not snr_ok:
        log(f"[METRIC] correctness gate failed → returning {-CORRECTNESS_PENALTY}")
        print(-CORRECTNESS_PENALTY)
        return

    rows: list[dict[str, Any]] = []
    for shape in SHAPES:
        for section in SECTIONS:
            tflops, status = measure_hk_one(shape, section)
            target = shape.target(section)            # = TARGET_MULTIPLIER × triton
            progress = min(tflops / target, 1.0) if target > 0 else 0.0
            rows.append({
                "name": shape.name, "section": section,
                "listed": shape.listed(section),
                "triton": shape.triton(section),
                "target": target,
                "tflops": tflops, "progress": progress, "status": status,
            })

    section_progress: dict[str, float] = {}
    for s in SECTIONS:
        items = [r for r in rows if r["section"] == s]
        section_progress[s] = sum(r["progress"] for r in items) / len(items) if items else 0.0

    overall = sum(section_progress.values()) / len(section_progress)
    score = round(overall * 1000)

    if VERBOSE:
        print_table(rows, section_progress, overall, score)

    print(score)


def print_table(rows, section_progress, overall, score) -> None:
    log("")
    log("# Per-shape × per-section results")
    log("")
    log(f"  Target = {TARGET_MULTIPLIER}× THIS-machine Triton baseline (not listed).")
    log(f"  Score = 1000 iff every (shape, section) hits its target.")
    log("")
    log(f"  {'Shape':<46} {'Sec':<6} {'Listed':>7} {'Triton':>7} {'Target':>7} {'HK':>7} {'Prog':>5}  Status")
    log("  " + "-" * 120)
    for r in rows:
        log(f"  {r['name']:<46} {r['section']:<6} "
            f"{r['listed']:>7.1f} {r['triton']:>7.1f} {r['target']:>7.1f} {r['tflops']:>7.1f} "
            f"{r['progress']*100:>4.0f}%  {r['status']}")

    log("")
    log("# Section progress")
    for s in SECTIONS:
        log(f"  {s:<6} ({LAYOUT_BY_SECTION[s]:<7})  {section_progress[s]*100:>5.1f}%")
    log(f"  {'overall':<6}            {overall*100:>5.1f}%   →  score = {score}")
    log("")


if __name__ == "__main__":
    main()
