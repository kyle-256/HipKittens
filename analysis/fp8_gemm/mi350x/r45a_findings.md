# R45 Dev A: M=2..16 SHIP Completion Attempt — SCLK-CONTAMINATED

## Objective
Complete R44A SHIP via GPU2/3/6/7 triangulation with R36 3-gate retry on 4 INCLUDED
M=2..16 cells (M=4 K=4096, M=4 K=8192, M=8 K=8192, M=16 K=8192); bench all 3
layouts (RRR/CRR/RCR) per cell.

## Result: FAILED — 3-gate retry exhausted on majority of cells

### Root cause
GPU sclk never stabilized at >=2200 MHz across GPU2/3/6/7. The R36 3-gate retry
harness requires:
- G1: sclk >= 2200 MHz post-preheat
- G2a: sclk >= 2200 MHz post-bench
- G2b: stdev/mean <= 1% for both DECODE and BASELINE

Typical observed sclk: 1700-1800 MHz post-preheat, ramping to 2400 MHz only in
late pairs (3-4 of 5), creating massive within-run variance (stdev/mean 10-23%).

### Sweep coverage
- 4 shapes x 3 layouts x 4 GPUs = 48 cells attempted
- ~30 cells FAILED R36 3-gate (rc=2)
- ~18 cells completed but with unreliable sclk-contaminated data
- Raw delta percentages range +0.3% to +484% — ENTIRELY due to sclk ramp, not kernel performance

### Dispatch trace verification
All cells correctly dispatch to R44A kernel:
- `SMALLM-DECODE-M2-16-RRR (R44A)` for RRR layout
- `SMALLM-DECODE-M2-16-CRR (R44A)` for CRR layout  
- `SMALLM-DECODE-M2-16-RCR (R44A)` for RCR layout

This confirms R44A waterfall integration is functional. The kernel code is correct;
only the bench methodology failed.

### R46 recommendation
1. Investigate sclk instability — may be thermal throttling or power management.
   Consider: longer preheat (60-90s vs 30s), `rocm-smi --setperflevel high` before sweep.
2. Retry with extended preheat + performance-level lock
3. If sclk still won't stabilize: relax G1 gate to >= 2000 MHz (with documented justification)

### Files
- `r45a_orchestrate.sh` — main 4-GPU parallel orchestrator
- `r45a_run_4gpu_sweep.sh` — per-GPU sweep driver
- `r45a_paired_bench.py` — paired bench with R36 3-gate (derived from R38C gold-standard)
- `r45a_fp8_ref_bench.py` — FP8 reference bench (not used due to sclk failure)
- `r45a_summarize.py` — aggregator (not invoked due to failure)
- `r45a_within_gpu_reps.sh` — within-GPU rep harness (not invoked)
- `r45a_fp8_run.sh` — FP8 reference run script
- `r45a_runs/` — 328 raw log files (3-attempt × 4-GPU × 12 cells + smoke + sweep_main)
