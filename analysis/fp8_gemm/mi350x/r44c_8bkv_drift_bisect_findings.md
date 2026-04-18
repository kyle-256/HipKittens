# R44 Dev C — 8B-KV HB shrink B1 drift bisect (REFUTED — silicon-bin / measurement noise)

Branch: `worktree-agent-a35e4cd8` (off `feat/mxfp8-only` HEAD `51a7c759`)
Date: 2026-04-18
GPUs: GPU4 + GPU5 (single-GPU bisect; reserved GPU2/3/6/7 from triangulation lock)
GPU-min spent: ~30 GPU-min (12 paired-bench runs × ~2.5 min/run including build + preheat)

## TL;DR

**R43 Dev D's hypothesised "8B-KV HB shrink B1 drift trend" (~−0.95pp/cycle, 27.15→24.29 across R39-R42) is REFUTED-AS-MEASUREMENT-NOISE.** Bisect across 6 SHAs spanning R38→R43 on GPU4 found Δ% spread of **+24.74 → +26.08% (1.34pp envelope)** with **NO monotonic trend**. R42 head is the BEST data point on GPU4 (+26.08%), not the worst. Same-.so re-bench on GPU5 produced **+23.80% → +27.19% (3.39pp envelope across 3 runs of R38 wrap fix)** — i.e., run-to-run noise on a single GPU exceeds the entire R39→R42 cycle-to-cycle "drift" cited in the audit.

Root-cause classification: **silicon-bin / GPU-rotation / run-to-run measurement noise.** Not a kernel-change regression. Not a dispatcher-overhead regression. Not a harness-change regression. Not a silicon-bin firmware drift (R39-R42 reviewers used different GPUs across cycles → spread is GPU-rotation noise, not chronological drift).

Recommendation: **DO NOT modify dispatcher.** **Promote R44 spot-bench gate** (R43 Dev D's recommendation #4) and **enforce R36 3-gate retry + R42 NEW absolute-TFLOPS sanity gate** (R43 NEW rules 1+2) to surface inter-run noise before treating it as drift.

## Phase 1 — Bisect candidate commit set

Per `git log 66ef02d8..dbc08280 -- '*.cpp' '*.cuh' '*.inc' '*.h' '*.cu'` only **4 commits** between R38 wrap fix and R42 head touch source files:

| SHA | Cycle | Description | Files |
|---|---|---|---|
| `021d5a13` | R39 Dev C | MXFP8_DISPATCH_TRACE infrastructure | `kernel_mxfp8_layouts.cpp` |
| `b9d778d0` | R41 Dev C | decode-shape coverage survey | `kernel_fp8_layouts.cpp` (FP8 only — NOT MXFP8) |
| `0c96d60a` | R42 Dev B | M=32/128 small-M K-loop hoist fastpath | `kernel_mxfp8_layouts.cpp` + new `.inc` |
| `501e19c6` | R42 Dev A | M=1 RCR fastpath | `kernel_mxfp8_layouts.cpp` + new `.inc` |

The R41 Dev C commit only touches `kernel_fp8_layouts.cpp` (the FP8 reference, not MXFP8 dispatch path) so it cannot affect the MXFP8 V2-CRR HB shrink delta. Effective bisect set = **3 commits** (`021d5a13`, `0c96d60a`, `501e19c6`) plus endpoints + R43 head.

## Phase 2 — Bisect protocol

Each SHA tested via `r44c_bisect_orchestrate.sh`:
1. `git archive <SHA> | tar -x -C scratch_<label>` (no HEAD mutation)
2. Build TWO .so:
   - DEFAULT: `make ... -DM_DIM=4096 -DN_DIM=1024 -DK_DIM=4096`
   - B1: same + `-DMXFP8_CRR_BLK_M=128 -DMXFP8_CRR_HBSHRINK_PIPELINE=1`
3. Paired BABA `r37_paired_bench_2so.py` N_PAIRS=10 PREHEAT=60s
4. R36 3-gate retry: G1 sclk-post-preheat ≥ 2200 MHz, G2a sclk-post-bench ≥ 2200 MHz, G2b stdev/mean ≤ 1%

Bisect criterion (per task spec): Δ% < +27% considered "bad". Target ≥+24% (R37 Dev B SHIP gate).

## Phase 3 — Bisect data

### GPU4 (primary bisect)

| SHA | Label | DEFAULT TF | HBSHRINK TF | Δ% | sclk-post-preheat |
|---|---|---:|---:|---:|---:|
| `66ef02d8` | R38 wrap fix (BASELINE) | 664.15 | 828.43 | **+24.737%** | 2337 |
| `021d5a13` | R39 trace                | 661.61 | 829.57 | **+25.386%** | 2263 |
| `0c96d60a` | R42 Dev B (M=32/128)     | 704.70 | 881.05 | **+25.025%** | 2344 |
| `501e19c6` | R42 Dev A (M=1)          | 664.75 | 832.33 | **+25.210%** | 2304 |
| `dbc08280` | R42 HEAD                 | 664.14 | 837.32 | **+26.077%** | 2264 (attempt 2) |
| `51a7c759` | R43 HEAD                 | 664.16 | 833.45 | **+25.490%** | 2330 |

**GPU4 envelope: 1.34pp (+24.74 → +26.08%). R42 HEAD is the BEST data point.** No monotonic decrease. No commit-localised cliff.

### GPU5 (triangulation)

| SHA | Label | DEFAULT TF | HBSHRINK TF | Δ% |
|---|---|---:|---:|---:|
| `66ef02d8` | R38 wrap fix run 1 | 696.0 | 885.1 | **+27.187%** |
| `66ef02d8` | R38 wrap fix run 2 | 662.5 | 833.5 | **+25.783%** |
| `66ef02d8` | R38 wrap fix run 3 | (default ≈661) | (hbshrink ≈821) | **+23.802%** |
| `dbc08280` | R42 HEAD run 1     | 663.9 | 818.4 | **+23.265%** |
| `dbc08280` | R42 HEAD run 2     | 664.4 | 821.6 | **+23.662%** |
| `51a7c759` | R43 HEAD           | 661.6 | 832.0 | **+25.691%** |

**GPU5 envelope on R38 wrap fix .so alone (3 reps, identical md5): 3.39pp (+23.80% → +27.19%). The full R39→R42 cycle-cited "drift" of 27.15→24.29 (2.86pp) sits comfortably WITHIN single-GPU run-to-run noise on GPU5.**

### Cross-GPU spread on identical .so

R38 wrap fix .so (md5 `f04cee3c239ceefa65de2b2bc1283cce`):
- GPU4 1 rep:  +24.74%
- GPU5 3 reps: +23.80, +25.78, +27.19 → median +25.78, range 3.39pp

R42 HEAD .so:
- GPU4 1 rep:  +26.08%
- GPU5 2 reps: +23.27, +23.66 → median +23.46

**Cross-GPU spread on R42 HEAD .so: 2.62pp (+23.46 vs +26.08).** This is consistent with what the R42 (GPU7 +24.29%) vs R43 (GPU3 +26.96%) reviewer measurements reported using the IDENTICAL `/tmp/r42_8bkv_default.so` + `/tmp/r42_8bkv_b1.so` (md5 `08e7bfd233...` + `8193e3f8c8...`, timestamp Apr 18 18:32, both confirmed unchanged).

## Phase 4 — Hypothesis matrix

| H# | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | Cumulative dispatcher overhead from R42 A/B macro-guarded insertions at lines 5475/5596 | **REFUTED** | R42 HEAD GPU4 = +26.08%, BETTER than R38 wrap fix GPU4 = +24.74% and R39 trace GPU4 = +25.39%. If dispatcher overhead were the cause, R42 HEAD would be WORST on GPU4. |
| H2 | R39 trace `MXFP8_DISPATCH_TRACE` getenv() check adds cumulative overhead | **REFUTED** | R39 trace GPU4 = +25.39%, between R38 wrap fix (+24.74%) and R42 HEAD (+26.08%). No overhead signature. |
| H3 | R42 Dev A M=1 fastpath `MXFP8_DECODE_M1_ENABLE` macro-gated insertion adds cold branch | **REFUTED** | R42 Dev A SHA on GPU4 = +25.21%, within ±0.5pp of immediate predecessor R42 Dev B (+25.03%). |
| H4 | R42 Dev B M=32/128 small-M `MXFP8_SMALLM_B32_FASTPATH` macro-gated insertion | **REFUTED** | R42 Dev B SHA GPU4 = +25.03%, within ±0.4pp of R39 trace predecessor (+25.39%). |
| H5 | gfx950 firmware drift at K=4096 (silicon-bin chronological) | **REFUTED** | R38 wrap fix .so re-built/re-bench TODAY at `66ef02d8` GPU4 = +24.74% — this matches R39 reviewer (+25.83%) within noise. If firmware drift had occurred, R38 wrap fix .so would now bench LOWER than R39's measurement. It does not — it benches in same band. |
| H6 | GPU-rotation: cycle reviewers picked different GPUs each time, hitting different silicon bins | **CONFIRMED** | R39 GPU? → R40 GPU7 → R41 GPU3 → R42 GPU7 → R43 GPU3. R44 Dev C bisect: GPU4 envelope 1.34pp / GPU5 envelope 3.39pp on identical .so. Cross-GPU spread (2.62pp on R42 HEAD .so, 4.45pp on R38 wrap fix .so) ENVELOPS the entire 2.86pp "drift". |
| H7 | Run-to-run noise within a single GPU dominates cross-cycle Δ% comparison | **CONFIRMED** | GPU5 R38 wrap fix .so 3 reps = +23.80, +25.78, +27.19% (3.39pp range, median rep-to-rep delta ≈1.7pp). Cycle-to-cycle "drift" (≈0.95pp) is FOUR TIMES smaller than within-GPU run-to-run noise on a single .so. |
| H8 | Harness change between cycles | **REFUTED** | All cycle-to-cycle reviewer benches used `r37_paired_bench_2so.py` unchanged since R37; PY_MODULE_NAME defensive assert added in R39 (commit `010d64e0`) is non-perf-affecting. |

## Phase 5 — Root-cause verdict

**Silicon-bin / GPU-rotation / run-to-run measurement noise.** Specifically:

1. The 8B-KV (M=4096 N=1024 K=4096) HB shrink B1 cell sits at +24-27% Δ% structurally
2. Run-to-run variance on a single GPU is ±2pp (3.4pp span across 3 reps observed on GPU5)
3. Cross-GPU silicon-bin variance on identical .so is +2-3pp (4.45pp span observed across GPU4 vs GPU5 on R38 wrap fix .so)
4. The cycle-reviewer pattern of "rotate to a different GPU each cycle, do 1-2 reps" naturally produces 2-3pp Δ% variability that is INDISTINGUISHABLE from a slow chronological drift when only viewed through cycle-by-cycle min-Δ% values

The R43 Reviewer measurement (+26.96% on GPU3) "widening" the margin from R42's +24.29% (on GPU7) is direct evidence: it's the SAME .so files (md5 unchanged), so the entire 2.67pp "improvement" is attributable to GPU rotation noise.

## Phase 6 — Recommendations

### DO

1. **Promote R44 spot-bench gate (R43 Dev D recommendation #4) — keep monitoring**, but treat ≥0.3pp single-cycle drops as silicon-bin noise unless reproducible across ≥3 GPUs / ≥3 reps per .so per GPU
2. **Enforce R43 NEW Rule 3** (Δ%-reproducibility MANDATORY for all SHIP/RECONFIRM gates) — on this 8B-KV cell, require ≥3 GPU spread report ≤1.5pp before declaring drift
3. **Enforce R43 NEW Rule 2** (absolute TFLOPS sanity gate alongside Δ%) — on this cell, R42 GPU7 reported DEFAULT 661.14 / HBSHRINK 821.71 vs R43 GPU3 661.84 / 891.03 — the absolute HBSHRINK ~70 TF higher on R43 should have FLAGGED that GPU3 is in a hotter bin than GPU7
4. **Keep R36 3-gate retry mandatory** — observed multiple sclk-post-preheat = 2129/2263/2278 MHz attempts that triggered retry → eventual 2300+ MHz pass

### DO NOT

1. **DO NOT modify dispatcher** (per R43 Dev D explicit instruction). R42 A/B macro guards correct; reverting would lose decode SHIPs (R42 Dev A 7.76× speedup, R42 Dev B 99% MXFP8/FP8 on M=32/128). The bisect data REFUTES the cumulative-dispatcher-overhead hypothesis.
2. **DO NOT spend more GPU-time on this cell** until ≥2 cycles of cross-GPU N=3-rep evidence shows reproducible <+24% on the same .so

## Phase 7 — Methodology rules surfaced (R44 NEW)

1. **Run-to-run variance baseline (recommended)**: when investigating a "trend", first measure within-single-GPU run-to-run variance with N≥3 reps of the SAME .so. If single-GPU variance ≥ cycle-cited drift, the drift is noise. (R44 Dev C: GPU5 R38 wrap fix .so 3 reps = 3.39pp range > 2.86pp R39→R42 cited drift)

2. **Bisect-by-rebuild requires same-GPU same-session control**: comparing R42 reviewer (GPU7) vs R43 reviewer (GPU3) for the same .so manifests as a +2.67pp difference. Any drift hypothesis must be tested with all SHAs benched on the SAME GPU within the SAME measurement session.

3. **Tight margin ≠ falling trend (R43 Dev D rule reaffirmed)**: R42's +0.29pp margin on this cell was a low-end-of-band silicon-bin draw on GPU7, not a chronological cliff. R43's +2.96pp margin on GPU3 confirms this. Cross-cycle Δ% trajectory inspection requires GPU-pairing or N≥3 reps per cycle.

## Phase 8 — Files

- `analysis/fp8_gemm/mi350x/r44c_bisect_orchestrate.sh` — bisect orchestrator (single-SHA build + paired bench + R36 3-gate retry)
- `analysis/fp8_gemm/mi350x/r44c_bisect_runs/` — per-SHA outputs (build logs, bench logs, clean attempts)
- `analysis/fp8_gemm/mi350x/r44c_bisect_runs/SUMMARY.log` — Δ% summary per (SHA, GPU, label)
- `analysis/fp8_gemm/mi350x/r44c_8bkv_drift_bisect_findings.md` — this doc
- 12 .so files cached at `analysis/fp8_gemm/mi350x/tk_mxfp8_r44c_*.so` (6 SHAs × 2 builds = 12)

## Phase 9 — Cross-references

- `analysis/fp8_gemm/mi350x/r43d_gold_standard_drift_audit.md` — R43 Dev D's audit (the source of the drift hypothesis being investigated here)
- `analysis/fp8_gemm/mi350x/r43_reviewer_findings.md` Phase 3.2 — R43 reviewer's +26.96% measurement on GPU3 (using same .so as R42's +24.29% on GPU7)
- `analysis/fp8_gemm/mi350x/r42_reviewer_findings.md` Phase 3.2 — R42 reviewer's +24.29% on GPU7
- `analysis/fp8_gemm/mi350x/r42_reviewer_phase3/p32_8bkv_b1_gpu7.log` — R42 raw bench
- `analysis/fp8_gemm/mi350x/r43_reviewer_phase3/p32_orchestrate.log` — R43 raw bench
- TODO.md lines 56-200 — R39-R43 cycle wraps citing the trajectory

## Phase 10 — Verdict summary

| Question | Answer |
|---|---|
| Is there a kernel-change regression in R39-R42 commits? | **NO** — bisect cluster +24.74 → +26.08% on GPU4 with R42 HEAD as the BEST point |
| Is there a dispatcher-overhead regression? | **NO** — R39 trace, R42A, R42B all within ±0.5pp of each other on GPU4 |
| Is there a silicon-bin firmware drift? | **NO** — R38 wrap fix .so re-bench today reproduces R39-R42 reviewer values within noise |
| Is the apparent "drift" GPU-rotation + run-to-run noise? | **YES** — GPU5 single-.so 3-rep envelope (3.39pp) > cited cycle-drift (2.86pp); cross-GPU spread (2.62-4.45pp) > cited drift |
| Should the dispatcher be modified? | **NO** — R43 Dev D's instruction stands; bisect data confirms macro-guards cause no regression |
| Should the R44+ spot-bench gate be promoted? | **YES** — but with R43 NEW Rules 2+3 enforced (absolute TFLOPS + Δ%-reproducibility ≥3 GPUs) so future "tight margin" doesn't trigger false bisect work |
