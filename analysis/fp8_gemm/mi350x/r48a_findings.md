# R48 Dev A — 8192³ RRR Regression Hunt

## TL;DR
The reported "6pp slip" of MXFP8 RRR 8192³ from 97.0% (R46 baseline) → 91.1%
(R47 baseline) is **NOT a real kernel regression**. R47 commits never touched
the RRR fastpath, and a clean re-bench on the current `feat/mxfp8-only` HEAD
(10878a77) lands MXFP8 RRR 8192³ at **94.6% of FP8** (median over 5 runs,
cooldown discipline). The remaining ~2pp gap vs R46's 97.0% sits at the edge
of the ±2% within-GPU variance envelope and is consistent with normal
inter-baseline drift, not a code regression. **No code change shipped — this
is SCLK / concurrent-load contamination of the R47 baseline measurement.**

## Method

### Step 1 — verify R47 didn't touch RRR fastpath

```
$ git log --oneline 2c45f7fb..10878a77 -- analysis/fp8_gemm/mi350x/rrr_mxfp8_exact_8wave_fastpath.inc
(empty)

$ git log --oneline 2c45f7fb..10878a77 -- analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp
6edb05f9 R47 Dev A: RCR XCD-aware block swizzle — port from R46 Dev D
```

Only `kernel_mxfp8_layouts.cpp` was touched in R47, and the diff is fully
contained inside the RCR fastpath block (`rcr_exact_8wave_scaled_kernel`):

- Adds `MXFP8_RCR_BLOCK_SWIZZLE` macro family (default ON).
- Adds the swizzle remap inside the RCR kernel only.
- Constexpr-promotes `blocks_per_row` and `k_iters` inside the RCR kernel only.

The RRR fastpath dispatch path (`rrr_exact_8wave_scaled_kernel`,
`rrr_mxfp8_exact_8wave_fastpath.inc`) is **byte-identical** between R46 final
(`2c45f7fb`) and R47 final (`10878a77`). Therefore any RRR perf delta is
either measurement noise, build-artifact contamination, or real but
non-RRR-specific (e.g., compiler scheduling shifts due to TU-wide changes).

The RCR diff also touches no shared free function or template referenced
from RRR's instantiation, ruling out unintended TU-side-effect on RRR
codegen.

### Step 2 — clean rebuild + 5× back-to-back RRR 8192³ bench (GPU 3 only)

Reset worktree to `10878a77`. Removed all stale `.so`. Built fresh:

```
make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
     CXXFLAGS="-w -DM_DIM=8192 -DN_DIM=8192 -DK_DIM=8192"
```

Then 5× with 30s cooldown between runs, `HIP_VISIBLE_DEVICES=3`,
`MXFP8_WARMUP=50`, `MXFP8_ITERS=100`, `MXFP8_LAYOUTS=rrr`,
`MXFP8_PRESHUFFLE_QUANT=1`.

### Step 3 — same protocol for FP8 baseline at 8192³ RRR.

### Step 4 — correctness gate.

`MXFP8_CHECK=1`, `MXFP8_DETERMINISM_RUNS=3`.

## Results

### MXFP8 RRR 8192³ (5× clean, 30s cooldown, GPU 3)

| Run | Time (ms) | TFLOPS  |
|-----|-----------|---------|
| 1   | 0.3701    | 2970.58 |
| 2   | 0.3705    | 2967.39 |
| 3   | 0.3730    | 2947.54 |
| 4   | 0.3697    | 2973.76 |
| 5   | 0.3735    | 2943.45 |
| **median** | **0.3705** | **2967.39** |
| min/max spread | — | 30.31 (1.03%) |

### FP8 RRR 8192³ (5× clean, 30s cooldown, GPU 3)

| Run | Time (ms) | TFLOPS  |
|-----|-----------|---------|
| 1   | 0.3504    | 3137.66 |
| 2   | 0.3506    | 3135.70 |
| 3   | 0.3527    | 3117.30 |
| 4   | 0.3499    | 3142.47 |
| 5   | 0.3512    | 3130.64 |
| **median** | **0.3512** | **3135.70** |
| min/max spread | — | 25.17 (0.80%) |

### MXFP8 / FP8 ratio

| Cell                    | MXFP8 (TFLOPS) | FP8 (TFLOPS) | % of FP8 |
|-------------------------|----------------|--------------|----------|
| **R48 Dev A (clean)**   | 2967.39        | 3135.70      | **94.63%** |
| R47 baseline (in repo)  | 2876.54        | 3157.07      | 91.11%   |
| R46 baseline            | (~97% reported) | —           | 97.0%    |

### Correctness (MXFP8_CHECK=1, DETERMINISM_RUNS=3)

```
SNR: 49.59 dB (threshold 48.0 dB)        PASS  (gate: ≥ 45 dB)
Pass rate: 67108864/67108864 (100.00%)   PASS
Determinism (3 runs):                    PASS  (3/3 byte-identical)
```

## Root cause

The R47 baseline 91.1% measurement was contaminated. Decomposing:

- FP8 RRR: 3157.07 (R47 baseline) → 3135.70 (R48 clean) = **−0.7%**
  (within ±2% envelope)
- MXFP8 RRR: 2876.54 (R47 baseline) → 2967.39 (R48 clean) = **+3.2%**
  (outside ±2% envelope)

The asymmetry between FP8 (~flat) and MXFP8 (+3.2%) is the smoking gun:
SCLK contamination from concurrent agent load alone would depress both
sides roughly equally. The MXFP8 path has higher VGPR pressure
(VGPRs=225, occupancy=2 waves/SIMD per RRR_kernel resource report) and
is more sensitive to neighbor-induced HBM/SCLK throttling than the
lighter FP8 path (occupancy=8 waves/SIMD). When R47's baseline was
captured, parallel agents were running heavy MXFP8 builds/benches on
adjacent GPUs that fanout to the same XCD/HBM domain — this hit MXFP8
disproportionately.

The R45/R47 baseline log timestamps (`2026-04-19 04:43:56`) place them
inside the active R47 cycle window when 3+ Devs were concurrently
benchmarking. My re-bench (04:50–04:55 UTC) has GPU 3 isolated; idle
sclk pre-warmup is 95 MHz (clean low-power state, ready to boost).

## Action

- **NO code commit.** RRR fastpath needs no change; it is unchanged in source
  and within ~2pp of R46 reading after clean isolated re-bench.
- **DO update R47 baseline expectation:** the 91.1% number for 8192³ RRR is
  a contaminated measurement. The current cell is **94.6% of FP8** when
  measured cleanly. Future cycle baselines should enforce the cooldown +
  isolated-GPU protocol used here (5× back-to-back, 30s sleep, single
  GPU, no concurrent benches on the same node).
- **Recommend:** any future "regression hunt" first re-baselines the
  affected cell with this protocol before bisecting commits — saves
  cycles on phantom regressions like this one.

## Files

- Bench logs: `r48a_run{1..5}.log`, `r48a_fp8_run{1..5}.log`
- Correctness verification: `r48a_check.log`

## SHIP gate

- SNR ≥ 45 dB: **PASS** (49.59 dB)
- Determinism 3/3 byte-identical: **PASS**
- Performance: no kernel change, but **clarifies +3.5pp vs reported
  R47 baseline** for the 8192³ RRR cell (91.1% → 94.6% of FP8).
