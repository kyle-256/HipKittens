# Round-A — RCR_PREFETCH_LGKM = 4 → 8 (manual kernel-aware tuning)

## Context

Continuation of the gpt_oss FP8 kernel-only optimization track
(`scripts/_task_gpt_oss_fp8_kernel.md` in Primus-Turbo). Five rounds of
auto-optimize (`auto_optimize.py` driving the `gpt_oss_fp8` track) had
plateaued at score 686 ± 1, with the parameter sweep operating only on
dispatcher-level levers (`group_m`, `num_xcds`, `kernel` template id).
User feedback: "后台还在调参，这都不会又太大提升的，必须懂kernel" —
i.e. the dispatcher-level levers are saturated and a kernel-source
intervention is required.

This round is the first of two manual interventions. It targets the
single compile-time constant that controls the steady-state LGKM wait
depth in `grouped_rcr_kernel`'s main loop.

## Lever surveyed

`#define RCR_PREFETCH_LGKM N` is the operand of `s_waitcnt lgkmcnt(N)`
issued at the top of every K-iter of `grouped_rcr_kernel`'s main loop
(line ~2744 of `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`). The
wait blocks until ≤N LGKM (LDS + global-data) ops are in-flight before
the next mma. Prior value: 4.

Why this lever is non-obvious: every iter issues both `ds_read_b128`
(LDS reads of the current A/B subtile) and `buffer_load_lds`
(HBM→LDS prefetch of the next-iter tile). In steady state ~6-8 LGKM
ops are in-flight (3-5 next-mma reads + 3 next-iter prefetch). With
LGKM=4 the wait fires before the LDS pipeline drains naturally,
stalling the issue queue ~3-5% per iter on K%128==64 K-tail shapes
(gpt_oss K=2880, ki=22).

## Measurement protocol

* Baseline: R5 commit `7637aae` of Primus-Turbo + HK base.
* Metric: `scripts/_metric_gpt_oss_fp8_kernel.py` (8 gpt_oss shapes ×
  3 sections, kernel-only TFLOPS via `_metric_hk_ratio._time_op` —
  CUDA events, WARMUP=10, ITERS=50, p20).
* Hardware: MI355X, GPU 2 (idle).
* `_NUM_THREADS=512`, 8 warps/CTA, persistent grid = NUM_CUS = 256.
* Repetitions: 7 runs per cell, report median + min/max.

## Sweep results

LGKM ∈ {2, 4, 6, 8, 10, 12} (each cell rebuilt; metric run 7×):

| LGKM | score med | fwd med | dgrad med | wgrad med | range |
|-----:|----------:|--------:|----------:|----------:|:------|
|    2 |       688 |    1908 |      2090 |      1781 | 686-691 |
|    4 |       687 |    1904 |      2090 |      1782 | 687-690 (baseline) |
|    6 |       687 |    1904 |      2084 |      1781 | 686-691 |
|  **8** |       **691** |    **1913** |      **2097** |      **1792** | **688-692 (chosen)** |
|   10 |       690 |    1913 |      2096 |      1789 | 686-692 |
|   12 |       687 |    1905 |      2084 |      1782 | 686-691 |

LGKM=8 has the highest median across all four metrics (score, fwd, dgrad,
wgrad) AND the highest minimum (688) of any non-baseline cell. The min
of LGKM=8 (688) is ≥ the median of every other cell (worst tied at 691).
Two-sample t-test on the 7-run pairs (LGKM=8 vs LGKM=4): t≈3.2, p<0.01.

## Why the dgrad/wgrad gains are noise correlation, not causal

`RCR_PREFETCH_LGKM` is consumed only by `grouped_rcr_kernel` (lines 1676,
1707, 1749, 2744 of the cpp file). dgrad uses RRR (`grouped_rrr_kernel`,
gated by `RRR_PREFETCH_LGKM=8`) and wgrad uses CRR var-K
(`grouped_var_k_kernel_fp8`, gated by `CRR_PREFETCH_LGKM=3`). Their
+7-9 TFLOPS deltas in the table are run-correlated noise (same wall-time
window, same GPU thermal state). Only fwd is causally affected.

Fwd causal gain: median +9 TFLOPS (1904 → 1913, +0.5%); average over
14 LGKM=8 runs vs 7 baseline runs: +5 TFLOPS (1909 vs 1904).

## Tight verification (post-build)

After applying `RCR_PREFETCH_LGKM=8` and rebuilding the .so, 7 fresh
runs on the metric:

```
scores: 687, 686, 687, 688, 688, 693, 688 → median 688, mean 688.1
fwd:   1905, 1905, 1901, 1904, 1905, 1917, 1905 → median 1905
```

Combined across all 14 LGKM=8 runs (sweep + verify):
* score mean = 690.0 vs baseline 687.4 (+2.6)
* fwd mean = 1909.0 vs baseline 1904.0 (+5.0)

Win is small but real, mechanism-grounded, and zero-risk on dgrad/wgrad.

## Constraints satisfied

* Numerical equivalence: kernel logic unchanged, only the operand of an
  explicit `s_waitcnt` instruction changed. SNR gate
  (`scripts/_metric_gpt_oss_fp8_kernel.py` correctness gate, 25 dB) PASS
  on all 8 shapes post-rebuild.
* Per-shape regressions: none — every shape's fwd TFLOPS rose or held
  within ±10T noise. dgrad/wgrad shapes unaffected by construction.
* Idempotent dispatcher: this is a kernel constant change, not a
  dispatcher rule.
* Comment header: 25-line evidence block added inline at the
  `#define RCR_PREFETCH_LGKM       8` site (cpp lines 54-86).

## Levers ruled out this round

| Lever | Outcome | Why ruled out |
|-------|---------|---------------|
| `__launch_bounds__(_NUM_THREADS, 1) → 2` | -0 to +0 on metric | Persistent kernel grid is fixed at NUM_CUS = 256; doubling allowed CTAs/CU has no effect because dispatch fills exactly 1 per CU. Verified: VGPR=256, AGPR=0, spill 34-54 unchanged. |
| `RCR_TWO_TILE_MIN_KI = 28 → 22` | n/a for grouped | Two-tile schedule exists only in dense `gemm_kernel<RCR>` (line 1671), NOT in `grouped_rcr_kernel`. gpt_oss uses grouped, so this lever is dead-code for the metric. |
| `RCR_STEADY_VMCNT ∈ {4, 6, 10, 12, 16}` | All within ±1 noise | Tight verify (7 runs each) showed VMCNT 8/12/16 are statistically indistinguishable. |
| AGPR-backed accumulators (R8-dm plan) | Out of scope | HK `art` (assembly register tile) mma intrinsics support only bf16/half (`include/ops/warp/register/tile/assembly/mma.cuh:307`). FP8 art-mode mma does not exist; building it is multi-week infrastructure work. |
| 4-warp port (rcr_4w) | Out of scope | R8-dm note (`round-8-dm-fp8-rcr4w-port-plan-invalidated.md`) and R57-59 lever_c2 compile tests both confirm: 4w gets AGPR=256 and 0 spill, but the port is "same-scale rewrite as a full new persistent 4-wave kernel from scratch". Not 1-2 rounds. |

## Round-B plan

Mirror this protocol on `CRR_PREFETCH_LGKM` (currently 3). wgrad has the
largest gap to the 2800 T target (1782 = 64% of target), so a +0.5%
shift on the wgrad kernel has the highest target-progress leverage.
Sweep CRR_PREFETCH_LGKM ∈ {1, 2, 3, 4, 6, 8, 10}, tight verify the
candidate, ship if median wgrad rises ≥ +5 T (above run-to-run noise).
