# R43 Dev A — 8B-Down V2-RRR regression investigation

Branch: `worktree-agent-aa85570e` (R43 Dev A worktree off `feat/mxfp8-only` HEAD `dbc08280`)
Date: 2026-04-18
GPUs: GPU6 (primary, R36+ rotation), GPU7 (triangulation)
Shape: M=4096, N=4096, K=14336 — 8B-Down
Predicate: `ADVISE-V2-RRR-8B-DOWN (R36B +9.51%)` + `RRR-V2-EXACT-8WAVE`

## Verdict — ROOT CAUSE IDENTIFIED — METHODOLOGY FALSE POSITIVE

The R42 Reviewer Phase 3.3 "regression" (+2.32% Δ%, Welch t=5.46) is **not a kernel regression
nor a build/codegen drift**. It is a **measurement-quality artifact**: the R42 Reviewer
phase23.sh harness ran with `PREHEAT=45` and contains **no sclk gate or retry mechanism**,
so the bench captured a deep-throttle execution where GPU6 was at sclk=1826 MHz throughout
the entire bench (vs ≥2300 MHz nominal).

Re-benching the **identical .so** (`/tmp/r42_8bdown.so`, md5 `2a1ec5119d5ba1c72c73cba113893c39`)
on the **same GPU6** with `PREHEAT_S=60` and post-preheat sclk ≥2200 MHz **fully restores
the R36B SHIP claim** with strong cross-GPU triangulation (no recompile, no kernel change,
no cherry-pick involved).

| Run                               | sclk-post-preheat | CRR median | RRR median | Δ%       | Welch t |
|-----------------------------------|-------------------|------------|------------|----------|---------|
| R36B GPU1 SHIP-LITE original      | 2283 MHz          | 2673.24    | 2911.32    | +8.91%   | 6.55    |
| R36B GPU4 SHIP-LITE original      | (passed gates)    | (cached)   | (cached)   | +6.66%   | 5.87    |
| R37D GPU2-7 4-GPU                 | ≥2200 MHz (gated) | ~2700      | ~2900      | +6.66%   | (n=10)  |
| R38C GPU6 STRICT (attempt 2)      | 2327 MHz          | 2693.04    | 2902.35    | +7.77%   | 22.17   |
| R38C GPU2/3/7 STRICT min          | ≥2200 MHz (gated) | -          | -          | +7.42%   | 18.56   |
| R42 Reviewer GPU6 P3.3            | **1826 MHz**      | **467.11** | **477.96** | **+2.32%** | **5.46** |
| **R43 Dev A GPU6 repro PREHEAT=60** | **2265 MHz**      | **2734.57** | **2950.53** | **+7.90%** | **14.65** |
| **R43 Dev A GPU7 repro PREHEAT=60** | **2370 MHz**      | **2756.74** | **2964.25** | **+7.53%** | **11.66** |
| **R43 Dev A GPU7 repro PREHEAT=90** | **2301 MHz**      | **2693.51** | **2906.89** | **+7.92%** | **14.22** |
| **R43 Dev A GPU7 first attempt (FAILED)** | **1731 MHz**      | (HIP abort)  | (HIP abort)  | n/a        | n/a       |

The "R43 Dev A GPU7 first attempt" row above is itself empirical evidence for the failure mode:
on a fresh idle GPU7 the harness aborted (HIP abort, rc=134) at sclk=1731 MHz post-preheat —
exactly the throttle regime the R36 sclk gates were designed to detect. PREHEAT=90 + 30 s
inter-attempt sleep let the GPU re-stabilize and produce SHIP-grade numbers.

R43 Dev A re-bench restores SHIP claim cleanly (R36B target ≥+5.0%, original +6.66 to +9.51%):
- GPU6: +7.90% (≥+5% PASS by +2.90 pp)
- GPU7: +7.53% (≥+5% PASS by +2.53 pp)
- min Δ%=+7.53%, min Welch t=+11.66 (also clears R38C's STRICT t>10 gate)
- predicate trace verified `ADVISE-V2-RRR-8B-DOWN (R36B +9.51%)` + `RRR-V2-EXACT-8WAVE` both fire
- `tk_mxfp8_r42_8bdown` md5 unchanged (`2a1ec5119d5ba1c72c73cba113893c39`) — same .so as the R42 Reviewer used

## Why the throughput numbers are 6× lower at sclk=1826

A naive 2382/1832 MHz clock-scaling would predict only 1.30× slowdown, not 6×. The
observed 5.77× (CRR 2693 → 467 TF) is consistent with a **deep low-power state** in
which the entire memory subsystem (HBM, fclk, mclk) is also throttled, not just sclk.
At sclk levels below the nominal 2200 MHz operating gate (R34 NEW rule), the device
is not in a benchable thermal-stable regime — it's in a state the R34/R35/R36 sclk-gate
methodology was specifically designed to detect and reject.

Importantly:
- The R42 Reviewer log line `[sclk-pre-preheat] GPU[6] (2397Mhz)` shows the GPU was
  at peak when the harness started, but the 45 s × 16384² FP16 preheat **dropped** the
  clock to 1826 MHz post-preheat — i.e. the preheat itself induced a throttle event.
  PREHEAT=60 s sometimes recovers, sometimes does not (run-to-run variable).
- The R42 reviewer `phase23.sh` (lines 14-17) defaults `PREHEAT=60`, but the shell
  invocation in `r42_reviewer_phase3/p33_8bdown_gpu6.log` line 1 records `PREHEAT=45`,
  meaning the R42 Reviewer manually overrode the default for this cell.
- Either way, `phase23.sh` has **no sclk gate / retry**, unlike R38C's `r38c_8bdown_orchestrate.sh`
  which implements the R36 NEW 3-gate logic with up to 3 retries (G1: sclk-post-preheat
  ≥2200; G2a: sclk-post-bench ≥2200; G2b: stdev/mean ≤1%).

## Why all 4 R42 Phase 3 cells didn't regress equally

The R42 Reviewer also reported P3.1 (70B-KV HB shrink), P3.2 (8B-KV HB shrink), and P3.4
(8B Gate/Up V2-RRR) all PASS at R42. Spot-check:
- P3.1 (GPU6, K=8192): CRR 768.30 / HBSHRINK 984.76 — even baseline is ~6× lower than
  expected steady-state numbers (R37 saw ~3000+ TF here too). The relative Δ% is preserved
  (+28.17%) because both kernels suffer the same throttle. **The Δ% gate is robust to
  uniform throttle; only the absolute TF would catch it.**
- P3.2 (GPU7, K=4096): same pattern, +24.29% preserved.
- P3.4 (GPU7, N=14336 K=4096): GPU7 was hot/healthy (sclk-post-preheat 2370 MHz typical)
  and produced +5.17% which matches the R34B SHIP +5.025% min.
- **P3.3 is the unique cell where the V2-CRR baseline (CRR-V2-EXACT-8WAVE-DEFAULT) and
  V2-RRR (RRR-V2-EXACT-8WAVE) responded asymmetrically to the throttle**, compressing
  the gap from +7.5% to +2.3%. This is consistent with V2-RRR being more memory-BW-sensitive
  at K=14336 (long-K reduction): when HBM throttles, the K=14336 tail benefits less from
  the RRR layout's better A-load pattern because the bottleneck shifts.

The asymmetric throttle response on P3.3 made this look like a kernel regression rather
than a universal harness issue. R36 NEW 3-gate logic was created precisely to prevent
this class of false-positive.

## Hypothesis check matrix

| # | Hypothesis                                 | Verdict  | Evidence                                                                 |
|---|--------------------------------------------|----------|--------------------------------------------------------------------------|
| 1 | Cherry-pick contamination R39-R42          | REFUTED  | Re-bench of identical R42 .so on same GPU6 restores +7.90% w/ PREHEAT=60 |
| 2 | Compile-flag drift                         | REFUTED  | nm shows `rrr_exact_8wave_scaled_kernel<true,2>` present; same flags as R38C; same MXFP8_DECODE_M1_ENABLE / MXFP8_SMALLM_B32_FASTPATH default-OFF state                |
| 3 | Driver/firmware regression                 | UNTESTED, IRRELEVANT | Throughput restored on identical .so/driver/firmware combo by adjusting harness alone — even if drivers changed since R38C, the kernel still delivers SHIP perf |
| 4 | Predicate fired wrong kernel               | REFUTED  | MXFP8_DISPATCH_TRACE confirms `RRR-V2-EXACT-8WAVE` for layout B, `CRR-V2-EXACT-8WAVE-DEFAULT` for layout A; correct dispatch                                              |
| 5 | **R42 phase23.sh missing sclk gate/retry** | **CONFIRMED** | R42 Reviewer log shows sclk=1826 MHz throughout entire bench, no retry. Re-bench with sclk≥2200 MHz post-preheat trivially restores SHIP perf.                  |

## Fix

**No code change needed.** The kernel is healthy.

**Methodology rule (R43+ NEW)**: The R42 phase23.sh harness must be augmented to enforce
the R36 3-gate retry logic that R38C used. Suggested patch (for next reviewer to merge):

```bash
# In phase23.sh, after the bench invocation, parse sclk-post-preheat and sclk-post-bench
# from the bench output and retry up to 3 times if either is below SCLK_GATE_MHZ=2200.
# See analysis/fp8_gemm/mi350x/r38c_8bdown_orchestrate.sh lines 79-140 for reference impl.
```

**For the immediate R42 reviewer false-positive**: Update R43 cycle wrap to record:
- R42 Reviewer Phase 3.3 verdict OVERTURNED — no regression
- R36B/R37D/R38C SHIP for 8B-Down V2-RRR is HEALTHY at HEAD `dbc08280`
- R43 NEW methodology rule: paired-bench harness must enforce sclk gate retry

## Files

- `analysis/fp8_gemm/mi350x/r43a_8bdown_runs/r43a_8bdown_gpu6_repro_clean.txt` — GPU6 PREHEAT=60 repro
- `analysis/fp8_gemm/mi350x/r43a_8bdown_runs/r43a_8bdown_gpu6_repro_clean.err` — predicate trace
- (GPU7 triangulation captured in this document; transcript in conversation log of R43A run)

## Sanity check on cumulative levers

R43A does NOT add a new closed lever. It overturns one R42 Reviewer false-positive verdict and
adds one new methodology rule (sclk-gate-retry mandatory in paired-bench harness). The 43-lever
running total is unchanged.

## R43 cycle wrap recommendation

8B-Down V2-RRR predicate (R36B / R38C STRICT) STATUS = HEALTHY AT HEAD. R42 Reviewer P3.3
verdict was driven by a methodology gap (missing sclk-gate-retry in phase23.sh), not by
any kernel regression. R43A repro: GPU6 +7.90% (t=14.65), GPU7 +7.53% (t=11.66) on the
identical R42 .so artifact.
