# R48 Dev C: CRR wide-N structural floor — REFUTED

## Summary

Investigated CRR layout's structural floor at wide-N shapes (Gate/Up family). 3
hypotheses tested across 7 compute-bound shapes:

1. **Step 1 — CRR XCD swizzle revalidation** (R47 Dev B's default-ON setting):
   confirmed swizzle ON is correct default. Net positive across all 7 CRR
   shapes (worst -0.20%, best +20.74% on 70B Gate/Up — note the latter is
   inflated by a contaminated OFF run1=1468 outlier).

2. **Step 2 — GROUP_M sweep on 70B Gate/Up CRR** (single-shape exploration):
   GROUP_M ∈ {2, 4, 8, 16}. Default = 4. Best alternative = GROUP_M=8 at
   +0.66% (below 2% SHIP gate).

3. **Step 3 — GROUP_M=8 full sweep across 7 CRR shapes**: REFUTED (worst
   -3.24% on 70B Down, exceeds ±2% envelope).

**Verdict: REFUTED.** No new SHIP lever for CRR wide-N. R47 Dev B's CRR XCD
swizzle remains the optimal default.

## Step 1: Swizzle ON vs OFF revalidation

Confirms R47 Dev B's `MXFP8_CRR_BLOCK_SWIZZLE` default-ON is net positive:

| Shape           | OFF avg | ON avg | Δ% (ON−OFF) |
|-----------------|--------:|-------:|------------:|
| 8192³           |  2763.4 | 2759.8 |       -0.13%|
| 8B Q/O          |  2045.7 | 2105.9 |       +2.94% ★|
| 8B Gate/Up      |  2292.7 | 2297.9 |       +0.23%|
| 8B Down         |  2645.1 | 2688.2 |       +1.63%|
| 70B Q/O         |  2654.0 | 2648.7 |       -0.20%|
| 70B Gate/Up     |  2058.8 | 2485.8 |      +20.74% ★†|
| 70B Down        |  2553.1 | 2626.4 |       +2.87% ★|

† 70B Gate/Up OFF run1 = 1468.2 is contaminated (other 2 runs = 2352, 2356).
Excluding the contaminated run, the real Δ is closer to **+5.6%**, still a win.

All deltas net positive or within ±0.5% — **swizzle ON remains optimal**.

## Step 2: GROUP_M sweep on 70B Gate/Up CRR

70B Gate/Up CRR is the worst FAIL cell (88.5%, gap 6.5pp). Tested GROUP_M
parameter on the swizzle (controls how many M-tiles share an XCD before
rotating to the next):

| GROUP_M | avg TFLOPS | Δ% vs default GROUP_M=4 |
|--------:|-----------:|------------------------:|
|       2 |     2411.9 |                  -3.13% |
|       4 |     2489.9 |                  default|
|       8 |     2506.4 |                  +0.66% |
|      16 |     2324.2 |                  -6.65% |

GROUP_M=8 marginally above default (+0.66%), below SHIP gate. GROUP_M=2 / 16
both regress badly. Default GROUP_M=4 is locally optimal for 70B Gate/Up.

## Step 3: GROUP_M=8 full sweep (no-regression check)

Despite GROUP_M=8 being below SHIP gate on its target shape, swept across all
7 CRR shapes to test if it's a net-positive default:

| Shape           | GM=4 avg | GM=8 avg |    Δ%    |
|-----------------|---------:|---------:|---------:|
| 8192³           |   2774.6 |   2714.4 |   -2.17% XX|
| 8B Q/O          |   2061.4 |   2057.4 |   -0.20% |
| 8B Gate/Up      |   2172.4 |   2291.2 |   +5.47% ★|
| 8B Down         |   2697.6 |   2674.2 |   -0.87% |
| 70B Q/O         |   2648.5 |   2634.8 |   -0.52% |
| 70B Gate/Up     |   2464.7 |   2503.2 |   +1.56% |
| 70B Down        |   2516.5 |   2434.8 |   -3.24% XX|

**REFUTED**: 2 wins (8B Gate/Up +5.47%, 70B Gate/Up +1.56%) but 2 losses
worse than ±2% envelope (8192³ -2.17%, 70B Down -3.24%).

The 70B Down regression is particularly bad — a -3.24% slip on an already-
SHIP'd cell (108.2% R47 baseline) would push it back from +PASS to FAIL.

## Root-cause interpretation

GROUP_M acts as a tradeoff between L2 reuse (larger group → more M-tiles per
XCD → tighter B-tile cache locality) and load balance across XCDs (smaller
group → finer-grained XCD assignment).

- 8B Gate/Up (M=4096, N=14336, K=4096): N-heavy, K-light → B-tile cache
  reuse benefits from larger groups → +5.47% on GM=8.
- 70B Down (M=4096, N=8192, K=28672): K-heavy → A-tile streaming dominates
  → larger M-groups starve some XCDs → -3.24% on GM=8.

There is no single GROUP_M that wins universally → would need per-shape gate.
Per-shape gate adds compile complexity for marginal cumulative gain (+0.7%
geomean) — not worth shipping.

## Hardware-ceiling alignment (per R48 Dev D §2.3)

R48 Dev D classified 70B Gate/Up CRR as a HEADROOM cell:
- Measured 88.5% vs predicted 92.5% → +4pp gap recoverable
- Predicted ceiling 92.5% reflects CRR's structural 6 v_lshrrev_b32 scale-shift
  ops per K-pair (vs RCR/RRR which use opsel = zero shifts)

GROUP_M tuning is a workgroup-dispatch lever — orthogonal to the scale-shift
structural floor. To recover the full +4pp would require restructuring CRR's
scale unpacking pipeline (R49+ kernel rewrite territory).

## Files added

- `r48c_swizzle_revalidation.sh` + `.run.log` — Step 1 sweep
- `r48c_groupm_sweep.sh` + `.run.log` — Step 2 sweep on 70B Gate/Up
- `r48c_groupm_full_sweep.sh` + `.run.log` — Step 3 sweep across 7 shapes
- `r48c_groupm_full_results/` — per-shape build/run logs (3 runs × 2 GMs × 7 shapes)

## Files modified

None. `MXFP8_CRR_BLOCK_SWIZZLE_GROUP_M` macro infrastructure already exists in
`crr_mxfp8_exact_8wave_fastpath.inc` from R47 Dev B (default = 4); no kernel
change shipped this cycle.

## Recommendations for R49+

1. **Per-shape GROUP_M gate** — if the +5.47% on 8B Gate/Up CRR is structurally
   reproducible (not contamination), per-shape gate via `#if N_DIM == 14336`
   could ship a +5.47% on one cell with no penalty elsewhere. Re-validate
   first under strict SCLK protocol.

2. **CRR scale-shift restructuring** — Dev D's structural floor finding (CRR
   has 6 scale-shift ops vs 0 for RCR/RRR opsel-friendly layouts) is the real
   lever for closing CRR wide-N gaps. R49 kernel rewrite to make scale layout
   opsel-friendly for CRR would lift the entire CRR ceiling from 92% → 95%.

3. **STOP list (per R48 Dev D)**: 8B Down CRR (94.6%, near hit), 70B Q/O CRR
   (93.6%) — both classified at ceiling. Don't burn cycles trying to lift
   these via dispatch tweaks.
