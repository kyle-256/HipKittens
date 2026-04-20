# R56 Dev B — 70B Gate/Up CRR PMC diagnostic — DIAGNOSTIC-CRR-NEW-CLASS-WAIT-DOMINANT

**Cycle:** R56
**Cell (primary):** 70B Gate/Up CRR (M=4096, N=28672, K=8192)
**Co-cells (cross-shape triangulation):** 70B Q/O CRR (4096×8192×8192, K=8192, same K different N), 70B Down CRR (4096×8192×28672, K=28672, large-K reference)
**Baseline gap (primary):** MX/FP8 = **86.46%** (HEADROOM -8.54pp vs 95% gate, -13.54pp vs 100%) — biggest open gap in the 9-cell sweep
**Lever:** PMC counter-pair capture (FP8 vs MXFP8) on CRR layout — `set1` (utilization) + `set3` (stall attribution). Observational only, mirrors R54B / R55F method.
**Origin:** R54B (70B Down RCR) → DIAGNOSTIC-SCALE-FETCH-WAIT (SQ_WAIT_ANY +71.6%, MfmaUtil -16pp). R55F (8B GU RRR) → DIAGNOSTIC-SCALE-FETCH-WAIT-SALU-DOMINANT (SQ_WAIT_ANY +31.2%, SQ_INSTS_SALU +50.6%). 70B Gate/Up CRR was the only un-PMC'd open HEADROOM cell. R55E showed an LDS-resident scale layout regressed (-7.98% median) due to LDS round-trip dominating the 6-cycle v_lshrrev saving — so the v_lshrrev shifts are *paid by both FP8 and MXFP8* in CRR (CRR floor), not the marginal MX delta.
**GPU:** MI355X (gfx950), HIP_VISIBLE_DEVICES=1 per task assignment.

## TL;DR — VERDICT: DIAGNOSTIC-CRR-NEW-CLASS-WAIT-DOMINANT (NOT-A-FIX)

**The 70B Gate/Up CRR -8.54pp HEADROOM is a wait-cycle-dominated bottleneck class, structurally distinct from both R54B's RCR signature and R55F's RRR signature.** The dominant PMC delta FP8→MXFP8 on the primary cell:

| Metric | FP8 | MXFP8 | Δ (MX-FP8) | Ratio | vs R54B RCR | vs R55F RRR |
|---|---:|---:|---:|---:|---:|---:|
| **SQ_WAIT_ANY** (cyc4) | 1.658e8 | 4.074e8 | +2.416e8 | **+145.7%** ← DOMINANT | 1.7× R54B's +71.6% | 4.7× R55F's +31.2% |
| **MfmaUtil_pct** | 70.98 | 51.23 | -19.75pp | 0.722 | larger than R54B's -16pp | far larger than R55F's -7pp |
| **SQ_INSTS_SALU** | 2.780e7 | 3.726e7 | +0.946e7 | **+34.0%** | 2.3× R54B's +15.1% | 0.67× R55F's +50.6% |
| **SQ_ACTIVE_INST_VALU** | 7.574e7 | 1.084e8 | +3.270e7 | **+43.2%** | smaller than R54B/R55F (~+84%) | smaller |
| SQ_INSTS_VALU | 7.574e7 | 7.906e7 | +0.333e7 | +4.4% | flat (vs R54B +18%) | flat |
| **VALUBusy_pct** | 22.89 | 23.65 | +0.76pp | 1.033 | flat (vs +7.5pp R55F) | nearly identical |
| SQ_INSTS_MFMA | 2.936e7 | 2.936e7 | 0 | 1.000 | identical | identical |
| SQ_INSTS_LDS | 4.404e7 | 4.404e7 | 0 | 1.000 | identical | identical |
| SQ_INSTS_VMEM_RD | 7.340e6 | 8.258e6 | +0.918e6 | +12.5% (= 9/8) | canonical | canonical |
| **TCC_hit_pct** | 72.56 | 80.54 | +7.98pp | 1.110 | MXFP8 IMPROVES L2 | improves |
| **FetchSize_KB** | 1013985 | 711246 | -302738 | **0.701** (MXFP8 fetches 30% LESS) | improves |
| LDSBankConflict_pct | 0 | 0 | 0 | n/a | not LDS | not LDS |
| TA_DATA_STALL_TC_per_grbm | 0.645 | 0.214 | -0.431 | 0.331 | LESS data stall in MXFP8 | less |

**Key signature: VALUBusy is essentially flat (+0.76pp), SQ_INSTS_VALU is essentially flat (+4.4%), but SQ_WAIT_ANY MORE THAN DOUBLES (2.46×).** The MFMA pipe is starved on something other than VALU/SALU instruction dispatch — the wait-cycle expansion happens *without* a corresponding work expansion. This is the **CRR-WAIT-DOMINANT** class.

This is a **DIAGNOSTIC**, not a SHIP/REFUTED — PMC profiling does not modify any kernel and the verdict gates no source change.

## Cross-shape PMC summary (CRR layout)

```
Cell             K       FP8 TFLOPS   MX TFLOPS   MX/FP8    MfmaUtil FP8   MfmaUtil MX   TCChit FP8   TCChit MX
70B_GateUp     8192        2955.7      2555.7     86.5%        70.98         51.23         72.56        80.54     <-- HEADROOM
70B_QO         8192        2933.1      2823.6     96.3%        69.50         61.02         80.57        80.59
70B_Down      28672        3123.7      2740.8     87.7%        76.03         56.10         81.01        80.05
```

**Key cross-shape observations (CRR):**

1. **N-dimension drives the gap at fixed K=8192**: 70B_GateUp (N=28672) is at 86.5%, 70B_QO (N=8192) is at 96.3%. **9.8pp gap from N alone.** This is the dominant cross-shape signal — it is *not* K-driven within CRR.

2. **K-driven gap (large K) is comparable**: 70B_Down at K=28672 sits at 87.7% — comparable to 70B_GateUp at K=8192. So the bottleneck has TWO independent levers — N and K both inflate it — but the relative weights flip.

3. **MfmaUtil collapse is uniform across the bottleneck cells** (-19pp at 70B_GateUp, -20pp at 70B_Down) but only -8.5pp at 70B_QO (the passing cell). **MfmaUtil collapse is the dominant TFLOPS-translatable effect.**

4. **L2 hit rate INVERTS in the wrong direction for the gap diagnosis**: at 70B_GateUp MXFP8 has +7.98pp BETTER hit rate and 30% LESS fetched bytes. So the gap is NOT L2 / cache miss / memory bandwidth — it's downstream of fetch arrival.

## Bottleneck class identification — comparing all three diagnostic cycles

| Counter signature | R54B 70B Down RCR | R55F 8B GU RRR | **R56B 70B GU CRR (this work)** |
|---|---|---|---|
| Layout | RCR | RRR (V2) | CRR (V2) |
| MX/FP8 ratio | 90.5% | 92.9% | **86.5%** ← worst |
| SQ_WAIT_ANY ratio | 1.716 (+71.6%) | 1.312 (+31.2%) | **2.457 (+145.7%)** ← dominant |
| SQ_INSTS_SALU ratio | 1.151 (+15.1%) | 1.506 (+50.6%) | 1.340 (+34.0%) |
| SQ_INSTS_VALU ratio | 1.181 (+18.1%) | 1.171 (+17.1%) | 1.044 (+4.4%) ← FLAT |
| VALUBusy delta | +5pp | +7.47pp | **+0.76pp** ← FLAT |
| SQ_ACTIVE_INST_VALU ratio | ~1.84 (+84%) | 1.844 (+84.4%) | 1.432 (+43.2%) ← smaller |
| MfmaUtil delta | -16pp | -7.04pp | **-19.75pp** ← largest collapse |
| TCC_hit delta | mostly flat | +1.06pp | **+7.98pp** ← MXFP8 wins L2 |
| FetchSize MX/FP8 | ~1.0 | 0.965 | **0.701** ← MXFP8 fetches 30% less |
| LDS bank conflict | 0 | 0 | 0 |
| Bottleneck class | SCALE-FETCH-WAIT | SCALE-FETCH-WAIT-SALU-DOMINANT | **WAIT-DOMINANT (no work expansion)** |

**The third class is qualitatively distinct.** R54B (RCR) and R55F (RRR) both show a **work-side expansion** (more VALU instructions, more VALU pipe occupancy, more SALU) that explains *why* WAIT cycles grow. The CRR class shows almost no work-side expansion — VALU instruction count grows only 4.4%, VALUBusy is flat, the kernel just *waits* 2.46× longer for an event that doesn't happen.

**Three candidate mechanisms for the WAIT-only signature** (predictions for R57+ to attack):

(a) **Scale operand vmcnt latency on critical path** — the +12.5% extra VMEM_RD instructions arrive late, and the MFMA dispatch stalls on `s_waitcnt vmcnt(0)` for the scale operand even though the data tile already arrived. Falsifiable by counter that distinguishes vmcnt vs lgkmcnt waits, or by ISA inspection of waitcnt placement.

(b) **Scale-tile LDS write-then-read across the MFMA dispatch boundary** — the V2 CRR scale layout still does an LDS round-trip per K-pair (R55E showed LDS-resident layout regresses, but the *baseline* V2 CRR uses LDS staging too). The lgkmcnt stall on the LDS read after staging may dominate. Falsifiable by `SQ_WAIT_INST_LDS_cyc4` — note this is actually -17.8% in MXFP8 (less LDS-wait), which weakens this hypothesis.

(c) **Scale-pack swizzle dependency chain serializes via SALU+vmcnt** — the +34% SALU expansion AND the -29.9% fetched bytes together suggest the kernel computes more scale-tile addresses (more SALU index math) but issues more efficient (coalesced) loads. The dependency chain on these SALU computations may serialize via `s_waitcnt`-style scalar memory waits. Falsifiable by `SQ_WAIT_INST_LGKM` or vector vs scalar memory wait breakdown.

## Lever choice and rationale

The 70B Gate/Up CRR cell (R54 reviewer measured 88.95% MX/FP8, R56B PMC capture confirms 86.46%) was the **largest single-cell open gap in the 9-cell baseline sweep** AND the only un-PMC'd open HEADROOM cell:

- 70B Down RCR was profiled by R54B → DIAGNOSTIC-SCALE-FETCH-WAIT
- 8B GU RRR was profiled by R55F → DIAGNOSTIC-SCALE-FETCH-WAIT-SALU-DOMINANT
- **70B GU CRR was unprofiled and structurally interesting** — biggest gap, CRR layout has the 6× `v_lshrrev_b32` conditional shift block as a known FP8+MXFP8 floor (R52C, R54A/E/F/I 6× closure)

Three cells were captured to triangulate the bottleneck class:

- **70B Gate/Up (N=28672, K=8192):** primary HEADROOM cell — MX/FP8 86.5%
- **70B Q/O (N=8192, K=8192):** same K, smaller N — MX/FP8 96.3% (+9.8pp better) — isolates N-dim contribution
- **70B Down (N=8192, K=28672):** large K, R54B's RCR primary cell shape, but on CRR — MX/FP8 87.7% — isolates K-dim contribution

Two PMC sets collected (rocprofv3 max 8 counters/run):

- **set1** — utilization (`SQ_WAVES, SQ_INSTS_*, SQ_WAIT_*, SQ_BUSY_CYCLES, SQ_VALU_MFMA_BUSY_CYCLES, GRBM_GUI_ACTIVE, TCC_HIT/MISS, TCP_PENDING/RFIFO/READ_TAGCONFLICT_STALL, SQ_LDS_BANK_CONFLICT, TA_TA_BUSY, SQ_ACTIVE_INST_*, GRBM_TA/TC_BUSY, VALUBusy, MfmaUtil, MemUnitStalled, LDSBankConflict, FetchSize`) — copied verbatim from R54B set1
- **set3** — stall attribution (`TA_ADDR_STALLED_BY_TC/TD, TA_DATA_STALLED_BY_TC, TCP_TCP_TA_DATA_STALL, TCC_BUBBLE/BUSY, TCP_PENDING_STALL/RFIFO_STALL, SQ_LDS_DATA_FIFO_FULL, SQ_VMEM_TA_ADDR_FIFO_FULL`) — copied verbatim from R54B set3

Note: this rocprofv3 build does not provide derived metrics `[ALUStalledByLDS, L2CacheHit, MemUnitBusy]` — `TCC_hit_pct` (computed from raw `TCC_HIT_sum`/`TCC_MISS_sum`) is the L2-hit proxy throughout.

## Phase 0 — Setup and capture mechanics

- `pmc_set1.txt`, `pmc_set3.txt` — counter-group manifests (≤8 counters/group), copied verbatim from R54B.
- `run_pmc.sh` — capture orchestrator. Per cell × dtype × set: builds the shape-specific kernel once (cached via `${SO}.shape` marker), runs PMC capture under `rocprofv3`, dumps per-dispatch CSV. `MXFP8_WARMUP=100 ITERS=200`, `MXFP8_PRESHUFFLE_QUANT=1`, `MXFP8_CRR_PRESHUFFLE_V2_RUNTIME=1`, 30s cooldown between rocprof invocations. Test scripts copied (not symlinked) per build dir to avoid the R55F symlink-resolution memory fault.
- `aggregate.py` — per-cell loader. Filters by kernel-name tag (`crr_exact_8wave_scaled_kernel` for V2 MXFP8, `crr_exact_8wave_kernel` for FP8), drops `gemm_tail_kernel` fallback via 5000us duration ceiling, computes per-counter medians across dispatches, computes derived TFLOPS / MfmaUtil / TCC_hit_pct etc., prints per-cell comparison and summary table. Also dumps `aggregated.json`.

Capture artifact tree: `r56b_pmc_results/{cell}_{dtype}_crr_set{1,3}/pmc_{1..5}/` contains rocprofv3 CSV outputs (3 cells × 2 dtypes × 2 sets × 5 dispatches ≈ 60 dispatches captured).

## Phase 1 — Per-cell PMC delta evidence

### Cell A — 70B Gate/Up (PRIMARY, M=4096 N=28672 K=8192)

```
Metric                                    FP8        MXFP8       MX-FP8       MX/FP8
duration_us                              651.0       752.9        +101.9        1.157
achieved_TFLOPS                         2955.7      2555.7        -400.1      0.86464  <-- the gap
GRBM_GUI_ACTIVE                       1.001e7     1.415e7      +0.414e7      1.414
SQ_BUSY_CYCLES                        3.885e7     5.564e7      +1.680e7      1.432
SQ_INSTS_VALU                         7.574e7     7.906e7      +0.333e7      1.044  <-- nearly flat
SQ_INSTS_MFMA                         2.936e7     2.936e7         0          1.000  <-- identical
SQ_INSTS_LDS                          4.404e7     4.404e7         0          1.000  <-- identical
SQ_INSTS_VMEM_RD                      7.340e6     8.258e6      +0.918e6      1.125  <-- +12.5% (=9/8 canonical)
SQ_INSTS_SALU                         2.780e7     3.726e7      +0.946e7      1.340  <-- +34.0% SALU
SQ_VALU_MFMA_BUSY_CYCLES              9.395e8     9.395e8         0          1.000  <-- identical
SQ_WAIT_INST_LDS_cyc4                 4.918e7     4.044e7      -0.874e7      0.822  <-- LDS wait DECREASES
SQ_WAIT_INST_ANY_cyc4                 2.839e8     2.563e8      -0.277e8      0.903  <-- INST any wait DECREASES
SQ_WAIT_ANY_cyc4                      1.658e8     4.074e8      +2.416e8      2.457  <-- DOMINANT, +145.7%
SQ_ACTIVE_INST_VALU                   7.574e7     1.084e8      +3.270e7      1.432  <-- +43.2%
SQ_ACTIVE_INST_VMEM                   7.340e6     9.627e6      +2.287e6      1.312
SQ_ACTIVE_INST_LDS                    4.404e7     4.404e7         0          1.000
TCC_HIT_sum                           4.795e7     5.470e7      +0.675e7      1.141
TCC_MISS_sum                          1.813e7     1.322e7      -0.491e7      0.729  <-- L2 misses DROP
TCC_hit_pct                           72.56       80.54        +7.98pp       1.110  <-- MXFP8 hit rate IMPROVES
FetchSize_KB                          1013984.6   711246.2    -302738.4      0.701  <-- MXFP8 fetches 30% LESS
VALUBusy_pct                          22.89       23.65        +0.76pp       1.033  <-- nearly flat
MfmaUtil_pct                          70.98       51.23       -19.75pp       0.722  <-- MFMA collapses
LDSBankConflict_pct                   0.000       0.000          0           n/a    <-- not LDS
TA_ADDR_STALL_TC_per_grbm             2.619       2.178        -0.441        0.831
TA_DATA_STALL_TC_per_grbm             0.645       0.214        -0.431        0.331  <-- LESS data stall
TCP_TA_DATA_STALL_per_grbm            0.650       0.391        -0.259        0.601
TCP_PENDING_per_grbm                  18.020      18.858       +0.838        1.047
TCP_RFIFO_per_grbm                    0.048       0.001        -0.047        0.014
TCC_BUBBLE_per_grbm                   0           0              0           n/a
TCC_BUSY_per_grbm                     15.704      15.764       +0.060        1.004
VMEM_FIFO_FULL_per_grbm               0.871       0.986        +0.115        1.132
LDS_FIFO_FULL_per_grbm                0           0              0           n/a
LDS_BANK_CONFLICT_per_grbm            0           0              0           n/a
```

**Diagnostic claims from the deltas:**

1. **L2 / fetch bandwidth is NOT the bottleneck** — actually IMPROVES in MXFP8.
   - TCC_hit_pct +7.98pp (72.56% → 80.54%)
   - FetchSize -29.9% (1.01 GB → 711 MB) — MXFP8 reuses scale tiles *and* the L2 footprint shrinks
   - TA_DATA_STALL_TC -66.9%, TCP_TA_DATA_STALL -39.9%, TCP_RFIFO -98.6%
   - All address+data pipeline stalls *down* in MXFP8

2. **LDS / bank-conflict is NOT the bottleneck.**
   - LDSBankConflict_pct = 0.000 in both
   - SQ_INSTS_LDS identical (44.04M each)
   - LDS_FIFO_FULL_per_grbm = 0 in both
   - SQ_WAIT_INST_LDS *decreases* by -17.8% in MXFP8 (49.2M → 40.4M cyc/4)

3. **MMA-side instruction stream is NOT the bottleneck.**
   - SQ_INSTS_MFMA identical (29.36M)
   - SQ_VALU_MFMA_BUSY_CYCLES identical (939.5M)
   - The MMA dispatch count is unchanged

4. **VALU work-side is NOT the bottleneck (departing from R54B/R55F!).**
   - SQ_INSTS_VALU only +4.4% (R54B/R55F both +17-18%)
   - VALUBusy only +0.76pp (R54B was +5pp, R55F was +7.5pp)
   - The VALU pipe is *not* doing much extra work — yet MFMA collapses by -19.75pp

5. **The bottleneck IS pure WAIT — kernel waves spend 2.46× more cycles in `s_waitcnt`-class drain.**
   - SQ_WAIT_ANY +145.7% (+241.6M cyc/4) — **2.46× the FP8 baseline**
   - SQ_BUSY_CYCLES +43.2% (kernel runs longer overall)
   - GRBM_GUI_ACTIVE +41.4% (frontend active for 41% more time)
   - But neither VALU instruction count nor VALUBusy% reflects extra work — the wait cycles are pure idle drain

### Cell B — 70B Q/O (N=8192, K=8192, "control" cell)

```
Metric                                    FP8        MXFP8       MX-FP8       MX/FP8
achieved_TFLOPS                       2933.1      2823.6        -109.5      0.96266  <-- 96.3% (passes 95%)
SQ_INSTS_VALU                         2.149e7     2.259e7      +0.110e7      1.051
SQ_INSTS_SALU                         8.163e6     1.064e7      +2.478e6      1.304  <-- similar SALU growth
SQ_INSTS_MFMA                         8.389e6     8.389e6         0          1.000
SQ_INSTS_VMEM_RD                      2.097e6     2.359e6      +0.262e6      1.125
SQ_WAIT_ANY_cyc4                      4.065e7     5.400e7      +1.335e7      1.329  <-- much smaller (+32.9%)
SQ_ACTIVE_INST_VALU                   2.149e7     3.098e7      +0.949e7      1.442
VALUBusy_pct                          22.26       28.17        +5.91pp       1.266  <-- VALU rises here
MfmaUtil_pct                          69.50       61.02        -8.48pp       0.878
TCC_hit_pct                           80.57       80.59        +0.02pp       1.000  <-- L2 unaffected
FetchSize_KB                          196670      202837       +6167         1.031
SQ_WAIT_INST_LDS_cyc4                 1.479e7     1.892e7      +0.413e7      1.279
```

**Inflection: at the same K=8192 but smaller N (8192), the cell sits at 96.3% MX/FP8 — passes the 95% gate. The dominant bottleneck shifts:**
- SQ_WAIT_ANY ratio drops from 2.457 (70B GU) to 1.329 (70B QO) — **the WAIT-DOMINANT signature is N-driven, not just shape**
- VALUBusy actually *rises* +5.91pp here (closer to the R54B/R55F signature)
- TCC hit/FetchSize essentially unchanged — the L2 fix in 70B GU MXFP8 doesn't appear here because the working set already fits

**Interpretation:** at N=8192 the V2 CRR scale-fetch+stage pipeline overlaps cleanly with MFMA dispatch. At N=28672 (3.5× larger N tile), the per-CTA work increases enough that the scale-fetch wait can no longer overlap with MFMA — MFMA pipe goes idle waiting on scale-operand readiness. This is consistent with N-dim driving the gap.

### Cell C — 70B Down CRR (N=8192, K=28672, large-K reference)

```
Metric                                    FP8        MXFP8       MX-FP8       MX/FP8
achieved_TFLOPS                       3123.7      2740.8        -382.8      0.87744  <-- 87.7% (HEADROOM)
SQ_INSTS_VALU                         7.123e7     7.436e7      +0.314e7      1.044  <-- flat (like 70B GU)
SQ_INSTS_SALU                         2.943e7     3.620e7      +0.677e7      1.230  <-- +23%
SQ_INSTS_VMEM_RD                      7.344e6     8.258e6      +0.913e6      1.124
SQ_WAIT_ANY_cyc4                      1.446e8     3.454e8      +2.008e8      2.389  <-- 2.39× (similar to 70B GU 2.46×)
VALUBusy_pct                          23.06       24.78        +1.72pp       1.075  <-- nearly flat (like 70B GU)
MfmaUtil_pct                          76.03       56.10       -19.92pp       0.738  <-- collapse like 70B GU
TCC_hit_pct                           81.01       80.05        -0.96pp       0.988  <-- essentially flat
FetchSize_KB                          688200      759303      +71103         1.103  <-- MXFP8 fetches 10% MORE
TA_ADDR_STALL_TC_per_grbm             0.913       1.711       +0.798         1.874  <-- ADDR stall RISES
VMEM_FIFO_FULL_per_grbm               0.197       0.360       +0.163         1.825  <-- VMEM FIFO fills more
```

**70B Down CRR shows the SAME WAIT-DOMINANT signature** (SQ_WAIT_ANY 2.39×, VALU 1.04×, VALUBusy +1.7pp, MfmaUtil -19.9pp) — but:
- L2 hit rate does NOT improve here (-0.96pp; 70B GU showed +7.98pp)
- FetchSize *increases* +10% in MXFP8 (vs -30% at 70B GU)
- ADDR stall RISES +87.4%, VMEM FIFO fills +82.5%

**Interpretation:** At large K (28672), the 30% L2 footprint reduction that 70B GU enjoyed disappears because the K-dim itself drives data fetch volume. The wait-cycle expansion is preserved (2.39× vs 2.46×) but the L2-fix bonus is K-attenuated. **This means the WAIT-DOMINANT class is K-orthogonal — it shows up at both K=8192 (70B GU) and K=28672 (70B Down) once N is large enough.**

### Per-cell summary

```
Cell             N       K       MX/FP8    SQ_WAIT_ANY ratio   VALUBusy delta   MfmaUtil delta   TCC_hit delta
70B_GateUp     28672   8192     86.5%      2.457                +0.76pp          -19.75pp         +7.98pp
70B_QO          8192   8192     96.3%      1.329                +5.91pp           -8.48pp          +0.02pp     <-- PASSES gate
70B_Down        8192  28672     87.7%      2.389                +1.72pp          -19.92pp          -0.96pp
```

**Pattern:** the WAIT-DOMINANT signature (WAIT_ANY ratio ≥ 2.0, VALUBusy delta ≤ +2pp, MfmaUtil drop ≥ -19pp) appears in both the 70B_GateUp (large N) and 70B_Down (large K) cells, but NOT in 70B_QO (where neither N nor K are large). The 70B_QO cell exhibits the more familiar work-side-expansion class (R54B/R55F-like). **Conclusion:** the third bottleneck class is triggered when (N × K) per CTA exceeds some threshold — when the per-tile work amortizes across enough MFMA dispatches to *expose* the scale-fetch latency floor on the critical path.

## Phase 2 — Resource summary table

PMC capture is observational; no build resource delta exists for this work. Reference resource summaries (from R52/R53/R54 baseline):

| Kernel | VGPR | Spill | LDS | Occupancy |
|---|---:|---:|---:|---:|
| `crr_exact_8wave_kernel` (FP8 baseline) | (per CRR baseline) | 0 | (per CRR baseline) | 2 |
| `crr_exact_8wave_scaled_kernel<true,2>` (V2 MXFP8 CRR) | (per V2 CRR) | 0 | (per V2 CRR) | 2 |

V2 CRR is at the CRR floor 6× closed (R54A/R54E/R54F/R54I): the 6× `v_lshrrev_b32` conditional shift block is paid by both FP8 and MXFP8 baselines (not the marginal MX gap). LDS-resident scale layout (R55E) was REFUTED-EMPIRICAL — the LDS round-trip dominates the 6-cycle v_lshrrev saving (-7.98% median).

## Reasoning — what the PMC delta tells us

The dominant signature on 70B Gate/Up CRR is:

```
SQ_WAIT_ANY:           +145.7%  (2.46× FP8) ← DOMINANT, qualitatively distinct
SQ_INSTS_SALU:         +34.0%
SQ_ACTIVE_INST_VALU:   +43.2%
SQ_INSTS_VALU:         +4.4%    ← FLAT (R54B/R55F both +17%)
VALUBusy:              +0.76pp  ← FLAT (R54B +5pp, R55F +7.5pp)
MfmaUtil:              -19.75pp ← largest collapse of the three classes
SQ_INSTS_VMEM_RD:      +12.5%   (canonical 9/8)
SQ_INSTS_MFMA:          0%
SQ_INSTS_LDS:           0%
TCC_hit_pct:           +7.98pp  ← MXFP8 IMPROVES L2
FetchSize:             -29.9%   ← MXFP8 fetches 30% LESS
LDSBankConflict:        0
TA_DATA_STALL_TC:      -66.9%   ← MXFP8 has LESS data-stall
```

**Bottleneck class identification: WAIT-DOMINANT, work-side-FLAT.** The V2 CRR scaled kernel adds +12.5% scale-fetch reads and +34% SALU index-math, but **does almost no extra VALU work** (+4.4% instructions, +0.76pp busy). Yet the kernel waits 2.46× longer in `s_waitcnt`-class drain, and MFMA pipe utilization collapses by -19.75pp.

**The structural explanation that fits all three diagnostic cycles** (R54B RCR / R55F RRR / R56B CRR):

| Layout | Bottleneck mechanism | Marginal MX delta concentrated in |
|---|---|---|
| RCR (R54B) | Scale-fetch arrival latency on the critical path; data-side already fine | WAIT + VALU work-expansion (similar magnitude) |
| RRR (R55F) | Scale-pointer SALU arithmetic + scale-derived VALU dependency chain | SALU explosion (+50%) + VALU active cycles (+84%) |
| **CRR (R56B)** | **Scale-fetch latency exposed by large per-CTA work; no SALU/VALU work-expansion to hide behind** | **Pure WAIT (+146%) with flat VALU work** |

**The CRR class is the "late scale-arrival on critical path" class** — it's qualitatively distinct because the V2 CRR kernel is already so efficient on the work-side (no extra VALU work to do) that the wait-cycle blocking dominates without any cover. The kernel is *idling* on `s_waitcnt`, not *computing*. R55E's LDS-resident layout attempt to compress the scale-handling hit a wall because moving scale to LDS just trades vmcnt waits for lgkmcnt waits (-7.98% confirmed regression).

**This bottleneck is FUNDAMENTALLY DIFFERENT from the CRR scale-shift floor.** The 6× `v_lshrrev_b32` shift block (R49A/R50A/R53A/R54A/R54E/R54F/R54I 6× closure) is the **CRR vs RRR** layout floor — paid by both FP8 and MXFP8 in CRR. R56B's WAIT-DOMINANT class is the **CRR MXFP8 vs CRR FP8** delta — a separate, marginal effect orthogonal to the shift floor. **Closing the shift floor would NOT close the WAIT-DOMINANT gap.**

## Falsifiable predictions for future cycles (R57+)

These are what should be attempted next on the 70B Gate/Up CRR cell to target the identified WAIT-DOMINANT class:

- **P1.** A lever that issues scale-tile loads earlier in the pipeline (e.g., async copy via a separate queue, or hoisting scale `buffer_load_*` ahead of the data `buffer_load_*` by ≥ N cycles via `sched_barrier(0)` placement) should reduce SQ_WAIT_ANY by ≥ 30% on 70B GU CRR. Predicted TFLOPS gain: ≥ +1.5pp on MX/FP8 ratio. **Falsification:** WAIT_ANY drops ≥30% but TFLOPS flat → wait is in `s_waitcnt` of a different class (e.g., lgkmcnt, not vmcnt) or scoreboard.

- **P2.** A lever that adds an explicit `s_waitcnt vmcnt(N)` barrier at the scale-fetch boundary AND splits the K-pair body to issue scale loads two K-pairs ahead of consumption (double-buffer scale tiles in VGPR) should specifically attack the late-arrival mechanism. Predicted MfmaUtil recovery: +5-10pp (from 51% toward 60%). **Falsification:** MfmaUtil flat after the change → scale arrival is not the gating dependency (look at SALU dependency chain instead).

- **P3.** A lever that *removes* the +34% SALU index math (e.g., precompute scale-tile base + per-K-pair stride into scratch SGPRs in prologue, similar to R55B SALU_HOIST attempt) should reduce SQ_INSTS_SALU by ≥ 20%. R55B already showed compiler LICM-hoists scale SRDs to prologue on RCR — does it on CRR too? Worth ISA-verifying. **Falsification:** SALU drops ≥20% but WAIT_ANY flat → SALU is in the address pipeline but not on the critical path (consistent with R55B finding for RCR).

- **P4.** Compare V1 CRR (preshuffle V1, no wave-tile reorder) vs V2 CRR on 70B Gate/Up — does V1 hit the same WAIT-DOMINANT class? V1 issues scalar `buffer_load_dword` for scales (no b128/b64 burst), so latency arrival per scale should be even worse. If V1 has worse WAIT_ANY but same MfmaUtil, then V2's wave-tile reorder is already doing the right pipelining and the residual is hardware-dominated. **Falsification:** V1 has same WAIT_ANY as V2 → wave-tile reorder is not the determining factor.

- **P5.** Test V2 CRR on a smaller N variant of 70B Gate/Up (M=4096 N=14336 K=8192 — half the N tile). Does MX/FP8 climb back toward 96% (matching 70B QO)? If yes, the bottleneck is N-tile-driven and reducing wave-tile N coverage may help. **Falsification:** half-N still sits at ~87% → bottleneck is K-driven (since K=8192 is shared with 70B QO at 96%, this would be inconsistent — but worth verifying).

## Strong negative predictions (what NOT to try on V2 CRR 70B Gate/Up)

- **NOT** any LDS-side restructuring (LDSBankConflict = 0; LDS instruction count identical FP8 vs MXFP8; LDS_WAIT actually *decreases* in MXFP8; LDS is not the bottleneck — already confirmed by R55E REFUTED-EMPIRICAL).
- **NOT** any L2/cache-pressure mitigation on 70B Gate/Up (TCC_hit_pct IMPROVED +7.98pp in MXFP8; FetchSize -29.9%; L2 is *better* in MXFP8).
- **NOT** any further scale-pack `>> 16` shift removal — 6× closed (R54A/E/F/I); the shifts are paid by both FP8 and MXFP8 (CRR layout floor), not the marginal MX delta.
- **NOT** any K-loop body restructure that adds VGPR pressure (V2 CRR shares the architecture pattern with V2 RRR and likely sits near a VGPR ceiling — verify before touching).
- **NOT** any VALU-side dependency-chain break (R55C-style sched_barrier(0) before setprio sites) — VALU instruction count is essentially flat (+4.4%), VALUBusy is essentially flat (+0.76pp); there's no VALU dependency chain to break.
- **NOT** any cachepolicy modifier on data loads (data-side is already fine — TA_DATA_STALL drops, TCP_RFIFO drops, fetched bytes drop; cachepolicy axis already triply closed for V2 RCR scale loads at R47C/R48F/R55D).

## Files

- `r56b_pmc_results/aggregate_output.txt` — primary human-readable per-cell comparison table (the table that drives this synthesis).
- `r56b_pmc_results/aggregated.json` — full per-cell raw + derived counters in JSON form.
- `r56b_pmc_results/aggregate.py` — aggregator script (kernel-name filtering: `crr_exact_8wave_scaled_kernel` for V2 MXFP8, `crr_exact_8wave_kernel` for FP8 baseline; fallback-kernel duration ceiling at 5000us; median-across-dispatches; set1+set3 merge; 3-cell sweep).
- `r56b_pmc_results/pmc_set1.txt`, `pmc_set3.txt` — counter-group manifests submitted to rocprofv3 (verbatim from R54B/R55F).
- `r56b_pmc_results/run_pmc.sh` — capture orchestrator (per-cell × dtype × set: build, PMC capture under rocprofv3, CSV output, 30s cooldown between rocprof invocations). MXFP8_WARMUP=100 ITERS=200, MXFP8_CRR_PRESHUFFLE_V2_RUNTIME=1, GPU=1.
- `r56b_pmc_results/orchestrator.log` — capture-run log.
- `r56b_pmc_results/{cell}_{dtype}_crr_set{1,3}/pmc_{1..5}/` — raw per-dispatch rocprofv3 CSVs (3 cells × 2 dtypes × 2 sets × 5 dispatches ≈ 60 dispatches captured).
- `r56b_workspace/` — per-shape build dirs (`builds/{LABEL}/`) with copied test_python.py / test_mxfp8_python.py and symlinked source `.cpp/.inc/.h/Makefile`.

No source files were modified. No commits to source were created. PMC profiling is strictly observational.

## Protocol note

PMC capture under rocprofv3 with multi-counter-set sweeps is rate-limited by GPU dispatch order; 5 dispatches per (cell, dtype, set) are taken with median aggregation. Test scripts copied per build dir (not symlinked) per R55F symlink-resolution memory-fault recovery.

This rocprofv3 build does not provide derived metrics `[ALUStalledByLDS, L2CacheHit, MemUnitBusy]` — `TCC_hit_pct` (computed from raw `TCC_HIT_sum`/`TCC_MISS_sum`) is used as the L2-hit proxy. LDSBankConflict_pct is reported (= 0 in all three cells, both dtypes).

This work follows the R47D / R53C / R54B / R55F precedent of using PMC capture as a **diagnostic** to scope the bottleneck class for downstream lever cycles, not as a SHIP gate by itself.

## Verdict line for cycle wrap

`R56 Dev B: 70B Gate/Up CRR PMC diagnostic — DIAGNOSTIC-CRR-NEW-CLASS-WAIT-DOMINANT — bottleneck class is pure SQ_WAIT_ANY expansion (+145.7%, 2.46× FP8) with FLAT VALU work-side (SQ_INSTS_VALU +4.4%, VALUBusy +0.76pp); MfmaUtil collapses -19.75pp (worst of three diagnostic cycles); SALU +34%, VMEM_RD +12.5% canonical, MFMA/LDS instruction streams identical; NOT cache (TCC_hit IMPROVES +7.98pp, FetchSize -29.9%), NOT LDS (bank-conflict 0, LDS instr identical, LDS_WAIT decreases); structurally distinct from R54B RCR (work-side expansion shares the WAIT cost) and R55F RRR (SALU 3.4× R54B); cross-shape: 70B QO at same K=8192 sits at 96.3% with WAIT only +32.9% (N-driven trigger), 70B Down at K=28672 reproduces the WAIT-DOMINANT signature (WAIT 2.39×) confirming K-orthogonal trigger; mechanism predicted as scale-fetch arrival latency exposed on critical path when (N × K) per CTA is large; opens R57+ attack family on scale-tile early-issue / double-buffer / vmcnt placement; CRR shift floor (6× v_lshrrev) confirmed orthogonal — paid by both FP8 and MXFP8 baselines (R54I 6× closure stands).`
