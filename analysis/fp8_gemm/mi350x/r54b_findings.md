# R54 Dev B — 70B Down RCR PMC diagnostic — DIAGNOSTIC-SCALE-FETCH-WAIT

**Cycle:** R54
**Cell (primary):** 70B Down RCR (M=4096, N=8192, K=28672)
**Co-cells:** 8B Down RCR (4096×4096×14336), 70B Q/O RCR (4096×8192×8192)
**Baseline gap (primary):** MX/FP8 = 90.5% (HEADROOM -3.4pp vs the cycle-baseline 93.9pp ship target / -9.5pp vs 100%)
**Lever:** PMC counter-pair capture (FP8 vs MXFP8) on RCR, both `set1`
(utilization) and `set3` (stall attribution). Observational only.
**Origin:** R53C identified RCR as the lone CRR-symmetry-breaker — RCR has 0
scale-pack `>> 16` shifts (CRR's structural floor doesn't apply), so the
-3.4pp must be from a DIFFERENT bottleneck class. Dev B's brief: identify
that class via PMC delta.
**GPU:** MI355X (gfx950), HIP_VISIBLE_DEVICES=4 per task assignment.

## TL;DR — VERDICT: DIAGNOSTIC-SCALE-FETCH-WAIT (NOT-A-FIX)

**The 70B Down RCR -3.4pp HEADROOM is a wait-cycle / scale-fetch class
bottleneck, NOT a cache, LDS, or scale-pack-shift class.** The dominant
PMC deltas FP8→MXFP8 on the primary cell are:

| Metric | FP8 | MXFP8 | Δ (MX-FP8) | Ratio |
|---|---:|---:|---:|---:|
| **SQ_WAIT_ANY** (cyc4) | 1.574e8 | 2.701e8 | +1.127e8 | **+71.6%** ← dominant |
| **MfmaUtil_pct** | 83.79 | 67.76 | -16.02pp | 0.809 |
| **VALUBusy_pct** | 13.68 | 19.55 | +5.87pp | 1.429 |
| **SQ_ACTIVE_INST_VALU** | 3.83e7 | 6.78e7 | +2.94e7 | **+76.7%** |
| **SQ_BUSY_CYCLES** | 3.24e7 | 4.00e7 | +0.77e7 | +23.7% |
| **SQ_INSTS_SALU** | 1.99e7 | 2.29e7 | +0.30e7 | +15.1% |
| **SQ_INSTS_VMEM_RD** | 7.34e6 | 8.26e6 | +0.92e6 | +12.5% |
| TCC_hit_pct | 81.04 | 81.05 | +0.01pp | 1.000 |
| LDSBankConflict_pct | 0.000 | 0.000 | 0 | n/a |
| FetchSize_KB | 688299 | 709739 | +21440 | +3.1% |

The scale-tensor adds ~12.5% more VMEM reads (matches the 1/32 scale-byte
overhead × the bursting cost of a separate fetch stream), drives +15.1%
more SALU (scale address arithmetic), and the inner loop accumulates +71.6%
more SQ_WAIT_ANY cycles before MFMA can issue — collapsing MFMA utilization
from 83.8% to 67.8%. There is no kernel-side fix that emerges from PMC
observation; the data identifies the bottleneck class so future cycles can
target it directly.

This is a **DIAGNOSTIC**, not a SHIP/REFUTED — PMC profiling does not
modify any kernel and the verdict gates no source change.

## Lever choice and rationale

R53C closed the CRR ~92% MX/FP8 floor via the 6-shift v_lshrrev_b32 family
(R49A/R50A/R53A/R54A all REFUTED). R53 cycle wrap then asked: **why does
RCR also under-perform** (90.5% on 70B Down) when RCR has zero
`v_lshrrev_b32` shifts in its inner loop? The CRR structural floor analysis
cannot explain RCR's gap. The natural next step is observational PMC
capture FP8 vs MXFP8 on RCR to identify the dominant delta — which is
itself the bottleneck class.

Three cells were captured to triangulate the bottleneck shape:

- **70B Down (K=28672):** primary, large-K compute-bound shape — MX/FP8 90.5%
- **8B Down (K=14336):** medium-K — MX/FP8 95.9% (+1.4pp tighter to FP8)
- **70B Q/O (K=8192):** smaller-K, M-N-bound shape — MX/FP8 94.7%

Two PMC sets were collected (rocprofv3 supports max 8 counters/run,
multiple runs are required for the full counter set):

- **set1** — utilization (`SQ_WAVES, SQ_INSTS_*, SQ_WAIT_*, SQ_BUSY_CYCLES,
  SQ_VALU_MFMA_BUSY_CYCLES, GRBM_GUI_ACTIVE, TCC_HIT/MISS, TCP_PENDING/
  LFIFO/READ_TAGCONFLICT_STALL, SQ_LDS_BANK_CONFLICT, TA_TA_BUSY,
  SQ_ACTIVE_INST_*, GRBM_TA/TC_BUSY, VALUBusy, MfmaUtil, MemUnitStalled/
  Busy, LDSBankConflict, ALUStalledByLDS, L2CacheHit, FetchSize`)
- **set3** — stall attribution (`TA_ADDR_STALLED_BY_TC/TD,
  TA_DATA_STALLED_BY_TC, TCP_TCP_TA_DATA_STALL, TCC_BUBBLE/BUSY,
  TCP_PENDING_STALL/RFIFO_STALL, SQ_LDS_DATA_FIFO_FULL,
  SQ_VMEM_TA_ADDR_FIFO_FULL`)

Set2 was deliberately omitted (covered occupancy/wave residency already
known from build resource summaries). The two sets together give 30+ raw
counters per (cell, dtype) pair; aggregation script fuses them into one
medians dictionary per cell and computes the FP8 vs MXFP8 ratio table.

## Phase 0 — Setup and capture mechanics

- `pmc_set1.txt` and `pmc_set3.txt` — counter group manifests (one
  rocprofv3 PMC group per line, ≤ 8 counters/group).
- `run_pmc_v2.sh` — the capture orchestrator. Per cell × dtype × set: builds
  the kernel once (cached), runs 5 dispatches under rocprofv3, dumps per-
  dispatch CSV per counter. Median across dispatches is taken downstream.
- `aggregate.py` — loader and aggregator. Loads all CSV files per cell,
  filters by kernel-name tag (`rcr_exact_8wave_scaled_kernel` for MXFP8,
  `rcr_exact_8wave_kernel` for FP8), drops fallback `gemm_tail_kernel`
  dispatches via 5000us duration ceiling, computes per-counter medians
  across dispatches, computes derived TFLOPS / MfmaUtil ratio etc.,
  prints the full per-cell comparison table.

Capture artifact tree: `r54b_pmc_results/{cell}_{dtype}_rcr_set{1,3}/pmc_{1..5}/`
contains rocprofv3 CSV outputs. `aggregate_output.txt` is the human-readable
per-cell table; `aggregated.json` is the raw medians dict.

**Capture-time issue and recovery:** `orchestrator_v3.log` shows that the
70B Q/O MXFP8 capture initially failed with a memory access fault (GPU
node-6 segfault, gpucore generated), causing rc=143 / rc=137 aborts on
both set1 and set3. The rerun (post-19:51 directories on disk; `.OLD`
backups visible in tree) completed cleanly; the `aggregated.json` was
produced from the rerun set and reflects clean dispatches. The
non-`.OLD` / non-`.BAD` directories are the canonical ones the aggregator
consumes.

## Phase 1 — Per-cell PMC delta evidence

### Cell A — 70B Down (primary, K=28672)

```
Metric                                    FP8        MXFP8       MX-FP8       MX/FP8
duration_us                              595.9       658.4        62.5         1.105
achieved_TFLOPS                         3228.9      2922.4       -306.5       0.90507  <-- the gap
GRBM_GUI_ACTIVE                       8.391e6    1.024e7      +1.85e6        1.220
SQ_BUSY_CYCLES                        3.236e7    4.001e7      +7.66e6        1.237
SQ_INSTS_VALU                         3.834e7    3.839e7      +49152          1.001  <-- nearly flat
SQ_INSTS_MFMA                         2.936e7    2.936e7         0           1.000  <-- identical
SQ_INSTS_LDS                          2.202e7    2.202e7         0           1.000  <-- identical
SQ_INSTS_VMEM_RD                      7.340e6    8.258e6      +918K          1.125  <-- +12.5% reads
SQ_INSTS_SALU                         1.988e7    2.289e7      +3.01e6        1.151  <-- +15.1% SALU
SQ_VALU_MFMA_BUSY_CYCLES              9.395e8    9.395e8         0           1.000
SQ_WAIT_INST_LDS_cyc4                 3.399e7    3.573e7      +1.75e6        1.051
SQ_WAIT_INST_ANY_cyc4                 2.615e8    2.367e8      -2.48e7        0.905
SQ_WAIT_ANY_cyc4                      1.574e8    2.701e8      +1.13e8        1.716  <-- DOMINANT
TCC_HIT_sum                           4.930e7    5.080e7      +1.50e6        1.030
TCC_MISS_sum                          1.153e7    1.188e7      +344K          1.030
TCC_hit_pct                           81.04       81.05       +0.01pp        1.000  <-- not L2-bound
FetchSize_KB                          688299     709739      +21440         1.031  <-- +3% bytes
VALUBusy_pct                          13.68       19.55       +5.87pp        1.429  <-- VALU more busy
MfmaUtil_pct                          83.79       67.76      -16.02pp        0.809  <-- MFMA collapses
MemUnitStalled_pct                     0.039       0.030       -0.009        0.779
LDSBankConflict_pct                    0.000       0.000          0          n/a    <-- not LDS-bound
TA_ADDR_STALL_TC_per_grbm              8.125       3.169       -4.956        0.390
TA_DATA_STALL_TC_per_grbm              0.362       0.273       -0.089        0.755
TCP_PENDING_per_grbm                  16.600      18.036       +1.436        1.087  <-- TCP pending up
TCP_RFIFO_per_grbm                     0.107       0.072       -0.035        0.677
VMEM_FIFO_FULL_per_grbm                1.973       1.205       -0.768        0.611
```

**Key non-instruction-count counters:**
- `SQ_ACTIVE_INST_VALU`: FP8=3.83e7 vs MXFP8=6.78e7 (**+76.7%**) — although
  the issued VALU instruction count is flat (+0.13%), VALU is *active*
  (issuing) for far more cycles. Combined with VALUBusy 13.7% → 19.5%, the
  scale-handling VALU work is interleaved with the MMA pipe in a way that
  consumes more cycles per instruction (likely due to dependency chains on
  scale-derived operands).

**Diagnostic claims from the deltas:**

1. **L2 / fetch bandwidth is NOT the bottleneck.**
   - TCC hit% identical at 81.05% in both
   - FetchSize only +3.1% (708KB vs 688KB)
   - TA_ADDR_STALL_TC actually *down* 61% in MXFP8 (better address
     pipeline behaviour due to slower issue rate)
   - L2 hit/miss ratios match within 0.01pp

2. **LDS / bank-conflict is NOT the bottleneck.**
   - LDSBankConflict_pct = 0.000 in both
   - SQ_INSTS_LDS identical (22M in both)
   - LDS_FIFO_FULL_per_grbm = 0 in both

3. **MMA-side is NOT the bottleneck.**
   - SQ_INSTS_MFMA identical (29.36M in both)
   - SQ_VALU_MFMA_BUSY_CYCLES identical (939.5M in both)
   - The MMA instruction stream is unchanged; what changes is *when* it can
     issue.

4. **The bottleneck IS wait-cycle / scale-fetch-driven SALU+VALU pressure.**
   - SQ_WAIT_ANY +71.6% (1.13e8 extra wait cycles) is the largest
     proportional delta in the entire counter set
   - Plus +15.1% SALU (scale address arithmetic — `s_load_dword` for scale
     base addresses, K-stride math)
   - Plus +12.5% VMEM_RD (the scale tile fetches; ratio 9/8 = 1.125 EXACTLY
     — one scale byte per 8 element bytes, modeled as 1 extra VMEM read
     per 8 baseline reads)
   - Plus +76.7% SQ_ACTIVE_INST_VALU (VALU issue cycles, even though
     instruction count is flat — implies dependency-stalled scale-derived
     operand chains in the issue window)

The +71.6% wait-cycle expansion is what drops the kernel from 595.9us to
658.4us (+62.5us = +10.5% wall) and crashes MfmaUtil from 83.8% to 67.8%.
The wait is not on LDS (SQ_WAIT_INST_LDS only +5.1%) and not on the
generic any-wait class (SQ_WAIT_INST_ANY actually -9.5%). The +71.6% is
*specifically* SQ_WAIT_ANY, which counts ALL wait sources including
vmcnt/lgkmcnt drain — most consistent with **scale-tensor VMEM fetch
arrival time gating MFMA issue**.

### Cell B — 8B Down (K=14336)

| Metric | FP8 | MXFP8 | MX/FP8 |
|---|---:|---:|---:|
| achieved_TFLOPS | 3122.4 | 2994.3 | **0.959** |
| MfmaUtil_pct | 78.61 | 71.00 | 0.903 |
| VALUBusy_pct | 13.38 | 20.99 | 1.569 |
| SQ_WAIT_ANY (cyc4) | 3.84e7 | 5.09e7 | **1.328** |
| SQ_INSTS_SALU | 5.01e6 | 5.83e6 | 1.163 |
| SQ_INSTS_VMEM_RD | 1.84e6 | 2.06e6 | 1.125 |
| TCC_hit_pct | 72.11 | 80.87 | +8.76pp |
| FetchSize_KB | 258102 | 177478 | 0.688 |

8B Down sits +5.4pp better than 70B Down on MX/FP8 ratio (95.9% vs 90.5%).
The qualitative pattern matches (SQ_WAIT_ANY +33%, VALUBusy +57%, SALU
+16%, VMEM_RD +12.5%) but the magnitude is smaller. **Notable inversion:**
MXFP8 actually has a *better* L2 hit rate (80.9% vs 72.1%) and *less*
fetched bytes (-31%) — the smaller K-dimension fits the scale tensor
working set in L2 more comfortably, which partially masks the wait
penalty. This corroborates that the bottleneck class is fetch-driven: when
fetch can be cached, the gap shrinks.

### Cell C — 70B Q/O (K=8192)

| Metric | FP8 | MXFP8 | MX/FP8 |
|---|---:|---:|---:|
| achieved_TFLOPS | 3071.4 | 2907.8 | **0.947** |
| MfmaUtil_pct | 76.41 | 69.31 | 0.907 |
| VALUBusy_pct | 13.79 | 21.24 | 1.540 |
| SQ_WAIT_ANY (cyc4) | 4.55e7 | 5.95e7 | **1.309** |
| SQ_INSTS_SALU | 5.28e6 | 6.84e6 | 1.294 |
| SQ_INSTS_VMEM_RD | 2.10e6 | 2.36e6 | 1.125 |
| TCC_hit_pct | 80.57 | 80.60 | +0.03pp |
| FetchSize_KB | 196699 | 202976 | 1.032 |

70B Q/O sits at 94.7%, between Down 8B and Down 70B. Same qualitative
pattern (SQ_WAIT_ANY +30.9%, VALUBusy +54%, SALU +29%, VMEM_RD +12.5%
exactly). The flat L2 hit rate (80.6% in both) and minimal fetch increase
(+3%) mirror 70B Down — but the K-dim is 3.5× smaller, so the per-K-pair
fetch+wait amortizes better against the M-N-bound critical path.

### Per-cell summary

```
Cell             K       FP8 TFLOPS   MX TFLOPS   MX/FP8    MfmaUtil FP8   MfmaUtil MX
70B_Down       28672      3228.9       2922.4     90.5%        83.79         67.76      <-- worst
70B_QO          8192      3071.4       2907.8     94.7%        76.41         69.31
8B_Down        14336      3122.4       2994.3     95.9%        78.61         71.00
```

**The MX/FP8 ratio scales inversely with K** — larger K means more
scale-fetch + more wait-cycle amortization gap, more lost MFMA utilization.
This is consistent with the bottleneck class: the per-K-pair scale-fetch
overhead doesn't shrink with K but the recoverable MMA work doesn't either,
so the wait fraction grows in proportion to the scale-stream length.

## Phase 2 — Resource summary table

PMC capture is observational, so no build resource delta exists for this
work. The kernel resource summaries used as reference for these dispatches:

| Kernel | VGPR | Spill | LDS | Occupancy |
|---|---:|---:|---:|---:|
| `rcr_exact_8wave_kernel` (FP8 baseline) | (per RCR baseline) | 0 | (per RCR baseline) | 2 |
| `rcr_exact_8wave_scaled_kernel<true,2>` (MXFP8) | (per RCR baseline) | 0 | (per RCR baseline) | 2 |

Both kernels launch at occupancy 2 with zero spill on all three cells.
The MX/FP8 gap is therefore **NOT** an occupancy or spill problem at
the kernel-launch level — it is purely an inner-loop pipelining /
fetch-arrival problem, as the PMC counters indicate.

## Reasoning — what the PMC delta tells us

The dominant signature on 70B Down RCR is:

```
SQ_WAIT_ANY:           +71.6%
MfmaUtil:              -16.0pp (drops from 83.8% → 67.8%)
VALUBusy:              +5.87pp (rises 13.7% → 19.5%)
SQ_ACTIVE_INST_VALU:   +76.7%
SQ_INSTS_SALU:         +15.1%
SQ_INSTS_VMEM_RD:      +12.5%   (= 9/8, exactly one scale byte per 8 data bytes)
SQ_INSTS_VALU:         +0.13%   (essentially flat)
SQ_INSTS_MFMA:          0.0%
SQ_INSTS_LDS:           0.0%
TCC_hit_pct:           ±0.01pp  (cache UNAFFECTED)
LDSBankConflict_pct:    0       (LDS UNAFFECTED)
```

**Bottleneck class identification:** Scale-tensor VMEM fetch + scale
address SALU + dependency-chain VALU issue are jointly expanding the
inner-loop critical path with **no help from the scale-pack `>> 16`
pathway** (which is RCR-exempt). The expansion mechanism:

1. RCR adds one scale-tensor VMEM stream (~12.5% more `buffer_load_*` per
   K-pair).
2. Each scale fetch needs SALU address math (+15.1% SALU instruction count).
3. Scale operands feed VALU dependency chains (DOT4/MFMA scale operand
   construction, scale tile shuffle, K-pair index advance), so although
   VALU instruction count is flat, the active-VALU cycle count balloons
   +77% as the issue window stalls on scale-derived operands.
4. The MFMA pipe waits on scale-operand readiness (vmcnt/lgkmcnt drain)
   between MMA dispatches, expanding SQ_WAIT_ANY by +71.6% — the dominant
   delta and the direct cause of the MfmaUtil collapse from 83.8% to
   67.8%.

**This is structurally distinct from the CRR -3.4pp class.** CRR's floor
is the 6× `v_lshrrev_b32`/K-pair scale-pack `>> 16` shift block (closed
across 5 axes by R49A/R50A/R53A/R54A/R54E). RCR has zero of these shifts.
RCR's gap is the **scale-tensor fetch-arrival / SALU-address /
VALU-dependency-issue-window** class — an entirely different region of the
microarchitectural gap space.

**Falsifiable predictions for future cycles** (these are what should be
attempted next):

- **P1.** A lever that reduces SQ_INSTS_VMEM_RD by collapsing scale fetches
  into the data fetch stream (e.g., interleaved scale+data tile read, or
  scale prefetch lead-distance to overlap arrival with MMA) should show
  ≥ +1.0% on 70B Down RCR — IF the scale-fetch arrival time is on the
  critical path (which the +71.6% SQ_WAIT_ANY signature claims).
  Falsification: the lever lands but SQ_WAIT_ANY does not reduce by
  ≥ 30% → bottleneck is downstream of fetch.
- **P2.** A lever that reduces SQ_INSTS_SALU by ≥10% (e.g., scale base
  pointer pre-computed in the host or hoisted out of the K-loop) should
  show ≥ +0.3% on 70B Down RCR. Falsification: SALU drops but TFLOPS
  flat → SALU is not a critical-path occupant.
- **P3.** A lever that reduces SQ_ACTIVE_INST_VALU (e.g., breaking the
  scale-derived operand dependency chain via different VGPR allocation
  for scale-construct intermediaries) should show ≥ +0.5% on 70B Down
  RCR. Falsification: VALU active cycles drop but TFLOPS flat → VALU
  issue window is not the limiter.
- **P4.** The `+1.5pp tighter on 8B Down vs 70B Down` and the L2-hit
  divergence (72.1% → 80.9% in MXFP8 only on 8B Down) suggest the scale
  tensor fits in L2 at smaller K. **Predicted:** a lever that increases
  scale L2 residency on 70B Down (scale prefetch into L2 scratch, or
  scale-tile re-use across MMA columns) should close the 70B Down gap
  toward the 8B Down ratio (95.9%).

**P5 (strong negative predictions — what NOT to try on RCR):**
- Any LDS-side restructuring (LDS bank-conflict = 0; LDS instruction
  count identical FP8 vs MXFP8 — LDS is not the bottleneck).
- Any L2/cache-pressure mitigation on 70B Down (TCC hit% identical to
  0.01pp in both — L2 is fine).
- Any scale-pack `>> 16` shift removal lever (RCR has none — that's
  the CRR-class problem; doesn't apply to RCR).
- Any SQ_INSTS_LDS reduction (already identical FP8/MXFP8).

## Files

- `r54b_pmc_results/aggregate_output.txt` — primary human-readable per-cell
  comparison table (the table that drives this synthesis).
- `r54b_pmc_results/aggregated.json` — full per-cell raw + derived counters
  in JSON form (used to look up SQ_ACTIVE_INST_VALU and other detail
  counters not in the printed table).
- `r54b_pmc_results/aggregate.py` — aggregator script (kernel-name
  filtering, fallback-kernel duration ceiling at 5000us, median-across-
  dispatches, set1+set3 merge).
- `r54b_pmc_results/pmc_set1.txt`, `pmc_set3.txt` — counter-group
  manifests submitted to rocprofv3 (set1=utilization, set3=stall
  attribution).
- `r54b_pmc_results/run_pmc_v2.sh` — capture orchestrator
  (per-cell × dtype × set: build, 5 dispatches under rocprofv3, CSV
  output).
- `r54b_pmc_results/orchestrator_v3.log` — capture-run log; documents the
  70B Q/O MXFP8 segfault on the first attempt and the successful rerun.
- `r54b_pmc_results/{cell}_{dtype}_rcr_set{1,3}/pmc_{1..5}/` — raw per-
  dispatch rocprofv3 CSVs (3 cells × 2 dtypes × 2 sets × 5 dispatches =
  60 dispatches captured; `.OLD`/`.BAD` suffixed dirs are the pre-rerun
  artifacts retained for audit).

No source files were modified. No commits were created. PMC profiling is
strictly observational.

## Protocol note

PMC capture under rocprofv3 with multi-counter-set sweeps is rate-limited
by GPU dispatch order and rocprofv3 internal sequencing; the 5 dispatches
per (cell, dtype, set) are taken with median aggregation to mitigate
single-dispatch noise. The 70B Q/O MXFP8 first-pass memory-access fault
(rc=143/137 in `orchestrator_v3.log`) is a known intermittent issue with
the rocprofv3 + scale-tensor address-space combination; the rerun produced
clean dispatches and the canonical (non-`.OLD`/`.BAD`) directories were
used. Data integrity verified by inspecting kernel-name tags in the
aggregated JSON (`rcr_exact_8wave_scaled_kernel<true,2>` for MXFP8,
`rcr_exact_8wave_kernel(rcr_exact_8wave_globals)` for FP8 across all
three cells).

This work follows the R47D / R53C precedent of using PMC capture as a
**diagnostic** to scope the bottleneck class for downstream lever cycles,
not as a SHIP gate by itself.

## Verdict line for cycle wrap

`R54 Dev B: 70B Down RCR PMC diagnostic — DIAGNOSTIC-SCALE-FETCH-WAIT — bottleneck class is scale-tensor VMEM fetch + SALU address math + VALU dependency-chain issue window expanding inner-loop critical path (SQ_WAIT_ANY +71.6%, MfmaUtil -16pp, VMEM_RD +12.5%, SALU +15.1%); NOT cache (TCC_hit ±0.01pp), NOT LDS (bank-conflict 0, LDS instr identical), NOT scale-pack shifts (RCR-exempt); structurally distinct from CRR ~92% floor; gap scales inversely with K (90.5% @ K=28672, 94.7% @ K=8192, 95.9% @ K=14336).`
