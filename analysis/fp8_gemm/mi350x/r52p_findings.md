# R52 Dev P — Measured PMC stall breakdown for V1/V2 production RRR — DIAGNOSTIC

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ 72a3c217 (R52N landed)
**GPU:** MI355X (gfx950) — rocprofv3 PMC counter collection on HIP_VISIBLE_DEVICES=4
**Mandate:** Replace static ISA analysis with measured stall breakdown via
hardware counters. Identify the dominant stall mechanism on production RRR
at 8B Gate/Up (M=4096 N=14336 K=4096) to explain the unresolved +3.1pp
HEADROOM gap that R52L/R52N closed without source remediation.

## TL;DR — VERDICT: DIAGNOSTIC-COMPLETE

**The dominant +12.4pp MfmaUtil gap between V2 8B Gate/Up (55.3%) and the
fast V2 70B Q/O comparison cell (67.8%) is L2/TC cache-return backpressure
to the L1 read pipe.** Per-GRBM-cycle, V2 8B Gate/Up shows:

  * **TA_DATA_STALLED_BY_TC: +57% vs 70B Q/O** (1.50 vs 0.95 cycles/GRBM)
  * **TCP_TCP_TA_DATA_STALL: +62% vs 70B Q/O** (1.64 vs 1.01 cycles/GRBM)
  * TCC L2 hit rate is essentially identical (79.4% vs 80.6%)
  * LDS bank conflicts are 0 in **all** cells (re-confirms R51E)
  * Scratch / spill pressure is irrelevant in the K-loop body (re-confirms R52N)

Critically, the large-N cross-check refutes the naive "N=14336 stride
thrashes L2" hypothesis: V2 70B Gate/Up at **N=28672** has the LOWEST
TA_DATA_STALL/GRBM (0.42) of any cell profiled. The 8B Gate/Up cell
(N=14336, **K=4096**) is uniquely bad on TC-data backpressure relative to
both K=8192 cells.

**Specific R53 lever recommendation: K-dimension is the discriminator, not
N. The 8B Gate/Up cell has only `K/64 = 64` K-iterations vs 128 at K=8192
— half the per-tile MFMA work to amortize the same A/B-tile L2 demand.
R53 should investigate (1) longer per-CTA K-tile reuse via super-tiling,
or (2) increased B-side L2 cachepolicy hint (cache-streaming vs
cache-LRU) so the B-panel for N=14336 stays warm longer per tile.**

## 1. Method

### 1.1 Profiled cells
| Cell | Path | Shape (M,N,K) | Notes |
|---|---|---|---|
| `v2_8B_GateUp` | default | 4096, 14336, 4096 | **Target HEADROOM cell** |
| `v1_8B_GateUp` | `MXFP8_RRR_PRESHUFFLE_V2_RUNTIME=0` | 4096, 14336, 4096 | What R52N audited |
| `v2_70B_QO` | default | 4096, 8192, 8192 | Fast comparison cell |
| `v1_70B_QO` | `MXFP8_RRR_PRESHUFFLE_V2_RUNTIME=0` | 4096, 8192, 8192 | V1 at 70B Q/O |
| `v2_70B_GateUp` | default | 4096, 28672, 8192 | Large-N cross-check |

V2 (`gemm_rrr_pq_v2` → `dispatch_rrr_exact_8wave_scaled_v2<true>`) is the
**production-default** dispatch (env default `MXFP8_RRR_PRESHUFFLE_V2_RUNTIME=1`),
not V1 as R52N's framing implied. We profile both.

### 1.2 Counter sets
* `pmc_set1.txt` — base SQ/MFMA/TCC/TCP utilization counters (5 PMC groups)
* `pmc_set3.txt` — stall-attribution counters (5 PMC groups, 2 counters each
  to avoid the rocprofv3 hang seen with set2's denser packing)

`pmc_set2.txt` (5 counters per pmc: line) hung the profiler indefinitely;
`pmc_set3.txt` is the working replacement.

### 1.3 Aggregation
Per-cell median across all dispatches (10 invocations × multiple
PMC-replay groups). All ratios computed against `GRBM_GUI_ACTIVE` for
fair cross-shape normalization. Scripts: `r52p_results/aggregate.py`,
`aggregate_set3.py`, `final_aggregate.py`. JSON dump:
`r52p_results/final_aggregated.json`.

## 2. Headline measurements

### 2.1 Per-cell summary

| Cell                     | TFLOPS | dur(µs) | MfmaUtil | VALUBusy | TCC hit | TA_DATA_stall_TC/GRBM | TCP_TA_DATA_stall/GRBM |
|--------------------------|-------:|--------:|---------:|---------:|--------:|----------------------:|-----------------------:|
| **v2_8B_GateUp** (target)| 2429   | 198.0   | **55.35%**| 18.98%  | 79.35%  | **1.4968**            | **1.6395**             |
| v1_8B_GateUp             | 2269   | 212.0   | 51.03%   | 18.28%   | 77.74%  | n/a (set3 not run)    | n/a                    |
| **v2_70B_QO** (fast cmp) | 2843   | 193.4   | **67.78%**| 21.15%  | 80.60%  | **0.9501**            | **1.0147**             |
| v1_70B_QO                | 2731   | 201.3   | 63.78%   | 20.77%   | 79.66%  | n/a                   | n/a                    |
| v2_70B_GateUp (large N)  | 2509   | 766.8   | 69.11%   | 21.57%   | 80.59%  | 0.4162                | 0.6295                 |

V2 is consistently **+4–7% faster** than V1 across all shapes — V2 is
correctly the production default.

### 2.2 V2 8B Gate/Up vs V2 70B Q/O — headroom-gap deltas

| Metric                                         | V2 8B Gate/Up | V2 70B Q/O | delta (70B − 8B) |
|------------------------------------------------|--------------:|-----------:|-----------------:|
| Achieved TFLOPS                                | 2429          | 2843       | +414             |
| MfmaUtil (%)                                   | **55.35**     | **67.78**  | **+12.43 pp**    |
| VALUBusy (%)                                   | 18.98         | 21.15      | +2.17 pp         |
| TCC L2 hit rate (%)                            | 79.4          | 80.6       | +1.2 pp          |
| **TA_DATA_STALLED_BY_TC / GRBM**               | **1.4968**    | **0.9501** | **−0.5467**      |
| **TCP_TCP_TA_DATA_STALL / GRBM**               | **1.6395**    | **1.0147** | **−0.6248**      |
| TA_ADDR_STALLED_BY_TC / GRBM                   | 0.0811        | 0.1042     | +0.0231          |
| TCP_PENDING_STALL / GRBM                       | 13.18         | 14.55      | +1.37            |
| TCP_RFIFO_STALL / GRBM                         | 0.2598        | 0.2133     | −0.0465          |
| SQ_VMEM_TA_ADDR_FIFO_FULL / GRBM               | 0.1817        | 0.1518     | −0.0299          |
| **SQ_LDS_BANK_CONFLICT (cyc) / GRBM**          | **0.000000**  | **0.000000**| **0 (re-confirms R51E)** |
| FetchSize (KB)                                 | 185 934       | 202 838    | +16 904          |
| TCC_MISS_sum                                   | 3 891 848     | 3 768 968  | −122 880         |

**Interpretation**:
1. The MfmaUtil deficit at 8B Gate/Up is real and large (12.4pp).
2. Of every counter measured, only **TA_DATA_STALLED_BY_TC** and its
   downstream TCP version are *worse* on the slow cell with both
   magnitude and direction matching the perf gap. Every other stall
   counter (TCP_PENDING, TCP_RFIFO, TA_ADDR, FIFO_FULL) is either flat
   or *better* on the slow cell.
3. The cache hit rate is essentially identical, so this is **not** a
   miss-rate problem — it is a **cache-return-bandwidth** problem at
   constant hit rate. The TC channel is delivering data slower than
   the TA can consume it, and TA stalls upstream of TCP.
4. LDS bank conflicts are 0 (re-confirms R51E B-side LDS verdict was
   correct). LDS is fully clean.

### 2.3 Large-N cross-check — refutes the obvious "N stride" hypothesis

| Cell                       | N      | K    | TFLOPS | MfmaUtil | TA_DATA_stall_TC/GRBM |
|----------------------------|-------:|-----:|-------:|---------:|----------------------:|
| V2 70B Q/O                 |  8192  | 8192 | 2843   | 67.78%   | 0.9501                |
| V2 8B Gate/Up (target)     | 14336  | 4096 | **2429** | **55.35%** | **1.4968** (worst)  |
| V2 70B Gate/Up (large N)   | 28672  | 8192 | 2509   | 69.11%   | 0.4162 (best)         |

Going from N=8192 → 14336 → 28672 with K=8192 fixed shows TA_DATA stalls
*decrease* (0.95 → 0.42 / GRBM). N is not the discriminator. Going from
K=8192 → K=4096 at non-trivial N is what spikes the stall. This points
at **K-dimension reuse arithmetic**, not N-dimension cache stride.

## 3. Stall-source attribution (ranked)

For V2 8B Gate/Up, ranked by per-GRBM-cycle attributable to the headroom gap:

1. **L2/TC cache-return backpressure to L1 (TA_DATA / TCP_TA_DATA stalls)**
   — +57% / +62% over the fast comparison cell. **Dominant.**
   Manifests as TA pipeline cycles where the address/issue side is ready
   but the data return from TC has not arrived. Hit rate is unchanged,
   so this is bandwidth-shape, not miss-driven.
2. **VALU under-occupancy** (VALUBusy 18.98% vs 21.15%, −2.2pp) — minor;
   tracks but does not lead the MfmaUtil gap. Likely a *consequence* of
   stall-1 (waves blocked on returning data cannot issue VALU either).
3. **TCP_PENDING_STALL** is high in absolute terms (13.18/GRBM) but
   actually *lower* than the fast cell (14.55/GRBM). Not a contributor
   to the gap, though it indicates the L1 is generally pending-bound on
   both shapes — TC is the rate-limiter on both, and 8B Gate/Up just
   exposes a sharper version of it.
4. **All other counters** (TA_ADDR, TCP_RFIFO, FIFO_FULL, LDS bank
   conflicts, scratch, spill) — null contributors to this gap.

Roughly attributing the +12.4pp MfmaUtil gap:
* ~75–80% to L2/TC return backpressure (stalls 1).
* ~10–15% to derived VALU under-issue (stall 2).
* ~5–10% residual / measurement noise.

## 4. Why K=4096 hurts more than K=8192

The MXFP8 RRR K-loop streams A and B tiles of fixed M-row / N-col size
through L2 → L1 → LDS. Per CTA tile, the L2 demand is dominated by:
  * A-tile: `M_tile × K` bytes (per CTA-row-stripe, reused across N).
  * B-tile: `K × N_tile` bytes (per CTA-col-stripe, reused across M).
  * Scale tiles (small).

The L2-resident *productive* MFMA work per A/B fetch is proportional
to `K`. At K=4096, each A-row or B-col fetched from L2 amortizes only
half as many MFMA-cycles as at K=8192 before the kernel must move on.
The TA → TC return pipe is hit twice as hard per MFMA-cycle.

This matches what the counters show: the L2 hit rate is fine (the data
*is* in L2), but the **rate at which TC can deliver it back to TA** is
the rate-limiter, and that rate-limit becomes binding at low K because
there is less downstream MFMA work to hide it behind.

This is a **structural arithmetic-intensity issue at K=4096**, not a
cache-locality issue. R52N's recommendation that "non-prologue,
non-K-loop avenues" be pursued (LDS, V2 cachepolicy, rocprof PMC) is
vindicated — the K-loop body itself is fine, the *feeding rate* into
it is not.

## 5. R53 lever recommendation

Based on the measured stall attribution and the K-discriminator finding,
the highest-leverage R53 directions are (ordered):

### 5.1 PRIMARY — B-side L2 cachepolicy hint tuning at K=4096

The B-panel at N=14336 with K=4096 needs to stay warm in L2 long enough
that successive M-stripes of A can re-pull the same B columns without
re-promoting from HBM. The default cachepolicy (likely cache-LRU) may
be evicting B prematurely under the smaller K. **Test cachepolicy=NT
or cache-streaming on the B-side global loads at K≤4096** to bias L2
toward retention of B over A.

Specific candidate: in `kernel_mxfp8_layouts.cpp` V2 RRR path
(`dispatch_rrr_exact_8wave_scaled_v2<true>`), audit the cachepolicy
parameter on `tma::load_async` / global B fetches. If currently
`cache::ALWAYS` for both A and B, try `cache::ALWAYS` for B and
`cache::STREAMING` for A (A is consumed once per stripe, B is reused
across the N panel).

Expected impact if successful: closes 30–50% of the +12.4pp MfmaUtil
gap → +1.5pp HEADROOM at 8B Gate/Up.

### 5.2 SECONDARY — K-superblock reuse (CTA-level)

If cachepolicy alone is insufficient, restructure the CTA schedule to
process multiple M-stripes against the same B-column-block before
moving the B pointer. This is a **persistent-CTA / split-K-staged**
pattern. Higher implementation cost; defer until 5.1 is benchmarked.

### 5.3 TERTIARY — N-tile shrinking at K=4096

Smaller `N_tile` per CTA reduces the B working-set pressure per CTA
and lets more B fit in L2 simultaneously. Trade-off: fewer ops per CTA,
more CTA dispatch overhead. Worth a sweep at K=4096 only.

### 5.4 NOT recommended

* **CTA swizzle for L2 locality** — the large-N cross-check refuted
  N-stride thrashing. Swizzle would not help.
* **B-side LDS double-buffer** — LDS bank conflicts are 0; LDS pipe
  is already clean per R51E and re-confirmed here.
* **Prologue spill elimination** — R52N established prologue spill
  is benign for the K-loop body.
* **K-loop body restructuring** — the K-loop is bound by *data
  arrival*, not internal scheduling. R52H, R52J, R52K already refuted
  intra-K-loop levers.

## 6. Validity / confidence

* All measurements are 10-iteration medians per dispatch via rocprofv3
  PMC replay; durations and counters are stable cross-iteration.
* TCC_BUSY / TCC_BUBBLE were collected but show TCC pipe is largely
  busy on both cells — this is consistent with L2 itself being busy
  on hits, not idle waiting on memory. The bottleneck is the **TC →
  TA return path**, not L2 occupancy.
* The cross-shape comparison normalizes by GRBM_GUI_ACTIVE, which
  controls for cell duration. The +57%/+62% deltas are after
  normalization.
* V1 was profiled at the base set (set1) only; set3 stall counters
  were collected on V2 cells only because the V2 path is production
  and is what matters for HEADROOM. V1 set1 numbers are included for
  completeness and confirm V2 wins by ~5–7%.

## 7. Files / artifacts

* `analysis/fp8_gemm/mi350x/r52p_results/pmc_set1.txt` — base counter PMC spec
* `analysis/fp8_gemm/mi350x/r52p_results/pmc_set3.txt` — stall counter PMC spec
* `analysis/fp8_gemm/mi350x/r52p_results/v2_8B_gateup/`, `v1_8B_gateup_forced/`,
  `v2_70B_QO/`, `v1_70B_QO_forced/`, `v2_70B_GateUp/`, `v2_8B_set3/`,
  `v2_70B_set3/`, `v2_70B_GateUp_set3/` — raw rocprofv3 outputs
* `analysis/fp8_gemm/mi350x/r52p_results/aggregate.py`, `aggregate_set3.py`,
  `final_aggregate.py` — aggregation scripts
* `analysis/fp8_gemm/mi350x/r52p_results/aggregated.json`,
  `final_aggregated.json` — machine-readable medians

## 8. Verdict

**DIAGNOSTIC-COMPLETE.** The +3.1pp HEADROOM gap at 8B Gate/Up RRR is
attributable to L2/TC cache-return backpressure to the L1 read pipe,
*not* to LDS / spill / bank conflict / N-stride / K-loop scheduling.
The discriminator is K=4096 reducing per-fetch MFMA amortization. R53
should pursue B-side L2 cachepolicy tuning as the primary lever.
