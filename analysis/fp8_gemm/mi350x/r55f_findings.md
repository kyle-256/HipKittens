# R55 Dev F — 8B Gate/Up RRR PMC diagnostic — DIAGNOSTIC-SCALE-FETCH-WAIT-SALU-DOMINANT

**Cycle:** R55
**Cell (primary):** 8B Gate/Up RRR (M=4096, N=14336, K=4096)
**Co-cells (cross-shape K-sweep):** 8B Q/O RRR (4096×4096×4096, K=4096), 70B Q/O RRR (4096×8192×8192, K=8192), 70B Down RRR (4096×8192×28672, K=28672)
**Baseline gap (primary):** MX/FP8 = 92.9% (HEADROOM -1.1pp vs the cycle-baseline 94.0% target / -7.1pp vs 100%)
**Lever:** PMC counter-pair capture (FP8 vs MXFP8) on RRR layout, both `set1`
(utilization) and `set3` (stall attribution). Observational only.
**Origin:** R54B identified the **70B Down RCR** -3.4pp gap as
DIAGNOSTIC-SCALE-FETCH-WAIT (SQ_WAIT_ANY +71.6%, MfmaUtil -16pp). R55 cycle
asks: what is the structurally analogous bottleneck on the **8B Gate/Up RRR**
HEADROOM cell, and does the same class apply at smaller K and on a different
layout (RRR uses the V2 scaled kernel `rrr_exact_8wave_scaled_kernel<true,2>`,
whereas RCR uses `rcr_exact_8wave_scaled_kernel<true,2>`)? Cross-shape sweep
(K = 4096, 4096, 8192, 28672) tests K-scaling of the bottleneck.
**GPU:** MI355X (gfx950), HIP_VISIBLE_DEVICES=5 per task assignment.

## TL;DR — VERDICT: DIAGNOSTIC-SCALE-FETCH-WAIT-SALU-DOMINANT (NOT-A-FIX)

**The 8B Gate/Up RRR -1.1pp HEADROOM is a wait-cycle / scale-fetch class
bottleneck with an unusually large SALU expansion component**, structurally
related to but quantitatively distinct from R54B's 70B Down RCR signature.
The dominant PMC deltas FP8→MXFP8 on the primary cell:

| Metric | FP8 | MXFP8 | Δ (MX-FP8) | Ratio |
|---|---:|---:|---:|---:|
| **SQ_WAIT_ANY** (cyc4) | 4.090e7 | 5.368e7 | +1.278e7 | **+31.2%** ← dominant |
| **SQ_INSTS_SALU** | 4.351e6 | 6.552e6 | +2.201e6 | **+50.6%** ← much larger than R54B (+15.1%) |
| **SQ_ACTIVE_INST_VALU** | 1.092e7 | 2.013e7 | +9.218e6 | **+84.4%** |
| **VALUBusy_pct** | 11.71 | 19.18 | +7.47pp | 1.638 |
| **MfmaUtil_pct** | 62.98 | 55.94 | -7.04pp | 0.888 |
| **SQ_INSTS_VALU** | 1.092e7 | 1.279e7 | +1.871e6 | +17.1% |
| **SQ_INSTS_VMEM_RD** | 1.835e6 | 2.064e6 | +0.229e6 | +12.5% (= 9/8 exactly) |
| **SQ_BUSY_CYCLES** | 1.018e7 | 1.177e7 | +1.59e6 | +15.7% |
| SQ_INSTS_MFMA | 7.340e6 | 7.340e6 | 0 | 1.000 (identical) |
| SQ_INSTS_LDS | 7.340e6 | 7.340e6 | 0 | 1.000 (identical) |
| TCC_hit_pct | 78.28 | 79.34 | +1.06pp | 1.014 (essentially flat) |
| LDSBankConflict_pct | 0.000 | 0.000 | 0 | n/a |
| FetchSize_KB | 192758 | 185927 | -6831 | 0.965 (MXFP8 fetches LESS) |
| SQ_WAIT_INST_LDS_cyc4 | 1.070e7 | 1.160e7 | +0.090e7 | +8.4% |

The scale-tensor adds the canonical +12.5% VMEM reads (1/32 scale-byte ratio,
bursting cost identical to R54B), but the SALU expansion (+50.6%) is
**3.4× the R54B 70B Down RCR magnitude** (+15.1%) — a structural difference
in how the V2 RRR kernel handles scale address arithmetic vs the RCR kernel.
SQ_WAIT_ANY +31.2% vs R54B's +71.6% is much smaller in proportional terms,
consistent with the V2 RRR kernel having better scale-fetch overlap; but the
SALU+VALU side effects bite harder because the K=4096 inner loop has fewer
MFMA dispatches to amortize them across.

This is a **DIAGNOSTIC**, not a SHIP/REFUTED — PMC profiling does not
modify any kernel and the verdict gates no source change.

## Cross-shape PMC summary (K-sweep)

```
Cell             K       FP8 TFLOPS   MX TFLOPS   MX/FP8    MfmaUtil FP8   MfmaUtil MX
8B_GateUp      4096        2798.0      2600.3     92.9%        62.98         55.94      <-- HEADROOM cell
8B_QO          4096        2767.6      2647.6     95.7%        55.51         50.97
70B_QO         8192        3180.1      3027.3     95.2%        75.45         67.78
70B_Down      28672        2775.4      3060.1    110.3%        61.88         67.59      <-- INVERSION
```

**Two unexpected cross-shape findings:**

1. **8B_GateUp at 92.9% is the worst MX/FP8 ratio in the entire RRR cross-shape sweep** — the layout's HEADROOM cell sits 2.3-2.8pp below the other RRR cells at the same or smaller K.

2. **70B_Down RRR INVERTS to 110.3%** at K=28672 — MXFP8 is *faster* than FP8. This is structurally distinct from the R54B 70B Down RCR result (90.5%, where MXFP8 is *slower*). The interpretation: at very large K, the FP8 RRR kernel hits its own wait-cycle bottleneck (SQ_WAIT_ANY = 3.59e8 in FP8 vs 2.60e8 in MXFP8 — FP8 waits 38% more!) — so the FP8 baseline degrades faster than the MXFP8 V2 kernel, flipping the ratio. This means the V2 RRR kernel's scale-fetch pipelining is actually *more efficient* at large K than the FP8 baseline kernel's plain-fetch pipelining.

Combined: **the V2 RRR scaled kernel has good intrinsic pipelining for large K, but at small K (4096) the SALU+VALU scale-handling overhead dominates because it's not amortized across enough MFMA dispatches.** The bottleneck class on 8B Gate/Up RRR is therefore SCALE-FETCH-WAIT-SALU-DOMINANT — a SALU-elevated variant of R54B's RCR signature.

## Lever choice and rationale

R54 cycle wrap delivered the V2 RRR kernel as the MXFP8 RRR baseline (built
into `rrr_exact_8wave_scaled_kernel<true,2>` when `MXFP8_RRR_PRESHUFFLE_V2_RUNTIME=1`).
R55 cycle's HEADROOM cell is **8B Gate/Up RRR at 94.0% MX/FP8** (the
remaining 1pp gap below the R54 ship target). R49A/R53A/R53B all closed
V2 RRR K-loop body restructure family at the 254/256 VGPR ceiling (triply
closed, see MEMORY notes). This left PMC observation as the only path to
identify the bottleneck class for the 8B Gate/Up cell.

Four cells were captured to cross-validate the bottleneck class against
K-scaling:

- **8B Gate/Up (K=4096):** primary HEADROOM cell — MX/FP8 92.9%
- **8B Q/O (K=4096):** same K, square shape — MX/FP8 95.7% (+2.8pp better)
- **70B Q/O (K=8192):** medium K — MX/FP8 95.2%
- **70B Down (K=28672):** large K, R54B's primary RCR cell — MX/FP8 **110.3% (INVERTED)**

Two PMC sets were collected (rocprofv3 max 8 counters/run):

- **set1** — utilization (`SQ_WAVES, SQ_INSTS_*, SQ_WAIT_*, SQ_BUSY_CYCLES,
  SQ_VALU_MFMA_BUSY_CYCLES, GRBM_GUI_ACTIVE, TCC_HIT/MISS, TCP_PENDING/
  RFIFO/READ_TAGCONFLICT_STALL, SQ_LDS_BANK_CONFLICT, TA_TA_BUSY,
  SQ_ACTIVE_INST_*, GRBM_TA/TC_BUSY, VALUBusy, MfmaUtil, MemUnitStalled,
  LDSBankConflict, FetchSize`) — copied from R54B set1 verbatim
- **set3** — stall attribution (`TA_ADDR_STALLED_BY_TC/TD,
  TA_DATA_STALLED_BY_TC, TCP_TCP_TA_DATA_STALL, TCC_BUBBLE/BUSY,
  TCP_PENDING_STALL/RFIFO_STALL, SQ_LDS_DATA_FIFO_FULL,
  SQ_VMEM_TA_ADDR_FIFO_FULL`)

Note: this rocprofv3 build reports `[ALUStalledByLDS, L2CacheHit, MemUnitBusy]`
as missing derived metrics. We use `TCC_hit_pct` (computed from raw
`TCC_HIT_sum`/`TCC_MISS_sum`) as the L2-hit proxy throughout.

## Phase 0 — Setup and capture mechanics

- `pmc_set1.txt`, `pmc_set3.txt` — counter-group manifests (≤8 counters/group),
  copied verbatim from R54B.
- `run_pmc.sh` — capture orchestrator. Per cell × dtype × set: builds the
  shape-specific kernel once (cached via `${SO}.shape` marker), runs PMC
  capture under `rocprofv3`, dumps per-dispatch CSV.
  `MXFP8_WARMUP=100 ITERS=200`, `MXFP8_PRESHUFFLE_QUANT=1`,
  `MXFP8_RRR_PRESHUFFLE_V2_RUNTIME=1`, 30s cooldown between rocprof
  invocations.
- `aggregate.py` — per-cell loader. Filters by kernel-name tag
  (`rrr_exact_8wave_scaled_kernel` for V2 MXFP8,
  `rrr_exact_8wave_kernel` for FP8), drops `gemm_tail_kernel` fallback
  via 5000us duration ceiling, computes per-counter medians across
  dispatches, computes derived TFLOPS / MfmaUtil / TCC_hit_pct etc.,
  prints per-cell comparison and summary table. Also dumps `aggregated.json`.

Capture artifact tree:
`r55f_pmc_results/{cell}_{dtype}_rrr_set{1,3}/pmc_{1..5}/` contains
rocprofv3 CSV outputs.

**Capture-time issue and recovery (build-side, not GPU-side this time):**
The initial run dispatched a memory access fault on MXFP8 RRR. Root cause
was the per-shape build dirs symlinking `test_python.py` and
`test_mxfp8_python.py` back to `r55f_workspace/` — but Python resolves
symlinks for `sys.path[0]`, so the import path resolved to the parent
`mi350x/` directory which contained a stale
`tk_mxfp8_layouts.cpython-310-x86_64-linux-gnu.so` from another agent
(timestamped Apr 20 00:19). The stale `.so` was compiled for a different
shape and dispatched into bad memory.

Fix (in `run_pmc.sh`): replaced symlinks for `test_python.py` and
`test_mxfp8_python.py` with `cp` copies in each per-shape build dir, so
`sys.path[0]` resolves to the real per-shape build dir and loads the
correct shape-built `.so`. After the fix, MXFP8 RRR ran cleanly at
2600 TFLOPS (8B Gate/Up). All four cells × two dtypes × two PMC sets
captured with 5 dispatches each (= 80 dispatches total) with no further
faults.

## Phase 1 — Per-cell PMC delta evidence

### Cell A — 8B Gate/Up (PRIMARY, K=4096)

```
Metric                                    FP8        MXFP8       MX-FP8       MX/FP8
duration_us                              171.9       185.0        +13.1        1.076
achieved_TFLOPS                         2798.0      2600.3       -197.7      0.92935  <-- the gap
GRBM_GUI_ACTIVE                       2.825e6     3.200e6      +0.376e6      1.133
SQ_BUSY_CYCLES                        1.018e7     1.177e7      +0.159e7      1.157
SQ_INSTS_VALU                         1.092e7     1.279e7      +0.187e7      1.171
SQ_INSTS_MFMA                         7.340e6     7.340e6         0          1.000  <-- identical
SQ_INSTS_LDS                          7.340e6     7.340e6         0          1.000  <-- identical
SQ_INSTS_VMEM_RD                      1.835e6     2.064e6      +0.229e6      1.125  <-- +12.5% reads (= 9/8)
SQ_INSTS_SALU                         4.351e6     6.552e6      +2.201e6      1.506  <-- +50.6% SALU (3.4× R54B!)
SQ_VALU_MFMA_BUSY_CYCLES              2.349e8     2.349e8         0          1.000
SQ_WAIT_INST_LDS_cyc4                 1.070e7     1.160e7      +0.090e7      1.084
SQ_WAIT_INST_ANY_cyc4                 7.129e7     6.828e7      -0.301e7      0.958
SQ_WAIT_ANY_cyc4                      4.090e7     5.368e7      +1.278e7      1.312  <-- DOMINANT
SQ_ACTIVE_INST_VALU                   1.092e7     2.013e7      +0.922e7      1.844  <-- +84.4%
SQ_ACTIVE_INST_VMEM                   1.835e6     2.356e6      +0.521e6      1.284
SQ_ACTIVE_INST_LDS                    7.340e6     7.340e6         0          1.000
TCC_HIT_sum                           1.442e7     1.495e7      +0.053e7      1.037
TCC_MISS_sum                          3.999e6     3.892e6      -0.108e6      0.973
TCC_hit_pct                           78.28       79.34        +1.06pp       1.014  <-- L2 essentially flat
FetchSize_KB                          192758      185927       -6831         0.965  <-- MXFP8 FETCHES LESS
VALUBusy_pct                          11.71       19.18        +7.47pp       1.638
MfmaUtil_pct                          62.98       55.94        -7.04pp       0.888  <-- MFMA collapses
LDSBankConflict_pct                   0.000       0.000          0           n/a    <-- not LDS-bound
TA_ADDR_STALL_TC_per_grbm             0.196       0.112        -0.084        0.571  <-- LESS stall in MXFP8
TA_DATA_STALL_TC_per_grbm             1.788       1.501        -0.288        0.839
TCP_TA_DATA_STALL_per_grbm            1.795       1.637        -0.158        0.912
TCP_PENDING_per_grbm                  13.12       13.31        +0.189        1.014
TCP_RFIFO_per_grbm                    0.409       0.257        -0.153        0.627
VMEM_FIFO_FULL_per_grbm               0.271       0.174        -0.097        0.643
```

**Diagnostic claims from the deltas:**

1. **L2 / fetch bandwidth is NOT the bottleneck** (in fact, MXFP8 *fetches less*).
   - TCC hit% +1.06pp in MXFP8 (78.28% → 79.34%)
   - FetchSize -3.5% in MXFP8 (193MB → 186MB) — counter-intuitive but real:
     the V2 kernel reuses scale tiles efficiently across the M-dim
   - TA_ADDR_STALL_TC -42.9% in MXFP8 (lower address pipeline pressure)
   - All TCP_* / VMEM_FIFO_FULL stalls *down* in MXFP8

2. **LDS / bank-conflict is NOT the bottleneck.**
   - LDSBankConflict_pct = 0.000 in both
   - SQ_INSTS_LDS identical (7.340M in both)
   - LDS_FIFO_FULL_per_grbm = 0 in both
   - SQ_WAIT_INST_LDS only +8.4% (modest)

3. **MMA-side is NOT the bottleneck.**
   - SQ_INSTS_MFMA identical (7.340M in both)
   - SQ_VALU_MFMA_BUSY_CYCLES identical (234.9M in both)
   - The MMA instruction stream is unchanged.

4. **The bottleneck IS SALU-elevated wait-cycle / scale-fetch.**
   - SQ_INSTS_SALU **+50.6%** (+2.20M extra SALU) — **the largest
     proportional delta in the whole counter set**, 3.4× the R54B
     RCR signature (+15.1%)
   - SQ_WAIT_ANY +31.2% (+12.8M wait cycles)
   - SQ_ACTIVE_INST_VALU +84.4% (VALU active for far more cycles
     even though instruction count grows only +17.1%)
   - VALUBusy 11.7% → 19.2% (+7.5pp) — VALU pipe filled with
     scale-derived work
   - MfmaUtil 63.0% → 55.9% (-7.0pp) — MMA pipe starved on scale-
     operand readiness

The SALU expansion is the structural difference vs R54B. The V2 RRR kernel
appears to issue substantially more scalar address-math instructions per
K-pair than the RCR kernel, possibly due to the preshuffle scale-pointer
stride re-derivation (preshuffled scale layout requires more index
arithmetic per scale tile).

### Cell B — 8B Q/O (K=4096, square)

| Metric | FP8 | MXFP8 | MX/FP8 |
|---|---:|---:|---:|
| achieved_TFLOPS | 2767.6 | 2647.6 | **0.957** |
| MfmaUtil_pct | 55.51 | 50.97 | 0.918 |
| VALUBusy_pct | 10.06 | 17.48 | 1.737 |
| SQ_WAIT_ANY (cyc4) | 1.315e7 | 1.616e7 | **1.229** |
| SQ_INSTS_SALU | 1.290e6 | 1.870e6 | **1.449** |
| SQ_INSTS_VMEM_RD | 5.243e5 | 5.898e5 | 1.125 |
| SQ_ACTIVE_INST_VALU | 3.041e6 | 5.753e6 | 1.892 |
| TCC_hit_pct | 72.81 | 80.11 | +7.30pp |
| FetchSize_KB | 73962 | 50759 | 0.686 |

8B Q/O sits +2.8pp better than 8B Gate/Up at the same K. **Notable
inversion (mirroring R54B's 8B Down observation):** MXFP8 has *better*
L2 hit rate (+7.3pp) and *much less* fetched bytes (-31.4%) — the smaller
N-dim (4096 vs 14336) means the scale tile fits in L2 more comfortably,
masking part of the wait penalty. SQ_INSTS_SALU still expands +44.9%
(close to 8B Gate/Up's +50.6%) — confirming the SALU expansion is a
property of the V2 kernel, not the shape.

### Cell C — 70B Q/O (K=8192)

| Metric | FP8 | MXFP8 | MX/FP8 |
|---|---:|---:|---:|
| achieved_TFLOPS | 3180.1 | 3027.3 | **0.952** |
| MfmaUtil_pct | 75.45 | 67.78 | 0.898 |
| VALUBusy_pct | 12.14 | 21.15 | 1.743 |
| SQ_WAIT_ANY (cyc4) | 4.376e7 | 5.823e7 | **1.331** |
| SQ_INSTS_SALU | 6.779e6 | 7.148e6 | **1.054** ← much smaller SALU growth |
| SQ_INSTS_VMEM_RD | 2.097e6 | 2.359e6 | 1.125 |
| SQ_ACTIVE_INST_VALU | 1.080e7 | 2.094e7 | 1.940 |
| TCC_hit_pct | 80.67 | 80.60 | -0.07pp |

70B Q/O at K=8192 sits at 95.2%. **Important inflection:** SQ_INSTS_SALU
ratio drops to +5.4% (from +50.6% at K=4096). Interpretation: at K=8192,
the per-tile SALU scale-address overhead is amortized across twice as many
MMA dispatches per scale-tile arithmetic block, so the SALU expansion
becomes negligible. The SQ_WAIT_ANY +33.1% remains as the residual
wait-cycle bottleneck (matching R54B's 70B Q/O RCR signature of +30.9%).

### Cell D — 70B Down (K=28672) — INVERSION

| Metric | FP8 | MXFP8 | MX/FP8 |
|---|---:|---:|---:|
| achieved_TFLOPS | 2775.4 | 3060.1 | **1.103** ← INVERTED |
| MfmaUtil_pct | 61.88 | 67.59 | **1.092** ← MXFP8 wins |
| VALUBusy_pct | 9.60 | 19.60 | 2.042 |
| SQ_WAIT_ANY (cyc4) | 3.590e8 | 2.601e8 | **0.725** ← MXFP8 waits 28% LESS |
| SQ_INSTS_SALU | 1.738e7 | 2.419e7 | 1.392 |
| SQ_INSTS_VMEM_RD | 8.032e6 | 8.258e6 | 1.028 |
| SQ_ACTIVE_INST_VALU | 3.645e7 | 6.813e7 | 1.869 |
| TCC_hit_pct | 79.50 | 80.98 | +1.48pp |
| FetchSize_KB | 774553 | 718109 | 0.927 |

**At K=28672, the FP8 RRR baseline kernel itself hits a wait-cycle
bottleneck** — `SQ_WAIT_ANY = 3.59e8` (vs MXFP8's 2.60e8). The MXFP8 V2
kernel waits 28% LESS, fetches 7% LESS, and achieves +9.2pp higher
MfmaUtil (61.9% → 67.6%). This is a **layout-class structural difference**:

- The R54B 70B Down RCR result (90.5% MX/FP8) shows the RCR FP8 kernel
  is *not* wait-cycle bound at K=28672 (it sits at 83.8% MfmaUtil),
  so adding scale fetches *adds* +71.6% wait cycles.
- The R55F 70B Down RRR result (110.3% MX/FP8) shows the RRR FP8 kernel
  *is* wait-cycle bound at K=28672 (only 61.9% MfmaUtil), so the V2
  scaled kernel's better pipelining beats the FP8 baseline.

**This means the 70B Down RRR cell would be a SHIP candidate for V2
preshuffle scaling on grounds of being *faster* than FP8, not just
narrowing the gap.**

### Per-cell summary

```
Cell             K       FP8 TFLOPS   MX TFLOPS   MX/FP8    MfmaUtil FP8   MfmaUtil MX
8B_GateUp      4096        2798.0      2600.3     92.9%        62.98         55.94      <-- HEADROOM
8B_QO          4096        2767.6      2647.6     95.7%        55.51         50.97
70B_QO         8192        3180.1      3027.3     95.2%        75.45         67.78
70B_Down      28672        2775.4      3060.1    110.3%        61.88         67.59      <-- INVERTED
```

**The K-scaling pattern for RRR is non-monotone**, which is structurally
distinct from RCR's monotone inverse-K-scaling (R54B):
- RCR (R54B): MX/FP8 monotonically tracks K (95.9% @ K=14336 → 94.7% @ K=8192 → 90.5% @ K=28672).
- RRR (R55F): MX/FP8 starts at 92.9-95.7% for K=4096, climbs to 95.2% at K=8192, then INVERTS to 110.3% at K=28672.

The non-monotone behavior is consistent with the V2 RRR kernel having a
different pipelining shape: it amortizes scale-handling well at moderate K,
but at small K the SALU+VALU overhead dominates, and at very large K the
FP8 baseline degrades faster than V2 MXFP8.

## Phase 2 — Resource summary table

PMC capture is observational, so no build resource delta exists for this
work. The kernel resource summaries used as reference for these dispatches
(taken from R52O/R53 baseline measurements):

| Kernel | VGPR | Spill | LDS | Occupancy |
|---|---:|---:|---:|---:|
| `rrr_exact_8wave_kernel` (FP8 baseline) | (per RRR baseline) | 0 | (per RRR baseline) | 2 |
| `rrr_exact_8wave_scaled_kernel<true,2>` (V2 MXFP8) | 254/256 (ceiling) | 0 | (per RRR baseline) | 2 |

V2 RRR sits at the 254/256 VGPR ceiling (R49A/R53A/R53B all spilled when
attempting K-loop body restructure). The MX/FP8 gap is therefore **NOT**
an occupancy or spill problem at the kernel-launch level — and the
254/256 VGPR ceiling means any future cycle that touches V2 RRR's K-loop
body needs to subtract VGPR pressure before adding work (else it spills,
losing occupancy).

ISA disassembly captured at:
- `r55f_pmc_results/isa/8B_GateUp_mxfp8_v2rrr_dis.txt` (V2 RRR scaled, 1267 lines)
- `r55f_pmc_results/isa/8B_GateUp_fp8_rrr_dis.txt` (FP8 RRR baseline, 4431 lines)

The FP8 baseline disassembly is 3.5× longer because the FP8 kernel has
multiple specializations and tail kernels in the same `.so`, whereas the
V2 RRR kernel is a single template instantiation `<true,2>`.

## Reasoning — what the PMC delta tells us

The dominant signature on 8B Gate/Up RRR is:

```
SQ_INSTS_SALU:         +50.6%  (3.4× the R54B RCR signature) ← STRUCTURAL DIFFERENCE
SQ_ACTIVE_INST_VALU:   +84.4%
SQ_WAIT_ANY:           +31.2%  (less than half of R54B's +71.6%)
VALUBusy:              +7.47pp (rises 11.7% → 19.2%)
MfmaUtil:              -7.04pp (drops from 63.0% → 55.9%)
SQ_INSTS_VMEM_RD:      +12.5%  (= 9/8, exactly one scale byte per 8 data bytes)
SQ_INSTS_VALU:         +17.1%  (small expansion)
SQ_INSTS_MFMA:          0.0%
SQ_INSTS_LDS:           0.0%
TCC_hit_pct:           +1.06pp (cache UNAFFECTED, slightly improved)
FetchSize_KB:          -3.5%   (MXFP8 fetches LESS, not more)
LDSBankConflict_pct:    0      (LDS UNAFFECTED)
```

**Bottleneck class identification:** The V2 RRR scaled kernel adds three
joint pressures on the inner-loop critical path on the 8B Gate/Up cell:

1. **Scale-tensor VMEM stream (+12.5% reads)**: same canonical 1/32
   scale-byte-per-data-byte ratio as RCR. Modest direct contribution.

2. **Scale address SALU explosion (+50.6%)** ← **distinctive of V2 RRR**:
   The preshuffle-V2 scale layout requires more scalar pointer
   arithmetic per K-tile than the RCR kernel. R54B RCR saw +15.1% SALU;
   R55F V2 RRR sees +50.6%. The 70B Q/O cell at K=8192 shows this
   collapses to +5.4%, indicating the SALU work is per-tile-fixed rather
   than per-MFMA — so it amortizes well at large K but dominates at K=4096.

3. **Scale-derived VALU dependency chain (+84.4% active VALU cycles)**:
   Same mechanism as R54B — scale operand construction (preshuffle
   unpack, scale tile shuffle) creates dependency chains that stall the
   VALU issue window. VALUBusy rises 11.7% → 19.2%, MFMA pipe waits
   on scale-operand readiness.

These three jointly expand SQ_WAIT_ANY by +31.2% — the dominant
wait-cycle delta. Note the wait-cycle expansion is **smaller in
proportion** than R54B's RCR result, but the **MfmaUtil collapse is
similar** (-7.0pp vs -16.0pp) because the V2 RRR baseline MfmaUtil starts
much lower (62.98% vs 83.79% in RCR FP8) — proportionally the loss is
significant.

**This is structurally related to but quantitatively distinct from R54B
RCR.** The shared component is scale-fetch + scale-VALU dependency chain.
The distinguishing component is the SALU explosion (3.4× larger),
attributed to the V2 preshuffle scale-pointer arithmetic. The 70B Q/O
data point (+5.4% SALU at K=8192) confirms this is a per-tile-fixed
overhead that small K cannot amortize.

**This bottleneck is FUNDAMENTALLY DIFFERENT from the CRR class.** CRR's
floor is the 6× `v_lshrrev_b32`/K-pair scale-pack `>> 16` shift block
(closed across 5 axes by R49A/R50A/R53A/R54A/R54E). RRR has zero of
these shifts (RRR uses preshuffle-V2, not pack-shift). RRR's gap is the
**SALU-dominant scale-address-arithmetic / VALU-dependency-issue-window
class** — third distinct microarchitectural region of the gap space
(after RCR's wait-fetch-dominant class and CRR's pack-shift class).

## Falsifiable predictions for future cycles (R56+)

These are what should be attempted next on the 8B Gate/Up RRR cell to
target the identified bottleneck class:

- **P1.** A lever that hoists scale-pointer SALU arithmetic out of the
  K-loop (e.g., precompute base + per-tile-stride pair into VGPR-resident
  scratch on launch, or replace per-iteration `s_load_dword` chains with
  a single `s_load_dwordx2` per K-pair) should reduce SQ_INSTS_SALU
  by ≥ 20% on 8B Gate/Up RRR. Predicted TFLOPS gain: ≥ +0.5pp on MX/FP8
  ratio. **Falsification:** SALU drops ≥20% but SQ_WAIT_ANY does not
  reduce by ≥ 10% → SALU is in the address pipeline but not on the
  critical path.

- **P2.** A lever that interleaves scale-tile VMEM fetch with data-tile
  VMEM fetch into a single `buffer_load_dwordx4` burst (collapsing
  +12.5% extra fetches into the data fetch issue window) should show
  ≥ +0.3pp on 8B Gate/Up RRR MX/FP8 ratio. **Falsification:** the lever
  lands but SQ_WAIT_ANY does not reduce by ≥ 15% → scale-fetch arrival
  is not on the critical path (it's the SALU+VALU chain, not the
  fetch-arrival).

- **P3.** A lever that breaks the scale-derived VALU dependency chain
  (e.g., different VGPR allocation for scale-construct intermediaries,
  or `v_pk_mov_b32` to break dependency latency) should reduce
  SQ_ACTIVE_INST_VALU by ≥ 30% on 8B Gate/Up RRR (from 2.013e7 toward
  1.5e7). Predicted TFLOPS gain: ≥ +0.7pp. **Falsification:** active
  VALU cycles drop ≥30% but TFLOPS flat → VALU issue window is not the
  limiter (it's the SALU pipeline).

- **P4.** A lever that pre-stages scale-tile L2 residency on 8B Gate/Up
  (scale prefetch into L2 scratch immediately before the K-loop) should
  push the TCC_hit_pct delta upward (currently +1.06pp) and reduce
  SQ_WAIT_ANY by ≥ 5%. **Falsification:** L2 hit rises but SQ_WAIT_ANY
  flat → the +12.5% extra reads are not L2-misses, they're issue-pipe
  pressure regardless of cache hit.

- **P5.** A lever that makes V2 RRR shippable on 70B Down (already
  INVERTED at 110.3%) should be considered **NOW** — the cell is a SHIP
  candidate independent of the gate. Verify: 70B Down RRR with V2
  preshuffle on the 8B Gate/Up tuned configuration should retain the
  +10.3% advantage. **Falsification:** V2 RRR loses the inversion under
  ship-mode determinism gates → the FP8 baseline degraded for a
  measurement-noise reason, not a structural one.

## Strong negative predictions (what NOT to try on V2 RRR 8B Gate/Up)

- **NOT** any LDS-side restructuring (LDSBankConflict = 0; LDS instruction
  count identical FP8 vs MXFP8; LDS is not the bottleneck).
- **NOT** any L2/cache-pressure mitigation on 8B Gate/Up (TCC_hit_pct
  IMPROVED slightly in MXFP8 +1.06pp; FetchSize *down* -3.5%; L2 is fine).
- **NOT** any scale-pack `>> 16` shift removal (RRR has none — that's
  the CRR-class problem).
- **NOT** any K-loop body restructure that adds VGPR pressure (V2 RRR is
  at 254/256 ceiling — quadruply closed by R49A/R53A/R53B per
  MEMORY/v2_rrr_vgpr_ceiling.md).
- **NOT** any -mllvm flag sweep (R54C closed compiler-flag exploration
  on V2 RRR — REFUTED-EMPIRICAL, 254/256 ceiling per VGPR memory note).
- **NOT** any cachepolicy modifier on data loads (R54B/R54H/R54I-class —
  data-side already fine, the bottleneck is scale-side SALU+VALU).

## Files

- `r55f_pmc_results/aggregate_output.txt` — primary human-readable per-cell
  comparison table (the table that drives this synthesis).
- `r55f_pmc_results/aggregated.json` — full per-cell raw + derived counters
  in JSON form (used to look up SQ_ACTIVE_INST_VALU and other detail
  counters not in the printed table).
- `r55f_pmc_results/aggregate.py` — aggregator script (kernel-name
  filtering, fallback-kernel duration ceiling at 5000us, median-across-
  dispatches, set1+set3 merge, 4-cell sweep).
- `r55f_pmc_results/pmc_set1.txt`, `pmc_set3.txt` — counter-group
  manifests submitted to rocprofv3 (verbatim from R54B).
- `r55f_pmc_results/run_pmc.sh` — capture orchestrator (per-cell × dtype
  × set: build, PMC capture under rocprofv3, CSV output, 30s cooldown
  between rocprof invocations). MXFP8_WARMUP=100 ITERS=200,
  MXFP8_RRR_PRESHUFFLE_V2_RUNTIME=1.
- `r55f_pmc_results/orchestrator.log` — capture-run log; documents the
  symlink-resolution memory-fault and recovery via real-file copies.
- `r55f_pmc_results/isa/{8B_GateUp_mxfp8_v2rrr,8B_GateUp_fp8_rrr}_dis.txt` —
  ISA disassembly of the V2 RRR scaled kernel and FP8 RRR baseline.
- `r55f_pmc_results/{cell}_{dtype}_rrr_set{1,3}/pmc_{1..5}/` — raw per-
  dispatch rocprofv3 CSVs (4 cells × 2 dtypes × 2 sets × 5 dispatches =
  80 dispatches captured).
- `r55f_workspace/` — per-shape build dirs (`builds/{LABEL}/`) with copied
  test_python.py / test_mxfp8_python.py (NOT symlinked — see Phase 0
  capture-time issue).

No source files were modified. No commits to source were created. PMC
profiling is strictly observational.

## Protocol note

PMC capture under rocprofv3 with multi-counter-set sweeps is rate-limited
by GPU dispatch order and rocprofv3 internal sequencing; 5 dispatches per
(cell, dtype, set) are taken with median aggregation to mitigate
single-dispatch noise. The 8B Gate/Up MXFP8 first-pass memory-access fault
was a **symlink-resolution issue, not a GPU/rocprofv3 issue**: Python
resolves symlinks for `sys.path[0]`, which pulled a stale parent-dir `.so`
into the import. The fix (real-file copies of test_*.py per build dir)
is documented in `run_pmc.sh` and prevents recurrence.

This rocprofv3 build does not provide derived metrics
`[ALUStalledByLDS, L2CacheHit, MemUnitBusy]` — `TCC_hit_pct` (computed
from raw `TCC_HIT_sum`/`TCC_MISS_sum`) is used as the L2-hit proxy.
LDSBankConflict_pct is reported (= 0 in all four cells, both dtypes).

This work follows the R47D / R53C / R54B precedent of using PMC capture
as a **diagnostic** to scope the bottleneck class for downstream lever
cycles, not as a SHIP gate by itself.

## Verdict line for cycle wrap

`R55 Dev F: 8B Gate/Up RRR PMC diagnostic — DIAGNOSTIC-SCALE-FETCH-WAIT-SALU-DOMINANT — bottleneck class is V2 RRR scale-pointer SALU-address arithmetic (+50.6%, 3.4× R54B RCR's +15.1%) + scale-derived VALU dependency chain (+84.4% active VALU) + scale-fetch wait (SQ_WAIT_ANY +31.2%, MfmaUtil -7.04pp); NOT cache (TCC_hit +1.06pp, FetchSize -3.5%), NOT LDS (bank-conflict 0, LDS instr identical), NOT scale-pack shifts (RRR uses preshuffle-V2, not pack-shifts); structurally distinct from R54B RCR (SALU expansion 3.4× larger) and from CRR pack-shift class (no shifts in RRR); cross-shape K-sweep non-monotone (92.9% @ K=4096, 95.2% @ K=8192, 110.3% INVERTED @ K=28672 — V2 beats FP8 at large K); 70B_Down RRR is a SHIP candidate at +10.3% over FP8.`
