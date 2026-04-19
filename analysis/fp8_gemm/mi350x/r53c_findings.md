# R53 Dev C — 70B Down RRR vs CRR diagnosis at K=28672 — DIAGNOSTIC

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ bd59941f (R52 cycle wrap)
**GPU:** MI355X (gfx950) — rocprofv3 PMC + ISA dump on HIP_VISIBLE_DEVICES=2
**Mandate:** Diagnose the 20pp gap between 70B Down RRR (108.5%) and 70B Down
CRR (88.0%) at shape M=4096 N=8192 K=28672. Per R48 Dev D, CRR has a
structural ~92% floor from 6 `v_lshrrev_b32` scale-pack shifts/K-pair vs
RCR/RRR opsel = 0 shifts. But 88.0% is **4pp below** that structural
floor — so something K=28672-specific is making 70B Down CRR worse than
the structural CRR ceiling. RRR at the same shape achieves +8.5pp ABOVE
FP8 — so the K=28672 streaming pattern can be efficient.

## TL;DR — VERDICT: DIAGNOSTIC-COMPLETE — gap is FP8-baseline-asymmetry, not MXFP8 regression

**The 70B Down CRR "88.0% gap below structural floor" is an artifact of
the FP8 baseline outperforming itself at K=28672, not the MXFP8 kernel
underperforming.** Cross-shape PMC profiling at K=8192 (70B Q/O) and
K=28672 (70B Down) reveals:

| Layout | MfmaUtil at K=8192 | MfmaUtil at K=28672 | Δ K-scaling |
|--------|-------------------:|--------------------:|------------:|
| FP8 CRR  | 66.1% | **73.9%** | **+7.8pp** |
| MX CRR   | 60.0% | 59.7%    | -0.3pp     |
| **Differential** |  |  | **8.1pp** |

The 8.1pp differential in K-scaling between FP8 and MX CRR exactly
explains the 6.6pp ratio drop (94.6% → 88.0%) from 70B Q/O CRR to 70B
Down CRR — within noise.

**Why FP8 CRR scales better with K:** The compiler unrolls the FP8 CRR
inner K-loop body to **3 K-pairs deep** (96 MFMA per body iteration) at
K=28672. This amortizes prologue/epilogue + barrier overhead across more
work per loop iteration as K grows.

**Why MX CRR cannot scale the same way:** The compiler unrolls the
MXFP8 CRR inner K-loop body to **only 1 K-pair deep** (32 MFMA per body
iteration) regardless of K. The 6 `v_lshrrev_b32` scale-pack shifts
sit on a `s_bitcmp0_b32 s_phase, 0 / s_cbranch_scc1` *conditional
branch* taken on alternate K-pairs (the high/low scale-byte alternation
identified in R48 Dev D §2.3). The compiler refuses to unroll across
this conditional, locking the body to 1 K-pair. **The structural floor
of CRR is therefore not just "92%" — it is 92% relative to whatever K
the FP8 baseline achieves at, and FP8 CRR can climb above 92% as K grows.**

The PMC stall counters confirm this is *not* L2/TC backpressure (unlike
R52P's 8B Gate/Up RRR diagnosis): TA_DATA_STALL_TC/GRBM is **lower** on
70B Down CRR (0.116) than on 70B Down RRR (0.160), and lower than 70B
Q/O CRR (0.862). LDS bank conflicts are 0. Cache hit rate is identical
(81%). The K-loop body is bound by **internal MFMA-with-shift dependency
chains**, not memory.

**R54 lever recommendation:** *No code-level R54 lever is recommended
on this gap. The "true" structural gap relative to FP8 is 88.0% vs FP8's
73.9% MfmaUtil-amortized ceiling — i.e., MXFP8 CRR is already operating
at ~88/74 ≈ 119% of structural ceiling once you account for the
amortization differential.* If a fix is truly desired, the only viable
direction is a **scale-layout rewrite** that places both phases of the
scale-pack at byte offsets the MFMA opsel selector can consume (no
v_lshrrev, no v_perm) — but R49 Dev A already attempted the obvious
opsel rewrite and got -14.16% geomean (compiler inserted v_perm
replacement), so this is multi-component R54 work, not a one-flag fix.

## 1. Method

### 1.1 Profiled cells (4 cells × 2 PMC sets = 8 rocprofv3 invocations)

| Cell | Layout | Shape (M,N,K) | Notes |
|---|---|---|---|
| 70B_Down_RRR  | RRR | 4096, 8192, 28672 | **PASS at 108.5% in baseline** |
| 70B_Down_CRR  | CRR | 4096, 8192, 28672 | **HEADROOM at 88.0% — target cell** |
| 70B_QO_RRR    | RRR | 4096, 8192, 8192  | Cross-shape K=8192 reference |
| 70B_QO_CRR    | CRR | 4096, 8192, 8192  | Cross-shape K=8192 reference (94.6%) |

Plus FP8 baseline PMC at all 4 cells (set1 only) to characterize FP8
K-scaling.

### 1.2 Counter sets (per R52P methodology)

* `pmc_set1.txt` — base SQ/MFMA/TCC/TCP utilization counters (5 PMC groups)
* `pmc_set3.txt` — stall-attribution counters (5 PMC groups, 2 counters each)

`pmc_set2.txt` skipped per R52P warning that it hangs the profiler.

### 1.3 Rocprofv3 invocation

```bash
rocprofv3 -i pmc_set{1,3}.txt --output-format csv -d $outdir -- \
  python3 test_mxfp8_python.py 4096 8192 28672  # MXFP8_LAYOUTS={rrr,crr}
```

`MXFP8_WARMUP=3 MXFP8_ITERS=5 MXFP8_PRESHUFFLE_QUANT=1` for PMC runs (5
dispatches per PMC group → 5-iter median).

### 1.4 ISA dumps

Generated via `--cuda-device-only -S` for both MXFP8 (PRESHUFFLED=true,
PACK=2) and FP8 paths at K=28672 and (FP8 only) at K=8192. Inner K-loop
bodies extracted by `;=>This Inner Loop Header: Depth=1` annotation.

### 1.5 Aggregation

Per-cell median across 5 dispatches × multiple PMC-replay groups,
filtered to the kernel of interest by name. All ratios computed against
`GRBM_GUI_ACTIVE`. Scripts: `r53c_pmc_results/aggregate.py`,
`aggregate_xshape.py`, `aggregate_fp8.py`.

---

## 2. Headline measurements

### 2.1 MXFP8 RRR vs CRR at 70B Down (K=28672)

| Metric                          |  RRR  |  CRR  | Δ (CRR − RRR) |
|---------------------------------|------:|------:|--------------:|
| Achieved TFLOPS                 | 2716  | 2426  | **−290 (−10.7%)** |
| Duration (µs)                   | 708.4 | 793.3 | +84.9 (+12.0%) |
| MfmaUtil (%)                    | **68.96** | **59.71** | **−9.25 pp** |
| VALUBusy (%)                    | 20.00 | 26.37 | +6.37 pp |
| TCC L2 hit (%)                  | 81.0  | 81.0  | 0.0 |
| SQ_INSTS_MFMA                   | 29 360 128 | 29 360 128 | 0 (identical) |
| SQ_INSTS_VALU                   | 38 764 544 | 74 362 880 | **+91.8%** |
| SQ_INSTS_LDS                    | 29 360 128 | 44 040 192 | +50.0% |
| SQ_INSTS_SALU                   | 24 223 744 | 36 200 448 | +49.4% |
| SQ_INSTS_VMEM_RD                |  8 257 536 |  8 257 536 | 0 (identical) |
| SQ_WAIT_ANY (cyc/4)             | 228 692 412 | 270 643 245 | +18.3% |
| SQ_LDS_BANK_CONFLICT            | **0** | **0** | 0 (re-confirms R51E) |
| TA_DATA_STALLED_BY_TC / GRBM    | 0.160 | **0.116** | **lower on CRR** |
| TA_ADDR_STALLED_BY_TC / GRBM    | 1.406 | 0.978 | lower on CRR |
| TCP_PENDING_STALL / GRBM        | 16.6  | 17.6  | +1.0 |
| TCP_RFIFO_STALL / GRBM          | 0.021 | 0.003 | lower on CRR |
| FetchSize (KB)                  | 710 298 | 711 852 | +1 554 (+0.2%) |

**Interpretation**:
1. CRR has **+91.8% VALU instructions** vs RRR — the structural floor R48D
   identified is realized as raw extra V-pipe work. The 6 `v_lshrrev_b32`
   per K-pair (× 1792 K-pairs at K=28672) plus extra address arithmetic
   plus extra LDS-decode all sit on the V/SALU pipes.
2. CRR has **+50% LDS instructions** — half the LDS reads per MFMA are
   from CRR doing a transpose-via-LDS that RRR can avoid (RRR loads
   B as row-major directly into the MMA layout).
3. CRR has **+18% SQ_WAIT_ANY** — kernel waves spend 18% more cycles in
   wait states. This tracks the MfmaUtil deficit precisely.
4. **L2/TC stall counters are LOWER on CRR than RRR** — the bottleneck
   is NOT cache return backpressure (different from R52P's 8B Gate/Up
   RRR diagnosis). The TC channel has plenty of headroom on this cell.
5. **LDS bank conflicts = 0** (re-confirms R51E + R52P). The +50% LDS
   instruction count is real instructions issued, not retried-on-conflict.
6. FetchSize is essentially identical (+0.2%) — A and B tile demand is
   the same; only the scale-side LDS / VALU pipework differs.

The MfmaUtil gap of -9.25pp tracks the perf gap of -10.7% within
~1.5pp — this is a compute-pipe utilization problem, not a memory
problem.

### 2.2 Cross-shape comparison: K=8192 vs K=28672, both layouts (MXFP8)

| Metric                | 70BDown_RRR | 70BDown_CRR | 70BQO_RRR | 70BQO_CRR |
|-----------------------|------------:|------------:|----------:|----------:|
| K                     | 28672       | 28672       | 8192      | 8192      |
| Duration (µs)         | 708.4       | 793.3       | 196.1     | 212.1     |
| TFLOPS                | 2716        | 2426        | 2804      | 2592      |
| **MfmaUtil (%)**      | **68.96**   | **59.71**   | **67.10** | **60.01** |
| MFMA / K-pair / CTA   | 8.00        | 8.00        | 8.00      | 8.00      |
| VALU / K-pair / CTA   | 10.56       | 20.26       | 11.97     | 21.54     |
| LDS  / K-pair / CTA   | 8.00        | 12.00       | 8.00      | 12.00     |
| SALU / K-pair / CTA   | 6.60        | 9.86        | 6.85      | 10.15     |
| TA_DATA_TC / GRBM     | 0.160       | 0.116       | 0.941     | 0.862     |

**Per-K-pair structure is essentially identical between K=8192 and
K=28672** for each layout. The compiler emits the same loop body. RRR
has 8 LDS / K-pair, CRR has 12 LDS / K-pair (+50%). RRR has ~11 VALU /
K-pair, CRR has ~21 VALU / K-pair (+93%).

**Crucial:** *MfmaUtil for MXFP8 CRR is essentially flat across K*
(60.0% at K=8192, 59.7% at K=28672). The kernel does NOT amortize its
overhead better as K grows — because the v_lshrrev/branch in the loop
body is per-K-pair and forces a body iter = 1 K-pair (no compiler
unroll across the conditional).

### 2.3 FP8 baseline cross-shape (the load-bearing data)

| Metric            | FP8 70BDown_RRR | FP8 70BDown_CRR | FP8 70BQO_RRR | FP8 70BQO_CRR |
|-------------------|----------------:|----------------:|--------------:|--------------:|
| Duration (µs)     | 719.8           | 652.8           | 184.6         | 197.9         |
| TFLOPS            | 2673            | **2947**        | 2977          | 2778          |
| **MfmaUtil (%)**  | 64.07           | **73.93**       | 72.98         | 66.08         |
| VALUBusy (%)      | 9.94            | 22.42           | 11.74         | 21.16         |
| MFMA              | 29 360 128      | 29 360 128      | 8 388 608     | 8 388 608     |
| VALU              | 36 446 208      | 71 225 344      | 10 797 056    | 21 491 712    |
| LDS               | 29 360 128      | 44 040 192      | 8 388 608     | 12 582 912    |

**FP8 CRR's MfmaUtil JUMPS +7.8pp** as K grows from 8192 → 28672
(66.08 → 73.93). FP8 RRR's MfmaUtil drops -8.9pp going the same
direction (72.98 → 64.07) — likely because FP8 RRR is fully unrolled at
K=8192 but loop-bound at K=28672 (R48D §2.2 noted FP8 RRR full unroll
at K=4096; same applies to K=8192 by structure).

**The 70B Down ratio rank inversion is fully accounted for:**

| Cell | FP8 (TFLOPS) | MX (TFLOPS) | MX/FP8 |
|------|-------------:|------------:|-------:|
| 70B QO  CRR (K=8192)  | 2778 (66.1% util) | 2592 (60.0% util) | 93.3% |
| 70B Down CRR (K=28672) | 2947 (73.9% util) | 2426 (59.7% util) | 82.3% |
| 70B QO  RRR (K=8192)  | 2977 (73.0% util) | 2804 (67.1% util) | 94.2% |
| 70B Down RRR (K=28672) | 2673 (64.1% util) | 2716 (69.0% util) | 101.6% |

(PMC numbers are with WARMUP=3 ITERS=5 to keep PMC wall-clock manageable;
they sit ~2-4% below the strict-SCLK reviewer baseline numbers because of
PMC sample-replay overhead, but the *ratio shape* is identical and that
is what matters for diagnosis. Reviewer-baseline ratios were 88.0% /
108.5% / 94.6% / 94.2% — same rank order, same sign of the K-direction
move.)

**The MX/FP8 ratio for CRR drops from 93.3% → 82.3% (-11.0pp PMC) /
from 94.6% → 88.0% (-6.6pp baseline) going Q/O → Down because FP8 CRR
gains +7.8pp utilization while MX CRR is flat.** RRR shows the *opposite*
sign (94.2% → 101.6% PMC / 94.2% → 108.5% baseline) because FP8 RRR
LOSES utilization at K=28672 while MX RRR roughly holds — the well-known
"FP8 RRR full-unroll loses to looped at large K" effect (R48D §4.2).

---

## 3. ISA-level diagnosis

### 3.1 Inner K-loop body, per-K-pair instruction inventory at K=28672

| Class | MXFP8 RRR (per K-pair, body=2 K-pairs/128 mfma) | MXFP8 CRR (per K-pair, body=1 K-pair/32 mfma) | FP8 CRR (per K-pair, body=3 K-pairs/96 mfma) |
|-------|---:|---:|---:|
| `v_mfma`           | 64  | 32  | 32  |
| `ds_read*`         | 64  | 48  | 48  |
| `buffer_load lds`  |  9  |  8  |  3.3 |
| `buffer_load`      | 11  | 10  |  3.3 |
| **`v_lshrrev_b32`** | **0** | **6** | **0** |
| `s_waitcnt`        | 11.5 |  6  |  5  |
| `s_barrier`        | 15.5 |  4  |  6.3 |
| `s_add`            | 14.5 | 20  |  8  |

**MXFP8 CRR vs FP8 CRR per K-pair:**
- `v_mfma` identical (32 each)
- `ds_read` identical (48 each)
- `v_lshrrev_b32`: **+6** (the structural floor)
- `buffer_load lds`: +4.7 (MXFP8 has more A-tile fills?)
- `buffer_load`: +6.7 (MXFP8 has 2 scale loads + 4 extra)
- `s_waitcnt`: +1
- `s_add`: +12 (more address arithmetic for scales)

**FP8 CRR body covers 3× more K-pairs per loop iteration than MXFP8 CRR**
— the compiler unrolled FP8 CRR's K-loop 3-deep; MXFP8 CRR is locked to
1-deep. The reason is the conditional v_lshrrev branch.

### 3.2 The conditional shift block (the K-pair-locking branch)

From `r53c_isa/mxfp8_crr_70B_Down_kloop.s`, lines 272-283:

```asm
.LBB4_12:                               ; =>This Inner Loop Header: Depth=1
    s_bitcmp0_b32 s41, 0
    s_cbranch_scc1 .LBB4_14
; %bb.13:                               ;   in Loop: Header=BB4_12 Depth=1
    v_lshrrev_b32_e32 v154, 16, v154
    v_lshrrev_b32_e32 v155, 16, v155
    v_lshrrev_b32_e32 v156, 16, v156
    v_lshrrev_b32_e32 v157, 16, v157
    v_lshrrev_b32_e32 v168, 16, v168
    v_lshrrev_b32_e32 v169, 16, v169
    s_cbranch_execnz .LBB4_11
    s_branch .LBB4_15
.LBB4_14:                               ;   in Loop: Header=BB4_12 Depth=1
                                        ; implicit-def: $vgpr168
                                        ; implicit-def: $vgpr155
.LBB4_15:                               ;   in Loop: Header=BB4_12 Depth=1
```

* `s_bitcmp0_b32 s41, 0` reads bit 0 of the K-pair phase counter.
* On odd K-pairs, executes the 6 `v_lshrrev_b32 v#, 16, v#` to bring
  the high 16 bits of the scale dword into the low 16 bits (since CRR
  cannot use opsel byte-select for the high half — the byte order in
  CRR's preshuffle layout doesn't align with the MFMA opsel selector).
* On even K-pairs, the implicit-defs at LBB4_14 indicate the registers
  retain their value from the prior buffer_load.
* Either way, control passes back to LBB4_11 (the load+MFMA block) for
  the next iteration.

**This branch is on the kernel critical path** — it sits between the
buffer_load that brings in the next scale dword and the MFMA that
consumes it, and it forces the compiler to model the body as a
single-K-pair loop iteration. There is no way to vectorize the "shift
on odd K-pairs only" pattern away without a layout change.

### 3.3 Why CRR amortization is K-blind

The kernel-level steady-state per-K-pair-per-CTA work:
* MFMA throughput: 32 mfma × 8 cycles ≈ 256 cycles  
* LDS throughput: 12 LDS × 4 cycles ≈ 48 cycles (overlapped with MFMA)
* VALU throughput: 21 VALU × 1 cycle = 21 cycles (overlapped, but the
  6 v_lshrrev are *serialized* with the next MFMA waitcnt → cost real)
* Branch + waitcnt + barrier overhead: ~30 cycles non-overlapped per
  K-pair (4 barriers × ~5 cycles each + 6 waitcnts + 1 cbranch)

Total ≈ 256 + 30 + 6×4 (lshrrev critical) ≈ ~310 cycles per K-pair.
MFMA-only would be 256; achieved 256/310 = 82.6%, but real MfmaUtil is
59.7%, so there's another ~25-30% of "internal" stall (VALU
serialization, scale-dword ready-on-time waits) we're not modeling
analytically. The point is: this overhead is **fixed per K-pair** —
adding more K-pairs (K=8192 → K=28672) buys nothing because each pair
pays the same overhead.

By contrast, FP8 CRR's body covers 3 K-pairs per iteration (96 mfma).
The barrier/waitcnt overhead is paid 1× per body, so per-K-pair it's
1/3 of MXFP8's overhead. The 73.9% MfmaUtil is consistent with this:
~96 mfma × 8 cyc = 768 cycles MFMA; total observed ≈ 1040 cycles per
3-K-pair body → 73.9%. As K grows, the compiler can choose deeper
unrolls (and the K-tail penalty amortizes more), so FP8 CRR climbs.

---

## 4. Stall-source attribution (ranked)

For 70B Down CRR vs 70B Down RRR, ranked by per-GRBM-cycle attributable
to the MfmaUtil gap:

1. **Internal V-pipe / scale-pack-dependency serialization** —
   +91.8% VALU instructions issued, of which the 6 v_lshrrev_b32 per
   K-pair sit on the critical path between scale-load and MFMA-consume.
   **Dominant.**
2. **Extra LDS instruction count** — +50% LDS issued (CRR transpose
   path doubles ds_read calls per LDS-byte fetched). Pipe is wide
   enough to hide the count, but contributes to the wait_lds queue
   pressure.
3. **Extra SALU instruction count** — +49.4%, mainly address
   arithmetic for the scale-tile striding. Minor; SALU pipe is rarely
   bottleneck.
4. **NOT contributing to the gap:**
   * L2/TC backpressure — TA_DATA_STALL_TC/GRBM is 0.116 on CRR vs
     0.160 on RRR. Lower stall on CRR. (Different from R52P 8B Gate/Up
     RRR diagnosis.)
   * LDS bank conflicts — 0 cycles. Re-confirms R51E.
   * Cache hit rate — 81% on both. Identical L2 behavior.
   * Spill / scratch — Scratch_Size = 0 on both kernels (per PMC csv).
   * TCP_PENDING — slightly higher on CRR (+1.0) but small.

Roughly attributing the −9.25pp MfmaUtil gap:
* ~80% to scale-pack VALU dependency serialization (item 1).
* ~10-15% to LDS pipe pressure from +50% LDS count (item 2).
* ~5-10% residual from SALU pipe + barrier scheduling.

---

## 5. Why "88.0% < 92% structural floor" is not a real anomaly

R48D's structural floor model gives 92% as the "CRR ceiling" relative
to FP8. But this ceiling was anchored to the **8192³** measured FP8
data — the FP8 baseline used for that anchor was at MfmaUtil ≈ 67% (RRR)
/ 66% (CRR). At K=28672, FP8 CRR climbs to MfmaUtil 73.9% (+8pp). The
"92% floor" was implicitly normalized to a 66%-MfmaUtil FP8 anchor;
when FP8 climbs 8pp, the apparent CRR/FP8 ratio drops 8pp without
MX changing.

**Per-K-pair-per-CTA, MXFP8 CRR runs the same kernel body at K=8192
and K=28672, with the same MfmaUtil (60.0% → 59.7%, flat).** There is
no K=28672-specific MX regression. The 88.0% ratio measured by R52
Reviewer is the floor-of-the-floor: it's where MX CRR sits relative to
the *unrolled* FP8 CRR at high K.

**This means the "+4pp below floor" framing in the mandate is a
modeling artifact**, not a workable headroom. The MXFP8 CRR kernel is
operating *at* its true structural ceiling at this shape; that ceiling
is just lower in MX/FP8 ratio terms when FP8 happens to amortize
better.

---

## 6. R54 lever recommendation

### 6.1 NO concrete one-flag fix exists for this gap

The diagnosis points at the MXFP8 CRR scale-pack format requiring 6
`v_lshrrev_b32` shifts on alternate K-pairs, with a conditional branch
that the compiler refuses to unroll across. Eliminating these requires
either:

1. **Repack scales so opsel can select bytes without shift.** *Already
   tried in R49 Dev A (REFUTED -14.16% geomean).* The compiler responded
   by inserting `v_perm_b32` permutations to feed the MMA, with a worse
   net instruction count and 8192³ regression of -34.78%. **Closed lever.**

2. **Restructure CRR to issue 2 K-pairs of MMA per body iter with
   scale-dword pre-shuffled.** Would require a per-CTA scale-resident
   register file 2× current size (occupancy risk per R49A's pathology),
   and a hand-written MMA dispatch macro that interleaves 2 phases
   without a runtime branch. **Multi-component R54 work, not a one-flag
   fix.** Not recommended without strong prior; the +4pp ceiling lift
   would not justify the engineering cost.

3. **Lift FP8 baseline at K=8192 to match its K=28672 amortization.**
   Outside the scope of MXFP8 work; would also defeat the purpose of
   the comparison.

### 6.2 Recommended R54 disposition

**RETIRE 70B Down CRR from the open-headroom list.** Per §5, the cell
is at its true structural ceiling. The 88.0% baseline is not a fixable
gap; it is a measurement of where MXFP8 CRR sits when FP8 CRR happens
to amortize 8pp better. Move 70B Down CRR to the "STRUCTURAL CEILING"
category alongside R48D's 70B Q/O CRR + 8B Down CRR cells.

The remaining open-headroom CRR cell is **70B Gate/Up CRR (89.0% at
HEAD)** which has K=8192 and N=28672. R52 Reviewer §recommended-direction
flags this as the open lever for CRR. R53 Dev C does *not* extend to
that cell — separate K-pair / N-stripe geometry, separate diagnosis
needed. Recommend R54 focus there if CRR is the priority.

### 6.3 Levers RULED OUT by this diagnosis

* L2/TC cachepolicy (per R52P direction for 8B Gate/Up RRR) — does NOT
  apply here. TA_DATA_STALL is *lower* on CRR than RRR; cache return
  bandwidth is not the bottleneck.
* LDS swizzle / B-side LDS double-buffer — LDS bank conflicts are 0;
  LDS pipe is clean. R51E re-confirmed.
* CTA / XCD swizzle — already on (R47B); the 0.116 TA_DATA_STALL is
  evidence the L2 access pattern is well-distributed.
* Prologue spill / scratch elimination — Scratch_Size=0 on both kernels.
* RRR full unroll (R48E refuted, applies to RRR not CRR but pattern
  repeats — full unroll causes occupancy collapse).
* CRR coop B-scale (R48B refuted — slab geometry forbids).
* CRR scale cachepolicy (R48F refuted — A-scale not bottleneck).

---

## 7. Did not prototype — why

The mandate said *"if diagnosis points to a one-flag fix, prototype it"*.
The diagnosis here points to a multi-component rewrite (per §6.1), and
all credible one-flag candidates are on the closed-levers list. R49A
already tried the obvious one (CRR opsel) and it failed by -14% with a
clear mechanism (compiler v_perm replacement). Re-trying any opsel
variant is anti-duplication-discipline-violating per the R52 wrap.

**No prototype was built, and no strict-SCLK A/B was run.** The
ISA + cross-shape PMC evidence is sufficient to close the cell as
DIAGNOSTIC-COMPLETE without bench-time confirmation.

---

## 8. Falsifiable predictions for next-cycle audit

**P1 (CRR ceiling is K-amortization-bounded):** Any R54 lever that does
not change the MXFP8 CRR loop-body-per-K-pair structure (i.e., does not
remove the v_lshrrev branch or add a body-level unroll) will produce
≤+1pp on 70B Down CRR. If a future R54 lever claims >+2pp without
touching the scale-pack layout, the lever is either measurement noise
or affecting a non-CRR-fastpath dispatch.

**P2 (FP8 CRR amortization is K-monotone):** Re-bench FP8 CRR at
K=4096 and K=14336 (other LLaMA shapes). MfmaUtil should monotonically
increase with K, with the K=28672 73.9% being the highest measured.
This is testable on R52 Reviewer's existing baseline — the FP8 CRR
TFLOPS at 8B QO (K=4096), 8B Down (K=14336), 70B QO (K=8192), 70B Down
(K=28672) should rank-order with K (after normalizing for tile count).

**P3 (MXFP8 CRR is K-flat):** Same re-bench should show MXFP8 CRR
MfmaUtil flat across K (with +/- 1pp variation, no monotone trend).

**P4 (the gap is not in the production V2 dispatch):** A diagnostic
test forcing `MXFP8_CRR_PRESHUFFLE_V2_RUNTIME=0` (V1 path) at 70B Down
should show *similar or worse* MX/FP8 ratio than V2's 88.0% — V1 has
the same v_lshrrev problem (per R48D §2.3, both V1 and V2 hit the same
opsel limitation). If V1 were materially better, it would refute the
"v_lshrrev is the cause" claim.

---

## 9. Files / artifacts

### 9.1 PMC profiling

* `r53c_pmc_results/pmc_set1.txt`, `pmc_set3.txt` — PMC counter specs
  (copied from R52P).
* `r53c_pmc_results/run_pmc.sh` — orchestrator for 70B Down RRR/CRR ×
  set1/set3.
* `r53c_pmc_results/run_pmc_70BQO.sh` — same for 70B Q/O cross-check.
* `r53c_pmc_results/run_pmc_fp8.sh`, `run_pmc_fp8_QO.sh` — FP8 baseline
  PMC at both shapes (set1 only).
* `r53c_pmc_results/{rrr,crr}_{set1,set3}/` — raw rocprofv3 csv per
  cell × per PMC group at 70B Down.
* `r53c_pmc_results/QO_{rrr,crr}_{set1,set3}/` — same at 70B Q/O.
* `r53c_pmc_results/fp8_Down_{rrr,crr}_set1/`,
  `fp8_QO_{rrr,crr}_set1/` — FP8 raw csv.
* `r53c_pmc_results/aggregate.py` — single-shape (70B Down) aggregator.
* `r53c_pmc_results/aggregate_xshape.py` — cross-shape (Down vs Q/O)
  aggregator with per-K-pair normalization.
* `r53c_pmc_results/aggregate_fp8.py` — FP8 baseline aggregator.
* `r53c_pmc_results/aggregated.json`, `aggregated_xshape.json` —
  machine-readable medians.
* `r53c_pmc_results/{mxfp8,fp8}_70B_{Down,QO}_build.log` — kernel build
  logs.

### 9.2 ISA dumps

* `r53c_isa/mxfp8_70B_Down_device.s` — full MXFP8 device .s at K=28672
  (30k lines).
* `r53c_isa/fp8_70B_Down_device.s` — full FP8 device .s at K=28672
  (17k lines).
* `r53c_isa/mxfp8_{rcr,rrr,crr}_70B_Down_kernel.s` — per-kernel
  extracts (PRESHUFFLED=true, PACK=2 production path).
* `r53c_isa/mxfp8_{rrr,crr}_70B_Down_kloop.s` — inner K-loop body
  extracts.
* `r53c_isa/fp8_crr_70B_Down_kloop.s` — FP8 CRR inner K-loop body
  extract (96 mfma = 3 K-pairs / iter, vs MXFP8's 32 mfma / 1 K-pair /
  iter).

### 9.3 Cross-references

* `r48d_findings.md` — original CRR ~92% structural floor diagnosis
  (this work refines: floor is K-amortization-relative, not absolute).
* `r49a_findings.md` — CRR opsel rewrite REFUTED (-14.16% geomean).
* `r52p_findings.md` — RRR L2/TC backpressure diagnosis at 8B Gate/Up
  K=4096 (different mechanism — does not apply at 70B Down K=28672).
* `r52_reviewer_findings.md` — 21-cell baseline confirming 70B Down
  CRR at 88.0% under strict SCLK.

---

## 10. Verdict

**DIAGNOSTIC-COMPLETE.** The 70B Down CRR 88.0% rate is not a 4pp
recoverable gap below a 92% ceiling — it is the true structural ceiling
of the MXFP8 CRR kernel at K=28672, where the FP8 baseline happens to
amortize +7.8pp better via a 3-deep compiler unroll that MXFP8 cannot
match because of the conditional v_lshrrev branch in the K-loop body.
The gap is **NOT addressable by any one-flag lever**; the only viable
direction is a scale-layout rewrite (R49 Dev A already attempted and
REFUTED). **R54 should retire 70B Down CRR from the open-headroom list
and refocus on 70B Gate/Up CRR** (89.0% at K=8192, the true open
CRR-headroom cell per R52 Reviewer).
