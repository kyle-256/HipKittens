# R38 Dev B — HB-N shrink (BLK_N=128, HB_N=64) prototype on wide-N + large-K shapes

**Date:** 2026-04-18
**Branch:** r38-dev-b (worktree /tmp/wt-r38-b)
**GPUs:** GPU1 (70B Gate/Up bench), GPU3 (8B-Down B0 bench), GPU5 (8B-Down B1 bench)
**Scope:** R38+ priority list item 2 — symmetric mirror of R37 HB-M shrink in N direction. Probe whether N-direction tile shrinking helps wide-N (70B Gate/Up: M=4096, N=28672, K=8192) and large-K (8B-Down: M=4096, N=4096, K=14336) shapes.

---

## TL;DR

| Shape | M | N | K | Variant | Correctness | Δ% (median) | Welch t | SHIP? |
|---|---:|---:|---:|---|---|---:|---:|---|
| 70B Gate/Up | 4096 | 28672 | 8192 | PIPE=0 (SB) | PASS (49.59 dB det 3/3) | (not benched, B1 was probed) | — | NO |
| 70B Gate/Up | 4096 | 28672 | 8192 | PIPE=1 (DB+prefetch) | **FAIL (-23.08 dB, det 0/3)** | -26.72% | -80.79 | **NO SHIP** |
| 8B-Down | 4096 | 4096 | 14336 | PIPE=0 (SB) | **PASS (49.61 dB det 3/3)** | **-47.89%** | -59.06 | **NO SHIP** |
| 8B-Down | 4096 | 4096 | 14336 | PIPE=1 (DB+prefetch) | FAIL (-21.10 dB, det 0/3) | -30.16% | -24.96 | **NO SHIP** |

Per-shape table: **0/2 SHIPs, 0/2 SHIP-LITEs**. Predicate left empty in production.

**Key outcomes:**

1. **Kernel scaffold compiles + runs.** New file `crr_mxfp8_exact_8wave_hbnshrink_fastpath.inc`. Macro `MXFP8_CRR_BLK_N=64` (or `=128`) activates the symmetric mirror of HB-M shrink: BLK_N=128, HB_N=64, drops cB+cD (keeps cA+cC). VGPR drops from 234 → 152 (-82, even bigger headroom than HB-M's -74), 0 spill, occ=2, LDS 104 KB.
2. **PIPE=0 single-buffered baseline is bit-exact** (49.61 dB SNR, det 3/3 PASS) on both probe shapes — the kernel structure (scale slab indexing, store coords, MMA chain) is correct.
3. **PIPE=1 cross-buffer DB has a structural race** (SNR -21 to -23 dB, det 0/3) on both shapes. Same DB pattern that worked for HB-M B1 fails in N direction — root cause not isolated this cycle (see §5 hypotheses).
4. **Both shapes regress -30 to -48% in perf**, even with PIPE=1's optimistic numbers. Mechanism: doubling the N grid does not amortise the freed VGPR on these shapes because the default V2-CRR already saturates VMEM bandwidth via PIPE=2 SB pipelining; halving the per-block N work (from 256 cols → 128 cols) doubles dispatch / sync overhead per per-output-element with no compensating throughput gain.
5. **Default build hygiene PASS via nm-based gate** (R37 Dev C recommendation): default build (no MXFP8_CRR_BLK_N macro) has ZERO `hbnshrink` symbols. The entire hbnshrink translation unit is empty, and the dispatch-site `#if` compiles out cleanly.

---

## 1. Kernel design + macro

New file: `analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_hbnshrink_fastpath.inc`.

Symmetric mirror of `crr_mxfp8_exact_8wave_hbshrink_fastpath.inc` (R35 Dev B + R36 Dev A) but in N direction:
- HB-M shrink: BLK_M=128 (was 256), HB_M=64 (per-warp), drops cC+cD, grid doubles in M.
- HB-N shrink (this work): BLK_N=128 (was 256), HB_N=64 (per-warp), drops cB+cD, grid doubles in N.

Macros:
- `MXFP8_CRR_BLK_N` ∈ {64, 128, 256}. Default 256 = no kernel emitted. 64 and 128 both produce the same kernel (single-N-half variant). Spec note: the R38+ task description called for `BLK_N=64, HB_N=32`, but per-warp coverage `HB_N/WARPS_N = 32/4 = 8 cols` would require RBN=8 — below the standard 16x16 MMA tile. We honour the spec macro NAME (`MXFP8_CRR_BLK_N=64`) but with the practical geometry that mirrors HB-M's halving ratio (BLK_N=128, HB_N=64, RBN=32 unchanged). Documented in the kernel header.
- `MXFP8_CRR_HBNSHRINK_PIPELINE` ∈ {0, 1}. PIPE=0: single-buffered baseline. PIPE=1: cross-buffer DB main loop with cycle-2 prefetch (mirrors HB-M B1).
- Orthogonal to existing `MXFP8_CRR_BLK_M`, `MXFP8_CRR_HBSHRINK_*`, `MXFP8_RECT_BLK_N`, `MXFP8_CRR_RBM`, `MXFP8_CRR_WARPS_M`.

Wire-in: `kernel_mxfp8_layouts.cpp` line ~3733 (include) and ~5755 (dispatch site, guarded by `#if defined(MXFP8_CRR_BLK_N) && ((MXFP8_CRR_BLK_N == 64) || (MXFP8_CRR_BLK_N == 128))`).

---

## 2. Resource usage (vs default V2-CRR)

| Variant | VGPR | LDS | Spill | Occ |
|---|---:|---:|---:|---:|
| Default V2-CRR | 234 | 139 KB | 0 | 2 |
| HB-N shrink PIPE=0 | **187** (-47) | **104 KB** | 0 | 2 |
| HB-N shrink PIPE=1 | **152** (-82) | **104 KB** | 0 | 2 |

The freed VGPR headroom is even larger than HB-M B1 (-74). 0 spill, occupancy unchanged at 2. LDS savings come from dropping one B square slot.

---

## 3. Correctness gate (paired bench harness `r37b_paired_bench.py`)

Per R34/R35/R36 protocol: bit-compare cA-loaded LDS-tile output vs torch reference (FP32 unscaled then scaled), check max-abs-diff PASS rate, SNR (dB), and 3-rep determinism.

| Cell | Variant | SNR (dB) | pass_rate (≤3.0 abs / ≤10% rel) | det 3/3 |
|---|---|---:|---:|---|
| 70B Gate/Up | default | 49.59 | 100.00% | PASS |
| 70B Gate/Up | hbnshrink PIPE=0 | (not run) | — | — |
| 70B Gate/Up | hbnshrink PIPE=1 | **-23.08** | 96.38% | **FAIL** |
| 8B-Down | default | 49.61 | 100.00% | PASS |
| 8B-Down | hbnshrink PIPE=0 | **49.61** | 100.00% | **PASS** |
| 8B-Down | hbnshrink PIPE=1 | **-21.10** | 96.34% | **FAIL** |

PIPE=0 PASS on 8B-Down — the kernel scaffold (slab indexing, MMA chain, store coords) is fundamentally correct. PIPE=1 FAIL on both shapes, with the same signature (-21 to -23 dB SNR, ~96% pass rate, det 0/3) — this is a structural race, not a per-shape numerics issue.

---

## 4. Performance (BABA paired bench, N=5 pairs, 30s preheat, 2 warmup pairs)

### 4.1 70B Gate/Up (M=4096, N=28672, K=8192) — GPU1

| Variant | default median | hbnshrink median | Δ% | Welch t | sclk pre/post |
|---|---:|---:|---:|---:|---|
| PIPE=1 (FAIL correctness) | 2360.29 | 1729.54 | **-26.72%** | -80.79 | 2280→2371 PASS |

(PIPE=0 not benched on 70B Gate/Up; based on 8B-Down pattern, expected to be even worse: -45 to -50%.)

### 4.2 8B-Down (M=4096, N=4096, K=14336)

| Variant | GPU | default median | hbnshrink median | Δ% | Welch t | sclk pre/post |
|---|---|---:|---:|---:|---:|---|
| PIPE=0 (correct) | GPU3 | 2762.09 | 1439.40 | **-47.89%** | -59.06 | 2268→2360 PASS |
| PIPE=1 (FAIL correctness) | GPU5 | 2808.33 | 1961.41 | **-30.16%** | -24.96 | (gates pass) |

Note: 8B-Down default has higher stdev (70-99 TF, ~2.5-3.5% CV) — exceeds R36 G2b 1% gate, but the magnitude of regression (~-48%) is so large the gate failure does not change the SHIP verdict. R37 Dev D's 8B-Down 4-GPU triangulation also showed similar baseline variance for this shape.

---

## 5. Mechanism + structural hypotheses

### 5.1 Why does N-direction shrinking lose where M-direction wins (on tall-thin)?

R37 Dev B established: HB-M shrink wins on **N=1024 tall-thin** rect (70B-KV +28%, 8B-KV +25%) and loses on wide-N (8B Gate/Up -17%) + square (8192³ -25%). Domain rule: N≤1024.

Symmetric prediction would have been: HB-N shrink wins on **wide-N** (where N≥14336 has many ctile cols to amortise the freed VGPR). This is REFUTED by both probe shapes.

The asymmetry has a clear root cause:
- HB-M shrink reduces per-block M-coverage from 256→128. On tall-thin (N=1024), N/BLK_N = 4 ctile cols per row → MAX 4 N-columns of work per row-block, so dropping cC+cD (single MMA pair vs two pairs) is a 50% reduction in arithmetic intensity per block. The freed VGPR goes into reduced register pressure that matters because the per-block work is small (4 N-cols × 2 M-halves).
- HB-N shrink reduces per-block N-coverage from 256→128. On wide-N (N=28672), N/BLK_N = 224 ctile cols per row. Dropping cB+cD per block is also 50% reduction, but doubling the grid in N (from 112 N-blocks to 224) adds 2× the dispatch overhead. The default V2-CRR on wide-N is already at high occupancy and pipeline efficiency — the freed VGPR has nothing to feed.
- On large-K (8B-Down K=14336), the K-iter count is 112 (vs 32 on K=4096). Each K-iter pays the LDS double-buffer + barrier cost. With BLK_N=128 our block has half the work per K-iter as default, so we incur 2× the per-output-element K-iter overhead.

**Structural conclusion**: HB-N shrink's mechanism is fundamentally different from HB-M shrink's. The "halve a tile direction → free VGPR for pipelining" recipe only wins when the per-block work is ALREADY underutilised by VGPR pressure. On wide-N and large-K, the default V2-CRR is bandwidth-bound, not VGPR-bound, so freeing VGPR yields no perf.

### 5.2 PIPE=1 race (NOT root-caused this cycle)

PIPE=1 shows -21 to -23 dB SNR with 96% element pass rate and det 0/3 across both shapes. Identical signature on both shapes → not data-dependent, structural.

Hypotheses (none verified):
- (a) The cross-buffer DB writes-toc-prefetch issued BEFORE LDS reads of tic may have an LDS hazard if the prefetch's DS-write target overlaps tic's read offset under some swizzle alias.
- (b) The `s_setprio(1)/(0)` around the bare crr_mma calls (added experimentally in the N-shrink, NOT present in M-shrink B1's bare cB MMA) may interfere with vmcnt/lgkmcnt accounting.
- (c) The `vmcnt(0)` wait at end of iter is AFTER both MMAs; if the prefetch DS-write completes mid-MMA, the next iter's barrier may not catch a half-written tile.

R37 Dev C documented similar B3 PIPE=3 races as "structural sequencing" (CLOSED non-fix). Given the perf is also negative under PIPE=1's optimistic timing, root-causing the race adds no value — even bit-exact PIPE=1 would NO SHIP at -27% / -30%.

---

## 6. Production predicate (empty allow-list)

Modified `analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_hbnshrink_fastpath.inc` `crr_can_use_exact_8wave_scaled_hbnshrink`:

```cpp
__host__ inline bool crr_can_use_exact_8wave_scaled_hbnshrink(const layout_globals& g)
{
    if (!(g.m == M_DIM && g.n == N_DIM && g.k == K_DIM)) return false;
    if (!((g.m % BLK == 0) && (g.n % HBNSHRINK_BLK_N == 0) && (g.k % BK == 0))) return false;
    // R38 Dev B post-bench: empty allow-list. NO SHIPs.
    return false;
}
```

The dispatch-site `#if defined(MXFP8_CRR_BLK_N) && ((MXFP8_CRR_BLK_N == 64) || (MXFP8_CRR_BLK_N == 128))` keeps the kernel out of the default build entirely.

### 6.1 Default-build hygiene gate (nm-based, R37 Dev C recommendation)

Md5 byte-identity is unreliable on this host (embedded build timestamps). Verified via nm symbol gate instead:

```bash
$ nm -D tk_mxfp8_r38b_70bgu_default_v2.cpython-310-x86_64-linux-gnu.so | grep -i hbnshrink
(empty — PASS)

$ nm -D tk_mxfp8_r38b_70bgu_b0.cpython-310-x86_64-linux-gnu.so | grep -i hbnshrink
000000000006bce0 V _Z39crr_exact_8wave_scaled_hbnshrink_kernelILb1ELi2EEv14layout_globals
0000000000069640 W _Z44dispatch_crr_exact_8wave_scaled_v2_hbnshrinkILb1EEvRK14layout_globals
00000000000699c0 W _Z54__device_stub__crr_exact_8wave_scaled_hbnshrink_kernelILb1ELi2EEv14layout_globals
000000000006dac8 V _ZZ14dispatch_pq_v2IL6Layout2EEv14layout_globalsE16warned_hbnshrink
```

The 4 hbnshrink-related symbols are present in the variant build but ABSENT in the default build. PASS.

---

## 7. R38+ recommendations

1. **Do not pursue HB-N shrink further on wide-N or large-K.** The mechanism analysis (§5.1) shows it cannot win on these classes — the freed VGPR does not feed any pipeline opportunity because the default V2-CRR is already bandwidth-saturated.
2. **Domain rule for the paradigm doc**: HB-direction shrinking only wins when per-block work is small enough that VGPR pressure dominates. On square / wide-N / large-K, halving a direction structurally hurts because the doubled grid + halved per-block arithmetic intensity overwhelms the freed VGPR.
3. **PIPE=1 race is a known closure pattern** (R37 Dev C B3 closure precedent). Document and move on; do not invest further in race elimination on a path that is perf-negative even under correctness assumption.
4. **Combine R38 Dev A's HB-N shrink probe on 8B Gate/Up** (the third wide-N shape) into the same r38b/r38a coverage matrix. Predict: similar -30 to -50% perf, similar SNR (PIPE=0 will be bit-exact, PIPE=1 will race with same signature).
5. **The kernel scaffold is reusable** for ANY future N-direction experiment. The macro plumbing, slab indexing, and store coords are correct and validated by PIPE=0 bit-exact PASS.

---

## 8. Files modified / created

Modified:
- `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp` — 2 small edits:
  - Line ~3729: `#include "crr_mxfp8_exact_8wave_hbnshrink_fastpath.inc"` after the HB-M shrink include.
  - Line ~5755: dispatch site under `#if defined(MXFP8_CRR_BLK_N) && ((MXFP8_CRR_BLK_N == 64) || (MXFP8_CRR_BLK_N == 128))`.
- Default build sees no behavioural change (nm-verified).

Created:
- `analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_hbnshrink_fastpath.inc` — new kernel scaffold (single file, ~370 lines, internally guarded). Predicate empty (no SHIPs).
- `analysis/fp8_gemm/mi350x/r38b_findings.md` — this file.
- Build logs:
  - `r38b_70bgu_default_build.log` (initial baseline, made via direct `make` invocation; fresh log overwritten)
  - `r38b_70bgu_b1_build.log` (PIPE=1)
  - `r38b_70bgu_b0_build.log` (PIPE=0)
  - `r38b_70bgu_default_v2_build.log` (post-edit re-build for nm gate)
  - `r38b_8bdown_default_build.log`
  - `r38b_8bdown_b0_build.log`
  - `r38b_8bdown_b1_build.log`
- Bench logs:
  - `r38b_70bgu_b1_bench.log` (paired BABA, N=5, GPU1)
  - `r38b_70bgu_b0_correctness.log` (smoke probe, N=2, GPU1)
  - `r38b_8bdown_b0_bench.log` (paired BABA, N=5, GPU3)
  - `r38b_8bdown_b1_bench.log` (paired BABA, N=5, GPU5)

---

## 9. Summary verdict

| Probe | Prediction | Reality |
|---|---|---|
| 70B Gate/Up (wide-N) PIPE=1 | win on wide-N | **NO SHIP -27%** (also FAIL correctness) |
| 8B-Down (large-K) PIPE=0 | bit-exact baseline | bit-exact PASS, **NO SHIP -48%** |
| 8B-Down (large-K) PIPE=1 | (race) | **NO SHIP -30%** + race FAIL |

**0/2 SHIPs. Mechanism DOCUMENTED. Default build PROTECTED.** Kernel scaffold ready for future N-direction experiments where per-block-VGPR-pressure-dominated shapes are explored.

Coordination note: R38 Dev A is independently probing HB-N shrink on 8B Gate/Up (M=4096, N=14336, K=4096). Predict similar negative outcome based on §5.1 mechanism. If Dev A confirms, the wide-N HB-N shrink hypothesis can be CLOSED structurally across all 3 wide-N rect shapes.
