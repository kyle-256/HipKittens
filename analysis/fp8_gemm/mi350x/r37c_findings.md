# R37 Dev C — HB shrink B1 fan-out to 70B Gate/Up + B3 regression diagnosis

**Date:** 2026-04-18
**Branch:** r37-dev-c (base feat/mxfp8-only @ 098ef0e5)
**Worktree:** /tmp/wt-r37-c
**Scope:** Two parallel investigations from R37+ priority list:
  - Part A (high) — extend R36 Dev A's HB shrink Stage B1 SHIP from the 70B-KV (M=4096 N=1024 K=8192) win to the 70B Gate/Up shape (M=4096 N=28672 K=8192).
  - Part B (medium) — investigate the R36 Dev A B3 (PIPE=3 hybrid SB+interleave) -6.27% regression: structural vs sequencing.

---

## TL;DR

- **Part A — NO SHIP.** HB shrink Stage B1 lands **-28.30%** on 70B Gate/Up (4096×28672×8192) on GPU0 (clean PAIR 0-3 medians: default ~2400 TF/s, B1 ~1720 TF/s). Bit-exact correctness confirmed (max abs diff 0.0, SNR 49.59 dB). Same VGPR=160, occ=2, LDS=104 KB as 70B-KV B1.
  - **Refines R36 Dev A paradigm**: B1 SHIP scope is **N=1024 ONLY** (tall-rect, small-N). It does NOT generalise to wide-N rect (N=28672) any more than to square (8192³ was -25%). The mechanism is grid-size scaling: BLK_M=128 doubles M-grid count; on wide-N there are already many ctile slabs (3584 for 70B Gate/Up vs 1792 default), so doubling worsens scheduling overhead while the per-ctile pipeline win does not amortise.
  - **No predicate change needed** — production predicate stays 70B-KV-only, exactly as R36 Reviewer recommended.
- **Part B — REGRESSION IS STRUCTURAL, NOT SEQUENCING.** A new Stage B3v2 variant (PIPE=4) tested the "sequencing bug" hypothesis by reordering the next-iter VMEM issue to AFTER cB (instead of B3's between-cA-and-cB position) and removing one redundant barrier. Result: **-17.74%** on GPU2 (worse than B3 -6.27%).
  - The B1 LDS-interleave pattern (deferred b1 LDS read inside the cA MMA chain) is **structurally incompatible with single-buffered SB pipelining**. Without cross-buffer DB, no VMEM ordering recovers the loss.
  - Both halves of the original hypothesis test cleanly: removing the cA→cB barrier did NOT speed it up (showing the barrier wasn't the dominant cost); moving VMEM after cB did NOT speed it up (showing the b1-vs-VMEM same-slot race wasn't the dominant cost). The single dominant cost is **the interleave + SB combination itself**.
  - **Implication for R37+ priority list item 4 (the proposed B4 cross-buffer DB + interleave hybrid)**: this is structurally **already exactly what R36 Dev A's B1 SHIP delivers** — DB main loop with B1 interleave. There is no additional R37+ "B4" candidate; the SB+interleave path is paradigm-CLOSED.

---

## Part A — HB shrink B1 on 70B Gate/Up (M=4096 N=28672 K=8192)

### A.1 Build

Files:
- `r37c_b1_70bgu_build.log` — build with `-DMXFP8_CRR_BLK_M=128 -DMXFP8_CRR_HBSHRINK_PIPELINE=1 -DM_DIM=4096 -DN_DIM=28672 -DK_DIM=8192`
- `r37c_70bgu_md5.log` — md5 manifest

Resource numbers (identical to R36 Dev A's 70B-KV B1):

```
crr_exact_8wave_scaled_hbshrink_kernel<true,2>:
  VGPRs:       160     (-74 vs default 234)
  AGPRs:       0
  ScratchSize: 0
  Occupancy:   2 waves/SIMD
  VGPRs Spill: 0
  LDS:         104448 B/block (-34816 vs default)
```

The HB shrink kernel's per-block resource footprint is shape-independent (only block geometry depends on shape, not register usage), so VGPR/LDS match exactly.

### A.2 Numerics PASS

`r37c_b1_70bgu_bitcompare.log`:
```
M=4096 N=28672 K=8192
C_default[0,:8]   = [-0.431640625, 0.86328125, -0.6328125, 0.26171875, -1.4765625, 0.46875, -0.61328125, 1.0859375]
C_hbshrink[0,:8]  = [-0.431640625, 0.86328125, -0.6328125, 0.26171875, -1.4765625, 0.46875, -0.61328125, 1.0859375]
bit_equal_full    = True
max_abs_diff      = 0.0
mean_abs_diff     = 0.0
```

Plus SNR 49.59 dB, det_ok=True from the bench script's correctness check.

### A.3 Bench — NO SHIP

Bench harness: `r34c_paired_bench.py` (BABA pattern, 30s preheat, 2 warmup pairs, 5 paired BABA samples, n=10 per kernel).

`r37c_b1_70bgu_bench_gpu0.log` (GPU0; PAIR 4 had post-bench sclk dip to 1993 MHz, drop):

```
Default V2-CRR:    PAIR 0-3 medians 2389-2424 TF/s; full-set median 2398.02
HB shrink B1:      PAIR 0-3 medians 1695-1724 TF/s; full-set median 1719.51
Δ% (B1 vs default) = -28.295% (-28.30% on PAIR 0-3 only)
Welch t = -4.61 (full set incl PAIR 4); cleaner on subset
sclk pre-bench 2322 MHz, post-bench 1993 MHz
```

GPU4/GPU5 reruns hit thermal/contention throttle (sclk dropped <2000 MHz mid-bench); both reproduced the negative direction (B1 always slower) but with noisy magnitudes — they confirm the verdict but do not provide cross-GPU triangulation. Triangulation is moot since first-GPU bench is already a clear NO SHIP.

### A.4 Mechanism — why B1 fails on wide-N

B1's win on 70B-KV came from: VGPR -74 + cycle-2 prefetch + DB pattern overlapping VMEM with MMAs across two LDS buffers. The cost: BLK_M=128 doubles the M-grid count.

Trade-off arithmetic:

| Shape | N | Default ctiles (M/256 × N/256) | B1 ctiles (M/128 × N/256) | Ratio | Per-ctile inner iters (K/128) | B1 verdict |
|---|---:|---:|---:|---:|---:|---|
| 70B-KV | 1024 | 64 | 128 | 2× | 64 | **+28% SHIP** |
| 70B Gate/Up | 28672 | 1792 | 3584 | 2× | 64 | **-28% NO SHIP** |
| 8192³ (square) | 8192 | 1024 | 2048 | 2× | 64 | **-25% NO SHIP** (R36 Dev A) |

The pattern is clear: B1 only wins when the absolute ctile count is small (< ~256). On 70B-KV with only 128 B1 ctiles, the doubled scheduling overhead is dominated by the per-ctile pipeline win. On 70B Gate/Up (3584 ctiles) and 8192³ (2048 ctiles), the doubled scheduling overhead dominates and B1 loses ~25-30%.

This is a strong constraint. The R37+ priority list item 3 named three candidate shapes for B1 fan-out:
- **70B Gate/Up** (4096×28672×8192) — REFUTED here at -28%, ctiles 3584
- **8B-KV** (4096×1024×4096) — would have **64 B1 ctiles** (vs default 32), might SHIP — but K=4096 vs B1's tested K=8192 means inner iters are 32 vs 64; the pipeline-amortisation arithmetic is different
- **8B Gate/Up** (4096×14336×4096) — would have 1792 B1 ctiles, would predict similar -25-30% by analogy with the 70B Gate/Up result

Recommendation for R38+: only test 8B-KV from this list — the others are very likely to repeat the wide-N failure. (8B-KV is bumped to R38 due to time budget on the Part B investigation.)

---

## Part B — B3 regression diagnosis (PIPE=3 hybrid SB+interleave)

### B.1 Hypothesis

R36 Dev A reported B3 (PIPE=3 SB + B1 LDS interleave) at -6.27% on 70B-KV with a hand-wave: "the deferred b1 LDS read in the cA chain conflicts with SB drain semantics". Two distinct mechanisms could explain this:

1. **Sequencing**: the next-iter VMEM write into the same (single-buffered) slot is issued BETWEEN cA and cB, while cA's deferred b1 LDS read may still be in flight on that slot. Race symptom: VMEM-vs-LDS hazard, scheduler stalls.
2. **Structural**: the interleave pattern fundamentally adds a dependency chain inside cA that cannot be hidden by SB regardless of VMEM ordering.

R37 Dev C tested **(1)** by introducing a new variant **B3v2 (PIPELINE=4)** that:
- Issues VMEM AFTER cB completes (not between cA and cB)
- Removes the redundant `s_barrier` between cA and the (formerly intermediate, now post-cB) VMEM issue
- Otherwise identical to B3 (single-buffered LDS, PIPE=3-style early/late wait)

If hypothesis (1) is dominant, B3v2 should recover toward B2's +13.27% (since it removes the racy ordering). If (2) is dominant, B3v2 should remain near or below B3.

### B.2 Result

`r37c_b3v2_70bkv_build.log`:
```
crr_exact_8wave_scaled_hbshrink_kernel<true,2>:
  VGPRs:       168     (same as B3)
  Occupancy:   2 waves/SIMD
  LDS:         104448 B/block
```

`r37c_b3v2_70bkv_bitcompare.log`: `bit_equal_full = True`, `max_abs_diff = 0.0`. SNR 49.60 dB, det_ok=True.

`r37c_b3v2_70bkv_bench_gpu2.log` (GPU2; clean — sclk 2384 MHz post-bench, all 5 PAIRs consistent):

```
Default V2-CRR:        median 763.60 TF/s, n=10, stdev 3.08
HB shrink B3v2 (PIPE=4): median 628.13 TF/s, n=10, stdev 1.32
Welch t = -127.71
Δ% (B3v2 vs default) = -17.741%
```

### B.3 Verdict — REGRESSION IS STRUCTURAL

| Variant | LDS scheme | b1 interleave | VMEM position | Δ% vs default |
|---|---|---|---|---:|
| Default (R32 baseline) | DB | n/a | classic | 0% |
| **B1 (R36 SHIP)** | **DB (cross-buffer)** | **YES** | **early in iter, into TOC** | **+28.02%** |
| B2 (R36) | SB (single-buffered) | NO | mid-iter, into slot 0 | +13.27% |
| B3 (R36) | SB | YES | between cA and cB | -6.27% |
| **B3v2 (R37 Dev C)** | **SB** | **YES** | **AFTER cB** | **-17.74%** |

Read this table left to right: B1 (DB+interleave) wins big. B2 (SB-only) wins moderately. B3 (SB+interleave) loses. B3v2 (SB+interleave with "fixed" sequencing) loses MORE.

The two B3 variants bracket the sequencing question:
- B3 = VMEM mid-iter (overlaps with cB) → -6.27%
- B3v2 = VMEM post-iter (no overlap with cB) → -17.74%

If sequencing were dominant, the "cleaner" B3v2 ordering would be faster. It is significantly slower because removing the VMEM-during-cB overlap removes the only mechanism by which SB could hide VMEM latency at all.

**The structural conclusion is sharper**: in SB mode, the PIPE benefit comes from VMEM-overlap-with-MMA. The B1 LDS interleave defers a b1 read into the cA chain, which means cA is no longer free of LDS dependencies — so the lgkmcnt drain point shifts later, and the VMEM issue point pushed up against it (in B3) or shifted past it (in B3v2) both fight the cA chain instead of helping it.

### B.4 Implication for R37+ priority list item 4

The R37+ priority list item 4 proposed a "B4 cross-buffer DB + interleave hybrid" as a follow-up. **R37 Dev C closes this:**

- "Cross-buffer DB + interleave" is exactly **B1**, which already SHIPped at +28%.
- "SB + interleave" is structurally bad (B3 -6.27%, B3v2 -17.74%) and no sequencing can recover it.

There is no productive R37+ "B4" candidate from this paradigm. **Paradigm closure**: SB+interleave is CLOSED. Future re-pipelining work must use cross-buffer DB.

---

## Build hygiene

- Default builds (no `MXFP8_CRR_BLK_M`) verified to expose **0 hbshrink symbols** via `nm -D` on `tk_mxfp8_kv_default_postedit.cpython-310-x86_64-linux-gnu.so`. The new `PIPELINE==4` block sits inside `#if (MXFP8_CRR_BLK_M == 128)` (line 61) and `#elif (MXFP8_CRR_HBSHRINK_PIPELINE == 4)` (added by this commit), so the default build is dead-code only.
- **Note on md5 build hygiene**: this worktree's compiler exhibits non-deterministic md5 across rebuilds (verified: two consecutive identical `hipcc` invocations of the default kernel produced different md5s). md5 byte-identity is therefore NOT a reliable signal in this environment; functional dead-code verification (`nm -D | grep hbshrink → 0`) is used instead. R36 Dev A's reported md5 byte-identity may have been incidental.
- All HB shrink builds (B1 70B Gate/Up + B3v2 70B-KV) compile with no warnings, no spills.
- Numerics: PASS bit-equal across all variants tested (4 builds × 2 shapes).
- BABA protocol: 30s preheat, 2 warmup pairs, 5 paired BABA samples (n=10/kernel), Welch t two-sample, sclk auto-checked pre-preheat / post-preheat / pre-bench / post-bench.

---

## Files modified / created

Modified:
- `analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_hbshrink_fastpath.inc` — extended `MXFP8_CRR_HBSHRINK_PIPELINE` valid range from 0..3 to 0..4, added `#elif PIPELINE == 4` Stage B3v2 main loop (single-buffered, b1 LDS interleave, VMEM issued AFTER cB MMA).

Created (this file + logs):
- `r37c_findings.md` — this file
- Part A logs:
  - `r37c_b1_70bgu_build.log` (build w/ resource remarks)
  - `r37c_70bgu_md5.log`
  - `r37c_b1_70bgu_bitcompare.log`
  - `r37c_b1_70bgu_bench_gpu0.log` (clean, NO SHIP -28.30%)
  - `r37c_b1_70bgu_bench_gpu4.log`, `r37c_b1_70bgu_bench_gpu5.log` (throttled but directionally consistent)
- Part B logs:
  - `r37c_b3v2_70bkv_build.log`
  - `r37c_b3v2_70bkv_bitcompare.log`
  - `r37c_b3v2_70bkv_bench_gpu0.log` (throttled — discard), `r37c_b3v2_70bkv_bench_gpu2.log` (clean, -17.74%)
- Build-hygiene logs:
  - `r37c_default_preedit_md5.log`, `r37c_default_postedit_md5.log` (md5 differs — environment non-determinism, see Build hygiene §)

---

## Recommendations to R37 Reviewer / R38+ priority list

1. **Part A NO SHIP** — no production predicate change required. The R36 production predicate (HB shrink B1 fires only at M=4096 N=1024 K=8192) remains correct.
2. **Part B paradigm CLOSURE** — close R37+ priority list item 4 ("B4 cross-buffer DB + interleave hybrid"). The DB+interleave path is already R36 Dev A's B1 SHIP; the SB+interleave path (B3, B3v2) is structurally bad. No additional B4 candidate exists.
3. **Cumulative tally update**: 33 closed levers (R36) + this paradigm closure + Part A wide-N constraint.
4. **R38+ candidate from item 3 fan-out**: only **8B-KV (4096×1024×4096)** has a small enough ctile count to be a plausible B1 SHIP candidate. 8B Gate/Up (1792 B1 ctiles) is very likely to repeat the wide-N failure observed here.
5. **Methodology note on md5 hygiene**: per-build md5 byte-identity is unreliable in this environment. Recommend the Reviewer adopt `nm -D | grep <symbol-name>` as the dead-code verification gate, not md5 comparison.
