# R48 Dev B — Cooperative B-scale loading for V2-RCR / V2-CRR

**Date:** 2026-04-19
**Branch:** R48 worktree on feat/mxfp8-only @ 10878a77
**GPU:** HIP_VISIBLE_DEVICES=4 (MI355X / gfx950)
**Target cells:** 70B Down RCR (90.9% of FP8) + 70B Down CRR (88.2% of FP8)
**Shape:** M=4096, N=8192, K=28672 (B-scale = 7.34 MB > 8 MB L2 per XCD)

**Verdict:** **NO SHIP — REFUTED (STRUCTURAL + EMPIRICAL).**

The "cooperative B-scale loading" hypothesis as written ("multiple warps in
a CTA collectively prefetch B-scales for shared K-tiles") is structurally
incompatible with the V2 wave-tile preshuffle layout used by V2-RCR and
V2-CRR: each warp owns a per-warp B-scale slab (slab_idx_b = bc * WARPS_N
+ wn) covering DIFFERENT N rows, so there is no shared B-scale data
between warps in the same CTA to amortize. The two structurally-feasible
re-interpretations were tested and refuted.

---

## 1. Layout audit — there is no B-scale data shared between warps in a CTA

V2-RCR (`kernel_mxfp8_layouts.cpp:2742-2890`) and V2-CRR
(`crr_mxfp8_exact_8wave_fastpath.inc:407-447`) both use the same V2
wave-tile preshuffle slab geometry for B-scales:

```
slab_bytes_b   = 64 * padded_k_blocks       # 64 = pack_count(2) * 32
slab_idx_b     = bc * WARPS_N + wn          # one slab per (block_col, warp_n)
b_v2_base      = g.b_scale + slab_idx_b * slab_bytes_b
```

Each warp issues exactly one `buffer_load_b64` per k_pair per lane against
its own per-warp slab. WARPS_N = 4 (BLK_N=256, RBN=64). The 4 warps in a
CTA load 4 disjoint slabs covering 4 disjoint sets of N rows. Within a
single CTA there is **zero B-scale data sharing between warps**.

The cross-CTA reuse pattern (multiple `br` values sharing the same `bc`
slab) is already amortized at the L2 level by the R46 Dev D (RRR) /
R47 Dev A (RCR) / R47 Dev B (CRR) XCD-aware block swizzles, which are
default ON.

A literal "warp 0 loads ALL B-scales for the CTA's K-tile into LDS, then
all warps read from LDS" variant CAN be expressed, but it requires staging
4× as many bytes per warp and routing them through LDS. This is exactly
the `MXFP8_RCR_EXACT_PQ_SCALE_LDS_ENABLE` path that already exists
(`kernel_mxfp8_layouts.cpp:372-376, 2399-2890`), which the README records
as "Shared/LDS scale cache experiments: produced correctness problems and
did not survive validation" (R30 Dev C confirmed: V2-RCR/V2-CRR both have
zero `ds_write` in their production SASS — the entire scale path is
VMEM→VGPR→MFMA-direct, with no LDS round-trip to optimize).

A second feasible reinterpretation is "deeper VGPR prefetch ring across
k_pairs," but R31 Dev C (`r31c_findings.md`) already confirmed the
1-deep VGPR prefetch (`MXFP8_RCR_V2_SCALE_PREFETCH=1`) catastrophically
regresses V2-RCR (256 VGPR / 312 VGPR spill / 596 bytes scratch/lane;
4096³ collapses to 187 TFLOPS, -92%) — V2-RCR is at the structural
register-pressure ceiling.

## 2. Empirical confirmation: VGPR-prefetch repeats catastrophic regression on 70B Down RCR

Build: `MXFP8_RCR_V2_SCALE_PREFETCH=1`
Bench: M=4096 N=8192 K=28672, MXFP8_PRESHUFFLE_QUANT=1, 20 warmup / 50 iters

| Layout | Baseline TFLOPS | PREFETCH=1 TFLOPS | Δ |
|--------|-----------------|-------------------|---|
| RCR    | 2870            | **208**           | **-92.7%** |
| RRR    | 2934            | 2934              | 0% (untouched) |
| CRR    | 2616            | 2616              | 0% (untouched) |

Confirms R31 Dev C verdict on this specific shape — RCR at register ceiling,
prefetch is DOA on 70B Down too.

## 3. Tested lever: B-first issue reorder (MXFP8_RCR/CRR_COOPERATIVE_BSCALE)

Since the literal hypothesis is structurally infeasible and the obvious
pivot (deeper prefetch) is already-refuted, I implemented the
narrowest plausible cooperative-spirit lever: **issue the B-scale `b64`
BEFORE the A-scale `b128` inside `load_scale_buffer`** (V2-RCR + V2-CRR),
guarded by a `__builtin_amdgcn_sched_barrier(0)` to prevent the LLVM
scheduler from re-floating the order. Hypothesis: the smaller B-scale
transaction returns faster and unblocks MMA dispatch sooner; on large-K
shapes where B-scale residency exceeds L2, biasing the per-iter issue
order toward B-first might win at the VMEM-arbitration level.

Patch sites (gated, default OFF):
- `kernel_mxfp8_layouts.cpp:418-425` — `MXFP8_RCR_COOPERATIVE_BSCALE` macro decl
- `kernel_mxfp8_layouts.cpp:~2843, ~3327` — V2-RCR scale loaders (inline + lambda)
- `crr_mxfp8_exact_8wave_fastpath.inc:88-95` — `MXFP8_CRR_COOPERATIVE_BSCALE` macro decl
- `crr_mxfp8_exact_8wave_fastpath.inc:~458` — V2-CRR scale loader

VGPR / spill diff vs baseline at the RCR/CRR/RRR build with COOP=1:
identical resource counts, no new spilling.

### Bench results (3-run, 50 warmup / 100 iter, sleep 20s between)

70B Down M=4096 N=8192 K=28672:

| Layout | Baseline (3-run TFLOPS) | COOP (3-run TFLOPS) | Median Δ |
|--------|-------------------------|---------------------|----------|
| RCR    | 2922.35 / 2918.93 / 2932.51 (med 2922.35) | 2907.52 / 2927.67 / 2940.60 (med 2927.67) | **+0.18%** |
| CRR    | 2638.59 / 2628.44 / 2644.86 (med 2638.59) | 2632.42 / 2634.94 / 2630.22 (med 2632.42) | **-0.23%** |
| RRR    | 2946.66 / 2946.63 / 2936.84 (med 2946.63) | 2934.89 / 2927.77 / 2940.42 (med 2934.89) | -0.40% (RRR untouched, within run-to-run noise; cf. ±2% envelope) |

All deltas are within ±0.5% (well inside the ±2% within-GPU noise envelope
and far below the +2% SHIP gate on RCR/CRR). The B-first issue order is a
no-op — the LLVM scheduler is already producing an optimal interleaving of
the two VMEM transactions, and pinning the order with a sched-barrier
provides no measurable benefit.

### Correctness gate (sanity check)

`MXFP8_CHECK=1`, MXFP8_DETERMINISM_RUNS=3, COOP=1:

- RCR: SNR 49.61 dB (threshold 48.0 dB) — **PASS**
- CRR: SNR 49.60 dB (threshold 48.0 dB) — **PASS**

(Output is bit-stable across the 3 determinism runs since SNR is computed
across them; the kernel is deterministic.)

## 4. Conclusion

The hypothesis is REFUTED for V2-RCR / V2-CRR on 70B Down:

1. **Structural:** the V2 layout has no inter-warp B-scale sharing within a
   CTA; the warp-disjoint slab geometry is the design point that already
   minimizes B-scale fanout per warp's MFMA dispatch.
2. **Already-refuted alternatives:** LDS scale staging (R30 Dev C / README)
   and VGPR-prefetch ring (R31 Dev C, re-confirmed here on 70B Down) are
   both DOA — the kernel is at the register-pressure ceiling and has zero
   LDS scale path to second-buffer.
3. **Empirically tested narrow lever (B-first issue reorder, this work):**
   no measurable effect (±0.5%, well inside noise) — the LLVM scheduler is
   already issuing an optimal order.

The 70B Down RCR/CRR gap (4.1pp / 6.8pp vs FP8) is **not in the B-scale
loading path**. Productive directions now point elsewhere — likely the
A-side scale path (PC=4 b128 has 2× the bytes of B's b64 b128 and is the
bigger contributor to scale VMEM volume) or the FP8 tile loading itself
(28672 K-dim is also the largest K, so A-tile + B-tile data movement
dominates).

## 5. Macros left in the tree (default OFF)

- `MXFP8_RCR_COOPERATIVE_BSCALE` (default 0) — V2-RCR B-first reorder
- `MXFP8_CRR_COOPERATIVE_BSCALE` (default 0) — V2-CRR B-first reorder

These are kept (not reverted) so a future investigator can reproduce the
benchmark with one CXXFLAGS toggle without re-deriving the patch sites.
The existing ±0.5% no-op result is now the documented null prior for
"issue-order cooperative B-scale" — saving a future cycle from re-running
the same experiment.

## 6. Logs

- `r48b_baseline_70b_down.log` — 3-run baseline (RCR/CRR/RRR)
- `r48b_coop_70b_down.log`     — 3-run COOP=1 (RCR/CRR/RRR)
