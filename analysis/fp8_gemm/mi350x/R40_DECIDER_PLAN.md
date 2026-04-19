# R40 Decider Plan — Path to 42/42 Verified-Correct WIN

**Date**: 2026-04-19
**Gate**: R39B (`bench_all_42_R39B.py`): `wrong_cell_frac < 2% AND snr_med ≥ 10 dB AND finite ≥ 0.99`, random scale [-2,2].
**Current**: 6/42 verified-correct (3 WIN + 3 LOSE_CORRECT) — all small-K (≤4096) shapes that happen to have clean upper-left 1024×1024.

---

## Synthesis of R37→R39 (root-cause re-analysis)

### What R39A's data tells us

R39A (TAIL_SCALE_CLAMP — variant 0, freeze scale on tail iters) shipped 3 WIN / 8 LOSE / 26 WRONG / 5 ERR under the **old** finite-only gate (R37 was 14 WIN). **The R38F hypothesis ("scale advances bt+1 while data is clamped") is REFUTED as the singular root cause** — clamping the scale didn't recover the WRONG_OUTPUT shapes; it actively *demoted* R37 WINs by paying a perf cost without buying correctness. R39A confirms the bug is NOT scale-data tile misalignment on tail iters alone.

### The R39B per-cell-map gives us a much sharper picture

Three distinct failure clusters across the 36 broken shapes:

| Cluster | snr_med | wrong_cell_frac | Count | Shapes characteristic |
|---|---|---|---|---|
| **A — borderline** | 24-50 dB | 0.5-1.5% | ~7 | Most rows pristine; <2% cells are bf16-overflow garbage. Small per-shape fix away from PASS. |
| **B — systematic** | -4 to -12 dB | 2-15% | ~17 | Whole regions of output are BF16-overflow. Matches R35 "upper-left 128×128 of every 256×256 tile" pattern. |
| **C — catastrophic** | -12 dB, finite < 0.8 | 15-28% | ~8 | Large-K shapes where BF16 saturation cascades. Signal mostly destroyed. |
| **D — CRASH** | n/a | n/a | 4 | M ≥ 14336 + N ≥ 28672 + K=2048 or 4096. Aperture violation. |

### The root cause we should now believe (revised)

R35's diagnosis was **acc_A0Bl in mh=0/nh=0 (upper-left 128×128 quadrant) of every 256×256 output tile is corrupted on non-FUSED_STEP34 path**. R37's "FIX B" backports `kpair_64mfma_step34` (single asm block) into the default path expressly to close this. Yet R39B shows **the upper-left corruption persists in cluster B+C** despite R37_FIX_B being default ON.

This means **R37_FIX_B is INCOMPLETE**. The fused step34 closes one of the corruption sources (the mid-iter compiler interleaving between step3 and step4 asm blocks), but a *second* corruption source remains: the **step12→step34 boundary**. Specifically:

- `kpair_64mfma_step12` is a SEPARATE asm block from `kpair_64mfma_step34`.
- Between them sits an `s_waitcnt lgkmcnt(0)` + scalar extracts (`extract_tile`) + EARLY_SCALE_PF buffer_loads (lines 2737-2740, 3087-3090).
- The compiler is free to schedule loads (the `pf_*` buffer_load_to_lds intrinsics) BETWEEN those two asm blocks. R37 added an `asm volatile("" ::: "memory")` AFTER step34, but **NOT BEFORE step12**. The pf_*_p struct construction at the top of the iter (lines 2680-2687) sits free in scheduler-land.
- Result: a `buffer_load_to_lds` from this iter's prefetch can land in the LDS slot that step12/step34 is mid-MFMA-reading. Mid-tile corruption. Random per row. Manifests as the 17% deterministic-wrong cells.

### Connection between R35's "upper-left 128×128" and R38F's "scale/data misalign"

These are **the same underlying bug observed at two scales**:
- R35's pattern: which *cells* within an output tile are wrong (it's the cells whose accumulator was actively read-then-written across the contended LDS slot during the bad iter).
- R38F's signature: which *iters* trigger it (tail iters whose pf_bt clamps to k_byte_iters-1, doubling up the same LDS line; this is when the contention probability spikes).

R39A clamping the scale didn't help because the corruption is not in the scale operand — it's in the LDS data slot the next-iter's MFMA reads.

---

## Why the 6 PASS shapes pass

All 6 PASS shapes (3 WIN + 3 LOSE_CORRECT) have **small K** (2048-4096) and **the K-loop unroll factor places R25C_TAIL_PF_OFF_ITERS=4** (or PF gets disabled outright). Specifically:
- `R25C_TAIL_PF_OFF_ITERS=4` × `bytes/iter` = the LAST iter is the only contended one, and on these K-values the LDS contention window is too short for the load to cross MFMA reads.
- Crucially, all 6 PASS shapes use variant `_pfoff4` or `_pfoff14`, where `R25C_TAIL_PF_OFF_ITERS ≥ k_byte_iters - 4` → the tail-pf-off path **never even runs** on most iters. The LDS contention-window simply doesn't open.

This is consistent with R39B's wrong_cell_frac ≤ 0.91% on these PASSes (vs 5-25% on the same kernel structure with `_pfoff14` on K ≥ 8192).

---

## R40 ranked attack plan (4 parallel optimizers)

### R40_OPT_A — Fence the entire iter (highest-confidence)

**Hypothesis**: The compiler reorders `buffer_load_to_lds` (issued via the `pf_*_p` struct construction + emit_pf_tail) across the `kpair_64mfma_step12` asm boundary, landing in an LDS slot mid-MFMA. R37_FIX_B's post-step34 fence is necessary but not sufficient; the iter needs a fence BOTH BEFORE step12 AND AFTER step34, AND the pf_*_p construction must be re-ordered to AFTER step34.

**Surgical change** (`kernel_mxfp4_gluon_cpp.cpp`):
- Add `asm volatile("" ::: "memory")` immediately BEFORE the `kpair_64mfma_step12` call (line ~2731 and ~3083).
- MOVE the `make_pf_params` block (lines 2680-2687, 2932 etc.) from the TOP of the iter to AFTER `kpair_64mfma_step34` returns (line ~3110 in the R37_FIX_B branch).
- Wrap in macro `R40A_PF_FENCE` (default 1).

**Predicted outcome**: Recover ~7 cluster-A shapes (the 49.6 dB snr_med, <1.5% wrong shapes). Predicted R39B WIN: 13/42.

**Risk**: Pf_*_p construction post-step34 means the prefetch is issued LATER in the iter → less overlap. Predicted perf cost: 2-4% on K-bound shapes. If both fences serialize too aggressively, may LOSE perf even where correctness recovers.

**Falsification**: If wrong_cell_frac stays > 2% on ANY of {16384x4096x3072, 32768x4096x3072, 4096x32768x4096} (currently borderline), the LDS-contention hypothesis is wrong and the bug is in the MFMA register file (R35 hypothesis 3).

**Test protocol**:
- Smoke: `m16384_n6144_k2048` (currently 12.10% wrong, snr_med=0.7) + `m6144_n32768_k4096` (currently 0.64% wrong, on the boundary).
- Full: `bench_all_42_R39B.py`, 3-run consensus.

**Effort**: 2 agent-hours.

---

### R40_OPT_B — Fork BEST_VARIANTS to FUSED_STEP34=1 + drop pfoff

**Hypothesis**: R35 already proved `FUSED_STEP34=1` produces 0.06% non-finite vs 6-8% on the non-fused path; the only reason BEST_VARIANTS picked non-fused was that they were faster *while writing garbage*. R37_FIX_B was supposed to backport this but R39B shows it didn't fully. **Just enable the original FUSED_STEP34=1 path** (which has its own clean step34 emission) and accept the perf cost.

**Surgical change**:
- For every entry in `R38_BEST_VARIANTS_v3.py`: append `-DFUSED_STEP34=1 -DR25C_TAIL_PF_OFF_ITERS=0` (drops both R37_FIX_B path AND the pfoff tail-skip). Strip `_memc` (max-memory-clause) from variant name and CPPFLAGS.
- New variant suffix `_R40B_safe`.

**Predicted outcome**: Largest correctness recovery. Probably 25-35 of 42 shapes pass R39B. Perf hit: 5-15% per shape vs current "fast garbage" baselines, but still likely WIN vs aiter on M=4096 and M=6144 shapes (where we have headroom). Predicted R39B WIN: 18-25/42.

**Risk**: If FUSED_STEP34=1 has its OWN tail-iter race (was previously masked because non-fused path was the default), this won't help. R35 only verified one shape (n4096_k2048) for FUSED_STEP34. Need broader verification.

**Falsification**: Build 5 representative shapes; check wrong_cell_frac on all 5 < 2%. If even one fails, the hypothesis weakens.

**Test protocol**:
- Smoke: rebuild 4 shapes spanning K-clusters: `m4096_n4096_k8192` (cluster B), `m16384_n4096_k14336` (cluster C), `m32768_n6144_k2048` (already PASS, sanity), `m4096_n28672_k32768` (worst CRASH-borderline).
- Full: `bench_all_42_R39B.py`, 3-run consensus.

**Effort**: 3 agent-hours (build is 10 min × 5 retries; bench is 5 min × 3 reps).

---

### R40_OPT_C — LDS double-buffer slot quarantine

**Hypothesis**: The corruption is specifically in the `acc_A0Bl` register file (R35), and R35 hypothesis 3 (MFMA write-after-write hazard between step12's writes to acc_A0Bl and step3's reads of A1) is the real mechanism. The fix is not at the asm-block boundary but in the **double-buffer slot allocation**: force step3+step4 to read from a DIFFERENT LDS slot than step1+step2 wrote to.

**Surgical change**:
- Audit `A0_db[cur]` / `Bl_db[cur]` indexing in `make_pf_params` (lines 2680-2687) — currently `cur` flips parity per iter, so iter N writes to slot 0 and iter N+1 reads from slot 1. But `kpair_64mfma_step34` reads `tBl` (which was loaded into `tA0`/`tBl` at iter top from slot `1-cur`) — **verify no aliasing**.
- Insert an `asm volatile("ds_swizzle_b32 ...")` or explicit `s_waitcnt lgkmcnt(0)` between the LDS write (in `kpair_32mfma_with_lds_and_pf`) and the next-iter LDS read.
- New macro `R40C_LDS_DRAIN` — drains lgkmcnt(0) immediately before step12.

**Predicted outcome**: If the bug is the DS-write/DS-read race specifically (not the buffer_load contention), this recovers cluster B (the 17 systematic-wrong shapes). Predicted R39B WIN: 10-18/42 if hypothesis correct.

**Risk**: Adding lgkmcnt(0) drains LDS pipelining, which is the kernel's main latency-hide mechanism on K-bound shapes. Perf cost 8-15% likely. Also: R38F variant 4 already tested `s_waitcnt vmcnt(0)+lgkmcnt(0)` on tail iters and it didn't work — but R38F gated only to tail iters; this proposes drain on EVERY iter, which is structurally different.

**Falsification**: If R40A also fixes it, R40C is unnecessary. R40C only stays in the running if R40A's per-cell map shows residual wrong cells in `acc_A0Bl` quadrant specifically.

**Test protocol**:
- Smoke: `m4096_n4096_k8192` + r35-style `r35_nonfinite_pattern.py` analysis to verify the upper-left 128×128 quadrant clears.
- Full: `bench_all_42_R39B.py`, 3-run consensus.

**Effort**: 4 agent-hours (need to re-instrument the per-cell map to confirm quadrant-level recovery).

---

### R40_OPT_D — Strip ALL prefetch from non-tail K-loop (correctness-first baseline)

**Hypothesis**: Until we know which prefetch path is racing, eliminate all of them and establish a known-correct (slow) baseline. From there, incrementally re-add prefetches and measure the wrong_cell_frac at each step. This is binary-search-bisection of the responsible code path.

**Surgical change**:
- New macro `R40D_NO_PREFETCH=1`: in the K-loop body (lines 2929-3196), comment out:
  - `emit_pf_tail<0>(pf_a0_p, pf_a1_p)` calls
  - `emit_pf_tail<0>(pf_bl_p, pf_br_p)` calls
  - The `make_pf_params` block (use dummy/zero structs)
- Keep `load_pq_scale_x2_async` (those are scale loads, smaller bandwidth, less likely culprits).
- Force `STEP3_PF_N=0`, `STEP4_PF_N=0`.

**Predicted outcome**: This kernel will be SLOW (no prefetch overlap = ~30-50% perf regression) but should be CORRECT on most shapes. If R39B wrong_cell_frac drops to < 2% on ALL 42 shapes, the bug IS in the prefetch path → R40A is correct direction. If wrong cells PERSIST, the bug is elsewhere (MFMA / register file / store path) and R40C is needed.

**Risk**: This is purely diagnostic; no chance of WIN. But it gives the team a known-correct binary to bisect against.

**Falsification**: If R40D still has wrong_cell_frac > 2% on any shape, the prefetch-LDS hypothesis (R40A) is wrong.

**Test protocol**:
- Smoke: `m4096_n4096_k8192` (representative cluster B), `m16384_n4096_k28672` (worst cluster C).
- Full: `bench_all_42_R39B.py`. Skip TFLOPS comparison; only measure correctness.

**Effort**: 1.5 agent-hours.

---

## Reviewer protocol (post-R40)

For each R40_OPT_X that lands a candidate kernel, the reviewer agent must:

1. **Re-bench under R39B with 5 runs** (not 3) per shape. Report per-shape wrong_cell_frac stddev across runs. Flag any shape where stddev > 1% (gate-flake risk).

2. **Generate per-shape per-cell wrong-cell map** for the 7 cluster-A borderline shapes that R40A targets. Verify the wrong cells are NOT in the upper-left 128×128 quadrant (R35 signature) anymore — if they are, R37_FIX_B fence is still incomplete.

3. **Verify no R37 PASS shapes regressed** in the 6 currently-passing (under R39B). Any regression is a STOP-the-line: roll back and reconsider.

4. **TFLOPS sanity vs comp**: for any new PASS shape, report `tflops/comp` ratio. If < 90%, classify as LOSE_CORRECT and check whether perf headroom exists by re-tuning variant flags.

5. **Cross-compare R40A vs R40D**: on the 36 broken shapes, compute `wrong_cell_frac(R40A) - wrong_cell_frac(R40D)`. If R40A ≈ R40D wrong-cell-frac, R40A's fence works as expected. If R40A ≫ R40D wrong-cell-frac, R40A missed a prefetch path. If R40A ≪ R40D wrong-cell-frac, something is wrong with the test (impossible since R40D removes all prefetches).

6. **Commit on every WIN-delta**: if R40_OPT_X moves R39B verified-correct from 6 to N>6, commit immediately with leaderboard delta in message.

---

## Bench discipline

- All measurements: `bench_all_42_R39B.py`, warmup=200, iters=500, trim=10%.
- 3-run consensus (majority-vote) for verdicts; 5-run for reviewer audit.
- 8 GPUs available; check `rocm-smi` first.
- DO NOT use `bench_all_42_R37.py` (uniform-scale gate hides defect).

---

## Recommended dispatch order

Launch **all 4 in parallel** (independent kernel macros):
- A: most likely to add ~7 wins quickly.
- B: highest absolute potential (FUSED_STEP34=1 already known to be correct on 1 shape).
- C: only valuable if A doesn't fully recover cluster A; can be paused if A wins big.
- D: diagnostic; finishes fast (1.5h) and unblocks A/C analysis.

Total expected wall-time: 4 hours. Expected R39B verified-correct after R40: **18-30/42** depending on which hypothesis dominates.
