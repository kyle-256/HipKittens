# R26 Plan — Post-R25 Research Brief

Date: 2026-04-18  Branch: mxfp4  Decider model: claude-opus-4-7

## 1. Where R25 left us

| Class | Mechanism wired | Status |
|---|---|---|
| R25-C/D | `R25C_TAIL_PF_OFF_ITERS={4,14}` × `gm6/gm7` | committed (`7ada8c70`, `e5083bad`) |
| R25-F   | `gm7+pfoff14` for K=4096 (DLA2/DLA7) | committed |
| R25-G   | per-K `pfoff = K_iters - {4..8}` for K∈{14336,16384,28672,32768} | committed (`7f200b76`) |
| R25-H   | per-K `pfoff = K_iters - {4..6}` for K∈{2048,6144,7168,8192} | committed (`a2c85d86`) |

**Tail-prefetch axis is exhausted** within ±2 of `K_iters - 6`; verify3 shows wider sweep on SD (16384×4096×28672) flat or worse. **Don't touch this axis again.**

Confirmed already-implemented (do NOT re-investigate):
- Scale loads use `buffer_load_dwordx2` with SGPR SRD + `NONVOLATILE_SCALE_X2_POC=1` (kernel L695, ISA L186 — VGPR-resident `pf_a0/pf_a1/pf_bl/pf_br`, no LDS round-trip). This was previously proposed as the "MXFP8 SGPR-SRD trick" vector. **Mark dead.**
- LDS XOR swizzle, persistent-XCD, NT/cache-stream hints, extra L2 pf, outer-K pf, K-loop sync coarsening, EARLY_SCALE_PF, EARLY_BL_PF, DIRECT_BL → all DEAD on DLA shapes.
- B-tile `__builtin_prefetch` ≡ `emit_one_pf` (`emit_tile_pf` already issues `buffer_load_lds` for the next K-tile; tail of these is exactly what R25-G turns OFF). **Mark dead.**

## 2. Likely remaining LOSE shapes (post-R25 estimate)

DLA1 alone is the only structurally-untouched shape (R25C_K_LIMIT=32768 gates it OUT; R25-E peel was in flight). Other candidates that may still LOSE marginally if R25-G/H pfoff was sub-optimal at the wider extreme:
- **DLA1**: 4096×32768×128256 (K_iters=501, partial-unroll, branch is runtime — R25-C cannot fold)
- **DLA2** rerun: 128256×32768×4096 (gm7+pfoff14 already wired; K=4096 may be over-shooting)
- 4096×128256×32768 (large-N variant, never appears in R25-G's pfoff list)
- 28672×32768×4096 (DLA7 cousin, likely flipped by R25-F wires; verify)
- 4096×4096×32768 (K=32768 but small M/N — never tested with R25-G pfoff124)

## 3. R26 vectors, ranked by expected value

### Vector V1 — DLA1 K-loop "head/peel" with R25E_K_LOOP_PEEL (HIGH PRIORITY)
**Hypothesis**: DLA1's K=128256 → 501 K-iters. The kernel's pragma-unroll fails to fold the R25-C branch, so the tail-PF-off optimization is structurally impossible without splitting the loop. R25-E (in worktree `r25e-kpeel`) wired `R25E_K_LOOP_PEEL` macro that peels last N iters into a no-pf tail. The mechanism is identical to R25-C/D/F/G but expressed as two static loops instead of a runtime `if`. With K_iters≈501 and B-tile reuse pattern from R25-F mechanism (after ~2 iters B is L2-resident), **the entire steady-state pf is dead weight**.
**Test plan**:
- Pull R25-E worktree's macro changes; smoke-test `R25E_K_LOOP_PEEL ∈ {2,4,8,16,32,64,128,250,499}` on DLA1 alone (K_DIM=128256). Each variant ~3 min build + 5-rep smoke.
- Cross with `gm7` and `BARRIER_TO_WAITCNT_ALL=1`.
- If positive, broaden K-EXACT gate so it doesn't bleed into smaller K.
**Cost**: 9 builds × 3 min = 27 min build, 9 × 5-rep × 1.4ms × 700 = ~1 min GPU. Total ~30 min.
**Check vs R25-F/G/H**: Different axis — this is *static loop split*, not a runtime branch. R25-C does NOT fold for K_iters>32, so this is the only way to reach DLA1.
**Expected**: DLA1 88% → ≥97% (matching R25-F gain shape on K=4096; K=128256 has *more* iterations of dead pf).
**P_flip × magnitude / cost ≈ 0.6 × 10pp / 30min = HIGH**.

### Vector V2 — Per-K pfoff for shapes NOT covered by R25-G/H (MEDIUM-HIGH)
**Hypothesis**: R25-G/H covered specific K values, but each entry is gated by `R25C_K_EXACT == K_DIM`. Several still-LOSE shapes may have K values that don't match any K_EXACT, so they fall back to parents that don't carry pfoff. Specifically check:
- 4096×128256×32768 (K=32768) — does it pick up `_ts_lgk2_gm7_memc_pfoff124_kx32768_btw_all` (K_EXACT=32768) at autotune time? **Verify** by reading `bench_all42_results.json` once the wrap-up regression finishes.
- 4096×4096×32768 (K=32768) — likewise
- 16384×4096×7168, 32768×4096×14336 — do they pick up R25-G/H wires? Their (M,N) profile differs from the DLA-test set.
**Test plan**:
- Phase 1 (zero-build): wait for wrap-up regression JSON, grep `best_tag` per shape, identify which post-R25 LOSE shapes did NOT select an R25-G/H variant.
- Phase 2 (1-2 builds): for each such shape, copy the closest R25-G/H entry, change `R25C_K_EXACT` to the shape's K, smoke-test `pfoff ∈ {K_iters-{4,5,6,7,8}}`.
**Cost**: ≤6 builds × 3 min × 5 reps GPU ≈ 25 min.
**Expected**: 1-3 more flips at +10-17pp each.
**P × M / C ≈ 0.5 × 13pp / 25min = HIGH**.

### Vector V3 — `STEP3_PF_N` × shape sweep (MEDIUM)
**Hypothesis**: All R25 work tuned the *tail* prefetch (turning it off after iter K-N). The *steady-state* `STEP3_PF_N`/`STEP4_PF_N` early-iter prefetch depth (default 8) has only been swept at the global level and was found to regress on `1/2`. But the same R25-F mechanism — "B-tile becomes L2-resident after 2 iters" — implies that **STEP3_PF_N=4 or even 2** in steady state may *also* free VMEM bandwidth on K-bound shapes. R25 only tested it at iter-tail granularity; never combined `STEP3_PF_N=4` × `pfoff=K_iters-6` on the *non-tail* iterations.
**Test plan**:
- Smoke 6 variants on DLA2 + DLA7: `(STEP3_PF_N, STEP4_PF_N) ∈ {(8,4), (4,4), (4,2), (2,2), (6,4), (6,2)}` with R25-G stack.
- If positive on DLA2/DLA7, sweep on the per-K wires.
**Cost**: 6 builds × 3 min + 5-rep smoke = ~25 min.
**Risk**: TODO has prior result that PF_N=1/2 globally regressed 2-6%. New twist: combined with R25-G's tail-off; previously they were not stackable.
**Expected**: 1-2pp on DLA2/DLA7 if positive; flat → DEAD if not.
**P × M / C ≈ 0.25 × 2pp / 25min = MED**.

### Vector V4 — `TAIL_BARRIER_VMCNT × shape` sweep (MED)
**Hypothesis**: `TAIL_BARRIER_VMCNT` (kernel L70, defaults to `STEP3_BARRIER_VMCNT=8`) controls the s_waitcnt of the TAIL_SPLIT block. Per-shape SWEEP has never been run with the R25-F/G/H stack engaged. Past sweeps used `tv0/tv16` only (3-point) on the pre-R25 best variants. The R25 wires changed which iters issue VMEM, so the *optimal* TAIL VMCNT may have moved.
**Test plan**:
- Take the 6 R25-G shapes (SA/SB/SC/SD/SE/SF) + 4 R25-H shapes (SH1/SH2/SH4/SH6). For each, smoke `TAIL_BARRIER_VMCNT ∈ {0, 4, 8, 12, 16}` × current best pfoff. 5-pt sweep × 10 shapes = 50 builds total.
- If any +1.5pp+ flip, wire it.
**Cost**: 50 builds × 3 min = 150 min build (parallelizable across 4 GPUs); 50 × 5-rep × 1ms × 700 ≈ 3 min GPU.
**Expected**: 1-2pp on a couple of shapes; many DEADs.
**P × M / C ≈ 0.3 × 1.5pp × 2 shapes / 150min = MED**.

### Vector V5 — MFMA_32X32X64 tiling rewrite (LOW priority, HIGH ceiling)
**Hypothesis**: R5/Optimizer-D (Round 5) said "32×32 still needs 256 AGPRs because per-warp output is 4×64×64" — that's correct for *output* tile, but the K-loop's MFMA-issue-bandwidth and AGPR↔VGPR copy ratio differ. The B-tile broadcast pattern is: 32×32 MFMAs allow 4× more concurrent MFMAs in flight on a single CU (at the same total acc footprint), changing the VMEM/MFMA overlap balance. May relieve VMEM-issue saturation that R24B/C confirmed.
**Test plan**: kernel rewrite, ≥1 week of work. Out of scope for an R26 round; flag as **R27+ candidate**, not for this round.
**Cost**: ≥1 week.
**Expected**: 0-5pp on K-bound deep-LOSE; high uncertainty.
**P × M / C ≈ 0.15 × 4pp / 1wk = LOW** for this round but **only remaining structural axis**.

## 4. Vectors confirmed DEAD/duplicate

- **B-tile `__builtin_prefetch`** — `emit_tile_pf` already does `buffer_load_lds` for B; R25-G specifically *removes* the tail of these. DEAD.
- **scale-load buffer_load with SGPR SRD** — already implemented (kernel L695-L709, NONVOLATILE_SCALE_X2_POC=1, ISA confirms SGPR-SRD `buffer_load_dwordx2` at offset 0). DEAD.
- **SCALE_REG_CACHE round-trip removal** — DUPLICATE confirmed in TODO Round 4. DEAD.
- **WAVES_PER_EU=3** — TODO §284 says `waves_per_eu(2,2)` only +0.4pp on 1 shape, sub-best elsewhere. The R25 wires shifted occupancy via `__launch_bounds__(_NUM_THREADS, 1)` (already 1 wave/SIMD on K-bound). The "wpeu=3 × gm7 unexplored" combination is **theoretically inert** because the launch_bounds clamp dominates. SKIP.
- **per-shape async-prefetch coalescing into `s_waitcnt vmcnt(N)` group** — `emit_pf_tail` is already a tight `#pragma unroll` block; LLVM emits them adjacent and the trailing `s_waitcnt` is at the next consumer. Can verify via ISA dump (already-extant `kernel_mxfp4_gluon_cpp-hip-amdgcn-amd-amdhsa-gfx950.s`, lines 186-228 show clustered `buffer_load_dwordx2`). Likely DEAD.
- **early-iter-only prefetch (inverse of R25-G)** — equivalent to `emit_pf_tail<0>` for late iters with `if (bt < N)` early — same structural shape as R25-C just shifted; given R25-F's "B is L2-resident after iter 2" mechanism, *removing* early pf would force HBM re-fetches that have no cached source. Strong negative prior. SKIP unless V1/V2 dead-end.

## 5. Validation by ISA inspection (light, zero-impact)

Already loaded `kernel_mxfp4_gluon_cpp-hip-amdgcn-amd-amdhsa-gfx950.s`:
- L186-228: confirmed scale `buffer_load_dwordx2` use SGPR-SRD (s[0:3], s[4:7], s[8:11], s[12:15]) — vector "MXFP8 SGPR-SRD trick" is already in place. Cannot wring more.
- L995-1004: same pattern in K-loop body — scales pre-loaded into v[74:77, 106:109]; latency-hidden ≥512 cyc per Round 5 finding.

## 6. Recommended R26 work order (for orchestrator)

1. **Optimizer A → V1 (R25E peel for DLA1)** — pull worktree `r25e-kpeel`, sweep `R25E_K_LOOP_PEEL ∈ {2,4,8,16,32,64,128,250}`, gate `R25C_K_LIMIT_LO=65536`. Single shape, GPU 1.
2. **Reviewer/decider → V2 audit** (zero compile): once wrap-up regression finishes, grep `best_tag` per LOSE shape; produce gap list and propose minimum new K_EXACT entries.
3. **Optimizer B → V3 (`STEP3_PF_N` × R25-G stack)** on DLA2/DLA7. GPU 3.
4. **Optimizer C → V4 (TAIL_BARRIER_VMCNT × shape)** sweep on the 10 R25-G/H shapes. GPUs 4+5 in parallel.
5. **Defer V5** to R27 explicitly.

## 7. KPI / commit thresholds

- V1 success: DLA1 ≥+5pp (any N) → wire and commit
- V2 success: any post-R25 LOSE shape flips to WIN with new K_EXACT entry → wire and commit
- V3/V4 success: ≥+1.5pp on any committed R25 shape with stable std (≤25 TFLOPS) → wire and commit
- All variants must use the standard MXFP4 bench params: warmup=200, iters=500, trim=10%, isolated GPU.

## 8. Constraints reaffirmed

- DO NOT extend pfoff sweeps beyond ±2 of K_iters-6 — exhausted.
- DO NOT touch existing R25-C/D/F/G/H wires.
- DO NOT add cache hints, NT stores, persistent-XCD, EARLY_SCALE_PF — all confirmed DEAD.
- DO NOT touch DIRECT_BL — confirmed structurally inadequate.
- All commits must keep R25 wins intact (regression check via `bench_deep_lose.py` or 42-shape resample).
