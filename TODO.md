# MXFP4 Optimization TODO

**Last update:** 2026-04-20 (R65 — scoping round, no kernel changes)
**Status:** 12/42 WIN, mean ~93.6% of aiter. R65 scoped axes A/B/D + orphan audit; produced concrete R66 plan.
**Bench harness:** `analysis/fp8_gemm/mi350x/bench_all_42.py` (HipKittens-only, **11 variants**, parallel-GPU)

---

## Hard rules (do not violate)

1. **No aiter binary substitution.** Every reported number must come from a HipKittens kernel built from `kernel_mxfp4_gluon_cpp.cpp`. Past incident: R50–R61 used aiter `.co` to fake 42/42; user is intolerant of repeats.
2. **Bench protocol:** warmup=200, iters=500, trim_frac=0.10, GPU isolation via `HIP_VISIBLE_DEVICES`.
3. **Compete against `competitor_tflops`** (aiter ASM via Python dispatcher) embedded in `bench_all_42.py`.
4. **Commit only when there is measurable effect.** Inert scaffolding stays out of git.

## Current standing

| Cluster | Shapes | Status |
|---|---:|---|
| Easy WINs (M-large, K-small) | 12 | already winning (incl. 16384×6144×4096 R63 unlock via gm8) |
| Boundary (~95% to ~99.5%) | 6 | needs +0.5pp to +5pp; knob-tuning largely exhausted |
| LOSE: M=4096 K-large | 8 | -7% to -24% gap (root cause: inner-loop scheduling) |
| LOSE: K-large general | 16 | -10% to -25% gap |

Worst losers (4096×*×K, K≥16384): 67–82% of aiter. These dominate the headline gap.

R63 boundary residue (still LOSE):
  16384x14336x4096   95.6%   (no variant tried crosses 100%)
  4096x32768x4096    96.8%   (gm6 best; +0.5pp from R62)
  16384x28672x2048   96.1%   (gm6 best; -0.2pp noise from R62)
  4096x4096x8192     98.8%   (gm8 best; +2.4pp from R62 default)
  16384x4096x6144    98.1%   (unchanged; gb best now)
  32768x28672x2048   99.1%   (gm6 best; -0.3pp noise from R62)

---

## Productive axes (not yet exhausted)

### A. Inner-loop rewrite (HIGH priority — scoped in R65, ready for R66)
**R65 finding** (Opt-A scoping): the current `kpair_64mfma_step34` (line 863-1014) ALREADY emits 4:1 MFMA:ds_read interleaved within its single asm volatile block — comment at line 894 confirms. **What's missing is buffer_load interleaved into the MFMA stream** (true 4:1:1 pattern); currently all 16 prefetches are emitted via `emit_pf_tail<0>` AFTER the fused asm block (lines 2431-2432, 2640-2641, 2652-2653).

Two sub-options identified:
- **Option 1 (recommended for R66): SCHEDULING ONLY.** Extend `kpair_64mfma_step34`'s asm body to inline 16 `buffer_load_dwordx4 ... lds` between MFMA groups, replacing the post-block `emit_pf_tail<0>` calls. ~250 LOC, single helper.
- **Option 2: FULL DATA-FLOW REWRITE** (also flip A-tile data flow Global→VGPR). ~600 LOC. NOT recommended first — empirical data: `GLOBAL_B` (the existing Global→VGPR path for B) only nets ~1 boundary win across 42 shapes (per R64 `gb_unr2`), so data-flow flip alone is not the lever.

R66 touch points (Opt-A enumerated):
- `kernel_mxfp4_gluon_cpp.cpp:863-1014` — rewrite kpair_64mfma_step34
- `kernel_mxfp4_gluon_cpp.cpp:1414-1755` — DELETE broken orphan kpair_64mfma_step34_interleaved
- `kernel_mxfp4_gluon_cpp.cpp:2428-2432, 2438-2440, 2456-2463, 2636-2641, 2649-2653` — gut post-block pf emission, pass pf params into helper
- `bench_all_42.py:88-112` — gate via new STEP34_PF_INTERLEAVE=1 macro for A/B testing

R66 BLOCKERS to anticipate:
1. Inline-asm operand pool overflow: helper has 82 operands today; +16 prefetches needs +32-48 more. May exceed clang/LLVM limit. Workaround: reuse one srd per tile-group (4 srds total).
2. Scale-load contention on K=128256 ultra-loop (DLA1 path). Bench K=128256 shapes early.
3. AGPR-allocator hazard: separate-asm-block design hit `ds_read_b128` AGPR-address bug in R62. Keep all 64 MFMAs in a SINGLE asm block.

### B. MFMA 32×32×64 (DEFERRED — R65 NO-GO verdict)
- ISA support CONFIRMED on gfx950 (`v_mfma_scale_f32_32x32x64_f8f6f4`, cbsz=4 blgp=4, see `/opt/rocm/lib/llvm/include/clang/Basic/BuiltinsAMDGPU.def:451`).
- Cost: rewrite ~1500 lines of asm across 15 kpair helpers + re-derive ds_read interleave + re-tune op_sel.
- Reward: speculative. aiter (the SOTA) chose 16×16×128 explicitly. Switching against the SOTA's MFMA size is bet-against-the-house with no data.
- **Defer to R70+**: only revisit if Axis A plateaus, and then start with a single-helper proof-of-concept (`kpair_32mfma_pure` at line 678).

### C. Per-shape tile dimensions (192×256) — DEFERRED
- Same closure as R63: BLK=256 conflated as both M and N at ~30 sites; ~500-800 LOC parallel asm bodies needed.
- Defer until after Axis A makes the kpair body template-friendly.

### D. K-pair count tuning — RECLASSIFIED (was misleading TODO entry)
- **R65 Opt-D investigation: `KPAIRS_PER_ITER` is NOT a knob.** "KPair" is the name for the fundamental MFMA primitive, not a tunable batching count. The 4-step × 32-MFMA pipeline is the architectural choice (one K-tile produces 4 MN sub-tiles A0Bl/A0Br/A1Bl/A1Br).
- To change "KPair count per iteration" requires rewriting all 4 kpair_64mfma_step* and kpair_32mfma_* helpers (~600 LOC of asm) — i.e., it's not "cheap", it IS axis A.
- Removed from action list. Don't try to add `-DKPAIRS_PER_ITER` to bench autotune.

---

## Closed axes (don't reopen)

- **Fence positioning** in/around step34 (R45B, R47A, R49A, R49C, R50A — 5 closures)
- **MFMA↔ds_read 1:3/1:4 interleaving** inside existing `kpair_64mfma_step34` (R50A — ISA-verified emit but no win)
- **STEP34_INTERLEAVED scaffolding** (R62, 2026-04-20): added flag + 4 call-site gates, but `kpair_64mfma_step34_interleaved` function (already in tree at line 1432) does not compile when actually used — emits `ds_read_b128` with operands that resolve to AGPRs, producing 20+ "invalid operand for instruction" errors. Don't enable until that function is rewritten.
- **`UNROLL_K=2/16` per-shape variants** (R62): added to bench but unlock 0 NEW WINs (only re-rank already-winning shapes by ~0.5pp).
- **`asm_inline` "5084 TFLOPS" reference** — REVOKED 2026-04-17, that kernel is numerically incorrect (SNR -1.31 dB).
- **`STEP3_PF_N` / `STEP4_PF_N` > 8** (R63): static_assert `PF_N <= 2*PF_MPT=8` — pf=12,16 unbuildable.
- **`STEP3_BARRIER_VMCNT` ∈ {4, 12}** (R63): ≤0.5pp delta on boundary shapes; mostly regression. Default 8 stands.
- **`R25C_TAIL_PF_OFF_ITERS` ∈ {1, 2, 4}** (R63): macro is dead at FUSED_STEP34=1 (R63's `_BASE`). At FUSED_STEP34=0 it activates but causes HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION crashes at K=14336. Bug in tail-PF-off path; don't reopen without first fixing the in-flight buffer_load tracking.
- **`STEP4_EXTERNAL_BR_PREFETCH`** (R63): macro doesn't exist anymore (removed in cleanup commit a70e4a15 / 5405bdb4). Stale comment at line 2516.
- **`192×256` tile path** (R63 Opt-3 feasibility): NO-GO for now. Architectural rewrite — `BLK=256` is conflated as both M and N at ~30 sites; 10 `kpair_*` asm functions have 4×4 MFMA grid hard-numbered into operand slots; would need ~500-800 LOC parallel asm bodies. Must follow inner-loop rewrite (axis A) so we don't duplicate code we plan to throw away. `224×256` is strictly less feasible (224 is not a multiple of 64 for scale super-group).
- **`R50A_AITER_INTERLEAVE` macro** (R64): macro was removed in cleanup commit a70e4a15 along with ~250 lines of asm that resurrecting now would be wasted work — the schedule it implemented is the very same one the inner-loop rewrite (axis A) needs to redo from scratch. Don't resurrect; rewrite.
- **`AGPR_REGS_HINT_192` macro** (R64): neutral/weak across all sweeps — never sole best on any shape. Don't enable as default.
- **`WAVES_PER_EU_1`+`GM=8` and `WAVES_PER_EU_1`+`AGPR_REGS_HINT_192` combos** (R64): regress on most shapes; the individual `we1` variant is in autotune but the combos add nothing.
- **R64 isolated boundary "WINs"** (R64): three shapes (4096x4096x8192, 16384x4096x6144, 32768x28672x2048) crossed 100% on isolated re-bench but reverted to LOSE in the full sweep — boundary noise, not algorithmic progress. Don't claim future WINs without ≥3-run isolated confirmation AND a full-sweep confirmation.
- **`KPAIRS_PER_ITER` macro** (R65 Opt-D): does NOT exist as a knob. "KPair" names the fundamental MFMA primitive; the 4-step×32-MFMA pipeline is architectural. Don't try to add `-DKPAIRS_PER_ITER={1,4}` to bench autotune — there's nothing to gate. The TODO entry suggesting this as "cheap to try" was misleading; reclassified as part of axis-A.
- **`kpair_64mfma_step34_interleaved` orphan** (R62 + R65 re-confirmed): function at line 1414-1755 emits `ds_read_b128` with AGPR-address operands due to AGPR pressure in separate-asm-block design. Compile-broken. Useful as STRUCTURAL REFERENCE only — do not try to revive in-place. R66 axis-A rewrite must be from-scratch in a SINGLE asm block (mirror existing `kpair_64mfma_step34` block structure).
- **15 other orphan functions in kernel** (R65 Opt-Orphan, ~700 LOC): `store_bf16x2_packed`, `extract_dsread_tile`, `load_pq_scale_srd`, `compute_lds_base_addrs`, `emit_full_pf_l2only`, `emit_l2_pf_block`, `kpair_32mfma_with_pf`, `kpair_32mfma_pure`, `kpair_32mfma_with_16lds_and_pf`, `kpair_32mfma_with_pf_swapped_sel`, `kpair_32mfma_pure_swapped_plain`, `kpair_64mfma_step12_swapped_sel`, `kpair_32mfma_with_lds_and_pf_swapped_sel`, plus the dead `R37_FIX_B==0` else-branches (lines 2466-2528 and 2654-2680). Pure clarity cleanup, no perf. Optional R66 task.

---

## R66+ priorities

1. **Axis A Option 1 — interleave buffer_loads into kpair_64mfma_step34** (R65 SCOPED, R66 IMPLEMENT). Single helper rewrite, ~250 LOC. See "Axis A" section above for touch points and BLOCKERS. Expected 4-8pp mean uplift, biggest gains on 4096×*×K-large losers (currently 67-82% of aiter).
2. **Orphan dead-code cleanup** (R65 Opt-Orphan audit, ~700 LOC removable) — pure clarity gain, no perf. Optional; do alongside or after R66 to reduce future agent-search noise.
3. **Axis A Option 2 (full data-flow rewrite)** — only if Option 1 lands < 4pp mean uplift.
4. **Axis B (MFMA 32×32×64)** — defer to R70+. Single-helper PoC first, never wholesale.
5. **Axis C (192×256 tile)** — defer until after Axis A makes kpair body template-friendly.

## Bench script (R64 working set)
`bench_all_42.py` variants (11 total):
  `default | gm6 | gb | gb_gm6 | unr2 | unr16 | gm8 | we1 | tbv16 | unr2_gm6 | gb_unr2`
Run: `BENCH_GPUS=0,1,2,3,4,5,6,7 python3 bench_all_42.py` (~10 min for build+full sweep).
Single-shape autotune: `python3 bench_all_42.py M N K` picks best variant.

## Standing GPU/timing protocol
- GPUs 0-7 all available on this MI355X box; check with `rocm-smi --showuse` first.
- Bench result variance: ±2pp on borderline shapes is normal. Boundary WINs (within 1pp of 100%) need re-bench on isolated GPU to confirm.
- The 16384×4096×28672 shape showed a 43% reading in one full sweep that re-benched at 74.5% — interpret single-cell anomalies as contention noise, not regression.
- **R64 lesson**: full-sweep contention systematically depresses borderline TFLOPS by 2-4pp vs isolated runs. A WIN must show in BOTH isolated re-bench AND full sweep before being claimed — see R64 closures for three "WINs" that didn't survive.
