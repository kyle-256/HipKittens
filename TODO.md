# MXFP4 Optimization TODO

**Last update:** 2026-04-20 (R67 — step34pf×knob cross-product LANDED, +4 WINs; step12pf SPLIT failed correctness)
**Status:** **23/42 WIN, mean ~102.5%** of aiter. R67 added 5 cross-product variants on top of R66's `STEP34_PF_INTERLEAVE` base.
**Bench harness:** `analysis/fp8_gemm/mi350x/bench_all_42.py` (HipKittens-only, **17 variants**, parallel-GPU)
**Kernel commits:** 9b83a0e8 (R66 helper) + 3cc7f92a (R67 bench cross-product)

---

## Hard rules (do not violate)

1. **No aiter binary substitution.** Every reported number must come from a HipKittens kernel built from `kernel_mxfp4_gluon_cpp.cpp`. Past incident: R50–R61 used aiter `.co` to fake 42/42; user is intolerant of repeats.
2. **Bench protocol:** warmup=200, iters=500, trim_frac=0.10, GPU isolation via `HIP_VISIBLE_DEVICES`.
3. **Compete against `competitor_tflops`** (aiter ASM via Python dispatcher) embedded in `bench_all_42.py`.
4. **Commit only when there is measurable effect.** Inert scaffolding stays out of git.

## Current standing (R67 verified bench, full sweep)

| Cluster | Shapes | Status |
|---|---:|---|
| WIN | 23 | step34pf + 4 cross-product NEW WINs (gm6, unr2) |
| Boundary close (95-99.5%) | 10 | residue after R67 cross-product; knob ceiling on top of step34pf reached |
| Boundary mid (90-95%) | 5 | 14336/32768 N or K — needs structural rewrite (axis A Opt-2 or step12pf) |
| Hard LOSE (<90%) | 4 | 14336x4096x32768, 4096x28672x32768, 4096x32768x128256, 32768x4096x14336 |

R67 NEW WINs (4) on top of R66:
  16384×4096×6144     96.4% → 109.2%   [step34pf_unr2]
  16384×28672×2048    96.1% → 101.4%   [step34pf_gm6]
  32768×28672×2048    99.3% → 106.7%   [step34pf_gm6]
  4096×4096×16384     98.4% → 101.1%   [step34pf]

Closest boundary LOSE residue after R67 (R68 candidates):
  28672×4096×8192     97.8%
  32768×4096×7168     97.6%
  14336×32768×4096    97.4%
  4096×4096×32768     97.4%
  16384×4096×14336    97.2%
  6144×4096×16384     96.5%
  4096×14336×16384    95.7%
  16384×28672×4096    94.3%
  4096×32768×6144     93.7%
  128256×32768×4096   92.6%
  28672×4096×16384    92.4%
  4096×32768×28672    91.8%
  16384×4096×28672    91.8%
  4096×32768×14336    91.6%
  4096×32768×128256   87.0%
  14336×4096×32768    87.2%
  32768×4096×14336    82.5%
  4096×28672×32768    82.2%

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
- **`STEP12_PF_INTERLEAVE` SPLIT design** (R67 Opt-2, 2026-04-20): symmetric extension of R66 4:1:1 trick into `kpair_64mfma_step12` with a0+bl prefetches in step12 and a1+br post-step34. Compiled clean (Gate 1 PASS, SGPR 82, VGPR 240, no spills) but produced 5632 row-clustered NaN cells in correctness test (Gate 2 FAIL). Sampled cells [0,0]/[100,100]/[1024,2048] matched baseline; specific row ranges (13201-13215, 13265-13279, 15252-15255) corrupted. Root cause unidentified within time budget — hypotheses: (a) m0 clobber interaction across step12/step34 boundary, (b) operand-pool/SGPR rename pressure at 94 operands, (c) missed LDS race, (d) STEP3_BARRIER_VMCNT(8) tuned for 16 outstanding bufloads but step12 issued 8 already. Reverted entirely. **Don't re-attempt SPLIT design without first building isolated single-iter test (K=64, PF_DEPTH=1) to bisect the bug**, or try NON-SPLIT design (all 16 prefetches in step12, 0 in step34) to simplify dataflow at cost of doubled operand pool ~150. See `project_mxfp4_R67_step12pf_failed.md`.

---

## R68+ priorities

R67 LANDED step34pf×knob cross-product with +4 WINs (commit 3cc7f92a). step12pf SPLIT design failed Gate 2 correctness (see closures). Remaining 19 LOSE shapes:

1. **`kpair_64mfma_step12` interleave — NON-SPLIT retry** (R68 candidate, untried):
   - SPLIT design (a0+bl in step12, a1+br post-step34) FAILED with row-clustered NaNs. Don't repeat.
   - NON-SPLIT design: put ALL 16 prefetches in step12, 0 in step34 (and remove `emit_pf_tail<0>`). Simpler dataflow (step12 = "issue", step34 = "drain") at cost of operand pool ~150.
   - Pre-work REQUIRED: build isolated K=64 / PF_DEPTH=1 test to bisect bug before kernel integration. See `project_mxfp4_R67_step12pf_failed.md`.

2. **Axis A Option 2 (full data-flow rewrite — Global→VGPR for A tiles)** — for the 4 hard losers (<90%): 14336×4096×32768, 4096×28672×32768, 4096×32768×128256, 32768×4096×14336. ~600 LOC. Risk: VGPR pressure, LDS swizzle re-derivation. Bench cluster shapes specifically before committing.

3. **Orphan dead-code cleanup** (R65 Opt-Orphan audit, ~700 LOC removable) — pure clarity gain, no perf. Optional alongside R68 to reduce future agent-search noise.

4. **Triple cross-product (step34pf × 2 knobs)**: e.g., `step34pf_gm6_unr2`, `step34pf_gm8_we1`, `step34pf_tbv16_unr2`. Some boundary residue (e.g., 32768×4096×7168 at 97.6%, 28672×4096×8192 at 97.8%) may cross with stacked knobs. Cheap; ~30 min round.

5. **Axis B (MFMA 32×32×64)** — still defer to R70+. R66/R67 confirmed the 4:1:1 schedule was the lever, not the MFMA size.

6. **Axis C (192×256 tile)** — defer; cross-product unlocked enough 4096×K-large that the motivation is weaker.

## Bench script (R67 working set)
`bench_all_42.py` variants (17 total):
  `default | gm6 | gb | gb_gm6 | unr2 | unr16 | gm8 | we1 | tbv16 | unr2_gm6 | gb_unr2 | step34pf | step34pf_gm6 | step34pf_gm8 | step34pf_unr2 | step34pf_we1 | step34pf_tbv16`
Run: `BENCH_GPUS=0,1,2,3,4,5,6,7 python3 bench_all_42.py` (~10-15 min for build+full sweep with 17 variants).
Single-shape autotune: `python3 bench_all_42.py M N K` picks best variant.
**Cross-product variants (`step34pf_*`) are autotune-best on 18/42 shapes after R67.**

## Standing GPU/timing protocol
- GPUs 0-7 all available on this MI355X box; check with `rocm-smi --showuse` first.
- Bench result variance: ±2pp on borderline shapes is normal. Boundary WINs (within 1pp of 100%) need re-bench on isolated GPU to confirm.
- The 16384×4096×28672 shape showed a 43% reading in one full sweep that re-benched at 74.5% — interpret single-cell anomalies as contention noise, not regression.
- **R64 lesson**: full-sweep contention systematically depresses borderline TFLOPS by 2-4pp vs isolated runs. A WIN must show in BOTH isolated re-bench AND full sweep before being claimed — see R64 closures for three "WINs" that didn't survive.
