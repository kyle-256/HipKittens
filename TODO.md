# MXFP4 Optimization TODO

**Last update:** 2026-04-21 (R70 GLOBAL_A DEAD; F1 NO-OP; F3 128×512 VGPR-spill; 8-wave ping-pong < 4-wave; UNROLL_K=4 landed; exhaustive knob sweep done)
**Status:** **26/42 WIN, mean ~103.2%** of aiter. 1 shape < 90% (14336×4096×32768 = 88.5%, knob ceiling). K=128256 actual = 92.5% isolated (89.2% in contended sweep = noise).
**Bench harness:** `analysis/fp8_gemm/mi350x/bench_all_42.py` (HipKittens-only, **27 variants** incl unr4, parallel-GPU)
**Kernel commits:** 9b83a0e8 (R66) + 3cc7f92a (R67) + fc0e6ef1 (R68 cleanup) + d0778ba6 (R68 step12pf) + 830ae4c9 (R68 triple) + d76ad328 (R70 unr4 bench variants).

---

## Hard rules (do not violate)

1. **No aiter binary substitution.** Every reported number must come from a HipKittens kernel built from `kernel_mxfp4_gluon_cpp.cpp`. Past incident: R50–R61 used aiter `.co` to fake 42/42; user is intolerant of repeats.
2. **Bench protocol:** warmup=200, iters=500, trim_frac=0.10, GPU isolation via `HIP_VISIBLE_DEVICES`.
3. **Compete against `competitor_tflops`** (aiter ASM via Python dispatcher) embedded in `bench_all_42.py`.
4. **Commit only when there is measurable effect.** Inert scaffolding stays out of git.

## Current standing (R68 verified bench, full sweep)

| Cluster | Shapes | Status |
|---|---:|---|
| WIN | 26 | step34pf + R67 single-knob cross-product + R68 triple-knob cross-product NEW WINs (gm6, gm8, gm6_tbv16) |
| Boundary close (95-99.5%) | ~10 | residue after R68 triple cross-product; gm6_we1 / gb_unr2 / gm8_unr2 untried (R69 candidates) |
| Boundary mid (90-95%) | ~3 | 32768/14336 K — needs structural rewrite (axis A Opt-2) or quad cross-product |
| Hard LOSE (<90%) | 3 | 14336×4096×32768 (88.1%), 4096×32768×128256 (86.9%), 4096×28672×32768 (92.0% borderline) |

R68 NEW WINs (3) on top of R67 (all confirmed in isolated re-bench, GATE 2):
  28672×4096×8192     97.8% → 100.8%   [step34pf_gm6]            (iso 100.9%)
  32768×4096×7168     97.6% → 100.7%   [step34pf_gm8]            (iso 101.7% on step34pf_gm6_tbv16)
  16384×28672×4096    94.3% → 100.1%   [step34pf_gm6_tbv16]      (iso 101.3%, triple-only WIN)

R68 big LOSE uplifts (no WIN flip, but knob-ceiling extension demonstrated on hard losers):
  32768×4096×14336    82.5% → 95.1%   (+12.6pp) [step34pf_gm6_tbv16]
  4096×28672×32768    82.2% → 92.0%   (+9.8pp)  [step34pf_unr2_tbv16]

Closest boundary LOSE residue after R68 (R69 candidates):
  14336×32768×4096    ~97.4%   (re-confirm with quad)
  4096×4096×32768     ~97.4%
  16384×4096×14336    ~97.2%
  6144×4096×16384     ~96.5%
  4096×14336×16384    ~95.7%
  4096×32768×6144     ~93.7%
  128256×32768×4096   ~92.6%
  28672×4096×16384    ~92.4%
  16384×4096×28672    ~91.8%
  4096×32768×28672    ~91.8%
  4096×32768×14336    ~91.6%
  4096×32768×128256   ~86.9%
  14336×4096×32768    ~88.1%

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

## R69 closure summary

- **Opt-A (quad+triple cross-product, 10 new variants on GPUs 0-3)**: 26/42 WIN, mean 103.2%. **0 NEW WINs vs R68.** Hard losers unchanged (14336×4096×32768 88.3%, 4096×32768×128256 86.9%, 4096×28672×32768 92.0%). Bench changes REVERTED (no measurable effect → no commit per rule). 5 of 10 new variants displaced incumbents on individual shapes by ≤1pp but did not move WIN/LOSE landscape. **Knob ceiling confirmed reached on the 16 LOSE shapes.**
- **Opt-B (Axis-A Opt-2 read-only scoping)**: GO-WITH-CAVEATS. 17 touchpoints across `kernel_mxfp4_gluon_cpp.cpp`, ~480 LOC, ~10-14h. NEW helper `kpair_64mfma_step34_pf_interleaved_globalA` mirrors R66 with 8 ds_reads for A removed, 4 buffer_load_dwordx4 (no `lds`) interleaved in single asm block. Top blocker B3b: A-side raw-row mapping (~80 LOC) since A is not preshuffled in pipeline. Saved as `project_mxfp4_R69_axis_a_opt2_scoped.md`.
- **Opt-C (GLOBAL_A VGPR stub-compile gate, isolated worktree)**: **PASS.** Stub helper compiled clean. Compile remarks: VGPR=253 (under 256 ceiling), SGPR=91, AGPR=256, 0 spills, occupancy 1 wave/SIMD, LDS=128 KB. Voff-VGPR-reuse mitigation NOT needed at default occupancy. Forced `we1` NOT needed. Worktree throwaway.

## R70 priorities — RETRY plan (multi-agent)

**R70 first attempt** (single-agent monolithic): NO COMMIT. Identified `compute_a_global_load_voffs` (B3b raw-row mapping) as session-scope blocker — must produce 8 per-lane voffs that yield bytes a `ds_read_b128` would have produced from swizzled-LDS-loaded `A_row_reg`. No twin in codebase; authoring this in one shot has same failure profile as R67 SPLIT (5632 NaN cells, root cause unidentified). Macro+bench-variant scaffolding compiled clean (default WIN preserved at 109.9%) and was reverted as inert. See `project_mxfp4_R70_globalA_session_findings.md`.

**R70 RETRY: split into 3 agents (per session findings recommendation):**
- **Opt-A (B3b bisect harness)** — standalone ~150 LOC `.cpp` that loads one A half-tile via existing LDS path → reference `fp4_intx8_t[4]` per lane, then issues per-lane `buffer_load_dwordx4` (no `lds`) using candidate `compute_a_global_load_voffs` and bytewise-compares ALL 256 threads. Bisect by lane-group / (row, k-phase) / (warp_m, warp_n) until correct. Output: verified mapping function + pass log. ~3-4h. Worktree: `agent-a8d66e1e`.
- **Opt-B (diff scoping)** — read-only — produce EXACT line-by-line diff between `kpair_64mfma_step34_pf_interleaved` (L1004-1203) and proposed `_globalA` variant: which 8 ds_read constraints get replaced, operand-pool delta (122→?), AGPR-block check, srd/soff reuse plan. Output: precise patch outline. ~1-2h. Worktree: `agent-ad42c716`.
- **Opt-C (integration)** — sequential AFTER A+B: apply mapping (from A) to patch outline (from B), preflight `GLOBAL_A=0` byte-identical check, K=128256 isolated bench, full sweep, commit-gate. ~3-4h.
- Total ~7-10h split across 3 agents (matches R69 Opt-B prediction of 10-14h).

R70 implementation plan touchpoints (per `project_mxfp4_R69_axis_a_opt2_scoped.md`, still valid):
1. Macro `GLOBAL_A` near `kernel_mxfp4_gluon_cpp.cpp:99-101` (default 0).
2. New helpers `load_a_global_8` + `compute_a_global_load_voffs` near `:287` (B3b raw-row mapping, ~80 LOC).
3. Driver gates at `:1574-1779` and `:1813-2073` to bypass A LDS path under `#if GLOBAL_A`.
4. NEW variant `kpair_64mfma_step34_pf_interleaved_globalA` paralleling `:1004-1203` (~250 LOC, single asm volatile block).
5. Bench `step34pf_globalA = -DSTEP34_PF_INTERLEAVE=1 -DGLOBAL_A=1` (one variant only, NO knob cross-product in PoC).
6. Preflight: build with `GLOBAL_A=0`, diff `.s` against current main → must be byte-identical.
7. Bench K=128256 first in isolation (R66 cadence).
8. Use 10-run @ 80% cohort-race protocol (FINITE_GATE 0.97).
9. Commit only on ≥+1 NEW WIN AND no regression on the 19 R66 winners.

R71+ pre-scoped (R69 staged-gate pattern, R71 read-only scoping landed in R70 window):
- **R71 OPTION A — Knob cross-product OVER GLOBAL_A** (`project_mxfp4_R71_globalA_xprod_scoped.md`): top-3 single-knobs are `globalA_we1` / `_gm6` / `_tbv16`. **`we1` is promoted to highest-EV** because GLOBAL_A pushes VGPR to ~252/256 — `we1` is the safety net that unblocks the rest. Top-2 doubles: `globalA_gm6_tbv16` and `globalA_we1_gm6`. **RULE OUT: `globalA + gb` (operand pool 146→170, infeasible), `globalA + unr16` (VGPR spill), `globalA + AGPR_REGS_HINT_*` (R62/R65/R66 AGPR hazard). GO trigger: ≥1 NEW WIN at R70 + no R66-winner regression. Projected: +1-4 NEW WINs (27-30/42).**
- **R71 OPTION B — Hard-loser ISA-diff findings** (`project_mxfp4_R71_hardloser_isa_diff.md`): aiter loop body 231 lines vs HK 726 lines (3.1× longer). Three portable findings:
  - ~~**F1: Hoist LDS-addr swizzle out of K-loop**~~ **CLOSED — NO-OP** (`project_mxfp4_R71_F1_no_op.md`). Built `.s` for K=32768 + K=128256 + multiple variants → **0 in-loop XOR-swizzle ops** in K-loop body. `compute_lds_base_addrs<>` (L1699-1708) already pre-computes; per-iter is just 8 v_cndmask + 4 s_cselect + 4 s_add (~<2% of cycles). The R71 ISA-diff memo's "49 ops/iter" was a counting error (likely C-store epilogue mis-classified as in-loop). DO NOT relaunch.
  - **F2: step12 PF interleave** — DEEP-SCOPED (`project_mxfp4_R72_F2_step12_pf_globalA_scoped.md`). **CRITICAL FINDING: R68 already landed dormant F2 infrastructure** (commit d0778ba6): NON-SPLIT helper `kpair_64mfma_step12_pf_interleaved` exists at `kernel_mxfp4_gluon_cpp.cpp:644-840` (197 LOC, 0 spills, VGPR 240, 114 operands), macro guard at L640-642, driver gates at L1996-2006/L2017-2034/L2163-2172/L2183-2201. R72 F2 = **MODIFY** existing dormant code, not write from scratch. Recommended: NON-SPLIT (preserves m0-block discipline). Operand 114→108 (drops 8 A-side lds_addrs, fits R66's 122 ceiling). VGPR ~330 at we1 → **forces WAVES_PER_EU=1**. R67 SPLIT NaN root cause likely H3 (A-side LDS race) — GLOBAL_A structurally REMOVES that. New fragility H5 (B-side race) is bisect-harness gated. LOC ~310 total (Phase 1 kernel + bench variant + harness + Phase 2 xprod). Macro design: `#error STEP12_PF_INTERLEAVE && !GLOBAL_A` enforces dependency at compile time. **GO trigger: R70 GLOBAL_A landed ≥1 WIN AND hard losers still <95%. F2 is dead-on-arrival without GLOBAL_A.**
  - **F3: Per-shape 128×512 tile template** — DEEP-SCOPED (`project_mxfp4_R72_F3_per_shape_tile_scoped.md`). LOC refined to ~1340 (BLK templating 70 + 3 NEW asm helpers 1040 + K-loop dispatch 120 + LDS mitigation 60 + bench infra 30). **Helps only 5 strip shapes** (`4096×{32768×128256, 32768×28672, 32768×14336, 28672×32768, 128256×32768}`); the 14336×4096×32768 and 128256×32768×4096 hard losers stay 256×256 and need F1+GLOBAL_A instead. **VGPR=480 (fits 2-wave band).** **LDS=160KB exactly at gfx950 cap** — tight; mitigation = single-buffered B (96KB) risks -5-10pp on B-bound shapes. 4 NO-GO triggers (VGPR>256 we1 / opspool>200 / LDS>160KB / first-helper smoke fail at M=N=K=4096). **GO trigger: R70 LANDED + (R70+F1 leaves 4096×32768×128256 < 95%) + 4096×28672×32768 < 95%; need ≥2 NEW WINs OR ≥0.5pp mean uplift to ship.**
- **`kpair_64mfma_step12` interleave SPLIT** (R67 closure): row-clustered NaN, don't reopen without isolated bisect harness.
- **Axis B (MFMA 32×32×64)** — still defer; cross-product confirms 4:1:1 schedule (and now data-flow) is the lever, not MFMA size.
- **Axis C (192×256 tile)** — defer; F3 (128×512) is more targeted at the actual losers (K=128256 strips).

## Bench script (R68 working set)
`bench_all_42.py` variants (23 total):
  `default | gm6 | gb | gb_gm6 | unr2 | unr16 | gm8 | we1 | tbv16 | unr2_gm6 | gb_unr2 | step34pf | step34pf_gm6 | step34pf_gm8 | step34pf_unr2 | step34pf_we1 | step34pf_tbv16 | step34pf_gm6_unr2 | step34pf_gm8_we1 | step34pf_gm6_tbv16 | step34pf_unr2_tbv16 | step34pf_gb_gm6 | step34pf_we1_unr2`
Run: `BENCH_GPUS=0,1,2,3,4,5,6,7 python3 bench_all_42.py` (~12-18 min for build+full sweep with 23 variants).
Single-shape autotune: `python3 bench_all_42.py M N K` picks best variant.
Optional env: `BENCH_BUILD_DIR` / `BENCH_WORK_DIR` to isolate parallel runs from the shared `build_all42`/`work_all42` dirs.
**Cross-product variants (`step34pf_*`) are autotune-best on 21+/42 shapes after R68.**

## Standing GPU/timing protocol
- GPUs 0-7 all available on this MI355X box; check with `rocm-smi --showuse` first.
- Bench result variance: ±2pp on borderline shapes is normal. Boundary WINs (within 1pp of 100%) need re-bench on isolated GPU to confirm.
- The 16384×4096×28672 shape showed a 43% reading in one full sweep that re-benched at 74.5% — interpret single-cell anomalies as contention noise, not regression.
- **R64 lesson**: full-sweep contention systematically depresses borderline TFLOPS by 2-4pp vs isolated runs. A WIN must show in BOTH isolated re-bench AND full sweep before being claimed — see R64 closures for three "WINs" that didn't survive.

## R66/R67/R68/R69 lessons learned

- **R66 / R67**: structural change (step34pf) unblocked 7 WINs; cross-product on top added 4 more. Always re-sweep knob axes across new structural axes — don't assume previous knob ranking holds.
- **R68 lesson 1 (TRIPLE CROSS-PRODUCT VALIDATED)**: another +3 WINs after R66/R67 cross-producted single knobs. Largest gains on `step34pf_gm6_tbv16` and `step34pf_gm8`. Confirms gm6+tbv16 and gm8 alone are strong on tail-shape clusters.
- **R68 lesson 2 (HARD-LOSER UPLIFT)**: 32768×4096×14336 jumped 82.5% → 95.1% (+12.6pp) and 4096×28672×32768 jumped 82.2% → 92.0% (+9.8pp) via triple combos — knob ceiling on hard losers was NOT yet reached after R67. Don't claim "structural rewrite required" until quad cross-product has been tried.
- **R68 lesson 3 (step12pf NON-SPLIT closed by dominance)**: the iso harness (commit d0778ba6) proved the standalone helper is correctness-SAFE, but in the integrated kernel it is uniformly DOMINATED by step34pf+R68 triples. Different from R67 SPLIT (correctness-unsafe). Don't reopen unless we find a shape cluster step34pf can't reach.
- **R68 lesson 4 (orphan cleanup, -703 LOC)**: removed the 15 orphan helpers from R65 audit (commit fc0e6ef1). Pure clarity gain; reduces future agent-search noise.
- **R69 lesson 1 (KNOB CEILING NOW REACHED)**: quad+triple cross-product (10 new variants on step34pf base) delivered exactly 0 NEW WINs over R68 baseline, mean unchanged at 103.2%. R68's lesson 2 was about triples-after-singles; R69 inverts it: quads-after-triples bring NOTHING. Knob ceiling that R65/R67/R68 dismissed is now twice-confirmed. **Only structural data-flow change can move WINs from here.**
- **R69 lesson 2 (STAGED GATE PATTERN)**: Decider launched Opt-A (perf-test the cheap axis), Opt-B (read-only scope the expensive axis), Opt-C (compile-only gate the expensive axis) IN PARALLEL. Result: by the time Opt-A returned with "no effect", Opt-B/C had already cleared the next round's biggest unknowns. Cost: ~3 agents × ~1.5h = 4.5h wallclock; payoff: R70 can launch with no remaining decision blockers. Pattern reusable when next round's pivot is hypothetically known.
- **R69 lesson 3 (`-effect → -commit`)**: Opt-A's bench changes added +22 LOC, +40% bench time, and 0 measurable WIN delta. Reverted. Stick to "有效果就进行commit" even when the work was non-trivial; an inert diff is still inert.
