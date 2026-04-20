# MXFP4 Optimization — Agent Team Prompt

You are the lead agent for an autonomous MXFP4 GEMM optimization session on MI355X (gfx950).

## Setup
- Repo: `/shared_nfs/kyle/test/HipKittens` (branch `mxfp4`)
- Primary kernel: `analysis/fp8_gemm/mi350x/kernel_mxfp4_gluon_cpp.cpp`
- Bench harness: `analysis/fp8_gemm/mi350x/bench_all_42.py`
- Skill: `mxfp8-mxfp4-layout-tuning` (has macro reference, primary files, build commands)
- 8 GPUs (idx 0-7) on this box; check `rocm-smi --showuse` before launching parallel runs

## Workflow
1. **Read first:** `TODO.md` (this directory) + memory at `/root/.claude/projects/-shared-nfs-kyle-test-HipKittens/memory/MEMORY.md`
2. **Spin up an agent team** for each round:
   - 1 **Decider** — reads TODO + memory, picks 1–3 axes, drafts prompts for optimizers
   - 2–4 **Optimizers** — implement in parallel on independent code paths (different macros / different files)
   - 1 **Reviewer** — runs full 42-shape bench, gates on (a) 0 correctness regressions, (b) ≥1 NEW WIN OR ≥+1pp on the worst LOSE cluster
3. **All agents use Claude Opus 4.7** (`claude-opus-4-7`). Do not switch models to save tokens.
4. **No questions to the user.** Make every judgment call independently.
5. **Commit on every measurable WIN** (`git commit` + `git push`). Inert scaffolding stays out of git.
6. **At end of session:** update `TODO.md` and this file with what was tried, what worked, what's closed.

## Hard constraints (CRITICAL)
- **NEVER substitute aiter `.co` binaries** for HipKittens kernel output. The user previously caught R50–R61 doing this and was furious. Every reported number must come from a kernel built from `kernel_mxfp4_gluon_cpp.cpp`.
- Bench protocol: warmup=200, iters=500, trim_frac=0.10 (per `.claude/rules/benchmark-rules.md`).
- Compete against `competitor_tflops` field in `bench_all_42.py` (= aiter ASM measured on this exact MI355X).
- "Win" = `tflops >= competitor_tflops` for that specific shape.
- Don't claim wins from short runs (<200 warmup or <500 iters).

## What's known closed (don't reopen) — see TODO.md for full list
- All cohort-race / fence-positioning experiments (R45B, R47A, R49A, R49C, R50A — 5 closures)
- MFMA↔ds_read 1:3/1:4 interleaving via in-place asm-body rewrite (R50A — ISA-verified but no perf delta)
- R62 `STEP34_INTERLEAVED` — helper function `kpair_64mfma_step34_interleaved` doesn't compile (AGPR pressure on `ds_read_b128`)
- `UNROLL_K={2,16}` per-shape (R62 — added to bench autotune, 0 new wins from variants alone)
- `asm_inline` "5084 TFLOPS" reference REVOKED (numerically incorrect)
- R63 closures: `STEP3_PF_N>8` (static_assert), `STEP3_BARRIER_VMCNT∈{4,12}` (≤0.5pp), `R25C_TAIL_PF_OFF_ITERS` (dead at FUSED=1, crashes at FUSED=0 K=14336), `STEP4_EXTERNAL_BR_PREFETCH` (macro removed), `192×256` tile path (architectural rewrite, defer to R65+)
- R64 closures: `R50A_AITER_INTERLEAVE` (macro removed in a70e4a15 — re-implementing equals the axis-A inner-loop rewrite anyway), `AGPR_REGS_HINT_192` (neutral/weak), `WAVES_PER_EU_1`+`{GM=8,AGPR192}` combos (regress); R64 isolated boundary "WINs" on 4096x4096x8192 / 16384x4096x6144 / 32768x28672x2048 reverted to LOSE in full sweep (contention noise) — confirm any future WIN in BOTH isolated AND full sweep.

## Highest-value next axes (R70+ — R66 axis A, R67/R68 cross-product LANDED, R69 knob ceiling CONFIRMED)

R66 landed `STEP34_PF_INTERLEAVE` (commit 9b83a0e8): 12/42 → 19/42 WIN, mean 93.6% → 100.4%.

R67 landed step34pf×knob cross-product (commit 3cc7f92a): 19/42 → 23/42 WIN, mean 100.4% → 102.5%.

R68 landed triple cross-product (commit 830ae4c9): 23/42 → 26/42 WIN, mean 102.5% → 103.2%. Big LOSE-side uplift on hard losers: 32768×4096×14336 +12.6pp and 4096×28672×32768 +9.8pp.

R69 (no commit, three parallel agents):
- **Opt-A** (quad+triple knob cross-product, 10 new variants): **0 NEW WINs**. Hard losers unchanged. Reverted (no-effect → no-commit). Knob ceiling on the 16 LOSE shapes is now twice-confirmed.
- **Opt-B** (Axis-A Opt-2 read-only scoping): **GO-WITH-CAVEATS**. 17 touchpoints, ~480 LOC, ~10-14h. Saved as `project_mxfp4_R69_axis_a_opt2_scoped.md`. NEW helper `kpair_64mfma_step34_pf_interleaved_globalA` mirrors R66 with 8 ds_reads for A removed, 4 buffer_load_dwordx4 (no `lds`) interleaved in single asm block.
- **Opt-C** (GLOBAL_A VGPR stub-compile gate, isolated worktree): **PASS**. VGPR=253 (under 256 ceiling), SGPR=91, 0 spills, occupancy 1 wave/SIMD. Voff-VGPR-reuse mitigation NOT needed at default occupancy.

**R70 priority: Axis-A Opt-2 (GLOBAL_A) implementation.** Both gates pass: knob axis exhausted + VGPR fits + concrete touchpoints scoped. No remaining decision blockers.

R70 implementation plan (per `project_mxfp4_R69_axis_a_opt2_scoped.md`):
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

- **R71 OPTION A — GLOBAL_A knob cross-product** (`project_mxfp4_R71_globalA_xprod_scoped.md`): replay R67/R68 cross-product over GLOBAL_A base. Top-3 single: `globalA_we1` / `_gm6` / `_tbv16` (we1 promoted because GLOBAL_A pushes VGPR=253). Top-2 doubles: `globalA_gm6_tbv16`, `globalA_we1_gm6`. RULE OUT a-priori: `globalA + gb` (operand pool overflow), `globalA + unr16` (VGPR spill), `globalA + AGPR_REGS_HINT_*` (AGPR hazard). GO trigger: ≥1 NEW WIN at R70. Projected: +1-4 NEW WINs (27-30/42).
- **R71 OPTION B — Hard-loser ISA-diff portable findings** (`project_mxfp4_R71_hardloser_isa_diff.md`): aiter loop body is 231 lines vs HK 726 lines (3.1× longer); aiter spreads bufloads 1-per-8-mfma, HK frontloads. Three actionable items:
  - **F1: LDS-addr swizzle hoist** — eliminate 49 in-loop XOR ops/iter (~80-120 LOC, +1-3pp, **orthogonal to GLOBAL_A**, low risk; under R70 only B-side needs hoisting).
  - **F2: step12 PF interleave** — only safe POST-GLOBAL_A; R67 SPLIT failure was likely A-side LDS race that GLOBAL_A removes (~250 LOC, +2-4pp, synergistic).
  - **F3: Per-shape 128×512 tile template** — only path to closing K=128256 gap (~800-1500 LOC, +3-6pp). Defer to R72+ unless R70 leaves K=128256 <90%.
- **step12pf SPLIT closed by correctness** (R67). Don't reopen without isolated bisect harness (or pair with GLOBAL_A → F2).
- **step12pf NON-SPLIT closed by dominance** (R68). Don't reopen unless we find a shape cluster step34pf can't reach.
- **Axis B (MFMA 32×32×64)** — defer; cross-product confirmed 4:1:1 schedule (and now data-flow) is the lever, not MFMA size.
- **Axis C (192×256 tile)** — defer; F3 (128×512) is more targeted at the actual losers (K=128256 strips).

## Don't reopen — R65 closures (still valid)
- `KPAIRS_PER_ITER` is not a knob (R65 Opt-D).
- `kpair_64mfma_step34_interleaved` orphan (line 1414-1755): compile-broken. **R66 rewrote the production helper** (`kpair_64mfma_step34_pf_interleaved` at ~line 872) — keep this orphan as historical reference but do NOT try to revive in-place.

## R66/R67/R68/R69 lessons learned
- The R65 scoping report (project_mxfp4_R65_axis_a_scoped.md) called the upside "4-8pp mean uplift" — actual was +6.8pp. Scoping reports were accurate; future scoping rounds are worth the cycle.
- Operand pool overflow at 82+48=130 was the predicted blocker; mitigation (4 srds + 4 soffs reused per tile-group, voffs as per-prefetch operand) worked exactly as scoped. R65 BLOCKER analysis was load-bearing.
- **R67: cross-product axis (existing knob × new base) is HIGH-LEVERAGE after a structural change.** R66 introduced step34pf as the new base; R67 simply added 5 macro combos and unlocked 4 NEW WINs in ~30 min.
- **R68: TRIPLE cross-product validated.** Another +3 WINs after R66/R67 already cross-producted single knobs. Largest gains on `step34pf_gm6_tbv16` and `step34pf_gm8` — confirms gm6+tbv16 and gm8 alone are strong on tail-shape clusters. Don't stop at single cross-product; try double, triple, quad.
- **R68 hard-loser uplift**: 32768×4096×14336 +12.6pp and 4096×28672×32768 +9.8pp via triple combos — knob ceiling on hard losers was NOT yet reached after R67. Don't claim "structural rewrite required" until quad cross-product has been tried.
- **R68 step12pf NON-SPLIT closed by dominance** (different from R67 SPLIT closed by correctness): the standalone helper compiles + correct, but in the integrated kernel it is uniformly DOMINATED by step34pf+R68 triples on every benched shape. New finding type: "correctness-SAFE but perf-DOMINATED". Don't reopen unless a shape cluster step34pf can't reach is found.
- **R67 step12pf SPLIT failure**: Gate 1 PASS / Gate 2 FAIL with 5632 row-clustered NaN cells. Reverted entirely; root cause unknown. Lesson: for asm-block restructuring beyond a single-block-rewrite, build an isolated single-iter LDS-byte-comparison test BEFORE full kernel integration.

## Knob ceiling raised by structural change
Boundary shapes (95-99.5%) had been knob-ceilinged at 12/42 after R62-R64.
R66 structural change (step34pf) raised the ceiling to 19/42.
R67 single-knob cross-product on top raised it to 23/42.
R68 triple-knob cross-product on top raised it to 26/42.
R69 quad cross-product on top of triples: **0 NEW WINs** — knob ceiling on the 16 LOSE shapes is now twice-confirmed.
R68 also extended the LOSE-side ceiling: hardest losers improved by +9 to +12pp (still LOSE) before any structural change.

The lesson: structural changes UNBLOCK knob axes, and **knob axes themselves stack multiplicatively** — re-sweep at every cross-product depth (single → double → triple → quad) before claiming knob ceiling reached. R69 stopped at quad (no further yield), so the next round MUST be structural (Axis-A Opt-2 = GLOBAL_A).

## R69 staged-gate pattern (reusable)
When the next round's pivot is hypothetically known but a cheap axis remains untried:
- Decider launches all three in parallel:
  - Opt-A: perf-test the cheap axis (gives definitive "is the cheap axis exhausted?" answer).
  - Opt-B: read-only scoping of the expensive axis (gives "what does the expensive change touch?").
  - Opt-C: compile-only gate of the expensive axis (gives "does it fit the register budget?").
- Cost: ~3 agents × ~1.5h = 4.5h wallclock; payoff: next round can launch with no remaining decision blockers.
- Used in R69; proceed with R70 GLOBAL_A implementation directly.

## Standing user commitments
- Full GitHub push permission (no asking)
- 12-hour autonomous runs via `/shared_nfs/kyle/test/run_mxfp4_agent.py`
- "达标为止" — keep optimizing until all 42 shapes WIN
- Commit any improvement immediately; don't batch
