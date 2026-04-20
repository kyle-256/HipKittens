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

## Highest-value next axes (R69+ — R66 axis A, R67 cross-product, R68 triple cross-product LANDED)

R66 landed `STEP34_PF_INTERLEAVE` (commit 9b83a0e8): 12/42 → 19/42 WIN, mean 93.6% → 100.4%. New `kpair_64mfma_step34_pf_interleaved` helper inlines all 16 `buffer_load_dwordx4 ... lds` prefetches into the existing single asm block.

R67 landed step34pf×knob cross-product (commit 3cc7f92a): 19/42 → 23/42 WIN, mean 100.4% → 102.5%. Added 5 cross-product variants (gm6/gm8/unr2/we1/tbv16). NEW WINs from `step34pf_unr2`, `step34pf_gm6`, and bare `step34pf`.

R68 landed triple cross-product (commit 830ae4c9): 23/42 → 26/42 WIN, mean 102.5% → 103.2%. Added 6 triple-knob variants on top of step34pf base (gm6_unr2, gm8_we1, gm6_tbv16, unr2_tbv16, gb_gm6, we1_unr2). NEW WINs on 28672×4096×8192 (step34pf_gm6, +3.0pp), 32768×4096×7168 (step34pf_gm8, +3.1pp), and 16384×28672×4096 (step34pf_gm6_tbv16, +5.8pp — triple-only WIN). All 3 confirmed via isolated re-bench (GATE 2). Big LOSE-side uplift on hard losers: 32768×4096×14336 +12.6pp and 4096×28672×32768 +9.8pp.

R68 also landed: (a) orphan dead-code cleanup, -703 LOC (commit fc0e6ef1), and (b) step12pf NON-SPLIT iso harness (commit d0778ba6) — verdict: correctness-SAFE but perf-DOMINATED in the integrated kernel; CLOSED.

R67 step12pf SPLIT design (separate, earlier) FAILED Gate 2 (5632 row-clustered NaN cells, see `project_mxfp4_R67_step12pf_failed.md`). DON'T retry SPLIT; correctness root cause unidentified.

1. **Quadruple cross-product** (R69 high-priority, cheap ~30 min): `step34pf_gm6_tbv16_we1`, `step34pf_gm6_unr2_tbv16`, `step34pf_gb_gm6_unr2`, etc. R68 demonstrated triple-only WIN on 16384×28672×4096; quad may yield another 1-3 WINs.
2. **Untried triple combos** for boundary residue at 95-99%: `step34pf_gm6_we1`, `step34pf_gb_unr2`, `step34pf_gm8_unr2`, `step34pf_we1_tbv16`. Cheap.
3. **Axis A Option 2 — full data-flow flip (Global→VGPR for A)** — only attempt for the 3 hard losers <90%: 14336×4096×32768 (88.1%), 4096×32768×128256 (86.9%), 4096×28672×32768 (92.0% borderline). ~600 LOC. High risk. Note: 4096×28672×32768 jumped 82.2% → 92.0% via triple, may not need axis-A.
4. **step12pf NON-SPLIT closed by dominance** (R68). Don't reopen unless we find a shape cluster step34pf can't reach.
5. **step12pf SPLIT closed by correctness** (R67). Don't reopen without isolated bisect harness.
6. **Triple/quad knob extensions across non-step34pf base helpers** (e.g., `gb_gm6` cluster). Some may have unmined boundary shapes.
7. **Axis B (MFMA 32×32×64)** — defer to R71+. R66-R68 confirmed the 4:1:1 schedule was the lever, not MFMA size.
8. **Axis C (192×256 tile)** — defer; cross-product unlocked enough that motivation is weaker.

## Don't reopen — R65 closures (still valid)
- `KPAIRS_PER_ITER` is not a knob (R65 Opt-D).
- `kpair_64mfma_step34_interleaved` orphan (line 1414-1755): compile-broken. **R66 rewrote the production helper** (`kpair_64mfma_step34_pf_interleaved` at ~line 872) — keep this orphan as historical reference but do NOT try to revive in-place.

## R66/R67/R68 lessons learned
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
R68 also extended the LOSE-side ceiling: hardest losers improved by +9 to +12pp (still LOSE) before any structural change.

The lesson: structural changes UNBLOCK knob axes, and **knob axes themselves stack multiplicatively** — re-sweep at every cross-product depth (single → double → triple → quad) before claiming knob ceiling reached. Don't conclude "knob ceiling reached" without first confirming knobs were re-swept at the next combo depth on top of the latest base.

## Standing user commitments
- Full GitHub push permission (no asking)
- 12-hour autonomous runs via `/shared_nfs/kyle/test/run_mxfp4_agent.py`
- "达标为止" — keep optimizing until all 42 shapes WIN
- Commit any improvement immediately; don't batch
