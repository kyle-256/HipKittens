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

## Highest-value next axes (R64+)
1. **Full inner-loop rewrite** matching aiter's 4:1:1 MFMA:buffer_load:ds_read schedule (not a knob — replace `kpair_64mfma_step34` body wholesale, see `project_mxfp4_aiter_disasm_findings.md`)
2. **MFMA 32×32×64** as a structural alternative to 16×16×128 (untested)
3. **192×256 tile path** for 4096×K-large LOSE cluster — defer to R65+ after axis 1 makes the kpair body template-friendly

## Knob ceiling reached for boundary shapes
After R62+R63+R64, boundary shapes (95-99.5%) have been swept across:
GROUP_SIZE_M ∈ {4,6,8}, UNROLL_K ∈ {2,16}, STEP3_PF_N/STEP4_PF_N ∈ {4,8}, STEP3_BARRIER_VMCNT ∈ {4,8,12}, GLOBAL_B ∈ {0,1}, WAVES_PER_EU_1, TAIL_BARRIER_VMCNT ∈ {8,16}, AGPR_REGS_HINT_192, plus all 2-knob combos thereof.
R63 unlocked 16384×6144×4096 via gm8 (only NEW WIN since R45B). R64 added autotune coverage (mean +0.5pp) but **no NEW WIN** — boundary "WINs" from isolated runs reverted under full-sweep contention. **Knob ceiling at 12/42 confirmed twice.** Path to >12 requires the inner-loop rewrite (axis A).

## Standing user commitments
- Full GitHub push permission (no asking)
- 12-hour autonomous runs via `/shared_nfs/kyle/test/run_mxfp4_agent.py`
- "达标为止" — keep optimizing until all 42 shapes WIN
- Commit any improvement immediately; don't batch
