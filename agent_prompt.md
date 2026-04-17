# Agent Prompt — FP8 / BF16 GEMM on MI350X

Use this document when spawning subagents to continue the FP8/BF16 work.
Read the related skills under `.cursor/skills/` first:
- `cdna4-gemm-kernel-design/SKILL.md` (architecture fundamentals)
- `fp8-strict-layout-tuning/SKILL.md` (FP8 rules + known findings)
- `fp8-rcr-autotune-optimization/SKILL.md` (FP8 current state + next steps)
- `bf16-gemm-optimization/SKILL.md` (BF16 current state + next steps)

## Agent Team Structure

Use this team structure for non-trivial optimization tasks.

### Roles

1. **Decider (主 agent)** — plans work, assigns subagents, reviews results,
   merges changes, writes commits, updates TODO and skill docs. Does not
   itself do heavy kernel edits.

2. **Dev (2-4 subagents)** — each owns one optimization direction (a
   specific layout × weak-shape family, or a specific idea like M↔N swap).
   Each runs in isolation with its own GPU.

3. **Reviewer / QA (1 subagent)** — after all Dev changes merge, runs the
   full 48-shape benchmark, the SNR/determinism gate, and confirms
   non-regression of the strong layouts.

### GPU Assignment

Hardware has 8 MI350X GPUs (GPU 0-7). Assign one GPU per agent:
- FP8 Dev agents: GPU 0, 2, 3
- BF16 Dev agents: GPU 1, 4, 5
- Reviewer: any idle GPU

Never have two agents share a GPU — hipcc invocations overwrite the shared
`.so` binary and invalidate each other's measurements.

## Prompt Template for Dev Subagents

Every Dev subagent prompt MUST include:

### 1. Hard constraints

- **NO JIT** — single `.so` per target. No per-shape `-DM_DIM` compile flags.
- **NO git commits** — leave changes in working tree; Decider commits.
- **GPU isolation** — use only the assigned `HIP_VISIBLE_DEVICES` value.
- **SNR ≥ 48 dB**, **bit-exact determinism** across 3+ runs.
- **Non-regression gates** — cite the current numbers and the minimum
  acceptable numbers for the layouts not being changed.

### 2. Context

- Repo path, branch name.
- Working directory.
- Current measured baseline (print actual TFLOPS, not just ratios).
- List of the specific weak shapes being targeted.
- Which files are the single source, which files are OK to delete.

### 3. Concrete strategies, ranked

Prefer 3-5 ranked ideas over one open-ended task. Each idea should include:
- Expected outcome
- Files to change
- A smoke test + full test that confirms success

### 4. Deliverables

- Exact output format: benchmark JSON path, final summary format.
- "Do NOT commit" in bold.
- Print summary fields: geo_means, top 5 wins, remaining weak shapes,
  VGPRs/occupancy of any new kernel variants.

## Decider Checklist (before commit)

After subagents return:

1. `git status` — list modified + deleted + untracked.
2. Delete transient files: `bench_*.log`, `build_*.log`, `probe_*.py`,
   any `.so` binary, any `bench_v*_baseline.json` (keep only one
   "final" json per directory).
3. Run the Reviewer agent: re-benchmark the modified directory, confirm
   SNR + determinism + non-regression. **The Reviewer MUST use a
   different `HIP_VISIBLE_DEVICES` than the Dev** who claimed the win,
   to rule out per-GPU DVFS thermal artifacts. P9 demonstrated that
   the same code can show +1.2pp on one GPU and -0.88pp on another
   when clocks are not actually pinned (and `rocm-smi --setperflevel high`
   is silently broken on this host).
4. Ensure the strong layouts didn't regress (FP8 RRR ≥ 1.4x, CRR ≥ 1.8x;
   BF16 keep whatever baseline was).
5. Update `TODO.md` with new state and remaining items.
6. Update relevant skill under `.cursor/skills/` — **no more than what's
   needed to reflect reality**. Delete stale guidance.
7. Stage changes explicitly, use a HEREDOC commit message.
8. Never add `--no-verify` or `-i`.

## Example Agent Task (copy-paste starting point)

```
You are <role> on HipKittens <FP8|BF16> GEMM for AMD MI350X.

Environment:
- Repo: /workspace/code/Hipkittens_per_tensor on branch <branch>
- Working dir: analysis/<fp8_gemm|bf16_gemm>/mi350x
- GPU: use HIP_VISIBLE_DEVICES=<N> only

Baseline (measured <date>):
- RCR geo_mean=X.XXXx ; RRR=X.XXXx ; CRR=X.XXXx

Your target: <specific layout / shape family / number>

STRICT CONSTRAINTS:
- NO JIT per-shape compilation; single .so only.
- NO git commits.
- SNR ≥ 48 dB; bit-exact determinism.
- Don't regress RRR (must stay ≥ X.XXx) or CRR (≥ X.XXx).

Strategies in priority order:
  A. <idea> — <expected +/- %>
  B. <idea>
  C. <idea>

Deliverables:
1. Modified files list
2. Final benchmark JSON at <path>
3. Print FINAL SUMMARY: geo_means, wins, top 5 shape improvements,
   remaining weak shapes, VGPRs.
4. DO NOT commit.

Tips:
- <hint about primus_turbo import>
- <hint about cleaning stale .so>
- <hint about watching VGPR remarks>
```

## Session Logging

Every Decider session should append a one-line dated entry to `TODO.md`
"Closed / Completed" section when a target is hit, and update the
"Current Status" table numbers.

Every skill update should reflect only what was verified in the current
session, not aspirational targets.

## Session Log

### 2026-04-17 — P9 (3 Devs + 1 Reviewer in worktrees, all model=opus)

**Outcome: nothing landed. All three optimization directions bottomed out
at DVFS noise after cross-GPU validation.**

- **BF16 CRR Dev (worktree `agent-aa1be9f2`, GPU 4)** — KI=296
  `#pragma unroll 1` specialization. Build log confirms SGPR spills
  26 → 0 at KI=296. On Dev's GPU 4: +1.2pp on (8192,3584,18944).
  Reviewer on GPU 2: SAME shape, SAME code → -0.88pp (regression).
  Geo-mean across the layout was -0.28pp on CRR. The unroll-2 → unroll-1
  trade reduces spills but loses barrier-hiding; net negative for CRR
  on this kernel. Cosmetic `readfirstlane` hoist of `row*2/col*2` had
  zero effect on spill count (compiler already factored it). Diff
  preserved in worktree, NOT landed.
- **BF16 RCR/RRR Dev (worktree `team-bf16-rcrrrr-mn`, GPU 5)** — per-shape
  `vmcnt`/`lgkmcnt` autotune via 4-profile `WAITCNT_PROFILE` template
  arg. +0.25pp / +0.17pp consistent across 3 runs but inside the
  calibrated DVFS noise band (per-shape stdev 0.66pp from a CRR
  identical-code calibration). Bloats the .so by ~7×. Diff preserved,
  not landed.
- **FP8 RCR Dev (worktree `agent-a693b720`, GPU 0)** — re-bench of the
  per-shape NUM_XCDS strategy with `--warmup 30 --iters 100 --trials 5`
  on every weak shape. Result: xcd=8 wins on every weak shape by
  0.1-2.5%; the P8 xcd=16 "wins" were thermal noise. Closed item;
  no diff.
- **Decider** — applied Dev 1's diff to main worktree, ran Reviewer
  agent on GPU 2 (different from Dev 1's GPU 4 to rule out thermals),
  Reviewer rejected, working tree reverted. Updated TODO.md +
  agent_prompt.md to record findings. Committed docs only as P9.

**Lessons (additive to P8 lessons):**

1. **`rocm-smi --setperflevel high` is silently broken on this host.**
   Both with and without sudo it returns success but perf level stays
   "auto". Confirmed across P8 + P9. Without clock pinning, ±2pp DVFS
   noise dominates any single-knob effect.
2. **Always cross-validate on a 2nd GPU before committing**, even if
   the change shows a real ISA-level effect (like SGPR spill count
   dropping). Dev 1's KI=296 unroll-1 change ABSOLUTELY dropped spills
   from 26 to 0 — but the wall-clock effect on the target shape was
   *opposite* on a different GPU. The mechanism was real; the
   interpretation was wrong.
3. **SGPR spill count is a means, not an end.** Reducing spills can
   regress perf if the trade (e.g. losing #pragma unroll for spill
   reduction) costs more in barrier-hiding than it saves in load
   pressure. Always measure the wall clock.
4. **+0.25pp consistent across 3 runs is still noise** when the
   calibrated per-shape stdev is 0.66pp. Always calibrate the noise
   floor with an identical-code A/B run before claiming a win in the
   sub-1pp band.
5. **Reviewer must use a different GPU than the Dev** who claimed the
   win. Codifying this in the Decider Checklist now (see step 3).

### 2026-04-17 — P8 (3 Devs in worktrees, all model=opus)
- **FP8 Dev (a67f50ee, GPU 0)** — implemented runtime `g.num_xcds` end-to-end
  in `kernel_fp8_layouts.cpp` + autotune two-phase + bench passthrough.
  4 RCR shapes prefer xcd=16 with measurable wins, but per-shape noise on
  the other 44 cancels at the geo-mean. Mechanism is sound; **not landed**
  pending more aggressive coverage. Diff preserved in worktree.
- **BF16 CRR Dev (a7cbd1fb, GPU 4)** — exhaustive sweep of CRR-only knobs
  (CRR_MAIN_VMCNT/LGKMCNT/UNROLL/NUM_XCDS/CHUNK). All within ±2pp DVFS
  noise band. Reverted. Recommendation for next iteration: pin GPU clocks
  before sweeping; root cause of CRR gap is SGPR spill on KI=128/296.
- **BF16 RCR/RRR Dev (ac9f516a, GPU 5)** — applied same NUM_XCDS-as-runtime
  strategy that FP8 dev tried; **on BF16 it works**: RCR +1.0pp, RRR +1.6pp,
  CRR +1.7pp at the geo-mean, no regressions, SNR/det pass. Landed in P8.
- **Decider** — verified on GPU2/3 in main worktree, committed P8 with both
  the BF16 NUM_XCDS infra and the FP8 MID=6 correction.

Lessons:
- A strategy that's neutral on one kernel can win on another (NUM_XCDS
  was geo-mean-neutral on FP8 because FP8 RCR autotune already pushes
  the strong shapes hard; BF16 had more slack).
- Always pin GPU clocks before doing CRR-class sweeps (the 2pp band
  swallowed several real attempts).
- Sub-agents must `git fetch && git reset --hard <branch>` first if their
  worktree was created from a different branch — `main` doesn't have the
  bf16/fp8 dirs.
