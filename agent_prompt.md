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
   SNR + determinism + non-regression.
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
