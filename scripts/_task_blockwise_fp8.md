# Task — blockwise FP8 GEMM kernel optimization

You are a kernel optimization collaborator on **HipKittens** (this checkout,
branch `feat/fp8-blockwise-mi300x`). The daemon `scripts/auto_optimize_blockwise_fp8.py`
schedules you for **one focused round** per invocation. After you exit, the daemon
runs the metric and decides whether your change improved the score.

## What you're optimizing

**Kernel**: `kernels/gemm/fp8fp32/mi300x/blockwise_8192/blockwise.cpp`
DeepSeek-V3 1×128 / 128×128 block-scaled FP8 GEMM. Covers fwd (RCR) / dgrad
(routed through fwd via caller B-transpose) / wgrad (routed through fwd via
K-contig col-T inputs + per-element b_scale).

**Target**: 18 production (M,N,K) shapes × 3 sections (fwd/dgrad/wgrad) = 54
configs. See `scripts/_shapes_target.py:SHAPES` and `TRITON_BASELINE`.
HK target per config = `1.25 × THIS-machine Triton`.

**Canonical metric**: `scripts/_metric_blockwise_fp8_target_shapes.py`
Score = `round(mean(min(HK/target, 1.0)) × 1000)`. Single-shape sensitivity ≈ 0.6%.

You may run the metric locally for debugging:
```
METRIC_TRIALS=3 python3 scripts/_metric_blockwise_fp8_target_shapes.py
```
But the **daemon's own run** after you exit is the score that counts.

## Where the headroom is

Most shapes are at or near 100% of target (capped by formula). The remaining
gap concentrates in a handful of LOSER pairs (HK below Triton). For loser-
focused rounds, the daemon uses `_task_blockwise_fp8_losers_extra.md` + the
`_metric_blockwise_fp8_loser_shapes.py` metric (~8× per-pair sensitivity).

See `scripts/_goal_blockwise_fp8.md` for current score, exhausted knobs,
and remaining structural attacks (KBPT=2 interleaved, dedicated wgrad
kernel, persistent kernel, 32×32×16 MFMA).

## Hard rules

1. **One commit per round** in this checkout. ONE hypothesis =
   ONE diff = ONE commit. Negative results also commit
   (`docs(blockwise-fp8): round-N — <X> FALSIFIED`).
2. **FROZEN files** (daemon depends on them; never modify):
   * `scripts/auto_optimize_*.py`
   * `scripts/_task_*.md` (this file)
   * `scripts/_metric_*.py`
   * `scripts/_bench_*.py`
   * `scripts/_shapes_target.py`
3. **Never** `git push` and **never** `git rebase` / `git reset --hard`.
4. **Commit message format**:
   * `perf(blockwise-fp8): round-{N} — {hypothesis} (+X T or +Y score)`
   * `feat(blockwise-fp8): round-{N} — {hypothesis}` (structural change)
   * `docs(blockwise-fp8): round-{N} — {hypothesis} FALSIFIED ({why})`
5. **Correctness gate**: SNR ≥ 48 dB vs fp32 reference. Breaking the gate
   short-circuits the metric to `-2000` (heavy regression).

## Mandatory first steps (every round)

1. Read this file.
2. Read `scripts/_goal_blockwise_fp8.md` — current score, exhausted knobs,
   remaining attacks.
3. Read `scripts/_shapes_target.py` — shape table, TRITON_BASELINE,
   `hk_fp8_unsupported_reason()`.
4. Read `scripts/_metric_blockwise_fp8_target_shapes.py` — score formula.
5. Skim **last 5 rounds** in `auto_optimize_logs/<run>/round_NNN/` (daemon
   gives SHAs in prompt) — `git show <sha>` for actual diffs. Avoid
   re-trying recently FALSIFIED ideas.
6. Read current `kernels/gemm/fp8fp32/mi300x/blockwise_8192/blockwise.cpp`.

Only then decide your hypothesis.

## Per-round budget

~20 min including reasoning + edits + commit. The daemon's metric run after
you exit is another ~5-15 min. Don't burn the budget reading files — focus
on ONE focused change.

If partway through a multi-round structural change, commit a partial step
that compiles + passes correctness even if it doesn't unlock new shapes
yet. Next round picks up.

## What to commit when

* **ACCEPTED** (score went up, OR kernel capability grew even if score
  didn't move yet): commit the code change.
* **FALSIFIED** (tried it, score dropped or unchanged AND you reverted):
  commit a `docs(blockwise-fp8): round-N — <X> FALSIFIED` with a 2-3 line
  body note. The note prevents future rounds from re-trying the same dead
  end. Look at recent FALSIFIED commits before picking ideas.
* **PROBE** (collected diagnostic data, no kernel change): commit a
  `docs(blockwise-fp8): round-N — probe <X>` with the data. Use sparingly.

## Output requirements

Before exiting, print to stdout:

```markdown
## Round {N} summary
- **Hypothesis**: <one line>
- **Mechanism**: <why this should help / which prior round informed this>
- **Files changed**: <list>
- **Decision**: ACCEPT / FALSIFIED / PROBE
- **Local metric (debug)**: <score / per-section breakdown if you ran it>
- **Commit**: <SHA>
- **Next-round suggestion**: <what to try regardless of outcome>
```

Then exit. The daemon takes over.
