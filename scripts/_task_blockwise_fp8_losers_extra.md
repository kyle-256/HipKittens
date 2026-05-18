# EXTRA — loser-focused round

The active metric this round is `_metric_blockwise_fp8_loser_shapes.py`,
which scores only the (shape, section) pairs in `LOSER_PAIRS` where HK
runs *below* this-machine Triton. One +30 T win on a loser shape moves
this metric ~8% vs ~0.6% in the full-target metric.

## Live state

- Loser pair set: see `LOSER_PAIRS` in `scripts/_metric_blockwise_fp8_loser_shapes.py`
  (refresh by running the full-target metric and dropping any pair ≥ 105%
  of Triton).
- Per-shape best configs already landed: `kernels/.../blockwise_8192/test_python.py:TUNED_REGISTRY`.
- Exhausted single-knob space (BM/BN/NW/WGM/CHUNK/RAW/PRE etc.) +
  remaining structural attacks: `scripts/_goal_blockwise_fp8.md`.

## Pattern

All losers share at least one of: M ≥ 16384 (big M) and/or K ≥ 4096
(big K).
- **fwd losers (big K)**: per-K SCALE_DRAIN amortization is the bottleneck.
- **wgrad losers**: route through fwd with M_fwd=N (small) and K_fwd=M
  (huge), suffering both low CU coverage and huge per-WG K-loop.

## Per-round guidance

- Each round is ~20 min. For multi-round structural attacks, commit
  PARTIAL progress that compiles + passes correctness on the existing
  8192³ fwd path even if the new path doesn't yet measure better.
- A round that ADDS structural surface (new template overload, new build
  flag, new dispatch path) without lifting the metric is ACCEPTED — tag
  it `feat(blockwise-fp8): round-N — <step> (scaffold)`.
- Pure-measure / pure-analyze rounds (PROBE) are acceptable but should
  be the exception. Most rounds should advance code.
- Before picking an idea, check recent rounds and the FALSIFIED list in
  `_goal_blockwise_fp8.md` to avoid re-trying dead ends.
