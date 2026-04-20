# R59 Cohort J-3 (Opt V) Verdict — L1 noise-edge re-attack alt-tile probe

**Date:** 2026-04-20
**Worker:** C
**GPUs used:** 5, 6 (verified idle pre-launch; released post-bench)
**Wall-clock:** ~50 s SMOKE pair (parallel, 5 seeds × ~9 s/seed)

---

## Summary

| Cell | Tile | aiter eff | SMOKE pct_comp | Decision | Verdict |
|---|---|---:|---:|---|---|
| V-1 | 96×640 | 83.5 | **87.42%** | STOP_DEAD | **DEAD** |
| V-2 | 64×1024 | 60.2 | **66.05%** | STOP_DEAD | **DEAD** |

**Cohort outcome:** 0/2 PROMOTE / 0 ACCEPT_FALLBACK / **2 DEAD**.

L1 baseline (R57J1_L1_AITER 256×256, ITERS=500) = **99.94%** pct_comp.
- D-3A-1 SMOKE PROMOTE gate = > 100.94% (current + 1.0pp)
- STOP_DEAD threshold = < 94.94% (current - 5.0pp)

**V-1 perf delta = −12.52pp; V-2 perf delta = −33.89pp.** Both far below STOP_DEAD threshold; no 10-run escalation per STOP rule.

---

## Correctness

Both candidates clean across 5 seeds:
- n_OK = 5/5
- fin_min = 1.0 (all elements finite)
- wcf_max = 0.0 (bit-deterministic; no catastrophic cells)
- snr_med ∈ [55.59, 55.62] dB (well above 10 dB gate)

Kernels execute correctly under R50D shim AS-IS dispatch on the alternate `.co` files; performance, not correctness, is the failure mode.

---

## Mechanism analysis

R59 decider hypothesis: K=14336 is intermediate-K (not the K-bound regime where 256×256 wins). Wider-N tiles (96×640, 64×1024) reduce grid_y by 5-8× which may improve XCD load balance for M=4096 small-M dispatches.

**Hypothesis FALSIFIED.** Both wider-N tiles dramatically underperform the 256×256 baseline:
- 96×640: grid = (43, 52) — grid_y 52 vs 256×256 grid_y = 128. Despite the supposedly-favorable smaller grid_y, eff penalty (83.5 vs 256×256's higher eff) dominates.
- 64×1024: grid = (64, 32) — grid_y 32 vs 256×256 grid_y = 128. Even smaller grid but eff=60.2 collapses kernel TFLOPS.

aiter heuristic prediction (eff < 100 disfavored on this shape) was empirically validated. The aiter hand-tuned 256×256 dispatch remains best for L1.

---

## Axis closure

**`(96×640, 64×1024)` alt-tile axis CLOSED on L1 `(4096,32768,14336)`.**

Combined with R56 G-4 confirming 256×256 best on L1 neighbor cells, AND R57 H-2 axis closure on 192×256 for HK cells (L1 not part of that probe but the mechanism logic transfers — wider/narrower tiles all underperform 256×256 at K=14336+ on aiter), L1 alt-tile space is **fully exhausted**.

L1 remains at **R57J1_L1_AITER 256×256 99.94% LOSE-edge** under R45+ default ITERS=500. Per Opt R disposition (recommendation A), no mixed-protocol ITERS=1000 bump; L1 stays as the single noise-edge LOSE cell with documented bit-deterministic guarantee (wcf_max=0).

---

## D-3A-1 protection action

ACCEPT_FALLBACK applies — preserve R57J1_L1_AITER 256×256 manifest entry verbatim. No fragment merge; no manifest delta from R59 J-3.

---

## Files emitted

- `bench_R59J3_V1.py` — 96×640 bench
- `bench_R59J3_V2.py` — 64×1024 bench
- `R59_OPT_V1_SMOKE.{json,log}` — V-1 5-seed SMOKE results
- `R59_OPT_V2_SMOKE.{json,log}` — V-2 5-seed SMOKE results
- `R59J3_V1_INTEGRATION_FRAGMENT.json` — V-1 integration fragment (DEAD)
- `R59J3_V2_INTEGRATION_FRAGMENT.json` — V-2 integration fragment (DEAD)
- `R59_OPT_J3_VERDICT.md` — this verdict

---

## GPU release

GPUs 5, 6 released. No python processes remain (`pgrep -af bench_R59J3` empty).
