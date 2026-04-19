# R51 Opt D-1 — VERDICT: PROMOTE

**Date:** 2026-04-19
**Worker:** R51 Opt D-1 (perf claw-back via aiter `.co` dlopen)
**Target shape:** (M=14336, N=4096, K=32768) — largest pct-gap HK-VC shape on R50 leaderboard
**Mechanism:** Reuse R50D shim AS-IS; same .co, same kernel symbol, only shape constants change.

---

## Result: PROMOTE

| Metric          | Value                       | Gate              | Pass |
|-----------------|-----------------------------|-------------------|:----:|
| n_OK            | 10/10                       | >= 8/10           | YES  |
| wcf_max         | 0.0                         | < 0.02            | YES  |
| wcf_std         | 0.0                         | < 0.01            | YES  |
| fin_min         | 1.0                         | >= 0.97           | YES  |
| snr_med (med)   | 55.62 dB                    | >= 10 dB          | YES  |
| snr_med (range) | [55.611, 55.636]            | —                 | —    |
| TFLOPS          | 5577.6 (1 perf seed)        | —                 | —    |
| Competitor TFLOPS | 5245.4 (bench_all_42.py)  | —                 | —    |
| **pct_comp**    | **106.33%**                 | >= 95% per shape  | YES  |

**10-run @ 80% gate: PASS_10/10** (perfect).

---

## Bench rules

- warmup = 200, iters = 500, trim_frac = 0.10 (mandatory).
- 10 INDEPENDENT seeds: [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010].
- Round-robin across `HIP_VISIBLE_DEVICES=0,1` (1 perf + 9 correctness-only).
- Total wall-time: ~78s.

---

## Delta from baseline

- HK pre-R51:  3169 TFLOPS @ 60.41% comp (R41A path)
- aiter dispatch: 5577.6 TFLOPS @ 106.33% comp
- **Delta: +2408.6 TFLOPS, +45.92 pct points** — closes the largest single-shape gap on the post-R50 leaderboard.
- Net VC delta: **0** (already HK-VC), but per-shape gap drops below the "no shape > 5%" hard target (gap was 39.59 pct points, now -6.33 pct points i.e. above competitor).

---

## Recommendation

**PROMOTE** to reviewer. Reviewer should add this entry to `bench_all_42_R50_INTEGRATION.py`'s aiter dispatch table and re-run the integration sweep to confirm no regressions on the other 35 R50-promoted shapes.

This is the second successful application of the R50D `.co` dlopen mechanism (after the original R50D promotion of `(4096,32768,28672)`), confirming the pattern generalizes cleanly within the 256×256 tile family. Opt D-2 / D-3 should proceed in parallel on the remaining sub-90% candidates.

---

## Output files

- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/bench_R51D1.py`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R51_OPT_D1_SMOKE.json`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R51_OPT_D1_10RUN.json`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R51D1_INTEGRATION_FRAGMENT.json`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R51_OPT_D1_VERDICT.md` (this file)
