# R52 Opt D-2A — VERDICT: PROMOTE

**Date:** 2026-04-19
**Worker:** R52 Opt D-2A (perf claw-back via aiter `.co` dlopen)
**Target shape:** (M=4096, N=28672, K=32768) — sub-90% HK-VC shape on R51 leaderboard (~61.9% comp)
**Mechanism:** Reuse R50D shim AS-IS; same .co, same kernel symbol, only shape constants change.

---

## Result: PROMOTE

| Metric            | Value                       | Gate              | Pass |
|-------------------|-----------------------------|-------------------|:----:|
| n_OK              | 10/10                       | >= 8/10           | YES  |
| wcf_max           | 0.0                         | < 0.02            | YES  |
| wcf_std           | 0.0                         | < 0.01            | YES  |
| fin_min           | 1.0                         | >= 0.97           | YES  |
| snr_med (med)     | 55.62 dB                    | >= 10 dB          | YES  |
| snr_med (range)   | [55.610, 55.647]            | —                 | —    |
| TFLOPS            | 5736.7 (1 perf seed)        | —                 | —    |
| Competitor TFLOPS | 5649.9 (bench_all_42.py)    | —                 | —    |
| **pct_comp**      | **101.54%**                 | >= 95% per shape  | YES  |

**10-run @ 80% gate: PASS_10/10** (perfect).

---

## Bench rules

- warmup = 200, iters = 500, trim_frac = 0.10 (mandatory).
- 10 INDEPENDENT seeds: [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010].
- Round-robin across `HIP_VISIBLE_DEVICES=4,5` (1 perf + 9 correctness-only).
- Total wall-time: ~80 s.

---

## Delta from baseline

- HK pre-R52:  ~3497 TFLOPS @ ~61.9% comp
- aiter dispatch: 5736.7 TFLOPS @ 101.54% comp
- **Delta: +~2240 TFLOPS, +~39.6 pct points** — closes a major sub-90% gap on the post-R51 leaderboard.
- Net VC delta: **0** (already HK-VC under R44D gate), but per-shape gap drops below the "no shape > 5%" hard target (was 38.1 pct points below comp; now 1.54 pct points above comp).

---

## Per-shape grid params

- `gdx = ceil(N/256) = ceil(28672/256) = 112`
- `gdy = ceil(M/256) = ceil(4096/256) = 16`
- `gdz = 1`
- `bdx = 256`
- KernelArgs: M=4096, N=28672, K=32768 (passed via shim ABI)

---

## Recommendation

**PROMOTE** to reviewer. Reviewer should add this entry to the aiter dispatch table in `bench_all_42_R51_INTEGRATION.py` (or its R52 successor) and re-run the integration sweep to confirm no regressions on the other R51-promoted shapes.

This is the **third** successful application of the R50D `.co` dlopen mechanism (after R50D's original `(4096,32768,28672)` and R51 D-1's `(14336,4096,32768)`), strongly confirming the pattern generalizes cleanly within the 256×256 tile family with **zero shim rebuilds** required — only per-shape grid arithmetic and shape constants need to change.

---

## Output files

- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/bench_R52D2A.py`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R52_OPT_D2A_SMOKE.json`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R52_OPT_D2A_SMOKE.log`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R52_OPT_D2A_10RUN.json`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R52_OPT_D2A_10RUN.log`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R52D2A_INTEGRATION_FRAGMENT.json`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R52_OPT_D2A_VERDICT.md` (this file)
