# R56 Opt G-4 — Worker Verdict

**Date:** 2026-04-19
**Cohort:** G-4 (low-medium-confidence HK->AITER + alt-tile probes)
**GPUs used:** 6, 7
**Shim:** build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so (AS-IS, no rebuild)

## Summary

| Cand | Cell | Shape (MxNxK) | Current | Alt tile | Current pct_comp | Alt pct_comp | Delta (pp) | D-3A-1 | Verdict |
|------|------|---------------|---------|----------|------------------|--------------|-----------:|--------|---------|
| C1 | L7 | 4096x128256x32768 | HK R41A | 256x256 | 97.59% | **176.51%** (10-run) | **+78.92** | PASS (>98.59) | **PROMOTE** |
| C2 | L8 | 4096x32768x128256 | AITER 256x256 (R52D2B) | 128x512 | 98.32% | 85.01% (smoke) | -13.31 | FAIL (<99.32) | ACCEPT_FALLBACK |
| C3 | L8 | 4096x32768x128256 | AITER 256x256 (R52D2B) | 192x256 | 98.32% | 86.48% (smoke) | -11.84 | FAIL (<99.32) | ACCEPT_FALLBACK |

**Result: 1 PROMOTE / 2 ACCEPT_FALLBACK / 0 DEAD.**

## C1 PROMOTE (L7 HK->AITER 256x256)

This is the **first successful HK->AITER swap on a kept-HK cell** in R55 baseline. Notable:
- 10-run @ 80% INDEPENDENT seeds [101..1010 step 101]: n_OK=10/10, wcf_max=0, wcf_std=0, fin_min=1.0
- snr_med_db: 55.598 to 55.63 (range)
- tflops: 5640.0 (first seed perf measurement)
- pct_comp: **176.51%** vs HK baseline 97.59% = **+78.92pp claw-back** (largest single-cell delta this round; HK was leaving ~3.5x perf on the table here)
- Bit-determinism auto-pass via AITER `.co` dlopen
- AITER bit-deterministic share grows from 38 -> 39 of 42 if integrated
- 4 surviving HK cells reduced to 3 -> cohort-race surface from 4 -> 3 cells

## C2/C3 ACCEPT_FALLBACK (L8 alt-tile probes)

L8 baseline AITER 256x256 already at 98.32% — both alt tiles (128x512, 192x256) underperformed substantially (-11 to -13pp). Mechanism: 256x256 has highest compute2mem_efficiency (128.0 vs 102.4 / 109.7); the wider/narrower alts trade efficiency for grid-fit but K=128256 K-pipeline appears not to favor the alt-tile shapes here. D-3A-1 protection triggers: keep current baseline.

Both C2 and C3 stopped at smoke (no 10-run); shim is tile-generic and worked correctly with NEW 128x512 and 192x256 tiles (first project use). No shim modifications proposed.

## Files emitted

- bench_R56G4_C{1,2,3}.py
- R56_OPT_G4_C1_SMOKE.{json,log}, R56_OPT_G4_C2_SMOKE.{json,log}, R56_OPT_G4_C3_SMOKE.{json,log}
- R56_OPT_G4_C1_10RUN.{json,log}
- R56G4_C{1,2,3}_INTEGRATION_FRAGMENT.json
- R56_OPT_G4_VERDICT.md (this file)

## Bench rules compliance

- warmup=200, iters=500, trim=0.10 — confirmed
- HIP_VISIBLE_DEVICES=6,7 only; rocm-smi pre-check showed all 8 GPUs idle
- Shim AS-IS (no rebuild) — 7th consecutive round of R50D AS-IS reuse continues for any candidates promoted
