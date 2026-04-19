# R57 Cohort H-3 (Opt K) — VERDICT

## Summary
**ACCEPT_FALLBACK** — 224×256 alt-tile probe on L8 = `(4096, 32768, 128256)` is **CORRECT but UNDERPERFORMS** the AITER 256×256 baseline. D-3A-1 protection holds R52D2B 256×256.

## Pre-flight checks
- `.co` file present: **YES** — `f4gemm_bf16_per1x32Fp4_BpreShuffle_224x256.co`
- Kernel symbol resolved: **YES** — `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_224x256E`
- R50D shim accepts `tile_M=224`: **YES** — shim parameterizes tile_M (gdy = ceil(M/tile_M)); launched cleanly
- GPU 4 idle: verified
- Bench params: warmup=200, iters=500, trim=0.10 (per benchmark-rules.md)

## SMOKE result (seed=101)
| metric | value |
|---|---|
| status | OK |
| finite_frac | 1.000000 |
| wrong_cell_frac | 0.000000 |
| snr_med_db | 55.63 |
| tflops | 5064.8 |
| competitor_tflops | 5781.1 |
| **pct_comp** | **87.61%** |

## Gate check
- Smoke-first gate threshold: pct_comp ≥ 99.30% (current AITER 256×256 baseline 98.30 + 1.0pp)
- Observed: **87.61%** → **FAILS by ~11.7pp**
- Per protocol (token probe), STOP — no 10-run escalation

## Verdict: ACCEPT_FALLBACK
- Keep R52D2B AITER 256×256 on L8 (D-3A-1 protected)
- Abandon 224×256 alt-tile for this shape

## Mechanism interpretation
- 224×256 has theoretical eff = 224·256/(224+256) = **119.5** vs 256×256 eff = 128.0
- Observed perf gap (~11pp) consistent with the eff drop plus poor tile/grid divisibility
  (M=4096 / tile_M=224 → gdy=19 with tail row coverage 19·224=4256, wasted 160-row partial tile)
- This corroborates R56 G-4 closure: lower-eff alt-tiles uniformly LOSE on L8;
  256×256 strictly best across 192/224/128/64 tile axis

## Files emitted
- `bench_R57H3_K1.py` (template adapted from `bench_R56G4_C3.py`)
- `R57_OPT_K1_SMOKE.json`, `R57_OPT_K1_SMOKE.log`
- `R57H3_K1_INTEGRATION_FRAGMENT.json`
- `R57_OPT_H3_VERDICT.md` (this file)

## No kernel/shim changes
- R50D shim used AS-IS (no rebuild) — 9th candidate AS-IS reuse round
- No HK kernel rebuild
