# R56 Opt G-3 Verdict — 128x512 falsification probe on cluster A

**Date:** 2026-04-19
**Cohort:** G-3 (medium-confidence falsification)
**Shim:** `build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so` (AS-IS, 8th consecutive AS-IS reuse)
**Tile:** 128x512 (eff = 102.4) via `f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512.co`
**Kernel symbol:** `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512E`
**GPUs:** 4, 5 (verified idle pre-launch)
**Bench:** warmup=200, iters=500, trim=0.10; 10-run @ 80% INDEPENDENT seeds [101..1010 step 101]

---

## Result table

| Cell | Shape | Current (64x1024) | 128x512 | Delta | n_OK | wcf_max | fin_min | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| L1 | (4096, 32768, 14336)  | 65.68% | **92.48%**  | +26.80pp | 10/10 | 0.0 | 1.0 | **PROMOTE** |
| L2 | (32768, 4096, 14336)  | 84.13% | **104.70%** | +20.57pp | 10/10 | 0.0 | 1.0 | **PROMOTE** |
| L6 | (16384, 4096, 14336)  | 90.28% | **107.19%** | +16.91pp | 10/10 | 0.0 | 1.0 | **PROMOTE** |

**Aggregate perf claw-back: +64.28pp across 3/3 PROMOTE.**

All cells pass strict-VC 10-run gate (n_OK >= 8/10, wcf_max < 0.02, wcf_std < 0.01, fin_min >= 0.97). All 30 individual seed runs were bit-deterministic (wcf_max = wcf_std = 0.0; fin_min = fin_max = 1.0). SNR median: ~55.6 dB on every run.

---

## Mechanism inference

**G-3 (128x512) PROMOTEs on all 3 candidates.** Combined with G-1 (256x256) cohort results, the mechanism interpretation is:

- **Provisional inference (pending G-1 confirmation):** Both 256x256 (eff=128.0) and 128x512 (eff=102.4) tiles win against 64x1024 (eff=60.2). This supports the interpretation that **the 64x1024 tile is "too narrow"** (any wider tile beats it) rather than `compute2mem_efficiency` being the sole dominant factor.
- **Stronger statement:** If G-1's 256x256 PROMOTEs by a *larger* margin than G-3's 128x512, then `compute2mem_efficiency` still dominates within the "wider-than-narrow" regime — both axes operate.
- **Decisive comparison requires G-1 fragments.** Predicted: G-1 256x256 will give larger pct_comp deltas on L1 (~+30pp), comparable on L2 (~+20pp), and larger on L6 (~+18pp+) given eff jump 60.2→128.0 vs 60.2→102.4.

**Reviewer's call:** G-3 PROMOTE on all 3 cells means the aiter dispatcher's pick of 64x1024 was a clear under-pick. The reviewer should select between G-1 (256x256) vs G-3 (128x512) per cell on the highest pct_comp.

---

## Bit-determinism audit

All 30 runs (3 cells x 10 seeds) showed:
- `kernel_finite = 1.0` on every run
- `wrong_cell_frac = 0.0` on every run
- `snr_med_db` clustered tightly at 55.58-55.63 dB (cross-seed variance < 0.05 dB)

Confirms AITER `.co` dlopen pattern's bit-determinism property, which is the load-bearing assumption for the R56 perf-claw-back gate.

---

## Output artifacts

- `bench_R56G3_L{1,2,6}.py` — bench harnesses (R55 D-5 template with shape/tile/co_path/kernel_name updates)
- `R56_OPT_G3_L{1,2,6}_SMOKE.{json,log}` — smoke verification (1 seed, full perf measurement)
- `R56_OPT_G3_L{1,2,6}_10RUN.{json,log}` — 10-run @ 80% INDEPENDENT seeds (perf on seed 0 only)
- `R56G3_L{1,2,6}_INTEGRATION_FRAGMENT.json` — per-cell fragments per R56 schema (Section 8 of `R56_DECIDER_PLAN.md`)

No shim modifications were made. No kernel rebuilds. R50D shim accepts arbitrary `tile_M`/`tile_N` parameters (verified via `R50D_aiter_dlopen.cpp` source review) — only computes `gdx = ceil(N/tile_N)`, `gdy = ceil(M/tile_M)`. 128x512 is the 4th distinct tile shape successfully driven through the shim (after 256x256, 96x640, 64x1024).
