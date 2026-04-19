# R52 Opt D-2C — VERDICT: PROMOTE  (PERF CLAW-BACK +28.42 pp)

**Date:** 2026-04-19
**Target shape:** `(M=4096, N=4096, K=32768)`
**Mechanism:** Direct port of R50D `.co` dlopen pattern (no shim rebuild) on perf claw-back candidate.

## Result: **PROMOTE** — 0 VC delta (already YES-VC), +28.42 pp pct_comp uplift

The previously-YES-VC HipKittens R41A path is replaced by the aiter
`f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` binary launched via the
existing R50D shim. All 10 INDEPENDENT seeds pass the cohort-race gate
deterministically AND perf is well above the 95% hard floor.

## Numbers

| Metric | Value | Gate | Pass |
|---|---:|---|:--|
| n_OK / n_runs | **10 / 10** | >= 8/10 | YES |
| wcf_max | **0.0** | < 0.02 | YES |
| wcf_std | **0.0** | < 0.01 | YES |
| fin_min | **1.0** | >= 0.97 | YES |
| snr_med (range) | **55.610 - 55.647 dB** | >= 10 dB | YES |
| tflops (10run seed 101, warmup=200, iters=500, trim=0.10) | **5437.2** | -- | -- |
| tflops (smoke seed 101) | **5444.9** | -- | -- |
| competitor_tflops | 5152.8 | -- | -- |
| **pct_comp** | **105.52% (10run) / 105.67% (smoke)** | >= 95% (>=100% R52 stop crit) | YES |

## Comparison to previous state

| | R51 (HipKittens R41A) | R52 D-2C (aiter `.co`) | Delta |
|---|:---:|:---:|:---:|
| Verdict | YES_VC_10/10 | YES_VC_10/10 | unchanged |
| pct_comp | 77.10% | 105.52% | **+28.42 pp** |
| TFLOPS | 3973.4 | 5437.2 | +1463.8 |
| VC contribution | +1 | +1 | 0 NET VC delta |

## Recommendation

**INTEGRATE.** Pure perf claw-back; no VC count change but +28.42 pp comp on a sub-90% shape.
Mechanism is identical to R50D / R51 D-1/2/3; reviewer just appends the shape entry from
`R52D2C_INTEGRATION_FRAGMENT.json` to the integration manifest.

- Same `.co` and `kernel_name` as R50D / R51 D-1/2/3 — no shim rebuild needed.
- Same shim (`R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`) AS-IS — shim
  auto-computes grid from M/N/tile (gdx=16, gdy=16, single-round).
- Same B preshuffle (`shuffle_weight(layout=(16,16))`) and scale prep
  (`get_triton_quant(per_1x32)(..., shuffle=True)`) as R50D.
- 5.52% above competitor — comfortably above the R52 100% stop criterion AND the
  "no shape > 5% gap" hard target (we are 5.52% ABOVE competitor, not below).

## Files

- `bench_R52D2C.py` — bench harness (clone of `bench_R51D3.py`, shape changed,
  competitor_tflops=5152.8, default GPU=2).
- `R52_OPT_D2C_SMOKE.json` — 1-seed smoke (seed=101), `pct_comp=105.67%`.
- `R52_OPT_D2C_10RUN.json` — 10 INDEPENDENT seeds [101..1010] gate result.
- `R52D2C_INTEGRATION_FRAGMENT.json` — drop-in dispatch entry for the
  R52 integration manifest.

## GPU isolation

GPU 2 used (per assignment HIP_VISIBLE_DEVICES=2; D-2A on GPU 4,5; we picked GPU 2 to
avoid contention). GPU 2 verified idle via `rocm-smi --showuse` before run start.
Total wall time 10-run = ~78s.
