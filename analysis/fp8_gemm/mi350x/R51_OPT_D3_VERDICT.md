# R51 Opt D-3 — VERDICT: PROMOTE  (+1 NET VC)

**Date:** 2026-04-19
**Target shape:** `(M=28672, N=4096, K=16384)`
**Mechanism:** Direct port of R50D `.co` dlopen pattern (no shim rebuild) on FLAKE_7/10 conversion candidate.

## Result: **+1 NET VC** (R50 36/42 → R51 ≥ 37/42 if integrated)

The previously-FLAKE-7/10 HipKittens R40B path is replaced by the aiter
`f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` binary launched via the
existing R50D shim. All 10 INDEPENDENT seeds pass the cohort-race gate
with deterministic correctness AND perf above the 95% hard floor.

## Numbers

| Metric | Value | Gate | Pass |
|---|---:|---|:--|
| n_OK / n_runs | **10 / 10** | ≥ 8/10 | YES |
| wcf_max | **0.0** | < 0.02 | YES |
| wcf_std | **0.0** | < 0.01 | YES |
| fin_min | **1.0** | ≥ 0.97 | YES |
| snr_med (range) | **55.586 – 55.610 dB** | ≥ 10 dB | YES |
| tflops (10run seed 101, warmup=200, iters=500, trim=0.10) | **5515.4** | — | — |
| tflops (smoke seed 101) | **5526.1** | — | — |
| competitor_tflops | 5350.6 | — | — |
| **pct_comp** | **103.08% (10run) / 103.28% (smoke)** | ≥ 95% | YES |

## Comparison to previous state

| | R50 (HipKittens R40B) | R51 D-3 (aiter `.co`) | Delta |
|---|:---:|:---:|:---:|
| Verdict | FLAKE_7/10 | PASS_10/10 | conversion |
| pct_comp | 82.18% | 103.08% | **+20.90 pp** |
| VC contribution | 0 | **+1** | **+1 NET VC** |

## Recommendation

**INTEGRATE.** This is the highest-leverage R51 candidate (only one of D-1/D-2/D-3
that increases the VC count). Mechanism is identical to R50D; reviewer just appends
the shape entry from `R51D3_INTEGRATION_FRAGMENT.json` to the integration manifest.

- Same `.co` and `kernel_name` as R50D — no shim rebuild needed.
- Same shim (`R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`) AS-IS — shim
  auto-computes grid from M/N/tile (gdx=16, gdy=112).
- Same B preshuffle (`shuffle_weight(layout=(16,16))`) and scale prep
  (`get_triton_quant(per_1x32)(..., shuffle=True)`) as R50D.
- Above the "no shape > 5% gap" hard target (we're 3.08% ABOVE competitor).

## Files

- `bench_R51D3.py` — bench harness (copy of `bench_R50D.py`, shape changed,
  competitor_tflops=5350.6).
- `R51_OPT_D3_SMOKE.json` — 1-seed smoke (seed=101), `pct_comp=103.28%`.
- `R51_OPT_D3_10RUN.json` — 10 INDEPENDENT seeds [101..1010] gate result.
- `R51D3_INTEGRATION_FRAGMENT.json` — drop-in dispatch entry for the
  R51 integration manifest.

## GPU isolation

GPU 4 used (per assignment HIP_VISIBLE_DEVICES=4,5; only one device is needed
since runs are serial and the 10-run finished in ~80s wall time on a single GPU).
GPU 4 verified idle via `rocm-smi --showuse` before run start.
