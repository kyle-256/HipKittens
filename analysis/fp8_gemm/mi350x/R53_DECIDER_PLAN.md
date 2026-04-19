# R53 DECIDER PLAN

**Round:** R53 (post R52 commit `0a20ed2b`)
**Date:** 2026-04-19
**Baseline:** 36/42 strict 10-run VC; 7 shapes already promoted to aiter `.co` dispatch via R50D shim
**Mandate:** strict 10-run @ 80% gate; warmup=200, iters=500, trim=0.10
**Compute:** MI355X gfx950, 4 GPUs (0,1,2,3), seeds [101..1010]

---

## Key finding from heuristic analysis

The R50D shim (`build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`)
**already accepts** `tile_M`, `tile_N`, `co_path`, `kernel_name` as launch
kwargs (see `R50D_aiter_dlopen.cpp:147-203`). Default is 256x256, but ANY
single-kernel `.co` from `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/`
can be dispatched via the same `.so` AS-IS. Caveat: `bdx=256` is hardcoded;
non-256x256 tiles must verify wave/block layout in smoke.

Aiter heuristic computed for all sub-95% HK-backed shapes (NUM_CU=256,
formula in `asm_gemm_a4w4.cu:85-152`). **None pick 256x256** — most pick
`64x1024` or `96x640` because smaller-N tiles minimize `local_round`
when N>>M. This means **D-extended-3 (256x256-only reuse) is exhausted**.
R53 must pivot to D-non-256x256.

---

## Per-candidate analysis (sub-95% HK-backed shapes)

| Shape (MxNxK) | R52 backend | comp% | Aiter heuristic tile | splitK | rounds | Target backend | Confidence |
|---|---|---:|---|---:|---:|---|---|
| 4096x14336x16384 | R40B | 84.53 | 96x640 | 1 | 4 | aiter `96x640.co` | HIGH |
| 6144x4096x16384 | R40B | 83.89 | 96x640 | 1 | 2 | aiter `96x640.co` | HIGH |
| 4096x6144x32768 | R41A | 82.39 | 96x640 | 1 | 2 | aiter `96x640.co` | HIGH |
| 4096x32768x14336 (NO_VC) | R40B | 82.85 | 64x1024 | 1 | 8 | aiter `64x1024.co` | HIGH (rescue) |
| 32768x4096x14336 (NO_VC) | R40B | 84.83 | 64x1024 | 1 | 8 | aiter `64x1024.co` | HIGH (rescue) |
| 128256x32768x4096 | R40B | 85.95 | 64x1024 | 1 | 251 | aiter `64x1024.co` | MED (deep K=4096) |
| 14336x32768x4096 | R40B | 86.06 | 64x1024 | 1 | 28 | aiter `64x1024.co` | MED |
| 28672x32768x4096 | R40B | 87.31 | 64x1024 | 1 | 56 | aiter `64x1024.co` | MED |
| 16384x4096x14336 | R41B | 87.50 | 64x1024 | 1 | 4 | aiter `64x1024.co` | MED |
| 32768x4096x7168 | R40B | 88.43 | 64x1024 | 1 | 8 | aiter `64x1024.co` | MED |
| 6144x4096x8192 | R40B | 89.57 | 96x640 | 1 | 2 | aiter `96x640.co` | MED |
| 4096x32768x4096 | R40B | 90.82 | 64x1024 | 1 | 8 | aiter `64x1024.co` | LOW |
| 4096x14336x8192 | R40B | 91.74 | 96x640 | 1 | 4 | aiter `96x640.co` | LOW |
| 28672x4096x8192 | R40B | 91.90 | 64x1024 | 1 | 7 | aiter `96x640.co` | LOW |
| 4096x32768x6144 | R40B | 91.87 | 64x1024 | 1 | 8 | aiter `64x1024.co` | LOW |
| 16384x14336x4096 (NO_VC) | R40B | 94.49 | 64x1024 | 1 | 14 | aiter `64x1024.co` | LOW |

(Higher comp% = less headroom = lower confidence; first 5 rows have largest
gap to 100%.)

---

## R53 priority ranking

### Tier-1 (highest confidence, biggest gap, includes 2 NO_VC rescues)

1. **D-3A (96x640 cohort)** — 3 shapes: `4096x14336x16384` (84.53%),
   `6144x4096x16384` (83.89%), `4096x6144x32768` (82.39%). All pick 96x640
   per heuristic, all in low-80s comp. R52 D-2 saw +27-40 pp swings on
   similar gaps; conservative target +10-25 pp each.

2. **D-3B (64x1024 cohort + NO_VC rescue)** — 3 shapes:
   `4096x32768x14336` (NO_VC, 82.85%), `32768x4096x14336` (NO_VC, 84.83%),
   `128256x32768x4096` (85.95%). Two NO_VC rescues plus largest "true VC
   but slow" deep-K shape. If aiter `.co` solves — +1 to +2 NET VC plus
   perf gain on third.

3. **D-3C (64x1024 medium-K)** — 3 shapes: `14336x32768x4096` (86.06%),
   `28672x32768x4096` (87.31%), `16384x4096x14336` (87.50%). Mid-tier
   gaps; lower per-shape upside but high integration probability since
   no fence/race issues at K<32k for 64x1024 tile.

### Tier-2 (only if Tier-1 over-saturates parallelism budget)

4. Additional 64x1024 mid-K: `32768x4096x7168` (88.43%),
   `6144x4096x8192` (89.57%).

### Tier-3 (defer — low headroom or speculative)

5. 90-95% comp shapes: `4096x32768x4096`, `4096x14336x8192`,
   `28672x4096x8192`, `4096x32768x6144` — likely cohort-noise band.

6. Opt B (32×32×64 MFMA HK kernel rewrite) — DEFER. ~1-2 day cost vs
   Tier-1's hours. Only invoke if D-3A/B/C all DEAD.

---

## Final 3-worker dispatch (parallel)

| Worker | Assignment | Shapes | Shim rebuild? | Estimated wall |
|---|---|---|---|---|
| **W1** | R53 Opt D-3A (96x640 cohort) | 4096x14336x16384, 6144x4096x16384, 4096x6144x32768 | NO (R50D AS-IS, kwargs only) | ~30 min smoke + 10-run |
| **W2** | R53 Opt D-3B (64x1024 + NO_VC rescue) | 4096x32768x14336, 32768x4096x14336, 128256x32768x4096 | NO (R50D AS-IS, kwargs only) | ~30 min smoke + 10-run |
| **W3** | R53 Opt D-3C (64x1024 medium-K) | 14336x32768x4096, 28672x32768x4096, 16384x4096x14336 | NO (R50D AS-IS, kwargs only) | ~30 min smoke + 10-run |

### Per-worker protocol

1. **Smoke** each shape via `shim.launch(..., tile_M=T_M, tile_N=T_N,
   co_path=AITER_CO, kernel_name=KNL_NAME)` against torch reference.
   If finite_frac < 0.97 OR SNR < 30 dB → SKIP candidate (likely needs
   tile-specific bdx; flag for shim rebuild in R54).
2. **5-run** for shapes that smoke-pass; promote to **strict 10-run**
   only if 5/5 PASS @ wcf<0.02.
3. **PROMOTE only on**: 10/10 PASS, wcf_max<0.02, wcf_std<0.01,
   fin_min≥0.97 (R52 strict gate).
4. Reuse R52 integration manifest pattern (`R52_INTEGRATION_MANIFEST.json`)
   as template; new manifest will be `R53_INTEGRATION_MANIFEST.json`.

### Kernel/symbol map for workers

| Tile | `.co` file | Symbol (kernel_name) |
|---|---|---|
| 96x640 | `f4gemm_bf16_per1x32Fp4_BpreShuffle_96x640.co` | `_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_96x640E` |
| 64x1024 | `f4gemm_bf16_per1x32Fp4_BpreShuffle_64x1024.co` | `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_64x1024E` |

Symbol-length prefix differs (`41` vs `42`) — must use exactly as in
`asm_f4gemm_configs.hpp` / `f4gemm_bf16_per1x32Fp4.csv`.

### Fallback if Tier-1 has DEAD smokes

If W1/W2/W3 smokes show `bdx=256` ABI mismatch on non-256x256 tiles
(SNR < 0 dB or fault), the shim **does** need rebuild to parameterize
`bdx`. Rebuild path: edit `R50D_aiter_dlopen.cpp:187` to take `bdx` as
arg, rebuild via `build_R50D.py`. Cost: ~5 min. Defer to R54 if any
worker hits this.

---

## Risks

- **bdx ABI mismatch**: aiter's non-256x256 kernels may launch with
  block size != 256. Mitigation: smoke first; rebuild shim for R54 if
  needed.
- **stride_A0/B0 differences**: 256x256 expects `stride * 2` for fp4_x2.
  Smaller tiles likely match, but confirm via aiter `.cu` if smoke fails.
- **Cohort tail-draw churn**: per `project_mxfp4_R45_cohort_tail_draw.md`,
  expect +/- 1-2 VC churn on UNCHANGED .so shapes. Net delta is what
  matters.

## Expected R53 outcome (success path)

- **+3 to +5 NET VC** (1 from each NO_VC rescue if either lands; 2-3
  from perf-claw-back PROMOTEs reaching 100% comp).
- **+5 to +15 pp mean perf delta** on 6 of the 9 candidate shapes.
- Methodology proof: aiter `.co` dlopen pattern generalizes beyond
  256x256 → unlocks 35-tile aiter library for R54+.
