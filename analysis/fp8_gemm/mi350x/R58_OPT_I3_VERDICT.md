# R58 Opt P (Cohort I-3) — 128×256 alt-tile probe on 3 HK survivor cells

**Date:** 2026-04-20
**Worker:** I-3
**Cohort:** I-3 (Opt P, OPTIONAL closure)
**GPUs used:** 2, 3 (both idle, verified pre-launch)
**Wall budget:** ~10 min total (3 SMOKEs in parallel + 1 escalated 10-run)
**Bench rules:** warmup=200, iters=500, trim_frac=0.10 (R58 ITERS=500 default)

---

## 1. Binary availability

**128×256 .co VERIFIED present:** `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256.co` (27704 bytes). Symbol: `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256E`. Reused via R50D shim AS-IS (10th consecutive AS-IS reuse round preserved).

---

## 2. Per-cell SMOKE results

| Cell | Shape | HK baseline | SMOKE pct_comp | Threshold (HK +1pp) | Decision |
|---|---|---:|---:|---:|---|
| **P-1** | 16384×4096×2048 | 109.57% | **105.89%** | 110.57% | STOP_ACCEPT_FALLBACK (-3.68pp) |
| **P-2** | 16384×4096×3072 | 103.25% | **106.35%** | 104.25% | **ESCALATE** (+3.10pp) |
| **P-3** | 32768×14336×2048 | 100.53% | **75.49%** | 101.53% | STOP_ACCEPT_FALLBACK (catastrophic -25.04pp) |

All 3 SMOKEs were correctness-clean (snr_med ≈ 55.5 dB, wcf=0, finite=1.0) — no correctness regressions. The differentiator is purely perf.

---

## 3. P-2 escalation — 10-run @ INDEPENDENT seeds

10 INDEPENDENT seeds [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]:
- **n_OK = 10/10**
- **wcf_max = 0.0**, **wcf_std = 0.0** — bit-deterministic across all 10 seeds
- **fin_min = 1.0**
- **snr_med ∈ [55.55, 55.58] dB** (tight cluster)
- **tflops_first = 3728.0**, **pct_comp_first = 106.75%**
- **passes_10run_gate = true** (R45+ strict gate)
- **passes_D-3A-1_perf_gate = true** (>104.25% required)
- **passes_WIN_gate = true** (>=100%)
- **perf_delta vs HK = +3.50pp** (103.25 → 106.75)

This is a clean PROMOTE under all R58 Opt P gates AND the D-3A-1 AITER→HK swap special clause:
- Trades HK cohort-race surface for AITER bit-determinism
- HK→AITER swap is allowed per the AITER→HK clause (the inverse direction has no analogous prohibition; we are *gaining* bit-determinism, not losing it)
- Strictly beats HK by ≥+1.0pp (+3.50pp actual)

---

## 4. Per-cell verdicts

| Cell | Verdict | Rationale |
|---|---|---|
| **P-1** | ACCEPT_FALLBACK | 128×256 -3.68pp under HK 109.57%; HK 256×256 is the strictly-better tile choice on the (16384, 4096, 2048) M-major skinny shape. |
| **P-2** | **PROMOTE** | 128×256 +3.50pp over HK 103.25%; 10-run n_OK=10/10, wcf_max=0, pct_comp=106.75%; HK→AITER swap converts cohort-race cell to bit-deterministic. **+0 NEW WIN cell** (already WIN at 103.25%); +1 AITER bit-deterministic share (39→40); -1 HK cell (3→2). |
| **P-3** | ACCEPT_FALLBACK | 128×256 catastrophic -25.04pp under HK 100.53% (75.49%); 128×256 grid coverage inadequate for M=32768/N=14336 large-grid shape. SMOKE-DEAD STOP rule triggered. |

---

## 5. Axis-closure status — 128×256 alt-tile on HK kept-cell pool

**Status: PARTIALLY CLOSED.**

| Cell | Shape | 128×256 outcome | Axis status for this cell |
|---|---|---|---|
| P-1 | 16384×4096×2048 | DEAD -3.68pp | CLOSED (128×256 inferior to HK 256×256) |
| P-2 | 16384×4096×3072 | **WIN +3.50pp** | **OPEN/PROMOTED** (128×256 strictly better than HK) |
| P-3 | 32768×14336×2048 | DEAD -25.04pp catastrophic | CLOSED (128×256 grossly inferior; do not retry) |

**Combined with R57 H-2 closure of 192×256 on the same 3 cells (-8 to -14pp on all 3):**
- For P-1 and P-3: both 192×256 AND 128×256 are CLOSED on the AITER alt-tile axis. HK 256×256 remains the manifest entry. R59+ should not re-attempt 96×640 / 64×1024 here (both lower-eff than 128×256 / 192×256 already DEAD).
- For P-2: a successful AITER alt-tile WIN was found at 128×256 (eff=85.3) despite 192×256 failing earlier. The mechanism: 128×256 yields a different gridx geometry (gdx=128 vs 86) on M=16384 K=3072 and may better saturate the XCD ring; the eff-only model under-predicts here.

**Net round delta from this cohort:** +1 PROMOTE (P-2), 2 ACCEPT_FALLBACK (P-1, P-3), 0 DEAD. The +1 PROMOTE preserves WIN count (P-2 was already WIN at 103.25%) and gains +1 AITER bit-deterministic share.

---

## 6. Methodology notes

- All 3 SMOKEs ran clean to completion in ~8s wall each on idle GPUs 2, 3 (verified `rocm-smi --showuse` pre-launch).
- 10-run for P-2 ran sequentially on GPU 3 (~80s wall total) and was bit-deterministic across all 10 INDEPENDENT seeds — strong evidence that 128×256 on (16384, 4096, 3072) AITER inherits the same bit-determinism profile as the other 38 AITER cells.
- No kernel rebuild, no shim modification (R50D shim AS-IS, 10th consecutive round when applied to P-2).
- ITERS=500 R58 default used (no protocol bump).
- The Opt P probe was framed as "very low confidence axis closure" but produced 1 unexpected PROMOTE — supports the methodology principle that low-confidence axis-closure probes are still worth running when SMOKE cost is bounded.

---

## 7. Files produced

- `bench_R58I3_P1.py`, `bench_R58I3_P2.py`, `bench_R58I3_P3.py` (bench scripts)
- `R58_OPT_P1_SMOKE.{json,log}`, `R58_OPT_P2_SMOKE.{json,log}`, `R58_OPT_P3_SMOKE.{json,log}` (SMOKE outputs)
- `R58_OPT_P2_10RUN.{json,log}` (escalated 10-run for P-2)
- `R58I3_P1_INTEGRATION_FRAGMENT.json`, `R58I3_P2_INTEGRATION_FRAGMENT.json`, `R58I3_P3_INTEGRATION_FRAGMENT.json` (per-cell fragments)
- `R58_OPT_I3_VERDICT.md` (this file)
