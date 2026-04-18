# R27 Diagnose — Shape Profile of the 9 Residual LOSE Shapes

Date: 2026-04-18  Branch: `mxfp4`  Scope: pure no-GPU/no-build research
Source data: `bench_all42_results_R25_FINAL.json`, `bench_all_42.py:519-576`,
`R26D_AUDIT_WIRING_BUG.md`, `R27_V5_MFMA32_SCOUT.md`, `kernel_mxfp4_gluon_cpp.cpp`.

---

## 0. Top-line finding

**5 of the 9 LOSE shapes are wiring bugs already fixed by `r26d_build_missing.py`** —
not algorithmic deficits. R26-D AUDIT directly measured these flips at 5-rep
warmup=200/iters=500/trim=10% with std ≤ 0.5%. Re-running the FINAL bench with
the now-built `.so` files is projected at **38/42 WIN** with zero new code work.

| # | Shape (M×N×K) | FINAL ratio | R26-D verified ratio after fix | Mechanism |
|---|---|---|---|---|
| 4 | 4096×14336×16384 | 97.2% | **112.3%** | `_kx16384` for n=14336 was missing |
| 2 | 4096×28672×32768 | 96.5% | **113.0%** | `_kx32768` (tv0) for n=28672 was missing |
| 3 | 4096×32768×6144 | 99.7% | **115.7%** | `_kx6144` for n=32768 was missing |
| 4 | 4096×32768×14336 | 98.9% | **113.8%** | `_kx14336` (tv0) for n=32768 was missing |
| 7 | 14336×4096×32768 | 94.7% | **111.7%** | `_kx32768` (lgk2) for n=4096 was missing |

The 4 shapes NOT covered by R26-D audit are the actual R27 candidates:

| # | Shape | Ratio | K | Why R26-D didn't fix |
|---|---|---|---|---|
| 5 | 4096×32768×28672 | 94.7% | 28672 | Wrong parent: `_kx28672` is `lgk2` parent; this shape's best parent is `tv0` family (`ts_v12_tv0_memc_btw_all` ran 5266 vs `gm8_step3` 5272) — R25-G K_EXACT only built one parent variant per K |
| 6 | 4096×32768×128256 | 92.8% | **128256** | **No K_EXACT entry exists for K=128256** in `bench_all_42.py:519-576`. K_iters=501 → would need pfoff≈494-497 |
| 8 | 16384×4096×14336 | 97.9% | 14336 | Best is `ts_u16` (vector-width 16 path), not lgk2 — `_kx14336` parent mismatch (built tv0+lgk2 only) |
| 9 | 16384×4096×28672 | 95.0% | 28672 | `_kx28672` lgk2 variant ran in `per_variant` at 263 TFLOPS (a launch failure, not a perf result) → effectively missing from R26-D audit set; rebuild-and-retest needed |

---

## 1. Theoretical bound profile (no-GPU compute)

Constants used (MI355X / gfx950):
- 256 CUs × ~2.4 GHz, HBM3 peak ≈ **5.3 TB/s**.
- Per-CU peak (16×16×128 mfma_scale, 1 inst/16 cyc): 16·16·128·2 / 16 = 4 096 FLOPs/cyc → **~2.5 PFLOPS @ 256 CU × 2.4 GHz** (single-issue worst case; realistic peak ~6-8 PFLOPS with parallel warps).
- Per-block reads per outer K-iter (BLK=256, BK=128, fp4): A=128·128 B + B=128·128 B = 32 KB/block/K-iter (kernel_mxfp4_gluon_cpp.cpp:529-546).
- buffer_load_dwordx4 issues per block per K-iter ≈ 2048 (16 B/load × 2048 = 32 KB).

For each LOSE shape, computing `(M·K/2 + N·K/2 + M·N·2 + (M+N)·K/32) bytes / time-at-comp-TFLOPS`:

| Shape (M×N×K) | K-iter | HBM req | %peak HBM | MFMA-issue util | Tightest bound |
|---|---:|---:|---:|---:|---|
| 4096×14336×16384  | 64  | 0.72 TB/s | 13.7% | 99.6% | **MFMA-issue** |
| 4096×28672×32768  | 128 | 0.59 TB/s | 11.2% | 112.3% | **MFMA-issue** (saturated) |
| 4096×32768×6144   | 24  | 1.07 TB/s | 20.2% | 90.4% | MFMA-issue (some headroom) |
| 4096×32768×14336  | 56  | 0.76 TB/s | 14.3% | 105.2% | **MFMA-issue** (saturated) |
| 4096×32768×28672  | 112 | 0.60 TB/s | 11.3% | 110.6% | **MFMA-issue** (saturated) |
| 4096×32768×128256 | 501 | 0.47 TB/s |  8.8% | 114.9% | **MFMA-issue** (saturated) |
| 14336×4096×32768  | 128 | 0.60 TB/s | 11.3% | 104.2% | **MFMA-issue** |
| 16384×4096×14336  | 56  | 0.78 TB/s | 14.6% | 102.2% | **MFMA-issue** |
| 16384×4096×28672  | 112 | 0.64 TB/s | 12.1% | 109.8% | **MFMA-issue** |

(util%>100 means competitor is doing better than the back-of-envelope ceiling — i.e. competitor is exploiting parallel issue across multiple warps that the simple model under-counts. Both interpretations point to the same conclusion: **MFMA-issue is the tightest port**, HBM is at ~10-20% utilisation, never the bound.)

This corroborates R24B/C ("VMEM-issue-bound, not VMEM-latency-bound"). It also
matches R25-F/G's mechanism: tail-PF-off helped because tail prefetches were
**stealing VMEM-issue slots** in the K-loop tail, where MFMA was already
saturated and didn't need any more loads.

---

## 2. Empirical signal: K_EXACT match table (cross-reference vs `bench_all_42.py:519-576`)

K_EXACT entries currently shipped (R25-G/H), each gated by `R25C_K_EXACT == K_DIM`
(`kernel_mxfp4_gluon_cpp.cpp:103-107`):

| K_EXACT | Suffix | Parent stack | Tuned-on shape (M×N×K) |
|---|---|---|---|
| 32768 | `_ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all` | tv0 | 4096×28672×32768 |
| 32768 | `_ts_lgk2_gm7_memc_pfoff124_kx32768_btw_all`       | lgk2 | 14336×4096×32768 |
| 28672 | `_ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all`       | lgk2 | 16384×4096×28672 |
| 16384 | `_ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all`        | lgk2 | 4096×14336×16384, 28672×4096×16384 |
| 14336 | `_ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all`  | tv0 | 4096×32768×14336 |
| 8192 / 7168 / 6144 / 2048 | various v12+gm7 | gm2-v12 / tv0 | small-K WIN shapes |

**`per_variant` audit (FINAL JSON)** — for each LOSE shape, did the matching K_EXACT entry actually run?

| LOSE shape | K_EXACT expected | Found in `per_variant`? | TFLOPS observed |
|---|---|---|---|
| 4096×14336×16384 | `_kx16384` (n=14336) | **NO** (.so absent) | n/a → R26-D built it |
| 4096×28672×32768 | `_kx32768` tv0 (n=28672) | **NO** (.so absent) | n/a → R26-D built it |
| 4096×32768×6144 | `_kx6144` (n=32768) | YES | **78 TFLOPS — broken launch** (.so was rebuilt by R26-D) |
| 4096×32768×14336 | `_kx14336` tv0 (n=32768) | **NO** (.so absent) | n/a → R26-D built it |
| 4096×32768×28672 | none correct (only `lgk2` parent built, this shape needs `tv0`) | partial | — |
| 4096×32768×128256 | **does not exist** in source | n/a | — |
| 14336×4096×32768 | `_kx32768` lgk2 (n=4096) | **NO** (.so absent) | n/a → R26-D built it |
| 16384×4096×14336 | `_kx14336` (n=4096, lgk2) | **NO** | n/a — and `ts_u16` parent best, not lgk2 |
| 16384×4096×28672 | `_kx28672` (n=4096, lgk2) | YES | **263 TFLOPS — broken launch** (this is the wire R25-G was tuned ON) |

Conclusion: the 5 R26-D shapes are pure missing-.so wiring bugs.
The remaining 4 split into two classes:
- **Wrong parent** (LOSE 5, 8): K_EXACT only ships one parent per K; the right parent for these specific (M,N) was never gated.
- **Genuinely uncovered K** (LOSE 6 K=128256): no entry exists at all.
- **Failed launch / broken K_EXACT** (LOSE 9 16384×4096×28672): `kx28672` lgk2 ran at 263 TFLOPS in FINAL — needs root-cause inspection (timeout? OOM? unrelated bug?). The shape it was *tuned on*. R26-D didn't audit this one explicitly.

---

## 3. Predicted next-vector matrix

| # | Shape | Best Tag (FINAL) | Bound | K_EXACT match? | Recommended Next Vector | Est Gain |
|---|---|---|---|---|---|---|
| 1 | 4096×14336×16384 | `ts_lgk2_memc_btw_all` | MFMA-issue | NO (.so missing) — R26-D **already built `_kx16384`** | **Re-run FINAL with R26-D builds** (zero code change) | **+15.5pp → 112.3% (verified)** |
| 2 | 4096×28672×32768 | `ts_pf4_memc_btw_step3` | MFMA-issue | NO (.so missing) — R26-D **already built `_kx32768` tv0** | Re-run FINAL with R26-D builds | **+17.2pp → 113.0% (verified)** |
| 3 | 4096×32768×6144 | `ts_gm2_v12_memc_btw_all` | MFMA-issue (margin) | partial (broken launch in FINAL) — R26-D **rebuilt** | Re-run FINAL with R26-D builds | **+16.1pp → 115.7% (verified)** |
| 4 | 4096×32768×14336 | `ts_lgk2_memc_btw_all` | MFMA-issue | NO (.so missing) — R26-D **already built `_kx14336` tv0** | Re-run FINAL with R26-D builds | **+15.0pp → 113.8% (verified)** |
| 5 | 4096×32768×28672 | `ts_gm8_v12_btw_step3` | MFMA-issue (saturated) | wrong parent (only lgk2 built) | **Add `_ts_v12_tv0_memc_dc_gm7_pfoff104_kx28672_btw_all` (clone existing kx28672 with tv0 parent)** | +5-8pp (unverified) |
| 6 | 4096×32768×128256 | `ts_gm8_v12_btw_step3` | MFMA-issue (most saturated) | none exist | **Add `_kx128256` entries for both lgk2 and tv0 parents at pfoff = K_iters - {4..8} = 493-497**; choose the closer one based on dispatch family. Cheap copy-paste. | +5-8pp |
| 7 | 14336×4096×32768 | `ts_pf4_memc_btw_step3` | MFMA-issue | NO (.so missing) — R26-D **already built `_kx32768` lgk2** | Re-run FINAL with R26-D builds | **+18.0pp → 111.7% (verified)** |
| 8 | 16384×4096×14336 | `ts_u16` (vector-16 path!) | MFMA-issue | wrong parent (`u16` family not in K_EXACT set) | **Add `_ts_u16_gm7_pfoff52_kx14336_btw_all` (clone u16 family with gm7+pfoff)**, OR accept (gap is only 2.1pp, near noise) | +1-3pp (uncertain) |
| 9 | 16384×4096×28672 | `v20_memc_btw_step3` | MFMA-issue (saturated) | broken (263 TFLOPS in FINAL) | **Investigate why `_kx28672` (the wire tuned ON this shape!) launched broken; rebuild and 5-rep audit per R26-D protocol** | **+15-20pp** if launch fixes |

### Why `STEP3_BARRIER_VMCNT × M=4096-large-N sweep` is *not* the recommendation
LOSE 5 / 6 use `_btw_step3` (= STEP3 path active) already. R26-C confirmed the
TAIL_VMCNT axis is dead on R25-G stack. Speculative.

### Why `MFMA_32X32X64 (V5)` stays BACKBURNER
R27_V5_MFMA32_SCOUT.md §6 verdict: 1.5-2 weeks engineering + macro re-tune,
estimated p50 at 34-35/42 (modest gain), with regression risk on the 33 currently-WIN
shapes. The R26-D audit alone projects 38/42 with **zero risk**. V5 EV is only
worth funding *after* the cheap wins are realized.

### Why `Tail-MFMA split` is not the recommendation
MFMA-issue saturation at 100-115% means the issue port is the bottleneck — splitting
the tail does not add issue capacity. Would need 32×32 (V5) to actually reduce
total instruction count.

---

## 4. Ranking by `P(flip) × magnitude / cost`

`P(flip)` = probability the recommended action moves shape ≥1pp.
`Magnitude` = expected pp gain.
`Cost` = engineering hours.

| Rank | Shape | P(flip) | Mag (pp) | Cost (hr) | Score |
|---|---|---:|---:|---:|---:|
| **1** | 14336×4096×32768 | 1.00 | +18.0 | 0.1 (re-run) | **180** |
| **2** | 4096×28672×32768 | 1.00 | +17.2 | 0.1 | **172** |
| **3** | 4096×32768×6144  | 1.00 | +16.1 | 0.1 | **161** |
| 4 | 4096×32768×14336 | 1.00 | +15.0 | 0.1 | 150 |
| 5 | 4096×14336×16384 | 1.00 | +15.5 | 0.1 | 155 |
| 6 | 16384×4096×28672 | 0.6  | +18 (if launch fixes)  | 2  | 5.4 |
| 7 | 4096×32768×128256| 0.5  | +6  | 4  | 0.75 |
| 8 | 4096×32768×28672 | 0.5  | +6  | 4  | 0.75 |
| 9 | 16384×4096×14336 | 0.4  | +2  | 4  | 0.20 |

**Top 3 R27 candidates** (essentially all 5 R26-D shapes — re-run is one operation):

1. **Re-run `bench_all42_parallel_R25_FINAL.py`** with the `.so` files
   already built by `r26d_build_missing.py`. **EV: +5 WIN flips → 38/42, ~1 hour
   bench wall-clock, zero risk.** This is the entire R27-A.
2. **Investigate 16384×4096×28672 launch failure** (the FINAL run measured the
   `kx28672` wire — tuned *on this very shape* — at 263 TFLOPS, meaning the
   subprocess loaded the .so but the kernel produced ~50× slowdown). Likely
   driver-state / build-cache poisoning. 5-rep audit per R26-D protocol; if the
   wire actually works clean, it flips this shape to ~+15-20pp WIN. **EV: +1
   flip, ~2 hours, low risk.** This is R27-B.
3. **Add a single new `_kx128256` entry** (tv0 + lgk2 parents) at
   `R25C_TAIL_PF_OFF_ITERS = 497` (= K_iters - 4 = 501 - 4) for shape
   4096×32768×128256. Mechanically identical to the existing R25-G K_EXACT
   recipe. **EV: +1 flip (P=0.5, mag +6pp), ~4 hours.** This is R27-C.

LOSE shapes 5, 8, 9 (`4096×32768×28672`, `16384×4096×14336`, residual
`4096×32768×128256` if R27-C fails) are then candidates for V5 (32×32) only if
no other lever remains. Shape 9 is at 97.9%, near noise — likely accept-and-move-on.

---

## 5. References

- `analysis/fp8_gemm/mi350x/bench_all42_results_R25_FINAL.json` — 9 LOSE rows + per_variant
- `analysis/fp8_gemm/mi350x/bench_all_42.py:519-576` — R25-G/H K_EXACT entries
- `analysis/fp8_gemm/mi350x/bench_all42_parallel_R25_FINAL.py:215-219` — silent-skip on missing .so
- `analysis/fp8_gemm/mi350x/R26D_AUDIT_WIRING_BUG.md` — root-cause + 5-shape verified flips
- `analysis/fp8_gemm/mi350x/r26d_build_missing.py` — 9 .so files added (already on disk)
- `analysis/fp8_gemm/mi350x/R27_V5_MFMA32_SCOUT.md` — V5 BACKBURNER reasoning
- `analysis/fp8_gemm/mi350x/kernel_mxfp4_gluon_cpp.cpp:95-107` — `R25C_K_EXACT` gate
- `analysis/fp8_gemm/mi350x/kernel_mxfp4_gluon_cpp.cpp:529-546` — block/tile constants
- `analysis/fp8_gemm/mi350x/kernel_mxfp4_gluon_cpp-hip-amdgcn-amd-amdhsa-gfx950.s:5778, 5812-5815` — register footprint (256/256 AGPR ceiling)
