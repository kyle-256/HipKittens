# R36 Dev A — HB shrink Stage 2 re-pipelining (B1/B2/B3) on V2-CRR

**Date:** 2026-04-18
**Branch:** r36-a (base feat/mxfp8-only @ 40d77d98)
**GPU:** HIP_VISIBLE_DEVICES=3 (physical GPU 3, MI355X / gfx950)
**Scope:** Re-pipeline the R35 Dev B HB shrink (BLK_M=128, HB_M=64) skeleton into a competitive V2-CRR variant. Convert the -66 VGPR / -34 KB LDS headroom into positive perf vs. the default V2-CRR.

---

## TL;DR

- **R35 baseline (recap):** The HB shrink skeleton (single-buffered LDS, no SB pipelining, no LDS interleave) was VGPR 168, 0 spill, occ=2, LDS 104 KB, **bit-exact correctness**, **but -12.16% slower** than default V2-CRR on the 70B-KV shape (M=4096 N=1024 K=8192).
- **R36 Dev A delivered three pipelining variants** behind the existing `MXFP8_CRR_HBSHRINK_PIPELINE` macro switch:
  - **Stage B1: classic cross-buffer DB main loop + B1 LDS interleave + cycle-2 prefetch.**
  - **Stage B2: SB PIPE=3 form** (single-buffered, early-issue VMEM, late-wait).
  - **Stage B3: B1 + B2 hybrid** (PIPE=3 SB + B1 LDS interleave inside the cA chain).
- **Stage B1 is the SHIP** on 70B-KV V2-CRR (4096×1024×8192):
  - Build hygiene PASS — default `.so` byte-identical (md5 `3c060f2536a4d71b41b146eb6c1e6114` pre+post my hbshrink-inc edits).
  - Correctness PASS — full bit-equality with default (max abs diff = 0.0, SNR 49.60 dB, det 3/3).
  - Perf PASS — Stage B1 median **981.05 TF/s** vs. default **766.32 TF/s** → **Δ% = +28.02%** (Welch t = +96.4, sclk steady ≥2393 MHz throughout).
- **Stage B2 also positive** (Δ% = +13.27%, 870.20 vs 768.28 TF/s) but worse than B1.
- **Stage B3 NEGATIVE** (Δ% = -6.27%, 745.34 vs 795.19 TF/s) — combining cycle-2 prefetch with PIPE=3 SB regressed; the in-place LDS prefetch inside the cA chain conflicts with the SB drain semantics.
- **Stage B1 8192³ V2-CRR confirmation:** B1 is **-25.03% slower** on the square shape — the rect 70B-KV win does NOT generalise to square 8192³ (square shape has more BLK_M=128 ctile slabs to process; the cycle-2 prefetch headroom does not amortise across the full sweep).

**Recommendation: SHIP Stage B1 with a 70B-KV-specific predicate** (M=4096, N=1024, K=8192) only — do NOT enable on 8192³. The R36 Reviewer should triangulate Stage B1 across remaining GPUs with the same predicate.

---

## 1. Stage 1 (R35 Dev B) recap and Stage 2 plan

R35 Dev B closed Stage A1-A5 of the HB shrink work:

- A1: scaffolding parallel template `crr_mxfp8_exact_8wave_hbshrink_fastpath.inc` behind `MXFP8_CRR_BLK_M=128`.
- A2: per-warp coverage halved → only cA/cB accumulators (no cC/cD).
- A3: VGPR 234 → 168 (-66), 0 spill, occ=2, LDS -34 KB.
- A4: full bit-equality with default V2-CRR.
- A5: BABA paired bench shows **-12.16% perf** (the skeleton form has no SB pipelining, no LDS interleave, no cycle-2 prefetch — purely VMEM→barrier→LDS→barrier→MMA serial).

R34 Dev D §3.2 Option 1 hypothesis (which R35 Dev B confirmed structurally) was that the freed VGPR/LDS would unblock SB pipelining patterns that were too costly under the production accumulator footprint. The hypothesis remained UNTESTED in R35 — that is the R36 Dev A scope.

### 1.1 Three pipelining variants

The existing inc file already defined three macro-selected variants:

```cpp
#ifndef MXFP8_CRR_HBSHRINK_PIPELINE
#define MXFP8_CRR_HBSHRINK_PIPELINE 0  // 0 = R35 skeleton, 1 = B1 DB, 2 = B2 SB-PIPE3, 3 = B3 hybrid
#endif
```

Each variant is gated only when `MXFP8_CRR_BLK_M=128`. Default builds (no `MXFP8_CRR_BLK_M` define, or `=256`) are unaffected.

---

## 2. Stage B1 — Classic cross-buffer DB + B1 LDS interleave + cycle-2 prefetch

### 2.1 Design

Stage B1 mirrors the production V2-CRR DB main loop pattern adapted to the single-cA-cB accumulator footprint:

1. **Two LDS slots** (`As[2]`, `Bs[2][2]`) with `tic`/`toc` swap each iteration.
2. **Initial prime** loads k=0 into `As[tic]/Bs[tic][0..1]`, then `s_waitcnt vmcnt(0) + s_barrier`.
3. **Main loop** (k=0 to k_iters-2): scale-pack handling (load_raw_scales for even k, shift for odd), `a_sel_packs` selection by `mhalf`, then issue prefetch `k+1` into the OTHER slot (`As[toc]/Bs[toc]`), then read tic-slot LDS (b0, a) and run `crr_hbshrink_cA_with_b1_interleave_fixed_phase` (this template loads b1 from LDS at INSERT_AFTER position INSIDE the cA MMA chain), then `crr_mma_scaled_from_packs_fixed_phase(cB, ...)`, then `s_waitcnt vmcnt(0) + s_barrier`.
4. **Final tail** (k=k_iters-1): no prefetch.

Critical change vs. R35 skeleton:

- **Cycle-2 prefetch is back** (B1 issues k+1 VMEM→LDS *before* the MMA chain so VMEM overlaps MMA), but writes go into the TOC slot (different from TIC being consumed) → no in-place race.

### 2.2 The numerics bug in the first attempt (and the fix)

My initial Stage B1 v1 used **in-place LDS prefetch** (write to `As[tic]` / `Bs[tic]` after consuming them, all in the same iteration). Per-iter perf jumped to ~1000 TF/s (+30.96%), but correctness FAILED — SNR 5.29 dB, max abs diff 4.02, det_ok=False. The issue: while the LDS reads complete before the writes are issued, the consumer reads from the SAME slot and the producer writes there — under PIPE=k re-ordering, MMAs from iter k can race against LDS writes for iter k+1 in the same `tic` slot.

**Fix:** classic cross-buffer DB pattern — read TIC, prefetch into TOC. Confirmed via bit-compare:

```
C_default[0,:8]   = [2.1875, -1.4765625, 0.37109375, -1.0703125, ...]
C_hbshrink[0,:8]  = [2.1875, -1.4765625, 0.37109375, -1.0703125, ...]
bit_equal_C[0,:8] = True
bit_equal_full    = True
max_abs_diff      = 0.0
mean_abs_diff     = 0.0
```

### 2.3 Resource numbers (Stage B1)

```
crr_exact_8wave_scaled_hbshrink_kernel<true,2>:
  TotalSGPRs:  46
  VGPRs:       160       (vs. default 234 → -74)
  AGPRs:       0
  ScratchSize: 0
  Occupancy:   2 waves/SIMD
  VGPRs Spill: 0
  LDS:         104448 B/block  (vs. default 139264 → -34816 B)
```

VGPR went from R35 skeleton 168 down to **160** (-8 more saved by the DB form vs. SB), 0 spill maintained, occupancy 2 unchanged, LDS 104 KB unchanged.

### 2.4 BABA paired bench (70B-KV V2-CRR, GPU 3)

Files:
- Build: `r36a_b1_70bkv_build_v2.log` (md5 `3c320b0662ef0b53bc8c699c85860106`)
- Bit-compare: `r36a_b1_70bkv_bitcompare_v2.log` (full bit-equal)
- Bench: `r36a_b1_70bkv_bench_v2.log`

```
Default V2-CRR:    median 766.32 TF/s, mean 764.79, stdev 3.43, n=10
HB shrink B1:      median 981.05 TF/s, mean 980.49, stdev 6.18, n=10
Welch t = +96.444
Δ% (B1 vs default) = +28.021%
sclk: pre=2407 MHz, post=2404 MHz (steady, no throttle)
```

**SHIP eligible.** Δ% ≥ 0% gate cleared with massive headroom. Welch t is overwhelming.

---

## 3. Stage B2 — SB PIPE=3 (no DB)

### 3.1 Design

Stage B2 keeps single-buffered LDS (use `As[0]`/`Bs[0]` only) but issues the next-iter VMEM writes BEFORE the MMA pair, deferring `s_waitcnt vmcnt(0)` to the iteration tail. This is the R32 Dev B Approach 3 PIPE=3 form adapted to the single cA/cB footprint.

### 3.2 Resource numbers (Stage B2)

```
crr_exact_8wave_scaled_hbshrink_kernel<true,2>:
  VGPRs:       177       (-57 vs. default 234)
  Occupancy:   2 waves/SIMD
  VGPRs Spill: 0
  LDS:         104448 B/block
```

### 3.3 BABA bench (Stage B2)

Files: `r36a_b2_70bkv_build.log` (md5 `71da713cb0a9dd3881749dc1ae7c5f10`), `r36a_b2_70bkv_bitcompare.log` (bit-equal), `r36a_b2_70bkv_bench.log`.

```
Default V2-CRR:    median 768.28 TF/s, mean 766.49, stdev 5.72
HB shrink B2:      median 870.20 TF/s, mean 870.34, stdev 1.72
Welch t = +55.011
Δ% (B2 vs default) = +13.266%
sclk steady at 2400-2407 MHz
```

Positive but **less than B1**. PIPE=3 SB without DB cannot fully hide VMEM latency on the rect shape — the per-iter drain barrier still serialises VMEM with the next MMA. B1's cross-buffer DB lets MMA/LDS-of-tic overlap with VMEM-into-toc for the full iteration.

---

## 4. Stage B3 — B1 + B2 hybrid (PIPE=3 SB + B1 LDS interleave)

### 4.1 Design

Stage B3 keeps the SB single-slot scheme of B2 but adds the B1 LDS interleave (defer b1 LDS read into the cA MMA chain). Idea: hide the second LDS read latency under cA MMAs.

### 4.2 Resource numbers (Stage B3)

```
crr_exact_8wave_scaled_hbshrink_kernel<true,2>:
  VGPRs:       168       (-66 vs. default 234)
  Occupancy:   2 waves/SIMD
  VGPRs Spill: 0
  LDS:         104448 B/block
```

### 4.3 BABA bench (Stage B3)

Files: `r36a_b3_70bkv_build.log` (md5 `99db91eb66aee29f0194721fa670ee95`), `r36a_b3_70bkv_bitcompare.log` (bit-equal), `r36a_b3_70bkv_bench.log`.

```
Default V2-CRR:    median 795.19 TF/s, mean 792.90, stdev 6.53
HB shrink B3:      median 745.34 TF/s, mean 745.24, stdev 1.50
Welch t = -22.500
Δ% (B3 vs default) = -6.269%
```

**NEGATIVE.** Combining the SB form with the B1 interleave regressed below baseline. Hypothesis: the B1 interleave inside the cA chain changes the LDS issue ordering vs. the VMEM prefetch in B2 — the deferred b1 read may cause the next-iter VMEM write to race with the lingering b1 LDS read on the same slot.

---

## 5. Stage B1 8192³ V2-CRR confirmation (out-of-domain probe)

To check whether the +28% rect-shape win generalises, I rebuilt Stage B1 for the square 8192³ shape:

Files: `r36a_default_8k_build.log` (md5 `8118af3570f7f4504899f277d72de364`), `r36a_b1_8k_build.log` (md5 `186cdcc0d9ba20019bedaf8eb88ab0f1`), `r36a_b1_8k_bitcompare.log` (bit-equal), `r36a_b1_8k_bench.log`.

```
Default V2-CRR:    median 2444.84 TF/s, mean 2439.43, stdev 15.50
HB shrink B1:      median 1832.90 TF/s, mean 1833.05, stdev 2.80
Welch t = -121.737
Δ% (B1 vs default) = -25.030%
```

**B1 does NOT win on 8192³.** The square shape has 32× more ctile slabs (8192/256 vs. 1024/256 = 4×, then squared = 16× × 2 from BLK_M halving). The default's cC/cD-rich kernel amortises VMEM latency across more inner iterations per launch; the BLK_M=128 halving doubles the launch grid count, and the per-launch pipeline gain doesn't make up for the doubled scheduling overhead.

**Implication for productionization:** Stage B1 SHIP must be predicated on the 70B-KV shape (M=4096, N=1024, K=8192) only. The kernel_mxfp8_layouts.cpp dispatch should NOT enable HB shrink for square shapes.

---

## 6. Summary table

| Variant | VGPR | LDS (KB) | Bit-equal | TF/s (median) | Δ% vs default | Welch t | Verdict |
|---|---|---|---|---|---|---|---|
| Default V2-CRR (70B-KV) | 234 | 139.3 | (ref) | 766.32-795.19 | 0% | n/a | (baseline) |
| R35 skeleton (PIPE=0) | 168 | 104.4 | YES | 675.28 | -12.16% | -71.3 | NO SHIP |
| **B1 DB (PIPE=1)** | **160** | **104.4** | **YES** | **981.05** | **+28.02%** | **+96.4** | **SHIP** |
| B2 SB-PIPE3 (PIPE=2) | 177 | 104.4 | YES | 870.20 | +13.27% | +55.0 | dominated by B1 |
| B3 hybrid (PIPE=3) | 168 | 104.4 | YES | 745.34 | -6.27% | -22.5 | NO SHIP |
| B1 on 8192³ | 160 | 104.4 | YES | 1832.90 | -25.03% | -121.7 | NO SHIP (out-of-domain) |

---

## 7. Build hygiene

- **Default-off invariant verified.** Pre-edit (HEAD) and post-edit default builds both produce md5 `3c060f2536a4d71b41b146eb6c1e6114` for `tk_mxfp8_kv_default.cpython-310-x86_64-linux-gnu.so`. Logs: `r36a_default_build_postedit.log` / `r36a_default_build_unstashed.log` (byte-identical).
- All HB shrink builds compile with no warnings and no spills.
- Numerics: SNR 49.60 dB at 70B-KV, 49.59 dB at 8192³; pass_rate 100%; deterministic 3/3.
- BABA protocol: 30s preheat, 2 warmup pairs, 5 paired BABA samples (n=10/kernel), Welch t two-sample, sclk auto-checked pre/post each pair. No throttle observed.

---

## 8. Recommendations to R36 Reviewer

1. **SHIP Stage B1 with a 70B-KV-only predicate.** Add a dispatch guard that selects HB shrink Stage B1 only when (M=4096, N=1024, K=8192) AND `MXFP8_CRR_BLK_M=128`. Defer to default V2-CRR for all other shapes.
2. **Triangulate Stage B1 on remaining GPUs** (the R36 Reviewer 4-GPU orchestrate). The +28% on GPU 3 is far above the +1% SHIP threshold, but cross-GPU consistency should be verified.
3. **Investigate the B3 regression in a follow-up cycle** (Stage 3). The hypothesis that "B1 + B2 = best" was wrong — the interleave + SB-only combination appears to expose a hazard. Worth bisecting whether deferring the b1 LDS to the cA chain is the cause vs. the absence of cross-buffer DB.
4. **Square-shape (8192³) follow-up.** Test whether a different BLK_N partitioning (e.g. BLK_N=128 instead of 256) recovers the win on square shapes; current Stage B1 uses BLK_M=128 × BLK_N=256, so the column count is unchanged from default, but row count doubles.

## 9. Files modified / created

Modified:
- `analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_hbshrink_fastpath.inc` — Rewrote Stage B1 (PIPELINE==1) section to use classic cross-buffer DB instead of in-place prefetch; existing Stage B2 / Stage B3 sections retained as-is from the file's pre-existing macro skeleton.

Created (logs/scripts):
- `analysis/fp8_gemm/mi350x/r36a_bitcompare.py` — generalised bit-compare wrapper (MOD_A/MOD_B/SO_A/SO_B/M/N/K env vars)
- `analysis/fp8_gemm/mi350x/r36a_findings.md` — this file
- Build/bench/md5/bit-compare logs:
  - `r36a_default_build_postedit{.log,_md5.log}`, `r36a_default_build_unstashed{.log,_md5.log}` (byte-identity proof)
  - `r36a_default_70bkv_build.log`, `r36a_default_70bkv_md5.log`
  - `r36a_b0_70bkv_build.log`, `r36a_b0_70bkv_md5.log`
  - `r36a_b1_70bkv_build_v2.log`, `r36a_b1_70bkv_md5_v2.log`, `r36a_b1_70bkv_bitcompare_v2.log`, `r36a_b1_70bkv_bench_v2.log`
  - `r36a_b2_70bkv_build.log`, `r36a_b2_70bkv_md5.log`, `r36a_b2_70bkv_bitcompare.log`, `r36a_b2_70bkv_bench.log`
  - `r36a_b3_70bkv_build.log`, `r36a_b3_70bkv_md5.log`, `r36a_b3_70bkv_bitcompare.log`, `r36a_b3_70bkv_bench.log`
  - `r36a_default_8k_build.log`, `r36a_default_8k_md5.log`
  - `r36a_b1_8k_build.log`, `r36a_b1_8k_md5.log`, `r36a_b1_8k_bitcompare.log`, `r36a_b1_8k_bench.log`
