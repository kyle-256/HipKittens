# R37 Dev B — HB shrink Stage B1 fan-out to 8B-KV + 8B Gate/Up

**Date:** 2026-04-18
**Branch:** r37-dev-b (worktree /tmp/wt-r37-b, base feat/mxfp8-only @ 098ef0e5 R36 wrap)
**GPU:** primary HIP_VISIBLE_DEVICES=1 (PHYS_GPU=1); 3rd-rep triangulation HIP_VISIBLE_DEVICES=4 (PHYS_GPU=4)
**Scope:** Extend R36 Dev A's HB shrink (BLK_M=128, HB_M=64) Stage B1 cross-buffer DB pattern (PIPE=1) — which delivered +28.02% on 70B-KV — to two additional rect LLaMA shapes:
  - 8B-KV  (M=4096, N=1024, K=4096) — square-K rect, same N=1024 as 70B-KV
  - 8B Gate/Up (M=4096, N=14336, K=4096) — wide-N rect

---

## TL;DR

| Shape | M | N | K | Δ% (median-of-medians) | Min Welch t | SHIP? |
|---|---:|---:|---:|---:|---:|---|
| 8B-KV | 4096 | 1024 | 4096 | **+24.96% (median of +23.88, +24.96, +26.93)** | **+32.1** | **STRICT SHIP** |
| 8B Gate/Up | 4096 | 14336 | 4096 | **-17.26%** (rep 1) / **-17.12%** (rep 2) | -23.30 / -22.61 | **NO SHIP** (predicate must exclude) |

- **8B-KV STRICTLY SHIPS** under the same Stage B1 cross-buffer DB pattern. 3-rep triangulation across GPU1 (rep 1+2) and GPU4 (rep 3) all pass STRICT gate (Δ% ≥ +5, Welch t > 10, SNR ≥ 48 dB det 3/3). VGPR 160 (-74 vs default 234), 0 spill, occ=2, LDS 104 KB — same resource numbers as R36 Dev A's 70B-KV success. Bit-exact with default (max abs diff = 0.0, SNR 49.61 dB).
- **8B Gate/Up does NOT SHIP**, regressing -17.26% (rep 1) and -17.12% (rep 2). Wide-N rect shape behaves like the R36 Dev A 8192³ probe (-25.03%): when N is large, the BLK_M=128 grid-doubling overhead is not amortised by the freed VGPR headroom. Bit-exact correctness PASS (49.61 dB) — this is purely a perf regression.
- **Production predicate wired**: extended `crr_can_use_exact_8wave_scaled_hbshrink` to an explicit allow-list `{(4096,1024,8192), (4096,1024,4096)}`. Production .so verification on GPU4 confirms:
  - 8B-KV: +25.87%, t=+64.7 (predicate fires → HB shrink path).
  - 8B Gate/Up: +0.04%, t=-0.27 (predicate refuses → default V2-CRR path; no regression).

---

## 1. Build pattern (followed R36 Dev A `30d298e8`)

Per-shape rebuild via direct hipcc invocation with `-DPY_MODULE_NAME=...` to produce uniquely-named .so per (shape × variant). Build flags:

- Default V2-CRR baseline: `-DM_DIM=… -DN_DIM=… -DK_DIM=…` (no HB shrink defines).
- HB shrink Stage B1 variant: add `-DMXFP8_CRR_BLK_M=128 -DMXFP8_CRR_HBSHRINK_PIPELINE=1`.

All builds compile clean, no warnings, no spills.

### 1.1 8B-KV (M=4096, N=1024, K=4096)

| Variant | md5 | VGPR | LDS | Spill |
|---|---|---:|---:|---:|
| Default V2-CRR | `8aa732d20caffa254b681ae0b041475a` | 234 | 139.3 KB | 0 |
| HB shrink B1 | `231aacd63ca704dc0a984c2ba36121fc` | **160** | **104.4 KB** | 0 |

Logs: `r37b_default_kv8b_build.log`, `r37b_b1_kv8b_build.log`, plus md5 files.

### 1.2 8B Gate/Up (M=4096, N=14336, K=4096)

| Variant | md5 | VGPR | LDS | Spill |
|---|---|---:|---:|---:|
| Default V2-CRR | `ceb91134fc156b1c6ab66fb6956b04d6` | 234 | 139.3 KB | 0 |
| HB shrink B1 | `526292d87b3178a356012fb4b6723c1e` | **160** | **104.4 KB** | 0 |

Logs: `r37b_default_gateup8b_build.log`, `r37b_b1_gateup8b_build.log`.

---

## 2. Correctness (bit-compare vs default)

`r36a_bitcompare.py` reused with per-shape `MOD_A`, `MOD_B`, `M`, `N`, `K`.

- 8B-KV (`r37b_b1_kv8b_bitcompare.log`): `bit_equal_full=True`, `max_abs_diff=0.0`, SNR 49.61 dB, det 3/3 PASS.
- 8B Gate/Up (`r37b_b1_gateup8b_bitcompare.log`): `bit_equal_full=True`, `max_abs_diff=0.0`, SNR 49.61 dB, det 3/3 PASS.

Numerics gate cleared on both shapes; the regression on 8B Gate/Up is purely structural perf.

---

## 3. Performance (BABA paired bench, R37 paired_bench harness — copy of `r34c_paired_bench.py`)

Protocol per R34/R35/R36 rules:
- 30s preheat (16k × 16k FP16 matmul) before sclk gate.
- 2 warmup pairs + 5 BABA pairs (n=10/kernel).
- Welch two-sample t.
- sclk gate G1 (post-preheat ≥ 2200 MHz) and G2a (post-bench ≥ 2200 MHz) verified per run.

### 3.1 8B-KV — STRICT SHIP (3-rep triangulated)

| Rep | GPU | default median | hbshrink_b1 median | Δ% | Welch t | sclk pre/post |
|---|---|---:|---:|---:|---:|---|
| 1 | GPU1 | 663.11 | 821.48 | **+23.88%** | +57.40 | 2338→2390 PASS |
| 2 | GPU1 | 663.75 | 829.39 | **+24.96%** | +54.68 | 2338→2387 PASS |
| 3 | GPU4 | 698.76 | 886.96 | **+26.93%** | +32.11 | 2356→2387 PASS |

- Median-of-medians (R36 Dev D rule): default 663.75, hbshrink 829.39 → **+24.96%**.
- Min Δ% across 3 reps: **+23.88%** (well above STRICT +5%).
- Min Welch t: **+32.11** (well above STRICT +10).
- Cross-GPU consistency: GPU1 vs GPU4 spread 3.05 pp — well within R33 stochastic-band hypothesis. GPU4 baseline 698.76 is +5.4% higher than GPU1 baseline 663.11, but the ratio (Δ%) is consistent.
- All sclk-post-bench gates PASS.

(Note: rep 3 was first attempted on GPU1 but caught contention — sclk dropped to 1698 MHz post-bench, default median 130 TF, regressed -64.7%. Per R34 sclk-post-preheat ≥ 2200 MHz auto-retry rule combined with R35 G2a sclk-post-bench gate, this rep was DISCARDED as gate-fail; rebenched on idle GPU4 which passed all gates. Discarded log retained at `r37b_b1_kv8b_bench_rep3.log` for audit.)

Logs: `r37b_b1_kv8b_bench.log` (rep 1), `r37b_b1_kv8b_bench_rep2.log` (rep 2), `r37b_b1_kv8b_bench_rep3_gpu4.log` (rep 3).

**SHIP gates**: SNR ≥ 48 dB ✓ (49.61 dB) — det 3/3 ✓ — perf MXFP8 ≥ FP8×95% ✓ (+24.96% over default V2-CRR is strongly above the gate; this rect shape is well over FP8 baseline) — Welch t > 3.0 ✓ (min +32.1).

### 3.2 8B Gate/Up — NO SHIP

| Rep | GPU | default median | hbshrink_b1 median | Δ% | Welch t |
|---|---|---:|---:|---:|---:|
| 1 | GPU1 | 2378.78 | 1968.21 | **-17.26%** | -23.30 |
| 2 | GPU1 (3 warmup pairs) | 2381.96 | 1974.11 | **-17.12%** | -22.61 |

- Both reps confirm the regression at -17.1 to -17.3%.
- Welch t is -22 to -23 (overwhelmingly significant negative). Numerics PASS bit-exact, so this is structural.
- Hypothesis: 8B Gate/Up has N=14336 (vs 1024 in 70B/8B-KV) → 14× more BLK_N=256 column tiles per row. The BLK_M=128 doubling adds grid overhead that the freed VGPR headroom cannot amortise once column tile count grows. This matches the R36 Dev A 8192³ -25% out-of-domain finding (square 8192³ has 32× more ctile slabs than 70B-KV).
- **Domain conclusion**: HB shrink Stage B1 wins on **N=1024 tall-thin rect** shapes only, regardless of K.

Logs: `r37b_b1_gateup8b_bench.log` (rep 1), `r37b_b1_gateup8b_bench_rep2.log` (rep 2).

---

## 4. Production predicate wire-in

Modified `analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_hbshrink_fastpath.inc` lines 683-705 — extended `crr_can_use_exact_8wave_scaled_hbshrink` to add an explicit shape allow-list:

```cpp
__host__ inline bool crr_can_use_exact_8wave_scaled_hbshrink(const layout_globals& g)
{
    if (!(g.m == M_DIM && g.n == N_DIM && g.k == K_DIM))
        return false;
    if (!((g.m % HBSHRINK_BLK_M == 0) && (g.n % BLK == 0) && (g.k % BK == 0)))
        return false;
    // R37 Dev B production shape allow-list — only fire on confirmed SHIP shapes.
    const bool is_70b_kv = (g.m == 4096 && g.n == 1024 && g.k == 8192);
    const bool is_8b_kv  = (g.m == 4096 && g.n == 1024 && g.k == 4096);
    return is_70b_kv || is_8b_kv;
}
```

The edit lives entirely inside `#if (MXFP8_CRR_BLK_M == 128)`, so the default build (BLK_M=256) is structurally unaffected — the entire HB shrink translation unit is empty under the default macro.

### 4.1 Build hygiene (default build kernel resources unchanged)

Direct md5 byte-identity comparison is unreliable on this host — repeated identical builds yield different .so md5s due to embedded build timestamps (verified by 4 consecutive identical default builds: `7c4d0e3645f8…`, `9dcea1cac79f…`, `48e1e18e6c36…`, `11f2aa8c3254…`). Instead, verified hygiene via per-kernel resource-usage remarks:

- Pre-edit default build (`r37b_default_preedit_build.log`): `crr_exact_8wave_scaled_kernel<true,2>` VGPR 234, LDS 139264 B.
- Post-edit default build (`r37b_default_postedit_build_v2.log`): `crr_exact_8wave_scaled_kernel<true,2>` VGPR 234, LDS 139264 B.
- All 6 kernels in the default build are byte-identical at the resource-usage level (RRR/CRR/RCR fastpaths and tail kernels) pre vs post.
- The HB shrink fastpath kernel `crr_exact_8wave_scaled_hbshrink_kernel` does NOT exist in the default build (BLK_M=256 → empty translation unit).

### 4.2 Production behavior verification

Built per-shape production .so with `-DMXFP8_CRR_BLK_M=128 -DMXFP8_CRR_HBSHRINK_PIPELINE=1` AND the per-shape `-DM_DIM=… -DN_DIM=… -DK_DIM=…`:

- 8B-KV production (`r37b_b1_kv8b_prod_bench.log`): +25.87%, Welch t +64.73 — predicate fires, HB shrink active.
- 8B Gate/Up production (`r37b_b1_gateup8b_prod_bench.log`): **+0.04%, Welch t -0.27** — predicate refuses, falls through to default V2-CRR. (Default median 2421.28 vs hbshrink_b1_prod median 2422.26 — within sclk noise; the kernel never dispatched into HB shrink despite the build flags being on.)

The 8B Gate/Up production result definitively confirms the allow-list works: with the HB shrink kernel compiled into the .so but the predicate guarding it, callers see baseline V2-CRR perf on shapes outside the allow-list.

---

## 5. Summary table

| Shape | M | N | K | Δ% (rep 1) | Δ% (rep 2) | Δ% (rep 3) | Min t | SHIP | Production predicate |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 70B-KV (R36 Dev A) | 4096 | 1024 | 8192 | +28.02 | (4-GPU triangulation pending) | — | +96.4 | STRICT (R36) | allow-listed |
| **8B-KV (this work)** | 4096 | 1024 | 4096 | **+23.88** | **+24.96** | **+26.93** | **+32.1** | **STRICT** | **NEW: allow-listed** |
| **8B Gate/Up (this work)** | 4096 | 14336 | 4096 | **-17.26** | **-17.12** | — | -23.3 | **NO SHIP** | refused |
| 8192³ (R36 Dev A out-of-domain probe) | 8192 | 8192 | 8192 | -25.03 | — | — | -121.7 | NO SHIP | refused |

Confirmed domain: HB shrink Stage B1 wins on **rect shapes with N=1024**; loses on **wide-N rect** (N=14336) and **square** (N=8192). The structural pattern: the freed VGPR headroom (-74 vs default) is amortised over the **per-block** workload, so blocks with small N (single ctile column → single launch per M-tile) maximize the win, while large N (many ctile columns per M-tile) is dominated by grid-launch overhead from the BLK_M=128 row doubling.

---

## 6. Methodology notes

- **R36 Dev D median-of-medians applied** for 8B-KV (3 reps). Median-of-medians = 24.96%; min = 23.88%; spread = 3.05 pp. Within healthy range.
- **R34 sclk-post-preheat + R35 G2a sclk-post-bench gates** caught a GPU1 contention event during 8B-KV rep 3 (sclk dropped 1698 MHz, default tflops 130 vs ~660 expected); per the auto-retry rule, the rep was DISCARDED and re-run on idle GPU4 which passed all gates. Audit log: `r37b_b1_kv8b_bench_rep3.log` (gate-fail) vs `r37b_b1_kv8b_bench_rep3_gpu4.log` (gate-pass).
- **R37+ rules carried**: All R29-R36 rules. No new methodology gaps surfaced.

---

## 7. R37+ recommendations

1. **HB shrink Stage B1 production predicate now covers 2 LLaMA cells** — 70B-KV (R36) + 8B-KV (R37). Both shapes share `(M=4096, N=1024)` — predicate is "tall-thin V2-CRR rect with N=1024" parameterised by K∈{4096,8192}. Recommend Reviewer triangulate the 8B-KV result on additional GPUs (this commit covers GPU1+GPU4; ideally 4-GPU triangulation per R34 rule).
2. **Wide-N rect (8B Gate/Up) and big-K (8B-Down K=14336) NOT covered**. Future cycles should:
   - Investigate whether a different BLK_M/BLK_N partitioning recovers wide-N rect (R36 Dev A's recommendation #4 of trying BLK_N=128 to compensate). 
   - Check 70B Gate/Up (M=4096, N=28672, K=8192): if the ratio scales like 8B Gate/Up, expect ~-17%. Suggest skipping prototype and going directly to wider partitioning experiments.
3. **8B Gate/Up correctness PASSES bit-exact** even with the perf regression — the kernel is functionally correct on this shape, just slow. The predicate-based allow-list is the right gating mechanism (don't rely on numerical guards).
4. **Domain finding for paradigm doc**: HB shrink Stage B1's win is **N-bandwidth-bounded**. The +28% / +25% on N=1024 shapes is close to the structural max for 2-warp single-ctile-column launches. As N grows to 14336+ the per-row grid-launch overhead amortises out the VGPR headroom. This matches the R36 Dev A finding that 8192³ regresses -25% — square 8192 has N/BLK=32 ctile columns vs 4 for 8B-KV (1024/256=4) and 4 for 70B-KV (1024/256=4). **Recommended paradigm assertion**: HB shrink predicate must be `N ≤ 1024` (the BLK=256 boundary that splits to ≤4 ctile columns).

---

## 8. Files modified / created

Modified:
- `analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_hbshrink_fastpath.inc` (lines 683-705) — production shape allow-list inside `crr_can_use_exact_8wave_scaled_hbshrink`. Default build unaffected (entire HB shrink TU is empty under BLK_M=256).

Created (this findings doc + scripts + logs):
- `analysis/fp8_gemm/mi350x/r37b_findings.md` — this file
- `analysis/fp8_gemm/mi350x/r37b_paired_bench.py` — copy of `r34c_paired_bench.py` for R37 fan-out reuse
- 8B-KV build/bench/md5/bit-compare logs:
  - `r37b_default_kv8b_{build,md5}.log`
  - `r37b_b1_kv8b_{build,md5,bitcompare,bench,bench_rep2,bench_rep3,bench_rep3_gpu4}.log`
  - `r37b_b1_kv8b_prod_{build,build_v2,md5,md5_v2,bench}.log`
- 8B Gate/Up build/bench/md5/bit-compare logs:
  - `r37b_default_gateup8b_{build,md5}.log`
  - `r37b_b1_gateup8b_{build,md5,bitcompare,bench,bench_rep2}.log`
  - `r37b_b1_gateup8b_prod_{build,build_v2,md5,md5_v2,bench}.log`
- 70B-KV regression-check production .so (no bench, just build to confirm predicate):
  - `r37b_default_kv70b_{build,md5}.log`
  - `r37b_b1_kv70b_prod_{build,build_v2,md5,md5_v2}.log`
- Default build hygiene logs:
  - `r37b_default_preedit_{build,md5}.log` (pre-edit baseline)
  - `r37b_default_postedit_{build,build_v2,md5,md5_v2}.log` (post-edit verification)
