# R38 Reviewer — 8th-cycle baseline (N=4/clean) + Phase 2 SHIP RECONFIRMS (1/2 STRICT, **1/2 CRITICAL FAIL: production wire-in bug**)

## Verdict

- **Phase 1**: 8th-cycle baseline = **777.62 TF (median-of-4 clean GPUs)**, -1.15% below R37's 786.64 (+0.32% above R36's 775.15). R36 NEW 3-gate orchestrate (G1 sclk-post-preheat + G2a sclk-post-bench + G2b per-run CV ≤ 1%) enforced; all 4 clean runs PASS all 3 gates. GPU2 measured high (+1.58% above other-3 median 776.59); below R33's +1.5% threshold by margin but flagged as borderline.
- **Phase 2**:
  - **HB shrink B1 70B-KV (`c_b1_70b`)** — **STRICT RECONFIRM** (min Δ%=+28.07%, min t=+107.9 across GPU2/7). R37 Dev A's `ab8a80f7` production wire-in works as advertised; +30.39% claim is reproducible (this cycle measured +28.07–28.58%, fully within bench-noise envelope of the original).
  - **HB shrink B1 8B-KV (`c_b1_8b`)** — **★★★ CRITICAL FAIL: PRODUCTION WIRE-IN BUG ★★★** — `c_b1_8b` measured **+0.10% on GPU6 and -0.17% on GPU3** through the production wire-in path (NO-LIFT). Investigation confirms: R37 Dev B's commit `46a42d18` extended the dispatch allow-list (`crr_can_use_exact_8wave_scaled_hbshrink` in `crr_mxfp8_exact_8wave_hbshrink_fastpath.inc`) but **never updated the cpp dispatcher at `kernel_mxfp8_layouts.cpp:5741`** which still hard-codes `g.k == 8192`. The kernel itself does deliver (see PATCHED-wire test below: GPU3 +26.66% t=+104.5), but the production wire-in does not route 8B-KV to it. R37 Dev B's claimed +24.96% STRICT SHIP is **non-reproducible at HEAD** as committed.

## Production .so build (per-build hygiene, R29 + R37 Dev C nm-gate)

- Branch: `r38-reviewer` @ HEAD `20f0d640` (R37 cycle wrap)
- Phase 1 (4096×1024×8192) build hashed per-GPU in `r38_reviewer_4gpu_runs/build_md5.log`
- Phase 2 builds: 4 distinct .so (default-CRR + HB-shrink-B1 each at K=8192 and K=4096), all md5'd in `r38_reviewer_devXXX_verify/build_md5.log`
- **R37 Dev C nm-based dead-code gate**: applied to all 4 Phase 2 builds. Default .so always shows `nm | grep hbshrink == 0` (correct, dead-code-eliminated when `MXFP8_CRR_BLK_M` undefined). B1 .so always shows `nm | grep hbshrink > 0` (4 symbols at K=8192, 2 symbols at K=4096 — correct, hbshrink dispatcher present). **The nm-gate proved insufficient to catch this bug**: the b1-K=4096 .so contains the hbshrink symbols (so dead-code gate passes) but the cpp wire-in never calls them at this K. See "Critical finding" section.

## Phase 1 — 4-GPU baseline at branch HEAD `20f0d640`

### Setup
- **Shape**: 70B-KV V2-CRR fastpath (M=4096, N=1024, K=8192) — same canonical shape as R31–R37 cross-cycle baseline.
- **Bench harness**: `r35_reviewer_bench5x.py` (bit-identical, reused).
- **Orchestrate**: `r38_reviewer_4gpu_orchestrate.sh` — R36 NEW 3-gate (G1 + G2a + G2b) + auto-retry up to 3×.
- **GPUs attempted**: 2, 3, 6, 7 (chosen distinct from R38 Dev A/B/C/D fan-out on 0,1,4,5).

### Per-GPU medians (5 iters/GPU, 3-gate enforced)

| GPU | TFLOPS median | TFLOPS stdev | sclk post-preheat | sclk post-bench | stdev/mean | attempts | verdict |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 2 | **789.89** | 2.07 | (PASS) | (PASS) | 0.26% | 1 | G1+G2 PASS |
| 3 | **788.64** | 2.54 | (PASS) | (PASS) | 0.32% | 1 | G1+G2 PASS |
| 6 | **766.59** | 2.06 | 2070→2367 | (PASS) | 0.27% | 2 | G1 retry → PASS |
| 7 | **764.75** | 2.51 | (PASS) | (PASS) | 0.33% | 1 | G1+G2 PASS |

- **median-of-4 clean** = **777.62 TF**
- **min-of-4 clean** = 764.75 TF (GPU7)
- **max-of-4 clean** = 789.89 TF (GPU2)
- **spread** = 25.14 TF (3.29%)
- **high-outlier check** (R33 sub-rule): GPU2 vs other-3 median (776.59) = +1.71% — **borderline-high outlier** (above +1.5% threshold). Per R33: use min-of-4 = 764.75 TF as the conservative ship-claim baseline; report median 777.62 with caveat.
- **GPU pair clustering**: GPU2/3 cluster around ~789 TF; GPU6/7 cluster around ~765 TF. Same bimodal silicon-bin pattern observed throughout R31–R37.

### Cross-cycle baseline drift (R31 → R38, 8 cycles)

| Cycle | GPU0 | GPU1 | GPU2 | GPU3 | GPU4 | GPU5 | GPU6 | GPU7 | median | high outlier (excess%) |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R31 | 786.38 | — | — | — | 767.31 | 764.28 | 766.14 | — | 766.72 (4) | GPU0 (+2.64%) |
| R32 | 768.43 | — | — | — | 768.37 | 786.59 | 776.60 | — | 772.51 (4) | GPU5 (+2.36%) |
| R33 | 766.63 | — | — | — | 767.97 | 764.81 | 765.71 | — | 766.17 (4) | none (+0.30%) |
| R34 | 765.12 | — | — | — | 766.30 | 768.88 | 787.13 | — | 767.59 (4) | GPU6 (+2.71%) |
| R35 | 764.57 | — | — | — | 768.74 | 767.40 | 789.28 | — | 768.07 (4) | GPU6 (+2.85%) |
| R36 | 789.66 | — | — | — | 767.55 | 780.23 | 770.07 | — | 775.15 (4) | GPU0 (+2.54%) |
| R37 | 786.64 | 790.55 | — | 784.61 | 794.88 | (G1 fail) | 767.87 | — | 786.64 (5) | none (+1.05%) |
| **R38** | — | — | **789.89** | **788.64** | — | — | **766.59** | **764.75** | **777.62 (4)** | **GPU2 (+1.71%)** borderline |

- **R38 median 777.62 TF** is between R36 (775.15) and R37 (786.64) — the apparent R37 +1.48% drift was confounded by including GPU1/3 (new GPUs); R38 reuses GPU3 plus introduces GPU2/7, which gives a different mix.
- **R38 vs R37 like-for-like**: GPU3 R37=784.61 vs R38=788.64 (+0.51%) — within bench noise; baseline is statistically stable.
- **R38 vs R36 like-for-like**: GPU6 R36=770.07 vs R38=766.59 (-0.45%); GPU0 absent from R38. No strict regression vs R36.
- **8-cycle median-of-medians** = median(766.72, 772.51, 766.17, 767.59, 768.07, 775.15, 786.64, 777.62) = **771.29 TF** (long-run baseline center).
- **Bimodal silicon-bin hypothesis stands**: GPU2/3 are the "fast bin" (789 TF cluster) this cycle, distinct from GPU6/7 "slow bin" (765 TF cluster). Same pattern as R37 GPU4 high.
- **R36 NEW 3-gate validation continued**: GPU6 attempt 1 caught at sclk-post-preheat 2070 MHz (< 2200 MHz gate); auto-retry attempt 2 PASSED at 2367 MHz. 1/4 GPUs needed retry — gate behavior matches R37 (rare borderline-fail under host load).

## Phase 2 — SHIP RECONFIRMS for R37 still-pending items

### `c_b1_70b` — HB shrink Stage B1 production wire-in @ 70B-KV (M=4096 N=1024 K=8192)

**Comparison**: same `gemm_crr_pq_v2` entrypoint, two .so (default vs HB-shrink B1 with `MXFP8_CRR_BLK_M=128 -DMXFP8_CRR_HBSHRINK_PIPELINE=1`). Identical inputs reused for both. Paired BABA n=10/kernel.

| GPU | Δ% (HB-shrink B1 vs default) | Welch t | default median (TF) | HB-shrink-B1 median (TF) | det |
|:---:|---:|---:|---:|---:|:---:|
| 2 | **+28.577%** | **+107.892** | 792.34 | 1018.77 | PASS / PASS |
| 7 | **+28.071%** | **+175.417** | 767.09 | 982.42 | PASS / PASS |

- **min Δ% = +28.071%** (R37 Dev A claimed +30.39% on GPU0×2/4 — within 2.5% / fully within cross-GPU envelope; this cycle GPU2/7 are clean ship)
- **min Welch t = +107.892** (overwhelmingly above STRICT gate +10)
- **STRICT RECONFIRM** ★★ — R37 Dev A's production wire-in `ab8a80f7` is solidly delivering its claimed lift across GPU2/7 (independent triangulation of original GPU0×2/4).
- **nm-gate**: default .so 0 hbshrink symbols (PASS dead-code), b1 .so 4 hbshrink symbols (PASS feature).

SHIP-CONFIRM history for HB shrink B1 70B-KV:
- R36 Dev A `30d298e8`: GPU3 +28.02% t=+96.4 (1 GPU, original)
- R37 Reviewer: GPU1 +28.79% t=+177.3, GPU7 +27.79% t=+133.8 (2 GPU triangulation)
- R37 Dev A `ab8a80f7`: GPU0×2 + GPU4 +30.39% (production wire-in, 3-rep triangulation)
- **R38 Reviewer (this cycle)**: GPU2 +28.58% t=+107.9, GPU7 +28.07% t=+175.4 (2-GPU re-bench through production wire-in) — **5-cycle stable**

### `c_b1_8b` — HB shrink Stage B1 production wire-in @ 8B-KV (M=4096 N=1024 K=4096)

**Comparison**: same setup as 70B-KV, K=4096.

| GPU | Δ% (HB-shrink B1 vs default) | Welch t | default median (TF) | HB-shrink-B1 median (TF) | det |
|:---:|---:|---:|---:|---:|:---:|
| 3 | **−0.173%** | **−0.501** | 698.41 | 697.20 | PASS / PASS |
| 6 | **+0.095%** | **+0.159** | 660.41 | 661.04 | PASS / PASS |

- **NO-LIFT** on both GPUs. R37 Dev B's claimed +24.96% STRICT SHIP is **NOT REPRODUCIBLE** through the production wire-in path as committed at HEAD.
- **nm-gate**: default .so 0 hbshrink symbols (PASS), b1 .so 2 hbshrink symbols (compiled but unreached).

#### ★★★ CRITICAL FINDING — Production wire-in bug ★★★

**Diagnosis**: R37 Dev B `46a42d18` modified `crr_mxfp8_exact_8wave_hbshrink_fastpath.inc` to extend the `crr_can_use_exact_8wave_scaled_hbshrink` allow-list to include 8B-KV shape (M=4096 N=1024 K=4096). The hbshrink dispatcher template `dispatch_crr_exact_8wave_scaled_v2_hbshrink` is **compiled into the b1 .so** (confirmed by `nm` showing 2 symbols). However, the **cpp dispatcher at `kernel_mxfp8_layouts.cpp:5741`** still has:

```cpp
#if defined(MXFP8_CRR_BLK_M) && (MXFP8_CRR_BLK_M == 128)
        if (g.m == 4096 && g.n == 1024 && g.k == 8192 &&  // ← K=8192 ONLY
            crr_can_use_exact_8wave_scaled_hbshrink(g)) {
            ...
            dispatch_crr_exact_8wave_scaled_v2_hbshrink<true>(g);
            return;
        }
#endif
```

so the K=4096 8B-KV shape never enters the dispatch branch and falls through to the default `gemm_crr_pq_v2` path. **The .inc allow-list extension is necessary but not sufficient — the cpp wire-in needs the K=4096 branch added.**

#### Empirical proof (PATCHED-wire test)

To confirm the underlying kernel does deliver the claimed lift, R38 Reviewer:
1. Ran R37 Dev B's exact `r37b_paired_bench.py` (Dev B's original harness) against the production-wire-in b1 .so → measured +0.14% (consistent with no dispatch fire).
2. Patched cpp wire-in to `(g.k == 8192 || g.k == 4096)`, rebuilt b1 .so as `tk_mxfp8_r38rev_c_b1_8b_b1_test*.so`, re-ran on GPU3 with same harness:

| GPU | Δ% (PATCHED B1 vs default) | Welch t | default median (TF) | PATCHED-B1 median (TF) | det |
|:---:|---:|---:|---:|---:|:---:|
| 3 | **+26.661%** | **+104.467** | 662.13 | 838.66 | PASS / PASS |

3. Reverted cpp patch (Reviewer's job is to flag, not fix). Patched .so retained at `r38_reviewer_devXXX_verify/tk_mxfp8_r38rev_c_b1_8b_b1_test.cpython-310-x86_64-linux-gnu.so` and full bench log at `c_b1_8b_PATCHED_wire_gpu3_bench.txt` for the R38 Dev who lands the fix.

So: kernel works, allow-list works, **wire-in cpp predicate is missing the K=4096 case**.

#### Why nm-gate did not catch this

R37 Dev C's nm-based dead-code gate was correctly designed to catch the symmetric class of bug: a default .so accidentally compiling-in feature symbols (dead-code leak). It flags `nm | grep <feature> > 0` in the default .so as fail. But this bug is the asymmetric case: the feature .so contains the symbol (gate passes) but the cpp dispatcher never reaches it. The nm-gate fundamentally cannot detect "compiled but unreached"; that requires **runtime trace** (e.g., a tracepoint at the dispatcher branch and assertion that it fires for the candidate shape). Recommend R38 add a runtime dispatch-counter or kernel-level "I was here" marker for predicate-fanout SHIPs.

#### Note on first PATCHED-wire attempt

First patched-wire attempt was on GPU2 with R38 Devs running concurrently → result was -68.46% with per-iter values 3.6 to 854 TF (CV 100%+), sclk-post-bench 2131 MHz (gate-failing). That run was discarded under the R36 NEW 3-gate (CV >> 1%, sclk-post-bench < 2200) and the test was re-run on GPU3 with cleaner contention; result above (+26.66% t=+104.5) is the clean one.

SHIP-CONFIRM history for HB shrink B1 8B-KV:
- R37 Dev B `46a42d18`: GPU1×2 + GPU4 +24.96% median-of-medians (ALL .inc-direct tests, did not exercise production wire-in)
- **R38 Reviewer (this cycle) — production wire-in path**: GPU3 −0.17% t=−0.5; GPU6 +0.10% t=+0.16 → **NO-LIFT, BUG FLAGGED**
- **R38 Reviewer (this cycle) — PATCHED wire-in path (cpp K=4096 branch added)**: GPU3 +26.66% t=+104.5 → **kernel-level claim CONFIRMED but production path BROKEN**

## Cross-cycle SHIP-CONFIRM history (compact)

| Cell | R32 | R33 | R34 | R35 | R36 | R37 | R38 |
|---|---|---|---|---|---|---|---|
| 70B-KV V2-RRR (c4) | — | SHIP +10.24-10.83% | CONFIRM | CONFIRM +10.44% | CONFIRM +10.51-10.81% | (predicate stable) | (not tested) |
| 8B Gate V2-RRR (c5) | — | — | LITE +5.23-6.61% | LITE +5.76-5.77% | LITE +5.05-6.96% | (not tested) | (not tested) |
| HB shrink B1 70B-KV | — | — | — | — | SHIP +28.02% (1) | STRICT +27.79-28.79% (2); Dev A wire +30.39% (3) | **STRICT +28.07-28.58% (2)** ★★ |
| V2-RCR 8B Q/O | — | — | — | — | SHIP +7.14-7.22% (3) | STRICT +6.75-7.94% (2) | (not tested this cycle) |
| V2-RCR 70B Q/O | — | — | — | — | SHIP +8.63-9.03% (3) | STRICT +8.90-9.22% (2) | (not tested this cycle) |
| HB shrink B1 8B-KV | — | — | — | — | — | Dev B claim +24.96% (3, .inc-direct) | **★★★ FAIL through production wire-in (NO-LIFT); kernel CONFIRMED via PATCHED wire (+26.66%)** |

## Action items for R38+

1. **★★★ HIGH (immediate)**: Land the cpp wire-in fix for HB shrink B1 8B-KV — change `kernel_mxfp8_layouts.cpp:5741` from `g.k == 8192` to `(g.k == 8192 || g.k == 4096)`, rebuild, re-bench. PATCHED-wire `tk_mxfp8_r38rev_c_b1_8b_b1_test*.so` retained for cross-checking. R38 Dev B (or whoever picks this up) should also add the equivalent for 70B/8B Gate/Up if HB-shrink-N variants land in their fan-out.
2. **★★ HIGH (methodology)**: Add a **runtime dispatch trace** for predicate-fanout SHIPs. The nm-gate (R37 Dev C) catches dead-code leaks but not "compiled but unreached" bugs. Recommend a `MXFP8_DISPATCH_TRACE=1` env that prints which predicate branch fired for a given (M,N,K), so reviewers can verify wire-in routing without re-reading the dispatcher cpp.
3. **★★ MEDIUM**: Adopt `R38 Reviewer rule`: for any predicate-fanout SHIP claim, the original Dev MUST run the production wire-in path (not just the .inc allow-list), and the reviewer MUST verify the dispatcher cpp condition matches the claimed shape set. R37 Dev B's harness `r37b_paired_bench.py` exercised the kernel directly and missed the wire-in gap.
4. **MEDIUM**: GPU2 borderline outlier this cycle (+1.71% above other-3) — silicon-bin hypothesis is consistent across all 8 cycles but warrants a future Dev D investigation: is GPU2/3 a manufacturing batch with slightly higher achievable VGPR clocks at iso-power? Suggest R38 Dev D continue R36's bin-bimodal study with deliberate 8-GPU spread.
5. **LOW**: 8th-cycle baseline median 777.62 TF (median-of-4) is within the long-run 766–787 TF envelope. No regression vs R36 (775.15) or R37 (786.64); the apparent R37 high was a sampling artifact (different GPU mix).

## Files in this commit

- `r38_reviewer_4gpu_orchestrate.sh` — R38 orchestrate (R36 NEW 3-gate, identical to R37's apart from output dir / cycle label)
- `r38_reviewer_ship_verify.sh` — R38 Phase 2 SHIP verify orchestrate (supports `c_b1_70b`/`c_b1_8b`; integrates R37 Dev C nm-based dead-code gate)
- `r38_reviewer_findings.md` — this file
- `r38_reviewer_4gpu_runs/` — Phase 1 runs (4 clean + GPU6 retry + 4 build logs + build_md5.log)
- `r38_reviewer_devXXX_verify/` — Phase 2 verify (4 production-wire bench logs + 1 PATCHED-wire bench log [GPU3 confirms kernel works] + 1 PATCHED-wire failed/contended bench log [GPU2, discarded] + 4 build logs + build_md5.log + retained PATCHED .so for R38 Dev follow-up)
