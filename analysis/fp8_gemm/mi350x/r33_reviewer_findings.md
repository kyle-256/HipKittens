# R33 Reviewer Findings

**Date:** 2026-04-18
**Branch:** r33-rev (base feat/mxfp8-only @ `2945ed91`)
**GPU:** Phase 1: GPU 4/5/6/0 (4-GPU triangulation; GPU0 last to deconflict with R33 Dev A on /tmp/wt-r33-a). Phase 2: GPU5/GPU6 for Dev A Task 1 (Dev used 0/4) + Dev C 4 STRICT SHIPs (Dev used 2/3); GPU4/GPU5 for Dev B numerics (Dev used 1/2).
**Methodology:** Per-process 8s 16k FP16 preheat → 5x in-process bench (warmup=50, iters=100), single shape (70B KV V2-CRR M=4096 N=1024 K=8192). `r33_reviewer_bench5x.py` is bit-identical to `r32_reviewer_bench5x.py` (R31 sclk-fix carried forward, no methodology change this cycle).
**Build-cache hygiene (R29 Dev C rule):** `rm -f tk_*.so` + `make clean` before each compile + per-build md5 logged in `r33_reviewer_4gpu_runs/build_md5.log`. **All 4 Phase-1 builds produced bit-identical .so md5 = `b1df8e374fc9b49722b9b85941c6d2e2`** (single shape, single source ⇒ build deterministic). Md5 differs from R32 (`a945fd99…`) because the source moved from R31 wrap `1e1e03a6` → R32 wrap `2945ed91` (R32 SHIPs added the V2-RRR autotune-data note + Dev D rect-V2 RCR Stage A1 scaffolding, which doesn't affect the V2-CRR fastpath but does change the .so byte layout). Source is bit-stable across the 4 R33 Phase-1 builds.

---

## Phase 1 — 4-GPU baseline reverify (3rd cross-cycle data point)

### R33 Phase 1 raw data (sclk-correct, R31 fix carried forward)

Run sequentially on GPUs 4, 5, 6, 0. Identical orchestrate (`r33_reviewer_4gpu_orchestrate.sh`), single source @ `2945ed91`, single .so md5.

| GPU  | TFLOPS median | mean | stdev | bench SNR (dB) | corr SNR (dB) | det 3/3 | sclk pre-bench | sclk post-bench |
|---|---:|---:|---:|---:|---:|:---:|---:|---:|
| GPU0 | **766.63** | 764.84 | 4.99 | 43.69 | 49.60 | PASS | 2352 MHz | 2385 MHz |
| GPU4 | **767.97** | 767.77 | 0.55 | 62.91 | 49.60 | PASS | 2342 MHz | 2381 MHz |
| GPU5 | **764.81** | 764.24 | 2.00 | 51.65 | 49.60 | PASS | 2363 MHz | 2394 MHz |
| GPU6 | **765.71** | 766.08 | 1.15 | 56.49 | 49.60 | PASS | 2333 MHz | 2384 MHz |

Build md5 verification (all 4 runs identical):
```
[build_mxfp8 gpu=4 4096x1024x8192 rc=0 md5=b1df8e374fc9b49722b9b85941c6d2e2]
[build_mxfp8 gpu=5 4096x1024x8192 rc=0 md5=b1df8e374fc9b49722b9b85941c6d2e2]
[build_mxfp8 gpu=6 4096x1024x8192 rc=0 md5=b1df8e374fc9b49722b9b85941c6d2e2]
[build_mxfp8 gpu=0 4096x1024x8192 rc=0 md5=b1df8e374fc9b49722b9b85941c6d2e2]
```

### Cross-GPU triangulation table (R29–R33)

| GPU | R29 | R30 | R31 | R32 | **R33** | R33 Δ vs R32 | R33 Δ vs R31 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPU0 | (n/a) | 791.25 | 786.38 | 768.43 | **766.63** | -1.80 (-0.23%) | -19.75 (-2.51%) |
| GPU4 | 766.86 | (n/a) | 767.31 | 768.37 | **767.97** | -0.40 (-0.05%) | +0.66 (+0.09%) |
| GPU5 | (n/a) | 767.23 | 764.28 | 786.59 | **764.81** | **-21.78 (-2.77%)** | +0.53 (+0.07%) |
| GPU6 | (n/a) | (n/a) | 766.14 | 776.60 | **765.71** | -10.89 (-1.40%) | -0.43 (-0.06%) |
| **median of 4 GPUs** | | | **766.72** | **772.51** | **766.17** | **-6.34 (-0.82%)** | **-0.55 (-0.07%)** |
| **mean of 4 medians** | | | 771.03 | 775.00 | 766.28 | -8.72 (-1.13%) | -4.75 (-0.62%) |

R33 spread: max 767.97, min 764.81 ⇒ **3.16 TFLOPS spread = 0.41%** (R31: 22.10 TF / 2.89%; R32: 18.22 TF / 2.37%; R33 6× tighter than either prior cycle).

### Phase 1 verdict

**The R32 "high regime rotates per-cycle" hypothesis is FALSIFIED by R33.** Specifically:

1. **No R33 GPU is in the high regime.** R33 spread is 0.41% — well within bench precision (~0.3-0.5%). The high-regime outlier (~786 TF, +2.5-2.9% above the cluster) that appeared in **both** R30 (GPU0) **and** R31 (GPU0) **and** R32 (GPU5) **does not exist** in R33.
2. **R32's high-outlier GPU5 (786.59) collapsed -21.78 TFLOPS (-2.77%)** R32→R33 — directly back into the cluster band (764.81). All four GPUs now sit in 764.8–768.0 TF.
3. **GPU6 also dropped -10.89 (-1.40%)** R32→R33 (was +1.37% above R31; now back near R31 level).
4. **GPU0/GPU4 essentially flat** R32→R33 (-0.23%, -0.05%), and very close to their R31 levels too (GPU4 +0.09%, GPU0 -2.51% (still below R31 high-regime)).

**Three-cycle rotating-high-outlier table:**

| Cycle | High GPU | High median | Other-3 median | High excess | Verdict |
|---|---|---:|---:|---:|---|
| R31 | GPU0 | 786.38 | 766.14 | **+2.64%** | high-regime outlier present |
| R32 | GPU5 | 786.59 | 768.43 | **+2.36%** | high-regime outlier present, rotated GPU0→GPU5 |
| **R33** | **(none)** | **767.97** | **765.71** | **+0.30%** | **NO high-regime outlier; all 4 GPUs cluster ~766 TF** |

**Refined hypothesis (R33 Reviewer):** the high-regime ~+2.5% offset is **transient** — it appears in some cycles on some GPU and not in others. R30/R31/R32 each had a high outlier (GPU0, GPU0, GPU5); R33 has none. This is **not** a stable per-cycle "one-of-four" rotation — it is a stochastic firmware/DPM-residency state that some cycle-runs land on for one GPU and other cycles do not. The R32 hypothesis ("rotates one per cycle") was an over-fit to N=2 cycles where exactly one GPU was high; R33 N=3 makes this a clear stochastic per-(GPU × cycle) phenomenon with no guarantee at least one GPU is high.

5. **sclk readings (R31 fix verified):** all 4 GPUs read distinct, plausible per-GPU clocks (GPU0 2352, GPU4 2342, GPU5 2363, GPU6 2333 MHz pre-bench; all in normal DPM step ~level-1 2333-2363 MHz cluster). The integer Mhz tier values are essentially identical across GPUs and across cycles, yet R30/R31/R32 had a >2% spread among GPUs — confirming R32's observation that the high-regime offset is sub-step / boost-residency / firmware-state and **not** visible in the level-1 sclk readout. R33's tight cluster suggests this firmware-state was uniform across the 4 GPUs at the time of measurement.

**Methodology revision (R33 Reviewer):**

- The R32-introduced rule "use median-of-N≥3 GPUs as the SHIP gate baseline" is **strengthened by R33 data**: R31 median-of-4 = 766.72, R32 = 772.51, R33 = 766.17 — three-cycle drift = 6.34 TFLOPS = 0.83% peak-to-peak. R32's 5.79 TF rise looks like it was the cycle-wide bias from one GPU sitting in the high regime; R33 collapses back to near the R31 median (-0.55 TF drift, well within precision).
- The R32 "any single-GPU SHIP claim must be measured on ≥2 distinct physical GPUs and all must clear +1% gate" is **not relaxed** by R33 — even though R33 had no high outlier, there is no a priori guarantee future cycles won't, so the cross-GPU rule should remain.
- **New sub-rule (R33):** if a cross-GPU SHIP run encounters one GPU reading anomalously high (>+1.5% above the other GPUs), prefer the **min-of-GPUs** for the SHIP gate, not the mean — a single high-state GPU is not representative of production deployment.

**Recommended canonical baseline for R34+ (70B KV V2-CRR):**

- **Live baseline: 768 TFLOPS** (rounded mean of R31/R32/R33 medians-of-4: (766.72 + 772.51 + 766.17)/3 = 768.47).
- **SHIP gate target for the rect-V2 path 0.92 ratio:** still ≥840 TFLOPS (FP8 baseline 911-941 TFLOPS × 0.92), unchanged from R31/R32 prescription — the ratio target dominates over the small baseline drift.
- **Per-GPU normalization rule (R32 carried forward, R33 sub-rule added):** any single-GPU SHIP claim must be measured on ≥2 distinct physical GPUs **and** include a same-cycle on-the-same-GPUs baseline reverify (not a previous-cycle baseline) before calling SHIP. The gate is `min(measured_TFLOPS_across_GPUs) ≥ baseline_min × 1.01` AND `Welch t > 3.0` against the same-cycle paired baseline. R33 sub-rule: discount any GPU sitting >+1.5% above the others as a transient high-state outlier; use min-of-GPUs not mean.

### Phase 1 artifacts

- `analysis/fp8_gemm/mi350x/r33_reviewer_kv_4gpu.json` (full structured JSON)
- `analysis/fp8_gemm/mi350x/r33_reviewer_4gpu_orchestrate.sh` (driver script with PHYS_GPU sclk-fix carried forward)
- `analysis/fp8_gemm/mi350x/r33_reviewer_4gpu_aggregate.py` (aggregator with R31/R32/R33 cross-cycle drift comparison)
- `analysis/fp8_gemm/mi350x/r33_reviewer_bench5x.py` (5x bench, identical to R32)
- `analysis/fp8_gemm/mi350x/r33_reviewer_4gpu_runs/70b_kv_crr_mxfp8_gpu{0,4,5,6}.txt` (raw per-GPU bench logs)
- `analysis/fp8_gemm/mi350x/r33_reviewer_4gpu_runs/build_md5.log` (4-build md5 hygiene log)

---

## Phase 2 — Dev SHIP-candidate verdicts

R33 dev branches polled every ~5-10 minutes during Phase 2 window (09:14-10:07 UTC). Final state (4 of 4 dev branches reported):
- **r33-a:** `142370b6 R33 Dev A: SHIP V2-RRR autotune for 70B Down + Stage A2 Path 1 status` — Phase 2 verification REQUIRED (Task 1 SHIP wired-in autotune). Task 2 = NO SHIP (perf, -8.5%).
- **r33-b:** `b81a6e16 R33 Dev B: SHIP Stage A2 — rect-V2 RCR correct numerics @ 4096³` — Phase 2 verification REQUIRED (correctness SHIP). Stage A2d perf = NO SHIP (-33-36%).
- **r33-c:** `1d4bf291 R33 Dev C: per-shape RRR sweep — 6 SHIP candidates identified` — Phase 2 verification REQUIRED for 4 STRICT SHIP cells. SHIP-LITE 2 cells deferred (per Dev C's recommendation).
- **r33-d:** `19d219aa R33 Dev D: sub-RBM operand-tile Stage 1 scaffolding — RBM=32 build status` — no Phase 2 verification (no "SHIP" in commit message; per task brief Dev D is "scaffolding only, no SHIP expected this cycle").

### Verdict 1: R33 Dev A Task 1 — SHIP CONFIRMED (V2-RRR autotune entry for 70B Down 4096×8192×28672)

Dev A wired in a host-side autotune entry inside `dispatch_pq_v2<CRR>` (kernel_mxfp8_layouts.cpp lines 5522-5532) that emits a one-time stderr advisory: "shape (M=4096,N=8192,K=28672) is +12.14% faster on V2-RRR — prefer gemm_rrr_pq_v2". The entry does NOT transparently reroute (V2-CRR's A=(K,M) and V2-RRR's A=(M,K) are incompatible memory layouts; transparent reroute would require a transpose that erodes the +12% gain).

Reverify methodology (cross-GPU per R32 rule):
- Cherry-picked `142370b6` to r33-rev temporarily.
- Built two .so variants with PY_MODULE_NAME override: `tk_mxfp8_r33rev_deva_crr.so` and `tk_mxfp8_r33rev_deva_rrr.so`, both with `M=4096 N=8192 K=28672`. Both rc=0 (md5: crr=`e4af078c…`, rrr=`3f52996…`).
- Ran `r32c_c4_paired_bench.py` on **GPU5 + GPU6** (Dev used GPU0+GPU4 — fully orthogonal cross-GPU triangulation).

| Run                  | CRR median (TF) | RRR median (TF) | Δ%      | Welch t |
|----------------------|---------------:|---------------:|--------:|--------:|
| Dev A GPU0 (n=10)    | 2559.34        | 2854.66        | +11.539% | +56.63  |
| Dev A GPU4 (n=10)    | 2517.50        | 2824.56        | +12.197% | +61.41  |
| **R33 Reviewer GPU5 (n=10)** | **2527.40** | **2833.24** | **+12.101%** | **+76.37** |
| **R33 Reviewer GPU6 (n=10)** | **2515.75** | **2825.51** | **+12.313%** | **+15.72** |

**4-GPU triangulation (full GPU 0/4/5/6 coverage)**: median(Δ%) = +12.15%, range [+11.54, +12.31]; spread 0.77 pp. All correctness PASS (snr_db=49.60-49.61, det 3/3) on all runs. The autotune-entry side-effect (one-time stderr warning) is also verified PRESENT and FIRING ONCE per process on both GPU5 and GPU6 runs.

R32's V2-RRR pivot recommendation (CONFIRMED in R32 Reviewer cycle on GPU2/GPU4/GPU6) is now also CONFIRMED in R33 with the host-side wire-in across GPU0/GPU4/GPU5/GPU6 — 7 distinct GPU runs across 2 cycles all agree on +11.5-12.5%.

**Verdict: CONFIRM SHIP** for Dev A Task 1 (V2-RRR autotune-entry wire-in). The previously-data-only SHIP from R32 Dev C is now host-side-realized in `dispatch_pq_v2<CRR>`. Cherry-pick recommendation to `feat/mxfp8-only`: **YES** — this is exactly what R32's recommendation called for ("autotune-entry recommendation, follow-up commit after this Reviewer note is in"). R33 Dev A is that follow-up.

Artifacts: `analysis/fp8_gemm/mi350x/r33_reviewer_deva_verify/{build_deva_crr,build_deva_rrr}.log`, `build_md5.log`, `c4_gpu{5,6}.txt`.

### Verdict 2: R33 Dev B — SHIP CONFIRMED (rect-V2 RCR Stage A2 numerics @ 4096³)

Dev B fixed Dev D's R32 Stage A1 scaffolding bug (PC=2 B-side scale slab layout, b64 load voff/soff shifts, slab_bytes_b_rect = 64 not 32) such that the rect-V2 RCR fastpath now produces **bit-identical correct output** vs square V2-RCR on 4096³. Stage A2d perf is NO SHIP (-33-36% slower than square via Dev's own bench).

Reverify methodology (cross-GPU correctness only — perf is NO SHIP, no need to triangulate perf):
- Cherry-picked `b81a6e16` to r33-rev temporarily.
- Built rect (`-DM_DIM=4096 -DN_DIM=4096 -DK_DIM=4096 -DMXFP8_RECT_BLK_N=64`): rc=0, md5=`8bea31b222412eb1d3c34d03e60da913`. (Differs from Dev B's rect md5 `434f08cd…` for the same R32-Reviewer-known LLVM `__FILE__`-dependent reason — different worktree paths inject different `__FILE__` strings into kernel-resource-usage remarks. The byte-identical *property* still holds in each environment.)
- Ran `r33b_bench.py rect 4096 4096 4096` on **GPU4** (Dev used GPU1) and **GPU5** (Dev used GPU2 — fully orthogonal cross-GPU triangulation):

| Run | snr_db | pass_rate | det 3/3 | C[0,:8] match Dev's |
|---|---:|---:|:---:|:---:|
| Dev B GPU1 | 49.61 | 100.00 | PASS | (reference) |
| Dev B GPU2 | 49.61 | 100.00 | PASS | (reference) |
| **R33 Reviewer GPU4** | **49.61** | **100.00** | **PASS** | **bit-identical** |
| **R33 Reviewer GPU5** | **49.61** | **100.00** | **PASS** | **bit-identical** |

C[0,:8] = `[-0.53515625, 0.54296875, -0.51171875, -0.11376953125, -0.047119140625, -0.48828125, 0.1474609375, -0.37890625]` on **all 4 GPUs**. SNR matches to 2 decimal places. Determinism PASS on all 4 GPUs.

Bench TFLOPS (informational — perf NO SHIP per Dev's claim):
- R33 Reviewer GPU4: rect 1407.19 TF (vs Dev's GPU1 rect ~1488)
- R33 Reviewer GPU5: rect 1378.37 TF (vs Dev's GPU2 rect ~1534)

The rect kernel is consistently 30-40% slower than square (~2280-2330 TF) on all 4 GPUs — Dev's NO-SHIP perf claim is independently confirmed.

**Verdict: CONFIRM SHIP** for Dev B Stage A2a/b/c (correctness/numerics). Cherry-pick recommendation to `feat/mxfp8-only`: **YES** — Dev D's R32 Stage A1 scaffolding had a B-side load bug (Dev B identified and fixed it); shipping correct rect-V2 RCR numerics is a structural prerequisite for any future rect-V2 RCR perf work. Stage A2d perf NO SHIP recommendation (close rect-V2 RCR perf lever as paradigm closure) is also independently sensible — rect lacks PIPELINE_SCALE/KPAIR_LOOP/PHASE_U16_CACHE/REMAP_ONCE that the square kernel uses.

Artifacts: `analysis/fp8_gemm/mi350x/r33_reviewer_devb_verify/{build_rect.log,md5_rect.txt,a2c_gpu{4,5}.txt}`.

### Verdict 3: R33 Dev C — 4-of-4 STRICT SHIPs CONFIRMED (V2-RRR layout pivots)

Dev C ran a per-shape RRR sweep on the 8 LLaMA cells. Their classification:
- **STRICT SHIP** (Δ ≥ +5% AND Welch t > 10 on BOTH Dev's GPUs): 4 cells.
- **SHIP-LITE** (Δ ≥ +5% on BOTH but t < 10 on at least one): 2 cells (8B Gate, 8B Up @ 4096×14336×4096).
- **NO SHIP** (RCR > RRR on Q/O cells): 2 cells.

Phase 2 verification scope (per task budget): the 4 STRICT SHIPs only. The 2 SHIP-LITE cells are deferred per Dev C's recommendation ("recommend Reviewer 4-GPU"); 4-GPU triangulation on those is a clean R34 follow-up.

Reverify methodology (cross-GPU per R32 rule):
- Cherry-picked `1d4bf291` to r33-rev temporarily.
- Built 4 .so variants (one per cell) with PY_MODULE_NAME override.
- Ran `r33c_paired_bench.py` on **GPU5 + GPU6** (Dev used GPU2+GPU3 — fully orthogonal cross-GPU triangulation).

**Per-cell 4-GPU triangulation table** (Dev's Δ% (Welch t) || Reviewer's Δ% (Welch t)):

| Cell | Shape | Layout pivot | Dev GPU2 | Dev GPU3 | **Rev GPU5** | **Rev GPU6** | Min Δ% | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---|
| c1 70B Gate | 4096×28672×8192 | RRR vs CRR | +7.50% (t=30) | +7.56% (t=54) | **+7.99% (t=29)** | **+7.18% (t=43)** | **+7.18%** | **CONFIRM** |
| c2 70B Up | 4096×28672×8192 | RRR vs CRR | +7.81% (t=32) | +7.20% (t=44) | **+7.37% (t=13)** | **+7.55% (t=11)** | **+7.20%** | **CONFIRM** |
| c4 70B KV | 4096×1024×8192 | RRR vs CRR | +10.68% (t=30) | +10.77% (t=24) | **+10.24% (t=36)** | **+10.83% (t=32)** | **+10.24%** | **CONFIRM** |
| c8 8B KV | 4096×1024×4096 | RRR vs CRR | +8.30% (t=21) | +8.36% (t=22) | **+8.13% (t=34)** | **+8.63% (t=15)** | **+8.13%** | **CONFIRM** |

All correctness PASS (snr_db ≥ 49.59 / det 3/3) on all 16 paired-bench runs (4 cells × 4 GPUs). The min-Δ%-across-GPUs (R33 sub-rule) is +7.18% on c1 — well above the +5% gate. Welch t lowest = 11 (c2 GPU6), still above the +10 STRICT SHIP gate.

**4-GPU triangulation**: each cell measured on 4 distinct physical GPUs (Dev's 2/3 + Reviewer's 5/6). Cross-GPU Δ-medians agree within ±0.81 pp for all 4 cells. The pattern is *exceptionally* consistent — the V2-RRR > V2-CRR layout pivot holds robustly across cycles, GPUs, and shapes (K=4096 c8, K=8192 c4, K=8192 c1/c2).

**Verdict: CONFIRM SHIP** for all 4 Dev C STRICT SHIPs. Combined with R32 Dev C C4 (70B Down 4096×8192×28672) and R33 Dev A Task 1 (autotune wire-in for that R32 cell), the V2-RRR autotune fan-out for `dispatch_pq_v2<CRR>` should now route 5 cells to V2-RRR:
```
if (M==4096 && K==28672 && N==8192)  return Layout::RRR; // 70B Down  (R32, wired R33)
if (M==4096 && K==8192  && N==28672) return Layout::RRR; // 70B Gate  (R33 c1)
if (M==4096 && K==8192  && N==28672) return Layout::RRR; // 70B Up    (R33 c2 — same shape as Gate)
if (M==4096 && K==8192  && N==1024)  return Layout::RRR; // 70B KV    (R33 c4)
if (M==4096 && K==4096  && N==1024)  return Layout::RRR; // 8B  KV    (R33 c8)
```

Note c1/c2 are the same shape (K=8192, N=28672) — only one autotune predicate needed for both. So 4 effective predicates cover 5 cells.

R33 Dev C closures 5/6/7 (V2-RRR layout pivot broadened; V2-RCR remains optimal for square-ish Q/O cells; "RCR > RRR" is independent of K) are also independently sensible and **CONFIRMED** by the cross-GPU triangulation pattern: the 4 SHIP cells are all "non-square" (N ≠ M), the 2 NO SHIP cells are both Q/O (N = M = 4096 or 8192). Pattern holds.

SHIP-LITE cells (not Phase 2 verified this cycle):
- 8B Gate 4096×14336×4096: Dev GPU2 +6.50% (t=6.7), GPU3 +5.71% (t=8.2)
- 8B Up   4096×14336×4096: Dev GPU2 +6.18% (t=6.2), GPU3 +5.25% (t=5.4)

These cells passed Δ ≥ +5% on both Dev's GPUs but failed t > 10 — Reviewer 4-GPU triangulation (deferred to R34) is the appropriate next step before SHIP.

Cherry-pick recommendation to `feat/mxfp8-only`: **YES** — Dev C's full commit (data + 8 build logs + 16 BABA bench results + 4 closures + per-shape autotune recommendation) plus the autotune-entry source patch should land together. R33 Dev A wired one of these (R32 Dev C C4 shape); the R34 cycle should land the remaining 3 STRICT SHIP autotune predicates as a follow-up.

Artifacts: `analysis/fp8_gemm/mi350x/r33_reviewer_devc_verify/{build_*.log, build_md5.log, c{1,2,4,8}_*_gpu{5,6}.txt}` (4 builds + 8 paired-bench runs).

### Verdict 4: R33 Dev D — NO SHIP, no verification needed

Dev D's commit `19d219aa` is "sub-RBM operand-tile Stage 1 scaffolding — RBM=32 build status" — explicit scaffolding, no "SHIP" in commit message, per task brief "(sub-RBM scaffolding only, no SHIP expected this cycle)". No Phase 2 verification work.

---

## Cross-cycle summary (R28 → R33)

### SHIPs in flight (cumulative)

- **R28 SHIP (cachepolicy auto-select N≥28672 && K≥8192 V2-CRR)**: still in production. Not separately re-tested in R33 Phase 1 (single-shape budget).
- **R31 SHIP (rect-V2 CRR Stage A1 scaffolding)**: extended in R32 by Dev D's parallel rect-V2 RCR Stage A1 scaffolding SHIP (note: Dev D's R32 scaffold had a B-side scale slab bug; **R33 Dev B Stage A2a/b/c** fixes that bug and ships correct numerics @ 4096³).
- **R32 SHIP (V2-RRR autotune pivot for 70B Down 4096×8192×28672)**: data SHIP from R32; **R33 Dev A Task 1 wires the host-side autotune entry** in `dispatch_pq_v2<CRR>`. Re-confirmed across 4 GPUs (R32 GPU2/4/6 + R33 GPU0/4/5/6 = 7 distinct GPU runs).
- **R33 SHIP (V2-RRR autotune pivot broadened to 4 STRICT-SHIP cells)**: NEW (Dev C, CONFIRMED across 4 GPUs each). 70B Gate, 70B Up, 70B KV, 8B KV — all 4096×K×N variants with K ∈ {4096, 8192} and N ∈ {1024, 28672}.
- **R33 SHIP (rect-V2 RCR Stage A2 numerics)**: NEW (Dev B, CONFIRMED across 4 GPUs). Stage A2d perf is NO SHIP (rect -33-36% slower than square; Dev recommends closing the rect-V2 RCR perf lever as paradigm closure — R33 Reviewer concurs).

R33 net: **6 new SHIPs CONFIRMED** (Dev A Task 1 + Dev C 4 STRICT + Dev B numerics).

### Methodology bugs surfaced this cycle (R33 Reviewer)

1. **R32's "high regime rotates one-per-cycle" hypothesis FALSIFIED by R33 (N=3 cross-cycle data).** Action: the rotation is **not** a deterministic round-robin (one GPU high per cycle). It is a stochastic per-(GPU × cycle) firmware/DPM state where some cycles have zero high-state GPUs and other cycles have one. **The R31 GPU0-discount rule (already deprecated by R32) remains deprecated; the R32 cross-GPU triangulation rule (≥2 GPUs, all clear +1% gate) remains in force. New R33 sub-rule: prefer min-of-GPUs for SHIP gate (not mean) to be robust against transient high-state outliers.**

2. **No new methodology bug surfaced in R33's Phase 1 baseline run**: sclk readings correct on all 4 GPUs, build md5 byte-identical across all 4 GPUs, correctness PASS on all 4 (snr 49.60 dB / det 3/3), bench SNR ≥43 dB on all (range 43.7-62.9 dB), no determinism failures.

3. **Dev D R32 scaffolding had a real architectural bug (Dev B finding in R33).** Dev D R32's rect-V2 RCR Stage A1 scaffold was "GPU-fault-clean" but produced denormal output — *which the R32 Reviewer reverify also observed* ("Output `C[0, :8]` is denormals (expected — host preshuffle is square layout, rect kernel reads rect-slab geometry).") The R32 Reviewer accepted this as "expected" given the scaffolding's stated scope (compile + GPU-fault-clean only, A1c "correct numerics" out-of-scope). R33 Dev B's deeper analysis identified that Dev D's b64 load against PC=1 slab geometry was the architectural bug (b64 read 8 bytes, slab geometry only had 4 bytes per (lane, k_pair) per pack ⇒ adjacent lanes' 8-byte reads OVERLAPPED). **Lesson for the methodology: a "scaffolding SHIP with denormal output" can mask an architectural-level bug that only a numerics-targeted follow-up identifies.** This is not a new rule — it's a flag that "Stage A1 SHIP" status should be more cautiously interpreted in future cycles. R33 Reviewer recommendation: when Dev claims "scaffolding SHIP" with denormal/wrong output, the next-cycle Dev (or Reviewer) should explicitly call out the residual numerics-bug-risk in their cycle plan.

### R34 priorities (rebuilt)

1. **【high / immediate】Land the 5-cell V2-RRR autotune fan-out** in `dispatch_pq_v2<CRR>` (R32 Dev C C4 + R33 Dev C c1/c2/c4/c8 + R33 Dev A wire-in). 4 effective predicates (c1+c2 share shape). Plus run the LLaMA matrix to verify no regression on neighboring shapes.

2. **【medium / 1-2 day】R34 Reviewer or R34 Dev: 4-GPU triangulation on R33 Dev C SHIP-LITE cells** (8B Gate 4096×14336×4096, 8B Up 4096×14336×4096). Both showed Δ ≥ +5% on both Dev's GPUs but t < 10 — needs ≥4-GPU bench to clear t > 10 STRICT SHIP gate.

3. **【medium / 1-2 day】rect-V2 CRR Stage A2 (numerics)** — R33 Dev A Path 1 establishes square-LDS + halve-N is correct (snr 49.60 dB, det 3/3) but slow (-8.5%). Per Dev A's recommendation, attempt Path 2 (rect LDS HB=64 with K_HALF=0-only helper) or revisit K_HALF=1 helper to remove the OOB indexing.

4. **【close / paradigm】rect-V2 RCR perf lever** — R33 Dev B recommends closure (rect lacks PIPELINE_SCALE/KPAIR_LOOP/PHASE_U16_CACHE/REMAP_ONCE; rect 33-36% slower than square). R33 Reviewer **concurs**. Add to closures list as "rect-V2 RCR perf-via-MXFP8_RECT_BLK_N=64 lever".

5. **【methodology / R34 standing rule】R33 sub-rule**: when running cross-GPU SHIP triangulation, if any GPU reads >+1.5% above the others, prefer min-of-GPUs (not mean) for the SHIP gate. Add to bench harness comments.

### Closures-list cumulative count

R32 closed 21 cumulative levers. R33 adds:
- R33 Dev B closure: rect-V2 RCR perf (lacks PIPELINE_SCALE/KPAIR_LOOP/PHASE_U16_CACHE/REMAP_ONCE; -33-36% slower than square; closure recommended).
- R33 Dev C closures 5/6/7: V2-RRR layout pivot broadened (FALSIFIED R32 K-magnitude-specificity); RCR > RRR for square-ish Q/O cells; pattern holds independent of K.
- R33 Dev A Task 2 closure: rect-V2 CRR Path 1 numerics correct but -8.5% perf.

= **26 cumulative closed levers** (21 R32 + 5 R33 sub-closures across Dev A/B/C). Open levers remaining: rect-V2 CRR Stage A2 Path 2/3 perf (only structural lever for 0.84 V2-CRR ratio band); R33 Dev D sub-RBM operand-tile (still scaffolding, R34 Stage 2 numerics + perf TBD).

### Files (R33 Reviewer artifacts)

- `analysis/fp8_gemm/mi350x/r33_reviewer_findings.md` (this doc)
- `analysis/fp8_gemm/mi350x/r33_reviewer_kv_4gpu.json` (Phase 1 structured data)
- `analysis/fp8_gemm/mi350x/r33_reviewer_4gpu_orchestrate.sh` (driver, sclk-fix carried forward)
- `analysis/fp8_gemm/mi350x/r33_reviewer_4gpu_aggregate.py` (aggregator, R31/R32/R33 cross-cycle drift)
- `analysis/fp8_gemm/mi350x/r33_reviewer_bench5x.py` (5x bench, identical to R32)
- `analysis/fp8_gemm/mi350x/r33_reviewer_4gpu_runs/` (Phase 1 raw bench logs + build_md5.log)
- `analysis/fp8_gemm/mi350x/r33_reviewer_deva_verify/` (Phase 2 Dev A reverify: 2 builds + 2 paired-bench runs on GPU5/GPU6)
- `analysis/fp8_gemm/mi350x/r33_reviewer_devb_verify/` (Phase 2 Dev B reverify: 1 rect build + 2 correctness checks on GPU4/GPU5)
- `analysis/fp8_gemm/mi350x/r33_reviewer_devc_verify/` (Phase 2 Dev C reverify: 4 builds + 8 paired-bench runs on GPU5/GPU6)
