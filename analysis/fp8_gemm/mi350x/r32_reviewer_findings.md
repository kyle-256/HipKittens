# R32 Reviewer Findings

**Date:** 2026-04-18
**Branch:** r32-rev (base feat/mxfp8-only @ `1e1e03a6`)
**GPU:** GPU 4/5/6/0 in serial for Phase 1 (4-GPU triangulation; GPU0 last to deconflict with R32 Dev A), GPU4 primary for Phase 2 SHIP verification
**Methodology:** Per-process 8s 16k FP16 preheat → 5x in-process bench (warmup=50, iters=100), single shape (70B KV V2-CRR M=4096 N=1024 K=8192). `r32_reviewer_bench5x.py` differs from R29/R30/R31 only in the **rocm-smi -d $PHYS_GPU fix** (R31 paradigm correction applied).
**Build-cache hygiene (R29 Dev C rule):** `rm -f tk_*.so` + `make clean` before each compile + per-build md5 logged in `r32_reviewer_4gpu_runs/build_md5.log`. **All 4 builds produced bit-identical .so md5 = `a945fd9955f444be3da15f443279c113`** (single shape, single source ⇒ build deterministic). Md5 differs from R31 (`095e12e2…`) because the source moved from R30 base `3dddbd5e` → R31 wrap `1e1e03a6` (R31 Dev B/D added `MXFP8_CRR_LDS_SINGLE_BUFFER` and `MXFP8_RECT_BLK_N` gates whose default values keep V2-CRR fastpath behavior unchanged but emit additional bytes). Source is bit-stable across the 4 R32 builds.

---

## Phase 1 — 4-GPU baseline reverify with R31 sclk-fix applied

### R31 paradigm correction applied (R32 verification)

R31 Reviewer surfaced that `rocm-smi --showclocks -d 0` always reads physical GPU0 regardless of `ROCR_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES`. R32 fix:

```diff
- ["rocm-smi", "--showclocks", "-d", "0"]
+ ["rocm-smi", "--showclocks", "-d", str(phys_gpu)]   # PHYS_GPU env var
```

R32 orchestrate exports `PHYS_GPU=$PHYS_GPU` separately from `ROCR_VISIBLE_DEVICES`. **Verified empirically:** `rocm-smi --showclocks -d 0/4/5/6` returns 4 distinct sclk readings simultaneously, and the per-GPU pre/post-bench sclk values now correspond to the working GPU (e.g. GPU4 reads 169 MHz idle → 2319 MHz post-preheat → 2366 MHz pre-bench → 2385 MHz post-bench, all under `PHYS_GPU=4`).

### R32 Phase 1 raw data (sclk-correct)

Run sequentially on GPUs 4, 5, 6, 0. Identical orchestrate (`r32_reviewer_4gpu_orchestrate.sh`), single source @ `1e1e03a6`, single .so md5.

| GPU  | TFLOPS median | mean | stdev | bench SNR (dB) | corr SNR (dB) | det 3/3 | sclk pre-bench | sclk post-bench |
|---|---:|---:|---:|---:|---:|:---:|---:|---:|
| GPU0 | **768.43** | 766.60 | 4.59 | 44.4 | 49.60 | PASS | 2366 MHz | 2390 MHz |
| GPU4 | **768.37** | 767.11 | 2.07 | 51.4 | 49.60 | PASS | 2366 MHz | 2385 MHz |
| GPU5 | **786.59** | 787.57 | 3.68 | 46.6 | 49.60 | PASS | 2359 MHz | 2386 MHz |
| GPU6 | **776.60** | 776.51 | 1.10 | 56.9 | 49.60 | PASS | 2361 MHz | 2397 MHz |

Build md5 verification (all 4 runs identical):
```
[build_mxfp8 gpu=4 4096x1024x8192 rc=0 md5=a945fd9955f444be3da15f443279c113]
[build_mxfp8 gpu=5 4096x1024x8192 rc=0 md5=a945fd9955f444be3da15f443279c113]
[build_mxfp8 gpu=6 4096x1024x8192 rc=0 md5=a945fd9955f444be3da15f443279c113]
[build_mxfp8 gpu=0 4096x1024x8192 rc=0 md5=a945fd9955f444be3da15f443279c113]
```

### Cross-GPU triangulation table (R29–R32)

| GPU | R27 | R29 | R30 | R31 | **R32** | R32 Δ vs R31 |
|---|---:|---:|---:|---:|---:|---:|
| GPU0 | (n/a) | (n/a) | 791.25 | 786.38 | **768.43** | **-17.95 (-2.28%)** |
| GPU4 | 787.96 | 766.86 | (n/a) | 767.31 | **768.37** | +1.06 (+0.14%) |
| GPU5 | (n/a) | (n/a) | 767.23 | 764.28 | **786.59** | **+22.31 (+2.92%)** |
| GPU6 | (n/a) | (n/a) | (n/a) | 766.14 | **776.60** | +10.46 (+1.37%) |
| **median of 4 GPUs** | | | | **766.72** | **772.51** | **+5.79 (+0.76%)** |
| **mean of 4 medians** | | | | 771.03 | 775.00 | +3.97 (+0.52%) |

R32 spread: max 786.59, min 768.37 ⇒ **18.22 TFLOPS spread = 2.37%** (R31: 22.10 / 2.89%; very similar magnitude).

### Phase 1 verdict

**The R31 "GPU0 is the high-regime outlier" hypothesis is FALSIFIED by R32.** Specifically:

1. **GPU0 dropped 17.95 TFLOPS (-2.28%) R31→R32** — directly into the low-regime band that R31 said was the GPU4/5/6 cluster.
2. **GPU5 jumped 22.31 TFLOPS (+2.92%) R31→R32** — became the new high-regime outlier (786.59 vs 764.28).
3. The "high regime" thus rotated cycle-to-cycle: R30 → GPU0; R31 → GPU0; **R32 → GPU5**.
4. The mid-cluster (GPU4, GPU6) stayed flat (R32 vs R31: +0.14%, +1.37%) — well within bench-precision.

This means the high-regime offset is **not a per-GPU bias** (e.g. node-airflow asymmetry); it is **per-(GPU × cycle) DPM-state variance** that picks one GPU per cycle to sit at a higher steady-state DPM step. All 4 GPUs ended R32 at sclk-pre-bench 2359-2366 MHz (very tight cluster) and sclk-post-bench 2385-2397 MHz, so the difference is *not* visible in the level-1 sclk readout — it must be a sub-step / boost-residency / firmware-state effect that doesn't show in the integer Mhz tier readout but does show in 2-3% TFLOPS.

**Methodology revision (R32 Reviewer):**

- **Discount the SHIP-claim "GPU0 high regime" 2.6% normalization** that R31 introduced. R31's prescription was based on N=2 cycles where GPU0 happened to be high; R32 N=3 falsifies it. Use **whichever-of-4-GPUs-is-the-outlier-this-cycle** approach: when running a SHIP claim, run on at least **2 different GPUs**, and if one reads anomalously high (≥+1.5% above the other 3 cluster median), discount it.
- More robustly: **always use the median of N≥3 GPUs** as the SHIP gate baseline, not a single-GPU number. The cross-cycle median-of-4 has been very stable: R31 766.72, R32 772.51 (+0.76%, well within bench-precision when measured across 4 GPUs).

**Recommended canonical baseline for R33+ (70B KV V2-CRR):**

- **Live baseline: 770 TFLOPS** (rounded mean of R31/R32 medians-of-4: (766.72 + 772.51)/2 = 769.62).
- **SHIP gate target for the rect-V2 path 0.92 ratio:** still ≥840 TFLOPS (FP8 baseline 911-941 TFLOPS × 0.92), unchanged from R31 prescription — the ratio target dominates over the small baseline drift.
- **Per-GPU normalization rule (revised):** any single-GPU SHIP claim must be measured on ≥2 distinct physical GPUs **and** include a same-cycle on-the-same-GPUs baseline reverify (not a previous-cycle baseline) before calling SHIP. The gate is `min(measured_TFLOPS_across_GPUs) ≥ baseline_min × 1.01` AND `Welch t > 3.0` against the same-cycle paired baseline.

### Phase 1 artifacts

- `analysis/fp8_gemm/mi350x/r32_reviewer_kv_4gpu.json` (full structured JSON)
- `analysis/fp8_gemm/mi350x/r32_reviewer_4gpu_orchestrate.sh` (driver script with PHYS_GPU sclk-fix)
- `analysis/fp8_gemm/mi350x/r32_reviewer_4gpu_aggregate.py` (aggregator)
- `analysis/fp8_gemm/mi350x/r32_reviewer_bench5x.py` (5x bench with R31 sclk-fix applied)
- `analysis/fp8_gemm/mi350x/r32_reviewer_4gpu_runs/70b_kv_crr_mxfp8_gpu{0,4,5,6}.txt` (raw per-GPU bench logs)
- `analysis/fp8_gemm/mi350x/r32_reviewer_4gpu_runs/build_md5.log` (4-build md5 hygiene log)

---

## Phase 2 — Dev SHIP-candidate verdicts

R32 dev branches polled every 2-3 minutes during Phase 2 window (07:51-08:36 UTC). Final state (4 of 4 dev branches reported):
- **r32-a:** `2ce6844b R32 Dev A NO SHIP — rect-V2 CRR Stage A2c numerics blocked` — no Phase 2 verification (NO SHIP).
- **r32-b:** `49896046 R32 Dev B: V2-CRR LDS SB pipelining recovery — NO SHIP / paradigm closure` — no Phase 2 verification (NO SHIP).
- **r32-c:** `f33e106c R32 Dev C: K-large MLP shapes — V2-RRR SHIP for 70B Down + 3 closures` — Phase 2 verification REQUIRED (SHIP).
- **r32-d:** `bc5a7b0c SHIP Stage A1 — rect-V2 RCR fastpath scaffolded, GPU-fault-clean` — Phase 2 verification REQUIRED (SHIP — scaffolding gates only).

### Verdict 1: R32 Dev D — SHIP CONFIRMED (scaffolding gates)

Dev D shipped Stage A1 (a + b) per their task brief: scaffolding-only SHIP (clean compile, GPU-fault-clean). **Stage A1c (correct numerics) explicitly out-of-scope** for Stage A1 per the orchestrator brief; no perf SHIP gate to evaluate (no paired bench, no Welch t).

Reverify methodology (cross-GPU per R31 rule):
- Cherry-picked `bc5a7b0c` to r32-rev temporarily.
- Built **default** (`-DM_DIM=4096 -DN_DIM=4096 -DK_DIM=4096`) **pre-** and **post-** dev D's source changes — both md5 = `660e0e2a1258b8a36e0e325a306b8823` ⇒ **byte-identical PASS**.
  - Note: dev D claims md5 `7d6c1ae78ee0001b45930835237673e6`. Our absolute md5 differs but our pre↔post comparison still establishes byte-identical (the absolute md5 difference is reproducer-environment dependent — different worktree paths inject different `__FILE__` strings into LLVM kernel-resource-usage remarks, which the dev's `-Rpass-analysis` printout includes verbatim into the .so even though it's a build-time diagnostic). The byte-identical claim **holds in our environment**.
- Built **rect** (`-DMXFP8_RECT_BLK_N=64`): rc=0, `rcr_exact_8wave_scaled_rect_kernel` reports VGPRs=137, AGPRs=0, ScratchSize=0, Occupancy=2, no spills — matches dev D's resource report exactly.
- Ran `r32d_stage_a1b_gpufault_test.py` on **GPU4** (dev used GPU3): `STAGE_A1b_RESULT: NO_FAULT`, kernel completes in ~1ms. Output `C[0, :8]` is denormals (expected — host preshuffle is square layout, rect kernel reads rect-slab geometry).
- Ran the same test on **GPU5** (second cross-GPU triangulation): `STAGE_A1b_RESULT: NO_FAULT`, identical denormal output.

**Verdict: CONFIRM SHIP** (Stage A1a + A1b scaffolding gates).
- Default-build byte-identical: PASS (confirmed in our environment).
- Rect-build clean compile + matching resource report: PASS (137 VGPR / 0 spills / occ=2 — exact match).
- Cross-GPU (GPU4, GPU5) no-GPU-fault: PASS on both.
- Identical-bytes denormal output across GPU4/5 also confirms deterministic execution.

Artifacts: `analysis/fp8_gemm/mi350x/r32_reviewer_devd_verify/{default,rect}_build.log`, `{default,rect}_md5.txt`, `a1b_gpu{4,5}.txt`.

### Verdict 2: R32 Dev C — SHIP CONFIRMED (V2-RRR layout pivot for 70B Down 4096×8192×28672)

Dev C shipped a **layout-pivot autotune entry**: V2-RRR vs V2-CRR for the specific shape (M=4096, N=8192, K=28672). Reported median 2816.55 vs 2511.66 TFLOPS (+12.14%, Welch t = +49.24, n=10, BABA-paired, 60s preheat, GPU2). No source change — only a recommendation to add a per-shape autotune entry routing to `gemm_rrr_pq_v2`.

Reverify methodology:
- Cherry-picked `f33e106c` to r32-rev temporarily.
- Built two .so variants with PY_MODULE_NAME override (matches dev's harness exactly): `tk_mxfp8_layouts_r32rev_c4_crr.so` and `tk_mxfp8_layouts_r32rev_c4_rrr.so`, both with `M=4096 N=8192 K=28672`. Both built rc=0.
- Ran `r32c_c4_paired_bench.py` on **GPU6** (dev used GPU2): see table below.
- Ran the same bench on **GPU4** (second cross-GPU triangulation).

| Run                  | CRR median (TF) | RRR median (TF) | Δ%      | Welch t |
|----------------------|---------------:|---------------:|--------:|--------:|
| Dev C (GPU2, n=10)   | 2511.66        | 2816.55        | +12.14% | +49.24  |
| **R32 Reviewer GPU6 (n=10)** | **2507.29** | **2822.56** | **+12.57%** | **+76.22** |
| **R32 Reviewer GPU4 (n=10)** | **2511.04** | **2817.56** | **+12.21%** | **+28.16** |

Three independent GPUs (2, 4, 6), three independent processes, all agree on RRR≫CRR within 0.4 pp. Both correctness PASS (snr ≈ 49.6 dB, det 3/3) on all runs. sclk stable 2249-2367 MHz pre-bench across all three runs.

Apply R31 SHIP-claim normalization rule (discount GPU0 by 2.6%): N/A — none of the three runs used GPU0. Even with the most aggressive R31 normalization counterfactual (e.g. if we discounted +12.14% → ~+9.5%), the result still vastly exceeds the +1% gate at t > 3.

R32 Phase 1 update to the normalization rule: my Phase 1 falsified the "GPU0 always high" hypothesis (R32 GPU0 was actually -2.28% vs R31 GPU0; R32 GPU5 was the new high outlier at +2.92%). **The R31 GPU0 2.6% discount rule is now obsolete; the safer policy (cross-GPU triangulation on ≥2 GPUs and require all to clear the +1% gate) was applied here and PASSES on 3-of-3 GPUs.**

**Verdict: CONFIRM SHIP** for V2-RRR autotune pivot at (M=4096, N=8192, K=28672). Recommended autotune entry (per Dev C) is a single-shape pivot — do NOT broaden to other K-large shapes (Dev C's bonus 8B-Down probe showed -0.88% at K=14336, no extension).

Cherry-pick recommendation to `feat/mxfp8-only`: **YES** — Dev C's full commit including the autotune-entry recommendation, all 3 closures (C1/C3 cp lever closures, C2 K-iter resource audit), and all paired-bench data. The SHIP is data + recommendation, not yet a kernel patch; the orchestrator can land the actual `dispatch_pq_v2` entry in a follow-up commit after this Reviewer note is in.

Artifacts: `analysis/fp8_gemm/mi350x/r32_reviewer_devc_verify/{build_crr,build_rrr}.log`, `md5_{crr,rrr}.txt`, `c4_gpu{4,6}.txt`.

### Verdicts 3 + 4: R32 Dev A and Dev B — NO SHIP, no verification needed

- **Dev A:** rect-V2 CRR Stage A2c numerics blocked (architectural finding: K_HALF param of `load_col_from_v2_st_half` indexes N-rows, not K-rows; rect mode's HB_N=64 makes the K_HALF=1 register slot read OOB N-rows). NO SHIP, no Phase 2 work.
- **Dev B:** V2-CRR LDS SB pipelining recovery (PIPE=1/2/3 variants on top of R31's PIPE=0). Best variant (PIPE=3) recovers half of R31's loss but still -15-18% vs baseline at Welch t=-64 to -175. Closure: V2-CRR's accumulator-tile shape leaves no headroom for a second concurrent A_col_reg without VGPR spill. NO SHIP, no Phase 2 work.

---

## Cross-cycle summary (R28 → R32)

### SHIPs in flight
- **R28 SHIP (cachepolicy auto-select N≥28672 && K≥8192 V2-CRR)**: still in production. Not separately re-tested in R32 Phase 1 (single-shape budget). Dev C's C1 (which probed 70B Down K=28672 N=8192, falling outside the gate) confirms the gate boundary is correct — 70B Down does NOT need cp=2 (cp=2 is neutral).
- **R31 SHIP (rect-V2 CRR Stage A1 scaffolding)**: extended this cycle by R32 Dev D's parallel rect-V2 RCR Stage A1 scaffolding SHIP (CONFIRMED, this report).
- **R32 SHIP (V2-RRR autotune pivot for 70B Down 4096×8192×28672)**: NEW (Dev C, CONFIRMED, this report). +12.14-12.57% across 3 GPUs.

### Methodology bugs surfaced this cycle (R32 Reviewer)

1. **R31 "GPU0 high outlier" hypothesis FALSIFIED by R32 N=3 data**: GPU0 dropped 2.28% R31→R32 and was no longer the high regime; GPU5 became the new high outlier at +2.92%. **Action: deprecate the R31 "discount GPU0 by 2.6%" SHIP-claim normalization rule.** Replace with: SHIP claims must be measured on ≥2 distinct physical GPUs and all of them must clear the +1% gate.

2. **md5 reproducibility caveat**: dev D's claimed default-build md5 (`7d6c1ae78…`) differs from our reproducer's (`660e0e2a…`) for the same source under the same compile flags. Suspected cause: LLVM `-Rpass-analysis=kernel-resource-usage` remarks include `__FILE__`-dependent line markers that vary by worktree path. The byte-identical pre↔post-edit *property* still holds in each environment; only absolute md5 strings cross-environment do not. **Action: future md5-hygiene logs should include the worktree path + hipcc version, and absolute md5 comparison should be intra-environment only.**

3. **The 30-minute Phase 2 polling window is sufficient for this cycle**: both SHIP candidates landed within 30 minutes of Phase 2 start (Dev D at +9 min, Dev C at +30 min). For future cycles, a 45-minute polling window is recommended as the upper-bound covers spread.

### R33 priorities (rebuilt)

1. **【critical / 2-3 day】rect-V2 CRR Stage A2 (numerics)** — Dev A blocked at K_HALF helper architectural issue. Two paths forward (per Dev A's NO SHIP report): (a) square LDS tile + halve kernel N-work, (b) refactor `load_col_from_v2_st_half` to index K-rows instead of N-rows. Paired with R32 Dev D's rect-V2 RCR Stage A2 (host preshuffle for rect-B + tile coordinate audit).

2. **【medium / immediate】Land the V2-RRR autotune entry for (4096, 8192, 28672)** in the upstream `dispatch_pq_v2` autotune fan-out. Single-line shape predicate per Dev C's recommendation; +12.14% with very tight CI. After landing, re-run the LLaMA matrix to ensure no regression on neighboring shapes.

3. **【low / 1 day】R32 Phase 1 GPU0/GPU5 cross-cycle drift further investigation**: would benefit from per-cycle N=10 same-GPU repeat (current N=1 / cycle is too noisy to characterize the per-cycle DPM rotation precisely). Defer until rect-V2 priority 1 lands.

4. **【methodology / R33 standing rule】R32 normalization-rule deprecation**: any new bench harness must (a) read `rocm-smi --showclocks -d $PHYS_GPU` per the R31 fix, AND (b) require ≥2-GPU SHIP measurement per the R32 deprecation of the GPU0 2.6% discount.

### Closures-list cumulative count

R31 closed 18 levers; R32 closes 3 more (Dev C C1/C2/C3) = **21 cumulative closed levers**. Open levers remaining: rect-V2 CRR/RCR Stage A2/A3 (the only structural lever for the 0.84 V2-CRR ratio band).

### Files (R32 Reviewer artifacts)

- `analysis/fp8_gemm/mi350x/r32_reviewer_findings.md` (this doc)
- `analysis/fp8_gemm/mi350x/r32_reviewer_kv_4gpu.json` (Phase 1 structured data)
- `analysis/fp8_gemm/mi350x/r32_reviewer_4gpu_orchestrate.sh` (driver, sclk-fix applied)
- `analysis/fp8_gemm/mi350x/r32_reviewer_4gpu_aggregate.py` (aggregator)
- `analysis/fp8_gemm/mi350x/r32_reviewer_bench5x.py` (5x bench, sclk-fix applied)
- `analysis/fp8_gemm/mi350x/r32_reviewer_4gpu_runs/` (Phase 1 raw bench logs + build_md5.log)
- `analysis/fp8_gemm/mi350x/r32_reviewer_devc_verify/` (Phase 2 Dev C reverify: 2 builds + 2 paired-bench runs on GPU4/GPU6)
- `analysis/fp8_gemm/mi350x/r32_reviewer_devd_verify/` (Phase 2 Dev D reverify: 2 builds + 2 GPU-fault tests on GPU4/GPU5)

