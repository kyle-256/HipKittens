# R31 Reviewer Findings

**Date:** 2026-04-18
**Branch:** r31-rev (base feat/mxfp8-only @ `3dddbd5e`)
**GPU:** GPU 0/4/5/6 in serial for Phase 1 (4-GPU triangulation), GPU4 primary for Phase 2 SHIP verification
**Methodology:** Per-process 8s 16k FP16 preheat → 5x in-process bench (warmup=50, iters=100), single shape (70B KV V2-CRR M=4096 N=1024 K=8192). Identical harness to R30 Reviewer (`r29_reviewer_bench5x.py`).
**Build-cache hygiene (R29 Dev C rule):** `rm -f tk_*.so` + `make clean` before each compile + per-build md5 logged in `r31_4gpu_runs/build_md5.log`. **All 4 builds produced bit-identical .so md5 = `095e12e2f7e9300e657b593338749341`** (single shape, single source ⇒ build deterministic). No stale-`.so` contamination.

Source unchanged from R30 base — `3dddbd5e` (R30 cycle wrap) is the head of `feat/mxfp8-only` at the moment of R31 spawn. Therefore any cross-GPU delta in measured perf is environmental (per-GPU DPM / cooling / firmware state), not source-driven.

---

## Phase 1 — 4-GPU baseline triangulation for 70B KV V2-CRR

### Background

R30 Reviewer documented per-GPU dispersion on this shape:
- GPU0 (R30 Dev A baseline) = 791.25 TFLOPS
- GPU4 (R27 Reviewer baseline) = 787.96 TFLOPS
- GPU5 (R30 Reviewer cross-GPU) = 767.23 TFLOPS
- GPU4 (R29 Reviewer baseline) = 766.86 TFLOPS

R30 verdict: a **per-GPU and per-cycle drift** that splits "high regime" (~788) from "low regime" (~767), bridged across GPUs and time but not driven by source. Goal of R31 Phase 1: nail down the live cross-GPU spread *in this cycle* and recommend a GPU-agnostic baseline.

### R31 Phase 1 raw data

Run sequentially on GPUs 4, 5, 6, 0 (GPU0 last to deconflict with R31 Dev A on `/tmp/wt-r31-a`). Each run = identical orchestrate script (`r31_reviewer_4gpu_orchestrate.sh`), same source, same .so md5.

| GPU  | TFLOPS median | mean | stdev | bench SNR (dB) | corr SNR (dB) | det 3/3 | sclk reading (note) |
|---|---:|---:|---:|---:|---:|:---:|---|
| GPU0 | **786.38** | 785.80 | 4.63 | 44.6 | 49.60 | PASS | 2345 MHz (sclk reading via rocm-smi -d 0 = phys GPU0 always; only meaningful here) |
| GPU4 | **767.31** | 766.22 | 2.09 | 51.3 | 49.60 | PASS | (sclk reading is GPU0's clock state, see note below) |
| GPU5 | **764.28** | 763.70 | 2.74 | 48.9 | 49.60 | PASS | (sclk reading is GPU0's clock state) |
| GPU6 | **766.14** | 766.42 | 1.92 | 52.0 | 49.60 | PASS | (sclk reading is GPU0's clock state) |

Build md5 verification (all 4 runs):
```
[build_mxfp8 gpu=4 4096x1024x8192 rc=0 md5=095e12e2f7e9300e657b593338749341]
[build_mxfp8 gpu=5 4096x1024x8192 rc=0 md5=095e12e2f7e9300e657b593338749341]
[build_mxfp8 gpu=6 4096x1024x8192 rc=0 md5=095e12e2f7e9300e657b593338749341]
[build_mxfp8 gpu=0 4096x1024x8192 rc=0 md5=095e12e2f7e9300e657b593338749341]
```

### sclk caveat (paradigm correction proposed)

The R29/R30/R31 bench script calls `rocm-smi --showclocks -d 0`. Because we use `ROCR_VISIBLE_DEVICES=$PHYS_GPU HIP_VISIBLE_DEVICES=0`, the HIP runtime sees the physical GPU as logical 0, but `rocm-smi -d 0` **always reads physical GPU0** regardless of HIP visibility. Therefore the `[sclk-*]` lines printed in the GPU4/5/6 runs reflect GPU0's idle state, not the working GPU's clock. The only sclk line that actually corresponds to the working GPU is the GPU0 run, which shows ~2345 MHz at bench-time vs ~94 MHz idle — confirming the working GPU did boost.

For R32+ harnesses, `rocm-smi -d $PHYS_GPU` (where $PHYS_GPU is the kernel-driver-visible index, set outside the rocr mapping) should be added. This sclk-misread does not invalidate any TFLOPS measurement; it only means we cannot directly verify the DPM state of GPU4/5/6 at bench time from these logs. The convergent within-run stdevs (1.9–4.6 TFLOPS, all bench SNR > 44 dB) are consistent with the GPUs being in a stable post-preheat state regardless of which DPM step.

### Cross-GPU triangulation table (all R31 + historical comparison)

| GPU | R27 | R29 | R30 | **R31** | R31 Δ% vs R27 | R31 Δ% vs R30 |
|---|---:|---:|---:|---:|---:|---:|
| GPU0 | (n/a) | (n/a) | 791.25 | **786.38** | (n/a) | -0.62% |
| GPU4 | 787.96 | 766.86 | (n/a) | **767.31** | -2.62% | (n/a) |
| GPU5 | (n/a) | (n/a) | 767.23 | **764.28** | (n/a) | -0.38% |
| GPU6 | (n/a) | (n/a) | (n/a) | **766.14** | (n/a) | (n/a) |
| **median** | 787.96 | 766.86 | 779.24 | **766.72** | | |
| **mean of 4 R31 medians** | | | | **771.03** | | |

R31 spread: max 786.38, min 764.28 ⇒ **22.1 TFLOPS spread = 2.89%**. This exceeds bench-precision (within-run stdev 0.27%-0.60% ⇒ bench SNR > 44 dB) by an order of magnitude.

### Phase 1 verdict

**Cross-GPU dispersion is real, persistent across R30→R31 cycles, and stratified into two regimes:**
- **High regime ~786-791 TFLOPS:** GPU0 (R30, R31). Difference R30→R31 only -0.62%, well within bench-precision.
- **Low regime ~764-767 TFLOPS:** GPU4 (R29, R31), GPU5 (R30, R31), GPU6 (R31). Three independent GPUs, two cycles, all converge to the 764-767 band.

R30 hypothesized "per-GPU + time component". R31 narrows this: **GPU0 is the outlier** (sits ~2.6% above the GPU4/5/6 cluster), not GPU4/5/6 collectively dropping. Most plausible root cause is per-GPU thermal/firmware/DPM bias for the very-small-N (N=1024) underutilized regime — at 21% wave-fill (R30 Phase 1 closure), a small clock-state difference between GPUs translates directly to TFLOPS differences in proportion to the SIMD DPM step level. GPU0 likely sits a partial step higher, by chance or by node-level airflow asymmetry.

**Live baseline recommendation for 70B KV V2-CRR (R32+):**
- **Median of 4 GPUs = 766.72 TFLOPS** (preferred — robust to GPU0 high outlier).
- Mean of 4 GPU medians = 771.03 TFLOPS (sensitive to outlier; less preferred).
- A SHIP claim for this cell that requires reading on a single GPU should be normalized to the 766-767 regime regardless of which GPU is used. A R31+ Path A rectangular-V2 fastpath that lifts the 0.84 ratio band to ≥0.92 needs to push **above ~840 TFLOPS** (FP8 baseline ~913 TFLOPS × 0.92), independent of which absolute baseline is selected as "live".
- If a SHIP claim is instead measured on GPU0, **discount by 2.6%** before applying the +1% improvement gate; otherwise GPU0's natural high-regime bias is mis-attributed to the change under test.

R30's "real_regression_or_persistent" verdict is now refined: **persistent per-GPU drift, not a regression**. The R27 GPU4 reading of 787.96 is now the "outlier high" — GPU4 in R29/R31 measures the low regime, suggesting that R27's GPU4 was either (a) transiently boosted (cold-DPM artifact) or (b) GPU4 had a similar firmware/DPM state to today's GPU0 in October's window. We have no way to discriminate (a) vs (b) without time-machine data, but the R27 number is no longer the canonical baseline.

### Phase 1 artifacts

- `analysis/fp8_gemm/mi350x/r31_reviewer_kv_4gpu.json` (full structured JSON)
- `analysis/fp8_gemm/mi350x/r31_reviewer_4gpu_orchestrate.sh` (driver script)
- `analysis/fp8_gemm/mi350x/r31_reviewer_4gpu_aggregate.py` (aggregator)
- `analysis/fp8_gemm/mi350x/r31_4gpu_runs/70b_kv_crr_mxfp8_gpu{0,4,5,6}.txt` (raw per-GPU bench logs)
- `analysis/fp8_gemm/mi350x/r31_4gpu_runs/build_md5.log` (4-build md5 hygiene log; all 4 md5 = `095e12e2f7e9300e657b593338749341`)

---

## Phase 2 — Dev SHIP-candidate verdicts

R31 spawned 4 dev agents in parallel (per orchestrator brief):
- Dev A (GPU0): rect-V2 Stage A1 — start of full Path A rectangular BLK_M=256/N=128 V2-CRR fastpath
- Dev B (GPU1): LDS reduction for V2-CRR (R30 Dev B identified LDS, not VGPR, as binding occupancy constraint)
- Dev C (GPU2): PIPELINE_SCALE / scale-pipeline audit V2-RCR
- Dev D (GPU3): persistent-CU dispatch geometry V2-RCR

**No SHIP candidates were submitted to Reviewer for verification by the time Phase 2 was invoked.** Each `/tmp/wt-r31-{a,b,c,d}` worktree at the time of this writing has no `r31{a,b,c,d}_findings.md` and no new commits beyond the R30 wrap (worktrees b/c/d still at `3dddbd5e`; worktree a is at the carry-over R30 Dev A commit `bcb22bd0`). Orchestrator has not signalled a SHIP candidate ready, so the SHIP gate (SNR ≥ 48, det 3/3, MXFP8 ≥ FP8 × 0.95, ≥ baseline + 1%, no regression > -1% in 8192³, Welch t > 3.0) was not exercised this cycle.

**Per-dev verdicts: pending dev completion** — if a dev produces a SHIP candidate after this Reviewer report is committed, a follow-up R31 Reviewer addendum is the appropriate vehicle (orchestrator can re-invoke Phase 2 in a separate R31-rev commit). The Phase 1 baseline established above (766.72 TFLOPS) is the gate value any SHIP claim must clear for 70B KV V2-CRR in R31.

---

## Phase 3 — R32 priorities + cherry-pick recommendations

### Cherry-pick to feat/mxfp8-only

**RECOMMENDED:**

1. **Reviewer artifacts (Phase 1):** `r31_reviewer_kv_4gpu.json`, `r31_reviewer_4gpu_orchestrate.sh`, `r31_reviewer_4gpu_aggregate.py`, `r31_reviewer_findings.md`, and the `r31_4gpu_runs/` log directory. Pure analysis artifacts; no kernel impact. Establishes the 766.72 TFLOPS live baseline.

2. **R31 paradigm correction #1 (sclk reading):** `agent_prompt.md` should record that `rocm-smi -d 0` in any future bench harness misreads the working GPU's clock when ROCR_VISIBLE_DEVICES is used to remap. R32 harness change recommendation: pass the kernel-driver phys index separately and call `rocm-smi -d $PHYS_GPU`. (Cherry-pick as a one-line addition to the R29 Dev C build-hygiene rule in agent_prompt.md.)

**NOT cherry-picked (this cycle):** none — no Phase 2 dev SHIPs to weigh in on.

### R32 priority list (rebuilt from R30 closures + R31 baseline correction)

1. **【critical / 2-3 day】Rectangular BLK_M=256/BLK_N=128 V2-CRR fastpath kernel** (Path A): unchanged from R29/R30/R31 priority #1 — still the only structural lever for 70B KV (0.84 ratio band). R30 Dev A's refined breakdown remains the starting point. Reviewer's Phase 1 baseline correction adjusts the SHIP gate target for this work: **need ≥ 840 TFLOPS** on the GPU-agnostic median to claim the 0.92 ratio (FP8 ~913 TFLOPS as documented in R30 Phase 1).

2. **【medium / immediate】8B Down + 70B Down K-large MLP shapes** (still 0.91-0.94 band, untouched R28-R31). Open levers: (a) PIPELINE_SCALE second-buffer (R28D scaffolding on r28-d), (b) K-axis prefetch bump beyond current `BLK*K_HALF*2` window, (c) per-shape cachepolicy table refinement.

3. **【low / 1 day】4096³ V2-RCR GRID under-occupancy** (R29 Dev C structural finding): defer until rect-V2 (priority 1) lands.

4. **【methodology / R32 standing rule】sclk-d-fix in bench harnesses** (R31 Reviewer): add `--showclocks -d $PHYS_GPU` reading to `r29_reviewer_bench5x.py` (and any descendant) so future cross-GPU runs have direct DPM evidence per working GPU, not GPU0 idle telemetry.

5. **【methodology / R32 standing rule】Cross-GPU baseline normalization for SHIP gates**: any SHIP claim for the 70B KV V2-CRR cell (and analogously for any low-N small-block cell that R32+ identifies as DPM-sensitive) must be measured on at least 2 physical GPUs OR explicitly compared to the per-GPU-of-measurement baseline (GPU0 ≈ 786, GPU4/5/6 ≈ 765-767 in current environment), not the global median 766.72.

### Levers not changed in R31 (R30 closures still hold)

- `GEMM_MIN_BLOCKS_PER_CU > 2 for V2-CRR` (R30 Dev B): LDS-bound. NEVER prototype VGPR-side occ lift for V2-CRR until LDS shrinks below ~80 KB/block.
- `buffer_load_dword_lds for V2 scales` (R30 Dev C): scales have zero LDS hop. NEVER prototype "scale LDS direct path".
- `B-tile load reorder for V2-RCR` (R30 Dev D): correctness-load-bearing.

### Cycle wrap (Reviewer side, Phase 2 may add)

- **0 SHIPs from R31 Reviewer Phase 2 verifications at time of writing** (consistent with R29/R30 cycles' 0-SHIP pattern; pending dev completion in this cycle).
- **R28 SHIP (cachepolicy auto-select)** still in production; no regression detected (no 8192³ baseline check run this cycle — Phase 1 was targeted single-shape per orchestrator brief; R32 should resume periodic 8192³ probe).
- **1 paradigm correction** (R31 sclk-d-fix in bench harness).
- **Live baseline for 70B KV V2-CRR consolidated to 766.72 TFLOPS** (median of 4 GPUs); R27 787.96 reading deprecated as outlier.
- **R32 priorities unchanged** in ordering; only the SHIP gate value for priority 1 (rect-V2) is revised down from the R27-era ~788 → effective ~767 baseline-of-record (median).

### Files

- `analysis/fp8_gemm/mi350x/r31_reviewer_kv_4gpu.json`
- `analysis/fp8_gemm/mi350x/r31_reviewer_4gpu_orchestrate.sh`
- `analysis/fp8_gemm/mi350x/r31_reviewer_4gpu_aggregate.py`
- `analysis/fp8_gemm/mi350x/r31_reviewer_findings.md` (this doc)
- `analysis/fp8_gemm/mi350x/r31_4gpu_runs/` (raw bench logs + build_md5.log)
