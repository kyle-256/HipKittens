# R30 Reviewer Findings

**Date:** 2026-04-18
**Branch:** r30-rev (base feat/mxfp8-only @ `cd630fd8`)
**GPU:** GPU4 owned by Reviewer + **GPU5 cross-GPU reverify** for Phase 1
**Methodology:** Per-process 8s 16k FP16 preheat → 5x in-process bench (warmup=50, iters=100), 2-sigma outlier filter, MXFP8 V2 (`gemm_*_pq_v2`) vs FP8 per-tensor. Identical to R29 Reviewer harness (`r29_reviewer_bench5x.py`).
**Build-cache hygiene (R29 Dev C rule):** `rm -f tk_*.so` before each compile + per-build md5 logged in `r30_crossgpu_runs/build_md5.log`. Six unique md5s observed across the six (kind × shape) builds — no stale-`.so` contamination.

Source unchanged from R29 base — `cd630fd8` (R29 cycle wrap) is the head of `feat/mxfp8-only` at the moment of R30 spawn. Therefore any cross-cycle delta in measured perf is environmental (GPU state / DPM / driver / scheduler), not source-driven.

---

## Phase 1 — Cross-GPU reverify of R29 negative-t cells

### Background

R29 Reviewer (`r29_reviewer_findings.md`) flagged 3 cells with statistically significant MXFP8 absolute drops on GPU4 vs R27 GPU4 (no source change since R28):

| Cell | Shape | R29 Δpct vs R27 | R29 Welch t | R29 Verdict |
|---|---|---:|---:|---|
| 8b_gate_crr | 4096×14336×4096 | -0.63% | -4.11 | flagged |
| 8b_down_crr | 4096×4096×14336 | -2.42% | -3.66 | flagged |
| 70b_kv_crr  | 4096×1024×8192  | -2.68% | -8.51 | flagged (largest) |

**Hypothesis:** GPU4-state / DPM artifact, not real source-driven regression. Re-running on a different physical GPU (GPU5) under identical bench conditions should reproduce the R27 numbers if hypothesis holds.

### Phase 1 raw data

Run on **GPU5** via `ROCR_VISIBLE_DEVICES=5 PHYS_GPU=5 ./r30_reviewer_crossgpu_orchestrate.sh`. Outputs in `r30_crossgpu_runs/`. Build md5s (all distinct, build-cache hygiene confirmed):

```
build_fp8   4096x14336x4096 md5=add15a2e84d5b5c4877b9bf1677b0b46
build_mxfp8 4096x14336x4096 md5=9622ab4be9a221e0a2e84e1793a417d8
build_fp8   4096x4096x14336 md5=6316157befc499c7e909fbb0c1e8e513
build_mxfp8 4096x4096x14336 md5=6b5b456949986c83ad71cab57370ca64
build_fp8   4096x1024x8192  md5=839eaa4d9813715a5b4cc8ba9acf098c
build_mxfp8 4096x1024x8192  md5=ac063d2ac676f1f88653939e296d0777
```

All 6 cells: SNR ≥ 49.60 dB, determinism 3/3 PASS.

### Phase 1 cross-cycle comparison table

MXFP8 V2 medians (TFLOPS) and Welch t-tests (R30 GPU5 vs R29 GPU4 / R27 GPU4):

| Cell | R27 GPU4 | R29 GPU4 | **R30 GPU5** | R30 stdev | Δ% vs R27 | t vs R27 | Δ% vs R29 | t vs R29 | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8b_gate_crr | 2405.79 | 2390.72 | **2382.04** | 2.09 | -0.99% | -9.23 | -0.36% | -4.14 | **inconclusive (small persistent shift)** |
| 8b_down_crr | 2779.39 | 2712.16 | **2763.81** | 65.27 | -0.56% | -0.60 | +1.90% | +1.50 | **gpu_state_artifact** |
| 70b_kv_crr  |  787.96 |  766.86 | **767.23**  | 2.35 | -2.63% | -8.24 | +0.05% | +0.56 | **real_regression_or_persistent** |

For comparison, FP8 medians (no source change either):

| Cell | R27 FP8 | R29 FP8 | R30 FP8 | t (FP8) vs R27 | t (FP8) vs R29 |
|---|---:|---:|---:|---:|---:|
| 8b_gate_crr | 2610.48 | 2660.37 | 2604.36 | -0.89 | -12.84 |
| 8b_down_crr | 2958.56 | 2993.91 | 3028.38 | +2.81 |  +1.06 |
| 70b_kv_crr  |  911.31 |  940.81 |  913.42 | +1.25 | -22.79 |

### Per-cell verdict

**1. `8b_down_crr` — `gpu_state_artifact` (R29 was wrong about this one)**

R30 GPU5 MXFP8 median 2763.81 matches R27 GPU4 2779.39 within bench-to-bench noise (-0.56%, Welch t=-0.60). The R29 GPU4 figure of 2712.16 (-2.42% vs R27, t=-3.66) was the outlier — most likely a transient GPU4-state/DPM effect or cooling drift specific to R29's measurement window. R30 confirms no source-driven regression.

**Action:** drop from R30+ watch list. The R28 cherry-picks did not regress this shape.

**2. `70b_kv_crr` — `real_regression_or_persistent` (R29 confirmed by R30)**

R30 GPU5 MXFP8 median 767.23 is statistically indistinguishable from R29 GPU4 766.86 (Δ +0.05%, Welch t=+0.56), and both differ from R27 GPU4 787.96 by ~-2.63% (R30 t vs R27 = -8.24, R29 t vs R27 = -8.51). The drop is reproducible on a fresh GPU.

FP8 on the same cell is flat across all three cycles (R30 vs R27 t=+1.25, ratio 0.84 vs R27 0.8646), so the ratio drop is MXFP8-specific. Two consecutive cycles measuring -2.63% on different GPUs with the same source ⇒ this is **not** a per-cycle GPU-state artifact. Likely root causes (in order of plausibility):
   1. Cross-cycle environmental shift (driver / firmware / cooling between the R27 run and R29) that disproportionately affects very small N=1024 shapes where the V2-CRR kernel is grid-underoccupied (32 blocks × 8 waves = 256 waves vs 1216 SIMDs ⇒ 21% utilization, very sensitive to DPM step-down on idle SIMDs).
   2. The R27 GPU4 measurement may itself have been transiently high due to a cold-DPM artifact (R26 documented this exact cell as cold-DPM-sensitive — `agent_prompt.md` R26 paradigm correction #1 had to invalidate the R25 KV CRR cell entirely for the same reason).
   3. Some lower-precedence factor (cache pre-warm, neighbor-CU activity) that we cannot isolate without controlled clock pinning.

The R29 Reviewer's hypothesis "GPU4-state effect" is **partially refuted** for this cell: the same number reproduces on GPU5. Calling the absolute value "the new baseline" is more honest than calling it a regression. The MXFP8 perf at this shape is genuinely 767 TFLOPS in our current environment.

**Action:** treat 767 TFLOPS as the live baseline for `70b_kv_crr` V2-CRR. The R27 reading of 788 was either transient or environment-dependent and is not reproducible. **The R30+ priority for rectangular BLK_M=256/BLK_N=128 is unchanged** — it remains the only meaningful path to lift this cell out of the 0.84 ratio band, regardless of whether the baseline is 767 or 788.

**3. `8b_gate_crr` — `inconclusive` (small persistent shift, well within multi-cycle noise)**

R30 GPU5 MXFP8 median 2382.04 is between R27 (2405.79) and R29 (2390.72), and the R30 stdev is extremely tight (2.09 TFLOPS = 0.09%) which inflates Welch t-statistics on otherwise small absolute deltas:
   - R30 vs R27: -0.99%, t=-9.23 (significant by t but only ~1% absolute)
   - R30 vs R29: -0.36%, t=-4.14 (significant by t but only ~0.36% absolute)

FP8 on the same cell is flat (R30 vs R27 t=-0.89), so the ratio dropped 0.7pp (0.9216 → 0.9146) consistent across R29 and R30. This is the smallest of the three flagged cells and is closer to "measurement-precision detected a real ~1% shift" than "regression" or "artifact".

**Action:** demote from "regression" status. Continue to track in the LLaMA shape baseline matrix; do not invest in dedicated remediation. Future cycles measuring this cell in the 2380-2405 band should treat both extremes as plausible.

### Phase 1 summary

| Cell | R29 hypothesis (GPU-state artifact) | R30 verdict |
|---|---|---|
| 8b_down_crr | yes | **CONFIRMED artifact** — R30 matches R27 |
| 70b_kv_crr  | yes | **REFUTED** — R30 matches R29 (both differ from R27); persistent shift, not GPU-state |
| 8b_gate_crr | yes | **INCONCLUSIVE** — small ~1% persistent shift, low practical impact |

**Net Phase 1 outcome:** 1 of 3 R29 negative-t flags was a GPU4-state artifact; the other 2 reproduce on GPU5 and represent persistent cross-cycle environmental drift (no source change). **Source-driven regression: 0/3.**

### Phase 1 artifacts

- `analysis/fp8_gemm/mi350x/r30_reviewer_crossgpu_reverify.json` (full cross-cycle JSON)
- `analysis/fp8_gemm/mi350x/r30_reviewer_crossgpu_orchestrate.sh` (driver script)
- `analysis/fp8_gemm/mi350x/r30_reviewer_crossgpu_aggregate.py` (aggregator)
- `analysis/fp8_gemm/mi350x/r30_crossgpu_runs/{8b_gate_crr,8b_down_crr,70b_kv_crr}_{fp8,mxfp8}.txt` (raw per-cell logs)
- `analysis/fp8_gemm/mi350x/r30_crossgpu_runs/build_md5.log` (per-build md5 hygiene log)

---

## Phase 2 — Dev SHIP-candidate verdicts

R30 spawned 4 dev agents in parallel:
- Dev A (GPU0): rect-V2 fastpath kernel (R30 priority #1 from R29)
- Dev B (GPU1): VGPR reduction for V2-CRR occ=3 (R30 priority #2)
- Dev C (GPU2): `buffer_load_dword_lds` direct VMEM→LDS audit (R30 priority #3)
- Dev D (GPU3): B-tile load reorder (H7) for V2-RCR (R30 priority #4)

**No SHIP candidates were submitted for Reviewer verification.** Per-dev outcomes (read from each worktree's `r30{a,b,c}_findings.md` + Dev D's raw bench logs):

### Dev A — rectangular BLK_M=256/BLK_N=128 V2-CRR fastpath → **NO SHIP (Path A infeasible in 90-min budget)**

Worktree `/tmp/wt-r30-a` head `7702f000`. Dev A's audit re-confirms R28D's 2.5-day estimate (1.5 day kernel rewrite + 0.5 day Python + 0.5 day validation), then re-establishes baselines and refines the work breakdown for the next cycle author. No new fastpath kernel written. Baselines:

- 70B KV V2-CRR (4096×1024×8192): 791.25 TFLOPS median (5x preheat-bench, GPU0)
- 8192³ V2-CRR no-regression check: 2841.64 TFLOPS median, Δ -0.08% vs R28 (within noise)
- All correctness PASS (SNR 49.59-49.60, det 3/3)

Verdict: **NO SHIP**. Scaffolding from R28D + R29A preserved on r30-a; refined Path A breakdown documented for R31+.

Note: Dev A's GPU0 baseline of 791.25 TFLOPS for 70B KV is consistent with R27 GPU4's 787.96 (within +0.4%). This contradicts the R30 GPU5 reading of 767.23 by ~3%, suggesting the cross-cycle drift on 70b_kv_crr may have a per-GPU component (GPU5 happens to measure ~3% lower than GPU0/GPU4 historically). Even so, R30 GPU5 matches R29 GPU4 within 0.05% — both R29 and R30 are *measuring* a ~767 TFLOPS regime, while R27 / R30-A are measuring a ~788 TFLOPS regime. This adds nuance to the "real_regression_or_persistent" verdict above: it may be a **GPU- and time-dependent drift** rather than a true source-driven regression. **The Path A rectangular kernel remains the structural lever, independent of which absolute baseline is correct today.**

### Dev B — V2-CRR VGPR reduction for occ=3 → **NO SHIP — STRUCTURAL CLOSURE (paradigm correction)**

Worktree `/tmp/wt-r30-b` (uncommitted findings `r30b_findings.md`). Dev B disproves the R30 brief's premise that "VGPR reduction can lift V2-CRR occ from 2 to 3":
- Hardware ground truth: gfx950 LDS per CU = 163,840 B (160 KB).
- V2-CRR exact-8-wave kernel uses **234 VGPR + 139,264 B LDS per block** (no spill).
- **LDS, not VGPR, is the binding occupancy constraint**: 1 block/CU = 85% LDS util; 2 blocks/CU = 170% (overflow).
- "Occupancy 3" with an 8-wave block requires 24 waves/CU = non-integer multiple of 8 ⇒ infeasible without restructuring to non-8-wave blocks.
- Mechanistic experiment with `-DGEMM_MIN_BLOCKS_PER_CU=3` produces a binary with **bit-identical kernel resources** to the mb=2 baseline, and on 70B Gate it **regresses -0.55% with Welch t=-3.26** — pure overhead from the launch_bounds hint with no occupancy benefit.

Verdict: **NO SHIP + paradigm correction**. Adds `GEMM_MIN_BLOCKS_PER_CU > 2 for V2-CRR` to the lever-closure list. Rationale: any future "increase V2-CRR occupancy" attempt must first reduce LDS budget below ~80 KB/block (currently 139 KB) or restructure to a non-8-wave block — both multi-day rewrites.

### Dev C — `buffer_load_dword_lds` for V2-CRR scales → **AUDIT-ONLY (lever N/A)**

Worktree `/tmp/wt-r30-c` head `1849bc4e`. Dev C's data-flow trace proves:
- V2-CRR scales follow VMEM→VGPR→MMA path with **zero LDS round-trip** (scales declared as per-thread automatic `fp8e8m0_4` arrays at `crr_mxfp8_exact_8wave_fastpath.inc:221-224`, populated via `llvm_amdgcn_raw_buffer_load_b128/b64` direct to VGPR at lines 316-358).
- `buffer_load_*_lds` is only useful when destination is LDS — there is no eligible call site for scales.
- A/B tile fills (the only LDS-bound VMEM traffic in V2-CRR) **already use** `llvm_amdgcn_raw_buffer_load_lds` via TK's `G::load`. Lever is also maxed for residual LDS traffic.

Verdict: **AUDIT-ONLY (lever closed)**. No source change. Confirms R29 Dev D's recommendation was based on a misreading of the V2 scale data flow (which has zero LDS hop). The Dev D audit had assumed scales went VMEM→LDS→VGPR, but R27 paradigm correction #1 had already documented "V2 has NO scale LDS" — Dev C's trace is independent confirmation.

### Dev D — B-tile load reorder (H7) for V2-RCR → **NO SHIP — paradigm correction (correctness FAIL)**

Worktree `/tmp/wt-r30-d` (uncommitted source change: 36-line `MXFP8_RCR_V2_BLOAD_REORDER` macro scaffolding in `kernel_mxfp8_layouts.cpp`). Dev D ran 5 cells on GPU3:

| cell | flag | SNR (dB) | det | tflops median | verdict |
|---|---|---:|:---:|---:|---|
| A0_base | (none) | 49.59 | True | (baseline) | OK |
| A1_v1 | `-DMXFP8_RCR_V2_BLOAD_REORDER=1` |  7.82 | False | 2357.24 | **CORRECTNESS FAIL** |
| A2_v2 | `-DMXFP8_RCR_V2_BLOAD_REORDER=2` |  ~  | False | (broken) | **CORRECTNESS FAIL** |
| A3_v3 | `-DMXFP8_RCR_V2_BLOAD_REORDER=3` | 19.33 | False | 2454.17 | **CORRECTNESS FAIL** |
| A4_v4 | `-DMXFP8_RCR_V2_BLOAD_REORDER=4` | 24.40 | False | 2487.16 | **CORRECTNESS FAIL** |
| B0_base | (none) on 8192³ | 49.59 | True | 3016.02 | OK (baseline check) |

All 4 reorder variants (1-4) fail the SNR≥48 dB gate (best is 24.40 dB, i.e. 24+ dB below floor) AND fail determinism. Reordering the B-tile load sequence on V2-RCR breaks the lockstep between B-load completion and the MFMA dependency it feeds. The ordering is load-bearing for correctness; "reorder for perf" cannot be done without an MFMA-side restructure.

Verdict: **NO SHIP — paradigm correction**. Adds `B-tile load reorder for V2-RCR is correctness-load-bearing (any reorder breaks SNR < 25 dB and determinism)` to lever-closure list.

### Phase 2 net outcome

**0 of 4 R30 dev agents produced a SHIP candidate.** No SHIP gate verification was performed; the gate (SNR ≥ 48, det 3/3, Welch t > 3, no regression > -1%) had no candidates to apply against.

Two new paradigm corrections (Dev B + Dev D) extend the closure list. One audit lever closed (Dev C). The R30 priority list for R31 is rebuilt below.

---

## Phase 3 — R31 priorities + cherry-pick recommendations

### Cherry-pick to feat/mxfp8-only

**RECOMMENDED:**

1. **Reviewer artifacts (Phase 1):** `r30_reviewer_crossgpu_reverify.json`, `r30_reviewer_crossgpu_orchestrate.sh`, `r30_reviewer_crossgpu_aggregate.py`, `r30_reviewer_findings.md`, and the `r30_crossgpu_runs/` log directory. Pure analysis artifacts; no kernel impact. The build-cache hygiene rule (rm + md5 log) is a reusable harness pattern.
2. **Dev A artifacts (rev-only):** `r30a_findings.md` + refined Path A breakdown. Useful R31 starting point for the rect-V2 attempt. No kernel cherry-pick (no new code).
3. **Dev B paradigm correction:** record in `agent_prompt.md` R30 section as paradigm correction #1 ("VGPR reduction is N/A for V2-CRR occ; LDS is the binding constraint at 139 KB/block"). No source cherry-pick.
4. **Dev C audit:** record in `agent_prompt.md` R30 section as paradigm correction #2 ("`buffer_load_dword_lds` is N/A for V2 scales; scales are VMEM→VGPR direct, no LDS hop"). No source cherry-pick.
5. **Dev D paradigm correction:** record in `agent_prompt.md` R30 section as paradigm correction #3 ("B-tile load reorder for V2-RCR is correctness-load-bearing; any reorder breaks SNR < 25 dB and determinism"). No source cherry-pick (Dev D's macro scaffolding is dead code at default `=0` but provides no value as audit-trail since the lever is closed).

**NOT cherry-picked:**

- Dev D's `MXFP8_RCR_V2_BLOAD_REORDER` macro scaffolding: closed lever, no audit-trail value, defaults to no-op but adds noise to the source. Leave on r30-d side branch.
- Dev B's experimental `GEMM_MIN_BLOCKS_PER_CU=3` build configs: same — closed lever.

### R31 priority list (rebuilt from R30 closures)

1. **【critical / 2-3 day】Rectangular BLK_M=256/BLK_N=128 V2-CRR fastpath kernel** (Path A): unchanged from R29/R30 priority #1 — this remains the only structural lever for 70B KV (0.84 band) and likely 8B Gate / 70B Gate. R30 Dev A's refined breakdown is the starting point. Need full sprint-budget commitment, not 90-min slot.

2. **【medium / re-verify】70b_kv_crr cross-GPU baseline triangulation**: R30 cross-GPU revealed a per-GPU dispersion of ~3% on this shape (GPU0=791, GPU4=788, GPU5=767, GPU4-R29=767). Consider running a 4-GPU baseline (GPU0/4/5/6) once before R31 to fix the live baseline number for go/no-go on Path A's projected ≥0.92 ratio. Cheap (~10 min/GPU).

3. **【medium】8B Down + 70B Down K-large MLP shapes**: still in 0.91-0.94 band, untouched by R28-R30. Now that R30 Dev D closed B-tile reorder for V2-RCR, the open levers for these K-heavy CRR shapes are: (a) PIPELINE_SCALE second-buffer (R28D scaffolding on r28-d, requires LDS budget analysis), (b) K-axis prefetch bump beyond current `BLK*K_HALF*2` window, (c) per-shape cachepolicy table refinement (R29B says cp gate is tight; only neutral or losing variants found within (N,K) corners explored).

4. **【low / 1 day】4096³ V2-RCR GRID under-occupancy** (R29 Dev C's structural finding): 256 blocks at BLK=256 / 304 CUs = 0.84 wave-fill, 16% CUs idle. Per-kernel optimization is structurally bounded; closing the 0.92 gap likely requires dispatch-geometry changes (block reshape or streamk). Defer until rect-V2 (priority 1) lands.

5. **【methodology / standing rule】Build-cache hygiene + per-build md5 log** (R29 Dev C): R30 Reviewer cross-GPU run is a clean reference impl (`r30_reviewer_crossgpu_orchestrate.sh`). All R31+ orchestrate scripts must follow.

### Levers permanently closed in R30 (extends R29 closure list)

- `GEMM_MIN_BLOCKS_PER_CU > 2 for V2-CRR` (R30 Dev B): LDS-bound, not VGPR-bound. NEVER prototype "increase V2-CRR occupancy via VGPR reduction" — must restructure to non-8-wave block or shrink LDS below 80 KB/block first.
- `buffer_load_dword_lds for V2 scales` (R30 Dev C): scales have zero LDS hop in V2; lever has no eligible call site. NEVER prototype "scale LDS direct path".
- `B-tile load reorder for V2-RCR` (R30 Dev D): correctness-load-bearing; any reorder collapses SNR to < 25 dB and breaks determinism. NEVER prototype "reorder B-side `buffer_load` for V2-RCR perf".

### Cycle wrap

- **0 SHIPs from R30 dev agents** (consistent with R28's 1-SHIP-then-3-NO-SHIP and R29's 0-SHIP cycles).
- **0 of 3 R29 negative-t cells confirmed as source regressions**; 1 is a GPU4-state artifact, 2 are persistent cross-cycle environmental drift not attributable to source.
- **3 new lever closures** (VGPR/occ for V2-CRR, buffer_load_dword_lds for scales, B-tile reorder for V2-RCR) narrow R31 search space further.
- **R28 SHIP (cachepolicy auto-select)** still in production; no regression detected.
- **Dev A's refined Path A breakdown** is the recommended R31 priority #1.

### Files

- `analysis/fp8_gemm/mi350x/r30_reviewer_crossgpu_reverify.json`
- `analysis/fp8_gemm/mi350x/r30_reviewer_crossgpu_orchestrate.sh`
- `analysis/fp8_gemm/mi350x/r30_reviewer_crossgpu_aggregate.py`
- `analysis/fp8_gemm/mi350x/r30_reviewer_findings.md` (this doc)
- `analysis/fp8_gemm/mi350x/r30_crossgpu_runs/` (raw bench logs + build_md5.log)
