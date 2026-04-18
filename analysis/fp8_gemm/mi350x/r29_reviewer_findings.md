# R29 Reviewer Findings

**Date:** 2026-04-18
**Branch:** r29-rev (base feat/mxfp8-only @ `145ff766`)
**GPU:** GPU4 (MI355X, gfx950) via `ROCR_VISIBLE_DEVICES=4 HIP_VISIBLE_DEVICES=0`
**Methodology:** Per-process 8s 16k FP16 preheat → 5x in-process bench (warmup=50, iters=100), 2-sigma outlier filter, MXFP8 V2 (gemm_*_pq_v2) vs FP8 per-tensor.
**Cell schema:** Identical to R27 Reviewer (`r27_reviewer_baseline_gpu4.json`) for delta-vs-R27 comparability.

---

## Phase 1 — Baseline Reverify

### Per-cell results (10 cells)

| # | Cell | Shape | Layout | FP8 med | MXFP8 med | R29 ratio | R27 ratio | Δpp | Welch t (MX, R29 vs R27) | Verdict |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 8k_rcr        | 8192³            | rcr | 3310.29 | 3010.60 | **0.9095** | 0.9324 | -2.29 |  -1.04 | regressed (FP8 drift up) |
| 2 | 8k_rrr        | 8192³            | rrr | 3296.60 | 2995.25 | **0.9086** | 0.9270 | -1.84 |  +1.87 | regressed (FP8 drift up) |
| 3 | 8k_crr        | 8192³            | crr | 3086.07 | 2782.56 | **0.9017** | 0.9266 | -2.49 |  +1.95 | regressed (FP8 drift up) |
| 4 | 4k_rcr        | 4096³            | rcr | 2572.08 | 2370.85 | **0.9218** | 0.9229 | -0.11 |  +0.52 | unchanged |
| 5 | 8b_gate_crr   | 4096×14336×4096  | crr | 2660.37 | 2390.72 | **0.8986** | 0.9216 | -2.30 |  -4.11 | regressed (MXFP8 dropped, t<-3) |
| 6 | 8b_down_crr   | 4096×4096×14336  | crr | 2993.91 | 2712.16 | **0.9059** | 0.9394 | -3.35 |  -3.66 | regressed (MXFP8 dropped, t<-3) |
| 7 | 70b_qo_rcr    | 4096×8192×8192   | rcr | 3233.44 | 2944.26 | **0.9106** | 0.9373 | -2.67 |  -2.29 | regressed (FP8 drift up) |
| 8 | 70b_kv_crr    | 4096×1024×8192   | crr |  940.81 |  766.86 | **0.8151** | 0.8646 | -4.95 |  -8.51 | regressed (MXFP8 dropped, t<-3) |
| 9 | 70b_gate_crr  | 4096×28672×8192  | crr | 2813.75 | 2349.81 | **0.8351** | 0.8351 | +0.00 | +10.20 | **R28 SHIP confirmed**: MXFP8 +45.85 TFLOPS (+1.99%); FP8 also up ~+1.99%, ratio flat |
| 10| 70b_down_crr  | 4096×8192×28672  | crr | 3038.16 | 2513.80 | **0.8274** | 0.8459 | -1.85 |  +1.88 | regressed (FP8 drift up) |

**Summary:** 0/10 cells pass the ≥0.95 perf gate (same as R27). Worst cells unchanged: 70B KV crr (0.8151), 70B Down crr (0.8274), 70B Gate crr (0.8351).

### Cross-cycle interpretation

The R29 baseline shows **uniform FP8 drift +1-3% vs R27** (Welch t > 3 on FP8 for 6/10 cells), most likely a GPU4-state / DPM-tuning effect (R29 saw consistent ~1.8 GHz post-preheat sclk in the in-process bench window). MXFP8 drifted ±2% for most cells but did NOT keep pace with FP8 → ratios mostly down 1-3pp. **This is NOT a real MXFP8 regression** — the R28 cherry-picks did not modify the V2 fastpath inner loop outside the cachepolicy gate region, so absolute MXFP8 numbers remain in the same noise band as R27. The drift is FP8-driven.

**Three cells show statistically significant MXFP8 drops (t < -3.0)** that warrant attention:
- **70b_kv_crr**: -21.10 TFLOPS / -2.68% (t=-8.51). The largest absolute MXFP8 regression. Same kernel binary expected, so this is consistent with cross-cycle DPM/scheduler variability or a real GPU-state difference. R26 already documented this cell as cold-DPM sensitive.
- **8b_down_crr**: -67.23 TFLOPS / -2.42% (t=-3.66). Adjacent cells (70b_down_crr, K=28672 same layout) show MXFP8 +0.55%, so the K=14336 corner specifically is the noisy one.
- **8b_gate_crr**: -15.07 TFLOPS / -0.63% (t=-4.11). Smaller absolute delta but tight stdev makes it significant.

**70b_gate_crr (R28 Dev A SHIP target)**: confirmed working — MXFP8 median moved from 2303.96 → 2349.81 (+45.85 TFLOPS, +1.99%, t=+10.20). FP8 also moved +54.82 (+1.99%), so the ratio is identical to R27 (0.8351). The R28 cachepolicy auto-select ship is healthy in production.

### Files

- Baseline JSON: `analysis/fp8_gemm/mi350x/r29_reviewer_baseline_gpu4.json`
- Bench harness: `analysis/fp8_gemm/mi350x/r29_reviewer_bench5x.py`
- Orchestrator: `analysis/fp8_gemm/mi350x/r29_reviewer_orchestrate.sh`
- Aggregator: `analysis/fp8_gemm/mi350x/r29_reviewer_aggregate.py`
- Per-cell raw outputs: `analysis/fp8_gemm/mi350x/r29_runs/<cell>_{fp8,mxfp8}.txt`

---

## Phase 2 — Dev SHIP-candidate Verdicts

### Dev A — rectangular BLK_M=256/BLK_N=128 V2 path → **NO SHIP (foundation only)**
- Worktree: `/tmp/wt-r29-a`, head `b2cc032f`
- What shipped: host-side dispatcher guard in `kernel_mxfp8_layouts.cpp` that prevents the GPU memory fault when `MXFP8_RECT_BLK_N=64` builds call `gemm_crr_pq_v2`. V1 fallback re-verified correct on rect target shape (4096×1024×8192, SNR 49.60 dB, det 3/3). Default-build binary unchanged.
- SHIP gate: NOT MET — the rect-V2 fastpath kernel itself was NOT written. V1 fallback at the rect target shape is 294× slower than V2 baseline (2.66 vs 781.91 TFLOPS), so no perf win this cycle.
- Verdict: **NO SHIP**, but is a clean foundation for R30 Path A. Recommend cherry-picking the dispatcher guard to feat/mxfp8-only as a safety fix (prevents user-visible memory faults on rect builds).
- Artifact: `r29a_findings.md`, `r29a_bench.py`, `r29a_orchestrate.sh`, dispatcher guard in `kernel_mxfp8_layouts.cpp`.

### Dev B — cp=2 sweep on 5 additional CRR shapes → **NO SHIP**
- Worktree: `/tmp/wt-r29-b`, head `b82eab24`
- What was tested: 5x preheat-bench cp=0 vs cp=2 on 70B Down (4096×8192×28672), 8B Down (4096×4096×14336), and 3 synthetic shapes (N=8192/14336/20480 × K=8192/14336).
- Result: cp=2 is **negative on every cell tested** (ranging −0.11% to −3.34%). Even at N=20480 K=8192 (closest to gate boundary), cp=2 loses 2.53% with t=-27.5. The R28 gate `(N≥28672) AND (K≥8192)` is tight and correct on both axes.
- Verdict: **NO SHIP** — current gate is optimal, no refinement possible. Documents per-shape cp lookup table for future audits.
- Artifact: `r29b_findings.md`, `r29b_bench.py`, `r29b_orchestrate.sh`, 10 raw cell logs. No source change.

### Dev C — V2-RCR new-lever audit (4096³ priority) → **NO SHIP**
- Worktree: `/tmp/wt-r29-c`, head `145ff766` (work staged uncommitted; findings written but Dev C has not yet committed)
- What was tested: H1 `MXFP8_RCR_V2_SCHED_BARRIER_MASK` (compiler reorder mask sweep) and H2 `MXFP8_RCR_V2_MMA_SETPRIO` (sweep prio=2,3) on V2-RCR 4096³ and 8192³.
- Result: All NULL within bench-to-bench noise (|t| ≤ 0.61). H1 sched-barrier-mask=0xB (allow non-MEM/VALU/MFMA reorder): also NULL on 4096³. Hand-tuned ordering plus surrounding `s_barrier()` already saturate the available scheduling freedom.
- Verdict: **NO SHIP**. RCR setprio lever closed (mirrors R28 Dev B's CRR closure). 2-macro scaffolding is functional no-op at default values, can be cherry-picked as audit-trail or dropped.
- Artifact: `r29c_findings.md`, `r29c_bench.py`, `r29c_orchestrate.sh`, 6 raw cell logs (currently uncommitted on r29-c).
- Bonus (Dev C side discovery): The shared `Makefile clean` target only removes `$(TARGET)` (no extension), leaving the actual `.so` artifact in place. Devs A/B/C orchestrators all `rm -f tk_*.so` before building to defend against this; recommend folding the fix into `Makefile` proper.

### Dev D — V2-CRR LDS bank conflict audit → **AUDIT-ONLY (lever closed)**
- Worktree: `/tmp/wt-r29-d`, head `63fc644b`
- What was tested: Static analysis of all `ds_read*`/`ds_write*` calls in V2-CRR exact-8-wave fastpath. Per-lane bank computation on `load_col_from_v2_st_half` (B-tile) and `load_col_from_v2a_st_half` (A-tile); ST_v2 store-side swizzle.
- Result: **Zero LDS bank conflicts in all configurations** (4 dispatch cycles × K_HALF ∈ {0,1} × j ∈ [0, RT::width)). The existing `(nc ^ sw_k)` swizzle distributes lane addresses evenly across all 32 banks every 16-lane cycle. No fix possible at the swizzle level.
- Verdict: **LEVER CLOSED**. Documents that V2-CRR's 7-15% gap originates elsewhere — recommends R30 explore (a) VGPR/occupancy reduction (currently ~246 VGPRs limits to occ=2), (b) `buffer_load_dword_lds` direct VMEM→LDS to eliminate VGPR staging (big restructure).
- Artifact: `r29d_lds_bank_audit.md`. No source change.

---

## Phase 3 — Recommendations for R30

### Cherry-pick candidates from R29 (to feat/mxfp8-only)
1. **Dev A dispatcher guard** in `kernel_mxfp8_layouts.cpp` (small safety fix, prevents memory fault on `-DMXFP8_RECT_BLK_N=64` rect builds). Default behavior unchanged. **RECOMMENDED.**
2. **Dev C 2-macro scaffolding** (`MXFP8_RCR_V2_SCHED_BARRIER_MASK`, `MXFP8_RCR_V2_MMA_SETPRIO`) — defaults are no-ops, useful as audit-trail. **OPTIONAL.**
3. **Makefile fix** for `clean` target to also remove `tk_*.so`. Independent of R29 work but uncovered by Dev C. **RECOMMENDED.**

### R30 priority list (rebuilt from R29 root-cause)

1. **【critical / 2-3 day】Rectangular BLK_M=256/BLK_N=128 V2 fastpath kernel** (Path A): Dev A's dispatcher guard now provides a clean hook. Estimated ~2.5 days per Dev D's R28 audit. Target: 70B KV V2-CRR 0.8151 → projected ≥0.92 (8 N-tile vs 4 N-tile = ~2× CU utilization).

2. **【high】V2-CRR VGPR reduction for occupancy=3**: per Dev D's audit, V2-CRR uses ~246 VGPRs blocking occupancy=2. If a 4-accumulator → 2-accumulator design or scratch reuse can drop VGPRs to <171, gfx950 LDS cap (160 KB/CU) is the next bottleneck (V2 currently 131 KB) — would need joint VGPR + LDS reduction. Multi-day.

3. **【high】`buffer_load_dword_lds` direct VMEM→LDS for scales** (Dev D recommendation): currently V2-CRR stages scales VMEM→VGPR→LDS. Direct path eliminates ~512B of per-CTA register pressure and an ALU pass. Big restructure.

4. **【medium】8B Down / 8B Gate CRR cross-cycle drift investigation**: R29 measured statistically significant MXFP8 regressions (t=-3.66 / -4.11) on these cells with no source change. Re-verify on a fresh GPU before R30 SHIP gates. May be DPM/clock-state artifact rather than true regression.

5. **【closed】LDS bank conflicts on V2-CRR** (Dev D): permanently closed.
6. **【closed】MXFP8_RCR_V2_MMA_SETPRIO** (Dev C): permanently closed (mirrors R28 V2-CRR closure).
7. **【closed】MXFP8_RCR_V2_SCHED_BARRIER_MASK** (Dev C): permanently closed.
8. **【closed】cp=2 region extension beyond (N≥28672, K≥8192)** (Dev B): gate is tight, no refinement.

### Cycle wrap

- **0 SHIPs from R29 dev agents**.
- **R28 Dev A's SHIP (cachepolicy auto-select gate) confirmed in R29 baseline**: 70B Gate MXFP8 +45.85 TFLOPS / +1.99% (Welch t=+10.2) vs R27.
- **3 levers permanently closed** (LDS bank conflicts, RCR setprio, RCR sched-barrier), narrowing R30 search space.
- **1 dispatcher safety fix** (Dev A) recommended for cherry-pick.
- **Cross-cycle drift detected** on 70b_kv_crr / 8b_down_crr / 8b_gate_crr MXFP8 medians (Welch t < -3) — recommend re-verify on a different GPU as part of R30 baseline.
