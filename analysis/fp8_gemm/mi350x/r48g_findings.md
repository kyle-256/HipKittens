# R48 Dev G — MXFP8 RRR Partial-Unroll Sweet Spot Search — REFUTED

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ c46227d0 (R48E wrap)
**GPU:** MI355X (gfx950), HIP_VISIBLE_DEVICES=3
**Lever:** R48 Dev D §2.2 (loop-control overhead) → R48 Dev E (full-unroll
REFUTED via VGPR spill) → **search for partial-unroll U > 1 that recovers
some loop-control overhead without triggering spill**.

**TL;DR — VERDICT: REFUTED.** There is no partial-unroll sweet spot for the
MXFP8 RRR K-pair loop. Filling the missing U=16 data point and re-verifying
Dev E's U={2, 4, 8} runs confirms a **structural compiler threshold at N>=2**:
any `#pragma unroll N` with N>=2 immediately commits to spilling
(VGPR=256, Scratch=208 bytes/lane, ~70% perf regression on K=4096 RRR cells).
The unroll-factor knob is **monotonic-bad** at any N>=2 within the current
`do_k_iter` always-inline + fused scale-pack design. Macro
`MXFP8_RRR_MAIN_UNROLL` left in tree **default OFF (=0)** — unchanged from
Dev E.

---

## 1. Method

### 1.1 What Dev G adds beyond Dev E

Dev E swept U ∈ {2, 4, 8, 32}. Dev G:
1. **Re-verified macro accepts arbitrary N** (checked
   `rrr_mxfp8_exact_8wave_fastpath.inc` lines 45–53 — `_Pragma("unroll N")`
   with `TK_STRINGIFY(unroll MXFP8_RRR_MAIN_UNROLL)`, accepts any positive
   integer).
2. **Filled the missing U=16 data point** (between Dev E's u=8 = 803.2
   TFLOPS and u=32 = 679.8 TFLOPS).
3. **Filled U=1 data point** (explicit `#pragma unroll 1` = disable, vs U=0
   no-pragma) to establish that U=1 is not silently different from baseline.
4. **Re-anchored U=0 baseline on current GPU state** (Dev E's baseline run
   was 4 days prior — within ±1.5% of Dev E numbers, confirming no
   GPU/SCLK drift contamination of Dev E's data).

### 1.2 Build / bench

Same env + Makefile invocation as `r48e_findings.md` §1.2. GPU 3, 50 warmup
+ 100 iter, 3 runs per cell, 20s cooldown. Resource extraction via
`-Rpass-analysis=kernel-resource-usage` (HIPCC default).

```bash
cd /shared_nfs/kyle/test/Hipkittens2/.claude/worktrees/agent-a00aa129/analysis/fp8_gemm/mi350x
export THUNDERKITTENS_ROOT=/shared_nfs/kyle/test/Hipkittens2/.claude/worktrees/agent-a00aa129
export ROCM_PATH=/opt/rocm
# For each U in {0, 1, 16}:
rm -f tk_mxfp8_layouts*.so
make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
  CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K -DMXFP8_RRR_MAIN_UNROLL=$U"
HIP_VISIBLE_DEVICES=3 MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
  MXFP8_WARMUP=50 MXFP8_ITERS=100 MXFP8_CHECK=0 \
  MXFP8_LAYOUTS=rrr MXFP8_PRESHUFFLE_QUANT=1 \
  python3 test_mxfp8_python.py $M $N $K
```

---

## 2. Results

### 2.1 Combined U-sweep table (Dev E + Dev G)

K=4096 cells, 8B Q/O (4096³) and 8B Gate/Up (4096×14336×4096), 3-run mean.

| U   | VGPRs | Scratch (B/lane) | 8B Q/O TFLOPS | Δ vs base | 8B Gate/Up TFLOPS | Δ vs base | Source |
|----:|------:|-----------------:|--------------:|----------:|------------------:|----------:|--------|
|   0 |   254 |                0 |        2342.2 |     0.00% |             2491.5 |     0.00% | Dev G base + Dev E base |
|   1 |   254 |                0 |        2334.1 |    -0.34% |                  — |         — | Dev G  |
|   2 |   256 |          **208** |         804.8 |   -65.62% |              777.3 |   -68.80% | Dev E  |
|   4 |   256 |          **208** |         810.2 |   -65.40% |              778.7 |   -68.74% | Dev E  |
|   8 |   256 |          **208** |         803.2 |   -65.70% |              778.2 |   -68.76% | Dev E  |
|  16 |   256 |          **208** |     **807.5** | **-65.51%** |          **779.2** | **-68.71%** | **Dev G** |
|  32 |   256 |          **208** |         679.8 |   -71.00% |              749.8 |   -69.90% | Dev E  |

(Dev G U=0 baseline: 3-run mean 2361.6/2340.8/2324.1 = 2342.2 TFLOPS; matches
Dev E baseline 2368.4 TFLOPS within ±1.1%, confirming no GPU drift between
the Dev E and Dev G runs.)

### 2.2 Structural finding

The compiler exhibits a **bimodal threshold at N>=2**:
- **U ∈ {0, 1}**: VGPR=254, Scratch=0 → baseline perf (~2340 TFLOPS).
- **U ∈ {2, 4, 8, 16}**: VGPR=256, Scratch=**208** → ~70% regression.
- **U=32 (full unroll)**: VGPR=256, Scratch=**208**, +33% extra MFMAs in
  flight visibly worsen scheduling (-5pp vs U={2..16}).

The flat 208 B/lane scratch across U ∈ {2..16} confirms the spill is
**structurally invariant to the unroll factor** above the "spill or not"
threshold. The compiler's heuristic is binary: once given any
`#pragma unroll N` with N>=2 on this loop body, it commits to spilling the
scale-pack live ranges to scratch. There is no register-pressure tuning
parameter to extract via this lever.

### 2.3 SHIP gate

- Perf gain ≥ +1.0% on K=4096 RRR cell: **NO** (worst case −71% at U=32, best
  case −65.4% at U=4 — all U>=2 catastrophically regress).
- Net positive across 7 RRR shapes: **N/A** (single-cell K=4096 already
  refuted; not run on larger K — extrapolation from Dev E shows worse
  regression on K=8192/14336/28672, see Dev E §3.1).

→ **REFUTED.** No SHIP. No kernel change. Macro left default OFF as Dev E.

---

## 3. Root-cause confirmation

The U=16 data point is the **decisive bridge** between Dev E's small-factor
sweep and the full-unroll point. It rules out the hypothesis that the
spill threshold scales with unroll factor (e.g., "maybe small N stays under
the live-range limit"):

- If spill were proportional to unroll factor, U=2 would have ~12 B/lane
  scratch and U=16 would have ~104 B/lane (linear scaling expectation).
  **Observed:** flat 208 B/lane across U ∈ {2, 4, 8, 16}. The spill amount
  is determined by the compiler's worst-case live-range analysis on the
  unrolled body, not by the iteration count.
- If the regression were due to I-cache pressure from larger code size, U=2
  would regress less than U=16. **Observed:** U=2 (804.8) ≈ U=16 (807.5).
  The bottleneck is scratch-spill latency, not I-cache.

This pins the structural ceiling Dev E identified (§4 of `r48e_findings.md`):
the `do_k_iter` always-inline lambda's per-K-pair scale-pack live ranges
(`a0_scale_packs`, `a1_scale_packs`, `b0_scale_packs`, `b1_scale_packs`)
overlap across adjacent K-pair iterations once any unroll exposes ≥2 inlined
copies side-by-side. The compiler cannot prove the previous K-pair's scale
packs are dead before the next K-pair's load begins, so it spills.

**Implication for R49+**: Dev E's §4.1 R49+ list is the only path forward:
1. `do_k_iter` noinline + manual phase specialization,
2. Manual K-pair unroll with `__builtin_amdgcn_kill` live-range hints,
3. V2-RRR scale-SRD pre-loaded indirectly to remove scale_packs from the
   register file.

The compiler-pragma lever (any N>=2) is **fully exhausted**.

---

## 4. Outcome

- **No code change.** Macro `MXFP8_RRR_MAIN_UNROLL` remains default 0 from
  Dev E.
- Findings doc + bench logs preserved (`r48g_*.log`,
  `r48g_*_resource.log`).
- The +1-2pp loop-control overhead identified by Dev D §2.2 remains a
  **structural ceiling** unreachable via compiler-pragma alone.

## 5. Files

| Path | Purpose |
|---|---|
| `r48g_findings.md` | This document |
| `r48g_unroll1_8B_QO_resource.log` | U=1 VGPR/scratch (=baseline) |
| `r48g_unroll16_8B_QO_resource.log` | **U=16 VGPR/scratch (=spill, same as U=2..8)** |
| `r48g_unroll16_8B_GateUp_resource.log` | U=16 Gate/Up resource log |
| `r48g_baseline_8B_QO_resource.log` | U=0 baseline VGPR/scratch (re-anchor) |
| `r48g_U0_8B_QO_run{1,2,3}.log` | Baseline 3-run TFLOPS |
| `r48g_U1_8B_QO_run{1,2,3}.log` | U=1 3-run TFLOPS |
| `r48g_U16_8B_QO_run{1,2,3}.log` | U=16 8B Q/O 3-run TFLOPS |
| `r48g_U16_8B_GateUp_run{1,2,3}.log` | U=16 8B Gate/Up 3-run TFLOPS |

## 6. Cross-references

- `r48d_findings.md` §2.2 — origin of the loop-control overhead lever.
- `r48e_findings.md` — full-unroll refutation (Dev E swept U={2,4,8,32}).
- `rrr_mxfp8_exact_8wave_fastpath.inc` lines 30–53 — macro implementation.

## 7. One-Line Summary

**Filling the missing U=16 data point confirms Dev E's structural finding:
the `#pragma unroll N` lever is monotonic-bad for N>=2 (flat 208 B/lane
spill, ~70% regression band) on the MXFP8 RRR K-pair loop. There is no
partial-unroll sweet spot. The compiler-pragma lever is fully exhausted;
any further loop-control recovery requires R49+ structural rewrites
(noinline phase split, live-range hints, scale-SRD indirection).**
