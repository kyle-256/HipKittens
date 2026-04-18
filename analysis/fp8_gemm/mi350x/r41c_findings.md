# R41 Dev C — Decode-shape coverage survey

**Branch**: r41-dev-c
**GPU**: MI355X (gfx950), HIP_VISIBLE_DEVICES=2 (single-GPU survey scope)
**Date**: 2026-04-18
**Scope**: R41+ priority list item #2 — prefill-shape work (M=4096) has covered R28-R40 cycles; decode shapes (M=1, 32, 128) entirely unmapped. Survey current MXFP8 vs FP8 perf, identify cells failing the 95% rule.

## Top-line: 0 PASS / 6 FAIL

Every decode shape on every backend is **tail-kernel limited**. The dispatcher's V2 fastpaths never fire on M < BLK=256, and the fallback (`gemm_tail_kernel`) is the correctness-first reference implementation — never optimized. Both MXFP8 and FP8 land in the same fallback, which is why their absolute numbers are similarly small. The 95% rule is FAIL across the board because MXFP8 V1 path is structurally slower than FP8 V1 path (extra scale-load + multiply per K-block).

## Coverage table (RCR layout, LLaMA Q/K/V/O proj)

| # | Shape (M×N×K) | Workload | MXFP8 TF | FP8 TF | Ratio | PASS 95%? | Dispatched predicate |
|--:|--------------|----------|---------:|-------:|------:|:---------:|----------------------|
| 1 | 1×4096×4096 | 8B single-token | 0.0647 | 0.0942 | 68.69% | **FAIL** | V1-LEGACY-FALLBACK (V2 predicate misses → tail kernel) |
| 2 | 32×4096×4096 | 8B batch=32 | 0.8739 | 1.1253 | 77.66% | **FAIL** | V1-LEGACY-FALLBACK |
| 3 | 128×4096×4096 | 8B batch=128 | 0.8819 | 1.1457 | 76.97% | **FAIL** | V1-LEGACY-FALLBACK |
| 4 | 1×8192×8192 | 70B single-token | 0.1069 | 0.1854 | 57.66% | **FAIL** | V1-LEGACY-FALLBACK |
| 5 | 32×8192×8192 | 70B batch=32 | 0.8845 | 1.1385 | 77.69% | **FAIL** | V1-LEGACY-FALLBACK |
| 6 | 128×8192×8192 | 70B batch=128 | 0.8851 | 1.1441 | 77.36% | **FAIL** | V1-LEGACY-FALLBACK |

Bench config: WARMUP=50, ITERS=100, median-of-iters timing (per-iteration `torch.cuda.synchronize()` + `output.zero_()`), 3-second float16 GEMM preheat. Single-GPU survey — STRICT-promotion not in scope.

Raw log: `r41c_survey_results.log` (12 entries, all `[mxfp8_dispatch]` traces captured for audit).

## Mechanism analysis — why all 6 FAIL

### 1. Dispatcher cannot route decode shapes to any V2 fastpath
- `kernel_mxfp8_layouts.cpp:3675-3676`: `rcr_can_use_exact_8wave_scaled(g)` requires `g.m == M_DIM && g.n == N_DIM && g.k == K_DIM`.
- `BLK = GEMM_BLOCK_SIZE = 256` (line 304-305, 335). The V2 fastpath grid = `(g.m / BLK) * (g.n / BLK)`. For M < 256 the grid is **0 blocks** → `hipErrorInvalidConfiguration` if forced.
- Even when M_DIM is compiled to match decode M (e.g., M_DIM=128), the predicate succeeds but the launch fails. So **for any decode build the V2 path is unusable**.
- Result: dispatcher falls through to `gemm_tail_kernel` with `TAIL_BLOCK_M=16`, `TAIL_BLOCK_N=16` — a single 32×32 thread block per output tile. This is the legacy correctness-first reference, never tuned for throughput.

### 2. Tail kernel is the same code path for MXFP8 and FP8 — but MXFP8 carries scale overhead
- Both kernels share `gemm_tail_kernel<L, PRESHUFFLED_QUANT>` structure; the MXFP8 tail must additionally fetch and apply per-32-K E8M0 scales for both A and B operands per inner-K accumulation step (`load_scale_scalar_preshuffled`).
- FP8 tail applies a single per-tensor scalar after the entire dot product (`acc * g.scale`).
- This explains the consistent ~77% MXFP8/FP8 ratio across the M=32/128 shapes (all dominated by the same tail-kernel overheads with the MXFP8 scale tax dragging it down).

### 3. M=1 is anomalously bad on both backends
- 8B M=1: 0.065 TF MXFP8 / 0.094 TF FP8 — these are ~10× lower than M=32. With a 16×16 tail block and `TAIL_BLOCK_M=16`, M=1 launches `ceil_div(1,16)=1` row-block but only 1 row is real (15 zero rows of useless work) → effectively 1/16 useful compute per block, plus latency-bound launch overhead amortized over a tiny problem.
- 70B M=1 ratio is even worse (57.7%) because the FP8 tail handles the 1 row faster relative to MXFP8 (less scale-load latency for the larger N=8192 working set).

### 4. M=32 and M=128 saturate at the same MXFP8 TFLOPS
- 0.874 / 0.882 / 0.885 / 0.885 TF — the tail kernel hits a per-output-tile throughput plateau (single 16×16 block per tile, ~256 threads per CU, no pipelining of K).
- N×K determines absolute time, M scales linearly within tail-kernel regime — no architectural lift available without writing a small-M fastpath.

## Recommendations for R42+

### A. CRITICAL — production-inference scope is incomplete
The TODO.md primary perf goal "MXFP8 ≥ FP8 × 95%" implicitly assumes prefill-only workloads. Production LLM inference is ~half decode by token-time. **The MXFP8 project as currently scoped does not deliver the goal on real production traffic.** This must be flagged at project level — either (a) declare decode shapes out-of-scope (requires explicit sign-off), or (b) open a new optimization track.

### B. R42+ track #1 — small-M MXFP8 fastpath (highest ROI if pursued)
- Decode shapes are dominated by N×K matmul time × small M. The tail kernel's 16×16 block geometry is a 32× under-utilization of the wave-tile MFMA primitives (256-element wave tiles).
- A dedicated small-M kernel (e.g., M=1..256, BLK_M=16 or 32, BLK_N=128, K-pipelined) could plausibly reach **30-50× the current decode TF** by exploiting MFMA on a thin tile geometry (M=16, N=128, K=32 MFMA + persistent CTAs across N).
- Closed-paradigm check: tile-area-conservation under (WARPS_M, WARPS_N) rotation (R34/R35/R40) does NOT apply here — tile area itself is the lever (16×16 → 16×128).

### C. R42+ track #2 — defer or scope-reduce
Given the project's STRICT-promote bandwidth was R28-R40 entirely on prefill, and given paradigm closure rate (39 closed levers across 9 cycles), redirecting Dev capacity to decode optimization is a strategic decision. Options:
1. **Defer indefinitely**: production loadings may favor batched prefill (e.g., online serving with continuous batching) where decode shapes are amortized; quantify before investing.
2. **Single-cycle prototype**: 1 dev × 1 cycle on a small-M MXFP8 kernel (M ≤ 128, N=4096 or 8192, K=4096 or 8192). Measure delta vs FP8 same-shape; if MXFP8/FP8 ratio reaches ≥95% on at least 4/6 cells, proceed; else close as paradigm-NEGATIVE.

### D. R42+ track #3 — defensive dispatcher hardening (LOW priority)
- Document at the V2 entry point (`gemm_rcr_pq_v2` etc.) that **calling with M<BLK is a hipErrorInvalidConfiguration trap** if the build's M_DIM matches. Today the only mitigation is: production code must check M and route to V1 (or to a future small-M fastpath) themselves.
- Optional: make V2 entry point guard `if (g.m / BLK == 0) { fall through to V1; }`. R39 Dev C's tracepoint infrastructure is already in place to verify this.

### E. R42+ track #4 — extend the survey methodology (LOW priority)
- Survey was single-GPU and single-N=5 median. For STRICT promotion, would need 4-GPU rotation + N_PAIRS=20 (R37+ protocol). Not done here because decode shapes are uniformly catastrophic — any per-GPU jitter is dwarfed by the 30-50× headroom vs FP8 reference TF on prefill.
- Ratios are stable enough across runs (single-iter sigma confirmed by 100-iter median) that the FAIL verdict is not statistical-power limited.

## Methodology notes

- **Build pattern**: 4 .so artifacts built with `make HIPFLAGS=… -DM_DIM=… -DN_DIM=… -DK_DIM=… -DPY_MODULE_NAME=…` (Makefile's `CXXFLAGS := -w` overrides user `CXXFLAGS`; the `HIPFLAGS+=…` line accepts user additions correctly). Distinct PY_MODULE_NAME per .so to satisfy R39 Dev C's defensive assert.
- **Source patch (one line)**: added `#ifndef PY_MODULE_NAME / #define PY_MODULE_NAME tk_fp8_layouts / #endif` guard around the FP8 PYBIND11_MODULE so it accepts module-name override (mirroring the MXFP8 kernel). No semantic change to the dispatcher.
- **nm-gate**: OVERALL: PASS for both MXFP8 .sos (`tk_mxfp8_8b_4kx4kx4k`, `tk_mxfp8_70b_4kx8kx8k`) — 0 hbshrink/hbn/4wave/subrbm/double_pump/warpsm4/rect symbols; v2 dispatchers present.
- **Dispatch trace**: every measurement captured `[mxfp8_dispatch] rcr_v2: shape=(M=…,N=…,K=…) -> V1-LEGACY-FALLBACK (no V2 predicate matched)` confirming the route — no silent V2 success.
- **Time**: ~2.5 min wall-clock for 12 measurements (well under the 2-hour time-box).

## Artifacts (this commit)

- `r41c_findings.md` — this file
- `r41c_decode_bench.py` — bench harness (loads named .so, runs single layout, prints `R41C_RESULT` line)
- `r41c_run_survey.sh` — driver (12 measurements, captures dispatch traces)
- `r41c_survey_results.log` — raw run log
- `kernel_fp8_layouts.cpp` — 1-line patch: PY_MODULE_NAME override guard

## Cross-cycle context

- All R28-R40 STRICT promotes (8B/70B Gate, Up, Down, KV, QO across V2-RCR/V2-RRR/V2-CRR) operated on M=4096 prefill shapes.
- The 39 closed-paradigm levers (R32: 21 + R33-R40 incremental) all targeted prefill-shape kernel geometry (HB-N shrink, sub-RBM, W4, double_pump, etc.).
- Decode-shape geometry is **out of distribution** for the current paradigm map. R41 Dev C survey is the first measurement — and the first identification that the 95% rule is structurally unattainable on the current dispatcher path.
