# R29 L6 V8 (R25E K-loop static peel) — VERDICT: DEAD

**Date**: 2026-04-18
**Optimizer**: B
**Target**: L6 / DLA1 = 4096×32768×128256 (currently 5353.9 TFLOPS via `ts_lgk2_v12_memc_btw_all`, 92.6% of comp 5781.1)
**Goal**: ≥94% via static loop split

## TL;DR

V8 is **DEAD** for L6/DLA1. ISA audit was clean (compiler emitted two distinct loops, fully unrolled peel TAIL, no runtime branches), but at runtime PEEL=2 and PEEL=4 hit `Memory access fault` (write to read-only page) and PEEL=8 ran at 1529 TFLOPS (28% of baseline). All three failure modes ≪ baseline. **STOP** per R29 abort criteria.

## Steps executed

1. **Patch port** (5 min). The R25E worktree (`/shared_nfs/kyle/test/HipKittens/.claude/worktrees/r25e-kpeel/`) had a complete ~150-line implementation that applies cleanly via `patch` to the current kernel (3260 → 3414 lines, +154). Macros: `R25E_K_LOOP_PEEL` (default 0), `R25E_K_LIMIT_LO=65536` (default 65536). When active, main HEAD loop terminates at `bt + 1 + PEEL < k_byte_iters` (full PF), then a `#pragma unroll` PEEL loop runs the last PEEL iters with all `kpair_*<0,...>` calls (PF_N=0 baked compile-time), then the original TAIL block runs `bt = k_byte_iters - 1`.

2. **Builds** (`build_r29_v8_peel.py`, parallel, 5 s wall):
   - PEEL=2: PASS, 260.7 KB, VGPR=212, no spills
   - PEEL=4: PASS, 270.5 KB, VGPR=212
   - PEEL=8: PASS, 290.0 KB, VGPR=212

3. **ISA audit** (PEEL=2 vs PEEL=8): extracted `.hip_fatbin`, unbundled gfx950 image, disassembled.
   - peel2 GPU kernel: 3495 lines, 1152 v_mfma; peel8: 5843 lines, 2688 v_mfma.
   - Δ = +1536 MFMAs = 6 extra peel iters × 256 MFMAs/iter — **math checks exactly**.
   - Only ONE backward branch in K-region (`s_cbranch_scc0 61590` at offset ~0x1080, the main HEAD loop backedge).
   - **No `s_cbranch_execz` or runtime test inside the hot loop.** Compiler did NOT merge.

   ISA AUDIT: **PASS**. The compile-time approach is structurally correct.

4. **Smoke + 5-rep bench** (`bench_r29_v8_peel.py`, GPUs 2,3, warmup=200, iters=500, trim=10%):
   - SMOKE PEEL=2: completed (no fault on 1 launch), SNR = nan (likely correctness collapse, but more importantly the test is huge so reference path may have its own issues — not the headline result here)
   - Run 1 PEEL=2: **`Memory access fault by GPU node-4 ... Write access to a read-only page`** (address 0xbb7c342000)
   - Run 1 PEEL=4: **`Memory access fault by GPU node-5`** (address 0x90366000, "Reason: Unknown")
   - Run 1 PEEL=8: 1529.18 TFLOPS = **28.6% of baseline** (massive perf regression, no fault)

   Aborted bench at this point per R29 stop rule: peel{2,4} faulted, peel8 < 4000.

## Diagnosis

The peel TAIL inlined in the worktree:
- correctly clamps `pf_bt = min(bt+2, k_byte_iters-1)` for global PF param construction (defensive, since `kpair_*<0>` doesn't issue them)
- correctly issues `load_pq_scale_x2_async` with `nxt_scale = (bt+1) << 9` for `bt ∈ [k_byte_iters-1-PEEL, k_byte_iters-2]`, leaving the final scale ready for the existing TAIL at bt=k_byte_iters-1

The issue is NOT in the structure visible at C++ level. The smoking-gun pattern (peel2/4 fault randomly, peel8 doesn't fault but tanks perf) points to:
- Register pressure / scheduling pathology when the unrolled peel TAIL is glued to the unroll-8 main loop. With 2/4 peel iters the compiler may interleave their VMEM/LDS issues with the main-loop tail in a way that produces invalid SRD state at runtime.
- peel8 = 8 iters exactly matches the main loop's `#pragma unroll 8`, so the compiler can keep the same scheduling template — no fault, but it's effectively a duplicate copy of one main-loop unrolled chunk with PF removed → no win, in fact severe slowdown (likely L2 thrash from prefetches issued just before the peel that never get consumed).

This matches the worktree's own warning: the original R25-E "segfaulted" attempt is documented at lines 119-127 of the worktree kernel. The R26-A "robustness fixes" referenced there did not, in fact, make the design robust on this kernel — the worktree never successfully benched at K=128256 (only `r25_reviewer_baseline_check.md` exists, no peel results).

## Verdict

| PEEL | ISA two-loops | runtime  | TFLOPS    | %comp  | %base  |
|------|---------------|----------|-----------|--------|--------|
| 2    | yes           | FAULT    | -         | -      | -      |
| 4    | yes           | FAULT    | -         | -      | -      |
| 8    | yes           | clean    | 1529.18   | 26.45% | 28.56% |

**Verdict: V8 DEAD via this approach.**

ISA-level approach is sound (no runtime branch, two loops emitted). But the peel TAIL inlined body interacts pathologically with the existing tail iteration / scale-loading pattern at runtime. Two of three variants fault; the surviving one regresses to ~28% baseline.

## Recommendation

1. **Do not pursue R25E peel for L6/DLA1 in this kernel shape.** The fault pattern + perf collapse is symptomatic of the K-loop body's stateful dependence on the `pf_*` scale registers being written by the immediately-preceding main-loop iter. Any code that runs between the unroll-8 main loop and the original `bt = k_byte_iters - 1` tail breaks that dependence.

2. **L6 92.6% may be the practical ceiling** under the current `ts_lgk2_v12_memc_btw_all` kernel template. The mega-K nature (501 K-iters) makes K-loop tail PF cost a small fraction of total runtime; the 7.4% gap to comp likely comes from elsewhere (L2 footprint, XCD scheduling, or the aiter ASM having a fundamentally different K-loop epilogue we cannot replicate without a true K_EXACT bypass — which itself triggers aperture violation per R27-C).

3. **If retrying:** a fundamentally different approach would be to build a separate "DLA1-only" kernel that ONLY contains the peeled tail (no main loop). i.e. partial epilogue pre-baked with a smaller `k_byte_iters` constant. This bypasses the inter-loop state hazard but requires nearly a full kernel rewrite.

## Artifacts (uncommitted)

- `analysis/fp8_gemm/mi350x/kernel_mxfp4_gluon_cpp.cpp` — patched (R25E macros + peel TAIL inlined). REVERT before commit.
- `analysis/fp8_gemm/mi350x/build_r29_v8_peel.py` — build script
- `analysis/fp8_gemm/mi350x/bench_r29_v8_peel.py` — bench script (smoke + 5-rep)
- `analysis/fp8_gemm/mi350x/build_all42/tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_v12_memc_btw_all_v8peel{2,4,8}.cpython-310-x86_64-linux-gnu.so`
- `analysis/fp8_gemm/mi350x/R29_V8_BUILD.log`, `R29_V8_BUILD_RESULTS.json`
- `analysis/fp8_gemm/mi350x/R29_V8_BENCH.log`
- ISA dumps: `/tmp/v8peel{2,8}_gpu.s` (3495 / 5843 lines)

Time spent: ~30 min.
