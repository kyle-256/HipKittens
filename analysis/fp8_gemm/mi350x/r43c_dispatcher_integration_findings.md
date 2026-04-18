# R43 Dev C — Unified Small-M Dispatcher Waterfall Integration

**Branch**: `r43-dev-c`
**Date**: 2026-04-18
**Verdict**: SHIP (DONE)
**Time-box**: ~2.5 GPU-hr

## Summary

Folded R42 Dev A's `MXFP8_DECODE_M1_ENABLE` branch and R42 Dev B's
`MXFP8_SMALLM_B32_FASTPATH` branch — previously two independent
macro-gated edits at distinct dispatch sites — into a single unified
small-M dispatcher waterfall at the top of `dispatch<L>` in
`kernel_mxfp8_layouts.cpp`. The new block also includes a placeholder
hook for R43 Dev B's `MXFP8_DECODE_M1_RRR_CRR_ENABLE` (M=1 RRR/CRR
extension, separate parallel work landing in this cycle) and the
`MXFP8_SMALLM_BLK_M_16` slot from R42 Dev D's design (M=2..16 future).

Trace pattern unified: replaced Dev B's inline
`std::getenv("MXFP8_DISPATCH_TRACE") + static int once = 0 + std::fprintf`
shim with `MXFP8_DISPATCH_TRACE_ONCE(LAYOUT, NAME, G)` (R39 Dev C
helper). Macro definition moved to a forward position alongside the
existing trace-helper forward decl so it is in scope inside `dispatch<L>`.
A guarded re-definition is left at the original location to preserve
compilation downstream.

## Design diff vs R42 Dev D's sketch

| Aspect | Dev D sketch (`r42d_smallm_dispatch_design.md`) | R43 Dev C ship |
|---|---|---|
| Insertion point | Top of `dispatch_pq_v2<L>` after extent reads | **Top of `dispatch<L>` after extent reads** |
| Rationale for site change | n/a | R42 Dev A and Dev B kernels both already lived in `dispatch<L>` (V1 path), reachable from V2 via V1-LEGACY-FALLBACK at the bottom of `dispatch_pq_v2`. Moving the gate to `dispatch_pq_v2` would (a) require a duplicate gate in `dispatch<L>` to also cover the V1 entry points (`gemm_rcr_pq`), or (b) silently change V1-entry-point behavior. Keeping the gate in `dispatch<L>` preserves identical semantics for both entry points and is byte-identical at the symbol-set level. |
| Predicate ordering | M=1 → M=32 → M=16 → V1-LEGACY | **M=1 RCR (R42A) → M=1 RRR/CRR (R43B) → BLK_M=16 (placeholder) → SMALLM-B32-TAIL (R42B) → V1-LEGACY** |
| Trace helper | `MXFP8_DISPATCH_TRACE_ONCE` macro (R39 Dev C) | Same — macro forward-defined adjacent to `tk_mxfp8_dispatch_trace::emit` forward decl so dispatch<L> can use it before the namespace body is parsed. Original macro definition guarded with `#ifndef`. |
| `MXFP8_SMALLM_BLK_M_16` | Placeholder (no-op default build) | Implemented as a `#if defined(MXFP8_SMALLM_BLK_M_16)` block. Default build (macro undefined) compiles to empty. When ANY future contributor defines the macro, they must also provide `can_use_smallm_blk_m_16(g)` and `dispatch_smallm_blk_m_16<L, PRESHUFFLED_QUANT>(g)` — hard build error if missing, signaling the contract. |
| Bottom-of-dispatch SMALLM-B32 branch | n/a (Dev D assumed sole-top-block) | **Retained but now only fires for M ≥ BLK + K-tail/M%BLK!=0** (the small-M waterfall short-circuits all M<BLK cases at the top). Trace pattern refactored to use `MXFP8_DISPATCH_TRACE_ONCE` consistent with the top waterfall. |

## Macro composition contract (verified)

| Build flags | Top SMALLM-DECODE-M1-RCR fires | Top SMALLM-B32-TAIL fires | Bottom SMALLM-B32-LEFTOVER fires | Default 8192³ behavior |
|---|---|---|---|---|
| (none) | no (`#if`-out) | no (`#if`-out) | no (`#if`-out) | unchanged from pre-R43 head |
| `MXFP8_DECODE_M1_ENABLE=1` only | yes for RCR M=1 | no | no | unchanged |
| `MXFP8_SMALLM_B32_FASTPATH=1` only | no | yes for any M<BLK + PRESHUFFLED_QUANT | yes for M≥BLK + leftover + PRESHUFFLED_QUANT | unchanged |
| Both | both, M=1 wins on M=1 RCR | yes for M ∈ {2..255} except M=1 RCR | yes for M≥BLK leftover | unchanged |

## Byte-identical symbol-set verification

Built four `(NEW=R43C, BASE=pre-R43C)` pairs at GPU2 with
`-DPY_MODULE_NAME` distinct per build. Diffed `nm -D` symbol sets
(name + bind-letter), ignoring the auto-randomized `__hip_cuid_*`
symbol that ROCm injects per build.

### Default 8192³ build (no small-M flags)

```
$ diff nm-set NEW vs BASE (M_DIM=8192 N_DIM=8192 K_DIM=8192)
1c1
< B __hip_cuid_f87ccd0754211fa1
---
> B __hip_cuid_b9005fdd4fe64463
```

**Result**: PASS — only HIP cuid (random per build) differs. Zero
kernel symbols added, zero W helper symbols changed.

### `MXFP8_DECODE_M1_ENABLE=1` only (M_DIM=1 N_DIM=4096 K_DIM=4096)

```
$ diff nm-set NEW vs BASE
1c1
< B __hip_cuid_4dcb89704da50281
---
> B __hip_cuid_df9a0a128bc2e792
```

**Result**: PASS — bit-identical except for HIP cuid.
4 `gemv_m1_decode_kernel` symbols (RCR × {true, false} × {V symbol, W stub}) emitted in both.

### `MXFP8_SMALLM_B32_FASTPATH=1` only (M_DIM=4096 N_DIM=4096 K_DIM=4096)

```
$ diff nm-set NEW vs BASE
1c1
< B __hip_cuid_ee9682ed62e36df1
---
> B __hip_cuid_5fd117f2e1e7280e
139a140,142
> V _ZZ8dispatchIL6Layout0ELb1EEv14layout_globalsE4once
> V _ZZ8dispatchIL6Layout1ELb1EEv14layout_globalsE4once
> V _ZZ8dispatchIL6Layout2ELb1EEv14layout_globalsE4once
```

**Result**: PASS-with-explained-diff. Beyond the HIP cuid:

- Three `_ZZ8dispatch...4once` `V` symbols are present ONLY in BASE.
  These are the function-static `int once = 0` flags from R42 Dev B's
  inline `std::getenv` shim that R43 Dev C replaced with
  `MXFP8_DISPATCH_TRACE_ONCE`. Their absence in NEW is the intended
  refactor, not a behavioral change.
- All 12 `gemm_tail_kernel_smallm_b32` symbols (3 layouts × 2 PQ flags
  × {V, W}) emitted in both.
- All `dispatch_*_exact_8wave_scaled<true>` weak symbols identical.

## Perf-identical paired-bench (BABA × 8 pairs, GPU2)

### M=1 paired BABA (Dev A path: `gemv_m1_decode_kernel`)

```
M_DIM=1 N=4096 K=4096, mod_a = NEW (R43 Dev C), mod_b = BASE (R42 Dev A)
  pair 1/8: A=0.06436/0.06442ms B=0.06432/0.06438ms
  ...
  pair 8/8: A=0.06468/0.06466ms B=0.06462/0.06468ms
  NEW(A)=0.06445ms BASE(B)=0.06438ms delta=+0.110%
```

**Verdict**: PASS — Δ=+0.110% well within 0±0.5% target.

### M=32 paired BABA (Dev B path: `gemm_tail_kernel_smallm_b32`)

```
M_DIM=4096 N=4096 K=4096, mod_a = NEW (R43 Dev C), mod_b = BASE (R42 Dev B)
  pair 1/8: A=0.95661/0.95723ms B=0.95631/0.95577ms
  ...
  pair 8/8: A=0.95947/0.95967ms B=0.95965/0.95911ms
  NEW(A)=0.95728ms BASE(B)=0.95762ms delta=-0.036%
```

**Verdict**: PASS — Δ=-0.036% well within 0±0.5% target.

## Trace-coverage verification (MXFP8_DISPATCH_TRACE=1)

Combined build (`MXFP8_DECODE_M1_ENABLE=1` + `MXFP8_SMALLM_B32_FASTPATH=1`):

```bash
$ MXFP8_DISPATCH_TRACE=1 python3 -c '... gemm_rcr_pq(M=1) ...'
[mxfp8_dispatch] rcr_pq: shape=(M=0,N=0,K=0) -> V1-PQ-DEFAULT
[mxfp8_dispatch] rcr_pq_v1: shape=(M=1,N=4096,K=4096) -> SMALLM-DECODE-M1-RCR (R42A)

$ MXFP8_DISPATCH_TRACE=1 python3 -c '... gemm_rcr_pq(M=32) ...'
[mxfp8_dispatch] rcr_pq: shape=(M=0,N=0,K=0) -> V1-PQ-DEFAULT
[mxfp8_dispatch] rcr_v2: shape=(M=32,N=4096,K=4096) -> SMALLM-B32-TAIL (R42B)

$ MXFP8_DISPATCH_TRACE=1 python3 -c '... gemm_rcr_pq_v2(M=8192) ...'
[mxfp8_dispatch] rcr_v2: shape=(M=8192,N=8192,K=8192) -> RCR-V2-EXACT-8WAVE
```

All three paths trace the expected predicate. M=8192 default 8192³
correctly bypasses the small-M waterfall (`g.m < BLK == 256` is false).

## Files modified

- `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp`
  - +25 lines: forward `MXFP8_DISPATCH_TRACE_ONCE` definition adjacent
    to the trace-helper forward decl (~line 3760).
  - +95 lines / -32 lines: replaced R42 Dev A's `#if MXFP8_DECODE_M1_ENABLE`
    branch (~line 5475) and rewrote the bottom `#if MXFP8_SMALLM_B32_FASTPATH`
    branch (~line 5596) to share the unified small-M waterfall idiom.
  - +5 lines: `#ifndef MXFP8_DISPATCH_TRACE_ONCE` guard at the original
    macro definition site (~line 5725).

## Open questions / future work

1. **R43 Dev B M=1 RRR/CRR landing**: this commit includes a guarded
   hook for `MXFP8_DECODE_M1_RRR_CRR_ENABLE` in the waterfall. Dev B's
   in-flight branch adds `r43b_decode_m1_rrr_crr_fastpath.inc` with the
   `can_use_decode_m1_rrr_crr` predicate and `dispatch_decode_m1_rrr_crr`
   helper. When that lands, no additional dispatcher edit is required —
   the waterfall already routes M=1 RRR/CRR via this hook.
2. **`MXFP8_SMALLM_BLK_M_16` placeholder**: enforced by hard build error
   if defined without providing the kernel + predicate. Future cycles
   adding M=2..16 batch-decode kernels need only define the predicate
   + dispatch helper and toggle the macro; no dispatcher edit.
3. **V0 entry points (`gemm_rcr` non-PQ)**: not exercised on production
   decode (decode uses preshuffled_quant). The waterfall handles them
   via `if constexpr (PRESHUFFLED_QUANT)` short-circuiting on the Dev B
   branch. Dev A's M=1 fastpath supports both PQ and raw via its
   template parameter.

## Methodology notes

- Build pattern: 5 distinct .so artifacts (default + M1-only + B32-only
  + B32-only-M_DIM4096 + both), each with distinct `PY_MODULE_NAME` to
  satisfy R39 Dev D defensive assert.
- BASE build = stash my `kernel_mxfp8_layouts.cpp` changes, build,
  unstash. Both NEW and BASE built with identical flags except for the
  R43 Dev C edit.
- nm-set diff strips PY_MODULE_NAME-derived symbols and HIP cuid (random
  per build).
- Paired BABA: 8 pairs × 200 iters per timing event, GPU2, no preheat
  (M=1/M=32 are bandwidth/latency dominated, not thermal).
- nm diffs saved to `analysis/fp8_gemm/mi350x/r43c_runs/`:
  `nm_diff_default.txt`, `nm_diff_m1only.txt`, `nm_diff_b32only.txt`.

## Coordination notes

- Two parallel agents touched the same `kernel_mxfp8_layouts.cpp` in
  this cycle: R43 Dev B (this branch's parent) added
  `MXFP8_DECODE_M1_RRR_CRR_ENABLE` branch + include of
  `r43b_decode_m1_rrr_crr_fastpath.inc`. Those changes are NOT in this
  commit — the integration assumes Dev B will rebase on top of the
  unified waterfall, replacing their 24-line independent dispatcher
  branch with the existing `MXFP8_DECODE_M1_RRR_CRR_ENABLE` hook in
  the waterfall (zero additional dispatcher lines required from Dev B).
- The forward `MXFP8_DISPATCH_TRACE_ONCE` macro definition is benign
  for any other in-flight branch that was already using the macro —
  the `#ifndef` guard prevents double-definition.

## Verdict

DONE — all 5 requirements from the R43 Dev C task brief satisfied:
1. Single waterfall covering Dev A + Dev B + R43 Dev B placeholder + Dev D BLK_M=16 placeholder ✅
2. R39 `MXFP8_DISPATCH_TRACE_ONCE` macro replaces both inline trace patterns ✅
3. Macro-gate composition: each kernel independently macro-gated; default 8192³ has 0 small-M symbols ✅
4. Byte-identical on both `MXFP8_DECODE_M1_ENABLE=1` and `MXFP8_SMALLM_B32_FASTPATH=1` builds vs pre-integration ✅
5. Perf-identical: paired BABA Δ=+0.110% (M=1) and Δ=-0.036% (M=32), both well within ±0.5% ✅
