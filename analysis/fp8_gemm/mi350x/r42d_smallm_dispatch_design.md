# R42 Dev D — Phase 1: Small-M dispatcher gating design

**Branch**: r42-dev-d
**Date**: 2026-04-18
**Scope**: Design (NOT apply) the dispatcher gating change required for R42 Dev A (M=1) and Dev B (M=32/128) small-M MXFP8 fastpath kernels to be reachable without breaking the existing M≥BLK=256 V2 path.

## Source-of-truth pointers

- `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp:5634-5876` — `dispatch_pq_v2<L>` template (entry from `gemm_{rcr,rrr,crr}_pq_v2`)
- `kernel_mxfp8_layouts.cpp:3675-3677` — `rcr_can_use_exact_8wave_scaled` predicate (`g.m == M_DIM`)
- `kernel_mxfp8_layouts.cpp:3681` — V2 grid math: `(g.m / BLK) * (g.n / BLK)` (returns 0 grid when M < BLK)
- `kernel_mxfp8_layouts.cpp:5505-5532` — V1 dispatch (uses `g.fast_m = (g.m/BLK)*BLK` and falls through to `gemm_tail_kernel` for the leftover M < BLK rows)
- `kernel_mxfp8_layouts.cpp:5340-5403` — `gemm_tail_kernel<L,PRESHUFFLED_QUANT>` (16×16 block, MXFP8 scale-fetch per kk loop iteration — the ~77% MXFP8/FP8 ratio source for M=32/128)
- `kernel_mxfp8_layouts.cpp:5544-5632` — MXFP8_DISPATCH_TRACE infrastructure (R39 Dev C), `MXFP8_DISPATCH_TRACE_ONCE(LAYOUT, NAME, G)` macro

## Problem statement (from R41 Dev C `r41c_findings.md`)

For all M < `BLK = GEMM_BLOCK_SIZE = 256`, the V2 predicate
`rcr_can_use_exact_8wave_scaled(g) := (g.m == M_DIM && g.n == N_DIM && g.k == K_DIM)`
fails (M_DIM compiled to ≥256 in production .so), so the dispatcher reaches the
`V1-LEGACY-FALLBACK` branch (`dispatch<L,true>(g)` at line 5875) → V1 path → because
`g.fast_m = (g.m/BLK)*BLK = 0` for M<256, the V1 launch is skipped and the entire
problem falls into `gemm_tail_kernel`. Tail kernel has no scale-pipelining, no
LDS staging, no MFMA — pure scalar accumulate per output cell. This is the
~0.07–0.88 TFLOPS MXFP8 measurement for the 6 decode shapes.

## Design goals (mandate from R42 task brief)

1. **Reachability**: dispatcher selects new small-M fastpath when M < 256 AND a
   small-M predicate matches.
2. **Fallback discipline**: when no small-M predicate matches (e.g. M ∈ {2..15} or
   N indivisible by BLK_N), route to V1-LEGACY-FALLBACK as today.
3. **Byte-identity for default 8192³**: nm-gate clean — when none of the new
   small-M flags are defined, the dispatcher branches compile to dead code
   (mirror of R35 Dev B / R37 Dev A `MXFP8_CRR_BLK_M=128` pattern). Default
   build sees zero new symbols.
4. **Trace coverage**: every small-M dispatch emits `MXFP8_DISPATCH_TRACE_ONCE`
   with predicate name + BLK_M, so Reviewer Phase 2 grep-verification works
   identically to R37/R38 production wire-ins.

## Selection algorithm (waterfall, ordered)

The new gate is inserted at the **top** of `dispatch_pq_v2<L>` (immediately after
the layout-conditional `g.m=...; g.n=...; g.k=...` extent reads), before the
existing rect / V2-EXACT / V2-HBSHRINK / V2-HBNSHRINK / V2-EXACT-DEFAULT chain.
Selection waterfall (most-specific first):

```text
if (g.m < BLK) {                               // NEW: small-M domain guard
    // R42 Dev A: M=1 (single-token decode) — BLK_M=1 specialised
    if MXFP8_SMALLM_BLK_M_1 defined and predicate matches
        -> dispatch SMALLM-V1-FASTPATH BLK_M=1
    // R42 Dev B: M=32 / M=128 (small-batch decode) — BLK_M ∈ {16, 32}
    if MXFP8_SMALLM_BLK_M_32 defined and predicate matches
        -> dispatch SMALLM-V1-FASTPATH BLK_M=32
    if MXFP8_SMALLM_BLK_M_16 defined and predicate matches
        -> dispatch SMALLM-V1-FASTPATH BLK_M=16
    // No small-M predicate matched — fall through to V1-LEGACY-FALLBACK
    // (this preserves the R41 Dev C measured behavior for unmatched shapes)
    goto v1_legacy_fallback;
}
// existing M≥BLK chain unchanged (rect, V2-EXACT, HBSHRINK, HBNSHRINK, default)
```

Critical property: the entire NEW block is enclosed in a single
`if (g.m < BLK)` guard, so when M ≥ BLK the existing dispatcher chain is
reached **at the same line** as today — byte-identical instruction trace for
the LLaMA prefill 4096³/4096×N×K shapes.

## Patch sketch (Devs A/B integrate)

The patch lives entirely inside the `dispatch_pq_v2<L>` template body. RCR is
the canonical worked example below; RRR/CRR mirror exactly with their own
predicate names + dispatch helper symbols.

```cpp
// kernel_mxfp8_layouts.cpp, dispatch_pq_v2<L> body
//
// Insert immediately after:
//     if constexpr (L == Layout::RCR) {
//         g.m = static_cast<int>(g.c.rows());
//         g.n = static_cast<int>(g.c.cols());
//         g.k = static_cast<int>(g.a.cols());
// (and analogously inside the RRR / CRR `if constexpr` blocks)

// ============================================================================
// R42 Dev D Phase 1 — Small-M dispatcher gating
// ============================================================================
// Routes M < BLK shapes to a dedicated small-M fastpath when one matches.
// Falls through to V1-LEGACY-FALLBACK otherwise (preserves R41 Dev C measured
// behavior for unmatched small-M shapes). Default build with NONE of the
// MXFP8_SMALLM_BLK_M_* macros defined sees this branch compile to a single
// `if (g.m < BLK) { goto v1_fallback; }` short-circuit + zero kernel symbols
// (nm-gate clean). M ≥ BLK is byte-identical to pre-R42 dispatcher.
//
// MXFP8_DISPATCH_TRACE example output (env-gated, R39 Dev C infra):
//   [mxfp8_dispatch] rcr_v2: shape=(M=1,N=4096,K=4096) -> SMALLM-V1-FASTPATH BLK_M=1
//   [mxfp8_dispatch] rcr_v2: shape=(M=32,N=4096,K=4096) -> SMALLM-V1-FASTPATH BLK_M=32
//   [mxfp8_dispatch] rcr_v2: shape=(M=128,N=4096,K=4096) -> SMALLM-V1-FASTPATH BLK_M=32
if (g.m < BLK) {
#if defined(MXFP8_SMALLM_BLK_M_1)
    if (rcr_can_use_smallm_blk_m_1(g)) {                 // Dev A predicate
        MXFP8_DISPATCH_TRACE_ONCE("rcr_v2",
            "SMALLM-V1-FASTPATH BLK_M=1 (R42A)", g);
        dispatch_rcr_smallm_blk_m_1<true>(g);
        return;
    }
#endif
#if defined(MXFP8_SMALLM_BLK_M_32)
    if (rcr_can_use_smallm_blk_m_32(g)) {                // Dev B predicate
        MXFP8_DISPATCH_TRACE_ONCE("rcr_v2",
            "SMALLM-V1-FASTPATH BLK_M=32 (R42B)", g);
        dispatch_rcr_smallm_blk_m_32<true>(g);
        return;
    }
#endif
#if defined(MXFP8_SMALLM_BLK_M_16)
    if (rcr_can_use_smallm_blk_m_16(g)) {                // Dev B alt
        MXFP8_DISPATCH_TRACE_ONCE("rcr_v2",
            "SMALLM-V1-FASTPATH BLK_M=16 (R42B)", g);
        dispatch_rcr_smallm_blk_m_16<true>(g);
        return;
    }
#endif
    // No small-M predicate matched — fall straight to V1-LEGACY-FALLBACK.
    // We do NOT attempt rect/HBSHRINK/EXACT below, because all of those
    // require g.m >= BLK to launch a non-empty grid (R41 Dev C
    // `(g.m / BLK) == 0 -> hipErrorInvalidConfiguration` trap analysis).
    if (::tk_mxfp8_dispatch_trace::trace_enabled()) {
        MXFP8_DISPATCH_TRACE_ONCE("rcr_v2",
            "V1-LEGACY-FALLBACK (M<BLK, no small-M predicate matched)", g);
    }
    dispatch<Layout::RCR, true>(g);
    return;
}
// ============================================================================
// END R42 Dev D Phase 1 — fall through to existing M >= BLK chain unchanged
// ============================================================================
```

## Predicate sketch (Dev A / Dev B implement)

```cpp
// Devs A/B own these — sketched only for the dispatcher integration contract.
// Defined in dedicated .inc files (mirror crr_mxfp8_exact_8wave_hbshrink_fastpath.inc):
//   include/mxfp8_smallm_blk_m_1.inc       (Dev A)
//   include/mxfp8_smallm_blk_m_32.inc      (Dev B)
//   include/mxfp8_smallm_blk_m_16.inc      (Dev B alt)
//
// Each .inc is internally guarded by `#if defined(MXFP8_SMALLM_BLK_M_X)` so
// default build sees an empty translation unit.

__host__ inline bool rcr_can_use_smallm_blk_m_1(const layout_globals& g) {
    return g.m == 1
        && (g.n % BLK_N_SMALLM) == 0          // Dev B picks BLK_N_SMALLM
        && (g.k % BK_SMALLM) == 0
        && g.k >= 2 * BK_SMALLM;              // K-pipelining sanity
}

__host__ inline bool rcr_can_use_smallm_blk_m_32(const layout_globals& g) {
    return g.m == 32
        && (g.n % BLK_N_SMALLM) == 0
        && (g.k % BK_SMALLM) == 0;
}

template<bool PRESHUFFLED_QUANT>
__host__ inline void dispatch_rcr_smallm_blk_m_1(const layout_globals& g) {
    // Grid covers N-direction only (M=1 fits in 1 BLK_M tile)
    const dim3 grid(g.n / BLK_N_SMALLM);
    rcr_smallm_kernel<PRESHUFFLED_QUANT, /*BLK_M=*/1>
        <<<grid, dim3(SMALLM_NUM_THREADS), 0, g.stream>>>(g);
}
```

## RRR / CRR mirror

Identical structure inside the `if constexpr (L == Layout::RRR)` and
`if constexpr (L == Layout::CRR)` blocks of `dispatch_pq_v2`. RRR insertion
point is line 5687-5689 (after the extent reads); CRR insertion point is
line 5699-5701. Each layout gets its own predicate (`rrr_can_use_smallm_*`,
`crr_can_use_smallm_*`) and dispatch helper. Devs A/B may choose to scope to
RCR only initially (decode is RCR-dominant per LLaMA Q/K/V/O proj) and add
RRR/CRR in a follow-up.

## Byte-identity / nm-gate proof

The new block compiles to `if (g.m < BLK) { dispatch<L,true>(g); return; }` when
no `MXFP8_SMALLM_BLK_M_*` macro is defined. For the default 8192³ build:
- `g.m == 8192`, `BLK == 256` → `g.m < BLK` evaluates false at runtime.
- Compiler may keep the comparison + branch (1 SALU instruction), OR may
  fold it (via PGO / inlining of `g.m`). Either way, no new kernel symbols
  emitted.
- Symbol-table check (Dev A/B in their integration commit):
  ```
  $ nm tk_mxfp8_layouts.so | grep -E 'smallm|SMALLM' | wc -l
  0   # default build
  ```
- The `r38_nm_gate.sh` regex set already includes `hbshrink/hbn/4wave/subrbm/double_pump/warpsm4/rect`. Add `smallm` to that regex when integrating.

## Trace verification (Reviewer Phase 2 protocol)

Standard R39+ verification:
```bash
MXFP8_DISPATCH_TRACE=1 python3 r41c_decode_bench.py \
    --shape 1x4096x4096 --layout rcr --kind mxfp8 \
    --module tk_mxfp8_8b_4kx4kx4k_smallm_blk_m_1 2> trace.err
grep '\[mxfp8_dispatch\]' trace.err
# Expect: rcr_v2: shape=(M=1,N=4096,K=4096) -> SMALLM-V1-FASTPATH BLK_M=1 (R42A)
```

If grep is empty for the expected predicate name → CRITICAL: dispatcher did
not reach the predicate. Same failure mode R38 Reviewer caught for the 8B-KV
HB shrink wire-in (commit `66ef02d8`). The MXFP8_DISPATCH_TRACE infrastructure
is unchanged from R39 — small-M extension is purely additive.

## Order-of-operations check

The CRR dispatcher (lines 5697-5841) currently has 8 V2-RRR/V2-RCR advisories
(lines 5756-5781) that fire only under MXFP8_DISPATCH_TRACE. These advisories
fire at M=4096 only — they will not interfere with the small-M block (which
short-circuits at `g.m < BLK`). No advisory restructure needed.

The RECT-V2 CRR fallback (lines 5848-5867) emits an unconditional fprintf for
non-matching shapes when `MXFP8_RECT_BLK_N=64` is defined. The small-M block
sits **above** the existing chain (after extent reads, before any `_rect` or
`_hbshrink` predicate), so it intercepts before the RECT error path. Default
production builds with `MXFP8_RECT_BLK_N=128` are unaffected.

## Open questions for R43+

1. **Persistent-CU dispatch for small-M**: M=1 with BLK_N=128 launches only
   `g.n / 128` blocks (e.g. 32 blocks for N=4096) — heavily underutilizes
   the 304 CUs of MI355X. Persistent-CU pattern (mirror of
   `MXFP8_RCR_V2_PERSISTENT` at line 3690-3699) may apply: launch
   `MXFP8_RCR_V2_PERSISTENT_GRID = 608` and have the kernel early-exit when
   `bid >= g.n / BLK_N_SMALLM`. Defer to Dev A/B based on their measured
   utilization.

2. **K-axis blocking** (Dev A/B kernel decision, dispatcher-neutral):
   `BK_SMALLM=64` vs `BK=128` choice affects `(g.k % BK_SMALLM) == 0`
   predicate but is dispatcher-transparent.

3. **Layout coverage**: LLaMA decode uses RCR for all 4 attn projections
   (Q/K/V/O) and RRR for the 3 MLP proj (gate/up/down). RCR-first is
   correct; RRR is the next priority. CRR is rare on decode (LLaMA does not
   use the (K,M) A-layout on decode).

## Time-box notes

Phase 1 alone: ~1.5 hr design + writeup. Zero kernel changes. Zero GPU usage.
The .inc file scaffolds for predicates / dispatch helpers / kernel templates
are explicitly Devs A/B's responsibility — this doc only specifies the
**dispatcher integration contract** (predicate name conventions + macro names
+ trace string format).
