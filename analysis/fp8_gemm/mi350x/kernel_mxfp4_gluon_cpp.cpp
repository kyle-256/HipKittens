// MXFP4 Gluon-architecture kernel: C++ implementation
//
// Replicates Gluon a4w4 core algorithm:
//   - N-split: B tile → left/right halves (128 each)
//   - DOT_left / DOT_right pipeline per K-iteration
//   - Double-buffered async tile copy (buffer_load_to_lds)
//   - KPAIR MFMAs (op_sel for sub-group, op_sel_hi for K-phase, no remap_phase)
//   - Preshuffle-quant scales
//
// Per K-iteration:
//   Load A+B_left from LDS → DOT_left (A×B_left, 64 MFMAs)
//   Load B_right from LDS → DOT_right (A×B_right, 64 MFMAs)
//   Barrier → Prefetch next tiles

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <type_traits>
#include <cstdlib>
#include <cstdio>
using namespace kittens;

#ifndef M_DIM
#define M_DIM 8192
#endif
#ifndef K_DIM
#define K_DIM 8192
#endif
#ifndef N_DIM
#define N_DIM 8192
#endif

#ifndef STEP3_EMBED_BARRIER
#define STEP3_EMBED_BARRIER 1
#endif
#ifndef STEP3_BARRIER_VMCNT
#define STEP3_BARRIER_VMCNT 8
#endif
#ifndef SWAP_STEP34_MAIN
#define SWAP_STEP34_MAIN 0
#endif
#ifndef SWAP_STEP12_MAIN
#define SWAP_STEP12_MAIN 0
#endif
#if SWAP_STEP12_MAIN && !SWAP_STEP34_MAIN
#error "SWAP_STEP12_MAIN requires SWAP_STEP34_MAIN"
#endif
#ifndef SPREAD_LDS
#define SPREAD_LDS 0
#endif
#ifndef NONVOLATILE_SCALE_X2_POC
#define NONVOLATILE_SCALE_X2_POC 1
#endif
// R34-A Finding C: scale-load granularity reduction — replace 2× buffer_load_dwordx2
// per K-iter with 4× single buffer_load_dword (same call sites, different asm).
// When SCALE_LOAD_X1=1, load_pq_scale_x2_async issues two single-dword loads (offset
// 0 and offset +4) instead of one dwordx2. This matches aiter's pattern that decouples
// scale-arrival latency from the iter-boundary stall by feeding the LSU smaller, more
// frequent loads. Default 0 → bit-for-bit identical to current 41/42 incumbent.
#ifndef SCALE_LOAD_X1
#define SCALE_LOAD_X1 0
#endif
#ifndef STEP4_EXTERNAL_BR_PREFETCH
#define STEP4_EXTERNAL_BR_PREFETCH 0
#endif
#ifndef MAIN_PERMLANE_BF16_STORE_POC
#define MAIN_PERMLANE_BF16_STORE_POC 0
#endif
#if MAIN_PERMLANE_BF16_STORE_POC && !SWAP_STEP34_MAIN
#error "MAIN_PERMLANE_BF16_STORE_POC requires SWAP_STEP34_MAIN"
#endif
#if MAIN_PERMLANE_BF16_STORE_POC && !SWAP_STEP12_MAIN
#error "MAIN_PERMLANE_BF16_STORE_POC requires SWAP_STEP12_MAIN"
#endif

#ifndef STEP12_BR_LGKMCNT
#define STEP12_BR_LGKMCNT 0
#endif

#ifndef TAIL_BARRIER_VMCNT
#define TAIL_BARRIER_VMCNT STEP3_BARRIER_VMCNT
#endif

#ifndef FUSED_STEP34
#define FUSED_STEP34 0
#endif

// R43 Opt A.fix1 (2026-04-19): R43A_GATE_PF_TAIL_KBOUND
// Default OFF. When enabled together with FUSED_STEP34=1 + TAIL_SPLIT=1, gates
// the unconditional `emit_pf_tail<0>(pf_a0_p, pf_a1_p)` and
// `emit_pf_tail<0>(pf_bl_p, pf_br_p)` calls that follow the fused step34 asm
// block (kernel line ~3222). Skips the prefetch on the LAST steady-state iter
// (bt == k_byte_iters - 2) where pf_bt clamps to k_byte_iters-1 and the
// in-flight LDS write races the TAIL_SPLIT tail handler's reads. Closes the
// HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION on the K=28672 CRASH shapes
// without exposing the 17%-bf16-overflow correctness bug that stripping
// FUSED_STEP34 or TAIL_SPLIT alone would re-introduce. See R43_OPT_A_VERDICT.md.
// No effect when FUSED_STEP34=0; no behavior change on non-CRASH variants when
// the macro is unset.
#ifndef R43A_GATE_PF_TAIL_KBOUND
#define R43A_GATE_PF_TAIL_KBOUND 0
#endif

// R37 Fix B (2026-04-19): the non-fused step3+step4 path emits 4 separate
// `asm volatile` blocks; the compiler is free to interleave clobbering moves
// between them which corrupts the upper-left 128x128 quadrant of every 256x256
// output tile (acc_A0Bl). Fix: on the default (non-FUSED_STEP34) path, fuse
// step3+step4 into a single asm block via kpair_64mfma_step34 — preserving the
// R25-C tail-pf-off, BARRIER_TO_WAITCNT_ALL, and K_EXACT branching that the
// original FUSED_STEP34=1 path bypassed. Defaults ON; set R37_FIX_B=0 to revert
// to the legacy buggy code path (for comparison only).
#ifndef R37_FIX_B
#define R37_FIX_B 1
#endif

// R38 Opt B (2026-04-19): tail-iter prefetch hardening.
// Background: 9/42 R37 BEST_VARIANTS shapes CRASH with HSA_STATUS_ERROR_MEMORY
// _APERTURE_VIOLATION after a few hundred kernel invocations. All 9 use a
// `_pfoff*` variant that drives `R25C_TAIL_PF_OFF_ITERS > 0` together with the
// fused step34 backport. The single-rep correctness check passes (finite ≈
// 0.999) — the fault is intermittent and only appears under stress.
//
// Mechanism: in the R37 fix-B path, `make_pf_params(...)` is called
// UNCONDITIONALLY at the top of every K-loop iteration even when
// `_r25c_tail_no_pf` is true. Building the `tile_pf_params` struct (~32 bytes
// of `voffs`/`lds_addrs` per tile, x4 tiles = 128 bytes/iter) inflates VGPR
// pressure for the rest of the iter. With `-mllvm -amdgpu-disable-clustered-
// low-occupancy-reschedule` (in 5/9 CRASH variants) the spill scheduler
// produces a frame index that, after the (bt+2 vs k_byte_iters) clamping
// branch is folded by the unrolled loop, ends up referencing scratch slots
// that were sized for the non-tail path. Those scratch loads/stores fault
// once the GPU's scratch-bound speculation is exercised by repeat launches.
//
// Fix B1 (chosen): hoist `make_pf_params` INSIDE the `if (!_r25c_tail_no_pf)`
// branch on the R37 path so the struct construction (and its per-iter
// register footprint) is skipped on tail iters. Also wrap the PFs in an
// `asm volatile("" ::: "memory")` pair so the compiler cannot speculate the
// load-lds intrinsics across the tail-skip branch.
//
// Default OFF; the R38B builder sets R38B_TAIL_FIX=1 explicitly. This change
// only touches the R37 fix-B branch (kernel line ~2863), not the FUSED_STEP34
// branch nor the legacy path.
#ifndef R38B_TAIL_FIX
#define R38B_TAIL_FIX 0
#endif

// R38 Opt C (2026-04-19): Fix B3 — route tail-iter prefetches through the
// L2-only path (`emit_full_pf_l2only<>`), instead of either skipping (R37
// default = crash) or always-emitting LDS-bound prefetches (R38B = perf
// regression). The L2-only path issues the SAME `buffer_load_dwordx4` GMEM
// fetches (so the compiler-tracked vmcnt remains consistent with the iter's
// scheduler footprint) but discards the result into a scratch VGPR (no LDS
// write, no double-buffer slot collision). This should:
//   - eliminate CRASH (the wait-count metadata stays in sync, no in-flight
//     buffer-load-to-LDS racing the next iter's s_barrier);
//   - preserve R37's WINs (no extra LDS write traffic, the tail iter's MFMAs
//     still consume the SAME stale data they did under R25-C/R37, plus the
//     new L2 fetches warm the cache for the prologue/epilogue Bl direct-load).
// The scope of the change is the same `R37_FIX_B && !FUSED_STEP34` branch
// that R38B targets. R38C must NOT be enabled together with R38B.
// Default OFF; the R38C builder sets R38C_TAIL_L2ONLY=1 explicitly.
#ifndef R38C_TAIL_L2ONLY
#define R38C_TAIL_L2ONLY 0
#endif
#if R38C_TAIL_L2ONLY && R38B_TAIL_FIX
#error "R38C_TAIL_L2ONLY and R38B_TAIL_FIX are mutually exclusive"
#endif

// R38 Opt F (2026-04-19): explicit tail-iter LDS-write drain. Per R38C verdict,
// the structural fix for the 6 WIN→WRONG_OUTPUT and 6 CRASH→WRONG_OUTPUT demotions
// is in the R37_FIX_B fused-step34 path's tail iters: the next iter's MFMA reads
// from an LDS double-buffer slot whose write from the prior iter's
// buffer_load_to_lds prefetch hasn't fully drained (the in-flight load lands
// AFTER the next iter's s_barrier already released the slot for reuse). Without
// fixing this, R37_FIX_B + R25-C produces stale-LDS reads → WRONG_OUTPUT.
//
// R38F inserts an explicit `s_waitcnt vmcnt(N) [+ s_barrier]` at the END of
// each tail iter (after both step12 and step34/prefetch emission), guaranteeing
// all outstanding GMEM→LDS loads land before the next iter's s_barrier releases
// the LDS slot. Three variants:
//   F1 (R38F_VARIANT=1): `s_waitcnt vmcnt(0)` only — drain GMEM, no WG sync.
//   F2 (R38F_VARIANT=2): `s_waitcnt vmcnt(0)` + `s_barrier` — drain + sync.
//   F3 (R38F_VARIANT=3): `s_waitcnt 0` (vmcnt+lgkmcnt+expcnt) — most conservative.
//   F4 (R38F_VARIANT=4): `s_waitcnt vmcnt(0) lgkmcnt(0)` — drain GMEM + LDS.
// Default OFF (R38F_TAIL_DRAIN=0); the R38F builder sets it explicitly.
// Drain fires ONLY on the same tail iters R25-C identifies (when
// _r25c_tail_no_pf is true), AND only on the R37_FIX_B path. No effect on
// FUSED_STEP34 or on legacy paths. INDEPENDENT of R38B/R38C — those should
// remain default OFF when testing R38F.
#ifndef R38F_TAIL_DRAIN
#define R38F_TAIL_DRAIN 0
#endif
#ifndef R38F_VARIANT
#define R38F_VARIANT 2  // default to F2 (drain + barrier) when ON
#endif
// R38F may be combined with R38B (always-emit) — in fact it must be, to close
// the underlying CRASH first; R38F then fixes the stale-LDS WRONG_OUTPUT that
// R38B reveals. Keep mutex only with R38C (L2-only path takes a different
// emit_full_pf_l2only target).
#if R38F_TAIL_DRAIN && R38C_TAIL_L2ONLY
#error "R38F_TAIL_DRAIN and R38C_TAIL_L2ONLY are mutually exclusive (different prefetch targets)"
#endif

// R39 Opt A (2026-04-19): TAIL_SCALE_CLAMP. Per R38_OPT_F_VERDICT root-cause
// analysis: in R37_FIX_B + R25-C, the data-tile prefetch is suppressed on the
// last R25C_TAIL_PF_OFF_ITERS iters (`if (!_r25c_tail_no_pf) emit_pf_tail<0>`),
// while `load_pq_scale_x2_async(... bt+1 ...)` keeps advancing the scale index.
// On those tail iters the next-iter MFMA reads STALE data (the last-prefetched
// tile, which is iter `k_byte_iters - 1 - R25C_TAIL_PF_OFF_ITERS`) but a FRESH
// scale (pointing at iter `bt+1` past the last fetched data tile). Scale-vs-data
// mismatch under uniform scale=-4 produces BF16-overflow garbage on ~17% of
// cells (matches the project memory `MXFP4 17%-deterministic-wrong cells`).
//
// R39A clamps the `nxt_scale` index to match the data-tile clamp pattern: when
// `_r25c_tail_no_pf` is true, do not advance the scale; reuse the current iter's
// scale slot (idx = bt). When R25-C is inactive (default tail_off=0), the
// natural `bt+1` index is preserved (no behavior change on R37 WIN shapes).
//
// Macro is gated `R37_FIX_B && !FUSED_STEP34` (same scope as R38B/C/F). Default
// OFF; the R39A builder sets `R39A_TAIL_SCALE_CLAMP=1` explicitly.
//
// Mutex: independent of R38B/C/F (operates on scale-load index, not data-pf
// emission). May be combined with any of them if needed for incremental tests.
#ifndef R39A_TAIL_SCALE_CLAMP
#define R39A_TAIL_SCALE_CLAMP 0
#endif

// R40 Opt A (2026-04-19): per-iter PF FENCE for the R37_FIX_B path.
//
// Hypothesis: the compiler reorders the per-iter `buffer_load_to_lds` issue
// (driven by `pf_*_p` struct construction at the TOP of the iter +
// `emit_pf_tail` after step34) ACROSS the `kpair_64mfma_step12` asm boundary.
// R37 added a fence AFTER step34, but did NOT add one BEFORE step12 — and the
// pf_*_p struct construction sits free in scheduler-land at the top of the
// iter, before step12. Result: a buffer_load_to_lds may land in an LDS slot
// the in-flight MFMA is reading. Manifests as the cluster-A "borderline"
// 24-50 dB SNR / 0.5-1.5% wrong cells on ~7 shapes.
//
// R40A surgical change in the `R37_FIX_B && !FUSED_STEP34` branch:
//   1) Insert `asm volatile("" ::: "memory")` IMMEDIATELY BEFORE every call to
//      `kpair_64mfma_step12` in the R37_FIX_B branch (TAIL_SPLIT and
//      no-TAIL_SPLIT main loops).
//   2) MOVE the `make_pf_params` block from the TOP of the iter to AFTER
//      `kpair_64mfma_step34` returns. Construction is then folded into the
//      same pre-emit_pf_tail span the R38B_TAIL_FIX branch already uses.
//
// Default OFF. The R40A builder sets `-DR40A_PF_FENCE=1` explicitly. Has no
// effect on the R37_FIX_B == 0 legacy path nor on FUSED_STEP34 == 1.
//
// Notes:
//  - When R40A_PF_FENCE && !R38B_TAIL_FIX: skip the per-iter top-of-loop
//    `make_pf_params` (defer to inside the R37_FIX_B branch, mirroring R38B's
//    deferred-construction pattern). When R38B_TAIL_FIX is ON, R38B already
//    defers the construction, so R40A only adds the pre-step12 fence.
//  - This change is INDEPENDENT of R39A/R38C/R38F. It can be combined with
//    R38B (R38B already builds the params after step34 inside the always-emit
//    branch); R40A then only adds the pre-step12 fence on top.
#ifndef R40A_PF_FENCE
#define R40A_PF_FENCE 0
#endif

// R38 Opt A (2026-04-19): replace the `__builtin_amdgcn_raw_buffer_load_lds`
// intrinsic in emit_one_pf with an `asm volatile("buffer_load_dwordx4 ... lds")`
// block carrying a "memory" clobber. The intrinsic, being a regular call, is
// schedulable by LLVM (and especially by `-mllvm -amdgpu-sched-strategy=
// max-memory-clause`) across iteration boundaries. Inline asm with "memory"
// clobber is a hard ordering barrier — the compiler cannot reorder loads/stores
// across it. Targets the 19 WRONG_OUTPUT shapes from R37 whose BEST_VARIANTS
// flag stack still contains some scheduler-aggressive flag.
//
// IMPORTANT: when ON, the cache_hint argument is dropped (the inline asm uses
// the default cache mode — sc0=sc1=nt=0). All R37 BEST_VARIANTS use cache_all
// (=0), so this is a no-op for the R37 set. Default OFF; the R38A builder sets
// R38A_INLINE_BUFLOAD_LDS=1 explicitly.
#ifndef R38A_INLINE_BUFLOAD_LDS
#define R38A_INLINE_BUFLOAD_LDS 0
#endif

// R25-C: K-loop tail epilogue specialization. When set to N>0, the last N
// iterations of the steady-state main loop use PF_N=0 (no global prefetch) for
// the Step3/Step4 KPAIR calls. Rationale: clamped pf_bt re-fetches the same
// trailing K-tile, wasting saturated VMEM slots on stale lines. Default 0 →
// baseline (no behavior change). Only takes effect on TAIL_SPLIT=1, non-FUSED
// path (the path used by all DLA shapes).
//
// Empirical (R25C smoke 2026-04-18): runtime tail branch FOLDS for shapes that
// fully unroll (k_byte_iters ≤ 32, e.g. K=4096 → 16 iters, DLA2/DLA7) → WIN.
// For shapes where the K-loop is `pragma unroll 8`-only (k_byte_iters > 32,
// e.g. K=128256 → 501 iters, DLA1) the branch becomes a runtime check inside
// the hot loop and code-size doubles → catastrophic regression.
// Therefore we gate R25C on K_DIM ≤ 32768 (≤ 128 K-tiles), where pragma unroll
// fully unrolls and the branch folds.
#ifndef R25C_TAIL_PF_OFF_ITERS
#define R25C_TAIL_PF_OFF_ITERS 0
#endif

// R41 Opt A (2026-04-19): Cluster C deep-K tail-pf-off SWEEP + extract_tile fence.
// Targets the 5 catastrophic K=32768 shapes (~97% wrong cells, ~10% finite under R40B).
// Hypothesis: R25C_TAIL_PF_OFF_ITERS=120 with K_DIM=32768 (k_byte_iters=128) means PF
// runs only on first 8 iters, then kernel rides on extract_tile-staged registers for
// >100 iters. Combined with FUSED_STEP34=1's fewer iter-boundary fences, the
// tile-register liveness across deep K is the dominant corruption vector.
// R40D's data (no-prefetch IMPROVES 4 of 5 cluster-C shapes) supports this.
//
// R41A_DEEP_K_FIX (master gate, default 0): when 0, behavior is bit-identical to R40B.
// R41A_PFOFF_OVERRIDE (int, default 0): when nonzero AND R41A_DEEP_K_FIX AND
//   K_DIM >= 16384, OVERRIDES the variant-supplied R25C_TAIL_PF_OFF_ITERS at
//   compile time (via #undef + #define).
// R41A_EXTRACT_TILE_FENCE (bool, default 0): when 1 AND R41A_DEEP_K_FIX AND
//   FUSED_STEP34 AND K_DIM >= 16384, inserts s_waitcnt vmcnt(0) immediately
//   BEFORE every extract_tile(nxt_a0_d, tA0) and extract_tile(nxt_bl_d, tBl)
//   call in the K-loop body (3 sites: lines ~2851, ~3364, ~3598).
#ifndef R41A_DEEP_K_FIX
#define R41A_DEEP_K_FIX 0
#endif
#ifndef R41A_PFOFF_OVERRIDE
#define R41A_PFOFF_OVERRIDE 0
#endif
#ifndef R41A_EXTRACT_TILE_FENCE
#define R41A_EXTRACT_TILE_FENCE 0
#endif

// R42 Opt C (2026-04-19): broaden the R41A extract_tile vmcnt fence beyond
// deep-K only. R41A's gating was (FUSED_STEP34 && K_DIM >= 16384). The same
// load->extract race may exist at smaller K with a smaller window. Two opt-in
// broadenings:
//   R42C_FENCE_NO_K_GUARD (default 0): when 1 AND R41A_DEEP_K_FIX AND
//     R41A_EXTRACT_TILE_FENCE AND FUSED_STEP34, drop the K_DIM>=16384 guard
//     so the fence applies for any K. (C1)
//   R42C_FENCE_ANY_PATH (default 0): when 1 AND R41A_DEEP_K_FIX AND
//     R41A_EXTRACT_TILE_FENCE, drop BOTH the FUSED_STEP34 and K_DIM guards.
//     Fence emitted at every extract_tile site. (C2)
// Default behavior is BIT-IDENTICAL to R41A (both default OFF).
//
// ORIGINAL (R41A) gate, for reviewer reference (non-default change verifier):
//   #if R41A_DEEP_K_FIX && R41A_EXTRACT_TILE_FENCE && FUSED_STEP34 && (K_DIM >= 16384)
//     #define R41A_FENCE_BEFORE_EXTRACT() asm volatile("s_waitcnt vmcnt(0)" ::: "memory")
//   #else
//     #define R41A_FENCE_BEFORE_EXTRACT() ((void)0)
//   #endif
#ifndef R42C_FENCE_NO_K_GUARD
#define R42C_FENCE_NO_K_GUARD 0
#endif
#ifndef R42C_FENCE_ANY_PATH
#define R42C_FENCE_ANY_PATH 0
#endif

#if R41A_DEEP_K_FIX && (R41A_PFOFF_OVERRIDE != 0) && (K_DIM >= 16384)
  #undef R25C_TAIL_PF_OFF_ITERS
  #define R25C_TAIL_PF_OFF_ITERS R41A_PFOFF_OVERRIDE
#endif

#if R41A_DEEP_K_FIX && R41A_EXTRACT_TILE_FENCE && R42C_FENCE_ANY_PATH
  // C2: any K, any path (fused or not)
  #define R41A_FENCE_BEFORE_EXTRACT() \
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory")
#elif R41A_DEEP_K_FIX && R41A_EXTRACT_TILE_FENCE && FUSED_STEP34 && R42C_FENCE_NO_K_GUARD
  // C1: any K, but only on FUSED_STEP34 path
  #define R41A_FENCE_BEFORE_EXTRACT() \
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory")
#elif R41A_DEEP_K_FIX && R41A_EXTRACT_TILE_FENCE && FUSED_STEP34 && (K_DIM >= 16384)
  // R41A original: deep-K only on FUSED_STEP34 path
  #define R41A_FENCE_BEFORE_EXTRACT() \
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory")
#else
  #define R41A_FENCE_BEFORE_EXTRACT() ((void)0)
#endif

#ifndef R25C_K_LIMIT
#define R25C_K_LIMIT 32768
#endif
// R25-G: optional EXACT-K gate. When R25C_K_EXACT > 0, R25C only fires for the
// exact specified K_DIM. Used to ship per-K-iter-count tuned pfoff values
// (e.g. K=14336→pfoff54, K=32768→pfoff124) without their flag set bleeding
// into other shapes (where the wrong pfoff would zero out all prefetches).
// Default 0 → behavior unchanged (only the K_LIMIT gate applies).
#ifndef R25C_K_EXACT
#define R25C_K_EXACT 0
#endif
#define R25C_ACTIVE ((R25C_TAIL_PF_OFF_ITERS > 0) && (K_DIM <= R25C_K_LIMIT) \
                     && (R25C_K_EXACT == 0 || K_DIM == R25C_K_EXACT))

#ifndef DIRECT_BL
#define DIRECT_BL 0
#endif
#if DIRECT_BL && FUSED_STEP34
#error "DIRECT_BL is incompatible with FUSED_STEP34"
#endif

// EARLY_BL_PF: when DIRECT_BL=1, issue the next-iter Bl buffer_load_dwordx4 BEFORE
// Step12 instead of inside Step4. This gives ~128 MFMAs (~512 cycles) of latency
// hiding for the ~400-cycle buffer_load, vs the original ~32 cycles in Step4 alone.
// Step4 then runs as pure MFMAs + Br prefetch (Bl already in VGPR).
#ifndef EARLY_BL_PF
#define EARLY_BL_PF 0
#endif
#if EARLY_BL_PF && !DIRECT_BL
#error "EARLY_BL_PF requires DIRECT_BL=1"
#endif

// EARLY_SCALE_PF: hoist next-iter scale buffer_load_dwordx2 to before _raw copy
// using shadow regs nxt_pf_* with writeback at iter end.
//
// ⚠ BROKEN — DO NOT ENABLE. The compiler aliases pf_* and nxt_pf_* to the same
// VGPRs (since `pf_* = nxt_pf_*` writeback looks like a no-op assignment), so the
// outstanding VMEM clobbers pf_* before _raw reads it → race condition (different
// NaN pattern vs baseline; torch.equal == False on 1024×1024×4096 corr test).
//
// Tried fixes that don't work: (a) volatile asm + 8 extra VGPRs → exceeds 256-VGPR
// cap, no occupancy gain; (b) restructure pf_* as [2] array → invasive AND
// no perf benefit since baseline ASM already issues scale loads at iter top
// (NONVOLATILE_SCALE_X2_POC=1 lets the compiler schedule them ~512 cyc before use).
//
// Round 5 dead-end (2026-04-17). Bench (broken variant, for completeness):
//   14336x4096x32768   baseline 4742.6 → early_scale 4702.3  (-0.85%)
//   16384x4096x28672   baseline 5005.9 → early_scale 4982.3  (-0.47%)
//   4096x32768x128256  baseline 5105.9 → early_scale 5123.9  (+0.35%)
// All deltas within run-to-run noise. See test_early_scale_pf.py for repro.
#ifndef EARLY_SCALE_PF
#define EARLY_SCALE_PF 0
#endif
#if EARLY_SCALE_PF
#error "EARLY_SCALE_PF is BROKEN (compiler aliases shadow regs → race)."
#endif

#ifndef NT_STORE
#define NT_STORE 0
#endif

#ifndef PACKED_STORE
#define PACKED_STORE 0
#endif

// PERSISTENT_XCD: launch a fixed-size grid (PERSISTENT_GRID workgroups), each WG
// pulling tile_id atomically from a global counter. Maintains XCD-aware ordering
// by constructing raw_bid = tile_id (so the existing % NUM_XCDS swizzle still works).
#ifndef PERSISTENT_XCD
#define PERSISTENT_XCD 0
#endif

// STATIC_XCD_REMAP (Round 7, Optimizer B): atomic-free static bid->(m,n) remap
// that constrains each XCD to a narrow N-strip of width (bpc / NUM_XCDS).
// Within the XCD: GROUP_M m-tiles × n_per_xcd n-tiles tiled walk for L2 B-tile reuse.
// Requires bpc % NUM_XCDS == 0 — falls back to default mapping otherwise.
#ifndef STATIC_XCD_REMAP
#define STATIC_XCD_REMAP 0
#endif
#ifndef PERSISTENT_GRID
// Default: 8 XCDs * 38 CUs * 2 WGs/CU = 608.  MI355X has 304 CUs total.
#define PERSISTENT_GRID 608
#endif
#ifndef PERSISTENT_BATCH
// Tiles claimed per atomicAdd (1 = one-at-a-time, 4 = grab 4 sequential tiles).
#define PERSISTENT_BATCH 1
#endif

// OPTC flags (default-off, source-level scheduling/codegen hints)
#ifndef WAVE_PRIO_HIGH
#define WAVE_PRIO_HIGH 0
#endif
#ifndef WAVE_PRIO_LOW_TAIL
#define WAVE_PRIO_LOW_TAIL 0
#endif
#ifndef SCHED_GROUP_BARRIERS
#define SCHED_GROUP_BARRIERS 0
#endif
#ifndef EXPLICIT_S_NOP
#define EXPLICIT_S_NOP 0
#endif

// ───── R22A: LDS sub-arbitration probes ─────
// LDS_RD_STAGGER_NOP=N inserts `s_nop N-1` after each ds_read_b128 in the 3
// hot-path KPAIR functions (kpair_32mfma_with_lds_and_pf,
// kpair_32mfma_with_lds_rowspread_pf, kpair_32mfma_with_lds_and_pf_swapped_sel).
// Default 0 means empty string, so default-built kernel is bit-for-bit identical.
//
// Hypothesis (R21-recon): TCP_TA_DATA_STALL = 167-294 % of GRBM with 0 % LDS bank
// conflict means port-side sub-arbitration (multiple ds_read_b128 per cycle
// exceeding LDS port bandwidth). Spreading them by 1-2 cycles via s_nop should
// give the LDS arbiter time to drain. NOP cost is hidden by MFMA pipeline (16
// cycles per MFMA), so as long as MFMA throughput remains saturated this is
// expected to be free.
#ifndef LDS_RD_STAGGER_NOP
#define LDS_RD_STAGGER_NOP 0
#endif
#if LDS_RD_STAGGER_NOP == 1
  #define LDS_NOP_STR "s_nop 0\n"
#elif LDS_RD_STAGGER_NOP == 2
  #define LDS_NOP_STR "s_nop 1\n"
#elif LDS_RD_STAGGER_NOP == 3
  #define LDS_NOP_STR "s_nop 2\n"
#else
  #define LDS_NOP_STR ""
#endif

// ───── R21B: macro hook helpers (no-op when defaults are 0) ─────
// These expand to nothing in baseline so a default-built kernel is bit-for-bit
// identical. Site-call macros are dropped at the 4 K-iter end points + tail +
// pre-store-C for opt-in scheduling-hint experiments.
#if EXPLICIT_S_NOP > 0
  #if EXPLICIT_S_NOP >= 3
    #define MXFP4_R21B_S_NOP_HOOK \
      do { asm volatile("s_nop 0"); asm volatile("s_nop 0"); asm volatile("s_nop 0"); } while (0)
  #elif EXPLICIT_S_NOP == 2
    #define MXFP4_R21B_S_NOP_HOOK \
      do { asm volatile("s_nop 0"); asm volatile("s_nop 0"); } while (0)
  #else
    #define MXFP4_R21B_S_NOP_HOOK do { asm volatile("s_nop 0"); } while (0)
  #endif
#else
  #define MXFP4_R21B_S_NOP_HOOK do {} while (0)
#endif

// ───── R22C: finer-mask sched_group_barrier control ─────
// R21B used mask=0xff (ALL), which over-constrained the post-RA scheduler and
// regressed -1.6 to -3.5%. R22C adds finer per-class masks and tunable group
// size + per-site enable bitmask. SCHED_GROUP_BARRIERS=1 still works as the
// R21B "wide mask=0xff" path; setting R22C_SCHED_MASK!=0 overrides the mask.
//   R22C_SCHED_MASK : AMDGCN inst-class mask (0x004=MFMA, 0x040=DS_R, 0x080=DS_W,
//                     0x008=VMEM, 0x044=MFMA+DS_R, 0x0c0=DS_R+DS_W, etc.).
//                     Set non-zero to opt in to R22C.
//   R22C_SCHED_SIZE : group size (2nd arg to sched_group_barrier). R21B used 1.
//   R22C_HOOK_MASK  : bitmask of hook sites
//                     bit0..bit2 = the 3 K-iter end sites (lines ~2380/2605/2808)
//                     bit3       = pre-Store-C site (line ~2823)
//                     Defaults to 0xf = all 4 sites (matches R21B coverage).
#ifndef R22C_SCHED_MASK
#define R22C_SCHED_MASK 0
#endif
#ifndef R22C_SCHED_SIZE
#define R22C_SCHED_SIZE 1
#endif
#ifndef R22C_HOOK_MASK
#define R22C_HOOK_MASK 0xf
#endif

#if R22C_SCHED_MASK
  // R22C overrides: caller picked an explicit mask. Implies SCHED_GROUP_BARRIERS.
  #define MXFP4_R21B_SCHED_GROUP_HOOK \
    do { __builtin_amdgcn_sched_group_barrier(R22C_SCHED_MASK, R22C_SCHED_SIZE, 0); } while (0)
#elif SCHED_GROUP_BARRIERS
  // R21B legacy path: mask=0xff matches all instruction classes; size=1 group.
  #define MXFP4_R21B_SCHED_GROUP_HOOK \
    do { __builtin_amdgcn_sched_group_barrier(0xff, 1, 0); } while (0)
#else
  #define MXFP4_R21B_SCHED_GROUP_HOOK do {} while (0)
#endif

#if WAVE_PRIO_LOW_TAIL
  #define MXFP4_R21B_PRIO_LOW_HOOK \
    do { asm volatile("s_setprio 0" ::: "memory"); } while (0)
#else
  #define MXFP4_R21B_PRIO_LOW_HOOK do {} while (0)
#endif

// R22C indexed iter-end hooks: each call site uses an _IDX form so individual
// sites can be disabled via R22C_HOOK_MASK. The R21B convenience macro is
// retained as a no-op-by-default base. EXPLICIT_S_NOP is independent of the
// sched_group hook and is always emitted (gated by its own macro) regardless
// of R22C_HOOK_MASK so existing R21B s_nop combos still work.
#define MXFP4_R21B_ITER_END_HOOK \
  do { MXFP4_R21B_S_NOP_HOOK; MXFP4_R21B_SCHED_GROUP_HOOK; } while (0)

#if (R22C_HOOK_MASK) & 0x1
  #define MXFP4_R22C_ITER_END_HOOK_0 \
    do { MXFP4_R21B_S_NOP_HOOK; MXFP4_R21B_SCHED_GROUP_HOOK; } while (0)
#else
  #define MXFP4_R22C_ITER_END_HOOK_0 do { MXFP4_R21B_S_NOP_HOOK; } while (0)
#endif
#if (R22C_HOOK_MASK) & 0x2
  #define MXFP4_R22C_ITER_END_HOOK_1 \
    do { MXFP4_R21B_S_NOP_HOOK; MXFP4_R21B_SCHED_GROUP_HOOK; } while (0)
#else
  #define MXFP4_R22C_ITER_END_HOOK_1 do { MXFP4_R21B_S_NOP_HOOK; } while (0)
#endif
#if (R22C_HOOK_MASK) & 0x4
  #define MXFP4_R22C_ITER_END_HOOK_2 \
    do { MXFP4_R21B_S_NOP_HOOK; MXFP4_R21B_SCHED_GROUP_HOOK; } while (0)
#else
  #define MXFP4_R22C_ITER_END_HOOK_2 do { MXFP4_R21B_S_NOP_HOOK; } while (0)
#endif
#if (R22C_HOOK_MASK) & 0x8
  #define MXFP4_R22C_PRESTORE_HOOK MXFP4_R21B_SCHED_GROUP_HOOK
#else
  #define MXFP4_R22C_PRESTORE_HOOK do {} while (0)
#endif

// ───── R22B: B-tile / A-tile cache-hint streaming (frees L2 BW for the partner) ─────
// Hypothesis (R21-recon PMC): TCP_DATA_STALL = 290%+ on DLA2/DLA7 with HBM at only
// 17–20% of peak — TCP/L2 throughput is the limit, not bandwidth. B-tile is
// broadcast across all M-tile rows of the same (BlockX,BlockY); on large-N shapes
// (DLA2 N=32768, DLA7 N=32768) B-tile traffic evicts A-tile L2 lines. Marking
// B-tile loads as `cache_stream` (GLC=1) bypasses L2 for B → frees L2 for A.
//
// Values:
//   B_LOAD_NONTEMPORAL=1 → B-tile uses `coherency::non_temporal`  (slc + glc bypass)
//   B_LOAD_NONTEMPORAL=2 → B-tile uses `coherency::cache_stream`  (glc only)
//   A_LOAD_NONTEMPORAL=1/2 → same for A-tile (likely worse on high-A-reuse shapes)
#ifndef B_LOAD_NONTEMPORAL
#define B_LOAD_NONTEMPORAL 0
#endif
#ifndef A_LOAD_NONTEMPORAL
#define A_LOAD_NONTEMPORAL 0
#endif

#if B_LOAD_NONTEMPORAL == 0
  #define MXFP4_R22B_B_HINT (kittens::coherency::cache_all)
#elif B_LOAD_NONTEMPORAL == 1
  #define MXFP4_R22B_B_HINT (kittens::coherency::non_temporal)
#elif B_LOAD_NONTEMPORAL == 2
  #define MXFP4_R22B_B_HINT (kittens::coherency::cache_stream)
#else
  #error "B_LOAD_NONTEMPORAL must be 0, 1, or 2"
#endif

#if A_LOAD_NONTEMPORAL == 0
  #define MXFP4_R22B_A_HINT (kittens::coherency::cache_all)
#elif A_LOAD_NONTEMPORAL == 1
  #define MXFP4_R22B_A_HINT (kittens::coherency::non_temporal)
#elif A_LOAD_NONTEMPORAL == 2
  #define MXFP4_R22B_A_HINT (kittens::coherency::cache_stream)
#else
  #error "A_LOAD_NONTEMPORAL must be 0, 1, or 2"
#endif

#define MXFP4_STR_IMPL(x) #x
#define MXFP4_STR(x) MXFP4_STR_IMPL(x)

// ───── R18A-P3: replace inner-loop s_barrier with cheaper s_waitcnt lgkmcnt(0) ─────
// The inner-loop `s_waitcnt vmcnt(N) s_barrier` synchronizes the producer chain
// (buffer_load_to_lds → LDS) with the consumer (Step3 ds_read). The s_barrier
// also provides cross-wave LDS visibility. If wave-private LDS partitions allow
// safe drop of cross-wave sync (each wave only reads what it wrote), we can drop
// the barrier and only wait for in-wave LDS+VMEM completion via lgkmcnt+vmcnt.
//
// SAFETY: drop only via SNR check at M=N=K=4096. If output diverges, the cross-
// wave barrier IS load-bearing — discard variant.
#ifndef BARRIER_TO_WAITCNT_STEP3
#define BARRIER_TO_WAITCNT_STEP3 0
#endif
#ifndef BARRIER_TO_WAITCNT_STEP12
#define BARRIER_TO_WAITCNT_STEP12 0
#endif
#ifndef BARRIER_TO_WAITCNT_ALL
#define BARRIER_TO_WAITCNT_ALL 0
#endif

// ───── R20C: K-loop sync coarsening ─────
// Hypothesis (from R17A DLA1 profile): for very-large-K shapes, the per-iter
// inner-loop sync (vmcnt+lgkmcnt[+s_barrier]) dominates the K-loop epilogue
// overhead (2004 iters × per-iter sync ≈ 0.6-1.2 ms wasted on DLA1).
//
// K_LOOP_SYNC_EVERY_2: call the embedded-barrier (or waitcnt-only)
// kpair_32mfma_with_lds_and_pf_swapped_sel on EVEN-bt iters, and call the
// no-barrier variant on ODD-bt iters. Existing 2-buffer LDS rotation:
// iter N writes A*_db[N&1], iter N reads A*_db[1-(N&1)] (i.e. what was
// written in iter N-1). Skipping the barrier on ODD bt means iter N+1 may
// start its ds_read of A*_db[N&1] before iter N's buffer_load_to_lds for
// that slot has finished — UNLESS the natural lgkmcnt+vmcnt of step12+step3
// + the s_waitcnt lgkmcnt(0) at line 2235/2283 + the inherent SIMD
// scheduling of MFMAs covers the gap.
//
// SAFETY: this is a CORRECTNESS-RISKY rewrite. Validate via SNR (both
// uniform and random-scale aperture probe per R18A's lesson).
#ifndef K_LOOP_SYNC_EVERY_2
#define K_LOOP_SYNC_EVERY_2 0
#endif
#ifndef K_LOOP_SYNC_EVERY_4
#define K_LOOP_SYNC_EVERY_4 0
#endif

#if BARRIER_TO_WAITCNT_ALL
#undef BARRIER_TO_WAITCNT_STEP3
#define BARRIER_TO_WAITCNT_STEP3 1
#undef BARRIER_TO_WAITCNT_STEP12
#define BARRIER_TO_WAITCNT_STEP12 1
#endif

#if BARRIER_TO_WAITCNT_STEP3
// Drop s_barrier from STEP3 producer→consumer sync — wait only for VMEM + LDS.
#define MXFP4_STEP3_BARRIER_INST "s_waitcnt vmcnt(" MXFP4_STR(STEP3_BARRIER_VMCNT) ") lgkmcnt(0)\n"
#else
#define MXFP4_STEP3_BARRIER_INST "s_waitcnt vmcnt(" MXFP4_STR(STEP3_BARRIER_VMCNT) ")\ns_barrier\n"
#endif

#if BARRIER_TO_WAITCNT_STEP12
// Drop s_barrier from TAIL_BARRIER (last-iter sync before tail Step3/4).
#define MXFP4_TAIL_BARRIER_INST "s_waitcnt vmcnt(" MXFP4_STR(TAIL_BARRIER_VMCNT) ") lgkmcnt(0)\n"
#else
#define MXFP4_TAIL_BARRIER_INST "s_waitcnt vmcnt(" MXFP4_STR(TAIL_BARRIER_VMCNT) ")\ns_barrier\n"
#endif

// ───── R19B: per-site (and vmcnt-relax) finer-grained barrier control ─────
// Allow individual sites to be flipped to waitcnt-only without affecting other sites.
// Each per-site macro defaults to the corresponding aggregate macro (STEP3 or STEP12)
// so existing R18A behavior is preserved bit-exactly.
//
// STEP3 sites (live with default STEP3_EMBED_BARRIER=1):
//   _S2 = template kpair_32mfma_with_lds_and_pf            (line ~1259)
//   _S3 = template kpair_32mfma_with_lds_rowspread_pf      (line ~1452)
//   _S4 = template kpair_32mfma_with_lds_and_pf_swapped_sel (line ~1677)
// STEP3 sites (dead with default STEP3_EMBED_BARRIER=1):
//   _S1 = kpair_64mfma_step34 entry (FUSED_STEP34=1 only,  line ~1022)
//   _S5 = TAIL_SPLIT inner            (!STEP3_EMBED_BARRIER, line ~2127)
//   _S6 = TAIL_SPLIT outer            (!STEP3_EMBED_BARRIER, line ~2314)
//   _S7 = no-TAIL_SPLIT outer         (!STEP3_EMBED_BARRIER, line ~2505)
// TAIL/STEP12 sites:
//   STEP12_S1 = TAIL_SPLIT==1 path (line ~2213, live for all 4 parents)
//   STEP12_S2 = TAIL_SPLIT==0 path (line ~2400, dead with TAIL_SPLIT=1)

#ifndef BARRIER_TO_WAITCNT_STEP3_S1
#define BARRIER_TO_WAITCNT_STEP3_S1 (BARRIER_TO_WAITCNT_STEP3)
#endif

// R37 Fix B (and FUSED_STEP34): the fused kpair_64mfma_step34 reads from LDS
// double-buffer slots that the previous iteration's buffer_load_to_lds writes.
// Cross-warp synchronization REQUIRES an actual s_barrier — converting the S1
// barrier to a waitcnt-only causes ~10-30% non-finite output (R36 _f34 builds
// with BARRIER_TO_WAITCNT_ALL=1 produced finite_frac ≈ 0.87 for that reason).
// Force S1 back to s_barrier whenever the fused path is active, regardless of
// BARRIER_TO_WAITCNT_ALL / BARRIER_TO_WAITCNT_STEP3 / explicit S1 override.
#if (R37_FIX_B || FUSED_STEP34) && BARRIER_TO_WAITCNT_STEP3_S1
#undef BARRIER_TO_WAITCNT_STEP3_S1
#define BARRIER_TO_WAITCNT_STEP3_S1 0
#endif
#ifndef BARRIER_TO_WAITCNT_STEP3_S2
#define BARRIER_TO_WAITCNT_STEP3_S2 (BARRIER_TO_WAITCNT_STEP3)
#endif
#ifndef BARRIER_TO_WAITCNT_STEP3_S3
#define BARRIER_TO_WAITCNT_STEP3_S3 (BARRIER_TO_WAITCNT_STEP3)
#endif
#ifndef BARRIER_TO_WAITCNT_STEP3_S4
#define BARRIER_TO_WAITCNT_STEP3_S4 (BARRIER_TO_WAITCNT_STEP3)
#endif
#ifndef BARRIER_TO_WAITCNT_STEP3_S5
#define BARRIER_TO_WAITCNT_STEP3_S5 (BARRIER_TO_WAITCNT_STEP3)
#endif
#ifndef BARRIER_TO_WAITCNT_STEP3_S6
#define BARRIER_TO_WAITCNT_STEP3_S6 (BARRIER_TO_WAITCNT_STEP3)
#endif
#ifndef BARRIER_TO_WAITCNT_STEP3_S7
#define BARRIER_TO_WAITCNT_STEP3_S7 (BARRIER_TO_WAITCNT_STEP3)
#endif
#ifndef BARRIER_TO_WAITCNT_STEP12_S1
#define BARRIER_TO_WAITCNT_STEP12_S1 (BARRIER_TO_WAITCNT_STEP12)
#endif
#ifndef BARRIER_TO_WAITCNT_STEP12_S2
#define BARRIER_TO_WAITCNT_STEP12_S2 (BARRIER_TO_WAITCNT_STEP12)
#endif

// vmcnt override: when nonzero, replaces STEP3_BARRIER_VMCNT in the per-site STEP3
// strings (and TAIL_BARRIER_VMCNT in the STEP12 strings). 0 (default) = use existing.
// NB: applies regardless of barrier-vs-waitcnt mode — relaxed/tight vmcnt can be
// independently swept.
#ifndef BARRIER_TO_WAITCNT_RELAXED_VMCNT
#define BARRIER_TO_WAITCNT_RELAXED_VMCNT 0
#endif

#if BARRIER_TO_WAITCNT_RELAXED_VMCNT > 0
#define MXFP4_R19B_VMCNT_STR MXFP4_STR(BARRIER_TO_WAITCNT_RELAXED_VMCNT)
#define MXFP4_R19B_TAIL_VMCNT_STR MXFP4_STR(BARRIER_TO_WAITCNT_RELAXED_VMCNT)
#else
#define MXFP4_R19B_VMCNT_STR MXFP4_STR(STEP3_BARRIER_VMCNT)
#define MXFP4_R19B_TAIL_VMCNT_STR MXFP4_STR(TAIL_BARRIER_VMCNT)
#endif

// Per-site barrier strings. Each picks waitcnt-only OR vmcnt+s_barrier per its own gate.
#if BARRIER_TO_WAITCNT_STEP3_S1
#define MXFP4_STEP3_BARRIER_INST_S1 "s_waitcnt vmcnt(" MXFP4_R19B_VMCNT_STR ") lgkmcnt(0)\n"
#else
#define MXFP4_STEP3_BARRIER_INST_S1 "s_waitcnt vmcnt(" MXFP4_R19B_VMCNT_STR ")\ns_barrier\n"
#endif
#if BARRIER_TO_WAITCNT_STEP3_S2
#define MXFP4_STEP3_BARRIER_INST_S2 "s_waitcnt vmcnt(" MXFP4_R19B_VMCNT_STR ") lgkmcnt(0)\n"
#else
#define MXFP4_STEP3_BARRIER_INST_S2 "s_waitcnt vmcnt(" MXFP4_R19B_VMCNT_STR ")\ns_barrier\n"
#endif
#if BARRIER_TO_WAITCNT_STEP3_S3
#define MXFP4_STEP3_BARRIER_INST_S3 "s_waitcnt vmcnt(" MXFP4_R19B_VMCNT_STR ") lgkmcnt(0)\n"
#else
#define MXFP4_STEP3_BARRIER_INST_S3 "s_waitcnt vmcnt(" MXFP4_R19B_VMCNT_STR ")\ns_barrier\n"
#endif
#if BARRIER_TO_WAITCNT_STEP3_S4
#define MXFP4_STEP3_BARRIER_INST_S4 "s_waitcnt vmcnt(" MXFP4_R19B_VMCNT_STR ") lgkmcnt(0)\n"
#else
#define MXFP4_STEP3_BARRIER_INST_S4 "s_waitcnt vmcnt(" MXFP4_R19B_VMCNT_STR ")\ns_barrier\n"
#endif
#if BARRIER_TO_WAITCNT_STEP3_S5
#define MXFP4_STEP3_BARRIER_INST_S5 "s_waitcnt vmcnt(" MXFP4_R19B_VMCNT_STR ") lgkmcnt(0)\n"
#else
#define MXFP4_STEP3_BARRIER_INST_S5 "s_waitcnt vmcnt(" MXFP4_R19B_VMCNT_STR ")\ns_barrier\n"
#endif
#if BARRIER_TO_WAITCNT_STEP3_S6
#define MXFP4_STEP3_BARRIER_INST_S6 "s_waitcnt vmcnt(" MXFP4_R19B_VMCNT_STR ") lgkmcnt(0)\n"
#else
#define MXFP4_STEP3_BARRIER_INST_S6 "s_waitcnt vmcnt(" MXFP4_R19B_VMCNT_STR ")\ns_barrier\n"
#endif
#if BARRIER_TO_WAITCNT_STEP3_S7
#define MXFP4_STEP3_BARRIER_INST_S7 "s_waitcnt vmcnt(" MXFP4_R19B_VMCNT_STR ") lgkmcnt(0)\n"
#else
#define MXFP4_STEP3_BARRIER_INST_S7 "s_waitcnt vmcnt(" MXFP4_R19B_VMCNT_STR ")\ns_barrier\n"
#endif

#if BARRIER_TO_WAITCNT_STEP12_S1
#define MXFP4_TAIL_BARRIER_INST_S1 "s_waitcnt vmcnt(" MXFP4_R19B_TAIL_VMCNT_STR ") lgkmcnt(0)\n"
#else
#define MXFP4_TAIL_BARRIER_INST_S1 "s_waitcnt vmcnt(" MXFP4_R19B_TAIL_VMCNT_STR ")\ns_barrier\n"
#endif
#if BARRIER_TO_WAITCNT_STEP12_S2
#define MXFP4_TAIL_BARRIER_INST_S2 "s_waitcnt vmcnt(" MXFP4_R19B_TAIL_VMCNT_STR ") lgkmcnt(0)\n"
#else
#define MXFP4_TAIL_BARRIER_INST_S2 "s_waitcnt vmcnt(" MXFP4_R19B_TAIL_VMCNT_STR ")\ns_barrier\n"
#endif

constexpr int BLK = 256;
constexpr int BK  = 128;
constexpr int WARPS_M = 2, WARPS_N = 2;
constexpr int _NUM_WARPS   = WARPS_M * WARPS_N;
constexpr int _NUM_THREADS = _NUM_WARPS * WARP_THREADS;
constexpr int HB = BLK / 2;
constexpr int RBM = HB / WARPS_M;
constexpr int RBN = HB / WARPS_N;

constexpr int K_BYTES = K_DIM / 2;
constexpr int k_byte_iters = K_BYTES / BK;

using ST_tile = st_fp8e4m3<HB, BK, st_16x128_s>;
using A_row_reg = rt_fp8e4m3<RBM, BK, row_l, rt_16x128_s>;
using B_row_reg = rt_fp8e4m3<RBN, BK, row_l, rt_16x128_s>;
using RT_C = rt_fl<RBM, RBN, col_l, rt_16x16_s>;

using G = kittens::group<_NUM_WARPS>;
using _gl_fp4   = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_scale = gl<fp8e8m0, -1, -1, -1, -1>;
using _gl_bf16  = gl<bf16, -1, -1, -1, -1>;

struct gluon_globals {
    _gl_fp4 a, b;
    _gl_scale a_scale, b_scale;
    _gl_bf16 c;
#if DIRECT_BL
    _gl_fp4 b_ps;  // Preshuffled B for half-direct-Bl loading
#endif
    float scale = 1.0f;
};

using fp4_intx8_t   = int __attribute__((__vector_size__(8 * sizeof(int))));
using fp4_intx4_t   = int __attribute__((__vector_size__(4 * sizeof(int))));
using fp4_floatx4_t = float __attribute__((__vector_size__(4 * sizeof(float))));
using u32x2_t       = unsigned int __attribute__((ext_vector_type(2)));
static_assert(sizeof(u32x2_t) == 8);

__device__ __forceinline__ unsigned int pack_bf16x2(float x, float y) {
    unsigned int out;
    asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2"
        : "=v"(out) : "v"(x), "v"(y));
    return out;
}

// Store bf16 with optional non-temporal hint (NT_STORE=1 bypasses L2 for writes)
static __device__ __forceinline__ void store_bf16_val(bf16* addr, float val) {
    bf16 v = base_types::convertor<bf16, float>::convert(val);
#if NT_STORE
    unsigned short u;
    __builtin_memcpy(&u, &v, 2);
    __builtin_nontemporal_store(u, reinterpret_cast<unsigned short*>(addr));
#else
    *addr = v;
#endif
}

// Packed dword store: two bf16 values packed via v_cvt_pk_bf16_f32
static __device__ __forceinline__ void store_bf16x2_packed(bf16* addr, float v0, float v1) {
    unsigned int packed = pack_bf16x2(v0, v1);
#if NT_STORE
    __builtin_nontemporal_store(packed, reinterpret_cast<unsigned int*>(addr));
#else
    *reinterpret_cast<unsigned int*>(addr) = packed;
#endif
}

__device__ __forceinline__ fp4_intx4_t fp4_lo4(const fp4_intx8_t& x) {
    return __builtin_shufflevector(x, x, 0, 1, 2, 3);
}
__device__ __forceinline__ fp4_intx4_t fp4_hi4(const fp4_intx8_t& x) {
    return __builtin_shufflevector(x, x, 4, 5, 6, 7);
}

struct alignas(16) gluon_acc {
    fp4_floatx4_t regs[32]; // [0..15] = A0×B_half, [16..31] = A1×B_half
};

// ── LDS → register tile load ──

template<ducks::rt::row_layout RT, ducks::st::all ST>
__device__ __forceinline__ void fp4_load_st_to_rt(RT &dst, const ST &src) {
    static_assert(RT::rows == ST::rows && RT::cols == ST::cols);
    using T = typename base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename ST::dtype;
    constexpr int packing = base_types::packing<typename RT::dtype>::num();
    static_assert(std::is_same_v<T, U>);
    const int laneid = kittens::laneid();
    const int row_offset = laneid % dst.base_tile_rows;
    const int col_offset = dst.base_tile_stride * (laneid / dst.base_tile_rows);
    const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);
    constexpr int reg_sub_row = ST::underlying_subtile_cols / RT::base_tile_cols;
    constexpr int reg_sub_col = ST::underlying_subtile_rows / RT::base_tile_rows;
    #pragma unroll 8
    for (int k = 0; k < RT::base_tile_num_strides; k++)
        #pragma unroll 8
        for (int i = 0; i < reg_sub_col; i++)
            #pragma unroll 8
            for (int j = 0; j < reg_sub_row; j++) {
                const int row = i * RT::base_tile_rows + row_offset;
                const int col = j * RT::base_tile_cols + col_offset +
                    k * RT::base_tile_elements_per_stride_group;
                const uint32_t offset = sizeof(U) * (src_ptr + row * ST::underlying_subtile_cols + col);
                const uint32_t addr = offset ^ (((offset % (16 * 128)) >> 8) << 4);
                const int idx = k * RT::base_tile_stride / packing;
                #pragma unroll 8
                for (int ii = 0; ii < ST::subtiles_per_col; ii++)
                    #pragma unroll 8
                    for (int jj = 0; jj < ST::subtiles_per_row; jj++) {
                        const int sid = ii * ST::underlying_subtiles_per_row + jj;
                        const int soff = sid * ST::underlying_subtile_bytes;
                        asm volatile(
                            "ds_read_b128 %0, %1 offset:%2\n"
                            : "=v"(*reinterpret_cast<float4*>(
                                  &dst.tiles[ii * reg_sub_col + i][jj * reg_sub_row + j].data[idx]))
                            : "v"(addr), "i"(soff) : "memory"
                        );
                    }
            }
}

template<ducks::rt::row_layout RT>
__device__ __forceinline__ fp4_intx8_t fp4_extract_tile(const RT &src, int tile_row) {
    return *reinterpret_cast<const fp4_intx8_t*>(&src.tiles[tile_row][0].data[0]);
}

__device__ __forceinline__ void extract_dsread_tile(const float4 d[8], fp4_intx8_t t[4]) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        auto lo = *reinterpret_cast<const fp4_intx4_t*>(&d[i]);
        auto hi = *reinterpret_cast<const fp4_intx4_t*>(&d[i + 4]);
        t[i][0] = lo[0]; t[i][1] = lo[1]; t[i][2] = lo[2]; t[i][3] = lo[3];
        t[i][4] = hi[0]; t[i][5] = hi[1]; t[i][6] = hi[2]; t[i][7] = hi[3];
    }
}

// ── Scale helpers ──

__device__ __forceinline__ const uint8_t* preshuffled_scale_row_base_ptr(
    const _gl_scale& src, int row_group) {
    return reinterpret_cast<const uint8_t*>(src.raw_ptr + src.idx(coord<>(row_group, 0)));
}

__device__ __forceinline__ i32x4 make_scale_srd(const uint8_t* ptr) {
    i32x4 srd = std::bit_cast<i32x4>(make_buffer_resource(
        static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(ptr)),
        0xFFFFFFFFu, 0x00110000u));
    srd[0] = __builtin_amdgcn_readfirstlane(srd[0]);
    srd[1] = __builtin_amdgcn_readfirstlane(srd[1]);
    srd[2] = __builtin_amdgcn_readfirstlane(srd[2]);
    srd[3] = __builtin_amdgcn_readfirstlane(srd[3]);
    return srd;
}

__device__ __forceinline__ fp8e8m0_4 load_pq_scale_srd(
    i32x4 srsrc, uint32_t voffset, uint32_t soffset) {
    return std::bit_cast<fp8e8m0_4>(
        llvm_amdgcn_raw_buffer_load_b32(srsrc, voffset, soffset, 0));
}

// Load two consecutive scale dwords via buffer_load_dwordx2 (merged preshuffle format).
// Non-volatile asm allows compiler scheduling flexibility while preserving dwordx2.
//
// R34-A: when SCALE_LOAD_X1=1, replace dwordx2 with 2× single dword loads (lo at
// soffset, hi at soffset+4). Aiter's pattern uses dword-granularity scale loads
// that interleave better with the MFMA chain than 8-byte dwordx2 loads pinching
// the LSU at the iter boundary. Per-iter byte budget is identical (8 bytes/SRD)
// but issue count doubles (2 issues/SRD vs 1).
__device__ __forceinline__ void load_pq_scale_x2_async(
    i32x4 srsrc, uint32_t voffset, uint32_t soffset,
    fp8e8m0_4 &out_lo, fp8e8m0_4 &out_hi) {
#if SCALE_LOAD_X1
    uint32_t lo, hi;
#if NONVOLATILE_SCALE_X2_POC
    asm(
        "buffer_load_dword %0, %1, %2, %3 offen"
        : "=v"(lo)
        : "v"(voffset), "s"(srsrc), "s"(soffset)
    );
    asm(
        "buffer_load_dword %0, %1, %2, %3 offen offset:4"
        : "=v"(hi)
        : "v"(voffset), "s"(srsrc), "s"(soffset)
    );
#else
    asm volatile(
        "buffer_load_dword %0, %1, %2, %3 offen"
        : "=v"(lo)
        : "v"(voffset), "s"(srsrc), "s"(soffset)
    );
    asm volatile(
        "buffer_load_dword %0, %1, %2, %3 offen offset:4"
        : "=v"(hi)
        : "v"(voffset), "s"(srsrc), "s"(soffset)
    );
#endif
    out_lo = std::bit_cast<fp8e8m0_4>(lo);
    out_hi = std::bit_cast<fp8e8m0_4>(hi);
#else
    uint64_t pair;
#if NONVOLATILE_SCALE_X2_POC
    asm(
        "buffer_load_dwordx2 %0, %1, %2, %3 offen"
        : "=v"(pair)
        : "v"(voffset), "s"(srsrc), "s"(soffset)
    );
#else
    asm volatile(
        "buffer_load_dwordx2 %0, %1, %2, %3 offen"
        : "=v"(pair)
        : "v"(voffset), "s"(srsrc), "s"(soffset)
    );
#endif
    out_lo = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(pair));
    out_hi = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(pair >> 32));
#endif
}

// ── Half-direct Bl loading from preshuffled global memory ──
#if DIRECT_BL
// Preshuffled B layout: [N0, K0, KLane=4, NLane=16, KPack_bytes=16]
// N0_stride = K0 * 4 * 16 * 16 = K0 * 1024 = 16 * K_BYTES
// K0_stride = 4 * 16 * 16 = 1024
// Per-lane voff: k_lane * 256 + n_lane * 16 = (lid/16)*256 + (lid%16)*16
// soff per (n0, k0): n0 * N0_stride + k0 * K0_stride
constexpr uint32_t BL_N0_STRIDE = 16 * K_BYTES;
constexpr uint32_t BL_K0_STRIDE = 1024;

// Issue 8 buffer_load_dwordx4 for one Bl tile from preshuffled memory.
// Loads 4 subtiles (N0 groups) × 2 K-phases, NO waitcnt (caller manages).
// R22B: B_LOAD_NONTEMPORAL drives cache hint for direct-Bl path.
__device__ __forceinline__ void load_bl_direct_async(
    float4 dst[8],
    i32x4 srd, uint32_t voff, uint32_t soff_base)
{
    constexpr int B_HINT = static_cast<int>(MXFP4_R22B_B_HINT);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t si = soff_base + i * BL_N0_STRIDE;
        dst[i]   = std::bit_cast<float4>(
            llvm_amdgcn_raw_buffer_load_b128(srd, voff, si, B_HINT));
        dst[i+4] = std::bit_cast<float4>(
            llvm_amdgcn_raw_buffer_load_b128(srd, voff, si + BL_K0_STRIDE, B_HINT));
    }
}

#endif

// ── Tile prefetch ──

static constexpr int PF_MPT = (HB * BK * sizeof(fp8e4m3)) / (16 * _NUM_THREADS);

// R22B: cache_hint defaults to cache_all (0); pass MXFP4_R22B_B_HINT for B-tile sites.
__device__ __forceinline__ void emit_tile_pf(
    auto &dst, const auto &src, const auto &idx,
    const uint32_t *so, i32x4 srd, const void *base, uint32_t lb,
    int cache_hint = static_cast<int>(coherency::cache_all))
{
    using ST = std::remove_reference_t<decltype(dst)>;
    using T = typename ST::dtype;
    coord<> uc = idx.template unit_coord<2, 3>();
    T* gptr = (T*)&src[uc];
    uint32_t soff = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<const char*>(gptr) - reinterpret_cast<const char*>(base)));
    asm volatile("" : "+s"(soff));
    const uint32_t lds_tile_base = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&dst.data[0])));
    const uint32_t warp_off = lb - lds_tile_base;
    constexpr int BPM = 16 * _NUM_THREADS;
    #pragma unroll
    for (int i = 0; i < PF_MPT; ++i) {
        const uint32_t lin = warp_off + i * BPM;
        const uint32_t sid = lin / ST::underlying_subtile_bytes;
        uint32_t lds_b = lds_tile_base + lin + sid * ST::subtile_padding;
        asm volatile("" : "+s"(lds_b));
        llvm_amdgcn_raw_buffer_load_lds(
            std::bit_cast<int32x4_t>(srd),
            (as3_uint32_ptr)(uintptr_t)lds_b,
            16, so[i], soff, 0,
            cache_hint);
    }
}

// ── LDS address computation for interleaved ds_read ──

template<ducks::rt::row_layout RT, ducks::st::all ST>
__device__ __forceinline__ void compute_lds_base_addrs(
    const ST &src, uint32_t &addr_p0, uint32_t &addr_p1)
{
    const int laneid = kittens::laneid();
    const int row_offset = laneid % RT::base_tile_rows;
    const int col_offset = RT::base_tile_stride * (laneid / RT::base_tile_rows);
    const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);
    using U = typename ST::dtype;
    constexpr int subcols = ST::underlying_subtile_cols;
    const uint32_t off0 = sizeof(U) * (src_ptr + row_offset * subcols + col_offset);
    addr_p0 = off0 ^ (((off0 % (16 * 128)) >> 8) << 4);
    const int col1 = col_offset + RT::base_tile_elements_per_stride_group;
    const uint32_t off1 = sizeof(U) * (src_ptr + row_offset * subcols + col1);
    addr_p1 = off1 ^ (((off1 % (16 * 128)) >> 8) << 4);
}

// ── Tile prefetch params (for individual emit_one_pf calls) ──

struct tile_pf_params {
    int32x4_t srd;
    uint32_t soff;
    uint32_t lds_addrs[PF_MPT];
    uint32_t voffs[PF_MPT];
    int      cache_hint;  // R22B: per-tile cache hint (cache_all default)
};

template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ __forceinline__ tile_pf_params make_pf_params(
    ST &dst, const GL &src, const COORD &idx,
    const uint32_t *so, i32x4 srd_in, const void *base_ptr, uint32_t lds_base,
    int cache_hint = static_cast<int>(coherency::cache_all))
{
    using T = typename ST::dtype;
    constexpr int BPM = 16 * _NUM_THREADS;
    coord<> uc = idx.template unit_coord<2, 3>();
    T* gptr = (T*)&src[uc];
    uint32_t soff = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<const char*>(gptr) - reinterpret_cast<const char*>(base_ptr)));
    const uint32_t lds_tile_base = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&dst.data[0])));
    const uint32_t warp_off = lds_base - lds_tile_base;
    tile_pf_params p;
    p.srd = std::bit_cast<int32x4_t>(srd_in);
    p.soff = soff;
    p.cache_hint = cache_hint;
    #pragma unroll
    for (int i = 0; i < PF_MPT; ++i) {
        p.voffs[i] = so[i];
        const uint32_t lin = warp_off + i * BPM;
        const uint32_t sid = lin / ST::underlying_subtile_bytes;
        p.lds_addrs[i] = lds_tile_base + lin + sid * ST::subtile_padding;
    }
    return p;
}

__device__ __forceinline__ void emit_one_pf(const tile_pf_params& p, int idx) {
#if R38A_INLINE_BUFLOAD_LDS
    // R38 Opt A: inline asm form. Sets m0 to the LDS address then issues
    // `buffer_load_dwordx4 voff, srd, soff offen lds`. The "memory" clobber
    // prevents the compiler from reordering this load across other LDS/buffer
    // operations. m0 is clobbered. cache_hint is dropped (default cache mode).
    uint32_t lds_addr = p.lds_addrs[idx];
    uint32_t voff = p.voffs[idx];
    asm volatile(
        "s_mov_b32 m0, %0\n"
        "buffer_load_dwordx4 %1, %2, %3 offen lds\n"
        :
        : "s"(lds_addr), "v"(voff), "s"(p.srd), "s"(p.soff)
        : "memory"
    );
#else
    llvm_amdgcn_raw_buffer_load_lds(
        std::bit_cast<int32x4_t>(p.srd),
        (as3_uint32_ptr)(uintptr_t)p.lds_addrs[idx],
        16, p.voffs[idx], p.soff, 0,
        p.cache_hint);
#endif
}

// ── R24C: Outer-K pull-forward L2 prefetch ──
// Issues a buffer_load_dwordx4 inline asm with no LDS write and discards
// the result, leaving data in L2 (and L1) for subsequent LDS-bound prefetches.
// The compiler cannot DCE inline asm volatile.
//
// `voff_extra` allows offsetting the per-tile voff by one outer-K stride so
// the load targets K+OUTER_K_PF_DEPTH instead of K+1. We compute the stride
// in bytes: K-step in GMEM equals `BK * sizeof(fp8e4m3) / 2 = 64` bytes per
// 16-thread group; a full tile prefetch step is `K_BYTES_PER_K0_STRIDE`.
// To stay simple we use the existing pf_a0_p (which already targets bt+2)
// and reissue with soff += K_OUTER_STRIDE_BYTES.
#ifndef OUTER_K_PF_DEPTH
#define OUTER_K_PF_DEPTH 1
#endif

#ifndef OUTER_K_PF_MODE
// 0 = no extra; 1 = L2-only via buffer_load discarded; 2 = (reserved future LDS triple-buffer)
#define OUTER_K_PF_MODE 0
#endif

// ── R24B: L2-only prefetch (always-available helper) ──
// Issue a buffer_load_dwordx4 into a scratch VGPR (discarded), targeting the
// SAME GMEM offsets as the supplied pf params. The data lands in L2 (and L1),
// prewarming the cache for a future LDS-bound prefetch.
__device__ __forceinline__ void emit_one_pf_l2only(const tile_pf_params& p, int idx)
{
    float4 sink;
    asm volatile(
        "buffer_load_dwordx4 %0, %1, %2, %3 offen\n"
        : "=v"(sink)
        : "v"(p.voffs[idx]), "s"(p.srd), "s"(p.soff)
        : "memory"
    );
}

template<int N>
__device__ __forceinline__ void emit_full_pf_l2only(const tile_pf_params& p) {
    #pragma unroll
    for (int i = 0; i < N; ++i) emit_one_pf_l2only(p, i);
}

// ── R24B: L2 prefetch macros (default 0 = inactive) ──
// L2_PF_A / L2_PF_B = 1 → 1 buffer_load_dwordx4 per thread per K-iter (cheap probe)
// L2_PF_A / L2_PF_B = 2 → PF_MPT/2 loads per K-iter (mid intensity)
// L2_PF_A / L2_PF_B = 3 → full PF_MPT loads per K-iter (full tile cover)
// Targets pf_bt+1 = bt+3 (one extra outer-K iter beyond the existing LDS prefetch).
#ifndef L2_PF_A
#define L2_PF_A 0
#endif
#ifndef L2_PF_B
#define L2_PF_B 0
#endif

template<int LEVEL>
__device__ __forceinline__ void emit_l2_pf_block(const tile_pf_params& p) {
    if constexpr (LEVEL == 1) {
        emit_one_pf_l2only(p, 0);
    } else if constexpr (LEVEL == 2) {
        constexpr int N = (PF_MPT > 1) ? (PF_MPT / 2) : 1;
        #pragma unroll
        for (int i = 0; i < N; ++i) emit_one_pf_l2only(p, i);
    } else if constexpr (LEVEL == 3) {
        #pragma unroll
        for (int i = 0; i < PF_MPT; ++i) emit_one_pf_l2only(p, i);
    }
}

#ifndef STEP3_PF_N
#define STEP3_PF_N 8
#endif
#ifndef STEP4_PF_N
#if STEP4_EXTERNAL_BR_PREFETCH
#define STEP4_PF_N 4
#else
#define STEP4_PF_N 8
#endif
#endif

static_assert(STEP3_PF_N >= 0 && STEP3_PF_N <= 2 * PF_MPT);
static_assert(STEP4_PF_N >= 0 && STEP4_PF_N <= 2 * PF_MPT);

template<int PF_N>
__device__ __forceinline__ void emit_pf_tail(const tile_pf_params& pf0, const tile_pf_params& pf1) {
    static_assert(PF_N >= 0 && PF_N <= 2 * PF_MPT);
    if constexpr (PF_N < PF_MPT) {
        #pragma unroll
        for (int pi = PF_N; pi < PF_MPT; ++pi) emit_one_pf(pf0, pi);
        #pragma unroll
        for (int pi = 0; pi < PF_MPT; ++pi) emit_one_pf(pf1, pi);
    } else if constexpr (PF_N < 2 * PF_MPT) {
        #pragma unroll
        for (int pi = PF_N - PF_MPT; pi < PF_MPT; ++pi) emit_one_pf(pf1, pi);
    }
}

// ── KPAIR shared operand setup macro ──
#define KPAIR_SETUP() \
    fp4_intx4_t a0l=fp4_lo4(A[0]), a1l=fp4_lo4(A[1]), a2l=fp4_lo4(A[2]), a3l=fp4_lo4(A[3]); \
    fp4_intx4_t a0h=fp4_hi4(A[0]), a1h=fp4_hi4(A[1]), a2h=fp4_hi4(A[2]), a3h=fp4_hi4(A[3]); \
    fp4_intx4_t b0l=fp4_lo4(B[0]), b1l=fp4_lo4(B[1]), b2l=fp4_lo4(B[2]), b3l=fp4_lo4(B[3]); \
    fp4_intx4_t b0h=fp4_hi4(B[0]), b1h=fp4_hi4(B[1]), b2h=fp4_hi4(B[2]), b3h=fp4_hi4(B[3]); \
    unsigned sa0 = std::bit_cast<unsigned>(a_raw[0]); \
    unsigned sa1 = std::bit_cast<unsigned>(a_raw[1]); \
    unsigned sb0 = std::bit_cast<unsigned>(b_raw[0]); \
    unsigned sb1 = std::bit_cast<unsigned>(b_raw[1])

// ── KPAIR constraint lists (all 16 acc + 20 A/B/scale inputs) ──
#define KPAIR_ACC_CLOBBER \
    "+a"(acc[0]), "+a"(acc[1]), "+a"(acc[2]), "+a"(acc[3]),   \
    "+a"(acc[4]), "+a"(acc[5]), "+a"(acc[6]), "+a"(acc[7]),   \
    "+a"(acc[8]), "+a"(acc[9]), "+a"(acc[10]), "+a"(acc[11]), \
    "+a"(acc[12]), "+a"(acc[13]), "+a"(acc[14]), "+a"(acc[15])

#define KPAIR_INPUTS \
    "v"(a0l), "v"(a1l), "v"(a2l), "v"(a3l), \
    "v"(a0h), "v"(a1h), "v"(a2h), "v"(a3h), \
    "v"(b0l), "v"(b1l), "v"(b2l), "v"(b3l), \
    "v"(b0h), "v"(b1h), "v"(b2h), "v"(b3h), \
    "v"(sa0), "v"(sa1), "v"(sb0), "v"(sb1)

// ── 32 KPAIR MFMAs + 8 interleaved ds_reads (single asm block) ──
// Outputs: %0..15=acc, %16..23=d0..d7.  Inputs: %24..27=a_lo, %28..31=a_hi,
//   %32..35=b_lo, %36..39=b_hi, %40=sa0, %41=sa1, %42=sb0, %43=sb1,
//   %44=lds_a0, %45=lds_a1.

__device__ __forceinline__ void kpair_32mfma_with_lds(
    fp4_floatx4_t acc[16],
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_raw[2], const fp8e8m0_4 b_raw[2],
    float4 &d0, float4 &d1, float4 &d2, float4 &d3,
    float4 &d4, float4 &d5, float4 &d6, float4 &d7,
    uint32_t lds_a0, uint32_t lds_a1)
{
    KPAIR_SETUP();
    asm volatile(
        // Row 0 Phase 0 — ALL 8 ds_reads front-loaded (1:1 with first 8 MFMAs)
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %24, %32, %0,  %40, %42 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %16, %44 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %24, %33, %1,  %40, %42 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %17, %44 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %24, %34, %2,  %40, %43 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %18, %44 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %24, %35, %3,  %40, %43 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %19, %44 offset:6144\n"
        // Row 0 Phase 1 — remaining 4 ds_reads
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %28, %36, %0,  %40, %42 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %20, %45 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %28, %37, %1,  %40, %42 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %21, %45 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %28, %38, %2,  %40, %43 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %22, %45 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %28, %39, %3,  %40, %43 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %23, %45 offset:6144\n"
        // Rows 1-3: pure MFMAs (24 total, reads already in-flight with 24+ cycles to complete)
        // Row 1 Phase 0
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %25, %32, %4,  %40, %42 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %25, %33, %5,  %40, %42 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %25, %34, %6,  %40, %43 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %25, %35, %7,  %40, %43 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Row 1 Phase 1
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %29, %36, %4,  %40, %42 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %29, %37, %5,  %40, %42 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %29, %38, %6,  %40, %43 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %29, %39, %7,  %40, %43 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Row 2 Phase 0
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %26, %32, %8,  %41, %42 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %26, %33, %9,  %41, %42 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %26, %34, %10, %41, %43 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %26, %35, %11, %41, %43 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Row 2 Phase 1
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %30, %36, %8,  %41, %42 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %30, %37, %9,  %41, %42 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %30, %38, %10, %41, %43 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %30, %39, %11, %41, %43 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Row 3 Phase 0
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %27, %32, %12, %41, %42 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %27, %33, %13, %41, %42 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %27, %34, %14, %41, %43 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %27, %35, %15, %41, %43 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Row 3 Phase 1
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %31, %36, %12, %41, %42 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %31, %37, %13, %41, %42 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %31, %38, %14, %41, %43 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %31, %39, %15, %41, %43 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER,
          "=&v"(d0), "=&v"(d1), "=&v"(d2), "=&v"(d3),
          "=&v"(d4), "=&v"(d5), "=&v"(d6), "=&v"(d7)
        : KPAIR_INPUTS,
          "v"(lds_a0), "v"(lds_a1)
    );
}

// ── 32 KPAIR MFMAs + 8 interleaved tile prefetch loads ──
// Uses same operand layout as original kpair_32mfma (%0..15=acc, %16..19=a_lo,
// %20..23=a_hi, %24..27=b_lo, %28..31=b_hi, %32..33=sa, %34..35=sb)
// but split into 4 row blocks with 2 pf calls between each.

template<int PF_N = 8>
__device__ __forceinline__ void kpair_32mfma_with_pf(
    fp4_floatx4_t acc[16],
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_raw[2], const fp8e8m0_4 b_raw[2],
    const tile_pf_params &pf0, const tile_pf_params &pf1)
{
    KPAIR_SETUP();
    // Row 0
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %16, %24, %0,  %32, %34 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %16, %25, %1,  %32, %34 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %16, %26, %2,  %32, %35 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %16, %27, %3,  %32, %35 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %20, %28, %0,  %32, %34 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %20, %29, %1,  %32, %34 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %20, %30, %2,  %32, %35 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %20, %31, %3,  %32, %35 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 0) emit_one_pf(pf0, 0);
    if constexpr (PF_N > 1) emit_one_pf(pf0, 1);
    // Row 1
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %17, %24, %4,  %32, %34 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %17, %25, %5,  %32, %34 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %17, %26, %6,  %32, %35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %17, %27, %7,  %32, %35 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %21, %28, %4,  %32, %34 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %21, %29, %5,  %32, %34 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %21, %30, %6,  %32, %35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %21, %31, %7,  %32, %35 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 2) emit_one_pf(pf0, 2);
    if constexpr (PF_N > 3) emit_one_pf(pf0, 3);
    // Row 2
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %18, %24, %8,  %33, %34 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %18, %25, %9,  %33, %34 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %18, %26, %10, %33, %35 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %18, %27, %11, %33, %35 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %22, %28, %8,  %33, %34 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %22, %29, %9,  %33, %34 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %22, %30, %10, %33, %35 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %22, %31, %11, %33, %35 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 4) emit_one_pf(pf1, 0);
    if constexpr (PF_N > 5) emit_one_pf(pf1, 1);
    // Row 3
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %19, %24, %12, %33, %34 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %19, %25, %13, %33, %34 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %19, %26, %14, %33, %35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %19, %27, %15, %33, %35 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %23, %28, %12, %33, %34 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %23, %29, %13, %33, %34 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %23, %30, %14, %33, %35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %23, %31, %15, %33, %35 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 6) emit_one_pf(pf1, 2);
    if constexpr (PF_N > 7) emit_one_pf(pf1, 3);
}

// ── 32 KPAIR MFMAs, no interleaved ops (single asm block) ──
__device__ __forceinline__ void kpair_32mfma_pure(
    fp4_floatx4_t acc[16],
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_raw[2], const fp8e8m0_4 b_raw[2])
{
    KPAIR_SETUP();
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %16, %24, %0,  %32, %34 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %16, %25, %1,  %32, %34 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %16, %26, %2,  %32, %35 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %16, %27, %3,  %32, %35 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %20, %28, %0,  %32, %34 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %20, %29, %1,  %32, %34 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %20, %30, %2,  %32, %35 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %20, %31, %3,  %32, %35 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %17, %24, %4,  %32, %34 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %17, %25, %5,  %32, %34 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %17, %26, %6,  %32, %35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %17, %27, %7,  %32, %35 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %21, %28, %4,  %32, %34 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %21, %29, %5,  %32, %34 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %21, %30, %6,  %32, %35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %21, %31, %7,  %32, %35 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %18, %24, %8,  %33, %34 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %18, %25, %9,  %33, %34 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %18, %26, %10, %33, %35 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %18, %27, %11, %33, %35 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %22, %28, %8,  %33, %34 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %22, %29, %9,  %33, %34 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %22, %30, %10, %33, %35 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %22, %31, %11, %33, %35 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %19, %24, %12, %33, %34 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %19, %25, %13, %33, %34 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %19, %26, %14, %33, %35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %19, %27, %15, %33, %35 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %23, %28, %12, %33, %34 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %23, %29, %13, %33, %34 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %23, %30, %14, %33, %35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %23, %31, %15, %33, %35 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
}

#if DIRECT_BL
// ── 32 KPAIR MFMAs + 8 interleaved buffer_load_dwordx4 for Bl (single asm block) ──
// Loads next Bl tile from preshuffled global memory while executing 32 MFMAs.
// The 8 buffer loads are interleaved 1:1 with the first 8 MFMAs (Row 0),
// so the 32 VGPRs (8×float4) are created within the asm scope.
// SGPR-optimized: computes N0 and K0 soffsets inside asm via s_add_u32 —
// only 2 SGPR inputs (soff_base, n0_stride). K0=1 offset (0x400) is literal.
//
// Operand map:
//   Outputs: %0..%15 = acc (AGPR), %16..%23 = d0..d7 (VGPR float4),
//            %24..%26 = s1/s2/s3 temps (SGPR)
//   Inputs:  %27..%46 = KPAIR_INPUTS (20 VGPRs: a_lo/hi, b_lo/hi, sa, sb),
//            %47 = bl_voff (VGPR), %48 = bl_srd (SGPR i32x4),
//            %49 = soff_base (SGPR), %50 = n0_stride (SGPR)
__device__ __forceinline__ void kpair_32mfma_with_vmem_bl(
    fp4_floatx4_t acc[16],
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_raw[2], const fp8e8m0_4 b_raw[2],
    float4 &d0, float4 &d1, float4 &d2, float4 &d3,
    float4 &d4, float4 &d5, float4 &d6, float4 &d7,
    uint32_t bl_voff, i32x4 bl_srd,
    uint32_t soff_base, uint32_t n0_stride)
{
    KPAIR_SETUP();
    uint32_t s1_tmp, s2_tmp, s3_tmp;
    asm volatile(
        // Compute N0 soffsets for K0=0: s1=base+stride, s2=base+2*stride, s3=base+3*stride
        "s_add_u32 %24, %49, %50\n"         // s1 = soff_base + n0_stride
        "s_add_u32 %25, %24, %50\n"         // s2 = s1 + n0_stride
        "s_add_u32 %26, %25, %50\n"         // s3 = s2 + n0_stride
        // Row 0 Phase 0 — 4 K0=0 buffer_load_dwordx4 interleaved with first 4 MFMAs
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %27, %35, %0,  %43, %45 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %16, %47, %48, %49 offen\n"             // d0: N0=0,K0=0
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %27, %36, %1,  %43, %45 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %17, %47, %48, %24 offen\n"             // d1: N0=1,K0=0
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %27, %37, %2,  %43, %46 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %18, %47, %48, %25 offen\n"             // d2: N0=2,K0=0
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %27, %38, %3,  %43, %46 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %19, %47, %48, %26 offen\n"             // d3: N0=3,K0=0
        // Row 0 Phase 1 — 4 K0=1 buffer_load_dwordx4 with s_add_u32 for soffsets
        // Reuse s1 (=%24) for K0=1 soffsets: s1 = soff_base + BL_K0_STRIDE
        "s_add_u32 %24, %49, 0x400\n"                                // s1 = soff_base + 1024 (N0=0,K0=1)
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %31, %39, %0,  %43, %45 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %20, %47, %48, %24 offen\n"             // d4: N0=0,K0=1
        "s_add_u32 %24, %24, %50\n"                                  // s1 += n0_stride (N0=1,K0=1)
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %31, %40, %1,  %43, %45 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %21, %47, %48, %24 offen\n"             // d5: N0=1,K0=1
        "s_add_u32 %24, %24, %50\n"                                  // s1 += n0_stride (N0=2,K0=1)
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %31, %41, %2,  %43, %46 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %22, %47, %48, %24 offen\n"             // d6: N0=2,K0=1
        "s_add_u32 %24, %24, %50\n"                                  // s1 += n0_stride (N0=3,K0=1)
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %31, %42, %3,  %43, %46 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %23, %47, %48, %24 offen\n"             // d7: N0=3,K0=1
        // Rows 1-3: pure MFMAs (24 total)
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %28, %35, %4,  %43, %45 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %28, %36, %5,  %43, %45 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %28, %37, %6,  %43, %46 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %28, %38, %7,  %43, %46 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %32, %39, %4,  %43, %45 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %32, %40, %5,  %43, %45 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %32, %41, %6,  %43, %46 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %32, %42, %7,  %43, %46 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %29, %35, %8,  %44, %45 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %29, %36, %9,  %44, %45 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %29, %37, %10, %44, %46 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %29, %38, %11, %44, %46 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %33, %39, %8,  %44, %45 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %33, %40, %9,  %44, %45 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %33, %41, %10, %44, %46 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %33, %42, %11, %44, %46 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %30, %35, %12, %44, %45 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %30, %36, %13, %44, %45 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %30, %37, %14, %44, %46 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %30, %38, %15, %44, %46 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %34, %39, %12, %44, %45 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %34, %40, %13, %44, %45 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %34, %41, %14, %44, %46 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %34, %42, %15, %44, %46 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER,
          "=&v"(d0), "=&v"(d1), "=&v"(d2), "=&v"(d3),
          "=&v"(d4), "=&v"(d5), "=&v"(d6), "=&v"(d7),
          "=&s"(s1_tmp), "=&s"(s2_tmp), "=&s"(s3_tmp)
        : KPAIR_INPUTS,
          "v"(bl_voff), "s"(bl_srd),
          "s"(soff_base), "s"(n0_stride)
        : "scc"
    );
}

__device__ __forceinline__ void kpair_32mfma_with_vmem_bl_wrap(
    fp4_floatx4_t acc[16],
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_raw[2], const fp8e8m0_4 b_raw[2],
    float4 &d0, float4 &d1, float4 &d2, float4 &d3,
    float4 &d4, float4 &d5, float4 &d6, float4 &d7,
    uint32_t bl_voff, i32x4 bl_srd, uint32_t soff_base)
{
    kpair_32mfma_with_vmem_bl(acc, A, B, a_raw, b_raw,
        d0, d1, d2, d3, d4, d5, d6, d7,
        bl_voff, bl_srd, soff_base, BL_N0_STRIDE);
}
#endif // DIRECT_BL

// ── Merged Steps 1+2: 64 MFMAs + 16 ds_reads (Br + A1) in one asm block ──
// Eliminates compiler transition between Steps 1 and 2.
// Outputs: %0..15=acc_bl, %16..31=acc_br, %32..39=br_d, %40..47=a1_d
// Inputs: %48..55=A0_lo/hi, %56..63=Bl_lo/hi, %64..65=sa0/sa1,
//   %66..67=sb_bl0/sb_bl1, %68..69=sb_br0/sb_br1,
//   %70..71=br_lds_p0/p1, %72..73=a1_lds_p0/p1

__device__ __forceinline__ void kpair_64mfma_step12(
    fp4_floatx4_t acc_bl[16], fp4_floatx4_t acc_br[16],
    const fp4_intx8_t A0[4], const fp4_intx8_t Bl[4],
    const fp8e8m0_4 a_raw[2], const fp8e8m0_4 bl_raw[2], const fp8e8m0_4 br_raw[2],
    float4 br_d[8], float4 a1_d[8],
    uint32_t br_p0, uint32_t br_p1,
    uint32_t a1_p0, uint32_t a1_p1)
{
    fp4_intx4_t a0l=fp4_lo4(A0[0]), a1l=fp4_lo4(A0[1]), a2l=fp4_lo4(A0[2]), a3l=fp4_lo4(A0[3]);
    fp4_intx4_t a0h=fp4_hi4(A0[0]), a1h=fp4_hi4(A0[1]), a2h=fp4_hi4(A0[2]), a3h=fp4_hi4(A0[3]);
    fp4_intx4_t b0l=fp4_lo4(Bl[0]), b1l=fp4_lo4(Bl[1]), b2l=fp4_lo4(Bl[2]), b3l=fp4_lo4(Bl[3]);
    fp4_intx4_t b0h=fp4_hi4(Bl[0]), b1h=fp4_hi4(Bl[1]), b2h=fp4_hi4(Bl[2]), b3h=fp4_hi4(Bl[3]);
    unsigned sa0 = std::bit_cast<unsigned>(a_raw[0]);
    unsigned sa1 = std::bit_cast<unsigned>(a_raw[1]);
    unsigned sb_bl0 = std::bit_cast<unsigned>(bl_raw[0]);
    unsigned sb_bl1 = std::bit_cast<unsigned>(bl_raw[1]);
    unsigned sb_br0 = std::bit_cast<unsigned>(br_raw[0]);
    unsigned sb_br1 = std::bit_cast<unsigned>(br_raw[1]);

    asm volatile(
        // ═══ STEP 1: A0×Bl (32 MFMAs) + 8 ds_reads for Br ═══
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %48, %56, %0,  %64, %66 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %32, %70 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %48, %57, %1,  %64, %66 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %33, %70 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %48, %58, %2,  %64, %67 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %34, %70 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %48, %59, %3,  %64, %67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %35, %70 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %52, %60, %0,  %64, %66 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %36, %71 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %52, %61, %1,  %64, %66 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %37, %71 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %52, %62, %2,  %64, %67 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %38, %71 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %52, %63, %3,  %64, %67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %39, %71 offset:6144\n"
        // Rows 1-3: 24 pure Step 1 MFMAs
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %49, %56, %4,  %64, %66 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %49, %57, %5,  %64, %66 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %49, %58, %6,  %64, %67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %49, %59, %7,  %64, %67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %53, %60, %4,  %64, %66 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %53, %61, %5,  %64, %66 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %53, %62, %6,  %64, %67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %53, %63, %7,  %64, %67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %50, %56, %8,  %65, %66 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %50, %57, %9,  %65, %66 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %50, %58, %10, %65, %67 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %50, %59, %11, %65, %67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %54, %60, %8,  %65, %66 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %54, %61, %9,  %65, %66 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %54, %62, %10, %65, %67 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %54, %63, %11, %65, %67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %51, %56, %12, %65, %66 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %51, %57, %13, %65, %66 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %51, %58, %14, %65, %67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %51, %59, %15, %65, %67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %55, %60, %12, %65, %66 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %55, %61, %13, %65, %66 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %55, %62, %14, %65, %67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %55, %63, %15, %65, %67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Wait for Br ds_reads (tunable: lgkmcnt(STEP12_BR_LGKMCNT))
        "s_waitcnt lgkmcnt(" MXFP4_STR(STEP12_BR_LGKMCNT) ")\n"
        // ═══ STEP 2: A0×Br (32 MFMAs) + 8 ds_reads for A1 ═══
        "v_mfma_scale_f32_16x16x128_f8f6f4 %16, %48, %32, %16, %64, %68 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %40, %72 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %17, %48, %33, %17, %64, %68 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %41, %72 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %18, %48, %34, %18, %64, %69 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %42, %72 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %19, %48, %35, %19, %64, %69 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %43, %72 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %16, %52, %36, %16, %64, %68 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %44, %73 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %17, %52, %37, %17, %64, %68 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %45, %73 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %18, %52, %38, %18, %64, %69 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %46, %73 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %19, %52, %39, %19, %64, %69 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %47, %73 offset:6144\n"
        // Rows 1-3: 24 pure Step 2 MFMAs
        "v_mfma_scale_f32_16x16x128_f8f6f4 %20, %49, %32, %20, %64, %68 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %21, %49, %33, %21, %64, %68 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %22, %49, %34, %22, %64, %69 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %23, %49, %35, %23, %64, %69 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %20, %53, %36, %20, %64, %68 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %21, %53, %37, %21, %64, %68 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %22, %53, %38, %22, %64, %69 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %23, %53, %39, %23, %64, %69 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %24, %50, %32, %24, %65, %68 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %25, %50, %33, %25, %65, %68 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %26, %50, %34, %26, %65, %69 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %27, %50, %35, %27, %65, %69 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %24, %54, %36, %24, %65, %68 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %25, %54, %37, %25, %65, %68 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %26, %54, %38, %26, %65, %69 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %27, %54, %39, %27, %65, %69 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %28, %51, %32, %28, %65, %68 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %29, %51, %33, %29, %65, %68 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %30, %51, %34, %30, %65, %69 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %31, %51, %35, %31, %65, %69 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %28, %55, %36, %28, %65, %68 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %29, %55, %37, %29, %65, %68 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %30, %55, %38, %30, %65, %69 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %31, %55, %39, %31, %65, %69 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc_bl[0]),  "+a"(acc_bl[1]),  "+a"(acc_bl[2]),  "+a"(acc_bl[3]),
          "+a"(acc_bl[4]),  "+a"(acc_bl[5]),  "+a"(acc_bl[6]),  "+a"(acc_bl[7]),
          "+a"(acc_bl[8]),  "+a"(acc_bl[9]),  "+a"(acc_bl[10]), "+a"(acc_bl[11]),
          "+a"(acc_bl[12]), "+a"(acc_bl[13]), "+a"(acc_bl[14]), "+a"(acc_bl[15]),
          "+a"(acc_br[0]),  "+a"(acc_br[1]),  "+a"(acc_br[2]),  "+a"(acc_br[3]),
          "+a"(acc_br[4]),  "+a"(acc_br[5]),  "+a"(acc_br[6]),  "+a"(acc_br[7]),
          "+a"(acc_br[8]),  "+a"(acc_br[9]),  "+a"(acc_br[10]), "+a"(acc_br[11]),
          "+a"(acc_br[12]), "+a"(acc_br[13]), "+a"(acc_br[14]), "+a"(acc_br[15]),
          "=&v"(br_d[0]), "=&v"(br_d[1]), "=&v"(br_d[2]), "=&v"(br_d[3]),
          "=&v"(br_d[4]), "=&v"(br_d[5]), "=&v"(br_d[6]), "=&v"(br_d[7]),
          "=&v"(a1_d[0]), "=&v"(a1_d[1]), "=&v"(a1_d[2]), "=&v"(a1_d[3]),
          "=&v"(a1_d[4]), "=&v"(a1_d[5]), "=&v"(a1_d[6]), "=&v"(a1_d[7])
        : "v"(a0l), "v"(a1l), "v"(a2l), "v"(a3l),
          "v"(a0h), "v"(a1h), "v"(a2h), "v"(a3h),
          "v"(b0l), "v"(b1l), "v"(b2l), "v"(b3l),
          "v"(b0h), "v"(b1h), "v"(b2h), "v"(b3h),
          "v"(sa0), "v"(sa1), "v"(sb_bl0), "v"(sb_bl1),
          "v"(sb_br0), "v"(sb_br1),
          "v"(br_p0), "v"(br_p1), "v"(a1_p0), "v"(a1_p1)
    );
}

// ── Merged Steps 3+4: 64 MFMAs + 16 ds_reads (nxt_a0 + nxt_bl) in one asm block ──
// Eliminates compiler transition between Steps 3 and 4.
// Barrier emitted separately before the asm block (caller or inline).
// Outputs: %0..15=acc_bl, %16..31=acc_br, %32..39=nxt_a0_d, %40..47=nxt_bl_d
// Inputs: %48..51=A1_lo, %52..55=A1_hi, %56..59=Bl_lo, %60..63=Bl_hi,
//   %64..67=Br_lo, %68..71=Br_hi,
//   %72..73=sa0/sa1, %74..75=sbl0/sbl1, %76..77=sbr0/sbr1,
//   %78..79=a0_lds_p0/p1, %80..81=bl_lds_p0/p1

__device__ __forceinline__ void kpair_64mfma_step34(
    fp4_floatx4_t acc_bl[16], fp4_floatx4_t acc_br[16],
    const fp4_intx8_t A1[4],
    const fp4_intx8_t Bl[4], const fp4_intx8_t Br[4],
    const fp8e8m0_4 a1_raw[2], const fp8e8m0_4 bl_raw[2], const fp8e8m0_4 br_raw[2],
    float4 nxt_a0_d[8], float4 nxt_bl_d[8],
    uint32_t a0_p0, uint32_t a0_p1,
    uint32_t bl_p0, uint32_t bl_p1)
{
    // A1 tile splits (shared between Step3 and Step4)
    fp4_intx4_t a1_0l=fp4_lo4(A1[0]), a1_1l=fp4_lo4(A1[1]), a1_2l=fp4_lo4(A1[2]), a1_3l=fp4_lo4(A1[3]);
    fp4_intx4_t a1_0h=fp4_hi4(A1[0]), a1_1h=fp4_hi4(A1[1]), a1_2h=fp4_hi4(A1[2]), a1_3h=fp4_hi4(A1[3]);
    // Bl tile splits (Step3 only)
    fp4_intx4_t bl_0l=fp4_lo4(Bl[0]), bl_1l=fp4_lo4(Bl[1]), bl_2l=fp4_lo4(Bl[2]), bl_3l=fp4_lo4(Bl[3]);
    fp4_intx4_t bl_0h=fp4_hi4(Bl[0]), bl_1h=fp4_hi4(Bl[1]), bl_2h=fp4_hi4(Bl[2]), bl_3h=fp4_hi4(Bl[3]);
    // Br tile splits (Step4 only)
    fp4_intx4_t br_0l=fp4_lo4(Br[0]), br_1l=fp4_lo4(Br[1]), br_2l=fp4_lo4(Br[2]), br_3l=fp4_lo4(Br[3]);
    fp4_intx4_t br_0h=fp4_hi4(Br[0]), br_1h=fp4_hi4(Br[1]), br_2h=fp4_hi4(Br[2]), br_3h=fp4_hi4(Br[3]);
    // Scales
    unsigned sa0 = std::bit_cast<unsigned>(a1_raw[0]);
    unsigned sa1 = std::bit_cast<unsigned>(a1_raw[1]);
    unsigned sbl0 = std::bit_cast<unsigned>(bl_raw[0]);
    unsigned sbl1 = std::bit_cast<unsigned>(bl_raw[1]);
    unsigned sbr0 = std::bit_cast<unsigned>(br_raw[0]);
    unsigned sbr1 = std::bit_cast<unsigned>(br_raw[1]);

    // Barrier emitted separately (with memory clobber) so the MFMAs block stays lightweight
    // R19B: site _S1 (kpair_64mfma_step34, dead with default FUSED_STEP34=0)
    asm volatile(MXFP4_STEP3_BARRIER_INST_S1 ::: "memory");

    asm volatile(
        // ═══ STEP 3: A1×Bl (32 MFMAs) + 8 ds_reads for nxt_a0 ═══
        // Row 0 Phase 0: 4 MFMAs + 4 ds_reads
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %48, %56, %0,  %72, %74 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %32, %78 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %48, %57, %1,  %72, %74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %33, %78 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %48, %58, %2,  %72, %75 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %34, %78 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %48, %59, %3,  %72, %75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %35, %78 offset:6144\n"
        // Row 0 Phase 1: 4 MFMAs + 4 ds_reads
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %52, %60, %0,  %72, %74 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %36, %79 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %52, %61, %1,  %72, %74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %37, %79 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %52, %62, %2,  %72, %75 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %38, %79 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %52, %63, %3,  %72, %75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %39, %79 offset:6144\n"
        // Row 1: 8 pure Step3 MFMAs (odd, sa0)
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %49, %56, %4,  %72, %74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %49, %57, %5,  %72, %74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %49, %58, %6,  %72, %75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %49, %59, %7,  %72, %75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %53, %60, %4,  %72, %74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %53, %61, %5,  %72, %74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %53, %62, %6,  %72, %75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %53, %63, %7,  %72, %75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Row 2: 8 pure Step3 MFMAs (even, sa1)
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %50, %56, %8,  %73, %74 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %50, %57, %9,  %73, %74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %50, %58, %10, %73, %75 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %50, %59, %11, %73, %75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %54, %60, %8,  %73, %74 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %54, %61, %9,  %73, %74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %54, %62, %10, %73, %75 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %54, %63, %11, %73, %75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Row 3: 8 pure Step3 MFMAs (odd, sa1)
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %51, %56, %12, %73, %74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %51, %57, %13, %73, %74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %51, %58, %14, %73, %75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %51, %59, %15, %73, %75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %55, %60, %12, %73, %74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %55, %61, %13, %73, %74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %55, %62, %14, %73, %75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %55, %63, %15, %73, %75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // ═══ STEP 4: A1×Br (32 MFMAs) + 8 ds_reads for nxt_bl ═══
        // Row 0 Phase 0: 4 MFMAs + 4 ds_reads
        "v_mfma_scale_f32_16x16x128_f8f6f4 %16, %48, %64, %16, %72, %76 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %40, %80 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %17, %48, %65, %17, %72, %76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %41, %80 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %18, %48, %66, %18, %72, %77 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %42, %80 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %19, %48, %67, %19, %72, %77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %43, %80 offset:6144\n"
        // Row 0 Phase 1: 4 MFMAs + 4 ds_reads
        "v_mfma_scale_f32_16x16x128_f8f6f4 %16, %52, %68, %16, %72, %76 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %44, %81 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %17, %52, %69, %17, %72, %76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %45, %81 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %18, %52, %70, %18, %72, %77 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %46, %81 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %19, %52, %71, %19, %72, %77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %47, %81 offset:6144\n"
        // Row 1: 8 pure Step4 MFMAs (odd, sa0)
        "v_mfma_scale_f32_16x16x128_f8f6f4 %20, %49, %64, %20, %72, %76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %21, %49, %65, %21, %72, %76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %22, %49, %66, %22, %72, %77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %23, %49, %67, %23, %72, %77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %20, %53, %68, %20, %72, %76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %21, %53, %69, %21, %72, %76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %22, %53, %70, %22, %72, %77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %23, %53, %71, %23, %72, %77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Row 2: 8 pure Step4 MFMAs (even, sa1)
        "v_mfma_scale_f32_16x16x128_f8f6f4 %24, %50, %64, %24, %73, %76 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %25, %50, %65, %25, %73, %76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %26, %50, %66, %26, %73, %77 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %27, %50, %67, %27, %73, %77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %24, %54, %68, %24, %73, %76 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %25, %54, %69, %25, %73, %76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %26, %54, %70, %26, %73, %77 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %27, %54, %71, %27, %73, %77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Row 3: 8 pure Step4 MFMAs (odd, sa1)
        "v_mfma_scale_f32_16x16x128_f8f6f4 %28, %51, %64, %28, %73, %76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %29, %51, %65, %29, %73, %76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %30, %51, %66, %30, %73, %77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %31, %51, %67, %31, %73, %77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %28, %55, %68, %28, %73, %76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %29, %55, %69, %29, %73, %76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %30, %55, %70, %30, %73, %77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %31, %55, %71, %31, %73, %77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc_bl[0]),  "+a"(acc_bl[1]),  "+a"(acc_bl[2]),  "+a"(acc_bl[3]),
          "+a"(acc_bl[4]),  "+a"(acc_bl[5]),  "+a"(acc_bl[6]),  "+a"(acc_bl[7]),
          "+a"(acc_bl[8]),  "+a"(acc_bl[9]),  "+a"(acc_bl[10]), "+a"(acc_bl[11]),
          "+a"(acc_bl[12]), "+a"(acc_bl[13]), "+a"(acc_bl[14]), "+a"(acc_bl[15]),
          "+a"(acc_br[0]),  "+a"(acc_br[1]),  "+a"(acc_br[2]),  "+a"(acc_br[3]),
          "+a"(acc_br[4]),  "+a"(acc_br[5]),  "+a"(acc_br[6]),  "+a"(acc_br[7]),
          "+a"(acc_br[8]),  "+a"(acc_br[9]),  "+a"(acc_br[10]), "+a"(acc_br[11]),
          "+a"(acc_br[12]), "+a"(acc_br[13]), "+a"(acc_br[14]), "+a"(acc_br[15]),
          "=&v"(nxt_a0_d[0]), "=&v"(nxt_a0_d[1]), "=&v"(nxt_a0_d[2]), "=&v"(nxt_a0_d[3]),
          "=&v"(nxt_a0_d[4]), "=&v"(nxt_a0_d[5]), "=&v"(nxt_a0_d[6]), "=&v"(nxt_a0_d[7]),
          "=&v"(nxt_bl_d[0]), "=&v"(nxt_bl_d[1]), "=&v"(nxt_bl_d[2]), "=&v"(nxt_bl_d[3]),
          "=&v"(nxt_bl_d[4]), "=&v"(nxt_bl_d[5]), "=&v"(nxt_bl_d[6]), "=&v"(nxt_bl_d[7])
        : "v"(a1_0l), "v"(a1_1l), "v"(a1_2l), "v"(a1_3l),
          "v"(a1_0h), "v"(a1_1h), "v"(a1_2h), "v"(a1_3h),
          "v"(bl_0l), "v"(bl_1l), "v"(bl_2l), "v"(bl_3l),
          "v"(bl_0h), "v"(bl_1h), "v"(bl_2h), "v"(bl_3h),
          "v"(br_0l), "v"(br_1l), "v"(br_2l), "v"(br_3l),
          "v"(br_0h), "v"(br_1h), "v"(br_2h), "v"(br_3h),
          "v"(sa0), "v"(sa1), "v"(sbl0), "v"(sbl1), "v"(sbr0), "v"(sbr1),
          "v"(a0_p0), "v"(a0_p1), "v"(bl_p0), "v"(bl_p1)
    );
}

// ── 32 KPAIR MFMAs + 16 ds_reads (2 tiles) + 8 pf (row blocks) ──
// Each row: 8 MFMAs + 4 ds_reads (2 tiles × 2 reads) + 2 pf.
// Operands: %0..15=acc(+a), %16..19=ds_out(=v: da_lo,db_lo,da_hi,db_hi),
//   %20..39=KPAIR_INPUTS, %40=lds_a, %41=lds_b.

#define KPAIR_ROW_ACC_DS4 \
    KPAIR_ACC_CLOBBER, "=v"(da_lo), "=v"(db_lo), "=v"(da_hi), "=v"(db_hi)
#define KPAIR_INPUTS_2LDS \
    KPAIR_INPUTS, "v"(lds_a), "v"(lds_b)

template<int PF_N = 8>
__device__ __forceinline__ void kpair_32mfma_with_16lds_and_pf(
    fp4_floatx4_t acc[16],
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_raw[2], const fp8e8m0_4 b_raw[2],
    float4 da[8], float4 db[8],
    uint32_t lds_a_addrs[2], uint32_t lds_b_addrs[2],
    const tile_pf_params &pf0, const tile_pf_params &pf1)
{
    KPAIR_SETUP();
    { // Row 0 (even, sa0)
        float4 &da_lo=da[0], &db_lo=db[0], &da_hi=da[4], &db_hi=db[4];
        uint32_t lds_a=lds_a_addrs[0], lds_b=lds_b_addrs[0];
        asm volatile(
            "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %20, %28, %0,  %36, %38 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %20, %29, %1,  %36, %38 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "ds_read_b128 %16, %40 offset:0\n"
            "ds_read_b128 %17, %41 offset:0\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %20, %30, %2,  %36, %39 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %20, %31, %3,  %36, %39 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %24, %32, %0,  %36, %38 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %24, %33, %1,  %36, %38 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "ds_read_b128 %18, %40 offset:2048\n"
            "ds_read_b128 %19, %41 offset:2048\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %24, %34, %2,  %36, %39 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %24, %35, %3,  %36, %39 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            : KPAIR_ROW_ACC_DS4 : KPAIR_INPUTS_2LDS);
    }
    if constexpr (PF_N > 0) emit_one_pf(pf0, 0);
    if constexpr (PF_N > 1) emit_one_pf(pf0, 1);
    { // Row 1 (odd, sa0)
        float4 &da_lo=da[1], &db_lo=db[1], &da_hi=da[5], &db_hi=db[5];
        uint32_t lds_a=lds_a_addrs[0], lds_b=lds_b_addrs[0];
        asm volatile(
            "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %21, %28, %4,  %36, %38 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %21, %29, %5,  %36, %38 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "ds_read_b128 %16, %40 offset:4096\n"
            "ds_read_b128 %17, %41 offset:4096\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %21, %30, %6,  %36, %39 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %21, %31, %7,  %36, %39 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %25, %32, %4,  %36, %38 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %25, %33, %5,  %36, %38 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "ds_read_b128 %18, %40 offset:6144\n"
            "ds_read_b128 %19, %41 offset:6144\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %25, %34, %6,  %36, %39 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %25, %35, %7,  %36, %39 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            : KPAIR_ROW_ACC_DS4 : KPAIR_INPUTS_2LDS);
    }
    if constexpr (PF_N > 2) emit_one_pf(pf0, 2);
    if constexpr (PF_N > 3) emit_one_pf(pf0, 3);
    { // Row 2 (even, sa1)
        float4 &da_lo=da[2], &db_lo=db[2], &da_hi=da[6], &db_hi=db[6];
        uint32_t lds_a=lds_a_addrs[1], lds_b=lds_b_addrs[1];
        asm volatile(
            "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %22, %28, %8,  %37, %38 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %22, %29, %9,  %37, %38 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "ds_read_b128 %16, %40 offset:0\n"
            "ds_read_b128 %17, %41 offset:0\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %22, %30, %10, %37, %39 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %22, %31, %11, %37, %39 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %26, %32, %8,  %37, %38 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %26, %33, %9,  %37, %38 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "ds_read_b128 %18, %40 offset:2048\n"
            "ds_read_b128 %19, %41 offset:2048\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %26, %34, %10, %37, %39 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %26, %35, %11, %37, %39 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            : KPAIR_ROW_ACC_DS4 : KPAIR_INPUTS_2LDS);
    }
    if constexpr (PF_N > 4) emit_one_pf(pf1, 0);
    if constexpr (PF_N > 5) emit_one_pf(pf1, 1);
    { // Row 3 (odd, sa1)
        float4 &da_lo=da[3], &db_lo=db[3], &da_hi=da[7], &db_hi=db[7];
        uint32_t lds_a=lds_a_addrs[1], lds_b=lds_b_addrs[1];
        asm volatile(
            "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %23, %28, %12, %37, %38 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %23, %29, %13, %37, %38 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "ds_read_b128 %16, %40 offset:4096\n"
            "ds_read_b128 %17, %41 offset:4096\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %23, %30, %14, %37, %39 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %23, %31, %15, %37, %39 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %27, %32, %12, %37, %38 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %27, %33, %13, %37, %38 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "ds_read_b128 %18, %40 offset:6144\n"
            "ds_read_b128 %19, %41 offset:6144\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %27, %34, %14, %37, %39 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %27, %35, %15, %37, %39 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            : KPAIR_ROW_ACC_DS4 : KPAIR_INPUTS_2LDS);
    }
    if constexpr (PF_N > 6) emit_one_pf(pf1, 2);
    if constexpr (PF_N > 7) emit_one_pf(pf1, 3);
}

// ── 32 KPAIR MFMAs + 8 ds_reads + 8 pf (split into 4 row blocks) ──
// Each row: 8 MFMAs + 2 ds_reads (in asm) + 2 pf (C++ builtin).

template<int PF_N = 8, bool EMIT_BARRIER = false>
__device__ __forceinline__ void kpair_32mfma_with_lds_and_pf(
    fp4_floatx4_t acc[16],
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_raw[2], const fp8e8m0_4 b_raw[2],
    float4 &d0, float4 &d1, float4 &d2, float4 &d3,
    float4 &d4, float4 &d5, float4 &d6, float4 &d7,
    uint32_t lds_a0, uint32_t lds_a1,
    const tile_pf_params &pf0, const tile_pf_params &pf1)
{
    KPAIR_SETUP();
    // Row 0: 8 MFMAs + ALL 8 ds_reads front-loaded (1:1 interleave)
    // When EMIT_BARRIER: vmcnt+barrier at top, MFMAs overlap with any stall
    if constexpr (EMIT_BARRIER) {
        // R19B: site _S2 (kpair_32mfma_with_lds_and_pf, hot path)
        asm volatile(MXFP4_STEP3_BARRIER_INST_S2 ::: "memory");
    }
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %24, %32, %0,  %40, %42 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %16, %44 offset:0\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %24, %33, %1,  %40, %42 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %17, %44 offset:2048\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %24, %34, %2,  %40, %43 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %18, %44 offset:4096\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %24, %35, %3,  %40, %43 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %19, %44 offset:6144\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %28, %36, %0,  %40, %42 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %20, %45 offset:0\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %28, %37, %1,  %40, %42 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %21, %45 offset:2048\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %28, %38, %2,  %40, %43 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %22, %45 offset:4096\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %28, %39, %3,  %40, %43 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %23, %45 offset:6144\n" LDS_NOP_STR
        : KPAIR_ACC_CLOBBER,
          "=&v"(d0), "=&v"(d1), "=&v"(d2), "=&v"(d3),
          "=&v"(d4), "=&v"(d5), "=&v"(d6), "=&v"(d7)
        : KPAIR_INPUTS,
          "v"(lds_a0), "v"(lds_a1)
    );
    if constexpr (PF_N > 0) emit_one_pf(pf0, 0);
    if constexpr (PF_N > 1) emit_one_pf(pf0, 1);
    // Row 1 (odd, sa0) — pure MFMAs
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %17, %24, %4,  %32, %34 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %17, %25, %5,  %32, %34 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %17, %26, %6,  %32, %35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %17, %27, %7,  %32, %35 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %21, %28, %4,  %32, %34 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %21, %29, %5,  %32, %34 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %21, %30, %6,  %32, %35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %21, %31, %7,  %32, %35 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 2) emit_one_pf(pf0, 2);
    if constexpr (PF_N > 3) emit_one_pf(pf0, 3);
    // Row 2 (even, sa1) — pure MFMAs
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %18, %24, %8,  %33, %34 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %18, %25, %9,  %33, %34 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %18, %26, %10, %33, %35 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %18, %27, %11, %33, %35 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %22, %28, %8,  %33, %34 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %22, %29, %9,  %33, %34 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %22, %30, %10, %33, %35 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %22, %31, %11, %33, %35 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 4) emit_one_pf(pf1, 0);
    if constexpr (PF_N > 5) emit_one_pf(pf1, 1);
    // Row 3 (odd, sa1) — pure MFMAs
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %19, %24, %12, %33, %34 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %19, %25, %13, %33, %34 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %19, %26, %14, %33, %35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %19, %27, %15, %33, %35 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %23, %28, %12, %33, %34 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %23, %29, %13, %33, %34 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %23, %30, %14, %33, %35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %23, %31, %15, %33, %35 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 6) emit_one_pf(pf1, 2);
    if constexpr (PF_N > 7) emit_one_pf(pf1, 3);
}

// ── Swapped-operand MFMA variants ──
// These compute C^T instead of C by swapping A↔B operands and scales.
// The accumulator layout is transposed (row↔col).

template<int PF_N = 8>
__device__ __forceinline__ void kpair_32mfma_with_pf_swapped_sel(
    fp4_floatx4_t acc[16],
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_raw[2], const fp8e8m0_4 b_raw[2],
    const tile_pf_params &pf0, const tile_pf_params &pf1)
{
    KPAIR_SETUP();
    // Row 0
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %24, %16, %0,  %34, %32 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %25, %16, %1,  %34, %32 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %26, %16, %2,  %35, %32 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %27, %16, %3,  %35, %32 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %28, %20, %0,  %34, %32 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %29, %20, %1,  %34, %32 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %30, %20, %2,  %35, %32 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %31, %20, %3,  %35, %32 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 0) emit_one_pf(pf0, 0);
    if constexpr (PF_N > 1) emit_one_pf(pf0, 1);
    // Row 1
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %24, %17, %4,  %34, %32 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %25, %17, %5,  %34, %32 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %26, %17, %6,  %35, %32 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %27, %17, %7,  %35, %32 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %28, %21, %4,  %34, %32 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %29, %21, %5,  %34, %32 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %30, %21, %6,  %35, %32 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %31, %21, %7,  %35, %32 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 2) emit_one_pf(pf0, 2);
    if constexpr (PF_N > 3) emit_one_pf(pf0, 3);
    // Row 2
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %24, %18, %8,  %34, %33 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %25, %18, %9,  %34, %33 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %26, %18, %10, %35, %33 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %27, %18, %11, %35, %33 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %28, %22, %8,  %34, %33 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %29, %22, %9,  %34, %33 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %30, %22, %10, %35, %33 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %31, %22, %11, %35, %33 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 4) emit_one_pf(pf1, 0);
    if constexpr (PF_N > 5) emit_one_pf(pf1, 1);
    // Row 3
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %24, %19, %12, %34, %33 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %25, %19, %13, %34, %33 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %26, %19, %14, %35, %33 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %27, %19, %15, %35, %33 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %28, %23, %12, %34, %33 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %29, %23, %13, %34, %33 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %30, %23, %14, %35, %33 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %31, %23, %15, %35, %33 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 6) emit_one_pf(pf1, 2);
    if constexpr (PF_N > 7) emit_one_pf(pf1, 3);
}

__device__ __forceinline__ void kpair_32mfma_pure_swapped_plain(
    fp4_floatx4_t acc[16],
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_raw[2], const fp8e8m0_4 b_raw[2])
{
    KPAIR_SETUP();
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %24, %16, %0,  %34, %32 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %25, %16, %1,  %34, %32 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %26, %16, %2,  %35, %32 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %27, %16, %3,  %35, %32 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %28, %20, %0,  %34, %32 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %29, %20, %1,  %34, %32 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %30, %20, %2,  %35, %32 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %31, %20, %3,  %35, %32 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %24, %17, %4,  %34, %32 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %25, %17, %5,  %34, %32 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %26, %17, %6,  %35, %32 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %27, %17, %7,  %35, %32 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %28, %21, %4,  %34, %32 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %29, %21, %5,  %34, %32 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %30, %21, %6,  %35, %32 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %31, %21, %7,  %35, %32 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %24, %18, %8,  %34, %33 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %25, %18, %9,  %34, %33 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %26, %18, %10, %35, %33 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %27, %18, %11, %35, %33 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %28, %22, %8,  %34, %33 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %29, %22, %9,  %34, %33 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %30, %22, %10, %35, %33 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %31, %22, %11, %35, %33 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %24, %19, %12, %34, %33 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %25, %19, %13, %34, %33 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %26, %19, %14, %35, %33 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %27, %19, %15, %35, %33 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %28, %23, %12, %34, %33 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %29, %23, %13, %34, %33 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %30, %23, %14, %35, %33 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %31, %23, %15, %35, %33 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
}

// ── 32 KPAIR MFMAs + 8 ds_reads (spread: 2 per row) + 8 pf ──
// Instead of front-loading all 8 ds_reads in row 0, spread them
// 2 per row for better LDS pipeline utilization.
// Each row reads one lo/hi pair: (d_i, d_{i+4}) from lds_a0/lds_a1.

template<int PF_N = 8, bool EMIT_BARRIER = false>
__device__ __forceinline__ void kpair_32mfma_with_lds_rowspread_pf(
    fp4_floatx4_t acc[16],
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_raw[2], const fp8e8m0_4 b_raw[2],
    float4 &d0, float4 &d1, float4 &d2, float4 &d3,
    float4 &d4, float4 &d5, float4 &d6, float4 &d7,
    uint32_t lds_a0, uint32_t lds_a1,
    const tile_pf_params &pf0, const tile_pf_params &pf1)
{
    KPAIR_SETUP();
    if constexpr (EMIT_BARRIER) {
        // R19B: site _S3 (kpair_32mfma_with_lds_rowspread_pf, hot path)
        asm volatile(MXFP4_STEP3_BARRIER_INST_S3 ::: "memory");
    }
    // Row 0: 8 MFMAs + 2 ds_reads (d0, d4)
    {
        float4 &d_lo = d0, &d_hi = d4;
        asm volatile(
            "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %18, %26, %0,  %34, %36 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %18, %27, %1,  %34, %36 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "ds_read_b128 %16, %38 offset:0\n" LDS_NOP_STR
            "ds_read_b128 %17, %39 offset:0\n" LDS_NOP_STR
            "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %18, %28, %2,  %34, %37 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %18, %29, %3,  %34, %37 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %22, %30, %0,  %34, %36 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %22, %31, %1,  %34, %36 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %22, %32, %2,  %34, %37 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %22, %33, %3,  %34, %37 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            : KPAIR_ACC_CLOBBER, "=&v"(d_lo), "=&v"(d_hi)
            : KPAIR_INPUTS, "v"(lds_a0), "v"(lds_a1)
        );
    }
    if constexpr (PF_N > 0) emit_one_pf(pf0, 0);
    if constexpr (PF_N > 1) emit_one_pf(pf0, 1);
    // Row 1: 8 MFMAs + 2 ds_reads (d1, d5)
    {
        float4 &d_lo = d1, &d_hi = d5;
        asm volatile(
            "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %19, %26, %4,  %34, %36 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %19, %27, %5,  %34, %36 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "ds_read_b128 %16, %38 offset:2048\n" LDS_NOP_STR
            "ds_read_b128 %17, %39 offset:2048\n" LDS_NOP_STR
            "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %19, %28, %6,  %34, %37 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %19, %29, %7,  %34, %37 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %23, %30, %4,  %34, %36 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %23, %31, %5,  %34, %36 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %23, %32, %6,  %34, %37 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %23, %33, %7,  %34, %37 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            : KPAIR_ACC_CLOBBER, "=&v"(d_lo), "=&v"(d_hi)
            : KPAIR_INPUTS, "v"(lds_a0), "v"(lds_a1)
        );
    }
    if constexpr (PF_N > 2) emit_one_pf(pf0, 2);
    if constexpr (PF_N > 3) emit_one_pf(pf0, 3);
    // Row 2: 8 MFMAs + 2 ds_reads (d2, d6)
    {
        float4 &d_lo = d2, &d_hi = d6;
        asm volatile(
            "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %20, %26, %8,  %35, %36 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %20, %27, %9,  %35, %36 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "ds_read_b128 %16, %38 offset:4096\n" LDS_NOP_STR
            "ds_read_b128 %17, %39 offset:4096\n" LDS_NOP_STR
            "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %20, %28, %10, %35, %37 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %20, %29, %11, %35, %37 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %24, %30, %8,  %35, %36 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %24, %31, %9,  %35, %36 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %24, %32, %10, %35, %37 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %24, %33, %11, %35, %37 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            : KPAIR_ACC_CLOBBER, "=&v"(d_lo), "=&v"(d_hi)
            : KPAIR_INPUTS, "v"(lds_a0), "v"(lds_a1)
        );
    }
    if constexpr (PF_N > 4) emit_one_pf(pf1, 0);
    if constexpr (PF_N > 5) emit_one_pf(pf1, 1);
    // Row 3: 8 MFMAs + 2 ds_reads (d3, d7)
    {
        float4 &d_lo = d3, &d_hi = d7;
        asm volatile(
            "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %21, %26, %12, %35, %36 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %21, %27, %13, %35, %36 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "ds_read_b128 %16, %38 offset:6144\n" LDS_NOP_STR
            "ds_read_b128 %17, %39 offset:6144\n" LDS_NOP_STR
            "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %21, %28, %14, %35, %37 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %21, %29, %15, %35, %37 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %25, %30, %12, %35, %36 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %25, %31, %13, %35, %36 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %25, %32, %14, %35, %37 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %25, %33, %15, %35, %37 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            : KPAIR_ACC_CLOBBER, "=&v"(d_lo), "=&v"(d_hi)
            : KPAIR_INPUTS, "v"(lds_a0), "v"(lds_a1)
        );
    }
    if constexpr (PF_N > 6) emit_one_pf(pf1, 2);
    if constexpr (PF_N > 7) emit_one_pf(pf1, 3);
}

// Swapped Step12: fused ds_read schedule with operand-swapped MFMAs
__device__ __forceinline__ void kpair_64mfma_step12_swapped_sel(
    fp4_floatx4_t acc_bl[16], fp4_floatx4_t acc_br[16],
    const fp4_intx8_t A0[4], const fp4_intx8_t Bl[4],
    const fp8e8m0_4 a_raw[2], const fp8e8m0_4 bl_raw[2], const fp8e8m0_4 br_raw[2],
    float4 br_d[8], float4 a1_d[8],
    uint32_t br_p0, uint32_t br_p1,
    uint32_t a1_p0, uint32_t a1_p1)
{
    fp4_intx4_t a0l=fp4_lo4(A0[0]), a1l=fp4_lo4(A0[1]), a2l=fp4_lo4(A0[2]), a3l=fp4_lo4(A0[3]);
    fp4_intx4_t a0h=fp4_hi4(A0[0]), a1h=fp4_hi4(A0[1]), a2h=fp4_hi4(A0[2]), a3h=fp4_hi4(A0[3]);
    fp4_intx4_t b0l=fp4_lo4(Bl[0]), b1l=fp4_lo4(Bl[1]), b2l=fp4_lo4(Bl[2]), b3l=fp4_lo4(Bl[3]);
    fp4_intx4_t b0h=fp4_hi4(Bl[0]), b1h=fp4_hi4(Bl[1]), b2h=fp4_hi4(Bl[2]), b3h=fp4_hi4(Bl[3]);
    unsigned sa0 = std::bit_cast<unsigned>(a_raw[0]);
    unsigned sa1 = std::bit_cast<unsigned>(a_raw[1]);
    unsigned sb_bl0 = std::bit_cast<unsigned>(bl_raw[0]);
    unsigned sb_bl1 = std::bit_cast<unsigned>(bl_raw[1]);
    unsigned sb_br0 = std::bit_cast<unsigned>(br_raw[0]);
    unsigned sb_br1 = std::bit_cast<unsigned>(br_raw[1]);

    asm volatile(
        // STEP 1: A0*Bl (swapped) + 8 ds_reads for Br
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %56, %48, %0,  %66, %64 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %32, %70 offset:0\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %57, %48, %1,  %66, %64 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %33, %70 offset:2048\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %58, %48, %2,  %67, %64 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %34, %70 offset:4096\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %59, %48, %3,  %67, %64 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %35, %70 offset:6144\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %60, %52, %0,  %66, %64 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %36, %71 offset:0\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %61, %52, %1,  %66, %64 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %37, %71 offset:2048\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %62, %52, %2,  %67, %64 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %38, %71 offset:4096\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %63, %52, %3,  %67, %64 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %39, %71 offset:6144\n" LDS_NOP_STR
        // Rows 1-3: 24 pure Step 1 MFMAs
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %56, %49, %4,  %66, %64 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %57, %49, %5,  %66, %64 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %58, %49, %6,  %67, %64 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %59, %49, %7,  %67, %64 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %60, %53, %4,  %66, %64 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %61, %53, %5,  %66, %64 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %62, %53, %6,  %67, %64 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %63, %53, %7,  %67, %64 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %56, %50, %8,  %66, %65 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %57, %50, %9,  %66, %65 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %58, %50, %10, %67, %65 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %59, %50, %11, %67, %65 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %60, %54, %8,  %66, %65 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %61, %54, %9,  %66, %65 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %62, %54, %10, %67, %65 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %63, %54, %11, %67, %65 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %56, %51, %12, %66, %65 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %57, %51, %13, %66, %65 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %58, %51, %14, %67, %65 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %59, %51, %15, %67, %65 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %60, %55, %12, %66, %65 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %61, %55, %13, %66, %65 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %62, %55, %14, %67, %65 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %63, %55, %15, %67, %65 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Wait for Br ds_reads (tunable: lgkmcnt(STEP12_BR_LGKMCNT))
        "s_waitcnt lgkmcnt(" MXFP4_STR(STEP12_BR_LGKMCNT) ")\n"
        // STEP 2: A0*Br (swapped) + 8 ds_reads for A1
        "v_mfma_scale_f32_16x16x128_f8f6f4 %16, %32, %48, %16, %68, %64 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %40, %72 offset:0\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %17, %33, %48, %17, %68, %64 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %41, %72 offset:2048\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %18, %34, %48, %18, %69, %64 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %42, %72 offset:4096\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %19, %35, %48, %19, %69, %64 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %43, %72 offset:6144\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %16, %36, %52, %16, %68, %64 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %44, %73 offset:0\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %17, %37, %52, %17, %68, %64 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %45, %73 offset:2048\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %18, %38, %52, %18, %69, %64 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %46, %73 offset:4096\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %19, %39, %52, %19, %69, %64 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %47, %73 offset:6144\n" LDS_NOP_STR
        // Rows 1-3: 24 pure Step 2 MFMAs
        "v_mfma_scale_f32_16x16x128_f8f6f4 %20, %32, %49, %20, %68, %64 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %21, %33, %49, %21, %68, %64 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %22, %34, %49, %22, %69, %64 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %23, %35, %49, %23, %69, %64 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %20, %36, %53, %20, %68, %64 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %21, %37, %53, %21, %68, %64 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %22, %38, %53, %22, %69, %64 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %23, %39, %53, %23, %69, %64 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %24, %32, %50, %24, %68, %65 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %25, %33, %50, %25, %68, %65 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %26, %34, %50, %26, %69, %65 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %27, %35, %50, %27, %69, %65 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %24, %36, %54, %24, %68, %65 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %25, %37, %54, %25, %68, %65 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %26, %38, %54, %26, %69, %65 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %27, %39, %54, %27, %69, %65 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %28, %32, %51, %28, %68, %65 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %29, %33, %51, %29, %68, %65 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %30, %34, %51, %30, %69, %65 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %31, %35, %51, %31, %69, %65 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %28, %36, %55, %28, %68, %65 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %29, %37, %55, %29, %68, %65 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %30, %38, %55, %30, %69, %65 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %31, %39, %55, %31, %69, %65 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc_bl[0]),  "+a"(acc_bl[1]),  "+a"(acc_bl[2]),  "+a"(acc_bl[3]),
          "+a"(acc_bl[4]),  "+a"(acc_bl[5]),  "+a"(acc_bl[6]),  "+a"(acc_bl[7]),
          "+a"(acc_bl[8]),  "+a"(acc_bl[9]),  "+a"(acc_bl[10]), "+a"(acc_bl[11]),
          "+a"(acc_bl[12]), "+a"(acc_bl[13]), "+a"(acc_bl[14]), "+a"(acc_bl[15]),
          "+a"(acc_br[0]),  "+a"(acc_br[1]),  "+a"(acc_br[2]),  "+a"(acc_br[3]),
          "+a"(acc_br[4]),  "+a"(acc_br[5]),  "+a"(acc_br[6]),  "+a"(acc_br[7]),
          "+a"(acc_br[8]),  "+a"(acc_br[9]),  "+a"(acc_br[10]), "+a"(acc_br[11]),
          "+a"(acc_br[12]), "+a"(acc_br[13]), "+a"(acc_br[14]), "+a"(acc_br[15]),
          "=&v"(br_d[0]), "=&v"(br_d[1]), "=&v"(br_d[2]), "=&v"(br_d[3]),
          "=&v"(br_d[4]), "=&v"(br_d[5]), "=&v"(br_d[6]), "=&v"(br_d[7]),
          "=&v"(a1_d[0]), "=&v"(a1_d[1]), "=&v"(a1_d[2]), "=&v"(a1_d[3]),
          "=&v"(a1_d[4]), "=&v"(a1_d[5]), "=&v"(a1_d[6]), "=&v"(a1_d[7])
        : "v"(a0l), "v"(a1l), "v"(a2l), "v"(a3l),
          "v"(a0h), "v"(a1h), "v"(a2h), "v"(a3h),
          "v"(b0l), "v"(b1l), "v"(b2l), "v"(b3l),
          "v"(b0h), "v"(b1h), "v"(b2h), "v"(b3h),
          "v"(sa0), "v"(sa1), "v"(sb_bl0), "v"(sb_bl1),
          "v"(sb_br0), "v"(sb_br1),
          "v"(br_p0), "v"(br_p1), "v"(a1_p0), "v"(a1_p1)
    );
}

template<int PF_N = 8, bool EMIT_BARRIER = false>
__device__ __forceinline__ void kpair_32mfma_with_lds_and_pf_swapped_sel(
    fp4_floatx4_t acc[16],
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_raw[2], const fp8e8m0_4 b_raw[2],
    float4 &d0, float4 &d1, float4 &d2, float4 &d3,
    float4 &d4, float4 &d5, float4 &d6, float4 &d7,
    uint32_t lds_a0, uint32_t lds_a1,
    const tile_pf_params &pf0, const tile_pf_params &pf1)
{
    KPAIR_SETUP();
    if constexpr (EMIT_BARRIER) {
        // R19B: site _S4 (kpair_32mfma_with_lds_and_pf_swapped_sel, hot path)
        asm volatile(MXFP4_STEP3_BARRIER_INST_S4 ::: "memory");
    }
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %32, %24, %0,  %42, %40 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %16, %44 offset:0\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %33, %24, %1,  %42, %40 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %17, %44 offset:2048\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %34, %24, %2,  %43, %40 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %18, %44 offset:4096\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %35, %24, %3,  %43, %40 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %19, %44 offset:6144\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %36, %28, %0,  %42, %40 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %20, %45 offset:0\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %37, %28, %1,  %42, %40 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %21, %45 offset:2048\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %38, %28, %2,  %43, %40 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %22, %45 offset:4096\n" LDS_NOP_STR
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %39, %28, %3,  %43, %40 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %23, %45 offset:6144\n" LDS_NOP_STR
        : KPAIR_ACC_CLOBBER,
          "=&v"(d0), "=&v"(d1), "=&v"(d2), "=&v"(d3),
          "=&v"(d4), "=&v"(d5), "=&v"(d6), "=&v"(d7)
        : KPAIR_INPUTS,
          "v"(lds_a0), "v"(lds_a1)
    );
    if constexpr (PF_N > 0) emit_one_pf(pf0, 0);
    if constexpr (PF_N > 1) emit_one_pf(pf0, 1);
    // Row 1
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %24, %17, %4,  %34, %32 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %25, %17, %5,  %34, %32 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %26, %17, %6,  %35, %32 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %27, %17, %7,  %35, %32 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %28, %21, %4,  %34, %32 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %29, %21, %5,  %34, %32 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %30, %21, %6,  %35, %32 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %31, %21, %7,  %35, %32 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 2) emit_one_pf(pf0, 2);
    if constexpr (PF_N > 3) emit_one_pf(pf0, 3);
    // Row 2
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %24, %18, %8,  %34, %33 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %25, %18, %9,  %34, %33 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %26, %18, %10, %35, %33 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %27, %18, %11, %35, %33 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %28, %22, %8,  %34, %33 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %29, %22, %9,  %34, %33 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %30, %22, %10, %35, %33 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %31, %22, %11, %35, %33 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 4) emit_one_pf(pf1, 0);
    if constexpr (PF_N > 5) emit_one_pf(pf1, 1);
    // Row 3
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %24, %19, %12, %34, %33 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %25, %19, %13, %34, %33 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %26, %19, %14, %35, %33 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %27, %19, %15, %35, %33 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %28, %23, %12, %34, %33 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %29, %23, %13, %34, %33 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %30, %23, %14, %35, %33 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %31, %23, %15, %35, %33 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 6) emit_one_pf(pf1, 2);
    if constexpr (PF_N > 7) emit_one_pf(pf1, 3);
}

// ══════════════════════════════════════════════════════════════
// Main kernel
// ══════════════════════════════════════════════════════════════

#if PERSISTENT_XCD
// Global tile counter for persistent-kernel mode. Reset to 0 from host before each launch.
__device__ unsigned int g_persistent_tile_counter = 0;
#endif

#if defined(WAVES_PER_EU_1)
__attribute__((amdgpu_waves_per_eu(1, 1)))
#elif defined(WAVES_PER_EU_2)
__attribute__((amdgpu_waves_per_eu(2, 2)))
#endif
#if defined(AGPR_REGS_HINT_192)
__attribute__((amdgpu_num_agpr(192)))
#elif defined(AGPR_REGS_HINT_128)
__attribute__((amdgpu_num_agpr(128)))
#elif defined(AGPR_REGS_HINT_256)
__attribute__((amdgpu_num_agpr(256)))
#endif
__global__ __launch_bounds__(_NUM_THREADS, 1)
void mxfp4_gluon_cpp_kernel(const gluon_globals g) {
#if WAVE_PRIO_HIGH
    asm volatile("s_setprio 3" ::: "memory");
#endif
    static_assert(K_BYTES % BK == 0 && N_DIM % BLK == 0 && M_DIM % BLK == 0);

    constexpr int bpc = N_DIM / BLK;
    constexpr int a_packs = RBM / 32;
    constexpr int b_packs = RBN / 32;

    __shared__ ST_tile A0_db[2], A1_db[2], Bl_db[2], Br_db[2];

    // XCD-aware dispatch + GROUP_SIZE_M swizzle for L2 B-tile reuse
    constexpr int NUM_XCDS = 8;
#ifndef GROUP_SIZE_M
#define GROUP_SIZE_M 4
#endif
    constexpr int GROUP_M = GROUP_SIZE_M;
#if PERSISTENT_XCD
    // Persistent grid: total_blocks = real number of tiles in problem (M/BLK * N/BLK).
    // gridDim.x = PERSISTENT_GRID (e.g. 608). Each WG loops, atomically claiming tile_id.
    const int total_blocks = (M_DIM / BLK) * (N_DIM / BLK);
#else
    const int total_blocks = gridDim.x;
#endif
    const int bpr = total_blocks / bpc;

    const int pids_per_xcd = (total_blocks + NUM_XCDS - 1) / NUM_XCDS;
    int tall_xcds = total_blocks % NUM_XCDS;
    if (tall_xcds == 0) tall_xcds = NUM_XCDS;

#if PERSISTENT_XCD
    // ── Persistent loop entry ──
    // Per-WG cache of the next tile (for batch>1 we reuse claims).
    __shared__ unsigned int s_claim_base;
    unsigned int local_offset = PERSISTENT_BATCH;  // forces first-iter claim

    while (true) {
        // ── Claim next tile_id via atomicAdd on the global counter ──
        if (local_offset >= PERSISTENT_BATCH) {
            if (threadIdx.x == 0) {
                s_claim_base = atomicAdd(&g_persistent_tile_counter,
                                         (unsigned int)PERSISTENT_BATCH);
            }
            __syncthreads();
            local_offset = 0;
        }
        const int raw_bid = (int)(s_claim_base + local_offset);
        local_offset += 1;
        if (raw_bid >= total_blocks) break;
#else
    {
        // Static dispatch: raw_bid = blockIdx.x
        const int raw_bid = (int)blockIdx.x;
#endif

    // XCD pid remapping: Gluon-style "tall XCDs" for correct remainder handling
    const int xcd = raw_bid % NUM_XCDS;
    const int local_pid = raw_bid / NUM_XCDS;
    int bid;
    if (xcd < tall_xcds) {
        bid = xcd * pids_per_xcd + local_pid;
    } else {
        bid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid;
    }
#if PERSISTENT_XCD
    if (bid >= total_blocks) continue;
#else
    if (bid >= total_blocks) return;
#endif

    int br, bc;
#if STATIC_XCD_REMAP
    // Round 7: atomic-free static remap.
    // Each XCD owns a narrow N-strip of width n_per_xcd = bpc / NUM_XCDS.
    // The original `bid` is a contiguous range in [xcd*pids_per_xcd, ..),
    // length pids_per_xcd. We re-interpret it as a (M-stripe × n_per_xcd) walk:
    //   inner_bid = bid - xcd * pids_per_xcd   (0..pids_per_xcd-1)
    //   gid_m  = inner_bid / (GROUP_M * n_per_xcd)
    //   in_gp  = inner_bid % (GROUP_M * n_per_xcd)
    //   br     = gid_m*GROUP_M + (in_gp % gsm)
    //   bc_loc = in_gp / gsm
    //   bc     = xcd * n_per_xcd + bc_loc
    // Falls back to default if bpc not divisible by NUM_XCDS or tile counts mismatch.
    constexpr int n_per_xcd_const = (bpc) / NUM_XCDS;  // bpc is constexpr
    if constexpr ((bpc) % NUM_XCDS == 0) {
        const int xcd_pid_base = xcd * pids_per_xcd;
        const int inner_bid = bid - xcd_pid_base;
        const int g_m_size = GROUP_M * n_per_xcd_const;
        const int gid_m = inner_bid / g_m_size;
        const int in_gp = inner_bid - gid_m * g_m_size;
        const int fpm = gid_m * GROUP_M;
        const int gsm = (bpr - fpm < GROUP_M) ? (bpr - fpm) : GROUP_M;
        br = fpm + (in_gp % gsm);
        const int bc_loc = in_gp / gsm;
        bc = xcd * n_per_xcd_const + bc_loc;
    } else {
        // Non-divisible fallback: default GROUP_M swizzle on full bid space
        const int num_pig = GROUP_M * bpc;
        const int gid = bid / num_pig;
        const int fpm = gid * GROUP_M;
        const int gsm = (bpr - fpm < GROUP_M) ? (bpr - fpm) : GROUP_M;
        br = fpm + (bid % gsm);
        bc = (bid % num_pig) / gsm;
    }
#else
    // GROUP_SIZE_M swizzle within XCD's block range
    const int num_pig = GROUP_M * bpc;
    const int gid = bid / num_pig;
    const int fpm = gid * GROUP_M;
    const int gsm = (bpr - fpm < GROUP_M) ? (bpr - fpm) : GROUP_M;
    br = fpm + (bid % gsm);
    bc = (bid % num_pig) / gsm;
#endif
    const int wm = warpid() / WARPS_N, wn = warpid() % WARPS_N;

    uint32_t so_a[PF_MPT], so_b[PF_MPT];
    G::prefill_swizzled_offsets(A0_db[0], g.a, so_a);
    G::prefill_swizzled_offsets(Bl_db[0], g.b, so_b);

    // Scale SRDs — merged preshuffle format (64-row super-groups, dwordx2 loads)
    // lane_soff_x2: doubled offsets for merged format where each dword position is 8 bytes
    const uint32_t lane_soff_x2 =
        (static_cast<uint32_t>(kittens::laneid() / 16) << 7) |
        (static_cast<uint32_t>(kittens::laneid() % 16) << 3);

    // One SRD per tile-half, pointing to the 64-row super-group base
    i32x4 a0_srd = make_scale_srd(preshuffled_scale_row_base_ptr(
        g.a_scale, (br * BLK + wm * RBM) >> 6));
    i32x4 a1_srd = make_scale_srd(preshuffled_scale_row_base_ptr(
        g.a_scale, (br * BLK + HB + wm * RBM) >> 6));
    i32x4 bl_srd = make_scale_srd(preshuffled_scale_row_base_ptr(
        g.b_scale, (bc * BLK + wn * RBN) >> 6));
    i32x4 br_srd = make_scale_srd(preshuffled_scale_row_base_ptr(
        g.b_scale, (bc * BLK + HB + wn * RBN) >> 6));

    fp4_floatx4_t acc_A0Bl[16]={}, acc_A0Br[16]={}, acc_A1Bl[16]={}, acc_A1Br[16]={};

#if DIRECT_BL
    // Preshuffled B SRD and addressing for half-direct Bl loading
    const uint32_t bl_voff_ps =
        static_cast<uint32_t>(kittens::laneid() / 16) * 256 +
        static_cast<uint32_t>(kittens::laneid() % 16) * 16;
    const uint32_t bl_ps_n0_base = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>((bc * BLK + wn * RBN) / 16));
    const uint32_t bl_ps_n0_soff = bl_ps_n0_base * BL_N0_STRIDE;
#endif

    // Tile SRDs
    auto make_srd = [](const void* raw_ptr) {
        i32x4 s = std::bit_cast<i32x4>(make_buffer_resource(
            static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(raw_ptr)),
            0xFFFFFFFFu, 0x00110000u));
        s[0] = __builtin_amdgcn_readfirstlane(s[0]);
        s[1] = __builtin_amdgcn_readfirstlane(s[1]);
        s[2] = __builtin_amdgcn_readfirstlane(s[2]);
        s[3] = __builtin_amdgcn_readfirstlane(s[3]);
        return s;
    };
    i32x4 srd_a = make_srd(g.a.raw_ptr), srd_b = make_srd(g.b.raw_ptr);
#if DIRECT_BL
    i32x4 srd_b_ps = make_srd(g.b_ps.raw_ptr);
#endif
    const void *base_a = (const void*)g.a.raw_ptr, *base_b = (const void*)g.b.raw_ptr;

    constexpr int epw = 16 / sizeof(fp8e4m3) * WARP_THREADS;
    const uint32_t wlo = (warpid() % _NUM_WARPS) * epw * sizeof(fp8e4m3);
    auto lb = [&](auto &t) -> uint32_t {
        return __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(&t.data[0]) + wlo));
    };
    uint32_t lb_a0[2], lb_a1[2], lb_br[2];
#if !DIRECT_BL
    uint32_t lb_bl[2];
#endif
    for (int d = 0; d < 2; ++d) {
        lb_a0[d]=lb(A0_db[d]); lb_a1[d]=lb(A1_db[d]);
#if !DIRECT_BL
        lb_bl[d]=lb(Bl_db[d]);
#endif
        lb_br[d]=lb(Br_db[d]);
    }

    constexpr int R22B_A_HINT_VAL = static_cast<int>(MXFP4_R22B_A_HINT);
    constexpr int R22B_B_HINT_VAL = static_cast<int>(MXFP4_R22B_B_HINT);
    auto load_tiles = [&](int bt, int db) {
        emit_tile_pf(A0_db[db], g.a, coord<ST_tile>(0,0,br*2,    bt), so_a, srd_a, base_a, lb_a0[db], R22B_A_HINT_VAL);
        emit_tile_pf(A1_db[db], g.a, coord<ST_tile>(0,0,br*2+1,  bt), so_a, srd_a, base_a, lb_a1[db], R22B_A_HINT_VAL);
#if !DIRECT_BL
        emit_tile_pf(Bl_db[db], g.b, coord<ST_tile>(0,0,bc*2,    bt), so_b, srd_b, base_b, lb_bl[db], R22B_B_HINT_VAL);
#endif
        emit_tile_pf(Br_db[db], g.b, coord<ST_tile>(0,0,bc*2+1,  bt), so_b, srd_b, base_b, lb_br[db], R22B_B_HINT_VAL);
    };

    // Pre-compute LDS addresses as 16 static named variables (one per db-slot × phase).
    // Avoids runtime-indexed [2][2] arrays that compiler spills to LDS + ds_read_b64.
    // Selection via ternary (compiles to v_cndmask). No swap needed.
    uint32_t a0_0_p0, a0_0_p1, a0_1_p0, a0_1_p1;
#if !DIRECT_BL
    uint32_t bl_0_p0, bl_0_p1, bl_1_p0, bl_1_p1;
#endif
    uint32_t br_0_p0, br_0_p1, br_1_p0, br_1_p1;
    uint32_t a1_0_p0, a1_0_p1, a1_1_p0, a1_1_p1;
    compute_lds_base_addrs<A_row_reg>(kittens::subtile_inplace<RBM, BK>(A0_db[0], {wm, 0}), a0_0_p0, a0_0_p1);
    compute_lds_base_addrs<A_row_reg>(kittens::subtile_inplace<RBM, BK>(A0_db[1], {wm, 0}), a0_1_p0, a0_1_p1);
#if !DIRECT_BL
    compute_lds_base_addrs<B_row_reg>(kittens::subtile_inplace<RBN, BK>(Bl_db[0], {wn, 0}), bl_0_p0, bl_0_p1);
    compute_lds_base_addrs<B_row_reg>(kittens::subtile_inplace<RBN, BK>(Bl_db[1], {wn, 0}), bl_1_p0, bl_1_p1);
#endif
    compute_lds_base_addrs<B_row_reg>(kittens::subtile_inplace<RBN, BK>(Br_db[0], {wn, 0}), br_0_p0, br_0_p1);
    compute_lds_base_addrs<B_row_reg>(kittens::subtile_inplace<RBN, BK>(Br_db[1], {wn, 0}), br_1_p0, br_1_p1);
    compute_lds_base_addrs<A_row_reg>(kittens::subtile_inplace<RBM, BK>(A1_db[0], {wm, 0}), a1_0_p0, a1_0_p1);
    compute_lds_base_addrs<A_row_reg>(kittens::subtile_inplace<RBM, BK>(A1_db[1], {wm, 0}), a1_1_p0, a1_1_p1);

    // Tile extraction helper (ds_read float4[8] → fp4_intx8_t[4])
    auto extract_tile = [](const float4 d[8], fp4_intx8_t t[4]) __attribute__((always_inline)) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            auto lo = *reinterpret_cast<const fp4_intx4_t*>(&d[i]);
            auto hi = *reinterpret_cast<const fp4_intx4_t*>(&d[i + 4]);
            t[i][0]=lo[0]; t[i][1]=lo[1]; t[i][2]=lo[2]; t[i][3]=lo[3];
            t[i][4]=hi[0]; t[i][5]=hi[1]; t[i][6]=hi[2]; t[i][7]=hi[3];
        }
    };

    // ═══════════ Prologue ═══════════
    load_tiles(0, 0);
    if (k_byte_iters > 1) load_tiles(1, 1);

    fp8e8m0_4 pf_a0[a_packs], pf_a1[a_packs], pf_bl[b_packs], pf_br[b_packs];
    {
        load_pq_scale_x2_async(a0_srd, lane_soff_x2, 0, pf_a0[0], pf_a0[1]);
        load_pq_scale_x2_async(a1_srd, lane_soff_x2, 0, pf_a1[0], pf_a1[1]);
        load_pq_scale_x2_async(bl_srd, lane_soff_x2, 0, pf_bl[0], pf_bl[1]);
        load_pq_scale_x2_async(br_srd, lane_soff_x2, 0, pf_br[0], pf_br[1]);
    }

    // Pre-load A0+Bl for iteration 0 (2-tile software pipeline)
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();
    A_row_reg a0_rt;
    fp4_load_st_to_rt(a0_rt, kittens::subtile_inplace<RBM, BK>(A0_db[0], {wm, 0}));
#if DIRECT_BL
    // Load initial Bl directly from preshuffled global memory
    float4 bl_init_vmem[8];
    load_bl_direct_async(bl_init_vmem, srd_b_ps, bl_voff_ps,
        bl_ps_n0_soff + 0 * 2 * BL_K0_STRIDE);  // bt=0
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
    fp4_intx8_t tA0[4], tBl[4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        tA0[i] = fp4_extract_tile(a0_rt, i);
    extract_tile(bl_init_vmem, tBl);
#else
    B_row_reg bl_rt;
    fp4_load_st_to_rt(bl_rt, kittens::subtile_inplace<RBN, BK>(Bl_db[0], {wn, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    fp4_intx8_t tA0[4], tBl[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        tA0[i] = fp4_extract_tile(a0_rt, i);
        tBl[i] = fp4_extract_tile(bl_rt, i);
    }
#endif

    // ═══════════ Main loop (2-tile pipeline) ═══════════
#if SWAP_STEP34_MAIN
    // Split steady-state iterations from the final tail to keep the hot loop free
    // of last-iteration branches and reduce live-range pressure.
#ifdef UNROLL_K
  #if UNROLL_K == 0
    #pragma unroll
  #else
    #pragma unroll UNROLL_K
  #endif
#elif (K_DIM / 256) <= 16
    #pragma unroll
#elif (K_DIM / 256) <= 32
    #pragma unroll 16
#else
    #pragma unroll 8
#endif
    for (int bt = 0; bt + 1 < k_byte_iters; ++bt) {
        const int cur = bt & 1;
        const int nxt = 1 - cur;
        const uint32_t sel_br_p0 = cur ? br_1_p0 : br_0_p0;
        const uint32_t sel_br_p1 = cur ? br_1_p1 : br_0_p1;
        const uint32_t sel_a1_p0 = cur ? a1_1_p0 : a1_0_p0;
        const uint32_t sel_a1_p1 = cur ? a1_1_p1 : a1_0_p1;
        const uint32_t sel_a0_p0 = nxt ? a0_1_p0 : a0_0_p0;
        const uint32_t sel_a0_p1 = nxt ? a0_1_p1 : a0_0_p1;
#if !DIRECT_BL
#if !DIRECT_BL
        const uint32_t sel_bl_p0 = nxt ? bl_1_p0 : bl_0_p0;
        const uint32_t sel_bl_p1 = nxt ? bl_1_p1 : bl_0_p1;
#endif
#endif

        const int pf_bt = (bt + 2 < k_byte_iters) ? (bt + 2) : (k_byte_iters - 1);
        tile_pf_params pf_a0_p = make_pf_params(A0_db[cur], g.a, coord<ST_tile>(0,0,br*2,     pf_bt), so_a, srd_a, base_a, lb_a0[cur], R22B_A_HINT_VAL);
        tile_pf_params pf_a1_p = make_pf_params(A1_db[cur], g.a, coord<ST_tile>(0,0,br*2+1,   pf_bt), so_a, srd_a, base_a, lb_a1[cur], R22B_A_HINT_VAL);
#if !DIRECT_BL
#if !DIRECT_BL
        tile_pf_params pf_bl_p = make_pf_params(Bl_db[cur], g.b, coord<ST_tile>(0,0,bc*2,     pf_bt), so_b, srd_b, base_b, lb_bl[cur], R22B_B_HINT_VAL);
#endif
#endif
        tile_pf_params pf_br_p = make_pf_params(Br_db[cur], g.b, coord<ST_tile>(0,0,bc*2+1,   pf_bt), so_b, srd_b, base_b, lb_br[cur], R22B_B_HINT_VAL);

#if EARLY_SCALE_PF
        // Hoist scale loads ABOVE the _raw copy: write to shadow regs nxt_pf_* so the
        // outstanding VMEM does not collide with the current pf_* (still being copied
        // to _raw). pf_* is updated from nxt_pf_* at end of iteration after vmcnt.
        fp8e8m0_4 nxt_pf_a0[a_packs], nxt_pf_a1[a_packs], nxt_pf_bl[b_packs], nxt_pf_br[b_packs];
        {
            const uint32_t nxt_scale = static_cast<uint32_t>(bt + 1) << 9;
            load_pq_scale_x2_async(a0_srd, lane_soff_x2, nxt_scale, nxt_pf_a0[0], nxt_pf_a0[1]);
            load_pq_scale_x2_async(a1_srd, lane_soff_x2, nxt_scale, nxt_pf_a1[0], nxt_pf_a1[1]);
            load_pq_scale_x2_async(bl_srd, lane_soff_x2, nxt_scale, nxt_pf_bl[0], nxt_pf_bl[1]);
            load_pq_scale_x2_async(br_srd, lane_soff_x2, nxt_scale, nxt_pf_br[0], nxt_pf_br[1]);
        }
#endif

        fp8e8m0_4 a0_raw[a_packs], a1_raw[a_packs], bl_raw[b_packs], br_raw[b_packs];
        #pragma unroll
        for (int p = 0; p < a_packs; ++p) { a0_raw[p] = pf_a0[p]; a1_raw[p] = pf_a1[p]; }
        #pragma unroll
        for (int p = 0; p < b_packs; ++p) { bl_raw[p] = pf_bl[p]; br_raw[p] = pf_br[p]; }

#if !EARLY_SCALE_PF
        {
            const uint32_t nxt_scale = static_cast<uint32_t>(bt + 1) << 9;
            load_pq_scale_x2_async(a0_srd, lane_soff_x2, nxt_scale, pf_a0[0], pf_a0[1]);
            load_pq_scale_x2_async(a1_srd, lane_soff_x2, nxt_scale, pf_a1[0], pf_a1[1]);
            load_pq_scale_x2_async(bl_srd, lane_soff_x2, nxt_scale, pf_bl[0], pf_bl[1]);
            load_pq_scale_x2_async(br_srd, lane_soff_x2, nxt_scale, pf_br[0], pf_br[1]);
        }
#endif

        float4 nxt_bl_d[8];
#if DIRECT_BL && EARLY_BL_PF
        // Issue Bl buffer_load BEFORE Step12 — ~128 MFMAs (~512 cyc) hiding window
        load_bl_direct_async(nxt_bl_d, srd_b_ps, bl_voff_ps,
            bl_ps_n0_soff + static_cast<uint32_t>(bt + 1) * 2 * BL_K0_STRIDE);
#endif

        float4 br_d[8], a1_d[8];
#if SWAP_STEP12_MAIN
        kpair_64mfma_step12_swapped_sel(acc_A0Bl, acc_A0Br, tA0, tBl,
            a0_raw, bl_raw, br_raw, br_d, a1_d,
            sel_br_p0, sel_br_p1, sel_a1_p0, sel_a1_p1);
#else
        kpair_64mfma_step12(acc_A0Bl, acc_A0Br, tA0, tBl,
            a0_raw, bl_raw, br_raw, br_d, a1_d,
            sel_br_p0, sel_br_p1, sel_a1_p0, sel_a1_p1);
#endif

        asm volatile("s_waitcnt lgkmcnt(0)");
        fp4_intx8_t tBr[4], tA1[4];
        extract_tile(br_d, tBr);
        extract_tile(a1_d, tA1);

#if !STEP3_EMBED_BARRIER
        // R19B: site _S5 (TAIL_SPLIT inner !STEP3_EMBED_BARRIER, dead w/ default STEP3_EMBED_BARRIER=1)
        asm volatile(MXFP4_STEP3_BARRIER_INST_S5 ::: "memory");
#endif

        float4 nxt_a0_d[8];
#if K_LOOP_SYNC_EVERY_4
        // R20C: barrier only every 4 iters (bt%4==0). Compiler unrolls and
        // statically resolves the parity per unrolled copy.
        if ((bt & 3) == 0) {
            kpair_32mfma_with_lds_and_pf_swapped_sel<STEP3_PF_N, STEP3_EMBED_BARRIER>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw,
                nxt_a0_d[0], nxt_a0_d[1], nxt_a0_d[2], nxt_a0_d[3],
                nxt_a0_d[4], nxt_a0_d[5], nxt_a0_d[6], nxt_a0_d[7],
                sel_a0_p0, sel_a0_p1, pf_a0_p, pf_a1_p);
        } else {
            kpair_32mfma_with_lds_and_pf_swapped_sel<STEP3_PF_N, false>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw,
                nxt_a0_d[0], nxt_a0_d[1], nxt_a0_d[2], nxt_a0_d[3],
                nxt_a0_d[4], nxt_a0_d[5], nxt_a0_d[6], nxt_a0_d[7],
                sel_a0_p0, sel_a0_p1, pf_a0_p, pf_a1_p);
        }
#elif K_LOOP_SYNC_EVERY_2
        // R20C: barrier only on EVEN bt iters (every 2 iters). Compiler unrolls
        // and statically resolves the parity per unrolled copy.
        if ((bt & 1) == 0) {
            kpair_32mfma_with_lds_and_pf_swapped_sel<STEP3_PF_N, STEP3_EMBED_BARRIER>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw,
                nxt_a0_d[0], nxt_a0_d[1], nxt_a0_d[2], nxt_a0_d[3],
                nxt_a0_d[4], nxt_a0_d[5], nxt_a0_d[6], nxt_a0_d[7],
                sel_a0_p0, sel_a0_p1, pf_a0_p, pf_a1_p);
        } else {
            kpair_32mfma_with_lds_and_pf_swapped_sel<STEP3_PF_N, false>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw,
                nxt_a0_d[0], nxt_a0_d[1], nxt_a0_d[2], nxt_a0_d[3],
                nxt_a0_d[4], nxt_a0_d[5], nxt_a0_d[6], nxt_a0_d[7],
                sel_a0_p0, sel_a0_p1, pf_a0_p, pf_a1_p);
        }
#else
        kpair_32mfma_with_lds_and_pf_swapped_sel<STEP3_PF_N, STEP3_EMBED_BARRIER>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw,
            nxt_a0_d[0], nxt_a0_d[1], nxt_a0_d[2], nxt_a0_d[3],
            nxt_a0_d[4], nxt_a0_d[5], nxt_a0_d[6], nxt_a0_d[7],
            sel_a0_p0, sel_a0_p1, pf_a0_p, pf_a1_p);
#endif
        emit_pf_tail<STEP3_PF_N>(pf_a0_p, pf_a1_p);

#if DIRECT_BL && EARLY_BL_PF
        // Step4: pure MFMAs + Br prefetch only (Bl already in nxt_bl_d, no late vmem)
        {
            tile_pf_params dummy_pf = {};
            kpair_32mfma_with_pf_swapped_sel<PF_MPT>(acc_A1Br, tA1, tBr, a1_raw, br_raw,
                pf_br_p, dummy_pf);
        }
#elif DIRECT_BL
        kpair_32mfma_with_vmem_bl_wrap(acc_A1Br, tA1, tBr, a1_raw, br_raw,
            nxt_bl_d[0], nxt_bl_d[1], nxt_bl_d[2], nxt_bl_d[3],
            nxt_bl_d[4], nxt_bl_d[5], nxt_bl_d[6], nxt_bl_d[7],
            bl_voff_ps, srd_b_ps,
            bl_ps_n0_soff + static_cast<uint32_t>(bt + 1) * 2 * BL_K0_STRIDE);
        #pragma unroll
        for (int pi = 0; pi < PF_MPT; ++pi) emit_one_pf(pf_br_p, pi);
#else
        kpair_32mfma_with_lds_and_pf_swapped_sel<STEP4_PF_N>(acc_A1Br, tA1, tBr, a1_raw, br_raw,
            nxt_bl_d[0], nxt_bl_d[1], nxt_bl_d[2], nxt_bl_d[3],
            nxt_bl_d[4], nxt_bl_d[5], nxt_bl_d[6], nxt_bl_d[7],
            sel_bl_p0, sel_bl_p1, pf_bl_p, pf_br_p);
#if STEP4_EXTERNAL_BR_PREFETCH
        #pragma unroll
        for (int pi = 0; pi < PF_MPT; ++pi) emit_one_pf(pf_br_p, pi);
#else
        emit_pf_tail<STEP4_PF_N>(pf_bl_p, pf_br_p);
#endif
#endif // DIRECT_BL

#if DIRECT_BL
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
#else
        asm volatile("s_waitcnt lgkmcnt(0)");
#endif
        R41A_FENCE_BEFORE_EXTRACT();
        extract_tile(nxt_a0_d, tA0);
        R41A_FENCE_BEFORE_EXTRACT();
        extract_tile(nxt_bl_d, tBl);
        // R21B/R22C: opt-in scheduling hooks (no-op at defaults). Site 0.
        MXFP4_R22C_ITER_END_HOOK_0;

#if EARLY_SCALE_PF
        // Writeback shadow scales → pf_* (compiler inserts vmcnt as needed before next-iter use)
        #pragma unroll
        for (int p = 0; p < a_packs; ++p) { pf_a0[p] = nxt_pf_a0[p]; pf_a1[p] = nxt_pf_a1[p]; }
        #pragma unroll
        for (int p = 0; p < b_packs; ++p) { pf_bl[p] = nxt_pf_bl[p]; pf_br[p] = nxt_pf_br[p]; }
#endif
    }

    {
        const int bt = k_byte_iters - 1;
        const int cur = bt & 1;
        const uint32_t sel_br_p0 = cur ? br_1_p0 : br_0_p0;
        const uint32_t sel_br_p1 = cur ? br_1_p1 : br_0_p1;
        const uint32_t sel_a1_p0 = cur ? a1_1_p0 : a1_0_p0;
        const uint32_t sel_a1_p1 = cur ? a1_1_p1 : a1_0_p1;

        fp8e8m0_4 a0_raw[a_packs], a1_raw[a_packs], bl_raw[b_packs], br_raw[b_packs];
        #pragma unroll
        for (int p = 0; p < a_packs; ++p) { a0_raw[p] = pf_a0[p]; a1_raw[p] = pf_a1[p]; }
        #pragma unroll
        for (int p = 0; p < b_packs; ++p) { bl_raw[p] = pf_bl[p]; br_raw[p] = pf_br[p]; }

        float4 br_d[8], a1_d[8];
#if SWAP_STEP12_MAIN
        kpair_64mfma_step12_swapped_sel(acc_A0Bl, acc_A0Br, tA0, tBl,
            a0_raw, bl_raw, br_raw, br_d, a1_d,
            sel_br_p0, sel_br_p1, sel_a1_p0, sel_a1_p1);
#else
        kpair_64mfma_step12(acc_A0Bl, acc_A0Br, tA0, tBl,
            a0_raw, bl_raw, br_raw, br_d, a1_d,
            sel_br_p0, sel_br_p1, sel_a1_p0, sel_a1_p1);
#endif

        asm volatile("s_waitcnt lgkmcnt(0)");
        fp4_intx8_t tBr[4], tA1[4];
        extract_tile(br_d, tBr);
        extract_tile(a1_d, tA1);

        // Tail: always emit barrier (no embedded barrier in pure-MFMA Step3/4)
        // R19B: TAIL site _S1 (TAIL_SPLIT==1, live for parents using -DTAIL_SPLIT=1)
        asm volatile(MXFP4_TAIL_BARRIER_INST_S1 ::: "memory");

        tile_pf_params dummy_pf = {};
        kpair_32mfma_with_pf_swapped_sel<0>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw, dummy_pf, dummy_pf);
        kpair_32mfma_with_pf_swapped_sel<0>(acc_A1Br, tA1, tBr, a1_raw, br_raw, dummy_pf, dummy_pf);
    }
#else

#ifndef TAIL_SPLIT
#define TAIL_SPLIT 0
#endif

#if TAIL_SPLIT
    // ── Non-SWAP path: tail-split to eliminate wasted prefetch/LDS on last iter ──
    // Steady-state loop: bt = 0 .. k_byte_iters-2
    // Tail: bt = k_byte_iters-1 (no prefetch, no next-iter scale/LDS loads)
#ifdef UNROLL_K
  #if UNROLL_K == 0
    #pragma unroll
  #else
    #pragma unroll UNROLL_K
  #endif
#elif (K_DIM / 256) <= 16
    #pragma unroll
#elif (K_DIM / 256) <= 32
    #pragma unroll 16
#else
    #pragma unroll 8
#endif
    for (int bt = 0; bt + 1 < k_byte_iters; ++bt) {
        const int cur = bt & 1;
        const int nxt = 1 - cur;

#if R38F_TAIL_DRAIN && R37_FIX_B && !FUSED_STEP34
        // R38F Fix B4: drain in-flight buffer_load_to_lds loads at the TOP of
        // each iter that R25-C identified as a "tail" iter. Designed to be
        // combined with R38B (always-emit) — R38B closes the CRASH; R38F then
        // ensures the always-emitted prefetches drain before the next iter's
        // ds_reads consume them, fixing the stale-LDS-read WRONG_OUTPUT.
        // R25C_ACTIVE folds at compile-time when K-loop fully unrolls.
#if R25C_ACTIVE
        const bool _r38f_in_tail = (bt >= k_byte_iters - 1 - R25C_TAIL_PF_OFF_ITERS);
        if (_r38f_in_tail) {
#if R38F_VARIANT == 1
            asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
#elif R38F_VARIANT == 2
            asm volatile("s_waitcnt vmcnt(0)\ns_barrier\n" ::: "memory");
#elif R38F_VARIANT == 3
            asm volatile("s_waitcnt 0\n" ::: "memory");
#elif R38F_VARIANT == 4
            asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n" ::: "memory");
#else
#error "R38F_VARIANT must be 1, 2, 3, or 4"
#endif
        }
#endif // R25C_ACTIVE
#endif // R38F_TAIL_DRAIN

        const uint32_t sel_br_p0 = cur ? br_1_p0 : br_0_p0;
        const uint32_t sel_br_p1 = cur ? br_1_p1 : br_0_p1;
        const uint32_t sel_a1_p0 = cur ? a1_1_p0 : a1_0_p0;
        const uint32_t sel_a1_p1 = cur ? a1_1_p1 : a1_0_p1;
        const uint32_t sel_a0_p0 = nxt ? a0_1_p0 : a0_0_p0;
        const uint32_t sel_a0_p1 = nxt ? a0_1_p1 : a0_0_p1;
#if !DIRECT_BL
        const uint32_t sel_bl_p0 = nxt ? bl_1_p0 : bl_0_p0;
        const uint32_t sel_bl_p1 = nxt ? bl_1_p1 : bl_0_p1;
#endif

        const int pf_bt = (bt + 2 < k_byte_iters) ? (bt + 2) : (k_byte_iters - 1);
#if R38B_TAIL_FIX && R37_FIX_B && !FUSED_STEP34
        // R38B: defer make_pf_params construction to inside the no-tail
        // gate (see R37_FIX_B branch below). This eliminates ~128 bytes/iter
        // of struct construction (and the resulting VGPR/scratch pressure)
        // on tail iters, fixing intermittent HSA aperture violations on the
        // 9 R37 CRASH shapes.
#elif R40A_PF_FENCE && R37_FIX_B && !FUSED_STEP34
        // R40A: defer make_pf_params construction until AFTER kpair_64mfma_step34
        // returns (see R37_FIX_B branch below). Combined with the explicit
        // pre-step12 `asm volatile` fence, this prevents the compiler from
        // hoisting the per-iter buffer_load_to_lds prefetches across the
        // step12 asm boundary.
#else
        tile_pf_params pf_a0_p = make_pf_params(A0_db[cur], g.a, coord<ST_tile>(0,0,br*2,     pf_bt), so_a, srd_a, base_a, lb_a0[cur], R22B_A_HINT_VAL);
        tile_pf_params pf_a1_p = make_pf_params(A1_db[cur], g.a, coord<ST_tile>(0,0,br*2+1,   pf_bt), so_a, srd_a, base_a, lb_a1[cur], R22B_A_HINT_VAL);
#if !DIRECT_BL
        tile_pf_params pf_bl_p = make_pf_params(Bl_db[cur], g.b, coord<ST_tile>(0,0,bc*2,     pf_bt), so_b, srd_b, base_b, lb_bl[cur], R22B_B_HINT_VAL);
#endif
        tile_pf_params pf_br_p = make_pf_params(Br_db[cur], g.b, coord<ST_tile>(0,0,bc*2+1,   pf_bt), so_b, srd_b, base_b, lb_br[cur], R22B_B_HINT_VAL);
#endif // R38B_TAIL_FIX / R40A_PF_FENCE

#if OUTER_K_PF_DEPTH > 1 && OUTER_K_PF_MODE == 1
        // R24C: pull-forward L2-only prefetch for K+OUTER_K_PF_DEPTH (>=K+3 for depth=2).
        // We rebuild pf params at pf_bt+(OUTER_K_PF_DEPTH-1) clamped to k_byte_iters-1
        // and issue buffer_load_dwordx4 that lands in L2/L1 only (no LDS write).
        // The LDS slot we encode does not matter — the load result is discarded.
        {
            const int pf2_bt = (pf_bt + (OUTER_K_PF_DEPTH - 1) < k_byte_iters)
                              ? (pf_bt + (OUTER_K_PF_DEPTH - 1))
                              : (k_byte_iters - 1);
            tile_pf_params pf2_a0 = make_pf_params(A0_db[cur], g.a, coord<ST_tile>(0,0,br*2,     pf2_bt), so_a, srd_a, base_a, lb_a0[cur], R22B_A_HINT_VAL);
            tile_pf_params pf2_a1 = make_pf_params(A1_db[cur], g.a, coord<ST_tile>(0,0,br*2+1,   pf2_bt), so_a, srd_a, base_a, lb_a1[cur], R22B_A_HINT_VAL);
            tile_pf_params pf2_br = make_pf_params(Br_db[cur], g.b, coord<ST_tile>(0,0,bc*2+1,   pf2_bt), so_b, srd_b, base_b, lb_br[cur], R22B_B_HINT_VAL);
            // Issue all PF_MPT loads per tile for A0/A1/Br. Skip Bl when DIRECT_BL.
            emit_full_pf_l2only<PF_MPT>(pf2_a0);
            emit_full_pf_l2only<PF_MPT>(pf2_a1);
            emit_full_pf_l2only<PF_MPT>(pf2_br);
#if !DIRECT_BL
            tile_pf_params pf2_bl = make_pf_params(Bl_db[cur], g.b, coord<ST_tile>(0,0,bc*2,     pf2_bt), so_b, srd_b, base_b, lb_bl[cur], R22B_B_HINT_VAL);
            emit_full_pf_l2only<PF_MPT>(pf2_bl);
#endif
        }
#endif

#if (L2_PF_A > 0) || (L2_PF_B > 0)
        // R24B: L2-only prefetch one extra outer-K iter ahead of the LDS prefetch.
        // pf_bt = bt+2 (LDS-going); pf3_bt = bt+3 (cache-warming only).
        // Result is discarded by emit_one_pf_l2only (no LDS write, no live VGPR).
        {
            const int pf3_bt = (pf_bt + 1 < k_byte_iters) ? (pf_bt + 1) : (k_byte_iters - 1);
#if (L2_PF_A > 0)
            tile_pf_params pf3_a0 = make_pf_params(A0_db[cur], g.a, coord<ST_tile>(0,0,br*2,     pf3_bt), so_a, srd_a, base_a, lb_a0[cur], R22B_A_HINT_VAL);
            tile_pf_params pf3_a1 = make_pf_params(A1_db[cur], g.a, coord<ST_tile>(0,0,br*2+1,   pf3_bt), so_a, srd_a, base_a, lb_a1[cur], R22B_A_HINT_VAL);
            emit_l2_pf_block<L2_PF_A>(pf3_a0);
            emit_l2_pf_block<L2_PF_A>(pf3_a1);
#endif
#if (L2_PF_B > 0)
            tile_pf_params pf3_br = make_pf_params(Br_db[cur], g.b, coord<ST_tile>(0,0,bc*2+1,   pf3_bt), so_b, srd_b, base_b, lb_br[cur], R22B_B_HINT_VAL);
            emit_l2_pf_block<L2_PF_B>(pf3_br);
#if !DIRECT_BL
            tile_pf_params pf3_bl = make_pf_params(Bl_db[cur], g.b, coord<ST_tile>(0,0,bc*2,     pf3_bt), so_b, srd_b, base_b, lb_bl[cur], R22B_B_HINT_VAL);
            emit_l2_pf_block<L2_PF_B>(pf3_bl);
#endif
#endif
        }
#endif

        // R39A: skip the scale-pf advance on R25-C tail iters where the data
        // prefetch is suppressed. Without this, the next iter's MFMA reads STALE
        // data (last fetched tile, frozen by R25-C) but a FRESH scale (advancing
        // past the data), causing scale-vs-data mismatch and BF16-overflow garbage.
        //
        // Variants (compile-time):
        //   R39A_VARIANT=0 (default): freeze scale loads entirely on tail iters
        //                              (skip the load, keep previous pf_* values).
        //   R39A_VARIANT=1:           clamp scale index to (k_byte_iters - 1 -
        //                              R25C_TAIL_PF_OFF_ITERS) on tail iters
        //                              (re-load same scale every tail iter).
        //   R39A_VARIANT=2:           clamp to bt (one-back of natural bt+1).
        //
        // When R25-C is inactive (default tail_off=0) or R39A=0, the natural
        // `bt+1` index is preserved (no behavior change on R37 WIN shapes).
#ifndef R39A_VARIANT
#define R39A_VARIANT 0
#endif
#if R39A_TAIL_SCALE_CLAMP && R37_FIX_B && !FUSED_STEP34 && R25C_ACTIVE
        const bool _r39a_in_tail = (bt >= k_byte_iters - 1 - R25C_TAIL_PF_OFF_ITERS);
#else
        constexpr bool _r39a_in_tail = false;
#endif
#if R39A_VARIANT == 1
        const uint32_t _r39a_scale_idx = _r39a_in_tail
            ? static_cast<uint32_t>(k_byte_iters - 1 - R25C_TAIL_PF_OFF_ITERS)
            : static_cast<uint32_t>(bt + 1);
#elif R39A_VARIANT == 2
        const uint32_t _r39a_scale_idx = _r39a_in_tail ? static_cast<uint32_t>(bt) : static_cast<uint32_t>(bt + 1);
#else
        const uint32_t _r39a_scale_idx = static_cast<uint32_t>(bt + 1);
#endif

#if EARLY_SCALE_PF
        fp8e8m0_4 nxt_pf_a0[a_packs], nxt_pf_a1[a_packs], nxt_pf_bl[b_packs], nxt_pf_br[b_packs];
#if R39A_VARIANT == 0
        if (!_r39a_in_tail) {
            const uint32_t nxt_scale = static_cast<uint32_t>(bt + 1) << 9;
            load_pq_scale_x2_async(a0_srd, lane_soff_x2, nxt_scale, nxt_pf_a0[0], nxt_pf_a0[1]);
            load_pq_scale_x2_async(a1_srd, lane_soff_x2, nxt_scale, nxt_pf_a1[0], nxt_pf_a1[1]);
            load_pq_scale_x2_async(bl_srd, lane_soff_x2, nxt_scale, nxt_pf_bl[0], nxt_pf_bl[1]);
            load_pq_scale_x2_async(br_srd, lane_soff_x2, nxt_scale, nxt_pf_br[0], nxt_pf_br[1]);
        } else {
            // Carry forward existing pf_* (treated as nxt_pf_* shadow)
            #pragma unroll
            for (int p = 0; p < a_packs; ++p) { nxt_pf_a0[p] = pf_a0[p]; nxt_pf_a1[p] = pf_a1[p]; }
            #pragma unroll
            for (int p = 0; p < b_packs; ++p) { nxt_pf_bl[p] = pf_bl[p]; nxt_pf_br[p] = pf_br[p]; }
        }
#else
        {
            const uint32_t nxt_scale = _r39a_scale_idx << 9;
            load_pq_scale_x2_async(a0_srd, lane_soff_x2, nxt_scale, nxt_pf_a0[0], nxt_pf_a0[1]);
            load_pq_scale_x2_async(a1_srd, lane_soff_x2, nxt_scale, nxt_pf_a1[0], nxt_pf_a1[1]);
            load_pq_scale_x2_async(bl_srd, lane_soff_x2, nxt_scale, nxt_pf_bl[0], nxt_pf_bl[1]);
            load_pq_scale_x2_async(br_srd, lane_soff_x2, nxt_scale, nxt_pf_br[0], nxt_pf_br[1]);
        }
#endif // R39A_VARIANT == 0
#endif

        fp8e8m0_4 a0_raw[a_packs], a1_raw[a_packs], bl_raw[b_packs], br_raw[b_packs];
        #pragma unroll
        for (int p = 0; p < a_packs; ++p) { a0_raw[p] = pf_a0[p]; a1_raw[p] = pf_a1[p]; }
        #pragma unroll
        for (int p = 0; p < b_packs; ++p) { bl_raw[p] = pf_bl[p]; br_raw[p] = pf_br[p]; }

#if !EARLY_SCALE_PF
#if R39A_VARIANT == 0
        if (!_r39a_in_tail) {
            const uint32_t nxt_scale = static_cast<uint32_t>(bt + 1) << 9;
            load_pq_scale_x2_async(a0_srd, lane_soff_x2, nxt_scale, pf_a0[0], pf_a0[1]);
            load_pq_scale_x2_async(a1_srd, lane_soff_x2, nxt_scale, pf_a1[0], pf_a1[1]);
            load_pq_scale_x2_async(bl_srd, lane_soff_x2, nxt_scale, pf_bl[0], pf_bl[1]);
            load_pq_scale_x2_async(br_srd, lane_soff_x2, nxt_scale, pf_br[0], pf_br[1]);
        }
        // else: leave pf_* alone (frozen scale)
#else
        {
            const uint32_t nxt_scale = _r39a_scale_idx << 9;
            load_pq_scale_x2_async(a0_srd, lane_soff_x2, nxt_scale, pf_a0[0], pf_a0[1]);
            load_pq_scale_x2_async(a1_srd, lane_soff_x2, nxt_scale, pf_a1[0], pf_a1[1]);
            load_pq_scale_x2_async(bl_srd, lane_soff_x2, nxt_scale, pf_bl[0], pf_bl[1]);
            load_pq_scale_x2_async(br_srd, lane_soff_x2, nxt_scale, pf_br[0], pf_br[1]);
        }
#endif // R39A_VARIANT == 0
#endif

        // Steps 1+2 merged: A0*Bl (32 MFMAs) + ds_read Br + A0*Br (32 MFMAs) + ds_read A1
        float4 br_d[8], a1_d[8];
#if R40A_PF_FENCE && R37_FIX_B && !FUSED_STEP34
        // R40A: pre-step12 fence — prevent the compiler from hoisting any
        // upcoming buffer_load_to_lds prefetches (constructed AFTER step34)
        // across the step12 asm boundary. Combined with the deferred
        // make_pf_params (see top-of-iter block) this guarantees the LDS
        // prefetch issue cannot land in a slot mid-MFMA-read.
        asm volatile("" ::: "memory");
#endif
        kpair_64mfma_step12(acc_A0Bl, acc_A0Br, tA0, tBl,
            a0_raw, bl_raw, br_raw, br_d, a1_d,
            sel_br_p0, sel_br_p1, sel_a1_p0, sel_a1_p1);

        asm volatile("s_waitcnt lgkmcnt(0)");
        fp4_intx8_t tBr[4], tA1[4];
        extract_tile(br_d, tBr);
        extract_tile(a1_d, tA1);

#if FUSED_STEP34
        // Fused Step34: barrier + 64 MFMAs + 16 ds_reads in one asm block
        float4 nxt_a0_d[8];
        float4 nxt_bl_d[8];
        kpair_64mfma_step34(acc_A1Bl, acc_A1Br, tA1, tBl, tBr,
            a1_raw, bl_raw, br_raw, nxt_a0_d, nxt_bl_d,
            sel_a0_p0, sel_a0_p1, sel_bl_p0, sel_bl_p1);
#if R43A_GATE_PF_TAIL_KBOUND
        // R43 Opt A.fix1 — at K=28672 (k_byte_iters=112), the FUSED_STEP34+TAIL_SPLIT
        // conjunction triggers HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION at runtime.
        // R42 Opt B Phase-2A localized the CRASH to FUSED_STEP34=1 AND TAIL_SPLIT=1
        // (stripping either knob alone removes the CRASH but exposes a separate
        // 17%-bf16-overflow correctness bug — see R42_OPT_B_VERDICT.md).
        //
        // Variants (compile-time R43A_GATE_PF_TAIL_KBOUND value):
        //   1 = skip BOTH emit_pf_tail on bt == k_byte_iters-2 (kills A & B halves)
        //   2 = skip only B-half on the clamped iter
        //   3 = skip only A-half on the clamped iter
        //   4 = REPLACE the data prefetches with L2-only (cache-warming, no LDS race)
        //   5 = R38B-style: build pf_*_p HERE (after step34) instead of top-of-iter,
        //       then emit_pf_tail. Mirrors R37_FIX_B+R38B_TAIL_FIX inside FUSED_STEP34.
        //   6 = pre-emit-pf vmcnt fence: insert s_waitcnt vmcnt(0) BEFORE emit_pf_tail
        //       to drain in-flight buffer_load_to_lds from prior iters before issuing
        //       new ones (closes write-after-write race on shared LDS double-buffer).
        const bool _r43a_skip = (bt >= k_byte_iters - 2);
#if R43A_GATE_PF_TAIL_KBOUND == 1
        if (!_r43a_skip) {
            emit_pf_tail<0>(pf_a0_p, pf_a1_p);
            emit_pf_tail<0>(pf_bl_p, pf_br_p);
        }
#elif R43A_GATE_PF_TAIL_KBOUND == 2
        emit_pf_tail<0>(pf_a0_p, pf_a1_p);
        if (!_r43a_skip) {
            emit_pf_tail<0>(pf_bl_p, pf_br_p);
        }
#elif R43A_GATE_PF_TAIL_KBOUND == 3
        if (!_r43a_skip) {
            emit_pf_tail<0>(pf_a0_p, pf_a1_p);
        }
        emit_pf_tail<0>(pf_bl_p, pf_br_p);
#elif R43A_GATE_PF_TAIL_KBOUND == 4
        if (!_r43a_skip) {
            emit_pf_tail<0>(pf_a0_p, pf_a1_p);
            emit_pf_tail<0>(pf_bl_p, pf_br_p);
        } else {
            emit_full_pf_l2only<PF_MPT>(pf_a0_p);
            emit_full_pf_l2only<PF_MPT>(pf_a1_p);
            emit_full_pf_l2only<PF_MPT>(pf_bl_p);
            emit_full_pf_l2only<PF_MPT>(pf_br_p);
        }
#elif R43A_GATE_PF_TAIL_KBOUND == 5
        // R43 fix1 variant 5: defer make_pf_params construction to here (after step34)
        // so the live-range of the pf_*_p structs spans only the emit_pf_tail emission,
        // mirroring the R38B_TAIL_FIX pattern in the R37_FIX_B (non-FUSED) branch.
        // The original (top-of-iter) pf_*_p go unused on the FUSED path with this
        // variant — the compiler should DCE them. Suppress unused-warning only.
        {
            tile_pf_params pf_a0_p_late = make_pf_params(A0_db[cur], g.a, coord<ST_tile>(0,0,br*2,     pf_bt), so_a, srd_a, base_a, lb_a0[cur], R22B_A_HINT_VAL);
            tile_pf_params pf_a1_p_late = make_pf_params(A1_db[cur], g.a, coord<ST_tile>(0,0,br*2+1,   pf_bt), so_a, srd_a, base_a, lb_a1[cur], R22B_A_HINT_VAL);
#if !DIRECT_BL
            tile_pf_params pf_bl_p_late = make_pf_params(Bl_db[cur], g.b, coord<ST_tile>(0,0,bc*2,     pf_bt), so_b, srd_b, base_b, lb_bl[cur], R22B_B_HINT_VAL);
#endif
            tile_pf_params pf_br_p_late = make_pf_params(Br_db[cur], g.b, coord<ST_tile>(0,0,bc*2+1,   pf_bt), so_b, srd_b, base_b, lb_br[cur], R22B_B_HINT_VAL);
            emit_pf_tail<0>(pf_a0_p_late, pf_a1_p_late);
            emit_pf_tail<0>(pf_bl_p_late, pf_br_p_late);
        }
        (void)pf_a0_p; (void)pf_a1_p;
#if !DIRECT_BL
        (void)pf_bl_p;
#endif
        (void)pf_br_p;
#elif R43A_GATE_PF_TAIL_KBOUND == 6
        // Variant 6: vmcnt(0) fence BEFORE the unconditional emit_pf_tail to ensure
        // all prior buffer_load_to_lds have drained before issuing new ones.
        asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
        emit_pf_tail<0>(pf_a0_p, pf_a1_p);
        emit_pf_tail<0>(pf_bl_p, pf_br_p);
#else
#error "R43A_GATE_PF_TAIL_KBOUND must be 0, 1, 2, 3, 4, 5, or 6"
#endif
        (void)_r43a_skip;
#else
        emit_pf_tail<0>(pf_a0_p, pf_a1_p);
        emit_pf_tail<0>(pf_bl_p, pf_br_p);
#endif // R43A_GATE_PF_TAIL_KBOUND
#elif R37_FIX_B
        // R37 Fix B (default): use fused step3+step4 (correctness fix) while
        // preserving R25-C tail-pf-off + STEP4_EXTERNAL_BR_PREFETCH branching
        // that the FUSED_STEP34=1 path otherwise bypasses.
        float4 nxt_a0_d[8];
        float4 nxt_bl_d[8];
        kpair_64mfma_step34(acc_A1Bl, acc_A1Br, tA1, tBl, tBr,
            a1_raw, bl_raw, br_raw, nxt_a0_d, nxt_bl_d,
            sel_a0_p0, sel_a0_p1, sel_bl_p0, sel_bl_p1);
        // R37: fence the scheduler — `-mllvm -amdgpu-sched-strategy=max-memory-clause`
        // is otherwise free to hoist the upcoming buffer_load_to_lds prefetches across
        // iteration boundaries (these intrinsics aren't asm volatile), which corrupts
        // the LDS double-buffer state because the next iter's s_barrier vmcnt count
        // is now wrong. Plain memory clobber asm volatile prevents the hoist.
        asm volatile("" ::: "memory");
#if R40A_PF_FENCE && !R38B_TAIL_FIX
        // R40A: build pf_*_p AFTER step34 (deferred from top-of-iter). With the
        // pre-step12 fence above, the compiler cannot hoist the constructed
        // emit_pf_tail loads back across step12. The construction itself is
        // free of memory side effects; only the subsequent emit_pf_tail issues
        // the buffer_load_to_lds intrinsics.
        tile_pf_params pf_a0_p = make_pf_params(A0_db[cur], g.a, coord<ST_tile>(0,0,br*2,     pf_bt), so_a, srd_a, base_a, lb_a0[cur], R22B_A_HINT_VAL);
        tile_pf_params pf_a1_p = make_pf_params(A1_db[cur], g.a, coord<ST_tile>(0,0,br*2+1,   pf_bt), so_a, srd_a, base_a, lb_a1[cur], R22B_A_HINT_VAL);
#if !DIRECT_BL
        tile_pf_params pf_bl_p = make_pf_params(Bl_db[cur], g.b, coord<ST_tile>(0,0,bc*2,     pf_bt), so_b, srd_b, base_b, lb_bl[cur], R22B_B_HINT_VAL);
#endif
        tile_pf_params pf_br_p = make_pf_params(Br_db[cur], g.b, coord<ST_tile>(0,0,bc*2+1,   pf_bt), so_b, srd_b, base_b, lb_br[cur], R22B_B_HINT_VAL);
#endif // R40A_PF_FENCE && !R38B_TAIL_FIX

        // R25-C: in the last R25C_TAIL_PF_OFF_ITERS iters, drop global prefetch.
        // Branch folds to compile-time when K-loop fully unrolls (R25C_ACTIVE
        // gates K_DIM ≤ R25C_K_LIMIT); becomes constexpr false otherwise.
#if R25C_ACTIVE
        const bool _r25c_tail_no_pf = (bt >= k_byte_iters - 1 - R25C_TAIL_PF_OFF_ITERS);
#else
        constexpr bool _r25c_tail_no_pf = false;
#endif
#if R38B_TAIL_FIX
        // R38B: ALWAYS emit prefetches, even when R25C says "skip". Empirical
        // finding: in the R37 fix-B path, the runtime `if (!_r25c_tail_no_pf)`
        // branch around emit_pf_tail (combined with the per-iter struct
        // construction of pf_*_p) leaves the compiler-generated vmcnt /
        // s_barrier counts inconsistent across the tail iters, producing
        // intermittent HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION on stress.
        // Confirmed: setting R25C_TAIL_PF_OFF_ITERS=0 (always-emit) eliminates
        // the crash; this macro provides the same effect WITHOUT having to
        // strip the per-shape pfoff flag from every CRASH variant. The pf
        // targets the clamped pf_bt = k_byte_iters - 1 on tail iters — same
        // L2/LDS line we already loaded — so the perf cost is bounded
        // (≤ R25C_TAIL_PF_OFF_ITERS / k_byte_iters extra wasted-bandwidth iters).
        // Build the pf params HERE (deferred from loop top) so the construction
        // cost is paid only inside the always-emit branch (no behavior change
        // versus the original top-of-loop construction, just placement).
        {
            tile_pf_params pf_a0_p = make_pf_params(A0_db[cur], g.a, coord<ST_tile>(0,0,br*2,     pf_bt), so_a, srd_a, base_a, lb_a0[cur], R22B_A_HINT_VAL);
            tile_pf_params pf_a1_p = make_pf_params(A1_db[cur], g.a, coord<ST_tile>(0,0,br*2+1,   pf_bt), so_a, srd_a, base_a, lb_a1[cur], R22B_A_HINT_VAL);
#if !DIRECT_BL
            tile_pf_params pf_bl_p = make_pf_params(Bl_db[cur], g.b, coord<ST_tile>(0,0,bc*2,     pf_bt), so_b, srd_b, base_b, lb_bl[cur], R22B_B_HINT_VAL);
#endif
            tile_pf_params pf_br_p = make_pf_params(Br_db[cur], g.b, coord<ST_tile>(0,0,bc*2+1,   pf_bt), so_b, srd_b, base_b, lb_br[cur], R22B_B_HINT_VAL);
            emit_pf_tail<0>(pf_a0_p, pf_a1_p);
            emit_pf_tail<0>(pf_bl_p, pf_br_p);
        }
        (void)_r25c_tail_no_pf;  // suppress unused-variable warning
#else
        if (!_r25c_tail_no_pf) {
            // Issue full A0 + A1 prefetches (would have come from STEP3_PF_N).
            emit_pf_tail<0>(pf_a0_p, pf_a1_p);
            // Issue Bl + Br prefetches (would have come from STEP4_PF_N).
            // STEP4_EXTERNAL_BR_PREFETCH=1 reorders Br to fire as a separate
            // emit_one_pf burst after Bl; semantically the SAME loads — so we
            // emit both pf groups here uniformly.
            emit_pf_tail<0>(pf_bl_p, pf_br_p);
        }
#if R38C_TAIL_L2ONLY
        // R38C Fix B3: when R25-C says "skip" the tail prefetches, instead of
        // dropping them entirely (which leaves the compiler-tracked vmcnt out
        // of sync with the in-flight buffer-load-to-LDS state and causes the
        // next iter's s_barrier to fire while a load is still landing into a
        // double-buffer slot about to be reallocated → CRASH), route the SAME
        // GMEM addresses through `emit_full_pf_l2only<PF_MPT>`. That issues
        // identical buffer_load_dwordx4 instructions but discards the result
        // into a scratch VGPR (no `lds:1` modifier → no LDS write → no
        // double-buffer slot collision). The data lands in L2/L1 and warms
        // the cache for the prologue/epilogue Bl direct-load. Crucially the
        // compiler-tracked vmcnt now stays consistent with the iter's
        // scheduler footprint (the loads are emitted as inline-asm volatile
        // with a "memory" clobber, so they cannot be reordered or DCE'd).
        if (_r25c_tail_no_pf) {
            emit_full_pf_l2only<PF_MPT>(pf_a0_p);
            emit_full_pf_l2only<PF_MPT>(pf_a1_p);
#if !DIRECT_BL
            emit_full_pf_l2only<PF_MPT>(pf_bl_p);
#endif
            emit_full_pf_l2only<PF_MPT>(pf_br_p);
        }
#endif // R38C_TAIL_L2ONLY
#endif // R38B_TAIL_FIX
        // R37: fence again post-prefetch so the next iter's barrier vmcnt is correct.
        asm volatile("" ::: "memory");
#else // R37_FIX_B == 0 → legacy buggy non-fused step3+step4 path
#if !STEP3_EMBED_BARRIER
        // R19B: site _S6 (TAIL_SPLIT outer !STEP3_EMBED_BARRIER, dead w/ default STEP3_EMBED_BARRIER=1)
        asm volatile(MXFP4_STEP3_BARRIER_INST_S6 ::: "memory");
#endif

        // Step 3: A1*Bl (32 MFMAs) + ds_read A0[nxt] + prefetch
        float4 nxt_a0_d[8];
        float4 nxt_bl_d[8];

        // R25-C: in the last R25C_TAIL_PF_OFF_ITERS iters, drop global prefetch
        // (PF_N=0). Branch folds to compile-time when the K-loop fully unrolls
        // (R25C_ACTIVE gates K_DIM ≤ R25C_K_LIMIT); becomes constexpr false
        // otherwise.
#if R25C_ACTIVE
        const bool _r25c_tail_no_pf = (bt >= k_byte_iters - 1 - R25C_TAIL_PF_OFF_ITERS);
#else
        constexpr bool _r25c_tail_no_pf = false;
#endif

        if (_r25c_tail_no_pf) {
#if SPREAD_LDS
            kpair_32mfma_with_lds_rowspread_pf<0, STEP3_EMBED_BARRIER>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw,
#else
            kpair_32mfma_with_lds_and_pf<0, STEP3_EMBED_BARRIER>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw,
#endif
                nxt_a0_d[0], nxt_a0_d[1], nxt_a0_d[2], nxt_a0_d[3],
                nxt_a0_d[4], nxt_a0_d[5], nxt_a0_d[6], nxt_a0_d[7],
                sel_a0_p0, sel_a0_p1, pf_a0_p, pf_a1_p);
            // emit_pf_tail<0> is a no-op
        } else {
#if K_LOOP_SYNC_EVERY_4 || K_LOOP_SYNC_EVERY_2
        // R20C: barrier coarsening — emit barrier only every 2 (or 4) iters.
        // Compiler unrolls the K-loop and statically resolves the parity per copy.
#if K_LOOP_SYNC_EVERY_4
        const bool _r20c_emit_barrier = ((bt & 3) == 0);
#else
        const bool _r20c_emit_barrier = ((bt & 1) == 0);
#endif
        if (_r20c_emit_barrier) {
#if SPREAD_LDS
            kpair_32mfma_with_lds_rowspread_pf<STEP3_PF_N, STEP3_EMBED_BARRIER>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw,
#else
            kpair_32mfma_with_lds_and_pf<STEP3_PF_N, STEP3_EMBED_BARRIER>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw,
#endif
                nxt_a0_d[0], nxt_a0_d[1], nxt_a0_d[2], nxt_a0_d[3],
                nxt_a0_d[4], nxt_a0_d[5], nxt_a0_d[6], nxt_a0_d[7],
                sel_a0_p0, sel_a0_p1, pf_a0_p, pf_a1_p);
        } else {
#if SPREAD_LDS
            kpair_32mfma_with_lds_rowspread_pf<STEP3_PF_N, false>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw,
#else
            kpair_32mfma_with_lds_and_pf<STEP3_PF_N, false>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw,
#endif
                nxt_a0_d[0], nxt_a0_d[1], nxt_a0_d[2], nxt_a0_d[3],
                nxt_a0_d[4], nxt_a0_d[5], nxt_a0_d[6], nxt_a0_d[7],
                sel_a0_p0, sel_a0_p1, pf_a0_p, pf_a1_p);
        }
#else
#if SPREAD_LDS
        kpair_32mfma_with_lds_rowspread_pf<STEP3_PF_N, STEP3_EMBED_BARRIER>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw,
#else
        kpair_32mfma_with_lds_and_pf<STEP3_PF_N, STEP3_EMBED_BARRIER>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw,
#endif
            nxt_a0_d[0], nxt_a0_d[1], nxt_a0_d[2], nxt_a0_d[3],
            nxt_a0_d[4], nxt_a0_d[5], nxt_a0_d[6], nxt_a0_d[7],
            sel_a0_p0, sel_a0_p1, pf_a0_p, pf_a1_p);
#endif // K_LOOP_SYNC_EVERY_*
        emit_pf_tail<STEP3_PF_N>(pf_a0_p, pf_a1_p);
        } // end !_r25c_tail_no_pf (Step3)

        // Step 4: A1*Br (32 MFMAs) + load next Bl
#if DIRECT_BL
        kpair_32mfma_with_vmem_bl_wrap(acc_A1Br, tA1, tBr, a1_raw, br_raw,
            nxt_bl_d[0], nxt_bl_d[1], nxt_bl_d[2], nxt_bl_d[3],
            nxt_bl_d[4], nxt_bl_d[5], nxt_bl_d[6], nxt_bl_d[7],
            bl_voff_ps, srd_b_ps,
            bl_ps_n0_soff + static_cast<uint32_t>(bt + 1) * 2 * BL_K0_STRIDE);
        #pragma unroll
        for (int pi = 0; pi < PF_MPT; ++pi) emit_one_pf(pf_br_p, pi);
#else
        if (_r25c_tail_no_pf) {
#if SPREAD_LDS
            kpair_32mfma_with_lds_rowspread_pf<0>(acc_A1Br, tA1, tBr, a1_raw, br_raw,
#else
            kpair_32mfma_with_lds_and_pf<0>(acc_A1Br, tA1, tBr, a1_raw, br_raw,
#endif
                nxt_bl_d[0], nxt_bl_d[1], nxt_bl_d[2], nxt_bl_d[3],
                nxt_bl_d[4], nxt_bl_d[5], nxt_bl_d[6], nxt_bl_d[7],
                sel_bl_p0, sel_bl_p1, pf_bl_p, pf_br_p);
            // emit_pf_tail<0> and external Br pf are no-ops on tail-no-pf
        } else {
#if SPREAD_LDS
        kpair_32mfma_with_lds_rowspread_pf<STEP4_PF_N>(acc_A1Br, tA1, tBr, a1_raw, br_raw,
#else
        kpair_32mfma_with_lds_and_pf<STEP4_PF_N>(acc_A1Br, tA1, tBr, a1_raw, br_raw,
#endif
            nxt_bl_d[0], nxt_bl_d[1], nxt_bl_d[2], nxt_bl_d[3],
            nxt_bl_d[4], nxt_bl_d[5], nxt_bl_d[6], nxt_bl_d[7],
            sel_bl_p0, sel_bl_p1, pf_bl_p, pf_br_p);
#if STEP4_EXTERNAL_BR_PREFETCH
        #pragma unroll
        for (int pi = 0; pi < PF_MPT; ++pi) emit_one_pf(pf_br_p, pi);
#else
        emit_pf_tail<STEP4_PF_N>(pf_bl_p, pf_br_p);
#endif
        } // end !_r25c_tail_no_pf (Step4)
#endif // DIRECT_BL
#endif // FUSED_STEP34 / R37_FIX_B / legacy

#if DIRECT_BL
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
#else
        asm volatile("s_waitcnt lgkmcnt(0)");
#endif
        R41A_FENCE_BEFORE_EXTRACT();
        extract_tile(nxt_a0_d, tA0);
        R41A_FENCE_BEFORE_EXTRACT();
        extract_tile(nxt_bl_d, tBl);
        // R21B/R22C: opt-in scheduling hooks (no-op at defaults). Site 1.
        MXFP4_R22C_ITER_END_HOOK_1;

#if EARLY_SCALE_PF
        #pragma unroll
        for (int p = 0; p < a_packs; ++p) { pf_a0[p] = nxt_pf_a0[p]; pf_a1[p] = nxt_pf_a1[p]; }
        #pragma unroll
        for (int p = 0; p < b_packs; ++p) { pf_bl[p] = nxt_pf_bl[p]; pf_br[p] = nxt_pf_br[p]; }
#endif
    }

    // ── Tail iteration: no prefetch, no next-iter scale/LDS loads ──
    {
        const int bt = k_byte_iters - 1;
        const int cur = bt & 1;
        const uint32_t sel_br_p0 = cur ? br_1_p0 : br_0_p0;
        const uint32_t sel_br_p1 = cur ? br_1_p1 : br_0_p1;
        const uint32_t sel_a1_p0 = cur ? a1_1_p0 : a1_0_p0;
        const uint32_t sel_a1_p1 = cur ? a1_1_p1 : a1_0_p1;

        fp8e8m0_4 a0_raw[a_packs], a1_raw[a_packs], bl_raw[b_packs], br_raw[b_packs];
        #pragma unroll
        for (int p = 0; p < a_packs; ++p) { a0_raw[p] = pf_a0[p]; a1_raw[p] = pf_a1[p]; }
        #pragma unroll
        for (int p = 0; p < b_packs; ++p) { bl_raw[p] = pf_bl[p]; br_raw[p] = pf_br[p]; }

        // Steps 1+2: still need Br and A1 from current LDS
        float4 br_d[8], a1_d[8];
        kpair_64mfma_step12(acc_A0Bl, acc_A0Br, tA0, tBl,
            a0_raw, bl_raw, br_raw, br_d, a1_d,
            sel_br_p0, sel_br_p1, sel_a1_p0, sel_a1_p1);

        asm volatile("s_waitcnt lgkmcnt(0)");
        fp4_intx8_t tBr[4], tA1[4];
        extract_tile(br_d, tBr);
        extract_tile(a1_d, tA1);

        // Tail: always emit barrier (no embedded barrier in pure-MFMA Step3/4)
        // R19B: TAIL site _S2 (TAIL_SPLIT==0, dead for parents using -DTAIL_SPLIT=1)
        asm volatile(MXFP4_TAIL_BARRIER_INST_S2 ::: "memory");

        // Steps 3+4: pure MFMAs, no ds_reads, no prefetches
        tile_pf_params dummy_pf = {};
        kpair_32mfma_with_pf<0>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw, dummy_pf, dummy_pf);
        kpair_32mfma_with_pf<0>(acc_A1Br, tA1, tBr, a1_raw, br_raw, dummy_pf, dummy_pf);
    }

#else // TAIL_SPLIT == 0: original single-loop (better for large K)

#ifdef UNROLL_K
  #if UNROLL_K == 0
    #pragma unroll
  #else
    #pragma unroll UNROLL_K
  #endif
#elif (K_DIM / 256) <= 16
    #pragma unroll
#elif (K_DIM / 256) <= 32
    #pragma unroll 16
#else
    #pragma unroll 8
#endif
    for (int bt = 0; bt < k_byte_iters; ++bt) {
        const int cur = bt & 1;
        const int nxt = 1 - cur;

        const uint32_t sel_br_p0 = cur ? br_1_p0 : br_0_p0;
        const uint32_t sel_br_p1 = cur ? br_1_p1 : br_0_p1;
        const uint32_t sel_a1_p0 = cur ? a1_1_p0 : a1_0_p0;
        const uint32_t sel_a1_p1 = cur ? a1_1_p1 : a1_0_p1;
        const uint32_t sel_a0_p0 = nxt ? a0_1_p0 : a0_0_p0;
        const uint32_t sel_a0_p1 = nxt ? a0_1_p1 : a0_0_p1;
#if !DIRECT_BL
        const uint32_t sel_bl_p0 = nxt ? bl_1_p0 : bl_0_p0;
        const uint32_t sel_bl_p1 = nxt ? bl_1_p1 : bl_0_p1;
#endif

        const int pf_bt = (bt + 2 < k_byte_iters) ? (bt + 2) : (k_byte_iters - 1);
#if R40A_PF_FENCE && R37_FIX_B && !FUSED_STEP34 && !DIRECT_BL
        // R40A: defer make_pf_params construction until AFTER kpair_64mfma_step34
        // returns (see R37_FIX_B branch below).
#else
        tile_pf_params pf_a0_p = make_pf_params(A0_db[cur], g.a, coord<ST_tile>(0,0,br*2,     pf_bt), so_a, srd_a, base_a, lb_a0[cur], R22B_A_HINT_VAL);
        tile_pf_params pf_a1_p = make_pf_params(A1_db[cur], g.a, coord<ST_tile>(0,0,br*2+1,   pf_bt), so_a, srd_a, base_a, lb_a1[cur], R22B_A_HINT_VAL);
#if !DIRECT_BL
        tile_pf_params pf_bl_p = make_pf_params(Bl_db[cur], g.b, coord<ST_tile>(0,0,bc*2,     pf_bt), so_b, srd_b, base_b, lb_bl[cur], R22B_B_HINT_VAL);
#endif
        tile_pf_params pf_br_p = make_pf_params(Br_db[cur], g.b, coord<ST_tile>(0,0,bc*2+1,   pf_bt), so_b, srd_b, base_b, lb_br[cur], R22B_B_HINT_VAL);
#endif // R40A_PF_FENCE

#if EARLY_SCALE_PF
        fp8e8m0_4 nxt_pf_a0[a_packs], nxt_pf_a1[a_packs], nxt_pf_bl[b_packs], nxt_pf_br[b_packs];
        {
            const uint32_t nxt_scale = static_cast<uint32_t>(bt + 1 < k_byte_iters ? bt + 1 : bt) << 9;
            load_pq_scale_x2_async(a0_srd, lane_soff_x2, nxt_scale, nxt_pf_a0[0], nxt_pf_a0[1]);
            load_pq_scale_x2_async(a1_srd, lane_soff_x2, nxt_scale, nxt_pf_a1[0], nxt_pf_a1[1]);
            load_pq_scale_x2_async(bl_srd, lane_soff_x2, nxt_scale, nxt_pf_bl[0], nxt_pf_bl[1]);
            load_pq_scale_x2_async(br_srd, lane_soff_x2, nxt_scale, nxt_pf_br[0], nxt_pf_br[1]);
        }
#endif

        fp8e8m0_4 a0_raw[a_packs], a1_raw[a_packs], bl_raw[b_packs], br_raw[b_packs];
        #pragma unroll
        for (int p = 0; p < a_packs; ++p) { a0_raw[p] = pf_a0[p]; a1_raw[p] = pf_a1[p]; }
        #pragma unroll
        for (int p = 0; p < b_packs; ++p) { bl_raw[p] = pf_bl[p]; br_raw[p] = pf_br[p]; }

#if !EARLY_SCALE_PF
        {
            const uint32_t nxt_scale = static_cast<uint32_t>(bt + 1 < k_byte_iters ? bt + 1 : bt) << 9;
            load_pq_scale_x2_async(a0_srd, lane_soff_x2, nxt_scale, pf_a0[0], pf_a0[1]);
            load_pq_scale_x2_async(a1_srd, lane_soff_x2, nxt_scale, pf_a1[0], pf_a1[1]);
            load_pq_scale_x2_async(bl_srd, lane_soff_x2, nxt_scale, pf_bl[0], pf_bl[1]);
            load_pq_scale_x2_async(br_srd, lane_soff_x2, nxt_scale, pf_br[0], pf_br[1]);
        }
#endif

        float4 nxt_bl_d[8];
#if DIRECT_BL && EARLY_BL_PF
        // Issue Bl buffer_load BEFORE Step12 — ~128 MFMAs (~512 cyc) hiding window
        {
            const uint32_t bl_nxt_bt = static_cast<uint32_t>((bt + 1 < k_byte_iters) ? (bt + 1) : bt);
            load_bl_direct_async(nxt_bl_d, srd_b_ps, bl_voff_ps,
                bl_ps_n0_soff + bl_nxt_bt * 2 * BL_K0_STRIDE);
        }
#endif

        float4 br_d[8], a1_d[8];
#if R40A_PF_FENCE && R37_FIX_B && !FUSED_STEP34 && !DIRECT_BL
        // R40A: pre-step12 fence — see top-of-iter comment block.
        asm volatile("" ::: "memory");
#endif
        kpair_64mfma_step12(acc_A0Bl, acc_A0Br, tA0, tBl,
            a0_raw, bl_raw, br_raw, br_d, a1_d,
            sel_br_p0, sel_br_p1, sel_a1_p0, sel_a1_p1);

        asm volatile("s_waitcnt lgkmcnt(0)");
        fp4_intx8_t tBr[4], tA1[4];
        extract_tile(br_d, tBr);
        extract_tile(a1_d, tA1);

#if FUSED_STEP34
        // Fused Step34: barrier + 64 MFMAs + 16 ds_reads in one asm block
        float4 nxt_a0_d[8];
        // (nxt_bl_d already declared above)
        kpair_64mfma_step34(acc_A1Bl, acc_A1Br, tA1, tBl, tBr,
            a1_raw, bl_raw, br_raw, nxt_a0_d, nxt_bl_d,
            sel_a0_p0, sel_a0_p1, sel_bl_p0, sel_bl_p1);
        // All prefetches emitted after the fused block
        emit_pf_tail<0>(pf_a0_p, pf_a1_p);
        emit_pf_tail<0>(pf_bl_p, pf_br_p);
#elif R37_FIX_B && !DIRECT_BL
        // R37 Fix B (default, no-TAIL_SPLIT): use fused step3+step4 (correctness
        // fix). The no-TAIL_SPLIT path has no R25-C tail-pf-off branching, so we
        // unconditionally emit all prefetches after the fused block (matches the
        // FUSED_STEP34=1 emission shape).
        float4 nxt_a0_d[8];
        // (nxt_bl_d already declared above)
        kpair_64mfma_step34(acc_A1Bl, acc_A1Br, tA1, tBl, tBr,
            a1_raw, bl_raw, br_raw, nxt_a0_d, nxt_bl_d,
            sel_a0_p0, sel_a0_p1, sel_bl_p0, sel_bl_p1);
#if R40A_PF_FENCE
        // R40A: build pf_*_p AFTER step34 (deferred from top-of-iter); see
        // top-of-iter comment block. Plain memory clobber prevents the compiler
        // from sinking the construction back across step12.
        asm volatile("" ::: "memory");
        tile_pf_params pf_a0_p = make_pf_params(A0_db[cur], g.a, coord<ST_tile>(0,0,br*2,     pf_bt), so_a, srd_a, base_a, lb_a0[cur], R22B_A_HINT_VAL);
        tile_pf_params pf_a1_p = make_pf_params(A1_db[cur], g.a, coord<ST_tile>(0,0,br*2+1,   pf_bt), so_a, srd_a, base_a, lb_a1[cur], R22B_A_HINT_VAL);
        tile_pf_params pf_bl_p = make_pf_params(Bl_db[cur], g.b, coord<ST_tile>(0,0,bc*2,     pf_bt), so_b, srd_b, base_b, lb_bl[cur], R22B_B_HINT_VAL);
        tile_pf_params pf_br_p = make_pf_params(Br_db[cur], g.b, coord<ST_tile>(0,0,bc*2+1,   pf_bt), so_b, srd_b, base_b, lb_br[cur], R22B_B_HINT_VAL);
#endif // R40A_PF_FENCE
        emit_pf_tail<0>(pf_a0_p, pf_a1_p);
        emit_pf_tail<0>(pf_bl_p, pf_br_p);
#else // R37_FIX_B == 0 OR DIRECT_BL → legacy non-fused path
#if !STEP3_EMBED_BARRIER
        // R19B: site _S7 (no-TAIL_SPLIT outer !STEP3_EMBED_BARRIER, dead w/ default STEP3_EMBED_BARRIER=1)
        asm volatile(MXFP4_STEP3_BARRIER_INST_S7 ::: "memory");
#endif

        float4 nxt_a0_d[8];
        // (nxt_bl_d already declared above)
#if SPREAD_LDS
        kpair_32mfma_with_lds_rowspread_pf<STEP3_PF_N, STEP3_EMBED_BARRIER>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw,
#else
        kpair_32mfma_with_lds_and_pf<STEP3_PF_N, STEP3_EMBED_BARRIER>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw,
#endif
            nxt_a0_d[0], nxt_a0_d[1], nxt_a0_d[2], nxt_a0_d[3],
            nxt_a0_d[4], nxt_a0_d[5], nxt_a0_d[6], nxt_a0_d[7],
            sel_a0_p0, sel_a0_p1, pf_a0_p, pf_a1_p);
        emit_pf_tail<STEP3_PF_N>(pf_a0_p, pf_a1_p);

#if DIRECT_BL && EARLY_BL_PF
        // Step4: pure MFMAs + Br prefetch only (Bl already loaded into nxt_bl_d above)
        {
            tile_pf_params dummy_pf = {};
            kpair_32mfma_with_pf<PF_MPT>(acc_A1Br, tA1, tBr, a1_raw, br_raw,
                pf_br_p, dummy_pf);
        }
#elif DIRECT_BL
        {
            const uint32_t bl_nxt_bt = static_cast<uint32_t>((bt + 1 < k_byte_iters) ? (bt + 1) : bt);
            kpair_32mfma_with_vmem_bl_wrap(acc_A1Br, tA1, tBr, a1_raw, br_raw,
                nxt_bl_d[0], nxt_bl_d[1], nxt_bl_d[2], nxt_bl_d[3],
                nxt_bl_d[4], nxt_bl_d[5], nxt_bl_d[6], nxt_bl_d[7],
                bl_voff_ps, srd_b_ps,
                bl_ps_n0_soff + bl_nxt_bt * 2 * BL_K0_STRIDE);
        }
        #pragma unroll
        for (int pi = 0; pi < PF_MPT; ++pi) emit_one_pf(pf_br_p, pi);
#else
#if SPREAD_LDS
        kpair_32mfma_with_lds_rowspread_pf<STEP4_PF_N>(acc_A1Br, tA1, tBr, a1_raw, br_raw,
#else
        kpair_32mfma_with_lds_and_pf<STEP4_PF_N>(acc_A1Br, tA1, tBr, a1_raw, br_raw,
#endif
            nxt_bl_d[0], nxt_bl_d[1], nxt_bl_d[2], nxt_bl_d[3],
            nxt_bl_d[4], nxt_bl_d[5], nxt_bl_d[6], nxt_bl_d[7],
            sel_bl_p0, sel_bl_p1, pf_bl_p, pf_br_p);
#if STEP4_EXTERNAL_BR_PREFETCH
        #pragma unroll
        for (int pi = 0; pi < PF_MPT; ++pi) emit_one_pf(pf_br_p, pi);
#else
        emit_pf_tail<STEP4_PF_N>(pf_bl_p, pf_br_p);
#endif
#endif // DIRECT_BL
#endif // FUSED_STEP34 / R37_FIX_B / legacy

#if DIRECT_BL
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
#else
        asm volatile("s_waitcnt lgkmcnt(0)");
#endif
        R41A_FENCE_BEFORE_EXTRACT();
        extract_tile(nxt_a0_d, tA0);
        R41A_FENCE_BEFORE_EXTRACT();
        extract_tile(nxt_bl_d, tBl);
        // R21B/R22C: opt-in scheduling hooks (no-op at defaults). Site 2.
        MXFP4_R22C_ITER_END_HOOK_2;

#if EARLY_SCALE_PF
        #pragma unroll
        for (int p = 0; p < a_packs; ++p) { pf_a0[p] = nxt_pf_a0[p]; pf_a1[p] = nxt_pf_a1[p]; }
        #pragma unroll
        for (int p = 0; p < b_packs; ++p) { pf_bl[p] = nxt_pf_bl[p]; pf_br[p] = nxt_pf_br[p]; }
#endif
    }

#endif // TAIL_SPLIT
#endif // SWAP_STEP34_MAIN

    // ═══════════ Store C -- streamlined direct store ═══════════
    // R21B: optionally drop wave priority before the store epilogue.
    MXFP4_R21B_PRIO_LOW_HOOK;
    // R22C: optional sched_group_barrier just before Store-C (site bit3).
    MXFP4_R22C_PRESTORE_HOOK;
    // Process base tiles directly from accumulators without materializing RT_C.
    auto store_block = [&](const fp4_floatx4_t acc[16], int mh, int nh) {
        const int lid = kittens::laneid();
        const int tile_r = br * WARPS_M * 2 + WARPS_M * mh + wm;
        const int tile_c = bc * WARPS_N * 2 + WARPS_N * nh + wn;
        bf16 *dst_ptr = g.c.raw_ptr + static_cast<size_t>(tile_r * 64) * g.c.cols()
                        + static_cast<size_t>(tile_c * 64);
        const int row_stride = g.c.cols();
        const int row_off = 4 * (lid / 16);
        const int col_off = lid % 16;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                fp4_floatx4_t s = acc[i * 4 + j] * g.scale;
                const int row_base = i * 16 + row_off;
                const int col = j * 16 + col_off;
                store_bf16_val(&dst_ptr[(row_base + 0) * row_stride + col], s[0]);
                store_bf16_val(&dst_ptr[(row_base + 1) * row_stride + col], s[1]);
                store_bf16_val(&dst_ptr[(row_base + 2) * row_stride + col], s[2]);
                store_bf16_val(&dst_ptr[(row_base + 3) * row_stride + col], s[3]);
            }
        }
    };

#if SWAP_STEP34_MAIN
    auto store_block_inner = [&](const fp4_floatx4_t acc[16], int mh, int nh) {
        const int lid = kittens::laneid();
        const int tile_r = br * WARPS_M * 2 + WARPS_M * mh + wm;
        const int tile_c = bc * WARPS_N * 2 + WARPS_N * nh + wn;
        bf16 *dst_ptr = g.c.raw_ptr + static_cast<size_t>(tile_r * 64) * g.c.cols()
                        + static_cast<size_t>(tile_c * 64);
        const int row_stride = g.c.cols();
        const int lane_group = lid / 16;
        const int lane_pos = lid % 16;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                fp4_floatx4_t s = acc[i * 4 + j] * g.scale;
                const int row = i * 16 + lane_pos;
                const int col = j * 16 + 4 * lane_group;
#if PACKED_STORE
                // Pack 4 bf16 into 2 dword stores (4x fewer store instructions)
                store_bf16x2_packed(&dst_ptr[row * row_stride + col + 0], s[0], s[1]);
                store_bf16x2_packed(&dst_ptr[row * row_stride + col + 2], s[2], s[3]);
#else
                store_bf16_val(&dst_ptr[row * row_stride + col + 0], s[0]);
                store_bf16_val(&dst_ptr[row * row_stride + col + 1], s[1]);
                store_bf16_val(&dst_ptr[row * row_stride + col + 2], s[2]);
                store_bf16_val(&dst_ptr[row * row_stride + col + 3], s[3]);
#endif // PACKED_STORE
            }
        }
    };
#if SWAP_STEP12_MAIN
    store_block_inner(acc_A0Bl, 0, 0);
    store_block_inner(acc_A0Br, 0, 1);
    store_block_inner(acc_A1Bl, 1, 0);
    store_block_inner(acc_A1Br, 1, 1);
#else
    store_block(acc_A0Bl, 0, 0);
    store_block(acc_A0Br, 0, 1);
    store_block_inner(acc_A1Bl, 1, 0);
    store_block_inner(acc_A1Br, 1, 1);
#endif
#else
    store_block(acc_A0Bl, 0, 0);
    store_block(acc_A0Br, 0, 1);
    store_block(acc_A1Bl, 1, 0);
    store_block(acc_A1Br, 1, 1);
#endif

#if PERSISTENT_XCD
    __syncthreads();   // Fix B: ensure all stores for the current tile are visible
                       // and all threads have left the store loop before the next
                       // atomicAdd claim on the global tile counter.
    } // end while(true) persistent loop
#else
    } // end static-dispatch block
#endif
}

void dispatch_gluon_cpp(gluon_globals g) {
    int m = static_cast<int>(g.c.rows());
    int n = static_cast<int>(g.c.cols());
#if PERSISTENT_XCD
    // Fix A: robust counter reset (checked symbol lookup + synchronous memset).
    unsigned int* counter_dev = nullptr;
    hipError_t err = hipGetSymbolAddress((void**)&counter_dev,
                                         HIP_SYMBOL(g_persistent_tile_counter));
    if (err != hipSuccess || counter_dev == nullptr) {
        fprintf(stderr, "PERSISTENT_XCD: hipGetSymbolAddress failed (%d) - aborting\n",
                (int)err);
        std::abort();
    }
    hipError_t merr = hipMemset(counter_dev, 0, sizeof(unsigned int));
    if (merr != hipSuccess) {
        fprintf(stderr, "PERSISTENT_XCD: hipMemset failed (%d)\n", (int)merr);
        std::abort();
    }
    // Fix C: cap grid by problem size so we never over-launch on small problems.
    const int total_tiles = (m / BLK) * (n / BLK);
    const int grid_x = total_tiles < PERSISTENT_GRID ? total_tiles : PERSISTENT_GRID;
    const dim3 grid(grid_x);
#else
    const dim3 grid((m / BLK) * (n / BLK));
#endif
    mxfp4_gluon_cpp_kernel<<<grid, dim3(_NUM_THREADS), 0>>>(g);
}

PYBIND11_MODULE(tk_mxfp4_gluon_cpp, m) {
    m.doc() = "MXFP4 Gluon-arch kernel (C++ reimplementation)";
#if DIRECT_BL
    py::bind_function<dispatch_gluon_cpp>(m, "gemm_rcr",
        &gluon_globals::a, &gluon_globals::b,
        &gluon_globals::a_scale, &gluon_globals::b_scale,
        &gluon_globals::c, &gluon_globals::b_ps);
#else
    py::bind_function<dispatch_gluon_cpp>(m, "gemm_rcr",
        &gluon_globals::a, &gluon_globals::b,
        &gluon_globals::a_scale, &gluon_globals::b_scale,
        &gluon_globals::c);
#endif
}
