# Round 6 — FP8 grouped: dense 2-tile main loop port FAILS

**Date**: 2026-05-01
**HK SHA (pre)**: aaee8d26 (round-5 docs only)
**HK SHA (post)**: this commit (notes-only; code unchanged)
**Primus-Turbo SHA**: c3b70e374 (unchanged)
**Round-6 baseline metric**: 791
**Round-6 result metric**: 791 (no code change shipped)

## 1. Hypothesis

Round-3/4/5 roadmap item (b): port dense `gemm_kernel<RCR>`'s 2-tile main
loop body (lines 1256-1326 in `kernel_fp8_layouts.cpp`) into
`grouped_rcr_kernel`. Dense uses 2-tile when `ki ≥ RCR_TWO_TILE_MIN_KI = 28`;
gpt_oss FP8 has `ki = K/BK = 2880/128 = 22` (even). Lowering the
grouped-side threshold to 22 would steer all 8 gpt_oss FP8 shapes to
the 2-tile body. Estimated yield (round-3 notes): +3-5 pp ratio.

Mechanism the 2-tile body is supposed to provide: each iter processes
2 K-iters with explicit `Bs[0]/Bs[1]` + `As[0]/As[1]` indexing (no
`tic^toc` swap). The compiler sees a longer instruction sequence and
can pack `mfma` + `load` + `barrier` more tightly within the 2-tile
window. Dense kernel measurements (round-1 notes) showed +1.4 % TF on
isolated dense ki=22 probes.

## 2. Patch

`analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp:2082-2120` (grouped
RCR main loop). Replaced the single-tile body with a branched form:

```cpp
if ((ki_dyn & 1) == 0 && ki_dyn >= 22) {
    auto main_loop_iter = [&](int tile) {
        // 8 mma + 14 rcr_8w_load_hoist + 16 s_barrier
        // Mirrors dense lines 1257-1318 1:1.
    };
    TK_PRAGMA_UNROLL(RCR_MAIN_UNROLL)
    for (int tile = 0; tile < ki_dyn - 2; tile += 2) {
        main_loop_iter(tile);
    }
    TK_WAIT_VMCNT(0); __builtin_amdgcn_s_barrier();
} else {
    // Existing 1-tile loop (round-4 sched_barrier-removed)
}
```

Build resources stayed identical (VGPRs 256 / Spill 67 / LDS / Occ 2).
Static instruction count grew (the 2-tile body is ~2× the 1-tile body).

## 3. Result — CATASTROPHIC FAIL

Probe (gpt_oss-GateUP-B32-M4096): HK 1252 → **797 TF (-36.4 %)**.
Metric: score **791 → 647 (-144)**. grp_FP8 geomean **0.864 → 0.577
(-28.7 pp)**. DSV3 [watch] below_1.0 went 5/16 → 8/16 (no correctness
FAIL — pure perf regression).

REVERTED via `git checkout --`. State restored to aaee8d26.

## 4. Why dense's 2-tile body doesn't port to grouped

The dense kernel's 2-tile body was hand-tuned for the **dense** kernel
prologue/epilog/main-loop interaction:

* **Dense prologue is short** — single-shape, no group binary-search,
  no group-by-M swizzle. Prologue loads complete fast, so the 2-tile
  body's first `TK_WAIT_LGKM(RCR_PREFETCH_LGKM)` finds prologue vmem
  already done.
* **Grouped prologue is long** — binary-search for group_idx +
  `m_subtile_A/C` derivation + group-by-M `br/bc` swizzle + dispatch
  on `bpr_g/num_pid_n` bound. Adds ~50-100 cyc of SALU work between
  the persistent-loop top and the prologue's first `rcr_8w_load_hoist`.
  Combined with the 2-tile body's PRELOAD of K-iter `tile+2/+3` data
  in the SAME iter that consumes K-iter `tile/tile+1`, the prologue's
  2-staged `As[0..1] / Bs[0..1]` aren't fully ready when the 2-tile
  body's first MMA fires → extended `s_waitcnt vmcnt` per iter.
* **`RCR_TWO_TILE_MID_VMCNT = 6` is dense-tuned**. Grouped's per-iter
  vmem outstanding profile differs (extra LDS reads from group caches
  `s_offs[]` / `s_cum_tiles[]` indirectly through the persistent loop
  bookkeeping). The mid-iter wait at vmcnt=6 fires when there are
  more outstanding loads than dense would have, so the wait blocks
  longer.
* **The 2-tile body's preload pattern is "ahead by 2 K-iters"**. For
  dense with deep prologues (ki >= 28), this is fine — load latency
  is amortised across many iters. For ki=22 with 10 outer iters and
  RCR_MAIN_UNROLL=2 (= 5 unrolled outer iters), the preload window
  is too tight; the SQ vmem queue saturates and stalls the wave.

## 5. What might work instead

The 2-tile body itself is sound (correctness PASSed; just slow). To
make it work for grouped, would need:

* **Tune `RCR_TWO_TILE_MID_VMCNT` for grouped** — sweep over {2, 4, 6,
  8, 10}. Dense-optimal is 6; grouped may want higher (8-10) to allow
  more outstanding vmem before draining.
* **Disable `RCR_MAIN_UNROLL` for grouped 2-tile body** — code size
  pressure at ki=22 × unroll 2 = 5 × 2-tile body iterations may exceed
  I-cache lines for the persistent grouped kernel which already has
  K-tail + binary-search bookkeeping.
* **Adapt the prologue** — load 4 K-iters worth of B/A data in the
  prologue (instead of 2) so the first 2-tile iter's preloads aren't
  on the critical path.

These are 2-4 hr of careful sweep work each. Out of scope for round 6.

## 6. Round-6 alternative attempts (also flat or worse)

After reverting the 2-tile port, no other code change was attempted
this round (round-5 already exhausted RCR_STEADY_VMCNT, RCR_PREFETCH_LGKM,
and the Down-B4-M4096 config sweep). Round-6 ships notes only.

## 7. Updated next-round roadmap

The "easy" 2-tile port is **closed** (catastrophic fail when
naively ported). Remaining levers:

* **(b') 2-tile body port + grouped-tuned `MID_VMCNT` + unroll=1** —
  retry the 2-tile body with grouped-specific wait threshold and
  unroll disable. Estimated 3-4 hr; may recover the +3 pp yield, may
  flat-line.
* **(c) Skip A-tile LDS staging** — Triton-style direct HBM → reg main
  loop. Removes 8 cross-warp `s_barrier` per K-iter (the 64-cyc cost
  identified in round-5 §2 as the load-bearing barrier set). 4-8 hr
  structural rewrite. Estimated yield: +5-7 pp ratio (closes most of
  MFMA Util gap 35 → 42 %).
* **(d) Port FP8 round-3 single-wait pattern to BF16 grouped K-tail**.
  +0.3-0.5 pp on grpBF16 geomean. Same vgpr-add risk as FP8 round-3.

## 8. Updated rule for register-pressure changes

Round-3 single-wait: +1 register tile (`a_kt1`), +0.7 pp.
Round-5 full hoist: +3 register tiles, **−55 score**.
Round-6 2-tile port: 0 register tiles but +static instruction count,
**−144 score**.

**Lesson updated**: each round must change AT MOST one of:
  1. +1 register tile, OR
  2. ≤5 instruction-count delta in main loop, OR
  3. wait-counter constant tweak.

Larger structural changes need multi-round prototyping with a
roll-back plan baked in.

## 9. Files touched

- `analysis/_notes/round-6-fp8-grouped-2tile-port-fail.md` (this file)
- (No kernel or config code changes shipped this round.)
