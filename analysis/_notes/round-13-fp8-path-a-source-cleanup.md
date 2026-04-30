# Round-13 — FP8 fuse path-A source-clarity cleanup + plateau review

**Date**: 2026-04-30 (continued from round-12 rocprof breakdown)
**Round**: 13 / 100 (auto_optimize)
**Score**: 833 → 833 (4-round plateau)
**Primus-Turbo HEAD**: 0cff2388 (no Primus changes this round)
**HipKittens HEAD before**: e022c2f3 (round-12 rocprof note)

## TL;DR

Round-12 confirmed via `rocprofv3` that the score plateau at 833 is
**architectural** — the 50 µs/iter gap on the worst FP8 shape lives
entirely inside the HK GEMM kernel itself (198 µs HK vs 148 µs Triton),
not in dispatch / quantize / scale overhead. Closing that gap requires
a kernel-template rewrite (new BM/BN/BK + register layout re-derivation),
not a micro-optimisation.

Round-13 ships a **source-clarity** chore: delete the dead
`prefill_swizzled_offsets_partial_K` helper (~120 lines) plus the ~80-line
historical path-A comment block in the FP8 grouped fuse epilog. Round-11
(commit f9d591cb) had already dropped the dead callers but left the
function definition + stale comments behind. Codegen is unchanged (DCE
was already firing on the unreferenced symbol — see round-11 commit
message); the file shrinks 5759 → 5621 lines.

## Why no metric-moving change this round

Per round-10 / round-11 / round-12 documentation, the K-tail-fuse
**forward main line is fully saturated**:

| dtype | layout | fuse status                          | path | round |
|-------|--------|--------------------------------------|------|-------|
| BF16  | RCR    | ✓ shipped — direct HBM→Reg, K_REM=64 | B    |  5    |
| BF16  | RRR    | ✗ path A/B numerically failed        | A/B  |  5-7  |
| BF16  | RRR    | ✓ workaround — H4 reroute via b.T    | H4   |  9    |
| FP8   | RCR    | ✓ shipped — direct HBM→Reg + per-grp | B    |  3-7  |
| FP8   | RRR    | ✗ not done (dA bwd, not metric)      | —    |   —   |

All 32 metric shapes go through fuse (RCR forward) with `FUSED_KTAIL=true`
and never touch the standalone `grouped_ktail_kernel_*` external launches.
The remaining FP8 RRR fuse is a backward path that the metric does not
measure (verified by `scripts/_metric_grouped_only.py` only timing
forward TFLOPS).

The task-body's main line ("K-tail fuse forward into main kernel epilog")
is therefore complete. The remaining items in the architectural-ceiling
bucket are:

1. **Kernel-template rewrite** for `BN ∈ {192, 144}` so N=2880=15·192=20·144
   (gpt_oss) avoids the 256-tile 90 % occupancy quirk. Multi-round work:
   register layout re-derivation + LDS swizzle re-design + main-loop +
   fuse-epilog re-fitting + numerical re-verification. Score upside:
   est. +50–100 (would lift FP8 gpt_oss section from 0.83 → 1.00–1.05;
   but the 1.20 PASS bar would still need either a Triton-beating GEMM
   kernel or ratio progress capped at 1.0).
2. **dB var-K fuse** (FP8 + BF16). Backward; doesn't move metric.
   `grouped_var_k_kernel_fp8` shows 256 VGPR / 67 spill — register-pressure-
   bound. Long-term codebase health.
3. **Accept the plateau**. Score 833 reflects a kernel-template ceiling
   that exists across multiple metric shapes; further wedge attempts in
   the existing template (rounds 11 store hoist, round 12 SRD hoist)
   were net-neutral or net-negative.

## Round-13 changes

* `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp` (-138 lines):
  - Removed `template<...> __device__ __forceinline__ void
    prefill_swizzled_offsets_partial_K(...)` (former path-A scaffolding).
  - Replaced ~10-line stale block-comment in the kernel-template header
    (line ~1980) with a concise reference to round-3 path-B and
    `analysis/_notes/round-{2,3}-fp8-ktail-*.md`.
  - Replaced ~80-line stale path-A historical block comment in
    `grouped_rcr_kernel`'s fuse epilog (around line ~2250) with a
    history-pointing summary.
  - Replaced ~12-line round-10 cleanup comment in the prologue (around
    line ~2060) with a 6-line summary.

* `analysis/_notes/round-13-fp8-path-a-source-cleanup.md` (this file).

## Verification

```
$ make -j  # tk_fp8_layouts build
[ok] no errors; warnings unchanged from round-12
[grouped_rcr_kernel] VGPRs: 256, AGPRs: 0, Spill: 67 dwords (BF16-FUSED + FP8-FUSED templates byte-identical SASS to round-12)
[grouped_var_k_kernel_fp8] VGPRs: 256, Spill: 67 — unchanged

$ python3 scripts/_metric_grouped_only.py
score=833  weights=grpBF16:1 grpFP8:1
  grp_BF16 geomean=1.0926 (vs round-12 1.0887; noise-level identical)
  grp_FP8  geomean=0.9146 (vs round-12 1.0888; noise-level identical)
```

Codegen is byte-identical to round-12 (DCE on the unreferenced symbol
was firing well before this cleanup); the only delta is source clarity.

## Next-round angles (in priority order)

1. **Accept plateau + document** (this round + round-12 already did this).
2. **dB var-K fuse for FP8** — long-term codebase health. The
   `grouped_var_k_kernel_fp8` has 67 VGPR spill / 2 waves/SIMD; reducing
   spill via path-A LDS-staged or path-B direct-HBM K-tail (mirroring
   FP8 RCR round-3 / BF16 RCR round-5) would halve backward dB cost on
   gpt_oss-Down (currently 27–38 % Triton per task-body P2 note). Does
   not move metric (forward TFLOPS only) but the work is well-scoped.
3. **New kernel template for `BN=192`** — 1–2 rounds, very risky. Would
   need:
   - New `ST_v2_192` layout (192-col equivalent of `ST_v2`).
   - Re-derive `rt_16x128_s` lane mapping for BN=192 register tile.
   - New `rcr_8w_load_hoist` + `rcr_mma` specialised for the 192-col tile.
   - Numerical verification across all 32 metric shapes.
   - Upside capped: even 0 % overhead lifts FP8 gpt_oss to ~1.00, still
     below the 1.20 PASS bar. Would need _both_ this template _and_ the
     main-loop microarch wedge to reach PASS.
4. **Do NOT** chase rule tune (round-9..12 docs all confirm saturated;
   task body forbids).
5. **Do NOT** chase main-K-loop micro-tweaks (round-11 store hoist
   regressed −7..−10 score; round-12 SRD hoist net-neutral). The
   compiler already CSEs every wave-uniform constant the source can
   reasonably hoist.

## Resource snapshot for the next agent

* `grouped_rcr_kernel<KI=0, N_MASKED=true,  FUSED=true>`: VGPRs 256, Spill 82, Occupancy 2 waves/SIMD
* `grouped_rcr_kernel<KI=0, N_MASKED=false, FUSED=true>`: VGPRs 256, Spill 72, Occupancy 2 waves/SIMD
* `grouped_var_k_kernel_fp8<KI=0>`: VGPRs 256, Spill 67, Occupancy 2 waves/SIMD (dB hot path)
* All `grouped_ktail_kernel_*` external-launch variants: 32–64 VGPR, 0 spill, but they sit on dead BF16/FP8 dispatch fall-back paths that the current metric never reaches.

The "main K-loop is the gap" diagnosis from round-12 still holds: at 256 VGPR / 2 waves the kernel is occupancy-bound, and the fuse epilog only contributes ~5 % of total wall — the remaining ~95 % is in the K=2..ki_dyn-1 main loop body. Closing the 50 µs/iter gap would require either reducing main-loop spill (= 67 dwords = 268 bytes/lane scratch, dominant cost in occupancy) or unlocking 4 waves/SIMD via VGPR reduction, neither of which is reachable inside the existing 256-BM × 256-BN × 128-BK kernel template (every spilled register has been measured and ruled out individually across rounds 1-12).
