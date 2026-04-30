# Round 15 — FP8 RCR `RCR_TWO_TILE_MIN_KI` 28 → 20 probe (no-op)

## Context

Score plateau at 833 for 5 consecutive rounds (10/11/12/13/14 — round 14
H4 FP8 reroute moved bwd +8.4 % avg but metric is forward-only, score
unchanged). Round 12 rocprof identified the dominant 50 µs/iter gap
between HK and Triton FP8 forward sits in the **main K-loop**, not the
K-tail fuse epilog or prologue.

Round 15 probe: lower the kernel-internal schedule branch threshold
`RCR_TWO_TILE_MIN_KI` from 28 to 20 to cover gpt_oss FP8 forward
shapes (K=2880, fast_k=2816, **ki_dyn=22**), which currently fall
through to the single-tile main-loop fallback because `ki_dyn=22 < 28`.

## Hypothesis

Two-tile schedule (`kernel_fp8_layouts.cpp:1247-1316`) advances 2 K-blocks
per iter with 8 mma cells and 4 LDS slots in flight, vs single-tile
(`:1320-1389`) which advances 1 K-block per iter with 4 mma cells and
2 LDS slots. The deeper register-cell pipeline should hide more HBM
latency on long-enough K-loops.

`28` was an empirical break-even from earlier dense kernel tuning.
gpt_oss `ki_dyn=22` sits 21 % below it. If the two-tile pipeline still
amortizes at ki=22, lowering the threshold should win across all 8
gpt_oss FP8 cases (which currently sit at ratio 0.82–0.88).

## Constraint compliance

`RCR_TWO_TILE_MIN_KI` is a **kernel-template constant** (constexpr int
inside the kernel template), gating a runtime `if` branch in
`grouped_rcr_kernel`. Both schedule paths are compiled into the binary
(verified — VGPR spill numbers identical with both 28 and 20: spill
91 / 83 / 72 / 82 across the four `<KI=0, N_MASKED, FUSED_KTAIL>`
template instantiations). Lowering the threshold only changes which
schedule the runtime takes for a given `ki_dyn`, not the codegen.

This is **NOT** rule-tune (host-side dispatch decision per shape) —
which the task body forbids. It IS a main-K-loop schedule probe within
the kernel itself, which round 12 rocprof identified as the wedge.

## Result: NO-OP within bench noise

Metric (after RCR_TWO_TILE_MIN_KI = 20):

| metric | before (round-14 commit 4871e4f) | after (probe) | Δ |
|--------|----------------------------------|---------------|---|
| score  | 834 | 833 | -1 (noise) |
| BF16 geomean | 1.0926 | 1.0913 | -0.13 % |
| FP8 geomean  | 0.9171 | 0.9147 | -0.24 % |

bench (FP8, 16 cases × ~30 iter each):

| metric | before | after | Δ |
|--------|--------|-------|---|
| avg fwd TFLOPS | 1207.25 | 1206.24 | -0.08 % |
| avg bwd TFLOPS | 967.43  | 962.38  | -0.52 % |

Per-case gpt_oss FP8 fwd (8 cases, the targets of the probe):

| case | before | after | Δ |
|------|--------|-------|---|
| GateUP B=4  M=2048 | 960.18 | 958.94 | -0.13 % |
| Down   B=4  M=2048 | 732.44 | 730.72 | -0.23 % |
| GateUP B=4  M=4096 | 1106.92| 1107.35| +0.04 % |
| Down   B=4  M=4096 | 987.99 | 988.26 | +0.03 % |
| GateUP B=32 M=2048 | 1068.50| 1068.89| +0.04 % |
| Down   B=32 M=2048 | 926.63 | 926.67 | 0      % |
| GateUP B=32 M=4096 | 1234.39| 1234.43| 0      % |
| Down   B=32 M=4096 | 1084.59| 1083.94| -0.06 % |

**All 8 ratios within ±0.23 % — pure noise. No measurable wedge.**

## Why no wedge?

ki_dyn = 22 is too short for two-tile amortization to net-win:

1. **Loop count**: two-tile runs `(22 - 2) / 2 = 10` loop bodies (each
   covering 2 K-blocks); single-tile runs `22 - 2 = 20` loop bodies.
   Same total mma count (88 + 8 epilog = 96 cells either way).

2. **Epilog fraction**: 8 mma in epilog 1 + epilog 2 / 96 total mma =
   8.3 %. Same in both paths. So epilog isn't the discriminator.

3. **Prologue fraction**: two-tile has same prologue (2 K-blocks
   prefetched ahead) as single-tile. Same.

4. **Pipeline depth**: two-tile's deeper 4-LDS-slot pipeline makes
   sense when the HBM latency across (load → mma → load) is large
   relative to the number of mma cells per iter. With ki_dyn=22 the
   per-iter HBM throughput requirement is moderate; single-tile's
   2-LDS-slot pipeline already saturates the HBM bandwidth here.

5. **Issue queue / register port pressure**: two-tile's reformulated
   register dependency chain (each mma cell reads from a longer
   prefetch chain) makes the SIMD issue queue marginally busier.
   At ki=11 (10 main loop iterations + 1 epilog) the increased issue
   pressure offsets the 2× larger per-iter cell budget.

The 28 break-even threshold is approximately correct for this hardware
+ schedule combination; gpt_oss ki=22 sits below it for a real reason.

## Conclusion / decision

**Reverted** RCR_TWO_TILE_MIN_KI back to 28 (commit). Round-15 leaves
the binary unchanged; this note documents the probe so future rounds
don't re-attempt the same threshold sweep.

## Implications for the score plateau

The 50 µs/iter HK-vs-Triton gap on gpt_oss FP8 forward is NOT closable
by tuning the two-tile schedule threshold. The wedge truly sits in
either:

* Triton's `_grouped_fp8_persistent_gemm_kernel` MFMA scheduling (uses
  `mfma_scale_f32_32x32x64_f8f6f4` per 32×32 cell — 50 % fewer mfma
  instructions per K-step than HK's `mfma_scale_f32_16x16x128_f8f6f4`
  per 16×16 cell).
* A different prefetch pattern (Triton's `tl.dot(a, b)` lowers to a
  more aggressive HBM→register-file pipeline that bypasses the LDS
  staging overhead in the inner loop).

Either of those is a multi-round kernel-template rewrite, not a
threshold tweak. The score plateau remains at 833 ± 1.

## Next-round suggestions

1. **MFMA cell-shape probe**: replace `mfma_scale_f32_16x16x128_f8f6f4`
   with `mfma_scale_f32_32x32x64_f8f6f4` in `rcr_mma`. Requires
   rewriting `rt_fl<RBM=64, RBN=32, col_l, rt_16x16_s>` → `rt_32x32_s`
   accumulator tile + corresponding A/B register tile + lane-cell
   mapping for store. ~1-2 round project.

2. **K-tail fuse on FP8 RRR (dA bwd)** — round-14 H4 reroute relieves
   the metric-irrelevant external-launch slow path; a real RRR fuse
   that avoids the transpose would remove the M_total × N_orig fp8
   reformat pass. Bench-only impact, doesn't move metric.

3. **Continue plateau analysis**: rocprof per-instruction trace
   (rocm-llvm-objdump + roc-systrace) to compare HK vs Triton main
   K-loop assembly cycle-by-cycle, identify the specific
   instruction-level inefficiency.
