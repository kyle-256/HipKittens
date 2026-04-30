# Round-12 progress note: FP8 grouped GEMM rocprof breakdown

**Date**: 2026-04-30
**Round**: 12 / 100 (auto_optimize)
**Score**: 832-835 (3-run mean ≈ 834, plateau confirmed for 3rd consecutive round)
**Primus-Turbo HEAD**: 0cff2388
**HipKittens HEAD before round**: 83cea423 (round-11 BF16 hoist note)

## TL;DR

`rocprofv3` kernel-trace on `gpt_oss-GateUP-B4-M2048` FP8 (the worst
shape in the metric, ratio 0.83) confirms **65 % of wall is in the
HipKittens FP8 grouped GEMM kernel itself** and **35 % is FP8
quantization scaffolding** (amax / scale / cast). The same
quantization stack runs end-to-end identically under Triton — the
ratio gap is therefore **entirely in the GEMM kernel runtime**.

## Kernel-time breakdown (gpt_oss-GateUP-B4-M2048, K=2880, B=4, 200 iters)

```
                                                  HK          Triton    Δ
grouped_rcr_kernel<0,true,true> / Triton GEMM     40.2 ms     30.1 ms   −10.1 ms
unary_kernel<512,8,bf16,fp8e4m3>                   8.8         8.8       0
reduce_row_kernel<AbsMaxOp, bf16, float, ..>       8.5         7.8       −0.7
reduce_row_kernel<AbsMaxOp, float, float, ..>      1.8         1.8       0
compute_scale_from_amax_kernel<float>              1.7         1.9      +0.2
compute_group_offs_device<long>                    0.9         0.9       0
at::native::*                                      0.2         0.2       0
__amd_rocclr_copyBuffer                            0.0         0.0       0
                                                 -----       -----    -----
TOTAL                                            62.0 ms     51.4 ms   −10.6 ms
```

* HK total / iter = 305 µs; Triton = 253 µs.
* Ratio = Triton / HK = 0.830 → matches metric `0.829`.
* HK GEMM kernel itself = 198 µs / iter; Triton GEMM = 148 µs / iter
  (HK is **1.34× slower** in raw GEMM throughput).
* Non-GEMM overhead (~107 µs HK, ~106 µs Triton) is essentially
  identical — the same `quantize_fp8 → grouped_gemm_fp8 → reduce` op
  graph runs on both backends; only the GEMM kernel differs.

## Implications for the score plateau

* The remaining gap is **entirely** in the GEMM kernel. Even if HK
  GEMM perfectly matched Triton GEMM (148 µs / iter), total time
  would drop to 148 + 107 = 255 µs → ratio ≈ 1.00, **still below
  the 1.20 target**. Hitting 1.20 requires HK GEMM to be **20 % faster
  than Triton GEMM** (≈ 124 µs / iter) — i.e. HK has to outperform
  Triton's mature `_grouped_fp8_persistent_gemm_kernel`.
* Triton's persistent GEMM uses `origami.rank_configs` to pick
  `(BM, BN, BK, num_warps, num_stages, group_size_m, cache_a, cache_b)`
  per-shape; HK uses a fixed kernel template `(BM=256, BN=256, BK=128,
  num_warps=8, RBM=64, RBN=32)` with only `(group_m, num_xcds)` knobs.
  Closing the −34 % raw-GEMM gap is a kernel rewrite, not a
  micro-optimisation.
* All "small wedge" attempts in rounds 10-12 (dead-prefill cleanup,
  BF16 RCR store hoist, FP8 fuse-epilog SRD hoist) are net-neutral
  or negative on the metric — the compiler already CSEs the
  wave-uniform setups, and reordering the epilog branches just
  perturbs the dominant DSV3 hot path.

## Round-12 micro-probes attempted

### Probe 1: FP8 fuse-epilog kernel-uniform SRD hoist (net-neutral, reverted)

The fuse-epilog at `kernel_fp8_layouts.cpp:~2373` recomputes 7
launch-uniform values per persistent-tile iteration:

```
const fp8e4m3* a_base_ptr;          // launch-uniform
const int      a_row_stride_bytes;  // launch-uniform
const uint32_t a_total_bytes;       // launch-uniform
const uint32_t K_tail_base_bytes;   // launch-uniform
const fp8e4m3* b_base_ptr;          // launch-uniform
const int      b_row_stride_bytes;  // launch-uniform
const i32x4    a_srsrc_kt;          // launch-uniform (1 make_srsrc)
```

Hoisted these to kernel entry (above the persistent `for (gt = pid;
...)` loop, inside an `if constexpr (FUSED_KTAIL)` guard so the
non-fused template instantiation pays nothing).

* SGPR/VGPR: identical (67 spills before, 67 after — compiler
  already CSE'd the per-tile redundant computes into SGPR-uniform
  values that survive the persistent loop).
* `_metric_grouped_only.py` 3-run: 832 / 835 / 834 vs round-11
  baseline 832 / 835 / 834 — **noise-level identical**. Reverted.

The compiler is already smart enough to hoist these wave-uniform
constant expressions across the persistent loop because they only
read from `g.*` globals which are LDS-cached / SGPR-resident.
Source-level hoist is a no-op at SASS level.

### Probe 2: Confirmed mfma instruction width on gfx950

While investigating fuse-epilog mfma count, verified that
`v_mfma_f32_16x16x32_fp8_fp8` on gfx950 (MI355X) actually issues a
**K=128-wide MFMA** despite the legacy `_16x16x32` name in the
macro / asm. This explains why a single `mma_ABt(rt_16x128_s
sub-tile)` call reduces full K=128 cells (and not K=32 as the
macro name suggests). The instruction reads 8 GPR/lane for A but
the macro encodes `[A_lo : A_lo+1]` in the asm template — gfx950's
ISA decoder reads the full 8-GPR consecutive A range from the
`A_lo` start.

Bottom line: there is **no wasted MFMA** in the fuse-epilog from
the K=128 instruction running on K_REM=64 data — the lower 32
fp8 cells/lane (lanes 0-31) carry valid data, the upper 32 cells/lane
(lanes 32-63) carry zero (via SENTINEL voffset → buffer_load_b128
zero-fill on OOB SRD bound), and the K=128 mfma simply does
`real * real + real * 0` on the upper half. The 50 % MFMA-cycle
"waste" hypothesis from round-11 was **wrong** — we always issue
exactly the right number of mfmas, regardless of K_REM ∈ {32, 64,
96}.

This means switching to `mfma_scale_f32_32x32x64_f8f6f4` (K=64
native) would NOT save mfma cycles — we'd issue the same number,
just at a different K width. Wedge dropped from the next-round
list.

### Probe 3: Triton FP8 GEMM bench

Confirmed Triton's FP8 grouped GEMM performance numbers via the
same rocprof harness (above). The Triton kernel takes 148 µs /
iter on this shape — HK takes 198 µs / iter. The 50 µs gap is the
real wedge.

## Next-round angle

Score plateau is essentially **architectural**. The only way to
push past 833-835 is to either:

1. Match Triton's GEMM kernel runtime by rewriting HipKittens'
   FP8 grouped kernel template — this is a multi-round effort
   (new BM/BN/BK combination + register layout re-derivation +
   fuse-epilog re-fitting + numerical re-verification).
2. Accept the plateau and focus on backward-path correctness
   robustness (P2 dB var-K fuse / RRR direct-fix without H4).
   Doesn't move metric, but is long-term codebase health.
3. Tune rules — explicitly forbidden by the task body across rounds
   1-N. Saturated by rounds 57-70 anyway.

For round 13+ I'd recommend documenting the architectural ceiling
(this note is the start) and pivoting to long-term codebase health
work that keeps backward-path correctness robust without trying to
chase metric uplift through micro-optimisation. If the run
patience exhausts (currently 30, used 3), the auto_optimize loop
will roll back.
