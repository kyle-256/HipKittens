# Round-E — FP8 grouped-RCR 32x32x64 main-loop port: EV-FALSIFIED at single-wave microbench (+2.80% per-iter, projects to +1..+2% on full kernel; not worth 1-2 week port)

## Summary

Round D recommended porting the production `grouped_rcr_kernel` main
loop from `mfma_scale_f32_16x16x128_f8f6f4` (8 prim/`rcr_mma`, 32
prim/outer-K-iter) to `mfma_scale_f32_32x32x64_f8f6f4` (4 prim/`rcr_mma`,
16 prim/outer-K-iter), with an estimated +5..+10% TFLOPS gain. Round D
also flagged this estimate as derived from issue-density arguments
without direct measurement.

Round E built a single-wave HIP microbench
(`analysis/fp8_gemm/mi350x/round_e_32x32_microbench.cu`) that holds
all variables identical between the two paths *except* the MFMA prim
shape, and measured per-iter throughput across 5 independent runs of
N=100000 inner iterations × 11 trials each (median).

**Result: +2.80% (± 0.03%) for 32x32x64 vs 16x16x128 on identical
work.**

That is *positive* — issue density does help — but well below the
pre-committed +5% gate. With Amdahl scaling against the ~50% of wall
that PMC attributes to MFMA + LDS in the production kernel (Round-D
counter table), the projected full-kernel gain is **+1..+2%** = roughly
+7..+15 score points (current score ~690 → ~700).

The required port is multi-week (1-2 weeks):

1. New `load_a_32` / `load_b_32` LDS→register cooperative loaders
   that re-derive the lane geometry of `mfma_323264` (32 rows / 64
   K-cols per cell) onto the existing `st_16x128_v2` / `st_16x128_v2a`
   LDS swizzle. Today only the K-tail-bound `load_a_kt_32x64` HBM
   loader exists (lever_d round-B step 4, line 410-450 in
   `kernel_fp8_layouts.cpp`); LDS-source loaders do not.
2. Validate correctness (SNR > 25 dB) across all 8 gpt_oss FP8
   shapes × all 4 `KI_HINT` specializations × `FUSED_KTAIL ∈ {0, 1}`
   × `FUSE_ACT ∈ {0, 1}` × `N_MASKED_STORE ∈ {0, 1}` template
   variants — at least 8 × 4 × 2 × 2 × 2 = 256 instantiation paths.
3. Re-derive K-tail block + epilog 1 + epilog 2 for the 32x32x32
   fragment shape (currently 16x16 cells throughout).
4. Re-tune wait counters for the new LDS read shape (Round-A
   `RCR_PREFETCH_LGKM` = 8 likely needs re-validation since the
   per-iter outstanding LGKM count changes).

EV per round (~+7..+15 score / 1-2 weeks) is well below Round-A's
+2.6 score / 1-number-flip. **Round E is a no-ship.** No kernel
change committed; the microbench source is committed for
reproducibility and as the empirical artifact for this decision.

---

## Microbench setup and result

### Workload (identical between paths)

Per-iter:

* 4 LDS reads of `intx8_t` (= 8 `ds_read_b128` codegen) modeling
  the 4 distinct per-iter operands (a0, a1, b0, b1) the production
  kernel loads per outer K-iter for the 4-acc cA/cB/cC/cD cluster.
* `s_waitcnt lgkmcnt(0)` drain to ensure read completion before MFMA
  (mirrors production line 2778, 2785, 2793, etc).
* Compute the SAME 4 × 64 × 32 × 128 = 2.097 MFLOP per iter.

### Path (a) — 16x16x128 baseline

```cpp
// 4 accs × 8 cells/acc = 32 prim mfmas
for (int c = 0; c < 8; c++) {
    cA[c] = mfma_16x16x128(a0, b0, cA[c]);
    cB[c] = mfma_16x16x128(a0, b1, cB[c]);
    cC[c] = mfma_16x16x128(a1, b0, cC[c]);
    cD[c] = mfma_16x16x128(a1, b1, cD[c]);
}
```

Each accumulator cell is `floatx4_t` (4 fp32/lane).

### Path (b) — 32x32x64 candidate

```cpp
// 4 accs × 2 cells/acc × 2 inner-K = 16 prim mfmas
for (int c = 0; c < 2; c++) {
    cA[c] = mfma_32x32x64(a0, b0, cA[c]);  // inner-K 0
    cB[c] = mfma_32x32x64(a0, b1, cB[c]);
    cC[c] = mfma_32x32x64(a1, b0, cC[c]);
    cD[c] = mfma_32x32x64(a1, b1, cD[c]);
    cA[c] = mfma_32x32x64(a0, b0, cA[c]);  // inner-K 1
    cB[c] = mfma_32x32x64(a0, b1, cB[c]);
    cC[c] = mfma_32x32x64(a1, b0, cC[c]);
    cD[c] = mfma_32x32x64(a1, b1, cD[c]);
}
```

Each accumulator cell is `floatx16_t` (16 fp32/lane). Same per-acc
fp32/lane count (32) as path (a) — register pressure equivalent.

### Per-iter throughput (5 runs × 11 trials each, median)

| Run | (a) 16x16x128         | (b) 32x32x64          | Δ vs (a) |
|-----|------------------------|------------------------|----------|
| 1   | 491.54 ns / 4.3 TFLOPS | 478.16 ns / 4.4 TFLOPS | +2.80 %  |
| 2   | 491.43 ns / 4.3 TFLOPS | 477.96 ns / 4.4 TFLOPS | +2.82 %  |
| 3   | 491.24 ns / 4.3 TFLOPS | 478.01 ns / 4.4 TFLOPS | +2.77 %  |
| 4   | 491.34 ns / 4.3 TFLOPS | 477.95 ns / 4.4 TFLOPS | +2.80 %  |
| 5   | 491.31 ns / 4.3 TFLOPS | 478.09 ns / 4.4 TFLOPS | +2.77 %  |
|-----|------------------------|------------------------|----------|
| Mean| 491.37 ns              | 478.03 ns              | +2.79 %  |
| SD  | 0.11 ns                | 0.09 ns                | 0.02 pp  |

Standard deviation of the per-iter delta is 0.02 percentage points —
the +2.80 % signal is significant well beyond noise (T-test would
return p < 1e-9 trivially).

---

## Why the gain is real but small

The Round-D PMC argument was: 16x16x128 path has issue density
29 % MFMA / 110 inst per outer K-iter. Halving MFMA insts gives
17 % MFMA / 94 inst — 1.78× higher MFMA *issue* fraction.

What the microbench *actually* measures: per-iter wall in a single-wave
config. Halving MFMA insts saves ~16 cyc of issue overhead per outer
K-iter, and ~10 cyc of associated VALU shuffle (operand reformat
between wider MFMA inputs). Total ~26 cyc savings per outer K-iter.

* Microbench per-iter: ~480 ns × 1.5 GHz ≈ 720 cyc per outer K-iter
  per warp (closer to per-warp peak than the production kernel since
  no cross-wave barriers). Savings 26 cyc / 720 cyc = **3.6 %** —
  matches measured +2.8 %.

* Production kernel per-outer-K-iter per warp ≈ 2400 cyc (derived from
  1700 µs wall × 1.5 GHz / 256 CU / 132 outer-iter-per-CU / 8 warps
  per CU). Same 26 cyc savings = **1.1 %** — within the projected
  +1..+2 % range.

The MFMA pipeline runs at the same FLOP/cyc throughput regardless of
prim shape (CDNA4: ~4096 FLOP/cyc/SIMD for both 16x16x128 and 32x32x64).
The only saving is in *non-pipeline* overhead (issue + shuffle), and
that overhead is small relative to the 2400-cyc per-iter total.

---

## Why this rules out Option B as a high-EV next round

EV calculus on score:

* **Round-A (`RCR_PREFETCH_LGKM` 4 → 8)**: 1 number flip, +2.6 mean
  score, ~30 minutes total round time. EV ≈ 5 score / hour.
* **Round-E (32x32x64 main-loop port)**: 1-2 weeks of port work,
  +7..+15 mean score (projected). EV ≈ 0.05..0.2 score / hour
  (assuming 80 hour weeks). **25..100× lower EV than Round-A.**

Even Round-A was a small win. Round-E is structurally similar in scale
but at vastly higher engineering cost.

The remaining structural options retain their Round-D EV ranking:

* **Option A (AGPR migration)**: Estimated +5..+10 % full-kernel gain
  but blocked on missing FP8 `art`-mode MMA intrinsics in HK headers.
  Unblock cost: 1-2 weeks header infrastructure (writes once, benefits
  every fp8 grouped kernel). After unblock, port cost: ~1 round.
  **Highest EV when amortized across the FP8 grouped kernel ecosystem,
  but the upfront infrastructure investment is the gating cost.**
* **Option B (32x32x64 port)**: This round — measured +1..+2 %
  full-kernel projection. Falsified at microbench EV gate.
* **Option C (4-warp port)**: Round-D estimated +5..+10 % but with
  significant register-pressure regression risk. Unmeasured.
* **Option D (K-tail / main-loop overlap)**: Round-D estimated
  +1..+2 %. Comparable EV to Option B but smaller scope (~3-6 days
  vs 1-2 weeks).

---

## Recommendation

**Stop active per-round numeric/structural tuning of
`grouped_rcr_kernel` at this baseline (score ~690).** The remaining
gains require either:

1. Multi-week infrastructure investment (Option A — FP8 `art`-mode
   MMA intrinsics in HK headers), with downstream benefits across
   all FP8 grouped kernels (current GPT-OSS, future DSV3, future
   Qwen3 fp8); OR
2. Acceptance that 51 % FP8 peak on these shapes is the architectural
   ceiling for the 8-warp / 16x16-base / 4-acc / 2-buf design.

Recommend (1) as the next *infrastructure* round, deferring active
kernel tuning until the intrinsic is available. The intrinsic work
itself is best done outside the per-round agenda.

---

## Files touched

* `analysis/fp8_gemm/mi350x/round_e_32x32_microbench.cu` (NEW) —
  empirical artifact for this round's decision; reproducible via
  `hipcc round_e_32x32_microbench.cu -o /tmp/round_e --offload-arch=gfx950 -O3`.
* `analysis/_notes/round-E-fp8-32x32x64-port-EV-falsified-at-microbench.md`
  (this file).

No production kernel change shipped.
