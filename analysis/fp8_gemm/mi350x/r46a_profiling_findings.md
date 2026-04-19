# R46 Dev A -- MXFP8 vs FP8 RCR 8-Wave Fastpath Kernel Comparison

**Date:** 2026-04-19
**Branch:** feat/mxfp8-only
**GPU:** MI355X (gfx950)
**Scope:** Identify exactly where the MXFP8 scale overhead comes from in the RCR 8-wave fastpath vs the FP8 (no-scale) RCR 8-wave fastpath, and why 70B Gate/Up (N=28672) and 70B Down (K=28672) have the worst MXFP8/FP8 ratios (~90%).

---

## 1. Kernel Architecture Summary

Both kernels share identical tile geometry:
- **BLK=256, BK=128, WARPS_M=2, WARPS_N=4** (8 warps/block = 512 threads)
- **RBM=64 (per-warp M), RBN=32 (per-warp N)**
- Double-buffered A[2][2] and B[2][2] shared tiles (each half-block = 128x128 fp8)
- 4-quadrant accumulator: cA(wm, wn), cB(wm, wn+WARPS_N), cC(wm+WARPS_M, wn), cD(wm+WARPS_M, wn+WARPS_N)
- Each K-iteration: load 128 K-elements from LDS into registers, perform 4 MMA operations (one per quadrant)

### FP8 Kernel (`rcr_exact_8wave_kernel`)

Per K-iteration (steady state), the sequence is:
1. Load B[tic][0] subtile -> register b0 (ds_read_b128 from LDS)
2. Load A[tic][0] subtile -> register a (ds_read_b128 from LDS)
3. Prefetch A[toc][1] from global -> LDS (buffer_load...lds)
4. `mma_ABt(cA, a, b0, cA)` -- 1 MFMA instruction (16x16x128, unscaled)
5. Load B[tic][1] subtile -> register b1
6. Prefetch B[tic][0] next-K from global -> LDS
7. `mma_ABt(cB, a, b1, cB)` -- 1 MFMA instruction
8. Load A[tic][1] subtile -> register a
9. Prefetch A[tic][0] next-K from global -> LDS
10. `mma_ABt(cC, a, b0, cC)` -- 1 MFMA instruction
11. Prefetch B[tic][1] next-K from global -> LDS
12. `mma_ABt(cD, a, b1, cD)` -- 1 MFMA instruction

**Per K-iteration: 4 MFMA instructions (unscaled `v_mfma_f32_16x16x128_f8f6f4`)**
Each MFMA computes one 16x16 output tile contributing to one quadrant of the 64x32 wave tile.

### MXFP8 Kernel (`rcr_exact_8wave_scaled_kernel<true, 2>`)

Per K-iteration (steady state), identical data movement PLUS scale handling:
1-12: Same LDS reads + global prefetches as FP8
PLUS:
- Scale load: 1x `buffer_load_dwordx4` (b128, 16 bytes, A-side scales for all 4 packs) per K-pair
- Scale load: 1x `buffer_load_dwordx2` (b64, 8 bytes, B-side scales for 2 packs) per K-pair
- Phase remapping: `v_lshrrev_b32` (shift by 16 for k_phase=1) -- avoided with HOIST_HI_ENABLE=1 via opsel encoding
- Each `mma_ABt` replaced with `v_mfma_scale_f32_16x16x128_f8f6f4` -- the SCALED variant

**Per K-iteration: 4 scaled MFMA instructions + amortized scale loads**

The scaled MFMA is the SAME instruction class -- each `v_mfma_scale_f32_16x16x128_f8f6f4` takes an A-scale and B-scale VGPR operand in addition to the standard A, B, D operands. HOWEVER:

Each MFMA in the MXFP8 kernel decomposes differently because RBM=64 means 4 row groups (rows 0-15, 16-31, 32-47, 48-63) and RBN=32 means 2 column groups. So per quadrant:

**Per quadrant per K-iter: 4 rows x 2 cols = 8 MFMA instructions** (not 1!)

The `rcr_mma_scaled_from_packs_fixed_phase_impl` calls `rcr_mma_scaled_from_packs_fixed_phase_row<ROW>` for ROW=0..3, each issuing 2 `rcr_exact_mfma_scale_builtin_inplace` (for the 2 B column sub-tiles).

**Per K-iteration: 4 quadrants x 8 MFMAs = 32 scaled MFMAs**

The FP8 kernel's `mma_ABt(cA, a, b0, cA)` also decomposes into the same 8 unscaled MFMA sub-instructions internally (the register tile is 64x32 in 16x16 subtiles = 4x2 = 8 MFMAs per call). So both kernels issue **32 MFMAs per K-iteration**.

The critical difference: the MXFP8 kernel's 32 MFMAs are `v_mfma_scale_f32_16x16x128_f8f6f4` (with scale operands), while the FP8 kernel's 32 MFMAs are `v_mfma_f32_16x16x128_f8f6f4` (no scale operands).

### SASS Inventory (V2-RCR, 4096^3 build, from R31 Dev C SASS audit)

Total kernel SASS (MXFP8 V2-RCR): 1314 lines / 4310 SASS lines (full file including prologue/epilogue)
- 384 `v_mfma_scale_f32_16x16x128_f8f6f4` instructions total
- 392 `ds_read*` (LDS tile reads)
- 112 `buffer_load*` total (96 tile fills via buffer_load_dwordx4...lds + 16 scale loads via buffer_load_dwordx4/x2 offen, no lds flag)
- 0 `ds_write` / `ds_store` (ZERO LDS writes -- all tile fills use gfx950 buffer_load_dword_lds direct path)

For K=4096 (32 K-iterations at BK=128): 384/32 = **12 MFMAs/iteration** ... wait, 384 total MFMAs for the entire kernel including prologue + epilogue.

Analyzing more carefully:
- K=4096, BK=128 => k_iters = 32
- Main loop: k_iters-2 = 30 iterations, each with 32 MFMAs = 960
- Pre-tail: 1 iteration with 32 MFMAs = 32
- Tail: 1 iteration with 32 MFMAs = 32
- But the loop is unrolled by 2 (pair loop), so: 15 k_pairs * 2 * 32 + 32 + 32 = 1024
- Wait, 384 total MFMAs in SASS. This means for 4096^3, k_iters = 32, and the compiler unrolled by 2, giving 16 iterations (k_pairs=15 main + 1 remainder + tail). The 384 MFMA count in the binary is the STATIC instruction count, not dynamic count. With loop unroll by 2: body has 2*32=64 MFMAs, pre-tail has 32, tail has 32. But the loop body appears once in binary (it loops), plus pre-tail and tail are peeled. So: 1 loop body * 64 + pre-tail 32 + tail 32 = 128 static MFMAs? No, 384 is more.

Actually: the K-pair loop (`do_k_iter_body<0>` and `do_k_iter_body<1>`) is templated on K_PHASE, so the compiler generates 2 bodies. With KPAIR_LOOP_ENABLE=1 and unroll=2 (the k_pairs loop itself may have some unrolling), plus the pre-tail (k-2) block and tail (k-1) block.

Let me re-examine: the SASS shows 384 MFMA instructions in total. Given:
- Main k_pair loop body: do_k_iter_body<0> (32 MFMAs) + do_k_iter_body<1> (32 MFMAs) = 64 MFMAs per k_pair
- Pre-tail block (k_iters-2): 32 MFMAs (4 quadrants, each 8 MFMAs)
- Tail block (k_iters-1): 32 MFMAs
- Plus pre-tail has extra LDS-only block at end combining cC+cD: 16 MFMAs

But 384 doesn't divide cleanly. The key point is that the loop body runs dynamically k_pairs times (15 for K=4096), so dynamic MFMA count = 15*64 + 32 + 32 = 1024 per wave.

---

## 2. Resource Usage Comparison

| Resource | FP8 RCR 8-wave | MXFP8 RCR V2 8-wave | Delta |
|---|---:|---:|---:|
| VGPRs | 252 | 246 | -6 (MXFP8 lower!) |
| SGPRs | 30 | 52 | +22 |
| AGPRs | 0 | 0 | 0 |
| Scratch/lane | 0 | 0 | 0 |
| VGPR Spill | 0 | 0 | 0 |
| LDS bytes/block | 131072 | 131072 | 0 |
| Occupancy | 2 waves/SIMD | 2 waves/SIMD | **Same** |
| `__launch_bounds__` | (512, 2) | (512, 2) | Same |

Source: r45_shape_comparison build logs for 8192^3 shape.

**Critical finding: Occupancy is identical.** Both kernels run at 2 waves/SIMD (= 1 block/CU for the 8-wave blocks). Occupancy is limited by LDS (131072 B < 163840 B gfx950 limit for 1 block; 2 blocks = 262144 B which overflows).

The MXFP8 kernel actually uses FEWER VGPRs (246 vs 252) because the scaled MFMA variant `v_mfma_scale_f32_16x16x128_f8f6f4` encodes scale operands in the instruction itself (VGPR source operands), and the `rcr_exact_acc` struct uses a flat `rcr_exact_floatx4_t regs[8]` array for accumulation that may be more register-efficient than the kittens rt_fl tile representation.

---

## 3. Scale Overhead Breakdown

### 3A. Scale Load Overhead

The MXFP8 kernel adds **2 global memory loads per K-pair** (every 2 K-iterations):
- A-side: `buffer_load_dwordx4` (b128, 16 bytes) fetches all 4 A scale packs via V2 SRD
- B-side: `buffer_load_dwordx2` (b64, 8 bytes) fetches both B scale packs via V2 SRD

Per K-iteration amortized: **1 b128 + 1 b64 = 24 bytes per K-pair = 12 bytes/K-iter**.

SASS confirms these are issued ~30 instructions before the first consuming MFMA, providing ample VMEM latency hiding. The compiler naturally schedules them at the top of the K-pair body.

### 3B. Phase Remapping Overhead

MXFP8 scales are packed as k_pair (2 phases in a 32-bit word, 16 bits per phase). With `MXFP8_RCR_EXACT_PQ_HOIST_HI_ENABLE=1` (the default/production path), the phase selection is encoded directly in the MFMA `opsel` field:

```c
constexpr int OPSEL_A_FULL = (OPSEL_A & 1) | ((K_PHASE & 1) << 1);
constexpr int OPSEL_B_FULL = (OPSEL_B & 1) | ((K_PHASE & 1) << 1);
```

With `KPAIR_LOOP_ENABLE=1`, `do_k_iter_body` is templated on `K_PHASE`, so this becomes a **compile-time constant**. This means:
- **Zero v_lshrrev_b32 instructions** for phase remapping in the production path
- The phase is baked into the opsel encoding of each MFMA instruction

### 3C. Instruction-Level Overhead: Scaled vs Unscaled MFMA

The core question: **does `v_mfma_scale_f32_16x16x128_f8f6f4` take more cycles than `v_mfma_f32_16x16x128_f8f6f4`?**

On gfx950 (CDNA4/MI355X), the scaled variant (`v_mfma_scale_f32_16x16x128_f8f6f4`) takes additional VGPR source operands for scale_a and scale_b. The per-32-element scale multiply is fused into the MFMA pipeline, but:

1. **The instruction encoding is wider** -- the scaled MFMA has 2 extra VGPR source operands (scale_a, scale_b) and 2 opsel fields. This means more instruction bytes to fetch/decode.

2. **The MFMA execution likely takes the same number of compute cycles** on the matrix core itself (the scale multiply can be pipelined with the dot-product accumulation), but the **instruction issue rate may be lower** due to:
   - Additional source operand reads from the VGPR file (scale_a, scale_b need to be read)
   - Longer instruction decode

3. **Scale operand reuse**: each scale VGPR is read 8 times per quadrant (once per MFMA sub-tile). The compiler can schedule consecutive MFMAs sharing the same scale operand back-to-back, but the VGPR read ports may create minor stalls.

### 3D. K-Loop Structure Overhead

The MXFP8 K-loop has additional synchronization and control flow:

1. **K-pair grouping**: MXFP8 processes K-iterations in pairs (`k_pair`), with scale loads issued per k_pair and `do_k_iter_body` invoked twice per pair (phase 0 and phase 1). This adds:
   - An inner `for k_pair` loop with a `load_scale_buffer(k_pair)` call at top
   - Two template-instantiated bodies per k_pair

2. **`ensure_scale_packs` caching**: a `cached_k_pair` variable with branch to skip redundant loads (likely optimized away in the V2 pipeline-scale path since `load_scale_buffer` is called explicitly)

3. **SRD setup**: The V2 path computes A and B scale SRDs (`i32x4 a_v2_srsrc, b_v2_srsrc`) in the prologue, requiring `readfirstlane` to broadcast base addresses from VGPR to SGPR for buffer resource descriptors. This is a one-time cost.

### 3E. Epilogue Overhead

MXFP8 epilogue has additional work:
- FP8: direct `store(g.c, cA, ...)` for each quadrant
- MXFP8: `rcr_exact_acc_to_rt(c_store, cA, g.scale)` which multiplies each accumulator by a scalar scale, then stores. This adds 8 f32 multiplies per quadrant (32 total) before the 4 global stores.

---

## 4. Quantifying the Overhead Sources

### 4A. Steady-State Cycle Budget Analysis

For a compute-bound shape (8192^3, K=8192, k_iters=64):

**FP8 RCR**: 
- 62 steady-state K-iterations (k_iters - 2) + 2 epilogue iterations
- Per iteration: 32 MFMA instructions + LDS reads + global prefetches + barriers
- The MFMA v_mfma_f32_16x16x128_f8f6f4 at 16x16x128 throughput on gfx950

**MXFP8 V2-RCR**:
- Same iteration count
- Per iteration: 32 scaled MFMA instructions + LDS reads + global prefetches + barriers + (amortized) 2 scale loads per 2 iterations + SRD address computation

The overhead sources and estimated impact:

| Source | Per K-iter instructions | Fraction of FP8 iter | Status |
|---|---|---|---|
| Scale loads (amortized) | ~0.5 buffer_load (b128) + ~0.5 buffer_load (b64) | ~3% | Overlapped with MFMAs |
| Scaled vs unscaled MFMA | 32 instructions (same count) | 0% (same count) | Issue-rate delta TBD |
| Phase remap (v_lshrrev) | 0 (eliminated by opsel encoding) | 0% | Optimized away |
| Epilogue scale mul | 32 f32 muls (one-time, end of kernel) | <0.1% | Negligible |
| K-pair control flow | ~2-4 s_branch/s_cmp per k_pair | <0.5% | Negligible |

### 4B. Where Does 5-10% Go?

The dominant cost is almost certainly **the scaled MFMA instruction's longer issue latency or lower throughput** compared to unscaled MFMA on gfx950. Even a 5% slower MFMA issue rate across all 32 instructions/iteration fully explains the observed gap.

Supporting evidence:
1. Occupancy is identical (2 waves/SIMD)
2. VGPR count is actually lower for MXFP8 (246 vs 252)
3. LDS size is identical (131072)
4. Scale loads are well-overlapped (SASS shows 30-instruction gap)
5. Phase remapping is free (opsel encoding)
6. The only remaining variable is the MFMA instruction itself

---

## 5. Why 70B Gate/Up (N=28672) and 70B Down (K=28672) Are Worst

### 5A. 70B Gate/Up: M=4096, N=28672, K=8192, RCR ratio = 90.6%

**Tile geometry**: (M/BLK) x (N/BLK) = 16 x 112 = **1792 blocks**
**K iterations**: K/BK = 8192/128 = 64

**Wave fill**: 1792 blocks / 304 CUs = 5.89 waves/CU (at occ=1 block/CU).
- Wave 1: 304 CUs busy, 1488 blocks remaining
- Wave 2: 304 CUs busy, 1184 remaining
- Wave 3: 304 CUs busy, 880 remaining
- Wave 4: 304 CUs busy, 576 remaining
- Wave 5: 304 CUs busy, 272 remaining
- Wave 6: 272 CUs busy, **32 CUs idle (10.5% idle)**

The last dispatch wave has 272/304 = 89.5% utilization. This is a **tail effect** that penalizes both FP8 and MXFP8, but penalizes MXFP8 MORE because:

1. **The absolute TFLOPS gap is amplified by tail inefficiency**: FP8 achieves 2933 TF, MXFP8 achieves 2632 TF. The per-CU compute efficiency loss from scaled MFMA (say 5%) is applied across all waves, but the tail effect means those last 272 CUs are 100% compute-bound (no prefetch overlap from the next wave), so the per-MFMA overhead is fully exposed in the tail.

2. **N=28672 means more B-side scale loads per K-iteration**: With N=28672, each block computes a 256x256 output tile, but the wave tile N-dimension (RBN=32) means each warp processes 32 N-elements. The B-side scale array is larger (28672/32 = 896 row-groups), increasing L2 cache pressure for scale fetches. However, each warp only loads its own 2 B scale packs per K-pair, so the per-warp scale volume is identical -- the issue is L2 cache set conflicts across CTAs.

3. **L2 scale cache thrashing**: With 1792 CTAs each loading unique B-scale rows, the total B-scale working set is 28672 * (K/32) = 28672 * 256 = 7.3 MB. For A-scales: 4096 * 256 = 1 MB. The MI355X L2 cache is likely insufficient to hold all scale data for 1792 concurrent CTAs, causing scale-load latency to increase. The FP8 kernel has NO scale loads, so it is immune to this cache pressure.

### 5B. 70B Down: M=4096, N=8192, K=28672, RCR ratio = 90.4%

**Tile geometry**: (M/BLK) x (N/BLK) = 16 x 32 = **512 blocks**
**K iterations**: K/BK = 28672/128 = **224 iterations**

**Wave fill**: 512 / 304 = 1.68 waves/CU.
- Wave 1: 304 CUs busy, 208 remaining
- Wave 2: 208 CUs busy, **96 CUs idle (31.6% idle!)**

The tail effect here is brutal -- 31.6% of CUs are idle in the last wave.

But more importantly, the **K-dimension is 3.5x longer** than 8192^3:
- 224 K-iterations = 112 K-pairs, each requiring a scale load
- **112 scale loads per wave (A-side: 112 x buffer_load_dwordx4 = 1792 bytes; B-side: 112 x buffer_load_dwordx2 = 896 bytes)**
- Total scale traffic per wave: 2688 bytes = 2.6 KB (modest, but the latency matters)

The key issue for K=28672:

1. **Scale load latency is no longer perfectly hidden**: With K=28672, the K-loop runs for 224 iterations. The scale loads issue every 2 iterations. In the FP8 kernel, the K-loop is purely MFMA + LDS read + global tile prefetch. Adding 2 buffer_load instructions per K-pair means 2 extra VMEM operations that compete with the 4 global tile prefetch loads per iteration on the VMEM return path. At K=28672, the tile data volume per CTA is: 256*28672 = 7.3 MB (A) + 256*28672 = 7.3 MB (B). Scale data is 2.6 KB/wave -- negligible in volume but it adds VMEM queue contention.

2. **The per-MFMA overhead accumulates linearly with K**: If each scaled MFMA is ~5% slower than unscaled, the total time delta is 5% * 224 * 32 MFMAs/iter = 5% overhead on a longer kernel. The 90.4% ratio (9.6% gap) is explained by: 5% MFMA overhead + ~5% tail effect from 31.6% CU idleness in the last wave.

Actually, the tail effect affects BOTH kernels equally, so the MXFP8/FP8 ratio should be relatively constant regardless of tile count. The worse ratio for K=28672 shapes must come from:

3. **Scale traffic competes with tile traffic for memory bandwidth**: With K=28672, the tile data volume per CTA is much larger. The memory subsystem is working harder to feed the tiles. Adding scale loads (even at 12 bytes/K-iter average) adds to the memory traffic and can cause cache line conflicts. The FP8 kernel has strictly less memory traffic.

4. **Longer K-loop means more scale SRD address arithmetic**: The `a_soff = k_pair << 10` and `b_soff = k_pair << 9` offset computations happen every K-pair. These are scalar instructions (fast), but they contribute to instruction-stream pressure.

### 5C. Summary: Why Large N or Large K Hurts MXFP8 Disproportionately

| Factor | 8192^3 (95%) | 70B Gate/Up (90.6%) | 70B Down (90.4%) |
|---|---|---|---|
| Tile count | 1024 | 1792 | 512 |
| Wave fill last wave | 304/304 (100%) | 272/304 (89.5%) | 208/304 (68.4%) |
| K iterations | 64 | 64 | 224 |
| K pairs (scale loads) | 32 | 32 | 112 |
| Scale working set | 2x 8192*256 = 4 MB | A: 1MB + B: 7.3MB | A: 1MB + B: 2MB + long K chain |
| Primary overhead source | MFMA latency delta | MFMA + L2 scale pressure | MFMA + VMEM contention |

---

## 6. Optimization Opportunities

### 6A. Already Optimized / Closed Levers

These have been investigated and closed in prior cycles:

1. **Scale LDS staging** -- CLOSED (R27, R30 Dev C, R31 Dev C). V2 path has zero LDS round-trip for scales (SASS confirms 0 ds_write). Nothing to double-buffer.

2. **VGPR prefetch second-buffer** -- CLOSED (R31 Dev C). Adding next-iter scale VGPRs causes 312 VGPR spill, -92% regression. The 4-quadrant accumulator design cannot accommodate parallel scale-pack live ranges.

3. **Phase remapping** -- OPTIMIZED (HOIST_HI_ENABLE + KPAIR_LOOP_ENABLE). Phase selection is compile-time via opsel encoding. Zero runtime v_lshrrev_b32 cost.

4. **Scale load scheduling** -- NATURALLY OPTIMIZED. Compiler issues scale loads ~30 instructions before first consumer. VMEM latency is hidden.

5. **V2 preshuffle layout** -- OPTIMIZED. Single b128 fetch per K-pair loads all 4 A scale packs; single b64 loads both B scale packs. Minimal load count.

### 6B. Potentially Open Optimization Opportunities

1. **Scaled MFMA throughput on gfx950**: If the hardware scaled MFMA truly has lower throughput than unscaled, this is a hardware ceiling. However, it may be possible to:
   - Verify whether the issue is instruction decode bandwidth (instruction is wider) or actual matrix-core throughput
   - If decode-bound: reduce instruction count per K-iteration via loop unrolling or instruction packing changes
   - If matrix-core-bound: this is a hardware limitation with no software mitigation

2. **Reduce K-loop overhead for large K**: For K=28672 (224 iterations):
   - Current loop unroll is 2 (K-pair). Consider unrolling by 4 or 8 to amortize loop control overhead
   - BUT: higher unroll increases code size and may hurt instruction cache. The current 384 static MFMAs already span significant I-cache working set
   - **Estimated gain: <1%** (loop overhead is already minimal)

3. **Scale load cachepolicy tuning**: Currently `MXFP8_RCR_V2_SCALE_CACHEPOLICY=0` (default). For shapes with large N (28672), setting SLC (skip L2 cache) for scale loads might reduce L2 pollution and improve tile-data cache hit rate. Conversely, for long-K shapes, keeping scales in L2 (default) is beneficial since the same scale is reused.
   - **Estimated gain: 0-2% for N=28672 shapes** (speculative; needs bench)

4. **Dispatch geometry for 4096-M shapes**: The 512-block (70B Down) and 1792-block (70B Gate/Up) shapes have significant tail waste. Persistent CTA scheduling (`MXFP8_RCR_V2_PERSISTENT`) was explored in R31 Dev D but only helped when grid > 304 CUs. For 512 blocks, persistent dispatch with 304-CU grid would give 1.68 loops/CU, which still has tail waste. For 1792 blocks, persistent dispatch with 608 grid (2x occ) would give 1792/608 = 2.95 loops, with 3rd pass doing 576 blocks on 608 grid (5.3% tail waste vs 10.5% current).
   - **This improves both FP8 and MXFP8 equally** -- it doesn't close the MXFP8/FP8 gap specifically.

5. **LDS reduction to enable occ=2 blocks/CU**: Current LDS=131072 B, gfx950 limit=163840 B. Two blocks need 262144 B (overflow). Reducing LDS to 81920 B (e.g., single-buffer one of A/B) would enable 2 blocks/CU. This is a major structural change:
   - Would require removing double-buffering for one side (A or B)
   - Net effect unclear: double-buffering hides VMEM latency; removing it may hurt compute-bound shapes
   - **This affects both FP8 and MXFP8 equally** unless the MXFP8 kernel benefits more from higher occupancy (unlikely since both are compute-bound)

6. **Accumulator layout optimization**: The `rcr_exact_acc` uses a flat `rcr_exact_floatx4_t regs[8]` array (8 x float4 = 32 floats = 128 bytes = 32 VGPRs per quadrant, 128 VGPRs for 4 quadrants). The FP8 kernel uses `rt_fl<64, 32, col_l, rt_16x16_s>` for each quadrant, which may have different register allocation characteristics.
   - The `rcr_exact_acc` approach appears to give better VGPR utilization (246 vs 252 for FP8) -- this is already an advantage.

7. **Scale-aware tile scheduling**: For shapes with large N (28672), CTAs that share the same A-tile rows could be scheduled to the same CU cluster to improve A-scale cache reuse. This requires a custom CTA scheduler (not available in standard HIP dispatch).

---

## 7. Root Cause Summary

The MXFP8/FP8 gap of 5-14% across shapes is composed of:

| Component | Estimated contribution | Applies to |
|---|---|---|
| Scaled MFMA instruction overhead (decode/issue/execute) | **3-7%** | All shapes uniformly |
| Scale VMEM loads competing with tile loads | **0-3%** | Worse for large K (28672) |
| Scale data L2 cache pressure | **0-3%** | Worse for large N (28672) |
| SRD setup + address arithmetic | **<0.5%** | All shapes |
| Epilogue scale multiply | **<0.1%** | All shapes |

The 8192^3 shape at 95.0% (5% gap) is closest to the pure scaled-MFMA overhead since it has balanced dimensions, good wave fill (1024/304 = 3.37 waves, last wave 112/304 = 37% idle but same for both kernels), and moderate K.

The 70B Gate/Up at 90.6% (9.4% gap) adds L2 scale cache pressure from N=28672 (7.3 MB B-scale working set).

The 70B Down at 90.4% (9.6% gap) adds VMEM contention from K=28672 (112 scale load pairs competing with 224 iterations of tile loads).

---

## 8. Recommendation for Closing the Gap to 95%

The **only lever likely to close the gap for the worst shapes** is reducing the per-MFMA overhead, which requires:

1. **Micro-benchmarking scaled vs unscaled MFMA throughput in isolation** to establish the hardware ceiling. If scaled MFMA is inherently 5% slower, the 8192^3 RCR result (95.0%) is already at ceiling, and the 70B shapes need the secondary overheads (L2 pressure, VMEM contention) addressed.

2. **For 70B Gate/Up (N=28672)**: Scale cachepolicy experiment (`MXFP8_RCR_V2_SCALE_CACHEPOLICY=2` for SLC) to reduce L2 pollution. Target: recover 2-3% from current 90.6% to ~93%.

3. **For 70B Down (K=28672)**: The RCR ratio is 90.4% which is 4.6pp worse than 8192^3 (95.0%). With 224 K-iterations, the scale VMEM overhead has more iterations to accumulate. A larger K-unroll factor or explicit scale prefetch into the MFMA latency shadow could help, but R31 Dev C showed VGPR prefetch catastrophically fails. The only viable approach may be **reducing the K-loop instruction count** by further SASS-level scheduling optimization.

4. **Accept the hardware ceiling**: If scaled MFMA throughput is genuinely 5% lower, then the 8192^3 RCR at 95.0% is already at the theoretical maximum, and the goal of ">=95% for all shapes" requires either hardware changes or a fundamentally different MXFP8 algorithm (e.g., fusing scale multiplication into a separate pass, which would be worse).

---

## 9. Files Referenced

- FP8 RCR fastpath: `analysis/fp8_gemm/mi350x/rcr_exact_8wave_fastpath.inc`
- MXFP8 RCR kernel: `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp` (line 2354, `rcr_exact_8wave_scaled_kernel`)
- FP8 dispatcher: `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`
- V2-RCR SASS disassembly: `analysis/fp8_gemm/mi350x/r31c_v2_rcr.s` (1314 lines, 384 scaled MFMAs)
- R31 Dev C scale-pipeline audit: `analysis/fp8_gemm/mi350x/r31c_findings.md`
- R31 Dev C SASS inventory: `analysis/fp8_gemm/mi350x/r30c_sass_inventory.log`
- Build logs (FP8): `analysis/fp8_gemm/mi350x/r45_shape_comparison/*_fp8_build.log`
- Build logs (MXFP8): `analysis/fp8_gemm/mi350x/r45_shape_comparison/*_mxfp8_build.log`
- Runtime results: `analysis/fp8_gemm/mi350x/r45_full_results/SUMMARY.txt`
