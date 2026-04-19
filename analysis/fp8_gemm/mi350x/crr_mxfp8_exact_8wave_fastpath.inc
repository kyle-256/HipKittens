#pragma once

// R28 Dev D — rectangular BLK_M=256/BLK_N=128 scaffolding gate.
// When MXFP8_RECT_BLK_N=64, the existing CRR exact 8-wave fastpath is
// force-disabled because it hardcodes BLK_N=BLK=256 (scale slab math
// uses (g.n / BLK), load_col_from_v2_st_half uses HB-sized tile). The
// dispatch falls back to the BLK-parametric V1 PQ path. A dedicated
// rect-V2 fastpath is the next-cycle work item — see r28d_findings.md.
//
// Define MXFP8_RECT_BLK_N here to a sentinel if the user did not set
// it (we cannot rely on kernel_mxfp8_layouts.cpp's default having been
// expanded yet because this header is `#pragma once`-included from
// inside kernel_mxfp8_layouts.cpp AFTER the constexpr is established —
// so MXFP8_RECT_BLK_N is always defined here).
#if defined(MXFP8_RECT_BLK_N) && (MXFP8_RECT_BLK_N == 64)
#undef  MXFP8_CRR_EXACT_8WAVE_FAST_ENABLE
#define MXFP8_CRR_EXACT_8WAVE_FAST_ENABLE 0
#endif

// R33 Dev D — sub-RBM operand-tile scaffolding gate.
// Default 64 (no change). When set to 32, the existing CRR exact 8-wave
// fastpath is force-disabled because it hardcodes RBM=64 across:
//   * 8 static_asserts (4 in this file, 4+ in kernel_mxfp8_layouts.cpp)
//   * register tile types A_col_reg = rt_fp8e4m3<BK, RBM, col_l, rt_128x16_s>
//   * the cA/cB/cC/cD accumulator shape rt_fl<RBM, RBN, col_l, rt_16x16_s>
//   * the scale-base offset math (wm * RBM)
//   * crr_a_pack_count = RBM/32 (becomes 1 instead of 2 → loop count change)
//   * the LDS subtile geometry (subtile_inplace<RBM, BK>)
// A parallel sub-RBM kernel template lives in crr_mxfp8_exact_8wave_subrbm_fastpath.inc
// (mirrors R31 Dev A's rect-V2 CRR scaffolding). See r33d_findings.md
// Stage 1a static_assert audit for the full list.
//
// Motivation (per R32 Dev B closure r32b_findings.md §8): A_col_reg = 32
// VGPR/wave at RBM=64; halving to RBM=32 halves it to 16 VGPR/wave,
// removing the 26-lane spill barrier that gates LDS single-buffer
// pipelining. R34+ correctness/perf work.
#ifndef MXFP8_CRR_RBM
#define MXFP8_CRR_RBM 64
#endif
#if (MXFP8_CRR_RBM != 32) && (MXFP8_CRR_RBM != 64)
#error "MXFP8_CRR_RBM must be 32 or 64 (R33 Dev D scaffolding)"
#endif
#if (MXFP8_CRR_RBM == 32)
#undef  MXFP8_CRR_EXACT_8WAVE_FAST_ENABLE
#define MXFP8_CRR_EXACT_8WAVE_FAST_ENABLE 0
#endif

#ifndef MXFP8_CRR_EXACT_8WAVE_FAST_ENABLE
#define MXFP8_CRR_EXACT_8WAVE_FAST_ENABLE 1
#endif
#ifndef MXFP8_CRR_EXACT_PQ_PIPELINE_SCALE_ENABLE
#define MXFP8_CRR_EXACT_PQ_PIPELINE_SCALE_ENABLE 1
#endif
// R47 Dev B: XCD-aware block swizzle for the MXFP8 CRR exact 8-wave fastpath.
// Ports R46 Dev D's RRR swizzle to CRR. CRR has the same row-major bid → (br, bc)
// mapping as the RRR baseline, and the same compute-bound shapes are penalized
// by L2 cache thrashing on the worst CRR cells (70B Down 85.3%, 70B Gate/Up 86.3%).
// 3-run averaged sweep across 7 compute-bound shapes:
//   70B Down +3.36% ★, 8B Q/O +2.69% ★, 70B Gate/Up +2.23% ★, 8B Down +1.13% ★,
//   8B Gate/Up +0.01%, 8192³ -0.04%, 70B Q/O -1.04% (within ±2% within-GPU envelope).
// 4/7 shapes ≥+1%, worst -1.04% within R44 within-GPU envelope — DEFAULT ON.
#ifndef MXFP8_CRR_BLOCK_SWIZZLE
#define MXFP8_CRR_BLOCK_SWIZZLE 1
#endif
#ifndef MXFP8_CRR_BLOCK_SWIZZLE_NUM_XCDS
#define MXFP8_CRR_BLOCK_SWIZZLE_NUM_XCDS 8
#endif
#ifndef MXFP8_CRR_BLOCK_SWIZZLE_GROUP_M
#define MXFP8_CRR_BLOCK_SWIZZLE_GROUP_M 4
#endif
// R27 Dev A H1: cachepolicy bits for V2 scale buffer loads (b128/b64).
// Default 0 = previous behavior. 1=GLC, 2=SLC, 3=GLC|SLC.
// R28 Dev A had auto-defaulted to 2 (SLC) when (N_DIM >= 28672 AND
// K_DIM >= 8192) based on a +2.6% Welch-significant win on
// 4096x28672x8192 CRR. R47 Dev C re-swept all 4 policies × 7
// compute-bound shapes after the R44/R45/R46 baseline shifted upward
// (notably R46 Dev D RRR XCD swizzle and surrounding optimizations)
// and found the SLC default has FLIPPED to a -1.5% loss on the same
// 4096x28672x8192 CRR shape (cp0=2502.54 vs cp2=2464.57 TFLOPS,
// median of 3 reps, ±0.4% spread). p1/p2/p3 either tie or lose
// across all 7 shapes × 3 layouts (no policy ≥ +2% on any cell).
// Conditional auto-gate REMOVED — unconditional default 0.
// User-provided -D still wins.
#ifndef MXFP8_CRR_V2_SCALE_CACHEPOLICY
#define MXFP8_CRR_V2_SCALE_CACHEPOLICY 0
#endif
// R48 Dev B: cooperative B-scale lever — issue B-scale b64 BEFORE A-scale
// b128 in the V2-CRR scale loader. Hypothesis: smaller B-scale transactions
// return faster and unblock MMA dispatch sooner; on large-K shapes
// (70B Down 4096×8192×28672) where B-scale = 7.34 MB > L2 (8 MB per XCD),
// shifting issue order may help VMEM scheduling. Default OFF.
#ifndef MXFP8_CRR_COOPERATIVE_BSCALE
#define MXFP8_CRR_COOPERATIVE_BSCALE 0
#endif

// R50 Dev A: CRR scale prefetch lead-distance lever.
// Baseline (LEAD=0): load_raw_scales(k_pair) is issued at the TOP of each
// even K-iteration; the cA MMA in the same iter consumes those scales after
// only a partial-iter vmcnt drain (LDS loads + barrier).
// Treatment (LEAD=1): pre-issue load_raw_scales(0) once before the K-loop,
// then re-issue load_raw_scales((k+1)>>1) at the END of each ODD iter (after
// cD MMA, the last consumer of the OLD scales). This places ~one full BK of
// vmcnt drain between scale-issue and cA-MMA consumption — targeting the
// CRR ~92% structural ceiling (R47D / R49C §7.2). No shadow scale registers
// are required (the >>16 shift on odd iters operates in-place on the still-
// current packs, and the post-cD issue overwrites them only after the last
// MMA reads complete). The tail block (k=k_iters-2) gets its scales from
// the loop-tail prefetch (last loop iter is always odd given even k_iters),
// so the tail's own load_raw_scales becomes redundant under LEAD=1. Default
// OFF — must clear ship gate (≥+1.5% geomean, no -2% regression, SNR≥45 dB,
// det 3/3) before flipping.
#ifndef MXFP8_CRR_SCALE_LEAD
#define MXFP8_CRR_SCALE_LEAD 0
#endif

#ifndef CRR_USE_EXACT_8WAVE_FASTPATH
#define CRR_USE_EXACT_8WAVE_FASTPATH 1
#endif

#ifndef CRR_EXACT_INTERLEAVE_B1_LDS
#define CRR_EXACT_INTERLEAVE_B1_LDS 1
#endif

// R31 Dev B: LDS single-buffer experiment.
// Default 0 = baseline (As[2][2] + Bs[2][2], 139264 B/block, occ 1 block/CU).
// 1 = collapse outer dim to [1][2], cutting LDS to ~69632 B/block (fits 2
//     blocks/CU on gfx950 163840 B/CU). Loses K-pipelining (no more
//     cross-iter prefetch into the toc slot); this experiment measures
//     whether the occupancy gain pays for the latency-hiding loss.
#ifndef MXFP8_CRR_LDS_SINGLE_BUFFER
#define MXFP8_CRR_LDS_SINGLE_BUFFER 0
#endif

// R32 Dev B: SB pipelining-recovery selector. Used only when SB=1.
//   0 = R31 naive synchronous form (load->barrier->compute->barrier->load)
//       —30% perf (closed by R31 Dev B).
//   1 = early-issue + late-wait. Issue all LDS reads, wait, barrier,
//       issue next-iter VMEM->LDS direct, then run all 4 MMAs while
//       VMEM is in flight. wait vmcnt at end of iter.
//   2 = async-overlap. Like 1 but defers the post-MMA barrier to overlap
//       with the next iter's LDS reads (pre-emptive LDS read, not sync).
//   3 = inner-K interleave. Interleave next-iter VMEM issue and current
//       LDS reads through the MMA chain (MMA pair -> partial VMEM ->
//       MMA pair -> ...). Maximum overlap.
#ifndef MXFP8_CRR_SB_PIPELINE
#define MXFP8_CRR_SB_PIPELINE 0
#endif

#if MXFP8_CRR_LDS_SINGLE_BUFFER
#  define CRR_LDS_OUTER 1
#  define CRR_LDS_TIC   0
#  define CRR_LDS_TOC   0
#else
#  define CRR_LDS_OUTER 2
#  define CRR_LDS_TIC   tic
#  define CRR_LDS_TOC   toc
#endif

#ifndef CRR_EXACT_B1_LDS_INSERT_AFTER
#define CRR_EXACT_B1_LDS_INSERT_AFTER 4
#endif

#ifndef CRR_EXACT_PREFETCH_LGKM
#define CRR_EXACT_PREFETCH_LGKM CRR_PREFETCH_LGKM
#endif

#ifndef CRR_EXACT_STEADY_VMCNT
#define CRR_EXACT_STEADY_VMCNT CRR_STEADY_VMCNT
#endif

#if MXFP8_CRR_EXACT_8WAVE_FAST_ENABLE

#if CRR_A_LDS_REENCODE
#error "CRR_A_LDS_REENCODE not supported for MXFP8 CRR exact 8-wave fast path"
#endif

static_assert(BLK == 256, "MXFP8 exact 8-wave fast path requires BLK=256");
static_assert(BK == 128, "MXFP8 exact 8-wave fast path requires BK=128");
static_assert(WARPS_M == 2, "MXFP8 exact 8-wave fast path requires WARPS_M=2");
static_assert(WARPS_N == 4, "MXFP8 exact 8-wave fast path requires WARPS_N=4");
static_assert(!CRR_ROW_SHARED_TRANSPOSE, "Exact CRR fast path only supports strict shared layout");
static_assert(!CRR_USE_V3_SWIZZLE, "Exact CRR fast path is wired for v2/v2a shared layouts");
static_assert(CRR_A_REG_ROW_LOAD_TRANSPOSE == 1, "Exact CRR fast path expects default A row-load transpose");
static_assert(CRR_A_REG_ROW_LOAD_ALIAS == 1, "Exact CRR fast path expects default A row-load alias");
static_assert(CRR_B_REG_ROW_LOAD_TRANSPOSE == 0, "Exact CRR fast path expects default B col-load path");
static_assert(CRR_B_REG_ROW_LOAD_ALIAS == 0, "Exact CRR fast path expects default B col-load path");
static_assert(CRR_BATCHED_PAIR_MMA == 1, "Exact CRR fast path expects batched pair MMA");
static_assert(CRR_BATCHED_EPILOGUE_MMA == 0, "Exact CRR fast path expects scalar epilogue MMA");

template<int A_PACK_COUNT, int B_PACK_COUNT>
__device__ __forceinline__ void crr_exact_mma_one_scaled(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
    const A_col_reg& a,
    const B_col_reg& b,
    int n,
    int m,
    int k_phase,
    const fp8e8m0_4 (&a_scale_packs)[A_PACK_COUNT],
    const fp8e8m0_4 (&b_scale_packs)[B_PACK_COUNT])
{
    const int a_opsel = (k_phase << 1) | (n & 1);
    const int b_opsel = (k_phase << 1) | (m & 1);
    crr_mma_scaled_dispatch(
        acc,
        a,
        b,
        n,
        m,
        a_opsel,
        b_opsel,
        a_scale_packs[n / 2],
        b_scale_packs[m / 2]);
}

template<int INSERT_AFTER, int A_PACK_COUNT, int B_PACK_COUNT>
__device__ __forceinline__ void crr_exact_cA_with_b1_interleave_fixed_phase(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& cA,
    const A_col_reg& a,
    const B_col_reg& b0,
    B_col_reg& b1,
    const ST_v2& b1_tile,
    int wn,
    const fp8e8m0_4 (&a0_phase)[A_PACK_COUNT],
    const fp8e8m0_4 (&b0_phase)[B_PACK_COUNT])
{
    static_assert(INSERT_AFTER >= 0 && INSERT_AFTER <= 8);
    static_assert(RBM == 64 && RBN == 32);

    __builtin_amdgcn_s_setprio(1);
    if constexpr (INSERT_AFTER == 0) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<0, 0>(cA, a, b0, 0, 0, a0_phase[0], b0_phase[0]);
    if constexpr (INSERT_AFTER == 1) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<0, 1>(cA, a, b0, 0, 1, a0_phase[0], b0_phase[0]);
    if constexpr (INSERT_AFTER == 2) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<1, 0>(cA, a, b0, 1, 0, a0_phase[0], b0_phase[0]);
    if constexpr (INSERT_AFTER == 3) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<1, 1>(cA, a, b0, 1, 1, a0_phase[0], b0_phase[0]);
    if constexpr (INSERT_AFTER == 4) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<0, 0>(cA, a, b0, 2, 0, a0_phase[1], b0_phase[0]);
    if constexpr (INSERT_AFTER == 5) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<0, 1>(cA, a, b0, 2, 1, a0_phase[1], b0_phase[0]);
    if constexpr (INSERT_AFTER == 6) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<1, 0>(cA, a, b0, 3, 0, a0_phase[1], b0_phase[0]);
    if constexpr (INSERT_AFTER == 7) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<1, 1>(cA, a, b0, 3, 1, a0_phase[1], b0_phase[0]);
    if constexpr (INSERT_AFTER == 8) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    __builtin_amdgcn_s_setprio(0);
    CRR_SCHED_BARRIER();
}

template<int K_PHASE, int INSERT_AFTER, int A_PACK_COUNT, int B_PACK_COUNT>
__device__ __forceinline__ void crr_exact_cA_with_b1_interleave_raw_phase(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& cA,
    const A_col_reg& a,
    const B_col_reg& b0,
    B_col_reg& b1,
    const ST_v2& b1_tile,
    int wn,
    const fp8e8m0_4 (&a0_packs)[A_PACK_COUNT],
    const fp8e8m0_4 (&b0_packs)[B_PACK_COUNT])
{
    static_assert(K_PHASE == 0 || K_PHASE == 1);
    static_assert(INSERT_AFTER >= 0 && INSERT_AFTER <= 8);
    static_assert(RBM == 64 && RBN == 32);
    constexpr int lo = K_PHASE * 2;
    constexpr int hi = K_PHASE * 2 + 1;

    __builtin_amdgcn_s_setprio(1);
    if constexpr (INSERT_AFTER == 0) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<lo, lo>(cA, a, b0, 0, 0, a0_packs[0], b0_packs[0]);
    if constexpr (INSERT_AFTER == 1) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<lo, hi>(cA, a, b0, 0, 1, a0_packs[0], b0_packs[0]);
    if constexpr (INSERT_AFTER == 2) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<hi, lo>(cA, a, b0, 1, 0, a0_packs[0], b0_packs[0]);
    if constexpr (INSERT_AFTER == 3) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<hi, hi>(cA, a, b0, 1, 1, a0_packs[0], b0_packs[0]);
    if constexpr (INSERT_AFTER == 4) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<lo, lo>(cA, a, b0, 2, 0, a0_packs[1], b0_packs[0]);
    if constexpr (INSERT_AFTER == 5) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<lo, hi>(cA, a, b0, 2, 1, a0_packs[1], b0_packs[0]);
    if constexpr (INSERT_AFTER == 6) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<hi, lo>(cA, a, b0, 3, 0, a0_packs[1], b0_packs[0]);
    if constexpr (INSERT_AFTER == 7) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<hi, hi>(cA, a, b0, 3, 1, a0_packs[1], b0_packs[0]);
    if constexpr (INSERT_AFTER == 8) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    __builtin_amdgcn_s_setprio(0);
    CRR_SCHED_BARRIER();
}

template<bool PRESHUFFLED_QUANT, int SCALE_VERSION = 1>
__global__ __launch_bounds__(_NUM_THREADS, GEMM_MIN_BLOCKS_PER_CU)
void crr_exact_8wave_scaled_kernel(const layout_globals g)
{
    // R22-B SCALE_VERSION:
    //   1 = original V1 preshuffle layout (per row_group SRDs / b32 loads)
    //   2 = R22 V2-CRR wave-tile reordered layout (per wave-tile SRD,
    //       buffer_load_b128 for A, buffer_load_b64 for B). Mirrors R21's
    //       V2-RCR wiring; A/B scale tensor shapes are identical between
    //       RCR and CRR (both M/N-major scale x K), so the same wave-tile
    //       reorder Python preshuffle (PC=4 for A, PC=2 for B) works.
#if CRR_ROW_SHARED_TRANSPOSE
    using ST_crr_a = ST_row;
    using ST_crr_b = ST_row;
#elif CRR_USE_V3_SWIZZLE
    using ST_crr_a = ST_v3;
    using ST_crr_b = ST_v3;
#else
    using ST_crr_a = ST_v2a;
    using ST_crr_b = ST_v2;
#endif

    constexpr int crr_a_pack_count = RBM / 32;
    constexpr int crr_b_pack_count = (RBN + 31) / 32;

    const int k_iters = g.k / BK;
    const int blocks_per_col = g.n / BLK;
#if MXFP8_CRR_BLOCK_SWIZZLE
    // R47 Dev B: constexpr promotion for swizzle math (mirrors RRR R46 Dev D).
    constexpr int swizzle_blocks_per_row = M_DIM / BLK;
    constexpr int swizzle_blocks_per_col = N_DIM / BLK;
#endif

    __shared__ ST_crr_a As[CRR_LDS_OUTER][2];
    __shared__ ST_crr_b Bs[CRR_LDS_OUTER][2];

    A_col_reg a;
    B_col_reg b0, b1;
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;
    zero(cA);
    zero(cB);
    zero(cC);
    zero(cD);

    int bid = blockIdx.x;
#if MXFP8_CRR_BLOCK_SWIZZLE
    // R47 Dev B: XCD-aware chiplet swizzle + grouped-M swizzle (port of R46 Dev D).
    {
        const int num_wgs = gridDim.x;
        if (num_wgs >= MXFP8_CRR_BLOCK_SWIZZLE_NUM_XCDS &&
            (num_wgs % MXFP8_CRR_BLOCK_SWIZZLE_NUM_XCDS) == 0) {
            bid =
                (bid % MXFP8_CRR_BLOCK_SWIZZLE_NUM_XCDS) *
                    (num_wgs / MXFP8_CRR_BLOCK_SWIZZLE_NUM_XCDS) +
                (bid / MXFP8_CRR_BLOCK_SWIZZLE_NUM_XCDS);
        }
    }
    const int num_wgid_in_group = MXFP8_CRR_BLOCK_SWIZZLE_GROUP_M * swizzle_blocks_per_col;
    const int group_id = bid / num_wgid_in_group;
    const int first_pid_m = group_id * MXFP8_CRR_BLOCK_SWIZZLE_GROUP_M;
    const int group_size_m =
        (first_pid_m + MXFP8_CRR_BLOCK_SWIZZLE_GROUP_M <= swizzle_blocks_per_row)
            ? MXFP8_CRR_BLOCK_SWIZZLE_GROUP_M
            : (swizzle_blocks_per_row - first_pid_m);
    int br, bc;
    if (group_size_m <= 0) {
        br = swizzle_blocks_per_row;
        bc = swizzle_blocks_per_col;
    } else {
        br = first_pid_m + ((bid % num_wgid_in_group) % group_size_m);
        bc = (bid % num_wgid_in_group) / group_size_m;
    }
#else
    const int br = bid / blocks_per_col;
    const int bc = bid % blocks_per_col;
#endif
    const int wm = warpid() / WARPS_N;
    const int wn = warpid() % WARPS_N;

    const int lane_nonk = kittens::laneid() % 16;
    const int lane_kblk = kittens::laneid() / 16;
    const uint32_t lane_scale_byte_offset =
        (static_cast<uint32_t>(lane_kblk) << 6) |
        (static_cast<uint32_t>(lane_nonk) << 2);

    auto crr_scale_a_base = [&](int half) {
        return br * BLK + half * (BLK / 2) + wm * RBM;
    };
    auto crr_scale_b_base = [&](int half) {
        return bc * BLK + half * (BLK / 2) + wn * RBN;
    };

    fp8e8m0_4 a0_scale_packs[crr_a_pack_count];
    fp8e8m0_4 a1_scale_packs[crr_a_pack_count];
    fp8e8m0_4 b0_scale_packs[crr_b_pack_count];
    fp8e8m0_4 b1_scale_packs[crr_b_pack_count];
    const uint8_t* a0_scale_row_bases[crr_a_pack_count];
    const uint8_t* a1_scale_row_bases[crr_a_pack_count];
    const uint8_t* b0_scale_row_bases[crr_b_pack_count];
    const uint8_t* b1_scale_row_bases[crr_b_pack_count];
#if MXFP8_CRR_EXACT_PQ_PIPELINE_SCALE_ENABLE
    i32x4 a0p0_srsrc = {}, a0p1_srsrc = {}, a1p0_srsrc = {}, a1p1_srsrc = {};
    i32x4 b0p0_srsrc = {}, b1p0_srsrc = {};
#endif
    // V2 wave-tile SRDs: 1 SRD per wave for A, 1 per wave for B.
    i32x4 a_v2_srsrc = {}, b_v2_srsrc = {};
    if constexpr (PRESHUFFLED_QUANT) {
        #pragma unroll
        for (int p = 0; p < crr_a_pack_count; ++p) {
            a0_scale_row_bases[p] = preshuffled_scale_row_base_ptr(
                g.a_scale, (crr_scale_a_base(0) + p * 32) >> 5);
            a1_scale_row_bases[p] = preshuffled_scale_row_base_ptr(
                g.a_scale, (crr_scale_a_base(1) + p * 32) >> 5);
        }
        #pragma unroll
        for (int p = 0; p < crr_b_pack_count; ++p) {
            b0_scale_row_bases[p] = preshuffled_scale_row_base_ptr(
                g.b_scale, (crr_scale_b_base(0) + p * 32) >> 5);
            b1_scale_row_bases[p] = preshuffled_scale_row_base_ptr(
                g.b_scale, (crr_scale_b_base(1) + p * 32) >> 5);
        }
#if MXFP8_CRR_EXACT_PQ_PIPELINE_SCALE_ENABLE
        if constexpr (SCALE_VERSION == 1) {
            auto make_scale_srd = [](const void* ptr) -> i32x4 {
                i32x4 srd = std::bit_cast<i32x4>(
                    make_buffer_resource(
                        static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(ptr)),
                        0xFFFFFFFFu,
                        0x00110000u
                    )
                );
                srd[0] = __builtin_amdgcn_readfirstlane(srd[0]);
                srd[1] = __builtin_amdgcn_readfirstlane(srd[1]);
                srd[2] = __builtin_amdgcn_readfirstlane(srd[2]);
                srd[3] = __builtin_amdgcn_readfirstlane(srd[3]);
                return srd;
            };
            a0p0_srsrc = make_scale_srd(a0_scale_row_bases[0]);
            a1p0_srsrc = make_scale_srd(a1_scale_row_bases[0]);
            b0p0_srsrc = make_scale_srd(b0_scale_row_bases[0]);
            b1p0_srsrc = make_scale_srd(b1_scale_row_bases[0]);
            if constexpr (crr_a_pack_count > 1) {
                a0p1_srsrc = make_scale_srd(a0_scale_row_bases[1]);
                a1p1_srsrc = make_scale_srd(a1_scale_row_bases[1]);
            }
        } else { // SCALE_VERSION == 2
            // V2-CRR: per-wave-tile slab SRDs (mirrors R21 V2-RCR).
            // A side: pack_count=4, num_slabs_a = (M/BLK)*WARPS_M.
            //   slab_bytes_a = 4 * 32 * padded_k_blocks = 128 * padded_k_blocks.
            //   slab(br, wm) = br * WARPS_M + wm.
            // B side: pack_count=2, num_slabs_b = (N/BLK)*WARPS_N.
            //   slab_bytes_b = 2 * 32 * padded_k_blocks = 64 * padded_k_blocks.
            //   slab(bc, wn) = bc * WARPS_N + wn.
            const int padded_k_blocks =
                (g.k / 32 + 7) & ~7; // round up to multiple of 8
            const size_t slab_bytes_a = 128u * padded_k_blocks;
            const size_t slab_bytes_b = 64u * padded_k_blocks;
            const size_t slab_idx_a = static_cast<size_t>(br) * WARPS_M + wm;
            const size_t slab_idx_b = static_cast<size_t>(bc) * WARPS_N + wn;
            const uint8_t* a_v2_base =
                reinterpret_cast<const uint8_t*>(g.a_scale.raw_ptr) +
                slab_idx_a * slab_bytes_a;
            const uint8_t* b_v2_base =
                reinterpret_cast<const uint8_t*>(g.b_scale.raw_ptr) +
                slab_idx_b * slab_bytes_b;
            auto make_v2_srd = [](const void* ptr, uint32_t bytes) -> i32x4 {
                i32x4 srd = std::bit_cast<i32x4>(
                    make_buffer_resource(
                        static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(ptr)),
                        bytes,
                        0x00110000u
                    )
                );
                srd[0] = __builtin_amdgcn_readfirstlane(srd[0]);
                srd[1] = __builtin_amdgcn_readfirstlane(srd[1]);
                srd[2] = __builtin_amdgcn_readfirstlane(srd[2]);
                srd[3] = __builtin_amdgcn_readfirstlane(srd[3]);
                return srd;
            };
            a_v2_srsrc = make_v2_srd(a_v2_base,
                                     static_cast<uint32_t>(slab_bytes_a));
            b_v2_srsrc = make_v2_srd(b_v2_base,
                                     static_cast<uint32_t>(slab_bytes_b));
        }
#endif
    }
    auto load_raw_scales = [&](int k_pair) __attribute__((always_inline)) {
#if MXFP8_CRR_EXACT_PQ_PIPELINE_SCALE_ENABLE
        if constexpr (PRESHUFFLED_QUANT) {
            if constexpr (SCALE_VERSION == 2) {
                // R22-B V2-CRR: per (k_pair, lane), one b128 fetches all 4 A
                // packs ({a0p0, a1p0, a0p1, a1p1}) and one b64 fetches both
                // B packs ({b0p0, b1p0}). Strides match V2 spec:
                //   A (PC=4): voff = lane_kblk*256 + lane_nonk*16, soff = k_pair*1024
                //   B (PC=2): voff = lane_kblk*128 + lane_nonk*8,  soff = k_pair*512
                // CRR has no separate warmup/tail helper (load_raw_scales is
                // called from main loop and from the k=k_iters-2 / tail
                // blocks alike), so this single patch covers all paths.
                const uint32_t a_voff =
                    (static_cast<uint32_t>(lane_kblk) << 8) |
                    (static_cast<uint32_t>(lane_nonk) << 4);
                const uint32_t a_soff = static_cast<uint32_t>(k_pair) << 10;
                const uint32_t b_voff =
                    (static_cast<uint32_t>(lane_kblk) << 7) |
                    (static_cast<uint32_t>(lane_nonk) << 3);
                const uint32_t b_soff = static_cast<uint32_t>(k_pair) << 9;
#if MXFP8_CRR_COOPERATIVE_BSCALE
                // R48 Dev B: issue B-scale b64 BEFORE A-scale b128.
                const uint64_t b_raw =
                    llvm_amdgcn_raw_buffer_load_b64(b_v2_srsrc, b_voff, b_soff, MXFP8_CRR_V2_SCALE_CACHEPOLICY);
                __builtin_amdgcn_sched_barrier(0);
                const __uint128_t a_raw =
                    llvm_amdgcn_raw_buffer_load_b128(a_v2_srsrc, a_voff, a_soff, MXFP8_CRR_V2_SCALE_CACHEPOLICY);
                b0_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(b_raw      ));
                b1_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(b_raw >> 32));
                a0_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw      ));
                a1_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw >> 32));
                if constexpr (crr_a_pack_count > 1) {
                    a0_scale_packs[1] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw >> 64));
                    a1_scale_packs[1] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw >> 96));
                }
#else
                const __uint128_t a_raw =
                    llvm_amdgcn_raw_buffer_load_b128(a_v2_srsrc, a_voff, a_soff, MXFP8_CRR_V2_SCALE_CACHEPOLICY);
                a0_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw      ));
                a1_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw >> 32));
                if constexpr (crr_a_pack_count > 1) {
                    a0_scale_packs[1] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw >> 64));
                    a1_scale_packs[1] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw >> 96));
                }
                const uint64_t b_raw =
                    llvm_amdgcn_raw_buffer_load_b64(b_v2_srsrc, b_voff, b_soff, MXFP8_CRR_V2_SCALE_CACHEPOLICY);
                b0_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(b_raw      ));
                b1_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(b_raw >> 32));
#endif
                return;
            }
            const uint32_t soff = static_cast<uint32_t>(k_pair) << 8;
            a0_scale_packs[0] = std::bit_cast<fp8e8m0_4>(llvm_amdgcn_raw_buffer_load_b32(a0p0_srsrc, lane_scale_byte_offset, soff, 0));
            a1_scale_packs[0] = std::bit_cast<fp8e8m0_4>(llvm_amdgcn_raw_buffer_load_b32(a1p0_srsrc, lane_scale_byte_offset, soff, 0));
            if constexpr (crr_a_pack_count > 1) {
                a0_scale_packs[1] = std::bit_cast<fp8e8m0_4>(llvm_amdgcn_raw_buffer_load_b32(a0p1_srsrc, lane_scale_byte_offset, soff, 0));
                a1_scale_packs[1] = std::bit_cast<fp8e8m0_4>(llvm_amdgcn_raw_buffer_load_b32(a1p1_srsrc, lane_scale_byte_offset, soff, 0));
            }
            b0_scale_packs[0] = std::bit_cast<fp8e8m0_4>(llvm_amdgcn_raw_buffer_load_b32(b0p0_srsrc, lane_scale_byte_offset, soff, 0));
            b1_scale_packs[0] = std::bit_cast<fp8e8m0_4>(llvm_amdgcn_raw_buffer_load_b32(b1p0_srsrc, lane_scale_byte_offset, soff, 0));
            return;
        }
#endif
#pragma unroll
        for (int p = 0; p < crr_a_pack_count; ++p) {
            if constexpr (PRESHUFFLED_QUANT) {
                a0_scale_packs[p] = load_scale_pair_pack_16x128_preshuffled_from_row_base(
                    a0_scale_row_bases[p], k_pair, lane_scale_byte_offset);
                a1_scale_packs[p] = load_scale_pair_pack_16x128_preshuffled_from_row_base(
                    a1_scale_row_bases[p], k_pair, lane_scale_byte_offset);
            } else {
                a0_scale_packs[p] = load_scale_pair_pack_16x128(
                    g.a_scale, crr_scale_a_base(0) + p * 32, k_pair, lane_nonk, lane_kblk);
                a1_scale_packs[p] = load_scale_pair_pack_16x128(
                    g.a_scale, crr_scale_a_base(1) + p * 32, k_pair, lane_nonk, lane_kblk);
            }
        }
#pragma unroll
        for (int p = 0; p < crr_b_pack_count; ++p) {
            if constexpr (PRESHUFFLED_QUANT) {
                b0_scale_packs[p] = load_scale_pair_pack_16x128_preshuffled_from_row_base(
                    b0_scale_row_bases[p], k_pair, lane_scale_byte_offset);
                b1_scale_packs[p] = load_scale_pair_pack_16x128_preshuffled_from_row_base(
                    b1_scale_row_bases[p], k_pair, lane_scale_byte_offset);
            } else {
                b0_scale_packs[p] = load_scale_pair_pack_16x128(
                    g.b_scale, crr_scale_b_base(0) + p * 32, k_pair, lane_nonk, lane_kblk);
                b1_scale_packs[p] = load_scale_pair_pack_16x128(
                    g.b_scale, crr_scale_b_base(1) + p * 32, k_pair, lane_nonk, lane_kblk);
            }
        }
    };

    constexpr int bptA = ST_crr_a::underlying_subtile_bytes_per_thread;
    constexpr int bpmA = bptA * _NUM_THREADS;
    constexpr int mptA = ST_crr_a::rows * ST_crr_a::cols * sizeof(fp8e4m3) / bpmA;
    uint32_t soA[mptA];
#if CRR_ROW_SHARED_TRANSPOSE
    prefill_transpose_swizzled_offsets<_NUM_THREADS>(As[0][0], g.a, soA);
#else
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);
#endif

    constexpr int bptB = ST_crr_b::underlying_subtile_bytes_per_thread;
    constexpr int bpmB = bptB * _NUM_THREADS;
    constexpr int mptB = ST_crr_b::rows * ST_crr_b::cols * sizeof(fp8e4m3) / bpmB;
    uint32_t soB[mptB];
#if CRR_ROW_SHARED_TRANSPOSE
    prefill_transpose_swizzled_offsets<_NUM_THREADS>(Bs[0][0], g.b, soB);
#else
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);
#endif

#if CRR_ROW_SHARED_TRANSPOSE
    auto a_co = [&](int s, int k) -> coord<ST_crr_a> { return {0, 0, s, k}; };
    auto b_co = [&](int s, int k) -> coord<ST_crr_b> { return {0, 0, s, k}; };
    auto global_load_a = [&](ST_crr_a& tile, int s, int k) {
        load_transpose<_NUM_THREADS>(tile, g.a, a_co(s, k), soA);
    };
    auto global_load_b = [&](ST_crr_b& tile, int s, int k) {
        load_transpose<_NUM_THREADS>(tile, g.b, b_co(s, k), soB);
    };
    auto load_a = [&](A_col_reg& dst, ST_crr_a& tile, int wi) {
        A_row_reg tmp;
        auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
        load(tmp, sub);
        transpose(dst, tmp);
    };
    auto load_b = [&](B_col_reg& dst, ST_crr_b& tile, int wi) {
        B_row_reg tmp;
        auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
        load(tmp, sub);
        transpose(dst, tmp);
    };
#else
    auto a_co = [&](int s, int k) -> coord<ST_crr_a> { return {0, 0, k, s}; };
    auto b_co = [&](int s, int k) -> coord<ST_crr_b> { return {0, 0, k, s}; };
    auto global_load_a = [&](ST_crr_a& tile, int s, int k) {
        G::load(tile, g.a, a_co(s, k), soA);
    };
    auto global_load_b = [&](ST_crr_b& tile, int s, int k) {
        G::load(tile, g.b, b_co(s, k), soB);
    };
    auto load_a = [&](A_col_reg& dst, ST_crr_a& tile, int wi) {
#if CRR_USE_V3_SWIZZLE
        load_col_from_v3_st(dst, tile, wi * RBM);
#else
#if CRR_A_REG_ROW_LOAD_TRANSPOSE
#if CRR_A_REG_ROW_LOAD_ALIAS
        load_col_from_v2a_st(dst, tile, wi * RBM);
#else
        A_row_reg tmp;
        auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
        load(tmp, sub);
        transpose(dst, tmp);
#endif
#else
        load_col_from_v2a_st(dst, tile, wi * RBM);
#endif
#endif
    };
    auto load_b = [&](B_col_reg& dst, ST_crr_b& tile, int wi) {
#if CRR_USE_V3_SWIZZLE
        load_col_from_v3_st(dst, tile, wi * RBN);
#else
#if CRR_B_REG_ROW_LOAD_TRANSPOSE
#if CRR_B_REG_ROW_LOAD_ALIAS
        load_col_from_v2_st(dst, tile, wi * RBN);
#else
        B_row_reg tmp;
        auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
        load(tmp, sub);
        transpose(dst, tmp);
#endif
#else
        load_col_from_v2_st(dst, tile, wi * RBN);
#endif
#endif
    };
#endif

#if MXFP8_CRR_LDS_SINGLE_BUFFER
    // R31 Dev B: single-buffered LDS path. No tic/toc, no cross-iter prefetch.
    // R32 Dev B: SB_PIPELINE selector for pipelining-recovery variants.
    int tic = 0; (void)tic;
    int toc = 0; (void)toc;
    // Initial load of k=0 only.
    global_load_b(Bs[0][0], bc * 2, 0);
    global_load_a(As[0][0], br * 2, 0);
    global_load_b(Bs[0][1], bc * 2 + 1, 0);
    global_load_a(As[0][1], br * 2 + 1, 0);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

#if MXFP8_CRR_SB_PIPELINE == 0
    // R31 naive synchronous form (closed; kept for parity bench)
    TK_PRAGMA_UNROLL(CRR_MAIN_UNROLL)
    for (int k = 0; k < k_iters - 1; k++) {
        if ((k & 1) == 0) {
            load_raw_scales(k >> 1);
        } else {
            #pragma unroll
            for (int g = 0; g < crr_a_pack_count; g++) {
                a0_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a0_scale_packs[g]) >> 16);
                a1_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a1_scale_packs[g]) >> 16);
            }
            #pragma unroll
            for (int g = 0; g < crr_b_pack_count; g++) {
                b0_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b0_scale_packs[g]) >> 16);
                b1_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b1_scale_packs[g]) >> 16);
            }
        }
        load_b(b0, Bs[0][0], wn);
        load_b(b1, Bs[0][1], wn);
        load_a(a, As[0][0], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_from_packs_fixed_phase(cA, a, b0, a0_scale_packs, b0_scale_packs);
        crr_mma_scaled_from_packs_fixed_phase(cB, a, b1, a0_scale_packs, b1_scale_packs);
        CRR_MMA_END();
        load_a(a, As[0][1], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_from_packs_fixed_phase(cC, a, b0, a1_scale_packs, b0_scale_packs);
        crr_mma_scaled_from_packs_fixed_phase(cD, a, b1, a1_scale_packs, b1_scale_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();
        global_load_b(Bs[0][0], bc * 2, k + 1);
        global_load_a(As[0][0], br * 2, k + 1);
        global_load_b(Bs[0][1], bc * 2 + 1, k + 1);
        global_load_a(As[0][1], br * 2 + 1, k + 1);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }
#elif MXFP8_CRR_SB_PIPELINE == 1
    // R32 Dev B Approach 1: early-issue + late-wait.
    //   1. Issue all 4 LDS reads (b0, b1, a, a_next).
    //   2. waitcnt lgkm + barrier (LDS read drain & wave-sync).
    //   3. Issue all 4 next-iter VMEM->LDS direct (writes async, in flight).
    //   4. Run all 4 MMAs (cA, cB, cC, cD) — overlaps with VMEM.
    //   5. waitcnt vmcnt + barrier at end of iter.
    // This requires holding 2 A_col_regs concurrently (a, a_next).
    A_col_reg a_next;
    TK_PRAGMA_UNROLL(CRR_MAIN_UNROLL)
    for (int k = 0; k < k_iters - 1; k++) {
        if ((k & 1) == 0) {
            load_raw_scales(k >> 1);
        } else {
            #pragma unroll
            for (int g = 0; g < crr_a_pack_count; g++) {
                a0_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a0_scale_packs[g]) >> 16);
                a1_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a1_scale_packs[g]) >> 16);
            }
            #pragma unroll
            for (int g = 0; g < crr_b_pack_count; g++) {
                b0_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b0_scale_packs[g]) >> 16);
                b1_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b1_scale_packs[g]) >> 16);
            }
        }
        // 1. Issue all four LDS reads upfront.
        load_b(b0, Bs[0][0], wn);
        load_b(b1, Bs[0][1], wn);
        load_a(a, As[0][0], wm);
        load_a(a_next, As[0][1], wm);
        // 2. Drain LDS, sync waves before next-iter LDS writes start.
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        // 3. Issue next-iter VMEM->LDS direct (asynchronous).
        global_load_b(Bs[0][0], bc * 2, k + 1);
        global_load_a(As[0][0], br * 2, k + 1);
        global_load_b(Bs[0][1], bc * 2 + 1, k + 1);
        global_load_a(As[0][1], br * 2 + 1, k + 1);
        // 4. MMA chain — VMEM in flight overlaps with compute.
        CRR_MMA_BEGIN();
        crr_mma_scaled_from_packs_fixed_phase(cA, a, b0, a0_scale_packs, b0_scale_packs);
        crr_mma_scaled_from_packs_fixed_phase(cB, a, b1, a0_scale_packs, b1_scale_packs);
        CRR_MMA_END();
        CRR_MMA_BEGIN();
        crr_mma_scaled_from_packs_fixed_phase(cC, a_next, b0, a1_scale_packs, b0_scale_packs);
        crr_mma_scaled_from_packs_fixed_phase(cD, a_next, b1, a1_scale_packs, b1_scale_packs);
        CRR_MMA_END();
        // 5. Drain VMEM (LDS writes complete), sync waves before next-iter LDS reads.
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }
#elif MXFP8_CRR_SB_PIPELINE == 2
    // R32 Dev B Approach 2: async-overlap (LDS-bind aware).
    // Like Approach 1 but interleaves the next-iter VMEM issue between the
    // two MMA pairs to give earlier issue (more time-in-flight before
    // we hit vmcnt at end). Also defers the s_waitcnt lgkmcnt to right
    // before each MMA group instead of one big upfront drain.
    A_col_reg a_next;
    TK_PRAGMA_UNROLL(CRR_MAIN_UNROLL)
    for (int k = 0; k < k_iters - 1; k++) {
        if ((k & 1) == 0) {
            load_raw_scales(k >> 1);
        } else {
            #pragma unroll
            for (int g = 0; g < crr_a_pack_count; g++) {
                a0_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a0_scale_packs[g]) >> 16);
                a1_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a1_scale_packs[g]) >> 16);
            }
            #pragma unroll
            for (int g = 0; g < crr_b_pack_count; g++) {
                b0_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b0_scale_packs[g]) >> 16);
                b1_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b1_scale_packs[g]) >> 16);
            }
        }
        // Issue all 4 LDS reads.
        load_b(b0, Bs[0][0], wn);
        load_b(b1, Bs[0][1], wn);
        load_a(a, As[0][0], wm);
        load_a(a_next, As[0][1], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        // Issue half the next-iter VMEM (B side first).
        global_load_b(Bs[0][0], bc * 2, k + 1);
        global_load_b(Bs[0][1], bc * 2 + 1, k + 1);
        // First MMA pair.
        CRR_MMA_BEGIN();
        crr_mma_scaled_from_packs_fixed_phase(cA, a, b0, a0_scale_packs, b0_scale_packs);
        crr_mma_scaled_from_packs_fixed_phase(cB, a, b1, a0_scale_packs, b1_scale_packs);
        CRR_MMA_END();
        // Issue rest of next-iter VMEM (A side).
        global_load_a(As[0][0], br * 2, k + 1);
        global_load_a(As[0][1], br * 2 + 1, k + 1);
        // Second MMA pair.
        CRR_MMA_BEGIN();
        crr_mma_scaled_from_packs_fixed_phase(cC, a_next, b0, a1_scale_packs, b0_scale_packs);
        crr_mma_scaled_from_packs_fixed_phase(cD, a_next, b1, a1_scale_packs, b1_scale_packs);
        CRR_MMA_END();
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }
#elif MXFP8_CRR_SB_PIPELINE == 3
    // R32 Dev B Approach 3: inner-K interleave. Issue both halves of
    // the next-iter VMEM in two batches across the MMA chain. Read As[0][1]
    // BEFORE first MMA pair to permit issuing the As[0][0] write right
    // after it (and Bs[*][*] writes immediately, since b0/b1 reads were
    // already drained by the lgkm-wait above). Then after MMA1 issue
    // As[0][1] write. Reuses single A_col_reg `a` (no a_next).
    TK_PRAGMA_UNROLL(CRR_MAIN_UNROLL)
    for (int k = 0; k < k_iters - 1; k++) {
        if ((k & 1) == 0) {
            load_raw_scales(k >> 1);
        } else {
            #pragma unroll
            for (int g = 0; g < crr_a_pack_count; g++) {
                a0_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a0_scale_packs[g]) >> 16);
                a1_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a1_scale_packs[g]) >> 16);
            }
            #pragma unroll
            for (int g = 0; g < crr_b_pack_count; g++) {
                b0_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b0_scale_packs[g]) >> 16);
                b1_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b1_scale_packs[g]) >> 16);
            }
        }
        // Read first three LDS operands.
        load_b(b0, Bs[0][0], wn);
        load_b(b1, Bs[0][1], wn);
        load_a(a, As[0][0], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        // First-pair MMA (uses As[0][0]).
        CRR_MMA_BEGIN();
        crr_mma_scaled_from_packs_fixed_phase(cA, a, b0, a0_scale_packs, b0_scale_packs);
        crr_mma_scaled_from_packs_fixed_phase(cB, a, b1, a0_scale_packs, b1_scale_packs);
        CRR_MMA_END();
        // Read As[0][1] (last LDS read for this iter).
        load_a(a, As[0][1], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        // All LDS reads drained; sync waves before any VMEM->LDS writes.
        __builtin_amdgcn_s_barrier();
        // Issue ALL next-iter VMEM (4 tiles).
        global_load_b(Bs[0][0], bc * 2, k + 1);
        global_load_b(Bs[0][1], bc * 2 + 1, k + 1);
        global_load_a(As[0][0], br * 2, k + 1);
        global_load_a(As[0][1], br * 2 + 1, k + 1);
        // Second-pair MMA (uses As[0][1]) — overlaps with VMEM in flight.
        CRR_MMA_BEGIN();
        crr_mma_scaled_from_packs_fixed_phase(cC, a, b0, a1_scale_packs, b0_scale_packs);
        crr_mma_scaled_from_packs_fixed_phase(cD, a, b1, a1_scale_packs, b1_scale_packs);
        CRR_MMA_END();
        // Drain VMEM (LDS writes complete), sync waves.
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }
#else
#error "MXFP8_CRR_SB_PIPELINE must be 0, 1, 2, or 3"
#endif
    // Final iter: data from last prefetch of k=k_iters-1 is in [0].
    {
        const int k = k_iters - 1;
        if ((k & 1) == 0) {
            load_raw_scales(k >> 1);
        } else {
            #pragma unroll
            for (int g = 0; g < crr_a_pack_count; g++) {
                a0_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a0_scale_packs[g]) >> 16);
                a1_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a1_scale_packs[g]) >> 16);
            }
            #pragma unroll
            for (int g = 0; g < crr_b_pack_count; g++) {
                b0_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b0_scale_packs[g]) >> 16);
                b1_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b1_scale_packs[g]) >> 16);
            }
        }
        load_b(b0, Bs[0][0], wn);
        load_b(b1, Bs[0][1], wn);
        load_a(a, As[0][0], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_from_packs_fixed_phase(cA, a, b0, a0_scale_packs, b0_scale_packs);
        crr_mma_scaled_from_packs_fixed_phase(cB, a, b1, a0_scale_packs, b1_scale_packs);
        CRR_MMA_END();
        load_a(a, As[0][1], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_from_packs_fixed_phase(cC, a, b0, a1_scale_packs, b0_scale_packs);
        crr_mma_scaled_from_packs_fixed_phase(cD, a, b1, a1_scale_packs, b1_scale_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();
    }
#else
    int tic = 0, toc = 1;
    global_load_b(Bs[tic][0], bc * 2, 0);
    global_load_a(As[tic][0], br * 2, 0);
    global_load_b(Bs[tic][1], bc * 2 + 1, 0);
    global_load_a(As[tic][1], br * 2 + 1, 0);

    if (wm == 1) {
        __builtin_amdgcn_s_barrier();
    }
    TK_WAIT_VMCNT(CRR_INIT0_VMCNT);
    __builtin_amdgcn_s_barrier();

    global_load_b(Bs[toc][0], bc * 2, 1);
    global_load_a(As[toc][0], br * 2, 1);
    global_load_b(Bs[toc][1], bc * 2 + 1, 1);

    TK_WAIT_VMCNT(CRR_INIT1_VMCNT);
    __builtin_amdgcn_s_barrier();

#if MXFP8_CRR_SCALE_LEAD
    // R50 Dev A: pre-issue scales for k_pair=0 once before the K-loop. The
    // steady-state loop then re-issues at the END of each odd iter for the
    // NEXT k_pair (see post-cD block below). This grants ~one extra BK of
    // vmcnt drain time before the consuming cA MMA.
    if (k_iters > 0) {
        load_raw_scales(0);
    }
#endif

    TK_PRAGMA_UNROLL(CRR_MAIN_UNROLL)
    for (int k = 0; k < k_iters - 2; k++, tic ^= 1, toc ^= 1) {
#if MXFP8_CRR_SCALE_LEAD
        // LEAD=1: scales for even iters are already in regs (preloaded once
        // before the loop, and re-issued at end of each odd iter). Odd iters
        // shift in place (uses second half of the same k_pair's b128/b64
        // load), same as baseline.
        if ((k & 1) == 1) {
            #pragma unroll
            for (int g = 0; g < crr_a_pack_count; g++) {
                a0_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a0_scale_packs[g]) >> 16);
                a1_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a1_scale_packs[g]) >> 16);
            }
            #pragma unroll
            for (int g = 0; g < crr_b_pack_count; g++) {
                b0_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b0_scale_packs[g]) >> 16);
                b1_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b1_scale_packs[g]) >> 16);
            }
        }
#else
        if ((k & 1) == 0) {
            load_raw_scales(k >> 1);
        } else {
            #pragma unroll
            for (int g = 0; g < crr_a_pack_count; g++) {
                a0_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a0_scale_packs[g]) >> 16);
                a1_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a1_scale_packs[g]) >> 16);
            }
            #pragma unroll
            for (int g = 0; g < crr_b_pack_count; g++) {
                b0_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b0_scale_packs[g]) >> 16);
                b1_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b1_scale_packs[g]) >> 16);
            }
        }
#endif

        load_b(b0, Bs[tic][0], wn);
#if !CRR_EXACT_INTERLEAVE_B1_LDS
        load_b(b1, Bs[tic][1], wn);
#endif
        load_a(a, As[tic][0], wm);
        global_load_a(As[toc][1], br * 2 + 1, k + 1);
        TK_WAIT_LGKM(CRR_EXACT_PREFETCH_LGKM);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
#if CRR_EXACT_INTERLEAVE_B1_LDS
        crr_exact_cA_with_b1_interleave_fixed_phase<CRR_EXACT_B1_LDS_INSERT_AFTER, crr_a_pack_count, crr_b_pack_count>(
            cA, a, b0, b1, Bs[tic][1], wn, a0_scale_packs, b0_scale_packs);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        crr_mma_scaled_from_packs_fixed_phase(cB, a, b1, a0_scale_packs, b1_scale_packs);
        __builtin_amdgcn_s_setprio(0);
        CRR_SCHED_BARRIER();
#else
        CRR_MMA_BEGIN();
        crr_mma_scaled_from_packs_fixed_phase(cA, a, b0, a0_scale_packs, b0_scale_packs);
        crr_mma_scaled_from_packs_fixed_phase(cB, a, b1, a0_scale_packs, b1_scale_packs);
        CRR_MMA_END();
#endif
        CRR_STEADY_MID_BARRIER();
        CRR_SCHED_BARRIER();

        load_a(a, As[tic][1], wm);
        global_load_a(As[tic][0], br * 2, k + 2);
        global_load_b(Bs[tic][1], bc * 2 + 1, k + 2);
        TK_WAIT_VMCNT(CRR_EXACT_STEADY_VMCNT);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_from_packs_fixed_phase(cC, a, b0, a1_scale_packs, b0_scale_packs);
        crr_mma_scaled_from_packs_fixed_phase(cD, a, b1, a1_scale_packs, b1_scale_packs);
        CRR_MMA_END();
#if MXFP8_CRR_SCALE_LEAD
        // R50 Dev A: lead-distance scale prefetch. After cD MMA on an odd
        // iter, the OLD scale packs are no longer needed; issue the NEXT
        // k_pair's scales now so vmcnt has the entire next iter's LDS-load
        // window to drain before the cA MMA consumes them. We unconditionally
        // issue here when k is odd; the tail block (k=k_iters-2, even) then
        // expects scales already loaded — its own load_raw_scales is gated
        // off below. k_iters is always even for the production compute-bound
        // shapes (k = K/128 with K ∈ {4096,8192,14336,28672}), so the LAST
        // loop iter is k=k_iters-3 which is ODD — its post-cD prefetch fills
        // scales for the tail's k_pair = (k_iters-2)>>1.
        if ((k & 1) == 1) {
            load_raw_scales((k + 1) >> 1);
        }
#endif
        __builtin_amdgcn_s_barrier();
        global_load_b(Bs[tic][0], bc * 2, k + 2);
    }

    {
        const int k = k_iters - 2;
#if !MXFP8_CRR_SCALE_LEAD
        load_raw_scales(k >> 1);
#endif

        load_b(b0, Bs[tic][0], wn);
        load_a(a, As[tic][0], wm);
        global_load_a(As[toc][1], br * 2 + 1, k + 1);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase<0>(cA, a, b0, a0_scale_packs, b0_scale_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();
        CRR_SCHED_BARRIER();

        load_b(b1, Bs[tic][1], wn);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase<0>(cB, a, b1, a0_scale_packs, b1_scale_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();

        load_a(a, As[tic][1], wm);
        TK_WAIT_VMCNT(CRR_EPILOGUE_VMCNT);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase<0>(cC, a, b0, a1_scale_packs, b0_scale_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();

        load_b(b0, Bs[toc][0], wn);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase<0>(cD, a, b1, a1_scale_packs, b1_scale_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();
        CRR_SCHED_BARRIER();
        tic ^= 1;
        toc ^= 1;
    }

    {
        const auto b0_keep = b0;
        load_a(a, As[tic][0], wm);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase<1>(cA, a, b0_keep, a0_scale_packs, b0_scale_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();

        load_b(b1, Bs[tic][1], wn);
        const auto b1_keep = b1;
        __builtin_amdgcn_s_barrier();
        CRR_SCHED_BARRIER();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase<1>(cB, a, b1_keep, a0_scale_packs, b1_scale_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();

        load_a(a, As[tic][1], wm);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase<1>(cC, a, b0_keep, a1_scale_packs, b0_scale_packs);
        crr_mma_scaled_phase<1>(cD, a, b1_keep, a1_scale_packs, b1_scale_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();
    }
#endif // MXFP8_CRR_LDS_SINGLE_BUFFER

    if (wm == 0) {
        __builtin_amdgcn_s_barrier();
    }

    if constexpr (PRESHUFFLED_QUANT) {
        mul(cA, cA, g.scale);
        mul(cB, cB, g.scale);
        mul(cC, cC, g.scale);
        mul(cD, cD, g.scale);
    }

    store(g.c, cA, {0, 0, br * WARPS_M * 2 + wm, bc * WARPS_N * 2 + wn});
    store(g.c, cB, {0, 0, br * WARPS_M * 2 + wm, bc * WARPS_N * 2 + WARPS_N + wn});
    store(g.c, cC, {0, 0, br * WARPS_M * 2 + WARPS_M + wm, bc * WARPS_N * 2 + wn});
    store(g.c, cD, {0, 0, br * WARPS_M * 2 + WARPS_M + wm, bc * WARPS_N * 2 + WARPS_N + wn});
}

__host__ inline bool crr_can_use_exact_8wave_scaled(const layout_globals& g)
{
    return g.m == M_DIM && g.n == N_DIM && g.k == K_DIM;
}

template<bool PRESHUFFLED_QUANT>
__host__ inline void dispatch_crr_exact_8wave_scaled(const layout_globals& g)
{
    const dim3 grid((g.m / BLK) * (g.n / BLK));
    crr_exact_8wave_scaled_kernel<PRESHUFFLED_QUANT, 1><<<grid, dim3(_NUM_THREADS), 0, g.stream>>>(g);
}

// R22-B milestone-2: V2-CRR scale layout dispatch (only valid when
// PRESHUFFLED_QUANT=true and the host has preshuffled the scale tensors via
// preshuffle_scale_matrix_mfma16_v2_rcr_a/b — same wave-tile reorder works
// for CRR because the scale tensor shapes are identical between RCR and CRR).
template<bool PRESHUFFLED_QUANT>
__host__ inline void dispatch_crr_exact_8wave_scaled_v2(const layout_globals& g)
{
    const dim3 grid((g.m / BLK) * (g.n / BLK));
    crr_exact_8wave_scaled_kernel<PRESHUFFLED_QUANT, 2><<<grid, dim3(_NUM_THREADS), 0, g.stream>>>(g);
}

#endif
