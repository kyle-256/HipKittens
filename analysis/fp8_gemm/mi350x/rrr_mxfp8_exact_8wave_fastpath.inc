#pragma once

#ifndef MXFP8_RRR_EXACT_8WAVE_FAST_ENABLE
#define MXFP8_RRR_EXACT_8WAVE_FAST_ENABLE 1
#endif
#ifndef MXFP8_RRR_EXACT_PQ_PIPELINE_SCALE_ENABLE
#define MXFP8_RRR_EXACT_PQ_PIPELINE_SCALE_ENABLE 0
#endif
// R27 Dev A H1: cachepolicy bits for V2 scale buffer loads (b128/b64).
#ifndef MXFP8_RRR_V2_SCALE_CACHEPOLICY
#define MXFP8_RRR_V2_SCALE_CACHEPOLICY 0
#endif
// R46 Dev D: XCD-aware block swizzle for the MXFP8 RRR exact 8-wave fastpath.
// Improves L2 cache locality for N-heavy shapes (Gate/Up: N=28672, K=8192)
// where the default row-major block order causes B-tile cache thrashing.
// Sweep across 7 compute-bound shapes (R46) showed:
//   70B Gate/Up +6.51% ★, 70B Down +2.99% ★, 8B Q/O +1.43%, 8B Down +1.14%,
//   70B Q/O +0.19%, 8192³ -0.25%, 8B Gate/Up -0.74% (worst, within noise).
// Net positive for production prefill — enabled by default.
#ifndef MXFP8_RRR_BLOCK_SWIZZLE
#define MXFP8_RRR_BLOCK_SWIZZLE 1
#endif
#ifndef MXFP8_RRR_BLOCK_SWIZZLE_NUM_XCDS
#define MXFP8_RRR_BLOCK_SWIZZLE_NUM_XCDS 8
#endif
#ifndef MXFP8_RRR_BLOCK_SWIZZLE_GROUP_M
#define MXFP8_RRR_BLOCK_SWIZZLE_GROUP_M 4
#endif

// R48 Dev E: K-pair unroll factor for the main K-pair loop.
// Default = 0 (no #pragma unroll directive — preserves R48 baseline behavior
// where MXFP8 RRR keeps the K-loop intact at K=4096 while FP8 RRR fully unrolls).
// Set to a positive integer to emit `#pragma unroll N` on the K-pair loop.
// At K=4096, k_pairs=15 — values >= 15 force full unroll.
// Set to 1 to disable unrolling explicitly.
//
// Lever motivation (r48d_findings.md §2.2):
//   ISA dump showed FP8 RRR @ K=4096 fully unrolled (1024 MFMAs straight-line)
//   but MXFP8 RRR keeps loop intact, costing ~3 loop-control instr per K-pair
//   (s_cbranch + s_cmp + s_add). Estimated +1-2pp recoverable on 8B Q/O RRR
//   and 8B Gate/Up RRR (both K=4096) by forcing full unroll.
// Risk:
//   Larger K (K>=8192 has k_pairs >= 31) plus full unroll may explode VGPR
//   spill or hurt I-cache locality; benchmark all 7 RRR shapes before SHIP.
#ifndef MXFP8_RRR_MAIN_UNROLL
#define MXFP8_RRR_MAIN_UNROLL 0
#endif

#if MXFP8_RRR_MAIN_UNROLL > 0
#define MXFP8_RRR_PRAGMA_UNROLL_MAIN _Pragma(TK_STRINGIFY(unroll MXFP8_RRR_MAIN_UNROLL))
#else
#define MXFP8_RRR_PRAGMA_UNROLL_MAIN
#endif

// R50 Dev D: Optional inter-phase sched_barrier inside do_k_iter to gate the
// LLVM scheduler at the load-issue -> MMA-dispatch boundary without
// materializing a call frame (the noinline tool from R49B was too coarse,
// see r49b_findings.md "Diagnosis"). The barrier is emitted right after the
// post-load s_barrier() and before the s_setprio(1)/s_waitcnt that opens
// the MFMA dispatch window.
//
// MXFP8_RRR_SCHED_BARRIER values (default 0 = OFF, baseline behavior):
//   0 = OFF                         (no extra sched_barrier emitted)
//   1 = mask 0x0000 (strictest: nothing crosses)
//   2 = mask 0x0008 (allow MFMA/WMMA only)
//   3 = mask 0x0010 (allow ALL VMEM only)
//   4 = mask 0x0080 (allow ALL DS/LDS only)
// (Mask values per /opt/rocm/lib/llvm/include/llvm/IR/IntrinsicsAMDGPU.td.)
#ifndef MXFP8_RRR_SCHED_BARRIER
#define MXFP8_RRR_SCHED_BARRIER 0
#endif

#if MXFP8_RRR_SCHED_BARRIER == 1
#define MXFP8_RRR_PHASE_SB() __builtin_amdgcn_sched_barrier(0x0000)
#elif MXFP8_RRR_SCHED_BARRIER == 2
#define MXFP8_RRR_PHASE_SB() __builtin_amdgcn_sched_barrier(0x0008)
#elif MXFP8_RRR_SCHED_BARRIER == 3
#define MXFP8_RRR_PHASE_SB() __builtin_amdgcn_sched_barrier(0x0010)
#elif MXFP8_RRR_SCHED_BARRIER == 4
#define MXFP8_RRR_PHASE_SB() __builtin_amdgcn_sched_barrier(0x0080)
#else
#define MXFP8_RRR_PHASE_SB() do {} while (0)
#endif

// R49 Dev B: Phase-split do_k_iter MMA quartet into two noinline halves to
// break the compiler's bimodal VGPR-spill threshold (R48 Dev E/G: any unroll
// >= 2 commits to flat 208 B/lane scratch). The lambda's 4 MFMAs (cA, cB,
// cC, cD) consume {a0_scale_packs, a1_scale_packs} × {b0_scale_packs,
// b1_scale_packs}; the noinline boundary forces the compiler to drop a0
// scale-pack live ranges before a1 scale-pack live ranges become live.
//
// Default OFF (=0). When MXFP8_RRR_PHASE_SPLIT=1, the lambda routes its 4
// MFMAs through `do_mma_quartet_first_half` (cA+cB, uses a0_scale_packs)
// and `do_mma_quartet_second_half` (cC+cD, uses a1_scale_packs), both
// marked __attribute__((noinline)).
#ifndef MXFP8_RRR_PHASE_SPLIT
#define MXFP8_RRR_PHASE_SPLIT 0
#endif

// R52 Dev H: Soft scheduler barrier alternative to R49 Dev B's noinline
// phase split. The R49 Dev B noinline boundary (between do_mma_quartet_first_half
// = cA+cB consuming a0_scale_packs and do_mma_quartet_second_half = cC+cD
// consuming a1_scale_packs) caused catastrophic spills (-97%) on most cells
// because clang materializes a real call frame. R49 Dev B's findings.md
// suggested replacing the noinline with `asm volatile("" ::: "memory")` —
// a compiler barrier with no call frame, just a scheduler-ordering boundary.
// This is intended to gate the LLVM scheduler at the natural cB→cC boundary
// without forcing a stack frame, hopefully allowing scale-pack v_lshrrev
// issue to overlap better with VMEM/LDS latency.
//
// Default OFF (=0). When MXFP8_RRR_SOFT_BARRIER=1, an empty inline asm
// barrier is emitted at the cB→cC boundary in the baseline (non-phase-split)
// body of do_k_iter.
#ifndef MXFP8_RRR_SOFT_BARRIER
#define MXFP8_RRR_SOFT_BARRIER 0
#endif

#if MXFP8_RRR_SOFT_BARRIER
#define MXFP8_RRR_SOFT_BAR() __asm__ volatile("" ::: "memory")
#else
#define MXFP8_RRR_SOFT_BAR() do {} while (0)
#endif

#if MXFP8_RRR_EXACT_8WAVE_FAST_ENABLE

#if MXFP8_RRR_PHASE_SPLIT
// Noinline halves of the do_k_iter MMA quartet. Each takes only the scale
// packs it actually uses (a0 or a1, plus both b packs). Barriers/setprio
// surrounding each MFMA pair stay inside the function so the live-range
// graph at the call site is just {a, b0, b1, scale_packs, accumulators}.
//
// The accumulators (rt_fl<RBM,RBN,...>) are passed by reference; the input
// `a` and `b` regs are passed by const-reference; scale packs are
// arrays-by-reference (already preserved through rrr_mma_scaled_phase).
//
// `__attribute__((noinline))` is the load-bearing attribute. We also add
// `__attribute__((flatten))` so any inner inlinable helpers (e.g.
// rrr_mma_scaled_phase) still get inlined into the helper, but the helper
// itself remains a function call.
template<int K_PHASE, int A_PACK_COUNT, int B_PACK_COUNT>
__attribute__((noinline)) __device__
void do_mma_quartet_first_half(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& cA,
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& cB,
    const A_row_reg& a,
    const B_col_reg& b0,
    const B_col_reg& b1,
    const fp8e8m0_4 (&a0_scale_packs)[A_PACK_COUNT],
    const fp8e8m0_4 (&b0_scale_packs)[B_PACK_COUNT],
    const fp8e8m0_4 (&b1_scale_packs)[B_PACK_COUNT])
{
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_setprio(1);
    rrr_mma_scaled_phase<K_PHASE>(cA, a, b0, a0_scale_packs, b0_scale_packs);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    RRR_SCHED_BARRIER();
    // Caller is responsible for the prefetch + barrier between cA and cB.
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_setprio(1);
    rrr_mma_scaled_phase<K_PHASE>(cB, a, b1, a0_scale_packs, b1_scale_packs);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
}

template<int K_PHASE, int A_PACK_COUNT, int B_PACK_COUNT>
__attribute__((noinline)) __device__
void do_mma_quartet_second_half(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& cC,
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& cD,
    const A_row_reg& a,
    const B_col_reg& b0,
    const B_col_reg& b1,
    const fp8e8m0_4 (&a1_scale_packs)[A_PACK_COUNT],
    const fp8e8m0_4 (&b0_scale_packs)[B_PACK_COUNT],
    const fp8e8m0_4 (&b1_scale_packs)[B_PACK_COUNT])
{
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_setprio(1);
    rrr_mma_scaled_phase<K_PHASE>(cC, a, b0, a1_scale_packs, b0_scale_packs);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    RRR_SCHED_BARRIER();
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_setprio(1);
    rrr_mma_scaled_phase<K_PHASE>(cD, a, b1, a1_scale_packs, b1_scale_packs);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
}
#endif // MXFP8_RRR_PHASE_SPLIT

static_assert(BLK == 256, "MXFP8 exact 8-wave fast path requires BLK=256");
static_assert(BK == 128, "MXFP8 exact 8-wave fast path requires BK=128");
static_assert(WARPS_M == 2, "MXFP8 exact 8-wave fast path requires WARPS_M=2");
static_assert(WARPS_N == 4, "MXFP8 exact 8-wave fast path requires WARPS_N=4");

template<bool PRESHUFFLED_QUANT, int SCALE_VERSION = 1>
__global__ __launch_bounds__(_NUM_THREADS, GEMM_MIN_BLOCKS_PER_CU)
void rrr_exact_8wave_scaled_kernel(const layout_globals g) {
    // SCALE_VERSION:
    //   1 = original V1 preshuffle layout (per row_group / b32 loads)
    //   2 = R22 V2-RRR wave-tile reordered layout (per wave-tile SRD,
    //       buffer_load_b128 for A, buffer_load_b64 for B). Mirrors the
    //       R21 V2-RCR wiring. The host must preshuffle the scale tensors
    //       via preshuffle_scale_matrix_mfma16_v2_rcr_a/b (same layout as
    //       RCR — A/B scale base formulas are identical between RCR/RRR).
    // R46 Dev D: use constexpr for the exact fastpath to enable full
    // loop unrolling and better compile-time scheduling.
    constexpr int blocks_per_row = M_DIM / BLK;
    constexpr int blocks_per_col = N_DIM / BLK;
    constexpr int k_iters = K_DIM / BK;

    __shared__ ST_row As[2][2];
#if RRR_ROW_SHARED_TRANSPOSE
    __shared__ ST_row Bs[2][2];
#else
    __shared__ ST_rrr_b Bs[2][2];
#endif

    A_row_reg a;
    B_col_reg b0, b1;
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;
    zero(cA); zero(cB); zero(cC); zero(cD);

    int bid = blockIdx.x;
#if MXFP8_RRR_BLOCK_SWIZZLE
    // R46 Dev D: XCD-aware chiplet swizzle + grouped-M swizzle.
    // Chiplet swizzle: interleave blocks across XCDs for better L2 locality.
    {
        const int num_wgs = gridDim.x;
        if (num_wgs >= MXFP8_RRR_BLOCK_SWIZZLE_NUM_XCDS &&
            (num_wgs % MXFP8_RRR_BLOCK_SWIZZLE_NUM_XCDS) == 0) {
            bid =
                (bid % MXFP8_RRR_BLOCK_SWIZZLE_NUM_XCDS) *
                    (num_wgs / MXFP8_RRR_BLOCK_SWIZZLE_NUM_XCDS) +
                (bid / MXFP8_RRR_BLOCK_SWIZZLE_NUM_XCDS);
        }
    }
    // Grouped-M swizzle: group M-tiles so adjacent blocks share B-tiles.
    const int num_wgid_in_group = MXFP8_RRR_BLOCK_SWIZZLE_GROUP_M * blocks_per_col;
    const int group_id = bid / num_wgid_in_group;
    const int first_pid_m = group_id * MXFP8_RRR_BLOCK_SWIZZLE_GROUP_M;
    const int group_size_m =
        (first_pid_m + MXFP8_RRR_BLOCK_SWIZZLE_GROUP_M <= blocks_per_row)
            ? MXFP8_RRR_BLOCK_SWIZZLE_GROUP_M
            : (blocks_per_row - first_pid_m);
    int br, bc;
    if (group_size_m <= 0) {
        br = blocks_per_row;
        bc = blocks_per_col;
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

    auto rrr_scale_a_base = [&](int half) {
        return br * BLK + half * HB + wm * RBM;
    };
    auto rrr_scale_b_base = [&](int half) {
        return bc * BLK + half * HB + wn * RBN;
    };
    const int lane_nonk = kittens::laneid() % 16;
    const int lane_kblk = kittens::laneid() / 16;
    const uint32_t lane_scale_byte_offset =
        (static_cast<uint32_t>(lane_kblk) << 6) |
        (static_cast<uint32_t>(lane_nonk) << 2);

    constexpr int rrr_a_pack_count = RBM / 32;
    constexpr int rrr_b_pack_count = (RBN + 31) / 32;
    fp8e8m0_4 a0_scale_packs[rrr_a_pack_count], a1_scale_packs[rrr_a_pack_count];
    fp8e8m0_4 b0_scale_packs[rrr_b_pack_count], b1_scale_packs[rrr_b_pack_count];
    const uint8_t* a0_scale_row_bases[rrr_a_pack_count];
    const uint8_t* a1_scale_row_bases[rrr_a_pack_count];
    const uint8_t* b0_scale_row_bases[rrr_b_pack_count];
    const uint8_t* b1_scale_row_bases[rrr_b_pack_count];
#if MXFP8_RRR_EXACT_PQ_PIPELINE_SCALE_ENABLE
    i32x4 a0p0_srsrc = {}, a0p1_srsrc = {}, a1p0_srsrc = {}, a1p1_srsrc = {};
    i32x4 b0p0_srsrc = {}, b1p0_srsrc = {};
#endif
    // R22 V2 wave-tile SRDs: 1 SRD per wave for A, 1 per wave for B.
    i32x4 a_v2_srsrc = {}, b_v2_srsrc = {};
    if constexpr (PRESHUFFLED_QUANT) {
        #pragma unroll
        for (int p = 0; p < rrr_a_pack_count; ++p) {
            a0_scale_row_bases[p] = preshuffled_scale_row_base_ptr(
                g.a_scale, (rrr_scale_a_base(0) + p * 32) >> 5);
            a1_scale_row_bases[p] = preshuffled_scale_row_base_ptr(
                g.a_scale, (rrr_scale_a_base(1) + p * 32) >> 5);
        }
        #pragma unroll
        for (int p = 0; p < rrr_b_pack_count; ++p) {
            b0_scale_row_bases[p] = preshuffled_scale_row_base_ptr(
                g.b_scale, (rrr_scale_b_base(0) + p * 32) >> 5);
            b1_scale_row_bases[p] = preshuffled_scale_row_base_ptr(
                g.b_scale, (rrr_scale_b_base(1) + p * 32) >> 5);
        }
        if constexpr (SCALE_VERSION == 2) {
            // V2-RRR: per-wave-tile slab SRDs (mirrors V2-RCR exactly; A/B
            // scale base formulas are identical between RCR and RRR so the
            // same preshuffle layout works).
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
#if MXFP8_RRR_EXACT_PQ_PIPELINE_SCALE_ENABLE
        {
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
            if constexpr (rrr_a_pack_count > 1) {
                a0p1_srsrc = make_scale_srd(a0_scale_row_bases[1]);
                a1p1_srsrc = make_scale_srd(a1_scale_row_bases[1]);
            }
        }
#endif
    }
    auto load_scale_packs = [&](int k_pair) __attribute__((always_inline)) {
        if constexpr (PRESHUFFLED_QUANT && SCALE_VERSION == 2) {
            // R22 V2-RRR: per (k_pair, lane), one b128 fetches all 4 A
            // packs ({a0p0, a1p0, a0p1, a1p1}) and one b64 fetches both
            // B packs ({b0p0, b1p0}). Mirrors V2-RCR exactly.
            //   A side (PC=4): voff = lane_kblk*256 + lane_nonk*16,
            //                  soff = k_pair * 1024
            //   B side (PC=2): voff = lane_kblk*128 + lane_nonk*8,
            //                  soff = k_pair * 512
            const uint32_t a_voff =
                (static_cast<uint32_t>(lane_kblk) << 8) |
                (static_cast<uint32_t>(lane_nonk) << 4);
            const uint32_t a_soff = static_cast<uint32_t>(k_pair) << 10;
            const __uint128_t a_raw =
                llvm_amdgcn_raw_buffer_load_b128(a_v2_srsrc, a_voff, a_soff, MXFP8_RRR_V2_SCALE_CACHEPOLICY);
            a0_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw      ));
            a1_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw >> 32));
            if constexpr (rrr_a_pack_count > 1) {
                a0_scale_packs[1] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw >> 64));
                a1_scale_packs[1] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw >> 96));
            }
            const uint32_t b_voff =
                (static_cast<uint32_t>(lane_kblk) << 7) |
                (static_cast<uint32_t>(lane_nonk) << 3);
            const uint32_t b_soff = static_cast<uint32_t>(k_pair) << 9;
            const uint64_t b_raw =
                llvm_amdgcn_raw_buffer_load_b64(b_v2_srsrc, b_voff, b_soff, MXFP8_RRR_V2_SCALE_CACHEPOLICY);
            b0_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(b_raw      ));
            b1_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(b_raw >> 32));
            return;
        }
#if MXFP8_RRR_EXACT_PQ_PIPELINE_SCALE_ENABLE
        if constexpr (PRESHUFFLED_QUANT) {
            const uint32_t soff = static_cast<uint32_t>(k_pair) << 8;
            a0_scale_packs[0] = std::bit_cast<fp8e8m0_4>(llvm_amdgcn_raw_buffer_load_b32(a0p0_srsrc, lane_scale_byte_offset, soff, 0));
            a1_scale_packs[0] = std::bit_cast<fp8e8m0_4>(llvm_amdgcn_raw_buffer_load_b32(a1p0_srsrc, lane_scale_byte_offset, soff, 0));
            if constexpr (rrr_a_pack_count > 1) {
                a0_scale_packs[1] = std::bit_cast<fp8e8m0_4>(llvm_amdgcn_raw_buffer_load_b32(a0p1_srsrc, lane_scale_byte_offset, soff, 0));
                a1_scale_packs[1] = std::bit_cast<fp8e8m0_4>(llvm_amdgcn_raw_buffer_load_b32(a1p1_srsrc, lane_scale_byte_offset, soff, 0));
            }
            b0_scale_packs[0] = std::bit_cast<fp8e8m0_4>(llvm_amdgcn_raw_buffer_load_b32(b0p0_srsrc, lane_scale_byte_offset, soff, 0));
            b1_scale_packs[0] = std::bit_cast<fp8e8m0_4>(llvm_amdgcn_raw_buffer_load_b32(b1p0_srsrc, lane_scale_byte_offset, soff, 0));
            return;
        }
#endif
        #pragma unroll
        for (int p = 0; p < rrr_a_pack_count; ++p) {
            if constexpr (PRESHUFFLED_QUANT) {
                a0_scale_packs[p] = load_scale_pair_pack_16x128_preshuffled_from_row_base(
                    a0_scale_row_bases[p], k_pair, lane_scale_byte_offset);
                a1_scale_packs[p] = load_scale_pair_pack_16x128_preshuffled_from_row_base(
                    a1_scale_row_bases[p], k_pair, lane_scale_byte_offset);
            } else {
                a0_scale_packs[p] = load_scale_pair_pack_16x128(
                    g.a_scale, rrr_scale_a_base(0) + p * 32, k_pair, lane_nonk, lane_kblk);
                a1_scale_packs[p] = load_scale_pair_pack_16x128(
                    g.a_scale, rrr_scale_a_base(1) + p * 32, k_pair, lane_nonk, lane_kblk);
            }
        }
        #pragma unroll
        for (int p = 0; p < rrr_b_pack_count; ++p) {
            if constexpr (PRESHUFFLED_QUANT) {
                b0_scale_packs[p] = load_scale_pair_pack_16x128_preshuffled_from_row_base(
                    b0_scale_row_bases[p], k_pair, lane_scale_byte_offset);
                b1_scale_packs[p] = load_scale_pair_pack_16x128_preshuffled_from_row_base(
                    b1_scale_row_bases[p], k_pair, lane_scale_byte_offset);
            } else {
                b0_scale_packs[p] = load_scale_pair_pack_16x128(
                    g.b_scale, rrr_scale_b_base(0) + p * 32, k_pair, lane_nonk, lane_kblk);
                b1_scale_packs[p] = load_scale_pair_pack_16x128(
                    g.b_scale, rrr_scale_b_base(1) + p * 32, k_pair, lane_nonk, lane_kblk);
            }
        }
    };

    constexpr int bptA = ST_row::underlying_subtile_bytes_per_thread;
    constexpr int bpmA = bptA * _NUM_THREADS;
    constexpr int mptA = ST_row::rows * ST_row::cols * sizeof(fp8e4m3) / bpmA;
    uint32_t soA[mptA];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);

    constexpr int bptB =
#if RRR_ROW_SHARED_TRANSPOSE
        ST_row::underlying_subtile_bytes_per_thread;
#else
        ST_rrr_b::underlying_subtile_bytes_per_thread;
#endif
    constexpr int bpmB = bptB * _NUM_THREADS;
    constexpr int mptB =
#if RRR_ROW_SHARED_TRANSPOSE
        ST_row::rows * ST_row::cols * sizeof(fp8e4m3) / bpmB;
#else
        ST_rrr_b::rows * ST_rrr_b::cols * sizeof(fp8e4m3) / bpmB;
#endif
    uint32_t soB[mptB];
#if RRR_ROW_SHARED_TRANSPOSE
    prefill_transpose_swizzled_offsets<_NUM_THREADS>(Bs[0][0], g.b, soB);
#else
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);
#endif

    auto a_co = [&](int s, int k) -> coord<ST_row> { return {0, 0, s, k}; };
#if RRR_ROW_SHARED_TRANSPOSE
    auto b_co = [&](int s, int k) -> coord<ST_row> { return {0, 0, s, k}; };
#else
    auto b_co = [&](int s, int k) -> coord<ST_rrr_b> { return {0, 0, k, s}; };
#endif

    auto load_a = [&](A_row_reg& dst, ST_row& tile, int wi) {
        auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
        load(dst, sub);
    };
#if RRR_ROW_SHARED_TRANSPOSE
    auto load_b = [&](B_col_reg& dst, ST_row& tile, int wi) {
        B_row_reg tmp;
        auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
        load(tmp, sub);
        transpose(dst, tmp);
    };
#else
    auto load_b = [&](B_col_reg& dst, ST_rrr_b& tile, int wi) {
    #if RRR_B_REG_ROW_LOAD_TRANSPOSE
    #if RRR_B_REG_ROW_LOAD_ALIAS
    #if RRR_USE_V2A_SWIZZLE
        load_col_from_v2a_st(dst, tile, wi * RBN);
    #else
        load_col_from_v2_st(dst, tile, wi * RBN);
    #endif
    #else
        auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
        B_row_reg tmp;
        load(tmp, sub);
        transpose(dst, tmp);
    #endif
    #else
    #if RRR_USE_V2A_SWIZZLE
        load_col_from_v2a_st(dst, tile, wi * RBN);
    #else
        load_col_from_v2_st(dst, tile, wi * RBN);
    #endif
    #endif
    };
#endif

    int tic = 0, toc = 1;
#if RRR_ROW_SHARED_TRANSPOSE
    load_transpose<_NUM_THREADS>(Bs[tic][0], g.b, b_co(bc * 2, 0), soB);
#else
    G::load(Bs[tic][0], g.b, b_co(bc * 2, 0), soB);
#endif
    G::load(As[tic][0], g.a, a_co(br * 2, 0), soA);
#if RRR_ROW_SHARED_TRANSPOSE
    load_transpose<_NUM_THREADS>(Bs[tic][1], g.b, b_co(bc * 2 + 1, 0), soB);
#else
    G::load(Bs[tic][1], g.b, b_co(bc * 2 + 1, 0), soB);
#endif
    G::load(As[tic][1], g.a, a_co(br * 2 + 1, 0), soA);

    if (wm == 1) {
        __builtin_amdgcn_s_barrier();
    }
    TK_WAIT_VMCNT(RRR_INIT0_VMCNT);
    __builtin_amdgcn_s_barrier();

#if RRR_ROW_SHARED_TRANSPOSE
    load_transpose<_NUM_THREADS>(Bs[toc][0], g.b, b_co(bc * 2, 1), soB);
#else
    G::load(Bs[toc][0], g.b, b_co(bc * 2, 1), soB);
#endif
    G::load(As[toc][0], g.a, a_co(br * 2, 1), soA);
#if RRR_ROW_SHARED_TRANSPOSE
    load_transpose<_NUM_THREADS>(Bs[toc][1], g.b, b_co(bc * 2 + 1, 1), soB);
#else
    G::load(Bs[toc][1], g.b, b_co(bc * 2 + 1, 1), soB);
#endif

    TK_WAIT_VMCNT(RRR_INIT1_VMCNT);
    __builtin_amdgcn_s_barrier();

    auto do_k_iter = [&](int k, auto phase_tag) __attribute__((always_inline)) {
        constexpr int phase = decltype(phase_tag)::value;

#if MXFP8_RRR_PHASE_SPLIT
        // R49 Dev B: Phase-split path. The MFMA quartet is delivered through
        // two noinline halves (cA+cB use a0_scale_packs; cC+cD use
        // a1_scale_packs). The function-call boundary forces the compiler to
        // drop a0 scale-pack live ranges before a1 scale-pack live ranges
        // become live, breaking the bimodal spill threshold (R48 Dev E/G).
        //
        // Loads needed for the first MFMA pair (cA, cB) are issued before
        // the first noinline call. Loads needed for the second MFMA pair
        // (cC, cD) are issued between the two noinline calls.
        load_b(b0, Bs[tic][0], wn);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        const auto b0_keep = b0;
#endif
        load_a(a, As[tic][0], wm);
        G::load(As[toc][1], g.a, a_co(br * 2 + 1, k + 1), soA);
        TK_WAIT_LGKM(RRR_PREFETCH_LGKM);
        __builtin_amdgcn_s_barrier();

        // Pre-issue loads for cB so the first half can issue both MFMAs
        // back-to-back.
        load_b(b1, Bs[tic][1], wn);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        const auto b1_keep = b1;
#endif
#if RRR_ROW_SHARED_TRANSPOSE
        load_transpose<_NUM_THREADS>(Bs[tic][0], g.b, b_co(bc * 2, k + 2), soB);
#else
        G::load(Bs[tic][0], g.b, b_co(bc * 2, k + 2), soB);
#endif

#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        do_mma_quartet_first_half<phase>(
            cA, cB, a, b0_keep, b1_keep,
            a0_scale_packs, b0_scale_packs, b1_scale_packs);
#else
        do_mma_quartet_first_half<phase>(
            cA, cB, a, b0, b1,
            a0_scale_packs, b0_scale_packs, b1_scale_packs);
#endif

        // Loads for cC and cD.
        load_a(a, As[tic][1], wm);
#if RRR_ROW_SHARED_TRANSPOSE
        load_transpose<_NUM_THREADS>(Bs[tic][1], g.b, b_co(bc * 2 + 1, k + 2), soB);
#else
        G::load(Bs[tic][1], g.b, b_co(bc * 2 + 1, k + 2), soB);
#endif
        G::load(As[tic][0], g.a, a_co(br * 2, k + 2), soA);
        TK_WAIT_VMCNT(RRR_STEADY_VMCNT);
        __builtin_amdgcn_s_barrier();

#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        do_mma_quartet_second_half<phase>(
            cC, cD, a, b0_keep, b1_keep,
            a1_scale_packs, b0_scale_packs, b1_scale_packs);
#else
        do_mma_quartet_second_half<phase>(
            cC, cD, a, b0, b1,
            a1_scale_packs, b0_scale_packs, b1_scale_packs);
#endif
#else // !MXFP8_RRR_PHASE_SPLIT — original baseline body
        load_b(b0, Bs[tic][0], wn);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        const auto b0_keep = b0;
#endif
        load_a(a, As[tic][0], wm);
        G::load(As[toc][1], g.a, a_co(br * 2 + 1, k + 1), soA);
        TK_WAIT_LGKM(RRR_PREFETCH_LGKM);
        __builtin_amdgcn_s_barrier();
        MXFP8_RRR_PHASE_SB();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        rrr_mma_scaled_phase<phase>(cA, a, b0_keep, a0_scale_packs, b0_scale_packs);
#else
        rrr_mma_scaled_phase<phase>(cA, a, b0, a0_scale_packs, b0_scale_packs);
#endif
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        RRR_SCHED_BARRIER();

        load_b(b1, Bs[tic][1], wn);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        const auto b1_keep = b1;
#endif
#if RRR_ROW_SHARED_TRANSPOSE
        load_transpose<_NUM_THREADS>(Bs[tic][0], g.b, b_co(bc * 2, k + 2), soB);
#else
        G::load(Bs[tic][0], g.b, b_co(bc * 2, k + 2), soB);
#endif
        __builtin_amdgcn_s_barrier();
        MXFP8_RRR_PHASE_SB();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        rrr_mma_scaled_phase<phase>(cB, a, b1_keep, a0_scale_packs, b1_scale_packs);
#else
        rrr_mma_scaled_phase<phase>(cB, a, b1, a0_scale_packs, b1_scale_packs);
#endif
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        // R52 Dev H: soft scheduler barrier at the cB->cC quartet boundary
        // (the same boundary R49 Dev B's noinline used). Default OFF.
        MXFP8_RRR_SOFT_BAR();

        load_a(a, As[tic][1], wm);
#if RRR_ROW_SHARED_TRANSPOSE
        load_transpose<_NUM_THREADS>(Bs[tic][1], g.b, b_co(bc * 2 + 1, k + 2), soB);
#else
        G::load(Bs[tic][1], g.b, b_co(bc * 2 + 1, k + 2), soB);
#endif
        __builtin_amdgcn_s_barrier();
        MXFP8_RRR_PHASE_SB();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        rrr_mma_scaled_phase<phase>(cC, a, b0_keep, a1_scale_packs, b0_scale_packs);
#else
        rrr_mma_scaled_phase<phase>(cC, a, b0, a1_scale_packs, b0_scale_packs);
#endif
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        RRR_SCHED_BARRIER();

        G::load(As[tic][0], g.a, a_co(br * 2, k + 2), soA);
        TK_WAIT_VMCNT(RRR_STEADY_VMCNT);
        __builtin_amdgcn_s_barrier();
        MXFP8_RRR_PHASE_SB();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        rrr_mma_scaled_phase<phase>(cD, a, b1_keep, a1_scale_packs, b1_scale_packs);
#else
        rrr_mma_scaled_phase<phase>(cD, a, b1, a1_scale_packs, b1_scale_packs);
#endif
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
#endif // MXFP8_RRR_PHASE_SPLIT
    };

    {
        constexpr int k_iters_main = k_iters - 2;
        constexpr int k_pairs = k_iters_main / 2;
        MXFP8_RRR_PRAGMA_UNROLL_MAIN
        for (int kp = 0; kp < k_pairs; kp++) {
            load_scale_packs(kp);
            do_k_iter(kp * 2, phase_t<0>{});
            tic ^= 1; toc ^= 1;
            do_k_iter(kp * 2 + 1, phase_t<1>{});
            tic ^= 1; toc ^= 1;
        }
    }

    {
        const int k = k_iters - 2;
        load_scale_packs(k >> 1);

        load_b(b0, Bs[tic][0], wn);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        const auto b0_keep = b0;
#endif
        load_a(a, As[tic][0], wm);
        G::load(As[toc][1], g.a, a_co(br * 2 + 1, k + 1), soA);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        rrr_mma_scaled_phase<0>(cA, a, b0_keep, a0_scale_packs, b0_scale_packs);
#else
        rrr_mma_scaled_phase<0>(cA, a, b0, a0_scale_packs, b0_scale_packs);
#endif
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        RRR_SCHED_BARRIER();

        load_b(b1, Bs[tic][1], wn);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        const auto b1_keep = b1;
#endif
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        rrr_mma_scaled_phase<0>(cB, a, b1_keep, a0_scale_packs, b1_scale_packs);
#else
        rrr_mma_scaled_phase<0>(cB, a, b1, a0_scale_packs, b1_scale_packs);
#endif
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a(a, As[tic][1], wm);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        rrr_mma_scaled_phase<0>(cC, a, b0_keep, a1_scale_packs, b0_scale_packs);
#else
        rrr_mma_scaled_phase<0>(cC, a, b0, a1_scale_packs, b0_scale_packs);
#endif
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_b(b0, Bs[toc][0], wn);
        TK_WAIT_VMCNT(RRR_EPILOGUE_VMCNT);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        rrr_mma_scaled_phase<0>(cD, a, b1_keep, a1_scale_packs, b1_scale_packs);
#else
        rrr_mma_scaled_phase<0>(cD, a, b1, a1_scale_packs, b1_scale_packs);
#endif
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        RRR_SCHED_BARRIER();
        tic ^= 1;
        toc ^= 1;
    }

    {
        load_a(a, As[tic][0], wm);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        const auto b0_keep = b0;
#endif
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        rrr_mma_scaled_phase<1>(cA, a, b0_keep, a0_scale_packs, b0_scale_packs);
#else
        rrr_mma_scaled_phase<1>(cA, a, b0, a0_scale_packs, b0_scale_packs);
#endif
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_b(b1, Bs[tic][1], wn);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        const auto b1_keep = b1;
#endif
        __builtin_amdgcn_s_barrier();
        RRR_SCHED_BARRIER();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        rrr_mma_scaled_phase<1>(cB, a, b1_keep, a0_scale_packs, b1_scale_packs);
#else
        rrr_mma_scaled_phase<1>(cB, a, b1, a0_scale_packs, b1_scale_packs);
#endif
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a(a, As[tic][1], wm);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        rrr_mma_scaled_phase<1>(cC, a, b0_keep, a1_scale_packs, b0_scale_packs);
        rrr_mma_scaled_phase<1>(cD, a, b1_keep, a1_scale_packs, b1_scale_packs);
#else
        rrr_mma_scaled_phase<1>(cC, a, b0, a1_scale_packs, b0_scale_packs);
        rrr_mma_scaled_phase<1>(cD, a, b1, a1_scale_packs, b1_scale_packs);
#endif
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

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

template<bool PRESHUFFLED_QUANT>
__host__ inline void dispatch_rrr_exact_8wave_scaled(const layout_globals& g) {
    const dim3 grid((g.m / BLK) * (g.n / BLK));
    rrr_exact_8wave_scaled_kernel<PRESHUFFLED_QUANT, 1>
        <<<grid, dim3(_NUM_THREADS), 0, g.stream>>>(g);
}

// R22 milestone-1: V2-RRR scale layout dispatch (only valid when
// PRESHUFFLED_QUANT=true and the host has preshuffled the scale tensors via
// preshuffle_scale_matrix_mfma16_v2_rcr_a/b — same layout works for RRR).
template<bool PRESHUFFLED_QUANT>
__host__ inline void dispatch_rrr_exact_8wave_scaled_v2(const layout_globals& g) {
    const dim3 grid((g.m / BLK) * (g.n / BLK));
    rrr_exact_8wave_scaled_kernel<PRESHUFFLED_QUANT, 2>
        <<<grid, dim3(_NUM_THREADS), 0, g.stream>>>(g);
}

__host__ inline bool rrr_can_use_exact_8wave_scaled(const layout_globals& g) {
    return g.m == M_DIM && g.n == N_DIM && g.k == K_DIM;
}

#endif
