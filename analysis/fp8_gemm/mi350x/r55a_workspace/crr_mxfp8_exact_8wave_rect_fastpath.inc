#pragma once

// R33 Dev A — Stage A2 Path 1: square-LDS rect-V2 CRR fastpath.
//
// REWRITE of R31/R32 Stage A1 scaffolding which used HB_N=64 LDS tile and hit
// the load_col_from_v2_st_half K_HALF architectural blocker (R32 Dev A
// findings). Per R32 Dev A's recommendation, Path 1:
//
//   - KEEP square ST_v2 LDS tile (HB=128, BK=128) — same as default fastpath.
//   - HALVE the per-block N work: each block now processes BLK_M=256 × HB=128
//     N-cols (one N-half-tile) instead of BLK_M=256 × BLK=256.
//   - Drop b1, cB, cD; only b0, cA, cC remain (1 N-half × 2 M-halves).
//   - Grid doubles in N: blocks_per_col = N / HB instead of N / BLK.
//   - Host preshuffle is UNCHANGED — the LDS tile geometry didn't change,
//     and we re-derive scale-slab indexing to address the same square slab
//     layout from the doubled-grid kernel.
//
// Correctness gate (Stage A2c): SNR >= 48 dB + det 3/3 PASS at 70B KV
// 4096x1024x8192. If correct → paired BABA bench rect (this file) vs square.
//
// Implementation notes:
//   - bc indexes HB-cols (0..N/HB-1). bc_orig = bc / 2 indexes the original
//     square BLK-col. nhalf = bc & 1 picks which N-half of the square block.
//   - Scale slab indexing: slab_idx_b = bc_orig * WARPS_N + wn (same as
//     square — host preshuffle unchanged). The slab contains BOTH N-halves
//     packed (pack_count=2). We select b{0,1}_scale_packs based on nhalf.
//   - LDS B-tile load: each block loads ONE square ST_v2 tile (HB=128,
//     BK=128). Coordinate `bc * 2 + nhalf` in the original kernel
//     corresponds to bc (HB-id) here directly: just use `bc` as the
//     N-tile id (which equals the original `bc_orig * 2 + nhalf`).
//   - Output store: bc * WARPS_N + wn (HB-cols of WARPS_N*RBN=128 each
//     correspond to single N-halves; cA stores at row=br*WM*2+wm,
//     col=bc*WARPS_N+wn; cC at row=br*WM*2+WM+wm, same col).

#if defined(MXFP8_RECT_BLK_N) && (MXFP8_RECT_BLK_N == 64)

#ifndef MXFP8_CRR_RECT_8WAVE_FAST_ENABLE
#define MXFP8_CRR_RECT_8WAVE_FAST_ENABLE 1
#endif

// R34 Dev C — Path 2 selector. When 0 (default), build Path 1 (square LDS,
// halved N work) — byte-identical to R33. When 1, build Path 2 (HB_N=64 LDS
// for B, two K-serialised K_HALF=0 reads per K-iter from two HB_N=64 tiles
// staging K=0..63 and K=64..127 respectively). See r34c_findings.md.
#ifndef MXFP8_RECT_PATH2_K_SERIALIZE
#define MXFP8_RECT_PATH2_K_SERIALIZE 0
#endif

#if MXFP8_CRR_RECT_8WAVE_FAST_ENABLE

#if !MXFP8_RECT_PATH2_K_SERIALIZE
// =====================================================================
// PATH 1 (R33 Dev A): square LDS HB=128 + halved N-work + grid doubled
// =====================================================================

// Path 1 keeps the square primitives. Sanity-check the square invariants we
// rely on.
static_assert(BLK == 256, "Path 1 requires BLK_M=256");
static_assert(BK == 128, "Path 1 requires BK=128");
static_assert(WARPS_M == 2, "Path 1 requires WARPS_M=2");
static_assert(WARPS_N == 4, "Path 1 requires WARPS_N=4");
static_assert(HB == 128, "Path 1 requires HB=128 (square LDS tile)");
static_assert(RBN == 32, "Path 1 requires RBN=32 (square reg tile)");
static_assert(!CRR_ROW_SHARED_TRANSPOSE, "Path 1 only supports strict shared layout");
static_assert(!CRR_USE_V3_SWIZZLE, "Path 1 wired for v2/v2a shared layouts");

// Path 1 rect-N CRR exact 8-wave scaled kernel.
//   Block work: BLK_M=256 rows of M, HB=128 cols of N (ONE N-half-tile).
//   Grid: (M/BLK) * (N/HB)  — 2x more blocks vs square in N.
template<bool PRESHUFFLED_QUANT>
__global__ __launch_bounds__(_NUM_THREADS, GEMM_MIN_BLOCKS_PER_CU)
void crr_exact_8wave_scaled_rect_kernel(const layout_globals g)
{
    using ST_crr_a = ST_v2a;     // square A LDS tile (HB=128, BK=128)
    using ST_crr_b = ST_v2;      // square B LDS tile (HB=128, BK=128)

    constexpr int crr_a_pack_count = RBM / 32;       // 2
    constexpr int crr_b_pack_count = (RBN + 31) / 32; // 1 for RBN=32

    const int k_iters = g.k / BK;
    // Path 1: blocks_per_col counts HB-cols (N/HB), not BLK-cols (N/BLK).
    const int blocks_per_col_hb = g.n / HB;

    // Path 1: only ONE B-tile per LDS double-buffer slot (vs square's TWO).
    __shared__ ST_crr_a As[2][2];
    __shared__ ST_crr_b Bs[2];

    A_col_reg a;
    B_col_reg b0;
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cC;
    zero(cA);
    zero(cC);

    const int bid = blockIdx.x;
    const int br = bid / blocks_per_col_hb;
    const int bc = bid % blocks_per_col_hb;       // HB-col id (0..N/HB-1)
    const int bc_orig = bc >> 1;                  // square BLK-col id
    const int nhalf = bc & 1;                     // 0 or 1: which N-half
    const int wm = warpid() / WARPS_N;
    const int wn = warpid() % WARPS_N;

    const int lane_nonk = kittens::laneid() % 16;
    const int lane_kblk = kittens::laneid() / 16;

    // Scale base: A side unchanged (br * BLK = original M-block); B side uses
    // bc_orig * BLK to address the original square BLK-block scale slab.
    auto crr_scale_a_base = [&](int half) {
        return br * BLK + half * (BLK / 2) + wm * RBM;
    };
    auto crr_scale_b_base = [&](int half) {
        return bc_orig * BLK + half * (BLK / 2) + wn * RBN;
    };

    fp8e8m0_4 a0_scale_packs[crr_a_pack_count];
    fp8e8m0_4 a1_scale_packs[crr_a_pack_count];
    fp8e8m0_4 b0_scale_packs[crr_b_pack_count];
    fp8e8m0_4 b1_scale_packs[crr_b_pack_count];
    const uint8_t* a0_scale_row_bases[crr_a_pack_count];
    const uint8_t* a1_scale_row_bases[crr_a_pack_count];
    const uint8_t* b0_scale_row_bases[crr_b_pack_count];
    const uint8_t* b1_scale_row_bases[crr_b_pack_count];

    // V2 wave-tile SRDs — same slab geometry as square.
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
        // V2-CRR slab geometry — identical to square (host preshuffle unchanged).
        // A side: pack_count=2, slab_bytes_a = 128 * padded_k_blocks.
        //   slab_idx_a = br * WARPS_M + wm
        // B side: pack_count=2, slab_bytes_b = 64 * padded_k_blocks.
        //   slab_idx_b = bc_orig * WARPS_N + wn  (bc_orig, not bc)
        const int padded_k_blocks =
            (g.k / 32 + 7) & ~7;
        const size_t slab_bytes_a = 128u * padded_k_blocks;
        const size_t slab_bytes_b = 64u * padded_k_blocks;
        const size_t slab_idx_a = static_cast<size_t>(br) * WARPS_M + wm;
        const size_t slab_idx_b = static_cast<size_t>(bc_orig) * WARPS_N + wn;
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

    auto load_raw_scales = [&](int k_pair) __attribute__((always_inline)) {
        if constexpr (PRESHUFFLED_QUANT) {
            // V2-CRR: 1 b128 fetches 4 A packs, 1 b64 fetches 2 B packs.
            // Strides match square spec — slab layout unchanged.
            const uint32_t a_voff =
                (static_cast<uint32_t>(lane_kblk) << 8) |
                (static_cast<uint32_t>(lane_nonk) << 4);
            const uint32_t a_soff = static_cast<uint32_t>(k_pair) << 10;
            const __uint128_t a_raw =
                llvm_amdgcn_raw_buffer_load_b128(a_v2_srsrc, a_voff, a_soff, MXFP8_CRR_V2_SCALE_CACHEPOLICY);
            a0_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw      ));
            a1_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw >> 32));
            if constexpr (crr_a_pack_count > 1) {
                a0_scale_packs[1] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw >> 64));
                a1_scale_packs[1] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw >> 96));
            }
            const uint32_t b_voff =
                (static_cast<uint32_t>(lane_kblk) << 7) |
                (static_cast<uint32_t>(lane_nonk) << 3);
            const uint32_t b_soff = static_cast<uint32_t>(k_pair) << 9;
            const uint64_t b_raw =
                llvm_amdgcn_raw_buffer_load_b64(b_v2_srsrc, b_voff, b_soff, MXFP8_CRR_V2_SCALE_CACHEPOLICY);
            b0_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(b_raw      ));
            b1_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(b_raw >> 32));
            return;
        }
    };

    constexpr int bptA = ST_crr_a::underlying_subtile_bytes_per_thread;
    constexpr int bpmA = bptA * _NUM_THREADS;
    constexpr int mptA = ST_crr_a::rows * ST_crr_a::cols * sizeof(fp8e4m3) / bpmA;
    uint32_t soA[mptA];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);

    constexpr int bptB = ST_crr_b::underlying_subtile_bytes_per_thread;
    constexpr int bpmB = bptB * _NUM_THREADS;
    constexpr int mptB = ST_crr_b::rows * ST_crr_b::cols * sizeof(fp8e4m3) / bpmB;
    uint32_t soB[mptB];
    G::prefill_swizzled_offsets(Bs[0], g.b, soB);

    auto a_co = [&](int s, int k) -> coord<ST_crr_a> { return {0, 0, k, s}; };
    auto b_co = [&](int s, int k) -> coord<ST_crr_b> { return {0, 0, k, s}; };
    auto global_load_a = [&](ST_crr_a& tile, int s, int k) {
        G::load(tile, g.a, a_co(s, k), soA);
    };
    auto global_load_b = [&](ST_crr_b& tile, int s, int k) {
        G::load(tile, g.b, b_co(s, k), soB);
    };
    auto load_a = [&](A_col_reg& dst, ST_crr_a& tile, int wi) {
        load_col_from_v2a_st(dst, tile, wi * RBM);
    };
    auto load_b = [&](B_col_reg& dst, ST_crr_b& tile, int wi) {
        load_col_from_v2_st(dst, tile, wi * RBN);
    };

    // Path 1 simple single-buffered loop. Each iter:
    //   (1) load b0 from Bs[tic], a from As[tic][0], a' from As[tic][1].
    //   (2) MMA cA and cC against b0.
    //   (3) prefetch next-iter LDS-direct.
    // The B-side N-tile coord is `bc` (HB-col-id); for nhalf-aware scale-pack
    // selection we compile-time-branch on nhalf below by passing the right
    // {b0,b1}_scale_packs based on a runtime nhalf.
    int tic = 0, toc = 1;
    global_load_b(Bs[tic], bc, 0);
    global_load_a(As[tic][0], br * 2,     0);
    global_load_a(As[tic][1], br * 2 + 1, 0);

    if (wm == 1) {
        __builtin_amdgcn_s_barrier();
    }
    TK_WAIT_VMCNT(CRR_INIT0_VMCNT);
    __builtin_amdgcn_s_barrier();

    global_load_b(Bs[toc], bc, 1);
    global_load_a(As[toc][0], br * 2,     1);
    global_load_a(As[toc][1], br * 2 + 1, 1);

    TK_WAIT_VMCNT(CRR_INIT1_VMCNT);
    __builtin_amdgcn_s_barrier();

    // Pick the right scale-pack pair for this nhalf at runtime.
    // (b0_scale_packs and b1_scale_packs both populated by load_raw_scales;
    // we just pick the right one. The compiler can't constant-fold nhalf
    // here, so this is a runtime select per K-iter.)
    auto b_packs_for_nhalf = [&]() -> fp8e8m0_4* {
        return (nhalf == 0) ? b0_scale_packs : b1_scale_packs;
    };

    TK_PRAGMA_UNROLL(CRR_MAIN_UNROLL)
    for (int k = 0; k < k_iters - 2; k++, tic ^= 1, toc ^= 1) {
        if ((k & 1) == 0) {
            load_raw_scales(k >> 1);
        } else {
            #pragma unroll
            for (int gp = 0; gp < crr_a_pack_count; gp++) {
                a0_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a0_scale_packs[gp]) >> 16);
                a1_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a1_scale_packs[gp]) >> 16);
            }
            #pragma unroll
            for (int gp = 0; gp < crr_b_pack_count; gp++) {
                b0_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b0_scale_packs[gp]) >> 16);
                b1_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b1_scale_packs[gp]) >> 16);
            }
        }
        // Build a local nhalf-selected B-pack array.
        fp8e8m0_4 b_sel_packs[crr_b_pack_count];
        #pragma unroll
        for (int p = 0; p < crr_b_pack_count; p++) {
            b_sel_packs[p] = (nhalf == 0) ? b0_scale_packs[p] : b1_scale_packs[p];
        }

        load_b(b0, Bs[tic], wn);
        load_a(a, As[tic][0], wm);
        TK_WAIT_LGKM(CRR_EXACT_PREFETCH_LGKM);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_from_packs_fixed_phase(cA, a, b0, a0_scale_packs, b_sel_packs);
        CRR_MMA_END();
        CRR_STEADY_MID_BARRIER();
        CRR_SCHED_BARRIER();

        load_a(a, As[tic][1], wm);
        global_load_a(As[tic][0], br * 2,     k + 2);
        global_load_b(Bs[tic],    bc,         k + 2);
        TK_WAIT_VMCNT(CRR_EXACT_STEADY_VMCNT);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_from_packs_fixed_phase(cC, a, b0, a1_scale_packs, b_sel_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();
        global_load_a(As[tic][1], br * 2 + 1, k + 2);
    }

    // Tail: k = k_iters - 2 and k = k_iters - 1.
    {
        const int k = k_iters - 2;
        load_raw_scales(k >> 1);
        fp8e8m0_4 b_sel_packs[crr_b_pack_count];
        #pragma unroll
        for (int p = 0; p < crr_b_pack_count; p++) {
            b_sel_packs[p] = (nhalf == 0) ? b0_scale_packs[p] : b1_scale_packs[p];
        }

        load_b(b0, Bs[tic], wn);
        load_a(a, As[tic][0], wm);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase<0>(cA, a, b0, a0_scale_packs, b_sel_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();

        load_a(a, As[tic][1], wm);
        TK_WAIT_VMCNT(CRR_EPILOGUE_VMCNT);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase<0>(cC, a, b0, a1_scale_packs, b_sel_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();
        tic ^= 1;
        toc ^= 1;
    }

    {
        // Final iter — second K-half (phase=1) using already-loaded b0_keep
        // from previous iter is not needed in Path 1's simple loop; instead
        // we re-load b0 for k_iters-1 from Bs[tic] which was prefetched at
        // k_iters-2 → Bs[toc] (now Bs[tic] after swap above).
        const int k = k_iters - 1;
        // Apply the >>16 phase shift for odd k (phase 1 on the loaded packs).
        if ((k & 1) == 1) {
            #pragma unroll
            for (int gp = 0; gp < crr_a_pack_count; gp++) {
                a0_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a0_scale_packs[gp]) >> 16);
                a1_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a1_scale_packs[gp]) >> 16);
            }
            #pragma unroll
            for (int gp = 0; gp < crr_b_pack_count; gp++) {
                b0_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b0_scale_packs[gp]) >> 16);
                b1_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b1_scale_packs[gp]) >> 16);
            }
        } else {
            load_raw_scales(k >> 1);
        }
        fp8e8m0_4 b_sel_packs[crr_b_pack_count];
        #pragma unroll
        for (int p = 0; p < crr_b_pack_count; p++) {
            b_sel_packs[p] = (nhalf == 0) ? b0_scale_packs[p] : b1_scale_packs[p];
        }

        load_b(b0, Bs[tic], wn);
        load_a(a, As[tic][0], wm);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase<0>(cA, a, b0, a0_scale_packs, b_sel_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();

        load_a(a, As[tic][1], wm);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase<0>(cC, a, b0, a1_scale_packs, b_sel_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();
    }

    if (wm == 0) {
        __builtin_amdgcn_s_barrier();
    }

    if constexpr (PRESHUFFLED_QUANT) {
        mul(cA, cA, g.scale);
        mul(cC, cC, g.scale);
    }

    // Path 1 store coords. Each block stores BLK_M=256 rows × HB=128 cols.
    //   cA: rows [br*BLK,        br*BLK+RBM*WARPS_M),
    //       cols [bc*HB+wn*RBN,  +RBN)
    //   cC: rows [br*BLK+RBM*WARPS_M, br*BLK+BLK),
    //       cols same as cA
    // Store coord units are RBM=64-row × RBN=32-col tiles. Each WARPS_M-step
    // = WARPS_M tiles (so row index = br*WARPS_M*2 + wm for cA; +WARPS_M for
    // cC). col index = bc * WARPS_N + wn (since each HB-col block has
    // WARPS_N=4 RBN=32-col tiles).
    store(g.c, cA, {0, 0, br * WARPS_M * 2 + wm,           bc * WARPS_N + wn});
    store(g.c, cC, {0, 0, br * WARPS_M * 2 + WARPS_M + wm, bc * WARPS_N + wn});
}

__host__ inline bool crr_can_use_exact_8wave_scaled_rect(const layout_globals& g)
{
    // Path 1: shape gate matches default fastpath; require N divisible by HB
    // (=128) instead of BLK (=256) since our grid uses HB-sized N-tiles.
    return g.m == M_DIM && g.n == N_DIM && g.k == K_DIM
        && (g.m % BLK == 0)
        && (g.n % HB == 0)
        && (g.k % BK == 0);
}

template<bool PRESHUFFLED_QUANT>
__host__ inline void dispatch_crr_exact_8wave_scaled_v2_rect(const layout_globals& g)
{
    // Path 1: grid = (M/BLK_M) * (N/HB) — 2x more blocks vs square in N.
    const dim3 grid((g.m / BLK) * (g.n / HB));
    crr_exact_8wave_scaled_rect_kernel<PRESHUFFLED_QUANT>
        <<<grid, dim3(_NUM_THREADS), 0, g.stream>>>(g);
}

#else  // MXFP8_RECT_PATH2_K_SERIALIZE
// =====================================================================
// PATH 2 (R34 Dev C): HB_N=64 LDS for B + serialised K_HALF=0 reads
// =====================================================================
//
// Strategy: keep the rect-V2 LDS tile geometry (HB_N=64) for B-side, but
// stage the K=0..63 portion in Bs[0][slot] and the K=64..127 portion in
// Bs[1][slot]. Each K-iter then performs:
//   - global_load_b for both half-K tiles (2 loads of 8704 B each — same
//     total bytes as the square 17408 B tile but issued as two ds_writes)
//   - load_col_from_v2_st_half_rect_idx<RT, 0> on Bs[0][tic] — fills data[0..3]
//   - load_col_from_v2_st_half_rect_idx<RT, 4> on Bs[1][tic] — fills data[4..7]
//   - one MMA per accumulator (cA, cC) on the fully-populated b0 register
//
// LDS budget vs Path 1: B-side becomes 4 * 8704 = 34,816 B (Path 1: 2 *
// 17408 = 34,816 B) — bit-identical, so occupancy is unchanged at 2/CU
// (subject to compiler VGPR lift). Verified by build log.
//
// Expected perf: ≥1 LDS-write per global B fetch is doubled (two staging
// tiles per slot rather than one). The two ds_read_b64_tr_b8 issues per
// helper call become four (two per K-half). Path 2 is therefore expected
// to be SLOWER than Path 1; this implementation exists to close out the
// last open lever for rect-V2 CRR — see r34c_findings.md SHIP/NO-SHIP
// verdict.

static_assert(BLK == 256, "Path 2 requires BLK_M=256");
static_assert(BK == 128, "Path 2 requires BK=128");
static_assert(WARPS_M == 2, "Path 2 requires WARPS_M=2");
static_assert(WARPS_N == 4, "Path 2 requires WARPS_N=4");
static_assert(HB == 128, "Path 2 requires HB=128 (square A LDS tile)");
static_assert(HB_N == 64, "Path 2 requires HB_N=64 (rect B LDS tile)");
static_assert(RBN == 32, "Path 2 requires RBN=32 (square reg tile)");
static_assert(!CRR_ROW_SHARED_TRANSPOSE, "Path 2 only supports strict shared layout");
static_assert(!CRR_USE_V3_SWIZZLE, "Path 2 wired for v2/v2a shared layouts");

template<bool PRESHUFFLED_QUANT>
__global__ __launch_bounds__(_NUM_THREADS, GEMM_MIN_BLOCKS_PER_CU)
void crr_exact_8wave_scaled_rect_kernel(const layout_globals g)
{
    using ST_crr_a = ST_v2a;                                              // square A: HB=128, BK=128
    using ST_crr_b_rect = st_fp8e4m3<HB_N, BK, st_16x128_v2_s>;           // rect B: 64x128 (HB_N=64)

    constexpr int crr_a_pack_count = RBM / 32;       // 2
    constexpr int crr_b_pack_count = (RBN + 31) / 32; // 1

    const int k_iters = g.k / BK;
    const int blocks_per_col_hb = g.n / HB;

    // Path 2 LDS: A unchanged (square), B uses TWO HB_N=64 tiles per slot.
    __shared__ ST_crr_a As[2][2];
    __shared__ ST_crr_b_rect Bs[2][2];   // Bs[k_half][slot]

    A_col_reg a;
    B_col_reg b0;
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cC;
    zero(cA);
    zero(cC);

    const int bid = blockIdx.x;
    const int br = bid / blocks_per_col_hb;
    const int bc = bid % blocks_per_col_hb;
    const int bc_orig = bc >> 1;
    const int nhalf = bc & 1;
    const int wm = warpid() / WARPS_N;
    const int wn = warpid() % WARPS_N;

    const int lane_nonk = kittens::laneid() % 16;
    const int lane_kblk = kittens::laneid() / 16;

    auto crr_scale_a_base = [&](int half) {
        return br * BLK + half * (BLK / 2) + wm * RBM;
    };
    auto crr_scale_b_base = [&](int half) {
        return bc_orig * BLK + half * (BLK / 2) + wn * RBN;
    };

    fp8e8m0_4 a0_scale_packs[crr_a_pack_count];
    fp8e8m0_4 a1_scale_packs[crr_a_pack_count];
    fp8e8m0_4 b0_scale_packs[crr_b_pack_count];
    fp8e8m0_4 b1_scale_packs[crr_b_pack_count];

    i32x4 a_v2_srsrc = {}, b_v2_srsrc = {};
    if constexpr (PRESHUFFLED_QUANT) {
        const int padded_k_blocks =
            (g.k / 32 + 7) & ~7;
        const size_t slab_bytes_a = 128u * padded_k_blocks;
        const size_t slab_bytes_b = 64u * padded_k_blocks;
        const size_t slab_idx_a = static_cast<size_t>(br) * WARPS_M + wm;
        const size_t slab_idx_b = static_cast<size_t>(bc_orig) * WARPS_N + wn;
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
        a_v2_srsrc = make_v2_srd(a_v2_base, static_cast<uint32_t>(slab_bytes_a));
        b_v2_srsrc = make_v2_srd(b_v2_base, static_cast<uint32_t>(slab_bytes_b));
    }

    auto load_raw_scales = [&](int k_pair) __attribute__((always_inline)) {
        if constexpr (PRESHUFFLED_QUANT) {
            const uint32_t a_voff =
                (static_cast<uint32_t>(lane_kblk) << 8) |
                (static_cast<uint32_t>(lane_nonk) << 4);
            const uint32_t a_soff = static_cast<uint32_t>(k_pair) << 10;
            const __uint128_t a_raw =
                llvm_amdgcn_raw_buffer_load_b128(a_v2_srsrc, a_voff, a_soff, MXFP8_CRR_V2_SCALE_CACHEPOLICY);
            a0_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw      ));
            a1_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw >> 32));
            if constexpr (crr_a_pack_count > 1) {
                a0_scale_packs[1] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw >> 64));
                a1_scale_packs[1] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw >> 96));
            }
            const uint32_t b_voff =
                (static_cast<uint32_t>(lane_kblk) << 7) |
                (static_cast<uint32_t>(lane_nonk) << 3);
            const uint32_t b_soff = static_cast<uint32_t>(k_pair) << 9;
            const uint64_t b_raw =
                llvm_amdgcn_raw_buffer_load_b64(b_v2_srsrc, b_voff, b_soff, MXFP8_CRR_V2_SCALE_CACHEPOLICY);
            b0_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(b_raw      ));
            b1_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(b_raw >> 32));
            return;
        }
    };

    constexpr int bptA = ST_crr_a::underlying_subtile_bytes_per_thread;
    constexpr int bpmA = bptA * _NUM_THREADS;
    constexpr int mptA = ST_crr_a::rows * ST_crr_a::cols * sizeof(fp8e4m3) / bpmA;
    uint32_t soA[mptA];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);

    constexpr int bptB = ST_crr_b_rect::underlying_subtile_bytes_per_thread;
    constexpr int bpmB = bptB * _NUM_THREADS;
    constexpr int mptB = ST_crr_b_rect::rows * ST_crr_b_rect::cols * sizeof(fp8e4m3) / bpmB;
    uint32_t soB[mptB];
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

    // B coord: row=k (K-block id of HB_N=64 K-rows), col=s (N-block id of BK=128 N-cols).
    // For square BK=128, each K-iter k spans BK=128 K-elements; with HB_N=64 we
    // need 2 K-blocks per K-iter. K_HALF=0 is K-block (2k+0); K_HALF=1 is (2k+1).
    auto a_co = [&](int s, int k) -> coord<ST_crr_a> { return {0, 0, k, s}; };
    auto b_co_lo = [&](int k) -> coord<ST_crr_b_rect> { return {0, 0, 2*k,     bc}; };
    auto b_co_hi = [&](int k) -> coord<ST_crr_b_rect> { return {0, 0, 2*k + 1, bc}; };
    auto global_load_a = [&](ST_crr_a& tile, int s, int k) {
        G::load(tile, g.a, a_co(s, k), soA);
    };
    auto global_load_b_lo = [&](ST_crr_b_rect& tile, int k) {
        G::load(tile, g.b, b_co_lo(k), soB);
    };
    auto global_load_b_hi = [&](ST_crr_b_rect& tile, int k) {
        G::load(tile, g.b, b_co_hi(k), soB);
    };
    auto load_a = [&](A_col_reg& dst, ST_crr_a& tile, int wi) {
        load_col_from_v2a_st(dst, tile, wi * RBM);
    };
    // Path 2 B-side load: 2 K_HALF=0 reads from 2 separate HB_N=64 tiles.
    auto load_b = [&](B_col_reg& dst,
                      ST_crr_b_rect& tile_lo,
                      ST_crr_b_rect& tile_hi,
                      int wi) {
        load_col_from_v2_st_half_rect_idx<B_col_reg, 0, HB_N>(dst, tile_lo, wi * RBN);
        load_col_from_v2_st_half_rect_idx<B_col_reg, 4, HB_N>(dst, tile_hi, wi * RBN);
    };

    int tic = 0, toc = 1;
    global_load_b_lo(Bs[0][tic], 0);
    global_load_b_hi(Bs[1][tic], 0);
    global_load_a(As[tic][0], br * 2,     0);
    global_load_a(As[tic][1], br * 2 + 1, 0);

    if (wm == 1) {
        __builtin_amdgcn_s_barrier();
    }
    TK_WAIT_VMCNT(CRR_INIT0_VMCNT);
    __builtin_amdgcn_s_barrier();

    global_load_b_lo(Bs[0][toc], 1);
    global_load_b_hi(Bs[1][toc], 1);
    global_load_a(As[toc][0], br * 2,     1);
    global_load_a(As[toc][1], br * 2 + 1, 1);

    TK_WAIT_VMCNT(CRR_INIT1_VMCNT);
    __builtin_amdgcn_s_barrier();

    auto b_packs_for_nhalf = [&]() -> fp8e8m0_4* {
        return (nhalf == 0) ? b0_scale_packs : b1_scale_packs;
    };

    TK_PRAGMA_UNROLL(CRR_MAIN_UNROLL)
    for (int k = 0; k < k_iters - 2; k++, tic ^= 1, toc ^= 1) {
        if ((k & 1) == 0) {
            load_raw_scales(k >> 1);
        } else {
            #pragma unroll
            for (int gp = 0; gp < crr_a_pack_count; gp++) {
                a0_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a0_scale_packs[gp]) >> 16);
                a1_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a1_scale_packs[gp]) >> 16);
            }
            #pragma unroll
            for (int gp = 0; gp < crr_b_pack_count; gp++) {
                b0_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b0_scale_packs[gp]) >> 16);
                b1_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b1_scale_packs[gp]) >> 16);
            }
        }
        fp8e8m0_4 b_sel_packs[crr_b_pack_count];
        #pragma unroll
        for (int p = 0; p < crr_b_pack_count; p++) {
            b_sel_packs[p] = (nhalf == 0) ? b0_scale_packs[p] : b1_scale_packs[p];
        }

        load_b(b0, Bs[0][tic], Bs[1][tic], wn);
        load_a(a, As[tic][0], wm);
        TK_WAIT_LGKM(CRR_EXACT_PREFETCH_LGKM);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_from_packs_fixed_phase(cA, a, b0, a0_scale_packs, b_sel_packs);
        CRR_MMA_END();
        CRR_STEADY_MID_BARRIER();
        CRR_SCHED_BARRIER();

        load_a(a, As[tic][1], wm);
        global_load_a(As[tic][0], br * 2,     k + 2);
        global_load_b_lo(Bs[0][tic], k + 2);
        global_load_b_hi(Bs[1][tic], k + 2);
        TK_WAIT_VMCNT(CRR_EXACT_STEADY_VMCNT);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_from_packs_fixed_phase(cC, a, b0, a1_scale_packs, b_sel_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();
        global_load_a(As[tic][1], br * 2 + 1, k + 2);
    }

    // Tail
    {
        const int k = k_iters - 2;
        load_raw_scales(k >> 1);
        fp8e8m0_4 b_sel_packs[crr_b_pack_count];
        #pragma unroll
        for (int p = 0; p < crr_b_pack_count; p++) {
            b_sel_packs[p] = (nhalf == 0) ? b0_scale_packs[p] : b1_scale_packs[p];
        }

        load_b(b0, Bs[0][tic], Bs[1][tic], wn);
        load_a(a, As[tic][0], wm);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase<0>(cA, a, b0, a0_scale_packs, b_sel_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();

        load_a(a, As[tic][1], wm);
        TK_WAIT_VMCNT(CRR_EPILOGUE_VMCNT);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase<0>(cC, a, b0, a1_scale_packs, b_sel_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();
        tic ^= 1;
        toc ^= 1;
    }

    {
        const int k = k_iters - 1;
        if ((k & 1) == 1) {
            #pragma unroll
            for (int gp = 0; gp < crr_a_pack_count; gp++) {
                a0_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a0_scale_packs[gp]) >> 16);
                a1_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a1_scale_packs[gp]) >> 16);
            }
            #pragma unroll
            for (int gp = 0; gp < crr_b_pack_count; gp++) {
                b0_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b0_scale_packs[gp]) >> 16);
                b1_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b1_scale_packs[gp]) >> 16);
            }
        } else {
            load_raw_scales(k >> 1);
        }
        fp8e8m0_4 b_sel_packs[crr_b_pack_count];
        #pragma unroll
        for (int p = 0; p < crr_b_pack_count; p++) {
            b_sel_packs[p] = (nhalf == 0) ? b0_scale_packs[p] : b1_scale_packs[p];
        }

        load_b(b0, Bs[0][tic], Bs[1][tic], wn);
        load_a(a, As[tic][0], wm);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase<0>(cA, a, b0, a0_scale_packs, b_sel_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();

        load_a(a, As[tic][1], wm);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase<0>(cC, a, b0, a1_scale_packs, b_sel_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();
    }

    if (wm == 0) {
        __builtin_amdgcn_s_barrier();
    }

    if constexpr (PRESHUFFLED_QUANT) {
        mul(cA, cA, g.scale);
        mul(cC, cC, g.scale);
    }

    store(g.c, cA, {0, 0, br * WARPS_M * 2 + wm,           bc * WARPS_N + wn});
    store(g.c, cC, {0, 0, br * WARPS_M * 2 + WARPS_M + wm, bc * WARPS_N + wn});
}

__host__ inline bool crr_can_use_exact_8wave_scaled_rect(const layout_globals& g)
{
    return g.m == M_DIM && g.n == N_DIM && g.k == K_DIM
        && (g.m % BLK == 0)
        && (g.n % HB == 0)
        && (g.k % BK == 0);
}

template<bool PRESHUFFLED_QUANT>
__host__ inline void dispatch_crr_exact_8wave_scaled_v2_rect(const layout_globals& g)
{
    const dim3 grid((g.m / BLK) * (g.n / HB));
    crr_exact_8wave_scaled_rect_kernel<PRESHUFFLED_QUANT>
        <<<grid, dim3(_NUM_THREADS), 0, g.stream>>>(g);
}

#endif // MXFP8_RECT_PATH2_K_SERIALIZE

#endif // MXFP8_CRR_RECT_8WAVE_FAST_ENABLE
#endif // MXFP8_RECT_BLK_N == 64
