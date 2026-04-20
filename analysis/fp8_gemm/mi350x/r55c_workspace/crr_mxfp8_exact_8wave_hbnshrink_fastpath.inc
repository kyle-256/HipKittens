#pragma once

// R38 Dev A — HB-N shrink (BLK_N=128, HB_N=64) V2-CRR exact 8-wave fastpath.
//
// SYMMETRIC MIRROR of R36 Dev A's HB-M shrink (BLK_M=128, HB_M=64).
// Whereas HB-M shrink halves the per-block M-coverage (drops cC/cD), HB-N
// shrink halves the per-block N-coverage (drops cB/cD). This is targeted at
// wide-N shapes (8B Gate/Up 4096×14336×4096, 70B Gate/Up 4096×28672×8192,
// 8B-Down 4096×4096×14336) where the M-direction grid-doubling overhead of
// HB-M shrink kills perf (R37 Dev B/C established -17% to -28% on wide-N),
// but where the N grid is already enormous so an additional 2x N-grid
// expansion is amortizable.
//
// PARADIGM:
//   HB-M shrink:  drops bottom M-half (no As[1]/cC/cD); doubles M-grid.
//                 Wins on N=1024 tall-thin rect (R37 SHIP).
//                 Loses on wide-N (M-grid 2x → ctile count 2x → scheduling
//                 overhead dominates per-ctile pipeline win).
//   HB-N shrink:  drops second N-half (no Bs[1]/cB/cD); doubles N-grid.
//                 Hypothesis: wins on wide-N (where the N-grid was
//                 already much larger than the M-grid, so doubling it is
//                 fractionally smaller scheduling overhead than M-doubling
//                 was on tall-thin shapes).
//
// HB-N SHRINK APPROACH (mirror of HB-M):
//   * BLK_N = 128 (was BLK = 256). Block now covers HB_N = BLK_N / 2 = 64
//     cols per N-half. Each warp's N-coverage = HB_N / WARPS_N = 16... no,
//     each warp gets RBN=32 cols (this is the per-warp tile minimum, must
//     not change). With WARPS_N=4 and RBN=32, the natural N-half coverage
//     is WARPS_N * RBN = 128 cols, which is exactly BLK_N=128. So WE DROP
//     THE SECOND HALF — the per-block N-coverage becomes ONE half-tile
//     of 128 cols = 4 warps × 32 cols.
//   * WARPS_M = 2, WARPS_N = 4 (unchanged).
//   * Accumulator vectors: 2 (cA, cC) instead of 4 (cA, cB, cC, cD).
//     cA = top M-half × first N-half;  cC = bot M-half × first N-half.
//     cB and cD (which used the second N-half) are dropped.
//   * Grid doubles in N: blocks_per_col = N / BLK_N = 2 × default.
//
// VGPR REDUCTION (forecast, mirror of HB-M):
//   * Default V2-CRR: 234 VGPR.
//   * HB-M shrink: 160 VGPR (-74) at PIPE=1 cross-buffer DB.
//   * HB-N shrink (forecast): 160 VGPR similar — each accumulator vector
//     is the same size; we drop 2 of 4, saving same accumulator footprint.
//
// MXFP8_CRR_HBNSHRINK_PIPELINE selects pipelining strategy:
//   0 = Stage A skeleton (single-buffered, no cross-iter prefetch). Match
//       HB-M Stage A, expected -10% to -15% perf as bare skeleton.
//   1 = Stage B1: classic cross-buffer DB main loop with cycle-2 prefetch.
//       Mirror of HB-M Stage B1 which delivered +28-30% on tall-thin rect.
//
// GATING: Macro MXFP8_CRR_BLK_N controls block N size. Default 256 = no
// change. =128 selects HB-N shrink path. ORTHOGONAL to MXFP8_CRR_BLK_M
// (HB-M shrink) — both can be enabled independently in production .so.
//
// Default build (no MXFP8_CRR_BLK_N define) compiles to an EMPTY translation
// unit — verified via `nm -D | grep hbnshrink` = 0 symbols (R37 Dev C's
// recommended dead-code gate, since md5 unreliable in this env).

#ifndef MXFP8_CRR_BLK_N
#define MXFP8_CRR_BLK_N 256
#endif
#if (MXFP8_CRR_BLK_N != 128) && (MXFP8_CRR_BLK_N != 256)
#error "MXFP8_CRR_BLK_N must be 128 or 256 (R38 Dev A HB-N shrink scaffolding)"
#endif

#if (MXFP8_CRR_BLK_N == 128)

#ifndef MXFP8_CRR_HBNSHRINK_8WAVE_FAST_ENABLE
#define MXFP8_CRR_HBNSHRINK_8WAVE_FAST_ENABLE 1
#endif

// R38 Dev A pipelining selector. 0 = skeleton (single-buffered, no
// prefetch). 1 = Stage B1 DB main loop + cycle-2 prefetch.
#ifndef MXFP8_CRR_HBNSHRINK_PIPELINE
#define MXFP8_CRR_HBNSHRINK_PIPELINE 0
#endif
#if (MXFP8_CRR_HBNSHRINK_PIPELINE < 0) || (MXFP8_CRR_HBNSHRINK_PIPELINE > 1)
#error "MXFP8_CRR_HBNSHRINK_PIPELINE must be 0 or 1 (R38 Dev A)"
#endif

#if MXFP8_CRR_HBNSHRINK_8WAVE_FAST_ENABLE

// HB-N shrink invariants. We KEEP square B LDS tile (HB=128, BK=128) so the
// host preshuffle is unchanged — we just consume HALF of each square B
// LDS tile per block (nhalf selects first vs second N-half). Mirror of
// HB-M shrink which kept square A LDS tile and consumed half via mhalf.
static_assert(BLK == 256, "HB-N shrink CRR fast path requires BLK_N (default) = 256 — we halve to BLK_N=128 internally");
static_assert(BK == 128, "HB-N shrink CRR fast path requires BK=128");
static_assert(WARPS_M == 2, "HB-N shrink CRR fast path requires WARPS_M=2");
static_assert(WARPS_N == 4, "HB-N shrink CRR fast path requires WARPS_N=4");
static_assert(HB == 128, "HB-N shrink CRR fast path requires HB=128 (square LDS tile)");
static_assert(RBM == 64, "HB-N shrink CRR fast path requires RBM=64 (square reg tile)");
static_assert(RBN == 32, "HB-N shrink CRR fast path requires RBN=32 (square reg tile)");
static_assert(!CRR_ROW_SHARED_TRANSPOSE, "HB-N shrink CRR fast path only supports strict shared layout");
static_assert(!CRR_USE_V3_SWIZZLE, "HB-N shrink CRR fast path is wired for v2/v2a shared layouts");

// HB-N shrink internal block-tile constants. These name the *effective*
// block geometry the HB-N shrink kernel produces; the LDS tile geometry
// is still the square HB=128.
constexpr int HBNSHRINK_BLK_N = 128;          // halved N-direction block
constexpr int HBNSHRINK_HB_N  = HBNSHRINK_BLK_N / 2;  // 64 = per N-half cols
// B LDS tile geometry kept square (HB=128); we consume only HALF per block
// via the nhalf selector below.

// ---------------------------------------------------------------------------
// HB-N shrink V2-CRR exact 8-wave scaled kernel.
//
// Each block processes BLK_M=256 (M) × HBNSHRINK_BLK_N=128 (N) of C.
// Compared to default (BLK_M=256 × BLK_N=256):
//   * Per-block N-coverage: HBNSHRINK_BLK_N=128 (was BLK_N=256).
//   * B LDS tiles: 1 (top half only) instead of 2.
//   * Accumulator vectors: 2 (cA, cC) instead of 4 (cA, cB, cC, cD).
//   * Grid: blocks_per_col = N / HBNSHRINK_BLK_N = 2 × default.
//
// MXFP8_CRR_HBNSHRINK_PIPELINE selects pipelining:
//   0 = Stage A skeleton (single-buffered, R38 Dev A baseline).
//   1 = Stage B1: DB main loop + cycle-2 prefetch (mirror of HB-M B1).
// ---------------------------------------------------------------------------
template<bool PRESHUFFLED_QUANT, int SCALE_VERSION = 2>
__global__ __launch_bounds__(_NUM_THREADS, GEMM_MIN_BLOCKS_PER_CU)
void crr_exact_8wave_scaled_hbnshrink_kernel(const layout_globals g)
{
    using ST_crr_a = ST_v2a;     // square A LDS tile (HB=128, BK=128)
    using ST_crr_b = ST_v2;      // square B LDS tile (HB=128, BK=128)

    constexpr int crr_a_pack_count = RBM / 32;        // 2
    constexpr int crr_b_pack_count = (RBN + 31) / 32; // 1 for RBN=32

    const int k_iters = g.k / BK;
    // N-direction grid is doubled: blocks_per_col_eff steps at HBNSHRINK_BLK_N.
    const int blocks_per_col = g.n / HBNSHRINK_BLK_N;

    // LDS: A unchanged (two M-halves per block); B single-tile per slot
    // (we only need ONE N-half-tile per block). Two slots for double-buffering
    // in Stage B1; only slot [0] used in Stage 0.
    __shared__ ST_crr_a As[2][2];
    __shared__ ST_crr_b Bs[2];

    A_col_reg a;
    B_col_reg b0;
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cC;
    zero(cA);
    zero(cC);

    const int bid = blockIdx.x;
    // N-direction grid is doubled: bc indexes HBNSHRINK_BLK_N=128 cols.
    // bc_orig = bc/2 indexes the original square BLK=256 col-block;
    // nhalf = bc & 1 picks which N-half (0 = first, 1 = second).
    const int br      = bid / blocks_per_col;
    const int bc      = bid % blocks_per_col;
    const int bc_orig = bc >> 1;
    const int nhalf   = bc & 1;
    const int wm      = warpid() / WARPS_N;
    const int wn      = warpid() % WARPS_N;

    const int lane_nonk = kittens::laneid() % 16;
    const int lane_kblk = kittens::laneid() / 16;
    (void)lane_nonk; (void)lane_kblk;

    // Scale base: A side unchanged. B side uses bc_orig * BLK + nhalf*(BLK/2)
    // to address the original square BLK-block scale slab — host preshuffle
    // unchanged; we still load the same scale layout.
    auto crr_scale_a_base = [&](int half) {
        return br * BLK + half * (BLK / 2) + wm * RBM;
    };
    auto crr_scale_b_base = [&](int local_half) {
        (void)local_half;
        return bc_orig * BLK + nhalf * (BLK / 2) + wn * RBN;
    };

    // Scale-pack arrays. B side: only ONE half active (we only do first-half
    // MMA in this kernel); we still allocate b0/b1 to keep control flow
    // parallel to default, b1 will be DCEd.
    fp8e8m0_4 a0_scale_packs[crr_a_pack_count];
    fp8e8m0_4 a1_scale_packs[crr_a_pack_count];
    fp8e8m0_4 b0_scale_packs[crr_b_pack_count];
    fp8e8m0_4 b1_scale_packs[crr_b_pack_count];
    const uint8_t* a0_scale_row_bases[crr_a_pack_count];
    const uint8_t* a1_scale_row_bases[crr_a_pack_count];
    const uint8_t* b0_scale_row_bases[crr_b_pack_count];
    const uint8_t* b1_scale_row_bases[crr_b_pack_count];

    // V2 wave-tile SRDs — same slab geometry as square (host preshuffle
    // unchanged). slab_idx_b uses bc_orig (NOT bc) to address the original
    // square BLK-block.
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
        const int padded_k_blocks =
            (g.k / 32 + 7) & ~7;
        const size_t slab_bytes_a = 128u * padded_k_blocks;
        const size_t slab_bytes_b = 64u * padded_k_blocks;
        const size_t slab_idx_a =
            static_cast<size_t>(br) * WARPS_M + wm;
        const size_t slab_idx_b =
            static_cast<size_t>(bc_orig) * WARPS_N + wn;
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

    // B coord: each block loads ONE square B LDS tile (HB=128, BK=128).
    // The "col" dimension uses (bc_orig * 2 + nhalf) — for nhalf=0 we load
    // the first half of the square BLK-block (cols 0..127), for nhalf=1
    // the second half (cols 128..255). The square B LDS tile is 128 cols,
    // so this single load covers exactly one N-half.
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

    // The B-side global K-tile coord. With BLK_N=128 we want the first half
    // OR second half of the original BLK=256 col-block. (bc_orig * 2 + nhalf)
    // gives the col-tile id at HB=128 granularity.
    auto b_tile_col = [&]() { return bc_orig * 2 + nhalf; };

#if (MXFP8_CRR_HBNSHRINK_PIPELINE == 0)
    // -----------------------------------------------------------------------
    // Stage A skeleton (R38 Dev A baseline; expected ~-10-15% vs default).
    // Single-buffered LDS schedule: load → barrier → compute → barrier → load.
    // -----------------------------------------------------------------------
    global_load_a(As[0][0], br * 2,     0);
    global_load_b(Bs[0],    b_tile_col(), 0);
    global_load_a(As[0][1], br * 2 + 1, 0);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    for (int k = 0; k < k_iters; k++) {
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

        load_b(b0, Bs[0], wn);
        load_a(a, As[0][0], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        crr_mma_scaled_from_packs_fixed_phase(cA, a, b0, a0_scale_packs, b_sel_packs);
        load_a(a, As[0][1], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        crr_mma_scaled_from_packs_fixed_phase(cC, a, b0, a1_scale_packs, b_sel_packs);

        __builtin_amdgcn_s_barrier();
        if (k + 1 < k_iters) {
            global_load_a(As[0][0], br * 2,     k + 1);
            global_load_b(Bs[0],    b_tile_col(), k + 1);
            global_load_a(As[0][1], br * 2 + 1, k + 1);
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_s_barrier();
        }
    }

#elif (MXFP8_CRR_HBNSHRINK_PIPELINE == 1)
    // -----------------------------------------------------------------------
    // Stage B1: double-buffered LDS main loop with classic cross-buffer
    // prefetch (read tic, prefetch into toc for k+1). Two LDS outer slots
    // eliminate the in-place-prefetch race that R36 Dev A v1 hit on HB-M.
    //
    // Schedule per iter k (0 .. k_iters-2):
    //   read tic for k  →  cA → cC
    //   prefetch toc for k+1 (issued before the wait so VMEM overlaps MMA)
    //   wait vmcnt + lgkm; barrier; tic ^= 1, toc ^= 1
    // Final iter k = k_iters-1 reads tic only (no prefetch).
    // -----------------------------------------------------------------------
    int tic = 0, toc = 1;
    // Initial loads: prime k=0 in As[tic]/Bs[tic].
    global_load_a(As[tic][0], br * 2,     0);
    global_load_b(Bs[tic],    b_tile_col(), 0);
    global_load_a(As[tic][1], br * 2 + 1, 0);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    TK_PRAGMA_UNROLL(CRR_MAIN_UNROLL)
    for (int k = 0; k < k_iters - 1; k++, tic ^= 1, toc ^= 1) {
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

        // Issue prefetch for k+1 into toc (different slot than tic).
        global_load_a(As[toc][0], br * 2,     k + 1);
        global_load_b(Bs[toc],    b_tile_col(), k + 1);
        global_load_a(As[toc][1], br * 2 + 1, k + 1);

        // LDS reads + MMAs from tic — same as skeleton MMA chain order.
        load_b(b0, Bs[tic], wn);
        load_a(a, As[tic][0], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        crr_mma_scaled_from_packs_fixed_phase(cA, a, b0, a0_scale_packs, b_sel_packs);
        load_a(a, As[tic][1], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        crr_mma_scaled_from_packs_fixed_phase(cC, a, b0, a1_scale_packs, b_sel_packs);

        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }

    // Final iter k = k_iters - 1: read tic (toc was just primed by last
    // prefetch). No more VMEM.
    {
        const int k = k_iters - 1;
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
        load_b(b0, Bs[tic], wn);
        load_a(a, As[tic][0], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        crr_mma_scaled_from_packs_fixed_phase(cA, a, b0, a0_scale_packs, b_sel_packs);
        load_a(a, As[tic][1], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        crr_mma_scaled_from_packs_fixed_phase(cC, a, b0, a1_scale_packs, b_sel_packs);
        __builtin_amdgcn_s_barrier();
    }

#endif  // MXFP8_CRR_HBNSHRINK_PIPELINE

    if (wm == 0) {
        __builtin_amdgcn_s_barrier();
    }

    if constexpr (PRESHUFFLED_QUANT) {
        mul(cA, cA, g.scale);
        mul(cC, cC, g.scale);
    }

    // Store coords: each block stores BLK_M=256 rows × HBNSHRINK_BLK_N=128 cols.
    // Default kernel store layout (from rect/HB-M shrink reading):
    //   cA at (br*WARPS_M*2 + 0*WARPS_M + wm, bc*WARPS_N*2 + nhalf*WARPS_N + wn)
    //     — top M-half × selected N-half
    //   cC at (br*WARPS_M*2 + 1*WARPS_M + wm, bc*WARPS_N*2 + nhalf*WARPS_N + wn)
    //     — bot M-half × selected N-half
    // bc here is bc_orig (the original 256-col block index); nhalf selects
    // which N-half (first 4 wn slots, or second 4 wn slots).
    const int macro_col =
        bc_orig * WARPS_N * 2 + nhalf * WARPS_N + wn;
    store(g.c, cA, {0, 0, br * WARPS_M * 2 + 0 * WARPS_M + wm, macro_col});
    store(g.c, cC, {0, 0, br * WARPS_M * 2 + 1 * WARPS_M + wm, macro_col});
}

// Host-side dispatch entry — wired into gemm_*_pq_v2 path in
// kernel_mxfp8_layouts.cpp behind MXFP8_CRR_BLK_N=128.
//
// R38 Dev A — production shape gating: HB-N shrink targets wide-N rect
// shapes (8B Gate/Up, 70B Gate/Up, 8B-Down). Predicate gates on the
// specific shape to avoid affecting square 8192³ or tall-thin shapes
// where HB-N shrink is paradigm-incompatible (all the existing HB-M
// SHIPS prove out for those shapes).
//
// Initial probe: 8B Gate/Up (M=4096, N=14336, K=4096). If positive,
// extend allow-list to 70B Gate/Up (4096×28672×8192) and 8B-Down
// (4096×4096×14336) in follow-on cycles. Initially conservative.
__host__ inline bool crr_can_use_exact_8wave_scaled_hbnshrink(const layout_globals& g)
{
    if (!(g.m == M_DIM && g.n == N_DIM && g.k == K_DIM))
        return false;
    if (!((g.m % BLK == 0) && (g.n % HBNSHRINK_BLK_N == 0) && (g.k % BK == 0)))
        return false;
    // R38 Dev A initial probe shape allow-list — 8B Gate/Up only until
    // proven on additional wide-N shapes.
    const bool is_8b_gateup  = (g.m == 4096 && g.n == 14336 && g.k == 4096);
    const bool is_70b_gateup = (g.m == 4096 && g.n == 28672 && g.k == 8192);
    // R49 Dev E — extend allow-list to 70B Down (4096x4096x14336) for the
    // CRR-variant re-eval sweep. Note 70B Down is K-heavy (not strictly
    // wide-N), so HB-N is paradigm-questionable here; benchmark documents
    // the result rather than claiming a win.
    const bool is_70b_down   = (g.m == 4096 && g.n == 4096 && g.k == 14336);
    return is_8b_gateup || is_70b_gateup || is_70b_down;
}

template<bool PRESHUFFLED_QUANT>
__host__ inline void dispatch_crr_exact_8wave_scaled_v2_hbnshrink(const layout_globals& g)
{
    const dim3 grid((g.m / BLK) * (g.n / HBNSHRINK_BLK_N));
    crr_exact_8wave_scaled_hbnshrink_kernel<PRESHUFFLED_QUANT, 2>
        <<<grid, dim3(_NUM_THREADS), 0, g.stream>>>(g);
}

#ifndef MXFP8_CRR_HBNSHRINK_PROBE
#define MXFP8_CRR_HBNSHRINK_PROBE 0
#endif
#if MXFP8_CRR_HBNSHRINK_PROBE
__host__ inline void* crr_hbnshrink_probe_addr_handle()
{
    return reinterpret_cast<void*>(&dispatch_crr_exact_8wave_scaled_v2_hbnshrink<true>);
}
#endif

#endif  // MXFP8_CRR_HBNSHRINK_8WAVE_FAST_ENABLE

#endif  // MXFP8_CRR_BLK_N == 128
