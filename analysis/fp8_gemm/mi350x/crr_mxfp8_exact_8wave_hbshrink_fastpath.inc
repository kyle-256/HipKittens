#pragma once

// R35 Dev B — HB shrink (BLK_M=128, HB_M=64) V2-CRR exact 8-wave fastpath
// scaffolding. Parallel template to crr_mxfp8_exact_8wave_fastpath.inc with
// the M-direction block geometry halved. Mirrors the structural pattern of
// the rect-V2 CRR Path 1 scaffolding (crr_mxfp8_exact_8wave_rect_fastpath.inc)
// which halves the N-direction block geometry; here we mirror the same
// approach to halve the M-direction block geometry.
//
// PARADIGM PIVOT (R34 Dev D §3.2 Option 1):
//   R32 Dev B identified that V2-CRR LDS SB pipelining hits a -15% structural
//   ceiling at PIPE=3 because A_col_reg=32 VGPR/wave + accumulator=128 VGPR
//   leaves no room for the `a_next` register PIPE=2..3 needs. R34 Dev D Stage
//   2 REFUTED the sub-RBM (RBM=32) hypothesis: shrinking the operand-tile
//   doubles the accumulator vector count, leaving total VGPR worse not
//   better. The pivot identified by R34 Dev D is to instead shrink the
//   per-warp M-coverage (HB) — this actually reduces accumulator vector
//   count.
//
// HB SHRINK (this file) APPROACH:
//   * BLK_M = 128 (was 256). Block now covers HB_M = BLK_M / 2 = 64 rows
//     per M-half. Each warp's M-coverage = HB_M = 64 rows.
//   * WARPS_M = 2 (unchanged). Each warp covers 1 M-half-tile via RBM=64
//     (single stride per warp; was 2 strides at production HB=128).
//   * Grid doubles in M: blocks_per_row = M / BLK_M (= 2x default).
//   * Drop cC/cD: only cA/cB remain (top-half only). Accumulator footprint
//     drops from 4 × 32 = 128 VGPR to 2 × 32 = 64 VGPR per warp.
//
// EXPECTED RESOURCE DELTA (Stage A3 prediction):
//   * Accumulator VGPR/wave:  128 → 64    (-64)
//   * Operand A VGPR/wave:    32  → 32    (0; A_col_reg unchanged at RBM=64)
//   * Operand B VGPR/wave:    16+16 → 16+16 (0; b0/b1 unchanged)
//   * Net VGPR/wave:          ~234 → ~170 (predicted; saves ~64 VGPR)
//   * LDS bytes/block:        139264 → ~69632 (cuts in half — A side drops
//     from As[2][2] to As[2][1] since each block is only one M-half-tile)
//
// SCOPE / CAVEATS (Stage A1+A2 scaffolding only):
//   * Only the V2 SCALE_VERSION=2 path is provided (PRESHUFFLED_QUANT=true).
//     V1 path can be added as straightforward extension.
//   * Host-side preshuffle is UNCHANGED (we mirror rect-V2 Path 1's trick of
//     keeping square LDS tile geometry and re-deriving the slab indexing
//     from the doubled-grid kernel coordinates).
//   * Single-buffered LDS schedule (mirror SUBRBM Stage 2b skeleton). The
//     full DB main loop + SB pipelining variants can be added in R36+.
//   * Dispatch is NOT wired into kernel_mxfp8_layouts.cpp's gemm_*_pq_v2 path
//     this cycle. Wiring up is a single edit when the next cycle picks this
//     up — see r35b_findings.md §X.
//
// Reference (R34 Dev D r34d_findings.md §3.2):
// > Real path past the structural ceiling: smaller M-extent per warp.
// > Option 1: Halve HB → new BLK=128 (M-direction smaller block) — halves
// > both stride count AND accumulator count. This is a different scaffolding
// > (touches block-tile geometry, scale slab math, dispatch grid).
// This file IS that "smaller M-extent" lever as a parallel template.

// HB shrink macro: when defined to 128, the parallel HB shrink kernel
// activates. Default 256 = no change (default fastpath remains source of
// truth). Independent of MXFP8_CRR_RBM (R33 Dev D sub-RBM scaffolding) and
// MXFP8_RECT_BLK_N (R28 Dev D rect-V2 scaffolding) — those are orthogonal
// progressive macros.
#ifndef MXFP8_CRR_BLK_M
#define MXFP8_CRR_BLK_M 256
#endif
#if (MXFP8_CRR_BLK_M != 128) && (MXFP8_CRR_BLK_M != 256)
#error "MXFP8_CRR_BLK_M must be 128 or 256 (R35 Dev B scaffolding)"
#endif

#if (MXFP8_CRR_BLK_M == 128)

#ifndef MXFP8_CRR_HBSHRINK_8WAVE_FAST_ENABLE
#define MXFP8_CRR_HBSHRINK_8WAVE_FAST_ENABLE 1
#endif

#if MXFP8_CRR_HBSHRINK_8WAVE_FAST_ENABLE

// HB shrink invariants. We KEEP square A LDS tile (HB=128, BK=128) so the
// host preshuffle is unchanged — we just consume HALF of each square A
// LDS tile per block (mhalf selects top vs bottom). This mirrors rect-V2
// Path 1's halved N-work approach (Dev A R33 Stage A2).
static_assert(BLK == 256, "HB shrink CRR fast path requires BLK_M (default) = 256 — we halve to BLK_M=128 internally");
static_assert(BK == 128, "HB shrink CRR fast path requires BK=128");
static_assert(WARPS_M == 2, "HB shrink CRR fast path requires WARPS_M=2");
static_assert(WARPS_N == 4, "HB shrink CRR fast path requires WARPS_N=4");
static_assert(HB == 128, "HB shrink CRR fast path requires HB=128 (square LDS tile)");
static_assert(RBM == 64, "HB shrink CRR fast path requires RBM=64 (square reg tile)");
static_assert(RBN == 32, "HB shrink CRR fast path requires RBN=32 (square reg tile)");
static_assert(!CRR_ROW_SHARED_TRANSPOSE, "HB shrink CRR fast path only supports strict shared layout");
static_assert(!CRR_USE_V3_SWIZZLE, "HB shrink CRR fast path is wired for v2/v2a shared layouts");

// HB shrink internal block-tile constants. These name the *effective*
// block geometry the HB shrink kernel produces; the LDS tile geometry
// is still the square HB=128.
constexpr int HBSHRINK_BLK_M = 128;          // halved M-direction block
constexpr int HBSHRINK_HB_M  = HBSHRINK_BLK_M / 2;  // 64 = per-warp M coverage
// A LDS tile geometry kept square (HB=128); we consume only HALF per block
// via the mhalf selector below.

// ---------------------------------------------------------------------------
// Stage A2: HB shrink V2-CRR exact 8-wave scaled kernel skeleton.
//
// Each block processes HB_M=64 (M) × BLK_N=256 (N) of C. Compared to the
// default kernel (BLK_M=256 × BLK_N=256):
//   * Per-warp M-coverage: HB_M=64 (was HB=128).
//   * M-strides per warp:  1 (was 2).
//   * Accumulator vectors: 2 (cA, cB) instead of 4 (cA, cB, cC, cD) —
//     drops 64 VGPR/wave from the accumulator footprint.
//   * Grid:  blocks_per_row = M / HBSHRINK_BLK_M = 2 × default.
//
// Uses single-buffered LDS schedule for skeleton simplicity. R36+ can
// rewire to DB main loop + SB pipelining variants.
// ---------------------------------------------------------------------------
template<bool PRESHUFFLED_QUANT, int SCALE_VERSION = 2>
__global__ __launch_bounds__(_NUM_THREADS, GEMM_MIN_BLOCKS_PER_CU)
void crr_exact_8wave_scaled_hbshrink_kernel(const layout_globals g)
{
    using ST_crr_a = ST_v2a;     // square A LDS tile (HB=128, BK=128)
    using ST_crr_b = ST_v2;      // square B LDS tile (HB=128, BK=128)

    constexpr int crr_a_pack_count = RBM / 32;        // 2
    constexpr int crr_b_pack_count = (RBN + 31) / 32; // 1 for RBN=32

    const int k_iters = g.k / BK;
    // Grid uses HBSHRINK_BLK_M (= 128) for M-direction. blocks_per_col is
    // unchanged since N geometry is unchanged.
    const int blocks_per_col = g.n / BLK;

    // LDS: one A tile per slot (we only need ONE M-half-tile per block);
    // B unchanged (two N-halves per block).
    __shared__ ST_crr_a As[2];
    __shared__ ST_crr_b Bs[2][2];

    A_col_reg a;
    B_col_reg b0, b1;
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB;
    zero(cA);
    zero(cB);

    const int bid = blockIdx.x;
    // M-direction grid is doubled: br indexes HBSHRINK_BLK_M=128 rows.
    // br_orig = br/2 indexes the original square BLK=256 row-block; mhalf
    // = br & 1 picks which M-half (0 = top, 1 = bot).
    const int br      = bid / blocks_per_col;
    const int bc      = bid % blocks_per_col;
    const int br_orig = br >> 1;
    const int mhalf   = br & 1;
    const int wm      = warpid() / WARPS_N;
    const int wn      = warpid() % WARPS_N;

    const int lane_nonk = kittens::laneid() % 16;
    const int lane_kblk = kittens::laneid() / 16;
    (void)lane_nonk; (void)lane_kblk;

    // Scale base: B side unchanged. A side uses br_orig * BLK + mhalf*(BLK/2)
    // to address the original square BLK-block scale slab — we still load
    // the same scale layout since host preshuffle is unchanged.
    auto crr_scale_a_base = [&](int local_half) {
        // local_half is always 0 in the HB shrink kernel (each block covers
        // ONE M-half = HB_M=64 rows = a single RBM=64 stride per warp).
        // The "global" half of the original square BLK-block is mhalf,
        // which selects which scale-pack of the square slab we use.
        (void)local_half;
        return br_orig * BLK + mhalf * (BLK / 2) + wm * RBM;
    };
    auto crr_scale_b_base = [&](int half) {
        return bc * BLK + half * (BLK / 2) + wn * RBN;
    };

    // Scale-pack arrays. A side: only ONE half active (we only do top-half
    // MMA in this kernel); we still allocate a0/a1 to keep the control
    // flow parallel to the default kernel and let the slab loader emit
    // the same b128 fetch (the unused half is then DCEd).
    fp8e8m0_4 a0_scale_packs[crr_a_pack_count];
    fp8e8m0_4 a1_scale_packs[crr_a_pack_count];
    fp8e8m0_4 b0_scale_packs[crr_b_pack_count];
    fp8e8m0_4 b1_scale_packs[crr_b_pack_count];
    const uint8_t* a0_scale_row_bases[crr_a_pack_count];
    const uint8_t* a1_scale_row_bases[crr_a_pack_count];
    const uint8_t* b0_scale_row_bases[crr_b_pack_count];
    const uint8_t* b1_scale_row_bases[crr_b_pack_count];

    // V2 wave-tile SRDs — same slab geometry as square (host preshuffle
    // unchanged). slab_idx_a uses br_orig (NOT br) to address the original
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
        // V2-CRR slab geometry — identical to square (host preshuffle
        // unchanged).
        //   slab_idx_a = br_orig * WARPS_M + wm  (br_orig, not br)
        //   slab_idx_b = bc      * WARPS_N + wn
        const int padded_k_blocks =
            (g.k / 32 + 7) & ~7;
        const size_t slab_bytes_a = 128u * padded_k_blocks;
        const size_t slab_bytes_b = 64u * padded_k_blocks;
        const size_t slab_idx_a =
            static_cast<size_t>(br_orig) * WARPS_M + wm;
        const size_t slab_idx_b =
            static_cast<size_t>(bc) * WARPS_N + wn;
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
            // V2-CRR: 1 b128 fetches 4 A packs (a0p0, a1p0, a0p1, a1p1),
            // 1 b64 fetches 2 B packs (b0p0, b1p0). Layout unchanged from
            // square — host preshuffle unchanged.
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
    G::prefill_swizzled_offsets(As[0], g.a, soA);

    constexpr int bptB = ST_crr_b::underlying_subtile_bytes_per_thread;
    constexpr int bpmB = bptB * _NUM_THREADS;
    constexpr int mptB = ST_crr_b::rows * ST_crr_b::cols * sizeof(fp8e4m3) / bpmB;
    uint32_t soB[mptB];
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

    // A coord: each block loads ONE square A LDS tile (HB=128, BK=128).
    // The "row" dimension uses (br_orig * 2 + mhalf) — for mhalf=0 we load
    // the top half of the square BLK-block (rows 0..127), for mhalf=1 the
    // bottom half (rows 128..255). The square A LDS tile is 128 rows, so
    // this single load covers exactly one M-half.
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

    // Single-buffered LDS schedule (mirrors SUBRBM Stage 2b skeleton).
    // The simplest correct form: load → barrier → compute → barrier → load.
    // Per-iter we issue the next-iter VMEM->LDS at end and waitcnt at top.
    // The A-side global coord uses (br_orig * 2 + mhalf) — mhalf-aware.
    global_load_b(Bs[0][0], bc * 2,     0);
    global_load_a(As[0],    br_orig * 2 + mhalf, 0);
    global_load_b(Bs[0][1], bc * 2 + 1, 0);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    // Pick the right A scale-pack pair for this mhalf.
    auto a_packs_for_mhalf = [&]() -> fp8e8m0_4* {
        return (mhalf == 0) ? a0_scale_packs : a1_scale_packs;
    };
    (void)a_packs_for_mhalf;

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
        // Build a local mhalf-selected A-pack array (compiler can't const-
        // fold mhalf, so this is a runtime select per K-iter — same pattern
        // as rect-V2 Path 1's b_sel_packs for nhalf).
        fp8e8m0_4 a_sel_packs[crr_a_pack_count];
        #pragma unroll
        for (int p = 0; p < crr_a_pack_count; p++) {
            a_sel_packs[p] = (mhalf == 0) ? a0_scale_packs[p] : a1_scale_packs[p];
        }

        load_b(b0, Bs[0][0], wn);
        load_b(b1, Bs[0][1], wn);
        load_a(a, As[0], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        crr_mma_scaled_from_packs_fixed_phase(cA, a, b0, a_sel_packs, b0_scale_packs);
        crr_mma_scaled_from_packs_fixed_phase(cB, a, b1, a_sel_packs, b1_scale_packs);

        __builtin_amdgcn_s_barrier();
        if (k + 1 < k_iters) {
            global_load_b(Bs[0][0], bc * 2,     k + 1);
            global_load_a(As[0],    br_orig * 2 + mhalf, k + 1);
            global_load_b(Bs[0][1], bc * 2 + 1, k + 1);
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_s_barrier();
        }
    }

    if (wm == 0) {
        __builtin_amdgcn_s_barrier();
    }

    if constexpr (PRESHUFFLED_QUANT) {
        mul(cA, cA, g.scale);
        mul(cB, cB, g.scale);
    }

    // Store coords. Each block stores HBSHRINK_BLK_M=128 rows × BLK_N=256
    // cols. Row index uses (br_orig * 2 + mhalf) * WARPS_M + wm — the
    // doubled-grid M-index hits the right RBM=64 macro-row of the output.
    //   cA: rows = (br_orig*2 + mhalf)*WARPS_M + wm,
    //       cols = bc*WARPS_N*2 + wn
    //   cB: rows same as cA,
    //       cols = bc*WARPS_N*2 + WARPS_N + wn
    // Note: original kernel used "br * WARPS_M * 2 + wm" for cA (top-half)
    // and "br * WARPS_M * 2 + WARPS_M + wm" for cC (bot-half). With HB
    // shrink, br_orig replaces br, and (mhalf * WARPS_M) replaces the
    // top/bot offset.
    const int macro_row =
        br_orig * WARPS_M * 2 + mhalf * WARPS_M + wm;
    store(g.c, cA, {0, 0, macro_row, bc * WARPS_N * 2 + wn});
    store(g.c, cB, {0, 0, macro_row, bc * WARPS_N * 2 + WARPS_N + wn});
}

// Host-side dispatch entry: NOT wired into gemm_*_pq_v2 path in
// kernel_mxfp8_layouts.cpp. Kept here so future authors can switch the
// dispatcher over with a single edit.
__host__ inline bool crr_can_use_exact_8wave_scaled_hbshrink(const layout_globals& g)
{
    // Shape gate: M divisible by HBSHRINK_BLK_M=128 (i.e., even number of
    // BLK=256-blocks in M). N divisible by BLK=256 (unchanged). K
    // divisible by BK=128 (unchanged).
    return g.m == M_DIM && g.n == N_DIM && g.k == K_DIM
        && (g.m % HBSHRINK_BLK_M == 0)
        && (g.n % BLK == 0)
        && (g.k % BK == 0);
}

template<bool PRESHUFFLED_QUANT>
__host__ inline void dispatch_crr_exact_8wave_scaled_v2_hbshrink(const layout_globals& g)
{
    // Grid: (M/HBSHRINK_BLK_M) * (N/BLK) — 2x more blocks vs square in M.
    const dim3 grid((g.m / HBSHRINK_BLK_M) * (g.n / BLK));
    crr_exact_8wave_scaled_hbshrink_kernel<PRESHUFFLED_QUANT, 2>
        <<<grid, dim3(_NUM_THREADS), 0, g.stream>>>(g);
}

// Force-instantiation probe: when MXFP8_CRR_HBSHRINK_PROBE=1 we emit a
// host stub that takes the address of the dispatcher to force template
// instantiation and produce a kernel symbol in the resource-usage report.
// Default 0 (no probe — Stage A1 builds just need the file to parse).
// Set to 1 to confirm the kernel emits with realistic VGPR/LDS numbers.
#ifndef MXFP8_CRR_HBSHRINK_PROBE
#define MXFP8_CRR_HBSHRINK_PROBE 0
#endif
#if MXFP8_CRR_HBSHRINK_PROBE
__host__ inline void* crr_hbshrink_probe_addr_handle()
{
    return reinterpret_cast<void*>(&dispatch_crr_exact_8wave_scaled_v2_hbshrink<true>);
}
#endif

#endif  // MXFP8_CRR_HBSHRINK_8WAVE_FAST_ENABLE

#endif  // MXFP8_CRR_BLK_M == 128
