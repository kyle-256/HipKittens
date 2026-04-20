#pragma once

// R35 Dev C — WARPS_M=4 (× WARPS_N=2) V2-CRR exact 8-wave fastpath
// scaffolding. Parallel template to crr_mxfp8_exact_8wave_fastpath.inc with
// the warp-grid topology rotated from production (WARPS_M=2 × WARPS_N=4)
// to (WARPS_M=4 × WARPS_N=2). Same 8 waves total per block, same BLK=256
// × BK=128 LDS tile geometry — only the per-warp partition changes.
//
// MOTIVATION (R34 cycle wrap, r34d_findings.md §3.2 option 2):
//   The R32 Dev B SB-pipelining ceiling at PIPE=3 was ascribed to the
//   accumulator footprint (128 VGPR/wave for cA/cB/cC/cD at RBM=64) +
//   A_col_reg=32 VGPR/wave on top. R34 Dev D's sub-RBM rewrite REFUTED
//   the "operand-tile shrink saves accumulator VGPR" hypothesis — the
//   accumulator inflates 1:1 with the operand shrink because per-warp
//   M-coverage (HB=128) stays the same while RBM halves, doubling the
//   stride count.
//
//   WARPS_M=4 is the OTHER lever in r34d_findings.md §3.2: instead of
//   shrinking the operand tile, change the warp-grid so each warp covers
//   HALF the M extent (HB=128 → 64) while keeping the same RBM-per-stride
//   count (1 stride per Bs half).
//
// FIRST-ORDER VGPR ANALYSIS (R35 Dev C, before measurement)
// ----------------------------------------------------------
//   Per-warp tile work — production:
//     M-coverage = BLK / WARPS_M     = 256 / 2 = 128 (= HB)
//     N-coverage = BLK / WARPS_N     = 256 / 4 = 64
//     Per-warp output elements       = 128 × 64 = 8192
//
//   Per-warp tile work — WARPS_M=4 / WARPS_N=2:
//     M-coverage = BLK / WARPS_M_W4  = 256 / 4 = 64
//     N-coverage = BLK / WARPS_N_W4  = 256 / 2 = 128
//     Per-warp output elements       = 64 × 128 = 8192   (UNCHANGED — same total work)
//
//   Operand register (per warp):
//     A_col_reg   = rt_fp8e4m3<128, 64> = 32 VGPR/wave (BK=128 × RBM=64)
//     A_col_reg_w4 = rt_fp8e4m3<128, 32> = 16 VGPR/wave (BK=128 × RBM_W4=32)  -16
//     B_col_reg   = rt_fp8e4m3<128, 32> = 16 VGPR/wave (BK=128 × RBN=32)
//     B_col_reg_w4 = rt_fp8e4m3<128, 64> = 32 VGPR/wave (BK=128 × RBN_W4=64)  +16
//     Net delta on operand registers = ZERO.
//     (The b0 + b1 pair widens the same +16 again, so total holding
//      both b0/b1 widens by +32 VGPR/wave net on B side.)
//
//   Accumulator register (per warp):
//     Production: 4 accs × rt_fl<64, 32, col_l, rt_16x16_s>
//                       = 4 × (RBM/16)·(RBN/16) sub-tiles
//                       = 4 × 4·2 = 32 sub-tiles × 4 fp32-VGPR/sub = 128 VGPR
//     WARPS_M=4 : 4 accs × rt_fl<32, 64, col_l, rt_16x16_s>
//                       = 4 × 2·4 = 32 sub-tiles × 4 fp32-VGPR/sub = 128 VGPR
//     Net delta on accumulators = ZERO.
//
//   First-order conclusion: WARPS_M=4 does NOT save accumulator VGPR (the
//   savings reasoning in the R35 task description was first-order
//   incorrect, mirroring the r34d sub-RBM finding). Per-warp tile work
//   is conserved when (WARPS_M, WARPS_N) reorder while WARPS_M·WARPS_N
//   stays constant.
//
//   POSSIBLE SECOND-ORDER WINS (need empirical confirmation):
//     1. Operand-A reuse: with B widened to 64 cols, a single A_col_reg
//        (16 VGPR) drives 2 N-tiles via opsel within the wider intrinsic.
//        Net: A LDS reads halve (per-iter), B reads stay same (1 b0_w4 +
//        1 b1_w4 vs 1 b0 + 1 b1).
//     2. Half the M-stride global loads (k-iter). Bs traffic doubles per
//        warp but total Bs bytes/block unchanged.
//     3. Different scheduler pressure pattern; may unblock SB pipelining
//        even if total VGPR is the same.
//
//   We scaffold + measure even though first-order says ZERO acc-VGPR
//   savings — empirical resource numbers are the deliverable (R34 Dev D
//   pattern: scaffold the lever to closure even when prediction is null).

#ifndef MXFP8_CRR_WARPS_M
#define MXFP8_CRR_WARPS_M 2
#endif
#if (MXFP8_CRR_WARPS_M != 2) && (MXFP8_CRR_WARPS_M != 4)
#error "MXFP8_CRR_WARPS_M must be 2 (production) or 4 (R35 Dev C scaffolding)"
#endif

#if defined(MXFP8_CRR_WARPS_M) && (MXFP8_CRR_WARPS_M == 4)

// When the user selects WARPS_M=4, force-disable the production fastpath
// in the same way MXFP8_CRR_RBM=32 does (the production fastpath has 8
// static_asserts requiring WARPS_M=2 / WARPS_N=4). The W4 kernel below
// supplies a parallel kernel symbol; dispatcher wiring is not changed
// (Stage A2-A4 deliverable: scaffold + resource numbers).
#undef  MXFP8_CRR_EXACT_8WAVE_FAST_ENABLE
#define MXFP8_CRR_EXACT_8WAVE_FAST_ENABLE 0

#ifndef MXFP8_CRR_WARPSM4_8WAVE_FAST_ENABLE
#define MXFP8_CRR_WARPSM4_8WAVE_FAST_ENABLE 1
#endif

#if MXFP8_CRR_WARPSM4_8WAVE_FAST_ENABLE

// Sanity: this WARPS_M=4 kernel only makes sense when the global
// parametrization matches BLK=256, BK=128. The global WARPS_M / WARPS_N
// constexprs are still 2 / 4 (we don't rebind them — the rest of the
// translation unit's helpers depend on them); the W4 kernel uses
// INTERNAL constants WARPS_M_W4=4 / WARPS_N_W4=2 for its own warp-grid.
static_assert(BLK == 256, "MXFP8 WARPS_M=4 CRR fast path requires BLK=256");
static_assert(BK == 128, "MXFP8 WARPS_M=4 CRR fast path requires BK=128");
static_assert(WARPS_M == 2, "MXFP8 WARPS_M=4 path coexists with global WARPS_M=2");
static_assert(WARPS_N == 4, "MXFP8 WARPS_M=4 path coexists with global WARPS_N=4");
static_assert(!CRR_ROW_SHARED_TRANSPOSE,
              "WARPS_M=4 CRR fast path only supports strict shared layout");
static_assert(!CRR_USE_V3_SWIZZLE,
              "WARPS_M=4 CRR fast path is wired for v2/v2a shared layouts");

// W4 constants. The INTERNAL warp-grid is 4 × 2 (still 8 waves total).
// Per-warp register tile geometry follows from the warp partition:
//   RBM_W4 = BLK / WARPS_M_W4 / 2 = 256 / 4 / 2 = 32   (was 64 at WARPS_M=2)
//   RBN_W4 = BLK / WARPS_N_W4 / 2 = 256 / 2 / 2 = 64   (was 32 at WARPS_N=4)
constexpr int WARPS_M_W4 = 4;
constexpr int WARPS_N_W4 = 2;
constexpr int RBM_W4 = BLK / WARPS_M_W4 / 2;
constexpr int RBN_W4 = BLK / WARPS_N_W4 / 2;
static_assert(RBM_W4 == 32, "WARPS_M=4 expects per-warp RBM=32");
static_assert(RBN_W4 == 64, "WARPS_M=4 expects per-warp RBN=64");
static_assert(WARPS_M_W4 * WARPS_N_W4 == _NUM_WARPS,
              "WARPS_M=4 / WARPS_N=2 must keep 8 waves total");

// Per-warp pack counts. RBM_W4 / 32 = 1; RBN_W4 / 32 = 2. Note the
// asymmetry vs production: A side now has ONE pack (was 2), B side now
// has TWO packs (was 1). This means scale-load balance shifts: A pack
// volume halves, B pack volume doubles, net traffic identical.
constexpr int crr_a_pack_count_w4 = RBM_W4 / 32;          // 1
constexpr int crr_b_pack_count_w4 = (RBN_W4 + 31) / 32;   // 2
static_assert(crr_a_pack_count_w4 == 1, "W4 expects A pack count 1");
static_assert(crr_b_pack_count_w4 == 2, "W4 expects B pack count 2");

// W4 register tile types. NEW types (suffix _w4) — they do NOT replace
// the global A_col_reg / B_col_reg defined in kernel_mxfp8_layouts.cpp.
using A_col_reg_w4 = rt_fp8e4m3<BK, RBM_W4, col_l, rt_128x16_s>;  // 128x32
using B_col_reg_w4 = rt_fp8e4m3<BK, RBN_W4, col_l, rt_128x16_s>;  // 128x64

// Row-layout aliases for the col→row reinterpret bridge (mirrors R34 Dev D
// Stage 2a in subrbm + production crr_mma_scaled_base bridge at
// kernel_mxfp8_layouts.cpp:1378-1388). The MMA scaled intrinsic
// (mma_ABt_base_scaled) requires both A and B as row-layout tiles.
//
//   A_row_reg_w4: tiles[RBM_W4/16=2][BK/128=1] = tiles[2][1]
//   A_col_reg_w4: tiles[BK/128=1][RBM_W4/16=2] = tiles[1][2]
//   B_row_reg_w4: tiles[RBN_W4/16=4][BK/128=1] = tiles[4][1]
//   B_col_reg_w4: tiles[BK/128=1][RBN_W4/16=4] = tiles[1][4]
using A_row_reg_w4 = rt_fp8e4m3<RBM_W4, BK, row_l, rt_16x128_s>;  // 32x128
using B_row_reg_w4 = rt_fp8e4m3<RBN_W4, BK, row_l, rt_16x128_s>;  // 64x128
static_assert(sizeof(A_row_reg_w4) == sizeof(A_col_reg_w4),
              "W4 A row/col aliases must have identical size");
static_assert(alignof(A_row_reg_w4) == alignof(A_col_reg_w4),
              "W4 A row/col aliases must have identical alignment");
static_assert(sizeof(B_row_reg_w4) == sizeof(B_col_reg_w4),
              "W4 B row/col aliases must have identical size");
static_assert(alignof(B_row_reg_w4) == alignof(B_col_reg_w4),
              "W4 B row/col aliases must have identical alignment");

// W4 mma_scaled_base: same per-quadrant 16x16x128 scaled MFMA intrinsic
// as production crr_mma_scaled_base, only the surrounding tile container
// shape changes. Reuses the col→row reinterpret bridge.
template<int opsel_a, int opsel_b>
__device__ __forceinline__ void crr_mma_scaled_base_w4(
    rt_fl<RBM_W4, RBN_W4, col_l, rt_16x16_s>& acc,
    const A_col_reg_w4& a,
    const B_col_reg_w4& b,
    int n,
    int m,
    const fp8e8m0_4 a_scale_pack,
    const fp8e8m0_4 b_scale_pack)
{
    const auto& a_row = reinterpret_cast<const A_row_reg_w4&>(a);
    const auto& b_row = reinterpret_cast<const B_row_reg_w4&>(b);
    mma_ABt_base_scaled<opsel_a, opsel_b>(
        acc.tiles[n][m],
        a_row.tiles[n][0],
        b_row.tiles[m][0],
        acc.tiles[n][m],
        a_scale_pack,
        b_scale_pack
    );
}

// W4 fixed-phase wrapper. Iteration shape:
//   RBM_W4=32 → acc_h = RBM_W4/16 = 2   (was 4 at RBM=64)
//   RBN_W4=64 → acc_w = RBN_W4/16 = 4   (was 2 at RBN=32)
// MMA chain count = acc_h × acc_w = 8 per fixed_phase call (same as
// production 4×2=8, but now indexed differently).
//
// Pack indexing:
//   A pack[0] covers all RBM_W4=32 rows
//   B pack[m/2] covers RBN_W4=32 cols (so m=0,1 → pack[0]; m=2,3 → pack[1])
template<int A_PACK_COUNT, int B_PACK_COUNT>
__device__ __forceinline__ void crr_mma_scaled_from_packs_fixed_phase_w4(
    rt_fl<RBM_W4, RBN_W4, col_l, rt_16x16_s>& acc,
    const A_col_reg_w4& a,
    const B_col_reg_w4& b,
    const fp8e8m0_4 (&a_scale_packs)[A_PACK_COUNT],
    const fp8e8m0_4 (&b_scale_packs)[B_PACK_COUNT])
{
    static_assert(RBM_W4 == 32 && RBN_W4 == 64,
                  "fixed_phase_w4 helper expects 32x64 tile");
    static_assert(A_PACK_COUNT >= 1 && B_PACK_COUNT >= 2,
                  "W4 fixed_phase needs A_PACK_COUNT>=1, B_PACK_COUNT>=2");
    // 8-MMA chain (same count as production).
    // Layout: opsel encodes (k_phase=0 fixed, n_or_m & 1) — see production
    // crr_exact_mma_one_scaled at crr_mxfp8_exact_8wave_fastpath.inc:144.
    crr_mma_scaled_base_w4<0, 0>(acc, a, b, 0, 0, a_scale_packs[0], b_scale_packs[0]);
    crr_mma_scaled_base_w4<0, 1>(acc, a, b, 0, 1, a_scale_packs[0], b_scale_packs[0]);
    crr_mma_scaled_base_w4<1, 0>(acc, a, b, 1, 0, a_scale_packs[0], b_scale_packs[0]);
    crr_mma_scaled_base_w4<1, 1>(acc, a, b, 1, 1, a_scale_packs[0], b_scale_packs[0]);
    // Second B-pack half (m=2,3 → b_scale_packs[1]).
    crr_mma_scaled_base_w4<0, 0>(acc, a, b, 0, 2, a_scale_packs[0], b_scale_packs[1]);
    crr_mma_scaled_base_w4<0, 1>(acc, a, b, 0, 3, a_scale_packs[0], b_scale_packs[1]);
    crr_mma_scaled_base_w4<1, 0>(acc, a, b, 1, 2, a_scale_packs[0], b_scale_packs[1]);
    crr_mma_scaled_base_w4<1, 1>(acc, a, b, 1, 3, a_scale_packs[0], b_scale_packs[1]);
}

// W4 K-phase variant (mirrors production crr_mma_scaled_phase).
template<int K_PHASE, int A_PACK_COUNT, int B_PACK_COUNT>
__device__ __forceinline__ void crr_mma_scaled_phase_w4(
    rt_fl<RBM_W4, RBN_W4, col_l, rt_16x16_s>& acc,
    const A_col_reg_w4& a,
    const B_col_reg_w4& b,
    const fp8e8m0_4 (&a_scale_packs)[A_PACK_COUNT],
    const fp8e8m0_4 (&b_scale_packs)[B_PACK_COUNT])
{
    static_assert(K_PHASE == 0 || K_PHASE == 1);
    static_assert(RBM_W4 == 32 && RBN_W4 == 64);
    static_assert(A_PACK_COUNT >= 1 && B_PACK_COUNT >= 2);
    constexpr int lo = K_PHASE * 2;
    constexpr int hi = K_PHASE * 2 + 1;
    crr_mma_scaled_base_w4<lo, lo>(acc, a, b, 0, 0, a_scale_packs[0], b_scale_packs[0]);
    crr_mma_scaled_base_w4<lo, hi>(acc, a, b, 0, 1, a_scale_packs[0], b_scale_packs[0]);
    crr_mma_scaled_base_w4<hi, lo>(acc, a, b, 1, 0, a_scale_packs[0], b_scale_packs[0]);
    crr_mma_scaled_base_w4<hi, hi>(acc, a, b, 1, 1, a_scale_packs[0], b_scale_packs[0]);
    crr_mma_scaled_base_w4<lo, lo>(acc, a, b, 0, 2, a_scale_packs[0], b_scale_packs[1]);
    crr_mma_scaled_base_w4<lo, hi>(acc, a, b, 0, 3, a_scale_packs[0], b_scale_packs[1]);
    crr_mma_scaled_base_w4<hi, lo>(acc, a, b, 1, 2, a_scale_packs[0], b_scale_packs[1]);
    crr_mma_scaled_base_w4<hi, hi>(acc, a, b, 1, 3, a_scale_packs[0], b_scale_packs[1]);
}

// W4 CRR exact 8-wave scaled kernel — Stage A3 skeleton (R35 Dev C).
//
// Translated from crr_exact_8wave_scaled_kernel with the WARPS_M=4 /
// WARPS_N=2 internal warp-grid:
//
//   Per-warp partition (INTERNAL — global WARPS_M/WARPS_N still 2/4):
//     wm_w4 = warpid() / WARPS_N_W4 = warpid() / 2 ∈ [0, 4)
//     wn_w4 = warpid() % WARPS_N_W4 = warpid() % 2 ∈ [0, 2)
//
//   Per-warp coverage (within the BLK=256 × BLK=256 block):
//     M = RBM_W4 × 2 halves = 32 × 2 = 64    (was HB=128 in production)
//     N = RBN_W4 × 2 halves = 64 × 2 = 128   (was 64 in production)
//
//   LDS tile layout unchanged from production:
//     As[2][2] of ST_v2a<HB=128, BK=128>  — As[*][0]=rows 0..127, As[*][1]=128..255
//     Bs[2][2] of ST_v2 <HB=128, BK=128>  — Bs[*][0]=cols 0..127, Bs[*][1]=128..255
//
//   Per-warp loads:
//     load_a_w4(a, As[*][half], wm_w4) → reads RBM_W4=32 rows starting at
//                                         wm_w4 * RBM_W4 within the half
//     load_b_w4(b0_w4, Bs[*][0], wn_w4) → reads RBN_W4=64 cols starting at
//                                          wn_w4 * RBN_W4 within Bs[*][0]
//     load_b_w4(b1_w4, Bs[*][1], wn_w4) → similarly within Bs[*][1]
//
//   Accumulators: 4 (cA, cB, cC, cD) of rt_fl<32, 64> each.
//     cA: top half × Bs[*][0]; cB: top half × Bs[*][1]
//     cC: bot half × Bs[*][0]; cD: bot half × Bs[*][1]
//
//   Epilogue store coordinates (per-rt-fl stores at macro tile granularity):
//     The output C tile is stored as 16x16 sub-tiles, with macro coords
//     based on (wm_w4, wn_w4) rather than global (wm, wn). The store
//     macro-row index needs new arithmetic:
//       out_macro_rows_per_block = BLK / RBM_W4 = 8
//       out_macro_cols_per_block = BLK / RBN_W4 = 4
//     Top-half: macro_row = br * 8 + wm_w4 * 2 + 0
//     Bot-half: macro_row = br * 8 + wm_w4 * 2 + 1
//          (NOTE: production scheme uses + WARPS_M offset, but with W4 the
//           halves are interleaved differently — see store calls below.)
//
//   Scale wire-in (Stage A3): STUBBED to unity (0x7F7F7F7F). Real V1/V2
//   scale fetch needs:
//     V1: per-warp scale row bases at offsets br*BLK + half*HB + wm_w4*RBM_W4
//         (and bc*BLK + half*HB + wn_w4*RBN_W4 + p*32 for the 2 B packs)
//         — straightforward but the slab indexing changes.
//     V2: slab geometry must be recomputed — production uses
//         slab_idx_a = br * WARPS_M + wm. With W4 it would be br * WARPS_M_W4
//         + wm_w4, which means the host preshuffle would need a re-tile.
//         The slab_bytes_a stays 4 × 32 × padded_k_blocks since
//         crr_a_pack_count_w4 × something = ... wait, with PC_A=1 the slab
//         shrinks to 32 × padded_k_blocks per warp, which is incompatible
//         with the production V2 host preshuffle. Either re-shuffle or
//         use SCALE_VERSION=1 for W4 numerics.
//
//   The Stage A3 skeleton uses unity-scale stubs (matches R34 Dev D
//   sub-RBM Stage 2b pattern) — this gates type-system validation and
//   gives a real VGPR/LDS resource report we can compare to production.
template<bool PRESHUFFLED_QUANT, int SCALE_VERSION = 2>
__global__ __launch_bounds__(_NUM_THREADS, GEMM_MIN_BLOCKS_PER_CU)
void crr_exact_8wave_scaled_w4_kernel(const layout_globals g)
{
    using ST_crr_a = ST_v2a;
    using ST_crr_b = ST_v2;

    constexpr int crr_a_pack_count_local = crr_a_pack_count_w4;  // 1
    constexpr int crr_b_pack_count_local = crr_b_pack_count_w4;  // 2

    const int k_iters = g.k / BK;
    const int blocks_per_col = g.n / BLK;

    // LDS tiles unchanged from production (As/Bs are block-wide tiles,
    // independent of warp-grid topology).
    __shared__ ST_crr_a As[2][2];
    __shared__ ST_crr_b Bs[2][2];

    // W4 operand tiles.
    A_col_reg_w4 a_w4;
    B_col_reg_w4 b0_w4, b1_w4;

    // Accumulators (4 vectors, rt_fl<32, 64>).
    rt_fl<RBM_W4, RBN_W4, col_l, rt_16x16_s> cA, cB, cC, cD;
    zero(cA); zero(cB); zero(cC); zero(cD);

    const int bid = blockIdx.x;
    const int br = bid / blocks_per_col;
    const int bc = bid % blocks_per_col;

    // INTERNAL warp partition (WARPS_M_W4 × WARPS_N_W4).
    const int wm_w4 = warpid() / WARPS_N_W4;  // 0..3
    const int wn_w4 = warpid() % WARPS_N_W4;  // 0..1

    const int lane_nonk = kittens::laneid() % 16;
    const int lane_kblk = kittens::laneid() / 16;
    const uint32_t lane_scale_byte_offset =
        (static_cast<uint32_t>(lane_kblk) << 6) |
        (static_cast<uint32_t>(lane_nonk) << 2);

    // Scale base offsets (W4 partition).
    auto crr_scale_a_base_w4 = [&](int half) {
        // Each warp covers RBM_W4=32 rows starting at wm_w4 * RBM_W4
        // within the half (half ∈ {0,1} selects the 128-row sub-block).
        return br * BLK + half * (BLK / 2) + wm_w4 * RBM_W4;
    };
    auto crr_scale_b_base_w4 = [&](int half) {
        return bc * BLK + half * (BLK / 2) + wn_w4 * RBN_W4;
    };

    // STUBBED scale packs (Stage A3 — type validation only).
    // crr_a_pack_count_w4 = 1, crr_b_pack_count_w4 = 2.
    fp8e8m0_4 a0_scale_packs[crr_a_pack_count_local];
    fp8e8m0_4 a1_scale_packs[crr_a_pack_count_local];
    fp8e8m0_4 b0_scale_packs[crr_b_pack_count_local];
    fp8e8m0_4 b1_scale_packs[crr_b_pack_count_local];
    #pragma unroll
    for (int p = 0; p < crr_a_pack_count_local; ++p) {
        a0_scale_packs[p] = std::bit_cast<fp8e8m0_4>(uint32_t{0x7F7F7F7F});
        a1_scale_packs[p] = std::bit_cast<fp8e8m0_4>(uint32_t{0x7F7F7F7F});
    }
    #pragma unroll
    for (int p = 0; p < crr_b_pack_count_local; ++p) {
        b0_scale_packs[p] = std::bit_cast<fp8e8m0_4>(uint32_t{0x7F7F7F7F});
        b1_scale_packs[p] = std::bit_cast<fp8e8m0_4>(uint32_t{0x7F7F7F7F});
    }

    // Prefill swizzled offsets (unchanged — block-wide tile loaders).
    constexpr int bptA = ST_crr_a::underlying_subtile_bytes_per_thread;
    constexpr int bpmA = bptA * _NUM_THREADS;
    constexpr int mptA = ST_crr_a::rows * ST_crr_a::cols * sizeof(fp8e4m3) / bpmA;
    uint32_t soA[mptA];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);
    constexpr int bptB = ST_crr_b::underlying_subtile_bytes_per_thread;
    constexpr int bpmB = bptB * _NUM_THREADS;
    constexpr int mptB = ST_crr_b::rows * ST_crr_b::cols * sizeof(fp8e4m3) / bpmB;
    uint32_t soB[mptB];
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

    auto a_co = [&](int s, int k) -> coord<ST_crr_a> { return {0, 0, k, s}; };
    auto b_co = [&](int s, int k) -> coord<ST_crr_b> { return {0, 0, k, s}; };
    auto global_load_a = [&](ST_crr_a& tile, int s, int k) {
        G::load(tile, g.a, a_co(s, k), soA);
    };
    auto global_load_b = [&](ST_crr_b& tile, int s, int k) {
        G::load(tile, g.b, b_co(s, k), soB);
    };
    // W4 per-warp loaders.
    auto load_a_w4 = [&](A_col_reg_w4& dst, ST_crr_a& tile) {
        // wm_w4 ∈ [0, WARPS_M_W4=4); each warp reads RBM_W4=32 rows
        // starting at wm_w4 * RBM_W4 within the HB=128 tile half.
        load_col_from_v2a_st(dst, tile, wm_w4 * RBM_W4);
    };
    auto load_b_w4 = [&](B_col_reg_w4& dst, ST_crr_b& tile) {
        // wn_w4 ∈ [0, WARPS_N_W4=2); each warp reads RBN_W4=64 cols
        // starting at wn_w4 * RBN_W4 within the HB=128 tile half.
        load_col_from_v2_st(dst, tile, wn_w4 * RBN_W4);
    };

    // Single-buffered LDS schedule (mirrors MXFP8_CRR_LDS_SINGLE_BUFFER=1
    // mode of the production kernel — simplest correct schedule for an
    // A3 skeleton). Stage A6+ can revisit pipelining.
    global_load_b(Bs[0][0], bc * 2, 0);
    global_load_a(As[0][0], br * 2, 0);
    global_load_b(Bs[0][1], bc * 2 + 1, 0);
    global_load_a(As[0][1], br * 2 + 1, 0);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    for (int k = 0; k < k_iters; k++) {
        // Suppress unused-warning for stubs; full V1/V2 wire-in deferred.
        (void)crr_scale_a_base_w4; (void)crr_scale_b_base_w4;
        (void)lane_scale_byte_offset;

        // Top half (As[*][0] = rows 0..127, but warp's 32-row slice
        // selected by wm_w4 * RBM_W4 within).
        load_a_w4(a_w4, As[0][0]);
        load_b_w4(b0_w4, Bs[0][0]);
        load_b_w4(b1_w4, Bs[0][1]);
        asm volatile("s_waitcnt lgkmcnt(0)");

        crr_mma_scaled_from_packs_fixed_phase_w4(
            cA, a_w4, b0_w4, a0_scale_packs, b0_scale_packs);
        crr_mma_scaled_from_packs_fixed_phase_w4(
            cB, a_w4, b1_w4, a0_scale_packs, b1_scale_packs);

        // Bottom half (As[*][1] = rows 128..255).
        load_a_w4(a_w4, As[0][1]);
        asm volatile("s_waitcnt lgkmcnt(0)");

        crr_mma_scaled_from_packs_fixed_phase_w4(
            cC, a_w4, b0_w4, a1_scale_packs, b0_scale_packs);
        crr_mma_scaled_from_packs_fixed_phase_w4(
            cD, a_w4, b1_w4, a1_scale_packs, b1_scale_packs);

        __builtin_amdgcn_s_barrier();
        if (k + 1 < k_iters) {
            global_load_b(Bs[0][0], bc * 2, k + 1);
            global_load_a(As[0][0], br * 2, k + 1);
            global_load_b(Bs[0][1], bc * 2 + 1, k + 1);
            global_load_a(As[0][1], br * 2 + 1, k + 1);
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_s_barrier();
        }
    }

    if (wm_w4 == 0) {
        __builtin_amdgcn_s_barrier();
    }

    if constexpr (PRESHUFFLED_QUANT) {
        mul(cA, cA, g.scale);
        mul(cB, cB, g.scale);
        mul(cC, cC, g.scale);
        mul(cD, cD, g.scale);
    }

    // Epilogue store. Macro-tile coords use the W4 grid:
    //   out_macro_rows_per_block = BLK / RBM_W4 = 8
    //   out_macro_cols_per_block = BLK / RBN_W4 = 4
    //   Per-warp covers 2 macro-rows (top + bot half) × 2 macro-cols (Bs[0]+Bs[1]).
    //
    // Production scheme has wm ∈ [0,2), so macro_row_top = br * (WARPS_M*2) + wm,
    // macro_row_bot = ... + WARPS_M. With W4 each warp owns:
    //   macro_row_top = br * 8 + wm_w4
    //   macro_row_bot = br * 8 + WARPS_M_W4 + wm_w4
    //   macro_col_left  = bc * 4 + wn_w4
    //   macro_col_right = bc * 4 + WARPS_N_W4 + wn_w4
    //
    // Each rt_fl<32,64> store covers 2 (M-axis) × 4 (N-axis) sub-tiles, so
    // a single store call writes to the macro_row × macro_col coordinate.
    const int macro_row_top = br * (BLK / RBM_W4) + wm_w4;
    const int macro_row_bot = br * (BLK / RBM_W4) + WARPS_M_W4 + wm_w4;
    const int macro_col_left  = bc * (BLK / RBN_W4) + wn_w4;
    const int macro_col_right = bc * (BLK / RBN_W4) + WARPS_N_W4 + wn_w4;
    store(g.c, cA, {0, 0, macro_row_top, macro_col_left});
    store(g.c, cB, {0, 0, macro_row_top, macro_col_right});
    store(g.c, cC, {0, 0, macro_row_bot, macro_col_left});
    store(g.c, cD, {0, 0, macro_row_bot, macro_col_right});
}

__host__ inline bool crr_can_use_exact_8wave_scaled_w4(const layout_globals& g)
{
    return g.m == M_DIM && g.n == N_DIM && g.k == K_DIM;
}

template<bool PRESHUFFLED_QUANT>
__host__ inline void dispatch_crr_exact_8wave_scaled_v2_w4(const layout_globals& g)
{
    const dim3 grid((g.m / BLK) * (g.n / BLK));
    crr_exact_8wave_scaled_w4_kernel<PRESHUFFLED_QUANT, 2>
        <<<grid, dim3(_NUM_THREADS), 0, g.stream>>>(g);
}

// Force-instantiation probe (mirrors R33 Dev D / R34 Dev D pattern):
// when MXFP8_CRR_WARPSM4_PROBE=1 we emit a tiny host stub that takes
// the address of the dispatcher, which forces the kernel template to
// be instantiated and the symbol to appear in the resource-usage report.
#ifndef MXFP8_CRR_WARPSM4_PROBE
#define MXFP8_CRR_WARPSM4_PROBE 0
#endif
#if MXFP8_CRR_WARPSM4_PROBE
__host__ inline void* crr_warpsm4_probe_addr_handle()
{
    return reinterpret_cast<void*>(&dispatch_crr_exact_8wave_scaled_v2_w4<true>);
}
#endif

#endif  // MXFP8_CRR_WARPSM4_8WAVE_FAST_ENABLE

#endif  // MXFP8_CRR_WARPS_M == 4
