#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLOCK_SIZE       = 256;
constexpr int HALF_BLOCK_SIZE  = BLOCK_SIZE / 2;
constexpr int K_STEP           = 64;
constexpr int WARPS_M          = 2;
constexpr int WARPS_N          = 4;
constexpr int REG_BLOCK_M      = BLOCK_SIZE / WARPS_M;
constexpr int REG_BLOCK_N      = BLOCK_SIZE / WARPS_N;
constexpr int HALF_REG_BLOCK_M = REG_BLOCK_M / 2;
constexpr int HALF_REG_BLOCK_N = REG_BLOCK_N / 2;

// Tail-kernel tile (mirror FP8 dense kernel_fp8_layouts.cpp:34-35). The
// scalar fp32 fallback runs one (row, col) per thread; 16x16 was chosen
// in FP8 to keep launch overhead bounded for the small tail region while
// still hitting full occupancy. Used when M/N/K are not a multiple of
// the main-kernel block size.
constexpr int TAIL_BLOCK_M     = 16;
constexpr int TAIL_BLOCK_N     = 16;
// Two-tile schedule of the BF16 main kernel (`for tile = 0; tile < num_tiles - 2; tile += 2`)
// requires `ki = K / K_STEP` to be EVEN, i.e. K must be a multiple of
// `2 * K_STEP = 128`. Anything else has to fall through to the tail.
constexpr int K_TWO_TILE       = 2 * K_STEP;

#define NUM_WARPS (WARPS_M * WARPS_N)
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)


using _gl = gl<bf16, -1, -1, -1, -1>;
using G = kittens::group<NUM_WARPS>;

enum class Layout { RCR, RRR, CRR };

// P21 Dev D — port the FP8 RCR m0-broadcast hoist (P19 commit 49647b11) to
// BF16 RCR + RRR. The baseline kernel emits one `s_mov_b32 m0, sX` plus one
// `s_mov_b32 sX, s_next` rotation per `buffer_load_dwordx4 ... offen lds`.
// The intrinsic path lets LLVM rotate m0 via a single working SGPR which
// inserts an extra scalar move per load. Inline asm pre-computes a fully
// unrolled SGPR ramp of LDS byte addresses in the prologue and issues
//   s_mov_b32 m0, <per-pass SGPR>
//   buffer_load_dwordx4 <vphantom>, <SRD>, <SOFF> offen lds
// which removes the SGPR rotate and leaves only the unavoidable m0 write.
//
// CRR is left on the stock G::load path: its DTL pattern is structurally
// different (b64_tr_b16 dominant, 8x more buffer_loads) and any change there
// must be measured separately.
//
// Default 0 — flip to 1 only on a measured wall-clock win with SNR > 5 dB
// AND zero-byte CRR delta.
#ifndef BF16_HOIST_M0
#define BF16_HOIST_M0 0
#endif

// P23 Session 2 Dev C — RCR Route 1 padded-b128 wiring (PREP).
//
// When `RCR_PADDED_B128_MODE` is 1, the RCR branch swaps ST_A/ST_B from
//   st_bf<128, 64, st_16x32_s>     (subtile 16x32, 0-byte padding,
//                                    underlying_subtile_stride_bytes=1024)
// to
//   st_bf<128, 64, st_64x32_padded_b128_s>
//                                   (subtile 64x32, 32-byte padding,
//                                    underlying_subtile_stride_bytes=4128).
//
// Goals: break the 128-B LDS-bank alias for stride-128 ds_read_b128 on
// gfx950's 32-bank x 4-B banks (matches BL's RCR design). Within-subtile
// swizzle is identity; the padding lives BETWEEN subtiles.
//
// LDS budget (per-block, both A and B allocated [2][2]):
//   PADDED=0:  4 * sizeof(st_bf<128,64,st_16x32_s>)         * 2 (A+B)
//             = 4 * 16,384 * 2 = 131,072 B   (~128 KiB, MAX=160,000)
//   PADDED=1:  4 * sizeof(st_bf<128,64,st_64x32_padded_b128>) * 2 (A+B)
//             = 4 * 16,512 * 2 = 132,096 B   (~129 KiB, MAX=160,000)
// Per-tile size derived from st.cuh:83
//   = underlying_subtiles_per_col * underlying_subtiles_per_row
//     * (underlying_subtile_elements + subtile_padding/sizeof(T)) * sizeof(T)
//   PADDED=1: 2 * 2 * (64*32 + 32/2) * 2 = 16,512 B  (matches design doc).
//
// The RCR_PADDED_B128_MODE=1 build is EXPECTED to compile-error at the
// `load_a_subtile`/`load_b_subtile` calls until Dev A or Dev B lands the
// b128 dispatch branch in `include/ops/warp/memory/tile/shared_to_register.cuh`.
// Default 0 — flag=0 path must be byte-identical to the pre-Step-6 baseline.
#ifndef RCR_PADDED_B128_MODE
#define RCR_PADDED_B128_MODE 0
#endif

// Convenience: exposes the RCR ST_A/ST_B swap to all RCR-only callsites
// (typedefs, allocations, prefill, subtile lambdas). Orthogonal to BF16_HOIST_M0.
#if RCR_PADDED_B128_MODE
using rcr_padded_st_shape = kittens::st_64x32_padded_b128_s;
#endif

#if BF16_HOIST_M0
namespace bf16_dev_d {
using as3_uint32_ptr = __attribute__((address_space(3))) unsigned int*;

template<int N_THREADS,
         ducks::st::all ST,
         ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>>
__device__ __forceinline__ void load_hoist(
    ST& dst, const GL& src, const COORD& idx,
    const uint32_t* __restrict__ swizzled_offsets,
    i32x4 SRD, const void* base_ptr, const uint32_t lds_base)
{
    using T = typename ST::dtype;
    static_assert(sizeof(T) == 2, "bf16 hoist expects 2-byte dtype");

    constexpr int bytes_per_thread = 16;
    constexpr int bytes_per_memcpy = bytes_per_thread * N_THREADS;
    constexpr int memcpy_per_tile  =
        (ST::rows * ST::cols * sizeof(T)) / bytes_per_memcpy;
    static_assert(bytes_per_memcpy % 16 == 0, "LDS bump must be 16-aligned");

    // Wave-uniform SOFF (pulled into SGPR; mirrors stock to_sgpr_u32 helper).
    coord<> unit_coord = idx.template unit_coord<2, 3>();
    T* __restrict__ gptr = (T*)&src[unit_coord];
    uint32_t SOFF = static_cast<uint32_t>(
        reinterpret_cast<const char*>(gptr) -
        reinterpret_cast<const char*>(base_ptr));
    SOFF = __builtin_amdgcn_readfirstlane(SOFF);
    asm volatile("" : "+s"(SOFF));

    // Wave-uniform LDS tile base (matches the per-warp `lds_base` arg's
    // tile origin; we recompute warp_offset from the same delta as the stock
    // path so the two layouts are byte-identical when BF16_HOIST_M0=0).
    uint32_t lds_tile_base3 = static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&dst.data[0]));
    lds_tile_base3 = __builtin_amdgcn_readfirstlane(lds_tile_base3);
    asm volatile("" : "+s"(lds_tile_base3));
    const uint32_t warp_offset = lds_base - lds_tile_base3;

    // Hoist per-pass scalar LDS-byte ramp into SGPRs in the prologue.
    uint32_t lds_addrs[memcpy_per_tile > 0 ? memcpy_per_tile : 1];
    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; ++i) {
        const uint32_t linear_offset =
            warp_offset + static_cast<uint32_t>(i) * bytes_per_memcpy;
        const uint32_t subtile_id_lds = linear_offset / ST::underlying_subtile_bytes;
        uint32_t lds_byte = lds_tile_base3 + linear_offset +
                            subtile_id_lds * ST::subtile_padding;
        lds_byte = __builtin_amdgcn_readfirstlane(lds_byte);
        asm volatile("" : "+s"(lds_byte));
        lds_addrs[i] = lds_byte;
    }

    // Inline-asm DTL — set m0 from the SGPR-hoisted per-pass offset and issue
    // buffer_load_dwordx4 ... offen lds. The intrinsic
    // `__builtin_amdgcn_raw_buffer_load_lds` lets the scheduler reschedule
    // the m0 write back through a vector intermediate; inline asm forecloses
    // that and keeps the per-iter cluster down to 1 scalar move + 1 DTL.
    // Operand binding mirrors P19 Dev A's working FP8 8w pattern:
    //   %0 = s "lds_off" (SGPR)  %1 = v "goff" (per-lane VGPR offset)
    //   %2 = s "SRD" (4-SGPR buffer resource)  %3 = s "SOFF" (scalar offset)
    // v0 satisfies the asm constraint; `buffer_load_dwordx4 ... lds` does not
    // actually write a VGPR.
    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; ++i) {
        const uint32_t lds_off = lds_addrs[i];
        const uint32_t goff    = swizzled_offsets[i];
        asm volatile(
            "s_mov_b32 m0, %0\n\t"
            "buffer_load_dwordx4 %1, %2, %3 offen lds\n\t"
            :
            : "s"(lds_off), "v"(goff), "s"(SRD), "s"(SOFF)
            : "memory");
    }
}
} // namespace bf16_dev_d
#endif // BF16_HOIST_M0

struct layout_globals {
    _gl a, b, c;
    hipStream_t stream;
    int m, n, k, ki, bpr, bpc, group_m, num_xcds;
    // Aligned-region dimensions consumed by the main kernel:
    //   fast_m = (m / BLOCK_SIZE) * BLOCK_SIZE
    //   fast_n = (n / BLOCK_SIZE) * BLOCK_SIZE
    //   fast_k = (k / K_TWO_TILE) * K_TWO_TILE
    // Cells outside [0,fast_m) × [0,fast_n) plus the K-tail in
    // [fast_k, k) are handled by `gemm_tail_kernel` (scalar fp32).
    int fast_m, fast_n, fast_k;
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

// Scalar bf16 helpers used by `gemm_tail_kernel` (mirror the FP8 versions
// in analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp:168-174). One element
// per call; intentionally NOT vectorised because the tail region is small
// (<=2*BLOCK_SIZE rows/cols + K_TWO_TILE-1 K-tail) and the launch is rare.
__device__ __forceinline__ float load_bf16_scalar(const _gl& src, int row, int col) {
    return base_types::convertor<float, bf16>::convert(src[coord<>(row, col)]);
}

__device__ __forceinline__ void store_bf16_scalar(const _gl& dst, int row, int col, float value) {
    dst[coord<>(row, col)] = base_types::convertor<bf16, float>::convert(value);
}

// Scalar bf16 load for a 3D-grouped tensor (B in grouped GEMM is laid out
// as `[1, G, *, *]`, where the 2nd axis indexes the group). Used by
// `grouped_tail_kernel` to read B at `(group_idx, row, col)`.
__device__ __forceinline__ float load_bf16_scalar_grp(const _gl& src, int g_idx, int row, int col) {
    return base_types::convertor<float, bf16>::convert(src[coord<>{0, g_idx, row, col}]);
}

// Packed 4 × bf16 = 8 bytes for vectorised tail-kernel K-loop. The HIP
// compiler emits a single `global_load_dwordx2` for a load through this
// type when the source pointer is 8-byte aligned, replacing 4 separate
// scalar bf16 loads (4× fewer VMEM transactions). Used by the RCR fast
// path inside `gemm_tail_kernel` / `grouped_tail_kernel` where both
// operands are stride-1 in K.
struct alignas(8) bf16x4 {
    bf16_2 lo, hi;
};

// =============================================================================
// N-mask C-store helper (Phase 4 enabler):
//
// Lets the main GEMM kernel cover the N-tail directly. Caller sets
// ``g.bpc = ceil_div(n, BLOCK_SIZE)``; the last tile-column may straddle the
// real N boundary, so its store must drop OOB columns.
//
// Three-way fast path:
//   * n0 >= n_limit               -> entire tile OOB, skip (no work).
//   * n1 <= n_limit               -> tile fully in-bounds, dispatch the
//                                    original ``store(g_c, src, ...)`` (zero
//                                    overhead vs. pre-Phase-4 baseline).
//   * partial (n0 < n_limit < n1) -> lane-level skip on per-column OOB.
//
// Garbage data the main loop reads from B in OOB columns is harmless: the
// MFMA still writes a result to C_accum's OOB columns, but those columns are
// dropped here and never reach global memory.
// =============================================================================
using C_rt_accum_t =
    rt_fl<HALF_REG_BLOCK_M, HALF_REG_BLOCK_N, col_l, rt_16x16_s>;

template<ducks::gl::all GL>
__device__ __forceinline__ void store_c_tile_n_masked(
    const GL& g_c, const C_rt_accum_t& src,
    int r_tile, int c_tile, int n_limit) {
    using T = base_types::packing<typename C_rt_accum_t::dtype>::unpacked_type;
    using U = typename GL::dtype;
    constexpr int packing = base_types::packing<typename C_rt_accum_t::dtype>::num();
    static_assert(std::is_same_v<U, bf16>, "C is bf16 global");

    const int n0 = c_tile * C_rt_accum_t::cols;
    const int n1 = n0 + C_rt_accum_t::cols;
    if (n0 >= n_limit) return;
    if (n1 <= n_limit) {
        store(g_c, src, {0, 0, r_tile, c_tile});
        return;
    }

    constexpr int axis = 2;
    U* dst_ptr = (U*)&g_c[(coord<C_rt_accum_t>{0, 0, r_tile, c_tile}
                            .template unit_coord<axis, 3>())];
    const int row_stride = g_c.template stride<axis>();
    const int laneid = kittens::laneid();
    const int row_offset = src.base_tile_stride * (laneid / src.base_tile_cols);
    const int col_offset = laneid % src.base_tile_cols;

    #pragma unroll
    for (int i = 0; i < src.height; i++) {
        #pragma unroll
        for (int j = 0; j < src.width; j++) {
            const int col = j * src.base_tile_cols + col_offset;
            if (n0 + col >= n_limit) continue;
            #pragma unroll
            for (int k = 0; k < src.base_tile_num_strides; k++) {
                int row = i * src.base_tile_rows + row_offset +
                          k * src.base_tile_elements_per_stride_group;
                #pragma unroll
                for (int l = 0; l < src.base_tile_stride / packing; l++) {
                    int idx = l + k * src.base_tile_stride / packing;
                    dst_ptr[(row + l * 2) * row_stride + col] =
                        base_types::convertor<U, T>::convert(
                            src.tiles[i][j].data[idx].x);
                    dst_ptr[(row + l * 2 + 1) * row_stride + col] =
                        base_types::convertor<U, T>::convert(
                            src.tiles[i][j].data[idx].y);
                }
            }
        }
    }
}

// =============================================================================
// store_c_tile_mn_masked_grouped — two-axis-masked C store for the
// persistent variable-K (CRR / dB) grouped kernel.
//
// Output layout is 3D-grouped ``[G, m_kernel, n_kernel]``; per-group
// ``m_kernel`` (= N_fwd) and ``n_kernel`` (= K_fwd) can both be partially
// misaligned (gpt_oss-Down has both = 2880, neither a 256-multiple).
// Caller sets ``bpr = ceil_div(m_kernel, BLOCK_SIZE)`` and ``bpc =
// ceil_div(n_kernel, BLOCK_SIZE)``; the last tile in either axis may
// straddle the real boundary.
//
// 4-way fast path:
//   * fully OOB (m0 >= m_limit OR n0 >= n_limit): no-op.
//   * fully in-bounds (m1 <= m_limit AND n1 <= n_limit): forward to
//     the original ``store(...)``. Aligned shapes pay zero overhead.
//   * partial in N only: ~mirror ``store_c_tile_n_masked`` but with
//     the depth axis (group_idx) propagated.
//   * partial in M (with or without N partial): per-row + per-col mask.
//
// MMA reads OOB cols/rows from A and B (full-tensor SRD bound covers
// the contiguous tensors so no fault), produces garbage for OOB
// (m, n) cells, then we drop those cells here. In-bounds cells still
// reduce over the correct K range, so their values are exact.
// =============================================================================
template<ducks::gl::all GL>
__device__ __forceinline__ void store_c_tile_mn_masked_grouped(
    const GL& g_c, const C_rt_accum_t& src,
    int group_idx, int r_tile, int c_tile,
    int m_limit, int n_limit) {
    using T = base_types::packing<typename C_rt_accum_t::dtype>::unpacked_type;
    using U = typename GL::dtype;
    constexpr int packing = base_types::packing<typename C_rt_accum_t::dtype>::num();
    static_assert(std::is_same_v<U, bf16>, "C is bf16 global");

    const int m0 = r_tile * C_rt_accum_t::rows;
    const int m1 = m0 + C_rt_accum_t::rows;
    const int n0 = c_tile * C_rt_accum_t::cols;
    const int n1 = n0 + C_rt_accum_t::cols;

    if (m0 >= m_limit || n0 >= n_limit) return;
    if (m1 <= m_limit && n1 <= n_limit) {
        store(g_c, src, {0, group_idx, r_tile, c_tile});
        return;
    }

    constexpr int axis = 2;
    U* dst_ptr = (U*)&g_c[(coord<C_rt_accum_t>{0, group_idx, r_tile, c_tile}
                            .template unit_coord<axis, 3>())];
    const int row_stride = g_c.template stride<axis>();
    const int laneid = kittens::laneid();
    const int row_offset = src.base_tile_stride * (laneid / src.base_tile_cols);
    const int col_offset = laneid % src.base_tile_cols;

    #pragma unroll
    for (int i = 0; i < src.height; i++) {
        #pragma unroll
        for (int j = 0; j < src.width; j++) {
            const int col = j * src.base_tile_cols + col_offset;
            if (n0 + col >= n_limit) continue;
            #pragma unroll
            for (int k = 0; k < src.base_tile_num_strides; k++) {
                int row = i * src.base_tile_rows + row_offset +
                          k * src.base_tile_elements_per_stride_group;
                #pragma unroll
                for (int l = 0; l < src.base_tile_stride / packing; l++) {
                    int idx = l + k * src.base_tile_stride / packing;
                    int row_a = row + l * 2;
                    int row_b = row + l * 2 + 1;
                    if (m0 + row_a < m_limit) {
                        dst_ptr[row_a * row_stride + col] =
                            base_types::convertor<U, T>::convert(
                                src.tiles[i][j].data[idx].x);
                    }
                    if (m0 + row_b < m_limit) {
                        dst_ptr[row_b * row_stride + col] =
                            base_types::convertor<U, T>::convert(
                                src.tiles[i][j].data[idx].y);
                    }
                }
            }
        }
    }
}

// =============================================================================
// device_gemm_tile_body — shared GEMM main-loop body (Phase 2 refactor).
//
// The dense `gemm_kernel<L, KI_HINT>` and the persistent grouped
// `grouped_kernel<L, KI_HINT>` previously contained two byte-identical
// copies of: subtile-load lambdas + DO_MMA macro + prologue (4 G::load) +
// main_loop_iter (~85 line lambda over a two-tile schedule) + epilog 1 +
// epilog 2. This single `__forceinline__` device function is the shared
// implementation; both callers see the identical instruction schedule
// after inlining (constants like `m_subtile_A=0, group_idx=0` are folded
// in the dense path).
//
// Caller responsibilities:
//   * Allocate ST_A As[2][2] and ST_B Bs[2][2] in shared memory.
//   * Compute SRD bases (a_srsrc_base, b_srsrc_base) and element bases
//     (a_base, b_base) covering one tile (dense) or the full A / B
//     tensors (grouped).
//   * Compute the 8 LDS double-buffer offsets (a_lds_{00,01,10,11},
//     b_lds_{00,01,10,11}) for the wave's per-warp slots.
//   * Prefill swizzled_offsets_A / swizzled_offsets_B once.
//   * `zero(C_accum[i][j])` before invoking this helper.
//   * Pass `m_subtile_A = 0` and `group_idx = 0` from the dense kernel;
//     pass `m_start_g / HALF_BLOCK_SIZE` and the persistent group index
//     from the grouped kernel.
//   * On return, store C_accum to the output tensor (plus optional row
//     shift for grouped) and, for grouped, drain in-flight ops before
//     the next persistent iteration.
//
// `num_tiles_dyn` is only consulted when KI_HINT == 0 (dynamic K). For
// KI_HINT > 0 the loop bound is constant-folded.
// =============================================================================
template<Layout L, int KI_HINT,
         typename ST_A_T, typename ST_B_T,
         typename A_reg_t, typename B_reg_t>
__device__ __forceinline__ void device_gemm_tile_body(
    const _gl& a_gl, const _gl& b_gl,
    int m_subtile_A, int group_idx, int k_offset_tiles,
    ST_A_T (&As)[2][2], ST_B_T (&Bs)[2][2],
    const uint32_t* swizzled_offsets_A,
    const uint32_t* swizzled_offsets_B,
    i32x4 a_srsrc_base, i32x4 b_srsrc_base,
    const bf16* a_base, const bf16* b_base,
    uint32_t a_lds_00, uint32_t a_lds_01,
    uint32_t a_lds_10, uint32_t a_lds_11,
    uint32_t b_lds_00, uint32_t b_lds_01,
    uint32_t b_lds_10, uint32_t b_lds_11,
    int row, int col,
    int warp_row, int warp_col,
    int num_tiles_dyn,
    rt_fl<HALF_REG_BLOCK_M, HALF_REG_BLOCK_N, col_l, rt_16x16_s> (&C_accum)[2][2])
{
    A_reg_t A_tile;
    B_reg_t B_tile_0, B_tile_1;
    int tic = 0, toc = 1;

    // Coord helpers — `m_subtile_A` shifts A's M-axis (= 0 for dense,
    // = m_start_g/HALF_BLOCK_SIZE for grouped); `group_idx` indexes B's
    // depth axis (= 0 for dense, = persistent group index for grouped);
    // `k_offset_tiles` shifts the K-axis in K_STEP units (= 0 for dense
    // and forward grouped where K is fixed; = m_start_g/K_STEP for
    // variable-K dB grouped where the K-reduction dimension is the
    // per-group M_g segment of a [M_total, *] input).
    auto a_coord = [&](int spatial, int k) {
        if constexpr (L == Layout::CRR)
            return coord<ST_A_T>{0, 0, k_offset_tiles + k, m_subtile_A + spatial};
        else
            return coord<ST_A_T>{0, 0, m_subtile_A + spatial, k_offset_tiles + k};
    };
    auto b_coord = [&](int spatial, int k) {
        if constexpr (L == Layout::RCR)
            return coord<ST_B_T>{0, group_idx, spatial, k_offset_tiles + k};
        else
            return coord<ST_B_T>{0, group_idx, k_offset_tiles + k, spatial};
    };

    auto load_a_subtile = [&](A_reg_t& dst, auto& smem_tile, int warp_idx) {
        if constexpr (L == Layout::CRR) {
            auto sub = subtile_inplace<K_STEP, HALF_REG_BLOCK_M>(smem_tile, {0, warp_idx});
            load(dst, sub);
        } else {
            auto sub = subtile_inplace<HALF_REG_BLOCK_M, K_STEP>(smem_tile, {warp_idx, 0});
            load(dst, sub);
        }
    };
    auto load_b_subtile = [&](B_reg_t& dst, auto& smem_tile, int warp_idx) {
        if constexpr (L == Layout::RCR) {
            auto sub = subtile_inplace<HALF_REG_BLOCK_N, K_STEP>(smem_tile, {warp_idx, 0});
            load(dst, sub);
        } else {
            auto sub = subtile_inplace<K_STEP, HALF_REG_BLOCK_N>(smem_tile, {0, warp_idx});
            load(dst, sub);
        }
    };

    // MMA dispatch (same as dense baseline). For CRR we use mma_AtB_base
    // directly to skip the register-transpose dance; the macro form keeps
    // the outer #pragma unroll over the main-loop lambda transparent.
    #define DO_MMA(D, A, B, C) \
        do { \
            if constexpr (L == Layout::RCR) { mma_ABt(D, A, B, C); } \
            else if constexpr (L == Layout::RRR) { mma_AB(D, A, B, C); } \
            else { \
                constexpr int NH = std::remove_reference_t<decltype(D)>::height; \
                constexpr int NW = std::remove_reference_t<decltype(D)>::width; \
                constexpr int KH = std::remove_reference_t<decltype(A)>::height; \
                _Pragma("unroll") \
                for (int _n = 0; _n < NH; _n++) { \
                    _Pragma("unroll") \
                    for (int _m = 0; _m < NW; _m++) { \
                        mma_AtB_base(D.tiles[_n][_m], A.tiles[0][_n], B.tiles[0][_m], C.tiles[_n][_m]); \
                        _Pragma("unroll") \
                        for (int _k = 1; _k < KH; _k++) { \
                            mma_AtB_base(D.tiles[_n][_m], A.tiles[_k][_n], B.tiles[_k][_m], D.tiles[_n][_m]); \
                        } \
                    } \
                } \
            } \
        } while(0)

    /********** Prologue: load first two K-tiles **********/
    G::load(Bs[tic][0], b_gl, b_coord(col*2, 0), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_00);
    G::load(As[tic][0], a_gl, a_coord(row*2, 0), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_00);
    G::load(Bs[tic][1], b_gl, b_coord(col*2+1, 0), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_01);
    G::load(As[tic][1], a_gl, a_coord(row*2+1, 0), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_01);

    if (warp_row == 1) { __builtin_amdgcn_s_barrier(); }
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    G::load(Bs[toc][0], b_gl, b_coord(col*2, 1), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_10);
    G::load(As[toc][0], a_gl, a_coord(row*2, 1), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_10);
    G::load(Bs[toc][1], b_gl, b_coord(col*2+1, 1), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_11);

    asm volatile("s_waitcnt vmcnt(6)");
    __builtin_amdgcn_s_barrier();

    /********** Main loop **********/
    auto main_loop_iter = [&](int tile) {
        load_b_subtile(B_tile_0, Bs[0][0], warp_col);
        load_a_subtile(A_tile, As[0][0], warp_row);
        G::load(As[1][1], a_gl, a_coord(row*2+1, tile+1), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_11);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load_b_subtile(B_tile_1, Bs[0][1], warp_col);
        G::load(Bs[0][0], b_gl, b_coord(col*2, tile+2), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_00);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a_subtile(A_tile, As[0][1], warp_row);
        G::load(As[0][0], a_gl, a_coord(row*2, tile+2), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_00);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load_b_subtile(B_tile_0, Bs[1][0], warp_col);
        G::load(Bs[0][1], b_gl, b_coord(col*2+1, tile+2), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_01);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a_subtile(A_tile, As[1][0], warp_row);
        G::load(As[0][1], a_gl, a_coord(row*2+1, tile+2), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_01);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load_b_subtile(B_tile_1, Bs[1][1], warp_col);
        G::load(Bs[1][0], b_gl, b_coord(col*2, tile+3), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_10);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a_subtile(A_tile, As[1][1], warp_row);
        G::load(As[1][0], a_gl, a_coord(row*2, tile+3), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_10);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        G::load(Bs[1][1], b_gl, b_coord(col*2+1, tile+3), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_11);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    };

    // Schedule selection (matches the original baseline):
    //   RCR/RRR + KI_HINT > 0:  full #pragma unroll over compile-time KI.
    //   CRR     + KI_HINT > 0:  #pragma unroll 2 (some KIs see 7-26 SGPR
    //                            spills under unroll-2, but barrier-hiding
    //                            still wins).
    //   KI_HINT == 0 dynamic:    #pragma unroll 2 (all layouts).
    if constexpr (KI_HINT > 0) {
        constexpr int num_tiles = KI_HINT;
        if constexpr (L == Layout::CRR) {
            #pragma unroll 2
            for (int tile = 0; tile < num_tiles - 2; tile += 2) main_loop_iter(tile);
        } else {
            #pragma unroll
            for (int tile = 0; tile < num_tiles - 2; tile += 2) main_loop_iter(tile);
        }
    } else {
        const int num_tiles = num_tiles_dyn;
        #pragma unroll 2
        for (int tile = 0; tile < num_tiles - 2; tile += 2) main_loop_iter(tile);
    }

    /********** Epilog 1: second-to-last K-tile pair **********/
    {
        const int tile = (KI_HINT > 0) ? (KI_HINT - 2) : (num_tiles_dyn - 2);
        load_b_subtile(B_tile_0, Bs[tic][0], warp_col);
        load_a_subtile(A_tile, As[tic][0], warp_row);
        G::load(As[toc][1], a_gl, a_coord(row*2+1, tile+1), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_11);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_b_subtile(B_tile_1, Bs[tic][1], warp_col);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a_subtile(A_tile, As[tic][1], warp_row);
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        DO_MMA(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        tic ^= 1; toc ^= 1;
    }

    /********** Epilog 2: last K-tile **********/
    {
        load_b_subtile(B_tile_0, Bs[tic][0], warp_col);
        load_a_subtile(A_tile, As[tic][0], warp_row);
        asm volatile("s_waitcnt vmcnt(2)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_b_subtile(B_tile_1, Bs[tic][1], warp_col);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a_subtile(A_tile, As[tic][1], warp_row);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        DO_MMA(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    #undef DO_MMA
}

// KI_HINT > 0: compile-time num_tiles (K / K_STEP) -> full #pragma unroll
// KI_HINT == 0: dynamic num_tiles from g.ki, #pragma unroll 2
template<Layout L, int KI_HINT>
__global__ __launch_bounds__(NUM_THREADS, 2)
void gemm_kernel(const layout_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // Shared memory tile types: "normal" = <128,64,st_16x32_s>, "transposed" = <64,128,st_32x16_s>
    // The swizzle must match: row_l registers use rt_16x32 -> st_16x32_s;
    //                         col_l registers use rt_32x16 -> st_32x16_s.
    //
    // P23 S2 Dev C: when RCR_PADDED_B128_MODE=1, the RCR-specific shape is
    // `st_64x32_padded_b128_s` (Route 1). RRR/CRR are unchanged. The shape
    // swap stays inside `if constexpr (L == Layout::RCR)` so RRR/CRR codegen
    // is byte-identical regardless of the flag.
#if RCR_PADDED_B128_MODE
    using ST_A_RCR = st_bf<HALF_BLOCK_SIZE, K_STEP, rcr_padded_st_shape>;
    using ST_B_RCR = st_bf<HALF_BLOCK_SIZE, K_STEP, rcr_padded_st_shape>;
#else
    using ST_A_RCR = st_bf<HALF_BLOCK_SIZE, K_STEP, st_16x32_s>;
    using ST_B_RCR = st_bf<HALF_BLOCK_SIZE, K_STEP, st_16x32_s>;
#endif

    using ST_A = std::conditional_t<L == Layout::CRR,
        st_bf<K_STEP, HALF_BLOCK_SIZE, st_32x16_s>,
        std::conditional_t<L == Layout::RCR,
            ST_A_RCR,
            st_bf<HALF_BLOCK_SIZE, K_STEP, st_16x32_s>>>;
    using ST_B = std::conditional_t<L == Layout::RCR,
        ST_B_RCR,
        st_bf<K_STEP, HALF_BLOCK_SIZE, st_32x16_s>>;

    ST_A (&As)[2][2] = al.allocate<ST_A, 2, 2>();
    ST_B (&Bs)[2][2] = al.allocate<ST_B, 2, 2>();

    // Register tile types
    using A_reg_t = std::conditional_t<L == Layout::CRR,
        rt_bf<K_STEP, HALF_REG_BLOCK_M, col_l, rt_32x16_s>,
        rt_bf<HALF_REG_BLOCK_M, K_STEP, row_l, rt_16x32_s>>;
    using B_reg_t = std::conditional_t<L == Layout::RCR,
        rt_bf<HALF_REG_BLOCK_N, K_STEP, row_l, rt_16x32_s>,
        rt_bf<K_STEP, HALF_REG_BLOCK_N, col_l, rt_32x16_s>>;

    // C accumulators stay in this outer scope so they survive the helper
    // call and are still live during the store epilog below.
    rt_fl<HALF_REG_BLOCK_M, HALF_REG_BLOCK_N, col_l, rt_16x16_s> C_accum[2][2];
    zero(C_accum[0][0]); zero(C_accum[0][1]);
    zero(C_accum[1][0]); zero(C_accum[1][1]);

    const int total_tiles = g.bpr * g.bpc;
    int wgid = blockIdx.x;

    // Block mapping with XCD swizzle. Dual strategy: for tall-N problems
    // (bpc > bpr), group-by-N so WGs in a super-block share pid_n and
    // cycle pid_m — this optimizes B-reuse (B is larger than A on tall-N).
    // For tall-M / square, use group-by-M so WGs share pid_m and cycle
    // pid_n — this optimizes A-reuse. The user-specified group_m becomes
    // WGM on tall-M path or WGN on tall-N path.
    //
    // The split encodes g.group_m meaning as a generic "super-block length
    // along the narrower dimension". Threshold bpc > bpr picks tall-N vs
    // non-tall-N. The two paths produce different (pid_m, pid_n) mappings
    // but both preserve the XCD-swizzled traversal order.
    const int NUM_WGS = total_tiles;
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, g.num_xcds, 64);
    const int num_pid_m = g.bpr;
    const int num_pid_n = g.bpc;
    const int WG  = g.group_m;
    int pid_m, pid_n;
    if (g.bpc > g.bpr) {
        // Tall-N: group-by-N. Super-block = all_M × WGN
        const int WGN = WG;
        const int num_wgid_in_group = num_pid_m * WGN;
        int group_id = wgid / num_wgid_in_group;
        int first_pid_n = group_id * WGN;
        int group_size_n = min(num_pid_n - first_pid_n, WGN);
        if (group_size_n <= 0) return;
        pid_n = first_pid_n + ((wgid % num_wgid_in_group) % group_size_n);
        pid_m = (wgid % num_wgid_in_group) / group_size_n;
    } else {
        // Tall-M / square: group-by-M. Super-block = WGM × all_N
        const int WGM = WG;
        const int num_wgid_in_group = WGM * num_pid_n;
        int group_id = wgid / num_wgid_in_group;
        int first_pid_m = group_id * WGM;
        int group_size_m = min(num_pid_m - first_pid_m, WGM);
        if (group_size_m <= 0) return;
        pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
        pid_n = (wgid % num_wgid_in_group) / group_size_m;
    }
    if (pid_m >= g.bpr || pid_n >= g.bpc) return;
    int row = pid_m;
    int col = pid_n;

    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;

    // K-specialization: compile-time vs dynamic.
    // For the KI_HINT>0 path the helper sees constexpr KI_HINT so the
    // main loop can unroll fully. For the KI_HINT==0 path we forward
    // g.ki via num_tiles_dyn and the helper uses #pragma unroll 2.

    /********** SRD setup **********/
    const bf16* a_base = (bf16*)&g.a[{0, 0, 0, 0}];
    const bf16* b_base = (bf16*)&g.b[{0, 0, 0, 0}];
    const int a_row_stride = g.a.template stride<2>() * sizeof(bf16);
    const int b_row_stride = g.b.template stride<2>() * sizeof(bf16);
    // For "normal" layout (M×K or N×K): num_rows = M or N
    // For "transposed" layout (K×M or K×N): num_rows = K
    const int a_num_rows = (L == Layout::CRR) ? g.k : g.m;
    const int b_num_rows = (L == Layout::RCR) ? g.n : g.k;
    i32x4 a_srsrc_base = make_srsrc(a_base, a_num_rows * a_row_stride, a_row_stride);
    i32x4 b_srsrc_base = make_srsrc(b_base, b_num_rows * b_row_stride, b_row_stride);

    const int wid = warpid() % NUM_WARPS;
    constexpr int elem_per_warp = (16 / sizeof(bf16)) * kittens::WARP_THREADS;
    constexpr uint32_t A_TILE_LDS = sizeof(ST_A);
    constexpr uint32_t B_TILE_LDS = sizeof(ST_B);
    uint32_t a_lds = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&As[0][0].data[0]) + wid * elem_per_warp * sizeof(bf16)));
    uint32_t b_lds = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&Bs[0][0].data[0]) + wid * elem_per_warp * sizeof(bf16)));
    const uint32_t a_lds_00 = a_lds;
    const uint32_t a_lds_01 = a_lds + A_TILE_LDS;
    const uint32_t a_lds_10 = a_lds + 2 * A_TILE_LDS;
    const uint32_t a_lds_11 = a_lds + 3 * A_TILE_LDS;
    const uint32_t b_lds_00 = b_lds;
    const uint32_t b_lds_01 = b_lds + B_TILE_LDS;
    const uint32_t b_lds_10 = b_lds + 2 * B_TILE_LDS;
    const uint32_t b_lds_11 = b_lds + 3 * B_TILE_LDS;

    using T = typename st_bf<BLOCK_SIZE, K_STEP, st_32x16_s>::dtype;
    constexpr int bytes_per_thread = st_32x16_s::template bytes_per_thread<T>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile = BLOCK_SIZE * K_STEP * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile/2];
    uint32_t swizzled_offsets_B[memcpy_per_tile/2];
    G::prefill_swizzled_offsets(As[0][0], g.a, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[0][0], g.b, swizzled_offsets_B);

    // Phase 2: shared device function (above) runs the prologue + main loop
    // + epilog 1/2. Dense passes m_subtile_A=0 and group_idx=0; the
    // compiler folds those constants and emits the same code as the
    // pre-refactor kernel.
    device_gemm_tile_body<L, KI_HINT, ST_A, ST_B, A_reg_t, B_reg_t>(
        g.a, g.b,
        /*m_subtile_A=*/0, /*group_idx=*/0, /*k_offset_tiles=*/0,
        As, Bs,
        swizzled_offsets_A, swizzled_offsets_B,
        a_srsrc_base, b_srsrc_base,
        a_base, b_base,
        a_lds_00, a_lds_01, a_lds_10, a_lds_11,
        b_lds_00, b_lds_01, b_lds_10, b_lds_11,
        row, col, warp_row, warp_col,
        g.ki,
        C_accum);

    if (warp_row == 0) { __builtin_amdgcn_s_barrier(); }

    store_c_tile_n_masked(g.c, C_accum[0][0],
        (row * 2) * WARPS_M + warp_row,
        col * 2 * WARPS_N + warp_col,
        g.n);
    store_c_tile_n_masked(g.c, C_accum[0][1],
        (row * 2) * WARPS_M + warp_row,
        col * 2 * WARPS_N + WARPS_N + warp_col,
        g.n);
    store_c_tile_n_masked(g.c, C_accum[1][0],
        (row * 2) * WARPS_M + WARPS_M + warp_row,
        col * 2 * WARPS_N + warp_col,
        g.n);
    store_c_tile_n_masked(g.c, C_accum[1][1],
        (row * 2) * WARPS_M + WARPS_M + warp_row,
        col * 2 * WARPS_N + WARPS_N + warp_col,
        g.n);
}

// ---- Explicit instantiations ----

// Scalar fp32 tail kernel — one thread per (row, col) of g.c.
//
// Mirror of the FP8 dense tail in analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp:1515-1547,
// minus the FP8 scale epilog. Two distinct cases based on (row, col):
//
//   1) `interior_mn`  (row < fast_m AND col < fast_n) AND `needs_k_tail`
//      (fast_k < g.k):  the main kernel already wrote the [0..fast_k) inner
//      product to g.c[row, col]. We add the K-tail [fast_k..k) on top.
//
//   2) Boundary cell (row >= fast_m OR col >= fast_n) OR main kernel did
//      not run at all (`fast_m`/`fast_n`/`fast_k` == 0): compute the full
//      K reduction from scratch and overwrite g.c[row, col].
//
// `fast_covers_cell && !needs_k_tail` early-returns: those cells are the
// fully-aligned interior already produced by the main kernel.
template<Layout L>
__global__ void gemm_tail_kernel(const layout_globals g) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= g.m || col >= g.n) {
        return;
    }

    // Phase 4: main kernel coverage depends on K alignment (see
    // dispatch_gemm bpc computation):
    //   * K aligned (fast_k == k): main covers [0, fast_m) × [0, n) via
    //     ceil_div bpc + column-masked C store.
    //   * K not aligned: main covers [0, fast_m) × [0, fast_n) (OG behavior);
    //     cols [fast_n, n) need full K reduction in tail kernel.
    const bool main_covers_n   = (g.fast_k == g.k);
    const bool main_ran_for_cell =
        row < g.fast_m && g.fast_m > 0 && g.fast_k > 0 &&
        (main_covers_n || col < g.fast_n);
    const bool fast_covers_cell = main_ran_for_cell;
    const bool needs_k_tail     = g.fast_k < g.k;
    if (fast_covers_cell && !needs_k_tail) {
        return;
    }

    const int k0 = fast_covers_cell ? g.fast_k : 0;
    float acc = 0.0f;

    if constexpr (L == Layout::RCR) {
        // Vec4 fast path (8-byte loads = 4 bf16) for RCR's stride-1 K
        // axis on both A and B. Mirrors the grouped-tail optimisation;
        // see ``grouped_tail_kernel`` for the rationale + bench (~4×
        // fewer VMEM transactions per K element on the K-tail / N-tail
        // paths). Falls back to scalar when k0 or g.k isn't multiple
        // of 4. Dense rarely runs the tail (host-aligned LLM shapes are
        // 4096 / 8192 multiples), so this is mostly a code-symmetry win
        // — keeps dense + grouped tail kernel logic identical.
        const bf16* a_row = &g.a[coord<>(row, 0)];
        const bf16* b_row = &g.b[coord<>(col, 0)];
        int kk = k0;
        if ((g.k % 4 == 0) && ((k0 & 3) == 0)) {
            const bf16x4* a_v4 = reinterpret_cast<const bf16x4*>(a_row);
            const bf16x4* b_v4 = reinterpret_cast<const bf16x4*>(b_row);
            const int j_start = k0 >> 2;
            const int j_end   = g.k >> 2;
            #pragma unroll 4
            for (int j = j_start; j < j_end; ++j) {
                bf16x4 a4 = a_v4[j];
                bf16x4 b4 = b_v4[j];
                acc += float(a4.lo.x) * float(b4.lo.x)
                     + float(a4.lo.y) * float(b4.lo.y)
                     + float(a4.hi.x) * float(b4.hi.x)
                     + float(a4.hi.y) * float(b4.hi.y);
            }
            kk = j_end << 2;
        }
        for (; kk < g.k; ++kk) {
            acc += load_bf16_scalar(g.a, row, kk) * load_bf16_scalar(g.b, col, kk);
        }
    } else if constexpr (L == Layout::RRR) {
        for (int kk = k0; kk < g.k; ++kk) {
            acc += load_bf16_scalar(g.a, row, kk) * load_bf16_scalar(g.b, kk, col);
        }
    } else { // CRR
        for (int kk = k0; kk < g.k; ++kk) {
            acc += load_bf16_scalar(g.a, kk, row) * load_bf16_scalar(g.b, kk, col);
        }
    }

    if (fast_covers_cell && needs_k_tail) {
        store_bf16_scalar(g.c, row, col,
                          load_bf16_scalar(g.c, row, col) + acc);
    } else {
        store_bf16_scalar(g.c, row, col, acc);
    }
}

template __global__ void gemm_tail_kernel<Layout::RCR>(const layout_globals);
template __global__ void gemm_tail_kernel<Layout::RRR>(const layout_globals);
template __global__ void gemm_tail_kernel<Layout::CRR>(const layout_globals);

// KI_HINT = 0 dynamic fallback
template __global__ void gemm_kernel<Layout::RCR, 0>(const layout_globals);
template __global__ void gemm_kernel<Layout::RRR, 0>(const layout_globals);
template __global__ void gemm_kernel<Layout::CRR, 0>(const layout_globals);
// KI_HINT > 0: specialized for common LLM K values (K / K_STEP).
// K values: 3584, 4096, 8192, 11008, 14336, 16384, 18944, 28672, 29568, 53248
// -> ki = 56, 64, 128, 172, 224, 256, 296, 448, 462, 832
#define INSTANTIATE_K(KI) \
    template __global__ void gemm_kernel<Layout::RCR, KI>(const layout_globals); \
    template __global__ void gemm_kernel<Layout::RRR, KI>(const layout_globals); \
    template __global__ void gemm_kernel<Layout::CRR, KI>(const layout_globals)
INSTANTIATE_K(56);
INSTANTIATE_K(64);
INSTANTIATE_K(128);
INSTANTIATE_K(172);
INSTANTIATE_K(224);
INSTANTIATE_K(256);
INSTANTIATE_K(296);
INSTANTIATE_K(448);
INSTANTIATE_K(462);
INSTANTIATE_K(832);
#undef INSTANTIATE_K

template<Layout L, int KI>
static inline void launch_one(layout_globals& g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    // Set shared-mem attribute once per (L, KI) function pointer: idempotent,
    // avoids per-launch HIP runtime overhead.
    static bool attr_set = false;
    if (!attr_set) {
        hipFuncSetAttribute((void*)gemm_kernel<L, KI>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
        attr_set = true;
    }
    int total_blocks = g.bpr * g.bpc;
    gemm_kernel<L, KI><<<dim3(total_blocks), g.block(), mem_size, g.stream>>>(g);
}

template<Layout L>
void dispatch_gemm(layout_globals g) {
    g.m = static_cast<int>(g.c.rows());
    g.n = static_cast<int>(g.c.cols());
    if constexpr (L == Layout::CRR) g.k = static_cast<int>(g.a.rows());
    else g.k = static_cast<int>(g.a.cols());

    // Native non-aligned-shape support (mirror FP8 dense dispatch in
    // analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp:1900-1953):
    //   * `fast_*` is the largest aligned sub-region the BLOCK_SIZE-tiled
    //     two-tile-K main kernel can cover. Misaligned M / N / K is no
    //     longer pre-padded by the host — the kernel handles it natively
    //     via a scalar fp32 tail kernel.
    //   * K alignment is `K_TWO_TILE = 2 * K_STEP = 128` (NOT just K_STEP)
    //     because the main loop is two-tile (`tile += 2`) and silently
    //     reads OOB on odd `ki` (root cause of the SNR=16.55 dB regression
    //     the task spec calls out for K=2880).
    //   * If the main kernel can't run at all (fast region empty), we still
    //     drop into the tail kernel below to compute the full output.
    g.fast_m = (g.m / BLOCK_SIZE) * BLOCK_SIZE;
    g.fast_n = (g.n / BLOCK_SIZE) * BLOCK_SIZE;
    g.fast_k = (g.k / K_TWO_TILE) * K_TWO_TILE;
    g.bpr = g.fast_m / BLOCK_SIZE;
    // Phase 4: main kernel covers the entire N range via column-masked C
    // store. Last tile-column may straddle [fast_n, n) boundary.
    //
    // Restriction: ceil_div coverage of N is only enabled when K is fully
    // aligned (fast_k == k). When BOTH N and K are misaligned, the partial
    // col-tile interacts with the K-tail correction in a way that triggers
    // a memory fault on certain shape combinations (e.g., N=K=2880 with
    // M>=1024 in KI_HINT=0 dynamic kernel). In that case we fall back to
    // bpc = fast_n / BLOCK_SIZE and the scalar tail kernel covers cols
    // [fast_n, n).
    g.bpc = (g.fast_k == g.k)
        ? kittens::ceil_div(g.n, BLOCK_SIZE)
        : (g.fast_n / BLOCK_SIZE);
    g.ki  = g.fast_k / K_STEP;

    if (g.bpr > 0 && g.bpc > 0 && g.ki >= 2) {
        switch (g.ki) {
            case 56:  launch_one<L, 56> (g); break;
            case 64:  launch_one<L, 64> (g); break;
            case 128: launch_one<L, 128>(g); break;
            case 172: launch_one<L, 172>(g); break;
            case 224: launch_one<L, 224>(g); break;
            case 256: launch_one<L, 256>(g); break;
            case 296: launch_one<L, 296>(g); break;
            case 448: launch_one<L, 448>(g); break;
            case 462: launch_one<L, 462>(g); break;
            case 832: launch_one<L, 832>(g); break;
            default:  launch_one<L, 0>  (g); break;
        }
    } else {
        // Fast region empty (M < BLOCK_SIZE or N < BLOCK_SIZE or K < K_TWO_TILE)
        // — let the tail kernel compute the entire output. Reset fast_* so
        // tail logic treats every cell as "kernel never ran here".
        g.fast_m = 0;
        g.fast_n = 0;
        g.fast_k = 0;
        g.ki     = 0;
    }

    // Tail kernel runs when:
    //   * M-tail (row >= fast_m): full K reduction.
    //   * K-tail (fast_k < k): K-tail correction (and N-tail full reduction
    //     when K is misaligned and main bpc is fast_n / BLOCK_SIZE).
    //   * N-tail with K aligned: handled inline by main kernel masked store.
    const bool main_covers_n = (g.fast_k == g.k);
    const bool need_tail =
        (g.fast_m != g.m) || (g.fast_k != g.k) ||
        (!main_covers_n && g.fast_n != g.n);
    if (need_tail) {
        dim3 tail_block(TAIL_BLOCK_N, TAIL_BLOCK_M);
        dim3 tail_grid(
            kittens::ceil_div(g.n, TAIL_BLOCK_N),
            kittens::ceil_div(g.m, TAIL_BLOCK_M)
        );
        gemm_tail_kernel<L><<<tail_grid, tail_block, 0, g.stream>>>(g);
    }
}

static void gemm_dispatch(pybind11::object a, pybind11::object b, pybind11::object c,
                          int gm, int num_xcds, const char* layout_name) {
    auto c_gl = py::from_object<_gl>::make(c);
    layout_globals g{py::from_object<_gl>::make(a), py::from_object<_gl>::make(b),
                     c_gl, {},
                     /* m, n, k, ki, bpr, bpc */ 0, 0, 0, 0, 0, 0,
                     /* group_m, num_xcds */ gm, num_xcds,
                     /* fast_m, fast_n, fast_k */ 0, 0, 0};

    if (layout_name[0] == 'r' && layout_name[1] == 'c') dispatch_gemm<Layout::RCR>(g);
    else if (layout_name[0] == 'r' && layout_name[1] == 'r') dispatch_gemm<Layout::RRR>(g);
    else dispatch_gemm<Layout::CRR>(g);
}

static void rcr(pybind11::object a, pybind11::object b, pybind11::object c, int gm, int num_xcds) {
    gemm_dispatch(a, b, c, gm, num_xcds, "rcr");
}
static void rrr(pybind11::object a, pybind11::object b, pybind11::object c, int gm, int num_xcds) {
    gemm_dispatch(a, b, c, gm, num_xcds, "rrr");
}
static void crr(pybind11::object a, pybind11::object b, pybind11::object c, int gm, int num_xcds) {
    gemm_dispatch(a, b, c, gm, num_xcds, "crr");
}

// =============================================================================
// Persistent grouped GEMM (CPU-sync-free).
//
// Mirror of the Triton ``_grouped_bf16_persistent_gemm_kernel`` design (see
// ``primus_turbo/triton/grouped_gemm/grouped_gemm_kernel.py`` for the full
// reference). One launch with ``grid_x = NUM_CUS = 256`` programs covers
// ALL groups × ALL tiles. Each program:
//
//   1. Pulls G+1 int64 offsets from a device tensor and computes total tile
//      count via O(G) scan (no host sync).
//   2. Iterates ``global_tile = pid; gt < total; gt += NUM_CUS`` so the same
//      block streams through many (group, tile) pairs without re-launch.
//   3. Per iteration: O(G) scan to recover (group_idx, m_start_g, M_g),
//      then runs the existing dense GEMM tile body with coord shifts:
//         * A/C  : spatial += m_start_g / SUBTILE_ROWS
//         * B    : depth   = group_idx (B is treated as ``[G, N, K]`` for
//                 RCR or ``[G, K, N]`` for RRR/CRR)
//
// Inner body is byte-identical to the dense kernel's prologue + main_loop +
// epilog; the only differences are coord shifts and per-iteration C_accum
// reset. SRD bounds are widened to span the full A / B tensors instead of
// one-tile worth so the same SRD is valid across iterations.
// =============================================================================

struct grouped_layout_globals {
    _gl a;                       // [M_total, K]
    _gl b;                       // [G, N, K] (RCR) or [G, K, N] (RRR/CRR)
    _gl c;                       // [M_total, N]
    const int64_t* group_offs;   // [G+1] int64 prefix-sum on device
    hipStream_t stream;
    int G;                       // number of groups
    int n;                       // N
    int k;                       // K
    int ki;                      // fast_k / K_STEP
    int bpc;                     // fast_n / BLOCK_SIZE
    int group_m;                 // tile-scheduling super-block factor
    int num_xcds;                // XCD swizzle factor
    int M_total;                 // sum of group sizes (= a.shape[0])
    // Aligned-region dimensions consumed by the main grouped kernel
    // (mirror dense kernel layout_globals).
    //   fast_n = (n / BLOCK_SIZE) * BLOCK_SIZE
    //   fast_k = (k / K_TWO_TILE) * K_TWO_TILE
    // Per-group M tail (M_g % BLOCK_SIZE != 0) is detected on-device by
    // `grouped_tail_kernel` reading `group_offs`. Cells outside
    // [0, fast_m_g) × [0, fast_n) (per group) plus the K-tail in
    // [fast_k, k) are handled by the tail kernel (scalar fp32).
    int fast_n, fast_k;
    // ``m_per_group`` is set by the host caller from the uniform-M check
    // (``_uniform_group_m`` on the python side). If the groups are uniform
    // *and* ``m_per_group`` is a multiple of ``TAIL_BLOCK_M``, then every
    // tail-kernel block of (TAIL_BLOCK_M, TAIL_BLOCK_N) lies inside a single
    // group, so the LDS-staged K-tail kernel ``grouped_ktail_kernel_lds``
    // can use one ``group_idx`` for the whole block (faster B-strip load).
    // 0 means "non-uniform / not aligned" → LDS path is unsafe and the
    // scalar tail covers every cell as before. The scalar tail kernel
    // ALSO consults this flag to decide whether to skip case-2 (interior
    // K-tail correction) — when LDS has handled it, we mustn't double-add.
    int m_per_group;
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

// Scalar fp32 tail kernel for grouped GEMM — one (row, col) per thread.
//
// Mirror of dense `gemm_tail_kernel<L>` but with B 3D-grouped (per-thread
// O(G) LDS scan to recover `group_idx` for B indexing).
//
// Caller contract (Phase 3): each group's M_g is a BLOCK_SIZE multiple,
// so the main kernel covers full rows of every group. This tail kernel
// only handles N-tail (col >= fast_n) and the K-tail correction
// (fast_k < k for interior cells). Per-group M-tail is a Phase ≥ 4
// concern that requires reworking the main kernel's tile addressing
// (m_subtile_A is in HALF_BLOCK_SIZE=128 units, so non-128-aligned
// m_start_g of a subsequent group truncates and silently corrupts that
// group's tiles — the fix is non-trivial and beyond Phase 3 scope).
//
// Three cases per cell:
//   * col < fast_n  AND fast_k == k  → main covers fully → early-return.
//   * col < fast_n  AND fast_k <  k  → main wrote partial; add K-tail.
//   * col >= fast_n                  → main did not run; full-K reduction.
template<Layout L>
__global__ void grouped_tail_kernel(const grouped_layout_globals g) {
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        #pragma unroll 1
        for (int gi = 0; gi <= g.G; ++gi) {
            s_offs[gi] = static_cast<int>(g.group_offs[gi]);
        }
    }
    __syncthreads();

    // Round-11: RCR's ``dispatch_grouped`` always launches the main
    // kernel with ``bpc = ceil_div(g.n, BLOCK_SIZE)``; the per-group
    // bounded B SRD + column-masked C store cover the entire [0, g.n)
    // column range natively, regardless of K alignment. In that mode
    // the tail kernel only runs for K-tail (which applies to ALL cols
    // [0, n) including the partial last col-tile) or per-group M-tail.
    // RRR/CRR layouts can't activate ceil_div N coverage (B has N on
    // the column axis, where SRD-clamp doesn't trigger for OOB N), so
    // their N-tail still flows through the per-cell full-K reduction.
    constexpr bool layout_supports_main_n = (L == Layout::RCR);
    const bool main_covers_n =
        layout_supports_main_n && (g.fast_k > 0);
    const bool needs_k_tail = g.fast_k < g.k;
    const bool needs_n_tail = !main_covers_n && (g.fast_n < g.n);
    if (!needs_k_tail && !needs_n_tail) return;

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= g.M_total || col >= g.n) return;

    // Locate the group that contains `row` (needed to index B).
    int group_idx = 0;
    #pragma unroll 1
    for (int gi = 0; gi < g.G; ++gi) {
        if (row < s_offs[gi + 1]) { group_idx = gi; break; }
    }

    // ``interior_n``: column lies inside the region the main kernel covered.
    //   * main_covers_n (ceil_div bpc + masked store): main wrote [0, g.n).
    //   * otherwise (bpc = fast_n / BLOCK_SIZE): main wrote [0, fast_n).
    const bool interior_n       = main_covers_n
                                      ? (col < g.n)
                                      : (col < g.fast_n);
    const bool fast_covers_cell = interior_n && g.fast_k > 0;
    if (fast_covers_cell && !needs_k_tail) return;
    // When the LDS-staged K-tail kernel runs (round-9 ``feat(bf16-grouped):
    // LDS-staged K-tail correction``), it has already added the
    // [fast_k, k) K-tail to interior cells. Skip case-2 here to avoid
    // double-counting. The ``m_per_group >= TBM && % TBM == 0`` test is
    // identical to the dispatcher's launch condition for the LDS kernel.
    //
    // Round-6 cross-group safety: the LDS K-tail kernel internally
    // early-exits when its block straddles a group boundary
    // (row_block_base + TBM > s_offs[group_idx + 1]) — those cells are
    // NOT written by LDS, so scalar tail must NOT skip them. We
    // replicate the same per-block group-containment check here so the
    // skip predicate matches what LDS actually executed (rather than
    // what the host hint told us was safe).
    if constexpr (L == Layout::RCR) {
        const bool lds_k_tail_safe = (g.m_per_group >= TAIL_BLOCK_M) &&
                                     ((g.m_per_group % TAIL_BLOCK_M) == 0);
        const bool lds_k_rem_match = ((g.k - g.fast_k) == 64);
        const int row_block_base = (row / TAIL_BLOCK_M) * TAIL_BLOCK_M;
        const bool block_in_group =
            (row_block_base + TAIL_BLOCK_M <= s_offs[group_idx + 1]);
        if (fast_covers_cell && needs_k_tail && lds_k_tail_safe &&
            lds_k_rem_match && block_in_group) {
            return;  // LDS K-tail kernel already wrote the corrected value.
        }
        // [round-7] LDS-staged N-tail full-K reduction skip. The N-tail
        // LDS kernel (grouped_ntail_kernel_lds) wrote ABSOLUTE values for
        // col >= fast_n cells in blocks fully contained in one group,
        // mirroring the dispatcher's launch gate (m_per_group passes
        // alignment) plus the device-side cross-group check. Same gates
        // here.
        if (!main_covers_n && col >= g.fast_n && lds_k_tail_safe &&
            block_in_group) {
            return;  // LDS N-tail kernel already wrote the absolute value.
        }
    }

    const int k0 = fast_covers_cell ? g.fast_k : 0;
    float acc = 0.0f;

    if constexpr (L == Layout::RCR) {
        // Vectorised K-loop fast path. Both A[row, kk] and B[group_idx,
        // col, kk] are stride-1 in K; the row strides (g.k) on each are
        // a bf16-element count, so when g.k % 4 == 0 the start of any
        // (row, 0) and (group_idx, col, 0) is 8-byte aligned and the
        // entire K range can be split into a vec4 prefix + scalar tail.
        // This replaces 4 separate scalar bf16 loads per K with a single
        // 8-byte global_load_dwordx2 — ~4× fewer VMEM transactions on
        // the gpt_oss K=2880 / K-tail=64 path which was HBM-issue-bound
        // at ~50 TF (probe). For K-tail correction (k0=fast_k mod 128)
        // and N-tail full reduction (k0=0), k0 is always even when g.k
        // is non-zero, and (g.k - k0) % 4 == 0 when g.k % 4 == 0; the
        // scalar tail handles non-multiple-of-4 K.
        const bf16* a_row = &g.a[coord<>(row, 0)];
        const bf16* b_row = &g.b[coord<>{0, group_idx, col, 0}];
        int kk = k0;
        if ((g.k % 4 == 0) && ((k0 & 3) == 0)) {
            const bf16x4* a_v4 = reinterpret_cast<const bf16x4*>(a_row);
            const bf16x4* b_v4 = reinterpret_cast<const bf16x4*>(b_row);
            const int j_start = k0 >> 2;
            const int j_end   = g.k >> 2;
            #pragma unroll 4
            for (int j = j_start; j < j_end; ++j) {
                bf16x4 a4 = a_v4[j];
                bf16x4 b4 = b_v4[j];
                acc += float(a4.lo.x) * float(b4.lo.x)
                     + float(a4.lo.y) * float(b4.lo.y)
                     + float(a4.hi.x) * float(b4.hi.x)
                     + float(a4.hi.y) * float(b4.hi.y);
            }
            kk = j_end << 2;
        }
        for (; kk < g.k; ++kk) {
            acc += load_bf16_scalar(g.a, row, kk) *
                   load_bf16_scalar_grp(g.b, group_idx, col, kk);
        }
    } else if constexpr (L == Layout::RRR) {
        // RRR: A stride-1 in K, B[group_idx, kk, col] stride-N in K — B
        // not vectorisable. Stay scalar (RRR is grad-X path; not in any
        // current grouped metric shape).
        for (int kk = k0; kk < g.k; ++kk) {
            acc += load_bf16_scalar(g.a, row, kk) *
                   load_bf16_scalar_grp(g.b, group_idx, kk, col);
        }
    } else {
        // CRR: A stride-M in K, B stride-N in K — neither vectorisable.
        // Stay scalar (CRR is dB grouped variable-K; not used in current
        // grouped metric).
        for (int kk = k0; kk < g.k; ++kk) {
            acc += load_bf16_scalar(g.a, kk, row) *
                   load_bf16_scalar_grp(g.b, group_idx, kk, col);
        }
    }

    if (fast_covers_cell && needs_k_tail) {
        store_bf16_scalar(g.c, row, col,
                          load_bf16_scalar(g.c, row, col) + acc);
    } else {
        store_bf16_scalar(g.c, row, col, acc);
    }
}

template __global__ void grouped_tail_kernel<Layout::RCR>(const grouped_layout_globals);
template __global__ void grouped_tail_kernel<Layout::RRR>(const grouped_layout_globals);
template __global__ void grouped_tail_kernel<Layout::CRR>(const grouped_layout_globals);

// =============================================================================
// LDS-staged K-tail correction kernel for the INTERIOR region (RCR only).
//
// Background: the scalar `grouped_tail_kernel` is HBM-issue-bound on gpt_oss
// K=2880 / K_remainder=64 — each (row, col) thread independently fetches its
// own A row × B col K-strip, with zero data reuse across threads. The 16×16
// thread block's 256 cells therefore issue 256 × 2 × K_rem global loads,
// reading the same A-row 16× (once per col) and the same B-col 16× (once per
// row). Round-7 vec4 brought this to ~80 TF on gpt_oss, but the host-pad
// fast path runs at 600-900 TF — a 10× gap that's purely no-LDS-reuse.
//
// This kernel handles the dominant case — ~98 % of tail cells on metric
// shapes (interior K-tail correction on uniform-M aligned grouped) — by:
//
//   1. Cooperatively loading the (TBM, K_REM) A K-strip + (TBN, K_REM) B
//      K-strip into LDS, with each thread fetching K_REM/TBN=4 elements
//      per side. 256 threads → 4 KB of LDS, 4 dword loads/thread/side.
//   2. After one __syncthreads(), each (rib, cib) thread reads its
//      A_lds[rib][*] × B_lds[cib][*] strip from LDS (LDS BW ~10× HBM)
//      and accumulates a scalar fp32 dot product over K_REM elements.
//   3. K-tail correction add: load existing C[row, col] (which the main
//      kernel already wrote with the [0, fast_k) reduction), add `acc`,
//      store back.
//
// Caller assumptions (failure → fall back to scalar `grouped_tail_kernel`):
//   * `g.fast_k < g.k` (kernel only runs when there IS a K-tail).
//   * `g.fast_n > 0` (interior region exists).
//   * Each row block (TBM=16 consecutive rows) lies inside ONE group.
//     For uniform-M groups with M_g a multiple of TBM=16 (the metric case:
//     M_per_group ∈ {2048, 4096}), this always holds. The LDS load picks
//     the group_idx of the block's first row and uses it for all B loads,
//     so cross-group blocks would corrupt B_lds. Caller must restrict the
//     launch grid to row blocks where this holds; the boundary tail kernel
//     handles the rest.
//   * `g.k - g.fast_k <= K_REM` (fits in LDS). Since `fast_k =
//     (k / K_TWO_TILE) * K_TWO_TILE` and `K_TWO_TILE = 128`, the K-tail is
//     always in [0, 128); template specialisation at K_REM=64 covers
//     k ≡ 64 mod 128 (gpt_oss K=2880); a future K_REM=128 instantiation
//     would handle k ≡ 0 (no K-tail, never reached) symmetrically. K-tail
//     of size 0 (k aligned) — caller skips this kernel entirely.
//
// Numerics: identical to the scalar tail kernel modulo (a) fp32-accumulator
// reduction order across K (associativity-only, ≤ 1 ULP difference at
// K_REM=64) and (b) LDS-load aliasing has no ULP effect.
//
// Bench (round 9 /tmp/bench_grouped_pad_vs_native.py on gpt_oss, post-LDS):
//   Down-B4-M2048   tail-only:  ~80 → ~XXX TF (target ~10×)
//   Down-B32-M4096  tail-only:  ~81 → ~XXX TF
// Closes ~80 % of the 10× pad-vs-native gap; remainder is the N-boundary
// path (col >= fast_n, full-K reduction) that still uses the scalar tail
// — but on metric shapes those are only n_tail × M_total cells (~2 % of
// tail work for gpt_oss N=2880).
//
// LDS bank-conflict: A_lds and B_lds are laid out [row][k] with K_REM=64.
// The 16-thread "row" group (cib varies, rib fixed) reads A_lds[rib][kk]
// — same row, same kk → broadcast (no conflict). The 16-thread "col"
// group (rib varies, cib fixed) reads B_lds[cib][kk] for kk fixed →
// same row, same kk → broadcast. K-loop within a thread reads
// A_lds[rib][kk++] sequentially → consecutive within one bank, no
// conflict.
template<Layout L, int K_REM>
__global__ void grouped_ktail_kernel_lds(const grouped_layout_globals g) {
    static_assert(L == Layout::RCR,
        "grouped_ktail_kernel_lds: RCR only — RRR/CRR fall back to scalar tail.");
    constexpr int TBM = TAIL_BLOCK_M;
    constexpr int TBN = TAIL_BLOCK_N;
    constexpr int NTHR = TBM * TBN;          // 256
    constexpr int A_TOTAL = TBM * K_REM;     // 1024 for K_REM=64
    constexpr int A_PER_THR = (A_TOTAL + NTHR - 1) / NTHR;
    constexpr int B_TOTAL = TBN * K_REM;
    constexpr int B_PER_THR = (B_TOTAL + NTHR - 1) / NTHR;

    // Round-14: LDS row padding to break the (cib * K_REM) bank-conflict
    // pattern. With K_REM=64 bf16, the row stride is 128 bytes = 32 banks,
    // so all 16 ``cib`` lanes within a wave hit banks {0,1} on B (16-way
    // conflict) and all 4 ``rib`` lanes hit banks {0,1} on A (4-way
    // conflict). Padding the row to 68 bf16 = 136 bytes = 34 banks (= 2
    // mod 32) makes ``cib * 34 mod 32 = cib * 2``, so each cib lane lands
    // on a distinct (even, odd) bank pair → 32 banks fully covered, no
    // conflict on either operand. Rocprof on gpt_oss-GateUP-B32-M4096
    // (round-13): K-tail kernel was 77.3 % of total wall (12.4 ms / call,
    // ~6 TF on 64-fma cells), this fix brings ds_read_b64 from 8-cycle
    // serialised banks down to 1-cycle parallel banks.
    constexpr int K_REM_LDS = K_REM + 4;     // 68 bf16 / 136 bytes / 34 banks
    __shared__ bf16 A_lds[TBM * K_REM_LDS];
    __shared__ bf16 B_lds[TBN * K_REM_LDS];
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];

    const int rib = threadIdx.y;
    const int cib = threadIdx.x;
    const int tid = rib * blockDim.x + cib;

    // Cooperative G+1 offsets prefill (once per block).
    if (tid < MAX_G_PLUS_1) {
        s_offs[tid] = (tid <= g.G) ? static_cast<int>(g.group_offs[tid]) : 0;
    }
    __syncthreads();

    const int row_block_base = blockIdx.y * TBM;
    const int col_block_base = blockIdx.x * TBN;
    // Round-11: cover the ENTIRE [0, g.n) col range, not just
    // [0, g.fast_n). The partial last col-tile (col_block_base + TBN
    // straddling g.n) is handled per-thread via the ``col < g.n`` check
    // below, mirroring main kernel's ``store_c_tile_n_masked`` behaviour.
    if (row_block_base >= g.M_total || col_block_base >= g.n) return;

    // Pick a single group_idx for the entire block (assumes block within one
    // group — see caller assumptions above). Use the FIRST row of the block.
    int group_idx = 0;
    #pragma unroll 1
    for (int gi = 0; gi < g.G; ++gi) {
        if (row_block_base < s_offs[gi + 1]) { group_idx = gi; break; }
    }

    const int k0 = g.fast_k;
    const int K_rem_dyn = g.k - k0;
    // K_rem_dyn must equal K_REM for this template instantiation (callers
    // dispatch by g.k - g.fast_k). Guard against accidental mismatch (e.g.,
    // dispatch bug) — fall back to early-exit, scalar tail will fix things.
    if (K_rem_dyn != K_REM) return;

    // [round-6 cross-group safety] If the block straddles a group
    // boundary, fall back to per-row scalar K-tail correction in this
    // same kernel. Each row uses its own group_idx for B indexing.
    // [round-10] Inlined fallback so the host can drop the scalar tail
    // kernel launch entirely on the RCR uniform-aligned path. The
    // host's ``m_per_group`` gate ensures cross_boundary is rare (only
    // happens for non-uniform group_lens whose avg happens to be
    // m_per_group-aligned).
    const bool cross_boundary = (row_block_base + TBM > s_offs[group_idx + 1]);
    if (cross_boundary) {
        const int row = row_block_base + rib;
        const int col = col_block_base + cib;
        if (row < g.M_total && col < g.n) {
            int row_group = 0;
            #pragma unroll 1
            for (int gi = 0; gi < g.G; ++gi) {
                if (row < s_offs[gi + 1]) { row_group = gi; break; }
            }
            // Round-18: cross_boundary fallback now uses the same
            // ``__builtin_amdgcn_fdot2_f32_bf16`` packed dot + 2-way
            // parallel acc as the fast LDS path (see round-15 of bf16
            // K-tail). This branch only fires when a tail block
            // straddles a group boundary (non-uniform group_lens at
            // launch); the metric uses uniform group_lens so it does
            // not exercise this branch — but a real bench with skewed
            // group_lens benefits ~2x on the cross_boundary cells.
            typedef __attribute__((__vector_size__(2 * sizeof(__bf16)))) __bf16 bf16x2_v;
            float acc_s_lo = 0.0f, acc_s_hi = 0.0f;
            const bf16* a_row = &g.a[coord<>(row, 0)];
            const bf16* b_row = &g.b[coord<>{0, row_group, col, 0}];
            int kk = k0;
            if ((g.k % 4 == 0) && ((k0 & 3) == 0)) {
                const bf16x4* a_v4 = reinterpret_cast<const bf16x4*>(a_row);
                const bf16x4* b_v4 = reinterpret_cast<const bf16x4*>(b_row);
                const int j_start = k0 >> 2;
                const int j_end   = g.k >> 2;
                for (int j = j_start; j < j_end; ++j) {
                    bf16x4 a4 = a_v4[j];
                    bf16x4 b4 = b_v4[j];
                    acc_s_lo = __builtin_amdgcn_fdot2_f32_bf16(
                        *reinterpret_cast<const bf16x2_v*>(&a4.lo),
                        *reinterpret_cast<const bf16x2_v*>(&b4.lo),
                        acc_s_lo, false);
                    acc_s_hi = __builtin_amdgcn_fdot2_f32_bf16(
                        *reinterpret_cast<const bf16x2_v*>(&a4.hi),
                        *reinterpret_cast<const bf16x2_v*>(&b4.hi),
                        acc_s_hi, false);
                }
                kk = j_end << 2;
            }
            float acc_s = acc_s_lo + acc_s_hi;
            for (; kk < g.k; ++kk) {
                acc_s += load_bf16_scalar(g.a, row, kk) *
                         load_bf16_scalar_grp(g.b, row_group, col, kk);
            }
            store_bf16_scalar(g.c, row, col,
                              load_bf16_scalar(g.c, row, col) + acc_s);
        }
        return;
    }

    // [round-9] Vec4 cooperative load (1 dwordx2 / thread instead of 4
    // separate scalar bf16 loads). With NTHR=256 threads and A_TOTAL=1024
    // bf16 = 256 vec4, each thread owns exactly one vec4. Address pattern:
    //   tid=0  -> A_lds[0..3]   (row 0, k 0..3)
    //   tid=1  -> A_lds[4..7]   (row 0, k 4..7)
    //   tid=15 -> A_lds[60..63] (row 0, k 60..63)
    //   tid=16 -> A_lds[64..67] (row 1, k 0..3)
    //   ...
    // 8-byte alignment: g.a's row stride is g.k bf16 elements; for K=2880
    // (multiple of 4) every (r_global, k0 + 4*j) start is 8-byte aligned.
    constexpr int VEC = 4;
    constexpr int VECS_PER_ROW = K_REM / VEC;
    static_assert(K_REM % VEC == 0, "K_REM must be vec4-aligned");
    static_assert(NTHR == TBM * VECS_PER_ROW,
        "Each thread must own exactly one vec4 of A.");
    {
        const int r_in_blk = tid / VECS_PER_ROW;
        const int kk_v = tid - r_in_blk * VECS_PER_ROW;
        const int kk_start = kk_v * VEC;
        const int r_global = row_block_base + r_in_blk;
        bf16x4 va{};
        if (r_global < g.M_total) {
            const bf16* ap = &g.a[coord<>(r_global, k0 + kk_start)];
            va = *reinterpret_cast<const bf16x4*>(ap);
        }
        *reinterpret_cast<bf16x4*>(&A_lds[r_in_blk * K_REM_LDS + kk_start]) = va;
    }
    {
        const int c_in_blk = tid / VECS_PER_ROW;
        const int kk_v = tid - c_in_blk * VECS_PER_ROW;
        const int kk_start = kk_v * VEC;
        const int c_global = col_block_base + c_in_blk;
        bf16x4 vb{};
        // Round-11: load up to ``g.n`` (was ``g.fast_n``). For the
        // partial last col-tile (col >= g.n) we use the zero pad below;
        // the main kernel's ``store_c_tile_n_masked`` ensures we don't
        // read uninitialised C cells in the col >= g.n region either.
        if (c_global < g.n) {
            const bf16* bp = &g.b[coord<>{0, group_idx, c_global, k0 + kk_start}];
            vb = *reinterpret_cast<const bf16x4*>(bp);
        }
        *reinterpret_cast<bf16x4*>(&B_lds[c_in_blk * K_REM_LDS + kk_start]) = vb;
    }
    __syncthreads();

    const int row = row_block_base + rib;
    const int col = col_block_base + cib;
    // Round-11: skip cells whose col is beyond the real ``g.n`` (the
    // partial last col-tile). Main kernel did NOT write those cells
    // (column-masked C store), and we must not RMW them.
    if (row >= g.M_total || col >= g.n) return;

    // Round-15: replace the scalar fp32 fma chain with
    // ``v_dot2_f32_bf16`` (CDNA4 packed bf16 dot-product), which does
    // 2 bf16 muls + 1 fp32 add in 1 cycle on the VALU, vs 2 separate
    // fp32 fmas (2 cycles) in the bf16-cast-to-float path. K_REM=64 was
    // 64 scalar fp32 fmas per thread (~64 cycles); now it's 32
    // ``v_dot2`` ops (~32 cycles) — halves the K-tail compute latency
    // and saves the 4 bf16->fp32 converts per kk_v.
    //
    // Numerics: identical math (a.x*b.x + a.y*b.y added to acc) to the
    // fp32-cast path; the intrinsic accumulates in fp32 as well.
    // Compiler emits ``v_dot2c_f32_bf16`` directly (verified by
    // -save-temps assembly).
    typedef __attribute__((__vector_size__(2 * sizeof(__bf16)))) __bf16 bf16x2_v;
    constexpr int FMA_VEC = 4;
    constexpr int K_VECS = K_REM / FMA_VEC;
    static_assert(K_REM % FMA_VEC == 0, "K_REM must be vec4-aligned for inner fma");
    float acc = 0.0f;
    #pragma unroll
    for (int kk_v = 0; kk_v < K_VECS; ++kk_v) {
        bf16x4 a4 = *reinterpret_cast<const bf16x4*>(
            &A_lds[rib * K_REM_LDS + kk_v * FMA_VEC]);
        bf16x4 b4 = *reinterpret_cast<const bf16x4*>(
            &B_lds[cib * K_REM_LDS + kk_v * FMA_VEC]);
        acc = __builtin_amdgcn_fdot2_f32_bf16(
            *reinterpret_cast<const bf16x2_v*>(&a4.lo),
            *reinterpret_cast<const bf16x2_v*>(&b4.lo),
            acc, false);
        acc = __builtin_amdgcn_fdot2_f32_bf16(
            *reinterpret_cast<const bf16x2_v*>(&a4.hi),
            *reinterpret_cast<const bf16x2_v*>(&b4.hi),
            acc, false);
    }

    // K-tail correction add. Main grouped kernel already stored the
    // [0, fast_k) reduction at C[row, col]; we add the [fast_k, k) part.
    store_bf16_scalar(g.c, row, col,
                      load_bf16_scalar(g.c, row, col) + acc);
}

template __global__ void grouped_ktail_kernel_lds<Layout::RCR, 64>(const grouped_layout_globals);

// =============================================================================
// Round-19 (BF16, mirrors FP8 round-18): MFMA-based K-tail correction kernel
// for RCR.
//
// rocprof on BF16 grouped gpt_oss-GateUP B=32 M=4096 traced the LDS-staged
// fdot2_f32_bf16 K-tail kernel (round-15) at the dominant-fraction-of-wall
// at ~30 TFLOPS (vs 1138 TF Triton total → BF16 gpt_oss ratio stalled at
// 0.41-0.65). The packed bf16x2 dot-product (2 bf16 muls + 1 fp32 add per
// cycle) is already 2× the scalar fma path, but still 50× below MFMA
// throughput. The main grouped kernel sweeping [0, fast_k) is fine; the
// 64-element K-tail dominates because of zero data reuse + scalar VALU.
//
// This kernel fuses the K=64 tail into TWO ``v_mfma_f32_16x16x32_bf16``
// calls per (16M × 16N) cell-tile. Each call covers K=32 (8 bf16/lane);
// for K=64 we shift the K offset between calls. No K-zero-padding (unlike
// FP8 which used the K=128 mfma_scale variant) — bf16 only has the K=32
// mfma so 2 back-to-back calls give 100% mfma utilisation.
//
// Geometry mirrors ``grouped_ktail_kernel_lds<RCR, 64>`` exactly:
//   * Block: 1 wave (64 threads). blockDim = (64,).
//   * Grid: ceil_div(n, 16) × ceil_div(M_total, 16) — same as scalar/LDS.
//   * Per lane: load 16 bf16 of A (= 2× bf16x8 = 32 bytes) + 16 bf16 of B
//     and run 2× mfma_f32_16x16x32_bf16; standard CDNA 16x16x32 layout
//     gives each lane 4 output cells C[(t/16)*4 + 0..3, t%16].
//   * RMW into bf16 g.c (no FP8 combined_scale).
//
// Cross-group fallback (block straddles a group boundary in M): preserved
// via the same per-row vec4 fdot2_f32_bf16 + 2-way ILP scalar dot loop the
// LDS variant uses (round-15/18). Host hint ``m_per_group % 16 == 0``
// keeps this branch unreachable on uniform group_lens (the metric path).
//
// FP8-only RCR caveat applies symmetrically: B is row-major [G, N, K] so
// ABt MFMA directly applies; RRR/CRR have B not stride-1 in K and stay on
// the existing scalar tail.
// =============================================================================
template<Layout L, int K_REM>
__global__ void grouped_ktail_kernel_mfma(const grouped_layout_globals g) {
    static_assert(L == Layout::RCR,
        "grouped_ktail_kernel_mfma (BF16): RCR only — RRR/CRR fall back to scalar tail.");
    static_assert(K_REM == 64,
        "grouped_ktail_kernel_mfma (BF16): K_REM must be 64 (= 2 × mfma_16x16x32_bf16).");
    constexpr int TBM = TAIL_BLOCK_M;       // 16
    constexpr int TBN = TAIL_BLOCK_N;       // 16
    constexpr int K_PER_LANE_CHUNK = 8;     // mfma_16x16x32_bf16: 8 bf16/lane per call
    constexpr int K_PER_MFMA = 32;          // K dim per mfma call
    constexpr int N_MFMA = K_REM / K_PER_MFMA;  // = 2 calls
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];

    const int tid = threadIdx.x;            // single-wave block, 64 threads
    if (tid <= g.G && tid < MAX_G_PLUS_1) {
        s_offs[tid] = static_cast<int>(g.group_offs[tid]);
    }
    __syncthreads();

    const int row_block_base = blockIdx.y * TBM;
    const int col_block_base = blockIdx.x * TBN;
    if (row_block_base >= g.M_total || col_block_base >= g.n) return;

    int group_idx = 0;
    #pragma unroll 1
    for (int gi = 0; gi < g.G; ++gi) {
        if (row_block_base < s_offs[gi + 1]) { group_idx = gi; break; }
    }

    const int K_rem_dyn = g.k - g.fast_k;
    if (K_rem_dyn != K_REM) return;
    const int k0 = g.fast_k;

    // Cross-group fallback (per-row vec4 fdot2_f32_bf16 + 2-way ILP).
    // Mirror ``grouped_ktail_kernel_lds`` cross_boundary path. Single
    // wave: 64 lanes cover the 256 cells in 4 passes (4 cells/lane).
    const bool cross_boundary = (row_block_base + TBM > s_offs[group_idx + 1]);
    if (cross_boundary) {
        typedef __attribute__((__vector_size__(2 * sizeof(__bf16)))) __bf16 bf16x2_v;
        #pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            const int rib = (slot * 16) + (tid / TBN);
            const int cib = tid % TBN;
            const int row = row_block_base + rib;
            const int col = col_block_base + cib;
            if (row >= g.M_total || col >= g.n) continue;
            int row_group = 0;
            #pragma unroll 1
            for (int gi = 0; gi < g.G; ++gi) {
                if (row < s_offs[gi + 1]) { row_group = gi; break; }
            }
            float acc_s_lo = 0.f, acc_s_hi = 0.f;
            const bf16* a_row = &g.a[coord<>(row, 0)];
            const bf16* b_row = &g.b[coord<>{0, row_group, col, 0}];
            int kk = k0;
            if ((g.k % 4 == 0) && ((k0 & 3) == 0)) {
                const bf16x4* a_v4 = reinterpret_cast<const bf16x4*>(a_row);
                const bf16x4* b_v4 = reinterpret_cast<const bf16x4*>(b_row);
                const int j_start = k0 >> 2;
                const int j_end   = g.k >> 2;
                for (int j = j_start; j < j_end; ++j) {
                    bf16x4 a4 = a_v4[j];
                    bf16x4 b4 = b_v4[j];
                    acc_s_lo = __builtin_amdgcn_fdot2_f32_bf16(
                        *reinterpret_cast<const bf16x2_v*>(&a4.lo),
                        *reinterpret_cast<const bf16x2_v*>(&b4.lo),
                        acc_s_lo, false);
                    acc_s_hi = __builtin_amdgcn_fdot2_f32_bf16(
                        *reinterpret_cast<const bf16x2_v*>(&a4.hi),
                        *reinterpret_cast<const bf16x2_v*>(&b4.hi),
                        acc_s_hi, false);
                }
                kk = j_end << 2;
            }
            float acc_s = acc_s_lo + acc_s_hi;
            for (; kk < g.k; ++kk) {
                acc_s += load_bf16_scalar(g.a, row, kk) *
                         load_bf16_scalar_grp(g.b, row_group, col, kk);
            }
            store_bf16_scalar(g.c, row, col,
                              load_bf16_scalar(g.c, row, col) + acc_s);
        }
        return;
    }

    // ----- Fast MFMA path ------------------------------------------------
    // Lane (t):  row_in_blk = t % 16,  k_chunk = t / 16  (0..3).
    //   For mfma_16x16x32_bf16 (K=32 per call):
    //     A operand: lane t holds A[t%16, k_off + (t/16)*8 .. (t/16)*8+7]
    //                = bf16x8 = 8 bf16 = 16 bytes.
    //     B operand (RCR ABt): lane t holds B[t%16, k_off + (t/16)*8 ..]
    //                          = same layout, B treated as [N, K] row-major.
    //   Two calls cover K=64: k_off = k0 (first), k_off = k0+32 (second).
    typedef __attribute__((__vector_size__(8 * sizeof(__bf16)))) __bf16 bf16x8_t;
    typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;

    const int row_in_blk = tid % TBM;
    const int chunk      = tid / TBM;       // 0..3

    const int g_row = row_block_base + row_in_blk;
    const int g_col = col_block_base + row_in_blk;

    const int k_lane_offset = chunk * K_PER_LANE_CHUNK;  // 0, 8, 16, 24

    bf16x8_t a_pack0, a_pack1;
    bf16x8_t b_pack0, b_pack1;
    if (g_row < g.M_total) {
        const bf16* a_ptr0 = &g.a[coord<>(g_row, k0 + k_lane_offset)];
        const bf16* a_ptr1 = &g.a[coord<>(g_row, k0 + K_PER_MFMA + k_lane_offset)];
        a_pack0 = *reinterpret_cast<const bf16x8_t*>(a_ptr0);
        a_pack1 = *reinterpret_cast<const bf16x8_t*>(a_ptr1);
    } else {
        a_pack0 = bf16x8_t{};
        a_pack1 = bf16x8_t{};
    }
    if (g_col < g.n) {
        const bf16* b_ptr0 = &g.b[coord<>{0, group_idx, g_col, k0 + k_lane_offset}];
        const bf16* b_ptr1 = &g.b[coord<>{0, group_idx, g_col, k0 + K_PER_MFMA + k_lane_offset}];
        b_pack0 = *reinterpret_cast<const bf16x8_t*>(b_ptr0);
        b_pack1 = *reinterpret_cast<const bf16x8_t*>(b_ptr1);
    } else {
        b_pack0 = bf16x8_t{};
        b_pack1 = bf16x8_t{};
    }

    // Two mfma_f32_16x16x32_bf16 calls accumulating over K=64.
    floatx4_t acc = floatx4_t{0.f, 0.f, 0.f, 0.f};
    acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a_pack0, b_pack0, acc, 0, 0, 0);
    acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a_pack1, b_pack1, acc, 0, 0, 0);

    // Output: lane t → cells C[(t/16)*4 + (0..3), t%16].
    const int out_row_base = row_block_base + chunk * 4;
    const int out_col      = col_block_base + row_in_blk;
    if (out_col >= g.n) return;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = out_row_base + i;
        if (r >= g.M_total) break;
        const float existing = load_bf16_scalar(g.c, r, out_col);
        const float new_val  = existing + acc[i];
        store_bf16_scalar(g.c, r, out_col, new_val);
    }
}

template __global__ void grouped_ktail_kernel_mfma<Layout::RCR, 64>(const grouped_layout_globals);

// =============================================================================
// Round-21 (BF16): 32x32x16 MFMA-based K-tail correction kernel for RCR.
//
// Round-19 used 2× ``v_mfma_f32_16x16x32_bf16`` per (16M × 16N) cell tile —
// already 100 % MFMA utilization (no zero padding because BF16 has a K=32
// mfma natively). But the 16x16 cell layout means each (M, N) tile is
// loaded into 256 cells per block; for N-stripe sweeps the same B[col_block,
// k0..k0+64) tile is re-fetched by every M-row block. K-tail grid for
// gpt_oss-GateUP B=32-M=4096: 8192 row_blocks × 360 col_blocks ≈ 2.95 M
// blocks, each reading 4 KB of A+B = ~12 GB total HBM traffic on the K-tail
// path alone.
//
// Mirror of FP8 round-20: 32x32 cell tile via 4× ``v_mfma_f32_32x32x16_bf16``
// (K=16 each, K=64 → 4 calls, 100 % util). Per-block cell tile expands
// 4× (256 → 1024 cells); K-tail grid shrinks 4× while per-mfma useful
// work doubles by sharing A/B operands across more cells:
//   * Per-block A read: 16x64=1024 bf16 → 32x64=2048 bf16 (2× per block)
//   * Per-block B read: 16x64=1024 bf16 → 32x64=2048 bf16 (2× per block)
//   * Per-cell read: 4 KB / 256 = 16 byte/cell → 8 KB / 1024 = 8 byte/cell (½)
// I.e. mem traffic per output cell halves while mfma rate stays at 100 %
// utilization — net throughput improvement on the K-tail dominated path.
//
// Layout (verified by /tmp/mfma_bf16_3232x16_test, gfx950):
//   * Input A[32, 16] bf16:
//       lane t (0..63): row = t % 32, K-chunk = t / 32 (0 or 1).
//       lane t supplies a_pack = 8 bf16 (bf16x8_t) of A[t%32, K_chunk*8 + 0..7].
//   * Input B[32, 16] bf16: same, B treated as [N, K] row-major (B^T = [K, N]).
//   * Output D[32, 32] fp32, 16 floats per lane (floatx16_t):
//       For lane t at d[i] (i = 0..15):
//         col       = t % 32
//         chunk     = t / 32
//         row_group = i / 4
//         row_in_grp= (i % 4) + chunk * 4
//         row       = row_group * 8 + row_in_grp
//   * Same output layout as FP8 mfma_32x32x64_f8f6f4 (round-20). Both are
//     standard CDNA 32x32 fp32 outputs — verified on gfx950.
//
// Host hint requirements:
//   * ``g.m_per_group >= 32 && g.m_per_group % 32 == 0`` (TBM=32). Falls
//     back to round-19 16x16 mfma kernel otherwise.
// =============================================================================
template<Layout L, int K_REM>
__global__ void grouped_ktail_kernel_mfma32x32(const grouped_layout_globals g) {
    static_assert(L == Layout::RCR,
        "grouped_ktail_kernel_mfma32x32 (BF16): RCR only.");
    static_assert(K_REM == 64,
        "grouped_ktail_kernel_mfma32x32 (BF16): K_REM must be 64 (= 4 × mfma_32x32x16_bf16).");
    constexpr int TBM = 32;
    constexpr int TBN = 32;
    constexpr int K_PER_MFMA      = 16;                  // mfma_32x32x16_bf16 K dim
    constexpr int N_MFMA          = K_REM / K_PER_MFMA;  // 4 calls
    constexpr int K_PER_LANE_CHUNK = 8;                  // 8 bf16/lane per mfma call
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];

    const int tid = threadIdx.x;            // single-wave block, 64 threads
    if (tid <= g.G && tid < MAX_G_PLUS_1) {
        s_offs[tid] = static_cast<int>(g.group_offs[tid]);
    }
    __syncthreads();

    const int row_block_base = blockIdx.y * TBM;
    const int col_block_base = blockIdx.x * TBN;
    if (row_block_base >= g.M_total || col_block_base >= g.n) return;

    int group_idx = 0;
    #pragma unroll 1
    for (int gi = 0; gi < g.G; ++gi) {
        if (row_block_base < s_offs[gi + 1]) { group_idx = gi; break; }
    }

    const int K_rem_dyn = g.k - g.fast_k;
    if (K_rem_dyn != K_REM) return;
    const int k0 = g.fast_k;

    // Cross-group fallback (block straddles a group boundary in M, or hits
    // M_total tail). 64 lanes × 16 cells = 1024 cells covered via per-row
    // vec4 fdot2_f32_bf16 + 2-way ILP scalar dot loop. Unreachable on
    // uniform group_lens with M_g % 32 == 0 (the metric path).
    const bool cross_boundary = (row_block_base + TBM > s_offs[group_idx + 1]);
    if (cross_boundary) {
        typedef __attribute__((__vector_size__(2 * sizeof(__bf16)))) __bf16 bf16x2_v;
        // Each lane covers col = tid % 32 for 16 distinct rows. Split 32
        // rows across the 2 chunks: chunk-0 lane serves even rows (0,2,...),
        // chunk-1 serves odd rows (1,3,...). 16 rows per lane — half work
        // each — keeps the fallback path simple. This branch only fires on
        // non-uniform group_lens which the metric path never hits.
        const int col = col_block_base + (tid % 32);
        if (col < g.n) {
            #pragma unroll
            for (int rr = 0; rr < TBM; ++rr) {
                if ((rr % 2) != (tid / 32)) continue;
                const int row = row_block_base + rr;
                if (row >= g.M_total) break;
                int row_group = 0;
                #pragma unroll 1
                for (int gi = 0; gi < g.G; ++gi) {
                    if (row < s_offs[gi + 1]) { row_group = gi; break; }
                }
                float acc_s_lo = 0.f, acc_s_hi = 0.f;
                const bf16* a_row = &g.a[coord<>(row, 0)];
                const bf16* b_row = &g.b[coord<>{0, row_group, col, 0}];
                int kk = k0;
                if ((g.k % 4 == 0) && ((k0 & 3) == 0)) {
                    const bf16x4* a_v4 = reinterpret_cast<const bf16x4*>(a_row);
                    const bf16x4* b_v4 = reinterpret_cast<const bf16x4*>(b_row);
                    const int j_start = k0 >> 2;
                    const int j_end   = g.k >> 2;
                    for (int j = j_start; j < j_end; ++j) {
                        bf16x4 a4 = a_v4[j];
                        bf16x4 b4 = b_v4[j];
                        acc_s_lo = __builtin_amdgcn_fdot2_f32_bf16(
                            *reinterpret_cast<const bf16x2_v*>(&a4.lo),
                            *reinterpret_cast<const bf16x2_v*>(&b4.lo),
                            acc_s_lo, false);
                        acc_s_hi = __builtin_amdgcn_fdot2_f32_bf16(
                            *reinterpret_cast<const bf16x2_v*>(&a4.hi),
                            *reinterpret_cast<const bf16x2_v*>(&b4.hi),
                            acc_s_hi, false);
                    }
                    kk = j_end << 2;
                }
                float acc_s = acc_s_lo + acc_s_hi;
                for (; kk < g.k; ++kk) {
                    acc_s += load_bf16_scalar(g.a, row, kk) *
                             load_bf16_scalar_grp(g.b, row_group, col, kk);
                }
                store_bf16_scalar(g.c, row, col,
                                  load_bf16_scalar(g.c, row, col) + acc_s);
            }
        }
        return;
    }

    // ----- Fast MFMA path ------------------------------------------------
    // 4× mfma_f32_32x32x16_bf16 over K=64 → 100 % util, 2× cell-tile vs
    // round-19 → halved per-cell mem traffic.
    typedef __attribute__((__vector_size__(8 * sizeof(__bf16)))) __bf16 bf16x8_t;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;

    const int row_in_blk = tid % 32;
    const int chunk      = tid / 32;        // 0 or 1
    const int k_lane_offset = chunk * K_PER_LANE_CHUNK;  // 0 or 8

    const int g_row = row_block_base + row_in_blk;
    const int g_col = col_block_base + row_in_blk;

    bf16x8_t a_pack[N_MFMA];
    bf16x8_t b_pack[N_MFMA];
    if (g_row < g.M_total) {
        #pragma unroll
        for (int j = 0; j < N_MFMA; ++j) {
            const int k_off = k0 + j * K_PER_MFMA + k_lane_offset;
            const bf16* a_ptr = &g.a[coord<>(g_row, k_off)];
            a_pack[j] = *reinterpret_cast<const bf16x8_t*>(a_ptr);
        }
    } else {
        #pragma unroll
        for (int j = 0; j < N_MFMA; ++j) a_pack[j] = bf16x8_t{};
    }
    if (g_col < g.n) {
        #pragma unroll
        for (int j = 0; j < N_MFMA; ++j) {
            const int k_off = k0 + j * K_PER_MFMA + k_lane_offset;
            const bf16* b_ptr = &g.b[coord<>{0, group_idx, g_col, k_off}];
            b_pack[j] = *reinterpret_cast<const bf16x8_t*>(b_ptr);
        }
    } else {
        #pragma unroll
        for (int j = 0; j < N_MFMA; ++j) b_pack[j] = bf16x8_t{};
    }

    // 4 mfma calls accumulate K=[0,16), [16,32), [32,48), [48,64).
    floatx16_t acc{};
    #pragma unroll
    for (int j = 0; j < N_MFMA; ++j) {
        acc = __builtin_amdgcn_mfma_f32_32x32x16_bf16(
            a_pack[j], b_pack[j], acc, 0, 0, 0);
    }

    // Output: lane t → cells C[r, t%32] for 16 r's via the standard
    // CDNA 32x32 fp32 layout (verified on gfx950, see header comment).
    const int out_col = col_block_base + row_in_blk;
    if (out_col >= g.n) return;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int row_group     = i >> 2;
        const int row_in_group  = (i & 3) + chunk * 4;
        const int local_row     = row_group * 8 + row_in_group;
        const int r             = row_block_base + local_row;
        if (r >= g.M_total) continue;
        const float existing = load_bf16_scalar(g.c, r, out_col);
        const float new_val  = existing + acc[i];
        store_bf16_scalar(g.c, r, out_col, new_val);
    }
}

template __global__ void grouped_ktail_kernel_mfma32x32<Layout::RCR, 64>(const grouped_layout_globals);

// =============================================================================
// LDS-staged N-tail full-K reduction kernel for the partial last col-tile
// region [fast_n, n) (RCR only).
//
// Background: when the main grouped kernel uses bpc = fast_n / BLOCK_SIZE
// (the safe pre-round-6 path; ceil_div coverage of N is gated to fast_k == k
// because dual-misaligned shapes triggered a memory fault in KI=0 dynamic
// main kernel — see comment in dispatch_grouped), the N-tail region
// [fast_n, n) is computed by `grouped_tail_kernel`'s scalar fp32 vec4-loaded
// path: per-(row, col) thread, K-loop over [0, k), 8-byte load A & B with
// no LDS reuse. On gpt_oss N=2880/5760 K=2880 this hits ~5 TF (HBM-bound).
//
// This kernel handles the same cells with cooperative LDS staging:
//   1. Block of (TBM=16, TBN=16) cells, 256 threads.
//   2. Loop K in chunks of K_CHUNK=64. Per chunk:
//        a. Coop-load A[block_rows, k_chunk] -> A_lds (16*64*2 = 2 KB,
//           4 bf16 elements per thread).
//        b. Coop-load B[group_idx, block_cols, k_chunk] -> B_lds (2 KB).
//        c. __syncthreads().
//        d. Per (rib, cib) thread: K_CHUNK fma over LDS data (row
//           broadcast across cib lane-group, col broadcast across rib).
//        e. __syncthreads() before next chunk overwrites LDS.
//   3. Store full result to C[row, col] (main did NOT write here).
//
// Cross-group safety mirrors `grouped_ktail_kernel_lds` (round-6): per-block
// `row_block_base + TBM <= s_offs[group_idx + 1]` check, fall back to scalar
// tail for cross-group blocks (which pick group_idx per-row).
//
// LDS bank-conflict: A_lds laid out [row][k]. For fixed `rib`, threads with
// different `cib` read A_lds[rib][kk] for the SAME kk -> broadcast (no
// conflict). For fixed `cib`, threads with different `rib` read
// B_lds[cib][kk] for the SAME kk -> broadcast.
// =============================================================================
template<Layout L, int K_CHUNK>
__global__ void grouped_ntail_kernel_lds(const grouped_layout_globals g) {
    static_assert(L == Layout::RCR,
        "grouped_ntail_kernel_lds: RCR only — RRR/CRR fall back to scalar tail.");
    constexpr int TBM = TAIL_BLOCK_M;        // 16
    constexpr int TBN = TAIL_BLOCK_N;        // 16
    constexpr int NTHR = TBM * TBN;           // 256
    constexpr int A_CHUNK = TBM * K_CHUNK;    // 16 * 64 = 1024 elements
    constexpr int B_CHUNK = TBN * K_CHUNK;    // 16 * 64 = 1024 elements
    constexpr int A_PER_THR = (A_CHUNK + NTHR - 1) / NTHR;
    constexpr int B_PER_THR = (B_CHUNK + NTHR - 1) / NTHR;

    __shared__ bf16 A_lds[A_CHUNK];
    __shared__ bf16 B_lds[B_CHUNK];
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];

    const int rib = threadIdx.y;
    const int cib = threadIdx.x;
    const int tid = rib * blockDim.x + cib;

    if (tid < MAX_G_PLUS_1) {
        s_offs[tid] = (tid <= g.G) ? static_cast<int>(g.group_offs[tid]) : 0;
    }
    __syncthreads();

    const int row_block_base = blockIdx.y * TBM;
    // grid.x indexes the n-tail region [fast_n, n); shift by fast_n.
    const int col_block_base = g.fast_n + blockIdx.x * TBN;
    if (row_block_base >= g.M_total || col_block_base >= g.n) return;

    int group_idx = 0;
    #pragma unroll 1
    for (int gi = 0; gi < g.G; ++gi) {
        if (row_block_base < s_offs[gi + 1]) { group_idx = gi; break; }
    }

    // [round-7] Cross-group safety mirror. Single-group_idx B-strip load
    // would feed wrong rows for the upper part of a cross-boundary block.
    // [round-10] Inlined scalar fallback so the host can drop the scalar
    // tail launch on the RCR uniform-aligned path. Per-row group_idx +
    // full-K reduction (vec4 inner where K%4==0).
    const bool cross_boundary = (row_block_base + TBM > s_offs[group_idx + 1]);
    if (cross_boundary) {
        const int row_s = row_block_base + rib;
        const int col_s = col_block_base + cib;
        if (row_s < g.M_total && col_s < g.n) {
            int row_group = 0;
            #pragma unroll 1
            for (int gi = 0; gi < g.G; ++gi) {
                if (row_s < s_offs[gi + 1]) { row_group = gi; break; }
            }
            float acc_s = 0.0f;
            const bf16* a_row = &g.a[coord<>(row_s, 0)];
            const bf16* b_row = &g.b[coord<>{0, row_group, col_s, 0}];
            int kk = 0;
            if (g.k % 4 == 0) {
                const bf16x4* a_v4 = reinterpret_cast<const bf16x4*>(a_row);
                const bf16x4* b_v4 = reinterpret_cast<const bf16x4*>(b_row);
                const int j_end = g.k >> 2;
                for (int j = 0; j < j_end; ++j) {
                    bf16x4 a4 = a_v4[j];
                    bf16x4 b4 = b_v4[j];
                    acc_s += float(a4.lo.x) * float(b4.lo.x)
                           + float(a4.lo.y) * float(b4.lo.y)
                           + float(a4.hi.x) * float(b4.hi.x)
                           + float(a4.hi.y) * float(b4.hi.y);
                }
                kk = j_end << 2;
            }
            for (; kk < g.k; ++kk) {
                acc_s += load_bf16_scalar(g.a, row_s, kk) *
                         load_bf16_scalar_grp(g.b, row_group, col_s, kk);
            }
            store_bf16_scalar(g.c, row_s, col_s, acc_s);
        }
        return;
    }

    const int row = row_block_base + rib;
    const int col = col_block_base + cib;
    const bool active_cell = (row < g.M_total) && (col < g.n);

    float acc = 0.0f;

    // [round-9] Vec4 cooperative load (1 dwordx2 / thread / chunk). With
    // K_CHUNK=64 and TBM=16: 1024 bf16 / chunk / operand = 256 vec4 / chunk
    // / operand. NTHR=256 -> exactly 1 vec4 per thread per chunk per operand.
    constexpr int VEC = 4;
    constexpr int VECS_PER_ROW = K_CHUNK / VEC;
    static_assert(K_CHUNK % VEC == 0, "K_CHUNK must be vec4-aligned");
    static_assert(NTHR == TBM * VECS_PER_ROW,
        "Each thread must own exactly one vec4 of A per chunk.");
    const int r_in_blk_a = tid / VECS_PER_ROW;
    const int kk_v_a = tid - r_in_blk_a * VECS_PER_ROW;
    const int kk_start_a = kk_v_a * VEC;
    const int c_in_blk_b = r_in_blk_a;
    const int kk_start_b = kk_start_a;

    // Loop K in K_CHUNK slices. The compile-time partial loop over K_CHUNK
    // unrolls fully (kk in [0, K_CHUNK)). The runtime outer loop iterates
    // ceil_div(g.k, K_CHUNK) times (e.g. K=2880, K_CHUNK=64 -> 45 chunks).
    for (int k_chunk_start = 0; k_chunk_start < g.k; k_chunk_start += K_CHUNK) {
        {
            const int r_global = row_block_base + r_in_blk_a;
            const int k_global = k_chunk_start + kk_start_a;
            bf16x4 va{};
            if (r_global < g.M_total && k_global + VEC <= g.k) {
                const bf16* ap = &g.a[coord<>(r_global, k_global)];
                va = *reinterpret_cast<const bf16x4*>(ap);
            }
            *reinterpret_cast<bf16x4*>(&A_lds[r_in_blk_a * K_CHUNK + kk_start_a]) = va;
        }
        {
            const int c_global = col_block_base + c_in_blk_b;
            const int k_global = k_chunk_start + kk_start_b;
            bf16x4 vb{};
            if (c_global < g.n && k_global + VEC <= g.k) {
                const bf16* bp = &g.b[coord<>{0, group_idx, c_global, k_global}];
                vb = *reinterpret_cast<const bf16x4*>(bp);
            }
            *reinterpret_cast<bf16x4*>(&B_lds[c_in_blk_b * K_CHUNK + kk_start_b]) = vb;
        }
        __syncthreads();

        if (active_cell) {
            // [round-10] Vec4 LDS inner fma. K_CHUNK=64 -> 16 vec4 reads per
            // chunk per operand. 0.0f pad on the load above keeps the last
            // partial chunk (k_iters_v < K_CHUNK_VECS) numerically safe; the
            // ``break`` guard also short-circuits the unrolled loop on the
            // last chunk for K not a multiple of K_CHUNK.
            constexpr int FMA_VEC = 4;
            constexpr int K_CHUNK_VECS = K_CHUNK / FMA_VEC;
            static_assert(K_CHUNK % FMA_VEC == 0,
                "K_CHUNK must be vec4-aligned for inner fma");
            const int k_left = g.k - k_chunk_start;
            const int k_iters_v = (k_left < K_CHUNK)
                ? (k_left + FMA_VEC - 1) / FMA_VEC : K_CHUNK_VECS;
            #pragma unroll
            for (int kk_v = 0; kk_v < K_CHUNK_VECS; ++kk_v) {
                if (kk_v >= k_iters_v) break;
                bf16x4 a4 = *reinterpret_cast<const bf16x4*>(
                    &A_lds[rib * K_CHUNK + kk_v * FMA_VEC]);
                bf16x4 b4 = *reinterpret_cast<const bf16x4*>(
                    &B_lds[cib * K_CHUNK + kk_v * FMA_VEC]);
                acc += float(a4.lo.x) * float(b4.lo.x)
                     + float(a4.lo.y) * float(b4.lo.y)
                     + float(a4.hi.x) * float(b4.hi.x)
                     + float(a4.hi.y) * float(b4.hi.y);
            }
        }
        __syncthreads();  // before next chunk overwrites LDS
    }

    // Store ABSOLUTE value (main kernel did NOT write [fast_n, n) when
    // bpc = fast_n / BLOCK_SIZE; this kernel is the sole contributor).
    if (active_cell) {
        store_bf16_scalar(g.c, row, col, acc);
    }
}

template __global__ void grouped_ntail_kernel_lds<Layout::RCR, 64>(const grouped_layout_globals);

// Persistent kernel: grid_x = NUM_CUS. One block per CU; each block iterates
// many (group, tile) pairs in a single launch.
//
// The inner per-tile body is duplicated from ``gemm_kernel<L, KI_HINT>`` with
// minor coord adjustments. Code style follows the dense kernel for review
// parity; mechanical changes are flagged with ``[grouped]`` comments.
template<Layout L, int KI_HINT>
__global__ __launch_bounds__(NUM_THREADS, 1)
void grouped_kernel(const grouped_layout_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // [grouped] LDS cache for device group_offs (int32 view) + per-group
    // tile-cumsum. group_offs is read O(N_iter * G) times by the per-tile
    // inner scan; caching to LDS once at kernel entry replaces ~640 cycles
    // of HBM-cached ld/iter with ~320 cycles of LDS ld/iter, ~3-5% kernel
    // speedup on shapes with low ki / many tiles. Cap MAX_G_PLUS_1 = 65 to
    // cover G ≤ 64 (metric uses G ≤ 32). 8×65 = 520 bytes LDS, negligible.
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];
    __shared__ int s_cum_tiles[MAX_G_PLUS_1];
    __shared__ int s_total_tiles;

    // Same shared-memory tile types as gemm_kernel (no padded-b128 path here;
    // grouped path keeps RCR_PADDED_B128_MODE=0 to avoid pulling in the
    // experimental wiring while we're stabilising the persistent design).
    using ST_A = std::conditional_t<L == Layout::CRR,
        st_bf<K_STEP, HALF_BLOCK_SIZE, st_32x16_s>,
        st_bf<HALF_BLOCK_SIZE, K_STEP, st_16x32_s>>;
    using ST_B = std::conditional_t<L == Layout::RCR,
        st_bf<HALF_BLOCK_SIZE, K_STEP, st_16x32_s>,
        st_bf<K_STEP, HALF_BLOCK_SIZE, st_32x16_s>>;

    ST_A (&As)[2][2] = al.allocate<ST_A, 2, 2>();
    ST_B (&Bs)[2][2] = al.allocate<ST_B, 2, 2>();

    // Register tile types
    using A_reg_t = std::conditional_t<L == Layout::CRR,
        rt_bf<K_STEP, HALF_REG_BLOCK_M, col_l, rt_32x16_s>,
        rt_bf<HALF_REG_BLOCK_M, K_STEP, row_l, rt_16x32_s>>;
    using B_reg_t = std::conditional_t<L == Layout::RCR,
        rt_bf<HALF_REG_BLOCK_N, K_STEP, row_l, rt_16x32_s>,
        rt_bf<K_STEP, HALF_REG_BLOCK_N, col_l, rt_32x16_s>>;

    // C accumulators stay in this outer scope so they survive the helper
    // call and are still live during the per-tile store epilog inside the
    // persistent loop below.
    rt_fl<HALF_REG_BLOCK_M, HALF_REG_BLOCK_N, col_l, rt_16x16_s> C_accum[2][2];

    // [grouped] Persistent: chiplet-swizzle pid against full grid (NUM_CUS).
    int pid = chiplet_transform_chunked(blockIdx.x, NUM_CUS, g.num_xcds, 64);

    const int num_pid_n = g.bpc;

    // [grouped] Cooperative init of the LDS group-metadata caches. Single
    // thread does the O(G) scan once; then everyone uses s_offs / s_cum_tiles.
    // group_offs is in element units; per-group tile count = (M_g/256)*bpc.
    if (threadIdx.x == 0) {
        int prev = static_cast<int>(g.group_offs[0]);
        s_offs[0] = prev;
        s_cum_tiles[0] = 0;
        int t = 0;
        #pragma unroll 1
        for (int gi = 0; gi < g.G; ++gi) {
            const int next = static_cast<int>(g.group_offs[gi + 1]);
            s_offs[gi + 1] = next;
            t += ((next - prev) / BLOCK_SIZE) * num_pid_n;
            s_cum_tiles[gi + 1] = t;
            prev = next;
        }
        s_total_tiles = t;
    }
    __syncthreads();
    const int total_tiles = s_total_tiles;

    // [grouped] SRD setup. Bounds span the FULL A and C tensors (across all
    // groups) so the same A SRD is valid across persistent iterations.
    // The B SRD is computed PER PERSISTENT ITERATION below — its bound is
    // ``(group_idx + 1) * <inner_rows> * sizeof(bf16)`` so a partial col-tile
    // (from ``g.bpc = ceil_div(g.n, BLOCK_SIZE)`` when N is misaligned and K
    // is aligned) cannot wrap into the NEXT group's region. With a global
    // SRD bound the OOB row would still land inside ``[0, G*N*K)`` and read
    // garbage; clipping to the current group's slice forces ``buffer_load_lds``
    // to clamp those lanes to 0 — the column-masked C store below then
    // drops the OOB cells from the write-back.
    const bf16* a_base = (bf16*)&g.a[{0, 0, 0, 0}];
    const bf16* b_base = (bf16*)&g.b[{0, 0, 0, 0}];
    const int a_row_stride = g.a.template stride<2>() * sizeof(bf16);
    const int b_row_stride = g.b.template stride<2>() * sizeof(bf16);
    // For "normal" A layout (M×K): A_total_rows = M_total. For CRR A (K×M_total):
    // A_total_rows = K (M dimension lives on the col axis).
    const int a_total_rows = (L == Layout::CRR) ? g.k : g.M_total;
    i32x4 a_srsrc_base = make_srsrc(a_base, a_total_rows * a_row_stride, a_row_stride);
    // ``b_inner_rows`` is the number of rows per group along B's row axis:
    //   * RCR     : B is [G, N, K] — row axis is N → ``g.n``.
    //   * RRR/CRR : B is [G, K, N] — row axis is K → ``g.k``.
    // We multiply by ``(group_idx + 1)`` per-iteration to get the per-group
    // SRD upper bound.
    const int b_inner_rows = (L == Layout::RCR) ? g.n : g.k;

    const int wid = warpid() % NUM_WARPS;
    constexpr int elem_per_warp = (16 / sizeof(bf16)) * kittens::WARP_THREADS;
    constexpr uint32_t A_TILE_LDS = sizeof(ST_A);
    constexpr uint32_t B_TILE_LDS = sizeof(ST_B);
    uint32_t a_lds = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&As[0][0].data[0]) + wid * elem_per_warp * sizeof(bf16)));
    uint32_t b_lds = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&Bs[0][0].data[0]) + wid * elem_per_warp * sizeof(bf16)));
    const uint32_t a_lds_00 = a_lds;
    const uint32_t a_lds_01 = a_lds + A_TILE_LDS;
    const uint32_t a_lds_10 = a_lds + 2 * A_TILE_LDS;
    const uint32_t a_lds_11 = a_lds + 3 * A_TILE_LDS;
    const uint32_t b_lds_00 = b_lds;
    const uint32_t b_lds_01 = b_lds + B_TILE_LDS;
    const uint32_t b_lds_10 = b_lds + 2 * B_TILE_LDS;
    const uint32_t b_lds_11 = b_lds + 3 * B_TILE_LDS;

    using T = typename st_bf<BLOCK_SIZE, K_STEP, st_32x16_s>::dtype;
    constexpr int bytes_per_thread = st_32x16_s::template bytes_per_thread<T>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile = BLOCK_SIZE * K_STEP * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile/2];
    uint32_t swizzled_offsets_B[memcpy_per_tile/2];
    G::prefill_swizzled_offsets(As[0][0], g.a, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[0][0], g.b, swizzled_offsets_B);

    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;

    // Per-group bounded B SRD cache. The persistent loop streams many
    // (group, tile) pairs through this CU; ``group_idx`` typically stays
    // constant for a run of 8-32 tiles before advancing. We keep the
    // most recently constructed SRD and only rebuild on group change,
    // which lifts the 4-SGPR ``make_srsrc`` cost out of the inner store
    // path on aligned DeepSeek shapes.
    //
    // Round-11: B SRD must NOT enable the cache-swizzle stride field —
    // when ``b_row_stride`` is not a power-of-2 (gpt_oss K=2880 → stride
    // 5760 bytes, K=7168 → stride 14336 bytes), the buffer-LDS path's
    // OOB-clamp behaviour is undefined for the partial last col-tile
    // (rows >= b_inner_rows but < range_bytes / row_stride). Empirically
    // this leads to a memory access fault on K=2880 + N=2880 when main
    // sweeps cols [0, ceil_div(g.n, BLOCK_SIZE)*BLOCK_SIZE). Passing
    // ``row_stride=0`` to ``make_srsrc`` keeps the linear range-bytes
    // bound check (which clamps OOB lanes to 0 reliably) without the
    // swizzle. The cost is the lost cache-swizzle benefit on the
    // aligned interior tiles — we accept that on the grouped path
    // because (a) DSV3 N=4096/7168-aligned still hits ~1.15 vs Triton,
    // and (b) without the disable, gpt_oss N=K=2880 crashes.
    int last_group_idx = -1;
    i32x4 b_srsrc_curr = make_srsrc(b_base, b_inner_rows * b_row_stride, /*row_stride_bytes=*/0);

    // [grouped] Persistent outer loop: stream (group, tile) pairs through this CU.
    for (int gt = pid; gt < total_tiles; gt += NUM_CUS) {

        // [grouped] O(G) linear scan over LDS-cached cumsum to map gt →
        // (group_idx, local_tile, m_start_g, M_g). LDS reads (~5 cyc) replace
        // HBM-cached g.group_offs ld pairs (~10 cyc each), and the per-iter
        // recompute of `(M_g/BLOCK_SIZE) * num_pid_n` is hoisted to the
        // kernel-entry init.
        int group_idx = 0;
        int tile_start = 0;
        #pragma unroll 1
        for (int gi = 0; gi < g.G; ++gi) {
            const int new_cum = s_cum_tiles[gi + 1];
            if (gt >= new_cum) {
                group_idx = gi + 1;
                tile_start = new_cum;
            }
        }
        const int local_tile = gt - tile_start;
        const int m_start_g = s_offs[group_idx];
        const int M_g = s_offs[group_idx + 1] - m_start_g;
        const int bpr_g = M_g / BLOCK_SIZE;

        // Group-by-M / group-by-N swizzle (matches dense kernel's tile mapping).
        int pid_m, pid_n;
        if (g.bpc > bpr_g) {
            const int WGN = g.group_m;
            const int num_wgid_in_group = bpr_g * WGN;
            int group_id = local_tile / num_wgid_in_group;
            int first_pid_n = group_id * WGN;
            int group_size_n = min(num_pid_n - first_pid_n, WGN);
            if (group_size_n <= 0) continue;
            pid_n = first_pid_n + ((local_tile % num_wgid_in_group) % group_size_n);
            pid_m = (local_tile % num_wgid_in_group) / group_size_n;
        } else {
            const int WGM = g.group_m;
            const int num_wgid_in_group = WGM * num_pid_n;
            int group_id = local_tile / num_wgid_in_group;
            int first_pid_m = group_id * WGM;
            int group_size_m = min(bpr_g - first_pid_m, WGM);
            if (group_size_m <= 0) continue;
            pid_m = first_pid_m + ((local_tile % num_wgid_in_group) % group_size_m);
            pid_n = (local_tile % num_wgid_in_group) / group_size_m;
        }
        if (pid_m >= bpr_g || pid_n >= num_pid_n) continue;
        const int row = pid_m;
        const int col = pid_n;

        // [grouped] m_start_g is always BLOCK_SIZE-aligned (callers guarantee
        // group_lens are multiples of 256), so these divisions are exact.
        //
        // Unit derivations (verified against gemm_kernel store coords + the
        // ``unit_coord<row_axis=2, col_axis=3>`` definition in
        // ``include/types/global/util.cuh:51``):
        //
        //   * ST_A non-CRR : st_bf<HALF_BLOCK_SIZE=128, K_STEP=64, st_16x32_s>
        //                    → BASE::rows = 128 → r-coord unit = 128 elements
        //                    → m_subtile_A = m_start_g / HALF_BLOCK_SIZE
        //
        //   * ST_A CRR     : st_bf<K_STEP=64, HALF_BLOCK_SIZE=128, st_32x16_s>
        //                    A is laid out [K, M_total]; m sits on the COLUMN
        //                    axis. unit_coord uses BASE::cols = 128, so the
        //                    same divisor (128) maps element-row offset to
        //                    coord-col offset. (m_subtile_A reused as a c-coord.)
        //
        //   * C (RT store) : rt_fl<HALF_REG_BLOCK_M=64, HALF_REG_BLOCK_N=32, ...>
        //                    BASE::rows = 64 (RT::rows = full register-tile
        //                    height, NOT base_tile_rows=16). r-coord unit = 64
        //                    elements ⇒ m_subtile_C = m_start_g / HALF_REG_BLOCK_M.
        //                    The previous ``/16`` (assuming the 16-row base
        //                    tile was the unit) was off by 4× and produced
        //                    out-of-bounds writes for B≥2 (SNR=3 dB on B=2 /
        //                    GPU memory fault on B=2 M_g=512; see
        //                    ``_PERSISTENT_GROUPED_WIP_NOTES.md``).
        const int m_subtile_A = m_start_g / HALF_BLOCK_SIZE;
        const int m_subtile_C = m_start_g / HALF_REG_BLOCK_M;

        // Reset accumulators for this tile.
        zero(C_accum[0][0]); zero(C_accum[0][1]);
        zero(C_accum[1][0]); zero(C_accum[1][1]);

        // Per-group bounded B SRD: limits the buffer-flat range to the
        // current group's slice ``[group_idx * <inner_rows>, (group_idx+1)
        // * <inner_rows>)``. ``buffer_load_lds`` then SRD-clamps any OOB
        // row (from a partial last col-tile when ``g.bpc =
        // ceil_div(g.n, BLOCK_SIZE)``) to 0 instead of wrapping into the
        // NEXT group. Rebuilt only on ``group_idx`` transitions to avoid
        // 4-SGPR-op overhead on every persistent iteration.
        if (group_idx != last_group_idx) {
            const int b_grp_total_rows = (group_idx + 1) * b_inner_rows;
            b_srsrc_curr = make_srsrc(
                b_base, b_grp_total_rows * b_row_stride,
                /*row_stride_bytes=*/0);
            last_group_idx = group_idx;
        }

        // Phase 2: shared device function above runs the same prologue +
        // main_loop + epilog 1/2 as the dense kernel. Grouped passes
        // m_subtile_A (A row shift in HALF_BLOCK_SIZE units) and group_idx
        // (B depth axis) so the helper's a_coord / b_coord land on the
        // correct (group, M-slice) sub-tensor.
        device_gemm_tile_body<L, KI_HINT, ST_A, ST_B, A_reg_t, B_reg_t>(
            g.a, g.b,
            m_subtile_A, group_idx, /*k_offset_tiles=*/0,
            As, Bs,
            swizzled_offsets_A, swizzled_offsets_B,
            a_srsrc_base, b_srsrc_curr,
            a_base, b_base,
            a_lds_00, a_lds_01, a_lds_10, a_lds_11,
            b_lds_00, b_lds_01, b_lds_10, b_lds_11,
            row, col, warp_row, warp_col,
            g.ki,
            C_accum);

        if (warp_row == 0) { __builtin_amdgcn_s_barrier(); }

        // [grouped] Store with C row shifted by m_subtile_C. When
        // ``g.bpc = ceil_div(g.n, BLOCK_SIZE)`` (round 5 path, K aligned)
        // the last col-tile may straddle ``[fast_n, n)``; the masked store
        // drops OOB columns. The launch-uniform branch on
        // ``g.n % BLOCK_SIZE == 0`` keeps the aligned path on the raw
        // ``store(...)`` call (zero compare/branch in the store inner
        // loop) — DeepSeek-V3 N=4096/7168 hit this and pay no masked-store
        // cost. Misaligned shapes (gpt_oss N=2880/5760) take the
        // column-masked branch.
        const int r0 = m_subtile_C + (row * 2) * WARPS_M + warp_row;
        const int r1 = m_subtile_C + (row * 2) * WARPS_M + WARPS_M + warp_row;
        const int c0 = col * 2 * WARPS_N + warp_col;
        const int c1 = col * 2 * WARPS_N + WARPS_N + warp_col;
        if ((g.n % BLOCK_SIZE) == 0) {
            store(g.c, C_accum[0][0], {0, 0, r0, c0});
            store(g.c, C_accum[0][1], {0, 0, r0, c1});
            store(g.c, C_accum[1][0], {0, 0, r1, c0});
            store(g.c, C_accum[1][1], {0, 0, r1, c1});
        } else {
            store_c_tile_n_masked(g.c, C_accum[0][0], r0, c0, g.n);
            store_c_tile_n_masked(g.c, C_accum[0][1], r0, c1, g.n);
            store_c_tile_n_masked(g.c, C_accum[1][0], r1, c0, g.n);
            store_c_tile_n_masked(g.c, C_accum[1][1], r1, c1, g.n);
        }

        // [grouped] Drain in-flight ops before the next persistent iteration so
        // the next tile's prologue starts from a clean state.
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }
}

// Explicit instantiations — same KI specialization set as gemm_kernel.
template __global__ void grouped_kernel<Layout::RCR, 0>(const grouped_layout_globals);
template __global__ void grouped_kernel<Layout::RRR, 0>(const grouped_layout_globals);
template __global__ void grouped_kernel<Layout::CRR, 0>(const grouped_layout_globals);
#define INSTANTIATE_K_GRP(KI) \
    template __global__ void grouped_kernel<Layout::RCR, KI>(const grouped_layout_globals); \
    template __global__ void grouped_kernel<Layout::RRR, KI>(const grouped_layout_globals); \
    template __global__ void grouped_kernel<Layout::CRR, KI>(const grouped_layout_globals)
INSTANTIATE_K_GRP(56);
INSTANTIATE_K_GRP(64);
INSTANTIATE_K_GRP(112);
INSTANTIATE_K_GRP(128);
INSTANTIATE_K_GRP(172);
INSTANTIATE_K_GRP(224);
INSTANTIATE_K_GRP(256);
INSTANTIATE_K_GRP(296);
INSTANTIATE_K_GRP(448);
INSTANTIATE_K_GRP(462);
INSTANTIATE_K_GRP(832);
#undef INSTANTIATE_K_GRP

template<Layout L, int KI>
static inline void launch_one_grouped(grouped_layout_globals& g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    static bool attr_set = false;
    if (!attr_set) {
        hipFuncSetAttribute((void*)grouped_kernel<L, KI>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
        attr_set = true;
    }
    grouped_kernel<L, KI><<<dim3(NUM_CUS), g.block(), mem_size, g.stream>>>(g);
}

template<Layout L>
void dispatch_grouped(grouped_layout_globals g) {
    g.n = static_cast<int>(g.c.cols());
    g.M_total = static_cast<int>(g.c.rows());
    if constexpr (L == Layout::CRR) g.k = static_cast<int>(g.a.rows());
    else g.k = static_cast<int>(g.a.cols());

    // Phase 3: native non-aligned N/K. Per-group M tail (M_g % BLOCK_SIZE != 0)
    // is handled by `grouped_tail_kernel`.
    //
    // Phase 4 (round 5): main kernel covers the entire N range via column-
    // masked C store + per-group bounded B SRD. Previously the docstring
    // here warned that bpc = ceil_div was unsafe for grouped because a
    // partial col-tile (spatial >= N) would issue B loads at
    // coord{0, group_idx, spatial_OOB, k_tile} where the buffer-flat
    // address wraps into the NEXT group's region (still within the
    // shared global SRD bound, no SRD clamp-to-zero) and segfaults
    // through the swizzle/cache path.
    //
    // Round-5 fix: in ``grouped_kernel`` we now reconstruct the B SRD
    // per-iteration with a bound of ``(group_idx + 1) * <inner-rows> *
    // sizeof(bf16)`` so the SOFF for any OOB-row tile lands beyond
    // the SRD limit and the hardware clamps the load to 0.
    //
    // Round-11 (auto_optimize): RCR uses ``ceil_div(g.n, BLOCK_SIZE)``
    // unconditionally. The previous restriction (only when K is also
    // aligned) was lifted by (a) extending the LDS K-tail kernel grid
    // to ``ceil_div(g.n, TBN)`` (covers the partial last col-tile too),
    // and (b) DISABLING the cache-swizzle stride field on the per-group
    // B SRD inside ``grouped_kernel`` — non-power-of-2 K strides
    // (gpt_oss K=2880 → 5760 bytes) made the swizzled OOB clamp
    // unreliable, leading to a GPU memory fault for the OOB partial
    // col-tile rows. The unswizzled SRD's linear range-bytes bound
    // clamps reliably; the cost is lost cache-swizzle on the
    // interior tiles, but DSV3 still hits ~1.15 vs Triton on the
    // K-power-of-2 path. The N-tail LDS kernel is no longer launched
    // on RCR because the main kernel + LDS K-tail combination now
    // writes every cell in [0, g.n) × [0, g.k) natively. This removes
    // the entire ~32% wall-time spent on grouped_ntail_kernel_lds for
    // gpt_oss N=2880/5760, K=2880.
    g.fast_n = (g.n / BLOCK_SIZE) * BLOCK_SIZE;
    g.fast_k = (g.k / K_TWO_TILE) * K_TWO_TILE;
    // ceil_div N coverage is RCR-only:
    //   * RCR     : B is [G, N, K] — row axis is N, ``b_row_stride`` =
    //               K bytes, so an OOB N row's buffer-flat offset
    //               exceeds the per-group SRD bound and gets clamped
    //               to 0. The masked C store then drops the OOB cells.
    //               Round 5 path: the K-tail correction is added on top
    //               of the masked main kernel write by ``grouped_tail_kernel``
    //               for ALL cells col in [0, g.n) — including the partial
    //               last col-tile — because main now wrote the
    //               [0, fast_k) partial reduction there.
    //   * RRR/CRR : B is [G, K, N] — N lives on the COLUMN axis, so
    //               an OOB N column lands at byte-offset
    //               ``row*N_stride + col_oob``, which is still inside
    //               the per-group SRD (just wraps to the next K row's
    //               valid columns) — no clamp triggers, garbage data
    //               feeds the MMA. Until we add a column-mask path on
    //               the B load itself (Phase 6+), RRR/CRR keep the
    //               legacy ``bpc = fast_n / BLOCK_SIZE`` and N-tail
    //               flows through ``grouped_tail_kernel``.
    // Round-11: ceil_div N coverage is enabled unconditionally for RCR
    // (no K-alignment gate). The unswizzled per-group B SRD inside
    // ``grouped_kernel`` clamps OOB rows reliably, and the LDS K-tail
    // kernel grid ``ceil_div(g.n, TBN)`` covers the partial last
    // col-tile of the K-tail correction.
    if constexpr (L == Layout::RCR) {
        g.bpc = kittens::ceil_div(g.n, BLOCK_SIZE);
    } else {
        g.bpc = g.fast_n / BLOCK_SIZE;
    }
    g.ki     = g.fast_k / K_STEP;

    if (g.bpc > 0 && g.ki >= 2) {
        switch (g.ki) {
            case 56:  launch_one_grouped<L, 56> (g); break;
            case 64:  launch_one_grouped<L, 64> (g); break;
            case 112: launch_one_grouped<L, 112>(g); break;
            case 128: launch_one_grouped<L, 128>(g); break;
            case 172: launch_one_grouped<L, 172>(g); break;
            case 224: launch_one_grouped<L, 224>(g); break;
            case 256: launch_one_grouped<L, 256>(g); break;
            case 296: launch_one_grouped<L, 296>(g); break;
            case 448: launch_one_grouped<L, 448>(g); break;
            case 462: launch_one_grouped<L, 462>(g); break;
            case 832: launch_one_grouped<L, 832>(g); break;
            default:  launch_one_grouped<L, 0>  (g); break;
        }
    } else {
        // Main kernel can't run (N < BLOCK_SIZE or K < K_TWO_TILE). Reset
        // fast_* so the tail kernel treats every cell as "kernel never
        // ran here" and computes the full output from scratch.
        g.fast_n = 0;
        g.fast_k = 0;
        g.bpc    = 0;
        g.ki     = 0;
    }

    // Round-11: main kernel always covers [0, g.n) × [0, g.fast_k) on
    // RCR. The LDS K-tail kernel (when launched) covers the K-tail
    // correction for ALL cols [0, g.n), including the partial last
    // col-tile via a wider grid. The dedicated LDS N-tail kernel is
    // therefore no longer launched on RCR — main + LDS K-tail together
    // cover every cell.
    // RRR/CRR can't activate ceil_div N coverage (see dispatch comment
    // above) — N-tail still runs through the scalar tail kernel for
    // those layouts.
    constexpr bool layout_supports_main_n = (L == Layout::RCR);
    const bool main_covers_n = layout_supports_main_n;
    const bool need_tail_run =
        (g.fast_k != g.k) ||
        (!main_covers_n && g.fast_n != g.n);
    if (need_tail_run) {
        // Fast path: LDS-staged interior K-tail correction (round-9). Runs
        // when the K-tail size matches a templated specialisation AND each
        // (TAIL_BLOCK_M × TAIL_BLOCK_N) tail block sits inside a single
        // group (uniform M with M_g a TAIL_BLOCK_M multiple — true for
        // metric uniform-M=2048/4096 grouped shapes). Cuts interior K-tail
        // from ~4 TF (scalar HBM-bound, one A row + one B col fetched per
        // thread) to ~50-80 TF (cooperative LDS staging, 16× reuse).
        const int K_rem = g.k - g.fast_k;
        const bool lds_k_tail_safe = (g.m_per_group >= TAIL_BLOCK_M) &&
                                     ((g.m_per_group % TAIL_BLOCK_M) == 0);
        if constexpr (L == Layout::RCR) {
            // Round-11: LDS K-tail grid covers ALL cols [0, g.n), not
            // just [0, fast_n). The partial last col-tile of the K-tail
            // RMW is now handled here too (kernel skips ``col >= g.n``);
            // the dedicated N-tail kernel below is dropped because main
            // already wrote the partial col-tile via
            // ``store_c_tile_n_masked`` and this kernel adds the K-tail
            // correction on top.
            if (K_rem == 64 && lds_k_tail_safe) {
                // Round-21: prefer 32x32x16 mfma kernel when m_per_group is
                // 32-aligned. Same 100 % MFMA util as round-19 16x16x32 but
                // 4× larger cell tile (32x32 → 1024 cells/block) → grid
                // shrinks 4× and per-cell mem traffic halves (8 byte/cell
                // vs 16 byte/cell), net throughput on the K-tail dominated
                // gpt_oss path. Mirror of FP8 round-20. Falls back to
                // round-19 16x16x32 path for 16-aligned but not 32-aligned
                // (none in metric — gpt_oss M_per ∈ {2048, 4096} both
                // satisfy 32-alignment).
                constexpr int TBM_32x32 = 32;
                const bool mfma32_safe = (g.m_per_group >= TBM_32x32) &&
                                         ((g.m_per_group % TBM_32x32) == 0);
                if (mfma32_safe) {
                    dim3 mfma_block(64);
                    dim3 mfma_grid(
                        kittens::ceil_div(g.n, TBM_32x32),
                        kittens::ceil_div(g.M_total, TBM_32x32)
                    );
                    grouped_ktail_kernel_mfma32x32<Layout::RCR, 64>
                        <<<mfma_grid, mfma_block, 0, g.stream>>>(g);
                } else {
                    // Round-19: 16x16x32 mfma kernel (still 100 % util).
                    dim3 mfma_block(64);
                    dim3 mfma_grid(
                        kittens::ceil_div(g.n, TAIL_BLOCK_N),
                        kittens::ceil_div(g.M_total, TAIL_BLOCK_M)
                    );
                    grouped_ktail_kernel_mfma<Layout::RCR, 64>
                        <<<mfma_grid, mfma_block, 0, g.stream>>>(g);
                }
            }
            // Round-11: LDS N-tail kernel removed for RCR. Main kernel's
            // ``bpc = ceil_div(g.n, BLOCK_SIZE)`` + per-group bounded B
            // SRD (unswizzled, see grouped_kernel comment) + column-
            // masked C store handles the partial last col-tile natively;
            // the LDS K-tail above adds the K-tail correction for all
            // cols in [0, g.n) including that partial col-tile.
            // Profiling (rocprof on gpt_oss-Down B=4-M2048) showed the
            // dropped N-tail kernel was 32% of total wall-time; this
            // change removes that whole kernel from the launch graph.
        }

        // [round-10] Skip scalar tail launch when LDS K-tail covers
        // every cell. Round-11: ``g.fast_n > 0`` is dropped because
        // main now ALWAYS covers [0, g.n) for RCR — the LDS K-tail
        // grid is ``ceil_div(g.n, TBN)``, never empty.
        const bool lds_handles_all =
            (L == Layout::RCR) &&
            (K_rem == 64) &&
            lds_k_tail_safe;
        if (!lds_handles_all) {
            dim3 tail_block(TAIL_BLOCK_N, TAIL_BLOCK_M);
            dim3 tail_grid(
                kittens::ceil_div(g.n, TAIL_BLOCK_N),
                kittens::ceil_div(g.M_total, TAIL_BLOCK_M)
            );
            grouped_tail_kernel<L><<<tail_grid, tail_block, 0, g.stream>>>(g);
        }
    }
}

static void grouped_dispatch(pybind11::object a, pybind11::object b, pybind11::object c,
                             pybind11::object group_offs, int gm, int num_xcds,
                             int m_per_group, const char* layout_name) {
    auto group_offs_ptr = group_offs.attr("data_ptr")().cast<uintptr_t>();
    int G = group_offs.attr("numel")().cast<int>() - 1;

    grouped_layout_globals g{
        py::from_object<_gl>::make(a),
        py::from_object<_gl>::make(b),
        py::from_object<_gl>::make(c),
        reinterpret_cast<const int64_t*>(group_offs_ptr),
        {},
        G, 0, 0, 0, 0, gm, num_xcds, 0,
        0, 0, // fast_n, fast_k — populated inside dispatch_grouped<L>.
        m_per_group,
    };

    if (layout_name[0] == 'r' && layout_name[1] == 'c') dispatch_grouped<Layout::RCR>(g);
    else if (layout_name[0] == 'r' && layout_name[1] == 'r') dispatch_grouped<Layout::RRR>(g);
    else dispatch_grouped<Layout::CRR>(g);
}

// =============================================================================
// Persistent CPU-sync-free grouped variable-K (CRR / dB) kernel.
//
// Math (per group ``g``):
//
//     C[g, n, k] = sum_{m in [offs[g], offs[g+1]) }
//                      A[m, n] * B[m, k]
//
// where:
//   * A is the upstream gradient ``grad_out`` (2D ``[M_total, n]``) and is
//     reinterpreted as CRR-A ``[K=M_total, M=n]``.
//   * B is the activation tensor ``x``      (2D ``[M_total, k]``) and is
//     reinterpreted as CRR-B ``[K=M_total, N=k]``.
//   * C is the weight gradient ``grad_b``   (3D ``[G, n, k]``) — group_idx
//     is the depth axis, n is the kernel's M-output, k is the kernel's
//     N-output.
//
// Replaces the 32× per-group ``dense_run`` loop in
// ``GroupedGEMMVariableKHipKittenBackend.execute`` (Primus side); the
// per-group launch was the dominant bottleneck for backward dB
// (the breakdown probe — gpt_oss-Down B=4 M=2048 — showed dB took
//  92% of the total backward time and only 56 TF, vs ~650 TF for dA).
//
// Layout differences from the forward grouped kernel:
//   * n and k are *group-uniform* (output [G, n, k]); the variable axis
//     is the K-reduction dim ``M_g = offs[g+1] - offs[g]``.
//   * Per-group tile count is therefore *uniform*: ``bpr * bpc``. No
//     LDS-cached per-group cumsum is needed — group_idx is simply
//     ``gt / (bpr * bpc)`` and ``ki_g = M_g / K_STEP`` only matters
//     for the dynamic K-loop bound.
//   * The K-axis SHIFT per group is delivered through the new
//     ``k_offset_tiles = m_start_g / K_STEP`` parameter on
//     ``device_gemm_tile_body`` (forward grouped passes 0 there; the
//     constant folds away in dense and forward grouped codegen).
//
// CRR-only for now — the dB autograd path always wants CRR-trans_c. The
// kernel falls back to ``Layout::CRR`` ST/RT types directly to avoid the
// dispatcher boilerplate the forward grouped kernel needs for L-templating.
// =============================================================================
struct grouped_var_k_layout_globals {
    _gl a;                       // [1, 1, M_total, n] — grad_out
    _gl b;                       // [1, 1, M_total, k] — x
    _gl c;                       // [1, G, n, k]       — grad_b
    const int64_t* group_offs;   // [G+1] int64 device prefix-sum of M_g
    hipStream_t stream;
    int G;
    int M_total;
    int n;          // kernel M-output dim (= N_fwd)
    int k;          // kernel N-output dim (= K_fwd)
    int group_m;
    int num_xcds;
    int bpr;        // n / BLOCK_SIZE — output row tiles
    int bpc;        // k / BLOCK_SIZE — output col tiles
    int ki_max;     // upper bound on per-group ki (for KI_HINT specialization)
    // Aligned-region dims. v0 only: aligned-only. fast_{n,k} == n,k and
    // m_aligned == M_total are required by ``can_handle`` on the Python
    // side; the dispatcher otherwise falls back to the per-group loop.
    int fast_n, fast_k;
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

template<int KI_HINT>
__global__ __launch_bounds__(NUM_THREADS, 1)
void grouped_var_k_kernel(const grouped_var_k_layout_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // CRR-only ST/RT types (mirror the L=Layout::CRR branch of the
    // dense / forward-grouped kernel).
    using ST_A = st_bf<K_STEP, HALF_BLOCK_SIZE, st_32x16_s>;
    using ST_B = st_bf<K_STEP, HALF_BLOCK_SIZE, st_32x16_s>;
    ST_A (&As)[2][2] = al.allocate<ST_A, 2, 2>();
    ST_B (&Bs)[2][2] = al.allocate<ST_B, 2, 2>();
    using A_reg_t = rt_bf<K_STEP, HALF_REG_BLOCK_M, col_l, rt_32x16_s>;
    using B_reg_t = rt_bf<K_STEP, HALF_REG_BLOCK_N, col_l, rt_32x16_s>;
    rt_fl<HALF_REG_BLOCK_M, HALF_REG_BLOCK_N, col_l, rt_16x16_s> C_accum[2][2];

    // [grouped-var-k] LDS-cached group_offs. Read O(N_iter * G) times by
    // the per-tile coord scan; caching to LDS once at kernel entry
    // mirrors the forward grouped kernel.
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];
    if (threadIdx.x == 0) {
        #pragma unroll 1
        for (int gi = 0; gi <= g.G; ++gi) {
            s_offs[gi] = static_cast<int>(g.group_offs[gi]);
        }
    }
    __syncthreads();

    // Persistent: chiplet-swizzle pid against full grid (NUM_CUS).
    int pid = chiplet_transform_chunked(blockIdx.x, NUM_CUS, g.num_xcds, 64);

    // Per-group tile count uniform: bpr * bpc.
    const int tiles_per_group = g.bpr * g.bpc;
    const int total_tiles = g.G * tiles_per_group;
    const int num_pid_m = g.bpr;
    const int num_pid_n = g.bpc;

    // Full-tensor SRDs (A and B span [M_total, *]; the K-axis shift per
    // group is delivered to ``device_gemm_tile_body`` via k_offset_tiles).
    const bf16* a_base = (bf16*)&g.a[{0, 0, 0, 0}];
    const bf16* b_base = (bf16*)&g.b[{0, 0, 0, 0}];
    const int a_row_stride = g.a.template stride<2>() * sizeof(bf16);  // = n * sizeof(bf16)
    const int b_row_stride = g.b.template stride<2>() * sizeof(bf16);  // = k * sizeof(bf16)
    const int a_total_rows = g.M_total;
    const int b_total_rows = g.M_total;
    i32x4 a_srsrc_base = make_srsrc(a_base, a_total_rows * a_row_stride, a_row_stride);
    i32x4 b_srsrc_base = make_srsrc(b_base, b_total_rows * b_row_stride, b_row_stride);

    // LDS double-buffered per-warp slots (mirror grouped_kernel).
    const int wid = warpid() % NUM_WARPS;
    constexpr int elem_per_warp = (16 / sizeof(bf16)) * kittens::WARP_THREADS;
    constexpr uint32_t A_TILE_LDS = sizeof(ST_A);
    constexpr uint32_t B_TILE_LDS = sizeof(ST_B);
    uint32_t a_lds = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&As[0][0].data[0]) + wid * elem_per_warp * sizeof(bf16)));
    uint32_t b_lds = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&Bs[0][0].data[0]) + wid * elem_per_warp * sizeof(bf16)));
    const uint32_t a_lds_00 = a_lds;
    const uint32_t a_lds_01 = a_lds + A_TILE_LDS;
    const uint32_t a_lds_10 = a_lds + 2 * A_TILE_LDS;
    const uint32_t a_lds_11 = a_lds + 3 * A_TILE_LDS;
    const uint32_t b_lds_00 = b_lds;
    const uint32_t b_lds_01 = b_lds + B_TILE_LDS;
    const uint32_t b_lds_10 = b_lds + 2 * B_TILE_LDS;
    const uint32_t b_lds_11 = b_lds + 3 * B_TILE_LDS;

    using T = typename st_bf<BLOCK_SIZE, K_STEP, st_32x16_s>::dtype;
    constexpr int bytes_per_thread = st_32x16_s::template bytes_per_thread<T>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile = BLOCK_SIZE * K_STEP * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile/2];
    uint32_t swizzled_offsets_B[memcpy_per_tile/2];
    G::prefill_swizzled_offsets(As[0][0], g.a, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[0][0], g.b, swizzled_offsets_B);

    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;

    // [grouped-var-k] Persistent outer loop: stream (group, tile) pairs through this CU.
    for (int gt = pid; gt < total_tiles; gt += NUM_CUS) {
        const int group_idx = gt / tiles_per_group;
        const int local_tile = gt - group_idx * tiles_per_group;

        const int m_start_g = s_offs[group_idx];
        const int M_g = s_offs[group_idx + 1] - m_start_g;
        const int ki_g = M_g / K_STEP;
        // Need at least 2 K-tiles for the prologue + epilog schedule; same
        // constraint as dense / forward grouped (caller enforces M_g >= 128).
        if (ki_g < 2) continue;

        // Within-group tile mapping (mirror dense gemm_compute_block_coords).
        // Both axes (m=output rows, n=output cols) are uniform; same dual
        // tall-N / tall-M swizzle as dense.
        int pid_m, pid_n;
        if (num_pid_n > num_pid_m) {
            const int WGN = g.group_m;
            const int num_wgid_in_group = num_pid_m * WGN;
            int group_id = local_tile / num_wgid_in_group;
            int first_pid_n = group_id * WGN;
            int group_size_n = min(num_pid_n - first_pid_n, WGN);
            if (group_size_n <= 0) continue;
            pid_n = first_pid_n + ((local_tile % num_wgid_in_group) % group_size_n);
            pid_m = (local_tile % num_wgid_in_group) / group_size_n;
        } else {
            const int WGM = g.group_m;
            const int num_wgid_in_group = WGM * num_pid_n;
            int group_id = local_tile / num_wgid_in_group;
            int first_pid_m = group_id * WGM;
            int group_size_m = min(num_pid_m - first_pid_m, WGM);
            if (group_size_m <= 0) continue;
            pid_m = first_pid_m + ((local_tile % num_wgid_in_group) % group_size_m);
            pid_n = (local_tile % num_wgid_in_group) / group_size_m;
        }
        if (pid_m >= num_pid_m || pid_n >= num_pid_n) continue;
        const int row = pid_m;
        const int col = pid_n;

        // K-axis offset in K_STEP units. m_start_g is BLK-aligned (256),
        // K_STEP=64 divides BLK so this is exact.
        const int k_offset_tiles = m_start_g / K_STEP;

        // Reset accumulators.
        zero(C_accum[0][0]); zero(C_accum[0][1]);
        zero(C_accum[1][0]); zero(C_accum[1][1]);

        // Reuse the shared GEMM body with CRR layout. m_subtile_A=0
        // (kernel n is group-uniform; row is the M-output tile) and
        // group_idx=0 (B is 2D, no depth axis). The variable-K piece is
        // delivered through k_offset_tiles which adds m_start_g/K_STEP
        // to every a_coord/b_coord K-axis lookup.
        device_gemm_tile_body<Layout::CRR, KI_HINT, ST_A, ST_B, A_reg_t, B_reg_t>(
            g.a, g.b,
            /*m_subtile_A=*/0, /*group_idx=*/0, k_offset_tiles,
            As, Bs,
            swizzled_offsets_A, swizzled_offsets_B,
            a_srsrc_base, b_srsrc_base,
            a_base, b_base,
            a_lds_00, a_lds_01, a_lds_10, a_lds_11,
            b_lds_00, b_lds_01, b_lds_10, b_lds_11,
            row, col, warp_row, warp_col,
            ki_g,
            C_accum);

        if (warp_row == 0) { __builtin_amdgcn_s_barrier(); }

        // Store C[group_idx, m=row*BLK..., n=col*BLK...]. Output is
        // 3D-grouped; depth axis is group_idx. Two-axis-masked store
        // drops OOB cells when ``row`` or ``col`` is the partial last
        // tile in m_kernel (= g.n) or n_kernel (= g.k); aligned tiles
        // hit the fast forward to ``store(...)`` with no overhead.
        store_c_tile_mn_masked_grouped(g.c, C_accum[0][0],
            group_idx,
            (row * 2) * WARPS_M + warp_row,
            col * 2 * WARPS_N + warp_col,
            g.n, g.k);
        store_c_tile_mn_masked_grouped(g.c, C_accum[0][1],
            group_idx,
            (row * 2) * WARPS_M + warp_row,
            col * 2 * WARPS_N + WARPS_N + warp_col,
            g.n, g.k);
        store_c_tile_mn_masked_grouped(g.c, C_accum[1][0],
            group_idx,
            (row * 2) * WARPS_M + WARPS_M + warp_row,
            col * 2 * WARPS_N + warp_col,
            g.n, g.k);
        store_c_tile_mn_masked_grouped(g.c, C_accum[1][1],
            group_idx,
            (row * 2) * WARPS_M + WARPS_M + warp_row,
            col * 2 * WARPS_N + WARPS_N + warp_col,
            g.n, g.k);

        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }
}

// KI=0 dynamic only for v0; round-2 can add specializations once the
// numerics are validated and bench data identifies hot ki values.
template __global__ void grouped_var_k_kernel<0>(const grouped_var_k_layout_globals);

static inline void launch_grouped_var_k(grouped_var_k_layout_globals& g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    static bool attr_set = false;
    if (!attr_set) {
        hipFuncSetAttribute((void*)grouped_var_k_kernel<0>,
                            hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
        attr_set = true;
    }
    grouped_var_k_kernel<0><<<dim3(NUM_CUS), g.block(), mem_size, g.stream>>>(g);
}

void dispatch_grouped_var_k(grouped_var_k_layout_globals g) {
    g.n = static_cast<int>(g.a.cols());     // kernel M-output dim
    g.k = static_cast<int>(g.b.cols());     // kernel N-output dim
    g.M_total = static_cast<int>(g.a.rows());

    // Round-2 native non-aligned: bpr / bpc are ceil_div'ed so the
    // last tile in each output axis can be partial; the kernel's
    // ``store_c_tile_mn_masked_grouped`` drops OOB (m, n) cells. The
    // partial tiles still issue OOB A/B input loads, which are safe
    // because (a) A and B are 2D contiguous with full-tensor SRDs —
    // no SRD wrap into a different group's data (the issue that
    // blocks the forward grouped kernel from doing this), and
    // (b) MMA output cells whose (m, n) are OOB are dropped before
    // any store.
    //
    // Still need M_g >= 2*K_STEP (= 128) per group to satisfy the
    // prologue + epilog 1 + epilog 2 schedule of
    // ``device_gemm_tile_body`` — caller (Primus uniform_M >= 128
    // gate) enforces this; the persistent loop's ``ki_g < 2`` skip
    // is a defensive fallback.
    g.fast_n = g.n;
    g.fast_k = g.k;
    g.bpr = kittens::ceil_div(g.n, BLOCK_SIZE);
    g.bpc = kittens::ceil_div(g.k, BLOCK_SIZE);

    if (g.bpr <= 0 || g.bpc <= 0 || g.G <= 0) return;

    launch_grouped_var_k(g);
}

static void grouped_var_k_crr_fn(pybind11::object a, pybind11::object b, pybind11::object c,
                                 pybind11::object group_offs, int gm, int num_xcds) {
    auto group_offs_ptr = group_offs.attr("data_ptr")().cast<uintptr_t>();
    int G = group_offs.attr("numel")().cast<int>() - 1;

    grouped_var_k_layout_globals g{
        py::from_object<_gl>::make(a),
        py::from_object<_gl>::make(b),
        py::from_object<_gl>::make(c),
        reinterpret_cast<const int64_t*>(group_offs_ptr),
        {},
        G, 0, 0, 0, gm, num_xcds, 0, 0, 0,
        0, 0,  // fast_n, fast_k populated in dispatch.
    };
    dispatch_grouped_var_k(g);
}

// Round-9 binding signature: extra ``m_per_group`` int defaults to 0
// (non-uniform). When > 0 and a TAIL_BLOCK_M multiple, the dispatcher
// activates the LDS-staged interior K-tail correction kernel. Existing
// Primus callers that haven't been updated will pass the default and
// silently fall back to the scalar tail (zero behavior change).
static void grouped_rcr_fn(pybind11::object a, pybind11::object b, pybind11::object c,
                           pybind11::object group_offs, int gm, int num_xcds,
                           int m_per_group) {
    grouped_dispatch(a, b, c, group_offs, gm, num_xcds, m_per_group, "rcr");
}
static void grouped_rrr_fn(pybind11::object a, pybind11::object b, pybind11::object c,
                           pybind11::object group_offs, int gm, int num_xcds,
                           int m_per_group) {
    grouped_dispatch(a, b, c, group_offs, gm, num_xcds, m_per_group, "rrr");
}
static void grouped_crr_fn(pybind11::object a, pybind11::object b, pybind11::object c,
                           pybind11::object group_offs, int gm, int num_xcds,
                           int m_per_group) {
    grouped_dispatch(a, b, c, group_offs, gm, num_xcds, m_per_group, "crr");
}

PYBIND11_MODULE(tk_bf16_layouts, m) {
    using namespace pybind11::literals;
    m.def("gemm_rcr", &rcr, "a"_a, "b"_a, "c"_a, "group_m"_a=4, "num_xcds"_a=8);
    m.def("gemm_rrr", &rrr, "a"_a, "b"_a, "c"_a, "group_m"_a=4, "num_xcds"_a=8);
    m.def("gemm_crr", &crr, "a"_a, "b"_a, "c"_a, "group_m"_a=4, "num_xcds"_a=8);
    // [grouped] Persistent + CPU-sync-free grouped launchers. ``group_offs``
    // is a [G+1] int64 device tensor (prefix-sum of per-group M); the kernel
    // consumes it on the GPU side via O(G) linear scan, no host reads.
    m.def("grouped_rcr", &grouped_rcr_fn, "a"_a, "b"_a, "c"_a,
          "group_offs"_a, "group_m"_a=4, "num_xcds"_a=8, "m_per_group"_a=0);
    m.def("grouped_rrr", &grouped_rrr_fn, "a"_a, "b"_a, "c"_a,
          "group_offs"_a, "group_m"_a=4, "num_xcds"_a=8, "m_per_group"_a=0);
    m.def("grouped_crr", &grouped_crr_fn, "a"_a, "b"_a, "c"_a,
          "group_offs"_a, "group_m"_a=4, "num_xcds"_a=8, "m_per_group"_a=0);
    // [grouped-var-k] Persistent CPU-sync-free grouped variable-K (CRR / dB)
    // launcher. Different from grouped_crr (forward CRR with variable
    // M_total): here the variable axis is the K-reduction (= M_g per
    // group), and outputs are 3D ``[G, n, k]`` instead of 2D
    // ``[M_total, n]``. Used by the BF16 dB autograd path. v0 requires
    // n % BLOCK_SIZE == 0 and k % BLOCK_SIZE == 0; misaligned shapes
    // must use the per-group ``gemm_crr`` fallback.
    m.def("grouped_variable_k_crr", &grouped_var_k_crr_fn, "a"_a, "b"_a, "c"_a,
          "group_offs"_a, "group_m"_a=4, "num_xcds"_a=8);
    m.attr("BLOCK_SIZE") = BLOCK_SIZE;
    m.attr("K_STEP") = K_STEP;
}
