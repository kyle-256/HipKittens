#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <cstdlib>
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

#ifndef BF16_HOIST_M0
#define BF16_HOIST_M0 0
#endif

#ifndef BF16_LOAD_HOIST_AVAILABLE
#define BF16_LOAD_HOIST_AVAILABLE 1
#endif

#ifndef RCR_PADDED_B128_MODE
#define RCR_PADDED_B128_MODE 0
#endif

// Convenience: exposes the RCR ST_A/ST_B swap to all RCR-only callsites
// (typedefs, allocations, prefill, subtile lambdas). Orthogonal to BF16_HOIST_M0.
#if RCR_PADDED_B128_MODE
using rcr_padded_st_shape = kittens::st_64x32_padded_b128_s;
#endif

#if BF16_HOIST_M0 || BF16_LOAD_HOIST_AVAILABLE
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
#endif // BF16_HOIST_M0 || BF16_LOAD_HOIST_AVAILABLE

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

__device__ __forceinline__ float load_bf16_scalar(const _gl& src, int row, int col) {
    const uint32_t buffer_size = src.batch() * src.depth() * src.rows() * src.cols() * sizeof(bf16);
    const std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(src.raw_ptr);
    const std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    const buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);
    const i32x4 srsrc = std::bit_cast<i32x4>(br);
    const uint32_t voffset = (row * src.cols() + col) * sizeof(bf16);
    const uint16_t bits = llvm_amdgcn_raw_buffer_load_b16(srsrc, voffset, 0, 0);
    return base_types::convertor<float, bf16>::convert(std::bit_cast<bf16>(bits));
}

__device__ __forceinline__ void store_bf16_scalar(const _gl& dst, int row, int col, float value) {
    const uint32_t buffer_size = dst.batch() * dst.depth() * dst.rows() * dst.cols() * sizeof(bf16);
    const std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(dst.raw_ptr);
    const std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    const buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);
    const i32x4 srsrc = std::bit_cast<i32x4>(br);
    const uint32_t voffset = (row * dst.cols() + col) * sizeof(bf16);
    const bf16 v = base_types::convertor<bf16, float>::convert(value);
    llvm_amdgcn_raw_buffer_store_b16(std::bit_cast<uint16_t>(v), srsrc, voffset, 0, 0);
}

// Scalar bf16 load for a 3D-grouped tensor (B in grouped GEMM is laid out
// as `[1, G, *, *]`, where the 2nd axis indexes the group). Used by
// `grouped_tail_kernel_bf16` to read B at `(group_idx, row, col)`.
__device__ __forceinline__ float load_bf16_scalar_grp(const _gl& src, int g_idx, int row, int col) {
    const uint32_t buffer_size = src.batch() * src.depth() * src.rows() * src.cols() * sizeof(bf16);
    const std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(src.raw_ptr);
    const std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    const buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);
    const i32x4 srsrc = std::bit_cast<i32x4>(br);
    const uint32_t idx = ((0 * src.depth() + g_idx) * src.rows() + row) * src.cols() + col;
    const uint32_t voffset = idx * sizeof(bf16);
    const uint16_t bits = llvm_amdgcn_raw_buffer_load_b16(srsrc, voffset, 0, 0);
    return base_types::convertor<float, bf16>::convert(std::bit_cast<bf16>(bits));
}

// Packed 4 × bf16 = 8 bytes for vectorised tail-kernel K-loop. The HIP
// compiler emits a single `global_load_dwordx2` for a load through this
// type when the source pointer is 8-byte aligned, replacing 4 separate
// scalar bf16 loads (4× fewer VMEM transactions). Used by the RCR fast
// path inside `gemm_tail_kernel` / `grouped_tail_kernel_bf16` where both
// operands are stride-1 in K.
struct alignas(8) bf16x4 {
    bf16_2 lo, hi;
};

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

    uint32_t buffer_size = g_c.batch() * g_c.depth() * g_c.rows() * g_c.cols() * sizeof(U);
    std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(dst_ptr);
    std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);
    i32x4 srsrc = std::bit_cast<i32x4>(br);

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
                    U v0 = base_types::convertor<U, T>::convert(
                            src.tiles[i][j].data[idx].x);
                    U v1 = base_types::convertor<U, T>::convert(
                            src.tiles[i][j].data[idx].y);
                    const uint32_t off0 = ((row + l * 2)     * row_stride + col) * sizeof(U);
                    const uint32_t off1 = ((row + l * 2 + 1) * row_stride + col) * sizeof(U);
                    llvm_amdgcn_raw_buffer_store_b16(std::bit_cast<uint16_t>(v0), srsrc, off0, 0, 0);
                    llvm_amdgcn_raw_buffer_store_b16(std::bit_cast<uint16_t>(v1), srsrc, off1, 0, 0);
                }
            }
        }
    }
}

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
    // NOTE: deliberately NOT taking kittens::store fast path even when the
    // tile is fully in-bounds (m1 <= m_limit && n1 <= n_limit). Reason:
    // kittens::store builds its SRD with bound = full gl<> tensor_size from
    // dst_ptr, which over-reaches past the actual allocated memory when
    // g_c is a per-group SHIFTED VIEW (rows_internal == M_g < M_total). The
    // per-element path below shrinks the SRD to (tensor_end - dst_ptr) so
    // OOB writes hardware-clamp. Cost: ~10% slower than fast path on
    // aligned tiles; correctness is non-negotiable.
    constexpr int axis = 2;
    U* dst_ptr = (U*)&g_c[(coord<C_rt_accum_t>{0, group_idx, r_tile, c_tile}
                            .template unit_coord<axis, 3>())];
    const int row_stride = g_c.template stride<axis>();
    const int laneid = kittens::laneid();
    const int row_offset = src.base_tile_stride * (laneid / src.base_tile_cols);
    const int col_offset = laneid % src.base_tile_cols;

    // SRD size = bytes from tile start to tensor end. Previous formulation
    // used full_tensor_size with mid-tensor base, which let late-tile stores
    // reach offsets beyond the actual mapped region → "Memory access fault
    // by GPU node-N" on shapes where any output dim was BLOCK_SIZE-misaligned
    // (gpt_oss N=5760, K=2880). The per-row m_limit/n_limit lambda guards
    // already drop OOB-of-tensor stores; this size shrink adds a fault-safe
    // outer bound for lanes that slip past the per-row guards.
    const std::uintptr_t tensor_base_int =
        reinterpret_cast<std::uintptr_t>(&g_c[(coord<>{0, 0, 0, 0})]);
    const std::uint64_t  tensor_size_total =
        static_cast<std::uint64_t>(g_c.batch()) *
        static_cast<std::uint64_t>(g_c.depth()) *
        static_cast<std::uint64_t>(g_c.rows())  *
        static_cast<std::uint64_t>(g_c.cols())  *
        static_cast<std::uint64_t>(sizeof(U));
    const std::uintptr_t tensor_end_int = tensor_base_int + tensor_size_total;
    const std::uintptr_t dst_int = reinterpret_cast<std::uintptr_t>(dst_ptr);
    const std::uint64_t  remaining_bytes64 =
        (dst_int < tensor_end_int) ? (tensor_end_int - dst_int) : 0;
    const uint32_t buffer_size = static_cast<uint32_t>(
        remaining_bytes64 > 0xFFFFFFFFu ? 0xFFFFFFFFu : remaining_bytes64);
    std::uint64_t  as_u64 = static_cast<std::uint64_t>(dst_int);
    buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);
    i32x4 srsrc = std::bit_cast<i32x4>(br);

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
                        U v0 = base_types::convertor<U, T>::convert(
                                src.tiles[i][j].data[idx].x);
                        const uint32_t off0 = (row_a * row_stride + col) * sizeof(U);
                        llvm_amdgcn_raw_buffer_store_b16(std::bit_cast<uint16_t>(v0), srsrc, off0, 0, 0);
                    }
                    if (m0 + row_b < m_limit) {
                        U v1 = base_types::convertor<U, T>::convert(
                                src.tiles[i][j].data[idx].y);
                        const uint32_t off1 = (row_b * row_stride + col) * sizeof(U);
                        llvm_amdgcn_raw_buffer_store_b16(std::bit_cast<uint16_t>(v1), srsrc, off1, 0, 0);
                    }
                }
            }
        }
    }
}

// =============================================================================
// device_gemm_tile_body — shared GEMM main-loop body (Phase 2 refactor).
//
// The dense `gemm_kernel<L>` and the persistent grouped
// `grouped_gemm_bf16_kernel<L>` previously contained two byte-identical
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
// `num_tiles_dyn` is the dynamic K-tile count (= g.fast_k / K_STEP),
// always the loop bound since KI_HINT specializations were removed.
// =============================================================================
template<Layout L,
         typename ST_A_T, typename ST_B_T,
         typename A_reg_t, typename B_reg_t,
         bool FUSED_KTAIL = false>
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
    rt_fl<HALF_REG_BLOCK_M, HALF_REG_BLOCK_N, col_l, rt_16x16_s> (&C_accum)[2][2],
    bool has_k_tail = false)
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

    // CRR-layout helpers: clamp the spatial arg to the last fully in-bounds
    // slab. raw_buffer_load_lds is supposed to no-op on OOB voffsets, but
    // on gfx950 a fully-OOB 64×128 BF16 prefetch via G::load reliably
    // faults a downstream store on shapes where the global tensor's last
    // dim is not a multiple of 2*HALF_BLOCK_SIZE (gpt_oss N=5760 K=2880
    // wgrad). Loaded data is downstream-discarded for the clamped slabs —
    // the store-helper's m_limit/n_limit early-return drops every output
    // cell that would have used those slabs.
    auto a_coord_safe = [&](int spatial, int k) {
        if constexpr (L == Layout::CRR) {
            const int spatial_abs  = m_subtile_A + spatial;
            const int last_safe    = static_cast<int>(a_gl.cols()) / HALF_BLOCK_SIZE - 1;
            const int spatial_safe = (spatial_abs <= last_safe) ? spatial_abs : last_safe;
            return a_coord(spatial_safe - m_subtile_A, k);
        } else {
            // Grouped path: a_gl is the per-group shifted view with
            // rows_internal = M_g. The spatial dim addresses sub-tiles of
            // HALF_BLOCK_SIZE rows each; the kernel's main-loop schedule
            // always issues spatial=0 AND spatial=1 loads. Two failure modes:
            //  (a) spatial completely past the last record (record_id_first
            //      >= num_records, i.e. spatial * HALF_BLOCK_SIZE >= M_g):
            //      swizzled SRD's record-index clamp is unreliable on gfx950
            //      → real OOB read fault. Must clamp.
            //  (b) spatial partially in-bounds (M_g not a HALF_BLOCK_SIZE
            //      multiple): byte-level SRD clamp on individual voffsets
            //      handles the OOB rows correctly, no clamp needed.
            // Use ceil_div so partial-last-subtile is treated as in-bounds.
            // Also floor at 0 — without the floor, M_g < HALF_BLOCK_SIZE
            // gave last_safe=-1 → negative SOFF → uint32 wrap to ~4GB fault.
            // Per-group A SRD uses non-swizzled bound (see make_srsrc call
            // at the group-transition block). Non-swizzled SRD HW-clamps
            // byte-level OOB voffsets to 0. So spatial fully-OOB or
            // partial-OOB both safe to load (OOB lanes return 0, the
            // store-side m_limit mask drops invalid rows).
            return a_coord(spatial, k);
        }
    };
    auto b_coord_safe = [&](int spatial, int k) {
        if constexpr (L == Layout::CRR) {
            const int last_safe    = static_cast<int>(b_gl.cols()) / HALF_BLOCK_SIZE - 1;
            const int spatial_safe = (spatial <= last_safe) ? spatial : last_safe;
            return b_coord(spatial_safe, k);
        } else {
            return b_coord(spatial, k);
        }
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
    G::load(Bs[tic][1], b_gl, b_coord_safe(col*2+1, 0), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_01);
    G::load(As[tic][1], a_gl, a_coord_safe(row*2+1, 0), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_01);

    if (warp_row == 1) { __builtin_amdgcn_s_barrier(); }
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    G::load(Bs[toc][0], b_gl, b_coord(col*2, 1), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_10);
    G::load(As[toc][0], a_gl, a_coord(row*2, 1), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_10);
    G::load(Bs[toc][1], b_gl, b_coord_safe(col*2+1, 1), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_11);

    asm volatile("s_waitcnt vmcnt(6)");
    __builtin_amdgcn_s_barrier();

    /********** Main loop **********/
    auto main_loop_iter = [&](int tile) {
        load_b_subtile(B_tile_0, Bs[0][0], warp_col);
        load_a_subtile(A_tile, As[0][0], warp_row);
        G::load(As[1][1], a_gl, a_coord_safe(row*2+1, tile+1), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_11);
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
        __builtin_amdgcn_sched_barrier(0);

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
        G::load(Bs[0][1], b_gl, b_coord_safe(col*2+1, tile+2), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_01);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load_a_subtile(A_tile, As[1][0], warp_row);
        G::load(As[0][1], a_gl, a_coord_safe(row*2+1, tile+2), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_01);
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
        __builtin_amdgcn_sched_barrier(0);

        load_a_subtile(A_tile, As[1][1], warp_row);
        G::load(As[1][0], a_gl, a_coord(row*2, tile+3), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_10);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Skip the tile+3 prefetch on the LAST main-loop iter (= tile+4 == num_tiles_dyn).
        // For ki_g >= 63 (gpt_oss B>=4 M>=4096 wgrad shapes), this prefetch
        // combined with the prior tile+3 prefetches in the same iter triggers
        // a downstream "Memory access fault by GPU node-N" on misaligned-N
        // shapes. The prefetched tile is K=tile+3 = ki_g-1 (already loaded
        // by the main loop's prior iter's tile+1 prefetch path or used by
        // Epilog 1 directly), so dropping it on the last iter is safe.
        G::load(Bs[1][1], b_gl, b_coord_safe(col*2+1, tile+3), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_11);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    };

    {
        const int num_tiles = num_tiles_dyn;
        #pragma unroll 2
        for (int tile = 0; tile < num_tiles - 2; tile += 2) main_loop_iter(tile);
    }

    /********** Epilog 1: second-to-last K-tile pair **********/
    {
        const int tile = num_tiles_dyn - 2;
        load_b_subtile(B_tile_0, Bs[tic][0], warp_col);
        load_a_subtile(A_tile, As[tic][0], warp_row);
        G::load(As[toc][1], a_gl, a_coord_safe(row*2+1, tile+1), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_11);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load_b_subtile(B_tile_1, Bs[tic][1], warp_col);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load_a_subtile(A_tile, As[tic][1], warp_row);
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        DO_MMA(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
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
        __builtin_amdgcn_sched_barrier(0);

        load_b_subtile(B_tile_1, Bs[tic][1], warp_col);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load_a_subtile(A_tile, As[tic][1], warp_row);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        DO_MMA(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    /********** Round-4 path A: fused K-tail epilog (RCR/RRR). When
     * `has_k_tail` is false at runtime, the entire block is no-op'd —
     * we still compile the code so a single template instance can serve
     * both K_rem==0 and K_rem==K_STEP shapes. **********/
    if constexpr (FUSED_KTAIL) {
        if (has_k_tail) {
        if constexpr (L == Layout::RCR) {
            const int k_tail_tile = num_tiles_dyn;
            const int laneid = ::kittens::laneid();
            const int row_lane = laneid % 16;
            const int k_lane_bytes =
                (laneid / 16) * 8 * (int)sizeof(bf16);
            const int K_tail_base_bytes =
                (k_offset_tiles + k_tail_tile) * K_STEP *
                (int)sizeof(bf16);

            // Row strides in BYTES (matches a_srsrc_base /
            // b_srsrc_base construction in the caller).
            const int a_row_stride_bytes =
                a_gl.template stride<2>() * (int)sizeof(bf16);
            const int b_row_stride_bytes =
                b_gl.template stride<2>() * (int)sizeof(bf16);
            const int b_inner_rows = b_gl.rows();  // N for RCR

            auto load_a_kt = [&](int m_slab) __attribute__((always_inline)) {
                const int M_warp_base =
                    ((m_subtile_A + row*2 + m_slab) * 2 + warp_row) *
                    HALF_REG_BLOCK_M;
                #pragma unroll
                for (int h = 0; h < A_reg_t::height; ++h) {
                    #pragma unroll
                    for (int w = 0; w < A_reg_t::width; ++w) {
                        const int A_row_idx = M_warp_base + h * 16 + row_lane;
                        const uint32_t v_offset = static_cast<uint32_t>(
                            A_row_idx * a_row_stride_bytes +
                            w * 32 * (int)sizeof(bf16) +
                            K_tail_base_bytes + k_lane_bytes);
                        __uint128_t v = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            a_srsrc_base, v_offset, 0, 0);
                        *reinterpret_cast<__uint128_t*>(
                            &A_tile.tiles[h][w].data[0]) = v;
                    }
                }
            };
            auto load_b_kt = [&](B_reg_t& B_tile, int n_strip) __attribute__((always_inline)) {
                const int N_warp_base =
                    (col * 8 + n_strip * 4 + warp_col) *
                    HALF_REG_BLOCK_N;
                #pragma unroll
                for (int h_b = 0; h_b < B_reg_t::height; ++h_b) {
                    #pragma unroll
                    for (int w = 0; w < B_reg_t::width; ++w) {
                        const int B_row_idx = N_warp_base + h_b * 16 + row_lane;
                        const uint32_t v_offset = static_cast<uint32_t>(
                            (group_idx * b_inner_rows + B_row_idx) *
                                b_row_stride_bytes +
                            w * 32 * (int)sizeof(bf16) +
                            K_tail_base_bytes + k_lane_bytes);
                        __uint128_t v = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            b_srsrc_base, v_offset, 0, 0);
                        *reinterpret_cast<__uint128_t*>(
                            &B_tile.tiles[h_b][w].data[0]) = v;
                    }
                }
            };

            // M slab 0
            load_a_kt(0);
            load_b_kt(B_tile_0, 0);
            load_b_kt(B_tile_1, 1);
            asm volatile("s_waitcnt vmcnt(0)");
            DO_MMA(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
            DO_MMA(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
            // M slab 1 (B tiles unchanged — share K-tail across slabs)
            load_a_kt(1);
            asm volatile("s_waitcnt vmcnt(0)");
            DO_MMA(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
            DO_MMA(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        } else if constexpr (L == Layout::RRR) {
            const int k_tail_tile = num_tiles_dyn;
            const int laneid = ::kittens::laneid();

            // ---- A: direct HBM → register (mirror RCR path B) ----
            const int row_lane = laneid % 16;
            const int k_lane_bytes =
                (laneid / 16) * 8 * (int)sizeof(bf16);
            const int K_tail_base_bytes =
                (k_offset_tiles + k_tail_tile) * K_STEP *
                (int)sizeof(bf16);
            const int a_row_stride_bytes =
                a_gl.template stride<2>() * (int)sizeof(bf16);

            auto load_a_kt = [&](int m_slab) __attribute__((always_inline)) {
                const int M_warp_base =
                    ((m_subtile_A + row*2 + m_slab) * 2 + warp_row) *
                    HALF_REG_BLOCK_M;
                #pragma unroll
                for (int h = 0; h < A_reg_t::height; ++h) {
                    #pragma unroll
                    for (int w = 0; w < A_reg_t::width; ++w) {
                        const int A_row_idx = M_warp_base + h * 16 + row_lane;
                        const uint32_t v_offset = static_cast<uint32_t>(
                            A_row_idx * a_row_stride_bytes +
                            w * 32 * (int)sizeof(bf16) +
                            K_tail_base_bytes + k_lane_bytes);
                        __uint128_t v = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            a_srsrc_base, v_offset, 0, 0);
                        *reinterpret_cast<__uint128_t*>(
                            &A_tile.tiles[h][w].data[0]) = v;
                    }
                }
            };

            G::load(Bs[1][0], b_gl, b_coord(col*2,   k_tail_tile),
                    swizzled_offsets_B, b_srsrc_base, b_base, b_lds_10);
            G::load(Bs[1][1], b_gl, b_coord_safe(col*2+1, k_tail_tile),
                    swizzled_offsets_B, b_srsrc_base, b_base, b_lds_11);
            asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
            __syncthreads();

            // Per-lane intra-subtile address (computed once).
            const int b_row_off_lane = ((laneid % 16) / 4) + ((laneid / 16) * 8);
            const int b_col_off_lane = (laneid % 4) * 4;
            const uint32_t b_intra_off = [&]() -> uint32_t {
                const uint32_t off = static_cast<uint32_t>(
                    2 * (b_row_off_lane * 16 + b_col_off_lane));
                const uint32_t sw = ((off % 1024u) >> 9) << 4;
                return off ^ sw;
            }();

            const uint32_t b_arr_base_10 = __builtin_amdgcn_readfirstlane(
                static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1][0].data[0])));
            const uint32_t b_arr_base_11 = __builtin_amdgcn_readfirstlane(
                static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1][1].data[0])));

            auto load_b_kt = [&](B_reg_t& B_tile, int n_strip) __attribute__((always_inline)) {
#ifndef BF16_RRR_FUSE_USE_KITTENS
#define BF16_RRR_FUSE_USE_KITTENS 0
#endif
#if BF16_RRR_FUSE_USE_KITTENS
                if (n_strip == 0) {
                    auto sub = subtile_inplace<K_STEP, HALF_REG_BLOCK_N>(Bs[1][0], {0, warp_col});
                    load(B_tile, sub);
                } else {
                    auto sub = subtile_inplace<K_STEP, HALF_REG_BLOCK_N>(Bs[1][1], {0, warp_col});
                    load(B_tile, sub);
                }
#else
                const uint32_t bs_base =
                    (n_strip == 0) ? b_arr_base_10 : b_arr_base_11;
                const uint32_t src_ptr =
                    bs_base + static_cast<uint32_t>(warp_col) * 2048u;
                const uint32_t addr = src_ptr + b_intra_off;
                #pragma unroll
                for (int h_b = 0; h_b < B_reg_t::height; ++h_b) {
                    #pragma unroll
                    for (int w = 0; w < B_reg_t::width; ++w) {
                        const int shared_subtile_id = h_b * 8 + w;
                        const int offset_bytes = shared_subtile_id * 1024;
                        asm volatile(
                            "ds_read_b64_tr_b16 %0, %2 offset:%3\n"
                            "ds_read_b64_tr_b16 %1, %2 offset:%4\n"
                            : "=v"(*reinterpret_cast<float2*>(
                                  &B_tile.tiles[h_b][w].data[0])),
                              "=v"(*reinterpret_cast<float2*>(
                                  &B_tile.tiles[h_b][w].data[2]))
                            : "v"(addr),
                              "i"(offset_bytes),
                              "i"(offset_bytes + 128)
                            : "memory"
                        );
                    }
                }
#endif
            };

            // ---- DO_MMA dispatch ----
#ifndef BF16_RRR_FUSE_SKIP_DO_MMA
#define BF16_RRR_FUSE_SKIP_DO_MMA 0
#endif
#if BF16_RRR_FUSE_SKIP_DO_MMA
            // Diagnostic: skip K-tail accumulation entirely. SNR floor =
            // main-only (K=[0, fast_k)). Compare with fuse-enabled SNR to
            // determine if fuse is contributing useful or harmful data.
            (void)load_a_kt;
            (void)load_b_kt;
#else
            load_a_kt(0);
            load_b_kt(B_tile_0, 0);
            load_b_kt(B_tile_1, 1);
            asm volatile("s_waitcnt vmcnt(0)");
            asm volatile("s_waitcnt lgkmcnt(0)");
            DO_MMA(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
            DO_MMA(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);

            load_a_kt(1);
            asm volatile("s_waitcnt vmcnt(0)");
            DO_MMA(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
            DO_MMA(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
#endif
        }
        }  // end if (has_k_tail)
    }

    #undef DO_MMA
}

// Dynamic K only — num_tiles read from g.ki, `#pragma unroll 2`.
template<Layout L>
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
    // The helper forwards g.ki as `num_tiles_dyn`; the main loop runs
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
    device_gemm_tile_body<L, ST_A, ST_B, A_reg_t, B_reg_t>(
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
        // see ``grouped_tail_kernel_bf16`` for the rationale + bench (~4×
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

template __global__ void gemm_kernel<Layout::RCR>(const layout_globals);
template __global__ void gemm_kernel<Layout::RRR>(const layout_globals);
template __global__ void gemm_kernel<Layout::CRR>(const layout_globals);

template<Layout L>
static inline void launch_one(layout_globals& g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    // Set shared-mem attribute once per (L, KI) function pointer: idempotent,
    // avoids per-launch HIP runtime overhead.
    static bool attr_set = false;
    if (!attr_set) {
        hipFuncSetAttribute((void*)gemm_kernel<L>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
        attr_set = true;
    }
    int total_blocks = g.bpr * g.bpc;
    gemm_kernel<L><<<dim3(total_blocks), g.block(), mem_size, g.stream>>>(g);
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
    // M>=1024 in the dynamic kernel). In that case we fall back to
    // bpc = fast_n / BLOCK_SIZE and the scalar tail kernel covers cols
    // [fast_n, n).
    g.bpc = (g.fast_k == g.k)
        ? kittens::ceil_div(g.n, BLOCK_SIZE)
        : (g.fast_n / BLOCK_SIZE);
    g.ki  = g.fast_k / K_STEP;

    if (g.bpr > 0 && g.bpc > 0 && g.ki >= 2) {
        launch_one<L>(g);
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

#ifndef PRIMUS_TURBO_HK_INTEGRATION
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
#endif  // PRIMUS_TURBO_HK_INTEGRATION

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
    // `grouped_tail_kernel_bf16` reading `group_offs`. Cells outside
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
    int* tile_counter;
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

// Single template — FUSED_KTAIL compile-time gate folded into a runtime
// `has_k_tail` check inside device_gemm_tile_body, so one kernel binary
// handles both K_rem==0 and K_rem==K_STEP shapes.
template<Layout L>
__global__ __launch_bounds__(NUM_THREADS, 1)
void grouped_gemm_bf16_kernel(const grouped_layout_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

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

    __shared__ int s_claim;
    int pid;
    if (g.tile_counter != nullptr) {
        if (threadIdx.x == 0) {
            s_claim = atomicAdd(g.tile_counter, 1);
        }
        __syncthreads();
        pid = s_claim;
    } else {
        pid = chiplet_transform_chunked(blockIdx.x, NUM_CUS, g.num_xcds, 64);
    }

    const int num_pid_n = g.bpc;

    if (threadIdx.x == 0) {
        int prev = static_cast<int>(g.group_offs[0]);
        s_offs[0] = prev;
        s_cum_tiles[0] = 0;
        int t = 0;
        #pragma unroll 1
        for (int gi = 0; gi < g.G; ++gi) {
            const int next = static_cast<int>(g.group_offs[gi + 1]);
            s_offs[gi + 1] = next;
            // ceil_div: include the partial-M tile when M_g % BLOCK_SIZE != 0
            // so the main kernel covers every group's M_g rows itself.
            t += kittens::ceil_div(next - prev, BLOCK_SIZE) * num_pid_n;
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
    const bf16* c_base = (bf16*)&g.c[{0, 0, 0, 0}];
    (void)c_base;  // only referenced when per-group rebuild fires below
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

    int last_group_idx = -1;
    i32x4 a_srsrc_curr = a_srsrc_base;
    i32x4 b_srsrc_curr = make_srsrc(b_base, b_inner_rows * b_row_stride, /*row_stride_bytes=*/0);
    // Per-group shifted base pointers; rebuilt on group_idx transitions.
    // Shifting a_base / c_base to m_start_g lets the kernel address the
    // current group's data via coord row = 0..bpr_g (no need for m_subtile_*
    // integer division — m_start_g can be arbitrary byte-level value).
    const bf16* a_base_g = a_base;
    const bf16* c_base_g = c_base;

    int gt = pid;
    while (gt < total_tiles) {

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
        // ceil_div so the final partial M-block (M_g % BLOCK_SIZE != 0) gets
        // its own pid_m; the store path masks OOB rows via m_limit, and the
        // per-group A SRD clamps OOB row loads to 0.
        const int bpr_g = kittens::ceil_div(M_g, BLOCK_SIZE);

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

        // Element-level addressing via per-group shifted GL views (below)
        // means m_subtile_A / m_subtile_C are always 0 — the shifted base
        // already locates the group's rows. Supports arbitrary byte-level
        // m_start_g (callers no longer need group_lens % HALF_BLOCK_SIZE).
        constexpr int m_subtile_A = 0;
        constexpr int m_subtile_C = 0;

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
            // Shift A/C base pointers to m_start_g (byte-level OK — pointer
            // arithmetic doesn't care about alignment). A SRD then bounds
            // [a_base_g, a_base_g + M_g*row_stride), so partial-tile OOB
            // row loads (M_g % BLOCK_SIZE != 0) hardware-clamp to 0.
            a_base_g = (const bf16*)&g.a[{0, 0, m_start_g, 0}];
            c_base_g = (const bf16*)&g.c[{0, 0, m_start_g, 0}];
            // NO swizzle on per-group A SRD: with swizzle, gfx950's
            // raw_buffer_load_lds doesn't reliably clamp OOB voffsets
            // (record_id >= num_records pass-through → fault on partial
            // last-tile loads). Non-swizzled SRD uses byte-level OOB
            // check, HW-clamps voffset>=range to 0 → safe.
            a_srsrc_curr = make_srsrc(
                a_base_g, M_g * a_row_stride, /*row_stride_bytes=*/0);
            const int b_grp_total_rows = (group_idx + 1) * b_inner_rows;
            b_srsrc_curr = make_srsrc(
                b_base, b_grp_total_rows * b_row_stride,
                /*row_stride_bytes=*/0);
            last_group_idx = group_idx;
        }
        // Per-tile shifted GL views: copy g.a / g.c then patch raw_ptr +
        // rows_internal to point at the group's slice. (gl<>'s host-only
        // constructor sets up TMA descriptors we don't use; copy-then-
        // mutate is the device-safe path.) Lets coord (0,*,*,*) inside
        // device_gemm_tile_body and store_c_tile_mn_masked_grouped resolve
        // to absolute row m_start_g + (coord_row * unit) without needing
        // m_start_g to be a multiple of any block size.
        _gl a_gl_g = g.a;
        a_gl_g.raw_ptr = (bf16*)a_base_g;
        a_gl_g.rows_internal = M_g;
        _gl c_gl_g = g.c;
        c_gl_g.raw_ptr = (bf16*)c_base_g;
        c_gl_g.rows_internal = M_g;

        device_gemm_tile_body<L, ST_A, ST_B, A_reg_t, B_reg_t, /*FUSED_KTAIL=*/true>(
            a_gl_g, g.b,
            m_subtile_A, group_idx, /*k_offset_tiles=*/0,
            As, Bs,
            swizzled_offsets_A, swizzled_offsets_B,
            a_srsrc_curr, b_srsrc_curr,
            a_base_g, b_base,
            a_lds_00, a_lds_01, a_lds_10, a_lds_11,
            b_lds_00, b_lds_01, b_lds_10, b_lds_11,
            row, col, warp_row, warp_col,
            g.ki,
            C_accum,
            /*has_k_tail=*/ (g.k - g.fast_k == K_STEP));

        if (warp_row == 0) { __builtin_amdgcn_s_barrier(); }

        // m_subtile_C = 0 (shifted c_gl_g base). r0/r1 in HALF_REG_BLOCK_M=64
        // unit relative to m_start_g. m_limit = M_g (group-relative element
        // count) — store_c_tile_mn_masked_grouped masks per-cell so partial
        // last-tile OOB rows + partial N-cols are skipped.
        const int r0 = (row * 2) * WARPS_M + warp_row;
        const int r1 = (row * 2) * WARPS_M + WARPS_M + warp_row;
        const int c0 = col * 2 * WARPS_N + warp_col;
        const int c1 = col * 2 * WARPS_N + WARPS_N + warp_col;
        const int m_limit = M_g;
        const int n_limit = g.n;
        store_c_tile_mn_masked_grouped(c_gl_g, C_accum[0][0], 0, r0, c0, m_limit, n_limit);
        store_c_tile_mn_masked_grouped(c_gl_g, C_accum[0][1], 0, r0, c1, m_limit, n_limit);
        store_c_tile_mn_masked_grouped(c_gl_g, C_accum[1][0], 0, r1, c0, m_limit, n_limit);
        store_c_tile_mn_masked_grouped(c_gl_g, C_accum[1][1], 0, r1, c1, m_limit, n_limit);

        // [grouped] Drain in-flight ops before the next persistent iteration so
        // the next tile's prologue starts from a clean state.
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        if (g.tile_counter != nullptr) {
            if (threadIdx.x == 0) {
                s_claim = atomicAdd(g.tile_counter, 1);
            }
            __syncthreads();
            gt = s_claim;
        } else {
            gt += NUM_CUS;
        }
    }
}

// Explicit instantiations: single instance per layout. FUSED_KTAIL is folded
// into a runtime `has_k_tail` check inside the kernel.
template __global__ void grouped_gemm_bf16_kernel<Layout::RCR>(const grouped_layout_globals);
template __global__ void grouped_gemm_bf16_kernel<Layout::RRR>(const grouped_layout_globals);

static int* grouped_tile_counter_buffer() {
    static int* d_counter = nullptr;
    if (d_counter == nullptr) {
        hipMalloc(&d_counter, sizeof(int));
    }
    return d_counter;
}

static inline bool should_use_work_stealing(int M_total, int bpc) {
    if (bpc <= 0 || M_total <= 0) return false;
    const int tiles = (M_total / BLOCK_SIZE) * bpc;
    return (tiles > 0) && (tiles < NUM_CUS * 4) && ((tiles % NUM_CUS) != 0);
}

static inline void prime_grouped_tile_counter(grouped_layout_globals& g) {
    if (!should_use_work_stealing(g.M_total, g.bpc)) {
        g.tile_counter = nullptr;
        return;
    }
    int* counter = grouped_tile_counter_buffer();
    g.tile_counter = counter;
    hipMemsetAsync(counter, 0, sizeof(int), g.stream);
}

template<Layout L>
static inline void launch_one_grouped(grouped_layout_globals& g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    static bool attr_set = false;
    if (!attr_set) {
        hipFuncSetAttribute((void*)grouped_gemm_bf16_kernel<L>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
        attr_set = true;
    }
    prime_grouped_tile_counter(g);
    grouped_gemm_bf16_kernel<L><<<dim3(NUM_CUS), g.block(), mem_size, g.stream>>>(g);
}

template<Layout L>
void dispatch_grouped(grouped_layout_globals g) {
    g.n = static_cast<int>(g.c.cols());
    g.M_total = static_cast<int>(g.c.rows());
    if constexpr (L == Layout::CRR) g.k = static_cast<int>(g.a.rows());
    else g.k = static_cast<int>(g.a.cols());

    g.fast_n = (g.n / BLOCK_SIZE) * BLOCK_SIZE;
    g.fast_k = (g.k / K_TWO_TILE) * K_TWO_TILE;
    if constexpr (L == Layout::RCR || L == Layout::RRR) {
        g.bpc = kittens::ceil_div(g.n, BLOCK_SIZE);
    } else {
        g.bpc = g.fast_n / BLOCK_SIZE;
    }
    g.ki     = g.fast_k / K_STEP;

    const int K_rem_for_fuse = g.k - g.fast_k;
    // Single kernel binary per layout. FUSED_KTAIL is folded into a runtime
    // `has_k_tail` check inside grouped_gemm_bf16_kernel; the dispatcher just
    // computes whether fuse would actually run (used below by the scalar tail
    // launch gate). K_rem==K_STEP is the only condition that triggers fuse.
    const bool fuse_ktail_eligible =
        (g.bpc > 0) && (g.ki >= 2) && (K_rem_for_fuse == K_STEP);

    if (g.bpc > 0 && g.ki >= 2) {
        launch_one_grouped<L>(g);
    } else {
        // Main kernel can't run (N < BLOCK_SIZE or K < K_TWO_TILE). Reset
        // fast_* so the tail kernel treats every cell as "kernel never
        // ran here" and computes the full output from scratch.
        g.fast_n = 0;
        g.fast_k = 0;
        g.bpc    = 0;
        g.ki     = 0;
    }

    // Single-kernel design: main kernel processes ceil_div(M_g, BLOCK_SIZE)
    // M-tiles per group (including the partial last tile), uses per-group
    // A SRD that clips to [0, m_start_g + M_g) so partial-tile OOB rows
    // hardware-clamp to 0, and stores via store_c_tile_mn_masked_grouped
    // which drops OOB cells from the write-back. K-tail (K_rem == K_STEP)
    // is folded in via the FUSED_KTAIL block (runtime has_k_tail check).
    // No scalar tail kernel — main alone covers every cell of the output.
    (void)fuse_ktail_eligible;  // computed earlier; unused now
}

#ifndef PRIMUS_TURBO_HK_INTEGRATION
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
    else dispatch_grouped<Layout::RRR>(g);
    // CRR not supported for grouped forward (PT routes CRR/wgrad through
    // grouped_gemm_var_k_bf16_kernel; HK-only test path treats CRR as RRR).
}
#endif  // PRIMUS_TURBO_HK_INTEGRATION

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
    // Aligned-region dims. v0 only: aligned-only. fast_{n,k} == n,k and
    // m_aligned == M_total are required by ``can_handle`` on the Python
    // side; the dispatcher otherwise falls back to the per-group loop.
    int fast_n, fast_k;
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

struct var_k_coord_result_t {
    int pid_m;
    int pid_n;
    int valid;  // 1 == live tile, 0 == skip (out-of-bounds partial group)
};

struct var_k_group_lookup_t {
    int group_idx;
    int local_tile;
    int m_start_g;
    int M_g;
    int ki_g;
    int valid;  // 0 if ceil_div(M_g, K_STEP) < 2 -> skip
};

#ifdef PROFILE_VAR_K_NOINLINE
#define VAR_K_HELPER_ATTR __attribute__((noinline))
#else
#define VAR_K_HELPER_ATTR __attribute__((always_inline))
#endif

__device__ VAR_K_HELPER_ATTR
var_k_coord_result_t compute_var_k_coords(int local_tile,
                                          int num_pid_m,
                                          int num_pid_n,
                                          int group_m) {
    int pid_m = 0, pid_n = 0;
    if (num_pid_n > num_pid_m) {
        const int WGN = group_m;
        const int num_wgid_in_group = num_pid_m * WGN;
        int group_id = local_tile / num_wgid_in_group;
        int first_pid_n = group_id * WGN;
        int group_size_n = min(num_pid_n - first_pid_n, WGN);
        if (group_size_n <= 0) return {0, 0, 0};
        pid_n = first_pid_n + ((local_tile % num_wgid_in_group) % group_size_n);
        pid_m = (local_tile % num_wgid_in_group) / group_size_n;
    } else {
        const int WGM = group_m;
        const int num_wgid_in_group = WGM * num_pid_n;
        int group_id = local_tile / num_wgid_in_group;
        int first_pid_m = group_id * WGM;
        int group_size_m = min(num_pid_m - first_pid_m, WGM);
        if (group_size_m <= 0) return {0, 0, 0};
        pid_m = first_pid_m + ((local_tile % num_wgid_in_group) % group_size_m);
        pid_n = (local_tile % num_wgid_in_group) / group_size_m;
    }
    if (pid_m >= num_pid_m || pid_n >= num_pid_n) return {0, 0, 0};
    return {pid_m, pid_n, 1};
}

__device__ VAR_K_HELPER_ATTR
var_k_group_lookup_t compute_var_k_group_lookup(int gt,
                                                int tiles_per_group,
                                                const int* s_offs) {
    const int group_idx = gt / tiles_per_group;
    const int local_tile = gt - group_idx * tiles_per_group;
    const int m_start_g = s_offs[group_idx];
    const int M_g = s_offs[group_idx + 1] - m_start_g;
    // ceil_div so the last partial K-tile (M_g % K_STEP != 0) gets its own
    // ki iteration. Per-group shifted A/B SRD + non-swizzled bound clamps
    // OOB rows in that partial tile to 0 (no contribution to accumulator).
    const int ki_g = kittens::ceil_div(M_g, K_STEP);
    if (ki_g <= 0) return {0, 0, 0, 0, 0, 0};
    // ki_g == 1 (M_g <= K_STEP) is allowed: per-group A/B SRDs are
    // bounded to M_g * row_stride bytes (non-swizzled), so the prologue's
    // tile-1 load and the main-loop's prefetches reading beyond row M_g
    // hardware-clamp their voffsets to 0. Main loop range (0..ki_g-2 step 2)
    // is empty for ki_g <= 2, and epilog 1/2 still complete using the
    // prefetched tile-0 data (with zeros for OOB tiles contributing
    // nothing to the accumulator). num_tiles_dyn = ki_g is passed verbatim.
    // Force ki_g to at least 2 so the K-loop bound covers prologue + epilog;
    // the second virtual tile is fully-OOB and clamped to 0, so its MMA
    // contribution is zero.
    const int ki_g_min = (ki_g < 2) ? 2 : ki_g;
    return {group_idx, local_tile, m_start_g, M_g, ki_g_min, 1};
}

__global__ __launch_bounds__(NUM_THREADS, 1)
void grouped_gemm_var_k_bf16_kernel(const grouped_var_k_layout_globals g) {
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

    // A and B are CRR-layout K-reduction tensors spanning [M_total, *];
    // the K-axis is the per-group row range [m_start_g, m_start_g + M_g).
    // Per-group: shift A/B base pointers to m_start_g (byte-level OK) and
    // build per-group SRDs bounded to M_g * row_stride. Rebuilt on
    // group_idx transitions to amortise SGPR build cost across persistent
    // iters of the same group. Non-swizzled (third arg = 0) so HW
    // byte-level OOB clamp handles partial last K-tile and any over-read.
    const bf16* a_base = (bf16*)&g.a[{0, 0, 0, 0}];
    const bf16* b_base = (bf16*)&g.b[{0, 0, 0, 0}];
    const int a_row_stride = g.a.template stride<2>() * sizeof(bf16);  // = n * sizeof(bf16)
    const int b_row_stride = g.b.template stride<2>() * sizeof(bf16);  // = k * sizeof(bf16)
    const bf16* a_base_g = a_base;
    const bf16* b_base_g = b_base;
    i32x4 a_srsrc_curr = make_srsrc(a_base, g.M_total * a_row_stride, /*row_stride_bytes=*/0);
    i32x4 b_srsrc_curr = make_srsrc(b_base, g.M_total * b_row_stride, /*row_stride_bytes=*/0);
    int last_group_idx = -1;

    // LDS double-buffered per-warp slots (mirror grouped_gemm_bf16_kernel).
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
        const auto gl = compute_var_k_group_lookup(gt, tiles_per_group, s_offs);
        if (!gl.valid) continue;
        const int group_idx = gl.group_idx;
        const int local_tile = gl.local_tile;
        const int m_start_g = gl.m_start_g;
        const int M_g = gl.M_g;
        const int ki_g = gl.ki_g;

        const auto coords = compute_var_k_coords(local_tile, num_pid_m, num_pid_n, g.group_m);
        if (!coords.valid) continue;
        const int pid_m = coords.pid_m;
        const int pid_n = coords.pid_n;
        const int row = pid_m;
        const int col = pid_n;

        // Per-group shifted A/B base + bounded SRD: rebuild only on
        // group_idx transitions to amortise across this group's tiles.
        // m_start_g may be ANY byte-level value (not necessarily a K_STEP
        // multiple) — pointer arithmetic handles that, and non-swizzled
        // SRD byte-level OOB clamp handles the partial last K-tile when
        // M_g is not a K_STEP multiple.
        if (group_idx != last_group_idx) {
            a_base_g = (const bf16*)&g.a[{0, 0, m_start_g, 0}];
            b_base_g = (const bf16*)&g.b[{0, 0, m_start_g, 0}];
            a_srsrc_curr = make_srsrc(a_base_g, M_g * a_row_stride, /*row_stride_bytes=*/0);
            b_srsrc_curr = make_srsrc(b_base_g, M_g * b_row_stride, /*row_stride_bytes=*/0);
            last_group_idx = group_idx;
        }
        // Per-tile shifted GL views (copy + patch raw_ptr + rows_internal).
        // CRR a_coord uses row = K-axis (= per-group M_g range); the shifted
        // view lets coord (0,0,k,*) resolve to absolute byte = a_base_g + k*...
        // so we can pass k_offset_tiles=0.
        _gl a_gl_g = g.a;
        a_gl_g.raw_ptr = (bf16*)a_base_g;
        a_gl_g.rows_internal = M_g;
        _gl b_gl_g = g.b;
        b_gl_g.raw_ptr = (bf16*)b_base_g;
        b_gl_g.rows_internal = M_g;

        // Reset accumulators.
        zero(C_accum[0][0]); zero(C_accum[0][1]);
        zero(C_accum[1][0]); zero(C_accum[1][1]);

        // Reuse the shared GEMM body with CRR layout. m_subtile_A=0
        // (kernel n is group-uniform; row is the M-output tile),
        // group_idx=0 (B is 2D, no depth axis), k_offset_tiles=0 because
        // a_gl_g/b_gl_g raw_ptr already point at the group's K-axis start.
        device_gemm_tile_body<Layout::CRR, ST_A, ST_B, A_reg_t, B_reg_t>(
            a_gl_g, b_gl_g,
            /*m_subtile_A=*/0, /*group_idx=*/0, /*k_offset_tiles=*/0,
            As, Bs,
            swizzled_offsets_A, swizzled_offsets_B,
            a_srsrc_curr, b_srsrc_curr,
            a_base_g, b_base_g,
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

static inline void launch_grouped_var_k(grouped_var_k_layout_globals& g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    static bool attr_set = false;
    if (!attr_set) {
        hipFuncSetAttribute((void*)grouped_gemm_var_k_bf16_kernel,
                            hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
        attr_set = true;
    }
    grouped_gemm_var_k_bf16_kernel<<<dim3(NUM_CUS), g.block(), mem_size, g.stream>>>(g);
}

void dispatch_grouped_var_k(grouped_var_k_layout_globals g) {
    g.n = static_cast<int>(g.a.cols());     // kernel M-output dim
    g.k = static_cast<int>(g.b.cols());     // kernel N-output dim
    g.M_total = static_cast<int>(g.a.rows());

    g.fast_n = g.n;
    g.fast_k = g.k;
    g.bpr = kittens::ceil_div(g.n, BLOCK_SIZE);
    g.bpc = kittens::ceil_div(g.k, BLOCK_SIZE);

    if (g.bpr <= 0 || g.bpc <= 0 || g.G <= 0) return;

    launch_grouped_var_k(g);
}

#ifndef PRIMUS_TURBO_HK_INTEGRATION
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
        G, 0, 0, 0, gm, num_xcds, 0, 0,
        0, 0,  // fast_n, fast_k populated in dispatch.
    };
    dispatch_grouped_var_k(g);
}

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
#endif  // PRIMUS_TURBO_HK_INTEGRATION
