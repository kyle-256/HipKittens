#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

// =============================================================================
// Tuning constants (BF16-style: single block of constexpr ints + plain #defines
// for values that need stringification into inline asm).
// =============================================================================

// Block / tile dims
constexpr int BLOCK_SIZE        = 256;
constexpr int K_BLOCK           = 128;
constexpr int WARPS_M           = 2;
constexpr int WARPS_N           = 4;
constexpr int MIN_BLOCKS_PER_CU = 2;

// Block-swizzle (XCD-aware tile mapping)
constexpr int BLOCK_SWIZZLE_NUM_XCDS = 8;

// 4-wave RCR dispatch thresholds
constexpr int RCR_4WAVE_MIN_GRID  = 3200;
constexpr int RCR_4WAVE_MAX_K     = 8192;

constexpr int RCR_TWO_TILE_MIN_KI = 28;

// Derived block dims
constexpr int BLK = BLOCK_SIZE, BK = K_BLOCK;
constexpr int HB  = BLK / 2;
constexpr int _NUM_WARPS    = WARPS_M * WARPS_N;
constexpr int _NUM_THREADS  = _NUM_WARPS * WARP_THREADS;
constexpr int RBM = BLK / WARPS_M / 2;   // 64
constexpr int RBN = BLK / WARPS_N / 2;   // 32
constexpr int TAIL_BLOCK_M  = 16;
constexpr int TAIL_BLOCK_N  = 16;

// Stringification utilities for inline-asm wait counters
#define TK_STRINGIFY_IMPL(x) #x
#define TK_STRINGIFY(x) TK_STRINGIFY_IMPL(x)
#define TK_WAIT_LGKM(x)  asm volatile("s_waitcnt lgkmcnt(" TK_STRINGIFY(x) ")")
#define TK_WAIT_VMCNT(x) asm volatile("s_waitcnt vmcnt("  TK_STRINGIFY(x) ")")
#define TK_PRAGMA_UNROLL(x) _Pragma(TK_STRINGIFY(unroll x))

#define RCR_PREFETCH_LGKM       8
#define RCR_INIT0_VMCNT         4
#define RCR_INIT1_VMCNT         6
#define RCR_STEADY_VMCNT        8
#define RCR_EPILOGUE_VMCNT      4
#define RCR_TWO_TILE_MID_VMCNT  6
#ifndef RCR_KTAIL_VMCNT
#define RCR_KTAIL_VMCNT         8
#endif
#define RRR_PREFETCH_LGKM       8
#define RRR_INIT0_VMCNT         4
#define RRR_INIT1_VMCNT         6
#define RRR_STEADY_VMCNT        4
#define RRR_EPILOGUE_VMCNT      2
#define CRR_PREFETCH_LGKM       3
#define CRR_INIT0_VMCNT         2
#define CRR_INIT1_VMCNT         6
#define CRR_STEADY_VMCNT        4
#define CRR_EPILOGUE_VMCNT      2

// Per-layout main-loop unroll counts (stringified into #pragma unroll N).
#define RCR_MAIN_UNROLL 2
#define RRR_MAIN_UNROLL 4
#define CRR_MAIN_UNROLL 1
#define VARK_MAIN_UNROLL 1

#ifndef VARK_DROP_REDUNDANT_LGKM_DRAIN
#define VARK_DROP_REDUNDANT_LGKM_DRAIN 0
#endif

#ifndef VARK_SW_PIPE_HOIST_AHEAD
#define VARK_SW_PIPE_HOIST_AHEAD 0
#endif

#ifndef VARK_DROP_BARRIER_2
#define VARK_DROP_BARRIER_2 0
#endif

#ifndef VARK_HOIST_PREFETCH_INTO_HALF1
#define VARK_HOIST_PREFETCH_INTO_HALF1 0
#endif

#ifndef VARK_DROP_BARRIER_4
#define VARK_DROP_BARRIER_4 0
#endif

// Sched/wait barrier helpers
#define RRR_SCHED_BARRIER() __builtin_amdgcn_sched_barrier(0)
#define RCR_SCHED_BARRIER() __builtin_amdgcn_sched_barrier(0)
#define CRR_STEADY_MID_BARRIER() __builtin_amdgcn_s_barrier()
#define CRR_MMA_BEGIN() __builtin_amdgcn_s_setprio(1)
#define CRR_MMA_END()   __builtin_amdgcn_s_setprio(0)

using G = kittens::group<_NUM_WARPS>;
using _gl_fp8  = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_bf16 = gl<bf16, -1, -1, -1, -1>;

enum class Layout { RCR, RRR, CRR };

// Row-layout shared/register tiles (for A in RCR/RRR, B in RCR)
using ST_row    = st_fp8e4m3<HB, BK, st_16x128_s>;    // 128×128, M/N rows × K cols
using A_row_reg = rt_fp8e4m3<RBM, BK, row_l, rt_16x128_s>;
using B_row_reg = rt_fp8e4m3<RBN, BK, row_l, rt_16x128_s>;

// Col-layout register tiles (for B in RRR, A/B in CRR)
using A_col_reg = rt_fp8e4m3<BK, RBM, col_l, rt_128x16_s>;  // 128×64
using B_col_reg = rt_fp8e4m3<BK, RBN, col_l, rt_128x16_s>;  // 128×32

using ST_v2     = st_fp8e4m3<HB, BK, st_16x128_v2_s>;
using ST_v2a    = st_fp8e4m3<HB, BK, st_16x128_v2a_s>;

static_assert(sizeof(A_row_reg) == sizeof(A_col_reg));
static_assert(alignof(A_row_reg) == alignof(A_col_reg));
static_assert(sizeof(B_row_reg) == sizeof(B_col_reg));
static_assert(alignof(B_row_reg) == alignof(B_col_reg));

// Cooperative col-major load from a v2/v2a-swizzled FP8 LDS tile.
// Two `ds_read_b64_tr_b8` per lane per K_HALF (offset:0 + offset:1024).
template<typename RT, int K_HALF, typename ST>
__device__ __forceinline__ void load_col_from_st_half(
    RT& dst, const ST& tile, int col_start)
{
    const int laneid = kittens::laneid();
    const int row_off = ((laneid % 16) / 2) + ((laneid / 16) * 16);
    const int col_off = (laneid % 2) * 8;
    const uint32_t tile_base = reinterpret_cast<uintptr_t>(&tile.data[0]);

    constexpr int idx = K_HALF * 4;
    const int k_row = row_off + K_HALF * 64;

    const uint32_t stidx = k_row >> 4;
    const uint32_t base_k = tile_base + (stidx << 11) + (stidx << 7) + ((k_row & 15) << 7);
    const uint32_t sw_k   = (k_row & 7) << 4;

    #pragma unroll
    for (int j = 0; j < RT::width; j++) {
        const uint32_t nc = col_start + j * 16 + col_off;
        const uint32_t addr = base_k + (nc ^ sw_k);

        asm volatile(
            "ds_read_b64_tr_b8 %0, %2 offset:0\n"
            "ds_read_b64_tr_b8 %1, %2 offset:1024\n"
            : "=&v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx])),
              "=&v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx + 2]))
            : "v"(addr)
            : "memory"
        );
    }
}

template<typename RT, typename ST>
__device__ __forceinline__ void load_col_from_st(
    RT& dst, const ST& tile, int col_start)
{
    load_col_from_st_half<RT, 0>(dst, tile, col_start);
    load_col_from_st_half<RT, 1>(dst, tile, col_start);
}

__device__ __forceinline__ void rrr_mma(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
    const A_row_reg& a,
    const B_col_reg& b)
{
    mma_AB(acc, a, b, acc);
}

__device__ __forceinline__ void rcr_mma(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
    const A_row_reg& a,
    const B_row_reg& b)
{
    mma_ABt(acc, a, b, acc);
}

__device__ __forceinline__ void rcr_mma_32(
    rt_fl<RBM, RBN, col_l, rt_32x32_s>& acc,
    const rt_fp8e4m3<RBM, 64, row_l, rt_32x64_s>& a,
    const rt_fp8e4m3<RBN, 64, row_l, rt_32x64_s>& b)
{
    mma_ABt(acc, a, b, acc);
}

template<typename A_RT_32x64>
__device__ __forceinline__ void load_a_kt_32x64(
    A_RT_32x64& A_tile,
    i32x4 a_srsrc_kt,
    int M_warp_base,
    int row_lane,
    int k_lane_byte,
    int a_row_stride_bytes,
    uint32_t K_tail_base_bytes,
    bool b128_lo_valid,
    bool b128_hi_valid)
{
    constexpr uint32_t SENTINEL = 0xFFFF0000u;

    #pragma unroll
    for (int h = 0; h < A_RT_32x64::height; ++h) {
        const int A_row_idx = M_warp_base + h * 32 + row_lane;
        const uint32_t v_base = static_cast<uint32_t>(
            A_row_idx * a_row_stride_bytes +
            K_tail_base_bytes + k_lane_byte);
        const uint32_t v_lo = b128_lo_valid ? v_base : SENTINEL;
        const uint32_t v_hi = b128_hi_valid ? (v_base + 16) : SENTINEL;
        __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
            a_srsrc_kt, v_lo, 0, 0);
        __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
            a_srsrc_kt, v_hi, 0, 0);
        *reinterpret_cast<__uint128_t*>(&A_tile.tiles[h][0].data[0]) = v0;
        *reinterpret_cast<__uint128_t*>(&A_tile.tiles[h][0].data[4]) = v1;
    }
}

template<typename B_RT_32x64>
__device__ __forceinline__ void load_b_kt_32x64(
    B_RT_32x64& B_tile,
    i32x4 b_srsrc_kt,
    int N_warp_base,
    int row_lane,
    int k_lane_byte,
    int b_row_stride_bytes,
    uint32_t b_group_byte_base,
    uint32_t K_tail_base_bytes,
    bool b128_lo_valid,
    bool b128_hi_valid)
{
    constexpr uint32_t SENTINEL = 0xFFFF0000u;

    #pragma unroll
    for (int h_b = 0; h_b < B_RT_32x64::height; ++h_b) {
        const int B_row_idx_in_group = N_warp_base + h_b * 32 + row_lane;
        const uint32_t v_base = b_group_byte_base + static_cast<uint32_t>(
            B_row_idx_in_group * b_row_stride_bytes +
            K_tail_base_bytes + k_lane_byte);
        const uint32_t v_lo = b128_lo_valid ? v_base : SENTINEL;
        const uint32_t v_hi = b128_hi_valid ? (v_base + 16) : SENTINEL;
        __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
            b_srsrc_kt, v_lo, 0, 0);
        __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
            b_srsrc_kt, v_hi, 0, 0);
        *reinterpret_cast<__uint128_t*>(&B_tile.tiles[h_b][0].data[0]) = v0;
        *reinterpret_cast<__uint128_t*>(&B_tile.tiles[h_b][0].data[4]) = v1;
    }
}

__attribute__((used)) [[maybe_unused]] static __device__ void
__lever_d_round_b_force_instantiate_rcr_mma_32() {
    rt_fl<RBM, RBN, col_l, rt_32x32_s> dummy_acc{};
    rt_fp8e4m3<RBM, 64, row_l, rt_32x64_s> dummy_a{};
    rt_fp8e4m3<RBN, 64, row_l, rt_32x64_s> dummy_b{};
    rcr_mma_32(dummy_acc, dummy_a, dummy_b);

    i32x4 dummy_srsrc{};
    load_a_kt_32x64(dummy_a, dummy_srsrc,
                    /*M_warp_base=*/0,
                    /*row_lane=*/0,
                    /*k_lane_byte=*/0,
                    /*a_row_stride_bytes=*/0,
                    /*K_tail_base_bytes=*/0u,
                    /*b128_lo_valid=*/true,
                    /*b128_hi_valid=*/true);
    load_b_kt_32x64(dummy_b, dummy_srsrc,
                    /*N_warp_base=*/0,
                    /*row_lane=*/0,
                    /*k_lane_byte=*/0,
                    /*b_row_stride_bytes=*/0,
                    /*b_group_byte_base=*/0u,
                    /*K_tail_base_bytes=*/0u,
                    /*b128_lo_valid=*/true,
                    /*b128_hi_valid=*/true);
}

__attribute__((used)) [[maybe_unused]] static __device__ void
__lever_d_round_b_force_instantiate_st_32x64() {
    using ST_32x64 = st_fp8e4m3<HB, 64, st_32x64_s>;
    __shared__ ST_32x64 dummy_st;

    // Touch the type's static-member infrastructure to force full
    // template-parameter validation. Static-asserts mirror the
    // kittens-internal checks in ``st<>::`` body.
    static_assert(ST_32x64::rows == HB, "ST_32x64 rows should equal HB=128");
    static_assert(ST_32x64::cols == 64, "ST_32x64 cols should equal 64");
    static_assert(ST_32x64::underlying_subtile_rows == 32,
                  "ST_32x64 underlying subtile rows should equal 32");
    static_assert(ST_32x64::underlying_subtile_cols == 64,
                  "ST_32x64 underlying subtile cols should equal 64");
    static_assert(ST_32x64::underlying_subtile_bytes_per_thread == 16,
                  "ST_32x64 should dispatch the fp8 bytes_per_thread=16 branch");

    // Exercise the swizzle functor at compile time via a device call
    // path; LLVM DCE removes the dead reference after instantiation.
    (void)ST_32x64::swizzle({0, 0});
    (void)dummy_st;
}

__device__ __forceinline__ void crr_mma(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
    const A_col_reg& a,
    const B_col_reg& b)
{
    const auto& a_row = reinterpret_cast<const A_row_reg&>(a);
    mma_AB(acc, a_row, b, acc);
}

__device__ __forceinline__ float load_fp8_scalar(const _gl_fp8& src, int row, int col) {
    const uint32_t buffer_size = src.batch() * src.depth() * src.rows() * src.cols() * sizeof(fp8e4m3);
    const std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(src.raw_ptr);
    const std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    const buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);
    const i32x4 srsrc = std::bit_cast<i32x4>(br);
    const uint32_t voffset = (row * src.cols() + col) * sizeof(fp8e4m3);
    const uint8_t bits = llvm_amdgcn_raw_buffer_load_b8(srsrc, voffset, 0, 0);
    return base_types::convertor<float, fp8e4m3>::convert(std::bit_cast<fp8e4m3>(bits));
}

__device__ __forceinline__ float load_bf16_scalar(const _gl_bf16& src, int row, int col) {
    const uint32_t buffer_size = src.batch() * src.depth() * src.rows() * src.cols() * sizeof(bf16);
    const std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(src.raw_ptr);
    const std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    const buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);
    const i32x4 srsrc = std::bit_cast<i32x4>(br);
    const uint32_t voffset = (row * src.cols() + col) * sizeof(bf16);
    const uint16_t bits = llvm_amdgcn_raw_buffer_load_b16(srsrc, voffset, 0, 0);
    return base_types::convertor<float, bf16>::convert(std::bit_cast<bf16>(bits));
}

__device__ __forceinline__ void store_bf16_scalar(const _gl_bf16& dst, int row, int col, float value) {
    const uint32_t buffer_size = dst.batch() * dst.depth() * dst.rows() * dst.cols() * sizeof(bf16);
    const std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(dst.raw_ptr);
    const std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    const buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);
    const i32x4 srsrc = std::bit_cast<i32x4>(br);
    const uint32_t voffset = (row * dst.cols() + col) * sizeof(bf16);
    const bf16 v = base_types::convertor<bf16, float>::convert(value);
    llvm_amdgcn_raw_buffer_store_b16(std::bit_cast<uint16_t>(v), srsrc, voffset, 0, 0);
}

// Per-group scalar FP8 load. ``b`` for grouped FP8 is logically
// [batch=1, G, N, K]; the 4D coord lets `grouped_tail_kernel` index B at
// (group_idx, row, col).
__device__ __forceinline__ float load_fp8_scalar_grp(const _gl_fp8& src, int g_idx, int row, int col) {
    const uint32_t buffer_size = src.batch() * src.depth() * src.rows() * src.cols() * sizeof(fp8e4m3);
    const std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(src.raw_ptr);
    const std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    const buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);
    const i32x4 srsrc = std::bit_cast<i32x4>(br);
    const uint32_t idx = ((0 * src.depth() + g_idx) * src.rows() + row) * src.cols() + col;
    const uint32_t voffset = idx * sizeof(fp8e4m3);
    const uint8_t bits = llvm_amdgcn_raw_buffer_load_b8(srsrc, voffset, 0, 0);
    return base_types::convertor<float, fp8e4m3>::convert(std::bit_cast<fp8e4m3>(bits));
}

// Packed 8 × fp8e4m3 = 8 bytes for vectorised tail-kernel K-loop. The
// HIP compiler emits a single `global_load_dwordx2` for a load through
// this type when the source pointer is 8-byte aligned, replacing 8
// separate scalar fp8 loads (8× fewer VMEM transactions). Used by the
// RCR fast path inside `grouped_tail_kernel` where both operands are
// stride-1 in K. Extraction via ``convertor<float4, fp8e4m3_4>`` gives
// 8 fp32 accumulator inputs per 8-byte load.
struct alignas(8) fp8e4m3_8 {
    fp8e4m3_4 lo, hi;
};

template<int N_THREADS, ducks::st::all ST, ducks::gl::all GL>
__device__ __forceinline__ void prefill_transpose_swizzled_offsets(
    ST& dst, const GL& src, uint32_t* swizzled_offsets)
{
    using T = typename ST::dtype;

    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    constexpr int memcpy_per_tile =
        ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);
    static_assert(
        ST::rows * ST::cols * sizeof(T) >= bytes_per_warp,
        "shared tile must be at least 1024 bytes"
    );

    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int laneid = kittens::laneid();
    const int warpid = kittens::warpid() % num_warps;
    const int row_stride = src.template stride<2>();

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {
        const int lane_byte_offset =
            (laneid * bytes_per_thread) +
            (warpid * bytes_per_warp) +
            (i * num_warps * bytes_per_warp);
        const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
        const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
        const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
        const int subtile_lane_byte_offset =
            lane_byte_offset % ST::underlying_subtile_bytes;

        const int row =
            subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
        const int col =
            (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);

        const uint32_t swizzled_shared_byte_offset = dst.swizzle({row, col});
        const int shared_row =
            swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes;
        const int shared_col =
            (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);

        const int transposed_global_row =
            shared_col + subtile_col * ST::underlying_subtile_cols;
        const int transposed_global_col =
            shared_row + subtile_row * ST::underlying_subtile_rows;

        swizzled_offsets[i] =
            (transposed_global_row * row_stride + transposed_global_col) * sizeof(T);
    }

    if constexpr (
        memcpy_per_tile * (bytes_per_thread * N_THREADS) !=
        ST::rows * ST::cols * sizeof(T)
    ) {
        constexpr int leftover_bytes =
            ST::rows * ST::cols * sizeof(T) -
            memcpy_per_tile * (bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;
        constexpr int leftover_warps = leftover_threads / kittens::WARP_THREADS;

        if (warpid < leftover_warps) {
            const int lane_byte_offset =
                (laneid * bytes_per_thread) +
                (warpid * bytes_per_warp) +
                (memcpy_per_tile * num_warps * bytes_per_warp);
            const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
            const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
            const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
            const int subtile_lane_byte_offset =
                lane_byte_offset % ST::underlying_subtile_bytes;

            const int row =
                subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
            const int col =
                (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);

            const uint32_t swizzled_shared_byte_offset = dst.swizzle({row, col});
            const int shared_row =
                swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes;
            const int shared_col =
                (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);

            const int transposed_global_row =
                shared_col + subtile_col * ST::underlying_subtile_cols;
            const int transposed_global_col =
                shared_row + subtile_row * ST::underlying_subtile_rows;

            swizzled_offsets[memcpy_per_tile] =
                (transposed_global_row * row_stride + transposed_global_col) * sizeof(T);
        }
    }
}

template<int N_THREADS,
         ducks::st::all ST,
         ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>>
__device__ __forceinline__ void load_transpose(
    ST& dst, const GL& src, const COORD& idx, const uint32_t* swizzled_offsets)
{
    using T = typename ST::dtype;

    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    constexpr int memcpy_per_tile =
        ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);
    static_assert(
        ST::rows * ST::cols * sizeof(T) >= bytes_per_warp,
        "shared tile must be at least 1024 bytes"
    );

    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int warpid = kittens::warpid() % num_warps;
    const int row_stride = src.template stride<2>();

    coord<> unit_coord(
        idx.template dim<0>(),
        idx.template dim<1>(),
        idx.template dim<3>() * ST::cols,
        idx.template dim<2>() * ST::rows
    );
    T* global_ptr = (T*)&src[unit_coord];
    i32x4 srsrc = make_srsrc(global_ptr, row_stride * ST::cols * sizeof(T));

    const uintptr_t lds_tile_base = reinterpret_cast<uintptr_t>(&dst.data[0]);

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {
        const int warp_linear_offset =
            (warpid * bytes_per_warp) + (i * num_warps * bytes_per_warp);
        const int lds_subtile_id = warp_linear_offset / ST::underlying_subtile_bytes;
        uintptr_t lds_addr =
            lds_tile_base + warp_linear_offset + lds_subtile_id * ST::subtile_padding;
        as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

        llvm_amdgcn_raw_buffer_load_lds(
            srsrc,
            lds_ptr,
            bytes_per_thread,
            swizzled_offsets[i],
            0,
            0,
            static_cast<int>(coherency::cache_all)
        );
    }

    if constexpr (
        memcpy_per_tile * (bytes_per_thread * N_THREADS) !=
        ST::rows * ST::cols * sizeof(T)
    ) {
        constexpr int leftover_bytes =
            ST::rows * ST::cols * sizeof(T) -
            memcpy_per_tile * (bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;
        constexpr int leftover_warps = leftover_threads / kittens::WARP_THREADS;

        if (warpid < leftover_warps) {
            const int warp_linear_offset =
                (warpid * bytes_per_warp) +
                (memcpy_per_tile * num_warps * bytes_per_warp);
            const int lds_subtile_id =
                warp_linear_offset / ST::underlying_subtile_bytes;
            uintptr_t lds_addr =
                lds_tile_base + warp_linear_offset + lds_subtile_id * ST::subtile_padding;
            as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

            llvm_amdgcn_raw_buffer_load_lds(
                srsrc,
                lds_ptr,
                bytes_per_thread,
                swizzled_offsets[memcpy_per_tile],
                0,
                0,
                static_cast<int>(coherency::cache_all)
            );
        }
    }
}

// P19 Dev A — m0-broadcast hoist for the 8-wave RCR DTL loads.
//
// Drop-in replacement for the kittens 4-arg G::load(dst, src, idx, swizzled_offsets)
// used inside the 8-wave RCR main loop. Behaviour:
//   * Pre-computes the per-iter LDS byte offset as an SGPR (readfirstlane
//     forces wave-uniform residency in SGPR class), bypassing LLVM's
//     tendency to recompute the address through a VGPR + v_readfirstlane
//     immediately before each DTL store.
//   * Issues `s_mov m0, <sgpr>` + `buffer_load_dwordx4 ... offen lds` via
//     inline asm. The LLVM intrinsic
//     `__builtin_amdgcn_raw_buffer_load_lds` lets the scheduler CSE the
//     m0 plumbing back into vector ops; inline asm forecloses that.
//   * Vector destination operand is a phantom — `buffer_load_dwordx4 ... lds`
//     does NOT write a VGPR, the operand merely satisfies LLVM's asm-binding
//     contract and gets dead-code-eliminated downstream.
//   * Preserves the leftover-warps tail handling from the kittens helper so
//     odd memcpy_per_tile counts still drain.
//
// Mirrors `kittens::load<2, false, ST, GL, COORD, N_THREADS>` from
// include/ops/warp/memory/tile/global_to_shared.cuh:187-249 (the 4-arg
// variant taking pre-computed swizzled_offsets).
template<int N_THREADS,
         ducks::st::all ST,
         ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>>
__device__ __forceinline__ void rcr_8w_load_hoist(
    ST& dst, const GL& src, const COORD& idx,
    const uint32_t* __restrict__ swizzled_offsets)
{
    using T = typename ST::dtype;

    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_warp   = bytes_per_thread * kittens::WARP_THREADS;
    constexpr int memcpy_per_tile  =
        ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);
    static_assert(
        ST::rows * ST::cols * sizeof(T) >= bytes_per_warp,
        "shared tile must be at least 1024 bytes"
    );

    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int warpid = kittens::warpid() % num_warps;

    coord<> unit_coord = idx.template unit_coord<2, 3>();
    T* tensor_base = (T*)src.raw_ptr;
    T* global_ptr  = (T*)&src[unit_coord];
    // Full-tensor SRD: bound the ENTIRE source (A or B) tensor so that
    // OOB column-tile loads — induced by ``bpc = ceil_div(n, BLK)`` for
    // partial last-N tiles — clamp to 0 instead of touching unmapped
    // pages. The OG ``make_srsrc(global_ptr, ST::rows * row_stride *
    // sizeof(T))`` was tile-local: ``global_ptr`` past tensor end took
    // the SRD bound into unmapped memory, faulting on the swizzled
    // raw_buffer_load_lds path (verified on N=384 K=2048 RCR/RRR/CRR
    // cumulative test, MI350X). The full-tensor SRD makes every load
    // safe regardless of where ``unit_coord`` points; OOB bytes are
    // returned as 0, the masked C store drops them.
    const uint32_t total_bytes = static_cast<uint32_t>(
        size_t(src.batch()) * size_t(src.depth()) *
        size_t(src.rows())  * size_t(src.cols())  * sizeof(T));
    i32x4 srsrc = make_srsrc(tensor_base, total_bytes);
    // Tile byte offset from tensor base, hoisted into an SGPR.
    // Used as SOFFSET of the buffer_load_dwordx4 to recover the per-tile
    // base. Wave-uniform because ``unit_coord`` only depends on (br, bc,
    // sub-tile, k), all of which are uniform.
    const uint32_t tile_byte_offset = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(global_ptr) -
                              reinterpret_cast<uintptr_t>(tensor_base)));

    const uintptr_t lds_tile_base =
        reinterpret_cast<uintptr_t>(&dst.data[0]);

    // A1: hoist scalar per-pass LDS-byte ramp into SGPRs in the prologue.
    // Each lds_addr below is wave-uniform (warpid + i are uniform), so we
    // make that explicit via readfirstlane and keep the value in SGPR.
    uint32_t lds_addrs[memcpy_per_tile + 1];  // +1 leftover slot (may be unused)
    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; ++i) {
        const int warp_linear_offset =
            (warpid * bytes_per_warp) + (i * num_warps * bytes_per_warp);
        const int lds_subtile_id = warp_linear_offset / ST::underlying_subtile_bytes;
        const uint32_t off32 = static_cast<uint32_t>(
            lds_tile_base + warp_linear_offset +
            lds_subtile_id * ST::subtile_padding);
        lds_addrs[i] = __builtin_amdgcn_readfirstlane(off32);
    }

    // A2: full inline-asm DTL — set m0 from the SGPR-hoisted per-pass offset
    // and issue buffer_load_dwordx4 ... offen lds. Operand binding mirrors
    // P18 Dev A's working 4-wave pattern (see rcr_4wave_dynamic.inc).
    // %0 = s "lds_off" (SGPR), %1 = v "goff" (per-lane VGPR offset),
    // %2 = s "srsrc" (4-SGPR buffer resource), %3 = s "tile_byte_offset"
    // (SGPR SOFFSET) — the per-tile base offset added to V_VOFFSET.
    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; ++i) {
        const uint32_t lds_off = lds_addrs[i];
        const uint32_t goff    = swizzled_offsets[i];
        asm volatile(
            "s_mov_b32 m0, %0\n\t"
            "buffer_load_dwordx4 %1, %2, %3 offen lds\n\t"
            :
            : "s"(lds_off), "v"(goff), "s"(srsrc), "s"(tile_byte_offset)
            : "memory");
    }

    if constexpr (memcpy_per_tile * (bytes_per_thread * N_THREADS) !=
                  ST::rows * ST::cols * sizeof(T)) {
        constexpr int leftover_bytes =
            ST::rows * ST::cols * sizeof(T) -
            memcpy_per_tile * (bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;
        constexpr int leftover_warps   = leftover_threads / kittens::WARP_THREADS;
        if (warpid < leftover_warps) {
            const int warp_linear_offset =
                (warpid * bytes_per_warp) +
                (memcpy_per_tile * num_warps * bytes_per_warp);
            const int lds_subtile_id =
                warp_linear_offset / ST::underlying_subtile_bytes;
            const uint32_t off32 = static_cast<uint32_t>(
                lds_tile_base + warp_linear_offset +
                lds_subtile_id * ST::subtile_padding);
            const uint32_t lds_off = __builtin_amdgcn_readfirstlane(off32);
            const uint32_t goff    = swizzled_offsets[memcpy_per_tile];
            asm volatile(
                "s_mov_b32 m0, %0\n\t"
                "buffer_load_dwordx4 %1, %2, %3 offen lds\n\t"
                :
                : "s"(lds_off), "v"(goff), "s"(srsrc), "s"(tile_byte_offset)
                : "memory");
        }
    }
}

struct layout_globals {
    _gl_fp8 a, b;
    _gl_bf16 c;
    float scale_a, scale_b;
    hipStream_t stream;
    int m, n, k;
    int bpr, bpc, ki;
    int fast_m, fast_n, fast_k;
    int group_m;
    const float* dscale_a;
    const float* dscale_b;
    dim3 grid()  { return dim3(bpr * bpc); }
    dim3 block() { return dim3(_NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

// Resolve the combined per-tensor scale at kernel epilogue time. When the
// host wrapper passed device-side scale tensors (dscale_{a,b} non-null) we
// load them with a scalar global-memory read; otherwise we fall back to
// the host-known floats baked into `g`. The branch is uniform across the
// wave so the compiler keeps it scalar; the load itself is one b32 from
// global memory and hits cache after the first wave issues it.
__device__ __forceinline__ float resolve_combined_scale(const layout_globals &g) {
    const float sa = g.dscale_a ? *g.dscale_a : g.scale_a;
    const float sb = g.dscale_b ? *g.dscale_b : g.scale_b;
    return sa * sb;
}

__device__ __forceinline__ int gemm_chiplet_swizzle_bid(int bid, int num_wgs) {
    if (num_wgs >= BLOCK_SWIZZLE_NUM_XCDS &&
        (num_wgs % BLOCK_SWIZZLE_NUM_XCDS) == 0) {
        return
            (bid % BLOCK_SWIZZLE_NUM_XCDS) *
                (num_wgs / BLOCK_SWIZZLE_NUM_XCDS) +
            (bid / BLOCK_SWIZZLE_NUM_XCDS);
    }
    return bid;
}

__device__ __forceinline__ void gemm_compute_block_coords(
    int bid, int bpr, int bpc, int group_m, int &br, int &bc) {
    bid = gemm_chiplet_swizzle_bid(bid, gridDim.x);
    const int num_wgid_in_group = group_m * bpc;
    const int group_id = bid / num_wgid_in_group;
    const int first_pid_m = group_id * group_m;
    const int group_size_m =
        (first_pid_m + group_m <= bpr)
            ? group_m
            : (bpr - first_pid_m);
    if (group_size_m <= 0) {
        br = bpr;
        bc = bpc;
        return;
    }
    br = first_pid_m + ((bid % num_wgid_in_group) % group_size_m);
    bc = (bid % num_wgid_in_group) / group_size_m;
}

// =============================================================================
// store_c_tile_n_masked — column-masked C-store for partial-N tiles (FP8 dense).
//
// Mirrors the BF16 helper in `analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`.
// Templated on the rt accum type because the FP8 file has two main kernels with
// different RC tile widths (rcr_4w uses RBN=64; the generic 8-wave gemm_kernel
// uses RBN=32).
//
// 3-way fast path:
//   * fully OOB tile-column (n0 >= n_limit): no-op (block won't be writing).
//   * fully in-bounds (n1 <= n_limit): forward to the original `store(...)`.
//     Aligned shapes pay zero overhead — same instruction sequence as before.
//   * partial: lane-level skip on per-column OOB. Each lane checks its column
//     index against `n_limit`. C-tile is bf16 in global memory.
//
// MFMA still writes a result to RC's OOB columns, but those columns are
// dropped here and never reach global memory. SRD bounds for the global B
// (set up by the caller as full-tensor bounds) clamp OOB B-loads to 0, so
// the MFMA accumulator just contains a partial-sum garbage that we discard.
// =============================================================================
template<ducks::gl::all GL, ducks::rt::all RT>
__device__ __forceinline__ void store_c_tile_n_masked(
    const GL& g_c, const RT& src,
    int r_tile, int c_tile, int n_limit) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;
    constexpr int packing = base_types::packing<typename RT::dtype>::num();
    static_assert(std::is_same_v<U, bf16>, "C is bf16 global");

    const int n0 = c_tile * RT::cols;
    const int n1 = n0 + RT::cols;
    if (n0 >= n_limit) return;
    if (n1 <= n_limit) {
        store(g_c, src, {0, 0, r_tile, c_tile});
        return;
    }

    constexpr int axis = 2;
    U* dst_ptr = (U*)&g_c[(coord<RT>{0, 0, r_tile, c_tile}
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

template<ducks::gl::all GL, ducks::rt::all RT>
__device__ __forceinline__ void store_c_tile_mn_masked_grouped(
    const GL& g_c, const RT& src,
    int group_idx, int r_tile, int c_tile,
    int m_limit, int n_limit) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;
    constexpr int packing = base_types::packing<typename RT::dtype>::num();
    static_assert(std::is_same_v<U, bf16>, "C is bf16 global");

    const int m0 = r_tile * RT::rows;
    const int m1 = m0 + RT::rows;
    const int n0 = c_tile * RT::cols;
    const int n1 = n0 + RT::cols;

    if (m0 >= m_limit || n0 >= n_limit) return;
    if (m1 <= m_limit && n1 <= n_limit) {
        store(g_c, src, {0, group_idx, r_tile, c_tile});
        return;
    }

    constexpr int axis = 2;
    U* dst_ptr = (U*)&g_c[(coord<RT>{0, group_idx, r_tile, c_tile}
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


// Runtime K-specialization: when KI_HINT>0 it matches g.ki exactly, enabling
// the compiler to fully unroll or uniformly unroll the main loop without
// branch overhead and with register allocation tuned to the known loop count.
template<Layout L, int KI_HINT = 0>
__global__ __launch_bounds__(_NUM_THREADS, MIN_BLOCKS_PER_CU)
void gemm_kernel(const layout_globals g) {
    int bid = blockIdx.x;
    int br, bc;
    gemm_compute_block_coords(bid, g.bpr, g.bpc, g.group_m, br, bc);
    const int ki_dyn = (KI_HINT > 0) ? KI_HINT : g.ki;
    if (br >= g.bpr || bc >= g.bpc || ki_dyn <= 0) {
        return;
    }
    int wm = warpid() / WARPS_N, wn = warpid() % WARPS_N;

    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;
    zero(cA); zero(cB); zero(cC); zero(cD);

    if constexpr (L == Layout::RCR) {
        using ST_rcr = ST_v2;
        __shared__ ST_rcr As[2][2];
        __shared__ ST_rcr Bs[2][2];
        A_row_reg a;
        B_row_reg b0, b1;

        constexpr int bpt = ST_rcr::underlying_subtile_bytes_per_thread;
        constexpr int bpm = bpt * _NUM_THREADS;
        constexpr int mpt = ST_rcr::rows * ST_rcr::cols * sizeof(fp8e4m3) / bpm;
        uint32_t soA[mpt], soB[mpt];
        G::prefill_swizzled_offsets(As[0][0], g.a, soA);
        G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

        auto a_co = [&](int s, int k) -> coord<ST_rcr> { return {0, 0, s, k}; };
        auto b_co = [&](int s, int k) -> coord<ST_rcr> { return {0, 0, s, k}; };

        auto load_a = [&](A_row_reg& dst, ST_rcr& tile, int wi) {
            auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
            load(dst, sub);
        };
        auto load_b = [&](B_row_reg& dst, ST_rcr& tile, int wi) {
            auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
            load(dst, sub);
        };

        auto b_tile = [&](int stage, int which) -> ST_rcr& {
            return Bs[stage][which];
        };

        int tic = 0, toc = 1;
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 0), g.b, b_co(bc*2,   0), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0], g.a, a_co(br*2,   0), soA);
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 1), g.b, b_co(bc*2+1, 0), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[tic][1], g.a, a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(RCR_INIT0_VMCNT);
        __builtin_amdgcn_s_barrier();

        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(toc, 0), g.b, b_co(bc*2,   1), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[toc][0], g.a, a_co(br*2,   1), soA);
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(toc, 1), g.b, b_co(bc*2+1, 1), soB);

        TK_WAIT_VMCNT(RCR_INIT1_VMCNT);
        __builtin_amdgcn_s_barrier();

        if ((ki_dyn & 1) == 0 && ki_dyn >= RCR_TWO_TILE_MIN_KI) {
            auto main_loop_iter = [&](int tile) {
                load_b(b0, Bs[0][0], wn);
                load_a(a, As[0][0], wm);
                rcr_8w_load_hoist<_NUM_THREADS>(As[1][1], g.a, a_co(br*2+1, tile+1), soA);
                TK_WAIT_LGKM(RCR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();

                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1); mma_ABt(cA, a, b0, cA); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

                load_b(b1, Bs[0][1], wn);
                rcr_8w_load_hoist<_NUM_THREADS>(Bs[0][0], g.b, b_co(bc*2, tile+2), soB);
                __builtin_amdgcn_s_barrier();

                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1); mma_ABt(cB, a, b1, cB); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();

                load_a(a, As[0][1], wm);
                rcr_8w_load_hoist<_NUM_THREADS>(As[0][0], g.a, a_co(br*2, tile+2), soA);
                __builtin_amdgcn_s_barrier();

                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1); mma_ABt(cC, a, b0, cC); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

                load_b(b0, Bs[1][0], wn);
                rcr_8w_load_hoist<_NUM_THREADS>(Bs[0][1], g.b, b_co(bc*2+1, tile+2), soB);
                TK_WAIT_VMCNT(RCR_TWO_TILE_MID_VMCNT); __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_s_setprio(1); mma_ABt(cD, a, b1, cD); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();

                load_a(a, As[1][0], wm);
                rcr_8w_load_hoist<_NUM_THREADS>(As[0][1], g.a, a_co(br*2+1, tile+2), soA);
                TK_WAIT_LGKM(RCR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();

                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1); mma_ABt(cA, a, b0, cA); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

                load_b(b1, Bs[1][1], wn);
                rcr_8w_load_hoist<_NUM_THREADS>(Bs[1][0], g.b, b_co(bc*2, tile+3), soB);
                __builtin_amdgcn_s_barrier();

                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1); mma_ABt(cB, a, b1, cB); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();

                load_a(a, As[1][1], wm);
                rcr_8w_load_hoist<_NUM_THREADS>(As[1][0], g.a, a_co(br*2, tile+3), soA);
                __builtin_amdgcn_s_barrier();

                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1); mma_ABt(cC, a, b0, cC); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

                rcr_8w_load_hoist<_NUM_THREADS>(Bs[1][1], g.b, b_co(bc*2+1, tile+3), soB);
                TK_WAIT_VMCNT(RCR_TWO_TILE_MID_VMCNT); __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_s_setprio(1); mma_ABt(cD, a, b1, cD); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();
            };

            TK_PRAGMA_UNROLL(RCR_MAIN_UNROLL)
            for (int tile = 0; tile < ki_dyn - 2; tile += 2) {
                main_loop_iter(tile);
            }
            TK_WAIT_VMCNT(0);
            __builtin_amdgcn_s_barrier();
        } else
        {
        TK_PRAGMA_UNROLL(RCR_MAIN_UNROLL)
        for (int k = 0; k < ki_dyn - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, b_tile(tic, 0), wn);
            load_a(a, As[tic][0], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], g.a, a_co(br*2+1, k+1), soA);
            TK_WAIT_LGKM(RCR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

            load_b(b1, b_tile(tic, 1), wn);
            rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 0), g.b, b_co(bc*2, k+2), soB);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0], g.a, a_co(br*2, k+2), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

            rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 1), g.b, b_co(bc*2+1, k+2), soB);
            TK_WAIT_VMCNT(RCR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_setprio(1); rcr_mma(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }
        }

        {
            load_b(b0, b_tile(tic, 0), wn);
            load_a(a, As[tic][0], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], g.a, a_co(br*2+1, ki_dyn-1), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

            load_b(b1, b_tile(tic, 1), wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(RCR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, b_tile(toc, 0), wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();
            tic ^= 1; toc ^= 1;
        }

        {
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, b_tile(tic, 1), wn);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            rcr_mma(cC, a, b0);
            rcr_mma(cD, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

    } else if constexpr (L == Layout::RRR) {
        __shared__ ST_row As[2][2];
        __shared__ ST_v2 Bs[2][2];
        A_row_reg a;
        B_col_reg b0, b1;

        constexpr int bptA = ST_row::underlying_subtile_bytes_per_thread;
        constexpr int bpmA = bptA * _NUM_THREADS;
        constexpr int mptA = ST_row::rows * ST_row::cols * sizeof(fp8e4m3) / bpmA;
        uint32_t soA[mptA];
        G::prefill_swizzled_offsets(As[0][0], g.a, soA);

        constexpr int bptB =
            ST_v2::underlying_subtile_bytes_per_thread;
        constexpr int bpmB = bptB * _NUM_THREADS;
        constexpr int mptB =
            ST_v2::rows * ST_v2::cols * sizeof(fp8e4m3) / bpmB;
        uint32_t soB[mptB];
        G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

        auto a_co = [&](int s, int k) -> coord<ST_row> { return {0, 0, s, k}; };
        auto b_co = [&](int s, int k) -> coord<ST_v2> { return {0, 0, k, s}; };

        auto load_a = [&](A_row_reg& dst, ST_row& tile, int wi) {
            auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
            load(dst, sub);
        };
        auto load_b = [&](B_col_reg& dst, ST_v2& tile, int wi) {
            load_col_from_st(dst, tile, wi * RBN);
        };

        int tic = 0, toc = 1;
        G::load(Bs[tic][0], g.b, b_co(bc*2,   0), soB);
        G::load(As[tic][0], g.a, a_co(br*2,   0), soA);
        G::load(Bs[tic][1], g.b, b_co(bc*2+1, 0), soB);
        G::load(As[tic][1], g.a, a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(RRR_INIT0_VMCNT);
        __builtin_amdgcn_s_barrier();

        G::load(Bs[toc][0], g.b, b_co(bc*2,   1), soB);
        G::load(As[toc][0], g.a, a_co(br*2,   1), soA);
        G::load(Bs[toc][1], g.b, b_co(bc*2+1, 1), soB);

        TK_WAIT_VMCNT(RRR_INIT1_VMCNT);
        __builtin_amdgcn_s_barrier();

        TK_PRAGMA_UNROLL(RRR_MAIN_UNROLL)
        for (int k = 0; k < ki_dyn - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, k+1), soA);
            TK_WAIT_LGKM(RRR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();

            load_b(b1, Bs[tic][1], wn);
            G::load(Bs[tic][0], g.b, b_co(bc*2, k+2), soB);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            rrr_mma(cB, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            G::load(Bs[tic][1], g.b, b_co(bc*2+1, k+2), soB);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();

            G::load(As[tic][0], g.a, a_co(br*2, k+2), soA);
            TK_WAIT_VMCNT(RRR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            rrr_mma(cD, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, ki_dyn-1), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            rrr_mma(cB, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[toc][0], wn);
            TK_WAIT_VMCNT(RRR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();
            tic ^= 1; toc ^= 1;
        }

        {
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            rrr_mma(cB, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            rrr_mma(cC, a, b0);
            rrr_mma(cD, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

    } else if constexpr (L == Layout::CRR) {
        using ST_crr_a = ST_v2a;
        using ST_crr_b = ST_v2;
        __shared__ ST_crr_a As[2][2];
        __shared__ ST_crr_b Bs[2][2];
        A_col_reg a;
        B_col_reg b0, b1;

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

        auto load_a = [&](A_col_reg& dst, ST_crr_a& tile, int wi) {
            load_col_from_st(dst, tile, wi * RBM);
        };
        auto load_b = [&](B_col_reg& dst, ST_crr_b& tile, int wi) {
            load_col_from_st(dst, tile, wi * RBN);
        };

        int tic = 0, toc = 1;
        global_load_b(Bs[tic][0], bc*2,   0);
        global_load_a(As[tic][0], br*2,   0);
        global_load_b(Bs[tic][1], bc*2+1, 0);
        global_load_a(As[tic][1], br*2+1, 0);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(CRR_INIT0_VMCNT);
        __builtin_amdgcn_s_barrier();

        global_load_b(Bs[toc][0], bc*2,   1);
        global_load_a(As[toc][0], br*2,   1);
        global_load_b(Bs[toc][1], bc*2+1, 1);

        TK_WAIT_VMCNT(CRR_INIT1_VMCNT);
        __builtin_amdgcn_s_barrier();


        TK_PRAGMA_UNROLL(CRR_MAIN_UNROLL)
        for (int k = 0; k < ki_dyn - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, Bs[tic][0], wn);
            load_b(b1, Bs[tic][1], wn);
            load_a(a, As[tic][0], wm);
            global_load_a(As[toc][1], br*2+1, k+1);
            TK_WAIT_LGKM(CRR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cA, a, b0);
            crr_mma(cB, a, b1);
            CRR_MMA_END();
            CRR_STEADY_MID_BARRIER();

            load_a(a, As[tic][1], wm);
            global_load_a(As[tic][0], br*2, k+2);
            global_load_b(Bs[tic][1], bc*2+1, k+2);
            TK_WAIT_VMCNT(CRR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cC, a, b0);
            crr_mma(cD, a, b1);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();
            global_load_b(Bs[tic][0], bc*2, k+2);
        }

        {
            load_b(b0, Bs[tic][0], wn);
            const auto b0_keep = b0;
            load_a(a, As[tic][0], wm);
            global_load_a(As[toc][1], br*2+1, ki_dyn-1);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cA, a, b0_keep);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
            const auto b1_keep = b1;
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cB, a, b1_keep);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(CRR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cC, a, b0_keep);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[toc][0], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cD, a, b1_keep);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();
            tic ^= 1; toc ^= 1;
        }

        {
            const auto b0_keep = b0;
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cA, a, b0_keep);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
            const auto b1_keep = b1;
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cB, a, b1_keep);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cC, a, b0_keep);
            crr_mma(cD, a, b1_keep);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();
        }    }

    const float combined_scale = resolve_combined_scale(g);
    mul(cA, cA, combined_scale);
    mul(cB, cB, combined_scale);
    mul(cC, cC, combined_scale);
    mul(cD, cD, combined_scale);

    // Store Output
    if (wm == 0) __builtin_amdgcn_s_barrier();
    store_c_tile_n_masked(g.c, cA, br*WARPS_M*2+wm,         bc*WARPS_N*2+wn,         g.n);
    store_c_tile_n_masked(g.c, cB, br*WARPS_M*2+wm,         bc*WARPS_N*2+WARPS_N+wn, g.n);
    store_c_tile_n_masked(g.c, cC, br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+wn,         g.n);
    store_c_tile_n_masked(g.c, cD, br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+WARPS_N+wn, g.n);
}

template<Layout L>
__global__ void gemm_tail_kernel(const layout_globals g) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= g.m || col >= g.n) {
        return;
    }

    // ``main_covers_n`` mirrors the dispatch decision (Phase 4): when the
    // generic 8-wave kernel ran with ``bpc = ceil_div(n, BLK)``, the
    // ``store_c_tile_n_masked`` helper already wrote cols [fast_n, n)
    // for rows in [0, fast_m) with the FULL K reduction. Tail must NOT
    // redo those cells. Detected from ``g.bpc`` itself:
    //   * bpc * BLK > fast_n  →  ceil_div path → main covered N.
    //   * bpc * BLK == fast_n →  fast path (4-wave RCR or K misaligned).
    const bool main_covers_n = (g.bpc * BLK > g.fast_n);
    const int n_main_limit = main_covers_n ? g.n : g.fast_n;
    const bool interior_mn = row < g.fast_m && col < n_main_limit;
    const bool fast_covers_cell = interior_mn && g.fast_m > 0 && g.fast_n > 0 && g.fast_k > 0;
    const bool needs_k_tail = g.fast_k < g.k;
    if (fast_covers_cell && !needs_k_tail) {
        return;
    }

    const int k0 = fast_covers_cell ? g.fast_k : 0;
    float acc = 0.0f;

    if constexpr (L == Layout::RCR) {
        // Vec8 fast path for RCR. See ``grouped_tail_kernel`` for the
        // rationale; mirror change to keep dense + grouped tail logic
        // in sync. Dense rarely runs the tail (LLM shapes are aligned
        // 4096 / 8192 multiples) so this is mostly a code-symmetry win.
        const fp8e4m3* a_row = &g.a[coord<>(row, 0)];
        const fp8e4m3* b_row = &g.b[coord<>(col, 0)];
        int kk = k0;
        if ((g.k % 8 == 0) && ((k0 & 7) == 0)) {
            const fp8e4m3_8* a_v8 = reinterpret_cast<const fp8e4m3_8*>(a_row);
            const fp8e4m3_8* b_v8 = reinterpret_cast<const fp8e4m3_8*>(b_row);
            const int j_start = k0 >> 3;
            const int j_end   = g.k >> 3;
            #pragma unroll 4
            for (int j = j_start; j < j_end; ++j) {
                fp8e4m3_8 a8 = a_v8[j];
                fp8e4m3_8 b8 = b_v8[j];
                float4 a_lo = base_types::convertor<float4, fp8e4m3_4>::convert(a8.lo);
                float4 a_hi = base_types::convertor<float4, fp8e4m3_4>::convert(a8.hi);
                float4 b_lo = base_types::convertor<float4, fp8e4m3_4>::convert(b8.lo);
                float4 b_hi = base_types::convertor<float4, fp8e4m3_4>::convert(b8.hi);
                acc += a_lo.x * b_lo.x + a_lo.y * b_lo.y
                     + a_lo.z * b_lo.z + a_lo.w * b_lo.w
                     + a_hi.x * b_hi.x + a_hi.y * b_hi.y
                     + a_hi.z * b_hi.z + a_hi.w * b_hi.w;
            }
            kk = j_end << 3;
        }
        for (; kk < g.k; ++kk) {
            acc += load_fp8_scalar(g.a, row, kk) * load_fp8_scalar(g.b, col, kk);
        }
    } else if constexpr (L == Layout::RRR) {
        for (int kk = k0; kk < g.k; ++kk) {
            acc += load_fp8_scalar(g.a, row, kk) * load_fp8_scalar(g.b, kk, col);
        }
    } else {
        for (int kk = k0; kk < g.k; ++kk) {
            acc += load_fp8_scalar(g.a, kk, row) * load_fp8_scalar(g.b, kk, col);
        }
    }

    const float scaled = acc * resolve_combined_scale(g);
    if (fast_covers_cell && needs_k_tail) {
        store_bf16_scalar(g.c, row, col, load_bf16_scalar(g.c, row, col) + scaled);
    } else {
        store_bf16_scalar(g.c, row, col, scaled);
    }
}

// Single dynamic K instantiation (KI_HINT=0). Experiments showed that
// compile-time KI specialization causes VGPR spills (64+ bytes/lane of
// scratch) because the two-tile main loop body is ~60 lines of asm and,
// when combined with `#pragma unroll RCR_MAIN_UNROLL` and a constexpr
// upper bound, the compiler emits many copies that exceed the register
// budget. The dynamic path holds up at 0 spills across all three layouts.
template __global__ void gemm_kernel<Layout::RCR, 0>(const layout_globals);
template __global__ void gemm_kernel<Layout::RRR, 0>(const layout_globals);
template __global__ void gemm_kernel<Layout::CRR, 0>(const layout_globals);

template __global__ void gemm_tail_kernel<Layout::RCR>(const layout_globals);
template __global__ void gemm_tail_kernel<Layout::RRR>(const layout_globals);
template __global__ void gemm_tail_kernel<Layout::CRR>(const layout_globals);

namespace fused_act_round4_compile_test {
// Definition inlined here (was forward-declared previously) so -fgpu-rdc
// builds (Primus-Turbo csrc integration) link cleanly without relying on
// late-defined hidden symbols.
__device__ __forceinline__ uint32_t cvt_bf16x4_to_fp8x4(
    bf16_2 lo, bf16_2 hi, float scale)
{
    float2 lo_f = __bfloat1622float2(lo);
    float2 hi_f = __bfloat1622float2(hi);
    lo_f.x *= scale; lo_f.y *= scale;
    hi_f.x *= scale; hi_f.y *= scale;
    int dummy_old;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
    uint32_t packed = __builtin_amdgcn_cvt_pk_fp8_f32(
        lo_f.x, lo_f.y, dummy_old, /*sel=*/false);
    packed = __builtin_amdgcn_cvt_pk_fp8_f32(
        hi_f.x, hi_f.y, packed, /*sel=*/true);
#pragma clang diagnostic pop
    return packed;
}
}  // namespace fused_act_round4_compile_test

struct grouped_layout_globals_fused_act {
    _gl_bf16 a;
    _gl_fp8 b;
    _gl_bf16 c;
    float scale_a, scale_b;
    const float* dscale_a;
    const float* dscale_b;
    const int64_t* group_offs;
    hipStream_t stream;
    int G;
    int n;
    int k;
    int ki;
    int bpc;
    int group_m;
    int num_xcds;
    int M_total;
    int fast_n, fast_k;
    int m_per_group;
    int chunk_size;              // Round-14 (gpt_oss FP8 kernel-only ceiling,
    dim3 block() { return dim3(_NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

namespace fused_act_round5_compile_test {

template<int N_THREADS,
         ducks::st::all ST_DST,
         ducks::gl::all GL_SRC,
         ducks::coord::tile COORD = coord<ST_DST>>
__device__ __forceinline__ void rcr_8w_load_hoist_fused_act(
    ST_DST& dst,
    const GL_SRC& src,
    const COORD& idx,
    const uint32_t* __restrict__ swizzled_offsets,
    float scale)
{
    using T_DST = typename ST_DST::dtype;
    using T_SRC = typename GL_SRC::dtype;
    static_assert(sizeof(T_SRC) == 2, "fused-act expects BF16 src");
    static_assert(sizeof(T_DST) == 1, "fused-act expects FP8 dst");

    constexpr int dst_bytes_per_thread =
        ST_DST::underlying_subtile_bytes_per_thread;
    constexpr int dst_bytes_per_warp =
        dst_bytes_per_thread * kittens::WARP_THREADS;
    constexpr int memcpy_per_tile =
        ST_DST::rows * ST_DST::cols * sizeof(T_DST) /
        (dst_bytes_per_thread * N_THREADS);
    static_assert(
        ST_DST::rows * ST_DST::cols * sizeof(T_DST) >= dst_bytes_per_warp,
        "shared tile must be at least 1024 bytes"
    );

    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int warpid = kittens::warpid() % num_warps;
    const int laneid = kittens::laneid();

    coord<> unit_coord = idx.template unit_coord<2, 3>();
    T_SRC* tensor_base = (T_SRC*)src.raw_ptr;
    T_SRC* global_ptr  = (T_SRC*)&src[unit_coord];
    const uint32_t total_bytes = static_cast<uint32_t>(
        size_t(src.batch()) * size_t(src.depth()) *
        size_t(src.rows())  * size_t(src.cols())  * sizeof(T_SRC));
    i32x4 srsrc = make_srsrc(tensor_base, total_bytes);
    const uint32_t tile_byte_offset = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(global_ptr) -
                              reinterpret_cast<uintptr_t>(tensor_base)));

    const uintptr_t lds_tile_base =
        reinterpret_cast<uintptr_t>(&dst.data[0]);

    uint32_t lds_addrs[memcpy_per_tile + 1];
    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; ++i) {
        const int warp_linear_offset =
            (warpid * dst_bytes_per_warp) + (i * num_warps * dst_bytes_per_warp);
        const int lds_subtile_id =
            warp_linear_offset / ST_DST::underlying_subtile_bytes;
        const uint32_t off32 = static_cast<uint32_t>(
            lds_tile_base + warp_linear_offset +
            lds_subtile_id * ST_DST::subtile_padding);
        lds_addrs[i] = __builtin_amdgcn_readfirstlane(off32);
    }

    const uint32_t lds_lane_off = static_cast<uint32_t>(laneid) * 16u;

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; ++i) {
        const uint32_t voff_lo = swizzled_offsets[i] * 2u;
        const uint32_t voff_hi = voff_lo + 16u;

        __uint128_t v_lo = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
            srsrc, voff_lo, tile_byte_offset, 0);
        __uint128_t v_hi = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
            srsrc, voff_hi, tile_byte_offset, 0);

        bf16_2* bf_lo = reinterpret_cast<bf16_2*>(&v_lo);
        bf16_2* bf_hi = reinterpret_cast<bf16_2*>(&v_hi);

        u32x4 fp8_pack = {
            fused_act_round4_compile_test::cvt_bf16x4_to_fp8x4(
                bf_lo[0], bf_lo[1], scale),
            fused_act_round4_compile_test::cvt_bf16x4_to_fp8x4(
                bf_lo[2], bf_lo[3], scale),
            fused_act_round4_compile_test::cvt_bf16x4_to_fp8x4(
                bf_hi[0], bf_hi[1], scale),
            fused_act_round4_compile_test::cvt_bf16x4_to_fp8x4(
                bf_hi[2], bf_hi[3], scale),
        };

        const uint32_t lds_addr = lds_addrs[i] + lds_lane_off;
        ::kittens::macros::ds_write_b128(fp8_pack, lds_addr, /*i_offset=*/0);
    }

    if constexpr (memcpy_per_tile * (dst_bytes_per_thread * N_THREADS) !=
                  ST_DST::rows * ST_DST::cols * sizeof(T_DST)) {
        constexpr int leftover_bytes =
            ST_DST::rows * ST_DST::cols * sizeof(T_DST) -
            memcpy_per_tile * (dst_bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / dst_bytes_per_thread;
        constexpr int leftover_warps   = leftover_threads / kittens::WARP_THREADS;
        if (warpid < leftover_warps) {
            const int warp_linear_offset =
                (warpid * dst_bytes_per_warp) +
                (memcpy_per_tile * num_warps * dst_bytes_per_warp);
            const int lds_subtile_id =
                warp_linear_offset / ST_DST::underlying_subtile_bytes;
            const uint32_t off32 = static_cast<uint32_t>(
                lds_tile_base + warp_linear_offset +
                lds_subtile_id * ST_DST::subtile_padding);
            const uint32_t lds_warp_addr = __builtin_amdgcn_readfirstlane(off32);
            const uint32_t voff_lo = swizzled_offsets[memcpy_per_tile] * 2u;
            const uint32_t voff_hi = voff_lo + 16u;

            __uint128_t v_lo = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                srsrc, voff_lo, tile_byte_offset, 0);
            __uint128_t v_hi = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                srsrc, voff_hi, tile_byte_offset, 0);

            bf16_2* bf_lo = reinterpret_cast<bf16_2*>(&v_lo);
            bf16_2* bf_hi = reinterpret_cast<bf16_2*>(&v_hi);
            u32x4 fp8_pack = {
                fused_act_round4_compile_test::cvt_bf16x4_to_fp8x4(
                    bf_lo[0], bf_lo[1], scale),
                fused_act_round4_compile_test::cvt_bf16x4_to_fp8x4(
                    bf_lo[2], bf_lo[3], scale),
                fused_act_round4_compile_test::cvt_bf16x4_to_fp8x4(
                    bf_hi[0], bf_hi[1], scale),
                fused_act_round4_compile_test::cvt_bf16x4_to_fp8x4(
                    bf_hi[2], bf_hi[3], scale),
            };

            const uint32_t lds_addr = lds_warp_addr + lds_lane_off;
            ::kittens::macros::ds_write_b128(fp8_pack, lds_addr, /*i_offset=*/0);
        }
    }
}

}  // namespace fused_act_round5_compile_test

struct grouped_layout_globals {
    _gl_fp8 a;                   // [M_total, K]
    _gl_fp8 b;                   // [G, N, K] (RCR)
    _gl_bf16 c;                  // [M_total, N]
    float scale_a, scale_b;
    const float* dscale_a;
    const float* dscale_b;
    const int64_t* group_offs;   // [G+1] int64 prefix-sum on device
    hipStream_t stream;
    int G;                       // number of groups
    int n;                       // N
    int k;                       // K
    int ki;                      // fast_k / K_BLOCK
    int bpc;                     // fast_n / BLOCK_SIZE
    int group_m;                 // tile-scheduling super-block factor
    int num_xcds;                // chiplet-swizzle XCD count (0 → default 8)
    int M_total;                 // sum of group sizes (= a.shape[0])
    // [grouped] Native non-aligned support (mirror of BF16 grouped Phase 3
    // and FP8 dense fast/tail). Main kernel only sweeps the largest aligned
    // interior:
    //   fast_n = (n / BLOCK_SIZE) * BLOCK_SIZE
    //   fast_k = (k / K_BLOCK) * K_BLOCK
    // `grouped_tail_kernel` (scalar fp32) handles cells with col >= fast_n
    // (full-K reduction) plus K-tail correction in [fast_k, k) for interior
    // cells. Per-group M-tail (M_g % BLOCK_SIZE != 0) is NOT handled in this
    // round (caller contract: each group's M is BLOCK_SIZE-aligned).
    int fast_n, fast_k;
    int m_per_group;
    int num_slots;
    int chunk_size;              // Round-14 (gpt_oss FP8 kernel-only ceiling,
    int fuse_ktail_off;
    int sk_split_n;
    int* sk_partial_buf;
    dim3 block() { return dim3(_NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

template<bool FUSE_ACT = false, typename GL>
__device__ __forceinline__ float resolve_combined_scale_grp(const GL &g) {
    const float sa_dev = g.dscale_a ? *g.dscale_a : g.scale_a;
    const float sb_dev = g.dscale_b ? *g.dscale_b : g.scale_b;
    if constexpr (FUSE_ACT) {
        // ``g.dscale_a`` for fused-act stores the FORWARD scale (output of
        // the R1 ``max_abs_bf16_to_fp8_scale`` binding). Invert to recover
        // the dequant scale that the FP8 epilog multiplies by.
        const float sa = (sa_dev > 0.0f) ? (1.0f / sa_dev) : 0.0f;
        return sa * sb_dev;
    } else {
        return sa_dev * sb_dev;
    }
}

template<int KI_HINT = 0, bool N_MASKED_STORE = false, bool FUSED_KTAIL = false,
         bool FUSE_ACT = false>
__global__ __launch_bounds__(_NUM_THREADS, 1)
void grouped_rcr_kernel(
    const std::conditional_t<FUSE_ACT,
                             grouped_layout_globals_fused_act,
                             grouped_layout_globals> g) {
    static_assert(!(FUSE_ACT && FUSED_KTAIL),
                  "FUSE_ACT=true requires FUSED_KTAIL=false");
    using ST_rcr = ST_v2;
    // [kyle-L1 single-buf DEBUG] As/Bs decl LEFT AS [2][2] for now; loop
    // body single-buffered (always reads [0]) to isolate loop-structure
    // correctness from LDS-decl side effects. Will collapse to [1][2]
    // once loop is verified.
    __shared__ ST_rcr As[2][2];
    __shared__ ST_rcr Bs[2][2];
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];
    __shared__ int s_cum_tiles[MAX_G_PLUS_1];
    __shared__ int s_total_tiles;
    A_row_reg a;
    B_row_reg b0, b1;
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;

    const int slots_eff = gridDim.x;
    const int xcds_eff = g.num_xcds > 0 ? g.num_xcds : BLOCK_SWIZZLE_NUM_XCDS;
    const int chunk_size_eff = g.chunk_size > 0 ? g.chunk_size : 64;
    int pid = chiplet_transform_chunked(
        blockIdx.x, slots_eff, xcds_eff, chunk_size_eff);

    int wm = warpid() / WARPS_N;
    int wn = warpid() % WARPS_N;
    const int num_pid_n = g.bpc;
    const int ki_dyn   = (KI_HINT > 0) ? KI_HINT : g.ki;

    if (threadIdx.x <= g.G && threadIdx.x < MAX_G_PLUS_1) {
        s_offs[threadIdx.x] = static_cast<int>(g.group_offs[threadIdx.x]);
    }
    if (threadIdx.x > g.G && threadIdx.x < MAX_G_PLUS_1) {
        s_cum_tiles[threadIdx.x] = 0x7FFFFFFF;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int prev = s_offs[0];
        s_cum_tiles[0] = 0;
        int t = 0;
        #pragma unroll 1
        for (int gi = 0; gi < g.G; ++gi) {
            const int next = s_offs[gi + 1];
            t += ((next - prev) / BLOCK_SIZE) * num_pid_n;
            s_cum_tiles[gi + 1] = t;
            prev = next;
        }
        s_total_tiles = t;
    }
    __syncthreads();
    const int total_tiles = s_total_tiles;

    // Prefill swizzled offsets ONCE (shared across all tiles & all groups —
    // depends only on the GL strides which are constant within the launch).
    constexpr int bpt = ST_rcr::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * _NUM_THREADS;
    constexpr int mpt = ST_rcr::rows * ST_rcr::cols * sizeof(fp8e4m3) / bpm;
    uint32_t soA[mpt], soB[mpt];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

    float scale_a_inv = 0.0f;
    if constexpr (FUSE_ACT) {
        scale_a_inv = (g.dscale_a != nullptr) ? *g.dscale_a : 0.0f;
    }

    for (int gt = pid; gt < total_tiles; gt += slots_eff) {
        // [grouped] 6-step branch-free binary search over LDS-cached cumsum
        // (covers G ∈ [1, 64] since 2^6 = 64 = MAX_G_PLUS_1-1). Sentinel
        // INT_MAX past g.G keeps the `gt >= s_cum_tiles[mid]` cmp false so
        // lo never advances past g.G. Compared to the linear O(G) scan,
        // this collapses ~32 LDS lds + cmp into 6 sequential lookups: ~70
        // cyc instead of ~320 cyc per outer iter (kernel-only saving ~3-5%
        // on shapes with low ki / many tiles).
        int lo = 0;
        int hi = MAX_G_PLUS_1 - 1;
        #pragma unroll
        for (int level = 0; level < 6; ++level) {
            const int mid = (lo + hi + 1) >> 1;
            if (gt >= s_cum_tiles[mid]) lo = mid;
            else hi = mid - 1;
        }
        const int group_idx = lo;
        const int tile_start = s_cum_tiles[lo];
        const int local_tile = gt - tile_start;
        const int m_start_g = s_offs[group_idx];
        const int M_g = s_offs[group_idx + 1] - m_start_g;
        const int bpr_g = M_g / BLOCK_SIZE;

        // Group-by-M / group-by-N swizzle (matches dense kernel mapping).
        int br, bc;
        if (g.bpc > bpr_g) {
            const int WGN = g.group_m;
            const int num_wgid_in_group = bpr_g * WGN;
            int group_id = local_tile / num_wgid_in_group;
            int first_pid_n = group_id * WGN;
            int group_size_n = min(num_pid_n - first_pid_n, WGN);
            if (group_size_n <= 0) continue;
            bc = first_pid_n + ((local_tile % num_wgid_in_group) % group_size_n);
            br = (local_tile % num_wgid_in_group) / group_size_n;
        } else {
            const int WGM = g.group_m;
            const int num_wgid_in_group = WGM * num_pid_n;
            int group_id = local_tile / num_wgid_in_group;
            int first_pid_m = group_id * WGM;
            int group_size_m = min(bpr_g - first_pid_m, WGM);
            if (group_size_m <= 0) continue;
            br = first_pid_m + ((local_tile % num_wgid_in_group) % group_size_m);
            bc = (local_tile % num_wgid_in_group) / group_size_m;
        }
        if (br >= bpr_g || bc >= num_pid_n) continue;

        // Coord shifts:
        //   ST_A: st_fp8e4m3<HB=128, BK=128, ...> → row-coord unit = HB = 128.
        //         m_subtile_A = m_start_g / HB.
        //   C (RT store): rt_fl<RBM=64, RBN=32, ...> → row-coord unit = RBM=64.
        //         m_subtile_C = m_start_g / RBM.
        const int m_subtile_A = m_start_g / HB;
        const int m_subtile_C = m_start_g / RBM;

        auto a_co = [&](int s, int k) -> coord<ST_rcr> {
            return {0, 0, m_subtile_A + s, k};
        };
        auto b_co = [&](int s, int k) -> coord<ST_rcr> {
            return {0, group_idx, s, k};
        };

        auto load_a = [&](A_row_reg& dst, ST_rcr& tile, int wi) {
            auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
            load(dst, sub);
        };
        auto load_b = [&](B_row_reg& dst, ST_rcr& tile, int wi) {
            auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
            load(dst, sub);
        };

        auto b_tile = [&](int stage, int which) -> ST_rcr& {
            return Bs[stage][which];
        };

        // Reset accumulators per tile.
        zero(cA); zero(cB); zero(cC); zero(cD);

        int tic = 0, toc = 1;
        // Prologue: load tile-0 + tile-1 (mirrors gemm_kernel<RCR> 1040-1054).
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 0), g.b, b_co(bc*2,   0), soB);
        if constexpr (FUSE_ACT) fused_act_round5_compile_test::rcr_8w_load_hoist_fused_act<_NUM_THREADS>(As[tic][0], g.a, a_co(br*2,   0), soA, scale_a_inv);
        else                    rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0],    g.a, a_co(br*2,   0), soA);
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 1), g.b, b_co(bc*2+1, 0), soB);
        if constexpr (FUSE_ACT) fused_act_round5_compile_test::rcr_8w_load_hoist_fused_act<_NUM_THREADS>(As[tic][1], g.a, a_co(br*2+1, 0), soA, scale_a_inv);
        else                    rcr_8w_load_hoist<_NUM_THREADS>(As[tic][1],    g.a, a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(RCR_INIT0_VMCNT);
        __builtin_amdgcn_s_barrier();

        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(toc, 0), g.b, b_co(bc*2,   1), soB);
        if constexpr (FUSE_ACT) fused_act_round5_compile_test::rcr_8w_load_hoist_fused_act<_NUM_THREADS>(As[toc][0], g.a, a_co(br*2,   1), soA, scale_a_inv);
        else                    rcr_8w_load_hoist<_NUM_THREADS>(As[toc][0],    g.a, a_co(br*2,   1), soA);
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(toc, 1), g.b, b_co(bc*2+1, 1), soB);

        TK_WAIT_VMCNT(RCR_INIT1_VMCNT);
        __builtin_amdgcn_s_barrier();

        TK_PRAGMA_UNROLL(RCR_MAIN_UNROLL)
        for (int k = 0; k < ki_dyn - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, b_tile(tic, 0), wn);
            load_a(a, As[tic][0], wm);
            if constexpr (FUSE_ACT) fused_act_round5_compile_test::rcr_8w_load_hoist_fused_act<_NUM_THREADS>(As[toc][1], g.a, a_co(br*2+1, k+1), soA, scale_a_inv);
            else                    rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], g.a, a_co(br*2+1, k+1), soA);
            TK_WAIT_LGKM(RCR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, b_tile(tic, 1), wn);
            rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 0), g.b, b_co(bc*2, k+2), soB);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            if constexpr (FUSE_ACT) fused_act_round5_compile_test::rcr_8w_load_hoist_fused_act<_NUM_THREADS>(As[tic][0], g.a, a_co(br*2, k+2), soA, scale_a_inv);
            else                    rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0], g.a, a_co(br*2, k+2), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 1), g.b, b_co(bc*2+1, k+2), soB);
            TK_WAIT_VMCNT(RCR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_setprio(1); rcr_mma(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        // Epilog 1: second-to-last K-tile (mirrors dense lines 1160-1187).
        {
            load_b(b0, b_tile(tic, 0), wn);
            load_a(a, As[tic][0], wm);
            if constexpr (FUSE_ACT) fused_act_round5_compile_test::rcr_8w_load_hoist_fused_act<_NUM_THREADS>(As[toc][1], g.a, a_co(br*2+1, ki_dyn-1), soA, scale_a_inv);
            else                    rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], g.a, a_co(br*2+1, ki_dyn-1), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

            load_b(b1, b_tile(tic, 1), wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(RCR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, b_tile(toc, 0), wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();
            tic ^= 1; toc ^= 1;
        }

        // Epilog 2: last K-tile (mirrors dense lines 1189-1210).
        {
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, b_tile(tic, 1), wn);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            rcr_mma(cC, a, b0);
            rcr_mma(cD, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        if constexpr (FUSED_KTAIL) {
            // Round-7-dm: scoped A register tile for K-tail M-slab 1
            // (moved from function-scope; round-3 introduced the reg
            // to save one ``vmcnt(0)`` wait on K-misaligned shapes).
            A_row_reg a_kt1;
            if (g.fast_k < g.k) {
                const int laneid = kittens::laneid();
                const int row_lane = laneid % 16;
                const int k_lane_byte = (laneid / 16) * 32;
                constexpr int KREM = 64;
                static_assert(KREM == 64,
                    "FUSED_KTAIL=true K_REM must be 64; see fuse_ktail_eligible");
                const bool both_valid = (laneid < 32);
                constexpr uint32_t SENTINEL = 0xFFFF0000u;

                const fp8e4m3* a_base_ptr = (const fp8e4m3*)&g.a[{0, 0, 0, 0}];
                const fp8e4m3* b_base_ptr = (const fp8e4m3*)&g.b[{0, 0, 0, 0}];
                const int a_row_stride_bytes = g.a.template stride<2>();
                const int b_row_stride_bytes = g.b.template stride<2>();
                const uint32_t a_total_bytes =
                    static_cast<uint32_t>(g.M_total) *
                    static_cast<uint32_t>(a_row_stride_bytes);
                const uint32_t b_per_group_bytes =
                    static_cast<uint32_t>(group_idx + 1) *
                    static_cast<uint32_t>(g.n) *
                    static_cast<uint32_t>(b_row_stride_bytes);
                i32x4 a_srsrc_kt = make_srsrc((const void*)a_base_ptr, a_total_bytes);
                i32x4 b_srsrc_kt = make_srsrc((const void*)b_base_ptr, b_per_group_bytes);

                const uint32_t K_tail_base_bytes =
                    static_cast<uint32_t>(g.fast_k);
                const uint32_t b_group_byte_base =
                    static_cast<uint32_t>(group_idx) *
                    static_cast<uint32_t>(g.n) *
                    static_cast<uint32_t>(b_row_stride_bytes);

                auto load_a_kt = [&](A_row_reg& A_tile, int slab)
                        __attribute__((always_inline)) {
                    const int M_warp_base =
                        (m_subtile_A + br * 2 + slab) * HB + wm * RBM;
                    #pragma unroll
                    for (int h = 0; h < A_row_reg::height; ++h) {
                        const int A_row_idx = M_warp_base + h * 16 + row_lane;
                        const uint32_t v_base = static_cast<uint32_t>(
                            A_row_idx * a_row_stride_bytes +
                            K_tail_base_bytes + k_lane_byte);
                        const uint32_t v_lo = both_valid ? v_base : SENTINEL;
                        const uint32_t v_hi = both_valid ? (v_base + 16) : SENTINEL;
                        __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            a_srsrc_kt, v_lo, 0, 0);
                        __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            a_srsrc_kt, v_hi, 0, 0);
                        *reinterpret_cast<__uint128_t*>(&A_tile.tiles[h][0].data[0]) = v0;
                        *reinterpret_cast<__uint128_t*>(&A_tile.tiles[h][0].data[4]) = v1;
                    }
                };

                // N_warp_base derivation:
                //   b_co(s, k) → coord {0, group_idx, s, k}
                //   unit_coord: N-row in tile = s * ST_rcr::rows = s * HB.
                //   warp wn picks rows wn*RBN..wn*RBN+RBN-1 within the 128-row tile.
                //   For h_b ∈ [0, B_row_reg::height = 2):
                //     B_row_idx_in_group = N_warp_base + h_b*16 + row_lane.
                //   Global byte = group_idx * N * K + B_row_idx_in_group * K + ...
                auto load_b_kt = [&](B_row_reg& B_tile, int n_strip) __attribute__((always_inline)) {
                    const int N_warp_base =
                        (bc * 2 + n_strip) * HB + wn * RBN;
                    #pragma unroll
                    for (int h_b = 0; h_b < B_row_reg::height; ++h_b) {
                        const int B_row_idx_in_group = N_warp_base + h_b * 16 + row_lane;
                        const uint32_t v_base = b_group_byte_base + static_cast<uint32_t>(
                            B_row_idx_in_group * b_row_stride_bytes +
                            K_tail_base_bytes + k_lane_byte);
                        const uint32_t v_lo = both_valid ? v_base : SENTINEL;
                        const uint32_t v_hi = both_valid ? (v_base + 16) : SENTINEL;
                        __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            b_srsrc_kt, v_lo, 0, 0);
                        __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            b_srsrc_kt, v_hi, 0, 0);
                        *reinterpret_cast<__uint128_t*>(&B_tile.tiles[h_b][0].data[0]) = v0;
                        *reinterpret_cast<__uint128_t*>(&B_tile.tiles[h_b][0].data[4]) = v1;
                    }
                };

                load_b_kt(b0,    0);   // 4 buffer_load → b0 (issued FIRST)
                load_b_kt(b1,    1);   // 4 buffer_load → b1
                load_a_kt(a,     0);   // 8 buffer_load → a (M slab 0)
                load_a_kt(a_kt1, 1);   // 8 buffer_load → a_kt1 (M slab 1, LAST)
                TK_WAIT_VMCNT(RCR_KTAIL_VMCNT);
                rcr_mma(cA, a,     b0);
                rcr_mma(cB, a,     b1);
                asm volatile("s_waitcnt vmcnt(0)");
                rcr_mma(cC, a_kt1, b0);
                rcr_mma(cD, a_kt1, b1);
            }
        }

        const float combined_scale = resolve_combined_scale_grp<FUSE_ACT>(g);

        if (wm == 0) __builtin_amdgcn_s_barrier();
        const int r0 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2+wm);
        const int r1 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2+WARPS_M+wm);
        const int c0 = __builtin_amdgcn_readfirstlane(bc*WARPS_N*2+wn);
        const int c1 = __builtin_amdgcn_readfirstlane(bc*WARPS_N*2+WARPS_N+wn);
        if constexpr (N_MASKED_STORE) {
            if ((bc + 1) * BLOCK_SIZE <= g.n) {
                mul(cA, cA, combined_scale);
                store(g.c, cA, {0, 0, r0, c0});
                mul(cB, cB, combined_scale);
                store(g.c, cB, {0, 0, r0, c1});
                mul(cC, cC, combined_scale);
                store(g.c, cC, {0, 0, r1, c0});
                mul(cD, cD, combined_scale);
                store(g.c, cD, {0, 0, r1, c1});
            } else {
                mul(cA, cA, combined_scale);
                store_c_tile_n_masked(g.c, cA, r0, c0, g.n);
                mul(cB, cB, combined_scale);
                store_c_tile_n_masked(g.c, cB, r0, c1, g.n);
                mul(cC, cC, combined_scale);
                store_c_tile_n_masked(g.c, cC, r1, c0, g.n);
                mul(cD, cD, combined_scale);
                store_c_tile_n_masked(g.c, cD, r1, c1, g.n);
            }
        } else {
            mul(cA, cA, combined_scale);
            store(g.c, cA, {0, 0, r0, c0});
            mul(cB, cB, combined_scale);
            store(g.c, cB, {0, 0, r0, c1});
            mul(cC, cC, combined_scale);
            store(g.c, cC, {0, 0, r1, c0});
            mul(cD, cD, combined_scale);
            store(g.c, cD, {0, 0, r1, c1});
        }

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }
}

template __global__ void grouped_rcr_kernel<0, false, false>(const grouped_layout_globals);
template __global__ void grouped_rcr_kernel<0, true , false>(const grouped_layout_globals);
template __global__ void grouped_rcr_kernel<0, false, true >(const grouped_layout_globals);
template __global__ void grouped_rcr_kernel<0, true , true >(const grouped_layout_globals);

// Force-instantiate. Compare resource report against the R57 step-2A
// baseline (V256 / A256 / Spill 0 / Scratch 0 — placeholder G::load).


#ifndef FP8_RRR_FUSE_PROBE
#define FP8_RRR_FUSE_PROBE 0
#endif
template<int KI_HINT = 0, bool N_MASKED_STORE = false, bool FUSED_KTAIL = false>
__global__ __launch_bounds__(_NUM_THREADS, 1)
void grouped_rrr_kernel(const grouped_layout_globals g) {
    __shared__ ST_row As[2][2];
    __shared__ ST_v2  Bs[2][2];
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];
    __shared__ int s_cum_tiles[MAX_G_PLUS_1];
    __shared__ int s_total_tiles;

    A_row_reg a;
    B_col_reg b0, b1;
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;

    // Round-2 (FP8 backward unblock): mirror RCR — read host-side
    // ``g.num_xcds`` knob, fall back to the default 8 when unset.
    const int xcds_eff = g.num_xcds > 0 ? g.num_xcds : BLOCK_SWIZZLE_NUM_XCDS;
    int pid = chiplet_transform_chunked(
        blockIdx.x, NUM_CUS, xcds_eff, 64);

    int wm = warpid() / WARPS_N;
    int wn = warpid() % WARPS_N;
    const int num_pid_n = g.bpc;
    const int ki_dyn   = (KI_HINT > 0) ? KI_HINT : g.ki;

    if (threadIdx.x <= g.G && threadIdx.x < MAX_G_PLUS_1) {
        s_offs[threadIdx.x] = static_cast<int>(g.group_offs[threadIdx.x]);
    }
    if (threadIdx.x > g.G && threadIdx.x < MAX_G_PLUS_1) {
        s_cum_tiles[threadIdx.x] = 0x7FFFFFFF;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int prev = s_offs[0];
        s_cum_tiles[0] = 0;
        int t = 0;
        #pragma unroll 1
        for (int gi = 0; gi < g.G; ++gi) {
            const int next = s_offs[gi + 1];
            t += ((next - prev) / BLOCK_SIZE) * num_pid_n;
            s_cum_tiles[gi + 1] = t;
            prev = next;
        }
        s_total_tiles = t;
    }
    __syncthreads();
    const int total_tiles = s_total_tiles;

    constexpr int bptA = ST_row::underlying_subtile_bytes_per_thread;
    constexpr int bpmA = bptA * _NUM_THREADS;
    constexpr int mptA = ST_row::rows * ST_row::cols * sizeof(fp8e4m3) / bpmA;
    uint32_t soA[mptA];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);

    constexpr int bptB = ST_v2::underlying_subtile_bytes_per_thread;
    constexpr int bpmB = bptB * _NUM_THREADS;
    constexpr int mptB = ST_v2::rows * ST_v2::cols * sizeof(fp8e4m3) / bpmB;
    uint32_t soB[mptB];
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

    for (int gt = pid; gt < total_tiles; gt += NUM_CUS) {
        int lo = 0;
        int hi = MAX_G_PLUS_1 - 1;
        #pragma unroll
        for (int level = 0; level < 6; ++level) {
            const int mid = (lo + hi + 1) >> 1;
            if (gt >= s_cum_tiles[mid]) lo = mid;
            else hi = mid - 1;
        }
        const int group_idx = lo;
        const int tile_start = s_cum_tiles[lo];
        const int local_tile = gt - tile_start;
        const int m_start_g = s_offs[group_idx];
        const int M_g = s_offs[group_idx + 1] - m_start_g;
        const int bpr_g = M_g / BLOCK_SIZE;

        int br, bc;
        if (g.bpc > bpr_g) {
            const int WGN = g.group_m;
            const int num_wgid_in_group = bpr_g * WGN;
            int group_id = local_tile / num_wgid_in_group;
            int first_pid_n = group_id * WGN;
            int group_size_n = min(num_pid_n - first_pid_n, WGN);
            if (group_size_n <= 0) continue;
            bc = first_pid_n + ((local_tile % num_wgid_in_group) % group_size_n);
            br = (local_tile % num_wgid_in_group) / group_size_n;
        } else {
            const int WGM = g.group_m;
            const int num_wgid_in_group = WGM * num_pid_n;
            int group_id = local_tile / num_wgid_in_group;
            int first_pid_m = group_id * WGM;
            int group_size_m = min(bpr_g - first_pid_m, WGM);
            if (group_size_m <= 0) continue;
            br = first_pid_m + ((local_tile % num_wgid_in_group) % group_size_m);
            bc = (local_tile % num_wgid_in_group) / group_size_m;
        }
        if (br >= bpr_g || bc >= num_pid_n) continue;

        const int m_subtile_A = m_start_g / HB;
        const int m_subtile_C = m_start_g / RBM;

        // RRR coord conventions (mirror of dense gemm_kernel<RRR>):
        //   a_co(s, k) : A is [M_total, K]      → row-shift by m_subtile_A.
        //   b_co(s, k) : B is [1, G, K, N]      → K on row, N on col, group
        //                                         depth = group_idx.
        auto a_co = [&](int s, int k) -> coord<ST_row> {
            return {0, 0, m_subtile_A + s, k};
        };
        auto b_co = [&](int s, int k) -> coord<ST_v2> {
            return {0, group_idx, k, s};
        };

        auto load_a = [&](A_row_reg& dst, ST_row& tile, int wi) {
            auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
            load(dst, sub);
        };
        auto load_b = [&](B_col_reg& dst, ST_v2& tile, int wi) {
            load_col_from_st(dst, tile, wi * RBN);
        };

        zero(cA); zero(cB); zero(cC); zero(cD);

        int tic = 0, toc = 1;
        // Prologue: tile-0 + tile-1 (mirrors dense gemm_kernel<RRR>
        // lines 1421-1435).
        G::load(Bs[tic][0], g.b, b_co(bc*2,   0), soB);
        G::load(As[tic][0], g.a, a_co(br*2,   0), soA);
        G::load(Bs[tic][1], g.b, b_co(bc*2+1, 0), soB);
        G::load(As[tic][1], g.a, a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(RRR_INIT0_VMCNT);
        __builtin_amdgcn_s_barrier();

        G::load(Bs[toc][0], g.b, b_co(bc*2,   1), soB);
        G::load(As[toc][0], g.a, a_co(br*2,   1), soA);
        G::load(Bs[toc][1], g.b, b_co(bc*2+1, 1), soB);

        TK_WAIT_VMCNT(RRR_INIT1_VMCNT);
        __builtin_amdgcn_s_barrier();

        // Single-tile main loop (mirror dense lines 1437-1470).
        TK_PRAGMA_UNROLL(RRR_MAIN_UNROLL)
        for (int k = 0; k < ki_dyn - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, k+1), soA);
            TK_WAIT_LGKM(RRR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();

            load_b(b1, Bs[tic][1], wn);
            G::load(Bs[tic][0], g.b, b_co(bc*2, k+2), soB);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            G::load(Bs[tic][1], g.b, b_co(bc*2+1, k+2), soB);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();

            G::load(As[tic][0], g.a, a_co(br*2, k+2), soA);
            TK_WAIT_VMCNT(RRR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        // Epilog 1 (mirror dense lines 1472-1501).
        {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, ki_dyn-1), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[toc][0], wn);
            TK_WAIT_VMCNT(RRR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();
            tic ^= 1; toc ^= 1;
        }

        // Epilog 2 (mirror dense lines 1503-1526).
        {
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            rrr_mma(cC, a, b0);
            rrr_mma(cD, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        // ===== FUSED_KTAIL (RRR FP8) =====
        // K-tail K_REM=64 reduction fused into main kernel epilog. Both A
        // AND B loaded direct-to-register via raw_buffer_load_b8 to avoid
        // the load_col_from_st (ds_read_b64_tr_b8) path that triggers a
        // mfma operand-forwarding quirk for RRR. Per-group SRD bounds
        // protect against cross-group/OOB garbage reads (K_row >= g.k → 0).
        // K-aligned-to-32 chunks: lane chunks 0,1 carry real data (K=0..64),
        // chunks 2,3 zeroed (K=64..128, mfma's K=128 zero-pad).
        if constexpr (FUSED_KTAIL) {
            if (g.fast_k < g.k) {
                typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
                A_row_reg a_kt0, a_kt1;
                B_col_reg b0_kt, b1_kt;

                const int laneid_fk  = kittens::laneid();
                const int row_lane_fk = laneid_fk % 16;
                const int chunk_fk    = laneid_fk / 16;          // 0..3
                const int k_lane_byte_fk = chunk_fk * 32;
                const bool ab_chunk_valid = (chunk_fk < 2);      // K=0..64 chunks have real K-tail data
                constexpr uint32_t SENTINEL_FK = 0xFFFF0000u;

                // ---- Per-group SRDs (so OOB voffset returns 0) ----
                const fp8e4m3* a_base_ptr_fk =
                    (const fp8e4m3*)&g.a[{0, 0, 0, 0}];
                fp8e4m3* b_grp_base = const_cast<fp8e4m3*>(
                    &g.b[{0, group_idx, 0, 0}]);
                const uint32_t a_row_stride_bytes_fk = g.a.template stride<2>();
                const uint32_t b_row_stride_bytes_fk = g.b.template stride<2>();
                const uint32_t a_total_bytes_fk =
                    static_cast<uint32_t>(g.M_total) * a_row_stride_bytes_fk;
                const uint32_t b_grp_total_bytes_fk =
                    static_cast<uint32_t>(g.k) * b_row_stride_bytes_fk;
                i32x4 a_srsrc_fk = make_srsrc(
                    (const void*)a_base_ptr_fk, a_total_bytes_fk);
                i32x4 b_srsrc_fk = make_srsrc(
                    (const void*)b_grp_base, b_grp_total_bytes_fk);
                const uint32_t K_tail_byte_fk =
                    static_cast<uint32_t>(g.fast_k);

                // ---- Custom A K-tail load (direct-to-reg, mirror RCR) ----
                auto load_a_kt_fk = [&](A_row_reg& A_tile, int slab)
                        __attribute__((always_inline)) {
                    const int M_warp_base =
                        (m_subtile_A + br * 2 + slab) * HB + wm * RBM;
                    #pragma unroll
                    for (int h = 0; h < A_row_reg::height; ++h) {
                        const int A_row_idx =
                            M_warp_base + h * 16 + row_lane_fk;
                        const uint32_t v_base = static_cast<uint32_t>(
                            A_row_idx * a_row_stride_bytes_fk +
                            K_tail_byte_fk + k_lane_byte_fk);
                        const uint32_t v_lo = ab_chunk_valid ? v_base : SENTINEL_FK;
                        const uint32_t v_hi = ab_chunk_valid ? (v_base + 16) : SENTINEL_FK;
                        __uint128_t va0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            a_srsrc_fk, v_lo, 0, 0);
                        __uint128_t va1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            a_srsrc_fk, v_hi, 0, 0);
                        *reinterpret_cast<__uint128_t*>(
                            &A_tile.tiles[h][0].data[0]) = va0;
                        *reinterpret_cast<__uint128_t*>(
                            &A_tile.tiles[h][0].data[4]) = va1;
                    }
                };

                // ---- Custom B K-tail load (direct-to-reg, strided gather) ----
                // For each base tile j (B_col_reg.tiles[0][j], j=0..1) of one
                // n-strip, lane (l) holds 32 fp8 along K direction at one N
                // column (col = col_block_base + n_strip*HB + wn*RBN + j*16 + l%16).
                // Each fp8 byte lives at byte position
                //   (K_idx * b_row_stride + N_col)
                // → 32 strided 1-byte loads per base tile per lane.
                auto load_b_kt_fk = [&](B_col_reg& B_tile, int n_strip)
                        __attribute__((always_inline)) {
                    const int N_warp_base =
                        (bc * 2 + n_strip) * HB + wn * RBN;
                    #pragma unroll
                    for (int j = 0; j < B_col_reg::width; ++j) {
                        const int n_col = N_warp_base + j * 16 + row_lane_fk;
                        // Each lane holds 32 fp8 along K, packed into
                        // .data[8] of fp8e4m3_4. Layout: data[idx][i] holds
                        // fp8 at K = K_tail + chunk*32 + idx*4 + i.
                        intx8_t b_pack = intx8_t{};
                        if (ab_chunk_valid) {
                            const uint32_t K_base_byte =
                                K_tail_byte_fk + k_lane_byte_fk;
                            uint8_t* bp_out = (uint8_t*)&b_pack;
                            #pragma unroll
                            for (int i = 0; i < 32; ++i) {
                                const uint32_t voffset =
                                    (K_base_byte + i) * b_row_stride_bytes_fk
                                    + static_cast<uint32_t>(n_col);
                                bp_out[i] = ::kittens::llvm_amdgcn_raw_buffer_load_b8(
                                    b_srsrc_fk, voffset, 0, 0);
                            }
                        }
                        *reinterpret_cast<intx8_t*>(
                            &B_tile.tiles[0][j].data[0]) = b_pack;
                    }
                };

                load_b_kt_fk(b0_kt, 0);
                load_b_kt_fk(b1_kt, 1);
                load_a_kt_fk(a_kt0, 0);
                load_a_kt_fk(a_kt1, 1);
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                rrr_mma(cA, a_kt0, b0_kt);
                rrr_mma(cB, a_kt0, b1_kt);
                rrr_mma(cC, a_kt1, b0_kt);
                rrr_mma(cD, a_kt1, b1_kt);
                __builtin_amdgcn_s_barrier();
            }
        }

#if FP8_RRR_FUSE_PROBE
        if (g.fast_k < g.k) {
            // ---- Cooperative pre-zero of Bs[tic][0/1] ----
            // ST_v2 has swizzle padding (= 17408 bytes); strip to 16-byte
            // alignment so we can b128-store the entire region incl. padding.
            constexpr int ST_V2_B128 = (sizeof(ST_v2) / 16);
            const int tid = threadIdx.x;
            __uint128_t* Bs0_ptr = reinterpret_cast<__uint128_t*>(&Bs[tic][0].data[0]);
            __uint128_t* Bs1_ptr = reinterpret_cast<__uint128_t*>(&Bs[tic][1].data[0]);
            #pragma unroll
            for (int idx = tid; idx < ST_V2_B128; idx += _NUM_THREADS) {
                Bs0_ptr[idx] = 0;
                Bs1_ptr[idx] = 0;
            }
            __syncthreads();

            // ---- B side: cooperative G::load on K-tail iter ----
            // OOB voffsets (k_row >= K_global) no-op on raw_buffer_load_lds →
            // pre-zeroed bytes preserved → effective zero-pad for K=[K_global,
            // fast_k + K_BLOCK). Cross-group contamination is OK for the G=1
            // probe shape (full-tensor SRD == per-group SRD); production hybrid
            // would need per-group SRD construction.
            G::load(Bs[tic][0], g.b, b_co(bc*2,   ki_dyn), soB);
            G::load(Bs[tic][1], g.b, b_co(bc*2+1, ki_dyn), soB);

            const int laneid = kittens::laneid();
            const int row_lane = laneid % 16;
            const int k_lane_byte = (laneid / 16) * 32;
            const int K_REM = g.k - g.fast_k;
            const bool b128_lo_valid = (k_lane_byte + 16) <= K_REM;
            const bool b128_hi_valid = (k_lane_byte + 32) <= K_REM;
            constexpr uint32_t SENTINEL = 0xFFFF0000u;
            const fp8e4m3* a_base_ptr = (const fp8e4m3*)&g.a[{0, 0, 0, 0}];
            const int a_row_stride_bytes = g.a.template stride<2>();
            const uint32_t a_total_bytes =
                static_cast<uint32_t>(g.M_total) *
                static_cast<uint32_t>(a_row_stride_bytes);
            i32x4 a_srsrc_kt = make_srsrc((const void*)a_base_ptr, a_total_bytes);
            const uint32_t K_tail_base_bytes =
                static_cast<uint32_t>(g.fast_k);

            auto load_a_kt = [&](int slab) __attribute__((always_inline)) {
                const int M_warp_base =
                    (m_subtile_A + br * 2 + slab) * HB + wm * RBM;
                #pragma unroll
                for (int h = 0; h < A_row_reg::height; ++h) {
                    const int A_row_idx = M_warp_base + h * 16 + row_lane;
                    const uint32_t v_base = static_cast<uint32_t>(
                        A_row_idx * a_row_stride_bytes +
                        K_tail_base_bytes + k_lane_byte);
                    const uint32_t v_lo = b128_lo_valid ? v_base : SENTINEL;
                    const uint32_t v_hi = b128_hi_valid ? (v_base + 16) : SENTINEL;
                    __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                        a_srsrc_kt, v_lo, 0, 0);
                    __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                        a_srsrc_kt, v_hi, 0, 0);
                    *reinterpret_cast<__uint128_t*>(&a.tiles[h][0].data[0]) = v0;
                    *reinterpret_cast<__uint128_t*>(&a.tiles[h][0].data[4]) = v1;
                }
            };

            // Wait B G::load + LDS visibility before B reads.
            asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
            __syncthreads();

#if !FP8_RRR_FUSE_PROBE_SKIP_A_LOAD
            load_b(b0, Bs[tic][0], wn);
            load_a_kt(0);
            asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)" ::: "memory");
            rrr_mma(cA, a, b0);

            load_b(b1, Bs[tic][1], wn);
            rrr_mma(cB, a, b1);

            load_a_kt(1);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            rrr_mma(cC, a, b0);
            rrr_mma(cD, a, b1);
#endif
            __builtin_amdgcn_s_barrier();
        }
#endif

        const float combined_scale = resolve_combined_scale_grp(g);
        mul(cA, cA, combined_scale);
        mul(cB, cB, combined_scale);
        mul(cC, cC, combined_scale);
        mul(cD, cD, combined_scale);

        if (wm == 0) __builtin_amdgcn_s_barrier();
        const int r0 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2+wm);
        const int r1 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2+WARPS_M+wm);
        const int c0 = __builtin_amdgcn_readfirstlane(bc*WARPS_N*2+wn);
        const int c1 = __builtin_amdgcn_readfirstlane(bc*WARPS_N*2+WARPS_N+wn);
        if constexpr (N_MASKED_STORE) {
            // Mirror RCR: bpc = ceil_div(g.n, BLOCK_SIZE) so the last bc
            // can extend past g.n; mask its OOB columns at store time.
            // OOB B reads earlier in the loop returned 0 via raw_buffer_load
            // OOB-no-op, so MMA contributions for OOB cells are 0.
            if ((bc + 1) * BLOCK_SIZE <= g.n) {
                store(g.c, cA, {0, 0, r0, c0});
                store(g.c, cB, {0, 0, r0, c1});
                store(g.c, cC, {0, 0, r1, c0});
                store(g.c, cD, {0, 0, r1, c1});
            } else {
                store_c_tile_n_masked(g.c, cA, r0, c0, g.n);
                store_c_tile_n_masked(g.c, cB, r0, c1, g.n);
                store_c_tile_n_masked(g.c, cC, r1, c0, g.n);
                store_c_tile_n_masked(g.c, cD, r1, c1, g.n);
            }
        } else {
            store(g.c, cA, {0, 0, r0, c0});
            store(g.c, cB, {0, 0, r0, c1});
            store(g.c, cC, {0, 0, r1, c0});
            store(g.c, cD, {0, 0, r1, c1});
        }

        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }
}

template __global__ void grouped_rrr_kernel<0, false, false>(const grouped_layout_globals);
template __global__ void grouped_rrr_kernel<0, true , false>(const grouped_layout_globals);
template __global__ void grouped_rrr_kernel<0, false, true >(const grouped_layout_globals);
template __global__ void grouped_rrr_kernel<0, true , true >(const grouped_layout_globals);

// =============================================================================
// Grouped tail kernel — scalar fp32 fixup for cells the main grouped kernel
// does not cover (col >= fast_n) and the K-tail correction in [fast_k, k)
// for interior cells. Mirror of `gemm_tail_kernel` (FP8 dense) but with
// per-group B indexing via `group_offs`. Templated over Layout to support
// both RCR (forward) and RRR (backward dA) — see also the BF16 mirror in
// kernel_bf16_dynamic.cpp::grouped_tail_kernel.
//
// Three cases per cell:
//   * col <  fast_n  AND fast_k == k  → main covers fully → early-return.
//   * col <  fast_n  AND fast_k <  k  → main wrote partial; add K-tail.
//   * col >= fast_n                   → main did not run; full-K reduction.
//
// Per-group M-tail (M_g % BLOCK_SIZE != 0) is NOT handled — caller contract.
//
// B layout per Layout L:
//   * RCR : g.b is [1, G, N, K]  → B[g_idx, col, kk] (stride-1 in K).
//           Vec8 fast path enabled: A and B both contiguous in K.
//   * RRR : g.b is [1, G, K, N]  → B[g_idx, kk, col] (stride-1 in N).
//           B not vectorisable along K → scalar K-loop only.
template<Layout L>
__global__ void grouped_tail_kernel(const grouped_layout_globals g) {
    static_assert(L == Layout::RCR || L == Layout::RRR,
                  "FP8 grouped tail kernel: RCR or RRR only.");
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        #pragma unroll 1
        for (int gi = 0; gi <= g.G; ++gi) {
            s_offs[gi] = static_cast<int>(g.group_offs[gi]);
        }
    }
    __syncthreads();

    const bool main_covers_n = (g.bpc * BLOCK_SIZE > g.fast_n);
    const bool needs_k_tail = g.fast_k < g.k;
    const bool needs_n_tail = !main_covers_n && (g.fast_n < g.n);
    if (!needs_k_tail && !needs_n_tail) return;

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= g.M_total || col >= g.n) return;

    int group_idx = 0;
    #pragma unroll 1
    for (int gi = 0; gi < g.G; ++gi) {
        if (row < s_offs[gi + 1]) { group_idx = gi; break; }
    }

    const bool interior_n       = main_covers_n ? true : (col < g.fast_n);
    const bool fast_covers_cell = interior_n && g.fast_n > 0 && g.fast_k > 0;
    if (fast_covers_cell && !needs_k_tail) return;

    const int k0 = fast_covers_cell ? g.fast_k : 0;
    float acc = 0.0f;

    if constexpr (L == Layout::RCR) {
        const fp8e4m3* a_row = &g.a[coord<>(row, 0)];
        const fp8e4m3* b_row = &g.b[coord<>{0, group_idx, col, 0}];
        int kk = k0;
        if ((g.k % 8 == 0) && ((k0 & 7) == 0)) {
            const fp8e4m3_8* a_v8 = reinterpret_cast<const fp8e4m3_8*>(a_row);
            const fp8e4m3_8* b_v8 = reinterpret_cast<const fp8e4m3_8*>(b_row);
            const int j_start = k0 >> 3;
            const int j_end   = g.k >> 3;
            #pragma unroll 4
            for (int j = j_start; j < j_end; ++j) {
                fp8e4m3_8 a8 = a_v8[j];
                fp8e4m3_8 b8 = b_v8[j];
                float4 a_lo = base_types::convertor<float4, fp8e4m3_4>::convert(a8.lo);
                float4 a_hi = base_types::convertor<float4, fp8e4m3_4>::convert(a8.hi);
                float4 b_lo = base_types::convertor<float4, fp8e4m3_4>::convert(b8.lo);
                float4 b_hi = base_types::convertor<float4, fp8e4m3_4>::convert(b8.hi);
                acc += a_lo.x * b_lo.x + a_lo.y * b_lo.y
                     + a_lo.z * b_lo.z + a_lo.w * b_lo.w
                     + a_hi.x * b_hi.x + a_hi.y * b_hi.y
                     + a_hi.z * b_hi.z + a_hi.w * b_hi.w;
            }
            kk = j_end << 3;
        }
        for (; kk < g.k; ++kk) {
            acc += load_fp8_scalar(g.a, row, kk) *
                   load_fp8_scalar_grp(g.b, group_idx, col, kk);
        }
    } else {
        // RRR (FP8): main kernel now sweeps all N up to ceil(g.n / BLOCK_SIZE)
        // tiles via N_MASKED_STORE, so every cell already holds the K=[0,
        // fast_k) reduction. We only need K-tail RMW; the LDS K-tail kernel
        // covers the aligned interior [0, fast_n), this scalar tail covers
        // the N-tail strip [fast_n, n).
        const bool lds_k_tail_safe = (g.m_per_group >= TAIL_BLOCK_M) &&
                                     ((g.m_per_group % TAIL_BLOCK_M) == 0);
        const int row_block_base = (row / TAIL_BLOCK_M) * TAIL_BLOCK_M;
        const bool block_in_group =
            (row_block_base + TAIL_BLOCK_M <= s_offs[group_idx + 1]);
        const bool lds_k_rem_match = ((g.k - g.fast_k) == 64);
        // LDS K-tail kernel now covers FULL N (interior + N-tail), so skip
        // any cell it already corrected.
        if (needs_k_tail &&
            lds_k_tail_safe && lds_k_rem_match && block_in_group) {
            return;
        }
        for (int kk = k0; kk < g.k; ++kk) {
            acc += load_fp8_scalar(g.a, row, kk) *
                   load_fp8_scalar_grp(g.b, group_idx, kk, col);
        }
    }

    const float scaled = acc * resolve_combined_scale_grp(g);
    if (fast_covers_cell && needs_k_tail) {
        store_bf16_scalar(g.c, row, col,
                          load_bf16_scalar(g.c, row, col) + scaled);
    } else {
        store_bf16_scalar(g.c, row, col, scaled);
    }
}

template __global__ void grouped_tail_kernel<Layout::RCR>(const grouped_layout_globals);
template __global__ void grouped_tail_kernel<Layout::RRR>(const grouped_layout_globals);

template<Layout L, int K_REM>
__global__ void grouped_ktail_kernel_lds(const grouped_layout_globals g) {
    static_assert(L == Layout::RCR,
        "grouped_ktail_kernel_lds (FP8): RCR only — RRR/CRR fall back to scalar tail.");
    constexpr int TBM = TAIL_BLOCK_M;       // 16
    constexpr int TBN = TAIL_BLOCK_N;       // 16
    constexpr int NTHR = TBM * TBN;         // 256

    constexpr int K_REM_LDS = K_REM + 8;
    __shared__ fp8e4m3 A_lds[TBM * K_REM_LDS];
    __shared__ fp8e4m3 B_lds[TBN * K_REM_LDS];
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
    const int col_block_base = blockIdx.x * TBN;
    if (row_block_base >= g.M_total || col_block_base >= g.n) return;

    int group_idx = 0;
    #pragma unroll 1
    for (int gi = 0; gi < g.G; ++gi) {
        if (row_block_base < s_offs[gi + 1]) { group_idx = gi; break; }
    }

    const int k0 = g.fast_k;
    const int K_rem_dyn = g.k - k0;
    if (K_rem_dyn != K_REM) return;

    // Cross-group fallback: per-thread scalar K-tail RMW correction
    // (mirror BF16 round-6). Each row uses its own ``row_group`` for B
    // indexing. Vec8 fast path when k0 and g.k are 8-aligned.
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
            typedef __attribute__((__vector_size__(2 * sizeof(float)))) float fp32x2_v;
            auto fp8x4_to_f32x4 = [](const fp8e4m3_4& u) -> float4 {
                int packed;
                __builtin_memcpy(&packed, &u, 4);
                fp32x2_v lo = __builtin_amdgcn_cvt_pk_f32_fp8(packed, false);
                fp32x2_v hi = __builtin_amdgcn_cvt_pk_f32_fp8(packed, true);
                return make_float4(lo[0], lo[1], hi[0], hi[1]);
            };
            float acc_s0 = 0.0f, acc_s1 = 0.0f, acc_s2 = 0.0f, acc_s3 = 0.0f;
            const fp8e4m3* a_row = &g.a[coord<>(row, 0)];
            const fp8e4m3* b_row = &g.b[coord<>{0, row_group, col, 0}];
            int kk = k0;
            if ((g.k % 8 == 0) && ((k0 & 7) == 0)) {
                const fp8e4m3_8* a_v8 = reinterpret_cast<const fp8e4m3_8*>(a_row);
                const fp8e4m3_8* b_v8 = reinterpret_cast<const fp8e4m3_8*>(b_row);
                const int j_start = k0 >> 3;
                const int j_end   = g.k >> 3;
                for (int j = j_start; j < j_end; ++j) {
                    fp8e4m3_8 a8 = a_v8[j];
                    fp8e4m3_8 b8 = b_v8[j];
                    float4 a_lo = fp8x4_to_f32x4(a8.lo);
                    float4 a_hi = fp8x4_to_f32x4(a8.hi);
                    float4 b_lo = fp8x4_to_f32x4(b8.lo);
                    float4 b_hi = fp8x4_to_f32x4(b8.hi);
                    acc_s0 += a_lo.x * b_lo.x + a_hi.x * b_hi.x;
                    acc_s1 += a_lo.y * b_lo.y + a_hi.y * b_hi.y;
                    acc_s2 += a_lo.z * b_lo.z + a_hi.z * b_hi.z;
                    acc_s3 += a_lo.w * b_lo.w + a_hi.w * b_hi.w;
                }
                kk = j_end << 3;
            }
            float acc_s = (acc_s0 + acc_s1) + (acc_s2 + acc_s3);
            for (; kk < g.k; ++kk) {
                acc_s += load_fp8_scalar(g.a, row, kk) *
                         load_fp8_scalar_grp(g.b, row_group, col, kk);
            }
            const float scaled_s = acc_s * resolve_combined_scale_grp(g);
            store_bf16_scalar(g.c, row, col,
                              load_bf16_scalar(g.c, row, col) + scaled_s);
        }
        return;
    }

    constexpr int VEC = 4;
    constexpr int VECS_PER_ROW = K_REM / VEC;       // 16
    static_assert(K_REM % VEC == 0, "K_REM must be vec4-aligned");
    static_assert(NTHR == TBM * VECS_PER_ROW,
        "Each thread must own exactly one vec4 of A.");
    {
        const int r_in_blk = tid / VECS_PER_ROW;
        const int kk_v     = tid - r_in_blk * VECS_PER_ROW;
        const int kk_start = kk_v * VEC;
        const int r_global = row_block_base + r_in_blk;
        fp8e4m3_4 va{};
        if (r_global < g.M_total) {
            const fp8e4m3* ap = &g.a[coord<>(r_global, k0 + kk_start)];
            va = *reinterpret_cast<const fp8e4m3_4*>(ap);
        }
        *reinterpret_cast<fp8e4m3_4*>(&A_lds[r_in_blk * K_REM_LDS + kk_start]) = va;
    }
    {
        const int c_in_blk = tid / VECS_PER_ROW;
        const int kk_v     = tid - c_in_blk * VECS_PER_ROW;
        const int kk_start = kk_v * VEC;
        const int c_global = col_block_base + c_in_blk;
        fp8e4m3_4 vb{};
        if (c_global < g.n) {
            const fp8e4m3* bp = &g.b[coord<>{0, group_idx, c_global, k0 + kk_start}];
            vb = *reinterpret_cast<const fp8e4m3_4*>(bp);
        }
        *reinterpret_cast<fp8e4m3_4*>(&B_lds[c_in_blk * K_REM_LDS + kk_start]) = vb;
    }
    __syncthreads();

    const int row = row_block_base + rib;
    const int col = col_block_base + cib;
    if (row >= g.M_total || col >= g.n) return;

    // Vec8 inner fma: K_REM=64 / 8 = 8 vec8 per cell. Each vec8 LDS read
    // is one ds_read_b64 (8 bytes, 2 banks broadcast). Per vec8: two
    // fp8e4m3_4 → float4 conversions for both A and B → 8 fma.
    constexpr int FMA_VEC = 8;
    constexpr int K_VECS = K_REM / FMA_VEC;          // 8
    static_assert(K_REM % FMA_VEC == 0, "K_REM must be vec8-aligned for inner fma");
    float acc = 0.0f;
    typedef __attribute__((__vector_size__(2 * sizeof(float)))) float fp32x2_v;
    auto fp8x4_to_f32x4 = [](const fp8e4m3_4& u) -> float4 {
        int packed;
        __builtin_memcpy(&packed, &u, 4);
        fp32x2_v lo = __builtin_amdgcn_cvt_pk_f32_fp8(packed, false);
        fp32x2_v hi = __builtin_amdgcn_cvt_pk_f32_fp8(packed, true);
        return make_float4(lo[0], lo[1], hi[0], hi[1]);
    };

    // Round-15: split into 4 parallel fp32 accumulators to break the
    // 8-deep dependency chain in the per-thread fma loop.
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    #pragma unroll
    for (int kk_v = 0; kk_v < K_VECS; ++kk_v) {
        fp8e4m3_8 a8 = *reinterpret_cast<const fp8e4m3_8*>(
            &A_lds[rib * K_REM_LDS + kk_v * FMA_VEC]);
        fp8e4m3_8 b8 = *reinterpret_cast<const fp8e4m3_8*>(
            &B_lds[cib * K_REM_LDS + kk_v * FMA_VEC]);
        float4 a_lo = fp8x4_to_f32x4(a8.lo);
        float4 a_hi = fp8x4_to_f32x4(a8.hi);
        float4 b_lo = fp8x4_to_f32x4(b8.lo);
        float4 b_hi = fp8x4_to_f32x4(b8.hi);
        acc0 += a_lo.x * b_lo.x + a_hi.x * b_hi.x;
        acc1 += a_lo.y * b_lo.y + a_hi.y * b_hi.y;
        acc2 += a_lo.z * b_lo.z + a_hi.z * b_hi.z;
        acc3 += a_lo.w * b_lo.w + a_hi.w * b_hi.w;
    }
    acc = (acc0 + acc1) + (acc2 + acc3);

    // K-tail RMW correction. Main grouped kernel already wrote
    // [0, fast_k) × combined_scale at C[row, col]; we add the
    // [fast_k, k) × combined_scale slice. Mirror BF16 store.
    const float scaled = acc * resolve_combined_scale_grp(g);
    store_bf16_scalar(g.c, row, col,
                      load_bf16_scalar(g.c, row, col) + scaled);
}

template __global__ void grouped_ktail_kernel_lds<Layout::RCR, 64>(const grouped_layout_globals);

template<int K_REM>
__global__ void grouped_ktail_kernel_lds_rrr(const grouped_layout_globals g) {
    constexpr int TBM = TAIL_BLOCK_M;       // 16
    constexpr int TBN = TAIL_BLOCK_N;       // 16
    constexpr int NTHR = TBM * TBN;         // 256

    // K_REM_LDS = K_REM + 8 padding (= 72 fp8 = 72 bytes = 18 banks mod 32 = 18):
    // makes ``cib * 18 mod 32`` distribute the 16 cib lanes across 16 distinct
    // even banks for the ds_read_b64 in the inner loop. Mirror round-17 RCR.
    constexpr int K_REM_LDS = K_REM + 8;
    __shared__ fp8e4m3 A_lds[TBM * K_REM_LDS];
    __shared__ fp8e4m3 B_lds[TBN * K_REM_LDS];
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
    const int col_block_base = blockIdx.x * TBN;
    if (row_block_base >= g.M_total || col_block_base >= g.n) return;

    int group_idx = 0;
    #pragma unroll 1
    for (int gi = 0; gi < g.G; ++gi) {
        if (row_block_base < s_offs[gi + 1]) { group_idx = gi; break; }
    }

    const int k0 = g.fast_k;
    const int K_rem_dyn = g.k - k0;
    if (K_rem_dyn != K_REM) return;

    const bool cross_boundary = (row_block_base + TBM > s_offs[group_idx + 1]);
    if (cross_boundary) {
        // Per-row scalar fallback (B is stride-N in K → no vec along K).
        const int row = row_block_base + rib;
        const int col = col_block_base + cib;
        if (row < g.M_total && col < g.n) {
            int row_group = 0;
            #pragma unroll 1
            for (int gi = 0; gi < g.G; ++gi) {
                if (row < s_offs[gi + 1]) { row_group = gi; break; }
            }
            float acc_s = 0.0f;
            for (int kk = k0; kk < g.k; ++kk) {
                acc_s += load_fp8_scalar(g.a, row, kk) *
                         load_fp8_scalar_grp(g.b, row_group, kk, col);
            }
            const float scaled_s = acc_s * resolve_combined_scale_grp(g);
            store_bf16_scalar(g.c, row, col,
                              load_bf16_scalar(g.c, row, col) + scaled_s);
        }
        return;
    }

    // ---- Coop load A: [TBM, K_REM] fp8 from A[r_global, k0..k0+K_REM).
    // Same pattern as RCR round-9 (A stride-1 in K for both layouts).
    constexpr int VEC = 4;
    constexpr int A_VECS_PER_ROW = K_REM / VEC;       // 16
    static_assert(NTHR == TBM * A_VECS_PER_ROW,
        "NTHR must equal TBM * (K_REM / VEC) for vec4 coop load of A");
    {
        const int r_in_blk = tid / A_VECS_PER_ROW;
        const int kk_v = tid - r_in_blk * A_VECS_PER_ROW;
        const int kk_start = kk_v * VEC;
        const int r_global = row_block_base + r_in_blk;
        fp8e4m3_4 va{};
        if (r_global < g.M_total) {
            const fp8e4m3* ap = &g.a[coord<>(r_global, k0 + kk_start)];
            va = *reinterpret_cast<const fp8e4m3_4*>(ap);
        }
        *reinterpret_cast<fp8e4m3_4*>(&A_lds[r_in_blk * K_REM_LDS + kk_start]) = va;
    }

    // ---- Coop load B (RRR): [K_REM, TBN] fp8 from B[group, k0+kk, col_base..+TBN).
    // 4 contiguous N cols at fixed K = vec4 fp8 (4 bytes = 1 dword) HBM load.
    // SCATTER to LDS in [TBN, K_REM_LDS] transposed layout: 4 separate scalar
    // LDS stores per thread, but inner loop then reads vec8 along K stride-1.
    constexpr int B_VECS_PER_K = TBN / VEC;           // 4
    static_assert(NTHR == K_REM * B_VECS_PER_K,
        "NTHR must equal K_REM * (TBN / VEC) for vec4 coop load of B");
    {
        const int kk_in_blk = tid / B_VECS_PER_K;
        const int n_in_blk = (tid - kk_in_blk * B_VECS_PER_K) * VEC;
        const int kk_global = k0 + kk_in_blk;
        const int col_global = col_block_base + n_in_blk;
        fp8e4m3_4 vb{};
        // Zero-pad cols >= g.n. n_in_blk is 4-aligned so the 4 cols are
        // either fully in-bounds or some are >= g.n.
        if (col_global + VEC <= g.n) {
            const fp8e4m3* bp = &g.b[coord<>{0, group_idx, kk_global, col_global}];
            vb = *reinterpret_cast<const fp8e4m3_4*>(bp);
        } else if (col_global < g.n) {
            #pragma unroll
            for (int i = 0; i < VEC; ++i) {
                const int cg = col_global + i;
                if (cg < g.n) {
                    fp8e4m3 v = g.b[coord<>{0, group_idx, kk_global, cg}];
                    reinterpret_cast<fp8e4m3*>(&vb)[i] = v;
                }
            }
        }
        // Scatter to B_lds in [TBN, K_REM_LDS] transposed layout.
        const fp8e4m3* vb_arr = reinterpret_cast<const fp8e4m3*>(&vb);
        B_lds[(n_in_blk + 0) * K_REM_LDS + kk_in_blk] = vb_arr[0];
        B_lds[(n_in_blk + 1) * K_REM_LDS + kk_in_blk] = vb_arr[1];
        B_lds[(n_in_blk + 2) * K_REM_LDS + kk_in_blk] = vb_arr[2];
        B_lds[(n_in_blk + 3) * K_REM_LDS + kk_in_blk] = vb_arr[3];
    }
    __syncthreads();

    const int row = row_block_base + rib;
    const int col = col_block_base + cib;
    if (row >= g.M_total || col >= g.n) return;

    // Inner vec8 fma: K_REM/8 = 8 vec8 per cell. Mirror the FP8 RCR LDS
    // K-tail (line ~2754 in this file): 2 packed cvt_pk_f32_fp8 per fp8e4m3_4
    // operand, 4 parallel fp32 accumulators to break dependency chains.
    constexpr int FMA_VEC = 8;
    constexpr int K_VECS = K_REM / FMA_VEC;           // 8
    static_assert(K_REM % FMA_VEC == 0, "K_REM must be vec8-aligned for inner fma");
    typedef __attribute__((__vector_size__(2 * sizeof(float)))) float fp32x2_v;
    auto fp8x4_to_f32x4 = [](const fp8e4m3_4& u) -> float4 {
        int packed;
        __builtin_memcpy(&packed, &u, 4);
        fp32x2_v lo = __builtin_amdgcn_cvt_pk_f32_fp8(packed, false);
        fp32x2_v hi = __builtin_amdgcn_cvt_pk_f32_fp8(packed, true);
        return make_float4(lo[0], lo[1], hi[0], hi[1]);
    };
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    #pragma unroll
    for (int kk_v = 0; kk_v < K_VECS; ++kk_v) {
        fp8e4m3_8 a8 = *reinterpret_cast<const fp8e4m3_8*>(
            &A_lds[rib * K_REM_LDS + kk_v * FMA_VEC]);
        fp8e4m3_8 b8 = *reinterpret_cast<const fp8e4m3_8*>(
            &B_lds[cib * K_REM_LDS + kk_v * FMA_VEC]);
        float4 a_lo = fp8x4_to_f32x4(a8.lo);
        float4 a_hi = fp8x4_to_f32x4(a8.hi);
        float4 b_lo = fp8x4_to_f32x4(b8.lo);
        float4 b_hi = fp8x4_to_f32x4(b8.hi);
        acc0 += a_lo.x * b_lo.x + a_hi.x * b_hi.x;
        acc1 += a_lo.y * b_lo.y + a_hi.y * b_hi.y;
        acc2 += a_lo.z * b_lo.z + a_hi.z * b_hi.z;
        acc3 += a_lo.w * b_lo.w + a_hi.w * b_hi.w;
    }
    const float acc = (acc0 + acc1) + (acc2 + acc3);

    const float scaled = acc * resolve_combined_scale_grp(g);
    store_bf16_scalar(g.c, row, col,
                      load_bf16_scalar(g.c, row, col) + scaled);
}

template __global__ void grouped_ktail_kernel_lds_rrr<64>(const grouped_layout_globals);

template<int K_CHUNK>
__global__ void grouped_ntail_kernel_lds_rrr(const grouped_layout_globals g) {
    constexpr int TBM = TAIL_BLOCK_M;            // 16
    constexpr int TBN = TAIL_BLOCK_N;            // 16
    constexpr int NTHR = TBM * TBN;              // 256
    constexpr int K_CHUNK_LDS = K_CHUNK + 8;
    __shared__ fp8e4m3 A_lds[TBM * K_CHUNK_LDS];
    __shared__ fp8e4m3 B_lds[TBN * K_CHUNK_LDS];
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
    const int col_block_base = g.fast_n + blockIdx.x * TBN;
    if (row_block_base >= g.M_total || col_block_base >= g.n) return;

    int group_idx = 0;
    #pragma unroll 1
    for (int gi = 0; gi < g.G; ++gi) {
        if (row_block_base < s_offs[gi + 1]) { group_idx = gi; break; }
    }

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
            for (int kk = 0; kk < g.k; ++kk) {
                acc_s += load_fp8_scalar(g.a, row_s, kk) *
                         load_fp8_scalar_grp(g.b, row_group, kk, col_s);
            }
            const float scaled_s = acc_s * resolve_combined_scale_grp(g);
            store_bf16_scalar(g.c, row_s, col_s, scaled_s);
        }
        return;
    }

    const int row = row_block_base + rib;
    const int col = col_block_base + cib;
    const bool active_cell = (row < g.M_total) && (col < g.n);

    constexpr int VEC = 4;

    // ---- A coop-load layout: vec4 along K, NTHR = TBM × (K_CHUNK / VEC).
    constexpr int A_VECS_PER_ROW = K_CHUNK / VEC;        // 16
    static_assert(K_CHUNK % VEC == 0, "K_CHUNK must be vec4-aligned");
    static_assert(NTHR == TBM * A_VECS_PER_ROW,
        "NTHR must equal TBM * (K_CHUNK / VEC) for A coop load");
    const int r_in_blk_a = tid / A_VECS_PER_ROW;
    const int kk_v_a = tid - r_in_blk_a * A_VECS_PER_ROW;
    const int kk_start_a = kk_v_a * VEC;

    // ---- B coop-load layout: vec4 along N at fixed K, scatter to
    // transposed [TBN, K_CHUNK_LDS] LDS. NTHR = K_CHUNK × (TBN / VEC).
    constexpr int B_VECS_PER_K = TBN / VEC;              // 4
    static_assert(NTHR == K_CHUNK * B_VECS_PER_K,
        "NTHR must equal K_CHUNK * (TBN / VEC) for B coop load");
    const int kk_in_blk_b = tid / B_VECS_PER_K;
    const int n_in_blk_b = (tid - kk_in_blk_b * B_VECS_PER_K) * VEC;

    // 4 parallel fp32 accumulators to break the inner-fma dependency
    // chain (mirror round-15 FP8 RCR LDS K-tail).
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    typedef __attribute__((__vector_size__(2 * sizeof(float)))) float fp32x2_v;
    auto fp8x4_to_f32x4 = [](const fp8e4m3_4& u) -> float4 {
        int packed;
        __builtin_memcpy(&packed, &u, 4);
        fp32x2_v lo = __builtin_amdgcn_cvt_pk_f32_fp8(packed, false);
        fp32x2_v hi = __builtin_amdgcn_cvt_pk_f32_fp8(packed, true);
        return make_float4(lo[0], lo[1], hi[0], hi[1]);
    };

    for (int k_chunk_start = 0; k_chunk_start < g.k; k_chunk_start += K_CHUNK) {
        // Coop-load A (vec4 fp8 = 4 bytes / thread).
        {
            const int r_global = row_block_base + r_in_blk_a;
            const int k_global = k_chunk_start + kk_start_a;
            fp8e4m3_4 va{};
            if (r_global < g.M_total && k_global + VEC <= g.k) {
                const fp8e4m3* ap = &g.a[coord<>(r_global, k_global)];
                va = *reinterpret_cast<const fp8e4m3_4*>(ap);
            }
            *reinterpret_cast<fp8e4m3_4*>(&A_lds[r_in_blk_a * K_CHUNK_LDS + kk_start_a]) = va;
        }
        // Coop-load B (vec4 fp8 along N; scatter to transposed LDS).
        {
            const int kk_global = k_chunk_start + kk_in_blk_b;
            const int col_global = col_block_base + n_in_blk_b;
            fp8e4m3_4 vb{};
            if (kk_global < g.k && col_global + VEC <= g.n) {
                const fp8e4m3* bp = &g.b[coord<>{0, group_idx, kk_global, col_global}];
                vb = *reinterpret_cast<const fp8e4m3_4*>(bp);
            } else if (kk_global < g.k && col_global < g.n) {
                #pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    const int cg = col_global + i;
                    if (cg < g.n) {
                        fp8e4m3 v = g.b[coord<>{0, group_idx, kk_global, cg}];
                        reinterpret_cast<fp8e4m3*>(&vb)[i] = v;
                    }
                }
            }
            const fp8e4m3* vb_arr = reinterpret_cast<const fp8e4m3*>(&vb);
            B_lds[(n_in_blk_b + 0) * K_CHUNK_LDS + kk_in_blk_b] = vb_arr[0];
            B_lds[(n_in_blk_b + 1) * K_CHUNK_LDS + kk_in_blk_b] = vb_arr[1];
            B_lds[(n_in_blk_b + 2) * K_CHUNK_LDS + kk_in_blk_b] = vb_arr[2];
            B_lds[(n_in_blk_b + 3) * K_CHUNK_LDS + kk_in_blk_b] = vb_arr[3];
        }
        __syncthreads();

        if (active_cell) {
            constexpr int FMA_VEC = 8;
            constexpr int K_CHUNK_VECS = K_CHUNK / FMA_VEC;
            static_assert(K_CHUNK % FMA_VEC == 0, "K_CHUNK must be vec8-aligned for inner fma");
            const int k_left = g.k - k_chunk_start;
            const int k_iters_v = (k_left < K_CHUNK)
                ? (k_left + FMA_VEC - 1) / FMA_VEC : K_CHUNK_VECS;
            #pragma unroll
            for (int kk_v = 0; kk_v < K_CHUNK_VECS; ++kk_v) {
                if (kk_v >= k_iters_v) break;
                fp8e4m3_8 a8 = *reinterpret_cast<const fp8e4m3_8*>(
                    &A_lds[rib * K_CHUNK_LDS + kk_v * FMA_VEC]);
                fp8e4m3_8 b8 = *reinterpret_cast<const fp8e4m3_8*>(
                    &B_lds[cib * K_CHUNK_LDS + kk_v * FMA_VEC]);
                float4 a_lo = fp8x4_to_f32x4(a8.lo);
                float4 a_hi = fp8x4_to_f32x4(a8.hi);
                float4 b_lo = fp8x4_to_f32x4(b8.lo);
                float4 b_hi = fp8x4_to_f32x4(b8.hi);
                acc0 += a_lo.x * b_lo.x + a_hi.x * b_hi.x;
                acc1 += a_lo.y * b_lo.y + a_hi.y * b_hi.y;
                acc2 += a_lo.z * b_lo.z + a_hi.z * b_hi.z;
                acc3 += a_lo.w * b_lo.w + a_hi.w * b_hi.w;
            }
        }
        __syncthreads();
    }

    if (active_cell) {
        const float acc = (acc0 + acc1) + (acc2 + acc3);
        const float scaled = acc * resolve_combined_scale_grp(g);
        store_bf16_scalar(g.c, row, col, scaled);
    }
}

template __global__ void grouped_ntail_kernel_lds_rrr<64>(const grouped_layout_globals);

// =============================================================================
// MFMA-based RRR N-tail. Replaces grouped_ntail_kernel_lds_rrr's scalar fp8
// fma inner loop with mfma_f32_16x16x128_f8f6f4. Big-tile design: 64×64
// cells per block (4×4 mfma sub-tiles), 4 waves cooperating, A and B both
// LDS-staged so each (16,16) sub-tile reads its slice from LDS.
//
// Layout: A is RRR row-major [M,K]; B is RRR row-major [G,K,N]. Both are
// staged into LDS each K_TILE iter (A row-major, B transposed to col-major
// in K so MFMA can b256-load 32 fp8 of B per lane in the right register
// shape).
//
// K-tail (g.k % K_TILE != 0): the last K_TILE iter zero-pads A and B beyond
// g.k. No downstream RMW.
//
// Restrictions (caller verifies):
//   * m_per_group is a TBM=64 multiple AND >= TBM (no cross-group cell can
//     straddle an m_per_group boundary on a TBM-aligned grid)
//   * Output store is OVERWRITE — main kernel does not write [fast_n, n)
//     for RRR.
// =============================================================================
template<int K_TILE = 128>
__global__ void grouped_ntail_kernel_mfma_rrr(const grouped_layout_globals g) {
    static_assert(K_TILE == 128, "K_TILE must equal mfma_16x16x128 K dim");
    constexpr int TBM = 64;
    constexpr int TBN = 64;
    constexpr int MFMA_M = 16, MFMA_N = 16;
    constexpr int M_TILES = TBM / MFMA_M;       // 4
    constexpr int N_TILES = TBN / MFMA_N;       // 4
    constexpr int K_PER_LANE = 32;              // 128 / 4 lane-chunks
    constexpr int K_LDS_PAD  = 8;
    constexpr int K_LDS      = K_TILE + K_LDS_PAD;
    constexpr int MAX_G_PLUS_1 = 65;
    constexpr int NTHR = 256;                   // 4 waves
    __shared__ fp8e4m3 A_lds[TBM * K_LDS];      // [row][k], row-major
    __shared__ fp8e4m3 B_lds[TBN * K_LDS];      // [col][k], col-major in K
    __shared__ int s_offs[MAX_G_PLUS_1];

    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int   intx8_t;
    typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;

    const int tid = threadIdx.x;
    if (tid <= g.G && tid < MAX_G_PLUS_1) {
        s_offs[tid] = static_cast<int>(g.group_offs[tid]);
    }
    __syncthreads();

    const int row_block_base = blockIdx.y * TBM;
    const int col_block_base = g.fast_n + blockIdx.x * TBN;
    if (row_block_base >= g.M_total || col_block_base >= g.n) return;

    int group_idx = 0;
    #pragma unroll 1
    for (int gi = 0; gi < g.G; ++gi) {
        if (row_block_base < s_offs[gi + 1]) { group_idx = gi; break; }
    }
    // Caller guarantees m_per_group % TBM == 0 → blocks lie wholly in one group.
    // Skip via early return if for any reason block straddles a group; the
    // scalar tail kernel will fill these cells (its skip predicate excludes
    // them via the `block_in_group` check).
    if (row_block_base + TBM > s_offs[group_idx + 1]) return;

    // ---- Persistent per-block state: 4×4 grid of float4 accumulators.
    floatx4_t acc[M_TILES][N_TILES];
    #pragma unroll
    for (int mi = 0; mi < M_TILES; ++mi)
    #pragma unroll
    for (int ni = 0; ni < N_TILES; ++ni)
        acc[mi][ni] = floatx4_t{0.f, 0.f, 0.f, 0.f};

    // Wave layout: 4 waves total. wave_id = tid / 64 ∈ [0,4).
    // We let wave w handle M-rows [w*16, w*16+16) of the block.
    // Each wave then iterates over its 4 N-tiles.
    const int wave_id = tid / 64;
    const int lane    = tid % 64;
    const int row_in_mfma = lane % MFMA_M;       // 0..15
    const int chunk       = lane / MFMA_M;       // 0..3 (K lane-chunk)
    const bool full_n_block = (col_block_base + TBN <= g.n);

    for (int k_off = 0; k_off < g.k; k_off += K_TILE) {
        // ---- A coop-load: NTHR=256 lanes, each loads vec-of-8 (intx8_t)
        // = 32 fp8 from A. 256 × 32 = 8192 fp8 = TBM(64) × K_TILE(128) ✓.
        // Lane (l) → row = l % 64, chunk_id = l / 64 (∈ 0..3).
        // Reads A[row_block_base + row, k_off + chunk_id*32 .. +32].
        {
            const int a_row = tid % TBM;                 // 0..63
            const int a_chunk = tid / TBM;               // 0..3
            const int g_row = row_block_base + a_row;
            const int k_chunk_off = k_off + a_chunk * K_PER_LANE;
            intx8_t a_pack = intx8_t{};
            if (g_row < g.M_total && k_chunk_off + K_PER_LANE <= g.k) {
                const fp8e4m3* ap = &g.a[coord<>(g_row, k_chunk_off)];
                a_pack = *reinterpret_cast<const intx8_t*>(ap);
            } else if (g_row < g.M_total && k_chunk_off < g.k) {
                const int k_avail = g.k - k_chunk_off;
                #pragma unroll
                for (int i = 0; i < K_PER_LANE; ++i) {
                    if (i < k_avail) {
                        reinterpret_cast<fp8e4m3*>(&a_pack)[i] =
                            g.a[coord<>(g_row, k_chunk_off + i)];
                    }
                }
            }
            *reinterpret_cast<intx8_t*>(&A_lds[a_row * K_LDS + a_chunk * K_PER_LANE]) = a_pack;
        }

        // ---- B coop-load + scatter: NTHR=256 lanes, each loads 1 b128
        // (16 fp8) from one K-row (one (k, 16-col strip)). 256 × 16 = 4096
        // fp8 per pass = 32 K rows × TBN(64). Need 4 passes for K=128.
        // Lane (l) → col_strip = l % 4, k_in_chunk = l / 4 (∈ 0..63).
        #pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            const int col_strip = (tid % 4);                        // 0..3
            const int k_local   = (tid / 4) + pass * 64;            // 0..127
            const int g_col_strip_base = col_block_base + col_strip * 16;
            const int k_global = k_off + k_local;
            int4 vbi = int4{0, 0, 0, 0};
            if (k_global < g.k) {
                if (full_n_block || g_col_strip_base + 16 <= g.n) {
                    const fp8e4m3* bp =
                        &g.b[coord<>{0, group_idx, k_global, g_col_strip_base}];
                    vbi = *reinterpret_cast<const int4*>(bp);
                } else if (g_col_strip_base < g.n) {
                    fp8e4m3* vbp = reinterpret_cast<fp8e4m3*>(&vbi);
                    #pragma unroll
                    for (int c = 0; c < 16; ++c) {
                        const int cg = g_col_strip_base + c;
                        if (cg < g.n) {
                            vbp[c] = g.b[coord<>{0, group_idx, k_global, cg}];
                        }
                    }
                }
            }
            const fp8e4m3* vb_arr = reinterpret_cast<const fp8e4m3*>(&vbi);
            #pragma unroll
            for (int c = 0; c < 16; ++c) {
                B_lds[(col_strip * 16 + c) * K_LDS + k_local] = vb_arr[c];
            }
        }
        __syncthreads();

        // ---- 4×4 grid of MFMAs: each wave w handles M-rows [w*16, w*16+16);
        // iterates over 4 N-tiles (col_strip 0..3).
        #pragma unroll
        for (int ni = 0; ni < N_TILES; ++ni) {
            const int b_lds_col = ni * MFMA_N + row_in_mfma;
            intx8_t b_pack = *reinterpret_cast<const intx8_t*>(
                &B_lds[b_lds_col * K_LDS + chunk * K_PER_LANE]);
            const int a_lds_row = wave_id * MFMA_M + row_in_mfma;
            intx8_t a_pack = *reinterpret_cast<const intx8_t*>(
                &A_lds[a_lds_row * K_LDS + chunk * K_PER_LANE]);
            acc[wave_id][ni] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                a_pack, b_pack, acc[wave_id][ni],
                /*cbsz=*/0, /*abid=*/0, /*blgp=*/0,
                /*scale_op_a=*/0, /*scale_op_b=*/0, /*scale_op_d=*/0);
        }
        __syncthreads();
    }

    // ---- Store. Each wave w writes its 16 M-rows × 64 N-cols (4 N-tiles).
    // Per-tile distribution: lane t in MFMA holds D[(t/16)*4 + 0..3, t%16].
    const float scale = resolve_combined_scale_grp(g);
    #pragma unroll
    for (int ni = 0; ni < N_TILES; ++ni) {
        const int out_col = col_block_base + ni * MFMA_N + row_in_mfma;
        if (out_col >= g.n) continue;
        const int out_row_base = row_block_base + wave_id * MFMA_M + chunk * 4;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = out_row_base + i;
            if (r >= g.M_total) continue;
            store_bf16_scalar(g.c, r, out_col, acc[wave_id][ni][i] * scale);
        }
    }
}

template __global__ void grouped_ntail_kernel_mfma_rrr<128>(const grouped_layout_globals);

template<Layout L, int K_REM>
__global__ void grouped_ktail_kernel_mfma(const grouped_layout_globals g) {
    static_assert(L == Layout::RCR,
        "grouped_ktail_kernel_mfma (FP8): RCR only — RRR/CRR fall back to scalar tail.");
    static_assert(K_REM == 64,
        "grouped_ktail_kernel_mfma (FP8): K_REM must be 64 (zero-padded to 128).");
    constexpr int TBM = TAIL_BLOCK_M;       // 16
    constexpr int TBN = TAIL_BLOCK_N;       // 16
    constexpr int K_PER_LANE_CHUNK = 32;    // mfma_16x16x128 distributes K across 4 lane-chunks
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

    // Cross-group fallback (per-row vec8 scalar fma + cvt_pk_f32_fp8).
    // Mirror ``grouped_ktail_kernel_lds`` cross_boundary path. Single
    // wave: each lane handles a (row, col) cell within the 16×16 block.
    const bool cross_boundary = (row_block_base + TBM > s_offs[group_idx + 1]);
    if (cross_boundary) {
        typedef __attribute__((__vector_size__(2 * sizeof(float)))) float fp32x2_v;
        auto fp8x4_to_f32x4 = [](const fp8e4m3_4& u) -> float4 {
            int packed;
            __builtin_memcpy(&packed, &u, 4);
            fp32x2_v lo = __builtin_amdgcn_cvt_pk_f32_fp8(packed, false);
            fp32x2_v hi = __builtin_amdgcn_cvt_pk_f32_fp8(packed, true);
            return make_float4(lo[0], lo[1], hi[0], hi[1]);
        };
        // 64 lanes cover the 256 cells in 4 passes (4 cells/lane).
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
            float acc_s0 = 0.f, acc_s1 = 0.f, acc_s2 = 0.f, acc_s3 = 0.f;
            const fp8e4m3* a_row = &g.a[coord<>(row, 0)];
            const fp8e4m3* b_row = &g.b[coord<>{0, row_group, col, 0}];
            int kk = k0;
            if ((g.k % 8 == 0) && ((k0 & 7) == 0)) {
                const fp8e4m3_8* a_v8 = reinterpret_cast<const fp8e4m3_8*>(a_row);
                const fp8e4m3_8* b_v8 = reinterpret_cast<const fp8e4m3_8*>(b_row);
                const int j_start = k0 >> 3;
                const int j_end   = g.k >> 3;
                for (int j = j_start; j < j_end; ++j) {
                    fp8e4m3_8 a8 = a_v8[j];
                    fp8e4m3_8 b8 = b_v8[j];
                    float4 a_lo = fp8x4_to_f32x4(a8.lo);
                    float4 a_hi = fp8x4_to_f32x4(a8.hi);
                    float4 b_lo = fp8x4_to_f32x4(b8.lo);
                    float4 b_hi = fp8x4_to_f32x4(b8.hi);
                    acc_s0 += a_lo.x * b_lo.x + a_hi.x * b_hi.x;
                    acc_s1 += a_lo.y * b_lo.y + a_hi.y * b_hi.y;
                    acc_s2 += a_lo.z * b_lo.z + a_hi.z * b_hi.z;
                    acc_s3 += a_lo.w * b_lo.w + a_hi.w * b_hi.w;
                }
                kk = j_end << 3;
            }
            float acc_s = (acc_s0 + acc_s1) + (acc_s2 + acc_s3);
            for (; kk < g.k; ++kk) {
                acc_s += load_fp8_scalar(g.a, row, kk) *
                         load_fp8_scalar_grp(g.b, row_group, col, kk);
            }
            const float scaled_s = acc_s * resolve_combined_scale_grp(g);
            store_bf16_scalar(g.c, row, col,
                              load_bf16_scalar(g.c, row, col) + scaled_s);
        }
        return;
    }

    // ----- Fast MFMA path ------------------------------------------------
    // Lane (t):  row_in_blk = t % 16,  k_chunk = t / 16  (0..3).
    //   * Lane 0..15  (chunk=0): A row=row_in_blk, K=[k0+0,  k0+32)  — real
    //   * Lane 16..31 (chunk=1): A row=row_in_blk, K=[k0+32, k0+64)  — real
    //   * Lane 32..47 (chunk=2): A row=row_in_blk, K=[k0+64, k0+96)  — pad 0
    //   * Lane 48..63 (chunk=3): A row=row_in_blk, K=[k0+96, k0+128) — pad 0
    //   B mirrors A with col=row_in_blk replacing row=row_in_blk (RCR ABt).
    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
    typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;

    const int row_in_blk = tid % TBM;
    const int chunk      = tid / TBM;

    intx8_t a_pack;
    intx8_t b_pack;
    if (chunk < 2) {
        const int k_off = k0 + chunk * K_PER_LANE_CHUNK;
        const int g_row = row_block_base + row_in_blk;
        const int g_col = col_block_base + row_in_blk;
        // K=2880, k0=2816 → 32-byte aligned. row strides = K = 2880 (fp8) →
        // 32-byte aligned. Single 32-byte buffer load = 2 × b128.
        if (g_row < g.M_total) {
            const fp8e4m3* a_ptr = &g.a[coord<>(g_row, k_off)];
            a_pack = *reinterpret_cast<const intx8_t*>(a_ptr);
        } else {
            a_pack = intx8_t{};
        }
        if (g_col < g.n) {
            const fp8e4m3* b_ptr = &g.b[coord<>{0, group_idx, g_col, k_off}];
            b_pack = *reinterpret_cast<const intx8_t*>(b_ptr);
        } else {
            b_pack = intx8_t{};
        }
    } else {
        a_pack = intx8_t{};
        b_pack = intx8_t{};
    }

    // mfma_scale_f32_16x16x128_f8f6f4 — D[16,16] = A[16,128] @ B^T[128,16] + 0
    floatx4_t acc = floatx4_t{0.f, 0.f, 0.f, 0.f};
    acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a_pack, b_pack, acc, /*cbsz=*/0, /*abid=*/0, /*blgp=*/0,
        /*scale_op_a=*/0, /*scale_op_b=*/0, /*scale_op_d=*/0);

    // Output: lane t → cells C[(t/16)*4 + (0..3), t%16].
    const float scale = resolve_combined_scale_grp(g);
    const int out_row_base = row_block_base + chunk * 4;
    const int out_col      = col_block_base + row_in_blk;
    if (out_col >= g.n) return;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = out_row_base + i;
        if (r >= g.M_total) break;
        const float existing = load_bf16_scalar(g.c, r, out_col);
        const float new_val  = existing + acc[i] * scale;
        store_bf16_scalar(g.c, r, out_col, new_val);
    }
}

template __global__ void grouped_ktail_kernel_mfma<Layout::RCR, 64>(const grouped_layout_globals);

template<Layout L, int K_REM>
__global__ void grouped_ktail_kernel_mfma32x32(const grouped_layout_globals g) {
    static_assert(L == Layout::RCR,
        "grouped_ktail_kernel_mfma32x32 (FP8): RCR only.");
    static_assert(K_REM == 64,
        "grouped_ktail_kernel_mfma32x32 (FP8): K_REM must be 64 (native mfma K).");
    constexpr int TBM = 32;
    constexpr int TBN = 32;
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

    const bool cross_boundary = (row_block_base + TBM > s_offs[group_idx + 1]);
    if (cross_boundary) {
        typedef __attribute__((__vector_size__(2 * sizeof(float)))) float fp32x2_v;
        auto fp8x4_to_f32x4 = [](const fp8e4m3_4& u) -> float4 {
            int packed;
            __builtin_memcpy(&packed, &u, 4);
            fp32x2_v lo = __builtin_amdgcn_cvt_pk_f32_fp8(packed, false);
            fp32x2_v hi = __builtin_amdgcn_cvt_pk_f32_fp8(packed, true);
            return make_float4(lo[0], lo[1], hi[0], hi[1]);
        };
        // 64 lanes cover the 1024 cells in 16 passes (16 cells/lane).
        // Pass slot s ∈ 0..15 → cell offset within block:
        //   rib = s / 2 * 2 + (tid / 32);  // 16 row stripes
        //   cib = (s % 2) * 32 + (tid % 32);  // 32 cols / pass
        // Hmm simpler: each lane does 16 passes, in pass s walks (rib, cib).
        // Use the standard tiled traversal: each lane "owns" col=tid%32
        // and walks 16 distinct rows per col.
        const int col = col_block_base + (tid % 32);
        if (col < g.n) {
            #pragma unroll
            for (int rr = 0; rr < TBM; ++rr) {
                if ((rr % 2) != ((tid / 32))) continue;
                const int row = row_block_base + rr;
                if (row >= g.M_total) break;
                int row_group = 0;
                #pragma unroll 1
                for (int gi = 0; gi < g.G; ++gi) {
                    if (row < s_offs[gi + 1]) { row_group = gi; break; }
                }
                float acc_s0 = 0.f, acc_s1 = 0.f, acc_s2 = 0.f, acc_s3 = 0.f;
                const fp8e4m3* a_row = &g.a[coord<>(row, 0)];
                const fp8e4m3* b_row = &g.b[coord<>{0, row_group, col, 0}];
                int kk = k0;
                if ((g.k % 8 == 0) && ((k0 & 7) == 0)) {
                    const fp8e4m3_8* a_v8 = reinterpret_cast<const fp8e4m3_8*>(a_row);
                    const fp8e4m3_8* b_v8 = reinterpret_cast<const fp8e4m3_8*>(b_row);
                    const int j_start = k0 >> 3;
                    const int j_end   = g.k >> 3;
                    for (int j = j_start; j < j_end; ++j) {
                        fp8e4m3_8 a8 = a_v8[j];
                        fp8e4m3_8 b8 = b_v8[j];
                        float4 a_lo = fp8x4_to_f32x4(a8.lo);
                        float4 a_hi = fp8x4_to_f32x4(a8.hi);
                        float4 b_lo = fp8x4_to_f32x4(b8.lo);
                        float4 b_hi = fp8x4_to_f32x4(b8.hi);
                        acc_s0 += a_lo.x * b_lo.x + a_hi.x * b_hi.x;
                        acc_s1 += a_lo.y * b_lo.y + a_hi.y * b_hi.y;
                        acc_s2 += a_lo.z * b_lo.z + a_hi.z * b_hi.z;
                        acc_s3 += a_lo.w * b_lo.w + a_hi.w * b_hi.w;
                    }
                    kk = j_end << 3;
                }
                float acc_s = (acc_s0 + acc_s1) + (acc_s2 + acc_s3);
                for (; kk < g.k; ++kk) {
                    acc_s += load_fp8_scalar(g.a, row, kk) *
                             load_fp8_scalar_grp(g.b, row_group, col, kk);
                }
                const float scaled_s = acc_s * resolve_combined_scale_grp(g);
                store_bf16_scalar(g.c, row, col,
                                  load_bf16_scalar(g.c, row, col) + scaled_s);
            }
        }
        return;
    }

    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;

    const int row_in_blk = tid % 32;
    const int chunk      = tid / 32;        // 0 or 1
    const int k_off      = k0 + chunk * 32;

    const int g_row = row_block_base + row_in_blk;
    const int g_col = col_block_base + row_in_blk;

    intx8_t a_pack;
    intx8_t b_pack;
    // K=2880 → k0=2816 → k_off ∈ {2816, 2848}; row stride = K = 2880 fp8.
    // Both 32-byte aligned (2816 % 32 = 0, 2848 % 32 = 0, 2880 % 32 = 0)
    // so one buffer load per pack = 2 × b128.
    if (g_row < g.M_total) {
        const fp8e4m3* a_ptr = &g.a[coord<>(g_row, k_off)];
        a_pack = *reinterpret_cast<const intx8_t*>(a_ptr);
    } else {
        a_pack = intx8_t{};
    }
    if (g_col < g.n) {
        const fp8e4m3* b_ptr = &g.b[coord<>{0, group_idx, g_col, k_off}];
        b_pack = *reinterpret_cast<const intx8_t*>(b_ptr);
    } else {
        b_pack = intx8_t{};
    }

    // mfma_scale_f32_32x32x64_f8f6f4 — D[32,32] = A[32,64] @ B^T[64,32] + C
    floatx16_t acc{};
    acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a_pack, b_pack, acc, /*cbsz=*/0, /*abid=*/0, /*blgp=*/0,
        /*scale_op_a=*/0, /*scale_op_b=*/0, /*scale_op_d=*/0);

    // Output: lane t → cells C[r, t%32] for 16 r's.
    //   chunk=0: rows 0..3, 8..11, 16..19, 24..27 in d[0..3], d[4..7], d[8..11], d[12..15]
    //   chunk=1: rows 4..7, 12..15, 20..23, 28..31
    // Unified: row_group = i/4; row_in_group = (i%4) + chunk*4; row = row_group*8 + row_in_group.
    const float scale = resolve_combined_scale_grp(g);
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
        const float new_val  = existing + acc[i] * scale;
        store_bf16_scalar(g.c, r, out_col, new_val);
    }
}

template __global__ void grouped_ktail_kernel_mfma32x32<Layout::RCR, 64>(const grouped_layout_globals);

template<Layout L, int K_REM>
__global__ void grouped_ktail_kernel_mfma32x32_M2(const grouped_layout_globals g) {
    static_assert(L == Layout::RCR,
        "grouped_ktail_kernel_mfma32x32_M2 (FP8): RCR only.");
    static_assert(K_REM == 64,
        "grouped_ktail_kernel_mfma32x32_M2 (FP8): K_REM must be 64 (native mfma K).");
    constexpr int TBM_TOTAL = 64;       // 2 stacked 32×32 sub-blocks
    constexpr int TBM_SUB   = 32;
    constexpr int TBN       = 32;
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];

    const int tid = threadIdx.x;            // single-wave block, 64 threads
    if (tid <= g.G && tid < MAX_G_PLUS_1) {
        s_offs[tid] = static_cast<int>(g.group_offs[tid]);
    }
    __syncthreads();

    const int row_block_base = blockIdx.y * TBM_TOTAL;
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

    // Cross-group fallback: if the 64-row block straddles a group boundary
    // (or the M_total tail), fall back to per-row scalar fma over BOTH
    // sub-blocks. Unreachable on uniform group_lens with M_g % 64 == 0
    // (gpt_oss M_per ∈ {2048, 4096} both satisfy).
    const bool cross_boundary = (row_block_base + TBM_TOTAL > s_offs[group_idx + 1]);
    if (cross_boundary) {
        typedef __attribute__((__vector_size__(2 * sizeof(float)))) float fp32x2_v;
        auto fp8x4_to_f32x4 = [](const fp8e4m3_4& u) -> float4 {
            int packed;
            __builtin_memcpy(&packed, &u, 4);
            fp32x2_v lo = __builtin_amdgcn_cvt_pk_f32_fp8(packed, false);
            fp32x2_v hi = __builtin_amdgcn_cvt_pk_f32_fp8(packed, true);
            return make_float4(lo[0], lo[1], hi[0], hi[1]);
        };
        // 64 lanes × 32 cells/lane (the 64-row × 32-col block has 2048
        // cells; chunk-0 lanes serve even-rr and chunk-1 lanes serve
        // odd-rr just like the round-20 M1 fallback, half work each).
        const int col = col_block_base + (tid % 32);
        if (col < g.n) {
            #pragma unroll 1
            for (int rr = 0; rr < TBM_TOTAL; ++rr) {
                if ((rr % 2) != ((tid / 32))) continue;
                const int row = row_block_base + rr;
                if (row >= g.M_total) break;
                int row_group = 0;
                #pragma unroll 1
                for (int gi = 0; gi < g.G; ++gi) {
                    if (row < s_offs[gi + 1]) { row_group = gi; break; }
                }
                float acc_s0 = 0.f, acc_s1 = 0.f, acc_s2 = 0.f, acc_s3 = 0.f;
                const fp8e4m3* a_row = &g.a[coord<>(row, 0)];
                const fp8e4m3* b_row = &g.b[coord<>{0, row_group, col, 0}];
                int kk = k0;
                if ((g.k % 8 == 0) && ((k0 & 7) == 0)) {
                    const fp8e4m3_8* a_v8 = reinterpret_cast<const fp8e4m3_8*>(a_row);
                    const fp8e4m3_8* b_v8 = reinterpret_cast<const fp8e4m3_8*>(b_row);
                    const int j_start = k0 >> 3;
                    const int j_end   = g.k >> 3;
                    for (int j = j_start; j < j_end; ++j) {
                        fp8e4m3_8 a8 = a_v8[j];
                        fp8e4m3_8 b8 = b_v8[j];
                        float4 a_lo = fp8x4_to_f32x4(a8.lo);
                        float4 a_hi = fp8x4_to_f32x4(a8.hi);
                        float4 b_lo = fp8x4_to_f32x4(b8.lo);
                        float4 b_hi = fp8x4_to_f32x4(b8.hi);
                        acc_s0 += a_lo.x * b_lo.x + a_hi.x * b_hi.x;
                        acc_s1 += a_lo.y * b_lo.y + a_hi.y * b_hi.y;
                        acc_s2 += a_lo.z * b_lo.z + a_hi.z * b_hi.z;
                        acc_s3 += a_lo.w * b_lo.w + a_hi.w * b_hi.w;
                    }
                    kk = j_end << 3;
                }
                float acc_s = (acc_s0 + acc_s1) + (acc_s2 + acc_s3);
                for (; kk < g.k; ++kk) {
                    acc_s += load_fp8_scalar(g.a, row, kk) *
                             load_fp8_scalar_grp(g.b, row_group, col, kk);
                }
                const float scaled_s = acc_s * resolve_combined_scale_grp(g);
                store_bf16_scalar(g.c, row, col,
                                  load_bf16_scalar(g.c, row, col) + scaled_s);
            }
        }
        return;
    }

    // ----- Fast MFMA path: shared B-pack across two stacked 32×32 sub-blocks
    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;

    const int row_in_blk = tid % 32;
    const int chunk      = tid / 32;        // 0 or 1
    const int k_off      = k0 + chunk * 32;

    const int g_col = col_block_base + row_in_blk;

    // Single B load shared across both sub-blocks. B is [G, N, K] row-major
    // and the K-tail slice [k0, k0+K_REM) is the same for sub-block 0 and
    // sub-block 1 (only the A row range differs). Halves the B-side HBM
    // bandwidth vs running two independent 32×32 M1 blocks.
    intx8_t b_pack;
    if (g_col < g.n) {
        const fp8e4m3* b_ptr = &g.b[coord<>{0, group_idx, g_col, k_off}];
        b_pack = *reinterpret_cast<const intx8_t*>(b_ptr);
    } else {
        b_pack = intx8_t{};
    }

    const float scale = resolve_combined_scale_grp(g);
    const int out_col = col_block_base + row_in_blk;
    if (out_col >= g.n) return;

    // Two stacked 32×32 sub-blocks. Same MFMA / lane layout as round-20
    // M1; we just shift the row base by +TBM_SUB on sub-block 1.
    #pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
        const int sub_row_base = row_block_base + sub * TBM_SUB;
        const int g_row = sub_row_base + row_in_blk;

        intx8_t a_pack;
        if (g_row < g.M_total) {
            const fp8e4m3* a_ptr = &g.a[coord<>(g_row, k_off)];
            a_pack = *reinterpret_cast<const intx8_t*>(a_ptr);
        } else {
            a_pack = intx8_t{};
        }

        floatx16_t acc{};
        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_pack, b_pack, acc, /*cbsz=*/0, /*abid=*/0, /*blgp=*/0,
            /*scale_op_a=*/0, /*scale_op_b=*/0, /*scale_op_d=*/0);

        // Output cell mapping (mirror M1): lane t at d[i] (i=0..15) →
        //   row_group     = i / 4
        //   row_in_group  = (i % 4) + chunk * 4
        //   row           = row_group * 8 + row_in_group
        // i.e. chunk-0 lane owns rows {0..3, 8..11, 16..19, 24..27} per col,
        //      chunk-1 lane owns rows {4..7, 12..15, 20..23, 28..31} per col.
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int row_group     = i >> 2;
            const int row_in_group  = (i & 3) + chunk * 4;
            const int local_row     = row_group * 8 + row_in_group;
            const int r             = sub_row_base + local_row;
            if (r >= g.M_total) continue;
            const float existing = load_bf16_scalar(g.c, r, out_col);
            const float new_val  = existing + acc[i] * scale;
            store_bf16_scalar(g.c, r, out_col, new_val);
        }
    }
}

template __global__ void grouped_ktail_kernel_mfma32x32_M2<Layout::RCR, 64>(const grouped_layout_globals);

template<Layout L, int K_REM>
__global__ void grouped_ktail_kernel_mfma32x32_M2N2(const grouped_layout_globals g) {
    static_assert(L == Layout::RCR,
        "grouped_ktail_kernel_mfma32x32_M2N2 (FP8): RCR only.");
    static_assert(K_REM == 64,
        "grouped_ktail_kernel_mfma32x32_M2N2 (FP8): K_REM must be 64 (native mfma K).");
    constexpr int TBM_TOTAL = 64;       // 2 stacked 32×32 sub-blocks
    constexpr int TBM_SUB   = 32;
    constexpr int TBN       = 32;
    constexpr int N_SUB     = 2;
    constexpr int TBN_TOTAL = TBN * N_SUB;  // 64 cols
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];

    const int tid = threadIdx.x;            // single-wave block, 64 threads
    if (tid <= g.G && tid < MAX_G_PLUS_1) {
        s_offs[tid] = static_cast<int>(g.group_offs[tid]);
    }
    __syncthreads();

    const int row_block_base = blockIdx.y * TBM_TOTAL;
    const int col_block_base = blockIdx.x * TBN_TOTAL;
    if (row_block_base >= g.M_total || col_block_base >= g.n) return;

    int group_idx = 0;
    #pragma unroll 1
    for (int gi = 0; gi < g.G; ++gi) {
        if (row_block_base < s_offs[gi + 1]) { group_idx = gi; break; }
    }

    const int K_rem_dyn = g.k - g.fast_k;
    if (K_rem_dyn != K_REM) return;
    const int k0 = g.fast_k;

    // Cross-group fallback: same as M2 (64-row block straddling a group
    // boundary). Iterate over BOTH 32-col sub-tiles per row using scalar
    // fma. Unreachable on uniform group_lens with M_g % 64 == 0.
    const bool cross_boundary = (row_block_base + TBM_TOTAL > s_offs[group_idx + 1]);
    if (cross_boundary) {
        typedef __attribute__((__vector_size__(2 * sizeof(float)))) float fp32x2_v;
        auto fp8x4_to_f32x4 = [](const fp8e4m3_4& u) -> float4 {
            int packed;
            __builtin_memcpy(&packed, &u, 4);
            fp32x2_v lo = __builtin_amdgcn_cvt_pk_f32_fp8(packed, false);
            fp32x2_v hi = __builtin_amdgcn_cvt_pk_f32_fp8(packed, true);
            return make_float4(lo[0], lo[1], hi[0], hi[1]);
        };
        const float scale_s = resolve_combined_scale_grp(g);
        // 64 lanes split: chunk-0 (tid<32) covers even-rr, chunk-1 odd-rr
        // for each of the 2 col sub-tiles.
        for (int nt = 0; nt < N_SUB; ++nt) {
            const int col = col_block_base + nt * TBN + (tid % 32);
            if (col >= g.n) continue;
            #pragma unroll 1
            for (int rr = 0; rr < TBM_TOTAL; ++rr) {
                if ((rr % 2) != ((tid / 32))) continue;
                const int row = row_block_base + rr;
                if (row >= g.M_total) break;
                int row_group = 0;
                #pragma unroll 1
                for (int gi = 0; gi < g.G; ++gi) {
                    if (row < s_offs[gi + 1]) { row_group = gi; break; }
                }
                float acc_s0 = 0.f, acc_s1 = 0.f, acc_s2 = 0.f, acc_s3 = 0.f;
                const fp8e4m3* a_row = &g.a[coord<>(row, 0)];
                const fp8e4m3* b_row = &g.b[coord<>{0, row_group, col, 0}];
                int kk = k0;
                if ((g.k % 8 == 0) && ((k0 & 7) == 0)) {
                    const fp8e4m3_8* a_v8 = reinterpret_cast<const fp8e4m3_8*>(a_row);
                    const fp8e4m3_8* b_v8 = reinterpret_cast<const fp8e4m3_8*>(b_row);
                    const int j_start = k0 >> 3;
                    const int j_end   = g.k >> 3;
                    for (int j = j_start; j < j_end; ++j) {
                        fp8e4m3_8 a8 = a_v8[j];
                        fp8e4m3_8 b8 = b_v8[j];
                        float4 a_lo = fp8x4_to_f32x4(a8.lo);
                        float4 a_hi = fp8x4_to_f32x4(a8.hi);
                        float4 b_lo = fp8x4_to_f32x4(b8.lo);
                        float4 b_hi = fp8x4_to_f32x4(b8.hi);
                        acc_s0 += a_lo.x * b_lo.x + a_hi.x * b_hi.x;
                        acc_s1 += a_lo.y * b_lo.y + a_hi.y * b_hi.y;
                        acc_s2 += a_lo.z * b_lo.z + a_hi.z * b_hi.z;
                        acc_s3 += a_lo.w * b_lo.w + a_hi.w * b_hi.w;
                    }
                    kk = j_end << 3;
                }
                float acc_s = (acc_s0 + acc_s1) + (acc_s2 + acc_s3);
                for (; kk < g.k; ++kk) {
                    acc_s += load_fp8_scalar(g.a, row, kk) *
                             load_fp8_scalar_grp(g.b, row_group, col, kk);
                }
                const float scaled_s = acc_s * scale_s;
                store_bf16_scalar(g.c, row, col,
                                  load_bf16_scalar(g.c, row, col) + scaled_s);
            }
        }
        return;
    }

    // ----- Fast MFMA path: 2 M sub-blocks × 2 N sub-tiles = 4 mfmas/thread.
    // The A-pack is loaded ONCE per M sub-block and reused across the 2 N
    // sub-tiles (vs M2 where the same A-pack drives just 1 mfma per
    // sub-block). Halves per-cell A-side HBM bandwidth on the K-tail.
    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;

    const int row_in_blk = tid % 32;
    const int chunk      = tid / 32;        // 0 or 1
    const int k_off      = k0 + chunk * 32;

    // Two B-packs, one per N sub-tile. Same K slice for both packs (A-pack
    // shared between them) → mfma decomposes the A·Bᵀ block into 2 separate
    // 32×32 outputs along N.
    intx8_t b_pack[N_SUB];
    #pragma unroll
    for (int nt = 0; nt < N_SUB; ++nt) {
        const int g_col = col_block_base + nt * TBN + row_in_blk;
        if (g_col < g.n) {
            const fp8e4m3* b_ptr = &g.b[coord<>{0, group_idx, g_col, k_off}];
            b_pack[nt] = *reinterpret_cast<const intx8_t*>(b_ptr);
        } else {
            b_pack[nt] = intx8_t{};
        }
    }

    const float scale = resolve_combined_scale_grp(g);

    #pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
        const int sub_row_base = row_block_base + sub * TBM_SUB;
        const int g_row = sub_row_base + row_in_blk;

        intx8_t a_pack;
        if (g_row < g.M_total) {
            const fp8e4m3* a_ptr = &g.a[coord<>(g_row, k_off)];
            a_pack = *reinterpret_cast<const intx8_t*>(a_ptr);
        } else {
            a_pack = intx8_t{};
        }

        #pragma unroll
        for (int nt = 0; nt < N_SUB; ++nt) {
            floatx16_t acc{};
            acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a_pack, b_pack[nt], acc, /*cbsz=*/0, /*abid=*/0, /*blgp=*/0,
                /*scale_op_a=*/0, /*scale_op_b=*/0, /*scale_op_d=*/0);

            const int out_col = col_block_base + nt * TBN + row_in_blk;
            if (out_col >= g.n) continue;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int row_group     = i >> 2;
                const int row_in_group  = (i & 3) + chunk * 4;
                const int local_row     = row_group * 8 + row_in_group;
                const int r             = sub_row_base + local_row;
                if (r >= g.M_total) continue;
                const float existing = load_bf16_scalar(g.c, r, out_col);
                const float new_val  = existing + acc[i] * scale;
                store_bf16_scalar(g.c, r, out_col, new_val);
            }
        }
    }
}

template __global__ void grouped_ktail_kernel_mfma32x32_M2N2<Layout::RCR, 64>(const grouped_layout_globals);

#if 0  // round-61 disabled: regressed metric 752 → 749 (see comment above)
template<Layout L, int K_REM>
__global__ void grouped_ktail_kernel_mfma32x32_M2N4(const grouped_layout_globals g) {
    static_assert(L == Layout::RCR,
        "grouped_ktail_kernel_mfma32x32_M2N4 (FP8): RCR only.");
    static_assert(K_REM == 64,
        "grouped_ktail_kernel_mfma32x32_M2N4 (FP8): K_REM must be 64 (native mfma K).");
    constexpr int TBM_TOTAL = 64;       // 2 stacked 32×32 sub-blocks
    constexpr int TBM_SUB   = 32;
    constexpr int TBN       = 32;
    constexpr int N_SUB     = 4;
    constexpr int TBN_TOTAL = TBN * N_SUB;  // 128 cols
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];

    const int tid = threadIdx.x;            // single-wave block, 64 threads
    if (tid <= g.G && tid < MAX_G_PLUS_1) {
        s_offs[tid] = static_cast<int>(g.group_offs[tid]);
    }
    __syncthreads();

    const int row_block_base = blockIdx.y * TBM_TOTAL;
    const int col_block_base = blockIdx.x * TBN_TOTAL;
    if (row_block_base >= g.M_total || col_block_base >= g.n) return;

    int group_idx = 0;
    #pragma unroll 1
    for (int gi = 0; gi < g.G; ++gi) {
        if (row_block_base < s_offs[gi + 1]) { group_idx = gi; break; }
    }

    const int K_rem_dyn = g.k - g.fast_k;
    if (K_rem_dyn != K_REM) return;
    const int k0 = g.fast_k;

    // Cross-group fallback: same as M2N2 (64-row block straddling a group
    // boundary). Iterate over ALL FOUR 32-col sub-tiles per row using
    // scalar fma. Unreachable on uniform group_lens with M_g % 64 == 0.
    const bool cross_boundary = (row_block_base + TBM_TOTAL > s_offs[group_idx + 1]);
    if (cross_boundary) {
        typedef __attribute__((__vector_size__(2 * sizeof(float)))) float fp32x2_v;
        auto fp8x4_to_f32x4 = [](const fp8e4m3_4& u) -> float4 {
            int packed;
            __builtin_memcpy(&packed, &u, 4);
            fp32x2_v lo = __builtin_amdgcn_cvt_pk_f32_fp8(packed, false);
            fp32x2_v hi = __builtin_amdgcn_cvt_pk_f32_fp8(packed, true);
            return make_float4(lo[0], lo[1], hi[0], hi[1]);
        };
        const float scale_s = resolve_combined_scale_grp(g);
        for (int nt = 0; nt < N_SUB; ++nt) {
            const int col = col_block_base + nt * TBN + (tid % 32);
            if (col >= g.n) continue;
            #pragma unroll 1
            for (int rr = 0; rr < TBM_TOTAL; ++rr) {
                if ((rr % 2) != ((tid / 32))) continue;
                const int row = row_block_base + rr;
                if (row >= g.M_total) break;
                int row_group = 0;
                #pragma unroll 1
                for (int gi = 0; gi < g.G; ++gi) {
                    if (row < s_offs[gi + 1]) { row_group = gi; break; }
                }
                float acc_s0 = 0.f, acc_s1 = 0.f, acc_s2 = 0.f, acc_s3 = 0.f;
                const fp8e4m3* a_row = &g.a[coord<>(row, 0)];
                const fp8e4m3* b_row = &g.b[coord<>{0, row_group, col, 0}];
                int kk = k0;
                if ((g.k % 8 == 0) && ((k0 & 7) == 0)) {
                    const fp8e4m3_8* a_v8 = reinterpret_cast<const fp8e4m3_8*>(a_row);
                    const fp8e4m3_8* b_v8 = reinterpret_cast<const fp8e4m3_8*>(b_row);
                    const int j_start = k0 >> 3;
                    const int j_end   = g.k >> 3;
                    for (int j = j_start; j < j_end; ++j) {
                        fp8e4m3_8 a8 = a_v8[j];
                        fp8e4m3_8 b8 = b_v8[j];
                        float4 a_lo = fp8x4_to_f32x4(a8.lo);
                        float4 a_hi = fp8x4_to_f32x4(a8.hi);
                        float4 b_lo = fp8x4_to_f32x4(b8.lo);
                        float4 b_hi = fp8x4_to_f32x4(b8.hi);
                        acc_s0 += a_lo.x * b_lo.x + a_hi.x * b_hi.x;
                        acc_s1 += a_lo.y * b_lo.y + a_hi.y * b_hi.y;
                        acc_s2 += a_lo.z * b_lo.z + a_hi.z * b_hi.z;
                        acc_s3 += a_lo.w * b_lo.w + a_hi.w * b_hi.w;
                    }
                    kk = j_end << 3;
                }
                float acc_s = (acc_s0 + acc_s1) + (acc_s2 + acc_s3);
                for (; kk < g.k; ++kk) {
                    acc_s += load_fp8_scalar(g.a, row, kk) *
                             load_fp8_scalar_grp(g.b, row_group, col, kk);
                }
                const float scaled_s = acc_s * scale_s;
                store_bf16_scalar(g.c, row, col,
                                  load_bf16_scalar(g.c, row, col) + scaled_s);
            }
        }
        return;
    }

    // ----- Fast MFMA path: 2 M sub-blocks × 4 N sub-tiles = 8 mfmas/thread.
    // Single A-pack per M sub-block reused across all 4 N sub-tiles.
    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;

    const int row_in_blk = tid % 32;
    const int chunk      = tid / 32;        // 0 or 1
    const int k_off      = k0 + chunk * 32;

    intx8_t b_pack[N_SUB];
    #pragma unroll
    for (int nt = 0; nt < N_SUB; ++nt) {
        const int g_col = col_block_base + nt * TBN + row_in_blk;
        if (g_col < g.n) {
            const fp8e4m3* b_ptr = &g.b[coord<>{0, group_idx, g_col, k_off}];
            b_pack[nt] = *reinterpret_cast<const intx8_t*>(b_ptr);
        } else {
            b_pack[nt] = intx8_t{};
        }
    }

    const float scale = resolve_combined_scale_grp(g);

    #pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
        const int sub_row_base = row_block_base + sub * TBM_SUB;
        const int g_row = sub_row_base + row_in_blk;

        intx8_t a_pack;
        if (g_row < g.M_total) {
            const fp8e4m3* a_ptr = &g.a[coord<>(g_row, k_off)];
            a_pack = *reinterpret_cast<const intx8_t*>(a_ptr);
        } else {
            a_pack = intx8_t{};
        }

        #pragma unroll
        for (int nt = 0; nt < N_SUB; ++nt) {
            floatx16_t acc{};
            acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a_pack, b_pack[nt], acc, /*cbsz=*/0, /*abid=*/0, /*blgp=*/0,
                /*scale_op_a=*/0, /*scale_op_b=*/0, /*scale_op_d=*/0);

            const int out_col = col_block_base + nt * TBN + row_in_blk;
            if (out_col >= g.n) continue;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int row_group     = i >> 2;
                const int row_in_group  = (i & 3) + chunk * 4;
                const int local_row     = row_group * 8 + row_in_group;
                const int r             = sub_row_base + local_row;
                if (r >= g.M_total) continue;
                const float existing = load_bf16_scalar(g.c, r, out_col);
                const float new_val  = existing + acc[i] * scale;
                store_bf16_scalar(g.c, r, out_col, new_val);
            }
        }
    }
}

template __global__ void grouped_ktail_kernel_mfma32x32_M2N4<Layout::RCR, 64>(const grouped_layout_globals);
#endif  // M2N4 disabled

#if 0  // M4 kernel definition kept disabled — see comment above
template<Layout L, int K_REM>
__global__ void grouped_ktail_kernel_mfma32x32_M4(const grouped_layout_globals g) {
    static_assert(L == Layout::RCR,
        "grouped_ktail_kernel_mfma32x32_M4 (FP8): RCR only.");
    static_assert(K_REM == 64,
        "grouped_ktail_kernel_mfma32x32_M4 (FP8): K_REM must be 64 (native mfma K).");
    constexpr int TBM_TOTAL = 128;      // 4 stacked 32×32 sub-blocks
    constexpr int TBM_SUB   = 32;
    constexpr int N_SUB     = 4;
    constexpr int TBN       = 32;
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];

    const int tid = threadIdx.x;            // single-wave block, 64 threads
    if (tid <= g.G && tid < MAX_G_PLUS_1) {
        s_offs[tid] = static_cast<int>(g.group_offs[tid]);
    }
    __syncthreads();

    const int row_block_base = blockIdx.y * TBM_TOTAL;
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

    // Cross-group fallback: if the 128-row block straddles a group boundary
    // (or the M_total tail), fall back to per-row scalar fma over ALL FOUR
    // sub-blocks. Unreachable on uniform group_lens with M_g % 128 == 0
    // (gpt_oss M_per ∈ {2048, 4096} both satisfy).
    const bool cross_boundary = (row_block_base + TBM_TOTAL > s_offs[group_idx + 1]);
    if (cross_boundary) {
        typedef __attribute__((__vector_size__(2 * sizeof(float)))) float fp32x2_v;
        auto fp8x4_to_f32x4 = [](const fp8e4m3_4& u) -> float4 {
            int packed;
            __builtin_memcpy(&packed, &u, 4);
            fp32x2_v lo = __builtin_amdgcn_cvt_pk_f32_fp8(packed, false);
            fp32x2_v hi = __builtin_amdgcn_cvt_pk_f32_fp8(packed, true);
            return make_float4(lo[0], lo[1], hi[0], hi[1]);
        };
        const int col = col_block_base + (tid % 32);
        if (col < g.n) {
            #pragma unroll 1
            for (int rr = 0; rr < TBM_TOTAL; ++rr) {
                if ((rr % 2) != ((tid / 32))) continue;
                const int row = row_block_base + rr;
                if (row >= g.M_total) break;
                int row_group = 0;
                #pragma unroll 1
                for (int gi = 0; gi < g.G; ++gi) {
                    if (row < s_offs[gi + 1]) { row_group = gi; break; }
                }
                float acc_s0 = 0.f, acc_s1 = 0.f, acc_s2 = 0.f, acc_s3 = 0.f;
                const fp8e4m3* a_row = &g.a[coord<>(row, 0)];
                const fp8e4m3* b_row = &g.b[coord<>{0, row_group, col, 0}];
                int kk = k0;
                if ((g.k % 8 == 0) && ((k0 & 7) == 0)) {
                    const fp8e4m3_8* a_v8 = reinterpret_cast<const fp8e4m3_8*>(a_row);
                    const fp8e4m3_8* b_v8 = reinterpret_cast<const fp8e4m3_8*>(b_row);
                    const int j_start = k0 >> 3;
                    const int j_end   = g.k >> 3;
                    for (int j = j_start; j < j_end; ++j) {
                        fp8e4m3_8 a8 = a_v8[j];
                        fp8e4m3_8 b8 = b_v8[j];
                        float4 a_lo = fp8x4_to_f32x4(a8.lo);
                        float4 a_hi = fp8x4_to_f32x4(a8.hi);
                        float4 b_lo = fp8x4_to_f32x4(b8.lo);
                        float4 b_hi = fp8x4_to_f32x4(b8.hi);
                        acc_s0 += a_lo.x * b_lo.x + a_hi.x * b_hi.x;
                        acc_s1 += a_lo.y * b_lo.y + a_hi.y * b_hi.y;
                        acc_s2 += a_lo.z * b_lo.z + a_hi.z * b_hi.z;
                        acc_s3 += a_lo.w * b_lo.w + a_hi.w * b_hi.w;
                    }
                    kk = j_end << 3;
                }
                float acc_s = (acc_s0 + acc_s1) + (acc_s2 + acc_s3);
                for (; kk < g.k; ++kk) {
                    acc_s += load_fp8_scalar(g.a, row, kk) *
                             load_fp8_scalar_grp(g.b, row_group, col, kk);
                }
                const float scaled_s = acc_s * resolve_combined_scale_grp(g);
                store_bf16_scalar(g.c, row, col,
                                  load_bf16_scalar(g.c, row, col) + scaled_s);
            }
        }
        return;
    }

    // ----- Fast MFMA path: shared B-pack across FOUR stacked 32×32 sub-blocks
    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;

    const int row_in_blk = tid % 32;
    const int chunk      = tid / 32;        // 0 or 1
    const int k_off      = k0 + chunk * 32;

    const int g_col = col_block_base + row_in_blk;

    // Single B load shared across all four sub-blocks. B is [G, N, K] row-
    // major and the K-tail slice [k0, k0+K_REM) is the same for sub-blocks
    // 0..3 (only the A row range differs). 4×reuse of the B-side HBM
    // bandwidth vs running 4 independent 32×32 M1 blocks; 2× reuse vs M2.
    intx8_t b_pack;
    if (g_col < g.n) {
        const fp8e4m3* b_ptr = &g.b[coord<>{0, group_idx, g_col, k_off}];
        b_pack = *reinterpret_cast<const intx8_t*>(b_ptr);
    } else {
        b_pack = intx8_t{};
    }

    const float scale = resolve_combined_scale_grp(g);
    const int out_col = col_block_base + row_in_blk;
    if (out_col >= g.n) return;

    // Four stacked 32×32 sub-blocks. Same MFMA / lane layout as round-20
    // M1 / round-53 M2; we shift the row base by +sub*TBM_SUB and re-issue
    // mfma_scale_f32_32x32x64_f8f6f4 with a fresh accumulator each sub.
    #pragma unroll
    for (int sub = 0; sub < N_SUB; ++sub) {
        const int sub_row_base = row_block_base + sub * TBM_SUB;
        const int g_row = sub_row_base + row_in_blk;

        intx8_t a_pack;
        if (g_row < g.M_total) {
            const fp8e4m3* a_ptr = &g.a[coord<>(g_row, k_off)];
            a_pack = *reinterpret_cast<const intx8_t*>(a_ptr);
        } else {
            a_pack = intx8_t{};
        }

        floatx16_t acc{};
        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_pack, b_pack, acc, /*cbsz=*/0, /*abid=*/0, /*blgp=*/0,
            /*scale_op_a=*/0, /*scale_op_b=*/0, /*scale_op_d=*/0);

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int row_group     = i >> 2;
            const int row_in_group  = (i & 3) + chunk * 4;
            const int local_row     = row_group * 8 + row_in_group;
            const int r             = sub_row_base + local_row;
            if (r >= g.M_total) continue;
            const float existing = load_bf16_scalar(g.c, r, out_col);
            const float new_val  = existing + acc[i] * scale;
            store_bf16_scalar(g.c, r, out_col, new_val);
        }
    }
}

template __global__ void grouped_ktail_kernel_mfma32x32_M4<Layout::RCR, 64>(const grouped_layout_globals);
#endif  // M4 kernel disabled

void dispatch_grouped_rcr(grouped_layout_globals g) {
    g.n = static_cast<int>(g.c.cols());
    g.M_total = static_cast<int>(g.c.rows());
    g.k = static_cast<int>(g.a.cols());

    g.fast_n = (g.n / BLOCK_SIZE) * BLOCK_SIZE;
    g.fast_k = (g.k / K_BLOCK)    * K_BLOCK;
    g.bpc    = kittens::ceil_div(g.n, BLOCK_SIZE);
    g.ki     = g.fast_k / K_BLOCK;

    const int K_rem_for_fuse = g.k - g.fast_k;
    const bool lds_k_tail_safe_for_fuse =
        (g.m_per_group >= TAIL_BLOCK_M) &&
        ((g.m_per_group % TAIL_BLOCK_M) == 0);
    const bool fuse_ktail_eligible =
        (g.bpc > 0) && (g.ki > 0) &&
        ((K_rem_for_fuse == 64) || (K_rem_for_fuse == 0)) &&
        lds_k_tail_safe_for_fuse;

    static const bool fuse_force_off = []() {
        if (const char* e = std::getenv("TK_GROUPED_RCR_FUSE_OFF")) {
            return std::atoi(e) >= 1;
        }
        return false;
    }();
    const bool fuse_ktail_active =
        fuse_ktail_eligible && !fuse_force_off && (g.fuse_ktail_off == 0);

    int* sk_partial_buf_owned = nullptr;
    if (g.sk_split_n > 0 && g.bpc > 0 && g.ki > 0 && g.sk_partial_buf == nullptr) {
        const int T_max = kittens::ceil_div(g.M_total, BLOCK_SIZE) * g.bpc;
        const size_t buf_bytes =
            static_cast<size_t>(T_max) * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
        hipMallocAsync(reinterpret_cast<void**>(&sk_partial_buf_owned),
                       buf_bytes, g.stream);
        hipMemsetAsync(sk_partial_buf_owned, 0, buf_bytes, g.stream);
        g.sk_partial_buf = sk_partial_buf_owned;
    }

    if (g.bpc > 0 && g.ki > 0) {
        const bool n_aligned = (g.bpc * BLOCK_SIZE == g.n);

        static const int rcr_slots_env = []() {
            if (const char* e = std::getenv("TK_RCR_NUM_CUS")) {
                const int v = std::atoi(e);
                if (v > 0 && v <= NUM_CUS) return v;
            }
            return NUM_CUS;
        }();
        const int rcr_slots = (g.num_slots > 0 && g.num_slots <= NUM_CUS)
            ? g.num_slots : rcr_slots_env;

        if (g.chunk_size <= 0 || g.chunk_size > NUM_CUS) {
            static const int env_chunk_size = []() {
                if (const char* e = std::getenv("TK_RCR_CHUNK_SIZE")) {
                    const int v = std::atoi(e);
                    if (v >= 1 && v <= 256) return v;
                }
                return 0;  // 0 → kernel uses default 64
            }();
            g.chunk_size = env_chunk_size;
        }

        if (fuse_ktail_active) {
            if (n_aligned) {
                grouped_rcr_kernel<0, false, true><<<dim3(rcr_slots), g.block(), 0, g.stream>>>(g);
            } else {
                grouped_rcr_kernel<0, true , true><<<dim3(rcr_slots), g.block(), 0, g.stream>>>(g);
            }
        } else {
            if (n_aligned) {
                grouped_rcr_kernel<0, false, false><<<dim3(rcr_slots), g.block(), 0, g.stream>>>(g);
            } else {
                grouped_rcr_kernel<0, true , false><<<dim3(rcr_slots), g.block(), 0, g.stream>>>(g);
            }
        }
    } else {
        // No aligned interior at all: main kernel cannot run; tail handles
        // every cell with a full-K reduction.
        g.fast_n = 0;
        g.fast_k = 0;
        g.bpc = 0;
        g.ki = 0;
    }

    if (!fuse_ktail_active && (g.fast_k != g.k || g.bpc == 0)) {
        const int K_rem = g.k - g.fast_k;
        constexpr int TBM_32x32 = 32;
        constexpr int TBM_M2    = 64;       // round-53: 2 stacked 32×32 sub-blocks
        constexpr int TBN_M2N2  = 64;       // round-60: 2 stacked × 2 N sub-tiles
        const bool mfma32_safe = (g.m_per_group >= TBM_32x32) &&
                                 ((g.m_per_group % TBM_32x32) == 0);
        const bool mfma32_m2_safe = (g.m_per_group >= TBM_M2) &&
                                    ((g.m_per_group % TBM_M2) == 0);
        const bool mfma32_m2n2_safe = mfma32_m2_safe &&
                                      ((g.n % TBN_M2N2) == 0);
        const bool lds_k_tail_safe = (g.m_per_group >= TAIL_BLOCK_M) &&
                                     ((g.m_per_group % TAIL_BLOCK_M) == 0);
        const bool mfma32_m2n2_handles_all =
            (g.bpc > 0) &&
            (K_rem == 64) &&
            mfma32_m2n2_safe;
        const bool mfma32_m2_handles_all =
            (g.bpc > 0) &&
            (K_rem == 64) &&
            mfma32_m2_safe;
        const bool mfma32_handles_all =
            (g.bpc > 0) &&
            (K_rem == 64) &&
            mfma32_safe;
        const bool mfma16_handles_all =
            (g.bpc > 0) &&
            (K_rem == 64) &&
            lds_k_tail_safe;
        if (mfma32_m2n2_handles_all) {
            dim3 mfma_block(64);
            dim3 mfma_grid(
                kittens::ceil_div(g.n, TBN_M2N2),
                kittens::ceil_div(g.M_total, TBM_M2)
            );
            grouped_ktail_kernel_mfma32x32_M2N2<Layout::RCR, 64>
                <<<mfma_grid, mfma_block, 0, g.stream>>>(g);
        } else if (mfma32_m2_handles_all) {
            dim3 mfma_block(64);
            dim3 mfma_grid(
                kittens::ceil_div(g.n, TBM_32x32),
                kittens::ceil_div(g.M_total, TBM_M2)
            );
            grouped_ktail_kernel_mfma32x32_M2<Layout::RCR, 64>
                <<<mfma_grid, mfma_block, 0, g.stream>>>(g);
        } else if (mfma32_handles_all) {
            dim3 mfma_block(64);
            dim3 mfma_grid(
                kittens::ceil_div(g.n, TBM_32x32),
                kittens::ceil_div(g.M_total, TBM_32x32)
            );
            grouped_ktail_kernel_mfma32x32<Layout::RCR, 64>
                <<<mfma_grid, mfma_block, 0, g.stream>>>(g);
        } else if (mfma16_handles_all) {
            // Round-18: 16x16x128 mfma kernel (50 % util via zero-pad).
            dim3 mfma_block(64);
            dim3 mfma_grid(
                kittens::ceil_div(g.n, TAIL_BLOCK_N),
                kittens::ceil_div(g.M_total, TAIL_BLOCK_M)
            );
            grouped_ktail_kernel_mfma<Layout::RCR, 64>
                <<<mfma_grid, mfma_block, 0, g.stream>>>(g);
        } else {
            dim3 tail_block(TAIL_BLOCK_N, TAIL_BLOCK_M);
            dim3 tail_grid(
                kittens::ceil_div(g.n, TAIL_BLOCK_N),
                kittens::ceil_div(g.M_total, TAIL_BLOCK_M)
            );
            grouped_tail_kernel<Layout::RCR>
                <<<tail_grid, tail_block, 0, g.stream>>>(g);
        }
    }

    if (sk_partial_buf_owned != nullptr) {
        hipFreeAsync(sk_partial_buf_owned, g.stream);
    }
}

void dispatch_grouped_rcr_fused_act(grouped_layout_globals_fused_act g) {
    g.n = static_cast<int>(g.c.cols());
    g.M_total = static_cast<int>(g.c.rows());
    g.k = static_cast<int>(g.a.cols());

    g.fast_n = (g.n / BLOCK_SIZE) * BLOCK_SIZE;
    g.fast_k = (g.k / K_BLOCK)    * K_BLOCK;
    g.bpc    = kittens::ceil_div(g.n, BLOCK_SIZE);
    g.ki     = g.fast_k / K_BLOCK;

    if (!(g.bpc > 0 && g.ki > 0 && g.fast_k == g.k)) {
        // Caller is responsible for falling back to the un-fused path.
        return;
    }

    const bool n_aligned = (g.bpc * BLOCK_SIZE == g.n);
    if (n_aligned) {
        grouped_rcr_kernel<0, /*N_MASKED_STORE=*/false,
                           /*FUSED_KTAIL=*/false, /*FUSE_ACT=*/true>
            <<<dim3(NUM_CUS), g.block(), 0, g.stream>>>(g);
    } else {
        grouped_rcr_kernel<0, /*N_MASKED_STORE=*/true,
                           /*FUSED_KTAIL=*/false, /*FUSE_ACT=*/true>
            <<<dim3(NUM_CUS), g.block(), 0, g.stream>>>(g);
    }
}

// =============================================================================
// Persistent grouped RRR dispatcher — FP8 (forward-A backward dA path).
//
// Mirror of ``dispatch_grouped_rcr``: aligned interior swept by the
// persistent main kernel (``grouped_rrr_kernel``), cells outside go
// through ``grouped_tail_kernel<Layout::RRR>`` (scalar fp32 with the
// same N-tail / K-tail correction logic as RCR — see template body).
//
// Per-group M_g must still be a BLOCK_SIZE multiple (the persistent
// loop derives ``bpr_g = M_g / BLOCK_SIZE`` and steps in HB units);
// other invariants are identical to the RCR path.
// =============================================================================
void dispatch_grouped_rrr(grouped_layout_globals g) {
    g.n = static_cast<int>(g.c.cols());
    g.M_total = static_cast<int>(g.c.rows());
    g.k = static_cast<int>(g.a.cols());

    g.fast_n = (g.n / BLOCK_SIZE) * BLOCK_SIZE;
    g.fast_k = (g.k / K_BLOCK)    * K_BLOCK;
    // Mirror RCR: bpc rounds UP so the main kernel sweeps the (possibly
    // partial) last N tile and masks its OOB cells via N_MASKED_STORE.
    // OOB B reads inside the loop return 0 from raw_buffer_load_lds.
    g.bpc    = kittens::ceil_div(g.n, BLOCK_SIZE);
    g.ki     = g.fast_k / K_BLOCK;

    if (g.M_total <= 0 || g.n <= 0 || g.k <= 0 || g.G <= 0) return;

    const bool n_aligned = (g.bpc * BLOCK_SIZE == g.n);
    const int K_rem_for_fuse = g.k - g.fast_k;
    const bool fuse_ktail_active =
        (g.bpc > 0) && (g.ki > 0) && (K_rem_for_fuse == 64) &&
        (g.m_per_group >= HB) && ((g.m_per_group % HB) == 0);
    if (g.bpc > 0 && g.ki > 0) {
        if (fuse_ktail_active) {
            if (n_aligned) {
                grouped_rrr_kernel<0, false, true>
                    <<<dim3(NUM_CUS), g.block(), 0, g.stream>>>(g);
            } else {
                grouped_rrr_kernel<0, true, true>
                    <<<dim3(NUM_CUS), g.block(), 0, g.stream>>>(g);
            }
        } else if (n_aligned) {
            grouped_rrr_kernel<0, false, false>
                <<<dim3(NUM_CUS), g.block(), 0, g.stream>>>(g);
        } else {
            grouped_rrr_kernel<0, true, false>
                <<<dim3(NUM_CUS), g.block(), 0, g.stream>>>(g);
        }
    } else {
        // No aligned interior at all — main kernel cannot run; tail handles
        // every cell with a full-K reduction.
        g.fast_n = 0;
        g.fast_k = 0;
        g.bpc = 0;
        g.ki = 0;
    }

    if (g.fast_k != g.k && !fuse_ktail_active) {
        // K-tail correction (RMW). When FUSED_KTAIL handled it inside the
        // main kernel, skip these launches (they'd double-add).
        const bool lds_k_tail_safe = (g.m_per_group >= TAIL_BLOCK_M) &&
                                     ((g.m_per_group % TAIL_BLOCK_M) == 0);
        const int K_rem = g.k - g.fast_k;
        if (K_rem == 64 && lds_k_tail_safe) {
            // LDS K-tail kernel covers FULL N (interior + N-tail). The
            // kernel's per-cell guards (col >= g.n) and zero-pad cooperative
            // load handle the N-tail strip [fast_n, n) safely.
            dim3 lds_block(TAIL_BLOCK_N, TAIL_BLOCK_M);
            dim3 lds_grid(
                kittens::ceil_div(g.n, TAIL_BLOCK_N),
                kittens::ceil_div(g.M_total, TAIL_BLOCK_M)
            );
            grouped_ktail_kernel_lds_rrr<64>
                <<<lds_grid, lds_block, 0, g.stream>>>(g);
        }

        // Scalar K-tail RMW for cells the LDS kernel didn't cover
        // (cross-group blocks, m_per_group misalign, K-tail != 64).
        dim3 tail_block(TAIL_BLOCK_N, TAIL_BLOCK_M);
        dim3 tail_grid(
            kittens::ceil_div(g.n, TAIL_BLOCK_N),
            kittens::ceil_div(g.M_total, TAIL_BLOCK_M)
        );
        grouped_tail_kernel<Layout::RRR>
            <<<tail_grid, tail_block, 0, g.stream>>>(g);
    }
}

// =============================================================================
// Persistent grouped variable-K (CRR / dB) GEMM (CPU-sync-free) — FP8.
//
// Mirror of the BF16 ``grouped_variable_k_crr`` kernel in
// analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp (rounds 1-2). Drives
// the FP8 backward dB computation: given ``a`` = grad_out [M_total, N_fwd]
// fp8 and ``b`` = x [M_total, K_fwd] fp8, produce
// ``c`` = grad_b [G, N_fwd, K_fwd] bf16 in a single CPU-sync-free launch.
// Per-group K-reduction over M_g rows is realised by adding a per-group
// ``k_offset_tiles = m_start_g / HB`` to the kernel's K-axis tile coords.
//
// Native non-aligned (n, k): bpr = ceil_div(n, BLK), bpc = ceil_div(k, BLK).
// Partial last tiles use ``store_c_tile_mn_masked_grouped`` to drop OOB
// (m, n) cells. Safe because (a) A and B are 2D contiguous tensors, NOT
// 3D-grouped — the SRD-wrap-into-next-group issue blocking forward
// grouped does not apply; and (b) MMA cells at OOB output positions are
// dropped before any global store.
//
// Per-group M_g >= 2*HB = 256 is required (prologue + 2 epilogues each
// consume HB-many K-rows; ki_g = M_g / HB and we need ki_g >= 2). The
// Primus-side uniform-M >= 256 gate enforces this; the kernel ``ki_g <
// 2`` skip is a defensive fallback.
// =============================================================================
struct grouped_var_k_layout_globals_fp8 {
    _gl_fp8 a;                     // [1, 1, M_total, n] — grad_out fp8
    _gl_fp8 b;                     // [1, 1, M_total, k] — x fp8
    _gl_bf16 c;                    // [1, G, n, k]       — grad_b bf16
    float scale_a, scale_b;        // host-side scales (used when no dscale)
    const float* dscale_a;
    const float* dscale_b;
    const int64_t* group_offs;     // [G+1] int64 device prefix-sum
    hipStream_t stream;
    int G;                         // number of groups
    int M_total;                   // sum M_g across groups
    int n;                         // kernel M-output dim (= N_fwd)
    int k;                         // kernel N-output dim (= K_fwd)
    int group_m;
    int bpr;                       // ceil_div(n, BLOCK_SIZE)
    int bpc;                       // ceil_div(k, BLOCK_SIZE)
    int fast_n, fast_k;
    int num_xcds;                  // chiplet-swizzle XCD count (0 → default 8)
    int num_slots;                 // Round-3 (gpt_oss FP8 kernel-only ceiling,
    int chunk_size;                // Round-13 (gpt_oss FP8 kernel-only ceiling,
    dim3 block() { return dim3(_NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

__device__ __forceinline__ float resolve_combined_scale_var_k_fp8(
    const grouped_var_k_layout_globals_fp8 &g) {
    const float sa = g.dscale_a ? *g.dscale_a : g.scale_a;
    const float sb = g.dscale_b ? *g.dscale_b : g.scale_b;
    return sa * sb;
}

template<int KI_HINT = 0>
__global__ __launch_bounds__(_NUM_THREADS, 1)
void grouped_var_k_kernel_fp8(const grouped_var_k_layout_globals_fp8 g) {
    using ST_crr_a = ST_v2a;
    using ST_crr_b = ST_v2;
    __shared__ ST_crr_a As[2][2];
    __shared__ ST_crr_b Bs[2][2];

    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];
    __shared__ int s_cum_tiles[MAX_G_PLUS_1];
    __shared__ int s_total_tiles;

    A_col_reg a;
#if VARK_SW_PIPE_HOIST_AHEAD
    // Round-24: extra A-tile register staging iter k+1's half-1 LDS A-read.
    A_col_reg a_next;
#endif
    B_col_reg b0, b1;
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;

    // Round-2 (FP8 backward unblock): mirror RCR line ~2020 — host-side
    // ``g.num_xcds`` knob with fallback to the default 8 when unset.
    const int xcds_eff = g.num_xcds > 0 ? g.num_xcds : BLOCK_SWIZZLE_NUM_XCDS;
    const int slots_eff = gridDim.x;
    const int chunk_size_eff = g.chunk_size > 0 ? g.chunk_size : 64;
    int pid = chiplet_transform_chunked(
        blockIdx.x, slots_eff, xcds_eff, chunk_size_eff);

    int wm = warpid() / WARPS_N;
    int wn = warpid() % WARPS_N;
    const int num_pid_n = g.bpc;

    const int tiles_per_group = g.bpr * g.bpc;
    if (threadIdx.x <= g.G && threadIdx.x < MAX_G_PLUS_1) {
        s_offs[threadIdx.x] = static_cast<int>(g.group_offs[threadIdx.x]);
        s_cum_tiles[threadIdx.x] =
            static_cast<int>(threadIdx.x) * tiles_per_group;
    }
    if (threadIdx.x > g.G && threadIdx.x < MAX_G_PLUS_1) {
        s_cum_tiles[threadIdx.x] = 0x7FFFFFFF;
    }
    if (threadIdx.x == 0) {
        s_total_tiles = g.G * tiles_per_group;
    }
    __syncthreads();
    const int total_tiles = s_total_tiles;

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

    for (int gt = pid; gt < total_tiles; gt += slots_eff) {
        const int group_idx = gt / tiles_per_group;
        const int tile_start = group_idx * tiles_per_group;
        const int local_tile = gt - tile_start;
        const int m_start_g = s_offs[group_idx];
        const int M_g = s_offs[group_idx + 1] - m_start_g;
        const int ki_g = M_g / HB;
        if (ki_g < 2) continue;
        const int bpr_g = g.bpr;

        int br, bc;
        if (g.bpc > bpr_g) {
            const int WGN = g.group_m;
            const int num_wgid_in_group = bpr_g * WGN;
            int group_id = local_tile / num_wgid_in_group;
            int first_pid_n = group_id * WGN;
            int group_size_n = min(num_pid_n - first_pid_n, WGN);
            if (group_size_n <= 0) continue;
            bc = first_pid_n + ((local_tile % num_wgid_in_group) % group_size_n);
            br = (local_tile % num_wgid_in_group) / group_size_n;
        } else {
            const int WGM = g.group_m;
            const int num_wgid_in_group = WGM * num_pid_n;
            int group_id = local_tile / num_wgid_in_group;
            int first_pid_m = group_id * WGM;
            int group_size_m = min(bpr_g - first_pid_m, WGM);
            if (group_size_m <= 0) continue;
            br = first_pid_m + ((local_tile % num_wgid_in_group) % group_size_m);
            bc = (local_tile % num_wgid_in_group) / group_size_m;
        }
        if (br >= bpr_g || bc >= num_pid_n) continue;

        // K-axis tile shift for variable-K: each group's K-reduction
        // starts at row m_start_g of the global flat A/B tensors.
        // ST_v2a / ST_v2 row dim = HB, so the tile-coord offset is
        // ``m_start_g / HB``. M_g must be a multiple of HB (= 128) for
        // the per-group ki_g to be exact (Primus uniform-M >= 256 gate
        // covers the bench cases; M_g = 2048 / 4096 / 8192 are all
        // 128-multiples).
        const int k_offset_tiles = m_start_g / HB;

        auto a_co = [&](int s, int k) -> coord<ST_crr_a> {
            return {0, 0, k_offset_tiles + k, s};
        };
        auto b_co = [&](int s, int k) -> coord<ST_crr_b> {
            return {0, 0, k_offset_tiles + k, s};
        };
        auto load_a = [&](A_col_reg& dst, ST_crr_a& tile, int wi) {
            load_col_from_st(dst, tile, wi * RBM);
        };
        auto load_b = [&](B_col_reg& dst, ST_crr_b& tile, int wi) {
            load_col_from_st(dst, tile, wi * RBN);
        };
        auto global_load_a = [&](ST_crr_a& tile, int s, int k) {
            rcr_8w_load_hoist<_NUM_THREADS>(tile, g.a, a_co(s, k), soA);
        };
        auto global_load_b = [&](ST_crr_b& tile, int s, int k) {
            rcr_8w_load_hoist<_NUM_THREADS>(tile, g.b, b_co(s, k), soB);
        };

        zero(cA); zero(cB); zero(cC); zero(cD);

        int tic = 0, toc = 1;
        global_load_b(Bs[tic][0], bc*2,   0);
        global_load_a(As[tic][0], br*2,   0);
        global_load_b(Bs[tic][1], bc*2+1, 0);
        global_load_a(As[tic][1], br*2+1, 0);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(CRR_INIT0_VMCNT);
        __builtin_amdgcn_s_barrier();

        global_load_b(Bs[toc][0], bc*2,   1);
        global_load_a(As[toc][0], br*2,   1);
        global_load_b(Bs[toc][1], bc*2+1, 1);

        TK_WAIT_VMCNT(CRR_INIT1_VMCNT);
        __builtin_amdgcn_s_barrier();

#if VARK_SW_PIPE_HOIST_AHEAD
        load_a(a_next, As[tic][0], wm);
#endif

        TK_PRAGMA_UNROLL(VARK_MAIN_UNROLL)
        for (int k = 0; k < ki_g - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, Bs[tic][0], wn);
            load_b(b1, Bs[tic][1], wn);
#if VARK_SW_PIPE_HOIST_AHEAD
            a = a_next;
#else
            load_a(a, As[tic][0], wm);
#endif
            global_load_a(As[toc][1], br*2+1, k+1);
            TK_WAIT_LGKM(CRR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
#if !VARK_DROP_REDUNDANT_LGKM_DRAIN
            asm volatile("s_waitcnt lgkmcnt(0)");
#endif
#if VARK_HOIST_PREFETCH_INTO_HALF1
            __builtin_amdgcn_s_barrier();
            global_load_a(As[tic][0], br*2, k+2);
            global_load_b(Bs[tic][1], bc*2+1, k+2);
#endif
            CRR_MMA_BEGIN();
            crr_mma(cA, a, b0);
            crr_mma(cB, a, b1);
            CRR_MMA_END();
#if !VARK_DROP_BARRIER_2 && !VARK_HOIST_PREFETCH_INTO_HALF1
            CRR_STEADY_MID_BARRIER();
#endif

            load_a(a, As[tic][1], wm);
#if !VARK_HOIST_PREFETCH_INTO_HALF1
            global_load_a(As[tic][0], br*2, k+2);
            global_load_b(Bs[tic][1], bc*2+1, k+2);
#endif
            TK_WAIT_VMCNT(CRR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
#if !VARK_DROP_REDUNDANT_LGKM_DRAIN
            asm volatile("s_waitcnt lgkmcnt(0)");
#endif
#if VARK_SW_PIPE_HOIST_AHEAD
            load_a(a_next, As[toc][0], wm);
#endif
            CRR_MMA_BEGIN();
            crr_mma(cC, a, b0);
            crr_mma(cD, a, b1);
            CRR_MMA_END();
#if !VARK_DROP_BARRIER_4
            __builtin_amdgcn_s_barrier();
#endif
            global_load_b(Bs[tic][0], bc*2, k+2);
        }

        {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            global_load_a(As[toc][1], br*2+1, ki_g-1);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cA, a, b0);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cB, a, b1);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(CRR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cC, a, b0);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[toc][0], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cD, a, b1);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();
            tic ^= 1; toc ^= 1;
        }

        {
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cA, a, b0);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cB, a, b1);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cC, a, b0);
            crr_mma(cD, a, b1);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();
        }

        // Apply scale + 2-axis-masked store. Output is per-group [n, k]
        // sub-tensor at depth = group_idx; no m_subtile_C row shift.
        const float combined_scale = resolve_combined_scale_var_k_fp8(g);
        mul(cA, cA, combined_scale);
        mul(cB, cB, combined_scale);
        mul(cC, cC, combined_scale);
        mul(cD, cD, combined_scale);

        if (wm == 0) __builtin_amdgcn_s_barrier();
        store_c_tile_mn_masked_grouped(g.c, cA, group_idx,
            br*WARPS_M*2+wm,         bc*WARPS_N*2+wn,         g.n, g.k);
        store_c_tile_mn_masked_grouped(g.c, cB, group_idx,
            br*WARPS_M*2+wm,         bc*WARPS_N*2+WARPS_N+wn, g.n, g.k);
        store_c_tile_mn_masked_grouped(g.c, cC, group_idx,
            br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+wn,         g.n, g.k);
        store_c_tile_mn_masked_grouped(g.c, cD, group_idx,
            br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+WARPS_N+wn, g.n, g.k);

        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }
}

template __global__ void grouped_var_k_kernel_fp8<0>(
    const grouped_var_k_layout_globals_fp8);

void dispatch_grouped_var_k_fp8(grouped_var_k_layout_globals_fp8 g) {
    g.n = static_cast<int>(g.a.cols());
    g.k = static_cast<int>(g.b.cols());
    g.M_total = static_cast<int>(g.a.rows());

    g.fast_n = g.n;
    g.fast_k = g.k;
    g.bpr = kittens::ceil_div(g.n, BLOCK_SIZE);
    g.bpc = kittens::ceil_div(g.k, BLOCK_SIZE);

    if (g.bpr <= 0 || g.bpc <= 0 || g.G <= 0) return;

    int slots_dispatch;
    if (g.num_slots > 0 && g.num_slots <= NUM_CUS) {
        slots_dispatch = g.num_slots;
    } else {
        static const int env_slots = []() {
            if (const char* e = std::getenv("TK_VARK_NUM_CUS")) {
                const int v = std::atoi(e);
                if (v > 0 && v <= NUM_CUS) return v;
            }
            return NUM_CUS;
        }();
        slots_dispatch = env_slots;
    }

    if (g.chunk_size <= 0 || g.chunk_size > NUM_CUS) {
        static const int env_chunk_size = []() {
            if (const char* e = std::getenv("TK_VARK_CHUNK_SIZE")) {
                const int v = std::atoi(e);
                if (v >= 1 && v <= 256) return v;
            }
            return 0;  // 0 → kernel uses default 64
        }();
        g.chunk_size = env_chunk_size;
    }

    grouped_var_k_kernel_fp8<0><<<dim3(slots_dispatch), g.block(), 0, g.stream>>>(g);
}

template<Layout L>
void dispatch(layout_globals g) {
    g.m = static_cast<int>(g.c.rows());
    g.n = static_cast<int>(g.c.cols());
    if constexpr (L == Layout::RCR) {
        g.k = static_cast<int>(g.a.cols());
    } else if constexpr (L == Layout::RRR) {
        g.k = static_cast<int>(g.a.cols());
    } else {
        g.k = static_cast<int>(g.b.rows());
    }

    g.fast_m = (g.m / BLK) * BLK;
    g.fast_n = (g.n / BLK) * BLK;
    g.fast_k = (g.k / BK) * BK;
    g.bpr = g.fast_m / BLK;
    g.bpc = g.fast_n / BLK;
    g.ki = g.fast_k / BK;

    // FP8 dense Phase 4: enable native N-tail via main-kernel column-
    // masked C store + ``bpc = ceil_div(n, BLK)``, mirroring BF16 dense.
    // Safe because ``rcr_8w_load_hoist`` was refactored to use a FULL-
    // tensor SRD with per-tile SOFFSET (see helper near line 400) — OOB
    // column reads from a partial last tile clamp to 0 from the SRD
    // bound rather than faulting on unmapped memory.
    //
    // Conditions to enable ``ceil_div`` bpc (mirror BF16):
    //   1. The 8-wave generic ``gemm_kernel<L, 0>`` is selected. The
    //      4-wave RCR fast path uses ``rcr_4w::prepare_g2s`` which still
    //      builds tile-local SRDs and is NOT yet refactored — keep its
    //      ``bpc = fast_n/BLK`` (its kernel is selected only for very
    //      large grids; tail kernel handles N misalignment in those
    //      large-grid cases).
    //   2. K is fully aligned (``fast_k == k``). When K is misaligned
    //      the tail kernel needs a deterministic fast/tail boundary in
    //      N; same as BF16 dense.
    bool main_covers_n = false;
    if (g.bpr > 0 && g.bpc > 0 && g.ki >= 2) {
        if (g.fast_k == g.k) {
            g.bpc = kittens::ceil_div(g.n, BLK);
            main_covers_n = true;
        }
        gemm_kernel<L, 0><<<g.grid(), g.block(), 0, g.stream>>>(g);
    } else {
        g.fast_k = 0;
        g.ki = 0;
    }

    const bool need_tail = (g.fast_m != g.m) || (g.fast_k != g.k) ||
                           (!main_covers_n && g.fast_n != g.n) || (g.ki == 0);
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
static float to_float(pybind11::object obj) {
    if (pybind11::hasattr(obj, "item"))
        return obj.attr("item")().cast<float>();
    return obj.cast<float>();
}

constexpr int DEFAULT_GROUP_M = 4;

template<Layout L>
static void gemm_wrapper(pybind11::object a, pybind11::object b, pybind11::object c,
                          pybind11::object scale_a_obj, pybind11::object scale_b_obj,
                          int group_m) {
    layout_globals g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        to_float(scale_a_obj),
        to_float(scale_b_obj),
        {}, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        group_m,
        nullptr, nullptr,
    };
    dispatch<L>(g);
}

// Variant of gemm_wrapper that takes the per-tensor FP8 scales as 0-d device
// tensors (one element each) instead of host-side scalars / 0-d host tensors.
// The Python-side host wrapper used to call `(a_scale_inv * b_scale_inv).item()`
// on every dispatch which is a stream sync that costs ~18us on small dense
// FP8 shapes (≈ 30% of the kernel itself) — this entry skips that sync by
// passing the device pointers straight through to the kernel's epilogue, which
// reads one b32 from global memory at scale-application time. The two scales
// are both used in the epilogue only, so the cost of the load is fully hidden
// behind the GEMM main loop.
template<Layout L>
static void gemm_wrapper_dscale(pybind11::object a, pybind11::object b, pybind11::object c,
                                 pybind11::object scale_a_obj, pybind11::object scale_b_obj,
                                 int group_m) {
    auto sa_ptr = scale_a_obj.attr("data_ptr")().cast<uintptr_t>();
    auto sb_ptr = scale_b_obj.attr("data_ptr")().cast<uintptr_t>();
    layout_globals g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        0.f, 0.f,
        {}, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        group_m,
        reinterpret_cast<const float*>(sa_ptr),
        reinterpret_cast<const float*>(sb_ptr),
    };
    dispatch<L>(g);
}

static void grouped_rcr_fn(pybind11::object a, pybind11::object b, pybind11::object c,
                           pybind11::object scale_a_obj, pybind11::object scale_b_obj,
                           pybind11::object group_offs_obj,
                           int group_m,
                           int m_per_group,
                           int num_xcds,
                           int num_slots,
                           int chunk_size = 0,
                           int fuse_ktail_off = 0,
                           int sk_split_n = 0,
                           uint64_t sk_workspace_ptr = 0) {
    auto group_offs_ptr = group_offs_obj.attr("data_ptr")().cast<uintptr_t>();
    int G = group_offs_obj.attr("numel")().cast<int>() - 1;
    grouped_layout_globals g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        to_float(scale_a_obj),
        to_float(scale_b_obj),
        nullptr,
        nullptr,
        reinterpret_cast<const int64_t*>(group_offs_ptr),
        {},
        /* G,n,k,ki,bpc,group_m,num_xcds,M_total,fast_n,fast_k,m_per_group,num_slots,chunk_size,fuse_ktail_off,sk_split_n */
        G, 0, 0, 0, 0, group_m, num_xcds, 0, 0, 0, m_per_group, num_slots, chunk_size, fuse_ktail_off, sk_split_n,
        // sk_partial_buf left default-init (nullptr); R13a alloc fills it when sk_split_n > 0.
    };
    // R17: caller-allocated workspace override. If sk_workspace_ptr != 0, the
    // dispatcher's per-call hipMallocAsync branch is skipped (gated below on
    // g.sk_partial_buf == nullptr). Cast through void* to silence -Wcast-align.
    if (sk_workspace_ptr != 0) {
        g.sk_partial_buf = reinterpret_cast<int*>(static_cast<uintptr_t>(sk_workspace_ptr));
    }
    dispatch_grouped_rcr(g);
}

static void grouped_rcr_dscale_fn(
    pybind11::object a, pybind11::object b, pybind11::object c,
    pybind11::object scale_a_obj, pybind11::object scale_b_obj,
    pybind11::object group_offs_obj,
    int group_m,
    int m_per_group,
    int num_xcds,
    int num_slots,
    int chunk_size = 0,
    int fuse_ktail_off = 0,
    int sk_split_n = 0,
    uint64_t sk_workspace_ptr = 0) {
    auto sa_ptr = scale_a_obj.attr("data_ptr")().cast<uintptr_t>();
    auto sb_ptr = scale_b_obj.attr("data_ptr")().cast<uintptr_t>();
    auto group_offs_ptr = group_offs_obj.attr("data_ptr")().cast<uintptr_t>();
    int G = group_offs_obj.attr("numel")().cast<int>() - 1;
    grouped_layout_globals g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        0.f, 0.f,
        reinterpret_cast<const float*>(sa_ptr),
        reinterpret_cast<const float*>(sb_ptr),
        reinterpret_cast<const int64_t*>(group_offs_ptr),
        {},
        /* G,n,k,ki,bpc,group_m,num_xcds,M_total,fast_n,fast_k,m_per_group,num_slots,chunk_size,fuse_ktail_off,sk_split_n */
        G, 0, 0, 0, 0, group_m, num_xcds, 0, 0, 0, m_per_group, num_slots, chunk_size, fuse_ktail_off, sk_split_n,
        // sk_partial_buf left default-init (nullptr); R13a alloc fills it when sk_split_n > 0.
    };
    // R17: caller-allocated workspace override (mirrors grouped_rcr_fn).
    if (sk_workspace_ptr != 0) {
        g.sk_partial_buf = reinterpret_cast<int*>(static_cast<uintptr_t>(sk_workspace_ptr));
    }
    dispatch_grouped_rcr(g);
}

static bool grouped_rcr_fused_act_dscale_fn(
    pybind11::object a, pybind11::object b, pybind11::object c,
    pybind11::object scale_a_inv_obj, pybind11::object scale_b_obj,
    pybind11::object group_offs_obj,
    int group_m,
    int m_per_group,
    int num_xcds) {
    auto sa_ptr = scale_a_inv_obj.attr("data_ptr")().cast<uintptr_t>();
    auto sb_ptr = scale_b_obj.attr("data_ptr")().cast<uintptr_t>();
    auto group_offs_ptr = group_offs_obj.attr("data_ptr")().cast<uintptr_t>();
    int G = group_offs_obj.attr("numel")().cast<int>() - 1;
    grouped_layout_globals_fused_act g{
        py::from_object<_gl_bf16>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        0.f, 0.f,
        reinterpret_cast<const float*>(sa_ptr),
        reinterpret_cast<const float*>(sb_ptr),
        reinterpret_cast<const int64_t*>(group_offs_ptr),
        {},
        G, 0, 0, 0, 0, group_m, num_xcds, 0, 0, 0, m_per_group,
    };

    const int K = static_cast<int>(g.a.cols());
    const int N = static_cast<int>(g.c.cols());
    if ((K % K_BLOCK) != 0 || (N <= 0) || (K <= 0)) {
        return false;
    }
    dispatch_grouped_rcr_fused_act(g);
    return true;
}

static void grouped_rrr_fn(pybind11::object a, pybind11::object b, pybind11::object c,
                           pybind11::object scale_a_obj, pybind11::object scale_b_obj,
                           pybind11::object group_offs_obj,
                           int group_m,
                           int m_per_group,
                           int num_xcds) {
    auto group_offs_ptr = group_offs_obj.attr("data_ptr")().cast<uintptr_t>();
    int G = group_offs_obj.attr("numel")().cast<int>() - 1;
    grouped_layout_globals g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        to_float(scale_a_obj),
        to_float(scale_b_obj),
        nullptr,
        nullptr,
        reinterpret_cast<const int64_t*>(group_offs_ptr),
        {},
        /* G,n,k,ki,bpc,group_m,num_xcds,M_total,fast_n,fast_k,m_per_group */
        G, 0, 0, 0, 0, group_m, num_xcds, 0, 0, 0, m_per_group,
    };
    dispatch_grouped_rrr(g);
}

static void grouped_rrr_dscale_fn(
    pybind11::object a, pybind11::object b, pybind11::object c,
    pybind11::object scale_a_obj, pybind11::object scale_b_obj,
    pybind11::object group_offs_obj,
    int group_m,
    int m_per_group,
    int num_xcds) {
    auto sa_ptr = scale_a_obj.attr("data_ptr")().cast<uintptr_t>();
    auto sb_ptr = scale_b_obj.attr("data_ptr")().cast<uintptr_t>();
    auto group_offs_ptr = group_offs_obj.attr("data_ptr")().cast<uintptr_t>();
    int G = group_offs_obj.attr("numel")().cast<int>() - 1;
    grouped_layout_globals g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        0.f, 0.f,
        reinterpret_cast<const float*>(sa_ptr),
        reinterpret_cast<const float*>(sb_ptr),
        reinterpret_cast<const int64_t*>(group_offs_ptr),
        {},
        /* G,n,k,ki,bpc,group_m,num_xcds,M_total,fast_n,fast_k,m_per_group */
        G, 0, 0, 0, 0, group_m, num_xcds, 0, 0, 0, m_per_group,
    };
    dispatch_grouped_rrr(g);
}

static void grouped_variable_k_crr_fp8_fn(
    pybind11::object a, pybind11::object b, pybind11::object c,
    pybind11::object scale_a_obj, pybind11::object scale_b_obj,
    pybind11::object group_offs_obj,
    int group_m,
    int num_xcds,
    int num_slots,
    int chunk_size = 0) {
    auto group_offs_ptr = group_offs_obj.attr("data_ptr")().cast<uintptr_t>();
    int G = group_offs_obj.attr("numel")().cast<int>() - 1;
    grouped_var_k_layout_globals_fp8 g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        to_float(scale_a_obj),
        to_float(scale_b_obj),
        nullptr,
        nullptr,
        reinterpret_cast<const int64_t*>(group_offs_ptr),
        {},
        /* G, M_total, n, k, group_m, bpr, bpc, fast_n, fast_k, num_xcds, num_slots, chunk_size */
        G, 0, 0, 0, group_m, 0, 0, 0, 0, num_xcds, num_slots, chunk_size,
    };
    dispatch_grouped_var_k_fp8(g);
}

static void grouped_variable_k_crr_dscale_fp8_fn(
    pybind11::object a, pybind11::object b, pybind11::object c,
    pybind11::object scale_a_obj, pybind11::object scale_b_obj,
    pybind11::object group_offs_obj,
    int group_m,
    int num_xcds,
    int num_slots,
    int chunk_size = 0) {
    auto sa_ptr = scale_a_obj.attr("data_ptr")().cast<uintptr_t>();
    auto sb_ptr = scale_b_obj.attr("data_ptr")().cast<uintptr_t>();
    auto group_offs_ptr = group_offs_obj.attr("data_ptr")().cast<uintptr_t>();
    int G = group_offs_obj.attr("numel")().cast<int>() - 1;
    grouped_var_k_layout_globals_fp8 g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        0.f, 0.f,
        reinterpret_cast<const float*>(sa_ptr),
        reinterpret_cast<const float*>(sb_ptr),
        reinterpret_cast<const int64_t*>(group_offs_ptr),
        {},
        /* G, M_total, n, k, group_m, bpr, bpc, fast_n, fast_k, num_xcds, num_slots, chunk_size */
        G, 0, 0, 0, group_m, 0, 0, 0, 0, num_xcds, num_slots, chunk_size,
    };
    dispatch_grouped_var_k_fp8(g);
}
#endif  // !PRIMUS_TURBO_HK_INTEGRATION (close range #1)

namespace fused_act_round4_compile_test {

// (cvt_bf16x4_to_fp8x4 definition relocated to line 1501 above for
// -fgpu-rdc compatibility; the test kernel below still uses it.)

// Compile-test kernel: round-trips one bf16x4 → fp8x4 cvt. Forces LLVM to
// emit codegen so any cvt-builtin issue surfaces at build time. Single-thread
// body keeps the resource-usage report focused on the cvt sequence.
__global__ __launch_bounds__(64, 1)
void cvt_bf16x4_to_fp8x4_compile_test(
    const bf16_2* __restrict__ src,    // length 2: 2 bf16_2 = 4 bf16
    uint32_t* __restrict__ dst,        // length 1: packed fp8x4
    float scale)
{
    if (threadIdx.x != 0) return;
    bf16_2 lo = src[0];
    bf16_2 hi = src[1];
    *dst = cvt_bf16x4_to_fp8x4(lo, hi, scale);
}

// Bulk version: each thread cvts its own bf16x4 group → fp8x4. Used by the
// Python-level numerical probe. Round-up the count to a multiple of 4 BF16.
__global__ __launch_bounds__(256, 1)
void cvt_bf16_to_fp8_bulk_compile_test(
    const bf16* __restrict__ src,
    fp8e4m3* __restrict__ dst,
    int64_t N,             // number of bf16 elements (must be multiple of 4)
    float scale)
{
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t N4 = N / 4;
    const bf16_2* src2 = reinterpret_cast<const bf16_2*>(src);
    uint32_t* dst4 = reinterpret_cast<uint32_t*>(dst);
    #pragma unroll 1
    for (int64_t i = tid; i < N4; i += stride) {
        bf16_2 lo = src2[2 * i + 0];
        bf16_2 hi = src2[2 * i + 1];
        dst4[i] = cvt_bf16x4_to_fp8x4(lo, hi, scale);
    }
}

}  // namespace fused_act_round4_compile_test

#ifndef PRIMUS_TURBO_HK_INTEGRATION
PYBIND11_MODULE(tk_fp8_layouts, m) {
    m.doc() = "FP8 per-tensor GEMM: C = A op B * scale_a * scale_b";
    m.def("gemm_rcr", &gemm_wrapper<Layout::RCR>,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M);
    m.def("gemm_rrr", &gemm_wrapper<Layout::RRR>,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M);
    m.def("gemm_crr", &gemm_wrapper<Layout::CRR>,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M);
    m.def("gemm_rcr_dscale", &gemm_wrapper_dscale<Layout::RCR>,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M);
    m.def("gemm_rrr_dscale", &gemm_wrapper_dscale<Layout::RRR>,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M);
    m.def("gemm_crr_dscale", &gemm_wrapper_dscale<Layout::CRR>,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M);
    m.def("supports_shape", [](int m, int n, int k) -> bool {
        return m > 0 && n > 0 && k > 0;
    });
    // [fused-act] BF16 max-abs reduction → single fp32 device scalar. Used by
    // Primus-Turbo's ``_fused_act_grouped_fp8_forward`` (Round 1 of the FP8
    // grouped fused-activation-quant lever) to produce the activation scale
    // BEFORE calling the C++ ``quantize_fp8_tensorwise(input, scale=...)`` —
    m.def("grouped_rcr", &grouped_rcr_fn,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M,
          pybind11::arg("m_per_group") = 0,
          pybind11::arg("num_xcds") = 0,
          pybind11::arg("num_slots") = 0,
          pybind11::arg("chunk_size") = 0,
          pybind11::arg("fuse_ktail_off") = 0,
          pybind11::arg("sk_split_n") = 0,
          pybind11::arg("sk_workspace_ptr") = uint64_t{0});
    m.def("grouped_rcr_dscale", &grouped_rcr_dscale_fn,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M,
          pybind11::arg("m_per_group") = 0,
          pybind11::arg("num_xcds") = 0,
          pybind11::arg("num_slots") = 0,
          pybind11::arg("chunk_size") = 0,
          pybind11::arg("fuse_ktail_off") = 0,
          pybind11::arg("sk_split_n") = 0,
          // R17: see grouped_rcr m.def above.
          pybind11::arg("sk_workspace_ptr") = uint64_t{0});
    m.def("grouped_rcr_fused_act_dscale", &grouped_rcr_fused_act_dscale_fn,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a_inv"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M,
          pybind11::arg("m_per_group") = 0,
          pybind11::arg("num_xcds") = 0);
    m.def("grouped_rrr", &grouped_rrr_fn,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M,
          pybind11::arg("m_per_group") = 0,
          pybind11::arg("num_xcds") = 0);
    m.def("grouped_rrr_dscale", &grouped_rrr_dscale_fn,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M,
          pybind11::arg("m_per_group") = 0,
          pybind11::arg("num_xcds") = 0);
    m.def("grouped_variable_k_crr", &grouped_variable_k_crr_fp8_fn,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M,
          pybind11::arg("num_xcds") = 0,
          pybind11::arg("num_slots") = 0,
          pybind11::arg("chunk_size") = 0);
    m.def("grouped_variable_k_crr_dscale",
          &grouped_variable_k_crr_dscale_fp8_fn,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M,
          pybind11::arg("num_xcds") = 0,
          pybind11::arg("num_slots") = 0,
          pybind11::arg("chunk_size") = 0);
    m.attr("DEFAULT_GROUP_M") = DEFAULT_GROUP_M;
    m.attr("BLOCK_SIZE") = BLK;
    m.attr("K_BLOCK") = BK;

}
#endif  // !PRIMUS_TURBO_HK_INTEGRATION
