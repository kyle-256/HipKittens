#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

#ifndef M_DIM
#define M_DIM 8192
#endif
#ifndef K_DIM
#define K_DIM 8192
#endif
#ifndef N_DIM
#define N_DIM 8192
#endif

#ifndef TK_FP8_LAYOUTS_MODULE_NAME
#define TK_FP8_LAYOUTS_MODULE_NAME tk_fp8_layouts
#endif

#define TK_STRINGIFY_IMPL(x) #x
#define TK_STRINGIFY(x) TK_STRINGIFY_IMPL(x)
#define TK_WAIT_LGKM(x) asm volatile("s_waitcnt lgkmcnt(" TK_STRINGIFY(x) ")")
#define TK_WAIT_VMCNT(x) asm volatile("s_waitcnt vmcnt(" TK_STRINGIFY(x) ")")
#define TK_PRAGMA_UNROLL(x) _Pragma(TK_STRINGIFY(unroll x))

#ifndef RCR_PREFETCH_LGKM
#define RCR_PREFETCH_LGKM 8
#endif
#ifndef RRR_PREFETCH_LGKM
#define RRR_PREFETCH_LGKM 8
#endif
#ifndef RRR_INIT0_VMCNT
#define RRR_INIT0_VMCNT 4
#endif
#ifndef RRR_INIT1_VMCNT
#define RRR_INIT1_VMCNT 6
#endif
#ifndef RRR_STEADY_VMCNT
#define RRR_STEADY_VMCNT 6
#endif
#ifndef RRR_EPILOGUE_VMCNT
#define RRR_EPILOGUE_VMCNT 4
#endif
#ifndef CRR_PREFETCH_LGKM
#define CRR_PREFETCH_LGKM 3
#endif

#ifndef RCR_MAIN_UNROLL
#define RCR_MAIN_UNROLL 2
#endif
#ifndef RRR_MAIN_UNROLL
#define RRR_MAIN_UNROLL 4
#endif
#ifndef CRR_MAIN_UNROLL
#define CRR_MAIN_UNROLL 1
#endif

#ifndef CRR_ENABLE_SCHED_BARRIER
#define CRR_ENABLE_SCHED_BARRIER 0
#endif

#ifndef CRR_INIT0_VMCNT
#define CRR_INIT0_VMCNT 4
#endif
#ifndef CRR_INIT1_VMCNT
#define CRR_INIT1_VMCNT 6
#endif
#ifndef CRR_STEADY_VMCNT
#define CRR_STEADY_VMCNT 4
#endif
#ifndef CRR_EPILOGUE_VMCNT
#define CRR_EPILOGUE_VMCNT 4
#endif
#ifndef CRR_BATCHED_PAIR_MMA
#define CRR_BATCHED_PAIR_MMA 0
#endif
#ifndef CRR_BATCHED_EPILOGUE_MMA
#define CRR_BATCHED_EPILOGUE_MMA 0
#endif
#ifndef CRR_ENABLE_STEADY_MID_BARRIER
#define CRR_ENABLE_STEADY_MID_BARRIER 1
#endif
#ifndef CRR_A_LDS_REENCODE
#define CRR_A_LDS_REENCODE 0
#endif
#ifndef CRR_ROW_SHARED_TRANSPOSE
#define CRR_ROW_SHARED_TRANSPOSE 0
#endif
#ifndef CRR_A_ROW_SHARED_TRANSPOSE
#define CRR_A_ROW_SHARED_TRANSPOSE CRR_ROW_SHARED_TRANSPOSE
#endif
#ifndef CRR_B_ROW_SHARED_TRANSPOSE
#define CRR_B_ROW_SHARED_TRANSPOSE CRR_ROW_SHARED_TRANSPOSE
#endif
#ifndef CRR_ROW_SHARED_ALIAS_VIEW
#define CRR_ROW_SHARED_ALIAS_VIEW 0
#endif
#ifndef TRANSPOSE_LOAD_USE_LOGICAL_COORDS
#define TRANSPOSE_LOAD_USE_LOGICAL_COORDS 0
#endif
#ifndef RRR_ROW_SHARED_TRANSPOSE
#define RRR_ROW_SHARED_TRANSPOSE 0
#endif
#ifndef RRR_B_REG_ROW_LOAD_TRANSPOSE
#define RRR_B_REG_ROW_LOAD_TRANSPOSE 1
#endif
#ifndef RRR_B_REG_ROW_LOAD_ALIAS
#define RRR_B_REG_ROW_LOAD_ALIAS 0
#endif
#ifndef RRR_B_CDNA4_PADDED_LAYOUT
#define RRR_B_CDNA4_PADDED_LAYOUT 0
#endif
#ifndef RRR_ROW_SHARED_DIRECT_BROW
#define RRR_ROW_SHARED_DIRECT_BROW 0
#endif
#ifndef CRR_USE_V3_SWIZZLE
#define CRR_USE_V3_SWIZZLE 0
#endif
#ifndef CRR_A_REG_ROW_LOAD_TRANSPOSE
#define CRR_A_REG_ROW_LOAD_TRANSPOSE 1
#endif
#ifndef CRR_B_REG_ROW_LOAD_TRANSPOSE
#define CRR_B_REG_ROW_LOAD_TRANSPOSE 1
#endif
#ifndef CRR_A_ALIAS_TILE_PERM
#define CRR_A_ALIAS_TILE_PERM 0
#endif

#if CRR_ENABLE_SCHED_BARRIER
#define CRR_SCHED_BARRIER() __builtin_amdgcn_sched_barrier(0)
#else
#define CRR_SCHED_BARRIER() do {} while (0)
#endif

#if CRR_ENABLE_STEADY_MID_BARRIER
#define CRR_STEADY_MID_BARRIER() __builtin_amdgcn_s_barrier()
#else
#define CRR_STEADY_MID_BARRIER() do {} while (0)
#endif

#ifndef RRR_ENABLE_SCHED_BARRIER
#define RRR_ENABLE_SCHED_BARRIER 0
#endif
#if RRR_ENABLE_SCHED_BARRIER
#define RRR_SCHED_BARRIER() __builtin_amdgcn_sched_barrier(0)
#else
#define RRR_SCHED_BARRIER() do {} while (0)
#endif

#if RRR_ROW_SHARED_DIRECT_BROW
#define RRR_MMA mma_ABt
#elif !RRR_ROW_SHARED_TRANSPOSE && RRR_B_REG_ROW_LOAD_TRANSPOSE
#define RRR_MMA mma_ABt
#else
#define RRR_MMA mma_AB
#endif

#if !CRR_A_ROW_SHARED_TRANSPOSE && !CRR_B_ROW_SHARED_TRANSPOSE && !CRR_A_LDS_REENCODE && !CRR_USE_V3_SWIZZLE && CRR_A_REG_ROW_LOAD_TRANSPOSE && !CRR_B_REG_ROW_LOAD_TRANSPOSE
#define CRR_STRICT_A_ROWREG_AB 1
#else
#define CRR_STRICT_A_ROWREG_AB 0
#endif

#if !CRR_A_ROW_SHARED_TRANSPOSE && !CRR_B_ROW_SHARED_TRANSPOSE && !CRR_A_LDS_REENCODE && !CRR_USE_V3_SWIZZLE && CRR_A_REG_ROW_LOAD_TRANSPOSE && CRR_B_REG_ROW_LOAD_TRANSPOSE
#define CRR_STRICT_ROWREG_ABT 1
#else
#define CRR_STRICT_ROWREG_ABT 0
#endif

#if CRR_STRICT_A_ROWREG_AB || CRR_STRICT_ROWREG_ABT
#define CRR_STRICT_A_ROWREG 1
#else
#define CRR_STRICT_A_ROWREG 0
#endif

#if CRR_STRICT_ROWREG_ABT
#define CRR_STRICT_MMA mma_ABt
#elif CRR_STRICT_A_ROWREG_AB
#define CRR_STRICT_MMA mma_AB
#else
#define CRR_STRICT_MMA mma_AtB
#endif

#define CRR_MMA_BEGIN() do { CRR_SCHED_BARRIER(); __builtin_amdgcn_s_setprio(1); } while (0)
#define CRR_MMA_END() do { __builtin_amdgcn_s_setprio(0); CRR_SCHED_BARRIER(); } while (0)

#if CRR_ROW_SHARED_TRANSPOSE && (!CRR_A_ROW_SHARED_TRANSPOSE || !CRR_B_ROW_SHARED_TRANSPOSE)
#error "CRR_ROW_SHARED_TRANSPOSE implies both CRR_A_ROW_SHARED_TRANSPOSE and CRR_B_ROW_SHARED_TRANSPOSE"
#endif

#if CRR_A_LDS_REENCODE && CRR_A_ROW_SHARED_TRANSPOSE
#error "CRR_A_LDS_REENCODE is only valid on the strict non-row-shared path"
#endif

#if CRR_A_LDS_REENCODE && CRR_BATCHED_EPILOGUE_MMA
#error "CRR_A_LDS_REENCODE currently requires CRR_BATCHED_EPILOGUE_MMA=0"
#endif

#if CRR_USE_V3_SWIZZLE && CRR_A_LDS_REENCODE
#error "CRR_USE_V3_SWIZZLE is only supported on the strict non-reencode path"
#endif

#if CRR_USE_V3_SWIZZLE && (CRR_A_ROW_SHARED_TRANSPOSE || CRR_B_ROW_SHARED_TRANSPOSE)
#error "CRR_USE_V3_SWIZZLE is only supported on the fully non-row-shared path"
#endif

#if (CRR_A_REG_ROW_LOAD_TRANSPOSE || CRR_B_REG_ROW_LOAD_TRANSPOSE) && (CRR_A_ROW_SHARED_TRANSPOSE || CRR_B_ROW_SHARED_TRANSPOSE || CRR_A_LDS_REENCODE || CRR_USE_V3_SWIZZLE)
#error "CRR_*_REG_ROW_LOAD_TRANSPOSE only supports the strict v2/v2a non-reencode path"
#endif

#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_ROW_SHARED_TRANSPOSE
#error "RRR_B_REG_ROW_LOAD_TRANSPOSE only supports the strict non-row-shared path"
#endif

#if RRR_B_CDNA4_PADDED_LAYOUT && RRR_ROW_SHARED_TRANSPOSE
#error "RRR_B_CDNA4_PADDED_LAYOUT only supports the strict non-row-shared path"
#endif

#if RRR_ROW_SHARED_DIRECT_BROW && !RRR_ROW_SHARED_TRANSPOSE
#error "RRR_ROW_SHARED_DIRECT_BROW requires RRR_ROW_SHARED_TRANSPOSE"
#endif

constexpr int BLK = 256, BK = 128;
constexpr int HB  = BLK / 2;
constexpr int WARPS_M = 2, WARPS_N = 4;
constexpr int _NUM_WARPS   = WARPS_M * WARPS_N;
constexpr int _NUM_THREADS = _NUM_WARPS * WARP_THREADS;
constexpr int RBM = BLK / WARPS_M / 2;   // 64
constexpr int RBN = BLK / WARPS_N / 2;   // 32
constexpr int BPR = M_DIM / BLK;
constexpr int BPC = N_DIM / BLK;
constexpr int KI  = K_DIM / BK;

using G = kittens::group<_NUM_WARPS>;
using _gl_fp8  = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_bf16 = gl<bf16, -1, -1, -1, -1>;

enum class Layout { RCR, RRR, CRR };

// Row-layout shared/register tiles (for A in RCR/RRR, B in RCR)
using ST_row = st_fp8e4m3<HB, BK, st_16x128_s>;    // 128×128, M/N rows × K cols
using A_row_reg = rt_fp8e4m3<RBM, BK, row_l, rt_16x128_s>;
using B_row_reg = rt_fp8e4m3<RBN, BK, row_l, rt_16x128_s>;
using A_src_row_reg = rt_fp8e4m3<BK, RBM, row_l, rt_128x16_s>;
using A_reenc_col_reg = rt_fp8e4m3<RBM, BK, col_l, rt_16x128_s>;
using ST_crr_a_reenc = st_fp8e4m3<RBM, BK, st_16x128_s>;

// Col-layout register tiles (for B in RRR, A/B in CRR)
using A_col_reg = rt_fp8e4m3<BK, RBM, col_l, rt_128x16_s>;  // 128×64
using B_col_reg = rt_fp8e4m3<BK, RBN, col_l, rt_128x16_s>;  // 128×32

using ST_v2  = st_fp8e4m3<HB, BK, st_16x128_v2_s>;
using ST_v2a = st_fp8e4m3<HB, BK, st_16x128_v2a_s>;
using ST_v3  = st_fp8e4m3<HB, BK, st_16x128_v3_s>;
using ST_cdna4_fp8_a = st_fp8e4m3<HB, BK, st_128x128_cdna4_fp8_a_s>;
using ST_cdna4_fp8_b = st_fp8e4m3<HB, BK, st_128x128_cdna4_fp8_b_s>;

template<typename RT, int K_HALF>
__device__ __forceinline__ void load_col_from_v2_st_half(
    RT& dst, const ST_v2& tile, int col_start)
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
        const uint32_t next_addr = base_k + 1024 + (nc ^ sw_k);

        asm volatile(
            "ds_read_b64_tr_b8 %0, %2 offset:%4\n"
            "ds_read_b64_tr_b8 %1, %3 offset:%4\n"
            : "=v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx])),
              "=v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx + 2]))
            : "v"(addr), "v"(next_addr), "i"(0)
            : "memory"
        );
    }
}

template<typename RT>
__device__ __forceinline__ void load_col_from_v2_st(
    RT& dst, const ST_v2& tile, int col_start)
{
    load_col_from_v2_st_half<RT, 0>(dst, tile, col_start);
    load_col_from_v2_st_half<RT, 1>(dst, tile, col_start);
}

template<typename RT, int K_HALF, typename ST>
__device__ __forceinline__ void load_col_from_v2a_st_half(
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
        const uint32_t next_addr = base_k + 1024 + (nc ^ sw_k);

        asm volatile(
            "ds_read_b64_tr_b8 %0, %2 offset:%4\n"
            "ds_read_b64_tr_b8 %1, %3 offset:%4\n"
            : "=v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx])),
              "=v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx + 2]))
            : "v"(addr), "v"(next_addr), "i"(0)
            : "memory"
        );
    }
}

template<typename RT, typename ST>
__device__ __forceinline__ void load_col_from_v2a_st(
    RT& dst, const ST& tile, int col_start)
{
    load_col_from_v2a_st_half<RT, 0>(dst, tile, col_start);
    load_col_from_v2a_st_half<RT, 1>(dst, tile, col_start);
}

template<typename RT, int K_HALF>
__device__ __forceinline__ void load_col_from_cdna4_fp8_b_st_half(
    RT& dst, const ST_cdna4_fp8_b& tile, int col_start)
{
    const int laneid = kittens::laneid();
    const int row_off = ((laneid % 16) / 2) + ((laneid / 16) * 16);
    const int col_off = (laneid % 2) * 8;
    const uint32_t tile_base = reinterpret_cast<uintptr_t>(&tile.data[0]);

    constexpr int idx = K_HALF * 4;
    const int k_row = row_off + K_HALF * 64;

    #pragma unroll
    for (int j = 0; j < RT::width; j++) {
        const int n_col = col_start + j * 16 + col_off;
        const uint32_t addr = tile_base + tile.swizzle({k_row, n_col});
        const uint32_t next_addr = tile_base + tile.swizzle({k_row + 8, n_col});

        asm volatile(
            "ds_read_b64_tr_b8 %0, %2 offset:%4\n"
            "ds_read_b64_tr_b8 %1, %3 offset:%4\n"
            : "=v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx])),
              "=v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx + 2]))
            : "v"(addr), "v"(next_addr), "i"(0)
            : "memory"
        );
    }
}

template<typename RT>
__device__ __forceinline__ void load_col_from_cdna4_fp8_b_st(
    RT& dst, const ST_cdna4_fp8_b& tile, int col_start)
{
    load_col_from_cdna4_fp8_b_st_half<RT, 0>(dst, tile, col_start);
    load_col_from_cdna4_fp8_b_st_half<RT, 1>(dst, tile, col_start);
}

template<typename ST>
__device__ __forceinline__ uint8_t read_st_v2_scalar_u8(
    const ST& tile, int row, int col)
{
    const int subtile_row = row / ST::underlying_subtile_rows;
    const int local_row = row % ST::underlying_subtile_rows;
    const uint32_t subtile_offset = subtile_row * ST::underlying_subtile_stride_bytes;
    const uint32_t scalar_offset = subtile_offset + tile.swizzle({local_row, col});
    const auto* addr = reinterpret_cast<const uint8_t*>(
        reinterpret_cast<const char*>(&tile.data[0]) + scalar_offset
    );
    return *addr;
}

template<typename ST>
__device__ __forceinline__ uint32_t read_st_v2_packed4_bits(
    const ST& tile, int row, int col)
{
    uint32_t bits = 0;
    bits |= static_cast<uint32_t>(read_st_v2_scalar_u8(tile, row + 0, col)) << 0;
    bits |= static_cast<uint32_t>(read_st_v2_scalar_u8(tile, row + 1, col)) << 8;
    bits |= static_cast<uint32_t>(read_st_v2_scalar_u8(tile, row + 2, col)) << 16;
    bits |= static_cast<uint32_t>(read_st_v2_scalar_u8(tile, row + 3, col)) << 24;
    return bits;
}

template<ducks::rt::row_layout RT, typename ST>
__device__ __forceinline__ void load_row_from_v2_st(
    RT& dst, const ST& tile, int row_start)
{
    static_assert(std::is_same_v<typename RT::T, fp8e4m3>, "FP8-only row loader");
    constexpr int packing = base_types::packing<typename RT::dtype>::num();
    const int laneid = kittens::laneid();
    const int row_offset = laneid % RT::base_tile_rows;
    const int col_offset = RT::base_tile_stride * (laneid / RT::base_tile_rows);

    #pragma unroll 1
    for (int i = 0; i < RT::height; ++i) {
        const int n_row = row_start + i * RT::base_tile_rows + row_offset;
        #pragma unroll 1
        for (int k = 0; k < RT::base_tile_num_strides; ++k) {
            const int k_col = k * RT::base_tile_elements_per_stride_group + col_offset;
            const int idx = k * RT::base_tile_stride / packing;
            #pragma unroll
            for (int p = 0; p < RT::base_tile_stride / packing; ++p) {
                const uint32_t bits = read_st_v2_packed4_bits(tile, k_col + p * packing, n_row);
                dst.tiles[i][0].data[idx + p] = std::bit_cast<fp8e4m3_4>(bits);
            }
        }
    }
}

template<bool ENABLE, typename AliasRT, typename StorageRT>
__device__ __forceinline__ decltype(auto) rowreg_alias(StorageRT& src)
{
    if constexpr (ENABLE) {
        return *reinterpret_cast<AliasRT*>(&src);
    } else {
        return (src);
    }
}

template<bool ENABLE, typename AliasRT, typename StorageRT>
__device__ __forceinline__ decltype(auto) rowreg_alias(const StorageRT& src)
{
    if constexpr (ENABLE) {
        return *reinterpret_cast<const AliasRT*>(&src);
    } else {
        return (src);
    }
}

template<int PERM, typename RT>
__device__ __forceinline__ void apply_width4_tile_perm(RT& dst)
{
    if constexpr (PERM != 0) {
        auto t0 = dst.tiles[0][0];
        auto t1 = dst.tiles[0][1];
        auto t2 = dst.tiles[0][2];
        auto t3 = dst.tiles[0][3];
        if constexpr (PERM == 1) {          // 0,2,1,3
            dst.tiles[0][0] = t0;
            dst.tiles[0][1] = t2;
            dst.tiles[0][2] = t1;
            dst.tiles[0][3] = t3;
        } else if constexpr (PERM == 2) {   // 2,3,0,1
            dst.tiles[0][0] = t2;
            dst.tiles[0][1] = t3;
            dst.tiles[0][2] = t0;
            dst.tiles[0][3] = t1;
        } else if constexpr (PERM == 3) {   // 1,0,3,2
            dst.tiles[0][0] = t1;
            dst.tiles[0][1] = t0;
            dst.tiles[0][2] = t3;
            dst.tiles[0][3] = t2;
        } else if constexpr (PERM == 4) {   // 3,2,1,0
            dst.tiles[0][0] = t3;
            dst.tiles[0][1] = t2;
            dst.tiles[0][2] = t1;
            dst.tiles[0][3] = t0;
        }
    }
}

template<typename RT, int K_HALF>
__device__ __forceinline__ void load_col_from_v3_st_half(
    RT& dst, const ST_v3& tile, int col_start)
{
    const int laneid = kittens::laneid();
    const int row_off = ((laneid % 16) / 2) + ((laneid / 16) * 16);
    const int col_off = (laneid % 2) * 8;
    const uint32_t tile_base = reinterpret_cast<uintptr_t>(&tile.data[0]);

    constexpr int idx = K_HALF * 4;
    const int k_row = row_off + K_HALF * 64;
    const uint32_t stidx = k_row >> 4;
    const uint32_t local_row = k_row & 15;
    const uint32_t base_k = tile_base + (stidx << 11) + (local_row << 7);
    const uint32_t sw_k = local_row << 3;
    const uint32_t base_n = base_k + 1024;
    const uint32_t sw_n = (local_row + 8) << 3;

    #pragma unroll
    for (int j = 0; j < RT::width; j++) {
        const uint32_t nc = col_start + j * 16 + col_off;
        const uint32_t addr = base_k + (nc ^ sw_k);
        const uint32_t next_addr = base_n + (nc ^ sw_n);

        asm volatile(
            "ds_read_b64_tr_b8 %0, %2 offset:%4\n"
            "ds_read_b64_tr_b8 %1, %3 offset:%4\n"
            : "=v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx])),
              "=v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx + 2]))
            : "v"(addr), "v"(next_addr), "i"(0)
            : "memory"
        );
    }
}

template<typename RT>
__device__ __forceinline__ void load_col_from_v3_st(
    RT& dst, const ST_v3& tile, int col_start)
{
    load_col_from_v3_st_half<RT, 0>(dst, tile, col_start);
    load_col_from_v3_st_half<RT, 1>(dst, tile, col_start);
}

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

#if TRANSPOSE_LOAD_USE_LOGICAL_COORDS
        const int transposed_global_row =
            col + subtile_col * ST::underlying_subtile_cols;
        const int transposed_global_col =
            row + subtile_row * ST::underlying_subtile_rows;
#else
        const int transposed_global_row =
            shared_col + subtile_col * ST::underlying_subtile_cols;
        const int transposed_global_col =
            shared_row + subtile_row * ST::underlying_subtile_rows;
#endif

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

#if TRANSPOSE_LOAD_USE_LOGICAL_COORDS
            const int transposed_global_row =
                col + subtile_col * ST::underlying_subtile_cols;
            const int transposed_global_col =
                row + subtile_row * ST::underlying_subtile_rows;
#else
            const int transposed_global_row =
                shared_col + subtile_col * ST::underlying_subtile_cols;
            const int transposed_global_col =
                shared_row + subtile_row * ST::underlying_subtile_rows;
#endif

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

struct layout_globals {
    _gl_fp8 a, b;
    _gl_bf16 c;
    float scale = 1.0f;
    hipStream_t stream;
    dim3 grid()  { return dim3(BPR * BPC); }
    dim3 block() { return dim3(_NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

template<Layout L>
__global__ __launch_bounds__(_NUM_THREADS, 2)
void gemm_kernel(const layout_globals g) {
    int bid = blockIdx.x;
    int br = bid / BPC, bc = bid % BPC;
    int wm = warpid() / WARPS_N, wn = warpid() % WARPS_N;

    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;
    zero(cA); zero(cB); zero(cC); zero(cD);

    if constexpr (L == Layout::RCR) {
        __shared__ ST_row As[2][2];
        __shared__ ST_row Bs[2][2];
        A_row_reg a;
        B_row_reg b0, b1;

        constexpr int bpt = ST_row::underlying_subtile_bytes_per_thread;
        constexpr int bpm = bpt * _NUM_THREADS;
        constexpr int mpt = ST_row::rows * ST_row::cols * sizeof(fp8e4m3) / bpm;
        uint32_t soA[mpt], soB[mpt];
        G::prefill_swizzled_offsets(As[0][0], g.a, soA);
        G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

        auto a_co = [&](int s, int k) -> coord<ST_row> { return {0, 0, s, k}; };
        auto b_co = [&](int s, int k) -> coord<ST_row> { return {0, 0, s, k}; };

        auto load_a = [&](A_row_reg& dst, ST_row& tile, int wi) {
            auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
            load(dst, sub);
        };
        auto load_b = [&](B_row_reg& dst, ST_row& tile, int wi) {
            auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
            load(dst, sub);
        };

        int tic = 0, toc = 1;
        G::load(Bs[tic][0], g.b, b_co(bc*2,   0), soB);
        G::load(As[tic][0], g.a, a_co(br*2,   0), soA);
        G::load(Bs[tic][1], g.b, b_co(bc*2+1, 0), soB);
        G::load(As[tic][1], g.a, a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(CRR_INIT0_VMCNT);
        __builtin_amdgcn_s_barrier();

        G::load(Bs[toc][0], g.b, b_co(bc*2,   1), soB);
        G::load(As[toc][0], g.a, a_co(br*2,   1), soA);
        G::load(Bs[toc][1], g.b, b_co(bc*2+1, 1), soB);

        TK_WAIT_VMCNT(CRR_INIT1_VMCNT);
        __builtin_amdgcn_s_barrier();

        TK_PRAGMA_UNROLL(RCR_MAIN_UNROLL)
        for (int k = 0; k < KI - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, k+1), soA);
            TK_WAIT_LGKM(RCR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cA, a, b0, cA); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);

            load_b(b1, Bs[tic][1], wn);
            G::load(Bs[tic][0], g.b, b_co(bc*2, k+2), soB);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cB, a, b1, cB); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            G::load(As[tic][0], g.a, a_co(br*2, k+2), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cC, a, b0, cC); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);

            G::load(Bs[tic][1], g.b, b_co(bc*2+1, k+2), soB);
            asm volatile("s_waitcnt vmcnt(6)"); __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_setprio(1); mma_ABt(cD, a, b1, cD); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, KI-1), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cA, a, b0, cA); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cB, a, b1, cB); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            asm volatile("s_waitcnt vmcnt(4)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cC, a, b0, cC); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[toc][0], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cD, a, b1, cD); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
            tic ^= 1; toc ^= 1;
        }

        {
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cA, a, b0, cA); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cB, a, b1, cB); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(cC, a, b0, cC);
            mma_ABt(cD, a, b1, cD);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

    } else if constexpr (L == Layout::RRR) {
        __shared__ ST_row As[2][2];
#if RRR_ROW_SHARED_TRANSPOSE
        __shared__ ST_row Bs[2][2];
#elif RRR_B_CDNA4_PADDED_LAYOUT
        __shared__ ST_cdna4_fp8_b Bs[2][2];
#else
        __shared__ ST_v2  Bs[2][2];
#endif
        A_row_reg a;
#if RRR_ROW_SHARED_TRANSPOSE && RRR_ROW_SHARED_DIRECT_BROW
        B_row_reg b;
#else
        B_col_reg b;
#endif

        constexpr int bptA = ST_row::underlying_subtile_bytes_per_thread;
        constexpr int bpmA = bptA * _NUM_THREADS;
        constexpr int mptA = ST_row::rows * ST_row::cols * sizeof(fp8e4m3) / bpmA;
        uint32_t soA[mptA];
        G::prefill_swizzled_offsets(As[0][0], g.a, soA);

        constexpr int bptB =
#if RRR_ROW_SHARED_TRANSPOSE
            ST_row::underlying_subtile_bytes_per_thread;
#elif RRR_B_CDNA4_PADDED_LAYOUT
            ST_cdna4_fp8_b::underlying_subtile_bytes_per_thread;
#else
            ST_v2::underlying_subtile_bytes_per_thread;
#endif
        constexpr int bpmB = bptB * _NUM_THREADS;
        constexpr int mptB =
#if RRR_ROW_SHARED_TRANSPOSE
            ST_row::rows * ST_row::cols * sizeof(fp8e4m3) / bpmB;
#elif RRR_B_CDNA4_PADDED_LAYOUT
            ST_cdna4_fp8_b::rows * ST_cdna4_fp8_b::cols * sizeof(fp8e4m3) / bpmB;
#else
            ST_v2::rows * ST_v2::cols * sizeof(fp8e4m3) / bpmB;
#endif
        uint32_t soB[mptB];
#if RRR_ROW_SHARED_TRANSPOSE
        prefill_transpose_swizzled_offsets<_NUM_THREADS>(Bs[0][0], g.b, soB);
#else
        G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);
#endif

        auto a_co = [&](int s, int k) -> coord<ST_row> { return {0, 0, s, k}; };
#if RRR_ROW_SHARED_TRANSPOSE
        // Load B from its original KxN layout, transpose it on-chip into row-friendly
        // shared tiles, then transpose once more in registers into the col_l MMA view.
        auto b_co = [&](int s, int k) -> coord<ST_row> { return {0, 0, s, k}; };
#elif RRR_B_CDNA4_PADDED_LAYOUT
        auto b_co = [&](int s, int k) -> coord<ST_cdna4_fp8_b> { return {0, 0, k, s}; };
#else
        auto b_co = [&](int s, int k) -> coord<ST_v2>  { return {0, 0, k, s}; };
#endif

        auto load_a = [&](A_row_reg& dst, ST_row& tile, int wi) {
            auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
            load(dst, sub);
        };
#if RRR_ROW_SHARED_TRANSPOSE
        auto load_b = [&](auto& dst, ST_row& tile, int wi) {
            auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
#if RRR_ROW_SHARED_DIRECT_BROW
            load(dst, sub);
#else
            B_row_reg tmp;
            load(tmp, sub);
            transpose(dst, tmp);
#endif
        };
#elif RRR_B_CDNA4_PADDED_LAYOUT
        auto load_b = [&](B_col_reg& dst, ST_cdna4_fp8_b& tile, int wi) {
            load_col_from_cdna4_fp8_b_st(dst, tile, wi * RBN);
        };
#else
        auto load_b = [&](B_col_reg& dst, ST_v2& tile, int wi) {
            load_col_from_v2_st(dst, tile, wi * RBN);
        };
#endif
        auto mma_b = [&](auto& src) -> decltype(auto) {
#if RRR_ROW_SHARED_TRANSPOSE && RRR_ROW_SHARED_DIRECT_BROW
            return (src);
#else
            return rowreg_alias<RRR_B_REG_ROW_LOAD_TRANSPOSE, B_row_reg>(src);
#endif
        };

        int tic = 0, toc = 1;
#if RRR_ROW_SHARED_TRANSPOSE
        load_transpose<_NUM_THREADS>(Bs[tic][0], g.b, b_co(bc*2,   0), soB);
#else
        G::load(Bs[tic][0], g.b, b_co(bc*2,   0), soB);
#endif
        G::load(As[tic][0], g.a, a_co(br*2,   0), soA);
#if RRR_ROW_SHARED_TRANSPOSE
        load_transpose<_NUM_THREADS>(Bs[tic][1], g.b, b_co(bc*2+1, 0), soB);
#else
        G::load(Bs[tic][1], g.b, b_co(bc*2+1, 0), soB);
#endif
        G::load(As[tic][1], g.a, a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(RRR_INIT0_VMCNT);
        __builtin_amdgcn_s_barrier();

#if RRR_ROW_SHARED_TRANSPOSE
        load_transpose<_NUM_THREADS>(Bs[toc][0], g.b, b_co(bc*2,   1), soB);
#else
        G::load(Bs[toc][0], g.b, b_co(bc*2,   1), soB);
#endif
        G::load(As[toc][0], g.a, a_co(br*2,   1), soA);
#if RRR_ROW_SHARED_TRANSPOSE
        load_transpose<_NUM_THREADS>(Bs[toc][1], g.b, b_co(bc*2+1, 1), soB);
#else
        G::load(Bs[toc][1], g.b, b_co(bc*2+1, 1), soB);
#endif

        TK_WAIT_VMCNT(RRR_INIT1_VMCNT);
        __builtin_amdgcn_s_barrier();

        TK_PRAGMA_UNROLL(RRR_MAIN_UNROLL)
        for (int k = 0; k < KI - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, k+1), soA);
            TK_WAIT_LGKM(RRR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); RRR_MMA(cA, a, mma_b(b), cA); __builtin_amdgcn_s_setprio(0);
            auto b0_keep = b;
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();

            load_a(a, As[tic][1], wm);
#if RRR_ROW_SHARED_TRANSPOSE
            load_transpose<_NUM_THREADS>(Bs[tic][0], g.b, b_co(bc*2, k+2), soB);
#else
            G::load(Bs[tic][0], g.b, b_co(bc*2, k+2), soB);
#endif
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); RRR_MMA(cC, a, mma_b(b0_keep), cC); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); RRR_MMA(cD, a, mma_b(b), cD); __builtin_amdgcn_s_setprio(0);
            auto b1_keep = b;
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();

            load_a(a, As[tic][0], wm);
#if RRR_ROW_SHARED_TRANSPOSE
            load_transpose<_NUM_THREADS>(Bs[tic][1], g.b, b_co(bc*2+1, k+2), soB);
#else
            G::load(Bs[tic][1], g.b, b_co(bc*2+1, k+2), soB);
#endif
            TK_WAIT_VMCNT(RRR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); RRR_MMA(cB, a, mma_b(b1_keep), cB); __builtin_amdgcn_s_setprio(0);
            // Keep As[tic][0] alive until cB consumes A0(k), then recycle it for A0(k+2).
            G::load(As[tic][0], g.a, a_co(br*2, k+2), soA);
            __builtin_amdgcn_s_barrier();
        }

        {
            load_b(b, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, KI-1), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); RRR_MMA(cA, a, mma_b(b), cA); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); RRR_MMA(cC, a, mma_b(b), cC); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); RRR_MMA(cD, a, mma_b(b), cD); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][0], wm);
            TK_WAIT_VMCNT(RRR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); RRR_MMA(cB, a, mma_b(b), cB); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            RRR_SCHED_BARRIER();
            tic ^= 1; toc ^= 1;
        }

        {
            load_b(b, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); RRR_MMA(cA, a, mma_b(b), cA); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); RRR_MMA(cC, a, mma_b(b), cC); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); RRR_MMA(cD, a, mma_b(b), cD); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][0], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            RRR_MMA(cB, a, mma_b(b), cB);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

    } else if constexpr (L == Layout::CRR) {
#if CRR_A_ROW_SHARED_TRANSPOSE
        using ST_crr_a = ST_row;
#else
    #if CRR_USE_V3_SWIZZLE
        using ST_crr_a = ST_v3;
    #else
        using ST_crr_a = ST_v2a;
    #endif
#endif
#if CRR_B_ROW_SHARED_TRANSPOSE
        using ST_crr_b = ST_row;
#else
    #if CRR_USE_V3_SWIZZLE
        using ST_crr_b = ST_v3;
    #else
        using ST_crr_b = ST_v2;
    #endif
#endif
        __shared__ ST_crr_a As[2][2];
        __shared__ ST_crr_b Bs[2][2];
#if CRR_A_LDS_REENCODE
        __shared__ ST_crr_a_reenc Aenc[2];
        A_row_reg a;
        B_col_reg b0, b1;
#else
        A_col_reg a;
        B_col_reg b0, b1;
#endif

        constexpr int bptA = ST_crr_a::underlying_subtile_bytes_per_thread;
        constexpr int bpmA = bptA * _NUM_THREADS;
        constexpr int mptA = ST_crr_a::rows * ST_crr_a::cols * sizeof(fp8e4m3) / bpmA;
        uint32_t soA[mptA];
#if CRR_A_ROW_SHARED_TRANSPOSE
        prefill_transpose_swizzled_offsets<_NUM_THREADS>(As[0][0], g.a, soA);
#else
        G::prefill_swizzled_offsets(As[0][0], g.a, soA);
#endif

        constexpr int bptB = ST_crr_b::underlying_subtile_bytes_per_thread;
        constexpr int bpmB = bptB * _NUM_THREADS;
        constexpr int mptB = ST_crr_b::rows * ST_crr_b::cols * sizeof(fp8e4m3) / bpmB;
        uint32_t soB[mptB];
#if CRR_B_ROW_SHARED_TRANSPOSE
        prefill_transpose_swizzled_offsets<_NUM_THREADS>(Bs[0][0], g.b, soB);
#else
        G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);
#endif

#if CRR_A_ROW_SHARED_TRANSPOSE
        auto a_co = [&](int s, int k) -> coord<ST_crr_a> { return {0, 0, s, k}; };
        auto global_load_a = [&](ST_crr_a& tile, int s, int k) {
            load_transpose<_NUM_THREADS>(tile, g.a, a_co(s, k), soA);
        };
#else
        auto a_co = [&](int s, int k) -> coord<ST_crr_a> { return {0, 0, k, s}; };
        auto global_load_a = [&](ST_crr_a& tile, int s, int k) {
            G::load(tile, g.a, a_co(s, k), soA);
        };
#endif
#if CRR_B_ROW_SHARED_TRANSPOSE
        auto b_co = [&](int s, int k) -> coord<ST_crr_b> { return {0, 0, s, k}; };
        auto global_load_b = [&](ST_crr_b& tile, int s, int k) {
            load_transpose<_NUM_THREADS>(tile, g.b, b_co(s, k), soB);
        };
#else
        auto b_co = [&](int s, int k) -> coord<ST_crr_b> { return {0, 0, k, s}; };
        auto global_load_b = [&](ST_crr_b& tile, int s, int k) {
            G::load(tile, g.b, b_co(s, k), soB);
        };
#endif

#if CRR_A_LDS_REENCODE
        auto reencode_a = [&](ST_crr_a& tile) {
            if (wn == 0) {
                A_col_reg tmp_src;
                A_row_reg tmp_dst;
                load_col_from_v2a_st(tmp_src, tile, wm * RBM);
                transpose(tmp_dst, tmp_src);
                store(Aenc[wm], tmp_dst);
            }
            __builtin_amdgcn_s_barrier();
        };
        auto load_a = [&](A_row_reg& dst, int wi) {
            load(dst, Aenc[wi]);
        };
#else
        auto load_a = [&](A_col_reg& dst, ST_crr_a& tile, int wi) {
#if CRR_A_ROW_SHARED_TRANSPOSE
            A_row_reg tmp;
            auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
            load(tmp, sub);
#if CRR_ROW_SHARED_ALIAS_VIEW
            dst = rowreg_alias<true, A_col_reg>(tmp);
#else
            transpose(dst, tmp);
#endif
#elif CRR_USE_V3_SWIZZLE
            load_col_from_v3_st(dst, tile, wi * RBM);
#else
            load_col_from_v2a_st(dst, tile, wi * RBM);
#if CRR_STRICT_A_ROWREG
            if (wi == 1) {
                apply_width4_tile_perm<CRR_A_ALIAS_TILE_PERM>(dst);
            }
#endif
#endif
        };
#endif
        auto load_b = [&](B_col_reg& dst, ST_crr_b& tile, int wi) {
#if CRR_B_ROW_SHARED_TRANSPOSE
            B_row_reg tmp;
            auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
            load(tmp, sub);
#if CRR_ROW_SHARED_ALIAS_VIEW
            dst = rowreg_alias<true, B_col_reg>(tmp);
#else
            transpose(dst, tmp);
#endif
#elif CRR_USE_V3_SWIZZLE
            load_col_from_v3_st(dst, tile, wi * RBN);
#else
            load_col_from_v2_st(dst, tile, wi * RBN);
#endif
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

#if !CRR_ROW_SHARED_TRANSPOSE && CRR_A_LDS_REENCODE
        reencode_a(As[tic][0]);
#endif

        TK_PRAGMA_UNROLL(CRR_MAIN_UNROLL)
        for (int k = 0; k < KI - 2; k++, tic ^= 1, toc ^= 1) {
#if CRR_BATCHED_PAIR_MMA
#if CRR_A_LDS_REENCODE
            load_b(b0, Bs[tic][0], wn);
            load_b(b1, Bs[tic][1], wn);
            load_a(a, wm);
            global_load_a(As[toc][1], br*2+1, k+1);
            global_load_b(Bs[tic][0], bc*2, k+2);
            TK_WAIT_LGKM(CRR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            mma_AB(cA, a, b0, cA);
            mma_AB(cB, a, b1, cB);
            CRR_MMA_END();
            CRR_STEADY_MID_BARRIER(); CRR_SCHED_BARRIER();

            reencode_a(As[tic][1]);
            load_a(a, wm);
            global_load_a(As[tic][0], br*2, k+2);
            global_load_b(Bs[tic][1], bc*2+1, k+2);
            TK_WAIT_VMCNT(CRR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            mma_AB(cC, a, b0, cC);
            mma_AB(cD, a, b1, cD);
            CRR_MMA_END();
            reencode_a(As[toc][0]);
#else
            load_b(b0, Bs[tic][0], wn);
            load_b(b1, Bs[tic][1], wn);
            load_a(a, As[tic][0], wm);
            global_load_a(As[toc][1], br*2+1, k+1);
            TK_WAIT_LGKM(CRR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            CRR_STRICT_MMA(
                cA,
                rowreg_alias<CRR_STRICT_A_ROWREG, A_row_reg>(a),
                rowreg_alias<CRR_STRICT_ROWREG_ABT, B_row_reg>(b0),
                cA
            );
            CRR_STRICT_MMA(
                cB,
                rowreg_alias<CRR_STRICT_A_ROWREG, A_row_reg>(a),
                rowreg_alias<CRR_STRICT_ROWREG_ABT, B_row_reg>(b1),
                cB
            );
            CRR_MMA_END();
            auto b0_keep = b0;
            global_load_b(Bs[tic][0], bc*2, k+2);
            CRR_STEADY_MID_BARRIER(); CRR_SCHED_BARRIER();

            load_a(a, As[tic][1], wm);
            global_load_a(As[tic][0], br*2, k+2);
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            CRR_STRICT_MMA(
                cC,
                rowreg_alias<CRR_STRICT_A_ROWREG, A_row_reg>(a),
                rowreg_alias<CRR_STRICT_ROWREG_ABT, B_row_reg>(b0_keep),
                cC
            );
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier(); CRR_SCHED_BARRIER();

            auto b1_keep = b1;
            global_load_b(Bs[tic][1], bc*2+1, k+2);
            TK_WAIT_VMCNT(CRR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            CRR_MMA_BEGIN();
            CRR_STRICT_MMA(
                cD,
                rowreg_alias<CRR_STRICT_A_ROWREG, A_row_reg>(a),
                rowreg_alias<CRR_STRICT_ROWREG_ABT, B_row_reg>(b1_keep),
                cD
            );
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();
#endif
#else
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            global_load_a(As[toc][1], br*2+1, k+1);
            TK_WAIT_LGKM(CRR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            CRR_STRICT_MMA(
                cA,
                rowreg_alias<CRR_STRICT_A_ROWREG, A_row_reg>(a),
                rowreg_alias<CRR_STRICT_ROWREG_ABT, B_row_reg>(b0),
                cA
            );
            CRR_MMA_END();
            auto b0_keep = b0;
            __builtin_amdgcn_s_barrier(); CRR_SCHED_BARRIER();

            load_b(b1, Bs[tic][1], wn);
            global_load_b(Bs[tic][0], bc*2, k+2);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            CRR_STRICT_MMA(
                cB,
                rowreg_alias<CRR_STRICT_A_ROWREG, A_row_reg>(a),
                rowreg_alias<CRR_STRICT_ROWREG_ABT, B_row_reg>(b1),
                cB
            );
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            global_load_a(As[tic][0], br*2, k+2);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            CRR_STRICT_MMA(
                cC,
                rowreg_alias<CRR_STRICT_A_ROWREG, A_row_reg>(a),
                rowreg_alias<CRR_STRICT_ROWREG_ABT, B_row_reg>(b0_keep),
                cC
            );
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier(); CRR_SCHED_BARRIER();

            global_load_b(Bs[tic][1], bc*2+1, k+2);
            TK_WAIT_VMCNT(CRR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            CRR_MMA_BEGIN();
            CRR_STRICT_MMA(
                cD,
                rowreg_alias<CRR_STRICT_A_ROWREG, A_row_reg>(a),
                rowreg_alias<CRR_STRICT_ROWREG_ABT, B_row_reg>(b1),
                cD
            );
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();
#endif
        }

        {
#if CRR_BATCHED_EPILOGUE_MMA
            load_b(b0, Bs[tic][0], wn);
            load_b(b1, Bs[tic][1], wn);
            load_a(a, As[tic][0], wm);
            global_load_a(As[toc][1], br*2+1, KI-1);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            CRR_STRICT_MMA(
                cA,
                rowreg_alias<CRR_STRICT_A_ROWREG, A_row_reg>(a),
                rowreg_alias<CRR_STRICT_ROWREG_ABT, B_row_reg>(b0),
                cA
            );
            CRR_STRICT_MMA(
                cB,
                rowreg_alias<CRR_STRICT_A_ROWREG, A_row_reg>(a),
                rowreg_alias<CRR_STRICT_ROWREG_ABT, B_row_reg>(b1),
                cB
            );
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier(); CRR_SCHED_BARRIER();

            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(CRR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            CRR_STRICT_MMA(
                cC,
                rowreg_alias<CRR_STRICT_A_ROWREG, A_row_reg>(a),
                rowreg_alias<CRR_STRICT_ROWREG_ABT, B_row_reg>(b0),
                cC
            );
            CRR_STRICT_MMA(
                cD,
                rowreg_alias<CRR_STRICT_A_ROWREG, A_row_reg>(a),
                rowreg_alias<CRR_STRICT_ROWREG_ABT, B_row_reg>(b1),
                cD
            );
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[toc][0], wn);
            tic ^= 1; toc ^= 1;
#else
#if CRR_A_LDS_REENCODE
            load_b(b0, Bs[tic][0], wn);
            load_a(a, wm);
            global_load_a(As[toc][1], br*2+1, KI-1);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_AB(cA, a, b0, cA); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); CRR_SCHED_BARRIER();

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_AB(cB, a, b1, cB); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            reencode_a(As[tic][1]);
            load_a(a, wm);
            TK_WAIT_VMCNT(CRR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_AB(cC, a, b0, cC); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            reencode_a(As[toc][0]);
            load_b(b0, Bs[toc][0], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_AB(cD, a, b1, cD); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); CRR_SCHED_BARRIER();
            tic ^= 1; toc ^= 1;
#else
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            global_load_a(As[toc][1], br*2+1, KI-1);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            CRR_STRICT_MMA(
                cA,
                rowreg_alias<CRR_STRICT_A_ROWREG, A_row_reg>(a),
                rowreg_alias<CRR_STRICT_ROWREG_ABT, B_row_reg>(b0),
                cA
            );
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier(); CRR_SCHED_BARRIER();

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            CRR_STRICT_MMA(
                cB,
                rowreg_alias<CRR_STRICT_A_ROWREG, A_row_reg>(a),
                rowreg_alias<CRR_STRICT_ROWREG_ABT, B_row_reg>(b1),
                cB
            );
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

#if CRR_BATCHED_PAIR_MMA
            load_b(b0, Bs[tic][0], wn);
#endif
            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(CRR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            CRR_STRICT_MMA(
                cC,
                rowreg_alias<CRR_STRICT_A_ROWREG, A_row_reg>(a),
                rowreg_alias<CRR_STRICT_ROWREG_ABT, B_row_reg>(b0),
                cC
            );
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[toc][0], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            CRR_STRICT_MMA(
                cD,
                rowreg_alias<CRR_STRICT_A_ROWREG, A_row_reg>(a),
                rowreg_alias<CRR_STRICT_ROWREG_ABT, B_row_reg>(b1),
                cD
            );
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier(); CRR_SCHED_BARRIER();
            tic ^= 1; toc ^= 1;
#endif
#endif
        }

        {
#if CRR_BATCHED_EPILOGUE_MMA
            load_a(a, As[tic][0], wm);
            load_b(b1, Bs[tic][1], wn);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            CRR_STRICT_MMA(
                cA,
                rowreg_alias<CRR_STRICT_A_ROWREG, A_row_reg>(a),
                rowreg_alias<CRR_STRICT_ROWREG_ABT, B_row_reg>(b0),
                cA
            );
            CRR_STRICT_MMA(
                cB,
                rowreg_alias<CRR_STRICT_A_ROWREG, A_row_reg>(a),
                rowreg_alias<CRR_STRICT_ROWREG_ABT, B_row_reg>(b1),
                cB
            );
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier(); CRR_SCHED_BARRIER();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            CRR_STRICT_MMA(
                cC,
                rowreg_alias<CRR_STRICT_A_ROWREG, A_row_reg>(a),
                rowreg_alias<CRR_STRICT_ROWREG_ABT, B_row_reg>(b0),
                cC
            );
            CRR_STRICT_MMA(
                cD,
                rowreg_alias<CRR_STRICT_A_ROWREG, A_row_reg>(a),
                rowreg_alias<CRR_STRICT_ROWREG_ABT, B_row_reg>(b1),
                cD
            );
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();
#else
#if CRR_A_LDS_REENCODE
            load_a(a, wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_AB(cA, a, b0, cA); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier(); CRR_SCHED_BARRIER();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_AB(cB, a, b1, cB); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            reencode_a(As[tic][1]);
            load_a(a, wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_AB(cC, a, b0, cC);
            mma_AB(cD, a, b1, cD);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
#else
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            CRR_STRICT_MMA(
                cA,
                rowreg_alias<CRR_STRICT_A_ROWREG, A_row_reg>(a),
                rowreg_alias<CRR_STRICT_ROWREG_ABT, B_row_reg>(b0),
                cA
            );
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier(); CRR_SCHED_BARRIER();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            CRR_STRICT_MMA(
                cB,
                rowreg_alias<CRR_STRICT_A_ROWREG, A_row_reg>(a),
                rowreg_alias<CRR_STRICT_ROWREG_ABT, B_row_reg>(b1),
                cB
            );
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            CRR_STRICT_MMA(
                cC,
                rowreg_alias<CRR_STRICT_A_ROWREG, A_row_reg>(a),
                rowreg_alias<CRR_STRICT_ROWREG_ABT, B_row_reg>(b0),
                cC
            );
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            CRR_STRICT_MMA(
                cD,
                rowreg_alias<CRR_STRICT_A_ROWREG, A_row_reg>(a),
                rowreg_alias<CRR_STRICT_ROWREG_ABT, B_row_reg>(b1),
                cD
            );
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();
#endif
        #endif
        }    }

    // Apply the tensorwise descale before the global store so the Python
    // bridge doesn't need to materialize an fp32 post-scale tensor.
    if (g.scale != 1.0f) {
        mul(cA, cA, g.scale);
        mul(cB, cB, g.scale);
        mul(cC, cC, g.scale);
        mul(cD, cD, g.scale);
    }

    // Store Output
    if (wm == 0) __builtin_amdgcn_s_barrier();
    store(g.c, cA, {0, 0, br*WARPS_M*2+wm,         bc*WARPS_N*2+wn});
    store(g.c, cB, {0, 0, br*WARPS_M*2+wm,         bc*WARPS_N*2+WARPS_N+wn});
    store(g.c, cC, {0, 0, br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+wn});
    store(g.c, cD, {0, 0, br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+WARPS_N+wn});
}

template __global__ void gemm_kernel<Layout::RCR>(const layout_globals);
template __global__ void gemm_kernel<Layout::RRR>(const layout_globals);
template __global__ void gemm_kernel<Layout::CRR>(const layout_globals);

template<Layout L>
void dispatch(layout_globals g) {
    gemm_kernel<L><<<g.grid(), g.block(), 0, g.stream>>>(g);
}

PYBIND11_MODULE(TK_FP8_LAYOUTS_MODULE_NAME, m) {
    m.doc() = "FP8 GEMM: RCR(mma_ABt), RRR(col_l+mma_AB), CRR(col_l+mma_AtB)";
    py::bind_function<dispatch<Layout::RCR>>(m, "gemm_rcr",
        &layout_globals::a, &layout_globals::b, &layout_globals::c);
    py::bind_function<dispatch<Layout::RCR>>(m, "gemm_rcr",
        &layout_globals::a, &layout_globals::b, &layout_globals::c, &layout_globals::scale);
    py::bind_function<dispatch<Layout::RRR>>(m, "gemm_rrr",
        &layout_globals::a, &layout_globals::b, &layout_globals::c);
    py::bind_function<dispatch<Layout::RRR>>(m, "gemm_rrr",
        &layout_globals::a, &layout_globals::b, &layout_globals::c, &layout_globals::scale);
    py::bind_function<dispatch<Layout::CRR>>(m, "gemm_crr",
        &layout_globals::a, &layout_globals::b, &layout_globals::c);
    py::bind_function<dispatch<Layout::CRR>>(m, "gemm_crr",
        &layout_globals::a, &layout_globals::b, &layout_globals::c, &layout_globals::scale);
}
