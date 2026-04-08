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

#define TK_STRINGIFY_IMPL(x) #x
#define TK_STRINGIFY(x) TK_STRINGIFY_IMPL(x)
#define TK_WAIT_LGKM(x) asm volatile("s_waitcnt lgkmcnt(" TK_STRINGIFY(x) ")")
#define TK_WAIT_VMCNT(x) asm volatile("s_waitcnt vmcnt(" TK_STRINGIFY(x) ")")
#define TK_PRAGMA_UNROLL(x) _Pragma(TK_STRINGIFY(unroll x))

#ifndef RCR_PREFETCH_LGKM
#define RCR_PREFETCH_LGKM 4
#endif
#ifndef RCR_INIT0_VMCNT
#define RCR_INIT0_VMCNT 4
#endif
#ifndef RCR_INIT1_VMCNT
#define RCR_INIT1_VMCNT 6
#endif
#ifndef RCR_STEADY_VMCNT
#define RCR_STEADY_VMCNT 4
#endif
#ifndef RCR_EPILOGUE_VMCNT
#define RCR_EPILOGUE_VMCNT 4
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
#define RRR_STEADY_VMCNT 4
#endif
#ifndef RRR_EPILOGUE_VMCNT
#define RRR_EPILOGUE_VMCNT 2
#endif
#ifndef CRR_PREFETCH_LGKM
#define CRR_PREFETCH_LGKM 3
#endif

#ifndef RCR_MAIN_UNROLL
#define RCR_MAIN_UNROLL 2
#endif
#ifndef RCR_BATCHED_READS
#define RCR_BATCHED_READS 0
#endif
#ifndef RCR_BATCHED_PAIR_MMA
#define RCR_BATCHED_PAIR_MMA 0
#endif
#ifndef RCR_BATCHED_EPILOGUE_MMA
#define RCR_BATCHED_EPILOGUE_MMA 0
#endif
#ifndef RCR_TWO_TILE_SCHEDULE
#define RCR_TWO_TILE_SCHEDULE 0
#endif
#ifndef RCR_SINGLE_STAGE
#define RCR_SINGLE_STAGE 0
#endif
#ifndef RCR_SINGLE_STAGE_INIT_VMCNT
#define RCR_SINGLE_STAGE_INIT_VMCNT 0
#endif
#ifndef RCR_SINGLE_STAGE_STEADY_VMCNT
#define RCR_SINGLE_STAGE_STEADY_VMCNT 0
#endif
#ifndef RCR_PAIR_B1_TOP_WAIT
#define RCR_PAIR_B1_TOP_WAIT 0
#endif
#ifndef RCR_PAIR_B1_TOP_VMCNT
#define RCR_PAIR_B1_TOP_VMCNT 0
#endif
#ifndef RCR_PAIR_INNER_SCHED_BARRIER
#define RCR_PAIR_INNER_SCHED_BARRIER 0
#endif
#ifndef RCR_PAIR_INNER_WG_BARRIER
#define RCR_PAIR_INNER_WG_BARRIER 0
#endif
#ifndef RCR_PAIR_KEEP_B_REGS
#define RCR_PAIR_KEEP_B_REGS 0
#endif
#ifndef RCR_PAIR_REVERSE_MMA_ORDER
#define RCR_PAIR_REVERSE_MMA_ORDER 0
#endif
#ifndef RCR_PAIR_POST_B1_WAIT
#define RCR_PAIR_POST_B1_WAIT 0
#endif
#ifndef RCR_B_REG_COL_LOAD
#define RCR_B_REG_COL_LOAD 0
#endif
#ifndef RCR_B_REG_MMA_ALIAS
#define RCR_B_REG_MMA_ALIAS 0
#endif
#ifndef RCR_B1_NOINLINE_MMA
#define RCR_B1_NOINLINE_MMA 0
#endif
#ifndef RCR_SWAP_STORE_HALVES
#define RCR_SWAP_STORE_HALVES 0
#endif
#ifndef RCR_PAIR_SWAP_B_LOADS
#define RCR_PAIR_SWAP_B_LOADS 0
#endif
#ifndef RCR_PAIR_SERIALIZE_FUTURE_B_LOADS
#define RCR_PAIR_SERIALIZE_FUTURE_B_LOADS 0
#endif
#ifndef RCR_PAIR_FUTURE_B_GAP_VMCNT
#define RCR_PAIR_FUTURE_B_GAP_VMCNT 0
#endif
#ifndef RCR_PAIR_SWAP_FUTURE_B_ORDER
#define RCR_PAIR_SWAP_FUTURE_B_ORDER 0
#endif
#ifndef RCR_PAIR_LATE_B1_PREFETCH
#define RCR_PAIR_LATE_B1_PREFETCH 0
#endif
#ifndef RCR_PAIR_SPLIT_CD
#define RCR_PAIR_SPLIT_CD 0
#endif
#ifndef RCR_PAIR_DUP_RHS_REGS
#define RCR_PAIR_DUP_RHS_REGS 0
#endif
#ifndef RCR_PAIR_NOINLINE_RHS_COPY
#define RCR_PAIR_NOINLINE_RHS_COPY 0
#endif
#ifndef RCR_PAIR_SPLIT_CB
#define RCR_PAIR_SPLIT_CB 0
#endif
#ifndef RCR_PAIR_CB_RELOAD_LAST_WN
#define RCR_PAIR_CB_RELOAD_LAST_WN 0
#endif
#ifndef RCR_PAIR_CB_NOINLINE_LAST_WN
#define RCR_PAIR_CB_NOINLINE_LAST_WN 0
#endif
#ifndef RCR_PAIR_CB_ALT_LOAD_LAST_WN
#define RCR_PAIR_CB_ALT_LOAD_LAST_WN 0
#endif
#ifndef RCR_SWAP_B_BUFFER_ASSIGN
#define RCR_SWAP_B_BUFFER_ASSIGN 0
#endif
#ifndef RCR_USE_V2_SHARED
#define RCR_USE_V2_SHARED 1
#endif
#ifndef RCR_USE_V2A_SHARED
#define RCR_USE_V2A_SHARED 0
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
#define CRR_INIT0_VMCNT 2
#endif
#ifndef CRR_INIT1_VMCNT
#define CRR_INIT1_VMCNT 6
#endif
#ifndef CRR_STEADY_VMCNT
#define CRR_STEADY_VMCNT 4
#endif
#ifndef CRR_EPILOGUE_VMCNT
#define CRR_EPILOGUE_VMCNT 2
#endif
#ifndef CRR_BATCHED_PAIR_MMA
#define CRR_BATCHED_PAIR_MMA 1
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
#ifndef RRR_ROW_SHARED_TRANSPOSE
#define RRR_ROW_SHARED_TRANSPOSE 0
#endif
#ifndef RRR_B_REG_ROW_LOAD_TRANSPOSE
#define RRR_B_REG_ROW_LOAD_TRANSPOSE 0
#endif
#ifndef RRR_B_REG_ROW_LOAD_ALIAS
#define RRR_B_REG_ROW_LOAD_ALIAS 0
#endif
#ifndef CRR_USE_V3_SWIZZLE
#define CRR_USE_V3_SWIZZLE 0
#endif
#ifndef CRR_A_REG_ROW_LOAD_TRANSPOSE
#define CRR_A_REG_ROW_LOAD_TRANSPOSE 1
#endif
#ifndef CRR_B_REG_ROW_LOAD_TRANSPOSE
#define CRR_B_REG_ROW_LOAD_TRANSPOSE 0
#endif
#ifndef CRR_A_REG_ROW_LOAD_ALIAS
#define CRR_A_REG_ROW_LOAD_ALIAS 1
#endif
#ifndef CRR_B_REG_ROW_LOAD_ALIAS
#define CRR_B_REG_ROW_LOAD_ALIAS 0
#endif
#ifndef RRR_USE_V2A_SWIZZLE
#define RRR_USE_V2A_SWIZZLE 0
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
#define RRR_ENABLE_SCHED_BARRIER 1
#endif
#if RRR_ENABLE_SCHED_BARRIER
#define RRR_SCHED_BARRIER() __builtin_amdgcn_sched_barrier(0)
#else
#define RRR_SCHED_BARRIER() do {} while (0)
#endif

#ifndef RCR_ENABLE_SCHED_BARRIER
#define RCR_ENABLE_SCHED_BARRIER 1
#endif
#if RCR_ENABLE_SCHED_BARRIER
#define RCR_SCHED_BARRIER() __builtin_amdgcn_sched_barrier(0)
#else
#define RCR_SCHED_BARRIER() do {} while (0)
#endif
#if RCR_PAIR_INNER_SCHED_BARRIER
#define RCR_PAIR_INNER_BARRIER() RCR_SCHED_BARRIER()
#else
#define RCR_PAIR_INNER_BARRIER() do {} while (0)
#endif
#if RCR_PAIR_INNER_WG_BARRIER
#define RCR_PAIR_INNER_WG_SYNC() do { __builtin_amdgcn_s_barrier(); asm volatile("s_waitcnt lgkmcnt(0)"); } while (0)
#else
#define RCR_PAIR_INNER_WG_SYNC() do {} while (0)
#endif

#define CRR_MMA_BEGIN() do { CRR_SCHED_BARRIER(); __builtin_amdgcn_s_setprio(1); } while (0)
#define CRR_MMA_END() do { __builtin_amdgcn_s_setprio(0); CRR_SCHED_BARRIER(); } while (0)

#if CRR_A_LDS_REENCODE && CRR_ROW_SHARED_TRANSPOSE
#error "CRR_A_LDS_REENCODE is only valid on the strict non-row-shared path"
#endif

#if CRR_A_LDS_REENCODE && CRR_BATCHED_EPILOGUE_MMA
#error "CRR_A_LDS_REENCODE currently requires CRR_BATCHED_EPILOGUE_MMA=0"
#endif

#if CRR_USE_V3_SWIZZLE && CRR_A_LDS_REENCODE
#error "CRR_USE_V3_SWIZZLE is only supported on the strict non-reencode path"
#endif

#if (CRR_A_REG_ROW_LOAD_TRANSPOSE || CRR_B_REG_ROW_LOAD_TRANSPOSE) && (CRR_ROW_SHARED_TRANSPOSE || CRR_A_LDS_REENCODE || CRR_USE_V3_SWIZZLE)
#error "CRR_*_REG_ROW_LOAD_TRANSPOSE only supports the strict v2/v2a non-reencode path"
#endif
#if CRR_B_REG_ROW_LOAD_ALIAS && !CRR_A_REG_ROW_LOAD_ALIAS
#error "CRR_B_REG_ROW_LOAD_ALIAS requires CRR_A_REG_ROW_LOAD_ALIAS"
#endif

#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_ROW_SHARED_TRANSPOSE
#error "RRR_B_REG_ROW_LOAD_TRANSPOSE only supports the strict non-row-shared path"
#endif

#ifndef GEMM_BLOCK_SIZE
#define GEMM_BLOCK_SIZE 256
#endif
#ifndef GEMM_K_BLOCK
#define GEMM_K_BLOCK 128
#endif
#ifndef GEMM_WARPS_M
#define GEMM_WARPS_M 2
#endif
#ifndef GEMM_WARPS_N
#define GEMM_WARPS_N 4
#endif

constexpr int BLK = GEMM_BLOCK_SIZE, BK = GEMM_K_BLOCK;
constexpr int HB  = BLK / 2;
constexpr int WARPS_M = GEMM_WARPS_M, WARPS_N = GEMM_WARPS_N;
constexpr int _NUM_WARPS   = WARPS_M * WARPS_N;
constexpr int _NUM_THREADS = _NUM_WARPS * WARP_THREADS;
constexpr int RBM = BLK / WARPS_M / 2;   // 64
constexpr int RBN = BLK / WARPS_N / 2;   // 32
constexpr int TAIL_BLOCK_M = 16;
constexpr int TAIL_BLOCK_N = 16;

#ifndef GEMM_MIN_BLOCKS_PER_CU
#define GEMM_MIN_BLOCKS_PER_CU 2
#endif
#ifndef GEMM_BLOCK_SWIZZLE
#define GEMM_BLOCK_SWIZZLE 0
#endif
#ifndef GEMM_BLOCK_SWIZZLE_NUM_XCDS
#define GEMM_BLOCK_SWIZZLE_NUM_XCDS 8
#endif
#ifndef GEMM_BLOCK_SWIZZLE_GROUP_M
#define GEMM_BLOCK_SWIZZLE_GROUP_M 4
#endif

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
#if RCR_B_REG_COL_LOAD && !RCR_B_REG_MMA_ALIAS
using RCR_B_reg = B_col_reg;
#else
using RCR_B_reg = B_row_reg;
#endif

using ST_v2  = st_fp8e4m3<HB, BK, st_16x128_v2_s>;
using ST_v2a = st_fp8e4m3<HB, BK, st_16x128_v2a_s>;
using ST_v3  = st_fp8e4m3<HB, BK, st_16x128_v3_s>;
#if RRR_USE_V2A_SWIZZLE
using ST_rrr_b = ST_v2a;
#else
using ST_rrr_b = ST_v2;
#endif

static_assert(sizeof(A_row_reg) == sizeof(A_col_reg));
static_assert(alignof(A_row_reg) == alignof(A_col_reg));
static_assert(sizeof(B_row_reg) == sizeof(B_col_reg));
static_assert(alignof(B_row_reg) == alignof(B_col_reg));

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
__device__ __forceinline__ void load_col_from_v2a_st(
    RT& dst, const ST& tile, int col_start)
{
    load_col_from_v2a_st_half<RT, 0>(dst, tile, col_start);
    load_col_from_v2a_st_half<RT, 1>(dst, tile, col_start);
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
            : "=&v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx])),
              "=&v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx + 2]))
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

__device__ __forceinline__ void rrr_mma(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
    const A_row_reg& a,
    const B_col_reg& b)
{
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
    const auto& b_row = reinterpret_cast<const B_row_reg&>(b);
    mma_ABt(acc, a, b_row, acc);
#else
    mma_AB(acc, a, b, acc);
#endif
}

__device__ __forceinline__ void rcr_mma(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
    const A_row_reg& a,
    const RCR_B_reg& b)
{
#if RCR_B_REG_COL_LOAD
#if RCR_B_REG_MMA_ALIAS
    const auto& b_col = reinterpret_cast<const B_col_reg&>(b);
    mma_AB(acc, a, b_col, acc);
#else
    mma_AB(acc, a, b, acc);
#endif
#else
    mma_ABt(acc, a, b, acc);
#endif
}

__device__ __noinline__ void rcr_rhs_mma(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
    const A_row_reg& a,
    const RCR_B_reg& b)
{
    rcr_mma(acc, a, b);
}

__device__ __noinline__ RCR_B_reg rcr_rhs_copy(const RCR_B_reg b)
{
    return b;
}

#if RCR_B1_NOINLINE_MMA
#define RCR_RHS_MMA(acc, a, b) rcr_rhs_mma(acc, a, b)
#else
#define RCR_RHS_MMA(acc, a, b) rcr_mma(acc, a, b)
#endif

#if RCR_PAIR_CB_NOINLINE_LAST_WN
#define RCR_PAIR_CB_MMA(acc, a, b, wn) do { \
    if ((wn) == WARPS_N - 1) rcr_rhs_mma((acc), (a), (b)); \
    else rcr_mma((acc), (a), (b)); \
} while (0)
#else
#define RCR_PAIR_CB_MMA(acc, a, b, wn) RCR_RHS_MMA((acc), (a), (b))
#endif

__device__ __forceinline__ void crr_mma(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
    const A_col_reg& a,
    const B_col_reg& b)
{
#if CRR_A_REG_ROW_LOAD_TRANSPOSE && CRR_A_REG_ROW_LOAD_ALIAS
    const auto& a_row = reinterpret_cast<const A_row_reg&>(a);
    #if CRR_B_REG_ROW_LOAD_TRANSPOSE && CRR_B_REG_ROW_LOAD_ALIAS
    const auto& b_row = reinterpret_cast<const B_row_reg&>(b);
    mma_ABt(acc, a_row, b_row, acc);
    #else
    mma_AB(acc, a_row, b, acc);
    #endif
#else
    mma_AtB(acc, a, b, acc);
#endif
}

__device__ __forceinline__ float load_fp8_scalar(const _gl_fp8& src, int row, int col) {
    return base_types::convertor<float, fp8e4m3>::convert(src[coord<>(row, col)]);
}

__device__ __forceinline__ float load_bf16_scalar(const _gl_bf16& src, int row, int col) {
    return base_types::convertor<float, bf16>::convert(src[coord<>(row, col)]);
}

__device__ __forceinline__ void store_bf16_scalar(const _gl_bf16& dst, int row, int col, float value) {
    dst[coord<>(row, col)] = base_types::convertor<bf16, float>::convert(value);
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

struct layout_globals {
    _gl_fp8 a, b;
    _gl_bf16 c;
    float scale_a, scale_b;
    hipStream_t stream;
    int m, n, k;
    int bpr, bpc, ki;
    int fast_m, fast_n, fast_k;
    int group_m;
    dim3 grid()  { return dim3(bpr * bpc); }
    dim3 block() { return dim3(_NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

__device__ __forceinline__ int gemm_chiplet_swizzle_bid(int bid, int num_wgs) {
#if GEMM_BLOCK_SWIZZLE
    if (num_wgs >= GEMM_BLOCK_SWIZZLE_NUM_XCDS &&
        (num_wgs % GEMM_BLOCK_SWIZZLE_NUM_XCDS) == 0) {
        return
            (bid % GEMM_BLOCK_SWIZZLE_NUM_XCDS) *
                (num_wgs / GEMM_BLOCK_SWIZZLE_NUM_XCDS) +
            (bid / GEMM_BLOCK_SWIZZLE_NUM_XCDS);
    }
#endif
    return bid;
}

__device__ __forceinline__ void gemm_compute_block_coords(
    int bid, int bpr, int bpc, int group_m, int &br, int &bc) {
#if GEMM_BLOCK_SWIZZLE
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
#else
    br = bid / bpc;
    bc = bid % bpc;
#endif
}

#include "rcr_exact_4wave_fastpath.inc"
#include "rcr_exact_8wave_fastpath.inc"
#include "rrr_exact_8wave_fastpath.inc"
#include "crr_exact_4wave_fastpath.inc"
#include "crr_exact_8wave_double_pump_fastpath.inc"
#include "crr_exact_8wave_fastpath.inc"

template<Layout L>
__global__ __launch_bounds__(_NUM_THREADS, GEMM_MIN_BLOCKS_PER_CU)
void gemm_kernel(const layout_globals g) {
    int bid = blockIdx.x;
    int br, bc;
    gemm_compute_block_coords(bid, g.bpr, g.bpc, g.group_m, br, bc);
    if (br >= g.bpr || bc >= g.bpc || g.ki <= 0) {
        return;
    }
    int wm = warpid() / WARPS_N, wn = warpid() % WARPS_N;

    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;
    zero(cA); zero(cB); zero(cC); zero(cD);

    if constexpr (L == Layout::RCR) {
#if RCR_USE_V2A_SHARED
        using ST_rcr = ST_v2a;
#elif RCR_USE_V2_SHARED
        using ST_rcr = ST_v2;
#else
        using ST_rcr = ST_row;
#endif
#if RCR_SINGLE_STAGE
        __shared__ ST_rcr As[2];
        __shared__ ST_rcr Bs[2];
        A_row_reg a;
        B_row_reg b0, b1;

        constexpr int bpt = ST_rcr::underlying_subtile_bytes_per_thread;
        constexpr int bpm = bpt * _NUM_THREADS;
        constexpr int mpt = ST_rcr::rows * ST_rcr::cols * sizeof(fp8e4m3) / bpm;
        uint32_t soA[mpt], soB[mpt];
        G::prefill_swizzled_offsets(As[0], g.a, soA);
        G::prefill_swizzled_offsets(Bs[0], g.b, soB);

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

        G::load(Bs[0], g.b, b_co(bc*2,   0), soB);
        G::load(As[0], g.a, a_co(br*2,   0), soA);
        G::load(Bs[1], g.b, b_co(bc*2+1, 0), soB);
        G::load(As[1], g.a, a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(RCR_SINGLE_STAGE_INIT_VMCNT);
        __builtin_amdgcn_s_barrier();

        for (int k = 0; k < g.ki; ++k) {
            load_b(b0, Bs[0], wn);
            load_a(a, As[0], wm);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cA, a, b0, cA); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

            load_b(b1, Bs[1], wn);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cB, a, b1, cB); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[1], wm);
            if (k + 1 < g.ki) {
                G::load(Bs[0], g.b, b_co(bc*2,   k+1), soB);
                G::load(As[0], g.a, a_co(br*2,   k+1), soA);
            }
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cC, a, b0, cC); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

            if (k + 1 < g.ki) {
                G::load(Bs[1], g.b, b_co(bc*2+1, k+1), soB);
                G::load(As[1], g.a, a_co(br*2+1, k+1), soA);
                TK_WAIT_VMCNT(RCR_SINGLE_STAGE_STEADY_VMCNT);
            }
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cD, a, b1, cD); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }
#else
        __shared__ ST_rcr As[2][2];
        __shared__ ST_rcr Bs[2][2];
        A_row_reg a;
        RCR_B_reg b0, b1;

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
        auto load_b = [&](RCR_B_reg& dst, ST_rcr& tile, int wi) {
#if RCR_B_REG_COL_LOAD && !RCR_B_REG_MMA_ALIAS
#if RCR_USE_V2A_SHARED
            load_col_from_v2a_st(dst, tile, wi * RBN);
#elif RCR_USE_V2_SHARED
            load_col_from_v2_st(dst, tile, wi * RBN);
#else
            B_row_reg tmp;
            auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
            load(tmp, sub);
            transpose(dst, tmp);
#endif
#else
            auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
            load(dst, sub);
#endif
        };

        constexpr int rcr_b0_slot = RCR_SWAP_B_BUFFER_ASSIGN ? 1 : 0;
        constexpr int rcr_b1_slot = RCR_SWAP_B_BUFFER_ASSIGN ? 0 : 1;
        auto b_tile = [&](int stage, int which) -> ST_rcr& {
            return Bs[stage][which == 0 ? rcr_b0_slot : rcr_b1_slot];
        };

        int tic = 0, toc = 1;
        G::load(b_tile(tic, 0), g.b, b_co(bc*2,   0), soB);
        G::load(As[tic][0], g.a, a_co(br*2,   0), soA);
        G::load(b_tile(tic, 1), g.b, b_co(bc*2+1, 0), soB);
        G::load(As[tic][1], g.a, a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(RCR_INIT0_VMCNT);
        __builtin_amdgcn_s_barrier();

        G::load(b_tile(toc, 0), g.b, b_co(bc*2,   1), soB);
        G::load(As[toc][0], g.a, a_co(br*2,   1), soA);
        G::load(b_tile(toc, 1), g.b, b_co(bc*2+1, 1), soB);

        TK_WAIT_VMCNT(RCR_INIT1_VMCNT);
        __builtin_amdgcn_s_barrier();

        #if RCR_TWO_TILE_SCHEDULE
        if ((g.ki & 1) == 0) {
            auto main_loop_iter = [&](int tile) {
                load_b(b0, Bs[0][0], wn);
                load_a(a, As[0][0], wm);
                G::load(As[1][1], g.a, a_co(br*2+1, tile+1), soA);
                TK_WAIT_LGKM(RCR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();

                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1); mma_ABt(cA, a, b0, cA); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

                load_b(b1, Bs[0][1], wn);
                G::load(Bs[0][0], g.b, b_co(bc*2, tile+2), soB);
                __builtin_amdgcn_s_barrier();

                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1); mma_ABt(cB, a, b1, cB); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();

                load_a(a, As[0][1], wm);
                G::load(As[0][0], g.a, a_co(br*2, tile+2), soA);
                __builtin_amdgcn_s_barrier();

                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1); mma_ABt(cC, a, b0, cC); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

                load_b(b0, Bs[1][0], wn);
                G::load(Bs[0][1], g.b, b_co(bc*2+1, tile+2), soB);
                asm volatile("s_waitcnt vmcnt(6)"); __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_s_setprio(1); mma_ABt(cD, a, b1, cD); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();

                load_a(a, As[1][0], wm);
                G::load(As[0][1], g.a, a_co(br*2+1, tile+2), soA);
                TK_WAIT_LGKM(RCR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();

                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1); mma_ABt(cA, a, b0, cA); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

                load_b(b1, Bs[1][1], wn);
                G::load(Bs[1][0], g.b, b_co(bc*2, tile+3), soB);
                __builtin_amdgcn_s_barrier();

                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1); mma_ABt(cB, a, b1, cB); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();

                load_a(a, As[1][1], wm);
                G::load(As[1][0], g.a, a_co(br*2, tile+3), soA);
                __builtin_amdgcn_s_barrier();

                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1); mma_ABt(cC, a, b0, cC); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

                G::load(Bs[1][1], g.b, b_co(bc*2+1, tile+3), soB);
                asm volatile("s_waitcnt vmcnt(6)"); __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_s_setprio(1); mma_ABt(cD, a, b1, cD); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();
            };

            TK_PRAGMA_UNROLL(RCR_MAIN_UNROLL)
            for (int tile = 0; tile < g.ki - 2; tile += 2) {
                main_loop_iter(tile);
            }

            {
                const int tile = g.ki - 2;
                load_b(b0, Bs[tic][0], wn);
                load_a(a, As[tic][0], wm);
                G::load(As[toc][1], g.a, a_co(br*2+1, tile+1), soA);
                __builtin_amdgcn_s_barrier();
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1); mma_ABt(cA, a, b0, cA); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();

                load_b(b1, Bs[tic][1], wn);
                __builtin_amdgcn_s_barrier();
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1); mma_ABt(cB, a, b1, cB); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();

                load_a(a, As[tic][1], wm);
                asm volatile("s_waitcnt vmcnt(4)"); __builtin_amdgcn_s_barrier();
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1);
                mma_ABt(cC, a, b0, cC);
                mma_ABt(cD, a, b1, cD);
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();
                tic ^= 1; toc ^= 1;
            }

            {
                load_b(b0, Bs[tic][0], wn);
                load_a(a, As[tic][0], wm);
                asm volatile("s_waitcnt vmcnt(2)"); __builtin_amdgcn_s_barrier();
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1); mma_ABt(cA, a, b0, cA); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();

                load_b(b1, Bs[tic][1], wn);
                asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
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
        } else
        #endif
        {
        TK_PRAGMA_UNROLL(RCR_MAIN_UNROLL)
        for (int k = 0; k < g.ki - 2; k++, tic ^= 1, toc ^= 1) {
#if RCR_BATCHED_PAIR_MMA
#if RCR_PAIR_SWAP_B_LOADS
            load_b(b0, b_tile(tic, 1), wn);
#else
            load_b(b0, b_tile(tic, 0), wn);
#endif
            load_a(a, As[tic][0], wm);
#if RCR_PAIR_B1_TOP_WAIT
            if (k >= 2) {
                TK_WAIT_VMCNT(RCR_PAIR_B1_TOP_VMCNT); __builtin_amdgcn_s_barrier();
            }
#endif
#if RCR_PAIR_SWAP_B_LOADS
            load_b(b1, b_tile(tic, 0), wn);
#else
            load_b(b1, b_tile(tic, 1), wn);
#endif
#if RCR_PAIR_POST_B1_WAIT
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
#endif
#if RCR_PAIR_KEEP_B_REGS
            const auto b0_pair = b0;
            const auto b1_pair = b1;
#endif
#if RCR_PAIR_NOINLINE_RHS_COPY
            const auto b1_pair_cB = rcr_rhs_copy(b1);
            const auto b1_pair_cD = rcr_rhs_copy(b1);
#elif RCR_PAIR_DUP_RHS_REGS
            const auto b1_pair_cB = b1;
            const auto b1_pair_cD = b1;
#endif
            G::load(As[toc][1], g.a, a_co(br*2+1, k+1), soA);
            TK_WAIT_LGKM(RCR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
#if RCR_PAIR_CB_RELOAD_LAST_WN
#if !RCR_PAIR_KEEP_B_REGS && !RCR_PAIR_DUP_RHS_REGS && !RCR_PAIR_NOINLINE_RHS_COPY
            if (wn == WARPS_N - 1) {
                load_b(b1, b_tile(tic, 1), wn);
                asm volatile("s_waitcnt lgkmcnt(0)");
            }
#endif
#endif
            __builtin_amdgcn_s_setprio(1);
#if RCR_PAIR_SPLIT_CB
#if RCR_PAIR_KEEP_B_REGS
            rcr_mma(cA, a, b0_pair);
#else
            rcr_mma(cA, a, b0);
#endif
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

            load_b(b1, b_tile(tic, 1), wn);
            G::load(b_tile(tic, 0), g.b, b_co(bc*2, k+2), soB);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
#if RCR_PAIR_KEEP_B_REGS
            RCR_PAIR_CB_MMA(cB, a, b1_pair, wn);
#elif RCR_PAIR_NOINLINE_RHS_COPY || RCR_PAIR_DUP_RHS_REGS
            RCR_PAIR_CB_MMA(cB, a, b1_pair_cB, wn);
#else
            RCR_PAIR_CB_MMA(cB, a, b1, wn);
#endif
#else
#if RCR_PAIR_REVERSE_MMA_ORDER
#if RCR_PAIR_KEEP_B_REGS
            RCR_PAIR_CB_MMA(cB, a, b1_pair, wn);
            RCR_PAIR_INNER_BARRIER();
            RCR_PAIR_INNER_WG_SYNC();
            rcr_mma(cA, a, b0_pair);
#else
#if RCR_PAIR_DUP_RHS_REGS
            RCR_PAIR_CB_MMA(cB, a, b1_pair_cB, wn);
            RCR_PAIR_INNER_BARRIER();
            RCR_PAIR_INNER_WG_SYNC();
            rcr_mma(cA, a, b0);
#else
#if RCR_PAIR_CB_ALT_LOAD_LAST_WN
            if (wn == WARPS_N - 1) {
                RCR_B_reg b1_lane;
                load_b(b1_lane, b_tile(tic, 1), wn);
                asm volatile("s_waitcnt lgkmcnt(0)");
                RCR_RHS_MMA(cB, a, b1_lane);
            } else {
                RCR_PAIR_CB_MMA(cB, a, b1, wn);
            }
#else
            RCR_PAIR_CB_MMA(cB, a, b1, wn);
#endif
            RCR_PAIR_INNER_BARRIER();
            RCR_PAIR_INNER_WG_SYNC();
            rcr_mma(cA, a, b0);
#endif
#endif
#else
#if RCR_PAIR_KEEP_B_REGS
            rcr_mma(cA, a, b0_pair);
            RCR_PAIR_INNER_BARRIER();
            RCR_PAIR_INNER_WG_SYNC();
            RCR_PAIR_CB_MMA(cB, a, b1_pair, wn);
#else
            rcr_mma(cA, a, b0);
            RCR_PAIR_INNER_BARRIER();
            RCR_PAIR_INNER_WG_SYNC();
#if RCR_PAIR_DUP_RHS_REGS
            RCR_PAIR_CB_MMA(cB, a, b1_pair_cB, wn);
#else
#if RCR_PAIR_CB_ALT_LOAD_LAST_WN
            if (wn == WARPS_N - 1) {
                RCR_B_reg b1_lane;
                load_b(b1_lane, b_tile(tic, 1), wn);
                asm volatile("s_waitcnt lgkmcnt(0)");
                RCR_RHS_MMA(cB, a, b1_lane);
            } else {
                RCR_PAIR_CB_MMA(cB, a, b1, wn);
            }
#else
            RCR_PAIR_CB_MMA(cB, a, b1, wn);
#endif
#endif
#endif
#endif
#endif
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

#if RCR_PAIR_SPLIT_CB
#if !RCR_PAIR_LATE_B1_PREFETCH
            G::load(b_tile(tic, 1), g.b, b_co(bc*2+1, k+2), soB);
#endif
#else
        #if RCR_PAIR_SWAP_FUTURE_B_ORDER
        #if !RCR_PAIR_LATE_B1_PREFETCH && !RCR_PAIR_SPLIT_CD
            G::load(b_tile(tic, 1), g.b, b_co(bc*2+1, k+2), soB);
        #endif
        #if RCR_PAIR_SERIALIZE_FUTURE_B_LOADS
            TK_WAIT_VMCNT(RCR_PAIR_FUTURE_B_GAP_VMCNT); __builtin_amdgcn_s_barrier();
        #endif
            G::load(b_tile(tic, 0), g.b, b_co(bc*2, k+2), soB);
        #else
            G::load(b_tile(tic, 0), g.b, b_co(bc*2, k+2), soB);
        #if RCR_PAIR_SERIALIZE_FUTURE_B_LOADS
            TK_WAIT_VMCNT(RCR_PAIR_FUTURE_B_GAP_VMCNT); __builtin_amdgcn_s_barrier();
        #endif
        #if !RCR_PAIR_LATE_B1_PREFETCH && !RCR_PAIR_SPLIT_CD
            G::load(b_tile(tic, 1), g.b, b_co(bc*2+1, k+2), soB);
        #endif
        #endif
#endif
            load_a(a, As[tic][1], wm);
            G::load(As[tic][0], g.a, a_co(br*2, k+2), soA);
            TK_WAIT_VMCNT(RCR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
#if RCR_PAIR_SPLIT_CD
#if RCR_PAIR_KEEP_B_REGS
            rcr_mma(cC, a, b0_pair);
#else
            rcr_mma(cC, a, b0);
#endif
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();
            G::load(b_tile(tic, 1), g.b, b_co(bc*2+1, k+2), soB);
            TK_WAIT_VMCNT(RCR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
#if RCR_PAIR_KEEP_B_REGS
            RCR_RHS_MMA(cD, a, b1_pair);
#else
            RCR_RHS_MMA(cD, a, b1);
#endif
#else
#if RCR_PAIR_REVERSE_MMA_ORDER
#if RCR_PAIR_KEEP_B_REGS
            RCR_RHS_MMA(cD, a, b1_pair);
            RCR_PAIR_INNER_BARRIER();
            rcr_mma(cC, a, b0_pair);
#else
#if RCR_PAIR_DUP_RHS_REGS
            RCR_RHS_MMA(cD, a, b1_pair_cD);
            RCR_PAIR_INNER_BARRIER();
            rcr_mma(cC, a, b0);
#else
            RCR_RHS_MMA(cD, a, b1);
            RCR_PAIR_INNER_BARRIER();
            rcr_mma(cC, a, b0);
#endif
#endif
#else
#if RCR_PAIR_KEEP_B_REGS
            rcr_mma(cC, a, b0_pair);
            RCR_PAIR_INNER_BARRIER();
            RCR_RHS_MMA(cD, a, b1_pair);
#else
            rcr_mma(cC, a, b0);
            RCR_PAIR_INNER_BARRIER();
#if RCR_PAIR_DUP_RHS_REGS
            RCR_RHS_MMA(cD, a, b1_pair_cD);
#else
            RCR_RHS_MMA(cD, a, b1);
#endif
#endif
#endif
#endif
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();
#if RCR_PAIR_LATE_B1_PREFETCH
            G::load(b_tile(tic, 1), g.b, b_co(bc*2+1, k+2), soB);
#endif
#elif RCR_BATCHED_READS
            A_row_reg a1;
            load_b(b0, b_tile(tic, 0), wn);
            load_b(b1, b_tile(tic, 1), wn);
            load_a(a, As[tic][0], wm);
            load_a(a1, As[tic][1], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, k+1), soA);
            TK_WAIT_LGKM(RCR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();

            G::load(b_tile(tic, 0), g.b, b_co(bc*2, k+2), soB);
            G::load(As[tic][0], g.a, a_co(br*2, k+2), soA);
            G::load(b_tile(tic, 1), g.b, b_co(bc*2+1, k+2), soB);

            __builtin_amdgcn_s_setprio(1);
            rcr_mma(cA, a, b0);
            RCR_SCHED_BARRIER();
            RCR_RHS_MMA(cB, a, b1);
            rcr_mma(cC, a1, b0);
            RCR_SCHED_BARRIER();
            RCR_RHS_MMA(cD, a1, b1);
            __builtin_amdgcn_s_setprio(0);

            TK_WAIT_VMCNT(RCR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
#else
            load_b(b0, b_tile(tic, 0), wn);
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, k+1), soA);
            TK_WAIT_LGKM(RCR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

            load_b(b1, b_tile(tic, 1), wn);
            G::load(b_tile(tic, 0), g.b, b_co(bc*2, k+2), soB);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); RCR_RHS_MMA(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            G::load(As[tic][0], g.a, a_co(br*2, k+2), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

            G::load(b_tile(tic, 1), g.b, b_co(bc*2+1, k+2), soB);
            TK_WAIT_VMCNT(RCR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_setprio(1); RCR_RHS_MMA(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
#endif
        }
        }

        {
#if RCR_BATCHED_READS
            A_row_reg a1_epi;
            load_b(b0, b_tile(tic, 0), wn);
            load_b(b1, b_tile(tic, 1), wn);
            load_a(a, As[tic][0], wm);
            load_a(a1_epi, As[tic][1], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, g.ki-1), soA);
            TK_WAIT_LGKM(RCR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");

            __builtin_amdgcn_s_setprio(1);
            rcr_mma(cA, a, b0);
            RCR_SCHED_BARRIER();
            RCR_RHS_MMA(cB, a, b1);
            rcr_mma(cC, a1_epi, b0);
            RCR_SCHED_BARRIER();
            RCR_RHS_MMA(cD, a1_epi, b1);
            __builtin_amdgcn_s_setprio(0);

            TK_WAIT_VMCNT(RCR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            tic ^= 1; toc ^= 1;
#elif RCR_BATCHED_EPILOGUE_MMA
            load_b(b0, b_tile(tic, 0), wn);
            load_b(b1, b_tile(tic, 1), wn);
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, g.ki-1), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(cA, a, b0, cA);
            mma_ABt(cB, a, b1, cB);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(RCR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(cC, a, b0, cC);
            mma_ABt(cD, a, b1, cD);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, b_tile(toc, 0), wn);
            tic ^= 1; toc ^= 1;
#else
            load_b(b0, b_tile(tic, 0), wn);
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, g.ki-1), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

            load_b(b1, b_tile(tic, 1), wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); RCR_RHS_MMA(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(RCR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, b_tile(toc, 0), wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); RCR_RHS_MMA(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();
            tic ^= 1; toc ^= 1;
#endif
        }

        {
#if RCR_BATCHED_READS
            A_row_reg a1_last;
            load_b(b0, b_tile(tic, 0), wn);
            load_b(b1, b_tile(tic, 1), wn);
            load_a(a, As[tic][0], wm);
            load_a(a1_last, As[tic][1], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");

            __builtin_amdgcn_s_setprio(1);
            rcr_mma(cA, a, b0);
            RCR_SCHED_BARRIER();
            RCR_RHS_MMA(cB, a, b1);
            rcr_mma(cC, a1_last, b0);
            RCR_SCHED_BARRIER();
            RCR_RHS_MMA(cD, a1_last, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
#elif RCR_BATCHED_EPILOGUE_MMA
            load_a(a, As[tic][0], wm);
            load_b(b1, b_tile(tic, 1), wn);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(cA, a, b0, cA);
            mma_ABt(cB, a, b1, cB);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(cC, a, b0, cC);
            mma_ABt(cD, a, b1, cD);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
#else
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, b_tile(tic, 1), wn);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); RCR_RHS_MMA(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            rcr_mma(cC, a, b0);
            RCR_RHS_MMA(cD, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
#endif
        }
#endif

    } else if constexpr (L == Layout::RRR) {
        __shared__ ST_row As[2][2];
#if RRR_ROW_SHARED_TRANSPOSE
        __shared__ ST_row Bs[2][2];
#else
        __shared__ ST_rrr_b Bs[2][2];
#endif
        A_row_reg a;
        B_col_reg b0, b1;

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
        // Load B from its original KxN layout and transpose it on-chip into the
        // row-friendly shared view consumed by `mma_ABt`.
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
        for (int k = 0; k < g.ki - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, Bs[tic][0], wn);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
            const auto b0_keep = b0;
#endif
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, k+1), soA);
            TK_WAIT_LGKM(RRR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
            __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0_keep); __builtin_amdgcn_s_setprio(0);
#else
            __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
#endif
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();

            load_b(b1, Bs[tic][1], wn);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
            const auto b1_keep = b1;
#endif
#if RRR_ROW_SHARED_TRANSPOSE
            load_transpose<_NUM_THREADS>(Bs[tic][0], g.b, b_co(bc*2, k+2), soB);
#else
            G::load(Bs[tic][0], g.b, b_co(bc*2, k+2), soB);
#endif
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
            rrr_mma(cB, a, b1_keep);
#else
            rrr_mma(cB, a, b1);
#endif
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
#if RRR_ROW_SHARED_TRANSPOSE
            load_transpose<_NUM_THREADS>(Bs[tic][1], g.b, b_co(bc*2+1, k+2), soB);
#else
            G::load(Bs[tic][1], g.b, b_co(bc*2+1, k+2), soB);
#endif
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
            __builtin_amdgcn_s_setprio(1); rrr_mma(cC, a, b0_keep); __builtin_amdgcn_s_setprio(0);
#else
            __builtin_amdgcn_s_setprio(1); rrr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
#endif
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();

            G::load(As[tic][0], g.a, a_co(br*2, k+2), soA);
            TK_WAIT_VMCNT(RRR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
            rrr_mma(cD, a, b1_keep);
#else
            rrr_mma(cD, a, b1);
#endif
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        {
            load_b(b0, Bs[tic][0], wn);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
            const auto b0_keep = b0;
#endif
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, g.ki-1), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
            __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0_keep); __builtin_amdgcn_s_setprio(0);
#else
            __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
#endif
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();

            load_b(b1, Bs[tic][1], wn);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
            const auto b1_keep = b1;
#endif
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
            rrr_mma(cB, a, b1_keep);
#else
            rrr_mma(cB, a, b1);
#endif
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
            __builtin_amdgcn_s_setprio(1); rrr_mma(cC, a, b0_keep); __builtin_amdgcn_s_setprio(0);
#else
            __builtin_amdgcn_s_setprio(1); rrr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
#endif
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[toc][0], wn);
            TK_WAIT_VMCNT(RRR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
            __builtin_amdgcn_s_setprio(1); rrr_mma(cD, a, b1_keep); __builtin_amdgcn_s_setprio(0);
#else
            __builtin_amdgcn_s_setprio(1); rrr_mma(cD, a, b1); __builtin_amdgcn_s_setprio(0);
#endif
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();
            tic ^= 1; toc ^= 1;
        }

        {
            load_a(a, As[tic][0], wm);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
            const auto b0_keep = b0;
#endif
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
            __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0_keep); __builtin_amdgcn_s_setprio(0);
#else
            __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
#endif
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
            const auto b1_keep = b1;
#endif
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
            rrr_mma(cB, a, b1_keep);
#else
            rrr_mma(cB, a, b1);
#endif
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
            rrr_mma(cC, a, b0_keep);
            rrr_mma(cD, a, b1_keep);
#else
            rrr_mma(cC, a, b0);
            rrr_mma(cD, a, b1);
#endif
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

    } else if constexpr (L == Layout::CRR) {
#if CRR_ROW_SHARED_TRANSPOSE
        __shared__ ST_row As[2][2];
        __shared__ ST_row Bs[2][2];
        A_col_reg a;
        B_col_reg b0, b1;

        constexpr int bptA = ST_row::underlying_subtile_bytes_per_thread;
        constexpr int bpmA = bptA * _NUM_THREADS;
        constexpr int mptA = ST_row::rows * ST_row::cols * sizeof(fp8e4m3) / bpmA;
        uint32_t soA[mptA];
        prefill_transpose_swizzled_offsets<_NUM_THREADS>(As[0][0], g.a, soA);

        constexpr int bptB = ST_row::underlying_subtile_bytes_per_thread;
        constexpr int bpmB = bptB * _NUM_THREADS;
        constexpr int mptB = ST_row::rows * ST_row::cols * sizeof(fp8e4m3) / bpmB;
        uint32_t soB[mptB];
        prefill_transpose_swizzled_offsets<_NUM_THREADS>(Bs[0][0], g.b, soB);

        // Load A^T / B from their original layouts and transpose them on-chip into
        // row-friendly shared tiles, then transpose in registers into the col_l view.
        auto a_co = [&](int s, int k) -> coord<ST_row> { return {0, 0, s, k}; };
        auto b_co = [&](int s, int k) -> coord<ST_row> { return {0, 0, s, k}; };
        auto global_load_a = [&](ST_row& tile, int s, int k) {
            load_transpose<_NUM_THREADS>(tile, g.a, a_co(s, k), soA);
        };
        auto global_load_b = [&](ST_row& tile, int s, int k) {
            load_transpose<_NUM_THREADS>(tile, g.b, b_co(s, k), soB);
        };

        auto load_a = [&](A_col_reg& dst, ST_row& tile, int wi) {
            A_row_reg tmp;
            auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
            load(tmp, sub);
            transpose(dst, tmp);
        };
        auto load_b = [&](B_col_reg& dst, ST_row& tile, int wi) {
            B_row_reg tmp;
            auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
            load(tmp, sub);
            transpose(dst, tmp);
        };
#else
    #if CRR_USE_V3_SWIZZLE
        using ST_crr_a = ST_v3;
        using ST_crr_b = ST_v3;
    #else
        using ST_crr_a = ST_v2a;
        using ST_crr_b = ST_v2;
    #endif
        __shared__ ST_crr_a As[2][2];
        __shared__ ST_crr_b Bs[2][2];
#if CRR_A_LDS_REENCODE
        __shared__ ST_crr_a_reenc Aenc[2];
        A_row_reg a;
#else
        A_col_reg a;
#endif
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
#endif
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
        for (int k = 0; k < g.ki - 2; k++, tic ^= 1, toc ^= 1) {
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
            crr_mma(cA, a, b0);
            crr_mma(cB, a, b1);
            CRR_MMA_END();
            CRR_STEADY_MID_BARRIER(); CRR_SCHED_BARRIER();

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
#endif
#else
            load_b(b0, Bs[tic][0], wn);
#if CRR_A_REG_ROW_LOAD_TRANSPOSE && CRR_A_REG_ROW_LOAD_ALIAS
            const auto b0_keep = b0;
#endif
            load_a(a, As[tic][0], wm);
            global_load_a(As[toc][1], br*2+1, k+1);
            TK_WAIT_LGKM(CRR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
#if CRR_A_REG_ROW_LOAD_TRANSPOSE && CRR_A_REG_ROW_LOAD_ALIAS
            crr_mma(cA, a, b0_keep);
#else
            crr_mma(cA, a, b0);
#endif
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier(); CRR_SCHED_BARRIER();

            load_b(b1, Bs[tic][1], wn);
#if CRR_A_REG_ROW_LOAD_TRANSPOSE && CRR_A_REG_ROW_LOAD_ALIAS
            const auto b1_keep = b1;
#endif
            global_load_b(Bs[tic][0], bc*2, k+2);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
#if CRR_A_REG_ROW_LOAD_TRANSPOSE && CRR_A_REG_ROW_LOAD_ALIAS
            crr_mma(cB, a, b1_keep);
#else
            crr_mma(cB, a, b1);
#endif
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            global_load_a(As[tic][0], br*2, k+2);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
#if CRR_A_REG_ROW_LOAD_TRANSPOSE && CRR_A_REG_ROW_LOAD_ALIAS
            crr_mma(cC, a, b0_keep);
#else
            crr_mma(cC, a, b0);
#endif
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier(); CRR_SCHED_BARRIER();

            global_load_b(Bs[tic][1], bc*2+1, k+2);
            TK_WAIT_VMCNT(CRR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            CRR_MMA_BEGIN();
#if CRR_A_REG_ROW_LOAD_TRANSPOSE && CRR_A_REG_ROW_LOAD_ALIAS
            crr_mma(cD, a, b1_keep);
#else
            crr_mma(cD, a, b1);
#endif
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();
#endif
        }

        {
#if CRR_BATCHED_EPILOGUE_MMA
            load_b(b0, Bs[tic][0], wn);
            load_b(b1, Bs[tic][1], wn);
            load_a(a, As[tic][0], wm);
            global_load_a(As[toc][1], br*2+1, g.ki-1);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cA, a, b0);
            crr_mma(cB, a, b1);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier(); CRR_SCHED_BARRIER();

            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(CRR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cC, a, b0);
            crr_mma(cD, a, b1);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[toc][0], wn);
            tic ^= 1; toc ^= 1;
#else
#if CRR_A_LDS_REENCODE
            load_b(b0, Bs[tic][0], wn);
            load_a(a, wm);
            global_load_a(As[toc][1], br*2+1, g.ki-1);
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
#if CRR_A_REG_ROW_LOAD_TRANSPOSE && CRR_A_REG_ROW_LOAD_ALIAS
            const auto b0_keep = b0;
#endif
            load_a(a, As[tic][0], wm);
            global_load_a(As[toc][1], br*2+1, g.ki-1);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
#if CRR_A_REG_ROW_LOAD_TRANSPOSE && CRR_A_REG_ROW_LOAD_ALIAS
            crr_mma(cA, a, b0_keep);
#else
            crr_mma(cA, a, b0);
#endif
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier(); CRR_SCHED_BARRIER();

            load_b(b1, Bs[tic][1], wn);
#if CRR_A_REG_ROW_LOAD_TRANSPOSE && CRR_A_REG_ROW_LOAD_ALIAS
            const auto b1_keep = b1;
#endif
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
#if CRR_A_REG_ROW_LOAD_TRANSPOSE && CRR_A_REG_ROW_LOAD_ALIAS
            crr_mma(cB, a, b1_keep);
#else
            crr_mma(cB, a, b1);
#endif
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(CRR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
#if CRR_A_REG_ROW_LOAD_TRANSPOSE && CRR_A_REG_ROW_LOAD_ALIAS
            crr_mma(cC, a, b0_keep);
#else
            crr_mma(cC, a, b0);
#endif
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[toc][0], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
#if CRR_A_REG_ROW_LOAD_TRANSPOSE && CRR_A_REG_ROW_LOAD_ALIAS
            crr_mma(cD, a, b1_keep);
#else
            crr_mma(cD, a, b1);
#endif
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
            crr_mma(cA, a, b0);
            crr_mma(cB, a, b1);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier(); CRR_SCHED_BARRIER();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cC, a, b0);
            crr_mma(cD, a, b1);
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
#if CRR_A_REG_ROW_LOAD_TRANSPOSE && CRR_A_REG_ROW_LOAD_ALIAS
            const auto b0_keep = b0;
#endif
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
#if CRR_A_REG_ROW_LOAD_TRANSPOSE && CRR_A_REG_ROW_LOAD_ALIAS
            crr_mma(cA, a, b0_keep);
#else
            crr_mma(cA, a, b0);
#endif
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
#if CRR_A_REG_ROW_LOAD_TRANSPOSE && CRR_A_REG_ROW_LOAD_ALIAS
            const auto b1_keep = b1;
#endif
            __builtin_amdgcn_s_barrier(); CRR_SCHED_BARRIER();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
#if CRR_A_REG_ROW_LOAD_TRANSPOSE && CRR_A_REG_ROW_LOAD_ALIAS
            crr_mma(cB, a, b1_keep);
#else
            crr_mma(cB, a, b1);
#endif
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
#if CRR_A_REG_ROW_LOAD_TRANSPOSE && CRR_A_REG_ROW_LOAD_ALIAS
            crr_mma(cC, a, b0_keep);
            crr_mma(cD, a, b1_keep);
#else
            crr_mma(cC, a, b0);
            crr_mma(cD, a, b1);
#endif
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();
#endif
        #endif
        }    }

    const float combined_scale = g.scale_a * g.scale_b;
    mul(cA, cA, combined_scale);
    mul(cB, cB, combined_scale);
    mul(cC, cC, combined_scale);
    mul(cD, cD, combined_scale);

    // Store Output
    if (wm == 0) __builtin_amdgcn_s_barrier();
#if RCR_SWAP_STORE_HALVES
    store(g.c, cB, {0, 0, br*WARPS_M*2+wm,         bc*WARPS_N*2+wn});
    store(g.c, cA, {0, 0, br*WARPS_M*2+wm,         bc*WARPS_N*2+WARPS_N+wn});
    store(g.c, cD, {0, 0, br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+wn});
    store(g.c, cC, {0, 0, br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+WARPS_N+wn});
#else
    store(g.c, cA, {0, 0, br*WARPS_M*2+wm,         bc*WARPS_N*2+wn});
    store(g.c, cB, {0, 0, br*WARPS_M*2+wm,         bc*WARPS_N*2+WARPS_N+wn});
    store(g.c, cC, {0, 0, br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+wn});
    store(g.c, cD, {0, 0, br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+WARPS_N+wn});
#endif
}

template<Layout L>
__global__ void gemm_tail_kernel(const layout_globals g) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= g.m || col >= g.n) {
        return;
    }

    const bool interior_mn = row < g.fast_m && col < g.fast_n;
    const bool fast_covers_cell = interior_mn && g.fast_m > 0 && g.fast_n > 0 && g.fast_k > 0;
    const bool needs_k_tail = g.fast_k < g.k;
    if (fast_covers_cell && !needs_k_tail) {
        return;
    }

    const int k0 = fast_covers_cell ? g.fast_k : 0;
    float acc = 0.0f;
    for (int kk = k0; kk < g.k; ++kk) {
        if constexpr (L == Layout::RCR) {
            acc += load_fp8_scalar(g.a, row, kk) * load_fp8_scalar(g.b, col, kk);
        } else if constexpr (L == Layout::RRR) {
            acc += load_fp8_scalar(g.a, row, kk) * load_fp8_scalar(g.b, kk, col);
        } else {
            acc += load_fp8_scalar(g.a, kk, row) * load_fp8_scalar(g.b, kk, col);
        }
    }

    const float scaled = acc * g.scale_a * g.scale_b;
    if (fast_covers_cell && needs_k_tail) {
        store_bf16_scalar(g.c, row, col, load_bf16_scalar(g.c, row, col) + scaled);
    } else {
        store_bf16_scalar(g.c, row, col, scaled);
    }
}

template __global__ void gemm_kernel<Layout::RCR>(const layout_globals);
template __global__ void gemm_kernel<Layout::RRR>(const layout_globals);
template __global__ void gemm_kernel<Layout::CRR>(const layout_globals);
template __global__ void gemm_tail_kernel<Layout::RCR>(const layout_globals);
template __global__ void gemm_tail_kernel<Layout::RRR>(const layout_globals);
template __global__ void gemm_tail_kernel<Layout::CRR>(const layout_globals);

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

#if RCR_USE_EXACT_4WAVE_FASTPATH
    if constexpr (L == Layout::RCR) {
        if (rcr_can_use_exact_4wave(g)) {
            dispatch_rcr_exact_4wave(g);
            return;
        }
    }
#endif
#if RCR_USE_EXACT_8WAVE_FASTPATH
    if constexpr (L == Layout::RCR) {
        if (rcr_can_use_exact_8wave(g)) {
            dispatch_rcr_exact_8wave(g);
            return;
        }
    }
#endif
#if RRR_USE_EXACT_8WAVE_FASTPATH
    if constexpr (L == Layout::RRR) {
        if (rrr_can_use_exact_8wave(g)) {
            dispatch_rrr_exact_8wave(g);
            return;
        }
    }
#endif
#if CRR_USE_EXACT_4WAVE_FASTPATH
    if constexpr (L == Layout::CRR) {
        if (crr_can_use_exact_4wave(g)) {
            dispatch_crr_exact_4wave(g);
            return;
        }
    }
#endif
#if CRR_USE_EXACT_8WAVE_DOUBLE_PUMP_FASTPATH
    if constexpr (L == Layout::CRR) {
        if (crr_can_use_exact_8wave_double_pump(g)) {
            dispatch_crr_exact_8wave_double_pump(g);
            return;
        }
    }
#endif
#if CRR_USE_EXACT_8WAVE_FASTPATH
    if constexpr (L == Layout::CRR) {
        if (crr_can_use_exact_8wave(g)) {
            dispatch_crr_exact_8wave(g);
            return;
        }
    }
#endif

    g.fast_m = (g.m / BLK) * BLK;
    g.fast_n = (g.n / BLK) * BLK;
    g.fast_k = (g.k / BK) * BK;
    g.bpr = g.fast_m / BLK;
    g.bpc = g.fast_n / BLK;
    g.ki = g.fast_k / BK;

    if (g.bpr > 0 && g.bpc > 0 && g.ki >= 2) {
        gemm_kernel<L><<<g.grid(), g.block(), 0, g.stream>>>(g);
    } else {
        g.fast_k = 0;
        g.ki = 0;
    }

    if (g.fast_m != g.m || g.fast_n != g.n || g.fast_k != g.k || g.ki == 0) {
        dim3 tail_block(TAIL_BLOCK_N, TAIL_BLOCK_M);
        dim3 tail_grid(
            kittens::ceil_div(g.n, TAIL_BLOCK_N),
            kittens::ceil_div(g.m, TAIL_BLOCK_M)
        );
        gemm_tail_kernel<L><<<tail_grid, tail_block, 0, g.stream>>>(g);
    }
}

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
    };
    dispatch<L>(g);
}

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
    m.def("supports_shape", [](int m, int n, int k) -> bool {
        return m > 0 && n > 0 && k > 0;
    });
    m.attr("DEFAULT_GROUP_M") = DEFAULT_GROUP_M;
    m.attr("BLOCK_SIZE") = BLK;
    m.attr("K_BLOCK") = BK;
}
