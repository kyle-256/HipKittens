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

// RCR two-tile schedule threshold (ki >= this → use two-tile main-loop schedule)
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

// Per-layout wait-counter budgets (#define since they're stringified into asm).
#define RCR_PREFETCH_LGKM       4
#define RCR_INIT0_VMCNT         4
#define RCR_INIT1_VMCNT         6
#define RCR_STEADY_VMCNT        8
#define RCR_EPILOGUE_VMCNT      4
#define RCR_TWO_TILE_MID_VMCNT  6
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

__device__ __forceinline__ void crr_mma(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
    const A_col_reg& a,
    const B_col_reg& b)
{
    const auto& a_row = reinterpret_cast<const A_row_reg&>(a);
    mma_AB(acc, a_row, b, acc);
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

// Per-group scalar FP8 load. ``b`` for grouped FP8 is logically
// [batch=1, G, N, K]; the 4D coord lets `grouped_tail_kernel` index B at
// (group_idx, row, col).
__device__ __forceinline__ float load_fp8_scalar_grp(const _gl_fp8& src, int g_idx, int row, int col) {
    return base_types::convertor<float, fp8e4m3>::convert(src[coord<>{0, g_idx, row, col}]);
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
    // Optional device-side scalar scales. When non-null, the kernel epilogue
    // reads `*dscale_a * *dscale_b` from device memory instead of using the
    // host-side `scale_a * scale_b` floats above. This lets the Python host
    // wrapper skip the `.item()` stream sync that would otherwise be required
    // to materialise the scales into host floats before kernel launch -- the
    // sync was responsible for ~18us of dispatch latency on small dense FP8
    // shapes (4096^3 etc) where it was bigger than the GEMM kernel itself
    // is over default hipBLASLt. nullptr selects the host-scale path so
    // existing call sites (gemm_{rcr,rrr,crr}) keep working unchanged.
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
// store_c_tile_mn_masked_grouped — two-axis-masked C store for the grouped
// variable-K (CRR / dB) kernel below. Mirror of the BF16 variant in
// analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp.
//
// Output layout is 3D-grouped ``[G, m_kernel, n_kernel]``; partial last
// tiles in either axis are dropped per-cell. Aligned tiles forward to
// the unmasked ``store(...)`` with no overhead.
// =============================================================================
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

namespace rcr_4w {

constexpr int WM = 2, WN = 2;
constexpr int NW = WM * WN;
constexpr int NT = NW * WARP_THREADS;

using G4 = kittens::group<NW>;
using ST4 = st_fp8e4m3<HB, BK, st_16x128_s>;
using RA = rt_fp8e4m3<HB / WM, BK, row_l, rt_16x128_s>;
using RB = rt_fp8e4m3<HB / WN, BK, row_l, rt_16x128_s>;
using RC = rt_fl<HB / WM, HB / WN, col_l, rt_16x16_s>;

struct g2s_desc {
    i32x4 srsrc;
    uintptr_t lds_base;
};

__device__ __forceinline__ g2s_desc prepare_g2s(
    ST4& dst, const void* tile_ptr, int row_stride)
{
    i32x4 srsrc = make_srsrc(const_cast<void*>(tile_ptr),
                              row_stride * ST4::rows * sizeof(fp8e4m3));
    constexpr int bpt = ST4::underlying_subtile_bytes_per_thread;
    constexpr int bpw = bpt * WARP_THREADS;
    uintptr_t lds_base =
        reinterpret_cast<uintptr_t>(&dst.data[0]) + (kittens::warpid() * bpw);
    return {srsrc, lds_base};
}

constexpr int G2S_PASSES = HB * BK * sizeof(fp8e4m3) /
    (ST4::underlying_subtile_bytes_per_thread * NT);

struct g2s_offsets {
    uint32_t global_off[G2S_PASSES];
};

__device__ __forceinline__ g2s_offsets prefill_g2s_offsets(
    ST4& dst, int row_stride)
{
    using T = fp8e4m3;
    constexpr int bpt = ST4::underlying_subtile_bytes_per_thread;
    constexpr int bpw = bpt * WARP_THREADS;
    const int lane = kittens::laneid();
    const int warp = kittens::warpid() % NW;

    g2s_offsets off;
    #pragma unroll
    for (int I = 0; I < G2S_PASSES; I++) {
        const int lbo = (lane * bpt) + (warp * bpw) + (I * NW * bpw);
        const int stid = lbo / ST4::underlying_subtile_bytes;
        const int str = stid / ST4::underlying_subtiles_per_row;
        const int stc = stid % ST4::underlying_subtiles_per_row;
        const int sub = lbo % ST4::underlying_subtile_bytes;
        const int row = sub / ST4::underlying_subtile_row_bytes;
        const int col = (sub % ST4::underlying_subtile_row_bytes) / sizeof(T);
        const uint32_t sw = dst.swizzle({row, col});
        const int swr = (sw / ST4::underlying_subtile_row_bytes) +
                         str * ST4::underlying_subtile_rows;
        const int swc = (sw % ST4::underlying_subtile_row_bytes) / sizeof(T) +
                         stc * ST4::underlying_subtile_cols;
        off.global_off[I] = (swr * row_stride + swc) * sizeof(T);
    }
    return off;
}

template<int I>
__device__ __forceinline__ void g2s_pass(
    const g2s_desc& addr, const g2s_offsets& off)
{
    constexpr int bpt = ST4::underlying_subtile_bytes_per_thread;
    constexpr int bpw = bpt * WARP_THREADS;
    uintptr_t lds_addr = addr.lds_base + (I * NW * bpw);
    as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);
    llvm_amdgcn_raw_buffer_load_lds(
        addr.srsrc, lds_ptr, bpt, off.global_off[I], 0, 0,
        static_cast<int>(coherency::cache_all));
}

template<int NUM_OFFSETS, typename RT, typename ST>
__device__ __forceinline__ void prefill_s2r_offsets(
    RT& dst, ST& src, uint32_t* off)
{
    using U = typename ST::dtype;
    constexpr int subtile_stride = RT::base_tile_cols * sizeof(U) / 2;
    const uint32_t st_offset =
        (kittens::laneid() % RT::base_tile_rows) * ST::underlying_cols +
        (kittens::laneid() / RT::base_tile_rows * 16 / sizeof(U));
    const uint32_t base_addr = reinterpret_cast<uintptr_t>(&src.data[st_offset]);
    off[0] = base_addr;
    off[0] ^= (((off[0] % (256 * 8)) >> 8) << 4);
    off[1] = base_addr + subtile_stride;
    off[1] ^= (((off[1] % (256 * 8)) >> 8) << 4);
}

template<int RR, int RC, int KS, typename RT, typename ST>
__device__ __forceinline__ void s2r_one(RT& dst, ST& src, uint32_t* off)
{
    constexpr int packing = base_types::packing<typename RT::dtype>::num();
    const int idx = KS * RT::base_tile_stride / packing;
    constexpr int row_stride =
        RT::base_tile_rows * ST::underlying_cols * sizeof(fp8e4m3);
    asm volatile(
        "ds_read_b128 %0, %1 offset:%2\n"
        : "=v"(*reinterpret_cast<float4*>(&dst.tiles[RR][RC].data[idx]))
        : "v"(off[KS]), "i"(RR * row_stride)
        : "memory");
}

template<typename D, typename A, typename B, typename C>
__device__ __forceinline__ void mma1(D& d, const A& a, const B& b, const C& c,
                                     int n, int m, int k)
{
    mma_ABt_base(d.tiles[n][m], a.tiles[n][k], b.tiles[m][k], c.tiles[n][m]);
}

template<typename S2R_RT, typename S2R_ST>
__device__ __forceinline__ void do_cluster(
    const g2s_desc& g2s_addr, const g2s_offsets& g2s_off,
    S2R_RT& s2r_dst, S2R_ST& s2r_src,
    RA& a, RB& b, RC& c)
{
    uint32_t s2r_off[2];
    prefill_s2r_offsets<2>(s2r_dst, s2r_src, s2r_off);

    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 0, 0, 0);
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 0, 1, 0);
    __builtin_amdgcn_sched_barrier(0);

    g2s_pass<0>(g2s_addr, g2s_off);
    s2r_one<0, 0, 0>(s2r_dst, s2r_src, s2r_off);

    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 0, 2, 0);
    __builtin_amdgcn_sched_barrier(0);

    s2r_one<0, 0, 1>(s2r_dst, s2r_src, s2r_off);

    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 0, 3, 0);
    __builtin_amdgcn_sched_barrier(0);

    g2s_pass<1>(g2s_addr, g2s_off);
    s2r_one<1, 0, 0>(s2r_dst, s2r_src, s2r_off);
    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 1, 0, 0);
    mma1(c, a, b, c, 1, 1, 0);
    __builtin_amdgcn_sched_barrier(0);

    s2r_one<1, 0, 1>(s2r_dst, s2r_src, s2r_off);
    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 1, 2, 0);
    mma1(c, a, b, c, 1, 3, 0);
    __builtin_amdgcn_sched_barrier(0);

    g2s_pass<2>(g2s_addr, g2s_off);
    s2r_one<2, 0, 0>(s2r_dst, s2r_src, s2r_off);
    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 2, 0, 0);
    mma1(c, a, b, c, 2, 1, 0);
    __builtin_amdgcn_sched_barrier(0);

    s2r_one<2, 0, 1>(s2r_dst, s2r_src, s2r_off);
    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 2, 2, 0);
    mma1(c, a, b, c, 2, 3, 0);
    __builtin_amdgcn_sched_barrier(0);

    g2s_pass<3>(g2s_addr, g2s_off);
    s2r_one<3, 0, 0>(s2r_dst, s2r_src, s2r_off);
    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 3, 0, 0);
    mma1(c, a, b, c, 3, 1, 0);
    __builtin_amdgcn_sched_barrier(0);

    s2r_one<3, 0, 1>(s2r_dst, s2r_src, s2r_off);
    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 3, 2, 0);
    mma1(c, a, b, c, 3, 3, 0);
    __builtin_amdgcn_sched_barrier(0);
}

template<ducks::rt::row_layout RT, ducks::st::all ST>
__device__ __forceinline__ void load_full_rt(RT& dst, const ST& src) {
    static_assert(RT::rows == ST::rows && RT::cols == ST::cols);
    using T2 = typename RT::dtype;
    using U2 = typename base_types::packing<typename ST::dtype>::packed_type;
    constexpr int packing = base_types::packing<T2>::num();
    const int laneid = kittens::laneid();
    const int row_offset = laneid % dst.base_tile_rows;
    const int col_offset = dst.base_tile_stride * (laneid / dst.base_tile_rows);
    const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);
    constexpr int rps_row = ST::underlying_subtile_cols / RT::base_tile_cols;
    constexpr int rps_col = ST::underlying_subtile_rows / RT::base_tile_rows;
    #pragma unroll
    for (int k = 0; k < RT::base_tile_num_strides; k++) {
        #pragma unroll
        for (int i = 0; i < rps_col; i++) {
            #pragma unroll
            for (int j = 0; j < rps_row; j++) {
                const int row = i * RT::base_tile_rows + row_offset;
                const int col = j * RT::base_tile_cols + col_offset +
                    k * RT::base_tile_elements_per_stride_group;
                const uint32_t offset =
                    sizeof(fp8e4m3) * (src_ptr + row * ST::underlying_subtile_cols + col);
                const uint32_t addr = offset ^ (((offset % (16 * 128)) >> 8) << 4);
                const int idx = k * RT::base_tile_stride / packing;
                #pragma unroll
                for (int ii = 0; ii < ST::subtiles_per_col; ii++) {
                    #pragma unroll
                    for (int jj = 0; jj < ST::subtiles_per_row; jj++) {
                        const int sid = ii * ST::underlying_subtiles_per_row + jj;
                        const int soff = sid * ST::underlying_subtile_bytes;
                        const int rr = ii * rps_col + i;
                        const int rc = jj * rps_row + j;
                        asm volatile("ds_read_b128 %0, %1 offset:%2\n"
                            : "=v"(*reinterpret_cast<float4*>(&dst.tiles[rr][rc].data[idx]))
                            : "v"(addr), "i"(soff) : "memory");
                    }
                }
            }
        }
    }
}

__global__ __launch_bounds__(NT, 2)
void kernel(const layout_globals g) {
    int br, bc;
    gemm_compute_block_coords(blockIdx.x, g.bpr, g.bpc, g.group_m, br, bc);
    if (br >= g.bpr || bc >= g.bpc || g.ki <= 0) return;

    const int wm = kittens::warpid() / WN;
    const int wn = kittens::warpid() % WN;
    const int ki = g.ki;

    const fp8e4m3* a_base = reinterpret_cast<const fp8e4m3*>(g.a.raw_ptr);
    const fp8e4m3* b_base = reinterpret_cast<const fp8e4m3*>(g.b.raw_ptr);
    const int a_stride = g.k;
    const int b_stride = g.k;

    auto a_ptr = [&](int row_tile, int k_tile) -> const void* {
        return a_base + row_tile * HB * a_stride + k_tile * BK;
    };
    auto b_ptr = [&](int col_tile, int k_tile) -> const void* {
        return b_base + col_tile * HB * b_stride + k_tile * BK;
    };

    __shared__ ST4 As[2][2];
    __shared__ ST4 Bs[2][2];
    RA a_reg[2];
    RB b_reg[2];
    RC c[2][2];
    zero(c[0][0]); zero(c[0][1]); zero(c[1][0]); zero(c[1][1]);

    constexpr int bpt = ST4::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * NT;
    constexpr int mpt = HB * BK * sizeof(fp8e4m3) / bpm;
    uint32_t soA[mpt], soB[mpt];
    G4::prefill_swizzled_offsets(As[0][0], g.a, soA);
    G4::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

    g2s_offsets g2s_off_A = prefill_g2s_offsets(As[0][0], a_stride);
    g2s_offsets g2s_off_B = prefill_g2s_offsets(Bs[0][0], b_stride);

    auto a_co = [&](int s, int k) -> coord<ST4> { return {0, 0, s, k}; };
    auto b_co = [&](int s, int k) -> coord<ST4> { return {0, 0, s, k}; };

    int cur = 0, nxt = 1;
    G4::load(As[cur][0], g.a, a_co(br * WM,     0), soA);
    G4::load(Bs[cur][0], g.b, b_co(bc * WN,     0), soB);
    G4::load(Bs[cur][1], g.b, b_co(bc * WN + 1, 0), soB);
    G4::load(As[cur][1], g.a, a_co(br * WM + 1, 0), soA);

    G4::load(As[nxt][0], g.a, a_co(br * WM,     1), soA);
    G4::load(Bs[nxt][0], g.b, b_co(bc * WN,     1), soB);
    G4::load(Bs[nxt][1], g.b, b_co(bc * WN + 1, 1), soB);
    G4::load(As[nxt][1], g.a, a_co(br * WM + 1, 1), soA);

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(28)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    auto a_sub0 = subtile_inplace<HB / WM, BK>(As[cur][0], {wm, 0});
    load_full_rt(a_reg[0], a_sub0);

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(24)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    auto b_sub0 = subtile_inplace<HB / WN, BK>(Bs[cur][0], {wn, 0});
    load_full_rt(b_reg[0], b_sub0);

    #pragma unroll 1
    for (int k = 0; k < ki - 2; ++k, cur ^= 1, nxt ^= 1) {
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(16)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto b_sub1 = subtile_inplace<HB / WN, BK>(Bs[cur][1], {wn, 0});
        g2s_desc addr_a0 = prepare_g2s(As[cur][0], a_ptr(br * WM, k + 2), a_stride);
        do_cluster(
            addr_a0, g2s_off_A,
            b_reg[1], b_sub1,
            a_reg[0], b_reg[0], c[0][0]);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_sub1 = subtile_inplace<HB / WM, BK>(As[cur][1], {wm, 0});
        g2s_desc addr_b0 = prepare_g2s(Bs[cur][0], b_ptr(bc * WN, k + 2), b_stride);
        do_cluster(
            addr_b0, g2s_off_B,
            a_reg[1], a_sub1,
            a_reg[0], b_reg[1], c[0][1]);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(16)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        a_sub0 = subtile_inplace<HB / WM, BK>(As[nxt][0], {wm, 0});
        g2s_desc addr_b1 = prepare_g2s(Bs[cur][1], b_ptr(bc * WN + 1, k + 2), b_stride);
        do_cluster(
            addr_b1, g2s_off_B,
            a_reg[0], a_sub0,
            a_reg[1], b_reg[0], c[1][0]);

        b_sub0 = subtile_inplace<HB / WN, BK>(Bs[nxt][0], {wn, 0});
        g2s_desc addr_a1 = prepare_g2s(As[cur][1], a_ptr(br * WM + 1, k + 2), a_stride);
        do_cluster(
            addr_a1, g2s_off_A,
            b_reg[0], b_sub0,
            a_reg[1], b_reg[1], c[1][1]);
    }

    // Epilogue: k = ki - 2
    {
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(16)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto b_sub1 = subtile_inplace<HB / WN, BK>(Bs[cur][1], {wn, 0});
        load_full_rt(b_reg[1], b_sub1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][0], a_reg[0], b_reg[0], c[0][0]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_sub1 = subtile_inplace<HB / WM, BK>(As[cur][1], {wm, 0});
        load_full_rt(a_reg[1], a_sub1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][1], a_reg[0], b_reg[1], c[0][1]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(8)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        a_sub0 = subtile_inplace<HB / WM, BK>(As[nxt][0], {wm, 0});
        load_full_rt(a_reg[0], a_sub0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][0], a_reg[1], b_reg[0], c[1][0]);
        __builtin_amdgcn_sched_barrier(0);

        b_sub0 = subtile_inplace<HB / WN, BK>(Bs[nxt][0], {wn, 0});
        load_full_rt(b_reg[0], b_sub0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][1], a_reg[1], b_reg[1], c[1][1]);
        __builtin_amdgcn_sched_barrier(0);

        cur ^= 1; nxt ^= 1;
    }

    // Last iteration
    {
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto b_sub1 = subtile_inplace<HB / WN, BK>(Bs[cur][1], {wn, 0});
        load_full_rt(b_reg[1], b_sub1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][0], a_reg[0], b_reg[0], c[0][0]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_sub1 = subtile_inplace<HB / WM, BK>(As[cur][1], {wm, 0});
        load_full_rt(a_reg[1], a_sub1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][1], a_reg[0], b_reg[1], c[0][1]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][0], a_reg[1], b_reg[0], c[1][0]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][1], a_reg[1], b_reg[1], c[1][1]);
        __builtin_amdgcn_sched_barrier(0);
    }

    const float sc = resolve_combined_scale(g);
    mul(c[0][0], c[0][0], sc);
    mul(c[0][1], c[0][1], sc);
    mul(c[1][0], c[1][0], sc);
    mul(c[1][1], c[1][1], sc);

    store_c_tile_n_masked(g.c, c[0][0], br * WM * 2 + wm,      bc * WN * 2 + wn,      g.n);
    store_c_tile_n_masked(g.c, c[0][1], br * WM * 2 + wm,      bc * WN * 2 + WN + wn, g.n);
    store_c_tile_n_masked(g.c, c[1][0], br * WM * 2 + WM + wm, bc * WN * 2 + wn,      g.n);
    store_c_tile_n_masked(g.c, c[1][1], br * WM * 2 + WM + wm, bc * WN * 2 + WN + wn, g.n);
}

} // namespace rcr_4w

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

// =============================================================================
// Persistent grouped GEMM (CPU-sync-free) — RCR layout only for round 12.
//
// Mirror of the BF16 grouped persistent kernel
// (analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp ``grouped_kernel<L,KI>``)
// ported to the FP8 register / shared tile types. One launch with grid_x =
// NUM_CUS programs covers ALL groups × ALL tiles. Each program:
//
//   1. Pulls G+1 int64 offsets from a device tensor and computes total tile
//      count via O(G) scan (no host sync).
//   2. Iterates ``gt = pid; gt < total; gt += NUM_CUS`` so the same block
//      streams through many (group, tile) pairs without re-launch.
//   3. Per iteration: O(G) scan to recover (group_idx, m_start_g, M_g),
//      then runs the existing dense RCR GEMM tile body with coord shifts:
//         * A   spatial += m_start_g / HB    (HB  = 128, ST_A row unit)
//         * B   depth   = group_idx          (b is treated as [G, N, K])
//         * C   row     += m_start_g / RBM   (RBM = 64,  RT::rows store unit)
//
// Inner body is the SINGLE-tile main loop + epilogs from the dense kernel
// (the ``else`` branch of ``gemm_kernel<Layout::RCR, ...>`` lines 1129-1210
// + scale epilog at 1500). Two-tile schedule (faster for ki>=28 with even
// ki) is intentionally not used in this round to keep the persistent path
// simple; can be added in a follow-up. RRR / CRR persistent variants ditto.
//
// Scale epilog: ``scale_a * scale_b`` applied per tile (matches dense).
// =============================================================================

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
    // Round-13: optional host-side hint — average per-group M in the
    // current launch. Consumed by the LDS-staged K-tail correction
    // kernel (``grouped_ktail_kernel_lds``) to gate the cooperative LDS
    // path: each tail block is (TBM × TBN); if ``m_per_group >= TBM``
    // and ``m_per_group % TBM == 0`` the per-block "all rows are in one
    // group" precondition holds for all blocks. Mirrors BF16 round-9/11
    // wiring; default 0 keeps the legacy scalar-tail fallback.
    int m_per_group;
    dim3 block() { return dim3(_NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

__device__ __forceinline__ float resolve_combined_scale_grp(
    const grouped_layout_globals &g) {
    const float sa = g.dscale_a ? *g.dscale_a : g.scale_a;
    const float sb = g.dscale_b ? *g.dscale_b : g.scale_b;
    return sa * sb;
}

// Persistent RCR kernel: grid_x = NUM_CUS. One block per CU; each block
// iterates many (group, tile) pairs in a single launch.
//
// Round-12: ``N_MASKED_STORE`` selects the C-store path at compile time.
// When ``false`` (N is BLOCK_SIZE-aligned, e.g. DSV3 N=4096/7168) the
// masked variant is dead-code-eliminated and the main kernel emits the
// same raw-store sequence as the round-11 path — keeping VGPR pressure
// low (no spill from the masked branch's lane-level row/col reconstruction).
// When ``true`` (N misaligned, e.g. gpt_oss N=2880/5760) we use
// ``store_c_tile_n_masked`` to drop OOB cols on the partial last tile.
template<int KI_HINT = 0, bool N_MASKED_STORE = false>
__global__ __launch_bounds__(_NUM_THREADS, 1)
void grouped_rcr_kernel(const grouped_layout_globals g) {
    using ST_rcr = ST_v2;
    __shared__ ST_rcr As[2][2];
    __shared__ ST_rcr Bs[2][2];
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
    A_row_reg a;
    B_row_reg b0, b1;
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;

    // [grouped] Persistent: chiplet-swizzle pid against full NUM_CUS grid.
    int pid = chiplet_transform_chunked(
        blockIdx.x, NUM_CUS, BLOCK_SWIZZLE_NUM_XCDS, 64);

    int wm = warpid() / WARPS_N;
    int wn = warpid() % WARPS_N;
    const int num_pid_n = g.bpc;
    const int ki_dyn   = (KI_HINT > 0) ? KI_HINT : g.ki;

    // [grouped] Cooperative init of the LDS group-metadata caches. Single
    // thread does the O(G) scan once; then everyone uses s_offs / s_cum_tiles.
    // Pad s_cum_tiles[g.G + 1 .. MAX_G_PLUS_1) with INT_MAX so a constant-
    // depth (6-step) branch-free binary search reading any mid > g.G never
    // updates lo (the cmp `gt >= INT_MAX` is always false for finite gt).
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
        #pragma unroll 1
        for (int gi = g.G + 1; gi < MAX_G_PLUS_1; ++gi) {
            s_cum_tiles[gi] = 0x7FFFFFFF;
        }
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

    // [grouped] Persistent outer loop.
    for (int gt = pid; gt < total_tiles; gt += NUM_CUS) {
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
        rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0],    g.a, a_co(br*2,   0), soA);
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 1), g.b, b_co(bc*2+1, 0), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[tic][1],    g.a, a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(RCR_INIT0_VMCNT);
        __builtin_amdgcn_s_barrier();

        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(toc, 0), g.b, b_co(bc*2,   1), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[toc][0],    g.a, a_co(br*2,   1), soA);
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(toc, 1), g.b, b_co(bc*2+1, 1), soB);

        TK_WAIT_VMCNT(RCR_INIT1_VMCNT);
        __builtin_amdgcn_s_barrier();

        // Single-tile main loop (mirrors dense gemm_kernel<RCR> else-branch
        // lines 1129-1158).
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

        // Epilog 1: second-to-last K-tile (mirrors dense lines 1160-1187).
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

        // Apply scale + store with m_subtile_C row shift.
        const float combined_scale = resolve_combined_scale_grp(g);
        mul(cA, cA, combined_scale);
        mul(cB, cB, combined_scale);
        mul(cC, cC, combined_scale);
        mul(cD, cD, combined_scale);

        if (wm == 0) __builtin_amdgcn_s_barrier();
        // Round-12: mirror BF16 grouped's column-masked C store. With
        // ``g.bpc = ceil_div(g.n, BLOCK_SIZE)`` (dispatch_grouped_rcr
        // round-12 path) the last col-tile may straddle ``[fast_n, n)``;
        // ``store_c_tile_n_masked`` drops OOB columns. ``rcr_8w_load_hoist``
        // already uses the full-tensor SRD (line ~432) so OOB rows in B
        // clamp to 0 — the masked C store then prevents those cells from
        // being written. ``N_MASKED_STORE`` is a compile-time template
        // parameter so the N-aligned dispatch path emits the raw store
        // (no spill from the masked variant's row/col reconstruction).
        const int r0 = m_subtile_C + br*WARPS_M*2+wm;
        const int r1 = m_subtile_C + br*WARPS_M*2+WARPS_M+wm;
        const int c0 = bc*WARPS_N*2+wn;
        const int c1 = bc*WARPS_N*2+WARPS_N+wn;
        if constexpr (N_MASKED_STORE) {
            store_c_tile_n_masked(g.c, cA, r0, c0, g.n);
            store_c_tile_n_masked(g.c, cB, r0, c1, g.n);
            store_c_tile_n_masked(g.c, cC, r1, c0, g.n);
            store_c_tile_n_masked(g.c, cD, r1, c1, g.n);
        } else {
            store(g.c, cA, {0, 0, r0, c0});
            store(g.c, cB, {0, 0, r0, c1});
            store(g.c, cC, {0, 0, r1, c0});
            store(g.c, cD, {0, 0, r1, c1});
        }

        // [grouped] Drain in-flight ops before the next persistent iteration
        // so the next tile's prologue starts from a clean state.
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }
}

template __global__ void grouped_rcr_kernel<0, false>(const grouped_layout_globals);
template __global__ void grouped_rcr_kernel<0, true >(const grouped_layout_globals);

// =============================================================================
// Persistent RRR grouped kernel — round-1 mirror of grouped_rcr_kernel.
//
// Identical persistent + CPU-sync-free skeleton (LDS group_offs cache, 6-step
// branch-free binary search over s_cum_tiles, group-by-M/N tile swizzle,
// chiplet pid permutation), with the inner per-tile body swapped for the
// FP8 RRR dense schedule (lines 1389-1526). Layout-specific differences:
//
//   * B layout : [1, G, K, N] row-major (K outer, N inner stride-1).
//                ``b_co(s, k)`` maps a tile coord to {0, g_idx, k, s}, i.e.
//                the K-tile index sits on row axis and the N-tile index sits
//                on col axis (vs RCR which puts N on row, K on col).
//
//   * Shared B : ST_v2 (st_fp8e4m3<HB, BK, st_16x128_v2_s>) — same row-tile
//                shape as ST_rcr but the underlying load/store patterns are
//                the v2 col-major-swizzle layout that the RRR mma reads
//                from.
//
//   * Shared A : ST_row (st_fp8e4m3<HB, BK, st_16x128_s>) — straight row-
//                major. RCR reuses ST_v2 for A; RRR uses the simpler ST_row.
//
//   * Register : A_row_reg + B_col_reg (B is column-loaded into the col
//                register layout via load_col_from_st).
//
//   * MMA      : rrr_mma (mma_AB) instead of rcr_mma (mma_ABt).
//
//   * Loads    : G::load (kittens::load with swizzled offsets) for both A
//                and B. The 8-wave m0-broadcast hoist (rcr_8w_load_hoist)
//                is RCR-specific; RRR keeps the standard kittens load path
//                used by the FP8 dense RRR kernel.
//
// Coord shifts inside the persistent loop:
//   * A row    : m_subtile_A = m_start_g / HB              (HB = 128)
//                a_co(s, k) -> {0, 0, m_subtile_A + s, k}
//   * B group  : b_co(s, k) -> {0, group_idx, k, s}       (G as depth axis)
//   * C row    : m_subtile_C = m_start_g / RBM             (RBM = 64)
//                store(g.c, ..., {0, 0, m_subtile_C + R, C})
//
// Aligned interior: ``g.fast_n = (n / BLOCK_SIZE) * BLOCK_SIZE``,
// ``g.fast_k = (k / K_BLOCK) * K_BLOCK``. Cells outside (col >= fast_n) and
// the K-tail correction in [fast_k, k) for interior cells are handled by
// ``grouped_tail_kernel<Layout::RRR>`` (scalar fp32, mirror BF16 RRR).
// =============================================================================
template<int KI_HINT = 0>
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

    int pid = chiplet_transform_chunked(
        blockIdx.x, NUM_CUS, BLOCK_SWIZZLE_NUM_XCDS, 64);

    int wm = warpid() / WARPS_N;
    int wn = warpid() % WARPS_N;
    const int num_pid_n = g.bpc;
    const int ki_dyn   = (KI_HINT > 0) ? KI_HINT : g.ki;

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
        #pragma unroll 1
        for (int gi = g.G + 1; gi < MAX_G_PLUS_1; ++gi) {
            s_cum_tiles[gi] = 0x7FFFFFFF;
        }
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

        const float combined_scale = resolve_combined_scale_grp(g);
        mul(cA, cA, combined_scale);
        mul(cB, cB, combined_scale);
        mul(cC, cC, combined_scale);
        mul(cD, cD, combined_scale);

        if (wm == 0) __builtin_amdgcn_s_barrier();
        store(g.c, cA, {0, 0, m_subtile_C + br*WARPS_M*2+wm,
                              bc*WARPS_N*2+wn});
        store(g.c, cB, {0, 0, m_subtile_C + br*WARPS_M*2+wm,
                              bc*WARPS_N*2+WARPS_N+wn});
        store(g.c, cC, {0, 0, m_subtile_C + br*WARPS_M*2+WARPS_M+wm,
                              bc*WARPS_N*2+wn});
        store(g.c, cD, {0, 0, m_subtile_C + br*WARPS_M*2+WARPS_M+wm,
                              bc*WARPS_N*2+WARPS_N+wn});

        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }
}

template __global__ void grouped_rrr_kernel<0>(const grouped_layout_globals);

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

    // Round-12: ``main_covers_n`` mirrors the dispatch decision (see
    // ``dispatch_grouped_rcr``). When ``g.bpc * BLOCK_SIZE > g.fast_n``,
    // the main kernel ran with ``bpc = ceil_div(g.n, BLOCK_SIZE)`` and
    // ``store_c_tile_n_masked`` already wrote cols [0, g.n) with the
    // [0, fast_k) partial K reduction. Tail must NOT redo full-K
    // reduction for those cells — only add the K-tail [fast_k, k)
    // correction. Detected from ``g.bpc`` itself (mirror dense
    // ``gemm_tail_kernel`` pattern).
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

    // Round-12: ``interior_n`` widens to ``col < g.n`` whenever
    // ``main_covers_n``. The fast_covers_cell branch then takes the
    // K-tail correction path (RMW + acc) for ALL cols [0, g.n) — the
    // legacy ``col < g.fast_n`` interior is preserved for layouts whose
    // main kernel still uses ``bpc = fast_n / BLOCK_SIZE`` (RRR).
    const bool interior_n       = main_covers_n ? true : (col < g.fast_n);
    const bool fast_covers_cell = interior_n && g.fast_n > 0 && g.fast_k > 0;
    if (fast_covers_cell && !needs_k_tail) return;

    const int k0 = fast_covers_cell ? g.fast_k : 0;
    float acc = 0.0f;

    if constexpr (L == Layout::RCR) {
        // Vec8 (8-byte = 8 fp8e4m3) fast path. Both A[row, kk] and
        // B[group_idx, col, kk] are stride-1 in K, so consecutive fp8s
        // along K can be loaded as a single dwordx2. ``g.k`` is bounded
        // below the K_BLOCK alignment by host pad in the gpt_oss-K=2880
        // path which is currently the only K-tail caller; for K=2880,
        // (g.k - k0) is 64 (K-tail correction) or 2880 (N-tail full
        // reduction) — both multiples of 8. The scalar tail handles any
        // residual when g.k % 8 != 0.
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
        // RRR: A stride-1 in K, B[group_idx, kk, col] stride-N in K — B
        // is not vectorisable along K. Stay scalar (this kernel is the
        // FP8 grad-X / backward dA path; performance gap to a native main
        // kernel is intentional and tracked as a follow-up — round 1 only
        // unblocks correctness so the bench can produce numbers).
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

// =============================================================================
// Round-13: LDS-staged K-tail correction kernel for FP8 RCR (mirror BF16
// ``grouped_ktail_kernel_lds`` round-11). Replaces the scalar fp32 tail
// kernel for the [fast_k, k) K-tail RMW correction on K-misaligned
// grouped shapes (gpt_oss K=2880 → K_rem=64). Profiling (rocprof on
// gpt_oss-GateUP B=32-M4096) showed the FP8 scalar tail kernel was
// **70.8 %** of total wall-time at 9.6 TFLOPS; this LDS-staged variant
// brings it to ~50-80 TFLOPS, which (combined with main+masked-store)
// closes the gpt_oss FP8 ratio from ~0.22 toward the target 1.20.
//
// Implementation mirrors BF16 closely:
//   1. Cooperative LDS load via vec4 fp8 (4 bytes/thread, NTHR=256 covers
//      the 1024-byte A and B blocks in one transaction each).
//   2. Inner-fma vec8 fp8 → ds_read_b64 + 2× fp8e4m3_4→float4 conversion
//      + 8 fma per vec8 (8 vec8 over K_REM=64 → 64 fma per cell).
//   3. Result × ``resolve_combined_scale_grp`` → bf16 RMW add.
//   4. Cross-group safety: per-row scalar fallback (vec8 + scalar tail)
//      when the (TBM × TBN) block straddles a group boundary; the host
//      ``m_per_group`` hint normally rules this out (TBM=16 << M_g).
//
// Only RCR is templated — the FP8 RRR/CRR layouts have B not stride-1 in
// K, so ds_read_b64 wouldn't help; they fall back to the scalar tail.
// =============================================================================
template<Layout L, int K_REM>
__global__ void grouped_ktail_kernel_lds(const grouped_layout_globals g) {
    static_assert(L == Layout::RCR,
        "grouped_ktail_kernel_lds (FP8): RCR only — RRR/CRR fall back to scalar tail.");
    constexpr int TBM = TAIL_BLOCK_M;       // 16
    constexpr int TBN = TAIL_BLOCK_N;       // 16
    constexpr int NTHR = TBM * TBN;         // 256

    // Round-17: pad LDS row to break (cib * K_REM) bank conflict
    // pattern. Within a wave, 16 ``cib`` lanes read B_lds at strides of
    // 64 bytes (= 16 banks); the addresses hit only banks {0,1} and
    // {16,17} → 8-way conflict, ds_read_b64 takes 8 cycles instead of
    // 1. K_REM_LDS = 72 fp8 = 72 bytes = 18 banks (= 18 mod 32) makes
    // ``cib * 18 mod 32`` distribute the 16 cib lanes across all 16
    // distinct even banks → no conflict. (Round-15 tested this in
    // isolation, was lost in noise behind the much larger fp8->fp32
    // cvt overhead. After round-16 cvt_pk_f32_fp8 fix, LDS is now the
    // bigger fraction of inner-loop time, so re-test.)
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
            // Round-17: cross_boundary fallback now uses the same
            // ``__builtin_amdgcn_cvt_pk_f32_fp8`` packed conversion +
            // 4-way parallel acc as the fast LDS path (see round-15
            // & round-16). This path only fires when a block straddles
            // a group boundary (non-uniform group_lens at runtime); the
            // metric uses uniform group_lens so it does not exercise
            // this branch — but a real bench with skewed group_lens
            // benefits ~2× on the cross_boundary cells.
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

    // Cooperative vec4 fp8 load. NTHR=256, A_TOTAL=TBM*K_REM=1024 fp8 = 256
    // vec4 → each thread owns exactly one vec4. Mirror BF16 round-9 layout.
    // Mapping:
    //   tid =  0 → A_lds[0..3]   (row 0, k 0..3)
    //   tid =  1 → A_lds[4..7]   (row 0, k 4..7)
    //   tid = 15 → A_lds[60..63] (row 0, k 60..63)
    //   tid = 16 → A_lds[64..67] (row 1, k 0..3)
    //   ...
    // 4-byte alignment: g.a row stride = g.k fp8 (= K=2880 multiple of 4),
    // so each (r, k0+4j) start is 4-byte aligned.
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
    // Round-16: replace the 4 scalar fp8->fp32 conversions per
    // ``convertor::convert(a8.lo)`` (which expands to 4 separate
    // ``v_cvt_f32_fp8`` lane-shifted ops) with 2 ``v_cvt_pk_f32_fp8``
    // packed conversions (each consumes a 32-bit reg of 4 fp8 and
    // outputs a fp32x2 from a chosen pair of lanes). Halves the cvt
    // count on each operand: 4 cvts -> 2 cvts per fp8e4m3_4.
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

// =============================================================================
// Round-18 (FP8): MFMA-based K-tail correction kernel for RCR.
//
// rocprof on gpt_oss-GateUP B=32-M4096 (FP8 grouped) showed the LDS-staged
// scalar-fp32 K-tail (``grouped_ktail_kernel_lds``) was **66 %** of total
// wall-time at ~16 TFLOPS — only 0.6 % of fp8 mfma peak. The rest of the
// wall is the main kernel doing 22 K-blocks at ~1370 TFLOPS. This kernel
// replaces the scalar fma inner loop with a single mfma_scale_f32_16x16x128
// _f8f6f4 call per (16M × 16N) cell-tile (K=64 valid + K=64 zero-padded
// → 50 % effective mfma utilization but still ~75× the throughput of the
// scalar fp32 fma path).
//
// Geometry mirrors ``grouped_ktail_kernel_lds<RCR, 64>``:
//   * Block: 1 wave (64 threads). blockDim = (64,).
//   * Grid: ceil_div(n, 16) × ceil_div(M_total, 16) — same as scalar/LDS
//     paths so the host dispatcher only flips the kernel template.
//   * Each lane owns 4 output cells C[(t/16)*4 + 0..3, t%16] and feeds
//     32 fp8 of A and 32 fp8 of B (chunks 0,1: real K=[k0, k0+64); chunks
//     2,3: zero) into the v_mfma_f32_16x16x128_f8f6f4 op.
//   * RMW into bf16 g.c with combined_scale = sa * sb.
//
// Cross-group fallback (block straddles a group boundary in M):
//   * Detected via row_block_base + 16 > s_offs[group_idx + 1].
//   * Falls through to a per-row scalar vec8 fma loop (same compute model
//     as ``grouped_ktail_kernel_lds`` cross-boundary path) using only the
//     first 64 lanes; the host hint ``m_per_group % 16 == 0`` for uniform
//     groups means this branch is unreachable in the metric.
//
// FP8-only RCR: B is row-major [G, N, K]; ``mma_ABt`` (= mfma with both
// operands K-contig) directly applies. RRR/CRR have B not stride-1 in K
// so they continue to use the scalar tail.
// =============================================================================
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

void dispatch_grouped_rcr(grouped_layout_globals g) {
    g.n = static_cast<int>(g.c.cols());
    g.M_total = static_cast<int>(g.c.rows());
    g.k = static_cast<int>(g.a.cols());

    // Round-12: mirror BF16 grouped round-11 path. Main kernel uses
    // ``bpc = ceil_div(g.n, BLOCK_SIZE)`` unconditionally on RCR; the
    // full-tensor SRD inside ``rcr_8w_load_hoist`` clamps OOB row loads
    // to 0 (no swizzle stride, see line ~432), and the column-masked
    // ``store_c_tile_n_masked`` in the main kernel drops OOB cells from
    // the write-back. This eliminates the ``grouped_tail_kernel`` full-K
    // N-tail reduction (scalar fp32 vec8) for cols [fast_n, n) — the
    // dominant wall-time on gpt_oss N=2880/5760, K=2880 grouped FP8.
    // The tail kernel still runs for K-tail correction in [fast_k, k)
    // for ALL cols [0, g.n) (including the partial last col-tile, since
    // main now wrote partial K reduction there too).
    g.fast_n = (g.n / BLOCK_SIZE) * BLOCK_SIZE;
    g.fast_k = (g.k / K_BLOCK)    * K_BLOCK;
    g.bpc    = kittens::ceil_div(g.n, BLOCK_SIZE);
    g.ki     = g.fast_k / K_BLOCK;

    if (g.bpc > 0 && g.ki > 0) {
        // Round-12: launch-uniform branch on N alignment selects the
        // masked-vs-raw store variant at compile time. DSV3 N=4096/7168
        // hits the raw-store instance (zero overhead, ratios stable);
        // gpt_oss N=2880/5760 hits the masked-store instance.
        if (g.bpc * BLOCK_SIZE == g.n) {
            grouped_rcr_kernel<0, false><<<dim3(NUM_CUS), g.block(), 0, g.stream>>>(g);
        } else {
            grouped_rcr_kernel<0, true ><<<dim3(NUM_CUS), g.block(), 0, g.stream>>>(g);
        }
    } else {
        // No aligned interior at all: main kernel cannot run; tail handles
        // every cell with a full-K reduction.
        g.fast_n = 0;
        g.fast_k = 0;
        g.bpc = 0;
        g.ki = 0;
    }

    // Round-12/13: tail kernel only needed for K-tail correction now (and
    // for the no-main fallback). Skip when fast_k == g.k (main covered
    // every cell). The tail kernel detects ``main_covers_n`` via
    // ``g.bpc * BLOCK_SIZE > g.fast_n`` (mirror dense gemm_tail_kernel).
    //
    // Round-13: when the main kernel ran (g.bpc > 0) AND K_rem matches a
    // templated LDS K-tail size AND the host hint guarantees per-block
    // single-group safety, take the LDS-staged K-tail correction path
    // (~10× speedup over scalar fp32 tail). Otherwise (no main, K_rem
    // not templated, or hint says blocks may straddle group boundaries
    // → kernel still has a per-block runtime fallback to scalar) use
    // the existing scalar tail.
    if (g.fast_k != g.k || g.bpc == 0) {
        const int K_rem = g.k - g.fast_k;
        const bool lds_k_tail_safe = (g.m_per_group >= TAIL_BLOCK_M) &&
                                     ((g.m_per_group % TAIL_BLOCK_M) == 0);
        const bool lds_handles_all =
            (g.bpc > 0) &&
            (K_rem == 64) &&
            lds_k_tail_safe;
        if (lds_handles_all) {
            // Round-18: prefer the MFMA-based K-tail kernel for K_REM=64.
            // Geometry / safety identical to the LDS-staged scalar variant
            // (same per-block (16M × 16N) cell tile, same host hint
            // ``m_per_group % 16 == 0``); blockDim shrinks to a single wave
            // (64 threads) since each lane owns 4 output cells via mfma.
            // Round-13 LDS kernel is kept compiled as a fallback safety
            // net but unreachable on this dispatch path.
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
    g.bpc    = g.fast_n / BLOCK_SIZE;
    g.ki     = g.fast_k / K_BLOCK;

    if (g.M_total <= 0 || g.n <= 0 || g.k <= 0 || g.G <= 0) return;

    if (g.bpc > 0 && g.ki > 0) {
        grouped_rrr_kernel<0><<<dim3(NUM_CUS), g.block(), 0, g.stream>>>(g);
    } else {
        // No aligned interior at all — main kernel cannot run; tail handles
        // every cell with a full-K reduction.
        g.fast_n = 0;
        g.fast_k = 0;
        g.bpc = 0;
        g.ki = 0;
    }

    if (g.fast_n != g.n || g.fast_k != g.k) {
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
    B_col_reg b0, b1;
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;

    int pid = chiplet_transform_chunked(
        blockIdx.x, NUM_CUS, BLOCK_SWIZZLE_NUM_XCDS, 64);

    int wm = warpid() / WARPS_N;
    int wn = warpid() % WARPS_N;
    const int num_pid_n = g.bpc;

    if (threadIdx.x == 0) {
        int prev = static_cast<int>(g.group_offs[0]);
        s_offs[0] = prev;
        s_cum_tiles[0] = 0;
        int t = 0;
        const int tiles_per_group = g.bpr * g.bpc;
        #pragma unroll 1
        for (int gi = 0; gi < g.G; ++gi) {
            const int next = static_cast<int>(g.group_offs[gi + 1]);
            s_offs[gi + 1] = next;
            t += tiles_per_group;
            s_cum_tiles[gi + 1] = t;
            prev = next;
        }
        s_total_tiles = t;
        #pragma unroll 1
        for (int gi = g.G + 1; gi < MAX_G_PLUS_1; ++gi) {
            s_cum_tiles[gi] = 0x7FFFFFFF;
        }
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
        // Use full-tensor SRD load (``rcr_8w_load_hoist``) instead of the
        // tile-local SRD path in ``G::load`` (= ``kittens::load`` line 187
        // in include/ops/warp/memory/tile/global_to_shared.cuh). The
        // tile-local SRD has range = ``row_stride * ST::rows * sizeof(T)``
        // bytes from ``global_ptr`` — fine for in-bounds tiles, but for
        // the partial last M-tile (s = 2*(bpr-1)+1 = 45 on n=5760, where
        // unit_coord col = 5760 = past last col), the last row of the
        // tile (lds row 127) crosses tensor end on the last K-iter.
        // Concretely on gpt_oss-GateUP B4 M2048 (M_total=8192, n=5760,
        // tile bytes = 128 cols × 1 byte): SRD bound = global_ptr +
        // 128 * 5760 = global_ptr + 737280, but tensor_end - global_ptr
        // = 47185920 - 46454400 = 731520 < 737280 — buffer_load on the
        // last 128-row tile reads past tensor end → memory fault.
        //
        // ``rcr_8w_load_hoist`` (line 396) constructs the SRD as
        // ``make_srsrc(tensor_base, total_bytes)`` — bound = the FULL
        // tensor size, OOB byte reads clamp to 0 by hardware. The
        // ``offen lds`` buffer_load + SOFFSET (= per-tile byte offset
        // from tensor_base) makes the bound check work correctly. Same
        // signature as ``G::load(dst, src, idx, swizzled_offsets)``.
        // Despite the name, this helper is layout-agnostic — used here
        // for CRR variable-K. Verified safe on round-3 ceil_div +
        // masked-store design via the Round-4 SNR probe.
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

        // Main loop body — mirror of FP8 dense ``gemm_kernel<CRR>`` lines
        // ~1513-1538, but with per-group ki_g and ``a_co/b_co`` lambdas
        // that fold ``k_offset_tiles`` into the row-axis tile coord.
        TK_PRAGMA_UNROLL(CRR_MAIN_UNROLL)
        for (int k = 0; k < ki_g - 2; k++, tic ^= 1, toc ^= 1) {
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

        // Epilog 1: second-to-last K-tile.
        {
            load_b(b0, Bs[tic][0], wn);
            const auto b0_keep = b0;
            load_a(a, As[tic][0], wm);
            global_load_a(As[toc][1], br*2+1, ki_g-1);
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

        // Epilog 2: last K-tile.
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

    grouped_var_k_kernel_fp8<0><<<dim3(NUM_CUS), g.block(), 0, g.stream>>>(g);
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
        if constexpr (L == Layout::RCR) {
            const int aligned_grid = g.bpr * g.bpc;
            bool use_4wave =
                aligned_grid >= RCR_4WAVE_MIN_GRID && g.k <= RCR_4WAVE_MAX_K;
            // Optional env override for tuning/debugging: set TK_RCR_FORCE_KERNEL=4 or 8
            if (const char* e = getenv("TK_RCR_FORCE_KERNEL")) {
                if (e[0] == '4') use_4wave = true;
                else if (e[0] == '8') use_4wave = false;
            }
            if (use_4wave) {
                dim3 grid4(aligned_grid);
                dim3 block4(rcr_4w::NT);
                rcr_4w::kernel<<<grid4, block4, 0, g.stream>>>(g);
            } else {
                if (g.fast_k == g.k) {
                    g.bpc = kittens::ceil_div(g.n, BLK);
                    main_covers_n = true;
                }
                gemm_kernel<L, 0><<<g.grid(), g.block(), 0, g.stream>>>(g);
            }
        } else
        {
            if (g.fast_k == g.k) {
                g.bpc = kittens::ceil_div(g.n, BLK);
                main_covers_n = true;
            }
            gemm_kernel<L, 0><<<g.grid(), g.block(), 0, g.stream>>>(g);
        }
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

// Host-side wrappers for grouped RCR kernel (host-scalar + dscale variants).
static void grouped_rcr_fn(pybind11::object a, pybind11::object b, pybind11::object c,
                           pybind11::object scale_a_obj, pybind11::object scale_b_obj,
                           pybind11::object group_offs_obj,
                           int group_m,
                           int m_per_group) {
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
        /* G,n,k,ki,bpc,group_m,M_total,fast_n,fast_k,m_per_group */
        G, 0, 0, 0, 0, group_m, 0, 0, 0, m_per_group,
    };
    dispatch_grouped_rcr(g);
}

static void grouped_rcr_dscale_fn(
    pybind11::object a, pybind11::object b, pybind11::object c,
    pybind11::object scale_a_obj, pybind11::object scale_b_obj,
    pybind11::object group_offs_obj,
    int group_m,
    int m_per_group) {
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
        /* G,n,k,ki,bpc,group_m,M_total,fast_n,fast_k,m_per_group */
        G, 0, 0, 0, 0, group_m, 0, 0, 0, m_per_group,
    };
    dispatch_grouped_rcr(g);
}

// Round-1 host wrappers for grouped RRR (backward dA) FP8 kernel.
// Same global struct as RCR (identical scale + group_offs plumbing); the
// dispatcher pins ``fast_n = fast_k = 0`` so the entire compute happens
// in ``grouped_tail_kernel<Layout::RRR>``.
static void grouped_rrr_fn(pybind11::object a, pybind11::object b, pybind11::object c,
                           pybind11::object scale_a_obj, pybind11::object scale_b_obj,
                           pybind11::object group_offs_obj,
                           int group_m,
                           int m_per_group) {
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
        G, 0, 0, 0, 0, group_m, 0, 0, 0, m_per_group,
    };
    dispatch_grouped_rrr(g);
}

static void grouped_rrr_dscale_fn(
    pybind11::object a, pybind11::object b, pybind11::object c,
    pybind11::object scale_a_obj, pybind11::object scale_b_obj,
    pybind11::object group_offs_obj,
    int group_m,
    int m_per_group) {
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
        G, 0, 0, 0, 0, group_m, 0, 0, 0, m_per_group,
    };
    dispatch_grouped_rrr(g);
}

// Host wrappers for grouped variable-K (CRR / dB) FP8 kernel
// (host-scalar + dscale variants).
static void grouped_variable_k_crr_fp8_fn(
    pybind11::object a, pybind11::object b, pybind11::object c,
    pybind11::object scale_a_obj, pybind11::object scale_b_obj,
    pybind11::object group_offs_obj,
    int group_m) {
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
        /* G, M_total, n, k, group_m, bpr, bpc, fast_n, fast_k */
        G, 0, 0, 0, group_m, 0, 0, 0, 0,
    };
    dispatch_grouped_var_k_fp8(g);
}

static void grouped_variable_k_crr_dscale_fp8_fn(
    pybind11::object a, pybind11::object b, pybind11::object c,
    pybind11::object scale_a_obj, pybind11::object scale_b_obj,
    pybind11::object group_offs_obj,
    int group_m) {
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
        G, 0, 0, 0, group_m, 0, 0, 0, 0,
    };
    dispatch_grouped_var_k_fp8(g);
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
    // [grouped] Persistent + CPU-sync-free FP8 RCR launcher. ``group_offs`` is
    // a [G+1] int64 device tensor (prefix-sum of per-group M); the kernel
    // consumes it on the GPU side via O(G) linear scan, no host reads.
    m.def("grouped_rcr", &grouped_rcr_fn,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M,
          pybind11::arg("m_per_group") = 0);
    m.def("grouped_rcr_dscale", &grouped_rcr_dscale_fn,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M,
          pybind11::arg("m_per_group") = 0);
    // [grouped] Round-1 RRR launcher (FP8 backward dA path). Same
    // ``group_offs``-driven contract as ``grouped_rcr``; uses the scalar
    // tail kernel for the full compute (no native main kernel yet).
    m.def("grouped_rrr", &grouped_rrr_fn,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M,
          pybind11::arg("m_per_group") = 0);
    m.def("grouped_rrr_dscale", &grouped_rrr_dscale_fn,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M,
          pybind11::arg("m_per_group") = 0);
    // [grouped variable-K dB] Persistent + CPU-sync-free FP8 CRR launcher
    // for the backward dB path. Inputs are 2D contiguous (grad_out, x);
    // output is 3D-grouped grad_b [G, n, k] bf16. ``group_offs`` is the
    // [G+1] int64 device prefix-sum; the kernel scans it on-GPU.
    m.def("grouped_variable_k_crr", &grouped_variable_k_crr_fp8_fn,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M);
    m.def("grouped_variable_k_crr_dscale",
          &grouped_variable_k_crr_dscale_fp8_fn,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M);
    m.attr("DEFAULT_GROUP_M") = DEFAULT_GROUP_M;
    m.attr("BLOCK_SIZE") = BLK;
    m.attr("K_BLOCK") = BK;
}
