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

    A_reg_t A_tile;
    B_reg_t B_tile_0, B_tile_1;
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
    // For the KI_HINT>0 path we use constexpr num_tiles so the main loop can unroll fully.
    // For the KI_HINT==0 path we use dynamic g.ki and #pragma unroll 2 (limited).

    // Coordinate helpers: coords are in tile units, scaled by ST::rows / ST::cols
    auto a_coord = [&](int spatial, int k) {
        if constexpr (L == Layout::CRR) return coord<ST_A>{0, 0, k, spatial};
        else                            return coord<ST_A>{0, 0, spatial, k};
    };
    auto b_coord = [&](int spatial, int k) {
        if constexpr (L == Layout::RCR) return coord<ST_B>{0, 0, spatial, k};
        else                            return coord<ST_B>{0, 0, k, spatial};
    };

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

    int tic = 0, toc = 1;

    using T = typename st_bf<BLOCK_SIZE, K_STEP, st_32x16_s>::dtype;
    constexpr int bytes_per_thread = st_32x16_s::template bytes_per_thread<T>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile = BLOCK_SIZE * K_STEP * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile/2];
    uint32_t swizzled_offsets_B[memcpy_per_tile/2];
    G::prefill_swizzled_offsets(As[0][0], g.a, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[0][0], g.b, swizzled_offsets_B);

    // P21 Dev D — gated DTL hoist. Layout-isolated via `if constexpr` so the
    // CRR codegen path is byte-identical to the BF16_HOIST_M0=0 build.
    auto bf16_dtl_load = [&]<typename DST>(DST& dst, const _gl& gl, auto coord_,
                                            const uint32_t* swo, i32x4 srd,
                                            const bf16* base, uint32_t lds_off) {
#if BF16_HOIST_M0
        if constexpr (L == Layout::RCR || L == Layout::RRR) {
            bf16_dev_d::load_hoist<NUM_THREADS>(dst, gl, coord_, swo, srd, base, lds_off);
        } else {
            G::load(dst, gl, coord_, swo, srd, base, lds_off);
        }
#else
        G::load(dst, gl, coord_, swo, srd, base, lds_off);
#endif
    };

    // Subtile extraction helpers.
    //
    // P23 S2 Dev C — when RCR_PADDED_B128_MODE=1, both `load(dst, sub)` calls
    // in the RCR branch dispatch into `kittens::load(rt_bf<...,row_l,...>&,
    // const st_subtile<st<bf16,128,64,st_64x32_padded_b128_s>,64,64>&)` at
    //   include/ops/warp/memory/tile/shared_to_register.cuh:29 (load(row_l, ST)).
    // Branch A (subtile >= register, line 50) is the live branch:
    //   ST::underlying_subtile_rows = 64 >= RT::base_tile_rows = 16, AND
    //   ST::underlying_subtile_cols = 32 >= RT::base_tile_cols = 32.
    //   register_subtiles_per_shared_subtile_col = 64/16 = 4,
    //   register_subtiles_per_shared_subtile_row = 32/32 = 1,
    //   ST::subtiles_per_col = 64/64 = 1, ST::subtiles_per_row = 64/32 = 2.
    //   underlying_subtile_stride_bytes = 4096 + 32 = 4128 (carries padding).
    // Dev A (Path A) or Dev B (Path B) is expected to add a new dispatch branch
    // here (or upgrade the existing Branch A) so the b128 LDS-write lane mapping
    // produced by the padded shape is actually consumed without bank conflicts.
    auto load_a_subtile = [&](A_reg_t& dst, auto& smem_tile, int warp_idx) {
        if constexpr (L == Layout::CRR) {
            auto sub = subtile_inplace<K_STEP, HALF_REG_BLOCK_M>(smem_tile, {0, warp_idx});
            load(dst, sub);
        } else {
            auto sub = subtile_inplace<HALF_REG_BLOCK_M, K_STEP>(smem_tile, {warp_idx, 0});
            // <<< Dev A/B b128 entry point will be needed HERE for RCR_PADDED_B128_MODE=1 >>>
            load(dst, sub);
        }
    };
    auto load_b_subtile = [&](B_reg_t& dst, auto& smem_tile, int warp_idx) {
        if constexpr (L == Layout::RCR) {
            auto sub = subtile_inplace<HALF_REG_BLOCK_N, K_STEP>(smem_tile, {warp_idx, 0});
            // <<< Dev A/B b128 entry point will be needed HERE for RCR_PADDED_B128_MODE=1 >>>
            load(dst, sub);
        } else {
            auto sub = subtile_inplace<K_STEP, HALF_REG_BLOCK_N>(smem_tile, {0, warp_idx});
            load(dst, sub);
        }
    };

    // MMA dispatch
    // For CRR: use mma_AtB directly (A in col_l, B in col_l) — no register transpose needed.
    // mma_AtB_base uses the same hardware instruction as mma_AB_base (mfma_f32_16x16x32_bf16)
    // but interprets A as transposed, eliminating the register shuffle overhead.
    // Expanded inline so the outer #pragma unroll on the main_loop_iter lambda
    // can see through to the base MMA calls (matches JIT path).
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
    G::load(Bs[tic][0], g.b, b_coord(col*2, 0), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_00);
    G::load(As[tic][0], g.a, a_coord(row*2, 0), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_00);
    G::load(Bs[tic][1], g.b, b_coord(col*2+1, 0), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_01);
    G::load(As[tic][1], g.a, a_coord(row*2+1, 0), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_01);

    if (warp_row == 1) { __builtin_amdgcn_s_barrier(); }
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    G::load(Bs[toc][0], g.b, b_coord(col*2, 1), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_10);
    G::load(As[toc][0], g.a, a_coord(row*2, 1), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_10);
    G::load(Bs[toc][1], g.b, b_coord(col*2+1, 1), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_11);

    asm volatile("s_waitcnt vmcnt(6)");
    __builtin_amdgcn_s_barrier();

    /********** Main loop **********/
    auto main_loop_iter = [&](int tile) {
        load_b_subtile(B_tile_0, Bs[0][0], warp_col);
        load_a_subtile(A_tile, As[0][0], warp_row);
        G::load(As[1][1], g.a, a_coord(row*2+1, tile+1), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_11);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load_b_subtile(B_tile_1, Bs[0][1], warp_col);
        G::load(Bs[0][0], g.b, b_coord(col*2, tile+2), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_00);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a_subtile(A_tile, As[0][1], warp_row);
        G::load(As[0][0], g.a, a_coord(row*2, tile+2), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_00);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load_b_subtile(B_tile_0, Bs[1][0], warp_col);
        G::load(Bs[0][1], g.b, b_coord(col*2+1, tile+2), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_01);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a_subtile(A_tile, As[1][0], warp_row);
        G::load(As[0][1], g.a, a_coord(row*2+1, tile+2), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_01);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load_b_subtile(B_tile_1, Bs[1][1], warp_col);
        G::load(Bs[1][0], g.b, b_coord(col*2, tile+3), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_10);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a_subtile(A_tile, As[1][1], warp_row);
        G::load(As[1][0], g.a, a_coord(row*2, tile+3), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_10);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        G::load(Bs[1][1], g.b, b_coord(col*2+1, tile+3), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_11);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        DO_MMA(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    };

    // Matching the JIT-path schedule:
    //   RCR/RRR: use full #pragma unroll (compile-time KI_HINT).
    //   CRR:     use #pragma unroll 2 (hides barrier latency). Some KI values
    //            (128, 172, 296) see 7-26 SGPR spills under unroll 2, but the
    //            barrier-hiding benefit still outweighs the spill cost.
    // KI_HINT==0 dynamic fallback always uses unroll 2.
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
        const int num_tiles = g.ki;
        #pragma unroll 2
        for (int tile = 0; tile < num_tiles - 2; tile += 2) main_loop_iter(tile);
    }

    /********** Epilog 1: second-to-last K-tile pair **********/
    {
        const int tile = (KI_HINT > 0) ? (KI_HINT - 2) : (g.ki - 2);
        load_b_subtile(B_tile_0, Bs[tic][0], warp_col);
        load_a_subtile(A_tile, As[tic][0], warp_row);
        G::load(As[toc][1], g.a, a_coord(row*2+1, tile+1), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_11);
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

    if (warp_row == 0) { __builtin_amdgcn_s_barrier(); }

    store(g.c, C_accum[0][0], {0, 0,
        (row * 2) * WARPS_M + warp_row,
        col * 2 * WARPS_N + warp_col});
    store(g.c, C_accum[0][1], {0, 0,
        (row * 2) * WARPS_M + warp_row,
        col * 2 * WARPS_N + WARPS_N + warp_col});
    store(g.c, C_accum[1][0], {0, 0,
        (row * 2) * WARPS_M + WARPS_M + warp_row,
        col * 2 * WARPS_N + warp_col});
    store(g.c, C_accum[1][1], {0, 0,
        (row * 2) * WARPS_M + WARPS_M + warp_row,
        col * 2 * WARPS_N + WARPS_N + warp_col});
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

    const bool interior_mn      = row < g.fast_m && col < g.fast_n;
    const bool fast_covers_cell = interior_mn && g.fast_m > 0 &&
                                  g.fast_n > 0 && g.fast_k > 0;
    const bool needs_k_tail     = g.fast_k < g.k;
    if (fast_covers_cell && !needs_k_tail) {
        return;
    }

    const int k0 = fast_covers_cell ? g.fast_k : 0;
    float acc = 0.0f;
    for (int kk = k0; kk < g.k; ++kk) {
        if constexpr (L == Layout::RCR) {
            acc += load_bf16_scalar(g.a, row, kk) * load_bf16_scalar(g.b, col, kk);
        } else if constexpr (L == Layout::RRR) {
            acc += load_bf16_scalar(g.a, row, kk) * load_bf16_scalar(g.b, kk, col);
        } else { // CRR
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
    g.bpc = g.fast_n / BLOCK_SIZE;
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

    if (g.fast_m != g.m || g.fast_n != g.n || g.fast_k != g.k) {
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
    int ki;                      // K / K_STEP
    int bpc;                     // n / BLOCK_SIZE  (constant across groups)
    int group_m;                 // tile-scheduling super-block factor
    int num_xcds;                // XCD swizzle factor
    int M_total;                 // sum of group sizes (= a.shape[0])
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

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

    A_reg_t A_tile;
    B_reg_t B_tile_0, B_tile_1;
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

    // [grouped] SRD setup. Bounds span the FULL A and B tensors (across all
    // groups) so the same SRD is valid across persistent iterations.
    const bf16* a_base = (bf16*)&g.a[{0, 0, 0, 0}];
    const bf16* b_base = (bf16*)&g.b[{0, 0, 0, 0}];
    const int a_row_stride = g.a.template stride<2>() * sizeof(bf16);
    const int b_row_stride = g.b.template stride<2>() * sizeof(bf16);
    // For "normal" A layout (M×K): A_total_rows = M_total. For CRR A (K×M_total):
    // A_total_rows = K (M dimension lives on the col axis).
    const int a_total_rows = (L == Layout::CRR) ? g.k : g.M_total;
    // For B (3D [G, N, K] (RCR) or [G, K, N] (RRR/CRR)): SRD must cover all
    // groups. ``b_row_stride`` already encodes the inner-row pitch (K or N).
    // The total row count is groups × per-group rows.
    const int b_total_rows = (L == Layout::RCR) ? (g.G * g.n) : (g.G * g.k);
    i32x4 a_srsrc_base = make_srsrc(a_base, a_total_rows * a_row_stride, a_row_stride);
    i32x4 b_srsrc_base = make_srsrc(b_base, b_total_rows * b_row_stride, b_row_stride);

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

        // Coordinate helpers — identical structure to dense kernel, but A/C
        // shift by m_subtile_* and B uses ``group_idx`` as the depth dim.
        auto a_coord = [&](int spatial, int kk) {
            if constexpr (L == Layout::CRR)
                return coord<ST_A>{0, 0, kk, m_subtile_A + spatial};
            else
                return coord<ST_A>{0, 0, m_subtile_A + spatial, kk};
        };
        auto b_coord = [&](int spatial, int kk) {
            if constexpr (L == Layout::RCR)
                return coord<ST_B>{0, group_idx, spatial, kk};
            else
                return coord<ST_B>{0, group_idx, kk, spatial};
        };

        // Reset accumulators + double-buffer indices for this tile.
        zero(C_accum[0][0]); zero(C_accum[0][1]);
        zero(C_accum[1][0]); zero(C_accum[1][1]);
        int tic = 0, toc = 1;

        auto bf16_dtl_load = [&]<typename DST>(DST& dst, const _gl& gl, auto coord_,
                                                const uint32_t* swo, i32x4 srd,
                                                const bf16* base, uint32_t lds_off) {
            G::load(dst, gl, coord_, swo, srd, base, lds_off);
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
        G::load(Bs[tic][0], g.b, b_coord(col*2, 0), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_00);
        G::load(As[tic][0], g.a, a_coord(row*2, 0), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_00);
        G::load(Bs[tic][1], g.b, b_coord(col*2+1, 0), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_01);
        G::load(As[tic][1], g.a, a_coord(row*2+1, 0), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_01);

        if (warp_row == 1) { __builtin_amdgcn_s_barrier(); }
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        G::load(Bs[toc][0], g.b, b_coord(col*2, 1), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_10);
        G::load(As[toc][0], g.a, a_coord(row*2, 1), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_10);
        G::load(Bs[toc][1], g.b, b_coord(col*2+1, 1), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_11);

        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        /********** Main loop **********/
        auto main_loop_iter = [&](int tile) {
            load_b_subtile(B_tile_0, Bs[0][0], warp_col);
            load_a_subtile(A_tile, As[0][0], warp_row);
            G::load(As[1][1], g.a, a_coord(row*2+1, tile+1), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_11);
            asm volatile("s_waitcnt lgkmcnt(8)");
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            DO_MMA(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load_b_subtile(B_tile_1, Bs[0][1], warp_col);
            G::load(Bs[0][0], g.b, b_coord(col*2, tile+2), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_00);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            DO_MMA(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a_subtile(A_tile, As[0][1], warp_row);
            G::load(As[0][0], g.a, a_coord(row*2, tile+2), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_00);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            DO_MMA(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load_b_subtile(B_tile_0, Bs[1][0], warp_col);
            G::load(Bs[0][1], g.b, b_coord(col*2+1, tile+2), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_01);
            asm volatile("s_waitcnt vmcnt(6)");
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(1);
            DO_MMA(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a_subtile(A_tile, As[1][0], warp_row);
            G::load(As[0][1], g.a, a_coord(row*2+1, tile+2), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_01);
            asm volatile("s_waitcnt lgkmcnt(8)");
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            DO_MMA(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load_b_subtile(B_tile_1, Bs[1][1], warp_col);
            G::load(Bs[1][0], g.b, b_coord(col*2, tile+3), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_10);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            DO_MMA(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a_subtile(A_tile, As[1][1], warp_row);
            G::load(As[1][0], g.a, a_coord(row*2, tile+3), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_10);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            DO_MMA(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            G::load(Bs[1][1], g.b, b_coord(col*2+1, tile+3), swizzled_offsets_B, b_srsrc_base, b_base, b_lds_11);
            asm volatile("s_waitcnt vmcnt(6)");
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(1);
            DO_MMA(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        };

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
            const int num_tiles = g.ki;
            #pragma unroll 2
            for (int tile = 0; tile < num_tiles - 2; tile += 2) main_loop_iter(tile);
        }

        /********** Epilog 1: second-to-last K-tile pair **********/
        {
            const int tile = (KI_HINT > 0) ? (KI_HINT - 2) : (g.ki - 2);
            load_b_subtile(B_tile_0, Bs[tic][0], warp_col);
            load_a_subtile(A_tile, As[tic][0], warp_row);
            G::load(As[toc][1], g.a, a_coord(row*2+1, tile+1), swizzled_offsets_A, a_srsrc_base, a_base, a_lds_11);
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

        if (warp_row == 0) { __builtin_amdgcn_s_barrier(); }

        // [grouped] Store with C row shifted by m_subtile_C.
        store(g.c, C_accum[0][0], {0, 0,
            m_subtile_C + (row * 2) * WARPS_M + warp_row,
            col * 2 * WARPS_N + warp_col});
        store(g.c, C_accum[0][1], {0, 0,
            m_subtile_C + (row * 2) * WARPS_M + warp_row,
            col * 2 * WARPS_N + WARPS_N + warp_col});
        store(g.c, C_accum[1][0], {0, 0,
            m_subtile_C + (row * 2) * WARPS_M + WARPS_M + warp_row,
            col * 2 * WARPS_N + warp_col});
        store(g.c, C_accum[1][1], {0, 0,
            m_subtile_C + (row * 2) * WARPS_M + WARPS_M + warp_row,
            col * 2 * WARPS_N + WARPS_N + warp_col});

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
    g.ki = g.k / K_STEP;
    g.bpc = g.n / BLOCK_SIZE;

    switch (g.ki) {
        case 56:  launch_one_grouped<L, 56> (g); return;
        case 64:  launch_one_grouped<L, 64> (g); return;
        case 112: launch_one_grouped<L, 112>(g); return;
        case 128: launch_one_grouped<L, 128>(g); return;
        case 172: launch_one_grouped<L, 172>(g); return;
        case 224: launch_one_grouped<L, 224>(g); return;
        case 256: launch_one_grouped<L, 256>(g); return;
        case 296: launch_one_grouped<L, 296>(g); return;
        case 448: launch_one_grouped<L, 448>(g); return;
        case 462: launch_one_grouped<L, 462>(g); return;
        case 832: launch_one_grouped<L, 832>(g); return;
        default:  launch_one_grouped<L, 0>  (g); return;
    }
}

static void grouped_dispatch(pybind11::object a, pybind11::object b, pybind11::object c,
                             pybind11::object group_offs, int gm, int num_xcds,
                             const char* layout_name) {
    auto group_offs_ptr = group_offs.attr("data_ptr")().cast<uintptr_t>();
    int G = group_offs.attr("numel")().cast<int>() - 1;

    grouped_layout_globals g{
        py::from_object<_gl>::make(a),
        py::from_object<_gl>::make(b),
        py::from_object<_gl>::make(c),
        reinterpret_cast<const int64_t*>(group_offs_ptr),
        {},
        G, 0, 0, 0, 0, gm, num_xcds, 0,
    };

    if (layout_name[0] == 'r' && layout_name[1] == 'c') dispatch_grouped<Layout::RCR>(g);
    else if (layout_name[0] == 'r' && layout_name[1] == 'r') dispatch_grouped<Layout::RRR>(g);
    else dispatch_grouped<Layout::CRR>(g);
}

static void grouped_rcr_fn(pybind11::object a, pybind11::object b, pybind11::object c,
                           pybind11::object group_offs, int gm, int num_xcds) {
    grouped_dispatch(a, b, c, group_offs, gm, num_xcds, "rcr");
}
static void grouped_rrr_fn(pybind11::object a, pybind11::object b, pybind11::object c,
                           pybind11::object group_offs, int gm, int num_xcds) {
    grouped_dispatch(a, b, c, group_offs, gm, num_xcds, "rrr");
}
static void grouped_crr_fn(pybind11::object a, pybind11::object b, pybind11::object c,
                           pybind11::object group_offs, int gm, int num_xcds) {
    grouped_dispatch(a, b, c, group_offs, gm, num_xcds, "crr");
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
          "group_offs"_a, "group_m"_a=4, "num_xcds"_a=8);
    m.def("grouped_rrr", &grouped_rrr_fn, "a"_a, "b"_a, "c"_a,
          "group_offs"_a, "group_m"_a=4, "num_xcds"_a=8);
    m.def("grouped_crr", &grouped_crr_fn, "a"_a, "b"_a, "c"_a,
          "group_offs"_a, "group_m"_a=4, "num_xcds"_a=8);
    m.attr("BLOCK_SIZE") = BLOCK_SIZE;
    m.attr("K_STEP") = K_STEP;
}
