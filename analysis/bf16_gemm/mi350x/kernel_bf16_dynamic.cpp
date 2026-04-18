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
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

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

    // Subtile extraction helpers
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
    g.ki = g.k / K_STEP;
    g.bpr = g.m / BLOCK_SIZE;
    g.bpc = g.n / BLOCK_SIZE;

    switch (g.ki) {
        case 56:  launch_one<L, 56> (g); return;
        case 64:  launch_one<L, 64> (g); return;
        case 128: launch_one<L, 128>(g); return;
        case 172: launch_one<L, 172>(g); return;
        case 224: launch_one<L, 224>(g); return;
        case 256: launch_one<L, 256>(g); return;
        case 296: launch_one<L, 296>(g); return;
        case 448: launch_one<L, 448>(g); return;
        case 462: launch_one<L, 462>(g); return;
        case 832: launch_one<L, 832>(g); return;
        default:  launch_one<L, 0>  (g); return;
    }
}

static void gemm_dispatch(pybind11::object a, pybind11::object b, pybind11::object c,
                          int gm, int num_xcds, const char* layout_name) {
    auto c_gl = py::from_object<_gl>::make(c);
    layout_globals g{py::from_object<_gl>::make(a), py::from_object<_gl>::make(b),
                     c_gl, {},
                     0, 0, 0, 0, 0, 0, gm, num_xcds};

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

PYBIND11_MODULE(tk_bf16_layouts, m) {
    using namespace pybind11::literals;
    m.def("gemm_rcr", &rcr, "a"_a, "b"_a, "c"_a, "group_m"_a=4, "num_xcds"_a=8);
    m.def("gemm_rrr", &rrr, "a"_a, "b"_a, "c"_a, "group_m"_a=4, "num_xcds"_a=8);
    m.def("gemm_crr", &crr, "a"_a, "b"_a, "c"_a, "group_m"_a=4, "num_xcds"_a=8);
    m.attr("BLOCK_SIZE") = BLOCK_SIZE;
    m.attr("K_STEP") = K_STEP;
}
