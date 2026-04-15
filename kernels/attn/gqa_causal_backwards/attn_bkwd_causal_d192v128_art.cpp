// ART-based Backward Kernel for Asymmetric Attention (D_QK=192, D_V=128)
// Uses Assigned Register Tiles (ART) for explicit VGPR/AGPR register management
// to eliminate gfx950 AGPR aliasing and compiler register allocation issues.
//
// Single fused kernel computing dK, dV, and dQ simultaneously.
// Based on the reference 128/128 backward kernel (attn_bkwd_causal.cpp).
//
// Key parameters:
//   WARP_SIZE_KV = 32 (reduced from 64 to fit D_QK=192)
//   DOT_SLICE_QO = 16
//   STEP_QO = 64 (= 4 × DOT_SLICE_QO)
//   BLOCK_SIZE_KV = 128 (= 4 warps × 32)
//   Grid: KV-parallel dim3(ATTN_H_KV, ATTN_N/BLOCK_SIZE_KV, ATTN_B)

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "utils.cpp"

// Custom atomic bf16 add for dQ with arbitrary stride (D_QK=192).
// Uses flat pointer atomics instead of buffer atomics.
//
// dQ_i_T is [32×16] col_l (mma_AtB accumulator). Its row_l transposed view
// dQ_i is [16×32]. The physical data follows col_l layout:
//   For each 16x16 tile: R[k] at (row=(laneid/16)*4+k, col=laneid%16)
//   dQ_i_T(r,c) = dQ(c,r) so dQ_row = laneid%16, dQ_col = col_offset + tile*16 + (laneid/16)*4 + k
//
// col_offset_elems = starting column in dQ for this 32-col block.
template<int axis, ducks::art::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void atomic_pk_add_bf16_2d(const GL &dst, const RT &src,
    const COORD &idx, int col_offset_elems) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;
    static_assert(std::is_same_v<U, bf16>, "only bf16 supported");
    static_assert(std::is_same_v<T, float>, "only float accumulators supported");

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    const int laneid = kittens::laneid();

    const int dQ_row = laneid % 16;
    const int dQiT_row_group = (laneid / 16) * 4;

    auto perform_atomic = [&]<int N, int M>() {
        using range_type = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
        static_assert(range_type::lo + 3 == range_type::hi, "need 4 consecutive registers");
        static_assert(range_type::hi < 256, "must be VGPRs");

        const int dQ_col_base = col_offset_elems + M * 16 + dQiT_row_group;

        // Read 4 floats from pinned ART registers, convert to bf16 pairs
        float f0 = macros::v_mov_b32_p2up<range_type::lo, float>();
        float f1 = macros::v_mov_b32_p2up<range_type::lo + 1, float>();
        float f2 = macros::v_mov_b32_p2up<range_type::lo + 2, float>();
        float f3 = macros::v_mov_b32_p2up<range_type::lo + 3, float>();

        // Pack into bf16 pairs and atomic-add via global_atomic_pk_add_bf16
        U *p = dst_ptr + dQ_row * row_stride + dQ_col_base;

        // Pack f0,f1 → bf16x2
        uint32_t pk0;
        asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(pk0) : "v"(f0), "v"(f1));
        asm volatile("global_atomic_pk_add_bf16 %0, %1, off" :: "v"(p), "v"(pk0) : "memory");

        // Pack f2,f3 → bf16x2
        uint32_t pk1;
        asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(pk1) : "v"(f2), "v"(f3));
        asm volatile("global_atomic_pk_add_bf16 %0, %1, off offset:4" :: "v"(p), "v"(pk1) : "memory");
    };

    [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
        ([&]<std::size_t N>() {
            [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                ([&]<std::size_t M>() {
                    perform_atomic.template operator()<N, M>();
                }.template operator()<Ms>(), ...);
            }(std::make_index_sequence<RT::width>{});
        }.template operator()<Ns>(), ...);
    }(std::make_index_sequence<RT::height>{});
}

#ifndef ATTN_B
constexpr int ATTN_B = 16;
#endif
#ifndef ATTN_H
constexpr int ATTN_H = 64;
#endif
#ifndef ATTN_H_KV
constexpr int ATTN_H_KV = 8;
#endif
constexpr int GROUP_SIZE = ATTN_H / ATTN_H_KV;
#ifndef ATTN_N
constexpr int ATTN_N = 4096;
#endif

constexpr int D_QK = 192;
constexpr int D_V  = 128;

constexpr int STEP_QO = 64;
constexpr int BLOCK_SIZE_KV = 128;
constexpr int SLICE_QO = 32;         // Not used directly (reference compat)
constexpr int DOT_SLICE_QO = 16;
constexpr int WARP_SIZE_KV = 32;     // Reduced from 64 for D_QK=192
constexpr bool causal = true;

#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using G = kittens::group<NUM_WARPS>;
using namespace kittens;

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
struct attn_bwd_d192v128_globals {
    gl<bf16, -1, -1, -1, -1> Q, K, V;
    gl<bf16, -1, -1, -1, -1> dOg, dQg, dKg, dVg;
    gl<float, -1, -1, -1, -1> L_vec, delta_vec;
    hipStream_t stream;
    dim3 grid()  { return dim3(ATTN_H_KV, (ATTN_N / BLOCK_SIZE_KV), ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
__launch_bounds__(NUM_THREADS, 1)
__global__ __attribute__((amdgpu_num_vgpr(29)))
void attend_bwd_d192v128_ker(const attn_bwd_d192v128_globals g) {

    const int kv_head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int first_q_head = kv_head_idx * GROUP_SIZE;

    const int warpid = kittens::warpid();
    const int j = seq_idx * NUM_WARPS + warpid;

    const int total_steps_per_head = ATTN_N / STEP_QO;
    const int j_min = seq_idx * NUM_WARPS;
    const int k_start_min = j_min * WARP_SIZE_KV;
    const int first_step = max(0, k_start_min / STEP_QO);
    const int num_steps_per_head = total_steps_per_head - first_step;
    const int num_steps = num_steps_per_head * GROUP_SIZE;
    const int k_pos = j * WARP_SIZE_KV;

    constexpr float L_SCALE_FACTOR = 1.44269504089f;
    constexpr float P_SCALE_FACTOR = 0.07216878365f * 1.44269504089f; // 1/sqrt(192) * log2(e)
    constexpr float dP_SCALE_FACTOR = 0.07216878365f;                 // 1/sqrt(192)

    // =====================================================================
    // Shared memory
    // =====================================================================
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // K in shared: [BLOCK_SIZE_KV × D_QK] = [128 × 192] = 48KB
    st_bf<BLOCK_SIZE_KV, D_QK, st_16x16_s> (&K_j_smem) = al.allocate<st_bf<BLOCK_SIZE_KV, D_QK, st_16x16_s>>();
    // Q double-buffered: [2][2] × [16 × 192] (2 buffers, 2 halves of STEP_QO=64)
    // Each st_bf<DOT_SLICE_QO=16, D_QK=192, st_16x32_s> but need to check if st_16x32_s works for D_QK=192
    // Actually for D_QK=192, we need st_16x32_s with width=192/32=6 tiles
    // SLICE_QO=32 = 2 × DOT_SLICE_QO: each half holds 2 consecutive dot-slices
    st_bf<SLICE_QO, D_QK, st_16x32_s> (&Q_i_smem)[2][2] = al.allocate<st_bf<SLICE_QO, D_QK, st_16x32_s>, 2, 2>();
    // dO double-buffered: [2][2] × [32 × 128]
    st_bf<SLICE_QO, D_V, st_16x32_s> (&dO_i_smem)[2][2] = al.allocate<st_bf<SLICE_QO, D_V, st_16x32_s>, 2, 2>();
    // Attention/dS transpose buffer: [BLOCK_SIZE_KV × DOT_SLICE_QO] = [128 × 16]
    st_bf<BLOCK_SIZE_KV, DOT_SLICE_QO, st_16x16_swizzled_s> (&attn_i_smem) = al.allocate<st_bf<BLOCK_SIZE_KV, DOT_SLICE_QO, st_16x16_swizzled_s>>();
    // L and delta double-buffered
    sv_fl<STEP_QO> (&L_smem)[2] = al.allocate<sv_fl<STEP_QO>, 2>();
    sv_fl<STEP_QO> (&delta_smem)[2] = al.allocate<sv_fl<STEP_QO>, 2>();

    // =====================================================================
    // ART Register Range Declarations
    // =====================================================================
    // Physical registers: 0-255 = VGPR, 256-511 = AGPR
    // __attribute__((amdgpu_num_vgpr(29))) reserves v[0:28] for compiler

    // --- Persistent accumulators ---
    // dK: [D_QK × WARP_SIZE_KV] = [192 × 32] col_l rt_32x32_s → 6 tiles × 16 = 96 AGPRs
    using dK_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<256, 351>>, 16>;  // a[0:95]
    // dV: [D_V × WARP_SIZE_KV] = [128 × 32] col_l rt_32x32_s → 4 tiles × 16 = 64 VGPRs
    using dV_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<128, 191>>, 16>;  // v[128:191]

    // --- K and V in registers (loaded from shared/global each iteration) ---
    // K_j: [WARP_SIZE_KV × D_QK] = [32 × 192] row_l rt_16x32_s → 2×6 = 12 tiles × 4 = 48 regs
    using K_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<352, 399>>, 4>;  // a[96:143]
    // V_j: [WARP_SIZE_KV × D_V] = [32 × 128] row_l rt_16x32_s → 2×4 = 8 tiles × 4 = 32 regs
    // OVERLAPS with dS_T_col/K_j_col (Phase 3 vs Phase 8-10, never simultaneous)
    using V_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<192, 223>>, 4>;  // v[192:223] (overlaps dS_T + K_col)

    // --- Q and dO tiles (loaded per dot-slice) ---
    // Q_i: [DOT_SLICE_QO × D_QK] = [16 × 192] row_l rt_16x32_s → 1×6 = 6 tiles × 4 = 24 regs
    using Q_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<432, 455>>, 4>;  // a[176:199]
    // dO_i: [DOT_SLICE_QO × D_V] = [16 × 128] row_l rt_16x32_s → 1×4 = 4 tiles × 4 = 16 regs
    using dO_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<62, 77>>, 4>;  // v[62:77]
    // dO_i_col: [DOT_SLICE_QO × D_V] = [16 × 128] col_l → 16 regs
    using dO_col_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<78, 93>>, 4>;  // v[78:93], 4 tiles of 4 regs (rt_16x32_s)
    // dO_i_col_16x16: same regs but split as rt_16x16_s (2 regs per tile, 8 tiles)
    using dO_col_16x16_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<78, 93>>, 2>;  // v[78:93], 8 tiles of 2 regs
    // Q_i_col: [DOT_SLICE_QO × D_QK] = [16 × 192] col_l → 24 regs
    using Q_col_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<94, 117>>, 4>;  // v[94:117], 6 tiles of 4 regs (rt_16x32_s)
    // Q_i_col_16x16: same regs but split as rt_16x16_s (2 regs per tile, 12 tiles)
    using Q_col_16x16_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<94, 117>>, 2>;  // v[94:117], 12 tiles of 2 regs

    // --- Attention tiles (temporary, heavily aliased) ---
    // P_ij: [DOT_SLICE_QO × WARP_SIZE_KV] = [16 × 32] col_l float → 2 tiles × 4 = 8 regs
    using P_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<30, 37>>, 4>;  // v[30:37]
    // dP_ij: [16 × 32] col_l float → 8 regs (can overlap K after K is consumed)
    using dP_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<38, 45>>, 4>;  // v[38:45]
    // P_bf16: [16 × 32] col_l bf16 → 4 regs
    using P_bf16_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<46, 49>>, 2>;  // v[46:49]
    // dP_bf16: [16 × 32] col_l bf16 → 4 regs
    using dP_bf16_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<50, 53>>, 2>;  // v[50:53]
    // P_bf16_col: for mma_AtB dV
    using P_bf16_col_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<46, 49>>, 4>;  // same as P_bf16 (aliased)
    // dP_bf16_col: for mma_AtB dK
    using dP_bf16_col_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<50, 53>>, 4>;  // same as dP_bf16

    // dS for dQ: [BLOCK_SIZE_KV × DOT_SLICE_QO] = [128 × 16] col_l rt_32x16_4_s
    // height=4, width=1 → 4 tiles × 4 regs = 16 regs
    using dS_T_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<192, 207>>, 4>;  // v[192:207], 16 regs

    // K_j_col: [BLOCK_SIZE_KV × 32] col_l rt_32x16_4_s for dQ mma_AtB
    // height=4, width=2 → 8 tiles × 4 regs = 32 regs
    using K_col_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<208, 239>>, 4>;  // v[208:239]

    // dQ_i_T: [32 × 16] col_l float → 8 regs
    using dQ_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<118, 125>>, 4>;  // v[118:125]

    // dP_bf16_accum_row (for shared store): row_l view of dP_bf16
    using dP_bf16_accum_row_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<50, 53>>, 4>;  // same physical regs

    // P_bf16_accum_row (for shared store): row_l view of P_bf16
    using P_bf16_accum_row_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<46, 49>>, 4>;  // same physical regs as P_bf16

    // --- dK epilogue chunk tiles (for AGPR->VGPR copy and store) ---
    // Chunk 2: dK tiles 4-5 (a[64:95]) → v[128:159], [64×32] col_l, row view [32×64]
    using dK_chunk2_col_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<128, 159>>, 16>;  // v[128:159], 2 tiles
    using dK_chunk2_row_ranges = ducks::art::transpose_2d<dK_chunk2_col_ranges, 2, 1>;

    // --- Clobber all ranges ---
    ducks::art::clobber<dK_ranges>();
    ducks::art::clobber<dV_ranges>();
    ducks::art::clobber<K_ranges>();
    ducks::art::clobber<V_ranges>();
    ducks::art::clobber<Q_ranges>();
    ducks::art::clobber<dO_ranges>();
    ducks::art::clobber<dO_col_ranges>();
    ducks::art::clobber<Q_col_ranges>();
    ducks::art::clobber<P_ranges>();
    ducks::art::clobber<dP_ranges>();
    ducks::art::clobber<P_bf16_ranges>();
    ducks::art::clobber<dP_bf16_ranges>();
    ducks::art::clobber<dS_T_ranges>();
    ducks::art::clobber<K_col_ranges>();
    ducks::art::clobber<dQ_ranges>();
    ducks::art::clobber<dK_chunk2_col_ranges>();

    // --- Declare ART tiles ---
    // Persistent accumulators
    art<float, D_QK, WARP_SIZE_KV, col_l, rt_32x32_s, dK_ranges> dK_j_T;  // 96 AGPRs
    art<float, D_V,  WARP_SIZE_KV, col_l, rt_32x32_s, dV_ranges> dV_j_T;  // 64 VGPRs

    // K and V register tiles
    art<bf16, WARP_SIZE_KV, D_QK, row_l, rt_16x32_s, K_ranges> K_j;  // 48 AGPRs
    art<bf16, WARP_SIZE_KV, D_V,  row_l, rt_16x32_s, V_ranges> V_j;  // 32 AGPRs

    // Q and dO tiles
    art<bf16, DOT_SLICE_QO, D_QK, row_l, rt_16x32_s, Q_ranges> Q_i;      // 24 AGPRs
    art<bf16, DOT_SLICE_QO, D_V,  row_l, rt_16x32_s, dO_ranges> dO_i;    // 16 VGPRs
    art<bf16, DOT_SLICE_QO, D_V,  col_l, rt_16x32_s, dO_col_ranges> dO_i_col;  // 16 VGPRs
    art<bf16, DOT_SLICE_QO, D_V,  col_l, rt_16x16_s, dO_col_16x16_ranges> dO_i_col_16x16;  // same regs, 8 tiles
    art<bf16, DOT_SLICE_QO, D_QK, col_l, rt_16x32_s, Q_col_ranges> Q_i_col;    // 24 VGPRs
    art<bf16, DOT_SLICE_QO, D_QK, col_l, rt_16x16_s, Q_col_16x16_ranges> Q_i_col_16x16;  // same regs, 12 tiles

    // Attention tiles
    art<float, DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s, P_ranges> P_ij;    // 8 VGPRs
    art<float, DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s, dP_ranges> dP_ij;  // 8 VGPRs
    art<bf16, DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s, P_bf16_ranges> P_ij_bf16;
    art<bf16, DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s, dP_bf16_ranges> dP_ij_bf16;
    art<bf16, DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x32_s, P_bf16_col_ranges> P_ij_bf16_col;
    art<bf16, DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x32_s, dP_bf16_col_ranges> dP_ij_bf16_col;

    // dS transposed (for dQ): [BLOCK_SIZE_KV × DOT_SLICE_QO] col_l
    art<bf16, BLOCK_SIZE_KV, DOT_SLICE_QO, col_l, rt_32x16_4_s, dS_T_ranges> dS_T_col;

    // K_j_col: [BLOCK_SIZE_KV × 32] col_l (for dQ mma_AtB)
    art<bf16, BLOCK_SIZE_KV, 32, col_l, rt_32x16_4_s, K_col_ranges> K_j_col;

    // dQ accumulator: [32 × DOT_SLICE_QO] col_l float
    art<float, 32, DOT_SLICE_QO, col_l, rt_16x16_s, dQ_ranges> dQ_i_T;
    // dQ row view (for atomic store)
    art<float, DOT_SLICE_QO, 32, row_l, rt_16x16_s,
        ducks::art::transpose_2d<dQ_ranges, 2, 1>> dQ_i;

    // Row view of dP_bf16 for shared store
    art<bf16, WARP_SIZE_KV, DOT_SLICE_QO, row_l, rt_16x16_s,
        ducks::art::transpose_2d<dP_bf16_ranges, 1, 2>> dP_ij_bf16_accum_row;

    // Row view of P_bf16 for shared store
    art<bf16, WARP_SIZE_KV, DOT_SLICE_QO, row_l, rt_16x16_s,
        ducks::art::transpose_2d<P_bf16_ranges, 1, 2>> P_ij_bf16_accum_row;

    // Row views for store
    // dV: [128x32] col_l 4x1 tiles -> [32x128] row_l 1x4
    art<float, WARP_SIZE_KV, D_V, row_l, rt_32x32_s,
        ducks::art::transpose_2d<dV_ranges, 4, 1>> dV_j;
    // dK: [192x32] col_l 6x1 tiles -> [32x192] row_l 1x6
    art<float, WARP_SIZE_KV, D_QK, row_l, rt_32x32_s,
        ducks::art::transpose_2d<dK_ranges, 6, 1>> dK_j;

    // dK chunk 2 for epilogue: [64×32] col_l in v[128:159], row view [32×64]
    art<float, 64, WARP_SIZE_KV, col_l, rt_32x32_s, dK_chunk2_col_ranges> dK_chunk2_T;
    art<float, WARP_SIZE_KV, 64, row_l, rt_32x32_s, dK_chunk2_row_ranges> dK_chunk2;

    // Scalar registers
    constexpr int L_i = 126;
    constexpr int delta_i = 127;
    constexpr int neg_inf_v = 29;
    kittens::macros::clobber_gpr<neg_inf_v>();
    kittens::macros::v_mov_b32_up2p<neg_inf_v>(0xff800000u);

    // =====================================================================
    // Prologue: Load K to shared, V to registers
    // =====================================================================
    int tic = 0, toc = 1;

    // Load K from global to shared memory
    G::load<1, false>(K_j_smem, g.K, {batch_idx, seq_idx, kv_head_idx, 0});

    // Load V from global directly to registers
    load<1, 0>(V_j, g.V, {batch_idx, 0, kv_head_idx, 0}, {0, j, 0, 0});

    // Prefetch first Q, dO, L, delta
    // Using standard kittens sv_fl load (may have issues on 64-lane warps)
    load(L_smem[tic], g.L_vec, {batch_idx, first_q_head, 0, first_step});
    load(delta_smem[tic], g.delta_vec, {batch_idx, first_q_head, 0, first_step});
    G::load<1, false>(Q_i_smem[tic][0],  g.Q,   {batch_idx, first_step * 2 + 0, first_q_head, 0});
    G::load<1, false>(dO_i_smem[tic][0], g.dOg, {batch_idx, first_step * 2 + 0, first_q_head, 0});
    G::load<1, false>(Q_i_smem[tic][1],  g.Q,   {batch_idx, first_step * 2 + 1, first_q_head, 0});
    G::load<1, false>(dO_i_smem[tic][1], g.dOg, {batch_idx, first_step * 2 + 1, first_q_head, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Compute addresses
    const uint32_t K_j_addr = get_address(K_j, subtile_inplace<WARP_SIZE_KV, D_QK>(K_j_smem, {warpid, 0}));
    auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
    const uint32_t dP_ij_bf16_accum_row_addr = get_address(attn_i_smem_subtile, dP_ij_bf16_accum_row);
    uint32_t dS_T_col_addr = get_address(dS_T_col, attn_i_smem);

    uint32_t Q_i_addr;
    uint32_t dO_i_addr;
    uint32_t dO_i_col_addr;
    uint32_t Q_i_col_addr;

    // K_j_col address computed per-column-block inside the dQ loop

    // =====================================================================
    // MAIN LOOP — Simplified (high-level MMA, correctness first)
    // =====================================================================
    zero(dK_j_T);
    zero(dV_j_T);

    for (int step = 0; step < num_steps; step++) {
        const int q_head_idx = step / num_steps_per_head + first_q_head;
        const int q_seq_idx = (step % num_steps_per_head) + first_step;
        const int q_pos_base = q_seq_idx * STEP_QO;

        // Process 4 dot-slices
        
        for (int ds = 0; ds < STEP_QO / DOT_SLICE_QO; ds++) {
            const int q_slice_pos = q_pos_base + ds * DOT_SLICE_QO;

            // Load Q_i[1×6] from shared
            {
              auto Qs = subtile_inplace<DOT_SLICE_QO, D_QK>(Q_i_smem[tic][ds/2], {ds%2, 0});
              Q_i_addr = get_address(Q_i, Qs);
              load<0, 0>(Q_i, Qs, Q_i_addr); load<0, 1>(Q_i, Qs, Q_i_addr);
              load<0, 2>(Q_i, Qs, Q_i_addr); load<0, 3>(Q_i, Qs, Q_i_addr);
              load<0, 4>(Q_i, Qs, Q_i_addr); load<0, 5>(Q_i, Qs, Q_i_addr);
            }
            load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[tic], ds));
            load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], ds));

            // Load K_j[2×6] from shared
            {
              auto Ks = subtile_inplace<WARP_SIZE_KV, D_QK>(K_j_smem, {warpid, 0});
              load<0, 0>(K_j, Ks, K_j_addr); load<0, 1>(K_j, Ks, K_j_addr);
              load<0, 2>(K_j, Ks, K_j_addr); load<0, 3>(K_j, Ks, K_j_addr);
              load<0, 4>(K_j, Ks, K_j_addr); load<0, 5>(K_j, Ks, K_j_addr);
              load<1, 0>(K_j, Ks, K_j_addr); load<1, 1>(K_j, Ks, K_j_addr);
              load<1, 2>(K_j, Ks, K_j_addr); load<1, 3>(K_j, Ks, K_j_addr);
              load<1, 4>(K_j, Ks, K_j_addr); load<1, 5>(K_j, Ks, K_j_addr);
            }
            asm volatile("s_waitcnt lgkmcnt(0)");

            // S = Q @ K^T
            mma_ABt(P_ij, Q_i, K_j);

            // Scale S by (1/sqrt(d) * log2(e)) and subtract L*log2(e)
            mul<0, 0>(P_ij, P_ij, P_SCALE_FACTOR);
            mul<0, 1>(P_ij, P_ij, P_SCALE_FACTOR);
            macros::mul::op<L_i, L_i>(L_SCALE_FACTOR);
            sub_row<0, 0, L_i>(P_ij, P_ij);
            sub_row<0, 1, L_i>(P_ij, P_ij);

            // Causal masking — set future positions to -inf (exp2(-inf) = 0)
            if constexpr (causal) {
                if (q_slice_pos + DOT_SLICE_QO <= k_pos) {
                    // Entire dot-slice is before K block — all masked
                    mov<neg_inf_v>(P_ij);
                } else if (q_slice_pos == k_pos) {
                    // Causal boundary in first 16x16 sub-tile, second fully masked
                    make_causal<0, 0, neg_inf_v>(P_ij, P_ij);
                    mov<0, 1, neg_inf_v>(P_ij);
                } else if (q_slice_pos == k_pos + DOT_SLICE_QO) {
                    // First 16x16 sub-tile fully visible, causal in second
                    make_causal<0, 1, neg_inf_v>(P_ij, P_ij);
                }
                // q_slice_pos >= k_pos + WARP_SIZE_KV: fully visible, no masking
            }
            exp2<0, 0>(P_ij, P_ij);
            exp2<0, 1>(P_ij, P_ij);
            copy<0, 0>(P_ij_bf16, P_ij);
            copy<0, 1>(P_ij_bf16, P_ij);
            swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);

            // Reload V_j (clobbered by dS_T/K_col in previous iteration's dQ phase)
            load<1, 0>(V_j, g.V, {batch_idx, 0, kv_head_idx, 0}, {0, j, 0, 0});

            // Load dO[1×4]
            {
              auto dOs = subtile_inplace<DOT_SLICE_QO, D_V>(dO_i_smem[tic][ds/2], {ds%2, 0});
              dO_i_addr = get_address(dO_i, dOs);
              load<0, 0>(dO_i, dOs, dO_i_addr); load<0, 1>(dO_i, dOs, dO_i_addr);
              load<0, 2>(dO_i, dOs, dO_i_addr); load<0, 3>(dO_i, dOs, dO_i_addr);
            }
            asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)");

            // dP = dO @ V^T
            mma_ABt(dP_ij, dO_i, V_j);

            // dS = P * (dP - delta)
            sub_row<0, 0, delta_i>(dP_ij, dP_ij);
            sub_row<0, 1, delta_i>(dP_ij, dP_ij);
            mul(dP_ij, dP_ij, P_ij);
            copy(dP_ij_bf16, dP_ij);
            swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);

            // dV += dO_col^T @ P_bf16_col (mma_AtB: D = A^T @ B + C)
            {
              auto dOcs = subtile_inplace<DOT_SLICE_QO, D_V>(dO_i_smem[tic][ds/2], {ds%2, 0});
              dO_i_col_addr = get_address(dO_i_col, dOcs);
              load<0, 0>(dO_i_col, dOcs, dO_i_col_addr); load<0, 1>(dO_i_col, dOcs, dO_i_col_addr);
              load<0, 2>(dO_i_col, dOcs, dO_i_col_addr); load<0, 3>(dO_i_col, dOcs, dO_i_col_addr);
            }
            asm volatile("s_waitcnt lgkmcnt(0)");
            mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);

            // dK += Q_col^T @ dS_bf16
            {
              auto Qcs = subtile_inplace<DOT_SLICE_QO, D_QK>(Q_i_smem[tic][ds/2], {ds%2, 0});
              Q_i_col_addr = get_address(Q_i_col, Qcs);
              load<0, 0>(Q_i_col, Qcs, Q_i_col_addr); load<0, 1>(Q_i_col, Qcs, Q_i_col_addr);
              load<0, 2>(Q_i_col, Qcs, Q_i_col_addr); load<0, 3>(Q_i_col, Qcs, Q_i_col_addr);
              load<0, 4>(Q_i_col, Qcs, Q_i_col_addr); load<0, 5>(Q_i_col, Qcs, Q_i_col_addr);
            }
            asm volatile("s_waitcnt lgkmcnt(0)");
            mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);

            // Store dS to shared for dQ transpose
            store<0, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
            store<1, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();

            // Load dS_T[4×1] from shared (same for all column blocks)
            load<0, 0>(dS_T_col, attn_i_smem, dS_T_col_addr);
            load<1, 0>(dS_T_col, attn_i_smem, dS_T_col_addr);
            load<2, 0>(dS_T_col, attn_i_smem, dS_T_col_addr);
            load<3, 0>(dS_T_col, attn_i_smem, dS_T_col_addr);
            asm volatile("s_waitcnt lgkmcnt(0)");

            // dQ: loop over 6 column blocks (D_QK/32 = 192/32 = 6)
            // 4 warps handle 4 blocks per pass; 2 passes needed for 6 blocks
            // dQ atomic adds disabled — enabling mma_AtB for dQ causes compiler
            // to reassign VGPR registers, corrupting dV/dK accumulators.
            // dQ should be computed in a separate kernel pass (like dispatch_bwd_dq).
            for (int col_pass = 0; col_pass < 0; col_pass++) {
                const int col_block = warpid + col_pass * NUM_WARPS;
                if (col_block >= D_QK / 32) break;

                // Compute K_j_col address for this column block
                const uint32_t kb_addr = [&] {
                    const int lid = kittens::laneid();
                    const uint32_t sp = reinterpret_cast<uintptr_t>(
                        &subtile_inplace<BLOCK_SIZE_KV, 32>(K_j_smem, {0, col_block}).data[0]);
                    const int ro = (lid % 16) / 4 + (lid / 16) * 4;
                    const int co = ((lid % 4) * 4);
                    const int boff = (ro * 16 + co) * sizeof(bf16);
                    return sp + boff;
                }();

                // Load K_j_col[4×2] for this column block
                {
                  auto Kcs = subtile_inplace<BLOCK_SIZE_KV, 32>(K_j_smem, {0, col_block});
                  load<0, 0>(K_j_col, Kcs, kb_addr); load<0, 1>(K_j_col, Kcs, kb_addr);
                  load<1, 0>(K_j_col, Kcs, kb_addr); load<1, 1>(K_j_col, Kcs, kb_addr);
                  load<2, 0>(K_j_col, Kcs, kb_addr); load<2, 1>(K_j_col, Kcs, kb_addr);
                  load<3, 0>(K_j_col, Kcs, kb_addr); load<3, 1>(K_j_col, Kcs, kb_addr);
                }
                asm volatile("s_waitcnt lgkmcnt(0)");

                // dQ_i_T = K_j_col^T @ dS_T_col  →  [32 × 16]
                mma_AtB(dQ_i_T, K_j_col, dS_T_col);
                mul(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);

                // Simple test: each thread writes 1.0 to dQ at its computed position
                // Only warp 0 (col_block=0), only lane 0
                {
                    const int lid = kittens::laneid();
                    if (warpid == 0 && lid == 0 && step == 0 && ds == 0) {
                        // Write bf16(1.0) to dQ[batch, head, 0, 0]
                        bf16 *p = g.dQg.raw_ptr + ((batch_idx * ATTN_H + q_head_idx) * ATTN_N) * D_QK;
                        bf16 one = __float2bfloat16(1.0f);
                        *p = one;
                    }
                }
            }

            __builtin_amdgcn_s_barrier();
        } // ds

        // Prefetch next step
        if (step + 1 < num_steps) {
            const int ns_head = (step + 1) / num_steps_per_head + first_q_head;
            const int ns_seq  = ((step + 1) % num_steps_per_head) + first_step;
            tic = 1 - tic; toc = 1 - toc;
            load(L_smem[tic], g.L_vec, {batch_idx, ns_head, 0, ns_seq});
            load(delta_smem[tic], g.delta_vec, {batch_idx, ns_head, 0, ns_seq});
            G::load<1, false>(Q_i_smem[tic][0],  g.Q,   {batch_idx, ns_seq * 2 + 0, ns_head, 0});
            G::load<1, false>(dO_i_smem[tic][0], g.dOg, {batch_idx, ns_seq * 2 + 0, ns_head, 0});
            G::load<1, false>(Q_i_smem[tic][1],  g.Q,   {batch_idx, ns_seq * 2 + 1, ns_head, 0});
            G::load<1, false>(dO_i_smem[tic][1], g.dOg, {batch_idx, ns_seq * 2 + 1, ns_head, 0});
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_s_barrier();
        }
    } // step

    // =====================================================================
    // Epilogue: Store dV, then dK (in two chunks via AGPR→VGPR copy)
    // =====================================================================
    // Store dV (already in VGPRs v[128:191], no scale needed — P is softmax probabilities)
    store<1>(g.dVg, dV_j, {batch_idx, 0, kv_head_idx, 0}, {0, j, 0, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    // --- dK store: copy AGPRs to VGPRs, scale, store ---
    // Chunk 2 first: a[64:95] → v[128:159], last 64 cols of dK
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        (macros::v_accvgpr_read_b32<128 + Is, 320 + Is>(), ...);
    }(std::make_index_sequence<32>{});
    mul(dK_chunk2_T, dK_chunk2_T, dP_SCALE_FACTOR);
    // Store chunk 2 at column 128: warp_idx col=2 → 2*64=128 elements
    store<1>(g.dKg, dK_chunk2, {batch_idx, 0, kv_head_idx, 0}, {0, j, 0, 2});
    __builtin_amdgcn_s_waitcnt(0);

    // Chunk 1: a[0:63] → v[128:191], first 128 cols of dK
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        (macros::v_accvgpr_read_b32<128 + Is, 256 + Is>(), ...);
    }(std::make_index_sequence<64>{});
    mul(dV_j_T, dV_j_T, dP_SCALE_FACTOR);
    store<1>(g.dKg, dV_j, {batch_idx, 0, kv_head_idx, 0}, {0, j, 0, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------
void dispatch_bwd_d192v128(attn_bwd_d192v128_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute(
        (void*)attend_bwd_d192v128_ker,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_bwd_d192v128_ker<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

// ---------------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------------
PYBIND11_MODULE(tk_kernel_bkwd, m) {
    m.doc() = "ART-based backward kernel D_QK=192 D_V=128";
    py::bind_function<dispatch_bwd_d192v128>(m, "dispatch_bwd_combined",
        &attn_bwd_d192v128_globals::Q,
        &attn_bwd_d192v128_globals::K,
        &attn_bwd_d192v128_globals::V,
        &attn_bwd_d192v128_globals::dOg,
        &attn_bwd_d192v128_globals::dQg,
        &attn_bwd_d192v128_globals::dKg,
        &attn_bwd_d192v128_globals::dVg,
        &attn_bwd_d192v128_globals::L_vec,
        &attn_bwd_d192v128_globals::delta_vec
    );
}
