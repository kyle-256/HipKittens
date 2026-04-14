// MLA Backward Kernel (D_QK=192, D_V=128) -- causal, dK/dV/dQ
// Uses ART (Assigned Register Tiles) for explicit register allocation
// to avoid AGPR aliasing on gfx950.
//
// Architecture:
//   DOT_SLICE_QO=16, STEP_QO=32 (2 dot slices per step)
//   KV_BLOCK=32 per warp, BLOCK_KV=128, 4 warps, 16x16 MFMA tiles

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "utils.cpp"

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

constexpr int KV_BLOCK = 32;           // per warp
constexpr int BLOCK_KV = KV_BLOCK * 4; // 128
constexpr int STEP_QO  = 32;           // Q rows per step (2 dot slices)
constexpr int DOT_SLICE_QO = 16;       // Q rows per dot slice
constexpr bool causal = true;

#define NUM_WARPS  4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;
using G   = kittens::group<NUM_WARPS>;
using _gl = gl<bf16, -1, -1, -1, -1>;

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
struct attn_bwd_combined_d192v128_globals {
    _gl Q, K, V;
    _gl dOg, dQg, dKg, dVg;
    gl<float, -1, -1, -1, -1> L_vec, delta_vec;
    hipStream_t stream;
    dim3 grid()  { return dim3(ATTN_H_KV, (ATTN_N / BLOCK_KV), ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
__launch_bounds__(NUM_THREADS, 1)
__global__ __attribute__((amdgpu_num_vgpr(29)))
void attend_bwd_combined_d192v128_ker(
    const attn_bwd_combined_d192v128_globals g)
{
    constexpr float P_SCALE  = 0.07216878365f * 1.44269504089f;
    constexpr float L_SCALE  = 1.44269504089f;
    constexpr float dP_SCALE = 0.07216878365f;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    st_bf<BLOCK_KV, D_QK, st_16x16_s> (&K_smem) = al.allocate<st_bf<BLOCK_KV, D_QK, st_16x16_s>>();
    st_bf<STEP_QO, D_QK, st_16x32_s> (&Q_smem)[2] = al.allocate<st_bf<STEP_QO, D_QK, st_16x32_s>, 2>();
    st_bf<STEP_QO, D_V,  st_16x32_s> (&dO_smem)[2] = al.allocate<st_bf<STEP_QO, D_V,  st_16x32_s>, 2>();
    st_bf<BLOCK_KV, DOT_SLICE_QO, st_16x16_swizzled_s> (&attn_smem) = al.allocate<st_bf<BLOCK_KV, DOT_SLICE_QO, st_16x16_swizzled_s>>();
    sv_fl<STEP_QO> (&L_smem)[2]    = al.allocate<sv_fl<STEP_QO>, 2>();
    sv_fl<STEP_QO> (&delta_smem)[2] = al.allocate<sv_fl<STEP_QO>, 2>();

    // ---- ART range declarations ----
    using dK_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<384, 479>>, 16>;
    using dV_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<128, 191>>, 16>;
    using K_ranges  = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<256, 303>>, 4>;
    using V_ranges  = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<304, 335>>, 4>;
    using Q_ranges  = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<336, 359>>, 4>;
    using dO_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<0, 15>>, 4>;
    using dO_col_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<16, 31>>, 4>;
    using Q_col_ranges  = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<360, 383>>, 4>;
    using P_ranges  = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<32, 39>>, 4>;
    using dP_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<40, 47>>, 4>;
    using P_bf16_ranges  = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<48, 51>>, 2>;
    using dP_bf16_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<52, 55>>, 2>;
    using P_bf16_col_ranges  = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<48, 51>>, 4>;
    using dP_bf16_col_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<52, 55>>, 4>;
    using dS_col_T_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<56, 71>>, 4>;
    using dQ_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<72, 79>>, 4>;
    using K_col_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<256, 287>>, 4>;

    ducks::art::clobber<dK_ranges>(); ducks::art::clobber<dV_ranges>();
    ducks::art::clobber<K_ranges>();  ducks::art::clobber<V_ranges>();
    ducks::art::clobber<Q_ranges>();  ducks::art::clobber<dO_ranges>();
    ducks::art::clobber<dO_col_ranges>(); ducks::art::clobber<Q_col_ranges>();
    ducks::art::clobber<P_ranges>();  ducks::art::clobber<dP_ranges>();
    ducks::art::clobber<P_bf16_ranges>(); ducks::art::clobber<dP_bf16_ranges>();
    ducks::art::clobber<dS_col_T_ranges>(); ducks::art::clobber<dQ_ranges>();

    // ---- ART tile declarations ----
    art<bf16, KV_BLOCK, D_QK, row_l, rt_16x32_s, K_ranges> K_j;
    art<bf16, KV_BLOCK, D_V,  row_l, rt_16x32_s, V_ranges> V_j;
    art<bf16, DOT_SLICE_QO, D_QK, row_l, rt_16x32_s, Q_ranges> Q_i;
    art<bf16, DOT_SLICE_QO, D_V,  row_l, rt_16x32_s, dO_ranges> dO_i;
    art<bf16, DOT_SLICE_QO, D_V,  col_l, rt_16x32_s, dO_col_ranges> dO_i_col;
    art<bf16, DOT_SLICE_QO, D_QK, col_l, rt_16x32_s, Q_col_ranges> Q_i_col;

    art<float, DOT_SLICE_QO, KV_BLOCK, col_l, rt_16x16_s, P_ranges> P_ij;
    art<float, DOT_SLICE_QO, KV_BLOCK, col_l, rt_16x16_s, dP_ranges> dP_ij;
    art<bf16, DOT_SLICE_QO, KV_BLOCK, col_l, rt_16x16_s, P_bf16_ranges> P_ij_bf16;
    art<bf16, DOT_SLICE_QO, KV_BLOCK, col_l, rt_16x16_s, dP_bf16_ranges> dP_ij_bf16;
    art<bf16, DOT_SLICE_QO, KV_BLOCK, col_l, rt_16x32_s, P_bf16_col_ranges> P_ij_bf16_col;
    art<bf16, DOT_SLICE_QO, KV_BLOCK, col_l, rt_16x32_s, dP_bf16_col_ranges> dP_ij_bf16_col;
    art<bf16, KV_BLOCK, DOT_SLICE_QO, row_l, rt_16x16_s,
        ducks::art::transpose_2d<dP_bf16_ranges, 1, 2>> dP_ij_bf16_accum_row;

    art<float, D_QK, KV_BLOCK, col_l, rt_32x32_s, dK_ranges> dK_j_T;
    art<float, D_V,  KV_BLOCK, col_l, rt_32x32_s, dV_ranges> dV_j_T;
    art<float, KV_BLOCK, D_V, row_l, rt_32x32_s,
        ducks::art::transpose_2d<dV_ranges, 4, 1>> dV_j;

    art<bf16, BLOCK_KV, KV_BLOCK, col_l, rt_32x16_4_s, K_col_ranges> K_j_col;
    art<bf16, BLOCK_KV, DOT_SLICE_QO, col_l, rt_32x16_4_s, dS_col_T_ranges> dS_col_T;
    art<float, 32, 16, col_l, rt_16x16_s, dQ_ranges> dQ_i_T;
    art<float, 16, 32, row_l, rt_16x16_s,
        ducks::art::transpose_2d<dQ_ranges, 2, 1>> dQ_i;

    // L, delta, neg_inf
    constexpr int L_i     = 80;
    constexpr int delta_i = 81;
    constexpr int neg_inf_v = 82;
    kittens::macros::clobber_gpr<L_i>();
    kittens::macros::clobber_gpr<delta_i>();
    kittens::macros::clobber_gpr<neg_inf_v>();
    // Set neg_inf to -inf (0xff800000)
    asm volatile("v_mov_b32 v[%0], 0xff800000" : : "n"(neg_inf_v));

    // ---- Indices ----
    const int kv_head   = blockIdx.x;
    const int seq_block = blockIdx.y;
    const int batch     = blockIdx.z;
    const int warpid    = kittens::warpid();
    const int j         = seq_block * NUM_WARPS + warpid;
    const int k_pos     = j * KV_BLOCK;

    const int first_q_head = kv_head * GROUP_SIZE;
    const int total_steps  = ATTN_N / STEP_QO;
    const int j_min        = seq_block * NUM_WARPS;
    const int k_start_min  = j_min * KV_BLOCK;
    const int first_step   = causal ? max(0, k_start_min / STEP_QO) : 0;
    const int num_steps_per_head = total_steps - first_step;
    const int num_steps    = num_steps_per_head * GROUP_SIZE;

    // ---- Swizzled offsets (separate for Q and dO since they have different D) ----
    constexpr int bytes_per_thread_q = st_16x32_s::template bytes_per_thread<bf16>();
    constexpr int memcpy_per_tile_q = STEP_QO * D_QK * sizeof(bf16) / (bytes_per_thread_q * NUM_THREADS);
    uint32_t swizzled_offsets_Q[memcpy_per_tile_q];
    G::prefill_swizzled_offsets<1, false>(Q_smem[0], g.Q, swizzled_offsets_Q);

    constexpr int memcpy_per_tile_do = STEP_QO * D_V * sizeof(bf16) / (bytes_per_thread_q * NUM_THREADS);
    uint32_t swizzled_offsets_dO[memcpy_per_tile_do];
    G::prefill_swizzled_offsets<1, false>(dO_smem[0], g.dOg, swizzled_offsets_dO);

    int tic = 0, toc = 1;

    // ---- Load K into shared, then to registers ----
    G::load<1, false>(K_smem, g.K, {batch, seq_block, kv_head, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    const uint32_t K_j_addr = get_address(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {warpid, 0}));
    // K_j is 32x192 row_l = height=2, width=6 = 12 tiles
    load<0, 0>(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {warpid, 0}), K_j_addr);
    load<0, 1>(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {warpid, 0}), K_j_addr);
    load<0, 2>(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {warpid, 0}), K_j_addr);
    load<0, 3>(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {warpid, 0}), K_j_addr);
    load<0, 4>(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {warpid, 0}), K_j_addr);
    load<0, 5>(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {warpid, 0}), K_j_addr);
    load<1, 0>(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {warpid, 0}), K_j_addr);
    load<1, 1>(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {warpid, 0}), K_j_addr);
    load<1, 2>(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {warpid, 0}), K_j_addr);
    load<1, 3>(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {warpid, 0}), K_j_addr);
    load<1, 4>(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {warpid, 0}), K_j_addr);
    load<1, 5>(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {warpid, 0}), K_j_addr);

    // Load V_j from global to registers (axis=1, row_l)
    load<1, 0>(V_j, g.V, {batch, 0, kv_head, 0}, {0, j, 0, 0});

    asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    // K_j_col address for dQ path
    const uint32_t K_j_col_addr = [&] {
        const int laneid = kittens::laneid();
        const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, warpid}).data[0]);
        const int row_offset = (laneid % 16) / 4 + (laneid / 16) * 4;
        const int col_offset = ((laneid % 4) * 4);
        const int lane_byte_offset = (row_offset * 16 + col_offset) * sizeof(bf16);
        return src_ptr + lane_byte_offset;
    }();

    auto attn_smem_subtile = subtile_inplace<KV_BLOCK, DOT_SLICE_QO>(attn_smem, {warpid, 0});
    const uint32_t dP_store_addr = get_address(attn_smem_subtile, dP_ij_bf16_accum_row);
    uint32_t dS_col_T_addr = get_address(dS_col_T, attn_smem);

    zero(dK_j_T);
    zero(dV_j_T);

    // ---- Prefetch first Q/dO/L/delta ----
    load(L_smem[tic], g.L_vec, {batch, first_q_head, 0, first_step});
    load(delta_smem[tic], g.delta_vec, {batch, first_q_head, 0, first_step});
    G::load<1, false>(Q_smem[tic], g.Q, {batch, first_step, first_q_head, 0}, swizzled_offsets_Q);
    G::load<1, false>(dO_smem[tic], g.dOg, {batch, first_step, first_q_head, 0}, swizzled_offsets_dO);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    uint32_t Q_i_addr, dO_i_addr, dO_i_col_addr, Q_i_col_addr;

    // ---- Macro to reload K_j from shared ----
    #define RELOAD_K_J() do { \
        load<0, 0>(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {warpid, 0}), K_j_addr); \
        load<0, 1>(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {warpid, 0}), K_j_addr); \
        load<0, 2>(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {warpid, 0}), K_j_addr); \
        load<0, 3>(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {warpid, 0}), K_j_addr); \
        load<0, 4>(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {warpid, 0}), K_j_addr); \
        load<0, 5>(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {warpid, 0}), K_j_addr); \
        load<1, 0>(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {warpid, 0}), K_j_addr); \
        load<1, 1>(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {warpid, 0}), K_j_addr); \
        load<1, 2>(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {warpid, 0}), K_j_addr); \
        load<1, 3>(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {warpid, 0}), K_j_addr); \
        load<1, 4>(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {warpid, 0}), K_j_addr); \
        load<1, 5>(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {warpid, 0}), K_j_addr); \
        asm volatile("s_waitcnt lgkmcnt(0)"); \
    } while(0)

    // =======================================================================
    // Main loop
    // =======================================================================
    for (int step = 0; step < num_steps; step++) {
        const int q_head = step / num_steps_per_head + first_q_head;
        const int q_seq_idx = (step % num_steps_per_head) + first_step;
        const int q_pos = q_seq_idx * STEP_QO;
        const bool has_next = (step + 1) < num_steps;
        const int next_q_head = has_next ? ((step + 1) / num_steps_per_head + first_q_head) : q_head;
        const int next_q_seq_idx = has_next ? (((step + 1) % num_steps_per_head) + first_step) : q_seq_idx;

        // Process 2 dot slices
        for (int slice_idx = 0; slice_idx < 2; slice_idx++) {
            const bool is_last_slice = (slice_idx == 1);
            const int effective_q_pos = q_pos + slice_idx * DOT_SLICE_QO;

            // Load Q_i from shared (Q_i: 16x192 row_l = 1x6 tiles)
            Q_i_addr = get_address(Q_i, subtile_inplace<DOT_SLICE_QO, D_QK>(Q_smem[tic], {slice_idx, 0}));
            load<0, 0>(Q_i, subtile_inplace<DOT_SLICE_QO, D_QK>(Q_smem[tic], {0, 0}), Q_i_addr);
            load<0, 1>(Q_i, subtile_inplace<DOT_SLICE_QO, D_QK>(Q_smem[tic], {0, 0}), Q_i_addr);
            load<0, 2>(Q_i, subtile_inplace<DOT_SLICE_QO, D_QK>(Q_smem[tic], {0, 0}), Q_i_addr);
            load<0, 3>(Q_i, subtile_inplace<DOT_SLICE_QO, D_QK>(Q_smem[tic], {0, 0}), Q_i_addr);
            load<0, 4>(Q_i, subtile_inplace<DOT_SLICE_QO, D_QK>(Q_smem[tic], {0, 0}), Q_i_addr);
            load<0, 5>(Q_i, subtile_inplace<DOT_SLICE_QO, D_QK>(Q_smem[tic], {0, 0}), Q_i_addr);

            load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[tic], slice_idx));
            load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], slice_idx));
            asm volatile("s_waitcnt lgkmcnt(0)");

            // S = Q @ K^T  (16x192 @ 192x32 -> 16x32, P_ij is 1x2 tiles of rt_16x16_s)
            mma_ABt<0, 0, 0>(P_ij, Q_i, K_j);
            mma_ABt<0, 0, 1>(P_ij, Q_i, K_j, P_ij);
            mma_ABt<0, 0, 2>(P_ij, Q_i, K_j, P_ij);
            mma_ABt<0, 0, 3>(P_ij, Q_i, K_j, P_ij);
            mma_ABt<0, 0, 4>(P_ij, Q_i, K_j, P_ij);
            mma_ABt<0, 0, 5>(P_ij, Q_i, K_j, P_ij);
            mma_ABt<0, 1, 0>(P_ij, Q_i, K_j);
            mma_ABt<0, 1, 1>(P_ij, Q_i, K_j, P_ij);
            mma_ABt<0, 1, 2>(P_ij, Q_i, K_j, P_ij);
            mma_ABt<0, 1, 3>(P_ij, Q_i, K_j, P_ij);
            mma_ABt<0, 1, 4>(P_ij, Q_i, K_j, P_ij);
            mma_ABt<0, 1, 5>(P_ij, Q_i, K_j, P_ij);

            mul<0, 0>(P_ij, P_ij, P_SCALE);
            mul<0, 1>(P_ij, P_ij, P_SCALE);
            mul<L_i, L_i>(L_SCALE);
            sub_row<0, 0, L_i>(P_ij, P_ij);
            sub_row<0, 1, L_i>(P_ij, P_ij);

            // Causal mask
            if constexpr (causal) {
                if (effective_q_pos < k_pos) {
                    mov<neg_inf_v>(P_ij);
                } else if (effective_q_pos == k_pos) {
                    make_causal<0, 0, neg_inf_v>(P_ij, P_ij);
                    mov<0, 1, neg_inf_v>(P_ij);
                }
            }

            exp2<0, 0>(P_ij, P_ij);
            exp2<0, 1>(P_ij, P_ij);

            // dP = dO @ V^T  (16x128 @ 128x32 -> 16x32)
            dO_i_addr = get_address(dO_i, subtile_inplace<DOT_SLICE_QO, D_V>(dO_smem[tic], {slice_idx, 0}));
            load<0, 0>(dO_i, subtile_inplace<DOT_SLICE_QO, D_V>(dO_smem[tic], {0, 0}), dO_i_addr);
            load<0, 1>(dO_i, subtile_inplace<DOT_SLICE_QO, D_V>(dO_smem[tic], {0, 0}), dO_i_addr);
            load<0, 2>(dO_i, subtile_inplace<DOT_SLICE_QO, D_V>(dO_smem[tic], {0, 0}), dO_i_addr);
            load<0, 3>(dO_i, subtile_inplace<DOT_SLICE_QO, D_V>(dO_smem[tic], {0, 0}), dO_i_addr);
            asm volatile("s_waitcnt lgkmcnt(0)");

            mma_ABt<0, 0, 0>(dP_ij, dO_i, V_j);
            mma_ABt<0, 0, 1>(dP_ij, dO_i, V_j, dP_ij);
            mma_ABt<0, 0, 2>(dP_ij, dO_i, V_j, dP_ij);
            mma_ABt<0, 0, 3>(dP_ij, dO_i, V_j, dP_ij);
            mma_ABt<0, 1, 0>(dP_ij, dO_i, V_j);
            mma_ABt<0, 1, 1>(dP_ij, dO_i, V_j, dP_ij);
            mma_ABt<0, 1, 2>(dP_ij, dO_i, V_j, dP_ij);
            mma_ABt<0, 1, 3>(dP_ij, dO_i, V_j, dP_ij);

            // dS = P * (dP - delta)
            sub_row<0, 0, delta_i>(dP_ij, dP_ij);
            sub_row<0, 1, delta_i>(dP_ij, dP_ij);
            mul<0, 0>(dP_ij, dP_ij, P_ij);
            mul<0, 1>(dP_ij, dP_ij, P_ij);

            // dV += dO_col^T @ P_bf16_col
            copy<0, 0>(P_ij_bf16, P_ij);
            copy<0, 1>(P_ij_bf16, P_ij);
            swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);

            dO_i_col_addr = [&] {
                const int laneid = kittens::laneid();
                const uint32_t src_ptr = reinterpret_cast<uintptr_t>(
                    &subtile_inplace<DOT_SLICE_QO, D_V>(dO_smem[tic], {slice_idx, 0}).data[0]);
                const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
                const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
                const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
                const int swizzled = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
                return src_ptr + swizzled;
            }();
            load<0, 0>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D_V>(dO_smem[tic], {0, 0}), dO_i_col_addr);
            load<0, 1>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D_V>(dO_smem[tic], {0, 0}), dO_i_col_addr);
            load<0, 2>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D_V>(dO_smem[tic], {0, 0}), dO_i_col_addr);
            load<0, 3>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D_V>(dO_smem[tic], {0, 0}), dO_i_col_addr);
            asm volatile("s_waitcnt lgkmcnt(0)");

            // dV_j_T height=4, width=1
            mma_AtB<0, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
            mma_AtB<1, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
            mma_AtB<2, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
            mma_AtB<3, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);

            // dK += Q_col^T @ dS_bf16_col
            copy<0, 0>(dP_ij_bf16, dP_ij);
            copy<0, 1>(dP_ij_bf16, dP_ij);

            // Store dS to shared for dQ path
            store<0, 0>(attn_smem_subtile, dP_ij_bf16_accum_row, dP_store_addr);
            store<1, 0>(attn_smem_subtile, dP_ij_bf16_accum_row, dP_store_addr);

            swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);

            Q_i_col_addr = [&] {
                const int laneid = kittens::laneid();
                const uint32_t src_ptr = reinterpret_cast<uintptr_t>(
                    &subtile_inplace<DOT_SLICE_QO, D_QK>(Q_smem[tic], {slice_idx, 0}).data[0]);
                const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
                const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
                const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
                const int swizzled = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
                return src_ptr + swizzled;
            }();
            load<0, 0>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D_QK>(Q_smem[tic], {0, 0}), Q_i_col_addr);
            load<0, 1>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D_QK>(Q_smem[tic], {0, 0}), Q_i_col_addr);
            load<0, 2>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D_QK>(Q_smem[tic], {0, 0}), Q_i_col_addr);
            load<0, 3>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D_QK>(Q_smem[tic], {0, 0}), Q_i_col_addr);
            load<0, 4>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D_QK>(Q_smem[tic], {0, 0}), Q_i_col_addr);
            load<0, 5>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D_QK>(Q_smem[tic], {0, 0}), Q_i_col_addr);
            asm volatile("s_waitcnt lgkmcnt(0)");

            // dK_j_T height=6, width=1
            mma_AtB<0, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
            mma_AtB<1, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
            mma_AtB<2, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
            mma_AtB<3, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
            mma_AtB<4, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
            mma_AtB<5, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);

            // === dQ path ===
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();

            // Load dS_col_T from shared (128x16 col_l = height=4, width=1)
            load<0, 0>(dS_col_T, attn_smem, dS_col_T_addr);
            load<1, 0>(dS_col_T, attn_smem, dS_col_T_addr);
            load<2, 0>(dS_col_T, attn_smem, dS_col_T_addr);
            load<3, 0>(dS_col_T, attn_smem, dS_col_T_addr);

            // Load K_j_col (128x32 col_l = height=4, width=2)
            load<0, 0>(K_j_col, subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, warpid}), K_j_col_addr);
            load<0, 1>(K_j_col, subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, warpid}), K_j_col_addr);
            load<1, 0>(K_j_col, subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, warpid}), K_j_col_addr);
            load<1, 1>(K_j_col, subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, warpid}), K_j_col_addr);
            load<2, 0>(K_j_col, subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, warpid}), K_j_col_addr);
            load<2, 1>(K_j_col, subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, warpid}), K_j_col_addr);
            load<3, 0>(K_j_col, subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, warpid}), K_j_col_addr);
            load<3, 1>(K_j_col, subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, warpid}), K_j_col_addr);
            asm volatile("s_waitcnt lgkmcnt(0)");

            // Prefetch next if last slice
            if (is_last_slice && has_next) {
                load(L_smem[toc], g.L_vec, {batch, next_q_head, 0, next_q_seq_idx});
                load(delta_smem[toc], g.delta_vec, {batch, next_q_head, 0, next_q_seq_idx});
                G::load<1, false>(Q_smem[toc], g.Q, {batch, next_q_seq_idx, next_q_head, 0}, swizzled_offsets_Q);
                G::load<1, false>(dO_smem[toc], g.dOg, {batch, next_q_seq_idx, next_q_head, 0}, swizzled_offsets_dO);
            }

            __builtin_amdgcn_s_barrier();

            // dQ_i_T = K_j_col^T @ dS_col_T  (32x16 = (128x32)^T @ (128x16))
            // dQ_i_T height=2, width=1; K_j_col height=4
            mma_AtB<0, 0, 0>(dQ_i_T, K_j_col, dS_col_T);
            mma_AtB<0, 0, 1>(dQ_i_T, K_j_col, dS_col_T, dQ_i_T);
            mma_AtB<0, 0, 2>(dQ_i_T, K_j_col, dS_col_T, dQ_i_T);
            mma_AtB<0, 0, 3>(dQ_i_T, K_j_col, dS_col_T, dQ_i_T);
            mma_AtB<1, 0, 0>(dQ_i_T, K_j_col, dS_col_T);
            mma_AtB<1, 0, 1>(dQ_i_T, K_j_col, dS_col_T, dQ_i_T);
            mma_AtB<1, 0, 2>(dQ_i_T, K_j_col, dS_col_T, dQ_i_T);
            mma_AtB<1, 0, 3>(dQ_i_T, K_j_col, dS_col_T, dQ_i_T);

            mul<0, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE);
            mul<0, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE);
            mul<0, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE);
            mul<0, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE);
            mul<1, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE);
            mul<1, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE);
            mul<1, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE);
            mul<1, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE);

            // Atomic add dQ (first 128 D columns covered by warpid 0-3)
            atomic_pk_add_bf16_with_warpid<2, 0, 0>(g.dQg, dQ_i, {batch, q_head, q_seq_idx * 2 + slice_idx, 0}, warpid);
            atomic_pk_add_bf16_with_warpid<2, 0, 1>(g.dQg, dQ_i, {batch, q_head, q_seq_idx * 2 + slice_idx, 0}, warpid);

            // Handle D columns 128-191 (chunks 4 and 5)
            // Each iteration: load K_col for that D-chunk, compute dQ, atomic add
            {
                // D-chunk 4: columns 128-159
                const uint32_t K_j_col_addr_4 = [&] {
                    const int laneid = kittens::laneid();
                    const uint32_t src_ptr = reinterpret_cast<uintptr_t>(
                        &subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, 4}).data[0]);
                    const int row_offset = (laneid % 16) / 4 + (laneid / 16) * 4;
                    const int col_offset = ((laneid % 4) * 4);
                    const int lane_byte_offset = (row_offset * 16 + col_offset) * sizeof(bf16);
                    return src_ptr + lane_byte_offset;
                }();
                load<0, 0>(K_j_col, subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, 4}), K_j_col_addr_4);
                load<0, 1>(K_j_col, subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, 4}), K_j_col_addr_4);
                load<1, 0>(K_j_col, subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, 4}), K_j_col_addr_4);
                load<1, 1>(K_j_col, subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, 4}), K_j_col_addr_4);
                load<2, 0>(K_j_col, subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, 4}), K_j_col_addr_4);
                load<2, 1>(K_j_col, subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, 4}), K_j_col_addr_4);
                load<3, 0>(K_j_col, subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, 4}), K_j_col_addr_4);
                load<3, 1>(K_j_col, subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, 4}), K_j_col_addr_4);
                asm volatile("s_waitcnt lgkmcnt(0)");

                mma_AtB<0, 0, 0>(dQ_i_T, K_j_col, dS_col_T);
                mma_AtB<0, 0, 1>(dQ_i_T, K_j_col, dS_col_T, dQ_i_T);
                mma_AtB<0, 0, 2>(dQ_i_T, K_j_col, dS_col_T, dQ_i_T);
                mma_AtB<0, 0, 3>(dQ_i_T, K_j_col, dS_col_T, dQ_i_T);
                mma_AtB<1, 0, 0>(dQ_i_T, K_j_col, dS_col_T);
                mma_AtB<1, 0, 1>(dQ_i_T, K_j_col, dS_col_T, dQ_i_T);
                mma_AtB<1, 0, 2>(dQ_i_T, K_j_col, dS_col_T, dQ_i_T);
                mma_AtB<1, 0, 3>(dQ_i_T, K_j_col, dS_col_T, dQ_i_T);

                mul<0, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE);
                mul<0, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE);
                mul<0, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE);
                mul<0, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE);
                mul<1, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE);
                mul<1, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE);
                mul<1, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE);
                mul<1, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE);

                atomic_pk_add_bf16_with_warpid<2, 0, 0>(g.dQg, dQ_i, {batch, q_head, q_seq_idx * 2 + slice_idx, 0}, 4);
                atomic_pk_add_bf16_with_warpid<2, 0, 1>(g.dQg, dQ_i, {batch, q_head, q_seq_idx * 2 + slice_idx, 0}, 4);
            }
            {
                // D-chunk 5: columns 160-191
                const uint32_t K_j_col_addr_5 = [&] {
                    const int laneid = kittens::laneid();
                    const uint32_t src_ptr = reinterpret_cast<uintptr_t>(
                        &subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, 5}).data[0]);
                    const int row_offset = (laneid % 16) / 4 + (laneid / 16) * 4;
                    const int col_offset = ((laneid % 4) * 4);
                    const int lane_byte_offset = (row_offset * 16 + col_offset) * sizeof(bf16);
                    return src_ptr + lane_byte_offset;
                }();
                load<0, 0>(K_j_col, subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, 5}), K_j_col_addr_5);
                load<0, 1>(K_j_col, subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, 5}), K_j_col_addr_5);
                load<1, 0>(K_j_col, subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, 5}), K_j_col_addr_5);
                load<1, 1>(K_j_col, subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, 5}), K_j_col_addr_5);
                load<2, 0>(K_j_col, subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, 5}), K_j_col_addr_5);
                load<2, 1>(K_j_col, subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, 5}), K_j_col_addr_5);
                load<3, 0>(K_j_col, subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, 5}), K_j_col_addr_5);
                load<3, 1>(K_j_col, subtile_inplace<BLOCK_KV, KV_BLOCK>(K_smem, {0, 5}), K_j_col_addr_5);
                asm volatile("s_waitcnt lgkmcnt(0)");

                mma_AtB<0, 0, 0>(dQ_i_T, K_j_col, dS_col_T);
                mma_AtB<0, 0, 1>(dQ_i_T, K_j_col, dS_col_T, dQ_i_T);
                mma_AtB<0, 0, 2>(dQ_i_T, K_j_col, dS_col_T, dQ_i_T);
                mma_AtB<0, 0, 3>(dQ_i_T, K_j_col, dS_col_T, dQ_i_T);
                mma_AtB<1, 0, 0>(dQ_i_T, K_j_col, dS_col_T);
                mma_AtB<1, 0, 1>(dQ_i_T, K_j_col, dS_col_T, dQ_i_T);
                mma_AtB<1, 0, 2>(dQ_i_T, K_j_col, dS_col_T, dQ_i_T);
                mma_AtB<1, 0, 3>(dQ_i_T, K_j_col, dS_col_T, dQ_i_T);

                mul<0, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE);
                mul<0, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE);
                mul<0, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE);
                mul<0, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE);
                mul<1, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE);
                mul<1, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE);
                mul<1, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE);
                mul<1, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE);

                atomic_pk_add_bf16_with_warpid<2, 0, 0>(g.dQg, dQ_i, {batch, q_head, q_seq_idx * 2 + slice_idx, 0}, 5);
                atomic_pk_add_bf16_with_warpid<2, 0, 1>(g.dQg, dQ_i, {batch, q_head, q_seq_idx * 2 + slice_idx, 0}, 5);
            }

            // Reload K_j from shared (K_j_col clobbered part of K_j AGPRs)
            RELOAD_K_J();

        } // slice_idx

        if (has_next) {
            tic ^= 1;
            toc ^= 1;
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_s_barrier();
        }
    } // step

    // ---- Epilogue: store dV and dK ----
    store<1>(g.dVg, dV_j, {batch, 0, kv_head, 0}, {0, j, 0, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    // Copy dK from AGPR -> VGPR, scale, store (2 passes)
    {
        using dK_first_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<384, 447>>, 16>;
        art<float, 128, KV_BLOCK, col_l, rt_32x32_s, dK_first_ranges> dK_agpr_p1;
        art<float, 128, KV_BLOCK, col_l, rt_32x32_s, dV_ranges> dK_vgpr_p1;
        accvgpr_read(dK_vgpr_p1, dK_agpr_p1);
        mul(dK_vgpr_p1, dK_vgpr_p1, dP_SCALE);
        art<float, KV_BLOCK, 128, row_l, rt_32x32_s,
            ducks::art::transpose_2d<dV_ranges, 4, 1>> dK_row_p1;
        store<1>(g.dKg, dK_row_p1, {batch, 0, kv_head, 0}, {0, j, 0, 0});
    }
    __builtin_amdgcn_s_waitcnt(0);
    {
        using dK_last_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<448, 479>>, 16>;
        using dK_vgpr_2_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<128, 159>>, 16>;
        art<float, 64, KV_BLOCK, col_l, rt_32x32_s, dK_last_ranges> dK_agpr_p2;
        art<float, 64, KV_BLOCK, col_l, rt_32x32_s, dK_vgpr_2_ranges> dK_vgpr_p2;
        accvgpr_read(dK_vgpr_p2, dK_agpr_p2);
        mul(dK_vgpr_p2, dK_vgpr_p2, dP_SCALE);
        art<float, KV_BLOCK, 64, row_l, rt_32x32_s,
            ducks::art::transpose_2d<dK_vgpr_2_ranges, 2, 1>> dK_row_p2;
        store<1>(g.dKg, dK_row_p2, {batch, 0, kv_head, 4}, {0, j, 0, 0});
    }
}

// ---------------------------------------------------------------------------
void dispatch_bwd_combined_d192v128(attn_bwd_combined_d192v128_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_bwd_combined_d192v128_ker,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_bwd_combined_d192v128_ker<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

PYBIND11_MODULE(tk_kernel_bkwd, m) {
    m.doc() = "tk_kernel python module - asymmetric backward D_QK=192 D_V=128";
    py::bind_function<dispatch_bwd_combined_d192v128>(m, "dispatch_bwd_combined",
        &attn_bwd_combined_d192v128_globals::Q,
        &attn_bwd_combined_d192v128_globals::K,
        &attn_bwd_combined_d192v128_globals::V,
        &attn_bwd_combined_d192v128_globals::dOg,
        &attn_bwd_combined_d192v128_globals::dQg,
        &attn_bwd_combined_d192v128_globals::dKg,
        &attn_bwd_combined_d192v128_globals::dVg,
        &attn_bwd_combined_d192v128_globals::L_vec,
        &attn_bwd_combined_d192v128_globals::delta_vec
    );
}
