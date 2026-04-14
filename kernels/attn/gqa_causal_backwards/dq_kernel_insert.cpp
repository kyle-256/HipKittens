// ===========================================================================
// Separate dQ kernel -- iterates Q-tile-major over KV blocks
// Uses rt_32x32_s base tiles matching the combined kernel's conventions.
//
// INSERT THIS CODE before the PYBIND11_MODULE block in attn_bkwd_causal_d192v128.cpp
// and add the dispatch_bwd_dq binding to the PYBIND11_MODULE.
// ===========================================================================

struct attn_bwd_dq_d192v128_globals {
    _gl Q, K, V;
    _gl dOg, dQg;
    gl<float, -1, -1, -1, -1> L_vec, delta_vec;
    hipStream_t stream;
    // Grid: one block per (q_head, q_tile_block, batch)
    dim3 grid()  { return dim3(ATTN_H, (ATTN_N / (Q_TILE * NUM_WARPS)), ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__launch_bounds__(NUM_THREADS, 1)
__global__ void attend_bwd_dq_d192v128_ker(
    const attn_bwd_dq_d192v128_globals g)
{
    constexpr int QKVO_AXIS = 1;
    constexpr float P_SCALE  = 0.07216878365f * 1.44269504089f;
    constexpr float L_SCALE  = 1.44269504089f;
    constexpr float dP_SCALE = 0.07216878365f;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    st_bf<BLOCK_KV, D_QK, st_32x32_s> (&K_smem)  = al.allocate<st_bf<BLOCK_KV, D_QK, st_32x32_s>>();
    st_bf<BLOCK_KV, D_V,  st_32x32_s> (&V_smem)  = al.allocate<st_bf<BLOCK_KV, D_V,  st_32x32_s>>();
    st_bf<Q_TILE, D_QK, st_32x32_s>   (&Q_smem)  = al.allocate<st_bf<Q_TILE, D_QK, st_32x32_s>>();
    st_bf<Q_TILE, D_V,  st_32x32_s>   (&dO_smem) = al.allocate<st_bf<Q_TILE, D_V,  st_32x32_s>>();
    sv_fl<Q_TILE> (&L_smem)     = al.allocate<sv_fl<Q_TILE>>();
    sv_fl<Q_TILE> (&delta_smem) = al.allocate<sv_fl<Q_TILE>>();

    // DUMMY persistent accumulators with MMA writes to force AGPR allocation
    rt<float, D_QK, KV_BLOCK, col_l, rt_32x32_s> _pad_K;
    rt<float, D_V,  KV_BLOCK, col_l, rt_32x32_s> _pad_V;
    zero(_pad_K); zero(_pad_V);
    // Dummy MMA to force AGPRs (zero inputs → zero output, no effect on correctness)
    {
        rt<bf16, Q_TILE, D_QK, col_l, rt_16x32_4_s> _z1; zero(_z1);
        rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_16x32_4_s> _z2; zero(_z2);
        mma_AtB(_pad_K, _z1, _z2, _pad_K);
        rt<bf16, Q_TILE, D_V, col_l, rt_16x32_4_s> _z3; zero(_z3);
        mma_AtB(_pad_V, _z3, _z2, _pad_V);
    }

    const int q_head   = blockIdx.x;
    const int kv_head  = q_head / GROUP_SIZE;
    const int q_block  = blockIdx.y;
    const int batch    = blockIdx.z;
    const int wid      = kittens::warpid();
    const int qi       = q_block * NUM_WARPS + wid;
    const int q_pos    = qi * Q_TILE;
    const int total_kv = ATTN_N / KV_BLOCK;
    const int last_kv  = causal ? min(total_kv, (q_pos + Q_TILE - 1) / KV_BLOCK + 1) : total_kv;

    // Load Q and dO for this Q tile
    G::load<QKVO_AXIS, false>(Q_smem, g.Q, {batch, q_block, q_head, 0});
    G::load<QKVO_AXIS, false>(dO_smem, g.dOg, {batch, q_block, q_head, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    rt<bf16, Q_TILE, D_QK, row_l, rt_32x16_4_s> Q_i;
    load(Q_i, Q_smem);
    rt<bf16, Q_TILE, D_V, row_l, rt_32x16_4_s> dO_i;
    load(dO_i, dO_smem);

    float *L_raw     = reinterpret_cast<float*>(&L_smem);
    float *delta_raw = reinterpret_cast<float*>(&delta_smem);
    load_L_delta_direct(L_raw, g.L_vec, batch, q_head, qi);
    load_L_delta_direct(delta_raw, g.delta_vec, batch, q_head, qi);
    __builtin_amdgcn_s_barrier();

    for (int kj = 0; kj < last_kv; kj++) {
        const int kv_start     = kj * KV_BLOCK;
        const int kv_seq_block = kj / NUM_WARPS;
        const int kv_sub       = kj % NUM_WARPS;

        G::load<QKVO_AXIS, false>(K_smem, g.K, {batch, kv_seq_block, kv_head, 0});
        G::load<QKVO_AXIS, false>(V_smem, g.V, {batch, kv_seq_block, kv_head, 0});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();

        // Phase 1: S = Q @ K^T * P_SCALE - L*L_SCALE, causal mask, exp2
        rt<bf16, KV_BLOCK, D_QK, row_l, rt_32x16_4_s> K_j;
        load(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {kv_sub, 0}));

        rt<float, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> S_ij;
        zero(S_ij);
        mma_ABt(S_ij, Q_i, K_j, S_ij);
        mul(S_ij, S_ij, P_SCALE);

        typename decltype(S_ij)::col_vec L_reg;
        load(L_reg, L_smem);
        mul(L_reg, L_reg, L_SCALE);
        sub_row(S_ij, S_ij, L_reg);

        if constexpr (causal) {
            const int lane = laneid();
            const int kv_c = lane & 31;
            const int G_id = lane >> 5;
            #pragma unroll
            for (int k = 0; k < S_ij.tiles[0][0].packed_per_thread; k++) {
                int ro   = ((k >> 1) << 3) + ((k & 1) << 1);
                int q_r0 = G_id * 4 + ro;
                int q_r1 = q_r0 + 1;
                if (q_pos + q_r0 < kv_start + kv_c)
                    S_ij.tiles[0][0].data[k].x = -__builtin_inff();
                if (q_pos + q_r1 < kv_start + kv_c)
                    S_ij.tiles[0][0].data[k].y = -__builtin_inff();
            }
        }

        #pragma unroll
        for (int ii = 0; ii < S_ij.height; ii++)
            #pragma unroll
            for (int jj = 0; jj < S_ij.width; jj++)
                #pragma unroll
                for (int k = 0; k < S_ij.tiles[ii][jj].packed_per_thread; k++) {
                    S_ij.tiles[ii][jj].data[k].x =
                        __builtin_fminf(S_ij.tiles[ii][jj].data[k].x, 0.f);
                    S_ij.tiles[ii][jj].data[k].y =
                        __builtin_fminf(S_ij.tiles[ii][jj].data[k].y, 0.f);
                    S_ij.tiles[ii][jj].data[k] =
                        base_ops::exp2::op(S_ij.tiles[ii][jj].data[k]);
                }

        // Phase 2: dP = dO @ V^T, dS = P * (dP - delta) * softmax_scale
        rt<bf16, KV_BLOCK, D_V, row_l, rt_32x16_4_s> V_j;
        load(V_j, subtile_inplace<KV_BLOCK, D_V>(V_smem, {kv_sub, 0}));

        rt<float, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> dP_ij;
        zero(dP_ij);
        mma_ABt(dP_ij, dO_i, V_j, dP_ij);

        {
            const int lane = laneid();
            const int G = lane >> 5;
            float *d_raw = reinterpret_cast<float*>(&delta_smem);
            #pragma unroll
            for (int k = 0; k < dP_ij.tiles[0][0].packed_per_thread; k++) {
                int ro = ((k >> 1) << 3) + ((k & 1) << 1);
                int row0 = G * 4 + ro;
                int row1 = row0 + 1;
                dP_ij.tiles[0][0].data[k].x -= d_raw[row0];
                dP_ij.tiles[0][0].data[k].y -= d_raw[row1];
            }
        }
        mul(dP_ij, dP_ij, S_ij);
        mul(dP_ij, dP_ij, dP_SCALE);

        // Phase 3: dQ += K^T @ dS via mma_AtB
        rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> dS_bf;
        copy(dS_bf, dP_ij);
        auto &dSm = *reinterpret_cast<
            rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_16x32_4_s>*>(&dS_bf);

        rt<bf16, KV_BLOCK, D_QK, col_l, rt_16x32_4_s> K_col;
        load(K_col, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {kv_sub, 0}));

        #pragma unroll
        for (int d = 0; d < D_QK / 32; d++) {
            auto &K_chunk = *reinterpret_cast<
                rt<bf16, KV_BLOCK, 32, col_l, rt_16x32_4_s>*>(&K_col.tiles[0][d]);

            rt<float, 32, Q_TILE, col_l, rt_32x32_s> dQ_chunk;
            zero(dQ_chunk);
            mma_AtB(dQ_chunk, K_chunk, dSm, dQ_chunk);

            atomic_add_dQ_col_l(g.dQg, dQ_chunk, batch, q_head, qi, d);
        }

        __builtin_amdgcn_s_barrier();
    } // kj

    // Keep dummy accumulators alive
    if (batch < -1) {
        store_col_l_direct<QKVO_AXIS, D_QK>(g.dQg, _pad_K, batch, 0, kv_head);
        store_col_l_direct<QKVO_AXIS, D_V>(g.dQg, _pad_V, batch, 0, kv_head);
    }
}

void dispatch_bwd_dq_d192v128(attn_bwd_dq_d192v128_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_bwd_dq_d192v128_ker,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_bwd_dq_d192v128_ker<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

// ---- PYBIND11 addition ----
// Add this to the existing PYBIND11_MODULE block:
//
//   py::bind_function<dispatch_bwd_dq_d192v128>(m, "dispatch_bwd_dq",
//       &attn_bwd_dq_d192v128_globals::Q,
//       &attn_bwd_dq_d192v128_globals::K,
//       &attn_bwd_dq_d192v128_globals::V,
//       &attn_bwd_dq_d192v128_globals::dOg,
//       &attn_bwd_dq_d192v128_globals::dQg,
//       &attn_bwd_dq_d192v128_globals::L_vec,
//       &attn_bwd_dq_d192v128_globals::delta_vec
//   );
