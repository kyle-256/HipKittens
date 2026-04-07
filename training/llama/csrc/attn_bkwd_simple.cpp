#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#ifndef ATTN_B
constexpr int ATTN_B = 8;
#endif
#ifndef ATTN_H
constexpr int ATTN_H = 16;
#endif
#ifndef ATTN_H_KV
constexpr int ATTN_H_KV = 16;
#endif
constexpr int GROUP_SIZE = ATTN_H / ATTN_H_KV;
#ifndef ATTN_N
constexpr int ATTN_N = 2048;
#endif

constexpr int ATTN_D = 128;
constexpr int KV_BLOCK = 64;
constexpr int BLOCK_KV = 256;
constexpr int Q_TILE = 32;
constexpr int STEP_QO = 64;
constexpr bool causal_flag = true;

#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;
using G = kittens::group<NUM_WARPS>;
using _gl = gl<bf16, -1, -1, -1, -1>;

#define BS(b, s, h, d) {SBHD ? (s) : (b), SBHD ? (b) : (s), (h), (d)}

template<int D, bool SBHD=false> struct bwd_globals {
    _gl Q, K, V, dOg, dQg, dKg, dVg;
    gl<float,-1,-1,-1,-1> L_vec, delta_vec;
    hipStream_t stream;
    dim3 grid() { return dim3(ATTN_H_KV, ATTN_N / BLOCK_KV, ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

template<int D, bool SBHD=false> __launch_bounds__(NUM_THREADS, 1)
__global__ void bwd_ker(const bwd_globals<D, SBHD> g) {

    constexpr int QKVO_AXIS = SBHD ? 0 : 1;
    constexpr float P_SCALE = (D == 128) ? 0.08838834764f * 1.44269504089f : 0.125f * 1.44269504089f;
    constexpr float L_SCALE = 1.44269504089f;
    constexpr float dP_SCALE = (D == 128) ? 0.08838834764f : 0.125f;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    st_bf<BLOCK_KV, D, st_32x32_s> (&KV_smem) = al.allocate<st_bf<BLOCK_KV, D, st_32x32_s>>();
    st_bf<Q_TILE, D, st_32x32_s> (&QdO_smem_row) = al.allocate<st_bf<Q_TILE, D, st_32x32_s>>();
    st_bf<Q_TILE, D, st_8x32_s>  (&QdO_smem_col) = al.allocate<st_bf<Q_TILE, D, st_8x32_s>>();
    sv_fl<STEP_QO> (&L_smem) = al.allocate<sv_fl<STEP_QO>>();
    sv_fl<STEP_QO> (&delta_smem) = al.allocate<sv_fl<STEP_QO>>();

    const int kv_head = blockIdx.x, seq_block = blockIdx.y, batch = blockIdx.z;
    const int wid = kittens::warpid(), j = seq_block * NUM_WARPS + wid;

    G::load<QKVO_AXIS, false>(KV_smem, g.K, BS(batch, seq_block, kv_head, 0));
    __builtin_amdgcn_s_waitcnt(0); __builtin_amdgcn_s_barrier();
    rt<bf16, KV_BLOCK, D, row_l, rt_32x16_s> K_j;
    load(K_j, subtile_inplace<KV_BLOCK, D>(KV_smem, {wid, 0}));
    __builtin_amdgcn_s_barrier();

    G::load<QKVO_AXIS, false>(KV_smem, g.V, BS(batch, seq_block, kv_head, 0));
    __builtin_amdgcn_s_waitcnt(0); __builtin_amdgcn_s_barrier();
    rt<bf16, KV_BLOCK, D, row_l, rt_32x16_s> V_j;
    load(V_j, subtile_inplace<KV_BLOCK, D>(KV_smem, {wid, 0}));
    __builtin_amdgcn_s_barrier();

    rt<float, D, KV_BLOCK, col_l, rt_32x32_s> dK_acc, dV_acc;
    zero(dK_acc); zero(dV_acc);

    const int kv_start = j * KV_BLOCK;
    const int total_q_tiles = ATTN_N / Q_TILE;
    const int block_kv_min = seq_block * BLOCK_KV;
    const int first_q_tile = causal_flag ? max(0, block_kv_min / STEP_QO) * (STEP_QO / Q_TILE) : 0;
    int prev_L_group = -1;

    for (int qi = first_q_tile; qi < total_q_tiles; qi++) {
        for (int q_head_offset = 0; q_head_offset < GROUP_SIZE; q_head_offset++) {
            const int q_head = kv_head * GROUP_SIZE + q_head_offset;
            const int q_pos = qi * Q_TILE;
            const bool warp_skip = causal_flag && (q_pos + Q_TILE <= kv_start);

            int L_group = qi / (STEP_QO / Q_TILE);
            if (L_group != prev_L_group || q_head_offset > 0) {
                load(L_smem, g.L_vec, {batch, q_head, 0, L_group});
                load(delta_smem, g.delta_vec, {batch, q_head, 0, L_group});
                prev_L_group = L_group;
            }

            // Load Q row_l
            G::load<QKVO_AXIS, false>(QdO_smem_row, g.Q, BS(batch, qi, q_head, 0));
            __builtin_amdgcn_s_waitcnt(0); __builtin_amdgcn_s_barrier();
            rt<bf16, Q_TILE, D, row_l, rt_32x16_s> Q_i;
            if (!warp_skip) load(Q_i, QdO_smem_row);

            // Load Q col_l (early — may be spilled but needed for dK)
            G::load<QKVO_AXIS, false>(QdO_smem_col, g.Q, BS(batch, qi, q_head, 0));
            __builtin_amdgcn_s_waitcnt(0); __builtin_amdgcn_s_barrier();
            rt<bf16, Q_TILE, D, col_l, rt_16x32_4_s> Q_i_col;
            if (!warp_skip) load(Q_i_col, QdO_smem_col);

            // Load dO row_l
            G::load<QKVO_AXIS, false>(QdO_smem_row, g.dOg, BS(batch, qi, q_head, 0));
            __builtin_amdgcn_s_waitcnt(0); __builtin_amdgcn_s_barrier();
            rt<bf16, Q_TILE, D, row_l, rt_32x16_s> dO_i;
            if (!warp_skip) load(dO_i, QdO_smem_row);

            // Load dO col_l (late — stays in registers for dV)
            G::load<QKVO_AXIS, false>(QdO_smem_col, g.dOg, BS(batch, qi, q_head, 0));
            __builtin_amdgcn_s_waitcnt(0); __builtin_amdgcn_s_barrier();
            rt<bf16, Q_TILE, D, col_l, rt_16x32_4_s> dO_i_col;
            if (!warp_skip) load(dO_i_col, QdO_smem_col);

            if (!warp_skip) {
                // S = Q @ K^T
                rt<float, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> S_ij;
                zero(S_ij); mma_ABt(S_ij, Q_i, K_j, S_ij); mul(S_ij, S_ij, P_SCALE);

                int L_sub = qi % (STEP_QO / Q_TILE);
                typename decltype(S_ij)::col_vec L_reg;
                load(L_reg, subvec_inplace<Q_TILE>(L_smem, L_sub));
                mul(L_reg, L_reg, L_SCALE); sub_row(S_ij, S_ij, L_reg);

                if constexpr (causal_flag) {
                    const int lane = laneid(); const int rb = (lane >> 5) << 2;
                    #pragma unroll
                    for (int i = 0; i < S_ij.height; i++)
                        #pragma unroll
                        for (int jj = 0; jj < S_ij.width; jj++) {
                            int akv = kv_start + jj * 32 + (lane & 31);
                            #pragma unroll
                            for (int k = 0; k < S_ij.tiles[i][jj].packed_per_thread; k++) {
                                int ro = ((k>>1)<<3)+((k&1)<<1);
                                int aq = q_pos + i*32 + rb + ro;
                                if (akv > aq)   S_ij.tiles[i][jj].data[k].x = -1e9f;
                                if (akv > aq+1) S_ij.tiles[i][jj].data[k].y = -1e9f;
                            }
                        }
                }

                #pragma unroll
                for (int i = 0; i < S_ij.height; i++)
                    #pragma unroll
                    for (int jj = 0; jj < S_ij.width; jj++)
                        #pragma unroll
                        for (int k = 0; k < S_ij.tiles[i][jj].packed_per_thread; k++)
                            S_ij.tiles[i][jj].data[k] = base_ops::exp2::op(S_ij.tiles[i][jj].data[k]);

                // dP = dO @ V^T
                rt<float, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> dP_ij;
                zero(dP_ij); mma_ABt(dP_ij, dO_i, V_j, dP_ij);
                typename decltype(dP_ij)::col_vec delta_reg;
                load(delta_reg, subvec_inplace<Q_TILE>(delta_smem, L_sub));
                sub_row(dP_ij, dP_ij, delta_reg);
                mul(dP_ij, dP_ij, S_ij);

                // dV^T += dO_col^T @ P_mma
                {
                    rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> P_bf16;
                    copy(P_bf16, S_ij);
                    auto &P_mma = *reinterpret_cast<rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_16x32_4_s>*>(&P_bf16);
                    mma_AtB(dV_acc, dO_i_col, P_mma, dV_acc);
                }

                // dK^T += Q_col^T @ dS_mma
                mul(dP_ij, dP_ij, dP_SCALE);
                {
                    rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> dS_bf16;
                    copy(dS_bf16, dP_ij);
                    auto &dS_mma = *reinterpret_cast<rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_16x32_4_s>*>(&dS_bf16);
                    mma_AtB(dK_acc, Q_i_col, dS_mma, dK_acc);
                }
            }

            __builtin_amdgcn_s_barrier();
        }
    }

    rt<float, KV_BLOCK, D, row_l, rt_32x32_s> dV_out, dK_out;
    transpose(dV_out, dV_acc); transpose(dK_out, dK_acc);
    rt<bf16, KV_BLOCK, D, row_l, rt_32x32_s> dV_bf16, dK_bf16;
    copy(dV_bf16, dV_out); copy(dK_bf16, dK_out);
    store<QKVO_AXIS>(g.dVg, dV_bf16, BS(batch, j, kv_head, 0));
    store<QKVO_AXIS>(g.dKg, dK_bf16, BS(batch, j, kv_head, 0));
}

template<int D, bool SBHD>
void dispatch_bwd(bwd_globals<D, SBHD> g) {
    unsigned long mem = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)bwd_ker<D, SBHD>, hipFuncAttributeMaxDynamicSharedMemorySize, mem);
    bwd_ker<D, SBHD><<<g.grid(), g.block(), mem, g.stream>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel_bkwd_simple, m) {
    m.doc() = "Simple backward kernel (no art, gfx950 compatible)";
    py::bind_function<dispatch_bwd<ATTN_D, false>>(m, "dispatch_bwd",
        &bwd_globals<ATTN_D, false>::Q, &bwd_globals<ATTN_D, false>::K,
        &bwd_globals<ATTN_D, false>::V, &bwd_globals<ATTN_D, false>::dOg,
        &bwd_globals<ATTN_D, false>::dQg, &bwd_globals<ATTN_D, false>::dKg,
        &bwd_globals<ATTN_D, false>::dVg, &bwd_globals<ATTN_D, false>::L_vec,
        &bwd_globals<ATTN_D, false>::delta_vec);
    py::bind_function<dispatch_bwd<ATTN_D, true>>(m, "dispatch_bwd_sbhd",
        &bwd_globals<ATTN_D, true>::Q, &bwd_globals<ATTN_D, true>::K,
        &bwd_globals<ATTN_D, true>::V, &bwd_globals<ATTN_D, true>::dOg,
        &bwd_globals<ATTN_D, true>::dQg, &bwd_globals<ATTN_D, true>::dKg,
        &bwd_globals<ATTN_D, true>::dVg, &bwd_globals<ATTN_D, true>::L_vec,
        &bwd_globals<ATTN_D, true>::delta_vec);
}
