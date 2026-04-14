// MLA Backward Kernel (D_QK=192, D_V=128) — causal, dK/dV only
// OPT2: Per-warp global loads for Q/dO (no barriers in inner loop)
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

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

constexpr int KV_BLOCK = 32;
constexpr int BLOCK_KV = KV_BLOCK * 4; // 128
constexpr int Q_TILE   = 32;
constexpr bool causal = true;

#define NUM_WARPS  4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;
using G   = kittens::group<NUM_WARPS>;
using _gl = gl<bf16, -1, -1, -1, -1>;

// store_col_l_direct: bypass broken transpose on gfx950
template<int QKVO_AXIS, int D_DIM>
__device__ __forceinline__ void store_col_l_direct(
    const _gl &dst,
    const rt<float, D_DIM, KV_BLOCK, col_l, rt_32x32_s> &src,
    int batch, int j, int kv_head)
{
    const int lane = laneid();
    const int kv_col = lane & 31;
    const int rb = (lane >> 5) << 2;

    bf16 *base = reinterpret_cast<bf16*>(dst.raw_ptr);
    const int stride_b = dst.template stride<0>();
    const int stride_s = dst.template stride<QKVO_AXIS>();
    const int stride_h = dst.template stride<2>();
    const int base_off = batch * stride_b + (j * KV_BLOCK + kv_col) * stride_s
                       + kv_head * stride_h;

    #pragma unroll
    for (int t = 0; t < D_DIM / 32; t++) {
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int ro = ((k >> 1) << 3) + ((k & 1) << 1);
            int d0 = t * 32 + rb + ro;
            base[base_off + d0]     = __float2bfloat16(src.tiles[t][0].data[k].x);
            base[base_off + d0 + 1] = __float2bfloat16(src.tiles[t][0].data[k].y);
        }
    }
}

// load_L_delta_direct
__device__ __forceinline__ void load_L_delta_direct(
    float *smem_dst,
    const gl<float, -1, -1, -1, -1> &src,
    int batch, int head, int qi)
{
    const int lane = laneid();
    float *src_ptr = (float*)src.raw_ptr;
    const int stride_b = src.template stride<0>();
    const int stride_h = src.template stride<1>();
    const int base_idx = batch * stride_b + head * stride_h + qi * Q_TILE;
    if (lane < Q_TILE) {
        smem_dst[lane] = src_ptr[base_idx + lane];
    }
}

struct attn_bwd_combined_d192v128_globals {
    _gl Q, K, V;
    _gl dOg, dQg, dKg, dVg;
    gl<float, -1, -1, -1, -1> L_vec, delta_vec;
    hipStream_t stream;
    dim3 grid()  { return dim3(ATTN_H_KV, (ATTN_N / BLOCK_KV), ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__launch_bounds__(NUM_THREADS, 1)
__global__ void attend_bwd_combined_d192v128_ker(
    const attn_bwd_combined_d192v128_globals g)
{
    constexpr int QKVO_AXIS = 1;
    constexpr float P_SCALE  = 0.07216878365f * 1.44269504089f;
    constexpr float L_SCALE  = 1.44269504089f;
    constexpr float dP_SCALE = 0.07216878365f;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // Only K/V in shared (loaded once). Q/dO loaded per-warp from global.
    st_bf<BLOCK_KV, D_QK, st_32x32_s> (&K_smem) = al.allocate<st_bf<BLOCK_KV, D_QK, st_32x32_s>>();
    st_bf<BLOCK_KV, D_V,  st_32x32_s> (&V_smem) = al.allocate<st_bf<BLOCK_KV, D_V,  st_32x32_s>>();
    // L/delta still in shared (small, 32 floats each)
    sv_fl<Q_TILE> (&L_smem)     = al.allocate<sv_fl<Q_TILE>>();
    sv_fl<Q_TILE> (&delta_smem) = al.allocate<sv_fl<Q_TILE>>();

    rt<float, D_QK, KV_BLOCK, col_l, rt_32x32_s> dK_acc;
    rt<float, D_V,  KV_BLOCK, col_l, rt_32x32_s> dV_acc;
    zero(dK_acc);
    zero(dV_acc);

    const int kv_head   = blockIdx.x;
    const int seq_block = blockIdx.y;
    const int batch     = blockIdx.z;
    const int wid       = kittens::warpid();
    const int j         = seq_block * NUM_WARPS + wid;
    const int kv_start  = j * KV_BLOCK;
    const int total_q   = ATTN_N / Q_TILE;
    const int first_q   = causal ? max(0, (int)(seq_block * BLOCK_KV / Q_TILE)) : 0;
    const int lane      = laneid();

    // Load K and V into shared memory (all warps cooperate)
    G::load<QKVO_AXIS, false>(K_smem, g.K, {batch, seq_block, kv_head, 0});
    G::load<QKVO_AXIS, false>(V_smem, g.V, {batch, seq_block, kv_head, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    // Load K_j and V_j into registers (hoisted, constant across inner loop)
    rt<bf16, KV_BLOCK, D_QK, row_l, rt_32x16_4_s> K_j;
    load(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {wid, 0}));
    rt<bf16, KV_BLOCK, D_V, row_l, rt_32x16_4_s> V_j;
    load(V_j, subtile_inplace<KV_BLOCK, D_V>(V_smem, {wid, 0}));

    // Main loop: per-warp, NO barriers needed
    #pragma unroll 1
    for (int qi = first_q; qi < total_q; qi++) {
        for (int qho = 0; qho < GROUP_SIZE; qho++) {
            const int q_head = kv_head * GROUP_SIZE + qho;
            const int q_pos  = qi * Q_TILE;
            const bool skip  = causal && (q_pos + Q_TILE <= kv_start);

            if (!skip) {
                // Phase 1: S = Q @ K^T
                // Load Q directly from global into registers (per-warp)
                rt<bf16, Q_TILE, D_QK, row_l, rt_32x16_4_s> Q_i;
                load<QKVO_AXIS, decltype(Q_i), _gl>(Q_i, g.Q, {batch, qi, q_head, 0});

                rt<float, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> S_ij;
                zero(S_ij);
                mma_ABt(S_ij, Q_i, K_j, S_ij);
                mul(S_ij, S_ij, P_SCALE);

                // Load L directly into registers (per-warp, bypassing shared)
                {
                    float *src_ptr = (float*)g.L_vec.raw_ptr;
                    const int stride_b = g.L_vec.template stride<0>();
                    const int stride_h = g.L_vec.template stride<1>();
                    const int base_idx = batch * stride_b + q_head * stride_h + qi * Q_TILE;
                    // col_vec of S_ij (col_l) has elements indexed by the row of S
                    // For col_l rt_32x32_s, col_vec is rv<float, Q_TILE/32, ...>
                    // We need to broadcast L values to match the col structure
                    const int G_id = lane >> 5;
                    #pragma unroll
                    for (int k = 0; k < S_ij.tiles[0][0].packed_per_thread; k++) {
                        int ro = ((k >> 1) << 3) + ((k & 1) << 1);
                        int row0 = G_id * 4 + ro;
                        int row1 = row0 + 1;
                        float l0 = src_ptr[base_idx + row0] * L_SCALE;
                        float l1 = src_ptr[base_idx + row1] * L_SCALE;
                        S_ij.tiles[0][0].data[k].x -= l0;
                        S_ij.tiles[0][0].data[k].y -= l1;
                    }
                }

                if constexpr (causal) {
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

                // Phase 2: dP = dO @ V^T
                rt<bf16, Q_TILE, D_V, row_l, rt_32x16_4_s> dO_i;
                load<QKVO_AXIS, decltype(dO_i), _gl>(dO_i, g.dOg, {batch, qi, q_head, 0});

                rt<float, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> dP_ij;
                zero(dP_ij);
                mma_ABt(dP_ij, dO_i, V_j, dP_ij);

                // dS = P * (dP - delta)
                {
                    float *d_ptr = (float*)g.delta_vec.raw_ptr;
                    const int stride_b = g.delta_vec.template stride<0>();
                    const int stride_h = g.delta_vec.template stride<1>();
                    const int base_idx = batch * stride_b + q_head * stride_h + qi * Q_TILE;
                    const int G = lane >> 5;
                    #pragma unroll
                    for (int k = 0; k < dP_ij.tiles[0][0].packed_per_thread; k++) {
                        int ro = ((k >> 1) << 3) + ((k & 1) << 1);
                        int row0 = G * 4 + ro;
                        int row1 = row0 + 1;
                        dP_ij.tiles[0][0].data[k].x -= d_ptr[base_idx + row0];
                        dP_ij.tiles[0][0].data[k].y -= d_ptr[base_idx + row1];
                    }
                }
                mul(dP_ij, dP_ij, S_ij);

                // Phase 3: dV += dO^T @ P
                rt<bf16, Q_TILE, D_V, col_l, rt_16x32_4_s> dO_col;
                load<QKVO_AXIS, decltype(dO_col), _gl>(dO_col, g.dOg, {batch, qi, q_head, 0});

                rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> P_bf;
                copy(P_bf, S_ij);
                auto &Pm = *reinterpret_cast<
                    rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_16x32_4_s>*>(&P_bf);

                mma_AtB(dV_acc, dO_col, Pm, dV_acc);

                // Phase 4: dK += Q^T @ dS
                mul(dP_ij, dP_ij, dP_SCALE);

                rt<bf16, Q_TILE, D_QK, col_l, rt_16x32_4_s> Q_col;
                load<QKVO_AXIS, decltype(Q_col), _gl>(Q_col, g.Q, {batch, qi, q_head, 0});

                rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> dS_bf;
                copy(dS_bf, dP_ij);
                auto &dSm = *reinterpret_cast<
                    rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_16x32_4_s>*>(&dS_bf);

                mma_AtB(dK_acc, Q_col, dSm, dK_acc);
            }
        } // qho
    } // qi

    // Epilogue
    store_col_l_direct<QKVO_AXIS, D_V>(g.dVg, dV_acc, batch, j, kv_head);
    store_col_l_direct<QKVO_AXIS, D_QK>(g.dKg, dK_acc, batch, j, kv_head);
}

void dispatch_bwd_combined_d192v128(attn_bwd_combined_d192v128_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute(
        (void*)attend_bwd_combined_d192v128_ker,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_bwd_combined_d192v128_ker<<<
        g.grid(), g.block(), mem_size, g.stream>>>(g);
}

// Dummy dQ kernel (kept for pybind compatibility, actual dQ is done by separate agent)
struct attn_bwd_dq_d192v128_globals {
    _gl Q, K, V;
    _gl dOg, dQg;
    gl<float, -1, -1, -1, -1> L_vec, delta_vec;
    hipStream_t stream;
    dim3 grid()  { return dim3(ATTN_H_KV, (ATTN_N / BLOCK_KV), ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__launch_bounds__(NUM_THREADS, 1)
__global__ void attend_bwd_dq_d192v128_ker(const attn_bwd_dq_d192v128_globals g) {
    // Placeholder - dQ handled by separate agent
}

void dispatch_bwd_dq_d192v128(attn_bwd_dq_d192v128_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute(
        (void*)attend_bwd_dq_d192v128_ker,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_bwd_dq_d192v128_ker<<<
        g.grid(), g.block(), mem_size, g.stream>>>(g);
}

PYBIND11_MODULE(tk_kernel_bkwd, m) {
    m.doc() = "tk_kernel python module - asymmetric backward D_QK=192 D_V=128 (opt2)";
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
    py::bind_function<dispatch_bwd_dq_d192v128>(m, "dispatch_bwd_dq",
        &attn_bwd_dq_d192v128_globals::Q,
        &attn_bwd_dq_d192v128_globals::K,
        &attn_bwd_dq_d192v128_globals::V,
        &attn_bwd_dq_d192v128_globals::dOg,
        &attn_bwd_dq_d192v128_globals::dQg,
        &attn_bwd_dq_d192v128_globals::L_vec,
        &attn_bwd_dq_d192v128_globals::delta_vec
    );
}
