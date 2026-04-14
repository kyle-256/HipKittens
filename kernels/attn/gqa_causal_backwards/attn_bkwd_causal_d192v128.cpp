// MLA Backward Kernel (D_QK=192, D_V=128) — causal, dK/dV only (no dQ yet)
// Modeled after attn_bkwd_mla_simple.cpp which works on gfx950.
// Uses rt_32x32_s base tiles, amdgpu_num_vgpr(29), hardcoded AGPR reads for dK.
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

// Hardcoded AGPR read epilogue for dK (192×32 = 96 AGPRs starting at a[0])
template<int QKVO_AXIS>
__device__ __forceinline__ void store_dK_from_agpr(
    const _gl &dKg, int batch, int j, int kv_head)
{
    const int lane = laneid();
    const int kv_col = lane & 31;
    const int rb = (lane >> 5) << 2;

    bf16 *base = reinterpret_cast<bf16*>(dKg.raw_ptr);
    const int stride_b = dKg.template stride<0>();
    const int stride_s = dKg.template stride<QKVO_AXIS>();
    const int stride_h = dKg.template stride<2>();
    const int base_off = batch * stride_b + (j * KV_BLOCK + kv_col) * stride_s + kv_head * stride_h;

    float vals[96];
    #pragma unroll
    for (int i = 0; i < 96; i++)
        asm volatile("v_accvgpr_read_b32 %0, a[%1]" : "=v"(vals[i]) : "n"(i));

    #pragma unroll
    for (int t = 0; t < D_QK / 32; t++) {
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int ro = ((k >> 1) << 3) + ((k & 1) << 1);
            int d0 = t * 32 + rb + ro;
            int reg_idx = t * 16 + k * 2;
            base[base_off + d0]     = __float2bfloat16(vals[reg_idx]);
            base[base_off + d0 + 1] = __float2bfloat16(vals[reg_idx + 1]);
        }
    }
}

struct attn_bwd_combined_d192v128_globals {
  _gl Q, K, V;
  _gl dOg, dQg, dKg, dVg;
  gl<float, -1, -1, -1, -1> L_vec, delta_vec;
  hipStream_t stream;
  dim3 grid() { return dim3(ATTN_H_KV, (ATTN_N / BLOCK_KV), ATTN_B); }
  dim3 block() { return dim3(NUM_THREADS); }
  size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__launch_bounds__(NUM_THREADS, 1)
__global__ __attribute__((amdgpu_num_vgpr(29))) void attend_bwd_combined_d192v128_ker(const attn_bwd_combined_d192v128_globals g) {

    constexpr int QKVO_AXIS = 1; // BNHD layout
    constexpr float P_SCALE  = 0.07216878365f * 1.44269504089f;
    constexpr float L_SCALE  = 1.44269504089f;
    constexpr float dP_SCALE = 0.07216878365f;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    st_bf<BLOCK_KV, D_QK, st_32x32_s> (&K_smem)      = al.allocate<st_bf<BLOCK_KV, D_QK, st_32x32_s>>();
    st_bf<BLOCK_KV, D_V,  st_32x32_s> (&V_smem)      = al.allocate<st_bf<BLOCK_KV, D_V,  st_32x32_s>>();
    st_bf<Q_TILE, D_QK, st_32x32_s>   (&Q_smem_row)  = al.allocate<st_bf<Q_TILE, D_QK, st_32x32_s>>();
    st_bf<Q_TILE, D_QK, st_8x32_s>    (&Q_smem_col)  = al.allocate<st_bf<Q_TILE, D_QK, st_8x32_s>>();
    st_bf<Q_TILE, D_V,  st_32x32_s>   (&dO_smem_row) = al.allocate<st_bf<Q_TILE, D_V, st_32x32_s>>();
    st_bf<Q_TILE, D_V,  st_8x32_s>    (&dO_smem_col) = al.allocate<st_bf<Q_TILE, D_V, st_8x32_s>>();
    sv_fl<Q_TILE> (&L_smem)     = al.allocate<sv_fl<Q_TILE>>();
    sv_fl<Q_TILE> (&delta_smem) = al.allocate<sv_fl<Q_TILE>>();

    rt<float, D_QK, KV_BLOCK, col_l, rt_32x32_s> dK_acc;
    rt<float, D_V,  KV_BLOCK, col_l, rt_32x32_s> dV_acc;
    zero(dK_acc);
    zero(dV_acc);

    const int kv_head = blockIdx.x, seq_block = blockIdx.y, batch = blockIdx.z;
    const int wid = kittens::warpid(), j = seq_block * NUM_WARPS + wid;
    const int kv_start = j * KV_BLOCK;
    const int total_q  = ATTN_N / Q_TILE;
    const int first_q = causal ? max(0, (int)(seq_block * BLOCK_KV / Q_TILE)) : 0;

    G::load<QKVO_AXIS, false>(K_smem, g.K, {batch, seq_block, kv_head, 0});
    G::load<QKVO_AXIS, false>(V_smem, g.V, {batch, seq_block, kv_head, 0});
    __builtin_amdgcn_s_waitcnt(0); __builtin_amdgcn_s_barrier();

    for (int qi = first_q; qi < total_q; qi++) {
        for (int qho = 0; qho < GROUP_SIZE; qho++) {
            const int q_head = kv_head * GROUP_SIZE + qho;
            const int q_pos = qi * Q_TILE;
            const bool skip = causal && (q_pos + Q_TILE <= kv_start);

            load(L_smem, g.L_vec, {batch, q_head, 0, qi});
            load(delta_smem, g.delta_vec, {batch, q_head, 0, qi});

            // Phase 1: S_ij = Q @ K^T * scale - L → P = exp2(S)
            G::load<QKVO_AXIS, false>(Q_smem_row, g.Q, {batch, qi, q_head, 0});
            __builtin_amdgcn_s_waitcnt(0); __builtin_amdgcn_s_barrier();
            rt<float, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> S_ij;
            if (!skip) {
                rt<bf16, Q_TILE, D_QK, row_l, rt_32x16_s> Q_i;
                load(Q_i, Q_smem_row);
                rt<bf16, KV_BLOCK, D_QK, row_l, rt_32x16_s> K_j;
                load(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {wid, 0}));
                zero(S_ij); mma_ABt(S_ij, Q_i, K_j, S_ij); mul(S_ij, S_ij, P_SCALE);
                typename decltype(S_ij)::col_vec L_reg;
                load(L_reg, L_smem); mul(L_reg, L_reg, L_SCALE); sub_row(S_ij, S_ij, L_reg);

                // Causal mask
                if constexpr (causal) {
                    #pragma unroll
                    for (int ii = 0; ii < S_ij.height; ii++)
                        #pragma unroll
                        for (int jj = 0; jj < S_ij.width; jj++)
                            #pragma unroll
                            for (int k = 0; k < S_ij.tiles[ii][jj].packed_per_thread; k++) {
                                int q_r = (ii * 32) + (laneid() & 31);
                                int k_c = (jj * 32) + ((laneid() >> 5) << 2) + ((k >> 1) << 3) + ((k & 1) << 1);
                                if (q_pos + q_r < kv_start + k_c)
                                    S_ij.tiles[ii][jj].data[k].x = -__builtin_inff();
                                if (q_pos + q_r < kv_start + k_c + 1)
                                    S_ij.tiles[ii][jj].data[k].y = -__builtin_inff();
                            }
                }

                // Clamp <= 0 then exp2
                #pragma unroll
                for (int ii = 0; ii < S_ij.height; ii++)
                    #pragma unroll
                    for (int jj = 0; jj < S_ij.width; jj++)
                        #pragma unroll
                        for (int k = 0; k < S_ij.tiles[ii][jj].packed_per_thread; k++) {
                            S_ij.tiles[ii][jj].data[k].x = __builtin_fminf(S_ij.tiles[ii][jj].data[k].x, 0.f);
                            S_ij.tiles[ii][jj].data[k].y = __builtin_fminf(S_ij.tiles[ii][jj].data[k].y, 0.f);
                            S_ij.tiles[ii][jj].data[k] = base_ops::exp2::op(S_ij.tiles[ii][jj].data[k]);
                        }
            }

            // Phase 2: dP = dO @ V^T, dS = P * (dP - delta)
            G::load<QKVO_AXIS, false>(dO_smem_row, g.dOg, {batch, qi, q_head, 0});
            __builtin_amdgcn_s_waitcnt(0); __builtin_amdgcn_s_barrier();
            rt<float, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> dP_ij;
            if (!skip) {
                rt<bf16, Q_TILE, D_V, row_l, rt_32x16_s> dO_i;
                load(dO_i, dO_smem_row);
                rt<bf16, KV_BLOCK, D_V, row_l, rt_32x16_s> V_j;
                load(V_j, subtile_inplace<KV_BLOCK, D_V>(V_smem, {wid, 0}));
                zero(dP_ij); mma_ABt(dP_ij, dO_i, V_j, dP_ij);
                typename decltype(dP_ij)::col_vec d_reg;
                load(d_reg, delta_smem); sub_row(dP_ij, dP_ij, d_reg);
                mul(dP_ij, dP_ij, S_ij); // dS = P * (dP - delta)
            }

            // Phase 3: dV += dO^T @ P
            G::load<QKVO_AXIS, false>(dO_smem_col, g.dOg, {batch, qi, q_head, 0});
            __builtin_amdgcn_s_waitcnt(0); __builtin_amdgcn_s_barrier();
            if (!skip) {
                rt<bf16, Q_TILE, D_V, col_l, rt_16x32_4_s> dO_col;
                load(dO_col, dO_smem_col);
                rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> P_bf;
                copy(P_bf, S_ij);
                auto &Pm = *reinterpret_cast<rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_16x32_4_s>*>(&P_bf);
                mma_AtB(dV_acc, dO_col, Pm, dV_acc);
            }

            // Phase 4: dK += Q^T @ dS
            G::load<QKVO_AXIS, false>(Q_smem_col, g.Q, {batch, qi, q_head, 0});
            __builtin_amdgcn_s_waitcnt(0); __builtin_amdgcn_s_barrier();
            if (!skip) {
                mul(dP_ij, dP_ij, dP_SCALE);
                rt<bf16, Q_TILE, D_QK, col_l, rt_16x32_4_s> Q_col;
                load(Q_col, Q_smem_col);
                rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> dS_bf;
                copy(dS_bf, dP_ij);
                auto &dSm = *reinterpret_cast<rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_16x32_4_s>*>(&dS_bf);
                mma_AtB(dK_acc, Q_col, dSm, dK_acc);
            }

            __builtin_amdgcn_s_barrier();
        }
    }

    // Epilogue: store dV via standard transpose, dK via hardcoded AGPR reads
    {
        rt<float, KV_BLOCK, D_V, row_l, rt_32x32_s> dV_out;
        transpose(dV_out, dV_acc);
        rt<bf16, KV_BLOCK, D_V, row_l, rt_32x32_s> dV_bf;
        copy(dV_bf, dV_out);
        store<QKVO_AXIS>(g.dVg, dV_bf, {batch, j, kv_head, 0});
    }
    store_dK_from_agpr<QKVO_AXIS>(g.dKg, batch, j, kv_head);
}

void dispatch_bwd_combined_d192v128(attn_bwd_combined_d192v128_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_bwd_combined_d192v128_ker, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_bwd_combined_d192v128_ker<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
    hipDeviceSynchronize();
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
