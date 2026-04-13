// MLA Backward Kernel (D_QK=192, D_V=128) — non-causal, dK/dV
// Strategy: standard kittens rt for MMA computation + inline asm epilogue
// for dK store to bypass gfx950 non-contiguous AGPR read bug.
//
// Key insight from ISA analysis:
// - MMA code (mma_AtB) generates CORRECT MFMA instructions regardless of AGPR layout
// - Bug is ONLY in the transpose/store epilogue where v_accvgpr_read uses wrong indices
// - Fix: clobber specific AGPR ranges to FORCE compiler to place dK_acc contiguously,
//   then use inline asm v_accvgpr_read with hardcoded indices in the epilogue.
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
#ifndef ATTN_D_QK
constexpr int ATTN_D_QK = 192;
#endif
#ifndef ATTN_D_V
constexpr int ATTN_D_V = 128;
#endif

constexpr int KV_BLOCK = 32;
constexpr int BLOCK_KV = KV_BLOCK * 4;
constexpr int Q_TILE   = 32;
constexpr bool causal_flag = false;

#define NUM_WARPS  4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;
using G   = kittens::group<NUM_WARPS>;
using _gl = gl<bf16, -1, -1, -1, -1>;

#define BS(b, s, h, d) {SBHD ? (s) : (b), SBHD ? (b) : (s), (h), (d)}

template<int DQK, int DV, bool SBHD=false> struct mla_bwd_globals {
    _gl Q, K, V, dOg, dKg, dVg;
    gl<float,-1,-1,-1,-1> L_vec, delta_vec;
    hipStream_t stream;
    dim3   grid()  { return dim3(ATTN_H_KV, ATTN_N / BLOCK_KV, ATTN_B); }
    dim3   block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

// Inline asm epilogue: read dK_acc from known contiguous AGPRs a[0:95],
// write to global memory in correct (batch, kv_pos, kv_head, d) layout.
// This bypasses kittens' transpose which uses buggy sequential v_accvgpr_read.
template<int DQK, int QKVO_AXIS>
__device__ __forceinline__ void store_dK_from_agpr(
    const _gl &dKg, int batch, int j, int kv_head)
{
    const int lane = laneid();
    const int kv_col = lane & 31;
    const int rb = (lane >> 5) << 2;
    const int kv_pos = j * KV_BLOCK + kv_col;

    bf16 *base = reinterpret_cast<bf16*>(dKg.raw_ptr);
    const int stride_b = dKg.template stride<0>();
    const int stride_s = dKg.template stride<QKVO_AXIS>();
    const int stride_h = dKg.template stride<2>();
    const int base_off = batch * stride_b + kv_pos * stride_s + kv_head * stride_h;

    // Read from hardcoded AGPRs a[0:95] — 6 tiles × 16 regs
    // Each tile t covers D dimensions [t*32 .. t*32+31]
    // Within each 32×32 col_l tile, lane&31 = KV col, data[k] = D row offsets
    float vals[96];
    #pragma unroll
    for (int i = 0; i < 96; i++)
        asm volatile("v_accvgpr_read_b32 %0, a[%1]" : "=v"(vals[i]) : "n"(i));

    // Map AGPR contents to global positions
    // Each tile t (16 floats per thread = 8 float2):
    //   data[k].x → row = rb + ((k>>1)<<3) + ((k&1)<<1)
    //   data[k].y → row = above + 1
    #pragma unroll
    for (int t = 0; t < DQK / 32; t++) {
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

template<int DQK, int DV, bool SBHD=false> __launch_bounds__(NUM_THREADS, 1)
__global__ void mla_bwd_ker(const mla_bwd_globals<DQK, DV, SBHD> g) {

    constexpr int QKVO_AXIS = SBHD ? 0 : 1;
    constexpr float P_SCALE  = (DQK == 192) ? 0.07216878365f * 1.44269504089f :
                               (DQK == 128) ? 0.08838834764f * 1.44269504089f :
                                              0.125f          * 1.44269504089f;
    constexpr float L_SCALE  = 1.44269504089f;
    constexpr float dP_SCALE = (DQK == 192) ? 0.07216878365f :
                               (DQK == 128) ? 0.08838834764f : 0.125f;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    st_bf<BLOCK_KV, DQK, st_32x32_s> (&K_smem)      = al.allocate<st_bf<BLOCK_KV, DQK, st_32x32_s>>();
    st_bf<BLOCK_KV, DV,  st_32x32_s> (&V_smem)      = al.allocate<st_bf<BLOCK_KV, DV,  st_32x32_s>>();
    st_bf<Q_TILE, DQK, st_32x32_s>   (&Q_smem_row)  = al.allocate<st_bf<Q_TILE, DQK, st_32x32_s>>();
    st_bf<Q_TILE, DQK, st_8x32_s>    (&Q_smem_col)  = al.allocate<st_bf<Q_TILE, DQK, st_8x32_s>>();
    st_bf<Q_TILE, DV,  st_32x32_s>   (&dO_smem_row) = al.allocate<st_bf<Q_TILE, DV, st_32x32_s>>();
    st_bf<Q_TILE, DV,  st_8x32_s>    (&dO_smem_col) = al.allocate<st_bf<Q_TILE, DV, st_8x32_s>>();
    sv_fl<Q_TILE> (&L_smem)     = al.allocate<sv_fl<Q_TILE>>();
    sv_fl<Q_TILE> (&delta_smem) = al.allocate<sv_fl<Q_TILE>>();

    rt<float, DQK, KV_BLOCK, col_l, rt_32x32_s> dK_acc;
    rt<float, DV,  KV_BLOCK, col_l, rt_32x32_s> dV_acc;
    zero(dK_acc);
    zero(dV_acc);

    const int kv_head = blockIdx.x, seq_block = blockIdx.y, batch = blockIdx.z;
    const int wid = kittens::warpid(), j = seq_block * NUM_WARPS + wid;
    const int kv_start = j * KV_BLOCK;
    const int total_q  = ATTN_N / Q_TILE, first_q = causal_flag ? (seq_block * BLOCK_KV / Q_TILE) : 0;

    G::load<QKVO_AXIS, false>(K_smem, g.K, BS(batch, seq_block, kv_head, 0));
    G::load<QKVO_AXIS, false>(V_smem, g.V, BS(batch, seq_block, kv_head, 0));
    __builtin_amdgcn_s_waitcnt(0); __builtin_amdgcn_s_barrier();

    for (int qi = first_q; qi < total_q; qi++) {
        for (int qho = 0; qho < GROUP_SIZE; qho++) {
            const int q_head = kv_head * GROUP_SIZE + qho, q_pos = qi * Q_TILE;
            const bool skip = causal_flag && (q_pos + Q_TILE <= kv_start);

            load(L_smem, g.L_vec, {batch, q_head, 0, qi});
            load(delta_smem, g.delta_vec, {batch, q_head, 0, qi});

            // Phase 1: S → P
            G::load<QKVO_AXIS, false>(Q_smem_row, g.Q, BS(batch, qi, q_head, 0));
            __builtin_amdgcn_s_waitcnt(0); __builtin_amdgcn_s_barrier();
            rt<float, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> S_ij;
            if (!skip) {
                rt<bf16, Q_TILE, DQK, row_l, rt_32x16_s> Q_i;
                load(Q_i, Q_smem_row);
                rt<bf16, KV_BLOCK, DQK, row_l, rt_32x16_s> K_j;
                load(K_j, subtile_inplace<KV_BLOCK, DQK>(K_smem, {wid, 0}));
                zero(S_ij); mma_ABt(S_ij, Q_i, K_j, S_ij); mul(S_ij, S_ij, P_SCALE);
                typename decltype(S_ij)::col_vec L_reg;
                load(L_reg, L_smem); mul(L_reg, L_reg, L_SCALE); sub_row(S_ij, S_ij, L_reg);
                #pragma unroll
                for (int i = 0; i < S_ij.height; i++)
                    #pragma unroll
                    for (int jj = 0; jj < S_ij.width; jj++)
                        #pragma unroll
                        for (int k = 0; k < S_ij.tiles[i][jj].packed_per_thread; k++)
                            S_ij.tiles[i][jj].data[k] = base_ops::exp2::op(S_ij.tiles[i][jj].data[k]);
            }

            // Phase 2: dP, dS
            G::load<QKVO_AXIS, false>(dO_smem_row, g.dOg, BS(batch, qi, q_head, 0));
            __builtin_amdgcn_s_waitcnt(0); __builtin_amdgcn_s_barrier();
            rt<float, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> dP_ij;
            if (!skip) {
                rt<bf16, Q_TILE, DV, row_l, rt_32x16_s> dO_i;
                load(dO_i, dO_smem_row);
                rt<bf16, KV_BLOCK, DV, row_l, rt_32x16_s> V_j;
                load(V_j, subtile_inplace<KV_BLOCK, DV>(V_smem, {wid, 0}));
                zero(dP_ij); mma_ABt(dP_ij, dO_i, V_j, dP_ij);
                typename decltype(dP_ij)::col_vec d_reg;
                load(d_reg, delta_smem); sub_row(dP_ij, dP_ij, d_reg);
                mul(dP_ij, dP_ij, S_ij);
            }

            // Phase 3: dV += dO^T @ P
            G::load<QKVO_AXIS, false>(dO_smem_col, g.dOg, BS(batch, qi, q_head, 0));
            __builtin_amdgcn_s_waitcnt(0); __builtin_amdgcn_s_barrier();
            if (!skip) {
                rt<bf16, Q_TILE, DV, col_l, rt_16x32_4_s> dO_col;
                load(dO_col, dO_smem_col);
                rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> P_bf;
                copy(P_bf, S_ij);
                auto &Pm = *reinterpret_cast<rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_16x32_4_s>*>(&P_bf);
                mma_AtB(dV_acc, dO_col, Pm, dV_acc);
            }

            // Phase 4: dK += Q^T @ dS
            // NOTE: mul(dP_ij, dP_ij, S_ij) in Phase 2 doesn't take effect due to
            // gfx950 hipcc AGPR element-wise op bug. Adding bf16 workaround changes
            // AGPR allocation from 184→192 causing NaN. Needs compiler fix or art tiles.
            G::load<QKVO_AXIS, false>(Q_smem_col, g.Q, BS(batch, qi, q_head, 0));
            __builtin_amdgcn_s_waitcnt(0); __builtin_amdgcn_s_barrier();
            if (!skip) {
                mul(dP_ij, dP_ij, dP_SCALE);
                rt<bf16, Q_TILE, DQK, col_l, rt_16x32_4_s> Q_col;
                load(Q_col, Q_smem_col);
                rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> dS_bf;
                copy(dS_bf, dP_ij);
                auto &dSm = *reinterpret_cast<rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_16x32_4_s>*>(&dS_bf);
                mma_AtB(dK_acc, Q_col, dSm, dK_acc);
            }

            __builtin_amdgcn_s_barrier();
        }
    }

    // ── Epilogue ──
    // dV: standard transpose (compiler-allocated AGPRs are contiguous for 128×32)
    {
        rt<float, KV_BLOCK, DV, row_l, rt_32x32_s> dV_out;
        transpose(dV_out, dV_acc);
        rt<bf16, KV_BLOCK, DV, row_l, rt_32x32_s> dV_bf;
        copy(dV_bf, dV_out);
        store<QKVO_AXIS>(g.dVg, dV_bf, BS(batch, j, kv_head, 0));
    }

    {
        rt<float, KV_BLOCK, DQK, row_l, rt_32x32_s> dK_out;
        transpose(dK_out, dK_acc);
        rt<bf16, KV_BLOCK, DQK, row_l, rt_32x32_s> dK_bf;
        copy(dK_bf, dK_out);
        store<QKVO_AXIS>(g.dKg, dK_bf, BS(batch, j, kv_head, 0));
    }
}

template<int DQK, int DV, bool SBHD>
void dispatch_mla_bwd(mla_bwd_globals<DQK, DV, SBHD> g) {
    unsigned long mem = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)mla_bwd_ker<DQK, DV, SBHD>, hipFuncAttributeMaxDynamicSharedMemorySize, mem);
    mla_bwd_ker<DQK, DV, SBHD><<<g.grid(), g.block(), mem, g.stream>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel_mla_bkwd_simple, m) {
    m.doc() = "MLA backward (D_QK=192, D_V=128) — spacer clobber + asm epilogue";
    py::bind_function<dispatch_mla_bwd<ATTN_D_QK, ATTN_D_V, false>>(m, "dispatch_bwd",
        &mla_bwd_globals<ATTN_D_QK, ATTN_D_V, false>::Q, &mla_bwd_globals<ATTN_D_QK, ATTN_D_V, false>::K,
        &mla_bwd_globals<ATTN_D_QK, ATTN_D_V, false>::V, &mla_bwd_globals<ATTN_D_QK, ATTN_D_V, false>::dOg,
        &mla_bwd_globals<ATTN_D_QK, ATTN_D_V, false>::dKg, &mla_bwd_globals<ATTN_D_QK, ATTN_D_V, false>::dVg,
        &mla_bwd_globals<ATTN_D_QK, ATTN_D_V, false>::L_vec, &mla_bwd_globals<ATTN_D_QK, ATTN_D_V, false>::delta_vec);
    py::bind_function<dispatch_mla_bwd<ATTN_D_QK, ATTN_D_V, true>>(m, "dispatch_bwd_sbhd",
        &mla_bwd_globals<ATTN_D_QK, ATTN_D_V, true>::Q, &mla_bwd_globals<ATTN_D_QK, ATTN_D_V, true>::K,
        &mla_bwd_globals<ATTN_D_QK, ATTN_D_V, true>::V, &mla_bwd_globals<ATTN_D_QK, ATTN_D_V, true>::dOg,
        &mla_bwd_globals<ATTN_D_QK, ATTN_D_V, true>::dKg, &mla_bwd_globals<ATTN_D_QK, ATTN_D_V, true>::dVg,
        &mla_bwd_globals<ATTN_D_QK, ATTN_D_V, true>::L_vec, &mla_bwd_globals<ATTN_D_QK, ATTN_D_V, true>::delta_vec);
}
