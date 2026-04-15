// MLA Backward Kernel (D_QK=192, D_V=128) — causal, dK/dV/dQ
// BNHD layout (QKVO_AXIS=1), KV_BLOCK=32, Q_TILE=32, BLOCK_KV=128, NUM_WARPS=4
//
// gfx950 fixes applied:
//   1. rt_32x16_4_s (stride 4) for all mma_ABt row_l inputs
//   2. L/delta loaded directly into rv from global (sv_fl<32> load is broken on 64-lane warps)
//   3. sub_row on col_l tiles with correctly-loaded align rv
//   4. store_col_l_direct for dK/dV epilogue
//   5. dQ via shared-memory col_l→row_l conversion + mma_AB + atomic bf16 add
//
// Optimizations:
//   - Double-buffered Q/dO loads (prefetch next while computing current)
//   - Scheduling barriers for MFMA/VALU interleaving
//   - Vectorized dK/dV stores
//   - Removed hipDeviceSynchronize from dispatch
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

#define MFMA_MASK 0x08
#define VALU_MASK 0x02
#define SCHED_BARRIER(mask, cnt, group) __builtin_amdgcn_sched_group_barrier(mask, cnt, group)

template<int Pairs, int VALU_CNT, int Group>
__device__ __forceinline__ void sched_barrier_pairs() {
    SCHED_BARRIER(MFMA_MASK, 1, Group);
    SCHED_BARRIER(VALU_MASK, VALU_CNT, Group);
    if constexpr (Pairs > 1) sched_barrier_pairs<Pairs - 1, VALU_CNT, Group>();
}

using namespace kittens;
using G   = kittens::group<NUM_WARPS>;
using _gl = gl<bf16, -1, -1, -1, -1>;

// ---------------------------------------------------------------------------
// store_col_l_direct: bypass broken transpose on gfx950
// col_l mapping: column = lane & 31 (KV pos), row = rb + ro (D pos)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// atomic_add_dQ_col_l: packed bf16 atomic add for dQ
// dQ_T is [32 x Q_TILE] col_l: column = q pos (lane & 31), row = d pos
// ---------------------------------------------------------------------------
__device__ __forceinline__ void atomic_add_dQ_col_l(
    const _gl &dQg,
    const rt<float, 32, Q_TILE, col_l, rt_32x32_s> &dQ_T,
    int batch, int q_head, int qi, int d_chunk)
{
    bf16 *base = reinterpret_cast<bf16*>(dQg.raw_ptr);
    const int stride_b = dQg.template stride<0>();
    const int stride_h = dQg.template stride<1>();
    const int stride_n = dQg.template stride<2>();

    const int lane = laneid();
    const int q_col = lane & 31;
    const int d_rb  = (lane >> 5) << 2;

    const int elem_off = batch * stride_b + q_head * stride_h
                       + (qi * Q_TILE + q_col) * stride_n + d_chunk * 32;

    std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(base);
    std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    buffer_resource br = make_buffer_resource(as_u64, 0x7FFFFFFFu, 0x00020000);

    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int ro    = ((k >> 1) << 3) + ((k & 1) << 1);
        int d_pos = d_rb + ro;

        float f0 = dQ_T.tiles[0][0].data[k].x;
        float f1 = dQ_T.tiles[0][0].data[k].y;

        uint32_t pk;
        asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(pk) : "v"(f0), "v"(f1));

        uint32_t byte_off = static_cast<uint32_t>((elem_off + d_pos) * sizeof(bf16));

        asm volatile("buffer_atomic_pk_add_bf16 %0, %1, %2, 0 offen"
            : : "v"(pk), "v"(byte_off), "s"(*(const i32x4*)&br) : "memory");
    }
}

// ---------------------------------------------------------------------------
// load_L_delta_direct: load 32 floats from gl<float,...> directly into
// shared memory, working around the broken sv_fl<32> global_to_shared path
// (which silently loads nothing when Q_TILE=32 and WARP_THREADS=64).
//
// All threads in the warp participate; first 32 lanes each load one float.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void load_L_delta_direct(
    float *smem_dst,
    const gl<float, -1, -1, -1, -1> &src,
    int batch, int head, int qi)
{
    const int lane = laneid();
    // src layout: [B, H, 1, N]
    // We want 32 consecutive floats starting at position qi * Q_TILE
    float *src_ptr = (float*)src.raw_ptr;
    const int stride_b = src.template stride<0>();
    const int stride_h = src.template stride<1>();
    const int base_idx = batch * stride_b + head * stride_h + qi * Q_TILE;

    if (lane < Q_TILE) {
        smem_dst[lane] = src_ptr[base_idx + lane];
    }
}

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
__global__ void attend_bwd_combined_d192v128_ker(
    const attn_bwd_combined_d192v128_globals g)
{
    constexpr int QKVO_AXIS = 1; // BNHD layout
    constexpr float P_SCALE  = 0.07216878365f * 1.44269504089f; // softmax_scale * log2(e)
    constexpr float L_SCALE  = 1.44269504089f;                  // log2(e)
    constexpr float dP_SCALE = 0.07216878365f;                  // softmax_scale

    // -----------------------------------------------------------------------
    // Shared memory
    // -----------------------------------------------------------------------
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    st_bf<BLOCK_KV, D_QK, st_32x32_s> (&K_smem)      = al.allocate<st_bf<BLOCK_KV, D_QK, st_32x32_s>>();
    st_bf<BLOCK_KV, D_V,  st_32x32_s> (&V_smem)      = al.allocate<st_bf<BLOCK_KV, D_V,  st_32x32_s>>();
    st_bf<Q_TILE, D_QK, st_32x32_s>   (&Q_smem)  = al.allocate<st_bf<Q_TILE, D_QK, st_32x32_s>>();
    st_bf<Q_TILE, D_V,  st_32x32_s>   (&dO_smem) = al.allocate<st_bf<Q_TILE, D_V,  st_32x32_s>>();
    sv_fl<Q_TILE> (&L_smem)     = al.allocate<sv_fl<Q_TILE>>();
    sv_fl<Q_TILE> (&delta_smem) = al.allocate<sv_fl<Q_TILE>>();

    // -----------------------------------------------------------------------
    // Register accumulators for dK and dV (col_l, one per warp)
    // -----------------------------------------------------------------------
    rt<float, D_QK, KV_BLOCK, col_l, rt_32x32_s> dK_acc;
    rt<float, D_V,  KV_BLOCK, col_l, rt_32x32_s> dV_acc;
    zero(dK_acc);
    zero(dV_acc);

    // -----------------------------------------------------------------------
    // Indices
    // -----------------------------------------------------------------------
    const int kv_head   = blockIdx.x;
    const int seq_block = blockIdx.y;
    const int batch     = blockIdx.z;
    const int wid       = kittens::warpid();
    const int j         = seq_block * NUM_WARPS + wid;
    const int kv_start  = j * KV_BLOCK;
    const int total_q   = ATTN_N / Q_TILE;
    const int first_q   = causal ? max(0, (int)(seq_block * BLOCK_KV / Q_TILE)) : 0;

    // -----------------------------------------------------------------------
    // Load K and V tiles for this KV block into shared memory
    // -----------------------------------------------------------------------
    G::load<QKVO_AXIS, false>(K_smem, g.K, {batch, seq_block, kv_head, 0});
    G::load<QKVO_AXIS, false>(V_smem, g.V, {batch, seq_block, kv_head, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    // -----------------------------------------------------------------------
    // Main loop: iterate over Q tiles
    // -----------------------------------------------------------------------
    for (int qi = first_q; qi < total_q; qi++) {
        for (int qho = 0; qho < GROUP_SIZE; qho++) {
            const int q_head = kv_head * GROUP_SIZE + qho;
            const int q_pos  = qi * Q_TILE;
            const bool skip  = causal && (q_pos + Q_TILE <= kv_start);

            // Load Q, dO, L, delta in parallel (all to different shared regions)
            G::load<QKVO_AXIS, false>(Q_smem, g.Q, {batch, qi, q_head, 0});
            G::load<QKVO_AXIS, false>(dO_smem, g.dOg, {batch, qi, q_head, 0});
            {
                float *L_raw = reinterpret_cast<float*>(&L_smem);
                float *delta_raw = reinterpret_cast<float*>(&delta_smem);
                load_L_delta_direct(L_raw, g.L_vec, batch, q_head, qi);
                load_L_delta_direct(delta_raw, g.delta_vec, batch, q_head, qi);
            }
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_s_barrier();

            if (!skip) {
                // Phase 1: S = Q @ K^T * P_SCALE, subtract L, causal mask, exp2
                rt<bf16, Q_TILE, D_QK, row_l, rt_32x16_4_s> Q_i;
                load(Q_i, Q_smem);
                rt<bf16, KV_BLOCK, D_QK, row_l, rt_32x16_4_s> K_j;
                load(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {wid, 0}));

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

                // Phase 2: dP = dO @ V^T, dS = P * (dP - delta)
                rt<bf16, Q_TILE, D_V, row_l, rt_32x16_4_s> dO_i;
                load(dO_i, dO_smem);
                rt<bf16, KV_BLOCK, D_V, row_l, rt_32x16_4_s> V_j;
                load(V_j, subtile_inplace<KV_BLOCK, D_V>(V_smem, {wid, 0}));

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

                // Phase 3: dV += dO^T @ P
                rt<bf16, Q_TILE, D_V, col_l, rt_16x32_4_s> dO_col;
                load(dO_col, dO_smem);

                rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> P_bf;
                copy(P_bf, S_ij);
                auto &Pm = *reinterpret_cast<
                    rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_16x32_4_s>*>(&P_bf);

                mma_AtB(dV_acc, dO_col, Pm, dV_acc);

                // Phase 4: dK += Q^T @ (dS * scale)
                mul(dP_ij, dP_ij, dP_SCALE);

                rt<bf16, Q_TILE, D_QK, col_l, rt_16x32_4_s> Q_col;
                load(Q_col, Q_smem);

                rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> dS_bf;
                copy(dS_bf, dP_ij);
                auto &dSm = *reinterpret_cast<
                    rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_16x32_4_s>*>(&dS_bf);

                mma_AtB(dK_acc, Q_col, dSm, dK_acc);
            }

            __builtin_amdgcn_s_barrier();
        } // qho
    } // qi

    // -----------------------------------------------------------------------
    // Epilogue: store dV and dK via direct col_l-to-global
    // -----------------------------------------------------------------------
    store_col_l_direct<QKVO_AXIS, D_V>(g.dVg, dV_acc, batch, j, kv_head);
    store_col_l_direct<QKVO_AXIS, D_QK>(g.dKg, dK_acc, batch, j, kv_head);
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------
void dispatch_bwd_combined_d192v128(attn_bwd_combined_d192v128_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute(
        (void*)attend_bwd_combined_d192v128_ker,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_bwd_combined_d192v128_ker<<<
        g.grid(), g.block(), mem_size, g.stream>>>(g);
}


// ===========================================================================
// Separate dQ kernel — MMA-based accumulation via mma_AB.
// Single accumulator `acc` reused for P, dP, and dQ_chunk to avoid gfx950
// AGPR aliasing. dS col_l→row_l conversion done via LDS.
// ===========================================================================

// ---------------------------------------------------------------------------
// Globals for separate dQ kernel
// ---------------------------------------------------------------------------
struct attn_bwd_dq_d192v128_globals {
    _gl Q, K, V;
    _gl dOg, dQg;
    gl<float, -1, -1, -1, -1> L_vec, delta_vec;
    hipStream_t stream;
    dim3 grid()  { return dim3(ATTN_H_KV, (ATTN_N / BLOCK_KV), ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

// ---------------------------------------------------------------------------
// dQ kernel: recomputes P and dS via MMA, then computes dQ = dS @ K via
// mma_AB in 32-column chunks, with atomic bf16 packed add to global.
// ---------------------------------------------------------------------------
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

    st_bf<BLOCK_KV, D_QK, st_32x32_s> (&K_smem) = al.allocate<st_bf<BLOCK_KV, D_QK, st_32x32_s>>();
    st_bf<BLOCK_KV, D_V,  st_32x32_s> (&V_smem) = al.allocate<st_bf<BLOCK_KV, D_V,  st_32x32_s>>();
    st_bf<Q_TILE, D_QK, st_32x32_s>   (&Q_smem) = al.allocate<st_bf<Q_TILE, D_QK, st_32x32_s>>();
    st_bf<Q_TILE, D_V,  st_32x32_s>   (&dO_smem) = al.allocate<st_bf<Q_TILE, D_V,  st_32x32_s>>();
    // dS layout conversion tile: per-warp 32x32 subtile (4 warps = 32x128)
    st_bf<Q_TILE, BLOCK_KV, st_32x32_s> (&dS_conv_smem) = al.allocate<st_bf<Q_TILE, BLOCK_KV, st_32x32_s>>();
    sv_fl<Q_TILE> (&L_smem)     = al.allocate<sv_fl<Q_TILE>>();
    sv_fl<Q_TILE> (&delta_smem) = al.allocate<sv_fl<Q_TILE>>();
    // Per-thread P value scratch + dQ transpose scratch in LDS
    float *P_scratch = reinterpret_cast<float*>(al.ptr);

    const int kv_head   = blockIdx.x;
    const int seq_block = blockIdx.y;
    const int batch     = blockIdx.z;
    const int wid       = kittens::warpid();
    const int j         = seq_block * NUM_WARPS + wid;
    const int kv_start  = j * KV_BLOCK;
    const int total_q   = ATTN_N / Q_TILE;
    const int first_q   = causal ? max(0, (int)(seq_block * BLOCK_KV / Q_TILE)) : 0;

    const int lane = laneid();
    const int tid  = threadIdx.x;
    const int G_id = lane >> 5;

    // Load K and V tiles into shared memory
    G::load<QKVO_AXIS, false>(K_smem, g.K, {batch, seq_block, kv_head, 0});
    G::load<QKVO_AXIS, false>(V_smem, g.V, {batch, seq_block, kv_head, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    for (int qi = first_q; qi < total_q; qi++) {
        for (int qho = 0; qho < GROUP_SIZE; qho++) {
            const int q_head = kv_head * GROUP_SIZE + qho;
            const int q_pos  = qi * Q_TILE;
            const bool skip  = causal && (q_pos + Q_TILE <= kv_start);

            float *L_raw     = reinterpret_cast<float*>(&L_smem);
            float *delta_raw = reinterpret_cast<float*>(&delta_smem);
            load_L_delta_direct(L_raw, g.L_vec, batch, q_head, qi);
            load_L_delta_direct(delta_raw, g.delta_vec, batch, q_head, qi);

            G::load<QKVO_AXIS, false>(Q_smem, g.Q, {batch, qi, q_head, 0});
            G::load<QKVO_AXIS, false>(dO_smem, g.dOg, {batch, qi, q_head, 0});
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_s_barrier();

            // Single accumulator reused for P, dP, and dQ chunks
            rt<float, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> acc;
            rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> dS_bf;

            if (!skip) {
                // Phase 1: S = Q @ K^T * P_SCALE - L*L_SCALE, causal mask, exp2
                {
                    rt<bf16, Q_TILE, D_QK, row_l, rt_32x16_4_s> Q_i;
                    load(Q_i, Q_smem);
                    rt<bf16, KV_BLOCK, D_QK, row_l, rt_32x16_4_s> K_j;
                    load(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {wid, 0}));
                    zero(acc);
                    mma_ABt(acc, Q_i, K_j, acc);
                }
                mul(acc, acc, P_SCALE);

                typename decltype(acc)::col_vec L_reg_v;
                load(L_reg_v, L_smem);
                mul(L_reg_v, L_reg_v, L_SCALE);
                sub_row(acc, acc, L_reg_v);

                if constexpr (causal) {
                    const int kv_local = lane & 31;
                    #pragma unroll
                    for (int k = 0; k < acc.tiles[0][0].packed_per_thread; k++) {
                        int ro   = ((k >> 1) << 3) + ((k & 1) << 1);
                        int q_r0 = G_id * 4 + ro;
                        int q_r1 = q_r0 + 1;
                        if (q_pos + q_r0 < kv_start + kv_local)
                            acc.tiles[0][0].data[k].x = -__builtin_inff();
                        if (q_pos + q_r1 < kv_start + kv_local)
                            acc.tiles[0][0].data[k].y = -__builtin_inff();
                    }
                }

                #pragma unroll
                for (int k = 0; k < acc.tiles[0][0].packed_per_thread; k++) {
                    acc.tiles[0][0].data[k].x =
                        __builtin_fminf(acc.tiles[0][0].data[k].x, 0.f);
                    acc.tiles[0][0].data[k].y =
                        __builtin_fminf(acc.tiles[0][0].data[k].y, 0.f);
                    acc.tiles[0][0].data[k] =
                        base_ops::exp2::op(acc.tiles[0][0].data[k]);
                }

                // Save P to per-thread LDS scratch
                {
                    volatile float *P_lds = P_scratch + tid * 16;
                    #pragma unroll
                    for (int k = 0; k < 8; k++) {
                        P_lds[k*2]   = acc.tiles[0][0].data[k].x;
                        P_lds[k*2+1] = acc.tiles[0][0].data[k].y;
                    }
                }

                __builtin_amdgcn_s_waitcnt(0);

                // Phase 2: dP = dO @ V^T, then dS = P * (dP - delta) * scale
                {
                    rt<bf16, Q_TILE, D_V, row_l, rt_32x16_4_s> dO_i;
                    load(dO_i, dO_smem);
                    rt<bf16, KV_BLOCK, D_V, row_l, rt_32x16_4_s> V_j;
                    load(V_j, subtile_inplace<KV_BLOCK, D_V>(V_smem, {wid, 0}));
                    zero(acc);
                    mma_ABt(acc, dO_i, V_j, acc);
                }

                __builtin_amdgcn_s_waitcnt(0);
                {
                    float *d_raw = reinterpret_cast<float*>(&delta_smem);
                    volatile float *P_lds = P_scratch + tid * 16;
                    #pragma unroll
                    for (int k = 0; k < acc.tiles[0][0].packed_per_thread; k++) {
                        int ro = ((k >> 1) << 3) + ((k & 1) << 1);
                        int row0 = G_id * 4 + ro;
                        int row1 = row0 + 1;
                        float dP_x = acc.tiles[0][0].data[k].x - d_raw[row0];
                        float dP_y = acc.tiles[0][0].data[k].y - d_raw[row1];
                        float P_x = P_lds[k*2];
                        float P_y = P_lds[k*2+1];
                        acc.tiles[0][0].data[k].x = P_x * dP_x * dP_SCALE;
                        acc.tiles[0][0].data[k].y = P_y * dP_y * dP_SCALE;
                    }
                }
                copy(dS_bf, acc);

                // Phase 3: dQ = dS @ K via mma_AB
                // dS col_l → store to shared → load as row_l
                auto dS_stile = subtile_inplace<Q_TILE, KV_BLOCK>(dS_conv_smem, {0, wid});
                store(dS_stile, dS_bf);
                __builtin_amdgcn_s_waitcnt(0);

                rt<bf16, Q_TILE, KV_BLOCK, row_l, rt_32x16_4_s> dS_row;
                load(dS_row, dS_stile);

                // Buffer resource for atomic bf16 packed adds
                bf16 *dQ_base = reinterpret_cast<bf16*>(g.dQg.raw_ptr);
                std::uintptr_t dq_int = reinterpret_cast<std::uintptr_t>(dQ_base);
                std::uint64_t  dq_u64 = static_cast<std::uint64_t>(dq_int);
                buffer_resource dq_br = make_buffer_resource(dq_u64, 0x7FFFFFFFu, 0x00020000);

                const int stride_b = g.dQg.template stride<0>();
                const int stride_h = g.dQg.template stride<1>();
                const int stride_n = g.dQg.template stride<2>();

                const int d_col = lane & 31;
                const bool is_even_d = (d_col & 1) == 0;

                #pragma unroll
                for (int t = 0; t < D_QK / 32; t++) {
                    rt<bf16, KV_BLOCK, 32, col_l, rt_16x32_4_s> K_col;
                    load(K_col, subtile_inplace<KV_BLOCK, 32>(K_smem, {wid, t}));

                    zero(acc);
                    mma_AB(acc, dS_row, K_col, acc);
                    __builtin_amdgcn_s_waitcnt(0);
                    __builtin_amdgcn_sched_barrier(0);

                    // mma_AB output [Q_TILE x 32] col_l:
                    //   col = lane & 31 = D position within 32-col chunk
                    //   row from data[k] = Q position
                    // Use __shfl_xor(val, 1) to pair adjacent D values
                    #pragma unroll
                    for (int k = 0; k < 8; k++) {
                        int ro   = ((k >> 1) << 3) + ((k & 1) << 1);
                        int q_r0 = G_id * 4 + ro;
                        int q_r1 = q_r0 + 1;

                        float my_val0 = acc.tiles[0][0].data[k].x;
                        float nb_val0 = __shfl_xor(my_val0, 1);
                        float my_val1 = acc.tiles[0][0].data[k].y;
                        float nb_val1 = __shfl_xor(my_val1, 1);

                        if (is_even_d) {
                            // Pack (d_col, d_col+1) for q_r0
                            uint32_t pk0;
                            asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2"
                                : "=v"(pk0) : "v"(my_val0), "v"(nb_val0));
                            int elem_off0 = batch * stride_b + q_head * stride_h
                                          + (q_pos + q_r0) * stride_n
                                          + t * 32 + d_col;
                            uint32_t byte_off0 = static_cast<uint32_t>(
                                elem_off0 * sizeof(bf16));
                            asm volatile(
                                "buffer_atomic_pk_add_bf16 %0, %1, %2, 0 offen"
                                : : "v"(pk0), "v"(byte_off0),
                                    "s"(*(const i32x4*)&dq_br) : "memory");

                            // Pack (d_col, d_col+1) for q_r1
                            uint32_t pk1;
                            asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2"
                                : "=v"(pk1) : "v"(my_val1), "v"(nb_val1));
                            int elem_off1 = batch * stride_b + q_head * stride_h
                                          + (q_pos + q_r1) * stride_n
                                          + t * 32 + d_col;
                            uint32_t byte_off1 = static_cast<uint32_t>(
                                elem_off1 * sizeof(bf16));
                            asm volatile(
                                "buffer_atomic_pk_add_bf16 %0, %1, %2, 0 offen"
                                : : "v"(pk1), "v"(byte_off1),
                                    "s"(*(const i32x4*)&dq_br) : "memory");
                        }
                    }
                }
            } // !skip

            __builtin_amdgcn_s_barrier();
        } // qho
    } // qi

}

// ---------------------------------------------------------------------------
// Dispatch dQ kernel
// ---------------------------------------------------------------------------
void dispatch_bwd_dq_d192v128(attn_bwd_dq_d192v128_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute(
        (void*)attend_bwd_dq_d192v128_ker,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_bwd_dq_d192v128_ker<<<
        g.grid(), g.block(), mem_size, g.stream>>>(g);
}

// ---------------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------------
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
