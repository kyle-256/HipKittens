// MLA Backward Kernel (D_QK=192, D_V=128) — causal, dK/dV/dQ
// BNHD layout (QKVO_AXIS=1), KV_BLOCK=32, Q_TILE=32, BLOCK_KV=128, NUM_WARPS=4
//
// gfx950 fixes applied:
//   1. rt_32x16_4_s (stride 4) for all mma_ABt row_l inputs
//   2. L/delta loaded directly into rv from global (sv_fl<32> load is broken on 64-lane warps)
//   3. sub_row on col_l tiles with correctly-loaded align rv
//   4. store_col_l_direct for dV; dK: transpose → bf16 → store<1> (agentA-style)
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
        asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(pk) : "v"(f1), "v"(f0));

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
// Kernel — Optimized: hoisted K/V reg loads, double-buffered Q/dO, sched barriers
// ---------------------------------------------------------------------------
__launch_bounds__(NUM_THREADS, 1)
__global__ void attend_bwd_combined_d192v128_ker(
    const attn_bwd_combined_d192v128_globals g)
{
    constexpr int QKVO_AXIS = 1; // BNHD layout
    constexpr float P_SCALE  = 0.07216878365f * 1.44269504089f;
    constexpr float L_SCALE  = 1.44269504089f;
    constexpr float dP_SCALE = 0.07216878365f;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // K/V in shared (loaded once)
    st_bf<BLOCK_KV, D_QK, st_32x32_s> (&K_smem)     = al.allocate<st_bf<BLOCK_KV, D_QK, st_32x32_s>>();
    st_bf<BLOCK_KV, D_V,  st_32x32_s> (&V_smem)     = al.allocate<st_bf<BLOCK_KV, D_V,  st_32x32_s>>();
    // Double-buffered Q/dO
    st_bf<Q_TILE, D_QK, st_32x32_s>   (&Q_smem)[2]  = al.allocate<st_bf<Q_TILE, D_QK, st_32x32_s>, 2>();
    st_bf<Q_TILE, D_V,  st_32x32_s>   (&dO_smem)[2] = al.allocate<st_bf<Q_TILE, D_V,  st_32x32_s>, 2>();
    sv_fl<Q_TILE> (&L_smem)[2]     = al.allocate<sv_fl<Q_TILE>, 2>();
    sv_fl<Q_TILE> (&delta_smem)[2] = al.allocate<sv_fl<Q_TILE>, 2>();

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

    // Load K and V into shared memory
    G::load<QKVO_AXIS, false>(K_smem, g.K, {batch, seq_block, kv_head, 0});
    G::load<QKVO_AXIS, false>(V_smem, g.V, {batch, seq_block, kv_head, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    // Hoist K_j and V_j into registers (used for ALL Q iterations)
    rt<bf16, KV_BLOCK, D_QK, row_l, rt_32x16_4_s> K_j;
    load(K_j, subtile_inplace<KV_BLOCK, D_QK>(K_smem, {wid, 0}));
    rt<bf16, KV_BLOCK, D_V, row_l, rt_32x16_4_s> V_j;
    load(V_j, subtile_inplace<KV_BLOCK, D_V>(V_smem, {wid, 0}));

    // Flatten (qi, qho) into single iteration space
    const int total_iters = (total_q - first_q) * GROUP_SIZE;

    if (total_iters > 0) {
        // Prefetch first Q/dO into buffer 0
        {
            const int qi0 = first_q;
            const int qh0 = kv_head * GROUP_SIZE;
            G::load<QKVO_AXIS, false>(Q_smem[0], g.Q, {batch, qi0, qh0, 0});
            G::load<QKVO_AXIS, false>(dO_smem[0], g.dOg, {batch, qi0, qh0, 0});
            load_L_delta_direct(reinterpret_cast<float*>(&L_smem[0]),
                                g.L_vec, batch, qh0, qi0);
            load_L_delta_direct(reinterpret_cast<float*>(&delta_smem[0]),
                                g.delta_vec, batch, qh0, qi0);
        }
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();

        #pragma unroll 1
        for (int iter = 0; iter < total_iters; iter++) {
            const int cur = iter & 1;
            const int nxt = 1 - cur;
            const int qi   = first_q + (iter / GROUP_SIZE);
            const int qho  = iter % GROUP_SIZE;
            const int q_head = kv_head * GROUP_SIZE + qho;
            const int q_pos  = qi * Q_TILE;
            const bool skip  = causal && (q_pos + Q_TILE <= kv_start);

            // Prefetch next Q/dO (overlap with compute)
            if (iter + 1 < total_iters) {
                const int nqi  = first_q + ((iter + 1) / GROUP_SIZE);
                const int nqho = (iter + 1) % GROUP_SIZE;
                const int nqh  = kv_head * GROUP_SIZE + nqho;
                G::load<QKVO_AXIS, false>(Q_smem[nxt], g.Q, {batch, nqi, nqh, 0});
                G::load<QKVO_AXIS, false>(dO_smem[nxt], g.dOg, {batch, nqi, nqh, 0});
                load_L_delta_direct(reinterpret_cast<float*>(&L_smem[nxt]),
                                    g.L_vec, batch, nqh, nqi);
                load_L_delta_direct(reinterpret_cast<float*>(&delta_smem[nxt]),
                                    g.delta_vec, batch, nqh, nqi);
            }

            if (!skip) {
                // Phase 1: S = Q @ K^T
                rt<bf16, Q_TILE, D_QK, row_l, rt_32x16_4_s> Q_i;
                load(Q_i, Q_smem[cur]);

                rt<float, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> S_ij;
                zero(S_ij);
                mma_ABt(S_ij, Q_i, K_j, S_ij);
                sched_barrier_pairs<12, 2, 0>();
                __builtin_amdgcn_sched_barrier(0);

                mul(S_ij, S_ij, P_SCALE);
                typename decltype(S_ij)::col_vec L_reg;
                load(L_reg, L_smem[cur]);
                mul(L_reg, L_reg, L_SCALE);
                sub_row(S_ij, S_ij, L_reg);

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
                load(dO_i, dO_smem[cur]);

                rt<float, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> dP_ij;
                zero(dP_ij);
                mma_ABt(dP_ij, dO_i, V_j, dP_ij);
                sched_barrier_pairs<8, 2, 1>();
                __builtin_amdgcn_sched_barrier(0);

                // dS = P * (dP - delta) * dP_SCALE
                {
                    const int G = lane >> 5;
                    float *d_raw = reinterpret_cast<float*>(&delta_smem[cur]);
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
                load(dO_col, dO_smem[cur]);

                rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> P_bf;
                copy(P_bf, S_ij);
                auto &Pm = *reinterpret_cast<
                    rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_16x32_4_s>*>(&P_bf);
                mma_AtB(dV_acc, dO_col, Pm, dV_acc);
                sched_barrier_pairs<4, 2, 2>();
                __builtin_amdgcn_sched_barrier(0);

                // Phase 4: dK += Q^T @ dS
                mul(dP_ij, dP_ij, dP_SCALE);

                rt<bf16, Q_TILE, D_QK, col_l, rt_16x32_4_s> Q_col;
                load(Q_col, Q_smem[cur]);
                // dK Phase 4: st_32x32_s shared→col_l path is wrong for D=32..63 on upper 16 Q rows;
                // narrow global→register col_l load directly into Q_col.tiles[1][1] (Q[16:32,32:64]).
                using Q_d32_patch_rt = rt<bf16, 16, 32, col_l, rt_16x32_4_s>;
                Q_d32_patch_rt &Q_d32_patch = reinterpret_cast<Q_d32_patch_rt &>(Q_col.tiles[1][1]);
                kittens::load<1>(Q_d32_patch, g.Q,
                    coord<Q_d32_patch_rt>{batch, qi * 2 + 1, q_head, 1});
                __builtin_amdgcn_s_waitcnt(0);

                rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> dS_bf;
                copy(dS_bf, dP_ij);
                auto &dSm = *reinterpret_cast<
                    rt<bf16, Q_TILE, KV_BLOCK, col_l, rt_16x32_4_s>*>(&dS_bf);
                mma_AtB(dK_acc, Q_col, dSm, dK_acc);
                sched_barrier_pairs<6, 2, 3>();
                __builtin_amdgcn_sched_barrier(0);
            }

            // Wait for prefetch and synchronize
            if (iter + 1 < total_iters) {
                __builtin_amdgcn_s_waitcnt(0);
            }
            __builtin_amdgcn_s_barrier();
        }
    }

    // Epilogue: dV keeps col_l direct store; dK uses agentA-style transpose → bf16 → store<1>
    store_col_l_direct<QKVO_AXIS, D_V>(g.dVg, dV_acc, batch, j, kv_head);

    rt<float, KV_BLOCK, D_QK, row_l, rt_32x32_s> dK_row;
    transpose(dK_row, dK_acc);
    rt<bf16, KV_BLOCK, D_QK, row_l, rt_32x32_s> dK_bf;
    copy(dK_bf, dK_row);
    __builtin_amdgcn_s_waitcnt(0);
    store<1>(g.dKg, dK_bf, {batch, j, kv_head, 0});
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
// Separate dQ kernel — SCALAR (VALU) accumulation, NO mma_AtB for dQ.
// Avoids gfx950 AGPR aliasing bug by computing dQ[q,d] = Σ_kv dS[q,kv]*K[kv,d]
// using scalar multiply-accumulate with K read from global memory.
// ===========================================================================

// ---------------------------------------------------------------------------
// Globals for separate dQ kernel
// ---------------------------------------------------------------------------
struct attn_bwd_dq_d192v128_globals {
    _gl Q, K, V;
    _gl dOg, dQg;
    gl<float, -1, -1, -1, -1> L_vec, delta_vec;
    hipStream_t stream;
    // Same grid as main kernel: iterate over KV blocks
    dim3 grid()  { return dim3(ATTN_H_KV, (ATTN_N / BLOCK_KV), ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

// ---------------------------------------------------------------------------
// dQ kernel: recomputes P and dS via MMA (mma_ABt only), then accumulates
// dQ using scalar VALU ops (no mma_AtB).
// K is read from global memory (not shared) to avoid swizzle complexity.
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
    sv_fl<Q_TILE> (&L_smem)     = al.allocate<sv_fl<Q_TILE>>();
    sv_fl<Q_TILE> (&delta_smem) = al.allocate<sv_fl<Q_TILE>>();
    // Per-thread P value scratch in LDS (16 floats * 256 threads = 16384 bytes)
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
    const int tid  = threadIdx.x;  // global thread id within block

    // col_l mapping for dS / S tiles (rt<float, 32, 32, col_l, rt_32x32_s>):
    //   kv_pos within sub-tile = lane & 31  (column)
    //   q offsets come from k index: G*4 + ((k>>1)<<3) + ((k&1)<<1) for .x, +1 for .y
    //   G = lane >> 5
    const int kv_local = lane & 31;                // position within this warp's KV_BLOCK
    const int G_id     = lane >> 5;

    // Set up buffer resource for dQ atomic adds
    bf16 *dQ_base = reinterpret_cast<bf16*>(g.dQg.raw_ptr);
    std::uintptr_t dq_int = reinterpret_cast<std::uintptr_t>(dQ_base);
    std::uint64_t  dq_u64 = static_cast<std::uint64_t>(dq_int);
    buffer_resource dq_br = make_buffer_resource(dq_u64, 0x7FFFFFFFu, 0x00020000);

    const int stride_b = g.dQg.template stride<0>();
    const int stride_h = g.dQg.template stride<1>();
    const int stride_n = g.dQg.template stride<2>();

    // Pre-load K row for this thread from GLOBAL memory into registers
    // K is [B, N, H_KV, D_QK] in BNHD layout
    // This thread's KV seq position = kv_start + kv_local
    const bf16 *K_ptr = reinterpret_cast<const bf16*>(g.K.raw_ptr);
    const int K_stride_b = g.K.template stride<0>();  // N * H_KV * D_QK
    const int K_stride_n = g.K.template stride<1>();  // H_KV * D_QK (axis=1 = seq)
    const int K_stride_h = g.K.template stride<2>();  // D_QK

    const int kv_seq_pos = kv_start + kv_local;
    const int K_base_off = batch * K_stride_b + kv_seq_pos * K_stride_n
                         + kv_head * K_stride_h;

    bf16 K_reg[D_QK];
    for (int d = 0; d < D_QK; d++) {
        K_reg[d] = K_ptr[K_base_off + d];
    }

    // Load K and V tiles into shared memory (needed for MMA recomputation of P and dP)
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

            if (!skip) {
                // Use a SINGLE MMA accumulator for both S and dP to avoid
                // gfx950 AGPR aliasing (compiler may overlap two accumulators)
                rt<float, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> acc;

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

                // Save P = exp2(S) into LDS scratch to avoid VGPR clobbering
                // during the second MMA's tile loads. 16 floats per thread.
                {
                    volatile float *P_lds = P_scratch + tid * 16;
                    #pragma unroll
                    for (int k = 0; k < 8; k++) {
                        P_lds[k*2]   = acc.tiles[0][0].data[k].x;
                        P_lds[k*2+1] = acc.tiles[0][0].data[k].y;
                    }
                }

                __builtin_amdgcn_s_waitcnt(0); // ensure P LDS stores complete
                // Phase 2: dP = dO @ V^T using the SAME accumulator
                {
                    rt<bf16, Q_TILE, D_V, row_l, rt_32x16_4_s> dO_i;
                    load(dO_i, dO_smem);
                    rt<bf16, KV_BLOCK, D_V, row_l, rt_32x16_4_s> V_j;
                    load(V_j, subtile_inplace<KV_BLOCK, D_V>(V_smem, {wid, 0}));
                    zero(acc);
                    mma_ABt(acc, dO_i, V_j, acc);
                }
                // acc now holds dP

                // dS = P * (dP - delta) * softmax_scale
                // Read P back from LDS scratch
                __builtin_amdgcn_s_waitcnt(0); // ensure dP MMA and P LDS loads don't conflict
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
                        // dS = P * (dP - delta) * dP_SCALE
                        acc.tiles[0][0].data[k].x = P_x * dP_x * dP_SCALE;
                        acc.tiles[0][0].data[k].y = P_y * dP_y * dP_SCALE;
                    }
                }
                // acc now holds dS (scaled)

                // Phase 3: SCALAR dQ accumulation
                // dQ[q,d] += dS[q,kv] * K[kv,d]
                // Each thread has 16 dS values (8 packed pairs) in acc.
                // For each dS element, multiply by K_reg[d] and atomic-add to dQ.
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    int ro    = ((k >> 1) << 3) + ((k & 1) << 1);
                    int q_r0  = G_id * 4 + ro;
                    int q_r1  = q_r0 + 1;

                    float dS_val0 = acc.tiles[0][0].data[k].x;
                    float dS_val1 = acc.tiles[0][0].data[k].y;

                    // Base offset for dQ[batch, q_head, q_pos + q_r0/q_r1, :]
                    int elem_off0 = batch * stride_b + q_head * stride_h
                                  + (q_pos + q_r0) * stride_n;
                    int elem_off1 = batch * stride_b + q_head * stride_h
                                  + (q_pos + q_r1) * stride_n;

                    // Accumulate over D_QK dimension in pairs
                    for (int dp = 0; dp < D_QK; dp += 2) {
                        float k0_f = __bfloat162float(K_reg[dp]);
                        float k1_f = __bfloat162float(K_reg[dp + 1]);

                        // dQ[q_r0, dp:dp+2] += dS_val0 * K[kv, dp:dp+2]
                        {
                            float f0 = dS_val0 * k0_f;
                            float f1 = dS_val0 * k1_f;
                            uint32_t pk;
                            asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2"
                                : "=v"(pk) : "v"(f0), "v"(f1));
                            uint32_t byte_off = static_cast<uint32_t>(
                                (elem_off0 + dp) * sizeof(bf16));
                            asm volatile(
                                "buffer_atomic_pk_add_bf16 %0, %1, %2, 0 offen"
                                : : "v"(pk), "v"(byte_off),
                                    "s"(*(const i32x4*)&dq_br) : "memory");
                        }

                        // dQ[q_r1, dp:dp+2] += dS_val1 * K[kv, dp:dp+2]
                        {
                            float f0 = dS_val1 * k0_f;
                            float f1 = dS_val1 * k1_f;
                            uint32_t pk;
                            asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2"
                                : "=v"(pk) : "v"(f0), "v"(f1));
                            uint32_t byte_off = static_cast<uint32_t>(
                                (elem_off1 + dp) * sizeof(bf16));
                            asm volatile(
                                "buffer_atomic_pk_add_bf16 %0, %1, %2, 0 offen"
                                : : "v"(pk), "v"(byte_off),
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
