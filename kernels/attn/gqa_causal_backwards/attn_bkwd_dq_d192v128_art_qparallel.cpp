// Q-Parallel dQ Backward Kernel (D_QK=192, D_V=128)
// No atomics. Each block owns 64 Q rows (2 warps * 32 Q rows each).
// Each warp accumulates its dQ [32 x 192] in registers across all KV blocks,
// then stores directly to global (no atomics, no inter-warp reduction).
//
// Uses 32x32x16 bf16 MFMA (known to work) instead of 16x16x32 (which has bugs
// on gfx950 for our col_l 32x16 B operand).
//
// Structure:
//   1. Per block: load 64 Q rows + 64 dO rows + L + delta (per-warp subtile).
//   2. For each KV block (N/KV_BLOCK iterations, KV_BLOCK=32):
//        - Cooperatively load K_j [32 x 192] and V_j [32 x 128] to shared.
//        - Compute S_ij = Q_i @ K_j^T    [32 x 32] col_l
//        - Apply scale, subtract L, causal mask, exp2 -> P_ij [32 x 32] col_l
//        - Compute dP_ij = dO_i @ V_j^T  [32 x 32] col_l
//        - dS_ij = P * (dP - delta) * dP_SCALE (bf16 col_l rt_32x32)
//        - Transpose dS col_l -> row_l via shared memory.
//        - dQ_i += dS_row @ K_j_col_chunk for each of 6 D-column chunks.
//   3. Epilogue: store dQ to global.

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

constexpr int D_QK      = 192;
constexpr int D_V       = 128;
constexpr int KV_BLOCK  = 32;     // KV rows per iteration
constexpr int Q_TILE    = 32;     // Q rows per warp
constexpr int STEP_Q    = 128;    // Q rows per block (4 warps * 32)
constexpr bool causal   = true;

#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;
using G   = kittens::group<NUM_WARPS>;
using _gl = gl<bf16, -1, -1, -1, -1>;

// ---------------------------------------------------------------------------
// Direct L/delta loader (sv_fl<32> is broken on 64-lane warps).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void load_L_delta_direct(
    float *smem_dst,
    const gl<float, -1, -1, -1, -1> &src,
    int batch, int head, int q_start)
{
    const int lane = laneid();
    const float *src_ptr = (const float*)src.raw_ptr;
    const int stride_b = src.template stride<0>();
    const int stride_h = src.template stride<1>();
    const int base_idx = batch * stride_b + head * stride_h + q_start;
    if (lane < Q_TILE) {
        smem_dst[lane] = src_ptr[base_idx + lane];
    }
}

struct attn_bwd_dq_qparallel_globals {
    _gl Q, K, V;
    _gl dOg, dQg;
    gl<float, -1, -1, -1, -1> L_vec, delta_vec;
    hipStream_t stream;
    dim3 grid()  { return dim3(ATTN_H, ATTN_N / STEP_Q, ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__launch_bounds__(NUM_THREADS, 1)
__global__ void attend_bwd_dq_qparallel_ker(const attn_bwd_dq_qparallel_globals g)
{
    constexpr float L_SCALE_FACTOR = 1.44269504089f;
    constexpr float P_SCALE_FACTOR = 0.07216878365f * 1.44269504089f;
    constexpr float dP_SCALE_FACTOR = 0.07216878365f;

    const int q_head   = blockIdx.x;
    // R35-A: reverse-Q ordering — process highest q_block first.
    // +1.8% lift on dQ kernel (~13.63 → ~13.39 ms) by reducing tail latency.
    // Numerics unchanged (algebra invariant). Grid axes already optimal — only this swap helps.
    const int q_block  = (ATTN_N / STEP_Q - 1) - blockIdx.y;
    const int batch    = blockIdx.z;
    const int kv_head  = q_head / GROUP_SIZE;
    const int warpid   = kittens::warpid();
    const int q_pos    = q_block * STEP_Q + warpid * Q_TILE;
    const int block_q_end = q_block * STEP_Q + STEP_Q;

    // =====================================================================
    // Shared memory allocation
    // =====================================================================
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    st_bf<KV_BLOCK, D_QK, st_32x32_s> (&K_smem) =
        al.allocate<st_bf<KV_BLOCK, D_QK, st_32x32_s>>();
    st_bf<KV_BLOCK, D_V, st_32x32_s> (&V_smem) =
        al.allocate<st_bf<KV_BLOCK, D_V, st_32x32_s>>();
    st_bf<STEP_Q, D_QK, st_32x32_s> (&Q_smem) =
        al.allocate<st_bf<STEP_Q, D_QK, st_32x32_s>>();
    st_bf<STEP_Q, D_V, st_32x32_s> (&dO_smem) =
        al.allocate<st_bf<STEP_Q, D_V, st_32x32_s>>();
    // attn_smem for dS layout conversion, per-warp sub-slot [32x32]
    // R94G: per-row-padded (pad4 = 8B/row stride bump = 2 LDS banks).
    // Re-tests R49-A's lever class at the post-R87/R89 baseline where dQ
    // SQ_WAIT_INST_LDS=31.67% (R94 PMC). Production stride 64B/row (32 banks);
    // pad4 stride 72B/row (= 9*8B; ds_read_b64 / ds_write_b16 alignment-safe).
    // R49-A had int32 stride bug at N=16384; R94G keeps post-R89 int64 epilogue.
    // +0.30% combined wall (379.03 → 377.90 ms); cos vs prod 1.000001.
    st_bf<STEP_Q, KV_BLOCK, st_32x32_pad4_s> (&attn_smem) =
        al.allocate<st_bf<STEP_Q, KV_BLOCK, st_32x32_pad4_s>>();
    // Per-warp L, delta (Q_TILE floats per warp)
    float *L_smem     = reinterpret_cast<float*>(al.ptr); al.ptr += NUM_WARPS * Q_TILE;
    float *delta_smem = reinterpret_cast<float*>(al.ptr); al.ptr += NUM_WARPS * Q_TILE;

    // =====================================================================
    // Register tiles
    // =====================================================================
    rt<bf16, Q_TILE, D_QK, row_l, rt_32x16_4_s> Q_i;
    rt<bf16, Q_TILE, D_V,  row_l, rt_32x16_4_s> dO_i;

    rt<float, Q_TILE, KV_BLOCK, col_l, rt_32x32_s> acc_fp;

    // dQ accumulators: 3 chunks of [32 x 64] col_l (total [32 x 192])
    rt<float, Q_TILE, 64, col_l, rt_32x32_s> dQa, dQb, dQc;
    zero(dQa); zero(dQb); zero(dQc);

    // K_col declared once; reused across kj iterations to keep the MFMA
    // pipeline "primed" (avoids 1 extra throwaway mma per kj iteration).
    rt<bf16, KV_BLOCK, 64, col_l, rt_16x32_4_s> K_col;

    // =====================================================================
    // Prologue: Load Q_i, dO_i, L, delta to shared
    // =====================================================================
    G::load<1, false>(Q_smem,  g.Q,   {batch, q_block, q_head, 0});
    G::load<1, false>(dO_smem, g.dOg, {batch, q_block, q_head, 0});

    if (warpid < NUM_WARPS) {
        const int qw_start = q_block * STEP_Q + warpid * Q_TILE;
        load_L_delta_direct(L_smem     + warpid * Q_TILE,
                            g.L_vec,     batch, q_head, qw_start);
        load_L_delta_direct(delta_smem + warpid * Q_TILE,
                            g.delta_vec, batch, q_head, qw_start);
    }

    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    {
        auto Q_sub  = subtile_inplace<Q_TILE, D_QK>(Q_smem,  {warpid, 0});
        auto dO_sub = subtile_inplace<Q_TILE, D_V>(dO_smem, {warpid, 0});
        load(Q_i,  Q_sub);
        load(dO_i, dO_sub);
    }

    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");

    // =====================================================================
    // Pre-loop prime: initialize MFMA HW state for dQa/dQb/dQc destinations
    // so the per-iter "throwaway" prime can be omitted. Inputs are zero so
    // accumulators remain zero even if the "first mma drops 8 rows" bug fires.
    // =====================================================================
    {
        rt<bf16, Q_TILE, KV_BLOCK, row_l, rt_32x16_4_s> dS_row_dummy;
        zero(dS_row_dummy);
        zero(K_col);
        mma_AB(dQa, dS_row_dummy, K_col, dQa);
        mma_AB(dQb, dS_row_dummy, K_col, dQb);
        mma_AB(dQc, dS_row_dummy, K_col, dQc);
        asm volatile("" :: "v"(K_col.tiles[0][0].data[0]), "v"(dQa.tiles[0][0].data[0]) : "memory");
    }

    // =====================================================================
    // Pre-load L from LDS to registers once (does not change per iteration).
    // (delta pre-load breaks due to register pressure -> keep delta in LDS.)
    // =====================================================================
    float L_reg[16];
    float delta_reg[16];
    {
        const int lane_pre = laneid();
        const int G_id_pre = (lane_pre >> 5) & 1;
        __attribute__((address_space(3))) float *my_L =
            (__attribute__((address_space(3))) float*)(L_smem + warpid * Q_TILE);
        __attribute__((address_space(3))) float *my_delta =
            (__attribute__((address_space(3))) float*)(delta_smem + warpid * Q_TILE);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int ro = ((k >> 1) << 3) + ((k & 1) << 1);
            int r0 = G_id_pre * 4 + ro, r1 = r0 + 1;
            L_reg[k * 2]         = my_L[r0]     * L_SCALE_FACTOR;
            L_reg[k * 2 + 1]     = my_L[r1]     * L_SCALE_FACTOR;
            delta_reg[k * 2]     = my_delta[r0];
            delta_reg[k * 2 + 1] = my_delta[r1];
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");


    // =====================================================================
    // Main loop over KV blocks
    // =====================================================================
    const int total_kv_blocks = ATTN_N / KV_BLOCK;
    const int last_kv_block   = causal
        ? min(total_kv_blocks, (block_q_end + KV_BLOCK - 1) / KV_BLOCK)
        : total_kv_blocks;

    // R88-pipeline: prefetch K_smem/V_smem for kj=0 BEFORE loop entry.
    // Per-iter, the next-iter's K/V load is issued at the END of the body
    // (after Phase 5's last MFMA), so its buffer_load_dwordx4 issue overlaps
    // with the dQc MFMA shadow + with the loop-carried branch+barrier delay.
    // Note: still single-buffered K_smem/V_smem; the prev iter's K_col reads
    // have all retired before the next G::load issues (Phase 5 last lgkmcnt(0)).
    if (last_kv_block > 0) {
        G::load<1, false>(K_smem, g.K, {batch, 0, kv_head, 0});
        G::load<1, false>(V_smem, g.V, {batch, 0, kv_head, 0});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
    }

    for (int kj = 0; kj < last_kv_block; kj++) {
        const int kv_start = kj * KV_BLOCK;

        // =================================================================
        // Phase 1: S_ij = Q_i @ K^T [32 x 32] col_l
        // Issue V_j load FIRST so its ds_reads keep streaming during the K_j
        // load and the Phase 1 MMA. We wait for K_j only (V_j outstanding); a
        // final lgkmcnt(0) before Phase 2 MMA ensures V_j is ready then.
        // =================================================================
        rt<bf16, KV_BLOCK, D_V, row_l, rt_32x16_4_s> V_j;
        {
            rt<bf16, KV_BLOCK, D_QK, row_l, rt_32x16_4_s> K_j;
            load(K_j, K_smem);   // 24 ds_reads (D_QK=192 / 16 chunks * 2)
            load(V_j, V_smem);   // 16 ds_reads (D_V=128 / 16 chunks * 2)
            // K_j is in front of V_j in the LDS FIFO. lgkmcnt(15) drains 25 reads
            // -> all K_j (24) done + 1 V_j done. K_j ready for MMA; V_j keeps
            // draining during Phase 1 ALU/MMA. Phase 2 takes a final lgkmcnt(0).
            asm volatile("s_waitcnt lgkmcnt(15)" ::: "memory");
            zero(acc_fp);
            mma_ABt(acc_fp, Q_i, K_j, acc_fp);
        }

        // acc_fp is [32 x 32] col_l rt_32x32_s. Layout for this tile:
        // rt_32x32 has stride=4, elements_per_thread=16, packed_per_thread=8.
        // For col_l with 32 reductions, threads_per_reduction = 32/16 = 2.
        // Actually more carefully: each lane holds 8 float2 packed = 16 floats.
        // For a 32x32 tile with 64 lanes, each lane has 16 elements.
        // Layout: with rt_32x32 stride=4, there are 4 stride groups (16/4=4).
        // Using the standard formula for rt_32x32 col_l:
        //   lane lid, register k: row = (lid >> 5) * 4 + (k & 3) + ((k >> 2) & 3) * 8,
        //                         col = lid & 31.  (Needs verification.)
        //
        // Let me use the formula as derived in the .bak2 reference file:
        //   ro = ((k >> 1) << 3) + ((k & 1) << 1);  for each of 8 packed floats.
        //   r0 = G_id * 4 + ro, r1 = r0 + 1
        //   where G_id = lid >> 5

        const int lane    = laneid();
        const int G_id    = (lane >> 5) & 1;      // 0 or 1 (group of 32)
        const int col_lo  = lane & 31;            // 0..31

        // Scale S_ij
        mul(acc_fp, acc_fp, P_SCALE_FACTOR);

        // Subtract L (pre-loaded to L_reg; already scaled by L_SCALE_FACTOR)
        #pragma unroll
        for (int k = 0; k < acc_fp.tiles[0][0].packed_per_thread; k++) {
            acc_fp.tiles[0][0].data[k].x -= L_reg[k * 2];
            acc_fp.tiles[0][0].data[k].y -= L_reg[k * 2 + 1];
        }

        if constexpr (causal) {
            // Skip mask loop entirely when this kj-block is fully below the diagonal
            // (q_pos >= kv_start + KV_BLOCK - 1 ensures all q_pos+r0 >= kv_start+col_lo
            // for r0 in [0..Q_TILE-1] and col_lo in [0..KV_BLOCK-1]).
            if (q_pos < kv_start + KV_BLOCK - 1) {
                #pragma unroll
                for (int k = 0; k < acc_fp.tiles[0][0].packed_per_thread; k++) {
                    int ro = ((k >> 1) << 3) + ((k & 1) << 1);
                    int r0 = G_id * 4 + ro, r1 = r0 + 1;
                    int kv_c = kv_start + col_lo;
                    if (q_pos + r0 < kv_c)
                        acc_fp.tiles[0][0].data[k].x = -__builtin_inff();
                    if (q_pos + r1 < kv_c)
                        acc_fp.tiles[0][0].data[k].y = -__builtin_inff();
                }
            }
        }

        // exp2 (clamp to 0)
        {
            #pragma unroll
            for (int k = 0; k < acc_fp.tiles[0][0].packed_per_thread; k++) {
                acc_fp.tiles[0][0].data[k].x = __builtin_fminf(acc_fp.tiles[0][0].data[k].x, 0.f);
                acc_fp.tiles[0][0].data[k].y = __builtin_fminf(acc_fp.tiles[0][0].data[k].y, 0.f);
                acc_fp.tiles[0][0].data[k] = base_ops::exp2::op(acc_fp.tiles[0][0].data[k]);
            }
        }

        // Keep P in registers (avoids LDS roundtrip). acc_fp[0][0] has 8 float2
        // = 16 floats per lane; copy into a small scratch array.
        float P_reg[16];
        #pragma unroll
        for (int k = 0; k < acc_fp.tiles[0][0].packed_per_thread; k++) {
            P_reg[k * 2]     = acc_fp.tiles[0][0].data[k].x;
            P_reg[k * 2 + 1] = acc_fp.tiles[0][0].data[k].y;
        }

        // =================================================================
        // Phase 2: dP_ij = dO @ V^T [32 x 32] col_l
        // V_j was loaded in Phase 1; ensure all V_j ds_reads have retired.
        // =================================================================
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        {
            zero(acc_fp);
            mma_ABt(acc_fp, dO_i, V_j, acc_fp);
        }

        // =================================================================
        // Phase 3: dS = P * (dP - delta) * dP_SCALE
        // delta pre-loaded into delta_reg in prologue (saves 16 ds_read2_b32 / iter).
        // =================================================================
        {
            #pragma unroll
            for (int k = 0; k < acc_fp.tiles[0][0].packed_per_thread; k++) {
                float dPx = acc_fp.tiles[0][0].data[k].x - delta_reg[k * 2];
                float dPy = acc_fp.tiles[0][0].data[k].y - delta_reg[k * 2 + 1];
                float Px  = P_reg[k * 2];
                float Py  = P_reg[k * 2 + 1];
                acc_fp.tiles[0][0].data[k].x = Px * dPx * dP_SCALE_FACTOR;
                acc_fp.tiles[0][0].data[k].y = Py * dPy * dP_SCALE_FACTOR;
            }
        }

        // =================================================================
        // Phase 4: dS col_l -> row_l via shared memory.
        // Store dS_bf (col_l rt_32x32) directly, no reinterpret (.bak2 style).
        // OPTIMIZATION: attn_smem is per-warp-partitioned. Each warp stores to
        // its own subslot and loads from its own subslot -- no cross-warp
        // communication, so no barrier needed. Only needs lgkmcnt for LDS.
        // =================================================================
        {
            // Fused fp32 col_l RT -> bf16 LDS store (TK store overload converts inline).
            // Eliminates the explicit copy(dS_bf, acc_fp) + bf16-tile-store roundtrip.
            auto my_attn_sub = subtile_inplace<Q_TILE, KV_BLOCK>(attn_smem, {warpid, 0});
            store(my_attn_sub, acc_fp);
        }
        // Wait for ds_write to land in LDS (single-warp visibility).
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");

        rt<bf16, Q_TILE, KV_BLOCK, row_l, rt_32x16_4_s> dS_row;
        {
            auto my_attn_sub = subtile_inplace<Q_TILE, KV_BLOCK>(attn_smem, {warpid, 0});
            load(dS_row, my_attn_sub);
        }
        // Issue K_col[0] load here so it overlaps with dS_row drain.
        load(K_col, subtile_inplace<KV_BLOCK, 64>(K_smem, {0, 0}));
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");

        // =================================================================
        // Phase 5: dQ_chunk += dS @ K for each 32-wide D column chunk
        // (Pre-loop prime handles the gfx950 "first mma drops 8 rows" bug.)
        // =================================================================
        {
            // Chunk 0: K_col already loaded above. Direct accumulation.
            mma_AB(dQa, dS_row, K_col, dQa);
            // Load K_col[1] NOW (before chunk 1 mma; K_col is free after mma
            // committed its operand).
            load(K_col, subtile_inplace<KV_BLOCK, 64>(K_smem, {0, 1}));

            // Chunk 1
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            mma_AB(dQb, dS_row, K_col, dQb);
            load(K_col, subtile_inplace<KV_BLOCK, 64>(K_smem, {0, 2}));

            // Chunk 2
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            mma_AB(dQc, dS_row, K_col, dQc);
        }

        // R88-pipeline: issue next-iter K/V global loads here. K_smem and
        // V_smem are now safe to overwrite (Phase 5's K_col reads completed
        // before the dQc MFMA started; we don't need them again until next
        // iter's Phase 1). The buffer_load_dwordx4 issue cycles overlap with
        // the dQc MFMA tail and the loop-back path.
        if (kj + 1 < last_kv_block) {
            G::load<1, false>(K_smem, g.K, {batch, kj + 1, kv_head, 0});
            G::load<1, false>(V_smem, g.V, {batch, kj + 1, kv_head, 0});
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_s_barrier();
        }
    }

    // =====================================================================
    // Epilogue: store dQ to global [B, H, N, D_QK] (BHND)
    //
    // dQ is 6 chunks of col_l rt_32x32_s fp32 (32x32 tile each).
    // For rt_32x32 col_l layout, each lane has 8 packed float2 (16 floats):
    //   register k (k=0..7): tile row = G_id*4 + ((k>>1)<<3) + ((k&1)<<1) + {0,1},
    //                        tile col = lane & 31
    // =====================================================================
    {
        bf16 *dQ_base = reinterpret_cast<bf16*>(g.dQg.raw_ptr);
        // int64 strides: at B=16 N=16384 H=64 D=192, s_b = 64*16384*192 = 2.0e8
        // and batch*s_b = 15*2e8 = 3.0e9 > INT32_MAX -> int overflow corrupts addr
        // and triggers "write to read-only page" GPU page fault.
        const long long s_b = (long long)g.dQg.template stride<0>();
        const long long s_h = (long long)g.dQg.template stride<1>();
        const long long s_n = (long long)g.dQg.template stride<2>();
        const int lane    = laneid();
        const int G_id    = (lane >> 5) & 1;
        const int col_lo  = lane & 31;

        // ATTN_N % STEP_Q == 0 and D_QK is multiple of 32, so q_r0/q_r1 < ATTN_N
        // and d_c < D_QK are always true. Drop branches for unconditional stores.
        auto store_chunk = [&](rt<float, Q_TILE, 64, col_l, rt_32x32_s> &tile,
                               int d_chunk_start)
        {
            const long long base_idx = (long long)batch * s_b
                                     + (long long)q_head * s_h
                                     + (long long)q_pos * s_n;
            #pragma unroll
            for (int j = 0; j < tile.width; j++) {
                #pragma unroll
                for (int k = 0; k < tile.tiles[0][j].packed_per_thread; k++) {
                    int ro = ((k >> 1) << 3) + ((k & 1) << 1);
                    int r0 = G_id * 4 + ro, r1 = r0 + 1;
                    int d_c = d_chunk_start + j * 32 + col_lo;
                    dQ_base[base_idx + (long long)r0 * s_n + d_c] =
                        __float2bfloat16(tile.tiles[0][j].data[k].x);
                    dQ_base[base_idx + (long long)r1 * s_n + d_c] =
                        __float2bfloat16(tile.tiles[0][j].data[k].y);
                }
            }
        };
        store_chunk(dQa, 0);
        store_chunk(dQb, 64);
        store_chunk(dQc, 128);
    }
}

void dispatch_bwd_dq_qparallel(attn_bwd_dq_qparallel_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_bwd_dq_qparallel_ker,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_bwd_dq_qparallel_ker<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

PYBIND11_MODULE(tk_kernel_bkwd_dq_qparallel, m) {
    m.doc() = "Q-parallel dQ backward kernel D_QK=192 D_V=128 (no atomics, 32x32 MFMA)";
    py::bind_function<dispatch_bwd_dq_qparallel>(m, "dispatch_bwd_dq",
        &attn_bwd_dq_qparallel_globals::Q,
        &attn_bwd_dq_qparallel_globals::K,
        &attn_bwd_dq_qparallel_globals::V,
        &attn_bwd_dq_qparallel_globals::dOg,
        &attn_bwd_dq_qparallel_globals::dQg,
        &attn_bwd_dq_qparallel_globals::L_vec,
        &attn_bwd_dq_qparallel_globals::delta_vec
    );
}
