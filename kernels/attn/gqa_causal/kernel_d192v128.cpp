#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#ifndef ATTN_B
constexpr int ATTN_B = 16; // batch size
#endif

#ifndef ATTN_H
constexpr int ATTN_H = 64; // number of heads
#endif

#ifndef ATTN_H_KV
constexpr int ATTN_H_KV = 8; // number of heads for key and value
#endif

constexpr int GROUP_SIZE = ATTN_H / ATTN_H_KV; // queries per KV head group

#ifndef ATTN_N
constexpr int ATTN_N = 8192; // sequence length
#endif

// Asymmetric head dimensions: Q/K use D_QK, V/O use D_V
#ifndef ATTN_D_QK
constexpr int ATTN_D_QK = 192; // Q/K head dimension
#endif

#ifndef ATTN_D_V
constexpr int ATTN_D_V = 128; // V/O head dimension
#endif

constexpr int D_QK = ATTN_D_QK;
constexpr int D_V  = ATTN_D_V;

constexpr int Q_BLOCK_SIZE = 32; // q block size
constexpr int KV_BLOCK_SIZE = 64; // kv block size
constexpr bool causal = true;

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

#define MFMA_MASK 0x08
#define VALU_MASK 0x02
#define EXP_MASK  0x400

using namespace kittens;
using _gl_QKVO = gl<bf16, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

#define SCHED_BARRIER(mask, cnt, group) __builtin_amdgcn_sched_group_barrier(mask, cnt, group)

template<int Pairs, int VALU_CNT, int Group>
__device__ __forceinline__ void sched_barrier_pairs() {
    SCHED_BARRIER(MFMA_MASK, 1, Group);
    SCHED_BARRIER(VALU_MASK, VALU_CNT, Group);
    if constexpr (Pairs > 1) sched_barrier_pairs<Pairs - 1, VALU_CNT, Group>();
}

template<int Pairs, int EXP_CNT, int Group>
__device__ __forceinline__ void sched_barrier_exp_pairs() {
    SCHED_BARRIER(MFMA_MASK, 1, Group);
    SCHED_BARRIER(EXP_MASK, EXP_CNT, Group);
    if constexpr (Pairs > 1) sched_barrier_exp_pairs<Pairs - 1, EXP_CNT, Group>();
}

template<typename T, ducks::rt_layout::all layout, ducks::rt_shape::all shape>
__device__ inline void exp2(rt_base<T, layout, shape> &dst, const rt_base<T, layout, shape> &src) {
    static_assert(std::is_same_v<shape, rt_32x32_s>, "Only 32x32 tiles are supported");

    #pragma unroll
    for(int k = 0; k < dst.packed_per_thread; k++) {
        dst.data[k] = base_ops::exp2::op(src.data[k]);
    }

}

constexpr float RESCALE_THRESHOLD = 8.0f;  // Only rescale if scale factor < 2^-4

template<typename RV>
__device__ __forceinline__ int rv_all_below(const RV& prev, const RV& cur, float T) {
    int ok = 1;
    #pragma unroll
    for (int o = 0; o < RV::outer_dim; ++o) {
        for (int i = 0; i < RV::inner_dim; ++i) {
            ok &= (float(cur.data[o][i]) - float(prev.data[o][i]) <= T);
        }
    }
    return ok;  // 1 or 0, per lane
}
// Wave-uniform: true iff EVERY lane says "all below"
__device__ __forceinline__ int wave_all_ok(int lane_ok) {
    return __all(lane_ok);  // 1 fast vote; no ds_bpermute ladder
}

// Q tile aliases (D_QK = 192)
template<typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using q_tile            = rt<T, Q_BLOCK_SIZE,  D_QK, L, S>;
template<typename T=bf16, typename L=col_l, typename S=rt_16x32_s> using q_tile_transposed  = rt<T, D_QK, Q_BLOCK_SIZE, L, S>;
// K half-tile: 32 rows x D_QK cols (one half of KV_BLOCK_SIZE=64)
// K is loaded from shared in halves to reduce register pressure
constexpr int K_HALF = KV_BLOCK_SIZE / 2; // 32
template<typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using k_half_tile            = rt<T, K_HALF, D_QK, L, S>;
template<typename T=bf16, typename L=col_l, typename S=rt_16x32_s> using k_half_tile_transposed = rt<T, D_QK, K_HALF, L, S>;

// V/O tile aliases (D_V = 128)
template<typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using v_tile             = rt<T, KV_BLOCK_SIZE, D_V, L, S>;
template<typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using o_tile             = rt<T, Q_BLOCK_SIZE, D_V, L, S>;
template<typename T=bf16, typename L=col_l, typename S=rt_16x32_s> using o_tile_transposed  = rt<T, D_V, Q_BLOCK_SIZE, L, S>;

// Attention tile is D-independent (KV_BLOCK_SIZE x Q_BLOCK_SIZE = 64x32)
template<typename T=float, typename L=col_l, typename S=rt_16x32_4_s> using attn_tile = rt<T, KV_BLOCK_SIZE, Q_BLOCK_SIZE, L, S>;

// Split QK MMA: loads K halves from shared memory to reduce register pressure.
// Instead of holding a full 192x64 k_reg_transposed (96 VGPRs), loads 192x32 halves (48 VGPRs).
// Computes att_block = K^T * Q^T (mma_AtB) in two passes (top 32 rows, bottom 32 rows of K).
// Full K tile type for loading from shared (then immediately split for MMA)
template<typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using k_tile = rt<T, KV_BLOCK_SIZE, D_QK, L, S>;
template<typename T=bf16, typename L=col_l, typename S=rt_16x32_s> using k_tile_transposed = rt<T, D_QK, KV_BLOCK_SIZE, L, S>;

template<ducks::st::all ST>
__device__ inline void mma_qk_split(
    attn_tile<float, col_l, rt_32x32_s> &att,
    ST &k_smem_tile,
    const q_tile_transposed<bf16> &q_t)
{
    // Load full K tile, transpose, then do MMA.
    // Scoping ensures k_reg dies after transpose to reduce register pressure.
    k_tile_transposed<bf16> k_t;
    {
        k_tile<bf16> k_reg;
        load(k_reg, k_smem_tile);
        asm volatile("s_waitcnt lgkmcnt(0)");
        transpose(k_t, k_reg);
    } // k_reg dies

    mma_AtB(att, k_t, q_t, att);
}

/**********************************************************/
template<int THR_X, int THR_Y>
__device__ inline void mask_vec2_imm(uint32_t rel_vgpr, uint32_t neg_inf_vgpr,
                                     uint32_t& x_ref, uint32_t& y_ref) {

    uint64_t x_mask, y_mask;
    asm volatile(
        // x: rel < THR_X ?
        "v_cmp_lt_i32_e64 %0, %6, %7\n\t"
        // y: rel < THR_Y ?
        "v_cmp_lt_i32_e64 %1, %6, %9\n\t"
        "v_cndmask_b32_e64 %2, %4, %8, %0\n\t"
        "v_cndmask_b32_e64 %3, %5, %8, %1\n\t"
        : "=s"(x_mask), "=s"(y_mask), "=v"(x_ref), "=v"(y_ref)
        : "v"(x_ref), "v"(y_ref), "v"(rel_vgpr),
          "n"(THR_X), "v"(neg_inf_vgpr), "n"(THR_Y)
        : "vcc"
    );
}

template<ducks::rt::col_layout RT>
__device__ inline void mask_kv_tile(RT &dst, int q_abs, int k_abs, uint32_t neg_inf_v, int lane) {
    const int col  = lane & 31;                 // 0..31 column within the 32-wide col tile

    // Absolute positions
    const int q_base = q_abs * Q_BLOCK_SIZE;    // start index for this Q tile
    const int k_base = k_abs * KV_BLOCK_SIZE;   // start index for this K/V tile

    // q position for this lane's column
    const int q_pos  = q_base + col;

    #pragma unroll
    for (int i = 0; i < dst.height; ++i) {
        // Row base of the 32x* chunk produced by MFMA
        const int row_base = (i * 32) + ((lane >> 5) << 2); // multiples of 4

        // Relative index of the FIRST element in this row-chunk w.r.t. q_pos
        const int rel0 = q_pos - (k_base + row_base);
        const uint32_t rel = static_cast<uint32_t>(rel0);

        #pragma unroll
        for (int j = 0; j < dst.width; ++j) {
            auto& d0x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[0].x);
            auto& d0y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[0].y);
            auto& d1x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[1].x);
            auto& d1y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[1].y);
            auto& d2x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[2].x);
            auto& d2y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[2].y);
            auto& d3x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[3].x);
            auto& d3y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[3].y);
            auto& d4x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[4].x);
            auto& d4y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[4].y);
            auto& d5x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[5].x);
            auto& d5y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[5].y);
            auto& d6x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[6].x);
            auto& d6y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[6].y);
            auto& d7x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[7].x);
            auto& d7y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[7].y);

            mask_vec2_imm< 0, 1 >(rel, neg_inf_v, d0x, d0y);
            mask_vec2_imm< 2, 3 >(rel, neg_inf_v, d1x, d1y);
            mask_vec2_imm< 8, 9 >(rel, neg_inf_v, d2x, d2y);
            mask_vec2_imm<10,11 >(rel, neg_inf_v, d3x, d3y);
            mask_vec2_imm<16,17 >(rel, neg_inf_v, d4x, d4y);
            mask_vec2_imm<18,19 >(rel, neg_inf_v, d5x, d5y);
            mask_vec2_imm<24,25 >(rel, neg_inf_v, d6x, d6y);
            mask_vec2_imm<26,27 >(rel, neg_inf_v, d7x, d7y);
        }
    }
}

/**********************************************************/

struct attn_globals {
    _gl_QKVO Qg, Kg, Vg, Og;
    gl<float, -1, -1, -1, -1> L_vec;
    hipStream_t stream;
    dim3 grid() { return dim3(ATTN_H, ((ATTN_N / Q_BLOCK_SIZE + NUM_WARPS - 1) / NUM_WARPS), ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

// Reduced occupancy from 2 to 1 due to increased register pressure with D_QK=192
__launch_bounds__(NUM_THREADS, 1)
__global__ void attend_ker(const attn_globals g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    // K uses D_QK=192, V uses D_V=128
    st_bf<KV_BLOCK_SIZE, D_QK, st_32x32_s> (&k_smem)[2] = al.allocate<st_bf<KV_BLOCK_SIZE, D_QK, st_32x32_s>, 2>();
    st_bf<KV_BLOCK_SIZE, D_V,  st_8x32_s>  (&v_smem)[2] = al.allocate<st_bf<KV_BLOCK_SIZE, D_V,  st_8x32_s>,  2>();
    // Total shared: 2*(64*192*2) + 2*(64*128*2) = 48KB + 32KB = 80KB (fits in 160KB CDNA4 LDS)

    const int head_idx = (blockIdx.x % ATTN_H_KV) * GROUP_SIZE + (blockIdx.x / ATTN_H_KV);
    const int batch_idx = blockIdx.z;
    const int head_idx_kv = head_idx / GROUP_SIZE;
    const int block_tile_idx = blockIdx.y;
    const int tile_idx = block_tile_idx * NUM_WARPS + warpid();
    const int stagger = warpid() / 4;
    const int lane = laneid();

    /********** Readfirstlane hoisting **********/
    // Create base buffer resources once
    const bf16* k_base = (bf16*)&g.Kg[{batch_idx, 0, head_idx_kv, 0}];
    const bf16* v_base = (bf16*)&g.Vg[{batch_idx, 0, head_idx_kv, 0}];
    const int k_row_stride = g.Kg.template stride<1>() * sizeof(bf16);
    const int v_row_stride = g.Vg.template stride<1>() * sizeof(bf16);
    i32x4 k_srsrc_base = make_srsrc(k_base, k_row_stride * ATTN_N, k_row_stride);
    i32x4 v_srsrc_base = make_srsrc(v_base, v_row_stride * ATTN_N, v_row_stride);

    const int wid = warpid() % NUM_WARPS;
    constexpr int elem_per_warp = (16 / sizeof(bf16)) * kittens::WARP_THREADS;
    uint32_t k_lds_base_0 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&k_smem[0].data[0]) + wid * elem_per_warp * sizeof(bf16)
    ));
    uint32_t v_lds_base_0 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&v_smem[0].data[0]) + wid * elem_per_warp * sizeof(bf16)
    ));
    uint32_t k_lds_base_1 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&k_smem[1].data[0]) + wid * elem_per_warp * sizeof(bf16)
    ));
    uint32_t v_lds_base_1 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&v_smem[1].data[0]) + wid * elem_per_warp * sizeof(bf16)
    ));
    /********** Swizzle **********/

    const int num_tiles = ATTN_N / KV_BLOCK_SIZE;
    const int max_tile_idx = block_tile_idx * NUM_WARPS + NUM_WARPS - 1;
    const int max_q_end_pos = (max_tile_idx + 1) * Q_BLOCK_SIZE;
    int max_num_tiles = (max_q_end_pos + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
    if constexpr (causal) max_num_tiles = min(max_num_tiles, num_tiles);
    else max_num_tiles = num_tiles;
    const int q_start_pos = tile_idx * Q_BLOCK_SIZE;

    // 1/sqrt(D_QK) * log2(e) for base-2 softmax
    constexpr float TEMPERATURE_SCALE = 0.07216878364870322f * 1.44269504089f; // 1/sqrt(192) * log2(e)
    uint32_t neg_inf_v = 0xff800000;

    // Initialize all of the register tiles.
    // Q uses D_QK=192, transposed for mma_AtB. K is loaded from shared on-the-fly (no persistent k_reg).
    q_tile_transposed<bf16> q_reg_transposed;

    // V/O tiles use D_V=128 (same as original D=128 kernel)
    v_tile<bf16, col_l, rt_16x32_4_s> v_reg;
    o_tile_transposed<float, col_l, rt_32x32_s> o_reg; // Output tile.
    attn_tile<float, col_l, rt_32x32_s> att_block[2]; // attention tile, in float.
    attn_tile<bf16, col_l, rt_32x32_s> att_block_bf16;
    attn_tile<bf16, col_l, rt_16x32_4_s> att_block_bf16_in;
    typename attn_tile<float, col_l, rt_32x32_s>::row_vec max_vec, norm_vec, max_vec_prev, scale_vec;

    zero(o_reg);
    zero(norm_vec);
    ones(scale_vec);

    // Separate memcpy_per_tile for K (D_QK) and V (D_V) since they have different widths
    using T_K = typename st_bf<KV_BLOCK_SIZE, D_QK, st_32x32_s>::dtype;
    constexpr int bytes_per_thread_K = st_32x32_s::template bytes_per_thread<T_K>();
    constexpr int bytes_per_memcpy_K = bytes_per_thread_K * NUM_THREADS;
    constexpr int memcpy_per_tile_K = KV_BLOCK_SIZE * D_QK * sizeof(T_K) / bytes_per_memcpy_K; // =3

    using T_V = typename st_bf<KV_BLOCK_SIZE, D_V, st_8x32_s>::dtype;
    constexpr int bytes_per_thread_V = st_8x32_s::template bytes_per_thread<T_V>();
    constexpr int bytes_per_memcpy_V = bytes_per_thread_V * NUM_THREADS;
    constexpr int memcpy_per_tile_V = KV_BLOCK_SIZE * D_V * sizeof(T_V) / bytes_per_memcpy_V; // =2

    uint32_t swizzled_offsets_K[memcpy_per_tile_K];
    uint32_t swizzled_offsets_V[memcpy_per_tile_V];
    G::prefill_swizzled_offsets<1, false>(k_smem[0], g.Kg, swizzled_offsets_K);
    G::prefill_swizzled_offsets<1, false>(v_smem[0], g.Vg, swizzled_offsets_V);

    // Load K0 to shared
    G::load<1, false>(k_smem[0], g.Kg, {batch_idx, 0, head_idx_kv, 0}, swizzled_offsets_K, k_srsrc_base, k_base, k_lds_base_0);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    // Load Q to registers (Q has D_QK=192 columns), scale, transpose
    {
        q_tile<float> q_reg_fl;
        load<1, q_tile<float>, _gl_QKVO>(q_reg_fl, g.Qg, {batch_idx, tile_idx, head_idx, 0});
        mul(q_reg_fl, q_reg_fl, TEMPERATURE_SCALE);
        q_tile<bf16> q_reg;
        copy(q_reg, q_reg_fl);
        transpose(q_reg_transposed, q_reg);
    } // q_reg_fl and q_reg go out of scope here, freeing their registers

    // Load K1 and V0 to shared
    G::load<1, false>(k_smem[1], g.Kg, {batch_idx, 1, head_idx_kv, 0}, swizzled_offsets_K, k_srsrc_base, k_base, k_lds_base_1);
    G::load<1, false>(v_smem[0], g.Vg, {batch_idx, 0, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base, v_base, v_lds_base_0);
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(3)"); // K1=3 + V0=2 = 5 outstanding, wait for 2 done, 3 remain
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    // QK0: K^T * Q^T = S[64x32], K loaded from shared in halves to reduce register pressure
    zero(att_block[0]);
    mma_qk_split(att_block[0], k_smem[0], q_reg_transposed);
    __builtin_amdgcn_sched_barrier(0);
    if constexpr (causal) {
        const int kv_end_pos = (1) * KV_BLOCK_SIZE;
        if (__builtin_expect(q_start_pos < kv_end_pos, 0)) {
            mask_kv_tile(att_block[0], tile_idx, 0, neg_inf_v, lane);
        }
    }
    // Partial softmax of QK0
    col_max(max_vec, att_block[0]);
    copy(max_vec_prev, max_vec);
    sub_col(att_block[0], att_block[0], max_vec);
    exp2(att_block[0].tiles[0][0], att_block[0].tiles[0][0]);
    __builtin_amdgcn_sched_barrier(0);

    if (stagger) {
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
    }

    // Load K2 and V1 to shared (K1 already in k_smem[1], will be loaded on-the-fly in MMA)
    G::load<1, false>(k_smem[0], g.Kg, {batch_idx, 2, head_idx_kv, 0}, swizzled_offsets_K, k_srsrc_base, k_base, k_lds_base_0);
    G::load<1, false>(v_smem[1], g.Vg, {batch_idx, 1, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base, v_base, v_lds_base_1);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(5)"); // K2=3 + V1=2 = 5 outstanding from this batch
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    int pending_scale = 0;

    // hot loop
    for (int j = 3; j < max_num_tiles - 1; j += 2) {
        // Cluster 0: QK_odd (K from k_smem[1]) + Finish softmax for QK_even
        zero(att_block[1]);
        mma_qk_split(att_block[1], k_smem[1], q_reg_transposed);
        exp2(att_block[0].tiles[1][0], att_block[0].tiles[1][0]);
        if (pending_scale) {
            mul(norm_vec, norm_vec, scale_vec);
        }
        col_sum(norm_vec, att_block[0], norm_vec);
        copy(att_block_bf16, att_block[0]);
        att_block_bf16_in = *reinterpret_cast<attn_tile<bf16, col_l, rt_16x32_4_s>*>(&att_block_bf16);
        sched_barrier_exp_pairs<6, 3, 1>();
        sched_barrier_pairs<10, 5, 1>();
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 1: Load next K to shared + Load V to registers
        G::load<1, false>(k_smem[1], g.Kg, {batch_idx, j, head_idx_kv, 0}, swizzled_offsets_K, k_srsrc_base, k_base, k_lds_base_1);
        load(v_reg, v_smem[0]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(5)"); // K=3 + V=2 outstanding from previous
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 2: A*V accumulation (PV, D_V=128, unchanged) + Partial softmax for QK_odd
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 0), subtile_inplace<16>(att_block_bf16_in, 0), o_reg);
        col_max(max_vec, att_block[1], max_vec_prev);
        sched_barrier_pairs<4, 5, 2>();
        int lane_ok = rv_all_below(max_vec_prev, max_vec, RESCALE_THRESHOLD);
        int all_ok  = wave_all_ok(lane_ok);
        if (__builtin_expect(all_ok, 1)) {
            copy(max_vec, max_vec_prev);
            pending_scale = 0;
        } else {
            sub(scale_vec, max_vec_prev, max_vec);
            exp2(scale_vec, scale_vec);
            mul_col(o_reg, o_reg, scale_vec);
            copy(max_vec_prev, max_vec);
            pending_scale = 1;
        }
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 1), subtile_inplace<16>(att_block_bf16_in, 1), o_reg);
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 2), subtile_inplace<16>(att_block_bf16_in, 2), o_reg);
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 3), subtile_inplace<16>(att_block_bf16_in, 3), o_reg);
        sub_col(att_block[1], att_block[1], max_vec);
        exp2(att_block[1].tiles[0][0], att_block[1].tiles[0][0]);
        sched_barrier_pairs<6, 5, 2>();
        sched_barrier_exp_pairs<6, 3, 2>();
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 3: Load V to shared (K stays in k_smem[0], loaded on-the-fly in MMA)
        G::load<1, false>(v_smem[0], g.Vg, {batch_idx, j - 1, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base, v_base, v_lds_base_0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 4: QK_even (K from k_smem[0]) + Finish softmax for QK_odd
        zero(att_block[0]);
        mma_qk_split(att_block[0], k_smem[0], q_reg_transposed);
        exp2(att_block[1].tiles[1][0], att_block[1].tiles[1][0]);
        if (pending_scale) {
            mul(norm_vec, norm_vec, scale_vec);
        }
        col_sum(norm_vec, att_block[1], norm_vec);
        copy(att_block_bf16, att_block[1]);
        att_block_bf16_in = *reinterpret_cast<attn_tile<bf16, col_l, rt_16x32_4_s>*>(&att_block_bf16);
        sched_barrier_exp_pairs<6, 3, 3>();
        sched_barrier_pairs<10, 5, 3>();
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 5: Load next K to shared + Load V to registers + causal mask
        G::load<1, false>(k_smem[0], g.Kg, {batch_idx, j + 1, head_idx_kv, 0}, swizzled_offsets_K, k_srsrc_base, k_base, k_lds_base_0);
        load(v_reg, v_smem[1]);
        if constexpr (causal) {
            const int kv_end_pos = (j) * KV_BLOCK_SIZE;
            if (q_start_pos < kv_end_pos) {
                mask_kv_tile(att_block[0], tile_idx, j - 1, neg_inf_v, lane);
            }
        }
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(5)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 6: A*V accumulation + Partial softmax for QK_even
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 0), subtile_inplace<16>(att_block_bf16_in, 0), o_reg);
        col_max(max_vec, att_block[0], max_vec_prev);
        sched_barrier_pairs<4, 5, 4>();
        lane_ok = rv_all_below(max_vec_prev, max_vec, RESCALE_THRESHOLD);
        all_ok  = wave_all_ok(lane_ok);
        if (__builtin_expect(all_ok, 1)) {
            copy(max_vec, max_vec_prev);
            pending_scale = 0;
        } else {
            sub(scale_vec, max_vec_prev, max_vec);
            exp2(scale_vec, scale_vec);
            mul_col(o_reg, o_reg, scale_vec);
            copy(max_vec_prev, max_vec);
            pending_scale = 1;
        }
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 1), subtile_inplace<16>(att_block_bf16_in, 1), o_reg);
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 2), subtile_inplace<16>(att_block_bf16_in, 2), o_reg);
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 3), subtile_inplace<16>(att_block_bf16_in, 3), o_reg);
        sub_col(att_block[0], att_block[0], max_vec);
        exp2(att_block[0].tiles[0][0], att_block[0].tiles[0][0]);
        sched_barrier_pairs<6, 5, 4>();
        sched_barrier_exp_pairs<6, 3, 4>();
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 7: Load V to shared (K stays in k_smem[1] for next iteration's Cluster 0)
        G::load<1, false>(v_smem[1], g.Vg, {batch_idx, j, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base, v_base, v_lds_base_1);
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    // Epilogue
    // Cluster 0: QK(n-3) from k_smem[1] + Finish softmax
    zero(att_block[1]);
    mma_qk_split(att_block[1], k_smem[1], q_reg_transposed);
    exp2(att_block[0].tiles[1][0], att_block[0].tiles[1][0]);
    mul(norm_vec, norm_vec, scale_vec);

    col_sum(norm_vec, att_block[0], norm_vec);
    copy(att_block_bf16, att_block[0]);
    att_block_bf16_in = *reinterpret_cast<attn_tile<bf16, col_l, rt_16x32_4_s>*>(&att_block_bf16);
    sched_barrier_exp_pairs<6, 3, 5>();
    sched_barrier_pairs<10, 5, 5>();
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 1: Load K(n-1), Load V(n-4)
    G::load<1, false>(k_smem[1], g.Kg, {batch_idx, max_num_tiles - 1, head_idx_kv, 0}, swizzled_offsets_K, k_srsrc_base, k_base, k_lds_base_1);
    load(v_reg, v_smem[0]);
    if constexpr (causal) {
        const int kv_end_pos = (max_num_tiles - 2) * KV_BLOCK_SIZE;
        if (__builtin_expect(q_start_pos < kv_end_pos, 0)) {
            mask_kv_tile(att_block[1], tile_idx, max_num_tiles - 3, neg_inf_v, lane);
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(5)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 2: A*V + Partial softmax
    __builtin_amdgcn_s_setprio(1);
    mma_AtB(o_reg, v_reg, att_block_bf16_in, o_reg);
    col_max(max_vec, att_block[1], max_vec_prev);
    sub(scale_vec, max_vec_prev, max_vec);
    copy(max_vec_prev, max_vec);
    exp2(scale_vec, scale_vec);
    sub_col(att_block[1], att_block[1], max_vec);
    exp2(att_block[1].tiles[0][0], att_block[1].tiles[0][0]);
    sched_barrier_pairs<10, 5, 6>();
    sched_barrier_exp_pairs<6, 3, 6>();
    __builtin_amdgcn_sched_barrier(0);
    mul_col(o_reg, o_reg, scale_vec);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 3: Load V(n-2) (K(n-2) stays in k_smem[0])
    G::load<1, false>(v_smem[0], g.Vg, {batch_idx, max_num_tiles - 2, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base, v_base, v_lds_base_0);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(2)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 4: QK(n-2) from k_smem[0] + Finish softmax
    zero(att_block[0]);
    mma_qk_split(att_block[0], k_smem[0], q_reg_transposed);
    exp2(att_block[1].tiles[1][0], att_block[1].tiles[1][0]);
    mul(norm_vec, norm_vec, scale_vec);
    col_sum(norm_vec, att_block[1], norm_vec);
    copy(att_block_bf16, att_block[1]);
    att_block_bf16_in = *reinterpret_cast<attn_tile<bf16, col_l, rt_16x32_4_s>*>(&att_block_bf16);
    sched_barrier_exp_pairs<6, 3, 7>();
    sched_barrier_pairs<10, 5, 7>();
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 5: Load V(n-3)
    load(v_reg, v_smem[1]);
    if constexpr (causal) {
        const int kv_end_pos = (max_num_tiles - 1) * KV_BLOCK_SIZE;
        if (__builtin_expect(q_start_pos < kv_end_pos, 1)) {
            mask_kv_tile(att_block[0], tile_idx, max_num_tiles - 2, neg_inf_v, lane);
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 6: A*V + Partial softmax
    __builtin_amdgcn_s_setprio(1);
    mma_AtB(o_reg, v_reg, att_block_bf16_in, o_reg);
    col_max(max_vec, att_block[0], max_vec_prev);
    sub(scale_vec, max_vec_prev, max_vec);
    copy(max_vec_prev, max_vec);
    exp2(scale_vec, scale_vec);
    sub_col(att_block[0], att_block[0], max_vec);
    exp2(att_block[0].tiles[0][0], att_block[0].tiles[0][0]);
    sched_barrier_pairs<10, 5, 8>();
    sched_barrier_exp_pairs<6, 3, 8>();
    __builtin_amdgcn_sched_barrier(0);
    mul_col(o_reg, o_reg, scale_vec);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 7: Load V(n-1) (K(n-1) already in k_smem[1] from Cluster 1)
    G::load<1, false>(v_smem[1], g.Vg, {batch_idx, max_num_tiles - 1, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base, v_base, v_lds_base_1);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(2)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 8: QK(n-1) from k_smem[1] + Finish softmax
    zero(att_block[1]);
    mma_qk_split(att_block[1], k_smem[1], q_reg_transposed);
    exp2(att_block[0].tiles[1][0], att_block[0].tiles[1][0]);
    mul(norm_vec, norm_vec, scale_vec);
    col_sum(norm_vec, att_block[0], norm_vec);
    copy(att_block_bf16, att_block[0]);
    att_block_bf16_in = *reinterpret_cast<attn_tile<bf16, col_l, rt_16x32_4_s>*>(&att_block_bf16);
    sched_barrier_exp_pairs<6, 3, 9>();
    sched_barrier_pairs<10, 5, 9>();
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 9: Load V(n-2) + causal mask
    load(v_reg, v_smem[0]);
    if constexpr (causal) {
        const int kv_end_pos = (max_num_tiles) * KV_BLOCK_SIZE;
        if (__builtin_expect(q_start_pos < kv_end_pos, 1)) {
            mask_kv_tile(att_block[1], tile_idx, max_num_tiles - 1, neg_inf_v, lane);
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 10: A*V + Full softmax for last QK
    mma_AtB(o_reg, v_reg, att_block_bf16_in, o_reg);
    col_max(max_vec, att_block[1], max_vec_prev);
    sub(scale_vec, max_vec_prev, max_vec);
    copy(max_vec_prev, max_vec);
    exp2(scale_vec, scale_vec);

    sub_col(att_block[1], att_block[1], max_vec);
    exp2(att_block[1].tiles[0][0], att_block[1].tiles[0][0]);
    sched_barrier_pairs<10, 5, 10>();
    sched_barrier_exp_pairs<6, 3, 10>();
    __builtin_amdgcn_sched_barrier(0);

    exp2(att_block[1].tiles[1][0], att_block[1].tiles[1][0]);
    mul(norm_vec, norm_vec, scale_vec);

    col_sum(norm_vec, att_block[1], norm_vec);
    copy(att_block_bf16, att_block[1]);
    att_block_bf16_in = *reinterpret_cast<attn_tile<bf16, col_l, rt_16x32_4_s>*>(&att_block_bf16);

    __builtin_amdgcn_sched_barrier(0);
    mul_col(o_reg, o_reg, scale_vec);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 11: Load V(n-1)
    load(v_reg, v_smem[1]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 12: Final A*V + normalize
    mma_AtB(o_reg, v_reg, att_block_bf16_in, o_reg);
    div_col(o_reg, o_reg, norm_vec);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Conclusion
    if (!stagger) {
        __builtin_amdgcn_s_barrier();
    }

    // Output is D_V=128 wide: transpose from (D_V x Q_BLOCK) col_l to (Q_BLOCK x D_V) row_l
    o_tile<float, row_l, rt_32x32_s> o_reg_transposed;
    transpose(o_reg_transposed, o_reg);
    store<1>(g.Og, o_reg_transposed, {batch_idx, tile_idx, head_idx, 0});

    // LSE output: convert from base-2 back to natural log
    mul(max_vec, max_vec, 0.69314718056f);
    log(norm_vec, norm_vec);
    add(norm_vec, norm_vec, max_vec);
    store(g.L_vec, norm_vec, {batch_idx, head_idx, 0, tile_idx});
}

void dispatch_micro(attn_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_ker, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_ker<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module - asymmetric causal attention D_QK=192 D_V=128";
    py::bind_function<dispatch_micro>(m, "dispatch_micro",
        &attn_globals::Qg,
        &attn_globals::Kg,
        &attn_globals::Vg,
        &attn_globals::Og,
        &attn_globals::L_vec
    );
}
