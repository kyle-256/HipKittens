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

// NUM_WARPS=4: 1 wave/SIMD, 512 VGPRs budget — enough for persistent k_reg+k_reg_transposed at D_QK=192
#define NUM_WARPS 4
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

constexpr float RESCALE_THRESHOLD = 8.0f;

template<typename RV>
__device__ __forceinline__ int rv_all_below(const RV& prev, const RV& cur, float T) {
    int ok = 1;
    #pragma unroll
    for (int o = 0; o < RV::outer_dim; ++o) {
        for (int i = 0; i < RV::inner_dim; ++i) {
            ok &= (float(cur.data[o][i]) - float(prev.data[o][i]) <= T);
        }
    }
    return ok;
}
__device__ __forceinline__ int wave_all_ok(int lane_ok) {
    return __all(lane_ok);
}

// Q/K tiles use D_QK=192 (same pattern as original kernel.cpp, just wider)
template<int D, typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using qo_tile = rt<T, Q_BLOCK_SIZE, D, L, S>;
template<int D, typename T=bf16, typename L=col_l, typename S=rt_16x32_s> using qo_tile_transposed = rt<T, D, Q_BLOCK_SIZE, L, S>;
template<int D, typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using kv_tile = rt<T, KV_BLOCK_SIZE, D, L, S>;
template<int D, typename T=bf16, typename L=col_l, typename S=rt_16x32_s> using kv_tile_transposed = rt<T, D, KV_BLOCK_SIZE, L, S>;
template<typename T=float, typename L=col_l, typename S=rt_16x32_4_s> using attn_tile = rt<T, KV_BLOCK_SIZE, Q_BLOCK_SIZE, L, S>;

/**********************************************************/
template<int THR_X, int THR_Y>
__device__ inline void mask_vec2_imm(uint32_t rel_vgpr, uint32_t neg_inf_vgpr,
                                     uint32_t& x_ref, uint32_t& y_ref) {

    uint64_t x_mask, y_mask;
    asm volatile(
        "v_cmp_lt_i32_e64 %0, %6, %7\n\t"
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
    const int col  = lane & 31;

    const int q_base = q_abs * Q_BLOCK_SIZE;
    const int k_base = k_abs * KV_BLOCK_SIZE;

    const int q_pos  = q_base + col;

    #pragma unroll
    for (int i = 0; i < dst.height; ++i) {
        const int row_base = (i * 32) + ((lane >> 5) << 2);

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
    // NUM_WARPS=4: grid.y doubles compared to NUM_WARPS=8
    dim3 grid() { return dim3(ATTN_H, ((ATTN_N / Q_BLOCK_SIZE + NUM_WARPS - 1) / NUM_WARPS), ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

// NUM_WARPS=4, 256 threads, 1 block/CU => 512 VGPRs, no spill expected
__launch_bounds__(256, 1)
__global__ void attend_ker(const attn_globals g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    // K uses D_QK=192, V uses D_V=128
    st_bf<KV_BLOCK_SIZE, D_QK, st_32x32_s> (&k_smem)[2] = al.allocate<st_bf<KV_BLOCK_SIZE, D_QK, st_32x32_s>, 2>();
    st_bf<KV_BLOCK_SIZE, D_V,  st_8x32_s>  (&v_smem)[2] = al.allocate<st_bf<KV_BLOCK_SIZE, D_V,  st_8x32_s>,  2>();
    // Total shared: 2*(64*192*2) + 2*(64*128*2) = 48KB + 32KB = 80KB

    const int head_idx = (blockIdx.x % ATTN_H_KV) * GROUP_SIZE + (blockIdx.x / ATTN_H_KV);
    const int batch_idx = blockIdx.z;
    const int head_idx_kv = head_idx / GROUP_SIZE;
    const int block_tile_idx = blockIdx.y;
    const int tile_idx = block_tile_idx * NUM_WARPS + warpid();
    // NUM_WARPS=4: stagger = warpid()/2 (was /4 with 8 warps)
    const int stagger = warpid() / 2;
    const int lane = laneid();

    /********** Readfirstlane hoisting **********/
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
    // With NUM_WARPS=4, early blocks can have max_num_tiles < 4.
    // The prologue+epilogue structure requires at least 4 tiles.
    // Extra tiles beyond causal boundary are fully masked (contribute zero).
    max_num_tiles = max(max_num_tiles, 4);
    const int q_start_pos = tile_idx * Q_BLOCK_SIZE;

    // 1/sqrt(D_QK=192) * log2(e) for base-2 softmax
    constexpr float TEMPERATURE_SCALE = 0.07216878364870322f * 1.44269504089f; // 1/sqrt(192) * log2(e)
    uint32_t neg_inf_v = 0xff800000;

    // Standard register tile layout (same as kernel.cpp, just with D_QK for Q/K, D_V for V/O)
    qo_tile<D_QK, bf16> q_reg;
    qo_tile_transposed<D_QK, bf16> q_reg_transposed;
    kv_tile<D_QK, bf16> k_reg;
    kv_tile_transposed<D_QK, bf16> k_reg_transposed;

    kv_tile<D_V, bf16, col_l, rt_16x32_4_s> v_reg;
    qo_tile_transposed<D_V, float, col_l, rt_32x32_s> o_reg;
    attn_tile<float, col_l, rt_32x32_s> att_block[2];
    attn_tile<bf16, col_l, rt_32x32_s> att_block_bf16;
    attn_tile<bf16, col_l, rt_16x32_4_s> att_block_bf16_in;
    typename attn_tile<float, col_l, rt_32x32_s>::row_vec max_vec, norm_vec, max_vec_prev, scale_vec;

    zero(o_reg);
    zero(norm_vec);
    ones(scale_vec);

    // With NUM_WARPS=4 (256 threads), bytes_per_memcpy halves:
    // K: 64*192*2 = 24576 bytes, bytes_per_memcpy = 16*256 = 4096, memcpy_per_tile_K = 6
    // V: 64*128*2 = 16384 bytes, bytes_per_memcpy = 16*256 = 4096, memcpy_per_tile_V = 4
    using T_K = typename st_bf<KV_BLOCK_SIZE, D_QK, st_32x32_s>::dtype;
    constexpr int bytes_per_thread_K = st_32x32_s::template bytes_per_thread<T_K>();
    constexpr int bytes_per_memcpy_K = bytes_per_thread_K * NUM_THREADS;
    constexpr int memcpy_per_tile_K = KV_BLOCK_SIZE * D_QK * sizeof(T_K) / bytes_per_memcpy_K; // =6

    using T_V = typename st_bf<KV_BLOCK_SIZE, D_V, st_8x32_s>::dtype;
    constexpr int bytes_per_thread_V = st_8x32_s::template bytes_per_thread<T_V>();
    constexpr int bytes_per_memcpy_V = bytes_per_thread_V * NUM_THREADS;
    constexpr int memcpy_per_tile_V = KV_BLOCK_SIZE * D_V * sizeof(T_V) / bytes_per_memcpy_V; // =4

    uint32_t swizzled_offsets_K[memcpy_per_tile_K];
    uint32_t swizzled_offsets_V[memcpy_per_tile_V];
    G::prefill_swizzled_offsets<1, false>(k_smem[0], g.Kg, swizzled_offsets_K);
    G::prefill_swizzled_offsets<1, false>(v_smem[0], g.Vg, swizzled_offsets_V);

    // Load K0 to shared (6 rounds)
    G::load<1, false>(k_smem[0], g.Kg, {batch_idx, 0, head_idx_kv, 0}, swizzled_offsets_K, k_srsrc_base, k_base, k_lds_base_0);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    // Load Q to registers (D_QK=192), scale, transpose
    qo_tile<D_QK, float> q_reg_fl;
    load<1, qo_tile<D_QK, float>, _gl_QKVO>(q_reg_fl, g.Qg, {batch_idx, tile_idx, head_idx, 0});
    mul(q_reg_fl, q_reg_fl, TEMPERATURE_SCALE);
    copy(q_reg, q_reg_fl);
    transpose(q_reg_transposed, q_reg);

    // Load K1 (6 rounds) and V0 (4 rounds) to shared
    G::load<1, false>(k_smem[1], g.Kg, {batch_idx, 1, head_idx_kv, 0}, swizzled_offsets_K, k_srsrc_base, k_base, k_lds_base_1);
    G::load<1, false>(v_smem[0], g.Vg, {batch_idx, 0, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base, v_base, v_lds_base_0);
    // K0 already done. K1=6 rounds + V0=4 rounds = 10 issued. Wait until K0 load from smem is safe.
    load(k_reg, k_smem[0]);
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(4)"); // wait for 6 of 10 to complete, 4 remain (V0)
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    // QK0: standard k_reg + transpose + mma_AtB
    zero(att_block[0]);
    transpose(k_reg_transposed, k_reg);
    mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);
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

    // Load K1 into registers from smem[1] (already loaded to shared)
    load(k_reg, k_smem[1]);
    // Load K2 (6 rounds) and V1 (4 rounds) to shared
    G::load<1, false>(k_smem[0], g.Kg, {batch_idx, 2, head_idx_kv, 0}, swizzled_offsets_K, k_srsrc_base, k_base, k_lds_base_0);
    G::load<1, false>(v_smem[1], g.Vg, {batch_idx, 1, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base, v_base, v_lds_base_1);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(4)"); // K2=6 + V1=4 = 10 outstanding, wait for 6, keep 4 (V1)
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    int pending_scale = 0;

    // hot loop
    #pragma unroll
    for (int j = 3; j < max_num_tiles - 1; j += 2) {
        // Cluster 0: QK_odd + Finish softmax for QK_even
        zero(att_block[1]);
        transpose(k_reg_transposed, k_reg);
        mma_AtB(att_block[1], k_reg_transposed, q_reg_transposed, att_block[1]);
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
        asm volatile("s_waitcnt vmcnt(6)"); // K=6 outstanding new, V still finishing = keep 6
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 2: A*V accumulation + Partial softmax for QK_odd
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

        // Cluster 3: Load V to shared + Load K from smem to registers
        G::load<1, false>(v_smem[0], g.Vg, {batch_idx, j - 1, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base, v_base, v_lds_base_0);
        load(k_reg, k_smem[0]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(4)"); // V=4 outstanding, wait all
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 4: QK_even + Finish softmax for QK_odd
        zero(att_block[0]);
        transpose(k_reg_transposed, k_reg);
        mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);
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
        asm volatile("s_waitcnt vmcnt(6)"); // K=6 outstanding, keep 6 for pipelining
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

        // Cluster 7: Load V to shared + Load K from smem to registers
        G::load<1, false>(v_smem[1], g.Vg, {batch_idx, j, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base, v_base, v_lds_base_1);
        load(k_reg, k_smem[1]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(4)"); // V=4 outstanding
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    // Epilogue
    // Cluster 0: QK(n-3) + Finish softmax
    zero(att_block[1]);
    transpose(k_reg_transposed, k_reg);
    mma_AtB(att_block[1], k_reg_transposed, q_reg_transposed, att_block[1]);
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

    // Cluster 1: Load K(n-1) to shared + Load V(n-4) to registers
    G::load<1, false>(k_smem[1], g.Kg, {batch_idx, max_num_tiles - 1, head_idx_kv, 0}, swizzled_offsets_K, k_srsrc_base, k_base, k_lds_base_1);
    load(v_reg, v_smem[0]);
    if constexpr (causal) {
        const int kv_end_pos = (max_num_tiles - 2) * KV_BLOCK_SIZE;
        if (__builtin_expect(q_start_pos < kv_end_pos, 0)) {
            mask_kv_tile(att_block[1], tile_idx, max_num_tiles - 3, neg_inf_v, lane);
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(6)"); // K(n-1)=6 outstanding
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

    // Cluster 3: Load V(n-2) to shared + Load K(n-2) from smem to registers
    G::load<1, false>(v_smem[0], g.Vg, {batch_idx, max_num_tiles - 2, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base, v_base, v_lds_base_0);
    load(k_reg, k_smem[0]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(4)"); // V=4 outstanding
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 4: QK(n-2) + Finish softmax
    zero(att_block[0]);
    transpose(k_reg_transposed, k_reg);
    mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);
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

    // Cluster 5: Load V(n-3) to registers + causal mask
    load(v_reg, v_smem[1]);
    if constexpr (causal) {
        const int kv_end_pos = (max_num_tiles - 1) * KV_BLOCK_SIZE;
        if (__builtin_expect(q_start_pos < kv_end_pos, 1)) {
            mask_kv_tile(att_block[0], tile_idx, max_num_tiles - 2, neg_inf_v, lane);
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(0)"); // wait for V(n-2) to finish
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

    // Cluster 7: Load V(n-1) to shared + Load K(n-1) from smem to registers
    G::load<1, false>(v_smem[1], g.Vg, {batch_idx, max_num_tiles - 1, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base, v_base, v_lds_base_1);
    load(k_reg, k_smem[1]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(4)"); // V=4 outstanding
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 8: QK(n-1) + Finish softmax
    zero(att_block[1]);
    transpose(k_reg_transposed, k_reg);
    mma_AtB(att_block[1], k_reg_transposed, q_reg_transposed, att_block[1]);
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

    // Cluster 9: Load V(n-2) to registers + causal mask
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

    // Cluster 11: Load V(n-1) to registers
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

    // Output is D_V=128 wide
    qo_tile<D_V, float, row_l, rt_32x32_s> o_reg_transposed;
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
    m.doc() = "tk_kernel python module - asymmetric causal attention D_QK=192 D_V=128 (v2 NUM_WARPS=4)";
    py::bind_function<dispatch_micro>(m, "dispatch_micro",
        &attn_globals::Qg,
        &attn_globals::Kg,
        &attn_globals::Vg,
        &attn_globals::Og,
        &attn_globals::L_vec
    );
}
