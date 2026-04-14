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

constexpr int STEP_QO = 64;
constexpr int BLOCK_SIZE_KV = 128;
constexpr int SLICE_QO = 32;
constexpr int DOT_SLICE_QO = 16;
constexpr int WARP_SIZE_KV = 32;
constexpr bool causal = true;

#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using G = kittens::group<NUM_WARPS>;
using namespace kittens;

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

// col_l rt_16x16_s: query = lane & 15, key_base = (lane >> 4) * 4
template<ducks::rt::col_layout RT>
__device__ inline void mask_causal_bwd(RT &dst, int q_pos, int k_pos, uint32_t neg_inf_v, int lane) {
    const int q_row      = lane & 15;
    const int k_row_base = (lane >> 4) << 2;
    #pragma unroll
    for (int jj = 0; jj < dst.width; jj++) {
        const int rel0 = (q_pos + q_row) - (k_pos + k_row_base + jj * 16);
        const uint32_t rel = static_cast<uint32_t>(rel0);
        auto& d0x = *reinterpret_cast<uint32_t*>(&dst.tiles[0][jj].data[0].x);
        auto& d0y = *reinterpret_cast<uint32_t*>(&dst.tiles[0][jj].data[0].y);
        auto& d1x = *reinterpret_cast<uint32_t*>(&dst.tiles[0][jj].data[1].x);
        auto& d1y = *reinterpret_cast<uint32_t*>(&dst.tiles[0][jj].data[1].y);
        mask_vec2_imm<0, 1>(rel, neg_inf_v, d0x, d0y);
        mask_vec2_imm<2, 3>(rel, neg_inf_v, d1x, d1y);
    }
}

// Atomic bf16 packed add for dQ.
// src is row_l rt_16x16_s float tile. For mfma_f32_16x16x32 row_l:
//   row = laneid % 16, col_base = (laneid / 16) * 4
//   data[0].x → (row, col_base+0), data[0].y → (row, col_base+1)
//   data[1].x → (row, col_base+2), data[1].y → (row, col_base+3)
template<int axis, ducks::rt::row_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void atomic_add_bf16_tile(const GL &dst, const RT &src, const COORD &idx, int warp_col_offset) {
    using U = typename GL::dtype;
    static_assert(std::is_same_v<U, bf16>, "only bf16 global");

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    const int laneid = kittens::laneid();
    const int lane_row = laneid % 16;
    const int lane_col_base = (laneid / 16) * 4;
    const int warp_col_base = warp_col_offset * src.width * src.base_tile_cols;

    const uint32_t buffer_size = static_cast<uint32_t>(row_stride * RT::rows * sizeof(U));
    buffer_resource br = make_buffer_resource(
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(dst_ptr)),
        buffer_size, 0x00020000);

    #pragma unroll
    for (int i = 0; i < src.height; i++) {
        #pragma unroll
        for (int j = 0; j < src.width; j++) {
            const int row = lane_row + i * src.base_tile_rows;
            const int col = lane_col_base + j * src.base_tile_cols + warp_col_base;

            float f0 = src.tiles[i][j].data[0].x;
            float f1 = src.tiles[i][j].data[0].y;
            float f2 = src.tiles[i][j].data[1].x;
            float f3 = src.tiles[i][j].data[1].y;

            uint32_t pk0, pk1;
            asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(pk0) : "v"(f1), "v"(f0));
            asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(pk1) : "v"(f3), "v"(f2));

            uint32_t off0 = static_cast<uint32_t>((row * row_stride + col) * sizeof(U));
            uint32_t off1 = static_cast<uint32_t>((row * row_stride + col + 2) * sizeof(U));

            asm volatile("buffer_atomic_pk_add_bf16 %0, %1, %2, 0 offen"
                : : "v"(pk0), "v"(off0), "s"(*(const i32x4*)&br) : "memory");
            asm volatile("buffer_atomic_pk_add_bf16 %0, %1, %2, 0 offen"
                : : "v"(pk1), "v"(off1), "s"(*(const i32x4*)&br) : "memory");
        }
    }
}

// Scalar store of row_l rt_32x32_s tile to bf16 global (bypasses buffer_resource OOB).
// row_l rt_32x32_s: row = lane & 31, col_base = (lane >> 5) * 4
template<typename RT>
__device__ inline void store_row_tile_scalar(bf16 *base, int base_off, int row_stride, const RT &src) {
    const int lane = kittens::laneid();
    const int row_lane = lane & 31;
    const int col_base = (lane >> 5) << 2;
    #pragma unroll
    for (int i = 0; i < src.height; i++) {
        #pragma unroll
        for (int j = 0; j < src.width; j++) {
            int row = row_lane + i * 32;
            int col = col_base + j * 32;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int co = col + ((k >> 1) << 3) + ((k & 1) << 1);
                float v0 = src.tiles[i][j].data[k].x;
                float v1 = src.tiles[i][j].data[k].y;
                base[base_off + row * row_stride + co] = __float2bfloat16(v0);
                base[base_off + row * row_stride + co + 1] = __float2bfloat16(v1);
            }
        }
    }
}

// Scalar store of col_l rt_32x32_s accumulator to bf16 global (bypasses gfx950 AGPR transpose bug).
// acc is [first_dim × WARP_SIZE_KV] col_l. Global is [B, N, H_KV, D] BNHD.
// For col_l rt_32x32_s: col (first dim) = lane & 31, row_base (second dim) = (lane >> 5) * 4
// data[k].x → row_base + ((k>>1)<<3) + ((k&1)<<1), data[k].y → row_base + ((k>>1)<<3) + ((k&1)<<1) + 1
template<typename ACC_T>
__device__ inline void store_col_accum_scalar(
    bf16 *base, int stride_0, int stride_1, int stride_2,
    int batch_idx, int kv_seq, int kv_head_idx,
    ACC_T &acc) {
    const int base_off = batch_idx * stride_0 + kv_seq * stride_1 + kv_head_idx * stride_2;
    const int lane = kittens::laneid();
    const int d_lane = lane & 31;
    const int row_base = (lane >> 5) << 2;
    #pragma unroll
    for (int i = 0; i < acc.height; i++) {
        const int d_base = i * 32 + d_lane;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            const int row_off = row_base + ((k >> 1) << 3) + ((k & 1) << 1);
            float val0 = acc.tiles[i][0].data[k].x;
            float val1 = acc.tiles[i][0].data[k].y;
            base[base_off + row_off * stride_1 + d_base] = __float2bfloat16(val0);
            base[base_off + (row_off + 1) * stride_1 + d_base] = __float2bfloat16(val1);
        }
    }
}

struct attn_bwd_combined_d192v128_globals {
  gl<bf16, -1, -1, -1, -1> Q, K, V;
  gl<bf16, -1, -1, -1, -1> dOg, dQg, dKg, dVg;
  gl<float, -1, -1, -1, -1> L_vec, delta_vec;
  hipStream_t stream;
  dim3 grid() { return dim3(ATTN_H_KV, (ATTN_N / BLOCK_SIZE_KV), ATTN_B); }
  dim3 block() { return dim3(NUM_THREADS); }
  size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__launch_bounds__(NUM_THREADS, 1)
__global__ __attribute__((amdgpu_num_vgpr(140))) void attend_bwd_combined_d192v128_ker(const attn_bwd_combined_d192v128_globals g) {

  const int kv_head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int batch_idx = blockIdx.z;
  const int first_q_head = kv_head_idx * GROUP_SIZE;
  const int warpid = kittens::warpid();
  const int j = seq_idx * NUM_WARPS + warpid;
  const int total_steps_per_head = ATTN_N / STEP_QO;
  const int j_min = seq_idx * NUM_WARPS;
  const int k_start_min = j_min * WARP_SIZE_KV;
  const int first_step = max(0, k_start_min / STEP_QO);
  const int num_steps_per_head = total_steps_per_head - first_step;
  const int num_steps = num_steps_per_head * GROUP_SIZE;
  const int k_pos = j * WARP_SIZE_KV;

  constexpr float L_SCALE_FACTOR = 1.44269504089f;
  constexpr float P_SCALE_FACTOR = 0.07216878364870322f * 1.44269504089f;
  constexpr float dP_SCALE_FACTOR = 0.07216878364870322f;

  extern __shared__ alignment_dummy __shm[];
  shared_allocator al((int*)&__shm[0]);

  st_bf<BLOCK_SIZE_KV, D_QK, st_16x16_s> (&K_j_smem) = al.allocate<st_bf<BLOCK_SIZE_KV, D_QK, st_16x16_s>>();
  st_bf<SLICE_QO, D_QK, st_16x32_s> (&Q_i_smem)[2][2] = al.allocate<st_bf<SLICE_QO, D_QK, st_16x32_s>, 2, 2>();
  st_bf<SLICE_QO, D_V, st_16x32_s> (&dO_i_smem)[2][2] = al.allocate<st_bf<SLICE_QO, D_V, st_16x32_s>, 2, 2>();
  st_bf<BLOCK_SIZE_KV, DOT_SLICE_QO, st_16x16_swizzled_s> (&attn_i_smem) = al.allocate<st_bf<BLOCK_SIZE_KV, DOT_SLICE_QO, st_16x16_swizzled_s>>();
  sv_fl<STEP_QO> (&L_smem)[2] = al.allocate<sv_fl<STEP_QO>, 2>();
  sv_fl<STEP_QO> (&delta_smem)[2] = al.allocate<sv_fl<STEP_QO>, 2>();

  rt_bf<DOT_SLICE_QO, D_QK, row_l, rt_16x32_s> Q_i;
  rt_bf<WARP_SIZE_KV, D_QK, row_l, rt_16x32_s> K_j;
  rt_bf<WARP_SIZE_KV, D_V, row_l, rt_16x32_s> V_j;
  rt_bf<DOT_SLICE_QO, D_V, row_l, rt_16x32_s> dO_i;

  rt_fl<DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s> P_ij;
  rt_fl<DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s> dP_ij;

  rt_bf<DOT_SLICE_QO, D_V, col_l, rt_16x32_4_s> dO_i_col;
  rt_bf<DOT_SLICE_QO, D_QK, col_l, rt_16x32_4_s> Q_i_col;
  rt_bf<DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x32_s> bf16_mma_scratch;

  rt_fl<D_V, WARP_SIZE_KV, col_l, rt_32x32_s> dV_j_T;
  rt_fl<D_QK, WARP_SIZE_KV, col_l, rt_32x32_s> dK_j_T;

  rt_bf<BLOCK_SIZE_KV, 32, col_l, rt_32x16_4_s> K_j_col;
  rt_bf<BLOCK_SIZE_KV, DOT_SLICE_QO, col_l, rt_32x16_4_s> dS_ij_bf16_col_T;
  rt_fl<32, DOT_SLICE_QO, col_l, rt_16x16_s> dQ_i_T;

  zero(dK_j_T);
  zero(dV_j_T);

  constexpr int bytes_per_thread_smem = st_16x32_s::template bytes_per_thread<bf16>();
  constexpr int memcpy_per_tile_Q  = SLICE_QO * D_QK * sizeof(bf16) / (bytes_per_thread_smem * NUM_THREADS);
  constexpr int memcpy_per_tile_dO = SLICE_QO * D_V  * sizeof(bf16) / (bytes_per_thread_smem * NUM_THREADS);
  uint32_t swizzled_offsets_Q[memcpy_per_tile_Q];
  G::prefill_swizzled_offsets<1, false>(Q_i_smem[0][0], g.Q, swizzled_offsets_Q);
  uint32_t swizzled_offsets_dO[memcpy_per_tile_dO];
  G::prefill_swizzled_offsets<1, false>(dO_i_smem[0][0], g.dOg, swizzled_offsets_dO);

  int tic = 0;

  G::load<1, false>(K_j_smem, g.K, {batch_idx, seq_idx, kv_head_idx, 0});
  load<1>(V_j, g.V, {batch_idx, j, kv_head_idx, 0});

  load(L_smem[tic], g.L_vec, {batch_idx, first_q_head, 0, first_step});
  load(delta_smem[tic], g.delta_vec, {batch_idx, first_q_head, 0, first_step});
  G::load<1, false>(Q_i_smem[tic][0],  g.Q,   {batch_idx, first_step * 2 + 0, first_q_head, 0}, swizzled_offsets_Q);
  G::load<1, false>(dO_i_smem[tic][0], g.dOg, {batch_idx, first_step * 2 + 0, first_q_head, 0}, swizzled_offsets_dO);
  G::load<1, false>(Q_i_smem[tic][1],  g.Q,   {batch_idx, first_step * 2 + 1, first_q_head, 0}, swizzled_offsets_Q);
  G::load<1, false>(dO_i_smem[tic][1], g.dOg, {batch_idx, first_step * 2 + 1, first_q_head, 0}, swizzled_offsets_dO);
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();

  for (int i = 0; i < num_steps; ++i) {
    const int q_head_idx = i / num_steps_per_head + first_q_head;
    const int q_seq_idx = (i % num_steps_per_head) + first_step;
    const int q_pos_base = q_seq_idx * STEP_QO;
    const bool is_last = (i == num_steps - 1);

    #pragma unroll 1
    for (int ds = 0; ds < 4; ds++) {
      const int q_pos = q_pos_base + ds * DOT_SLICE_QO;
      const int smem_half = ds / 2;
      const int smem_sub  = ds % 2;

      load(Q_i, subtile_inplace<DOT_SLICE_QO, D_QK>(Q_i_smem[tic][smem_half], {smem_sub, 0}));
      load(K_j, subtile_inplace<WARP_SIZE_KV, D_QK>(K_j_smem, {warpid, 0}));
      zero(P_ij);
      mma_ABt(P_ij, Q_i, K_j, P_ij);
      mul(P_ij, P_ij, P_SCALE_FACTOR);
      {
        typename decltype(P_ij)::col_vec L_i_vec;
        load(L_i_vec, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], ds));
        mul(L_i_vec, L_i_vec, L_SCALE_FACTOR);
        sub_row(P_ij, P_ij, L_i_vec);
      }
      if constexpr (causal) {
        if (q_pos + DOT_SLICE_QO <= k_pos) neg_infty(P_ij);
        else if (q_pos < k_pos + WARP_SIZE_KV) {
          const uint32_t neg_inf_v = 0xff800000u;
          mask_causal_bwd(P_ij, q_pos, k_pos, neg_inf_v, kittens::laneid());
        }
      }
      #pragma unroll
      for (int jj = 0; jj < P_ij.width; jj++)
        #pragma unroll
        for (int kk = 0; kk < P_ij.packed_per_base_tile; kk++) {
          P_ij.tiles[0][jj].data[kk].x = __builtin_fminf(P_ij.tiles[0][jj].data[kk].x, 0.0f);
          P_ij.tiles[0][jj].data[kk].y = __builtin_fminf(P_ij.tiles[0][jj].data[kk].y, 0.0f);
        }
      exp2(P_ij, P_ij);

      load(dO_i, subtile_inplace<DOT_SLICE_QO, D_V>(dO_i_smem[tic][smem_half], {smem_sub, 0}));
      zero(dP_ij);
      mma_ABt(dP_ij, dO_i, V_j, dP_ij);
      {
        typename decltype(dP_ij)::col_vec delta_vec;
        load(delta_vec, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], ds));
        sub_row(dP_ij, dP_ij, delta_vec);
      }
      mul(dP_ij, dP_ij, P_ij);

      rt_bf<DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s> P_ij_bf16;
      copy(P_ij_bf16, P_ij);
      rt_fl<DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s> dS_scaled;
      copy(dS_scaled, dP_ij);
      mul(dS_scaled, dS_scaled, dP_SCALE_FACTOR);
      rt_bf<DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s> dS_ij_bf16;
      copy(dS_ij_bf16, dP_ij);
      rt_bf<DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s> dS_scaled_bf16;
      copy(dS_scaled_bf16, dS_scaled);

      load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D_V>(dO_i_smem[tic][smem_half], {smem_sub, 0}));
      swap_layout(bf16_mma_scratch, P_ij_bf16);
      { auto &P_col_4s = *reinterpret_cast<rt_bf<DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x32_4_s>*>(&bf16_mma_scratch);
        mma_AtB(dV_j_T, dO_i_col, P_col_4s, dV_j_T); }

      // dQ skipped for now
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_s_barrier();

      load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D_QK>(Q_i_smem[tic][smem_half], {smem_sub, 0}));
      swap_layout(bf16_mma_scratch, dS_scaled_bf16);
      { auto &dS_col_4s = *reinterpret_cast<rt_bf<DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x32_4_s>*>(&bf16_mma_scratch);
        mma_AtB(dK_j_T, Q_i_col, dS_col_4s, dK_j_T); }
      __builtin_amdgcn_s_barrier();
    }

    if (!is_last) {
      const int nn_q_head_idx = (i + 1) / num_steps_per_head + first_q_head;
      const int nn_q_seq_idx  = ((i + 1) % num_steps_per_head) + first_step;
      G::load<1, false>(Q_i_smem[tic][0],  g.Q,   {batch_idx, nn_q_seq_idx * 2, nn_q_head_idx, 0}, swizzled_offsets_Q);
      G::load<1, false>(dO_i_smem[tic][0], g.dOg, {batch_idx, nn_q_seq_idx * 2, nn_q_head_idx, 0}, swizzled_offsets_dO);
      G::load<1, false>(Q_i_smem[tic][1],  g.Q,   {batch_idx, nn_q_seq_idx * 2 + 1, nn_q_head_idx, 0}, swizzled_offsets_Q);
      G::load<1, false>(dO_i_smem[tic][1], g.dOg, {batch_idx, nn_q_seq_idx * 2 + 1, nn_q_head_idx, 0}, swizzled_offsets_dO);
      load(L_smem[tic], g.L_vec, {batch_idx, nn_q_head_idx, 0, nn_q_seq_idx});
      load(delta_smem[tic], g.delta_vec, {batch_idx, nn_q_head_idx, 0, nn_q_seq_idx});
      __builtin_amdgcn_s_waitcnt(0);
      __builtin_amdgcn_s_barrier();
    }
  }

  // Epilogue: store dV and dK via scalar stores (bypasses buffer_resource OOB issue)
  {
    rt_fl<WARP_SIZE_KV, D_V, row_l, rt_32x32_s> dV_row;
    transpose(dV_row, dV_j_T);
    bf16 *dV_base = reinterpret_cast<bf16*>(g.dVg.raw_ptr);
    const int dV_s0 = g.dVg.template stride<0>(), dV_s1 = g.dVg.template stride<1>(), dV_s2 = g.dVg.template stride<2>();
    const int dV_off = batch_idx * dV_s0 + j * WARP_SIZE_KV * dV_s1 + kv_head_idx * dV_s2;
    store_row_tile_scalar(dV_base, dV_off, dV_s1, dV_row);
  }
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();
  // dK_j_T already has dP_SCALE_FACTOR baked in (pre-scaled dS)
  {
    rt_fl<WARP_SIZE_KV, D_QK, row_l, rt_32x32_s> dK_row;
    transpose(dK_row, dK_j_T);
    bf16 *dK_base = reinterpret_cast<bf16*>(g.dKg.raw_ptr);
    const int dK_s0 = g.dKg.template stride<0>(), dK_s1 = g.dKg.template stride<1>(), dK_s2 = g.dKg.template stride<2>();
    const int dK_off = batch_idx * dK_s0 + j * WARP_SIZE_KV * dK_s1 + kv_head_idx * dK_s2;
    store_row_tile_scalar(dK_base, dK_off, dK_s1, dK_row);
  }
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
