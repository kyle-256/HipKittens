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

// ============ Causal mask helpers (ported from forward kernel) ============
// Branchless per-element mask using signed comparison.
// If rel < THR_X, replace x_ref with neg_inf; same for y_ref vs THR_Y.
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

// Causal mask for backward attention tile: col_l rt_16x16_s, Q_rows x K_cols.
//
// CDNA4 v_mfma_f32_16x16x32 accumulator layout (64 lanes, 4 VGPRs/lane):
//   K_col_in_tile = lane % 16
//   Q_row_base    = (lane / 16) * 4
//   data[0].x → Q_row_base+0,  data[0].y → Q_row_base+1
//   data[1].x → Q_row_base+2,  data[1].y → Q_row_base+3
//
// Causal condition: mask element to -inf when q_absolute < k_absolute.
//
// Uses branchless asm (mask_vec2_imm) with a shifted relative index:
//   rel_shifted = (q_pos + q_row_base + 3) - (k_pos + jj*16 + k_col)
//   data[0].x masked when rel_shifted < 3  (i.e., q_row_base+0 < k_col_abs)
//   data[0].y masked when rel_shifted < 2  (i.e., q_row_base+1 < k_col_abs)
//   data[1].x masked when rel_shifted < 1  (i.e., q_row_base+2 < k_col_abs)
//   data[1].y masked when rel_shifted < 0  (i.e., q_row_base+3 < k_col_abs)
template<ducks::rt::col_layout RT>
__device__ inline void mask_causal_bwd(RT &dst, int q_pos, int k_pos, uint32_t neg_inf_v, int lane) {
    const int q_row_base = (lane >> 4) << 2;   // (lane/16)*4
    const int k_col      = lane & 15;           // lane % 16

    #pragma unroll
    for (int jj = 0; jj < dst.width; jj++) {
        const int rel0 = (q_pos + q_row_base + 3) - (k_pos + jj * 16 + k_col);
        const uint32_t rel = static_cast<uint32_t>(rel0);

        auto& d0x = *reinterpret_cast<uint32_t*>(&dst.tiles[0][jj].data[0].x);
        auto& d0y = *reinterpret_cast<uint32_t*>(&dst.tiles[0][jj].data[0].y);
        auto& d1x = *reinterpret_cast<uint32_t*>(&dst.tiles[0][jj].data[1].x);
        auto& d1y = *reinterpret_cast<uint32_t*>(&dst.tiles[0][jj].data[1].y);

        mask_vec2_imm<3, 2>(rel, neg_inf_v, d0x, d0y);
        mask_vec2_imm<1, 0>(rel, neg_inf_v, d1x, d1y);
    }
}

// ============ Non-ART atomic bf16 packed add ============
// Stores dQ via global atomic add using buffer_atomic_pk_add_bf16.
// Uses the same addressing as the reference D=128 kernel:
//   lane_offset = laneid * 2 + warp_col_offset * 512
//   tile_offset = i * row_stride * base_tile_rows + j * 256
// This requires the dQ tensor to have its rows along the axis dimension
// with row_stride equal to the column count (i.e., compact row storage).
template<int axis, ducks::rt::row_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void atomic_add_bf16_tile(const GL &dst, const RT &src, const COORD &idx, int warp_col_offset) {
    using U = typename GL::dtype;
    static_assert(std::is_same_v<U, bf16>, "only bf16 global");
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    static_assert(std::is_same_v<T, float>, "source must be float");

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    int laneid = kittens::laneid();

    const uint32_t buffer_size = row_stride * RT::rows * sizeof(U);
    std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(dst_ptr);
    std::uint64_t as_u64 = static_cast<std::uint64_t>(as_int);
    buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);

    int lane_offset = laneid * 2 + warp_col_offset * 512;

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            const int tile_offset = i * row_stride * src.base_tile_rows + j * 256;

            float f0 = src.tiles[i][j].data[0].x;
            float f1 = src.tiles[i][j].data[0].y;
            float f2 = src.tiles[i][j].data[1].x;
            float f3 = src.tiles[i][j].data[1].y;

            uint32_t pk0, pk1;
            asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(pk0) : "v"(f1), "v"(f0));
            asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(pk1) : "v"(f3), "v"(f2));

            uint32_t byte_offset_0 = static_cast<uint32_t>((tile_offset + lane_offset) * sizeof(U));
            uint32_t byte_offset_1 = static_cast<uint32_t>((tile_offset + lane_offset + 128) * sizeof(U));

            asm volatile("buffer_atomic_pk_add_bf16 %0, %1, %2, 0 offen"
                : : "v"(pk0), "v"(byte_offset_0), "s"(*(const i32x4*)&br) : "memory");
            asm volatile("buffer_atomic_pk_add_bf16 %0, %1, %2, 0 offen"
                : : "v"(pk1), "v"(byte_offset_1), "s"(*(const i32x4*)&br) : "memory");
        }
    }
}

// ============ Globals ============
struct attn_bwd_combined_d192v128_globals {
  gl<bf16, -1, -1, -1, -1> Q, K, V;
  gl<bf16, -1, -1, -1, -1> dOg, dQg, dKg, dVg;
  gl<float, -1, -1, -1, -1> L_vec, delta_vec;
  hipStream_t stream;
  dim3 grid() { return dim3(ATTN_H_KV, (ATTN_N / BLOCK_SIZE_KV), ATTN_B); }
  dim3 block() { return dim3(NUM_THREADS); }
  size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

// ============ Kernel ============
__launch_bounds__(NUM_THREADS, 1)
__global__ void attend_bwd_combined_d192v128_ker(const attn_bwd_combined_d192v128_globals g) {

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

  // ---- Shared memory ----
  extern __shared__ alignment_dummy __shm[];
  shared_allocator al((int*)&__shm[0]);

  st_bf<BLOCK_SIZE_KV, D_QK, st_16x16_s> (&K_j_smem) = al.allocate<st_bf<BLOCK_SIZE_KV, D_QK, st_16x16_s>>();
  st_bf<SLICE_QO, D_QK, st_16x32_s> (&Q_i_smem)[2][2] = al.allocate<st_bf<SLICE_QO, D_QK, st_16x32_s>, 2, 2>();
  st_bf<SLICE_QO, D_V, st_16x32_s> (&dO_i_smem)[2][2] = al.allocate<st_bf<SLICE_QO, D_V, st_16x32_s>, 2, 2>();
  st_bf<BLOCK_SIZE_KV, DOT_SLICE_QO, st_16x16_swizzled_s> (&attn_i_smem) = al.allocate<st_bf<BLOCK_SIZE_KV, DOT_SLICE_QO, st_16x16_swizzled_s>>();
  sv_fl<STEP_QO> (&L_smem)[2] = al.allocate<sv_fl<STEP_QO>, 2>();
  sv_fl<STEP_QO> (&delta_smem)[2] = al.allocate<sv_fl<STEP_QO>, 2>();

  // ---- Register tiles ----
  //
  // MFMA shape constraints for bf16:
  //   mma_ABt:  D=rt_16x16_s, A=row_l rt_16x32_s (16x32), B=row_l rt_16x32_s (16x32)
  //   mma_AtB:  D=rt_32x32_s, A=col_l rt_16x32_s (16x32), B=col_l rt_16x32_s (16x32)
  //
  // For mma_ABt (S = Q @ K^T, dP = dO @ V^T):
  //   Q_i: 16xD_QK=16x192, row_l, rt_16x32_s
  //   K_j: 32xD_QK=32x192, row_l, rt_16x32_s
  //   V_j: 32xD_V=32x128, row_l, rt_16x32_s
  //   dO_i: 16xD_V=16x128, row_l, rt_16x32_s
  //   P_ij accumulator: 16x32, float, col_l, rt_16x16_s
  //   dP_ij accumulator: 16x32, float, col_l, rt_16x16_s
  //
  // For mma_AtB (dV += dO^T @ P, dK += Q^T @ dP):
  //   dO_i_col: 16xD_V=16x128, col_l, rt_16x32_s
  //   Q_i_col: 16xD_QK=16x192, col_l, rt_16x32_s
  //   P_bf16: 16x32, bf16, col_l, rt_16x32_s
  //   dP_bf16: 16x32, bf16, col_l, rt_16x32_s
  //   dV_j_T accumulator: D_Vx32=128x32, float, col_l, rt_32x32_s
  //   dK_j_T accumulator: D_QKx32=192x32, float, col_l, rt_32x32_s

  // Row-layout inputs for mma_ABt
  rt_bf<DOT_SLICE_QO, D_QK, row_l, rt_16x32_s> Q_i;
  rt_bf<WARP_SIZE_KV, D_QK, row_l, rt_16x32_s> K_j;
  rt_bf<WARP_SIZE_KV, D_V, row_l, rt_16x32_s> V_j;
  rt_bf<DOT_SLICE_QO, D_V, row_l, rt_16x32_s> dO_i;

  // Attention accumulators (col_l, rt_16x16_s) for mma_ABt output
  rt_fl<DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s> P_ij;
  rt_fl<DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s> dP_ij;

  // Col-layout inputs for mma_AtB
  rt_bf<DOT_SLICE_QO, D_V, col_l, rt_16x32_s> dO_i_col;
  rt_bf<DOT_SLICE_QO, D_QK, col_l, rt_16x32_s> Q_i_col;

  // Gradient accumulators (col_l, rt_32x32_s)
  rt_fl<D_QK, WARP_SIZE_KV, col_l, rt_32x32_s> dK_j_T;
  rt_fl<D_V, WARP_SIZE_KV, col_l, rt_32x32_s> dV_j_T;

  // For dQ: K_j_col (128x32 col), dP_col_T (128x16 col) with rt_32x16_4_s base
  rt_bf<BLOCK_SIZE_KV, 32, col_l, rt_32x16_4_s> K_j_col;
  rt_bf<BLOCK_SIZE_KV, DOT_SLICE_QO, col_l, rt_32x16_4_s> dP_ij_bf16_col_T;
  rt_fl<32, DOT_SLICE_QO, col_l, rt_16x16_s> dQ_i_T;

  // Initialize accumulators
  zero(dK_j_T);
  zero(dV_j_T);


  // Prefill swizzled offsets
  // Q_i_smem is st_bf<SLICE_QO, D_QK, st_16x32_s>, dO_i_smem is st_bf<SLICE_QO, D_V, st_16x32_s>
  // They have different column counts, so different numbers of load iterations.
  constexpr int bytes_per_thread_smem = st_16x32_s::template bytes_per_thread<bf16>();
  constexpr int memcpy_per_tile_Q  = SLICE_QO * D_QK * sizeof(bf16) / (bytes_per_thread_smem * NUM_THREADS);
  constexpr int memcpy_per_tile_dO = SLICE_QO * D_V  * sizeof(bf16) / (bytes_per_thread_smem * NUM_THREADS);
  uint32_t swizzled_offsets_Q[memcpy_per_tile_Q];
  G::prefill_swizzled_offsets<1, false>(Q_i_smem[0][0], g.Q, swizzled_offsets_Q);
  uint32_t swizzled_offsets_dO[memcpy_per_tile_dO];
  G::prefill_swizzled_offsets<1, false>(dO_i_smem[0][0], g.dOg, swizzled_offsets_dO);

  int tic = 0, toc = 1;

  // Load K to shared (128 x 192)
  G::load<1, false>(K_j_smem, g.K, {batch_idx, seq_idx, kv_head_idx, 0});

  // Load V to registers (32 x 128, each warp its slice)
  load<1>(V_j, g.V, {batch_idx, j, kv_head_idx, 0});

  // Load first Q, dO, L, delta
  load(L_smem[tic], g.L_vec, {batch_idx, first_q_head, 0, first_step});
  load(delta_smem[tic], g.delta_vec, {batch_idx, first_q_head, 0, first_step});
  G::load<1, false>(Q_i_smem[tic][0],  g.Q,   {batch_idx, first_step * 2 + 0, first_q_head, 0}, swizzled_offsets_Q);
  G::load<1, false>(dO_i_smem[tic][0], g.dOg, {batch_idx, first_step * 2 + 0, first_q_head, 0}, swizzled_offsets_dO);
  G::load<1, false>(Q_i_smem[tic][1],  g.Q,   {batch_idx, first_step * 2 + 1, first_q_head, 0}, swizzled_offsets_Q);
  G::load<1, false>(dO_i_smem[tic][1], g.dOg, {batch_idx, first_step * 2 + 1, first_q_head, 0}, swizzled_offsets_dO);
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();

  // ==== Main loop ====
  for (int i = 0; i < num_steps; ++i) {
    const int q_head_idx = i / num_steps_per_head + first_q_head;
    const int q_seq_idx = (i % num_steps_per_head) + first_step;
    const int q_pos_base = q_seq_idx * STEP_QO;

    const bool is_last = (i == num_steps - 1);
    const int next_q_head_idx = is_last ? 0 : ((i + 1) / num_steps_per_head + first_q_head);
    const int next_q_seq_idx  = is_last ? 0 : (((i + 1) % num_steps_per_head) + first_step);

    #pragma unroll 1
    for (int ds = 0; ds < 4; ds++) {
      const int q_pos = q_pos_base + ds * DOT_SLICE_QO;
      const int smem_half = ds / 2;
      const int smem_sub  = ds % 2;

      // Load Q_i (16x192), K_j (32x192) from shared
      load(Q_i, subtile_inplace<DOT_SLICE_QO, D_QK>(Q_i_smem[tic][smem_half], {smem_sub, 0}));
      load(K_j, subtile_inplace<WARP_SIZE_KV, D_QK>(K_j_smem, {warpid, 0}));

      // S_ij = Q_i @ K_j^T (mma_ABt with rt_16x32_s inputs, rt_16x16_s accum)
      zero(P_ij);
      mma_ABt(P_ij, Q_i, K_j, P_ij);
      mul(P_ij, P_ij, P_SCALE_FACTOR);

      // Subtract L_i per query row
      // P_ij is col_l 16x32 rt_16x16_s: height=1, width=2
      // sub_row takes col_vec: outer_dim=height=1, inner covers 16 rows
      {
        typename decltype(P_ij)::col_vec L_i_vec;
        load(L_i_vec, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], ds));
        mul(L_i_vec, L_i_vec, L_SCALE_FACTOR);
        sub_row(P_ij, P_ij, L_i_vec);
      }

      // Causal mask (ported from forward kernel's mask_kv_tile)
      if constexpr (causal) {
        if (q_pos + DOT_SLICE_QO <= k_pos) {
          neg_infty(P_ij);
        } else if (q_pos < k_pos + WARP_SIZE_KV) {
          const uint32_t neg_inf_v = 0xff800000u; // -inf in IEEE 754
          const int lane = kittens::laneid();
          mask_causal_bwd(P_ij, q_pos, k_pos, neg_inf_v, lane);
        }
      }

      // Clamp P_ij to <= 0 before exp2 to prevent overflow from
      // numerical precision mismatch between forward and backward MMA paths.
      // In exact arithmetic P_ij = log2(softmax_weight) <= 0 always.
      // The forward scales Q in float then converts to bf16, while backward
      // computes Q@K^T in bf16 then scales in float, causing tiny positive
      // overshoot that can exp2-overflow to inf and propagate as NaN.
      #pragma unroll
      for (int jj = 0; jj < P_ij.width; jj++) {
        #pragma unroll
        for (int kk = 0; kk < P_ij.packed_per_base_tile; kk++) {
          P_ij.tiles[0][jj].data[kk].x = __builtin_fminf(P_ij.tiles[0][jj].data[kk].x, 0.0f);
          P_ij.tiles[0][jj].data[kk].y = __builtin_fminf(P_ij.tiles[0][jj].data[kk].y, 0.0f);
        }
      }

      exp2(P_ij, P_ij);

      // dP_ij = dO_i @ V_j^T
      load(dO_i, subtile_inplace<DOT_SLICE_QO, D_V>(dO_i_smem[tic][smem_half], {smem_sub, 0}));
      zero(dP_ij);
      mma_ABt(dP_ij, dO_i, V_j, dP_ij);

      // dS_ij = P_ij * (dP_ij - delta_i)
      {
        typename decltype(dP_ij)::col_vec delta_vec;
        load(delta_vec, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], ds));
        sub_row(dP_ij, dP_ij, delta_vec);
      }
      mul(dP_ij, dP_ij, P_ij);

      // dV_j += P_ij^T @ dO_i
      load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D_V>(dO_i_smem[tic][smem_half], {smem_sub, 0}));
      {
        rt_bf<DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s> P_ij_bf16_small;
        copy(P_ij_bf16_small, P_ij);
        auto &P_ij_bf16_col = swap_layout_inplace<col_l, rt_16x32_s>(P_ij_bf16_small);
        mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      }

      // dQ_i += dS_ij @ K_j (store dS to shared, compute dQ = K^T @ dS)
      {
        rt_bf<DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s> dP_bf16_small;
        copy(dP_bf16_small, dP_ij);
        rt_bf<WARP_SIZE_KV, DOT_SLICE_QO, row_l, rt_16x16_s> dP_row;
        transpose(dP_row, dP_bf16_small);
        auto attn_sub = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
        store(attn_sub, dP_row);
      }
      __builtin_amdgcn_s_barrier();

      {
        load(dP_ij_bf16_col_T, attn_i_smem);

        // dQ pass 1: cols 0..127
        load(K_j_col, subtile_inplace<BLOCK_SIZE_KV, 32>(K_j_smem, {0, warpid}));
        if (i == 0 && ds == 0) {
          zero(dQ_i_T);
        }
        mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        {
          rt_fl<32, DOT_SLICE_QO, col_l, rt_16x16_s> dQ_scaled;
          copy(dQ_scaled, dQ_i_T);
          mul(dQ_scaled, dQ_scaled, dP_SCALE_FACTOR);
          rt_fl<DOT_SLICE_QO, 32, row_l, rt_16x16_s> dQ_row;
          transpose(dQ_row, dQ_scaled);
          atomic_add_bf16_tile<2>(g.dQg, dQ_row, {batch_idx, q_head_idx, q_seq_idx * 4 + ds, 0}, warpid);
        }

        // dQ pass 2: warps 0,1 handle cols 128..191
        if (warpid < 2) {
          load(K_j_col, subtile_inplace<BLOCK_SIZE_KV, 32>(K_j_smem, {0, warpid + 4}));
          rt_fl<32, DOT_SLICE_QO, col_l, rt_16x16_s> dQ_i_T_2;
          zero(dQ_i_T_2);
          mma_AtB(dQ_i_T_2, K_j_col, dP_ij_bf16_col_T, dQ_i_T_2);
          mul(dQ_i_T_2, dQ_i_T_2, dP_SCALE_FACTOR);
          rt_fl<DOT_SLICE_QO, 32, row_l, rt_16x16_s> dQ_row_2;
          transpose(dQ_row_2, dQ_i_T_2);
          atomic_add_bf16_tile<2>(g.dQg, dQ_row_2, {batch_idx, q_head_idx, q_seq_idx * 4 + ds, 0}, warpid + 4);
        }
        zero(dQ_i_T);
      }
      __builtin_amdgcn_s_barrier();

      // dK_j += dS_ij^T @ Q_i (AFTER dQ to avoid register interference)
      load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D_QK>(Q_i_smem[tic][smem_half], {smem_sub, 0}));
      {
        rt_bf<DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s> dP_ij_bf16_small;
        copy(dP_ij_bf16_small, dP_ij);
        auto &dP_ij_bf16_col = swap_layout_inplace<col_l, rt_16x32_s>(dP_ij_bf16_small);
        mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      }
      __builtin_amdgcn_s_barrier();
    } // end dot slice

    // Load next iteration directly into tic (no double-buffering -- simpler and correct)
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
  } // end step

  // ==== Epilogue: Store dV and dK ====
  {
    rt_fl<WARP_SIZE_KV, D_V, row_l, rt_32x32_s> dV_row;
    transpose(dV_row, dV_j_T);
    store<1>(g.dVg, dV_row, {batch_idx, j, kv_head_idx, 0});
  }
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();
  {
    mul(dK_j_T, dK_j_T, dP_SCALE_FACTOR);
    rt_fl<WARP_SIZE_KV, D_QK, row_l, rt_32x32_s> dK_row;
    transpose(dK_row, dK_j_T);
    store<1>(g.dKg, dK_row, {batch_idx, j, kv_head_idx, 0});
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
