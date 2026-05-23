// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// =============================================================================
// HipKittens FP8 grouped RCR — V2 (4-acc baseline, attempt #1: FUSED noinline)
// =============================================================================
// Direction #1: keep v1's 4-acc K-loop for perf parity; attack the 37 VGPR
// spill via targeted edits. Stage 1: pull FUSED_KTAIL block into a separate
// __noinline__ __device__ function so its ~10 locals (a/b base ptrs, SRDs,
// K_tail_base_bytes, b_group_byte_base) don't share VGPR space with the
// K-loop hot path.

#include "kernel_fp8_layouts.cpp"  // pull v1 helpers + dispatchers into ns


// =============================================================================
// PT-facing struct.
// =============================================================================
struct grouped_layout_globals_v2 {
    const void* a_ptr;
    const void* b_ptr;
    void*       c_ptr;
    int M_total, G_b, bN, bK, cM, cN;
    const float* sa_ptr;
    const float* sb_ptr;
    const int64_t* group_offs_ptr;
    hipStream_t stream;
    int G, group_m, m_per_group, num_xcds;
    int num_slots, chunk_size;
};


// =============================================================================
// FUSED K-tail block extracted as __noinline__ device function. Compiler is
// forced to give it its own register/stack frame, freeing the K-loop's 4-acc
// hot path from co-locating these ~10 locals into shared VGPRs.
// =============================================================================
__device__ __attribute__((noinline)) void rcr_v2_fused_ktail(
        const grouped_layout_globals& g,
        rt_fl<RBM, RBN, col_l, rt_16x16_s>& cA,
        rt_fl<RBM, RBN, col_l, rt_16x16_s>& cB,
        rt_fl<RBM, RBN, col_l, rt_16x16_s>& cC,
        rt_fl<RBM, RBN, col_l, rt_16x16_s>& cD,
        A_row_reg& a, B_row_reg& b0, B_row_reg& b1,
        int group_idx, int m_start_g, int br, int bc, int wm, int wn) {
    const int a_row_stride_bytes = static_cast<int>(g.a.template stride<2>()) * sizeof(*g.a.raw_ptr);
    const int laneid = kittens::laneid();
    const int row_lane = laneid % 16;
    const int k_lane_byte = (laneid / 16) * 32;
    constexpr int KREM = 64;
    static_assert(KREM == 64, "FUSED K_REM must be 64");
    const bool both_valid = (laneid < 32);
    constexpr uint32_t SENTINEL = 0xFFFF0000u;

    const fp8e4m3* a_base_ptr = (const fp8e4m3*)&g.a[{0, 0, 0, 0}];
    const fp8e4m3* b_base_ptr = (const fp8e4m3*)&g.b[{0, 0, 0, 0}];
    const int b_row_stride_bytes = g.b.template stride<2>();
    const uint32_t a_total_bytes =
        static_cast<uint32_t>(g.M_total) * static_cast<uint32_t>(a_row_stride_bytes);
    const uint32_t b_per_group_bytes =
        static_cast<uint32_t>(group_idx + 1) *
        static_cast<uint32_t>(g.n) * static_cast<uint32_t>(b_row_stride_bytes);
    i32x4 a_srsrc_kt = make_srsrc((const void*)a_base_ptr, a_total_bytes);
    i32x4 b_srsrc_kt = make_srsrc((const void*)b_base_ptr, b_per_group_bytes);
    const uint32_t K_tail_base_bytes = static_cast<uint32_t>(g.fast_k);
    const uint32_t b_group_byte_base =
        static_cast<uint32_t>(group_idx) *
        static_cast<uint32_t>(g.n) * static_cast<uint32_t>(b_row_stride_bytes);

    auto load_a_kt = [&](A_row_reg& A_tile, int slab) __attribute__((always_inline)) {
        const int M_warp_base = m_start_g + (br * 2 + slab) * HB + wm * RBM;
        #pragma unroll
        for (int h = 0; h < A_row_reg::height; ++h) {
            const int A_row_idx = M_warp_base + h * 16 + row_lane;
            const uint32_t v_base = static_cast<uint32_t>(
                A_row_idx * a_row_stride_bytes + K_tail_base_bytes + k_lane_byte);
            const uint32_t v_lo = both_valid ? v_base : SENTINEL;
            const uint32_t v_hi = both_valid ? (v_base + 16) : SENTINEL;
            __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(a_srsrc_kt, v_lo, 0, 0);
            __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(a_srsrc_kt, v_hi, 0, 0);
            *reinterpret_cast<__uint128_t*>(&A_tile.tiles[h][0].data[0]) = v0;
            *reinterpret_cast<__uint128_t*>(&A_tile.tiles[h][0].data[4]) = v1;
        }
    };
    auto load_b_kt = [&](B_row_reg& B_tile, int n_strip) __attribute__((always_inline)) {
        const int N_warp_base = (bc * 2 + n_strip) * HB + wn * RBN;
        #pragma unroll
        for (int h_b = 0; h_b < B_row_reg::height; ++h_b) {
            const int B_row_idx_in_group = N_warp_base + h_b * 16 + row_lane;
            const uint32_t v_base = b_group_byte_base + static_cast<uint32_t>(
                B_row_idx_in_group * b_row_stride_bytes + K_tail_base_bytes + k_lane_byte);
            const uint32_t v_lo = both_valid ? v_base : SENTINEL;
            const uint32_t v_hi = both_valid ? (v_base + 16) : SENTINEL;
            __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(b_srsrc_kt, v_lo, 0, 0);
            __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(b_srsrc_kt, v_hi, 0, 0);
            *reinterpret_cast<__uint128_t*>(&B_tile.tiles[h_b][0].data[0]) = v0;
            *reinterpret_cast<__uint128_t*>(&B_tile.tiles[h_b][0].data[4]) = v1;
        }
    };

    load_b_kt(b0, 0);
    load_b_kt(b1, 1);
    load_a_kt(a,  0);
    asm volatile("s_waitcnt vmcnt(0)");
    rcr_mma(cA, a, b0);
    rcr_mma(cB, a, b1);
    load_a_kt(a,  1);
    asm volatile("s_waitcnt vmcnt(0)");
    rcr_mma(cC, a, b0);
    rcr_mma(cD, a, b1);
}


// =============================================================================
// V2 kernel body — copy of v1's grouped_rcr_kernel_body with the FUSED_KTAIL
// block replaced by a call to the __noinline__ function above.
// =============================================================================
template<bool N_MASKED_STORE = false, bool FUSED_KTAIL = false>
__device__ __forceinline__
void grouped_rcr_kernel_body_v2(const grouped_layout_globals g) {
    using ST_rcr = ST_v2;
    __shared__ ST_rcr As[2][2];
    __shared__ ST_rcr Bs[2][2];
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];
    __shared__ int s_cum_tiles[MAX_G_PLUS_1];
    __shared__ int s_total_tiles;
    A_row_reg a;
    B_row_reg b0, b1;
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;

    const int slots_eff = gridDim.x;
    const int xcds_eff = g.num_xcds > 0 ? g.num_xcds : BLOCK_SWIZZLE_NUM_XCDS;
    const int chunk_size_eff = g.chunk_size > 0 ? g.chunk_size : 64;
    int pid = chiplet_transform_chunked(blockIdx.x, slots_eff, xcds_eff, chunk_size_eff);

    int wm = warpid() / WARPS_N;
    int wn = warpid() % WARPS_N;
    const int num_pid_n = g.bpc;
    const int ki_dyn   = g.ki;

    init_group_cumsum_smem<MAX_G_PLUS_1>(g, s_offs, s_cum_tiles, s_total_tiles,
                                         num_pid_n, /*M_BLOCK_DIV=*/BLOCK_SIZE);
    const int total_tiles = s_total_tiles;

    constexpr int bpt = ST_rcr::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * _NUM_THREADS;
    constexpr int mpt = ST_rcr::rows * ST_rcr::cols * sizeof(fp8e4m3) / bpm;
    uint32_t soA[mpt], soB[mpt];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

    for (int gt = pid; gt < total_tiles; gt += slots_eff) {
        int group_idx, m_start_g, M_g, bpr_g, br, bc;
        if (!dispatch_tile_in_group<MAX_G_PLUS_1>(
                gt, s_cum_tiles, s_offs, num_pid_n, g.group_m,
                /*M_BLOCK_DIV=*/BLOCK_SIZE,
                group_idx, m_start_g, M_g, bpr_g, br, bc)) continue;

        // V2 spill attempt #2: NO persistent a_gl_g / c_gl_g view copies.
        // Build a fresh shifted view per-load via macro so the gl<> object
        // lives in tight SSA scope; compiler should rematerialize ptr+row
        // bound from g.a / g.c (which are uniform across the tile loop)
        // each call, not maintaining it across the entire K-loop.
        constexpr int m_subtile_A = 0;
        constexpr int m_subtile_C = 0;
        const int m_limit = M_g;

        auto a_co = [&](int s, int k) -> coord<ST_rcr> { return {0, 0, m_subtile_A + s, k}; };
        auto b_co = [&](int s, int k) -> coord<ST_rcr> { return {0, group_idx, s, k}; };

        // Construct the shifted view on demand in each load call so it
        // doesn't take persistent VGPRs across the K-loop body. The fresh
        // gl<> is only the patch parameters — ptr offset + row bound.
        #define V2_GL_A_SHIFTED() ({ \
            auto _gl = g.a; auto _c = g.c; \
            patch_per_group_gl_view(_gl, _c, m_start_g, M_g); \
            _gl; })
        #define V2_GL_C_SHIFTED() ({ \
            auto _gl = g.a; auto _c = g.c; \
            patch_per_group_gl_view(_gl, _c, m_start_g, M_g); \
            _c; })

        auto load_a = [&](A_row_reg& dst, ST_rcr& tile, int wi) {
            auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
            load(dst, sub);
        };
        auto load_b = [&](B_row_reg& dst, ST_rcr& tile, int wi) {
            auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
            load(dst, sub);
        };
        auto b_tile = [&](int stage, int which) -> ST_rcr& { return Bs[stage][which]; };

        zero(cA); zero(cB); zero(cC); zero(cD);

        int tic = 0, toc = 1;
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 0), g.b, b_co(bc*2,   0), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0], V2_GL_A_SHIFTED(), a_co(br*2,   0), soA);
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 1), g.b, b_co(bc*2+1, 0), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[tic][1], V2_GL_A_SHIFTED(), a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(RCR_INIT0_VMCNT);
        __builtin_amdgcn_s_barrier();

        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(toc, 0), g.b, b_co(bc*2,   1), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[toc][0], V2_GL_A_SHIFTED(), a_co(br*2,   1), soA);
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(toc, 1), g.b, b_co(bc*2+1, 1), soB);

        TK_WAIT_VMCNT(RCR_INIT1_VMCNT);
        __builtin_amdgcn_s_barrier();

        TK_PRAGMA_UNROLL(RCR_MAIN_UNROLL)
        for (int k = 0; k < ki_dyn - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, b_tile(tic, 0), wn);
            load_a(a, As[tic][0], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], V2_GL_A_SHIFTED(), a_co(br*2+1, k+1), soA);
            TK_WAIT_LGKM(RCR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, b_tile(tic, 1), wn);
            rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 0), g.b, b_co(bc*2, k+2), soB);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0], V2_GL_A_SHIFTED(), a_co(br*2, k+2), soA);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 1), g.b, b_co(bc*2+1, k+2), soB);
            TK_WAIT_VMCNT(RCR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        // Epilog 1
        {
            load_b(b0, b_tile(tic, 0), wn);
            load_a(a, As[tic][0], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], V2_GL_A_SHIFTED(), a_co(br*2+1, ki_dyn-1), soA);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

            load_b(b1, b_tile(tic, 1), wn);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(RCR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, b_tile(toc, 0), wn);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();
            tic ^= 1; toc ^= 1;
        }

        // Epilog 2
        {
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, b_tile(tic, 1), wn);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1);
            rcr_mma_agpr_t<!FUSED_KTAIL>(cC, a, b0);
            rcr_mma_agpr_t<!FUSED_KTAIL>(cD, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        if constexpr (FUSED_KTAIL) {
            if (g.fast_k < g.k) {
                rcr_v2_fused_ktail(g, cA, cB, cC, cD, a, b0, b1,
                                   group_idx, m_start_g, br, bc, wm, wn);
            }
        }

        const float combined_scale = resolve_combined_scale_grp(g);

        if (wm == 0) __builtin_amdgcn_s_barrier();
        const int r0 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2+wm);
        const int r1 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2+WARPS_M+wm);
        const int c0 = __builtin_amdgcn_readfirstlane(bc*WARPS_N*2+wn);
        const int c1 = __builtin_amdgcn_readfirstlane(bc*WARPS_N*2+WARPS_N+wn);
        mul(cA, cA, combined_scale);
        store_c_tile_mn_masked_grouped(V2_GL_C_SHIFTED(), cA, /*group_idx=*/0, r0, c0, m_limit, g.n);
        mul(cB, cB, combined_scale);
        store_c_tile_mn_masked_grouped(V2_GL_C_SHIFTED(), cB, /*group_idx=*/0, r0, c1, m_limit, g.n);
        mul(cC, cC, combined_scale);
        store_c_tile_mn_masked_grouped(V2_GL_C_SHIFTED(), cC, /*group_idx=*/0, r1, c0, m_limit, g.n);
        mul(cD, cD, combined_scale);
        store_c_tile_mn_masked_grouped(V2_GL_C_SHIFTED(), cD, /*group_idx=*/0, r1, c1, m_limit, g.n);

        MAYBE_DRAIN_LGKM();
        __builtin_amdgcn_s_barrier();
    }
}


// =============================================================================
// V2 kernel template entry.
// =============================================================================
template<bool N_MASKED_STORE = false, bool FUSED_KTAIL = false>
__global__ __launch_bounds__(_NUM_THREADS, 1)
void grouped_gemm_fp8_kernel_v2(const grouped_layout_globals g) {
    grouped_rcr_kernel_body_v2<N_MASKED_STORE, FUSED_KTAIL>(g);
}


// =============================================================================
// Dispatcher.
// =============================================================================
inline void dispatch_grouped_rcr_v2(grouped_layout_globals_v2 g_in) {
    grouped_layout_globals g{
        _gl_fp8(reinterpret_cast<fp8e4m3*>(const_cast<void*>(g_in.a_ptr)),
                1, 1, g_in.M_total, g_in.bK),
        _gl_fp8(reinterpret_cast<fp8e4m3*>(const_cast<void*>(g_in.b_ptr)),
                1, g_in.G_b, g_in.bN, g_in.bK),
        _gl_bf16(reinterpret_cast<bf16*>(g_in.c_ptr), 1, 1, g_in.cM, g_in.cN),
        0.f, 0.f, g_in.sa_ptr, g_in.sb_ptr,
        g_in.group_offs_ptr, g_in.stream,
        g_in.G, 0, 0, 0, 0,
        g_in.group_m, g_in.num_xcds, 0,
        0, 0,
        g_in.m_per_group, g_in.num_slots, g_in.chunk_size, 0,
        0, nullptr, 0,
    };
    g.n       = static_cast<int>(g.c.cols());
    g.M_total = static_cast<int>(g.c.rows());
    g.k       = static_cast<int>(g.a.cols());
    g.fast_n  = (g.n / BLOCK_SIZE) * BLOCK_SIZE;
    g.fast_k  = (g.k / K_BLOCK)    * K_BLOCK;
    g.bpc     = kittens::ceil_div(g.n, BLOCK_SIZE);
    g.ki      = g.fast_k / K_BLOCK;
    if (g.bpc == 0 || g.ki == 0) return;

    const int K_rem = g.k - g.fast_k;
    const bool fuse_on = (K_rem == 64);
    const bool n_aligned = (g.bpc * BLOCK_SIZE == g.n);

    static const int slots_env = []() {
        if (const char* e = std::getenv("TK_RCR_V2_NUM_CUS")) {
            const int v = std::atoi(e);
            if (v > 0 && v <= NUM_CUS) return v;
        }
        return NUM_CUS;
    }();
    const int slots = (g.num_slots > 0 && g.num_slots <= NUM_CUS)
        ? g.num_slots : slots_env;

    if (fuse_on) {
        if (n_aligned)
            grouped_gemm_fp8_kernel_v2<false, true><<<dim3(slots), g.block(), 0, g.stream>>>(g);
        else
            grouped_gemm_fp8_kernel_v2<true,  true><<<dim3(slots), g.block(), 0, g.stream>>>(g);
    } else {
        if (n_aligned)
            grouped_gemm_fp8_kernel_v2<false, false><<<dim3(slots), g.block(), 0, g.stream>>>(g);
        else
            grouped_gemm_fp8_kernel_v2<true,  false><<<dim3(slots), g.block(), 0, g.stream>>>(g);
    }
}
