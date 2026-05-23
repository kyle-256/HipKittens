// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// =============================================================================
// HipKittens FP8 grouped RCR — V2 (mxfp8 PR #330 pattern, 8-warp adapted)
// =============================================================================
// Campaign D pivot 2026-05-23: scratch the v1-copy-with-tweaks approach;
// rewrite RCR from scratch using lessons from turbo MXFP8 GEMM (PR #330
// in turbo_gemm_mxfp8_kernel.h) adapted to the HipKittens 8-warp mandate.
//
// Key learnings carried in (memory cross-references):
//   - [[gfx950-8wave-va-cap]]: 8-warp V+A ≤ 256 dwords/lane.
//   - [[8w-tw-phase-split-bug-observation]]: turbo's `"n"(PIN_A)` literal
//     asm constraint causes DCE on pinned-slot writes. Use HK's
//     mfma1616128_agpr_inplace SSA-visible wrapper (already what
//     rcr_mma_agpr_t uses).
//   - mxfp8 trick #2 (outer per-group ptr hoist), #4 (two-step prologue),
//     #11 (sched_group_barrier batched), #14 (s_nop pre-store).
//
// Structural change vs v1: 2-acc (cA, cC) + outer N-strip loop.
//   v1 BN=256 keeps 4 acc (cA cB cC cD) live across the entire K-loop →
//   AGPR pressure 4×32 = 128 → V+A=384-spillsalt → 37-67 VGPR spill.
//   v2 only keeps 2 acc (cA = a0×b, cC = a1×b for the current N strip)
//   live, and runs the K-loop twice (n_strip ∈ {0, 1}). AGPR drops to
//   2×32 = 64; V budget can fit fragments + scratch unrolled. Target:
//   spill = 0 on BN=256.
//
// Trade-off: A is reloaded per N-strip pass (2× A HBM read per tile).
// A volume per tile = BLK_M × BK = 32 KB; B per tile = 64 KB. So A
// reload adds ~33% read volume but spill=0 + occupancy may compensate.
// Outer N loop also halves LDS B requirement (single N strip per pass).
//
// 8-warp layout (mandate, unchanged): WARPS_M=2 × WARPS_N=4 = 512 thread.
// Per-warp output area 64 (M) × 32 (N) within a 256×256 tile.

#include "kernel_fp8_layouts.cpp"  // pull v1 helpers into this namespace


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
// V2 RCR kernel body: 2-acc + outer N-strip + persistent + single-buf B LDS.
// Borrows v1's load helpers (rcr_8w_load_hoist), mma wrapper (rcr_mma_agpr_t),
// per-group view patch, masked store. Differences vs v1 hot path:
//   - cA, cC only (no cB, cD)
//   - outer N strip loop runs the K-loop twice
//   - Bs[2] (single N-strip ping-pong) instead of Bs[2][2]
//   - As[2] (single ping-pong, both M-strips loaded into one tile) — both
//     m-strips' 64 rows fit in one ST_v2 (128-row tile)
//   - s_nop + sched_barrier before store (trick #14)
// =============================================================================
template<bool N_MASKED_STORE = false>
__device__ __forceinline__
void grouped_rcr_kernel_body_v2(const grouped_layout_globals g) {
    using ST_rcr = ST_v2;
    __shared__ ST_rcr As[2][2]; // [ping][m_chunk]
    __shared__ ST_rcr Bs[2];    // [ping]
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];
    __shared__ int s_cum_tiles[MAX_G_PLUS_1];
    __shared__ int s_total_tiles;

    A_row_reg a0, a1;
    B_row_reg b;
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cC;

    const int slots_eff       = gridDim.x;
    const int xcds_eff        = g.num_xcds > 0 ? g.num_xcds : BLOCK_SWIZZLE_NUM_XCDS;
    const int chunk_size_eff  = g.chunk_size > 0 ? g.chunk_size : 64;
    int pid = chiplet_transform_chunked(
        blockIdx.x, slots_eff, xcds_eff, chunk_size_eff);

    const int wm = warpid() / WARPS_N;
    const int wn = warpid() % WARPS_N;
    const int num_pid_n = g.bpc;
    const int ki_dyn    = g.ki;

    init_group_cumsum_smem<MAX_G_PLUS_1>(g, s_offs, s_cum_tiles, s_total_tiles,
                                         num_pid_n, /*M_BLOCK_DIV=*/BLOCK_SIZE);
    const int total_tiles = s_total_tiles;

    constexpr int bpt = ST_rcr::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * _NUM_THREADS;
    constexpr int mpt = ST_rcr::rows * ST_rcr::cols * sizeof(fp8e4m3) / bpm;
    uint32_t soA[mpt], soB[mpt];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);
    G::prefill_swizzled_offsets(Bs[0],    g.b, soB);

    for (int gt = pid; gt < total_tiles; gt += slots_eff) {
        int group_idx, m_start_g, M_g, bpr_g, br, bc;
        if (!dispatch_tile_in_group<MAX_G_PLUS_1>(
                gt, s_cum_tiles, s_offs, num_pid_n, g.group_m,
                /*M_BLOCK_DIV=*/BLOCK_SIZE,
                group_idx, m_start_g, M_g, bpr_g, br, bc)) continue;

        auto a_gl_g = g.a;
        auto c_gl_g = g.c;
        patch_per_group_gl_view(a_gl_g, c_gl_g, m_start_g, M_g);
        constexpr int m_subtile_A = 0;
        constexpr int m_subtile_C = 0;
        const int m_limit = M_g;

        // Per-strip coord lambdas (use the current strip index from the outer loop)
        auto a_co = [&](int s, int k) -> coord<ST_rcr> {
            return {0, 0, m_subtile_A + s, k};
        };
        auto b_co = [&](int s, int k) -> coord<ST_rcr> {
            return {0, group_idx, s, k};
        };

        auto load_a_reg = [&](A_row_reg& dst, ST_rcr& tile, int wi) {
            auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
            load(dst, sub);
        };
        auto load_b_reg = [&](B_row_reg& dst, ST_rcr& tile, int wi) {
            auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
            load(dst, sub);
        };

        // -----------------------------------------------------------------
        // Outer N-strip loop — runs full K-loop twice (n_strip = 0, 1).
        // Each pass writes 64 (M per-warp) × 32 (N per-warp) output tiles
        // to cA + cC (m strip 0 and m strip 1 of the current n strip).
        // -----------------------------------------------------------------
        #pragma unroll 1
        for (int n_strip = 0; n_strip < 2; ++n_strip) {
            zero(cA); zero(cC);

            int tic = 0, toc = 1;

            // ---- Single-stage prologue: K=0 FULL into [0] ----
            rcr_8w_load_hoist<_NUM_THREADS>(Bs[0],    g.b,    b_co(bc*2 + n_strip, 0), soB);
            rcr_8w_load_hoist<_NUM_THREADS>(As[0][0], a_gl_g, a_co(br*2,     0), soA);
            rcr_8w_load_hoist<_NUM_THREADS>(As[0][1], a_gl_g, a_co(br*2 + 1, 0), soA);
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_s_barrier();

            // ---- 2-stage pipeline K-loop, double-issue ds_read ----
            tic = 0; toc = 1;
            #pragma unroll 1
            for (int k = 0; k < ki_dyn; ++k) {
                const bool has_next = (k + 1 < ki_dyn);
                if (has_next) {
                    rcr_8w_load_hoist<_NUM_THREADS>(Bs[toc],    g.b,    b_co(bc*2 + n_strip, k+1), soB);
                    rcr_8w_load_hoist<_NUM_THREADS>(As[toc][0], a_gl_g, a_co(br*2,     k+1), soA);
                    rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], a_gl_g, a_co(br*2 + 1, k+1), soA);
                }
                load_b_reg(b,  Bs[tic],    wn);
                load_a_reg(a0, As[tic][0], wm);
                load_a_reg(a1, As[tic][1], wm);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1);
                rcr_mma_agpr_t<true>(cA, a0, b);
                rcr_mma_agpr_t<true>(cC, a1, b);
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_sched_barrier(0);
                if (has_next) {
                    asm volatile("s_waitcnt vmcnt(0)");
                    __builtin_amdgcn_s_barrier();
                }
                tic ^= 1; toc ^= 1;
            }

            // ---- FUSED K-tail (K_rem == 64) — port of v1 FUSED_KTAIL block ----
            // K_rem must be exactly 64; main K-loop covers K=[0, fast_k);
            // tail covers K=[fast_k, k). Per-group byte-level SRD with sentinel
            // OOB masking, mirroring v1's pattern at kernel_fp8_layouts.cpp:2082+.
            if (g.fast_k < g.k) {
                const int laneid     = kittens::laneid();
                const int row_lane   = laneid % 16;
                const int k_lane_byte = (laneid / 16) * 32;
                constexpr uint32_t SENTINEL = 0xFFFF0000u;
                const bool both_valid = (laneid < 32);

                const fp8e4m3* a_base_ptr = (const fp8e4m3*)&g.a[{0, 0, 0, 0}];
                const fp8e4m3* b_base_ptr = (const fp8e4m3*)&g.b[{0, 0, 0, 0}];
                const int a_row_stride_bytes = static_cast<int>(g.a.template stride<2>())
                    * sizeof(*g.a.raw_ptr);
                const int b_row_stride_bytes = g.b.template stride<2>();
                const uint32_t a_total_bytes =
                    static_cast<uint32_t>(g.M_total) * static_cast<uint32_t>(a_row_stride_bytes);
                const uint32_t b_per_group_bytes =
                    static_cast<uint32_t>(group_idx + 1) *
                    static_cast<uint32_t>(g.n) *
                    static_cast<uint32_t>(b_row_stride_bytes);
                i32x4 a_srsrc_kt = make_srsrc((const void*)a_base_ptr, a_total_bytes);
                i32x4 b_srsrc_kt = make_srsrc((const void*)b_base_ptr, b_per_group_bytes);
                const uint32_t K_tail_base_bytes = static_cast<uint32_t>(g.fast_k);
                const uint32_t b_group_byte_base =
                    static_cast<uint32_t>(group_idx) *
                    static_cast<uint32_t>(g.n) *
                    static_cast<uint32_t>(b_row_stride_bytes);

                auto load_a_kt = [&](A_row_reg& A_tile, int slab)
                        __attribute__((always_inline)) {
                    const int M_warp_base = m_start_g + (br * 2 + slab) * HB + wm * RBM;
                    #pragma unroll
                    for (int h = 0; h < A_row_reg::height; ++h) {
                        const int A_row_idx = M_warp_base + h * 16 + row_lane;
                        const uint32_t v_base = static_cast<uint32_t>(
                            A_row_idx * a_row_stride_bytes +
                            K_tail_base_bytes + k_lane_byte);
                        const uint32_t v_lo = both_valid ? v_base : SENTINEL;
                        const uint32_t v_hi = both_valid ? (v_base + 16) : SENTINEL;
                        __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            a_srsrc_kt, v_lo, 0, 0);
                        __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            a_srsrc_kt, v_hi, 0, 0);
                        *reinterpret_cast<__uint128_t*>(&A_tile.tiles[h][0].data[0]) = v0;
                        *reinterpret_cast<__uint128_t*>(&A_tile.tiles[h][0].data[4]) = v1;
                    }
                };
                auto load_b_kt = [&](B_row_reg& B_tile, int n_strip_kt)
                        __attribute__((always_inline)) {
                    const int N_warp_base = (bc * 2 + n_strip_kt) * HB + wn * RBN;
                    #pragma unroll
                    for (int h_b = 0; h_b < B_row_reg::height; ++h_b) {
                        const int B_row_idx_in_group = N_warp_base + h_b * 16 + row_lane;
                        const uint32_t v_base = b_group_byte_base + static_cast<uint32_t>(
                            B_row_idx_in_group * b_row_stride_bytes +
                            K_tail_base_bytes + k_lane_byte);
                        const uint32_t v_lo = both_valid ? v_base : SENTINEL;
                        const uint32_t v_hi = both_valid ? (v_base + 16) : SENTINEL;
                        __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            b_srsrc_kt, v_lo, 0, 0);
                        __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            b_srsrc_kt, v_hi, 0, 0);
                        *reinterpret_cast<__uint128_t*>(&B_tile.tiles[h_b][0].data[0]) = v0;
                        *reinterpret_cast<__uint128_t*>(&B_tile.tiles[h_b][0].data[4]) = v1;
                    }
                };

                load_b_kt(b, n_strip);
                load_a_kt(a0, 0);
                load_a_kt(a1, 1);
                asm volatile("s_waitcnt vmcnt(0)");
                rcr_mma(cA, a0, b);
                rcr_mma(cC, a1, b);
            }

            // ---- Store the 2 accumulators of this n strip ----
            const float combined_scale = resolve_combined_scale_grp(g);
            asm volatile("s_nop 7\n s_nop 7\n s_nop 7\n s_nop 7");  // trick #14
            __builtin_amdgcn_sched_barrier(0);
            mul(cA, cA, combined_scale);
            mul(cC, cC, combined_scale);

            const int r0 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2 + wm);
            const int r1 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2 + WARPS_M + wm);
            const int col = __builtin_amdgcn_readfirstlane(
                bc*WARPS_N*2 + n_strip*WARPS_N + wn);

            store_c_tile_mn_masked_grouped(c_gl_g, cA, /*group_idx=*/0, r0, col, m_limit, g.n);
            store_c_tile_mn_masked_grouped(c_gl_g, cC, /*group_idx=*/0, r1, col, m_limit, g.n);

            // Drain before next n_strip pass restarts the K loop on same A
            asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
        }
    }
}


// =============================================================================
// V2 kernel template entry + dispatcher.
// =============================================================================
template<bool N_MASKED_STORE = false>
__global__ __launch_bounds__(_NUM_THREADS, 1)
void grouped_gemm_fp8_kernel_v2(const grouped_layout_globals g) {
    grouped_rcr_kernel_body_v2<N_MASKED_STORE>(g);
}


inline void dispatch_grouped_rcr_v2(grouped_layout_globals_v2 g_in) {
    grouped_layout_globals g{
        _gl_fp8(reinterpret_cast<fp8e4m3*>(const_cast<void*>(g_in.a_ptr)),
                1, 1, g_in.M_total, g_in.bK),
        _gl_fp8(reinterpret_cast<fp8e4m3*>(const_cast<void*>(g_in.b_ptr)),
                1, g_in.G_b, g_in.bN, g_in.bK),
        _gl_bf16(reinterpret_cast<bf16*>(g_in.c_ptr),
                 1, 1, g_in.cM, g_in.cN),
        0.f, 0.f,
        g_in.sa_ptr, g_in.sb_ptr,
        g_in.group_offs_ptr, g_in.stream,
        g_in.G, /*n*/0, /*k*/0, /*ki*/0, /*bpc*/0,
        g_in.group_m, g_in.num_xcds, /*M_total*/0,
        /*fast_n*/0, /*fast_k*/0,
        g_in.m_per_group, g_in.num_slots, g_in.chunk_size,
        /*fuse_ktail_off*/0,
        /*sk_split_n*/0, /*sk_partial_buf*/nullptr,
        /*bn_block*/0,
    };
    g.n       = static_cast<int>(g.c.cols());
    g.M_total = static_cast<int>(g.c.rows());
    g.k       = static_cast<int>(g.a.cols());
    g.fast_n  = (g.n / BLOCK_SIZE) * BLOCK_SIZE;
    g.fast_k  = (g.k / K_BLOCK)    * K_BLOCK;
    g.bpc     = kittens::ceil_div(g.n, BLOCK_SIZE);
    g.ki      = g.fast_k / K_BLOCK;

    if (g.bpc == 0 || g.ki == 0) return;

    static const int rcr_slots_env = []() {
        if (const char* e = std::getenv("TK_RCR_V2_NUM_CUS")) {
            const int v = std::atoi(e);
            if (v > 0 && v <= NUM_CUS) return v;
        }
        return NUM_CUS;
    }();
    const int rcr_slots = (g.num_slots > 0 && g.num_slots <= NUM_CUS)
        ? g.num_slots : rcr_slots_env;

    const bool n_aligned = (g.bpc * BLOCK_SIZE == g.n);
    if (n_aligned) {
        grouped_gemm_fp8_kernel_v2<false><<<dim3(rcr_slots), g.block(), 0, g.stream>>>(g);
    } else {
        grouped_gemm_fp8_kernel_v2<true><<<dim3(rcr_slots), g.block(), 0, g.stream>>>(g);
    }
}
