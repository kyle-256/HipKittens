#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

#ifndef M_DIM
#define M_DIM 8192
#endif
#ifndef K_DIM
#define K_DIM 8192
#endif
#ifndef N_DIM
#define N_DIM 8192
#endif

// Architecture: occupancy 1, 128KB LDS, double-buffered, 4 warps
// Matches gluon a4w4 kernel structure for ~5300 TFLOPS FP4

constexpr int BLK = 256;
constexpr int BK  = 128; // bytes (= 256 FP4 elements)
constexpr int WARPS_M = 2, WARPS_N = 2;
constexpr int _NUM_WARPS   = WARPS_M * WARPS_N; // 4
constexpr int _NUM_THREADS = _NUM_WARPS * WARP_THREADS; // 256
constexpr int HB = BLK / 2; // 128 = half-block for M-halves and N-halves

// Per-warp tile dimensions (M-half × N-half)
constexpr int RBM = HB / WARPS_M; // 64
constexpr int RBN = HB / WARPS_N; // 64

static_assert(BLK == 256 && BK == 128, "V2 requires BLK=256, BK=128");
static_assert(RBM == 64 && RBN == 64, "Expected 64x64 per-warp tile");

constexpr int K_BYTES = K_DIM / 2;
constexpr int k_byte_iters = K_BYTES / BK;

// Tile types (same swizzled layout as V1)
using ST_tile = st_fp8e4m3<HB, BK, st_16x128_s>; // 128×128 = 16 KB per tile

// Register tile types for MFMA operands
using A_row_reg = rt_fp8e4m3<RBM, BK, row_l, rt_16x128_s>;
using B_row_reg = rt_fp8e4m3<RBN, BK, row_l, rt_16x128_s>;
using RT_C = rt_fl<RBM, RBN, col_l, rt_16x16_s>;

using G = kittens::group<_NUM_WARPS>;
using _gl_fp4   = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_scale = gl<fp8e8m0, -1, -1, -1, -1>;
using _gl_bf16  = gl<bf16, -1, -1, -1, -1>;

struct v2_globals {
    _gl_fp4 a, b;
    _gl_scale a_scale, b_scale;
    _gl_bf16 c;
    float scale = 1.0f;
    hipStream_t stream = nullptr;
    int m = 0, n = 0, k = 0;
};

using fp4_intx8_t   = int __attribute__((__vector_size__(8 * sizeof(int))));
using fp4_floatx4_t = float __attribute__((__vector_size__(4 * sizeof(float))));

// ── MFMA helpers (same as V1) ──

template<int OPSEL_A, int OPSEL_B>
__device__ __forceinline__ void fp4_mfma_scale_inplace(
    fp4_floatx4_t& d,
    const fp4_intx8_t& a,
    const fp4_intx8_t& b,
    fp8e8m0_4 scale_a,
    fp8e8m0_4 scale_b)
{
    d = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a, b, d, 4, 4,
        OPSEL_A, scale_a,
        OPSEL_B, scale_b
    );
}

__device__ __forceinline__ fp4_intx8_t fp4_upper_half(const fp4_intx8_t& src) {
    fp4_intx8_t dst;
    dst[0] = src[4]; dst[1] = src[5]; dst[2] = src[6]; dst[3] = src[7];
    dst[4] = 0;      dst[5] = 0;      dst[6] = 0;      dst[7] = 0;
    return dst;
}

template<ducks::rt::row_layout RT, ducks::st::all ST>
__device__ __forceinline__ void fp4_load_st_to_rt(RT &dst, const ST &src) {
    static_assert(RT::rows == ST::rows && RT::cols == ST::cols);
    using T2 = typename RT::dtype;
    using T = typename base_types::packing<T2>::unpacked_type;
    using U = typename ST::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    constexpr int packing = base_types::packing<typename RT::dtype>::num();
    static_assert(std::is_same_v<T, U>);
    const int laneid = kittens::laneid();
    const int row_offset = laneid % dst.base_tile_rows;
    const int col_offset = dst.base_tile_stride * (laneid / dst.base_tile_rows);
    const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);
    constexpr int reg_sub_row = ST::underlying_subtile_cols / RT::base_tile_cols;
    constexpr int reg_sub_col = ST::underlying_subtile_rows / RT::base_tile_rows;
    #pragma unroll
    for (int k = 0; k < RT::base_tile_num_strides; k++) {
        #pragma unroll
        for (int i = 0; i < reg_sub_col; i++) {
            #pragma unroll
            for (int j = 0; j < reg_sub_row; j++) {
                const int row = i * RT::base_tile_rows + row_offset;
                const int col = j * RT::base_tile_cols + col_offset +
                    k * RT::base_tile_elements_per_stride_group;
                const uint32_t offset = sizeof(U) * (src_ptr + row * ST::underlying_subtile_cols + col);
                const uint32_t addr = offset ^ (((offset % (16 * 128)) >> 8) << 4);
                const int idx = k * RT::base_tile_stride / packing;
                #pragma unroll
                for (int ii = 0; ii < ST::subtiles_per_col; ii++) {
                    #pragma unroll
                    for (int jj = 0; jj < ST::subtiles_per_row; jj++) {
                        const int sid = ii * ST::underlying_subtiles_per_row + jj;
                        const int soff = sid * ST::underlying_subtile_bytes;
                        const int rr = ii * reg_sub_col + i;
                        const int rc = jj * reg_sub_row + j;
                        asm volatile(
                            "ds_read_b128 %0, %1 offset:%2\n"
                            : "=v"(*reinterpret_cast<float4*>(
                                  &dst.tiles[rr][rc].data[idx]))
                            : "v"(addr), "i"(soff)
                            : "memory"
                        );
                    }
                }
            }
        }
    }
}

template<ducks::rt::row_layout RT>
__device__ __forceinline__ fp4_intx8_t fp4_extract_tile(const RT &src, int tile_row) {
    return *reinterpret_cast<const fp4_intx8_t*>(&src.tiles[tile_row][0].data[0]);
}

// ── Scale helpers (SRD-based buffer_load) ──

__device__ __forceinline__ const uint8_t* preshuffled_scale_row_base_ptr(
    const _gl_scale& src, int row_group)
{
    return reinterpret_cast<const uint8_t*>(src.raw_ptr + src.idx(coord<>(row_group, 0)));
}

__device__ __forceinline__ i32x4 make_scale_srd(const uint8_t* ptr) {
    i32x4 srd = std::bit_cast<i32x4>(
        make_buffer_resource(
            static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(ptr)),
            0xFFFFFFFFu, 0x00110000u));
    srd[0] = __builtin_amdgcn_readfirstlane(srd[0]);
    srd[1] = __builtin_amdgcn_readfirstlane(srd[1]);
    srd[2] = __builtin_amdgcn_readfirstlane(srd[2]);
    srd[3] = __builtin_amdgcn_readfirstlane(srd[3]);
    return srd;
}

__device__ __forceinline__ fp8e8m0_4 load_pq_scale_srd(
    i32x4 srsrc, uint32_t voffset, uint32_t soffset)
{
    return std::bit_cast<fp8e8m0_4>(
        llvm_amdgcn_raw_buffer_load_b32(srsrc, voffset, soffset, 0));
}

__device__ __forceinline__ fp8e8m0_4 remap_phase(fp8e8m0_4 src, int k_phase) {
    return std::bit_cast<fp8e8m0_4>(
        std::bit_cast<uint32_t>(src) >> (static_cast<uint32_t>(k_phase & 1) << 4));
}

// ── Per-warp accumulator (4 subtile-rows × 4 subtile-cols = 16 tiles × 4 floats) ──

struct alignas(16) fp4_acc_v2 {
    fp4_floatx4_t regs[(RBM / 16) * (RBN / 16)]; // 4×4 = 16 tiles
};

template<bool UPPER>
__device__ __forceinline__ void fp4_mma_v2(
    fp4_acc_v2& acc,
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_scales[], const fp8e8m0_4 b_scales[],
    int k_phase)
{
    constexpr int B_COLS = RBN / 16; // 4

    auto do_mfma = [&]<int AR, int BC>() {
        fp4_intx8_t a_data = A[AR], b_data = B[BC];
        if constexpr (UPPER) { a_data = fp4_upper_half(a_data); b_data = fp4_upper_half(b_data); }
        fp8e8m0_4 a_sc = remap_phase(a_scales[AR / 2], k_phase);
        fp8e8m0_4 b_sc = remap_phase(b_scales[BC / 2], k_phase);
        acc.regs[AR * B_COLS + BC] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_data, b_data, acc.regs[AR * B_COLS + BC], 4, 4,
            AR & 1, a_sc, BC & 1, b_sc);
    };
    do_mfma.template operator()<0,0>(); do_mfma.template operator()<0,1>();
    do_mfma.template operator()<0,2>(); do_mfma.template operator()<0,3>();
    __builtin_amdgcn_sched_barrier(0);
    do_mfma.template operator()<1,0>(); do_mfma.template operator()<1,1>();
    do_mfma.template operator()<1,2>(); do_mfma.template operator()<1,3>();
    __builtin_amdgcn_sched_barrier(0);
    do_mfma.template operator()<2,0>(); do_mfma.template operator()<2,1>();
    do_mfma.template operator()<2,2>(); do_mfma.template operator()<2,3>();
    __builtin_amdgcn_sched_barrier(0);
    do_mfma.template operator()<3,0>(); do_mfma.template operator()<3,1>();
    do_mfma.template operator()<3,2>(); do_mfma.template operator()<3,3>();
    __builtin_amdgcn_sched_barrier(0);
}

// ── Main kernel ──

__global__ __launch_bounds__(_NUM_THREADS, 1)
void mxfp4_rcr_v2_kernel(const v2_globals g) {
    static_assert(K_BYTES % BK == 0 && N_DIM % BLK == 0 && M_DIM % BLK == 0);

    constexpr int bpc = N_DIM / BLK;

    // ── Double-buffered LDS: 8 tiles × 16KB = 128KB ──
    __shared__ ST_tile A0_db[2]; // M-half 0, double-buffered
    __shared__ ST_tile A1_db[2]; // M-half 1, double-buffered
    __shared__ ST_tile Bl_db[2]; // B-left (N-half 0), double-buffered
    __shared__ ST_tile Br_db[2]; // B-right (N-half 1), double-buffered

    const int bid = blockIdx.x;
    const int br = bid / bpc, bc = bid % bpc;
    const int wm = warpid() / WARPS_N, wn = warpid() % WARPS_N;

    // ── Tile load offsets (computed once) ──
    constexpr int bpt = ST_tile::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * _NUM_THREADS;
    constexpr int mpt = HB * BK * sizeof(fp8e4m3) / bpm;
    uint32_t so_a[mpt], so_b[mpt];
    G::prefill_swizzled_offsets(A0_db[0], g.a, so_a);
    G::prefill_swizzled_offsets(Bl_db[0], g.b, so_b);

    // ── Scale SRDs (precomputed, in SGPRs) ──
    constexpr int a_packs = RBM / 32; // 2
    constexpr int b_packs = RBN / 32; // 2
    const int lane_nonk = kittens::laneid() % 16;
    const int lane_kblk = kittens::laneid() / 16;
    const uint32_t lane_soff =
        (static_cast<uint32_t>(lane_kblk) << 6) |
        (static_cast<uint32_t>(lane_nonk) << 2);

    i32x4 a0_srd[a_packs], a1_srd[a_packs];
    i32x4 bl_srd[b_packs], br_srd[b_packs];
    #pragma unroll
    for (int p = 0; p < a_packs; ++p) {
        a0_srd[p] = make_scale_srd(preshuffled_scale_row_base_ptr(
            g.a_scale, (br * BLK + 0 * HB + wm * RBM + p * 32) >> 5));
        a1_srd[p] = make_scale_srd(preshuffled_scale_row_base_ptr(
            g.a_scale, (br * BLK + 1 * HB + wm * RBM + p * 32) >> 5));
    }
    #pragma unroll
    for (int p = 0; p < b_packs; ++p) {
        bl_srd[p] = make_scale_srd(preshuffled_scale_row_base_ptr(
            g.b_scale, (bc * BLK + 0 * HB + wn * RBN + p * 32) >> 5));
        br_srd[p] = make_scale_srd(preshuffled_scale_row_base_ptr(
            g.b_scale, (bc * BLK + 1 * HB + wn * RBN + p * 32) >> 5));
    }

    // ── Accumulators: 4 quadrants (M-half × N-half) ──
    fp4_acc_v2 acc_A0Bl{}, acc_A0Br{}, acc_A1Bl{}, acc_A1Br{};

    // ── Register tile buffers ──
    A_row_reg a_rt;
    B_row_reg b_rt;

    // ── Lambda: load all 4 tiles for K-iteration bt into double-buffer slot db ──
    auto load_tiles = [&](int bt, int db) {
        G::load(A0_db[db], g.a, {0, 0, br * 2,     bt}, so_a);
        G::load(A1_db[db], g.a, {0, 0, br * 2 + 1, bt}, so_a);
        G::load(Bl_db[db], g.b, {0, 0, bc * 2,     bt}, so_b);
        G::load(Br_db[db], g.b, {0, 0, bc * 2 + 1, bt}, so_b);
    };

    // 16 buffer_load_dwordx4 per prefetch batch (4 tiles × 4 loads/tile/thread)
    constexpr int PREFETCH_VMCNT = 16;

    // ══════════════════ Prologue ══════════════════
    load_tiles(0, 0);
    if (k_byte_iters > 1) load_tiles(1, 1);

    // Prefetch scales for iteration 0
    fp8e8m0_4 pf_a0[a_packs], pf_a1[a_packs], pf_bl[b_packs], pf_br[b_packs];
    {
        const uint32_t soff0 = 0;
        #pragma unroll
        for (int p = 0; p < a_packs; ++p) {
            pf_a0[p] = load_pq_scale_srd(a0_srd[p], lane_soff, soff0);
            pf_a1[p] = load_pq_scale_srd(a1_srd[p], lane_soff, soff0);
        }
        #pragma unroll
        for (int p = 0; p < b_packs; ++p) {
            pf_bl[p] = load_pq_scale_srd(bl_srd[p], lane_soff, soff0);
            pf_br[p] = load_pq_scale_srd(br_srd[p], lane_soff, soff0);
        }
    }

    // ══════════════════ Main loop ══════════════════
    for (int bt = 0; bt < k_byte_iters; ++bt) {
        const int cur = bt & 1;

        asm volatile("s_waitcnt vmcnt(16)");
        __builtin_amdgcn_s_barrier();

        // Use prefetched scales (already in VGPRs from prev iteration)
        fp8e8m0_4 a0_raw[a_packs], a1_raw[a_packs];
        fp8e8m0_4 bl_raw[b_packs], br_raw[b_packs];
        #pragma unroll
        for (int p = 0; p < a_packs; ++p) { a0_raw[p] = pf_a0[p]; a1_raw[p] = pf_a1[p]; }
        #pragma unroll
        for (int p = 0; p < b_packs; ++p) { bl_raw[p] = pf_bl[p]; br_raw[p] = pf_br[p]; }

        // LDS reads for tiles
        A_row_reg a0_rt, a1_rt;
        B_row_reg bl_rt2, br_rt2;
        fp4_load_st_to_rt(a0_rt, kittens::subtile_inplace<RBM, BK>(A0_db[cur], {wm, 0}));
        fp4_load_st_to_rt(a1_rt, kittens::subtile_inplace<RBM, BK>(A1_db[cur], {wm, 0}));
        fp4_load_st_to_rt(bl_rt2, kittens::subtile_inplace<RBN, BK>(Bl_db[cur], {wn, 0}));
        fp4_load_st_to_rt(br_rt2, kittens::subtile_inplace<RBN, BK>(Br_db[cur], {wn, 0}));

        // Prefetch scales for bt+1 (overlaps with ds_reads above)
        {
            const uint32_t next_soff = static_cast<uint32_t>(bt + 1 < k_byte_iters ? bt + 1 : bt) << 8;
            #pragma unroll
            for (int p = 0; p < a_packs; ++p) {
                pf_a0[p] = load_pq_scale_srd(a0_srd[p], lane_soff, next_soff);
                pf_a1[p] = load_pq_scale_srd(a1_srd[p], lane_soff, next_soff);
            }
            #pragma unroll
            for (int p = 0; p < b_packs; ++p) {
                pf_bl[p] = load_pq_scale_srd(bl_srd[p], lane_soff, next_soff);
                pf_br[p] = load_pq_scale_srd(br_srd[p], lane_soff, next_soff);
            }
        }

        asm volatile("s_waitcnt lgkmcnt(0) vmcnt(8)");
        __builtin_amdgcn_sched_barrier(0);

        fp4_intx8_t tA0[4], tA1[4], tBl[4], tBr[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            tA0[i] = fp4_extract_tile(a0_rt, i);
            tA1[i] = fp4_extract_tile(a1_rt, i);
            tBl[i] = fp4_extract_tile(bl_rt2, i);
            tBr[i] = fp4_extract_tile(br_rt2, i);
        }

        // ── ALL Phase 0 (64 MFMAs) — scales already ready, no vmcnt stall ──
        fp4_mma_v2<false>(acc_A0Bl, tA0, tBl, a0_raw, bl_raw, 0);
        fp4_mma_v2<false>(acc_A0Br, tA0, tBr, a0_raw, br_raw, 0);
        fp4_mma_v2<false>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw, 0);
        fp4_mma_v2<false>(acc_A1Br, tA1, tBr, a1_raw, br_raw, 0);

        // ── Barrier + tile prefetch ──
        __builtin_amdgcn_s_barrier();
        {
            const int pf_bt = (bt + 2 < k_byte_iters) ? (bt + 2) : (k_byte_iters - 1);
            load_tiles(pf_bt, cur);
        }

        // ── ALL Phase 1 (64 MFMAs) ──
        fp4_mma_v2<true>(acc_A0Bl, tA0, tBl, a0_raw, bl_raw, 1);
        fp4_mma_v2<true>(acc_A0Br, tA0, tBr, a0_raw, br_raw, 1);
        fp4_mma_v2<true>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw, 1);
        fp4_mma_v2<true>(acc_A1Br, tA1, tBr, a1_raw, br_raw, 1);
    }

    // ══════════════════ Epilogue: store C ══════════════════
    auto store_acc = [&](const fp4_acc_v2& acc, int m_half, int n_half) {
        RT_C c_store;
        #pragma unroll
        for (int r = 0; r < (RBM / 16); ++r)
            #pragma unroll
            for (int c = 0; c < (RBN / 16); ++c)
                *reinterpret_cast<fp4_floatx4_t*>(&c_store.tiles[r][c].data[0]) =
                    acc.regs[r * (RBN / 16) + c] * g.scale;

        store(g.c, c_store, {0, 0,
            br * WARPS_M * 2 + WARPS_M * m_half + wm,
            bc * WARPS_N * 2 + WARPS_N * n_half + wn});
    };

    store_acc(acc_A0Bl, 0, 0);
    store_acc(acc_A0Br, 0, 1);
    store_acc(acc_A1Bl, 1, 0);
    store_acc(acc_A1Br, 1, 1);
}

void dispatch_rcr_v2(v2_globals g) {
    g.m = static_cast<int>(g.c.rows());
    g.n = static_cast<int>(g.c.cols());
    g.k = static_cast<int>(g.a.cols()) * 2;
    const dim3 grid((g.m / BLK) * (g.n / BLK));
    mxfp4_rcr_v2_kernel<<<grid, dim3(_NUM_THREADS), 0, g.stream>>>(g);
}

PYBIND11_MODULE(tk_mxfp4_v2, m) {
    m.doc() = "MXFP4 GEMM V2: occupancy-1 double-buffered pipeline";
    py::bind_function<dispatch_rcr_v2>(m, "gemm_rcr",
        &v2_globals::a, &v2_globals::b,
        &v2_globals::a_scale, &v2_globals::b_scale,
        &v2_globals::c);
}
