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

#define TK_STRINGIFY_IMPL(x) #x
#define TK_STRINGIFY(x) TK_STRINGIFY_IMPL(x)
#define TK_WAIT_LGKM(x) asm volatile("s_waitcnt lgkmcnt(" TK_STRINGIFY(x) ")")
#define TK_WAIT_VMCNT(x) asm volatile("s_waitcnt vmcnt(" TK_STRINGIFY(x) ")")

#ifndef GEMM_BLOCK_SIZE
#define GEMM_BLOCK_SIZE 256
#endif
#ifndef GEMM_K_BLOCK
#define GEMM_K_BLOCK 128
#endif
#ifndef GEMM_WARPS_M
#define GEMM_WARPS_M 2
#endif
#ifndef GEMM_WARPS_N
#define GEMM_WARPS_N 4
#endif
#ifndef GEMM_MIN_BLOCKS_PER_CU
#define GEMM_MIN_BLOCKS_PER_CU 2
#endif

constexpr int BLK = GEMM_BLOCK_SIZE, BK = GEMM_K_BLOCK;
constexpr int HB  = BLK / 2;
constexpr int WARPS_M = GEMM_WARPS_M, WARPS_N = GEMM_WARPS_N;
constexpr int _NUM_WARPS   = WARPS_M * WARPS_N;
constexpr int _NUM_THREADS = _NUM_WARPS * WARP_THREADS;
constexpr int RBM = BLK / WARPS_M / 2;
constexpr int RBN = BLK / WARPS_N / 2;

static_assert(BLK == 256 && BK == 128 && WARPS_M == 2 && WARPS_N == 4,
    "MXFP4 RCR 8-wave fast path requires BLK=256, BK=128, 2x4 warp grid");
static_assert(K_DIM % 2 == 0, "K_DIM must be even for FP4 packing");

constexpr int K_BYTES = K_DIM / 2;
constexpr int k_byte_iters = K_BYTES / BK;

using G = kittens::group<_NUM_WARPS>;
using _gl_fp4   = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_scale = gl<fp8e8m0, -1, -1, -1, -1>;
using _gl_bf16  = gl<bf16, -1, -1, -1, -1>;

using ST_A = st_fp8e4m3<HB, BK, st_16x128_s>;
using ST_B = st_fp8e4m3<HB, BK, st_16x128_s>;

using A_row_reg = rt_fp8e4m3<RBM, BK, row_l, rt_16x128_s>;
using B_row_reg = rt_fp8e4m3<RBN, BK, row_l, rt_16x128_s>;
using RT_C = rt_fl<RBM, RBN, col_l, rt_16x16_s>;

struct layout_globals {
    _gl_fp4 a, b;
    _gl_scale a_scale, b_scale;
    _gl_bf16 c;
    float scale = 1.0f;
    hipStream_t stream = nullptr;
    int m = 0, n = 0, k = 0;
};

using fp4_intx8_t   = int __attribute__((__vector_size__(8 * sizeof(int))));
using fp4_floatx4_t = float __attribute__((__vector_size__(4 * sizeof(float))));

struct alignas(16) fp4_acc {
    fp4_floatx4_t regs[(RBM / 16) * (RBN / 16)];
};

template<int OPSEL_A, int OPSEL_B>
__device__ __forceinline__ void fp4_mfma_scale_inplace(
    fp4_floatx4_t& d,
    const fp4_intx8_t& a,
    const fp4_intx8_t& b,
    fp8e8m0_4 scale_a,
    fp8e8m0_4 scale_b)
{
    d = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a, b, d,
        4, 4,
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
                        static_assert(std::is_same_v<U2, fp8e4m3_4>);
                        static_assert(RT::base_tile_stride == 16);
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

__device__ __forceinline__ const uint8_t* preshuffled_scale_row_base_ptr(
    const _gl_scale& src, int row_group)
{
    return reinterpret_cast<const uint8_t*>(src.raw_ptr + src.idx(coord<>(row_group, 0)));
}

__device__ __forceinline__ fp8e8m0_4 load_pq_scale(
    const uint8_t* row_base, int k_pair, uint32_t lane_byte_offset)
{
    const uint32_t off = (static_cast<uint32_t>(k_pair) << 8) + lane_byte_offset;
    return std::bit_cast<fp8e8m0_4>(
        *reinterpret_cast<const uint32_t*>(row_base + off)
    );
}

__device__ __forceinline__ fp8e8m0_4 remap_phase(fp8e8m0_4 src, int k_phase)
{
    return std::bit_cast<fp8e8m0_4>(
        std::bit_cast<uint32_t>(src) >> (static_cast<uint32_t>(k_phase & 1) << 4)
    );
}

template<bool UPPER>
__device__ __forceinline__ void fp4_mma_all_subtiles(
    fp4_acc& acc,
    const fp4_intx8_t& A0, const fp4_intx8_t& A1,
    const fp4_intx8_t& A2, const fp4_intx8_t& A3,
    const fp4_intx8_t& B0, const fp4_intx8_t& B1,
    fp8e8m0_4 a_sp0,
    fp8e8m0_4 a_sp1,
    fp8e8m0_4 b_sp)
{
    fp4_intx8_t a0, a1, a2, a3, b0, b1;
    if constexpr (UPPER) {
        a0 = fp4_upper_half(A0); a1 = fp4_upper_half(A1);
        a2 = fp4_upper_half(A2); a3 = fp4_upper_half(A3);
        b0 = fp4_upper_half(B0); b1 = fp4_upper_half(B1);
    } else {
        a0 = A0; a1 = A1; a2 = A2; a3 = A3;
        b0 = B0; b1 = B1;
    }

    fp4_mfma_scale_inplace<0, 0>(acc.regs[0], a0, b0, a_sp0, b_sp);
    fp4_mfma_scale_inplace<0, 1>(acc.regs[1], a0, b1, a_sp0, b_sp);
    fp4_mfma_scale_inplace<1, 0>(acc.regs[2], a1, b0, a_sp0, b_sp);
    fp4_mfma_scale_inplace<1, 1>(acc.regs[3], a1, b1, a_sp0, b_sp);
    fp4_mfma_scale_inplace<0, 0>(acc.regs[4], a2, b0, a_sp1, b_sp);
    fp4_mfma_scale_inplace<0, 1>(acc.regs[5], a2, b1, a_sp1, b_sp);
    fp4_mfma_scale_inplace<1, 0>(acc.regs[6], a3, b0, a_sp1, b_sp);
    fp4_mfma_scale_inplace<1, 1>(acc.regs[7], a3, b1, a_sp1, b_sp);
}

__device__ __forceinline__ void fp4_acc_to_rt(
    RT_C& dst, const fp4_acc& src, float s)
{
    #pragma unroll
    for (int r = 0; r < (RBM / 16); ++r)
        #pragma unroll
        for (int c = 0; c < (RBN / 16); ++c)
            *reinterpret_cast<fp4_floatx4_t*>(&dst.tiles[r][c].data[0]) =
                src.regs[r * (RBN / 16) + c] * s;
}

__global__ __launch_bounds__(_NUM_THREADS, GEMM_MIN_BLOCKS_PER_CU)
void mxfp4_rcr_pq_kernel(const layout_globals g) {
    static_assert(K_BYTES % BK == 0 && N_DIM % BLK == 0 && M_DIM % BLK == 0);

    constexpr int bpc = N_DIM / BLK;
    __shared__ ST_A As[2];
    __shared__ ST_B Bs[2];

    A_row_reg a_rt;
    B_row_reg b0_rt, b1_rt;
    fp4_acc cA{}, cB{}, cC{}, cD{};

    const int bid = blockIdx.x;
    const int br = bid / bpc, bc = bid % bpc;
    const int wm = warpid() / WARPS_N, wn = warpid() % WARPS_N;
    const int lane_nonk = kittens::laneid() % 16;
    const int lane_kblk = kittens::laneid() / 16;
    const uint32_t lane_soff =
        (static_cast<uint32_t>(lane_kblk) << 6) |
        (static_cast<uint32_t>(lane_nonk) << 2);

    constexpr int bpt = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * _NUM_THREADS;
    constexpr int mpt_a = HB * BK * sizeof(fp8e4m3) / bpm;
    constexpr int mpt_b = HB * BK * sizeof(fp8e4m3) / bpm;
    uint32_t so_a[mpt_a], so_b[mpt_b];
    G::prefill_swizzled_offsets(As[0], g.a, so_a);
    G::prefill_swizzled_offsets(Bs[0], g.b, so_b);

    constexpr int a_packs = RBM / 32;
    constexpr int b_packs = (RBN + 31) / 32;
    const uint8_t* a0_sb[a_packs], *a1_sb[a_packs];
    const uint8_t* b0_sb[b_packs], *b1_sb[b_packs];
    #pragma unroll
    for (int p = 0; p < a_packs; ++p) {
        a0_sb[p] = preshuffled_scale_row_base_ptr(
            g.a_scale, (br * BLK + 0 * HB + wm * RBM + p * 32) >> 5);
        a1_sb[p] = preshuffled_scale_row_base_ptr(
            g.a_scale, (br * BLK + 1 * HB + wm * RBM + p * 32) >> 5);
    }
    #pragma unroll
    for (int p = 0; p < b_packs; ++p) {
        b0_sb[p] = preshuffled_scale_row_base_ptr(
            g.b_scale, (bc * BLK + 0 * HB + wn * RBN + p * 32) >> 5);
        b1_sb[p] = preshuffled_scale_row_base_ptr(
            g.b_scale, (bc * BLK + 1 * HB + wn * RBN + p * 32) >> 5);
    }

    for (int bt = 0; bt < k_byte_iters; ++bt) {
        G::load(As[0], g.a, {0, 0, br * 2,     bt}, so_a);
        G::load(As[1], g.a, {0, 0, br * 2 + 1, bt}, so_a);
        G::load(Bs[0], g.b, {0, 0, bc * 2,     bt}, so_b);
        G::load(Bs[1], g.b, {0, 0, bc * 2 + 1, bt}, so_b);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();

        auto as0 = kittens::subtile_inplace<RBM, BK>(As[0], {wm, 0});
        auto bs0 = kittens::subtile_inplace<RBN, BK>(Bs[0], {wn, 0});
        auto bs1 = kittens::subtile_inplace<RBN, BK>(Bs[1], {wn, 0});

        fp8e8m0_4 a0_raw[a_packs], a1_raw[a_packs];
        fp8e8m0_4 b0_raw[b_packs], b1_raw[b_packs];
        #pragma unroll
        for (int p = 0; p < a_packs; ++p) {
            a0_raw[p] = load_pq_scale(a0_sb[p], bt, lane_soff);
            a1_raw[p] = load_pq_scale(a1_sb[p], bt, lane_soff);
        }
        #pragma unroll
        for (int p = 0; p < b_packs; ++p) {
            b0_raw[p] = load_pq_scale(b0_sb[p], bt, lane_soff);
            b1_raw[p] = load_pq_scale(b1_sb[p], bt, lane_soff);
        }

        fp4_load_st_to_rt(a_rt, as0);
        fp4_load_st_to_rt(b0_rt, bs0);
        fp4_load_st_to_rt(b1_rt, bs1);
        asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        fp4_intx8_t A0 = fp4_extract_tile(a_rt, 0);
        fp4_intx8_t A1 = fp4_extract_tile(a_rt, 1);
        fp4_intx8_t A2 = fp4_extract_tile(a_rt, 2);
        fp4_intx8_t A3 = fp4_extract_tile(a_rt, 3);
        fp4_intx8_t B0_0 = fp4_extract_tile(b0_rt, 0);
        fp4_intx8_t B0_1 = fp4_extract_tile(b0_rt, 1);
        fp4_intx8_t B1_0 = fp4_extract_tile(b1_rt, 0);
        fp4_intx8_t B1_1 = fp4_extract_tile(b1_rt, 1);

        fp8e8m0_4 a0p0_lo = remap_phase(a0_raw[0], 0);
        fp8e8m0_4 a0p1_lo = remap_phase(a0_raw[1], 0);
        fp8e8m0_4 b0p_lo  = remap_phase(b0_raw[0], 0);
        fp8e8m0_4 b1p_lo  = remap_phase(b1_raw[0], 0);

        // Fire A1 (M-half 1) reload early — overlaps with cA + cB MMA
        auto as1 = kittens::subtile_inplace<RBM, BK>(As[1], {wm, 0});
        fp4_load_st_to_rt(a_rt, as1);

        // Phase 0: M-half 0
        fp4_mma_all_subtiles<false>(cA, A0, A1, A2, A3, B0_0, B0_1, a0p0_lo, a0p1_lo, b0p_lo);
        fp4_mma_all_subtiles<false>(cB, A0, A1, A2, A3, B1_0, B1_1, a0p0_lo, a0p1_lo, b1p_lo);

        // Collect A1 data
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        fp4_intx8_t A1_0 = fp4_extract_tile(a_rt, 0);
        fp4_intx8_t A1_1 = fp4_extract_tile(a_rt, 1);
        fp4_intx8_t A1_2 = fp4_extract_tile(a_rt, 2);
        fp4_intx8_t A1_3 = fp4_extract_tile(a_rt, 3);

        fp8e8m0_4 a1p0_lo = remap_phase(a1_raw[0], 0);
        fp8e8m0_4 a1p1_lo = remap_phase(a1_raw[1], 0);

        fp4_mma_all_subtiles<false>(cC, A1_0, A1_1, A1_2, A1_3, B0_0, B0_1, a1p0_lo, a1p1_lo, b0p_lo);
        fp4_mma_all_subtiles<false>(cD, A1_0, A1_1, A1_2, A1_3, B1_0, B1_1, a1p0_lo, a1p1_lo, b1p_lo);

        // Phase 1: M-half 1 first — compute hi scales on demand
        fp8e8m0_4 a1p0_hi = remap_phase(a1_raw[0], 1);
        fp8e8m0_4 a1p1_hi = remap_phase(a1_raw[1], 1);
        fp8e8m0_4 b0p_hi  = remap_phase(b0_raw[0], 1);
        fp8e8m0_4 b1p_hi  = remap_phase(b1_raw[0], 1);

        fp4_mma_all_subtiles<true>(cC, A1_0, A1_1, A1_2, A1_3, B0_0, B0_1, a1p0_hi, a1p1_hi, b0p_hi);

        fp4_mma_all_subtiles<true>(cD, A1_0, A1_1, A1_2, A1_3, B1_0, B1_1, a1p0_hi, a1p1_hi, b1p_hi);

        // Phase 1: M-half 0 — reuse A0-A3 from initial extraction (still live)
        fp8e8m0_4 a0p0_hi = remap_phase(a0_raw[0], 1);
        fp8e8m0_4 a0p1_hi = remap_phase(a0_raw[1], 1);

        fp4_mma_all_subtiles<true>(cA, A0, A1, A2, A3, B0_0, B0_1, a0p0_hi, a0p1_hi, b0p_hi);
        fp4_mma_all_subtiles<true>(cB, A0, A1, A2, A3, B1_0, B1_1, a0p0_hi, a0p1_hi, b1p_hi);

        __builtin_amdgcn_s_barrier();
    }

    RT_C c_store;
    fp4_acc_to_rt(c_store, cA, g.scale);
    store(g.c, c_store, {0, 0, br * WARPS_M * 2 + wm,              bc * WARPS_N * 2 + wn});
    fp4_acc_to_rt(c_store, cB, g.scale);
    store(g.c, c_store, {0, 0, br * WARPS_M * 2 + wm,              bc * WARPS_N * 2 + WARPS_N + wn});
    fp4_acc_to_rt(c_store, cC, g.scale);
    store(g.c, c_store, {0, 0, br * WARPS_M * 2 + WARPS_M + wm,    bc * WARPS_N * 2 + wn});
    fp4_acc_to_rt(c_store, cD, g.scale);
    store(g.c, c_store, {0, 0, br * WARPS_M * 2 + WARPS_M + wm,    bc * WARPS_N * 2 + WARPS_N + wn});
}

void dispatch_rcr_pq(layout_globals g) {
    g.m = static_cast<int>(g.c.rows());
    g.n = static_cast<int>(g.c.cols());
    g.k = static_cast<int>(g.a.cols()) * 2;
    const dim3 grid((g.m / BLK) * (g.n / BLK));
    mxfp4_rcr_pq_kernel<<<grid, dim3(_NUM_THREADS), 0, g.stream>>>(g);
}

void dispatch_rcr(layout_globals g) { dispatch_rcr_pq(g); }

PYBIND11_MODULE(tk_mxfp4_layouts, m) {
    m.doc() = "MXFP4 GEMM with preshuffle-quant scales (RCR layout)";
    py::bind_function<dispatch_rcr>(m, "gemm_rcr",
        &layout_globals::a, &layout_globals::b,
        &layout_globals::a_scale, &layout_globals::b_scale,
        &layout_globals::c);
    py::bind_function<dispatch_rcr_pq>(m, "gemm_rcr_pq",
        &layout_globals::a, &layout_globals::b,
        &layout_globals::a_scale, &layout_globals::b_scale,
        &layout_globals::c);
}
