// MXFP4 128x128 tile kernel — minimal version for correctness
//
// BLK=128, HB=64, RBM=RBN=32, 4 accumulators per block.
// Simple loop: load all 4 tiles, compute all 4 blocks, repeat.

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <type_traits>
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

constexpr int BLK = 128;
constexpr int BK  = 128;
constexpr int WARPS_M = 2, WARPS_N = 2;
constexpr int _NUM_WARPS   = WARPS_M * WARPS_N;
constexpr int _NUM_THREADS = _NUM_WARPS * WARP_THREADS;
constexpr int HB = BLK / 2;        // 64
constexpr int RBM = HB / WARPS_M;  // 32
constexpr int RBN = HB / WARPS_N;  // 32

constexpr int K_BYTES = K_DIM / 2;
constexpr int k_byte_iters = K_BYTES / BK;

using ST_tile = st_fp8e4m3<HB, BK, st_16x128_s>;
using A_row_reg = rt_fp8e4m3<RBM, BK, row_l, rt_16x128_s>;
using B_row_reg = rt_fp8e4m3<RBN, BK, row_l, rt_16x128_s>;

using G = kittens::group<_NUM_WARPS>;
using _gl_fp4   = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_scale = gl<fp8e8m0, -1, -1, -1, -1>;
using _gl_bf16  = gl<bf16, -1, -1, -1, -1>;

struct gluon_globals {
    _gl_fp4 a, b;
    _gl_scale a_scale, b_scale;
    _gl_bf16 c;
    float scale = 1.0f;
};

using fp4_intx8_t   = int __attribute__((__vector_size__(8 * sizeof(int))));
using fp4_intx4_t   = int __attribute__((__vector_size__(4 * sizeof(int))));
using fp4_floatx4_t = float __attribute__((__vector_size__(4 * sizeof(float))));

__device__ __forceinline__ fp4_intx4_t fp4_lo4(const fp4_intx8_t& x) {
    return __builtin_shufflevector(x, x, 0, 1, 2, 3);
}
__device__ __forceinline__ fp4_intx4_t fp4_hi4(const fp4_intx8_t& x) {
    return __builtin_shufflevector(x, x, 4, 5, 6, 7);
}

// ── LDS → register tile load ──
template<ducks::rt::row_layout RT, ducks::st::all ST>
__device__ __forceinline__ void fp4_load_st_to_rt(RT &dst, const ST &src) {
    static_assert(RT::rows == ST::rows && RT::cols == ST::cols);
    using T = typename base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename ST::dtype;
    constexpr int packing = base_types::packing<typename RT::dtype>::num();
    static_assert(std::is_same_v<T, U>);
    const int laneid = kittens::laneid();
    const int row_offset = laneid % dst.base_tile_rows;
    const int col_offset = dst.base_tile_stride * (laneid / dst.base_tile_rows);
    const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);
    constexpr int reg_sub_row = ST::underlying_subtile_cols / RT::base_tile_cols;
    constexpr int reg_sub_col = ST::underlying_subtile_rows / RT::base_tile_rows;
    #pragma unroll 8
    for (int k = 0; k < RT::base_tile_num_strides; k++)
        #pragma unroll 8
        for (int i = 0; i < reg_sub_col; i++)
            #pragma unroll 8
            for (int j = 0; j < reg_sub_row; j++) {
                const int row = i * RT::base_tile_rows + row_offset;
                const int col = j * RT::base_tile_cols + col_offset +
                    k * RT::base_tile_elements_per_stride_group;
                const uint32_t offset = sizeof(U) * (src_ptr + row * ST::underlying_subtile_cols + col);
                const uint32_t addr = offset ^ (((offset % (16 * 128)) >> 8) << 4);
                const int idx = k * RT::base_tile_stride / packing;
                #pragma unroll 8
                for (int ii = 0; ii < ST::subtiles_per_col; ii++)
                    #pragma unroll 8
                    for (int jj = 0; jj < ST::subtiles_per_row; jj++) {
                        const int sid = ii * ST::underlying_subtiles_per_row + jj;
                        const int soff = sid * ST::underlying_subtile_bytes;
                        asm volatile(
                            "ds_read_b128 %0, %1 offset:%2\n"
                            : "=v"(*reinterpret_cast<float4*>(
                                  &dst.tiles[ii * reg_sub_col + i][jj * reg_sub_row + j].data[idx]))
                            : "v"(addr), "i"(soff) : "memory"
                        );
                    }
            }
}

template<ducks::rt::row_layout RT>
__device__ __forceinline__ fp4_intx8_t fp4_extract_tile(const RT &src, int tile_row) {
    return *reinterpret_cast<const fp4_intx8_t*>(&src.tiles[tile_row][0].data[0]);
}

// ── Scale helpers ──
__device__ __forceinline__ const uint8_t* preshuffled_scale_row_base_ptr(
    const _gl_scale& src, int row_group) {
    return reinterpret_cast<const uint8_t*>(src.raw_ptr + src.idx(coord<>(row_group, 0)));
}

__device__ __forceinline__ i32x4 make_scale_srd(const uint8_t* ptr) {
    i32x4 srd = std::bit_cast<i32x4>(make_buffer_resource(
        static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(ptr)),
        0xFFFFFFFFu, 0x00110000u));
    srd[0] = __builtin_amdgcn_readfirstlane(srd[0]);
    srd[1] = __builtin_amdgcn_readfirstlane(srd[1]);
    srd[2] = __builtin_amdgcn_readfirstlane(srd[2]);
    srd[3] = __builtin_amdgcn_readfirstlane(srd[3]);
    return srd;
}

__device__ __forceinline__ void load_pq_scale_x2_async(
    i32x4 srsrc, uint32_t voffset, uint32_t soffset,
    fp8e8m0_4 &out_lo, fp8e8m0_4 &out_hi) {
    uint64_t pair;
    asm volatile(
        "buffer_load_dwordx2 %0, %1, %2, %3 offen"
        : "=v"(pair)
        : "v"(voffset), "s"(srsrc), "s"(soffset)
    );
    out_lo = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(pair));
    out_hi = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(pair >> 32));
}

// ── Tile prefetch ──
static constexpr int PF_MPT = (HB * BK * sizeof(fp8e4m3)) / (16 * _NUM_THREADS);

__device__ __forceinline__ void emit_tile_pf(
    auto &dst, const auto &src, const auto &idx,
    const uint32_t *so, i32x4 srd, const void *base, uint32_t lb)
{
    using ST = std::remove_reference_t<decltype(dst)>;
    using T = typename ST::dtype;
    coord<> uc = idx.template unit_coord<2, 3>();
    T* gptr = (T*)&src[uc];
    uint32_t soff = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<const char*>(gptr) - reinterpret_cast<const char*>(base)));
    asm volatile("" : "+s"(soff));
    const uint32_t lds_tile_base = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&dst.data[0])));
    const uint32_t warp_off = lb - lds_tile_base;
    constexpr int BPM = 16 * _NUM_THREADS;
    #pragma unroll
    for (int i = 0; i < PF_MPT; ++i) {
        const uint32_t lin = warp_off + i * BPM;
        const uint32_t sid = lin / ST::underlying_subtile_bytes;
        uint32_t lds_b = lds_tile_base + lin + sid * ST::subtile_padding;
        asm volatile("" : "+s"(lds_b));
        llvm_amdgcn_raw_buffer_load_lds(
            std::bit_cast<int32x4_t>(srd),
            (as3_uint32_ptr)(uintptr_t)lds_b,
            16, so[i], soff, 0,
            static_cast<int>(coherency::cache_all));
    }
}

// ── 8 MFMAs pure (no interleaving) for 32x32 register block ──
__device__ __forceinline__ void mfma8_pure(
    fp4_floatx4_t acc[4],
    fp4_intx4_t a0l, fp4_intx4_t a1l,
    fp4_intx4_t a0h, fp4_intx4_t a1h,
    fp4_intx4_t b0l, fp4_intx4_t b1l,
    fp4_intx4_t b0h, fp4_intx4_t b1h,
    unsigned sa, unsigned sb)
{
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0, %4,  %8,  %0, %12, %13 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1, %4,  %9,  %1, %12, %13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2, %5,  %8,  %2, %12, %13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3, %5,  %9,  %3, %12, %13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0, %6,  %10, %0, %12, %13 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1, %6,  %11, %1, %12, %13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2, %7,  %10, %2, %12, %13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3, %7,  %11, %3, %12, %13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc[0]), "+a"(acc[1]), "+a"(acc[2]), "+a"(acc[3])
        : "v"(a0l), "v"(a1l), "v"(a0h), "v"(a1h),
          "v"(b0l), "v"(b1l), "v"(b0h), "v"(b1h),
          "v"(sa), "v"(sb)
    );
}

// ══════════════════════════════════════════════════════════
// Main kernel — simple single-buffer version
// ══════════════════════════════════════════════════════════

__global__ __launch_bounds__(_NUM_THREADS, 2)
void mxfp4_128tile_kernel(const gluon_globals g) {
    static_assert(K_BYTES % BK == 0 && N_DIM % BLK == 0 && M_DIM % BLK == 0);

    constexpr int bpc = N_DIM / BLK;

    // Single-buffered LDS (no double-buffer for simplicity)
    __shared__ ST_tile A0_sh, A1_sh, Bl_sh, Br_sh;

    // XCD-aware dispatch
    constexpr int NUM_XCDS = 8;
#ifndef GROUP_SIZE_M
#define GROUP_SIZE_M 4
#endif
    constexpr int GROUP_M = GROUP_SIZE_M;
    const int total_blocks = gridDim.x;
    const int bpr = total_blocks / bpc;
    const int raw_bid = blockIdx.x;
    const int pids_per_xcd = (total_blocks + NUM_XCDS - 1) / NUM_XCDS;
    int tall_xcds = total_blocks % NUM_XCDS;
    if (tall_xcds == 0) tall_xcds = NUM_XCDS;
    const int xcd = raw_bid % NUM_XCDS;
    const int local_pid = raw_bid / NUM_XCDS;
    int bid;
    if (xcd < tall_xcds) bid = xcd * pids_per_xcd + local_pid;
    else bid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid;
    if (bid >= total_blocks) return;

    const int num_pig = GROUP_M * bpc;
    const int gid = bid / num_pig;
    const int fpm = gid * GROUP_M;
    const int gsm = (bpr - fpm < GROUP_M) ? (bpr - fpm) : GROUP_M;
    const int br = fpm + (bid % gsm);
    const int bc = (bid % num_pig) / gsm;
    const int wm = warpid() / WARPS_N, wn = warpid() % WARPS_N;

    uint32_t so_a[PF_MPT], so_b[PF_MPT];
    G::prefill_swizzled_offsets(A0_sh, g.a, so_a);
    G::prefill_swizzled_offsets(Bl_sh, g.b, so_b);

    // Scale setup — use dwordx2 like the original kernel
    const uint32_t lane_soff_x2 =
        (static_cast<uint32_t>(kittens::laneid() / 16) << 7) |
        (static_cast<uint32_t>(kittens::laneid() % 16) << 3);

    const int a0_row = br * BLK + wm * RBM;
    const int a1_row = br * BLK + HB + wm * RBM;
    const int bl_row = bc * BLK + wn * RBN;
    const int br_row = bc * BLK + HB + wn * RBN;

    // block_sel: which dword within the dwordx2 pair (0=lo, 1=hi)
    const int a0_bsel = (a0_row >> 5) & 1;
    const int a1_bsel = (a1_row >> 5) & 1;
    const int bl_bsel = (bl_row >> 5) & 1;
    const int br_bsel = (br_row >> 5) & 1;

    i32x4 a0_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.a_scale, a0_row >> 6));
    i32x4 a1_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.a_scale, a1_row >> 6));
    i32x4 bl_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.b_scale, bl_row >> 6));
    i32x4 br_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.b_scale, br_row >> 6));

    fp4_floatx4_t acc_A0Bl[4]={}, acc_A0Br[4]={}, acc_A1Bl[4]={}, acc_A1Br[4]={};

    auto make_srd = [](const void* raw_ptr) {
        i32x4 s = std::bit_cast<i32x4>(make_buffer_resource(
            static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(raw_ptr)),
            0xFFFFFFFFu, 0x00110000u));
        s[0] = __builtin_amdgcn_readfirstlane(s[0]);
        s[1] = __builtin_amdgcn_readfirstlane(s[1]);
        s[2] = __builtin_amdgcn_readfirstlane(s[2]);
        s[3] = __builtin_amdgcn_readfirstlane(s[3]);
        return s;
    };
    i32x4 srd_a = make_srd(g.a.raw_ptr), srd_b = make_srd(g.b.raw_ptr);
    const void *base_a = (const void*)g.a.raw_ptr, *base_b = (const void*)g.b.raw_ptr;

    constexpr int epw = 16 / sizeof(fp8e4m3) * WARP_THREADS;
    const uint32_t wlo = (warpid() % _NUM_WARPS) * epw * sizeof(fp8e4m3);
    auto lb = [&](auto &t) -> uint32_t {
        return __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(&t.data[0]) + wlo));
    };
    uint32_t lb_a0 = lb(A0_sh), lb_a1 = lb(A1_sh), lb_bl = lb(Bl_sh), lb_br = lb(Br_sh);

    // ═══════════ Main loop — simple single-buffer ═══════════
    for (int bt = 0; bt < k_byte_iters; ++bt) {
        // Load all 4 tiles into LDS
        emit_tile_pf(A0_sh, g.a, coord<ST_tile>(0,0,br*2,    bt), so_a, srd_a, base_a, lb_a0);
        emit_tile_pf(A1_sh, g.a, coord<ST_tile>(0,0,br*2+1,  bt), so_a, srd_a, base_a, lb_a1);
        emit_tile_pf(Bl_sh, g.b, coord<ST_tile>(0,0,bc*2,    bt), so_b, srd_b, base_b, lb_bl);
        emit_tile_pf(Br_sh, g.b, coord<ST_tile>(0,0,bc*2+1,  bt), so_b, srd_b, base_b, lb_br);

        // Load scales for this K step
        fp8e8m0_4 sa0_lo, sa0_hi, sa1_lo, sa1_hi, sbl_lo, sbl_hi, sbr_lo, sbr_hi;
        const uint32_t soff = static_cast<uint32_t>(bt) << 9;
        load_pq_scale_x2_async(a0_srd, lane_soff_x2, soff, sa0_lo, sa0_hi);
        load_pq_scale_x2_async(a1_srd, lane_soff_x2, soff, sa1_lo, sa1_hi);
        load_pq_scale_x2_async(bl_srd, lane_soff_x2, soff, sbl_lo, sbl_hi);
        load_pq_scale_x2_async(br_srd, lane_soff_x2, soff, sbr_lo, sbr_hi);

        // Wait for everything
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();

        // Select correct scale dword
        unsigned sa0_v = std::bit_cast<unsigned>(a0_bsel ? sa0_hi : sa0_lo);
        unsigned sa1_v = std::bit_cast<unsigned>(a1_bsel ? sa1_hi : sa1_lo);
        unsigned sbl_v = std::bit_cast<unsigned>(bl_bsel ? sbl_hi : sbl_lo);
        unsigned sbr_v = std::bit_cast<unsigned>(br_bsel ? sbr_hi : sbr_lo);

        // Load all tiles from LDS into registers
        A_row_reg a0_rt, a1_rt;
        B_row_reg bl_rt, br_rt;
        fp4_load_st_to_rt(a0_rt, kittens::subtile_inplace<RBM, BK>(A0_sh, {wm, 0}));
        fp4_load_st_to_rt(a1_rt, kittens::subtile_inplace<RBM, BK>(A1_sh, {wm, 0}));
        fp4_load_st_to_rt(bl_rt, kittens::subtile_inplace<RBN, BK>(Bl_sh, {wn, 0}));
        fp4_load_st_to_rt(br_rt, kittens::subtile_inplace<RBN, BK>(Br_sh, {wn, 0}));
        asm volatile("s_waitcnt lgkmcnt(0)");

        // Extract tile data
        fp4_intx8_t tA0[2], tA1[2], tBl[2], tBr[2];
        for (int i = 0; i < 2; i++) {
            tA0[i] = fp4_extract_tile(a0_rt, i);
            tA1[i] = fp4_extract_tile(a1_rt, i);
            tBl[i] = fp4_extract_tile(bl_rt, i);
            tBr[i] = fp4_extract_tile(br_rt, i);
        }

        // Extract operands
        fp4_intx4_t a0l=fp4_lo4(tA0[0]),a0_1l=fp4_lo4(tA0[1]),a0h=fp4_hi4(tA0[0]),a0_1h=fp4_hi4(tA0[1]);
        fp4_intx4_t a1l=fp4_lo4(tA1[0]),a1_1l=fp4_lo4(tA1[1]),a1h=fp4_hi4(tA1[0]),a1_1h=fp4_hi4(tA1[1]);
        fp4_intx4_t bl0l=fp4_lo4(tBl[0]),bl1l=fp4_lo4(tBl[1]),bl0h=fp4_hi4(tBl[0]),bl1h=fp4_hi4(tBl[1]);
        fp4_intx4_t br0l=fp4_lo4(tBr[0]),br1l=fp4_lo4(tBr[1]),br0h=fp4_hi4(tBr[0]),br1h=fp4_hi4(tBr[1]);

        // 4 blocks x 8 MFMAs = 32 MFMAs total
        mfma8_pure(acc_A0Bl, a0l, a0_1l, a0h, a0_1h, bl0l, bl1l, bl0h, bl1h, sa0_v, sbl_v);
        mfma8_pure(acc_A0Br, a0l, a0_1l, a0h, a0_1h, br0l, br1l, br0h, br1h, sa0_v, sbr_v);
        mfma8_pure(acc_A1Bl, a1l, a1_1l, a1h, a1_1h, bl0l, bl1l, bl0h, bl1h, sa1_v, sbl_v);
        mfma8_pure(acc_A1Br, a1l, a1_1l, a1h, a1_1h, br0l, br1l, br0h, br1h, sa1_v, sbr_v);
    }

    // ═══════════ Store C ═══════════
    auto store_block = [&](const fp4_floatx4_t acc[4], int mh, int nh) {
        const int lid = kittens::laneid();
        const int base_row = br * BLK + mh * HB + wm * RBM;
        const int base_col = bc * BLK + nh * HB + wn * RBN;
        bf16 *dst_ptr = g.c.raw_ptr + static_cast<size_t>(base_row) * g.c.cols()
                        + static_cast<size_t>(base_col);
        const int row_stride = g.c.cols();
        const int row_off = 4 * (lid / 16);
        const int col_off = lid % 16;

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                fp4_floatx4_t s = acc[i * 2 + j] * g.scale;
                const int row_base = i * 16 + row_off;
                const int col = j * 16 + col_off;
                dst_ptr[(row_base + 0) * row_stride + col] = base_types::convertor<bf16, float>::convert(s[0]);
                dst_ptr[(row_base + 1) * row_stride + col] = base_types::convertor<bf16, float>::convert(s[1]);
                dst_ptr[(row_base + 2) * row_stride + col] = base_types::convertor<bf16, float>::convert(s[2]);
                dst_ptr[(row_base + 3) * row_stride + col] = base_types::convertor<bf16, float>::convert(s[3]);
            }
        }
    };

    store_block(acc_A0Bl, 0, 0);
    store_block(acc_A1Bl, 1, 0);
    store_block(acc_A0Br, 0, 1);
    store_block(acc_A1Br, 1, 1);
}

void dispatch_128tile(gluon_globals g) {
    int m = static_cast<int>(g.c.rows());
    int n = static_cast<int>(g.c.cols());
    const dim3 grid((m / BLK) * (n / BLK));
    mxfp4_128tile_kernel<<<grid, dim3(_NUM_THREADS), 0>>>(g);
}

PYBIND11_MODULE(tk_mxfp4_128tile, m) {
    m.doc() = "MXFP4 128x128 tile kernel for higher occupancy";
    py::bind_function<dispatch_128tile>(m, "gemm_rcr",
        &gluon_globals::a, &gluon_globals::b,
        &gluon_globals::a_scale, &gluon_globals::b_scale,
        &gluon_globals::c);
}
