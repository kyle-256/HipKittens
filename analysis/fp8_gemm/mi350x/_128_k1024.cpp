// MXFP4 128x128 tile kernel variant for higher occupancy
//
// Based on kernel_mxfp4_gluon_cpp.cpp but with:
//   BLK=128, HB=64, RBM=RBN=32 (register block 32x32 per warp)
//   4 acc per block (2x2 MFMAs) instead of 16 (4x4)
//   32 MFMAs per K-step (4 blocks x 4 MFMAs x 2 phases)
//   Target: occupancy=2 via smaller register footprint
//
// Per K-iteration:
//   Step 1: A0 x Bl (8 MFMAs) + ds_read Br
//   Step 2: A0 x Br (8 MFMAs) + ds_read A1
//   Step 3: A1 x Bl (8 MFMAs) + ds_read A0[nxt]
//   Step 4: A1 x Br (8 MFMAs) + ds_read Bl[nxt]

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

__device__ __forceinline__ fp8e8m0_4 load_pq_scale_srd(
    i32x4 srsrc, uint32_t voffset, uint32_t soffset) {
    return std::bit_cast<fp8e8m0_4>(
        llvm_amdgcn_raw_buffer_load_b32(srsrc, voffset, soffset, 0));
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

// ── LDS address computation ──

template<ducks::rt::row_layout RT, ducks::st::all ST>
__device__ __forceinline__ void compute_lds_base_addrs(
    const ST &src, uint32_t &addr_p0, uint32_t &addr_p1)
{
    const int laneid = kittens::laneid();
    const int row_offset = laneid % RT::base_tile_rows;
    const int col_offset = RT::base_tile_stride * (laneid / RT::base_tile_rows);
    const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);
    using U = typename ST::dtype;
    constexpr int subcols = ST::underlying_subtile_cols;
    const uint32_t off0 = sizeof(U) * (src_ptr + row_offset * subcols + col_offset);
    addr_p0 = off0 ^ (((off0 % (16 * 128)) >> 8) << 4);
    const int col1 = col_offset + RT::base_tile_elements_per_stride_group;
    const uint32_t off1 = sizeof(U) * (src_ptr + row_offset * subcols + col1);
    addr_p1 = off1 ^ (((off1 % (16 * 128)) >> 8) << 4);
}

// ── Tile prefetch params ──

struct tile_pf_params {
    int32x4_t srd;
    uint32_t soff;
    uint32_t lds_addrs[PF_MPT];
    uint32_t voffs[PF_MPT];
};

template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ __forceinline__ tile_pf_params make_pf_params(
    ST &dst, const GL &src, const COORD &idx,
    const uint32_t *so, i32x4 srd_in, const void *base_ptr, uint32_t lds_base)
{
    using T = typename ST::dtype;
    constexpr int BPM = 16 * _NUM_THREADS;
    coord<> uc = idx.template unit_coord<2, 3>();
    T* gptr = (T*)&src[uc];
    uint32_t soff = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<const char*>(gptr) - reinterpret_cast<const char*>(base_ptr)));
    const uint32_t lds_tile_base = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&dst.data[0])));
    const uint32_t warp_off = lds_base - lds_tile_base;
    tile_pf_params p;
    p.srd = std::bit_cast<int32x4_t>(srd_in);
    p.soff = soff;
    #pragma unroll
    for (int i = 0; i < PF_MPT; ++i) {
        p.voffs[i] = so[i];
        const uint32_t lin = warp_off + i * BPM;
        const uint32_t sid = lin / ST::underlying_subtile_bytes;
        p.lds_addrs[i] = lds_tile_base + lin + sid * ST::subtile_padding;
    }
    return p;
}

__device__ __forceinline__ void emit_one_pf(const tile_pf_params& p, int idx) {
    llvm_amdgcn_raw_buffer_load_lds(
        std::bit_cast<int32x4_t>(p.srd),
        (as3_uint32_ptr)(uintptr_t)p.lds_addrs[idx],
        16, p.voffs[idx], p.soff, 0,
        static_cast<int>(coherency::cache_all));
}

// ══════════════════════════════════════════════════════════
// 8-MFMA blocks for 32x32 register tiles (2x2 base tiles)
// ══════════════════════════════════════════════════════════
//
// With RBM=RBN=32: 2 MFMA tile rows x 2 MFMA tile cols = 4 accumulators
// A has 2 tile rows: A[0], A[1] → lo/hi halves each
// B has 2 tile rows: B[0], B[1] → lo/hi halves each
// 1 scale per A half, 1 scale per B half (RBM/32=1)
//
// Accumulator mapping:
//   acc[0] = A[0] x B[0]  op_sel:[0,0,0]
//   acc[1] = A[0] x B[1]  op_sel:[0,1,0]
//   acc[2] = A[1] x B[0]  op_sel:[1,0,0]
//   acc[3] = A[1] x B[1]  op_sel:[1,1,0]

// 8 MFMAs pure (no interleaving)
// Operands: %0..3=acc, %4..5=a_lo, %6..7=a_hi, %8..9=b_lo, %10..11=b_hi, %12=sa, %13=sb
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

// 8 MFMAs + 4 ds_reads interleaved (1:1 during phase 0)
// Outputs: %0..3=acc(+a), %4..7=ds_out(=&v)
// Inputs: %8..9=a_lo, %10..11=a_hi, %12..13=b_lo, %14..15=b_hi, %16=sa, %17=sb, %18..19=lds
__device__ __forceinline__ void mfma8_with_4lds(
    fp4_floatx4_t acc[4],
    fp4_intx4_t a0l, fp4_intx4_t a1l,
    fp4_intx4_t a0h, fp4_intx4_t a1h,
    fp4_intx4_t b0l, fp4_intx4_t b1l,
    fp4_intx4_t b0h, fp4_intx4_t b1h,
    unsigned sa, unsigned sb,
    float4 &d0, float4 &d1, float4 &d2, float4 &d3,
    uint32_t lds_p0, uint32_t lds_p1)
{
    asm volatile(
        // Phase 0 + 4 ds_reads
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0, %8,  %12, %0, %16, %17 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %4, %18 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1, %8,  %13, %1, %16, %17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %5, %18 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2, %9,  %12, %2, %16, %17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %6, %19 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3, %9,  %13, %3, %16, %17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %7, %19 offset:2048\n"
        // Phase 1 — pure MFMAs
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0, %10, %14, %0, %16, %17 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1, %10, %15, %1, %16, %17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2, %11, %14, %2, %16, %17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3, %11, %15, %3, %16, %17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc[0]), "+a"(acc[1]), "+a"(acc[2]), "+a"(acc[3]),
          "=&v"(d0), "=&v"(d1), "=&v"(d2), "=&v"(d3)
        : "v"(a0l), "v"(a1l), "v"(a0h), "v"(a1h),
          "v"(b0l), "v"(b1l), "v"(b0h), "v"(b1h),
          "v"(sa), "v"(sb),
          "v"(lds_p0), "v"(lds_p1)
    );
}

// ══════════════════════════════════════════════════════════
// Main kernel
// ══════════════════════════════════════════════════════════

__global__ __launch_bounds__(_NUM_THREADS, 2)
void mxfp4_128tile_kernel(const gluon_globals g) {
    static_assert(K_BYTES % BK == 0 && N_DIM % BLK == 0 && M_DIM % BLK == 0);

    constexpr int bpc = N_DIM / BLK;

    __shared__ ST_tile A0_db[2], A1_db[2], Bl_db[2], Br_db[2];

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
    if (xcd < tall_xcds) {
        bid = xcd * pids_per_xcd + local_pid;
    } else {
        bid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid;
    }
    if (bid >= total_blocks) return;

    const int num_pig = GROUP_M * bpc;
    const int gid = bid / num_pig;
    const int fpm = gid * GROUP_M;
    const int gsm = (bpr - fpm < GROUP_M) ? (bpr - fpm) : GROUP_M;
    const int br = fpm + (bid % gsm);
    const int bc = (bid % num_pig) / gsm;
    const int wm = warpid() / WARPS_N, wn = warpid() % WARPS_N;

    uint32_t so_a[PF_MPT], so_b[PF_MPT];
    G::prefill_swizzled_offsets(A0_db[0], g.a, so_a);
    G::prefill_swizzled_offsets(Bl_db[0], g.b, so_b);

    // Scale SRDs — with RBM=32, 1 scale group per warp half
    // In preshuffle_mfma16_merged format, each 64-row super-group interleaves
    // two 32-row blocks as dword pairs. A single dword load gets one block.
    // block_within_sg = ((row_base >> 5) & 1) determines which dword to load.
    const uint32_t lane_soff_base =
        (static_cast<uint32_t>(kittens::laneid() / 16) << 7) |
        (static_cast<uint32_t>(kittens::laneid() % 16) << 3);

    // Compute per-warp voffsets that select the correct 32-row block within super-group
    const int a0_row = br * BLK + wm * RBM;           // A0 half, warp wm
    const int a1_row = br * BLK + HB + wm * RBM;      // A1 half, warp wm
    const int bl_row = bc * BLK + wn * RBN;            // Bl half, warp wn
    const int br_row = bc * BLK + HB + wn * RBN;       // Br half, warp wn

    const uint32_t a0_soff_adj = lane_soff_base + (((a0_row >> 5) & 1) * 4);
    const uint32_t a1_soff_adj = lane_soff_base + (((a1_row >> 5) & 1) * 4);
    const uint32_t bl_soff_adj = lane_soff_base + (((bl_row >> 5) & 1) * 4);
    const uint32_t br_soff_adj = lane_soff_base + (((br_row >> 5) & 1) * 4);

    i32x4 a0_srd = make_scale_srd(preshuffled_scale_row_base_ptr(
        g.a_scale, a0_row >> 6));
    i32x4 a1_srd = make_scale_srd(preshuffled_scale_row_base_ptr(
        g.a_scale, a1_row >> 6));
    i32x4 bl_srd = make_scale_srd(preshuffled_scale_row_base_ptr(
        g.b_scale, bl_row >> 6));
    i32x4 br_srd = make_scale_srd(preshuffled_scale_row_base_ptr(
        g.b_scale, br_row >> 6));

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
    uint32_t lb_a0[2], lb_a1[2], lb_bl[2], lb_br[2];
    for (int d = 0; d < 2; ++d) {
        lb_a0[d]=lb(A0_db[d]); lb_a1[d]=lb(A1_db[d]);
        lb_bl[d]=lb(Bl_db[d]); lb_br[d]=lb(Br_db[d]);
    }

    auto load_tiles = [&](int bt, int db) {
        emit_tile_pf(A0_db[db], g.a, coord<ST_tile>(0,0,br*2,    bt), so_a, srd_a, base_a, lb_a0[db]);
        emit_tile_pf(A1_db[db], g.a, coord<ST_tile>(0,0,br*2+1,  bt), so_a, srd_a, base_a, lb_a1[db]);
        emit_tile_pf(Bl_db[db], g.b, coord<ST_tile>(0,0,bc*2,    bt), so_b, srd_b, base_b, lb_bl[db]);
        emit_tile_pf(Br_db[db], g.b, coord<ST_tile>(0,0,bc*2+1,  bt), so_b, srd_b, base_b, lb_br[db]);
    };

    // Pre-compute LDS addresses (static named vars)
    uint32_t a0_0_p0, a0_0_p1, a0_1_p0, a0_1_p1;
    uint32_t bl_0_p0, bl_0_p1, bl_1_p0, bl_1_p1;
    uint32_t br_0_p0, br_0_p1, br_1_p0, br_1_p1;
    uint32_t a1_0_p0, a1_0_p1, a1_1_p0, a1_1_p1;
    compute_lds_base_addrs<A_row_reg>(kittens::subtile_inplace<RBM, BK>(A0_db[0], {wm, 0}), a0_0_p0, a0_0_p1);
    compute_lds_base_addrs<A_row_reg>(kittens::subtile_inplace<RBM, BK>(A0_db[1], {wm, 0}), a0_1_p0, a0_1_p1);
    compute_lds_base_addrs<B_row_reg>(kittens::subtile_inplace<RBN, BK>(Bl_db[0], {wn, 0}), bl_0_p0, bl_0_p1);
    compute_lds_base_addrs<B_row_reg>(kittens::subtile_inplace<RBN, BK>(Bl_db[1], {wn, 0}), bl_1_p0, bl_1_p1);
    compute_lds_base_addrs<B_row_reg>(kittens::subtile_inplace<RBN, BK>(Br_db[0], {wn, 0}), br_0_p0, br_0_p1);
    compute_lds_base_addrs<B_row_reg>(kittens::subtile_inplace<RBN, BK>(Br_db[1], {wn, 0}), br_1_p0, br_1_p1);
    compute_lds_base_addrs<A_row_reg>(kittens::subtile_inplace<RBM, BK>(A1_db[0], {wm, 0}), a1_0_p0, a1_0_p1);
    compute_lds_base_addrs<A_row_reg>(kittens::subtile_inplace<RBM, BK>(A1_db[1], {wm, 0}), a1_1_p0, a1_1_p1);

    // Tile extraction: ds_read float4[4] → fp4_intx8_t[2]
    auto extract_tile_small = [](const float4 d[4], fp4_intx8_t t[2]) __attribute__((always_inline)) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            auto lo = *reinterpret_cast<const fp4_intx4_t*>(&d[i]);
            auto hi = *reinterpret_cast<const fp4_intx4_t*>(&d[i + 2]);
            t[i][0]=lo[0]; t[i][1]=lo[1]; t[i][2]=lo[2]; t[i][3]=lo[3];
            t[i][4]=hi[0]; t[i][5]=hi[1]; t[i][6]=hi[2]; t[i][7]=hi[3];
        }
    };

    // ═══════════ Prologue ═══════════
    load_tiles(0, 0);
    if (k_byte_iters > 1) load_tiles(1, 1);

    // Prefetch first scales (a_packs=1 for 32-row blocks → single buffer_load_b32)
    fp8e8m0_4 pf_a0, pf_a1, pf_bl, pf_br;
    {
        pf_a0 = load_pq_scale_srd(a0_srd, a0_soff_adj, 0);
        pf_a1 = load_pq_scale_srd(a1_srd, a1_soff_adj, 0);
        pf_bl = load_pq_scale_srd(bl_srd, bl_soff_adj, 0);
        pf_br = load_pq_scale_srd(br_srd, br_soff_adj, 0);
    }

    // Pre-load A0+Bl for iteration 0
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();
    A_row_reg a0_rt;
    B_row_reg bl_rt;
    fp4_load_st_to_rt(a0_rt, kittens::subtile_inplace<RBM, BK>(A0_db[0], {wm, 0}));
    fp4_load_st_to_rt(bl_rt, kittens::subtile_inplace<RBN, BK>(Bl_db[0], {wn, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    fp4_intx8_t tA0[2], tBl[2];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        tA0[i] = fp4_extract_tile(a0_rt, i);
        tBl[i] = fp4_extract_tile(bl_rt, i);
    }

    // ═══════════ Main loop ═══════════
#ifdef UNROLL_K
  #if UNROLL_K == 0
    #pragma unroll
  #else
    #pragma unroll UNROLL_K
  #endif
#elif (K_DIM / 256) <= 16
    #pragma unroll
#elif (K_DIM / 256) <= 32
    #pragma unroll 16
#else
    #pragma unroll 8
#endif
    for (int bt = 0; bt < k_byte_iters; ++bt) {
        const int cur = bt & 1;
        const int nxt = 1 - cur;

        const uint32_t sel_br_p0 = cur ? br_1_p0 : br_0_p0;
        const uint32_t sel_br_p1 = cur ? br_1_p1 : br_0_p1;
        const uint32_t sel_a1_p0 = cur ? a1_1_p0 : a1_0_p0;
        const uint32_t sel_a1_p1 = cur ? a1_1_p1 : a1_0_p1;
        const uint32_t sel_a0_p0 = nxt ? a0_1_p0 : a0_0_p0;
        const uint32_t sel_a0_p1 = nxt ? a0_1_p1 : a0_0_p1;
        const uint32_t sel_bl_p0 = nxt ? bl_1_p0 : bl_0_p0;
        const uint32_t sel_bl_p1 = nxt ? bl_1_p1 : bl_0_p1;

        const int pf_bt = (bt + 2 < k_byte_iters) ? (bt + 2) : (k_byte_iters - 1);
        tile_pf_params pf_a0_p = make_pf_params(A0_db[cur], g.a, coord<ST_tile>(0,0,br*2,     pf_bt), so_a, srd_a, base_a, lb_a0[cur]);
        tile_pf_params pf_a1_p = make_pf_params(A1_db[cur], g.a, coord<ST_tile>(0,0,br*2+1,   pf_bt), so_a, srd_a, base_a, lb_a1[cur]);
        tile_pf_params pf_bl_p = make_pf_params(Bl_db[cur], g.b, coord<ST_tile>(0,0,bc*2,     pf_bt), so_b, srd_b, base_b, lb_bl[cur]);
        tile_pf_params pf_br_p = make_pf_params(Br_db[cur], g.b, coord<ST_tile>(0,0,bc*2+1,   pf_bt), so_b, srd_b, base_b, lb_br[cur]);

        // Snapshot current scales
        fp8e8m0_4 a0_raw = pf_a0, a1_raw = pf_a1, bl_raw = pf_bl, br_raw = pf_br;

        // Prefetch next scales
        {
            const uint32_t nxt_scale = static_cast<uint32_t>(bt + 1 < k_byte_iters ? bt + 1 : bt) << 9;
            pf_a0 = load_pq_scale_srd(a0_srd, a0_soff_adj, nxt_scale);
            pf_a1 = load_pq_scale_srd(a1_srd, a1_soff_adj, nxt_scale);
            pf_bl = load_pq_scale_srd(bl_srd, bl_soff_adj, nxt_scale);
            pf_br = load_pq_scale_srd(br_srd, br_soff_adj, nxt_scale);
        }

        // Extract A0/Bl operands for this iteration
        fp4_intx4_t a0l = fp4_lo4(tA0[0]), a1l = fp4_lo4(tA0[1]);
        fp4_intx4_t a0h = fp4_hi4(tA0[0]), a1h = fp4_hi4(tA0[1]);
        fp4_intx4_t bl0l = fp4_lo4(tBl[0]), bl1l = fp4_lo4(tBl[1]);
        fp4_intx4_t bl0h = fp4_hi4(tBl[0]), bl1h = fp4_hi4(tBl[1]);
        unsigned sa0_v = std::bit_cast<unsigned>(a0_raw);
        unsigned sa1_v = std::bit_cast<unsigned>(a1_raw);
        unsigned sbl_v = std::bit_cast<unsigned>(bl_raw);
        unsigned sbr_v = std::bit_cast<unsigned>(br_raw);

        // ── Step 1: A0 x Bl (8 MFMAs) + ds_read Br ──
        float4 br_d[4];
        mfma8_with_4lds(acc_A0Bl, a0l, a1l, a0h, a1h, bl0l, bl1l, bl0h, bl1h,
                        sa0_v, sbl_v, br_d[0], br_d[1], br_d[2], br_d[3],
                        sel_br_p0, sel_br_p1);

        // Tile prefetch: A0, A1
        #pragma unroll
        for (int i = 0; i < PF_MPT; i++) emit_one_pf(pf_a0_p, i);

        asm volatile("s_waitcnt lgkmcnt(0)");
        fp4_intx8_t tBr[2];
        extract_tile_small(br_d, tBr);

        // ── Step 2: A0 x Br (8 MFMAs) + ds_read A1 ──
        float4 a1_d[4];
        {
            fp4_intx4_t br0l = fp4_lo4(tBr[0]), br1l = fp4_lo4(tBr[1]);
            fp4_intx4_t br0h = fp4_hi4(tBr[0]), br1h = fp4_hi4(tBr[1]);
            mfma8_with_4lds(acc_A0Br, a0l, a1l, a0h, a1h, br0l, br1l, br0h, br1h,
                            sa0_v, sbr_v, a1_d[0], a1_d[1], a1_d[2], a1_d[3],
                            sel_a1_p0, sel_a1_p1);
        }

        // Tile prefetch: A1
        #pragma unroll
        for (int i = 0; i < PF_MPT; i++) emit_one_pf(pf_a1_p, i);

        asm volatile("s_waitcnt lgkmcnt(0)");
        fp4_intx8_t tA1[2];
        extract_tile_small(a1_d, tA1);

        fp4_intx4_t a1_0l = fp4_lo4(tA1[0]), a1_1l = fp4_lo4(tA1[1]);
        fp4_intx4_t a1_0h = fp4_hi4(tA1[0]), a1_1h = fp4_hi4(tA1[1]);

        // vmcnt+barrier for next tile data
        asm volatile("s_waitcnt vmcnt(0)\ns_barrier\n" ::: "memory");

        // ── Step 3: A1 x Bl (8 MFMAs) + ds_read A0[nxt] ──
        float4 nxt_a0_d[4];
        mfma8_with_4lds(acc_A1Bl, a1_0l, a1_1l, a1_0h, a1_1h, bl0l, bl1l, bl0h, bl1h,
                        sa1_v, sbl_v, nxt_a0_d[0], nxt_a0_d[1], nxt_a0_d[2], nxt_a0_d[3],
                        sel_a0_p0, sel_a0_p1);

        // Tile prefetch: Bl
        #pragma unroll
        for (int i = 0; i < PF_MPT; i++) emit_one_pf(pf_bl_p, i);

        // ── Step 4: A1 x Br (8 MFMAs) + ds_read Bl[nxt] ──
        float4 nxt_bl_d[4];
        {
            fp4_intx4_t br0l = fp4_lo4(tBr[0]), br1l = fp4_lo4(tBr[1]);
            fp4_intx4_t br0h = fp4_hi4(tBr[0]), br1h = fp4_hi4(tBr[1]);
            mfma8_with_4lds(acc_A1Br, a1_0l, a1_1l, a1_0h, a1_1h, br0l, br1l, br0h, br1h,
                            sa1_v, sbr_v, nxt_bl_d[0], nxt_bl_d[1], nxt_bl_d[2], nxt_bl_d[3],
                            sel_bl_p0, sel_bl_p1);
        }

        // Tile prefetch: Br
        #pragma unroll
        for (int i = 0; i < PF_MPT; i++) emit_one_pf(pf_br_p, i);

        asm volatile("s_waitcnt lgkmcnt(0)");
        extract_tile_small(nxt_a0_d, tA0);
        extract_tile_small(nxt_bl_d, tBl);
    }

    // ═══════════ Store C ═══════════
    // 32x32 per accumulator block, 2x2 = 4 base tiles (16x16)
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

PYBIND11_MODULE(tk_128_k1024, m) {
    m.doc() = "MXFP4 128x128 tile kernel for higher occupancy";
    py::bind_function<dispatch_128tile>(m, "gemm_rcr",
        &gluon_globals::a, &gluon_globals::b,
        &gluon_globals::a_scale, &gluon_globals::b_scale,
        &gluon_globals::c);
}
