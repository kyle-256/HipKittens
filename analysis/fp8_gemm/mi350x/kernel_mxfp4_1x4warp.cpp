// MXFP4 1x4 warp layout kernel: WARPS_M=1, WARPS_N=4
//
// Based on kernel_mxfp4_gluon_cpp.cpp but with different warp decomposition:
//   - Each warp covers 128 M-rows x 32 N-columns per tile-half
//   - A tile per warp: 128x128 (8 subtiles of 16 rows)
//   - B tile per warp: 32x128 (2 subtiles of 16 rows)
//   - Accumulator per block: 8x2 = 16 MFMA tiles x 4 AGPRs = 64 AGPRs
//   - 4 accumulator blocks total: A0xBl, A0xBr, A1xBl, A1xBr = 256 AGPRs
//   - 32 MFMAs per block (8 A-sub x 2 B-sub x 2 phases)
//
// Key differences from 2x2 layout:
//   - A has 8 subtile rows -> 4 scale groups (sa0..sa3) instead of 2
//   - B has 2 subtile rows -> 1 scale group (sb0) instead of 2
//   - ds_reads per A tile: 16 (was 8); per B tile: 4 (was 8)
//   - All 4 warps share the same A data -> better L2 reuse
//   - Each warp has unique B data -> may help N-direction parallelism

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

constexpr int BLK = 256;
constexpr int BK  = 128;
constexpr int WARPS_M = 1, WARPS_N = 4;
constexpr int _NUM_WARPS   = WARPS_M * WARPS_N;
constexpr int _NUM_THREADS = _NUM_WARPS * WARP_THREADS;
constexpr int HB = BLK / 2;       // 128
constexpr int RBM = HB / WARPS_M; // 128
constexpr int RBN = HB / WARPS_N; // 32

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

// -- LDS -> register tile load --

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

// -- Scale helpers --

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

// -- Tile prefetch --

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

// -- LDS address computation --

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

// -- Tile prefetch params --

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

// ================================================================
// KPAIR functions for 8x2 grid (8 A-subtiles x 2 B-subtiles)
// ================================================================
//
// Accumulator layout: acc[i*2+j] where i=0..7 (A subtile), j=0..1 (B subtile)
// Scale layout: 4 A scales (sa0..sa3), 1 B scale (sb0)
//
// MFMA pattern per row pair (2 consecutive A subtiles x 2 B subtiles):
//   Phase 0: A_even x B0, A_even x B1, A_odd x B0, A_odd x B1
//   Phase 1: same with hi halves
//   = 8 MFMAs per row pair, 4 row pairs = 32 MFMAs total

// -- Pure 32 MFMAs (no interleaved ops) --
__device__ __forceinline__ void kpair_32mfma_pure_8x2(
    fp4_floatx4_t acc[16],
    const fp4_intx8_t A[8], const fp4_intx8_t B[2],
    const fp8e8m0_4 a_raw[4], const fp8e8m0_4 b_raw[1])
{
    fp4_intx4_t a0l=fp4_lo4(A[0]), a1l=fp4_lo4(A[1]), a2l=fp4_lo4(A[2]), a3l=fp4_lo4(A[3]);
    fp4_intx4_t a4l=fp4_lo4(A[4]), a5l=fp4_lo4(A[5]), a6l=fp4_lo4(A[6]), a7l=fp4_lo4(A[7]);
    fp4_intx4_t a0h=fp4_hi4(A[0]), a1h=fp4_hi4(A[1]), a2h=fp4_hi4(A[2]), a3h=fp4_hi4(A[3]);
    fp4_intx4_t a4h=fp4_hi4(A[4]), a5h=fp4_hi4(A[5]), a6h=fp4_hi4(A[6]), a7h=fp4_hi4(A[7]);
    fp4_intx4_t b0l=fp4_lo4(B[0]), b1l=fp4_lo4(B[1]);
    fp4_intx4_t b0h=fp4_hi4(B[0]), b1h=fp4_hi4(B[1]);
    unsigned sa0 = std::bit_cast<unsigned>(a_raw[0]);
    unsigned sa1 = std::bit_cast<unsigned>(a_raw[1]);
    unsigned sa2 = std::bit_cast<unsigned>(a_raw[2]);
    unsigned sa3 = std::bit_cast<unsigned>(a_raw[3]);
    unsigned sb0 = std::bit_cast<unsigned>(b_raw[0]);

    // Row pair 0: A0,A1 x B0,B1 (scale: sa0, sb0)
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %4,  %8,  %0,  %12, %13 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %4,  %9,  %1,  %12, %13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %5,  %8,  %2,  %12, %13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %5,  %9,  %3,  %12, %13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %6,  %10, %0,  %12, %13 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %6,  %11, %1,  %12, %13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %7,  %10, %2,  %12, %13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %7,  %11, %3,  %12, %13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc[0]), "+a"(acc[1]), "+a"(acc[2]), "+a"(acc[3])
        : "v"(a0l), "v"(a1l), "v"(a0h), "v"(a1h),
          "v"(b0l), "v"(b1l), "v"(b0h), "v"(b1h),
          "v"(sa0), "v"(sb0));

    // Row pair 1: A2,A3 x B0,B1 (scale: sa1, sb0)
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %4,  %8,  %0,  %12, %13 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %4,  %9,  %1,  %12, %13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %5,  %8,  %2,  %12, %13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %5,  %9,  %3,  %12, %13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %6,  %10, %0,  %12, %13 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %6,  %11, %1,  %12, %13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %7,  %10, %2,  %12, %13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %7,  %11, %3,  %12, %13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc[4]), "+a"(acc[5]), "+a"(acc[6]), "+a"(acc[7])
        : "v"(a2l), "v"(a3l), "v"(a2h), "v"(a3h),
          "v"(b0l), "v"(b1l), "v"(b0h), "v"(b1h),
          "v"(sa1), "v"(sb0));

    // Row pair 2: A4,A5 x B0,B1 (scale: sa2, sb0)
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %4,  %8,  %0,  %12, %13 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %4,  %9,  %1,  %12, %13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %5,  %8,  %2,  %12, %13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %5,  %9,  %3,  %12, %13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %6,  %10, %0,  %12, %13 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %6,  %11, %1,  %12, %13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %7,  %10, %2,  %12, %13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %7,  %11, %3,  %12, %13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc[8]), "+a"(acc[9]), "+a"(acc[10]), "+a"(acc[11])
        : "v"(a4l), "v"(a5l), "v"(a4h), "v"(a5h),
          "v"(b0l), "v"(b1l), "v"(b0h), "v"(b1h),
          "v"(sa2), "v"(sb0));

    // Row pair 3: A6,A7 x B0,B1 (scale: sa3, sb0)
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %4,  %8,  %0,  %12, %13 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %4,  %9,  %1,  %12, %13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %5,  %8,  %2,  %12, %13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %5,  %9,  %3,  %12, %13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %6,  %10, %0,  %12, %13 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %6,  %11, %1,  %12, %13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %7,  %10, %2,  %12, %13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %7,  %11, %3,  %12, %13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc[12]), "+a"(acc[13]), "+a"(acc[14]), "+a"(acc[15])
        : "v"(a6l), "v"(a7l), "v"(a6h), "v"(a7h),
          "v"(b0l), "v"(b1l), "v"(b0h), "v"(b1h),
          "v"(sa3), "v"(sb0));
}

// -- Step 1+2 merged: 64 MFMAs (A0xBl + A0xBr) + ds_reads for Br (4) + A1 (16) --
// This is the big merged function analogous to kpair_64mfma_step12 in the 2x2 kernel.
// Step 1: A0 x Bl (32 MFMAs) + 4 ds_reads for Br
// Step 2: A0 x Br (32 MFMAs) + 16 ds_reads for A1
//
// Br ds_reads: br_d[0..3], layout: [subtile0_p0, subtile1_p0, subtile0_p1, subtile1_p1]
// A1 ds_reads: a1_d[0..15], layout: [s0_p0..s7_p0, s0_p1..s7_p1]

__device__ __forceinline__ void kpair_step12_8x2(
    fp4_floatx4_t acc_bl[16], fp4_floatx4_t acc_br[16],
    const fp4_intx8_t A0[8], const fp4_intx8_t Bl[2],
    const fp8e8m0_4 a_raw[4], const fp8e8m0_4 bl_raw[1], const fp8e8m0_4 br_raw[1],
    float4 br_d[4], float4 a1_d[16],
    uint32_t br_p0, uint32_t br_p1,
    uint32_t a1_p0, uint32_t a1_p1)
{
    fp4_intx4_t a0l=fp4_lo4(A0[0]), a1l=fp4_lo4(A0[1]), a2l=fp4_lo4(A0[2]), a3l=fp4_lo4(A0[3]);
    fp4_intx4_t a4l=fp4_lo4(A0[4]), a5l=fp4_lo4(A0[5]), a6l=fp4_lo4(A0[6]), a7l=fp4_lo4(A0[7]);
    fp4_intx4_t a0h=fp4_hi4(A0[0]), a1h=fp4_hi4(A0[1]), a2h=fp4_hi4(A0[2]), a3h=fp4_hi4(A0[3]);
    fp4_intx4_t a4h=fp4_hi4(A0[4]), a5h=fp4_hi4(A0[5]), a6h=fp4_hi4(A0[6]), a7h=fp4_hi4(A0[7]);
    fp4_intx4_t b0l=fp4_lo4(Bl[0]), b1l=fp4_lo4(Bl[1]);
    fp4_intx4_t b0h=fp4_hi4(Bl[0]), b1h=fp4_hi4(Bl[1]);
    unsigned sa0 = std::bit_cast<unsigned>(a_raw[0]);
    unsigned sa1 = std::bit_cast<unsigned>(a_raw[1]);
    unsigned sa2 = std::bit_cast<unsigned>(a_raw[2]);
    unsigned sa3 = std::bit_cast<unsigned>(a_raw[3]);
    unsigned sb_bl = std::bit_cast<unsigned>(bl_raw[0]);
    unsigned sb_br = std::bit_cast<unsigned>(br_raw[0]);

    // === STEP 1: A0 x Bl (32 MFMAs) + 4 ds_reads for Br ===
    // Row pair 0 + Br subtile0,1 phase0
    // Outputs: %0..3=acc, %4..5=ds_read. Inputs start at %6.
    // %6,%7=A_even_lo,A_odd_lo  %8,%9=A_even_hi,A_odd_hi
    // %10,%11=B0_lo,B1_lo  %12,%13=B0_hi,B1_hi  %14,%15=sa,sb  %16=lds
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %6,  %10, %0,  %14, %15 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %4, %16 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %6,  %11, %1,  %14, %15 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %5, %16 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %7,  %10, %2,  %14, %15 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %7,  %11, %3,  %14, %15 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %8,  %12, %0,  %14, %15 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %8,  %13, %1,  %14, %15 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %9,  %12, %2,  %14, %15 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %9,  %13, %3,  %14, %15 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc_bl[0]), "+a"(acc_bl[1]), "+a"(acc_bl[2]), "+a"(acc_bl[3]),
          "=&v"(br_d[0]), "=&v"(br_d[1])
        : "v"(a0l), "v"(a1l), "v"(a0h), "v"(a1h),
          "v"(b0l), "v"(b1l), "v"(b0h), "v"(b1h),
          "v"(sa0), "v"(sb_bl),
          "v"(br_p0));

    // Row pair 1 + Br subtile0,1 phase1
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %6,  %10, %0,  %14, %15 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %4, %16 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %6,  %11, %1,  %14, %15 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %5, %16 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %7,  %10, %2,  %14, %15 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %7,  %11, %3,  %14, %15 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %8,  %12, %0,  %14, %15 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %8,  %13, %1,  %14, %15 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %9,  %12, %2,  %14, %15 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %9,  %13, %3,  %14, %15 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc_bl[4]), "+a"(acc_bl[5]), "+a"(acc_bl[6]), "+a"(acc_bl[7]),
          "=&v"(br_d[2]), "=&v"(br_d[3])
        : "v"(a2l), "v"(a3l), "v"(a2h), "v"(a3h),
          "v"(b0l), "v"(b1l), "v"(b0h), "v"(b1h),
          "v"(sa1), "v"(sb_bl),
          "v"(br_p1));

    // Row pair 2 (pure, no reads) — inputs start at %4
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %4,  %8,  %0,  %12, %13 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %4,  %9,  %1,  %12, %13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %5,  %8,  %2,  %12, %13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %5,  %9,  %3,  %12, %13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %6,  %10, %0,  %12, %13 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %6,  %11, %1,  %12, %13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %7,  %10, %2,  %12, %13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %7,  %11, %3,  %12, %13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc_bl[8]), "+a"(acc_bl[9]), "+a"(acc_bl[10]), "+a"(acc_bl[11])
        : "v"(a4l), "v"(a5l), "v"(a4h), "v"(a5h),
          "v"(b0l), "v"(b1l), "v"(b0h), "v"(b1h),
          "v"(sa2), "v"(sb_bl));

    // Row pair 3 (pure)
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %4,  %8,  %0,  %12, %13 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %4,  %9,  %1,  %12, %13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %5,  %8,  %2,  %12, %13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %5,  %9,  %3,  %12, %13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %6,  %10, %0,  %12, %13 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %6,  %11, %1,  %12, %13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %7,  %10, %2,  %12, %13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %7,  %11, %3,  %12, %13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc_bl[12]), "+a"(acc_bl[13]), "+a"(acc_bl[14]), "+a"(acc_bl[15])
        : "v"(a6l), "v"(a7l), "v"(a6h), "v"(a7h),
          "v"(b0l), "v"(b1l), "v"(b0h), "v"(b1h),
          "v"(sa3), "v"(sb_bl));

    // Wait for Br reads, extract Br tiles
    asm volatile("s_waitcnt lgkmcnt(0)");
    fp4_intx4_t br0l, br1l, br0h, br1h;
    {
        auto lo0 = *reinterpret_cast<const fp4_intx4_t*>(&br_d[0]);
        auto lo1 = *reinterpret_cast<const fp4_intx4_t*>(&br_d[1]);
        auto hi0 = *reinterpret_cast<const fp4_intx4_t*>(&br_d[2]);
        auto hi1 = *reinterpret_cast<const fp4_intx4_t*>(&br_d[3]);
        br0l = lo0; br1l = lo1; br0h = hi0; br1h = hi1;
    }

    // === STEP 2: A0 x Br (32 MFMAs) + 16 ds_reads for A1 ===
    // Each block: 4 acc (+a) + 4 ds_read (=&v) = 8 outputs, inputs start at %8
    // %8,%9=A_even_lo,A_odd_lo  %10,%11=A_even_hi,A_odd_hi
    // %12,%13=B0_lo,B1_lo  %14,%15=B0_hi,B1_hi  %16,%17=sa,sb  %18,%19=lds_p0,lds_p1

    // Row pair 0 + A1 subtiles 0,1 at phase0 and phase1
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %8,  %12, %0,  %16, %17 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %4, %18 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %8,  %13, %1,  %16, %17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %5, %18 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %9,  %12, %2,  %16, %17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %6, %19 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %9,  %13, %3,  %16, %17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %7, %19 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %10, %14, %0,  %16, %17 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %10, %15, %1,  %16, %17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %11, %14, %2,  %16, %17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %11, %15, %3,  %16, %17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc_br[0]), "+a"(acc_br[1]), "+a"(acc_br[2]), "+a"(acc_br[3]),
          "=&v"(a1_d[0]), "=&v"(a1_d[1]), "=&v"(a1_d[8]), "=&v"(a1_d[9])
        : "v"(a0l), "v"(a1l), "v"(a0h), "v"(a1h),
          "v"(br0l), "v"(br1l), "v"(br0h), "v"(br1h),
          "v"(sa0), "v"(sb_br),
          "v"(a1_p0), "v"(a1_p1));

    // Row pair 1 + A1 subtiles 2,3 at phase0 and phase1
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %8,  %12, %0,  %16, %17 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %4, %18 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %8,  %13, %1,  %16, %17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %5, %18 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %9,  %12, %2,  %16, %17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %6, %19 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %9,  %13, %3,  %16, %17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %7, %19 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %10, %14, %0,  %16, %17 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %10, %15, %1,  %16, %17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %11, %14, %2,  %16, %17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %11, %15, %3,  %16, %17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc_br[4]), "+a"(acc_br[5]), "+a"(acc_br[6]), "+a"(acc_br[7]),
          "=&v"(a1_d[2]), "=&v"(a1_d[3]), "=&v"(a1_d[10]), "=&v"(a1_d[11])
        : "v"(a2l), "v"(a3l), "v"(a2h), "v"(a3h),
          "v"(br0l), "v"(br1l), "v"(br0h), "v"(br1h),
          "v"(sa1), "v"(sb_br),
          "v"(a1_p0), "v"(a1_p1));

    // Row pair 2 + A1 subtiles 4,5 at phase0 and phase1
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %8,  %12, %0,  %16, %17 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %4, %18 offset:8192\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %8,  %13, %1,  %16, %17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %5, %18 offset:10240\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %9,  %12, %2,  %16, %17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %6, %19 offset:8192\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %9,  %13, %3,  %16, %17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %7, %19 offset:10240\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %10, %14, %0,  %16, %17 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %10, %15, %1,  %16, %17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %11, %14, %2,  %16, %17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %11, %15, %3,  %16, %17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc_br[8]), "+a"(acc_br[9]), "+a"(acc_br[10]), "+a"(acc_br[11]),
          "=&v"(a1_d[4]), "=&v"(a1_d[5]), "=&v"(a1_d[12]), "=&v"(a1_d[13])
        : "v"(a4l), "v"(a5l), "v"(a4h), "v"(a5h),
          "v"(br0l), "v"(br1l), "v"(br0h), "v"(br1h),
          "v"(sa2), "v"(sb_br),
          "v"(a1_p0), "v"(a1_p1));

    // Row pair 3 + A1 subtiles 6,7 at phase0 and phase1
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %8,  %12, %0,  %16, %17 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %4, %18 offset:12288\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %8,  %13, %1,  %16, %17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %5, %18 offset:14336\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %9,  %12, %2,  %16, %17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %6, %19 offset:12288\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %9,  %13, %3,  %16, %17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %7, %19 offset:14336\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %10, %14, %0,  %16, %17 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %10, %15, %1,  %16, %17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %11, %14, %2,  %16, %17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %11, %15, %3,  %16, %17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc_br[12]), "+a"(acc_br[13]), "+a"(acc_br[14]), "+a"(acc_br[15]),
          "=&v"(a1_d[6]), "=&v"(a1_d[7]), "=&v"(a1_d[14]), "=&v"(a1_d[15])
        : "v"(a6l), "v"(a7l), "v"(a6h), "v"(a7h),
          "v"(br0l), "v"(br1l), "v"(br0h), "v"(br1h),
          "v"(sa3), "v"(sb_br),
          "v"(a1_p0), "v"(a1_p1));
}

// -- Step 3: 32 MFMAs + 16 ds_reads for A0[nxt] + barrier at entry --
template<bool EMIT_BARRIER = false>
__device__ __forceinline__ void kpair_32mfma_with_16lds_and_pf_8x2(
    fp4_floatx4_t acc[16],
    const fp4_intx8_t A[8], const fp4_intx8_t B[2],
    const fp8e8m0_4 a_raw[4], const fp8e8m0_4 b_raw[1],
    float4 d[16],
    uint32_t lds_p0, uint32_t lds_p1,
    const tile_pf_params &pf0, const tile_pf_params &pf1)
{
    fp4_intx4_t a0l=fp4_lo4(A[0]), a1l=fp4_lo4(A[1]), a2l=fp4_lo4(A[2]), a3l=fp4_lo4(A[3]);
    fp4_intx4_t a4l=fp4_lo4(A[4]), a5l=fp4_lo4(A[5]), a6l=fp4_lo4(A[6]), a7l=fp4_lo4(A[7]);
    fp4_intx4_t a0h=fp4_hi4(A[0]), a1h=fp4_hi4(A[1]), a2h=fp4_hi4(A[2]), a3h=fp4_hi4(A[3]);
    fp4_intx4_t a4h=fp4_hi4(A[4]), a5h=fp4_hi4(A[5]), a6h=fp4_hi4(A[6]), a7h=fp4_hi4(A[7]);
    fp4_intx4_t b0l=fp4_lo4(B[0]), b1l=fp4_lo4(B[1]);
    fp4_intx4_t b0h=fp4_hi4(B[0]), b1h=fp4_hi4(B[1]);
    unsigned sa0 = std::bit_cast<unsigned>(a_raw[0]);
    unsigned sa1 = std::bit_cast<unsigned>(a_raw[1]);
    unsigned sa2 = std::bit_cast<unsigned>(a_raw[2]);
    unsigned sa3 = std::bit_cast<unsigned>(a_raw[3]);
    unsigned sb0 = std::bit_cast<unsigned>(b_raw[0]);

    if constexpr (EMIT_BARRIER) {
        asm volatile("s_waitcnt vmcnt(0)\ns_barrier\n" ::: "memory");
    }

    // Each block: 4 acc (+a) %0..3, 4 ds_read (=&v) %4..7, inputs start at %8
    // %8,%9=A_even_lo,A_odd_lo  %10,%11=A_even_hi,A_odd_hi
    // %12,%13=B0_lo,B1_lo  %14,%15=B0_hi,B1_hi  %16,%17=sa,sb  %18,%19=lds_p0,lds_p1

    // Row pair 0 + ds_read A subtiles 0,1 at both phases
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %8,  %12, %0,  %16, %17 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %4, %18 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %8,  %13, %1,  %16, %17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %5, %18 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %9,  %12, %2,  %16, %17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %6, %19 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %9,  %13, %3,  %16, %17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %7, %19 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %10, %14, %0,  %16, %17 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %10, %15, %1,  %16, %17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %11, %14, %2,  %16, %17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %11, %15, %3,  %16, %17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc[0]), "+a"(acc[1]), "+a"(acc[2]), "+a"(acc[3]),
          "=&v"(d[0]), "=&v"(d[1]), "=&v"(d[8]), "=&v"(d[9])
        : "v"(a0l), "v"(a1l), "v"(a0h), "v"(a1h),
          "v"(b0l), "v"(b1l), "v"(b0h), "v"(b1h),
          "v"(sa0), "v"(sb0),
          "v"(lds_p0), "v"(lds_p1));
    emit_one_pf(pf0, 0);
    emit_one_pf(pf0, 1);

    // Row pair 1 + ds_read A subtiles 2,3 at both phases
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %8,  %12, %0,  %16, %17 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %4, %18 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %8,  %13, %1,  %16, %17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %5, %18 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %9,  %12, %2,  %16, %17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %6, %19 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %9,  %13, %3,  %16, %17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %7, %19 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %10, %14, %0,  %16, %17 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %10, %15, %1,  %16, %17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %11, %14, %2,  %16, %17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %11, %15, %3,  %16, %17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc[4]), "+a"(acc[5]), "+a"(acc[6]), "+a"(acc[7]),
          "=&v"(d[2]), "=&v"(d[3]), "=&v"(d[10]), "=&v"(d[11])
        : "v"(a2l), "v"(a3l), "v"(a2h), "v"(a3h),
          "v"(b0l), "v"(b1l), "v"(b0h), "v"(b1h),
          "v"(sa1), "v"(sb0),
          "v"(lds_p0), "v"(lds_p1));
    emit_one_pf(pf0, 2);
    emit_one_pf(pf0, 3);

    // Row pair 2 + ds_read A subtiles 4,5 at both phases
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %8,  %12, %0,  %16, %17 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %4, %18 offset:8192\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %8,  %13, %1,  %16, %17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %5, %18 offset:10240\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %9,  %12, %2,  %16, %17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %6, %19 offset:8192\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %9,  %13, %3,  %16, %17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %7, %19 offset:10240\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %10, %14, %0,  %16, %17 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %10, %15, %1,  %16, %17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %11, %14, %2,  %16, %17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %11, %15, %3,  %16, %17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc[8]), "+a"(acc[9]), "+a"(acc[10]), "+a"(acc[11]),
          "=&v"(d[4]), "=&v"(d[5]), "=&v"(d[12]), "=&v"(d[13])
        : "v"(a4l), "v"(a5l), "v"(a4h), "v"(a5h),
          "v"(b0l), "v"(b1l), "v"(b0h), "v"(b1h),
          "v"(sa2), "v"(sb0),
          "v"(lds_p0), "v"(lds_p1));
    emit_one_pf(pf1, 0);
    emit_one_pf(pf1, 1);

    // Row pair 3 + ds_read A subtiles 6,7 at both phases
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %8,  %12, %0,  %16, %17 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %4, %18 offset:12288\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %8,  %13, %1,  %16, %17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %5, %18 offset:14336\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %9,  %12, %2,  %16, %17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %6, %19 offset:12288\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %9,  %13, %3,  %16, %17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %7, %19 offset:14336\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %10, %14, %0,  %16, %17 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %10, %15, %1,  %16, %17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %11, %14, %2,  %16, %17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %11, %15, %3,  %16, %17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc[12]), "+a"(acc[13]), "+a"(acc[14]), "+a"(acc[15]),
          "=&v"(d[6]), "=&v"(d[7]), "=&v"(d[14]), "=&v"(d[15])
        : "v"(a6l), "v"(a7l), "v"(a6h), "v"(a7h),
          "v"(b0l), "v"(b1l), "v"(b0h), "v"(b1h),
          "v"(sa3), "v"(sb0),
          "v"(lds_p0), "v"(lds_p1));
    emit_one_pf(pf1, 2);
    emit_one_pf(pf1, 3);
}

// -- Step 4: 32 MFMAs + 4 ds_reads for Bl[nxt] + prefetches --
__device__ __forceinline__ void kpair_32mfma_with_4lds_and_pf_8x2(
    fp4_floatx4_t acc[16],
    const fp4_intx8_t A[8], const fp4_intx8_t B[2],
    const fp8e8m0_4 a_raw[4], const fp8e8m0_4 b_raw[1],
    float4 d[4],
    uint32_t lds_p0, uint32_t lds_p1,
    const tile_pf_params &pf0, const tile_pf_params &pf1)
{
    fp4_intx4_t a0l=fp4_lo4(A[0]), a1l=fp4_lo4(A[1]), a2l=fp4_lo4(A[2]), a3l=fp4_lo4(A[3]);
    fp4_intx4_t a4l=fp4_lo4(A[4]), a5l=fp4_lo4(A[5]), a6l=fp4_lo4(A[6]), a7l=fp4_lo4(A[7]);
    fp4_intx4_t a0h=fp4_hi4(A[0]), a1h=fp4_hi4(A[1]), a2h=fp4_hi4(A[2]), a3h=fp4_hi4(A[3]);
    fp4_intx4_t a4h=fp4_hi4(A[4]), a5h=fp4_hi4(A[5]), a6h=fp4_hi4(A[6]), a7h=fp4_hi4(A[7]);
    fp4_intx4_t b0l=fp4_lo4(B[0]), b1l=fp4_lo4(B[1]);
    fp4_intx4_t b0h=fp4_hi4(B[0]), b1h=fp4_hi4(B[1]);
    unsigned sa0 = std::bit_cast<unsigned>(a_raw[0]);
    unsigned sa1 = std::bit_cast<unsigned>(a_raw[1]);
    unsigned sa2 = std::bit_cast<unsigned>(a_raw[2]);
    unsigned sa3 = std::bit_cast<unsigned>(a_raw[3]);
    unsigned sb0 = std::bit_cast<unsigned>(b_raw[0]);

    // Row pair 0 + Bl subtile 0,1 phase0
    // 4 acc (+a) %0..3, 2 ds_read (=&v) %4..5, inputs start at %6
    // %6,%7=A_even_lo,A_odd_lo  %8,%9=A_even_hi,A_odd_hi
    // %10,%11=B0_lo,B1_lo  %12,%13=B0_hi,B1_hi  %14,%15=sa,sb  %16=lds
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %6,  %10, %0,  %14, %15 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %4, %16 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %6,  %11, %1,  %14, %15 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %5, %16 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %7,  %10, %2,  %14, %15 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %7,  %11, %3,  %14, %15 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %8,  %12, %0,  %14, %15 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %8,  %13, %1,  %14, %15 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %9,  %12, %2,  %14, %15 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %9,  %13, %3,  %14, %15 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc[0]), "+a"(acc[1]), "+a"(acc[2]), "+a"(acc[3]),
          "=&v"(d[0]), "=&v"(d[1])
        : "v"(a0l), "v"(a1l), "v"(a0h), "v"(a1h),
          "v"(b0l), "v"(b1l), "v"(b0h), "v"(b1h),
          "v"(sa0), "v"(sb0),
          "v"(lds_p0));
    emit_one_pf(pf0, 0);
    emit_one_pf(pf0, 1);

    // Row pair 1 + Bl subtile 0,1 phase1
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %6,  %10, %0,  %14, %15 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %4, %16 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %6,  %11, %1,  %14, %15 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %5, %16 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %7,  %10, %2,  %14, %15 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %7,  %11, %3,  %14, %15 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %8,  %12, %0,  %14, %15 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %8,  %13, %1,  %14, %15 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %9,  %12, %2,  %14, %15 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %9,  %13, %3,  %14, %15 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc[4]), "+a"(acc[5]), "+a"(acc[6]), "+a"(acc[7]),
          "=&v"(d[2]), "=&v"(d[3])
        : "v"(a2l), "v"(a3l), "v"(a2h), "v"(a3h),
          "v"(b0l), "v"(b1l), "v"(b0h), "v"(b1h),
          "v"(sa1), "v"(sb0),
          "v"(lds_p1));
    emit_one_pf(pf0, 2);
    emit_one_pf(pf0, 3);

    // Row pair 2 (pure)
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %4,  %8,  %0,  %12, %13 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %4,  %9,  %1,  %12, %13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %5,  %8,  %2,  %12, %13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %5,  %9,  %3,  %12, %13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %6,  %10, %0,  %12, %13 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %6,  %11, %1,  %12, %13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %7,  %10, %2,  %12, %13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %7,  %11, %3,  %12, %13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc[8]), "+a"(acc[9]), "+a"(acc[10]), "+a"(acc[11])
        : "v"(a4l), "v"(a5l), "v"(a4h), "v"(a5h),
          "v"(b0l), "v"(b1l), "v"(b0h), "v"(b1h),
          "v"(sa2), "v"(sb0));
    emit_one_pf(pf1, 0);
    emit_one_pf(pf1, 1);

    // Row pair 3 (pure)
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %4,  %8,  %0,  %12, %13 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %4,  %9,  %1,  %12, %13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %5,  %8,  %2,  %12, %13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %5,  %9,  %3,  %12, %13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %6,  %10, %0,  %12, %13 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %6,  %11, %1,  %12, %13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %7,  %10, %2,  %12, %13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %7,  %11, %3,  %12, %13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc[12]), "+a"(acc[13]), "+a"(acc[14]), "+a"(acc[15])
        : "v"(a6l), "v"(a7l), "v"(a6h), "v"(a7h),
          "v"(b0l), "v"(b1l), "v"(b0h), "v"(b1h),
          "v"(sa3), "v"(sb0));
    emit_one_pf(pf1, 2);
    emit_one_pf(pf1, 3);
}

// ================================================================
// Main kernel
// ================================================================

__global__ __launch_bounds__(_NUM_THREADS, 1)
void mxfp4_1x4warp_kernel(const gluon_globals g) {
    static_assert(K_BYTES % BK == 0 && N_DIM % BLK == 0 && M_DIM % BLK == 0);

    constexpr int bpc = N_DIM / BLK;
    constexpr int a_packs = RBM / 32;  // 4
    constexpr int b_packs = RBN / 32;  // 1

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
    const int wm = warpid() / WARPS_N, wn = warpid() % WARPS_N;  // wm=0 always

    uint32_t so_a[PF_MPT], so_b[PF_MPT];
    G::prefill_swizzled_offsets(A0_db[0], g.a, so_a);
    G::prefill_swizzled_offsets(Bl_db[0], g.b, so_b);

    // Scale SRDs
    const uint32_t lane_soff_x2 =
        (static_cast<uint32_t>(kittens::laneid() / 16) << 7) |
        (static_cast<uint32_t>(kittens::laneid() % 16) << 3);

    // A0: 128 rows -> 2 super-groups of 64 rows -> 2 SRDs
    i32x4 a0_lo_srd = make_scale_srd(preshuffled_scale_row_base_ptr(
        g.a_scale, (br * BLK) >> 6));
    i32x4 a0_hi_srd = make_scale_srd(preshuffled_scale_row_base_ptr(
        g.a_scale, (br * BLK + 64) >> 6));
    // A1: next 128 rows -> 2 SRDs
    i32x4 a1_lo_srd = make_scale_srd(preshuffled_scale_row_base_ptr(
        g.a_scale, (br * BLK + HB) >> 6));
    i32x4 a1_hi_srd = make_scale_srd(preshuffled_scale_row_base_ptr(
        g.a_scale, (br * BLK + HB + 64) >> 6));
    // B: 32 rows per warp, within one 64-row super-group
    i32x4 bl_srd = make_scale_srd(preshuffled_scale_row_base_ptr(
        g.b_scale, (bc * BLK + wn * RBN) >> 6));
    i32x4 br_srd = make_scale_srd(preshuffled_scale_row_base_ptr(
        g.b_scale, (bc * BLK + HB + wn * RBN) >> 6));

    fp4_floatx4_t acc_A0Bl[16]={}, acc_A0Br[16]={}, acc_A1Bl[16]={}, acc_A1Br[16]={};

    // Tile SRDs
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

    // Pre-compute LDS addresses
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

    // Tile extraction helpers
    // A tile: 16 float4 -> 8 fp4_intx8_t (d[0..7]=phase0, d[8..15]=phase1)
    auto extract_a_tile = [](const float4 d[16], fp4_intx8_t t[8]) __attribute__((always_inline)) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            auto lo = *reinterpret_cast<const fp4_intx4_t*>(&d[i]);
            auto hi = *reinterpret_cast<const fp4_intx4_t*>(&d[i + 8]);
            t[i][0]=lo[0]; t[i][1]=lo[1]; t[i][2]=lo[2]; t[i][3]=lo[3];
            t[i][4]=hi[0]; t[i][5]=hi[1]; t[i][6]=hi[2]; t[i][7]=hi[3];
        }
    };

    // B tile: 4 float4 -> 2 fp4_intx8_t (d[0..1]=phase0, d[2..3]=phase1)
    auto extract_b_tile = [](const float4 d[4], fp4_intx8_t t[2]) __attribute__((always_inline)) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            auto lo = *reinterpret_cast<const fp4_intx4_t*>(&d[i]);
            auto hi = *reinterpret_cast<const fp4_intx4_t*>(&d[i + 2]);
            t[i][0]=lo[0]; t[i][1]=lo[1]; t[i][2]=lo[2]; t[i][3]=lo[3];
            t[i][4]=hi[0]; t[i][5]=hi[1]; t[i][6]=hi[2]; t[i][7]=hi[3];
        }
    };

    // =========== Prologue ===========
    load_tiles(0, 0);
    if (k_byte_iters > 1) load_tiles(1, 1);

    // B scale offset: in merged format, each dwordx2 contains two 32-row groups.
    // wn%2==0 uses the lo dword, wn%2==1 uses the hi dword.
    // For single-dword B scale load, add 4 bytes for the hi group.
    const uint32_t b_scale_soff_adj = (wn % 2) * 4;
    const uint32_t lane_soff_b = lane_soff_x2 + b_scale_soff_adj;

    fp8e8m0_4 pf_a0[a_packs], pf_a1[a_packs], pf_bl[b_packs], pf_br[b_packs];
    {
        load_pq_scale_x2_async(a0_lo_srd, lane_soff_x2, 0, pf_a0[0], pf_a0[1]);
        load_pq_scale_x2_async(a0_hi_srd, lane_soff_x2, 0, pf_a0[2], pf_a0[3]);
        load_pq_scale_x2_async(a1_lo_srd, lane_soff_x2, 0, pf_a1[0], pf_a1[1]);
        load_pq_scale_x2_async(a1_hi_srd, lane_soff_x2, 0, pf_a1[2], pf_a1[3]);
        pf_bl[0] = load_pq_scale_srd(bl_srd, lane_soff_b, 0);
        pf_br[0] = load_pq_scale_srd(br_srd, lane_soff_b, 0);
    }

    // Pre-load A0+Bl from LDS
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();
    A_row_reg a0_rt;
    B_row_reg bl_rt;
    fp4_load_st_to_rt(a0_rt, kittens::subtile_inplace<RBM, BK>(A0_db[0], {wm, 0}));
    fp4_load_st_to_rt(bl_rt, kittens::subtile_inplace<RBN, BK>(Bl_db[0], {wn, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    fp4_intx8_t tA0[8], tBl[2];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        tA0[i] = fp4_extract_tile(a0_rt, i);
    #pragma unroll
    for (int i = 0; i < 2; i++)
        tBl[i] = fp4_extract_tile(bl_rt, i);

    // =========== Main loop ===========
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

        fp8e8m0_4 a0_raw[a_packs], a1_raw[a_packs], bl_raw[b_packs], br_raw[b_packs];
        #pragma unroll
        for (int p = 0; p < a_packs; ++p) { a0_raw[p] = pf_a0[p]; a1_raw[p] = pf_a1[p]; }
        #pragma unroll
        for (int p = 0; p < b_packs; ++p) { bl_raw[p] = pf_bl[p]; br_raw[p] = pf_br[p]; }

        // Prefetch scales for next K-iteration
        {
            const uint32_t nxt_scale = static_cast<uint32_t>(bt + 1 < k_byte_iters ? bt + 1 : bt) << 9;
            load_pq_scale_x2_async(a0_lo_srd, lane_soff_x2, nxt_scale, pf_a0[0], pf_a0[1]);
            load_pq_scale_x2_async(a0_hi_srd, lane_soff_x2, nxt_scale, pf_a0[2], pf_a0[3]);
            load_pq_scale_x2_async(a1_lo_srd, lane_soff_x2, nxt_scale, pf_a1[0], pf_a1[1]);
            load_pq_scale_x2_async(a1_hi_srd, lane_soff_x2, nxt_scale, pf_a1[2], pf_a1[3]);
            pf_bl[0] = load_pq_scale_srd(bl_srd, lane_soff_b, nxt_scale);
            pf_br[0] = load_pq_scale_srd(br_srd, lane_soff_b, nxt_scale);
        }

        // Steps 1+2: A0xBl (32 MFMAs) + ds_read Br + A0xBr (32 MFMAs) + ds_read A1
        float4 br_d[4], a1_d[16];
        kpair_step12_8x2(acc_A0Bl, acc_A0Br, tA0, tBl,
            a0_raw, bl_raw, br_raw, br_d, a1_d,
            sel_br_p0, sel_br_p1, sel_a1_p0, sel_a1_p1);

        asm volatile("s_waitcnt lgkmcnt(0)");
        fp4_intx8_t tBr[2], tA1[8];
        extract_b_tile(br_d, tBr);
        extract_a_tile(a1_d, tA1);

        // Step 3: A1xBl (32 MFMAs) + ds_read A0[nxt] + barrier at entry + prefetches
        float4 nxt_a0_d[16];
        kpair_32mfma_with_16lds_and_pf_8x2<true>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw,
            nxt_a0_d, sel_a0_p0, sel_a0_p1, pf_a0_p, pf_a1_p);

        // Step 4: A1xBr (32 MFMAs) + ds_read Bl[nxt] + prefetches
        float4 nxt_bl_d[4];
        kpair_32mfma_with_4lds_and_pf_8x2(acc_A1Br, tA1, tBr, a1_raw, br_raw,
            nxt_bl_d, sel_bl_p0, sel_bl_p1, pf_bl_p, pf_br_p);

        asm volatile("s_waitcnt lgkmcnt(0)");
        extract_a_tile(nxt_a0_d, tA0);
        extract_b_tile(nxt_bl_d, tBl);
    }

    // =========== Store C ===========
    // 8x2 accumulator: 8 rows of 16, 2 cols of 16 = 128 x 32 per block
    auto store_block = [&](const fp4_floatx4_t acc[16], int mh, int nh) {
        const int lid = kittens::laneid();
        const int m_base = br * BLK + mh * HB;  // wm=0 always
        const int n_base = bc * BLK + nh * HB + wn * RBN;
        bf16 *dst_ptr = g.c.raw_ptr + static_cast<size_t>(m_base) * g.c.cols()
                        + static_cast<size_t>(n_base);
        const int row_stride = g.c.cols();
        const int row_off = 4 * (lid / 16);
        const int col_off = lid % 16;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
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

void dispatch_1x4warp(gluon_globals g) {
    int m = static_cast<int>(g.c.rows());
    int n = static_cast<int>(g.c.cols());
    const dim3 grid((m / BLK) * (n / BLK));
    mxfp4_1x4warp_kernel<<<grid, dim3(_NUM_THREADS), 0>>>(g);
}

PYBIND11_MODULE(tk_mxfp4_1x4warp, m) {
    m.doc() = "MXFP4 1x4 warp layout kernel";
    py::bind_function<dispatch_1x4warp>(m, "gemm_rcr",
        &gluon_globals::a, &gluon_globals::b,
        &gluon_globals::a_scale, &gluon_globals::b_scale,
        &gluon_globals::c);
}
