// MXFP4 128×128 tile kernel variant for higher occupancy
//
// Based on kernel_mxfp4_gluon_cpp.cpp but with:
//   BLK=128 (tile 128×128), HB=64, RBM=RBN=32 (register block 32×32)
//   4 accumulators per block (2×2 MFMAs) instead of 16 (4×4)
//   32 MFMAs per K-step (4 blocks × 4 MFMAs × 2 phases) instead of 128
//   Target: occupancy=2 via smaller register footprint
//
// Structure per K-iteration:
//   Step 1: A0×B_left  (8 MFMAs) + ds_read B_right
//   Step 2: A0×B_right (8 MFMAs) + ds_read A1
//   Step 3: A1×B_left  (8 MFMAs) + ds_read A0[next]
//   Step 4: A1×B_right (8 MFMAs) + ds_read B_left[next]

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

// Accumulator for 32×32 register block: 2×2 = 4 base tiles
struct alignas(16) gluon_acc_small {
    fp4_floatx4_t regs[4];
};

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

// ── LDS address computation for interleaved ds_read ──

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
// 8-MFMA KPAIR blocks for 32×32 register tiles (2×2 base tiles)
// ══════════════════════════════════════════════════════════
//
// For 32×32: A has 2 tile rows (A[0], A[1]), B has 2 tile rows (B[0], B[1])
// 2×2 = 4 accumulators, each phase 0 + phase 1 = 8 MFMAs total
//
// Accumulator layout (2×2):
//   acc[0] = A[0]×B[0]  (row0,col0)
//   acc[1] = A[0]×B[1]  (row0,col1)
//   acc[2] = A[1]×B[0]  (row1,col0)
//   acc[3] = A[1]×B[1]  (row1,col1)
//
// Scale: RBM=32 → 1 scale group → sa0 only, sb0 only

// ── KPAIR macros for 32×32 (4 acc + 8 A/B inputs + 2 scale inputs) ──
#define KPAIR_SETUP_SMALL() \
    fp4_intx4_t a0l=fp4_lo4(A[0]), a1l=fp4_lo4(A[1]); \
    fp4_intx4_t a0h=fp4_hi4(A[0]), a1h=fp4_hi4(A[1]); \
    fp4_intx4_t b0l=fp4_lo4(B[0]), b1l=fp4_lo4(B[1]); \
    fp4_intx4_t b0h=fp4_hi4(B[0]), b1h=fp4_hi4(B[1]); \
    unsigned sa0 = std::bit_cast<unsigned>(a_raw[0]); \
    unsigned sb0 = std::bit_cast<unsigned>(b_raw[0])

#define KPAIR_ACC_CLOBBER_SMALL \
    "+a"(acc[0]), "+a"(acc[1]), "+a"(acc[2]), "+a"(acc[3])

#define KPAIR_INPUTS_SMALL \
    "v"(a0l), "v"(a1l), \
    "v"(a0h), "v"(a1h), \
    "v"(b0l), "v"(b1l), \
    "v"(b0h), "v"(b1h), \
    "v"(sa0), "v"(sb0)

// ── 8 KPAIR MFMAs (pure, no interleave) for 32×32 ──
__device__ __forceinline__ void kpair_8mfma_pure(
    fp4_floatx4_t acc[4],
    const fp4_intx8_t A[2], const fp4_intx8_t B[2],
    const fp8e8m0_4 a_raw[1], const fp8e8m0_4 b_raw[1])
{
    KPAIR_SETUP_SMALL();
    asm volatile(
        // Phase 0: 4 MFMAs
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0, %4,  %8,  %0, %12, %13 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1, %4,  %9,  %1, %12, %13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2, %5,  %8,  %2, %12, %13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3, %5,  %9,  %3, %12, %13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Phase 1: 4 MFMAs
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0, %6,  %10, %0, %12, %13 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1, %6,  %11, %1, %12, %13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2, %7,  %10, %2, %12, %13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3, %7,  %11, %3, %12, %13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER_SMALL : KPAIR_INPUTS_SMALL);
}

// ── 8 KPAIR MFMAs + 4 ds_reads (for one tile) ──
// Interleaves ds_reads during phase 0, phase 1 pure MFMAs.
// ds_read outputs: d0..d3 (float4 for the next tile data)
__device__ __forceinline__ void kpair_8mfma_with_4lds(
    fp4_floatx4_t acc[4],
    const fp4_intx8_t A[2], const fp4_intx8_t B[2],
    const fp8e8m0_4 a_raw[1], const fp8e8m0_4 b_raw[1],
    float4 &d0, float4 &d1, float4 &d2, float4 &d3,
    uint32_t lds_p0, uint32_t lds_p1)
{
    KPAIR_SETUP_SMALL();
    asm volatile(
        // Phase 0 + 4 ds_reads interleaved
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0, %8,  %12, %0, %18, %19 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %4, %20 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1, %8,  %13, %1, %18, %19 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %5, %20 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2, %9,  %12, %2, %18, %19 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %6, %21 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3, %9,  %13, %3, %18, %19 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %7, %21 offset:2048\n"
        // Phase 1 — pure MFMAs
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0, %10, %14, %0, %18, %19 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1, %10, %15, %1, %18, %19 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2, %11, %14, %2, %18, %19 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3, %11, %15, %3, %18, %19 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER_SMALL,
          "=&v"(d0), "=&v"(d1), "=&v"(d2), "=&v"(d3)
        : KPAIR_INPUTS_SMALL,
          "v"(a0l), "v"(a1l),   // %14,%15 — redundant but needed for phase1 a_hi
          "v"(b0l), "v"(b1l),   // %16,%17 — redundant but needed for phase1 b_hi
          "v"(sa0), "v"(sb0),   // %18,%19
          "v"(lds_p0), "v"(lds_p1)
    );
}

// Let me redo this more carefully. The operand numbering in inline asm is tricky.
// For the 8-MFMA blocks, let me use a cleaner approach.

// ── Merged Steps 1+2: 16 MFMAs + 4 ds_reads for Br + 4 ds_reads for A1 ──
// Step 1: A0×Bl (8 MFMAs) + ds_read Br (4 reads)
// Step 2: A0×Br (8 MFMAs) + ds_read A1 (4 reads)
__device__ __forceinline__ void kpair_16mfma_step12(
    fp4_floatx4_t acc_bl[4], fp4_floatx4_t acc_br[4],
    const fp4_intx8_t A0[2], const fp4_intx8_t Bl[2],
    const fp8e8m0_4 a_raw[1], const fp8e8m0_4 bl_raw[1], const fp8e8m0_4 br_raw[1],
    float4 br_d[4], float4 a1_d[4],
    uint32_t br_p0, uint32_t br_p1,
    uint32_t a1_p0, uint32_t a1_p1)
{
    fp4_intx4_t a0l=fp4_lo4(A0[0]), a1l=fp4_lo4(A0[1]);
    fp4_intx4_t a0h=fp4_hi4(A0[0]), a1h=fp4_hi4(A0[1]);
    fp4_intx4_t b0l=fp4_lo4(Bl[0]), b1l=fp4_lo4(Bl[1]);
    fp4_intx4_t b0h=fp4_hi4(Bl[0]), b1h=fp4_hi4(Bl[1]);
    unsigned sa0 = std::bit_cast<unsigned>(a_raw[0]);
    unsigned sb_bl0 = std::bit_cast<unsigned>(bl_raw[0]);
    unsigned sb_br0 = std::bit_cast<unsigned>(br_raw[0]);

    asm volatile(
        // ═══ STEP 1: A0×Bl (8 MFMAs) + 4 ds_reads for Br ═══
        // Phase 0 + ds_reads
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %12, %16, %0,  %20, %21 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %8, %24 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %12, %17, %1,  %20, %21 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %9, %24 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %13, %16, %2,  %20, %21 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %10, %25 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %13, %17, %3,  %20, %21 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %11, %25 offset:2048\n"
        // Phase 1 — pure MFMAs
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %14, %18, %0,  %20, %21 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %14, %19, %1,  %20, %21 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %15, %18, %2,  %20, %21 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %15, %19, %3,  %20, %21 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Wait for Br ds_reads
        "s_waitcnt lgkmcnt(0)\n"
        // ═══ STEP 2: A0×Br (8 MFMAs) + 4 ds_reads for A1 ═══
        // Phase 0 + ds_reads
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %12, %8,  %4,  %20, %22 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %23, %26 offset:0\n"       // reuse %23 temp — NO, need separate outputs
        // Actually we need dedicated a1_d outputs. Let me restructure.
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %12, %9,  %5,  %20, %22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %13, %8,  %6,  %20, %22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %13, %9,  %7,  %20, %22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Phase 1 — pure MFMAs
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %14, %10, %4,  %20, %22 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %14, %11, %5,  %20, %22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %15, %10, %6,  %20, %22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %15, %11, %7,  %20, %22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc_bl[0]), "+a"(acc_bl[1]), "+a"(acc_bl[2]), "+a"(acc_bl[3]),   // %0..3
          "+a"(acc_br[0]), "+a"(acc_br[1]), "+a"(acc_br[2]), "+a"(acc_br[3]),   // %4..7
          "=&v"(br_d[0]), "=&v"(br_d[1]), "=&v"(br_d[2]), "=&v"(br_d[3])      // %8..11
        : "v"(a0l), "v"(a1l),          // %12..13  A0 lo halves
          "v"(a0h), "v"(a1h),          // %14..15  A0 hi halves
          "v"(b0l), "v"(b1l),          // %16..17  Bl lo halves
          "v"(b0h), "v"(b1h),          // %18..19  Bl hi halves
          "v"(sa0), "v"(sb_bl0),       // %20..21  scales
          "v"(sb_br0),                 // %22      Br scale
          "v"(0u),                     // %23      padding
          "v"(br_p0), "v"(br_p1),     // %24..25  Br LDS addrs
          "v"(a1_p0), "v"(a1_p1)      // %26..27  A1 LDS addrs
    );
}

// OK this is getting messy with the mixed br_d / a1_d outputs. Let me take a simpler
// approach: separate the steps and use individual 8-MFMA functions.

// ── 8 KPAIR MFMAs + 4 ds_reads interleaved + optional pf ──
template<bool EMIT_BARRIER = false>
__device__ __forceinline__ void kpair_8mfma_with_4lds_and_pf(
    fp4_floatx4_t acc[4],
    const fp4_intx8_t A[2], const fp4_intx8_t B[2],
    const fp8e8m0_4 a_raw[1], const fp8e8m0_4 b_raw[1],
    float4 &d0, float4 &d1, float4 &d2, float4 &d3,
    uint32_t lds_p0, uint32_t lds_p1,
    const tile_pf_params *pf = nullptr, int pf_start = 0)
{
    KPAIR_SETUP_SMALL();

    if constexpr (EMIT_BARRIER) {
        asm volatile("s_waitcnt vmcnt(4)\ns_barrier\n" ::: "memory");
    }

    // Phase 0 + 4 ds_reads
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0, %8,  %12, %0, %16, %17 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %4, %18 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1, %8,  %13, %1, %16, %17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %5, %18 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2, %9,  %12, %2, %16, %17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %6, %19 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3, %9,  %13, %3, %16, %17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %7, %19 offset:2048\n"
        : KPAIR_ACC_CLOBBER_SMALL,
          "=&v"(d0), "=&v"(d1), "=&v"(d2), "=&v"(d3)
        : KPAIR_INPUTS_SMALL,
          "v"(a0l), "v"(a1l),   // redundant %14,%15
          "v"(sa0), "v"(sb0),   // redundant %16,%17
          "v"(lds_p0), "v"(lds_p1)
    );

    // Prefetch between phases
    if (pf) {
        emit_one_pf(*pf, pf_start);
        emit_one_pf(*pf, pf_start + 1);
    }

    // Phase 1 — pure MFMAs
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0, %6,  %10, %0, %12, %13 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1, %6,  %11, %1, %12, %13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2, %7,  %10, %2, %12, %13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3, %7,  %11, %3, %12, %13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER_SMALL : KPAIR_INPUTS_SMALL,
          "v"(a0h), "v"(a1h), "v"(sa0), "v"(sb0) // redundant, %10..13 actually used as a_hi, b_hi
    );
}

// OK, I'm making this too complicated with inline asm operand numbering.
// Let me step back and write a clean, correct version with simpler asm blocks.

// ══════════════════════════════════════════════════════════
// CLEAN 8-MFMA implementations for 32×32 register blocks
// ══════════════════════════════════════════════════════════

// 8 MFMAs, pure computation, no interleaving
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

// 8 MFMAs + 4 ds_reads interleaved during phase 0
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
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0, %8,  %12, %0, %20, %21 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %4, %22 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1, %8,  %13, %1, %20, %21 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %5, %22 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2, %9,  %12, %2, %20, %21 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %6, %23 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3, %9,  %13, %3, %20, %21 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %7, %23 offset:2048\n"
        // Phase 1
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0, %10, %14, %0, %20, %21 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1, %10, %15, %1, %20, %21 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2, %11, %14, %2, %20, %21 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3, %11, %15, %3, %20, %21 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc[0]), "+a"(acc[1]), "+a"(acc[2]), "+a"(acc[3]),
          "=&v"(d0), "=&v"(d1), "=&v"(d2), "=&v"(d3)
        : "v"(a0l), "v"(a1l), "v"(a0h), "v"(a1h),     // %8..11
          "v"(b0l), "v"(b1l), "v"(b0h), "v"(b1h),     // %12..15
          "v"(0u), "v"(0u), "v"(0u), "v"(0u),         // %16..19 padding
          "v"(sa), "v"(sb),                             // %20..21
          "v"(lds_p0), "v"(lds_p1)                     // %22..23
    );
}

// ══════════════════════════════════════════════════════════
// Main kernel
// ══════════════════════════════════════════════════════════

__global__ __launch_bounds__(_NUM_THREADS, 2)
void mxfp4_128tile_kernel(const gluon_globals g) {
    static_assert(K_BYTES % BK == 0 && N_DIM % BLK == 0 && M_DIM % BLK == 0);

    constexpr int bpc = N_DIM / BLK;
    constexpr int a_packs = RBM / 32;  // = 1
    constexpr int b_packs = RBN / 32;  // = 1

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

    // Scale SRDs — with RBM=32, each warp covers exactly 1 scale group (32 rows)
    const uint32_t lane_soff_x2 =
        (static_cast<uint32_t>(kittens::laneid() / 16) << 7) |
        (static_cast<uint32_t>(kittens::laneid() % 16) << 3);

    // For 128-tile: br*BLK = br*128, HB=64, RBM=32
    // A0 covers rows [br*128 .. br*128+63], warp wm covers RBM=32 rows starting at wm*32
    // A1 covers rows [br*128+64 .. br*128+127], warp wm covers RBM=32 rows starting at wm*32
    i32x4 a0_srd = make_scale_srd(preshuffled_scale_row_base_ptr(
        g.a_scale, (br * BLK + wm * RBM) >> 6));
    i32x4 a1_srd = make_scale_srd(preshuffled_scale_row_base_ptr(
        g.a_scale, (br * BLK + HB + wm * RBM) >> 6));
    i32x4 bl_srd = make_scale_srd(preshuffled_scale_row_base_ptr(
        g.b_scale, (bc * BLK + wn * RBN) >> 6));
    i32x4 br_srd = make_scale_srd(preshuffled_scale_row_base_ptr(
        g.b_scale, (bc * BLK + HB + wn * RBN) >> 6));

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

    // Pre-compute LDS addresses (as static named vars, no arrays to spill)
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

    // Tile extraction helper (ds_read float4[4] → fp4_intx8_t[2])
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

    // Prefetch first scales (a_packs=1, b_packs=1 → single scale each)
    fp8e8m0_4 pf_a0[1], pf_a1[1], pf_bl[1], pf_br[1];
    {
        pf_a0[0] = load_pq_scale_srd(a0_srd, lane_soff_x2, 0);
        pf_a1[0] = load_pq_scale_srd(a1_srd, lane_soff_x2, 0);
        pf_bl[0] = load_pq_scale_srd(bl_srd, lane_soff_x2, 0);
        pf_br[0] = load_pq_scale_srd(br_srd, lane_soff_x2, 0);
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

        fp8e8m0_4 a0_raw[1], a1_raw[1], bl_raw[1], br_raw[1];
        a0_raw[0] = pf_a0[0]; a1_raw[0] = pf_a1[0];
        bl_raw[0] = pf_bl[0]; br_raw[0] = pf_br[0];

        // Prefetch next scales
        {
            const uint32_t nxt_scale = static_cast<uint32_t>(bt + 1 < k_byte_iters ? bt + 1 : bt) << 9;
            pf_a0[0] = load_pq_scale_srd(a0_srd, lane_soff_x2, nxt_scale);
            pf_a1[0] = load_pq_scale_srd(a1_srd, lane_soff_x2, nxt_scale);
            pf_bl[0] = load_pq_scale_srd(bl_srd, lane_soff_x2, nxt_scale);
            pf_br[0] = load_pq_scale_srd(br_srd, lane_soff_x2, nxt_scale);
        }

        // Extract operands once for use across steps
        fp4_intx4_t a0l = fp4_lo4(tA0[0]), a1l = fp4_lo4(tA0[1]);
        fp4_intx4_t a0h = fp4_hi4(tA0[0]), a1h = fp4_hi4(tA0[1]);
        fp4_intx4_t bl0l = fp4_lo4(tBl[0]), bl1l = fp4_lo4(tBl[1]);
        fp4_intx4_t bl0h = fp4_hi4(tBl[0]), bl1h = fp4_hi4(tBl[1]);
        unsigned sa0_v = std::bit_cast<unsigned>(a0_raw[0]);
        unsigned sbl_v = std::bit_cast<unsigned>(bl_raw[0]);
        unsigned sbr_v = std::bit_cast<unsigned>(br_raw[0]);

        // ── Step 1: A0×Bl (8 MFMAs) + ds_read Br ──
        float4 br_d[4];
        mfma8_with_4lds(acc_A0Bl, a0l, a1l, a0h, a1h, bl0l, bl1l, bl0h, bl1h,
                        sa0_v, sbl_v, br_d[0], br_d[1], br_d[2], br_d[3],
                        sel_br_p0, sel_br_p1);

        // Prefetch tiles A0, A1
        #pragma unroll
        for (int i = 0; i < PF_MPT; i++) emit_one_pf(pf_a0_p, i);

        asm volatile("s_waitcnt lgkmcnt(0)");
        fp4_intx8_t tBr[2];
        extract_tile_small(br_d, tBr);

        // ── Step 2: A0×Br (8 MFMAs) + ds_read A1 ──
        float4 a1_d[4];
        {
            fp4_intx4_t br0l = fp4_lo4(tBr[0]), br1l = fp4_lo4(tBr[1]);
            fp4_intx4_t br0h = fp4_hi4(tBr[0]), br1h = fp4_hi4(tBr[1]);
            mfma8_with_4lds(acc_A0Br, a0l, a1l, a0h, a1h, br0l, br1l, br0h, br1h,
                            sa0_v, sbr_v, a1_d[0], a1_d[1], a1_d[2], a1_d[3],
                            sel_a1_p0, sel_a1_p1);
        }

        // Prefetch tiles A1
        #pragma unroll
        for (int i = 0; i < PF_MPT; i++) emit_one_pf(pf_a1_p, i);

        asm volatile("s_waitcnt lgkmcnt(0)");
        fp4_intx8_t tA1[2];
        extract_tile_small(a1_d, tA1);

        fp4_intx4_t a1_0l = fp4_lo4(tA1[0]), a1_1l = fp4_lo4(tA1[1]);
        fp4_intx4_t a1_0h = fp4_hi4(tA1[0]), a1_1h = fp4_hi4(tA1[1]);
        unsigned sa1_v = std::bit_cast<unsigned>(a1_raw[0]);

        // vmcnt+barrier for next tiles
        asm volatile("s_waitcnt vmcnt(0)\ns_barrier\n" ::: "memory");

        // ── Step 3: A1×Bl (8 MFMAs) + ds_read A0[nxt] ──
        float4 nxt_a0_d[4];
        mfma8_with_4lds(acc_A1Bl, a1_0l, a1_1l, a1_0h, a1_1h, bl0l, bl1l, bl0h, bl1h,
                        sa1_v, sbl_v, nxt_a0_d[0], nxt_a0_d[1], nxt_a0_d[2], nxt_a0_d[3],
                        sel_a0_p0, sel_a0_p1);

        // Prefetch tiles Bl
        #pragma unroll
        for (int i = 0; i < PF_MPT; i++) emit_one_pf(pf_bl_p, i);

        // ── Step 4: A1×Br (8 MFMAs) + ds_read Bl[nxt] ──
        float4 nxt_bl_d[4];
        {
            fp4_intx4_t br0l = fp4_lo4(tBr[0]), br1l = fp4_lo4(tBr[1]);
            fp4_intx4_t br0h = fp4_hi4(tBr[0]), br1h = fp4_hi4(tBr[1]);
            mfma8_with_4lds(acc_A1Br, a1_0l, a1_1l, a1_0h, a1_1h, br0l, br1l, br0h, br1h,
                            sa1_v, sbr_v, nxt_bl_d[0], nxt_bl_d[1], nxt_bl_d[2], nxt_bl_d[3],
                            sel_bl_p0, sel_bl_p1);
        }

        // Prefetch tiles Br
        #pragma unroll
        for (int i = 0; i < PF_MPT; i++) emit_one_pf(pf_br_p, i);

        asm volatile("s_waitcnt lgkmcnt(0)");
        extract_tile_small(nxt_a0_d, tA0);
        extract_tile_small(nxt_bl_d, tBl);
    }

    // ═══════════ Store C ═══════════
    // 32×32 per accumulator block, 2×2 = 4 base tiles
    auto store_block = [&](const fp4_floatx4_t acc[4], int mh, int nh) {
        const int lid = kittens::laneid();
        // With BLK=128, HB=64, WARPS_M=2: each warp covers 32 rows
        // mh=0: rows [0..63], mh=1: rows [64..127]
        // Within each half: wm=0 → rows [0..31], wm=1 → rows [32..63]
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
