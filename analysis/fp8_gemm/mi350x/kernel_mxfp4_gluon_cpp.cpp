// MXFP4 Gluon-architecture kernel: C++ implementation
//
// Replicates Gluon a4w4 core algorithm:
//   - N-split: B tile → left/right halves (128 each)
//   - DOT_left / DOT_right pipeline per K-iteration
//   - Double-buffered async tile copy (buffer_load_to_lds)
//   - KPAIR MFMAs (op_sel for sub-group, op_sel_hi for K-phase, no remap_phase)
//   - Preshuffle-quant scales
//
// Per K-iteration:
//   Load A+B_left from LDS → DOT_left (A×B_left, 64 MFMAs)
//   Load B_right from LDS → DOT_right (A×B_right, 64 MFMAs)
//   Barrier → Prefetch next tiles

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
constexpr int WARPS_M = 2, WARPS_N = 2;
constexpr int _NUM_WARPS   = WARPS_M * WARPS_N;
constexpr int _NUM_THREADS = _NUM_WARPS * WARP_THREADS;
constexpr int HB = BLK / 2;
constexpr int RBM = HB / WARPS_M;
constexpr int RBN = HB / WARPS_N;

constexpr int K_BYTES = K_DIM / 2;
constexpr int k_byte_iters = K_BYTES / BK;

#ifndef STEP3_EMBED_BARRIER
#define STEP3_EMBED_BARRIER 1
#endif
#ifndef STEP3_BARRIER_VMCNT
#define STEP3_BARRIER_VMCNT 8
#endif
#ifndef STEP4_EXTERNAL_BR_PREFETCH
#define STEP4_EXTERNAL_BR_PREFETCH 0
#endif
#ifndef SWAP_STEP34_MAIN
#define SWAP_STEP34_MAIN 0
#endif
#ifndef SWAP_STEP12_MAIN
#define SWAP_STEP12_MAIN 0
#endif
#if SWAP_STEP12_MAIN && !SWAP_STEP34_MAIN
#error "SWAP_STEP12_MAIN requires SWAP_STEP34_MAIN"
#endif

#define MXFP4_STR_IMPL(x) #x
#define MXFP4_STR(x) MXFP4_STR_IMPL(x)

using ST_tile = st_fp8e4m3<HB, BK, st_16x128_s>;
using A_row_reg = rt_fp8e4m3<RBM, BK, row_l, rt_16x128_s>;
using B_row_reg = rt_fp8e4m3<RBN, BK, row_l, rt_16x128_s>;
using RT_C = rt_fl<RBM, RBN, col_l, rt_16x16_s>;

using G = kittens::group<_NUM_WARPS>;
using _gl_fp4   = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_scale = gl<fp8e8m0, -1, -1, -1, -1>;
using _gl_bf16  = gl<bf16, -1, -1, -1, -1>;
using _gl_float = gl<float, -1, -1, -1, -1>;

struct gluon_globals {
    _gl_fp4 a, b;
    _gl_scale a_scale, b_scale;
    _gl_bf16 c;
    float scale = 1.0f;
};

struct operand_swap_debug_globals {
    _gl_fp4 a, b;
    _gl_scale a_scale, b_scale;
    _gl_float out;
};

using fp4_intx8_t   = int __attribute__((__vector_size__(8 * sizeof(int))));
using fp4_intx4_t   = int __attribute__((__vector_size__(4 * sizeof(int))));
using fp4_floatx4_t = float __attribute__((__vector_size__(4 * sizeof(float))));
using u32x2_t       = unsigned int __attribute__((ext_vector_type(2)));

__device__ __forceinline__ fp4_intx4_t fp4_lo4(const fp4_intx8_t& x) {
    return __builtin_shufflevector(x, x, 0, 1, 2, 3);
}
__device__ __forceinline__ fp4_intx4_t fp4_hi4(const fp4_intx8_t& x) {
    return __builtin_shufflevector(x, x, 4, 5, 6, 7);
}

struct alignas(16) gluon_acc {
    fp4_floatx4_t regs[32]; // [0..15] = A0×B_half, [16..31] = A1×B_half
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

__device__ __forceinline__ void extract_dsread_tile(const float4 d[8], fp4_intx8_t t[4]) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        auto lo = *reinterpret_cast<const fp4_intx4_t*>(&d[i]);
        auto hi = *reinterpret_cast<const fp4_intx4_t*>(&d[i + 4]);
        t[i][0] = lo[0]; t[i][1] = lo[1]; t[i][2] = lo[2]; t[i][3] = lo[3];
        t[i][4] = hi[0]; t[i][5] = hi[1]; t[i][6] = hi[2]; t[i][7] = hi[3];
    }
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

// Load two consecutive scale dwords via buffer_load_dwordx2 (merged preshuffle format).
// Non-volatile asm allows compiler scheduling flexibility while preserving dwordx2.
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

// ── Tile prefetch params (for individual emit_one_pf calls) ──

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

// ── KPAIR shared operand setup macro ──
#define KPAIR_SETUP() \
    fp4_intx4_t a0l=fp4_lo4(A[0]), a1l=fp4_lo4(A[1]), a2l=fp4_lo4(A[2]), a3l=fp4_lo4(A[3]); \
    fp4_intx4_t a0h=fp4_hi4(A[0]), a1h=fp4_hi4(A[1]), a2h=fp4_hi4(A[2]), a3h=fp4_hi4(A[3]); \
    fp4_intx4_t b0l=fp4_lo4(B[0]), b1l=fp4_lo4(B[1]), b2l=fp4_lo4(B[2]), b3l=fp4_lo4(B[3]); \
    fp4_intx4_t b0h=fp4_hi4(B[0]), b1h=fp4_hi4(B[1]), b2h=fp4_hi4(B[2]), b3h=fp4_hi4(B[3]); \
    unsigned sa0 = std::bit_cast<unsigned>(a_raw[0]); \
    unsigned sa1 = std::bit_cast<unsigned>(a_raw[1]); \
    unsigned sb0 = std::bit_cast<unsigned>(b_raw[0]); \
    unsigned sb1 = std::bit_cast<unsigned>(b_raw[1])

// ── KPAIR constraint lists (all 16 acc + 20 A/B/scale inputs) ──
#define KPAIR_ACC_CLOBBER \
    "+a"(acc[0]), "+a"(acc[1]), "+a"(acc[2]), "+a"(acc[3]),   \
    "+a"(acc[4]), "+a"(acc[5]), "+a"(acc[6]), "+a"(acc[7]),   \
    "+a"(acc[8]), "+a"(acc[9]), "+a"(acc[10]), "+a"(acc[11]), \
    "+a"(acc[12]), "+a"(acc[13]), "+a"(acc[14]), "+a"(acc[15])

#define KPAIR_INPUTS \
    "v"(a0l), "v"(a1l), "v"(a2l), "v"(a3l), \
    "v"(a0h), "v"(a1h), "v"(a2h), "v"(a3h), \
    "v"(b0l), "v"(b1l), "v"(b2l), "v"(b3l), \
    "v"(b0h), "v"(b1h), "v"(b2h), "v"(b3h), \
    "v"(sa0), "v"(sa1), "v"(sb0), "v"(sb1)

// ── 32 KPAIR MFMAs + 8 interleaved ds_reads (single asm block) ──
// Outputs: %0..15=acc, %16..23=d0..d7.  Inputs: %24..27=a_lo, %28..31=a_hi,
//   %32..35=b_lo, %36..39=b_hi, %40=sa0, %41=sa1, %42=sb0, %43=sb1,
//   %44=lds_a0, %45=lds_a1.

__device__ __forceinline__ void kpair_32mfma_with_lds(
    fp4_floatx4_t acc[16],
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_raw[2], const fp8e8m0_4 b_raw[2],
    float4 &d0, float4 &d1, float4 &d2, float4 &d3,
    float4 &d4, float4 &d5, float4 &d6, float4 &d7,
    uint32_t lds_a0, uint32_t lds_a1)
{
    KPAIR_SETUP();
    asm volatile(
        // Row 0 Phase 0 — ALL 8 ds_reads front-loaded (1:1 with first 8 MFMAs)
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %24, %32, %0,  %40, %42 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %16, %44 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %24, %33, %1,  %40, %42 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %17, %44 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %24, %34, %2,  %40, %43 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %18, %44 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %24, %35, %3,  %40, %43 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %19, %44 offset:6144\n"
        // Row 0 Phase 1 — remaining 4 ds_reads
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %28, %36, %0,  %40, %42 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %20, %45 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %28, %37, %1,  %40, %42 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %21, %45 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %28, %38, %2,  %40, %43 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %22, %45 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %28, %39, %3,  %40, %43 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %23, %45 offset:6144\n"
        // Rows 1-3: pure MFMAs (24 total, reads already in-flight with 24+ cycles to complete)
        // Row 1 Phase 0
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %25, %32, %4,  %40, %42 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %25, %33, %5,  %40, %42 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %25, %34, %6,  %40, %43 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %25, %35, %7,  %40, %43 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Row 1 Phase 1
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %29, %36, %4,  %40, %42 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %29, %37, %5,  %40, %42 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %29, %38, %6,  %40, %43 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %29, %39, %7,  %40, %43 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Row 2 Phase 0
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %26, %32, %8,  %41, %42 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %26, %33, %9,  %41, %42 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %26, %34, %10, %41, %43 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %26, %35, %11, %41, %43 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Row 2 Phase 1
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %30, %36, %8,  %41, %42 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %30, %37, %9,  %41, %42 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %30, %38, %10, %41, %43 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %30, %39, %11, %41, %43 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Row 3 Phase 0
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %27, %32, %12, %41, %42 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %27, %33, %13, %41, %42 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %27, %34, %14, %41, %43 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %27, %35, %15, %41, %43 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Row 3 Phase 1
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %31, %36, %12, %41, %42 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %31, %37, %13, %41, %42 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %31, %38, %14, %41, %43 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %31, %39, %15, %41, %43 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER,
          "=&v"(d0), "=&v"(d1), "=&v"(d2), "=&v"(d3),
          "=&v"(d4), "=&v"(d5), "=&v"(d6), "=&v"(d7)
        : KPAIR_INPUTS,
          "v"(lds_a0), "v"(lds_a1)
    );
}

// ── 32 KPAIR MFMAs + 8 interleaved tile prefetch loads ──
// Uses same operand layout as original kpair_32mfma (%0..15=acc, %16..19=a_lo,
// %20..23=a_hi, %24..27=b_lo, %28..31=b_hi, %32..33=sa, %34..35=sb)
// but split into 4 row blocks with 2 pf calls between each.

template<int PF_N = 8>
__device__ __forceinline__ void kpair_32mfma_with_pf(
    fp4_floatx4_t acc[16],
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_raw[2], const fp8e8m0_4 b_raw[2],
    const tile_pf_params &pf0, const tile_pf_params &pf1)
{
    KPAIR_SETUP();
    // Row 0
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %16, %24, %0,  %32, %34 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %16, %25, %1,  %32, %34 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %16, %26, %2,  %32, %35 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %16, %27, %3,  %32, %35 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %20, %28, %0,  %32, %34 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %20, %29, %1,  %32, %34 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %20, %30, %2,  %32, %35 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %20, %31, %3,  %32, %35 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 0) emit_one_pf(pf0, 0);
    if constexpr (PF_N > 1) emit_one_pf(pf0, 1);
    // Row 1
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %17, %24, %4,  %32, %34 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %17, %25, %5,  %32, %34 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %17, %26, %6,  %32, %35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %17, %27, %7,  %32, %35 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %21, %28, %4,  %32, %34 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %21, %29, %5,  %32, %34 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %21, %30, %6,  %32, %35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %21, %31, %7,  %32, %35 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 2) emit_one_pf(pf0, 2);
    if constexpr (PF_N > 3) emit_one_pf(pf0, 3);
    // Row 2
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %18, %24, %8,  %33, %34 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %18, %25, %9,  %33, %34 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %18, %26, %10, %33, %35 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %18, %27, %11, %33, %35 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %22, %28, %8,  %33, %34 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %22, %29, %9,  %33, %34 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %22, %30, %10, %33, %35 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %22, %31, %11, %33, %35 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 4) emit_one_pf(pf1, 0);
    if constexpr (PF_N > 5) emit_one_pf(pf1, 1);
    // Row 3
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %19, %24, %12, %33, %34 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %19, %25, %13, %33, %34 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %19, %26, %14, %33, %35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %19, %27, %15, %33, %35 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %23, %28, %12, %33, %34 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %23, %29, %13, %33, %34 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %23, %30, %14, %33, %35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %23, %31, %15, %33, %35 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 6) emit_one_pf(pf1, 2);
    if constexpr (PF_N > 7) emit_one_pf(pf1, 3);
}

template<int PF_N = 8>
__device__ __forceinline__ void kpair_32mfma_with_pf_swapped_sel(
    fp4_floatx4_t acc[16],
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_raw[2], const fp8e8m0_4 b_raw[2],
    const tile_pf_params &pf0, const tile_pf_params &pf1)
{
    KPAIR_SETUP();
    // Row 0
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %24, %16, %0,  %34, %32 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %25, %16, %1,  %34, %32 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %26, %16, %2,  %35, %32 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %27, %16, %3,  %35, %32 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %28, %20, %0,  %34, %32 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %29, %20, %1,  %34, %32 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %30, %20, %2,  %35, %32 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %31, %20, %3,  %35, %32 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 0) emit_one_pf(pf0, 0);
    if constexpr (PF_N > 1) emit_one_pf(pf0, 1);
    // Row 1
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %24, %17, %4,  %34, %32 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %25, %17, %5,  %34, %32 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %26, %17, %6,  %35, %32 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %27, %17, %7,  %35, %32 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %28, %21, %4,  %34, %32 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %29, %21, %5,  %34, %32 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %30, %21, %6,  %35, %32 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %31, %21, %7,  %35, %32 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 2) emit_one_pf(pf0, 2);
    if constexpr (PF_N > 3) emit_one_pf(pf0, 3);
    // Row 2
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %24, %18, %8,  %34, %33 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %25, %18, %9,  %34, %33 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %26, %18, %10, %35, %33 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %27, %18, %11, %35, %33 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %28, %22, %8,  %34, %33 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %29, %22, %9,  %34, %33 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %30, %22, %10, %35, %33 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %31, %22, %11, %35, %33 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 4) emit_one_pf(pf1, 0);
    if constexpr (PF_N > 5) emit_one_pf(pf1, 1);
    // Row 3
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %24, %19, %12, %34, %33 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %25, %19, %13, %34, %33 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %26, %19, %14, %35, %33 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %27, %19, %15, %35, %33 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %28, %23, %12, %34, %33 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %29, %23, %13, %34, %33 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %30, %23, %14, %35, %33 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %31, %23, %15, %35, %33 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 6) emit_one_pf(pf1, 2);
    if constexpr (PF_N > 7) emit_one_pf(pf1, 3);
}

// ── 32 KPAIR MFMAs, no interleaved ops (single asm block) ──
__device__ __forceinline__ void kpair_32mfma_pure(
    fp4_floatx4_t acc[16],
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_raw[2], const fp8e8m0_4 b_raw[2])
{
    KPAIR_SETUP();
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %16, %24, %0,  %32, %34 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %16, %25, %1,  %32, %34 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %16, %26, %2,  %32, %35 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %16, %27, %3,  %32, %35 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %20, %28, %0,  %32, %34 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %20, %29, %1,  %32, %34 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %20, %30, %2,  %32, %35 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %20, %31, %3,  %32, %35 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %17, %24, %4,  %32, %34 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %17, %25, %5,  %32, %34 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %17, %26, %6,  %32, %35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %17, %27, %7,  %32, %35 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %21, %28, %4,  %32, %34 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %21, %29, %5,  %32, %34 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %21, %30, %6,  %32, %35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %21, %31, %7,  %32, %35 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %18, %24, %8,  %33, %34 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %18, %25, %9,  %33, %34 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %18, %26, %10, %33, %35 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %18, %27, %11, %33, %35 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %22, %28, %8,  %33, %34 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %22, %29, %9,  %33, %34 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %22, %30, %10, %33, %35 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %22, %31, %11, %33, %35 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %19, %24, %12, %33, %34 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %19, %25, %13, %33, %34 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %19, %26, %14, %33, %35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %19, %27, %15, %33, %35 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %23, %28, %12, %33, %34 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %23, %29, %13, %33, %34 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %23, %30, %14, %33, %35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %23, %31, %15, %33, %35 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
}

// Operand-swap POC variant 1:
// swap src/scale operands but keep the original op_sel bit assignment.
__device__ __forceinline__ void kpair_32mfma_pure_swapped_plain(
    fp4_floatx4_t acc[16],
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_raw[2], const fp8e8m0_4 b_raw[2])
{
    KPAIR_SETUP();
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %24, %16, %0,  %34, %32 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %25, %16, %1,  %34, %32 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %26, %16, %2,  %35, %32 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %27, %16, %3,  %35, %32 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %28, %20, %0,  %34, %32 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %29, %20, %1,  %34, %32 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %30, %20, %2,  %35, %32 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %31, %20, %3,  %35, %32 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %24, %17, %4,  %34, %32 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %25, %17, %5,  %34, %32 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %26, %17, %6,  %35, %32 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %27, %17, %7,  %35, %32 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %28, %21, %4,  %34, %32 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %29, %21, %5,  %34, %32 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %30, %21, %6,  %35, %32 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %31, %21, %7,  %35, %32 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %24, %18, %8,  %34, %33 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %25, %18, %9,  %34, %33 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %26, %18, %10, %35, %33 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %27, %18, %11, %35, %33 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %28, %22, %8,  %34, %33 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %29, %22, %9,  %34, %33 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %30, %22, %10, %35, %33 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %31, %22, %11, %35, %33 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %24, %19, %12, %34, %33 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %25, %19, %13, %34, %33 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %26, %19, %14, %35, %33 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %27, %19, %15, %35, %33 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %28, %23, %12, %34, %33 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %29, %23, %13, %34, %33 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %30, %23, %14, %35, %33 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %31, %23, %15, %35, %33 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
}

// Operand-swap POC variant 2:
// swap src/scale operands and also swap the first two op_sel/op_sel_hi bits.
__device__ __forceinline__ void kpair_32mfma_pure_swapped_sel(
    fp4_floatx4_t acc[16],
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_raw[2], const fp8e8m0_4 b_raw[2])
{
    KPAIR_SETUP();
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %24, %16, %0,  %34, %32 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %25, %16, %1,  %34, %32 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %26, %16, %2,  %35, %32 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %27, %16, %3,  %35, %32 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %28, %20, %0,  %34, %32 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %29, %20, %1,  %34, %32 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %30, %20, %2,  %35, %32 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %31, %20, %3,  %35, %32 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %24, %17, %4,  %34, %32 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %25, %17, %5,  %34, %32 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %26, %17, %6,  %35, %32 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %27, %17, %7,  %35, %32 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %28, %21, %4,  %34, %32 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %29, %21, %5,  %34, %32 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %30, %21, %6,  %35, %32 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %31, %21, %7,  %35, %32 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %24, %18, %8,  %34, %33 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %25, %18, %9,  %34, %33 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %26, %18, %10, %35, %33 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %27, %18, %11, %35, %33 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %28, %22, %8,  %34, %33 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %29, %22, %9,  %34, %33 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %30, %22, %10, %35, %33 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %31, %22, %11, %35, %33 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %24, %19, %12, %34, %33 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %25, %19, %13, %34, %33 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %26, %19, %14, %35, %33 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %27, %19, %15, %35, %33 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %28, %23, %12, %34, %33 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %29, %23, %13, %34, %33 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %30, %23, %14, %35, %33 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %31, %23, %15, %35, %33 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
}

__device__ __forceinline__ void dump_acc_raw(
    const _gl_float &out, const fp4_floatx4_t acc[16], int variant)
{
    const int lid = kittens::laneid();
    float *dst = out.raw_ptr
               + static_cast<size_t>(variant) * WARP_THREADS * 16 * 4
               + static_cast<size_t>(lid) * 16 * 4;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        dst[i * 4 + 0] = acc[i][0];
        dst[i * 4 + 1] = acc[i][1];
        dst[i * 4 + 2] = acc[i][2];
        dst[i * 4 + 3] = acc[i][3];
    }
}

__device__ __forceinline__ void dump_fp4_tile_rows(
    const _gl_float &out, const fp4_intx8_t t[4], int variant)
{
    const int tid = static_cast<int>(threadIdx.x);
    const int stride = out.cols();
    float *dst_ptr = out.raw_ptr
                   + static_cast<size_t>((variant * _NUM_THREADS + tid) * 4) * stride;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            dst_ptr[i * stride + j] = static_cast<float>(t[i][j]);
        }
    }
}

__device__ __forceinline__ void store_acc_block_standard(
    const _gl_float &out, const fp4_floatx4_t acc[16], int row_base)
{
    const int lid = kittens::laneid();
    const int row_off = 4 * (lid / 16);
    const int col_off = lid % 16;
    const int stride = out.cols();
    float *dst_ptr = out.raw_ptr + static_cast<size_t>(row_base) * stride;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            fp4_floatx4_t s = acc[i * 4 + j];
            const int row = i * 16 + row_off;
            const int col = j * 16 + col_off;
            dst_ptr[(row + 0) * stride + col] = s[0];
            dst_ptr[(row + 1) * stride + col] = s[1];
            dst_ptr[(row + 2) * stride + col] = s[2];
            dst_ptr[(row + 3) * stride + col] = s[3];
        }
    }
}

__device__ __forceinline__ void store_acc_block_inner_transpose(
    const _gl_float &out, const fp4_floatx4_t acc[16], int row_base)
{
    const int lid = kittens::laneid();
    const int lane_group = lid / 16;
    const int lane_pos = lid % 16;
    const int stride = out.cols();
    float *dst_ptr = out.raw_ptr + static_cast<size_t>(row_base) * stride;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            fp4_floatx4_t s = acc[i * 4 + j];
            const int row = i * 16 + lane_pos;
            const int col = j * 16 + 4 * lane_group;
            dst_ptr[row * stride + col + 0] = s[0];
            dst_ptr[row * stride + col + 1] = s[1];
            dst_ptr[row * stride + col + 2] = s[2];
            dst_ptr[row * stride + col + 3] = s[3];
        }
    }
}

__device__ __forceinline__ void store_acc_block_inner_permlane(
    const _gl_float &out, const fp4_floatx4_t acc[16], int row_base)
{
    const int lid = kittens::laneid();
    const int lane_group = lid / 16;
    const int lane_pos = lid % 16;
    const bool write_lane = (lane_group & 1) == 0;

    const int stride = out.cols();
    float *dst_ptr = out.raw_ptr + static_cast<size_t>(row_base) * stride;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            fp4_floatx4_t s = acc[i * 4 + j];
            u32x2_t sw0 = __builtin_amdgcn_permlane16_swap(
                std::bit_cast<unsigned int>(s[0]),
                std::bit_cast<unsigned int>(s[0]),
                false, false);
            u32x2_t sw1 = __builtin_amdgcn_permlane16_swap(
                std::bit_cast<unsigned int>(s[1]),
                std::bit_cast<unsigned int>(s[1]),
                false, false);
            u32x2_t sw2 = __builtin_amdgcn_permlane16_swap(
                std::bit_cast<unsigned int>(s[2]),
                std::bit_cast<unsigned int>(s[2]),
                false, false);
            u32x2_t sw3 = __builtin_amdgcn_permlane16_swap(
                std::bit_cast<unsigned int>(s[3]),
                std::bit_cast<unsigned int>(s[3]),
                false, false);
            if (!write_lane) continue;
            const int row = i * 16 + lane_pos;
            const int col = j * 16 + 8 * (lane_group / 2);
            dst_ptr[row * stride + col + 0] = std::bit_cast<float>(sw0[0]);
            dst_ptr[row * stride + col + 1] = std::bit_cast<float>(sw1[0]);
            dst_ptr[row * stride + col + 2] = std::bit_cast<float>(sw2[0]);
            dst_ptr[row * stride + col + 3] = std::bit_cast<float>(sw3[0]);
            dst_ptr[row * stride + col + 4] = std::bit_cast<float>(sw0[1]);
            dst_ptr[row * stride + col + 5] = std::bit_cast<float>(sw1[1]);
            dst_ptr[row * stride + col + 6] = std::bit_cast<float>(sw2[1]);
            dst_ptr[row * stride + col + 7] = std::bit_cast<float>(sw3[1]);
        }
    }
}

// ── Merged Steps 1+2: 64 MFMAs + 16 ds_reads (Br + A1) in one asm block ──
// Eliminates compiler transition between Steps 1 and 2.
// Outputs: %0..15=acc_bl, %16..31=acc_br, %32..39=br_d, %40..47=a1_d
// Inputs: %48..55=A0_lo/hi, %56..63=Bl_lo/hi, %64..65=sa0/sa1,
//   %66..67=sb_bl0/sb_bl1, %68..69=sb_br0/sb_br1,
//   %70..71=br_lds_p0/p1, %72..73=a1_lds_p0/p1

__device__ __forceinline__ void kpair_64mfma_step12(
    fp4_floatx4_t acc_bl[16], fp4_floatx4_t acc_br[16],
    const fp4_intx8_t A0[4], const fp4_intx8_t Bl[4],
    const fp8e8m0_4 a_raw[2], const fp8e8m0_4 bl_raw[2], const fp8e8m0_4 br_raw[2],
    float4 br_d[8], float4 a1_d[8],
    uint32_t br_p0, uint32_t br_p1,
    uint32_t a1_p0, uint32_t a1_p1)
{
    fp4_intx4_t a0l=fp4_lo4(A0[0]), a1l=fp4_lo4(A0[1]), a2l=fp4_lo4(A0[2]), a3l=fp4_lo4(A0[3]);
    fp4_intx4_t a0h=fp4_hi4(A0[0]), a1h=fp4_hi4(A0[1]), a2h=fp4_hi4(A0[2]), a3h=fp4_hi4(A0[3]);
    fp4_intx4_t b0l=fp4_lo4(Bl[0]), b1l=fp4_lo4(Bl[1]), b2l=fp4_lo4(Bl[2]), b3l=fp4_lo4(Bl[3]);
    fp4_intx4_t b0h=fp4_hi4(Bl[0]), b1h=fp4_hi4(Bl[1]), b2h=fp4_hi4(Bl[2]), b3h=fp4_hi4(Bl[3]);
    unsigned sa0 = std::bit_cast<unsigned>(a_raw[0]);
    unsigned sa1 = std::bit_cast<unsigned>(a_raw[1]);
    unsigned sb_bl0 = std::bit_cast<unsigned>(bl_raw[0]);
    unsigned sb_bl1 = std::bit_cast<unsigned>(bl_raw[1]);
    unsigned sb_br0 = std::bit_cast<unsigned>(br_raw[0]);
    unsigned sb_br1 = std::bit_cast<unsigned>(br_raw[1]);

    asm volatile(
        // ═══ STEP 1: A0×Bl (32 MFMAs) + 8 ds_reads for Br ═══
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %48, %56, %0,  %64, %66 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %32, %70 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %48, %57, %1,  %64, %66 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %33, %70 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %48, %58, %2,  %64, %67 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %34, %70 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %48, %59, %3,  %64, %67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %35, %70 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %52, %60, %0,  %64, %66 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %36, %71 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %52, %61, %1,  %64, %66 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %37, %71 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %52, %62, %2,  %64, %67 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %38, %71 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %52, %63, %3,  %64, %67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %39, %71 offset:6144\n"
        // Rows 1-3: 24 pure Step 1 MFMAs
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %49, %56, %4,  %64, %66 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %49, %57, %5,  %64, %66 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %49, %58, %6,  %64, %67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %49, %59, %7,  %64, %67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %53, %60, %4,  %64, %66 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %53, %61, %5,  %64, %66 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %53, %62, %6,  %64, %67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %53, %63, %7,  %64, %67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %50, %56, %8,  %65, %66 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %50, %57, %9,  %65, %66 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %50, %58, %10, %65, %67 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %50, %59, %11, %65, %67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %54, %60, %8,  %65, %66 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %54, %61, %9,  %65, %66 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %54, %62, %10, %65, %67 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %54, %63, %11, %65, %67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %51, %56, %12, %65, %66 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %51, %57, %13, %65, %66 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %51, %58, %14, %65, %67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %51, %59, %15, %65, %67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %55, %60, %12, %65, %66 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %55, %61, %13, %65, %66 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %55, %62, %14, %65, %67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %55, %63, %15, %65, %67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Wait for Br ds_reads
        "s_waitcnt lgkmcnt(0)\n"
        // ═══ STEP 2: A0×Br (32 MFMAs) + 8 ds_reads for A1 ═══
        "v_mfma_scale_f32_16x16x128_f8f6f4 %16, %48, %32, %16, %64, %68 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %40, %72 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %17, %48, %33, %17, %64, %68 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %41, %72 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %18, %48, %34, %18, %64, %69 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %42, %72 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %19, %48, %35, %19, %64, %69 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %43, %72 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %16, %52, %36, %16, %64, %68 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %44, %73 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %17, %52, %37, %17, %64, %68 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %45, %73 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %18, %52, %38, %18, %64, %69 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %46, %73 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %19, %52, %39, %19, %64, %69 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %47, %73 offset:6144\n"
        // Rows 1-3: 24 pure Step 2 MFMAs
        "v_mfma_scale_f32_16x16x128_f8f6f4 %20, %49, %32, %20, %64, %68 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %21, %49, %33, %21, %64, %68 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %22, %49, %34, %22, %64, %69 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %23, %49, %35, %23, %64, %69 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %20, %53, %36, %20, %64, %68 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %21, %53, %37, %21, %64, %68 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %22, %53, %38, %22, %64, %69 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %23, %53, %39, %23, %64, %69 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %24, %50, %32, %24, %65, %68 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %25, %50, %33, %25, %65, %68 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %26, %50, %34, %26, %65, %69 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %27, %50, %35, %27, %65, %69 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %24, %54, %36, %24, %65, %68 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %25, %54, %37, %25, %65, %68 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %26, %54, %38, %26, %65, %69 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %27, %54, %39, %27, %65, %69 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %28, %51, %32, %28, %65, %68 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %29, %51, %33, %29, %65, %68 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %30, %51, %34, %30, %65, %69 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %31, %51, %35, %31, %65, %69 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %28, %55, %36, %28, %65, %68 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %29, %55, %37, %29, %65, %68 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %30, %55, %38, %30, %65, %69 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %31, %55, %39, %31, %65, %69 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc_bl[0]),  "+a"(acc_bl[1]),  "+a"(acc_bl[2]),  "+a"(acc_bl[3]),
          "+a"(acc_bl[4]),  "+a"(acc_bl[5]),  "+a"(acc_bl[6]),  "+a"(acc_bl[7]),
          "+a"(acc_bl[8]),  "+a"(acc_bl[9]),  "+a"(acc_bl[10]), "+a"(acc_bl[11]),
          "+a"(acc_bl[12]), "+a"(acc_bl[13]), "+a"(acc_bl[14]), "+a"(acc_bl[15]),
          "+a"(acc_br[0]),  "+a"(acc_br[1]),  "+a"(acc_br[2]),  "+a"(acc_br[3]),
          "+a"(acc_br[4]),  "+a"(acc_br[5]),  "+a"(acc_br[6]),  "+a"(acc_br[7]),
          "+a"(acc_br[8]),  "+a"(acc_br[9]),  "+a"(acc_br[10]), "+a"(acc_br[11]),
          "+a"(acc_br[12]), "+a"(acc_br[13]), "+a"(acc_br[14]), "+a"(acc_br[15]),
          "=&v"(br_d[0]), "=&v"(br_d[1]), "=&v"(br_d[2]), "=&v"(br_d[3]),
          "=&v"(br_d[4]), "=&v"(br_d[5]), "=&v"(br_d[6]), "=&v"(br_d[7]),
          "=&v"(a1_d[0]), "=&v"(a1_d[1]), "=&v"(a1_d[2]), "=&v"(a1_d[3]),
          "=&v"(a1_d[4]), "=&v"(a1_d[5]), "=&v"(a1_d[6]), "=&v"(a1_d[7])
        : "v"(a0l), "v"(a1l), "v"(a2l), "v"(a3l),
          "v"(a0h), "v"(a1h), "v"(a2h), "v"(a3h),
          "v"(b0l), "v"(b1l), "v"(b2l), "v"(b3l),
          "v"(b0h), "v"(b1h), "v"(b2h), "v"(b3h),
          "v"(sa0), "v"(sa1), "v"(sb_bl0), "v"(sb_bl1),
          "v"(sb_br0), "v"(sb_br1),
          "v"(br_p0), "v"(br_p1), "v"(a1_p0), "v"(a1_p1)
    );
}

// Swapped Step12 helper:
// keep the fused ds_read schedule from kpair_64mfma_step12, but use the
// operand-swapped MFMA form that matches kpair_32mfma_pure_swapped_sel.
__device__ __forceinline__ void kpair_64mfma_step12_swapped_sel(
    fp4_floatx4_t acc_bl[16], fp4_floatx4_t acc_br[16],
    const fp4_intx8_t A0[4], const fp4_intx8_t Bl[4],
    const fp8e8m0_4 a_raw[2], const fp8e8m0_4 bl_raw[2], const fp8e8m0_4 br_raw[2],
    float4 br_d[8], float4 a1_d[8],
    uint32_t br_p0, uint32_t br_p1,
    uint32_t a1_p0, uint32_t a1_p1)
{
    fp4_intx4_t a0l=fp4_lo4(A0[0]), a1l=fp4_lo4(A0[1]), a2l=fp4_lo4(A0[2]), a3l=fp4_lo4(A0[3]);
    fp4_intx4_t a0h=fp4_hi4(A0[0]), a1h=fp4_hi4(A0[1]), a2h=fp4_hi4(A0[2]), a3h=fp4_hi4(A0[3]);
    fp4_intx4_t b0l=fp4_lo4(Bl[0]), b1l=fp4_lo4(Bl[1]), b2l=fp4_lo4(Bl[2]), b3l=fp4_lo4(Bl[3]);
    fp4_intx4_t b0h=fp4_hi4(Bl[0]), b1h=fp4_hi4(Bl[1]), b2h=fp4_hi4(Bl[2]), b3h=fp4_hi4(Bl[3]);
    unsigned sa0 = std::bit_cast<unsigned>(a_raw[0]);
    unsigned sa1 = std::bit_cast<unsigned>(a_raw[1]);
    unsigned sb_bl0 = std::bit_cast<unsigned>(bl_raw[0]);
    unsigned sb_bl1 = std::bit_cast<unsigned>(bl_raw[1]);
    unsigned sb_br0 = std::bit_cast<unsigned>(br_raw[0]);
    unsigned sb_br1 = std::bit_cast<unsigned>(br_raw[1]);

    asm volatile(
        // ═══ STEP 1: A0×Bl (swapped) + 8 ds_reads for Br ═══
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %56, %48, %0,  %66, %64 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %32, %70 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %57, %48, %1,  %66, %64 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %33, %70 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %58, %48, %2,  %67, %64 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %34, %70 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %59, %48, %3,  %67, %64 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %35, %70 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %60, %52, %0,  %66, %64 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %36, %71 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %61, %52, %1,  %66, %64 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %37, %71 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %62, %52, %2,  %67, %64 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %38, %71 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %63, %52, %3,  %67, %64 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %39, %71 offset:6144\n"
        // Rows 1-3: 24 pure Step 1 MFMAs
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %56, %49, %4,  %66, %64 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %57, %49, %5,  %66, %64 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %58, %49, %6,  %67, %64 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %59, %49, %7,  %67, %64 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %60, %53, %4,  %66, %64 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %61, %53, %5,  %66, %64 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %62, %53, %6,  %67, %64 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %63, %53, %7,  %67, %64 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %56, %50, %8,  %66, %65 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %57, %50, %9,  %66, %65 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %58, %50, %10, %67, %65 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %59, %50, %11, %67, %65 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %60, %54, %8,  %66, %65 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %61, %54, %9,  %66, %65 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %62, %54, %10, %67, %65 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %63, %54, %11, %67, %65 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %56, %51, %12, %66, %65 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %57, %51, %13, %66, %65 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %58, %51, %14, %67, %65 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %59, %51, %15, %67, %65 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %60, %55, %12, %66, %65 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %61, %55, %13, %66, %65 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %62, %55, %14, %67, %65 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %63, %55, %15, %67, %65 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Wait for Br ds_reads
        "s_waitcnt lgkmcnt(0)\n"
        // ═══ STEP 2: A0×Br (swapped) + 8 ds_reads for A1 ═══
        "v_mfma_scale_f32_16x16x128_f8f6f4 %16, %32, %48, %16, %68, %64 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %40, %72 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %17, %33, %48, %17, %68, %64 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %41, %72 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %18, %34, %48, %18, %69, %64 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %42, %72 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %19, %35, %48, %19, %69, %64 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %43, %72 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %16, %36, %52, %16, %68, %64 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %44, %73 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %17, %37, %52, %17, %68, %64 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %45, %73 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %18, %38, %52, %18, %69, %64 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %46, %73 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %19, %39, %52, %19, %69, %64 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %47, %73 offset:6144\n"
        // Rows 1-3: 24 pure Step 2 MFMAs
        "v_mfma_scale_f32_16x16x128_f8f6f4 %20, %32, %49, %20, %68, %64 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %21, %33, %49, %21, %68, %64 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %22, %34, %49, %22, %69, %64 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %23, %35, %49, %23, %69, %64 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %20, %36, %53, %20, %68, %64 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %21, %37, %53, %21, %68, %64 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %22, %38, %53, %22, %69, %64 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %23, %39, %53, %23, %69, %64 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %24, %32, %50, %24, %68, %65 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %25, %33, %50, %25, %68, %65 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %26, %34, %50, %26, %69, %65 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %27, %35, %50, %27, %69, %65 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %24, %36, %54, %24, %68, %65 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %25, %37, %54, %25, %68, %65 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %26, %38, %54, %26, %69, %65 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %27, %39, %54, %27, %69, %65 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %28, %32, %51, %28, %68, %65 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %29, %33, %51, %29, %68, %65 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %30, %34, %51, %30, %69, %65 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %31, %35, %51, %31, %69, %65 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %28, %36, %55, %28, %68, %65 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %29, %37, %55, %29, %68, %65 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %30, %38, %55, %30, %69, %65 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %31, %39, %55, %31, %69, %65 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc_bl[0]),  "+a"(acc_bl[1]),  "+a"(acc_bl[2]),  "+a"(acc_bl[3]),
          "+a"(acc_bl[4]),  "+a"(acc_bl[5]),  "+a"(acc_bl[6]),  "+a"(acc_bl[7]),
          "+a"(acc_bl[8]),  "+a"(acc_bl[9]),  "+a"(acc_bl[10]), "+a"(acc_bl[11]),
          "+a"(acc_bl[12]), "+a"(acc_bl[13]), "+a"(acc_bl[14]), "+a"(acc_bl[15]),
          "+a"(acc_br[0]),  "+a"(acc_br[1]),  "+a"(acc_br[2]),  "+a"(acc_br[3]),
          "+a"(acc_br[4]),  "+a"(acc_br[5]),  "+a"(acc_br[6]),  "+a"(acc_br[7]),
          "+a"(acc_br[8]),  "+a"(acc_br[9]),  "+a"(acc_br[10]), "+a"(acc_br[11]),
          "+a"(acc_br[12]), "+a"(acc_br[13]), "+a"(acc_br[14]), "+a"(acc_br[15]),
          "=&v"(br_d[0]), "=&v"(br_d[1]), "=&v"(br_d[2]), "=&v"(br_d[3]),
          "=&v"(br_d[4]), "=&v"(br_d[5]), "=&v"(br_d[6]), "=&v"(br_d[7]),
          "=&v"(a1_d[0]), "=&v"(a1_d[1]), "=&v"(a1_d[2]), "=&v"(a1_d[3]),
          "=&v"(a1_d[4]), "=&v"(a1_d[5]), "=&v"(a1_d[6]), "=&v"(a1_d[7])
        : "v"(a0l), "v"(a1l), "v"(a2l), "v"(a3l),
          "v"(a0h), "v"(a1h), "v"(a2h), "v"(a3h),
          "v"(b0l), "v"(b1l), "v"(b2l), "v"(b3l),
          "v"(b0h), "v"(b1h), "v"(b2h), "v"(b3h),
          "v"(sa0), "v"(sa1), "v"(sb_bl0), "v"(sb_bl1),
          "v"(sb_br0), "v"(sb_br1),
          "v"(br_p0), "v"(br_p1), "v"(a1_p0), "v"(a1_p1)
    );
}

// ── 32 KPAIR MFMAs + 16 ds_reads (2 tiles) + 8 pf (row blocks) ──
// Each row: 8 MFMAs + 4 ds_reads (2 tiles × 2 reads) + 2 pf.
// Operands: %0..15=acc(+a), %16..19=ds_out(=v: da_lo,db_lo,da_hi,db_hi),
//   %20..39=KPAIR_INPUTS, %40=lds_a, %41=lds_b.

#define KPAIR_ROW_ACC_DS4 \
    KPAIR_ACC_CLOBBER, "=v"(da_lo), "=v"(db_lo), "=v"(da_hi), "=v"(db_hi)
#define KPAIR_INPUTS_2LDS \
    KPAIR_INPUTS, "v"(lds_a), "v"(lds_b)

template<int PF_N = 8>
__device__ __forceinline__ void kpair_32mfma_with_16lds_and_pf(
    fp4_floatx4_t acc[16],
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_raw[2], const fp8e8m0_4 b_raw[2],
    float4 da[8], float4 db[8],
    uint32_t lds_a_addrs[2], uint32_t lds_b_addrs[2],
    const tile_pf_params &pf0, const tile_pf_params &pf1)
{
    KPAIR_SETUP();
    { // Row 0 (even, sa0)
        float4 &da_lo=da[0], &db_lo=db[0], &da_hi=da[4], &db_hi=db[4];
        uint32_t lds_a=lds_a_addrs[0], lds_b=lds_b_addrs[0];
        asm volatile(
            "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %20, %28, %0,  %36, %38 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %20, %29, %1,  %36, %38 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "ds_read_b128 %16, %40 offset:0\n"
            "ds_read_b128 %17, %41 offset:0\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %20, %30, %2,  %36, %39 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %20, %31, %3,  %36, %39 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %24, %32, %0,  %36, %38 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %24, %33, %1,  %36, %38 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "ds_read_b128 %18, %40 offset:2048\n"
            "ds_read_b128 %19, %41 offset:2048\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %24, %34, %2,  %36, %39 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %24, %35, %3,  %36, %39 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            : KPAIR_ROW_ACC_DS4 : KPAIR_INPUTS_2LDS);
    }
    if constexpr (PF_N > 0) emit_one_pf(pf0, 0);
    if constexpr (PF_N > 1) emit_one_pf(pf0, 1);
    { // Row 1 (odd, sa0)
        float4 &da_lo=da[1], &db_lo=db[1], &da_hi=da[5], &db_hi=db[5];
        uint32_t lds_a=lds_a_addrs[0], lds_b=lds_b_addrs[0];
        asm volatile(
            "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %21, %28, %4,  %36, %38 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %21, %29, %5,  %36, %38 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "ds_read_b128 %16, %40 offset:4096\n"
            "ds_read_b128 %17, %41 offset:4096\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %21, %30, %6,  %36, %39 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %21, %31, %7,  %36, %39 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %25, %32, %4,  %36, %38 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %25, %33, %5,  %36, %38 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "ds_read_b128 %18, %40 offset:6144\n"
            "ds_read_b128 %19, %41 offset:6144\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %25, %34, %6,  %36, %39 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %25, %35, %7,  %36, %39 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            : KPAIR_ROW_ACC_DS4 : KPAIR_INPUTS_2LDS);
    }
    if constexpr (PF_N > 2) emit_one_pf(pf0, 2);
    if constexpr (PF_N > 3) emit_one_pf(pf0, 3);
    { // Row 2 (even, sa1)
        float4 &da_lo=da[2], &db_lo=db[2], &da_hi=da[6], &db_hi=db[6];
        uint32_t lds_a=lds_a_addrs[1], lds_b=lds_b_addrs[1];
        asm volatile(
            "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %22, %28, %8,  %37, %38 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %22, %29, %9,  %37, %38 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "ds_read_b128 %16, %40 offset:0\n"
            "ds_read_b128 %17, %41 offset:0\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %22, %30, %10, %37, %39 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %22, %31, %11, %37, %39 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %26, %32, %8,  %37, %38 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %26, %33, %9,  %37, %38 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "ds_read_b128 %18, %40 offset:2048\n"
            "ds_read_b128 %19, %41 offset:2048\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %26, %34, %10, %37, %39 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %26, %35, %11, %37, %39 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            : KPAIR_ROW_ACC_DS4 : KPAIR_INPUTS_2LDS);
    }
    if constexpr (PF_N > 4) emit_one_pf(pf1, 0);
    if constexpr (PF_N > 5) emit_one_pf(pf1, 1);
    { // Row 3 (odd, sa1)
        float4 &da_lo=da[3], &db_lo=db[3], &da_hi=da[7], &db_hi=db[7];
        uint32_t lds_a=lds_a_addrs[1], lds_b=lds_b_addrs[1];
        asm volatile(
            "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %23, %28, %12, %37, %38 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %23, %29, %13, %37, %38 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "ds_read_b128 %16, %40 offset:4096\n"
            "ds_read_b128 %17, %41 offset:4096\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %23, %30, %14, %37, %39 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %23, %31, %15, %37, %39 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %27, %32, %12, %37, %38 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %27, %33, %13, %37, %38 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "ds_read_b128 %18, %40 offset:6144\n"
            "ds_read_b128 %19, %41 offset:6144\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %27, %34, %14, %37, %39 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %27, %35, %15, %37, %39 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            : KPAIR_ROW_ACC_DS4 : KPAIR_INPUTS_2LDS);
    }
    if constexpr (PF_N > 6) emit_one_pf(pf1, 2);
    if constexpr (PF_N > 7) emit_one_pf(pf1, 3);
}

// ── 32 KPAIR MFMAs + 8 ds_reads + 8 pf (split into 4 row blocks) ──
// Each row: 8 MFMAs + 2 ds_reads (in asm) + 2 pf (C++ builtin).

template<int PF_N = 8, bool EMIT_BARRIER = false>
__device__ __forceinline__ void kpair_32mfma_with_lds_and_pf(
    fp4_floatx4_t acc[16],
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_raw[2], const fp8e8m0_4 b_raw[2],
    float4 &d0, float4 &d1, float4 &d2, float4 &d3,
    float4 &d4, float4 &d5, float4 &d6, float4 &d7,
    uint32_t lds_a0, uint32_t lds_a1,
    const tile_pf_params &pf0, const tile_pf_params &pf1)
{
    KPAIR_SETUP();
    // Row 0: 8 MFMAs + ALL 8 ds_reads front-loaded (1:1 interleave)
    // When EMIT_BARRIER: vmcnt+barrier at top, MFMAs overlap with any stall
    if constexpr (EMIT_BARRIER) {
        asm volatile("s_waitcnt vmcnt(" MXFP4_STR(STEP3_BARRIER_VMCNT) ")\ns_barrier\n" ::: "memory");
    }
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %24, %32, %0,  %40, %42 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %16, %44 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %24, %33, %1,  %40, %42 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %17, %44 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %24, %34, %2,  %40, %43 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %18, %44 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %24, %35, %3,  %40, %43 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %19, %44 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %28, %36, %0,  %40, %42 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %20, %45 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %28, %37, %1,  %40, %42 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %21, %45 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %28, %38, %2,  %40, %43 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %22, %45 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %28, %39, %3,  %40, %43 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %23, %45 offset:6144\n"
        : KPAIR_ACC_CLOBBER,
          "=&v"(d0), "=&v"(d1), "=&v"(d2), "=&v"(d3),
          "=&v"(d4), "=&v"(d5), "=&v"(d6), "=&v"(d7)
        : KPAIR_INPUTS,
          "v"(lds_a0), "v"(lds_a1)
    );
    if constexpr (PF_N > 0) emit_one_pf(pf0, 0);
    if constexpr (PF_N > 1) emit_one_pf(pf0, 1);
    // Row 1 (odd, sa0) — pure MFMAs
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %17, %24, %4,  %32, %34 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %17, %25, %5,  %32, %34 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %17, %26, %6,  %32, %35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %17, %27, %7,  %32, %35 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %21, %28, %4,  %32, %34 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %21, %29, %5,  %32, %34 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %21, %30, %6,  %32, %35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %21, %31, %7,  %32, %35 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 2) emit_one_pf(pf0, 2);
    if constexpr (PF_N > 3) emit_one_pf(pf0, 3);
    // Row 2 (even, sa1) — pure MFMAs
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %18, %24, %8,  %33, %34 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %18, %25, %9,  %33, %34 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %18, %26, %10, %33, %35 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %18, %27, %11, %33, %35 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %22, %28, %8,  %33, %34 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %22, %29, %9,  %33, %34 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %22, %30, %10, %33, %35 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %22, %31, %11, %33, %35 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 4) emit_one_pf(pf1, 0);
    if constexpr (PF_N > 5) emit_one_pf(pf1, 1);
    // Row 3 (odd, sa1) — pure MFMAs
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %19, %24, %12, %33, %34 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %19, %25, %13, %33, %34 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %19, %26, %14, %33, %35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %19, %27, %15, %33, %35 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %23, %28, %12, %33, %34 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %23, %29, %13, %33, %34 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %23, %30, %14, %33, %35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %23, %31, %15, %33, %35 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 6) emit_one_pf(pf1, 2);
    if constexpr (PF_N > 7) emit_one_pf(pf1, 3);
}

template<int PF_N = 8, bool EMIT_BARRIER = false>
__device__ __forceinline__ void kpair_32mfma_with_lds_and_pf_swapped_sel(
    fp4_floatx4_t acc[16],
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_raw[2], const fp8e8m0_4 b_raw[2],
    float4 &d0, float4 &d1, float4 &d2, float4 &d3,
    float4 &d4, float4 &d5, float4 &d6, float4 &d7,
    uint32_t lds_a0, uint32_t lds_a1,
    const tile_pf_params &pf0, const tile_pf_params &pf1)
{
    KPAIR_SETUP();
    // Row 0 keeps the with_lds operand numbering where ds_read outputs
    // occupy %16..23, so the swapped MFMA indices are not a direct copy of
    // kpair_32mfma_with_pf_swapped_sel.
    if constexpr (EMIT_BARRIER) {
        asm volatile("s_waitcnt vmcnt(" MXFP4_STR(STEP3_BARRIER_VMCNT) ")\ns_barrier\n" ::: "memory");
    }
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %32, %24, %0,  %42, %40 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %16, %44 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %33, %24, %1,  %42, %40 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %17, %44 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %34, %24, %2,  %43, %40 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %18, %44 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %35, %24, %3,  %43, %40 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %19, %44 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %36, %28, %0,  %42, %40 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %20, %45 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %37, %28, %1,  %42, %40 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %21, %45 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %38, %28, %2,  %43, %40 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %22, %45 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %39, %28, %3,  %43, %40 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %23, %45 offset:6144\n"
        : KPAIR_ACC_CLOBBER,
          "=&v"(d0), "=&v"(d1), "=&v"(d2), "=&v"(d3),
          "=&v"(d4), "=&v"(d5), "=&v"(d6), "=&v"(d7)
        : KPAIR_INPUTS,
          "v"(lds_a0), "v"(lds_a1)
    );
    if constexpr (PF_N > 0) emit_one_pf(pf0, 0);
    if constexpr (PF_N > 1) emit_one_pf(pf0, 1);
    // Row 1
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %24, %17, %4,  %34, %32 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %25, %17, %5,  %34, %32 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %26, %17, %6,  %35, %32 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %27, %17, %7,  %35, %32 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %28, %21, %4,  %34, %32 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %29, %21, %5,  %34, %32 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %30, %21, %6,  %35, %32 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %31, %21, %7,  %35, %32 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 2) emit_one_pf(pf0, 2);
    if constexpr (PF_N > 3) emit_one_pf(pf0, 3);
    // Row 2
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %24, %18, %8,  %34, %33 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %25, %18, %9,  %34, %33 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %26, %18, %10, %35, %33 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %27, %18, %11, %35, %33 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %28, %22, %8,  %34, %33 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %29, %22, %9,  %34, %33 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %30, %22, %10, %35, %33 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %31, %22, %11, %35, %33 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 4) emit_one_pf(pf1, 0);
    if constexpr (PF_N > 5) emit_one_pf(pf1, 1);
    // Row 3
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %24, %19, %12, %34, %33 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %25, %19, %13, %34, %33 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %26, %19, %14, %35, %33 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %27, %19, %15, %35, %33 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %28, %23, %12, %34, %33 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %29, %23, %13, %34, %33 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %30, %23, %14, %35, %33 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %31, %23, %15, %35, %33 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : KPAIR_ACC_CLOBBER : KPAIR_INPUTS);
    if constexpr (PF_N > 6) emit_one_pf(pf1, 2);
    if constexpr (PF_N > 7) emit_one_pf(pf1, 3);
}

// ══════════════════════════════════════════════════════════════
// Main kernel
// ══════════════════════════════════════════════════════════════

__global__ __launch_bounds__(_NUM_THREADS, 1)
void mxfp4_gluon_cpp_kernel(const gluon_globals g) {
    static_assert(K_BYTES % BK == 0 && N_DIM % BLK == 0 && M_DIM % BLK == 0);

    constexpr int bpc = N_DIM / BLK;
    constexpr int a_packs = RBM / 32;
    constexpr int b_packs = RBN / 32;

    __shared__ ST_tile A0_db[2], A1_db[2], Bl_db[2], Br_db[2];

    // XCD-aware dispatch + GROUP_SIZE_M swizzle for L2 B-tile reuse
    constexpr int NUM_XCDS = 8;
#ifndef GROUP_SIZE_M
#define GROUP_SIZE_M 4
#endif
    constexpr int GROUP_M = GROUP_SIZE_M;
    const int total_blocks = gridDim.x;
    const int bpr = total_blocks / bpc;

    // XCD pid remapping: Gluon-style "tall XCDs" for correct remainder handling
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

    // GROUP_SIZE_M swizzle within XCD's block range
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

    // Scale SRDs — merged preshuffle format (64-row super-groups, dwordx2 loads)
    // lane_soff_x2: doubled offsets for merged format where each dword position is 8 bytes
    const uint32_t lane_soff_x2 =
        (static_cast<uint32_t>(kittens::laneid() / 16) << 7) |
        (static_cast<uint32_t>(kittens::laneid() % 16) << 3);

    // One SRD per tile-half, pointing to the 64-row super-group base
    i32x4 a0_srd = make_scale_srd(preshuffled_scale_row_base_ptr(
        g.a_scale, (br * BLK + wm * RBM) >> 6));
    i32x4 a1_srd = make_scale_srd(preshuffled_scale_row_base_ptr(
        g.a_scale, (br * BLK + HB + wm * RBM) >> 6));
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

    // Pre-compute LDS addresses as 16 static named variables (one per db-slot × phase).
    // Avoids runtime-indexed [2][2] arrays that compiler spills to LDS + ds_read_b64.
    // Selection via ternary (compiles to v_cndmask). No swap needed.
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

    // Tile extraction helper (ds_read float4[8] → fp4_intx8_t[4])
    auto extract_tile = [](const float4 d[8], fp4_intx8_t t[4]) __attribute__((always_inline)) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            auto lo = *reinterpret_cast<const fp4_intx4_t*>(&d[i]);
            auto hi = *reinterpret_cast<const fp4_intx4_t*>(&d[i + 4]);
            t[i][0]=lo[0]; t[i][1]=lo[1]; t[i][2]=lo[2]; t[i][3]=lo[3];
            t[i][4]=hi[0]; t[i][5]=hi[1]; t[i][6]=hi[2]; t[i][7]=hi[3];
        }
    };

    // ═══════════ Prologue ═══════════
    load_tiles(0, 0);
    if (k_byte_iters > 1) load_tiles(1, 1);

    fp8e8m0_4 pf_a0[a_packs], pf_a1[a_packs], pf_bl[b_packs], pf_br[b_packs];
    {
        load_pq_scale_x2_async(a0_srd, lane_soff_x2, 0, pf_a0[0], pf_a0[1]);
        load_pq_scale_x2_async(a1_srd, lane_soff_x2, 0, pf_a1[0], pf_a1[1]);
        load_pq_scale_x2_async(bl_srd, lane_soff_x2, 0, pf_bl[0], pf_bl[1]);
        load_pq_scale_x2_async(br_srd, lane_soff_x2, 0, pf_br[0], pf_br[1]);
    }

    // Pre-load A0+Bl for iteration 0 from LDS (2-tile software pipeline)
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();
    A_row_reg a0_rt;
    B_row_reg bl_rt;
    fp4_load_st_to_rt(a0_rt, kittens::subtile_inplace<RBM, BK>(A0_db[0], {wm, 0}));
    fp4_load_st_to_rt(bl_rt, kittens::subtile_inplace<RBN, BK>(Bl_db[0], {wn, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    fp4_intx8_t tA0[4], tBl[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        tA0[i] = fp4_extract_tile(a0_rt, i);
        tBl[i] = fp4_extract_tile(bl_rt, i);
    }

    // ═══════════ Main loop (2-tile pipeline) ═══════════
#if SWAP_STEP34_MAIN
    // Split steady-state iterations from the final tail to keep the hot loop free
    // of last-iteration branches and reduce live-range pressure.
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
    for (int bt = 0; bt + 1 < k_byte_iters; ++bt) {
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

        {
            const uint32_t nxt_scale = static_cast<uint32_t>(bt + 1) << 9;
            load_pq_scale_x2_async(a0_srd, lane_soff_x2, nxt_scale, pf_a0[0], pf_a0[1]);
            load_pq_scale_x2_async(a1_srd, lane_soff_x2, nxt_scale, pf_a1[0], pf_a1[1]);
            load_pq_scale_x2_async(bl_srd, lane_soff_x2, nxt_scale, pf_bl[0], pf_bl[1]);
            load_pq_scale_x2_async(br_srd, lane_soff_x2, nxt_scale, pf_br[0], pf_br[1]);
        }

        float4 br_d[8], a1_d[8];
#if SWAP_STEP12_MAIN
        kpair_64mfma_step12_swapped_sel(acc_A0Bl, acc_A0Br, tA0, tBl,
            a0_raw, bl_raw, br_raw, br_d, a1_d,
            sel_br_p0, sel_br_p1, sel_a1_p0, sel_a1_p1);
#else
        kpair_64mfma_step12(acc_A0Bl, acc_A0Br, tA0, tBl,
            a0_raw, bl_raw, br_raw, br_d, a1_d,
            sel_br_p0, sel_br_p1, sel_a1_p0, sel_a1_p1);
#endif

        asm volatile("s_waitcnt lgkmcnt(0)");
        fp4_intx8_t tBr[4], tA1[4];
        extract_tile(br_d, tBr);
        extract_tile(a1_d, tA1);

#if !STEP3_EMBED_BARRIER
        asm volatile("s_waitcnt vmcnt(0)\ns_barrier\n" ::: "memory");
#endif

        float4 nxt_a0_d[8];
        float4 nxt_bl_d[8];
        kpair_32mfma_with_lds_and_pf_swapped_sel<8, STEP3_EMBED_BARRIER>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw,
            nxt_a0_d[0], nxt_a0_d[1], nxt_a0_d[2], nxt_a0_d[3],
            nxt_a0_d[4], nxt_a0_d[5], nxt_a0_d[6], nxt_a0_d[7],
            sel_a0_p0, sel_a0_p1, pf_a0_p, pf_a1_p);

#if STEP4_EXTERNAL_BR_PREFETCH
        kpair_32mfma_with_lds_and_pf_swapped_sel<4>(acc_A1Br, tA1, tBr, a1_raw, br_raw,
            nxt_bl_d[0], nxt_bl_d[1], nxt_bl_d[2], nxt_bl_d[3],
            nxt_bl_d[4], nxt_bl_d[5], nxt_bl_d[6], nxt_bl_d[7],
            sel_bl_p0, sel_bl_p1, pf_bl_p, pf_br_p);
        #pragma unroll
        for (int pi = 0; pi < PF_MPT; ++pi) emit_one_pf(pf_br_p, pi);
#else
        kpair_32mfma_with_lds_and_pf_swapped_sel<8>(acc_A1Br, tA1, tBr, a1_raw, br_raw,
            nxt_bl_d[0], nxt_bl_d[1], nxt_bl_d[2], nxt_bl_d[3],
            nxt_bl_d[4], nxt_bl_d[5], nxt_bl_d[6], nxt_bl_d[7],
            sel_bl_p0, sel_bl_p1, pf_bl_p, pf_br_p);
#endif

        asm volatile("s_waitcnt lgkmcnt(0)");
        extract_tile(nxt_a0_d, tA0);
        extract_tile(nxt_bl_d, tBl);
    }

    {
        const int bt = k_byte_iters - 1;
        const int cur = bt & 1;
        const uint32_t sel_br_p0 = cur ? br_1_p0 : br_0_p0;
        const uint32_t sel_br_p1 = cur ? br_1_p1 : br_0_p1;
        const uint32_t sel_a1_p0 = cur ? a1_1_p0 : a1_0_p0;
        const uint32_t sel_a1_p1 = cur ? a1_1_p1 : a1_0_p1;

        fp8e8m0_4 a0_raw[a_packs], a1_raw[a_packs], bl_raw[b_packs], br_raw[b_packs];
        #pragma unroll
        for (int p = 0; p < a_packs; ++p) { a0_raw[p] = pf_a0[p]; a1_raw[p] = pf_a1[p]; }
        #pragma unroll
        for (int p = 0; p < b_packs; ++p) { bl_raw[p] = pf_bl[p]; br_raw[p] = pf_br[p]; }

        float4 br_d[8], a1_d[8];
#if SWAP_STEP12_MAIN
        kpair_64mfma_step12_swapped_sel(acc_A0Bl, acc_A0Br, tA0, tBl,
            a0_raw, bl_raw, br_raw, br_d, a1_d,
            sel_br_p0, sel_br_p1, sel_a1_p0, sel_a1_p1);
#else
        kpair_64mfma_step12(acc_A0Bl, acc_A0Br, tA0, tBl,
            a0_raw, bl_raw, br_raw, br_d, a1_d,
            sel_br_p0, sel_br_p1, sel_a1_p0, sel_a1_p1);
#endif

        asm volatile("s_waitcnt lgkmcnt(0)");
        fp4_intx8_t tBr[4], tA1[4];
        extract_tile(br_d, tBr);
        extract_tile(a1_d, tA1);

#if !STEP3_EMBED_BARRIER
        asm volatile("s_waitcnt vmcnt(0)\ns_barrier\n" ::: "memory");
#endif

        tile_pf_params dummy_pf = {};
        kpair_32mfma_with_pf_swapped_sel<0>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw, dummy_pf, dummy_pf);
        kpair_32mfma_with_pf_swapped_sel<0>(acc_A1Br, tA1, tBr, a1_raw, br_raw, dummy_pf, dummy_pf);
    }
#else
    // Unroll: override via -DUNROLL_K=N, else auto by K size
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

        {
            const uint32_t nxt_scale = static_cast<uint32_t>(bt + 1 < k_byte_iters ? bt + 1 : bt) << 9;
            load_pq_scale_x2_async(a0_srd, lane_soff_x2, nxt_scale, pf_a0[0], pf_a0[1]);
            load_pq_scale_x2_async(a1_srd, lane_soff_x2, nxt_scale, pf_a1[0], pf_a1[1]);
            load_pq_scale_x2_async(bl_srd, lane_soff_x2, nxt_scale, pf_bl[0], pf_bl[1]);
            load_pq_scale_x2_async(br_srd, lane_soff_x2, nxt_scale, pf_br[0], pf_br[1]);
        }

        // Steps 1+2 merged: A0×Bl (32 MFMAs) + ds_read Br + A0×Br (32 MFMAs) + ds_read A1
        float4 br_d[8], a1_d[8];
#if SWAP_STEP12_MAIN
        kpair_64mfma_step12_swapped_sel(acc_A0Bl, acc_A0Br, tA0, tBl,
            a0_raw, bl_raw, br_raw, br_d, a1_d,
            sel_br_p0, sel_br_p1, sel_a1_p0, sel_a1_p1);
#else
        kpair_64mfma_step12(acc_A0Bl, acc_A0Br, tA0, tBl,
            a0_raw, bl_raw, br_raw, br_d, a1_d,
            sel_br_p0, sel_br_p1, sel_a1_p0, sel_a1_p1);
#endif

        asm volatile("s_waitcnt lgkmcnt(0)");
        fp4_intx8_t tBr[4], tA1[4];
        extract_tile(br_d, tBr);
        extract_tile(a1_d, tA1);

        // Step3 barrier placement is configurable for A/B experiments.
#if !STEP3_EMBED_BARRIER
        asm volatile("s_waitcnt vmcnt(0)\ns_barrier\n" ::: "memory");
#endif

        // Step 3: A1×Bl (32 MFMAs) + ds_read A0[nxt] — barrier at entry
        float4 nxt_a0_d[8];
        float4 nxt_bl_d[8];
        kpair_32mfma_with_lds_and_pf<8, STEP3_EMBED_BARRIER>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw,
            nxt_a0_d[0], nxt_a0_d[1], nxt_a0_d[2], nxt_a0_d[3],
            nxt_a0_d[4], nxt_a0_d[5], nxt_a0_d[6], nxt_a0_d[7],
            sel_a0_p0, sel_a0_p1, pf_a0_p, pf_a1_p);

        // Step4 Br prefetch can either stay interleaved or be issued after compute.
#if STEP4_EXTERNAL_BR_PREFETCH
        kpair_32mfma_with_lds_and_pf<4>(acc_A1Br, tA1, tBr, a1_raw, br_raw,
            nxt_bl_d[0], nxt_bl_d[1], nxt_bl_d[2], nxt_bl_d[3],
            nxt_bl_d[4], nxt_bl_d[5], nxt_bl_d[6], nxt_bl_d[7],
            sel_bl_p0, sel_bl_p1, pf_bl_p, pf_br_p);
        #pragma unroll
        for (int pi = 0; pi < PF_MPT; ++pi) emit_one_pf(pf_br_p, pi);
#else
        kpair_32mfma_with_lds_and_pf<8>(acc_A1Br, tA1, tBr, a1_raw, br_raw,
            nxt_bl_d[0], nxt_bl_d[1], nxt_bl_d[2], nxt_bl_d[3],
            nxt_bl_d[4], nxt_bl_d[5], nxt_bl_d[6], nxt_bl_d[7],
            sel_bl_p0, sel_bl_p1, pf_bl_p, pf_br_p);
#endif

        asm volatile("s_waitcnt lgkmcnt(0)");
        extract_tile(nxt_a0_d, tA0);
        extract_tile(nxt_bl_d, tBl);
    }
#endif

    // ═══════════ Store C — streamlined direct store ═══════════
    // Process base tiles directly from accumulators without materializing RT_C.
    // Uses 252 VGPRs (4 fewer than original), no spills.
    auto store_block = [&](const fp4_floatx4_t acc[16], int mh, int nh) {
        const int lid = kittens::laneid();
        const int tile_r = br * WARPS_M * 2 + WARPS_M * mh + wm;
        const int tile_c = bc * WARPS_N * 2 + WARPS_N * nh + wn;
        bf16 *dst_ptr = g.c.raw_ptr + static_cast<size_t>(tile_r * 64) * g.c.cols()
                        + static_cast<size_t>(tile_c * 64);
        const int row_stride = g.c.cols();
        const int row_off = 4 * (lid / 16);
        const int col_off = lid % 16;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                fp4_floatx4_t s = acc[i * 4 + j] * g.scale;
                const int row_base = i * 16 + row_off;
                const int col = j * 16 + col_off;
                dst_ptr[(row_base + 0) * row_stride + col] = base_types::convertor<bf16, float>::convert(s[0]);
                dst_ptr[(row_base + 1) * row_stride + col] = base_types::convertor<bf16, float>::convert(s[1]);
                dst_ptr[(row_base + 2) * row_stride + col] = base_types::convertor<bf16, float>::convert(s[2]);
                dst_ptr[(row_base + 3) * row_stride + col] = base_types::convertor<bf16, float>::convert(s[3]);
            }
        }
    };

#if SWAP_STEP34_MAIN
    auto store_block_inner = [&](const fp4_floatx4_t acc[16], int mh, int nh) {
        const int lid = kittens::laneid();
        const int tile_r = br * WARPS_M * 2 + WARPS_M * mh + wm;
        const int tile_c = bc * WARPS_N * 2 + WARPS_N * nh + wn;
        bf16 *dst_ptr = g.c.raw_ptr + static_cast<size_t>(tile_r * 64) * g.c.cols()
                        + static_cast<size_t>(tile_c * 64);
        const int row_stride = g.c.cols();
        const int lane_group = lid / 16;
        const int lane_pos = lid % 16;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                fp4_floatx4_t s = acc[i * 4 + j] * g.scale;
                const int row = i * 16 + lane_pos;
                const int col = j * 16 + 4 * lane_group;
                dst_ptr[row * row_stride + col + 0] = base_types::convertor<bf16, float>::convert(s[0]);
                dst_ptr[row * row_stride + col + 1] = base_types::convertor<bf16, float>::convert(s[1]);
                dst_ptr[row * row_stride + col + 2] = base_types::convertor<bf16, float>::convert(s[2]);
                dst_ptr[row * row_stride + col + 3] = base_types::convertor<bf16, float>::convert(s[3]);
            }
        }
    };
#if SWAP_STEP12_MAIN
    store_block_inner(acc_A0Bl, 0, 0);
    store_block_inner(acc_A0Br, 0, 1);
    store_block_inner(acc_A1Bl, 1, 0);
    store_block_inner(acc_A1Br, 1, 1);
#else
    store_block(acc_A0Bl, 0, 0);
    store_block(acc_A0Br, 0, 1);
    store_block_inner(acc_A1Bl, 1, 0);
    store_block_inner(acc_A1Br, 1, 1);
#endif
#else
    store_block(acc_A0Bl, 0, 0);
    store_block(acc_A0Br, 0, 1);
    store_block(acc_A1Bl, 1, 0);
    store_block(acc_A1Br, 1, 1);
#endif
}

__global__ __launch_bounds__(_NUM_THREADS, 1)
void mxfp4_operand_swap_debug_kernel(const operand_swap_debug_globals g) {
    constexpr int a_packs = RBM / 32;
    constexpr int b_packs = RBN / 32;

    __shared__ ST_tile A0_dbg, B0_dbg;

    uint32_t so_a[PF_MPT], so_b[PF_MPT];
    G::prefill_swizzled_offsets(A0_dbg, g.a, so_a);
    G::prefill_swizzled_offsets(B0_dbg, g.b, so_b);

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
    const uint32_t lb_a0 = lb(A0_dbg), lb_b0 = lb(B0_dbg);

    emit_tile_pf(A0_dbg, g.a, coord<ST_tile>(0, 0, 0, 0), so_a, srd_a, base_a, lb_a0);
    emit_tile_pf(B0_dbg, g.b, coord<ST_tile>(0, 0, 0, 0), so_b, srd_b, base_b, lb_b0);

    const uint32_t lane_soff_x2 =
        (static_cast<uint32_t>(kittens::laneid() / 16) << 7) |
        (static_cast<uint32_t>(kittens::laneid() % 16) << 3);
    i32x4 a_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.a_scale, 0));
    i32x4 b_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.b_scale, 0));

    fp8e8m0_4 a_raw[a_packs], b_raw[b_packs];
    load_pq_scale_x2_async(a_srd, lane_soff_x2, 0, a_raw[0], a_raw[1]);
    load_pq_scale_x2_async(b_srd, lane_soff_x2, 0, b_raw[0], b_raw[1]);

    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    if (warpid() != 0) return;

    A_row_reg a_rt;
    B_row_reg b_rt;
    fp4_load_st_to_rt(a_rt, kittens::subtile_inplace<RBM, BK>(A0_dbg, {0, 0}));
    fp4_load_st_to_rt(b_rt, kittens::subtile_inplace<RBN, BK>(B0_dbg, {0, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");

    fp4_intx8_t tA[4], tB[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        tA[i] = fp4_extract_tile(a_rt, i);
        tB[i] = fp4_extract_tile(b_rt, i);
    }

    fp4_floatx4_t acc_orig[16] = {};
    fp4_floatx4_t acc_swap_plain[16] = {};
    fp4_floatx4_t acc_swap_sel[16] = {};

    kpair_32mfma_pure(acc_orig, tA, tB, a_raw, b_raw);
    kpair_32mfma_pure_swapped_plain(acc_swap_plain, tA, tB, a_raw, b_raw);
    kpair_32mfma_pure_swapped_sel(acc_swap_sel, tA, tB, a_raw, b_raw);

    dump_acc_raw(g.out, acc_orig, 0);
    dump_acc_raw(g.out, acc_swap_plain, 1);
    dump_acc_raw(g.out, acc_swap_sel, 2);
}

__global__ __launch_bounds__(_NUM_THREADS, 1)
void mxfp4_operand_swap_block_kernel(const operand_swap_debug_globals g) {
    constexpr int a_packs = RBM / 32;
    constexpr int b_packs = RBN / 32;

    __shared__ ST_tile A0_dbg, B0_dbg;

    uint32_t so_a[PF_MPT], so_b[PF_MPT];
    G::prefill_swizzled_offsets(A0_dbg, g.a, so_a);
    G::prefill_swizzled_offsets(B0_dbg, g.b, so_b);

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
    const uint32_t lb_a0 = lb(A0_dbg), lb_b0 = lb(B0_dbg);

    emit_tile_pf(A0_dbg, g.a, coord<ST_tile>(0, 0, 0, 0), so_a, srd_a, base_a, lb_a0);
    emit_tile_pf(B0_dbg, g.b, coord<ST_tile>(0, 0, 0, 0), so_b, srd_b, base_b, lb_b0);

    const uint32_t lane_soff_x2 =
        (static_cast<uint32_t>(kittens::laneid() / 16) << 7) |
        (static_cast<uint32_t>(kittens::laneid() % 16) << 3);
    i32x4 a_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.a_scale, 0));
    i32x4 b_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.b_scale, 0));

    fp8e8m0_4 a_raw[a_packs], b_raw[b_packs];
    load_pq_scale_x2_async(a_srd, lane_soff_x2, 0, a_raw[0], a_raw[1]);
    load_pq_scale_x2_async(b_srd, lane_soff_x2, 0, b_raw[0], b_raw[1]);

    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    if (warpid() != 0) return;

    A_row_reg a_rt;
    B_row_reg b_rt;
    fp4_load_st_to_rt(a_rt, kittens::subtile_inplace<RBM, BK>(A0_dbg, {0, 0}));
    fp4_load_st_to_rt(b_rt, kittens::subtile_inplace<RBN, BK>(B0_dbg, {0, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");

    fp4_intx8_t tA[4], tB[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        tA[i] = fp4_extract_tile(a_rt, i);
        tB[i] = fp4_extract_tile(b_rt, i);
    }

    fp4_floatx4_t acc_orig[16] = {};
    fp4_floatx4_t acc_swap_sel[16] = {};
    kpair_32mfma_pure(acc_orig, tA, tB, a_raw, b_raw);
    kpair_32mfma_pure_swapped_sel(acc_swap_sel, tA, tB, a_raw, b_raw);

    store_acc_block_standard(g.out, acc_orig, 0);
    store_acc_block_inner_transpose(g.out, acc_swap_sel, 64);
    store_acc_block_inner_permlane(g.out, acc_swap_sel, 128);
}

template<bool SWAP_STEP34, bool USE_LDS_STEP34 = false, bool USE_MAINLIKE_STEP34 = false>
__device__ __forceinline__ void mxfp4_active_block_debug_body(const operand_swap_debug_globals g) {
    constexpr int a_packs = RBM / 32;
    constexpr int b_packs = RBN / 32;

    __shared__ ST_tile A0_db, A1_db, Bl_db, Br_db;

    const int wm = warpid() / WARPS_N;
    const int wn = warpid() % WARPS_N;

    uint32_t so_a[PF_MPT], so_b[PF_MPT];
    G::prefill_swizzled_offsets(A0_db, g.a, so_a);
    G::prefill_swizzled_offsets(Bl_db, g.b, so_b);

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
    const uint32_t lb_a0 = lb(A0_db), lb_a1 = lb(A1_db);
    const uint32_t lb_bl = lb(Bl_db), lb_br = lb(Br_db);

    emit_tile_pf(A0_db, g.a, coord<ST_tile>(0, 0, 0, 0), so_a, srd_a, base_a, lb_a0);
    emit_tile_pf(A1_db, g.a, coord<ST_tile>(0, 0, 1, 0), so_a, srd_a, base_a, lb_a1);
    emit_tile_pf(Bl_db, g.b, coord<ST_tile>(0, 0, 0, 0), so_b, srd_b, base_b, lb_bl);
    emit_tile_pf(Br_db, g.b, coord<ST_tile>(0, 0, 1, 0), so_b, srd_b, base_b, lb_br);

    const uint32_t lane_soff_x2 =
        (static_cast<uint32_t>(kittens::laneid() / 16) << 7) |
        (static_cast<uint32_t>(kittens::laneid() % 16) << 3);
    i32x4 a0_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.a_scale, (wm * RBM) >> 6));
    i32x4 a1_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.a_scale, (HB + wm * RBM) >> 6));
    i32x4 bl_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.b_scale, (wn * RBN) >> 6));
    i32x4 br_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.b_scale, (HB + wn * RBN) >> 6));

    fp8e8m0_4 a0_raw[a_packs], a1_raw[a_packs], bl_raw[b_packs], br_raw[b_packs];
    load_pq_scale_x2_async(a0_srd, lane_soff_x2, 0, a0_raw[0], a0_raw[1]);
    load_pq_scale_x2_async(a1_srd, lane_soff_x2, 0, a1_raw[0], a1_raw[1]);
    load_pq_scale_x2_async(bl_srd, lane_soff_x2, 0, bl_raw[0], bl_raw[1]);
    load_pq_scale_x2_async(br_srd, lane_soff_x2, 0, br_raw[0], br_raw[1]);

    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    A_row_reg a0_rt;
    B_row_reg bl_rt;
    fp4_load_st_to_rt(a0_rt, kittens::subtile_inplace<RBM, BK>(A0_db, {wm, 0}));
    fp4_load_st_to_rt(bl_rt, kittens::subtile_inplace<RBN, BK>(Bl_db, {wn, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");

    fp4_intx8_t tA0[4], tBl[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        tA0[i] = fp4_extract_tile(a0_rt, i);
        tBl[i] = fp4_extract_tile(bl_rt, i);
    }

    uint32_t a0_p0, a0_p1, a1_p0, a1_p1, bl_p0, bl_p1, br_p0, br_p1;
    compute_lds_base_addrs<A_row_reg>(kittens::subtile_inplace<RBM, BK>(A0_db, {wm, 0}), a0_p0, a0_p1);
    compute_lds_base_addrs<A_row_reg>(kittens::subtile_inplace<RBM, BK>(A1_db, {wm, 0}), a1_p0, a1_p1);
    compute_lds_base_addrs<B_row_reg>(kittens::subtile_inplace<RBN, BK>(Bl_db, {wn, 0}), bl_p0, bl_p1);
    compute_lds_base_addrs<B_row_reg>(kittens::subtile_inplace<RBN, BK>(Br_db, {wn, 0}), br_p0, br_p1);

    tile_pf_params pf_a0_p = {}, pf_a1_p = {}, pf_bl_p = {}, pf_br_p = {};
    if constexpr (USE_MAINLIKE_STEP34) {
        pf_a0_p = make_pf_params(A0_db, g.a, coord<ST_tile>(0, 0, 0, 0), so_a, srd_a, base_a, lb_a0);
        pf_a1_p = make_pf_params(A1_db, g.a, coord<ST_tile>(0, 0, 1, 0), so_a, srd_a, base_a, lb_a1);
        pf_bl_p = make_pf_params(Bl_db, g.b, coord<ST_tile>(0, 0, 0, 0), so_b, srd_b, base_b, lb_bl);
        pf_br_p = make_pf_params(Br_db, g.b, coord<ST_tile>(0, 0, 1, 0), so_b, srd_b, base_b, lb_br);
    }

    auto extract_tile = [](const float4 d[8], fp4_intx8_t t[4]) __attribute__((always_inline)) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            auto lo = *reinterpret_cast<const fp4_intx4_t*>(&d[i]);
            auto hi = *reinterpret_cast<const fp4_intx4_t*>(&d[i + 4]);
            t[i][0] = lo[0]; t[i][1] = lo[1]; t[i][2] = lo[2]; t[i][3] = lo[3];
            t[i][4] = hi[0]; t[i][5] = hi[1]; t[i][6] = hi[2]; t[i][7] = hi[3];
        }
    };

    fp4_floatx4_t acc_A0Bl[16] = {}, acc_A0Br[16] = {}, acc_A1Bl[16] = {}, acc_A1Br[16] = {};
    float4 br_d[8], a1_d[8];
    kpair_64mfma_step12(acc_A0Bl, acc_A0Br, tA0, tBl,
        a0_raw, bl_raw, br_raw, br_d, a1_d,
        br_p0, br_p1, a1_p0, a1_p1);

    asm volatile("s_waitcnt lgkmcnt(0)");
    fp4_intx8_t tBr[4], tA1[4];
    extract_tile(br_d, tBr);
    extract_tile(a1_d, tA1);

    tile_pf_params dummy_pf = {};
    if constexpr (SWAP_STEP34) {
        if constexpr (USE_LDS_STEP34) {
            float4 nxt_a0_d[8], nxt_bl_d[8];
            if constexpr (USE_MAINLIKE_STEP34) {
                kpair_32mfma_with_lds_and_pf_swapped_sel<8, STEP3_EMBED_BARRIER>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw,
                    nxt_a0_d[0], nxt_a0_d[1], nxt_a0_d[2], nxt_a0_d[3],
                    nxt_a0_d[4], nxt_a0_d[5], nxt_a0_d[6], nxt_a0_d[7],
                    a0_p0, a0_p1, pf_a0_p, pf_a1_p);
#if STEP4_EXTERNAL_BR_PREFETCH
                kpair_32mfma_with_lds_and_pf_swapped_sel<4>(acc_A1Br, tA1, tBr, a1_raw, br_raw,
                    nxt_bl_d[0], nxt_bl_d[1], nxt_bl_d[2], nxt_bl_d[3],
                    nxt_bl_d[4], nxt_bl_d[5], nxt_bl_d[6], nxt_bl_d[7],
                    bl_p0, bl_p1, pf_bl_p, pf_br_p);
                #pragma unroll
                for (int pi = 0; pi < PF_MPT; ++pi) emit_one_pf(pf_br_p, pi);
#else
                kpair_32mfma_with_lds_and_pf_swapped_sel<8>(acc_A1Br, tA1, tBr, a1_raw, br_raw,
                    nxt_bl_d[0], nxt_bl_d[1], nxt_bl_d[2], nxt_bl_d[3],
                    nxt_bl_d[4], nxt_bl_d[5], nxt_bl_d[6], nxt_bl_d[7],
                    bl_p0, bl_p1, pf_bl_p, pf_br_p);
#endif
            } else {
                kpair_32mfma_with_lds_and_pf_swapped_sel<0>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw,
                    nxt_a0_d[0], nxt_a0_d[1], nxt_a0_d[2], nxt_a0_d[3],
                    nxt_a0_d[4], nxt_a0_d[5], nxt_a0_d[6], nxt_a0_d[7],
                    a0_p0, a0_p1, dummy_pf, dummy_pf);
                kpair_32mfma_with_lds_and_pf_swapped_sel<0>(acc_A1Br, tA1, tBr, a1_raw, br_raw,
                    nxt_bl_d[0], nxt_bl_d[1], nxt_bl_d[2], nxt_bl_d[3],
                    nxt_bl_d[4], nxt_bl_d[5], nxt_bl_d[6], nxt_bl_d[7],
                    bl_p0, bl_p1, dummy_pf, dummy_pf);
            }
            asm volatile("s_waitcnt lgkmcnt(0)");
        } else {
            kpair_32mfma_with_pf_swapped_sel<0>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw, dummy_pf, dummy_pf);
            kpair_32mfma_with_pf_swapped_sel<0>(acc_A1Br, tA1, tBr, a1_raw, br_raw, dummy_pf, dummy_pf);
        }
    } else {
        kpair_32mfma_with_pf<0>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw, dummy_pf, dummy_pf);
        kpair_32mfma_with_pf<0>(acc_A1Br, tA1, tBr, a1_raw, br_raw, dummy_pf, dummy_pf);
    }

    auto store_block = [&](const fp4_floatx4_t acc[16], int mh, int nh) {
        const int lid = kittens::laneid();
        const int tile_r = WARPS_M * mh + wm;
        const int tile_c = WARPS_N * nh + wn;
        float *dst_ptr = g.out.raw_ptr + static_cast<size_t>(tile_r * 64) * g.out.cols()
                       + static_cast<size_t>(tile_c * 64);
        const int row_stride = g.out.cols();
        const int row_off = 4 * (lid / 16);
        const int col_off = lid % 16;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                fp4_floatx4_t s = acc[i * 4 + j];
                const int row_base = i * 16 + row_off;
                const int col = j * 16 + col_off;
                dst_ptr[(row_base + 0) * row_stride + col] = s[0];
                dst_ptr[(row_base + 1) * row_stride + col] = s[1];
                dst_ptr[(row_base + 2) * row_stride + col] = s[2];
                dst_ptr[(row_base + 3) * row_stride + col] = s[3];
            }
        }
    };

    auto store_block_inner = [&](const fp4_floatx4_t acc[16], int mh, int nh) {
        const int lid = kittens::laneid();
        const int tile_r = WARPS_M * mh + wm;
        const int tile_c = WARPS_N * nh + wn;
        float *dst_ptr = g.out.raw_ptr + static_cast<size_t>(tile_r * 64) * g.out.cols()
                       + static_cast<size_t>(tile_c * 64);
        const int row_stride = g.out.cols();
        const int lane_group = lid / 16;
        const int lane_pos = lid % 16;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                fp4_floatx4_t s = acc[i * 4 + j];
                const int row = i * 16 + lane_pos;
                const int col = j * 16 + 4 * lane_group;
                dst_ptr[row * row_stride + col + 0] = s[0];
                dst_ptr[row * row_stride + col + 1] = s[1];
                dst_ptr[row * row_stride + col + 2] = s[2];
                dst_ptr[row * row_stride + col + 3] = s[3];
            }
        }
    };

    store_block(acc_A0Bl, 0, 0);
    store_block(acc_A0Br, 0, 1);
    if constexpr (SWAP_STEP34) {
        store_block_inner(acc_A1Bl, 1, 0);
        store_block_inner(acc_A1Br, 1, 1);
    } else {
        store_block(acc_A1Bl, 1, 0);
        store_block(acc_A1Br, 1, 1);
    }
}

__global__ __launch_bounds__(_NUM_THREADS, 1)
void mxfp4_active_block_debug_kernel(const operand_swap_debug_globals g) {
    mxfp4_active_block_debug_body<false, false, false>(g);
}

__global__ __launch_bounds__(_NUM_THREADS, 1)
void mxfp4_active_block_swap_step34_kernel(const operand_swap_debug_globals g) {
    mxfp4_active_block_debug_body<true, false, false>(g);
}

__global__ __launch_bounds__(_NUM_THREADS, 1)
void mxfp4_active_block_swap_step34_with_lds_kernel(const operand_swap_debug_globals g) {
    mxfp4_active_block_debug_body<true, true, false>(g);
}

__global__ __launch_bounds__(_NUM_THREADS, 1)
void mxfp4_active_block_swap_step34_mainlike_kernel(const operand_swap_debug_globals g) {
    mxfp4_active_block_debug_body<true, true, true>(g);
}

__global__ __launch_bounds__(_NUM_THREADS, 1)
void mxfp4_active_block_swap_all_nonfused_kernel(const operand_swap_debug_globals g) {
    constexpr int a_packs = RBM / 32;
    constexpr int b_packs = RBN / 32;

    __shared__ ST_tile A0_db, A1_db, Bl_db, Br_db;

    const int wm = warpid() / WARPS_N;
    const int wn = warpid() % WARPS_N;

    uint32_t so_a[PF_MPT], so_b[PF_MPT];
    G::prefill_swizzled_offsets(A0_db, g.a, so_a);
    G::prefill_swizzled_offsets(Bl_db, g.b, so_b);

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
    const uint32_t lb_a0 = lb(A0_db), lb_a1 = lb(A1_db);
    const uint32_t lb_bl = lb(Bl_db), lb_br = lb(Br_db);

    emit_tile_pf(A0_db, g.a, coord<ST_tile>(0, 0, 0, 0), so_a, srd_a, base_a, lb_a0);
    emit_tile_pf(A1_db, g.a, coord<ST_tile>(0, 0, 1, 0), so_a, srd_a, base_a, lb_a1);
    emit_tile_pf(Bl_db, g.b, coord<ST_tile>(0, 0, 0, 0), so_b, srd_b, base_b, lb_bl);
    emit_tile_pf(Br_db, g.b, coord<ST_tile>(0, 0, 1, 0), so_b, srd_b, base_b, lb_br);

    const uint32_t lane_soff_x2 =
        (static_cast<uint32_t>(kittens::laneid() / 16) << 7) |
        (static_cast<uint32_t>(kittens::laneid() % 16) << 3);
    i32x4 a0_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.a_scale, (wm * RBM) >> 6));
    i32x4 a1_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.a_scale, (HB + wm * RBM) >> 6));
    i32x4 bl_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.b_scale, (wn * RBN) >> 6));
    i32x4 br_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.b_scale, (HB + wn * RBN) >> 6));

    fp8e8m0_4 a0_raw[a_packs], a1_raw[a_packs], bl_raw[b_packs], br_raw[b_packs];
    load_pq_scale_x2_async(a0_srd, lane_soff_x2, 0, a0_raw[0], a0_raw[1]);
    load_pq_scale_x2_async(a1_srd, lane_soff_x2, 0, a1_raw[0], a1_raw[1]);
    load_pq_scale_x2_async(bl_srd, lane_soff_x2, 0, bl_raw[0], bl_raw[1]);
    load_pq_scale_x2_async(br_srd, lane_soff_x2, 0, br_raw[0], br_raw[1]);

    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    A_row_reg a0_rt, a1_rt;
    B_row_reg bl_rt, br_rt;
    fp4_load_st_to_rt(a0_rt, kittens::subtile_inplace<RBM, BK>(A0_db, {wm, 0}));
    fp4_load_st_to_rt(a1_rt, kittens::subtile_inplace<RBM, BK>(A1_db, {wm, 0}));
    fp4_load_st_to_rt(bl_rt, kittens::subtile_inplace<RBN, BK>(Bl_db, {wn, 0}));
    fp4_load_st_to_rt(br_rt, kittens::subtile_inplace<RBN, BK>(Br_db, {wn, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");

    fp4_intx8_t tA0[4], tA1[4], tBl[4], tBr[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        tA0[i] = fp4_extract_tile(a0_rt, i);
        tA1[i] = fp4_extract_tile(a1_rt, i);
        tBl[i] = fp4_extract_tile(bl_rt, i);
        tBr[i] = fp4_extract_tile(br_rt, i);
    }

    fp4_floatx4_t acc_A0Bl[16] = {}, acc_A0Br[16] = {}, acc_A1Bl[16] = {}, acc_A1Br[16] = {};
    tile_pf_params dummy_pf = {};
    kpair_32mfma_pure_swapped_sel(acc_A0Bl, tA0, tBl, a0_raw, bl_raw);
    kpair_32mfma_pure_swapped_sel(acc_A0Br, tA0, tBr, a0_raw, br_raw);
    kpair_32mfma_with_pf_swapped_sel<0>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw, dummy_pf, dummy_pf);
    kpair_32mfma_with_pf_swapped_sel<0>(acc_A1Br, tA1, tBr, a1_raw, br_raw, dummy_pf, dummy_pf);

    auto store_block_inner = [&](const fp4_floatx4_t acc[16], int mh, int nh) {
        const int lid = kittens::laneid();
        const int tile_r = WARPS_M * mh + wm;
        const int tile_c = WARPS_N * nh + wn;
        float *dst_ptr = g.out.raw_ptr + static_cast<size_t>(tile_r * 64) * g.out.cols()
                       + static_cast<size_t>(tile_c * 64);
        const int row_stride = g.out.cols();
        const int lane_group = lid / 16;
        const int lane_pos = lid % 16;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                fp4_floatx4_t s = acc[i * 4 + j];
                const int row = i * 16 + lane_pos;
                const int col = j * 16 + 4 * lane_group;
                dst_ptr[row * row_stride + col + 0] = s[0];
                dst_ptr[row * row_stride + col + 1] = s[1];
                dst_ptr[row * row_stride + col + 2] = s[2];
                dst_ptr[row * row_stride + col + 3] = s[3];
            }
        }
    };

    store_block_inner(acc_A0Bl, 0, 0);
    store_block_inner(acc_A1Bl, 1, 0);
    store_block_inner(acc_A0Br, 0, 1);
    store_block_inner(acc_A1Br, 1, 1);
}

__global__ __launch_bounds__(_NUM_THREADS, 1)
void mxfp4_active_block_swap_all_fused_kernel(const operand_swap_debug_globals g) {
    constexpr int a_packs = RBM / 32;
    constexpr int b_packs = RBN / 32;

    __shared__ ST_tile A0_db, A1_db, Bl_db, Br_db;

    const int wm = warpid() / WARPS_N;
    const int wn = warpid() % WARPS_N;

    uint32_t so_a[PF_MPT], so_b[PF_MPT];
    G::prefill_swizzled_offsets(A0_db, g.a, so_a);
    G::prefill_swizzled_offsets(Bl_db, g.b, so_b);

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
    const uint32_t lb_a0 = lb(A0_db), lb_a1 = lb(A1_db);
    const uint32_t lb_bl = lb(Bl_db), lb_br = lb(Br_db);

    emit_tile_pf(A0_db, g.a, coord<ST_tile>(0, 0, 0, 0), so_a, srd_a, base_a, lb_a0);
    emit_tile_pf(A1_db, g.a, coord<ST_tile>(0, 0, 1, 0), so_a, srd_a, base_a, lb_a1);
    emit_tile_pf(Bl_db, g.b, coord<ST_tile>(0, 0, 0, 0), so_b, srd_b, base_b, lb_bl);
    emit_tile_pf(Br_db, g.b, coord<ST_tile>(0, 0, 1, 0), so_b, srd_b, base_b, lb_br);

    const uint32_t lane_soff_x2 =
        (static_cast<uint32_t>(kittens::laneid() / 16) << 7) |
        (static_cast<uint32_t>(kittens::laneid() % 16) << 3);
    i32x4 a0_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.a_scale, (wm * RBM) >> 6));
    i32x4 a1_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.a_scale, (HB + wm * RBM) >> 6));
    i32x4 bl_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.b_scale, (wn * RBN) >> 6));
    i32x4 br_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.b_scale, (HB + wn * RBN) >> 6));

    fp8e8m0_4 a0_raw[a_packs], a1_raw[a_packs], bl_raw[b_packs], br_raw[b_packs];
    load_pq_scale_x2_async(a0_srd, lane_soff_x2, 0, a0_raw[0], a0_raw[1]);
    load_pq_scale_x2_async(a1_srd, lane_soff_x2, 0, a1_raw[0], a1_raw[1]);
    load_pq_scale_x2_async(bl_srd, lane_soff_x2, 0, bl_raw[0], bl_raw[1]);
    load_pq_scale_x2_async(br_srd, lane_soff_x2, 0, br_raw[0], br_raw[1]);

    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    A_row_reg a0_rt;
    B_row_reg bl_rt;
    fp4_load_st_to_rt(a0_rt, kittens::subtile_inplace<RBM, BK>(A0_db, {wm, 0}));
    fp4_load_st_to_rt(bl_rt, kittens::subtile_inplace<RBN, BK>(Bl_db, {wn, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");

    fp4_intx8_t tA0[4], tBl[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        tA0[i] = fp4_extract_tile(a0_rt, i);
        tBl[i] = fp4_extract_tile(bl_rt, i);
    }

    uint32_t br_p0, br_p1, a1_p0, a1_p1;
    compute_lds_base_addrs<B_row_reg>(kittens::subtile_inplace<RBN, BK>(Br_db, {wn, 0}), br_p0, br_p1);
    compute_lds_base_addrs<A_row_reg>(kittens::subtile_inplace<RBM, BK>(A1_db, {wm, 0}), a1_p0, a1_p1);

    auto extract_tile = [](const float4 d[8], fp4_intx8_t t[4]) __attribute__((always_inline)) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            auto lo = *reinterpret_cast<const fp4_intx4_t*>(&d[i]);
            auto hi = *reinterpret_cast<const fp4_intx4_t*>(&d[i + 4]);
            t[i][0] = lo[0]; t[i][1] = lo[1]; t[i][2] = lo[2]; t[i][3] = lo[3];
            t[i][4] = hi[0]; t[i][5] = hi[1]; t[i][6] = hi[2]; t[i][7] = hi[3];
        }
    };

    fp4_floatx4_t acc_A0Bl[16] = {}, acc_A0Br[16] = {}, acc_A1Bl[16] = {}, acc_A1Br[16] = {};
    float4 br_d[8], a1_d[8];
    kpair_64mfma_step12_swapped_sel(acc_A0Bl, acc_A0Br, tA0, tBl,
        a0_raw, bl_raw, br_raw, br_d, a1_d, br_p0, br_p1, a1_p0, a1_p1);

    asm volatile("s_waitcnt lgkmcnt(0)");
    fp4_intx8_t tBr[4], tA1[4];
    extract_tile(br_d, tBr);
    extract_tile(a1_d, tA1);

    tile_pf_params dummy_pf = {};
    kpair_32mfma_with_pf_swapped_sel<0>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw, dummy_pf, dummy_pf);
    kpair_32mfma_with_pf_swapped_sel<0>(acc_A1Br, tA1, tBr, a1_raw, br_raw, dummy_pf, dummy_pf);

    auto store_block_inner = [&](const fp4_floatx4_t acc[16], int mh, int nh) {
        const int lid = kittens::laneid();
        const int tile_r = WARPS_M * mh + wm;
        const int tile_c = WARPS_N * nh + wn;
        float *dst_ptr = g.out.raw_ptr + static_cast<size_t>(tile_r * 64) * g.out.cols()
                       + static_cast<size_t>(tile_c * 64);
        const int row_stride = g.out.cols();
        const int lane_group = lid / 16;
        const int lane_pos = lid % 16;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                fp4_floatx4_t s = acc[i * 4 + j];
                const int row = i * 16 + lane_pos;
                const int col = j * 16 + 4 * lane_group;
                dst_ptr[row * row_stride + col + 0] = s[0];
                dst_ptr[row * row_stride + col + 1] = s[1];
                dst_ptr[row * row_stride + col + 2] = s[2];
                dst_ptr[row * row_stride + col + 3] = s[3];
            }
        }
    };

    store_block_inner(acc_A0Bl, 0, 0);
    store_block_inner(acc_A1Bl, 1, 0);
    store_block_inner(acc_A0Br, 0, 1);
    store_block_inner(acc_A1Br, 1, 1);
}

// Dump the lane-local fp4 tiles produced by direct RT extraction vs fused ds_read
// extraction. Expected out shape: [4 * _NUM_THREADS * 4, 8]
// variants: 0=tBr_ref, 1=tBr_dsread, 2=tA1_ref, 3=tA1_dsread
__global__ __launch_bounds__(_NUM_THREADS, 1)
void mxfp4_step12_layout_debug_kernel(const operand_swap_debug_globals g) {
    constexpr int a_packs = RBM / 32;
    constexpr int b_packs = RBN / 32;

    __shared__ ST_tile A0_db, A1_db, Bl_db, Br_db;

    const int wm = warpid() / WARPS_N;
    const int wn = warpid() % WARPS_N;

    uint32_t so_a[PF_MPT], so_b[PF_MPT];
    G::prefill_swizzled_offsets(A0_db, g.a, so_a);
    G::prefill_swizzled_offsets(Bl_db, g.b, so_b);

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
    const uint32_t lb_a0 = lb(A0_db), lb_a1 = lb(A1_db);
    const uint32_t lb_bl = lb(Bl_db), lb_br = lb(Br_db);

    emit_tile_pf(A0_db, g.a, coord<ST_tile>(0, 0, 0, 0), so_a, srd_a, base_a, lb_a0);
    emit_tile_pf(A1_db, g.a, coord<ST_tile>(0, 0, 1, 0), so_a, srd_a, base_a, lb_a1);
    emit_tile_pf(Bl_db, g.b, coord<ST_tile>(0, 0, 0, 0), so_b, srd_b, base_b, lb_bl);
    emit_tile_pf(Br_db, g.b, coord<ST_tile>(0, 0, 1, 0), so_b, srd_b, base_b, lb_br);

    const uint32_t lane_soff_x2 =
        (static_cast<uint32_t>(kittens::laneid() / 16) << 7) |
        (static_cast<uint32_t>(kittens::laneid() % 16) << 3);
    i32x4 a0_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.a_scale, (wm * RBM) >> 6));
    i32x4 a1_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.a_scale, (HB + wm * RBM) >> 6));
    i32x4 bl_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.b_scale, (wn * RBN) >> 6));
    i32x4 br_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.b_scale, (HB + wn * RBN) >> 6));

    fp8e8m0_4 a0_raw[a_packs], a1_raw[a_packs], bl_raw[b_packs], br_raw[b_packs];
    load_pq_scale_x2_async(a0_srd, lane_soff_x2, 0, a0_raw[0], a0_raw[1]);
    load_pq_scale_x2_async(a1_srd, lane_soff_x2, 0, a1_raw[0], a1_raw[1]);
    load_pq_scale_x2_async(bl_srd, lane_soff_x2, 0, bl_raw[0], bl_raw[1]);
    load_pq_scale_x2_async(br_srd, lane_soff_x2, 0, br_raw[0], br_raw[1]);

    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    A_row_reg a0_rt, a1_rt;
    B_row_reg bl_rt, br_rt;
    fp4_load_st_to_rt(a0_rt, kittens::subtile_inplace<RBM, BK>(A0_db, {wm, 0}));
    fp4_load_st_to_rt(a1_rt, kittens::subtile_inplace<RBM, BK>(A1_db, {wm, 0}));
    fp4_load_st_to_rt(bl_rt, kittens::subtile_inplace<RBN, BK>(Bl_db, {wn, 0}));
    fp4_load_st_to_rt(br_rt, kittens::subtile_inplace<RBN, BK>(Br_db, {wn, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");

    fp4_intx8_t tA0[4], tA1_ref[4], tBl[4], tBr_ref[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        tA0[i] = fp4_extract_tile(a0_rt, i);
        tA1_ref[i] = fp4_extract_tile(a1_rt, i);
        tBl[i] = fp4_extract_tile(bl_rt, i);
        tBr_ref[i] = fp4_extract_tile(br_rt, i);
    }

    uint32_t br_p0, br_p1, a1_p0, a1_p1;
    compute_lds_base_addrs<B_row_reg>(kittens::subtile_inplace<RBN, BK>(Br_db, {wn, 0}), br_p0, br_p1);
    compute_lds_base_addrs<A_row_reg>(kittens::subtile_inplace<RBM, BK>(A1_db, {wm, 0}), a1_p0, a1_p1);

    fp4_floatx4_t acc_A0Bl[16] = {}, acc_A0Br[16] = {};
    float4 br_d[8], a1_d[8];
    kpair_64mfma_step12_swapped_sel(acc_A0Bl, acc_A0Br, tA0, tBl,
        a0_raw, bl_raw, br_raw, br_d, a1_d, br_p0, br_p1, a1_p0, a1_p1);

    asm volatile("s_waitcnt lgkmcnt(0)");
    fp4_intx8_t tBr_ds[4], tA1_ds[4];
    extract_dsread_tile(br_d, tBr_ds);
    extract_dsread_tile(a1_d, tA1_ds);

    dump_fp4_tile_rows(g.out, tBr_ref, 0);
    dump_fp4_tile_rows(g.out, tBr_ds, 1);
    dump_fp4_tile_rows(g.out, tA1_ref, 2);
    dump_fp4_tile_rows(g.out, tA1_ds, 3);
}

// Compare step12 alone: nonfused swapped pure helpers vs fused swapped helper.
// Expected out shape: [4 * WARP_THREADS * 16, 4]
// variants: 0=ref_bl, 1=fused_bl, 2=ref_br, 3=fused_br
__global__ __launch_bounds__(_NUM_THREADS, 1)
void mxfp4_step12_swap_compare_kernel(const operand_swap_debug_globals g) {
    constexpr int a_packs = RBM / 32;
    constexpr int b_packs = RBN / 32;

    __shared__ ST_tile A0_db, A1_db, Bl_db, Br_db;

    const int wm = warpid() / WARPS_N;
    const int wn = warpid() % WARPS_N;

    uint32_t so_a[PF_MPT], so_b[PF_MPT];
    G::prefill_swizzled_offsets(A0_db, g.a, so_a);
    G::prefill_swizzled_offsets(Bl_db, g.b, so_b);

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
    const uint32_t lb_a0 = lb(A0_db), lb_a1 = lb(A1_db);
    const uint32_t lb_bl = lb(Bl_db), lb_br = lb(Br_db);

    emit_tile_pf(A0_db, g.a, coord<ST_tile>(0, 0, 0, 0), so_a, srd_a, base_a, lb_a0);
    emit_tile_pf(A1_db, g.a, coord<ST_tile>(0, 0, 1, 0), so_a, srd_a, base_a, lb_a1);
    emit_tile_pf(Bl_db, g.b, coord<ST_tile>(0, 0, 0, 0), so_b, srd_b, base_b, lb_bl);
    emit_tile_pf(Br_db, g.b, coord<ST_tile>(0, 0, 1, 0), so_b, srd_b, base_b, lb_br);

    const uint32_t lane_soff_x2 =
        (static_cast<uint32_t>(kittens::laneid() / 16) << 7) |
        (static_cast<uint32_t>(kittens::laneid() % 16) << 3);
    i32x4 a0_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.a_scale, (wm * RBM) >> 6));
    i32x4 a1_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.a_scale, (HB + wm * RBM) >> 6));
    i32x4 bl_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.b_scale, (wn * RBN) >> 6));
    i32x4 br_srd = make_scale_srd(preshuffled_scale_row_base_ptr(g.b_scale, (HB + wn * RBN) >> 6));

    fp8e8m0_4 a0_raw[a_packs], a1_raw[a_packs], bl_raw[b_packs], br_raw[b_packs];
    load_pq_scale_x2_async(a0_srd, lane_soff_x2, 0, a0_raw[0], a0_raw[1]);
    load_pq_scale_x2_async(a1_srd, lane_soff_x2, 0, a1_raw[0], a1_raw[1]);
    load_pq_scale_x2_async(bl_srd, lane_soff_x2, 0, bl_raw[0], bl_raw[1]);
    load_pq_scale_x2_async(br_srd, lane_soff_x2, 0, br_raw[0], br_raw[1]);

    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    if (warpid() != 0) return;

    A_row_reg a0_rt;
    B_row_reg bl_rt, br_rt;
    fp4_load_st_to_rt(a0_rt, kittens::subtile_inplace<RBM, BK>(A0_db, {0, 0}));
    fp4_load_st_to_rt(bl_rt, kittens::subtile_inplace<RBN, BK>(Bl_db, {0, 0}));
    fp4_load_st_to_rt(br_rt, kittens::subtile_inplace<RBN, BK>(Br_db, {0, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");

    fp4_intx8_t tA0[4], tBl[4], tBr_ref[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        tA0[i] = fp4_extract_tile(a0_rt, i);
        tBl[i] = fp4_extract_tile(bl_rt, i);
        tBr_ref[i] = fp4_extract_tile(br_rt, i);
    }

    uint32_t br_p0, br_p1, a1_p0, a1_p1;
    compute_lds_base_addrs<B_row_reg>(kittens::subtile_inplace<RBN, BK>(Br_db, {0, 0}), br_p0, br_p1);
    compute_lds_base_addrs<A_row_reg>(kittens::subtile_inplace<RBM, BK>(A1_db, {0, 0}), a1_p0, a1_p1);

    fp4_floatx4_t acc_ref_bl[16] = {}, acc_fused_bl[16] = {};
    fp4_floatx4_t acc_ref_br[16] = {}, acc_fused_br[16] = {};
    float4 br_d[8], a1_d[8];

    kpair_32mfma_pure_swapped_sel(acc_ref_bl, tA0, tBl, a0_raw, bl_raw);
    kpair_32mfma_pure_swapped_sel(acc_ref_br, tA0, tBr_ref, a0_raw, br_raw);
    kpair_64mfma_step12_swapped_sel(acc_fused_bl, acc_fused_br, tA0, tBl,
        a0_raw, bl_raw, br_raw, br_d, a1_d, br_p0, br_p1, a1_p0, a1_p1);

    dump_acc_raw(g.out, acc_ref_bl, 0);
    dump_acc_raw(g.out, acc_fused_bl, 1);
    dump_acc_raw(g.out, acc_ref_br, 2);
    dump_acc_raw(g.out, acc_fused_br, 3);
}

void dispatch_gluon_cpp(gluon_globals g) {
    int m = static_cast<int>(g.c.rows());
    int n = static_cast<int>(g.c.cols());
    const dim3 grid((m / BLK) * (n / BLK));
    mxfp4_gluon_cpp_kernel<<<grid, dim3(_NUM_THREADS), 0>>>(g);
}

void dispatch_operand_swap_debug(operand_swap_debug_globals g) {
    mxfp4_operand_swap_debug_kernel<<<1, dim3(_NUM_THREADS), 0>>>(g);
}

void dispatch_operand_swap_block(operand_swap_debug_globals g) {
    mxfp4_operand_swap_block_kernel<<<1, dim3(_NUM_THREADS), 0>>>(g);
}

void dispatch_active_block_debug(operand_swap_debug_globals g) {
    mxfp4_active_block_debug_kernel<<<1, dim3(_NUM_THREADS), 0>>>(g);
}

void dispatch_active_block_swap_step34(operand_swap_debug_globals g) {
    mxfp4_active_block_swap_step34_kernel<<<1, dim3(_NUM_THREADS), 0>>>(g);
}

void dispatch_active_block_swap_step34_with_lds(operand_swap_debug_globals g) {
    mxfp4_active_block_swap_step34_with_lds_kernel<<<1, dim3(_NUM_THREADS), 0>>>(g);
}

void dispatch_active_block_swap_step34_mainlike(operand_swap_debug_globals g) {
    mxfp4_active_block_swap_step34_mainlike_kernel<<<1, dim3(_NUM_THREADS), 0>>>(g);
}

void dispatch_active_block_swap_all_nonfused(operand_swap_debug_globals g) {
    mxfp4_active_block_swap_all_nonfused_kernel<<<1, dim3(_NUM_THREADS), 0>>>(g);
}

void dispatch_active_block_swap_all_fused(operand_swap_debug_globals g) {
    mxfp4_active_block_swap_all_fused_kernel<<<1, dim3(_NUM_THREADS), 0>>>(g);
}

void dispatch_step12_layout_debug(operand_swap_debug_globals g) {
    mxfp4_step12_layout_debug_kernel<<<1, dim3(_NUM_THREADS), 0>>>(g);
}

void dispatch_step12_swap_compare(operand_swap_debug_globals g) {
    mxfp4_step12_swap_compare_kernel<<<1, dim3(_NUM_THREADS), 0>>>(g);
}

PYBIND11_MODULE(tk_mxfp4_gluon_cpp, m) {
    m.doc() = "MXFP4 Gluon-arch kernel (C++ reimplementation)";
    py::bind_function<dispatch_gluon_cpp>(m, "gemm_rcr",
        &gluon_globals::a, &gluon_globals::b,
        &gluon_globals::a_scale, &gluon_globals::b_scale,
        &gluon_globals::c);
    py::bind_function<dispatch_operand_swap_debug>(m, "debug_operand_swap_raw",
        &operand_swap_debug_globals::a, &operand_swap_debug_globals::b,
        &operand_swap_debug_globals::a_scale, &operand_swap_debug_globals::b_scale,
        &operand_swap_debug_globals::out);
    py::bind_function<dispatch_operand_swap_block>(m, "debug_operand_swap_block",
        &operand_swap_debug_globals::a, &operand_swap_debug_globals::b,
        &operand_swap_debug_globals::a_scale, &operand_swap_debug_globals::b_scale,
        &operand_swap_debug_globals::out);
    py::bind_function<dispatch_active_block_debug>(m, "debug_active_block",
        &operand_swap_debug_globals::a, &operand_swap_debug_globals::b,
        &operand_swap_debug_globals::a_scale, &operand_swap_debug_globals::b_scale,
        &operand_swap_debug_globals::out);
    py::bind_function<dispatch_active_block_swap_step34>(m, "debug_active_block_swap_step34",
        &operand_swap_debug_globals::a, &operand_swap_debug_globals::b,
        &operand_swap_debug_globals::a_scale, &operand_swap_debug_globals::b_scale,
        &operand_swap_debug_globals::out);
    py::bind_function<dispatch_active_block_swap_step34_with_lds>(m, "debug_active_block_swap_step34_with_lds",
        &operand_swap_debug_globals::a, &operand_swap_debug_globals::b,
        &operand_swap_debug_globals::a_scale, &operand_swap_debug_globals::b_scale,
        &operand_swap_debug_globals::out);
    py::bind_function<dispatch_active_block_swap_step34_mainlike>(m, "debug_active_block_swap_step34_mainlike",
        &operand_swap_debug_globals::a, &operand_swap_debug_globals::b,
        &operand_swap_debug_globals::a_scale, &operand_swap_debug_globals::b_scale,
        &operand_swap_debug_globals::out);
    py::bind_function<dispatch_active_block_swap_all_nonfused>(m, "debug_active_block_swap_all_nonfused",
        &operand_swap_debug_globals::a, &operand_swap_debug_globals::b,
        &operand_swap_debug_globals::a_scale, &operand_swap_debug_globals::b_scale,
        &operand_swap_debug_globals::out);
    py::bind_function<dispatch_active_block_swap_all_fused>(m, "debug_active_block_swap_all_fused",
        &operand_swap_debug_globals::a, &operand_swap_debug_globals::b,
        &operand_swap_debug_globals::a_scale, &operand_swap_debug_globals::b_scale,
        &operand_swap_debug_globals::out);
    py::bind_function<dispatch_step12_layout_debug>(m, "debug_step12_layout",
        &operand_swap_debug_globals::a, &operand_swap_debug_globals::b,
        &operand_swap_debug_globals::a_scale, &operand_swap_debug_globals::b_scale,
        &operand_swap_debug_globals::out);
    py::bind_function<dispatch_step12_swap_compare>(m, "debug_step12_swap_compare",
        &operand_swap_debug_globals::a, &operand_swap_debug_globals::b,
        &operand_swap_debug_globals::a_scale, &operand_swap_debug_globals::b_scale,
        &operand_swap_debug_globals::out);
}
