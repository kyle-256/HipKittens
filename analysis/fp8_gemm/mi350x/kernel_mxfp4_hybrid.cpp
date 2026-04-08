// MXFP4 Hybrid Kernel: C++ outer structure + inline ASM MFMA inner loops
//
// Based on V2 architecture (occupancy-1, 4 warps, 128KB LDS double-buffered)
// with pipeline restructuring inspired by gluon a4w4:
//   - Tile prefetch issued DURING MFMA execution (not after)
//   - 96 MFMAs overlap with buffer_load_to_lds (vs 0 overlap in V2)
//   - Scale prefetch overlapped with Phase 0 MFMAs
//
// Tile decomposition: A split by M-half (A0, A1), B split by N-half (Bl, Br)
// Per iteration: 128 MFMAs (64 phase-0 lo-K + 64 phase-1 hi-K)
// Phase 0 first 32 MFMAs interleave ds_reads for Br and A1 tiles.
// After all tile data is in VGPRs → barrier → issue tile prefetch → remaining 96 MFMAs

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

constexpr int BLK = 256;
constexpr int BK  = 128;
constexpr int WARPS_M = 2, WARPS_N = 2;
constexpr int _NUM_WARPS   = WARPS_M * WARPS_N;
constexpr int _NUM_THREADS = _NUM_WARPS * WARP_THREADS;
constexpr int HB = BLK / 2;

constexpr int RBM = HB / WARPS_M; // 64
constexpr int RBN = HB / WARPS_N; // 64

static_assert(BLK == 256 && BK == 128, "Hybrid requires BLK=256, BK=128");
static_assert(RBM == 64 && RBN == 64, "Expected 64x64 per-warp tile");

constexpr int K_BYTES = K_DIM / 2;
constexpr int k_byte_iters = K_BYTES / BK;

using ST_tile = st_fp8e4m3<HB, BK, st_16x128_s>;

using A_row_reg = rt_fp8e4m3<RBM, BK, row_l, rt_16x128_s>;
using B_row_reg = rt_fp8e4m3<RBN, BK, row_l, rt_16x128_s>;
using RT_C = rt_fl<RBM, RBN, col_l, rt_16x16_s>;

using G = kittens::group<_NUM_WARPS>;
using _gl_fp4   = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_scale = gl<fp8e8m0, -1, -1, -1, -1>;
using _gl_bf16  = gl<bf16, -1, -1, -1, -1>;

struct hybrid_globals {
    _gl_fp4 a, b;
    _gl_scale a_scale, b_scale;
    _gl_bf16 c;
    float scale = 1.0f;
    hipStream_t stream = nullptr;
    int m = 0, n = 0, k = 0;
};

// ── Vector types for MFMA operands ──

using fp4_intx8_t   = int __attribute__((__vector_size__(8 * sizeof(int))));
using fp4_intx4_t   = int __attribute__((__vector_size__(4 * sizeof(int))));
using fp4_floatx4_t = float __attribute__((__vector_size__(4 * sizeof(float))));

// ── MFMA helper: single mfma_scale via builtin ──

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

// ── Extract lo/hi 4-dword halves from 8-dword operand ──

__device__ __forceinline__ fp4_intx4_t fp4_lo4(const fp4_intx8_t& x) {
    return __builtin_shufflevector(x, x, 0, 1, 2, 3);
}
__device__ __forceinline__ fp4_intx4_t fp4_hi4(const fp4_intx8_t& x) {
    return __builtin_shufflevector(x, x, 4, 5, 6, 7);
}

// ── LDS → register tile load (ds_read_b128) ──

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
    #pragma unroll 8
    for (int k = 0; k < RT::base_tile_num_strides; k++) {
        #pragma unroll 8
        for (int i = 0; i < reg_sub_col; i++) {
            #pragma unroll 8
            for (int j = 0; j < reg_sub_row; j++) {
                const int row = i * RT::base_tile_rows + row_offset;
                const int col = j * RT::base_tile_cols + col_offset +
                    k * RT::base_tile_elements_per_stride_group;
                const uint32_t offset = sizeof(U) * (src_ptr + row * ST::underlying_subtile_cols + col);
                const uint32_t addr = offset ^ (((offset % (16 * 128)) >> 8) << 4);
                const int idx = k * RT::base_tile_stride / packing;
                #pragma unroll 8
                for (int ii = 0; ii < ST::subtiles_per_col; ii++) {
                    #pragma unroll 8
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

// ── Scale helpers ──

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

// ── Accumulator ──

struct alignas(16) fp4_acc_v2 {
    fp4_floatx4_t regs[(RBM / 16) * (RBN / 16)]; // 4×4 = 16 tiles
};

// ── LDS base addr helper for interleaved ds_read during MFMA ──

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

// ── 16 MFMAs + 8 ds_reads interleaved in one asm block ──
// Computes one 64×64 subtile (4 A-rows × 4 B-cols) while loading the next tile.

template<bool UPPER>
__device__ __forceinline__ void fp4_mma_with_lds(
    fp4_acc_v2& acc,
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_scales[], const fp8e8m0_4 b_scales[], int k_phase,
    float4 &d0, float4 &d1, float4 &d2, float4 &d3,
    float4 &d4, float4 &d5, float4 &d6, float4 &d7,
    uint32_t lds_a0, uint32_t lds_a1)
{
    constexpr int BC = RBN / 16;
    fp4_intx4_t a0,a1,a2,a3, b0,b1,b2,b3;
    if constexpr (!UPPER) {
        a0=fp4_lo4(A[0]); a1=fp4_lo4(A[1]); a2=fp4_lo4(A[2]); a3=fp4_lo4(A[3]);
        b0=fp4_lo4(B[0]); b1=fp4_lo4(B[1]); b2=fp4_lo4(B[2]); b3=fp4_lo4(B[3]);
    } else {
        a0=fp4_hi4(A[0]); a1=fp4_hi4(A[1]); a2=fp4_hi4(A[2]); a3=fp4_hi4(A[3]);
        b0=fp4_hi4(B[0]); b1=fp4_hi4(B[1]); b2=fp4_hi4(B[2]); b3=fp4_hi4(B[3]);
    }
    unsigned sa0 = std::bit_cast<unsigned>(remap_phase(a_scales[0], k_phase));
    unsigned sa1 = std::bit_cast<unsigned>(remap_phase(a_scales[1], k_phase));
    unsigned sb0 = std::bit_cast<unsigned>(remap_phase(b_scales[0], k_phase));
    unsigned sb1 = std::bit_cast<unsigned>(remap_phase(b_scales[1], k_phase));

    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %24, %28, %0,  %32, %34 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %24, %29, %1,  %32, %34 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %16, %36 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %24, %30, %2,  %32, %35 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %24, %31, %3,  %32, %35 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %17, %36 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %25, %28, %4,  %32, %34 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %25, %29, %5,  %32, %34 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %18, %36 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %25, %30, %6,  %32, %35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %25, %31, %7,  %32, %35 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %19, %36 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %26, %28, %8,  %33, %34 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %26, %29, %9,  %33, %34 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %20, %37 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %26, %30, %10, %33, %35 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %26, %31, %11, %33, %35 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %21, %37 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %27, %28, %12, %33, %34 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %27, %29, %13, %33, %34 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %22, %37 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %27, %30, %14, %33, %35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %27, %31, %15, %33, %35 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %23, %37 offset:6144\n"
        : "+a"(acc.regs[0*BC+0]), "+a"(acc.regs[0*BC+1]), "+a"(acc.regs[0*BC+2]), "+a"(acc.regs[0*BC+3]),
          "+a"(acc.regs[1*BC+0]), "+a"(acc.regs[1*BC+1]), "+a"(acc.regs[1*BC+2]), "+a"(acc.regs[1*BC+3]),
          "+a"(acc.regs[2*BC+0]), "+a"(acc.regs[2*BC+1]), "+a"(acc.regs[2*BC+2]), "+a"(acc.regs[2*BC+3]),
          "+a"(acc.regs[3*BC+0]), "+a"(acc.regs[3*BC+1]), "+a"(acc.regs[3*BC+2]), "+a"(acc.regs[3*BC+3]),
          "=v"(d0), "=v"(d1), "=v"(d2), "=v"(d3),
          "=v"(d4), "=v"(d5), "=v"(d6), "=v"(d7)
        : "v"(a0), "v"(a1), "v"(a2), "v"(a3),
          "v"(b0), "v"(b1), "v"(b2), "v"(b3),
          "v"(sa0), "v"(sa1), "v"(sb0), "v"(sb1),
          "v"(lds_a0), "v"(lds_a1)
    );
}

// ── Prefetch parameters for one tile (4 buffer_load_to_lds calls) ──

struct tile_pf_params {
    int32x4_t srd;
    uint32_t soff;
    uint32_t lds_addrs[4];
    uint32_t voffs[4];
};

static constexpr int PF_MPT = (HB * BK * sizeof(fp8e4m3)) / (16 * _NUM_THREADS); // 4

template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ __forceinline__ tile_pf_params make_pf_params(
    ST &dst, const GL &src, const COORD &idx,
    const uint32_t *so, i32x4 srd_in, const void *base_ptr, uint32_t lds_base)
{
    using T = typename ST::dtype;
    static_assert(PF_MPT == 4);
    constexpr int BPM = 16 * _NUM_THREADS;

    coord<> uc = idx.template unit_coord<2, 3>();
    T* gptr = (T*)&src[uc];
    uint32_t soff = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<const char*>(gptr) - reinterpret_cast<const char*>(base_ptr)));
    asm volatile("" : "+s"(soff));

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

__device__ __forceinline__ void emit_one_pf(const tile_pf_params &p, int idx) {
    uint32_t lds_b = p.lds_addrs[idx];
    asm volatile("" : "+s"(lds_b));
    llvm_amdgcn_raw_buffer_load_lds(
        std::bit_cast<int32x4_t>(p.srd),
        (as3_uint32_ptr)(uintptr_t)lds_b,
        16, p.voffs[idx], p.soff, 0,
        static_cast<int>(coherency::cache_all));
}

// ── 4 MFMAs for one row of the 4×4 output grid (ROW must be compile-time) ──

template<int ROW>
__device__ __forceinline__ void fp4_mma_row(
    fp4_acc_v2& acc,
    fp4_intx4_t a, fp4_intx4_t b0, fp4_intx4_t b1, fp4_intx4_t b2, fp4_intx4_t b3,
    unsigned sa, unsigned sb0, unsigned sb1)
{
    constexpr int BC = RBN / 16;
    constexpr int BASE = ROW * BC;
    if constexpr ((ROW & 1) == 0) {
        asm volatile(
            "v_mfma_scale_f32_16x16x128_f8f6f4 %0, %4, %5, %0, %8, %10 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %1, %4, %6, %1, %8, %10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %2, %4, %7, %2, %8, %11 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %3, %4, %9, %3, %8, %11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            : "+a"(acc.regs[BASE+0]), "+a"(acc.regs[BASE+1]),
              "+a"(acc.regs[BASE+2]), "+a"(acc.regs[BASE+3])
            : "v"(a), "v"(b0), "v"(b1), "v"(b2),
              "v"(sa), "v"(b3), "v"(sb0), "v"(sb1)
        );
    } else {
        asm volatile(
            "v_mfma_scale_f32_16x16x128_f8f6f4 %0, %4, %5, %0, %8, %10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %1, %4, %6, %1, %8, %10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %2, %4, %7, %2, %8, %11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %3, %4, %9, %3, %8, %11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            : "+a"(acc.regs[BASE+0]), "+a"(acc.regs[BASE+1]),
              "+a"(acc.regs[BASE+2]), "+a"(acc.regs[BASE+3])
            : "v"(a), "v"(b0), "v"(b1), "v"(b2),
              "v"(sa), "v"(b3), "v"(sb0), "v"(sb1)
        );
    }
}

// ── 16 MFMAs + 4 buffer_load_to_lds, split into 4 asm blocks ──
// Each block has 4 MFMAs and claims ALL 16 accumulators as "+a"
// to prevent hipcc from reusing AGPRs across blocks.
// buffer_load_to_lds uses compiler builtin (correct SRD handling).

template<bool UPPER, int PF_N = PF_MPT>
__device__ __forceinline__ void fp4_mma_with_tile_prefetch(
    fp4_acc_v2& acc,
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_scales[], const fp8e8m0_4 b_scales[],
    int k_phase,
    const tile_pf_params &pf, int pf_start = 0)
{
    constexpr int BC = RBN / 16;
    fp4_intx4_t a0, a1, a2, a3, b0, b1, b2, b3;
    if constexpr (!UPPER) {
        a0=fp4_lo4(A[0]); a1=fp4_lo4(A[1]); a2=fp4_lo4(A[2]); a3=fp4_lo4(A[3]);
        b0=fp4_lo4(B[0]); b1=fp4_lo4(B[1]); b2=fp4_lo4(B[2]); b3=fp4_lo4(B[3]);
    } else {
        a0=fp4_hi4(A[0]); a1=fp4_hi4(A[1]); a2=fp4_hi4(A[2]); a3=fp4_hi4(A[3]);
        b0=fp4_hi4(B[0]); b1=fp4_hi4(B[1]); b2=fp4_hi4(B[2]); b3=fp4_hi4(B[3]);
    }
    unsigned sa0 = std::bit_cast<unsigned>(remap_phase(a_scales[0], k_phase));
    unsigned sa1 = std::bit_cast<unsigned>(remap_phase(a_scales[1], k_phase));
    unsigned sb0 = std::bit_cast<unsigned>(remap_phase(b_scales[0], k_phase));
    unsigned sb1 = std::bit_cast<unsigned>(remap_phase(b_scales[1], k_phase));

    // 4 rows of 4 MFMAs each. Prefetch loads distributed based on pf_start/pf_count.
    // Row 0
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %16, %20, %0,  %24, %26 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %16, %21, %1,  %24, %26 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %16, %22, %2,  %24, %27 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %16, %23, %3,  %24, %27 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        : "+a"(acc.regs[0*BC+0]), "+a"(acc.regs[0*BC+1]), "+a"(acc.regs[0*BC+2]), "+a"(acc.regs[0*BC+3]),
          "+a"(acc.regs[1*BC+0]), "+a"(acc.regs[1*BC+1]), "+a"(acc.regs[1*BC+2]), "+a"(acc.regs[1*BC+3]),
          "+a"(acc.regs[2*BC+0]), "+a"(acc.regs[2*BC+1]), "+a"(acc.regs[2*BC+2]), "+a"(acc.regs[2*BC+3]),
          "+a"(acc.regs[3*BC+0]), "+a"(acc.regs[3*BC+1]), "+a"(acc.regs[3*BC+2]), "+a"(acc.regs[3*BC+3])
        : "v"(a0), "v"(a1), "v"(a2), "v"(a3),
          "v"(b0), "v"(b1), "v"(b2), "v"(b3),
          "v"(sa0), "v"(sa1), "v"(sb0), "v"(sb1)
    );
    if constexpr (PF_N > 0) emit_one_pf(pf, pf_start);
    // Row 1
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %17, %20, %4,  %24, %26 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %17, %21, %5,  %24, %26 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %17, %22, %6,  %24, %27 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %17, %23, %7,  %24, %27 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        : "+a"(acc.regs[0*BC+0]), "+a"(acc.regs[0*BC+1]), "+a"(acc.regs[0*BC+2]), "+a"(acc.regs[0*BC+3]),
          "+a"(acc.regs[1*BC+0]), "+a"(acc.regs[1*BC+1]), "+a"(acc.regs[1*BC+2]), "+a"(acc.regs[1*BC+3]),
          "+a"(acc.regs[2*BC+0]), "+a"(acc.regs[2*BC+1]), "+a"(acc.regs[2*BC+2]), "+a"(acc.regs[2*BC+3]),
          "+a"(acc.regs[3*BC+0]), "+a"(acc.regs[3*BC+1]), "+a"(acc.regs[3*BC+2]), "+a"(acc.regs[3*BC+3])
        : "v"(a0), "v"(a1), "v"(a2), "v"(a3),
          "v"(b0), "v"(b1), "v"(b2), "v"(b3),
          "v"(sa0), "v"(sa1), "v"(sb0), "v"(sb1)
    );
    if constexpr (PF_N > 1) emit_one_pf(pf, pf_start + 1);
    // Row 2
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %18, %20, %8,  %25, %26 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %18, %21, %9,  %25, %26 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %18, %22, %10, %25, %27 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %18, %23, %11, %25, %27 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        : "+a"(acc.regs[0*BC+0]), "+a"(acc.regs[0*BC+1]), "+a"(acc.regs[0*BC+2]), "+a"(acc.regs[0*BC+3]),
          "+a"(acc.regs[1*BC+0]), "+a"(acc.regs[1*BC+1]), "+a"(acc.regs[1*BC+2]), "+a"(acc.regs[1*BC+3]),
          "+a"(acc.regs[2*BC+0]), "+a"(acc.regs[2*BC+1]), "+a"(acc.regs[2*BC+2]), "+a"(acc.regs[2*BC+3]),
          "+a"(acc.regs[3*BC+0]), "+a"(acc.regs[3*BC+1]), "+a"(acc.regs[3*BC+2]), "+a"(acc.regs[3*BC+3])
        : "v"(a0), "v"(a1), "v"(a2), "v"(a3),
          "v"(b0), "v"(b1), "v"(b2), "v"(b3),
          "v"(sa0), "v"(sa1), "v"(sb0), "v"(sb1)
    );
    if constexpr (PF_N > 2) emit_one_pf(pf, pf_start + 2);
    // Row 3
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %19, %20, %12, %25, %26 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %19, %21, %13, %25, %26 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %19, %22, %14, %25, %27 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %19, %23, %15, %25, %27 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        : "+a"(acc.regs[0*BC+0]), "+a"(acc.regs[0*BC+1]), "+a"(acc.regs[0*BC+2]), "+a"(acc.regs[0*BC+3]),
          "+a"(acc.regs[1*BC+0]), "+a"(acc.regs[1*BC+1]), "+a"(acc.regs[1*BC+2]), "+a"(acc.regs[1*BC+3]),
          "+a"(acc.regs[2*BC+0]), "+a"(acc.regs[2*BC+1]), "+a"(acc.regs[2*BC+2]), "+a"(acc.regs[2*BC+3]),
          "+a"(acc.regs[3*BC+0]), "+a"(acc.regs[3*BC+1]), "+a"(acc.regs[3*BC+2]), "+a"(acc.regs[3*BC+3])
        : "v"(a0), "v"(a1), "v"(a2), "v"(a3),
          "v"(b0), "v"(b1), "v"(b2), "v"(b3),
          "v"(sa0), "v"(sa1), "v"(sb0), "v"(sb1)
    );
    if constexpr (PF_N > 3) emit_one_pf(pf, pf_start + 3);
}

// ── 16 MFMAs in one asm block (pure compute, no interleaved memory) ──

template<bool UPPER>
__device__ __forceinline__ void fp4_mma_pure(
    fp4_acc_v2& acc,
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_scales[], const fp8e8m0_4 b_scales[],
    int k_phase)
{
    constexpr int BC = RBN / 16;

    fp4_intx4_t a0, a1, a2, a3, b0, b1, b2, b3;
    if constexpr (!UPPER) {
        a0=fp4_lo4(A[0]); a1=fp4_lo4(A[1]); a2=fp4_lo4(A[2]); a3=fp4_lo4(A[3]);
        b0=fp4_lo4(B[0]); b1=fp4_lo4(B[1]); b2=fp4_lo4(B[2]); b3=fp4_lo4(B[3]);
    } else {
        a0=fp4_hi4(A[0]); a1=fp4_hi4(A[1]); a2=fp4_hi4(A[2]); a3=fp4_hi4(A[3]);
        b0=fp4_hi4(B[0]); b1=fp4_hi4(B[1]); b2=fp4_hi4(B[2]); b3=fp4_hi4(B[3]);
    }

    unsigned sa0 = std::bit_cast<unsigned>(remap_phase(a_scales[0], k_phase));
    unsigned sa1 = std::bit_cast<unsigned>(remap_phase(a_scales[1], k_phase));
    unsigned sb0 = std::bit_cast<unsigned>(remap_phase(b_scales[0], k_phase));
    unsigned sb1 = std::bit_cast<unsigned>(remap_phase(b_scales[1], k_phase));

    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %16, %20, %0,  %24, %26 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %16, %21, %1,  %24, %26 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %16, %22, %2,  %24, %27 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %16, %23, %3,  %24, %27 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %17, %20, %4,  %24, %26 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %17, %21, %5,  %24, %26 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %17, %22, %6,  %24, %27 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %17, %23, %7,  %24, %27 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %18, %20, %8,  %25, %26 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %18, %21, %9,  %25, %26 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %18, %22, %10, %25, %27 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %18, %23, %11, %25, %27 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %19, %20, %12, %25, %26 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %19, %21, %13, %25, %26 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %19, %22, %14, %25, %27 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %19, %23, %15, %25, %27 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        : "+a"(acc.regs[0*BC+0]), "+a"(acc.regs[0*BC+1]), "+a"(acc.regs[0*BC+2]), "+a"(acc.regs[0*BC+3]),
          "+a"(acc.regs[1*BC+0]), "+a"(acc.regs[1*BC+1]), "+a"(acc.regs[1*BC+2]), "+a"(acc.regs[1*BC+3]),
          "+a"(acc.regs[2*BC+0]), "+a"(acc.regs[2*BC+1]), "+a"(acc.regs[2*BC+2]), "+a"(acc.regs[2*BC+3]),
          "+a"(acc.regs[3*BC+0]), "+a"(acc.regs[3*BC+1]), "+a"(acc.regs[3*BC+2]), "+a"(acc.regs[3*BC+3])
        : "v"(a0), "v"(a1), "v"(a2), "v"(a3),
          "v"(b0), "v"(b1), "v"(b2), "v"(b3),
          "v"(sa0), "v"(sa1), "v"(sb0), "v"(sb1)
    );
}

// ── Main kernel ──

__global__ __launch_bounds__(_NUM_THREADS, 1)
void mxfp4_hybrid_kernel(const hybrid_globals g) {
    static_assert(K_BYTES % BK == 0 && N_DIM % BLK == 0 && M_DIM % BLK == 0);

    constexpr int bpc = N_DIM / BLK;

    // 8 tiles × 16KB = 128KB LDS, double-buffered
    __shared__ ST_tile A0_db[2];
    __shared__ ST_tile A1_db[2];
    __shared__ ST_tile Bl_db[2];
    __shared__ ST_tile Br_db[2];

    const int bid = blockIdx.x;
    const int br = bid / bpc, bc = bid % bpc;
    const int wm = warpid() / WARPS_N, wn = warpid() % WARPS_N;

    // ── Tile load swizzled offsets (computed once) ──
    constexpr int bpt = ST_tile::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * _NUM_THREADS;
    constexpr int mpt = HB * BK * sizeof(fp8e4m3) / bpm;
    uint32_t so_a[mpt], so_b[mpt];
    G::prefill_swizzled_offsets(A0_db[0], g.a, so_a);
    G::prefill_swizzled_offsets(Bl_db[0], g.b, so_b);

    // ── Scale SRDs ──
    constexpr int a_packs = RBM / 32;
    constexpr int b_packs = RBN / 32;
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

    // ── Accumulators ──
    fp4_acc_v2 acc_A0Bl{}, acc_A0Br{}, acc_A1Bl{}, acc_A1Br{};

    // ── Tile SRDs for fast buffer_load_to_lds ──
    i32x4 srd_a = std::bit_cast<i32x4>(make_buffer_resource(
        static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(g.a.raw_ptr)),
        0xFFFFFFFFu, 0x00110000u));
    srd_a[0] = __builtin_amdgcn_readfirstlane(srd_a[0]);
    srd_a[1] = __builtin_amdgcn_readfirstlane(srd_a[1]);
    srd_a[2] = __builtin_amdgcn_readfirstlane(srd_a[2]);
    srd_a[3] = __builtin_amdgcn_readfirstlane(srd_a[3]);
    const void* base_a = reinterpret_cast<const void*>(g.a.raw_ptr);

    i32x4 srd_b = std::bit_cast<i32x4>(make_buffer_resource(
        static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(g.b.raw_ptr)),
        0xFFFFFFFFu, 0x00110000u));
    srd_b[0] = __builtin_amdgcn_readfirstlane(srd_b[0]);
    srd_b[1] = __builtin_amdgcn_readfirstlane(srd_b[1]);
    srd_b[2] = __builtin_amdgcn_readfirstlane(srd_b[2]);
    srd_b[3] = __builtin_amdgcn_readfirstlane(srd_b[3]);
    const void* base_b = reinterpret_cast<const void*>(g.b.raw_ptr);

    constexpr int epw = 16 / sizeof(fp8e4m3) * WARP_THREADS;
    const uint32_t warp_lds_off = (warpid() % _NUM_WARPS) * epw * sizeof(fp8e4m3);
    auto lds_base_of = [&](auto &tile) -> uint32_t {
        return __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(&tile.data[0]) + warp_lds_off));
    };
    uint32_t lb_a0[2], lb_a1[2], lb_bl[2], lb_br[2];
    #pragma unroll
    for (int db = 0; db < 2; ++db) {
        lb_a0[db] = lds_base_of(A0_db[db]);
        lb_a1[db] = lds_base_of(A1_db[db]);
        lb_bl[db] = lds_base_of(Bl_db[db]);
        lb_br[db] = lds_base_of(Br_db[db]);
    }

    auto load_tiles = [&](int bt, int db) {
        G::load(A0_db[db], g.a, {0, 0, br * 2,     bt}, so_a, srd_a, base_a, lb_a0[db]);
        G::load(A1_db[db], g.a, {0, 0, br * 2 + 1, bt}, so_a, srd_a, base_a, lb_a1[db]);
        G::load(Bl_db[db], g.b, {0, 0, bc * 2,     bt}, so_b, srd_b, base_b, lb_bl[db]);
        G::load(Br_db[db], g.b, {0, 0, bc * 2 + 1, bt}, so_b, srd_b, base_b, lb_br[db]);
    };

    // ════════════════ Prologue: prefetch 2 iterations ════════════════
    load_tiles(0, 0);
    if (k_byte_iters > 1) load_tiles(1, 1);

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

    // ════════════════ Main loop ════════════════
    // Pipeline: after extracting all 4 tiles to VGPRs, issue barrier + prefetch,
    // then run remaining 96 MFMAs overlapping with prefetch completion.

    #pragma unroll 2
    for (int bt = 0; bt < k_byte_iters; ++bt) {
        const int cur = bt & 1;

        // Wait for current tiles in LDS.
        // k_byte_iters >= 2: prologue issued 32 tile loads + 8 scale loads = 40 vmem ops.
        //   vmcnt(16) keeps db[other] tiles (8) + scales (8) in flight. db[cur] ready.
        // k_byte_iters == 1: prologue issued 16 tile loads + 8 scale loads = 24 vmem ops.
        //   vmcnt(8) keeps only scales (8) in flight. All tiles ready.
        if constexpr (k_byte_iters >= 2) {
            asm volatile("s_waitcnt vmcnt(16)");
        } else {
            asm volatile("s_waitcnt vmcnt(8)");
        }
        __builtin_amdgcn_s_barrier();

        // Capture prefetched scales for this iteration
        fp8e8m0_4 a0_raw[a_packs], a1_raw[a_packs];
        fp8e8m0_4 bl_raw[b_packs], br_raw[b_packs];
        #pragma unroll
        for (int p = 0; p < a_packs; ++p) { a0_raw[p] = pf_a0[p]; a1_raw[p] = pf_a1[p]; }
        #pragma unroll
        for (int p = 0; p < b_packs; ++p) { bl_raw[p] = pf_bl[p]; br_raw[p] = pf_br[p]; }

        // ── Step 1: Load A0 + Bl from LDS to register tiles ──
        A_row_reg a0_rt;
        B_row_reg bl_rt;
        fp4_load_st_to_rt(a0_rt, kittens::subtile_inplace<RBM, BK>(A0_db[cur], {wm, 0}));
        fp4_load_st_to_rt(bl_rt, kittens::subtile_inplace<RBN, BK>(Bl_db[cur], {wn, 0}));

        // Compute LDS addresses for Br and A1 (to be loaded during MFMAs)
        auto br_sub = kittens::subtile_inplace<RBN, BK>(Br_db[cur], {wn, 0});
        uint32_t br_lds_p0, br_lds_p1;
        compute_lds_base_addrs<B_row_reg, std::remove_reference_t<decltype(br_sub)>>(
            br_sub, br_lds_p0, br_lds_p1);
        auto a1_sub = kittens::subtile_inplace<RBM, BK>(A1_db[cur], {wm, 0});
        uint32_t a1_lds_p0, a1_lds_p1;
        compute_lds_base_addrs<A_row_reg, std::remove_reference_t<decltype(a1_sub)>>(
            a1_sub, a1_lds_p0, a1_lds_p1);

        // Prefetch scales for next iteration
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

        // Wait for A0+Bl ds_reads, allow scale loads to stay in flight
        asm volatile("s_waitcnt lgkmcnt(0) vmcnt(8)");

        fp4_intx8_t tA0[4], tBl[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            tA0[i] = fp4_extract_tile(a0_rt, i);
            tBl[i] = fp4_extract_tile(bl_rt, i);
        }

        // ── Step 2: A0×Bl MFMAs + interleaved ds_read for Br ──
        float4 br_d[8];
        fp4_mma_with_lds<false>(acc_A0Bl, tA0, tBl, a0_raw, bl_raw, 0,
            br_d[0], br_d[1], br_d[2], br_d[3],
            br_d[4], br_d[5], br_d[6], br_d[7],
            br_lds_p0, br_lds_p1);

        asm volatile("s_waitcnt lgkmcnt(0)");

        fp4_intx8_t tBr[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            auto lo = *reinterpret_cast<fp4_intx4_t*>(&br_d[i]);
            auto hi = *reinterpret_cast<fp4_intx4_t*>(&br_d[i + 4]);
            tBr[i][0]=lo[0]; tBr[i][1]=lo[1]; tBr[i][2]=lo[2]; tBr[i][3]=lo[3];
            tBr[i][4]=hi[0]; tBr[i][5]=hi[1]; tBr[i][6]=hi[2]; tBr[i][7]=hi[3];
        }

        // ── Step 3: A0×Br MFMAs + interleaved ds_read for A1 ──
        float4 a1_d[8];
        fp4_mma_with_lds<false>(acc_A0Br, tA0, tBr, a0_raw, br_raw, 0,
            a1_d[0], a1_d[1], a1_d[2], a1_d[3],
            a1_d[4], a1_d[5], a1_d[6], a1_d[7],
            a1_lds_p0, a1_lds_p1);

        asm volatile("s_waitcnt lgkmcnt(0)");

        fp4_intx8_t tA1[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            auto lo = *reinterpret_cast<fp4_intx4_t*>(&a1_d[i]);
            auto hi = *reinterpret_cast<fp4_intx4_t*>(&a1_d[i + 4]);
            tA1[i][0]=lo[0]; tA1[i][1]=lo[1]; tA1[i][2]=lo[2]; tA1[i][3]=lo[3];
            tA1[i][4]=hi[0]; tA1[i][5]=hi[1]; tA1[i][6]=hi[2]; tA1[i][7]=hi[3];
        }

        // ══════════════════════════════════════════════════════════════
        // All tile data now in VGPRs (tA0, tBl, tBr, tA1).
        // LDS db[cur] no longer needed — safe to barrier and prefetch.
        // 16 buffer_load_to_lds interleaved across 96 MFMAs below.
        // ══════════════════════════════════════════════════════════════

        __builtin_amdgcn_s_barrier();

        const int pf_bt = (bt + 2 < k_byte_iters) ? (bt + 2) : (k_byte_iters - 1);

        tile_pf_params pf_a0_p = make_pf_params(A0_db[cur], g.a, {0, 0, br * 2,     pf_bt}, so_a, srd_a, base_a, lb_a0[cur]);
        tile_pf_params pf_a1_p = make_pf_params(A1_db[cur], g.a, {0, 0, br * 2 + 1, pf_bt}, so_a, srd_a, base_a, lb_a1[cur]);
        tile_pf_params pf_bl_p = make_pf_params(Bl_db[cur], g.b, {0, 0, bc * 2,     pf_bt}, so_b, srd_b, base_b, lb_bl[cur]);
        tile_pf_params pf_br_p = make_pf_params(Br_db[cur], g.b, {0, 0, bc * 2 + 1, pf_bt}, so_b, srd_b, base_b, lb_br[cur]);

        // ── Step 4-5: 96 MFMAs with 16 tile prefetch loads distributed across all 6 blocks ──
        // Blocks 3-6: 2 loads each = 8 (tiles A0, A1)
        // Blocks 7-8: 4 loads each = 8 (tiles Bl, Br)
        fp4_mma_with_tile_prefetch<false>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw, 0, pf_a0_p);
        fp4_mma_with_tile_prefetch<false>(acc_A1Br, tA1, tBr, a1_raw, br_raw, 0, pf_a1_p);
        fp4_mma_with_tile_prefetch<true>(acc_A0Bl, tA0, tBl, a0_raw, bl_raw, 1, pf_bl_p);
        fp4_mma_with_tile_prefetch<true>(acc_A0Br, tA0, tBr, a0_raw, br_raw, 1, pf_br_p);
        // Last 2 blocks: split into 4×4 MFMA (no loads) to give compiler scheduling freedom
        static const tile_pf_params dummy_pf{};
        fp4_mma_with_tile_prefetch<true, 0>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw, 1, dummy_pf);
        fp4_mma_with_tile_prefetch<true, 0>(acc_A1Br, tA1, tBr, a1_raw, br_raw, 1, dummy_pf);
    }

    // ════════════════ Epilogue: store C ════════════════

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

void dispatch_hybrid(hybrid_globals g) {
    g.m = static_cast<int>(g.c.rows());
    g.n = static_cast<int>(g.c.cols());
    g.k = static_cast<int>(g.a.cols()) * 2;
    const dim3 grid((g.m / BLK) * (g.n / BLK));
    mxfp4_hybrid_kernel<<<grid, dim3(_NUM_THREADS), 0, g.stream>>>(g);
}

PYBIND11_MODULE(tk_mxfp4_hybrid, m) {
    m.doc() = "MXFP4 hybrid kernel: C++ structure + inline ASM MFMAs";
    py::bind_function<dispatch_hybrid>(m, "gemm_rcr",
        &hybrid_globals::a, &hybrid_globals::b,
        &hybrid_globals::a_scale, &hybrid_globals::b_scale,
        &hybrid_globals::c);
}
