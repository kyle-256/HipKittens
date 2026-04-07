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
using fp4_intx4_t   = int __attribute__((__vector_size__(4 * sizeof(int))));
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

__device__ __forceinline__ fp4_intx4_t fp4_lo4(const fp4_intx8_t& x) {
    return __builtin_shufflevector(x, x, 0, 1, 2, 3);
}
__device__ __forceinline__ fp4_intx4_t fp4_hi4(const fp4_intx8_t& x) {
    return __builtin_shufflevector(x, x, 4, 5, 6, 7);
}

// ── LDS addr helper for interleaved load+compute ──
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

// ── Combined 16 MFMAs + 8 ds_reads in ONE asm block ──
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

    // Outputs 0-15: acc, 16-23: ds_read dests
    // Inputs 24-27: a0-a3, 28-31: b0-b3, 32-35: sa0,sa1,sb0,sb1, 36-37: lds addrs
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

// Single MFMA via inline asm — no sched_barrier, no remap_phase overhead
template<int OPSEL_A, int OPSEL_B>
__device__ __forceinline__ void mfma_asm(
    fp4_floatx4_t& acc, fp4_intx4_t a, fp4_intx4_t b,
    unsigned sa, unsigned sb)
{
    if constexpr (OPSEL_A == 0 && OPSEL_B == 0) {
        asm volatile("v_mfma_scale_f32_16x16x128_f8f6f4 %0, %1, %2, %0, %3, %4"
            " op_sel_hi:[0,0,0] cbsz:4 blgp:4" : "+a"(acc) : "v"(a),"v"(b),"v"(sa),"v"(sb));
    } else if constexpr (OPSEL_A == 0 && OPSEL_B == 1) {
        asm volatile("v_mfma_scale_f32_16x16x128_f8f6f4 %0, %1, %2, %0, %3, %4"
            " op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4" : "+a"(acc) : "v"(a),"v"(b),"v"(sa),"v"(sb));
    } else if constexpr (OPSEL_A == 1 && OPSEL_B == 0) {
        asm volatile("v_mfma_scale_f32_16x16x128_f8f6f4 %0, %1, %2, %0, %3, %4"
            " op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4" : "+a"(acc) : "v"(a),"v"(b),"v"(sa),"v"(sb));
    } else {
        asm volatile("v_mfma_scale_f32_16x16x128_f8f6f4 %0, %1, %2, %0, %3, %4"
            " op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4" : "+a"(acc) : "v"(a),"v"(b),"v"(sa),"v"(sb));
    }
}

template<bool UPPER>
__device__ __forceinline__ void fp4_mma_v2(
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

    // 16 MFMAs in ONE asm block — hardware scheduler sees continuous stream
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

// ── Fast tile load: bypass TK's G::load to avoid per-call readfirstlane ──
// Pre-computes LDS byte offsets once, reuses every K iteration.

template<ducks::st::all ST, ducks::gl::all GL>
struct fast_tile_loader {
    using T = typename ST::dtype;
    static constexpr int BPT = 16;
    static constexpr int BPM = BPT * _NUM_THREADS;
    static constexpr int MPT = (ST::rows * ST::cols * sizeof(T)) / BPM;

    i32x4 srd;
    const void* base_ptr;
    uint32_t swizzled_voffs[MPT]; // per-lane VGPR offsets
    uint32_t lds_bytes[MPT];      // pre-computed LDS byte offsets (SGPR-friendly)

    __device__ void init(ST &dst, const GL &src, const uint32_t* so, uint32_t lds_base) {
        srd = std::bit_cast<i32x4>(make_buffer_resource(
            static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(src.raw_ptr)),
            0xFFFFFFFFu, 0x00110000u));
        srd[0] = __builtin_amdgcn_readfirstlane(srd[0]);
        srd[1] = __builtin_amdgcn_readfirstlane(srd[1]);
        srd[2] = __builtin_amdgcn_readfirstlane(srd[2]);
        srd[3] = __builtin_amdgcn_readfirstlane(srd[3]);
        base_ptr = reinterpret_cast<const void*>(src.raw_ptr);

        const uint32_t lds_tile_base = static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(&dst.data[0]));
        const uint32_t warp_offset = lds_base - lds_tile_base;

        #pragma unroll
        for (int i = 0; i < MPT; ++i) {
            swizzled_voffs[i] = so[i];
            const uint32_t lin = warp_offset + i * BPM;
            const uint32_t sid = lin / ST::underlying_subtile_bytes;
            lds_bytes[i] = lds_tile_base + lin + sid * ST::subtile_padding;
        }
    }

    template<ducks::coord::tile COORD>
    __device__ void load(ST &dst, const GL &src, const COORD &idx) {
        coord<> uc = idx.template unit_coord<2, 3>();
        T* gptr = (T*)&src[uc];
        uint32_t soff = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
            reinterpret_cast<const char*>(gptr) - reinterpret_cast<const char*>(base_ptr)));
        asm volatile("" : "+s"(soff));

        #pragma unroll
        for (int i = 0; i < MPT; ++i) {
            uint32_t lds_b = lds_bytes[i];
            asm volatile("" : "+s"(lds_b));
            llvm_amdgcn_raw_buffer_load_lds(
                std::bit_cast<int32x4_t>(srd),
                (as3_uint32_ptr)(uintptr_t)lds_b,
                16, swizzled_voffs[i], soff, 0,
                static_cast<int>(coherency::cache_all));
        }
    }
};

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
    #pragma unroll 8
    for (int p = 0; p < a_packs; ++p) {
        a0_srd[p] = make_scale_srd(preshuffled_scale_row_base_ptr(
            g.a_scale, (br * BLK + 0 * HB + wm * RBM + p * 32) >> 5));
        a1_srd[p] = make_scale_srd(preshuffled_scale_row_base_ptr(
            g.a_scale, (br * BLK + 1 * HB + wm * RBM + p * 32) >> 5));
    }
    #pragma unroll 8
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

    // ── Pre-compute SRD + LDS bases for tile loads (avoid repeated readfirstlane) ──
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

    // 16 buffer_load_dwordx4 per prefetch batch (4 tiles × 4 loads/tile/thread)
    constexpr int PREFETCH_VMCNT = 16;

    // ══════════════════ Prologue ══════════════════
    load_tiles(0, 0);
    if (k_byte_iters > 1) load_tiles(1, 1);

    // Prefetch scales for iteration 0
    fp8e8m0_4 pf_a0[a_packs], pf_a1[a_packs], pf_bl[b_packs], pf_br[b_packs];
    {
        const uint32_t soff0 = 0;
        #pragma unroll 8
        for (int p = 0; p < a_packs; ++p) {
            pf_a0[p] = load_pq_scale_srd(a0_srd[p], lane_soff, soff0);
            pf_a1[p] = load_pq_scale_srd(a1_srd[p], lane_soff, soff0);
        }
        #pragma unroll 8
        for (int p = 0; p < b_packs; ++p) {
            pf_bl[p] = load_pq_scale_srd(bl_srd[p], lane_soff, soff0);
            pf_br[p] = load_pq_scale_srd(br_srd[p], lane_soff, soff0);
        }
    }

    // ══════════════════ Main loop — pipelined LDS reads with partial lgkmcnt ══════════════════
    #pragma unroll 2
    for (int bt = 0; bt < k_byte_iters; ++bt) {
        const int cur = bt & 1;

        asm volatile("s_waitcnt vmcnt(16)");
        __builtin_amdgcn_s_barrier();

        // Capture prefetched scales
        fp8e8m0_4 a0_raw[a_packs], a1_raw[a_packs];
        fp8e8m0_4 bl_raw[b_packs], br_raw[b_packs];
        #pragma unroll
        for (int p = 0; p < a_packs; ++p) { a0_raw[p] = pf_a0[p]; a1_raw[p] = pf_a1[p]; }
        #pragma unroll
        for (int p = 0; p < b_packs; ++p) { bl_raw[p] = pf_bl[p]; br_raw[p] = pf_br[p]; }

        // Load only A0+Bl (16 ds_reads); Br and A1 loaded during Phase 0 MFMAs
        A_row_reg a0_rt;
        B_row_reg bl_rt2;
        fp4_load_st_to_rt(a0_rt, kittens::subtile_inplace<RBM, BK>(A0_db[cur], {wm, 0}));
        fp4_load_st_to_rt(bl_rt2, kittens::subtile_inplace<RBN, BK>(Bl_db[cur], {wn, 0}));

        auto br_sub = kittens::subtile_inplace<RBN, BK>(Br_db[cur], {wn, 0});
        uint32_t br_lds_p0, br_lds_p1;
        compute_lds_base_addrs<B_row_reg, std::remove_reference_t<decltype(br_sub)>>(
            br_sub, br_lds_p0, br_lds_p1);
        auto a1_sub = kittens::subtile_inplace<RBM, BK>(A1_db[cur], {wm, 0});
        uint32_t a1_lds_p0, a1_lds_p1;
        compute_lds_base_addrs<A_row_reg, std::remove_reference_t<decltype(a1_sub)>>(
            a1_sub, a1_lds_p0, a1_lds_p1);

        // Prefetch scales for bt+1
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

        fp4_intx8_t tA0[4], tBl[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            tA0[i] = fp4_extract_tile(a0_rt, i);
            tBl[i] = fp4_extract_tile(bl_rt2, i);
        }

        // ── Phase 0: A0×Bl — 16 MFMAs + load Br (8 ds_reads interleaved) ──
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

        // ── Phase 0: A0×Br — 16 MFMAs + load A1 (8 ds_reads interleaved) ──
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

        fp4_mma_v2<false>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw, 0);
        fp4_mma_v2<false>(acc_A1Br, tA1, tBr, a1_raw, br_raw, 0);

        // ── Barrier + tile prefetch ──
        __builtin_amdgcn_s_barrier();
        {
            const int pf_bt = (bt + 2 < k_byte_iters) ? (bt + 2) : (k_byte_iters - 1);
            load_tiles(pf_bt, cur);
        }

        // ── Phase 1 (64 MFMAs) ──
        fp4_mma_v2<true>(acc_A0Bl, tA0, tBl, a0_raw, bl_raw, 1);
        fp4_mma_v2<true>(acc_A0Br, tA0, tBr, a0_raw, br_raw, 1);
        fp4_mma_v2<true>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw, 1);
        fp4_mma_v2<true>(acc_A1Br, tA1, tBr, a1_raw, br_raw, 1);
    }

    // ══════════════════ Epilogue: store C ══════════════════
    auto store_acc = [&](const fp4_acc_v2& acc, int m_half, int n_half) {
        RT_C c_store;
        #pragma unroll 8
        for (int r = 0; r < (RBM / 16); ++r)
            #pragma unroll 8
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
