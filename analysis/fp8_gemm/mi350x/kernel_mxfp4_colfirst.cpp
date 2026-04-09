// MXFP4 KPAIR Column-First Kernel: processes both K-phases per accumulator
//
// Same V2 architecture as hybrid (occupancy-1, 4 warps, 128KB LDS double-buffered)
// KPAIR pipeline: each 32-MFMA block computes lo+hi phases per accumulator.
// Steps 2-3: 32 MFMAs each (KPAIR A0×Bl, A0×Br) + ds_reads.
// Step 5: 32+32 MFMAs (KPAIR A1×Bl, A1×Br) + 16 tile prefetch loads.
// Raw 32-bit scales with op_sel_hi for phase selection (no remap_phase).

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

struct colfirst_globals {
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
        // Col 0: a[0..3] × b0
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %24, %28, %0,  %32, %34 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %25, %28, %4,  %32, %34 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %16, %36 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %26, %28, %8,  %33, %34 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %27, %28, %12, %33, %34 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %17, %36 offset:2048\n"
        // Col 1: a[0..3] × b1
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %24, %29, %1,  %32, %34 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %25, %29, %5,  %32, %34 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %18, %36 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %26, %29, %9,  %33, %34 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %27, %29, %13, %33, %34 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %19, %36 offset:6144\n"
        // Col 2: a[0..3] × b2
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %24, %30, %2,  %32, %35 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %25, %30, %6,  %32, %35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %20, %37 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %26, %30, %10, %33, %35 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %27, %30, %14, %33, %35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %21, %37 offset:2048\n"
        // Col 3: a[0..3] × b3
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %24, %31, %3,  %32, %35 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %25, %31, %7,  %32, %35 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %22, %37 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %26, %31, %11, %33, %35 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %27, %31, %15, %33, %35 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %23, %37 offset:6144\n"
        : "+a"(acc.regs[0*BC+0]), "+a"(acc.regs[0*BC+1]), "+a"(acc.regs[0*BC+2]), "+a"(acc.regs[0*BC+3]),
          "+a"(acc.regs[1*BC+0]), "+a"(acc.regs[1*BC+1]), "+a"(acc.regs[1*BC+2]), "+a"(acc.regs[1*BC+3]),
          "+a"(acc.regs[2*BC+0]), "+a"(acc.regs[2*BC+1]), "+a"(acc.regs[2*BC+2]), "+a"(acc.regs[2*BC+3]),
          "+a"(acc.regs[3*BC+0]), "+a"(acc.regs[3*BC+1]), "+a"(acc.regs[3*BC+2]), "+a"(acc.regs[3*BC+3]),
          "=&v"(d0), "=&v"(d1), "=&v"(d2), "=&v"(d3),
          "=&v"(d4), "=&v"(d5), "=&v"(d6), "=&v"(d7)
        : "v"(a0), "v"(a1), "v"(a2), "v"(a3),
          "v"(b0), "v"(b1), "v"(b2), "v"(b3),
          "v"(sa0), "v"(sa1), "v"(sb0), "v"(sb1),
          "v"(lds_a0), "v"(lds_a1)
    );
}

// ── KPAIR: 32 MFMAs + 8 ds_reads interleaved in one asm block ──
// Processes BOTH lo and hi K-phases per accumulator using raw 32-bit scales.
// op_sel_hi:[0,0,0] selects lo phase, op_sel_hi:[1,1,0] selects hi phase.

__device__ __forceinline__ void fp4_mma_kpair_with_lds(
    fp4_acc_v2& acc,
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_scales[], const fp8e8m0_4 b_scales[],
    float4 &d0, float4 &d1, float4 &d2, float4 &d3,
    float4 &d4, float4 &d5, float4 &d6, float4 &d7,
    uint32_t lds_a0, uint32_t lds_a1)
{
    constexpr int BC = RBN / 16;
    fp4_intx4_t a0l=fp4_lo4(A[0]), a1l=fp4_lo4(A[1]), a2l=fp4_lo4(A[2]), a3l=fp4_lo4(A[3]);
    fp4_intx4_t a0h=fp4_hi4(A[0]), a1h=fp4_hi4(A[1]), a2h=fp4_hi4(A[2]), a3h=fp4_hi4(A[3]);
    fp4_intx4_t b0l=fp4_lo4(B[0]), b1l=fp4_lo4(B[1]), b2l=fp4_lo4(B[2]), b3l=fp4_lo4(B[3]);
    fp4_intx4_t b0h=fp4_hi4(B[0]), b1h=fp4_hi4(B[1]), b2h=fp4_hi4(B[2]), b3h=fp4_hi4(B[3]);
    unsigned sa0 = std::bit_cast<unsigned>(a_scales[0]);
    unsigned sa1 = std::bit_cast<unsigned>(a_scales[1]);
    unsigned sb0 = std::bit_cast<unsigned>(b_scales[0]);
    unsigned sb1 = std::bit_cast<unsigned>(b_scales[1]);

    asm volatile(
        // Col 0: 4 rows × lo+hi (8 MFMAs)
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %24, %32, %0,  %40, %42 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %28, %36, %0,  %40, %42 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %25, %32, %4,  %40, %42 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %29, %36, %4,  %40, %42 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %26, %32, %8,  %41, %42 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %30, %36, %8,  %41, %42 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %27, %32, %12, %41, %42 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %31, %36, %12, %41, %42 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %16, %44 offset:0\n"
        "ds_read_b128 %17, %44 offset:2048\n"
        // Col 1: 4 rows × lo+hi (8 MFMAs)
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %24, %33, %1,  %40, %42 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %28, %37, %1,  %40, %42 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %25, %33, %5,  %40, %42 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %29, %37, %5,  %40, %42 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %26, %33, %9,  %41, %42 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %30, %37, %9,  %41, %42 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %27, %33, %13, %41, %42 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %31, %37, %13, %41, %42 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %18, %44 offset:4096\n"
        "ds_read_b128 %19, %44 offset:6144\n"
        // Col 2: 4 rows × lo+hi (8 MFMAs)
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %24, %34, %2,  %40, %43 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %28, %38, %2,  %40, %43 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %25, %34, %6,  %40, %43 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %29, %38, %6,  %40, %43 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %26, %34, %10, %41, %43 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %30, %38, %10, %41, %43 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %27, %34, %14, %41, %43 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %31, %38, %14, %41, %43 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %20, %45 offset:0\n"
        "ds_read_b128 %21, %45 offset:2048\n"
        // Col 3: 4 rows × lo+hi (8 MFMAs)
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %24, %35, %3,  %40, %43 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %28, %39, %3,  %40, %43 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %25, %35, %7,  %40, %43 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %29, %39, %7,  %40, %43 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %26, %35, %11, %41, %43 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %30, %39, %11, %41, %43 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %27, %35, %15, %41, %43 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %31, %39, %15, %41, %43 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %22, %45 offset:4096\n"
        "ds_read_b128 %23, %45 offset:6144\n"
        : "+a"(acc.regs[0*BC+0]), "+a"(acc.regs[0*BC+1]), "+a"(acc.regs[0*BC+2]), "+a"(acc.regs[0*BC+3]),
          "+a"(acc.regs[1*BC+0]), "+a"(acc.regs[1*BC+1]), "+a"(acc.regs[1*BC+2]), "+a"(acc.regs[1*BC+3]),
          "+a"(acc.regs[2*BC+0]), "+a"(acc.regs[2*BC+1]), "+a"(acc.regs[2*BC+2]), "+a"(acc.regs[2*BC+3]),
          "+a"(acc.regs[3*BC+0]), "+a"(acc.regs[3*BC+1]), "+a"(acc.regs[3*BC+2]), "+a"(acc.regs[3*BC+3]),
          "=&v"(d0), "=&v"(d1), "=&v"(d2), "=&v"(d3),
          "=&v"(d4), "=&v"(d5), "=&v"(d6), "=&v"(d7)
        : "v"(a0l), "v"(a1l), "v"(a2l), "v"(a3l),
          "v"(a0h), "v"(a1h), "v"(a2h), "v"(a3h),
          "v"(b0l), "v"(b1l), "v"(b2l), "v"(b3l),
          "v"(b0h), "v"(b1h), "v"(b2h), "v"(b3h),
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

    // 4 columns of 4 MFMAs each. Prefetch loads distributed after each column block.
    // Col 0: a[0..3] × b0
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %16, %20, %0,  %24, %26 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %17, %20, %4,  %24, %26 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %18, %20, %8,  %25, %26 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %19, %20, %12, %25, %26 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        : "+a"(acc.regs[0*BC+0]), "+a"(acc.regs[0*BC+1]), "+a"(acc.regs[0*BC+2]), "+a"(acc.regs[0*BC+3]),
          "+a"(acc.regs[1*BC+0]), "+a"(acc.regs[1*BC+1]), "+a"(acc.regs[1*BC+2]), "+a"(acc.regs[1*BC+3]),
          "+a"(acc.regs[2*BC+0]), "+a"(acc.regs[2*BC+1]), "+a"(acc.regs[2*BC+2]), "+a"(acc.regs[2*BC+3]),
          "+a"(acc.regs[3*BC+0]), "+a"(acc.regs[3*BC+1]), "+a"(acc.regs[3*BC+2]), "+a"(acc.regs[3*BC+3])
        : "v"(a0), "v"(a1), "v"(a2), "v"(a3),
          "v"(b0), "v"(b1), "v"(b2), "v"(b3),
          "v"(sa0), "v"(sa1), "v"(sb0), "v"(sb1)
    );
    if constexpr (PF_N > 0) emit_one_pf(pf, pf_start);
    // Col 1: a[0..3] × b1
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %16, %21, %1,  %24, %26 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %17, %21, %5,  %24, %26 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %18, %21, %9,  %25, %26 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %19, %21, %13, %25, %26 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        : "+a"(acc.regs[0*BC+0]), "+a"(acc.regs[0*BC+1]), "+a"(acc.regs[0*BC+2]), "+a"(acc.regs[0*BC+3]),
          "+a"(acc.regs[1*BC+0]), "+a"(acc.regs[1*BC+1]), "+a"(acc.regs[1*BC+2]), "+a"(acc.regs[1*BC+3]),
          "+a"(acc.regs[2*BC+0]), "+a"(acc.regs[2*BC+1]), "+a"(acc.regs[2*BC+2]), "+a"(acc.regs[2*BC+3]),
          "+a"(acc.regs[3*BC+0]), "+a"(acc.regs[3*BC+1]), "+a"(acc.regs[3*BC+2]), "+a"(acc.regs[3*BC+3])
        : "v"(a0), "v"(a1), "v"(a2), "v"(a3),
          "v"(b0), "v"(b1), "v"(b2), "v"(b3),
          "v"(sa0), "v"(sa1), "v"(sb0), "v"(sb1)
    );
    if constexpr (PF_N > 1) emit_one_pf(pf, pf_start + 1);
    // Col 2: a[0..3] × b2
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %16, %22, %2,  %24, %27 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %17, %22, %6,  %24, %27 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %18, %22, %10, %25, %27 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %19, %22, %14, %25, %27 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        : "+a"(acc.regs[0*BC+0]), "+a"(acc.regs[0*BC+1]), "+a"(acc.regs[0*BC+2]), "+a"(acc.regs[0*BC+3]),
          "+a"(acc.regs[1*BC+0]), "+a"(acc.regs[1*BC+1]), "+a"(acc.regs[1*BC+2]), "+a"(acc.regs[1*BC+3]),
          "+a"(acc.regs[2*BC+0]), "+a"(acc.regs[2*BC+1]), "+a"(acc.regs[2*BC+2]), "+a"(acc.regs[2*BC+3]),
          "+a"(acc.regs[3*BC+0]), "+a"(acc.regs[3*BC+1]), "+a"(acc.regs[3*BC+2]), "+a"(acc.regs[3*BC+3])
        : "v"(a0), "v"(a1), "v"(a2), "v"(a3),
          "v"(b0), "v"(b1), "v"(b2), "v"(b3),
          "v"(sa0), "v"(sa1), "v"(sb0), "v"(sb1)
    );
    if constexpr (PF_N > 2) emit_one_pf(pf, pf_start + 2);
    // Col 3: a[0..3] × b3
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %16, %23, %3,  %24, %27 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %17, %23, %7,  %24, %27 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %18, %23, %11, %25, %27 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
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

// ── KPAIR: 32 MFMAs + 8 buffer_load_to_lds, split into 4 column blocks ──
// Each block: 8 MFMAs (4 rows × lo+hi) + 2 prefetch loads.
// Two tile_pf_params: pf1 loads spread across cols 0-1, pf2 across cols 2-3.

__device__ __forceinline__ void fp4_mma_kpair_with_tile_prefetch(
    fp4_acc_v2& acc,
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_scales[], const fp8e8m0_4 b_scales[],
    const tile_pf_params &pf1, const tile_pf_params &pf2)
{
    constexpr int BC = RBN / 16;
    fp4_intx4_t a0l=fp4_lo4(A[0]), a1l=fp4_lo4(A[1]), a2l=fp4_lo4(A[2]), a3l=fp4_lo4(A[3]);
    fp4_intx4_t a0h=fp4_hi4(A[0]), a1h=fp4_hi4(A[1]), a2h=fp4_hi4(A[2]), a3h=fp4_hi4(A[3]);
    unsigned sa0 = std::bit_cast<unsigned>(a_scales[0]);
    unsigned sa1 = std::bit_cast<unsigned>(a_scales[1]);
    unsigned sb0 = std::bit_cast<unsigned>(b_scales[0]);
    unsigned sb1 = std::bit_cast<unsigned>(b_scales[1]);

    fp4_intx4_t b0l=fp4_lo4(B[0]), b0h=fp4_hi4(B[0]);
    // Col 0: 4 rows × lo+hi
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %16, %24, %0,  %26, %28 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %20, %25, %0,  %26, %28 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %17, %24, %4,  %26, %28 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %21, %25, %4,  %26, %28 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %18, %24, %8,  %27, %28 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %22, %25, %8,  %27, %28 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %19, %24, %12, %27, %28 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %23, %25, %12, %27, %28 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc.regs[0*BC+0]), "+a"(acc.regs[0*BC+1]), "+a"(acc.regs[0*BC+2]), "+a"(acc.regs[0*BC+3]),
          "+a"(acc.regs[1*BC+0]), "+a"(acc.regs[1*BC+1]), "+a"(acc.regs[1*BC+2]), "+a"(acc.regs[1*BC+3]),
          "+a"(acc.regs[2*BC+0]), "+a"(acc.regs[2*BC+1]), "+a"(acc.regs[2*BC+2]), "+a"(acc.regs[2*BC+3]),
          "+a"(acc.regs[3*BC+0]), "+a"(acc.regs[3*BC+1]), "+a"(acc.regs[3*BC+2]), "+a"(acc.regs[3*BC+3])
        : "v"(a0l), "v"(a1l), "v"(a2l), "v"(a3l),
          "v"(a0h), "v"(a1h), "v"(a2h), "v"(a3h),
          "v"(b0l), "v"(b0h),
          "v"(sa0), "v"(sa1), "v"(sb0)
    );
    emit_one_pf(pf1, 0);
    emit_one_pf(pf1, 1);

    fp4_intx4_t b1l=fp4_lo4(B[1]), b1h=fp4_hi4(B[1]);
    // Col 1: 4 rows × lo+hi
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %16, %24, %1,  %26, %28 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %20, %25, %1,  %26, %28 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %17, %24, %5,  %26, %28 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %21, %25, %5,  %26, %28 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %18, %24, %9,  %27, %28 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %22, %25, %9,  %27, %28 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %19, %24, %13, %27, %28 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %23, %25, %13, %27, %28 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc.regs[0*BC+0]), "+a"(acc.regs[0*BC+1]), "+a"(acc.regs[0*BC+2]), "+a"(acc.regs[0*BC+3]),
          "+a"(acc.regs[1*BC+0]), "+a"(acc.regs[1*BC+1]), "+a"(acc.regs[1*BC+2]), "+a"(acc.regs[1*BC+3]),
          "+a"(acc.regs[2*BC+0]), "+a"(acc.regs[2*BC+1]), "+a"(acc.regs[2*BC+2]), "+a"(acc.regs[2*BC+3]),
          "+a"(acc.regs[3*BC+0]), "+a"(acc.regs[3*BC+1]), "+a"(acc.regs[3*BC+2]), "+a"(acc.regs[3*BC+3])
        : "v"(a0l), "v"(a1l), "v"(a2l), "v"(a3l),
          "v"(a0h), "v"(a1h), "v"(a2h), "v"(a3h),
          "v"(b1l), "v"(b1h),
          "v"(sa0), "v"(sa1), "v"(sb0)
    );
    emit_one_pf(pf1, 2);
    emit_one_pf(pf1, 3);

    fp4_intx4_t b2l=fp4_lo4(B[2]), b2h=fp4_hi4(B[2]);
    // Col 2: 4 rows × lo+hi
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %16, %24, %2,  %26, %28 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %20, %25, %2,  %26, %28 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %17, %24, %6,  %26, %28 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %21, %25, %6,  %26, %28 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %18, %24, %10, %27, %28 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %22, %25, %10, %27, %28 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %19, %24, %14, %27, %28 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %23, %25, %14, %27, %28 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc.regs[0*BC+0]), "+a"(acc.regs[0*BC+1]), "+a"(acc.regs[0*BC+2]), "+a"(acc.regs[0*BC+3]),
          "+a"(acc.regs[1*BC+0]), "+a"(acc.regs[1*BC+1]), "+a"(acc.regs[1*BC+2]), "+a"(acc.regs[1*BC+3]),
          "+a"(acc.regs[2*BC+0]), "+a"(acc.regs[2*BC+1]), "+a"(acc.regs[2*BC+2]), "+a"(acc.regs[2*BC+3]),
          "+a"(acc.regs[3*BC+0]), "+a"(acc.regs[3*BC+1]), "+a"(acc.regs[3*BC+2]), "+a"(acc.regs[3*BC+3])
        : "v"(a0l), "v"(a1l), "v"(a2l), "v"(a3l),
          "v"(a0h), "v"(a1h), "v"(a2h), "v"(a3h),
          "v"(b2l), "v"(b2h),
          "v"(sa0), "v"(sa1), "v"(sb1)
    );
    emit_one_pf(pf2, 0);
    emit_one_pf(pf2, 1);

    fp4_intx4_t b3l=fp4_lo4(B[3]), b3h=fp4_hi4(B[3]);
    // Col 3: 4 rows × lo+hi
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %16, %24, %3,  %26, %28 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %20, %25, %3,  %26, %28 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %17, %24, %7,  %26, %28 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %21, %25, %7,  %26, %28 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %18, %24, %11, %27, %28 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %22, %25, %11, %27, %28 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %19, %24, %15, %27, %28 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %23, %25, %15, %27, %28 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc.regs[0*BC+0]), "+a"(acc.regs[0*BC+1]), "+a"(acc.regs[0*BC+2]), "+a"(acc.regs[0*BC+3]),
          "+a"(acc.regs[1*BC+0]), "+a"(acc.regs[1*BC+1]), "+a"(acc.regs[1*BC+2]), "+a"(acc.regs[1*BC+3]),
          "+a"(acc.regs[2*BC+0]), "+a"(acc.regs[2*BC+1]), "+a"(acc.regs[2*BC+2]), "+a"(acc.regs[2*BC+3]),
          "+a"(acc.regs[3*BC+0]), "+a"(acc.regs[3*BC+1]), "+a"(acc.regs[3*BC+2]), "+a"(acc.regs[3*BC+3])
        : "v"(a0l), "v"(a1l), "v"(a2l), "v"(a3l),
          "v"(a0h), "v"(a1h), "v"(a2h), "v"(a3h),
          "v"(b3l), "v"(b3h),
          "v"(sa0), "v"(sa1), "v"(sb1)
    );
    emit_one_pf(pf2, 2);
    emit_one_pf(pf2, 3);
}

// ── KPAIR: 32 MFMAs + 8 tile loads + 8 ds_reads (pipelined next-iter tiles) ──
// Each column block: 8 MFMAs + 2 ds_reads interleaved + 2 tile loads after asm.
// nd0-nd7: outputs for next iteration's tile data (8 float4).

__device__ __forceinline__ void fp4_mma_kpair_tile_pf_and_nextlds(
    fp4_acc_v2& acc,
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_scales[], const fp8e8m0_4 b_scales[],
    const tile_pf_params &pf1, const tile_pf_params &pf2,
    float4 &nd0, float4 &nd1, float4 &nd2, float4 &nd3,
    float4 &nd4, float4 &nd5, float4 &nd6, float4 &nd7,
    uint32_t next_lds0, uint32_t next_lds1)
{
    constexpr int BC = RBN / 16;
    fp4_intx4_t a0l=fp4_lo4(A[0]), a1l=fp4_lo4(A[1]), a2l=fp4_lo4(A[2]), a3l=fp4_lo4(A[3]);
    fp4_intx4_t a0h=fp4_hi4(A[0]), a1h=fp4_hi4(A[1]), a2h=fp4_hi4(A[2]), a3h=fp4_hi4(A[3]);
    unsigned sa0 = std::bit_cast<unsigned>(a_scales[0]);
    unsigned sa1 = std::bit_cast<unsigned>(a_scales[1]);
    unsigned sb0 = std::bit_cast<unsigned>(b_scales[0]);
    unsigned sb1 = std::bit_cast<unsigned>(b_scales[1]);

    fp4_intx4_t b0l=fp4_lo4(B[0]), b0h=fp4_hi4(B[0]);
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %18, %26, %0,  %28, %30 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %22, %27, %0,  %28, %30 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %19, %26, %4,  %28, %30 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %23, %27, %4,  %28, %30 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %16, %31 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %20, %26, %8,  %29, %30 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %24, %27, %8,  %29, %30 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %17, %31 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %21, %26, %12, %29, %30 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %25, %27, %12, %29, %30 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc.regs[0*BC+0]), "+a"(acc.regs[0*BC+1]), "+a"(acc.regs[0*BC+2]), "+a"(acc.regs[0*BC+3]),
          "+a"(acc.regs[1*BC+0]), "+a"(acc.regs[1*BC+1]), "+a"(acc.regs[1*BC+2]), "+a"(acc.regs[1*BC+3]),
          "+a"(acc.regs[2*BC+0]), "+a"(acc.regs[2*BC+1]), "+a"(acc.regs[2*BC+2]), "+a"(acc.regs[2*BC+3]),
          "+a"(acc.regs[3*BC+0]), "+a"(acc.regs[3*BC+1]), "+a"(acc.regs[3*BC+2]), "+a"(acc.regs[3*BC+3]),
          "=&v"(nd0), "=&v"(nd1)
        : "v"(a0l), "v"(a1l), "v"(a2l), "v"(a3l),
          "v"(a0h), "v"(a1h), "v"(a2h), "v"(a3h),
          "v"(b0l), "v"(b0h),
          "v"(sa0), "v"(sa1), "v"(sb0),
          "v"(next_lds0)
    );
    emit_one_pf(pf1, 0);
    emit_one_pf(pf1, 1);

    fp4_intx4_t b1l=fp4_lo4(B[1]), b1h=fp4_hi4(B[1]);
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %18, %26, %1,  %28, %30 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %22, %27, %1,  %28, %30 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %19, %26, %5,  %28, %30 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %23, %27, %5,  %28, %30 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %16, %31 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %20, %26, %9,  %29, %30 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %24, %27, %9,  %29, %30 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %17, %31 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %21, %26, %13, %29, %30 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %25, %27, %13, %29, %30 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc.regs[0*BC+0]), "+a"(acc.regs[0*BC+1]), "+a"(acc.regs[0*BC+2]), "+a"(acc.regs[0*BC+3]),
          "+a"(acc.regs[1*BC+0]), "+a"(acc.regs[1*BC+1]), "+a"(acc.regs[1*BC+2]), "+a"(acc.regs[1*BC+3]),
          "+a"(acc.regs[2*BC+0]), "+a"(acc.regs[2*BC+1]), "+a"(acc.regs[2*BC+2]), "+a"(acc.regs[2*BC+3]),
          "+a"(acc.regs[3*BC+0]), "+a"(acc.regs[3*BC+1]), "+a"(acc.regs[3*BC+2]), "+a"(acc.regs[3*BC+3]),
          "=&v"(nd2), "=&v"(nd3)
        : "v"(a0l), "v"(a1l), "v"(a2l), "v"(a3l),
          "v"(a0h), "v"(a1h), "v"(a2h), "v"(a3h),
          "v"(b1l), "v"(b1h),
          "v"(sa0), "v"(sa1), "v"(sb0),
          "v"(next_lds0)
    );
    emit_one_pf(pf1, 2);
    emit_one_pf(pf1, 3);

    fp4_intx4_t b2l=fp4_lo4(B[2]), b2h=fp4_hi4(B[2]);
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %18, %26, %2,  %28, %30 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %22, %27, %2,  %28, %30 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %19, %26, %6,  %28, %30 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %23, %27, %6,  %28, %30 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %16, %31 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %20, %26, %10, %29, %30 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %24, %27, %10, %29, %30 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %17, %31 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %21, %26, %14, %29, %30 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %25, %27, %14, %29, %30 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc.regs[0*BC+0]), "+a"(acc.regs[0*BC+1]), "+a"(acc.regs[0*BC+2]), "+a"(acc.regs[0*BC+3]),
          "+a"(acc.regs[1*BC+0]), "+a"(acc.regs[1*BC+1]), "+a"(acc.regs[1*BC+2]), "+a"(acc.regs[1*BC+3]),
          "+a"(acc.regs[2*BC+0]), "+a"(acc.regs[2*BC+1]), "+a"(acc.regs[2*BC+2]), "+a"(acc.regs[2*BC+3]),
          "+a"(acc.regs[3*BC+0]), "+a"(acc.regs[3*BC+1]), "+a"(acc.regs[3*BC+2]), "+a"(acc.regs[3*BC+3]),
          "=&v"(nd4), "=&v"(nd5)
        : "v"(a0l), "v"(a1l), "v"(a2l), "v"(a3l),
          "v"(a0h), "v"(a1h), "v"(a2h), "v"(a3h),
          "v"(b2l), "v"(b2h),
          "v"(sa0), "v"(sa1), "v"(sb1),
          "v"(next_lds1)
    );
    emit_one_pf(pf2, 0);
    emit_one_pf(pf2, 1);

    fp4_intx4_t b3l=fp4_lo4(B[3]), b3h=fp4_hi4(B[3]);
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %18, %26, %3,  %28, %30 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %22, %27, %3,  %28, %30 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %19, %26, %7,  %28, %30 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %23, %27, %7,  %28, %30 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %16, %31 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %20, %26, %11, %29, %30 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %24, %27, %11, %29, %30 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %17, %31 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %21, %26, %15, %29, %30 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %25, %27, %15, %29, %30 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc.regs[0*BC+0]), "+a"(acc.regs[0*BC+1]), "+a"(acc.regs[0*BC+2]), "+a"(acc.regs[0*BC+3]),
          "+a"(acc.regs[1*BC+0]), "+a"(acc.regs[1*BC+1]), "+a"(acc.regs[1*BC+2]), "+a"(acc.regs[1*BC+3]),
          "+a"(acc.regs[2*BC+0]), "+a"(acc.regs[2*BC+1]), "+a"(acc.regs[2*BC+2]), "+a"(acc.regs[2*BC+3]),
          "+a"(acc.regs[3*BC+0]), "+a"(acc.regs[3*BC+1]), "+a"(acc.regs[3*BC+2]), "+a"(acc.regs[3*BC+3]),
          "=&v"(nd6), "=&v"(nd7)
        : "v"(a0l), "v"(a1l), "v"(a2l), "v"(a3l),
          "v"(a0h), "v"(a1h), "v"(a2h), "v"(a3h),
          "v"(b3l), "v"(b3h),
          "v"(sa0), "v"(sa1), "v"(sb1),
          "v"(next_lds1)
    );
    emit_one_pf(pf2, 2);
    emit_one_pf(pf2, 3);
}

// ── Main kernel ──

__global__ __launch_bounds__(_NUM_THREADS, 1)
void mxfp4_colfirst_kernel(const colfirst_globals g) {
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

    // ════════════════ Main loop (KPAIR) ════════════════
    // Each MFMA block processes both lo+hi K-phases per accumulator.
    // Steps 2-3: 32 MFMAs each + ds_reads. Step 5: 32+32 MFMAs + tile prefetch.

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

        // ── Step 2: KPAIR A0×Bl (lo+hi) + interleaved ds_read for Br ──
        float4 br_d[8];
        fp4_mma_kpair_with_lds(acc_A0Bl, tA0, tBl, a0_raw, bl_raw,
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

        // ── Step 3: KPAIR A0×Br (lo+hi) + interleaved ds_read for A1 ──
        float4 a1_d[8];
        fp4_mma_kpair_with_lds(acc_A0Br, tA0, tBr, a0_raw, br_raw,
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
        // 16 buffer_load_to_lds interleaved across 64 KPAIR MFMAs below.
        // ══════════════════════════════════════════════════════════════

        __builtin_amdgcn_s_barrier();

        const int pf_bt = (bt + 2 < k_byte_iters) ? (bt + 2) : (k_byte_iters - 1);

        tile_pf_params pf_a0_p = make_pf_params(A0_db[cur], g.a, {0, 0, br * 2,     pf_bt}, so_a, srd_a, base_a, lb_a0[cur]);
        tile_pf_params pf_a1_p = make_pf_params(A1_db[cur], g.a, {0, 0, br * 2 + 1, pf_bt}, so_a, srd_a, base_a, lb_a1[cur]);
        tile_pf_params pf_bl_p = make_pf_params(Bl_db[cur], g.b, {0, 0, bc * 2,     pf_bt}, so_b, srd_b, base_b, lb_bl[cur]);
        tile_pf_params pf_br_p = make_pf_params(Br_db[cur], g.b, {0, 0, bc * 2 + 1, pf_bt}, so_b, srd_b, base_b, lb_br[cur]);

        // ── Step 4-5: KPAIR 64 MFMAs with 16 tile prefetch loads (8 per call) ──
        fp4_mma_kpair_with_tile_prefetch(acc_A1Bl, tA1, tBl, a1_raw, bl_raw, pf_a0_p, pf_a1_p);
        fp4_mma_kpair_with_tile_prefetch(acc_A1Br, tA1, tBr, a1_raw, br_raw, pf_bl_p, pf_br_p);
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

void dispatch_colfirst(colfirst_globals g) {
    g.m = static_cast<int>(g.c.rows());
    g.n = static_cast<int>(g.c.cols());
    g.k = static_cast<int>(g.a.cols()) * 2;
    const dim3 grid((g.m / BLK) * (g.n / BLK));
    mxfp4_colfirst_kernel<<<grid, dim3(_NUM_THREADS), 0, g.stream>>>(g);
}

PYBIND11_MODULE(tk_mxfp4_colfirst, m) {
    m.doc() = "MXFP4 column-first kernel: column-first MFMA order within hybrid pipeline";
    py::bind_function<dispatch_colfirst>(m, "gemm_rcr",
        &colfirst_globals::a, &colfirst_globals::b,
        &colfirst_globals::a_scale, &colfirst_globals::b_scale,
        &colfirst_globals::c);
}
