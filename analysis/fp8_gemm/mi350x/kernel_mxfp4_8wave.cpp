// MXFP4 8-wave kernel: occupancy-2 (512 threads), single-buffered LDS, inline ASM MFMAs
//
// Warp grid: WARPS_M=2 × WARPS_N=4 (8 warps). Per-warp tile RBM×RBN = 64×32.
// K loop (V1): G::load → wait → barrier → scales + ds_read (fp4_load_st_to_rt) →
//   MFMAs with optional interleaved ds_read prefetch → barrier.

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
constexpr int WARPS_M = 2, WARPS_N = 4;
constexpr int _NUM_WARPS   = WARPS_M * WARPS_N;
constexpr int _NUM_THREADS = _NUM_WARPS * WARP_THREADS;
constexpr int HB = BLK / 2;

constexpr int RBM = HB / WARPS_M;
constexpr int RBN = HB / WARPS_N;

static_assert(BLK == 256 && BK == 128, "8-wave requires BLK=256, BK=128");
static_assert(RBM == 64 && RBN == 32, "Expected 64×32 per-warp tile");

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

struct wave8_globals {
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

__device__ __forceinline__ fp4_intx4_t fp4_lo4(const fp4_intx8_t& x) {
    return __builtin_shufflevector(x, x, 0, 1, 2, 3);
}
__device__ __forceinline__ fp4_intx4_t fp4_hi4(const fp4_intx8_t& x) {
    return __builtin_shufflevector(x, x, 4, 5, 6, 7);
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

struct alignas(16) fp4_acc {
    fp4_floatx4_t regs[(RBM / 16) * (RBN / 16)];
};

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

// 8 MFMAs + 4 ds_read (RBN=32 Br tile), interleaved — prefetch Br while computing A0×Bl.
template<bool UPPER>
__device__ __forceinline__ void fp4_mma_with_lds_br(
    fp4_acc& acc,
    const fp4_intx8_t A[4], const fp4_intx8_t B[2],
    const fp8e8m0_4 a_scales[2], fp8e8m0_4 b_scale,
    int k_phase,
    float4 &d0, float4 &d1, float4 &d2, float4 &d3,
    uint32_t lds_br0)
{
    fp4_intx4_t a0, a1, a2, a3, b0, b1;
    if constexpr (!UPPER) {
        a0=fp4_lo4(A[0]); a1=fp4_lo4(A[1]); a2=fp4_lo4(A[2]); a3=fp4_lo4(A[3]);
        b0=fp4_lo4(B[0]); b1=fp4_lo4(B[1]);
    } else {
        a0=fp4_hi4(A[0]); a1=fp4_hi4(A[1]); a2=fp4_hi4(A[2]); a3=fp4_hi4(A[3]);
        b0=fp4_hi4(B[0]); b1=fp4_hi4(B[1]);
    }
    unsigned sa0 = std::bit_cast<unsigned>(remap_phase(a_scales[0], k_phase));
    unsigned sa1 = std::bit_cast<unsigned>(remap_phase(a_scales[1], k_phase));
    unsigned sb0 = std::bit_cast<unsigned>(remap_phase(b_scale, k_phase));

    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %12, %16, %0,  %18, %20 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %12, %17, %1,  %18, %20 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %8,  %21 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %13, %16, %2,  %18, %20 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %13, %17, %3,  %18, %20 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %9,  %21 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %14, %16, %4,  %19, %20 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %14, %17, %5,  %19, %20 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %10, %21 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %15, %16, %6,  %19, %20 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %15, %17, %7,  %19, %20 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %11, %21 offset:6144\n"
        : "+a"(acc.regs[0]), "+a"(acc.regs[1]), "+a"(acc.regs[2]), "+a"(acc.regs[3]),
          "+a"(acc.regs[4]), "+a"(acc.regs[5]), "+a"(acc.regs[6]), "+a"(acc.regs[7]),
          "=v"(d0), "=v"(d1), "=v"(d2), "=v"(d3)
        : "v"(a0), "v"(a1), "v"(a2), "v"(a3),
          "v"(b0), "v"(b1),
          "v"(sa0), "v"(sa1), "v"(sb0),
          "v"(lds_br0)
    );
}

// 8 MFMAs + 8 ds_read — prefetch A1 (RBM=64) while computing A0×Br.
template<bool UPPER>
__device__ __forceinline__ void fp4_mma_with_lds_a1(
    fp4_acc& acc,
    const fp4_intx8_t A[4], const fp4_intx8_t B[2],
    const fp8e8m0_4 a_scales[2], fp8e8m0_4 b_scale,
    int k_phase,
    float4 &d0, float4 &d1, float4 &d2, float4 &d3,
    float4 &d4, float4 &d5, float4 &d6, float4 &d7,
    uint32_t lds_a0, uint32_t lds_a1)
{
    fp4_intx4_t a0, a1, a2, a3, b0, b1;
    if constexpr (!UPPER) {
        a0=fp4_lo4(A[0]); a1=fp4_lo4(A[1]); a2=fp4_lo4(A[2]); a3=fp4_lo4(A[3]);
        b0=fp4_lo4(B[0]); b1=fp4_lo4(B[1]);
    } else {
        a0=fp4_hi4(A[0]); a1=fp4_hi4(A[1]); a2=fp4_hi4(A[2]); a3=fp4_hi4(A[3]);
        b0=fp4_hi4(B[0]); b1=fp4_hi4(B[1]);
    }
    unsigned sa0 = std::bit_cast<unsigned>(remap_phase(a_scales[0], k_phase));
    unsigned sa1 = std::bit_cast<unsigned>(remap_phase(a_scales[1], k_phase));
    unsigned sb0 = std::bit_cast<unsigned>(remap_phase(b_scale, k_phase));

    // Operands: %0-%7 acc, %8-%15 ds outs; %16-%19 a0-a3, %20-%21 b0-b1, %22-%23 sa0-sa1, %24 sb0, %25-%26 lds
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %16, %20, %0,  %22, %24 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %8,  %25 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %16, %21, %1,  %22, %24 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %9,  %25 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %17, %20, %2,  %22, %24 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %10, %25 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %17, %21, %3,  %22, %24 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %11, %25 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %18, %20, %4,  %23, %24 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %12, %26 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %18, %21, %5,  %23, %24 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %13, %26 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %19, %20, %6,  %23, %24 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %14, %26 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %19, %21, %7,  %23, %24 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %15, %26 offset:6144\n"
        : "+a"(acc.regs[0]), "+a"(acc.regs[1]), "+a"(acc.regs[2]), "+a"(acc.regs[3]),
          "+a"(acc.regs[4]), "+a"(acc.regs[5]), "+a"(acc.regs[6]), "+a"(acc.regs[7]),
          "=v"(d0), "=v"(d1), "=v"(d2), "=v"(d3),
          "=v"(d4), "=v"(d5), "=v"(d6), "=v"(d7)
        : "v"(a0), "v"(a1), "v"(a2), "v"(a3),
          "v"(b0), "v"(b1),
          "v"(sa0), "v"(sa1), "v"(sb0),
          "v"(lds_a0), "v"(lds_a1)
    );
}

template<bool UPPER>
__device__ __forceinline__ void fp4_mma_pure(
    fp4_acc& acc,
    const fp4_intx8_t A[4], const fp4_intx8_t B[2],
    const fp8e8m0_4 a_scales[2], fp8e8m0_4 b_scale,
    int k_phase)
{
    fp4_intx4_t a0, a1, a2, a3, b0, b1;
    if constexpr (!UPPER) {
        a0=fp4_lo4(A[0]); a1=fp4_lo4(A[1]); a2=fp4_lo4(A[2]); a3=fp4_lo4(A[3]);
        b0=fp4_lo4(B[0]); b1=fp4_lo4(B[1]);
    } else {
        a0=fp4_hi4(A[0]); a1=fp4_hi4(A[1]); a2=fp4_hi4(A[2]); a3=fp4_hi4(A[3]);
        b0=fp4_hi4(B[0]); b1=fp4_hi4(B[1]);
    }
    unsigned sa0 = std::bit_cast<unsigned>(remap_phase(a_scales[0], k_phase));
    unsigned sa1 = std::bit_cast<unsigned>(remap_phase(a_scales[1], k_phase));
    unsigned sb0 = std::bit_cast<unsigned>(remap_phase(b_scale, k_phase));

    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %8, %12, %0,  %14, %16 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %8, %13, %1,  %14, %16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %9, %12, %2,  %14, %16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %9, %13, %3,  %14, %16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %10, %12, %4,  %15, %16 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %10, %13, %5,  %15, %16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %11, %12, %6,  %15, %16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %11, %13, %7,  %15, %16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        : "+a"(acc.regs[0]), "+a"(acc.regs[1]), "+a"(acc.regs[2]), "+a"(acc.regs[3]),
          "+a"(acc.regs[4]), "+a"(acc.regs[5]), "+a"(acc.regs[6]), "+a"(acc.regs[7])
        : "v"(a0), "v"(a1), "v"(a2), "v"(a3),
          "v"(b0), "v"(b1),
          "v"(sa0), "v"(sa1), "v"(sb0)
    );
}

__global__ __launch_bounds__(_NUM_THREADS, 2)
void mxfp4_8wave_kernel(const wave8_globals g) {
    static_assert(K_BYTES % BK == 0 && N_DIM % BLK == 0 && M_DIM % BLK == 0);

    constexpr int bpc = N_DIM / BLK;

    __shared__ ST_tile A0_tile;
    __shared__ ST_tile A1_tile;
    __shared__ ST_tile Bl_tile;
    __shared__ ST_tile Br_tile;

    const int bid = blockIdx.x;
    const int br = bid / bpc, bc = bid % bpc;
    const int wm = warpid() / WARPS_N, wn = warpid() % WARPS_N;

    constexpr int bpt = ST_tile::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * _NUM_THREADS;
    constexpr int mpt = HB * BK * sizeof(fp8e4m3) / bpm;
    uint32_t so_a[mpt], so_b[mpt];
    G::prefill_swizzled_offsets(A0_tile, g.a, so_a);
    G::prefill_swizzled_offsets(Bl_tile, g.b, so_b);

    constexpr int a_packs = RBM / 32;
    constexpr int b_packs = (RBN + 31) / 32;
    static_assert(a_packs == 2 && b_packs == 1);

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

    fp4_acc acc_A0Bl{}, acc_A0Br{}, acc_A1Bl{}, acc_A1Br{};

    for (int bt = 0; bt < k_byte_iters; ++bt) {
        G::load(A0_tile, g.a, {0, 0, br * 2,     bt}, so_a);
        G::load(A1_tile, g.a, {0, 0, br * 2 + 1, bt}, so_a);
        G::load(Bl_tile, g.b, {0, 0, bc * 2,     bt}, so_b);
        G::load(Br_tile, g.b, {0, 0, bc * 2 + 1, bt}, so_b);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();

        fp8e8m0_4 a0_raw[a_packs], a1_raw[a_packs];
        fp8e8m0_4 bl_raw[b_packs], br_raw[b_packs];
        const uint32_t soff = static_cast<uint32_t>(bt) << 8;
        #pragma unroll
        for (int p = 0; p < a_packs; ++p) {
            a0_raw[p] = load_pq_scale_srd(a0_srd[p], lane_soff, soff);
            a1_raw[p] = load_pq_scale_srd(a1_srd[p], lane_soff, soff);
        }
        #pragma unroll
        for (int p = 0; p < b_packs; ++p) {
            bl_raw[p] = load_pq_scale_srd(bl_srd[p], lane_soff, soff);
            br_raw[p] = load_pq_scale_srd(br_srd[p], lane_soff, soff);
        }

        // Load all 4 subtiles from LDS to registers (simple, correct approach)
        A_row_reg a0_rt, a1_rt;
        B_row_reg bl_rt, br_rt;
        fp4_load_st_to_rt(a0_rt, kittens::subtile_inplace<RBM, BK>(A0_tile, {wm, 0}));
        fp4_load_st_to_rt(a1_rt, kittens::subtile_inplace<RBM, BK>(A1_tile, {wm, 0}));
        fp4_load_st_to_rt(bl_rt, kittens::subtile_inplace<RBN, BK>(Bl_tile, {wn, 0}));
        fp4_load_st_to_rt(br_rt, kittens::subtile_inplace<RBN, BK>(Br_tile, {wn, 0}));
        asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)");

        fp4_intx8_t tA0[4], tA1[4], tBl[2], tBr[2];
        #pragma unroll
        for (int i = 0; i < 4; i++) { tA0[i] = fp4_extract_tile(a0_rt, i); tA1[i] = fp4_extract_tile(a1_rt, i); }
        tBl[0] = fp4_extract_tile(bl_rt, 0); tBl[1] = fp4_extract_tile(bl_rt, 1);
        tBr[0] = fp4_extract_tile(br_rt, 0); tBr[1] = fp4_extract_tile(br_rt, 1);

        // Phase 0 (lo)
        fp4_mma_pure<false>(acc_A0Bl, tA0, tBl, a0_raw, bl_raw[0], 0);
        fp4_mma_pure<false>(acc_A0Br, tA0, tBr, a0_raw, br_raw[0], 0);
        fp4_mma_pure<false>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw[0], 0);
        fp4_mma_pure<false>(acc_A1Br, tA1, tBr, a1_raw, br_raw[0], 0);

        // Phase 1 (hi)
        fp4_mma_pure<true>(acc_A0Bl, tA0, tBl, a0_raw, bl_raw[0], 1);
        fp4_mma_pure<true>(acc_A0Br, tA0, tBr, a0_raw, br_raw[0], 1);
        fp4_mma_pure<true>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw[0], 1);
        fp4_mma_pure<true>(acc_A1Br, tA1, tBr, a1_raw, br_raw[0], 1);

        __builtin_amdgcn_s_barrier();
    }

    auto store_acc = [&](const fp4_acc& acc, int m_half, int n_half) {
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

void dispatch_8wave(wave8_globals g) {
    g.m = static_cast<int>(g.c.rows());
    g.n = static_cast<int>(g.c.cols());
    g.k = static_cast<int>(g.a.cols()) * 2;
    const dim3 grid((g.m / BLK) * (g.n / BLK));
    mxfp4_8wave_kernel<<<grid, dim3(_NUM_THREADS), 0, g.stream>>>(g);
}

PYBIND11_MODULE(tk_mxfp4_8wave, m) {
    m.doc() = "MXFP4 8-wave GEMM: single-buffer LDS, inline ASM MFMAs";
    py::bind_function<dispatch_8wave>(m, "gemm_rcr",
        &wave8_globals::a, &wave8_globals::b,
        &wave8_globals::a_scale, &wave8_globals::b_scale,
        &wave8_globals::c);
}
