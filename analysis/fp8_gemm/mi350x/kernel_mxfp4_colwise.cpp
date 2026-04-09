// MXFP4 Column-First KPAIR Kernel
//
// Architecture changes vs kernel_mxfp4_hybrid.cpp:
//   - Column-first B-tile loading: 2 ds_reads per B column (vs 8 bulk)
//   - KPAIR: lo+hi K-phases in one 8-MFMA block (no remap_phase → 0 VALU)
//   - op_sel[0:1] selects K-phase, op_sel_hi[0:1] selects A-row/B-col sub-group
//   - A0+A1 loaded upfront (16 ds_reads), B columns loaded incrementally
//   - Eliminates lgkmcnt stalls between hybrid's Steps 2-3-4

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

static_assert(BLK == 256 && BK == 128, "Colwise requires BLK=256, BK=128");
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

struct colwise_globals {
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

// ── Scale helpers (no remap_phase for KPAIR) ──

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

// ── LDS base addr helper ──

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

// ── Accumulator ──

struct alignas(16) fp4_acc_v2 {
    fp4_floatx4_t regs[(RBM / 16) * (RBN / 16)]; // 4×4 = 16 tiles
};

// ── Prefetch params (same as hybrid) ──

struct tile_pf_params {
    int32x4_t srd;
    uint32_t soff;
    uint32_t lds_addrs[4];
    uint32_t voffs[4];
};

static constexpr int PF_MPT = (HB * BK * sizeof(fp8e4m3)) / (16 * _NUM_THREADS);

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

// ════════════════════════════════════════════════════════════════════
// Column-First KPAIR MFMA blocks
//
// Each block: 8 MFMAs for 4 A-rows × 1 B-col, both lo+hi K-phases
// op_sel[0:1]: K-phase (0=lo, 1=hi)
// op_sel_hi[0]: A row sub-group (0=even, 1=odd within scale pack)
// op_sel_hi[1]: B col sub-group (0=even, 1=odd within scale pack)
//
// Constraint layout (%0-%15: acc, %16-%23: A lo/hi, %24-%25: B lo/hi,
//                    %26-%27: A scales, %28: B scale)
// ════════════════════════════════════════════════════════════════════

// Macro: 8 MFMAs for column BCOL with B-col sub-group BSH
// R0,R1,R2,R3 = operand indices for acc rows 0-3 at this column
//
// Preshuffle layout per 4-byte dword: [sub0-kp0, sub1-kp0, sub0-kp1, sub1-kp1]
//   op_sel[0/1]    selects A/B sub-group (byte 0 vs 1 within 16-bit half)
//   op_sel_hi[0/1] selects K-phase (lower 16 bits=kp0 vs upper 16 bits=kp1)
#define COL_KPAIR_8MFMA(R0, R1, R2, R3, BSH) \
    "v_mfma_scale_f32_16x16x128_f8f6f4 %" #R0 ", %16, %24, %" #R0 ", %26, %28 op_sel:[0," BSH ",0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 %" #R0 ", %17, %25, %" #R0 ", %26, %28 op_sel:[0," BSH ",0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 %" #R1 ", %18, %24, %" #R1 ", %26, %28 op_sel:[1," BSH ",0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 %" #R1 ", %19, %25, %" #R1 ", %26, %28 op_sel:[1," BSH ",0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 %" #R2 ", %20, %24, %" #R2 ", %27, %28 op_sel:[0," BSH ",0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 %" #R2 ", %21, %25, %" #R2 ", %27, %28 op_sel:[0," BSH ",0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 %" #R3 ", %22, %24, %" #R3 ", %27, %28 op_sel:[1," BSH ",0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 %" #R3 ", %23, %25, %" #R3 ", %27, %28 op_sel:[1," BSH ",0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"

// Constraint list for all 16 acc regs + 8 A operands + 2 B operands + 3 scales
#define COL_KPAIR_CONSTRAINTS(acc, a0l, a0h, a1l, a1h, a2l, a2h, a3l, a3h, bl, bh, sa0, sa1, sb) \
    : "+a"((acc).regs[0]),  "+a"((acc).regs[1]),  "+a"((acc).regs[2]),  "+a"((acc).regs[3]),  \
      "+a"((acc).regs[4]),  "+a"((acc).regs[5]),  "+a"((acc).regs[6]),  "+a"((acc).regs[7]),  \
      "+a"((acc).regs[8]),  "+a"((acc).regs[9]),  "+a"((acc).regs[10]), "+a"((acc).regs[11]), \
      "+a"((acc).regs[12]), "+a"((acc).regs[13]), "+a"((acc).regs[14]), "+a"((acc).regs[15]) \
    : "v"(a0l), "v"(a0h), "v"(a1l), "v"(a1h), \
      "v"(a2l), "v"(a2h), "v"(a3l), "v"(a3h), \
      "v"(bl), "v"(bh), \
      "v"(sa0), "v"(sa1), "v"(sb)

// 8 MFMAs for B col 0 (sub=0): acc rows at %0,%4,%8,%12
__device__ __forceinline__ void col_kpair_bcol0(
    fp4_acc_v2& acc,
    fp4_intx4_t a0l, fp4_intx4_t a0h, fp4_intx4_t a1l, fp4_intx4_t a1h,
    fp4_intx4_t a2l, fp4_intx4_t a2h, fp4_intx4_t a3l, fp4_intx4_t a3h,
    fp4_intx4_t bl, fp4_intx4_t bh,
    unsigned sa0, unsigned sa1, unsigned sb)
{
    asm volatile(
        COL_KPAIR_8MFMA(0, 4, 8, 12, "0")
        COL_KPAIR_CONSTRAINTS(acc, a0l,a0h,a1l,a1h,a2l,a2h,a3l,a3h,bl,bh,sa0,sa1,sb)
    );
}

// 8 MFMAs for B col 1 (sub=1): acc rows at %1,%5,%9,%13
__device__ __forceinline__ void col_kpair_bcol1(
    fp4_acc_v2& acc,
    fp4_intx4_t a0l, fp4_intx4_t a0h, fp4_intx4_t a1l, fp4_intx4_t a1h,
    fp4_intx4_t a2l, fp4_intx4_t a2h, fp4_intx4_t a3l, fp4_intx4_t a3h,
    fp4_intx4_t bl, fp4_intx4_t bh,
    unsigned sa0, unsigned sa1, unsigned sb)
{
    asm volatile(
        COL_KPAIR_8MFMA(1, 5, 9, 13, "1")
        COL_KPAIR_CONSTRAINTS(acc, a0l,a0h,a1l,a1h,a2l,a2h,a3l,a3h,bl,bh,sa0,sa1,sb)
    );
}

// 8 MFMAs for B col 2 (sub=0): acc rows at %2,%6,%10,%14
__device__ __forceinline__ void col_kpair_bcol2(
    fp4_acc_v2& acc,
    fp4_intx4_t a0l, fp4_intx4_t a0h, fp4_intx4_t a1l, fp4_intx4_t a1h,
    fp4_intx4_t a2l, fp4_intx4_t a2h, fp4_intx4_t a3l, fp4_intx4_t a3h,
    fp4_intx4_t bl, fp4_intx4_t bh,
    unsigned sa0, unsigned sa1, unsigned sb)
{
    asm volatile(
        COL_KPAIR_8MFMA(2, 6, 10, 14, "0")
        COL_KPAIR_CONSTRAINTS(acc, a0l,a0h,a1l,a1h,a2l,a2h,a3l,a3h,bl,bh,sa0,sa1,sb)
    );
}

// 8 MFMAs for B col 3 (sub=1): acc rows at %3,%7,%11,%15
__device__ __forceinline__ void col_kpair_bcol3(
    fp4_acc_v2& acc,
    fp4_intx4_t a0l, fp4_intx4_t a0h, fp4_intx4_t a1l, fp4_intx4_t a1h,
    fp4_intx4_t a2l, fp4_intx4_t a2h, fp4_intx4_t a3l, fp4_intx4_t a3h,
    fp4_intx4_t bl, fp4_intx4_t bh,
    unsigned sa0, unsigned sa1, unsigned sb)
{
    asm volatile(
        COL_KPAIR_8MFMA(3, 7, 11, 15, "1")
        COL_KPAIR_CONSTRAINTS(acc, a0l,a0h,a1l,a1h,a2l,a2h,a3l,a3h,bl,bh,sa0,sa1,sb)
    );
}

// Dispatch to the right bcol function
__device__ __forceinline__ void col_kpair_dispatch(
    fp4_acc_v2& acc, int bcol,
    fp4_intx4_t a0l, fp4_intx4_t a0h, fp4_intx4_t a1l, fp4_intx4_t a1h,
    fp4_intx4_t a2l, fp4_intx4_t a2h, fp4_intx4_t a3l, fp4_intx4_t a3h,
    fp4_intx4_t bl, fp4_intx4_t bh,
    unsigned sa0, unsigned sa1, unsigned sb)
{
    switch (bcol) {
        case 0: col_kpair_bcol0(acc, a0l,a0h,a1l,a1h,a2l,a2h,a3l,a3h, bl,bh, sa0,sa1,sb); break;
        case 1: col_kpair_bcol1(acc, a0l,a0h,a1l,a1h,a2l,a2h,a3l,a3h, bl,bh, sa0,sa1,sb); break;
        case 2: col_kpair_bcol2(acc, a0l,a0h,a1l,a1h,a2l,a2h,a3l,a3h, bl,bh, sa0,sa1,sb); break;
        case 3: col_kpair_bcol3(acc, a0l,a0h,a1l,a1h,a2l,a2h,a3l,a3h, bl,bh, sa0,sa1,sb); break;
    }
}

// ── Main kernel ──

__global__ __launch_bounds__(_NUM_THREADS, 1)
void mxfp4_colwise_kernel(const colwise_globals g) {
    static_assert(K_BYTES % BK == 0 && N_DIM % BLK == 0 && M_DIM % BLK == 0);

    constexpr int bpc = N_DIM / BLK;

    __shared__ ST_tile A0_db[2];
    __shared__ ST_tile A1_db[2];
    __shared__ ST_tile Bl_db[2];
    __shared__ ST_tile Br_db[2];

    const int bid = blockIdx.x;
    const int br = bid / bpc, bc = bid % bpc;
    const int wm = warpid() / WARPS_N, wn = warpid() % WARPS_N;

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

    // ── Tile SRDs for buffer_load_to_lds ──
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

    // ════════════════ Prologue ════════════════
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
    // All 4 tiles loaded upfront from LDS (same as hybrid dead zone).
    // Column-first KPAIR MFMAs: iterate B columns, no remap_phase.
    #pragma unroll 2
    for (int bt = 0; bt < k_byte_iters; ++bt) {
        const int cur = bt & 1;

        if constexpr (k_byte_iters >= 2) {
            asm volatile("s_waitcnt vmcnt(16)");
        } else {
            asm volatile("s_waitcnt vmcnt(8)");
        }
        __builtin_amdgcn_s_barrier();

        // Capture prefetched scales
        fp8e8m0_4 a0_raw[a_packs], a1_raw[a_packs];
        fp8e8m0_4 bl_raw[b_packs], br_raw[b_packs];
        #pragma unroll
        for (int p = 0; p < a_packs; ++p) { a0_raw[p] = pf_a0[p]; a1_raw[p] = pf_a1[p]; }
        #pragma unroll
        for (int p = 0; p < b_packs; ++p) { bl_raw[p] = pf_bl[p]; br_raw[p] = pf_br[p]; }

        // ── Step 1: Load all 4 tiles from LDS (32 ds_reads) ──
        A_row_reg a0_rt, a1_rt;
        B_row_reg bl_rt, br_rt;
        fp4_load_st_to_rt(a0_rt, kittens::subtile_inplace<RBM, BK>(A0_db[cur], {wm, 0}));
        fp4_load_st_to_rt(a1_rt, kittens::subtile_inplace<RBM, BK>(A1_db[cur], {wm, 0}));
        fp4_load_st_to_rt(bl_rt, kittens::subtile_inplace<RBN, BK>(Bl_db[cur], {wn, 0}));
        fp4_load_st_to_rt(br_rt, kittens::subtile_inplace<RBN, BK>(Br_db[cur], {wn, 0}));

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

        // Wait for all 32 ds_reads, keep scales in flight
        asm volatile("s_waitcnt lgkmcnt(0) vmcnt(8)");

        // Extract all tiles
        fp4_intx8_t tA0[4], tA1[4], tBl[4], tBr[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            tA0[i] = fp4_extract_tile(a0_rt, i);
            tA1[i] = fp4_extract_tile(a1_rt, i);
            tBl[i] = fp4_extract_tile(bl_rt, i);
            tBr[i] = fp4_extract_tile(br_rt, i);
        }

        // Split A into lo/hi halves for KPAIR
        fp4_intx4_t a0_lo[4], a0_hi[4], a1_lo[4], a1_hi[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            a0_lo[i] = fp4_lo4(tA0[i]);
            a0_hi[i] = fp4_hi4(tA0[i]);
            a1_lo[i] = fp4_lo4(tA1[i]);
            a1_hi[i] = fp4_hi4(tA1[i]);
        }

        // Raw scales for KPAIR (no remap_phase)
        unsigned sa0_A0 = std::bit_cast<unsigned>(a0_raw[0]);
        unsigned sa1_A0 = std::bit_cast<unsigned>(a0_raw[1]);
        unsigned sa0_A1 = std::bit_cast<unsigned>(a1_raw[0]);
        unsigned sa1_A1 = std::bit_cast<unsigned>(a1_raw[1]);

        // ══════════════════════════════════════════════════════════════
        // All tile data in VGPRs. LDS free for tile prefetch.
        // Barrier → prefetch setup → 128 MFMAs (column-first KPAIR)
        // ══════════════════════════════════════════════════════════════

        __builtin_amdgcn_s_barrier();

        const int pf_bt = (bt + 2 < k_byte_iters) ? (bt + 2) : (k_byte_iters - 1);
        tile_pf_params pf_a0_p = make_pf_params(A0_db[cur], g.a, {0, 0, br * 2,     pf_bt}, so_a, srd_a, base_a, lb_a0[cur]);
        tile_pf_params pf_a1_p = make_pf_params(A1_db[cur], g.a, {0, 0, br * 2 + 1, pf_bt}, so_a, srd_a, base_a, lb_a1[cur]);
        tile_pf_params pf_bl_p = make_pf_params(Bl_db[cur], g.b, {0, 0, bc * 2,     pf_bt}, so_b, srd_b, base_b, lb_bl[cur]);
        tile_pf_params pf_br_p = make_pf_params(Br_db[cur], g.b, {0, 0, bc * 2 + 1, pf_bt}, so_b, srd_b, base_b, lb_br[cur]);

        // ── 128 MFMAs: 8 B cols × (A0 block + A1 block) ──
        // Tile prefetch distributed: 1 per 8-MFMA block (16 total)
        int pf_tile = 0, pf_idx = 0;
        const tile_pf_params* pf_arr[4] = { &pf_a0_p, &pf_a1_p, &pf_bl_p, &pf_br_p };

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            fp4_acc_v2& acc_a0 = (j < 4) ? acc_A0Bl : acc_A0Br;
            fp4_acc_v2& acc_a1 = (j < 4) ? acc_A1Bl : acc_A1Br;
            const int lc = j & 3;

            const fp4_intx8_t& bcol = (j < 4) ? tBl[lc] : tBr[lc];
            fp4_intx4_t b_lo = fp4_lo4(bcol);
            fp4_intx4_t b_hi = fp4_hi4(bcol);

            unsigned sb_u = std::bit_cast<unsigned>(
                (j < 4) ? ((lc < 2) ? bl_raw[0] : bl_raw[1])
                         : ((lc < 2) ? br_raw[0] : br_raw[1]));

            // 8 MFMAs: A0 × B[j] with KPAIR
            col_kpair_dispatch(acc_a0, lc,
                a0_lo[0],a0_hi[0], a0_lo[1],a0_hi[1],
                a0_lo[2],a0_hi[2], a0_lo[3],a0_hi[3],
                b_lo, b_hi, sa0_A0, sa1_A0, sb_u);

            if (pf_tile < 4) { emit_one_pf(*pf_arr[pf_tile], pf_idx); if (++pf_idx >= PF_MPT) { pf_idx = 0; pf_tile++; } }

            // 8 MFMAs: A1 × B[j] with KPAIR
            col_kpair_dispatch(acc_a1, lc,
                a1_lo[0],a1_hi[0], a1_lo[1],a1_hi[1],
                a1_lo[2],a1_hi[2], a1_lo[3],a1_hi[3],
                b_lo, b_hi, sa0_A1, sa1_A1, sb_u);

            if (pf_tile < 4) { emit_one_pf(*pf_arr[pf_tile], pf_idx); if (++pf_idx >= PF_MPT) { pf_idx = 0; pf_tile++; } }
        }
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

void dispatch_colwise(colwise_globals g) {
    g.m = static_cast<int>(g.c.rows());
    g.n = static_cast<int>(g.c.cols());
    g.k = static_cast<int>(g.a.cols()) * 2;
    const dim3 grid((g.m / BLK) * (g.n / BLK));
    mxfp4_colwise_kernel<<<grid, dim3(_NUM_THREADS), 0, g.stream>>>(g);
}

PYBIND11_MODULE(tk_mxfp4_colwise, m) {
    m.doc() = "MXFP4 column-first KPAIR kernel";
    py::bind_function<dispatch_colwise>(m, "gemm_rcr",
        &colwise_globals::a, &colwise_globals::b,
        &colwise_globals::a_scale, &colwise_globals::b_scale,
        &colwise_globals::c);
}
