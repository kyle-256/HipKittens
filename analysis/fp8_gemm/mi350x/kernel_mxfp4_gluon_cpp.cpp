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
#include <cstdlib>
#include <cstdio>
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

#ifndef STEP3_EMBED_BARRIER
#define STEP3_EMBED_BARRIER 1
#endif

#ifndef STEP3_BARRIER_VMCNT
#define STEP3_BARRIER_VMCNT 8
#endif
#ifndef SPREAD_LDS
#define SPREAD_LDS 0
#endif
#ifndef NONVOLATILE_SCALE_X2_POC
#define NONVOLATILE_SCALE_X2_POC 1
#endif
#ifndef TAIL_BARRIER_VMCNT
#define TAIL_BARRIER_VMCNT STEP3_BARRIER_VMCNT
#endif

#ifndef FUSED_STEP34
#define FUSED_STEP34 0
#endif
// R37 Fix B (2026-04-19): the non-fused step3+step4 path emits 4 separate
// `asm volatile` blocks; the compiler is free to interleave clobbering moves
// between them which corrupts the upper-left 128x128 quadrant of every 256x256
// output tile (acc_A0Bl). Fix: on the default (non-FUSED_STEP34) path, fuse
// step3+step4 into a single asm block via kpair_64mfma_step34 — preserving the
// R25-C tail-pf-off and K_EXACT branching that the original FUSED_STEP34=1 path
// bypassed. Defaults ON; set R37_FIX_B=0 to revert to the legacy buggy code
// path (for comparison only).
#ifndef R37_FIX_B
#define R37_FIX_B 1
#endif
// R25-C: K-loop tail epilogue specialization. When set to N>0, the last N
// iterations of the steady-state main loop use PF_N=0 (no global prefetch) for
// the Step3/Step4 KPAIR calls. Rationale: clamped pf_bt re-fetches the same
// trailing K-tile, wasting saturated VMEM slots on stale lines. Default 0 →
// baseline (no behavior change). Only takes effect on TAIL_SPLIT=1, non-FUSED
// path (the path used by all DLA shapes).
//
// Empirical (R25C smoke 2026-04-18): runtime tail branch FOLDS for shapes that
// fully unroll (k_byte_iters ≤ 32, e.g. K=4096 → 16 iters, DLA2/DLA7) → WIN.
// For shapes where the K-loop is `pragma unroll 8`-only (k_byte_iters > 32,
// e.g. K=128256 → 501 iters, DLA1) the branch becomes a runtime check inside
// the hot loop and code-size doubles → catastrophic regression.
// Therefore we gate R25C on K_DIM ≤ 32768 (≤ 128 K-tiles), where pragma unroll
// fully unrolls and the branch folds.
#ifndef R25C_TAIL_PF_OFF_ITERS
#define R25C_TAIL_PF_OFF_ITERS 0
#endif

#ifndef R25C_K_LIMIT
#define R25C_K_LIMIT 32768
#endif
// R25-G: optional EXACT-K gate. When R25C_K_EXACT > 0, R25C only fires for the
// exact specified K_DIM. Used to ship per-K-iter-count tuned pfoff values
// (e.g. K=14336→pfoff54, K=32768→pfoff124) without their flag set bleeding
// into other shapes (where the wrong pfoff would zero out all prefetches).
// Default 0 → behavior unchanged (only the K_LIMIT gate applies).
#ifndef R25C_K_EXACT
#define R25C_K_EXACT 0
#endif
#define R25C_ACTIVE ((R25C_TAIL_PF_OFF_ITERS > 0) && (K_DIM <= R25C_K_LIMIT) \
                     && (R25C_K_EXACT == 0 || K_DIM == R25C_K_EXACT))

// GlobalB: load B tiles directly from global memory (pre-shuffled) via
// buffer_load_dwordx4 → VGPRs, bypassing LDS for B entirely.
// Requires B tensor to be pre-shuffled with aiter's shuffle_weight(layout=(16,16)).
// When enabled: no Bl_db/Br_db shared memory, no B barrier waits, no B ds_reads.
#ifndef GLOBAL_B
#define GLOBAL_B 0
#endif

#define MXFP4_STR_IMPL(x) #x
#define MXFP4_STR(x) MXFP4_STR_IMPL(x)


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

using ST_tile = st_fp8e4m3<HB, BK, st_16x128_s>;
using A_row_reg = rt_fp8e4m3<RBM, BK, row_l, rt_16x128_s>;
using B_row_reg = rt_fp8e4m3<RBN, BK, row_l, rt_16x128_s>;
using RT_C = rt_fl<RBM, RBN, col_l, rt_16x16_s>;

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
using u32x2_t       = unsigned int __attribute__((ext_vector_type(2)));
static_assert(sizeof(u32x2_t) == 8);

__device__ __forceinline__ unsigned int pack_bf16x2(float x, float y) {
    unsigned int out;
    asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2"
        : "=v"(out) : "v"(x), "v"(y));
    return out;
}

// Store bf16 value
static __device__ __forceinline__ void store_bf16_val(bf16* addr, float val) {
    bf16 v = base_types::convertor<bf16, float>::convert(val);
    *addr = v;
}

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

// ── GlobalB: direct B tile load from global memory ──
// Pre-shuffled layout (aiter shuffle_weight layout=(16,16)):
//   flat[n_block, k_block, k_phase, n_row, byte] =
//     n_block*16*K_bytes + k_block*512 + k_phase*256 + n_row*16 + byte
//
// MFMA lane mapping: row=laneid%16, group=laneid/16, k_block_local=group/2
// Each lane loads 16 bytes per buffer_load_dwordx4.
// 8 loads per half-tile: 4 tile-rows x 2 k-phases (lo4 + hi4 of fp4_intx8_t).
// Output d[0..3] = k_phase=0 (lo4), d[4..7] = k_phase=1 (hi4).
// Feeds directly into extract_tile (same d[8] → fp4_intx8_t[4] mapping).
#if GLOBAL_B

__device__ __forceinline__ void load_b_global_8(
    float4 d[8],
    const i32x4 &b_srd,
    const uint32_t voff[8],
    uint32_t k_soffset
) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        asm volatile(
            "buffer_load_dwordx4 %0, %1, %2, %3 offen\n"
            : "=&v"(d[i])
            : "v"(voff[i]), "s"(b_srd), "s"(k_soffset)
        );
    }
}
// Pre-compute 8 per-lane voffsets for one B half-tile.
// n_block_start = first n-block index for this half-tile (each n-block = 16 rows)
// K_bytes_full = K_DIM/2 (total K dimension in bytes)
//
// Preshuffle format: aiter shuffle_weight(layout=(16,16)):
//   flat = n_block * 16 * K_bytes + col_group * 256 + n_row * 16 + byte
//   where col_group ∈ {0..K_bytes/16-1} indexes 16-byte columns.
//
// TK's LDS path applies a swizzle on ds_read: addr ^ (((addr % 2048) >> 8) << 4)
// This XORs the 16-byte column group index with (n_row // 2).
// To match, we apply the same XOR when computing buffer_load voffsets:
//   swizzled_col_group = original_col_group ^ (row >> 1)
//
// MFMA lane mapping:
//   row = laneid % 16, group = laneid / 16 (0..3)
//   lo (voff[r]):   original_col_group = group
//   hi (voff[r+4]): original_col_group = group + 4
//
// voff[0..3] = tile-rows 0..3 at lo half (column groups 0..3 → swizzled)
// voff[4..7] = tile-rows 0..3 at hi half (column groups 4..7 → swizzled)
__device__ __forceinline__ void compute_b_global_load_voffs(
    uint32_t voff[8],
    int n_block_start,
    int K_bytes_full)
{
    const int laneid = kittens::laneid();
    const int row = laneid % 16;
    const int group = laneid / 16;
    const int row_half = row >> 1;  // 0..7: LDS swizzle XOR key

    // Apply TK LDS swizzle: XOR column group index with (row // 2)
    const uint32_t lo_cg = (uint32_t)(group ^ row_half);        // swizzled col group for lo half
    const uint32_t hi_cg = (uint32_t)((group + 4) ^ row_half);  // swizzled col group for hi half

    // col_group * 256 + row * 16 gives the offset within one n_block's 2048-byte region
    const uint32_t lo_kb = lo_cg * 256u + (uint32_t)row * 16u;
    const uint32_t hi_kb = hi_cg * 256u + (uint32_t)row * 16u;

    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const uint32_t n_base = (uint32_t)(n_block_start + r) * 16u * (uint32_t)K_bytes_full;
        voff[r]     = n_base + lo_kb;
        voff[r + 4] = n_base + hi_kb;
    }
}

#endif // GLOBAL_B

// ── Scale helpers ──

__device__ __forceinline__ const uint8_t* global_loadd_scale_row_base_ptr(
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

// Load two consecutive scale dwords via buffer_load_dwordx2 (merged global_load format).
// Non-volatile asm allows compiler scheduling flexibility while preserving dwordx2.
__device__ __forceinline__ void load_pq_scale_x2_async(
    i32x4 srsrc, uint32_t voffset, uint32_t soffset,
    fp8e8m0_4 &out_lo, fp8e8m0_4 &out_hi) {
    uint64_t pair;
#if NONVOLATILE_SCALE_X2_POC
    asm(
        "buffer_load_dwordx2 %0, %1, %2, %3 offen"
        : "=v"(pair)
        : "v"(voffset), "s"(srsrc), "s"(soffset)
    );
#else
    asm volatile(
        "buffer_load_dwordx2 %0, %1, %2, %3 offen"
        : "=v"(pair)
        : "v"(voffset), "s"(srsrc), "s"(soffset)
    );
#endif
    out_lo = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(pair));
    out_hi = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(pair >> 32));
}
static constexpr int PF_MPT = (HB * BK * sizeof(fp8e4m3)) / (16 * _NUM_THREADS);

// Tile prefetch: issues PF_MPT buffer_load_dwordx4 ... lds per warp.
__device__ __forceinline__ void emit_tile_pf(
    auto &dst, const auto &src, const auto &idx,
    const uint32_t *so, i32x4 srd, const void *base, uint32_t lb,
    int cache_hint = static_cast<int>(coherency::cache_all))
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
            cache_hint);
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
    int      cache_hint;  // R22B: per-tile cache hint (cache_all default)
};

template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ __forceinline__ tile_pf_params make_pf_params(
    ST &dst, const GL &src, const COORD &idx,
    const uint32_t *so, i32x4 srd_in, const void *base_ptr, uint32_t lds_base,
    int cache_hint = static_cast<int>(coherency::cache_all))
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
    p.cache_hint = cache_hint;
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
        p.cache_hint);
}
#ifndef STEP3_PF_N
#define STEP3_PF_N 8
#endif
#ifndef STEP4_PF_N
#define STEP4_PF_N 8
#endif

static_assert(STEP3_PF_N >= 0 && STEP3_PF_N <= 2 * PF_MPT);
static_assert(STEP4_PF_N >= 0 && STEP4_PF_N <= 2 * PF_MPT);

template<int PF_N>
__device__ __forceinline__ void emit_pf_tail(const tile_pf_params& pf0, const tile_pf_params& pf1) {
    static_assert(PF_N >= 0 && PF_N <= 2 * PF_MPT);
    if constexpr (PF_N < PF_MPT) {
        #pragma unroll
        for (int pi = PF_N; pi < PF_MPT; ++pi) emit_one_pf(pf0, pi);
        #pragma unroll
        for (int pi = 0; pi < PF_MPT; ++pi) emit_one_pf(pf1, pi);
        // templated function body so the fence cannot be hoisted above the
        // preceding loads by the compiler scheduler. Default OFF
        // (R44 baseline byte-compatible).
    } else if constexpr (PF_N < 2 * PF_MPT) {
        #pragma unroll
        for (int pi = PF_N - PF_MPT; pi < PF_MPT; ++pi) emit_one_pf(pf1, pi);
    }
    // PF_N == 2*PF_MPT path: zero loads emitted; do NOT emit a fence (no-op
    // function, fence would just stall for nothing).
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

#if !defined(SPREAD_DS_READ) || !SPREAD_DS_READ
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
#endif // !SPREAD_DS_READ

// ── SPREAD variant: 8 ds_reads distributed 1:4 across 32 MFMAs ──
// Same interface as kpair_32mfma_with_lds but with spread ds_read placement.
// Includes s_waitcnt lgkmcnt(0) at the END of the asm block to prevent
// the compiler from reading ds_read outputs before they complete.
#if defined(SPREAD_DS_READ) && SPREAD_DS_READ
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
        // Row 0 Phase 0 (4 MFMAs) + ds_read 0
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %24, %32, %0,  %40, %42 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %24, %33, %1,  %40, %42 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %24, %34, %2,  %40, %43 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %16, %44 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %24, %35, %3,  %40, %43 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Row 0 Phase 1 (4 MFMAs) + ds_read 1
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %28, %36, %0,  %40, %42 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %28, %37, %1,  %40, %42 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %28, %38, %2,  %40, %43 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %17, %44 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %28, %39, %3,  %40, %43 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Row 1 Phase 0 (4 MFMAs) + ds_read 2
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %25, %32, %4,  %40, %42 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %25, %33, %5,  %40, %42 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %25, %34, %6,  %40, %43 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %18, %44 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %25, %35, %7,  %40, %43 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Row 1 Phase 1 (4 MFMAs) + ds_read 3
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %29, %36, %4,  %40, %42 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %29, %37, %5,  %40, %42 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %29, %38, %6,  %40, %43 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %19, %44 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %29, %39, %7,  %40, %43 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Row 2 Phase 0 (4 MFMAs) + ds_read 4
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %26, %32, %8,  %41, %42 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %26, %33, %9,  %41, %42 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %26, %34, %10, %41, %43 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %20, %45 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %26, %35, %11, %41, %43 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Row 2 Phase 1 (4 MFMAs) + ds_read 5
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %30, %36, %8,  %41, %42 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %30, %37, %9,  %41, %42 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %30, %38, %10, %41, %43 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %21, %45 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %30, %39, %11, %41, %43 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Row 3 Phase 0 (4 MFMAs) + ds_read 6
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %27, %32, %12, %41, %42 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %27, %33, %13, %41, %42 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %27, %34, %14, %41, %43 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %22, %45 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %27, %35, %15, %41, %43 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Row 3 Phase 1 (4 MFMAs) + ds_read 7 + waitcnt
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %31, %36, %12, %41, %42 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %23, %45 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %31, %37, %13, %41, %42 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %31, %38, %14, %41, %43 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %31, %39, %15, %41, %43 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "s_waitcnt lgkmcnt(0)\n"
        : KPAIR_ACC_CLOBBER,
          "=&v"(d0), "=&v"(d1), "=&v"(d2), "=&v"(d3),
          "=&v"(d4), "=&v"(d5), "=&v"(d6), "=&v"(d7)
        : KPAIR_INPUTS,
          "v"(lds_a0), "v"(lds_a1)
    );
}
#endif // SPREAD_DS_READ

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

// ── R68 axis-step12pf NON-SPLIT: kpair_64mfma_step12_pf_interleaved ──
// Iso-test prototype. Same as kpair_64mfma_step12 (64 MFMAs + 16 ds_reads in
// single asm block) PLUS 16 buffer_load_dwordx4 ... lds prefetches (a0+a1+bl+br,
// 4 each) interleaved between MFMA groups. Mirror of R66's STEP34_PF_INTERLEAVE
// pattern, but applied to step12. step34 caller emits NO prefetches when this
// path is active.
//
// Operand additions vs base step12 (74 → 114):
//   %74..77   = "s" srd_a0/a1/bl/br        (sgpr-quad each)
//   %78..81   = "s" soff_a0/a1/bl/br
//   %82..97   = "s" lds_addrs[a0,a1,bl,br][0..3]   (16 sgpr)
//   %98..113  = "v" voffs[a0,a1,bl,br][0..3]       (16 vgpr)
// Clobbers m0 (each prefetch writes m0 = lds_addr).
//
// CRITICAL DATAFLOW: prefetches write to A0_db[cur], A1_db[cur], Bl_db[cur],
// Br_db[cur] (per call site convention pf_a0bl=cur). step12 ds_reads pull from
// Br_db[cur] and A1_db[cur] — same physical LDS slots. Buffer_load_to_lds
// writes (vmem) and ds_reads (lds) of overlapping LDS regions create a race.
// This helper exists to MEASURE the resulting correctness damage at iso scale.
// It is NOT expected to be safe.
#ifndef STEP12_PF_INTERLEAVE
#define STEP12_PF_INTERLEAVE 0
#endif

// ── STEP12_SPLIT_PF: split step12 into 2x kpair_32mfma_with_lds + 4 prefetches ──
// Splits the monolithic 64-MFMA step12 asm block into:
//   Step1: kpair_32mfma_with_lds(A0*Bl + ds_read Br)
//   waitcnt lgkmcnt(0) + extract_tile(Br)
//   4x emit_one_pf(A0_db + Bl_db) — SAFE: A0/Bl already in regs, no LDS race
//   Step2: kpair_32mfma_with_lds(A0*Br + ds_read A1)
// +1.7pp on K=128256, -1.5pp on K=32768. Autotune picks best per-shape.
// Requires STEP34_PF_INTERLEAVE=1 (redundant A0/Bl loads in step34 are harmless).
#ifndef STEP12_SPLIT_PF
#define STEP12_SPLIT_PF 0
#endif

#if STEP12_PF_INTERLEAVE
__device__ __forceinline__ void kpair_64mfma_step12_pf_interleaved(
    fp4_floatx4_t acc_bl[16], fp4_floatx4_t acc_br[16],
    const fp4_intx8_t A0[4], const fp4_intx8_t Bl[4],
    const fp8e8m0_4 a_raw[2], const fp8e8m0_4 bl_raw[2], const fp8e8m0_4 br_raw[2],
    float4 br_d[8], float4 a1_d[8],
    uint32_t br_p0, uint32_t br_p1,
    uint32_t a1_p0, uint32_t a1_p1,
    const tile_pf_params &pf_a0, const tile_pf_params &pf_a1,
    const tile_pf_params &pf_bl, const tile_pf_params &pf_br)
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
        // ═══ STEP 1: A0×Bl (32 MFMAs) + 8 ds_reads (Br) + 8 prefetches (a0+a1) ═══
        // Group 0
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %48, %56, %0,  %64, %66 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %32, %70 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %48, %57, %1,  %64, %66 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %82\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %48, %58, %2,  %64, %67 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %98, %74, %78 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %48, %59, %3,  %64, %67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Group 1
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %52, %60, %0,  %64, %66 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %33, %70 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %52, %61, %1,  %64, %66 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %83\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %52, %62, %2,  %64, %67 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %99, %74, %78 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %52, %63, %3,  %64, %67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Group 2
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %49, %56, %4,  %64, %66 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %34, %70 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %49, %57, %5,  %64, %66 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %84\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %49, %58, %6,  %64, %67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %100, %74, %78 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %49, %59, %7,  %64, %67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Group 3
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %53, %60, %4,  %64, %66 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %35, %70 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %53, %61, %5,  %64, %66 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %85\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %53, %62, %6,  %64, %67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %101, %74, %78 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %53, %63, %7,  %64, %67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Group 4 — switch to a1 prefetch
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %50, %56, %8,  %65, %66 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %36, %71 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %50, %57, %9,  %65, %66 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %86\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %50, %58, %10, %65, %67 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %102, %75, %79 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %50, %59, %11, %65, %67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Group 5
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %54, %60, %8,  %65, %66 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %37, %71 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %54, %61, %9,  %65, %66 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %87\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %54, %62, %10, %65, %67 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %103, %75, %79 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %54, %63, %11, %65, %67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Group 6
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %51, %56, %12, %65, %66 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %38, %71 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %51, %57, %13, %65, %66 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %88\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %51, %58, %14, %65, %67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %104, %75, %79 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %51, %59, %15, %65, %67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Group 7
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %55, %60, %12, %65, %66 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %39, %71 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %55, %61, %13, %65, %66 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %89\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %55, %62, %14, %65, %67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %105, %75, %79 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %55, %63, %15, %65, %67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "s_waitcnt lgkmcnt(0)\n"
        // ═══ STEP 2: A0×Br (32 MFMAs) + 8 ds_reads (A1) + 8 prefetches (bl+br) ═══
        // Group 8 — switch to bl prefetch
        "v_mfma_scale_f32_16x16x128_f8f6f4 %16, %48, %32, %16, %64, %68 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %40, %72 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %17, %48, %33, %17, %64, %68 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %90\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %18, %48, %34, %18, %64, %69 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %106, %76, %80 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %19, %48, %35, %19, %64, %69 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Group 9
        "v_mfma_scale_f32_16x16x128_f8f6f4 %16, %52, %36, %16, %64, %68 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %41, %72 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %17, %52, %37, %17, %64, %68 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %91\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %18, %52, %38, %18, %64, %69 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %107, %76, %80 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %19, %52, %39, %19, %64, %69 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Group 10
        "v_mfma_scale_f32_16x16x128_f8f6f4 %20, %49, %32, %20, %64, %68 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %42, %72 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %21, %49, %33, %21, %64, %68 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %92\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %22, %49, %34, %22, %64, %69 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %108, %76, %80 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %23, %49, %35, %23, %64, %69 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Group 11
        "v_mfma_scale_f32_16x16x128_f8f6f4 %20, %53, %36, %20, %64, %68 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %43, %72 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %21, %53, %37, %21, %64, %68 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %93\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %22, %53, %38, %22, %64, %69 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %109, %76, %80 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %23, %53, %39, %23, %64, %69 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Group 12 — switch to br prefetch
        "v_mfma_scale_f32_16x16x128_f8f6f4 %24, %50, %32, %24, %65, %68 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %44, %73 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %25, %50, %33, %25, %65, %68 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %94\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %26, %50, %34, %26, %65, %69 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %110, %77, %81 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %27, %50, %35, %27, %65, %69 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Group 13
        "v_mfma_scale_f32_16x16x128_f8f6f4 %24, %54, %36, %24, %65, %68 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %45, %73 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %25, %54, %37, %25, %65, %68 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %95\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %26, %54, %38, %26, %65, %69 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %111, %77, %81 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %27, %54, %39, %27, %65, %69 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Group 14
        "v_mfma_scale_f32_16x16x128_f8f6f4 %28, %51, %32, %28, %65, %68 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %46, %73 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %29, %51, %33, %29, %65, %68 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %96\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %30, %51, %34, %30, %65, %69 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %112, %77, %81 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %31, %51, %35, %31, %65, %69 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Group 15
        "v_mfma_scale_f32_16x16x128_f8f6f4 %28, %55, %36, %28, %65, %68 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %47, %73 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %29, %55, %37, %29, %65, %68 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %97\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %30, %55, %38, %30, %65, %69 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %113, %77, %81 offen lds\n"
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
          "v"(br_p0), "v"(br_p1), "v"(a1_p0), "v"(a1_p1),
          // %74..77: srds
          "s"(pf_a0.srd), "s"(pf_a1.srd), "s"(pf_bl.srd), "s"(pf_br.srd),
          // %78..81: soffs
          "s"(pf_a0.soff), "s"(pf_a1.soff), "s"(pf_bl.soff), "s"(pf_br.soff),
          // %82..85: lds_addrs a0
          "s"(pf_a0.lds_addrs[0]), "s"(pf_a0.lds_addrs[1]), "s"(pf_a0.lds_addrs[2]), "s"(pf_a0.lds_addrs[3]),
          // %86..89: lds_addrs a1
          "s"(pf_a1.lds_addrs[0]), "s"(pf_a1.lds_addrs[1]), "s"(pf_a1.lds_addrs[2]), "s"(pf_a1.lds_addrs[3]),
          // %90..93: lds_addrs bl
          "s"(pf_bl.lds_addrs[0]), "s"(pf_bl.lds_addrs[1]), "s"(pf_bl.lds_addrs[2]), "s"(pf_bl.lds_addrs[3]),
          // %94..97: lds_addrs br
          "s"(pf_br.lds_addrs[0]), "s"(pf_br.lds_addrs[1]), "s"(pf_br.lds_addrs[2]), "s"(pf_br.lds_addrs[3]),
          // %98..101: voffs a0
          "v"(pf_a0.voffs[0]), "v"(pf_a0.voffs[1]), "v"(pf_a0.voffs[2]), "v"(pf_a0.voffs[3]),
          // %102..105: voffs a1
          "v"(pf_a1.voffs[0]), "v"(pf_a1.voffs[1]), "v"(pf_a1.voffs[2]), "v"(pf_a1.voffs[3]),
          // %106..109: voffs bl
          "v"(pf_bl.voffs[0]), "v"(pf_bl.voffs[1]), "v"(pf_bl.voffs[2]), "v"(pf_bl.voffs[3]),
          // %110..113: voffs br
          "v"(pf_br.voffs[0]), "v"(pf_br.voffs[1]), "v"(pf_br.voffs[2]), "v"(pf_br.voffs[3])
        : "memory", "m0"
    );
}
#endif // STEP12_PF_INTERLEAVE

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

// ── Merged Steps 3+4: 64 MFMAs + 16 ds_reads (nxt_a0 + nxt_bl) in one asm block ──
// Eliminates compiler transition between Steps 3 and 4.
// Barrier emitted separately before the asm block (caller or inline).
// Outputs: %0..15=acc_bl, %16..31=acc_br, %32..39=nxt_a0_d, %40..47=nxt_bl_d
// Inputs: %48..51=A1_lo, %52..55=A1_hi, %56..59=Bl_lo, %60..63=Bl_hi,
//   %64..67=Br_lo, %68..71=Br_hi,
//   %72..73=sa0/sa1, %74..75=sbl0/sbl1, %76..77=sbr0/sbr1,
//   %78..79=a0_lds_p0/p1, %80..81=bl_lds_p0/p1

#ifndef STEP34_PF_INTERLEAVE
#define STEP34_PF_INTERLEAVE 0
#endif

#if STEP34_PF_INTERLEAVE
// ── R66 axis-A Opt-1: kpair_64mfma_step34_pf_interleaved ──
// Same as kpair_64mfma_step34 (4:1 MFMA:ds_read in single asm block) PLUS
// 16 buffer_load_dwordx4 ... lds prefetches interleaved between MFMA groups.
// Layout per 4-MFMA group: [mfma, ds_read|nothing, mfma, s_mov_b32 m0,
//   mfma, buffer_load_dwordx4 ... lds, mfma]. ds_read present in 8 of 16 groups
// (groups consuming nxt_a0/nxt_bl). Buffer_load present in ALL 16 groups.
// All in a SINGLE asm volatile to avoid the AGPR-allocator hazard from R62.
//
// Operand additions vs base helper (82 → 122):
//   %82..85  = "s" srd_a0/a1/bl/br        (sgpr-quad each)
//   %86..89  = "s" soff_a0/a1/bl/br
//   %90..105 = "s" lds_addrs[a0,a1,bl,br][0..3]   (16 sgpr)
//   %106..121 = "v" voffs[a0,a1,bl,br][0..3]      (16 vgpr)
// Clobbers m0 (each prefetch writes m0 = lds_addr).
__device__ __forceinline__ void kpair_64mfma_step34_pf_interleaved(
    fp4_floatx4_t acc_bl[16], fp4_floatx4_t acc_br[16],
    const fp4_intx8_t A1[4],
    const fp4_intx8_t Bl[4], const fp4_intx8_t Br[4],
    const fp8e8m0_4 a1_raw[2], const fp8e8m0_4 bl_raw[2], const fp8e8m0_4 br_raw[2],
    float4 nxt_a0_d[8], float4 nxt_bl_d[8],
    uint32_t a0_p0, uint32_t a0_p1,
    uint32_t bl_p0, uint32_t bl_p1,
    const tile_pf_params &pf_a0, const tile_pf_params &pf_a1,
    const tile_pf_params &pf_bl, const tile_pf_params &pf_br)
{
    // A1 tile splits (shared between Step3 and Step4)
    fp4_intx4_t a1_0l=fp4_lo4(A1[0]), a1_1l=fp4_lo4(A1[1]), a1_2l=fp4_lo4(A1[2]), a1_3l=fp4_lo4(A1[3]);
    fp4_intx4_t a1_0h=fp4_hi4(A1[0]), a1_1h=fp4_hi4(A1[1]), a1_2h=fp4_hi4(A1[2]), a1_3h=fp4_hi4(A1[3]);
    fp4_intx4_t bl_0l=fp4_lo4(Bl[0]), bl_1l=fp4_lo4(Bl[1]), bl_2l=fp4_lo4(Bl[2]), bl_3l=fp4_lo4(Bl[3]);
    fp4_intx4_t bl_0h=fp4_hi4(Bl[0]), bl_1h=fp4_hi4(Bl[1]), bl_2h=fp4_hi4(Bl[2]), bl_3h=fp4_hi4(Bl[3]);
    fp4_intx4_t br_0l=fp4_lo4(Br[0]), br_1l=fp4_lo4(Br[1]), br_2l=fp4_lo4(Br[2]), br_3l=fp4_lo4(Br[3]);
    fp4_intx4_t br_0h=fp4_hi4(Br[0]), br_1h=fp4_hi4(Br[1]), br_2h=fp4_hi4(Br[2]), br_3h=fp4_hi4(Br[3]);
    unsigned sa0  = std::bit_cast<unsigned>(a1_raw[0]);
    unsigned sa1  = std::bit_cast<unsigned>(a1_raw[1]);
    unsigned sbl0 = std::bit_cast<unsigned>(bl_raw[0]);
    unsigned sbl1 = std::bit_cast<unsigned>(bl_raw[1]);
    unsigned sbr0 = std::bit_cast<unsigned>(br_raw[0]);
    unsigned sbr1 = std::bit_cast<unsigned>(br_raw[1]);

    asm volatile("s_waitcnt vmcnt(" MXFP4_STR(STEP3_BARRIER_VMCNT) ")\ns_barrier\n" ::: "memory");
    asm volatile(
        // ═══ STEP 3: A1×Bl (32 MFMAs) + 8 ds_reads + 8 prefetches ═══
        // Group 0 (acc_bl[0..3], sa0, lo): mfma, ds_read, mfma, m0+pf, mfma, mfma
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %48, %56, %0,  %72, %74 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %32, %78 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %48, %57, %1,  %72, %74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %90\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %48, %58, %2,  %72, %75 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %106, %82, %86 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %48, %59, %3,  %72, %75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Group 1 (acc_bl[0..3], sa0, hi): mfma, ds_read, mfma, m0+pf, mfma, mfma
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %52, %60, %0,  %72, %74 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %33, %78 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %52, %61, %1,  %72, %74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %91\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %52, %62, %2,  %72, %75 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %107, %82, %86 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %52, %63, %3,  %72, %75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Group 2 (acc_bl[4..7], sa0, lo)
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %49, %56, %4,  %72, %74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %34, %78 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %49, %57, %5,  %72, %74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %92\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %49, %58, %6,  %72, %75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %108, %82, %86 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %49, %59, %7,  %72, %75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Group 3 (acc_bl[4..7], sa0, hi)
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %53, %60, %4,  %72, %74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %35, %78 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %53, %61, %5,  %72, %74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %93\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %53, %62, %6,  %72, %75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %109, %82, %86 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %53, %63, %7,  %72, %75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Group 4 (acc_bl[8..11], sa1, lo) — switch to a1_lds (%79)
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %50, %56, %8,  %73, %74 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %36, %79 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %50, %57, %9,  %73, %74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %94\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %50, %58, %10, %73, %75 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %110, %83, %87 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %50, %59, %11, %73, %75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Group 5
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %54, %60, %8,  %73, %74 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %37, %79 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %54, %61, %9,  %73, %74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %95\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %54, %62, %10, %73, %75 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %111, %83, %87 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %54, %63, %11, %73, %75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Group 6
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %51, %56, %12, %73, %74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %38, %79 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %51, %57, %13, %73, %74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %96\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %51, %58, %14, %73, %75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %112, %83, %87 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %51, %59, %15, %73, %75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Group 7
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %55, %60, %12, %73, %74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %39, %79 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %55, %61, %13, %73, %74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %97\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %55, %62, %14, %73, %75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %113, %83, %87 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %55, %63, %15, %73, %75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // ═══ STEP 4: A1×Br (32 MFMAs) + 8 ds_reads (nxt_bl) + 8 prefetches ═══
        // Group 8 (acc_br[0..3], sa0, lo) — bl_lds (%80)
        "v_mfma_scale_f32_16x16x128_f8f6f4 %16, %48, %64, %16, %72, %76 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %40, %80 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %17, %48, %65, %17, %72, %76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %98\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %18, %48, %66, %18, %72, %77 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %114, %84, %88 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %19, %48, %67, %19, %72, %77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Group 9
        "v_mfma_scale_f32_16x16x128_f8f6f4 %16, %52, %68, %16, %72, %76 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %41, %80 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %17, %52, %69, %17, %72, %76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %99\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %18, %52, %70, %18, %72, %77 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %115, %84, %88 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %19, %52, %71, %19, %72, %77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Group 10
        "v_mfma_scale_f32_16x16x128_f8f6f4 %20, %49, %64, %20, %72, %76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %42, %80 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %21, %49, %65, %21, %72, %76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %100\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %22, %49, %66, %22, %72, %77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %116, %84, %88 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %23, %49, %67, %23, %72, %77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Group 11
        "v_mfma_scale_f32_16x16x128_f8f6f4 %20, %53, %68, %20, %72, %76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %43, %80 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %21, %53, %69, %21, %72, %76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %101\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %22, %53, %70, %22, %72, %77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %117, %84, %88 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %23, %53, %71, %23, %72, %77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Group 12 (acc_br[8..11], sa1, lo) — switch to bl_p1 (%81)
        "v_mfma_scale_f32_16x16x128_f8f6f4 %24, %50, %64, %24, %73, %76 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %44, %81 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %25, %50, %65, %25, %73, %76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %102\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %26, %50, %66, %26, %73, %77 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %118, %85, %89 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %27, %50, %67, %27, %73, %77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Group 13
        "v_mfma_scale_f32_16x16x128_f8f6f4 %24, %54, %68, %24, %73, %76 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %45, %81 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %25, %54, %69, %25, %73, %76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %103\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %26, %54, %70, %26, %73, %77 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %119, %85, %89 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %27, %54, %71, %27, %73, %77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Group 14
        "v_mfma_scale_f32_16x16x128_f8f6f4 %28, %51, %64, %28, %73, %76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %46, %81 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %29, %51, %65, %29, %73, %76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %104\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %30, %51, %66, %30, %73, %77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %120, %85, %89 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %31, %51, %67, %31, %73, %77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Group 15
        "v_mfma_scale_f32_16x16x128_f8f6f4 %28, %55, %68, %28, %73, %76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %47, %81 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %29, %55, %69, %29, %73, %76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "s_mov_b32 m0, %105\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %30, %55, %70, %30, %73, %77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "buffer_load_dwordx4 %121, %85, %89 offen lds\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %31, %55, %71, %31, %73, %77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc_bl[0]),  "+a"(acc_bl[1]),  "+a"(acc_bl[2]),  "+a"(acc_bl[3]),
          "+a"(acc_bl[4]),  "+a"(acc_bl[5]),  "+a"(acc_bl[6]),  "+a"(acc_bl[7]),
          "+a"(acc_bl[8]),  "+a"(acc_bl[9]),  "+a"(acc_bl[10]), "+a"(acc_bl[11]),
          "+a"(acc_bl[12]), "+a"(acc_bl[13]), "+a"(acc_bl[14]), "+a"(acc_bl[15]),
          "+a"(acc_br[0]),  "+a"(acc_br[1]),  "+a"(acc_br[2]),  "+a"(acc_br[3]),
          "+a"(acc_br[4]),  "+a"(acc_br[5]),  "+a"(acc_br[6]),  "+a"(acc_br[7]),
          "+a"(acc_br[8]),  "+a"(acc_br[9]),  "+a"(acc_br[10]), "+a"(acc_br[11]),
          "+a"(acc_br[12]), "+a"(acc_br[13]), "+a"(acc_br[14]), "+a"(acc_br[15]),
          "=&v"(nxt_a0_d[0]), "=&v"(nxt_a0_d[1]), "=&v"(nxt_a0_d[2]), "=&v"(nxt_a0_d[3]),
          "=&v"(nxt_a0_d[4]), "=&v"(nxt_a0_d[5]), "=&v"(nxt_a0_d[6]), "=&v"(nxt_a0_d[7]),
          "=&v"(nxt_bl_d[0]), "=&v"(nxt_bl_d[1]), "=&v"(nxt_bl_d[2]), "=&v"(nxt_bl_d[3]),
          "=&v"(nxt_bl_d[4]), "=&v"(nxt_bl_d[5]), "=&v"(nxt_bl_d[6]), "=&v"(nxt_bl_d[7])
        : "v"(a1_0l), "v"(a1_1l), "v"(a1_2l), "v"(a1_3l),
          "v"(a1_0h), "v"(a1_1h), "v"(a1_2h), "v"(a1_3h),
          "v"(bl_0l), "v"(bl_1l), "v"(bl_2l), "v"(bl_3l),
          "v"(bl_0h), "v"(bl_1h), "v"(bl_2h), "v"(bl_3h),
          "v"(br_0l), "v"(br_1l), "v"(br_2l), "v"(br_3l),
          "v"(br_0h), "v"(br_1h), "v"(br_2h), "v"(br_3h),
          "v"(sa0), "v"(sa1), "v"(sbl0), "v"(sbl1), "v"(sbr0), "v"(sbr1),
          "v"(a0_p0), "v"(a0_p1), "v"(bl_p0), "v"(bl_p1),
          // %82..85: srds (sgpr-quad each)
          "s"(pf_a0.srd), "s"(pf_a1.srd), "s"(pf_bl.srd), "s"(pf_br.srd),
          // %86..89: soffs
          "s"(pf_a0.soff), "s"(pf_a1.soff), "s"(pf_bl.soff), "s"(pf_br.soff),
          // %90..93: lds_addrs for a0 (sgpr)
          "s"(pf_a0.lds_addrs[0]), "s"(pf_a0.lds_addrs[1]), "s"(pf_a0.lds_addrs[2]), "s"(pf_a0.lds_addrs[3]),
          // %94..97: lds_addrs for a1
          "s"(pf_a1.lds_addrs[0]), "s"(pf_a1.lds_addrs[1]), "s"(pf_a1.lds_addrs[2]), "s"(pf_a1.lds_addrs[3]),
          // %98..101: lds_addrs for bl
          "s"(pf_bl.lds_addrs[0]), "s"(pf_bl.lds_addrs[1]), "s"(pf_bl.lds_addrs[2]), "s"(pf_bl.lds_addrs[3]),
          // %102..105: lds_addrs for br
          "s"(pf_br.lds_addrs[0]), "s"(pf_br.lds_addrs[1]), "s"(pf_br.lds_addrs[2]), "s"(pf_br.lds_addrs[3]),
          // %106..109: voffs for a0 (vgpr)
          "v"(pf_a0.voffs[0]), "v"(pf_a0.voffs[1]), "v"(pf_a0.voffs[2]), "v"(pf_a0.voffs[3]),
          // %110..113: voffs for a1
          "v"(pf_a1.voffs[0]), "v"(pf_a1.voffs[1]), "v"(pf_a1.voffs[2]), "v"(pf_a1.voffs[3]),
          // %114..117: voffs for bl
          "v"(pf_bl.voffs[0]), "v"(pf_bl.voffs[1]), "v"(pf_bl.voffs[2]), "v"(pf_bl.voffs[3]),
          // %118..121: voffs for br
          "v"(pf_br.voffs[0]), "v"(pf_br.voffs[1]), "v"(pf_br.voffs[2]), "v"(pf_br.voffs[3])
        : "memory", "m0"
    );
}
#endif // STEP34_PF_INTERLEAVE

__device__ __forceinline__ void kpair_64mfma_step34(
    fp4_floatx4_t acc_bl[16], fp4_floatx4_t acc_br[16],
    const fp4_intx8_t A1[4],
    const fp4_intx8_t Bl[4], const fp4_intx8_t Br[4],
    const fp8e8m0_4 a1_raw[2], const fp8e8m0_4 bl_raw[2], const fp8e8m0_4 br_raw[2],
    float4 nxt_a0_d[8], float4 nxt_bl_d[8],
    uint32_t a0_p0, uint32_t a0_p1,
    uint32_t bl_p0, uint32_t bl_p1)
{
    // A1 tile splits (shared between Step3 and Step4)
    fp4_intx4_t a1_0l=fp4_lo4(A1[0]), a1_1l=fp4_lo4(A1[1]), a1_2l=fp4_lo4(A1[2]), a1_3l=fp4_lo4(A1[3]);
    fp4_intx4_t a1_0h=fp4_hi4(A1[0]), a1_1h=fp4_hi4(A1[1]), a1_2h=fp4_hi4(A1[2]), a1_3h=fp4_hi4(A1[3]);
    // Bl tile splits (Step3 only)
    fp4_intx4_t bl_0l=fp4_lo4(Bl[0]), bl_1l=fp4_lo4(Bl[1]), bl_2l=fp4_lo4(Bl[2]), bl_3l=fp4_lo4(Bl[3]);
    fp4_intx4_t bl_0h=fp4_hi4(Bl[0]), bl_1h=fp4_hi4(Bl[1]), bl_2h=fp4_hi4(Bl[2]), bl_3h=fp4_hi4(Bl[3]);
    // Br tile splits (Step4 only)
    fp4_intx4_t br_0l=fp4_lo4(Br[0]), br_1l=fp4_lo4(Br[1]), br_2l=fp4_lo4(Br[2]), br_3l=fp4_lo4(Br[3]);
    fp4_intx4_t br_0h=fp4_hi4(Br[0]), br_1h=fp4_hi4(Br[1]), br_2h=fp4_hi4(Br[2]), br_3h=fp4_hi4(Br[3]);
    // Scales
    unsigned sa0 = std::bit_cast<unsigned>(a1_raw[0]);
    unsigned sa1 = std::bit_cast<unsigned>(a1_raw[1]);
    unsigned sbl0 = std::bit_cast<unsigned>(bl_raw[0]);
    unsigned sbl1 = std::bit_cast<unsigned>(bl_raw[1]);
    unsigned sbr0 = std::bit_cast<unsigned>(br_raw[0]);
    unsigned sbr1 = std::bit_cast<unsigned>(br_raw[1]);

    // Barrier emitted separately (with memory clobber) so the MFMAs block stays lightweight
    // R19B: site _S1 (kpair_64mfma_step34, dead with default FUSED_STEP34=0)
    asm volatile("s_waitcnt vmcnt(" MXFP4_STR(STEP3_BARRIER_VMCNT) ")\ns_barrier\n" ::: "memory");
    asm volatile(
        // ═══ STEP 3: A1×Bl (32 MFMAs) + 8 ds_reads for nxt_a0 ═══
        // aiter-style 4:1 MFMA:ds_read spread — 1 ds_read after 1st MFMA per group of 4
        // Row 0 Phase 0 (even, sa0, lo): 4 MFMAs + 1 ds_read
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %48, %56, %0,  %72, %74 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %32, %78 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %48, %57, %1,  %72, %74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %48, %58, %2,  %72, %75 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %48, %59, %3,  %72, %75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Row 0 Phase 1 (even, sa0, hi): 4 MFMAs + 1 ds_read
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %52, %60, %0,  %72, %74 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %33, %78 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %52, %61, %1,  %72, %74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %52, %62, %2,  %72, %75 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %52, %63, %3,  %72, %75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Row 1 Phase 0 (odd, sa0, lo): 4 MFMAs + 1 ds_read
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %49, %56, %4,  %72, %74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %34, %78 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %49, %57, %5,  %72, %74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %49, %58, %6,  %72, %75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %49, %59, %7,  %72, %75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Row 1 Phase 1 (odd, sa0, hi): 4 MFMAs + 1 ds_read
        "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %53, %60, %4,  %72, %74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %35, %78 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %53, %61, %5,  %72, %74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %53, %62, %6,  %72, %75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %53, %63, %7,  %72, %75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Row 2 Phase 0 (even, sa1, lo): 4 MFMAs + 1 ds_read
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %50, %56, %8,  %73, %74 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %36, %79 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %50, %57, %9,  %73, %74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %50, %58, %10, %73, %75 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %50, %59, %11, %73, %75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Row 2 Phase 1 (even, sa1, hi): 4 MFMAs + 1 ds_read
        "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %54, %60, %8,  %73, %74 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %37, %79 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %54, %61, %9,  %73, %74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %54, %62, %10, %73, %75 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %54, %63, %11, %73, %75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Row 3 Phase 0 (odd, sa1, lo): 4 MFMAs + 1 ds_read
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %51, %56, %12, %73, %74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %38, %79 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %51, %57, %13, %73, %74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %51, %58, %14, %73, %75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %51, %59, %15, %73, %75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Row 3 Phase 1 (odd, sa1, hi): 4 MFMAs + 1 ds_read
        "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %55, %60, %12, %73, %74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %39, %79 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %55, %61, %13, %73, %74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %55, %62, %14, %73, %75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %55, %63, %15, %73, %75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // ═══ STEP 4: A1×Br (32 MFMAs) + 8 ds_reads for nxt_bl ═══
        // aiter-style 4:1 MFMA:ds_read spread — 1 ds_read after 1st MFMA per group of 4
        // Row 0 Phase 0 (even, sa0, lo): 4 MFMAs + 1 ds_read
        "v_mfma_scale_f32_16x16x128_f8f6f4 %16, %48, %64, %16, %72, %76 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %40, %80 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %17, %48, %65, %17, %72, %76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %18, %48, %66, %18, %72, %77 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %19, %48, %67, %19, %72, %77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Row 0 Phase 1 (even, sa0, hi): 4 MFMAs + 1 ds_read
        "v_mfma_scale_f32_16x16x128_f8f6f4 %16, %52, %68, %16, %72, %76 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %41, %80 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %17, %52, %69, %17, %72, %76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %18, %52, %70, %18, %72, %77 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %19, %52, %71, %19, %72, %77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Row 1 Phase 0 (odd, sa0, lo): 4 MFMAs + 1 ds_read
        "v_mfma_scale_f32_16x16x128_f8f6f4 %20, %49, %64, %20, %72, %76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %42, %80 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %21, %49, %65, %21, %72, %76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %22, %49, %66, %22, %72, %77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %23, %49, %67, %23, %72, %77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Row 1 Phase 1 (odd, sa0, hi): 4 MFMAs + 1 ds_read
        "v_mfma_scale_f32_16x16x128_f8f6f4 %20, %53, %68, %20, %72, %76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %43, %80 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %21, %53, %69, %21, %72, %76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %22, %53, %70, %22, %72, %77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %23, %53, %71, %23, %72, %77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Row 2 Phase 0 (even, sa1, lo): 4 MFMAs + 1 ds_read
        "v_mfma_scale_f32_16x16x128_f8f6f4 %24, %50, %64, %24, %73, %76 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %44, %81 offset:0\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %25, %50, %65, %25, %73, %76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %26, %50, %66, %26, %73, %77 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %27, %50, %67, %27, %73, %77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Row 2 Phase 1 (even, sa1, hi): 4 MFMAs + 1 ds_read
        "v_mfma_scale_f32_16x16x128_f8f6f4 %24, %54, %68, %24, %73, %76 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %45, %81 offset:2048\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %25, %54, %69, %25, %73, %76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %26, %54, %70, %26, %73, %77 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %27, %54, %71, %27, %73, %77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        // Row 3 Phase 0 (odd, sa1, lo): 4 MFMAs + 1 ds_read
        "v_mfma_scale_f32_16x16x128_f8f6f4 %28, %51, %64, %28, %73, %76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %46, %81 offset:4096\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %29, %51, %65, %29, %73, %76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %30, %51, %66, %30, %73, %77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %31, %51, %67, %31, %73, %77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
        // Row 3 Phase 1 (odd, sa1, hi): 4 MFMAs + 1 ds_read
        "v_mfma_scale_f32_16x16x128_f8f6f4 %28, %55, %68, %28, %73, %76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "ds_read_b128 %47, %81 offset:6144\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %29, %55, %69, %29, %73, %76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %30, %55, %70, %30, %73, %77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        "v_mfma_scale_f32_16x16x128_f8f6f4 %31, %55, %71, %31, %73, %77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
        : "+a"(acc_bl[0]),  "+a"(acc_bl[1]),  "+a"(acc_bl[2]),  "+a"(acc_bl[3]),
          "+a"(acc_bl[4]),  "+a"(acc_bl[5]),  "+a"(acc_bl[6]),  "+a"(acc_bl[7]),
          "+a"(acc_bl[8]),  "+a"(acc_bl[9]),  "+a"(acc_bl[10]), "+a"(acc_bl[11]),
          "+a"(acc_bl[12]), "+a"(acc_bl[13]), "+a"(acc_bl[14]), "+a"(acc_bl[15]),
          "+a"(acc_br[0]),  "+a"(acc_br[1]),  "+a"(acc_br[2]),  "+a"(acc_br[3]),
          "+a"(acc_br[4]),  "+a"(acc_br[5]),  "+a"(acc_br[6]),  "+a"(acc_br[7]),
          "+a"(acc_br[8]),  "+a"(acc_br[9]),  "+a"(acc_br[10]), "+a"(acc_br[11]),
          "+a"(acc_br[12]), "+a"(acc_br[13]), "+a"(acc_br[14]), "+a"(acc_br[15]),
          "=&v"(nxt_a0_d[0]), "=&v"(nxt_a0_d[1]), "=&v"(nxt_a0_d[2]), "=&v"(nxt_a0_d[3]),
          "=&v"(nxt_a0_d[4]), "=&v"(nxt_a0_d[5]), "=&v"(nxt_a0_d[6]), "=&v"(nxt_a0_d[7]),
          "=&v"(nxt_bl_d[0]), "=&v"(nxt_bl_d[1]), "=&v"(nxt_bl_d[2]), "=&v"(nxt_bl_d[3]),
          "=&v"(nxt_bl_d[4]), "=&v"(nxt_bl_d[5]), "=&v"(nxt_bl_d[6]), "=&v"(nxt_bl_d[7])
        : "v"(a1_0l), "v"(a1_1l), "v"(a1_2l), "v"(a1_3l),
          "v"(a1_0h), "v"(a1_1h), "v"(a1_2h), "v"(a1_3h),
          "v"(bl_0l), "v"(bl_1l), "v"(bl_2l), "v"(bl_3l),
          "v"(bl_0h), "v"(bl_1h), "v"(bl_2h), "v"(bl_3h),
          "v"(br_0l), "v"(br_1l), "v"(br_2l), "v"(br_3l),
          "v"(br_0h), "v"(br_1h), "v"(br_2h), "v"(br_3h),
          "v"(sa0), "v"(sa1), "v"(sbl0), "v"(sbl1), "v"(sbr0), "v"(sbr1),
          "v"(a0_p0), "v"(a0_p1), "v"(bl_p0), "v"(bl_p1)
    );
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
        // R19B: site _S2 (kpair_32mfma_with_lds_and_pf, hot path)
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

// ── 32 KPAIR MFMAs + 8 ds_reads (spread: 2 per row) + 8 pf ──
// Instead of front-loading all 8 ds_reads in row 0, spread them
// 2 per row for better LDS pipeline utilization.
// Each row reads one lo/hi pair: (d_i, d_{i+4}) from lds_a0/lds_a1.

template<int PF_N = 8, bool EMIT_BARRIER = false>
__device__ __forceinline__ void kpair_32mfma_with_lds_rowspread_pf(
    fp4_floatx4_t acc[16],
    const fp4_intx8_t A[4], const fp4_intx8_t B[4],
    const fp8e8m0_4 a_raw[2], const fp8e8m0_4 b_raw[2],
    float4 &d0, float4 &d1, float4 &d2, float4 &d3,
    float4 &d4, float4 &d5, float4 &d6, float4 &d7,
    uint32_t lds_a0, uint32_t lds_a1,
    const tile_pf_params &pf0, const tile_pf_params &pf1)
{
    KPAIR_SETUP();
    if constexpr (EMIT_BARRIER) {
        // R19B: site _S3 (kpair_32mfma_with_lds_rowspread_pf, hot path)
        asm volatile("s_waitcnt vmcnt(" MXFP4_STR(STEP3_BARRIER_VMCNT) ")\ns_barrier\n" ::: "memory");
    }
    // Row 0: 8 MFMAs + 2 ds_reads (d0, d4)
    {
        float4 &d_lo = d0, &d_hi = d4;
        asm volatile(
            "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %18, %26, %0,  %34, %36 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %18, %27, %1,  %34, %36 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "ds_read_b128 %16, %38 offset:0\n"
            "ds_read_b128 %17, %39 offset:0\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %18, %28, %2,  %34, %37 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %18, %29, %3,  %34, %37 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %22, %30, %0,  %34, %36 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %22, %31, %1,  %34, %36 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %22, %32, %2,  %34, %37 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %22, %33, %3,  %34, %37 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            : KPAIR_ACC_CLOBBER, "=&v"(d_lo), "=&v"(d_hi)
            : KPAIR_INPUTS, "v"(lds_a0), "v"(lds_a1)
        );
    }
    if constexpr (PF_N > 0) emit_one_pf(pf0, 0);
    if constexpr (PF_N > 1) emit_one_pf(pf0, 1);
    // Row 1: 8 MFMAs + 2 ds_reads (d1, d5)
    {
        float4 &d_lo = d1, &d_hi = d5;
        asm volatile(
            "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %19, %26, %4,  %34, %36 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %19, %27, %5,  %34, %36 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "ds_read_b128 %16, %38 offset:2048\n"
            "ds_read_b128 %17, %39 offset:2048\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %19, %28, %6,  %34, %37 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %19, %29, %7,  %34, %37 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %23, %30, %4,  %34, %36 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %23, %31, %5,  %34, %36 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %23, %32, %6,  %34, %37 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %23, %33, %7,  %34, %37 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            : KPAIR_ACC_CLOBBER, "=&v"(d_lo), "=&v"(d_hi)
            : KPAIR_INPUTS, "v"(lds_a0), "v"(lds_a1)
        );
    }
    if constexpr (PF_N > 2) emit_one_pf(pf0, 2);
    if constexpr (PF_N > 3) emit_one_pf(pf0, 3);
    // Row 2: 8 MFMAs + 2 ds_reads (d2, d6)
    {
        float4 &d_lo = d2, &d_hi = d6;
        asm volatile(
            "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %20, %26, %8,  %35, %36 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %20, %27, %9,  %35, %36 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "ds_read_b128 %16, %38 offset:4096\n"
            "ds_read_b128 %17, %39 offset:4096\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %20, %28, %10, %35, %37 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %20, %29, %11, %35, %37 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %24, %30, %8,  %35, %36 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %24, %31, %9,  %35, %36 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %10, %24, %32, %10, %35, %37 op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %11, %24, %33, %11, %35, %37 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            : KPAIR_ACC_CLOBBER, "=&v"(d_lo), "=&v"(d_hi)
            : KPAIR_INPUTS, "v"(lds_a0), "v"(lds_a1)
        );
    }
    if constexpr (PF_N > 4) emit_one_pf(pf1, 0);
    if constexpr (PF_N > 5) emit_one_pf(pf1, 1);
    // Row 3: 8 MFMAs + 2 ds_reads (d3, d7)
    {
        float4 &d_lo = d3, &d_hi = d7;
        asm volatile(
            "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %21, %26, %12, %35, %36 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %21, %27, %13, %35, %36 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "ds_read_b128 %16, %38 offset:6144\n"
            "ds_read_b128 %17, %39 offset:6144\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %21, %28, %14, %35, %37 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %21, %29, %15, %35, %37 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %12, %25, %30, %12, %35, %36 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %13, %25, %31, %13, %35, %36 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %14, %25, %32, %14, %35, %37 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %15, %25, %33, %15, %35, %37 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
            : KPAIR_ACC_CLOBBER, "=&v"(d_lo), "=&v"(d_hi)
            : KPAIR_INPUTS, "v"(lds_a0), "v"(lds_a1)
        );
    }
    if constexpr (PF_N > 6) emit_one_pf(pf1, 2);
    if constexpr (PF_N > 7) emit_one_pf(pf1, 3);
}

// ══════════════════════════════════════════════════════════════
// Main kernel
// ══════════════════════════════════════════════════════════════

#if defined(WAVES_PER_EU_1)
__attribute__((amdgpu_waves_per_eu(1, 1)))
#elif defined(WAVES_PER_EU_2)
__attribute__((amdgpu_waves_per_eu(2, 2)))
#endif
#if defined(AGPR_REGS_HINT_192)
__attribute__((amdgpu_num_agpr(192)))
#elif defined(AGPR_REGS_HINT_128)
__attribute__((amdgpu_num_agpr(128)))
#elif defined(AGPR_REGS_HINT_256)
__attribute__((amdgpu_num_agpr(256)))
#endif
__global__ __launch_bounds__(_NUM_THREADS, 1)
void mxfp4_gluon_cpp_kernel(const gluon_globals g) {
    static_assert(K_BYTES % BK == 0 && N_DIM % BLK == 0 && M_DIM % BLK == 0);

    constexpr int bpc = N_DIM / BLK;
    constexpr int a_packs = RBM / 32;
    constexpr int b_packs = RBN / 32;
    constexpr int A0_SLOTS = 2;
    constexpr int A1_SLOTS = 2;
    constexpr int BL_SLOTS = 2;
    constexpr int BR_SLOTS = 2;
#if GLOBAL_B
    __shared__ ST_tile A0_db[2], A1_db[2];  // B tiles loaded directly from global
#else
    __shared__ ST_tile A0_db[2], A1_db[2], Bl_db[2], Br_db[2];
#endif

    // XCD-aware dispatch + GROUP_SIZE_M swizzle for L2 B-tile reuse
    constexpr int NUM_XCDS = 8;
#ifndef GROUP_SIZE_M
#define GROUP_SIZE_M 4
#endif
    constexpr int GROUP_M = GROUP_SIZE_M;
    const int total_blocks = gridDim.x;
    const int bpr = total_blocks / bpc;

    const int pids_per_xcd = (total_blocks + NUM_XCDS - 1) / NUM_XCDS;
    int tall_xcds = total_blocks % NUM_XCDS;
    if (tall_xcds == 0) tall_xcds = NUM_XCDS;
    {
        // Static dispatch: raw_bid = blockIdx.x
        const int raw_bid = (int)blockIdx.x;

    // XCD pid remapping: Gluon-style "tall XCDs" for correct remainder handling
    const int xcd = raw_bid % NUM_XCDS;
    const int local_pid = raw_bid / NUM_XCDS;
    int bid;
    if (xcd < tall_xcds) {
        bid = xcd * pids_per_xcd + local_pid;
    } else {
        bid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid;
    }
    if (bid >= total_blocks) return;

    int br, bc;
    // GROUP_SIZE_M swizzle within XCD's block range
    const int num_pig = GROUP_M * bpc;
    const int gid = bid / num_pig;
    const int fpm = gid * GROUP_M;
    const int gsm = (bpr - fpm < GROUP_M) ? (bpr - fpm) : GROUP_M;
    br = fpm + (bid % gsm);
    bc = (bid % num_pig) / gsm;
    const int wm = warpid() / WARPS_N, wn = warpid() % WARPS_N;

    uint32_t so_a[PF_MPT], so_b[PF_MPT];
    G::prefill_swizzled_offsets(A0_db[0], g.a, so_a);
#if !GLOBAL_B
    G::prefill_swizzled_offsets(Bl_db[0], g.b, so_b);
#endif

    // Scale SRDs — merged global_load format (64-row super-groups, dwordx2 loads)
    // lane_soff_x2: doubled offsets for merged format where each dword position is 8 bytes
    const uint32_t lane_soff_x2 =
        (static_cast<uint32_t>(kittens::laneid() / 16) << 7) |
        (static_cast<uint32_t>(kittens::laneid() % 16) << 3);

    // One SRD per tile-half, pointing to the 64-row super-group base
    i32x4 a0_srd = make_scale_srd(global_loadd_scale_row_base_ptr(
        g.a_scale, (br * BLK + wm * RBM) >> 6));
    i32x4 a1_srd = make_scale_srd(global_loadd_scale_row_base_ptr(
        g.a_scale, (br * BLK + HB + wm * RBM) >> 6));
    i32x4 bl_srd = make_scale_srd(global_loadd_scale_row_base_ptr(
        g.b_scale, (bc * BLK + wn * RBN) >> 6));
    i32x4 br_srd = make_scale_srd(global_loadd_scale_row_base_ptr(
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
    uint32_t lb_a0[A0_SLOTS], lb_a1[A1_SLOTS];
#if !GLOBAL_B
    uint32_t lb_br[BR_SLOTS];
    uint32_t lb_bl[BL_SLOTS];
#endif
    for (int d = 0; d < A0_SLOTS; ++d) {
        lb_a0[d]=lb(A0_db[d]);
    }
    for (int d = 0; d < A1_SLOTS; ++d) {
        lb_a1[d]=lb(A1_db[d]);
    }
#if !GLOBAL_B
    for (int d = 0; d < BR_SLOTS; ++d) {
        lb_br[d]=lb(Br_db[d]);
    }
    for (int d = 0; d < BL_SLOTS; ++d) {
        lb_bl[d]=lb(Bl_db[d]);
    }
#endif

    auto load_tiles = [&](int bt, int db) {
        emit_tile_pf(A0_db[db], g.a, coord<ST_tile>(0,0,br*2,    bt), so_a, srd_a, base_a, lb_a0[db], static_cast<int>(kittens::coherency::cache_all));
        emit_tile_pf(A1_db[db], g.a, coord<ST_tile>(0,0,br*2+1,  bt), so_a, srd_a, base_a, lb_a1[db], static_cast<int>(kittens::coherency::cache_all));
#if !GLOBAL_B
        emit_tile_pf(Bl_db[db], g.b, coord<ST_tile>(0,0,bc*2,    bt), so_b, srd_b, base_b, lb_bl[db], static_cast<int>(kittens::coherency::cache_all));
        emit_tile_pf(Br_db[db], g.b, coord<ST_tile>(0,0,bc*2+1,  bt), so_b, srd_b, base_b, lb_br[db], static_cast<int>(kittens::coherency::cache_all));
#endif
    };

    // Pre-compute LDS addresses as 16 static named variables (one per db-slot × phase).
    // Avoids runtime-indexed [2][2] arrays that compiler spills to LDS + ds_read_b64.
    // Selection via ternary (compiles to v_cndmask). No swap needed.
    uint32_t a0_0_p0, a0_0_p1, a0_1_p0, a0_1_p1;
#if !GLOBAL_B
    uint32_t bl_0_p0, bl_0_p1, bl_1_p0, bl_1_p1;
    uint32_t br_0_p0, br_0_p1, br_1_p0, br_1_p1;
#endif
    uint32_t a1_0_p0, a1_0_p1, a1_1_p0, a1_1_p1;
    compute_lds_base_addrs<A_row_reg>(kittens::subtile_inplace<RBM, BK>(A0_db[0], {wm, 0}), a0_0_p0, a0_0_p1);
    compute_lds_base_addrs<A_row_reg>(kittens::subtile_inplace<RBM, BK>(A0_db[1], {wm, 0}), a0_1_p0, a0_1_p1);
#if !GLOBAL_B
    compute_lds_base_addrs<B_row_reg>(kittens::subtile_inplace<RBN, BK>(Bl_db[0], {wn, 0}), bl_0_p0, bl_0_p1);
    compute_lds_base_addrs<B_row_reg>(kittens::subtile_inplace<RBN, BK>(Bl_db[1], {wn, 0}), bl_1_p0, bl_1_p1);
    compute_lds_base_addrs<B_row_reg>(kittens::subtile_inplace<RBN, BK>(Br_db[0], {wn, 0}), br_0_p0, br_0_p1);
    compute_lds_base_addrs<B_row_reg>(kittens::subtile_inplace<RBN, BK>(Br_db[1], {wn, 0}), br_1_p0, br_1_p1);
#endif
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


#if GLOBAL_B
    // GlobalB: pre-compute per-lane voffsets for Bl and Br half-tiles.
    // These are constant across K-iterations; K advance uses soffset.
    const int bl_n_block_start = bc * (BLK / 16) + wn * (RBN / 16);
    const int br_n_block_start = bc * (BLK / 16) + (BLK / 2 / 16) + wn * (RBN / 16);
    uint32_t bl_voffs[8], br_voffs[8];
    compute_b_global_load_voffs(bl_voffs, bl_n_block_start, K_BYTES);
    compute_b_global_load_voffs(br_voffs, br_n_block_start, K_BYTES);
#endif

    // ═══════════ Prologue ═══════════
    load_tiles(0, 0);
    if (k_byte_iters > 1) load_tiles(1, 1);
#if GLOBAL_B
    // GLOBALB: no LDS load for B tiles — loaded via buffer_load in prologue
#endif

    fp8e8m0_4 pf_a0[a_packs], pf_a1[a_packs], pf_bl[b_packs], pf_br[b_packs];
    {
        load_pq_scale_x2_async(a0_srd, lane_soff_x2, 0, pf_a0[0], pf_a0[1]);
        load_pq_scale_x2_async(a1_srd, lane_soff_x2, 0, pf_a1[0], pf_a1[1]);
        load_pq_scale_x2_async(bl_srd, lane_soff_x2, 0, pf_bl[0], pf_bl[1]);
        load_pq_scale_x2_async(br_srd, lane_soff_x2, 0, pf_br[0], pf_br[1]);
    }

    // Wait for ALL tile loads (A + B prologue) and scales
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    // Load A0 + Bl from LDS
    A_row_reg a0_rt;
    fp4_load_st_to_rt(a0_rt, kittens::subtile_inplace<RBM, BK>(A0_db[0], {wm, 0}));
    fp4_intx8_t tA0[4], tBl[4];
#if GLOBAL_B
    // Load Bl directly from global_loadd global via buffer_load (NOT LDS)
    {
        // First extract A0 from LDS (done before any buffer_load to avoid VGPR conflict)
        asm volatile("s_waitcnt lgkmcnt(0)");
        #pragma unroll
        for (int i = 0; i < 4; i++) tA0[i] = fp4_extract_tile(a0_rt, i);
        // Force A0 data to be consumed (prevent compiler from reusing a0_rt VGPRs)
        asm volatile("" :: "v"(tA0[0]), "v"(tA0[1]), "v"(tA0[2]), "v"(tA0[3]));
        // Now load Bl from global_loadd global
        float4 bl_d0[8];
        load_b_global_8(bl_d0, srd_b, bl_voffs, 0);
        asm volatile("s_waitcnt vmcnt(0)");
        extract_tile(bl_d0, tBl);
    }
#else
    B_row_reg bl_rt;
    fp4_load_st_to_rt(bl_rt, kittens::subtile_inplace<RBN, BK>(Bl_db[0], {wn, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        tA0[i] = fp4_extract_tile(a0_rt, i);
        tBl[i] = fp4_extract_tile(bl_rt, i);
    }
#endif

#ifndef TAIL_SPLIT
#define TAIL_SPLIT 0
#endif

#if TAIL_SPLIT
    // ── Non-SWAP path: tail-split to eliminate wasted prefetch/LDS on last iter ──
    // Steady-state loop: bt = 0 .. k_byte_iters-2
    // Tail: bt = k_byte_iters-1 (no prefetch, no next-iter scale/LDS loads)
    // entry to reduce inter-wave de-schedule jitter. Placed BEFORE the
    // #pragma unroll so the pragma stays adjacent to the for-loop (clang
    // requires that adjacency).
#if GLOBAL_B
    // ═══ GLOBAL_B K-loop: B tiles loaded via buffer_load_dwordx4 from global ═══
    // Pipeline: each iteration loads Br(cur) and nxt_Bl from global,
    // A0 and A1 from LDS (same as baseline), prefetches only A tiles to LDS.
    // Preshuffle flat-byte stride per K-iteration:
    // BK=128 raw bytes = BK/32 k_blocks, each k_block = 512 flat bytes
    constexpr int BK_PRESHUFFLE_STRIDE = (BK / 32) * 512;  // = 2048

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
        const int pf_a0bl  = cur;

        const uint32_t sel_a1_p0 = cur ? a1_1_p0 : a1_0_p0;
        const uint32_t sel_a1_p1 = cur ? a1_1_p1 : a1_0_p1;
        const uint32_t sel_a0_p0 = nxt ? a0_1_p0 : a0_0_p0;
        const uint32_t sel_a0_p1 = nxt ? a0_1_p1 : a0_0_p1;

        const int pf_bt = (bt + 2 < k_byte_iters) ? (bt + 2) : (k_byte_iters - 1);
        tile_pf_params pf_a0_p = make_pf_params(A0_db[pf_a0bl], g.a, coord<ST_tile>(0,0,br*2,   pf_bt), so_a, srd_a, base_a, lb_a0[pf_a0bl], static_cast<int>(kittens::coherency::cache_all));
        tile_pf_params pf_a1_p = make_pf_params(A1_db[cur],     g.a, coord<ST_tile>(0,0,br*2+1, pf_bt), so_a, srd_a, base_a, lb_a1[cur],     static_cast<int>(kittens::coherency::cache_all));

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

        // Load Br from global memory
        float4 br_d[8];
        const uint32_t br_k_soff = (uint32_t)bt * BK_PRESHUFFLE_STRIDE;
        load_b_global_8(br_d, srd_b, br_voffs, br_k_soff);

        // Step 1: A0*Bl (32 pure MFMAs, Bl already in tBl)
        tile_pf_params dummy_pf = {};
        kpair_32mfma_with_pf<0>(acc_A0Bl, tA0, tBl, a0_raw, bl_raw, dummy_pf, dummy_pf);

        // Wait for Br loads, extract
        asm volatile("s_waitcnt vmcnt(0)");
        fp4_intx8_t tBr[4];
        extract_tile(br_d, tBr);

        // Step 2: A0*Br (32 MFMAs) + ds_read A1
        float4 a1_d[8];
        kpair_32mfma_with_lds(acc_A0Br, tA0, tBr, a0_raw, br_raw,
            a1_d[0], a1_d[1], a1_d[2], a1_d[3],
            a1_d[4], a1_d[5], a1_d[6], a1_d[7],
            sel_a1_p0, sel_a1_p1);

        asm volatile("s_waitcnt lgkmcnt(0)");
        fp4_intx8_t tA1[4];
        extract_tile(a1_d, tA1);

        // Barrier for A tile double-buffer sync
        asm volatile("s_waitcnt vmcnt(0)\ns_barrier\n" ::: "memory");

        // Issue nxt_Bl buffer_load EARLY — before Step3+4's 64 MFMAs to hide latency
        float4 nxt_bl_d[8];
        const uint32_t nxt_bl_k_soff = (uint32_t)(bt + 1) * BK_PRESHUFFLE_STRIDE;
        load_b_global_8(nxt_bl_d, srd_b, bl_voffs, nxt_bl_k_soff);

        // Step 3: A1*Bl (32 MFMAs) + ds_read nxt_A0
        // nxt_Bl buffer_loads are in-flight, overlapping with these MFMAs
        float4 nxt_a0_d[8];
        kpair_32mfma_with_lds(acc_A1Bl, tA1, tBl, a1_raw, bl_raw,
            nxt_a0_d[0], nxt_a0_d[1], nxt_a0_d[2], nxt_a0_d[3],
            nxt_a0_d[4], nxt_a0_d[5], nxt_a0_d[6], nxt_a0_d[7],
            sel_a0_p0, sel_a0_p1);

        // Step 4: A1*Br (32 pure MFMAs)
        kpair_32mfma_with_pf<0>(acc_A1Br, tA1, tBr, a1_raw, br_raw, dummy_pf, dummy_pf);

        // Prefetch A tiles to LDS for next+1 iteration
        emit_pf_tail<0>(pf_a0_p, pf_a1_p);
        asm volatile("" ::: "memory");

        // Wait for nxt_A0 ds_reads and nxt_Bl buffer_loads
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
        extract_tile(nxt_a0_d, tA0);
        extract_tile(nxt_bl_d, tBl);
    }

    // ── GLOBAL_B Tail iteration ──
    {
        const int bt = k_byte_iters - 1;
        const int cur = bt & 1;
        const uint32_t sel_a1_p0 = cur ? a1_1_p0 : a1_0_p0;
        const uint32_t sel_a1_p1 = cur ? a1_1_p1 : a1_0_p1;

        fp8e8m0_4 a0_raw[a_packs], a1_raw[a_packs], bl_raw[b_packs], br_raw[b_packs];
        #pragma unroll
        for (int p = 0; p < a_packs; ++p) { a0_raw[p] = pf_a0[p]; a1_raw[p] = pf_a1[p]; }
        #pragma unroll
        for (int p = 0; p < b_packs; ++p) { bl_raw[p] = pf_bl[p]; br_raw[p] = pf_br[p]; }

        // Load Br from global
        float4 br_d[8];
        const uint32_t br_k_soff = (uint32_t)bt * BK_PRESHUFFLE_STRIDE;
        load_b_global_8(br_d, srd_b, br_voffs, br_k_soff);

        // Step 1: A0*Bl
        tile_pf_params dummy_pf = {};
        kpair_32mfma_with_pf<0>(acc_A0Bl, tA0, tBl, a0_raw, bl_raw, dummy_pf, dummy_pf);

        // Wait for Br
        asm volatile("s_waitcnt vmcnt(0)");
        fp4_intx8_t tBr[4];
        extract_tile(br_d, tBr);

        // Step 2: A0*Br + ds_read A1
        float4 a1_d[8];
        kpair_32mfma_with_lds(acc_A0Br, tA0, tBr, a0_raw, br_raw,
            a1_d[0], a1_d[1], a1_d[2], a1_d[3],
            a1_d[4], a1_d[5], a1_d[6], a1_d[7],
            sel_a1_p0, sel_a1_p1);

        asm volatile("s_waitcnt lgkmcnt(0)");
        fp4_intx8_t tA1[4];
        extract_tile(a1_d, tA1);

        // Tail barrier
        asm volatile("s_waitcnt vmcnt(0)\ns_barrier\n" ::: "memory");

        // Steps 3+4: pure MFMAs, no ds_reads, no prefetches
        kpair_32mfma_with_pf<0>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw, dummy_pf, dummy_pf);
        kpair_32mfma_with_pf<0>(acc_A1Br, tA1, tBr, a1_raw, br_raw, dummy_pf, dummy_pf);
    }

#else // !GLOBAL_B — original LDS-based K-loop

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
        const int cur_a0bl = cur;
        const int nxt_a0bl = nxt;
        const int pf_a0bl  = cur;

        const uint32_t sel_br_p0 = cur ? br_1_p0 : br_0_p0;
        const uint32_t sel_br_p1 = cur ? br_1_p1 : br_0_p1;
        const uint32_t sel_a1_p0 = cur ? a1_1_p0 : a1_0_p0;
        const uint32_t sel_a1_p1 = cur ? a1_1_p1 : a1_0_p1;
        const uint32_t sel_a0_p0 = nxt ? a0_1_p0 : a0_0_p0;
        const uint32_t sel_a0_p1 = nxt ? a0_1_p1 : a0_0_p1;
        const uint32_t sel_bl_p0 = nxt ? bl_1_p0 : bl_0_p0;
        const uint32_t sel_bl_p1 = nxt ? bl_1_p1 : bl_0_p1;

        const int pf_bt = (bt + 2 < k_byte_iters) ? (bt + 2) : (k_byte_iters - 1);
        // A1/Br stay double-buffered (slot cur).
        tile_pf_params pf_a0_p = make_pf_params(A0_db[pf_a0bl], g.a, coord<ST_tile>(0,0,br*2,     pf_bt), so_a, srd_a, base_a, lb_a0[pf_a0bl], static_cast<int>(kittens::coherency::cache_all));
        tile_pf_params pf_a1_p = make_pf_params(A1_db[cur],     g.a, coord<ST_tile>(0,0,br*2+1,   pf_bt), so_a, srd_a, base_a, lb_a1[cur],     static_cast<int>(kittens::coherency::cache_all));
        tile_pf_params pf_bl_p = make_pf_params(Bl_db[pf_a0bl], g.b, coord<ST_tile>(0,0,bc*2,     pf_bt), so_b, srd_b, base_b, lb_bl[pf_a0bl], static_cast<int>(kittens::coherency::cache_all));
        tile_pf_params pf_br_p = make_pf_params(Br_db[cur],     g.b, coord<ST_tile>(0,0,bc*2+1,   pf_bt), so_b, srd_b, base_b, lb_br[cur],     static_cast<int>(kittens::coherency::cache_all));
        constexpr bool _r39a_in_tail = false;
        const uint32_t _r39a_scale_idx = static_cast<uint32_t>(bt + 1);

        fp8e8m0_4 a0_raw[a_packs], a1_raw[a_packs], bl_raw[b_packs], br_raw[b_packs];
        #pragma unroll
        for (int p = 0; p < a_packs; ++p) { a0_raw[p] = pf_a0[p]; a1_raw[p] = pf_a1[p]; }
        #pragma unroll
        for (int p = 0; p < b_packs; ++p) { bl_raw[p] = pf_bl[p]; br_raw[p] = pf_br[p]; }
        if (!_r39a_in_tail) {
            const uint32_t nxt_scale = static_cast<uint32_t>(bt + 1) << 9;
            load_pq_scale_x2_async(a0_srd, lane_soff_x2, nxt_scale, pf_a0[0], pf_a0[1]);
            load_pq_scale_x2_async(a1_srd, lane_soff_x2, nxt_scale, pf_a1[0], pf_a1[1]);
            load_pq_scale_x2_async(bl_srd, lane_soff_x2, nxt_scale, pf_bl[0], pf_bl[1]);
            load_pq_scale_x2_async(br_srd, lane_soff_x2, nxt_scale, pf_br[0], pf_br[1]);
        }
        // else: leave pf_* alone (frozen scale)

        // Steps 1+2 merged: A0*Bl (32 MFMAs) + ds_read Br + A0*Br (32 MFMAs) + ds_read A1
        float4 br_d[8], a1_d[8];
#if STEP12_PF_INTERLEAVE
        // R68 axis-step12pf NON-SPLIT: 16 prefetches embedded in step12 asm block.
        kpair_64mfma_step12_pf_interleaved(acc_A0Bl, acc_A0Br, tA0, tBl,
            a0_raw, bl_raw, br_raw, br_d, a1_d,
            sel_br_p0, sel_br_p1, sel_a1_p0, sel_a1_p1,
            pf_a0_p, pf_a1_p, pf_bl_p, pf_br_p);
#elif STEP12_SPLIT_PF
        // STEP12_SPLIT_PF: split step12 into 2x kpair_32mfma_with_lds + 4 prefetches.
        // Step1: A0*Bl (32 MFMAs) + ds_read Br
        kpair_32mfma_with_lds(acc_A0Bl, tA0, tBl, a0_raw, bl_raw,
            br_d[0], br_d[1], br_d[2], br_d[3],
            br_d[4], br_d[5], br_d[6], br_d[7],
            sel_br_p0, sel_br_p1);
        // Wait for Br ds_reads, extract
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        fp4_intx8_t tBr_s12[4];
        extract_tile(br_d, tBr_s12);
        // 4 prefetches: A0_db + Bl_db (SAFE — A0/Bl already in regs, no LDS race)
        emit_one_pf(pf_a0_p, 0);
        emit_one_pf(pf_a0_p, 1);
        emit_one_pf(pf_bl_p, 0);
        emit_one_pf(pf_bl_p, 1);
        asm volatile("" ::: "memory");
        // Step2: A0*Br (32 MFMAs) + ds_read A1
        kpair_32mfma_with_lds(acc_A0Br, tA0, tBr_s12, a0_raw, br_raw,
            a1_d[0], a1_d[1], a1_d[2], a1_d[3],
            a1_d[4], a1_d[5], a1_d[6], a1_d[7],
            sel_a1_p0, sel_a1_p1);
#else
        kpair_64mfma_step12(acc_A0Bl, acc_A0Br, tA0, tBl,
            a0_raw, bl_raw, br_raw, br_d, a1_d,
            sel_br_p0, sel_br_p1, sel_a1_p0, sel_a1_p1);
#endif

        asm volatile("s_waitcnt lgkmcnt(0)");
        fp4_intx8_t tBr[4], tA1[4];
#if STEP12_SPLIT_PF
        // Br already extracted above in the split path
        tBr[0] = tBr_s12[0]; tBr[1] = tBr_s12[1]; tBr[2] = tBr_s12[2]; tBr[3] = tBr_s12[3];
#else
        extract_tile(br_d, tBr);
#endif
        extract_tile(a1_d, tA1);

#if FUSED_STEP34
        // Fused Step34: barrier + 64 MFMAs + 16 ds_reads in one asm block
        float4 nxt_a0_d[8];
        float4 nxt_bl_d[8];
#if STEP12_PF_INTERLEAVE
        // R68: prefetches already issued in step12. step34 base path, no tail.
        kpair_64mfma_step34(acc_A1Bl, acc_A1Br, tA1, tBl, tBr,
            a1_raw, bl_raw, br_raw, nxt_a0_d, nxt_bl_d,
            sel_a0_p0, sel_a0_p1, sel_bl_p0, sel_bl_p1);
#elif STEP34_PF_INTERLEAVE
        // R66 axis-A: prefetches inlined into the asm block, no post-tail needed.
        // When STEP12_SPLIT_PF=1, the 4 extra A0/Bl loads from step12 split are
        // harmless duplicates — step34pf still emits all 16 prefetches.
        kpair_64mfma_step34_pf_interleaved(acc_A1Bl, acc_A1Br, tA1, tBl, tBr,
            a1_raw, bl_raw, br_raw, nxt_a0_d, nxt_bl_d,
            sel_a0_p0, sel_a0_p1, sel_bl_p0, sel_bl_p1,
            pf_a0_p, pf_a1_p, pf_bl_p, pf_br_p);
#else
        kpair_64mfma_step34(acc_A1Bl, acc_A1Br, tA1, tBl, tBr,
            a1_raw, bl_raw, br_raw, nxt_a0_d, nxt_bl_d,
            sel_a0_p0, sel_a0_p1, sel_bl_p0, sel_bl_p1);
        emit_pf_tail<0>(pf_a0_p, pf_a1_p);
        emit_pf_tail<0>(pf_bl_p, pf_br_p);
#endif
#elif R37_FIX_B
        // R37 Fix B (default): use fused step3+step4 (correctness fix) while
        // that the FUSED_STEP34=1 path otherwise bypasses.
        float4 nxt_a0_d[8];
        float4 nxt_bl_d[8];
        kpair_64mfma_step34(acc_A1Bl, acc_A1Br, tA1, tBl, tBr,
            a1_raw, bl_raw, br_raw, nxt_a0_d, nxt_bl_d,
            sel_a0_p0, sel_a0_p1, sel_bl_p0, sel_bl_p1);
        // R37: fence the scheduler — `-mllvm -amdgpu-sched-strategy=max-memory-clause`
        // is otherwise free to hoist the upcoming buffer_load_to_lds prefetches across
        // iteration boundaries (these intrinsics aren't asm volatile), which corrupts
        // the LDS double-buffer state because the next iter's s_barrier vmcnt count
        // is now wrong. Plain memory clobber asm volatile prevents the hoist.
        asm volatile("" ::: "memory");

        // R25-C: in the last R25C_TAIL_PF_OFF_ITERS iters, drop global prefetch.
        // Branch folds to compile-time when K-loop fully unrolls (R25C_ACTIVE
        // gates K_DIM ≤ R25C_K_LIMIT); becomes constexpr false otherwise.
#if R25C_ACTIVE
        const bool _r25c_tail_no_pf = (bt >= k_byte_iters - 1 - R25C_TAIL_PF_OFF_ITERS);
#else
        constexpr bool _r25c_tail_no_pf = false;
#endif
        if (!_r25c_tail_no_pf) {
            // Issue full A0 + A1 prefetches (would have come from STEP3_PF_N).
            emit_pf_tail<0>(pf_a0_p, pf_a1_p);
            // Issue Bl + Br prefetches (would have come from STEP4_PF_N).
            // emit_one_pf burst after Bl; semantically the SAME loads — so we
            // emit both pf groups here uniformly.
            emit_pf_tail<0>(pf_bl_p, pf_br_p);
        }
        // R37: fence again post-prefetch so the next iter's barrier vmcnt is correct.
        asm volatile("" ::: "memory");
#endif // FUSED_STEP34 / R37_FIX_B
        asm volatile("s_waitcnt lgkmcnt(0)");
        extract_tile(nxt_a0_d, tA0);
        extract_tile(nxt_bl_d, tBl);
        // R21B/R22C: opt-in scheduling hooks (no-op at defaults). Site 1.
    }

    // ── Tail iteration: no prefetch, no next-iter scale/LDS loads ──
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

        // Steps 1+2: still need Br and A1 from current LDS
        float4 br_d[8], a1_d[8];
        kpair_64mfma_step12(acc_A0Bl, acc_A0Br, tA0, tBl,
            a0_raw, bl_raw, br_raw, br_d, a1_d,
            sel_br_p0, sel_br_p1, sel_a1_p0, sel_a1_p1);

        asm volatile("s_waitcnt lgkmcnt(0)");
        fp4_intx8_t tBr[4], tA1[4];
        extract_tile(br_d, tBr);
        extract_tile(a1_d, tA1);

        // Tail: always emit barrier (no embedded barrier in pure-MFMA Step3/4)
        // R19B: TAIL site _S2 (TAIL_SPLIT==0, dead for parents using -DTAIL_SPLIT=1)
        asm volatile("s_waitcnt vmcnt(" MXFP4_STR(TAIL_BARRIER_VMCNT) ")\ns_barrier\n" ::: "memory");

        // Steps 3+4: pure MFMAs, no ds_reads, no prefetches
        tile_pf_params dummy_pf = {};
        kpair_32mfma_with_pf<0>(acc_A1Bl, tA1, tBl, a1_raw, bl_raw, dummy_pf, dummy_pf);
        kpair_32mfma_with_pf<0>(acc_A1Br, tA1, tBr, a1_raw, br_raw, dummy_pf, dummy_pf);
    }

#endif // GLOBAL_B

#else // TAIL_SPLIT == 0: original single-loop (better for large K)

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
        tile_pf_params pf_a0_p = make_pf_params(A0_db[cur], g.a, coord<ST_tile>(0,0,br*2,     pf_bt), so_a, srd_a, base_a, lb_a0[cur], static_cast<int>(kittens::coherency::cache_all));
        tile_pf_params pf_a1_p = make_pf_params(A1_db[cur], g.a, coord<ST_tile>(0,0,br*2+1,   pf_bt), so_a, srd_a, base_a, lb_a1[cur], static_cast<int>(kittens::coherency::cache_all));
        tile_pf_params pf_bl_p = make_pf_params(Bl_db[cur], g.b, coord<ST_tile>(0,0,bc*2,     pf_bt), so_b, srd_b, base_b, lb_bl[cur], static_cast<int>(kittens::coherency::cache_all));
        tile_pf_params pf_br_p = make_pf_params(Br_db[cur], g.b, coord<ST_tile>(0,0,bc*2+1,   pf_bt), so_b, srd_b, base_b, lb_br[cur], static_cast<int>(kittens::coherency::cache_all));

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

        float4 nxt_bl_d[8];

        float4 br_d[8], a1_d[8];
#if STEP12_PF_INTERLEAVE
        kpair_64mfma_step12_pf_interleaved(acc_A0Bl, acc_A0Br, tA0, tBl,
            a0_raw, bl_raw, br_raw, br_d, a1_d,
            sel_br_p0, sel_br_p1, sel_a1_p0, sel_a1_p1,
            pf_a0_p, pf_a1_p, pf_bl_p, pf_br_p);
#elif STEP12_SPLIT_PF
        // STEP12_SPLIT_PF: split step12 into 2x kpair_32mfma_with_lds + 4 prefetches.
        // Step1: A0*Bl (32 MFMAs) + ds_read Br
        kpair_32mfma_with_lds(acc_A0Bl, tA0, tBl, a0_raw, bl_raw,
            br_d[0], br_d[1], br_d[2], br_d[3],
            br_d[4], br_d[5], br_d[6], br_d[7],
            sel_br_p0, sel_br_p1);
        // Wait for Br ds_reads, extract
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        fp4_intx8_t tBr_s12[4];
        extract_tile(br_d, tBr_s12);
        // 4 prefetches: A0_db + Bl_db (SAFE — A0/Bl already in regs, no LDS race)
        emit_one_pf(pf_a0_p, 0);
        emit_one_pf(pf_a0_p, 1);
        emit_one_pf(pf_bl_p, 0);
        emit_one_pf(pf_bl_p, 1);
        asm volatile("" ::: "memory");
        // Step2: A0*Br (32 MFMAs) + ds_read A1
        kpair_32mfma_with_lds(acc_A0Br, tA0, tBr_s12, a0_raw, br_raw,
            a1_d[0], a1_d[1], a1_d[2], a1_d[3],
            a1_d[4], a1_d[5], a1_d[6], a1_d[7],
            sel_a1_p0, sel_a1_p1);
#else
        kpair_64mfma_step12(acc_A0Bl, acc_A0Br, tA0, tBl,
            a0_raw, bl_raw, br_raw, br_d, a1_d,
            sel_br_p0, sel_br_p1, sel_a1_p0, sel_a1_p1);
#endif

        asm volatile("s_waitcnt lgkmcnt(0)");
        fp4_intx8_t tBr[4], tA1[4];
#if STEP12_SPLIT_PF
        tBr[0] = tBr_s12[0]; tBr[1] = tBr_s12[1]; tBr[2] = tBr_s12[2]; tBr[3] = tBr_s12[3];
#else
        extract_tile(br_d, tBr);
#endif
        extract_tile(a1_d, tA1);

#if FUSED_STEP34
        // Fused Step34: barrier + 64 MFMAs + 16 ds_reads in one asm block
        float4 nxt_a0_d[8];
        // (nxt_bl_d already declared above)
#if STEP12_PF_INTERLEAVE
        // R68: prefetches already issued in step12. step34 base path, no tail.
        kpair_64mfma_step34(acc_A1Bl, acc_A1Br, tA1, tBl, tBr,
            a1_raw, bl_raw, br_raw, nxt_a0_d, nxt_bl_d,
            sel_a0_p0, sel_a0_p1, sel_bl_p0, sel_bl_p1);
#elif STEP34_PF_INTERLEAVE
        // R66 axis-A: prefetches inlined into the asm block, no post-tail needed.
        // When STEP12_SPLIT_PF=1, the 4 extra A0/Bl loads from step12 split are
        // harmless duplicates — step34pf still emits all 16 prefetches.
        kpair_64mfma_step34_pf_interleaved(acc_A1Bl, acc_A1Br, tA1, tBl, tBr,
            a1_raw, bl_raw, br_raw, nxt_a0_d, nxt_bl_d,
            sel_a0_p0, sel_a0_p1, sel_bl_p0, sel_bl_p1,
            pf_a0_p, pf_a1_p, pf_bl_p, pf_br_p);
#else
        kpair_64mfma_step34(acc_A1Bl, acc_A1Br, tA1, tBl, tBr,
            a1_raw, bl_raw, br_raw, nxt_a0_d, nxt_bl_d,
            sel_a0_p0, sel_a0_p1, sel_bl_p0, sel_bl_p1);
        // All prefetches emitted after the fused block
        emit_pf_tail<0>(pf_a0_p, pf_a1_p);
        emit_pf_tail<0>(pf_bl_p, pf_br_p);
#endif
#elif R37_FIX_B
        // R37 Fix B (default, no-TAIL_SPLIT): use fused step3+step4 (correctness
        // fix). The no-TAIL_SPLIT path has no R25-C tail-pf-off branching, so we
        // unconditionally emit all prefetches after the fused block (matches the
        // FUSED_STEP34=1 emission shape).
        float4 nxt_a0_d[8];
        // (nxt_bl_d already declared above)
        kpair_64mfma_step34(acc_A1Bl, acc_A1Br, tA1, tBl, tBr,
            a1_raw, bl_raw, br_raw, nxt_a0_d, nxt_bl_d,
            sel_a0_p0, sel_a0_p1, sel_bl_p0, sel_bl_p1);
        emit_pf_tail<0>(pf_a0_p, pf_a1_p);
        emit_pf_tail<0>(pf_bl_p, pf_br_p);
#endif // FUSED_STEP34 / R37_FIX_B
        asm volatile("s_waitcnt lgkmcnt(0)");
        extract_tile(nxt_a0_d, tA0);
        extract_tile(nxt_bl_d, tBl);
        // R21B/R22C: opt-in scheduling hooks (no-op at defaults). Site 2.
    }

#endif // TAIL_SPLIT

    // ═══════════ Store C -- streamlined direct store ═══════════
    // R21B: optionally drop wave priority before the store epilogue.
    // R22C: optional sched_group_barrier just before Store-C (site bit3).
    // Process base tiles directly from accumulators without materializing RT_C.
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
                store_bf16_val(&dst_ptr[(row_base + 0) * row_stride + col], s[0]);
                store_bf16_val(&dst_ptr[(row_base + 1) * row_stride + col], s[1]);
                store_bf16_val(&dst_ptr[(row_base + 2) * row_stride + col], s[2]);
                store_bf16_val(&dst_ptr[(row_base + 3) * row_stride + col], s[3]);
            }
        }
    };
    store_block(acc_A0Bl, 0, 0);
    store_block(acc_A0Br, 0, 1);
    store_block(acc_A1Bl, 1, 0);
    store_block(acc_A1Br, 1, 1);
    } // end static-dispatch block
}

void dispatch_gluon_cpp(gluon_globals g) {
    int m = static_cast<int>(g.c.rows());
    int n = static_cast<int>(g.c.cols());
    const dim3 grid((m / BLK) * (n / BLK));
    mxfp4_gluon_cpp_kernel<<<grid, dim3(_NUM_THREADS), 0>>>(g);
}

PYBIND11_MODULE(tk_mxfp4_gluon_cpp, m) {
    m.doc() = "MXFP4 Gluon-arch kernel (C++ reimplementation)";
    py::bind_function<dispatch_gluon_cpp>(m, "gemm_rcr",
        &gluon_globals::a, &gluon_globals::b,
        &gluon_globals::a_scale, &gluon_globals::b_scale,
        &gluon_globals::c);
}

