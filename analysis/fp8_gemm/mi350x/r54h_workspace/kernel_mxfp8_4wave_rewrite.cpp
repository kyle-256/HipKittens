// MXFP8 4-wave RCR kernel: pure-intrinsic MFMAs for C++ → .s → Python pipeline
//
// Based on rcr_mxfp8_4wave_fastpath.inc but with all MFMA inline ASM
// replaced by __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4.
// ds_read_b128 inline ASM is kept (simple, non-buggy).
// Buffer-load SRD for scales is kept (already intrinsic-based).
//
// Build (normal .so):
//   THUNDERKITTENS_ROOT=$(git rev-parse --show-toplevel) ROCM_PATH=/opt/rocm \
//     CPPFLAGS='-DM_DIM=8192 -DN_DIM=8192 -DK_DIM=8192' \
//     make -B TARGET=tk_mxfp8_rewrite SRC=kernel_mxfp8_4wave_rewrite.cpp
//
// Build (device .s for Python post-processing):
//   make -B tk_mxfp8_rewrite_device.s

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

// ════════════════════════ Constants & Types ════════════════════════

constexpr int BLK = 256;
constexpr int BK  = 128;
constexpr int WARPS_ROW = 2, WARPS_COL = 2;
constexpr int NUM_WARPS = WARPS_ROW * WARPS_COL;
constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
constexpr int HB = BLK / 2;    // 128
constexpr int RBM = HB / WARPS_ROW;  // 64
constexpr int RBN = HB / WARPS_COL;  // 64

constexpr int GROUP_M = 2;
constexpr int MY_NUM_XCDS = 8;

using G4 = kittens::group<NUM_WARPS>;
using ST_A = st_fp8e4m3<HB, BK, st_16x128_s>;
using ST_B = st_fp8e4m3<HB, BK, st_16x128_s>;
using RT_A = rt_fp8e4m3<RBM, BK, row_l, rt_16x128_s>;
using RT_B = rt_fp8e4m3<RBN, BK, row_l, rt_16x128_s>;
using RT_C = rt_fl<RBM, RBN, col_l, rt_16x16_s>;

using _gl_fp8   = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_scale = gl<fp8e8m0, -1, -1, -1, -1>;
using _gl_bf16  = gl<bf16, -1, -1, -1, -1>;

struct rewrite_globals {
    _gl_fp8 a, b;
    _gl_scale a_scale, b_scale;
    _gl_bf16 c;
    float scale = 1.0f;
    hipStream_t stream = nullptr;
    int m = 0, n = 0, k = 0;
};

static_assert(BLK == 256 && BK == 128, "Requires BLK=256, BK=128");
static_assert(RBM == 64 && RBN == 64, "Expected 64x64 per-warp tile");

// ════════════════════════ Vector Types ════════════════════════

using intx8_t   = int __attribute__((__vector_size__(8 * sizeof(int))));
using floatx4_t = float __attribute__((__vector_size__(4 * sizeof(float))));

// ════════════════════════ Clean MFMA Helper ════════════════════════
//
// Replaces the entire ACC16/EMIT_PHASE/if-constexpr dispatch tree.
// One function, one builtin call, no AGPR management, no macros.

template<int K_PHASE, int N, int M>
__device__ __forceinline__ void mxfp8_mfma(
    RT_C& acc, const RT_A& a, const RT_B& b,
    const fp8e8m0_4 (&a_scale)[RBM / 32],
    const fp8e8m0_4 (&b_scale)[(RBN + 31) / 32])
{
    floatx4_t* dp = (floatx4_t*)&acc.tiles[N][M].data[0];
    *dp = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        *(const intx8_t*)&a.tiles[N][0].data[0],
        *(const intx8_t*)&b.tiles[M][0].data[0],
        *dp,
        0, 0,  // cbsz=0, blgp=0 (FP8e4m3)
        (N & 1) | (K_PHASE << 1), a_scale[N / 2],
        (M & 1) | (K_PHASE << 1), b_scale[M / 2]);
}

template<int K_PHASE>
__device__ __forceinline__ void mxfp8_mfma_tile(
    RT_C& acc, const RT_A& a, const RT_B& b,
    const fp8e8m0_4 (&a_scale)[RBM / 32],
    const fp8e8m0_4 (&b_scale)[(RBN + 31) / 32])
{
    mxfp8_mfma<K_PHASE, 0, 0>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<K_PHASE, 0, 1>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<K_PHASE, 0, 2>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<K_PHASE, 0, 3>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<K_PHASE, 1, 0>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<K_PHASE, 1, 1>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<K_PHASE, 1, 2>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<K_PHASE, 1, 3>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<K_PHASE, 2, 0>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<K_PHASE, 2, 1>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<K_PHASE, 2, 2>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<K_PHASE, 2, 3>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<K_PHASE, 3, 0>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<K_PHASE, 3, 1>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<K_PHASE, 3, 2>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<K_PHASE, 3, 3>(acc, a, b, a_scale, b_scale);
}

// KPAIR: both k_phases per (N,M) back-to-back → same AGPR stays active
__device__ __forceinline__ void mxfp8_mfma_tile_kpair(
    RT_C& acc, const RT_A& a, const RT_B& b,
    const fp8e8m0_4 (&a_scale)[RBM / 32],
    const fp8e8m0_4 (&b_scale)[(RBN + 31) / 32])
{
    mxfp8_mfma<0, 0, 0>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<1, 0, 0>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<0, 0, 1>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<1, 0, 1>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<0, 0, 2>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<1, 0, 2>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<0, 0, 3>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<1, 0, 3>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<0, 1, 0>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<1, 1, 0>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<0, 1, 1>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<1, 1, 1>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<0, 1, 2>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<1, 1, 2>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<0, 1, 3>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<1, 1, 3>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<0, 2, 0>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<1, 2, 0>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<0, 2, 1>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<1, 2, 1>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<0, 2, 2>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<1, 2, 2>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<0, 2, 3>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<1, 2, 3>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<0, 3, 0>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<1, 3, 0>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<0, 3, 1>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<1, 3, 1>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<0, 3, 2>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<1, 3, 2>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<0, 3, 3>(acc, a, b, a_scale, b_scale);
    mxfp8_mfma<1, 3, 3>(acc, a, b, a_scale, b_scale);
}

// ════════════════════════ LDS → Register Load ════════════════════════

template<ducks::rt::row_layout RT, ducks::st::all ST>
__device__ __forceinline__ void load_st_to_rt(RT &dst, const ST &src) {
    static_assert(RT::rows == ST::rows && RT::cols == ST::cols);
    using T = typename base_types::packing<typename RT::dtype>::unpacked_type;
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

    #pragma unroll
    for (int k = 0; k < RT::base_tile_num_strides; k++) {
        #pragma unroll
        for (int i = 0; i < reg_sub_col; i++) {
            #pragma unroll
            for (int j = 0; j < reg_sub_row; j++) {
                const int row = i * RT::base_tile_rows + row_offset;
                const int col = j * RT::base_tile_cols + col_offset +
                                k * RT::base_tile_elements_per_stride_group;
                const uint32_t offset =
                    sizeof(U) * (src_ptr + row * ST::underlying_subtile_cols + col);
                const uint32_t addr = offset ^ (((offset % (16 * 128)) >> 8) << 4);
                const int idx = k * RT::base_tile_stride / packing;

                #pragma unroll
                for (int ii = 0; ii < ST::subtiles_per_col; ii++) {
                    #pragma unroll
                    for (int jj = 0; jj < ST::subtiles_per_row; jj++) {
                        const int shared_offset =
                            (ii * ST::underlying_subtiles_per_row + jj) *
                            ST::underlying_subtile_bytes;
                        const int rr = ii * reg_sub_col + i;
                        const int rc = jj * reg_sub_row + j;

                        asm volatile(
                            "ds_read_b128 %0, %1 offset:%2\n"
                            : "=v"(*reinterpret_cast<float4*>(
                                  &dst.tiles[rr][rc].data[idx]))
                            : "v"(addr), "i"(shared_offset)
                            : "memory"
                        );
                    }
                }
            }
        }
    }
}

// Individual ds_read for interleaved scheduling
template<int NUM_OFFSETS, typename RT, typename ST>
__device__ __forceinline__ void prefill_swizzled_offsets(
    RT& dst, ST& src, uint32_t* swizzled_offsets)
{
    static_assert(NUM_OFFSETS == RT::base_tile_num_strides);
    using U = typename ST::dtype;
    constexpr int subtile_stride = RT::base_tile_cols * sizeof(U) / 2;
    const uint32_t st_offset =
        (kittens::laneid() % RT::base_tile_rows) * ST::underlying_cols +
        (kittens::laneid() / RT::base_tile_rows * 16 / sizeof(U));
    const uint32_t base_addr = reinterpret_cast<uintptr_t>(&src.data[st_offset]);

    swizzled_offsets[0] = base_addr;
    swizzled_offsets[0] ^= (((swizzled_offsets[0] % (256 * 8)) >> 8) << 4);
    swizzled_offsets[1] = base_addr + subtile_stride;
    swizzled_offsets[1] ^= (((swizzled_offsets[1] % (256 * 8)) >> 8) << 4);
}

template<int REGISTER_ROW, int REGISTER_COL, int K_STEP, typename RT, typename ST>
__device__ __forceinline__ void load_one_s2r(
    RT& dst, ST& src, uint32_t* swizzled_offsets)
{
    using U = typename ST::dtype;
    constexpr int packing = base_types::packing<typename RT::dtype>::num();
    const int idx = K_STEP * RT::base_tile_stride / packing;
    constexpr int row_stride = RT::base_tile_rows * ST::underlying_cols * sizeof(U);
    asm volatile(
        "ds_read_b128 %0, %1 offset:%2\n"
        : "=v"(*reinterpret_cast<float4*>(
              &dst.tiles[REGISTER_ROW][REGISTER_COL].data[idx]))
        : "v"(swizzled_offsets[K_STEP]), "i"(REGISTER_ROW * row_stride)
        : "memory"
    );
}

// ════════════════════════ Tile Prefetch (Global → LDS) ════════════════════════

struct precomputed_addresses {
    i32x4 srsrc;
    uintptr_t lds_base;
};

template<typename ST, typename GL>
__device__ __forceinline__ precomputed_addresses precompute_addresses(
    ST& dst, const GL& src, const coord<ST>& idx)
{
    constexpr int axis = 2;
    using T = typename ST::dtype;
    const int row_stride = src.template stride<axis>();
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* global_ptr = (T*)&src[unit_coord];
    i32x4 srsrc = make_srsrc(global_ptr, row_stride * ST::rows * sizeof(T));

    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    uintptr_t lds_base =
        reinterpret_cast<uintptr_t>(&dst.data[0]) + (warpid() * bytes_per_warp);
    return {srsrc, lds_base};
}

template<int I, typename ST, typename GL>
__device__ __forceinline__ void load_one_g2s(
    ST& dst, const GL& src, const precomputed_addresses& addr)
{
    constexpr int axis = 2;
    using T = typename ST::dtype;
    constexpr int bpt = ST::underlying_subtile_bytes_per_thread;
    constexpr int bpw = bpt * kittens::WARP_THREADS;

    const int lane = kittens::laneid();
    const int warp = kittens::warpid() % NUM_WARPS;
    const int row_stride = src.template stride<axis>();

    const int lane_byte_offset = (lane * bpt) + (warp * bpw) + (I * NUM_WARPS * bpw);
    const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
    const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
    const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
    const int subtile_lane_byte_offset = lane_byte_offset % ST::underlying_subtile_bytes;

    const int row = subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
    const int col = (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);
    const uint32_t swizzled_shared = dst.swizzle({row, col});
    const int swizzled_global_row =
        (swizzled_shared / ST::underlying_subtile_row_bytes) +
        subtile_row * ST::underlying_subtile_rows;
    const int swizzled_global_col =
        (swizzled_shared % ST::underlying_subtile_row_bytes) / sizeof(T) +
        subtile_col * ST::underlying_subtile_cols;
    const uint32_t swizzled_global_byte_offset =
        (swizzled_global_row * row_stride + swizzled_global_col) * sizeof(T);

    uintptr_t lds_addr = addr.lds_base + (I * NUM_WARPS * bpw);
    as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

    llvm_amdgcn_raw_buffer_load_lds(
        addr.srsrc, lds_ptr, bpt,
        swizzled_global_byte_offset, 0, 0,
        static_cast<int>(coherency::cache_all));
}

// ════════════════════════ Scale Loading (SRD) ════════════════════════

__device__ __forceinline__ const uint8_t* preshuffled_scale_row_base(
    const _gl_scale& src, int row_group)
{
    const int index = src.idx(coord<>(row_group, 0));
    return reinterpret_cast<const uint8_t*>(src.raw_ptr + index);
}

__device__ __forceinline__ i32x4 make_scale_srd(const void* ptr) {
    i32x4 srd = std::bit_cast<i32x4>(make_buffer_resource(
        static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(ptr)),
        0xFFFFFFFFu, 0x00110000u));
    srd[0] = __builtin_amdgcn_readfirstlane(srd[0]);
    srd[1] = __builtin_amdgcn_readfirstlane(srd[1]);
    srd[2] = __builtin_amdgcn_readfirstlane(srd[2]);
    srd[3] = __builtin_amdgcn_readfirstlane(srd[3]);
    return srd;
}

// ════════════════════════ Interleaved Compute + Prefetch ════════════════════════
//
// Same logical structure as do_interleaved_cluster_scaled but with
// clean builtin MFMA calls instead of the ACC16 macro system.

template<int K_PHASE, typename ST_GL, typename GL_GL, typename ST, typename RT,
         typename RT_A_T, typename RT_B_T, typename RT_C_T,
         ducks::coord::tile COORD = coord<ST_GL>>
__device__ __forceinline__ void compute_interleaved(
    ST_GL& dst_gl, const GL_GL& src_gl, COORD idx,
    RT& dst, ST& src, RT_A_T& a, RT_B_T& b, RT_C_T& c,
    const fp8e8m0_4 (&a_scale)[RBM / 32],
    const fp8e8m0_4 (&b_scale)[(RBN + 31) / 32])
{
    mxfp8_mfma<K_PHASE, 0, 0>(c, a, b, a_scale, b_scale);

    precomputed_addresses addresses = precompute_addresses(dst_gl, src_gl, idx);
    uint32_t swizzled_offsets[2];
    prefill_swizzled_offsets<2>(dst, src, swizzled_offsets);

    mxfp8_mfma<K_PHASE, 0, 1>(c, a, b, a_scale, b_scale);

    load_one_g2s<0>(dst_gl, src_gl, addresses);
    load_one_s2r<0, 0, 0>(dst, src, swizzled_offsets);

    mxfp8_mfma<K_PHASE, 0, 2>(c, a, b, a_scale, b_scale);

    load_one_s2r<0, 0, 1>(dst, src, swizzled_offsets);

    mxfp8_mfma<K_PHASE, 0, 3>(c, a, b, a_scale, b_scale);

    load_one_g2s<1>(dst_gl, src_gl, addresses);
    load_one_s2r<1, 0, 0>(dst, src, swizzled_offsets);

    mxfp8_mfma<K_PHASE, 1, 0>(c, a, b, a_scale, b_scale);
    mxfp8_mfma<K_PHASE, 1, 1>(c, a, b, a_scale, b_scale);

    load_one_s2r<1, 0, 1>(dst, src, swizzled_offsets);

    mxfp8_mfma<K_PHASE, 1, 2>(c, a, b, a_scale, b_scale);
    mxfp8_mfma<K_PHASE, 1, 3>(c, a, b, a_scale, b_scale);

    load_one_g2s<2>(dst_gl, src_gl, addresses);
    load_one_s2r<2, 0, 0>(dst, src, swizzled_offsets);

    mxfp8_mfma<K_PHASE, 2, 0>(c, a, b, a_scale, b_scale);
    mxfp8_mfma<K_PHASE, 2, 1>(c, a, b, a_scale, b_scale);

    load_one_s2r<2, 0, 1>(dst, src, swizzled_offsets);

    mxfp8_mfma<K_PHASE, 2, 2>(c, a, b, a_scale, b_scale);
    mxfp8_mfma<K_PHASE, 2, 3>(c, a, b, a_scale, b_scale);

    load_one_g2s<3>(dst_gl, src_gl, addresses);
    load_one_s2r<3, 0, 0>(dst, src, swizzled_offsets);

    mxfp8_mfma<K_PHASE, 3, 0>(c, a, b, a_scale, b_scale);
    mxfp8_mfma<K_PHASE, 3, 1>(c, a, b, a_scale, b_scale);

    load_one_s2r<3, 0, 1>(dst, src, swizzled_offsets);

    mxfp8_mfma<K_PHASE, 3, 2>(c, a, b, a_scale, b_scale);
    mxfp8_mfma<K_PHASE, 3, 3>(c, a, b, a_scale, b_scale);
}

// ════════════════════════ Block Coordinate Swizzle ════════════════════════

__device__ __forceinline__ void compute_block_coords(int bid, int& br, int& bc) {
    int wgid = bid;
    const int num_wgs = gridDim.x;
    if (num_wgs >= MY_NUM_XCDS && (num_wgs % MY_NUM_XCDS) == 0) {
        wgid = (wgid % MY_NUM_XCDS) * (num_wgs / MY_NUM_XCDS) + (wgid / MY_NUM_XCDS);
    }
    constexpr int num_pid_m = M_DIM / BLK;
    constexpr int num_pid_n = N_DIM / BLK;
    const int num_wgid_in_group = GROUP_M * num_pid_n;
    const int group_id = wgid / num_wgid_in_group;
    const int first_pid_m = group_id * GROUP_M;
    const int group_size_m =
        (first_pid_m + GROUP_M <= num_pid_m) ? GROUP_M : (num_pid_m - first_pid_m);
    br = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    bc = (wgid % num_wgid_in_group) / group_size_m;
}

// ════════════════════════ Main Kernel ════════════════════════

__global__ __launch_bounds__(NUM_THREADS, 1)
void mxfp8_rewrite_kernel(const rewrite_globals g) {
    constexpr int k_iters = K_DIM / BK;
    constexpr int a_pack_count = RBM / 32;
    constexpr int b_pack_count = (RBN + 31) / 32;

    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];

    RT_A a[2];
    RT_B b[2];
    RT_C c[2][2];

    const int bid = blockIdx.x;
    int br, bc;
    compute_block_coords(bid, br, bc);

    const int warp_m = warpid() / WARPS_COL;
    const int warp_n = warpid() % WARPS_COL;
    const int lane_nonk = kittens::laneid() % 16;
    const int lane_kblk = kittens::laneid() / 16;
    const uint32_t lane_scale_byte_offset =
        (static_cast<uint32_t>(lane_kblk) << 6) |
        (static_cast<uint32_t>(lane_nonk) << 2);

    auto scale_a_base = [&](int half) {
        return br * BLK + half * HB + warp_m * RBM;
    };
    auto scale_b_base = [&](int half) {
        return bc * BLK + half * HB + warp_n * RBN;
    };

    // Scale SRD setup
    fp8e8m0_4 a0_scale[a_pack_count], a1_scale[a_pack_count];
    fp8e8m0_4 b0_scale[b_pack_count], b1_scale[b_pack_count];

    i32x4 a0_srd[a_pack_count], a1_srd[a_pack_count];
    i32x4 b0_srd[b_pack_count], b1_srd[b_pack_count];

    #pragma unroll
    for (int p = 0; p < a_pack_count; ++p) {
        a0_srd[p] = make_scale_srd(preshuffled_scale_row_base(
            g.a_scale, (scale_a_base(0) + p * 32) >> 5));
        a1_srd[p] = make_scale_srd(preshuffled_scale_row_base(
            g.a_scale, (scale_a_base(1) + p * 32) >> 5));
    }
    #pragma unroll
    for (int p = 0; p < b_pack_count; ++p) {
        b0_srd[p] = make_scale_srd(preshuffled_scale_row_base(
            g.b_scale, (scale_b_base(0) + p * 32) >> 5));
        b1_srd[p] = make_scale_srd(preshuffled_scale_row_base(
            g.b_scale, (scale_b_base(1) + p * 32) >> 5));
    }

    auto load_scales = [&](int k_pair) __attribute__((always_inline)) {
        const uint32_t soff = static_cast<uint32_t>(k_pair) << 8;
        #pragma unroll
        for (int p = 0; p < a_pack_count; ++p) {
            a0_scale[p] = std::bit_cast<fp8e8m0_4>(
                llvm_amdgcn_raw_buffer_load_b32(a0_srd[p], lane_scale_byte_offset, soff, 0));
            a1_scale[p] = std::bit_cast<fp8e8m0_4>(
                llvm_amdgcn_raw_buffer_load_b32(a1_srd[p], lane_scale_byte_offset, soff, 0));
        }
        #pragma unroll
        for (int p = 0; p < b_pack_count; ++p) {
            b0_scale[p] = std::bit_cast<fp8e8m0_4>(
                llvm_amdgcn_raw_buffer_load_b32(b0_srd[p], lane_scale_byte_offset, soff, 0));
            b1_scale[p] = std::bit_cast<fp8e8m0_4>(
                llvm_amdgcn_raw_buffer_load_b32(b1_srd[p], lane_scale_byte_offset, soff, 0));
        }
    };

    // ──── Prologue: load first two iterations' tiles ────

    int curr = 0, next = 1;

    G4::load(As[curr][0], g.a, {0, 0, br * WARPS_ROW, 0});
    G4::load(Bs[curr][0], g.b, {0, 0, bc * WARPS_COL, 0});
    G4::load(Bs[curr][1], g.b, {0, 0, bc * WARPS_COL + 1, 0});
    G4::load(As[curr][1], g.a, {0, 0, br * WARPS_ROW + 1, 0});

    zero(c[0][0]); zero(c[0][1]); zero(c[1][0]); zero(c[1][1]);

    G4::load(As[next][0], g.a, {0, 0, br * WARPS_ROW, 1});
    G4::load(Bs[next][0], g.b, {0, 0, bc * WARPS_COL, 1});
    G4::load(Bs[next][1], g.b, {0, 0, bc * WARPS_COL + 1, 1});
    G4::load(As[next][1], g.a, {0, 0, br * WARPS_ROW + 1, 1});

    asm volatile("s_waitcnt vmcnt(28)");
    __builtin_amdgcn_s_barrier();

    auto a_sub_0 = kittens::subtile_inplace<RBM, BK>(As[curr][0], {warp_m, 0});
    load_st_to_rt(a[0], a_sub_0);

    asm volatile("s_waitcnt vmcnt(24)");
    __builtin_amdgcn_s_barrier();

    auto b_sub_0 = kittens::subtile_inplace<RBN, BK>(Bs[curr][0], {warp_n, 0});
    load_st_to_rt(b[0], b_sub_0);

    load_scales(0);

    // ──── Main loop ────

    #pragma unroll 2
    for (int k = 0; k < k_iters - 2; ++k, curr ^= 1, next ^= 1) {
        const int k_pair = k >> 1;
        const int k_phase = k & 1;

        asm volatile("s_waitcnt vmcnt(16)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        auto b_sub_1 = kittens::subtile_inplace<RBN, BK>(Bs[curr][1], {warp_n, 0});
        if (k_phase == 0) {
            compute_interleaved<0>(
                As[curr][0], g.a, coord<ST_A>{0, 0, br * WARPS_ROW, k + 2},
                b[1], b_sub_1, a[0], b[0], c[0][0], a0_scale, b0_scale);
        } else {
            compute_interleaved<1>(
                As[curr][0], g.a, coord<ST_A>{0, 0, br * WARPS_ROW, k + 2},
                b[1], b_sub_1, a[0], b[0], c[0][0], a0_scale, b0_scale);
        }

        asm volatile("s_waitcnt lgkmcnt(0)");

        auto a_sub_1 = kittens::subtile_inplace<RBM, BK>(As[curr][1], {warp_m, 0});
        if (k_phase == 0) {
            compute_interleaved<0>(
                Bs[curr][0], g.b, coord<ST_B>{0, 0, bc * WARPS_COL, k + 2},
                a[1], a_sub_1, a[0], b[1], c[0][1], a0_scale, b1_scale);
        } else {
            compute_interleaved<1>(
                Bs[curr][0], g.b, coord<ST_B>{0, 0, bc * WARPS_COL, k + 2},
                a[1], a_sub_1, a[0], b[1], c[0][1], a0_scale, b1_scale);
        }

        asm volatile("s_waitcnt vmcnt(16)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        a_sub_0 = kittens::subtile_inplace<RBM, BK>(As[next][0], {warp_m, 0});
        if (k_phase == 0) {
            compute_interleaved<0>(
                Bs[curr][1], g.b, coord<ST_B>{0, 0, bc * WARPS_COL + 1, k + 2},
                a[0], a_sub_0, a[1], b[0], c[1][0], a1_scale, b0_scale);
        } else {
            compute_interleaved<1>(
                Bs[curr][1], g.b, coord<ST_B>{0, 0, bc * WARPS_COL + 1, k + 2},
                a[0], a_sub_0, a[1], b[0], c[1][0], a1_scale, b0_scale);
        }

        b_sub_0 = kittens::subtile_inplace<RBN, BK>(Bs[next][0], {warp_n, 0});
        if (k_phase == 0) {
            compute_interleaved<0>(
                As[curr][1], g.a, coord<ST_A>{0, 0, br * WARPS_ROW + 1, k + 2},
                b[0], b_sub_0, a[1], b[1], c[1][1], a1_scale, b1_scale);
        } else {
            compute_interleaved<1>(
                As[curr][1], g.a, coord<ST_A>{0, 0, br * WARPS_ROW + 1, k + 2},
                b[0], b_sub_0, a[1], b[1], c[1][1], a1_scale, b1_scale);
        }

        {
            const int next_k_pair = (k + 1) >> 1;
            if (next_k_pair != k_pair) {
                load_scales(next_k_pair);
            }
        }
    }

    // ──── Penultimate iteration: no prefetch, use kpair MFMAs ────
    {
        const int k = k_iters - 2;
        const int k_pair = k >> 1;
        load_scales(k_pair);

        asm volatile("s_waitcnt vmcnt(16)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        auto b_sub_1 = kittens::subtile_inplace<RBN, BK>(Bs[curr][1], {warp_n, 0});
        load_st_to_rt(b[1], b_sub_1);

        mxfp8_mfma_tile_kpair(c[0][0], a[0], b[0], a0_scale, b0_scale);

        asm volatile("s_waitcnt lgkmcnt(0)");

        auto a_sub_1 = kittens::subtile_inplace<RBM, BK>(As[curr][1], {warp_m, 0});
        load_st_to_rt(a[1], a_sub_1);

        mxfp8_mfma_tile_kpair(c[0][1], a[0], b[1], a0_scale, b1_scale);

        asm volatile("s_waitcnt vmcnt(8)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        a_sub_0 = kittens::subtile_inplace<RBM, BK>(As[next][0], {warp_m, 0});
        load_st_to_rt(a[0], a_sub_0);

        mxfp8_mfma_tile_kpair(c[1][0], a[1], b[0], a1_scale, b0_scale);

        b_sub_0 = kittens::subtile_inplace<RBN, BK>(Bs[next][0], {warp_n, 0});
        load_st_to_rt(b[0], b_sub_0);

        mxfp8_mfma_tile_kpair(c[1][1], a[1], b[1], a1_scale, b1_scale);

        curr ^= 1;
        next ^= 1;
    }

    // ──── Final iteration: drain pipeline ────
    {
        const int k_pair = (k_iters - 1) >> 1;
        load_scales(k_pair);

        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        auto b_sub_1 = kittens::subtile_inplace<RBN, BK>(Bs[curr][1], {warp_n, 0});
        load_st_to_rt(b[1], b_sub_1);

        mxfp8_mfma_tile_kpair(c[0][0], a[0], b[0], a0_scale, b0_scale);

        asm volatile("s_waitcnt lgkmcnt(0)");

        auto a_sub_1 = kittens::subtile_inplace<RBM, BK>(As[curr][1], {warp_m, 0});
        load_st_to_rt(a[1], a_sub_1);

        mxfp8_mfma_tile_kpair(c[0][1], a[0], b[1], a0_scale, b1_scale);

        asm volatile("s_waitcnt lgkmcnt(0)");

        mxfp8_mfma_tile_kpair(c[1][0], a[1], b[0], a1_scale, b0_scale);
        mxfp8_mfma_tile_kpair(c[1][1], a[1], b[1], a1_scale, b1_scale);
    }

    // ──── Epilogue: scale + store ────

    mul(c[0][0], c[0][0], g.scale);
    mul(c[0][1], c[0][1], g.scale);
    mul(c[1][0], c[1][0], g.scale);
    mul(c[1][1], c[1][1], g.scale);

    store(g.c, c[0][0], {0, 0, br * WARPS_ROW * 2 + warp_m,
                                bc * WARPS_COL * 2 + warp_n});
    store(g.c, c[0][1], {0, 0, br * WARPS_ROW * 2 + warp_m,
                                bc * WARPS_COL * 2 + WARPS_COL + warp_n});
    store(g.c, c[1][0], {0, 0, br * WARPS_ROW * 2 + WARPS_ROW + warp_m,
                                bc * WARPS_COL * 2 + warp_n});
    store(g.c, c[1][1], {0, 0, br * WARPS_ROW * 2 + WARPS_ROW + warp_m,
                                bc * WARPS_COL * 2 + WARPS_COL + warp_n});
}

// ════════════════════════ Host Dispatch ════════════════════════

void dispatch_rewrite(rewrite_globals g) {
    g.m = static_cast<int>(g.c.rows());
    g.n = static_cast<int>(g.c.cols());
    g.k = static_cast<int>(g.a.cols());
    const dim3 grid((g.m / BLK) * (g.n / BLK));
    mxfp8_rewrite_kernel<<<grid, dim3(NUM_THREADS), 0, g.stream>>>(g);
}

PYBIND11_MODULE(tk_mxfp8_rewrite, m) {
    m.doc() = "MXFP8 4-wave RCR kernel with pure-intrinsic MFMAs";
    py::bind_function<dispatch_rewrite>(m, "gemm_rcr_pq",
        &rewrite_globals::a, &rewrite_globals::b,
        &rewrite_globals::a_scale, &rewrite_globals::b_scale,
        &rewrite_globals::c);
}
