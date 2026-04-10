#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

#define TK_STRINGIFY_IMPL(x) #x
#define TK_STRINGIFY(x) TK_STRINGIFY_IMPL(x)
#define TK_WAIT_LGKM(x) asm volatile("s_waitcnt lgkmcnt(" TK_STRINGIFY(x) ")")
#define TK_WAIT_VMCNT(x) asm volatile("s_waitcnt vmcnt(" TK_STRINGIFY(x) ")")

#ifndef GEMM_BLOCK_SIZE
#define GEMM_BLOCK_SIZE 256
#endif
#ifndef GEMM_K_BLOCK
#define GEMM_K_BLOCK 128
#endif

constexpr int BLK = GEMM_BLOCK_SIZE, BK = GEMM_K_BLOCK;
constexpr int HB  = BLK / 2;

#ifndef RCR_4WAVE_GROUP_M
#define RCR_4WAVE_GROUP_M 4
#endif
#ifndef RCR_4WAVE_NUM_XCDS
#define RCR_4WAVE_NUM_XCDS 8
#endif
#ifndef RCR_4WAVE_ENABLE_XCD_SWIZZLE
#define RCR_4WAVE_ENABLE_XCD_SWIZZLE 1
#endif

namespace rcr_4wave_kspec {

constexpr int WARPS_ROW = 2;
constexpr int WARPS_COL = 2;
constexpr int NUM_WARPS = WARPS_ROW * WARPS_COL;
constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
constexpr int WGM = RCR_4WAVE_GROUP_M;
constexpr int NUM_XCDS = RCR_4WAVE_NUM_XCDS;
constexpr int K_ITERS = K_DIM / BK;

using G4 = kittens::group<NUM_WARPS>;

using a_gl = gl<fp8e4m3, 1, 1, -1, K_DIM>;
using b_gl = gl<fp8e4m3, 1, 1, -1, K_DIM>;
using c_gl = gl<bf16, 1, 1, -1, -1>;

using ST_A = st_fp8e4m3<BLK / 2, BK, st_16x128_s>;
using ST_B = st_fp8e4m3<BLK / 2, BK, st_16x128_s>;
using RT_A = rt_fp8e4m3<BLK / 2 / WARPS_ROW, BK, row_l, rt_16x128_s>;
using RT_B = rt_fp8e4m3<BLK / 2 / WARPS_COL, BK, row_l, rt_16x128_s>;
using RT_C = rt_fl<BLK / 2 / WARPS_ROW, BLK / 2 / WARPS_COL, col_l, rt_16x16_s>;

template<ducks::rt::row_layout RT, ducks::st::all ST>
__device__ __forceinline__ void load_st_to_rt(RT &dst, const ST &src) {
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
    constexpr int rps_row = ST::underlying_subtile_cols / RT::base_tile_cols;
    constexpr int rps_col = ST::underlying_subtile_rows / RT::base_tile_rows;

    #pragma unroll
    for (int k = 0; k < RT::base_tile_num_strides; k++) {
        #pragma unroll
        for (int i = 0; i < rps_col; i++) {
            #pragma unroll
            for (int j = 0; j < rps_row; j++) {
                const int row = i * RT::base_tile_rows + row_offset;
                const int col = j * RT::base_tile_cols + col_offset +
                    k * RT::base_tile_elements_per_stride_group;
                const uint32_t offset = sizeof(U) * (src_ptr + row * ST::underlying_subtile_cols + col);
                const uint32_t addr = offset ^ (((offset % (16 * 128)) >> 8) << 4);
                const int idx = k * RT::base_tile_stride / packing;
                #pragma unroll
                for (int ii = 0; ii < ST::subtiles_per_col; ii++) {
                    #pragma unroll
                    for (int jj = 0; jj < ST::subtiles_per_row; jj++) {
                        const int sid = ii * ST::underlying_subtiles_per_row + jj;
                        const int soff = sid * ST::underlying_subtile_bytes;
                        const int rr = ii * rps_col + i;
                        const int rc = jj * rps_row + j;
                        if constexpr (std::is_same_v<U2, fp8e4m3_4>) {
                            static_assert(RT::base_tile_stride == 16);
                            asm volatile("ds_read_b128 %0, %1 offset:%2\n"
                                : "=v"(*reinterpret_cast<float4*>(&dst.tiles[rr][rc].data[idx]))
                                : "v"(addr), "i"(soff) : "memory");
                        }
                    }
                }
            }
        }
    }
}

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
    constexpr int bpw = ST::underlying_subtile_bytes_per_thread * kittens::WARP_THREADS;
    uintptr_t lds_base = reinterpret_cast<uintptr_t>(&dst.data[0]) + (warpid() * bpw);
    return {srsrc, lds_base};
}

template<int I, typename ST, typename GL>
__device__ __forceinline__ void load_one_g2s(
    ST& dst, const GL& src, const precomputed_addresses& addresses)
{
    constexpr int axis = 2;
    using T = typename ST::dtype;
    constexpr int bpt = ST::underlying_subtile_bytes_per_thread;
    constexpr int bpw = bpt * kittens::WARP_THREADS;
    const int lane = kittens::laneid();
    const int warp = kittens::warpid() % NUM_WARPS;
    const int row_stride = src.template stride<axis>();
    const int lbo = (lane * bpt) + (warp * bpw) + (I * NUM_WARPS * bpw);
    const int sid = lbo / ST::underlying_subtile_bytes;
    const int sr = sid / ST::underlying_subtiles_per_row;
    const int sc = sid % ST::underlying_subtiles_per_row;
    const int slbo = lbo % ST::underlying_subtile_bytes;
    const int row = slbo / ST::underlying_subtile_row_bytes;
    const int col = (slbo % ST::underlying_subtile_row_bytes) / sizeof(T);
    const uint32_t ssbo = dst.swizzle({row, col});
    const int sgr = (ssbo / ST::underlying_subtile_row_bytes) + sr * ST::underlying_subtile_rows;
    const int sgc = (ssbo % ST::underlying_subtile_row_bytes) / sizeof(T) + sc * ST::underlying_subtile_cols;
    const uint32_t sgbo = (sgr * row_stride + sgc) * sizeof(T);
    uintptr_t lds_addr = addresses.lds_base + (I * NUM_WARPS * bpw);
    as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);
    llvm_amdgcn_raw_buffer_load_lds(addresses.srsrc, lds_ptr, bpt, sgbo, 0, 0,
        static_cast<int>(coherency::cache_all));
}

template<int RR, int RC, int KS, typename RT, typename ST>
__device__ __forceinline__ void load_one_s2r(RT& dst, ST& src, uint32_t* so) {
    using U = typename ST::dtype;
    constexpr int packing = base_types::packing<typename RT::dtype>::num();
    const int idx = KS * RT::base_tile_stride / packing;
    constexpr int rs = RT::base_tile_rows * ST::underlying_cols * sizeof(U);
    asm volatile("ds_read_b128 %0, %1 offset:%2\n"
        : "=v"(*reinterpret_cast<float4*>(&dst.tiles[RR][RC].data[idx]))
        : "v"(so[KS]), "i"(RR * rs) : "memory");
}

template<int NUM_OFFSETS, typename RT, typename ST>
__device__ __forceinline__ void prefill_swizzled_offsets(RT& dst, ST& src, uint32_t* so) {
    using U = typename ST::dtype;
    constexpr int ss = RT::base_tile_cols * sizeof(U) / 2;
    const uint32_t sto = (kittens::laneid() % RT::base_tile_rows) * ST::underlying_cols +
        (kittens::laneid() / RT::base_tile_rows * 16 / sizeof(U));
    const uint32_t ba = reinterpret_cast<uintptr_t>(&src.data[sto]);
    so[0] = ba; so[0] ^= (((so[0] % (256 * 8)) >> 8) << 4);
    so[1] = ba + ss; so[1] ^= (((so[1] % (256 * 8)) >> 8) << 4);
}

template<typename D, typename A, typename B, typename C>
__device__ __forceinline__ void mma_abt_one(D& d, const A& a, const B& b, const C& c, int n, int m, int k) {
    mma_ABt_base(d.tiles[n][m], a.tiles[n][k], b.tiles[m][k], c.tiles[n][m]);
}

template<typename ST_GL, typename GL_GL, typename ST, typename RT,
         typename RT_A, typename RT_B, typename RT_C,
         ducks::coord::tile COORD = coord<ST_GL>>
__device__ __forceinline__ void do_interleaved_cluster(
    ST_GL& dst_gl, const GL_GL& src_gl, COORD idx,
    RT& dst, ST& src, RT_A& a, RT_B& b, RT_C& c)
{
    __builtin_amdgcn_sched_barrier(0); mma_abt_one(c, a, b, c, 0, 0, 0); __builtin_amdgcn_sched_barrier(0);
    precomputed_addresses addresses = precompute_addresses(dst_gl, src_gl, idx);
    uint32_t so[2]; prefill_swizzled_offsets<2>(dst, src, so);
    __builtin_amdgcn_sched_barrier(0); mma_abt_one(c, a, b, c, 0, 1, 0); __builtin_amdgcn_sched_barrier(0);
    load_one_g2s<0>(dst_gl, src_gl, addresses); load_one_s2r<0, 0, 0>(dst, src, so);
    __builtin_amdgcn_sched_barrier(0); mma_abt_one(c, a, b, c, 0, 2, 0); __builtin_amdgcn_sched_barrier(0);
    load_one_s2r<0, 0, 1>(dst, src, so);
    __builtin_amdgcn_sched_barrier(0); mma_abt_one(c, a, b, c, 0, 3, 0); __builtin_amdgcn_sched_barrier(0);
    load_one_g2s<1>(dst_gl, src_gl, addresses); load_one_s2r<1, 0, 0>(dst, src, so);
    __builtin_amdgcn_sched_barrier(0); mma_abt_one(c, a, b, c, 1, 0, 0); mma_abt_one(c, a, b, c, 1, 1, 0); __builtin_amdgcn_sched_barrier(0);
    load_one_s2r<1, 0, 1>(dst, src, so);
    __builtin_amdgcn_sched_barrier(0); mma_abt_one(c, a, b, c, 1, 2, 0); mma_abt_one(c, a, b, c, 1, 3, 0); __builtin_amdgcn_sched_barrier(0);
    load_one_g2s<2>(dst_gl, src_gl, addresses); load_one_s2r<2, 0, 0>(dst, src, so);
    __builtin_amdgcn_sched_barrier(0); mma_abt_one(c, a, b, c, 2, 0, 0); mma_abt_one(c, a, b, c, 2, 1, 0); __builtin_amdgcn_sched_barrier(0);
    load_one_s2r<2, 0, 1>(dst, src, so);
    __builtin_amdgcn_sched_barrier(0); mma_abt_one(c, a, b, c, 2, 2, 0); mma_abt_one(c, a, b, c, 2, 3, 0); __builtin_amdgcn_sched_barrier(0);
    load_one_g2s<3>(dst_gl, src_gl, addresses); load_one_s2r<3, 0, 0>(dst, src, so);
    __builtin_amdgcn_sched_barrier(0); mma_abt_one(c, a, b, c, 3, 0, 0); mma_abt_one(c, a, b, c, 3, 1, 0); __builtin_amdgcn_sched_barrier(0);
    load_one_s2r<3, 0, 1>(dst, src, so);
    __builtin_amdgcn_sched_barrier(0); mma_abt_one(c, a, b, c, 3, 2, 0); mma_abt_one(c, a, b, c, 3, 3, 0); __builtin_amdgcn_sched_barrier(0);
}

struct globals {
    a_gl a; b_gl b; c_gl c;
    float scale_a, scale_b;
    int bpr, bpc;
    hipStream_t stream;
};

__device__ __forceinline__ void compute_block_coords(int bid, int bpr, int bpc, int& br, int& bc) {
    int wgid = bid;
    const int num_wgs = gridDim.x;
#if RCR_4WAVE_ENABLE_XCD_SWIZZLE
    if (num_wgs >= NUM_XCDS && (num_wgs % NUM_XCDS) == 0)
        wgid = (wgid % NUM_XCDS) * (num_wgs / NUM_XCDS) + (wgid / NUM_XCDS);
#endif
    const int num_pid_m = bpr;
    const int num_pid_n = bpc;
    const int num_wgid_in_group = WGM * num_pid_n;
    const int group_id = wgid / num_wgid_in_group;
    const int first_pid_m = group_id * WGM;
    const int gsm = (first_pid_m + WGM <= num_pid_m) ? WGM : (num_pid_m - first_pid_m);
    br = first_pid_m + ((wgid % num_wgid_in_group) % gsm);
    bc = (wgid % num_wgid_in_group) / gsm;
}

__global__ __launch_bounds__(NUM_THREADS, 1)
void kernel(const globals g) {
    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];
    RT_A a[2]; RT_B b[2]; RT_C c[2][2];

    int br, bc;
    compute_block_coords(blockIdx.x, g.bpr, g.bpc, br, bc);
    if (br >= g.bpr || bc >= g.bpc) return;

    const int warp_m = warpid() / WARPS_COL;
    const int warp_n = warpid() % WARPS_COL;
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

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(28)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    auto a_st0 = kittens::subtile_inplace<BLK/2/WARPS_ROW, BK>(As[curr][0], {warp_m, 0});
    load_st_to_rt(a[0], a_st0);

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(24)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    auto b_st0 = kittens::subtile_inplace<BLK/2/WARPS_COL, BK>(Bs[curr][0], {warp_n, 0});
    load_st_to_rt(b[0], b_st0);

    #pragma unroll
    for (int k = 0; k < K_ITERS - 2; ++k, curr ^= 1, next ^= 1) {
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(16)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto b_st1 = kittens::subtile_inplace<BLK/2/WARPS_COL, BK>(Bs[curr][1], {warp_n, 0});
        do_interleaved_cluster(As[curr][0], g.a, coord<ST_A>{0, 0, br*WARPS_ROW, k+2},
            b[1], b_st1, a[0], b[0], c[0][0]);

        __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt lgkmcnt(0)"); __builtin_amdgcn_sched_barrier(0);

        auto a_st1 = kittens::subtile_inplace<BLK/2/WARPS_ROW, BK>(As[curr][1], {warp_m, 0});
        do_interleaved_cluster(Bs[curr][0], g.b, coord<ST_B>{0, 0, bc*WARPS_COL, k+2},
            a[1], a_st1, a[0], b[1], c[0][1]);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(16)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        a_st0 = kittens::subtile_inplace<BLK/2/WARPS_ROW, BK>(As[next][0], {warp_m, 0});
        do_interleaved_cluster(Bs[curr][1], g.b, coord<ST_B>{0, 0, bc*WARPS_COL+1, k+2},
            a[0], a_st0, a[1], b[0], c[1][0]);

        b_st0 = kittens::subtile_inplace<BLK/2/WARPS_COL, BK>(Bs[next][0], {warp_n, 0});
        do_interleaved_cluster(As[curr][1], g.a, coord<ST_A>{0, 0, br*WARPS_ROW+1, k+2},
            b[0], b_st0, a[1], b[1], c[1][1]);
    }

    // Penultimate iteration
    {
        __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt vmcnt(16)"); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt lgkmcnt(0)"); __builtin_amdgcn_sched_barrier(0);
        auto b_st1 = kittens::subtile_inplace<BLK/2/WARPS_COL, BK>(Bs[curr][1], {warp_n, 0});
        load_st_to_rt(b[1], b_st1);
        __builtin_amdgcn_sched_barrier(0); mma_ABt(c[0][0], a[0], b[0], c[0][0]); __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt lgkmcnt(0)"); __builtin_amdgcn_sched_barrier(0);
        auto a_st1 = kittens::subtile_inplace<BLK/2/WARPS_ROW, BK>(As[curr][1], {warp_m, 0});
        load_st_to_rt(a[1], a_st1);
        __builtin_amdgcn_sched_barrier(0); mma_ABt(c[0][1], a[0], b[1], c[0][1]); __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt vmcnt(8)"); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt lgkmcnt(0)"); __builtin_amdgcn_sched_barrier(0);
        a_st0 = kittens::subtile_inplace<BLK/2/WARPS_ROW, BK>(As[next][0], {warp_m, 0});
        load_st_to_rt(a[0], a_st0);
        __builtin_amdgcn_sched_barrier(0); mma_ABt(c[1][0], a[1], b[0], c[1][0]); __builtin_amdgcn_sched_barrier(0);
        b_st0 = kittens::subtile_inplace<BLK/2/WARPS_COL, BK>(Bs[next][0], {warp_n, 0});
        load_st_to_rt(b[0], b_st0);
        __builtin_amdgcn_sched_barrier(0); mma_ABt(c[1][1], a[1], b[1], c[1][1]); __builtin_amdgcn_sched_barrier(0);
        curr ^= 1; next ^= 1;
    }

    // Last iteration
    {
        __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt lgkmcnt(0)"); __builtin_amdgcn_sched_barrier(0);
        auto b_st1 = kittens::subtile_inplace<BLK/2/WARPS_COL, BK>(Bs[curr][1], {warp_n, 0});
        load_st_to_rt(b[1], b_st1);
        __builtin_amdgcn_sched_barrier(0); mma_ABt(c[0][0], a[0], b[0], c[0][0]); __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt lgkmcnt(0)"); __builtin_amdgcn_sched_barrier(0);
        auto a_st1 = kittens::subtile_inplace<BLK/2/WARPS_ROW, BK>(As[curr][1], {warp_m, 0});
        load_st_to_rt(a[1], a_st1);
        __builtin_amdgcn_sched_barrier(0); mma_ABt(c[0][1], a[0], b[1], c[0][1]); __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt lgkmcnt(0)"); __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_sched_barrier(0); mma_ABt(c[1][0], a[1], b[0], c[1][0]); __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_sched_barrier(0); mma_ABt(c[1][1], a[1], b[1], c[1][1]); __builtin_amdgcn_sched_barrier(0);
    }

    const float sc = g.scale_a * g.scale_b;
    mul(c[0][0], c[0][0], sc); mul(c[0][1], c[0][1], sc);
    mul(c[1][0], c[1][0], sc); mul(c[1][1], c[1][1], sc);

    store(g.c, c[0][0], {0, 0, br*WARPS_ROW*2 + warp_m, bc*WARPS_COL*2 + warp_n});
    store(g.c, c[0][1], {0, 0, br*WARPS_ROW*2 + warp_m, bc*WARPS_COL*2 + WARPS_COL + warp_n});
    store(g.c, c[1][0], {0, 0, br*WARPS_ROW*2 + WARPS_ROW + warp_m, bc*WARPS_COL*2 + warp_n});
    store(g.c, c[1][1], {0, 0, br*WARPS_ROW*2 + WARPS_ROW + warp_m, bc*WARPS_COL*2 + WARPS_COL + warp_n});
}

} // namespace rcr_4wave_kspec

using _gl_fp8  = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_bf16 = gl<bf16, -1, -1, -1, -1>;

static float to_float(pybind11::object obj) {
    if (pybind11::hasattr(obj, "item")) return obj.attr("item")().cast<float>();
    return obj.cast<float>();
}

static void gemm_rcr(pybind11::object a, pybind11::object b, pybind11::object c,
                      pybind11::object scale_a_obj, pybind11::object scale_b_obj,
                      int group_m) {
    auto ga = py::from_object<_gl_fp8>::make(a);
    auto gb = py::from_object<_gl_fp8>::make(b);
    auto gc = py::from_object<_gl_bf16>::make(c);

    int M = static_cast<int>(gc.rows());
    int N = static_cast<int>(gc.cols());
    int K = static_cast<int>(ga.cols());
    int bpr = M / BLK, bpc = N / BLK;

    if (K != K_DIM || M % BLK != 0 || N % BLK != 0) {
        throw std::runtime_error("Shape mismatch: K must equal K_DIM, M/N must be multiples of 256");
    }

    rcr_4wave_kspec::globals g{
        make_gl<rcr_4wave_kspec::a_gl>(reinterpret_cast<uint64_t>(ga.raw_ptr), 1, 1, M, K),
        make_gl<rcr_4wave_kspec::b_gl>(reinterpret_cast<uint64_t>(gb.raw_ptr), 1, 1, N, K),
        make_gl<rcr_4wave_kspec::c_gl>(reinterpret_cast<uint64_t>(gc.raw_ptr), 1, 1, M, N),
        to_float(scale_a_obj), to_float(scale_b_obj),
        bpr, bpc, {},
    };
    dim3 grid(bpr * bpc), block(rcr_4wave_kspec::NUM_THREADS);
    rcr_4wave_kspec::kernel<<<grid, block, 0, g.stream>>>(g);
}

PYBIND11_MODULE(tk_fp8_layouts, m) {
    m.def("gemm_rcr", &gemm_rcr,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_m") = 4);
}
