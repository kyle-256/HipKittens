#pragma once

#ifndef RCR_USE_4WAVE_DYNAMIC
#define RCR_USE_4WAVE_DYNAMIC 0
#endif

#if RCR_USE_4WAVE_DYNAMIC

namespace rcr_4w {

constexpr int WM = 2, WN = 2;
constexpr int NW = WM * WN;
constexpr int NT = NW * WARP_THREADS;

using G4 = kittens::group<NW>;
using ST4 = st_fp8e4m3<HB, BK, st_16x128_s>;
using RA = rt_fp8e4m3<HB / WM, BK, row_l, rt_16x128_s>;
using RB = rt_fp8e4m3<HB / WN, BK, row_l, rt_16x128_s>;
using RC = rt_fl<HB / WM, HB / WN, col_l, rt_16x16_s>;

struct g2s_desc {
    i32x4 srsrc;
    uintptr_t lds_base;
};

__device__ __forceinline__ g2s_desc prepare_g2s(
    ST4& dst, const void* tile_ptr, int row_stride)
{
    i32x4 srsrc = make_srsrc(const_cast<void*>(tile_ptr),
                              row_stride * ST4::rows * sizeof(fp8e4m3));
    constexpr int bpt = ST4::underlying_subtile_bytes_per_thread;
    constexpr int bpw = bpt * WARP_THREADS;
    uintptr_t lds_base =
        reinterpret_cast<uintptr_t>(&dst.data[0]) + (kittens::warpid() * bpw);
    return {srsrc, lds_base};
}

constexpr int G2S_PASSES = HB * BK * sizeof(fp8e4m3) /
    (ST4::underlying_subtile_bytes_per_thread * NT);

struct g2s_offsets {
    uint32_t global_off[G2S_PASSES];
};

__device__ __forceinline__ g2s_offsets prefill_g2s_offsets(
    ST4& dst, int row_stride)
{
    using T = fp8e4m3;
    constexpr int bpt = ST4::underlying_subtile_bytes_per_thread;
    constexpr int bpw = bpt * WARP_THREADS;
    const int lane = kittens::laneid();
    const int warp = kittens::warpid() % NW;

    g2s_offsets off;
    #pragma unroll
    for (int I = 0; I < G2S_PASSES; I++) {
        const int lbo = (lane * bpt) + (warp * bpw) + (I * NW * bpw);
        const int stid = lbo / ST4::underlying_subtile_bytes;
        const int str = stid / ST4::underlying_subtiles_per_row;
        const int stc = stid % ST4::underlying_subtiles_per_row;
        const int sub = lbo % ST4::underlying_subtile_bytes;
        const int row = sub / ST4::underlying_subtile_row_bytes;
        const int col = (sub % ST4::underlying_subtile_row_bytes) / sizeof(T);
        const uint32_t sw = dst.swizzle({row, col});
        const int swr = (sw / ST4::underlying_subtile_row_bytes) +
                         str * ST4::underlying_subtile_rows;
        const int swc = (sw % ST4::underlying_subtile_row_bytes) / sizeof(T) +
                         stc * ST4::underlying_subtile_cols;
        off.global_off[I] = (swr * row_stride + swc) * sizeof(T);
    }
    return off;
}

template<int I>
__device__ __forceinline__ void g2s_pass(
    const g2s_desc& addr, const g2s_offsets& off)
{
    constexpr int bpt = ST4::underlying_subtile_bytes_per_thread;
    constexpr int bpw = bpt * WARP_THREADS;
    uintptr_t lds_addr = addr.lds_base + (I * NW * bpw);
    as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);
    llvm_amdgcn_raw_buffer_load_lds(
        addr.srsrc, lds_ptr, bpt, off.global_off[I], 0, 0,
        static_cast<int>(coherency::cache_all));
}

template<int NUM_OFFSETS, typename RT, typename ST>
__device__ __forceinline__ void prefill_s2r_offsets(
    RT& dst, ST& src, uint32_t* off)
{
    using U = typename ST::dtype;
    constexpr int subtile_stride = RT::base_tile_cols * sizeof(U) / 2;
    const uint32_t st_offset =
        (kittens::laneid() % RT::base_tile_rows) * ST::underlying_cols +
        (kittens::laneid() / RT::base_tile_rows * 16 / sizeof(U));
    const uint32_t base_addr = reinterpret_cast<uintptr_t>(&src.data[st_offset]);
    off[0] = base_addr;
    off[0] ^= (((off[0] % (256 * 8)) >> 8) << 4);
    off[1] = base_addr + subtile_stride;
    off[1] ^= (((off[1] % (256 * 8)) >> 8) << 4);
}

template<int RR, int RC, int KS, typename RT, typename ST>
__device__ __forceinline__ void s2r_one(RT& dst, ST& src, uint32_t* off)
{
    constexpr int packing = base_types::packing<typename RT::dtype>::num();
    const int idx = KS * RT::base_tile_stride / packing;
    constexpr int row_stride =
        RT::base_tile_rows * ST::underlying_cols * sizeof(fp8e4m3);
    asm volatile(
        "ds_read_b128 %0, %1 offset:%2\n"
        : "=v"(*reinterpret_cast<float4*>(&dst.tiles[RR][RC].data[idx]))
        : "v"(off[KS]), "i"(RR * row_stride)
        : "memory");
}

template<typename D, typename A, typename B, typename C>
__device__ __forceinline__ void mma1(D& d, const A& a, const B& b, const C& c,
                                     int n, int m, int k)
{
    mma_ABt_base(d.tiles[n][m], a.tiles[n][k], b.tiles[m][k], c.tiles[n][m]);
}

template<typename S2R_RT, typename S2R_ST>
__device__ __forceinline__ void do_cluster(
    const g2s_desc& g2s_addr, const g2s_offsets& g2s_off,
    S2R_RT& s2r_dst, S2R_ST& s2r_src,
    RA& a, RB& b, RC& c)
{
    uint32_t s2r_off[2];
    prefill_s2r_offsets<2>(s2r_dst, s2r_src, s2r_off);

    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 0, 0, 0);
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 0, 1, 0);
    __builtin_amdgcn_sched_barrier(0);

    g2s_pass<0>(g2s_addr, g2s_off);
    s2r_one<0, 0, 0>(s2r_dst, s2r_src, s2r_off);

    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 0, 2, 0);
    __builtin_amdgcn_sched_barrier(0);

    s2r_one<0, 0, 1>(s2r_dst, s2r_src, s2r_off);

    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 0, 3, 0);
    __builtin_amdgcn_sched_barrier(0);

    g2s_pass<1>(g2s_addr, g2s_off);
    s2r_one<1, 0, 0>(s2r_dst, s2r_src, s2r_off);
    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 1, 0, 0);
    mma1(c, a, b, c, 1, 1, 0);
    __builtin_amdgcn_sched_barrier(0);

    s2r_one<1, 0, 1>(s2r_dst, s2r_src, s2r_off);
    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 1, 2, 0);
    mma1(c, a, b, c, 1, 3, 0);
    __builtin_amdgcn_sched_barrier(0);

    g2s_pass<2>(g2s_addr, g2s_off);
    s2r_one<2, 0, 0>(s2r_dst, s2r_src, s2r_off);
    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 2, 0, 0);
    mma1(c, a, b, c, 2, 1, 0);
    __builtin_amdgcn_sched_barrier(0);

    s2r_one<2, 0, 1>(s2r_dst, s2r_src, s2r_off);
    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 2, 2, 0);
    mma1(c, a, b, c, 2, 3, 0);
    __builtin_amdgcn_sched_barrier(0);

    g2s_pass<3>(g2s_addr, g2s_off);
    s2r_one<3, 0, 0>(s2r_dst, s2r_src, s2r_off);
    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 3, 0, 0);
    mma1(c, a, b, c, 3, 1, 0);
    __builtin_amdgcn_sched_barrier(0);

    s2r_one<3, 0, 1>(s2r_dst, s2r_src, s2r_off);
    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 3, 2, 0);
    mma1(c, a, b, c, 3, 3, 0);
    __builtin_amdgcn_sched_barrier(0);
}

template<ducks::rt::row_layout RT, ducks::st::all ST>
__device__ __forceinline__ void load_full_rt(RT& dst, const ST& src) {
    static_assert(RT::rows == ST::rows && RT::cols == ST::cols);
    using T2 = typename RT::dtype;
    using U2 = typename base_types::packing<typename ST::dtype>::packed_type;
    constexpr int packing = base_types::packing<T2>::num();
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
                const uint32_t offset =
                    sizeof(fp8e4m3) * (src_ptr + row * ST::underlying_subtile_cols + col);
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
                        asm volatile("ds_read_b128 %0, %1 offset:%2\n"
                            : "=v"(*reinterpret_cast<float4*>(&dst.tiles[rr][rc].data[idx]))
                            : "v"(addr), "i"(soff) : "memory");
                    }
                }
            }
        }
    }
}

__global__ __launch_bounds__(NT, 1)
void kernel(const layout_globals g) {
    int br, bc;
    gemm_compute_block_coords(blockIdx.x, g.bpr, g.bpc, g.group_m, br, bc);
    if (br >= g.bpr || bc >= g.bpc || g.ki <= 0) return;

    const int wm = kittens::warpid() / WN;
    const int wn = kittens::warpid() % WN;
    const int ki = g.ki;

    const fp8e4m3* a_base = reinterpret_cast<const fp8e4m3*>(g.a.raw_ptr);
    const fp8e4m3* b_base = reinterpret_cast<const fp8e4m3*>(g.b.raw_ptr);
    const int a_stride = g.k;
    const int b_stride = g.k;

    auto a_ptr = [&](int row_tile, int k_tile) -> const void* {
        return a_base + row_tile * HB * a_stride + k_tile * BK;
    };
    auto b_ptr = [&](int col_tile, int k_tile) -> const void* {
        return b_base + col_tile * HB * b_stride + k_tile * BK;
    };

    __shared__ ST4 As[2][2];
    __shared__ ST4 Bs[2][2];
    RA a_reg[2];
    RB b_reg[2];
    RC c[2][2];
    zero(c[0][0]); zero(c[0][1]); zero(c[1][0]); zero(c[1][1]);

    constexpr int bpt = ST4::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * NT;
    constexpr int mpt = HB * BK * sizeof(fp8e4m3) / bpm;
    uint32_t soA[mpt], soB[mpt];
    G4::prefill_swizzled_offsets(As[0][0], g.a, soA);
    G4::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

    g2s_offsets g2s_off_A = prefill_g2s_offsets(As[0][0], a_stride);
    g2s_offsets g2s_off_B = prefill_g2s_offsets(Bs[0][0], b_stride);

    auto a_co = [&](int s, int k) -> coord<ST4> { return {0, 0, s, k}; };
    auto b_co = [&](int s, int k) -> coord<ST4> { return {0, 0, s, k}; };

    int cur = 0, nxt = 1;
    G4::load(As[cur][0], g.a, a_co(br * WM,     0), soA);
    G4::load(Bs[cur][0], g.b, b_co(bc * WN,     0), soB);
    G4::load(Bs[cur][1], g.b, b_co(bc * WN + 1, 0), soB);
    G4::load(As[cur][1], g.a, a_co(br * WM + 1, 0), soA);

    G4::load(As[nxt][0], g.a, a_co(br * WM,     1), soA);
    G4::load(Bs[nxt][0], g.b, b_co(bc * WN,     1), soB);
    G4::load(Bs[nxt][1], g.b, b_co(bc * WN + 1, 1), soB);
    G4::load(As[nxt][1], g.a, a_co(br * WM + 1, 1), soA);

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(28)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    auto a_sub0 = subtile_inplace<HB / WM, BK>(As[cur][0], {wm, 0});
    load_full_rt(a_reg[0], a_sub0);

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(24)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    auto b_sub0 = subtile_inplace<HB / WN, BK>(Bs[cur][0], {wn, 0});
    load_full_rt(b_reg[0], b_sub0);

    #pragma unroll 1
    for (int k = 0; k < ki - 2; ++k, cur ^= 1, nxt ^= 1) {
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(16)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto b_sub1 = subtile_inplace<HB / WN, BK>(Bs[cur][1], {wn, 0});
        g2s_desc addr_a0 = prepare_g2s(As[cur][0], a_ptr(br * WM, k + 2), a_stride);
        do_cluster(
            addr_a0, g2s_off_A,
            b_reg[1], b_sub1,
            a_reg[0], b_reg[0], c[0][0]);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_sub1 = subtile_inplace<HB / WM, BK>(As[cur][1], {wm, 0});
        g2s_desc addr_b0 = prepare_g2s(Bs[cur][0], b_ptr(bc * WN, k + 2), b_stride);
        do_cluster(
            addr_b0, g2s_off_B,
            a_reg[1], a_sub1,
            a_reg[0], b_reg[1], c[0][1]);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(16)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        a_sub0 = subtile_inplace<HB / WM, BK>(As[nxt][0], {wm, 0});
        g2s_desc addr_b1 = prepare_g2s(Bs[cur][1], b_ptr(bc * WN + 1, k + 2), b_stride);
        do_cluster(
            addr_b1, g2s_off_B,
            a_reg[0], a_sub0,
            a_reg[1], b_reg[0], c[1][0]);

        b_sub0 = subtile_inplace<HB / WN, BK>(Bs[nxt][0], {wn, 0});
        g2s_desc addr_a1 = prepare_g2s(As[cur][1], a_ptr(br * WM + 1, k + 2), a_stride);
        do_cluster(
            addr_a1, g2s_off_A,
            b_reg[0], b_sub0,
            a_reg[1], b_reg[1], c[1][1]);
    }

    // Epilogue: k = ki - 2
    {
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(16)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto b_sub1 = subtile_inplace<HB / WN, BK>(Bs[cur][1], {wn, 0});
        load_full_rt(b_reg[1], b_sub1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][0], a_reg[0], b_reg[0], c[0][0]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_sub1 = subtile_inplace<HB / WM, BK>(As[cur][1], {wm, 0});
        load_full_rt(a_reg[1], a_sub1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][1], a_reg[0], b_reg[1], c[0][1]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(8)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        a_sub0 = subtile_inplace<HB / WM, BK>(As[nxt][0], {wm, 0});
        load_full_rt(a_reg[0], a_sub0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][0], a_reg[1], b_reg[0], c[1][0]);
        __builtin_amdgcn_sched_barrier(0);

        b_sub0 = subtile_inplace<HB / WN, BK>(Bs[nxt][0], {wn, 0});
        load_full_rt(b_reg[0], b_sub0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][1], a_reg[1], b_reg[1], c[1][1]);
        __builtin_amdgcn_sched_barrier(0);

        cur ^= 1; nxt ^= 1;
    }

    // Last iteration
    {
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto b_sub1 = subtile_inplace<HB / WN, BK>(Bs[cur][1], {wn, 0});
        load_full_rt(b_reg[1], b_sub1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][0], a_reg[0], b_reg[0], c[0][0]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_sub1 = subtile_inplace<HB / WM, BK>(As[cur][1], {wm, 0});
        load_full_rt(a_reg[1], a_sub1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][1], a_reg[0], b_reg[1], c[0][1]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][0], a_reg[1], b_reg[0], c[1][0]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][1], a_reg[1], b_reg[1], c[1][1]);
        __builtin_amdgcn_sched_barrier(0);
    }

    const float sc = g.scale_a * g.scale_b;
    mul(c[0][0], c[0][0], sc);
    mul(c[0][1], c[0][1], sc);
    mul(c[1][0], c[1][0], sc);
    mul(c[1][1], c[1][1], sc);

    store(g.c, c[0][0], {0, 0, br * WM * 2 + wm,      bc * WN * 2 + wn});
    store(g.c, c[0][1], {0, 0, br * WM * 2 + wm,      bc * WN * 2 + WN + wn});
    store(g.c, c[1][0], {0, 0, br * WM * 2 + WM + wm, bc * WN * 2 + wn});
    store(g.c, c[1][1], {0, 0, br * WM * 2 + WM + wm, bc * WN * 2 + WN + wn});
}

} // namespace rcr_4w

#endif
