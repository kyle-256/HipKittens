#pragma once

#ifndef MXFP8_CRR_4WAVE_FAST_ENABLE
#define MXFP8_CRR_4WAVE_FAST_ENABLE 0
#endif

#ifndef MXFP8_CRR_4WAVE_GROUP_M
#define MXFP8_CRR_4WAVE_GROUP_M 4
#endif

#ifndef MXFP8_CRR_4WAVE_NUM_XCDS
#define MXFP8_CRR_4WAVE_NUM_XCDS 8
#endif

#ifndef MXFP8_CRR_4WAVE_ENABLE_XCD_SWIZZLE
#define MXFP8_CRR_4WAVE_ENABLE_XCD_SWIZZLE 1
#endif

#if MXFP8_CRR_4WAVE_FAST_ENABLE

namespace mxfp8_crr_4wave {

constexpr int WARPS_ROW = 2;
constexpr int WARPS_COL = 2;
constexpr int NUM_WARPS = WARPS_ROW * WARPS_COL;
constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
constexpr int WGM = MXFP8_CRR_4WAVE_GROUP_M;
constexpr int NUM_XCDS = MXFP8_CRR_4WAVE_NUM_XCDS;

using G4 = kittens::group<NUM_WARPS>;
using ST_A = st_fp8e4m3<BK, BLK / 2, st_16x128_v2a_s>;
using ST_B = st_fp8e4m3<BK, BLK / 2, st_16x128_v2a_s>;
using RT_A = rt_fp8e4m3<BK, BLK / 2 / WARPS_ROW, col_l, rt_128x16_s>;
using RT_B = rt_fp8e4m3<BK, BLK / 2 / WARPS_COL, col_l, rt_128x16_s>;
using RT_C = rt_fl<BLK / 2 / WARPS_ROW, BLK / 2 / WARPS_COL, col_l, rt_16x16_s>;

template<int K_PHASE, int N, int M>
__device__ __forceinline__ void mma_atb_scaled_one(
    RT_C& acc,
    const RT_A& a,
    const RT_B& b,
    const fp8e8m0_4 (&a_scale_packs)[BLK / 2 / WARPS_ROW / 32],
    const fp8e8m0_4 (&b_scale_packs)[(BLK / 2 / WARPS_COL + 31) / 32])
{
    constexpr int opsel_a = (K_PHASE << 1) | (N & 1);
    constexpr int opsel_b = (K_PHASE << 1) | (M & 1);
    const fp8e8m0_4 a_scale_pack = a_scale_packs[N / 2];
    const fp8e8m0_4 b_scale_pack = b_scale_packs[M / 2];
    mma_AtB_base_scaled<opsel_a, opsel_b>(
        acc.tiles[N][M],
        a.tiles[0][N],
        b.tiles[0][M],
        acc.tiles[N][M],
        a_scale_pack,
        b_scale_pack
    );
}

template<int K_PHASE>
__device__ __forceinline__ void mma_atb_scaled_tile(
    RT_C& acc,
    const RT_A& a,
    const RT_B& b,
    const fp8e8m0_4 (&a_scale_packs)[BLK / 2 / WARPS_ROW / 32],
    const fp8e8m0_4 (&b_scale_packs)[(BLK / 2 / WARPS_COL + 31) / 32])
{
    mma_atb_scaled_one<K_PHASE, 0, 0>(acc, a, b, a_scale_packs, b_scale_packs);
    mma_atb_scaled_one<K_PHASE, 0, 1>(acc, a, b, a_scale_packs, b_scale_packs);
    mma_atb_scaled_one<K_PHASE, 0, 2>(acc, a, b, a_scale_packs, b_scale_packs);
    mma_atb_scaled_one<K_PHASE, 0, 3>(acc, a, b, a_scale_packs, b_scale_packs);
    mma_atb_scaled_one<K_PHASE, 1, 0>(acc, a, b, a_scale_packs, b_scale_packs);
    mma_atb_scaled_one<K_PHASE, 1, 1>(acc, a, b, a_scale_packs, b_scale_packs);
    mma_atb_scaled_one<K_PHASE, 1, 2>(acc, a, b, a_scale_packs, b_scale_packs);
    mma_atb_scaled_one<K_PHASE, 1, 3>(acc, a, b, a_scale_packs, b_scale_packs);
    mma_atb_scaled_one<K_PHASE, 2, 0>(acc, a, b, a_scale_packs, b_scale_packs);
    mma_atb_scaled_one<K_PHASE, 2, 1>(acc, a, b, a_scale_packs, b_scale_packs);
    mma_atb_scaled_one<K_PHASE, 2, 2>(acc, a, b, a_scale_packs, b_scale_packs);
    mma_atb_scaled_one<K_PHASE, 2, 3>(acc, a, b, a_scale_packs, b_scale_packs);
    mma_atb_scaled_one<K_PHASE, 3, 0>(acc, a, b, a_scale_packs, b_scale_packs);
    mma_atb_scaled_one<K_PHASE, 3, 1>(acc, a, b, a_scale_packs, b_scale_packs);
    mma_atb_scaled_one<K_PHASE, 3, 2>(acc, a, b, a_scale_packs, b_scale_packs);
    mma_atb_scaled_one<K_PHASE, 3, 3>(acc, a, b, a_scale_packs, b_scale_packs);
}

static_assert(BLK == 256, "MXFP8 CRR 4-wave fast path requires BLK=256");
static_assert(BK == 128, "MXFP8 CRR 4-wave fast path requires BK=128");

__device__ __forceinline__ void compute_block_coords(int bid, int& br, int& bc) {
    int wgid = bid;
    const int num_wgs = gridDim.x;
#if MXFP8_CRR_4WAVE_ENABLE_XCD_SWIZZLE
    if (num_wgs >= NUM_XCDS && (num_wgs % NUM_XCDS) == 0) {
        wgid = (wgid % NUM_XCDS) * (num_wgs / NUM_XCDS) + (wgid / NUM_XCDS);
    }
#endif

    constexpr int num_pid_m = M_DIM / BLK;
    constexpr int num_pid_n = N_DIM / BLK;
    const int num_wgid_in_group = WGM * num_pid_n;
    const int group_id = wgid / num_wgid_in_group;
    const int first_pid_m = group_id * WGM;
    const int group_size_m =
        (first_pid_m + WGM <= num_pid_m) ? WGM : (num_pid_m - first_pid_m);

    br = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    bc = (wgid % num_wgid_in_group) / group_size_m;
}

template<bool PRESHUFFLED_QUANT>
__global__ __launch_bounds__(NUM_THREADS, 1)
void kernel(const layout_globals g) {
    constexpr int k_iters = K_DIM / BK;
    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];

    RT_A a[2];
    RT_B b[2];
    RT_C c[2][2];

    constexpr int bptA = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpmA = bptA * NUM_THREADS;
    constexpr int mptA = ST_A::rows * ST_A::cols * sizeof(fp8e4m3) / bpmA;
    uint32_t soA[mptA];
    G4::prefill_swizzled_offsets(As[0][0], g.a, soA);

    constexpr int bptB = ST_B::underlying_subtile_bytes_per_thread;
    constexpr int bpmB = bptB * NUM_THREADS;
    constexpr int mptB = ST_B::rows * ST_B::cols * sizeof(fp8e4m3) / bpmB;
    uint32_t soB[mptB];
    G4::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

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

    constexpr int a_m_sub = BLK / 2 / WARPS_ROW;
    constexpr int b_n_sub = BLK / 2 / WARPS_COL;

    auto scale_a_base = [&](int half) {
        return br * BLK + half * (BLK / 2) + warp_m * a_m_sub;
    };
    auto scale_b_base = [&](int half) {
        return bc * BLK + half * (BLK / 2) + warp_n * b_n_sub;
    };

    constexpr int a_pack_count = a_m_sub / 32;
    constexpr int b_pack_count = (b_n_sub + 31) / 32;
    fp8e8m0_4 a0_scale_packs[a_pack_count];
    fp8e8m0_4 a1_scale_packs[a_pack_count];
    fp8e8m0_4 b0_scale_packs[b_pack_count];
    fp8e8m0_4 b1_scale_packs[b_pack_count];
    const uint8_t* a0_scale_row_bases[a_pack_count];
    const uint8_t* a1_scale_row_bases[a_pack_count];
    const uint8_t* b0_scale_row_bases[b_pack_count];
    const uint8_t* b1_scale_row_bases[b_pack_count];

    if constexpr (PRESHUFFLED_QUANT) {
        #pragma unroll
        for (int pack_idx = 0; pack_idx < a_pack_count; ++pack_idx) {
            a0_scale_row_bases[pack_idx] = preshuffled_scale_row_base_ptr(
                g.a_scale, (scale_a_base(0) + pack_idx * 32) >> 5);
            a1_scale_row_bases[pack_idx] = preshuffled_scale_row_base_ptr(
                g.a_scale, (scale_a_base(1) + pack_idx * 32) >> 5);
        }
        #pragma unroll
        for (int pack_idx = 0; pack_idx < b_pack_count; ++pack_idx) {
            b0_scale_row_bases[pack_idx] = preshuffled_scale_row_base_ptr(
                g.b_scale, (scale_b_base(0) + pack_idx * 32) >> 5);
            b1_scale_row_bases[pack_idx] = preshuffled_scale_row_base_ptr(
                g.b_scale, (scale_b_base(1) + pack_idx * 32) >> 5);
        }
    }

    int cached_k_pair = -1;
    auto ensure_scale_packs = [&](int k_pair) {
        if (k_pair == cached_k_pair) return;
        if constexpr (PRESHUFFLED_QUANT) {
            #pragma unroll
            for (int pack_idx = 0; pack_idx < a_pack_count; ++pack_idx) {
                a0_scale_packs[pack_idx] = load_scale_pair_pack_16x128_preshuffled_from_row_base(
                    a0_scale_row_bases[pack_idx], k_pair, lane_scale_byte_offset);
                a1_scale_packs[pack_idx] = load_scale_pair_pack_16x128_preshuffled_from_row_base(
                    a1_scale_row_bases[pack_idx], k_pair, lane_scale_byte_offset);
            }
            #pragma unroll
            for (int pack_idx = 0; pack_idx < b_pack_count; ++pack_idx) {
                b0_scale_packs[pack_idx] = load_scale_pair_pack_16x128_preshuffled_from_row_base(
                    b0_scale_row_bases[pack_idx], k_pair, lane_scale_byte_offset);
                b1_scale_packs[pack_idx] = load_scale_pair_pack_16x128_preshuffled_from_row_base(
                    b1_scale_row_bases[pack_idx], k_pair, lane_scale_byte_offset);
            }
        } else {
            load_scale_packs_16x128<false>(
                a0_scale_packs, g.a_scale, scale_a_base(0), k_pair, lane_nonk, lane_kblk);
            load_scale_packs_16x128<false>(
                a1_scale_packs, g.a_scale, scale_a_base(1), k_pair, lane_nonk, lane_kblk);
            load_scale_packs_16x128<false>(
                b0_scale_packs, g.b_scale, scale_b_base(0), k_pair, lane_nonk, lane_kblk);
            load_scale_packs_16x128<false>(
                b1_scale_packs, g.b_scale, scale_b_base(1), k_pair, lane_nonk, lane_kblk);
        }
        cached_k_pair = k_pair;
    };

    auto a_co = [&](int m_tile, int k_tile) -> coord<ST_A> {
        return {0, 0, k_tile, m_tile};
    };
    auto b_co = [&](int n_tile, int k_tile) -> coord<ST_B> {
        return {0, 0, k_tile, n_tile};
    };

    auto load_a_s2r = [&](RT_A& dst, ST_A& tile) {
        load_col_from_v2a_st(dst, tile, warp_m * a_m_sub);
    };
    auto load_b_s2r = [&](RT_B& dst, ST_B& tile) {
        load_col_from_v2a_st(dst, tile, warp_n * b_n_sub);
    };

    int curr = 0, next = 1;

    G4::load(As[curr][0], g.a, a_co(br * WARPS_ROW, 0), soA);
    G4::load(Bs[curr][0], g.b, b_co(bc * WARPS_COL, 0), soB);
    G4::load(Bs[curr][1], g.b, b_co(bc * WARPS_COL + 1, 0), soB);
    G4::load(As[curr][1], g.a, a_co(br * WARPS_ROW + 1, 0), soA);

    zero(c[0][0]);
    zero(c[0][1]);
    zero(c[1][0]);
    zero(c[1][1]);

    G4::load(As[next][0], g.a, a_co(br * WARPS_ROW, 1), soA);
    G4::load(Bs[next][0], g.b, b_co(bc * WARPS_COL, 1), soB);
    G4::load(Bs[next][1], g.b, b_co(bc * WARPS_COL + 1, 1), soB);
    G4::load(As[next][1], g.a, a_co(br * WARPS_ROW + 1, 1), soA);

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    #pragma unroll 2
    for (int k = 0; k < k_iters - 2; ++k, curr ^= 1, next ^= 1) {
        const int k_pair = k >> 1;
        const int k_phase = k & 1;
        ensure_scale_packs(k_pair);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        load_a_s2r(a[0], As[curr][0]);

        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        load_b_s2r(b[0], Bs[curr][0]);
        load_b_s2r(b[1], Bs[curr][1]);
        load_a_s2r(a[1], As[curr][1]);

        asm volatile("s_waitcnt lgkmcnt(0)");

        G4::load(As[curr][0], g.a, a_co(br * WARPS_ROW, k + 2), soA);
        G4::load(Bs[curr][0], g.b, b_co(bc * WARPS_COL, k + 2), soB);
        G4::load(As[curr][1], g.a, a_co(br * WARPS_ROW + 1, k + 2), soA);
        G4::load(Bs[curr][1], g.b, b_co(bc * WARPS_COL + 1, k + 2), soB);

        __builtin_amdgcn_sched_barrier(0);
        if (k_phase == 0) {
            mma_atb_scaled_tile<0>(c[0][0], a[0], b[0], a0_scale_packs, b0_scale_packs);
            mma_atb_scaled_tile<0>(c[0][1], a[0], b[1], a0_scale_packs, b1_scale_packs);
            mma_atb_scaled_tile<0>(c[1][0], a[1], b[0], a1_scale_packs, b0_scale_packs);
            mma_atb_scaled_tile<0>(c[1][1], a[1], b[1], a1_scale_packs, b1_scale_packs);
        } else {
            mma_atb_scaled_tile<1>(c[0][0], a[0], b[0], a0_scale_packs, b0_scale_packs);
            mma_atb_scaled_tile<1>(c[0][1], a[0], b[1], a0_scale_packs, b1_scale_packs);
            mma_atb_scaled_tile<1>(c[1][0], a[1], b[0], a1_scale_packs, b0_scale_packs);
            mma_atb_scaled_tile<1>(c[1][1], a[1], b[1], a1_scale_packs, b1_scale_packs);
        }
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
    }

    {
        const int k = k_iters - 2;
        const int k_pair = k >> 1;
        const int k_phase = k & 1;
        ensure_scale_packs(k_pair);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(16)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        load_a_s2r(a[0], As[curr][0]);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load_b_s2r(b[0], Bs[curr][0]);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        load_b_s2r(b[1], Bs[curr][1]);

        __builtin_amdgcn_sched_barrier(0);
        if (k_phase == 0) {
            mma_atb_scaled_tile<0>(c[0][0], a[0], b[0], a0_scale_packs, b0_scale_packs);
        } else {
            mma_atb_scaled_tile<1>(c[0][0], a[0], b[0], a0_scale_packs, b0_scale_packs);
        }
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        load_a_s2r(a[1], As[curr][1]);

        __builtin_amdgcn_sched_barrier(0);
        if (k_phase == 0) {
            mma_atb_scaled_tile<0>(c[0][1], a[0], b[1], a0_scale_packs, b1_scale_packs);
        } else {
            mma_atb_scaled_tile<1>(c[0][1], a[0], b[1], a0_scale_packs, b1_scale_packs);
        }
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(8)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        load_a_s2r(a[0], As[next][0]);

        __builtin_amdgcn_sched_barrier(0);
        if (k_phase == 0) {
            mma_atb_scaled_tile<0>(c[1][0], a[1], b[0], a1_scale_packs, b0_scale_packs);
        } else {
            mma_atb_scaled_tile<1>(c[1][0], a[1], b[0], a1_scale_packs, b0_scale_packs);
        }
        __builtin_amdgcn_sched_barrier(0);

        load_b_s2r(b[0], Bs[next][0]);

        __builtin_amdgcn_sched_barrier(0);
        if (k_phase == 0) {
            mma_atb_scaled_tile<0>(c[1][1], a[1], b[1], a1_scale_packs, b1_scale_packs);
        } else {
            mma_atb_scaled_tile<1>(c[1][1], a[1], b[1], a1_scale_packs, b1_scale_packs);
        }
        __builtin_amdgcn_sched_barrier(0);

        curr ^= 1;
        next ^= 1;
    }

    {
        const int k = k_iters - 1;
        const int k_pair = k >> 1;
        const int k_phase = k & 1;
        ensure_scale_packs(k_pair);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        load_b_s2r(b[1], Bs[curr][1]);

        __builtin_amdgcn_sched_barrier(0);
        if (k_phase == 0) {
            mma_atb_scaled_tile<0>(c[0][0], a[0], b[0], a0_scale_packs, b0_scale_packs);
        } else {
            mma_atb_scaled_tile<1>(c[0][0], a[0], b[0], a0_scale_packs, b0_scale_packs);
        }
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        load_a_s2r(a[1], As[curr][1]);

        __builtin_amdgcn_sched_barrier(0);
        if (k_phase == 0) {
            mma_atb_scaled_tile<0>(c[0][1], a[0], b[1], a0_scale_packs, b1_scale_packs);
        } else {
            mma_atb_scaled_tile<1>(c[0][1], a[0], b[1], a0_scale_packs, b1_scale_packs);
        }
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        if (k_phase == 0) {
            mma_atb_scaled_tile<0>(c[1][0], a[1], b[0], a1_scale_packs, b0_scale_packs);
        } else {
            mma_atb_scaled_tile<1>(c[1][0], a[1], b[0], a1_scale_packs, b0_scale_packs);
        }
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        if (k_phase == 0) {
            mma_atb_scaled_tile<0>(c[1][1], a[1], b[1], a1_scale_packs, b1_scale_packs);
        } else {
            mma_atb_scaled_tile<1>(c[1][1], a[1], b[1], a1_scale_packs, b1_scale_packs);
        }
        __builtin_amdgcn_sched_barrier(0);
    }

    mul(c[0][0], c[0][0], g.scale);
    mul(c[0][1], c[0][1], g.scale);
    mul(c[1][0], c[1][0], g.scale);
    mul(c[1][1], c[1][1], g.scale);

    store(g.c, c[0][0], {0, 0, br * WARPS_ROW * 2 + warp_m, bc * WARPS_COL * 2 + warp_n});
    store(g.c, c[0][1], {0, 0, br * WARPS_ROW * 2 + warp_m, bc * WARPS_COL * 2 + WARPS_COL + warp_n});
    store(g.c, c[1][0], {0, 0, br * WARPS_ROW * 2 + WARPS_ROW + warp_m, bc * WARPS_COL * 2 + warp_n});
    store(g.c, c[1][1], {0, 0, br * WARPS_ROW * 2 + WARPS_ROW + warp_m, bc * WARPS_COL * 2 + WARPS_COL + warp_n});
}

__host__ inline bool can_use(const layout_globals& g) {
    return g.m == M_DIM && g.n == N_DIM && g.k == K_DIM;
}

template<bool PRESHUFFLED_QUANT>
__host__ inline void dispatch(const layout_globals& g) {
    const dim3 grid((g.m / BLK) * (g.n / BLK));
    kernel<PRESHUFFLED_QUANT><<<grid, dim3(NUM_THREADS), 0, g.stream>>>(g);
}

} // namespace mxfp8_crr_4wave

#endif
