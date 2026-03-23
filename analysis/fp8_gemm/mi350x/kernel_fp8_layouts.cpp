#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

#define M_DIM 8192
#define K_DIM 8192
#define N_DIM 8192

constexpr int BLK = 256, BK = 128;
constexpr int HB  = BLK / 2;
constexpr int WARPS_M = 2, WARPS_N = 4;
constexpr int _NUM_WARPS   = WARPS_M * WARPS_N;
constexpr int _NUM_THREADS = _NUM_WARPS * WARP_THREADS;
constexpr int RBM = BLK / WARPS_M / 2;   // 64
constexpr int RBN = BLK / WARPS_N / 2;   // 32
constexpr int BPR = M_DIM / BLK;
constexpr int BPC = N_DIM / BLK;
constexpr int KI  = K_DIM / BK;

using G = kittens::group<_NUM_WARPS>;
using _gl_fp8  = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_bf16 = gl<bf16, -1, -1, -1, -1>;

enum class Layout { RCR, RRR, CRR };

// Row-layout shared/register tiles (for A in RCR/RRR, B in RCR)
using ST_row = st_fp8e4m3<HB, BK, st_16x128_s>;    // 128×128, M/N rows × K cols
using A_row_reg = rt_fp8e4m3<RBM, BK, row_l, rt_16x128_s>;
using B_row_reg = rt_fp8e4m3<RBN, BK, row_l, rt_16x128_s>;

// Col-layout register tiles (for B in RRR, A/B in CRR)
using A_col_reg = rt_fp8e4m3<BK, RBM, col_l, rt_128x16_s>;  // 128×64
using B_col_reg = rt_fp8e4m3<BK, RBN, col_l, rt_128x16_s>;  // 128×32

using ST_v2  = st_fp8e4m3<HB, BK, st_16x128_v2_s>;
using ST_v2a = st_fp8e4m3<HB, BK, st_16x128_v2a_s>;

template<typename RT>
__device__ __forceinline__ void load_col_from_v2_st(
    RT& dst, const ST_v2& tile, int col_start)
{
    const int laneid = kittens::laneid();
    const int row_off = ((laneid % 16) / 2) + ((laneid / 16) * 16);
    const int col_off = (laneid % 2) * 8;
    const uint32_t tile_base = reinterpret_cast<uintptr_t>(&tile.data[0]);

    #pragma unroll
    for (int k = 0; k < 2; k++) {
        const int idx = k * 4;
        const int k_row = row_off + k * 64;
        const int k_next = k_row + 8;

        const uint32_t base_k = tile_base + ((k_row >> 4) << 11) + ((k_row & 15) << 7);
        const uint32_t sw_k   = ((k_row & 7)) << 4;
        const uint32_t base_n = tile_base + ((k_next >> 4) << 11) + ((k_next & 15) << 7);
        const uint32_t sw_n   = ((k_next & 7)) << 4;

        #pragma unroll
        for (int j = 0; j < RT::width; j++) {
            const uint32_t nc = col_start + j * 16 + col_off;
            const uint32_t addr = base_k + (nc ^ sw_k);
            const uint32_t next_addr = base_n + (nc ^ sw_n);

            asm volatile(
                "ds_read_b64_tr_b8 %0, %2 offset:%4\n"
                "ds_read_b64_tr_b8 %1, %3 offset:%4\n"
                : "=v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx])),
                  "=v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx + 2]))
                : "v"(addr), "v"(next_addr), "i"(0)
                : "memory"
            );
        }
    }
}

template<typename RT>
__device__ __forceinline__ void load_col_from_v2a_st(
    RT& dst, const ST_v2a& tile, int col_start)
{
    const int laneid = kittens::laneid();
    const int row_off = ((laneid % 16) / 2) + ((laneid / 16) * 16);
    const int col_off = (laneid % 2) * 8;
    const uint32_t tile_base = reinterpret_cast<uintptr_t>(&tile.data[0]);

    #pragma unroll
    for (int k = 0; k < 2; k++) {
        const int idx = k * 4;
        const int k_row = row_off + k * 64;
        const int k_next = k_row + 8;

        const uint32_t base_k = tile_base + ((k_row >> 4) << 11) + ((k_row & 15) << 7);
        const uint32_t sw_k   = ((k_row & 7)) << 4;
        const uint32_t base_n = tile_base + ((k_next >> 4) << 11) + ((k_next & 15) << 7);
        const uint32_t sw_n   = ((k_next & 7)) << 4;

        #pragma unroll
        for (int j = 0; j < RT::width; j++) {
            const uint32_t nc = col_start + j * 16 + col_off;
            const uint32_t addr = base_k + (nc ^ sw_k);
            const uint32_t next_addr = base_n + (nc ^ sw_n);

            asm volatile(
                "ds_read_b64_tr_b8 %0, %2 offset:%4\n"
                "ds_read_b64_tr_b8 %1, %3 offset:%4\n"
                : "=v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx])),
                  "=v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx + 2]))
                : "v"(addr), "v"(next_addr), "i"(0)
                : "memory"
            );
        }
    }
}

struct layout_globals {
    _gl_fp8 a, b;
    _gl_bf16 c;
    hipStream_t stream;
    dim3 grid()  { return dim3(BPR * BPC); }
    dim3 block() { return dim3(_NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

template<Layout L>
__global__ __launch_bounds__(_NUM_THREADS, 2)
void gemm_kernel(const layout_globals g) {
    int bid = blockIdx.x;
    int br = bid / BPC, bc = bid % BPC;
    int wm = warpid() / WARPS_N, wn = warpid() % WARPS_N;

    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;
    zero(cA); zero(cB); zero(cC); zero(cD);

    if constexpr (L == Layout::RCR) {
        __shared__ ST_row As[2][2];
        __shared__ ST_row Bs[2][2];
        A_row_reg a;
        B_row_reg b0, b1;

        constexpr int bpt = ST_row::underlying_subtile_bytes_per_thread;
        constexpr int bpm = bpt * _NUM_THREADS;
        constexpr int mpt = ST_row::rows * ST_row::cols * sizeof(fp8e4m3) / bpm;
        uint32_t soA[mpt], soB[mpt];
        G::prefill_swizzled_offsets(As[0][0], g.a, soA);
        G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

        auto a_co = [&](int s, int k) -> coord<ST_row> { return {0, 0, s, k}; };
        auto b_co = [&](int s, int k) -> coord<ST_row> { return {0, 0, s, k}; };

        auto load_a = [&](A_row_reg& dst, ST_row& tile, int wi) {
            auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
            load(dst, sub);
        };
        auto load_b = [&](B_row_reg& dst, ST_row& tile, int wi) {
            auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
            load(dst, sub);
        };

        int tic = 0, toc = 1;
        G::load(Bs[tic][0], g.b, b_co(bc*2,   0), soB);
        G::load(As[tic][0], g.a, a_co(br*2,   0), soA);
        G::load(Bs[tic][1], g.b, b_co(bc*2+1, 0), soB);
        G::load(As[tic][1], g.a, a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        G::load(Bs[toc][0], g.b, b_co(bc*2,   1), soB);
        G::load(As[toc][0], g.a, a_co(br*2,   1), soA);
        G::load(Bs[toc][1], g.b, b_co(bc*2+1, 1), soB);

        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        #pragma unroll 2
        for (int k = 0; k < KI - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, k+1), soA);
            asm volatile("s_waitcnt lgkmcnt(8)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cA, a, b0, cA); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);

            load_b(b1, Bs[tic][1], wn);
            G::load(Bs[tic][0], g.b, b_co(bc*2, k+2), soB);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cB, a, b1, cB); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            G::load(As[tic][0], g.a, a_co(br*2, k+2), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cC, a, b0, cC); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);

            G::load(Bs[tic][1], g.b, b_co(bc*2+1, k+2), soB);
            asm volatile("s_waitcnt vmcnt(6)"); __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_setprio(1); mma_ABt(cD, a, b1, cD); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, KI-1), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cA, a, b0, cA); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cB, a, b1, cB); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            asm volatile("s_waitcnt vmcnt(4)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cC, a, b0, cC); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[toc][0], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cD, a, b1, cD); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
            tic ^= 1; toc ^= 1;
        }

        {
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cA, a, b0, cA); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cB, a, b1, cB); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(cC, a, b0, cC);
            mma_ABt(cD, a, b1, cD);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

    } else if constexpr (L == Layout::RRR) {
        __shared__ ST_row As[2][2];
        __shared__ ST_v2  Bs[2][2];
        A_row_reg a;
        B_col_reg b0, b1;

        constexpr int bptA = ST_row::underlying_subtile_bytes_per_thread;
        constexpr int bpmA = bptA * _NUM_THREADS;
        constexpr int mptA = ST_row::rows * ST_row::cols * sizeof(fp8e4m3) / bpmA;
        uint32_t soA[mptA];
        G::prefill_swizzled_offsets(As[0][0], g.a, soA);

        constexpr int bptB = ST_v2::underlying_subtile_bytes_per_thread;
        constexpr int bpmB = bptB * _NUM_THREADS;
        constexpr int mptB = ST_v2::rows * ST_v2::cols * sizeof(fp8e4m3) / bpmB;
        uint32_t soB[mptB];
        G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

        auto a_co = [&](int s, int k) -> coord<ST_row> { return {0, 0, s, k}; };
        auto b_co = [&](int s, int k) -> coord<ST_v2>  { return {0, 0, k, s}; };

        auto load_a = [&](A_row_reg& dst, ST_row& tile, int wi) {
            auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
            load(dst, sub);
        };
        auto load_b = [&](B_col_reg& dst, ST_v2& tile, int wi) {
            load_col_from_v2_st(dst, tile, wi * RBN);
        };

        int tic = 0, toc = 1;
        G::load(Bs[tic][0], g.b, b_co(bc*2,   0), soB);
        G::load(As[tic][0], g.a, a_co(br*2,   0), soA);
        G::load(Bs[tic][1], g.b, b_co(bc*2+1, 0), soB);
        G::load(As[tic][1], g.a, a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        G::load(Bs[toc][0], g.b, b_co(bc*2,   1), soB);
        G::load(As[toc][0], g.a, a_co(br*2,   1), soA);
        G::load(Bs[toc][1], g.b, b_co(bc*2+1, 1), soB);

        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        #pragma unroll 2
        for (int k = 0; k < KI - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, k+1), soA);
            asm volatile("s_waitcnt lgkmcnt(8)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_AB(cA, a, b0, cA); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);

            load_b(b1, Bs[tic][1], wn);
            G::load(Bs[tic][0], g.b, b_co(bc*2, k+2), soB);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_AB(cB, a, b1, cB); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            G::load(As[tic][0], g.a, a_co(br*2, k+2), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_AB(cC, a, b0, cC); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);

            G::load(Bs[tic][1], g.b, b_co(bc*2+1, k+2), soB);
            asm volatile("s_waitcnt vmcnt(6)"); __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_setprio(1); mma_AB(cD, a, b1, cD); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, KI-1), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_AB(cA, a, b0, cA); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_AB(cB, a, b1, cB); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            asm volatile("s_waitcnt vmcnt(4)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_AB(cC, a, b0, cC); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[toc][0], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_AB(cD, a, b1, cD); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
            tic ^= 1; toc ^= 1;
        }

        {
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_AB(cA, a, b0, cA); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_AB(cB, a, b1, cB); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_AB(cC, a, b0, cC);
            mma_AB(cD, a, b1, cD);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

    } else if constexpr (L == Layout::CRR) {
        __shared__ ST_v2a As[2][2];
        __shared__ ST_v2  Bs[2][2];
        A_col_reg a;
        B_col_reg b0, b1;

        constexpr int bptA = ST_v2a::underlying_subtile_bytes_per_thread;
        constexpr int bpmA = bptA * _NUM_THREADS;
        constexpr int mptA = ST_v2a::rows * ST_v2a::cols * sizeof(fp8e4m3) / bpmA;
        uint32_t soA[mptA];
        G::prefill_swizzled_offsets(As[0][0], g.a, soA);

        constexpr int bptB = ST_v2::underlying_subtile_bytes_per_thread;
        constexpr int bpmB = bptB * _NUM_THREADS;
        constexpr int mptB = ST_v2::rows * ST_v2::cols * sizeof(fp8e4m3) / bpmB;
        uint32_t soB[mptB];
        G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

        auto a_co = [&](int s, int k) -> coord<ST_v2a> { return {0, 0, k, s}; };
        auto b_co = [&](int s, int k) -> coord<ST_v2>  { return {0, 0, k, s}; };

        auto load_a = [&](A_col_reg& dst, ST_v2a& tile, int wi) {
            load_col_from_v2a_st(dst, tile, wi * RBM);
        };
        auto load_b = [&](B_col_reg& dst, ST_v2& tile, int wi) {
            load_col_from_v2_st(dst, tile, wi * RBN);
        };

        int tic = 0, toc = 1;
        G::load(Bs[tic][0], g.b, b_co(bc*2,   0), soB);
        G::load(As[tic][0], g.a, a_co(br*2,   0), soA);
        G::load(Bs[tic][1], g.b, b_co(bc*2+1, 0), soB);
        G::load(As[tic][1], g.a, a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        G::load(Bs[toc][0], g.b, b_co(bc*2,   1), soB);
        G::load(As[toc][0], g.a, a_co(br*2,   1), soA);
        G::load(Bs[toc][1], g.b, b_co(bc*2+1, 1), soB);

        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        #pragma unroll 1
        for (int k = 0; k < KI - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, k+1), soA);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_AtB(cA, a, b0, cA); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);

            load_b(b1, Bs[tic][1], wn);
            G::load(Bs[tic][0], g.b, b_co(bc*2, k+2), soB);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_AtB(cB, a, b1, cB); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            G::load(As[tic][0], g.a, a_co(br*2, k+2), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_AtB(cC, a, b0, cC); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);

            G::load(Bs[tic][1], g.b, b_co(bc*2+1, k+2), soB);
            asm volatile("s_waitcnt vmcnt(6)"); __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_setprio(1); mma_AtB(cD, a, b1, cD); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, KI-1), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_AtB(cA, a, b0, cA); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_AtB(cB, a, b1, cB); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            asm volatile("s_waitcnt vmcnt(4)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_AtB(cC, a, b0, cC); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[toc][0], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_AtB(cD, a, b1, cD); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
            tic ^= 1; toc ^= 1;
        }

        {
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_AtB(cA, a, b0, cA); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_AtB(cB, a, b1, cB); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_AtB(cC, a, b0, cC);
            mma_AtB(cD, a, b1, cD);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }
    }

    // Store Output
    if (wm == 0) __builtin_amdgcn_s_barrier();
    store(g.c, cA, {0, 0, br*WARPS_M*2+wm,         bc*WARPS_N*2+wn});
    store(g.c, cB, {0, 0, br*WARPS_M*2+wm,         bc*WARPS_N*2+WARPS_N+wn});
    store(g.c, cC, {0, 0, br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+wn});
    store(g.c, cD, {0, 0, br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+WARPS_N+wn});
}

template __global__ void gemm_kernel<Layout::RCR>(const layout_globals);
template __global__ void gemm_kernel<Layout::RRR>(const layout_globals);
template __global__ void gemm_kernel<Layout::CRR>(const layout_globals);

template<Layout L>
void dispatch(layout_globals g) {
    gemm_kernel<L><<<g.grid(), g.block(), 0, g.stream>>>(g);
}

PYBIND11_MODULE(tk_fp8_layouts, m) {
    m.doc() = "FP8 GEMM: RCR(mma_ABt), RRR(col_l+mma_AB), CRR(col_l+mma_AtB)";
    py::bind_function<dispatch<Layout::RCR>>(m, "gemm_rcr",
        &layout_globals::a, &layout_globals::b, &layout_globals::c);
    py::bind_function<dispatch<Layout::RRR>>(m, "gemm_rrr",
        &layout_globals::a, &layout_globals::b, &layout_globals::c);
    py::bind_function<dispatch<Layout::CRR>>(m, "gemm_crr",
        &layout_globals::a, &layout_globals::b, &layout_globals::c);
}
