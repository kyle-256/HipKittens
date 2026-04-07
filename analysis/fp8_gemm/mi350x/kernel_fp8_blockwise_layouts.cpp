// FP8 blockwise GEMM — MFMA RCR/RRR/CRR with per-K-block scaling.
// Uses kittens mma_ABt with temp accumulator + scale-add per K-block.
// Scale layout: a_scale [Kb, M], b_scale [Kb, Nb] (pre-transposed).

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLK = 256, BK = 128, HB = BLK / 2;
constexpr int WARPS_M = 2, WARPS_N = 4;
constexpr int _NUM_WARPS = WARPS_M * WARPS_N;
constexpr int _NUM_THREADS = _NUM_WARPS * WARP_THREADS;
constexpr int RBM = BLK / WARPS_M / 2, RBN = BLK / WARPS_N / 2;

using G = kittens::group<_NUM_WARPS>;
using _gl_fp8 = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_bf16 = gl<bf16, -1, -1, -1, -1>;
using _gl_f32 = gl<float, -1, -1, -1, -1>;
using ST_rcr = st_fp8e4m3<HB, BK, st_16x128_v2_s>;
using A_row_reg = rt_fp8e4m3<RBM, BK, row_l, rt_16x128_s>;
using B_row_reg = rt_fp8e4m3<RBN, BK, row_l, rt_16x128_s>;
using acc_tile = rt_fl<RBM, RBN, col_l, rt_16x16_s>;
using B_col_reg = rt_fp8e4m3<BK, RBN, col_l, rt_128x16_s>;
using A_col_reg = rt_fp8e4m3<BK, RBM, col_l, rt_128x16_s>;

// Drain pb → accumulator tile, zero pb.
#define _BW_SFZ(pbarr, c_tile, sv_base, bs) do { \
    float *_pf=reinterpret_cast<float*>(pbarr), \
          *_cf=reinterpret_cast<float*>(c_tile.data); \
    _cf[0]+=_pf[0]*((sv_base)[0]*(bs)); _cf[1]+=_pf[1]*((sv_base)[1]*(bs)); \
    _cf[2]+=_pf[2]*((sv_base)[2]*(bs)); _cf[3]+=_pf[3]*((sv_base)[3]*(bs)); \
    _pf[0]=_pf[1]=_pf[2]=_pf[3]=0.f; \
} while(0)

// Quad-buffered 8-MFMA: 4-MFMA gap between write and drain.
// Costs 16 VGPRs for buffers (vs 12 for triple). Better latency hiding.
#define MMA_ABT_BSCALE(c, a, b, sv, bs) do { \
    float2 _b0[2]={}, _b1[2]={}, _b2[2]={}, _b3[2]={}; \
    mfma1616128(_b0, (a).tiles[0][0].data, (b).tiles[0][0].data, _b0); \
    mfma1616128(_b1, (a).tiles[0][0].data, (b).tiles[1][0].data, _b1); \
    mfma1616128(_b2, (a).tiles[1][0].data, (b).tiles[0][0].data, _b2); \
    mfma1616128(_b3, (a).tiles[1][0].data, (b).tiles[1][0].data, _b3); \
    _BW_SFZ(_b0, (c).tiles[0][0], &(sv)[0], bs); \
    mfma1616128(_b0, (a).tiles[2][0].data, (b).tiles[0][0].data, _b0); \
    _BW_SFZ(_b1, (c).tiles[0][1], &(sv)[0], bs); \
    mfma1616128(_b1, (a).tiles[2][0].data, (b).tiles[1][0].data, _b1); \
    _BW_SFZ(_b2, (c).tiles[1][0], &(sv)[4], bs); \
    mfma1616128(_b2, (a).tiles[3][0].data, (b).tiles[0][0].data, _b2); \
    _BW_SFZ(_b3, (c).tiles[1][1], &(sv)[4], bs); \
    mfma1616128(_b3, (a).tiles[3][0].data, (b).tiles[1][0].data, _b3); \
    _BW_SFZ(_b0, (c).tiles[2][0], &(sv)[8], bs); \
    _BW_SFZ(_b1, (c).tiles[2][1], &(sv)[8], bs); \
    _BW_SFZ(_b2, (c).tiles[3][0], &(sv)[12], bs); \
    _BW_SFZ(_b3, (c).tiles[3][1], &(sv)[12], bs); \
} while(0)

// Cross-quadrant interleaved 16-MFMA for TWO quadrants sharing the same B tile.
// Doubles the gap between MFMA write and SFZ drain (4-MFMA gap vs 2-MFMA gap).
// Processes (cA, cC) or (cB, cD) simultaneously.
#define MMA_ABT_BSCALE_PAIR(cT, cB_acc, aT, aB_reg, b, svT, svB, bs) do { \
    float2 _t0[2]={}, _t1[2]={}, _t2[2]={}; \
    float2 _b0[2]={}, _b1[2]={}, _b2[2]={}; \
    /* n=0: aT[0]×b[0], aB[0]×b[0], aT[0]×b[1], aB[0]×b[1] */ \
    mfma1616128(_t0, (aT).tiles[0][0].data, (b).tiles[0][0].data, _t0); \
    mfma1616128(_b0, (aB_reg).tiles[0][0].data, (b).tiles[0][0].data, _b0); \
    mfma1616128(_t1, (aT).tiles[0][0].data, (b).tiles[1][0].data, _t1); \
    mfma1616128(_b1, (aB_reg).tiles[0][0].data, (b).tiles[1][0].data, _b1); \
    /* n=1: aT[1]×b[0], aB[1]×b[0] — drain t0,b0 (4-gap) */ \
    mfma1616128(_t2, (aT).tiles[1][0].data, (b).tiles[0][0].data, _t2); \
    mfma1616128(_b2, (aB_reg).tiles[1][0].data, (b).tiles[0][0].data, _b2); \
    _BW_SFZ(_t0, (cT).tiles[0][0], &(svT)[0], bs); \
    _BW_SFZ(_b0, (cB_acc).tiles[0][0], &(svB)[0], bs); \
    /* n=1 cont: aT[1]×b[1], aB[1]×b[1] — drain t1,b1 */ \
    mfma1616128(_t0, (aT).tiles[1][0].data, (b).tiles[1][0].data, _t0); \
    mfma1616128(_b0, (aB_reg).tiles[1][0].data, (b).tiles[1][0].data, _b0); \
    _BW_SFZ(_t1, (cT).tiles[0][1], &(svT)[0], bs); \
    _BW_SFZ(_b1, (cB_acc).tiles[0][1], &(svB)[0], bs); \
    /* n=2: aT[2]×b[0], aB[2]×b[0] — drain t2,b2 */ \
    mfma1616128(_t1, (aT).tiles[2][0].data, (b).tiles[0][0].data, _t1); \
    mfma1616128(_b1, (aB_reg).tiles[2][0].data, (b).tiles[0][0].data, _b1); \
    _BW_SFZ(_t2, (cT).tiles[1][0], &(svT)[4], bs); \
    _BW_SFZ(_b2, (cB_acc).tiles[1][0], &(svB)[4], bs); \
    /* n=2 cont: aT[2]×b[1], aB[2]×b[1] — drain t0,b0 */ \
    mfma1616128(_t2, (aT).tiles[2][0].data, (b).tiles[1][0].data, _t2); \
    mfma1616128(_b2, (aB_reg).tiles[2][0].data, (b).tiles[1][0].data, _b2); \
    _BW_SFZ(_t0, (cT).tiles[1][1], &(svT)[4], bs); \
    _BW_SFZ(_b0, (cB_acc).tiles[1][1], &(svB)[4], bs); \
    /* n=3: aT[3]×b[0], aB[3]×b[0] — drain t1,b1 */ \
    mfma1616128(_t0, (aT).tiles[3][0].data, (b).tiles[0][0].data, _t0); \
    mfma1616128(_b0, (aB_reg).tiles[3][0].data, (b).tiles[0][0].data, _b0); \
    _BW_SFZ(_t1, (cT).tiles[2][0], &(svT)[8], bs); \
    _BW_SFZ(_b1, (cB_acc).tiles[2][0], &(svB)[8], bs); \
    /* n=3 cont: aT[3]×b[1], aB[3]×b[1] — drain t2,b2 */ \
    mfma1616128(_t1, (aT).tiles[3][0].data, (b).tiles[1][0].data, _t1); \
    mfma1616128(_b1, (aB_reg).tiles[3][0].data, (b).tiles[1][0].data, _b1); \
    _BW_SFZ(_t2, (cT).tiles[2][1], &(svT)[8], bs); \
    _BW_SFZ(_b2, (cB_acc).tiles[2][1], &(svB)[8], bs); \
    /* final drain */ \
    _BW_SFZ(_t0, (cT).tiles[3][0], &(svT)[12], bs); \
    _BW_SFZ(_b0, (cB_acc).tiles[3][0], &(svB)[12], bs); \
    _BW_SFZ(_t1, (cT).tiles[3][1], &(svT)[12], bs); \
    _BW_SFZ(_b1, (cB_acc).tiles[3][1], &(svB)[12], bs); \
} while(0)

template<typename RT, int K_HALF>
__device__ __forceinline__ void load_col_from_v2_st_half(
    RT& dst, const ST_rcr& tile, int col_start)
{
    const int lid = kittens::laneid();
    const int row_off = ((lid % 16) / 2) + ((lid / 16) * 16);
    const int col_off = (lid % 2) * 8;
    const uint32_t tile_base = reinterpret_cast<uintptr_t>(&tile.data[0]);
    constexpr int idx = K_HALF * 4;
    const int k_row = row_off + K_HALF * 64;
    const uint32_t stidx = k_row >> 4;
    const uint32_t base_k = tile_base + (stidx << 11) + (stidx << 7) + ((k_row & 15) << 7);
    const uint32_t sw_k   = (k_row & 7) << 4;
    #pragma unroll
    for (int j = 0; j < RT::width; j++) {
        const uint32_t nc = col_start + j * 16 + col_off;
        const uint32_t addr = base_k + (nc ^ sw_k);
        asm volatile(
            "ds_read_b64_tr_b8 %0, %2 offset:0\n"
            "ds_read_b64_tr_b8 %1, %2 offset:1024\n"
            : "=&v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx])),
              "=&v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx + 2]))
            : "v"(addr)
            : "memory"
        );
    }
}

template<typename RT>
__device__ __forceinline__ void load_col_from_v2_st(
    RT& dst, const ST_rcr& tile, int col_start)
{
    load_col_from_v2_st_half<RT, 0>(dst, tile, col_start);
    load_col_from_v2_st_half<RT, 1>(dst, tile, col_start);
}

// Fused mma_AB + scale-drain for RRR (a is row_l, b is col_l).
// Same quad-buffered 8-MFMA pattern as MMA_ABT_BSCALE but
// B tiles indexed as col_l: b.tiles[0][n] instead of b.tiles[n][0].
#define MMA_AB_BSCALE(c, a, b, sv, bs) do { \
    float2 _b0[2]={}, _b1[2]={}, _b2[2]={}, _b3[2]={}; \
    mfma1616128(_b0, (a).tiles[0][0].data, (b).tiles[0][0].data, _b0); \
    mfma1616128(_b1, (a).tiles[0][0].data, (b).tiles[0][1].data, _b1); \
    mfma1616128(_b2, (a).tiles[1][0].data, (b).tiles[0][0].data, _b2); \
    mfma1616128(_b3, (a).tiles[1][0].data, (b).tiles[0][1].data, _b3); \
    _BW_SFZ(_b0, (c).tiles[0][0], &(sv)[0], bs); \
    mfma1616128(_b0, (a).tiles[2][0].data, (b).tiles[0][0].data, _b0); \
    _BW_SFZ(_b1, (c).tiles[0][1], &(sv)[0], bs); \
    mfma1616128(_b1, (a).tiles[2][0].data, (b).tiles[0][1].data, _b1); \
    _BW_SFZ(_b2, (c).tiles[1][0], &(sv)[4], bs); \
    mfma1616128(_b2, (a).tiles[3][0].data, (b).tiles[0][0].data, _b2); \
    _BW_SFZ(_b3, (c).tiles[1][1], &(sv)[4], bs); \
    mfma1616128(_b3, (a).tiles[3][0].data, (b).tiles[0][1].data, _b3); \
    _BW_SFZ(_b0, (c).tiles[2][0], &(sv)[8], bs); \
    _BW_SFZ(_b1, (c).tiles[2][1], &(sv)[8], bs); \
    _BW_SFZ(_b2, (c).tiles[3][0], &(sv)[12], bs); \
    _BW_SFZ(_b3, (c).tiles[3][1], &(sv)[12], bs); \
} while(0)

// Fused mma_ABt + per-column B scale drain for CRR.
// Uses aliased col_l→row_l registers. bsv[0] for n=0 subtiles, bsv[1] for n=1.
#define MMA_ABT_BSCALE_PER_COL(c, a, b, sv, bsv) do { \
    float2 _b0[2]={}, _b1[2]={}, _b2[2]={}, _b3[2]={}; \
    mfma1616128(_b0, (a).tiles[0][0].data, (b).tiles[0][0].data, _b0); \
    mfma1616128(_b1, (a).tiles[0][0].data, (b).tiles[1][0].data, _b1); \
    mfma1616128(_b2, (a).tiles[1][0].data, (b).tiles[0][0].data, _b2); \
    mfma1616128(_b3, (a).tiles[1][0].data, (b).tiles[1][0].data, _b3); \
    _BW_SFZ(_b0, (c).tiles[0][0], &(sv)[0], (bsv)[0]); \
    mfma1616128(_b0, (a).tiles[2][0].data, (b).tiles[0][0].data, _b0); \
    _BW_SFZ(_b1, (c).tiles[0][1], &(sv)[0], (bsv)[1]); \
    mfma1616128(_b1, (a).tiles[2][0].data, (b).tiles[1][0].data, _b1); \
    _BW_SFZ(_b2, (c).tiles[1][0], &(sv)[4], (bsv)[0]); \
    mfma1616128(_b2, (a).tiles[3][0].data, (b).tiles[0][0].data, _b2); \
    _BW_SFZ(_b3, (c).tiles[1][1], &(sv)[4], (bsv)[1]); \
    mfma1616128(_b3, (a).tiles[3][0].data, (b).tiles[1][0].data, _b3); \
    _BW_SFZ(_b0, (c).tiles[2][0], &(sv)[8], (bsv)[0]); \
    _BW_SFZ(_b1, (c).tiles[2][1], &(sv)[8], (bsv)[1]); \
    _BW_SFZ(_b2, (c).tiles[3][0], &(sv)[12], (bsv)[0]); \
    _BW_SFZ(_b3, (c).tiles[3][1], &(sv)[12], (bsv)[1]); \
} while(0)

#define SCALE_ACC_ADD(dst, src, sv, bs) do { \
    _Pragma("unroll") \
    for (int _n = 0; _n < 4; ++_n) { \
        const float _s0 = (sv)[_n*4]   * (bs); \
        const float _s1 = (sv)[_n*4+1] * (bs); \
        const float _s2 = (sv)[_n*4+2] * (bs); \
        const float _s3 = (sv)[_n*4+3] * (bs); \
        _Pragma("unroll") \
        for (int _m = 0; _m < 2; ++_m) { \
            float       *_df = reinterpret_cast<float*>((dst).tiles[_n][_m].data); \
            const float *_sf = reinterpret_cast<const float*>((src).tiles[_n][_m].data); \
            _df[0] += _sf[0] * _s0; _df[1] += _sf[1] * _s1; \
            _df[2] += _sf[2] * _s2; _df[3] += _sf[3] * _s3; \
        } \
    } \
} while(0)

#define SCALE_ACC_ADD_PER_COL(dst, src, sv, bsv) do { \
    _Pragma("unroll") \
    for (int _n = 0; _n < 4; ++_n) { \
        _Pragma("unroll") \
        for (int _m = 0; _m < 2; ++_m) { \
            const float _bs = (bsv)[_m]; \
            const float _s0 = (sv)[_n*4]   * _bs; \
            const float _s1 = (sv)[_n*4+1] * _bs; \
            const float _s2 = (sv)[_n*4+2] * _bs; \
            const float _s3 = (sv)[_n*4+3] * _bs; \
            float       *_df = reinterpret_cast<float*>((dst).tiles[_n][_m].data); \
            const float *_sf = reinterpret_cast<const float*>((src).tiles[_n][_m].data); \
            _df[0] += _sf[0] * _s0; _df[1] += _sf[1] * _s1; \
            _df[2] += _sf[2] * _s2; _df[3] += _sf[3] * _s3; \
        } \
    } \
} while(0)

struct bw_rcr_globals {
    _gl_fp8 a; _gl_fp8 b; _gl_bf16 c;
    _gl_f32 a_scale; _gl_f32 b_scale;
    hipStream_t stream = nullptr;
    int m, n, k, bpr, bpc, ki;
    dim3 grid()  { return dim3(bpr * bpc); }
    dim3 block() { return dim3(_NUM_THREADS); }
};

__global__ __launch_bounds__(_NUM_THREADS, 1)
void gemm_rcr_blockwise_mfma(const bw_rcr_globals g) {
    const int bid = blockIdx.x;
    const int br = bid / g.bpc, bc = bid % g.bpc;
    if (br >= g.bpr || bc >= g.bpc) return;
    const int wm = warpid() / WARPS_N, wn = warpid() % WARPS_N;
    const int r16 = 4 * (laneid() / 16);

    acc_tile cA, cB, cC, cD;
    zero(cA); zero(cB); zero(cC); zero(cD);
    A_row_reg a_reg; B_row_reg b0_reg, b1_reg;

    __shared__ ST_rcr As[2][2], Bs[2][2];
    constexpr int bpt = ST_rcr::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * _NUM_THREADS;
    constexpr int mpt = ST_rcr::rows * ST_rcr::cols * sizeof(fp8e4m3) / bpm;
    uint32_t soA[mpt], soB[mpt];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);
    auto aco = [&](int s, int ki) -> coord<ST_rcr> { return {0,0,s,ki}; };
    auto bco = [&](int s, int ki) -> coord<ST_rcr> { return {0,0,s,ki}; };
    auto la = [&](A_row_reg& d, ST_rcr& t, int w){ load(d, subtile_inplace<RBM,BK>(t,{w,0})); };
    auto lb = [&](B_row_reg& d, ST_rcr& t, int w){ load(d, subtile_inplace<RBN,BK>(t,{w,0})); };
    const int mt = br*BLK + wm*RBM, mb_ = br*BLK + HB + wm*RBM;

    // === Init ===
    G::load(As[0][0], g.a, aco(br*2,0), soA);
    G::load(As[0][1], g.a, aco(br*2+1,0), soA);
    G::load(Bs[0][0], g.b, bco(bc*2,0), soB);
    G::load(Bs[0][1], g.b, bco(bc*2+1,0), soB);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    // === Main loop ===
    int tic = 0, toc = 1;
    for (int k = 0; k < g.ki; ++k, tic ^= 1, toc ^= 1) {
        const float bl  = g.b_scale[coord<>(k, bc*2)];
        const float br2 = g.b_scale[coord<>(k, bc*2+1)];
        float svt[16];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int d = 0; d < 4; ++d)
                svt[i*4+d] = g.a_scale[coord<>(k, mt + i*16 + r16 + d)];

        float svb[16];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int d = 0; d < 4; ++d)
                svb[i*4+d] = g.a_scale[coord<>(k, mb_ + i*16 + r16 + d)];

        if (k + 1 < g.ki) {
            G::load(As[toc][0], g.a, aco(br*2,   k+1), soA);
            G::load(As[toc][1], g.a, aco(br*2+1, k+1), soA);
            G::load(Bs[toc][0], g.b, bco(bc*2,   k+1), soB);
            G::load(Bs[toc][1], g.b, bco(bc*2+1, k+1), soB);
        }

        lb(b0_reg, Bs[tic][0], wn);
        la(a_reg,  As[tic][0], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        MMA_ABT_BSCALE(cA, a_reg, b0_reg, svt, bl);
        __builtin_amdgcn_s_setprio(0);

        lb(b1_reg, Bs[tic][1], wn);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        MMA_ABT_BSCALE(cB, a_reg, b1_reg, svt, br2);
        __builtin_amdgcn_s_setprio(0);

        la(a_reg, As[tic][1], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        MMA_ABT_BSCALE(cC, a_reg, b0_reg, svb, bl);
        __builtin_amdgcn_s_setprio(0);

        __builtin_amdgcn_s_setprio(1);
        MMA_ABT_BSCALE(cD, a_reg, b1_reg, svb, br2);
        __builtin_amdgcn_s_setprio(0);

        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }

    if (wm == 0) __builtin_amdgcn_s_barrier();
    store(g.c, cA, {0,0, br*WARPS_M*2+wm,         bc*WARPS_N*2+wn});
    store(g.c, cB, {0,0, br*WARPS_M*2+wm,         bc*WARPS_N*2+WARPS_N+wn});
    store(g.c, cC, {0,0, br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+wn});
    store(g.c, cD, {0,0, br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+WARPS_N+wn});
}

__global__ void gemm_rcr_blockwise_tail(const bw_rcr_globals g, int fm, int fn, int fk) {
    int row=blockIdx.y*blockDim.y+threadIdx.y, col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=g.m||col>=g.n) return;
    bool inf=row<fm&&col<fn; if(inf&&fk>=g.k) return;
    int nb=col/BK, k0=inf?fk:0; float acc=0.f;
    for(int kk=k0;kk<g.k;++kk){ int ki=kk/BK;
        acc+=base_types::convertor<float,fp8e4m3>::convert(g.a[coord<>(row,kk)])
            *base_types::convertor<float,fp8e4m3>::convert(g.b[coord<>(col,kk)])
            *g.a_scale[coord<>(ki,row)]*g.b_scale[coord<>(ki,nb)];}
    if(inf){float pv=base_types::convertor<float,bf16>::convert(g.c[coord<>(row,col)]);
        g.c[coord<>(row,col)]=base_types::convertor<bf16,float>::convert(pv+acc);}
    else g.c[coord<>(row,col)]=base_types::convertor<bf16,float>::convert(acc);
}

void dispatch_gemm_rcr_blockwise(bw_rcr_globals g) {
    g.m=g.c.rows(); g.n=g.c.cols(); g.k=g.a.cols();
    g.bpr=g.m/BLK; g.bpc=g.n/BLK; g.ki=g.k/BK;
    int fm=g.bpr*BLK,fn=g.bpc*BLK,fk=g.ki*BK;
    if(g.bpr>0&&g.bpc>0&&g.ki>=1)
        gemm_rcr_blockwise_mfma<<<g.grid(),g.block(),0,g.stream>>>(g);
    if(fm!=g.m||fn!=g.n||fk!=g.k){dim3 tb(16,16),tg((g.n+15)/16,(g.m+15)/16);
        gemm_rcr_blockwise_tail<<<tg,tb,0,g.stream>>>(g,fm,fn,fk);}
}

struct bw_rrr_globals {
    _gl_fp8 a; _gl_fp8 b; _gl_bf16 c;
    _gl_f32 a_scale; _gl_f32 b_scale;
    hipStream_t stream = nullptr;
    int m, n, k, bpr, bpc, ki;
    dim3 grid()  { return dim3(bpr * bpc); }
    dim3 block() { return dim3(_NUM_THREADS); }
};

__global__ __launch_bounds__(_NUM_THREADS, 1)
void gemm_rrr_blockwise_mfma(const bw_rrr_globals g) {
    const int bid = blockIdx.x;
    const int br = bid / g.bpc, bc = bid % g.bpc;
    if (br >= g.bpr || bc >= g.bpc) return;
    const int wm = warpid() / WARPS_N, wn = warpid() % WARPS_N;
    const int r16 = 4 * (laneid() / 16);

    acc_tile cA, cB, cC, cD;
    zero(cA); zero(cB); zero(cC); zero(cD);
    A_row_reg a_reg; B_col_reg b0_reg, b1_reg;

    __shared__ ST_rcr As[2][2], Bs[2][2];
    constexpr int bpt = ST_rcr::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * _NUM_THREADS;
    constexpr int mpt = ST_rcr::rows * ST_rcr::cols * sizeof(fp8e4m3) / bpm;
    uint32_t soA[mpt], soB[mpt];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);
    auto aco = [&](int s, int ki) -> coord<ST_rcr> { return {0,0,s,ki}; };
    auto bco = [&](int s, int ki) -> coord<ST_rcr> { return {0,0,ki,s}; };
    auto la = [&](A_row_reg& d, ST_rcr& t, int w){ load(d, subtile_inplace<RBM,BK>(t,{w,0})); };
    auto lb = [&](B_col_reg& d, ST_rcr& t, int w){ load_col_from_v2_st(d, t, w * RBN); };
    const int mt = br*BLK + wm*RBM, mb_ = br*BLK + HB + wm*RBM;

    // === Init ===
    G::load(As[0][0], g.a, aco(br*2,0), soA);
    G::load(As[0][1], g.a, aco(br*2+1,0), soA);
    G::load(Bs[0][0], g.b, bco(bc*2,0), soB);
    G::load(Bs[0][1], g.b, bco(bc*2+1,0), soB);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    // === Main loop ===
    int tic = 0, toc = 1;
    for (int k = 0; k < g.ki; ++k, tic ^= 1, toc ^= 1) {
        const float bl  = g.b_scale[coord<>(k, bc*2)];
        const float br2 = g.b_scale[coord<>(k, bc*2+1)];
        float svt[16];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int d = 0; d < 4; ++d)
                svt[i*4+d] = g.a_scale[coord<>(k, mt + i*16 + r16 + d)];

        float svb[16];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int d = 0; d < 4; ++d)
                svb[i*4+d] = g.a_scale[coord<>(k, mb_ + i*16 + r16 + d)];

        if (k + 1 < g.ki) {
            G::load(As[toc][0], g.a, aco(br*2,   k+1), soA);
            G::load(As[toc][1], g.a, aco(br*2+1, k+1), soA);
            G::load(Bs[toc][0], g.b, bco(bc*2,   k+1), soB);
            G::load(Bs[toc][1], g.b, bco(bc*2+1, k+1), soB);
        }

        lb(b0_reg, Bs[tic][0], wn);
        la(a_reg,  As[tic][0], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        MMA_AB_BSCALE(cA, a_reg, b0_reg, svt, bl);
        __builtin_amdgcn_s_setprio(0);

        lb(b1_reg, Bs[tic][1], wn);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        MMA_AB_BSCALE(cB, a_reg, b1_reg, svt, br2);
        __builtin_amdgcn_s_setprio(0);

        la(a_reg, As[tic][1], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        MMA_AB_BSCALE(cC, a_reg, b0_reg, svb, bl);
        __builtin_amdgcn_s_setprio(0);

        __builtin_amdgcn_s_setprio(1);
        MMA_AB_BSCALE(cD, a_reg, b1_reg, svb, br2);
        __builtin_amdgcn_s_setprio(0);

        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }

    if (wm == 0) __builtin_amdgcn_s_barrier();
    store(g.c, cA, {0,0, br*WARPS_M*2+wm,         bc*WARPS_N*2+wn});
    store(g.c, cB, {0,0, br*WARPS_M*2+wm,         bc*WARPS_N*2+WARPS_N+wn});
    store(g.c, cC, {0,0, br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+wn});
    store(g.c, cD, {0,0, br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+WARPS_N+wn});
}

__global__ void gemm_rrr_blockwise_tail(const bw_rrr_globals g, int fm, int fn, int fk) {
    int row=blockIdx.y*blockDim.y+threadIdx.y, col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=g.m||col>=g.n) return;
    bool inf=row<fm&&col<fn; if(inf&&fk>=g.k) return;
    int nb=col/BK, k0=inf?fk:0; float acc=0.f;
    for(int kk=k0;kk<g.k;++kk){ int ki=kk/BK;
        acc+=base_types::convertor<float,fp8e4m3>::convert(g.a[coord<>(row,kk)])
            *base_types::convertor<float,fp8e4m3>::convert(g.b[coord<>(kk,col)])
            *g.a_scale[coord<>(ki,row)]*g.b_scale[coord<>(ki,nb)];}
    if(inf){float pv=base_types::convertor<float,bf16>::convert(g.c[coord<>(row,col)]);
        g.c[coord<>(row,col)]=base_types::convertor<bf16,float>::convert(pv+acc);}
    else g.c[coord<>(row,col)]=base_types::convertor<bf16,float>::convert(acc);
}

void dispatch_gemm_rrr_blockwise(bw_rrr_globals g) {
    g.m=g.c.rows(); g.n=g.c.cols(); g.k=g.a.cols();
    g.bpr=g.m/BLK; g.bpc=g.n/BLK; g.ki=g.k/BK;
    int fm=g.bpr*BLK,fn=g.bpc*BLK,fk=g.ki*BK;
    if(g.bpr>0&&g.bpc>0&&g.ki>=1)
        gemm_rrr_blockwise_mfma<<<g.grid(),g.block(),0,g.stream>>>(g);
    if(fm!=g.m||fn!=g.n||fk!=g.k){dim3 tb(16,16),tg((g.n+15)/16,(g.m+15)/16);
        gemm_rrr_blockwise_tail<<<tg,tb,0,g.stream>>>(g,fm,fn,fk);}
}

struct bw_crr_globals {
    _gl_fp8 a; _gl_fp8 b; _gl_bf16 c;
    _gl_f32 a_scale; _gl_f32 b_scale;
    hipStream_t stream = nullptr;
    int m, n, k, bpr, bpc, ki;
    dim3 grid()  { return dim3(bpr * bpc); }
    dim3 block() { return dim3(_NUM_THREADS); }
};

__global__ __launch_bounds__(_NUM_THREADS, 1)
void gemm_crr_blockwise_mfma(const bw_crr_globals g) {
    const int bid = blockIdx.x;
    const int br = bid / g.bpc, bc = bid % g.bpc;
    if (br >= g.bpr || bc >= g.bpc) return;
    const int wm = warpid() / WARPS_N, wn = warpid() % WARPS_N;
    const int r16 = 4 * (laneid() / 16);
    const int c16 = laneid() % 16;

    acc_tile cA, cB, cC, cD;
    zero(cA); zero(cB); zero(cC); zero(cD);
    A_col_reg a_reg; B_col_reg b0_reg, b1_reg;

    __shared__ ST_rcr As[2][2], Bs[2][2];
    constexpr int bpt = ST_rcr::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * _NUM_THREADS;
    constexpr int mpt = ST_rcr::rows * ST_rcr::cols * sizeof(fp8e4m3) / bpm;
    uint32_t soA[mpt], soB[mpt];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);
    auto aco = [&](int s, int ki) -> coord<ST_rcr> { return {0,0,ki,s}; };
    auto bco = [&](int s, int ki) -> coord<ST_rcr> { return {0,0,ki,s}; };
    auto la = [&](A_col_reg& d, ST_rcr& t, int w){ load_col_from_v2_st(d, t, w * RBM); };
    auto lb = [&](B_col_reg& d, ST_rcr& t, int w){ load_col_from_v2_st(d, t, w * RBN); };
    const int mt = br*BLK + wm*RBM, mb_ = br*BLK + HB + wm*RBM;
    const int nt = bc*BLK + wn*RBN, nb_ = bc*BLK + HB + wn*RBN;

    // === Init ===
    G::load(As[0][0], g.a, aco(br*2,0), soA);
    G::load(As[0][1], g.a, aco(br*2+1,0), soA);
    G::load(Bs[0][0], g.b, bco(bc*2,0), soB);
    G::load(Bs[0][1], g.b, bco(bc*2+1,0), soB);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    // === Main loop ===
    int tic = 0, toc = 1;
    for (int k = 0; k < g.ki; ++k, tic ^= 1, toc ^= 1) {
        float bsvA[2], bsvB[2];
        bsvA[0] = g.b_scale[coord<>(k, nt  + c16)];
        bsvA[1] = g.b_scale[coord<>(k, nt  + 16 + c16)];
        bsvB[0] = g.b_scale[coord<>(k, nb_ + c16)];
        bsvB[1] = g.b_scale[coord<>(k, nb_ + 16 + c16)];

        float svt[16];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int d = 0; d < 4; ++d)
                svt[i*4+d] = g.a_scale[coord<>(k, mt + i*16 + r16 + d)];

        float svb[16];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int d = 0; d < 4; ++d)
                svb[i*4+d] = g.a_scale[coord<>(k, mb_ + i*16 + r16 + d)];

        if (k + 1 < g.ki) {
            G::load(As[toc][0], g.a, aco(br*2,   k+1), soA);
            G::load(As[toc][1], g.a, aco(br*2+1, k+1), soA);
            G::load(Bs[toc][0], g.b, bco(bc*2,   k+1), soB);
            G::load(Bs[toc][1], g.b, bco(bc*2+1, k+1), soB);
        }

        lb(b0_reg, Bs[tic][0], wn);
        la(a_reg,  As[tic][0], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        { const auto& ar = reinterpret_cast<const A_row_reg&>(a_reg);
          const auto& br_ = reinterpret_cast<const B_row_reg&>(b0_reg);
          MMA_ABT_BSCALE_PER_COL(cA, ar, br_, svt, bsvA); }
        __builtin_amdgcn_s_setprio(0);

        lb(b1_reg, Bs[tic][1], wn);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        { const auto& ar = reinterpret_cast<const A_row_reg&>(a_reg);
          const auto& br_ = reinterpret_cast<const B_row_reg&>(b1_reg);
          MMA_ABT_BSCALE_PER_COL(cB, ar, br_, svt, bsvB); }
        __builtin_amdgcn_s_setprio(0);

        la(a_reg, As[tic][1], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        { const auto& ar = reinterpret_cast<const A_row_reg&>(a_reg);
          const auto& br_ = reinterpret_cast<const B_row_reg&>(b0_reg);
          MMA_ABT_BSCALE_PER_COL(cC, ar, br_, svb, bsvA); }
        __builtin_amdgcn_s_setprio(0);

        __builtin_amdgcn_s_setprio(1);
        { const auto& ar = reinterpret_cast<const A_row_reg&>(a_reg);
          const auto& br_ = reinterpret_cast<const B_row_reg&>(b1_reg);
          MMA_ABT_BSCALE_PER_COL(cD, ar, br_, svb, bsvB); }
        __builtin_amdgcn_s_setprio(0);

        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }

    if (wm == 0) __builtin_amdgcn_s_barrier();
    store(g.c, cA, {0,0, br*WARPS_M*2+wm,         bc*WARPS_N*2+wn});
    store(g.c, cB, {0,0, br*WARPS_M*2+wm,         bc*WARPS_N*2+WARPS_N+wn});
    store(g.c, cC, {0,0, br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+wn});
    store(g.c, cD, {0,0, br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+WARPS_N+wn});
}

__global__ void gemm_crr_blockwise_tail(const bw_crr_globals g, int fm, int fn, int fk) {
    int row=blockIdx.y*blockDim.y+threadIdx.y, col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=g.m||col>=g.n) return;
    bool inf=row<fm&&col<fn; if(inf&&fk>=g.k) return;
    int k0=inf?fk:0; float acc=0.f;
    for(int kk=k0;kk<g.k;++kk){ int ki=kk/BK;
        acc+=base_types::convertor<float,fp8e4m3>::convert(g.a[coord<>(kk,row)])
            *base_types::convertor<float,fp8e4m3>::convert(g.b[coord<>(kk,col)])
            *g.a_scale[coord<>(ki,row)]*g.b_scale[coord<>(ki,col)];}
    if(inf){float pv=base_types::convertor<float,bf16>::convert(g.c[coord<>(row,col)]);
        g.c[coord<>(row,col)]=base_types::convertor<bf16,float>::convert(pv+acc);}
    else g.c[coord<>(row,col)]=base_types::convertor<bf16,float>::convert(acc);
}

void dispatch_gemm_crr_blockwise(bw_crr_globals g) {
    g.m=g.c.rows(); g.n=g.c.cols(); g.k=g.a.rows();
    g.bpr=g.m/BLK; g.bpc=g.n/BLK; g.ki=g.k/BK;
    int fm=g.bpr*BLK,fn=g.bpc*BLK,fk=g.ki*BK;
    if(g.bpr>0&&g.bpc>0&&g.ki>=1)
        gemm_crr_blockwise_mfma<<<g.grid(),g.block(),0,g.stream>>>(g);
    if(fm!=g.m||fn!=g.n||fk!=g.k){dim3 tb(16,16),tg((g.n+15)/16,(g.m+15)/16);
        gemm_crr_blockwise_tail<<<tg,tb,0,g.stream>>>(g,fm,fn,fk);}
}

PYBIND11_MODULE(tk_fp8_blockwise_layouts, m) {
    m.doc()="FP8 blockwise GEMM: MFMA RCR/RRR/CRR with per-K-block scaling";
    py::bind_function<dispatch_gemm_rcr_blockwise>(m,"gemm_rcr_blockwise",
        &bw_rcr_globals::a,&bw_rcr_globals::b,&bw_rcr_globals::c,&bw_rcr_globals::a_scale,&bw_rcr_globals::b_scale);
    py::bind_function<dispatch_gemm_rrr_blockwise>(m,"gemm_rrr_blockwise",
        &bw_rrr_globals::a,&bw_rrr_globals::b,&bw_rrr_globals::c,&bw_rrr_globals::a_scale,&bw_rrr_globals::b_scale);
    py::bind_function<dispatch_gemm_crr_blockwise>(m,"gemm_crr_blockwise",
        &bw_crr_globals::a,&bw_crr_globals::b,&bw_crr_globals::c,&bw_crr_globals::a_scale,&bw_crr_globals::b_scale);
}
