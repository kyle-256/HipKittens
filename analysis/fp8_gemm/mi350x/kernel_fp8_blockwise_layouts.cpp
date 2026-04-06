// FP8 blockwise GEMM — MFMA (RCR/NT), scalar reference (RRR/CRR).
// MFMA-instruction-level double-buffered partial (CK-style):
//   2 × rt_base partial (~4 VGPR), scale_fma interleaved between MFMA calls.
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

using base_acc = rt_base<float, ducks::rt_layout::col, ducks::rt_shape::rt_16x16>;
using base_a   = rt_base<fp8e4m3, ducks::rt_layout::row, ducks::rt_shape::rt_16x128>;
using base_b   = rt_base<fp8e4m3, ducks::rt_layout::row, ducks::rt_shape::rt_16x128>;

// Scale one base-tile partial → FMA into main → zero partial
__device__ __forceinline__ void base_scale_fma_zero(
    float2 (&c)[2], float2 (&p)[2],
    const _gl_f32& asc, float bs, int ki, int m_base, int r16)
{
    float* pf = reinterpret_cast<float*>(p);
    float* cf = reinterpret_cast<float*>(c);
    #pragma unroll
    for (int d = 0; d < 4; ++d) {
        float s = asc[coord<>(ki, m_base + r16 + d)] * bs;
        cf[d] += pf[d] * s;
        pf[d] = 0.f;
    }
}

// mma_ABt with per-ki blockscale fused at MFMA-instruction level.
// Double-buffered partial: p[2][2] (2 slots × 2 float2 each = 4 VGPR).
// For each MFMA call, accumulate into p[buf]; scale p[buf^1] from previous call.
__device__ __forceinline__ void mma_ABt_blockscale(
    acc_tile& c,
    const A_row_reg& a, const B_row_reg& b,
    const _gl_f32& asc, float bs,
    int ki, int m_base, int r16)
{
    float2 pbuf[2][2] = {};  // double-buffered partial, zeroed

    constexpr int H = acc_tile::height;  // 4
    constexpr int W = acc_tile::width;   // 2
    int buf = 0;
    int prev_n = 0, prev_m = 0;

    #pragma unroll
    for (int n = 0; n < H; ++n) {
        #pragma unroll
        for (int m = 0; m < W; ++m) {
            // MFMA into pbuf[buf]
            mfma1616128(pbuf[buf], a.tiles[n][0].data, b.tiles[m][0].data, pbuf[buf]);

            // Scale pbuf[buf^1] from PREVIOUS MFMA (has had ≥1 MFMA gap)
            if (n > 0 || m > 0) {
                base_scale_fma_zero(
                    c.tiles[prev_n][prev_m].data, pbuf[buf ^ 1],
                    asc, bs, ki, m_base + prev_n * 16, r16);
            }
            prev_n = n;
            prev_m = m;
            buf ^= 1;
        }
    }
    // Scale the last MFMA's partial
    base_scale_fma_zero(
        c.tiles[prev_n][prev_m].data, pbuf[buf ^ 1],
        asc, bs, ki, m_base + prev_n * 16, r16);
}

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

    G::load(As[0][0], g.a, aco(br*2,0), soA);
    G::load(As[0][1], g.a, aco(br*2+1,0), soA);
    G::load(Bs[0][0], g.b, bco(bc*2,0), soB);
    G::load(Bs[0][1], g.b, bco(bc*2+1,0), soB);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    int tic = 0, toc = 1;
    for (int k = 0; k < g.ki; ++k, tic ^= 1, toc ^= 1) {
        if (k + 1 < g.ki) {
            G::load(As[toc][0], g.a, aco(br*2,   k+1), soA);
            G::load(As[toc][1], g.a, aco(br*2+1, k+1), soA);
            G::load(Bs[toc][0], g.b, bco(bc*2,   k+1), soB);
            G::load(Bs[toc][1], g.b, bco(bc*2+1, k+1), soB);
        }
        const float bl = g.b_scale[coord<>(k, bc*2)];
        const float br2 = g.b_scale[coord<>(k, bc*2+1)];

        lb(b0_reg, Bs[tic][0], wn);
        la(a_reg,  As[tic][0], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt_blockscale(cA, a_reg, b0_reg, g.a_scale, bl, k, mt, r16);
        __builtin_amdgcn_s_setprio(0);

        lb(b1_reg, Bs[tic][1], wn);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt_blockscale(cB, a_reg, b1_reg, g.a_scale, br2, k, mt, r16);
        __builtin_amdgcn_s_setprio(0);

        la(a_reg, As[tic][1], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt_blockscale(cC, a_reg, b0_reg, g.a_scale, bl, k, mb_, r16);
        __builtin_amdgcn_s_setprio(0);

        __builtin_amdgcn_s_setprio(1);
        mma_ABt_blockscale(cD, a_reg, b1_reg, g.a_scale, br2, k, mb_, r16);
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

struct bw_rrr_globals{_gl_fp8 a,b;_gl_bf16 c;_gl_f32 a_scale,b_scale;hipStream_t stream=nullptr;};
__global__ void gemm_rrr_bw_s(bw_rrr_globals g,int M,int N,int K){
    int idx=blockIdx.x*blockDim.x+threadIdx.x,m=idx/N,n=idx%N;
    if(m>=M||n>=N)return;int Kb=(K+BK-1)/BK,nb=n/BK;float acc=0.f;
    for(int ki=0;ki<Kb;++ki){int k0=ki*BK,k1=k0+BK<K?k0+BK:K;float d=0.f;
        for(int kk=k0;kk<k1;++kk)d+=base_types::convertor<float,fp8e4m3>::convert(g.a[coord<>(m,kk)])*base_types::convertor<float,fp8e4m3>::convert(g.b[coord<>(kk,n)]);
        acc+=d*g.a_scale[coord<>(ki,m)]*g.b_scale[coord<>(ki,nb)];}
    g.c[coord<>(m,n)]=base_types::convertor<bf16,float>::convert(acc);}
void dispatch_gemm_rrr_blockwise(bw_rrr_globals g){int M=g.c.rows(),N=g.c.cols(),K=g.a.cols();gemm_rrr_bw_s<<<((M*N)+255)/256,256,0,g.stream>>>(g,M,N,K);}

struct bw_crr_globals{_gl_fp8 a,b;_gl_bf16 c;_gl_f32 a_scale,b_scale;hipStream_t stream=nullptr;};
__global__ void gemm_crr_bw_s(bw_crr_globals g,int M,int N,int K){
    int idx=blockIdx.x*blockDim.x+threadIdx.x,m=idx/N,n=idx%N;
    if(m>=M||n>=N)return;int Kb=(K+BK-1)/BK;float acc=0.f;
    for(int ki=0;ki<Kb;++ki){int k0=ki*BK,k1=k0+BK<K?k0+BK:K;float d=0.f;
        for(int kk=k0;kk<k1;++kk)d+=base_types::convertor<float,fp8e4m3>::convert(g.a[coord<>(kk,m)])*base_types::convertor<float,fp8e4m3>::convert(g.b[coord<>(kk,n)]);
        acc+=d*g.a_scale[coord<>(ki,m)]*g.b_scale[coord<>(ki,n)];}
    g.c[coord<>(m,n)]=base_types::convertor<bf16,float>::convert(acc);}
void dispatch_gemm_crr_blockwise(bw_crr_globals g){int M=g.c.rows(),N=g.c.cols(),K=g.a.rows();gemm_crr_bw_s<<<((M*N)+255)/256,256,0,g.stream>>>(g,M,N,K);}

PYBIND11_MODULE(tk_fp8_blockwise_layouts, m) {
    m.doc()="FP8 blockwise GEMM: MFMA RCR (CK-style double-buffered scale), scalar RRR/CRR";
    py::bind_function<dispatch_gemm_rcr_blockwise>(m,"gemm_rcr_blockwise",
        &bw_rcr_globals::a,&bw_rcr_globals::b,&bw_rcr_globals::c,&bw_rcr_globals::a_scale,&bw_rcr_globals::b_scale);
    py::bind_function<dispatch_gemm_rrr_blockwise>(m,"gemm_rrr_blockwise",
        &bw_rrr_globals::a,&bw_rrr_globals::b,&bw_rrr_globals::c,&bw_rrr_globals::a_scale,&bw_rrr_globals::b_scale);
    py::bind_function<dispatch_gemm_crr_blockwise>(m,"gemm_crr_blockwise",
        &bw_crr_globals::a,&bw_crr_globals::b,&bw_crr_globals::c,&bw_crr_globals::a_scale,&bw_crr_globals::b_scale);
}
