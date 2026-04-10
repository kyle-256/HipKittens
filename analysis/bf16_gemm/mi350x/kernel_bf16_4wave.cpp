#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLK = 256, HB = BLK / 2, KS = 64;
constexpr int WM = 2, WN = 2, NW = WM * WN;
constexpr int NT = NW * WARP_THREADS;
constexpr int HRM = BLK / (2 * WM);  // 64
constexpr int HRN = BLK / (2 * WN);  // 64

using _gl = gl<bf16, -1, -1, -1, -1>;
using G4 = group<NW>;

using ST = st_bf<HB, KS, st_16x32_s>;
using A_rt = rt_bf<HRM, KS, row_l, rt_16x32_s>;
using B_rt = rt_bf<HRN, KS, row_l, rt_16x32_s>;
using C_rt = rt_fl<HRM, HRN, col_l, rt_16x16_s>;

struct globals {
    _gl a, b, c;
    hipStream_t stream;
    int m, n, k, ki, bpr, bpc, group_m;
};

__global__ __launch_bounds__(NT, 1)
void bf16_rcr_4wave(const globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    ST (&As)[2][2] = al.allocate<ST, 2, 2>();
    ST (&Bs)[2][2] = al.allocate<ST, 2, 2>();

    int wgid = blockIdx.x;
    wgid = chiplet_transform_chunked(wgid, gridDim.x, NUM_XCDS, 64);
    const int nwig = g.group_m * g.bpc;
    int gid = wgid / nwig;
    int fpm = gid * g.group_m;
    int gsm = min(g.bpr - fpm, g.group_m);
    if (gsm <= 0) return;
    int br = fpm + ((wgid % nwig) % gsm);
    int bc = (wgid % nwig) / gsm;
    if (br >= g.bpr || bc >= g.bpc) return;

    const int wm = warpid() / WN, wn = warpid() % WN;
    A_rt a; B_rt b0, b1;
    C_rt c[2][2];
    zero(c[0][0]); zero(c[0][1]); zero(c[1][0]); zero(c[1][1]);

    constexpr int bpt = ST::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * NT;
    constexpr int mpt = ST::rows * ST::cols * sizeof(bf16) / bpm;
    uint32_t soA[mpt], soB[mpt];
    G4::prefill_swizzled_offsets(As[0][0], g.a, soA);
    G4::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

    auto ac = [&](int s, int k) -> coord<ST> { return {0,0,s,k}; };
    auto bc_ = [&](int s, int k) -> coord<ST> { return {0,0,s,k}; };

    auto load_a = [&](A_rt& d, ST& t, int w) {
        auto sub = subtile_inplace<HRM, KS>(t, {w, 0}); load(d, sub);
    };
    auto load_b = [&](B_rt& d, ST& t, int w) {
        auto sub = subtile_inplace<HRN, KS>(t, {w, 0}); load(d, sub);
    };

    int cur = 0, nxt = 1;
    G4::load(As[cur][0], g.a, ac(br*2, 0), soA);
    G4::load(Bs[cur][0], g.b, bc_(bc*2, 0), soB);
    G4::load(As[cur][1], g.a, ac(br*2+1, 0), soA);
    G4::load(Bs[cur][1], g.b, bc_(bc*2+1, 0), soB);

    if (wm == 1) __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    G4::load(As[nxt][0], g.a, ac(br*2, 1), soA);
    G4::load(Bs[nxt][0], g.b, bc_(bc*2, 1), soB);
    G4::load(As[nxt][1], g.a, ac(br*2+1, 1), soA);
    G4::load(Bs[nxt][1], g.b, bc_(bc*2+1, 1), soB);

    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    auto main_loop_iter = [&](int tile) {
        // K-tile 0 from buf[0]: consume, then prefetch tile+2 INTO buf[0]
        load_a(a, As[0][0], wm);
        load_b(b0, Bs[0][0], wn);
        load_b(b1, Bs[0][1], wn);

        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(c[0][0], a, b0, c[0][0]);
        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][1], a, b1, c[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a(a, As[0][1], wm);
        G4::load(As[0][0], g.a, ac(br*2, tile+2), soA);
        G4::load(Bs[0][0], g.b, bc_(bc*2, tile+2), soB);

        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(c[1][0], a, b0, c[1][0]);
        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][1], a, b1, c[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        G4::load(As[0][1], g.a, ac(br*2+1, tile+2), soA);
        G4::load(Bs[0][1], g.b, bc_(bc*2+1, tile+2), soB);

        // K-tile 1 from buf[1]: consume, then prefetch tile+3 INTO buf[1]
        load_a(a, As[1][0], wm);
        load_b(b0, Bs[1][0], wn);
        load_b(b1, Bs[1][1], wn);

        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(c[0][0], a, b0, c[0][0]);
        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][1], a, b1, c[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a(a, As[1][1], wm);
        G4::load(As[1][0], g.a, ac(br*2, tile+3), soA);
        G4::load(Bs[1][0], g.b, bc_(bc*2, tile+3), soB);

        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(c[1][0], a, b0, c[1][0]);
        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][1], a, b1, c[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        G4::load(As[1][1], g.a, ac(br*2+1, tile+3), soA);
        G4::load(Bs[1][1], g.b, bc_(bc*2+1, tile+3), soB);
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();
    };

    #pragma unroll 2
    for (int tile = 0; tile < g.ki - 2; tile += 2) {
        main_loop_iter(tile);
    }

    // Penultimate
    {
        load_a(a, As[cur][0], wm);
        load_b(b0, Bs[cur][0], wn);
        load_b(b1, Bs[cur][1], wn);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(c[0][0], a, b0, c[0][0]);
        mma_ABt(c[0][1], a, b1, c[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        load_a(a, As[cur][1], wm);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(c[1][0], a, b0, c[1][0]);
        mma_ABt(c[1][1], a, b1, c[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        cur ^= 1; nxt ^= 1;
    }

    // Last
    {
        load_a(a, As[cur][0], wm);
        load_b(b0, Bs[cur][0], wn);
        load_b(b1, Bs[cur][1], wn);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(c[0][0], a, b0, c[0][0]);
        mma_ABt(c[0][1], a, b1, c[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        load_a(a, As[cur][1], wm);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(c[1][0], a, b0, c[1][0]);
        mma_ABt(c[1][1], a, b1, c[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    if (wm == 0) __builtin_amdgcn_s_barrier();
    store(g.c, c[0][0], {0,0, br*2*WM+wm, bc*2*WN+wn});
    store(g.c, c[0][1], {0,0, br*2*WM+wm, bc*2*WN+WN+wn});
    store(g.c, c[1][0], {0,0, br*2*WM+WM+wm, bc*2*WN+wn});
    store(g.c, c[1][1], {0,0, br*2*WM+WM+wm, bc*2*WN+WN+wn});
}

static void rcr(pybind11::object a, pybind11::object b, pybind11::object c, int gm) {
    auto ga=py::from_object<_gl>::make(a);
    auto gb=py::from_object<_gl>::make(b);
    auto gc=py::from_object<_gl>::make(c);
    int M=gc.rows(),N=gc.cols(),K=ga.cols();
    globals g{ga,gb,gc,{},M,N,K,K/KS,M/BLK,N/BLK,gm};
    unsigned long mem=MAX_SHARED_MEMORY;
    hipFuncSetAttribute((void*)bf16_rcr_4wave,hipFuncAttributeMaxDynamicSharedMemorySize,mem);
    bf16_rcr_4wave<<<dim3(g.bpr*g.bpc),dim3(NT),mem,g.stream>>>(g);
}

PYBIND11_MODULE(tk_bf16_4wave, m) {
    using namespace pybind11::literals;
    m.def("gemm_rcr", &rcr, "a"_a, "b"_a, "c"_a, "group_m"_a = 4);
}
