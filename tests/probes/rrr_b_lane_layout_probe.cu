// rrr_b_lane_layout_probe.cu
//
// Session 1 of RRR v2 B-pretranspose campaign.
//
// Goal: empirically determine which (k, n) elements of one B base tile
// (128 K × 16 N, fp8e4m3) end up in each lane's mfma B operand register
// after the standard col-layout ds_read_b64_tr_b8 load path used by RRR.
//
// Strategy: fill an LDS tile with a known pattern (mode 0 = byte holds K
// index, mode 1 = byte holds N index), call the standard
// kittens::load(rt_128x16 col_l, st_128x16_s) (which uses ds_read_b64_tr_b8
// internally — same instruction the production RRR kernel uses for B),
// then dump every lane's 32 B-operand bytes back to HBM. Two passes give
// (k, n) per (lane, byte).
//
// We use the identity-swizzle st_128x16_s rather than the production
// st_16x128_v2_s. The swizzle only changes where bytes physically live in
// LDS; the (lane, byte_idx) → (k, n) mapping that mfma consumes is
// determined by the mfma instruction + ds_read_b64_tr_b8 semantics, which
// are swizzle-independent.

#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

using namespace kittens;

// One B base tile for fp8 mfma_f32_16x16x128: 128 K × 16 N.
using ST = st_fp8e4m3<128, 16, st_128x16_s>;
using RT = rt_fp8e4m3<128, 16, col_l, rt_128x16_s>;

static_assert(RT::base_tile_rows == 128, "expect rt_128x16 base tile");
static_assert(RT::base_tile_cols == 16,  "expect rt_128x16 base tile");
static_assert(RT::base_tile_stride == 16, "expect stride 16 (fp8 col_l path)");

static_assert(ST::underlying_subtile_rows == 128, "expect 128-row st subtile");
static_assert(ST::underlying_subtile_cols == 16,  "expect 16-col  st subtile");

// 64 lanes × 32 bytes per lane = 2048 bytes per base tile.
constexpr int LANES        = 64;
constexpr int BYTES_LANE   = 32;
constexpr int OUT_BYTES    = LANES * BYTES_LANE;

// mode 0 fills B[k][n] = k & 0xFF; mode 1 fills B[k][n] = n & 0xFF.
__global__ void __launch_bounds__(64, 1)
probe_b_lane_layout(uint8_t* __restrict__ out, int mode)
{
    __shared__ ST Bs;

    const int tid = threadIdx.x;

    // Fill Bs. Identity swizzle: byte[(k * 16 + n)] = data[k][n].
    // Each of 64 threads writes 32 bytes (128*16 = 2048 / 64).
    uint8_t* raw = reinterpret_cast<uint8_t*>(&Bs.data[0]);
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        int linear = tid * 32 + i;
        int k = linear / 16;     // 0..127
        int n = linear % 16;     // 0..15
        // Identity swizzle for st_128x16_s on fp8 = offset itself.
        uint32_t swz = ST::swizzle(int2{k, n});
        raw[swz] = (uint8_t)(((mode == 0) ? k : n) & 0xFF);
    }
    __syncthreads();

    // Load via the standard col-layout fp8 path (calls ds_read_b64_tr_b8).
    RT b;
    kittens::load(b, Bs);

    // Each lane: b.tiles[0][0].data[0..7] is fp8e4m3_4[8] = 32 bytes,
    // exactly the operand consumed by mfma_f32_16x16x128_f8f6f4.
    const uint8_t* lane_bytes = reinterpret_cast<const uint8_t*>(
        &b.tiles[0][0].data[0]);

    #pragma unroll
    for (int i = 0; i < BYTES_LANE; ++i) {
        out[tid * BYTES_LANE + i] = lane_bytes[i];
    }
}

#define CK(stmt) do { hipError_t e = (stmt); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,    \
            hipGetErrorString(e)); std::exit(1); } } while(0)

int main(int argc, char** argv)
{
    bool dump_table = (argc > 1 && std::string(argv[1]) == "--table");

    uint8_t* d_out = nullptr;
    CK(hipMalloc(&d_out, OUT_BYTES));

    std::vector<uint8_t> h_k(OUT_BYTES), h_n(OUT_BYTES);

    // Mode 0 → bytes are K index.
    CK(hipMemset(d_out, 0xFF, OUT_BYTES));
    probe_b_lane_layout<<<1, 64>>>(d_out, 0);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(h_k.data(), d_out, OUT_BYTES, hipMemcpyDeviceToHost));

    // Mode 1 → bytes are N index.
    CK(hipMemset(d_out, 0xFF, OUT_BYTES));
    probe_b_lane_layout<<<1, 64>>>(d_out, 1);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(h_n.data(), d_out, OUT_BYTES, hipMemcpyDeviceToHost));

    CK(hipFree(d_out));

    // Sanity: every byte should decode to k ∈ [0, 128) and n ∈ [0, 16).
    int bad_k = 0, bad_n = 0;
    for (int i = 0; i < OUT_BYTES; ++i) {
        if (h_k[i] >= 128) ++bad_k;
        if (h_n[i] >= 16)  ++bad_n;
    }
    if (bad_k || bad_n) {
        fprintf(stderr, "FAIL: out-of-range bytes (bad_k=%d, bad_n=%d)\n",
                bad_k, bad_n);
        return 1;
    }

    // Coverage: each (k, n) in the 128×16 grid should appear exactly once.
    std::vector<int> cnt(128 * 16, 0);
    for (int lane = 0; lane < LANES; ++lane) {
        for (int b = 0; b < BYTES_LANE; ++b) {
            int k = h_k[lane * BYTES_LANE + b];
            int n = h_n[lane * BYTES_LANE + b];
            ++cnt[k * 16 + n];
        }
    }
    int missing = 0, duplicate = 0;
    for (int v : cnt) {
        if (v == 0) ++missing;
        else if (v != 1) ++duplicate;
    }
    if (missing || duplicate) {
        fprintf(stderr, "FAIL: coverage broken (missing=%d, duplicate=%d)\n",
                missing, duplicate);
        // Still dump table for debug.
    } else {
        fprintf(stderr, "OK: 2048 (k,n) coordinates, each appears exactly once.\n");
    }

    if (dump_table) {
        // CSV: lane,byte,k,n
        printf("lane,byte,k,n\n");
        for (int lane = 0; lane < LANES; ++lane) {
            for (int b = 0; b < BYTES_LANE; ++b) {
                printf("%d,%d,%d,%d\n",
                       lane, b,
                       (int)h_k[lane * BYTES_LANE + b],
                       (int)h_n[lane * BYTES_LANE + b]);
            }
        }
    } else {
        // Compact summary per lane: show k and n sequences.
        printf("# B operand lane layout for mfma_f32_16x16x128_f8f6f4 (mma_AB, RRR B)\n");
        printf("# Base tile: 128 K × 16 N, fp8e4m3. Each lane holds 32 bytes.\n");
        printf("# B-operand register: fp8e4m3_4 b[8] = data[0..7] (4 bytes each).\n");
        printf("# Run with --table for the full lane,byte,k,n CSV.\n#\n");
        for (int lane = 0; lane < LANES; ++lane) {
            printf("lane %2d:", lane);
            printf(" k=[");
            for (int b = 0; b < BYTES_LANE; ++b) {
                if (b) printf(",");
                printf("%d", (int)h_k[lane * BYTES_LANE + b]);
            }
            printf("] n=[");
            for (int b = 0; b < BYTES_LANE; ++b) {
                if (b) printf(",");
                printf("%d", (int)h_n[lane * BYTES_LANE + b]);
            }
            printf("]\n");
        }
    }
    return (missing || duplicate) ? 2 : 0;
}
