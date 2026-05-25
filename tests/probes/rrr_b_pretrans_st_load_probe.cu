// rrr_b_pretrans_st_load_probe.cu
//
// Session 3 of RRR v2 B-pretranspose campaign.
//
// Validates the new (ST type `st_128x128_n_major` + corresponding
// `load(col_l RT, ST)` specialization in shared_to_register.cuh) against
// Session 1's mma_AB B-operand (lane, byte) -> (k, n) mapping.
//
// LDS layout (N-major, identity swizzle): byte_offset(n, k) = n*128 + k.
// Per-lane 2x ds_read_b128 path (closed-form from Session 2 probe):
//     n_val = j*16 + (lane & 15)        // per j (N-tile)
//     k_block = (lane >> 4) & 3
//     addr = base + n_val*128 + k_block*16          (offset:0)
//     addr_high = base + n_val*128 + k_block*16 + 64 (offset:64)
//
// Expected (lane, byte) -> (k, n) per Session 1 closed form:
//     n        = j*16 + (lane & 15)
//     k_block  = (lane >> 4) & 3
//     half     = byte >> 4
//     k        = k_block*16 + (byte & 15) + half*64
//
// Pass criterion: mismatch == 0 over 64 lanes x 32 bytes x 8 N-tiles
//                 = 16384 (lane, byte, j) tuples.
//
// Build (chi2811 gfx950):
//     make -C tests/probes rrr_b_pretrans_st_load_probe
// Run:
//     ./tests/probes/rrr_b_pretrans_st_load_probe

#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

using namespace kittens;

// Full N-major B tile: 128 N rows x 128 K cols, fp8e4m3, identity swizzle.
using ST = st_fp8e4m3<128, 128, st_128x128_n_major_s>;
// col-layout RT covering same 128 K x 128 N (height=1, width=8).
using RT = rt_fp8e4m3<128, 128, col_l, rt_128x16_s>;

static_assert(RT::height == 1, "RT must cover full K=128 in 1 base-tile row");
static_assert(RT::width  == 8, "RT must cover full N=128 as 8 base-tile cols");
static_assert(RT::base_tile_rows == 128 && RT::base_tile_cols == 16,
              "RT must use rt_128x16 base tile (col_l fp8 path)");

constexpr int LANES        = 64;
constexpr int BYTES_LANE   = 32;       // per base tile (8 quad-bytes)
constexpr int N_TILES      = RT::width; // 8
constexpr int OUT_BYTES    = LANES * BYTES_LANE * N_TILES;

// mode 0 = byte holds K index, mode 1 = byte holds N index.
__global__ void __launch_bounds__(64, 1)
probe_st_load(uint8_t* __restrict__ out, int mode)
{
    __shared__ ST Bs;

    const int tid = threadIdx.x;

    // Fill Bs with identity swizzle: byte_offset(n, k) = n*128 + k.
    // 128 N * 128 K = 16384 bytes; 64 threads each write 256 bytes.
    uint8_t* raw = reinterpret_cast<uint8_t*>(&Bs.data[0]);
    #pragma unroll
    for (int i = 0; i < 256; ++i) {
        int linear = tid * 256 + i;
        int n = linear / 128;     // 0..127
        int k = linear % 128;     // 0..127
        // st_128x128_n_major::swizzle is identity for fp8 (sizeof T == 1).
        // coord.x = row index = n in this N-major layout.
        // coord.y = col index = k in this N-major layout.
        uint32_t swz = ST::swizzle(int2{n, k});
        raw[swz] = (uint8_t)(((mode == 0) ? k : n) & 0xFF);
    }
    __syncthreads();

    // Dispatch to new b128 load specialization.
    RT b;
    kittens::load(b, Bs);

    // Each lane: b.tiles[0][j].data[0..7] = fp8e4m3_4[8] = 32 bytes per j.
    #pragma unroll
    for (int j = 0; j < N_TILES; ++j) {
        const uint8_t* lane_bytes = reinterpret_cast<const uint8_t*>(
            &b.tiles[0][j].data[0]);
        #pragma unroll
        for (int i = 0; i < BYTES_LANE; ++i) {
            int dst_off = (j * LANES + tid) * BYTES_LANE + i;
            out[dst_off] = lane_bytes[i];
        }
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

    CK(hipMemset(d_out, 0xFF, OUT_BYTES));
    probe_st_load<<<1, 64>>>(d_out, 0);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(h_k.data(), d_out, OUT_BYTES, hipMemcpyDeviceToHost));

    CK(hipMemset(d_out, 0xFF, OUT_BYTES));
    probe_st_load<<<1, 64>>>(d_out, 1);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(h_n.data(), d_out, OUT_BYTES, hipMemcpyDeviceToHost));

    CK(hipFree(d_out));

    // Verify (lane, byte, j) -> (k, n) matches Session 1 closed form.
    int mismatch = 0, bad_range = 0;
    for (int j = 0; j < N_TILES; ++j) {
        for (int lane = 0; lane < LANES; ++lane) {
            const int k_block      = (lane >> 4) & 3;
            const int n_val_expect = j * 16 + (lane & 15);
            for (int byte = 0; byte < BYTES_LANE; ++byte) {
                const int half = byte >> 4;
                const int k_expect = k_block * 16 + (byte & 15) + half * 64;
                const int off = (j * LANES + lane) * BYTES_LANE + byte;
                const int k_got = h_k[off];
                const int n_got = h_n[off];
                if (k_got >= 128 || n_got >= 128) { ++bad_range; continue; }
                if (k_got != k_expect || n_got != n_val_expect) {
                    if (mismatch < 16) {
                        fprintf(stderr,
                            "MISMATCH j=%d lane=%d byte=%d: got (k=%d,n=%d) "
                            "expect (k=%d,n=%d)\n",
                            j, lane, byte, k_got, n_got, k_expect, n_val_expect);
                    }
                    ++mismatch;
                }
            }
        }
    }

    // Coverage: each (k, n) in 128x128 must appear exactly once over j-range
    // (lane covers same n_val per j, all 128 n covered by j=0..7 x lane=0..15).
    std::vector<int> cnt(128 * 128, 0);
    for (int off = 0; off < OUT_BYTES; ++off) {
        int k = h_k[off];
        int n = h_n[off];
        if (k < 128 && n < 128) ++cnt[k * 128 + n];
    }
    int missing = 0, duplicate = 0;
    for (int v : cnt) {
        if (v == 0) ++missing;
        else if (v != 1) ++duplicate;
    }

    fprintf(stderr,
            "Session 3 ST+load probe: mismatch=%d bad_range=%d "
            "missing=%d duplicate=%d (out_bytes=%d, expected=%d)\n",
            mismatch, bad_range, missing, duplicate,
            OUT_BYTES, 128 * 128);

    if (mismatch == 0 && bad_range == 0 && missing == 0 && duplicate == 0) {
        fprintf(stderr, "OK: 128x128 (k,n) coordinates, each appears exactly once;\n");
        fprintf(stderr, "    new st_128x128_n_major + load() == Session 1 mma_AB mapping.\n");
    } else {
        fprintf(stderr, "FAIL: see counters above.\n");
    }

    if (dump_table) {
        printf("j,lane,byte,k,n\n");
        for (int j = 0; j < N_TILES; ++j)
            for (int lane = 0; lane < LANES; ++lane)
                for (int byte = 0; byte < BYTES_LANE; ++byte) {
                    int off = (j * LANES + lane) * BYTES_LANE + byte;
                    printf("%d,%d,%d,%d,%d\n", j, lane, byte,
                           (int)h_k[off], (int)h_n[off]);
                }
    }

    return (mismatch || bad_range || missing || duplicate) ? 2 : 0;
}
