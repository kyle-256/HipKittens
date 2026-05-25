// rrr_b_pretrans_load_subtile_probe.cu
//
// Session 5 of RRR v2 B-pretranspose campaign — subtile load probe.
//
// Validates that the *subtile* variant of the new st_128x128_n_major
// load specialization (taking a `col_start` N-offset and producing an
// RT of shape rt_fp8e4m3<128, 32, col_l, rt_128x16_s> with width=2)
// produces the same (lane, byte) -> (k, n) mapping as Session 1's
// mma_AB B-operand closed form, for every n_offset ∈ {0, 32, 64, 96}.
//
// This is the missing interface piece for Session 5.1 kernel body
// integration: the RRR main loop calls `load_b(dst, Bs[a][b], wi)` four
// times per K-iter (wi=0..3, n_offset = wi * 32). The kernel body
// templated load must dispatch to a function that takes (RT, ST, int).
//
// LDS layout (N-major, identity swizzle): byte_offset(n, k) = n*128 + k.
// Per-lane 2x ds_read_b128 path:
//     n_val = j*16 + (lane & 15) + col_start    // per j (N-tile)
//     k_block = (lane >> 4) & 3
//     addr = base + n_val*128 + k_block*16          (offset:0)
//     addr_high = base + n_val*128 + k_block*16 + 64 (offset:64)
//
// Expected (lane, byte) -> (k, n):
//     n        = j*16 + (lane & 15) + col_start
//     k_block  = (lane >> 4) & 3
//     half     = byte >> 4
//     k        = k_block*16 + (byte & 15) + half*64
//
// Build (chi2811 gfx950):
//     make -C tests/probes rrr_b_pretrans_load_subtile_probe
// Run:
//     ./tests/probes/rrr_b_pretrans_load_subtile_probe

#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

using namespace kittens;

// Full N-major B tile: 128 N rows x 128 K cols, fp8e4m3, identity swizzle.
using ST  = st_fp8e4m3<128, 128, st_128x128_n_major_s>;
// Width-2 col-layout RT covering 128 K x 32 N (one wi-slice).
using RT2 = rt_fp8e4m3<128, 32, col_l, rt_128x16_s>;

static_assert(RT2::height == 1, "RT2 must cover full K=128 in 1 base-tile row");
static_assert(RT2::width  == 2, "RT2 must cover 32 N as 2 base-tile cols");
static_assert(RT2::base_tile_rows == 128 && RT2::base_tile_cols == 16,
              "RT2 must use rt_128x16 base tile (col_l fp8 path)");

constexpr int LANES        = 64;
constexpr int BYTES_LANE   = 32;   // per base tile (8 quad-bytes)
constexpr int N_TILES_PER  = RT2::width;  // 2
constexpr int OUT_BYTES_PER = LANES * BYTES_LANE * N_TILES_PER;  // 4096 per wi

// Session 5 subtile load. Templated mirror of Session 3 spec (in
// shared_to_register.cuh) with a runtime `col_start` N-offset.
template<typename RT_, typename ST_>
__device__ __forceinline__ void load_col_from_st_n_major_subtile(
    RT_& dst, const ST_& tile, int col_start)
{
    const int laneid       = kittens::laneid();
    const int n_val_base   = laneid & 15;
    const int k_block      = (laneid >> 4) & 3;
    const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&tile.data[0]);
    constexpr int K_DIM    = ST_::cols;
    #pragma unroll
    for (int j = 0; j < RT_::width; ++j) {
        const int n_val = j * 16 + n_val_base + col_start;
        const uint32_t addr = src_ptr + n_val * K_DIM + k_block * 16;
        asm volatile(
            "ds_read_b128 %0, %2 offset:0\n"
            "ds_read_b128 %1, %2 offset:64\n"
            : "=&v"(*reinterpret_cast<float4*>(&dst.tiles[0][j].data[0])),
              "=&v"(*reinterpret_cast<float4*>(&dst.tiles[0][j].data[4]))
            : "v"(addr)
            : "memory"
        );
    }
}

// mode 0: byte holds K index, mode 1: byte holds N index.
__global__ void __launch_bounds__(64, 1)
probe_subtile_load(uint8_t* __restrict__ out, int mode, int col_start)
{
    __shared__ ST Bs;
    const int tid = threadIdx.x;

    // Identity-swizzle fill: byte_offset(n, k) = n*128 + k.
    uint8_t* raw = reinterpret_cast<uint8_t*>(&Bs.data[0]);
    #pragma unroll
    for (int i = 0; i < 256; ++i) {
        int linear = tid * 256 + i;
        int n = linear / 128;
        int k = linear % 128;
        uint32_t swz = ST::swizzle(int2{n, k});
        raw[swz] = (uint8_t)(((mode == 0) ? k : n) & 0xFF);
    }
    __syncthreads();

    // Hand-unrolled per-j with FULL SCOPE PER J. Each scope has its own
    // addr SGPR and own int2 locals → compiler cannot alias VGPRs across j.
    // Uses 4× ds_read_b64 (each needs only 2-VGPR alignment).
    {
        const int laneid       = kittens::laneid();
        const int n_val_base   = laneid & 15;
        const int k_block      = (laneid >> 4) & 3;
        const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&Bs.data[0]);
        constexpr int K_DIM    = 128;

        // j=0
        {
            const int n_val = 0 + n_val_base + col_start;
            const uint32_t addr = src_ptr + n_val * K_DIM + k_block * 16;
            int2 q0, q1, q2, q3;
            asm volatile("ds_read_b64 %0, %1 offset:0"  : "=&v"(q0) : "v"(addr) : "memory");
            asm volatile("ds_read_b64 %0, %1 offset:8"  : "=&v"(q1) : "v"(addr) : "memory");
            asm volatile("ds_read_b64 %0, %1 offset:64" : "=&v"(q2) : "v"(addr) : "memory");
            asm volatile("ds_read_b64 %0, %1 offset:72" : "=&v"(q3) : "v"(addr) : "memory");
            int dst_base = (0 * LANES + tid) * BYTES_LANE;
            const uint8_t* p0 = reinterpret_cast<const uint8_t*>(&q0);
            const uint8_t* p1 = reinterpret_cast<const uint8_t*>(&q1);
            const uint8_t* p2 = reinterpret_cast<const uint8_t*>(&q2);
            const uint8_t* p3 = reinterpret_cast<const uint8_t*>(&q3);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                out[dst_base + i]      = p0[i];
                out[dst_base + 8 + i]  = p1[i];
                out[dst_base + 16 + i] = p2[i];
                out[dst_base + 24 + i] = p3[i];
            }
            // Prevent compiler from reordering j=0 byte loads BELOW j=1's ds_read
            // (which would re-use the same VGPRs and clobber q0..q3).
            asm volatile("" ::: "memory");
        }
        // j=1
        {
            const int n_val = 16 + n_val_base + col_start;
            const uint32_t addr = src_ptr + n_val * K_DIM + k_block * 16;
            int2 q0, q1, q2, q3;
            asm volatile("ds_read_b64 %0, %1 offset:0"  : "=&v"(q0) : "v"(addr) : "memory");
            asm volatile("ds_read_b64 %0, %1 offset:8"  : "=&v"(q1) : "v"(addr) : "memory");
            asm volatile("ds_read_b64 %0, %1 offset:64" : "=&v"(q2) : "v"(addr) : "memory");
            asm volatile("ds_read_b64 %0, %1 offset:72" : "=&v"(q3) : "v"(addr) : "memory");
            int dst_base = (1 * LANES + tid) * BYTES_LANE;
            const uint8_t* p0 = reinterpret_cast<const uint8_t*>(&q0);
            const uint8_t* p1 = reinterpret_cast<const uint8_t*>(&q1);
            const uint8_t* p2 = reinterpret_cast<const uint8_t*>(&q2);
            const uint8_t* p3 = reinterpret_cast<const uint8_t*>(&q3);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                out[dst_base + i]      = p0[i];
                out[dst_base + 8 + i]  = p1[i];
                out[dst_base + 16 + i] = p2[i];
                out[dst_base + 24 + i] = p3[i];
            }
        }
    }
}

#define CK(stmt) do { hipError_t e = (stmt); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,    \
            hipGetErrorString(e)); std::exit(1); } } while(0)

static int verify_one(int col_start, bool verbose)
{
    uint8_t* d_out = nullptr;
    CK(hipMalloc(&d_out, OUT_BYTES_PER));
    std::vector<uint8_t> h_k(OUT_BYTES_PER), h_n(OUT_BYTES_PER);

    CK(hipMemset(d_out, 0xFF, OUT_BYTES_PER));
    probe_subtile_load<<<1, 64>>>(d_out, 0, col_start);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(h_k.data(), d_out, OUT_BYTES_PER, hipMemcpyDeviceToHost));

    CK(hipMemset(d_out, 0xFF, OUT_BYTES_PER));
    probe_subtile_load<<<1, 64>>>(d_out, 1, col_start);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(h_n.data(), d_out, OUT_BYTES_PER, hipMemcpyDeviceToHost));
    CK(hipFree(d_out));

    int mismatch = 0, bad_range = 0;
    for (int j = 0; j < N_TILES_PER; ++j) {
        for (int lane = 0; lane < LANES; ++lane) {
            const int k_block      = (lane >> 4) & 3;
            const int n_val_expect = j * 16 + (lane & 15) + col_start;
            for (int byte = 0; byte < BYTES_LANE; ++byte) {
                const int half = byte >> 4;
                const int k_expect = k_block * 16 + (byte & 15) + half * 64;
                const int off = (j * LANES + lane) * BYTES_LANE + byte;
                const int k_got = h_k[off];
                const int n_got = h_n[off];
                if (k_got >= 128 || n_got >= 128) { ++bad_range; continue; }
                if (k_got != k_expect || n_got != n_val_expect) {
                    if (verbose && mismatch < 8) {
                        fprintf(stderr,
                            "  col_start=%d MISMATCH j=%d lane=%d byte=%d: "
                            "got (k=%d,n=%d) expect (k=%d,n=%d)\n",
                            col_start, j, lane, byte, k_got, n_got,
                            k_expect, n_val_expect);
                    }
                    ++mismatch;
                }
            }
        }
    }

    // Coverage on a 128x32 sub-rectangle: each (k, n) in 128x32 must appear exactly once.
    std::vector<int> cnt(128 * 32, 0);
    for (int j = 0; j < N_TILES_PER; ++j) {
        for (int lane = 0; lane < LANES; ++lane) {
            for (int byte = 0; byte < BYTES_LANE; ++byte) {
                int off = (j * LANES + lane) * BYTES_LANE + byte;
                int k = h_k[off];
                int n = h_n[off] - col_start;
                if (k >= 0 && k < 128 && n >= 0 && n < 32) ++cnt[k * 32 + n];
            }
        }
    }
    int missing = 0, dup = 0;
    for (int v : cnt) { if (v == 0) ++missing; else if (v != 1) ++dup; }

    fprintf(stderr,
            "  col_start=%-3d: mismatch=%d bad_range=%d missing=%d dup=%d\n",
            col_start, mismatch, bad_range, missing, dup);
    return mismatch + bad_range + missing + dup;
}

int main(int argc, char** argv)
{
    bool verbose = (argc > 1 && std::string(argv[1]) == "-v");
    int total_bad = 0;
    fprintf(stderr, "Session 5 subtile load probe (RT::width=2, n_offset sweep):\n");

    // Debug dump: col_start=0, mode=0, lane 0..3, j=0
    {
        uint8_t* d_out = nullptr;
        CK(hipMalloc(&d_out, OUT_BYTES_PER));
        CK(hipMemset(d_out, 0xFF, OUT_BYTES_PER));
        probe_subtile_load<<<1, 64>>>(d_out, 0, 0);
        CK(hipDeviceSynchronize());
        std::vector<uint8_t> h(OUT_BYTES_PER);
        CK(hipMemcpy(h.data(), d_out, OUT_BYTES_PER, hipMemcpyDeviceToHost));
        CK(hipFree(d_out));
        fprintf(stderr, "DEBUG col_start=0 mode=0 (byte holds k):\n");
        for (int lane = 0; lane <= 17; lane += 16) {
            for (int j = 0; j < 2; ++j) {
                fprintf(stderr, "  lane=%2d j=%d: ", lane, j);
                for (int i = 0; i < 32; ++i) {
                    int off = (j * LANES + lane) * BYTES_LANE + i;
                    fprintf(stderr, "%02x ", h[off]);
                    if (i == 15) fprintf(stderr, "| ");
                }
                fprintf(stderr, "\n");
            }
        }
    }

    for (int cs : {0, 32, 64, 96}) {
        total_bad += verify_one(cs, verbose);
    }
    if (total_bad == 0) {
        fprintf(stderr,
            "OK: subtile load matches Session 1 mma_AB mapping for all "
            "n_offset ∈ {0,32,64,96}.\n"
            "    Ready for kernel body integration (Session 5.1).\n");
        return 0;
    }
    fprintf(stderr, "FAIL: see counters above (total_bad=%d).\n", total_bad);
    return 2;
}
