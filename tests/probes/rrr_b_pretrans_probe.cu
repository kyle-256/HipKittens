// rrr_b_pretrans_probe.cu
//
// Session 2 of RRR v2 B-pretranspose campaign.
//
// Goal: empirically verify that storing B in LDS as transposed N-major
// layout (LDS[n][k] = HBM[k][n]) and reading via 2× ds_read_b128 per
// lane produces a per-lane (lane, byte) → (k, n) mapping IDENTICAL to
// the Session 1 mma_AB B-operand mapping. If equal byte-for-byte, then
// mma_AB(A, B) with this new path == mma_AB(A, B) with the current
// ds_read_b64_tr_b8 path, without having to actually run an mfma.
//
// Tile: 128 K × 16 N, fp8e4m3 (one mma_f32_16x16x128 B base tile).
// LDS layout: 16 rows × 128 cols, byte_offset(n, k) = n*128 + k (identity).
// Lane L reads 32 bytes via 2× ds_read_b128 at offsets:
//   addr0 = base + (L & 15)*128 + ((L >> 4) & 3)*16
//   addr1 = addr0 + 64
//
// Per Session 1 closed-form mapping (feedback_rrr_b_lane_layout.md):
//   n        = L & 15
//   k_block  = (L >> 4) & 3
//   half     = byte >> 4
//   k        = k_block*16 + (byte & 15) + half*64
// Expected: byte 0..15 → K=k_block*16+(0..15), all at n.
//           byte 16..31 → K=k_block*16+64+(0..15), all at n.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

constexpr int K_DIM       = 128;
constexpr int N_DIM       = 16;
constexpr int TILE_BYTES  = K_DIM * N_DIM;        // 2048
constexpr int LANES       = 64;
constexpr int BYTES_LANE  = 32;
constexpr int OUT_BYTES   = LANES * BYTES_LANE;   // 2048

// Kernel:
//  - Each of 64 threads writes 32 bytes to LDS in TRANSPOSED layout.
//    For source byte HBM[k][n] (row-major K×N): LDS[n*K_DIM + k] = src.
//  - Source pattern: mode 0 → byte = k & 0xFF; mode 1 → byte = n & 0xFF.
//  - Each lane reads 2× ds_read_b128 → 32 bytes.
//  - Dump to HBM.
__global__ void __launch_bounds__(64, 1)
probe_b_pretrans(uint8_t* __restrict__ out, int mode)
{
    __shared__ __align__(16) uint8_t Bs[TILE_BYTES];

    const int tid = threadIdx.x;  // 0..63

    // Each thread writes 32 bytes. Use thread's "lane" view of K,N to
    // produce a simple deterministic fill. We need every (k, n) ∈
    // [0,128) × [0,16) to be written exactly once. 64 threads × 32 bytes
    // = 2048 = TILE_BYTES, perfect.
    //
    // Layout chosen: thread t writes K = (t*32 + i)/N_DIM and N =
    // (t*32 + i)%N_DIM, mapping linear src offset 0..2047 to (k, n).
    // Then transpose-store: LDS[n*K_DIM + k] = src_byte(k, n).
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        int lin = tid * 32 + i;          // 0..2047
        int k = lin / N_DIM;             // 0..127
        int n = lin % N_DIM;             // 0..15
        uint8_t v = (uint8_t)(((mode == 0) ? k : n) & 0xFF);
        Bs[n * K_DIM + k] = v;
    }
    __syncthreads();

    // Compute per-lane base address and do 2× ds_read_b128.
    const int n_val   = tid & 15;
    const int k_block = (tid >> 4) & 3;
    const uint32_t lds_base = reinterpret_cast<uintptr_t>(&Bs[0]);
    const uint32_t addr0 = lds_base + n_val * K_DIM + k_block * 16;

    uint8_t bytes[BYTES_LANE];

    asm volatile(
        "ds_read_b128 %0, %2 offset:0\n"
        "ds_read_b128 %1, %2 offset:64\n"
        "s_waitcnt lgkmcnt(0)\n"
        : "=&v"(*reinterpret_cast<int4*>(&bytes[0])),
          "=&v"(*reinterpret_cast<int4*>(&bytes[16]))
        : "v"(addr0)
        : "memory"
    );

    #pragma unroll
    for (int i = 0; i < BYTES_LANE; ++i) {
        out[tid * BYTES_LANE + i] = bytes[i];
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
    probe_b_pretrans<<<1, 64>>>(d_out, 0);
    CK(hipGetLastError()); CK(hipDeviceSynchronize());
    CK(hipMemcpy(h_k.data(), d_out, OUT_BYTES, hipMemcpyDeviceToHost));

    CK(hipMemset(d_out, 0xFF, OUT_BYTES));
    probe_b_pretrans<<<1, 64>>>(d_out, 1);
    CK(hipGetLastError()); CK(hipDeviceSynchronize());
    CK(hipMemcpy(h_n.data(), d_out, OUT_BYTES, hipMemcpyDeviceToHost));

    CK(hipFree(d_out));

    // Validate per Session 1 closed-form mapping.
    int mismatch = 0;
    for (int lane = 0; lane < LANES; ++lane) {
        const int n_val   = lane & 15;
        const int k_block = (lane >> 4) & 3;
        for (int b = 0; b < BYTES_LANE; ++b) {
            const int half  = b >> 4;             // 0 or 1
            const int exp_k = k_block * 16 + (b & 15) + half * 64;
            const int exp_n = n_val;
            const int got_k = h_k[lane * BYTES_LANE + b];
            const int got_n = h_n[lane * BYTES_LANE + b];
            if (got_k != exp_k || got_n != exp_n) {
                if (mismatch < 8) {
                    fprintf(stderr,
                        "MISMATCH lane=%d byte=%d  got=(k=%d,n=%d)  expected=(k=%d,n=%d)\n",
                        lane, b, got_k, got_n, exp_k, exp_n);
                }
                ++mismatch;
            }
        }
    }

    // Also verify coverage: each (k, n) ∈ [0,128)×[0,16) appears exactly once.
    std::vector<int> cnt(K_DIM * N_DIM, 0);
    for (int i = 0; i < OUT_BYTES; ++i) {
        int k = h_k[i], n = h_n[i];
        if (k < K_DIM && n < N_DIM) ++cnt[k * N_DIM + n];
    }
    int missing = 0, dup = 0;
    for (int v : cnt) { if (v == 0) ++missing; else if (v != 1) ++dup; }

    if (mismatch == 0 && missing == 0 && dup == 0) {
        fprintf(stderr,
            "OK: b128-from-pretranspose-LDS produces Session-1-equivalent\n"
            "    (lane, byte) → (k, n) mapping. Coverage 2048/2048 unique.\n");
    } else {
        fprintf(stderr,
            "FAIL: mismatch=%d  missing=%d  duplicate=%d\n",
            mismatch, missing, dup);
    }

    if (dump_table) {
        printf("lane,byte,k,n\n");
        for (int lane = 0; lane < LANES; ++lane) {
            for (int b = 0; b < BYTES_LANE; ++b) {
                printf("%d,%d,%d,%d\n", lane, b,
                       (int)h_k[lane * BYTES_LANE + b],
                       (int)h_n[lane * BYTES_LANE + b]);
            }
        }
    }

    return (mismatch == 0 && missing == 0 && dup == 0) ? 0 : 1;
}
