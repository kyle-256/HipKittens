// rrr_prolog_b_pretrans_probe.cu
//
// Session 12 of RRR v2 B-pretranspose campaign — prolog-PoC.
//
// Exercises the 4-tile B-pretranspose prolog pattern that production
// `grouped_rrr_kernel_body_pinned` will use under RRR_B_PRETRANS=1
// (wired in Session 13). This probe is standalone — production body is
// NOT touched in Session 12 — so we can validate the writer + LDS
// allocation pattern in isolation before integration.
//
// Production prolog (kernel_fp8_layouts2.cpp:1958-1972, RRR_B_PRETRANS=0)
// calls G::load 4 times with coords:
//   b_co(bc*2,   k=0)    → tile (k_block=0, n_strip=bc*2  )
//   b_co(bc*2+1, k=0)    → tile (k_block=0, n_strip=bc*2+1)
//   b_co(bc*2,   k=1)    → tile (k_block=1, n_strip=bc*2  )
//   b_co(bc*2+1, k=1)    → tile (k_block=1, n_strip=bc*2+1)
//
// Under RRR_B_PRETRANS=1 these become 4 writer calls:
//   write_b_transpose_n_major_path_L<ST_NM>(Bs_NM[i], hbm_ptr_i, ..., stage_lds);
// where Bs_NM[i] is an `st_fp8e4m3<128,128,st_128x128_n_major_s>` and
// hbm_ptr_i points at the (k_block,n_strip) 128×128 sub-tile in HBM.
//
// This probe uses bc=0 → n_strips ∈ {0,1}. HBM source B[K=256, N=256] is
// row-major (K-major: stride between rows = N=256 bytes). The 4 writer
// outputs are dumped to HBM and host-verified against:
//
//   expected[tile_idx][n*128 + k] = hbm[(k_block*128 + k)*256 + (n_strip*128 + n)]
//
// where tile_idx = k_block*2 + n_strip.
//
// LDS budget per WG (gfx950, cap 160 KiB):
//   Bs_NM[4] : 4 × 16384 = 65536 B   (replaces production Bs[2][2] ≈ 70 KiB)
//   stage_lds: 16384 B
//   Total   : ~82 KiB (probe only; production body adds As[2][2] etc.)
//
// Build (chi2811 gfx950):
//   make -C tests/probes rrr_prolog_b_pretrans_probe
// Run:
//   ./tests/probes/rrr_prolog_b_pretrans_probe

#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

using namespace kittens;

constexpr int K_DIM        = 128;
constexpr int N_DIM        = 128;
constexpr int TILE_BYTES   = K_DIM * N_DIM;   // 16384
constexpr int HBM_K        = 256;             // full HBM B has 256 K rows
constexpr int HBM_N        = 256;             // and 256 N cols (bc=0 → strips {0,1})
constexpr int HBM_BYTES    = HBM_K * HBM_N;   // 65536
constexpr int NUM_TILES    = 4;               // 2 k_blocks × 2 n_strips
constexpr int DUMP_BYTES   = NUM_TILES * TILE_BYTES; // 65536

using ST_NM = st_fp8e4m3<128, 128, st_128x128_n_major_s>;

static_assert(sizeof(ST_NM) == TILE_BYTES, "ST_NM size mismatch (expected 16384)");

// =====================================================================
// 4-tile prolog: 4 × write_b_transpose_n_major_path_L into Bs_NM[i].
// Dumps each tile byte-by-byte to dst_dump[i*TILE_BYTES ..].
// =====================================================================
__global__ void __launch_bounds__(512, 1)
prolog_b_pretrans_4tile(const uint8_t* __restrict__ hbm_b,    // [HBM_K][HBM_N]
                        uint8_t*       __restrict__ dst_dump) // [NUM_TILES][TILE_BYTES]
{
    __shared__ __align__(16) uint8_t stage_lds[TILE_BYTES];
    __shared__ ST_NM                 Bs_NM[NUM_TILES];

    // Mirror production prolog (bc=0):
    //   tile 0 : k_block=0, n_strip=0
    //   tile 1 : k_block=0, n_strip=1
    //   tile 2 : k_block=1, n_strip=0
    //   tile 3 : k_block=1, n_strip=1
    //
    // Each tile's HBM sub-pointer is &hbm[(k_block*K_DIM)*HBM_N + n_strip*N_DIM]
    // and hbm_k_stride_bytes = HBM_N (full row stride of the HBM B tensor).
    #pragma unroll
    for (int tile_idx = 0; tile_idx < NUM_TILES; ++tile_idx) {
        const int k_block = tile_idx >> 1;     // 0,0,1,1
        const int n_strip = tile_idx & 1;      // 0,1,0,1
        const size_t hbm_off = static_cast<size_t>(k_block) * K_DIM * HBM_N +
                               static_cast<size_t>(n_strip) * N_DIM;
        const fp8e4m3* hbm_tile_ptr =
            reinterpret_cast<const fp8e4m3*>(hbm_b + hbm_off);

        kittens::write_b_transpose_n_major_path_L<ST_NM>(
            Bs_NM[tile_idx],
            hbm_tile_ptr,
            /*hbm_k_stride_bytes=*/(uint32_t)HBM_N,
            stage_lds);
        // write_b_transpose_n_major_path_L ends with __syncthreads(); safe
        // to reuse stage_lds for the next tile in the loop.
    }

    // Dump all 4 tiles to HBM (byte-by-byte) for host verification.
    #pragma unroll
    for (int tile_idx = 0; tile_idx < NUM_TILES; ++tile_idx) {
        const uint8_t* tile_raw =
            reinterpret_cast<const uint8_t*>(&Bs_NM[tile_idx].data[0]);
        uint8_t* dst_raw = dst_dump + tile_idx * TILE_BYTES;
        for (int i = threadIdx.x; i < TILE_BYTES; i += 512) {
            dst_raw[i] = tile_raw[i];
        }
    }
}

#define CK(stmt) do { hipError_t e = (stmt); if (e != hipSuccess) {  \
    fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,     \
            hipGetErrorString(e)); std::exit(1); } } while(0)

int main(int /*argc*/, char** /*argv*/)
{
    // Build deterministic HBM B[K=256][N=256] (row-major, K-major).
    // Pattern: byte = ((k * 31) ^ (n * 17) ^ (k + n)) & 0xFF — varies in
    // both k and n so the (k,n) → byte mapping is bijective per byte
    // position (each tile sees unique content).
    std::vector<uint8_t> hbm(HBM_BYTES);
    for (int k = 0; k < HBM_K; ++k) {
        for (int n = 0; n < HBM_N; ++n) {
            hbm[k * HBM_N + n] =
                (uint8_t)(((k * 31) ^ (n * 17) ^ (k + n)) & 0xFF);
        }
    }

    uint8_t *d_hbm = nullptr, *d_dump = nullptr;
    CK(hipMalloc(&d_hbm,  HBM_BYTES));
    CK(hipMalloc(&d_dump, DUMP_BYTES));
    CK(hipMemcpy(d_hbm, hbm.data(), HBM_BYTES, hipMemcpyHostToDevice));
    CK(hipMemset(d_dump, 0xAA, DUMP_BYTES));

    prolog_b_pretrans_4tile<<<1, 512>>>(d_hbm, d_dump);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());

    std::vector<uint8_t> got(DUMP_BYTES);
    CK(hipMemcpy(got.data(), d_dump, DUMP_BYTES, hipMemcpyDeviceToHost));

    CK(hipFree(d_hbm));
    CK(hipFree(d_dump));

    // Verify each tile:
    //   expected_tile[k_block, n_strip][n*128 + k] =
    //       hbm[(k_block*128 + k) * 256 + (n_strip*128 + n)]
    int total_mismatch = 0;
    int per_tile_mismatch[NUM_TILES] = {0, 0, 0, 0};
    for (int tile_idx = 0; tile_idx < NUM_TILES; ++tile_idx) {
        const int k_block = tile_idx >> 1;
        const int n_strip = tile_idx & 1;
        const uint8_t* tile = got.data() + tile_idx * TILE_BYTES;
        int shown = 0;
        for (int n = 0; n < N_DIM; ++n) {
            for (int k = 0; k < K_DIM; ++k) {
                const int K_glob = k_block * K_DIM + k;
                const int N_glob = n_strip * N_DIM + n;
                const uint8_t expect = hbm[K_glob * HBM_N + N_glob];
                const uint8_t actual = tile[n * K_DIM + k];
                if (actual != expect) {
                    if (shown < 4) {
                        fprintf(stderr,
                            "MISMATCH tile=%d (kblk=%d,strip=%d) n=%d k=%d: "
                            "got=0x%02x expect=0x%02x\n",
                            tile_idx, k_block, n_strip, n, k,
                            (int)actual, (int)expect);
                        ++shown;
                    }
                    ++per_tile_mismatch[tile_idx];
                    ++total_mismatch;
                }
            }
        }
    }

    fprintf(stderr,
            "Session 12 prolog-B-pretrans probe (4-tile):\n"
            "  tile 0 (kblk=0,strip=0) mismatch=%d   /  %d bytes\n"
            "  tile 1 (kblk=0,strip=1) mismatch=%d   /  %d bytes\n"
            "  tile 2 (kblk=1,strip=0) mismatch=%d   /  %d bytes\n"
            "  tile 3 (kblk=1,strip=1) mismatch=%d   /  %d bytes\n"
            "  TOTAL mismatch=%d   /  %d bytes\n",
            per_tile_mismatch[0], TILE_BYTES,
            per_tile_mismatch[1], TILE_BYTES,
            per_tile_mismatch[2], TILE_BYTES,
            per_tile_mismatch[3], TILE_BYTES,
            total_mismatch, DUMP_BYTES);

    if (total_mismatch == 0) {
        fprintf(stderr,
            "OK: 4-tile B-pretranspose prolog produces N-major tiles that\n"
            "    byte-equal the host transpose of HBM B[K=256,N=256].\n"
            "    Session 13 (= Session 9.2) can now wire this prolog into\n"
            "    grouped_rrr_kernel_body_pinned under RRR_B_PRETRANS=1.\n");
        return 0;
    }
    fprintf(stderr, "FAIL: see counters above.\n");
    return 2;
}
