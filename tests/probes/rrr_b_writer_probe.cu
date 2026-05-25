// rrr_b_writer_probe.cu
//
// Session 4 of RRR v2 B-pretranspose campaign — Path L (LDS staging).
//
// Validates a HBM→LDS B-transpose writer that takes row-major B[K][N]
// (K stride=N bytes) from HBM and produces N-major Bs[N][K] in LDS
// (identity swizzle, byte_offset(n,k)=n*128+k — the layout Session 3
// st_128x128_n_major + load specialization expects).
//
// Algorithm (Path L = LDS staging, see session4_writer_design.md):
//   Phase 1: 8-warp coop load HBM → staging LDS (K-major, 16 KB)
//   Phase 2: __syncthreads()
//   Phase 3: 8-warp coop read staging (16 ds_read_b8/lane, gather N-col) →
//            register 16-byte → 1 ds_write_b128 to final Bs N-major (16 KB)
//
// Two verifications:
//   A. Direct byte-compare of final Bs against host reference
//      ref[n*128 + k] = hbm[k*128 + n].
//   B. Round-trip via Session 3 load(rt_128x16_s, st_128x128_n_major):
//      load final Bs → RT → dump RT → check against Session 1
//      (lane, byte) → (k, n) mapping.
//
// Build (chi2811 gfx950):
//   make -C tests/probes rrr_b_writer_probe
// Run:
//   ./tests/probes/rrr_b_writer_probe

#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

using namespace kittens;

constexpr int K_DIM = 128;
constexpr int N_DIM = 128;
constexpr int TILE_BYTES = K_DIM * N_DIM;   // 16384

using ST = st_fp8e4m3<128, 128, st_128x128_n_major_s>;
using RT = rt_fp8e4m3<128, 128, col_l, rt_128x16_s>;

static_assert(sizeof(ST) == TILE_BYTES, "ST size mismatch");

// =====================================================================
// Phase A: writer probe — verify writer produces correct Bs bytes.
// =====================================================================
__global__ void __launch_bounds__(512, 1)
b_writer_path_L(const uint8_t* __restrict__ hbm_b,
                uint8_t* __restrict__ dst_dump)
{
    __shared__ __align__(16) uint8_t Bs_stage[TILE_BYTES];
    __shared__ ST                    Bs_final;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 6;     // 0..7
    const int lane_id = tid & 63;     // 0..63

    // -------- Phase 1+2: HBM → staging LDS (K-major) --------
    //   warp w handles K rows [w*16, w*16+16).
    //   Per iter: each lane does 1 b128 load (16 contiguous N bytes for 1 K row).
    //   2 iter × 64 lanes/warp × 16 bytes/lane = 16K × 128N = 2048 bytes/warp covered.
    #pragma unroll
    for (int iter = 0; iter < 2; ++iter) {
        const int n_block_base = iter * 64;
        const int k_local      = lane_id & 15;
        const int n_chunk      = (lane_id >> 4) & 3;
        const int K_row        = warp_id * 16 + k_local;
        const int N_col_start  = n_block_base + n_chunk * 16;
        const size_t hbm_off   = (size_t)K_row * N_DIM + N_col_start;
        const size_t lds_off   = (size_t)K_row * N_DIM + N_col_start;
        __uint128_t v = *reinterpret_cast<const __uint128_t*>(&hbm_b[hbm_off]);
        *reinterpret_cast<__uint128_t*>(&Bs_stage[lds_off]) = v;
    }
    __syncthreads();

    // -------- Phase 3+4: staging → final Bs (N-major) --------
    //   warp w handles K-strip [w*16, w*16+16) of FINAL.
    //   Per iter: each lane handles 1 N-row in final. lane_id ∈ [0,64),
    //   n_block_base = iter*64 → N_row = n_block_base + lane_id ∈ [0,128).
    //   For (K=w*16+0..15, N=N_row): 16 strided bytes in staging at
    //     staging[(w*16 + k_in_strip)*128 + N_row] for k_in_strip=0..15.
    //   Gather via 16 ds_read_b8 → 1 ds_write_b128 to Bs_final.
    //   (Sub-optimal LDS pattern — Session 4.1 will replace with cross-lane.)
    uint8_t* Bs_final_raw = reinterpret_cast<uint8_t*>(&Bs_final.data[0]);
    #pragma unroll
    for (int iter = 0; iter < 2; ++iter) {
        const int n_block_base = iter * 64;
        const int N_row        = n_block_base + lane_id;
        const int K_strip      = warp_id * 16;

        uint8_t out16[16];
        #pragma unroll
        for (int k_in_strip = 0; k_in_strip < 16; ++k_in_strip) {
            const int K_global = K_strip + k_in_strip;
            out16[k_in_strip] = Bs_stage[(size_t)K_global * N_DIM + N_row];
        }
        const size_t final_off = (size_t)N_row * K_DIM + K_strip;
        *reinterpret_cast<__uint128_t*>(&Bs_final_raw[final_off]) =
            *reinterpret_cast<const __uint128_t*>(&out16[0]);
    }
    __syncthreads();

    // Dump Bs_final to HBM (byte-by-byte) for host verification.
    for (int i = tid; i < TILE_BYTES; i += 512) {
        dst_dump[i] = Bs_final_raw[i];
    }
}

// =====================================================================
// Phase B: round-trip — writer + Session 3 load → RT dump
// =====================================================================
__global__ void __launch_bounds__(512, 1)
b_writer_roundtrip(const uint8_t* __restrict__ hbm_b,
                   uint8_t* __restrict__ rt_dump)
{
    __shared__ __align__(16) uint8_t Bs_stage[TILE_BYTES];
    __shared__ ST                    Bs_final;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 6;
    const int lane_id = tid & 63;

    // ---- Same writer phases as b_writer_path_L ----
    #pragma unroll
    for (int iter = 0; iter < 2; ++iter) {
        const int n_block_base = iter * 64;
        const int k_local      = lane_id & 15;
        const int n_chunk      = (lane_id >> 4) & 3;
        const int K_row        = warp_id * 16 + k_local;
        const int N_col_start  = n_block_base + n_chunk * 16;
        const size_t hbm_off   = (size_t)K_row * N_DIM + N_col_start;
        const size_t lds_off   = (size_t)K_row * N_DIM + N_col_start;
        __uint128_t v = *reinterpret_cast<const __uint128_t*>(&hbm_b[hbm_off]);
        *reinterpret_cast<__uint128_t*>(&Bs_stage[lds_off]) = v;
    }
    __syncthreads();

    uint8_t* Bs_final_raw = reinterpret_cast<uint8_t*>(&Bs_final.data[0]);
    #pragma unroll
    for (int iter = 0; iter < 2; ++iter) {
        const int n_block_base = iter * 64;
        const int N_row        = n_block_base + lane_id;
        const int K_strip      = warp_id * 16;
        uint8_t out16[16];
        #pragma unroll
        for (int k_in_strip = 0; k_in_strip < 16; ++k_in_strip) {
            const int K_global = K_strip + k_in_strip;
            out16[k_in_strip] = Bs_stage[(size_t)K_global * N_DIM + N_row];
        }
        const size_t final_off = (size_t)N_row * K_DIM + K_strip;
        *reinterpret_cast<__uint128_t*>(&Bs_final_raw[final_off]) =
            *reinterpret_cast<const __uint128_t*>(&out16[0]);
    }
    __syncthreads();

    // ---- Session 3 load: Bs_final → RT b ----
    // Only 1 warp dumps RT (single warp == 64 lanes is sufficient since
    // the load() specialization is per-warp; using all 8 warps would dump
    // identical RT 8 times).
    if (warp_id == 0) {
        RT b;
        kittens::load(b, Bs_final);

        // Dump b's per-lane bytes: 8 N-tiles × 64 lanes × 32 bytes = 16384.
        #pragma unroll
        for (int j = 0; j < RT::width; ++j) {
            const uint8_t* lane_bytes = reinterpret_cast<const uint8_t*>(
                &b.tiles[0][j].data[0]);
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                int off = (j * 64 + lane_id) * 32 + i;
                rt_dump[off] = lane_bytes[i];
            }
        }
    }
}

#define CK(stmt) do { hipError_t e = (stmt); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,    \
            hipGetErrorString(e)); std::exit(1); } } while(0)

int main(int argc, char** argv)
{
    // Build input HBM B[K=128][N=128] with deterministic pattern.
    // For verify-A: byte = (k*8 + n) & 0xFF (k-dominant pattern).
    // For verify-B: separately use modeA (byte=k) and modeB (byte=n) so we
    //               can recover k and n from RT dump independently.
    std::vector<uint8_t> hbm_byk(TILE_BYTES), hbm_byn(TILE_BYTES);
    std::vector<uint8_t> hbm_mixed(TILE_BYTES);
    for (int k = 0; k < K_DIM; ++k) {
        for (int n = 0; n < N_DIM; ++n) {
            hbm_byk[k*N_DIM + n]   = (uint8_t)(k & 0xFF);
            hbm_byn[k*N_DIM + n]   = (uint8_t)(n & 0xFF);
            hbm_mixed[k*N_DIM + n] = (uint8_t)(((k*8 + n) & 0xFF));
        }
    }

    uint8_t *d_hbm = nullptr, *d_dump = nullptr;
    CK(hipMalloc(&d_hbm,  TILE_BYTES));
    CK(hipMalloc(&d_dump, TILE_BYTES));

    // ---------- VERIFY A: direct byte-compare of Bs_final vs host ref ----------
    CK(hipMemcpy(d_hbm, hbm_mixed.data(), TILE_BYTES, hipMemcpyHostToDevice));
    CK(hipMemset(d_dump, 0xAA, TILE_BYTES));
    b_writer_path_L<<<1, 512>>>(d_hbm, d_dump);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    std::vector<uint8_t> got_final(TILE_BYTES);
    CK(hipMemcpy(got_final.data(), d_dump, TILE_BYTES, hipMemcpyDeviceToHost));

    // Host ref: ref[n*128 + k] = hbm_mixed[k*128 + n].
    int mismatchA = 0;
    for (int n = 0; n < N_DIM; ++n) {
        for (int k = 0; k < K_DIM; ++k) {
            uint8_t expect = hbm_mixed[k*N_DIM + n];
            uint8_t got    = got_final[n*K_DIM + k];
            if (got != expect) {
                if (mismatchA < 8) {
                    fprintf(stderr,
                        "VERIFY-A MISMATCH n=%d k=%d: got=0x%02x expect=0x%02x\n",
                        n, k, (int)got, (int)expect);
                }
                ++mismatchA;
            }
        }
    }

    // ---------- VERIFY B: round-trip through Session 3 load ----------
    constexpr int RT_DUMP_BYTES = 8 * 64 * 32;  // = 16384
    uint8_t* d_rt_dump = nullptr;
    CK(hipMalloc(&d_rt_dump, RT_DUMP_BYTES));

    // Mode K (byte = k)
    CK(hipMemcpy(d_hbm, hbm_byk.data(), TILE_BYTES, hipMemcpyHostToDevice));
    CK(hipMemset(d_rt_dump, 0xFF, RT_DUMP_BYTES));
    b_writer_roundtrip<<<1, 512>>>(d_hbm, d_rt_dump);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    std::vector<uint8_t> rt_k(RT_DUMP_BYTES);
    CK(hipMemcpy(rt_k.data(), d_rt_dump, RT_DUMP_BYTES, hipMemcpyDeviceToHost));

    // Mode N (byte = n)
    CK(hipMemcpy(d_hbm, hbm_byn.data(), TILE_BYTES, hipMemcpyHostToDevice));
    CK(hipMemset(d_rt_dump, 0xFF, RT_DUMP_BYTES));
    b_writer_roundtrip<<<1, 512>>>(d_hbm, d_rt_dump);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    std::vector<uint8_t> rt_n(RT_DUMP_BYTES);
    CK(hipMemcpy(rt_n.data(), d_rt_dump, RT_DUMP_BYTES, hipMemcpyDeviceToHost));

    CK(hipFree(d_rt_dump));
    CK(hipFree(d_hbm));
    CK(hipFree(d_dump));

    // Session 1 closed-form: per (j, lane, byte):
    //   n_expect = j*16 + (lane & 15)
    //   k_block  = (lane >> 4) & 3
    //   half     = byte >> 4
    //   k_expect = k_block*16 + (byte & 15) + half*64
    int mismatchB = 0, bad_range = 0;
    for (int j = 0; j < 8; ++j) {
        for (int lane = 0; lane < 64; ++lane) {
            const int k_block = (lane >> 4) & 3;
            const int n_expect = j*16 + (lane & 15);
            for (int byte = 0; byte < 32; ++byte) {
                const int half = byte >> 4;
                const int k_expect = k_block*16 + (byte & 15) + half*64;
                const int off = (j*64 + lane)*32 + byte;
                const int got_k = rt_k[off];
                const int got_n = rt_n[off];
                if (got_k >= K_DIM || got_n >= N_DIM) { ++bad_range; continue; }
                if (got_k != k_expect || got_n != n_expect) {
                    if (mismatchB < 8) {
                        fprintf(stderr,
                            "VERIFY-B MISMATCH j=%d lane=%d byte=%d: "
                            "got=(k=%d,n=%d) expect=(k=%d,n=%d)\n",
                            j, lane, byte, got_k, got_n, k_expect, n_expect);
                    }
                    ++mismatchB;
                }
            }
        }
    }

    // Coverage check on round-trip (same as Session 3 probe).
    std::vector<int> cnt(128*128, 0);
    for (int j = 0; j < 8; ++j) {
        for (int lane = 0; lane < 64; ++lane) {
            for (int byte = 0; byte < 32; ++byte) {
                int off = (j*64 + lane)*32 + byte;
                int k = rt_k[off], n = rt_n[off];
                if (k < 128 && n < 128) ++cnt[k*128 + n];
            }
        }
    }
    int missing = 0, dup = 0;
    for (int v : cnt) { if (v == 0) ++missing; else if (v != 1) ++dup; }

    fprintf(stderr,
            "Session 4 writer probe:\n"
            "  VERIFY-A (direct byte-compare):  mismatch=%d   (tile=%d)\n"
            "  VERIFY-B (round-trip via load):  mismatch=%d   bad_range=%d\n"
            "                                   missing=%d   duplicate=%d\n",
            mismatchA, TILE_BYTES,
            mismatchB, bad_range, missing, dup);

    if (mismatchA == 0 && mismatchB == 0 && bad_range == 0 &&
        missing == 0 && dup == 0) {
        fprintf(stderr,
            "OK: writer produces N-major Bs identical to host transpose, AND\n"
            "    round-trip through Session 3 load == Session 1 mma_AB mapping.\n");
        return 0;
    }
    fprintf(stderr, "FAIL: see counters above.\n");
    return 2;
}
