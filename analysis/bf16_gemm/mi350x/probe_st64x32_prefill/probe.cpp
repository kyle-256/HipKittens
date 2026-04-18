// P23 Dev B — LDS-peek probe for st_64x32_padded_b128 via TK's prefill_swizzled_offsets path.
//
// Purpose: prove that
//   (a) `prefill_swizzled_offsets` + `load(..., swizzled_offsets)` over the
//       new shape places HBM element A[m,k] at LDS byte
//          subtile_id * (4096 + 32) + (m % 64) * 64 + (k % 32) * 2
//       where subtile_id = (m / 64) * 2 + (k / 32),
//   (b) the within-subtile swizzle is identity (no XOR, no transpose),
//   (c) the +32 B per-subtile padding lands between subtiles (not within).
//
// Per feedback_dtl_probe_kernel.md, the probe inserts s_waitcnt vmcnt(0)
// + lgkmcnt(0) + __syncthreads() before reading LDS back; otherwise half the
// lanes' DTL writes are silently lost.
//
// Build: see Makefile. Runs single-block grid, 8 warps (NUM_THREADS=512),
// matching the BF16 RCR kernel's group size.

#include "kittens.cuh"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

using namespace kittens;

#define HIPCHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "%s:%d HIP %s\n", __FILE__, __LINE__, hipGetErrorString(e)); std::exit(1); } } while(0)

constexpr int M_TILE        = 128;
constexpr int K_TILE        = 64;
constexpr int NUM_WARPS     = 8;
constexpr int NUM_THREADS   = WARP_THREADS * NUM_WARPS;  // 512
constexpr int SUBTILE_BYTES = 64 * 32 * 2;               // 4096
constexpr int PAD_BYTES     = 32;
constexpr int STRIDE_BYTES  = SUBTILE_BYTES + PAD_BYTES; // 4128
constexpr int NUM_SUBTILES  = (M_TILE / 64) * (K_TILE / 32); // 4
constexpr int LDS_BYTES     = NUM_SUBTILES * STRIDE_BYTES;   // 16,512

using ST   = st_bf<M_TILE, K_TILE, st_64x32_padded_b128_s>;
using GL   = gl<bf16, -1, -1, -1, -1>;
using G    = kittens::group<NUM_WARPS>;

// Static-instantiation gate: if the new shape ever fails the
// shared_to_register::load(row_l) / load(col_l) template branches
// (`shared_to_register.cuh:50` / `:117`), this gives a hard compile error
// at probe build time instead of inside the production kernel.
//
// Both ST_A=ST and a second instantiation matching ST_B in BF16 RCR
// (same shape) exercise the same branch (`underlying_subtile_rows >=
// RT::base_tile_rows` for the b128 path), but we instantiate two
// register-tile shapes to hit both compile-time branches.
namespace static_branch_check {
    using ST_AB = st_bf<M_TILE, K_TILE, st_64x32_padded_b128_s>;

    // Branch ":50" — large shared subtile vs register subtile (b128 read).
    // For RCR, A_reg = rt_bf<HALF_REG_BLOCK_M=128, K_STEP=64, row_l, rt_16x32_s>,
    // i.e. base_tile_rows=16, base_tile_cols=32 — st_64x32 (64 >= 16, 32 >= 32) → branch :50.
    static_assert(ST_AB::underlying_subtile_rows == 64);
    static_assert(ST_AB::underlying_subtile_cols == 32);
    static_assert(ST_AB::subtile_padding == 32);
    static_assert(ST_AB::underlying_subtile_stride_bytes == 4128);
    static_assert(ST_AB::underlying_subtile_bytes_per_thread == 16);

    // Branch ":117" — small shared subtile vs register subtile.
    // For shapes like rt_32x16, would route through :117. We only want to
    // confirm the shape compiles cleanly when we ASK the compiler to
    // resolve it; we don't actually call load() here (avoids needing a
    // running kernel). The static_asserts above are sufficient to prove
    // the type forms.

    // ST_B in RCR is the same shape — independently instantiate to force
    // a second specialization template path.
    using ST_B_clone = st_bf<M_TILE, K_TILE, st_64x32_padded_b128_s>;
    static_assert(ST_B_clone::underlying_subtile_stride_bytes == 4128);
}

struct probe_globals {
    GL a;
    bf16 *lds_dump;       // device buffer to receive LDS image (LDS_BYTES bytes)
    int  m, n, k;
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void probe_kernel(const probe_globals g)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    ST (&As)[1] = al.allocate<ST, 1>();

    // Zero LDS with a sentinel before any DTL.
    uint16_t *sm_u16 = reinterpret_cast<uint16_t*>(&As[0].data[0]);
    int tid = threadIdx.x;
    int total_u16 = sizeof(As[0]) / sizeof(uint16_t);
    for (int i = tid; i < total_u16; i += NUM_THREADS) sm_u16[i] = 0xDEAD;
    asm volatile("s_waitcnt lgkmcnt(0)\n\t");
    __syncthreads();

    // Phase 1: prefill_swizzled_offsets + load.
    // Per-thread private array (matches kernel_bf16_dynamic.cpp:264). Each
    // lane stores its own per-memcpy global byte offsets in registers.
    constexpr int memcpy_per_tile =
        ST::rows * ST::cols * sizeof(bf16) /
        (ST::underlying_subtile_bytes_per_thread * NUM_THREADS);
    static_assert(memcpy_per_tile == 2, "expect 2 memcpy iters for st_64x32_padded_b128");
    uint32_t swizzled_offsets_A[memcpy_per_tile];
    G::prefill_swizzled_offsets(As[0], g.a, swizzled_offsets_A);
    G::load(As[0], g.a, {0, 0, 0, 0}, swizzled_offsets_A);

    // CRITICAL gfx950 DTL barrier: vmcnt(0) for the buffer_load → LDS path,
    // lgkmcnt(0) for the LDS write retire, then __syncthreads() to fence
    // across waves. Skipping this silently drops half the lanes' writes
    // (per feedback_dtl_probe_kernel.md).
    asm volatile("s_waitcnt vmcnt(0)\n\t");
    asm volatile("s_waitcnt lgkmcnt(0)\n\t");
    __syncthreads();

    // Phase 2: dump full LDS region to global so the host can verify.
    uint16_t *out_u16 = reinterpret_cast<uint16_t*>(g.lds_dump);
    for (int i = tid; i < total_u16; i += NUM_THREADS) out_u16[i] = sm_u16[i];
    __syncthreads();

    // Phase 3: force shared_to_register::load(rt, st) template-branch
    // resolution at COMPILE TIME for the new shape. This instantiates the
    // ST::underlying_subtile_rows >= RT::base_tile_rows && cols >= cols
    // branch (`shared_to_register.cuh:50`). The stored result is dumped
    // through volatile to keep DCE from removing the call.
    //
    // Branch ":117" (small shared subtile) cannot be reached for
    // st_64x32 because no existing rt shape has both rows >= 64 and
    // cols >= 32; that branch is `if constexpr` gated and simply not
    // instantiated for our shape, so the only branch we MUST validate
    // for codegen is :50.
    using A_reg_t = rt_bf<64, 32, row_l, rt_16x32_s>;  // 1 wave's view of the new ST
    A_reg_t reg_tile;
    auto sub = subtile_inplace<64, 32>(As[0], {0, 0});
    load(reg_tile, sub);
    // Touch the loaded data so DCE can't remove it.
    if (tid == 0) {
        volatile bf16 sink = reg_tile.tiles[0][0].data[0].x;
        (void)sink;
    }
}

int main(int argc, char** argv)
{
    int M = M_TILE, K = K_TILE;

    // Sentinel: A[m,k] = bf16( ((m & 0x7f) << 7) | (k & 0x7f) ).
    //   - Encodes 14 bits of (m,k) info into the bf16 mantissa+exp slots.
    //   - We compare host-known bit patterns vs device-read bit patterns,
    //     so the bf16 numerical interpretation doesn't matter; raw u16
    //     compare suffices.
    std::vector<uint16_t> h_A(M * K);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
            h_A[m * K + k] = (uint16_t)(((m & 0x7f) << 7) | (k & 0x7f));

    bf16 *d_A = nullptr;
    bf16 *d_lds = nullptr;
    HIPCHECK(hipMalloc(&d_A, M * K * sizeof(uint16_t)));
    HIPCHECK(hipMalloc(&d_lds, LDS_BYTES));
    HIPCHECK(hipMemcpy(d_A, h_A.data(), M * K * sizeof(uint16_t), hipMemcpyHostToDevice));
    HIPCHECK(hipMemset(d_lds, 0xCC, LDS_BYTES));

    probe_globals g{
        GL((__hip_bfloat16*)d_A, 1, 1, M, K),
        d_lds,
        M, 0, K
    };

    constexpr size_t SHM = LDS_BYTES + 256;  // some slack for allocator
    HIPCHECK(hipFuncSetAttribute((const void*)probe_kernel,
        hipFuncAttributeMaxDynamicSharedMemorySize, SHM));
    probe_kernel<<<dim3(1), dim3(NUM_THREADS), SHM>>>(g);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());

    std::vector<uint16_t> h_lds(LDS_BYTES / 2);
    HIPCHECK(hipMemcpy(h_lds.data(), d_lds, LDS_BYTES, hipMemcpyDeviceToHost));

    // -------- Closed-form verification --------
    // For st_bf<128,64,st_64x32_padded_b128>:
    //   subtile_id(m,k) = (m / 64) * 2 + (k / 32)
    //   within_subtile(m,k) = (m % 64) * 64 + (k % 32) * 2     [bytes]
    //   lds_byte(m,k) = subtile_id * 4128 + within_subtile(m,k)
    int errs = 0, checked = 0;
    int first_err_logged = 0;
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            int subtile_id = (m / 64) * 2 + (k / 32);
            int within     = (m % 64) * 64 + (k % 32) * 2;
            int lds_byte   = subtile_id * STRIDE_BYTES + within;
            int slot       = lds_byte / 2;
            uint16_t got    = h_lds[slot];
            uint16_t expect = (uint16_t)(((m & 0x7f) << 7) | (k & 0x7f));
            if (got != expect) {
                errs++;
                if (first_err_logged < 10) {
                    printf("MISMATCH m=%3d k=%3d subt=%d byte=%5d slot=%5d "
                           "expect=0x%04x got=0x%04x\n",
                           m, k, subtile_id, lds_byte, slot, expect, got);
                    first_err_logged++;
                }
            }
            checked++;
        }
    }
    printf("Per-cell check: %d / %d cells matched closed form.\n",
           checked - errs, checked);

    // Verify pad bytes are still sentinel (0xDEAD u16) — proves padding
    // lives BETWEEN subtiles, not consumed by writes.
    int pad_errs = 0;
    for (int s = 0; s < NUM_SUBTILES - 1; ++s) {
        for (int b = 0; b < PAD_BYTES; b += 2) {
            int slot = (s * STRIDE_BYTES + SUBTILE_BYTES + b) / 2;
            if (h_lds[slot] != 0xDEAD) {
                pad_errs++;
                if (pad_errs <= 4) {
                    printf("PAD WRITTEN: subtile %d pad+%d slot=%d val=0x%04x (expect 0xDEAD)\n",
                           s, b, slot, h_lds[slot]);
                }
            }
        }
    }
    printf("Pad check: %d violations (expect 0).\n", pad_errs);

    bool ok = (errs == 0) && (pad_errs == 0);
    printf("\n=== %s ===\n", ok ? "PROBE PASS" : "PROBE FAIL");
    if (ok) {
        printf("  st_64x32_padded_b128 + prefill_swizzled_offsets:\n");
        printf("    - HBM voffset reconstruction is bit-identity (no swizzle inversion needed).\n");
        printf("    - +32 B padding correctly inserted between 4096-B subtiles.\n");
        printf("    - Within-subtile layout = standard row-major (matches identity swizzle).\n");
        printf("    - Per-cell formula:  lds_byte(m,k) = subtile_id*4128 + (m%%64)*64 + (k%%32)*2\n");
        printf("      where subtile_id = (m/64)*2 + (k/32).\n");
        return 0;
    }
    return 1;
}
