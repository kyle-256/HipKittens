// MLA Backward Kernel — art (allocated register tile) version
// D_QK=192, D_V=128, DOT_SLICE_QO=16, KV_BLOCK=32
//
// Uses explicit register allocation to guarantee contiguous AGPRs.
// This avoids the gfx950 compiler bug where non-contiguous AGPR allocation
// causes v_accvgpr_read to use wrong sequential indices during transpose.
//
// Register allocation plan:
// ═══════════════════════════════════════════════════════════════
// AGPR (art index 256+):
//   a[0:95]   (256:351)  — dK_acc  192×32 col_l rt_32x32_s  6 tiles×16
//   a[96:159] (352:415)  — dV_acc  128×32 col_l rt_32x32_s  4 tiles×16
//   a[160:255]           — compiler scratch / S_ij / dP_ij temps
//
// VGPR (art index 0+):
//   v[0:23]   (0:23)     — Q_i / Q_col      16×192 row_l/col_l     24 regs (alias)
//   v[24:71]  (24:71)    — K_j              32×192 row_l            48 regs (reload from LDS)
//   v[72:79]  (72:79)    — S_ij/P_ij        16×32  col_l float      8 regs
//   v[80:95]  (80:95)    — dO_i             16×128 row_l            16 regs
//   v[96:127] (96:127)   — V_j              32×128 row_l            32 regs (reload from LDS)
//   v[128:135](128:135)  — dP_ij            16×32  col_l float      8 regs
//   v[136:151](136:151)  — dO_col           16×128 col_l            16 regs
//   v[152:155](152:155)  — P_mma/dS_mma     16×32  col_l bf16       4 regs
//   v[156:175]           — misc (addresses, loop vars, L, delta, neg_inf)
//   v[176:255]           — compiler scratch
// ═══════════════════════════════════════════════════════════════
//
// MMA pattern (following HNB):
//   Phase 1: S = Q_row @ K_j^T using mma_ABt (S in VGPR float)
//   Phase 2: dP = dO_row @ V_j^T using mma_ABt (dP in VGPR float)
//   Phase 3: dV += dO_col^T @ P_mma using art mma_AtB
//   Phase 4: dK += Q_col^T  @ dS_mma using art mma_AtB
//
// Store pattern:
//   dV: accvgpr_read(VGPR, dV_acc_AGPR) → transpose → bf16 → store
//   dK: accvgpr_read(VGPR, dK_acc_AGPR) → transpose → bf16 → store
//
// TODO: This is a skeleton. The mma_AtB calls for Phase 3/4 need the full
//       art operand declarations and manual unrolling (like HNB lines 388-725).
//       For now, the key contribution is the register allocation plan and the
//       verified root cause (non-contiguous AGPR allocation).

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#ifndef ATTN_B
constexpr int ATTN_B = 8;
#endif
#ifndef ATTN_H
constexpr int ATTN_H = 16;
#endif
#ifndef ATTN_H_KV
constexpr int ATTN_H_KV = 16;
#endif
constexpr int GROUP_SIZE = ATTN_H / ATTN_H_KV;
#ifndef ATTN_N
constexpr int ATTN_N = 2048;
#endif
#ifndef ATTN_D_QK
constexpr int ATTN_D_QK = 192;
#endif
#ifndef ATTN_D_V
constexpr int ATTN_D_V = 128;
#endif

constexpr int DOT_SLICE_QO = 16;  // Q rows per MMA tile (HNB-style, halved from 32)
constexpr int KV_BLOCK     = 32;
constexpr int BLOCK_KV     = KV_BLOCK * 4;
constexpr int STEP_QO      = 32;  // Q step along sequence (2 × DOT_SLICE_QO)
constexpr bool causal_flag  = false;

#define NUM_WARPS  4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;
using G   = kittens::group<NUM_WARPS>;
using _gl = gl<bf16, -1, -1, -1, -1>;

#define BS(b, s, h, d) {SBHD ? (s) : (b), SBHD ? (b) : (s), (h), (d)}

template<int DQK, int DV, bool SBHD=false> struct mla_bwd_globals {
    _gl Q, K, V, dOg, dKg, dVg;
    gl<float,-1,-1,-1,-1> L_vec, delta_vec;
    hipStream_t stream;
    dim3   grid()  { return dim3(ATTN_H_KV, ATTN_N / BLOCK_KV, ATTN_B); }
    dim3   block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

template<int DQK, int DV, bool SBHD=false> __launch_bounds__(NUM_THREADS, 1)
__global__ void mla_bwd_ker(const mla_bwd_globals<DQK, DV, SBHD> g) {
    // ── Art register ranges ──
    // AGPR accumulators (contiguous, prevents gfx950 bug)
    using dK_ranges = ducks::art::split_many_t<
        ducks::art::type_list<ducks::art::range<256, 351>>, 16>;  // a[0:95], 6×16
    using dV_ranges = ducks::art::split_many_t<
        ducks::art::type_list<ducks::art::range<352, 415>>, 16>;  // a[96:159], 4×16

    // VGPR operands for mma_AtB (Phase 3/4)
    // Q_col / dO_col: max 24 VGPRs (6 tiles × 4 for 192-dim col_l)
    using Acol_QK_ranges = ducks::art::split_many_t<
        ducks::art::type_list<ducks::art::range<0, 23>>, 4>;     // v[0:23], 6×4
    using Acol_V_ranges = ducks::art::split_many_t<
        ducks::art::type_list<ducks::art::range<0, 15>>, 4>;     // v[0:15], 4×4 (alias Q_col)
    using Bmma_ranges = ducks::art::split_many_t<
        ducks::art::type_list<ducks::art::range<152, 155>>, 4>;  // v[152:155], 1×4

    // Clobber all art ranges
    ducks::art::clobber<dK_ranges>();
    ducks::art::clobber<dV_ranges>();
    ducks::art::clobber<Acol_QK_ranges>();
    ducks::art::clobber<Bmma_ranges>();

    // ── Declare art accumulators ──
    art<float, DQK, KV_BLOCK, col_l, rt_32x32_s, dK_ranges> dK_acc;
    art<float, DV,  KV_BLOCK, col_l, rt_32x32_s, dV_ranges> dV_acc;

    // Manual AGPR zero
    { uint32_t z = 0;
      #pragma unroll
      for (int i = 0; i < 160; i++)
          asm volatile("v_accvgpr_write_b32 a[%0], %1" : : "n"(i), "v"(z) : "memory");
    }

    // ── The main loop and Phase 1-4 would go here ──
    // Phase 1: standard rt for S computation (mma_ABt with Q_row, K_j → S_ij)
    // Phase 2: standard rt for dP computation (mma_ABt with dO_row, V_j → dP_ij)
    // Phase 3: art mma_AtB for dV (dO_col_art, P_mma_art → dV_acc)
    // Phase 4: art mma_AtB for dK (Q_col_art, dS_mma_art → dK_acc)
    //
    // For Phase 3/4, all MMA operands must be art tiles.
    // The operand data comes from shared memory loads into art VGPRs.
    // The B operand (P/dS) comes from shared memory staging:
    //   1. Compute P/dS as standard rt float
    //   2. Convert to bf16
    //   3. Store to shared memory
    //   4. Load from shared into art bf16 tile (Bmma_ranges)
    //
    // This "standard rt → shared → art" bridge is the key pattern.

    // ── Epilogue: store via accvgpr_read ──
    // (placeholder — actual implementation reads each AGPR to VGPR,
    //  then uses standard transpose + store)

    const int kv_head = blockIdx.x, seq_block = blockIdx.y, batch = blockIdx.z;
    const int wid = kittens::warpid(), j = seq_block * NUM_WARPS + wid;

    // Placeholder stores (zeros for now)
    constexpr int QKVO_AXIS = SBHD ? 0 : 1;
    rt<bf16, KV_BLOCK, DQK, row_l, rt_32x32_s> dK_out_bf;
    zero(dK_out_bf);
    store<QKVO_AXIS>(g.dKg, dK_out_bf, BS(batch, j, kv_head, 0));

    rt<bf16, KV_BLOCK, DV, row_l, rt_32x32_s> dV_out_bf;
    zero(dV_out_bf);
    store<QKVO_AXIS>(g.dVg, dV_out_bf, BS(batch, j, kv_head, 0));
}

template<int DQK, int DV, bool SBHD>
void dispatch_mla_bwd(mla_bwd_globals<DQK, DV, SBHD> g) {
    unsigned long mem = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)mla_bwd_ker<DQK, DV, SBHD>, hipFuncAttributeMaxDynamicSharedMemorySize, mem);
    mla_bwd_ker<DQK, DV, SBHD><<<g.grid(), g.block(), mem, g.stream>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel_mla_bkwd_art, m) {
    m.doc() = "MLA backward kernel — art skeleton (D_QK=192, D_V=128)";
    py::bind_function<dispatch_mla_bwd<ATTN_D_QK, ATTN_D_V, false>>(m, "dispatch_bwd",
        &mla_bwd_globals<ATTN_D_QK, ATTN_D_V, false>::Q, &mla_bwd_globals<ATTN_D_QK, ATTN_D_V, false>::K,
        &mla_bwd_globals<ATTN_D_QK, ATTN_D_V, false>::V, &mla_bwd_globals<ATTN_D_QK, ATTN_D_V, false>::dOg,
        &mla_bwd_globals<ATTN_D_QK, ATTN_D_V, false>::dKg, &mla_bwd_globals<ATTN_D_QK, ATTN_D_V, false>::dVg,
        &mla_bwd_globals<ATTN_D_QK, ATTN_D_V, false>::L_vec, &mla_bwd_globals<ATTN_D_QK, ATTN_D_V, false>::delta_vec);
}
