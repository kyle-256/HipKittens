// ART-based Backward Kernel for Asymmetric Attention (D_QK=192, D_V=128)
// Uses Assigned Register Tiles (ART) for explicit VGPR/AGPR register management
// to eliminate gfx950 AGPR aliasing and compiler register allocation issues.
//
// dK+dV ONLY kernel (dQ stripped for performance).
// Use dispatch_bwd_dq from attn_bkwd_causal_d192v128.cpp for dQ.
//
// Key parameters:
//   WARP_SIZE_KV = 32 (reduced from 64 to fit D_QK=192)
//   DOT_SLICE_QO = 16
//   STEP_QO = 64 (= 4 x DOT_SLICE_QO)
//   BLOCK_SIZE_KV = 128 (= 4 warps x 32)
//   Grid: KV-parallel dim3(ATTN_H_KV, ATTN_N/BLOCK_SIZE_KV, ATTN_B)
//
// vs fused dK+dV+dQ: removed 16 MFMAs + 1 barrier + dS store per dot-slice

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "utils.cpp"

#ifndef ATTN_B
constexpr int ATTN_B = 16;
#endif
#ifndef ATTN_H
constexpr int ATTN_H = 64;
#endif
#ifndef ATTN_H_KV
constexpr int ATTN_H_KV = 8;
#endif
constexpr int GROUP_SIZE = ATTN_H / ATTN_H_KV;
#ifndef ATTN_N
constexpr int ATTN_N = 4096;
#endif

constexpr int D_QK = 192;
constexpr int D_V  = 128;

constexpr int STEP_QO = 64;
constexpr int BLOCK_SIZE_KV = 128;
constexpr int SLICE_QO = 32;         // Not used directly (reference compat)
constexpr int DOT_SLICE_QO = 16;
constexpr int WARP_SIZE_KV = 32;     // Reduced from 64 for D_QK=192
constexpr bool causal = true;

#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using G = kittens::group<NUM_WARPS>;
using namespace kittens;

// ---------------------------------------------------------------------------
// Globals (no dQg — dQ handled by separate kernel)
// ---------------------------------------------------------------------------
struct attn_bwd_d192v128_globals {
    gl<bf16, -1, -1, -1, -1> Q, K, V;
    gl<bf16, -1, -1, -1, -1> dOg, dKg, dVg;
    gl<float, -1, -1, -1, -1> L_vec, delta_vec;
    hipStream_t stream;
    dim3 grid()  { return dim3(ATTN_H_KV, (ATTN_N / BLOCK_SIZE_KV), ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
__launch_bounds__(NUM_THREADS, 1)
__global__ __attribute__((amdgpu_num_vgpr(29)))
void attend_bwd_d192v128_ker(const attn_bwd_d192v128_globals g) {

    const int kv_head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int first_q_head = kv_head_idx * GROUP_SIZE;

    const int warpid = kittens::warpid();
    const int j = seq_idx * NUM_WARPS + warpid;

    const int total_steps_per_head = ATTN_N / STEP_QO;
    const int j_min = seq_idx * NUM_WARPS;
    const int k_start_min = j_min * WARP_SIZE_KV;
    const int first_step = max(0, k_start_min / STEP_QO);
    const int num_steps_per_head = total_steps_per_head - first_step;
    const int num_steps = num_steps_per_head * GROUP_SIZE;
    const int k_pos = j * WARP_SIZE_KV;

    constexpr float L_SCALE_FACTOR = 1.44269504089f;
    constexpr float P_SCALE_FACTOR = 0.07216878365f * 1.44269504089f; // 1/sqrt(192) * log2(e)
    constexpr float dP_SCALE_FACTOR = 0.07216878365f;                 // 1/sqrt(192)

    // =====================================================================
    // Shared memory (no attn_i_smem — dQ removed)
    // =====================================================================
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // K in shared: [BLOCK_SIZE_KV x D_QK] = [128 x 192] = 48KB
    st_bf<BLOCK_SIZE_KV, D_QK, st_16x16_s> (&K_j_smem) = al.allocate<st_bf<BLOCK_SIZE_KV, D_QK, st_16x16_s>>();
    // Q double-buffered: [2][2] x [32 x 192]
    st_bf<SLICE_QO, D_QK, st_16x32_s> (&Q_i_smem)[2][2] = al.allocate<st_bf<SLICE_QO, D_QK, st_16x32_s>, 2, 2>();
    // dO double-buffered: [2][2] x [32 x 128]
    st_bf<SLICE_QO, D_V, st_16x32_s> (&dO_i_smem)[2][2] = al.allocate<st_bf<SLICE_QO, D_V, st_16x32_s>, 2, 2>();
    // L and delta double-buffered
    sv_fl<STEP_QO> (&L_smem)[2] = al.allocate<sv_fl<STEP_QO>, 2>();
    sv_fl<STEP_QO> (&delta_smem)[2] = al.allocate<sv_fl<STEP_QO>, 2>();

    // =====================================================================
    // ART Register Range Declarations
    // =====================================================================
    // Physical registers: 0-255 = VGPR, 256-511 = AGPR
    // __attribute__((amdgpu_num_vgpr(29))) reserves v[0:28] for compiler

    // --- Persistent accumulators ---
    // dK: [D_QK x WARP_SIZE_KV] = [192 x 32] col_l rt_32x32_s -> 6 tiles x 16 = 96 AGPRs
    using dK_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<256, 351>>, 16>;  // a[0:95]
    // dV: [D_V x WARP_SIZE_KV] = [128 x 32] col_l rt_32x32_s -> 4 tiles x 16 = 64 VGPRs
    using dV_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<128, 191>>, 16>;  // v[128:191]

    // --- K and V in registers ---
    // K_j: [WARP_SIZE_KV x D_QK] = [32 x 192] row_l rt_16x32_s -> 2x6 = 12 tiles x 4 = 48 regs
    using K_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<352, 399>>, 4>;  // a[96:143]
    // V_j: [WARP_SIZE_KV x D_V] = [32 x 128] row_l rt_16x32_s -> 2x4 = 8 tiles x 4 = 32 regs
    // No longer overlaps dS_T/K_col (dQ removed) — V persists across dot-slices
    using V_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<192, 223>>, 4>;  // v[192:223]
    // V_j split into two 16-row halves for correct global load
    using V_lo_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<192, 207>>, 4>;  // v[192:207], rows 0-15
    using V_hi_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<208, 223>>, 4>;  // v[208:223], rows 16-31

    // --- Q and dO tiles (loaded per dot-slice) ---
    // Q_i: [DOT_SLICE_QO x D_QK] = [16 x 192] row_l rt_16x32_s -> 1x6 = 6 tiles x 4 = 24 regs
    using Q_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<432, 455>>, 4>;  // a[176:199]
    // dO_i: [DOT_SLICE_QO x D_V] = [16 x 128] row_l rt_16x32_s -> 1x4 = 4 tiles x 4 = 16 regs
    using dO_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<62, 77>>, 4>;  // v[62:77]
    // dO_i_col: [DOT_SLICE_QO x D_V] = [16 x 128] col_l -> 16 regs
    using dO_col_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<78, 93>>, 4>;  // v[78:93], 4 tiles of 4 regs (rt_16x32_s)
    // Q_i_col: [DOT_SLICE_QO x D_QK] = [16 x 192] col_l -> 24 regs
    using Q_col_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<94, 117>>, 4>;  // v[94:117], 6 tiles of 4 regs (rt_16x32_s)

    // --- Attention tiles (temporary, heavily aliased) ---
    // P_ij: [DOT_SLICE_QO x WARP_SIZE_KV] = [16 x 32] col_l float -> 2 tiles x 4 = 8 regs
    using P_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<30, 37>>, 4>;  // v[30:37]
    // dP_ij: [16 x 32] col_l float -> 8 regs
    using dP_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<38, 45>>, 4>;  // v[38:45]
    // P_bf16: [16 x 32] col_l bf16 -> 4 regs
    using P_bf16_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<46, 49>>, 2>;  // v[46:49]
    // dP_bf16: [16 x 32] col_l bf16 -> 4 regs
    using dP_bf16_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<50, 53>>, 2>;  // v[50:53]
    // P_bf16_col: for mma_AtB dV
    using P_bf16_col_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<46, 49>>, 4>;  // same as P_bf16 (aliased)
    // dP_bf16_col: for mma_AtB dK
    using dP_bf16_col_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<50, 53>>, 4>;  // same as dP_bf16

    // --- dK epilogue chunk tiles (for AGPR->VGPR copy and store) ---
    // Chunk 2: dK tiles 4-5 (a[64:95]) -> v[128:159], [64x32] col_l, row view [32x64]
    using dK_chunk2_col_ranges = ducks::art::split_many_t<ducks::art::type_list<
        ducks::art::range<128, 159>>, 16>;  // v[128:159], 2 tiles
    using dK_chunk2_row_ranges = ducks::art::transpose_2d<dK_chunk2_col_ranges, 2, 1>;

    // --- Clobber all ranges ---
    ducks::art::clobber<dK_ranges>();
    ducks::art::clobber<dV_ranges>();
    ducks::art::clobber<K_ranges>();
    ducks::art::clobber<V_ranges>();
    ducks::art::clobber<V_lo_ranges>();
    ducks::art::clobber<V_hi_ranges>();
    ducks::art::clobber<Q_ranges>();
    ducks::art::clobber<dO_ranges>();
    ducks::art::clobber<dO_col_ranges>();
    ducks::art::clobber<Q_col_ranges>();
    ducks::art::clobber<P_ranges>();
    ducks::art::clobber<dP_ranges>();
    ducks::art::clobber<P_bf16_ranges>();
    ducks::art::clobber<dP_bf16_ranges>();
    ducks::art::clobber<dK_chunk2_col_ranges>();
    // Clobber v[54:61] — only v[61] used (dP_SCALE_FACTOR for dK epilogue)
    kittens::macros::clobber_gpr<54>();
    kittens::macros::clobber_gpr<55>();
    kittens::macros::clobber_gpr<56>();
    kittens::macros::clobber_gpr<57>();
    kittens::macros::clobber_gpr<58>();
    kittens::macros::clobber_gpr<59>();
    kittens::macros::clobber_gpr<60>();
    kittens::macros::clobber_gpr<61>();

    // --- Declare ART tiles ---
    // Persistent accumulators
    art<float, D_QK, WARP_SIZE_KV, col_l, rt_32x32_s, dK_ranges> dK_j_T;  // 96 AGPRs
    art<float, D_V,  WARP_SIZE_KV, col_l, rt_32x32_s, dV_ranges> dV_j_T;  // 64 VGPRs

    // K and V register tiles
    art<bf16, WARP_SIZE_KV, D_QK, row_l, rt_16x32_s, K_ranges> K_j;  // 48 AGPRs
    art<bf16, WARP_SIZE_KV, D_V,  row_l, rt_16x32_s, V_ranges> V_j;  // 32 VGPRs
    // Half-V tiles for correct global load (16 rows each)
    art<bf16, DOT_SLICE_QO, D_V, row_l, rt_16x32_s, V_lo_ranges> V_j_lo;  // v[192:207], rows 0-15
    art<bf16, DOT_SLICE_QO, D_V, row_l, rt_16x32_s, V_hi_ranges> V_j_hi;  // v[208:223], rows 16-31

    // Q and dO tiles
    art<bf16, DOT_SLICE_QO, D_QK, row_l, rt_16x32_s, Q_ranges> Q_i;      // 24 AGPRs
    art<bf16, DOT_SLICE_QO, D_V,  row_l, rt_16x32_s, dO_ranges> dO_i;    // 16 VGPRs
    art<bf16, DOT_SLICE_QO, D_V,  col_l, rt_16x32_s, dO_col_ranges> dO_i_col;  // 16 VGPRs
    art<bf16, DOT_SLICE_QO, D_QK, col_l, rt_16x32_s, Q_col_ranges> Q_i_col;    // 24 VGPRs

    // Attention tiles
    art<float, DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s, P_ranges> P_ij;    // 8 VGPRs
    art<float, DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s, dP_ranges> dP_ij;  // 8 VGPRs
    art<bf16, DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s, P_bf16_ranges> P_ij_bf16;
    art<bf16, DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s, dP_bf16_ranges> dP_ij_bf16;
    art<bf16, DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x32_s, P_bf16_col_ranges> P_ij_bf16_col;
    art<bf16, DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x32_s, dP_bf16_col_ranges> dP_ij_bf16_col;

    // Row views for store
    // dV: [128x32] col_l 4x1 tiles -> [32x128] row_l 1x4
    art<float, WARP_SIZE_KV, D_V, row_l, rt_32x32_s,
        ducks::art::transpose_2d<dV_ranges, 4, 1>> dV_j;
    // dK: [192x32] col_l 6x1 tiles -> [32x192] row_l 1x6
    art<float, WARP_SIZE_KV, D_QK, row_l, rt_32x32_s,
        ducks::art::transpose_2d<dK_ranges, 6, 1>> dK_j;

    // dK chunk 2 for epilogue: [64x32] col_l in v[128:159], row view [32x64]
    art<float, 64, WARP_SIZE_KV, col_l, rt_32x32_s, dK_chunk2_col_ranges> dK_chunk2_T;
    art<float, WARP_SIZE_KV, 64, row_l, rt_32x32_s, dK_chunk2_row_ranges> dK_chunk2;

    // Scalar registers
    constexpr int L_i = 126;
    constexpr int delta_i = 127;
    constexpr int neg_inf_v = 29;
    kittens::macros::clobber_gpr<neg_inf_v>();
    kittens::macros::v_mov_b32_up2p<neg_inf_v>(0xff800000u);

    // =====================================================================
    // Prologue: Load K to shared, V to registers
    // =====================================================================
    int tic = 0, toc = 1;

    // Load K from global to shared memory
    G::load<1, false>(K_j_smem, g.K, {batch_idx, seq_idx, kv_head_idx, 0});

    // Load V from global directly to registers (split into two 16-row halves)
    load<1, 0>(V_j_lo, g.V, {batch_idx, 0, kv_head_idx, 0}, {0, j * 2,     0, 0});
    load<1, 0>(V_j_hi, g.V, {batch_idx, 0, kv_head_idx, 0}, {0, j * 2 + 1, 0, 0});

    // Prefetch first Q, dO, L, delta (reverse order: start from highest Q position)
    const int last_step = total_steps_per_head - 1;
    load(L_smem[tic], g.L_vec, {batch_idx, first_q_head, 0, last_step});
    load(delta_smem[tic], g.delta_vec, {batch_idx, first_q_head, 0, last_step});
    G::load<1, false>(Q_i_smem[tic][0],  g.Q,   {batch_idx, last_step * 2 + 0, first_q_head, 0});
    G::load<1, false>(dO_i_smem[tic][0], g.dOg, {batch_idx, last_step * 2 + 0, first_q_head, 0});
    G::load<1, false>(Q_i_smem[tic][1],  g.Q,   {batch_idx, last_step * 2 + 1, first_q_head, 0});
    G::load<1, false>(dO_i_smem[tic][1], g.dOg, {batch_idx, last_step * 2 + 1, first_q_head, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Compute addresses
    const uint32_t K_j_addr = get_address(K_j, subtile_inplace<WARP_SIZE_KV, D_QK>(K_j_smem, {warpid, 0}));

    uint32_t Q_i_addr;
    uint32_t dO_i_addr;
    uint32_t Q_i_col_addr;

    // R87: tracks whether Q_i_addr/dO_i_addr were precomputed in the
    // previous ds iteration's dK MFMA shadow. Reset at start of each step.
    bool addr_precomputed = false;

    // =====================================================================
    // Load K_j from LDS to AGPRs ONCE (K_j is persistent — same K block
    // for all dot-slices and steps, a[96:143] never clobbered by MFMAs).
    // =====================================================================
    asm volatile(
        // Group 1: tiles (0,0)-(0,5), offsets 0-5120
        "ds_read_b128 v[32:35], %0 offset:0\n"
        "ds_read_b128 v[44:47], %0 offset:1024\n"
        "ds_read_b128 v[48:51], %0 offset:2048\n"
        "ds_read_b128 v[52:55], %0 offset:3072\n"
        "ds_read_b128 v[56:59], %0 offset:4096\n"
        "ds_read_b128 v[60:63], %0 offset:5120\n"
        "s_waitcnt lgkmcnt(0)\n"
        "v_accvgpr_write_b32 a[96], v[32]\n"
        "v_accvgpr_write_b32 a[97], v[33]\n"
        "v_accvgpr_write_b32 a[98], v[34]\n"
        "v_accvgpr_write_b32 a[99], v[35]\n"
        "v_accvgpr_write_b32 a[100], v[44]\n"
        "v_accvgpr_write_b32 a[101], v[45]\n"
        "v_accvgpr_write_b32 a[102], v[46]\n"
        "v_accvgpr_write_b32 a[103], v[47]\n"
        "v_accvgpr_write_b32 a[104], v[48]\n"
        "v_accvgpr_write_b32 a[105], v[49]\n"
        "v_accvgpr_write_b32 a[106], v[50]\n"
        "v_accvgpr_write_b32 a[107], v[51]\n"
        "v_accvgpr_write_b32 a[108], v[52]\n"
        "v_accvgpr_write_b32 a[109], v[53]\n"
        "v_accvgpr_write_b32 a[110], v[54]\n"
        "v_accvgpr_write_b32 a[111], v[55]\n"
        "v_accvgpr_write_b32 a[112], v[56]\n"
        "v_accvgpr_write_b32 a[113], v[57]\n"
        "v_accvgpr_write_b32 a[114], v[58]\n"
        "v_accvgpr_write_b32 a[115], v[59]\n"
        "v_accvgpr_write_b32 a[116], v[60]\n"
        "v_accvgpr_write_b32 a[117], v[61]\n"
        "v_accvgpr_write_b32 a[118], v[62]\n"
        "v_accvgpr_write_b32 a[119], v[63]\n"
        // Group 2: tiles (1,0)-(1,5), offsets 6144-11264
        "ds_read_b128 v[32:35], %0 offset:6144\n"
        "ds_read_b128 v[44:47], %0 offset:7168\n"
        "ds_read_b128 v[48:51], %0 offset:8192\n"
        "ds_read_b128 v[52:55], %0 offset:9216\n"
        "ds_read_b128 v[56:59], %0 offset:10240\n"
        "ds_read_b128 v[60:63], %0 offset:11264\n"
        "s_waitcnt lgkmcnt(0)\n"
        "v_accvgpr_write_b32 a[120], v[32]\n"
        "v_accvgpr_write_b32 a[121], v[33]\n"
        "v_accvgpr_write_b32 a[122], v[34]\n"
        "v_accvgpr_write_b32 a[123], v[35]\n"
        "v_accvgpr_write_b32 a[124], v[44]\n"
        "v_accvgpr_write_b32 a[125], v[45]\n"
        "v_accvgpr_write_b32 a[126], v[46]\n"
        "v_accvgpr_write_b32 a[127], v[47]\n"
        "v_accvgpr_write_b32 a[128], v[48]\n"
        "v_accvgpr_write_b32 a[129], v[49]\n"
        "v_accvgpr_write_b32 a[130], v[50]\n"
        "v_accvgpr_write_b32 a[131], v[51]\n"
        "v_accvgpr_write_b32 a[132], v[52]\n"
        "v_accvgpr_write_b32 a[133], v[53]\n"
        "v_accvgpr_write_b32 a[134], v[54]\n"
        "v_accvgpr_write_b32 a[135], v[55]\n"
        "v_accvgpr_write_b32 a[136], v[56]\n"
        "v_accvgpr_write_b32 a[137], v[57]\n"
        "v_accvgpr_write_b32 a[138], v[58]\n"
        "v_accvgpr_write_b32 a[139], v[59]\n"
        "v_accvgpr_write_b32 a[140], v[60]\n"
        "v_accvgpr_write_b32 a[141], v[61]\n"
        "v_accvgpr_write_b32 a[142], v[62]\n"
        "v_accvgpr_write_b32 a[143], v[63]\n"
        :: "v"(K_j_addr) :);

    // =====================================================================
    // MAIN LOOP — dK+dV only (22 MFMAs per dot-slice)
    // =====================================================================
    zero(dK_j_T);
    zero(dV_j_T);

    // v[61] = dP_SCALE_FACTOR for dK epilogue scaling
    {
        float dp_scale = dP_SCALE_FACTOR;
        asm volatile("v_mov_b32 v[61], %0\n" :: "v"(dp_scale));
    }

    for (int step = 0; step < num_steps; step++) {
        const int q_head_idx = step / num_steps_per_head + first_q_head;
        // Reverse Q iteration: start from highest position, work down.
        // This maximizes L2 cache sharing across KV blocks, since all blocks
        // for the same (head, batch) start at the same highest Q position.
        const int q_seq_idx = total_steps_per_head - 1 - (step % num_steps_per_head);
        const int q_pos_base = q_seq_idx * STEP_QO;

        // Software-pipelined prefetch: issue loads for NEXT step at the
        // beginning of the current step so global load latency overlaps
        // with the 4 dot-slices of compute.
        if (step + 1 < num_steps) {
            const int ns_head = (step + 1) / num_steps_per_head + first_q_head;
            const int ns_seq  = total_steps_per_head - 1 - ((step + 1) % num_steps_per_head);
            toc = 1 - tic;
            load(L_smem[toc], g.L_vec, {batch_idx, ns_head, 0, ns_seq});
            load(delta_smem[toc], g.delta_vec, {batch_idx, ns_head, 0, ns_seq});
            G::load<1, false>(Q_i_smem[toc][0],  g.Q,   {batch_idx, ns_seq * 2 + 0, ns_head, 0});
            G::load<1, false>(dO_i_smem[toc][0], g.dOg, {batch_idx, ns_seq * 2 + 0, ns_head, 0});
            G::load<1, false>(Q_i_smem[toc][1],  g.Q,   {batch_idx, ns_seq * 2 + 1, ns_head, 0});
            G::load<1, false>(dO_i_smem[toc][1], g.dOg, {batch_idx, ns_seq * 2 + 1, ns_head, 0});
        }

        // Process 4 dot-slices (compute overlaps with prefetch loads above)

        addr_precomputed = false;  // R87: reset cross-iter precompute flag at step start
        #pragma unroll
        for (int ds = 0; ds < STEP_QO / DOT_SLICE_QO; ds++) {
            const int q_slice_pos = q_pos_base + ds * DOT_SLICE_QO;

            // Early exit for fully masked dot-slices (causal):
            // When q_slice_pos + DOT_SLICE_QO <= k_pos, all P values
            // would be -inf → exp2(-inf)=0, making dV and dK contributions zero.
            if constexpr (causal) {
                if (q_slice_pos + DOT_SLICE_QO <= k_pos) {
                    addr_precomputed = false;  // R87: skipped iter invalidates precompute
                    continue;
                }
            }

            // ============================================================
            // Pre-compute LDS addresses
            // ============================================================
            {
              if (!addr_precomputed) {
                auto Qs = subtile_inplace<DOT_SLICE_QO, D_QK>(Q_i_smem[tic][ds/2], {ds%2, 0});
                Q_i_addr = get_address(Q_i, Qs);
                auto dOs = subtile_inplace<DOT_SLICE_QO, D_V>(dO_i_smem[tic][ds/2], {ds%2, 0});
                dO_i_addr = get_address(dO_i, dOs);
              }
              addr_precomputed = false;  // R87: consume precompute flag
              asm volatile(
                "v_mov_b32 v[38], %0\n"
                "v_mov_b32 v[40], %1\n"
                :: "v"(Q_i_addr), "v"(dO_i_addr));
            }

            // ============================================================
            // Load Q_i [16x192] row_l from shared (6 AGPR tiles)
            // Pipelined: issue all 6 ds_reads, then L/delta loads,
            // wait for Q_i (lgkmcnt(2)), write AGPRs, then wait for L/delta.
            // Temps: v[32:35], v[44:47], v[48:51], v[52:55], v[56:59], v[60:63]
            // Address in v[38].
            // ============================================================
            asm volatile(
                "ds_read_b128 v[32:35], v[38] offset:0\n"
                "ds_read_b128 v[44:47], v[38] offset:1024\n"
                "ds_read_b128 v[48:51], v[38] offset:2048\n"
                "ds_read_b128 v[52:55], v[38] offset:3072\n"
                "ds_read_b128 v[56:59], v[38] offset:4096\n"
                "ds_read_b128 v[60:63], v[38] offset:5120\n"
                ::: "memory");

            // Issue L/delta loads while Q_i reads are in flight
            load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[tic], ds));
            load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], ds));

            // ============================================================
            // Interleaved Q_i AGPR writes + P = Q @ K^T MFMAs
            // Write each Q_i tile (4 regs) then immediately issue the 2
            // P MFMAs that use that tile. The AGPR writes for the next
            // tile overlap with the MFMA pipeline execution.
            //
            // MFMAs reordered to k-first: for each k, issue both m=0,m=1
            // P[0]=v[30:33], P[1]=v[34:37]
            // Q[k]=a[176+k*4 : 179+k*4]
            // K[0][k]=a[96+k*4 : 99+k*4], K[1][k]=a[120+k*4 : 123+k*4]
            //
            // Then dO reads as pipeline drain after the last P MFMAs.
            // ============================================================
            asm volatile(
                // Wait for Q_i reads (2 outstanding = L + delta)
                "s_waitcnt lgkmcnt(2)\n"
                // --- k=0: write Q_i tile 0, nop for AGPR hazard, issue 2 MFMAs ---
                "v_accvgpr_write_b32 a[176], v[32]\n"
                "v_accvgpr_write_b32 a[177], v[33]\n"
                "v_accvgpr_write_b32 a[178], v[34]\n"
                "v_accvgpr_write_b32 a[179], v[35]\n"
                "s_nop 0\n"  // AGPR write-to-read hazard (1 cycle, sufficient on gfx950)
                "v_mfma_f32_16x16x32_bf16 v[30:33], a[176:179], a[96:99], 0\n"
                "v_mfma_f32_16x16x32_bf16 v[34:37], a[176:179], a[120:123], 0\n"
                // --- k=1: write Q_i tile 1 (overlaps k=0 MFMA), issue 2 MFMAs ---
                "v_accvgpr_write_b32 a[180], v[44]\n"
                "v_accvgpr_write_b32 a[181], v[45]\n"
                "v_accvgpr_write_b32 a[182], v[46]\n"
                "v_accvgpr_write_b32 a[183], v[47]\n"
                "s_nop 0\n"
                "v_mfma_f32_16x16x32_bf16 v[30:33], a[180:183], a[100:103], v[30:33]\n"
                "v_mfma_f32_16x16x32_bf16 v[34:37], a[180:183], a[124:127], v[34:37]\n"
                // --- k=2: write Q_i tile 2, issue 2 MFMAs ---
                "v_accvgpr_write_b32 a[184], v[48]\n"
                "v_accvgpr_write_b32 a[185], v[49]\n"
                "v_accvgpr_write_b32 a[186], v[50]\n"
                "v_accvgpr_write_b32 a[187], v[51]\n"
                "s_nop 0\n"
                "v_mfma_f32_16x16x32_bf16 v[30:33], a[184:187], a[104:107], v[30:33]\n"
                "v_mfma_f32_16x16x32_bf16 v[34:37], a[184:187], a[128:131], v[34:37]\n"
                // --- k=3: write Q_i tile 3, issue 2 MFMAs ---
                "v_accvgpr_write_b32 a[188], v[52]\n"
                "v_accvgpr_write_b32 a[189], v[53]\n"
                "v_accvgpr_write_b32 a[190], v[54]\n"
                "v_accvgpr_write_b32 a[191], v[55]\n"
                "s_nop 0\n"
                "v_mfma_f32_16x16x32_bf16 v[30:33], a[188:191], a[108:111], v[30:33]\n"
                "v_mfma_f32_16x16x32_bf16 v[34:37], a[188:191], a[132:135], v[34:37]\n"
                // --- k=4: write Q_i tile 4, issue 2 MFMAs ---
                "v_accvgpr_write_b32 a[192], v[56]\n"
                "v_accvgpr_write_b32 a[193], v[57]\n"
                "v_accvgpr_write_b32 a[194], v[58]\n"
                "v_accvgpr_write_b32 a[195], v[59]\n"
                "s_nop 0\n"
                "v_mfma_f32_16x16x32_bf16 v[30:33], a[192:195], a[112:115], v[30:33]\n"
                "v_mfma_f32_16x16x32_bf16 v[34:37], a[192:195], a[136:139], v[34:37]\n"
                // --- k=5: write Q_i tile 5, wait for L/delta, issue 2 MFMAs ---
                "v_accvgpr_write_b32 a[196], v[60]\n"
                "v_accvgpr_write_b32 a[197], v[61]\n"
                "v_accvgpr_write_b32 a[198], v[62]\n"
                "v_accvgpr_write_b32 a[199], v[63]\n"
                "s_waitcnt lgkmcnt(0)\n"  // wait for L and delta (also covers AGPR hazard)
                "v_mfma_f32_16x16x32_bf16 v[30:33], a[196:199], a[116:119], v[30:33]\n"
                "v_mfma_f32_16x16x32_bf16 v[34:37], a[196:199], a[140:143], v[34:37]\n"
                // Pipeline drain: issue dO_i ds_reads (4 reads overlap MFMA drain)
                "ds_read_b128 v[62:65], v[40] offset:0\n"
                "ds_read_b128 v[66:69], v[40] offset:1024\n"
                "ds_read_b128 v[70:73], v[40] offset:2048\n"
                "ds_read_b128 v[74:77], v[40] offset:3072\n"
                ::: "memory");

            // Scale, mask, exp2
            mul<0, 0>(P_ij, P_ij, P_SCALE_FACTOR);
            mul<0, 1>(P_ij, P_ij, P_SCALE_FACTOR);
            macros::mul::op<L_i, L_i>(L_SCALE_FACTOR);
            sub_row<0, 0, L_i>(P_ij, P_ij);
            sub_row<0, 1, L_i>(P_ij, P_ij);
            if constexpr (causal) {
                if (q_slice_pos + DOT_SLICE_QO <= k_pos) {
                    mov<neg_inf_v>(P_ij);
                } else if (q_slice_pos == k_pos) {
                    make_causal<0, 0, neg_inf_v>(P_ij, P_ij);
                    mov<0, 1, neg_inf_v>(P_ij);
                } else if (q_slice_pos == k_pos + DOT_SLICE_QO) {
                    make_causal<0, 1, neg_inf_v>(P_ij, P_ij);
                }
            }
            exp2<0, 0>(P_ij, P_ij);
            exp2<0, 1>(P_ij, P_ij);
            // R96c-B (k=4, snop=0): all 4 cvts hoisted into shadow.
            // swap_layout_inplace moved to AFTER dP MFMA block.

            // dO_i ds_reads (issued in P MFMA asm above) overlap with
            // P scale/exp2/copy VALU work (~60 cycles); ds_read latency
            // ~30-50 cycles, so they complete before dP MFMAs read them.
            // No explicit wait — reduces serialization.

            // ============================================================
            // dP = dO @ V^T : 8 MFMAs (raw asm with hardcoded V_j regs)
            //
            // dO_i: v[62:77] (4 tiles of 4 regs each)
            // V_j:  v[192:223] (8 tiles: V[0]=v[192:195]..V[7]=v[220:223])
            // dP_ij: v[38:45] (2 tiles: dP[0]=v[38:41], dP[1]=v[42:45])
            //
            // dP[0] = sum_k dO[k] @ V[k]^T, k=0..3
            // dP[1] = sum_k dO[k] @ V[k+4]^T, k=0..3
            //
            // Pipeline drain: 8 ds_read_b64_tr_b16 for dO_col to v[78:93]
            // ============================================================
            {
              auto dOcs2 = subtile_inplace<DOT_SLICE_QO, D_V>(dO_i_smem[tic][ds/2], {ds%2, 0});
              uint32_t addr_col = get_address(dO_i_col, dOcs2);
              asm volatile("v_mov_b32 v[126], %0" :: "v"(addr_col));
            }
            // Keep V_j registers alive: read v[192:223] into C++ variables
            // so the compiler tracks liveness from prologue loads through
            // the dP MFMAs below. Without this, the compiler inserts
            // spurious buffer_load_dwordx4 to reload V every dot-slice.
            uint32_t vk0, vk1, vk2, vk3, vk4, vk5, vk6, vk7;
            uint32_t vk8, vk9, vk10, vk11, vk12, vk13, vk14, vk15;
            uint32_t vk16, vk17, vk18, vk19, vk20, vk21, vk22, vk23;
            uint32_t vk24, vk25, vk26, vk27, vk28, vk29, vk30, vk31;
            asm volatile(""
                : "={v192}"(vk0),  "={v193}"(vk1),  "={v194}"(vk2),  "={v195}"(vk3),
                  "={v196}"(vk4),  "={v197}"(vk5),  "={v198}"(vk6),  "={v199}"(vk7),
                  "={v200}"(vk8),  "={v201}"(vk9),  "={v202}"(vk10), "={v203}"(vk11),
                  "={v204}"(vk12), "={v205}"(vk13), "={v206}"(vk14), "={v207}"(vk15),
                  "={v208}"(vk16), "={v209}"(vk17), "={v210}"(vk18), "={v211}"(vk19),
                  "={v212}"(vk20), "={v213}"(vk21), "={v214}"(vk22), "={v215}"(vk23),
                  "={v216}"(vk24), "={v217}"(vk25), "={v218}"(vk26), "={v219}"(vk27),
                  "={v220}"(vk28), "={v221}"(vk29), "={v222}"(vk30), "={v223}"(vk31)
                ::);
            asm volatile(
                // dP[0] = dO @ V[0:3]^T
                "v_mfma_f32_16x16x32_bf16 v[38:41], v[62:65], v[192:195], 0\n"
                // R96c-A: redundant cvt in shadow cvt #1 (v[46] <- v[30], v[31])
                "v_cvt_pk_bf16_f32 v[46], v[30], v[31]\n"
                "v_mfma_f32_16x16x32_bf16 v[38:41], v[66:69], v[196:199], v[38:41]\n"
                // R96c-A: redundant cvt in shadow cvt #2 (v[47] <- v[32], v[33])
                "v_cvt_pk_bf16_f32 v[47], v[32], v[33]\n"
                "v_mfma_f32_16x16x32_bf16 v[38:41], v[70:73], v[200:203], v[38:41]\n"
                // R96c-A: redundant cvt in shadow cvt #3 (v[48] <- v[34], v[35])
                "v_cvt_pk_bf16_f32 v[48], v[34], v[35]\n"
                "v_mfma_f32_16x16x32_bf16 v[38:41], v[74:77], v[204:207], v[38:41]\n"
                // R96c-A: redundant cvt in shadow cvt #4 (v[49] <- v[36], v[37])
                "v_cvt_pk_bf16_f32 v[49], v[36], v[37]\n"
                // dP[1] = dO @ V[4:7]^T
                "v_mfma_f32_16x16x32_bf16 v[42:45], v[62:65], v[208:211], 0\n"
                "v_mfma_f32_16x16x32_bf16 v[42:45], v[66:69], v[212:215], v[42:45]\n"
                "v_mfma_f32_16x16x32_bf16 v[42:45], v[70:73], v[216:219], v[42:45]\n"
                "v_mfma_f32_16x16x32_bf16 v[42:45], v[74:77], v[220:223], v[42:45]\n"
                // Pipeline drain: 8 ds_read_b64_tr_b16 for dO_col to v[78:93]
                "ds_read_b64_tr_b16 v[78:79], v[126] offset:0\n"
                "ds_read_b64_tr_b16 v[80:81], v[126] offset:256\n"
                "ds_read_b64_tr_b16 v[82:83], v[126] offset:1024\n"
                "ds_read_b64_tr_b16 v[84:85], v[126] offset:1280\n"
                "ds_read_b64_tr_b16 v[86:87], v[126] offset:2048\n"
                "ds_read_b64_tr_b16 v[88:89], v[126] offset:2304\n"
                "ds_read_b64_tr_b16 v[90:91], v[126] offset:3072\n"
                "ds_read_b64_tr_b16 v[92:93], v[126] offset:3328\n"
                :
                : "{v192}"(vk0),  "{v193}"(vk1),  "{v194}"(vk2),  "{v195}"(vk3),
                  "{v196}"(vk4),  "{v197}"(vk5),  "{v198}"(vk6),  "{v199}"(vk7),
                  "{v200}"(vk8),  "{v201}"(vk9),  "{v202}"(vk10), "{v203}"(vk11),
                  "{v204}"(vk12), "{v205}"(vk13), "{v206}"(vk14), "{v207}"(vk15),
                  "{v208}"(vk16), "{v209}"(vk17), "{v210}"(vk18), "{v211}"(vk19),
                  "{v212}"(vk20), "{v213}"(vk21), "{v214}"(vk22), "{v215}"(vk23),
                  "{v216}"(vk24), "{v217}"(vk25), "{v218}"(vk26), "{v219}"(vk27),
                  "{v220}"(vk28), "{v221}"(vk29), "{v222}"(vk30), "{v223}"(vk31)
                : "memory");

            // R96c-B: swap_layout_inplace moved here (after dP MFMA block)
            swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);

            // dS = P * (dP - delta)
            sub_row<0, 0, delta_i>(dP_ij, dP_ij);
            sub_row<0, 1, delta_i>(dP_ij, dP_ij);
            mul(dP_ij, dP_ij, P_ij);
            copy(dP_ij_bf16, dP_ij);

            // No dS store to shared — dQ removed

            // R82C Option A: defer swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16)
            // until AFTER dV MFMAs. The 4 v_permlane16_swap instructions then run
            // in the dV MFMA shadow (~128 cy) instead of in the LDS-wait shadow.
            // dV MFMAs do NOT read v[50:53] (only v[46:49]=P_bf16_col), so this
            // is RAW-safe. dP_bf16_col is only consumed by dK MFMAs, which now
            // come after the deferred swap.

            // dO_i_col reads (8 in-flight) + Q_i_col reads (12, issued below)
            // = 20 momentarily in-flight. The 15-cap means ds_read_b64_tr_b16
            // FIFO ordering is unreliable above 15. Wait for dO_i_col to drain
            // BEFORE issuing more Q_i_col.
            asm volatile("s_waitcnt lgkmcnt(0)");

            // ============================================================
            // Issue Q_i_col loads, then dV and dK MFMAs as raw asm to
            // prevent compiler from inserting spurious NOPs between phases.
            //
            // Q_col: v[94:117] (no conflict with dV operands)
            // dO_col: v[78:93] (A operand for dV MFMAs)
            // P_bf16_col: v[46:49] (B operand for dV MFMAs)
            // dP_bf16_col: v[50:53] (B operand for dK MFMAs)
            // dV: v[128:191] (16-reg tiles)
            // dK: a[0:95] (16-reg tiles)
            // ============================================================
            {
              auto Qcs = subtile_inplace<DOT_SLICE_QO, D_QK>(Q_i_smem[tic][ds/2], {ds%2, 0});
              Q_i_col_addr = get_address(Q_i_col, Qcs);
              load<0, 0>(Q_i_col, Qcs, Q_i_col_addr); load<0, 1>(Q_i_col, Qcs, Q_i_col_addr);
              load<0, 2>(Q_i_col, Qcs, Q_i_col_addr); load<0, 3>(Q_i_col, Qcs, Q_i_col_addr);
              load<0, 4>(Q_i_col, Qcs, Q_i_col_addr); load<0, 5>(Q_i_col, Qcs, Q_i_col_addr);
            }

            // R82C Option A: split the dV+dK monolithic asm into two blocks so
            // the compiler can schedule swap_layout_inplace(dP_ij_bf16_col,
            // dP_ij_bf16) IN the dV MFMA shadow. Each v_permlane16_swap is a
            // pure VALU op; placing 4 of them in the ~128cy dV MFMA window is
            // exactly the "VALU-MFMA coexec" lift R65A identified.
            asm volatile(
                // dV += dO_col^T @ P_bf16_col: 4 MFMAs (~128 cycles issue)
                "v_mfma_f32_32x32x16_bf16 v[128:143], v[78:81], v[46:49], v[128:143]\n"
                "v_mfma_f32_32x32x16_bf16 v[144:159], v[82:85], v[46:49], v[144:159]\n"
                "v_mfma_f32_32x32x16_bf16 v[160:175], v[86:89], v[46:49], v[160:175]\n"
                "v_mfma_f32_32x32x16_bf16 v[176:191], v[90:93], v[46:49], v[176:191]\n"
                ::: "memory");

            // VALU work hand-placed in dV MFMA shadow:
            // 4x v_permlane16_swap on dP_bf16 (v[50:53]) to produce dP_bf16_col.
            // RAW-safe vs dV chain (dV writes v[128:191], reads v[78:93]+v[46:49]).
            swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);

            asm volatile(
                // Q_col reads (12 ds_read_b64_tr_b16) were issued before the
                // dV MFMAs; they complete during the ~128-cycle dV chain.
                // No waitcnt needed before dK MFMAs.
                // dK += Q_col^T @ dP_bf16_col: 6 MFMAs
                "v_mfma_f32_32x32x16_bf16 a[0:15], v[94:97], v[50:53], a[0:15]\n"
                "v_mfma_f32_32x32x16_bf16 a[16:31], v[98:101], v[50:53], a[16:31]\n"
                "v_mfma_f32_32x32x16_bf16 a[32:47], v[102:105], v[50:53], a[32:47]\n"
                "v_mfma_f32_32x32x16_bf16 a[48:63], v[106:109], v[50:53], a[48:63]\n"
                "v_mfma_f32_32x32x16_bf16 a[64:79], v[110:113], v[50:53], a[64:79]\n"
                "v_mfma_f32_32x32x16_bf16 a[80:95], v[114:117], v[50:53], a[80:95]\n"
                ::: "memory");

            // R87: hoist next-iter Q_i_addr/dO_i_addr compute INTO dK MFMA
            // shadow (~192cy of 6 dK MFMAs). With #pragma unroll on the ds
            // loop, the compiler resolves the addr_precomputed flag via
            // predicated select (v_cndmask) instead of exec-mask divergence.
            if (ds + 1 < STEP_QO / DOT_SLICE_QO) {
              int next_ds = ds + 1;
              auto Qs_n = subtile_inplace<DOT_SLICE_QO, D_QK>(Q_i_smem[tic][next_ds/2], {next_ds%2, 0});
              Q_i_addr = get_address(Q_i, Qs_n);
              auto dOs_n = subtile_inplace<DOT_SLICE_QO, D_V>(dO_i_smem[tic][next_ds/2], {next_ds%2, 0});
              dO_i_addr = get_address(dO_i, dOs_n);
              addr_precomputed = true;
            }

        } // ds

        // Wait for software-pipelined prefetch loads to complete, then barrier.
        // The loads were issued at the top of this step, so their latency
        // overlapped with the 4 dot-slices of compute.
        if (step + 1 < num_steps) {
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_s_barrier();
            tic = toc;  // swap: next step reads from toc (now tic)
        }
    } // step

    // Restore v[61] = dP_SCALE_FACTOR (clobbered by pipelined Q/K loads)
    {
        float dp_scale = dP_SCALE_FACTOR;
        asm volatile("v_mov_b32 v[61], %0\n" :: "v"(dp_scale));
    }

    // =====================================================================
    // Epilogue: Store dV, then dK (in two chunks via AGPR->VGPR copy)
    // =====================================================================
    // Store dV (already in VGPRs v[128:191], no scale needed)
    store<1>(g.dVg, dV_j, {batch_idx, 0, kv_head_idx, 0}, {0, j, 0, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    // --- dK store: copy AGPRs to VGPRs, scale, store ---
    // dV store + s_waitcnt(0) + s_barrier above provide ample pipeline drain
    // for last MFMA writes to AGPRs (no extra s_nop 15 needed).
    // Chunk 2 first: a[64:95] -> v[128:159], last 64 cols of dK
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        (macros::v_accvgpr_read_b32<128 + Is, 320 + Is>(), ...);
    }(std::make_index_sequence<32>{});
    // Scale dK chunk 2 by dP_SCALE_FACTOR (v[61])
    asm volatile(
        "v_mul_f32 v[128], v[61], v[128]\n" "v_mul_f32 v[129], v[61], v[129]\n"
        "v_mul_f32 v[130], v[61], v[130]\n" "v_mul_f32 v[131], v[61], v[131]\n"
        "v_mul_f32 v[132], v[61], v[132]\n" "v_mul_f32 v[133], v[61], v[133]\n"
        "v_mul_f32 v[134], v[61], v[134]\n" "v_mul_f32 v[135], v[61], v[135]\n"
        "v_mul_f32 v[136], v[61], v[136]\n" "v_mul_f32 v[137], v[61], v[137]\n"
        "v_mul_f32 v[138], v[61], v[138]\n" "v_mul_f32 v[139], v[61], v[139]\n"
        "v_mul_f32 v[140], v[61], v[140]\n" "v_mul_f32 v[141], v[61], v[141]\n"
        "v_mul_f32 v[142], v[61], v[142]\n" "v_mul_f32 v[143], v[61], v[143]\n"
        "v_mul_f32 v[144], v[61], v[144]\n" "v_mul_f32 v[145], v[61], v[145]\n"
        "v_mul_f32 v[146], v[61], v[146]\n" "v_mul_f32 v[147], v[61], v[147]\n"
        "v_mul_f32 v[148], v[61], v[148]\n" "v_mul_f32 v[149], v[61], v[149]\n"
        "v_mul_f32 v[150], v[61], v[150]\n" "v_mul_f32 v[151], v[61], v[151]\n"
        "v_mul_f32 v[152], v[61], v[152]\n" "v_mul_f32 v[153], v[61], v[153]\n"
        "v_mul_f32 v[154], v[61], v[154]\n" "v_mul_f32 v[155], v[61], v[155]\n"
        "v_mul_f32 v[156], v[61], v[156]\n" "v_mul_f32 v[157], v[61], v[157]\n"
        "v_mul_f32 v[158], v[61], v[158]\n" "v_mul_f32 v[159], v[61], v[159]\n"
        :::);
    // Store chunk 2 at column 128: warp_idx col=2 -> 2*64=128 elements
    store<1>(g.dKg, dK_chunk2, {batch_idx, 0, kv_head_idx, 0}, {0, j, 0, 2});
    __builtin_amdgcn_s_waitcnt(0);

    // Chunk 1: a[0:63] -> v[128:191], first 128 cols of dK
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        (macros::v_accvgpr_read_b32<128 + Is, 256 + Is>(), ...);
    }(std::make_index_sequence<64>{});
    // Scale dK chunk 1 by dP_SCALE_FACTOR (v[61])
    asm volatile(
        "v_mul_f32 v[128], v[61], v[128]\n" "v_mul_f32 v[129], v[61], v[129]\n"
        "v_mul_f32 v[130], v[61], v[130]\n" "v_mul_f32 v[131], v[61], v[131]\n"
        "v_mul_f32 v[132], v[61], v[132]\n" "v_mul_f32 v[133], v[61], v[133]\n"
        "v_mul_f32 v[134], v[61], v[134]\n" "v_mul_f32 v[135], v[61], v[135]\n"
        "v_mul_f32 v[136], v[61], v[136]\n" "v_mul_f32 v[137], v[61], v[137]\n"
        "v_mul_f32 v[138], v[61], v[138]\n" "v_mul_f32 v[139], v[61], v[139]\n"
        "v_mul_f32 v[140], v[61], v[140]\n" "v_mul_f32 v[141], v[61], v[141]\n"
        "v_mul_f32 v[142], v[61], v[142]\n" "v_mul_f32 v[143], v[61], v[143]\n"
        "v_mul_f32 v[144], v[61], v[144]\n" "v_mul_f32 v[145], v[61], v[145]\n"
        "v_mul_f32 v[146], v[61], v[146]\n" "v_mul_f32 v[147], v[61], v[147]\n"
        "v_mul_f32 v[148], v[61], v[148]\n" "v_mul_f32 v[149], v[61], v[149]\n"
        "v_mul_f32 v[150], v[61], v[150]\n" "v_mul_f32 v[151], v[61], v[151]\n"
        "v_mul_f32 v[152], v[61], v[152]\n" "v_mul_f32 v[153], v[61], v[153]\n"
        "v_mul_f32 v[154], v[61], v[154]\n" "v_mul_f32 v[155], v[61], v[155]\n"
        "v_mul_f32 v[156], v[61], v[156]\n" "v_mul_f32 v[157], v[61], v[157]\n"
        "v_mul_f32 v[158], v[61], v[158]\n" "v_mul_f32 v[159], v[61], v[159]\n"
        "v_mul_f32 v[160], v[61], v[160]\n" "v_mul_f32 v[161], v[61], v[161]\n"
        "v_mul_f32 v[162], v[61], v[162]\n" "v_mul_f32 v[163], v[61], v[163]\n"
        "v_mul_f32 v[164], v[61], v[164]\n" "v_mul_f32 v[165], v[61], v[165]\n"
        "v_mul_f32 v[166], v[61], v[166]\n" "v_mul_f32 v[167], v[61], v[167]\n"
        "v_mul_f32 v[168], v[61], v[168]\n" "v_mul_f32 v[169], v[61], v[169]\n"
        "v_mul_f32 v[170], v[61], v[170]\n" "v_mul_f32 v[171], v[61], v[171]\n"
        "v_mul_f32 v[172], v[61], v[172]\n" "v_mul_f32 v[173], v[61], v[173]\n"
        "v_mul_f32 v[174], v[61], v[174]\n" "v_mul_f32 v[175], v[61], v[175]\n"
        "v_mul_f32 v[176], v[61], v[176]\n" "v_mul_f32 v[177], v[61], v[177]\n"
        "v_mul_f32 v[178], v[61], v[178]\n" "v_mul_f32 v[179], v[61], v[179]\n"
        "v_mul_f32 v[180], v[61], v[180]\n" "v_mul_f32 v[181], v[61], v[181]\n"
        "v_mul_f32 v[182], v[61], v[182]\n" "v_mul_f32 v[183], v[61], v[183]\n"
        "v_mul_f32 v[184], v[61], v[184]\n" "v_mul_f32 v[185], v[61], v[185]\n"
        "v_mul_f32 v[186], v[61], v[186]\n" "v_mul_f32 v[187], v[61], v[187]\n"
        "v_mul_f32 v[188], v[61], v[188]\n" "v_mul_f32 v[189], v[61], v[189]\n"
        "v_mul_f32 v[190], v[61], v[190]\n" "v_mul_f32 v[191], v[61], v[191]\n"
        :::);
    store<1>(g.dKg, dV_j, {batch_idx, 0, kv_head_idx, 0}, {0, j, 0, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------
void dispatch_bwd_d192v128(attn_bwd_d192v128_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute(
        (void*)attend_bwd_d192v128_ker,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_bwd_d192v128_ker<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

// ---------------------------------------------------------------------------
// Python bindings (dK+dV only — no dQg)
// ---------------------------------------------------------------------------
PYBIND11_MODULE(tk_kernel_bkwd, m) {
    m.doc() = "ART-based backward kernel D_QK=192 D_V=128 (dK+dV only)";
    py::bind_function<dispatch_bwd_d192v128>(m, "dispatch_bwd_combined",
        &attn_bwd_d192v128_globals::Q,
        &attn_bwd_d192v128_globals::K,
        &attn_bwd_d192v128_globals::V,
        &attn_bwd_d192v128_globals::dOg,
        &attn_bwd_d192v128_globals::dKg,
        &attn_bwd_d192v128_globals::dVg,
        &attn_bwd_d192v128_globals::L_vec,
        &attn_bwd_d192v128_globals::delta_vec
    );
}
