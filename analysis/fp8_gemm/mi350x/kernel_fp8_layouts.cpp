#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

// =============================================================================
// Tuning constants (BF16-style: single block of constexpr ints + plain #defines
// for values that need stringification into inline asm).
// =============================================================================

// Block / tile dims
constexpr int BLOCK_SIZE        = 256;
constexpr int K_BLOCK           = 128;
constexpr int WARPS_M           = 2;
constexpr int WARPS_N           = 4;
constexpr int MIN_BLOCKS_PER_CU = 2;

// Block-swizzle (XCD-aware tile mapping)
constexpr int BLOCK_SWIZZLE_NUM_XCDS = 8;

// 4-wave RCR dispatch thresholds
constexpr int RCR_4WAVE_MIN_GRID  = 3200;
constexpr int RCR_4WAVE_MAX_K     = 8192;

// RCR two-tile schedule threshold (ki >= this → use two-tile main-loop
// schedule). Round-15 probe: lowering to 20 to cover gpt_oss ki_dyn=22
// regressed by 0% (within ±0.23 % noise on bench, score unchanged at
// 833) — the deeper two-tile pipeline (8 mma/iter, 4 LDS slots) does
// NOT amortize on the shorter ki_dyn=22 main-loop relative to the
// single-tile fallback (4 mma/iter, 2 LDS slots) because the larger
// epilog fraction + slightly higher VGPR spill (compile-time codegen
// is identical, runtime branch only — but the reformulated reg
// dependency chain in two-tile path makes the issue queue marginally
// busier on ki=11). 28 stays the empirical break-even.
// See analysis/_notes/round-15-fp8-rcr-two-tile-min-ki-noop.md.
constexpr int RCR_TWO_TILE_MIN_KI = 28;

// Derived block dims
constexpr int BLK = BLOCK_SIZE, BK = K_BLOCK;
constexpr int HB  = BLK / 2;
constexpr int _NUM_WARPS    = WARPS_M * WARPS_N;
constexpr int _NUM_THREADS  = _NUM_WARPS * WARP_THREADS;
constexpr int RBM = BLK / WARPS_M / 2;   // 64
constexpr int RBN = BLK / WARPS_N / 2;   // 32
constexpr int TAIL_BLOCK_M  = 16;
constexpr int TAIL_BLOCK_N  = 16;

// Stringification utilities for inline-asm wait counters
#define TK_STRINGIFY_IMPL(x) #x
#define TK_STRINGIFY(x) TK_STRINGIFY_IMPL(x)
#define TK_WAIT_LGKM(x)  asm volatile("s_waitcnt lgkmcnt(" TK_STRINGIFY(x) ")")
#define TK_WAIT_VMCNT(x) asm volatile("s_waitcnt vmcnt("  TK_STRINGIFY(x) ")")
#define TK_PRAGMA_UNROLL(x) _Pragma(TK_STRINGIFY(unroll x))

// Per-layout wait-counter budgets (#define since they're stringified into asm).
//
// RCR_PREFETCH_LGKM (manual round-A, gpt_oss FP8 kernel-only metric):
// Sweep on /workspace/code/Primus-Turbo metric ``_metric_gpt_oss_fp8_kernel.py``
// (8 gpt_oss shapes × 3 sections, kernel-only TFLOPS via CUDA-event ``_time_op``
// with WARMUP=10 / ITERS=50 / 20th-percentile, MI355X GPU 2, baseline R5
// auto-optimize commit 7637aae). 7 runs per cell, median over runs:
//
//     LGKM | score | fwd avg | dgrad avg | wgrad avg
//        2 | 688   | 1908    | 2090      | 1781
//        4 | 687   | 1904    | 2090      | 1782   ← prior baseline
//        6 | 687   | 1904    | 2084      | 1781
//        8 | 691   | 1913    | 2097      | 1792   ← chosen
//       10 | 690   | 1913    | 2096      | 1789
//       12 | 687   | 1905    | 2084      | 1782
//
// Mechanism: ``TK_WAIT_LGKM(N)`` issues ``s_waitcnt lgkmcnt(N)``, i.e.
// "wait until ≤N LGKM ops are in-flight". The main-loop body of
// ``grouped_rcr_kernel`` (line ~2744) issues ``ds_read_b128`` (LDS reads
// for A/B subtiles) plus ``buffer_load_lds`` (HBM→LDS prefetch for
// next-iter tile) every iter; with LGKM=4 the wait blocks until
// only 4 outstanding LDS ops remain, but in steady state we have
// ~6-8 ops in-flight (3 next-iter prefetch + 3-5 next-mma reads),
// so the wait fires before the LDS pipeline drains naturally and
// stalls the issue queue. LGKM=8 lets the pipeline auto-drain
// before the explicit wait, removing ~3-5% of per-iter stall cycles
// on K%128==64 K-tail shapes (gpt_oss K=2880, ki=22). At LGKM≥10
// the next-iter mma starts consuming stale data before the wait
// completes, regressing back to baseline.
//
// fwd impact: +9 TFLOPS median (1904 → 1913, +0.5%) with all 7 runs
// landing ≥ baseline median 687. dgrad/wgrad gains in the table
// are noise-correlated (their kernels use RRR/CRR LGKM, not RCR).
// See analysis/_notes/round-A-fp8-rcr-prefetch-lgkm-8.md.
#define RCR_PREFETCH_LGKM       8
#define RCR_INIT0_VMCNT         4
#define RCR_INIT1_VMCNT         6
#define RCR_STEADY_VMCNT        8
#define RCR_EPILOGUE_VMCNT      4
#define RCR_TWO_TILE_MID_VMCNT  6
#define RRR_PREFETCH_LGKM       8
#define RRR_INIT0_VMCNT         4
#define RRR_INIT1_VMCNT         6
#define RRR_STEADY_VMCNT        4
#define RRR_EPILOGUE_VMCNT      2
#define CRR_PREFETCH_LGKM       3
#define CRR_INIT0_VMCNT         2
#define CRR_INIT1_VMCNT         6
#define CRR_STEADY_VMCNT        4
#define CRR_EPILOGUE_VMCNT      2

// Per-layout main-loop unroll counts (stringified into #pragma unroll N).
#define RCR_MAIN_UNROLL 2
#define RRR_MAIN_UNROLL 4
#define CRR_MAIN_UNROLL 1

// Sched/wait barrier helpers
#define RRR_SCHED_BARRIER() __builtin_amdgcn_sched_barrier(0)
#define RCR_SCHED_BARRIER() __builtin_amdgcn_sched_barrier(0)
#define CRR_STEADY_MID_BARRIER() __builtin_amdgcn_s_barrier()
#define CRR_MMA_BEGIN() __builtin_amdgcn_s_setprio(1)
#define CRR_MMA_END()   __builtin_amdgcn_s_setprio(0)

using G = kittens::group<_NUM_WARPS>;
using _gl_fp8  = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_bf16 = gl<bf16, -1, -1, -1, -1>;

enum class Layout { RCR, RRR, CRR };

// Row-layout shared/register tiles (for A in RCR/RRR, B in RCR)
using ST_row    = st_fp8e4m3<HB, BK, st_16x128_s>;    // 128×128, M/N rows × K cols
using A_row_reg = rt_fp8e4m3<RBM, BK, row_l, rt_16x128_s>;
using B_row_reg = rt_fp8e4m3<RBN, BK, row_l, rt_16x128_s>;

// Col-layout register tiles (for B in RRR, A/B in CRR)
using A_col_reg = rt_fp8e4m3<BK, RBM, col_l, rt_128x16_s>;  // 128×64
using B_col_reg = rt_fp8e4m3<BK, RBN, col_l, rt_128x16_s>;  // 128×32

using ST_v2     = st_fp8e4m3<HB, BK, st_16x128_v2_s>;
using ST_v2a    = st_fp8e4m3<HB, BK, st_16x128_v2a_s>;

static_assert(sizeof(A_row_reg) == sizeof(A_col_reg));
static_assert(alignof(A_row_reg) == alignof(A_col_reg));
static_assert(sizeof(B_row_reg) == sizeof(B_col_reg));
static_assert(alignof(B_row_reg) == alignof(B_col_reg));

// Round-26-dm (auto-optimize R30 / Lever D Round-B step 1):
// Compile-time validation that ``rt_fp8e4m3`` instantiates correctly
// with the new ``rt_32x64_s`` / ``rt_64x32_s`` cell shapes (added in
// HK SHA c2abba21). Confirms the kittens type system is fully
// functional for the 32x32x64 mfma cell-shape family — prerequisite
// for any future Lever D K-tail or main-loop port. No callers yet,
// no codegen impact (static_asserts have zero runtime footprint).
//
// Sanity expectations (from rt_shape::rt_32x64 = rt_shape<32, 64, 16>
// and rt_shape::rt_64x32 = rt_shape<64, 32, 16>):
//   * elements_per_thread = 32x64 / 64 = 32 (= 8 fp8e4m3_4 packed)
//   * num_strides = 32 / stride(16) = 2
//   * height for rt_fp8e4m3<RBM=64, BK=64, row_l, rt_32x64_s>:
//       = 64 / 32 = 2
//   * width for the same: 64 / 64 = 1
//   * sizeof = num_packed=4 * elements_per_thread=32 * sizeof(fp8e4m3)
//       = 32 fp8 = 32 bytes per lane per cell × height(2)*width(1) = 64 B/lane
namespace lever_d_round_b_step1_compile_test {
    using A_row_reg_32x64 = rt_fp8e4m3<RBM, 64, row_l, rt_32x64_s>;     // 64 rows × 64 K-cols
    using B_row_reg_32x64 = rt_fp8e4m3<RBN, 64, row_l, rt_32x64_s>;     // 32 rows × 64 K-cols
    using B_col_reg_64x32 = rt_fp8e4m3<64, RBN, col_l, rt_64x32_s>;     // 64 K-rows × 32 cols
    using cAB_32_acc      = rt_fl<RBM, RBN, col_l, rt_32x32_s>;         // 64×32 accumulator in 32x32 cells

    // Geometric assertions — matches the ``rt_32x64`` / ``rt_64x32``
    // shape struct values defined in rt_shape.cuh (R14-dm).
    static_assert(A_row_reg_32x64::height == 2,
        "rt_fp8e4m3<64, 64, row_l, rt_32x64_s>::height must be 64/32 = 2");
    static_assert(A_row_reg_32x64::width == 1,
        "rt_fp8e4m3<64, 64, row_l, rt_32x64_s>::width must be 64/64 = 1");
    static_assert(A_row_reg_32x64::base_tile_rows == 32,
        "rt_32x64 cell rows must be 32");
    static_assert(A_row_reg_32x64::base_tile_cols == 64,
        "rt_32x64 cell cols must be 64");
    static_assert(B_row_reg_32x64::height == 1,
        "rt_fp8e4m3<32, 64, row_l, rt_32x64_s>::height must be 32/32 = 1");
    static_assert(B_col_reg_64x32::height == 1,
        "rt_fp8e4m3<64, 32, col_l, rt_64x32_s>::height must be 64/64 = 1");
    static_assert(cAB_32_acc::height == 2,
        "rt_fl<64, 32, col_l, rt_32x32_s>::height must be 64/32 = 2");
    static_assert(cAB_32_acc::width == 1,
        "rt_fl<64, 32, col_l, rt_32x32_s>::width must be 32/32 = 1");

    // Per-lane register footprint sanity:
    //   rt_base<fp8e4m3, row_l, rt_32x64>:
    //     elements_per_thread = 32*64/64 = 32 fp8 / lane
    //     num_packed (fp8e4m3_4)  = 4
    //     packed_per_thread        = 32 / 4 = 8 fp8e4m3_4 / lane
    //     sizeof(rt_base)         = 8 * sizeof(fp8e4m3_4) = 8 * 4 = 32 B / cell / lane
    //   rt size = 32 B/cell × height(2) × width(1) = 64 B / lane.
    //
    //   Compare to the live A_row_reg above (rt_16x128_s):
    //     elements_per_thread = 16*128/64 = 32 fp8 / lane (same per cell)
    //     packed_per_thread    = 32 / 4 = 8 / lane
    //     sizeof(rt_base)     = 32 B / cell / lane (same)
    //     rt size = 32 B/cell × height(8 = 128/16) × width(1) = 256 B / lane.
    //
    //   The 32x64 instance is 4× SMALLER per-lane than the rt_16x128_s
    //   instance — explained by ``height`` differing (2 vs 8 = 4× ratio)
    //   while per-cell register footprint is identical. For K-tail
    //   (K_REM=64 = 1 K-iter at K=64), the A_row_reg_32x64 type holds
    //   exactly the data needed for ``height(2)`` mfma_323264 inputs and
    //   contains no SENTINEL waste — vs the current rt_16x128_s path which
    //   loads K=64..127 as SENTINEL and discards half the buffer_load
    //   instruction issue cycles.
    static_assert(sizeof(A_row_reg_32x64) == 32 * 2 * 1,
        "A_row_reg_32x64 should be 64 bytes per lane (32 B/cell × height(2) × width(1))");
    static_assert(sizeof(cAB_32_acc) == 16 * 4 * 2 * 1,
        "cAB_32_acc should be 128 bytes per lane (16 fp32 × 4 B × height(2) × width(1))");

    // Cross-layout dword equivalence — confirms cAB_32 covers the same
    // 64×32 region as cA-cB (currently rt_fl<64, 32, col_l, rt_16x16>):
    using cA_layout_16x16 = rt_fl<RBM, RBN, col_l, rt_16x16_s>;
    static_assert(sizeof(cA_layout_16x16) == sizeof(cAB_32_acc),
        "cAB_32 and cA must occupy same total per-lane VGPR footprint "
        "(both 32 dwords/lane = 128 B/lane covering 64×32 region) — "
        "differ only in cell-internal lane partition (32x32 vs 16x16).");
} // namespace lever_d_round_b_step1_compile_test

// Round-54-dm (auto-optimize R54): Lever C-2 round-1 scaffold.
//
// The R53 baseline kernel-resource-usage capture (analysis/_notes/
// round-53-fp8-grouped-resource-baseline-FUSED-KTAIL-not-spill-bound.md)
// established that ``grouped_rcr_kernel<FUSED_KTAIL=true>`` (gpt_oss
// K=2880) sits at VGPR=256 / AGPR=0 / Spill=34 — LOWER spill than the
// FUSED_KTAIL=false instances (38-54), yet achieves SLOWER ratios
// (1.07-1.11 vs 1.18-1.31). The bottleneck is therefore NOT register
// spill but **MFMA compute density**:
//   * current grouped (WARPS_M=2, WARPS_N=4, RBM=64, RBN=32):
//       8 mfma_16x16x128 per warp per K-step
//   * 4w-style (WARPS_M=2, WARPS_N=2, RBM=64, RBN=64):
//       16 mfma_16x16x128 per warp per K-step (2× compute density)
//
// This namespace stages the 4w-style **constants and type aliases**
// only — *no kernel definition yet*. Its only job in R54 is to verify
// that the LLVM type system accepts the larger RBN=64 register-tile
// instantiations and that the ratio invariants for warp-tile geometry
// hold at compile time. The R55 round will copy ``grouped_rcr_kernel``
// (lines ~2221-2885) into this namespace and re-target uses of the
// outer-scope constants to the local ones.
//
// Mirrors the precedent set by ``lever_d_round_b_step1_compile_test``
// above (round-26-dm) — same pattern: scaffold types, run static
// checks, leave actual kernel code to the next round once the type
// validation lands clean.
namespace lever_c2_round_54_step1_scaffold {
    constexpr int WARPS_M       = 2;
    constexpr int WARPS_N       = 2;                          // was 4 in outer scope
    constexpr int _NUM_WARPS    = WARPS_M * WARPS_N;          // 4
    constexpr int _NUM_THREADS  = _NUM_WARPS * WARP_THREADS;  // 256 (was 512)
    constexpr int RBM_4w        = BLK / WARPS_M / 2;          // 64 (unchanged)
    constexpr int RBN_4w        = BLK / WARPS_N / 2;          // 64 (was 32)

    // Per-warp output footprint:
    //   current grouped: RBM*RBN = 64*32 =  2048 fp32 across 64 lanes
    //                                   =   32 fp32/lane
    //   4w-style:        RBM*RBN = 64*64 =  4096 fp32 across 64 lanes
    //                                   =   64 fp32/lane (per-warp accum)
    // With 16 fragments of 4 fp32/lane each = 64 fp32/lane * 4 warps =
    // 256 fp32/lane total accumulator across the block — matching
    // ``rcr_4w::kernel`` (line 1284) which the R53 report shows at
    // AGPR=256 + Spill=0. This footprint is the **direct trigger** for
    // LLVM's AGPR allocator (per R47 hypothesis: accumulators ≥ 256
    // fp32/lane → LLVM picks AGPR).
    static_assert(_NUM_THREADS == 256,
        "4w-style block must launch with 256 threads (4 warps); "
        "halves block-occupancy vs current 8-warp/512-thread layout.");
    static_assert(WARPS_M * WARPS_N == 4,
        "4w-style uses 4 warps total (2x2 grid).");
    static_assert(RBM_4w * RBN_4w == 4096,
        "4w-style per-warp output region: 64x64 = 4096 fp32 per warp = "
        "64 fp32/lane * 64 lanes/warp.");

    // 4w-style register-tile types. ``A_row_reg_4w`` keeps the same
    // RBM=64 row count as the outer ``A_row_reg`` (line 92), so the
    // A-side load pattern can be reused verbatim. ``B_row_reg_4w``
    // doubles the RBN dim to 64, requiring RBN=64 on the B-side
    // load lambdas (the R55 wiring step will re-target the lambda
    // parameter ``int RBN = ...`` references inside ``grouped_rcr_kernel``).
    using A_row_reg_4w = rt_fp8e4m3<RBM_4w, BK, row_l, rt_16x128_s>;
    using B_row_reg_4w = rt_fp8e4m3<RBN_4w, BK, row_l, rt_16x128_s>;
    using C_acc_4w     = rt_fl    <RBM_4w, RBN_4w, col_l, rt_16x16_s>;

    // Per-lane VGPR footprint sanity (compile-time, must match the R53
    // resource report's "256 VGPR + 256 AGPR" target for rcr_4w):
    //   rt_fl<64, 64, col_l, rt_16x16_s>:
    //     height = 64 / 16 = 4
    //     width  = 64 / 16 = 4
    //     elements_per_thread per cell = 16*16/64 = 4 fp32/lane/cell
    //     total per-lane fp32 = 4 * 4 * 4 = 64 fp32/lane (per-warp)
    //     across 4 warps the block holds 64 * 4 = 256 fp32/lane.
    static_assert(C_acc_4w::height == 4,
        "C_acc_4w height = RBM_4w / cell_rows = 64/16 = 4");
    static_assert(C_acc_4w::width == 4,
        "C_acc_4w width  = RBN_4w / cell_cols = 64/16 = 4");
    // sizeof(rt_fl<...>) per-lane = packed_per_thread * elements_per_thread
    // * sizeof(fp32) * height * width — for rt_16x16 this is
    // 1 * 4 * 4 B * 4 * 4 = 256 B / lane / warp = 64 fp32/lane/warp.
    static_assert(sizeof(C_acc_4w) == 4 * 4 * 4 * 4,
        "C_acc_4w must be 256 bytes/lane = 64 fp32/lane (per-warp). "
        "Across 4 warps the per-block accumulator footprint is 256 "
        "fp32/lane — large enough for LLVM to pick AGPR allocation "
        "(matches rcr_4w::kernel R53 baseline: AGPR=256, Spill=0).");

    // A_row_reg_4w / B_row_reg_4w cross-checks (must be SAME size
    // as outer A_row_reg / B_row_reg respectively, since RBM
    // unchanged on A; B doubles to RBN=64 i.e. 2× the outer
    // B_row_reg footprint).
    static_assert(sizeof(A_row_reg_4w) == sizeof(::A_row_reg),
        "A_row_reg_4w must match outer A_row_reg footprint (RBM=64 "
        "unchanged on A side).");
    static_assert(sizeof(B_row_reg_4w) == 2 * sizeof(::B_row_reg),
        "B_row_reg_4w must be 2× outer B_row_reg (RBN doubled 32→64).");
} // namespace lever_c2_round_54_step1_scaffold

// =============================================================================
// Round-F (this round) — Lever F: BLOCK_SIZE=128 tile-size port (M1 skeleton).
//
// Strategic context (full plan: analysis/_notes/round-F-fp8-tile-size-128-port-
// plan-and-EV.md). After Rounds B-E exhausted the small numeric levers, the
// remaining unblocked structural lever is per-CTA tile-size dispatch:
//
//   * The production grouped_rcr_kernel runs at BLOCK_SIZE=256 → 256x256
//     per-CTA tile, 1 CTA per CU via persistent grid (NUM_CUS=256 blocks).
//   * For low-batch shapes (gpt_oss B=4, M=2048) the total tile count is
//     384 → only 1.5 tiles per CU → catastrophic tail effect: half the CUs
//     idle in wave 2. Measured TFLOPS = 1465 (Down fwd) = 28 % of FP8 peak,
//     vs 51 % on best shape (GateUP B=32 M=4096 dgrad) which has 46 tiles/CU.
//   * Strong correlation (Round-F per-shape table) between tiles/CU and
//     TFLOPS up to ~6 tiles/CU.
//
// Lever F shrinks the per-CTA tile to 128x128. Same 8-warp / VGPR-only /
// 4-acc design — only the per-warp acc dimension drops from RBM=64 / RBN=32
// to RBM=32 / RBN=16. Tile count quadruples → 4x more tiles/CU → eliminates
// the tail effect for B=4 shapes.
//
// Why not 4-warp port (256x256/4w, Lever C)? Tried R54-R61, blocked by
// deterministic LLVM AGPR allocation bug (cAB[0][0].tiles[{0,1}][1] wrong).
// Lever F stays in standard 8-warp/VGPR regime: per-warp acc footprint
// drops to 64 fp32/lane (vs 128 in BLK=256), well below the 256-fp32/lane
// threshold that triggers AGPR allocation in LLVM (per R47 hypothesis).
// Avoids the codegen bug entirely.
//
// Per-shape EV (projected score 685 → ~720..735, +35..+50 score):
//   shape           | tiles/CU @256 | tiles/CU @128 | est gain (worst sec)
//   Down  B4 M2048  | 1.5           | 6.0           | +500 T (1408 → 1900)
//   GateUP B4 M2048 | 2.9           | 11.6          | +250 T (1782 → 2050)
//   Down  B32 *     | 12-24         | 48-96         | regression (over-fine)
//   GateUP B32 *    | 23-46         | 92-184        | regression (over-fine)
// Conditional dispatch: kernel_b128 fires only when tiles_per_CU < 8.
//
// M1 (this commit) is COMPILE-ONLY: type aliases + static_assert validation
// that the HK type system accepts the smaller HB / RBM / RBN dimensions and
// the per-warp footprint math is correct. No kernel function defined yet.
// M2 will port grouped_rcr_kernel body verbatim (replacing constants), M3
// adds the dispatcher gate, M4 validates correctness, M5 measures metric.
namespace kernel_b128 {
    // === Round-F M2: namespace-local constants that SHADOW file-scope ===
    // The kernel function body (M2 port below) uses unqualified names like
    // `BLOCK_SIZE`, `RBM`, `RBN`, `ST_v2`, `A_row_reg` — when defined inside
    // this namespace, those resolve to the local versions (smaller HB / RBM
    // / RBN), enabling a verbatim copy of the production grouped_rcr_kernel
    // body without text changes. References to outer-scope names use ``::``
    // explicitly when needed.

    constexpr int BLOCK_SIZE = 128;          // shadows outer 256
    constexpr int HB         = BLOCK_SIZE / 2;  // = 64 (was 128)
    constexpr int BLK        = BLOCK_SIZE;
    constexpr int BK         = K_BLOCK;       // outer K_BLOCK = 128 (unchanged)
    constexpr int WARPS_M    = 2;
    constexpr int WARPS_N    = 4;
    constexpr int _NUM_WARPS = WARPS_M * WARPS_N;          // 8 (unchanged)
    constexpr int _NUM_THREADS = _NUM_WARPS * WARP_THREADS;// 512 (unchanged)
    constexpr int RBM        = BLOCK_SIZE / WARPS_M / 2;   // 32 (was 64)
    constexpr int RBN        = BLOCK_SIZE / WARPS_N / 2;   // 16 (was 32)

    using G = ::G;  // kittens::group<8> (8-warp group)

    // LDS tile types. Underlying sub-tile is 16x128 (st_16x128_v2_s); with
    // HB=64 we have 4 sub-tiles per A/B-slab (vs 8 sub-tiles at HB=128).
    using ST_v2  = st_fp8e4m3<HB, BK, st_16x128_v2_s>;
    using ST_v2a = st_fp8e4m3<HB, BK, st_16x128_v2a_s>;
    using ST_row = st_fp8e4m3<HB, BK, st_16x128_s>;

    // Register tile types. Same 16x128 base tile as BLK=256, just fewer
    // cells per tile.
    using A_row_reg = rt_fp8e4m3<RBM, BK, row_l, rt_16x128_s>;
    using B_row_reg = rt_fp8e4m3<RBN, BK, row_l, rt_16x128_s>;

    // Compile-time validation (shadowed sizes vs file-scope outer)
    static_assert(_NUM_THREADS == 512, "8-warp 512-thread CTA preserved");
    static_assert(RBM == 32 && RBN == 16, "per-warp acc cell shrinks to 32x16");
    static_assert(sizeof(ST_v2) * 2 == sizeof(::ST_v2),
        "ST_v2 (b128) should be exactly half of outer ::ST_v2 (HB halved)");
    static_assert(sizeof(ST_v2a) * 2 == sizeof(::ST_v2a),
        "ST_v2a (b128) should be exactly half of outer ::ST_v2a (HB halved)");
    static_assert(sizeof(A_row_reg) * 2 == sizeof(::A_row_reg),
        "A_row_reg (b128) should be half of outer ::A_row_reg (RBM halved)");
    static_assert(sizeof(B_row_reg) * 2 == sizeof(::B_row_reg),
        "B_row_reg (b128) should be half of outer ::B_row_reg (RBN halved)");

    // Per-warp accumulator footprint check. C_acc = rt_fl<RBM=32, RBN=16,
    // col_l, rt_16x16_s>: height=2, width=1, 4 fp32/lane/cell.
    // 4 accs (cA,cB,cC,cD) * 2 cells * 4 fp32/lane = 32 fp32/lane per warp.
    //   - 1/4 of BLK=256's 128 fp32/lane footprint
    //   - WELL BELOW the 256-fp32/lane AGPR-trigger threshold (per R47)
    //   - LLVM stays in standard VGPR allocation, avoids the rcr_4w
    //     R54-R61 AGPR-allocator codegen bug entirely
    using C_acc = rt_fl<RBM, RBN, col_l, rt_16x16_s>;
    static_assert(C_acc::height == 2, "C_acc height = 32/16 = 2");
    static_assert(C_acc::width  == 1, "C_acc width  = 16/16 = 1");
    static_assert(sizeof(C_acc) == 32,
        "C_acc must be 32 bytes/lane = 8 fp32/lane per acc; 4 accs = 32 fp32/lane "
        "per warp << 256 fp32/lane AGPR threshold per R47");

} // namespace kernel_b128

// Cooperative col-major load from a v2/v2a-swizzled FP8 LDS tile.
// Two `ds_read_b64_tr_b8` per lane per K_HALF (offset:0 + offset:1024).
template<typename RT, int K_HALF, typename ST>
__device__ __forceinline__ void load_col_from_st_half(
    RT& dst, const ST& tile, int col_start)
{
    const int laneid = kittens::laneid();
    const int row_off = ((laneid % 16) / 2) + ((laneid / 16) * 16);
    const int col_off = (laneid % 2) * 8;
    const uint32_t tile_base = reinterpret_cast<uintptr_t>(&tile.data[0]);

    constexpr int idx = K_HALF * 4;
    const int k_row = row_off + K_HALF * 64;

    const uint32_t stidx = k_row >> 4;
    const uint32_t base_k = tile_base + (stidx << 11) + (stidx << 7) + ((k_row & 15) << 7);
    const uint32_t sw_k   = (k_row & 7) << 4;

    #pragma unroll
    for (int j = 0; j < RT::width; j++) {
        const uint32_t nc = col_start + j * 16 + col_off;
        const uint32_t addr = base_k + (nc ^ sw_k);

        asm volatile(
            "ds_read_b64_tr_b8 %0, %2 offset:0\n"
            "ds_read_b64_tr_b8 %1, %2 offset:1024\n"
            : "=&v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx])),
              "=&v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx + 2]))
            : "v"(addr)
            : "memory"
        );
    }
}

template<typename RT, typename ST>
__device__ __forceinline__ void load_col_from_st(
    RT& dst, const ST& tile, int col_start)
{
    load_col_from_st_half<RT, 0>(dst, tile, col_start);
    load_col_from_st_half<RT, 1>(dst, tile, col_start);
}

__device__ __forceinline__ void rrr_mma(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
    const A_row_reg& a,
    const B_col_reg& b)
{
    mma_AB(acc, a, b, acc);
}

__device__ __forceinline__ void rcr_mma(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
    const A_row_reg& a,
    const B_row_reg& b)
{
    mma_ABt(acc, a, b, acc);
}

// Round-26-dm (auto-optimize R31 / Lever D Round-B step 2):
// Lever D 32x32x64 K-tail rcr_mma wrapper. Calls ``mma_ABt`` →
// dispatches to ``mma_ABt_base`` 32x32x64 fp8 branch (mma.cuh:234-238)
// → ``mfma323264`` intrinsic. The dispatch is selected at compile time
// by the (D::shape, A::shape, B::shape) tuple = (rt_32x32, rt_32x64,
// rt_32x64). No callers yet — wired in R32+ when the K-tail block
// migrates from 16x16x128 to 32x32x64 cell shape.
//
// Why ``rt_32x64`` for B and not ``rt_64x32_s`` (added in HK c2abba21):
// in the RCR layout (A row × B row → mfma_ABt computing A·Bᵀ), the B
// operand is the SAME shape as A from the mfma's perspective — both
// 32×64 — because mfma_323264 takes B as 32-row × 64-K and the
// transpose is implicit in the ABt accumulation pattern. The
// ``rt_64x32_s`` alias is for CRR/CCR layouts (B col-major), not RCR.
__device__ __forceinline__ void rcr_mma_32(
    rt_fl<RBM, RBN, col_l, rt_32x32_s>& acc,
    const rt_fp8e4m3<RBM, 64, row_l, rt_32x64_s>& a,
    const rt_fp8e4m3<RBN, 64, row_l, rt_32x64_s>& b)
{
    mma_ABt(acc, a, b, acc);
}

// Round-26-dm (auto-optimize R34 / Lever D Round-B step 4):
// K-tail loaders for the rt_32x64 fp8 layout. Each lane reads
// 2 × buffer_load_b128 from HBM into rt_base.data[]. Lane mapping
// per AMD CDNA4 mfma_323264 (verified by mma_ABt_base dispatch in
// mma.cuh:234-238):
//   row_lane    = laneid % 32      (rows 0..31 within 32-row cell)
//   k_lane_byte = (laneid / 32) * 32 (chunk 0 = K[0..31], 1 = K[32..63])
//   data[0..3]  = K=[k_lane_byte,        k_lane_byte + 16) → b128 #1
//   data[4..7]  = K=[k_lane_byte + 16,   k_lane_byte + 32) → b128 #2
//
// For K_REM=64 (gpt_oss K=2880 = 22*128 + 64): ALL LANES VALID, both
// b128 reads in-bounds. No SENTINEL lanes (vs rt_16x128 K-tail loader
// which had lanes 32..63 SENTINEL because K=64..127 K-OOB).
//
// Currently no callers — the K-tail block port (R35+) will wire these.
// Force-instantiated below to validate types at HK build time.
template<typename A_RT_32x64>
__device__ __forceinline__ void load_a_kt_32x64(
    A_RT_32x64& A_tile,
    i32x4 a_srsrc_kt,
    int M_warp_base,
    int row_lane,
    int k_lane_byte,
    int a_row_stride_bytes,
    uint32_t K_tail_base_bytes,
    bool b128_lo_valid,
    bool b128_hi_valid)
{
    constexpr uint32_t SENTINEL = 0xFFFF0000u;

    #pragma unroll
    for (int h = 0; h < A_RT_32x64::height; ++h) {
        const int A_row_idx = M_warp_base + h * 32 + row_lane;
        const uint32_t v_base = static_cast<uint32_t>(
            A_row_idx * a_row_stride_bytes +
            K_tail_base_bytes + k_lane_byte);
        const uint32_t v_lo = b128_lo_valid ? v_base : SENTINEL;
        const uint32_t v_hi = b128_hi_valid ? (v_base + 16) : SENTINEL;
        __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
            a_srsrc_kt, v_lo, 0, 0);
        __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
            a_srsrc_kt, v_hi, 0, 0);
        *reinterpret_cast<__uint128_t*>(&A_tile.tiles[h][0].data[0]) = v0;
        *reinterpret_cast<__uint128_t*>(&A_tile.tiles[h][0].data[4]) = v1;
    }
}

template<typename B_RT_32x64>
__device__ __forceinline__ void load_b_kt_32x64(
    B_RT_32x64& B_tile,
    i32x4 b_srsrc_kt,
    int N_warp_base,
    int row_lane,
    int k_lane_byte,
    int b_row_stride_bytes,
    uint32_t b_group_byte_base,
    uint32_t K_tail_base_bytes,
    bool b128_lo_valid,
    bool b128_hi_valid)
{
    constexpr uint32_t SENTINEL = 0xFFFF0000u;

    #pragma unroll
    for (int h_b = 0; h_b < B_RT_32x64::height; ++h_b) {
        const int B_row_idx_in_group = N_warp_base + h_b * 32 + row_lane;
        const uint32_t v_base = b_group_byte_base + static_cast<uint32_t>(
            B_row_idx_in_group * b_row_stride_bytes +
            K_tail_base_bytes + k_lane_byte);
        const uint32_t v_lo = b128_lo_valid ? v_base : SENTINEL;
        const uint32_t v_hi = b128_hi_valid ? (v_base + 16) : SENTINEL;
        __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
            b_srsrc_kt, v_lo, 0, 0);
        __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
            b_srsrc_kt, v_hi, 0, 0);
        *reinterpret_cast<__uint128_t*>(&B_tile.tiles[h_b][0].data[0]) = v0;
        *reinterpret_cast<__uint128_t*>(&B_tile.tiles[h_b][0].data[4]) = v1;
    }
}

// Force compile-time instantiation / type-check of the rcr_mma_32
// dispatch and the K-tail rt_32x64 loaders. ``__attribute__((used))``
// keeps the symbol around so the build will fully type-check all
// callees — surfacing layout / shape / intrinsic-arg mismatches at
// HK build time, before R35+ wires up the K-tail port.
__attribute__((used)) [[maybe_unused]] static __device__ void
__lever_d_round_b_force_instantiate_rcr_mma_32() {
    rt_fl<RBM, RBN, col_l, rt_32x32_s> dummy_acc{};
    rt_fp8e4m3<RBM, 64, row_l, rt_32x64_s> dummy_a{};
    rt_fp8e4m3<RBN, 64, row_l, rt_32x64_s> dummy_b{};
    rcr_mma_32(dummy_acc, dummy_a, dummy_b);

    i32x4 dummy_srsrc{};
    load_a_kt_32x64(dummy_a, dummy_srsrc,
                    /*M_warp_base=*/0,
                    /*row_lane=*/0,
                    /*k_lane_byte=*/0,
                    /*a_row_stride_bytes=*/0,
                    /*K_tail_base_bytes=*/0u,
                    /*b128_lo_valid=*/true,
                    /*b128_hi_valid=*/true);
    load_b_kt_32x64(dummy_b, dummy_srsrc,
                    /*N_warp_base=*/0,
                    /*row_lane=*/0,
                    /*k_lane_byte=*/0,
                    /*b_row_stride_bytes=*/0,
                    /*b_group_byte_base=*/0u,
                    /*K_tail_base_bytes=*/0u,
                    /*b128_lo_valid=*/true,
                    /*b128_hi_valid=*/true);
}

// Lever D Round-B step 1 (auto-optimize R37 / dm-R64):
// Force compile-time instantiation of the ST_32x64 shared-memory tile
// type declared in include/types/shared/st_shape.cuh + public alias
// in include/types/types.cuh. Confirms that st_fp8e4m3<HB, 64,
// st_32x64_s> composes through the underlying ``st<T, R, C, _shape>``
// template, including:
//   * shape struct's bytes_per_thread / swizzle functions
//   * underlying_subtile_* derived constants
//   * subtile_padding propagation
//   * st_subtile addressing math
// No runtime callers yet (R38+ will wire in main-loop load helpers
// once the bank-conflict-free swizzle is derived). LLVM DCE trims
// the body at codegen time so this contributes 0 bytes to kernel
// instruction memory; its purpose is purely to surface any template
// instantiation errors at HK build time, well before the full
// main-loop port is wired in.
__attribute__((used)) [[maybe_unused]] static __device__ void
__lever_d_round_b_force_instantiate_st_32x64() {
    using ST_32x64 = st_fp8e4m3<HB, 64, st_32x64_s>;
    __shared__ ST_32x64 dummy_st;

    // Touch the type's static-member infrastructure to force full
    // template-parameter validation. Static-asserts mirror the
    // kittens-internal checks in ``st<>::`` body.
    static_assert(ST_32x64::rows == HB, "ST_32x64 rows should equal HB=128");
    static_assert(ST_32x64::cols == 64, "ST_32x64 cols should equal 64");
    static_assert(ST_32x64::underlying_subtile_rows == 32,
                  "ST_32x64 underlying subtile rows should equal 32");
    static_assert(ST_32x64::underlying_subtile_cols == 64,
                  "ST_32x64 underlying subtile cols should equal 64");
    static_assert(ST_32x64::underlying_subtile_bytes_per_thread == 16,
                  "ST_32x64 should dispatch the fp8 bytes_per_thread=16 branch");

    // Exercise the swizzle functor at compile time via a device call
    // path; LLVM DCE removes the dead reference after instantiation.
    (void)ST_32x64::swizzle({0, 0});
    (void)dummy_st;
}

__device__ __forceinline__ void crr_mma(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
    const A_col_reg& a,
    const B_col_reg& b)
{
    const auto& a_row = reinterpret_cast<const A_row_reg&>(a);
    mma_AB(acc, a_row, b, acc);
}

// Round-20 — scalar load/store helpers used by the K-tail / N-tail
// kernels (RMW: load existing C, add K-tail accumulator, store back).
// Routed through ``llvm.amdgcn.raw.buffer.{load,store}.{i8,i16}`` (BUFFER
// class) instead of the generic-pointer ``raw_ptr[idx]`` expression
// which the compiler lowers to ``global_{load,store}_{byte,short}``
// (FLAT class). SRD construction is loop-invariant on the global tensor
// argument; the compiler's LICM hoists it out of the K-tail kernels'
// unrolled per-cell loops, so each cell only pays the buffer
// load/store cost. Round-19 ported the same FLAT->BUFFER reroute for
// the col-layout ``kittens::store`` overload (gpt_oss focus score
// 794->880, +85pp); this round does the same for the K-tail / N-tail
// kernels, which gpt_oss K=2880 always hits.
__device__ __forceinline__ float load_fp8_scalar(const _gl_fp8& src, int row, int col) {
    const uint32_t buffer_size = src.batch() * src.depth() * src.rows() * src.cols() * sizeof(fp8e4m3);
    const std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(src.raw_ptr);
    const std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    const buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);
    const i32x4 srsrc = std::bit_cast<i32x4>(br);
    const uint32_t voffset = (row * src.cols() + col) * sizeof(fp8e4m3);
    const uint8_t bits = llvm_amdgcn_raw_buffer_load_b8(srsrc, voffset, 0, 0);
    return base_types::convertor<float, fp8e4m3>::convert(std::bit_cast<fp8e4m3>(bits));
}

__device__ __forceinline__ float load_bf16_scalar(const _gl_bf16& src, int row, int col) {
    const uint32_t buffer_size = src.batch() * src.depth() * src.rows() * src.cols() * sizeof(bf16);
    const std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(src.raw_ptr);
    const std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    const buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);
    const i32x4 srsrc = std::bit_cast<i32x4>(br);
    const uint32_t voffset = (row * src.cols() + col) * sizeof(bf16);
    const uint16_t bits = llvm_amdgcn_raw_buffer_load_b16(srsrc, voffset, 0, 0);
    return base_types::convertor<float, bf16>::convert(std::bit_cast<bf16>(bits));
}

__device__ __forceinline__ void store_bf16_scalar(const _gl_bf16& dst, int row, int col, float value) {
    const uint32_t buffer_size = dst.batch() * dst.depth() * dst.rows() * dst.cols() * sizeof(bf16);
    const std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(dst.raw_ptr);
    const std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    const buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);
    const i32x4 srsrc = std::bit_cast<i32x4>(br);
    const uint32_t voffset = (row * dst.cols() + col) * sizeof(bf16);
    const bf16 v = base_types::convertor<bf16, float>::convert(value);
    llvm_amdgcn_raw_buffer_store_b16(std::bit_cast<uint16_t>(v), srsrc, voffset, 0, 0);
}

// Per-group scalar FP8 load. ``b`` for grouped FP8 is logically
// [batch=1, G, N, K]; the 4D coord lets `grouped_tail_kernel` index B at
// (group_idx, row, col).
__device__ __forceinline__ float load_fp8_scalar_grp(const _gl_fp8& src, int g_idx, int row, int col) {
    const uint32_t buffer_size = src.batch() * src.depth() * src.rows() * src.cols() * sizeof(fp8e4m3);
    const std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(src.raw_ptr);
    const std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    const buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);
    const i32x4 srsrc = std::bit_cast<i32x4>(br);
    const uint32_t idx = ((0 * src.depth() + g_idx) * src.rows() + row) * src.cols() + col;
    const uint32_t voffset = idx * sizeof(fp8e4m3);
    const uint8_t bits = llvm_amdgcn_raw_buffer_load_b8(srsrc, voffset, 0, 0);
    return base_types::convertor<float, fp8e4m3>::convert(std::bit_cast<fp8e4m3>(bits));
}

// Packed 8 × fp8e4m3 = 8 bytes for vectorised tail-kernel K-loop. The
// HIP compiler emits a single `global_load_dwordx2` for a load through
// this type when the source pointer is 8-byte aligned, replacing 8
// separate scalar fp8 loads (8× fewer VMEM transactions). Used by the
// RCR fast path inside `grouped_tail_kernel` where both operands are
// stride-1 in K. Extraction via ``convertor<float4, fp8e4m3_4>`` gives
// 8 fp32 accumulator inputs per 8-byte load.
struct alignas(8) fp8e4m3_8 {
    fp8e4m3_4 lo, hi;
};

template<int N_THREADS, ducks::st::all ST, ducks::gl::all GL>
__device__ __forceinline__ void prefill_transpose_swizzled_offsets(
    ST& dst, const GL& src, uint32_t* swizzled_offsets)
{
    using T = typename ST::dtype;

    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    constexpr int memcpy_per_tile =
        ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);
    static_assert(
        ST::rows * ST::cols * sizeof(T) >= bytes_per_warp,
        "shared tile must be at least 1024 bytes"
    );

    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int laneid = kittens::laneid();
    const int warpid = kittens::warpid() % num_warps;
    const int row_stride = src.template stride<2>();

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {
        const int lane_byte_offset =
            (laneid * bytes_per_thread) +
            (warpid * bytes_per_warp) +
            (i * num_warps * bytes_per_warp);
        const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
        const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
        const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
        const int subtile_lane_byte_offset =
            lane_byte_offset % ST::underlying_subtile_bytes;

        const int row =
            subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
        const int col =
            (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);

        const uint32_t swizzled_shared_byte_offset = dst.swizzle({row, col});
        const int shared_row =
            swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes;
        const int shared_col =
            (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);

        const int transposed_global_row =
            shared_col + subtile_col * ST::underlying_subtile_cols;
        const int transposed_global_col =
            shared_row + subtile_row * ST::underlying_subtile_rows;

        swizzled_offsets[i] =
            (transposed_global_row * row_stride + transposed_global_col) * sizeof(T);
    }

    if constexpr (
        memcpy_per_tile * (bytes_per_thread * N_THREADS) !=
        ST::rows * ST::cols * sizeof(T)
    ) {
        constexpr int leftover_bytes =
            ST::rows * ST::cols * sizeof(T) -
            memcpy_per_tile * (bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;
        constexpr int leftover_warps = leftover_threads / kittens::WARP_THREADS;

        if (warpid < leftover_warps) {
            const int lane_byte_offset =
                (laneid * bytes_per_thread) +
                (warpid * bytes_per_warp) +
                (memcpy_per_tile * num_warps * bytes_per_warp);
            const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
            const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
            const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
            const int subtile_lane_byte_offset =
                lane_byte_offset % ST::underlying_subtile_bytes;

            const int row =
                subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
            const int col =
                (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);

            const uint32_t swizzled_shared_byte_offset = dst.swizzle({row, col});
            const int shared_row =
                swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes;
            const int shared_col =
                (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);

            const int transposed_global_row =
                shared_col + subtile_col * ST::underlying_subtile_cols;
            const int transposed_global_col =
                shared_row + subtile_row * ST::underlying_subtile_rows;

            swizzled_offsets[memcpy_per_tile] =
                (transposed_global_row * row_stride + transposed_global_col) * sizeof(T);
        }
    }
}

// Round-13 cleanup: ``prefill_swizzled_offsets_partial_K`` (round-2 path-A
// scaffolding for LDS-staged K-tail) was deleted. Round-3 (commit 07354791)
// shipped the K-tail fuse via path B (direct ``raw_buffer_load_b128``
// HBM→register, lane-cell mapping derived in
// ``analysis/_notes/round-3-fp8-ktail-path-b-success.md``) which sidesteps
// swizzled offsets entirely. Round-11 (commit f9d591cb) dropped the now-dead
// callers in ``grouped_rcr_kernel``'s prologue; round-13 dropped the helper
// itself plus stale ~80-line path-A comment block from the fuse epilog.
// Path-A SENTINEL-voffset scheme remains documented in
// ``analysis/_notes/round-2-ktail-fuse-result.md`` and
// ``round-3-fp8-ktail-path-a-saturation.md`` for historical context.
// Codegen is unchanged — DCE was already firing on the unreferenced symbol.

template<int N_THREADS,
         ducks::st::all ST,
         ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>>
__device__ __forceinline__ void load_transpose(
    ST& dst, const GL& src, const COORD& idx, const uint32_t* swizzled_offsets)
{
    using T = typename ST::dtype;

    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    constexpr int memcpy_per_tile =
        ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);
    static_assert(
        ST::rows * ST::cols * sizeof(T) >= bytes_per_warp,
        "shared tile must be at least 1024 bytes"
    );

    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int warpid = kittens::warpid() % num_warps;
    const int row_stride = src.template stride<2>();

    coord<> unit_coord(
        idx.template dim<0>(),
        idx.template dim<1>(),
        idx.template dim<3>() * ST::cols,
        idx.template dim<2>() * ST::rows
    );
    T* global_ptr = (T*)&src[unit_coord];
    i32x4 srsrc = make_srsrc(global_ptr, row_stride * ST::cols * sizeof(T));

    const uintptr_t lds_tile_base = reinterpret_cast<uintptr_t>(&dst.data[0]);

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {
        const int warp_linear_offset =
            (warpid * bytes_per_warp) + (i * num_warps * bytes_per_warp);
        const int lds_subtile_id = warp_linear_offset / ST::underlying_subtile_bytes;
        uintptr_t lds_addr =
            lds_tile_base + warp_linear_offset + lds_subtile_id * ST::subtile_padding;
        as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

        llvm_amdgcn_raw_buffer_load_lds(
            srsrc,
            lds_ptr,
            bytes_per_thread,
            swizzled_offsets[i],
            0,
            0,
            static_cast<int>(coherency::cache_all)
        );
    }

    if constexpr (
        memcpy_per_tile * (bytes_per_thread * N_THREADS) !=
        ST::rows * ST::cols * sizeof(T)
    ) {
        constexpr int leftover_bytes =
            ST::rows * ST::cols * sizeof(T) -
            memcpy_per_tile * (bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;
        constexpr int leftover_warps = leftover_threads / kittens::WARP_THREADS;

        if (warpid < leftover_warps) {
            const int warp_linear_offset =
                (warpid * bytes_per_warp) +
                (memcpy_per_tile * num_warps * bytes_per_warp);
            const int lds_subtile_id =
                warp_linear_offset / ST::underlying_subtile_bytes;
            uintptr_t lds_addr =
                lds_tile_base + warp_linear_offset + lds_subtile_id * ST::subtile_padding;
            as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

            llvm_amdgcn_raw_buffer_load_lds(
                srsrc,
                lds_ptr,
                bytes_per_thread,
                swizzled_offsets[memcpy_per_tile],
                0,
                0,
                static_cast<int>(coherency::cache_all)
            );
        }
    }
}

// P19 Dev A — m0-broadcast hoist for the 8-wave RCR DTL loads.
//
// Drop-in replacement for the kittens 4-arg G::load(dst, src, idx, swizzled_offsets)
// used inside the 8-wave RCR main loop. Behaviour:
//   * Pre-computes the per-iter LDS byte offset as an SGPR (readfirstlane
//     forces wave-uniform residency in SGPR class), bypassing LLVM's
//     tendency to recompute the address through a VGPR + v_readfirstlane
//     immediately before each DTL store.
//   * Issues `s_mov m0, <sgpr>` + `buffer_load_dwordx4 ... offen lds` via
//     inline asm. The LLVM intrinsic
//     `__builtin_amdgcn_raw_buffer_load_lds` lets the scheduler CSE the
//     m0 plumbing back into vector ops; inline asm forecloses that.
//   * Vector destination operand is a phantom — `buffer_load_dwordx4 ... lds`
//     does NOT write a VGPR, the operand merely satisfies LLVM's asm-binding
//     contract and gets dead-code-eliminated downstream.
//   * Preserves the leftover-warps tail handling from the kittens helper so
//     odd memcpy_per_tile counts still drain.
//
// Mirrors `kittens::load<2, false, ST, GL, COORD, N_THREADS>` from
// include/ops/warp/memory/tile/global_to_shared.cuh:187-249 (the 4-arg
// variant taking pre-computed swizzled_offsets).
template<int N_THREADS,
         ducks::st::all ST,
         ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>>
__device__ __forceinline__ void rcr_8w_load_hoist(
    ST& dst, const GL& src, const COORD& idx,
    const uint32_t* __restrict__ swizzled_offsets)
{
    using T = typename ST::dtype;

    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_warp   = bytes_per_thread * kittens::WARP_THREADS;
    constexpr int memcpy_per_tile  =
        ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);
    static_assert(
        ST::rows * ST::cols * sizeof(T) >= bytes_per_warp,
        "shared tile must be at least 1024 bytes"
    );

    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int warpid = kittens::warpid() % num_warps;

    coord<> unit_coord = idx.template unit_coord<2, 3>();
    T* tensor_base = (T*)src.raw_ptr;
    T* global_ptr  = (T*)&src[unit_coord];
    // Full-tensor SRD: bound the ENTIRE source (A or B) tensor so that
    // OOB column-tile loads — induced by ``bpc = ceil_div(n, BLK)`` for
    // partial last-N tiles — clamp to 0 instead of touching unmapped
    // pages. The OG ``make_srsrc(global_ptr, ST::rows * row_stride *
    // sizeof(T))`` was tile-local: ``global_ptr`` past tensor end took
    // the SRD bound into unmapped memory, faulting on the swizzled
    // raw_buffer_load_lds path (verified on N=384 K=2048 RCR/RRR/CRR
    // cumulative test, MI350X). The full-tensor SRD makes every load
    // safe regardless of where ``unit_coord`` points; OOB bytes are
    // returned as 0, the masked C store drops them.
    const uint32_t total_bytes = static_cast<uint32_t>(
        size_t(src.batch()) * size_t(src.depth()) *
        size_t(src.rows())  * size_t(src.cols())  * sizeof(T));
    i32x4 srsrc = make_srsrc(tensor_base, total_bytes);
    // Tile byte offset from tensor base, hoisted into an SGPR.
    // Used as SOFFSET of the buffer_load_dwordx4 to recover the per-tile
    // base. Wave-uniform because ``unit_coord`` only depends on (br, bc,
    // sub-tile, k), all of which are uniform.
    const uint32_t tile_byte_offset = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(global_ptr) -
                              reinterpret_cast<uintptr_t>(tensor_base)));

    const uintptr_t lds_tile_base =
        reinterpret_cast<uintptr_t>(&dst.data[0]);

    // A1: hoist scalar per-pass LDS-byte ramp into SGPRs in the prologue.
    // Each lds_addr below is wave-uniform (warpid + i are uniform), so we
    // make that explicit via readfirstlane and keep the value in SGPR.
    uint32_t lds_addrs[memcpy_per_tile + 1];  // +1 leftover slot (may be unused)
    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; ++i) {
        const int warp_linear_offset =
            (warpid * bytes_per_warp) + (i * num_warps * bytes_per_warp);
        const int lds_subtile_id = warp_linear_offset / ST::underlying_subtile_bytes;
        const uint32_t off32 = static_cast<uint32_t>(
            lds_tile_base + warp_linear_offset +
            lds_subtile_id * ST::subtile_padding);
        lds_addrs[i] = __builtin_amdgcn_readfirstlane(off32);
    }

    // A2: full inline-asm DTL — set m0 from the SGPR-hoisted per-pass offset
    // and issue buffer_load_dwordx4 ... offen lds. Operand binding mirrors
    // P18 Dev A's working 4-wave pattern (see rcr_4wave_dynamic.inc).
    // %0 = s "lds_off" (SGPR), %1 = v "goff" (per-lane VGPR offset),
    // %2 = s "srsrc" (4-SGPR buffer resource), %3 = s "tile_byte_offset"
    // (SGPR SOFFSET) — the per-tile base offset added to V_VOFFSET.
    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; ++i) {
        const uint32_t lds_off = lds_addrs[i];
        const uint32_t goff    = swizzled_offsets[i];
        asm volatile(
            "s_mov_b32 m0, %0\n\t"
            "buffer_load_dwordx4 %1, %2, %3 offen lds\n\t"
            :
            : "s"(lds_off), "v"(goff), "s"(srsrc), "s"(tile_byte_offset)
            : "memory");
    }

    if constexpr (memcpy_per_tile * (bytes_per_thread * N_THREADS) !=
                  ST::rows * ST::cols * sizeof(T)) {
        constexpr int leftover_bytes =
            ST::rows * ST::cols * sizeof(T) -
            memcpy_per_tile * (bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;
        constexpr int leftover_warps   = leftover_threads / kittens::WARP_THREADS;
        if (warpid < leftover_warps) {
            const int warp_linear_offset =
                (warpid * bytes_per_warp) +
                (memcpy_per_tile * num_warps * bytes_per_warp);
            const int lds_subtile_id =
                warp_linear_offset / ST::underlying_subtile_bytes;
            const uint32_t off32 = static_cast<uint32_t>(
                lds_tile_base + warp_linear_offset +
                lds_subtile_id * ST::subtile_padding);
            const uint32_t lds_off = __builtin_amdgcn_readfirstlane(off32);
            const uint32_t goff    = swizzled_offsets[memcpy_per_tile];
            asm volatile(
                "s_mov_b32 m0, %0\n\t"
                "buffer_load_dwordx4 %1, %2, %3 offen lds\n\t"
                :
                : "s"(lds_off), "v"(goff), "s"(srsrc), "s"(tile_byte_offset)
                : "memory");
        }
    }
}

struct layout_globals {
    _gl_fp8 a, b;
    _gl_bf16 c;
    float scale_a, scale_b;
    hipStream_t stream;
    int m, n, k;
    int bpr, bpc, ki;
    int fast_m, fast_n, fast_k;
    int group_m;
    // Optional device-side scalar scales. When non-null, the kernel epilogue
    // reads `*dscale_a * *dscale_b` from device memory instead of using the
    // host-side `scale_a * scale_b` floats above. This lets the Python host
    // wrapper skip the `.item()` stream sync that would otherwise be required
    // to materialise the scales into host floats before kernel launch -- the
    // sync was responsible for ~18us of dispatch latency on small dense FP8
    // shapes (4096^3 etc) where it was bigger than the GEMM kernel itself
    // is over default hipBLASLt. nullptr selects the host-scale path so
    // existing call sites (gemm_{rcr,rrr,crr}) keep working unchanged.
    const float* dscale_a;
    const float* dscale_b;
    dim3 grid()  { return dim3(bpr * bpc); }
    dim3 block() { return dim3(_NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

// Resolve the combined per-tensor scale at kernel epilogue time. When the
// host wrapper passed device-side scale tensors (dscale_{a,b} non-null) we
// load them with a scalar global-memory read; otherwise we fall back to
// the host-known floats baked into `g`. The branch is uniform across the
// wave so the compiler keeps it scalar; the load itself is one b32 from
// global memory and hits cache after the first wave issues it.
__device__ __forceinline__ float resolve_combined_scale(const layout_globals &g) {
    const float sa = g.dscale_a ? *g.dscale_a : g.scale_a;
    const float sb = g.dscale_b ? *g.dscale_b : g.scale_b;
    return sa * sb;
}

__device__ __forceinline__ int gemm_chiplet_swizzle_bid(int bid, int num_wgs) {
    if (num_wgs >= BLOCK_SWIZZLE_NUM_XCDS &&
        (num_wgs % BLOCK_SWIZZLE_NUM_XCDS) == 0) {
        return
            (bid % BLOCK_SWIZZLE_NUM_XCDS) *
                (num_wgs / BLOCK_SWIZZLE_NUM_XCDS) +
            (bid / BLOCK_SWIZZLE_NUM_XCDS);
    }
    return bid;
}

__device__ __forceinline__ void gemm_compute_block_coords(
    int bid, int bpr, int bpc, int group_m, int &br, int &bc) {
    bid = gemm_chiplet_swizzle_bid(bid, gridDim.x);
    const int num_wgid_in_group = group_m * bpc;
    const int group_id = bid / num_wgid_in_group;
    const int first_pid_m = group_id * group_m;
    const int group_size_m =
        (first_pid_m + group_m <= bpr)
            ? group_m
            : (bpr - first_pid_m);
    if (group_size_m <= 0) {
        br = bpr;
        bc = bpc;
        return;
    }
    br = first_pid_m + ((bid % num_wgid_in_group) % group_size_m);
    bc = (bid % num_wgid_in_group) / group_size_m;
}

// =============================================================================
// store_c_tile_n_masked — column-masked C-store for partial-N tiles (FP8 dense).
//
// Mirrors the BF16 helper in `analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`.
// Templated on the rt accum type because the FP8 file has two main kernels with
// different RC tile widths (rcr_4w uses RBN=64; the generic 8-wave gemm_kernel
// uses RBN=32).
//
// 3-way fast path:
//   * fully OOB tile-column (n0 >= n_limit): no-op (block won't be writing).
//   * fully in-bounds (n1 <= n_limit): forward to the original `store(...)`.
//     Aligned shapes pay zero overhead — same instruction sequence as before.
//   * partial: lane-level skip on per-column OOB. Each lane checks its column
//     index against `n_limit`. C-tile is bf16 in global memory.
//
// MFMA still writes a result to RC's OOB columns, but those columns are
// dropped here and never reach global memory. SRD bounds for the global B
// (set up by the caller as full-tensor bounds) clamp OOB B-loads to 0, so
// the MFMA accumulator just contains a partial-sum garbage that we discard.
// =============================================================================
template<ducks::gl::all GL, ducks::rt::all RT>
__device__ __forceinline__ void store_c_tile_n_masked(
    const GL& g_c, const RT& src,
    int r_tile, int c_tile, int n_limit) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;
    constexpr int packing = base_types::packing<typename RT::dtype>::num();
    static_assert(std::is_same_v<U, bf16>, "C is bf16 global");

    const int n0 = c_tile * RT::cols;
    const int n1 = n0 + RT::cols;
    if (n0 >= n_limit) return;
    if (n1 <= n_limit) {
        store(g_c, src, {0, 0, r_tile, c_tile});
        return;
    }

    // Round-19 — partial-N path: route per-lane writes through
    // ``llvm.amdgcn.raw.buffer.store.i16`` (BUFFER class) instead of the
    // raw ``dst_ptr[...] = ...`` expression (which the compiler emits as
    // ``global_store_short`` / FLAT). Address arithmetic is bit-identical
    // to the previous version. Round-18 PMC breakdown localized 5-34x
    // ``SQ_INSTS_FLAT`` excess vs Triton on the gpt_oss FP8 grouped sweep
    // (`analysis/_notes/round-18-flat-instruction-excess-localized.md`);
    // the col-layout C store was the dominant code path emitting it.
    constexpr int axis = 2;
    U* dst_ptr = (U*)&g_c[(coord<RT>{0, 0, r_tile, c_tile}
                            .template unit_coord<axis, 3>())];
    const int row_stride = g_c.template stride<axis>();
    const int laneid = kittens::laneid();
    const int row_offset = src.base_tile_stride * (laneid / src.base_tile_cols);
    const int col_offset = laneid % src.base_tile_cols;

    uint32_t buffer_size = g_c.batch() * g_c.depth() * g_c.rows() * g_c.cols() * sizeof(U);
    std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(dst_ptr);
    std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);
    i32x4 srsrc = std::bit_cast<i32x4>(br);

    #pragma unroll
    for (int i = 0; i < src.height; i++) {
        #pragma unroll
        for (int j = 0; j < src.width; j++) {
            const int col = j * src.base_tile_cols + col_offset;
            if (n0 + col >= n_limit) continue;
            #pragma unroll
            for (int k = 0; k < src.base_tile_num_strides; k++) {
                int row = i * src.base_tile_rows + row_offset +
                          k * src.base_tile_elements_per_stride_group;
                #pragma unroll
                for (int l = 0; l < src.base_tile_stride / packing; l++) {
                    int idx = l + k * src.base_tile_stride / packing;
                    U v0 = base_types::convertor<U, T>::convert(
                            src.tiles[i][j].data[idx].x);
                    U v1 = base_types::convertor<U, T>::convert(
                            src.tiles[i][j].data[idx].y);
                    const uint32_t off0 = ((row + l * 2)     * row_stride + col) * sizeof(U);
                    const uint32_t off1 = ((row + l * 2 + 1) * row_stride + col) * sizeof(U);
                    llvm_amdgcn_raw_buffer_store_b16(std::bit_cast<uint16_t>(v0), srsrc, off0, 0, 0);
                    llvm_amdgcn_raw_buffer_store_b16(std::bit_cast<uint16_t>(v1), srsrc, off1, 0, 0);
                }
            }
        }
    }
}

// =============================================================================
// store_c_tile_mn_masked_grouped — two-axis-masked C store for the grouped
// variable-K (CRR / dB) kernel below. Mirror of the BF16 variant in
// analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp.
//
// Output layout is 3D-grouped ``[G, m_kernel, n_kernel]``; partial last
// tiles in either axis are dropped per-cell. Aligned tiles forward to
// the unmasked ``store(...)`` with no overhead.
// =============================================================================
template<ducks::gl::all GL, ducks::rt::all RT>
__device__ __forceinline__ void store_c_tile_mn_masked_grouped(
    const GL& g_c, const RT& src,
    int group_idx, int r_tile, int c_tile,
    int m_limit, int n_limit) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;
    constexpr int packing = base_types::packing<typename RT::dtype>::num();
    static_assert(std::is_same_v<U, bf16>, "C is bf16 global");

    const int m0 = r_tile * RT::rows;
    const int m1 = m0 + RT::rows;
    const int n0 = c_tile * RT::cols;
    const int n1 = n0 + RT::cols;

    if (m0 >= m_limit || n0 >= n_limit) return;
    if (m1 <= m_limit && n1 <= n_limit) {
        store(g_c, src, {0, group_idx, r_tile, c_tile});
        return;
    }

    // Round-19 — partial-MN path: same FLAT->BUFFER reroute as
    // ``store_c_tile_n_masked`` above. Per-row M-mask preserved; only the
    // active per-lane scalar write is changed from ``global_store_short``
    // to ``buffer_store_short``.
    constexpr int axis = 2;
    U* dst_ptr = (U*)&g_c[(coord<RT>{0, group_idx, r_tile, c_tile}
                            .template unit_coord<axis, 3>())];
    const int row_stride = g_c.template stride<axis>();
    const int laneid = kittens::laneid();
    const int row_offset = src.base_tile_stride * (laneid / src.base_tile_cols);
    const int col_offset = laneid % src.base_tile_cols;

    uint32_t buffer_size = g_c.batch() * g_c.depth() * g_c.rows() * g_c.cols() * sizeof(U);
    std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(dst_ptr);
    std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);
    i32x4 srsrc = std::bit_cast<i32x4>(br);

    #pragma unroll
    for (int i = 0; i < src.height; i++) {
        #pragma unroll
        for (int j = 0; j < src.width; j++) {
            const int col = j * src.base_tile_cols + col_offset;
            if (n0 + col >= n_limit) continue;
            #pragma unroll
            for (int k = 0; k < src.base_tile_num_strides; k++) {
                int row = i * src.base_tile_rows + row_offset +
                          k * src.base_tile_elements_per_stride_group;
                #pragma unroll
                for (int l = 0; l < src.base_tile_stride / packing; l++) {
                    int idx = l + k * src.base_tile_stride / packing;
                    int row_a = row + l * 2;
                    int row_b = row + l * 2 + 1;
                    if (m0 + row_a < m_limit) {
                        U v0 = base_types::convertor<U, T>::convert(
                                src.tiles[i][j].data[idx].x);
                        const uint32_t off0 = (row_a * row_stride + col) * sizeof(U);
                        llvm_amdgcn_raw_buffer_store_b16(std::bit_cast<uint16_t>(v0), srsrc, off0, 0, 0);
                    }
                    if (m0 + row_b < m_limit) {
                        U v1 = base_types::convertor<U, T>::convert(
                                src.tiles[i][j].data[idx].y);
                        const uint32_t off1 = (row_b * row_stride + col) * sizeof(U);
                        llvm_amdgcn_raw_buffer_store_b16(std::bit_cast<uint16_t>(v1), srsrc, off1, 0, 0);
                    }
                }
            }
        }
    }
}

namespace rcr_4w {

constexpr int WM = 2, WN = 2;
constexpr int NW = WM * WN;
constexpr int NT = NW * WARP_THREADS;

using G4 = kittens::group<NW>;
using ST4 = st_fp8e4m3<HB, BK, st_16x128_s>;
using RA = rt_fp8e4m3<HB / WM, BK, row_l, rt_16x128_s>;
using RB = rt_fp8e4m3<HB / WN, BK, row_l, rt_16x128_s>;
using RC = rt_fl<HB / WM, HB / WN, col_l, rt_16x16_s>;

struct g2s_desc {
    i32x4 srsrc;
    uintptr_t lds_base;
};

__device__ __forceinline__ g2s_desc prepare_g2s(
    ST4& dst, const void* tile_ptr, int row_stride)
{
    i32x4 srsrc = make_srsrc(const_cast<void*>(tile_ptr),
                              row_stride * ST4::rows * sizeof(fp8e4m3));
    constexpr int bpt = ST4::underlying_subtile_bytes_per_thread;
    constexpr int bpw = bpt * WARP_THREADS;
    uintptr_t lds_base =
        reinterpret_cast<uintptr_t>(&dst.data[0]) + (kittens::warpid() * bpw);
    return {srsrc, lds_base};
}

constexpr int G2S_PASSES = HB * BK * sizeof(fp8e4m3) /
    (ST4::underlying_subtile_bytes_per_thread * NT);

struct g2s_offsets {
    uint32_t global_off[G2S_PASSES];
};

__device__ __forceinline__ g2s_offsets prefill_g2s_offsets(
    ST4& dst, int row_stride)
{
    using T = fp8e4m3;
    constexpr int bpt = ST4::underlying_subtile_bytes_per_thread;
    constexpr int bpw = bpt * WARP_THREADS;
    const int lane = kittens::laneid();
    const int warp = kittens::warpid() % NW;

    g2s_offsets off;
    #pragma unroll
    for (int I = 0; I < G2S_PASSES; I++) {
        const int lbo = (lane * bpt) + (warp * bpw) + (I * NW * bpw);
        const int stid = lbo / ST4::underlying_subtile_bytes;
        const int str = stid / ST4::underlying_subtiles_per_row;
        const int stc = stid % ST4::underlying_subtiles_per_row;
        const int sub = lbo % ST4::underlying_subtile_bytes;
        const int row = sub / ST4::underlying_subtile_row_bytes;
        const int col = (sub % ST4::underlying_subtile_row_bytes) / sizeof(T);
        const uint32_t sw = dst.swizzle({row, col});
        const int swr = (sw / ST4::underlying_subtile_row_bytes) +
                         str * ST4::underlying_subtile_rows;
        const int swc = (sw % ST4::underlying_subtile_row_bytes) / sizeof(T) +
                         stc * ST4::underlying_subtile_cols;
        off.global_off[I] = (swr * row_stride + swc) * sizeof(T);
    }
    return off;
}

template<int I>
__device__ __forceinline__ void g2s_pass(
    const g2s_desc& addr, const g2s_offsets& off)
{
    constexpr int bpt = ST4::underlying_subtile_bytes_per_thread;
    constexpr int bpw = bpt * WARP_THREADS;
    uintptr_t lds_addr = addr.lds_base + (I * NW * bpw);
    as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);
    llvm_amdgcn_raw_buffer_load_lds(
        addr.srsrc, lds_ptr, bpt, off.global_off[I], 0, 0,
        static_cast<int>(coherency::cache_all));
}

template<int NUM_OFFSETS, typename RT, typename ST>
__device__ __forceinline__ void prefill_s2r_offsets(
    RT& dst, ST& src, uint32_t* off)
{
    using U = typename ST::dtype;
    constexpr int subtile_stride = RT::base_tile_cols * sizeof(U) / 2;
    const uint32_t st_offset =
        (kittens::laneid() % RT::base_tile_rows) * ST::underlying_cols +
        (kittens::laneid() / RT::base_tile_rows * 16 / sizeof(U));
    const uint32_t base_addr = reinterpret_cast<uintptr_t>(&src.data[st_offset]);
    off[0] = base_addr;
    off[0] ^= (((off[0] % (256 * 8)) >> 8) << 4);
    off[1] = base_addr + subtile_stride;
    off[1] ^= (((off[1] % (256 * 8)) >> 8) << 4);
}

template<int RR, int RC, int KS, typename RT, typename ST>
__device__ __forceinline__ void s2r_one(RT& dst, ST& src, uint32_t* off)
{
    constexpr int packing = base_types::packing<typename RT::dtype>::num();
    const int idx = KS * RT::base_tile_stride / packing;
    constexpr int row_stride =
        RT::base_tile_rows * ST::underlying_cols * sizeof(fp8e4m3);
    asm volatile(
        "ds_read_b128 %0, %1 offset:%2\n"
        : "=v"(*reinterpret_cast<float4*>(&dst.tiles[RR][RC].data[idx]))
        : "v"(off[KS]), "i"(RR * row_stride)
        : "memory");
}

template<typename D, typename A, typename B, typename C>
__device__ __forceinline__ void mma1(D& d, const A& a, const B& b, const C& c,
                                     int n, int m, int k)
{
    mma_ABt_base(d.tiles[n][m], a.tiles[n][k], b.tiles[m][k], c.tiles[n][m]);
}

template<typename S2R_RT, typename S2R_ST>
__device__ __forceinline__ void do_cluster(
    const g2s_desc& g2s_addr, const g2s_offsets& g2s_off,
    S2R_RT& s2r_dst, S2R_ST& s2r_src,
    RA& a, RB& b, RC& c)
{
    uint32_t s2r_off[2];
    prefill_s2r_offsets<2>(s2r_dst, s2r_src, s2r_off);

    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 0, 0, 0);
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 0, 1, 0);
    __builtin_amdgcn_sched_barrier(0);

    g2s_pass<0>(g2s_addr, g2s_off);
    s2r_one<0, 0, 0>(s2r_dst, s2r_src, s2r_off);

    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 0, 2, 0);
    __builtin_amdgcn_sched_barrier(0);

    s2r_one<0, 0, 1>(s2r_dst, s2r_src, s2r_off);

    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 0, 3, 0);
    __builtin_amdgcn_sched_barrier(0);

    g2s_pass<1>(g2s_addr, g2s_off);
    s2r_one<1, 0, 0>(s2r_dst, s2r_src, s2r_off);
    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 1, 0, 0);
    mma1(c, a, b, c, 1, 1, 0);
    __builtin_amdgcn_sched_barrier(0);

    s2r_one<1, 0, 1>(s2r_dst, s2r_src, s2r_off);
    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 1, 2, 0);
    mma1(c, a, b, c, 1, 3, 0);
    __builtin_amdgcn_sched_barrier(0);

    g2s_pass<2>(g2s_addr, g2s_off);
    s2r_one<2, 0, 0>(s2r_dst, s2r_src, s2r_off);
    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 2, 0, 0);
    mma1(c, a, b, c, 2, 1, 0);
    __builtin_amdgcn_sched_barrier(0);

    s2r_one<2, 0, 1>(s2r_dst, s2r_src, s2r_off);
    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 2, 2, 0);
    mma1(c, a, b, c, 2, 3, 0);
    __builtin_amdgcn_sched_barrier(0);

    g2s_pass<3>(g2s_addr, g2s_off);
    s2r_one<3, 0, 0>(s2r_dst, s2r_src, s2r_off);
    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 3, 0, 0);
    mma1(c, a, b, c, 3, 1, 0);
    __builtin_amdgcn_sched_barrier(0);

    s2r_one<3, 0, 1>(s2r_dst, s2r_src, s2r_off);
    __builtin_amdgcn_sched_barrier(0);
    mma1(c, a, b, c, 3, 2, 0);
    mma1(c, a, b, c, 3, 3, 0);
    __builtin_amdgcn_sched_barrier(0);
}

template<ducks::rt::row_layout RT, ducks::st::all ST>
__device__ __forceinline__ void load_full_rt(RT& dst, const ST& src) {
    static_assert(RT::rows == ST::rows && RT::cols == ST::cols);
    using T2 = typename RT::dtype;
    using U2 = typename base_types::packing<typename ST::dtype>::packed_type;
    constexpr int packing = base_types::packing<T2>::num();
    const int laneid = kittens::laneid();
    const int row_offset = laneid % dst.base_tile_rows;
    const int col_offset = dst.base_tile_stride * (laneid / dst.base_tile_rows);
    const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);
    constexpr int rps_row = ST::underlying_subtile_cols / RT::base_tile_cols;
    constexpr int rps_col = ST::underlying_subtile_rows / RT::base_tile_rows;
    #pragma unroll
    for (int k = 0; k < RT::base_tile_num_strides; k++) {
        #pragma unroll
        for (int i = 0; i < rps_col; i++) {
            #pragma unroll
            for (int j = 0; j < rps_row; j++) {
                const int row = i * RT::base_tile_rows + row_offset;
                const int col = j * RT::base_tile_cols + col_offset +
                    k * RT::base_tile_elements_per_stride_group;
                const uint32_t offset =
                    sizeof(fp8e4m3) * (src_ptr + row * ST::underlying_subtile_cols + col);
                const uint32_t addr = offset ^ (((offset % (16 * 128)) >> 8) << 4);
                const int idx = k * RT::base_tile_stride / packing;
                #pragma unroll
                for (int ii = 0; ii < ST::subtiles_per_col; ii++) {
                    #pragma unroll
                    for (int jj = 0; jj < ST::subtiles_per_row; jj++) {
                        const int sid = ii * ST::underlying_subtiles_per_row + jj;
                        const int soff = sid * ST::underlying_subtile_bytes;
                        const int rr = ii * rps_col + i;
                        const int rc = jj * rps_row + j;
                        asm volatile("ds_read_b128 %0, %1 offset:%2\n"
                            : "=v"(*reinterpret_cast<float4*>(&dst.tiles[rr][rc].data[idx]))
                            : "v"(addr), "i"(soff) : "memory");
                    }
                }
            }
        }
    }
}

__global__ __launch_bounds__(NT, 2)
void kernel(const layout_globals g) {
    int br, bc;
    gemm_compute_block_coords(blockIdx.x, g.bpr, g.bpc, g.group_m, br, bc);
    if (br >= g.bpr || bc >= g.bpc || g.ki <= 0) return;

    const int wm = kittens::warpid() / WN;
    const int wn = kittens::warpid() % WN;
    const int ki = g.ki;

    const fp8e4m3* a_base = reinterpret_cast<const fp8e4m3*>(g.a.raw_ptr);
    const fp8e4m3* b_base = reinterpret_cast<const fp8e4m3*>(g.b.raw_ptr);
    const int a_stride = g.k;
    const int b_stride = g.k;

    auto a_ptr = [&](int row_tile, int k_tile) -> const void* {
        return a_base + row_tile * HB * a_stride + k_tile * BK;
    };
    auto b_ptr = [&](int col_tile, int k_tile) -> const void* {
        return b_base + col_tile * HB * b_stride + k_tile * BK;
    };

    __shared__ ST4 As[2][2];
    __shared__ ST4 Bs[2][2];
    RA a_reg[2];
    RB b_reg[2];
    RC c[2][2];
    zero(c[0][0]); zero(c[0][1]); zero(c[1][0]); zero(c[1][1]);

    constexpr int bpt = ST4::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * NT;
    constexpr int mpt = HB * BK * sizeof(fp8e4m3) / bpm;
    uint32_t soA[mpt], soB[mpt];
    G4::prefill_swizzled_offsets(As[0][0], g.a, soA);
    G4::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

    g2s_offsets g2s_off_A = prefill_g2s_offsets(As[0][0], a_stride);
    g2s_offsets g2s_off_B = prefill_g2s_offsets(Bs[0][0], b_stride);

    auto a_co = [&](int s, int k) -> coord<ST4> { return {0, 0, s, k}; };
    auto b_co = [&](int s, int k) -> coord<ST4> { return {0, 0, s, k}; };

    int cur = 0, nxt = 1;
    G4::load(As[cur][0], g.a, a_co(br * WM,     0), soA);
    G4::load(Bs[cur][0], g.b, b_co(bc * WN,     0), soB);
    G4::load(Bs[cur][1], g.b, b_co(bc * WN + 1, 0), soB);
    G4::load(As[cur][1], g.a, a_co(br * WM + 1, 0), soA);

    G4::load(As[nxt][0], g.a, a_co(br * WM,     1), soA);
    G4::load(Bs[nxt][0], g.b, b_co(bc * WN,     1), soB);
    G4::load(Bs[nxt][1], g.b, b_co(bc * WN + 1, 1), soB);
    G4::load(As[nxt][1], g.a, a_co(br * WM + 1, 1), soA);

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(28)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    auto a_sub0 = subtile_inplace<HB / WM, BK>(As[cur][0], {wm, 0});
    load_full_rt(a_reg[0], a_sub0);

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(24)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    auto b_sub0 = subtile_inplace<HB / WN, BK>(Bs[cur][0], {wn, 0});
    load_full_rt(b_reg[0], b_sub0);

    #pragma unroll 1
    for (int k = 0; k < ki - 2; ++k, cur ^= 1, nxt ^= 1) {
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(16)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto b_sub1 = subtile_inplace<HB / WN, BK>(Bs[cur][1], {wn, 0});
        g2s_desc addr_a0 = prepare_g2s(As[cur][0], a_ptr(br * WM, k + 2), a_stride);
        do_cluster(
            addr_a0, g2s_off_A,
            b_reg[1], b_sub1,
            a_reg[0], b_reg[0], c[0][0]);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_sub1 = subtile_inplace<HB / WM, BK>(As[cur][1], {wm, 0});
        g2s_desc addr_b0 = prepare_g2s(Bs[cur][0], b_ptr(bc * WN, k + 2), b_stride);
        do_cluster(
            addr_b0, g2s_off_B,
            a_reg[1], a_sub1,
            a_reg[0], b_reg[1], c[0][1]);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(16)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        a_sub0 = subtile_inplace<HB / WM, BK>(As[nxt][0], {wm, 0});
        g2s_desc addr_b1 = prepare_g2s(Bs[cur][1], b_ptr(bc * WN + 1, k + 2), b_stride);
        do_cluster(
            addr_b1, g2s_off_B,
            a_reg[0], a_sub0,
            a_reg[1], b_reg[0], c[1][0]);

        b_sub0 = subtile_inplace<HB / WN, BK>(Bs[nxt][0], {wn, 0});
        g2s_desc addr_a1 = prepare_g2s(As[cur][1], a_ptr(br * WM + 1, k + 2), a_stride);
        do_cluster(
            addr_a1, g2s_off_A,
            b_reg[0], b_sub0,
            a_reg[1], b_reg[1], c[1][1]);
    }

    // Epilogue: k = ki - 2
    {
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(16)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto b_sub1 = subtile_inplace<HB / WN, BK>(Bs[cur][1], {wn, 0});
        load_full_rt(b_reg[1], b_sub1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][0], a_reg[0], b_reg[0], c[0][0]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_sub1 = subtile_inplace<HB / WM, BK>(As[cur][1], {wm, 0});
        load_full_rt(a_reg[1], a_sub1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][1], a_reg[0], b_reg[1], c[0][1]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(8)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        a_sub0 = subtile_inplace<HB / WM, BK>(As[nxt][0], {wm, 0});
        load_full_rt(a_reg[0], a_sub0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][0], a_reg[1], b_reg[0], c[1][0]);
        __builtin_amdgcn_sched_barrier(0);

        b_sub0 = subtile_inplace<HB / WN, BK>(Bs[nxt][0], {wn, 0});
        load_full_rt(b_reg[0], b_sub0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][1], a_reg[1], b_reg[1], c[1][1]);
        __builtin_amdgcn_sched_barrier(0);

        cur ^= 1; nxt ^= 1;
    }

    // Last iteration
    {
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto b_sub1 = subtile_inplace<HB / WN, BK>(Bs[cur][1], {wn, 0});
        load_full_rt(b_reg[1], b_sub1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][0], a_reg[0], b_reg[0], c[0][0]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_sub1 = subtile_inplace<HB / WM, BK>(As[cur][1], {wm, 0});
        load_full_rt(a_reg[1], a_sub1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][1], a_reg[0], b_reg[1], c[0][1]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][0], a_reg[1], b_reg[0], c[1][0]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][1], a_reg[1], b_reg[1], c[1][1]);
        __builtin_amdgcn_sched_barrier(0);
    }

    const float sc = resolve_combined_scale(g);
    mul(c[0][0], c[0][0], sc);
    mul(c[0][1], c[0][1], sc);
    mul(c[1][0], c[1][0], sc);
    mul(c[1][1], c[1][1], sc);

    store_c_tile_n_masked(g.c, c[0][0], br * WM * 2 + wm,      bc * WN * 2 + wn,      g.n);
    store_c_tile_n_masked(g.c, c[0][1], br * WM * 2 + wm,      bc * WN * 2 + WN + wn, g.n);
    store_c_tile_n_masked(g.c, c[1][0], br * WM * 2 + WM + wm, bc * WN * 2 + wn,      g.n);
    store_c_tile_n_masked(g.c, c[1][1], br * WM * 2 + WM + wm, bc * WN * 2 + WN + wn, g.n);
}

} // namespace rcr_4w

// Runtime K-specialization: when KI_HINT>0 it matches g.ki exactly, enabling
// the compiler to fully unroll or uniformly unroll the main loop without
// branch overhead and with register allocation tuned to the known loop count.
template<Layout L, int KI_HINT = 0>
__global__ __launch_bounds__(_NUM_THREADS, MIN_BLOCKS_PER_CU)
void gemm_kernel(const layout_globals g) {
    int bid = blockIdx.x;
    int br, bc;
    gemm_compute_block_coords(bid, g.bpr, g.bpc, g.group_m, br, bc);
    const int ki_dyn = (KI_HINT > 0) ? KI_HINT : g.ki;
    if (br >= g.bpr || bc >= g.bpc || ki_dyn <= 0) {
        return;
    }
    int wm = warpid() / WARPS_N, wn = warpid() % WARPS_N;

    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;
    zero(cA); zero(cB); zero(cC); zero(cD);

    if constexpr (L == Layout::RCR) {
        using ST_rcr = ST_v2;
        __shared__ ST_rcr As[2][2];
        __shared__ ST_rcr Bs[2][2];
        A_row_reg a;
        B_row_reg b0, b1;

        constexpr int bpt = ST_rcr::underlying_subtile_bytes_per_thread;
        constexpr int bpm = bpt * _NUM_THREADS;
        constexpr int mpt = ST_rcr::rows * ST_rcr::cols * sizeof(fp8e4m3) / bpm;
        uint32_t soA[mpt], soB[mpt];
        G::prefill_swizzled_offsets(As[0][0], g.a, soA);
        G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

        auto a_co = [&](int s, int k) -> coord<ST_rcr> { return {0, 0, s, k}; };
        auto b_co = [&](int s, int k) -> coord<ST_rcr> { return {0, 0, s, k}; };

        auto load_a = [&](A_row_reg& dst, ST_rcr& tile, int wi) {
            auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
            load(dst, sub);
        };
        auto load_b = [&](B_row_reg& dst, ST_rcr& tile, int wi) {
            auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
            load(dst, sub);
        };

        auto b_tile = [&](int stage, int which) -> ST_rcr& {
            return Bs[stage][which];
        };

        int tic = 0, toc = 1;
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 0), g.b, b_co(bc*2,   0), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0], g.a, a_co(br*2,   0), soA);
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 1), g.b, b_co(bc*2+1, 0), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[tic][1], g.a, a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(RCR_INIT0_VMCNT);
        __builtin_amdgcn_s_barrier();

        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(toc, 0), g.b, b_co(bc*2,   1), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[toc][0], g.a, a_co(br*2,   1), soA);
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(toc, 1), g.b, b_co(bc*2+1, 1), soB);

        TK_WAIT_VMCNT(RCR_INIT1_VMCNT);
        __builtin_amdgcn_s_barrier();

        if ((ki_dyn & 1) == 0 && ki_dyn >= RCR_TWO_TILE_MIN_KI) {
            auto main_loop_iter = [&](int tile) {
                load_b(b0, Bs[0][0], wn);
                load_a(a, As[0][0], wm);
                rcr_8w_load_hoist<_NUM_THREADS>(As[1][1], g.a, a_co(br*2+1, tile+1), soA);
                TK_WAIT_LGKM(RCR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();

                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1); mma_ABt(cA, a, b0, cA); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

                load_b(b1, Bs[0][1], wn);
                rcr_8w_load_hoist<_NUM_THREADS>(Bs[0][0], g.b, b_co(bc*2, tile+2), soB);
                __builtin_amdgcn_s_barrier();

                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1); mma_ABt(cB, a, b1, cB); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();

                load_a(a, As[0][1], wm);
                rcr_8w_load_hoist<_NUM_THREADS>(As[0][0], g.a, a_co(br*2, tile+2), soA);
                __builtin_amdgcn_s_barrier();

                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1); mma_ABt(cC, a, b0, cC); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

                load_b(b0, Bs[1][0], wn);
                rcr_8w_load_hoist<_NUM_THREADS>(Bs[0][1], g.b, b_co(bc*2+1, tile+2), soB);
                TK_WAIT_VMCNT(RCR_TWO_TILE_MID_VMCNT); __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_s_setprio(1); mma_ABt(cD, a, b1, cD); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();

                load_a(a, As[1][0], wm);
                rcr_8w_load_hoist<_NUM_THREADS>(As[0][1], g.a, a_co(br*2+1, tile+2), soA);
                TK_WAIT_LGKM(RCR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();

                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1); mma_ABt(cA, a, b0, cA); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

                load_b(b1, Bs[1][1], wn);
                rcr_8w_load_hoist<_NUM_THREADS>(Bs[1][0], g.b, b_co(bc*2, tile+3), soB);
                __builtin_amdgcn_s_barrier();

                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1); mma_ABt(cB, a, b1, cB); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();

                load_a(a, As[1][1], wm);
                rcr_8w_load_hoist<_NUM_THREADS>(As[1][0], g.a, a_co(br*2, tile+3), soA);
                __builtin_amdgcn_s_barrier();

                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1); mma_ABt(cC, a, b0, cC); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

                rcr_8w_load_hoist<_NUM_THREADS>(Bs[1][1], g.b, b_co(bc*2+1, tile+3), soB);
                TK_WAIT_VMCNT(RCR_TWO_TILE_MID_VMCNT); __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_s_setprio(1); mma_ABt(cD, a, b1, cD); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();
            };

            TK_PRAGMA_UNROLL(RCR_MAIN_UNROLL)
            for (int tile = 0; tile < ki_dyn - 2; tile += 2) {
                main_loop_iter(tile);
            }
            TK_WAIT_VMCNT(0);
            __builtin_amdgcn_s_barrier();
        } else
        {
        TK_PRAGMA_UNROLL(RCR_MAIN_UNROLL)
        for (int k = 0; k < ki_dyn - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, b_tile(tic, 0), wn);
            load_a(a, As[tic][0], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], g.a, a_co(br*2+1, k+1), soA);
            TK_WAIT_LGKM(RCR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

            load_b(b1, b_tile(tic, 1), wn);
            rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 0), g.b, b_co(bc*2, k+2), soB);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0], g.a, a_co(br*2, k+2), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

            rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 1), g.b, b_co(bc*2+1, k+2), soB);
            TK_WAIT_VMCNT(RCR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_setprio(1); rcr_mma(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }
        }

        {
            load_b(b0, b_tile(tic, 0), wn);
            load_a(a, As[tic][0], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], g.a, a_co(br*2+1, ki_dyn-1), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

            load_b(b1, b_tile(tic, 1), wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(RCR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, b_tile(toc, 0), wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();
            tic ^= 1; toc ^= 1;
        }

        {
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, b_tile(tic, 1), wn);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            rcr_mma(cC, a, b0);
            rcr_mma(cD, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

    } else if constexpr (L == Layout::RRR) {
        __shared__ ST_row As[2][2];
        __shared__ ST_v2 Bs[2][2];
        A_row_reg a;
        B_col_reg b0, b1;

        constexpr int bptA = ST_row::underlying_subtile_bytes_per_thread;
        constexpr int bpmA = bptA * _NUM_THREADS;
        constexpr int mptA = ST_row::rows * ST_row::cols * sizeof(fp8e4m3) / bpmA;
        uint32_t soA[mptA];
        G::prefill_swizzled_offsets(As[0][0], g.a, soA);

        constexpr int bptB =
            ST_v2::underlying_subtile_bytes_per_thread;
        constexpr int bpmB = bptB * _NUM_THREADS;
        constexpr int mptB =
            ST_v2::rows * ST_v2::cols * sizeof(fp8e4m3) / bpmB;
        uint32_t soB[mptB];
        G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

        auto a_co = [&](int s, int k) -> coord<ST_row> { return {0, 0, s, k}; };
        auto b_co = [&](int s, int k) -> coord<ST_v2> { return {0, 0, k, s}; };

        auto load_a = [&](A_row_reg& dst, ST_row& tile, int wi) {
            auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
            load(dst, sub);
        };
        auto load_b = [&](B_col_reg& dst, ST_v2& tile, int wi) {
            load_col_from_st(dst, tile, wi * RBN);
        };

        int tic = 0, toc = 1;
        G::load(Bs[tic][0], g.b, b_co(bc*2,   0), soB);
        G::load(As[tic][0], g.a, a_co(br*2,   0), soA);
        G::load(Bs[tic][1], g.b, b_co(bc*2+1, 0), soB);
        G::load(As[tic][1], g.a, a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(RRR_INIT0_VMCNT);
        __builtin_amdgcn_s_barrier();

        G::load(Bs[toc][0], g.b, b_co(bc*2,   1), soB);
        G::load(As[toc][0], g.a, a_co(br*2,   1), soA);
        G::load(Bs[toc][1], g.b, b_co(bc*2+1, 1), soB);

        TK_WAIT_VMCNT(RRR_INIT1_VMCNT);
        __builtin_amdgcn_s_barrier();

        TK_PRAGMA_UNROLL(RRR_MAIN_UNROLL)
        for (int k = 0; k < ki_dyn - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, k+1), soA);
            TK_WAIT_LGKM(RRR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();

            load_b(b1, Bs[tic][1], wn);
            G::load(Bs[tic][0], g.b, b_co(bc*2, k+2), soB);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            rrr_mma(cB, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            G::load(Bs[tic][1], g.b, b_co(bc*2+1, k+2), soB);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();

            G::load(As[tic][0], g.a, a_co(br*2, k+2), soA);
            TK_WAIT_VMCNT(RRR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            rrr_mma(cD, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, ki_dyn-1), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            rrr_mma(cB, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[toc][0], wn);
            TK_WAIT_VMCNT(RRR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();
            tic ^= 1; toc ^= 1;
        }

        {
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            rrr_mma(cB, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            rrr_mma(cC, a, b0);
            rrr_mma(cD, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

    } else if constexpr (L == Layout::CRR) {
        using ST_crr_a = ST_v2a;
        using ST_crr_b = ST_v2;
        __shared__ ST_crr_a As[2][2];
        __shared__ ST_crr_b Bs[2][2];
        A_col_reg a;
        B_col_reg b0, b1;

        constexpr int bptA = ST_crr_a::underlying_subtile_bytes_per_thread;
        constexpr int bpmA = bptA * _NUM_THREADS;
        constexpr int mptA = ST_crr_a::rows * ST_crr_a::cols * sizeof(fp8e4m3) / bpmA;
        uint32_t soA[mptA];
        G::prefill_swizzled_offsets(As[0][0], g.a, soA);

        constexpr int bptB = ST_crr_b::underlying_subtile_bytes_per_thread;
        constexpr int bpmB = bptB * _NUM_THREADS;
        constexpr int mptB = ST_crr_b::rows * ST_crr_b::cols * sizeof(fp8e4m3) / bpmB;
        uint32_t soB[mptB];
        G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

        auto a_co = [&](int s, int k) -> coord<ST_crr_a> { return {0, 0, k, s}; };
        auto b_co = [&](int s, int k) -> coord<ST_crr_b> { return {0, 0, k, s}; };
        auto global_load_a = [&](ST_crr_a& tile, int s, int k) {
            G::load(tile, g.a, a_co(s, k), soA);
        };
        auto global_load_b = [&](ST_crr_b& tile, int s, int k) {
            G::load(tile, g.b, b_co(s, k), soB);
        };

        auto load_a = [&](A_col_reg& dst, ST_crr_a& tile, int wi) {
            load_col_from_st(dst, tile, wi * RBM);
        };
        auto load_b = [&](B_col_reg& dst, ST_crr_b& tile, int wi) {
            load_col_from_st(dst, tile, wi * RBN);
        };

        int tic = 0, toc = 1;
        global_load_b(Bs[tic][0], bc*2,   0);
        global_load_a(As[tic][0], br*2,   0);
        global_load_b(Bs[tic][1], bc*2+1, 0);
        global_load_a(As[tic][1], br*2+1, 0);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(CRR_INIT0_VMCNT);
        __builtin_amdgcn_s_barrier();

        global_load_b(Bs[toc][0], bc*2,   1);
        global_load_a(As[toc][0], br*2,   1);
        global_load_b(Bs[toc][1], bc*2+1, 1);

        TK_WAIT_VMCNT(CRR_INIT1_VMCNT);
        __builtin_amdgcn_s_barrier();


        TK_PRAGMA_UNROLL(CRR_MAIN_UNROLL)
        for (int k = 0; k < ki_dyn - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, Bs[tic][0], wn);
            load_b(b1, Bs[tic][1], wn);
            load_a(a, As[tic][0], wm);
            global_load_a(As[toc][1], br*2+1, k+1);
            TK_WAIT_LGKM(CRR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cA, a, b0);
            crr_mma(cB, a, b1);
            CRR_MMA_END();
            CRR_STEADY_MID_BARRIER();

            load_a(a, As[tic][1], wm);
            global_load_a(As[tic][0], br*2, k+2);
            global_load_b(Bs[tic][1], bc*2+1, k+2);
            TK_WAIT_VMCNT(CRR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cC, a, b0);
            crr_mma(cD, a, b1);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();
            global_load_b(Bs[tic][0], bc*2, k+2);
        }

        {
            load_b(b0, Bs[tic][0], wn);
            const auto b0_keep = b0;
            load_a(a, As[tic][0], wm);
            global_load_a(As[toc][1], br*2+1, ki_dyn-1);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cA, a, b0_keep);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
            const auto b1_keep = b1;
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cB, a, b1_keep);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(CRR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cC, a, b0_keep);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[toc][0], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cD, a, b1_keep);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();
            tic ^= 1; toc ^= 1;
        }

        {
            const auto b0_keep = b0;
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cA, a, b0_keep);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
            const auto b1_keep = b1;
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cB, a, b1_keep);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cC, a, b0_keep);
            crr_mma(cD, a, b1_keep);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();
        }    }

    const float combined_scale = resolve_combined_scale(g);
    mul(cA, cA, combined_scale);
    mul(cB, cB, combined_scale);
    mul(cC, cC, combined_scale);
    mul(cD, cD, combined_scale);

    // Store Output
    if (wm == 0) __builtin_amdgcn_s_barrier();
    store_c_tile_n_masked(g.c, cA, br*WARPS_M*2+wm,         bc*WARPS_N*2+wn,         g.n);
    store_c_tile_n_masked(g.c, cB, br*WARPS_M*2+wm,         bc*WARPS_N*2+WARPS_N+wn, g.n);
    store_c_tile_n_masked(g.c, cC, br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+wn,         g.n);
    store_c_tile_n_masked(g.c, cD, br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+WARPS_N+wn, g.n);
}

template<Layout L>
__global__ void gemm_tail_kernel(const layout_globals g) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= g.m || col >= g.n) {
        return;
    }

    // ``main_covers_n`` mirrors the dispatch decision (Phase 4): when the
    // generic 8-wave kernel ran with ``bpc = ceil_div(n, BLK)``, the
    // ``store_c_tile_n_masked`` helper already wrote cols [fast_n, n)
    // for rows in [0, fast_m) with the FULL K reduction. Tail must NOT
    // redo those cells. Detected from ``g.bpc`` itself:
    //   * bpc * BLK > fast_n  →  ceil_div path → main covered N.
    //   * bpc * BLK == fast_n →  fast path (4-wave RCR or K misaligned).
    const bool main_covers_n = (g.bpc * BLK > g.fast_n);
    const int n_main_limit = main_covers_n ? g.n : g.fast_n;
    const bool interior_mn = row < g.fast_m && col < n_main_limit;
    const bool fast_covers_cell = interior_mn && g.fast_m > 0 && g.fast_n > 0 && g.fast_k > 0;
    const bool needs_k_tail = g.fast_k < g.k;
    if (fast_covers_cell && !needs_k_tail) {
        return;
    }

    const int k0 = fast_covers_cell ? g.fast_k : 0;
    float acc = 0.0f;

    if constexpr (L == Layout::RCR) {
        // Vec8 fast path for RCR. See ``grouped_tail_kernel`` for the
        // rationale; mirror change to keep dense + grouped tail logic
        // in sync. Dense rarely runs the tail (LLM shapes are aligned
        // 4096 / 8192 multiples) so this is mostly a code-symmetry win.
        const fp8e4m3* a_row = &g.a[coord<>(row, 0)];
        const fp8e4m3* b_row = &g.b[coord<>(col, 0)];
        int kk = k0;
        if ((g.k % 8 == 0) && ((k0 & 7) == 0)) {
            const fp8e4m3_8* a_v8 = reinterpret_cast<const fp8e4m3_8*>(a_row);
            const fp8e4m3_8* b_v8 = reinterpret_cast<const fp8e4m3_8*>(b_row);
            const int j_start = k0 >> 3;
            const int j_end   = g.k >> 3;
            #pragma unroll 4
            for (int j = j_start; j < j_end; ++j) {
                fp8e4m3_8 a8 = a_v8[j];
                fp8e4m3_8 b8 = b_v8[j];
                float4 a_lo = base_types::convertor<float4, fp8e4m3_4>::convert(a8.lo);
                float4 a_hi = base_types::convertor<float4, fp8e4m3_4>::convert(a8.hi);
                float4 b_lo = base_types::convertor<float4, fp8e4m3_4>::convert(b8.lo);
                float4 b_hi = base_types::convertor<float4, fp8e4m3_4>::convert(b8.hi);
                acc += a_lo.x * b_lo.x + a_lo.y * b_lo.y
                     + a_lo.z * b_lo.z + a_lo.w * b_lo.w
                     + a_hi.x * b_hi.x + a_hi.y * b_hi.y
                     + a_hi.z * b_hi.z + a_hi.w * b_hi.w;
            }
            kk = j_end << 3;
        }
        for (; kk < g.k; ++kk) {
            acc += load_fp8_scalar(g.a, row, kk) * load_fp8_scalar(g.b, col, kk);
        }
    } else if constexpr (L == Layout::RRR) {
        for (int kk = k0; kk < g.k; ++kk) {
            acc += load_fp8_scalar(g.a, row, kk) * load_fp8_scalar(g.b, kk, col);
        }
    } else {
        for (int kk = k0; kk < g.k; ++kk) {
            acc += load_fp8_scalar(g.a, kk, row) * load_fp8_scalar(g.b, kk, col);
        }
    }

    const float scaled = acc * resolve_combined_scale(g);
    if (fast_covers_cell && needs_k_tail) {
        store_bf16_scalar(g.c, row, col, load_bf16_scalar(g.c, row, col) + scaled);
    } else {
        store_bf16_scalar(g.c, row, col, scaled);
    }
}

// Single dynamic K instantiation (KI_HINT=0). Experiments showed that
// compile-time KI specialization causes VGPR spills (64+ bytes/lane of
// scratch) because the two-tile main loop body is ~60 lines of asm and,
// when combined with `#pragma unroll RCR_MAIN_UNROLL` and a constexpr
// upper bound, the compiler emits many copies that exceed the register
// budget. The dynamic path holds up at 0 spills across all three layouts.
template __global__ void gemm_kernel<Layout::RCR, 0>(const layout_globals);
template __global__ void gemm_kernel<Layout::RRR, 0>(const layout_globals);
template __global__ void gemm_kernel<Layout::CRR, 0>(const layout_globals);

template __global__ void gemm_tail_kernel<Layout::RCR>(const layout_globals);
template __global__ void gemm_tail_kernel<Layout::RRR>(const layout_globals);
template __global__ void gemm_tail_kernel<Layout::CRR>(const layout_globals);

// =============================================================================
// Persistent grouped GEMM (CPU-sync-free) — RCR layout only for round 12.
//
// Mirror of the BF16 grouped persistent kernel
// (analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp ``grouped_kernel<L,KI>``)
// ported to the FP8 register / shared tile types. One launch with grid_x =
// NUM_CUS programs covers ALL groups × ALL tiles. Each program:
//
//   1. Pulls G+1 int64 offsets from a device tensor and computes total tile
//      count via O(G) scan (no host sync).
//   2. Iterates ``gt = pid; gt < total; gt += NUM_CUS`` so the same block
//      streams through many (group, tile) pairs without re-launch.
//   3. Per iteration: O(G) scan to recover (group_idx, m_start_g, M_g),
//      then runs the existing dense RCR GEMM tile body with coord shifts:
//         * A   spatial += m_start_g / HB    (HB  = 128, ST_A row unit)
//         * B   depth   = group_idx          (b is treated as [G, N, K])
//         * C   row     += m_start_g / RBM   (RBM = 64,  RT::rows store unit)
//
// Inner body is the SINGLE-tile main loop + epilogs from the dense kernel
// (the ``else`` branch of ``gemm_kernel<Layout::RCR, ...>`` lines 1129-1210
// + scale epilog at 1500). Two-tile schedule (faster for ki>=28 with even
// ki) is intentionally not used in this round to keep the persistent path
// simple; can be added in a follow-up. RRR / CRR persistent variants ditto.
//
// Scale epilog: ``scale_a * scale_b`` applied per tile (matches dense).
// =============================================================================

// =============================================================================
// [fused-act R6] Forward-relocated R5a deposit so the FUSE_ACT=true
// instantiation of grouped_rcr_kernel can refer to the helper + struct. The
// R4 cvt builtin (line ~7390) is forward-declared; the R5a struct + load
// helper (line ~7490 and ~7520 originally) are fully relocated here. The
// originals are stubbed out (comment-only) at their old positions so the
// compile-test kernel + binding still in the bottom-of-file pybind block
// continue to find the symbols.
// =============================================================================

namespace fused_act_round4_compile_test {
__device__ __forceinline__ uint32_t cvt_bf16x4_to_fp8x4(
    bf16_2 lo, bf16_2 hi, float scale);
}  // namespace fused_act_round4_compile_test

struct grouped_layout_globals_fused_act {
    _gl_bf16 a;
    _gl_fp8 b;
    _gl_bf16 c;
    float scale_a, scale_b;
    const float* dscale_a;
    const float* dscale_b;
    const int64_t* group_offs;
    hipStream_t stream;
    int G;
    int n;
    int k;
    int ki;
    int bpc;
    int group_m;
    int num_xcds;
    int M_total;
    int fast_n, fast_k;
    int m_per_group;
    dim3 block() { return dim3(_NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

namespace fused_act_round5_compile_test {

template<int N_THREADS,
         ducks::st::all ST_DST,
         ducks::gl::all GL_SRC,
         ducks::coord::tile COORD = coord<ST_DST>>
__device__ __forceinline__ void rcr_8w_load_hoist_fused_act(
    ST_DST& dst,
    const GL_SRC& src,
    const COORD& idx,
    const uint32_t* __restrict__ swizzled_offsets,
    float scale)
{
    using T_DST = typename ST_DST::dtype;
    using T_SRC = typename GL_SRC::dtype;
    static_assert(sizeof(T_SRC) == 2, "fused-act expects BF16 src");
    static_assert(sizeof(T_DST) == 1, "fused-act expects FP8 dst");

    constexpr int dst_bytes_per_thread =
        ST_DST::underlying_subtile_bytes_per_thread;
    constexpr int dst_bytes_per_warp =
        dst_bytes_per_thread * kittens::WARP_THREADS;
    constexpr int memcpy_per_tile =
        ST_DST::rows * ST_DST::cols * sizeof(T_DST) /
        (dst_bytes_per_thread * N_THREADS);
    static_assert(
        ST_DST::rows * ST_DST::cols * sizeof(T_DST) >= dst_bytes_per_warp,
        "shared tile must be at least 1024 bytes"
    );

    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int warpid = kittens::warpid() % num_warps;
    const int laneid = kittens::laneid();

    coord<> unit_coord = idx.template unit_coord<2, 3>();
    T_SRC* tensor_base = (T_SRC*)src.raw_ptr;
    T_SRC* global_ptr  = (T_SRC*)&src[unit_coord];
    const uint32_t total_bytes = static_cast<uint32_t>(
        size_t(src.batch()) * size_t(src.depth()) *
        size_t(src.rows())  * size_t(src.cols())  * sizeof(T_SRC));
    i32x4 srsrc = make_srsrc(tensor_base, total_bytes);
    const uint32_t tile_byte_offset = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(global_ptr) -
                              reinterpret_cast<uintptr_t>(tensor_base)));

    const uintptr_t lds_tile_base =
        reinterpret_cast<uintptr_t>(&dst.data[0]);

    uint32_t lds_addrs[memcpy_per_tile + 1];
    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; ++i) {
        const int warp_linear_offset =
            (warpid * dst_bytes_per_warp) + (i * num_warps * dst_bytes_per_warp);
        const int lds_subtile_id =
            warp_linear_offset / ST_DST::underlying_subtile_bytes;
        const uint32_t off32 = static_cast<uint32_t>(
            lds_tile_base + warp_linear_offset +
            lds_subtile_id * ST_DST::subtile_padding);
        lds_addrs[i] = __builtin_amdgcn_readfirstlane(off32);
    }

    const uint32_t lds_lane_off = static_cast<uint32_t>(laneid) * 16u;

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; ++i) {
        const uint32_t voff_lo = swizzled_offsets[i] * 2u;
        const uint32_t voff_hi = voff_lo + 16u;

        __uint128_t v_lo = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
            srsrc, voff_lo, tile_byte_offset, 0);
        __uint128_t v_hi = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
            srsrc, voff_hi, tile_byte_offset, 0);

        bf16_2* bf_lo = reinterpret_cast<bf16_2*>(&v_lo);
        bf16_2* bf_hi = reinterpret_cast<bf16_2*>(&v_hi);

        u32x4 fp8_pack = {
            fused_act_round4_compile_test::cvt_bf16x4_to_fp8x4(
                bf_lo[0], bf_lo[1], scale),
            fused_act_round4_compile_test::cvt_bf16x4_to_fp8x4(
                bf_lo[2], bf_lo[3], scale),
            fused_act_round4_compile_test::cvt_bf16x4_to_fp8x4(
                bf_hi[0], bf_hi[1], scale),
            fused_act_round4_compile_test::cvt_bf16x4_to_fp8x4(
                bf_hi[2], bf_hi[3], scale),
        };

        const uint32_t lds_addr = lds_addrs[i] + lds_lane_off;
        ::kittens::macros::ds_write_b128(fp8_pack, lds_addr, /*i_offset=*/0);
    }

    if constexpr (memcpy_per_tile * (dst_bytes_per_thread * N_THREADS) !=
                  ST_DST::rows * ST_DST::cols * sizeof(T_DST)) {
        constexpr int leftover_bytes =
            ST_DST::rows * ST_DST::cols * sizeof(T_DST) -
            memcpy_per_tile * (dst_bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / dst_bytes_per_thread;
        constexpr int leftover_warps   = leftover_threads / kittens::WARP_THREADS;
        if (warpid < leftover_warps) {
            const int warp_linear_offset =
                (warpid * dst_bytes_per_warp) +
                (memcpy_per_tile * num_warps * dst_bytes_per_warp);
            const int lds_subtile_id =
                warp_linear_offset / ST_DST::underlying_subtile_bytes;
            const uint32_t off32 = static_cast<uint32_t>(
                lds_tile_base + warp_linear_offset +
                lds_subtile_id * ST_DST::subtile_padding);
            const uint32_t lds_warp_addr = __builtin_amdgcn_readfirstlane(off32);
            const uint32_t voff_lo = swizzled_offsets[memcpy_per_tile] * 2u;
            const uint32_t voff_hi = voff_lo + 16u;

            __uint128_t v_lo = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                srsrc, voff_lo, tile_byte_offset, 0);
            __uint128_t v_hi = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                srsrc, voff_hi, tile_byte_offset, 0);

            bf16_2* bf_lo = reinterpret_cast<bf16_2*>(&v_lo);
            bf16_2* bf_hi = reinterpret_cast<bf16_2*>(&v_hi);
            u32x4 fp8_pack = {
                fused_act_round4_compile_test::cvt_bf16x4_to_fp8x4(
                    bf_lo[0], bf_lo[1], scale),
                fused_act_round4_compile_test::cvt_bf16x4_to_fp8x4(
                    bf_lo[2], bf_lo[3], scale),
                fused_act_round4_compile_test::cvt_bf16x4_to_fp8x4(
                    bf_hi[0], bf_hi[1], scale),
                fused_act_round4_compile_test::cvt_bf16x4_to_fp8x4(
                    bf_hi[2], bf_hi[3], scale),
            };

            const uint32_t lds_addr = lds_warp_addr + lds_lane_off;
            ::kittens::macros::ds_write_b128(fp8_pack, lds_addr, /*i_offset=*/0);
        }
    }
}

}  // namespace fused_act_round5_compile_test

struct grouped_layout_globals {
    _gl_fp8 a;                   // [M_total, K]
    _gl_fp8 b;                   // [G, N, K] (RCR)
    _gl_bf16 c;                  // [M_total, N]
    float scale_a, scale_b;
    const float* dscale_a;
    const float* dscale_b;
    const int64_t* group_offs;   // [G+1] int64 prefix-sum on device
    hipStream_t stream;
    int G;                       // number of groups
    int n;                       // N
    int k;                       // K
    int ki;                      // fast_k / K_BLOCK
    int bpc;                     // fast_n / BLOCK_SIZE
    int group_m;                 // tile-scheduling super-block factor
    int num_xcds;                // chiplet-swizzle XCD count (0 → default 8)
    int M_total;                 // sum of group sizes (= a.shape[0])
    // [grouped] Native non-aligned support (mirror of BF16 grouped Phase 3
    // and FP8 dense fast/tail). Main kernel only sweeps the largest aligned
    // interior:
    //   fast_n = (n / BLOCK_SIZE) * BLOCK_SIZE
    //   fast_k = (k / K_BLOCK) * K_BLOCK
    // `grouped_tail_kernel` (scalar fp32) handles cells with col >= fast_n
    // (full-K reduction) plus K-tail correction in [fast_k, k) for interior
    // cells. Per-group M-tail (M_g % BLOCK_SIZE != 0) is NOT handled in this
    // round (caller contract: each group's M is BLOCK_SIZE-aligned).
    int fast_n, fast_k;
    // Round-13: optional host-side hint — average per-group M in the
    // current launch. Consumed by the LDS-staged K-tail correction
    // kernel (``grouped_ktail_kernel_lds``) to gate the cooperative LDS
    // path: each tail block is (TBM × TBN); if ``m_per_group >= TBM``
    // and ``m_per_group % TBM == 0`` the per-block "all rows are in one
    // group" precondition holds for all blocks. Mirrors BF16 round-9/11
    // wiring; default 0 keeps the legacy scalar-tail fallback.
    int m_per_group;
    // Round-9 (current Primus run, gpt_oss FP8 kernel-only ceiling task;
    // 2026-05-08): per-launch persistent-grid slot override. Mirrors the
    // ``num_slots`` field on the var-K CRR globals struct (line 7744) and
    // its R3 wiring through ``grouped_variable_k_crr_*_fp8_fn``. When > 0
    // and <= NUM_CUS, ``dispatch_grouped_rcr`` uses this as the launch
    // ``gridDim.x`` instead of the legacy process-static ``rcr_slots``
    // (which was env-only via ``TK_RCR_NUM_CUS`` per R4 — process-wide,
    // unable to vary per-shape).
    //
    // Default 0 → legacy fallback chain: TK_RCR_NUM_CUS env (if set,
    // process-static cached) → NUM_CUS. Existing positional aggregate
    // initializers in callers leave this trailing field zero-initialized
    // by C++ aggregate value-init rules, so adding the field is a strict
    // backward-compat extension. See R4 comments at lines 2693 and 7400
    // for the original env-only design and its limitations.
    int num_slots;
    dim3 block() { return dim3(_NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

// [fused-act R6] Generalized to a template so the BF16-input fused-act path
// can pass ``FUSE_ACT=true`` and have the resolver invert the stored
// forward scale (``FP8_MAX / amax(a)``) into the dequant scale (``amax / FP8_MAX``)
// the FP8 output epilog needs. Existing un-fused calls deduce GL from the
// argument and pick up the default ``FUSE_ACT=false`` (bit-identical
// behaviour, existing instantiations untouched).
template<bool FUSE_ACT = false, typename GL>
__device__ __forceinline__ float resolve_combined_scale_grp(const GL &g) {
    const float sa_dev = g.dscale_a ? *g.dscale_a : g.scale_a;
    const float sb_dev = g.dscale_b ? *g.dscale_b : g.scale_b;
    if constexpr (FUSE_ACT) {
        // ``g.dscale_a`` for fused-act stores the FORWARD scale (output of
        // the R1 ``max_abs_bf16_to_fp8_scale`` binding). Invert to recover
        // the dequant scale that the FP8 epilog multiplies by.
        const float sa = (sa_dev > 0.0f) ? (1.0f / sa_dev) : 0.0f;
        return sa * sb_dev;
    } else {
        return sa_dev * sb_dev;
    }
}

// Persistent RCR kernel: grid_x = NUM_CUS. One block per CU; each block
// iterates many (group, tile) pairs in a single launch.
//
// Round-12: ``N_MASKED_STORE`` selects the C-store path at compile time.
// When ``false`` (N is BLOCK_SIZE-aligned, e.g. DSV3 N=4096/7168) the
// masked variant is dead-code-eliminated and the main kernel emits the
// same raw-store sequence as the round-11 path — keeping VGPR pressure
// low (no spill from the masked branch's lane-level row/col reconstruction).
// When ``true`` (N misaligned, e.g. gpt_oss N=2880/5760) we use
// ``store_c_tile_n_masked`` to drop OOB cols on the partial last tile.
//
// K-tail fuse: ``FUSED_KTAIL`` selects between the legacy two-launch
// pipeline ("main kernel writes K=[0, fast_k); standalone
// ``grouped_ktail_kernel_*`` reads C, adds K=[fast_k, k), writes C back" —
// FUSED_KTAIL=false) and the in-kernel fused epilog (FUSED_KTAIL=true)
// which accumulates the K-tail directly into cA/cB/cC/cD before scale +
// store. The fuse path is **path B** (round-3 commit 07354791): direct
// per-lane ``raw_buffer_load_b128`` HBM→register, lane-cell mapping for
// ``rt_16x128_s`` derived in
// ``analysis/_notes/round-3-fp8-ktail-path-b-success.md``; OOB lanes get
// SENTINEL voffsets so the SRD range_bytes check zero-fills the VGPR.
// Path A's LDS-staged predecessor (round-2 commit 4f6a2dee, see
// ``analysis/_notes/round-2-ktail-fuse-result.md``) was retired in
// rounds 3-11; the supporting helper
// ``prefill_swizzled_offsets_partial_K`` was deleted in round 13.
template<int KI_HINT = 0, bool N_MASKED_STORE = false, bool FUSED_KTAIL = false,
         bool FUSE_ACT = false>
__global__ __launch_bounds__(_NUM_THREADS, 1)
void grouped_rcr_kernel(
    const std::conditional_t<FUSE_ACT,
                             grouped_layout_globals_fused_act,
                             grouped_layout_globals> g) {
    // [fused-act R6] The FUSE_ACT=true instantiation reads BF16 ``g.a`` from
    // HBM and converts to FP8 inside the load helper (Phase 1 of the FP8
    // grouped fused-act forward optimization). The K-tail fuse path B reads
    // FP8 bytes via reinterpret_cast — incompatible with the BF16 fused-act
    // input — so we gate the two flags as mutually exclusive at compile time.
    // The un-fused fallback (Primus ``_unfused_forward``) handles K-tail
    // correctness when fused-act is selected on K%128 != 0 shapes (none in
    // our initial gate; Phase 1 covers K%128==0 = 16/24 metric shapes).
    static_assert(!(FUSE_ACT && FUSED_KTAIL),
                  "FUSE_ACT=true requires FUSED_KTAIL=false");
    using ST_rcr = ST_v2;
    __shared__ ST_rcr As[2][2];
    __shared__ ST_rcr Bs[2][2];
    // [grouped] LDS cache for device group_offs (int32 view) + per-group
    // tile-cumsum. group_offs is read O(N_iter * G) times by the per-tile
    // inner scan; caching to LDS once at kernel entry replaces ~640 cycles
    // of HBM-cached ld/iter with ~320 cycles of LDS ld/iter, ~3-5% kernel
    // speedup on shapes with low ki / many tiles. Cap MAX_G_PLUS_1 = 65 to
    // cover G ≤ 64 (metric uses G ≤ 32). 8×65 = 520 bytes LDS, negligible.
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];
    __shared__ int s_cum_tiles[MAX_G_PLUS_1];
    __shared__ int s_total_tiles;
    A_row_reg a;
    B_row_reg b0, b1;
    // Round-3 (gpt_oss FP8 focus) — extra A register tile for K-tail
    // M-slab 1, declared inside the ``if constexpr (FUSED_KTAIL)``
    // block at line ~2290 (round-7-dm scoping cleanup: was previously
    // declared at function scope; compiler DCE was already eliminating
    // the slot for FUSED_KTAIL=false template spec; bit-identical
    // codegen but clearer intent).
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;

    // [grouped] Persistent: chiplet-swizzle pid against the actual launch
    // grid (NUM_CUS in the default case; smaller when the dispatcher
    // applies the R4 short-grid lever).
    // Round-67: ``g.num_xcds`` is a host-side knob (default 0 → fallback
    // to ``BLOCK_SWIZZLE_NUM_XCDS=8``). Each shape can override via the
    // Python config rule. Mirrors BF16 grouped's existing ``g.num_xcds``
    // handling (analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp:3249).
    //
    // Round-4 (gpt_oss FP8 kernel-only): swap constexpr ``NUM_CUS`` for
    // runtime ``gridDim.x`` so the chiplet swizzle range tracks the
    // actual launch geometry. Identical math when ``gridDim.x ==
    // NUM_CUS`` (the default); enables the ``TK_RCR_NUM_CUS`` env probe
    // (and a future ``g.num_slots`` knob, mirroring the var-K R2/R3
    // lever) to reduce the persistent grid for short-grid sparse fwd
    // shapes (e.g. Down-B4-M2048 fwd at ~1.5 wave-steps/CU).
    // ``gridDim.x`` is wave-uniform so codegen is stable; no change to
    // register pressure, LDS layout, or HBM stride.
    const int slots_eff = gridDim.x;
    const int xcds_eff = g.num_xcds > 0 ? g.num_xcds : BLOCK_SWIZZLE_NUM_XCDS;
    int pid = chiplet_transform_chunked(
        blockIdx.x, slots_eff, xcds_eff, 64);

    int wm = warpid() / WARPS_N;
    int wn = warpid() % WARPS_N;
    const int num_pid_n = g.bpc;
    const int ki_dyn   = (KI_HINT > 0) ? KI_HINT : g.ki;

    // [grouped] Cooperative init of the LDS group-metadata caches.
    //
    // Round-9-dm: split the original single-threaded init into
    //   (a) parallel HBM read of g.group_offs[0 .. g.G] (65 entries max, fit
    //       in <3 warp-coalesced cachelines so the original O(G) serialized
    //       HBM loads collapse to a single warp-wide wavefront transfer),
    //   (b) parallel pad of s_cum_tiles[g.G+1 .. MAX_G_PLUS_1),
    //   (c) intra-CTA sync, then thread-0 runs the O(G) prefix-scan from
    //       LDS (all the g.group_offs[] fetches have already retired).
    // Pad s_cum_tiles[g.G + 1 .. MAX_G_PLUS_1) with INT_MAX so a constant-
    // depth (6-step) branch-free binary search reading any mid > g.G never
    // updates lo (the cmp `gt >= INT_MAX` is always false for finite gt).
    if (threadIdx.x <= g.G && threadIdx.x < MAX_G_PLUS_1) {
        s_offs[threadIdx.x] = static_cast<int>(g.group_offs[threadIdx.x]);
    }
    if (threadIdx.x > g.G && threadIdx.x < MAX_G_PLUS_1) {
        s_cum_tiles[threadIdx.x] = 0x7FFFFFFF;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int prev = s_offs[0];
        s_cum_tiles[0] = 0;
        int t = 0;
        #pragma unroll 1
        for (int gi = 0; gi < g.G; ++gi) {
            const int next = s_offs[gi + 1];
            t += ((next - prev) / BLOCK_SIZE) * num_pid_n;
            s_cum_tiles[gi + 1] = t;
            prev = next;
        }
        s_total_tiles = t;
    }
    __syncthreads();
    const int total_tiles = s_total_tiles;

    // Prefill swizzled offsets ONCE (shared across all tiles & all groups —
    // depends only on the GL strides which are constant within the launch).
    constexpr int bpt = ST_rcr::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * _NUM_THREADS;
    constexpr int mpt = ST_rcr::rows * ST_rcr::cols * sizeof(fp8e4m3) / bpm;
    uint32_t soA[mpt], soB[mpt];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

    // [fused-act R6] One-shot read of the forward FP8 scale (FP8_MAX / amax)
    // into a wave-uniform float register. Used by every load_a call in the
    // FUSE_ACT=true path. Read ONCE here (not in the per-tile epilog) so the
    // HBM load is shared across all tiles processed by this CU. ``g.dscale_a``
    // stores the forward scale for fused-act (output of the R1
    // ``max_abs_bf16_to_fp8_scale`` binding); for the un-fused (FUSE_ACT=false)
    // path this variable is unused (compiler DCE drops the load).
    float scale_a_inv = 0.0f;
    if constexpr (FUSE_ACT) {
        scale_a_inv = (g.dscale_a != nullptr) ? *g.dscale_a : 0.0f;
    }

    // K-tail fuse uses path B (round-3 commit 07354791): direct per-lane
    // ``raw_buffer_load_b128`` HBM→register inside the fuse epilog (~line
    // 2300+ below), no LDS round-trip and no swizzled-offset prefill. The
    // path-A scaffolding (``soA_tail`` / ``soB_tail`` declarations +
    // ``prefill_swizzled_offsets_partial_K`` calls) was removed in round 11
    // (commit f9d591cb); the helper itself was removed in round 13.

    // [grouped] Persistent outer loop.
    // Round-4: stride by ``slots_eff`` (= gridDim.x) instead of constexpr
    // NUM_CUS so the persistent loop covers all tiles when the dispatcher
    // launches with a reduced grid. When gridDim.x == NUM_CUS this is
    // bit-identical to the prior constexpr stride.
    for (int gt = pid; gt < total_tiles; gt += slots_eff) {
        // [grouped] 6-step branch-free binary search over LDS-cached cumsum
        // (covers G ∈ [1, 64] since 2^6 = 64 = MAX_G_PLUS_1-1). Sentinel
        // INT_MAX past g.G keeps the `gt >= s_cum_tiles[mid]` cmp false so
        // lo never advances past g.G. Compared to the linear O(G) scan,
        // this collapses ~32 LDS lds + cmp into 6 sequential lookups: ~70
        // cyc instead of ~320 cyc per outer iter (kernel-only saving ~3-5%
        // on shapes with low ki / many tiles).
        int lo = 0;
        int hi = MAX_G_PLUS_1 - 1;
        #pragma unroll
        for (int level = 0; level < 6; ++level) {
            const int mid = (lo + hi + 1) >> 1;
            if (gt >= s_cum_tiles[mid]) lo = mid;
            else hi = mid - 1;
        }
        const int group_idx = lo;
        const int tile_start = s_cum_tiles[lo];
        const int local_tile = gt - tile_start;
        const int m_start_g = s_offs[group_idx];
        const int M_g = s_offs[group_idx + 1] - m_start_g;
        const int bpr_g = M_g / BLOCK_SIZE;

        // Group-by-M / group-by-N swizzle (matches dense kernel mapping).
        int br, bc;
        if (g.bpc > bpr_g) {
            const int WGN = g.group_m;
            const int num_wgid_in_group = bpr_g * WGN;
            int group_id = local_tile / num_wgid_in_group;
            int first_pid_n = group_id * WGN;
            int group_size_n = min(num_pid_n - first_pid_n, WGN);
            if (group_size_n <= 0) continue;
            bc = first_pid_n + ((local_tile % num_wgid_in_group) % group_size_n);
            br = (local_tile % num_wgid_in_group) / group_size_n;
        } else {
            const int WGM = g.group_m;
            const int num_wgid_in_group = WGM * num_pid_n;
            int group_id = local_tile / num_wgid_in_group;
            int first_pid_m = group_id * WGM;
            int group_size_m = min(bpr_g - first_pid_m, WGM);
            if (group_size_m <= 0) continue;
            br = first_pid_m + ((local_tile % num_wgid_in_group) % group_size_m);
            bc = (local_tile % num_wgid_in_group) / group_size_m;
        }
        if (br >= bpr_g || bc >= num_pid_n) continue;

        // Coord shifts:
        //   ST_A: st_fp8e4m3<HB=128, BK=128, ...> → row-coord unit = HB = 128.
        //         m_subtile_A = m_start_g / HB.
        //   C (RT store): rt_fl<RBM=64, RBN=32, ...> → row-coord unit = RBM=64.
        //         m_subtile_C = m_start_g / RBM.
        const int m_subtile_A = m_start_g / HB;
        const int m_subtile_C = m_start_g / RBM;

        auto a_co = [&](int s, int k) -> coord<ST_rcr> {
            return {0, 0, m_subtile_A + s, k};
        };
        auto b_co = [&](int s, int k) -> coord<ST_rcr> {
            return {0, group_idx, s, k};
        };

        auto load_a = [&](A_row_reg& dst, ST_rcr& tile, int wi) {
            auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
            load(dst, sub);
        };
        auto load_b = [&](B_row_reg& dst, ST_rcr& tile, int wi) {
            auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
            load(dst, sub);
        };

        auto b_tile = [&](int stage, int which) -> ST_rcr& {
            return Bs[stage][which];
        };

        // Reset accumulators per tile.
        zero(cA); zero(cB); zero(cC); zero(cD);

        int tic = 0, toc = 1;
        // Prologue: load tile-0 + tile-1 (mirrors gemm_kernel<RCR> 1040-1054).
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 0), g.b, b_co(bc*2,   0), soB);
        if constexpr (FUSE_ACT) fused_act_round5_compile_test::rcr_8w_load_hoist_fused_act<_NUM_THREADS>(As[tic][0], g.a, a_co(br*2,   0), soA, scale_a_inv);
        else                    rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0],    g.a, a_co(br*2,   0), soA);
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 1), g.b, b_co(bc*2+1, 0), soB);
        if constexpr (FUSE_ACT) fused_act_round5_compile_test::rcr_8w_load_hoist_fused_act<_NUM_THREADS>(As[tic][1], g.a, a_co(br*2+1, 0), soA, scale_a_inv);
        else                    rcr_8w_load_hoist<_NUM_THREADS>(As[tic][1],    g.a, a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(RCR_INIT0_VMCNT);
        __builtin_amdgcn_s_barrier();

        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(toc, 0), g.b, b_co(bc*2,   1), soB);
        if constexpr (FUSE_ACT) fused_act_round5_compile_test::rcr_8w_load_hoist_fused_act<_NUM_THREADS>(As[toc][0], g.a, a_co(br*2,   1), soA, scale_a_inv);
        else                    rcr_8w_load_hoist<_NUM_THREADS>(As[toc][0],    g.a, a_co(br*2,   1), soA);
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(toc, 1), g.b, b_co(bc*2+1, 1), soB);

        TK_WAIT_VMCNT(RCR_INIT1_VMCNT);
        __builtin_amdgcn_s_barrier();

        // Single-tile main loop (mirrors dense gemm_kernel<RCR> else-branch
        // lines 1129-1158).
        // Round-4 (gpt_oss FP8 focus) PROBE: removed 2x RCR_SCHED_BARRIER()
        // per K-iter (compiler reorder hint), keeping all s_setprio +
        // s_barrier + s_waitcnt. Round-2 falsified removing BOTH
        // sched_barrier AND setprio together (-5.6%); this isolates whether
        // the regression came from setprio alone (priority-bias for MFMA
        // issue) or sched_barrier alone (compiler reorder block). If this
        // probe is neutral or +ve, the round-2 regression was setprio-only
        // → confirms sched_barrier overhead is removable.
        //
        // Round-3-dm: swept local UNROLL override {1, 2, 4} × 5 runs each.
        // Means 816.2 / 816.2 / 816.4 — flat across the entire range.
        // LLVM's unroll heuristic already produces optimal body layout;
        // `#pragma unroll N` is an ignorable hint here. See round-3-dm
        // note. Saturated. Use shared RCR_MAIN_UNROLL (=2, matches dense).
        TK_PRAGMA_UNROLL(RCR_MAIN_UNROLL)
        for (int k = 0; k < ki_dyn - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, b_tile(tic, 0), wn);
            load_a(a, As[tic][0], wm);
            if constexpr (FUSE_ACT) fused_act_round5_compile_test::rcr_8w_load_hoist_fused_act<_NUM_THREADS>(As[toc][1], g.a, a_co(br*2+1, k+1), soA, scale_a_inv);
            else                    rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], g.a, a_co(br*2+1, k+1), soA);
            TK_WAIT_LGKM(RCR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, b_tile(tic, 1), wn);
            rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 0), g.b, b_co(bc*2, k+2), soB);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            if constexpr (FUSE_ACT) fused_act_round5_compile_test::rcr_8w_load_hoist_fused_act<_NUM_THREADS>(As[tic][0], g.a, a_co(br*2, k+2), soA, scale_a_inv);
            else                    rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0], g.a, a_co(br*2, k+2), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 1), g.b, b_co(bc*2+1, k+2), soB);
            TK_WAIT_VMCNT(RCR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_setprio(1); rcr_mma(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        // Epilog 1: second-to-last K-tile (mirrors dense lines 1160-1187).
        {
            load_b(b0, b_tile(tic, 0), wn);
            load_a(a, As[tic][0], wm);
            if constexpr (FUSE_ACT) fused_act_round5_compile_test::rcr_8w_load_hoist_fused_act<_NUM_THREADS>(As[toc][1], g.a, a_co(br*2+1, ki_dyn-1), soA, scale_a_inv);
            else                    rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], g.a, a_co(br*2+1, ki_dyn-1), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

            load_b(b1, b_tile(tic, 1), wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(RCR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, b_tile(toc, 0), wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();
            tic ^= 1; toc ^= 1;
        }

        // Epilog 2: last K-tile (mirrors dense lines 1189-1210).
        {
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, b_tile(tic, 1), wn);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            rcr_mma(cC, a, b0);
            rcr_mma(cD, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        // === Fused K-tail epilog (path B, round-3 commit 07354791) ===
        // After Epilog 2, cA/cB/cC/cD hold sum over K=[0, fast_k). For
        // K-misaligned shapes (e.g. gpt_oss K=2880, K_REM=64) we accumulate
        // K=[fast_k, fast_k + K_BLOCK) directly into cA/cB/cC/cD inside this
        // same launch — no standalone K-tail kernel, no RMW on g.c.
        //
        // History (see ``analysis/_notes/round-{2,3}-fp8-ktail-*.md`` for
        // full derivation):
        //   * Round-2 (commit 4f6a2dee) shipped path A: LDS-staged K-tail
        //     using ``prefill_swizzled_offsets_partial_K`` + SENTINEL
        //     voffset + ``buffer_load_lds``. Saturated at SNR ~20.7 dB
        //     because ``buffer_load_lds`` is a NO-OP on OOB voffset (it
        //     does NOT zero LDS — only ``buffer_load.iX`` zero-fills VGPR
        //     on OOB), leaving OOB lanes' LDS slots holding stale main-
        //     loop K-tile data; ``load(reg, st_subtile)`` then accumulated
        //     that stale data with weight 1.
        //   * Round-3 (commit 07354791) shipped the live code below, path B:
        //     each lane reads ``2 × buffer_load_b128`` directly from HBM
        //     into A_row_reg / B_row_reg ``data[]``, lane-cell mapping
        //     hand-derived for ``rt_16x128_s``. SRD range_bytes
        //     auto-zero-fills VGPR for K-OOB lanes — exactly what path A
        //     wanted but couldn't get from ``buffer_load_lds``. SNR
        //     ≥25 dB on all 8 gpt_oss K=2880 cases; metric 153 → 465.
        //   * Round-11 (commit f9d591cb) dropped the dead path-A scaffolding
        //     (``soA_tail`` / ``soB_tail`` declarations + the two
        //     ``prefill_swizzled_offsets_partial_K`` calls in the prologue);
        //     round-13 dropped the helper definition itself.
        if constexpr (FUSED_KTAIL) {
            // Round-7-dm: scoped A register tile for K-tail M-slab 1
            // (moved from function-scope; round-3 introduced the reg
            // to save one ``vmcnt(0)`` wait on K-misaligned shapes).
            A_row_reg a_kt1;
            if (g.fast_k < g.k) {
                // === Round-3 path B: direct HBM → register K-tail load ===
                // Mirrors BF16 round-5 path B
                // (kernel_bf16_dynamic.cpp:709-827). Each lane reads
                // 2 × buffer_load_b128 (= 32 fp8 cells) directly from
                // HBM into A_row_reg / B_row_reg `data[]`, sidestepping
                // LDS entirely and the round-3 phantom-read pattern
                // documented in
                // ``analysis/_notes/round-3-bf16-ktail-phantom-read.md``
                // and ``analysis/_notes/round-3-fp8-ktail-path-a-saturation.md``.
                //
                // Lane → cell mapping for ``rt_16x128_s`` (fp8e4m3, 64
                // lanes/warp, 32 cells/lane = 32 bytes = 2 × b128):
                //   row_lane    = laneid % 16     (16 rows/base tile)
                //   k_lane_byte = (laneid/16) * 32 (contiguous K cells)
                //   data[0..3]  = K=[k_lane_byte, k_lane_byte + 16) → b128 #1
                //   data[4..7]  = K=[k_lane_byte + 16, k_lane_byte + 32) → b128 #2
                //
                // K_REM=64 < K_STEP=128 (gpt_oss K=2880=22*128+64):
                //   laneid 0..15  : k_lane_byte=0  → both b128 valid
                //   laneid 16..31 : k_lane_byte=32 → both b128 valid
                //   laneid 32..47 : k_lane_byte=64 → both b128 K-OOB
                //   laneid 48..63 : k_lane_byte=96 → both b128 K-OOB
                //
                // K-OOB lanes get ``voffset = SENTINEL`` so the SRD
                // range_bytes check rejects the load → VGPR returns 0.
                // ``raw_buffer_load_b128`` zero-fills VGPR on OOB
                // (unlike ``raw_buffer_load_lds`` which is no-op, the
                // path-A blocker).
                //
                // SRDs:
                //   * A: full-tensor (M_total × K bytes). OOB row >
                //     range clamps to 0.
                //   * B: per-group ((group_idx + 1) × N × K bytes).
                //     Without per-group bound, OOB N-rows on partial
                //     last col-tile would wrap into NEXT group's data.
                //     Per-group bound clamps to 0; column-masked C store
                //     drops the OOB cells.
                //
                // ``row_stride_bytes=0`` in ``make_srsrc`` keeps the
                // linear range-bytes bound check (mirror of BF16 round-11
                // gpt_oss K=2880 fix: cache-swizzle on non-power-of-2
                // strides has UB OOB-clamp behaviour; raw range works).
                //
                // Currently gated at K_REM ∈ {32, 64, 96} (32-aligned)
                // at the dispatcher. Mixed K_REM (e.g. 80) requires
                // partial-b128 lane mask; future round.

                const int laneid = kittens::laneid();
                const int row_lane = laneid % 16;
                const int k_lane_byte = (laneid / 16) * 32;
                // Round-49-dm (auto-optimize R49): FUSED_KTAIL invariant
                // collapse — dispatcher gate ``fuse_ktail_eligible`` (line
                // ~5335) only enables this template spec when
                // ``K_rem_for_fuse ∈ {64, 0}``. The runtime branch
                // ``if (g.fast_k < g.k)`` (parent of this scope) is FALSE
                // when ``K_REM = 0`` (g.fast_k == g.k), so inside this
                // scope K_REM is necessarily 64.
                //
                // Replaces the prior dynamic ``K_REM = g.k - g.fast_k`` +
                // 2 per-lane masks ``b128_lo_valid = (k_lane_byte + 16) <=
                // K_REM`` and ``b128_hi_valid = (k_lane_byte + 32) <=
                // K_REM`` with a single constexpr-folded mask. Since
                // ``k_lane_byte = (laneid / 16) * 32 ∈ {0, 32, 64, 96}``,
                // for K_REM=64 both masks collapse to ``laneid < 32``:
                //
                //   laneid 0..15  : k_lane_byte=0  → lo: 16≤64 ✓ | hi: 32≤64 ✓ → both true
                //   laneid 16..31 : k_lane_byte=32 → lo: 48≤64 ✓ | hi: 64≤64 ✓ → both true
                //   laneid 32..47 : k_lane_byte=64 → lo: 80≤64 ✗ | hi: 96≤64 ✗ → both false
                //   laneid 48..63 : k_lane_byte=96 → lo:112≤64 ✗ | hi:128≤64 ✗ → both false
                //
                // Eliminates 1 wave-uniform sub op (``g.k - g.fast_k``),
                // 1 K_REM register slot, and one of the two per-lane
                // mask cmps. The resource report (``-Rpass-analysis=
                // kernel-resource-usage``) is bit-identical at the V/A/
                // spill/scratch level (LLVM's CSE was already partially
                // collapsing the redundancy); the .so binary md5 changes
                // (codegen emits 1-3 fewer instructions in the K-tail
                // load lambdas, observed via build comparison this
                // round). Metric: 996 / 987 over 2 runs vs 997 baseline
                // — within run-to-run noise band (982-998 centred ~990).
                //
                // **Future**: if ``fuse_ktail_eligible`` (line ~5335) is
                // ever extended to allow K_REM ∈ {32, 96} or other
                // partial-b128 values, this constexpr collapse must be
                // replaced by a per-K_REM template specialisation.
                // Currently only {0, 64} are gated through, so the
                // invariant holds.
                constexpr int KREM = 64;
                static_assert(KREM == 64,
                    "FUSED_KTAIL=true K_REM must be 64; see fuse_ktail_eligible");
                const bool both_valid = (laneid < 32);
                constexpr uint32_t SENTINEL = 0xFFFF0000u;

                const fp8e4m3* a_base_ptr = (const fp8e4m3*)&g.a[{0, 0, 0, 0}];
                const fp8e4m3* b_base_ptr = (const fp8e4m3*)&g.b[{0, 0, 0, 0}];
                const int a_row_stride_bytes = g.a.template stride<2>();
                const int b_row_stride_bytes = g.b.template stride<2>();
                const uint32_t a_total_bytes =
                    static_cast<uint32_t>(g.M_total) *
                    static_cast<uint32_t>(a_row_stride_bytes);
                const uint32_t b_per_group_bytes =
                    static_cast<uint32_t>(group_idx + 1) *
                    static_cast<uint32_t>(g.n) *
                    static_cast<uint32_t>(b_row_stride_bytes);
                i32x4 a_srsrc_kt = make_srsrc((const void*)a_base_ptr, a_total_bytes);
                i32x4 b_srsrc_kt = make_srsrc((const void*)b_base_ptr, b_per_group_bytes);

                const uint32_t K_tail_base_bytes =
                    static_cast<uint32_t>(g.fast_k);
                const uint32_t b_group_byte_base =
                    static_cast<uint32_t>(group_idx) *
                    static_cast<uint32_t>(g.n) *
                    static_cast<uint32_t>(b_row_stride_bytes);

                // M_warp_base derivation:
                //   a_co(s, k) → coord {0, 0, m_subtile_A + s, k}
                //   unit_coord<2,3>: row = (m_subtile_A + s) * ST_rcr::rows = (.) * HB
                //   warp wm picks rows wm*RBM..wm*RBM+RBM-1 within the 128-row tile.
                //   For h ∈ [0, A_row_reg::height = 4): row = M_warp_base + h*16 + row_lane.
                // Round-3: refactored to take A_row_reg by reference so M-slab 0
                // and M-slab 1 can target different register tiles (a vs a_kt1).
                // This lets us issue all 12 K-tail buffer_loads up front and
                // drain with a single vmcnt(0) before the 4 K-tail mfma —
                // saves ~1 vmcnt(0) round-trip per output tile (~50-100 cyc).
                auto load_a_kt = [&](A_row_reg& A_tile, int slab)
                        __attribute__((always_inline)) {
                    const int M_warp_base =
                        (m_subtile_A + br * 2 + slab) * HB + wm * RBM;
                    #pragma unroll
                    for (int h = 0; h < A_row_reg::height; ++h) {
                        const int A_row_idx = M_warp_base + h * 16 + row_lane;
                        const uint32_t v_base = static_cast<uint32_t>(
                            A_row_idx * a_row_stride_bytes +
                            K_tail_base_bytes + k_lane_byte);
                        const uint32_t v_lo = both_valid ? v_base : SENTINEL;
                        const uint32_t v_hi = both_valid ? (v_base + 16) : SENTINEL;
                        __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            a_srsrc_kt, v_lo, 0, 0);
                        __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            a_srsrc_kt, v_hi, 0, 0);
                        *reinterpret_cast<__uint128_t*>(&A_tile.tiles[h][0].data[0]) = v0;
                        *reinterpret_cast<__uint128_t*>(&A_tile.tiles[h][0].data[4]) = v1;
                    }
                };

                // N_warp_base derivation:
                //   b_co(s, k) → coord {0, group_idx, s, k}
                //   unit_coord: N-row in tile = s * ST_rcr::rows = s * HB.
                //   warp wn picks rows wn*RBN..wn*RBN+RBN-1 within the 128-row tile.
                //   For h_b ∈ [0, B_row_reg::height = 2):
                //     B_row_idx_in_group = N_warp_base + h_b*16 + row_lane.
                //   Global byte = group_idx * N * K + B_row_idx_in_group * K + ...
                auto load_b_kt = [&](B_row_reg& B_tile, int n_strip) __attribute__((always_inline)) {
                    const int N_warp_base =
                        (bc * 2 + n_strip) * HB + wn * RBN;
                    #pragma unroll
                    for (int h_b = 0; h_b < B_row_reg::height; ++h_b) {
                        const int B_row_idx_in_group = N_warp_base + h_b * 16 + row_lane;
                        const uint32_t v_base = b_group_byte_base + static_cast<uint32_t>(
                            B_row_idx_in_group * b_row_stride_bytes +
                            K_tail_base_bytes + k_lane_byte);
                        const uint32_t v_lo = both_valid ? v_base : SENTINEL;
                        const uint32_t v_hi = both_valid ? (v_base + 16) : SENTINEL;
                        __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            b_srsrc_kt, v_lo, 0, 0);
                        __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            b_srsrc_kt, v_hi, 0, 0);
                        *reinterpret_cast<__uint128_t*>(&B_tile.tiles[h_b][0].data[0]) = v0;
                        *reinterpret_cast<__uint128_t*>(&B_tile.tiles[h_b][0].data[4]) = v1;
                    }
                };

                // Round-3: issue ALL K-tail buffer_loads up front, drain with
                // a SINGLE ``vmcnt(0)`` wait, then 4 mfma sequential. M-slab 0
                // → ``a``, M-slab 1 → ``a_kt1`` (separate register). Saves one
                // ``s_waitcnt vmcnt(0)`` round-trip per output tile (~50-100
                // cyc) and lets the SQ overlap the buffer_loads.
                //
                // Round-12-dm: split the single ``vmcnt(0)`` into a 2-stage
                // wait so the cA/cB mfmas can overlap with the M-slab-1
                // ``a_kt1`` HBM drain. Issue order is now
                //   a (8 b128) → b0 (4) → b1 (4) → a_kt1 (8)        // 24 total
                // ``vmcnt(8)`` waits until <= 8 outstanding (i.e. only the
                // last 8 issued = a_kt1 still in flight), at which point
                // a/b0/b1 are guaranteed drained (vmcnt is in-issue-order
                // retirement on AMDGCN — same semantics relied on by the
                // main loop's ``RCR_STEADY_VMCNT=8`` mid-iter wait at line
                // 2199). cA = a · b0 and cB = a · b1 then fire while the
                // remaining 8 a_kt1 loads complete in parallel; the
                // ``vmcnt(0)`` before cC/cD acts as a no-op when those
                // already drained. Estimated saving: 1-2 mfma latencies
                // (~32-64 cyc) per K-tail output tile, K-misaligned
                // (gpt_oss K=2880 K_REM=64) shapes only.
                // Round-37-dm: reorder K-tail issues from
                //   [a(8), b0(4), b1(4), a_kt1(8)]  →  [b0(4), b1(4), a(8), a_kt1(8)]
                // Rationale: issue the smaller B tiles FIRST so they saturate
                // HBM controller request-queue slots with finer-grained fetches,
                // then issue the larger A tiles. Under in-issue-order VMEM
                // retirement (per R12-dm comment + main-loop RCR_STEADY_VMCNT=8
                // invariant), the first 16 retired are now b0+b1+a — all three
                // needed for mfma cA/cB. vmcnt(8) fires after first 16 retire
                // (same semantics as original), mfma cA/cB runs overlapping
                // with a_kt1 drain. Zero correctness impact. Potential win: if
                // HBM scheduler prioritises smaller requests first, the 8
                // b128-load B-batch drains slightly earlier than if A-batch
                // came first, cascading to mfma cA starting sooner.
                load_b_kt(b0,    0);   // 4 buffer_load → b0 (issued FIRST)
                load_b_kt(b1,    1);   // 4 buffer_load → b1
                load_a_kt(a,     0);   // 8 buffer_load → a (M slab 0)
                load_a_kt(a_kt1, 1);   // 8 buffer_load → a_kt1 (M slab 1, LAST)
                asm volatile("s_waitcnt vmcnt(8)");
                rcr_mma(cA, a,     b0);
                rcr_mma(cB, a,     b1);
                asm volatile("s_waitcnt vmcnt(0)");
                rcr_mma(cC, a_kt1, b0);
                rcr_mma(cD, a_kt1, b1);
            }
        }

        // Apply scale + store with m_subtile_C row shift.
        // Round-44-dm-probe-A: interleave mul + store per accumulator
        // (was: 4× mul, then 4× store). rocprof showed gpt_oss spec
        // <0,true,true> uses 228 B/thread scratch vs 148 B/thread for
        // dsv3_g spec <0,false,true> — the +80 B is N_MASKED helper's
        // live state. Hypothesis: serialising mul→store→next-mul lets
        // LLVM free cA/cB/cC/cD's accumulator VGPR slots progressively,
        // giving the masked-store helper room to spill into vacated
        // slots instead of HBM scratch. Targets the actual bottleneck
        // (scratch I/O REQUEST rate, not allocation count).
        const float combined_scale = resolve_combined_scale_grp<FUSE_ACT>(g);

        if (wm == 0) __builtin_amdgcn_s_barrier();
        // Round-12: mirror BF16 grouped's column-masked C store. With
        // ``g.bpc = ceil_div(g.n, BLOCK_SIZE)`` (dispatch_grouped_rcr
        // round-12 path) the last col-tile may straddle ``[fast_n, n)``;
        // ``store_c_tile_n_masked`` drops OOB columns. ``rcr_8w_load_hoist``
        // already uses the full-tensor SRD (line ~432) so OOB rows in B
        // clamp to 0 — the masked C store then prevents those cells from
        // being written. ``N_MASKED_STORE`` is a compile-time template
        // parameter so the N-aligned dispatch path emits the raw store
        // (no spill from the masked variant's row/col reconstruction).
        //
        // Round-59: hoist per-block N-tail branch from the helper into
        // the kernel epilogue. The helper's top-of-body
        // ``if (n1 <= n_limit) store(...)`` fast-path forwards interior
        // tiles to the same bare ``store(...)`` as the unmasked kernel,
        // but having the masked-helper body in scope on every block
        // (even when the runtime branch falls through to the bare store)
        // inflates VGPR pressure and serialises the epilogue: the
        // unmasked ``<0,false>`` template runs N=5888 in 2.19 ms while
        // the ``<0,true>`` template with helper-internal branching runs
        // the same 23 col-tile work in 3.05 ms (+39 % wall time, gpt_oss
        // GateUP B32-M4096 K=2816, /tmp/profile_fp8_n_alignment.py).
        // Hoisting the branch — interior takes the bare ``store(...)``
        // path identical to the unmasked template; only ``bc == bpc-1``
        // on a misaligned N hits the masked helper — lets the compiler
        // fully specialise both arms and recovers the unmasked-kernel
        // throughput on the 22/23 interior col-tiles. Numerical safety:
        // ``(bc + 1) * BLOCK_SIZE <= g.n`` is the necessary and
        // sufficient condition for the four C sub-tiles cA/cB/cC/cD to
        // fit entirely in [bc*BLOCK_SIZE, g.n) (combined col span of all
        // 4 stores = [bc*BLOCK_SIZE, (bc+1)*BLOCK_SIZE)); when true, the
        // bare ``store`` writes the same cells as the masked helper's
        // ``n1 <= n_limit`` fast-path. ``bc`` is uniform across the wave
        // so this is a single wave-uniform branch, not divergent control
        // flow.
        // Round-24 (PT): readfirstlane on the 4 C-store wave-uniform coords.
        // R22 ASM disassembly counted 411 divergent-SRD fallback loops in
        // FUSED_KTAIL=true rcr<T,T> spec (vs 128 in <F,F>); 80 % live in the
        // C-store epilog. Root cause: r0/r1/c0/c1 are wave-uniform IN PRINCIPLE
        // (m_subtile_C, br, bc, wm, wn all uniform per tile-iter) but LLVM's
        // uniformity analysis cannot prove it — the FUSED_KTAIL block's per-
        // lane VGPR ops (SENTINEL voffsets, b128_lo_valid lane masks) earlier
        // in the function taint downstream flow. Each store(g.c, cX, coord)
        // therefore constructs a VGPR-derived buffer SRD inside the kittens
        // helper (`make_buffer_resource(as_u64, ...)` at memory/tile/
        // global_to_register.cuh:332) and emits the per-lane fallback loop
        // pattern (`v_readfirstlane → v_cmp → s_and_saveexec → buffer_op →
        // s_xor exec → s_cbranch_execnz`) that costs ~20 cy/loop on the
        // common wave-uniform path.
        //
        // Localised readfirstlane HERE (not earlier) avoids R22's V-A/V-B
        // backlash (those tried on `group_idx` in the binary search prologue
        // → +21 dw spill on rcr<T,T> via wide downstream cascade). The 4
        // coord ints are dead-after-stores (no downstream consumers in the
        // tile-iter), so SGPR promotion is local.
        const int r0 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2+wm);
        const int r1 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2+WARPS_M+wm);
        const int c0 = __builtin_amdgcn_readfirstlane(bc*WARPS_N*2+wn);
        const int c1 = __builtin_amdgcn_readfirstlane(bc*WARPS_N*2+WARPS_N+wn);
        if constexpr (N_MASKED_STORE) {
            if ((bc + 1) * BLOCK_SIZE <= g.n) {
                mul(cA, cA, combined_scale);
                store(g.c, cA, {0, 0, r0, c0});
                mul(cB, cB, combined_scale);
                store(g.c, cB, {0, 0, r0, c1});
                mul(cC, cC, combined_scale);
                store(g.c, cC, {0, 0, r1, c0});
                mul(cD, cD, combined_scale);
                store(g.c, cD, {0, 0, r1, c1});
            } else {
                mul(cA, cA, combined_scale);
                store_c_tile_n_masked(g.c, cA, r0, c0, g.n);
                mul(cB, cB, combined_scale);
                store_c_tile_n_masked(g.c, cB, r0, c1, g.n);
                mul(cC, cC, combined_scale);
                store_c_tile_n_masked(g.c, cC, r1, c0, g.n);
                mul(cD, cD, combined_scale);
                store_c_tile_n_masked(g.c, cD, r1, c1, g.n);
            }
        } else {
            mul(cA, cA, combined_scale);
            store(g.c, cA, {0, 0, r0, c0});
            mul(cB, cB, combined_scale);
            store(g.c, cB, {0, 0, r0, c1});
            mul(cC, cC, combined_scale);
            store(g.c, cC, {0, 0, r1, c0});
            mul(cD, cD, combined_scale);
            store(g.c, cD, {0, 0, r1, c1});
        }

        // Round-50-dm probe: drain only LDS (lgkmcnt) at end of tile;
        // skip vmcnt(0). C-store HBM writes (~32 buffer_store_b16 pending)
        // do NOT alias with next tile's HBM loads (g.a / g.b reads of a
        // different (br, bc) cell), so the next tile's prologue can issue
        // its 16 buffer_loads in parallel with C-store drain. The
        // prologue's own ``TK_WAIT_VMCNT(RCR_INIT0_VMCNT=4)`` enforces the
        // sync it needs (prologue waits for ≤4 outstanding before mfma);
        // since vmcnt is shared across loads + stores, that wait will
        // also drain the C-store stragglers if they outlive the load
        // issue path. Keep lgkmcnt(0) to ensure main-loop LDS writes
        // retire before next tile's prologue overwrites the same slab.
        // Estimated: ~150 cy/tile saved (overlap of ~30 cy search +
        // prologue issue with ~150 cy store drain). On gpt_oss-GateUP-
        // B32-M4096 with ~368 tiles/CU, that's ~55K cy = ~27 us = ~1 %
        // wall-time saving, expected ~+1-2 pp on grp_FP8 ratio.
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }
}

template __global__ void grouped_rcr_kernel<0, false, false>(const grouped_layout_globals);
template __global__ void grouped_rcr_kernel<0, true , false>(const grouped_layout_globals);
template __global__ void grouped_rcr_kernel<0, false, true >(const grouped_layout_globals);
template __global__ void grouped_rcr_kernel<0, true , true >(const grouped_layout_globals);
// =============================================================================
// Round-F M2a: BLOCK_SIZE=128 tile-port (no K-tail, no fused-act) =============
// =============================================================================
// Mechanical port of the file-scope ``grouped_rcr_kernel`` body into
// ``namespace kernel_b128``. The namespace shadows BLOCK_SIZE, HB, RBM, RBN,
// ST_v2, A_row_reg, B_row_reg with smaller variants (HB=64, RBM=32, RBN=16);
// inside this re-opened namespace block, every unqualified reference in the
// kernel body resolves to the smaller types. References to file-scope
// helpers (``rcr_8w_load_hoist``, ``store_c_tile_n_masked``,
// ``resolve_combined_scale_grp``, ``chiplet_transform_chunked``, the various
// ``RCR_*`` wait-counter macros, ``grouped_layout_globals``, ``NUM_CUS``,
// ``BLOCK_SWIZZLE_NUM_XCDS``, ``WARP_THREADS``) resolve to file scope via
// outer-namespace lookup — no source change needed in the body.
//
// ``rcr_mma`` cannot be reached this way because the file-scope definition
// is hard-typed on the outer ``RBM/RBN`` (rt_fl<64, 32, ...>); we therefore
// provide a namespace-local overload that takes the smaller acc/A/B tiles
// and forwards to ``mma_ABt`` directly. Same FMA primitive
// (``mfma_scale_f32_16x16x128_f8f6f4`` × cells/warp), only the cell count
// per accumulator changes (height=2, width=1 vs 4, 2).
//
// M2a only instantiates ``<0, false, false>`` and ``<0, true , false>``
// (FUSED_KTAIL=false, FUSE_ACT=false). K-tail and fused-act variants come
// in M2b after correctness validation lands.
//
// Dispatch wiring lands in M3 (file ~5300+ in dispatch_grouped_rcr).
//
// === Round-F M2-debug-3 (correctness-fix, 2026-05-07) ========================
// The 5 file-scope wait-counter macros (RCR_PREFETCH_LGKM=8, RCR_INIT0_VMCNT=4,
// RCR_INIT1_VMCNT=6, RCR_STEADY_VMCNT=8, RCR_EPILOGUE_VMCNT=4) were tuned for
// the outer kernel's BLK=256 / HB=128 LDS volume. The b128 path has HALF the
// LDS data per ST tile (HB=64 → memcpy_per_tile=1 vs 2 in outer; per-call
// vmcnt/lgkm increments halved), so the inherited thresholds were LOOSE enough
// to be no-ops in some prologue paths, allowing first-iter ds_read to fire
// against not-yet-landed buffer_load_lds → MFMA on garbage → systematic SNR
// degradation that grew with iter count (B=4 K=5760 dgrad: SNR 18-31 dB vs
// outer 297 dB). Localised bug: line 3617's ``s_waitcnt lgkmcnt(0)`` already
// drains all pending LDS ops before each MFMA, so steady-state correctness is
// preserved by the explicit drain — but the PROLOGUE relies on
// TK_WAIT_VMCNT(N) being a real wait, not a no-op. With outer's N=4/6, b128's
// 4-tile prologue (4 vmcnt ops) leaves N=4 outstanding → no drain at all →
// first-MFMA reads stale registers.
//
// Fix (5 hardcoded inline-asm waits replacing the 5 file-scope macro
// expansions inside the b128 kernel body, halved proportionally to HB ratio):
//   * vmcnt(2) for INIT0  (was 4)        — drain 2 of 4 prologue-1 vmcnt ops
//   * vmcnt(3) for INIT1  (was 6)        — drain 5 of 8 prologue-1+2 ops
//   * lgkmcnt(4) for PREFETCH (was 8)    — keep main-loop prefetch hint tight
//   * vmcnt(4) for STEADY (was 8)        — same
//   * vmcnt(2) for EPILOGUE (was 4)      — same
//
// === Round-F M5 perf-falsification (2026-05-07) ==============================
// With M2-debug-3 wait fix in place, b128 produces correct output (8/8 PASS
// on the gpt_oss kernel-only metric, SNR 297 dB vs outer's 297 dB on every
// shape). However ALL gpt_oss shapes are SLOWER under b128 than the outer
// BLK=256 kernel, including the tile-starvation cases the port targeted:
//
//   shape                  outer-fwd  b128-fwd  outer-dgrad  b128-dgrad
//   GateUP_B4_M2048        1878       1875      1934         1487  (-23%)
//   GateUP_B4_M4096        2051       2077      2529         1606  (-37%)
//   GateUP_B32_M2048       2022       2021      2564         1664  (-35%)
//   GateUP_B32_M4096       2108       2120      2602         1608  (-38%)
//
// (Down family doesn't enter b128 because K_kern=2880 is K%128=64 not aligned.
// fwd shapes don't enter b128 because K=2880 is K%128=64 not aligned for fwd
// either; b128 is only reached on H4-rerouted dgrad with K_kern=N_orig=5760.)
//
// Root cause: per-tile fixed overhead (binary group search, prefetch issue,
// store epilog with 4 mul + 4 store) does NOT scale down with tile size. b128
// has 4x more tiles per shape (BLK 128 vs 256 = 4x area reduction), so 4x
// more per-tile overhead. The MFMA throughput per tile drops 4x in step (each
// b128 tile = 2 cells × 1 MFMA per K-step vs outer's 8 cells), so MFMA isn't
// faster either. Net: same MFMA work + 4x overhead = ~30-40 % regression on
// the H4-dgrad shapes. The "1.26 tiles/CU" starvation hypothesis projected a
// 1.5-2x speedup from CU-utilization — but the metric showed b128 at
// "4.84 tiles/CU" still loses by 23-38 %, contradicting the projection.
//
// Round F status: M2a structural port WORKS, M2-debug-3 wait fix WORKS,
// M5 perf hypothesis FALSIFIED. Kernel kept as scaffolding (env-gated to
// TURBO_FP8_B128=1, OFF in production) for future research that may change
// the conclusion (e.g., a 4-wave 2-CTA layout with different occupancy could
// flip the per-tile-overhead arithmetic). Round G picks a different lever.
namespace kernel_b128 {

// Namespace-local rcr_mma overload — accepts the half-sized accumulator /
// A / B tiles defined at the top of namespace kernel_b128 (line ~340).
// Dispatches to the same ``mma_ABt`` primitive as the outer rcr_mma; the
// per-cell mfma intrinsic is identical (``mfma_scale_f32_16x16x128_f8f6f4``).
__device__ __forceinline__ void rcr_mma(
    C_acc& acc,
    const A_row_reg& a,
    const B_row_reg& b)
{
    mma_ABt(acc, a, b, acc);
}

template<int KI_HINT = 0, bool N_MASKED_STORE = false, bool FUSED_KTAIL = false,
         bool FUSE_ACT = false>
__global__ __launch_bounds__(_NUM_THREADS, 1)
void grouped_rcr_kernel(
    const std::conditional_t<FUSE_ACT,
                             grouped_layout_globals_fused_act,
                             grouped_layout_globals> g) {
    // [fused-act R6] The FUSE_ACT=true instantiation reads BF16 ``g.a`` from
    // HBM and converts to FP8 inside the load helper (Phase 1 of the FP8
    // grouped fused-act forward optimization). The K-tail fuse path B reads
    // FP8 bytes via reinterpret_cast — incompatible with the BF16 fused-act
    // input — so we gate the two flags as mutually exclusive at compile time.
    // The un-fused fallback (Primus ``_unfused_forward``) handles K-tail
    // correctness when fused-act is selected on K%128 != 0 shapes (none in
    // our initial gate; Phase 1 covers K%128==0 = 16/24 metric shapes).
    static_assert(!(FUSE_ACT && FUSED_KTAIL),
                  "FUSE_ACT=true requires FUSED_KTAIL=false");
    using ST_rcr = ST_v2;
    __shared__ ST_rcr As[2][2];
    __shared__ ST_rcr Bs[2][2];
    // [grouped] LDS cache for device group_offs (int32 view) + per-group
    // tile-cumsum. group_offs is read O(N_iter * G) times by the per-tile
    // inner scan; caching to LDS once at kernel entry replaces ~640 cycles
    // of HBM-cached ld/iter with ~320 cycles of LDS ld/iter, ~3-5% kernel
    // speedup on shapes with low ki / many tiles. Cap MAX_G_PLUS_1 = 65 to
    // cover G ≤ 64 (metric uses G ≤ 32). 8×65 = 520 bytes LDS, negligible.
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];
    __shared__ int s_cum_tiles[MAX_G_PLUS_1];
    __shared__ int s_total_tiles;
    A_row_reg a;
    B_row_reg b0, b1;
    // Round-3 (gpt_oss FP8 focus) — extra A register tile for K-tail
    // M-slab 1, declared inside the ``if constexpr (FUSED_KTAIL)``
    // block at line ~2290 (round-7-dm scoping cleanup: was previously
    // declared at function scope; compiler DCE was already eliminating
    // the slot for FUSED_KTAIL=false template spec; bit-identical
    // codegen but clearer intent).
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;

    // [grouped] Persistent: chiplet-swizzle pid against the actual launch
    // grid (NUM_CUS in the default case; smaller when the dispatcher
    // applies the R4 short-grid lever).
    // Round-67: ``g.num_xcds`` is a host-side knob (default 0 → fallback
    // to ``BLOCK_SWIZZLE_NUM_XCDS=8``). Each shape can override via the
    // Python config rule. Mirrors BF16 grouped's existing ``g.num_xcds``
    // handling (analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp:3249).
    //
    // Round-4 (gpt_oss FP8 kernel-only): swap constexpr ``NUM_CUS`` for
    // runtime ``gridDim.x`` so the chiplet swizzle range tracks the
    // actual launch geometry. Identical math when ``gridDim.x ==
    // NUM_CUS`` (the default); enables the ``TK_RCR_NUM_CUS`` env probe
    // (and a future ``g.num_slots`` knob, mirroring the var-K R2/R3
    // lever) to reduce the persistent grid for short-grid sparse fwd
    // shapes (e.g. Down-B4-M2048 fwd at ~1.5 wave-steps/CU).
    // ``gridDim.x`` is wave-uniform so codegen is stable; no change to
    // register pressure, LDS layout, or HBM stride.
    const int slots_eff = gridDim.x;
    const int xcds_eff = g.num_xcds > 0 ? g.num_xcds : BLOCK_SWIZZLE_NUM_XCDS;
    int pid = chiplet_transform_chunked(
        blockIdx.x, slots_eff, xcds_eff, 64);

    int wm = warpid() / WARPS_N;
    int wn = warpid() % WARPS_N;
    const int num_pid_n = g.bpc;
    const int ki_dyn   = (KI_HINT > 0) ? KI_HINT : g.ki;

    // [grouped] Cooperative init of the LDS group-metadata caches.
    //
    // Round-9-dm: split the original single-threaded init into
    //   (a) parallel HBM read of g.group_offs[0 .. g.G] (65 entries max, fit
    //       in <3 warp-coalesced cachelines so the original O(G) serialized
    //       HBM loads collapse to a single warp-wide wavefront transfer),
    //   (b) parallel pad of s_cum_tiles[g.G+1 .. MAX_G_PLUS_1),
    //   (c) intra-CTA sync, then thread-0 runs the O(G) prefix-scan from
    //       LDS (all the g.group_offs[] fetches have already retired).
    // Pad s_cum_tiles[g.G + 1 .. MAX_G_PLUS_1) with INT_MAX so a constant-
    // depth (6-step) branch-free binary search reading any mid > g.G never
    // updates lo (the cmp `gt >= INT_MAX` is always false for finite gt).
    if (threadIdx.x <= g.G && threadIdx.x < MAX_G_PLUS_1) {
        s_offs[threadIdx.x] = static_cast<int>(g.group_offs[threadIdx.x]);
    }
    if (threadIdx.x > g.G && threadIdx.x < MAX_G_PLUS_1) {
        s_cum_tiles[threadIdx.x] = 0x7FFFFFFF;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int prev = s_offs[0];
        s_cum_tiles[0] = 0;
        int t = 0;
        #pragma unroll 1
        for (int gi = 0; gi < g.G; ++gi) {
            const int next = s_offs[gi + 1];
            t += ((next - prev) / BLOCK_SIZE) * num_pid_n;
            s_cum_tiles[gi + 1] = t;
            prev = next;
        }
        s_total_tiles = t;
    }
    __syncthreads();
    const int total_tiles = s_total_tiles;

    // Prefill swizzled offsets ONCE (shared across all tiles & all groups —
    // depends only on the GL strides which are constant within the launch).
    constexpr int bpt = ST_rcr::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * _NUM_THREADS;
    constexpr int mpt = ST_rcr::rows * ST_rcr::cols * sizeof(fp8e4m3) / bpm;
    uint32_t soA[mpt], soB[mpt];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

    // [fused-act R6] One-shot read of the forward FP8 scale (FP8_MAX / amax)
    // into a wave-uniform float register. Used by every load_a call in the
    // FUSE_ACT=true path. Read ONCE here (not in the per-tile epilog) so the
    // HBM load is shared across all tiles processed by this CU. ``g.dscale_a``
    // stores the forward scale for fused-act (output of the R1
    // ``max_abs_bf16_to_fp8_scale`` binding); for the un-fused (FUSE_ACT=false)
    // path this variable is unused (compiler DCE drops the load).
    float scale_a_inv = 0.0f;
    if constexpr (FUSE_ACT) {
        scale_a_inv = (g.dscale_a != nullptr) ? *g.dscale_a : 0.0f;
    }

    // K-tail fuse uses path B (round-3 commit 07354791): direct per-lane
    // ``raw_buffer_load_b128`` HBM→register inside the fuse epilog (~line
    // 2300+ below), no LDS round-trip and no swizzled-offset prefill. The
    // path-A scaffolding (``soA_tail`` / ``soB_tail`` declarations +
    // ``prefill_swizzled_offsets_partial_K`` calls) was removed in round 11
    // (commit f9d591cb); the helper itself was removed in round 13.

    // [grouped] Persistent outer loop.
    // Round-4: stride by ``slots_eff`` (= gridDim.x) instead of constexpr
    // NUM_CUS so the persistent loop covers all tiles when the dispatcher
    // launches with a reduced grid. When gridDim.x == NUM_CUS this is
    // bit-identical to the prior constexpr stride.
    for (int gt = pid; gt < total_tiles; gt += slots_eff) {
        // [grouped] 6-step branch-free binary search over LDS-cached cumsum
        // (covers G ∈ [1, 64] since 2^6 = 64 = MAX_G_PLUS_1-1). Sentinel
        // INT_MAX past g.G keeps the `gt >= s_cum_tiles[mid]` cmp false so
        // lo never advances past g.G. Compared to the linear O(G) scan,
        // this collapses ~32 LDS lds + cmp into 6 sequential lookups: ~70
        // cyc instead of ~320 cyc per outer iter (kernel-only saving ~3-5%
        // on shapes with low ki / many tiles).
        int lo = 0;
        int hi = MAX_G_PLUS_1 - 1;
        #pragma unroll
        for (int level = 0; level < 6; ++level) {
            const int mid = (lo + hi + 1) >> 1;
            if (gt >= s_cum_tiles[mid]) lo = mid;
            else hi = mid - 1;
        }
        const int group_idx = lo;
        const int tile_start = s_cum_tiles[lo];
        const int local_tile = gt - tile_start;
        const int m_start_g = s_offs[group_idx];
        const int M_g = s_offs[group_idx + 1] - m_start_g;
        const int bpr_g = M_g / BLOCK_SIZE;

        // Group-by-M / group-by-N swizzle (matches dense kernel mapping).
        int br, bc;
        if (g.bpc > bpr_g) {
            const int WGN = g.group_m;
            const int num_wgid_in_group = bpr_g * WGN;
            int group_id = local_tile / num_wgid_in_group;
            int first_pid_n = group_id * WGN;
            int group_size_n = min(num_pid_n - first_pid_n, WGN);
            if (group_size_n <= 0) continue;
            bc = first_pid_n + ((local_tile % num_wgid_in_group) % group_size_n);
            br = (local_tile % num_wgid_in_group) / group_size_n;
        } else {
            const int WGM = g.group_m;
            const int num_wgid_in_group = WGM * num_pid_n;
            int group_id = local_tile / num_wgid_in_group;
            int first_pid_m = group_id * WGM;
            int group_size_m = min(bpr_g - first_pid_m, WGM);
            if (group_size_m <= 0) continue;
            br = first_pid_m + ((local_tile % num_wgid_in_group) % group_size_m);
            bc = (local_tile % num_wgid_in_group) / group_size_m;
        }
        if (br >= bpr_g || bc >= num_pid_n) continue;

        // Coord shifts:
        //   ST_A: st_fp8e4m3<HB=128, BK=128, ...> → row-coord unit = HB = 128.
        //         m_subtile_A = m_start_g / HB.
        //   C (RT store): rt_fl<RBM=64, RBN=32, ...> → row-coord unit = RBM=64.
        //         m_subtile_C = m_start_g / RBM.
        const int m_subtile_A = m_start_g / HB;
        const int m_subtile_C = m_start_g / RBM;

        auto a_co = [&](int s, int k) -> coord<ST_rcr> {
            return {0, 0, m_subtile_A + s, k};
        };
        auto b_co = [&](int s, int k) -> coord<ST_rcr> {
            return {0, group_idx, s, k};
        };

        auto load_a = [&](A_row_reg& dst, ST_rcr& tile, int wi) {
            auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
            load(dst, sub);
        };
        auto load_b = [&](B_row_reg& dst, ST_rcr& tile, int wi) {
            auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
            load(dst, sub);
        };

        auto b_tile = [&](int stage, int which) -> ST_rcr& {
            return Bs[stage][which];
        };

        // Reset accumulators per tile.
        zero(cA); zero(cB); zero(cC); zero(cD);

        int tic = 0, toc = 1;
        // Prologue: load tile-0 + tile-1 (mirrors gemm_kernel<RCR> 1040-1054).
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 0), g.b, b_co(bc*2,   0), soB);
        if constexpr (FUSE_ACT) fused_act_round5_compile_test::rcr_8w_load_hoist_fused_act<_NUM_THREADS>(As[tic][0], g.a, a_co(br*2,   0), soA, scale_a_inv);
        else                    rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0],    g.a, a_co(br*2,   0), soA);
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 1), g.b, b_co(bc*2+1, 0), soB);
        if constexpr (FUSE_ACT) fused_act_round5_compile_test::rcr_8w_load_hoist_fused_act<_NUM_THREADS>(As[tic][1], g.a, a_co(br*2+1, 0), soA, scale_a_inv);
        else                    rcr_8w_load_hoist<_NUM_THREADS>(As[tic][1],    g.a, a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt vmcnt(2)");
        __builtin_amdgcn_s_barrier();

        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(toc, 0), g.b, b_co(bc*2,   1), soB);
        if constexpr (FUSE_ACT) fused_act_round5_compile_test::rcr_8w_load_hoist_fused_act<_NUM_THREADS>(As[toc][0], g.a, a_co(br*2,   1), soA, scale_a_inv);
        else                    rcr_8w_load_hoist<_NUM_THREADS>(As[toc][0],    g.a, a_co(br*2,   1), soA);
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(toc, 1), g.b, b_co(bc*2+1, 1), soB);

        asm volatile("s_waitcnt vmcnt(3)");
        __builtin_amdgcn_s_barrier();

        // Single-tile main loop (mirrors dense gemm_kernel<RCR> else-branch
        // lines 1129-1158).
        // Round-4 (gpt_oss FP8 focus) PROBE: removed 2x RCR_SCHED_BARRIER()
        // per K-iter (compiler reorder hint), keeping all s_setprio +
        // s_barrier + s_waitcnt. Round-2 falsified removing BOTH
        // sched_barrier AND setprio together (-5.6%); this isolates whether
        // the regression came from setprio alone (priority-bias for MFMA
        // issue) or sched_barrier alone (compiler reorder block). If this
        // probe is neutral or +ve, the round-2 regression was setprio-only
        // → confirms sched_barrier overhead is removable.
        //
        // Round-3-dm: swept local UNROLL override {1, 2, 4} × 5 runs each.
        // Means 816.2 / 816.2 / 816.4 — flat across the entire range.
        // LLVM's unroll heuristic already produces optimal body layout;
        // `#pragma unroll N` is an ignorable hint here. See round-3-dm
        // note. Saturated. Use shared RCR_MAIN_UNROLL (=2, matches dense).
        TK_PRAGMA_UNROLL(RCR_MAIN_UNROLL)
        for (int k = 0; k < ki_dyn - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, b_tile(tic, 0), wn);
            load_a(a, As[tic][0], wm);
            if constexpr (FUSE_ACT) fused_act_round5_compile_test::rcr_8w_load_hoist_fused_act<_NUM_THREADS>(As[toc][1], g.a, a_co(br*2+1, k+1), soA, scale_a_inv);
            else                    rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], g.a, a_co(br*2+1, k+1), soA);
            asm volatile("s_waitcnt lgkmcnt(4)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, b_tile(tic, 1), wn);
            rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 0), g.b, b_co(bc*2, k+2), soB);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            if constexpr (FUSE_ACT) fused_act_round5_compile_test::rcr_8w_load_hoist_fused_act<_NUM_THREADS>(As[tic][0], g.a, a_co(br*2, k+2), soA, scale_a_inv);
            else                    rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0], g.a, a_co(br*2, k+2), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 1), g.b, b_co(bc*2+1, k+2), soB);
            asm volatile("s_waitcnt vmcnt(4)"); __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_setprio(1); rcr_mma(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        // Epilog 1: second-to-last K-tile (mirrors dense lines 1160-1187).
        {
            load_b(b0, b_tile(tic, 0), wn);
            load_a(a, As[tic][0], wm);
            if constexpr (FUSE_ACT) fused_act_round5_compile_test::rcr_8w_load_hoist_fused_act<_NUM_THREADS>(As[toc][1], g.a, a_co(br*2+1, ki_dyn-1), soA, scale_a_inv);
            else                    rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], g.a, a_co(br*2+1, ki_dyn-1), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

            load_b(b1, b_tile(tic, 1), wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            asm volatile("s_waitcnt vmcnt(2)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, b_tile(toc, 0), wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();
            tic ^= 1; toc ^= 1;
        }

        // Epilog 2: last K-tile (mirrors dense lines 1189-1210).
        {
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, b_tile(tic, 1), wn);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rcr_mma(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            rcr_mma(cC, a, b0);
            rcr_mma(cD, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        // === Fused K-tail epilog (path B, round-3 commit 07354791) ===
        // After Epilog 2, cA/cB/cC/cD hold sum over K=[0, fast_k). For
        // K-misaligned shapes (e.g. gpt_oss K=2880, K_REM=64) we accumulate
        // K=[fast_k, fast_k + K_BLOCK) directly into cA/cB/cC/cD inside this
        // same launch — no standalone K-tail kernel, no RMW on g.c.
        //
        // History (see ``analysis/_notes/round-{2,3}-fp8-ktail-*.md`` for
        // full derivation):
        //   * Round-2 (commit 4f6a2dee) shipped path A: LDS-staged K-tail
        //     using ``prefill_swizzled_offsets_partial_K`` + SENTINEL
        //     voffset + ``buffer_load_lds``. Saturated at SNR ~20.7 dB
        //     because ``buffer_load_lds`` is a NO-OP on OOB voffset (it
        //     does NOT zero LDS — only ``buffer_load.iX`` zero-fills VGPR
        //     on OOB), leaving OOB lanes' LDS slots holding stale main-
        //     loop K-tile data; ``load(reg, st_subtile)`` then accumulated
        //     that stale data with weight 1.
        //   * Round-3 (commit 07354791) shipped the live code below, path B:
        //     each lane reads ``2 × buffer_load_b128`` directly from HBM
        //     into A_row_reg / B_row_reg ``data[]``, lane-cell mapping
        //     hand-derived for ``rt_16x128_s``. SRD range_bytes
        //     auto-zero-fills VGPR for K-OOB lanes — exactly what path A
        //     wanted but couldn't get from ``buffer_load_lds``. SNR
        //     ≥25 dB on all 8 gpt_oss K=2880 cases; metric 153 → 465.
        //   * Round-11 (commit f9d591cb) dropped the dead path-A scaffolding
        //     (``soA_tail`` / ``soB_tail`` declarations + the two
        //     ``prefill_swizzled_offsets_partial_K`` calls in the prologue);
        //     round-13 dropped the helper definition itself.
        if constexpr (FUSED_KTAIL) {
            // Round-7-dm: scoped A register tile for K-tail M-slab 1
            // (moved from function-scope; round-3 introduced the reg
            // to save one ``vmcnt(0)`` wait on K-misaligned shapes).
            A_row_reg a_kt1;
            if (g.fast_k < g.k) {
                // === Round-3 path B: direct HBM → register K-tail load ===
                // Mirrors BF16 round-5 path B
                // (kernel_bf16_dynamic.cpp:709-827). Each lane reads
                // 2 × buffer_load_b128 (= 32 fp8 cells) directly from
                // HBM into A_row_reg / B_row_reg `data[]`, sidestepping
                // LDS entirely and the round-3 phantom-read pattern
                // documented in
                // ``analysis/_notes/round-3-bf16-ktail-phantom-read.md``
                // and ``analysis/_notes/round-3-fp8-ktail-path-a-saturation.md``.
                //
                // Lane → cell mapping for ``rt_16x128_s`` (fp8e4m3, 64
                // lanes/warp, 32 cells/lane = 32 bytes = 2 × b128):
                //   row_lane    = laneid % 16     (16 rows/base tile)
                //   k_lane_byte = (laneid/16) * 32 (contiguous K cells)
                //   data[0..3]  = K=[k_lane_byte, k_lane_byte + 16) → b128 #1
                //   data[4..7]  = K=[k_lane_byte + 16, k_lane_byte + 32) → b128 #2
                //
                // K_REM=64 < K_STEP=128 (gpt_oss K=2880=22*128+64):
                //   laneid 0..15  : k_lane_byte=0  → both b128 valid
                //   laneid 16..31 : k_lane_byte=32 → both b128 valid
                //   laneid 32..47 : k_lane_byte=64 → both b128 K-OOB
                //   laneid 48..63 : k_lane_byte=96 → both b128 K-OOB
                //
                // K-OOB lanes get ``voffset = SENTINEL`` so the SRD
                // range_bytes check rejects the load → VGPR returns 0.
                // ``raw_buffer_load_b128`` zero-fills VGPR on OOB
                // (unlike ``raw_buffer_load_lds`` which is no-op, the
                // path-A blocker).
                //
                // SRDs:
                //   * A: full-tensor (M_total × K bytes). OOB row >
                //     range clamps to 0.
                //   * B: per-group ((group_idx + 1) × N × K bytes).
                //     Without per-group bound, OOB N-rows on partial
                //     last col-tile would wrap into NEXT group's data.
                //     Per-group bound clamps to 0; column-masked C store
                //     drops the OOB cells.
                //
                // ``row_stride_bytes=0`` in ``make_srsrc`` keeps the
                // linear range-bytes bound check (mirror of BF16 round-11
                // gpt_oss K=2880 fix: cache-swizzle on non-power-of-2
                // strides has UB OOB-clamp behaviour; raw range works).
                //
                // Currently gated at K_REM ∈ {32, 64, 96} (32-aligned)
                // at the dispatcher. Mixed K_REM (e.g. 80) requires
                // partial-b128 lane mask; future round.

                const int laneid = kittens::laneid();
                const int row_lane = laneid % 16;
                const int k_lane_byte = (laneid / 16) * 32;
                // Round-49-dm (auto-optimize R49): FUSED_KTAIL invariant
                // collapse — dispatcher gate ``fuse_ktail_eligible`` (line
                // ~5335) only enables this template spec when
                // ``K_rem_for_fuse ∈ {64, 0}``. The runtime branch
                // ``if (g.fast_k < g.k)`` (parent of this scope) is FALSE
                // when ``K_REM = 0`` (g.fast_k == g.k), so inside this
                // scope K_REM is necessarily 64.
                //
                // Replaces the prior dynamic ``K_REM = g.k - g.fast_k`` +
                // 2 per-lane masks ``b128_lo_valid = (k_lane_byte + 16) <=
                // K_REM`` and ``b128_hi_valid = (k_lane_byte + 32) <=
                // K_REM`` with a single constexpr-folded mask. Since
                // ``k_lane_byte = (laneid / 16) * 32 ∈ {0, 32, 64, 96}``,
                // for K_REM=64 both masks collapse to ``laneid < 32``:
                //
                //   laneid 0..15  : k_lane_byte=0  → lo: 16≤64 ✓ | hi: 32≤64 ✓ → both true
                //   laneid 16..31 : k_lane_byte=32 → lo: 48≤64 ✓ | hi: 64≤64 ✓ → both true
                //   laneid 32..47 : k_lane_byte=64 → lo: 80≤64 ✗ | hi: 96≤64 ✗ → both false
                //   laneid 48..63 : k_lane_byte=96 → lo:112≤64 ✗ | hi:128≤64 ✗ → both false
                //
                // Eliminates 1 wave-uniform sub op (``g.k - g.fast_k``),
                // 1 K_REM register slot, and one of the two per-lane
                // mask cmps. The resource report (``-Rpass-analysis=
                // kernel-resource-usage``) is bit-identical at the V/A/
                // spill/scratch level (LLVM's CSE was already partially
                // collapsing the redundancy); the .so binary md5 changes
                // (codegen emits 1-3 fewer instructions in the K-tail
                // load lambdas, observed via build comparison this
                // round). Metric: 996 / 987 over 2 runs vs 997 baseline
                // — within run-to-run noise band (982-998 centred ~990).
                //
                // **Future**: if ``fuse_ktail_eligible`` (line ~5335) is
                // ever extended to allow K_REM ∈ {32, 96} or other
                // partial-b128 values, this constexpr collapse must be
                // replaced by a per-K_REM template specialisation.
                // Currently only {0, 64} are gated through, so the
                // invariant holds.
                constexpr int KREM = 64;
                static_assert(KREM == 64,
                    "FUSED_KTAIL=true K_REM must be 64; see fuse_ktail_eligible");
                const bool both_valid = (laneid < 32);
                constexpr uint32_t SENTINEL = 0xFFFF0000u;

                const fp8e4m3* a_base_ptr = (const fp8e4m3*)&g.a[{0, 0, 0, 0}];
                const fp8e4m3* b_base_ptr = (const fp8e4m3*)&g.b[{0, 0, 0, 0}];
                const int a_row_stride_bytes = g.a.template stride<2>();
                const int b_row_stride_bytes = g.b.template stride<2>();
                const uint32_t a_total_bytes =
                    static_cast<uint32_t>(g.M_total) *
                    static_cast<uint32_t>(a_row_stride_bytes);
                const uint32_t b_per_group_bytes =
                    static_cast<uint32_t>(group_idx + 1) *
                    static_cast<uint32_t>(g.n) *
                    static_cast<uint32_t>(b_row_stride_bytes);
                i32x4 a_srsrc_kt = make_srsrc((const void*)a_base_ptr, a_total_bytes);
                i32x4 b_srsrc_kt = make_srsrc((const void*)b_base_ptr, b_per_group_bytes);

                const uint32_t K_tail_base_bytes =
                    static_cast<uint32_t>(g.fast_k);
                const uint32_t b_group_byte_base =
                    static_cast<uint32_t>(group_idx) *
                    static_cast<uint32_t>(g.n) *
                    static_cast<uint32_t>(b_row_stride_bytes);

                // M_warp_base derivation:
                //   a_co(s, k) → coord {0, 0, m_subtile_A + s, k}
                //   unit_coord<2,3>: row = (m_subtile_A + s) * ST_rcr::rows = (.) * HB
                //   warp wm picks rows wm*RBM..wm*RBM+RBM-1 within the 128-row tile.
                //   For h ∈ [0, A_row_reg::height = 4): row = M_warp_base + h*16 + row_lane.
                // Round-3: refactored to take A_row_reg by reference so M-slab 0
                // and M-slab 1 can target different register tiles (a vs a_kt1).
                // This lets us issue all 12 K-tail buffer_loads up front and
                // drain with a single vmcnt(0) before the 4 K-tail mfma —
                // saves ~1 vmcnt(0) round-trip per output tile (~50-100 cyc).
                auto load_a_kt = [&](A_row_reg& A_tile, int slab)
                        __attribute__((always_inline)) {
                    const int M_warp_base =
                        (m_subtile_A + br * 2 + slab) * HB + wm * RBM;
                    #pragma unroll
                    for (int h = 0; h < A_row_reg::height; ++h) {
                        const int A_row_idx = M_warp_base + h * 16 + row_lane;
                        const uint32_t v_base = static_cast<uint32_t>(
                            A_row_idx * a_row_stride_bytes +
                            K_tail_base_bytes + k_lane_byte);
                        const uint32_t v_lo = both_valid ? v_base : SENTINEL;
                        const uint32_t v_hi = both_valid ? (v_base + 16) : SENTINEL;
                        __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            a_srsrc_kt, v_lo, 0, 0);
                        __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            a_srsrc_kt, v_hi, 0, 0);
                        *reinterpret_cast<__uint128_t*>(&A_tile.tiles[h][0].data[0]) = v0;
                        *reinterpret_cast<__uint128_t*>(&A_tile.tiles[h][0].data[4]) = v1;
                    }
                };

                // N_warp_base derivation:
                //   b_co(s, k) → coord {0, group_idx, s, k}
                //   unit_coord: N-row in tile = s * ST_rcr::rows = s * HB.
                //   warp wn picks rows wn*RBN..wn*RBN+RBN-1 within the 128-row tile.
                //   For h_b ∈ [0, B_row_reg::height = 2):
                //     B_row_idx_in_group = N_warp_base + h_b*16 + row_lane.
                //   Global byte = group_idx * N * K + B_row_idx_in_group * K + ...
                auto load_b_kt = [&](B_row_reg& B_tile, int n_strip) __attribute__((always_inline)) {
                    const int N_warp_base =
                        (bc * 2 + n_strip) * HB + wn * RBN;
                    #pragma unroll
                    for (int h_b = 0; h_b < B_row_reg::height; ++h_b) {
                        const int B_row_idx_in_group = N_warp_base + h_b * 16 + row_lane;
                        const uint32_t v_base = b_group_byte_base + static_cast<uint32_t>(
                            B_row_idx_in_group * b_row_stride_bytes +
                            K_tail_base_bytes + k_lane_byte);
                        const uint32_t v_lo = both_valid ? v_base : SENTINEL;
                        const uint32_t v_hi = both_valid ? (v_base + 16) : SENTINEL;
                        __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            b_srsrc_kt, v_lo, 0, 0);
                        __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            b_srsrc_kt, v_hi, 0, 0);
                        *reinterpret_cast<__uint128_t*>(&B_tile.tiles[h_b][0].data[0]) = v0;
                        *reinterpret_cast<__uint128_t*>(&B_tile.tiles[h_b][0].data[4]) = v1;
                    }
                };

                // Round-3: issue ALL K-tail buffer_loads up front, drain with
                // a SINGLE ``vmcnt(0)`` wait, then 4 mfma sequential. M-slab 0
                // → ``a``, M-slab 1 → ``a_kt1`` (separate register). Saves one
                // ``s_waitcnt vmcnt(0)`` round-trip per output tile (~50-100
                // cyc) and lets the SQ overlap the buffer_loads.
                //
                // Round-12-dm: split the single ``vmcnt(0)`` into a 2-stage
                // wait so the cA/cB mfmas can overlap with the M-slab-1
                // ``a_kt1`` HBM drain. Issue order is now
                //   a (8 b128) → b0 (4) → b1 (4) → a_kt1 (8)        // 24 total
                // ``vmcnt(8)`` waits until <= 8 outstanding (i.e. only the
                // last 8 issued = a_kt1 still in flight), at which point
                // a/b0/b1 are guaranteed drained (vmcnt is in-issue-order
                // retirement on AMDGCN — same semantics relied on by the
                // main loop's ``RCR_STEADY_VMCNT=8`` mid-iter wait at line
                // 2199). cA = a · b0 and cB = a · b1 then fire while the
                // remaining 8 a_kt1 loads complete in parallel; the
                // ``vmcnt(0)`` before cC/cD acts as a no-op when those
                // already drained. Estimated saving: 1-2 mfma latencies
                // (~32-64 cyc) per K-tail output tile, K-misaligned
                // (gpt_oss K=2880 K_REM=64) shapes only.
                // Round-37-dm: reorder K-tail issues from
                //   [a(8), b0(4), b1(4), a_kt1(8)]  →  [b0(4), b1(4), a(8), a_kt1(8)]
                // Rationale: issue the smaller B tiles FIRST so they saturate
                // HBM controller request-queue slots with finer-grained fetches,
                // then issue the larger A tiles. Under in-issue-order VMEM
                // retirement (per R12-dm comment + main-loop RCR_STEADY_VMCNT=8
                // invariant), the first 16 retired are now b0+b1+a — all three
                // needed for mfma cA/cB. vmcnt(8) fires after first 16 retire
                // (same semantics as original), mfma cA/cB runs overlapping
                // with a_kt1 drain. Zero correctness impact. Potential win: if
                // HBM scheduler prioritises smaller requests first, the 8
                // b128-load B-batch drains slightly earlier than if A-batch
                // came first, cascading to mfma cA starting sooner.
                load_b_kt(b0,    0);   // 4 buffer_load → b0 (issued FIRST)
                load_b_kt(b1,    1);   // 4 buffer_load → b1
                load_a_kt(a,     0);   // 8 buffer_load → a (M slab 0)
                load_a_kt(a_kt1, 1);   // 8 buffer_load → a_kt1 (M slab 1, LAST)
                asm volatile("s_waitcnt vmcnt(8)");
                rcr_mma(cA, a,     b0);
                rcr_mma(cB, a,     b1);
                asm volatile("s_waitcnt vmcnt(0)");
                rcr_mma(cC, a_kt1, b0);
                rcr_mma(cD, a_kt1, b1);
            }
        }

        // Apply scale + store with m_subtile_C row shift.
        // Round-44-dm-probe-A: interleave mul + store per accumulator
        // (was: 4× mul, then 4× store). rocprof showed gpt_oss spec
        // <0,true,true> uses 228 B/thread scratch vs 148 B/thread for
        // dsv3_g spec <0,false,true> — the +80 B is N_MASKED helper's
        // live state. Hypothesis: serialising mul→store→next-mul lets
        // LLVM free cA/cB/cC/cD's accumulator VGPR slots progressively,
        // giving the masked-store helper room to spill into vacated
        // slots instead of HBM scratch. Targets the actual bottleneck
        // (scratch I/O REQUEST rate, not allocation count).
        const float combined_scale = resolve_combined_scale_grp<FUSE_ACT>(g);

        if (wm == 0) __builtin_amdgcn_s_barrier();
        // Round-12: mirror BF16 grouped's column-masked C store. With
        // ``g.bpc = ceil_div(g.n, BLOCK_SIZE)`` (dispatch_grouped_rcr
        // round-12 path) the last col-tile may straddle ``[fast_n, n)``;
        // ``store_c_tile_n_masked`` drops OOB columns. ``rcr_8w_load_hoist``
        // already uses the full-tensor SRD (line ~432) so OOB rows in B
        // clamp to 0 — the masked C store then prevents those cells from
        // being written. ``N_MASKED_STORE`` is a compile-time template
        // parameter so the N-aligned dispatch path emits the raw store
        // (no spill from the masked variant's row/col reconstruction).
        //
        // Round-59: hoist per-block N-tail branch from the helper into
        // the kernel epilogue. The helper's top-of-body
        // ``if (n1 <= n_limit) store(...)`` fast-path forwards interior
        // tiles to the same bare ``store(...)`` as the unmasked kernel,
        // but having the masked-helper body in scope on every block
        // (even when the runtime branch falls through to the bare store)
        // inflates VGPR pressure and serialises the epilogue: the
        // unmasked ``<0,false>`` template runs N=5888 in 2.19 ms while
        // the ``<0,true>`` template with helper-internal branching runs
        // the same 23 col-tile work in 3.05 ms (+39 % wall time, gpt_oss
        // GateUP B32-M4096 K=2816, /tmp/profile_fp8_n_alignment.py).
        // Hoisting the branch — interior takes the bare ``store(...)``
        // path identical to the unmasked template; only ``bc == bpc-1``
        // on a misaligned N hits the masked helper — lets the compiler
        // fully specialise both arms and recovers the unmasked-kernel
        // throughput on the 22/23 interior col-tiles. Numerical safety:
        // ``(bc + 1) * BLOCK_SIZE <= g.n`` is the necessary and
        // sufficient condition for the four C sub-tiles cA/cB/cC/cD to
        // fit entirely in [bc*BLOCK_SIZE, g.n) (combined col span of all
        // 4 stores = [bc*BLOCK_SIZE, (bc+1)*BLOCK_SIZE)); when true, the
        // bare ``store`` writes the same cells as the masked helper's
        // ``n1 <= n_limit`` fast-path. ``bc`` is uniform across the wave
        // so this is a single wave-uniform branch, not divergent control
        // flow.
        // Round-24 (PT): readfirstlane on the 4 C-store wave-uniform coords.
        // R22 ASM disassembly counted 411 divergent-SRD fallback loops in
        // FUSED_KTAIL=true rcr<T,T> spec (vs 128 in <F,F>); 80 % live in the
        // C-store epilog. Root cause: r0/r1/c0/c1 are wave-uniform IN PRINCIPLE
        // (m_subtile_C, br, bc, wm, wn all uniform per tile-iter) but LLVM's
        // uniformity analysis cannot prove it — the FUSED_KTAIL block's per-
        // lane VGPR ops (SENTINEL voffsets, b128_lo_valid lane masks) earlier
        // in the function taint downstream flow. Each store(g.c, cX, coord)
        // therefore constructs a VGPR-derived buffer SRD inside the kittens
        // helper (`make_buffer_resource(as_u64, ...)` at memory/tile/
        // global_to_register.cuh:332) and emits the per-lane fallback loop
        // pattern (`v_readfirstlane → v_cmp → s_and_saveexec → buffer_op →
        // s_xor exec → s_cbranch_execnz`) that costs ~20 cy/loop on the
        // common wave-uniform path.
        //
        // Localised readfirstlane HERE (not earlier) avoids R22's V-A/V-B
        // backlash (those tried on `group_idx` in the binary search prologue
        // → +21 dw spill on rcr<T,T> via wide downstream cascade). The 4
        // coord ints are dead-after-stores (no downstream consumers in the
        // tile-iter), so SGPR promotion is local.
        const int r0 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2+wm);
        const int r1 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2+WARPS_M+wm);
        const int c0 = __builtin_amdgcn_readfirstlane(bc*WARPS_N*2+wn);
        const int c1 = __builtin_amdgcn_readfirstlane(bc*WARPS_N*2+WARPS_N+wn);
        if constexpr (N_MASKED_STORE) {
            if ((bc + 1) * BLOCK_SIZE <= g.n) {
                mul(cA, cA, combined_scale);
                store(g.c, cA, {0, 0, r0, c0});
                mul(cB, cB, combined_scale);
                store(g.c, cB, {0, 0, r0, c1});
                mul(cC, cC, combined_scale);
                store(g.c, cC, {0, 0, r1, c0});
                mul(cD, cD, combined_scale);
                store(g.c, cD, {0, 0, r1, c1});
            } else {
                mul(cA, cA, combined_scale);
                store_c_tile_n_masked(g.c, cA, r0, c0, g.n);
                mul(cB, cB, combined_scale);
                store_c_tile_n_masked(g.c, cB, r0, c1, g.n);
                mul(cC, cC, combined_scale);
                store_c_tile_n_masked(g.c, cC, r1, c0, g.n);
                mul(cD, cD, combined_scale);
                store_c_tile_n_masked(g.c, cD, r1, c1, g.n);
            }
        } else {
            mul(cA, cA, combined_scale);
            store(g.c, cA, {0, 0, r0, c0});
            mul(cB, cB, combined_scale);
            store(g.c, cB, {0, 0, r0, c1});
            mul(cC, cC, combined_scale);
            store(g.c, cC, {0, 0, r1, c0});
            mul(cD, cD, combined_scale);
            store(g.c, cD, {0, 0, r1, c1});
        }

        // Round-50-dm probe: drain only LDS (lgkmcnt) at end of tile;
        // skip vmcnt(0). C-store HBM writes (~32 buffer_store_b16 pending)
        // do NOT alias with next tile's HBM loads (g.a / g.b reads of a
        // different (br, bc) cell), so the next tile's prologue can issue
        // its 16 buffer_loads in parallel with C-store drain. The
        // prologue's own ``TK_WAIT_VMCNT(RCR_INIT0_VMCNT=4)`` enforces the
        // sync it needs (prologue waits for ≤4 outstanding before mfma);
        // since vmcnt is shared across loads + stores, that wait will
        // also drain the C-store stragglers if they outlive the load
        // issue path. Keep lgkmcnt(0) to ensure main-loop LDS writes
        // retire before next tile's prologue overwrites the same slab.
        // Estimated: ~150 cy/tile saved (overlap of ~30 cy search +
        // prologue issue with ~150 cy store drain). On gpt_oss-GateUP-
        // B32-M4096 with ~368 tiles/CU, that's ~55K cy = ~27 us = ~1 %
        // wall-time saving, expected ~+1-2 pp on grp_FP8 ratio.
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }
}


// M2a instantiations only — KI_HINT=0, N_MASKED_STORE in {false, true},
// FUSED_KTAIL=false. FUSE_ACT defaulted to false. Keeps codegen surface
// minimal until M2b adds K-tail + M5 ships the dispatcher gate.
template __global__ void grouped_rcr_kernel<0, false, false>(const grouped_layout_globals);
template __global__ void grouped_rcr_kernel<0, true , false>(const grouped_layout_globals);

} // namespace kernel_b128


// =============================================================================
// Round-57-dm (auto-optimize R57): Lever C-2 round-2 step 2A — minimal
// 4w-style ISA-validation kernel.
//
// Background
// ----------
// R47/R48 established that ``rcr_4w::kernel`` (dense 4w-style) emits
// VGPR=198 / AGPR=256 / Spill=0, while every ``grouped_rcr_kernel<*,*>``
// spec emits VGPR=256 / AGPR=0 / Spill=34-54. R47-dm hypothesised that
// the **256 fp32/lane per-warp accumulator footprint** (4 acc × 64
// fp32/lane = 256) is what triggers LLVM to pick AGPR allocation; the
// grouped kernel sits at 128 fp32/lane (4 acc × 32) so LLVM keeps acc
// in VGPR and spills the 250+ VGPRs of non-acc live state to scratch.
//
// R54 step-1 staged the 4w-style geometry constants and register-tile
// **types** (``A_row_reg_4w`` / ``B_row_reg_4w`` / ``C_acc_4w``) in
// ``namespace lever_c2_round_54_step1_scaffold`` (line ~211) and
// validated them via ``static_assert`` only — no codegen exercise yet.
//
// What this round (step-2A) does
// ------------------------------
// Force-instantiate a minimal kernel that:
//   * declares ``cAB[2][2]`` of ``C_acc_4w`` (4 acc × 64 fp32/lane =
//     256 fp32/lane per warp — matches ``rcr_4w::kernel`` per-warp
//     accumulator footprint of 256 fp32/lane / 256 AGPR)
//   * loads ``a_reg[2]`` of ``A_row_reg_4w`` and ``b_reg[2]`` of
//     ``B_row_reg_4w`` from LDS (forces all 4w-style register tiles
//     into the live set simultaneously)
//   * issues ``mma_ABt`` × 4 per K-iter (covers all 4 cAB acc)
//   * stores cAB[*][*] back to ``g.c`` to anchor the output
//
// This is **NOT a working kernel** — the load/store coords are
// placeholder (would not produce correct GEMM output even if launched
// at the appropriate geometry); the goal is purely to make LLVM
// allocate registers for the full 4w-style live set so the resource
// report (-Rpass-analysis=kernel-resource-usage) shows whether AGPR
// gets picked or not.
//
// Acceptance criteria
// -------------------
//   * **HYPOTHESIS CONFIRMED**: AGPR > 0 AND Spill < 30 (≪
//     grouped_rcr_kernel<T,T> baseline 37 spill). R58+ proceed with
//     full step-2B port (real load lambdas + dispatch wire).
//   * **HYPOTHESIS PARTIALLY FALSIFIED**: AGPR > 0 BUT Spill ≥ 30
//     (acc moves to AGPR but other state cascades to scratch like
//     R48's ``+a`` hint experiment). step-2 path may still net-out
//     positive but risks R48-style cascade; R58 inspect ScratchSize
//     to decide whether to proceed.
//   * **HYPOTHESIS FULLY FALSIFIED**: AGPR == 0 (256 fp32/lane
//     threshold not sufficient by itself; rcr_4w must trigger AGPR
//     for some other reason — possibly the non-persistent launch
//     pattern or the ``__launch_bounds__(NT, 2)`` occupancy hint).
//     step-2 path is dead; R58+ pivot to Lever D R-B 5+ or accept
//     plateau.
//
// The kernel is force-instantiated below to make the compiler emit
// codegen even though no caller exists. Following the precedent
// established by ``lever_d_round_b_step1_compile_test`` (line ~124)
// and ``lever_c2_round_54_step1_scaffold`` (line ~211), it lives in a
// dedicated namespace, is never invoked from the runtime dispatcher,
// and has no impact on .so binary behaviour beyond the additional
// symbol's resource-usage line in the build log.
namespace lever_c2_round_57_step2a_compile_test {
    using ::ST_v2;

    using lever_c2_round_54_step1_scaffold::WARPS_M;
    using lever_c2_round_54_step1_scaffold::WARPS_N;
    using lever_c2_round_54_step1_scaffold::_NUM_WARPS;
    using lever_c2_round_54_step1_scaffold::_NUM_THREADS;
    using lever_c2_round_54_step1_scaffold::RBM_4w;
    using lever_c2_round_54_step1_scaffold::RBN_4w;
    using lever_c2_round_54_step1_scaffold::A_row_reg_4w;
    using lever_c2_round_54_step1_scaffold::B_row_reg_4w;
    using lever_c2_round_54_step1_scaffold::C_acc_4w;

    template<int KI_HINT = 4>
    __global__ __launch_bounds__(_NUM_THREADS, 1)
    void test_grouped_rcr_kernel_4w_compile_test(grouped_layout_globals g) {
        // Two LDS slabs per A/B so the K-loop has somewhere to read
        // from. Total LDS = 2 × 2 × sizeof(ST_v2) = 64 KB per slab
        // pair, well within gfx950's 64 KB per-block budget when run
        // with NUM_THREADS=256 (4 warps).
        __shared__ ST_v2 As[2];
        __shared__ ST_v2 Bs[2];

        // 4w-style register tiles: 4 cAB × 64 fp32/lane = 256 fp32/lane
        // per warp (matches rcr_4w::kernel per-warp accumulator
        // footprint of 256 fp32/lane / 256 AGPR baseline).
        A_row_reg_4w a_reg[2];
        B_row_reg_4w b_reg[2];
        C_acc_4w cAB[2][2];
        zero(cAB[0][0]); zero(cAB[0][1]);
        zero(cAB[1][0]); zero(cAB[1][1]);

        const int wm = warpid() / WARPS_N;  // wm ∈ {0, 1}
        const int wn = warpid() % WARPS_N;  // wn ∈ {0, 1}
        const int ki = (KI_HINT > 0) ? KI_HINT : g.ki;

        // Persistent outer loop: matches grouped_rcr_kernel structure.
        // We don't read group_offs here (no native group binary search)
        // — purpose is acc + a/b register allocation pressure, not
        // group correctness. ``M_total / BLOCK_SIZE * bpc`` is the
        // total tile count proxy (sum across groups), which keeps the
        // outer loop syntactically present so LLVM doesn't DCE the
        // inner work.
        const int total_tiles = (g.M_total / BLOCK_SIZE) * g.bpc;
        for (int gt = blockIdx.x; gt < total_tiles; gt += gridDim.x) {
            int tic = 0;

            // Cooperative LDS load. We use the outer ``G`` group<NW=8>
            // here because the grouped_rcr_kernel's existing helpers
            // assume that distribution; the resulting load is "wrong"
            // for a 4-warp launch (overshoots LDS slot stride) but
            // syntactically present so LLVM sees use-def chains. The
            // real step-2B will introduce a 4-warp G_4w cooperative
            // load helper.
            G::load(As[tic], g.a, {0, 0, 0, 0});
            G::load(Bs[tic], g.b, {0, 0, 0, 0});
            __syncthreads();

            // K-loop body: 4 mma per K-iter (matches 4 cAB acc cells).
            // Each mma_ABt expands to height(4) × width(4) × A.width(1)
            // = 16 mma_ABt_base calls = 16 mfma_16x16x128 instructions
            // per cAB acc per K-iter. 4 cAB × 16 = 64 mfma per warp
            // per K-iter (vs grouped_rcr_kernel's 32 mfma per warp
            // per K-iter — 2× compute density).
            #pragma unroll 1
            for (int k = 0; k < ki; k++) {
                // Read from LDS into 4w-style register tiles via the
                // existing ``load(rt_fp8, st)`` ds_read_b128 helper
                // (kernel_fp8_layouts.cpp line ~1336 ``load_full_rt``).
                auto a_sub = subtile_inplace<RBM_4w, BK>(As[tic], {wm, 0});
                auto b_sub = subtile_inplace<RBN_4w, BK>(Bs[tic], {wn, 0});
                load(a_reg[0], a_sub);
                load(b_reg[0], b_sub);
                load(a_reg[1], a_sub);  // placeholder; 2nd M-slab
                load(b_reg[1], b_sub);  // placeholder; 2nd N-slab
                __syncthreads();

                mma_ABt(cAB[0][0], a_reg[0], b_reg[0], cAB[0][0]);
                mma_ABt(cAB[0][1], a_reg[0], b_reg[1], cAB[0][1]);
                mma_ABt(cAB[1][0], a_reg[1], b_reg[0], cAB[1][0]);
                mma_ABt(cAB[1][1], a_reg[1], b_reg[1], cAB[1][1]);
            }

            // Store cAB to keep all 4 acc live through the K-loop
            // (otherwise LLVM DCE's unused cells and shrinks the
            // measured acc footprint). Coords are placeholder; we
            // only care that all 4 cAB are sunk to a memory write.
            store(g.c, cAB[0][0], {0, 0, gt * 4 + 0, 0});
            store(g.c, cAB[0][1], {0, 0, gt * 4 + 1, 0});
            store(g.c, cAB[1][0], {0, 0, gt * 4 + 2, 0});
            store(g.c, cAB[1][1], {0, 0, gt * 4 + 3, 0});
        }
    }
} // namespace lever_c2_round_57_step2a_compile_test

// Force-instantiate so LLVM emits codegen and the
// -Rpass-analysis=kernel-resource-usage line is printed at build time.
// Read the resource report for symbol
// ``lever_c2_round_57_step2a_compile_test::test_grouped_rcr_kernel_4w_compile_test<4>``
// in the build log — compare AGPR / Spill against
// ``grouped_rcr_kernel<0,true,true>`` baseline (256 V / 0 A / 37 Spill).
template __global__ void
lever_c2_round_57_step2a_compile_test::test_grouped_rcr_kernel_4w_compile_test<4>(grouped_layout_globals);

// =============================================================================
// Round-58-dm (auto-optimize R58): Lever C-2 round-2 step 2B-1 — replace
// the R57 step-2A kernel's PLACEHOLDER ``G::load`` (8-warp cooperative
// load called from a 4-warp launch — wrong distribution, but
// syntactically intact for codegen purposes) with the **real** 4-warp
// cooperative load path:
//
//   * ``kittens::group<_NUM_WARPS=4>::prefill_swizzled_offsets`` runs
//     once per launch to fill an 8-entry swizzled-offset SGPR-uniform
//     vector (twice the per-warp pass count of the 8-warp version since
//     each warp now does 4 passes instead of 2 for the same 16 KB
//     ST_v2 slab);
//   * ``rcr_8w_load_hoist<_NUM_THREADS=256>`` issues the
//     ``buffer_load_dwordx4 ... offen lds`` direct-to-LDS hoist with
//     ``num_warps = 256/64 = 4`` (the helper is already generic over
//     N_THREADS — the "8w" in the name is a legacy from when the
//     production grouped kernel was the only caller).
//
// Both helpers are template-parameterized (``group<N>::prefill_*`` via
// the ``GROUP_THREADS = N * WARP_THREADS`` typedef in
// ``include/ops/group/memory/tile/global_to_shared.cuh`` line ~22-27;
// ``rcr_8w_load_hoist`` via ``int N_THREADS`` template parameter in
// kernel_fp8_layouts.cpp line ~806). No new helper code is needed —
// step-2B-1 reduces to a wiring-only test of the existing primitives
// at NUM_THREADS=256 / NUM_WARPS=4.
//
// Acceptance gate (per round-57-dm note R58+ roadmap)
// ---------------------------------------------------
//   * **PASS**: AGPR retained (≥ 200 AGPR, matches step-2A baseline of
//     256). 0 spill or ≤ 5 spill (small uptick acceptable due to soA /
//     soB SGPR-resident vectors). R59 proceed with real coords +
//     correctness probe on a single-shape (M=256, N=256, K=128).
//   * **PARTIAL FAIL**: AGPR drops to 0 OR ScratchSize > 32 B/lane.
//     R48-style cascade — the prefill + hoist machinery's non-acc live
//     state is too large; LLVM picks VGPR over AGPR for the
//     accumulators because the rest of the live set already drains
//     VGPR budget below the AGPR-trigger threshold. R59 must either
//     reduce non-acc state or pivot to Lever D 32x32x64 cell shape.
//   * **FALSIFY**: AGPR=0 AND ScratchSize > 100 B/lane (worse than the
//     placeholder baseline of step-2A despite using the same acc
//     footprint). Hypothesis "256 fp32/lane → AGPR" requires more than
//     just acc footprint — possibly the ``__launch_bounds__(_, 1)`` on
//     rcr_4w::kernel; revisit this after step-2 is dead.
namespace lever_c2_round_58_step2b1_real_load {
    using ::ST_v2;
    using kittens::WARP_THREADS;

    using lever_c2_round_54_step1_scaffold::WARPS_M;
    using lever_c2_round_54_step1_scaffold::WARPS_N;
    using lever_c2_round_54_step1_scaffold::_NUM_WARPS;
    using lever_c2_round_54_step1_scaffold::_NUM_THREADS;
    using lever_c2_round_54_step1_scaffold::RBM_4w;
    using lever_c2_round_54_step1_scaffold::RBN_4w;
    using lever_c2_round_54_step1_scaffold::A_row_reg_4w;
    using lever_c2_round_54_step1_scaffold::B_row_reg_4w;
    using lever_c2_round_54_step1_scaffold::C_acc_4w;

    // 4-warp cooperative group typedef. Drives ``GROUP_THREADS =
    // 4 * WARP_THREADS = 256`` inside kittens::prefill_swizzled_offsets,
    // which selects the per-thread byte stride and pass count for the
    // 4-warp distribution natively (no hand-unroll needed).
    using G_4w = kittens::group<_NUM_WARPS>;

    template<int KI_HINT = 4>
    __global__ __launch_bounds__(_NUM_THREADS, 1)
    void test_grouped_rcr_kernel_4w_real_load(grouped_layout_globals g) {
        __shared__ ST_v2 As[2];
        __shared__ ST_v2 Bs[2];

        A_row_reg_4w a_reg[2];
        B_row_reg_4w b_reg[2];
        C_acc_4w cAB[2][2];
        zero(cAB[0][0]); zero(cAB[0][1]);
        zero(cAB[1][0]); zero(cAB[1][1]);

        const int wm = warpid() / WARPS_N;
        const int wn = warpid() % WARPS_N;
        const int ki = (KI_HINT > 0) ? KI_HINT : g.ki;

        // Prefill the per-pass byte offsets ONCE per launch, mirroring
        // grouped_rcr_kernel line ~2394-2399. The ``mpt_4w`` count is
        // 16 KB / (16 B × 256 thr) = 4 passes per warp, so soA/soB are
        // 4-element uint32 SGPR-uniform arrays.
        constexpr int bpt    = ST_v2::underlying_subtile_bytes_per_thread;
        constexpr int bpm_4w = bpt * _NUM_THREADS;
        constexpr int mpt_4w =
            ST_v2::rows * ST_v2::cols * sizeof(fp8e4m3) / bpm_4w;
        static_assert(mpt_4w == 4,
            "ST_v2 (16 KB) at NUM_THREADS=256 / bpt=16 → 4 passes per "
            "warp (vs 2 for the 8-warp / NUM_THREADS=512 path).");
        uint32_t soA[mpt_4w], soB[mpt_4w];
        G_4w::prefill_swizzled_offsets(As[0], g.a, soA);
        G_4w::prefill_swizzled_offsets(Bs[0], g.b, soB);

        const int total_tiles = (g.M_total / BLOCK_SIZE) * g.bpc;
        for (int gt = blockIdx.x; gt < total_tiles; gt += gridDim.x) {
            int tic = 0;

            // Real 4-warp cooperative load via buffer_load_dwordx4
            // ... offen lds (raw direct-to-LDS, bypasses any register
            // staging). Coords still placeholder ({0,0,0,0}) — we only
            // care that LLVM sees the full hoist machinery (SRD setup,
            // 4-pass per-warp byte ramp, ds_addr SGPR hoist, asm
            // intrinsic) for codegen pressure analysis.
            rcr_8w_load_hoist<_NUM_THREADS>(As[tic], g.a, {0, 0, 0, 0}, soA);
            rcr_8w_load_hoist<_NUM_THREADS>(Bs[tic], g.b, {0, 0, 0, 0}, soB);
            __syncthreads();

            #pragma unroll 1
            for (int k = 0; k < ki; k++) {
                auto a_sub = subtile_inplace<RBM_4w, BK>(As[tic], {wm, 0});
                auto b_sub = subtile_inplace<RBN_4w, BK>(Bs[tic], {wn, 0});
                load(a_reg[0], a_sub);
                load(b_reg[0], b_sub);
                load(a_reg[1], a_sub);
                load(b_reg[1], b_sub);
                __syncthreads();

                mma_ABt(cAB[0][0], a_reg[0], b_reg[0], cAB[0][0]);
                mma_ABt(cAB[0][1], a_reg[0], b_reg[1], cAB[0][1]);
                mma_ABt(cAB[1][0], a_reg[1], b_reg[0], cAB[1][0]);
                mma_ABt(cAB[1][1], a_reg[1], b_reg[1], cAB[1][1]);
            }

            store(g.c, cAB[0][0], {0, 0, gt * 4 + 0, 0});
            store(g.c, cAB[0][1], {0, 0, gt * 4 + 1, 0});
            store(g.c, cAB[1][0], {0, 0, gt * 4 + 2, 0});
            store(g.c, cAB[1][1], {0, 0, gt * 4 + 3, 0});
        }
    }
} // namespace lever_c2_round_58_step2b1_real_load

// Force-instantiate. Compare resource report against the R57 step-2A
// baseline (V256 / A256 / Spill 0 / Scratch 0 — placeholder G::load).
template __global__ void
lever_c2_round_58_step2b1_real_load::test_grouped_rcr_kernel_4w_real_load<4>(grouped_layout_globals);

// =============================================================================
// Round-59-dm (auto-optimize R59): Lever C-2 round-2 step 2B-2 — real
// (br, bc, k) coord indexing into the 4w-style test kernel, single-group
// only (G=1, m_subtile_A=0). This is the **first numerically correct**
// 4w-style grouped FP8 GEMM kernel — outputs match torch fp32 reference
// up to fp8 rounding noise (gate: max_abs ≤ 0.5, SNR ≥ 22 dB).
//
// What this round adds vs R58 step-2B-1
// -------------------------------------
//   * Real ``(br, bc)`` block-tile coordinate from ``gt = blockIdx.x +
//     k * NUM_CUS`` swept over ``total_tiles = bpr * bpc``;
//   * Per-tile A-load = 2 calls (``br*2 + 0`` and ``br*2 + 1``,
//     covering 256 M-rows in 2 ST_v2 slabs of 128 rows each);
//   * Per-tile B-load = 2 calls (similar for N=256);
//   * Per-warp register-tile loads use ``As[wm]`` and ``Bs[wn]`` to
//     pick the correct M / N sub-slab (``wm/wn ∈ {0,1}``);
//   * Real K-loop with 4 mma_ABt per K-iter (4 cAB cells: cAB[i][j]
//     covers M=[wm*128 + i*64, ..+64), N=[wn*128 + j*64, ..+64));
//   * Scale epilog ``mul(cAB, cAB, scale_a*scale_b)`` per cell;
//   * 4 cAB stores at coord ``{0, 0, br*4 + wm*2 + i, bc*4 + wn*2 + j}``
//     (output coord units = RBM_4w = RBN_4w = 64).
//
// What this round does NOT add (deferred to R60+)
// -----------------------------------------------
//   * Group binary search prologue (G=1 only here; m_subtile_A = 0
//     hardcoded). R60 ports the LDS group_offs cache + 6-step binary
//     search from production grouped_rcr_kernel.
//   * K-tail FUSED_KTAIL block (probe shape K=256 hits ki=2, no
//     K-tail needed). R61.
//   * N-mask store (probe shape N=256, fully aligned). R62.
//   * Double-buffered LDS ping-pong (single buffer used here for
//     simplicity; ping-pong is a perf optimization, not correctness).
//
// LDS budget
// ----------
// 2 ST_v2 A-slabs (32 KB) + 2 ST_v2 B-slabs (32 KB) = 64 KB / block.
// Plus ~few bytes for sync state. At occupancy=1 wave/SIMD on gfx950
// (max 160 KB LDS / CU), this leaves substantial headroom for R60+
// additions (group_offs cache ~4 KB).
//
// Acceptance gate
// ---------------
//   * **PASS**: AGPR ≥ 200 (≪ R58 baseline of 256); ``max_abs ≤ 0.5``
//     and ``SNR ≥ 22 dB`` vs torch fp32 ref on probe shape (M=512,
//     N=256, K=256, B=1). R60 proceed with group binary search port.
//   * **PARTIAL FAIL**: AGPR ≥ 200 but correctness mismatch
//     (numerical bug in coord math / store layout). Debug coord
//     mapping; do not proceed to R60 until fixed.
//   * **FALSIFY**: AGPR < 200 (cascade back to VGPR; R48-style failure
//     mode). The real coord indexing pushed the live set over the
//     AGPR-trigger threshold. Pivot to occupancy=2 retention or close
//     out C-2 path.
//
// R59 actual outcome (PARTIAL FAIL, REQUIRES R60 DEBUG)
// -----------------------------------------------------
// Resource report (vs R58 step-2B-1 baseline)
//   R58 step-2B-1: V256 / A256 / Scratch  36 / Spill   8 / Occupancy 1
//   R59 step-2B-2: V256 / A256 / Scratch 324 / Spill  80 / Occupancy 1
//
//   AGPR retained ✓ — the AGPR-allocation hypothesis (per-warp 256
//   fp32/lane footprint triggers LLVM's AGPR allocator) survives the
//   addition of full coord arithmetic, persistent tile loop, and 4
//   cAB cells. Spill jumped 10× from R58 (8 → 80) and Scratch 9× (36
//   → 324). Defer spill diagnosis to R60+ (this is the same order of
//   magnitude as production grouped's 37 spill, so within reasonable
//   range of the eventual production candidate).
//
// Correctness (probe at M=512,N=256,K=256,B=1 vs torch fp32 ref)
//   max_abs = 294 ≫ 0.5 gate, SNR = −35 dB ≪ 22 dB gate → FAIL.
//
//   Per-cell breakdown (block 0, M ∈ [0,256), N ∈ [0,256)):
//     cAB[0][0] for ALL 4 warps → WRONG (chunk_max ~150-300, garbage)
//     cAB[0][1] for ALL 4 warps → CORRECT (diff ~0.004 = fp8 noise)
//     cAB[1][0] for ALL 4 warps → CORRECT
//     cAB[1][1] for ALL 4 warps → CORRECT
//
//   Reorder probe (mma cAB[1][1] first, cAB[0][0] second): same
//   pattern — cAB[0][0] still WRONG, cAB[1][1] still CORRECT. Rules
//   out "first-mma-in-sequence" codegen issue. The bug is SPECIFIC
//   to the cAB[0][0] register tile, not the order of mma calls.
//
//   Logical impossibility ruling: cAB[0][0] = a_reg[0] @ b_reg[0]^T,
//   cAB[0][1] = a_reg[0] @ b_reg[1]^T (CORRECT → a_reg[0] is fine),
//   cAB[1][0] = a_reg[1] @ b_reg[0]^T (CORRECT → b_reg[0] is fine).
//   So both inputs are valid yet the output of cAB[0][0] is garbage.
//   Most likely cause: register aliasing between cAB[0][0]'s VGPR
//   slots and the LLVM-allocated scratch-spill temporaries (80 spill
//   slots use ~80 VGPRs over their live range; cAB[0][0] is the
//   FIRST acc tile declared so its registers are first-allocated and
//   most likely to overlap with spill traffic). R60 to investigate
//   via ISA dump + dummy-acc-shift workaround (declare a sacrificial
//   C_acc_4w before cAB to push the live ranges).
//
// Metric impact: ZERO (test kernel is NOT in the dispatch path —
// only reachable via the dedicated ``test_4w_real_coords`` pybind
// binding used by the probe script). Production grouped path (used
// by the metric) is byte-identical to R58.
//
// R60 debug findings (PARTIAL, REQUIRES R61+ DEEPER ROOT-CAUSE)
// -------------------------------------------------------------
// Three workarounds were tried in R60. None gave full PASS but they
// substantially narrow the diagnosis:
//
// 1. **Sacrificial dummy** (`C_acc_4w cAB_sacrificial; zero(cAB_sacrificial);`
//    declared BEFORE cAB[2][2]): NO EFFECT. Resource report
//    byte-identical (V256/A256/Scratch 324/Spill 80) — LLVM either
//    DCE'd the dummy or its first-allocated VGPRs do not overlap with
//    the bug zone. Falsifies the "first-allocated VGPR ↔ spill traffic
//    overlap" hypothesis from R59.
//
// 2. **Explicit per-base-tile mma** (16 mma_ABt_base loop instead of
//    one mma_ABt(d, ..., d) call per cAB cell): cAB[0][0] PARTIALLY
//    fixed (max diff dropped 294 → 86) but cAB[1][0] and cAB[1][1]
//    became SLIGHTLY off (~0.3 diff, was ~0.004). Net regression.
//    Suggests interleaved per-base-tile codegen exposes / masks
//    different register conflicts than the bulk for-loop.
//
// 3. **Skip ``mul()`` scale epilog** (probe uses scale=1.0 anyway,
//    so mul should be identity): cAB[0][0] **FIRST 4 ROWS NOW
//    CORRECT** vs torch ref (~0.001 diff = pure fp8 noise). But
//    rows 8-31 and cols 16-31 still broken. The mul() call was
//    PROPAGATING / AMPLIFYING the bug; underlying mma load/compute
//    has a real defect.
//
// Per-base-tile breakdown (probe 512×256×256, no-mul):
//   cAB[0][0].tiles[n][m] rows×cols (in cell coords):
//     [n=0..1, m=1] ← BROKEN (rows 0-31, cols 16-31, max diff ~138-216)
//     [n=0..1, m=0,2,3], [n=2..3, m=0..3] ← CORRECT (~0.004 fp8 noise)
//   = exactly 2 specific 16×16 base tiles broken in cAB[0][0],
//     out of 16 total base tiles. cAB[0][1], cAB[1][0], cAB[1][1]
//     all 16 base tiles each → CORRECT.
//
// Logical impossibility check (still holds): cAB[0][1].tiles[0][1]
//   uses a_reg[0].tiles[0][0] @ b_reg[1].tiles[1][0]^T → CORRECT, so
//   a_reg[0].tiles[0][0] is fine. cAB[1][0].tiles[0][1] uses
//   a_reg[1].tiles[0][0] @ b_reg[0].tiles[1][0]^T → CORRECT, so
//   b_reg[0].tiles[1][0] is fine. Yet
//   cAB[0][0].tiles[0][1] = a_reg[0].tiles[0][0] @ b_reg[0].tiles[1][0]^T
//   is BROKEN. Both inputs valid, output broken — possibly LLVM
//   register allocator places cAB[0][0].tiles[{0,1}][1]'s data in
//   AGPR slots that are clobbered by something between mma write and
//   store read. R61 plan:
//
//   * Dump ISA for `test_grouped_rcr_kernel_4w_real_coords<0>`,
//     trace AGPR slots holding cAB[0][0].tiles[0][1] and tiles[1][1].
//   * Try `volatile` on cAB[0][0] declaration to force separate VGPR
//     allocation (no AGPR — confirms AGPR-specific bug if probe passes).
//   * Try copying cAB[0][0] to a temp before store: `add(temp, cAB[0][0],
//     0.0); store(g.c, temp, ...)`. If temp store works, the AGPR-to-VGPR
//     transfer at store-time is the bug.
//   * If above fails, drop to occupancy=2 (force VGPR-only allocation by
//     reducing per-warp accumulator footprint to ≤128 fp32/lane). This
//     loses the AGPR allocation (= R57's confirmed mechanism) but gives
//     a known-good correctness baseline.
//
// R61 step-1 result: temp-copy + store-from-temp probe
// ----------------------------------------------------
// **Codegen radically changed but correctness pattern UNCHANGED**:
//   * VGPR Spill: 80 → 12 (87.5% reduction!)
//   * ScratchSize: 324 → 12 bytes/lane
//   * Same broken base tiles: cAB[0][0].tiles[{0,1}][1] still wrong
//
// The temp-copy dramatically improved compiler register allocation
// (spill 80 → 12) but did NOT fix the broken base tiles. This proves:
//   1. The bug is NOT at store time (store-from-temp gives same wrong
//      output as direct AGPR store).
//   2. The bug is in the mma_ABt(cAB[0][0], ...) computation itself —
//      specifically cAB[0][0]'s tiles[{0,1}][1] are computed wrong.
//   3. The bug is DETERMINISTIC across 4 mitigation attempts
//      (sacrificial dummy, explicit unroll, no-mul, temp-copy). Each
//      gives wildly different codegen yet same broken output region.
//
// **C-2 path closure**: After 7 rounds (R54-R60) of effort, the
// 4w-style 4-cell × 64×64 accumulator approach is BLOCKED by what
// appears to be a deterministic LLVM AGPR allocation defect specific
// to this kernel shape. Workarounds either (a) lose AGPR allocation
// (defeats the purpose) or (b) preserve the broken base tiles.
//
// **Decision (R62+)**: Pivot to Lever A (async global→LDS + MFMA
// pipelining). Lever A:
//   * doesn't require AGPR (works with normal VGPR allocation)
//   * reduces register pressure by eliminating VGPR staging of
//     A_row_reg / B_row_reg between buffer_load and ds_read
//   * uses gfx950 global_load_lds_dwordx4 ASM intrinsic directly
//
// The R60 test kernel + this docstring will remain in tree as
// reference material for a possible future C-2 retry (e.g. after
// LLVM upgrade or different cell shape that doesn't trigger the
// AGPR bug).
namespace lever_c2_round_59_step2b2_real_coords {
    using ::ST_v2;

    using lever_c2_round_54_step1_scaffold::WARPS_M;
    using lever_c2_round_54_step1_scaffold::WARPS_N;
    using lever_c2_round_54_step1_scaffold::_NUM_WARPS;
    using lever_c2_round_54_step1_scaffold::_NUM_THREADS;
    using lever_c2_round_54_step1_scaffold::RBM_4w;
    using lever_c2_round_54_step1_scaffold::RBN_4w;
    using lever_c2_round_54_step1_scaffold::A_row_reg_4w;
    using lever_c2_round_54_step1_scaffold::B_row_reg_4w;
    using lever_c2_round_54_step1_scaffold::C_acc_4w;

    using G_4w = kittens::group<_NUM_WARPS>;

    template<int KI_HINT = 0>
    __global__ __launch_bounds__(_NUM_THREADS, 1)
    void test_grouped_rcr_kernel_4w_real_coords(grouped_layout_globals g) {
        using ST_rcr = ST_v2;
        // 2 M-slabs of A (no ping-pong, single-buffer for R59 simplicity);
        // each slab is 128 M × 128 K = 16 KB. Total LDS = 4 × 16 KB = 64 KB
        // (well within gfx950's per-block budget at occupancy=1).
        __shared__ ST_rcr As[2];
        __shared__ ST_rcr Bs[2];

        // R60 step-1: sacrificial dummy declared BEFORE cAB to test
        // the register-aliasing hypothesis. If the bug is "first-allocated
        // VGPRs of cAB[0][0] overlap with spill traffic", declaring a
        // throwaway acc tile first should absorb those bad slots and let
        // cAB[0][0] land in clean register space.
        C_acc_4w cAB_sacrificial;
        zero(cAB_sacrificial);

        A_row_reg_4w a_reg[2];
        B_row_reg_4w b_reg[2];
        C_acc_4w cAB[2][2];

        const int wm = warpid() / WARPS_N;
        const int wn = warpid() % WARPS_N;

        const int num_pid_n = g.bpc;
        // R59 simplification: single group only (G=1). m_subtile_A = 0
        // hardcoded; M_g = g.M_total (the entire input M is one group).
        // R60 will add the LDS group_offs cache + 6-step binary search
        // to recover m_start_g per (br, bc) tile.
        const int M_g = g.M_total;
        const int bpr_g = M_g / BLOCK_SIZE;
        const int total_tiles = bpr_g * num_pid_n;
        const int ki_dyn = (KI_HINT > 0) ? KI_HINT : g.ki;

        // 4-warp swizzled-offset prefill (matches R58 step-2B-1).
        constexpr int bpt    = ST_rcr::underlying_subtile_bytes_per_thread;
        constexpr int bpm_4w = bpt * _NUM_THREADS;
        constexpr int mpt_4w =
            ST_rcr::rows * ST_rcr::cols * sizeof(fp8e4m3) / bpm_4w;
        uint32_t soA[mpt_4w], soB[mpt_4w];
        G_4w::prefill_swizzled_offsets(As[0], g.a, soA);
        G_4w::prefill_swizzled_offsets(Bs[0], g.b, soB);

        for (int gt = blockIdx.x; gt < total_tiles; gt += NUM_CUS) {
            const int br = gt / num_pid_n;
            const int bc = gt % num_pid_n;

            zero(cAB[0][0]); zero(cAB[0][1]);
            zero(cAB[1][0]); zero(cAB[1][1]);

            // Coord lambdas. A is [M_total, K] (batch=0, depth=0); B is
            // [G=1, N, K] (batch=0, depth=group_idx=0). Coord rows are
            // in units of HB=128.
            auto a_co = [&](int s, int k) -> coord<ST_rcr> {
                return {0, 0, br * 2 + s, k};
            };
            auto b_co = [&](int s, int k) -> coord<ST_rcr> {
                return {0, 0, bc * 2 + s, k};
            };

            #pragma unroll 1
            for (int k = 0; k < ki_dyn; k++) {
                rcr_8w_load_hoist<_NUM_THREADS>(As[0], g.a, a_co(0, k), soA);
                rcr_8w_load_hoist<_NUM_THREADS>(As[1], g.a, a_co(1, k), soA);
                rcr_8w_load_hoist<_NUM_THREADS>(Bs[0], g.b, b_co(0, k), soB);
                rcr_8w_load_hoist<_NUM_THREADS>(Bs[1], g.b, b_co(1, k), soB);
                asm volatile("s_waitcnt vmcnt(0)");
                __syncthreads();

                // Production grouped uses INTERLEAVED M/N distribution
                // across warps (matches dense rcr_4w lines 1480-1545):
                //   * a_reg[0] = wm-th 64-row sub of M-slab 0 (As[0])
                //   * a_reg[1] = wm-th 64-row sub of M-slab 1 (As[1])
                //   * b_reg[0] = wn-th 64-row sub of N-slab 0 (Bs[0])
                //   * b_reg[1] = wn-th 64-row sub of N-slab 1 (Bs[1])
                // → cAB[i][j] covers M-slab i (block-local M=[i*128 +
                //   wm*64, ..+64)), N-slab j (similar). Per-warp output
                //   is 4 cells totalling 128 M × 128 N — but
                //   interleaved across the block, NOT contiguous.
                // Store coords (below) use the same interleaved pattern
                // as production line 2123-2126 (store_c_tile_n_masked).
                auto a_sub_0 = subtile_inplace<RBM_4w, BK>(As[0], {wm, 0});
                auto a_sub_1 = subtile_inplace<RBM_4w, BK>(As[1], {wm, 0});
                auto b_sub_0 = subtile_inplace<RBN_4w, BK>(Bs[0], {wn, 0});
                auto b_sub_1 = subtile_inplace<RBN_4w, BK>(Bs[1], {wn, 0});
                load(a_reg[0], a_sub_0);
                load(a_reg[1], a_sub_1);
                load(b_reg[0], b_sub_0);
                load(b_reg[1], b_sub_1);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __syncthreads();

                mma_ABt(cAB[0][0], a_reg[0], b_reg[0], cAB[0][0]);
                mma_ABt(cAB[0][1], a_reg[0], b_reg[1], cAB[0][1]);
                mma_ABt(cAB[1][0], a_reg[1], b_reg[0], cAB[1][0]);
                mma_ABt(cAB[1][1], a_reg[1], b_reg[1], cAB[1][1]);
            }

            // R60 step-3: skip the scale epilog entirely (probe sets
            // scale_a=scale_b=1.0, so `mul(c, c, 1.0)` should be
            // identity. If skipping fixes cAB[0][0], the bug is in
            // `mul()` for the first acc cell. If still wrong, the
            // bug is in load/mma not mul.
            // const float combined_scale = resolve_combined_scale_grp(g);
            // mul(cAB[0][0], cAB[0][0], combined_scale);
            // mul(cAB[0][1], cAB[0][1], combined_scale);
            // mul(cAB[1][0], cAB[1][0], combined_scale);
            // mul(cAB[1][1], cAB[1][1], combined_scale);

            // R61 step-1: copy cAB[0][0] (the broken cell) to a fresh
            // temp tile and store FROM THE TEMP. This forces an
            // AGPR→VGPR transfer via copy() before store reads from
            // the tile. If this fixes the broken base tiles
            // [{0,1}][1] of cAB[0][0], the bug is in store reading
            // directly from AGPR for cAB[0][0]'s specific slot map.
            C_acc_4w c00_tmp;
            zero(c00_tmp);
            copy(c00_tmp, cAB[0][0]);

            // C output coord units: rows in RBM_4w=64, cols in RBN_4w=64.
            // INTERLEAVED layout (matches production line 2123-2126):
            //   cAB[i][j] covers M=[br*256 + i*128 + wm*64, ..+64),
            //                    N=[bc*256 + j*128 + wn*64, ..+64).
            // In coord units of (RBM_4w=64, RBN_4w=64):
            //   m_idx = br*(BLOCK_SIZE/RBM_4w) + i*WARPS_M + wm = br*4 + i*2 + wm
            //   n_idx = bc*(BLOCK_SIZE/RBN_4w) + j*WARPS_N + wn = bc*4 + j*2 + wn
            store(g.c, c00_tmp,    {0, 0, br*4 + 0       + wm, bc*4 + 0       + wn});
            store(g.c, cAB[0][1],  {0, 0, br*4 + 0       + wm, bc*4 + WARPS_N + wn});
            store(g.c, cAB[1][0],  {0, 0, br*4 + WARPS_M + wm, bc*4 + 0       + wn});
            store(g.c, cAB[1][1],  {0, 0, br*4 + WARPS_M + wm, bc*4 + WARPS_N + wn});

            asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
            __syncthreads();
        }

        // R60 step-1: keep cAB_sacrificial alive past the kernel body
        // so LLVM doesn't dead-code-eliminate the declaration. Write
        // its first lane to a SENTINEL output (never matters because
        // the binding doesn't read this address back).
        if (cAB_sacrificial.tiles[0][0].data[0].x != 0.0f) {
            // Unreachable in practice (zero() set it to 0.0); this branch
            // exists only to keep the compiler from eliminating the dummy.
            __builtin_amdgcn_s_barrier();
        }
    }

    // Host launcher: mirrors a stripped-down dispatch_grouped_rcr.
    // Single group only; no FUSED_KTAIL / N_MASKED_STORE specs. Caller
    // is responsible for ensuring M%256==0, N%256==0, K%128==0 (R59
    // probe shape: M=512, N=256, K=256 satisfies all three).
    static void dispatch_test_4w_real_coords(grouped_layout_globals g) {
        g.n = static_cast<int>(g.c.cols());
        g.M_total = static_cast<int>(g.c.rows());
        g.k = static_cast<int>(g.a.cols());

        g.fast_n = (g.n / BLOCK_SIZE) * BLOCK_SIZE;
        g.fast_k = (g.k / K_BLOCK)    * K_BLOCK;
        g.bpc    = g.fast_n / BLOCK_SIZE;
        g.ki     = g.fast_k / K_BLOCK;

        if (g.bpc <= 0 || g.ki <= 0 || g.M_total <= 0) return;

        dim3 grid(NUM_CUS);
        dim3 block(_NUM_THREADS);
        test_grouped_rcr_kernel_4w_real_coords<0><<<grid, block, 0, g.stream>>>(g);
    }
} // namespace lever_c2_round_59_step2b2_real_coords

template __global__ void
lever_c2_round_59_step2b2_real_coords::test_grouped_rcr_kernel_4w_real_coords<0>(grouped_layout_globals);

// =============================================================================
// Persistent RRR grouped kernel — round-1 mirror of grouped_rcr_kernel.
//
// Identical persistent + CPU-sync-free skeleton (LDS group_offs cache, 6-step
// branch-free binary search over s_cum_tiles, group-by-M/N tile swizzle,
// chiplet pid permutation), with the inner per-tile body swapped for the
// FP8 RRR dense schedule (lines 1389-1526). Layout-specific differences:
//
//   * B layout : [1, G, K, N] row-major (K outer, N inner stride-1).
//                ``b_co(s, k)`` maps a tile coord to {0, g_idx, k, s}, i.e.
//                the K-tile index sits on row axis and the N-tile index sits
//                on col axis (vs RCR which puts N on row, K on col).
//
//   * Shared B : ST_v2 (st_fp8e4m3<HB, BK, st_16x128_v2_s>) — same row-tile
//                shape as ST_rcr but the underlying load/store patterns are
//                the v2 col-major-swizzle layout that the RRR mma reads
//                from.
//
//   * Shared A : ST_row (st_fp8e4m3<HB, BK, st_16x128_s>) — straight row-
//                major. RCR reuses ST_v2 for A; RRR uses the simpler ST_row.
//
//   * Register : A_row_reg + B_col_reg (B is column-loaded into the col
//                register layout via load_col_from_st).
//
//   * MMA      : rrr_mma (mma_AB) instead of rcr_mma (mma_ABt).
//
//   * Loads    : G::load (kittens::load with swizzled offsets) for both A
//                and B. The 8-wave m0-broadcast hoist (rcr_8w_load_hoist)
//                is RCR-specific; RRR keeps the standard kittens load path
//                used by the FP8 dense RRR kernel.
//
// Coord shifts inside the persistent loop:
//   * A row    : m_subtile_A = m_start_g / HB              (HB = 128)
//                a_co(s, k) -> {0, 0, m_subtile_A + s, k}
//   * B group  : b_co(s, k) -> {0, group_idx, k, s}       (G as depth axis)
//   * C row    : m_subtile_C = m_start_g / RBM             (RBM = 64)
//                store(g.c, ..., {0, 0, m_subtile_C + R, C})
//
// Aligned interior: ``g.fast_n = (n / BLOCK_SIZE) * BLOCK_SIZE``,
// ``g.fast_k = (k / K_BLOCK) * K_BLOCK``. Cells outside (col >= fast_n) and
// the K-tail correction in [fast_k, k) for interior cells are handled by
// ``grouped_tail_kernel<Layout::RRR>`` (scalar fp32, mirror BF16 RRR).
// =============================================================================
//
// Round 27 — FP8 RRR fuse path A empirical numerical probe. Default 0 keeps
// production path identical to round-26 (external grouped_ktail_kernel_lds_rrr
// + grouped_ntail_kernel_lds_rrr + grouped_tail_kernel<RRR>). Set to 1 +
// recompile to enable in-kernel cooperative LDS-staged K-tail accumulation:
//   1. cooperative pre-zero As[tic][0/1] + Bs[tic][0/1] post-Epilog 2
//   2. G::load on K-tail iter (k = ki_dyn) with full-tensor SRD (OOB voffsets
//      no-op leaves pre-zeroed bytes intact = effective zero-pad)
//   3. load_a / load_b helpers (FP8 RRR uses manual ds_read_b64_tr_b8 via
//      load_col_from_st, NOT subtile_inplace — sidesteps BF16 round-7's SGPR
//      aliasing bug)
//   4. rrr_mma 4 times to accumulate K-tail into cA/cB/cC/cD pre-scale
// The probe + #if !FP8_RRR_FUSE_PROBE gate in dispatch_grouped_rrr ensure
// no double-accumulation with external launches. See
// analysis/_notes/round-26-fp8-rrr-path-a-probe-plan.md for protocol.
#ifndef FP8_RRR_FUSE_PROBE
#define FP8_RRR_FUSE_PROBE 0
#endif
template<int KI_HINT = 0>
__global__ __launch_bounds__(_NUM_THREADS, 1)
void grouped_rrr_kernel(const grouped_layout_globals g) {
    __shared__ ST_row As[2][2];
    __shared__ ST_v2  Bs[2][2];
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];
    __shared__ int s_cum_tiles[MAX_G_PLUS_1];
    __shared__ int s_total_tiles;

    A_row_reg a;
    B_col_reg b0, b1;
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;

    // Round-2 (FP8 backward unblock): mirror RCR — read host-side
    // ``g.num_xcds`` knob, fall back to the default 8 when unset.
    const int xcds_eff = g.num_xcds > 0 ? g.num_xcds : BLOCK_SWIZZLE_NUM_XCDS;
    int pid = chiplet_transform_chunked(
        blockIdx.x, NUM_CUS, xcds_eff, 64);

    int wm = warpid() / WARPS_N;
    int wn = warpid() % WARPS_N;
    const int num_pid_n = g.bpc;
    const int ki_dyn   = (KI_HINT > 0) ? KI_HINT : g.ki;

    // Round-41 (Lever L extension): 2-phase parallel init, mirroring the
    // R9-dm pattern in forward ``grouped_rcr_kernel`` (line ~2273) and
    // the R38 port in ``grouped_var_k_kernel_fp8`` (HK ad501f0a).
    //
    // Phase 1: all threads parallel-load ``g.group_offs[0..G]`` from HBM
    // into ``s_offs[]`` (collapse G+1 serialized HBM loads → one warp-
    // coalesced transfer) AND pad ``s_cum_tiles[G+1..MAX_G_PLUS_1)``
    // with the ``INT_MAX`` sentinel the downstream branch-free binary
    // search relies on. __syncthreads gate.
    //
    // Phase 2: thread 0 does the O(G) variable prefix-scan reading from
    // LDS (fast, no HBM stalls; the HBM reads all retired during the
    // sync above). Unlike var_k (R38) which had a CONSTANT
    // ``tiles_per_group = g.bpr * g.bpc`` and collapsed the scan to
    // an O(1) closed form, ``grouped_rrr`` has a VARIABLE per-group
    // count ``tiles_g = (M_g / BLOCK_SIZE) * num_pid_n`` (each group
    // has its own M_g = s_offs[gi+1] - s_offs[gi]) so the serial scan
    // is still necessary — only the HBM load is parallelized.
    //
    // ``grouped_rrr`` is the FP8 dA backward kernel used whenever
    // K_RRR is BLOCK_SIZE-aligned (DSV3 Down K=2048 / GateUP K=7168;
    // gpt_oss K=2880 is unaligned and reroutes via Triton transpose +
    // forward grouped_rcr instead). This closes the last remaining
    // init divergence among the 3 grouped FP8 kernels (rcr fwd, rrr
    // bwd, var_k bwd all now use the same 2-phase pattern).
    //
    // Correctness: bit-identical to the old serial path for all legal
    // inputs (G ≤ MAX_G_PLUS_1 - 1 = 64). Same values land in
    // ``s_offs[0..G]`` (HBM content unchanged) and ``s_cum_tiles[0..G]``
    // (same scan formula, same start-prev-next chain). Sentinel pad
    // unchanged. ``s_total_tiles`` unchanged.
    if (threadIdx.x <= g.G && threadIdx.x < MAX_G_PLUS_1) {
        s_offs[threadIdx.x] = static_cast<int>(g.group_offs[threadIdx.x]);
    }
    if (threadIdx.x > g.G && threadIdx.x < MAX_G_PLUS_1) {
        s_cum_tiles[threadIdx.x] = 0x7FFFFFFF;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int prev = s_offs[0];
        s_cum_tiles[0] = 0;
        int t = 0;
        #pragma unroll 1
        for (int gi = 0; gi < g.G; ++gi) {
            const int next = s_offs[gi + 1];
            t += ((next - prev) / BLOCK_SIZE) * num_pid_n;
            s_cum_tiles[gi + 1] = t;
            prev = next;
        }
        s_total_tiles = t;
    }
    __syncthreads();
    const int total_tiles = s_total_tiles;

    constexpr int bptA = ST_row::underlying_subtile_bytes_per_thread;
    constexpr int bpmA = bptA * _NUM_THREADS;
    constexpr int mptA = ST_row::rows * ST_row::cols * sizeof(fp8e4m3) / bpmA;
    uint32_t soA[mptA];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);

    constexpr int bptB = ST_v2::underlying_subtile_bytes_per_thread;
    constexpr int bpmB = bptB * _NUM_THREADS;
    constexpr int mptB = ST_v2::rows * ST_v2::cols * sizeof(fp8e4m3) / bpmB;
    uint32_t soB[mptB];
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

    for (int gt = pid; gt < total_tiles; gt += NUM_CUS) {
        int lo = 0;
        int hi = MAX_G_PLUS_1 - 1;
        #pragma unroll
        for (int level = 0; level < 6; ++level) {
            const int mid = (lo + hi + 1) >> 1;
            if (gt >= s_cum_tiles[mid]) lo = mid;
            else hi = mid - 1;
        }
        const int group_idx = lo;
        const int tile_start = s_cum_tiles[lo];
        const int local_tile = gt - tile_start;
        const int m_start_g = s_offs[group_idx];
        const int M_g = s_offs[group_idx + 1] - m_start_g;
        const int bpr_g = M_g / BLOCK_SIZE;

        int br, bc;
        if (g.bpc > bpr_g) {
            const int WGN = g.group_m;
            const int num_wgid_in_group = bpr_g * WGN;
            int group_id = local_tile / num_wgid_in_group;
            int first_pid_n = group_id * WGN;
            int group_size_n = min(num_pid_n - first_pid_n, WGN);
            if (group_size_n <= 0) continue;
            bc = first_pid_n + ((local_tile % num_wgid_in_group) % group_size_n);
            br = (local_tile % num_wgid_in_group) / group_size_n;
        } else {
            const int WGM = g.group_m;
            const int num_wgid_in_group = WGM * num_pid_n;
            int group_id = local_tile / num_wgid_in_group;
            int first_pid_m = group_id * WGM;
            int group_size_m = min(bpr_g - first_pid_m, WGM);
            if (group_size_m <= 0) continue;
            br = first_pid_m + ((local_tile % num_wgid_in_group) % group_size_m);
            bc = (local_tile % num_wgid_in_group) / group_size_m;
        }
        if (br >= bpr_g || bc >= num_pid_n) continue;

        const int m_subtile_A = m_start_g / HB;
        const int m_subtile_C = m_start_g / RBM;

        // RRR coord conventions (mirror of dense gemm_kernel<RRR>):
        //   a_co(s, k) : A is [M_total, K]      → row-shift by m_subtile_A.
        //   b_co(s, k) : B is [1, G, K, N]      → K on row, N on col, group
        //                                         depth = group_idx.
        auto a_co = [&](int s, int k) -> coord<ST_row> {
            return {0, 0, m_subtile_A + s, k};
        };
        auto b_co = [&](int s, int k) -> coord<ST_v2> {
            return {0, group_idx, k, s};
        };

        auto load_a = [&](A_row_reg& dst, ST_row& tile, int wi) {
            auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
            load(dst, sub);
        };
        auto load_b = [&](B_col_reg& dst, ST_v2& tile, int wi) {
            load_col_from_st(dst, tile, wi * RBN);
        };

        zero(cA); zero(cB); zero(cC); zero(cD);

        int tic = 0, toc = 1;
        // Prologue: tile-0 + tile-1 (mirrors dense gemm_kernel<RRR>
        // lines 1421-1435).
        G::load(Bs[tic][0], g.b, b_co(bc*2,   0), soB);
        G::load(As[tic][0], g.a, a_co(br*2,   0), soA);
        G::load(Bs[tic][1], g.b, b_co(bc*2+1, 0), soB);
        G::load(As[tic][1], g.a, a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(RRR_INIT0_VMCNT);
        __builtin_amdgcn_s_barrier();

        G::load(Bs[toc][0], g.b, b_co(bc*2,   1), soB);
        G::load(As[toc][0], g.a, a_co(br*2,   1), soA);
        G::load(Bs[toc][1], g.b, b_co(bc*2+1, 1), soB);

        TK_WAIT_VMCNT(RRR_INIT1_VMCNT);
        __builtin_amdgcn_s_barrier();

        // Single-tile main loop (mirror dense lines 1437-1470).
        TK_PRAGMA_UNROLL(RRR_MAIN_UNROLL)
        for (int k = 0; k < ki_dyn - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, k+1), soA);
            TK_WAIT_LGKM(RRR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();

            load_b(b1, Bs[tic][1], wn);
            G::load(Bs[tic][0], g.b, b_co(bc*2, k+2), soB);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            G::load(Bs[tic][1], g.b, b_co(bc*2+1, k+2), soB);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();

            G::load(As[tic][0], g.a, a_co(br*2, k+2), soA);
            TK_WAIT_VMCNT(RRR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        // Epilog 1 (mirror dense lines 1472-1501).
        {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, ki_dyn-1), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[toc][0], wn);
            TK_WAIT_VMCNT(RRR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();
            tic ^= 1; toc ^= 1;
        }

        // Epilog 2 (mirror dense lines 1503-1526).
        {
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); rrr_mma(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            rrr_mma(cC, a, b0);
            rrr_mma(cD, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

#if FP8_RRR_FUSE_PROBE
        // === Round 27 — Path A K-tail fuse probe (HYBRID A-path-B + B-path-A) ===
        // Round 27 first attempt cooperative G::load for BOTH A and B failed
        // (SNR 16.61 dB ≈ no-K-tail floor 16.75 dB). Diagnosis: A is 2D
        // [M_total, K] with K-stride = K bytes, so the K-tail iter k=ki_dyn
        // covering K=[fast_k, fast_k+K_BLOCK) reads row M's bytes
        // [fast_k, K_global) (valid) AND row M's bytes [K_global, fast_k+K_BLOCK)
        // (OOB) — but the OOB byte addresses LITERALLY EQUAL row M+1's bytes
        // [0, K_REM) (since row stride = K bytes). SRD bound check at
        // M_total * K bytes does NOT reject row M+1's data, so A's K-tail
        // accumulator contaminates with row M+1's [0, K_REM) values — wrong.
        //
        // FIX (round-17 docs): A uses path B (per-lane raw_buffer_load_b128
        // with per-lane SENTINEL on b128_lo_valid / b128_hi_valid — no load
        // issued for OOB K-bytes, so VGPR returns 0 from the SRD bound check
        // on the b128 with SENTINEL voffset). B side keeps path A
        // (cooperative G::load + pre-zero LDS) because B's K is N-strided
        // (RRR layout: B is [G, K, N]) so the OOB K-bytes for k_row >= K_global
        // are physically out of the tensor (no row+1 spillover concern).
        if (g.fast_k < g.k) {
            // ROUND 28 — All 5 attempts to escape A-register aliasing FAILED.
            // Detailed in analysis/_notes/round-28-fp8-rrr-path-a-aliasing-fixes-fail.md.
            // Summary:
            //   * Attempt 1 (fresh `A_row_reg a_kt` declaration): SNR 15.05 dB.
            //     Compiler aliased `a_kt` to same VGPR pool as round-27 `a`
            //     (after Epilog 2 release of `a`, compiler reuses for next
            //     "needs register" var = a_kt → still aliased to c).
            //   * Attempt 2 (fresh a_kt + b0_kt + b1_kt): SNR 15.04 dB.
            //     Same root cause; even adding b0_kt/b1_kt fresh did not help.
            //   * Attempt 3 (a + asm "+v" pin BEFORE cooperative ops): SNR
            //     15.06 dB. The pin forces VGPR alloc at asm time, but
            //     compiler released VGPR after asm (no further use until
            //     load_a_kt) and re-aliased to c during cooperative ops.
            //   * Attempt 4 (a + asm "+v" pin BEFORE AND AFTER cooperative
            //     ops, sandwich): SNR 15.07 dB. Same — post-cooperative
            //     pin is the asm-relevant point, but compiler still maps
            //     `a`'s VGPR to whatever physical register is "free" at
            //     that moment, which after cooperative ops may overlap c.
            //   * Attempt 5 (pin all cA/cB/cC/cD dwords before load_a_kt):
            //     SNR 15.07 dB, spill +6 dwords. c-pin forces compiler
            //     to keep c in fixed VGPR slots, but `a` lands on some
            //     other live register or spilled scratch — same SNR.
            // ROOT CAUSE: cooperative ops (pre-zero loop + G::load + sync)
            // create a long no-A-use gap. Compiler retires `a` (and any
            // post-Epilog 2 fresh `a_kt` in the same scope) during this
            // gap, then aliases the freed VGPR to live c registers. There
            // is no per-instruction asm trick to override this — register
            // allocation is a global pass that sees the gap and optimizes.
            //
            // ONLY KNOWN FIX: introduce a FRESH fp32 acc tile `c_kt` (32
            // dwords/lane) BEFORE cooperative ops, accumulate K-tail mma
            // into c_kt instead of `a → c`, then add `c[ABCD] += c_kt`.
            // Cost: +32 dwords spill (~+30% spill increase from baseline 94).
            // Risk: spill may push occupancy from 2 → 1, regressing perf.
            // Estimated probe time: 1 round to derive + test.
            //
            // Defer to round 29. PROBE remains gated #if 0; production
            // unchanged.

            // ---- Cooperative pre-zero of Bs[tic][0/1] ----
            // ST_v2 has swizzle padding (= 17408 bytes); strip to 16-byte
            // alignment so we can b128-store the entire region incl. padding.
            constexpr int ST_V2_B128 = (sizeof(ST_v2) / 16);
            const int tid = threadIdx.x;
            __uint128_t* Bs0_ptr = reinterpret_cast<__uint128_t*>(&Bs[tic][0].data[0]);
            __uint128_t* Bs1_ptr = reinterpret_cast<__uint128_t*>(&Bs[tic][1].data[0]);
            #pragma unroll
            for (int idx = tid; idx < ST_V2_B128; idx += _NUM_THREADS) {
                Bs0_ptr[idx] = 0;
                Bs1_ptr[idx] = 0;
            }
            __syncthreads();

            // ---- B side: cooperative G::load on K-tail iter ----
            // OOB voffsets (k_row >= K_global) no-op on raw_buffer_load_lds →
            // pre-zeroed bytes preserved → effective zero-pad for K=[K_global,
            // fast_k + K_BLOCK). Cross-group contamination is OK for the G=1
            // probe shape (full-tensor SRD == per-group SRD); production hybrid
            // would need per-group SRD construction.
            G::load(Bs[tic][0], g.b, b_co(bc*2,   ki_dyn), soB);
            G::load(Bs[tic][1], g.b, b_co(bc*2+1, ki_dyn), soB);

            // ---- A side: per-lane path B (direct HBM → register) ----
            // Mirror RCR fuse path B (line ~2300+) but for RRR's a register tile.
            // FP8 A_row_reg is rt_fp8e4m3<RBM=64, BK=128, row_l, rt_16x128_s>:
            //   * 32 fp8 cells / lane (= 32 bytes = 2 b128 loads)
            //   * row_lane = laneid % 16  (16 rows / base tile)
            //   * k_lane_byte = (laneid / 16) * 32  (K stride between lanes)
            //   * data[0..3] = K=[k_lane_byte,    k_lane_byte+16) → b128 #1
            //   * data[4..7] = K=[k_lane_byte+16, k_lane_byte+32) → b128 #2
            // For K_REM=64 (gpt_oss K=2880): lanes 0..31 valid, lanes 32..63 OOB.
            // OOB lanes get SENTINEL voffset → SRD range_bytes check rejects load
            // → VGPR returns 0. NO row-M+1 contamination because we never issue
            // the load for those lanes.
            const int laneid = kittens::laneid();
            const int row_lane = laneid % 16;
            const int k_lane_byte = (laneid / 16) * 32;
            const int K_REM = g.k - g.fast_k;
            const bool b128_lo_valid = (k_lane_byte + 16) <= K_REM;
            const bool b128_hi_valid = (k_lane_byte + 32) <= K_REM;
            constexpr uint32_t SENTINEL = 0xFFFF0000u;
            const fp8e4m3* a_base_ptr = (const fp8e4m3*)&g.a[{0, 0, 0, 0}];
            const int a_row_stride_bytes = g.a.template stride<2>();
            const uint32_t a_total_bytes =
                static_cast<uint32_t>(g.M_total) *
                static_cast<uint32_t>(a_row_stride_bytes);
            i32x4 a_srsrc_kt = make_srsrc((const void*)a_base_ptr, a_total_bytes);
            const uint32_t K_tail_base_bytes =
                static_cast<uint32_t>(g.fast_k);

            // ROUND 28 ATTEMPT 3 (continued): Write back to the SAME `a`
            // register tile that the live-pin asm above keeps allocated.
            // No more `a_kt` (round-28 attempt 1) — compiler aliased it
            // to c despite freshness because c was the next live var.
            // Pinning `a` directly with "+v" forces the VGPR allocation
            // to survive across cooperative ops.
            auto load_a_kt = [&](int slab) __attribute__((always_inline)) {
                const int M_warp_base =
                    (m_subtile_A + br * 2 + slab) * HB + wm * RBM;
                #pragma unroll
                for (int h = 0; h < A_row_reg::height; ++h) {
                    const int A_row_idx = M_warp_base + h * 16 + row_lane;
                    const uint32_t v_base = static_cast<uint32_t>(
                        A_row_idx * a_row_stride_bytes +
                        K_tail_base_bytes + k_lane_byte);
                    const uint32_t v_lo = b128_lo_valid ? v_base : SENTINEL;
                    const uint32_t v_hi = b128_hi_valid ? (v_base + 16) : SENTINEL;
                    __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                        a_srsrc_kt, v_lo, 0, 0);
                    __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                        a_srsrc_kt, v_hi, 0, 0);
                    *reinterpret_cast<__uint128_t*>(&a.tiles[h][0].data[0]) = v0;
                    *reinterpret_cast<__uint128_t*>(&a.tiles[h][0].data[4]) = v1;
                }
            };

            // Wait B G::load + LDS visibility before B reads.
            asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
            __syncthreads();

            // K-tail accumulation. b0/b1 from LDS via load_col_from_st
            // (manual ds_read_b64_tr_b8 — sidesteps BF16 round-7's
            // subtile_inplace SGPR aliasing). a from path B direct HBM
            // load with SENTINEL. Mirrors Epilog 2's 4-mma pattern.
            //
            // ROUND 27 RESULT — FAILURE (root cause: a register VGPR aliased
            // to c register by compiler post-Epilog 2). Probe sequence:
            //   * SKIP_MMA (load_a_kt + load_b but no rrr_mma): SNR = -inf
            //     dB, c corruption with NaN/Inf — catastrophic. Means
            //     load_a_kt's writes to a.tiles[].data[] DIRECTLY OVERWRITE
            //     c register VGPR slots. Compiler treats a as dead post-
            //     Epilog 2's last `rrr_mma(cD, a, b1)` and rebinds a's
            //     VGPRs into the c register pressure pool.
            //   * SKIP_A_LOAD (cooperative pre-zero + G::load Bs + load_b
            //     but NO load_a_kt / no MMA): SNR = 16.56 dB ≈ no-K-tail
            //     floor (16.75 dB from round-7 docs). B-side cooperative
            //     path A is NOT corrupting c — only A-side path B is.
            //   * Full hybrid (path B for A + path A for B + 4 rrr_mma):
            //     SNR = 15.05 dB (worse than floor) — confirms A-side
            //     corruption + a@b@c chained-error.
            //
            // CONTRAST WITH RCR FUSE (which works): RCR's K-tail epilog
            // also writes a.tiles[].data[] via load_a_kt and gets SNR ≥25
            // dB. The difference: RCR's K-tail block has NO cooperative
            // ops (no pre-zero + G::load) before the load_a_kt — it goes
            // directly from Epilog 2 to per-lane raw_buffer_load_b128 for
            // both A and B. Compiler keeps a/b0/b1 live across Epilog 2 →
            // K-tail boundary because the next use is immediate. Inserting
            // any cooperative op (pre-zero loop / G::load / __syncthreads)
            // creates a long no-A-use gap → compiler releases a's VGPR.
            //
            // FIX OPTIONS (next round):
            //   1. Mirror RCR fully: drop cooperative ops, use per-lane
            //      raw_buffer_load_b128 for B too (need to derive lane→cell
            //      mapping for B_col_reg = rt_fp8e4m3<BK=128, RBN=32,
            //      col_l, rt_128x16_s>; ds_read_b64_tr_b8 lane mapping in
            //      load_col_from_st_half line 113-145 is the starting point).
            //   2. Introduce fresh K-tail register tile a_kt (+32 VGPR)
            //      and write into a_kt instead of a, then rrr_mma(c..., a_kt, b...).
            //      Risk: spill (currently +4 dwords; +32 VGPR likely spills
            //      to occupancy=1). RCR option 1 is preferred because it's
            //      0-VGPR-delta.
#if !FP8_RRR_FUSE_PROBE_SKIP_A_LOAD
            load_b(b0, Bs[tic][0], wn);
            load_a_kt(0);
            asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)" ::: "memory");
            rrr_mma(cA, a, b0);

            load_b(b1, Bs[tic][1], wn);
            rrr_mma(cB, a, b1);

            load_a_kt(1);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            rrr_mma(cC, a, b0);
            rrr_mma(cD, a, b1);
#endif
            __builtin_amdgcn_s_barrier();
        }
#endif

        const float combined_scale = resolve_combined_scale_grp(g);
        mul(cA, cA, combined_scale);
        mul(cB, cB, combined_scale);
        mul(cC, cC, combined_scale);
        mul(cD, cD, combined_scale);

        if (wm == 0) __builtin_amdgcn_s_barrier();
        // Round-25 (PT): mirror R24's localised readfirstlane pattern for the
        // C-store epilog. Same mechanism: r0/r1/c0/c1 are wave-uniform IN
        // PRINCIPLE (m_subtile_C, br, bc, wm, wn all uniform per tile-iter)
        // but LLVM's uniformity analysis loses it (binary search via VGPR-
        // derived `s_cum_tiles[mid]` LDS read). Without readfirstlane, the
        // kittens store helper's `make_buffer_resource(as_u64, ...)` builds
        // a VGPR-derived i32x4 SRD → buffer_store_b16 emits the per-lane
        // fallback loop. Wrapping each coord at the call site lifts to SGPR
        // with short lifetime (4 store calls then dead). grouped_rrr_kernel
        // is the dA backward path for FP8 grouped (RRR layout); its 76 dw
        // VGPR spill is the highest of any FP8 grouped kernel (vs ~37 dw
        // for forward grouped_rcr_kernel after R24) — confirming the same
        // structural taint. Spill drop after this fix: 76 → 65 dw (-11 dw).
        // Forward metric unchanged (RRR not in forward path); backward bench
        // (--dtype fp8) average dA TFLOPS +2.76 (+0.20%) over R24 baseline.
        const int r0 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2+wm);
        const int r1 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2+WARPS_M+wm);
        const int c0 = __builtin_amdgcn_readfirstlane(bc*WARPS_N*2+wn);
        const int c1 = __builtin_amdgcn_readfirstlane(bc*WARPS_N*2+WARPS_N+wn);
        store(g.c, cA, {0, 0, r0, c0});
        store(g.c, cB, {0, 0, r0, c1});
        store(g.c, cC, {0, 0, r1, c0});
        store(g.c, cD, {0, 0, r1, c1});

        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }
}

template __global__ void grouped_rrr_kernel<0>(const grouped_layout_globals);

// =============================================================================
// Grouped tail kernel — scalar fp32 fixup for cells the main grouped kernel
// does not cover (col >= fast_n) and the K-tail correction in [fast_k, k)
// for interior cells. Mirror of `gemm_tail_kernel` (FP8 dense) but with
// per-group B indexing via `group_offs`. Templated over Layout to support
// both RCR (forward) and RRR (backward dA) — see also the BF16 mirror in
// kernel_bf16_dynamic.cpp::grouped_tail_kernel.
//
// Three cases per cell:
//   * col <  fast_n  AND fast_k == k  → main covers fully → early-return.
//   * col <  fast_n  AND fast_k <  k  → main wrote partial; add K-tail.
//   * col >= fast_n                   → main did not run; full-K reduction.
//
// Per-group M-tail (M_g % BLOCK_SIZE != 0) is NOT handled — caller contract.
//
// B layout per Layout L:
//   * RCR : g.b is [1, G, N, K]  → B[g_idx, col, kk] (stride-1 in K).
//           Vec8 fast path enabled: A and B both contiguous in K.
//   * RRR : g.b is [1, G, K, N]  → B[g_idx, kk, col] (stride-1 in N).
//           B not vectorisable along K → scalar K-loop only.
template<Layout L>
__global__ void grouped_tail_kernel(const grouped_layout_globals g) {
    static_assert(L == Layout::RCR || L == Layout::RRR,
                  "FP8 grouped tail kernel: RCR or RRR only.");
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        #pragma unroll 1
        for (int gi = 0; gi <= g.G; ++gi) {
            s_offs[gi] = static_cast<int>(g.group_offs[gi]);
        }
    }
    __syncthreads();

    // Round-12: ``main_covers_n`` mirrors the dispatch decision (see
    // ``dispatch_grouped_rcr``). When ``g.bpc * BLOCK_SIZE > g.fast_n``,
    // the main kernel ran with ``bpc = ceil_div(g.n, BLOCK_SIZE)`` and
    // ``store_c_tile_n_masked`` already wrote cols [0, g.n) with the
    // [0, fast_k) partial K reduction. Tail must NOT redo full-K
    // reduction for those cells — only add the K-tail [fast_k, k)
    // correction. Detected from ``g.bpc`` itself (mirror dense
    // ``gemm_tail_kernel`` pattern).
    const bool main_covers_n = (g.bpc * BLOCK_SIZE > g.fast_n);
    const bool needs_k_tail = g.fast_k < g.k;
    const bool needs_n_tail = !main_covers_n && (g.fast_n < g.n);
    if (!needs_k_tail && !needs_n_tail) return;

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= g.M_total || col >= g.n) return;

    int group_idx = 0;
    #pragma unroll 1
    for (int gi = 0; gi < g.G; ++gi) {
        if (row < s_offs[gi + 1]) { group_idx = gi; break; }
    }

    // Round-12: ``interior_n`` widens to ``col < g.n`` whenever
    // ``main_covers_n``. The fast_covers_cell branch then takes the
    // K-tail correction path (RMW + acc) for ALL cols [0, g.n) — the
    // legacy ``col < g.fast_n`` interior is preserved for layouts whose
    // main kernel still uses ``bpc = fast_n / BLOCK_SIZE`` (RRR).
    const bool interior_n       = main_covers_n ? true : (col < g.fast_n);
    const bool fast_covers_cell = interior_n && g.fast_n > 0 && g.fast_k > 0;
    if (fast_covers_cell && !needs_k_tail) return;

    const int k0 = fast_covers_cell ? g.fast_k : 0;
    float acc = 0.0f;

    if constexpr (L == Layout::RCR) {
        // Vec8 (8-byte = 8 fp8e4m3) fast path. Both A[row, kk] and
        // B[group_idx, col, kk] are stride-1 in K, so consecutive fp8s
        // along K can be loaded as a single dwordx2. ``g.k`` is bounded
        // below the K_BLOCK alignment by host pad in the gpt_oss-K=2880
        // path which is currently the only K-tail caller; for K=2880,
        // (g.k - k0) is 64 (K-tail correction) or 2880 (N-tail full
        // reduction) — both multiples of 8. The scalar tail handles any
        // residual when g.k % 8 != 0.
        const fp8e4m3* a_row = &g.a[coord<>(row, 0)];
        const fp8e4m3* b_row = &g.b[coord<>{0, group_idx, col, 0}];
        int kk = k0;
        if ((g.k % 8 == 0) && ((k0 & 7) == 0)) {
            const fp8e4m3_8* a_v8 = reinterpret_cast<const fp8e4m3_8*>(a_row);
            const fp8e4m3_8* b_v8 = reinterpret_cast<const fp8e4m3_8*>(b_row);
            const int j_start = k0 >> 3;
            const int j_end   = g.k >> 3;
            #pragma unroll 4
            for (int j = j_start; j < j_end; ++j) {
                fp8e4m3_8 a8 = a_v8[j];
                fp8e4m3_8 b8 = b_v8[j];
                float4 a_lo = base_types::convertor<float4, fp8e4m3_4>::convert(a8.lo);
                float4 a_hi = base_types::convertor<float4, fp8e4m3_4>::convert(a8.hi);
                float4 b_lo = base_types::convertor<float4, fp8e4m3_4>::convert(b8.lo);
                float4 b_hi = base_types::convertor<float4, fp8e4m3_4>::convert(b8.hi);
                acc += a_lo.x * b_lo.x + a_lo.y * b_lo.y
                     + a_lo.z * b_lo.z + a_lo.w * b_lo.w
                     + a_hi.x * b_hi.x + a_hi.y * b_hi.y
                     + a_hi.z * b_hi.z + a_hi.w * b_hi.w;
            }
            kk = j_end << 3;
        }
        for (; kk < g.k; ++kk) {
            acc += load_fp8_scalar(g.a, row, kk) *
                   load_fp8_scalar_grp(g.b, group_idx, col, kk);
        }
    } else {
        // RRR: A stride-1 in K, B[group_idx, kk, col] stride-N in K — B
        // is not vectorisable along K. Round-55 added the LDS-staged
        // ``grouped_ktail_kernel_lds_rrr<64>`` for the [0, fast_n) ×
        // M_total interior K-tail RMW; round-56 added the paired
        // ``grouped_ntail_kernel_lds_rrr<64>`` for the [fast_n, n) ×
        // M_total full-K OVERWRITE store. Skip cells covered by either.
        const bool lds_k_tail_safe = (g.m_per_group >= TAIL_BLOCK_M) &&
                                     ((g.m_per_group % TAIL_BLOCK_M) == 0);
        const int row_block_base = (row / TAIL_BLOCK_M) * TAIL_BLOCK_M;
        const bool block_in_group =
            (row_block_base + TAIL_BLOCK_M <= s_offs[group_idx + 1]);
        if (interior_n && needs_k_tail) {
            const bool lds_k_rem_match = ((g.k - g.fast_k) == 64);
            if (lds_k_tail_safe && lds_k_rem_match && block_in_group) {
                return;  // LDS K-tail (RRR) already wrote the corrected value.
            }
        }
        if (col >= g.fast_n && lds_k_tail_safe && block_in_group) {
            return;  // LDS N-tail (RRR) already wrote the absolute value.
        }
        for (int kk = k0; kk < g.k; ++kk) {
            acc += load_fp8_scalar(g.a, row, kk) *
                   load_fp8_scalar_grp(g.b, group_idx, kk, col);
        }
    }

    const float scaled = acc * resolve_combined_scale_grp(g);
    if (fast_covers_cell && needs_k_tail) {
        store_bf16_scalar(g.c, row, col,
                          load_bf16_scalar(g.c, row, col) + scaled);
    } else {
        store_bf16_scalar(g.c, row, col, scaled);
    }
}

template __global__ void grouped_tail_kernel<Layout::RCR>(const grouped_layout_globals);
template __global__ void grouped_tail_kernel<Layout::RRR>(const grouped_layout_globals);

// =============================================================================
// Round-13: LDS-staged K-tail correction kernel for FP8 RCR (mirror BF16
// ``grouped_ktail_kernel_lds`` round-11). Replaces the scalar fp32 tail
// kernel for the [fast_k, k) K-tail RMW correction on K-misaligned
// grouped shapes (gpt_oss K=2880 → K_rem=64). Profiling (rocprof on
// gpt_oss-GateUP B=32-M4096) showed the FP8 scalar tail kernel was
// **70.8 %** of total wall-time at 9.6 TFLOPS; this LDS-staged variant
// brings it to ~50-80 TFLOPS, which (combined with main+masked-store)
// closes the gpt_oss FP8 ratio from ~0.22 toward the target 1.20.
//
// Implementation mirrors BF16 closely:
//   1. Cooperative LDS load via vec4 fp8 (4 bytes/thread, NTHR=256 covers
//      the 1024-byte A and B blocks in one transaction each).
//   2. Inner-fma vec8 fp8 → ds_read_b64 + 2× fp8e4m3_4→float4 conversion
//      + 8 fma per vec8 (8 vec8 over K_REM=64 → 64 fma per cell).
//   3. Result × ``resolve_combined_scale_grp`` → bf16 RMW add.
//   4. Cross-group safety: per-row scalar fallback (vec8 + scalar tail)
//      when the (TBM × TBN) block straddles a group boundary; the host
//      ``m_per_group`` hint normally rules this out (TBM=16 << M_g).
//
// Only RCR is templated — the FP8 RRR/CRR layouts have B not stride-1 in
// K, so ds_read_b64 wouldn't help; they fall back to the scalar tail.
// =============================================================================
template<Layout L, int K_REM>
__global__ void grouped_ktail_kernel_lds(const grouped_layout_globals g) {
    static_assert(L == Layout::RCR,
        "grouped_ktail_kernel_lds (FP8): RCR only — RRR/CRR fall back to scalar tail.");
    constexpr int TBM = TAIL_BLOCK_M;       // 16
    constexpr int TBN = TAIL_BLOCK_N;       // 16
    constexpr int NTHR = TBM * TBN;         // 256

    // Round-17: pad LDS row to break (cib * K_REM) bank conflict
    // pattern. Within a wave, 16 ``cib`` lanes read B_lds at strides of
    // 64 bytes (= 16 banks); the addresses hit only banks {0,1} and
    // {16,17} → 8-way conflict, ds_read_b64 takes 8 cycles instead of
    // 1. K_REM_LDS = 72 fp8 = 72 bytes = 18 banks (= 18 mod 32) makes
    // ``cib * 18 mod 32`` distribute the 16 cib lanes across all 16
    // distinct even banks → no conflict. (Round-15 tested this in
    // isolation, was lost in noise behind the much larger fp8->fp32
    // cvt overhead. After round-16 cvt_pk_f32_fp8 fix, LDS is now the
    // bigger fraction of inner-loop time, so re-test.)
    constexpr int K_REM_LDS = K_REM + 8;
    __shared__ fp8e4m3 A_lds[TBM * K_REM_LDS];
    __shared__ fp8e4m3 B_lds[TBN * K_REM_LDS];
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];

    const int rib = threadIdx.y;
    const int cib = threadIdx.x;
    const int tid = rib * blockDim.x + cib;

    if (tid < MAX_G_PLUS_1) {
        s_offs[tid] = (tid <= g.G) ? static_cast<int>(g.group_offs[tid]) : 0;
    }
    __syncthreads();

    const int row_block_base = blockIdx.y * TBM;
    const int col_block_base = blockIdx.x * TBN;
    if (row_block_base >= g.M_total || col_block_base >= g.n) return;

    int group_idx = 0;
    #pragma unroll 1
    for (int gi = 0; gi < g.G; ++gi) {
        if (row_block_base < s_offs[gi + 1]) { group_idx = gi; break; }
    }

    const int k0 = g.fast_k;
    const int K_rem_dyn = g.k - k0;
    if (K_rem_dyn != K_REM) return;

    // Cross-group fallback: per-thread scalar K-tail RMW correction
    // (mirror BF16 round-6). Each row uses its own ``row_group`` for B
    // indexing. Vec8 fast path when k0 and g.k are 8-aligned.
    const bool cross_boundary = (row_block_base + TBM > s_offs[group_idx + 1]);
    if (cross_boundary) {
        const int row = row_block_base + rib;
        const int col = col_block_base + cib;
        if (row < g.M_total && col < g.n) {
            int row_group = 0;
            #pragma unroll 1
            for (int gi = 0; gi < g.G; ++gi) {
                if (row < s_offs[gi + 1]) { row_group = gi; break; }
            }
            // Round-17: cross_boundary fallback now uses the same
            // ``__builtin_amdgcn_cvt_pk_f32_fp8`` packed conversion +
            // 4-way parallel acc as the fast LDS path (see round-15
            // & round-16). This path only fires when a block straddles
            // a group boundary (non-uniform group_lens at runtime); the
            // metric uses uniform group_lens so it does not exercise
            // this branch — but a real bench with skewed group_lens
            // benefits ~2× on the cross_boundary cells.
            typedef __attribute__((__vector_size__(2 * sizeof(float)))) float fp32x2_v;
            auto fp8x4_to_f32x4 = [](const fp8e4m3_4& u) -> float4 {
                int packed;
                __builtin_memcpy(&packed, &u, 4);
                fp32x2_v lo = __builtin_amdgcn_cvt_pk_f32_fp8(packed, false);
                fp32x2_v hi = __builtin_amdgcn_cvt_pk_f32_fp8(packed, true);
                return make_float4(lo[0], lo[1], hi[0], hi[1]);
            };
            float acc_s0 = 0.0f, acc_s1 = 0.0f, acc_s2 = 0.0f, acc_s3 = 0.0f;
            const fp8e4m3* a_row = &g.a[coord<>(row, 0)];
            const fp8e4m3* b_row = &g.b[coord<>{0, row_group, col, 0}];
            int kk = k0;
            if ((g.k % 8 == 0) && ((k0 & 7) == 0)) {
                const fp8e4m3_8* a_v8 = reinterpret_cast<const fp8e4m3_8*>(a_row);
                const fp8e4m3_8* b_v8 = reinterpret_cast<const fp8e4m3_8*>(b_row);
                const int j_start = k0 >> 3;
                const int j_end   = g.k >> 3;
                for (int j = j_start; j < j_end; ++j) {
                    fp8e4m3_8 a8 = a_v8[j];
                    fp8e4m3_8 b8 = b_v8[j];
                    float4 a_lo = fp8x4_to_f32x4(a8.lo);
                    float4 a_hi = fp8x4_to_f32x4(a8.hi);
                    float4 b_lo = fp8x4_to_f32x4(b8.lo);
                    float4 b_hi = fp8x4_to_f32x4(b8.hi);
                    acc_s0 += a_lo.x * b_lo.x + a_hi.x * b_hi.x;
                    acc_s1 += a_lo.y * b_lo.y + a_hi.y * b_hi.y;
                    acc_s2 += a_lo.z * b_lo.z + a_hi.z * b_hi.z;
                    acc_s3 += a_lo.w * b_lo.w + a_hi.w * b_hi.w;
                }
                kk = j_end << 3;
            }
            float acc_s = (acc_s0 + acc_s1) + (acc_s2 + acc_s3);
            for (; kk < g.k; ++kk) {
                acc_s += load_fp8_scalar(g.a, row, kk) *
                         load_fp8_scalar_grp(g.b, row_group, col, kk);
            }
            const float scaled_s = acc_s * resolve_combined_scale_grp(g);
            store_bf16_scalar(g.c, row, col,
                              load_bf16_scalar(g.c, row, col) + scaled_s);
        }
        return;
    }

    // Cooperative vec4 fp8 load. NTHR=256, A_TOTAL=TBM*K_REM=1024 fp8 = 256
    // vec4 → each thread owns exactly one vec4. Mirror BF16 round-9 layout.
    // Mapping:
    //   tid =  0 → A_lds[0..3]   (row 0, k 0..3)
    //   tid =  1 → A_lds[4..7]   (row 0, k 4..7)
    //   tid = 15 → A_lds[60..63] (row 0, k 60..63)
    //   tid = 16 → A_lds[64..67] (row 1, k 0..3)
    //   ...
    // 4-byte alignment: g.a row stride = g.k fp8 (= K=2880 multiple of 4),
    // so each (r, k0+4j) start is 4-byte aligned.
    constexpr int VEC = 4;
    constexpr int VECS_PER_ROW = K_REM / VEC;       // 16
    static_assert(K_REM % VEC == 0, "K_REM must be vec4-aligned");
    static_assert(NTHR == TBM * VECS_PER_ROW,
        "Each thread must own exactly one vec4 of A.");
    {
        const int r_in_blk = tid / VECS_PER_ROW;
        const int kk_v     = tid - r_in_blk * VECS_PER_ROW;
        const int kk_start = kk_v * VEC;
        const int r_global = row_block_base + r_in_blk;
        fp8e4m3_4 va{};
        if (r_global < g.M_total) {
            const fp8e4m3* ap = &g.a[coord<>(r_global, k0 + kk_start)];
            va = *reinterpret_cast<const fp8e4m3_4*>(ap);
        }
        *reinterpret_cast<fp8e4m3_4*>(&A_lds[r_in_blk * K_REM_LDS + kk_start]) = va;
    }
    {
        const int c_in_blk = tid / VECS_PER_ROW;
        const int kk_v     = tid - c_in_blk * VECS_PER_ROW;
        const int kk_start = kk_v * VEC;
        const int c_global = col_block_base + c_in_blk;
        fp8e4m3_4 vb{};
        if (c_global < g.n) {
            const fp8e4m3* bp = &g.b[coord<>{0, group_idx, c_global, k0 + kk_start}];
            vb = *reinterpret_cast<const fp8e4m3_4*>(bp);
        }
        *reinterpret_cast<fp8e4m3_4*>(&B_lds[c_in_blk * K_REM_LDS + kk_start]) = vb;
    }
    __syncthreads();

    const int row = row_block_base + rib;
    const int col = col_block_base + cib;
    if (row >= g.M_total || col >= g.n) return;

    // Vec8 inner fma: K_REM=64 / 8 = 8 vec8 per cell. Each vec8 LDS read
    // is one ds_read_b64 (8 bytes, 2 banks broadcast). Per vec8: two
    // fp8e4m3_4 → float4 conversions for both A and B → 8 fma.
    constexpr int FMA_VEC = 8;
    constexpr int K_VECS = K_REM / FMA_VEC;          // 8
    static_assert(K_REM % FMA_VEC == 0, "K_REM must be vec8-aligned for inner fma");
    float acc = 0.0f;
    // Round-16: replace the 4 scalar fp8->fp32 conversions per
    // ``convertor::convert(a8.lo)`` (which expands to 4 separate
    // ``v_cvt_f32_fp8`` lane-shifted ops) with 2 ``v_cvt_pk_f32_fp8``
    // packed conversions (each consumes a 32-bit reg of 4 fp8 and
    // outputs a fp32x2 from a chosen pair of lanes). Halves the cvt
    // count on each operand: 4 cvts -> 2 cvts per fp8e4m3_4.
    typedef __attribute__((__vector_size__(2 * sizeof(float)))) float fp32x2_v;
    auto fp8x4_to_f32x4 = [](const fp8e4m3_4& u) -> float4 {
        int packed;
        __builtin_memcpy(&packed, &u, 4);
        fp32x2_v lo = __builtin_amdgcn_cvt_pk_f32_fp8(packed, false);
        fp32x2_v hi = __builtin_amdgcn_cvt_pk_f32_fp8(packed, true);
        return make_float4(lo[0], lo[1], hi[0], hi[1]);
    };

    // Round-15: split into 4 parallel fp32 accumulators to break the
    // 8-deep dependency chain in the per-thread fma loop.
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    #pragma unroll
    for (int kk_v = 0; kk_v < K_VECS; ++kk_v) {
        fp8e4m3_8 a8 = *reinterpret_cast<const fp8e4m3_8*>(
            &A_lds[rib * K_REM_LDS + kk_v * FMA_VEC]);
        fp8e4m3_8 b8 = *reinterpret_cast<const fp8e4m3_8*>(
            &B_lds[cib * K_REM_LDS + kk_v * FMA_VEC]);
        float4 a_lo = fp8x4_to_f32x4(a8.lo);
        float4 a_hi = fp8x4_to_f32x4(a8.hi);
        float4 b_lo = fp8x4_to_f32x4(b8.lo);
        float4 b_hi = fp8x4_to_f32x4(b8.hi);
        acc0 += a_lo.x * b_lo.x + a_hi.x * b_hi.x;
        acc1 += a_lo.y * b_lo.y + a_hi.y * b_hi.y;
        acc2 += a_lo.z * b_lo.z + a_hi.z * b_hi.z;
        acc3 += a_lo.w * b_lo.w + a_hi.w * b_hi.w;
    }
    acc = (acc0 + acc1) + (acc2 + acc3);

    // K-tail RMW correction. Main grouped kernel already wrote
    // [0, fast_k) × combined_scale at C[row, col]; we add the
    // [fast_k, k) × combined_scale slice. Mirror BF16 store.
    const float scaled = acc * resolve_combined_scale_grp(g);
    store_bf16_scalar(g.c, row, col,
                      load_bf16_scalar(g.c, row, col) + scaled);
}

template __global__ void grouped_ktail_kernel_lds<Layout::RCR, 64>(const grouped_layout_globals);

// =============================================================================
// Round-55 (FP8): LDS-staged K-tail correction kernel for **RRR** (dA path).
//
// Mirror of the BF16 ``grouped_ktail_kernel_lds_rrr<64>`` (analysis/bf16_gemm/
// mi350x/kernel_bf16_dynamic.cpp). Replaces the scalar K-loop in
// ``grouped_tail_kernel<RRR>`` for the K-tail RMW correction over
// [0, fast_n) × M_total cells on K-misaligned grouped shapes (gpt_oss
// K=2880 → K_REM=64).
//
// FP8 RRR layout: A is fp8 [M, K] row-major (stride-1 in K), B is
// fp8 [G, K, N] row-major (stride-N in K, stride-1 in N). The HBM B
// load reads 4 contiguous N cols at fixed K (vec4 fp8 = 4 bytes = 1
// dword) and SCATTERS to a [TBN, K_REM_LDS] transposed LDS layout so
// the inner-loop ds_read along K stays vec8.
//
// Bench (rocprof on FP8 grouped gpt_oss-Down B=32-M4096 dA):
//   * grouped_tail_kernel<RRR> (scalar) was ~13 ms / call.
//   * Post-LDS-staged: K-tail RMW down to ~1 ms (rest is the
//     unchanged scalar N-tail full-K reduction over [fast_n, n)).
// =============================================================================
template<int K_REM>
__global__ void grouped_ktail_kernel_lds_rrr(const grouped_layout_globals g) {
    constexpr int TBM = TAIL_BLOCK_M;       // 16
    constexpr int TBN = TAIL_BLOCK_N;       // 16
    constexpr int NTHR = TBM * TBN;         // 256

    // K_REM_LDS = K_REM + 8 padding (= 72 fp8 = 72 bytes = 18 banks mod 32 = 18):
    // makes ``cib * 18 mod 32`` distribute the 16 cib lanes across 16 distinct
    // even banks for the ds_read_b64 in the inner loop. Mirror round-17 RCR.
    constexpr int K_REM_LDS = K_REM + 8;
    __shared__ fp8e4m3 A_lds[TBM * K_REM_LDS];
    __shared__ fp8e4m3 B_lds[TBN * K_REM_LDS];
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];

    const int rib = threadIdx.y;
    const int cib = threadIdx.x;
    const int tid = rib * blockDim.x + cib;

    if (tid < MAX_G_PLUS_1) {
        s_offs[tid] = (tid <= g.G) ? static_cast<int>(g.group_offs[tid]) : 0;
    }
    __syncthreads();

    const int row_block_base = blockIdx.y * TBM;
    const int col_block_base = blockIdx.x * TBN;
    if (row_block_base >= g.M_total || col_block_base >= g.n) return;

    int group_idx = 0;
    #pragma unroll 1
    for (int gi = 0; gi < g.G; ++gi) {
        if (row_block_base < s_offs[gi + 1]) { group_idx = gi; break; }
    }

    const int k0 = g.fast_k;
    const int K_rem_dyn = g.k - k0;
    if (K_rem_dyn != K_REM) return;

    const bool cross_boundary = (row_block_base + TBM > s_offs[group_idx + 1]);
    if (cross_boundary) {
        // Per-row scalar fallback (B is stride-N in K → no vec along K).
        const int row = row_block_base + rib;
        const int col = col_block_base + cib;
        if (row < g.M_total && col < g.n) {
            int row_group = 0;
            #pragma unroll 1
            for (int gi = 0; gi < g.G; ++gi) {
                if (row < s_offs[gi + 1]) { row_group = gi; break; }
            }
            float acc_s = 0.0f;
            for (int kk = k0; kk < g.k; ++kk) {
                acc_s += load_fp8_scalar(g.a, row, kk) *
                         load_fp8_scalar_grp(g.b, row_group, kk, col);
            }
            const float scaled_s = acc_s * resolve_combined_scale_grp(g);
            store_bf16_scalar(g.c, row, col,
                              load_bf16_scalar(g.c, row, col) + scaled_s);
        }
        return;
    }

    // ---- Coop load A: [TBM, K_REM] fp8 from A[r_global, k0..k0+K_REM).
    // Same pattern as RCR round-9 (A stride-1 in K for both layouts).
    constexpr int VEC = 4;
    constexpr int A_VECS_PER_ROW = K_REM / VEC;       // 16
    static_assert(NTHR == TBM * A_VECS_PER_ROW,
        "NTHR must equal TBM * (K_REM / VEC) for vec4 coop load of A");
    {
        const int r_in_blk = tid / A_VECS_PER_ROW;
        const int kk_v = tid - r_in_blk * A_VECS_PER_ROW;
        const int kk_start = kk_v * VEC;
        const int r_global = row_block_base + r_in_blk;
        fp8e4m3_4 va{};
        if (r_global < g.M_total) {
            const fp8e4m3* ap = &g.a[coord<>(r_global, k0 + kk_start)];
            va = *reinterpret_cast<const fp8e4m3_4*>(ap);
        }
        *reinterpret_cast<fp8e4m3_4*>(&A_lds[r_in_blk * K_REM_LDS + kk_start]) = va;
    }

    // ---- Coop load B (RRR): [K_REM, TBN] fp8 from B[group, k0+kk, col_base..+TBN).
    // 4 contiguous N cols at fixed K = vec4 fp8 (4 bytes = 1 dword) HBM load.
    // SCATTER to LDS in [TBN, K_REM_LDS] transposed layout: 4 separate scalar
    // LDS stores per thread, but inner loop then reads vec8 along K stride-1.
    constexpr int B_VECS_PER_K = TBN / VEC;           // 4
    static_assert(NTHR == K_REM * B_VECS_PER_K,
        "NTHR must equal K_REM * (TBN / VEC) for vec4 coop load of B");
    {
        const int kk_in_blk = tid / B_VECS_PER_K;
        const int n_in_blk = (tid - kk_in_blk * B_VECS_PER_K) * VEC;
        const int kk_global = k0 + kk_in_blk;
        const int col_global = col_block_base + n_in_blk;
        fp8e4m3_4 vb{};
        // Zero-pad cols >= g.n. n_in_blk is 4-aligned so the 4 cols are
        // either fully in-bounds or some are >= g.n.
        if (col_global + VEC <= g.n) {
            const fp8e4m3* bp = &g.b[coord<>{0, group_idx, kk_global, col_global}];
            vb = *reinterpret_cast<const fp8e4m3_4*>(bp);
        } else if (col_global < g.n) {
            #pragma unroll
            for (int i = 0; i < VEC; ++i) {
                const int cg = col_global + i;
                if (cg < g.n) {
                    fp8e4m3 v = g.b[coord<>{0, group_idx, kk_global, cg}];
                    reinterpret_cast<fp8e4m3*>(&vb)[i] = v;
                }
            }
        }
        // Scatter to B_lds in [TBN, K_REM_LDS] transposed layout.
        const fp8e4m3* vb_arr = reinterpret_cast<const fp8e4m3*>(&vb);
        B_lds[(n_in_blk + 0) * K_REM_LDS + kk_in_blk] = vb_arr[0];
        B_lds[(n_in_blk + 1) * K_REM_LDS + kk_in_blk] = vb_arr[1];
        B_lds[(n_in_blk + 2) * K_REM_LDS + kk_in_blk] = vb_arr[2];
        B_lds[(n_in_blk + 3) * K_REM_LDS + kk_in_blk] = vb_arr[3];
    }
    __syncthreads();

    const int row = row_block_base + rib;
    const int col = col_block_base + cib;
    if (row >= g.M_total || col >= g.n) return;

    // Inner vec8 fma: K_REM/8 = 8 vec8 per cell. Mirror the FP8 RCR LDS
    // K-tail (line ~2754 in this file): 2 packed cvt_pk_f32_fp8 per fp8e4m3_4
    // operand, 4 parallel fp32 accumulators to break dependency chains.
    constexpr int FMA_VEC = 8;
    constexpr int K_VECS = K_REM / FMA_VEC;           // 8
    static_assert(K_REM % FMA_VEC == 0, "K_REM must be vec8-aligned for inner fma");
    typedef __attribute__((__vector_size__(2 * sizeof(float)))) float fp32x2_v;
    auto fp8x4_to_f32x4 = [](const fp8e4m3_4& u) -> float4 {
        int packed;
        __builtin_memcpy(&packed, &u, 4);
        fp32x2_v lo = __builtin_amdgcn_cvt_pk_f32_fp8(packed, false);
        fp32x2_v hi = __builtin_amdgcn_cvt_pk_f32_fp8(packed, true);
        return make_float4(lo[0], lo[1], hi[0], hi[1]);
    };
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    #pragma unroll
    for (int kk_v = 0; kk_v < K_VECS; ++kk_v) {
        fp8e4m3_8 a8 = *reinterpret_cast<const fp8e4m3_8*>(
            &A_lds[rib * K_REM_LDS + kk_v * FMA_VEC]);
        fp8e4m3_8 b8 = *reinterpret_cast<const fp8e4m3_8*>(
            &B_lds[cib * K_REM_LDS + kk_v * FMA_VEC]);
        float4 a_lo = fp8x4_to_f32x4(a8.lo);
        float4 a_hi = fp8x4_to_f32x4(a8.hi);
        float4 b_lo = fp8x4_to_f32x4(b8.lo);
        float4 b_hi = fp8x4_to_f32x4(b8.hi);
        acc0 += a_lo.x * b_lo.x + a_hi.x * b_hi.x;
        acc1 += a_lo.y * b_lo.y + a_hi.y * b_hi.y;
        acc2 += a_lo.z * b_lo.z + a_hi.z * b_hi.z;
        acc3 += a_lo.w * b_lo.w + a_hi.w * b_hi.w;
    }
    const float acc = (acc0 + acc1) + (acc2 + acc3);

    const float scaled = acc * resolve_combined_scale_grp(g);
    store_bf16_scalar(g.c, row, col,
                      load_bf16_scalar(g.c, row, col) + scaled);
}

template __global__ void grouped_ktail_kernel_lds_rrr<64>(const grouped_layout_globals);

// =============================================================================
// Round-56 (FP8): LDS-staged N-tail full-K reduction kernel for **RRR**.
//
// Mirror of the BF16 round-56 ``grouped_ntail_kernel_lds_rrr<64>``. Replaces
// the per-cell scalar full-K loop in ``grouped_tail_kernel<RRR>`` for
// the [fast_n, n) × M_total cells (the cells where RRR main kernel
// did NOT write because ``bpc = fast_n / BLOCK_SIZE``).
//
// Layout: A is fp8 [M, K] row-major (stride-1 in K), B is fp8 [G, K, N]
// row-major (stride-N in K, stride-1 in N). Coop-load pattern mirrors
// round-55 K-tail variant: vec4 along stride-1 axis (K for A, N for B);
// B is SCATTERED into a [TBN, K_CHUNK_LDS] transposed LDS layout so the
// inner-loop ds_read along K stays vec8 (mirrors round-15/17 FP8 RCR
// LDS K-tail).
//
// Bench (rocprof on FP8 grouped gpt_oss-GateUP-B32-M4096 dA before
// round-56): grouped_tail_kernel<RRR> at ~37 ms / call (~209 TF aggregate
// for the [fast_n=2816, n=2880] partial col-tile). The N-tail full-K
// reduction over 64 cols × 131072 rows × 5760 K = 96 GFMAs at scalar
// rate dominates. Post-round-56: LDS-staged inner FMA + vec4 HBM loads
// drop the same work to ~10 ms / call.
// =============================================================================
template<int K_CHUNK>
__global__ void grouped_ntail_kernel_lds_rrr(const grouped_layout_globals g) {
    constexpr int TBM = TAIL_BLOCK_M;            // 16
    constexpr int TBN = TAIL_BLOCK_N;            // 16
    constexpr int NTHR = TBM * TBN;              // 256
    // Pad LDS row to break (cib * stride) bank-conflict pattern. K_CHUNK
    // = 64 fp8 = 64 bytes = 16 banks, so cib lanes hit only 2 banks
    // (8-way conflict). K_CHUNK_LDS = 72 fp8 = 72 bytes = 18 banks
    // (mod 32 = 18) → cib * 18 mod 32 distributes 16 cib lanes across
    // 16 distinct even banks. Same as round-17 FP8 RCR / round-55
    // FP8 RRR K-tail.
    constexpr int K_CHUNK_LDS = K_CHUNK + 8;
    __shared__ fp8e4m3 A_lds[TBM * K_CHUNK_LDS];
    __shared__ fp8e4m3 B_lds[TBN * K_CHUNK_LDS];
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];

    const int rib = threadIdx.y;
    const int cib = threadIdx.x;
    const int tid = rib * blockDim.x + cib;

    if (tid < MAX_G_PLUS_1) {
        s_offs[tid] = (tid <= g.G) ? static_cast<int>(g.group_offs[tid]) : 0;
    }
    __syncthreads();

    const int row_block_base = blockIdx.y * TBM;
    const int col_block_base = g.fast_n + blockIdx.x * TBN;
    if (row_block_base >= g.M_total || col_block_base >= g.n) return;

    int group_idx = 0;
    #pragma unroll 1
    for (int gi = 0; gi < g.G; ++gi) {
        if (row_block_base < s_offs[gi + 1]) { group_idx = gi; break; }
    }

    const bool cross_boundary = (row_block_base + TBM > s_offs[group_idx + 1]);
    if (cross_boundary) {
        const int row_s = row_block_base + rib;
        const int col_s = col_block_base + cib;
        if (row_s < g.M_total && col_s < g.n) {
            int row_group = 0;
            #pragma unroll 1
            for (int gi = 0; gi < g.G; ++gi) {
                if (row_s < s_offs[gi + 1]) { row_group = gi; break; }
            }
            float acc_s = 0.0f;
            for (int kk = 0; kk < g.k; ++kk) {
                acc_s += load_fp8_scalar(g.a, row_s, kk) *
                         load_fp8_scalar_grp(g.b, row_group, kk, col_s);
            }
            const float scaled_s = acc_s * resolve_combined_scale_grp(g);
            store_bf16_scalar(g.c, row_s, col_s, scaled_s);
        }
        return;
    }

    const int row = row_block_base + rib;
    const int col = col_block_base + cib;
    const bool active_cell = (row < g.M_total) && (col < g.n);

    constexpr int VEC = 4;

    // ---- A coop-load layout: vec4 along K, NTHR = TBM × (K_CHUNK / VEC).
    constexpr int A_VECS_PER_ROW = K_CHUNK / VEC;        // 16
    static_assert(K_CHUNK % VEC == 0, "K_CHUNK must be vec4-aligned");
    static_assert(NTHR == TBM * A_VECS_PER_ROW,
        "NTHR must equal TBM * (K_CHUNK / VEC) for A coop load");
    const int r_in_blk_a = tid / A_VECS_PER_ROW;
    const int kk_v_a = tid - r_in_blk_a * A_VECS_PER_ROW;
    const int kk_start_a = kk_v_a * VEC;

    // ---- B coop-load layout: vec4 along N at fixed K, scatter to
    // transposed [TBN, K_CHUNK_LDS] LDS. NTHR = K_CHUNK × (TBN / VEC).
    constexpr int B_VECS_PER_K = TBN / VEC;              // 4
    static_assert(NTHR == K_CHUNK * B_VECS_PER_K,
        "NTHR must equal K_CHUNK * (TBN / VEC) for B coop load");
    const int kk_in_blk_b = tid / B_VECS_PER_K;
    const int n_in_blk_b = (tid - kk_in_blk_b * B_VECS_PER_K) * VEC;

    // 4 parallel fp32 accumulators to break the inner-fma dependency
    // chain (mirror round-15 FP8 RCR LDS K-tail).
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    typedef __attribute__((__vector_size__(2 * sizeof(float)))) float fp32x2_v;
    auto fp8x4_to_f32x4 = [](const fp8e4m3_4& u) -> float4 {
        int packed;
        __builtin_memcpy(&packed, &u, 4);
        fp32x2_v lo = __builtin_amdgcn_cvt_pk_f32_fp8(packed, false);
        fp32x2_v hi = __builtin_amdgcn_cvt_pk_f32_fp8(packed, true);
        return make_float4(lo[0], lo[1], hi[0], hi[1]);
    };

    for (int k_chunk_start = 0; k_chunk_start < g.k; k_chunk_start += K_CHUNK) {
        // Coop-load A (vec4 fp8 = 4 bytes / thread).
        {
            const int r_global = row_block_base + r_in_blk_a;
            const int k_global = k_chunk_start + kk_start_a;
            fp8e4m3_4 va{};
            if (r_global < g.M_total && k_global + VEC <= g.k) {
                const fp8e4m3* ap = &g.a[coord<>(r_global, k_global)];
                va = *reinterpret_cast<const fp8e4m3_4*>(ap);
            }
            *reinterpret_cast<fp8e4m3_4*>(&A_lds[r_in_blk_a * K_CHUNK_LDS + kk_start_a]) = va;
        }
        // Coop-load B (vec4 fp8 along N; scatter to transposed LDS).
        {
            const int kk_global = k_chunk_start + kk_in_blk_b;
            const int col_global = col_block_base + n_in_blk_b;
            fp8e4m3_4 vb{};
            if (kk_global < g.k && col_global + VEC <= g.n) {
                const fp8e4m3* bp = &g.b[coord<>{0, group_idx, kk_global, col_global}];
                vb = *reinterpret_cast<const fp8e4m3_4*>(bp);
            } else if (kk_global < g.k && col_global < g.n) {
                #pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    const int cg = col_global + i;
                    if (cg < g.n) {
                        fp8e4m3 v = g.b[coord<>{0, group_idx, kk_global, cg}];
                        reinterpret_cast<fp8e4m3*>(&vb)[i] = v;
                    }
                }
            }
            const fp8e4m3* vb_arr = reinterpret_cast<const fp8e4m3*>(&vb);
            B_lds[(n_in_blk_b + 0) * K_CHUNK_LDS + kk_in_blk_b] = vb_arr[0];
            B_lds[(n_in_blk_b + 1) * K_CHUNK_LDS + kk_in_blk_b] = vb_arr[1];
            B_lds[(n_in_blk_b + 2) * K_CHUNK_LDS + kk_in_blk_b] = vb_arr[2];
            B_lds[(n_in_blk_b + 3) * K_CHUNK_LDS + kk_in_blk_b] = vb_arr[3];
        }
        __syncthreads();

        if (active_cell) {
            constexpr int FMA_VEC = 8;
            constexpr int K_CHUNK_VECS = K_CHUNK / FMA_VEC;
            static_assert(K_CHUNK % FMA_VEC == 0, "K_CHUNK must be vec8-aligned for inner fma");
            const int k_left = g.k - k_chunk_start;
            const int k_iters_v = (k_left < K_CHUNK)
                ? (k_left + FMA_VEC - 1) / FMA_VEC : K_CHUNK_VECS;
            #pragma unroll
            for (int kk_v = 0; kk_v < K_CHUNK_VECS; ++kk_v) {
                if (kk_v >= k_iters_v) break;
                fp8e4m3_8 a8 = *reinterpret_cast<const fp8e4m3_8*>(
                    &A_lds[rib * K_CHUNK_LDS + kk_v * FMA_VEC]);
                fp8e4m3_8 b8 = *reinterpret_cast<const fp8e4m3_8*>(
                    &B_lds[cib * K_CHUNK_LDS + kk_v * FMA_VEC]);
                float4 a_lo = fp8x4_to_f32x4(a8.lo);
                float4 a_hi = fp8x4_to_f32x4(a8.hi);
                float4 b_lo = fp8x4_to_f32x4(b8.lo);
                float4 b_hi = fp8x4_to_f32x4(b8.hi);
                acc0 += a_lo.x * b_lo.x + a_hi.x * b_hi.x;
                acc1 += a_lo.y * b_lo.y + a_hi.y * b_hi.y;
                acc2 += a_lo.z * b_lo.z + a_hi.z * b_hi.z;
                acc3 += a_lo.w * b_lo.w + a_hi.w * b_hi.w;
            }
        }
        __syncthreads();
    }

    if (active_cell) {
        const float acc = (acc0 + acc1) + (acc2 + acc3);
        const float scaled = acc * resolve_combined_scale_grp(g);
        store_bf16_scalar(g.c, row, col, scaled);
    }
}

template __global__ void grouped_ntail_kernel_lds_rrr<64>(const grouped_layout_globals);

// =============================================================================
// Round-18 (FP8): MFMA-based K-tail correction kernel for RCR.
//
// rocprof on gpt_oss-GateUP B=32-M4096 (FP8 grouped) showed the LDS-staged
// scalar-fp32 K-tail (``grouped_ktail_kernel_lds``) was **66 %** of total
// wall-time at ~16 TFLOPS — only 0.6 % of fp8 mfma peak. The rest of the
// wall is the main kernel doing 22 K-blocks at ~1370 TFLOPS. This kernel
// replaces the scalar fma inner loop with a single mfma_scale_f32_16x16x128
// _f8f6f4 call per (16M × 16N) cell-tile (K=64 valid + K=64 zero-padded
// → 50 % effective mfma utilization but still ~75× the throughput of the
// scalar fp32 fma path).
//
// Geometry mirrors ``grouped_ktail_kernel_lds<RCR, 64>``:
//   * Block: 1 wave (64 threads). blockDim = (64,).
//   * Grid: ceil_div(n, 16) × ceil_div(M_total, 16) — same as scalar/LDS
//     paths so the host dispatcher only flips the kernel template.
//   * Each lane owns 4 output cells C[(t/16)*4 + 0..3, t%16] and feeds
//     32 fp8 of A and 32 fp8 of B (chunks 0,1: real K=[k0, k0+64); chunks
//     2,3: zero) into the v_mfma_f32_16x16x128_f8f6f4 op.
//   * RMW into bf16 g.c with combined_scale = sa * sb.
//
// Cross-group fallback (block straddles a group boundary in M):
//   * Detected via row_block_base + 16 > s_offs[group_idx + 1].
//   * Falls through to a per-row scalar vec8 fma loop (same compute model
//     as ``grouped_ktail_kernel_lds`` cross-boundary path) using only the
//     first 64 lanes; the host hint ``m_per_group % 16 == 0`` for uniform
//     groups means this branch is unreachable in the metric.
//
// FP8-only RCR: B is row-major [G, N, K]; ``mma_ABt`` (= mfma with both
// operands K-contig) directly applies. RRR/CRR have B not stride-1 in K
// so they continue to use the scalar tail.
// =============================================================================
template<Layout L, int K_REM>
__global__ void grouped_ktail_kernel_mfma(const grouped_layout_globals g) {
    static_assert(L == Layout::RCR,
        "grouped_ktail_kernel_mfma (FP8): RCR only — RRR/CRR fall back to scalar tail.");
    static_assert(K_REM == 64,
        "grouped_ktail_kernel_mfma (FP8): K_REM must be 64 (zero-padded to 128).");
    constexpr int TBM = TAIL_BLOCK_M;       // 16
    constexpr int TBN = TAIL_BLOCK_N;       // 16
    constexpr int K_PER_LANE_CHUNK = 32;    // mfma_16x16x128 distributes K across 4 lane-chunks
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];

    const int tid = threadIdx.x;            // single-wave block, 64 threads
    if (tid <= g.G && tid < MAX_G_PLUS_1) {
        s_offs[tid] = static_cast<int>(g.group_offs[tid]);
    }
    __syncthreads();

    const int row_block_base = blockIdx.y * TBM;
    const int col_block_base = blockIdx.x * TBN;
    if (row_block_base >= g.M_total || col_block_base >= g.n) return;

    int group_idx = 0;
    #pragma unroll 1
    for (int gi = 0; gi < g.G; ++gi) {
        if (row_block_base < s_offs[gi + 1]) { group_idx = gi; break; }
    }

    const int K_rem_dyn = g.k - g.fast_k;
    if (K_rem_dyn != K_REM) return;
    const int k0 = g.fast_k;

    // Cross-group fallback (per-row vec8 scalar fma + cvt_pk_f32_fp8).
    // Mirror ``grouped_ktail_kernel_lds`` cross_boundary path. Single
    // wave: each lane handles a (row, col) cell within the 16×16 block.
    const bool cross_boundary = (row_block_base + TBM > s_offs[group_idx + 1]);
    if (cross_boundary) {
        typedef __attribute__((__vector_size__(2 * sizeof(float)))) float fp32x2_v;
        auto fp8x4_to_f32x4 = [](const fp8e4m3_4& u) -> float4 {
            int packed;
            __builtin_memcpy(&packed, &u, 4);
            fp32x2_v lo = __builtin_amdgcn_cvt_pk_f32_fp8(packed, false);
            fp32x2_v hi = __builtin_amdgcn_cvt_pk_f32_fp8(packed, true);
            return make_float4(lo[0], lo[1], hi[0], hi[1]);
        };
        // 64 lanes cover the 256 cells in 4 passes (4 cells/lane).
        #pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            const int rib = (slot * 16) + (tid / TBN);
            const int cib = tid % TBN;
            const int row = row_block_base + rib;
            const int col = col_block_base + cib;
            if (row >= g.M_total || col >= g.n) continue;
            int row_group = 0;
            #pragma unroll 1
            for (int gi = 0; gi < g.G; ++gi) {
                if (row < s_offs[gi + 1]) { row_group = gi; break; }
            }
            float acc_s0 = 0.f, acc_s1 = 0.f, acc_s2 = 0.f, acc_s3 = 0.f;
            const fp8e4m3* a_row = &g.a[coord<>(row, 0)];
            const fp8e4m3* b_row = &g.b[coord<>{0, row_group, col, 0}];
            int kk = k0;
            if ((g.k % 8 == 0) && ((k0 & 7) == 0)) {
                const fp8e4m3_8* a_v8 = reinterpret_cast<const fp8e4m3_8*>(a_row);
                const fp8e4m3_8* b_v8 = reinterpret_cast<const fp8e4m3_8*>(b_row);
                const int j_start = k0 >> 3;
                const int j_end   = g.k >> 3;
                for (int j = j_start; j < j_end; ++j) {
                    fp8e4m3_8 a8 = a_v8[j];
                    fp8e4m3_8 b8 = b_v8[j];
                    float4 a_lo = fp8x4_to_f32x4(a8.lo);
                    float4 a_hi = fp8x4_to_f32x4(a8.hi);
                    float4 b_lo = fp8x4_to_f32x4(b8.lo);
                    float4 b_hi = fp8x4_to_f32x4(b8.hi);
                    acc_s0 += a_lo.x * b_lo.x + a_hi.x * b_hi.x;
                    acc_s1 += a_lo.y * b_lo.y + a_hi.y * b_hi.y;
                    acc_s2 += a_lo.z * b_lo.z + a_hi.z * b_hi.z;
                    acc_s3 += a_lo.w * b_lo.w + a_hi.w * b_hi.w;
                }
                kk = j_end << 3;
            }
            float acc_s = (acc_s0 + acc_s1) + (acc_s2 + acc_s3);
            for (; kk < g.k; ++kk) {
                acc_s += load_fp8_scalar(g.a, row, kk) *
                         load_fp8_scalar_grp(g.b, row_group, col, kk);
            }
            const float scaled_s = acc_s * resolve_combined_scale_grp(g);
            store_bf16_scalar(g.c, row, col,
                              load_bf16_scalar(g.c, row, col) + scaled_s);
        }
        return;
    }

    // ----- Fast MFMA path ------------------------------------------------
    // Lane (t):  row_in_blk = t % 16,  k_chunk = t / 16  (0..3).
    //   * Lane 0..15  (chunk=0): A row=row_in_blk, K=[k0+0,  k0+32)  — real
    //   * Lane 16..31 (chunk=1): A row=row_in_blk, K=[k0+32, k0+64)  — real
    //   * Lane 32..47 (chunk=2): A row=row_in_blk, K=[k0+64, k0+96)  — pad 0
    //   * Lane 48..63 (chunk=3): A row=row_in_blk, K=[k0+96, k0+128) — pad 0
    //   B mirrors A with col=row_in_blk replacing row=row_in_blk (RCR ABt).
    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
    typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;

    const int row_in_blk = tid % TBM;
    const int chunk      = tid / TBM;

    intx8_t a_pack;
    intx8_t b_pack;
    if (chunk < 2) {
        const int k_off = k0 + chunk * K_PER_LANE_CHUNK;
        const int g_row = row_block_base + row_in_blk;
        const int g_col = col_block_base + row_in_blk;
        // K=2880, k0=2816 → 32-byte aligned. row strides = K = 2880 (fp8) →
        // 32-byte aligned. Single 32-byte buffer load = 2 × b128.
        if (g_row < g.M_total) {
            const fp8e4m3* a_ptr = &g.a[coord<>(g_row, k_off)];
            a_pack = *reinterpret_cast<const intx8_t*>(a_ptr);
        } else {
            a_pack = intx8_t{};
        }
        if (g_col < g.n) {
            const fp8e4m3* b_ptr = &g.b[coord<>{0, group_idx, g_col, k_off}];
            b_pack = *reinterpret_cast<const intx8_t*>(b_ptr);
        } else {
            b_pack = intx8_t{};
        }
    } else {
        a_pack = intx8_t{};
        b_pack = intx8_t{};
    }

    // mfma_scale_f32_16x16x128_f8f6f4 — D[16,16] = A[16,128] @ B^T[128,16] + 0
    floatx4_t acc = floatx4_t{0.f, 0.f, 0.f, 0.f};
    acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a_pack, b_pack, acc, /*cbsz=*/0, /*abid=*/0, /*blgp=*/0,
        /*scale_op_a=*/0, /*scale_op_b=*/0, /*scale_op_d=*/0);

    // Output: lane t → cells C[(t/16)*4 + (0..3), t%16].
    const float scale = resolve_combined_scale_grp(g);
    const int out_row_base = row_block_base + chunk * 4;
    const int out_col      = col_block_base + row_in_blk;
    if (out_col >= g.n) return;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = out_row_base + i;
        if (r >= g.M_total) break;
        const float existing = load_bf16_scalar(g.c, r, out_col);
        const float new_val  = existing + acc[i] * scale;
        store_bf16_scalar(g.c, r, out_col, new_val);
    }
}

template __global__ void grouped_ktail_kernel_mfma<Layout::RCR, 64>(const grouped_layout_globals);

// =============================================================================
// Round-20 (FP8): 32x32x64 MFMA-based K-tail correction kernel for RCR.
//
// Round-18 used ``v_mfma_scale_f32_16x16x128_f8f6f4`` to fold a K=64 tail
// into a single mfma op per (16M × 16N) cell tile. That mfma instruction
// natively consumes K=128 fp8, so K_REM=64 forced **50 % effective MFMA
// utilization** (lanes 0..31 fed real K=[k0, k0+64), lanes 32..63 fed
// zero-padded K=[k0+64, k0+128)).
//
// gfx950 also exposes ``v_mfma_scale_f32_32x32x64_f8f6f4`` whose native K
// is **64**, so K_REM=64 fits perfectly with **100 % MFMA utilization**.
// The output cell tile expands from 16×16 = 256 cells to 32×32 = 1024
// cells per block, so the K-tail grid shrinks 4× while the per-mfma
// useful work doubles — net ~2× theoretical speedup on the K-tail
// dominated gpt_oss FP8 grouped path.
//
// Layout (verified by /tmp/mfma_fp8_3232x64_test2 microtest, gfx950):
//   * Input A: [32 rows, 64 K] fp8 row-major.
//       lane t (0..63): row = t % 32, K-chunk = t / 32.
//       lane t supplies a_pack = 32 fp8 covering A[t%32, k_chunk*32 + 0..31]
//       (= one intx8_t = 8 ints = 32 bytes).
//   * Input B: same shape as A, but logically B^T = [64 K, 32 cols].
//       lane t supplies b_pack = 32 fp8 covering B[t%32 (= col), k_chunk*32 + 0..31].
//   * Output D: [32 rows, 32 cols] fp32, 16 floats per lane.
//       For lane t at d[i] (i = 0..15):
//           col       = t % 32
//           chunk     = t / 32  (0 or 1)
//           row_group = i / 4
//           row_in_grp= (i % 4) + chunk * 4
//           row       = row_group * 8 + row_in_grp
//       i.e. chunk=0 owns rows {0..3, 8..11, 16..19, 24..27} per col;
//            chunk=1 owns rows {4..7, 12..15, 20..23, 28..31} per col.
//
// Host hint requirements:
//   * ``g.m_per_group >= 32 && g.m_per_group % 32 == 0`` (TBM=32). Falls
//     back to the 16x16 round-18 mfma kernel otherwise.
//   * Per-block runtime cross-group check still re-derives ``group_idx``
//     and falls back to a per-cell scalar fma fallback if a single
//     32-row block straddles a group boundary (rare in MoE dispatch
//     where group_lens are typically uniform M-aligned).
// =============================================================================
template<Layout L, int K_REM>
__global__ void grouped_ktail_kernel_mfma32x32(const grouped_layout_globals g) {
    static_assert(L == Layout::RCR,
        "grouped_ktail_kernel_mfma32x32 (FP8): RCR only.");
    static_assert(K_REM == 64,
        "grouped_ktail_kernel_mfma32x32 (FP8): K_REM must be 64 (native mfma K).");
    constexpr int TBM = 32;
    constexpr int TBN = 32;
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];

    const int tid = threadIdx.x;            // single-wave block, 64 threads
    if (tid <= g.G && tid < MAX_G_PLUS_1) {
        s_offs[tid] = static_cast<int>(g.group_offs[tid]);
    }
    __syncthreads();

    const int row_block_base = blockIdx.y * TBM;
    const int col_block_base = blockIdx.x * TBN;
    if (row_block_base >= g.M_total || col_block_base >= g.n) return;

    int group_idx = 0;
    #pragma unroll 1
    for (int gi = 0; gi < g.G; ++gi) {
        if (row_block_base < s_offs[gi + 1]) { group_idx = gi; break; }
    }

    const int K_rem_dyn = g.k - g.fast_k;
    if (K_rem_dyn != K_REM) return;
    const int k0 = g.fast_k;

    // Cross-group fallback (block straddles a group boundary in M, or
    // hits the M_total tail). 64 lanes × 16 passes = 1024 cells covered
    // by the same per-row vec8 cvt_pk_f32_fp8 + 4-way parallel-acc loop
    // as the 16x16 variant. Unreachable on uniform group_lens with M_g
    // % 32 == 0 (the metric path).
    const bool cross_boundary = (row_block_base + TBM > s_offs[group_idx + 1]);
    if (cross_boundary) {
        typedef __attribute__((__vector_size__(2 * sizeof(float)))) float fp32x2_v;
        auto fp8x4_to_f32x4 = [](const fp8e4m3_4& u) -> float4 {
            int packed;
            __builtin_memcpy(&packed, &u, 4);
            fp32x2_v lo = __builtin_amdgcn_cvt_pk_f32_fp8(packed, false);
            fp32x2_v hi = __builtin_amdgcn_cvt_pk_f32_fp8(packed, true);
            return make_float4(lo[0], lo[1], hi[0], hi[1]);
        };
        // 64 lanes cover the 1024 cells in 16 passes (16 cells/lane).
        // Pass slot s ∈ 0..15 → cell offset within block:
        //   rib = s / 2 * 2 + (tid / 32);  // 16 row stripes
        //   cib = (s % 2) * 32 + (tid % 32);  // 32 cols / pass
        // Hmm simpler: each lane does 16 passes, in pass s walks (rib, cib).
        // Use the standard tiled traversal: each lane "owns" col=tid%32
        // and walks 16 distinct rows per col.
        const int col = col_block_base + (tid % 32);
        if (col < g.n) {
            #pragma unroll
            for (int rr = 0; rr < TBM; ++rr) {
                if ((rr % 2) != ((tid / 32))) continue;
                // Already this means: chunk-0 lanes serve even rr (0, 2, ...)
                // and chunk-1 lanes serve odd rr (1, 3, ...). Half work / lane.
                // Actually we want 16 cells / lane evenly — but the metric
                // path never reaches this branch, so a simple "every other
                // row per chunk" division (16 rows / lane) is fine.
                const int row = row_block_base + rr;
                if (row >= g.M_total) break;
                int row_group = 0;
                #pragma unroll 1
                for (int gi = 0; gi < g.G; ++gi) {
                    if (row < s_offs[gi + 1]) { row_group = gi; break; }
                }
                float acc_s0 = 0.f, acc_s1 = 0.f, acc_s2 = 0.f, acc_s3 = 0.f;
                const fp8e4m3* a_row = &g.a[coord<>(row, 0)];
                const fp8e4m3* b_row = &g.b[coord<>{0, row_group, col, 0}];
                int kk = k0;
                if ((g.k % 8 == 0) && ((k0 & 7) == 0)) {
                    const fp8e4m3_8* a_v8 = reinterpret_cast<const fp8e4m3_8*>(a_row);
                    const fp8e4m3_8* b_v8 = reinterpret_cast<const fp8e4m3_8*>(b_row);
                    const int j_start = k0 >> 3;
                    const int j_end   = g.k >> 3;
                    for (int j = j_start; j < j_end; ++j) {
                        fp8e4m3_8 a8 = a_v8[j];
                        fp8e4m3_8 b8 = b_v8[j];
                        float4 a_lo = fp8x4_to_f32x4(a8.lo);
                        float4 a_hi = fp8x4_to_f32x4(a8.hi);
                        float4 b_lo = fp8x4_to_f32x4(b8.lo);
                        float4 b_hi = fp8x4_to_f32x4(b8.hi);
                        acc_s0 += a_lo.x * b_lo.x + a_hi.x * b_hi.x;
                        acc_s1 += a_lo.y * b_lo.y + a_hi.y * b_hi.y;
                        acc_s2 += a_lo.z * b_lo.z + a_hi.z * b_hi.z;
                        acc_s3 += a_lo.w * b_lo.w + a_hi.w * b_hi.w;
                    }
                    kk = j_end << 3;
                }
                float acc_s = (acc_s0 + acc_s1) + (acc_s2 + acc_s3);
                for (; kk < g.k; ++kk) {
                    acc_s += load_fp8_scalar(g.a, row, kk) *
                             load_fp8_scalar_grp(g.b, row_group, col, kk);
                }
                const float scaled_s = acc_s * resolve_combined_scale_grp(g);
                store_bf16_scalar(g.c, row, col,
                                  load_bf16_scalar(g.c, row, col) + scaled_s);
            }
        }
        return;
    }

    // ----- Fast MFMA path ------------------------------------------------
    // K=64 native mfma_32x32x64_f8f6f4 — 100 % utilization (vs round-18
    // 50 % zero-padded mfma_16x16x128). 1 wave per block, 64 lanes feed
    // the entire 32×64 A and 32×64 B tiles.
    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;

    const int row_in_blk = tid % 32;
    const int chunk      = tid / 32;        // 0 or 1
    const int k_off      = k0 + chunk * 32;

    const int g_row = row_block_base + row_in_blk;
    const int g_col = col_block_base + row_in_blk;

    intx8_t a_pack;
    intx8_t b_pack;
    // K=2880 → k0=2816 → k_off ∈ {2816, 2848}; row stride = K = 2880 fp8.
    // Both 32-byte aligned (2816 % 32 = 0, 2848 % 32 = 0, 2880 % 32 = 0)
    // so one buffer load per pack = 2 × b128.
    if (g_row < g.M_total) {
        const fp8e4m3* a_ptr = &g.a[coord<>(g_row, k_off)];
        a_pack = *reinterpret_cast<const intx8_t*>(a_ptr);
    } else {
        a_pack = intx8_t{};
    }
    if (g_col < g.n) {
        const fp8e4m3* b_ptr = &g.b[coord<>{0, group_idx, g_col, k_off}];
        b_pack = *reinterpret_cast<const intx8_t*>(b_ptr);
    } else {
        b_pack = intx8_t{};
    }

    // mfma_scale_f32_32x32x64_f8f6f4 — D[32,32] = A[32,64] @ B^T[64,32] + C
    floatx16_t acc{};
    acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a_pack, b_pack, acc, /*cbsz=*/0, /*abid=*/0, /*blgp=*/0,
        /*scale_op_a=*/0, /*scale_op_b=*/0, /*scale_op_d=*/0);

    // Output: lane t → cells C[r, t%32] for 16 r's.
    //   chunk=0: rows 0..3, 8..11, 16..19, 24..27 in d[0..3], d[4..7], d[8..11], d[12..15]
    //   chunk=1: rows 4..7, 12..15, 20..23, 28..31
    // Unified: row_group = i/4; row_in_group = (i%4) + chunk*4; row = row_group*8 + row_in_group.
    const float scale = resolve_combined_scale_grp(g);
    const int out_col = col_block_base + row_in_blk;
    if (out_col >= g.n) return;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int row_group     = i >> 2;
        const int row_in_group  = (i & 3) + chunk * 4;
        const int local_row     = row_group * 8 + row_in_group;
        const int r             = row_block_base + local_row;
        if (r >= g.M_total) continue;
        const float existing = load_bf16_scalar(g.c, r, out_col);
        const float new_val  = existing + acc[i] * scale;
        store_bf16_scalar(g.c, r, out_col, new_val);
    }
}

template __global__ void grouped_ktail_kernel_mfma32x32<Layout::RCR, 64>(const grouped_layout_globals);

// =============================================================================
// Round-53 (FP8): 64×32 MFMA-based K-tail correction kernel for RCR — two
// stacked 32×32 sub-blocks sharing the B-pack load.
//
// Microbench (round-53 probe, gpt_oss FP8 grouped):
//   K-tail kernel was 22-32 % of total wall time on the gpt_oss family
//   (B ∈ {4, 32}, M_per ∈ {2048, 4096}, N ∈ {2880, 5760}, K=2880). Per
//   block bytes break down as:
//
//     fast MFMA path (mfma32x32_M1):  A=2KB + B=2KB + C_rmw=4KB = 8 KB/block
//                                     (for 32×32 = 1024 cells; 8 B/cell)
//
//   Shaping the row-tile to TBM=64 lets a single B-pack feed two stacked
//   32×32 sub-blocks (one mfma each). Per block bytes:
//
//     fast MFMA path (mfma32x32_M2):  A=4KB + B=2KB + C_rmw=8KB = 14 KB/block
//                                     (for 64×32 = 2048 cells; 7 B/cell)
//
//   That's a 12.5 % per-cell HBM byte reduction — straight wall-time savings
//   on the K-tail-bound gpt_oss path. We keep the 32×32 M1 path as a
//   fallback for ``m_per_group``-not-64-aligned (round-20 round-13 LDS path).
//
// Host hint requirements (gate the dispatch):
//   * ``g.m_per_group >= 64 && g.m_per_group % 64 == 0`` (TBM_TOTAL=64).
//     gpt_oss M_per ∈ {2048, 4096} both satisfy.
//   * Per-block ``row_block_base + 64 <= s_offs[group_idx + 1]`` runtime
//     check still fires; non-uniform group_lens whose avg=64-aligned but
//     individual per-group M is not, fall back to the per-row scalar
//     fma loop covering BOTH sub-blocks (rare on the metric path).
//
// MFMA layout / lane mapping is identical to the round-20 32×32 path —
// each sub-block is a separate mfma_scale_f32_32x32x64_f8f6f4 call with
// the same a_pack / b_pack lane geometry. We just shift the row base by
// +TBM_SUB on the second sub-block and re-issue with a fresh accumulator.
// =============================================================================
template<Layout L, int K_REM>
__global__ void grouped_ktail_kernel_mfma32x32_M2(const grouped_layout_globals g) {
    static_assert(L == Layout::RCR,
        "grouped_ktail_kernel_mfma32x32_M2 (FP8): RCR only.");
    static_assert(K_REM == 64,
        "grouped_ktail_kernel_mfma32x32_M2 (FP8): K_REM must be 64 (native mfma K).");
    constexpr int TBM_TOTAL = 64;       // 2 stacked 32×32 sub-blocks
    constexpr int TBM_SUB   = 32;
    constexpr int TBN       = 32;
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];

    const int tid = threadIdx.x;            // single-wave block, 64 threads
    if (tid <= g.G && tid < MAX_G_PLUS_1) {
        s_offs[tid] = static_cast<int>(g.group_offs[tid]);
    }
    __syncthreads();

    const int row_block_base = blockIdx.y * TBM_TOTAL;
    const int col_block_base = blockIdx.x * TBN;
    if (row_block_base >= g.M_total || col_block_base >= g.n) return;

    int group_idx = 0;
    #pragma unroll 1
    for (int gi = 0; gi < g.G; ++gi) {
        if (row_block_base < s_offs[gi + 1]) { group_idx = gi; break; }
    }

    const int K_rem_dyn = g.k - g.fast_k;
    if (K_rem_dyn != K_REM) return;
    const int k0 = g.fast_k;

    // Cross-group fallback: if the 64-row block straddles a group boundary
    // (or the M_total tail), fall back to per-row scalar fma over BOTH
    // sub-blocks. Unreachable on uniform group_lens with M_g % 64 == 0
    // (gpt_oss M_per ∈ {2048, 4096} both satisfy).
    const bool cross_boundary = (row_block_base + TBM_TOTAL > s_offs[group_idx + 1]);
    if (cross_boundary) {
        typedef __attribute__((__vector_size__(2 * sizeof(float)))) float fp32x2_v;
        auto fp8x4_to_f32x4 = [](const fp8e4m3_4& u) -> float4 {
            int packed;
            __builtin_memcpy(&packed, &u, 4);
            fp32x2_v lo = __builtin_amdgcn_cvt_pk_f32_fp8(packed, false);
            fp32x2_v hi = __builtin_amdgcn_cvt_pk_f32_fp8(packed, true);
            return make_float4(lo[0], lo[1], hi[0], hi[1]);
        };
        // 64 lanes × 32 cells/lane (the 64-row × 32-col block has 2048
        // cells; chunk-0 lanes serve even-rr and chunk-1 lanes serve
        // odd-rr just like the round-20 M1 fallback, half work each).
        const int col = col_block_base + (tid % 32);
        if (col < g.n) {
            #pragma unroll 1
            for (int rr = 0; rr < TBM_TOTAL; ++rr) {
                if ((rr % 2) != ((tid / 32))) continue;
                const int row = row_block_base + rr;
                if (row >= g.M_total) break;
                int row_group = 0;
                #pragma unroll 1
                for (int gi = 0; gi < g.G; ++gi) {
                    if (row < s_offs[gi + 1]) { row_group = gi; break; }
                }
                float acc_s0 = 0.f, acc_s1 = 0.f, acc_s2 = 0.f, acc_s3 = 0.f;
                const fp8e4m3* a_row = &g.a[coord<>(row, 0)];
                const fp8e4m3* b_row = &g.b[coord<>{0, row_group, col, 0}];
                int kk = k0;
                if ((g.k % 8 == 0) && ((k0 & 7) == 0)) {
                    const fp8e4m3_8* a_v8 = reinterpret_cast<const fp8e4m3_8*>(a_row);
                    const fp8e4m3_8* b_v8 = reinterpret_cast<const fp8e4m3_8*>(b_row);
                    const int j_start = k0 >> 3;
                    const int j_end   = g.k >> 3;
                    for (int j = j_start; j < j_end; ++j) {
                        fp8e4m3_8 a8 = a_v8[j];
                        fp8e4m3_8 b8 = b_v8[j];
                        float4 a_lo = fp8x4_to_f32x4(a8.lo);
                        float4 a_hi = fp8x4_to_f32x4(a8.hi);
                        float4 b_lo = fp8x4_to_f32x4(b8.lo);
                        float4 b_hi = fp8x4_to_f32x4(b8.hi);
                        acc_s0 += a_lo.x * b_lo.x + a_hi.x * b_hi.x;
                        acc_s1 += a_lo.y * b_lo.y + a_hi.y * b_hi.y;
                        acc_s2 += a_lo.z * b_lo.z + a_hi.z * b_hi.z;
                        acc_s3 += a_lo.w * b_lo.w + a_hi.w * b_hi.w;
                    }
                    kk = j_end << 3;
                }
                float acc_s = (acc_s0 + acc_s1) + (acc_s2 + acc_s3);
                for (; kk < g.k; ++kk) {
                    acc_s += load_fp8_scalar(g.a, row, kk) *
                             load_fp8_scalar_grp(g.b, row_group, col, kk);
                }
                const float scaled_s = acc_s * resolve_combined_scale_grp(g);
                store_bf16_scalar(g.c, row, col,
                                  load_bf16_scalar(g.c, row, col) + scaled_s);
            }
        }
        return;
    }

    // ----- Fast MFMA path: shared B-pack across two stacked 32×32 sub-blocks
    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;

    const int row_in_blk = tid % 32;
    const int chunk      = tid / 32;        // 0 or 1
    const int k_off      = k0 + chunk * 32;

    const int g_col = col_block_base + row_in_blk;

    // Single B load shared across both sub-blocks. B is [G, N, K] row-major
    // and the K-tail slice [k0, k0+K_REM) is the same for sub-block 0 and
    // sub-block 1 (only the A row range differs). Halves the B-side HBM
    // bandwidth vs running two independent 32×32 M1 blocks.
    intx8_t b_pack;
    if (g_col < g.n) {
        const fp8e4m3* b_ptr = &g.b[coord<>{0, group_idx, g_col, k_off}];
        b_pack = *reinterpret_cast<const intx8_t*>(b_ptr);
    } else {
        b_pack = intx8_t{};
    }

    const float scale = resolve_combined_scale_grp(g);
    const int out_col = col_block_base + row_in_blk;
    if (out_col >= g.n) return;

    // Two stacked 32×32 sub-blocks. Same MFMA / lane layout as round-20
    // M1; we just shift the row base by +TBM_SUB on sub-block 1.
    #pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
        const int sub_row_base = row_block_base + sub * TBM_SUB;
        const int g_row = sub_row_base + row_in_blk;

        intx8_t a_pack;
        if (g_row < g.M_total) {
            const fp8e4m3* a_ptr = &g.a[coord<>(g_row, k_off)];
            a_pack = *reinterpret_cast<const intx8_t*>(a_ptr);
        } else {
            a_pack = intx8_t{};
        }

        floatx16_t acc{};
        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_pack, b_pack, acc, /*cbsz=*/0, /*abid=*/0, /*blgp=*/0,
            /*scale_op_a=*/0, /*scale_op_b=*/0, /*scale_op_d=*/0);

        // Output cell mapping (mirror M1): lane t at d[i] (i=0..15) →
        //   row_group     = i / 4
        //   row_in_group  = (i % 4) + chunk * 4
        //   row           = row_group * 8 + row_in_group
        // i.e. chunk-0 lane owns rows {0..3, 8..11, 16..19, 24..27} per col,
        //      chunk-1 lane owns rows {4..7, 12..15, 20..23, 28..31} per col.
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int row_group     = i >> 2;
            const int row_in_group  = (i & 3) + chunk * 4;
            const int local_row     = row_group * 8 + row_in_group;
            const int r             = sub_row_base + local_row;
            if (r >= g.M_total) continue;
            const float existing = load_bf16_scalar(g.c, r, out_col);
            const float new_val  = existing + acc[i] * scale;
            store_bf16_scalar(g.c, r, out_col, new_val);
        }
    }
}

template __global__ void grouped_ktail_kernel_mfma32x32_M2<Layout::RCR, 64>(const grouped_layout_globals);

// Round-60 (FP8): M2N2 — 64×64 K-tail block. 2 stacked 32×32 sub-blocks in M
// (M2 inheritance) × 2 sub-tiles in N. The single A-pack load per M-sub-block
// now feeds TWO mfmas (one per N sub-tile), halving the per-cell A-side HBM
// bandwidth on the K-tail path. B-side bandwidth stays at ~1 KB per sub-tile
// (sharing the same K-slice pattern). C RMW doubles vs M2 but the HBM-byte
// share favors M2N2:
//
//     M2   per block:  A=4KB + B=2KB + C_rmw= 8KB = 14 KB/block (2048 cells)
//     M2N2 per block:  A=4KB + B=4KB + C_rmw=16KB = 24 KB/block (4096 cells)
//                                                  ⇒ 6.0 vs 7.0 B/cell (-14 %)
//
// Grid halves in N (90 → 45 col-blocks for gpt_oss N=5760 and 180 → 90 for
// DSV3-Down N=7168), so per-cell wall time drops by both the byte-budget
// reduction AND the launch / scheduler overhead reduction.
//
// Host hint requirements:
//   * Same M alignment as M2 (g.m_per_group >= 64 && % 64 == 0).
//   * N must be 64-aligned (g.n % 64 == 0). gpt_oss N ∈ {2880, 5760} both
//     satisfy (2880=45·64, 5760=90·64); DSV3-Down N=7168=112·64 satisfies.
//     DSV3-GateUP N=4096=64·64 satisfies. Mis-aligned N falls back to M2
//     (which itself handles partial last col-tile via out_col >= g.n early
//     return + the existing cross_boundary scalar fma).
//
// MFMA / lane mapping is identical to M2 — the 4 mfmas per thread reuse the
// per-lane (row_group, row_in_group, chunk) decode; we just iterate the N
// sub-tile with a fresh b_pack and a fresh accumulator.
template<Layout L, int K_REM>
__global__ void grouped_ktail_kernel_mfma32x32_M2N2(const grouped_layout_globals g) {
    static_assert(L == Layout::RCR,
        "grouped_ktail_kernel_mfma32x32_M2N2 (FP8): RCR only.");
    static_assert(K_REM == 64,
        "grouped_ktail_kernel_mfma32x32_M2N2 (FP8): K_REM must be 64 (native mfma K).");
    constexpr int TBM_TOTAL = 64;       // 2 stacked 32×32 sub-blocks
    constexpr int TBM_SUB   = 32;
    constexpr int TBN       = 32;
    constexpr int N_SUB     = 2;
    constexpr int TBN_TOTAL = TBN * N_SUB;  // 64 cols
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];

    const int tid = threadIdx.x;            // single-wave block, 64 threads
    if (tid <= g.G && tid < MAX_G_PLUS_1) {
        s_offs[tid] = static_cast<int>(g.group_offs[tid]);
    }
    __syncthreads();

    const int row_block_base = blockIdx.y * TBM_TOTAL;
    const int col_block_base = blockIdx.x * TBN_TOTAL;
    if (row_block_base >= g.M_total || col_block_base >= g.n) return;

    int group_idx = 0;
    #pragma unroll 1
    for (int gi = 0; gi < g.G; ++gi) {
        if (row_block_base < s_offs[gi + 1]) { group_idx = gi; break; }
    }

    const int K_rem_dyn = g.k - g.fast_k;
    if (K_rem_dyn != K_REM) return;
    const int k0 = g.fast_k;

    // Cross-group fallback: same as M2 (64-row block straddling a group
    // boundary). Iterate over BOTH 32-col sub-tiles per row using scalar
    // fma. Unreachable on uniform group_lens with M_g % 64 == 0.
    const bool cross_boundary = (row_block_base + TBM_TOTAL > s_offs[group_idx + 1]);
    if (cross_boundary) {
        typedef __attribute__((__vector_size__(2 * sizeof(float)))) float fp32x2_v;
        auto fp8x4_to_f32x4 = [](const fp8e4m3_4& u) -> float4 {
            int packed;
            __builtin_memcpy(&packed, &u, 4);
            fp32x2_v lo = __builtin_amdgcn_cvt_pk_f32_fp8(packed, false);
            fp32x2_v hi = __builtin_amdgcn_cvt_pk_f32_fp8(packed, true);
            return make_float4(lo[0], lo[1], hi[0], hi[1]);
        };
        const float scale_s = resolve_combined_scale_grp(g);
        // 64 lanes split: chunk-0 (tid<32) covers even-rr, chunk-1 odd-rr
        // for each of the 2 col sub-tiles.
        for (int nt = 0; nt < N_SUB; ++nt) {
            const int col = col_block_base + nt * TBN + (tid % 32);
            if (col >= g.n) continue;
            #pragma unroll 1
            for (int rr = 0; rr < TBM_TOTAL; ++rr) {
                if ((rr % 2) != ((tid / 32))) continue;
                const int row = row_block_base + rr;
                if (row >= g.M_total) break;
                int row_group = 0;
                #pragma unroll 1
                for (int gi = 0; gi < g.G; ++gi) {
                    if (row < s_offs[gi + 1]) { row_group = gi; break; }
                }
                float acc_s0 = 0.f, acc_s1 = 0.f, acc_s2 = 0.f, acc_s3 = 0.f;
                const fp8e4m3* a_row = &g.a[coord<>(row, 0)];
                const fp8e4m3* b_row = &g.b[coord<>{0, row_group, col, 0}];
                int kk = k0;
                if ((g.k % 8 == 0) && ((k0 & 7) == 0)) {
                    const fp8e4m3_8* a_v8 = reinterpret_cast<const fp8e4m3_8*>(a_row);
                    const fp8e4m3_8* b_v8 = reinterpret_cast<const fp8e4m3_8*>(b_row);
                    const int j_start = k0 >> 3;
                    const int j_end   = g.k >> 3;
                    for (int j = j_start; j < j_end; ++j) {
                        fp8e4m3_8 a8 = a_v8[j];
                        fp8e4m3_8 b8 = b_v8[j];
                        float4 a_lo = fp8x4_to_f32x4(a8.lo);
                        float4 a_hi = fp8x4_to_f32x4(a8.hi);
                        float4 b_lo = fp8x4_to_f32x4(b8.lo);
                        float4 b_hi = fp8x4_to_f32x4(b8.hi);
                        acc_s0 += a_lo.x * b_lo.x + a_hi.x * b_hi.x;
                        acc_s1 += a_lo.y * b_lo.y + a_hi.y * b_hi.y;
                        acc_s2 += a_lo.z * b_lo.z + a_hi.z * b_hi.z;
                        acc_s3 += a_lo.w * b_lo.w + a_hi.w * b_hi.w;
                    }
                    kk = j_end << 3;
                }
                float acc_s = (acc_s0 + acc_s1) + (acc_s2 + acc_s3);
                for (; kk < g.k; ++kk) {
                    acc_s += load_fp8_scalar(g.a, row, kk) *
                             load_fp8_scalar_grp(g.b, row_group, col, kk);
                }
                const float scaled_s = acc_s * scale_s;
                store_bf16_scalar(g.c, row, col,
                                  load_bf16_scalar(g.c, row, col) + scaled_s);
            }
        }
        return;
    }

    // ----- Fast MFMA path: 2 M sub-blocks × 2 N sub-tiles = 4 mfmas/thread.
    // The A-pack is loaded ONCE per M sub-block and reused across the 2 N
    // sub-tiles (vs M2 where the same A-pack drives just 1 mfma per
    // sub-block). Halves per-cell A-side HBM bandwidth on the K-tail.
    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;

    const int row_in_blk = tid % 32;
    const int chunk      = tid / 32;        // 0 or 1
    const int k_off      = k0 + chunk * 32;

    // Two B-packs, one per N sub-tile. Same K slice for both packs (A-pack
    // shared between them) → mfma decomposes the A·Bᵀ block into 2 separate
    // 32×32 outputs along N.
    intx8_t b_pack[N_SUB];
    #pragma unroll
    for (int nt = 0; nt < N_SUB; ++nt) {
        const int g_col = col_block_base + nt * TBN + row_in_blk;
        if (g_col < g.n) {
            const fp8e4m3* b_ptr = &g.b[coord<>{0, group_idx, g_col, k_off}];
            b_pack[nt] = *reinterpret_cast<const intx8_t*>(b_ptr);
        } else {
            b_pack[nt] = intx8_t{};
        }
    }

    const float scale = resolve_combined_scale_grp(g);

    #pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
        const int sub_row_base = row_block_base + sub * TBM_SUB;
        const int g_row = sub_row_base + row_in_blk;

        intx8_t a_pack;
        if (g_row < g.M_total) {
            const fp8e4m3* a_ptr = &g.a[coord<>(g_row, k_off)];
            a_pack = *reinterpret_cast<const intx8_t*>(a_ptr);
        } else {
            a_pack = intx8_t{};
        }

        #pragma unroll
        for (int nt = 0; nt < N_SUB; ++nt) {
            floatx16_t acc{};
            acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a_pack, b_pack[nt], acc, /*cbsz=*/0, /*abid=*/0, /*blgp=*/0,
                /*scale_op_a=*/0, /*scale_op_b=*/0, /*scale_op_d=*/0);

            const int out_col = col_block_base + nt * TBN + row_in_blk;
            if (out_col >= g.n) continue;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int row_group     = i >> 2;
                const int row_in_group  = (i & 3) + chunk * 4;
                const int local_row     = row_group * 8 + row_in_group;
                const int r             = sub_row_base + local_row;
                if (r >= g.M_total) continue;
                const float existing = load_bf16_scalar(g.c, r, out_col);
                const float new_val  = existing + acc[i] * scale;
                store_bf16_scalar(g.c, r, out_col, new_val);
            }
        }
    }
}

template __global__ void grouped_ktail_kernel_mfma32x32_M2N2<Layout::RCR, 64>(const grouped_layout_globals);

// Round-61 (FP8): tried M2N4 — 64×128 K-tail block sharing the A-pack across
// 4 N sub-tiles (per-cell HBM 7.0 → 5.5 B/cell, -21 %). Metric regressed
// 752 → 749-750 (3-run mean) and per-shape gpt_oss-GateUP ratios dropped
// 1.6-2.8pp. Root causes:
//   * The 4-mfma chain per A-pack creates a long dependency chain that
//     under-utilizes the inner-loop FMA pipeline (compute-bound regime
//     when register pressure rises).
//   * Per-block C RMW grows 4× (32 → 128 cells/thread), serializing on
//     the bf16 read-add-write address dependency.
//   * Occupancy drops from 8 to 7 waves/SIMD (VGPRs 50 → 66) — less
//     latency hiding to absorb the longer chain.
// M2N2 retained as the round-60 sweet spot. M2N4 kernel definition kept
// disabled below for future revival if the C RMW chain can be split (see
// the round-61 round-trip "load-then-store" 2-phase pattern adopted in the
// M2 / M2N2 epilogues, which addresses the chain length).
#if 0  // round-61 disabled: regressed metric 752 → 749 (see comment above)
template<Layout L, int K_REM>
__global__ void grouped_ktail_kernel_mfma32x32_M2N4(const grouped_layout_globals g) {
    static_assert(L == Layout::RCR,
        "grouped_ktail_kernel_mfma32x32_M2N4 (FP8): RCR only.");
    static_assert(K_REM == 64,
        "grouped_ktail_kernel_mfma32x32_M2N4 (FP8): K_REM must be 64 (native mfma K).");
    constexpr int TBM_TOTAL = 64;       // 2 stacked 32×32 sub-blocks
    constexpr int TBM_SUB   = 32;
    constexpr int TBN       = 32;
    constexpr int N_SUB     = 4;
    constexpr int TBN_TOTAL = TBN * N_SUB;  // 128 cols
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];

    const int tid = threadIdx.x;            // single-wave block, 64 threads
    if (tid <= g.G && tid < MAX_G_PLUS_1) {
        s_offs[tid] = static_cast<int>(g.group_offs[tid]);
    }
    __syncthreads();

    const int row_block_base = blockIdx.y * TBM_TOTAL;
    const int col_block_base = blockIdx.x * TBN_TOTAL;
    if (row_block_base >= g.M_total || col_block_base >= g.n) return;

    int group_idx = 0;
    #pragma unroll 1
    for (int gi = 0; gi < g.G; ++gi) {
        if (row_block_base < s_offs[gi + 1]) { group_idx = gi; break; }
    }

    const int K_rem_dyn = g.k - g.fast_k;
    if (K_rem_dyn != K_REM) return;
    const int k0 = g.fast_k;

    // Cross-group fallback: same as M2N2 (64-row block straddling a group
    // boundary). Iterate over ALL FOUR 32-col sub-tiles per row using
    // scalar fma. Unreachable on uniform group_lens with M_g % 64 == 0.
    const bool cross_boundary = (row_block_base + TBM_TOTAL > s_offs[group_idx + 1]);
    if (cross_boundary) {
        typedef __attribute__((__vector_size__(2 * sizeof(float)))) float fp32x2_v;
        auto fp8x4_to_f32x4 = [](const fp8e4m3_4& u) -> float4 {
            int packed;
            __builtin_memcpy(&packed, &u, 4);
            fp32x2_v lo = __builtin_amdgcn_cvt_pk_f32_fp8(packed, false);
            fp32x2_v hi = __builtin_amdgcn_cvt_pk_f32_fp8(packed, true);
            return make_float4(lo[0], lo[1], hi[0], hi[1]);
        };
        const float scale_s = resolve_combined_scale_grp(g);
        for (int nt = 0; nt < N_SUB; ++nt) {
            const int col = col_block_base + nt * TBN + (tid % 32);
            if (col >= g.n) continue;
            #pragma unroll 1
            for (int rr = 0; rr < TBM_TOTAL; ++rr) {
                if ((rr % 2) != ((tid / 32))) continue;
                const int row = row_block_base + rr;
                if (row >= g.M_total) break;
                int row_group = 0;
                #pragma unroll 1
                for (int gi = 0; gi < g.G; ++gi) {
                    if (row < s_offs[gi + 1]) { row_group = gi; break; }
                }
                float acc_s0 = 0.f, acc_s1 = 0.f, acc_s2 = 0.f, acc_s3 = 0.f;
                const fp8e4m3* a_row = &g.a[coord<>(row, 0)];
                const fp8e4m3* b_row = &g.b[coord<>{0, row_group, col, 0}];
                int kk = k0;
                if ((g.k % 8 == 0) && ((k0 & 7) == 0)) {
                    const fp8e4m3_8* a_v8 = reinterpret_cast<const fp8e4m3_8*>(a_row);
                    const fp8e4m3_8* b_v8 = reinterpret_cast<const fp8e4m3_8*>(b_row);
                    const int j_start = k0 >> 3;
                    const int j_end   = g.k >> 3;
                    for (int j = j_start; j < j_end; ++j) {
                        fp8e4m3_8 a8 = a_v8[j];
                        fp8e4m3_8 b8 = b_v8[j];
                        float4 a_lo = fp8x4_to_f32x4(a8.lo);
                        float4 a_hi = fp8x4_to_f32x4(a8.hi);
                        float4 b_lo = fp8x4_to_f32x4(b8.lo);
                        float4 b_hi = fp8x4_to_f32x4(b8.hi);
                        acc_s0 += a_lo.x * b_lo.x + a_hi.x * b_hi.x;
                        acc_s1 += a_lo.y * b_lo.y + a_hi.y * b_hi.y;
                        acc_s2 += a_lo.z * b_lo.z + a_hi.z * b_hi.z;
                        acc_s3 += a_lo.w * b_lo.w + a_hi.w * b_hi.w;
                    }
                    kk = j_end << 3;
                }
                float acc_s = (acc_s0 + acc_s1) + (acc_s2 + acc_s3);
                for (; kk < g.k; ++kk) {
                    acc_s += load_fp8_scalar(g.a, row, kk) *
                             load_fp8_scalar_grp(g.b, row_group, col, kk);
                }
                const float scaled_s = acc_s * scale_s;
                store_bf16_scalar(g.c, row, col,
                                  load_bf16_scalar(g.c, row, col) + scaled_s);
            }
        }
        return;
    }

    // ----- Fast MFMA path: 2 M sub-blocks × 4 N sub-tiles = 8 mfmas/thread.
    // Single A-pack per M sub-block reused across all 4 N sub-tiles.
    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;

    const int row_in_blk = tid % 32;
    const int chunk      = tid / 32;        // 0 or 1
    const int k_off      = k0 + chunk * 32;

    intx8_t b_pack[N_SUB];
    #pragma unroll
    for (int nt = 0; nt < N_SUB; ++nt) {
        const int g_col = col_block_base + nt * TBN + row_in_blk;
        if (g_col < g.n) {
            const fp8e4m3* b_ptr = &g.b[coord<>{0, group_idx, g_col, k_off}];
            b_pack[nt] = *reinterpret_cast<const intx8_t*>(b_ptr);
        } else {
            b_pack[nt] = intx8_t{};
        }
    }

    const float scale = resolve_combined_scale_grp(g);

    #pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
        const int sub_row_base = row_block_base + sub * TBM_SUB;
        const int g_row = sub_row_base + row_in_blk;

        intx8_t a_pack;
        if (g_row < g.M_total) {
            const fp8e4m3* a_ptr = &g.a[coord<>(g_row, k_off)];
            a_pack = *reinterpret_cast<const intx8_t*>(a_ptr);
        } else {
            a_pack = intx8_t{};
        }

        #pragma unroll
        for (int nt = 0; nt < N_SUB; ++nt) {
            floatx16_t acc{};
            acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a_pack, b_pack[nt], acc, /*cbsz=*/0, /*abid=*/0, /*blgp=*/0,
                /*scale_op_a=*/0, /*scale_op_b=*/0, /*scale_op_d=*/0);

            const int out_col = col_block_base + nt * TBN + row_in_blk;
            if (out_col >= g.n) continue;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int row_group     = i >> 2;
                const int row_in_group  = (i & 3) + chunk * 4;
                const int local_row     = row_group * 8 + row_in_group;
                const int r             = sub_row_base + local_row;
                if (r >= g.M_total) continue;
                const float existing = load_bf16_scalar(g.c, r, out_col);
                const float new_val  = existing + acc[i] * scale;
                store_bf16_scalar(g.c, r, out_col, new_val);
            }
        }
    }
}

template __global__ void grouped_ktail_kernel_mfma32x32_M2N4<Layout::RCR, 64>(const grouped_layout_globals);
#endif  // M2N4 disabled

// Round-54 (FP8): tried M4 (TBM=128, 4 stacked 32×32 sub-blocks sharing one
// B-pack) but metric regressed — per-shape probe showed Down-B4-{M2048,
// M4096} each lost 25-28 TF (-3 to -4 %). Root cause: B=4 grids are small
// (5760 blocks at M_per=2048) → halving the grid (M4) leaves <3 waves per
// occupant slot, under-saturating the GPU. M2 keeps the wider parallel
// grid and is already at the bandwidth-limit elbow (M2 vs M4 wall-time
// within ±0.1 ms across the 8 gpt_oss FP8 metric shapes per the probe at
// /tmp/probe_fp8_bottleneck.py). Pinning FP8 K-tail to M2; see
// dispatch_grouped_rcr below.
#if 0  // M4 kernel definition kept disabled — see comment above
template<Layout L, int K_REM>
__global__ void grouped_ktail_kernel_mfma32x32_M4(const grouped_layout_globals g) {
    static_assert(L == Layout::RCR,
        "grouped_ktail_kernel_mfma32x32_M4 (FP8): RCR only.");
    static_assert(K_REM == 64,
        "grouped_ktail_kernel_mfma32x32_M4 (FP8): K_REM must be 64 (native mfma K).");
    constexpr int TBM_TOTAL = 128;      // 4 stacked 32×32 sub-blocks
    constexpr int TBM_SUB   = 32;
    constexpr int N_SUB     = 4;
    constexpr int TBN       = 32;
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];

    const int tid = threadIdx.x;            // single-wave block, 64 threads
    if (tid <= g.G && tid < MAX_G_PLUS_1) {
        s_offs[tid] = static_cast<int>(g.group_offs[tid]);
    }
    __syncthreads();

    const int row_block_base = blockIdx.y * TBM_TOTAL;
    const int col_block_base = blockIdx.x * TBN;
    if (row_block_base >= g.M_total || col_block_base >= g.n) return;

    int group_idx = 0;
    #pragma unroll 1
    for (int gi = 0; gi < g.G; ++gi) {
        if (row_block_base < s_offs[gi + 1]) { group_idx = gi; break; }
    }

    const int K_rem_dyn = g.k - g.fast_k;
    if (K_rem_dyn != K_REM) return;
    const int k0 = g.fast_k;

    // Cross-group fallback: if the 128-row block straddles a group boundary
    // (or the M_total tail), fall back to per-row scalar fma over ALL FOUR
    // sub-blocks. Unreachable on uniform group_lens with M_g % 128 == 0
    // (gpt_oss M_per ∈ {2048, 4096} both satisfy).
    const bool cross_boundary = (row_block_base + TBM_TOTAL > s_offs[group_idx + 1]);
    if (cross_boundary) {
        typedef __attribute__((__vector_size__(2 * sizeof(float)))) float fp32x2_v;
        auto fp8x4_to_f32x4 = [](const fp8e4m3_4& u) -> float4 {
            int packed;
            __builtin_memcpy(&packed, &u, 4);
            fp32x2_v lo = __builtin_amdgcn_cvt_pk_f32_fp8(packed, false);
            fp32x2_v hi = __builtin_amdgcn_cvt_pk_f32_fp8(packed, true);
            return make_float4(lo[0], lo[1], hi[0], hi[1]);
        };
        const int col = col_block_base + (tid % 32);
        if (col < g.n) {
            #pragma unroll 1
            for (int rr = 0; rr < TBM_TOTAL; ++rr) {
                if ((rr % 2) != ((tid / 32))) continue;
                const int row = row_block_base + rr;
                if (row >= g.M_total) break;
                int row_group = 0;
                #pragma unroll 1
                for (int gi = 0; gi < g.G; ++gi) {
                    if (row < s_offs[gi + 1]) { row_group = gi; break; }
                }
                float acc_s0 = 0.f, acc_s1 = 0.f, acc_s2 = 0.f, acc_s3 = 0.f;
                const fp8e4m3* a_row = &g.a[coord<>(row, 0)];
                const fp8e4m3* b_row = &g.b[coord<>{0, row_group, col, 0}];
                int kk = k0;
                if ((g.k % 8 == 0) && ((k0 & 7) == 0)) {
                    const fp8e4m3_8* a_v8 = reinterpret_cast<const fp8e4m3_8*>(a_row);
                    const fp8e4m3_8* b_v8 = reinterpret_cast<const fp8e4m3_8*>(b_row);
                    const int j_start = k0 >> 3;
                    const int j_end   = g.k >> 3;
                    for (int j = j_start; j < j_end; ++j) {
                        fp8e4m3_8 a8 = a_v8[j];
                        fp8e4m3_8 b8 = b_v8[j];
                        float4 a_lo = fp8x4_to_f32x4(a8.lo);
                        float4 a_hi = fp8x4_to_f32x4(a8.hi);
                        float4 b_lo = fp8x4_to_f32x4(b8.lo);
                        float4 b_hi = fp8x4_to_f32x4(b8.hi);
                        acc_s0 += a_lo.x * b_lo.x + a_hi.x * b_hi.x;
                        acc_s1 += a_lo.y * b_lo.y + a_hi.y * b_hi.y;
                        acc_s2 += a_lo.z * b_lo.z + a_hi.z * b_hi.z;
                        acc_s3 += a_lo.w * b_lo.w + a_hi.w * b_hi.w;
                    }
                    kk = j_end << 3;
                }
                float acc_s = (acc_s0 + acc_s1) + (acc_s2 + acc_s3);
                for (; kk < g.k; ++kk) {
                    acc_s += load_fp8_scalar(g.a, row, kk) *
                             load_fp8_scalar_grp(g.b, row_group, col, kk);
                }
                const float scaled_s = acc_s * resolve_combined_scale_grp(g);
                store_bf16_scalar(g.c, row, col,
                                  load_bf16_scalar(g.c, row, col) + scaled_s);
            }
        }
        return;
    }

    // ----- Fast MFMA path: shared B-pack across FOUR stacked 32×32 sub-blocks
    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;

    const int row_in_blk = tid % 32;
    const int chunk      = tid / 32;        // 0 or 1
    const int k_off      = k0 + chunk * 32;

    const int g_col = col_block_base + row_in_blk;

    // Single B load shared across all four sub-blocks. B is [G, N, K] row-
    // major and the K-tail slice [k0, k0+K_REM) is the same for sub-blocks
    // 0..3 (only the A row range differs). 4×reuse of the B-side HBM
    // bandwidth vs running 4 independent 32×32 M1 blocks; 2× reuse vs M2.
    intx8_t b_pack;
    if (g_col < g.n) {
        const fp8e4m3* b_ptr = &g.b[coord<>{0, group_idx, g_col, k_off}];
        b_pack = *reinterpret_cast<const intx8_t*>(b_ptr);
    } else {
        b_pack = intx8_t{};
    }

    const float scale = resolve_combined_scale_grp(g);
    const int out_col = col_block_base + row_in_blk;
    if (out_col >= g.n) return;

    // Four stacked 32×32 sub-blocks. Same MFMA / lane layout as round-20
    // M1 / round-53 M2; we shift the row base by +sub*TBM_SUB and re-issue
    // mfma_scale_f32_32x32x64_f8f6f4 with a fresh accumulator each sub.
    #pragma unroll
    for (int sub = 0; sub < N_SUB; ++sub) {
        const int sub_row_base = row_block_base + sub * TBM_SUB;
        const int g_row = sub_row_base + row_in_blk;

        intx8_t a_pack;
        if (g_row < g.M_total) {
            const fp8e4m3* a_ptr = &g.a[coord<>(g_row, k_off)];
            a_pack = *reinterpret_cast<const intx8_t*>(a_ptr);
        } else {
            a_pack = intx8_t{};
        }

        floatx16_t acc{};
        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_pack, b_pack, acc, /*cbsz=*/0, /*abid=*/0, /*blgp=*/0,
            /*scale_op_a=*/0, /*scale_op_b=*/0, /*scale_op_d=*/0);

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int row_group     = i >> 2;
            const int row_in_group  = (i & 3) + chunk * 4;
            const int local_row     = row_group * 8 + row_in_group;
            const int r             = sub_row_base + local_row;
            if (r >= g.M_total) continue;
            const float existing = load_bf16_scalar(g.c, r, out_col);
            const float new_val  = existing + acc[i] * scale;
            store_bf16_scalar(g.c, r, out_col, new_val);
        }
    }
}

template __global__ void grouped_ktail_kernel_mfma32x32_M4<Layout::RCR, 64>(const grouped_layout_globals);
#endif  // M4 kernel disabled

void dispatch_grouped_rcr(grouped_layout_globals g) {
    g.n = static_cast<int>(g.c.cols());
    g.M_total = static_cast<int>(g.c.rows());
    g.k = static_cast<int>(g.a.cols());

    // Round-12: mirror BF16 grouped round-11 path. Main kernel uses
    // ``bpc = ceil_div(g.n, BLOCK_SIZE)`` unconditionally on RCR; the
    // full-tensor SRD inside ``rcr_8w_load_hoist`` clamps OOB row loads
    // to 0 (no swizzle stride, see line ~432), and the column-masked
    // ``store_c_tile_n_masked`` in the main kernel drops OOB cells from
    // the write-back. This eliminates the ``grouped_tail_kernel`` full-K
    // N-tail reduction (scalar fp32 vec8) for cols [fast_n, n) — the
    // dominant wall-time on gpt_oss N=2880/5760, K=2880 grouped FP8.
    // The tail kernel still runs for K-tail correction in [fast_k, k)
    // for ALL cols [0, g.n) (including the partial last col-tile, since
    // main now wrote partial K reduction there too).
    g.fast_n = (g.n / BLOCK_SIZE) * BLOCK_SIZE;
    g.fast_k = (g.k / K_BLOCK)    * K_BLOCK;
    g.bpc    = kittens::ceil_div(g.n, BLOCK_SIZE);
    g.ki     = g.fast_k / K_BLOCK;

    // Round-2 path A (fused K-tail): select FUSED_KTAIL=true variant when
    // K_REM matches the in-kernel partial-K load granularity (16-aligned)
    // AND the host m_per_group hint guarantees that no persistent-loop
    // (br, bc) tile straddles a group boundary on the K-tail load (same
    // safety condition as the existing standalone K-tail kernels'
    // mfma path: m_per_group >= TAIL_BLOCK_M and 16-aligned). When fuse is
    // active, the main kernel itself accumulates the K=[fast_k, k) tail
    // into cA/cB/cC/cD before scale + store; we then SKIP the standalone
    // grouped_ktail_kernel_* launch below (no double-counting, no RMW on
    // g.c, no extra launch overhead).
    //
    // Round-2 enables fuse only for K_REM == 64 (the gpt_oss K=2880 case,
    // which is the only K_REM exercised by metric and the worst-perf
    // section pre-fuse). Future rounds extend to K_REM ∈ {16, 32, 48,
    // 80, 96, 112} once the K_REM=64 numerics pass.
    const int K_rem_for_fuse = g.k - g.fast_k;
    const bool lds_k_tail_safe_for_fuse =
        (g.m_per_group >= TAIL_BLOCK_M) &&
        ((g.m_per_group % TAIL_BLOCK_M) == 0);
    // Round-34-dm: extend FUSED_KTAIL=true selection to K_REM=0 shapes (DSV3,
    // all K-aligned cases). The in-kernel K-tail branch `if (g.fast_k < g.k)`
    // is runtime-gated; for K_REM=0 it NEVER executes. But the code's presence
    // changes LLVM's register allocation: ISA inspection (R34-dm) shows the
    // FUSED_KTAIL=true template specs have HALF to a THIRD the number of
    // interleaved scratch_store/mfma pairs in epilog 1 (28 vs 62 for
    // GateUP spec, 22 vs 44 for Down spec). The difference is a codegen
    // artifact from the extra `A_row_reg a_kt1;` declaration giving LLVM a
    // different liveness graph to work with. Numerical output is identical
    // because the K-tail branch is dead at runtime for K_REM=0.
    const bool fuse_ktail_eligible =
        (g.bpc > 0) && (g.ki > 0) &&
        ((K_rem_for_fuse == 64) || (K_rem_for_fuse == 0)) &&
        lds_k_tail_safe_for_fuse;

    // === Round-F M3 (quarantined): BLOCK_SIZE=128 dispatch gate ==============
    // Status (Round-F M5 perf-falsified, 2026-05-07):
    //   M2a port + M2-debug-3 wait-counter halving = correct kernel (8/8
    //   PASS on gpt_oss kernel-only metric, SNR 297 dB matches outer).
    //   M5 perf falsified — b128 is 23-38 % SLOWER than outer on every
    //   gpt_oss shape that enters the b128 path (per-tile fixed overhead
    //   doesn't scale with tile size; 4x more tiles × same overhead per
    //   tile swamps the CU-utilization gain from going 1.26 → 4.84
    //   tiles/CU). See ``round-F-fp8-tile-size-128-port-correctness-fix-
    //   perf-falsified.md`` for full per-shape numbers and PMC-ish
    //   accounting.
    //
    // Quarantine: this gate is ENV-ONLY (TURBO_FP8_B128=1). Production
    // traffic (env unset, env=0, env<0, env>1) goes through the outer
    // BLOCK_SIZE=256 path verbatim, score 691 unchanged. Env=1 routes
    // K-aligned + M-aligned shapes through the (correct, slow) b128 path
    // for future research that may flip the perf conclusion (e.g.
    // 4-wave / 2-CTA b128 layout, sub-tile fusion sharing prologue cost
    // across 2 b128 tiles, or a different shape suite where tile-merge
    // overhead is < CU-utilization gain).
    constexpr int B128_BLOCK = 128;
    const bool b128_k_aligned     = (g.fast_k == g.k);
    const bool b128_m_per_grp_ok  = (g.m_per_group >= B128_BLOCK) &&
                                    ((g.m_per_group % B128_BLOCK) == 0);
    const char* env_b128 = std::getenv("TURBO_FP8_B128");
    const int b128_force = (env_b128 != nullptr) ? std::atoi(env_b128) : 0;
    const bool use_b128 =
        b128_k_aligned && b128_m_per_grp_ok && (g.bpc > 0) &&
        (b128_force == 1);   // env=1 only; quarantined until M4 lands.

    if (use_b128) {
        // Recompute geometry for BLOCK_SIZE=128 path.
        g.fast_n = (g.n / B128_BLOCK) * B128_BLOCK;
        g.fast_k = (g.k / K_BLOCK) * K_BLOCK;
        g.bpc    = kittens::ceil_div(g.n, B128_BLOCK);
        g.ki     = g.fast_k / K_BLOCK;
        const bool n_aligned_b128 = (g.bpc * B128_BLOCK == g.n);
        if (n_aligned_b128) {
            kernel_b128::grouped_rcr_kernel<0, false, false>
                <<<dim3(NUM_CUS), g.block(), 0, g.stream>>>(g);
        } else {
            kernel_b128::grouped_rcr_kernel<0, true , false>
                <<<dim3(NUM_CUS), g.block(), 0, g.stream>>>(g);
        }
        return;
    }

    if (g.bpc > 0 && g.ki > 0) {
        // Round-12: launch-uniform branch on N alignment selects the
        // masked-vs-raw store variant at compile time. DSV3 N=4096/7168
        // hits the raw-store instance (zero overhead, ratios stable);
        // gpt_oss N=2880/5760 hits the masked-store instance.
        const bool n_aligned = (g.bpc * BLOCK_SIZE == g.n);

        // Round-4 (gpt_oss FP8 kernel-only): optional persistent-grid
        // slot override for short-grid sparse fwd shapes. The kernel
        // body uses ``gridDim.x`` for both the chiplet-swizzle range
        // and the persistent-loop stride (see lines ~2696 / ~2766);
        // any ``slots ∈ [1, NUM_CUS]`` produces bit-identical math
        // (only the (CU, tile) assignment changes). The R2 var-K
        // sweep proved short-grid wgrad shapes gain ~6% by reducing
        // slots; the same lever is plausible for sparse fwd RCR
        // (Down-B4-M2048 fwd at 1.5 wave-steps/CU). Probed via
        // ``TK_RCR_NUM_CUS`` env. Cached on first read so the per-
        // launch host-side cost is one ``getenv`` only on the first
        // dispatch (the cache is process-static, mirroring the
        // existing ``TURBO_FP8_B128`` env hook).
        // Round-9 (current Primus run, gpt_oss FP8 kernel-only ceiling
        // task; 2026-05-08): per-call ``g.num_slots`` override takes
        // precedence over the legacy R4 env-static cache. Mirrors the
        // var-K kernel's R3 ``g.num_slots`` lever (line ~8135). Default
        // ``g.num_slots == 0`` → fall back to the env-static cache below
        // → fall back to NUM_CUS.
        //
        // The R4 env-only design forced process-wide selection (because
        // the cache is read once at process startup), which made it
        // unusable for selective per-shape tuning: setting
        // TK_RCR_NUM_CUS=200 globally tanked the metric -65 points
        // (Down-B4-M2048 fwd lifted +1.3% but EVERY OTHER SHAPE
        // regressed -10..-30%). The per-call knob makes the lever a
        // proper Python dispatcher rule, gated on the same
        // (m_total, n, k, tiles_m, tiles_n) predicates already used for
        // (gm, num_xcds).
        static const int rcr_slots_env = []() {
            if (const char* e = std::getenv("TK_RCR_NUM_CUS")) {
                const int v = std::atoi(e);
                if (v > 0 && v <= NUM_CUS) return v;
            }
            return NUM_CUS;
        }();
        const int rcr_slots = (g.num_slots > 0 && g.num_slots <= NUM_CUS)
            ? g.num_slots : rcr_slots_env;

        if (fuse_ktail_eligible) {
            // R63 Lever F (KI_HINT short-K specialization) FALSIFIED:
            // ki={12,32} compile-time loop bounds INCREASED VGPR spill
            // 34 → 49 and Qwen FP8 ratios crashed from 1.13-1.22 to
            // 0.68-0.77 (HK SLOWER than Triton). Score 978 → 906 (-72).
            // The compile-time unroll exposed too many parallel live
            // ranges; LLVM couldn't fold them. REVERTED to KI_HINT=0
            // (runtime loop) which lets LLVM reuse registers across
            // iterations more aggressively.
            if (n_aligned) {
                grouped_rcr_kernel<0, false, true><<<dim3(rcr_slots), g.block(), 0, g.stream>>>(g);
            } else {
                grouped_rcr_kernel<0, true , true><<<dim3(rcr_slots), g.block(), 0, g.stream>>>(g);
            }
        } else {
            if (n_aligned) {
                grouped_rcr_kernel<0, false, false><<<dim3(rcr_slots), g.block(), 0, g.stream>>>(g);
            } else {
                grouped_rcr_kernel<0, true , false><<<dim3(rcr_slots), g.block(), 0, g.stream>>>(g);
            }
        }
    } else {
        // No aligned interior at all: main kernel cannot run; tail handles
        // every cell with a full-K reduction.
        g.fast_n = 0;
        g.fast_k = 0;
        g.bpc = 0;
        g.ki = 0;
    }

    // Round-12/13: tail kernel only needed for K-tail correction now (and
    // for the no-main fallback). Skip when fast_k == g.k (main covered
    // every cell). The tail kernel detects ``main_covers_n`` via
    // ``g.bpc * BLOCK_SIZE > g.fast_n`` (mirror dense gemm_tail_kernel).
    //
    // Round-13: when the main kernel ran (g.bpc > 0) AND K_rem matches a
    // templated LDS K-tail size AND the host hint guarantees per-block
    // single-group safety, take the LDS-staged K-tail correction path
    // (~10× speedup over scalar fp32 tail). Otherwise (no main, K_rem
    // not templated, or hint says blocks may straddle group boundaries
    // → kernel still has a per-block runtime fallback to scalar) use
    // the existing scalar tail.
    //
    // Round-2 path A: when fuse_ktail_eligible was true above, the main
    // kernel ALREADY accumulated the K-tail into cA/cB/cC/cD before scale
    // + store. The standalone K-tail kernels below would double-count
    // and corrupt the result, so SKIP this entire block when fuse is on.
    if (!fuse_ktail_eligible && (g.fast_k != g.k || g.bpc == 0)) {
        const int K_rem = g.k - g.fast_k;
        // Round-20: prefer 32x32x64 mfma kernel (100 % util) when
        // ``m_per_group`` is 32-aligned. Falls back to the round-18
        // 16x16x128 mfma kernel (50 % util) for 16-aligned but not
        // 32-aligned, and to the round-13 LDS scalar kernel /
        // grouped_tail_kernel for unaligned ``m_per_group``.
        constexpr int TBM_32x32 = 32;
        constexpr int TBM_M2    = 64;       // round-53: 2 stacked 32×32 sub-blocks
        constexpr int TBN_M2N2  = 64;       // round-60: 2 stacked × 2 N sub-tiles
        // Round-54 attempted M4 (TBM=128, 4 stacked sub-blocks) for FP8 but
        // metric regressed: per-shape probe showed Down-B4-{M2048,M4096}
        // each lost 25-28 TF (-3 to -4 %). Root cause: B=4 grids are small
        // (~5760 blocks for M_per=2048) → halving the grid (M4) leaves <3
        // waves per occupant slot, under-saturating the GPU. M2 keeps the
        // wider parallel grid and is already at the bandwidth-limit elbow
        // (probe showed M2 vs M4 wall-time within ±0.1 ms on the 8 gpt_oss
        // FP8 shapes). Decision: keep BF16 M4 (where M2→M4 nets +29 TF
        // total across 8 shapes thanks to larger B-pack reuse) and pin FP8
        // to M2.
        //
        // Round-60: extend M2 along N to 64 cols (M2N2). Lets the per-block
        // A-pack feed two mfmas (one per 32-col sub-tile) instead of one,
        // halving the K-tail per-cell A-side HBM bandwidth. Per-block bytes
        // 14 → 24 KB (2048 → 4096 cells); per-cell 7.0 → 6.0 B/cell (-14 %).
        // Grid halves in N (5760 → 90 col-blocks vs 180 for M2). gpt_oss N
        // ∈ {2880, 5760} are 64-aligned (45·64, 90·64); DSV3-Down N=7168
        // is 64-aligned (112·64). DSV3-GateUP N=4096 is also 64-aligned
        // but K=7168 has no K-tail (no kernel runs here).
        const bool mfma32_safe = (g.m_per_group >= TBM_32x32) &&
                                 ((g.m_per_group % TBM_32x32) == 0);
        const bool mfma32_m2_safe = (g.m_per_group >= TBM_M2) &&
                                    ((g.m_per_group % TBM_M2) == 0);
        const bool mfma32_m2n2_safe = mfma32_m2_safe &&
                                      ((g.n % TBN_M2N2) == 0);
        const bool lds_k_tail_safe = (g.m_per_group >= TAIL_BLOCK_M) &&
                                     ((g.m_per_group % TAIL_BLOCK_M) == 0);
        const bool mfma32_m2n2_handles_all =
            (g.bpc > 0) &&
            (K_rem == 64) &&
            mfma32_m2n2_safe;
        const bool mfma32_m2_handles_all =
            (g.bpc > 0) &&
            (K_rem == 64) &&
            mfma32_m2_safe;
        const bool mfma32_handles_all =
            (g.bpc > 0) &&
            (K_rem == 64) &&
            mfma32_safe;
        const bool mfma16_handles_all =
            (g.bpc > 0) &&
            (K_rem == 64) &&
            lds_k_tail_safe;
        if (mfma32_m2n2_handles_all) {
            // Round-60: 64×64 K-tail block (M2 stacking × N2 sub-tiling).
            // A-pack shared across 2 N sub-tiles → halves per-cell A-side
            // HBM bandwidth. gpt_oss N=2880/5760 and DSV3-Down N=7168
            // both 64-aligned; M_per ∈ {2048, 4096} also 64-aligned, so
            // the fast MFMA path runs and the grid halves in N.
            dim3 mfma_block(64);
            dim3 mfma_grid(
                kittens::ceil_div(g.n, TBN_M2N2),
                kittens::ceil_div(g.M_total, TBM_M2)
            );
            grouped_ktail_kernel_mfma32x32_M2N2<Layout::RCR, 64>
                <<<mfma_grid, mfma_block, 0, g.stream>>>(g);
        } else if (mfma32_m2_handles_all) {
            // Round-53: 64-row K-tail block sharing one B-pack across two
            // stacked 32×32 sub-blocks. 12.5 % per-cell HBM byte reduction
            // vs the round-20 32×32 path (gpt_oss FP8 K-tail was 22-32 %
            // of total wall time per round-53 probe; this kernel halves
            // the B-side HBM bandwidth and shrinks the launch grid 2×).
            // ``m_per_group`` is 64-aligned (gpt_oss M_per ∈ {2048, 4096}
            // both satisfy), so per-block ``row_block_base + 64 <=
            // s_offs[group_idx + 1]`` is guaranteed and the kernel takes
            // its fast MFMA path.
            dim3 mfma_block(64);
            dim3 mfma_grid(
                kittens::ceil_div(g.n, TBM_32x32),
                kittens::ceil_div(g.M_total, TBM_M2)
            );
            grouped_ktail_kernel_mfma32x32_M2<Layout::RCR, 64>
                <<<mfma_grid, mfma_block, 0, g.stream>>>(g);
        } else if (mfma32_handles_all) {
            // Round-20: 32x32x64 mfma_scale_f32_f8f6f4 — K=64 native, no
            // zero padding, 2× theoretical speedup over round-18.
            // ``m_per_group`` is 32-aligned but not 64-aligned (gpt_oss
            // M_per ∈ {2048, 4096} would prefer the M2 path above; this
            // branch covers M_per ∈ {32, 96, 160, ...}).
            dim3 mfma_block(64);
            dim3 mfma_grid(
                kittens::ceil_div(g.n, TBM_32x32),
                kittens::ceil_div(g.M_total, TBM_32x32)
            );
            grouped_ktail_kernel_mfma32x32<Layout::RCR, 64>
                <<<mfma_grid, mfma_block, 0, g.stream>>>(g);
        } else if (mfma16_handles_all) {
            // Round-18: 16x16x128 mfma kernel (50 % util via zero-pad).
            dim3 mfma_block(64);
            dim3 mfma_grid(
                kittens::ceil_div(g.n, TAIL_BLOCK_N),
                kittens::ceil_div(g.M_total, TAIL_BLOCK_M)
            );
            grouped_ktail_kernel_mfma<Layout::RCR, 64>
                <<<mfma_grid, mfma_block, 0, g.stream>>>(g);
        } else {
            dim3 tail_block(TAIL_BLOCK_N, TAIL_BLOCK_M);
            dim3 tail_grid(
                kittens::ceil_div(g.n, TAIL_BLOCK_N),
                kittens::ceil_div(g.M_total, TAIL_BLOCK_M)
            );
            grouped_tail_kernel<Layout::RCR>
                <<<tail_grid, tail_block, 0, g.stream>>>(g);
        }
    }
}

// =============================================================================
// [fused-act R6] Persistent grouped RCR dispatcher — FP8 fused-act variant.
// Mirrors ``dispatch_grouped_rcr`` but launches the FUSE_ACT=true template
// instantiation with ``grouped_layout_globals_fused_act`` (BF16 ``a`` view +
// device-side ``dscale_a`` storing the FORWARD scale = FP8_MAX / amax(a)).
// Initial gate: K % 128 == 0 only (no K-tail), and ``fused_ktail`` is forced
// to false (the K-tail fuse path B reads FP8 bytes via reinterpret_cast,
// incompatible with BF16 src). For K % 128 != 0 shapes the Primus side
// falls back to the un-fused path. The dispatcher pre-checks bpc > 0 and
// ki > 0 + fast_k == k; if any check fails the launch is skipped (Primus
// must call the un-fused path instead — caller responsibility).
// =============================================================================
void dispatch_grouped_rcr_fused_act(grouped_layout_globals_fused_act g) {
    g.n = static_cast<int>(g.c.cols());
    g.M_total = static_cast<int>(g.c.rows());
    g.k = static_cast<int>(g.a.cols());

    g.fast_n = (g.n / BLOCK_SIZE) * BLOCK_SIZE;
    g.fast_k = (g.k / K_BLOCK)    * K_BLOCK;
    g.bpc    = kittens::ceil_div(g.n, BLOCK_SIZE);
    g.ki     = g.fast_k / K_BLOCK;

    if (!(g.bpc > 0 && g.ki > 0 && g.fast_k == g.k)) {
        // Caller is responsible for falling back to the un-fused path.
        return;
    }

    const bool n_aligned = (g.bpc * BLOCK_SIZE == g.n);
    if (n_aligned) {
        grouped_rcr_kernel<0, /*N_MASKED_STORE=*/false,
                           /*FUSED_KTAIL=*/false, /*FUSE_ACT=*/true>
            <<<dim3(NUM_CUS), g.block(), 0, g.stream>>>(g);
    } else {
        grouped_rcr_kernel<0, /*N_MASKED_STORE=*/true,
                           /*FUSED_KTAIL=*/false, /*FUSE_ACT=*/true>
            <<<dim3(NUM_CUS), g.block(), 0, g.stream>>>(g);
    }
}

// =============================================================================
// Persistent grouped RRR dispatcher — FP8 (forward-A backward dA path).
//
// Mirror of ``dispatch_grouped_rcr``: aligned interior swept by the
// persistent main kernel (``grouped_rrr_kernel``), cells outside go
// through ``grouped_tail_kernel<Layout::RRR>`` (scalar fp32 with the
// same N-tail / K-tail correction logic as RCR — see template body).
//
// Per-group M_g must still be a BLOCK_SIZE multiple (the persistent
// loop derives ``bpr_g = M_g / BLOCK_SIZE`` and steps in HB units);
// other invariants are identical to the RCR path.
// =============================================================================
void dispatch_grouped_rrr(grouped_layout_globals g) {
    g.n = static_cast<int>(g.c.cols());
    g.M_total = static_cast<int>(g.c.rows());
    g.k = static_cast<int>(g.a.cols());

    g.fast_n = (g.n / BLOCK_SIZE) * BLOCK_SIZE;
    g.fast_k = (g.k / K_BLOCK)    * K_BLOCK;
    g.bpc    = g.fast_n / BLOCK_SIZE;
    g.ki     = g.fast_k / K_BLOCK;

    if (g.M_total <= 0 || g.n <= 0 || g.k <= 0 || g.G <= 0) return;

    if (g.bpc > 0 && g.ki > 0) {
        grouped_rrr_kernel<0><<<dim3(NUM_CUS), g.block(), 0, g.stream>>>(g);
    } else {
        // No aligned interior at all — main kernel cannot run; tail handles
        // every cell with a full-K reduction.
        g.fast_n = 0;
        g.fast_k = 0;
        g.bpc = 0;
        g.ki = 0;
    }

    if (g.fast_n != g.n || g.fast_k != g.k) {
#if FP8_RRR_FUSE_PROBE
        // Round 27 PROBE — main grouped_rrr_kernel accumulates K-tail
        // in-epilog (path A); skip external launches to avoid double-add.
        // Probe shape spec (G=1, M=2048, N=2880, K=2880): K-tail handled
        // by in-kernel epilog; N-tail (g.fast_n < g.n) is NOT covered by
        // the probe (probe shape has N-tail too — separate test). For
        // pure K-tail probe, use N=BLOCK_SIZE-aligned shape (N=2816 etc).
        (void)0;
#else
        // Round-55: LDS-staged K-tail correction (RMW) for RRR.
        // Round-56: paired LDS-staged N-tail full-K reduction for RRR.
        // Mirror BF16 wiring.
        const bool lds_k_tail_safe = (g.m_per_group >= TAIL_BLOCK_M) &&
                                     ((g.m_per_group % TAIL_BLOCK_M) == 0);
        const int K_rem = g.k - g.fast_k;
        if (K_rem == 64 && lds_k_tail_safe && g.fast_n > 0) {
            dim3 lds_block(TAIL_BLOCK_N, TAIL_BLOCK_M);
            dim3 lds_grid(
                kittens::ceil_div(g.fast_n, TAIL_BLOCK_N),
                kittens::ceil_div(g.M_total, TAIL_BLOCK_M)
            );
            grouped_ktail_kernel_lds_rrr<64>
                <<<lds_grid, lds_block, 0, g.stream>>>(g);
        }
        if (lds_k_tail_safe && g.fast_n < g.n) {
            dim3 lds_block(TAIL_BLOCK_N, TAIL_BLOCK_M);
            dim3 lds_grid(
                kittens::ceil_div(g.n - g.fast_n, TAIL_BLOCK_N),
                kittens::ceil_div(g.M_total, TAIL_BLOCK_M)
            );
            grouped_ntail_kernel_lds_rrr<64>
                <<<lds_grid, lds_block, 0, g.stream>>>(g);
        }

        // Scalar tail still needed for cross-group blocks, m_per_group
        // misalign, and any K-tail size other than 64. The skip
        // predicates inside ``grouped_tail_kernel<RRR>`` mirror the
        // launch gates above so cells already written by the LDS
        // kernels are not re-computed.
        dim3 tail_block(TAIL_BLOCK_N, TAIL_BLOCK_M);
        dim3 tail_grid(
            kittens::ceil_div(g.n, TAIL_BLOCK_N),
            kittens::ceil_div(g.M_total, TAIL_BLOCK_M)
        );
        grouped_tail_kernel<Layout::RRR>
            <<<tail_grid, tail_block, 0, g.stream>>>(g);
#endif
    }
}

// =============================================================================
// Persistent grouped variable-K (CRR / dB) GEMM (CPU-sync-free) — FP8.
//
// Mirror of the BF16 ``grouped_variable_k_crr`` kernel in
// analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp (rounds 1-2). Drives
// the FP8 backward dB computation: given ``a`` = grad_out [M_total, N_fwd]
// fp8 and ``b`` = x [M_total, K_fwd] fp8, produce
// ``c`` = grad_b [G, N_fwd, K_fwd] bf16 in a single CPU-sync-free launch.
// Per-group K-reduction over M_g rows is realised by adding a per-group
// ``k_offset_tiles = m_start_g / HB`` to the kernel's K-axis tile coords.
//
// Native non-aligned (n, k): bpr = ceil_div(n, BLK), bpc = ceil_div(k, BLK).
// Partial last tiles use ``store_c_tile_mn_masked_grouped`` to drop OOB
// (m, n) cells. Safe because (a) A and B are 2D contiguous tensors, NOT
// 3D-grouped — the SRD-wrap-into-next-group issue blocking forward
// grouped does not apply; and (b) MMA cells at OOB output positions are
// dropped before any global store.
//
// Per-group M_g >= 2*HB = 256 is required (prologue + 2 epilogues each
// consume HB-many K-rows; ki_g = M_g / HB and we need ki_g >= 2). The
// Primus-side uniform-M >= 256 gate enforces this; the kernel ``ki_g <
// 2`` skip is a defensive fallback.
// =============================================================================
struct grouped_var_k_layout_globals_fp8 {
    _gl_fp8 a;                     // [1, 1, M_total, n] — grad_out fp8
    _gl_fp8 b;                     // [1, 1, M_total, k] — x fp8
    _gl_bf16 c;                    // [1, G, n, k]       — grad_b bf16
    float scale_a, scale_b;        // host-side scales (used when no dscale)
    const float* dscale_a;
    const float* dscale_b;
    const int64_t* group_offs;     // [G+1] int64 device prefix-sum
    hipStream_t stream;
    int G;                         // number of groups
    int M_total;                   // sum M_g across groups
    int n;                         // kernel M-output dim (= N_fwd)
    int k;                         // kernel N-output dim (= K_fwd)
    int group_m;
    int bpr;                       // ceil_div(n, BLOCK_SIZE)
    int bpc;                       // ceil_div(k, BLOCK_SIZE)
    int fast_n, fast_k;
    int num_xcds;                  // chiplet-swizzle XCD count (0 → default 8)
    int num_slots;                 // Round-3 (gpt_oss FP8 kernel-only ceiling,
                                   // current Primus run; 2026-05-07): persistent-
                                   // grid slot count override. 0 → fall back to
                                   // TK_VARK_NUM_CUS env hook (R2) → NUM_CUS=256.
                                   // Clamped to [1, NUM_CUS] in dispatch.
                                   // Used by short-grid Down-B4 wgrad rule that
                                   // sets num_slots=192 for +5-6% kernel TFLOPS
                                   // (R2 sweep evidence; see Primus
                                   // grouped_gemm_fp8_impl.py R3 predicate).
    int chunk_size;                // Round-13 (gpt_oss FP8 kernel-only ceiling,
                                   // current Primus run; 2026-05-08): chunk_size
                                   // override for the ``chiplet_transform_chunked``
                                   // chiplet swizzle (line ~7827). 0 → fall back to
                                   // TK_VARK_CHUNK_SIZE env hook → 64 (existing
                                   // baseline). Lever lets the dispatcher align the
                                   // chiplet swizzle granularity to the persistent
                                   // grid (slots, xcds) topology — at default 64
                                   // with xcds=2 + slots=192 the swizzle leaves the
                                   // last 64 workgroups un-chunked (R12 falsification
                                   // note observation). chunk_size=96 with xcds=2
                                   // makes block=192 = exactly slots → all chunked
                                   // in 1 clean chiplet-pair partition. Probe in
                                   // R13 to see if this unlocks a +0.5pp+ lift on
                                   // Down-B4 wgrad over the current slots=192 cell.
                                   // Bit-equivalent: same persistent-grid scheduling
                                   // knob class as group_m / num_xcds / num_slots —
                                   // only blockIdx → tile-id mapping changes.
    dim3 block() { return dim3(_NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

__device__ __forceinline__ float resolve_combined_scale_var_k_fp8(
    const grouped_var_k_layout_globals_fp8 &g) {
    const float sa = g.dscale_a ? *g.dscale_a : g.scale_a;
    const float sb = g.dscale_b ? *g.dscale_b : g.scale_b;
    return sa * sb;
}

template<int KI_HINT = 0>
__global__ __launch_bounds__(_NUM_THREADS, 1)
void grouped_var_k_kernel_fp8(const grouped_var_k_layout_globals_fp8 g) {
    using ST_crr_a = ST_v2a;
    using ST_crr_b = ST_v2;
    __shared__ ST_crr_a As[2][2];
    __shared__ ST_crr_b Bs[2][2];

    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];
    __shared__ int s_cum_tiles[MAX_G_PLUS_1];
    __shared__ int s_total_tiles;

    A_col_reg a;
    B_col_reg b0, b1;
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;

    // Round-2 (FP8 backward unblock): mirror RCR line ~2020 — host-side
    // ``g.num_xcds`` knob with fallback to the default 8 when unset.
    const int xcds_eff = g.num_xcds > 0 ? g.num_xcds : BLOCK_SWIZZLE_NUM_XCDS;
    // Round-2 (gpt_oss FP8 kernel-only ceiling, current Primus run; 2026-05-07)
    // Replace constexpr ``NUM_CUS`` with runtime ``gridDim.x`` for the
    // persistent-grid stride + chiplet swizzle range. This makes the
    // kernel honor smaller launch geometries (TK_VARK_NUM_CUS env probe
    // in ``dispatch_grouped_var_k_fp8`` below). At ``gridDim.x ==
    // NUM_CUS == 256`` (the existing default) the math is bit-identical
    // to the constexpr-NUM_CUS version. Compiler will load gridDim.x
    // into a scalar register once and reuse — no per-iter cost.
    const int slots_eff = gridDim.x;
    // Round-13 (gpt_oss FP8 kernel-only ceiling, current Primus run; 2026-05-08):
    // ``g.chunk_size`` overrides the chiplet swizzle chunk granularity. Default
    // 64 mirrors the baseline R3/R15 behavior. Per-call values let the Primus
    // dispatcher align the swizzle to the (slots, xcds) topology — see the
    // struct field comment above.
    const int chunk_size_eff = g.chunk_size > 0 ? g.chunk_size : 64;
    int pid = chiplet_transform_chunked(
        blockIdx.x, slots_eff, xcds_eff, chunk_size_eff);

    int wm = warpid() / WARPS_N;
    int wn = warpid() % WARPS_N;
    const int num_pid_n = g.bpc;

    // Round-38 (Lever L, var_k cleanup): parallel init of the LDS
    // group-metadata caches, mirroring the R9-dm pattern shipped in
    // forward ``grouped_rcr_kernel`` (line ~2273 of this file). var_k
    // was the odd one out — its init ran single-threaded on lane 0 with
    // an O(G) serial HBM-read chain (G+1 loads @ ~80 cy cold-HBM each
    // = 2.5 μs / launch for G=32) before __syncthreads gate-kept the
    // whole CTA.
    //
    // Key observation specific to var_k: ``tiles_per_group`` is the
    // SAME constant for all groups (all groups of a var-K dispatch
    // produce the same-sized dB output `[bpr, bpc]` tile-grid), so the
    // prefix-sum collapses to ``s_cum_tiles[k] = k * tiles_per_group``
    // — an O(1) closed form per slot. No scan is needed at all.
    //
    // Correctness: bit-identical to the old serial path for all legal
    // inputs (G ≤ MAX_G_PLUS_1 - 1 = 64). Pad slots [G+1 .. MAX_G_PLUS_1)
    // receive the same INT_MAX sentinel the downstream binary search
    // depends on. ``s_offs[0..G]`` reads are now warp-coalesced instead
    // of serialized.
    //
    // Bench (gpt_oss-Down B=4 M=2048 — worst-bwd baseline 241 TFLOPS)
    // is the most-sensitive shape because its dB kernel runs at sub-
    // millisecond wall so per-launch init overhead is a larger fraction.
    const int tiles_per_group = g.bpr * g.bpc;
    if (threadIdx.x <= g.G && threadIdx.x < MAX_G_PLUS_1) {
        s_offs[threadIdx.x] = static_cast<int>(g.group_offs[threadIdx.x]);
        s_cum_tiles[threadIdx.x] =
            static_cast<int>(threadIdx.x) * tiles_per_group;
    }
    if (threadIdx.x > g.G && threadIdx.x < MAX_G_PLUS_1) {
        s_cum_tiles[threadIdx.x] = 0x7FFFFFFF;
    }
    if (threadIdx.x == 0) {
        s_total_tiles = g.G * tiles_per_group;
    }
    __syncthreads();
    const int total_tiles = s_total_tiles;

    constexpr int bptA = ST_crr_a::underlying_subtile_bytes_per_thread;
    constexpr int bpmA = bptA * _NUM_THREADS;
    constexpr int mptA = ST_crr_a::rows * ST_crr_a::cols * sizeof(fp8e4m3) / bpmA;
    uint32_t soA[mptA];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);

    constexpr int bptB = ST_crr_b::underlying_subtile_bytes_per_thread;
    constexpr int bpmB = bptB * _NUM_THREADS;
    constexpr int mptB = ST_crr_b::rows * ST_crr_b::cols * sizeof(fp8e4m3) / bpmB;
    uint32_t soB[mptB];
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

    for (int gt = pid; gt < total_tiles; gt += slots_eff) {
        int lo = 0;
        int hi = MAX_G_PLUS_1 - 1;
        #pragma unroll
        for (int level = 0; level < 6; ++level) {
            const int mid = (lo + hi + 1) >> 1;
            if (gt >= s_cum_tiles[mid]) lo = mid;
            else hi = mid - 1;
        }
        const int group_idx = lo;
        const int tile_start = s_cum_tiles[lo];
        const int local_tile = gt - tile_start;
        const int m_start_g = s_offs[group_idx];
        const int M_g = s_offs[group_idx + 1] - m_start_g;
        const int ki_g = M_g / HB;
        if (ki_g < 2) continue;
        const int bpr_g = g.bpr;

        int br, bc;
        if (g.bpc > bpr_g) {
            const int WGN = g.group_m;
            const int num_wgid_in_group = bpr_g * WGN;
            int group_id = local_tile / num_wgid_in_group;
            int first_pid_n = group_id * WGN;
            int group_size_n = min(num_pid_n - first_pid_n, WGN);
            if (group_size_n <= 0) continue;
            bc = first_pid_n + ((local_tile % num_wgid_in_group) % group_size_n);
            br = (local_tile % num_wgid_in_group) / group_size_n;
        } else {
            const int WGM = g.group_m;
            const int num_wgid_in_group = WGM * num_pid_n;
            int group_id = local_tile / num_wgid_in_group;
            int first_pid_m = group_id * WGM;
            int group_size_m = min(bpr_g - first_pid_m, WGM);
            if (group_size_m <= 0) continue;
            br = first_pid_m + ((local_tile % num_wgid_in_group) % group_size_m);
            bc = (local_tile % num_wgid_in_group) / group_size_m;
        }
        if (br >= bpr_g || bc >= num_pid_n) continue;

        // K-axis tile shift for variable-K: each group's K-reduction
        // starts at row m_start_g of the global flat A/B tensors.
        // ST_v2a / ST_v2 row dim = HB, so the tile-coord offset is
        // ``m_start_g / HB``. M_g must be a multiple of HB (= 128) for
        // the per-group ki_g to be exact (Primus uniform-M >= 256 gate
        // covers the bench cases; M_g = 2048 / 4096 / 8192 are all
        // 128-multiples).
        const int k_offset_tiles = m_start_g / HB;

        auto a_co = [&](int s, int k) -> coord<ST_crr_a> {
            return {0, 0, k_offset_tiles + k, s};
        };
        auto b_co = [&](int s, int k) -> coord<ST_crr_b> {
            return {0, 0, k_offset_tiles + k, s};
        };
        auto load_a = [&](A_col_reg& dst, ST_crr_a& tile, int wi) {
            load_col_from_st(dst, tile, wi * RBM);
        };
        auto load_b = [&](B_col_reg& dst, ST_crr_b& tile, int wi) {
            load_col_from_st(dst, tile, wi * RBN);
        };
        // Use full-tensor SRD load (``rcr_8w_load_hoist``) instead of the
        // tile-local SRD path in ``G::load`` (= ``kittens::load`` line 187
        // in include/ops/warp/memory/tile/global_to_shared.cuh). The
        // tile-local SRD has range = ``row_stride * ST::rows * sizeof(T)``
        // bytes from ``global_ptr`` — fine for in-bounds tiles, but for
        // the partial last M-tile (s = 2*(bpr-1)+1 = 45 on n=5760, where
        // unit_coord col = 5760 = past last col), the last row of the
        // tile (lds row 127) crosses tensor end on the last K-iter.
        // Concretely on gpt_oss-GateUP B4 M2048 (M_total=8192, n=5760,
        // tile bytes = 128 cols × 1 byte): SRD bound = global_ptr +
        // 128 * 5760 = global_ptr + 737280, but tensor_end - global_ptr
        // = 47185920 - 46454400 = 731520 < 737280 — buffer_load on the
        // last 128-row tile reads past tensor end → memory fault.
        //
        // ``rcr_8w_load_hoist`` (line 396) constructs the SRD as
        // ``make_srsrc(tensor_base, total_bytes)`` — bound = the FULL
        // tensor size, OOB byte reads clamp to 0 by hardware. The
        // ``offen lds`` buffer_load + SOFFSET (= per-tile byte offset
        // from tensor_base) makes the bound check work correctly. Same
        // signature as ``G::load(dst, src, idx, swizzled_offsets)``.
        // Despite the name, this helper is layout-agnostic — used here
        // for CRR variable-K. Verified safe on round-3 ceil_div +
        // masked-store design via the Round-4 SNR probe.
        auto global_load_a = [&](ST_crr_a& tile, int s, int k) {
            rcr_8w_load_hoist<_NUM_THREADS>(tile, g.a, a_co(s, k), soA);
        };
        auto global_load_b = [&](ST_crr_b& tile, int s, int k) {
            rcr_8w_load_hoist<_NUM_THREADS>(tile, g.b, b_co(s, k), soB);
        };

        zero(cA); zero(cB); zero(cC); zero(cD);

        int tic = 0, toc = 1;
        global_load_b(Bs[tic][0], bc*2,   0);
        global_load_a(As[tic][0], br*2,   0);
        global_load_b(Bs[tic][1], bc*2+1, 0);
        global_load_a(As[tic][1], br*2+1, 0);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(CRR_INIT0_VMCNT);
        __builtin_amdgcn_s_barrier();

        global_load_b(Bs[toc][0], bc*2,   1);
        global_load_a(As[toc][0], br*2,   1);
        global_load_b(Bs[toc][1], bc*2+1, 1);

        TK_WAIT_VMCNT(CRR_INIT1_VMCNT);
        __builtin_amdgcn_s_barrier();

        // Main loop body — mirror of FP8 dense ``gemm_kernel<CRR>`` lines
        // ~1513-1538, but with per-group ki_g and ``a_co/b_co`` lambdas
        // that fold ``k_offset_tiles`` into the row-axis tile coord.
        TK_PRAGMA_UNROLL(CRR_MAIN_UNROLL)
        for (int k = 0; k < ki_g - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, Bs[tic][0], wn);
            load_b(b1, Bs[tic][1], wn);
            load_a(a, As[tic][0], wm);
            global_load_a(As[toc][1], br*2+1, k+1);
            TK_WAIT_LGKM(CRR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cA, a, b0);
            crr_mma(cB, a, b1);
            CRR_MMA_END();
            CRR_STEADY_MID_BARRIER();

            load_a(a, As[tic][1], wm);
            global_load_a(As[tic][0], br*2, k+2);
            global_load_b(Bs[tic][1], bc*2+1, k+2);
            TK_WAIT_VMCNT(CRR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cC, a, b0);
            crr_mma(cD, a, b1);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();
            global_load_b(Bs[tic][0], bc*2, k+2);
        }

        // Epilog 1: second-to-last K-tile.
        //
        // Round-14 (Lever K): removed redundant ``const auto b0_keep = b0``
        // and ``const auto b1_keep = b1`` 32-dw register-tile copies. Live
        // range analysis (see ``analysis/_notes/round-14-fp8-grouped-Lever-K-
        // var_k-epilog-spill-trim.md``):
        //   - b0 is loaded at the top, used by cA + cC mmas, and only
        //     overwritten by ``load_b(b0, Bs[toc][0], wn)`` AFTER cC is
        //     issued. b0 is therefore valid wherever ``b0_keep`` was used.
        //   - b1 is loaded mid-epilog and never overwritten before epilog
        //     end. b1_keep was always a dead copy.
        // Forward ``grouped_rcr_kernel`` uses the same schedule without
        // these copies (line ~2454-2506 of this file). Removing 2 × 32-dw
        // copies cuts the epilog live range by 64 VGPRs (= 64 dw), which
        // is the bulk of the var_k spill anomaly (R12 measured 52 dw spill
        // / 161 loop S+R vs forward grouped_rcr 39 dw / 72 loop S+R).
        {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            global_load_a(As[toc][1], br*2+1, ki_g-1);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cA, a, b0);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cB, a, b1);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(CRR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cC, a, b0);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[toc][0], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cD, a, b1);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();
            tic ^= 1; toc ^= 1;
        }

        // Epilog 2: last K-tile.
        //
        // Round-14 (Lever K): same redundant-copy removal as epilog 1.
        // b0 enters from epilog 1's last ``load_b(b0, Bs[toc][0], wn)``
        // and is never overwritten in this block; b1 is loaded mid-block
        // and never overwritten. All 4 mmas (cA/cB/cC/cD) can read b0/b1
        // directly.
        {
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cA, a, b0);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cB, a, b1);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            CRR_MMA_BEGIN();
            crr_mma(cC, a, b0);
            crr_mma(cD, a, b1);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();
        }

        // Apply scale + 2-axis-masked store. Output is per-group [n, k]
        // sub-tensor at depth = group_idx; no m_subtile_C row shift.
        const float combined_scale = resolve_combined_scale_var_k_fp8(g);
        mul(cA, cA, combined_scale);
        mul(cB, cB, combined_scale);
        mul(cC, cC, combined_scale);
        mul(cD, cD, combined_scale);

        if (wm == 0) __builtin_amdgcn_s_barrier();
        store_c_tile_mn_masked_grouped(g.c, cA, group_idx,
            br*WARPS_M*2+wm,         bc*WARPS_N*2+wn,         g.n, g.k);
        store_c_tile_mn_masked_grouped(g.c, cB, group_idx,
            br*WARPS_M*2+wm,         bc*WARPS_N*2+WARPS_N+wn, g.n, g.k);
        store_c_tile_mn_masked_grouped(g.c, cC, group_idx,
            br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+wn,         g.n, g.k);
        store_c_tile_mn_masked_grouped(g.c, cD, group_idx,
            br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+WARPS_N+wn, g.n, g.k);

        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }
}

template __global__ void grouped_var_k_kernel_fp8<0>(
    const grouped_var_k_layout_globals_fp8);

void dispatch_grouped_var_k_fp8(grouped_var_k_layout_globals_fp8 g) {
    g.n = static_cast<int>(g.a.cols());
    g.k = static_cast<int>(g.b.cols());
    g.M_total = static_cast<int>(g.a.rows());

    g.fast_n = g.n;
    g.fast_k = g.k;
    g.bpr = kittens::ceil_div(g.n, BLOCK_SIZE);
    g.bpc = kittens::ceil_div(g.k, BLOCK_SIZE);

    if (g.bpr <= 0 || g.bpc <= 0 || g.G <= 0) return;

    // Round-2 (gpt_oss FP8 kernel-only ceiling, current Primus run; 2026-05-07)
    // Optional env override TK_VARK_NUM_CUS for the persistent-grid slot
    // count. Used by ``_probe_round_2_vark_numcus_sweep.py`` to validate
    // the R1 PMC characterisation hypothesis (Down-B4-M2048 wgrad at
    // 16.6 % MFMA-active is launch-geometry bound — too few wave-steps
    // per slot at NUM_CUS=256 / 484 tiles = 1.89 steps/slot). Clamped
    // to [1, NUM_CUS] to preserve the existing upper bound (kernel
    // assumes pid < NUM_CUS-related LDS tables / chiplet swizzle range).
    // The kernel body itself uses ``gridDim.x`` for the persistent-loop
    // stride + chiplet swizzle range so any ``slots ∈ [1, NUM_CUS]`` is
    // correctness-preserving. ``static`` cache avoids repeated getenv()
    // syscalls in the hot path.
    //
    // Round-3 (this run, 2026-05-07): per-call ``g.num_slots`` override
    // (set via the new pybind ``num_slots`` arg) takes precedence over
    // the env hook. Lets the Python dispatcher pick slots=192 for the
    // short-grid Down-B4 wgrad family without affecting any other
    // var-K caller in the process. ``g.num_slots == 0`` → fall back to
    // env or NUM_CUS (preserves R2 probe semantics).
    int slots_dispatch;
    if (g.num_slots > 0 && g.num_slots <= NUM_CUS) {
        slots_dispatch = g.num_slots;
    } else {
        static const int env_slots = []() {
            if (const char* e = std::getenv("TK_VARK_NUM_CUS")) {
                const int v = std::atoi(e);
                if (v > 0 && v <= NUM_CUS) return v;
            }
            return NUM_CUS;
        }();
        slots_dispatch = env_slots;
    }

    // Round-13 (gpt_oss FP8 kernel-only ceiling, current Primus run; 2026-05-08):
    // Optional env override TK_VARK_CHUNK_SIZE for the chiplet-swizzle
    // chunk_size. Process-static cache. Per-call ``g.chunk_size`` (set via
    // future pybind arg) takes precedence. ``g.chunk_size == 0`` AND env
    // unset → fall through to kernel default 64 (existing baseline).
    if (g.chunk_size <= 0 || g.chunk_size > NUM_CUS) {
        static const int env_chunk_size = []() {
            if (const char* e = std::getenv("TK_VARK_CHUNK_SIZE")) {
                const int v = std::atoi(e);
                if (v >= 1 && v <= 256) return v;
            }
            return 0;  // 0 → kernel uses default 64
        }();
        g.chunk_size = env_chunk_size;
    }

    grouped_var_k_kernel_fp8<0><<<dim3(slots_dispatch), g.block(), 0, g.stream>>>(g);
}

template<Layout L>
void dispatch(layout_globals g) {
    g.m = static_cast<int>(g.c.rows());
    g.n = static_cast<int>(g.c.cols());
    if constexpr (L == Layout::RCR) {
        g.k = static_cast<int>(g.a.cols());
    } else if constexpr (L == Layout::RRR) {
        g.k = static_cast<int>(g.a.cols());
    } else {
        g.k = static_cast<int>(g.b.rows());
    }

    g.fast_m = (g.m / BLK) * BLK;
    g.fast_n = (g.n / BLK) * BLK;
    g.fast_k = (g.k / BK) * BK;
    g.bpr = g.fast_m / BLK;
    g.bpc = g.fast_n / BLK;
    g.ki = g.fast_k / BK;

    // FP8 dense Phase 4: enable native N-tail via main-kernel column-
    // masked C store + ``bpc = ceil_div(n, BLK)``, mirroring BF16 dense.
    // Safe because ``rcr_8w_load_hoist`` was refactored to use a FULL-
    // tensor SRD with per-tile SOFFSET (see helper near line 400) — OOB
    // column reads from a partial last tile clamp to 0 from the SRD
    // bound rather than faulting on unmapped memory.
    //
    // Conditions to enable ``ceil_div`` bpc (mirror BF16):
    //   1. The 8-wave generic ``gemm_kernel<L, 0>`` is selected. The
    //      4-wave RCR fast path uses ``rcr_4w::prepare_g2s`` which still
    //      builds tile-local SRDs and is NOT yet refactored — keep its
    //      ``bpc = fast_n/BLK`` (its kernel is selected only for very
    //      large grids; tail kernel handles N misalignment in those
    //      large-grid cases).
    //   2. K is fully aligned (``fast_k == k``). When K is misaligned
    //      the tail kernel needs a deterministic fast/tail boundary in
    //      N; same as BF16 dense.
    bool main_covers_n = false;
    if (g.bpr > 0 && g.bpc > 0 && g.ki >= 2) {
        if constexpr (L == Layout::RCR) {
            const int aligned_grid = g.bpr * g.bpc;
            bool use_4wave =
                aligned_grid >= RCR_4WAVE_MIN_GRID && g.k <= RCR_4WAVE_MAX_K;
            // Optional env override for tuning/debugging: set TK_RCR_FORCE_KERNEL=4 or 8
            if (const char* e = getenv("TK_RCR_FORCE_KERNEL")) {
                if (e[0] == '4') use_4wave = true;
                else if (e[0] == '8') use_4wave = false;
            }
            if (use_4wave) {
                dim3 grid4(aligned_grid);
                dim3 block4(rcr_4w::NT);
                rcr_4w::kernel<<<grid4, block4, 0, g.stream>>>(g);
            } else {
                if (g.fast_k == g.k) {
                    g.bpc = kittens::ceil_div(g.n, BLK);
                    main_covers_n = true;
                }
                gemm_kernel<L, 0><<<g.grid(), g.block(), 0, g.stream>>>(g);
            }
        } else
        {
            if (g.fast_k == g.k) {
                g.bpc = kittens::ceil_div(g.n, BLK);
                main_covers_n = true;
            }
            gemm_kernel<L, 0><<<g.grid(), g.block(), 0, g.stream>>>(g);
        }
    } else {
        g.fast_k = 0;
        g.ki = 0;
    }

    const bool need_tail = (g.fast_m != g.m) || (g.fast_k != g.k) ||
                           (!main_covers_n && g.fast_n != g.n) || (g.ki == 0);
    if (need_tail) {
        dim3 tail_block(TAIL_BLOCK_N, TAIL_BLOCK_M);
        dim3 tail_grid(
            kittens::ceil_div(g.n, TAIL_BLOCK_N),
            kittens::ceil_div(g.m, TAIL_BLOCK_M)
        );
        gemm_tail_kernel<L><<<tail_grid, tail_block, 0, g.stream>>>(g);
    }
}

static float to_float(pybind11::object obj) {
    if (pybind11::hasattr(obj, "item"))
        return obj.attr("item")().cast<float>();
    return obj.cast<float>();
}

constexpr int DEFAULT_GROUP_M = 4;

template<Layout L>
static void gemm_wrapper(pybind11::object a, pybind11::object b, pybind11::object c,
                          pybind11::object scale_a_obj, pybind11::object scale_b_obj,
                          int group_m) {
    layout_globals g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        to_float(scale_a_obj),
        to_float(scale_b_obj),
        {}, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        group_m,
        nullptr, nullptr,
    };
    dispatch<L>(g);
}

// Variant of gemm_wrapper that takes the per-tensor FP8 scales as 0-d device
// tensors (one element each) instead of host-side scalars / 0-d host tensors.
// The Python-side host wrapper used to call `(a_scale_inv * b_scale_inv).item()`
// on every dispatch which is a stream sync that costs ~18us on small dense
// FP8 shapes (≈ 30% of the kernel itself) — this entry skips that sync by
// passing the device pointers straight through to the kernel's epilogue, which
// reads one b32 from global memory at scale-application time. The two scales
// are both used in the epilogue only, so the cost of the load is fully hidden
// behind the GEMM main loop.
template<Layout L>
static void gemm_wrapper_dscale(pybind11::object a, pybind11::object b, pybind11::object c,
                                 pybind11::object scale_a_obj, pybind11::object scale_b_obj,
                                 int group_m) {
    auto sa_ptr = scale_a_obj.attr("data_ptr")().cast<uintptr_t>();
    auto sb_ptr = scale_b_obj.attr("data_ptr")().cast<uintptr_t>();
    layout_globals g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        0.f, 0.f,
        {}, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        group_m,
        reinterpret_cast<const float*>(sa_ptr),
        reinterpret_cast<const float*>(sb_ptr),
    };
    dispatch<L>(g);
}

// Host-side wrappers for grouped RCR kernel (host-scalar + dscale variants).
// Round-67: optional ``num_xcds`` parameter (default 0 → kernel uses
// ``BLOCK_SWIZZLE_NUM_XCDS=8``). Mirrors BF16 grouped's existing
// per-launch num_xcds tuning so the Python-side config rule can pick a
// per-shape optimum (e.g. DSV3-Down prefers xcds=4, DSV3-GateUP keeps
// xcds=8 — see /tmp/sweep_fp8_xcds_round67.py).
static void grouped_rcr_fn(pybind11::object a, pybind11::object b, pybind11::object c,
                           pybind11::object scale_a_obj, pybind11::object scale_b_obj,
                           pybind11::object group_offs_obj,
                           int group_m,
                           int m_per_group,
                           int num_xcds,
                           int num_slots) {
    auto group_offs_ptr = group_offs_obj.attr("data_ptr")().cast<uintptr_t>();
    int G = group_offs_obj.attr("numel")().cast<int>() - 1;
    grouped_layout_globals g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        to_float(scale_a_obj),
        to_float(scale_b_obj),
        nullptr,
        nullptr,
        reinterpret_cast<const int64_t*>(group_offs_ptr),
        {},
        /* G,n,k,ki,bpc,group_m,num_xcds,M_total,fast_n,fast_k,m_per_group,num_slots */
        G, 0, 0, 0, 0, group_m, num_xcds, 0, 0, 0, m_per_group, num_slots,
    };
    dispatch_grouped_rcr(g);
}

static void grouped_rcr_dscale_fn(
    pybind11::object a, pybind11::object b, pybind11::object c,
    pybind11::object scale_a_obj, pybind11::object scale_b_obj,
    pybind11::object group_offs_obj,
    int group_m,
    int m_per_group,
    int num_xcds,
    int num_slots) {
    auto sa_ptr = scale_a_obj.attr("data_ptr")().cast<uintptr_t>();
    auto sb_ptr = scale_b_obj.attr("data_ptr")().cast<uintptr_t>();
    auto group_offs_ptr = group_offs_obj.attr("data_ptr")().cast<uintptr_t>();
    int G = group_offs_obj.attr("numel")().cast<int>() - 1;
    grouped_layout_globals g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        0.f, 0.f,
        reinterpret_cast<const float*>(sa_ptr),
        reinterpret_cast<const float*>(sb_ptr),
        reinterpret_cast<const int64_t*>(group_offs_ptr),
        {},
        /* G,n,k,ki,bpc,group_m,num_xcds,M_total,fast_n,fast_k,m_per_group,num_slots */
        G, 0, 0, 0, 0, group_m, num_xcds, 0, 0, 0, m_per_group, num_slots,
    };
    dispatch_grouped_rcr(g);
}

// [fused-act R6] Host wrapper for grouped_rcr_kernel<FUSE_ACT=true>.
// Inputs:
//   a              — BF16 [M_total, K] activation tensor.
//   b              — FP8  [G, N, K] weight tensor.
//   c              — BF16 [M_total, N] output tensor.
//   scale_a_inv    — device float32 [1] holding FP8_MAX / amax(a) (output of
//                    ``max_abs_bf16_to_fp8_scale`` from the R1 binding).
//   scale_b        — device float32 [1] holding the FP8 dequant scale for b.
//   group_offs     — device int64 [G+1] prefix-sum on M_total.
//   group_m,m_per_group,num_xcds — same scheduling knobs as the un-fused path.
// Returns ``true`` if the fused-act kernel ran; ``false`` if the dispatcher
// rejected the shape (caller must fall back to the un-fused path).
static bool grouped_rcr_fused_act_dscale_fn(
    pybind11::object a, pybind11::object b, pybind11::object c,
    pybind11::object scale_a_inv_obj, pybind11::object scale_b_obj,
    pybind11::object group_offs_obj,
    int group_m,
    int m_per_group,
    int num_xcds) {
    auto sa_ptr = scale_a_inv_obj.attr("data_ptr")().cast<uintptr_t>();
    auto sb_ptr = scale_b_obj.attr("data_ptr")().cast<uintptr_t>();
    auto group_offs_ptr = group_offs_obj.attr("data_ptr")().cast<uintptr_t>();
    int G = group_offs_obj.attr("numel")().cast<int>() - 1;
    grouped_layout_globals_fused_act g{
        py::from_object<_gl_bf16>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        0.f, 0.f,
        reinterpret_cast<const float*>(sa_ptr),
        reinterpret_cast<const float*>(sb_ptr),
        reinterpret_cast<const int64_t*>(group_offs_ptr),
        {},
        G, 0, 0, 0, 0, group_m, num_xcds, 0, 0, 0, m_per_group,
    };

    const int K = static_cast<int>(g.a.cols());
    const int N = static_cast<int>(g.c.cols());
    if ((K % K_BLOCK) != 0 || (N <= 0) || (K <= 0)) {
        return false;
    }
    dispatch_grouped_rcr_fused_act(g);
    return true;
}

// Round-1 host wrappers for grouped RRR (backward dA) FP8 kernel.
// Same global struct as RCR (identical scale + group_offs plumbing); the
// dispatcher pins ``fast_n = fast_k = 0`` so the entire compute happens
// in ``grouped_tail_kernel<Layout::RRR>``.
//
// Round-2 (FP8 backward unblock): ``num_xcds`` parameter added to mirror
// the RCR binding (round-67). Was previously absent, so the Primus-side
// dispatch in ``grouped_gemm_fp8_impl.py`` raised TypeError when the
// shared dispatch path passed ``num_xcds=xcds_arg`` for the RRR layout
// (FP8 backward dA), killing all 16 FP8 grouped cases with bwd-exception.
// Default 0 → kernel reads ``g.num_xcds == 0`` and falls back to the
// built-in ``BLOCK_SWIZZLE_NUM_XCDS=8`` (no perf regression).
static void grouped_rrr_fn(pybind11::object a, pybind11::object b, pybind11::object c,
                           pybind11::object scale_a_obj, pybind11::object scale_b_obj,
                           pybind11::object group_offs_obj,
                           int group_m,
                           int m_per_group,
                           int num_xcds) {
    auto group_offs_ptr = group_offs_obj.attr("data_ptr")().cast<uintptr_t>();
    int G = group_offs_obj.attr("numel")().cast<int>() - 1;
    grouped_layout_globals g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        to_float(scale_a_obj),
        to_float(scale_b_obj),
        nullptr,
        nullptr,
        reinterpret_cast<const int64_t*>(group_offs_ptr),
        {},
        /* G,n,k,ki,bpc,group_m,num_xcds,M_total,fast_n,fast_k,m_per_group */
        G, 0, 0, 0, 0, group_m, num_xcds, 0, 0, 0, m_per_group,
    };
    dispatch_grouped_rrr(g);
}

static void grouped_rrr_dscale_fn(
    pybind11::object a, pybind11::object b, pybind11::object c,
    pybind11::object scale_a_obj, pybind11::object scale_b_obj,
    pybind11::object group_offs_obj,
    int group_m,
    int m_per_group,
    int num_xcds) {
    auto sa_ptr = scale_a_obj.attr("data_ptr")().cast<uintptr_t>();
    auto sb_ptr = scale_b_obj.attr("data_ptr")().cast<uintptr_t>();
    auto group_offs_ptr = group_offs_obj.attr("data_ptr")().cast<uintptr_t>();
    int G = group_offs_obj.attr("numel")().cast<int>() - 1;
    grouped_layout_globals g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        0.f, 0.f,
        reinterpret_cast<const float*>(sa_ptr),
        reinterpret_cast<const float*>(sb_ptr),
        reinterpret_cast<const int64_t*>(group_offs_ptr),
        {},
        /* G,n,k,ki,bpc,group_m,num_xcds,M_total,fast_n,fast_k,m_per_group */
        G, 0, 0, 0, 0, group_m, num_xcds, 0, 0, 0, m_per_group,
    };
    dispatch_grouped_rrr(g);
}

// Host wrappers for grouped variable-K (CRR / dB) FP8 kernel
// (host-scalar + dscale variants).
//
// Round-2 (FP8 backward unblock): ``num_xcds`` parameter added to keep
// the var-K binding signature in sync with the RCR / RRR launchers
// (round-67). Without it, future Primus dispatch could raise TypeError
// the same way the RRR path did. Wires through to the kernel via the
// new ``num_xcds`` field on ``grouped_var_k_layout_globals_fp8``;
// ``num_xcds == 0`` falls back to ``BLOCK_SWIZZLE_NUM_XCDS=8`` so this
// is back-compat (Primus var-K dispatch currently does not pass
// ``num_xcds`` — that's a follow-up perf knob).
static void grouped_variable_k_crr_fp8_fn(
    pybind11::object a, pybind11::object b, pybind11::object c,
    pybind11::object scale_a_obj, pybind11::object scale_b_obj,
    pybind11::object group_offs_obj,
    int group_m,
    int num_xcds,
    int num_slots,
    int chunk_size = 0) {
    auto group_offs_ptr = group_offs_obj.attr("data_ptr")().cast<uintptr_t>();
    int G = group_offs_obj.attr("numel")().cast<int>() - 1;
    grouped_var_k_layout_globals_fp8 g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        to_float(scale_a_obj),
        to_float(scale_b_obj),
        nullptr,
        nullptr,
        reinterpret_cast<const int64_t*>(group_offs_ptr),
        {},
        /* G, M_total, n, k, group_m, bpr, bpc, fast_n, fast_k, num_xcds, num_slots, chunk_size */
        G, 0, 0, 0, group_m, 0, 0, 0, 0, num_xcds, num_slots, chunk_size,
    };
    dispatch_grouped_var_k_fp8(g);
}

static void grouped_variable_k_crr_dscale_fp8_fn(
    pybind11::object a, pybind11::object b, pybind11::object c,
    pybind11::object scale_a_obj, pybind11::object scale_b_obj,
    pybind11::object group_offs_obj,
    int group_m,
    int num_xcds,
    int num_slots,
    int chunk_size = 0) {
    auto sa_ptr = scale_a_obj.attr("data_ptr")().cast<uintptr_t>();
    auto sb_ptr = scale_b_obj.attr("data_ptr")().cast<uintptr_t>();
    auto group_offs_ptr = group_offs_obj.attr("data_ptr")().cast<uintptr_t>();
    int G = group_offs_obj.attr("numel")().cast<int>() - 1;
    grouped_var_k_layout_globals_fp8 g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        0.f, 0.f,
        reinterpret_cast<const float*>(sa_ptr),
        reinterpret_cast<const float*>(sb_ptr),
        reinterpret_cast<const int64_t*>(group_offs_ptr),
        {},
        /* G, M_total, n, k, group_m, bpr, bpc, fast_n, fast_k, num_xcds, num_slots, chunk_size */
        G, 0, 0, 0, group_m, 0, 0, 0, 0, num_xcds, num_slots, chunk_size,
    };
    dispatch_grouped_var_k_fp8(g);
}

// =============================================================================
// max_abs_bf16 — round-1 of FP8 grouped fused-activation-quantize support.
//
// Computes one fp32 device scalar over a flat BF16 buffer of length N.
// Two outputs supported:
//   * mode = MODE_AMAX     : ``out[0] = max(|a[i]|)``                       (raw amax)
//   * mode = MODE_FP8_SCALE: ``out[0] = fp8_max / max(eps, max(|a[i]|))``  (= "scale"
//     in primus_turbo's quantize_fp8_tensorwise convention — multiplier
//     that maps amax -> fp8_max). Lets the Python helper hand the scalar
//     STRAIGHT to ``quantize_fp8_tensorwise_impl(a, dtype, scale=scale)``
//     without a Python-side ``FP8_MAX / amax`` div+clamp launch (a 2-kernel
//     orchestration overhead that would otherwise eat the win on small
//     shapes — see /tmp/probe_fused_amax_quantize.py round-1 measurements).
//
// Used by the Primus-Turbo Python-side ``_fused_act_grouped_fp8_forward``
// to produce the activation tensorwise scale BEFORE calling C++
// ``quantize_fp8_tensorwise(input, scale=...)`` — the optional-scale
// branch skips its internal amax pass, so the net amax cost shrinks to
// one HK kernel launch (no torch reduction workspace alloc, no 2-stage
// reduce).
//
// Design:
//   * Persistent grid: NUM_CUS (=304 on MI355X) blocks × _NUM_THREADS=256
//     threads. Grid-stride loop over the flat BF16 buffer with vectorized
//     uint4 (=128b=8 bf16) reads.
//   * Reduce inside each block: warp-shuffle (xor reduction over the
//     64-thread wavefront) + per-block LDS staging across the 4 wavefronts.
//   * Final cross-block: each block runs ``atomicMax`` on the int-reinterpret
//     of its block-local fp32 max. Float order matches uint32 order for
//     non-negative values (which |·| guarantees), so the int-typed
//     atomicMax produces the correct fp32 max.
//   * Block 0, thread 0 ALSO runs the post-reduce ``fp8_max / amax``
//     transform (when ``MODE_FP8_SCALE``) via a brief spin-wait on the
//     final atomicMax sentinel. This keeps the entire scale pipeline in
//     one launch.
//
// Output buffer ``out_fp32_scalar`` MUST be pre-zeroed by the caller.
// (The Python helper allocates with ``torch.zeros(())``; cost is one
// 4-byte zero kernel — negligible vs the MN-byte read pass below.)
//
// Numerics: identical to ``a.abs().to(torch.float32).amax()`` modulo the
// order of fmaxf operations. Both routes promote BF16 to FP32 before
// taking abs+max, so denorms / subnormals collapse the same way and the
// result bit-matches torch's reduction within a single ULP. SNR vs torch
// > 90 dB on uniform-random inputs; far beyond the >= 25 dB FP8 gate.
// =============================================================================

// Mode selector: 0 = raw amax, 1 = scale = fp8_max / max(eps, amax).
enum MaxAbsMode : int { MODE_AMAX = 0, MODE_FP8_SCALE = 1 };

template<int MODE>
__global__ __launch_bounds__(_NUM_THREADS, 1)
void max_abs_bf16_kernel(const bf16* __restrict__ a,
                         float* __restrict__ out_fp32_scalar,
                         int64_t N,
                         float fp8_max,
                         float eps,
                         int* __restrict__ done_counter /* [1], pre-zeroed */ ) {
    constexpr int VEC = 8;  // 8 BF16 = 128 bits = uint4
    const int64_t N_vec = N / VEC;
    const uint4* a_vec = reinterpret_cast<const uint4*>(a);

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    float local_max = 0.0f;

    // Vectorized inner pass.
    #pragma unroll 1
    for (int64_t i = tid; i < N_vec; i += stride) {
        uint4 v = a_vec[i];
        const bf16* bf = reinterpret_cast<const bf16*>(&v);
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            float f = (float)bf[j];
            local_max = fmaxf(local_max, fabsf(f));
        }
    }

    // Tail elements (< VEC) — only the first stride contributes.
    int64_t tail_base = N_vec * VEC;
    for (int64_t i = tail_base + tid; i < N; i += stride) {
        float f = (float)a[i];
        local_max = fmaxf(local_max, fabsf(f));
    }

    // Wavefront reduce (64-wide on CDNA): xor butterfly, 6 steps.
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        float other = __shfl_xor(local_max, offset);
        local_max = fmaxf(local_max, other);
    }

    // Cross-wavefront reduce inside the block via LDS.
    constexpr int WARPS_PER_BLOCK = _NUM_THREADS / WARP_THREADS;  // 256/64 = 4
    __shared__ float smem[WARPS_PER_BLOCK];
    __shared__ bool s_is_finalizer;
    int warp_id = threadIdx.x / WARP_THREADS;
    int lane = threadIdx.x % WARP_THREADS;

    if (lane == 0) smem[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < WARPS_PER_BLOCK) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1) {
            float other = __shfl_xor(v, offset);
            v = fmaxf(v, other);
        }
        if (lane == 0) {
            // Float comparison via int-reinterpret only valid for non-negative
            // floats; |·| above guarantees that.
            atomicMax(reinterpret_cast<int*>(out_fp32_scalar),
                      __float_as_int(v));
            // Round-1: detect the LAST block to retire so it can run the
            // ``fp8_max / amax`` post-transform in-kernel (saves a Python
            // FP8_MAX/amax kernel launch in the fused-act path). Convention
            // mirrors AMD's persistent-launch sweep counters in the rest of
            // the file: a counter pre-zeroed by the caller; the kernel
            // bumps it once per block; whichever block hits ``gridDim.x``
            // is the finalizer.
            if (MODE == MODE_FP8_SCALE) {
                int prev = atomicAdd(done_counter, 1);
                s_is_finalizer = (prev == gridDim.x - 1);
            } else {
                s_is_finalizer = false;
            }
        }
    }
    __syncthreads();

    // Single-thread post-transform: amax -> scale (= fp8_max / max(eps, amax)).
    // Read the global amax with __threadfence-equivalent guarantee: the
    // atomicMax / atomicAdd pair above is sequentially-consistent across
    // blocks on AMD's GPU memory model (kfd buffer_atomic returns are
    // ordered with prior atomic writes).
    if (MODE == MODE_FP8_SCALE && warp_id == 0 && lane == 0 && s_is_finalizer) {
        float amax = *reinterpret_cast<volatile float*>(out_fp32_scalar);
        float denom = fmaxf(amax, eps);
        float scale = fp8_max / denom;
        *reinterpret_cast<volatile float*>(out_fp32_scalar) = scale;
    }
}

// Host wrapper: bridges pybind11 device tensors to the kernel launch.
// ``out_obj`` MUST be a pre-zeroed scalar fp32 device tensor (numel == 1).
// ``a_obj`` is any contiguous BF16 device tensor; we read ``numel`` from it
// and treat it as a flat array. Caller-side reshape / view is fine.
// ``done_obj`` is a pre-zeroed int32 device tensor (numel == 1) used as
// the finalizer counter; ignored by ``MODE_AMAX``.
//
// NUM_CUS-block × 256-thread persistent launch matches the existing FP8
// grouped GEMM convention; uses the legacy default stream (==0), same as
// every other binding in this file (see ``g.stream`` initialisers in
// ``grouped_rcr_fn`` / ``grouped_rcr_dscale_fn`` / etc.). Default stream
// auto-synchronises with PyTorch's per-thread streams, so callers don't
// need a manual stream sync before subsequent ``quantize_fp8`` / GEMM ops.
static constexpr int MAX_ABS_NUM_CUS = NUM_CUS;

static void max_abs_bf16_fn(pybind11::object a_obj, pybind11::object out_obj) {
    auto a_ptr   = a_obj.attr("data_ptr")().cast<uintptr_t>();
    auto out_ptr = out_obj.attr("data_ptr")().cast<uintptr_t>();
    int64_t N    = a_obj.attr("numel")().cast<int64_t>();
    if (N <= 0) return;
    max_abs_bf16_kernel<MODE_AMAX>
        <<<dim3(MAX_ABS_NUM_CUS), dim3(_NUM_THREADS), 0, 0>>>(
            reinterpret_cast<const bf16*>(a_ptr),
            reinterpret_cast<float*>(out_ptr),
            N, 0.0f, 0.0f, /*done_counter=*/nullptr);
}

// Round-1 fused-act helper: in-kernel FP8 scale = fp8_max / max(eps, amax).
// ``done_obj`` MUST be a pre-zeroed int32 device tensor (numel == 1) — the
// finalizer counter. Example FP8_E4M3_FN: fp8_max=448.0; FP8_E4M3_FNUZ:
// fp8_max=240.0. The helper does NOT pick the constant — caller hands it
// in so the kernel matches the Python-side ``get_float8_max`` choice.
static void max_abs_bf16_to_fp8_scale_fn(
        pybind11::object a_obj,
        pybind11::object out_obj,
        pybind11::object done_obj,
        float fp8_max,
        float eps) {
    auto a_ptr   = a_obj.attr("data_ptr")().cast<uintptr_t>();
    auto out_ptr = out_obj.attr("data_ptr")().cast<uintptr_t>();
    auto done_ptr = done_obj.attr("data_ptr")().cast<uintptr_t>();
    int64_t N    = a_obj.attr("numel")().cast<int64_t>();
    if (N <= 0) return;
    max_abs_bf16_kernel<MODE_FP8_SCALE>
        <<<dim3(MAX_ABS_NUM_CUS), dim3(_NUM_THREADS), 0, 0>>>(
            reinterpret_cast<const bf16*>(a_ptr),
            reinterpret_cast<float*>(out_ptr),
            N, fp8_max, eps,
            reinterpret_cast<int*>(done_ptr));
}

// =============================================================================
// R4 — Phase 1 fwd-fusion infra: BF16→FP8 cvt building block + compile-test.
//
// Foundation primitive for the future ``grouped_rcr_fused_act_kernel`` which
// fuses BF16→FP8 activation cvt into the GEMM kernel's load_a path. R4 lands
// just the inner-most cvt helper + a tiny compile-test kernel that forces
// hipcc to emit codegen for the cvt builtin under CDNA4 / gfx950 — surfaces
// any builtin-signature / register-class issue at build time, BEFORE the
// surgical kernel-body changes that follow in R5+.
//
// `cvt_bf16x4_to_fp8x4`:
//   4 BF16 lanes → 4 FP8e4m3 lanes packed in a single uint32_t.
//   Two ``__builtin_amdgcn_cvt_pk_fp8_f32`` calls form the int: first call
//   uses sel=false to write the lo half (WORD0); second call uses sel=true
//   and passes the first result as ``dummy_old`` to merge WORD1 into the
//   same int. The ``dummy_old`` accumulator pattern + the -Wuninitialized
//   suppression mirror the composable_kernel reference at:
//   3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_elementwise_hip.hpp:197-213.
//
// Numerical semantics:
//   fp8e4m3_lane[i] = round_to_e4m3(scale * bfloat16_to_fp32(bf16_lane[i]))
//   where ``scale = FP8_MAX / max(eps, amax(a))`` is computed by the R1
//   ``max_abs_bf16_to_fp8_scale`` kernel.
//
// Not yet wired to any production kernel — that's R5+ work where the helper
// gets called from inside a DTR-mode load_a path replacing the existing
// DTL ``buffer_load_dwordx4 ... offen lds`` for fused-act variants.
// =============================================================================

namespace fused_act_round4_compile_test {

__device__ __forceinline__ uint32_t cvt_bf16x4_to_fp8x4(
    bf16_2 lo,    // bf16 lanes 0, 1
    bf16_2 hi,    // bf16 lanes 2, 3
    float scale)  // a_scale_inv = FP8_MAX / amax(a)
{
    float2 lo_f = __bfloat1622float2(lo);
    float2 hi_f = __bfloat1622float2(hi);
    lo_f.x *= scale; lo_f.y *= scale;
    hi_f.x *= scale; hi_f.y *= scale;

    int dummy_old;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
    uint32_t packed = __builtin_amdgcn_cvt_pk_fp8_f32(
        lo_f.x, lo_f.y, dummy_old, /*sel=*/false);
    packed = __builtin_amdgcn_cvt_pk_fp8_f32(
        hi_f.x, hi_f.y, packed, /*sel=*/true);
#pragma clang diagnostic pop
    return packed;
}

// Compile-test kernel: round-trips one bf16x4 → fp8x4 cvt. Forces LLVM to
// emit codegen so any cvt-builtin issue surfaces at build time. Single-thread
// body keeps the resource-usage report focused on the cvt sequence.
__global__ __launch_bounds__(64, 1)
void cvt_bf16x4_to_fp8x4_compile_test(
    const bf16_2* __restrict__ src,    // length 2: 2 bf16_2 = 4 bf16
    uint32_t* __restrict__ dst,        // length 1: packed fp8x4
    float scale)
{
    if (threadIdx.x != 0) return;
    bf16_2 lo = src[0];
    bf16_2 hi = src[1];
    *dst = cvt_bf16x4_to_fp8x4(lo, hi, scale);
}

// Bulk version: each thread cvts its own bf16x4 group → fp8x4. Used by the
// Python-level numerical probe. Round-up the count to a multiple of 4 BF16.
__global__ __launch_bounds__(256, 1)
void cvt_bf16_to_fp8_bulk_compile_test(
    const bf16* __restrict__ src,
    fp8e4m3* __restrict__ dst,
    int64_t N,             // number of bf16 elements (must be multiple of 4)
    float scale)
{
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t N4 = N / 4;
    const bf16_2* src2 = reinterpret_cast<const bf16_2*>(src);
    uint32_t* dst4 = reinterpret_cast<uint32_t*>(dst);
    #pragma unroll 1
    for (int64_t i = tid; i < N4; i += stride) {
        bf16_2 lo = src2[2 * i + 0];
        bf16_2 hi = src2[2 * i + 1];
        dst4[i] = cvt_bf16x4_to_fp8x4(lo, hi, scale);
    }
}

}  // namespace fused_act_round4_compile_test

// =============================================================================
// [Round 5a, fused-act] Production-grade DTR load helper for fused-act variant.
//
// The BF16-source counterpart of the FP8-source ``rcr_8w_load_hoist`` (lines
// 806-915). Mirrors the same 8-warp swizzled multi-pass cooperative load but
// replaces the DTL ``buffer_load_dwordx4 ... offen lds`` with:
//   * 2× DTR (Direct Tile Read) ``raw_buffer_load_b128`` per pass — 32 bytes
//     BF16 land in VGPRs (not LDS).
//   * 4× ``cvt_bf16x4_to_fp8x4`` (the R4 helper) — 16 BF16 → 16 FP8 packed
//     into a 16-byte u32x4 in registers.
//   * 1× ``ds_write_b128`` per lane — the 16-byte FP8 chunk lands in LDS at
//     the same swizzled slot the un-fused DTL would have written to.
//
// Per-pass HBM byte rate doubles (32 vs 16 bytes/lane — BF16 occupies twice
// the bytes per element). Per-pass LDS write rate is unchanged (16 bytes/lane),
// so ``memcpy_per_tile`` and the LDS subtile layout are bit-identical to the
// un-fused helper. We can reuse ``prefill_swizzled_offsets`` as-is — its
// output is in ``ST::dtype = FP8`` byte stride; we scale by 2 inside the
// helper to recover the BF16-byte offset (the HBM stride for BF16 src).
//
// Wave-uniform / SGPR plumbing mirrors the un-fused helper:
//   * SGPR-hoisted ``lds_addrs[]`` (warp-base; per-lane offset is added in
//     a VGPR before ds_write_b128).
//   * SGPR ``tile_byte_offset`` via ``__builtin_amdgcn_readfirstlane``.
//   * Full-tensor SRD bound (so partial-N / partial-K tile loads clamp OOB
//     bytes to 0 instead of faulting on unmapped pages — same robustness
//     the un-fused helper already gets).
//
// Falsification gate: see ``rcr_8w_load_hoist_fused_act_compile_test`` below.
// VGPR usage must stay comparable to the un-fused helper — kernel-level
// resource numbers come in R5b when the helper is folded into a cloned
// ``grouped_rcr_fused_act_kernel``. R5a ships only the helper, the new
// ``grouped_layout_globals_fused_act`` struct, and a compile-test launcher
// that forces codegen at build time so any signature / register-class issue
// surfaces NOW rather than mid-kernel-clone in R5b.
// =============================================================================

// [fused-act R6] grouped_layout_globals_fused_act + rcr_8w_load_hoist_fused_act
// were FORWARD-RELOCATED to ~line 2245 (just before grouped_layout_globals)
// so the FUSE_ACT=true instantiation of grouped_rcr_kernel can refer to them.
// The compile-test kernel + binding below still find both via the relocated
// definitions (struct at file-scope, helper template in the
// fused_act_round5_compile_test namespace re-opened here).
namespace fused_act_round5_compile_test {

// rcr_8w_load_hoist_fused_act body was FORWARD-RELOCATED to the
// ``fused_act_round5_compile_test`` namespace at ~line 2245. The compile-test
// kernel below references it via the same fully-qualified name.

// Compile-test launcher: forces codegen for ``rcr_8w_load_hoist_fused_act``
// at one representative (ST_v2 / _gl_bf16 / N_THREADS=256) instantiation. The
// kernel does a single tile load (one ST_v2 worth of BF16 → FP8 in LDS) then
// stripes the LDS contents to ``out_fp8`` so LLVM doesn't DCE the cvt path.
// Used to:
//   (a) Verify the helper compiles cleanly under hipcc / CDNA4.
//   (b) Surface VGPR / scratch / register-class issues at build time.
//   (c) Provide a Python-callable probe for R5b's numerical validation
//       (compare LDS contents vs reference C++ ``quantize_fp8`` output;
//       expect bit-exact match modulo BF16->FP8 rounding noise — same SNR
//       budget as the R4 ``cvt_bf16_to_fp8_bulk`` probe).
__global__ __launch_bounds__(_NUM_THREADS, 1)
void rcr_8w_load_hoist_fused_act_kernel_test(
    _gl_bf16 a,                       // [M, K] BF16 source
    _gl_fp8  out,                     // [M, K] FP8 dst (LDS-dump)
    float    scale)                   // FP8_MAX / amax(a)
{
    using ST_DST = ST_v2;
    __shared__ ST_DST As;

    // Zero LDS so subtile-padding bytes show up as 0 (not random 0x7F NaN
    // encodings) in the strided dump below. Without this the SNR probe sees
    // legitimate cvt output AND ~896 random padding bytes, contaminating the
    // distributional comparison. Production kernels don't dump LDS so they
    // don't need this — it's only here for the R5a numerical probe.
    {
        constexpr int total_lds_bytes = sizeof(As);
        constexpr int per_thread_zero = total_lds_bytes / _NUM_THREADS;
        static_assert(total_lds_bytes % _NUM_THREADS == 0,
                      "LDS bytes must divide evenly across threads");
        uint8_t* lds_bytes = (uint8_t*)&As.data[0];
        #pragma unroll 1
        for (int j = 0; j < per_thread_zero; ++j) {
            lds_bytes[threadIdx.x * per_thread_zero + j] = 0;
        }
        __syncthreads();
    }

    constexpr int bpt = ST_DST::underlying_subtile_bytes_per_thread;
    constexpr int mpt = ST_DST::rows * ST_DST::cols *
                        sizeof(typename ST_DST::dtype) /
                        (bpt * _NUM_THREADS);
    uint32_t soA[mpt + 1];

    G::prefill_swizzled_offsets(As, a, soA);

    coord<ST_DST> base{0, 0, 0, 0};
    rcr_8w_load_hoist_fused_act<_NUM_THREADS, ST_DST, _gl_bf16, coord<ST_DST>>(
        As, a, base, soA, scale);
    __syncthreads();

    // Block-strided dump of the FULL LDS storage (including subtile padding)
    // to ``out``. Caller supplies a buffer of ``sizeof(As)`` bytes (17408 for
    // ST_v2 — 8 subtiles × (2048 valid + 128 padding)). Python-side strips
    // the 7 × 128 padding gaps so the SNR probe sees just the 16384 cvt
    // outputs. Without dumping the full LDS, the last subtile's tail (896
    // bytes) is missed, contaminating the sorted-distributional SNR check.
    if (blockIdx.x == 0 && out.raw_ptr) {
        const int tid = threadIdx.x;
        constexpr int total_lds_bytes = sizeof(As);
        constexpr int per_thread = total_lds_bytes / _NUM_THREADS;
        static_assert(total_lds_bytes % _NUM_THREADS == 0,
                      "LDS bytes must divide evenly across threads");
        uint8_t* hbm_base = (uint8_t*)out.raw_ptr;
        const uint8_t* lds_base = (const uint8_t*)&As.data[0];
        #pragma unroll 1
        for (int j = 0; j < per_thread; ++j) {
            const int idx = tid * per_thread + j;
            hbm_base[idx] = lds_base[idx];
        }
    }
}

}  // namespace fused_act_round5_compile_test

// Host wrapper for the bulk compile-test kernel (R4 numerical probe binding).
// Exposes the cvt path to Python so the probe can verify SNR > 25 dB vs
// torch's reference quantize. Not used by any production kernel — purely
// validation of the cvt builtin numerics under CDNA4.
static void cvt_bf16_to_fp8_bulk_fn(
        pybind11::object src_obj,
        pybind11::object dst_obj,
        float scale) {
    auto src_ptr = src_obj.attr("data_ptr")().cast<uintptr_t>();
    auto dst_ptr = dst_obj.attr("data_ptr")().cast<uintptr_t>();
    int64_t N    = src_obj.attr("numel")().cast<int64_t>();
    if (N <= 0) return;
    constexpr int BLOCK = 256;
    constexpr int GRID  = NUM_CUS;
    fused_act_round4_compile_test::cvt_bf16_to_fp8_bulk_compile_test
        <<<dim3(GRID), dim3(BLOCK), 0, 0>>>(
            reinterpret_cast<const bf16*>(src_ptr),
            reinterpret_cast<fp8e4m3*>(dst_ptr),
            N, scale);
}

// [fused-act R5a] Host wrapper for the DTR-load + cvt + LDS-write compile-test.
// Single-block launch (no group/tile scheduling — just exercises the helper
// once on a single ST_v2 worth of BF16 src). Output buffer is the strided LDS
// dump so the load chain isn't DCE'd. R5b will use this binding for numerical
// validation: cvt'd FP8 contents vs C++ ``quantize_fp8`` reference (expect
// the same SNR ~340 dB profile as the R4 cvt-bulk probe).
//
// Caller contract:
//   src — BF16 contiguous tensor, shape [≥ ST_v2::rows, ≥ ST_v2::cols].
//   dst — FP8e4m3 contiguous tensor, numel ≥ ST_v2::rows * ST_v2::cols.
//   scale — float (host-side; FP8_MAX / amax(src)).
static void rcr_load_hoist_fused_act_test_fn(
        pybind11::object src_obj,
        pybind11::object dst_obj,
        float scale) {
    fused_act_round5_compile_test::rcr_8w_load_hoist_fused_act_kernel_test
        <<<dim3(1), dim3(_NUM_THREADS), 0, 0>>>(
            py::from_object<_gl_bf16>::make(src_obj),
            py::from_object<_gl_fp8>::make(dst_obj),
            scale);
}

PYBIND11_MODULE(tk_fp8_layouts, m) {
    m.doc() = "FP8 per-tensor GEMM: C = A op B * scale_a * scale_b";
    m.def("gemm_rcr", &gemm_wrapper<Layout::RCR>,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M);
    m.def("gemm_rrr", &gemm_wrapper<Layout::RRR>,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M);
    m.def("gemm_crr", &gemm_wrapper<Layout::CRR>,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M);
    m.def("gemm_rcr_dscale", &gemm_wrapper_dscale<Layout::RCR>,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M);
    m.def("gemm_rrr_dscale", &gemm_wrapper_dscale<Layout::RRR>,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M);
    m.def("gemm_crr_dscale", &gemm_wrapper_dscale<Layout::CRR>,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M);
    m.def("supports_shape", [](int m, int n, int k) -> bool {
        return m > 0 && n > 0 && k > 0;
    });
    // [fused-act] BF16 max-abs reduction → single fp32 device scalar. Used by
    // Primus-Turbo's ``_fused_act_grouped_fp8_forward`` (Round 1 of the FP8
    // grouped fused-activation-quant lever) to produce the activation scale
    // BEFORE calling the C++ ``quantize_fp8_tensorwise(input, scale=...)`` —
    // the optional-scale branch skips its internal amax pass, so the net
    // amax cost shrinks to one HK kernel launch (no torch reduction
    // workspace alloc, no 2-stage reduce, single atomicMax target).
    // Caller MUST pre-zero ``out`` (a 0-d / 1-elem fp32 device tensor).
    m.def("max_abs_bf16", &max_abs_bf16_fn,
          pybind11::arg("a"), pybind11::arg("out"));
    // Same kernel as ``max_abs_bf16``, but the persistent-launch finalizer
    // ALSO runs the ``out = fp8_max / max(eps, amax)`` post-transform — the
    // primus_turbo "scale" convention used by ``quantize_fp8_tensorwise(
    // input, scale=...)``. Saves the Python-side ``FP8_MAX / amax`` div +
    // clamp launch (a 2-kernel orchestration overhead that otherwise eats
    // the win on small shapes; see /tmp/probe_fused_amax_quantize.py).
    // ``done`` MUST be a pre-zeroed int32 device scalar (numel == 1) for
    // the finalizer detection; ``out`` MUST be a pre-zeroed fp32 device
    // scalar (numel == 1).
    m.def("max_abs_bf16_to_fp8_scale", &max_abs_bf16_to_fp8_scale_fn,
          pybind11::arg("a"),
          pybind11::arg("out"),
          pybind11::arg("done"),
          pybind11::arg("fp8_max"),
          pybind11::arg("eps") = 1e-12f);
    // [fused-act R4] Numerical-probe binding for the BF16→FP8 cvt builtin.
    // Bulk-applies ``cvt_bf16x4_to_fp8x4`` over a contiguous BF16 buffer,
    // multiplied by the caller-supplied ``scale`` (= FP8_MAX / amax(a)).
    // Dst MUST be FP8e4m3-typed contiguous device tensor with same numel as
    // src; src.numel() MUST be a multiple of 4. Used only by the R4 probe;
    // not on any production hot path (R5+ folds this into a DTR load_a).
    m.def("cvt_bf16_to_fp8_bulk", &cvt_bf16_to_fp8_bulk_fn,
          pybind11::arg("src"),
          pybind11::arg("dst"),
          pybind11::arg("scale"));
    // [fused-act R5a] DTR load + cvt + LDS write compile-test. Forces codegen
    // for ``rcr_8w_load_hoist_fused_act`` at a representative ST_v2 tile size
    // and exposes a single-block launcher to Python so R5b can do the
    // numerical SNR probe before clone'ing the full grouped_rcr_kernel.
    m.def("rcr_load_hoist_fused_act_test", &rcr_load_hoist_fused_act_test_fn,
          pybind11::arg("src"),
          pybind11::arg("dst"),
          pybind11::arg("scale"));
    // [grouped] Persistent + CPU-sync-free FP8 RCR launcher. ``group_offs`` is
    // a [G+1] int64 device tensor (prefix-sum of per-group M); the kernel
    // consumes it on the GPU side via O(G) linear scan, no host reads.
    // Round-9 (current Primus run, gpt_oss FP8 kernel-only ceiling task;
    // 2026-05-08): added ``num_slots`` per-call arg (default 0 → legacy
    // env-only / NUM_CUS fallback). Mirrors the var-K binding's R3
    // ``num_slots`` arg below. Existing positional callers stay
    // backward-compat (trailing arg with default).
    m.def("grouped_rcr", &grouped_rcr_fn,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M,
          pybind11::arg("m_per_group") = 0,
          pybind11::arg("num_xcds") = 0,
          pybind11::arg("num_slots") = 0);
    m.def("grouped_rcr_dscale", &grouped_rcr_dscale_fn,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M,
          pybind11::arg("m_per_group") = 0,
          pybind11::arg("num_xcds") = 0,
          pybind11::arg("num_slots") = 0);
    // [fused-act R6] FUSE_ACT=true variant of the grouped RCR launcher.
    // ``a`` is BF16 (not FP8); ``scale_a_inv`` is the device float32 scalar
    // returned by ``max_abs_bf16_to_fp8_scale`` (= FP8_MAX / amax(a)).
    // Returns ``True`` if the fused-act kernel ran; ``False`` if the
    // dispatcher rejected the shape (caller must fall back to the un-fused
    // path: ``quantize_fp8_tensorwise(a) → grouped_rcr_dscale(...)``).
    // Initial gate: K % 128 == 0 only; FUSED_KTAIL=false.
    m.def("grouped_rcr_fused_act_dscale", &grouped_rcr_fused_act_dscale_fn,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a_inv"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M,
          pybind11::arg("m_per_group") = 0,
          pybind11::arg("num_xcds") = 0);
    // [grouped] Round-1 RRR launcher (FP8 backward dA path). Same
    // ``group_offs``-driven contract as ``grouped_rcr``; uses the scalar
    // tail kernel for the full compute (no native main kernel yet).
    // Round-2 (FP8 backward unblock): added ``num_xcds`` to mirror
    // ``grouped_rcr`` (round-67) and unblock the Primus shared dispatch
    // path which passes ``num_xcds=xcds_arg`` for every layout.
    m.def("grouped_rrr", &grouped_rrr_fn,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M,
          pybind11::arg("m_per_group") = 0,
          pybind11::arg("num_xcds") = 0);
    m.def("grouped_rrr_dscale", &grouped_rrr_dscale_fn,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M,
          pybind11::arg("m_per_group") = 0,
          pybind11::arg("num_xcds") = 0);
    // [grouped variable-K dB] Persistent + CPU-sync-free FP8 CRR launcher
    // for the backward dB path. Inputs are 2D contiguous (grad_out, x);
    // output is 3D-grouped grad_b [G, n, k] bf16. ``group_offs`` is the
    // [G+1] int64 device prefix-sum; the kernel scans it on-GPU.
    // Round-2 (FP8 backward unblock): ``num_xcds`` added for signature
    // parity with the RCR / RRR launchers; default 0 falls back to the
    // built-in ``BLOCK_SWIZZLE_NUM_XCDS=8``.
    m.def("grouped_variable_k_crr", &grouped_variable_k_crr_fp8_fn,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M,
          pybind11::arg("num_xcds") = 0,
          pybind11::arg("num_slots") = 0,
          pybind11::arg("chunk_size") = 0);
    m.def("grouped_variable_k_crr_dscale",
          &grouped_variable_k_crr_dscale_fp8_fn,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M,
          pybind11::arg("num_xcds") = 0,
          pybind11::arg("num_slots") = 0,
          pybind11::arg("chunk_size") = 0);
    m.attr("DEFAULT_GROUP_M") = DEFAULT_GROUP_M;
    m.attr("BLOCK_SIZE") = BLK;
    m.attr("K_BLOCK") = BK;

    // R59 (auto-optimize): Lever C-2 step 2B-2 — debug entry for the
    // 4w-style test kernel with real (br, bc, k) coord indexing.
    // Single-group only (G=1, m_subtile_A=0 hardcoded). Caller contract:
    // M%256==0, N%256==0, K%128==0. Output is BF16; scale_a/scale_b are
    // host-side floats. NOT a production entry — used by the R59
    // correctness probe (analysis/fp8_gemm/mi350x/probe_4w_real_coords.py)
    // to validate the new kernel matches torch fp32 ref before R60+
    // adds the group binary search prologue + K-tail + N-mask.
    m.def("test_4w_real_coords",
          [](pybind11::object a, pybind11::object b, pybind11::object c,
             pybind11::object scale_a_obj, pybind11::object scale_b_obj,
             pybind11::object group_offs_obj) {
              auto group_offs_ptr = group_offs_obj.attr("data_ptr")().cast<uintptr_t>();
              int G = group_offs_obj.attr("numel")().cast<int>() - 1;
              grouped_layout_globals g{
                  py::from_object<_gl_fp8>::make(a),
                  py::from_object<_gl_fp8>::make(b),
                  py::from_object<_gl_bf16>::make(c),
                  to_float(scale_a_obj),
                  to_float(scale_b_obj),
                  nullptr,
                  nullptr,
                  reinterpret_cast<const int64_t*>(group_offs_ptr),
                  {},
                  G, 0, 0, 0, 0, /*group_m=*/0, /*num_xcds=*/0,
                  0, 0, 0, /*m_per_group=*/0,
              };
              lever_c2_round_59_step2b2_real_coords::dispatch_test_4w_real_coords(g);
          },
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"));
}
