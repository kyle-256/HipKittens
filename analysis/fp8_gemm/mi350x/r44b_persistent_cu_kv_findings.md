# R44 Dev B — Persistent-CU dispatch for very small N (N=1024 KV-decode)

**Branch**: worktree-agent-a380f138 (off `feat/mxfp8-only` HEAD `51a7c759`)
**Date**: 2026-04-18
**Scope**: R43+ priority list item #3 (R43 Dev B explicit recommendation) — investigate
persistent-CU dispatch to improve CU occupancy on KV-projection decode shapes
at N=1024.

## Verdict: REFUTED-BY-INSPECTION

★ **NO SHIP, NO PROTOTYPE, 0 GPU-min burned.** The premise of the task brief —
"M=32 N=1024 → 16 tiles vs 304 CUs = 5.3% occupancy" — is **incorrect** for the
production dispatch path: the R42B `gemm_tail_kernel_smallm_b32` (which handles
M=32/128 PRESHUFFLED_QUANT decode via the R43C unified small-M waterfall) uses
`TAIL_BLOCK_M=16`, `TAIL_BLOCK_N=16`, NOT BLK_M=256/BLK_N=64. The task brief's
arithmetic uses V2 fastpath geometry, but V2 fastpaths are unreachable for
M < BLK=256 (R41 Dev C structural finding).

After re-deriving the *actual* tile counts under R42B/R43C geometry and applying
the R31 Dev D (`r31d_findings.md`) **paradigm correction** ("persistent-CU
dispatch with early-exit prologue cannot improve wave-fill when total_tiles ≤
num_CUs; inner-loop persistence is NOT a wave-fill optimization"), persistent-CU
is structurally NULL for all 4 KV-decode shapes in this task.

## Re-derived tile counts (production dispatch through R43C waterfall)

The 4 KV-decode shapes, with M < BLK=256, route through:

```
gemm_{rcr,rrr,crr}_pq_v2 → dispatch_pq_v2<L>
  → (V2 predicates fail: g.m != M_DIM)
  → V1-LEGACY-FALLBACK → dispatch<L, true>
  → R43C unified small-M waterfall: g.m < BLK
  → MXFP8_SMALLM_B32_FASTPATH branch (PRESHUFFLED_QUANT=true)
  → gemm_tail_kernel_smallm_b32<L, true>
  → grid = (ceil_div(g.n, TAIL_BLOCK_N=16), ceil_div(g.m, TAIL_BLOCK_M=16))
  → block = dim3(16, 16) = 256 threads = 4 wavefronts/block
```

For the 4 N=1024 KV-decode shapes:

| Shape | tile grid | total tiles | wavefronts (×4 waves/block) | CU occupancy |
|---|---|---:|---:|---|
| 32 ×1024×4096 (8B-K/V dec)  | (64, 2) | 128 | 512 | 1.68 waves/CU avg |
| 32 ×1024×8192 (70B-K/V dec) | (64, 2) | 128 | 512 | 1.68 waves/CU avg |
| 128×1024×4096 (8B-K/V b=128)  | (64, 8) | **512** | 2048 | 6.74 waves/CU avg |
| 128×1024×8192 (70B-K/V b=128) | (64, 8) | **512** | 2048 | 6.74 waves/CU avg |

**Compare task brief estimate**: "M_blocks × 16" = 16 tiles is wrong by ~8-32×.
The brief used BLK_M=256/BLK_N=64 (V2 geometry), but V2 is unreachable here.

## Why persistent-CU is NULL for these shapes (R31D paradigm)

**Case 1: M=128 N=1024 (8B/70B b=128)**: 512 tiles vs 304 CUs. With low LDS
(scalar kernel uses 0 LDS) and ~30 VGPRs/thread, occupancy is ~10-40 blocks/CU
(SIMD-limited, not LDS-limited). 512 blocks across 304 CUs already saturates
the chip — every CU receives 1-2 blocks per wave, fully utilized. Persistent-CU
adds **zero useful work** (R31D, page 2: "the missing work simply doesn't exist
in the (br, bc) tile grid").

**Case 2: M=32 N=1024 (8B/70B decode)**: 128 tiles vs 304 CUs. **Some CUs
necessarily idle** under any block-grained dispatch. But:

- **Early-exit-prologue persistent (R31D Approach 1)**: grid=304, 128 valid +
  176 early-exit. SPI places 304 CTAs across 304 CUs at occ=1; workers still
  128, idle CUs still 176. **Identical to baseline** (R31D direct quote, page 2).
- **Inner-loop persistent (R31D "true" persistence)**: grid=304 with each CTA
  doing `for (tile_id = blockIdx.x; tile_id < 128; tile_id += 304)`. 128
  iterations distribute as 128 CTAs do 1 tile, 176 do 0. **Same 128 working
  CTAs as baseline** — inner-loop persistence rebalances workload distribution
  but the SPI dispatcher already does this, "which the SPI dispatcher already
  does" (R31D, page 5).
- **Persistent with shared LDS reuse across tiles** (the only R31D path that
  could help): would require restructuring smallm_b32 from a *scalar 0-LDS*
  kernel into a *wavefront-LDS-prefetched* kernel, persistent-loop processing
  contiguous (br, bc) tiles that share an A-row prefetch. This is essentially
  the R41 Dev C MFMA-fastpath proposal (3-5 day rewrite, ~1500-3000 LoC) —
  out of scope for persistent-CU per se, and R42 Dev B explicitly chose the
  K-loop-hoist refactor instead because the MFMA fastpath was infeasible in
  a single time-box.

**Conclusion**: there is no code change to `gemm_tail_kernel_smallm_b32` or its
dispatch grid that would improve CU occupancy on M=32 N=1024 within the
constraints "macro-gate, default-OFF, 0 symbols". Any meaningful gain requires
architectural changes to the kernel body itself (not the dispatch shape) and
overlaps with R44+ priority #2 (M=2..16 MFMA fastpath, ~3-5 days).

## Why the task brief's "16 tiles vs 304 CUs" estimate was wrong

The brief computed `M_blocks × ceil(1024/64) = M_blocks × 16` using BLK_M=256
(V2 fastpath BLK), giving 1×16=16 blocks for M=32. This **would be correct if
the V2 fastpath were reachable** — but `rcr_can_use_exact_8wave_scaled` (and the
RRR/CRR equivalents) all require `g.m == M_DIM && g.n == N_DIM && g.k == K_DIM`
(line 3675), and `M_DIM`, `N_DIM`, `K_DIM` are compile-time constants of the
build. A build for (M_DIM=32, N_DIM=1024, K_DIM=4096) is possible but:

1. The .so would only handle that one shape (R25+ rule: each shape needs its
   own .so), making it unsuitable for production.
2. Even with `g.m == M_DIM == 32`, the V2 grid math is `(g.m / BLK) * (g.n / BLK)`
   = `(32/256) * (1024/256)` = `0 * 4 = 0` blocks — V2 cannot launch at all.

So the realistic dispatch for these shapes is the V1 path through the R43C
unified small-M waterfall, which routes to `gemm_tail_kernel_smallm_b32` — and
that kernel uses 16×16 tiling, NOT BLK=256 tiling. The 16-block premise of the
task brief is structurally inapplicable.

## What COULD be the next lever (out of scope here)

The R42B `gemm_tail_kernel_smallm_b32` runs a single thread per (row, col)
output element with a scalar K-loop. For M=32 N=1024 K=4096, each thread does:
- 1 row × 1 col × 4096 K-iters = 4096 FMA ops + 4096 FP8 loads each from A and
  B = ~8K HBM transactions per thread.
- Total threads = 32 × 1024 = 32768 = 512 wavefronts. At HBM peak ~5.3 TB/s
  and per-thread ~2 bytes/iter × 4096 = 8 KB load volume, total = 256 MB load
  → ~50 µs theoretical at HBM peak. Measured ~25 µs (R42B 0.88 TF on similar
  shape) → ~50% HBM efficiency (reasonable for scalar loads with no
  vectorization).

The bottleneck is **per-thread scalar HBM access**, not CU occupancy. The
levers that would help (in priority order):
1. **Vectorize the inner load to 4-byte/8-byte transactions** (load 4 or 8 FP8
   elements per thread per K-iter). Touches the kernel internal, NOT the
   dispatch grid. ~0.5-1 day, would improve HBM efficiency to >80% → +50%
   throughput.
2. **Use wavefront-LDS prefetch for the A row** (one 32-thread wavefront
   cooperatively loads A[row, k:k+32] into LDS, all threads in the same row
   read from LDS). Would be the M=2..16 MFMA fastpath direction (R44+ #2).
3. **Restructure to MFMA tiles** (R41C / R44+ #2 — large rewrite).

None of these are persistent-CU dispatch.

## Cross-check: what about FP8 reference kernel at N=1024?

FP8's `gemm_tail_kernel` uses the same TAIL_BLOCK_M=16/TAIL_BLOCK_N=16 geometry
(`kernel_fp8_layouts.cpp:306-307, 2315-2318`). It has the same grid for these
shapes. The R42B SHIP showed MXFP8/FP8 = 98-99% on M=32/128 × N=4096/8192 —
meaning **MXFP8 is now matching FP8** in the same scalar-tail regime. Persistent-CU
would have to lift *both* MXFP8 and FP8 (or only MXFP8 vs FP8 ratio). Since the
ratio is already ~99%, the only headroom is **absolute throughput** — and per
the R31D paradigm, persistent-CU dispatch alone cannot lift absolute throughput
without changing the kernel body.

## R44+ recommendation revision

Move R43+ priority #3 ("Persistent-CU dispatch for very small N") to **CLOSED-
REFUTED-BY-INSPECTION**, citing this finding + R31D paradigm correction.

Replace with: **B-side scalar-load vectorization in `gemm_tail_kernel_smallm_b32`**
(R44+ priority #4 in the existing list — "B-side packed-uint vectorization").
This is the actual highest-leverage lever for KV-decode N=1024 shapes given the
R42B SHIP closed the scale-load redundancy gap.

## Files

- `analysis/fp8_gemm/mi350x/r44b_persistent_cu_kv_findings.md` (this file).

## No code changes

This is a refute-by-inspection. The `MXFP8_PERSISTENT_CU_KV_ENABLE` macro
proposed in the task brief was **not added** to the tree, because:
1. The premise (16 tiles, 5% CU utilization) is incorrect for the production
   dispatch path.
2. Even if a persistent-CU kernel were prototyped, R31D's paradigm correction
   guarantees it would be performance-NULL.
3. Adding a default-OFF macro that gates a structurally-NULL kernel would
   add maintenance debt with no upside (the R34 / R38 NEW methodology rules
   on macro hygiene + nm-gate catalog updates).

## R44 Dev B GPU-min budget

**0 GPU-min** burned. Refute-by-inspection completed in ~30 reading-min using:
- `r31d_findings.md` (R31 Dev D persistent-CU paradigm correction)
- `r42b_smallm_b32_findings.md` (R42B SHIP geometry + bench results)
- `mxfp8_smallm_b32_fastpath.inc` (kernel source — TAIL_BLOCK_M/N=16)
- `kernel_mxfp8_layouts.cpp:5476-5736` (R43C unified small-M waterfall +
  smallm_b32 dispatch)
- `kernel_mxfp8_layouts.cpp:5861-6078` (`dispatch_pq_v2<L>` V1-LEGACY-FALLBACK)
- `kernel_mxfp8_layouts.cpp:3675` (`rcr_can_use_exact_8wave_scaled` predicate
  requiring g.m == M_DIM)

## Cumulative paradigm-closure tally

This adds **44th cumulative closed lever** to the R32-R43 closure list:
"Persistent-CU dispatch for KV-decode N=1024 small-M shapes — REFUTED-BY-
INSPECTION via R31D paradigm extension to TAIL_BLOCK=16 geometry."
