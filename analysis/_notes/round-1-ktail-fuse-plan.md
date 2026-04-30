# Round 1 — K-tail fuse plan & architecture survey

Goal: Replace the 4 standalone `grouped_ktail_kernel_*` launches (RMW
on `g.c`) with a single in-kernel fused K-tail epilog inside the
persistent `grouped_rcr_kernel` (BF16 + FP8 RCR), per the run's hard
constraint: "K-tail must be native to the main kernel; no second
launch, no host-pad K".

## Current architecture (post round-67, HEAD `fde4b692`)

Main grouped kernels:

- `analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp::grouped_kernel<L, KI_HINT>`
  (single template covers RCR/RRR/CRR; uses shared `device_gemm_tile_body`).
  - LDS slots: `ST_A As[2][2]`, `ST_B Bs[2][2]` (4 × 8 KB = 32 KB each
    side, 64 KB total). Double-buffer × 2 sub-tiles per side.
  - Register tiles in body: `A_reg_t A_tile`, `B_reg_t B_tile_0/1`,
    plus the 4 `rt_fl<HALF_REG_BLOCK_M=64, HALF_REG_BLOCK_N=32, col_l, rt_16x16_s>`
    accumulators `C_accum[2][2]` held by `grouped_kernel`.
  - Schedule: prologue (4 G::load) → main_loop_iter ×(num_tiles-2)/2 →
    Epilog 1 (second-to-last K-tile pair) → Epilog 2 (last K-tile pair).
  - At end of Epilog 2: all `As/Bs` LDS slots have been consumed into
    `A_tile/B_tile_*` registers; the LDS bytes are physically idle and
    can be re-used for K-tail staging.

- `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp::grouped_rcr_kernel<KI_HINT, N_MASKED_STORE>`
  (RCR-only; line 1864). Equivalent layout: `ST_v2 As[2][2]; ST_v2 Bs[2][2]`,
  4 register accumulators `cA/cB/cC/cD` of `rt_fl<RBM=64, RBN=32, col_l, rt_16x16_s>`,
  `A_row_reg a; B_row_reg b0, b1`. Schedule: prologue → main loop
  `for k in [0, ki_dyn-2)` → Epilog 1 → Epilog 2. Same LDS-idle-after-Epilog-2
  property.

- BF16 RRR has its own `grouped_kernel<RRR>` (same shared body); FP8 has
  `grouped_rrr_kernel` (line 2236) and a `grouped_tail_kernel` for full
  K reduction outside main coverage.

K-tail kernels (currently RMW on `g.c` after main kernel writes it):

- BF16: `grouped_ktail_kernel_lds`, `grouped_ktail_kernel_mfma`,
  `grouped_ktail_kernel_mfma32x32`, `_M2`, `_M2N2`, `_M4`,
  `grouped_ktail_kernel_lds_rrr` (for RRR backward dA path).
- FP8: same set + `_M2N4` (FP8-only).
- Each K-tail kernel does: cooperative load A_tail[16, K_REM=64] and
  B_tail[16, K_REM=64] into LDS → mfma_32x32 (or scalar fma) → multiply
  by `combined_scale` → **read C[row, col] from HBM, add, write back**.
- Profiled: M4 / M2N4 wins in BF16 / FP8 respectively; takes ~30-66%
  of total wall time on gpt_oss N=2880, K=2880 cases.

Dispatch (`dispatch_grouped` BF16, `dispatch_grouped_rcr` FP8):

```
g.fast_n = (g.n / BLOCK_SIZE) * BLOCK_SIZE;          # 256-aligned floor
g.fast_k = (g.k / K_TWO_TILE) * K_TWO_TILE;          # 128-aligned floor
g.bpc    = ceil_div(g.n, BLOCK_SIZE);                # ceil cover N (RCR only)
g.ki     = g.fast_k / K_STEP;                        # main K iters

if g.bpc > 0 && g.ki >= 2:
    launch main grouped kernel (covers [0, g.M_total) x [0, g.n) x [0, g.fast_k))
if g.fast_k < g.k:
    launch grouped_ktail_kernel_*  ← THIS is the launch we want to fuse away
```

## Metric reality (round 1 baseline)

```
grpBF16 geomean = 0.9943   progress=0.829  FAIL
grpFP8  geomean = 0.8207   progress=0.684  FAIL
score = 753 (target 1000)
```

Per-shape breakdown:

- DSV3 (K=2048/4096/7168, K%128==0 → K_REM=0): no K-tail launched. BF16 ratios
  1.11-1.16, FP8 ratios 0.93-1.04. K-tail fuse irrelevant on these.
- gpt_oss (K=2880, K%128=64 → K_REM=64): all 16 cases activate ktail.
  BF16 ratios 0.82-0.87, FP8 ratios 0.65-0.77. **This is where the fuse
  win lives.**

Estimated upper bound: removing the ktail launch saves the launch
fixed overhead (~5 µs) + the C HBM read (~16-24 µs on M*N=8M-23M cells).
On gpt_oss-Down-B4-M2048 wall ~80 µs, ktail ~30%, save ~24 µs → ratio
0.65 → ~0.93. Multiply by similar per-cell speedup on other gpt_oss
cases → projected FP8 geomean 0.95+, BF16 geomean 1.10+.

## Path A — LDS-staged in-kernel K-tail (recommended starting point)

Sketch (FP8 RCR shown; BF16 mirrors with tile constants):

```cpp
// === After Epilog 2 (cA/cB/cC/cD hold sum over K=[0, fast_k)) ===

if constexpr (...)  /* compile-time gate; runtime fast_k<g.k check */ {
    const int K_REM = g.k - g.fast_k;     // 64 for gpt_oss K=2880
    if (K_REM > 0) {
        // STEP 1: zero-init LDS slots that we're about to use.
        // As[tic][0..1] + Bs[tic][0..1] = 4 × 16 KB = 64 KB. Cooperative
        // 64-byte vec store per thread × 256 threads × 4 tiles = 64 KB
        // covered in 1 store per thread.
        zero_lds_tile(As[tic][0]);
        zero_lds_tile(As[tic][1]);
        zero_lds_tile(Bs[tic][0]);
        zero_lds_tile(Bs[tic][1]);
        __syncthreads();

        // STEP 2: cooperative load K=[fast_k, fast_k + K_REM) into the
        // K=[0, K_REM) bytes of the LDS tile. K=[K_REM, K_BLOCK) stays 0
        // from STEP 1, so a full-K rcr_mma sees real_K + zero_pad.
        //
        // PROBLEM: ST_v2 LDS layout is swizzled (st_16x128_v2_s); writing
        // raw fp8e4m3 at "K=k" position in the swizzle requires hitting
        // the right swizzled byte offset. Two paths:
        //   (a) Use the existing G::load with a partial K-bound by
        //       building a per-call SRD that has range_bytes capped to
        //       (M_total * g.k) — but we showed this clamps the buffer
        //       VOFFSET against M_total*K bytes, NOT against per-row K
        //       bytes; OOB K lanes wrap into the next row.
        //   (b) mirror grouped_ktail_kernel_lds's manual vec4-fp8
        //       cooperative load, but write into the SWIZZLED `As[tic][0]`
        //       buffer at the swizzle's `(row, K=k)` byte address.
        //       This requires inverting / reusing st_16x128_v2_s's
        //       coord-to-LDS-byte map.
        //
        // OR (c): manual coop-load to a SECOND, plain-row-major LDS
        // staging area, then a copy step that writes through subtile
        // store into As/Bs. Costs an extra LDS roundtrip but keeps the
        // swizzle math hidden inside the ST helper.

        // STEP 3: load_a / load_b feed the same a/b0/b1 register tiles
        // we used in main loop (lifetime ended after Epilog 2 mma).
        load_b(b0, Bs[tic][0], wn);
        load_a(a,  As[tic][0], wm);
        load_b(b1, Bs[tic][1], wn);
        asm volatile("s_waitcnt lgkmcnt(0)");
        rcr_mma(cA, a, b0);
        rcr_mma(cB, a, b1);

        load_a(a,  As[tic][1], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        rcr_mma(cC, a, b0);
        rcr_mma(cD, a, b1);
    }
}
// scale + store-C as before
```

**Register increment**: 0 (no new array hoisted; reuses cA-cD, a, b0, b1).
This is the win condition the run task body specifies.

## Hard fact discovered this round

`buffer_load_lds` SRD-bound clamping is range-bytes against the SOFF in
linear address space. For row-major K layout (g.a is `[M, K]` fp8 with
row_stride = K bytes), reading K=[fast_k, fast_k + K_BLOCK) when
`g.k = fast_k + K_REM` produces:

- K_in_tile in [0, K_REM): VOFFSET = row*K + fast_k + k → valid bytes
  in this row.
- K_in_tile in [K_REM, K_BLOCK): VOFFSET = row*K + (fast_k + K_BLOCK + k_in)
  → **falls into the next row's leading K bytes; SRD does NOT clamp
  to zero, returns the next-row-K=[0, K_BLOCK-K_REM) data instead.**

So path A cannot rely on "G::load + SRD clamp" to zero-pad the K tail.
The fuse logic MUST manually zero-init the LDS slots first, then
cooperative-load only K=[0, K_REM) bytes.

## Round 1 execution plan

Round 1 (this round):
- **Survey + notes** (this file).
- baseline metric = 753 confirmed (no improvement, no regression).
- Commit notes only; no kernel change.

Round 2 onwards:
- Build a `zero_lds_tile_st_v2` cooperative helper (256 threads × 64 B
  store = 16 KB, single store per thread).
- Build a `coop_load_partial_K_into_st_v2` helper that mirrors
  `grouped_ktail_kernel_lds`'s vec4 fp8 cooperative load but writes
  into the swizzled `ST_v2` LDS via `subtile_inplace<>` byte-store.
  (Path A, route a; we don't have ST_v2 byte-coord; use route c —
  plain LDS staging + copy through `store(...)` if needed.)
- Wire the helper into `grouped_rcr_kernel`'s epilog (FP8 first, since
  the FP8 gap is wider, then mirror to BF16).
- Drop the `grouped_ktail_kernel_*` launches from the host dispatcher
  for FP8 K_REM=64 once the in-kernel path covers it; keep the standalone
  kernels for K_REM=other or as fallback.
- Iterate on `s_waitcnt` placement and `__builtin_amdgcn_s_setprio`
  hints to overlap K-tail compute with the in-flight epilog 2 mma stalls.

## Spill-watch checklist (per Spill survival rules)

Before every kernel-mod build, run:

```
cd /workspace/code/HipKittens/analysis/fp8_gemm/mi350x
source ../../../env.src && make -j 2>&1 | grep -E '(spill|local mem|VGPR|hipcc)'
```

If `spill` count > 0: revert and try lower-pressure variant.

Variants in order of register-pressure (try lowest first):

1. **Path A LDS-staged**: register increment 0; only LDS bytes. ← Round 2 focus.
2. **Path B partial-K direct register load**: register increment 1
   a/b register tile (reuse single iteration of main_loop tile).
3. **Path C scalar epilog accumulate**: lane-level scalar fma direct
   into `cA/cB/cC/cD.tiles[h][w].data[idx].{x,y}`. Vec8 fp8e4m3_8 only;
   inner v loop runtime; avoid 16-cell parallel hoist (which spilled
   in pre-round-1 attempts per task body).

## Off-limits (do NOT revert to)

- Per-shape `num_xcds` / `group_m` rule tuning (saturated through round-67).
- Tightening `can_handle` to drop K_REM!=0 shapes (would clip ratio to
  0.01, sink score by ~100/case).
- Host-pad K (`torch.empty(K + 64)` etc.) — banned by run constraints.
- Multi-stream / per-group launch in dispatch — banned.
- `runtime json.load` / dict caches in dispatch — banned.

