# R70 RETRY Opt-A: B3b raw-row mapping — VERDICT = SUCCESS

**Date**: 2026-04-20
**Agent**: R70 RETRY Optimizer-A
**Worktree**: `.claude/worktrees/agent-a8d66e1e`  branch `worktree-agent-a8d66e1e`
**Source**: `analysis/fp8_gemm/mi350x/iso_b3b_mapping/iso_b3b.cpp`

## TL;DR

**HYP=1 (identity row/col mapping) is correct.** The LDS-write swizzle and the
LDS-read swizzle exactly cancel each other; for a lane in the production
`fp4_load_st_to_rt` + `fp4_extract_tile` path, the bytes it sees are the
identity-mapped bytes from the lane's MFMA-consumption (row, col) within the
warp's RBM × BK sub-slice of g.a (raw row-major). No swizzle XOR needs to be
applied during voff computation for `buffer_load_dwordx4` (no lds).

**All 16 test cases pass — 4096/4096 lane-comparisons match bytewise.**

```
=== iso_b3b_mapping  HYP=1 (compile-time) ===
    M=512 N=256 K=1024  K_BYTES=512  HB=128 BK=128  WARPS=4  THREADS=256
    Test grid: br_idx ∈ [0,1], half_idx ∈ [0,1], bt ∈ [0..3]   (16 cases)
  case br=0 half=0 bt=0  : OK    pass=256 fail=0
  case br=0 half=0 bt=1  : OK    pass=256 fail=0
  ... (all 16 cases pass) ...
  case br=1 half=1 bt=3  : OK    pass=256 fail=0

  Total cases tested      : 16
  Cases passed all 256 lanes: 16
  Cases with at least 1 fail: 0
  Total lanes passed      : 4096 / 4096

=== VERDICT (HYP=1): ALL_CASES_PASS ===
```

## VERIFIED `compute_a_global_load_voffs`

Drop-in for the `kpair_64mfma_step34_pf_interleaved_globalA` helper that R70 plan
asks for. Mirrors the 8-voff layout of the existing `compute_b_global_load_voffs`,
but for raw row-major A (no preshuffle). Per-K-iter advance is delivered via
the `k_soffset` parameter to `buffer_load_dwordx4` (matching the GLOBAL_B
convention).

```cpp
// Per-call invariant inputs:
//   - wm:           warp's M index (0..WARPS_M-1)
//   - half_idx:     0 = A0 (rows br*BLK + [0..HB-1]),
//                   1 = A1 (rows br*BLK + [HB..BLK-1])
//   - br_idx:       block-row index (units of BLK rows)
//   - K_bytes_full: K_DIM/2 (raw row stride in g.a)
//
// Outputs voff[8]:
//   voff[0..3] = lane's 16-byte chunks for slots 0..3 at k=0
//   voff[4..7] = lane's 16-byte chunks for slots 0..3 at k=1
// where "slot" = base-tile index (0..3 = vertical stack of 4 base tiles
// covering RBM=64 rows of the warp's sub-slice).
//
// Caller: pass k_soffset = bt * BK to buffer_load_dwordx4 (or pre-advance SRD).
__device__ __forceinline__ void compute_a_global_load_voffs(
    uint32_t voff[8],
    int wm,
    int half_idx,
    int br_idx,
    int K_bytes_full)
{
    const int laneid = kittens::laneid();
    const int row_offset = laneid % 16;
    const int col_offset = 16 * (laneid / 16);  // 0,16,32,48
    // Warp's base row in global g.a:
    //   = br_idx * BLK + half_idx * HB + wm * RBM
    const int warp_base_row = br_idx * BLK + half_idx * HB + wm * RBM;
    #pragma unroll
    for (int ii = 0; ii < 4; ++ii) {
        const int row = warp_base_row + ii * 16 + row_offset;
        const int col_lo = col_offset;          // k=0
        const int col_hi = col_offset + 64;     // k=1 (stride_group)
        voff[ii    ] = (uint32_t)(row * K_bytes_full + col_lo);
        voff[ii + 4] = (uint32_t)(row * K_bytes_full + col_hi);
    }
}
```

### Constants (from kernel_mxfp4_gluon_cpp.cpp)
- `BLK = 256`, `BK = 128`, `HB = BLK/2 = 128`, `RBM = HB/WARPS_M = 64`
- `WARPS_M = WARPS_N = 2`, `NUM_WARPS = 4`
- `K_BYTES = K_DIM / 2`

### Lane→bytes derivation
For lane L in warp wm, the production reference path resolves to:
- `row_offset = L % 16`     (which row inside a base tile)
- `col_offset = 16 * (L/16)` ∈ {0, 16, 32, 48}  (column-base inside base tile)
- 4 base tiles stacked vertically: `ii ∈ [0..3]`, base-row = `ii*16 + row_offset`
- For each (ii, k ∈ {0,1}): col = `col_offset + k*64`
- Global byte address: `(warp_base_row + ii*16 + row_offset) * K_bytes + col`

That formula is what `compute_a_global_load_voffs` emits; the harness verified
it produces byte-identical 16-byte chunks vs the reference path for **all 256
lanes × 4 slots × 2 k-strides** at every test case.

## Test methodology

1. Allocate g.a with shape (1, 1, M=512, K_BYTES=512) covering 2 block-row
   pairs and 4 K-iters worth.
2. Fill bytes with a per-row × per-col-chunk fingerprint:
   `bytes[r][c_chunk*16 + k*4 + i] = ((r&0xFF, r>>8, c_chunk&0xFF, c_chunk>>8)[i])`
   for i ∈ [0..3], replicated across all 16 bytes of each chunk.
3. For each test case `(br_idx, half_idx, bt)`:
   a. Reference: load A_db tile via `G::load(A_db, g.a, coord(0,0,br*2+half, bt))`,
      then `fp4_load_st_to_rt` + `fp4_extract_tile` → `fp4_intx8_t[4]` per lane.
   b. Candidate: call `compute_a_global_load_voffs(voff, wm, half, br, K_BYTES)`,
      then 8 `buffer_load_dwordx4` (no lds) at `(srd_a, voff[i], bt*BK)` →
      assemble `fp4_intx8_t[4]` via the same lo4/hi4 extract pattern.
   c. Bytewise-compare reference vs candidate per (lane, slot, byte).
4. Iterate over `br ∈ [0,1] × half ∈ [0,1] × bt ∈ [0..3]` = 16 cases × 256
   lanes × 4 slots × 32 bytes = **524288 bytes verified per run** — all match.

## Reproduction

```bash
cd /shared_nfs/kyle/test/HipKittens/.claude/worktrees/agent-a8d66e1e/analysis/fp8_gemm/mi350x

# Build
THUNDERKITTENS_ROOT=/shared_nfs/kyle/test/HipKittens/.claude/worktrees/agent-a8d66e1e \
  ROCM_PATH=/opt/rocm \
  /opt/rocm/bin/hipcc iso_b3b_mapping/iso_b3b.cpp \
    -DKITTENS_CDNA4 --offload-arch=gfx950 \
    -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
    -I/opt/rocm/include/rocrand -std=c++20 \
    -I/shared_nfs/kyle/test/HipKittens/.claude/worktrees/agent-a8d66e1e/include \
    -I/opt/rocm/include/hip \
    -I/shared_nfs/kyle/test/HipKittens/.claude/worktrees/agent-a8d66e1e/prototype \
    -w -o iso_b3b_mapping/iso_b3b

# Run on idle GPU (check rocm-smi first; we used GPU 4)
HIP_VISIBLE_DEVICES=4 ./iso_b3b_mapping/iso_b3b
# Exit code 0 = ALL_CASES_PASS
```

## Bisect framework details

The harness uses a compile-time `HYP` macro (default 1) selecting which
candidate hypothesis is compiled in. During iteration:
- HYP=1 (identity row/col): **VERIFIED correct**.
- HYP=2 (col-swap k=0/k=1): tested early on smaller M case during dev,
  produced systematic 64-byte col-flip mismatch — rejected.
- HYP=3 (explicit XOR-swizzle in voff): tested early — produced
  partial mismatches in the bytes the swizzle XOR's high bit affects;
  confirmed unnecessary because writer+reader swizzles cancel.

Initial single-shape run with M=256/K=128 surfaced a `warp_base_row` bug for
`wm=1` warps (fix: include `half_idx * HB + wm * RBM` instead of just `wm * RBM`).
After fix, the canonical case passed → extended to full 16-case grid → all passed.

## Disposition for R70/R71 integration (Opt-B/Opt-C)

This `compute_a_global_load_voffs` is the **structural hole identified in
`project_mxfp4_R70_globalA_session_findings.md`**. The blocker described
there ("no existing function in the codebase performs this mapping") is
now resolved.

Next-session integration steps (Opt-B/Opt-C of the R70 retry plan):
1. Copy the function into `kernel_mxfp4_gluon_cpp.cpp` near L287
   (alongside `compute_b_global_load_voffs`).
2. Author `kpair_64mfma_step34_pf_interleaved_globalA` by mechanically
   replacing the 8 A `ds_read_b128` operands+constraints in
   `kpair_64mfma_step34_pf_interleaved` with `buffer_load_dwordx4 (no lds)`
   operands+constraints using the verified voffs.
3. Wire it via `#if GLOBAL_A` macro at the call sites.
4. Preflight: build with `GLOBAL_A=0` → byte-identical .s vs current main.
5. Bench K=128256 first in isolation (R66 cadence) → full sweep.

## Files in this directory
- `iso_b3b.cpp` — the harness source (~520 LOC including comments)
- `iso_b3b`    — compiled standalone executable
- `VERDICT.md` — this file

## Reference notes
- `project_mxfp4_R70_globalA_session_findings.md` — describes the original
  R70 attempt that exited without committing because this mapping was unverified
- `project_mxfp4_R69_axis_a_opt2_scoped.md` — the 17-touchpoint scoping
  that calls out B3b mapping as the recommended approach
- `project_mxfp4_R66_axis_a_landed.md` — the LANDED axis-A pattern that
  GLOBAL_A mirrors (single-asm-block discipline)
