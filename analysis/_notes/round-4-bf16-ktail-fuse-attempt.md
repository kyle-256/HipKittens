# Round 4 — BF16 RCR K-tail fuse (path A) — multiple attempts, all 18.57 dB

## TL;DR

Three new attempts at the BF16 RCR K-tail fuse this round, all converge on
the same broken SNR (~18.5 dB vs 44.5 dB legacy):

| variant | LDS stage | G::load form | fuse-block scope | SNR (dB) |
|---|---|---|---|---|
| round-3 attempt #1 (prior round) | stage 0 (Bs[0][.]) | 4-arg | `grouped_kernel` outer | 18.57 |
| round-4 #1 | stage 1 (Bs[1][.]) | 8-arg precomputed offsets | `grouped_kernel` outer | 18.57 |
| round-4 #2 | stage 1 (Bs[1][.]) | 8-arg precomputed offsets | inside `device_gemm_tile_body` | 18.57 |
| round-4 #3 | stage 0 (Bs[0][.]) | 8-arg precomputed offsets | inside `device_gemm_tile_body` | 18.60 |
| round-4 zero-init diagnostic | stage 1, pre-zeroed | 8-arg precomputed offsets | inside `device_gemm_tile_body` | 18.11 (≈ no-correction baseline) |

Stage choice, G::load form, and fuse-block scope are NOT the bug. The
phantom-read is independent of all three.

The diagnostic experiment (cooperative pre-zero of stage-1 LDS slots
before G::load) collapses SNR to ≈ no-correction-baseline (16.53 dB
when the K-tail body is empty). This means: **after pre-zero, the
K-tail's ``load(reg, st_subtile)`` reads zeros from LDS — i.e., the
G::load did NOT actually write K-tile-44 data into the bytes that
``subtile_inplace + load(reg, st)`` reads from**. Without pre-zero, the
load reads stale K-tile-43 data left in stage-1 by epilog 2's last
read, so the K-tail's mma_ABt accumulates K-tile 43's contribution
on top of C_accum's [0, fast_k) reduction — wrong but not zero,
producing the 18.5 dB pattern.

This is a NEW finding vs the round-3 doc, which claimed lane 0 of
warp 0 could observe correct K-tail bytes via raw bf16* reads. That
diagnostic only verified row 0 of LDS; the K-tail load for warps 1, 3
reads rows [32, 64) and [96, 128) of LDS, which the round-3 lane-0
probe did not cover.

## What we tried

### Attempt #1 — stage 1 + 8-arg G::load + outer scope (`grouped_kernel`)

Replace the empty FUSED_KTAIL block in `grouped_kernel` with:

```cpp
if constexpr (FUSED_KTAIL) {
    if constexpr (L == Layout::RCR) {
        if (g.fast_k < g.k) {
            const int k_tail_tile = g.ki;
            // 8-arg G::load to stage 1 (matches FP8 path A).
            G::load(Bs[1][0], g.b, ..., swizzled_offsets_B, b_srsrc_curr, b_base, b_lds_10);
            G::load(As[1][0], g.a, ..., swizzled_offsets_A, a_srsrc_base, a_base, a_lds_10);
            G::load(Bs[1][1], g.b, ..., swizzled_offsets_B, b_srsrc_curr, b_base, b_lds_11);
            G::load(As[1][1], g.a, ..., swizzled_offsets_A, a_srsrc_base, a_base, a_lds_11);
            asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
            __builtin_amdgcn_s_barrier();
            // subtile_inplace + load(reg, st) + mma_ABt × 4
        }
    }
}
```

Result: SNR 18.57 dB (same as round-3 #1).

### Attempt #2 — move fuse INSIDE `device_gemm_tile_body`

Hypothesis (round-3): "shared_base_offset capture mismatch when
`subtile_inplace + load(reg, st_subtile)` straddles the inline
boundary between `__forceinline__` `device_gemm_tile_body` and
`grouped_kernel`. Sharing the function scope with the main loop's
lambdas + coord helpers should keep the inlining state consistent."

Refactor: add `bool FUSED_KTAIL = false` template parameter to
`device_gemm_tile_body`, place the K-tail block right after epilog 2
inside that function, reuse the `load_a_subtile` / `load_b_subtile` /
`DO_MMA` helpers verbatim. `grouped_kernel<L, KI_HINT, true>`
instantiates `device_gemm_tile_body<L, KI_HINT, ..., true>` and the
fuse block lives in the same compilation unit as the working main
loop.

Result: SNR 18.57 dB. Hypothesis was wrong.

### Attempt #3 — stage 0 + 8-arg G::load (combination not previously tried)

Round-3 used stage 0 + 4-arg G::load. Round-4 #1/#2 used stage 1 +
8-arg G::load. The stage-0 + 8-arg combination had not been tried.

Result: SNR 18.60 dB. Same bug.

### Diagnostic — explicit pre-zero of stage-1 LDS

Cooperatively zero the stage-1 LDS slots before the G::load:

```cpp
const int tid = threadIdx.x;
bf16x4 z{};
bf16* As10 = (bf16*)&As[1][0].data[0];
bf16* As11 = (bf16*)&As[1][1].data[0];
bf16* Bs10 = (bf16*)&Bs[1][0].data[0];
bf16* Bs11 = (bf16*)&Bs[1][1].data[0];
#pragma unroll
for (int i = 0; i < 4; ++i) {
    const int idx = tid * 4 + i;
    *reinterpret_cast<bf16x4*>(&As10[idx * 4]) = z;
    *reinterpret_cast<bf16x4*>(&As11[idx * 4]) = z;
    *reinterpret_cast<bf16x4*>(&Bs10[idx * 4]) = z;
    *reinterpret_cast<bf16x4*>(&Bs11[idx * 4]) = z;
}
__builtin_amdgcn_s_barrier();
asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
// ... then G::load + load(reg, st) + mma ...
```

Result: SNR 18.11 dB (vs 16.53 empty-body baseline, vs 18.57
non-zeroed full body, vs 44.27 legacy K-tail kernel).

The 18.11 dB ≈ 16.53 dB equivalence (1.5 dB difference is small,
consistent with a tiny fraction of K-tail bytes being correctly
written) is the SMOKING GUN: the K-tail's load(reg, st_subtile)
reads MOSTLY zeros from a pre-zeroed LDS, even though the G::load
ostensibly wrote K-tile-44 to those LDS bytes.

## Implication

`G::load` (with either 4-arg or 8-arg form) does NOT reliably write
K-tile-44 to stage-1 (or stage-0) LDS slots in the post-epilog-2
state of the persistent grouped kernel. The bytes the subsequent
`subtile_inplace + load(reg, st)` reads from are mostly NOT updated.
The data IS being fetched from HBM (s_waitcnt vmcnt(0) drains), but
the LDS write side of `buffer_load_lds` somehow doesn't land at the
right addresses.

Hypothesis (round-5 to verify): the m0-broadcast register that
buffer_load_lds uses to compute the LDS write address may be
corrupted in the post-epilog-2 state, causing buffer_load_lds writes
to land at unexpected LDS offsets. This would explain why nearly all
LDS bytes the read targets are unchanged (the writes went somewhere
else). FP8 path A works because `rcr_8w_load_hoist` uses inline asm
to explicitly set m0 via `s_mov m0, <sgpr>` before each
`buffer_load_dwordx4 ... offen lds`, bypassing whatever compiler-CSE
issue may exist with `__builtin_amdgcn_raw_buffer_load_lds`.

## Round 5 plan: skip path A, do path B

Path B = direct HBM-to-register K-tail load via per-lane
`buffer_load_dwordx4`, no LDS intermediate. Each lane reads bf16x8
of A and bf16x8 of B at the lane-specific (row, K-offset) cells
required by mfma_f32_16x16x32_bf16's operand layout, populates
A_tile/B_tile registers, and issues mma_ABt into C_accum.

Lane→cell mapping (CDNA4 mfma_f32_16x16x32_bf16):

* A operand bf16x8 per lane → A[lane % 16, (lane / 16) * 8 + (0..7)].
* B operand bf16x8 per lane → B[lane % 16, (lane / 16) * 8 + (0..7)].

For the K-tail (K_REM=64, fast_k=2816):

* A_tile = `rt_bf<HALF_REG_BLOCK_M=64, K_STEP=64, row_l, rt_16x32_s>`,
  4 row-base × 2 K-base of 16x32 tiles. For warp_row=r, h ∈ [0, 4),
  w ∈ [0, 2):
    * Row range = m_start_g + (row*2 + ?) * HALF_BLOCK_SIZE + r*64 + h*16 + (lane % 16)
    * K-offset = fast_k + w * 32 + (lane / 16) * 8.

The complication: ``device_gemm_tile_body`` issues mma_ABt for
C_accum[0][.] using A_tile from `As[X][0]` (the first 128-row M
slab) and C_accum[1][.] using A_tile from `As[X][1]` (the second
128-row M slab). Path B needs to load the K-tail twice — once for
each M slab — into A_tile, alternating with B_tile loads (B_tile
covers all 4 N strips via warp_col).

Implementation skeleton:

```cpp
if constexpr (FUSED_KTAIL && L == Layout::RCR) {
    // Path B: direct HBM-to-register K-tail load.
    const int k_tail = fast_k;  // K-cell offset into A/B rows
    // For each (m_slab=0..1, n_strip=0..1, K-base w=0..1):
    //   - Each lane reads bf16x8 from A at row depending on m_slab/h/lane,
    //     K-cell at k_tail + w * 32 + (lane / 16) * 8.
    //   - Each lane reads bf16x8 from B at col depending on n_strip/lane,
    //     K-cell same as above.
    //   - Stuff into A_tile.tiles[h][w] / B_tile.tiles[*][w].
    // Then mma_ABt(C_accum[m_slab][n_strip], A_tile, B_tile, C_accum).
}
```

This sidesteps the LDS-write bug of path A entirely. No
`subtile_inplace`, no `load(reg, st)`, no `G::load`. Only direct
`buffer_load_dwordx4` per lane. Register pressure: ~1 A_tile + 1
B_tile worth of VGPRs (already live during epilog 2, can be
reused).

## Files touched (round 4, infrastructure + diagnostics)

* `analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`:
  * Added `bool FUSED_KTAIL = false` template param to
    `device_gemm_tile_body`.
  * Added FUSED_KTAIL block at end of `device_gemm_tile_body` (empty
    body for now, comment with round-5 path-B plan).
  * Removed the round-3 outer-scope FUSED_KTAIL block from
    `grouped_kernel` (replaced with a comment pointing into the helper).
  * Forward `FUSED_KTAIL` template param to
    `device_gemm_tile_body<..., FUSED_KTAIL>` from `grouped_kernel`.
  * Dispatcher: kept `fuse_ktail_eligible` gated to false (path A
    failed; path B not yet implemented).

No primus-turbo changes; legacy K-tail launch path still active.
Metric stays at 793 (round-2/3 number).
