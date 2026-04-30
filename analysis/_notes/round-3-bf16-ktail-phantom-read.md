# Round 3 — BF16 RCR K-tail fuse (path A) — phantom LDS read bug

## TL;DR

Mirroring the round-2 FP8 path-A fuse to BF16 RCR's `grouped_kernel`
hits a numerical correctness regression that we did NOT root-cause this
round. The fuse is **infrastructure only** in this commit:
`FUSED_KTAIL=true` instantiations exist + the dispatcher has the
eligibility gate but `fuse_ktail_eligible` is hard-set to `false`, so
the standalone `grouped_ktail_kernel_lds` launch path remains active
exactly as round-2 baseline. Metric back to round-2 number (792).

Round 4 plan: switch to **path B (direct HBM-to-register K-tail load)**.
Path B side-steps the LDS round-trip entirely so the phantom-read
diagnostic does not apply.

## What the fuse does (when re-enabled)

```cpp
template<Layout L, int KI_HINT = 0, bool FUSED_KTAIL = false>
__global__ void grouped_kernel(const grouped_layout_globals g) {
    // ... shared_allocator: As[2][2] + Bs[2][2] of bf16 64x64 tiles ...
    device_gemm_tile_body<...>(g, ...);   // main loop K=[0, fast_k)

    if constexpr (FUSED_KTAIL) {
        if constexpr (L == Layout::RCR) {
            __syncthreads();

            // Cooperative G::load — write K=[fast_k, fast_k + K_STEP)
            // into Bs[0][0/1] / As[0][0/1].
            G::load(Bs[0][0], g.b, coord<ST_B>{0, group_idx, col*2,   g.ki});
            G::load(As[0][0], g.a, coord<ST_A>{0, 0, m_subtile_A + row*2,   g.ki});
            G::load(Bs[0][1], g.b, coord<ST_B>{0, group_idx, col*2+1, g.ki});
            G::load(As[0][1], g.a, coord<ST_A>{0, 0, m_subtile_A + row*2+1, g.ki});

            asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
            __syncthreads();

            A_reg_t A_tile;
            B_reg_t B_tile_0, B_tile_1;
            load(B_tile_0, subtile_inplace<HALF_REG_BLOCK_N, K_STEP>(Bs[0][0], {warp_col, 0}));
            load(A_tile,   subtile_inplace<HALF_REG_BLOCK_M, K_STEP>(As[0][0], {warp_row, 0}));
            load(B_tile_1, subtile_inplace<HALF_REG_BLOCK_N, K_STEP>(Bs[0][1], {warp_col, 0}));
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
            mma_ABt(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);

            load(A_tile,   subtile_inplace<HALF_REG_BLOCK_M, K_STEP>(As[0][1], {warp_row, 0}));
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            mma_ABt(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
            mma_ABt(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        }
    }
    // ... store C_accum to g.c ...
}
```

Eligibility:
```cpp
const bool fuse_ktail_eligible =
    false &&  // disabled — see phantom read bug below
    (L == Layout::RCR) && (g.bpc > 0) && (g.ki >= 2) &&
    (K_rem_for_fuse == K_STEP) &&
    (g.m_per_group >= TAIL_BLOCK_M) &&
    ((g.m_per_group % TAIL_BLOCK_M) == 0);
```

## What we observe (numerical probe)

Probe shape: `M_total=8192, N=2880, K=2880, G=4` (gpt_oss-GateUP-B4-M2048).
Reference: torch.matmul fp32. Compare `out_hk` to `out_ref`.

```
                                 SNR(dB)   max_abs_err
no fuse (legacy K-tail kernel)   44.5       6.3e-3
path-A fuse enabled              18.6       3.7e-1
```

A 26 dB drop indicates the K-tail's contribution is being applied to
the wrong subset of warps' output cells. C_accum is structured
`[2][2]` per warp where `[wm][wn]` covers an `RBM x RBN` tile; we'd
expect a uniform contribution if the fuse were correct.

## Targeted printf diagnostics (dropped this round)

We instrumented the kernel with `printf` from select lanes/warps to
isolate the corruption point. Findings (probe shape gpt_oss
GateUP-B4-M2048, K=2880 → fast_k=2816, ki=44, K_REM=64=K_STEP):

1. **Cooperative G::load writes correct data to LDS.** Lane 0 of warp 0
   reads `Bs[0][0].data[0..15]` (raw bf16 bytes) and observes the
   expected K=[2816, 2816+64) values for ALL warps (printed by lane 0
   of every warp post-`s_waitcnt vmcnt(0) lgkmcnt(0)` +
   `__builtin_amdgcn_s_barrier()`).

2. **Register tile after `load(reg, st_subtile)` differs by warp.**
   For B_tile_0 = `load(Bs[0][0], {warp_col, 0})`:
   - `warp_row=0, warp_col ∈ {0, 2}` → register has the correct
     K=[2816, 2880) data ✓
   - `warp_row=0, warp_col ∈ {1, 3}` → register has STALE data
     (K=[2688, 2752), the main loop's penultimate Bs[0][0] write) ✗
   - `warp_row=1, all warp_col` → register has the correct K=[2816,
     2880) data ✓

   Same pattern for B_tile_1 (Bs[0][1]). Rules out a write-side bug
   (LDS observably fresh from every warp) and rules out a swizzle bug
   (warp_row=1 reads the same LDS region successfully). Even/odd
   warp_col split rules out simple bank conflict.

3. **Synchronization combinations that did NOT fix it**:
   - Barriers tried: `__builtin_amdgcn_s_barrier()` only,
     `__syncthreads()` only, both, around individual loads.
   - Wait counts tried: `vmcnt(0)`, `vmcnt(0) lgkmcnt(0)`, with/without
     `:::"memory"` clobber.
   - Compiler scheduling: `__builtin_amdgcn_sched_barrier(0)` before
     and after every load.
   - MMA wrapper: replaced `mma_ABt` with explicit
     `mma_ABt_base` unrolled loops — no change.
   - Accumulator: tried `D = C` aliasing (matches FP8 fuse) AND
     `D = zero(); D += A*B^T; C += D` temporary — same SNR.
   - All combinations land at SNR ≈ 18.6 dB (modulo NaN when
     accumulator zero-init is wrong).

## Hypothesis for round 4

The phantom read pattern (warp_row=0 wc∈{1,3} only) is consistent with
`load(reg, st_subtile<>(Bs[0][0], {warp_col, 0}))` computing its
LDS source addresses based on a stale value of an `__shared__`
allocator state OR with `subtile_inplace` capturing a stale tile-base
pointer. Since the FP8 path uses `rcr_8w_load_hoist` which is a
custom inline-asm cooperative load+register-write that bypasses the
generic `load(reg, st)` helper, it doesn't hit this code path —
explains why FP8 fuse worked first try and BF16 doesn't.

Path B (direct HBM-to-register, no LDS round-trip) avoids
`subtile_inplace` + `load(reg, st)` entirely. Round 4 will use
`buffer_load_b128` + per-lane K_REM mask to write directly into
`B_reg_t`/`A_reg_t` register tiles using the lane→cell mapping derived
in round 3 (rt_fl<RBM=64, RBN=32, col_l, rt_16x16_s>: lane in
warp_col=c covers cells at `(local_col, k_idx)` where
`local_col = lane.lo32 % 16`, `k_idx = (lane.lo32 / 16) * 4 + lane.hi32`).

## Fallback plan

If path B also fails, we revisit:
1. Re-derive `subtile_inplace` view of `Bs[0][0]` with
   `warp_col ∈ {1, 3}` to confirm the LDS-byte address it computes
   matches what cooperative G::load wrote.
2. Check if `device_gemm_tile_body`'s register pressure forces a
   re-spill that races with the K-tail's `subtile_inplace` LDS read.

## Files touched (this round, infrastructure only)

`analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`:
- `grouped_kernel<L, KI_HINT, FUSED_KTAIL>` template signature.
- Empty FUSED_KTAIL body (placeholder for path B).
- `<RCR, 0, true>` instantiation.
- `dispatch_grouped` eligibility gate (forced false).

No primus-turbo changes; legacy K-tail launch path still active.
