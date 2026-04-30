# Round 2 — Path A K-tail fuse (FP8 RCR forward) — landed

**Result**: metric `score 753 → 793` (+40), `grp_FP8` geomean `0.821 →
0.909` (+9pp). All 8 gpt_oss FP8 forward shapes lifted by 9-20pp;
DSV3 (K_REM=0) shapes unchanged; reject 0/32; BF16 unchanged.

## Key insight that simplified the design

Round 1 notes claimed path A required explicit LDS zero-init followed
by partial-K cooperative load. **That was wrong** — there's a much
simpler trick:

- `llvm.amdgcn.raw.buffer.load.lds` clamps `SOFFSET + VOFFSET >= SRD.range_bytes`
  to 0 and **still issues the LDS write** with that zero value.
- So if we tag a lane's `voffset` with a SENTINEL value larger than any
  legitimate buffer offset (`0x7FFFFFFFu` works — well above the M*K
  byte total of any FP8 grouped tensor we hit), the lane's LDS slot
  gets written 0 with no extra cost. No explicit zero-init pass.

This is the basis of `prefill_swizzled_offsets_partial_K` — a
drop-in replacement for `kittens::prefill_swizzled_offsets` that
pre-tags each lane's per-pass voffset based on its post-XOR-swizzle
K-col position vs `K_REM_runtime`.

For ST_v2 (`st_fp8e4m3<HB=128, BK=128, st_16x128_v2_s>`):
- `bytes_per_thread = 16` → each lane covers a 16-cell K chunk per pass.
- The XOR swizzle (`((offset >> 7) & 7) << 4`) maps logical
  `(r, c)` to `swizzled_global_col = c XOR ((r%8) * 16)`.
- For `K_REM = 64`, lanes with `swizzled_global_col >= 64` (i.e.,
  `(L%8) XOR (r%8) >= 4`) get the sentinel; the rest get the
  real swizzled global byte offset.

K_REM is required to be `bytes_per_thread / sizeof(T) = 16`-aligned
in the fp8 path (no partial-validity 16-cell chunks supported). For
metric, K_REM is always 64 (gpt_oss K=2880) or 0 (DSV3) so this is
automatic.

## What landed in `grouped_rcr_kernel`

```cpp
template<int KI_HINT = 0, bool N_MASKED_STORE = false, bool FUSED_KTAIL = false>
__global__ void grouped_rcr_kernel(const grouped_layout_globals g) {
    // ... existing prelude ...
    uint32_t soA[mpt], soB[mpt];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

    // Round-2 path A additions:
    uint32_t soA_tail[mpt], soB_tail[mpt];
    if constexpr (FUSED_KTAIL) {
        const int K_REM = g.k - g.fast_k;
        prefill_swizzled_offsets_partial_K<_NUM_THREADS>(As[0][0], g.a, soA_tail, K_REM);
        prefill_swizzled_offsets_partial_K<_NUM_THREADS>(Bs[0][0], g.b, soB_tail, K_REM);
    }

    for (gt = pid; gt < total_tiles; gt += NUM_CUS) {
        // ... existing main loop + Epilog 1 + Epilog 2 ...

        // Round-2 path A: in-kernel K-tail accumulation into cA/cB/cC/cD.
        if constexpr (FUSED_KTAIL) {
            if (g.fast_k < g.k) {
                const int k_tail_tile = g.ki;
                rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 0), g.b, b_co(bc*2,   k_tail_tile), soB_tail);
                rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0],     g.a, a_co(br*2,   k_tail_tile), soA_tail);
                rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 1), g.b, b_co(bc*2+1, k_tail_tile), soB_tail);
                rcr_8w_load_hoist<_NUM_THREADS>(As[tic][1],     g.a, a_co(br*2+1, k_tail_tile), soA_tail);
                asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();

                load_b(b0, b_tile(tic, 0), wn);
                load_a(a, As[tic][0], wm);
                load_b(b1, b_tile(tic, 1), wn);
                __builtin_amdgcn_s_barrier();
                asm volatile("s_waitcnt lgkmcnt(0)");
                rcr_mma(cA, a, b0); rcr_mma(cB, a, b1);
                __builtin_amdgcn_s_barrier();

                load_a(a, As[tic][1], wm);
                __builtin_amdgcn_s_barrier();
                asm volatile("s_waitcnt lgkmcnt(0)");
                rcr_mma(cC, a, b0); rcr_mma(cD, a, b1);
                __builtin_amdgcn_s_barrier();
            }
        }

        // ... scale + store as before ...
    }
}
```

Plus instantiations:
```cpp
template __global__ void grouped_rcr_kernel<0, false, false>(...);  // legacy
template __global__ void grouped_rcr_kernel<0, true , false>(...);  // legacy
template __global__ void grouped_rcr_kernel<0, false, true >(...);  // fused
template __global__ void grouped_rcr_kernel<0, true , true >(...);  // fused
```

## Dispatcher gate (`dispatch_grouped_rcr`)

```cpp
const bool fuse_ktail_eligible =
    (g.bpc > 0) && (g.ki > 0) &&
    (g.k - g.fast_k == 64) &&
    (g.m_per_group >= TAIL_BLOCK_M) &&
    ((g.m_per_group % TAIL_BLOCK_M) == 0);

// Launch with FUSED_KTAIL=true when eligible, FUSED_KTAIL=false otherwise.
// When FUSED_KTAIL=true is selected, skip the standalone grouped_ktail_kernel_*
// launches (gated by `if (!fuse_ktail_eligible && ...)`).
```

The condition `K_REM == 64` is round-2-only for now. Round-3+ extend
to all 16-aligned K_REM ∈ {16, 32, 48, 80, 96, 112}.

## Spill diff

```
                            VGPR  spill   occ
<0, false, false>  legacy   256     91     2
<0, true , false>  legacy   256     83     2
<0, false, true >  fused    256     95     2   (+4)
<0, true , true >  fused    256     99     2   (+16)
```

Fuse adds +4 to +16 VGPR spills (the soA_tail/soB_tail arrays
plus the K-tail epilog's instruction sequence). Occupancy unchanged
at 2 waves/SIMD. The added register traffic is hidden by the
launch + RMW HBM-read savings (per-shape probe: gpt_oss-Down-B4-M2048
fwd 786.9 → 947.1 TFLOPS, +20%, way more than spill cost would
take back).

## Per-shape result (gpt_oss FP8 forward)

```
                          before  after  delta
B4-M2048-GateUP            0.675  0.803  +0.128
B4-M2048-Down              0.769  0.860  +0.091
B4-M4096-GateUP            0.649  0.817  +0.168
B4-M4096-Down              0.673  0.801  +0.128
B32-M2048-GateUP           0.680  0.862  +0.182
B32-M2048-Down             0.705  0.868  +0.163
B32-M4096-GateUP           0.646  0.842  +0.196
B32-M4096-Down             0.682  0.862  +0.180

geomean (FP8 16 shapes)    0.821  0.909  +0.088
```

DSV3 8 shapes (K_REM=0) unchanged; BF16 16 shapes unchanged.

## Limitations / next rounds

- **BF16 RCR fuse** (round 3): same trick mirrors to BF16. Expected
  uplift: gpt_oss BF16 geomean 0.85 → 1.00+. BF16's `device_gemm_tile_body`
  is shared with `grouped_kernel<L, KI_HINT>` so the fuse needs to be
  added inside `grouped_kernel` not inside the helper, or fork the
  helper. Care needed.
- **FP8 RRR fuse** (round 4 / dA backward): same pattern but K-tile
  layout differs (B is K x N instead of N x K), so the partial-K
  prefill needs an axis-flipped variant.
- **FP8 var-K dB fuse** (round 5 / dB backward): tile shapes differ
  again (CRR layout). Lower priority since metric only measures fwd.
- **Removing the standalone `grouped_ktail_kernel_*` definitions**:
  not done this round — they're still compiled but no longer
  launched on the K_REM=64 fuse path. Round 6+ once BF16 + FP8
  RRR fuses land we can delete the standalone kernels entirely.
- **Spill reduction** (later rounds): if we hoist soA_tail/soB_tail
  computation outside the kernel (precompute on host and pass as
  pointer to constant memory), the +10 VGPR/lane footprint goes
  away. Risk: HBM lookup latency in the kernel-entry hot path.
  Probably not worth it given the spill cost is already hidden.

