# Round 3 — FP8 RCR K-tail fuse (path A) saturates at SNR 20.7 dB

Mirrors `round-3-bf16-ktail-phantom-read.md` for the FP8 side.

## TL;DR

The round-2 commit `4f6a2dee perf(grouped-fp8): fuse K-tail epilog into
grouped_rcr_kernel (path A)` shipped a path-A FP8 K-tail fuse based on
the (incorrect) belief that `llvm.amdgcn.raw.buffer.load.lds` zero-fills
LDS on OOB voffset, and was reported as correctness-preserving in its
commit log + `round-2-ktail-fuse-result.md`.

**That report was wrong.** The fuse is structurally numerically broken
in exactly the same way BF16 path A was (`round-3-bf16-ktail-phantom-read.md`):

* `buffer_load_lds` with `SOFFSET + VOFFSET > range_bytes` is a NO-OP,
  not a zero-fill. OOB lanes' LDS slots therefore inherit the previous
  main-loop K-tile data (K=[fast_k - K_BLOCK, fast_k)), and rcr_mma
  accumulates those stale cells with weight 1 instead of the required
  weight 0 zero-pad. SNR ~16.8 dB.
* On top of that, `load(reg, st_subtile)` in the post-epilog-2 SGPR
  state returns stale data on warp subset `warp_row=0 ∧ warp_col∈{1,3}`
  (the BF16 round-3 phantom-read pattern). `rcr_8w_load_hoist` (the
  cooperative HBM→LDS load FP8 uses) sidesteps the BF16 G::load helper
  but the read-back via `subtile_inplace + load(reg, st)` is the same
  pattern and exhibits the same phantom-read failure.

The round-2 metric uplift (`grp_FP8 0.821 → 0.909`) was real for
performance (the standalone K-tail kernel really did stop running) but
the correctness uplift was an artefact of a looser SNR gate at that
time. With the current 25 dB FP8 SNR threshold (see
`scripts/_metric_grouped_only.py:_FP8_SNR_THRESHOLD_DB`), all 8 gpt_oss
K=2880 cases register `*FAIL[fwd-snr<XX]` and clip to ratio 0.

## What this round (round 3) shipped — partial defence on path A

```diff
+ // Cooperative zero of As[tic][0..1] + Bs[tic][0..1] before partial-K
+ // load. Covers full sizeof(ST_rcr) = 17408 bytes/tile (rows*cols +
+ // 8 × subtile_padding=128). Forces OOB lanes to read 0 in LDS even
+ // when buffer_load_lds is a no-op.
+ for (int i = tid; i < (int)sizeof(ST_rcr)/4; i += _NUM_THREADS) {
+     reinterpret_cast<int*>(&As[tic][0])[i] = 0;
+     reinterpret_cast<int*>(&As[tic][1])[i] = 0;
+     reinterpret_cast<int*>(&Bs[tic][0])[i] = 0;
+     reinterpret_cast<int*>(&Bs[tic][1])[i] = 0;
+ }
+ asm volatile("s_waitcnt lgkmcnt(0)");  // drain ds_writes
+ __builtin_amdgcn_s_barrier();
```

```diff
+ // Restore the 3 inner barriers commit 2035f1a1 pruned. With the
+ // cooperative zero in place those barriers are still required for
+ // the second load_a → rcr_mma to see the first rcr_mma's MFMA
+ // outputs settled.
  load_b(b0, ...); load_a(a, ...); load_b(b1, ...);
+ __builtin_amdgcn_s_barrier();
  rcr_mma(cA, ...); rcr_mma(cB, ...);
+ __builtin_amdgcn_s_barrier();

  load_a(a, ...);
+ __builtin_amdgcn_s_barrier();
  rcr_mma(cC, ...); rcr_mma(cD, ...);
```

## Numerical probe (HK fp8 grouped, gpt_oss-GateUP-B4-M2048, K=2880)

```
                                                 SNR (dB)
no fuse (external ktail kernel runs)             28.38
path-A fuse, no cooperative zero                 16.84   ← round-2 baseline
path-A + 16384 byte coop zero                    19.99
path-A + 17408 byte coop zero (full data[])      20.04
path-A + coop zero + lgkmcnt sync                20.05
path-A + coop zero + sync + restored barriers    20.69   ← this round
no fuse target                                   28.38
metric SNR gate                                  25.00
gap to gate                                      4.31
```

Ablation per row range (this round, with cC/cD rcr_mma disabled to
locate the broken accumulator subset):

```
rows [  0,  64): diff_mean=5.4   (high, cA broken)
rows [ 64, 128): diff_mean=1.6   (low , cB OK)
rows [128, 192): diff_mean=6.6   (high, cC disabled)
rows [192, 256): diff_mean=6.5   (high, cD disabled)
```

cB row range is at SNR ~28 dB; cA at ~17 dB. **One of cA/cB on the same
warp split into two correct/broken halves** — this is the BF16 round-3
warp_row=0 wc∈{1,3} pattern visible in the FP8 register tile layout.

## Why path A cannot reach 25 dB

BF16 round-3 doc enumerated (and failed) all of:

* `__builtin_amdgcn_s_barrier()` only / `__syncthreads()` only / both
* `vmcnt(0)` / `vmcnt(0) lgkmcnt(0)` / `:::"memory"` clobber
* `__builtin_amdgcn_sched_barrier(0)` before/after every load
* MMA wrapper variants (rcr_mma direct, `mma_ABt_base` unrolled)
* Accumulator aliasing (D=C vs D=zero+D+=A·B^T+C+=D)

None of those broke through the saturation. We re-confirm that on FP8:
the cooperative zero + restored barriers + lgkmcnt sync stack stops at
SNR 20.7 dB, ~4-5 dB short of the 25 dB metric gate.

The hypothesis from BF16 round-3 stands: `subtile_inplace` is capturing
a stale tile-base pointer in the post-epilog-2 SGPR state, so for
certain (warp_row, warp_col) the LDS read addresses don't match the
LDS bytes the cooperative load wrote. Fixing this requires bypassing
`load(reg, st)` entirely.

## Round 4 plan (next round) — path B

Mirror BF16 round-5 path B (BF16 file `kernel_bf16_dynamic.cpp:711+`).
Each lane reads its K-tail register slot directly from HBM via
`buffer_load_b128` into the A/B register tile, sidestepping LDS and
the phantom-read pattern entirely.

FP8-specific work needed:

1. **Lane → cell mapping for `rt_16x128_s` / `A_row_reg` / `B_row_reg`.**
   Round-3 derived the OUTPUT mapping
   `rt_fl<RBM=64, RBN=32, col_l, rt_16x16_s>`:
   `local_col = lane.lo32 % 16, k_idx = (lane.lo32 / 16) * 4 + lane.hi32`.
   The INPUT mapping is what path B needs:
   - 32 fp8 cells/lane per K=128 K-block (= 2 × b128 per lane).
   - Cell layout depends on the `mfma_scale_f32_f8f6f4_16x16x128`
     operand format on CDNA4.
2. **K_REM mask.** FP8 K_STEP=128 but K_REM=64 → half-block load.
   Either gate dispatcher to K_REM == K_STEP only (BF16 approach,
   leaves K_REM ∈ {16, 32, 48, 80, 96, 112} unfused) OR build per-lane
   `if constexpr (k_idx >= K_REM) zero else load`.
3. **Per-group SRD bound for B (RCR layout).** B is `[G, N, K]` with
   per-group N rows. Path B uses `(group_idx + 1) * N * K * 2` bound
   so OOB N rows clamp to 0 (same trick as the main-loop B SRD).
4. **Register pressure.** The 4 prior FP8 K-tail register attempts
   (vec8 hoist + parallel cells, cA+cB serial + outer unroll,
   cell-by-cell with runtime loop, scalar acc) all hit spill. Path B
   reuses the SAME register tiles A_tile / B_tile_0 / B_tile_1 from
   the main loop (dead at this point post-epilog-2), so the increment
   should be ≤ 1 a/b tile worth of VGPR pressure — comparable to one
   K-step of the main loop.

## Files touched (this round)

`analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`:
* `grouped_rcr_kernel<...,FUSED_KTAIL=true>` K-tail epilog: cooperative
  zero of `As[tic][0..1] + Bs[tic][0..1]` (all 4 LDS slots, full
  `sizeof(ST_rcr)` = 17408 bytes/tile to cover `subtile_padding=128`),
  `s_waitcnt lgkmcnt(0)` to drain ds_writes, restored 3 inner
  s_barriers between load_b/load_a/rcr_mma stages.

No primus-turbo changes. No changes to other kernels (BF16 grouped
unchanged; FP8 RRR / var-K dB / dense path unchanged). Metric unchanged
at 153 (the 25 dB gap remains, all 8 gpt_oss FP8 cases still clip to
ratio 0; the SNR moved from 16.84 → 20.7 dB but not across the 25 dB
gate).

## Backward verification

This round only touches `grouped_rcr_kernel` (FP8 forward path; trans_b
= True). FP8 dA path goes through `grouped_rrr_kernel` (no K-tail fuse
yet) and FP8 dB path goes through `grouped_var_k_kernel_fp8` (var-K dB);
both untouched this round so no `bench_grouped_gemm_turbo.py` rerun was
required. The `_metric_grouped_only.py` correctness suite checks fwd +
dA + dB allclose / SNR per-shape and re-confirmed `reject=0/32` (no
NaN/Inf, no exceptions), so backward correctness is intact.
