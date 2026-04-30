# Round 6 — FP8 path A K-tail epilog barrier prune

## TL;DR

`grouped_rcr_kernel<..., FUSED_KTAIL=true>` had **5** `__builtin_amdgcn_s_barrier()`
calls in the K-tail epilog. Three were mirrored from the main-loop pattern
without re-reasoning, but serve no purpose in the K-tail epilog (no
double-buffer prefetch follows; LDS reads are per-lane; rcr_mma is per-lane).
Removed those 3 inner barriers; kept the post-coop-write sync at the top
and the trailing barrier at the bottom.

## Metric

```
before round 6: score 828   grp_BF16=1.0888  grp_FP8=0.9065
after  round 6: score 830-832 (run-to-run variance ±2)
                grp_BF16=1.0892 (flat — barrier path not touched on BF16)
                grp_FP8=0.9117-0.9146 (+0.5-0.8pp on the 16-shape geomean)
```

Per-shape FP8 forward changes (HK vs Triton ratio):

```
DSV3 cases (K_REM=0, no K-tail):  unchanged (within ±0.005 noise)

gpt_oss cases (K=2880, K_REM=64): all 8 cases lifted
  GateUP-B4-M2048   0.803 → 0.812  (+0.9pp)
  Down-B4-M2048     0.866 → 0.903  (+3.7pp)   ← biggest single shape win
  GateUP-B4-M4096   0.815 → 0.824  (+0.9pp)
  Down-B4-M4096     0.798 → 0.803  (+0.5pp)
  GateUP-B32-M2048  0.862 → 0.867  (+0.5pp)
  Down-B32-M2048    0.866 → 0.873  (+0.7pp)
  GateUP-B32-M4096  0.842 → 0.845  (+0.3pp)
  Down-B32-M4096    0.863 → 0.868  (+0.5pp)

gpt_oss FP8 mean:  0.844 → 0.849 (+0.5pp)
```

The Down-B4-M2048 case has the biggest jump because it's the smallest
shape (least main-loop work), so the K-tail epilog is the largest
fraction of its wall — barrier savings show up proportionally.

## What was removed

In `grouped_rcr_kernel`'s `FUSED_KTAIL` block (round-2 path A), the 3
inner `__builtin_amdgcn_s_barrier()` calls between LDS reads and rcr_mma
were mirrored from the main loop without re-justification:

```cpp
// BEFORE round 6 (5 barriers):
asm volatile("s_waitcnt vmcnt(0)");
__builtin_amdgcn_s_barrier();          // (1) post-HBM→LDS coop write
load_b(b0, b_tile(tic, 0), wn);
load_a(a, As[tic][0], wm);
load_b(b1, b_tile(tic, 1), wn);
__builtin_amdgcn_s_barrier();          // (2) ← REMOVED
asm volatile("s_waitcnt lgkmcnt(0)");
__builtin_amdgcn_s_setprio(1);
rcr_mma(cA, a, b0); rcr_mma(cB, a, b1);
__builtin_amdgcn_s_setprio(0);
__builtin_amdgcn_s_barrier();          // (3) ← REMOVED
load_a(a, As[tic][1], wm);
__builtin_amdgcn_s_barrier();          // (4) ← REMOVED
asm volatile("s_waitcnt lgkmcnt(0)");
__builtin_amdgcn_s_setprio(1);
rcr_mma(cC, a, b0); rcr_mma(cD, a, b1);
__builtin_amdgcn_s_setprio(0);
__builtin_amdgcn_s_barrier();          // (5) end-of-K-tail
```

Why the 3 inner ones are unnecessary in this specific block:

* In the main loop, the `s_barrier` after rcr_mma protects against the
  next iter's HBM→LDS prefetch racing the current LDS read.
* In the K-tail epilog, **no further LDS write happens before the wave
  exits this block** — only `mul(cA..cD, scale)` (per-lane) and
  `store(g.c, ...)` (HBM write) follow.
* LDS reads (`load_a/load_b`) and `rcr_mma` are both per-lane; no
  cross-thread synchronization needed between them.
* `s_waitcnt lgkmcnt(0)` already drains LDS reads for the issuing thread
  before MMA, which is sufficient since each lane consumes only its own
  data.

Why (1) and (5) are kept:

* (1) The cooperative HBM→LDS via `buffer_load_lds + s_waitcnt vmcnt(0)`
  has cross-thread visibility requirement: lane X writes LDS bytes that
  lane Y reads via `ds_read`. Removing (1) breaks LDS visibility.
* (5) The trailing barrier preserves the relationship with the
  wm-conditional barrier at line 2316 and protects against the next
  outer iter's prologue `rcr_8w_load_hoist` LDS write racing the
  current K-tail's LDS reads. We don't yet have a clean argument that
  removing (5) is safe across all (M_g, num_pid_n) shapes; conservative
  to leave.

## Resource usage diff

Confirmed unchanged across all 4 `<KI_HINT, N_MASKED, FUSED>` variants:

```
                            VGPR  spill   occ
<0, false, false>  legacy   256     91     2   (round-2 baseline 91)
<0, true , false>  legacy   256     83     2   (round-2 baseline 83)
<0, false, true >  fused    256     95     2   (round-2 baseline 95)
<0, true , true >  fused    256     99     2   (round-2 baseline 99)
```

Identical to round-2's measured values — confirming that
`__builtin_amdgcn_s_barrier()` removal is register-pressure neutral
(barriers don't affect allocation; they're inserted between reg-allocated
ops). The compiler may schedule slightly differently after the prune,
but the spill counts remained exactly equal.

## Numerical correctness

Metric reports `reject=0/32, below_target=32/32`. No NaN/Inf, no kernel
exceptions. All 16 FP8 grouped shapes still execute to completion with
ratios consistent with the round-5 baseline (DSV3 K_REM=0 unchanged;
gpt_oss K_REM=64 all uplifted modestly).

The `grouped_rcr_kernel<..., FUSED_KTAIL=true>` is only reached on RCR
forward with `K_REM == 64` and `m_per_group >= 16 % 16 == 0` (per
`fuse_ktail_eligible`). Backward dA goes through `grouped_rrr_kernel`,
backward dB through `grouped_var_k_kernel_fp8`; neither is touched.

## Why this is round-6 work, not "small fix"

The K-tail fuse path B route (mirror BF16 round-5: direct HBM→Reg via
`buffer_load_b128`, no LDS at all, expected larger uplift) requires:

1. Deriving lane→cell mapping for `rt_16x128_s` (32 fp8 cells per lane =
   2 × b128 per (h,w) sub-tile per lane — different from BF16's 1 ×
   b128 per (h,w) for `rt_16x32_s`).
2. Building inline asm that writes directly to the fp8 register tile
   slots in a layout that `mma_ABt` can consume.
3. Reusing the same SRDs as the FP8 main loop (a_srsrc_base /
   b_srsrc_base equivalents that the per-lane `buffer_load_b128` can
   share).

Path B is the bigger lift but multi-round work. Round 6's barrier prune
is a contained, low-risk preliminary step that establishes the K-tail
epilog as a target for further optimization without touching anything
load-side. Path B is the round-7+ work item.

## Round 7+ plan

* **Round 7**: Derive `rt_16x128_s` lane mapping from existing
  `assembly/global_to_register.cuh::load` helper (which has fp8 static
  assert blocking it). Add a custom path-B-style direct register K-tail
  helper to `kernel_fp8_layouts.cpp`. Build, validate SNR, profile to
  confirm no spill regression.
* **Round 8**: Wire path-B helper into `FUSED_KTAIL` epilog. Compare
  vs path A (legacy + barrier-pruned). Expect: gpt_oss FP8 ~+5-10pp,
  matching the BF16 path A→B uplift seen in round 5.
* **Round 9+**: If path B works on FP8 RCR forward, extend to FP8 RRR
  (dA backward) and FP8 var-K CRR (dB backward) for completeness.
