R31 Dev A — Stage A1 SHIP: rect-V2 CRR fastpath kernel scaffolded, dispatched, GPU-fault-clean
=============================================================================================

Status: **SHIP Stage A1a + A1b**. Stage A1c (correct numerics) explicitly
deferred to next cycle (requires Stage A2 host preshuffle work — out of
scope for R31).

TL;DR
-----
- New file `crr_mxfp8_exact_8wave_rect_fastpath.inc` (~470 lines) contains a
  parallel template `crr_exact_8wave_scaled_rect_kernel` with rect-N register
  types, rect interleave helpers, rect MMA helpers, rect V2 scale slab geometry,
  and host-side `dispatch_crr_exact_8wave_scaled_v2_rect` + predicate
  `crr_can_use_exact_8wave_scaled_rect`.
- Default build (`MXFP8_RECT_BLK_N=128`) byte-identical to head:
  md5 = `79c2816c54a00cf0680d32a36af25c98` (matches r30-a HEAD pre-Stage-A1).
- Rect build (`-DMXFP8_RECT_BLK_N=64`) compiles clean with no warnings beyond
  the existing -Rpass=kernel-resource-usage remarks.
- Rect kernel runs on GPU0 at 70B KV target shape (4096×1024×8192) without
  GPU fault, segfault, or kernel timeout. Numerics produce NaNs (expected —
  scale slab geometry mismatch with current host preshuffle, which is the
  Stage A2 work).
- **Bonus structural finding**: rect kernel uses **148 VGPRs** (vs 234 for
  square) and **104,448 LDS bytes/block** (vs 139,264 for square) — a 37%
  VGPR drop and 25% LDS drop. Both the rect and square fastpaths report
  `Occupancy [waves/SIMD]: 2` after compilation. Pushing
  `MIN_BLOCKS_PER_CU=3` does NOT increase the reported occupancy (LDS-bound
  per CU at 2 blocks/CU). Real occupancy gain will require either a tighter
  LDS layout or split-K.

Files added/touched this cycle
------------------------------
+ `analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_rect_fastpath.inc` (NEW —
  rect-V2 CRR fastpath kernel, helpers, dispatcher).
+ `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp`:
  - line 3529: `#include "crr_mxfp8_exact_8wave_rect_fastpath.inc"` (header
    is internally guarded by `MXFP8_RECT_BLK_N == 64`, so default builds
    see an empty translation unit and remain byte-identical).
  - lines 5384-5404: replaced R29 Dev A's hard-guard (host-side `fprintf` +
    `return`) with a real dispatch to `dispatch_crr_exact_8wave_scaled_v2_rect`
    when shape matches `crr_can_use_exact_8wave_scaled_rect`.
+ `analysis/fp8_gemm/mi350x/r31a_orchestrate.sh` (NEW — 4-cell runner).
+ `analysis/fp8_gemm/mi350x/r31a_stage_a1b_gpufault_test.py` (NEW — minimal
  GPU-fault test with intentionally over-allocated B-scale buffer).
+ `analysis/fp8_gemm/mi350x/r31a_cell{1,2,3,4}_*.txt` (NEW — cell outputs).
+ `analysis/fp8_gemm/mi350x/r31a_findings.md` (NEW — this file).

Stage A1 deliverables (per task spec)
-------------------------------------
**Stage A1a — clean compile of rect-V2 fastpath with new B-side helper variant.**
SHIPPED. Cell 1 confirms default build md5 unchanged from r30-a head; Cell 2
confirms rect build compiles cleanly (rc=0).

**Stage A1b — rect build runs without GPU fault on 70B KV shape (4096×1024×8192).**
SHIPPED. Cell 3 (`r31a_stage_a1b_gpufault_test.py`) launches the rect kernel
via `gemm_crr_pq_v2` with `HIP_VISIBLE_DEVICES=0` and an intentionally
over-allocated B-scale buffer (32 slabs × 8192 bytes = 262,144 bytes for the
rect slab geometry, vs the square's 4 slabs × 16384 = 65,536). Kernel
completes in ~2ms with no fault. Output line:
```
[r31a-A1b] STAGE_A1b_RESULT: NO_FAULT
```

**Stage A1c — correct numerics on at least one configuration.** **NOT
SHIPPED.** This requires Stage A2 host preshuffle to be wired (current
`preshuffle_v2_b` produces square slab layout; rect kernel reads square
geometry → garbage values → NaN sum). See "What blocks Stage A2 next" below.

Build resource report (the major bonus finding)
-----------------------------------------------
Cell 2 captures `-Rpass-analysis=kernel-resource-usage` remarks for both
the new rect kernel and the existing square V2-CRR kernel:

| Kernel                                  | VGPRs | LDS bytes/block | Occupancy waves/SIMD | Spills |
|-----------------------------------------|------:|----------------:|---------------------:|-------:|
| Square V2-CRR (`crr_exact_8wave_scaled_kernel<true,2>`) |   234 |         139,264 |                    2 |      0 |
| **Rect V2-CRR** (`crr_exact_8wave_scaled_rect_kernel<true>`) | **148** |     **104,448** |                **2** |  **0** |
| Δ vs square                             |   -86 |         -34,816 |                    0 |      0 |
| Δ %                                     |  -37% |             -25% |                    — |      — |

Interpretation:
- **VGPR drop -37% is dramatic** and exceeds expectations. It comes from
  (a) `B_col_reg_rect` being half the size of `B_col_reg`
  (`rt_fp8e4m3<BK, RBN_RECT=16, …>` vs `RBN=32`), (b) all four accumulator
  tiles `cA/cB/cC/cD` of type `rt_fl<RBM, RBN_RECT, …>` being half the size,
  and (c) `crr_b_pack_count_rect=1` collapsing several scale pack arrays.
- **LDS drop -25% comes mainly from the smaller B shared tile**
  (`ST_v2_rect = st_fp8e4m3<HB_N=64, BK=128>` is half the size of
  `ST_v2 = st_fp8e4m3<HB=128, BK=128>`). With 2-deep `Bs[2][2]` double-buffer
  this gives a -32 KB B-side LDS saving (= 4 tiles × 8 KB).
- **`Occupancy [waves/SIMD]: 2` does not increase** despite the resource
  drops. Bumping `GEMM_MIN_BLOCKS_PER_CU` from 2→3 (Cell 4) does not raise
  the compiler's reported occupancy either. **The LDS budget per CU is
  binding**: 304 CUs × 160 KB LDS = 48,640 KB total. At 104,448 bytes/block
  (=102 KB), 2 blocks/CU = 204 KB which exceeds the 160-KB-per-CU LDS limit.
  So 1 block/CU is the LDS-binding cap on rect, and the launch_bounds(256, 2)
  request can't actually be satisfied — the compiler's "waves/SIMD: 2"
  appears to be a per-thread VGPR/SGPR-only metric independent of LDS-per-CU.
- **R30 Dev B's prediction (LDS shrink → 2-blocks/CU eligibility) does NOT
  pan out** at the current LDS layout. To unlock 2 blocks/CU we'd need to
  drive LDS below 80 KB/block. That requires either a single-buffered Bs
  or a 1-deep `Bs[1][2]` (no double-buffer prefetch). Both are deferred to
  Stage A3+ kernel-tuning.

Per the R30 paradigm correction #1 the rect tile alone targets ~1.5x speedup
on 70B KV (2x N-tiles × ~75% of square per-block throughput). The 37% VGPR
drop and 25% LDS drop strongly suggest there IS structural headroom for the
1.5x target once Stages A2/A3 wire up correct numerics + perf tuning.

What I actually changed in the kernel
-------------------------------------
1. **Rect-N register / shared types** (in
   `crr_mxfp8_exact_8wave_rect_fastpath.inc:48-50`):
   - `ST_v2_rect = st_fp8e4m3<HB_N, BK, st_16x128_v2_s>` (64×128 vs 128×128).
   - `B_col_reg_rect = rt_fp8e4m3<BK, RBN_RECT, col_l, rt_128x16_s>` (128×16
     vs 128×32).

2. **Rect B-side load helper** (`load_col_from_v2_st_rect`,
   `crr_mxfp8_exact_8wave_rect_fastpath.inc:104-144`):
   - Calls existing `load_col_from_v2_st_half_rect<RT, 0, HB_N>` for
     K_HALF=0 (R28D scaffolded helper, math validated).
   - For K_HALF=1 path: re-issues the same K_HALF=0 ds_read math but writes
     into the K_HALF=1 register slot (idx=4). **THIS IS A STAGE A1a STUB
     ONLY** — duplicates K=0..63 data into K=64..127 register slots, which
     guarantees no GPU fault but produces wrong numerics. Stage A2 must
     replace this with a real K=64..127 read (likely from a second 64-row
     shared tile or via a redesigned helper that addresses K-rows directly).

3. **Rect MMA primitives** (`crr_mma_scaled_base_rect`,
   `crr_mma_scaled_phase_rect`, `crr_mma_scaled_from_packs_fixed_phase_rect`)
   — bodies structurally identical to the square versions in
   `kernel_mxfp8_layouts.cpp:1344-1656` but using `rt_fl<RBM, RBN_RECT, …>`
   accumulator and `B_col_reg_rect`. The underlying
   `mma_AB_base_scaled<opsel_a, opsel_b>` / `mma_ABt_base_scaled` / etc.
   primitives are template-on-tile-types so no further plumbing was needed.

4. **Rect interleave helper** (`crr_exact_cA_with_b1_interleave_fixed_phase_rect`)
   — same 8-MMA + 8-load-slot interleave structure as the square version
   but uses `_rect` types and `_rect` MMA primitives. Pack-count compresses
   to 1 (RBN_RECT=16 < 32) so all m-iteration scale lookups collapse to
   `b0_phase[0]`.

5. **Rect kernel body** (`crr_exact_8wave_scaled_rect_kernel`,
   `crr_mxfp8_exact_8wave_rect_fastpath.inc:283-540`) — same structure as
   `crr_exact_8wave_scaled_kernel` in the square fastpath, but:
   - `crr_b_pack_count_rect = (RBN_RECT + 31) / 32 = 1`.
   - `blocks_per_col = g.n / BLK_N` (= 8 for N=1024 instead of 4).
   - `crr_scale_b_base_rect(half) = bc * BLK_N + half * (BLK_N/2) + wn * RBN_RECT`.
   - V2 SRD slab geometry: `slab_bytes_b_rect = 32 * padded_k_blocks`
     (vs 64 * padded_k_blocks for square — half the size because
     pack_count_b dropped from 2 to 1).
   - V1 SCALE_VERSION path is NOT implemented (rect kernel always runs V2).
     If ever needed, would compose from the existing `load_scale_pair_pack_*`
     non-pipelined helpers.

6. **Dispatcher wiring** (`kernel_mxfp8_layouts.cpp:5384-5404`):
   - Removed R29A's `fprintf+return` hard-guard.
   - Added shape-conditional dispatch to
     `dispatch_crr_exact_8wave_scaled_v2_rect<true>(g)` when
     `crr_can_use_exact_8wave_scaled_rect(g)` returns true.
   - Falls through to a host-side error message if the shape is rect-mode
     but doesn't match the rect predicate (M%256!=0, N%128!=0, K%128!=0,
     or M_DIM/N_DIM/K_DIM mismatch).

What blocks Stage A2 next
-------------------------
Two hard problems to solve in Stage A2 (in priority order):

**A2.1 — Host-side preshuffle for rect B (~2-3 hours)**:
- `r29a_bench.py:114-138` `preshuffle_v2_b` is hardcoded to `blk=256, hb=128,
  rbn=32, warps_n=4` with `pack_count=2`. Need a parallel
  `preshuffle_v2_b_rect(scale_exp, blk_n=128, hb_n=64, rbn_rect=16,
  warps_n=4)` with `pack_count=1`.
- The slab indexing math at `r29a_bench.py:122-132` becomes:
    `num_slabs_b_rect = num_ctiles * warps_n` where `num_ctiles = N // BLK_N`.
    `pack_count_rect = 1`. The `perm_view` becomes shape `(num_ctiles,
    warps_n, 1, 32, padded_kb)` (vs `(…, 2, 32, padded_kb)` for square).
    The "rg_hi_off = hb // 32" hi-pack copy disappears (no second pack).
- Final shape: `(num_slabs_rect, 1 * 32 * padded_kb)`. For N=1024, K=8192:
  num_slabs_rect = 32, slab_size = 8192 → total 262,144 bytes (matches what
  Stage A1b over-allocated as a placeholder).
- Output bench harness: `r31a_bench.py` should call
  `tk_mxfp8_layouts.gemm_crr_pq_v2` with this rect-preshuffled B (and
  unchanged A from `preshuffle_v2_a`).

**A2.2 — Fix the K_HALF=1 stub in `load_col_from_v2_st_rect` (~3-4 hours)**:
- The Stage A1a stub at `crr_mxfp8_exact_8wave_rect_fastpath.inc:115-143`
  duplicates K=0..63 data into the K_HALF=1 register slot. Stage A2 must
  replace this with a real K=64..127 read.
- Two design options:
  - **Option 1 (preferred)**: keep the 64-row tile but redesign the helper
    to address K-rows directly (k_row = K_HALF * 64 + row_off where K_HALF
    indexes the K-half within the SAME 64-row tile). The current helper
    treats K_HALF as an N-half offset (k_row = row_off + K_HALF * 64) which
    only works for HB_N=128. For HB_N=64 we'd need k_row = row_off and the
    K-axis offset bakes into a *different shared tile* per K-half.
  - **Option 2**: keep the existing helper math but use TWO separate B
    shared tiles per K-pair (one per K-half), each HB_N=64 × BK=64.
    Increases LDS footprint and scheduling complexity but matches the
    helper's existing math.
- **Option 1 is preferred** because LDS is already the binding constraint
  and Option 2 doubles the B-side LDS allocation (back to square's ~32 KB).

After Stages A2.1 + A2.2 are done, Stage A2 SHIP gate is `SNR ≥ 48 dB on
4096×1024×8192` and `det 3/3` on the rect kernel. Stage A3+A4 (perf SHIP +
LLaMA matrix) follow per `r30a_findings.md:140-208`.

Exact next-cycle starting point
-------------------------------
1. Read this file.
2. `git checkout r31-a` and verify head is the Stage A1 commit.
3. Implement Stage A2.1 (host preshuffle rect-B variant in `r31a_bench.py`).
   Cross-check buffer size against `slab_bytes_b_rect = 32 * padded_k_blocks`
   in `crr_mxfp8_exact_8wave_rect_fastpath.inc:354`.
4. Implement Stage A2.2 (real K_HALF=1 path in `load_col_from_v2_st_rect`).
   Recommend Option 1 above; if time-pressed, Option 2 is a fallback.
5. Run rect kernel on 4096×1024×8192 with rect-preshuffled scales; compare
   SNR vs reference. Target ≥ 48 dB.

R30 Dev D's 2.5-day estimate updated: A1 cost ~1.0 hour (this cycle, much
faster than estimated due to scaffolding payoff), A2 ~5 hours (estimated),
A3 ~4 hours (perf SHIP), A4 ~4 hours (LLaMA matrix + Welch-t). Total
remaining: ~13 hours, or ~2 R31-budget cycles after this one.

SHIP gate evaluation
--------------------
- Correctness: SNR ≥ 48 dB on rect → **N/A** (Stage A2 work).
- Determinism: 3/3 → **N/A** (numerics are NaN, det check meaningless).
- Performance: 70B KV V2-CRR rect ≥ square × 1.5 → **N/A** (Stage A3 work).
- No regression: default build byte-identical → **PASS** (md5 verified).
- Welch t > 3.0 between rect-V2 and square-V2 → **N/A** (Stage A3 work).

Verdict: **SHIP Stage A1a + A1b** (the only gates evaluable for Stage A1
per the task spec). Three SHIP gates (correctness/perf/Welch-t) are
explicitly out-of-scope for Stage A1 per the task brief.

Print stamp: `=== R31 DEV A SHIP STAGE A1a+A1b — RECT-V2 CRR FASTPATH SCAFFOLDED, GPU-FAULT-CLEAN ===`
