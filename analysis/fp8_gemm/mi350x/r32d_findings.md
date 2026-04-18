R32 Dev D — Stage A1 SHIP: rect-V2 RCR fastpath kernel scaffolded, dispatched, GPU-fault-clean
==============================================================================================

Status: **SHIP Stage A1a + A1b**. Stage A1c (correct numerics) explicitly
deferred to next cycle (matches Dev A's R31 timeline for the CRR side —
host preshuffle wiring + Stage A2 K_HALF audit).

TL;DR
-----
- New file `rcr_mxfp8_exact_8wave_rect_fastpath.inc` (~470 lines) contains a
  parallel template `rcr_exact_8wave_scaled_rect_kernel<true>` that mirrors
  the structure of the square `rcr_exact_8wave_scaled_kernel<true,2>` but
  with rect-N register types (`B_row_reg_rect`), rect shared B tile
  (`ST_B_rcr_rect = st_fp8e4m3<HB_N=64, BK=128, st_16x128_s>`), rect
  accumulator (`rcr_exact_acc_rect` with 4 floatx4 regs vs square's 8),
  and rect V2 scale slab geometry (pack_count_b=1, slab_bytes_b=32*kp).
- New helpers: `rcr_mma_scaled_from_packs_fixed_phase_row_rect<ROW_BASE>` and
  `rcr_mma_scaled_from_packs_fixed_phase_impl_rect` (reuse the existing
  `rcr_exact_mfma_scale_builtin_inplace` MFMA primitive). Also
  `rcr_exact_acc_rect_to_rt` (rect store helper).
- Host: `rcr_can_use_exact_8wave_scaled_rect` predicate + `dispatch_rcr_exact_8wave_scaled_v2_rect<PRESHUFFLED_QUANT>` dispatcher.
- Default build (`MXFP8_RECT_BLK_N=128`) byte-identical to head:
  md5 = `7d6c1ae78ee0001b45930835237673e6` (matches r32-d HEAD pre-Stage-A1
  baseline, pre- and post-changes verified).
- Rect build (`-DMXFP8_RECT_BLK_N=64`) compiles clean (rc=0) with no new
  warnings beyond the existing `-Rpass=kernel-resource-usage` remarks.
  Rect .so md5 = `830dbf96b29b9e1d5c241a82cfc7fabd` (different from default,
  as expected — the rect kernel symbol is materialized).
- Rect kernel runs on GPU3 (HIP_VISIBLE_DEVICES=3) at the V2-RCR target shape
  4096x4096x4096 without GPU fault, segfault, or kernel timeout. Numerics
  produce zeros / denormals (expected — the host preshuffle is the SQUARE
  layout but the rect kernel reads rect-slab geometry; this is the Stage A2
  work). Output line: `[r32d-A1b] STAGE_A1b_RESULT: NO_FAULT`.
- **Bonus structural finding**: rect RCR kernel uses **137 VGPRs** (vs 246
  for square V2-RCR) and **98,304 LDS bytes/block** (vs 131,072 for square)
  — a **44% VGPR drop** and **25% LDS drop**. Both rect and square fastpaths
  report `Occupancy [waves/SIMD]: 2` — same LDS-binding ceiling as Dev A
  observed for CRR rect. Rect-RCR LDS-per-block = 96 KB; 2 blocks = 192 KB
  which exceeds the 160-KB-per-CU LDS limit, so 1 block/CU remains the real
  per-CU LDS-binding cap. Rect VGPR drop is **even larger than CRR rect**
  (-44% RCR vs -37% CRR), because RCR's `rcr_exact_acc` is a flat
  `floatx4 regs[8]` and halving RBN halves it directly to `regs[4]`.

Files added/touched this cycle
------------------------------
+ `analysis/fp8_gemm/mi350x/rcr_mxfp8_exact_8wave_rect_fastpath.inc` (NEW —
  rect-V2 RCR fastpath kernel, helpers, dispatcher).
+ `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp`:
  - new `#include "rcr_mxfp8_exact_8wave_rect_fastpath.inc"` after the
    existing CRR rect include (header internally guarded by
    `MXFP8_RECT_BLK_N == 64`, so default builds see an empty translation
    unit and remain byte-identical).
  - extended the V2-RCR branch of `dispatch_pq_v2<Layout::RCR>` so that
    when `MXFP8_RECT_BLK_N=64` AND the shape predicate matches, dispatch
    routes to `dispatch_rcr_exact_8wave_scaled_v2_rect<true>(g)` instead
    of the square `dispatch_rcr_exact_8wave_scaled_v2<true>(g)`. Falls
    through to the square dispatch otherwise.
+ `analysis/fp8_gemm/mi350x/r32d_orchestrate.sh` (NEW — 3-cell runner).
+ `analysis/fp8_gemm/mi350x/r32d_stage_a1b_gpufault_test.py` (NEW — minimal
  GPU-fault test on V2-RCR 4096^3 with intentionally over-allocated rect-B
  scale buffer).
+ `analysis/fp8_gemm/mi350x/r32d_baseline_default_build.log`,
  `r32d_default_postchange_build.log`, `r32d_rect_build.log` — build logs.
+ `analysis/fp8_gemm/mi350x/r32d_stage_a1b_result.txt` — Cell 3 output.
+ `analysis/fp8_gemm/mi350x/r32d_findings.md` (NEW — this file).

Stage A1 deliverables (per task spec)
-------------------------------------
**Stage A1a — clean compile of rect-V2 RCR fastpath with new B-side helper variants.**
SHIPPED. Default build md5 matches head baseline both pre- and post-edits;
rect build compiles cleanly (rc=0) and emits the rect kernel symbol. Resource
report (next section) shows no spills.

**Stage A1b — rect build runs without GPU fault on V2-RCR 4096x4096x4096.**
SHIPPED. `r32d_stage_a1b_gpufault_test.py` launches the rect kernel via
`gemm_rcr_pq_v2` with `HIP_VISIBLE_DEVICES=3` and an over-allocated
zero-filled B-scale buffer (`num_slabs_rect=128 x slab_bytes=32*16=4096
bytes/slab = 524 KB total`). Kernel completes in ~4ms with no fault. Output:
```
[r32d-A1b] STAGE_A1b_RESULT: NO_FAULT
```

**Stage A1c — correct numerics on at least one configuration.** **NOT
SHIPPED.** Two issues block A1c (and therefore Stage A2 work, deferred to
R33+):

1. **Host preshuffle for rect B**: `r29a_bench.py:114-138` `preshuffle_v2_b`
   is hardcoded to square layout (`blk=256, hb=128, rbn=32, warps_n=4`,
   `pack_count=2`). Need a parallel `preshuffle_v2_b_rect` with `blk_n=128,
   hb_n=64, rbn_rect=16, pack_count=1`. The slab indexing math drops the
   second-pack copy entirely. Final shape: `(num_slabs_rect, 1*32*padded_kb)
   = (128, 4096)` for 4096^3 (the over-allocated buffer in this test
   matches that size exactly).
2. **K-axis is fine for RCR rect** (no K_HALF stub needed): RCR rect uses
   the generic `rcr_exact_load_st_to_rt` (a generic ST→RT copy) on a 64x128
   `st_16x128_s` tile, so a single load covers all of K=0..127 for the
   warp's 16 N-rows. **This is a key simplification vs Dev A's CRR rect**,
   where the v2_s ds_read_b64_tr_b8 path required a K_HALF stub; the RCR
   path's row-layout B reads do not have that issue.

Build resource report (the major bonus finding)
-----------------------------------------------
Cell 2 captures `-Rpass-analysis=kernel-resource-usage` remarks for both
the new rect RCR kernel and the existing square V2-RCR kernel:

| Kernel                                  | VGPRs | LDS bytes/block | Occupancy waves/SIMD | Spills |
|-----------------------------------------|------:|----------------:|---------------------:|-------:|
| Square V2-RCR (`rcr_exact_8wave_scaled_kernel<true,2>`) |   246 |         131,072 |                    2 |      0 |
| **Rect V2-RCR** (`rcr_exact_8wave_scaled_rect_kernel<true>`) | **137** |     **98,304** |                **2** |  **0** |
| Δ vs square                             |  -109 |         -32,768 |                    0 |      0 |
| Δ %                                     |  -44% |             -25% |                    — |      — |

Comparison with Dev A's R31 CRR rect numbers:

| Kernel                  | VGPRs | LDS B/blk | Occ |
|-------------------------|------:|----------:|----:|
| Square V2-CRR           |   234 |   139,264 |   2 |
| Rect V2-CRR             |   148 |   104,448 |   2 |
| Square V2-RCR           |   246 |   131,072 |   2 |
| **Rect V2-RCR**         | **137** | **98,304** |   2 |

Interpretation:
- **Rect RCR drops MORE VGPRs than rect CRR** (-44% RCR vs -37% CRR). Reason:
  RCR's `rcr_exact_acc` is `floatx4 regs[8]` (8 VGPR groups for the 64x32
  accumulator). Halving RBN to RBN_RECT=16 collapses regs[8] → regs[4],
  saving 4 floatx4 = 16 VGPRs of accumulator state. The R32D rect kernel
  also omits the optional configurability paths (PIPELINE_SCALE,
  KPAIR_LOOP, PHASE_U16_CACHE, REMAP_ONCE) that bloat the square kernel
  with conditional VGPR allocations.
- **LDS drop -25% comes mainly from the smaller B shared tile**
  (`ST_B_rcr_rect = st_fp8e4m3<HB_N=64, BK=128>` is half the size of
  `ST_B = st_fp8e4m3<HB=128, BK=128>`). With the 2-deep `Bs[2][2]`
  double-buffer this gives a -32 KB B-side LDS saving (= 4 tiles x 8 KB).
- **`Occupancy [waves/SIMD]: 2` does not increase** despite the resource
  drops. The compiler's "waves/SIMD: 2" is per-thread VGPR/SGPR-only and
  ignores the per-CU LDS budget. Real LDS-binding cap: 304 CUs x 160 KB
  LDS = 48,640 KB total. At 98,304 bytes/block (= 96 KB), 2 blocks/CU =
  192 KB which exceeds the 160-KB-per-CU LDS limit. So 1 block/CU is the
  LDS-binding cap on rect RCR — same paradigm closed by R31 Dev D for
  V2-RCR persistent-CU.
- **R31 Dev D's structural-CU prediction holds**: rect RCR at 4096^3 yields
  16 x 32 = 512 tiles vs square's 256. With persistent-CU constraint
  `total_tiles >= num_CUs` now satisfied (512 > 304), wave-fill becomes
  ~512/(304*1) = 1.68 waves at occupancy=1, vs square's 256/304 = 0.84
  waves. Rect halves per-tile work, doubles tile count → at first-order
  same total work but eliminates the 16% idle gap. **Net potential gain
  ≈ 25%** before per-tile-overhead penalties from rect.

What blocks Stage A2 next
-------------------------
Two hard problems for Stage A2 (in priority order):

**A2.1 — Host-side preshuffle for rect B (~1-2 hours)**:
- Adapt `r29a_bench.py:114-138` `preshuffle_v2_b` for rect: drop the
  second-pack copy (`pack_count_rect=1` instead of 2), use `blk_n_rect=128`
  and `hb_n_rect=64` instead of `blk=256, hb=128`, and use `rbn_rect=16`
  instead of `rbn=32`.
- Final shape: `(num_slabs_rect, 1*32*padded_kb)`. For 4096^3:
  `num_slabs_rect = 32 * 4 = 128`, `slab_size = 32 * 128 = 4096` (since
  padded_k_blocks = 4096/32 = 128 → padded to 128, no further rounding).
  Total = 524 KB (matches what Stage A1b over-allocated as a placeholder).
- Output bench harness: `r32d_bench.py` should call
  `tk_mxfp8_layouts.gemm_rcr_pq_v2` with this rect-preshuffled B (and
  unchanged A from `preshuffle_v2_a`).

**A2.2 — Verify B-tile load coordinate semantics for rect (~1-2 hours)**:
- The rect kernel uses `kittens::subtile_inplace<RBN_RECT, BK>(Bs[tic][0],
  {wn, 0})` to extract the warp's 16x128 sub-region from the 64x128 rect
  tile. With WARPS_N=4 and RBN_RECT=16 each warp gets a different 16-row
  N-stripe. Need to verify (a) `prefill_swizzled_offsets` produces correct
  byte offsets for the rect tile shape, (b) the subtile coordinate `{wn, 0}`
  selects the correct 16-row slice, (c) `G::load(Bs, g.b, {0, 0, bc*2, k})`
  loads the right global B columns for the rect tile (rect uses 2 N-half
  tiles per BLK_N, same pattern as square).
- After A2.1+A2.2 are done, Stage A2 SHIP gate is `SNR ≥ 48 dB on
  4096^3` and `det 3/3` on the rect kernel. Stage A3 perf SHIP and Stage A4
  LLaMA matrix follow per Dev A's CRR template.

Methodology rules followed (R31 closures)
-----------------------------------------
1. **`rm -f tk_mxfp8_layouts*.so` + log per-build md5** (R29 Dev C rule):
   followed for both default (md5 `7d6c1ae78...`) and rect (md5
   `830dbf96b...`) builds. Default md5 matches HEAD baseline both
   pre- and post-changes.
2. **`rocm-smi -d 3` (matches HIP_VISIBLE_DEVICES=3 on physical GPU3)**
   (R31 Reviewer rule): used in the test harness; sclk 1828 MHz observed
   at A1b run time (low-power state — kernel ran fine in 4ms, perf timing
   not relevant for fault test).
3. **All `MXFP8_*_PERSISTENT_GRID` style macros build-time-assert
   `grid >= total_tiles`** (R31 Dev D rule from prior cycle): not needed
   this cycle because the rect kernel uses the natural grid `(M/BLK) *
   (N/BLK_N) = 16 * 32 = 512` which IS `total_tiles` exactly (no
   persistent-CU dispatch involved). Documented in dispatcher comment.
4. **All in-process A/B: BABA pattern + 30s preheat** (R31 Dev D rule):
   not exercised this cycle because Stage A1b is a single GPU-fault test,
   not a perf measurement. Will apply to Stage A3+.

CLOSED levers respected (per R32D prompt — 18 total)
-----------------------------------------------------
None of the 18 closed levers were prototyped this cycle. The rect-V2 RCR
fastpath is the **only structural lever still open** for the 4096^3 V2-RCR
gap (per R31 Dev D's closure analysis). All other tested paths are NULL
or DEAD-END.

SHIP gate evaluation
--------------------
- **Default build byte-identical**: PASS (md5 `7d6c1ae78ee0001b45930835237673e6`
  pre- and post-changes verified).
- **Rect build compiles**: PASS (rc=0, no new warnings beyond existing
  resource-usage remarks).
- **Rect kernel runs without GPU fault**: PASS (Cell 3 output line
  `STAGE_A1b_RESULT: NO_FAULT`).
- Correctness (SNR ≥ 48 dB): N/A (Stage A2 work, deferred).
- Determinism (3/3): N/A (numerics garbage, det meaningless).
- Performance (Welch t > 3.0): N/A (Stage A3 work).

**Verdict: SHIP Stage A1a + A1b** (the only gates evaluable for Stage A1
per the task spec). Three SHIP gates (correctness/perf/Welch-t) are
explicitly out-of-scope for Stage A1 per the task brief.

Print stamp: `=== R32 DEV D SHIP STAGE A1a+A1b — RECT-V2 RCR FASTPATH SCAFFOLDED, GPU-FAULT-CLEAN ===`

Time budget actuals
-------------------
- Stage A1a (file creation + dispatcher wiring): ~2.5 hours.
- Stage A1b (gpufault test + GPU validation): ~0.5 hours.
- Documentation + commit: ~1.0 hour.
- **Total: ~4 hours** (under the 6h budget).

Exact next-cycle starting point (for R33 Dev D continuation)
------------------------------------------------------------
1. Read this file + `r31a_findings.md` (CRR rect Stage A1 — same paradigm).
2. `git checkout r32-d` and verify HEAD is the Stage A1 commit.
3. Implement Stage A2.1 (host preshuffle rect-B variant in a new
   `r33d_bench.py` adapted from `r29a_bench.py`). Cross-check buffer size
   against `slab_bytes_b_rect = 32 * padded_k_blocks` in
   `rcr_mxfp8_exact_8wave_rect_fastpath.inc`.
4. Run rect kernel on 4096x4096x4096 with rect-preshuffled scales; compare
   SNR vs reference. Target ≥ 48 dB.
5. If SNR fails, audit the B-tile coordinate semantics (Stage A2.2 above).
6. Once correctness passes, proceed to Stage A3 (perf SHIP via paired
   in-process A/B vs square V2-RCR baseline).

R32 budget update: A1 cost ~4 hours (this cycle), A2 ~3-4 hours (estimated),
A3 ~4 hours (perf SHIP), A4 ~4 hours (LLaMA matrix). Total remaining: ~12
hours = ~2 R32-budget cycles after this one.
