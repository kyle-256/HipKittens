R33 Dev B — rect-V2 RCR Stage A2: SHIP correctness (A2c) / NO-SHIP perf (A2d)
============================================================================

Author: Claude (Opus 4) acting as R33 Dev B
GPU: AMD MI355X (gfx950), 4 GPUs available
Branch: `r33-b` (worktree `/tmp/wt-r33-b`), base `feat/mxfp8-only @ 2945ed91`

TL;DR
-----
- **Stage A2a/A2b/A2c (correctness): SHIP.** rect-V2 RCR fastpath at 4096×4096×4096
  achieves **SNR 49.61 dB** (gate ≥48 dB), **pass_rate 100.00%**, **DETERMINISM 3/3 PASS**.
  Output bit-equal to square V2-RCR for `C[0,:8]`.
- **Stage A2d (perf): NO SHIP.** Cross-GPU BABA bench at 4096³ shows rect is
  **−35.6%** slower on GPU1 (Welch t = −26.4) and **−32.6%** slower on GPU2
  (Welch t = −95.1). Both GPUs CONFIRM perf REGRESSION; the structural argument
  (4× tiles → +25% from CU wave-fill) is overwhelmed by the lack of inner-loop
  optimizations (PIPELINE_SCALE / KPAIR_LOOP / PHASE_U16_CACHE / REMAP_ONCE)
  that the square kernel uses.

Stage A2 deliverables status
----------------------------
| Stage | Description                                              | Status |
|-------|----------------------------------------------------------|--------|
| A2a   | `preshuffle_v2_b_rect` host fn for V2-RCR (BLK_N=128)    | **DONE** (in `r33b_bench.py`, `r33b_baba_bench.py`) |
| A2b   | V2-RCR helper variant audit / rect adaptation            | **DONE** (kernel B-side load fixed; see below) |
| A2c   | SNR ≥ 48 dB + det 3/3 PASS @ 4096×4096×4096              | **PASS** (49.61 dB, 100% pass_rate, det 3/3) |
| A2d   | Paired BABA bench rect vs square @ 4096³ + cross-GPU triangulation | **FAIL** (rect −33% to −36% slower on 2 GPUs) |

Critical Stage A2b finding — Dev D's B-side load was wrong
----------------------------------------------------------
Dev D's R32 Stage A1 scaffolding designed the B-side scale slab with
`pack_count_b_rect=1` and `slab_bytes_b_rect = 32 * padded_k_blocks`. The
kernel issued a `b64` load (8 bytes per lane) reading two 4-byte halves
({b0p0, b1p0}) — but the slab layout had only 4 bytes per (lane, k_pair)
per pack. With `b_voff = (lane_kblk<<6) | (lane_nonk<<2)` and `b_soff = k_pair<<8`,
adjacent lanes' 8-byte reads OVERLAPPED in memory. The high 4 bytes of each
b64 read corresponded to the next k_pair's data of the next lane, not to
b1p0 as the kernel intended.

**Fix (Stage A2b)**: revert rect's B-side slab geometry to **PC=2** (b0+b1
packs in same slab, identical to square V2-RCR layout). The rect kernel now
uses the same `b_voff = (lane_kblk<<7) | (lane_nonk<<3)`, `b_soff = k_pair<<9`,
`b64` load as the square kernel. The only structural difference is that there
are 2× more B slabs (one per BLK_N=128 ctile vs one per BLK=256 ctile in
square): `num_slabs_b_rect = (N/BLK_N) * WARPS_N = 32*4 = 128` for N=4096
(vs 64 for square).

`slab_bytes_b_rect = 64 * padded_k_blocks` (was 32 in Dev D's scaffold).
For 4096^3 the rect B scale buffer is `128 slabs × 64 × 128 = 1 MB` (vs
`64 × 64 × 128 = 512 KB` for square — 2× total bytes, but each slab is the
SAME size).

Stage A2a — `preshuffle_v2_b_rect` host fn
------------------------------------------
Located in `r33b_bench.py` (lines ~117-152) and `r33b_baba_bench.py`.
Mirrors `preshuffle_v2_b` (square) but with rect dims:
- `blk_n=128` (vs `blk=256`)
- `hb_n=64` (vs `hb=128`)
- `rbn_rect=16` (vs `rbn=32`)
- `warps_n=4` (unchanged)
- `pack_count=2` (unchanged — kept SAME as square so the kernel's b_voff/b_soff
  strides are unchanged)

Critical detail: each warp's RBN_RECT=16 is HALF of a 32-row e8m0 row_group.
The 2 warps that share a 32-row scale row_group (e.g., wn=0 + wn=1 within
half=0) need DIFFERENT slabs that BOTH reference the same source row_group's
data — but each warp's slab stores ONLY ITS 16-row sub-range in the first
16 lane_nonk slots of the 32-row pack. The remaining 16 rows of each pack are
padded with `0x7F` (e8m0 = 1.0 — never read because `lane_nonk ∈ [0,15]`
covers only the 16 valid rows). This duplicates each row_group's data across
2 slabs — total scale memory doubles vs square (acceptable trade-off).

Mapping (matches kernel `rcr_scale_b_base_rect(half) = bc*128 + half*64 + wn*16`):
- `slab(ctile=bc, wn).pack[0]` = e8m0 rows `[bc*128 + wn*16 : bc*128 + wn*16 + 16)` (b0/half=0)
- `slab(ctile=bc, wn).pack[1]` = e8m0 rows `[bc*128 + 64 + wn*16 : bc*128 + 64 + wn*16 + 16)` (b1/half=1)

Stage A2c — correctness PASS
---------------------------
Build: `tk_mxfp8_layouts.cpython-310-x86_64-linux-gnu.so` md5 `434f08cd0f23dabc697a98673bc24cdc`
(rect, MXFP8_RECT_BLK_N=64). Default build md5 `337687a66067c7ff60ba5d7f5d08e7c4`.

Resource report (rect kernel, unchanged from R32 Dev D baseline):
- VGPRs: 137 (vs 246 square) — −44%
- LDS bytes/block: 98,304 (vs 131,072 square) — −25%
- Occupancy waves/SIMD: 2 (LDS-binding, same as square)
- Spills: 0 SGPR / 0 VGPR

3-run determinism cross-check (same input, same script, GPU1):
```
=== run 1 ===
CORRECTNESS mode=rect M=4096 N=4096 K=4096 snr_db=49.61 pass_rate_pct=100.00
DETERMINISM ok=True
=== run 2 ===
CORRECTNESS mode=rect M=4096 N=4096 K=4096 snr_db=49.61 pass_rate_pct=100.00
DETERMINISM ok=True
=== run 3 ===
CORRECTNESS mode=rect M=4096 N=4096 K=4096 snr_db=49.61 pass_rate_pct=100.00
DETERMINISM ok=True
```

Reference output match (8-element slice):
```
C[0, :8]     = [-0.535, 0.543, -0.512, -0.114, -0.047, -0.488, 0.147, -0.379]
C_ref[0, :8] = [-0.536, 0.545, -0.513, -0.114, -0.047, -0.490, 0.148, -0.379]
```
Bit-equal to the square V2-RCR output on the same inputs.

Stage A2d — paired BABA bench: NO SHIP perf
--------------------------------------------
Methodology:
- 30s sustained 16k matmul preheat per process (R31 Dev D rule).
- BABA pattern: alternating square/rect across 3 cycles (6 measurements total).
- Each measurement = 50 warmup iters + 100 timed iters of `gemm_rcr_pq_v2`
  on 4096×4096×4096.
- Two-process orchestration in `r33b_orchestrate.sh`: pre-builds both .so files
  (default + rect), copies the right one in for each cycle, runs 1 measurement
  per cycle. md5 verified per cycle.
- `rocm-smi -d $PHYS_GPU` (R31 Reviewer rule): sclk monitored at 2.2-2.4 GHz
  throughout, no thermal throttling observed.

GPU1 (`HIP_VISIBLE_DEVICES=1`, `PHYS_GPU=1`):
```
SQUARE_LIST: [2333.89, 2249.00, 2348.94] TFLOPS
RECT_LIST:   [1487.52, 1491.32, 1486.47] TFLOPS
SQUARE: median=2333.89  mean=2310.61  sd=53.88
RECT:   median=1487.52  mean=1488.44  sd=2.55
DELTA:  -35.58% (rect vs square)
WELCH_T: -26.40
```

GPU2 (`HIP_VISIBLE_DEVICES=2`, `PHYS_GPU=2`):
```
SQUARE_LIST: [2290.70, 2265.53, 2280.24] TFLOPS
RECT_LIST:   [1541.94, 1533.35, 1533.92] TFLOPS
SQUARE: median=2280.24  mean=2278.82  sd=12.64
RECT:   median=1533.92  mean=1536.40  sd=4.80
DELTA:  -32.58% (rect vs square)
WELCH_T: -95.07
```

Cross-GPU triangulation: BOTH GPUs CONFIRM the rect kernel is significantly
SLOWER than square (−33% to −36%). Welch t-statistics on both GPUs are
strongly negative — the regression is real and not measurement noise.

Per R32 normalization rule (cross-GPU triangulation, no per-GPU discount):
the cross-GPU consistent regression is conclusive. rect-V2 RCR is **NOT a
perf SHIP** at 4096³.

Why the structural prediction failed
-------------------------------------
R32 Dev D's structural argument: 4096³ at BLK=256 produces 256 tiles vs 304
CUs (16% idle). Rect BLK_N=128 → 16×32=512 tiles → CU-bound now ≈ 512/304 =
1.68 waves per CU at occupancy=1, eliminating the 16% gap. Predicted gain:
~25% before rect overhead penalties.

What actually happens — three compounding factors hurt the rect kernel:

1. **Per-tile overhead doubles**: rect halves per-tile work (1 BLK_N=128 ctile
   does 4 quadrant MMAs vs square's 8 for BLK_N=256). The cA/cB/cC/cD epilogue,
   prologue load fences (TK_WAIT_VMCNT), and `s_barrier` count are roughly
   constant per tile. Doubling tile count doubles overhead, undoing the
   wave-fill gain.

2. **Lack of inner-loop optimizations**: per Dev D's R32 scaffolding, rect
   kernel OMITS PIPELINE_SCALE, KPAIR_LOOP, PHASE_U16_CACHE, REMAP_ONCE,
   SCALE_LDS — all of which the square kernel uses. These optimizations save
   a significant fraction of the steady-state cycle count by amortizing scale
   loads, hoisting phase remap, etc.

3. **B scale memory traffic doubles per warp**: rect uses 1 b64 load per
   k_pair per warp (same as square in BYTES per load), but there are 2× more
   slabs because BLK_N halves. Per CU, total bytes fetched from B scale buffer
   is roughly the same, BUT the per-block b_v2_srsrc setup cost (4× scalar
   readfirstlane + slab base computation) is now incurred 2× more often per CU.

4. **LDS-binding occupancy unchanged**: 98 KB rect tile + 2 buffers × 2 halves =
   192 KB which exceeds the 160 KB-per-CU LDS limit, so 1 block/CU remains the
   per-CU cap. The 25% LDS savings does NOT translate to 2 blocks/CU.

The combined effect: rect reaches 65% of square's TFLOPS, exactly the inverse
of the predicted +25% gain. Net = −33% perf at the same shape.

SHIP / NO-SHIP verdict
----------------------
- **Stage A2a (host preshuffle): SHIP** — `preshuffle_v2_b_rect` produces
  numerically correct results.
- **Stage A2b (helper audit + rect kernel B-side fix): SHIP** — kernel updated
  to use SAME b_voff/b_soff/PC=2 layout as square, fixing the original
  scaffolding's b64-vs-PC1 mismatch.
- **Stage A2c (correctness): SHIP** — SNR 49.61 dB, pass_rate 100%, det 3/3
  PASS at 4096³. Output bit-equal to square V2-RCR.
- **Stage A2d (perf): NO SHIP** — cross-GPU triangulation on GPU1/GPU2 CONFIRMS
  rect is 33-36% slower than square at 4096³. Structural wave-fill gain is
  overwhelmed by rect's missing inner-loop optimizations.

Closure recommendation for R34
------------------------------
The rect-V2 RCR fastpath WORKS (correctness PASS) but is structurally slower
than square at the target shape. To reach a perf SHIP, the rect kernel would
need the same inner-loop optimizations as square (PIPELINE_SCALE, KPAIR_LOOP,
PHASE_U16_CACHE, REMAP_ONCE). That work is NOT free — each optimization adds
~50-100 lines of conditionally-compiled code and ~10-30 VGPRs. With rect
already at 137 VGPRs (vs 246 square), adding all optimizations could push
rect to ~200 VGPRs while still capping at occupancy=2 (LDS-bound).

A more promising direction: investigate whether the wave-fill gain at LARGER
shapes (8192³, 16384³) where total_tiles >> num_CUs gives rect a perf edge.
At 4096³, square already has 256 tiles vs 304 CUs (84% wave-fill in 1 wave),
so the gain is bounded by ~16%. At 8192³, square has 1024 tiles → 3.4 waves,
rect would have 2048 tiles → 6.7 waves; the wave-fill GAP IS GONE for both,
so rect cannot gain anything from CU utilization. The 4096³ shape was the
ONLY structural-CU-underutilization target — and the rect's overhead penalty
exceeds the wave-fill payoff there.

Closing this lever: **rect-V2 RCR fastpath at 4096³ is a paradigm closure**.
Correctness path is unblocked (R33 Dev B fix), but perf cannot beat square
without significant additional invasive work that R32 Dev D's "no rewrite
needed" hope did not anticipate. Add to closed levers list: `rect-V2 RCR @ 4096^3
perf inferior to square (33-36% slower, cross-GPU confirmed)`.

Methodology rules followed (R29/R31/R32 closures)
-------------------------------------------------
1. **`rm -f tk_mxfp8_layouts*.so` + per-build md5** (R29 Dev C):
   - Default build md5: `337687a66067c7ff60ba5d7f5d08e7c4`
   - Rect build md5: `434f08cd0f23dabc697a98673bc24cdc`
   - Md5 verified at every cycle's `cp` step in orchestrate.sh.
2. **`rocm-smi -d $PHYS_GPU` not `-d 0`** (R31 Reviewer): GPU1/GPU2 sclk
   tracked via `PHYS_GPU` env var; sclk consistently 2.1-2.4 GHz post-preheat.
3. **30s preheat + BABA pattern** (R31 Dev D): each python invocation does
   30s sustained 16k matmul preheat before measurement; BABA cycle = SQUARE,
   RECT, SQUARE, RECT, SQUARE, RECT (alternating, NOT batched).
4. **Cross-GPU triangulation** (R32 Reviewer): GPU1 + GPU2 BOTH show -33% to
   -36% rect regression. NO per-GPU discount applied (R31 GPU0 rule
   DEPRECATED per R32 update).
5. **No closed levers prototyped** (R33 prompt): all 25 closed levers
   respected. No s_setprio, no sched_barrier, no cachepolicy, no scale LDS,
   no split-K, no occupancy-3, no V2-CRR LDS SB, no V2-RCR PIPELINE_SCALE.

Files added/modified this cycle
-------------------------------
- **MODIFIED** `analysis/fp8_gemm/mi350x/rcr_mxfp8_exact_8wave_rect_fastpath.inc`:
  - Changed B-side scale slab from PC=1 (Dev D) to PC=2 (matches square layout).
  - Updated `slab_bytes_b_rect` from `32 * padded_k_blocks` to `64 * padded_k_blocks`.
  - Updated `b_voff` shifts from `<<6 | <<2` to `<<7 | <<3` (matches square).
  - Updated `b_soff` shift from `<<8` to `<<9` (matches square).
- **NEW** `analysis/fp8_gemm/mi350x/r33b_bench.py`: standalone correctness +
  bench harness with `preshuffle_v2_b_rect` host fn.
- **NEW** `analysis/fp8_gemm/mi350x/r33b_baba_bench.py`: paired BABA bench
  harness (single-mode-per-process, controlled by orchestrator).
- **NEW** `analysis/fp8_gemm/mi350x/r33b_orchestrate.sh`: BABA pattern
  orchestrator with two-build .so swap.
- **NEW** `analysis/fp8_gemm/mi350x/r33b_baba_gpu{1,2}_*.log`: bench output
  logs (12 files: 2 GPUs × 3 cycles × 2 modes).
- **NEW** `analysis/fp8_gemm/mi350x/r33b_baba_gpu{1,2}_md5_{default,rect}.txt`:
  per-build md5 records.
- **NEW** `analysis/fp8_gemm/mi350x/r33b_baba_gpu{1,2}_build_{default,rect}.log`:
  build logs.
- **NEW** `analysis/fp8_gemm/mi350x/r33b_findings.md` (this file).

Time budget actuals
-------------------
- Stage A2a (preshuffle_v2_b_rect host fn): ~1.5h (including Dev D scaffold audit)
- Stage A2b (kernel B-side fix): ~0.5h
- Stage A2c (correctness validation, 3x det): ~0.5h
- Stage A2d (BABA build + 2-GPU bench): ~1.5h
- Documentation + commit: ~1h
- **Total: ~5h** (under 6h budget)

Author identity
---------------
Author: Claude (Opus 4) acting as R33 Dev B
Co-Authored-By: Claude Opus 4 <noreply@anthropic.com>
