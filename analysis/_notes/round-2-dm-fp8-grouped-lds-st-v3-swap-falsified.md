# Round 2-dm — FP8 grouped LDS layout `ST_v2 → ST_v3` swap FALSIFIED (correctness)

**Date**: 2026-05-01 (Round 2 of 100 in `dm` run)
**HK HEAD on entry**: `3c679aa9` (round-1-dm docs)
**HK HEAD after revert**: same (no net code change this round)
**Primus-Turbo HEAD**: `08b307a` (round-1-dm docs; unchanged this round)

## TL;DR

Probed the **only** unused `st_16x128` LDS variant — `st_16x128_v3_s`
(row-swizzle `(r & 15) << 3`, `subtile_padding = 0`) — as a drop-in
replacement for the current `st_16x128_v2_s` (subtile swizzle
`((offset >> 7) & 7) << 4`, `subtile_padding = 128`) in
`grouped_rcr_kernel`. Task-body lever **B** ("LDS bank conflicts on FP8
K_BLOCK=128 LDS. ds_read_b128 layout, swizzle stride, padding").

**Result**: FP8 grouped forward SNR collapses on all 16 metric shapes
(`-0.6 to -0.7` vs threshold `+25`). Score drops `812 → 8`. Reverted in
place; metric restored to `812`.

**Root cause** (deduced from revert-behavior, not disassembled): the
`load(reg, subtile_inplace<RBM, BK>(tile, {wi, 0}))` + `rcr_mma` chain
in the main loop uses the subtile's hand-computed LDS-byte offsets
(via `ST::subtile_padding` + linear layout), but the `ds_read_b64_tr_b8`
cooperative loads assume **the specific address-XOR granularity of
st_16x128_v2's swizzle** (`(r & 7) << 4`, 8-row × 16-byte rotation).
`st_16x128_v3`'s `(r & 15) << 3` (16-row × 8-byte rotation) produces
a different bit-scrambled layout that the ds_read lane-to-register
mapping does not invert. The `prefill_swizzled_offsets` HBM-voffset
generation is parameterized by `ST::swizzle<T>()` and does update
correctly — but the consuming ds_read side has hard-coded 16-byte stride
assumptions, so the HBM→LDS→register round-trip no longer recovers the
logical `(r, c)` operand.

This is the same class of wiring dependency that round-11 flagged in
the `prefill_swizzled_offsets_partial_K` → path-A K-tail fuse
(deleted round-13 after the sentinel-tagging math had to be hand-coded
for ST_v2's swizzle exactly). ST_v3 would require re-deriving both
prefill AND ds_read/rt-register mapping.

## What was done

Single-line edit at line 1974 of `kernel_fp8_layouts.cpp`:

```cpp
-    using ST_rcr = ST_v2;
+    using ST_rcr = st_fp8e4m3<HB, BK, st_16x128_v3_s>;
```

Build succeeded with slight resource improvements (due to
`subtile_padding=0` saving 8 KB LDS/tile):

| metric | ST_v2 (baseline) | ST_v3 |
|---|---|---|
| VGPRs | 256 | 256 |
| VGPR Spill | 67 | 62 (-5) |
| LDS bytes/block | 139796 | 131604 (-8192, -5.9%) |
| Occupancy | 2 waves/SIMD | 2 waves/SIMD |

Would have been a nice win for register pressure if correctness held.

## Metric (with ST_v3 swapped in)

```
grpFP8 (all 16 shapes)  ratio = 0.0000  [fwd-snr < -0.6 or -0.7]
grpBF16 (8 shapes)      ratio = 1.12-1.25 [unchanged; kernel untouched]
score = 8 (floor from 16/16 correctness FAIL)
```

Specifically: HK forward returns values whose SNR against fp32-ref
is `-0.6 to -0.7 dB` — i.e. the output is anti-correlated noise,
not a small numerical drift. Definitive swizzle mismatch in ds_read.

## Revert verification

After reverting `ST_rcr = ST_v2` and rebuilding:

```
grpFP8 segment geomean = 0.9746
score = 812 (restored to round-2 baseline)
```

No net code change from this round; revert is a pure rebuild.

## Why this was worth probing

Round-25 `saturated-knobs inventory` does NOT include LDS-layout swap
as a tested lever — all 10+ single-knob tests in rounds 4-25 were
VMCNT / LGKM / sched_barrier / group_m / num_xcds / chunk_size /
`TWO_TILE_MIN_KI`. ST-type swap was never probed because the FP8
kernel was initially bootstrapped with ST_v2 hand-picked for dense
perf and never revisited when the grouped codepath was added.

This falsification adds a new entry:

| knob | direction tested | result | note |
|---|---|---|---|
| `ST_rcr` LDS layout (v2 → v3) | one-liner swap | **CORRECTNESS FAIL** | ds_read hard-codes v2 swizzle; swap requires full prefill + ds_read co-port |

## What a real ST-layout swap would require

If the FP8 team *wanted* to test ST_v3 (or any non-ST_v2 variant), the
following would all need to be updated together:

1. `ST_rcr` / `ST_row` type alias — 2-5 lines touched.
2. `prefill_swizzled_offsets` — already ST-parameterized, no change
   (verified by round-4 + round-11 docs).
3. **`ds_read_b64_tr_b8` address computation in `load<reg, ST>` +
   `subtile_inplace<RBM, BK>`** — this is the **hard** piece. The
   cooperative 64-lane B64 stride formula at the call site is derived
   assuming `st_16x128_v2`'s 8-row × 16-byte XOR granularity; ST_v3's
   16-row × 8-byte granularity would need a new lane-to-(r,c) mapping
   table in `global_to_register.cuh::load<>`.
4. `rcr_mma` register-tile operand layout — may be unaffected if the
   `rt_16x16_s<RBM,RBN>` operand is position-correct, but worth
   auditing the K-axis reduction order (fp8e4m3 mfma's K=128 inner
   layout may also assume a specific lane-to-col mapping).

**Estimated cost**: 2-4 rounds of careful disassembly + numerical
probe, with a high risk of hitting the round-27 VGPR-live-range cliff
when the new layout cascades through register allocation. Not
recommended as a round-3 target.

## Next-round orientation

All exposed single-knob levers now formally falsified (rounds 4, 5, 8,
14-16, 22-25, 27, 2-dm). Round-25 docs still call out the three
multi-round structural projects as the only path forward:

1. **BF16 BK=64 → BK=32 + ns=3** — out of scope per task body (BF16 is
   `[watch]`, score doesn't move).
2. **FP8 MFMA cell-shape 16x16x128 → 32x32x64** — 2-3 rounds; partially
   falsified by round-12 rocprof (same MMA count at different K width).
3. **K-tail epilog single-load merge / K-tail amortize across
   multi-tile-M** — 2 rounds; related to round-27 `load_a_kt` falsified,
   but the `amortize` variant (shared epilog across M-slab) is
   untouched.

Plus task-body lever **E** (direct HBM→reg main loop, skip LDS
cross-warp sharing) — 4-8 rounds; the most disruptive but highest
documented yield (+5-7pp per round-4 §9.1.c).

Round-3 suggestion: pick one of (3) or (E) and commit to multi-round
delivery with WIP-commits at each intermediate step. Do NOT continue
single-knob probing — nothing left.

## Files touched this round

- `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp:1974` — ST_v3 probe
  + revert (net no change).
- `analysis/_notes/round-2-dm-fp8-grouped-lds-st-v3-swap-falsified.md`
  (this file).

## Commit

```
docs(round-2-dm): FP8 grouped LDS ST_v2 → ST_v3 swap falsified (SNR fail, ds_read hard-codes v2 swizzle)
```
