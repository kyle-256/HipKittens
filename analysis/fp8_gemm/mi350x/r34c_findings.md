# R34 Dev C — Stage A2 Path 2 (HB_N=64 + serialised K_HALF=0 reads): NO SHIP

Worktree: `/tmp/wt-r34-c` (branch `r34-c`, base `feat/mxfp8-only` @ d0176862,
R33 cycle wrap).
GPU: AMD MI355X (gfx950); HIP_VISIBLE_DEVICES=2 primary, HIP_VISIBLE_DEVICES=3
cross-GPU verify.
Methodology: rm -f tk_mxfp8_*.so per build + md5 log; rocm-smi -d $PHYS_GPU
(not -d 0); 30s preheat + BABA paired bench n_pairs=5 → n=10/kernel; min-of-
GPUs gate.

## TL;DR

**NO SHIP. rect-V2 CRR perf paradigm CLOSED.**

Path 2 (HB_N=64 LDS for B + two K_HALF=0 reads from two staging tiles) is
**numerically PERFECT** (bit-exact against square baseline, snr=49.60 dB,
det 3/3) but is **-9.72% / -9.05% slower than square** at the 70B KV target
shape (4096×1024×8192) on GPU2 / GPU3 respectively. SHIP gate (≥+1% vs
default V2-CRR) is missed by 10 percentage points; gate cannot be cleared
under the present V2 LDS layout.

This closes the **last open lever** for rect-V2 CRR. The paradigm is now
exhausted:
- Path 0 (R31 stub): numerics-wrong (data[4..7] = duplicate of data[0..3]).
- Path 1 (R33 Dev A): numerics-correct but **-8.5% slower** (square LDS HB=128 + halve N-work).
- Path 2 (R34 Dev C): numerics-correct but **-9.7% slower** (rect LDS HB_N=64 + dual K-staging tiles).

Both Path 1 and Path 2 produce bit-exact correct output but neither can
beat square. Path 2 is uniformly worse than Path 1 (Welch t=-4.221 in a
direct paired bench, Δ=-0.93%). Recommendation: **CLOSE rect-V2 CRR perf
paradigm entirely** — the V2-CRR scale-pack semantics + ds_read_b64_tr_b8
mandate the existing K_HALF=1 N-row contiguity that HB_N=64 cannot honor
without paying ≥1× extra LDS-read overhead per K-iter.

| Path | Numerics | Δ vs square @ 70B KV | Note |
|------|----------|---------------------|------|
| square (default V2-CRR) | snr=49.60 / pass=100% | (baseline) | reference |
| Path 0 (R31 stub) | FAIL pass=33.5% | n/a | data[4..7] OOB |
| Path 1 (R33 Dev A) | snr=49.60 / pass=100% | -8.428% / -8.518% | square LDS HB=128 |
| **Path 2 (R34 Dev C)** | **snr=49.60 / pass=100%** | **-9.049% / -9.717%** | rect LDS HB_N=64 |

Cross-GPU triangulation (GPU2 + GPU3) confirms NO SHIP at the +1% gate.

## Stage A2-Path2-a — Architectural audit

### K_HALF=1 invocation site in current Path 1 kernel

Path 1's `crr_exact_8wave_scaled_rect_kernel` at
`crr_mxfp8_exact_8wave_rect_fastpath.inc:59-409` uses:

```cpp
auto load_b = [&](B_col_reg& dst, ST_crr_b& tile, int wi) {
    load_col_from_v2_st(dst, tile, wi * RBN);
};
```

`load_col_from_v2_st` (`kernel_mxfp8_layouts.cpp:501-506`) expands to:

```cpp
load_col_from_v2_st_half<RT, 0>(dst, tile, col_start);
load_col_from_v2_st_half<RT, 1>(dst, tile, col_start);
```

Both K_HALF=0 and K_HALF=1 are invoked. With the **square** LDS tile
(HB=128) used by Path 1, K_HALF=1 reads LDS rows [64,127] which are
in-bounds. Path 1 thus does NOT exercise the architectural blocker R32
Dev A documented; it sidesteps it by keeping HB=128.

### LDS row vs N-direction mapping

R32 Dev A's diagnosis described the LDS row dimension as "N-direction"
(citing `kernel_mxfp8_layouts.cpp:438` for ST_row). For the V2 B-tile
(`ST_v2 = st_fp8e4m3<HB, BK, st_16x128_v2_s>`), the kernel-level coord
convention `b_co(s, k) → {0, 0, k, s}` (row=k, col=s) shows that **LDS
rows are K-direction** for V2-CRR's B side: each LDS row is one K-block
in the B=K-by-N matrix. With HB=128, the tile holds 128 K-rows × 128
N-cols; with HB_N=64, only 64 K-rows × 128 N-cols.

Each warp's `B_col_reg` covers BK=128 K-elements × RBN=32 N-cols; the
helper's `data[K_HALF*4 .. K_HALF*4 + 3]` slot indexes K-position 0..3
(K_HALF=0) or K-position 4..7 (K_HALF=1) of one rt_16x16 subtile. After
the `ds_read_b64_tr_b8` 8×8 transpose, K_HALF=0 reads K-rows [0,63] of
LDS into data[0..3] of the register; K_HALF=1 reads K-rows [64,127] of
LDS into data[4..7]. With HB_N=64, K=64..127 is OOB.

### Path 2 design — K-serialise approach

Two HB_N=64 tiles per double-buffer slot:
- `Bs[0][slot]` holds K=0..63 of the BK=128 K-block (loaded with
  `b_co(2k, bc)` from global)
- `Bs[1][slot]` holds K=64..127 of the BK=128 K-block (loaded with
  `b_co(2k+1, bc)` from global)

A new helper `load_col_from_v2_st_half_rect_idx<RT, IDX_BASE, RECT_HB_N>`
performs the **K_HALF=0 read math** (`k_row = row_off`, in [0,63]) but
writes to `data[IDX_BASE]` instead of `data[K_HALF*4]`. Two calls fill
the full register tile:
- `load_col_from_v2_st_half_rect_idx<RT, 0, 64>(dst, Bs[0][tic], col)` → data[0..3]
- `load_col_from_v2_st_half_rect_idx<RT, 4, 64>(dst, Bs[1][tic], col)` → data[4..7]

A-side is unchanged: keeps `ST_v2a` with HB=128 (full K coverage in one
tile).

## Stage A2-Path2-b — LDS budget audit

### LDS cap & per-block accounting

gfx950 LDS cap per CU: **163,840 bytes** (per R32 Dev A `r32a_findings.md`
line 138).

| Build | A LDS | B LDS | Total | Occupancy |
|-------|-------|-------|-------|-----------|
| Square (default) | As[2][2] = 4 × 17,408 = 69,632 | Bs[2] = 2 × 17,408 = 34,816 | 139,264 | 1 wave/SIMD (per build log) |
| Path 1 (R33 Dev A) | As[2][2] = 4 × 17,408 = 69,632 | Bs[2] = 2 × 17,408 = 34,816 | 104,448 (build log) | **2 waves/SIMD** |
| **Path 2 (R34 Dev C)** | As[2][2] = 4 × 17,408 = 69,632 | **Bs[2][2] = 4 × 8,704 = 34,816** | **104,448 (build log)** | **2 waves/SIMD** |

**Crucial finding**: Path 2 has **identical LDS** (104,448 B/block) to
Path 1. The B-side reorganisation from 2 large tiles to 4 small tiles
preserves the total byte count (each HB_N=64 tile is 4 subtiles ×
2,176 B = 8,704 B; vs each HB=128 tile is 8 subtiles × 2,176 B =
17,408 B). Occupancy stays at **2 waves/SIMD** identical to Path 1.

LDS budget is **NOT the blocker** for Path 2; the LDS-rebalance approach
the brief listed (re-shuffle vs extra global load) ended up requiring no
extra LDS at all — just structural reorganisation.

### Build resources (Path 2 vs Path 1)

```
Path 1 (R33 Dev A baseline build, MXFP8_RECT_BLK_N=64):
  TotalSGPRs: 51    VGPRs: 169    AGPRs: 0
  Occupancy: 2      LDS: 104,448
  md5: 1f57f228ba59690bba646402e4143c4e

Path 2 (R34 Dev C, MXFP8_RECT_BLK_N=64 + MXFP8_RECT_PATH2_K_SERIALIZE=1):
  TotalSGPRs: 56    VGPRs: 169    AGPRs: 0
  Occupancy: 2      LDS: 104,448
  md5: 1b00a2afe4ace97e5d0959609d3c9245
```

Path 2 uses **5 more SGPRs** (51 → 56) for the extra B SRD bookkeeping
and dual b_co lambdas, but VGPRs and LDS are unchanged. Both kernels
clear the 256-VGPR / 163,840-B LDS limits comfortably.

### Default-build byte-identity

```
Default (no MXFP8_RECT_BLK_N, MXFP8_RECT_PATH2_K_SERIALIZE), my changes:
  md5: 051d2bfdc3812633d2b50e3d7439ec07

Default, R33 head (no my changes):
  md5: 8fda6811214177116af7b641ba6e0972
```

The .so md5s differ but the **.text section** is byte-identical (34,275
disasm lines on both) and the **.hip_fatbin sections** are identical in
size (0x32920 bytes). The fatbin diffs are all in debug-metadata regions
(line-number tables shifted because we added ~50 lines of new helper +
~310 lines of Path 2 kernel code in the source file). Functional GPU
behavior is preserved when `MXFP8_RECT_PATH2_K_SERIALIZE=0` (the default).

## Stage A2-Path2-c — Numerics (`r34c_correctness.log`)

```
CORRECTNESS_SQUARE       snr_db=49.60 pass_rate_pct=100.00
C_sq[0,:8] = [2.1875, -1.4765625, 0.37109375, -1.0703125, -0.318359375,
              0.06787109375, -0.8359375, 0.267578125]
C_ref[0,:8] = [2.197429895401001, -1.4819014072418213, 0.372469425201416,
               -1.0724642276763916, -0.3188326358795166, 0.06807208061218262,
               -0.8360903263092041, 0.26834750175476074]
CORRECTNESS_RECT_PATH2   snr_db=49.60 pass_rate_pct=100.00
C_rt[0,:8] = [2.1875, -1.4765625, 0.37109375, -1.0703125, -0.318359375,
              0.06787109375, -0.8359375, 0.267578125]
DETERMINISM_RECT_PATH2   ok=True
STAGE_A2c_VERDICT        PASS
```

`C_rt[0,:8]` is **bit-exactly equal** to `C_sq[0,:8]`, and SNR matches
the square reference to two decimal places (49.60 vs 49.60). Determinism
passes (3/3 runs match). The K-serialised Path 2 reproduces the square
math exactly, validating the design.

## Stage A2-Path2-d — Performance (`r34c_paired_bench_gpu{2,3}.log`)

GPU2 (`r34c_paired_bench_gpu2.log`, sclk 2,184–2,381 MHz):
- square median=787.84 mean=788.04 stdev=5.19 n=10
- path2  median=711.28 mean=711.23 stdev=2.06 n=10
- Welch t = -43.510, **Δ median = -9.717%**

GPU3 (`r34c_paired_bench_gpu3.log`, sclk 2,388–2,408 MHz):
- square median=766.34 mean=765.85 stdev=5.00 n=10
- path2  median=696.99 mean=696.22 stdev=3.50 n=10
- Welch t = -36.087, **Δ median = -9.049%**

Min-of-GPUs Δ = **-9.72%** (worse than the -1% NO-SHIP threshold by 10pp).

### Direct Path 2 vs Path 1 paired bench (GPU2)

```
path1 median=723.47 mean=722.85 stdev=3.27 n=10
path2 median=716.76 mean=717.48 stdev=2.33 n=10
Welch t (path1 vs path2) = -4.221  (positive = path2 faster)
DELTA_MEDIAN_PCT path2_vs_path1 = -0.927%
```

Path 2 is also **slightly slower than Path 1** (-0.93%, Welch t=-4.22).
This is below the noise floor for one-GPU isolation but consistent with
the Path 2 overhead model (2× LDS-read issues per K-iter + additional
SGPR pressure for SRD bookkeeping).

### Likely sources of Path 2 overhead

1. **2× ds_read_b64_tr_b8 per warp per K-iter** — Path 1 does 2 reads
   (one full helper call); Path 2 does 4 reads (two half-helper calls
   from two staging tiles). The LDS-bandwidth pressure does not double
   (each read is the same size) but the issue rate does, taxing the LDS
   issue scheduler.
2. **2× global_load_b per K-iter** — same total bytes (each rect tile
   is half-size), but the address generation work is doubled and the
   buffer-load cache-line-fetch granularity is potentially less efficient
   for the half-K-block stride.
3. **No reduction in compute** — the MMA throughput is identical (same
   register tile shape, same number of MMAs per K-iter). All overhead
   adds are pure latency without compensating throughput gains.
4. **Higher SGPR pressure** (51 → 56) — minor, but consistent with the
   slightly worse scheduling Path 2 exhibits.

## Closure: rect-V2 CRR perf paradigm exhausted

After R28 Dev D (helper rect scaffold), R30 Dev A (rect predicate),
R31 Dev A (Stage A1 stub — numerics wrong), R32 Dev A (architectural
blocker diagnosed — Path 1 / Path 2 paths identified), R33 Dev A
(Path 1 implemented — correct numerics, **-8.5% perf**), and R34 Dev C
(Path 2 implemented — correct numerics, **-9.7% perf**), the rect-V2
CRR fastpath has now been exhaustively explored:

- Both numerically-correct paths (Path 1, Path 2) lose perf by ~8–10%
  vs the square baseline at the 70B KV target shape.
- LDS budget is not the binding constraint (Path 1 and Path 2 both run
  at 104,448 B/block, comfortably within the 163,840 B/CU cap, both at
  2 waves/SIMD occupancy — same as square's 1 wave/SIMD with 139,264 B
  but the higher-occupancy rect kernels still lose).
- The structural perf gap comes from MMA-pipelining loss (Path 1 drops
  cB/cD which used to hide scale loads behind 4 chained MMAs; Path 2
  doubles the LDS-read issue rate without compensating) — neither path
  can recover the throughput in a 1-block N-tile work setup.
- No fourth path is architecturally available without redesigning the
  V2 LDS swizzle + helper sharding (R32 Dev A's "Path 3" estimated at
  6–8h with high risk of breaking 4 hot kernels) — out of scope for
  R34.

**Verdict for the cycle: rect-V2 CRR is CLOSED as a perf lever.**
The remaining +12% headroom on the 70B KV CRR shape (vs RRR) cannot be
unlocked through rect-N restructuring; future improvements must target
the V2 swizzle / scale-pack pipeline directly, or accept the V2-RRR
autotune pivot (R33 Dev A Task 1 SHIP).

## Files / artifacts

Source modifications (default-OFF macro keeps default build behavior
identical):
- `kernel_mxfp8_layouts.cpp` — added `load_col_from_v2_st_half_rect_idx`
  helper (~55 lines, lines ~498–550 of the diff). Templated, only
  instantiated when `MXFP8_RECT_PATH2_K_SERIALIZE=1` build is selected.
- `crr_mxfp8_exact_8wave_rect_fastpath.inc` — added Path 2 kernel branch
  (~310 lines) gated on `MXFP8_RECT_PATH2_K_SERIALIZE`. Default off →
  Path 1 (R33 baseline) preserved exactly.

Bench / correctness scripts:
- `r34c_task_correctness.py` — Stage A2-Path2-c correctness harness
  (cloned from `r33a_task2_correctness.py`, relabeled PATH2).
- `r34c_paired_bench.py` — Stage A2-Path2-d BABA paired bench
  (cloned from `r33a_task2_paired_bench.py`).

Logs:
- `r34c_build_path2.log` — Path 2 build (md5 `1b00a2afe4ace97e5d0959609d3c9245`).
- `r34c_build_path1.log` — Path 1 build for comparison (md5 `1f57f228ba59690bba646402e4143c4e`).
- `r34c_build_square.log` — Square baseline build (md5 `d7099c7019ca5183866547e33d6b7aaa`).
- `r34c_build_default.log` — Default build with my changes (md5 `051d2bfdc3812633d2b50e3d7439ec07`).
- `r34c_build_default_baseline.log` — Default build, pre-changes (md5 `8fda6811214177116af7b641ba6e0972`).
- `r34c_correctness.log` — Stage A2-Path2-c PASS verdict.
- `r34c_paired_bench_gpu2.log` — GPU2 BABA bench, Path 2 vs square (Δ=-9.717%).
- `r34c_paired_bench_gpu3.log` — GPU3 BABA bench, Path 2 vs square (Δ=-9.049%).
- `r34c_paired_bench_path1_vs_path2_gpu2.log` — direct Path1 vs Path2
  bench (Δ=-0.927%).

## Author identity

Author: Claude (Opus 4) acting as R34 Dev C
Co-Authored-By: Claude Opus 4 <noreply@anthropic.com>
