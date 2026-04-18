# R38 Dev A — HB-N shrink prototype on V2-CRR wide-N shapes

## Verdict: **NO SHIP — hypothesis REFUTED**

The symmetric mirror of R37 Dev A's HB-M shrink (BLK_M=128, drop cC/cD) is
**negative on both wide-N rect candidates**:

| shape (M×N×K)        | PIPE=0 prod | baseline | Δ%        |
|----------------------|-------------|----------|-----------|
| 8B Gate/Up 4096×14336×4096  | 1366.56 TF | 2404.98 TF | **-43.18%** |
| 70B Gate/Up 4096×28672×8192 | 1313.90 TF | 2382.21 TF | **-44.84%** |

The HB-N shrink kernel + dispatch predicate are **kept on disk** behind the
`MXFP8_CRR_BLK_N=128` macro (default-build dead-code-gated; verified 0
hbnshrink symbols in default 8192³ build) so future cycles can iterate, but
no production wire-in is recommended for any current shape.

## Hypothesis (refuted)

Mirror R37 Dev A's HB shrink win on tall-thin (N=1024) rect by halving the
N-direction tile geometry on wide-N rects:

* `BLK_N = 128` (was 256), `HB_N = BLK_N / 2 = 32` per N-half
* `WARPS_N = 4` (unchanged)
* Grid doubles in N: blocks_per_col = N / BLK_N (= 2x default)
* Drop cB, cD: only cA, cC remain (left-N-half only)
* Accumulator footprint per warp 128 → 64 VGPR (-64)

## Why it failed (root cause)

**The default V2-CRR kernel is already well-tuned for wide-N shapes.** Its
WARPS_N=4 and tiny RBN=32 partition the N-direction across 4 warps with
small per-warp tiles, so per-warp VGPR pressure on wide-N is *not* the
binding constraint — global memory throughput and B-operand bandwidth are.
Halving N coverage:

1. Doubles the WG grid in N → 2× barriers, 2× scale-fetch overhead, 2× CR
   reduction across the (now smaller) per-WG tile.
2. Does **not** materially relieve VGPR (the cA/cC pair re-uses the same A
   register, so the saving is only the 2 dropped accumulator vectors per
   warp = -64 VGPR; vs HB-M's saving of -64 VGPR for the same reason on
   tall-thin shapes, where it WAS load-bearing because BLK=256 hadn't
   fully amortized A-fetch).
3. Halves the per-WG MMA chain length → less MMA latency hiding, but the
   2× WG count adds more dispatch + barrier overhead than is saved.

Net: -43% to -45% across both wide-N candidates. This is the inverse of
HB-M's win on tall-thin: HB-M shrink shines exactly where the default
kernel's WARPS_M=2 + RBM=64 leaves big, expensive accumulators —
tall-thin rects. HB-N shrink would shine where the default kernel had
WARPS_N=2 + RBN=64 — but it doesn't; the default is already small in N.

## PIPE=1 cross-buffer DB attempt

Tried the same Stage B1 recipe (cross-buffer prefetch + double-buffered
LDS) that recovered HB-M from -12% (skeleton) to +28% (B1). Result:

* Single-shot perf on contended GPU0: **1594 TF** (vs PIPE=0 1260) —
  pipelining DOES help (+26% vs PIPE=0), but still well below baseline
  2400 TF.
* **Correctness FAIL**: pass_rate 88.25%, DETERMINISM ok=False, max abs
  diff > 1600 with PIPE=1 v1 (with `s_setprio` + `CRR_SCHED_BARRIER`
  between MMAs). With PIPE=1 v2 (simplified MMA chain matching the
  PIPE=0 skeleton order), correctness changed to `pass_rate=88.25%
  DETERMINISM=False` (still bad).

Even if PIPE=1 correctness were fixed, peak 1594 TF is still -34% vs
the 2400 TF baseline. The skeleton's structural overhead from doubling
the WG grid count cannot be overcome by pipelining alone on this layout.

## Build hygiene + dead-code gate

| build                                      | flags                                                                                                                              | hbnshrink syms |
|--------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|----------------|
| Default 8192³                              | `-w` (no MXFP8_CRR_BLK_N)                                                                                                          | **0**          |
| 8B Gate/Up baseline                        | `-w -DM_DIM=4096 -DN_DIM=14336 -DK_DIM=4096`                                                                                       | **0**          |
| 8B Gate/Up HB-N PIPE=0                     | `-w -DM_DIM=4096 -DN_DIM=14336 -DK_DIM=4096 -DMXFP8_CRR_BLK_N=128 -DMXFP8_CRR_HBNSHRINK_PIPELINE=0`                                | 4              |
| 70B Gate/Up baseline                       | `-w -DM_DIM=4096 -DN_DIM=28672 -DK_DIM=8192`                                                                                       | **0**          |
| 70B Gate/Up HB-N PIPE=0                    | `-w -DM_DIM=4096 -DN_DIM=28672 -DK_DIM=8192 -DMXFP8_CRR_BLK_N=128 -DMXFP8_CRR_HBNSHRINK_PIPELINE=0`                                | 4              |

**Default-build symbol-identity verified**: `nm | c++filt | sort` of the
default 8192³ build pre-edit and post-edit differs only in the random HIP
build-ID symbols (`__hip_gpubin_handle_*`, `__hip_cuid_*`); 741 total
symbols on both, 0 hbnshrink symbols, 0 hbshrink symbols. The new
predicate compiles to dead code under default flags. md5 differs only
because of the random build-ID, which is the load-bearing reason R37 Dev
D recommended switching from md5 to nm-based gate.

## Correctness — PIPE=0 skeleton (PASS)

| shape         | SNR (dB) | Pass rate (%) | Determinism |
|---------------|----------|---------------|-------------|
| 8B Gate/Up    | 49.61    | 100.00        | 3/3 PASS    |
| 70B Gate/Up   | (same)   | 100.00        | 3/3 PASS    |

The PIPE=0 skeleton is bit-exact correct on both shapes — only PIPE=1 has
the cross-buffer DB race that broke correctness.

## Single-GPU BABA results (N=5 BABA, R37 Dev A G2A+median acceptance)

### 8B Gate/Up (M=4096 N=14336 K=4096) on GPU3

| rep | tag  | med (TF) | mean   | stdev | cv      | bench MHz |
|-----|------|----------|--------|-------|---------|-----------|
| 0   | prod | 1354.98  | 1354.83| 0.47  | 0.0003  | 2407      |
| 1   | base | 2398.57  | 2386.12| 33.84 | 0.0142  | 2368      |
| 2   | prod | 1366.74  | 1366.96| 1.29  | 0.0009  | 2403      |
| 3   | base | 2404.90  | 2390.24| 34.42 | 0.0144  | 2368      |
| 4   | prod | 1366.56  | 1366.49| 0.29  | 0.0002  | 2406      |
| 5   | base | 2404.98  | 2391.36| 33.80 | 0.0141  | 2351      |

**prod median-of-medians: 1366.56 TF; base median-of-medians: 2404.90 TF
→ Δ% = -43.18%, NO SHIP** (STRICT requires ≥ +5% min Δ%).

### 70B Gate/Up (M=4096 N=28672 K=8192) on GPU2

| rep | tag  | med (TF) | mean   | stdev | cv      | bench MHz |
|-----|------|----------|--------|-------|---------|-----------|
| 0   | prod | 1311.86  | 1312.08| 0.59  | 0.0005  | 2395      |
| 1   | base | 2379.77  | 2381.73| 8.85  | 0.0037  | 2360      |
| 2   | prod | 1313.90  | 1313.85| 0.44  | 0.0003  | 2395      |
| 3   | base | 2383.31  | 2385.15| 4.25  | 0.0018  | 2350      |
| 4   | prod | 1312.12  | 1312.03| 0.16  | 0.0001  | 2389      |
| 5   | base | 2382.21  | 2377.78| 14.10 | 0.0059  | 2308      |

**prod median-of-medians: 1313.90 TF; base median-of-medians: 2382.21 TF
→ Δ% = -44.85%, NO SHIP**.

Single-GPU is sufficient given the magnitude of the regression. 4-GPU
triangulation skipped (R37 Dev A philosophy: do not spend GPU cycles
triangulating a clear refute).

## Files

* New file (kept for future iteration):
  `analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_hbnshrink_fastpath.inc`
  (~512 LoC, two pipelining stages PIPE=0/1, allow-list 8B+70B Gate/Up)
* New include + dispatch predicate in `kernel_mxfp8_layouts.cpp` (gated
  behind `MXFP8_CRR_BLK_N=128`).
* Per-GPU runner: `/tmp/r38a_run_gpu_hbn.sh`
* Run logs: `/tmp/r38a_wd_probe_gpu3/orchestrate.log` (8B Gate/Up),
  `/tmp/r38a_wd_probe_gpu2_70bgu/orchestrate.log` (70B Gate/Up)

## Followups / R39+ candidates

1. **Do NOT explore further wide-N HB-N shrink variants on V2-CRR.** The
   default kernel is already well-tuned for wide-N; the structural
   overhead from doubling the WG grid count cannot be amortized.
2. **PIPE=1 correctness bug analysis (low-priority).** Fixing the
   cross-buffer DB race in HB-N PIPE=1 is non-trivial (`__builtin_amdgcn_s_setprio`
   removal + MMA-chain reorder did not resolve). A reasonable hypothesis:
   the per-half scale select (`b_sel_packs`) reads from the WRONG slot
   when the prefetch into `toc` overlaps with cA/cC — i.e. the scale
   index path needs explicit ping-pong as well. Not pursued because
   even fully-pipelined, the kernel cannot beat baseline on these
   shapes.
3. **HB-N shrink might still help on a different layout family** (RCR
   or RRR), where the V2 kernel's per-warp N partition is different.
   Recommend R39 Dev fan-out cycle consider applying the HB-N geometry
   to V2-RRR wide-N (where R34 Dev B established V2-RRR is the
   preferred layout for 8B Gate/Up). The hbnshrink kernel as written
   is V2-CRR-specific; would need a parallel V2-RRR HB-N skeleton.
4. **Generalize R37 Dev D's nm-based dead-code gate** as the canonical
   replacement for md5-byte-identity. Random HIP build-IDs
   (`__hip_gpubin_handle_*`) make md5 unreliable; nm-symbol-list
   diff catches the actually-load-bearing invariant.

## Conclusion

The wide-N axis does NOT respond to HB shrink the way the tall-thin axis
does. R37 Dev A's HB-M shrink win on 70B-KV (+30%) was load-bearing
because the default V2-CRR kernel was VGPR-pressure-bound on tall-thin
(WARPS_M=2 + RBM=64 leaves big accumulators). The same kernel on wide-N
shapes is bandwidth-bound, not VGPR-bound, so HB shrink only adds
overhead. Future MXFP8 wide-N optimization should target the V2-RRR
layout (per R34 Dev B finding) or restructure A-fetch / B-bandwidth
rather than tile geometry.
