# R33 Dev A — findings

Worktree: `/tmp/wt-r33-a` (branch `r33-a`, base `feat/mxfp8-only` @ 2945ed91)
GPU: AMD MI355X (gfx950)
Methodology: rm -f tk_mxfp8_layouts*.so per build + md5 log; rocm-smi -d $PHYS_GPU
sclk gates; 30s preheat + BABA paired bench (n_pairs=5 → n=10/kernel);
cross-GPU triangulation on GPU0 + GPU4 per R32 Reviewer rule (R31 GPU0 discount
DEPRECATED).

## TL;DR

| Task | Status | Cross-GPU triangulation |
|------|--------|--------------------------|
| 1 — V2-RRR autotune for 70B Down (M=4096, N=8192, K=28672) | **SHIP** | GPU0 +11.539% t=+56.63; GPU4 +12.197% t=+61.41 |
| 2 — rect-V2 CRR Stage A2 Path 1 (square LDS HB=128, halve N-work) | **Numerics PASS, perf NO SHIP** | snr=49.60 dB / det 3/3 both GPUs; perf -8.518% / -8.428% |

## Task 1 — V2-RRR autotune for 70B Down

R32 Dev C SHIP claim verified independently and cross-GPU triangulated:
V2-RRR is **+11.5% to +12.2% faster** than V2-CRR on the 70B Down shape
(M=4096, N=8192, K=28672), with Welch t > 50 on both GPUs and identical
correctness (snr_db=49.60–49.61, pass=100%, det 3/3) on both layouts.

Wired host-side autotune entry at `kernel_mxfp8_layouts.cpp` lines 5522–5532
inside `dispatch_pq_v2<CRR>`:

```cpp
if (g.m == 4096 && g.n == 8192 && g.k == 28672) {
    static int warned = 0;
    if (!warned) {
        std::fprintf(stderr,
            "[tk_mxfp8_layouts] gemm_crr_pq_v2: shape (M=4096, N=8192, "
            "K=28672) is +12.14%% faster on V2-RRR (R32 Dev C SHIP, "
            "cross-GPU triangulated). Prefer gemm_rrr_pq_v2 with A "
            "row-major (M,K). See analysis/fp8_gemm/mi350x/r32c_findings.md.\n");
        warned = 1;
    }
}
```

**Why a warning rather than transparent reroute**: V2-CRR expects
A=(K,M) and V2-RRR expects A=(M,K) — incompatible memory layouts.
Transparent rerouting would require a transpose copy that erodes the +12% gain.
The autotune entry is therefore a documented host-side advisory: any caller
that lands on the 70B Down shape via `gemm_crr_pq_v2` is told once per process
that the same math is +12% faster via `gemm_rrr_pq_v2` with A row-major.

**Cross-GPU triangulation results** (`r33a_task1_C4_gpu{0,4}.txt`):

GPU0 (sclk 2310–2378 MHz throughout bench):
- CRR median=2559.34 mean=2556.18 stdev=11.08 n=10
- RRR median=2854.66 mean=2859.56 stdev=12.81 n=10
- Welch t = +56.63, **Δ median = +11.539%**

GPU4 (sclk 2317–2388 MHz throughout bench):
- CRR median=2517.50 mean=2512.94 stdev=13.79 n=10
- RRR median=2824.56 mean=2825.61 stdev=8.32 n=10
- Welch t = +61.41, **Δ median = +12.197%**

SHIP gate (≥+10% Δ, t>10, correctness PASS, ≥2 GPUs): all clear.

**Verdict: SHIP** — autotune entry merged in `dispatch_pq_v2<CRR>`; one-time
host warning fires when the shape lands on V2-CRR.

## Task 2 — rect-V2 CRR Stage A2 Path 1

Implementation (`crr_mxfp8_exact_8wave_rect_fastpath.inc`, complete rewrite,
~330 lines): keep `ST_v2 = st_fp8e4m3<HB=128, BK=128>` (square LDS), drop
right-half operand tile `b1` and right-half accumulators `cB`/`cD`, only
process `b0` + `cA` + `cC`. Block now produces M=BLK_M=256 rows × N=HB_N=128
cols; grid is doubled in N to `(M/BLK)*(N/HB)`. Block index in N becomes
`bc_orig = bc >> 1`, `nhalf = bc & 1`; runtime `b_sel_packs = (nhalf == 0)
? b0_scale_packs : b1_scale_packs` selects the right scale pack on each
K-iter. Host preshuffle layout (slab_idx_b = bc_orig * WARPS_N + wn) is
unchanged from square — Path 1 is purely a runtime-side restructure.

Build: 169 VGPRs, 104,448 B LDS, 0 spills, occupancy=2.

### Stage A2c — correctness (`r33a_task2_correctness.log`)

```
CORRECTNESS_SQUARE         snr_db=49.60 pass_rate_pct=100.00
CORRECTNESS_RECT_PATH1     snr_db=49.60 pass_rate_pct=100.00
DETERMINISM_RECT_PATH1     ok=True
STAGE_A2c_VERDICT          PASS
```

`C_rt[0,:8]` matches `C_sq[0,:8]` bit-exactly. SNR matches the square
reference to two decimal places.

### Stage A2d — perf (BABA paired bench, n=10/kernel, GPU0 + GPU4)

GPU0 (`r33a_task2_paired_bench_gpu0.log`, sclk 2378–2392 MHz):
- square median=788.39 mean=787.81 stdev=5.58 n=10
- rect_path1 median=721.23 mean=721.19 stdev=2.88 n=10
- Welch t = -33.536, **Δ median = -8.518%**

GPU4 (`r33a_task2_paired_bench_gpu4.log`, sclk 2320–2388 MHz):
- square median=794.63 mean=792.49 stdev=6.75 n=10
- rect_path1 median=727.66 mean=726.99 stdev=1.85 n=10
- Welch t = -29.578, **Δ median = -8.428%**

Both GPUs agree: Path 1 is **-8.4% to -8.5% slower than square at 70B KV**
(4096×1024×8192). Cross-GPU consistency rules out a single-GPU sclk artifact.

**Likely sources of overhead** (not addressed under closed-levers list):
1. Runtime `nhalf` branch selecting `b_sel_packs` per K-iter (square has the
   selection statically determined).
2. Reduced MMA interleaving — Path 1 has 2 MMAs/block (cA, cC) vs square's
   4 (cA, cB, cC, cD). Square hides scale loads behind 4 chained MMAs;
   Path 1 has only 2 to hide behind.
3. Doubled grid in N halves work-per-block, increasing scheduling overhead
   relative to square's larger blocks.

**Verdict**:
- Stage A2c (numerics): **PASS** — Path 1 is architecturally viable.
  Square HB=128 + halved-N strategy produces bit-exact correct output.
- Stage A2d (perf SHIP gate ≥+10%): **NO SHIP** — Path 1 is -8.5% slower
  than square baseline at the 70B KV target shape on both GPUs.

## Files / artifacts

- `kernel_mxfp8_layouts.cpp` — Task 1 autotune entry (lines 5522–5532).
- `crr_mxfp8_exact_8wave_rect_fastpath.inc` — Task 2 Path 1 rewrite.
- `r33a_task1_orchestrate.sh` — Task 1 cross-GPU bench driver.
- `r33a_task2_correctness.py` — Task 2 Stage A2c correctness harness.
- `r33a_task2_paired_bench.py` — Task 2 Stage A2d BABA paired bench harness.
- `r33a_task1_C4_gpu0.txt`, `r33a_task1_C4_gpu4.txt` — Task 1 raw results.
- `r33a_task2_correctness.log` — Task 2 correctness raw output.
- `r33a_task2_paired_bench_gpu0.log`, `r33a_task2_paired_bench_gpu4.log` —
  Task 2 perf raw output.
- `r33a_build_task1_crr.log`, `r33a_build_task1_rrr.log` — Task 1 build logs
  (md5s captured).

## Recommendation for next cycle

- **Task 1**: Promote V2-RRR as the recommended layout for the 70B Down
  shape. Consider extending the autotune entry to other RRR-dominant shapes
  if R32+ data supports it.
- **Task 2**: Path 1 is correct but slower. If rect-V2 CRR is still desired,
  next attempts should explore Path 2 (rect LDS HB=64 with K_HALF=0-only
  helper), or revisit the K_HALF=1 helper to remove the OOB indexing that
  was the original Path 0 blocker. Path 1's -8.5% gap is large enough that
  a Path 2 attempt is justified before declaring rect-V2 CRR infeasible.
