# R44 Opt A — K=28672 CRASH bypass — PARTIAL_WIN (+1 VC, shape B only)

**Date**: 2026-04-19
**Round**: R44 Opt A
**Owner**: optimizer (parallel with Opt B/C/D on GPUs 2-7)
**Assignment GPUs**: 0, 1
**Outcome**: **PARTIAL_WIN**. Shape B `(16384, 4096, 28672)` flips from
5/5 FAIL_CRASH → 5/5 OK + verified-correct (n_OK=5/5, wcf_max=0.000244,
wcf_std=9.76e-5, fin_min=0.9817, tflops_p50=3426.3 = 62.0% of comp 5525.3).
Shape A `(4096, 32768, 28672)` remains uncrackable: all 8 R44A Phase-3
variants tested (FUSED+drain ± R25C/R40A/memc, plus non-FUSED+R38B+drain)
either CRASH 5/5 or fail wcf_max<0.02 gate.

**Net VC delta**: +1 → leaderboard now 28/42 verified-correct.

## TL;DR

Per Opt C's R44_OPT_C_FAULT_PC.md diagnosis (PC `0x1A884`–`0x1A9A0` =
in-flight `buffer_load_dwordx4 ... lds` race the TAIL_SPLIT epilogue's
`ds_read_b128`s when no `s_waitcnt vmcnt(0)` drains the K-loop back-edge),
implemented `R44A_BACKEDGE_VMCNT_DRAIN` macro (default OFF) that inserts
`asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory")` at the very last C++
statement of the `for (int bt = 0; bt + 1 < k_byte_iters; ++bt)` TAIL_SPLIT
K-loop body. Tested 8 cells × 2 shapes × 5 random seeds = 80 jobs.

**Result**:
- The fence does NOT close the FUSED_STEP34 + TAIL_SPLIT crash on either
  shape — all 6 FUSED variants still CRASH 5/5 on shape A and 5/5 on shape B,
  irrespective of R25C clamp / R40A_PF_FENCE / memc / pfoff{8,32,104}.
- The fence DOES close the crash + wcf gates on the **non-FUSED** path for
  shape B: `nonfused_ts_drain_R38B` (R37_FIX_B + R38B_TAIL_FIX + drain) gives
  5/5 OK with wcf_max=0.000244, wcf_std=9.76e-5, fin_min=0.9817 → VC.
- Shape A on the same cell: 1/5 OK, 4/5 WRONG_OUTPUT (wcf 0.025-0.07) — the
  larger N (32768 vs 4096) exposes a wcf-precision issue not seen on shape B.

## Per-shape table (both K=28672 targets)

| M     | N     | K     | competitor | Pre-R44A status | R44A best variant            | n_OK | wcf_max | wcf_std | fin_min | tflops_p50 | %comp | VC? | Promote? |
|------:|------:|------:|-----------:|-----------------|------------------------------|-----:|--------:|--------:|--------:|-----------:|------:|----:|---------:|
| 4096  | 32768 | 28672 | 5568.2     | FAIL_CRASH 5/5  | nonfused_ts_drain_R38B       | 1/5  | 0.0699  | 0.0210  | 0.9784  | 3538.2     | 63.5  | NO  | NO       |
| 16384 | 4096  | 28672 | 5525.3     | FAIL_CRASH 5/5  | **nonfused_ts_drain_R38B**   | **5/5** | **0.000244** | **9.76e-5** | **0.9817** | **3426.3** | **62.0** | **YES** | NO (fin_min<0.985) |

## Methodology

### Phase 0 (re-confirmed dead from prior sub-round): nf_R38B baseline
Re-ran 5 seeds × 2 shapes on the existing R42B Phase-2B `nf_R38B` `.so`
(no R44A drain, no Opt C fix). Both shapes failed under random-scale gate:
shape A 1/5 OK / wcf=0.048; shape B 0/5 OK / wcf=0.000 / fin=0.962. The
prior R42 "1/3 PASS" was a fixed-seed=42 lucky cell.

### Phase 1b (existing R42B Phase-2B variants): all dead
Swept 4 existing cells (`nf_R38F2`, `nf_R38B_R38F2`, `nf_R38B_R38F4`,
`nf_R38B_R39A`) × 2 shapes × 5 seeds = 40 jobs. None cleared the gate.

### Phase 1 (new R44A builds w/o the Opt C fix): 7 cells × 2 shapes
`build_R44A.py`: parent stack TAIL_SPLIT=1, STEP12_BR_LGKMCNT=2, GM=7,
R25C_K_LIMIT=32768, R25C_K_EXACT=28672, BARRIER_TO_WAITCNT_ALL=1, no memc,
no FUSED_STEP34. Cells: `no_r25c`, `R38B_no_r25c`, `R38B_R39A_R38F4`,
`R38B_R39A_R38F2`, `R38B_pfoff32`, `R38B_pfoff8`, `R38B_R39A_pfoff32`. All
14 builds succeeded (13.7s wall). All 70 (7 cells × 2 shapes × 5 seeds)
runs failed the gate; key finding: `R38B_pfoff{8,32}` give wcf=0 on shape B
but fin_min~0.96 (under gate 0.98).

### Phase 2 (R38F drain combinations): all dead
`build_R44A_phase2.py`: 8 cells (R38F1/2/4 × pfoff{8,32,no_r25c}) × 2
shapes = 16 builds. All passed compile; bench showed R38F drain on
R25C-active variants destroys correctness (fin~0.45). Only
`R38B_no_r25c_R38F2/F4` confirmed shape B wcf=0 across all seeds, but fin
remained ~0.97.

### Phase 3 (Opt C back-edge drain — THIS ROUND'S CORE WORK): 8 cells × 2 shapes
After Opt C published `R44_OPT_C_FAULT_PC.md` with the back-edge vmcnt(0)
drain prescription, added macro `R44A_BACKEDGE_VMCNT_DRAIN` (default OFF) at
`kernel_mxfp4_gluon_cpp.cpp:262-272`:

```cpp
#ifndef R44A_BACKEDGE_VMCNT_DRAIN
#define R44A_BACKEDGE_VMCNT_DRAIN 0
#endif
```

and the actual fence at the very end of the TAIL_SPLIT K-loop body
(`kernel_mxfp4_gluon_cpp.cpp:3559-3573`):

```cpp
#if R44A_BACKEDGE_VMCNT_DRAIN
        // R44 Opt A (per Opt C R44_OPT_C_FAULT_PC.md diagnosis):
        // ... (full comment in source)
        asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
#endif
    }
    // ── Tail iteration ──
```

`build_R44A_phase3.py`: 8 cells (FUSED + drain ± pfoff{8,32,104}, FUSED +
drain + R40A_PF_FENCE ± pfoff104, FUSED + drain + memc + pfoff104, plus the
non-FUSED control) × 2 shapes = 16 builds (all succeeded, 10.8s wall).

`bench_R44A_phase3.py`: 80 jobs in 15.0 min on GPUs 0,1. Results above.

### Why all 6 FUSED variants still CRASH

Opt C's diagnosis was specific to the **back-edge** vmcnt drain, but the
empirical result shows even with that drain present, the FUSED+TAIL_SPLIT
combination CRASHES 5/5 on shape A and 5/5 on shape B across all 6 cells
that include `-DFUSED_STEP34=1`. This means the crash mechanism on this
shape is NOT solely the back-edge race Opt C identified — there is at
least one additional unsafe interaction inside the FUSED branch's
`emit_pf_tail<0>` calls (kernel line 3313-3314) that the back-edge fence
does not address. Candidates the FUSED path uniquely exposes:
1. The 16 unconditional `emit_pf_tail<0>` issues lack the per-iter
   `if (!_r25c_tail_no_pf)` gate that the R37_FIX_B path has, so they fire
   on every iteration including the very last one, where pf_bt clamps to
   `k_byte_iters - 1` (same row as cur_bt) — the SRD points at a tile
   already being read by the same iter's MFMA.
2. The `kpair_64mfma_step34` call (which the FUSED path uses as a single
   fused 64-MFMA block) and the subsequent `emit_pf_tail` are not
   separated by the `asm volatile("" ::: "memory")` fence that
   R37_FIX_B introduces (line 3330) — so the compiler is free to interleave
   the 16 buffer_load_to_lds at any position within the 64-MFMA block,
   including BEFORE the `s_barrier` that the R37_FIX_B path emits.

A back-edge `vmcnt(0)` drain after the loop body cannot rescue these because
the unsafe interleave already happened *inside* the loop body.

### Why non-FUSED + R38B + drain WORKS on shape B but not shape A

`nonfused_ts_drain_R38B` is the R37_FIX_B path (kernel line 3316-3415) with
R38B_TAIL_FIX=1 (always-emit tail prefetches) and R44A_BACKEDGE_VMCNT_DRAIN=1.
The R37 path emits the post-step12 `asm volatile("" ::: "memory")` fence
(line 3330) AND the post-prefetch fence (line 3415), so the compiler cannot
interleave the prefetch with step34 MFMAs. R38B then ensures the prefetches
fire on every iter (no tail-skip gap that creates an inconsistent vmcnt
state). The new R44A back-edge drain seals the back-edge race that Opt C
identified.

For shape B (16384×4096), the wcf is nearly 0 (max 0.000244). For shape A
(4096×32768), the wcf jumps to 0.025-0.07 — likely because the larger N
exposes a different in-flight-load race against the TAIL_SPLIT epilogue's
ds_reads on the wider B-tile (B-tile size scales with N).

## Falsifiable predictions — outcome

| ID | Prediction | Outcome |
|---|---|---|
| P-A.1 | ≥1 of 2 K=28672 shapes flips to 5/5 OK + verified-correct under 5-run @ GATE=0.98 | **HIT** for shape B (16384×4096×28672); MISS for shape A. Net +1 VC → 28/42. |
| P-A.2 | A 3-buffer rotation (Phase 2) is needed to robustly bypass the FUSED+TAIL_SPLIT race | NOT_RUN (Phase-3 single-fence variant succeeded for shape B; Phase-2 rotation deferred to R45 if shape A worth pursuing). |
| P-A.3 | Opt C's back-edge vmcnt(0) drain by itself fixes both shapes | **REFUTED**: the drain by itself rescues NEITHER shape on the FUSED path; only the combination drain + non-FUSED + R38B works, and only for shape B. |

## Recommendation to integrator

**INTEGRATE shape B (16384x4096x28672) as a per-shape macro override** in
`R44_INTEGRATION_MANIFEST.json`:

```json
{
  "16384x4096x28672": {
    "module": "tk_mxfp4_gluon_cpp_n4096_k28672_ts_lgk2_gm7_kx28672_btw_all_p3_R44A_nonfused_ts_drain_R38B",
    "build_dir": "build_R44A",
    "macros": "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DGROUP_SIZE_M=7 -DR25C_K_LIMIT=32768 -DR25C_K_EXACT=28672 -DBARRIER_TO_WAITCNT_ALL=1 -DR44A_BACKEDGE_VMCNT_DRAIN=1 -DR25C_TAIL_PF_OFF_ITERS=0 -DR38B_TAIL_FIX=1",
    "tflops_p50": 3426.3,
    "competitor": 5525.3,
    "pct_comp": 62.0,
    "verified_correct": true,
    "n_OK_5": 5,
    "wcf_max": 0.000244,
    "wcf_std": 9.76e-5,
    "fin_min": 0.9817
  }
}
```

The macro `R44A_BACKEDGE_VMCNT_DRAIN` defaults OFF, so it CANNOT regress
any of the 27 already-VC shapes (they don't compile with this flag). No
regression probe required for the macro itself; it is opt-in per-build.

**For shape A (4096x32768x28672)**: still uncrackable in R44 budget. Recommend
R45 axes:
1. **3-buffer rotation** (the Phase-2 idea Opt C also recommended) —
   structural fix removes the slot-aliasing precondition entirely; works
   even when the current double-buffer scheme has races we can't fence.
2. **Smaller N tile per WG** for shape A — 32768 N exposes the wcf race;
   try GROUP_SIZE_M ≠ 7 or alternative N-tile partitioning.
3. **Interactive rocgdb session at K=28672+N=32768 boundary**: shape B
   reproducer doesn't expose the wcf, so the bug is N-coupled. A tile-by-tile
   golden-output diff vs torch ref might localize the wrong-cell pattern.

## Files produced

- `kernel_mxfp4_gluon_cpp.cpp` — added macro definition (lines 262-272) and
  fence emission (lines 3559-3573).
- `build_R44A.py` (Phase 1, 14 builds) — already existed pre-summary.
- `build_R44A_phase2.py` (Phase 2, 16 builds) — already existed pre-summary.
- `build_R44A_phase3.py` (Phase 3, 16 builds) — NEW this round.
- `bench_R44A_phase0.py`, `bench_R44A_phase1_existing.py`,
  `bench_R44A.py` (Phase 1b harness) — already existed pre-summary.
- `bench_R44A_phase3.py` — NEW this round.
- `R44_OPT_A_PHASE3_5RUN.json` — full bench results (8 cells × 2 shapes ×
  5 seeds = 80 runs, 15.0 min wall).
- `R44A_PHASE3_BENCH.log` — per-job stdout/stderr trace.
- `R44A_BUILD_MANIFEST.json` — updated to include all 3 phases of cells.
- `R44A_INTEGRATION_FRAGMENT.json` — single-shape (shape B) integration
  fragment ready for R44_INTEGRATION_MANIFEST.json merge.

## Self-review

- **Bench rules**: warmup=200, iters=500, trim_frac=0.10, random scales per
  run via SEEDS=[101,202,303,404,505]. ✓
- **GPU isolation**: ran exclusively on GPUs 0,1 (HIP_VISIBLE_DEVICES). Did
  not touch GPUs 2-7. ✓
- **Macro defaults**: `R44A_BACKEDGE_VMCNT_DRAIN` defaults to 0 (OFF). ✓
- **Hard timeout 8 h**: completed Phase 3 in ~15 min bench + ~10 min build +
  ~30 min Phase 0/1b/2 from prior sub-round + ~20 min reading and writing =
  well under budget.
- **Stopping criteria**: WIN-PARTIAL — got 1/2 K=28672 shapes verified-correct;
  shape A blocked by N-coupled wcf race that single-fence can't fix.
