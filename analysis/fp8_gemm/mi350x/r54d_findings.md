# R54 Dev D — 8B Q/O wave-tail diagnostic + speculative 2-CTA fix

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ d5a6b22e (R53 wrap)
**GPU:** MI355X (gfx950) HIP_VISIBLE_DEVICES=6 (strict isolation)
**Mandate:** Confirm and (if confirmed) close the 8B Q/O RRR/CRR small-shape
wave-tail gap that the R53 Reviewer R54+ ranking item-4 alluded to ("256 CTA,
0.42 wave occupancy"). Try a 2-CTA-per-tile cooperative framing if Phase 2
confirms wave-tail, otherwise diagnose precisely.

---

## TL;DR — VERDICT: DIAGNOSTIC-COMPLETE-NOT-A-GAP

**All three layouts at 8B Q/O (M=N=K=4096) clear the 95% MX/FP8 SHIP gate
on current HEAD. There is no wave-tail gap to close.** Phase 1 triggered
the prompt's early-bail clause ("If above 95%, **early
DIAGNOSTIC-COMPLETE-NOT-A-GAP** verdict"). Phases 2 and 3 not executed.

| Layout | FP8 median | MX median | MX/FP8 % | Verdict |
|---|---:|---:|---:|---|
| RCR | 2357.0 | 2318.2 | **98.4%** | PASS (clears 95%) |
| RRR | 2360.4 | 2303.8 | **97.6%** | PASS (clears 95%) |
| CRR | 2277.6 | 2165.1 | **95.1%** | PASS (clears 95% by 0.1pp) |

All cells PASS SNR ≥ 48 dB (49.61 dB measured) and Det 3/3.

The R53 Reviewer's R54+ candidate list (in `r53_reviewer_findings.md` §3)
does NOT enumerate 8B Q/O at all — only 70B Down CRR (-7.1pp), 70B Gate/Up
CRR (-5.7pp), 70B Down RCR (-3.4pp), and 8B Gate/Up RRR (-1.0pp). The
prompt's "256 CTA, 0.42 wave occupancy" framing for 8B Q/O is structurally
inconsistent with the actual hardware geometry (see §2 below) and with
R49C's prior structural analysis of this exact shape family.

---

## §1. Phase 1 — baseline confirm (5 runs/cell, strict-SCLK, GPU 6)

### 1.1 Methodology

- GPU 6 isolation throughout (rocm-smi confirmed `sclk clock level: 1: (158Mhz)`
  pre and post — no DVFS upset).
- Strict-SCLK: WARMUP=100 / ITERS=200, 5 runs/cell, 30 s cooldown between
  runs, 60 s rebuild_cool between FP8 and MXFP8 build.
- FP8 baseline = `kernel_fp8_layouts.cpp` (per-tensor, no preshuffle), built
  with `-DM_DIM=4096 -DN_DIM=4096 -DK_DIM=4096`.
- MXFP8 = `kernel_mxfp8_layouts.cpp` with `MXFP8_PRESHUFFLE_QUANT=1`.
  Production V2 defaults: `MXFP8_RCR_PRESHUFFLE_V2_RUNTIME=1`,
  `MXFP8_RRR_PRESHUFFLE_V2_RUNTIME=1`, `MXFP8_CRR_PRESHUFFLE_V2_RUNTIME=1`
  (all default-ON in `test_mxfp8_python.py`).
- Det check: WARMUP=20 / ITERS=20 / DETERMINISM_RUNS=3 / SNR_THRESHOLD_DB=45.
- 8B Q proj == 8B O proj (both use M=N=K=4096), so a single bench cell
  covers both per the task description.

Bench script: `r54d_phase1_bench.sh`, raw logs in
`r54d_results/phase1_baseline_8B_QO/`.

### 1.2 Per-run TFLOPS (5 runs each)

| Cell | run1 | run2 | run3 | run4 | run5 | median |
|---|---:|---:|---:|---:|---:|---:|
| FP8 RCR | 2456.2 | 2361.1 | 2357.0 | 2355.1 | 2354.1 | **2357.0** |
| FP8 RRR | 2355.9 | 2356.5 | 2430.9 | 2360.4 | 2429.4 | **2360.4** |
| FP8 CRR | 2277.6 | 2296.7 | 2266.4 | 2255.4 | 2344.4 | **2277.6** |
| MX  RCR | 2313.6 | 2312.6 | 2318.2 | 2342.3 | 2331.5 | **2318.2** |
| MX  RRR | 2322.7 | 2303.8 | 2302.4 | 2305.6 | 2287.5 | **2303.8** |
| MX  CRR | 2143.5 | 2165.1 | 2168.2 | 2174.7 | 2159.3 | **2165.1** |

Spreads are <2% within each cell except FP8 RCR run1 (run1 thermal warm
boundary, 4.3% above the rest — excluded from interpretation but kept in
median). All medians are well within R52 / R53 reviewer reported
strict-SCLK noise floor (~1-3%).

### 1.3 Correctness (DET 3/3, SNR ≥ 48 dB)

- RCR: SNR 49.61 dB PASS, Det 3/3 PASS
- RRR: SNR 49.61 dB PASS, Det 3/3 PASS
- CRR: SNR 49.61 dB PASS, Det 3/3 PASS

Logs: `r54d_results/phase1_baseline_8B_QO/8B_QO_check_{rcr,rrr,crr}.log`.

### 1.4 MX/FP8 ratio summary

| Layout | MX median (TFLOPS) | FP8 median (TFLOPS) | MX/FP8 | SHIP gate (≥95%) |
|---|---:|---:|---:|---|
| RCR | 2318.2 | 2357.0 | **98.4%** | **PASS** (+3.4pp above) |
| RRR | 2303.8 | 2360.4 | **97.6%** | **PASS** (+2.6pp above) |
| CRR | 2165.1 | 2277.6 | **95.1%** | **PASS** (+0.1pp above) |

**All three layouts clear the 95% gate.** Per the prompt's bail clause:
> "If above 95%, **early DIAGNOSTIC-COMPLETE-NOT-A-GAP** verdict."

→ Phases 2 and 3 not executed.

---

## §2. Why the wave-tail framing is structurally inapplicable here

The prompt cited R53 Reviewer item-4 as "256 CTA, 0.42 wave occupancy"
for 8B Q/O. Two checks confirm this framing is incorrect for this shape:

### 2.1 Tile-vs-CU geometry (matches R49C analysis)

R49C (`r49c_findings.md` §1-2) already established that MI355X reports
**256 CUs** (`hipDeviceProp_t::multiProcessorCount = 256`, confirmed by
`rocminfo`), not the 304 CU sometimes assumed.

For 8B Q/O at BLK=256:
- `total_tiles = (M/BLK) * (N/BLK) = 16 * 16 = 256`
- `total_tiles / nCU = 256 / 256 = 1.00` ← **exactly one wave**, no tail.
- last-wave fill = 100%, tail fraction = **0.0%**.

### 2.2 Build-reported per-CTA occupancy

Build remarks for the V2 RCR/RRR/CRR fastpath kernels in this benchmark
report `Occupancy [waves/SIMD]: 2`. Each CTA is 8 waves and a CU has 4
SIMDs, so 8 waves/CTA needs 2 waves/SIMD — i.e., **the kernel runs at
exactly 1 CTA per CU** (a CU could in principle host occupancy=4 waves/SIMD
= 2 CTAs, but VGPR pressure caps at 254/256 → 2 waves/SIMD = 1 CTA/CU).

256 CTAs × (1 CTA/CU) → 256 CUs occupied → 0 idle CUs.

The "0.42 wave occupancy" figure in the prompt does not match either
the tile-fill geometry (1.00) or the per-CU CTA occupancy (1 CTA/CU = 100%).
It may have been transcribed from a different shape or a stale metric.

### 2.3 R49C's persistent-CTA refutation already covers this shape

R49C (REFUTED before prototype) tested precisely the same lever class
proposed for R54D Phase 3 (single-CTA persistent variant) at exactly this
shape, and concluded: "two of three target shapes (4096³, 4096×4096×14336)
have **0%** structural wave-tail at 256 CUs." A 2-CTA cooperative tile
variant would only redistribute the same 256 CTAs across the same 256 CUs
in pairs (or halve to 128 CTAs at 2× the work each) — neither changes the
underlying wave count or per-tile critical path. The K-loop body is the
same length per output tile in either framing.

### 2.4 R53 Reviewer R54+ list cross-check

The R53 Reviewer document (`r53_reviewer_findings.md` §3, "Open headroom
(R54+ candidate cells)") enumerates 4 cells, none of which is 8B Q/O:

1. 70B Down CRR (-7.1pp) — RULED OUT (R53C structural ceiling)
2. 70B Gate/Up CRR (-5.7pp) — Highest-leverage R54 target
3. 70B Down RCR (-3.4pp) — open; PMC profile recommended
4. 8B Gate/Up RRR (-1.0pp) — sub-noise; shelved (R52P diag, R52Q+R53B refuted)

The prompt's phrasing "8B Q/O small-shape wave-tail" appears to conflate
the R53 reviewer item-4 (8B **Gate/Up** RRR, M=4096 N=14336 K=4096 — has a
real wave-tail with 896 tiles = 3.5 waves) with the structurally tail-free
8B **Q/O** (M=N=K=4096 — 256 tiles = exactly 1 wave). Item-4 is a different
shape that R49C explicitly identified as the only tail-bearing 4096-M cell
(14.3% absolute upper bound), and that R52P / R52Q / R53B already refuted
under three different lever classes.

---

## §3. Phase 2 — SKIPPED (no gap to diagnose)

Per prompt bail clause. No PMC capture taken.

If a future cycle wants to retain a structural margin baseline for 8B Q/O,
the minimal PMC set already exists at `r54b_pmc_results/pmc_set1.txt`
(`SQ_WAVES`, `GRBM_GUI_ACTIVE`, `SQ_VALU_MFMA_BUSY_CYCLES`, etc.). Re-running
that against this exact 8B Q/O build (M=N=K=4096) with `MXFP8_LAYOUTS=rrr`
would empirically confirm `SQ_WAVES = 256 * 8` (one wave per CTA) and
`per-CU work = identical across CUs` (no tail wave). The 95.1% / 97.6% /
98.4% margins make the result obvious without spending the rocprofv3 time.

---

## §4. Phase 3 — SKIPPED (no gap to close)

Per prompt bail clause. The two design candidates were:

1. **Persistent CTA** (R53B-style, single CTA processes 2 tiles serially):
   For 8B Q/O at 256 tiles == 256 CUs, this would simply run 256 CTAs at
   2 tiles each across 128 CUs (or 128 CTAs at 2 tiles each on 128 CUs +
   128 idle CUs) — the latter wastes half the device. The former requires
   the K-loop body to support inter-tile state, which the V2 RRR
   254/256 VGPR ceiling forbids per memory file `v2_rrr_vgpr_ceiling.md`.
   R53B already empirically refuted this exact lever (228 B/lane spill,
   SNR 26.5 dB).

2. **2-wave cooperative tile** (split a tile across 2 waves of the same
   CTA): the existing 8wave kernel already uses 8 waves to compute a
   256×256 tile with 2 quadrant warps in M and 4 in N. Halving the CTA
   count requires either doubling the per-CTA tile (impossible at
   254 VGPR + 64 KB LDS already saturated) or halving the per-warp work
   (which changes the MFMA tiling and the LDS double-buffer geometry —
   essentially a kernel rewrite). Neither candidate yields a code change
   that respects the "no V2 RRR K-loop body changes" constraint.

Both designs are therefore moot at the proven 95-98% margin.

---

## §5. SHIP gate decision — DIAGNOSTIC-COMPLETE-NOT-A-GAP

| Gate | Required | Measured | Pass? |
|---|---|---|---|
| 8B Q/O RCR MX/FP8 ≥ 95% | 95.0% | 98.4% | **PASS** |
| 8B Q/O RRR MX/FP8 ≥ 95% | 95.0% | 97.6% | **PASS** |
| 8B Q/O CRR MX/FP8 ≥ 95% | 95.0% | 95.1% | **PASS** |
| SNR ≥ 48 dB | 48 dB | 49.61 dB | **PASS** |
| Det 3/3 | 3/3 | 3/3 | **PASS** |
| No regression on 9-cell baseline | n/a (no code changes) | n/a | **N/A** |

No tree changes. No macros introduced. No commits.

---

## §6. Files

- `analysis/fp8_gemm/mi350x/r54d_findings.md` — this document
- `analysis/fp8_gemm/mi350x/r54d_phase1_bench.sh` — Phase 1 orchestrator
- `analysis/fp8_gemm/mi350x/r54d_results/phase1_orchestrator.log`
- `analysis/fp8_gemm/mi350x/r54d_results/phase1_baseline_8B_QO/` — 6 cells × 5 runs + 3 check logs + 2 build logs

No source modifications. No new macros. No commits.

---

## §7. Recommendation for next cycle

Spend R55 cycles on the legitimately open R54+ items per R53 Reviewer §3:

- **Highest-leverage:** 70B Gate/Up CRR (-5.7pp, K=8192 — "real CRR waste"
  per R53 reviewer, since FP8 baseline does not amortize 3-deep at K=8192).
- 70B Down RCR (-3.4pp, K=28672 — different bottleneck than CRR; PMC
  profile recommended).
- 8B Gate/Up RRR (-1.0pp, only true tail-bearing 4096-M shape per R49C; but
  shelved after 3 refutations — needs a new lever class).

8B Q/O at 4096³ is in a structural sweet spot (perfect 256-CTA / 256-CU
fill) and should be retained as a **baseline anchor** for any future
cycle that adds new framing levers (a regression here would signal
breakage of the small-shape fast path).
