# MXFP4 Optimization TODO

**Last update:** 2026-04-20 (R62)
**Status:** 11/42 WIN, mean 93.1% of aiter — same as R61 baseline (12/42, 93.2%; 1-shape boundary noise)
**Bench harness:** `analysis/fp8_gemm/mi350x/bench_all_42.py` (HipKittens-only, 6 variants, parallel-GPU)

---

## Hard rules (do not violate)

1. **No aiter binary substitution.** Every reported number must come from a HipKittens kernel built from `kernel_mxfp4_gluon_cpp.cpp`. Past incident: R50–R61 used aiter `.co` to fake 42/42; user is intolerant of repeats.
2. **Bench protocol:** warmup=200, iters=500, trim_frac=0.10, GPU isolation via `HIP_VISIBLE_DEVICES`.
3. **Compete against `competitor_tflops`** (aiter ASM via Python dispatcher) embedded in `bench_all_42.py`.
4. **Commit only when there is measurable effect.** Inert scaffolding stays out of git.

## Current standing

| Cluster | Shapes | Status |
|---|---:|---|
| Easy WINs (M-large, K-small) | 11 | already winning |
| Boundary (~99% to ~102%) | 4 | run-to-run noise |
| LOSE: M=4096 K-large | 8 | -7% to -24% gap (root cause: inner-loop scheduling) |
| LOSE: K-large general | 19 | -10% to -25% gap |

Worst losers (4096×*×K, K≥16384): 67–82% of aiter. These dominate the headline gap.

---

## Productive axes (not yet exhausted)

### A. Inner-loop rewrite (HIGH priority)
- The current `kpair_64mfma_step34` emits 8 ds_reads then 24 pure MFMAs per Step.
- aiter emits an evenly-spread MFMA:ds_read:buffer_load 4:1:1 pattern.
- R50A tried to spread inside the existing asm volatile body but only addressed cohort-race; no perf win.
- **Next try:** rewrite the entire 64-MFMA inner kernel from scratch using aiter's emission template (see `project_mxfp4_aiter_disasm_findings.md`). Do NOT try to splice — the schedule is fundamental, not a knob.

### B. Different MFMA shape
- Current uses 16×16×128. UNTRIED: 32×32×64 with `mfma_scale` cbsz/blgp variants.
- Structurally different AGPR forwarding chain — would either close cohort race or open a new performance regime.

### C. Per-shape tile dimensions
- aiter uses 224×256 / 192×256 for the worst clusters.
- HipKittens is locked to BLK_M=BLK_N=256.
- Adding a 192×256 tile path may unlock 4096×K-large shapes specifically.

### D. K-pair count tuning
- Currently 2 KPairs per K iteration. aiter uses different KPair counts per shape.
- Cheap to try: extend `bench_all_42.py` variants with `-DKPAIRS_PER_ITER={1,2,4}`.

---

## Closed axes (don't reopen)

- **Fence positioning** in/around step34 (R45B, R47A, R49A, R49C, R50A — 5 closures)
- **MFMA↔ds_read 1:3/1:4 interleaving** inside existing `kpair_64mfma_step34` (R50A — ISA-verified emit but no win)
- **STEP34_INTERLEAVED scaffolding** (R62, 2026-04-20): added flag + 4 call-site gates, but `kpair_64mfma_step34_interleaved` function (already in tree at line 1432) does not compile when actually used — emits `ds_read_b128` with operands that resolve to AGPRs, producing 20+ "invalid operand for instruction" errors. Don't enable until that function is rewritten.
- **`UNROLL_K=2/16` per-shape variants** (R62): added to bench but unlock 0 NEW WINs (only re-rank already-winning shapes by ~0.5pp).
- **`asm_inline` "5084 TFLOPS" reference** — REVOKED 2026-04-17, that kernel is numerically incorrect (SNR -1.31 dB).

---

## R63+ priorities

1. **Inner-loop rewrite** (Axis A) — biggest unrealized lever; do this before micro-knobs.
2. **192×256 tile path** (Axis C) — focused at 4096×K-large losers.
3. **MFMA 32×32×64** (Axis B) — exploratory; only if Axis A stalls.

## Bench script (R62 working set)
`bench_all_42.py` variants: `default | gm6 | gb | gb_gm6 | unr2 | unr16`
Run: `BENCH_GPUS=0,1,2,3,4,5,6,7 python3 bench_all_42.py` (~10 min for build+full sweep).
Single-shape autotune: `python3 bench_all_42.py M N K` picks best variant.
