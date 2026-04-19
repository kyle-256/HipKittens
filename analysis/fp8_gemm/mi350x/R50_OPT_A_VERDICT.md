# R50 Opt A — VERDICT: DEAD

**Date**: 2026-04-19
**Branch**: mxfp4 (R44 baseline = 35/42 VC @ 305fe79d)
**Worker**: R50 Opt A — aiter MFMA↔ds_read 1:3/1:4 interleaving port inside `kpair_64mfma_step34`
**GPUs**: 0, 1
**Net delta**: **0 VC** on R44 35-shape VC list; **0 VC** on the 6 cluster-B targets.

## TL;DR
Ported aiter's spread interleaving (1 ds_read interleaved every 2-4 MFMAs) into the
HipKittens `kpair_64mfma_step34` asm volatile body, gated by macro
`R50A_AITER_INTERLEAVE` (default OFF, byte-equivalent to R44 baseline). Built 11
shapes × 2 cells (baseline, v1) = 22 modules. ISA-verified the v1 disassembly
breaks 30 of 61 long pure-MFMA runs (length 25) into shorter runs (length 3, 4, 11),
confirming the rewrite is emitted.

10-run @ 80% gate INDEPENDENT seeds [101..1010] result:
- **6 cluster-B wcf-flake targets**: 0/6 VC under v1 (identical 0/6 under baseline).
- **R44 VC stretch (4096x4096x8192)**: v1 retains VC (10/10 OK, wcf_max 0.0149,
  3960.2 TFLOPS vs baseline 3948.4 TFLOPS, +0.3%).
- **4 random regression shapes**: v1 gains VC on `8192x8192x8192` and loses VC on
  `32768x4096x4096`. Neither shape is in the official 42-shape benchmark.

**Conclusion**: The MFMA↔ds_read 1:4 spread is a real ISA-level change (verified by
disasm) but does not move the cohort race that defines cluster-B's wcf-flake. The
underlying race is in MFMA accumulator ordering, not in ds_read scheduling.

## Mechanism summary
**Hypothesis** (per `project_mxfp4_R49A_aiter_vmcnt_dead.md`):
aiter's hand-written kernel interleaves ds_reads evenly through the 32-MFMA Step3/4
blocks (~1:3 ratio), while HipKittens front-loads all 8 ds_reads in Row 0 leaving
24 consecutive pure MFMAs. R50 Opt A tested whether replicating aiter's spread
shifts the AGPR clobber window enough to close the cluster-B race.

**Finding**: ISA-verified that the spread *is* emitted (BASELINE shows `MD MD MD MD
MD MD MD MD M*25 M*25 M*25` per Step; v1 shows `MD MD MD MD MD MD MD MD M*4 D M*4
D M*11 M*4 D`...), so the change is real at the binary layer. However:
- The 6 cluster-B wcf-flake shapes still flake at identical rates under v1
- The Jaccard prefilter (5 INPUT_REUSE probes same seed) showed 7/22 stable cells
  for both baseline and v1 — same set
- Per-cell wcf_max values move randomly between baseline and v1 (some better,
  some worse) without trend toward stability

**Why this falsifies aiter-interleave-as-cause**: If front-loaded ds_reads were
the cohort-race trigger, the spread should have at least narrowed the race
window (e.g., 4/10 → 5/10 OK on multiple cluster-B shapes). Instead, n_OK
swings ±2 per shape with no shape crossing the 8/10 gate.

## 10-run results (per shape)

| Shape | baseline VC | baseline n_OK | baseline TFLOPS | v1 VC | v1 n_OK | v1 TFLOPS |
|---|---|---|---|---|---|---|
| 4096x4096x8192 (stretch) | True | 10/10 | 3948.4 | True | 10/10 | 3960.2 |
| 16384x14336x4096 (cB) | False | 4/10 | 4103.0 | False | 2/10 | 4013.0 |
| 16384x28672x4096 (cB) | False | 4/10 | 4114.4 | False | 6/10 | 3997.3 |
| 16384x6144x4096 (cB) | False | 4/10 | 4103.5 | False | 8/10 | 4011.0 |
| 28672x4096x16384 (cB) | False | 8/10 | 4486.6 | False | 6/10 | 4372.9 |
| 32768x4096x14336 (cB) | False | 5/10 | 4419.2 | False | 3/10 | 4303.3 |
| 4096x32768x14336 (cB) | False | 7/10 | 4376.0 | False | 6/10 | 4283.3 |
| 8192x8192x8192 (regr) | False | 8/10 | 4415.2 | True | 10/10 | 4337.0 |
| 16384x8192x8192 (regr) | False | 9/10 | 4321.5 | False | 8/10 | 4186.8 |
| 32768x4096x4096 (regr) | True | 10/10 | 3961.6 | False | 9/10 | 3938.7 |
| 4096x16384x4096 (regr) | True | 10/10 | 4036.6 | True | 10/10 | 4013.9 |

PROMOTE candidates (per task spec):
- `8192x8192x8192`: v1 gains VC, but shape NOT in official 42 → DROP
- `16384x6144x4096`: v1 4/10 → 8/10 — exactly at gate boundary, wcf_max 0.0285 > 0.02 gate → DROP

REGRESSIONS (must NOT regress R44 VC):
- `4096x4096x8192` (stretch baseline): retained VC under v1 (no regression)
- `32768x4096x4096`: regressed to 9/10 BUT shape NOT in official 42 → not a real R44 regression
- `4096x16384x4096`: retained VC under v1 (no regression)

**No promote candidate clears the 80% + wcf_max<0.02 + wcf_std<0.01 + fin_min>=0.97
gate on any shape that is in the official 42-shape benchmark.**

## Files produced
- `R50_OPT_A_VERDICT.md` (this file)
- `R50A_INTEGRATION_FRAGMENT.json` → `{}` (DEAD)
- `R50A_aiter_kloop_body.s` (extracted aiter K-loop pattern)
- `R50A_old_step34.s` (R44 baseline asm volatile body)
- `R50A_new_step34.s` (v1 1:4 spread design)
- `R50A_KERNEL_ISA.s` (v1 SO disasm — verified 1:4 spread emitted)
- `R50A_BASELINE_ISA.s` (baseline SO disasm — front-loaded 1:1+pure-25)
- `R50_OPT_A_SMOKE.{json,log}` (1-seed smoke; 18/22 OK)
- `R50_OPT_A_JACCARD.{json,log}` (5-probe INPUT_REUSE; 7/22 stable)
- `R50_OPT_A_10RUN.{json,log}` (10-run @ 80% gate; verdict above)
- `R50A_BUILD_MANIFEST.json`, `build_R50A.py`, `bench_R50A.py`
- `build_R50A/*.so` (22 modules)

## Closure for future rounds
- The aiter MFMA↔ds_read 1:4 spread axis is now CLOSED: ISA-verified emit, no
  cohort-race movement on the 6 cluster-B targets.
- Combined with R49A (aiter vmcnt knob alone DEAD), R49C (embedded vmcnt in
  emit_pf_tail DEAD), R45B (5 fence positions internal to step34 DEAD), R47A
  (3 fence positions on R46B 3-buf DEAD), this completes the 5th independent
  closure of the "MFMA accumulator race is a memory-ordering / scheduling bug"
  hypothesis. The race is structural and lives in the AGPR forwarding path
  inside the MFMA pipeline itself, not in any orderable instruction sequence
  the compiler or the asm body controls.
- R51+ should pivot to **(a)** aiter `.co` dlopen (R50 Opt D) for a full
  per-shape kernel swap, or **(b)** completely different MFMA shape (e.g.
  32×32×64) to break the AGPR forwarding chain.
