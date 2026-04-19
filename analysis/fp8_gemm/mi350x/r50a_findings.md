# R50 Dev A — CRR scale prefetch lead-distance — REFUTED (analytical bail before bench)

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ 03c94ab3
**GPU:** MI355X (gfx950), HIP_VISIBLE_DEVICES=1 (exclusive — bench not exercised)
**Hypothesis:** Lift the CRR `load_raw_scales(k_pair)` issue from the TOP of
the consuming even iter to the END of the preceding odd iter (after the cD
MMA, the last consumer of the OLD scales). This places ~one full BK of
vmcnt drain time between scale-issue and the cA-MMA dependency edge,
targeting the structural CRR ~92% ceiling (R47D / R49C §7.2). Gated by
`MXFP8_CRR_SCALE_LEAD=1` (default OFF); baseline kept as `#else` arm.

## TL;DR — VERDICT: REFUTED (analytical, pre-bench)

Treatment **introduces 5 VGPR spill + 24 bytes/lane scratch** on every
production CRR shape (V1 SCALE_VERSION=1 and V2 SCALE_VERSION=2 dispatch).
This is the exact bail condition called out in the R50 prompt and in
R49 Dev B's lesson (lifting load issue distance inflates live ranges,
which the compiler resolves by spilling against the 256-VGPR ceiling).
Per protocol, REFUTE before bench.

## Build remarks (CRR PRESHUFFLED_QUANT=1, both V1 and V2)

| Shape              | LEAD=0 (baseline)            | LEAD=1 (treatment)              |
|--------------------|------------------------------|---------------------------------|
| 8192×8192×8192     | V1: 225 VGPR, 0 spill, 0 scratch / V2: 227 VGPR, 0 spill, 0 scratch | V1: 256 VGPR, **5 spill**, **24 scratch** / V2: 256 VGPR, **5 spill**, **24 scratch** |
| 4096×4096×4096     | 225 / 227 VGPR, 0 spill      | 256 VGPR, **5 spill**, **24 scratch** |
| 4096×14336×4096    | 225 / 227 VGPR, 0 spill      | 256 VGPR, **5 spill**, **24 scratch** |
| 4096×4096×14336    | 225 / 227 VGPR, 0 spill      | 256 VGPR, **5 spill**, **24 scratch** |
| 4096×8192×8192     | 225 / 227 VGPR, 0 spill      | 256 VGPR, **5 spill**, **24 scratch** |
| 4096×28672×8192    | 225 / 227 VGPR, 0 spill      | 256 VGPR, **5 spill**, **24 scratch** |
| 4096×8192×28672    | 225 / 227 VGPR, 0 spill      | 256 VGPR, **5 spill**, **24 scratch** |

Pattern is identical across all 7 production CRR cells and across both
SCALE_VERSION dispatches: VGPRs jump by ~30 (from 225/227 to the 256 cap),
clearing the ceiling and forcing a 5-VGPR spill into per-lane scratch
(24 bytes/lane). Occupancy (waves/SIMD) does NOT drop — both stay at 2 —
but the inner K-loop now carries scratch reads/writes against the spilled
slots, which adds latency on every odd iter (where the scale prefetch is
re-issued).

Full per-shape resource dump in `r50a_results/SUMMARY.txt`; raw build
remarks per shape in `r50a_results/build_<shape>_lead{0,1}.log`.

## Diagnosis

The K-loop body in the production CRR exact 8-wave kernel (lines 928–1017
of `crr_mxfp8_exact_8wave_fastpath.inc` after the patch) advances scales
once per K-pair. Baseline issues `load_raw_scales(k>>1)` at the TOP of
each even iter; the cA MMA in the same iter then waits implicitly on
vmcnt before consuming the b128/b64 scale loads. The hypothesis was that
moving the issue to the END of the preceding odd iter (after cD MMA) would
extend the vmcnt drain window by ~one full BK.

The patch achieves the issue-point shift correctly — but in doing so, it
extends the live ranges of the four scale-pack VGPR clusters
(`a0_scale_packs[2]`, `a1_scale_packs[2]`, `b0_scale_packs[1]`,
`b1_scale_packs[1]`, ~6 u32 VGPRs total) across the entire interleaved
cA / cB MMA chain plus the steady-mid barrier and the load_a / global_load
sequence preceding the cC/cD MMAs. The compiler's register allocator
saturates the 256-VGPR cap (up from a clean 225/227) and spills 5 VGPRs
into per-lane scratch, generating implicit `scratch_load`/`scratch_store`
traffic in the inner loop. This is exactly the failure mode flagged in the
R50 prompt:

> If the change causes VGPR spill regression (build remarks show new
> "VGPRs Spill" line), REFUTE before benching — that's R49 Dev B's lesson:
> noinline annotations expose call frames; lifting load issue distance
> can similarly inflate live ranges.

Even though the spill count is small (5 VGPRs vs R49B's call-frame storm),
the inner-loop scratch traffic on every iteration would conservatively
cost ≥1.5–3% of throughput on a kernel already in HEADROOM territory —
swamping the ~5–8% upside the lever was projected to deliver, and almost
certainly forfeiting the ≥+1.5% geomean / no -2% regression ship gate.

The mechanism is structural to the lever: any approach that places the
scale-issue at iteration k-1 (or earlier) for consumption at iteration k
must hold the scale-pack VGPRs across the intervening MMAs — there is
no way to recover the live-range savings without losing the lead-distance
the lever depends on. A shadow-register variant (separate `_next` scale
packs that get promoted at the top of the next even iter) would *double*
the scale-pack footprint (~6 → ~12 VGPRs), which would saturate the
ceiling even harder.

## Decision

**Bail without bench.** The kernel patch and the build evidence are kept in
the tree (gated by `MXFP8_CRR_SCALE_LEAD`, default OFF) so future cycles
can audit the analytical refutation. The CRR scale prefetch lead-distance
lever is closed.

## Forward look

CRR's structural ~92% ceiling is now more tightly explained. The remaining
~3pp gap to RCR's ~95% is split between:
1. The 6×`v_lshrrev_b32` scale-pack shifts per K-pair (R48 Dev D
   hardware-ceiling analysis; opsel re-pack lever closed by R49 Dev A).
2. Implicit vmcnt drain stalls between scale-issue and cA-MMA (this
   cycle's lever; closed by VGPR pressure).

Both leverage points sit inside a tight 256-VGPR / 8-wave / occupancy-2
budget that has no slack for either layout-driven re-packs (force extra
v_perm) or live-range-driven prefetches (force VGPR spill). The CRR
ceiling is treated as structurally bounded by the kernel's register
budget, not by the scale-fetch micro-architecture.

Adjacent levers still on the candidate list:
- 4096³-specific persistent / split-K variant gated on `total_tiles == nCU`
  (r49c_findings.md §7.3) — the second occupancy slot is empty at 4096³,
  so VGPR/LDS budget can be spent more aggressively without occupancy
  penalty. Has not been prototyped.
- B-tile LDS swizzle audit at N=4096 specifically (r49c_findings.md §7.2,
  RRR-side counterpart).

## Files

- `crr_mxfp8_exact_8wave_fastpath.inc` — added `MXFP8_CRR_SCALE_LEAD` gate
  (default 0) and the LEAD=1 K-loop variant (pre-loop scale preload +
  post-cD scale prefetch on odd iters + tail load_raw_scales gate). LEAD=0
  arm matches the prior baseline byte-for-byte (validated by re-reading
  the build remark: 225 VGPR / 0 spill, identical to pre-patch).
- `r50a_results/SUMMARY.txt` — per-shape build resource summary
- `r50a_results/build_<MxNxK>_lead{0,1}.log` — full hipcc remarks per
  (shape, LEAD) pair (7 shapes × 2 LEAD = 14 build logs)

## Protocol note

Bench was NOT run (5×/30s/60s strict SCLK). The bail trigger
(VGPR spill regression, build remark) is observable at compile time and
satisfies the R50 prompt's bail-condition, parallel to the R49 Dev C
persistent-kernel analytical refutation and the R49 Dev B lesson.
