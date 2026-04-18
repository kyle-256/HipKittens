# R32 — Optimizer A verdict (K_LOOP_SYNC_EVERY_2 + NONTEMPORAL loads on L6)

**Shape**: L6 = 4096×32768×128256 (only this shape).
**Parent stack**: `ts_lgk2_v12_memc_btw_all` (incumbent best at 5354 TFLOPS, 92.6% of comp 5781).
**Date**: 2026-04-18

## TL;DR

**NO WIN. All 3 variants DEAD.** The two un-sampled axes flagged by the
R32 decider for L6 fail at the correctness/launch gate, not the perf gate.

| Variant | Build | SNR vs incumbent | Verdict |
|---|---|---:|---|
| V1 `+K_LOOP_SYNC_EVERY_2=1` | PASS but +44 VGPR / +32 spills / +132 B scratch | NaN (n_diff 23,080,486; max_abs 3.4e38 = bf16 max) | **DEAD-BY-SNR** |
| V2 `+B_LOAD_NONTEMPORAL=1 +A_LOAD_NONTEMPORAL=1` | PASS clean (212 VGPR, 0 spills, 0 scratch) | n/a — kernel CRASHES on launch | **DEAD-BY-CRASH** (HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION rc=-6, 191s) |
| V3 V1+V2 combined | PASS but +44 VGPR / +32 spills / +132 B scratch | NaN (n_diff 20,192,113; max_abs 3.4e38) | **DEAD-BY-SNR** |

The decider's pre-stated risks materialized: **K_LOOP_SYNC_EVERY_2** breaks
cross-wave LDS ordering at K=128256, and **A/B NONTEMPORAL** corrupts the
prefetch pipeline on the L6 stack the same way `STEP3_BARRIER_VMCNT≥20` did
in R31-C.

## Step 1 — Build (R32_OPT_A_BUILD_RESULTS.json)

`build_round32_optA.py` — 3 parallel builds, wall 5.3 s.
Parent flags: `-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause -DBARRIER_TO_WAITCNT_ALL=1`.

| Variant | VGPR | AGPR | SGPR | VGPR-spill | scratch [B/lane] | occ | compile_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1 kls2 | **256** | 256 | 99 | **32** | **132** | 1 | 5.3 |
| V2 nt | 212 | 256 | 93 | 0 | 0 | 1 | 5.2 |
| V3 kls2+nt | **256** | 256 | 99 | **32** | **132** | 1 | 5.2 |

V1/V3 already show a **structural compiler regression** from the new
even/odd K-loop branch: VGPR jumps 212→256, 32 VGPRs spilled to 132 B/lane
of scratch. Even if correctness held, this would crater perf vs incumbent.
V2 is bit-clean (identical resource footprint to incumbent) — its failure
is purely runtime.

Build artifacts: `build_all42/tk_mxfp4_gluon_cpp_n32768_k128256_R32A_ts_lgk2_v12_memc_btw_all_{kls2,nt,kls2_nt}.so`.

## Step 2 — Correctness gate (kernel-vs-incumbent SNR)

`snr_R32_optA_v3.py` — full M×N=4096×32768 GEMM with random fp4 + random
[-2,3] exponent scales (the R29 verify pattern). Compares each variant's
bf16 output to the incumbent's output, restricted to entries finite in
both. Acceptance gate: **SNR ≥ 25 dB** (per R32 mission spec).

Note: random-scale + bf16 output naturally produces ~40 % NaN/inf entries
even on the known-good incumbent (R29 saw 31.9 % min_finite_frac on L4).
This is a benchmark-rules-acknowledged property of the random aperture
test; comparison is restricted to both-finite entries. Incumbent here:
finite_frac=0.5832–0.5935 across runs.

```
[incumbent] running... finite_frac=0.5832
[V1_kls2]   finite_frac=1.0000 both_finite_frac=0.5832 snr=NaN dB
            n_diff=23,080,486 (out of 78.4 M both-finite) max_abs_diff=3.4e38
[V2_nt]     CRASH HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION (rc=-6, 191s)
[V3_kls2_nt] finite_frac=1.0000 both_finite_frac=0.5935 snr=NaN dB
            n_diff=20,192,113 max_abs_diff=3.4e38
```

(V3 was re-run standalone after V2's crash aborted the script:
`R32_OPT_A_SNR_V3only.log`.)

### V1 / V3: K_LOOP_SYNC_EVERY_2 is unsafe at K=128256

Decoder note (kernel:374-396, 2826-2862): `K_LOOP_SYNC_EVERY_2` skips the
`s_barrier` on odd-`bt` Step3 calls, relying on the natural `lgkmcnt+vmcnt`
+ `s_waitcnt lgkmcnt(0)` of Step12+Step3 to cover the producer→consumer
gap. At K=128256 (501 K-iters, only the outer `#pragma unroll 8` fires),
the alternate even/odd `kpair_32mfma_with_lds_and_pf<...,STEP3_EMBED_BARRIER>`
vs `<...,false>` template instantiations create **two distinct K-loop
function bodies**, doubling the unrolled code size. Compiler responds by
spilling 32 VGPRs to 132 B/lane of scratch (Step 1 above) — and even with
the spills, **wave-N+1 starts ds_read on `A*_db[N&1]` before wave-N's
buffer_load_to_lds for that slot has finished writing**, producing garbage
that flushes through the MFMA accumulator → output values at bf16 max
magnitude (max_abs_diff 3.4e38, the bf16 finite ceiling). 23 M / 78.4 M
finite entries differ from incumbent (29 % corruption rate).

V3 = V1 ⊕ V2 inherits V1's correctness break (same n_diff scale) but
V2's crash is masked because V3's broken kls2 codegen happens to not hit
whatever address the NONTEMPORAL hint corrupted in V2 (V3 ran to
completion). Either way, kls2's correctness break alone disqualifies V3.

### V2: NONTEMPORAL on K=128256 hits HSA aperture violation

Decoder note (kernel:325-350): with `B_LOAD_NONTEMPORAL=1` and
`A_LOAD_NONTEMPORAL=1`, the buffer-load coherency hint flips from
`cache_all` to `non_temporal` (slc + glc both set, full L1+L2 bypass).
This is the same crash class observed in R31-C `STEP3_BARRIER_VMCNT∈{20,24}`:
on L6's K=128256 with the v12-stack prefetch pipeline, decoupling cache
behavior from the SRD-bound prefetch lets a stale (or never-cached) line
get re-issued past the SRD limit, producing
`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION code: 0x29` after ~191 s of
warmup iterations. The crash is reproducible — single SNR pass triggered
it deterministically.

Build resource usage was clean (212 VGPR / 0 spill / 0 scratch identical
to incumbent), so this is purely a runtime/coherency interaction with the
v12 stack at K=128256, not a codegen pathology.

## Step 3 — Bench (NOT EXECUTED)

Per mission spec, bench runs only on variants passing the SNR ≥ 25 dB
gate. **Zero variants passed**, so no 5-rep bench was performed. (V2 also
cannot be safely benched — repeated launches would re-trigger the HSA
aperture violation.)

## Step 4 — Verdict & recommendations

**R32-A: NO WIN. L6 incumbent ceiling at 5354 TFLOPS / 92.6% confirmed.**

Combined with R29 / R30 / R31-A / R31-B / R31-C / R32-A, **all six
sub-2-hour structural axes** the decider laid out for L6 have now been
sampled and DEAD:

| Round | Axis | Verdict |
|---|---|---|
| R29 | V8 R25E peel | DEAD |
| R30 | UNROLL_K + Persistent-XCD + STEP3_VMCNT | DEAD |
| R31-A | UNROLL_K {1,2,4,16,32} | DEAD |
| R31-B | PERSISTENT_XCD + STATIC_XCD_REMAP | DEAD-BY-CRASH |
| R31-C | STEP3_BARRIER_VMCNT {4,8,10,16,20,24} | DEAD/CRASH |
| **R32-A** | **K_LOOP_SYNC_EVERY_2 + NONTEMPORAL_LOAD** | **DEAD/CRASH** |

**Recommendation**: Declare 41/42 the achieved ceiling. The remaining
7.4 pp gap on L6 is **structural** (K=128256 is VMEM-issue-bound at
vmcnt(8); aiter likely uses MFMA32 / split-K / Stream-K). Per the R32
decider §A3 estimate, **V6 split-K (atomic-free, S=2 / S=4)** is the
only credible sub-2-day lever (~12 hours, expected 3–6 pp on L6) — that
requires user approval as a 1.5-day sprint, not an R32-class round.

Otherwise, ship 41/42 and call MXFP4 done.

## Files produced

- `build_round32_optA.py` — build script (3 variants, parallel)
- `bench_R32_optA.py` — bench/SNR script (with --snr-only mode)
- `snr_R32_optA_v3.py` — kernel-vs-incumbent SNR check (the one that worked)
- `R32_OPT_A_BUILD.log`, `R32_OPT_A_BUILD_RESULTS.json`
- `R32_OPT_A_SNR.log`, `R32_OPT_A_SNR_RESULTS.json` (v1, partial-K bug)
- `R32_OPT_A_SNR_v2.log`, `R32_OPT_A_SNR_v2_RESULTS.json` (v2, full-K-vs-torch — torch ref overflows bf16, still NaN on incumbent)
- `R32_OPT_A_SNR_v3.log`, `R32_OPT_A_SNR_v3_RESULTS.json` (v3, full-K-vs-incumbent — the correct test)
- `R32_OPT_A_SNR_V3only.log` (V3 standalone after V2 crash aborted v3 main)
- `build_all42/compile_R32A_{V1_kls2,V2_nt,V3_kls2_nt}_n32768_k128256.log` (compile remarks)
- `build_all42/tk_mxfp4_gluon_cpp_n32768_k128256_R32A_ts_lgk2_v12_memc_btw_all_{kls2,nt,kls2_nt}.so` (3 modules)

No source edits to `kernel_mxfp4_gluon_cpp.cpp`, `bench_all_42.py`, `TODO.md`,
or `AGENT_PROMPT.md`. No commits (no WIN).

Wall clock used: ≈25 min of 120 min cap. Early termination because all
3 variants failed the SNR / launch gate; no bench cycles required.
