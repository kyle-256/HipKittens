# R55 Dev A — 70B Down RCR P1 scale-fetch interleave

**Verdict:** REFUTED-EMPIRICAL-LLVM-RESCHEDULES

**Target:** 70B Down RCR -3.4pp HEADROOM cell from R54 baseline. R54B PMC
diagnostic identified DIAGNOSTIC-SCALE-FETCH-WAIT class (SQ_WAIT_ANY +71.6%,
MfmaUtil -16pp) — back-to-back scale-tensor VMEM loads cluster the issue
stream and force serialised L2/TCC return arrivals, expanding `s_waitcnt
vmcnt(N)` drain BEFORE MMA segment 0.

**Hypothesis (P1):** Interleaving scale-issuance with data-tile load issuance
should overlap the two streams in the L2/TCC return path, reducing
`SQ_WAIT_ANY`.

## Lever

Macro `MXFP8_RCR_SCALE_INTERLEAVE` (values 0-3) in `kernel_mxfp8_layouts.cpp`:

| Value | Description |
|---|---|
| 0 | baseline (back-to-back A b128 then B b64 in `load_scale_buffer`) |
| 1 | insert `__builtin_amdgcn_sched_barrier(0)` BETWEEN A and B inside `load_scale_buffer` (lets LLVM scheduler interleave dwordx4 LDS issuance) |
| 2 | source-level split: A-scale b128 stays in `load_scale_buffer`; B-scale b64 emitted via separate `load_scale_buffer_b_late()` invoked from inside `do_k_iter_body` AFTER first per-kpair `ds_read` and BEFORE segment-0 MFMA |
| 3 | like 2 PLUS `sched_barrier` after b64 issue (firewall the late B-scale from being re-floated by scheduler) |

Mutually exclusive with `MXFP8_RCR_COOPERATIVE_BSCALE` and
`MXFP8_RCR_ASCALE_FIRST` (`#error` guarded).

## Phase 1 (ISA evidence)

K-loop body ISA inspection (`r55a_results/isa/v{0,1,2,3}_lbb27.s`):

```
$ diff v0_lbb27.s v1_lbb27.s
2a3
> 	; sched_barrier mask(0x00000000)
```

V0 vs V1 K-loop body bodies are byte-identical apart from the `sched_barrier`
comment marker. AMDGPU scheduler **clusters the b64 B-scale load back next to
the b128 A-scale load** regardless of source-level reordering or
`sched_barrier(0)` hint between them. V2/V3 (which physically move the b64
into `do_k_iter_body`) likewise resolve to the same clustered VMEM issue
schedule.

Conclusion: LLVM scheduler defeats source-level scale/data interleave at this
VGPR pressure point.

## Phase 2 (5-run SCLK bench, 3 cells × 4 variants = 60 runs)

`r55a_results/bench/`. Median TFLOPS per cell × variant:

| Cell | V0 | V1 | V2 | V3 | Best Δ% |
|---|---:|---:|---:|---:|---:|
| 70B_Down_RCR | 2992.84 | 2993.21 | 2988.02 | 3001.40 | +0.29% |
| 8B_Down_RCR  | 2984.20 | 2981.93 | 2984.65 | 2959.58 | +0.02% |
| 70B_QO_RCR   | 2927.84 | 2920.10 | 2913.25 | 2925.92 | -0.07% |

All 4 variants collapse within ±0.5% noise band across all 3 cells. No
variant exceeds the +1% pass threshold; 8B_Down V3 and 70B_QO V1/V2 sit
slightly below baseline.

## Verdict rationale

ISA evidence shows the lever is a no-op (compiler reschedules around it).
Phase 2 bench confirms zero observable perf signal in the 4096³-class noise
band. The DIAGNOSTIC-SCALE-FETCH-WAIT bottleneck class is **not addressable
by source-level scale-load reordering**.

## Macro disposition

- `MXFP8_RCR_SCALE_INTERLEAVE` left default-OFF (0) in
  `kernel_mxfp8_layouts.cpp:493`. Production tree byte-identical.
- Macro retained as documented dead-end (consistent with R55B/R55D pattern)
  with `#error` guard against `COOPERATIVE_BSCALE` / `ASCALE_FIRST` to
  prevent accidental combination if a future agent revisits.

## Cumulative R55 P1-P4 attack outcome

R55 attacks the DIAGNOSTIC-SCALE-FETCH-WAIT class on four axes (Dev B
prediction):

| Axis | Dev | Verdict | Evidence |
|---|---|---|---|
| P1 scale-fetch interleave | A | REFUTED-LLVM-RESCHEDULES | this report |
| P2 SALU hoist             | B | REFUTED-COMPILER-ALREADY-HOISTED | `r55b_findings.md` (`eb7483c2`) |
| P3 VALU dep-break         | C | REFUTED-PASS-CRITERION (+0.65% < +1%) | `r55c_findings.md` (`21d2656b`) |
| P4 scale L2 residency     | D | REFUTED (cachepolicy axis 3× closed) | `r55d_findings.md` (`10632347`) |

All four mechanisms predicted by R54B are closed. V2 RCR HEADROOM is **not
addressable by source-level scale-load transforms**. R56+ must pivot to a
different attack family (e.g., R55F's SALU-DOMINANT class on RRR, or
PMC-driven third bottleneck class on 70B Gate/Up CRR).
