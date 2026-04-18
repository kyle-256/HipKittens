# R31 DECIDER VERDICT — L6 (4096×32768×128256), 41/42 → ?

**Date**: 2026-04-18  Decider: Opus 4.7 (no-build pure analysis)
**State**: L6 only LOSE, 5353.9 / 5781.1 = 92.6%. 7.4pp gap.
Sources: `kernel_mxfp4_gluon_cpp.cpp:80-110, 2437-2450, 2805-2825`,
`bench_all42_results_R25_FINAL_v2.json` L6 entry, `R30_DECIDER_VERDICT.md`,
`R29_L6_V8_VERDICT.md`, `R29_L6_DECIDER_VERDICT.md`.

---

## A1 — UNROLL_K factor sweep on L6 — **DEAD-by-prior-evidence**

The kernel default at K=128256 falls through to `#pragma unroll 8`
(`kernel_mxfp4_gluon_cpp.cpp:2443-2448`); `UNROLL_K` macro at line 2437-2442
overrides it. The L6 per_variant table already contains: `u8`=4996.5,
`u16`=5004.9, `u32`=4997.7, `gm8u8`=5068.8, `gm8u16`=5060.5,
`gm16u16`=5009.2, `u32_btw_all`=5181.5. All are **≥170 TFLOPS BELOW** the
current best (5353.9). Compiler gives the steady-state loop the same SGPR
budget regardless of unroll factor (212 VGPR observed in V8 build per
`R29_L6_V8_VERDICT.md` step 2), so occupancy is fixed at the 16×16×KPAIR
template. Pragma-unroll{1,2,4} would *increase* loop overhead in a
501-iter loop — strictly negative EV. **No new step.**

## A2 — Mini-split-K POC (host-side) — **DEAD-by-prior-evidence**

The store path at `kernel_mxfp4_gluon_cpp.cpp:575-587` (`store_bf16_val`,
`store_bf16x2_packed`) writes bf16 directly to the output tile — **no
atomicAdd, no FP32 workspace path exists**. `bench_all_42.py:90`
(`build_for_nk`) builds a single .so per (N,K); the dispatcher launches
one kernel per shape. To do a 2-launch split:
(1) add a workspace ptr + split-index kernel arg (signature change cascades
through `make_pf_params`, the persistent-XCD launcher, and the Python
binding); (2) write a 2nd-pass FP32→bf16 add+cast kernel; (3) re-validate
SNR on full DLA1. R30 §Q3 already costed this at "≥3 days, HIGH risk"
(atomic contention nullifies the gain). The "2-3 hr hack" framing is wrong
— the bf16 direct-store path means there's no FP32 buffer to atomic-add
INTO; the entire epilogue must be replaced. **Same multi-day project as
R30 V6, just renamed.** No 2-hr action exists here.

## A3 — Kernel fork: `#pragma unroll 1` + R25C tail-PF-off — **NEW (low-tier)**

Forcing `UNROLL_K=1` removes the unroll-8 fold issue that blocks
`R25C_ACTIVE` at K_iters=501 (`kernel:80-110` comment block:
"runtime check inside the hot loop and code-size doubles → catastrophic
regression"). With unroll=1 the runtime branch at `kernel:2810`
(`_r25c_tail_no_pf = (bt >= k_byte_iters - 1 - R25C_TAIL_PF_OFF_ITERS)`)
becomes a single trivial check per iter rather than 8 inlined copies. BUT:
unroll=1 itself reverts to one full kpair body per iter with no
software-pipelined LDS-issue overlap → R20A regressed to 4996 TFLOPS as
shown in A1 data (`u32`=4997 ≈ unroll=1-equivalent). Best-case math:
unroll=1 baseline ~5000 + R25C-style tail-pf-off win (best observed +14.5%
on K=14336 K_EXACT) is structurally bounded by the **fraction of dead PF
loads**: 6/501 = 1.2% of K-iters → at most ~1pp recovery. Net: 5000 ×
1.012 ≈ 5060, still **293 TFLOPS BELOW** current 5353. **NEW but
recommend SKIP** — the math doesn't close.

## A4 — Persistent-XCD remap on L6 — **NEW (highest EV)**

`PERSISTENT_XCD` (kernel:160-181, machinery at 2150/2186/2199/2240,
launcher 3209-3239) and `STATIC_XCD_REMAP` (167-173) macros exist and are
default-OFF. **L6's per_variant table contains ZERO entries with
xcd/persistent/remap substrings** (filtered, empty result). The standard
variant list never compiles `-DPERSISTENT_XCD=1` or
`-DSTATIC_XCD_REMAP=1` for the L6 (N=32768, K=128256) build. L6's
4096×32768 grid = 32×128 = 4096 tiles on 304 CUs has a known tail effect
(R29 decider §V7 explicitly flagged this as L6's pathology). Persistent-
XCD with `PERSISTENT_GRID=608` (kernel:176) directly addresses tail-
draining via atomic tile claim. Concrete next step: build 4 variants
on top of the existing winner `ts_lgk2_v12_memc_btw_all`:
`+PERSISTENT_XCD=1`, `+STATIC_XCD_REMAP=1`, `+PERSISTENT_XCD=1
PERSISTENT_BATCH=4`, `+STATIC_XCD_REMAP=1 GROUP_SIZE_M=8`. ~30 min build
+ 5-rep × 4 variants ≈ 1.5 hr. Expected gain: **0-3pp p50, 5pp p90.**

## A5 — Compiler axis untried — **NEW (low-tier)**

Current best L6 build uses `-mllvm -amdgpu-sched-strategy=max-memory-
clause` (verified `r30_opt_a_build.py:36`). Untried `-mllvm
-amdgpu-sched-strategy=` values: `iterative-ilp` (CAUTION: caused HSA
aperture per user note → SKIP), `iterative-minreg`, `iterative-maxocc`,
`max-ilp`. Other untried: `-mllvm -amdgpu-disable-clustered-low-occupancy-
reschedule` (already in `_dc` variants, L6 has `lgk2_dc_btw_all`=5190.7
— **regresses** vs 5353); `-mllvm -amdgpu-enable-power-sched`. Concrete
next step: build 3 variants `_memc_minreg`, `_memc_maxocc`, `_memc_maxilp`
on top of the winner. ~20 min build + 5-rep bench ≈ 45 min. Expected
gain: 0-2pp p50; high probability of zero or regression (the existing
`_dc` data shows compiler-axis perturbations move L6 by <1pp).

## A6 — Free response: STEP3_BARRIER_VMCNT axis untried at K=128256

The winner uses `STEP3_BARRIER_VMCNT=12` (variant name `_v12_`). The L6
table contains `v12`, `v16` (via `wpe1_memc`), `v20` entries but **no
v4 / v6 / v8 / v10 / v24 / v28 sweep** at the `ts_lgk2_*_btw_all` parent
stack. At K=501 iters, the steady-state vmcnt-fence value affects how
many in-flight VMEM loads stall the next kpair_step3. Concrete next step:
build `ts_lgk2_v{4,8,10,16,20,24}_memc_btw_all`. ~30 min build + 5-rep ×
6 ≈ 1.5 hr. Expected gain: **0-2pp p50, 4pp p90** (R25 finding was that
vmcnt sweeps move ≤2pp on K-bound shapes).

---

## Final ranked recommendation

**R31-A — Persistent-XCD / STATIC_XCD_REMAP scout on L6 (axis A4).**
Highest EV per hour because (a) L6's per_variant XCD coverage is **zero**,
(b) machinery exists and compiles cleanly, (c) directly targets the
known-tail-effect pathology cited in R29 §V7. ~1.5 hr.

**R31-B — STEP3_BARRIER_VMCNT sweep on the winner parent stack (axis A6).**
Cheap parallel build, fully orthogonal to A4, no infrastructure risk. The
v12-vs-v20 spread on L6 (5354 vs 5193) is 3pp — non-trivial — and
intermediate values are unsampled. ~1.5 hr.

**Honest assessment**: probability that A4 ≥ 1pp gain ≈ **35%**;
A6 ≥ 1pp gain ≈ **25%**; both fail ≈ **49%**. If both return zero,
**STOP and confirm 41/42 ceiling** — remaining levers (V5 MFMA32, V6
split-K, V7 Stream-K) are all multi-day to multi-week. Do NOT
re-attempt V8/R26-A/R27-C variants — those failure modes are
structural, not noise.

(word count ~915)
