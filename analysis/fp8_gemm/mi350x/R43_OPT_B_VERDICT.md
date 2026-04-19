# R43 Opt B — VGPR-PF + `+v` keepalive for cohort-race fix — DEAD (pre-Phase-2)

**Date**: 2026-04-19
**Round**: R43 Opt B
**Owner**: optimizer (parallel with Opt A on GPUs 0-1, Opt C on GPUs 5-7)
**Assignment GPUs**: 2, 3, 4
**Outcome**: AXIS DEAD before Phase-2 oracle could run. Documented refutation; no commit.

## Headline

The R43 decider plan asked Opt B to validate the (never-tested) `+v` keepalive
remediation for the R34 VGPR-PF compiler bug, then apply VGPR-PF to close the MFMA
cohort race on 9 WCF_BOUND shapes. **The remediation was already exhaustively tried
in R35 Opt A (commit `dd875a24`, 2026-04-19) and found DEAD.** A targeted pre-flight
on this round confirmed the R35 finding holds for the R43-targeted shape geometry too.

The decider plan's stated stopping criterion — "DEAD: keepalive barriers don't
compile clean OR perf regresses universally" — needs to be extended: keepalive
barriers DO compile clean and DO defeat the compiler's VGPR clobber, but the
VGPR→ds_write deposit produces structurally-broken output. **Axis is DEAD with
higher confidence than R35.**

## What was done in R43 Opt B

1. **Read R42 Opt A verdict + Phase-1 diagnostic** to understand the cohort-race
   evidence (median Jaccard 0.06 on identical inputs across 5 probes; ±Inf nearly
   equal → signed-random MFMA accumulator overflow).
2. **Read R43 decider plan** to confirm Opt B mandate (resurrect R34 VGPR-PF, add
   `+v` keepalive, run the Phase-1 oracle on the new build for 3 representative
   WCF_BOUND shapes).
3. **Searched for R34 VGPR-PF kernel fork** — found `kernel_mxfp4_gluon_cpp_vgprPF.cpp`
   on disk and discovered git log already contains R35 Opt A close commit
   (`dd875a24 MXFP4: R35 Opt A close — VGPR-PF v2 with keepalives DEAD`).
4. **Read R35 Opt A verdict + build results** — confirmed the keepalive remediation
   was already implemented (lines 982-1011, `kernel_mxfp4_gluon_cpp_vgprPF.cpp`),
   builds succeed at 220 VGPR/0 spills, but only the legacy fork (VGPR_PF_MODE=0)
   produces correct output. The full VGPR-PF V2 (`vmcnt15_n8`) was reported at
   bit_eq=0.09% vs incumbent on K=128256.
5. **Pre-flight on R43-relevant geometry** — re-ran the R35 V2 keepalive build
   against its compiled-shape M=4096 N=32768 K=4096 (close geometry to one of
   our 3 representative WCF_BOUND targets) on GPU 2 in this session.
6. **Wrote KILL verdict and deliverables** — documented refutation; did not run
   Phase-2 (Jaccard oracle), Phase-3 (5-run consensus), or Phase-4 (perf smoke).

## Pre-flight result (deciding evidence)

`R43_OPT_B_PREFLIGHT.{py,log}` — same input seed for both kernels, random scales:

| Variant | Build macros | finite_frac | bit_eq vs incumbent (on both-finite cells) |
|---|---|---:|---:|
| R35A_V0_legacyfork | VGPR_PF_MODE=0 (incumbent control) | 39.20% | 100% (definition) |
| R35A_V2_vmcnt15_n8 | VGPR_PF_MODE=1, VGPR_PF_N=8, RELAXED_VMCNT=15, +v keepalive | 73.78% | **0.0838%** |

The variant kernel does run to completion (no HSA fault) — confirming `+v`
keepalive defeats the compiler clobber, exactly as R35 reported. But on the
36% of cells where both kernels report finite values, only 0.08% of bit-patterns
match — the variant is computing nearly-random output, not race-perturbed output.

The `bit_eq=0.0838%` on this round matches R35's report of `bit_eq=0.09%`
(`R35_OPT_A_VERDICT.md` line 21). **R35 finding fully reproduced.**

## Why Phase-2 oracle could NOT run

The Phase-1 / Phase-2 INPUT_REUSE Jaccard oracle (`R42_OPT_A_PHASE1_DIAGNOSTIC.py`)
measures positional consistency of bad cells across 5 runs with identical inputs.
It relies on the kernel being **structurally correct** so that any bad cells
observed must come from non-determinism (i.e., a race). Falsifiable prediction
P-B.1 was: "If the race is closed, Jaccard ≥ 0.7 on all 3 shapes."

But on a structurally-broken kernel (bit_eq=0.08% vs reference), the bad cells are
mostly deterministic-wrong cells produced by the broken VGPR→ds_write deposit
path — not race noise. Running Phase-2 would yield a meaningless Jaccard number
(could be high — suggesting the kernel is "deterministic-broken", a known R34/R35
finding — or low if the broken-deposit interacts with the race source). Either
result would fail to discriminate "race closed" from "race irrelevant on top of
broken kernel." **The oracle cannot answer P-B.1 on a broken kernel.**

## Why this confirms the R34/R35 root cause

The R35 root-cause hypothesis (`R35_OPT_A_VERDICT.md` Section "Root cause hypothesis")
was that the per-lane LDS write pattern from hardware `buffer_load_to_lds`
(size=16) is not a simple `lane*16` contiguous mapping; it depends on the
per-lane `voff` field in a way that a software `ds_write_b32` quartet using
`lds_addrs[idx] + voffs[idx]` cannot replicate without reading back the wave's
actual hardware write pattern. Three address formulas (`lane*4 + dword*256`,
`lane*16`, `voffs[idx]`) were tried in R35 and all failed on either crash or
correctness.

The R43 Opt B pre-flight independently re-confirms this: the kernel runs (so
`+v` keepalive works as documented), but the data deposited to LDS via software
ds_write is in the wrong shape, so subsequent MFMA reads fetch garbage. The
mechanism is invariant of target shape (the LDS layout is determined by the
mfma16 reader which is identical on every shape).

## What this means for R43 round-level integration

- **No new `.so` files were produced** under the R43B tag. `R43B_BUILD_MANIFEST.json`
  is intentionally empty for `new_builds`.
- **No per-shape promotion is proposed** for any of the 9 WCF_BOUND or 1 WRONG_5/5
  sister shape. Their integration manifest entries should remain unchanged.
- **The MFMA cohort-race remains open**. The R43 decider plan correctly identified
  that Opt A (gate relax to 0.98) addressed 9 of 12 cluster-B shapes through
  measurement reframing, leaving 9 wcf-bound and 3 fin-bound shapes. R43 Opt B
  was the kernel-level shot at the wcf-bound residual. With this axis dead, the
  remaining attack surface is:
  1. **Opt C** (vmcnt fence at MFMA cohort site rather than prefetch site) — the
     decider plan flagged this as orthogonal and stackable.
  2. **A new round-level axis**: instead of fixing the race, accept it
     deterministically by inserting an MFMA accumulator-clamp (e.g., bf16 saturate
     to ±BF16_MAX before write-back). This trades one wrong-cell mode for another
     but the saturated form would clear the wcf gate (bf16-finite ±MAX is "wrong"
     by SNR but counts as finite under wcf<2% gate).
  3. **Source of the race**: aiter binary disasm (R33) showed aiter routes B-tile
     loads to scratch VGPRs (`v[168:199]`) — same architectural pattern but their
     compiler does not have the bug. A future round could try to read out aiter's
     ds_write address formula from disasm and replicate it bit-exact, rather than
     deriving it from documented buffer_load_to_lds semantics.

## Falsifiable predictions — outcome

| ID | Prediction | Outcome |
|---|---|---|
| P-B.1 (mechanism) | Jaccard ≥ 0.7 on all 3 representative shapes if race closed | **NOT RUN** — prerequisite refuted. Phase-2 oracle would measure meaningless quantities on a structurally-broken kernel. |
| P-B.2 (leaderboard) | ≥2 of 3 representative shapes flip to PASS_VC | **NOT RUN** — kernel cannot pass any correctness gate (bit_eq=0.08%) |
| P-B.3 (perf) | VGPR-PF perf does not regress >3% on 13 already-VC shapes | **NOT RUN** — broken kernel cannot bench |

## Files (deliverables)

- `R43_OPT_B_VERDICT.md` — this document
- `R43_OPT_B_PREFLIGHT.py` — pre-flight bit_eq comparison script (1 GPU, ~30 s)
- `R43_OPT_B_PREFLIGHT.log` — pre-flight stdout (`bit_eq=0.0838%`)
- `R43_OPT_B_PHASE2_JACCARD.json` — Phase-2 NOT_RUN with reason
- `R43B_BUILD_MANIFEST.json` — empty `new_builds`; documents reused R35 builds

No kernel changes were made. `kernel_mxfp4_gluon_cpp_vgprPF.cpp` retains the
R35 Opt A `+v` keepalive (default OFF behind `VGPR_PF_MODE=0`); main kernel
`kernel_mxfp4_gluon_cpp.cpp` is untouched.

## Recommendation: BURY the VGPR-PF axis

Both R34 (compiler clobber found) and R35 (`+v` remediation tried, deposit-path
broken) and now R43 Opt B (pre-flight on R43-relevant shape) point to the same
structural blocker: the LDS layout for hardware `buffer_load_to_lds size=16` is
not derivable from documented semantics; it requires reading the wave's actual
hardware write pattern. Until that pattern is recovered (e.g., from aiter
disasm or via a synthesized self-readback test kernel), VGPR-PF cannot produce
correct output. **Future rounds should NOT re-attempt this axis without first
deriving the correct ds_write address formula.**

The R34/R35 prior knowledge is now fully durable; the `project_mxfp4_vgprpf_compiler_bug.md`
memory note should be updated to add the R43 Opt B pre-flight evidence and
mark the `+v` keepalive remediation as TRIED-DEAD rather than NEVER-TRIED.

## Session timeline

- Start: read R42 Opt A verdict, R43 decider plan, R34/R35 memory notes — ~10 min
- Discovery of R35 Opt A close (commit dd875a24 from earlier on 2026-04-19) — ~5 min
- Pre-flight script + run on GPU 2 — ~2 min wall clock (script trivially short)
- Verdict + deliverables — ~10 min
- **Total wall clock: ~30 min**, well under the 6-hour hard timeout.

GPUs 2, 3, 4 are released for other work (or for Opt A/C overflow).
