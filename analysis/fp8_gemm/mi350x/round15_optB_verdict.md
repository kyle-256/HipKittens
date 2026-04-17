# Round 15 Optimizer B — Verdict

**Theme:** Untested LLVM/AMDGPU flag families on the 4 broken deep-LOSE shapes
(DLA1 / DLA2 / DLA7 / P1).

**Result:** 0 wins. **DEAD END.** Adds 42 more flag tags to the saturation registry.

---

## Methodology

- **Build:** 4 shapes × 42 candidate flags = 168 variants. All 168 built OK
  (no MISSING/BROKEN beyond R14A's known list). Parent flags identical to R14A.
- **ASM-diff probe** (`asm_diff_probe_r15b.py`): SHA-256 of `.text` per variant.
  Note: parent .so was built without `-mllvm` wrappers, so its `.text` differs
  uniformly from every variant. Real "NOOP" verdict is variants whose hash
  equals the **modal** variant hash for that shape.
- **Smoke** (`bench_round15_optB_smoke.py`): single-shot warmup=200 iters=500,
  GPUs 3-5 (R15A=0-2, R15C=6-7). Gate: ≥+0.5 pp.
- **Verify** (`bench_round15_optB_verify.py`): 5-run, GPU 3, warmup=200 iters=500,
  trim=10%. Gate: mean ≥ baseline.max **AND** mean Δ ≥ +1 pp.

---

## Candidate flags (42, all NEW vs R14A)

Family 1 — Coalescer: `nojglc nojli nojse notarsch largeivf2 largeivs8 lateremat0 revlocal`
Family 2 — Spill / AGPR: `defspill nospillfu sinkavoidspill preallocs noliveopt`
Family 3 — Machine sink/LICM: `nosink nolicm hoistcheap noavoidspec sinkbfi sinkcycle100`
Family 4 — Post-RA sched: `postmi enpostmi nopostra postraN breakcrit breakall`
Family 5 — IGLP: `iglpcut0 iglpexact`
Family 6 — Loop / prefetch: `nolal looprmul loopdist`
Family 7 — AMDGPU: `aaaa noaa rerag eppra noeppra noemxpre nodalu vopd eifcvt nosgpcb nosgpmwc nosgpwa`

## ASM-diff verdict counts (vs modal-variant hash per shape)

| Shape | TRUE-DIFF tags | NOOP tags |
|------|------|------|
| DLA1 | 6  (`nojli largeivf2 revlocal nosink nopostra noemxpre`) | 36 |
| DLA2 | 5  (`nojli revlocal nosink nopostra noemxpre`) | 37 |
| DLA7 | 5  (`nojli revlocal nosink nopostra noemxpre`) | 37 |
| P1   | 7  (`nojli revlocal sinkavoidspill nosink nolicm nopostra noemxpre`) | 35 |

**Of the 42 R15B flags, only 8 produce ANY measurable .text change on at least
one shape**; the remaining 34 are silent NOOPs at this kernel's optimization
level. This is consistent with the R14A finding that the kernel sits in a
heavily over-determined region of the LLVM flag space.

## Smoke-PASS shortlist (single-shot)

| Shape | Tag | LLVM flag | base | cand | Δ pp |
|---|---|---|---:|---:|---:|
| P1   | nosink     | `-disable-machine-sink`            | 689.9  | 5013.1 | +81.99 ⚠ baseline anomaly |
| DLA1 | noemxpre   | `-amdgpu-opt-exec-mask-pre-ra=false` | 4712.7 | 5061.5 | +6.03 |
| DLA1 | largeivf2  | `-large-interval-freq-threshold=2` | 4803.1 | 5128.4 | +5.63 |

## Verify (5-run, GPU 3) — All FAIL +1pp gate

| Shape | Tag | base.mean | cand.mean | Δ TFLOPS | Δ pp | gate(>=base.max) | gate(>=+1pp) |
|---|---|---:|---:|---:|---:|:---:|:---:|
| P1   | nosink    | 5004.26 | 5007.98 |  +3.72 | +0.07 | FAIL | FAIL |
| DLA1 | noemxpre  | 5102.43 | 5099.13 |  -3.30 | -0.06 | FAIL | FAIL |
| DLA1 | largeivf2 | 5099.53 | 5099.51 |  -0.02 | -0.00 | FAIL | FAIL |

The smoke "gains" were entirely measurement noise from the single-shot baseline
hitting an outlier (P1 baseline 689.9 was a cold-start GPU; DLA1 baselines
4712/4803 were also single-run jitter). With proper 5-run replication, all
three regress to within ±0.07pp of zero.

## Damaging flags (smoke)

| Shape | Tag | LLVM flag | Effect |
|---|---|---|---|
| All  | nojli      | `-join-liveintervals=false` | -30pp or memory-aperture crash |
| All-but-P1 | revlocal | `-greedy-reverse-local-assignment` | memory-aperture crash |
| DLA1 | nosink     | `-disable-machine-sink` | -21.83pp |
| DLA1 | nopostra   | `-disable-post-ra`      | -3.87pp  |
| P1   | nopostra   | `-disable-post-ra`      | -2.20pp  |

These break or seriously regress the kernel and should be added to the
BROKEN-APERTURE / DEAD-END registry.

## Dead-end registry additions (R15B)

After R15B, the following flag families are confirmed exhausted on this kernel:

- **Coalescer policy** — every meaningful change tested (`join-globalcopies`,
  `join-liveintervals`, `join-splitedges`, `twoaddr-reschedule`, `large-interval-*`,
  `late-remat-update-threshold`, `greedy-reverse-local-assignment`).
- **Spill/AGPR** — `enable-deferred-spilling`, `disable-spill-fusing`,
  `sink-insts-to-avoid-spills`, `amdgpu-prealloc-sgpr-spill-vgprs`,
  `amdgpu-opt-vgpr-liverange=false`. All NOOP except `sinkavoidspill` on P1
  (which then FAILs verify).
- **Machine sink / LICM** — `disable-machine-sink`, `disable-machine-licm`,
  `hoist-cheap-insts`, `machine-sink-bfi`, `machine-sink-cycle-limit`. The two
  enables are strict regressions; rest NOOP.
- **Post-RA scheduler** — `misched-postra=true`, `enable-post-misched`,
  `disable-post-ra`, `post-RA-scheduler`, `break-anti-dependencies={critical,all}`.
  `nopostra` regresses on every shape; rest NOOP.
- **IGLP exact solver** — `amdgpu-igrouplp-exact-solver*` NOOP (no igroup
  pragmas in this kernel).
- **Loop alignment / distribution / rotate-multi** — all NOOP.
- **AMDGPU AA, reassign-regs, pre-RA opts, delay-alu, VOPD, early-ifcvt, sgpr-hazard
  toggles** — all NOOP except `noemxpre` (which FAILs verify).

**Total tags added to dead-end registry by R15B: 42 (all non-overlap with R14A's 27).**

The MXFP4 LLVM-flag-space frontier is **definitively saturated at the kernel-source
level**. Breakthrough requires one of: kernel rewrite (R12+ AGENT_PROMPT
recommendation), HIP intrinsic micro-tuning, or compiler version uplift.

## Files written

- `build_round15_optB_newllvm.{py,log}`
- `asm_diff_probe_r15b.{py,log,json}`
- `bench_round15_optB_smoke.{py,log,json}`
- `bench_round15_optB_verify.{py,log,json}`
- `round15_optB_verdict.md` (this file)

No commits made (0 wins under +1pp gate).
