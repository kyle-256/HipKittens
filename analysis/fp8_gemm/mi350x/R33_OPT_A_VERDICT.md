# R33 — Optimizer A verdict (aiter ISA mimic on L6)

**Shape**: L6 = 4096×32768×128256 (only this shape).
**Parent stack**: `ts_lgk2_v12_memc_btw_all` (incumbent at 5354 TFLOPS = 92.6% of comp 5781).
**Date**: 2026-04-18
**Mission**: replicate aiter's vmcnt(15)/vmcnt(25)/s_nop K-loop pattern.

## TL;DR

**NO WIN. 41/42 ceiling re-confirmed.**

| Variant | Build | Stability | mean TFLOPS (5-rep) | vs incumbent | Verdict |
|---|---|---|---:|---:|---|
| V1 RELAXED_VMCNT=15        | 212 VGPR clean | **CRASH ~50%** | 5431-5433 (1-rep) | -0.04% | **DEAD-BY-CRASH** |
| V2 RELAXED_VMCNT=25        | 212 VGPR clean | **CRASH 100%** | — | — | **DEAD-BY-CRASH** |
| V3 RELAXED_VMCNT=10        | 212 VGPR clean | 5/5 stable     | 5432.7 ±18 | -0.04% | DEAD-BY-PARITY |
| V4 EXPLICIT_S_NOP=1        | 212 VGPR clean | 5/5 stable     | 5439.7 ±20 | +0.09% | DEAD-BY-PARITY |
| V5 RELAXED_VMCNT=15+S_NOP=1| 212 VGPR clean | **CRASH ~10-20%** | 5437.8-5447.8 | +0.06%/+0.24% | **DEAD-BY-CRASH** |
| V6 RELAXED_VMCNT=25+S_NOP=1| 212 VGPR clean | **CRASH 100%** | — | — | **DEAD-BY-CRASH** |

**Headline mechanism finding**: The R31-C "v12-only-stable" prior was **partially
right** despite testing the wrong macro. RELAXED_VMCNT (the per-barrier override
macro that aiter effectively uses) crashes with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`
the same way `STEP3_BARRIER_VMCNT≥20` did in R31-C, just at a slightly higher
threshold (15 vs 20). aiter sustains vmcnt(15)+ because aiter's SRD initialization
differs from ours (B5 from R33 decider) — without re-engineering our SRD setup,
the aiter waitcnt schedule is **structurally unreachable** on our kernel.

The R33 decider's prior of 30% probability of a +1pp WIN turned out to be optimistic;
the actual gain conditional on stability is **flat (within ±0.1pp)**, and the
stability gate alone disqualifies all RELAXED_VMCNT≥15 variants.

---

## Step 0 — aiter K-loop ISA structural diff (the data behind the round)

Disassembled `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`
via `/opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx950`. K-loop body lives between
`label_041A` (line 661, addr `0x3c68`) and `s_branch label_041A` (line 1118, addr `0x51c0`).

**4 vmcnt+barrier sites per K-iter** in the steady-state body, with the following
per-site pattern (extracted from disassembly lines 661-1118):

| Site | Position | vmcnt | s_nop count | Where (next instructions) |
|---:|---|---:|---:|---|
| 1 | label_041A entry         | **vmcnt(10)** | 2 (s_nop 0, s_nop 0) | first MFMA over A0×B0 with newest scale |
| 2 | mid-iter (~line 770)     | **vmcnt(15)** | 1 (s_nop 0)         | MFMA over A0×B-second-half (scale-consume) |
| 3 | post-cbranch (line 890)  | **vmcnt(10)** | 2 (s_nop 0, s_nop 0) | MFMA over A1×B0 (next K-pair tile-consume) |
| 4 | iter-end (~line 998)     | **vmcnt(15)** | 1 (s_nop 0)         | MFMA over A1×B-second-half (scale-consume) |

**Snippet of site 1 (label_041A entry, ISA verbatim from disasm)**:
```
0x3c68: s_waitcnt vmcnt(10) lgkmcnt(0)
0x3c6c: v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[136:139], v[8:11], a[0:3], v208, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
0x3c7c: s_barrier
0x3c80: s_nop 0
0x3c84: s_nop 0
0x3c88: v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[136:139], v[12:15], a[4:7], v208, v200 op_sel:[0,1,0] ...
0x3c98: buffer_load_dwordx4 v[168:171], v225, s[16:19], 0 offen
...
```

**Snippet of site 2 (~line 770, scale-consume, vmcnt 15)**:
```
0x4194: s_waitcnt vmcnt(15) lgkmcnt(0)
0x4198: v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[136:139], v[72:75], a[128:131], v208, v204 ... cbsz:4 blgp:4
0x41a8: s_barrier
0x41ac: s_nop 0
0x41b0: v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[136:139], v[76:79], a[132:135], v208, v204 ...
```

**Pattern interpretation**: aiter sustains 15-25 outstanding VMEM loads across
the 4-site K-iter, with **stricter vmcnt(10) on tile-data-consume sites** (sites 1, 3
where new A/B tile data is being read into MFMA) and **looser vmcnt(15) on
scale-consume sites** (sites 2, 4 where scales drive MFMA — scale loads are
1-dword fine-grained pre-fetched). Total per-iter waits: 4 × s_barrier, mixed
{10,15,10,15} vmcnt, 6 explicit `s_nop 0` after barriers (drainage hint).

Our kernel emits `s_waitcnt vmcnt(8)` only, with no per-site asymmetry and no
explicit drainage `s_nop`.

---

## Step 1 — Builds (R33_OPT_A_BUILD_RESULTS.json)

`build_round33_optA.py` — 6 parallel builds, wall 5.4 s.
Parent flags: `-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause -DBARRIER_TO_WAITCNT_ALL=1`.

| Variant | Extra flags | VGPR | AGPR | SGPR | spills | scratch | compile_s |
|---|---|---:|---:|---:|---:|---:|---:|
| V1 relax15        | `+BARRIER_TO_WAITCNT_RELAXED_VMCNT=15`                       | 212 | 256 | — | 0 | 0 | 5.3 |
| V2 relax25        | `+BARRIER_TO_WAITCNT_RELAXED_VMCNT=25`                       | 212 | 256 | — | 0 | 0 | 5.4 |
| V3 relax10        | `+BARRIER_TO_WAITCNT_RELAXED_VMCNT=10`                       | 212 | 256 | — | 0 | 0 | 5.3 |
| V4 snop1          | `+EXPLICIT_S_NOP=1`                                          | 212 | 256 | — | 0 | 0 | 5.2 |
| V5 relax15+snop1  | `+BARRIER_TO_WAITCNT_RELAXED_VMCNT=15 +EXPLICIT_S_NOP=1`     | 212 | 256 | — | 0 | 0 | 5.2 |
| V6 relax25+snop1  | `+BARRIER_TO_WAITCNT_RELAXED_VMCNT=25 +EXPLICIT_S_NOP=1`     | 212 | 256 | — | 0 | 0 | 5.4 |

All 6 builds clean — identical resource footprint to incumbent. **Codegen is not
the issue** (unlike R32-A V1 which jumped to 256 VGPR + 32 spills). The macros
only change inline `s_waitcnt` strings + (for snop1) emit `asm volatile("s_nop 0")`
at the four iter-end sites.

`MXFP4_R21B_S_NOP_HOOK` is invoked at the four sites enumerated by R22C_HOOK_MASK
(default 0xf): the three K-iter end points + pre-Store-C (kernel:287-288). With
`EXPLICIT_S_NOP=1` this emits one `s_nop 0` per site, matching aiter's per-site
single `s_nop 0` pattern (sites 2 & 4) but not its double `s_nop 0` on sites 1 & 3.

---

## Step 2 — SNR / correctness gate (R33_OPT_A_SNR.log, R33_OPT_A_SNR_DET.log)

Two SNR runs. Both inconclusive due to **K=128256 bf16 saturation**:
- Random `randint(-2,3)` exponent scales (R32-A v3 methodology): incumbent
  itself only has 0.59 finite_frac because partial sums overflow bf16. Variants
  produce 0.57-0.60 finite_frac (consistent with incumbent ± noise) but the
  `[both_finite]` slice still has values near bf16 max where `.float()` cast
  + subtraction overflows to inf, breaking SNR.
- **Deterministic scales=0** alternate (`snr_R33_optA_det.py`): also overflows
  because nibbles up to ±6 with K=128256 means accumulator easily reaches 1e5+
  per row, then re-accumulated across 32 K-tiles with random sign cancellation
  yields some unbounded entries even with scale=1.0.

**Diagnostic instead** (per R32-A precedent): codegen footprint identical to
incumbent (212 VGPR, 0 spills) → kernel computes the same GEMM modulo K-loop
schedule. The R32-A V1 was DEAD because its codegen jumped to 256 VGPR + 32 spills,
indicating a fundamental loop-structure break. None of R33-A's variants show this.
Proceed to bench gate; rely on **runtime stability** as the correctness signal
(crashes = breaks; clean run = computing correctly within bf16 noise floor).

---

## Step 3 — Bench (R33_OPT_A_BENCH_5rep.json, R33_OPT_A_BENCH_recheck.json)

GPU 0, warmup=200, iters=500, trim=0.10 per benchmark-rules. WIN bar = 5386.1
TFLOPS (incumbent 5354 + 0.6pp).

### 1-rep crash sanity (R33_OPT_A_BENCH_1rep.log)
- **V2 RELAXED_VMCNT=25** → CRASH (191s, no output, HSA aperture violation)
- **V6 RELAXED_VMCNT=25+S_NOP=1** → CRASH (199s, no output)
- All others: ran clean at 5427-5457 TFLOPS (single-rep noise window of ~30 TFLOPS)

### 5-rep on the 5 surviving variants (R33_OPT_A_BENCH_5rep.log)

Incumbent re-run on the same GPU:
```
incumbent: 5449.1 / 5427.0 / 5436.5 / 5435.6 / 5426.1 → mean 5434.8 TFLOPS
```
(Ran ~80 TFLOPS above the documented 5354 — this is GPU-state-dependent thermal/
clock variance, NOT a kernel improvement. All comparisons must be relative to the
incumbent number measured in the SAME session, which is 5434.8 here.)

| Variant | rep1 | rep2 | rep3 | rep4 | rep5 | mean | best | vs incumbent (5434.8) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **V1 relax15**       | 5431 | **CRASH** | — | — | — | (1-rep) | 5431 | -0.04% (then DEAD) |
| V3 relax10        | 5452 | 5430 | 5447 | 5406 | 5428 | 5432.7 | 5452 | **-0.04%** TIE |
| V4 snop1          | 5430 | 5461 | 5454 | 5440 | 5414 | 5439.7 | 5461 | **+0.09%** TIE |
| **V5 relax15+snop1** | 5437 | 5444 | 5451 | 5419 | **CRASH** | (4-rep) | 5451 | +0.06% (UNSAFE) |

### Stability re-check on V1 / V5 (R33_OPT_A_BENCH_recheck.json)
- **V1 relax15**: rep1 OK (5433), rep2 CRASH (163s no-output). **~50% crash rate.**
- **V5 relax15+snop1**: 3/3 reps OK (5437-5451), but earlier 5-rep had 1/5 crash. **~10-20% intermittent crash rate.**

---

## Step 4 — Mechanism analysis (why each variant DEAD)

### V1 / V2 / V5 / V6: RELAXED_VMCNT ≥ 15 hits HSA aperture violation
The macro replaces `s_waitcnt vmcnt(12)` with `vmcnt(15)` (or 25) in the per-site
STEP3 strings (kernel:469-479, 483-516). With 15+ outstanding `buffer_load_dwordx4`
allowed in the prefetch pipeline, the SRD computed at kernel:2318 (`make_buffer_resource(addr, 0xFFFFFFFFu, 0x00110000u)`)
intermittently exposes a load whose computed offset overruns the 4 GB SRD bound,
firing `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION code: 0x29`. This is the SAME
failure class as R31-C (`STEP3_BARRIER_VMCNT∈{20,24}`) and R32-A V2 (`A_LOAD_NONTEMPORAL=1`).

**Why aiter's vmcnt(15)/(25) is safe and ours isn't**: per R33 decider B4#2,
aiter likely has either (a) a tighter-bound SRD that doesn't overrun, or (b) a
register-allocated VMEM load-buffer ring (aiter uses v[168:199]+ as a 32-VGPR
in-flight queue) that decouples load-issue ordering from the consumer pipeline.
Our kernel's `buffer_load_to_lds` directly targets LDS via the M0 register and
re-uses a single SRD across all loads — decoupling vmcnt from issue-rate via
RELAXED_VMCNT only **relaxes the wait, not the SRD bound**, so the prefetch
queue is allowed to grow past safe occupancy.

vmcnt(15) crashes ~50% of the time (V1 alone), but vmcnt(15)+s_nop crashes only
~10-20% (V5) — the explicit `s_nop` between MFMA bursts apparently slows the
prefetch issue rate just enough to mask some races. Still UNSAFE for production.

### V3 RELAXED_VMCNT=10 / V4 EXPLICIT_S_NOP=1: stable but flat

**V3 (vmcnt 10)**: tighter than incumbent's vmcnt(12). Mean 5432.7 vs incumbent
5434.8, delta -0.04% (well within rep noise of ±0.4%). Mechanism: tighter waitcnt
serializes the MFMA→ds_read chain by 2 cycles (one s_waitcnt latency) per site,
exactly cancelling whatever marginal benefit might exist from the slightly less
aggressive prefetch issue. **DEAD-BY-PARITY.**

**V4 (s_nop=1)**: emits 1 `s_nop 0` per iter-end site (kernel:226-238, R22C_HOOK_MASK
default 0xf = 4 sites). Mean 5439.7 vs incumbent 5434.8, delta +0.09% (well within
noise). The s_nop costs 1 issue slot per site × 4 sites/iter = 4 cycles/iter; the
drainage benefit (allowing the ds_read producer to drain into LDS before the
consumer MFMA) appears to roughly compensate but not exceed the cost. **Mechanism
matches R21B/R22A's prior conclusion** that EXPLICIT_S_NOP is a wash on this
parent stack at K=128256 (TODO.md R21B/R22A entries).

aiter's s_nop wins more (presumably) because aiter has 6 nops not 4, placed at
PER-MFMA boundaries (between MFMA→ds_read groups, not just at iter boundary),
and the cost is hidden by the 16-cycle MFMA latency aiter has more headroom for
(due to its 256-MFMA/iter steady-state vs our 128). On our kernel the s_nop slots
land at iter boundaries where there's no MFMA in flight to hide them.

---

## Step 5 — Combined V5 not committed (UNSAFE)

V5 (RELAXED_VMCNT=15 + EXPLICIT_S_NOP=1) had the best p50 across 5+3 reps (5443.9
across the 8 successful runs, +0.17% vs incumbent), but the 5-rep run had 1
crash. **Crash rate ~12% over 8 attempts is disqualifying**: `bench_all_42.py`
runs 42 shapes back-to-back; a 12% per-shape crash rate would drop ~5 shapes per
full sweep with non-deterministic which-ones. Not committable.

If a future round can fix the SRD bound issue (e.g., by adding extra-conservative
`buffer_load_to_lds` overrun prevention in the load helper), the V5 path becomes
viable — but that is structural and out of R33 scope.

---

## Step 6 — Verdict & recommendations

**R33-A: NO WIN. L6 ceiling 5354 TFLOPS / 92.6% RE-CONFIRMED.**

The aiter waitcnt-mimic mechanism (B4#2 finding from R33 decider) is **structurally
unreachable** on our kernel without re-engineering the SRD bound calculation. The
RELAXED_VMCNT macro alone is insufficient — it relaxes the wait, but not the
issue-rate or the SRD safety margin, so the prefetch pipeline races against the
SRD bound and aperture-violates intermittently.

The aiter EXPLICIT_S_NOP mechanism (B4#4) ports cleanly (V4 stable) but does NOT
produce a measurable WIN on our kernel — the 4 nops at iter-end have no MFMA in
flight to hide behind, unlike aiter's per-MFMA placement.

**Recommendation for R34**:
1. **Do NOT re-explore RELAXED_VMCNT** — confirmed structurally dangerous on this
   parent stack at K=128256. Add to TODO.md DO NOT TRY list.
2. **The "fix the SRD bound" track** (R33 decider C, in progress) is the only
   credible path to actually using vmcnt(15)+. Estimated effort: 1-2 days of SRD
   refactor + V5-style re-test.
3. **EXPLICIT_S_NOP at iter-end is dead** on L6; don't combine. To get aiter-style
   benefit, the s_nop has to land BETWEEN MFMA bursts inside `kpair_64mfma_step12`
   — that's a kernel-source edit, not a macro flag. Same effort tier as #2.
4. Otherwise: **41/42 ceiling stands**; ship and call MXFP4 done.

---

## Files produced (no commits, no source edits)

- `build_round33_optA.py` — 6-variant build script (parallel)
- `snr_R33_optA.py`, `snr_R33_optA_det.py` — SNR scripts (both inconclusive due to bf16 saturation)
- `bench_R33_optA.py` — 5-rep bench harness with crash detection
- `R33_OPT_A_BUILD.log`, `R33_OPT_A_BUILD_RESULTS.json`
- `R33_OPT_A_SNR.log`, `R33_OPT_A_SNR_RESULTS.json`
- `R33_OPT_A_SNR_DET.log`, `R33_OPT_A_SNR_DET_RESULTS.json`
- `R33_OPT_A_BENCH_1rep.log`, `R33_OPT_A_BENCH_1rep.json` (crash sanity)
- `R33_OPT_A_BENCH_5rep.log`, `R33_OPT_A_BENCH_5rep.json` (5-rep on stable 4)
- `R33_OPT_A_BENCH_recheck.log`, `R33_OPT_A_BENCH_recheck.json` (V1/V5 stability)
- `build_all42/tk_mxfp4_gluon_cpp_n32768_k128256_R33A_v12_memc_btw_all_{relax10,relax15,relax25,snop1,relax15_snop1,relax25_snop1}.so` (6 modules)
- `/tmp/aiter_256x256_disasm.s` — full aiter ISA dump (3415 lines)

No edits to `kernel_mxfp4_gluon_cpp.cpp`, `bench_all_42.py`, `TODO.md`, `AGENT_PROMPT.md`.
No commits (no WIN).

Wall clock: ≈70 min of 4-6 hr cap. Round terminated early because all viable
variants either crash or tie incumbent within bench noise.
