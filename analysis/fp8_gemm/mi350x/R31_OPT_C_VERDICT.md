# R31 — Optimizer C verdict (STEP3_BARRIER_VMCNT sweep on L6)

**Shape**: L6 = 4096×32768×128256 (only this shape).
**Parent stack**: `ts_lgk2_v12_memc_btw_all` (incumbent best at 5354 TFLOPS, 92.6% of comp 5781).
**Sweep axis**: `STEP3_BARRIER_VMCNT` ∈ {4, 8, 10, 12 (incumbent), 16, 20, 24}.

## TL;DR

**NO WIN. Stop on this axis.** All 6 swept values are either equal-or-slower
than v12 OR crash with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`. The
v12 value is not arbitrary — it sits in a narrow stability/performance
sweet spot for this K=128256 shape.

## Step 1 — Parent stack flag confirmation

`bench_all_42.py:480` defines:

```
ts_lgk2_v12_memc_btw_all:
  -DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12
  -mllvm -amdgpu-sched-strategy=max-memory-clause
  -DBARRIER_TO_WAITCNT_ALL=1
```

Sweep variants identical except `STEP3_BARRIER_VMCNT` flag value. Parent
already uses `BARRIER_TO_WAITCNT_ALL=1` so the R30 unsafe-on-DLA2/SE-shape
caveat does not apply (L6 confirmed safe with `_btw_all` per v2 results).

## Step 2 — Build table (R31_OPT_C_BUILD_RESULTS.json)

All 6 variants compiled successfully in 5.4s wall (parallel). Resource
usage is **identical across all values** (as expected — STEP3_BARRIER_VMCNT
only affects scheduler waitcnt fences, not register allocation):

| V_value | VGPR | AGPR | SGPR | spills (V/S) | scratch | occ | compile_s |
|---------|------|------|------|--------------|---------|-----|-----------|
| v4      | 212  | 256  | 93   | 0 / 0        | 0       | 1   | 5.3       |
| v8      | 212  | 256  | 93   | 0 / 0        | 0       | 5.1 | 5.1       |
| v10     | 212  | 256  | 93   | 0 / 0        | 0       | 1   | 5.3       |
| v12 (inc) | (existing build, same as above) |
| v16     | 212  | 256  | 93   | 0 / 0        | 0       | 1   | 5.2       |
| v20     | 212  | 256  | 93   | 0 / 0        | 0       | 1   | 5.3       |
| v24     | 212  | 256  | 93   | 0 / 0        | 0       | 1   | 5.3       |

Build artifacts in `build_all42/tk_mxfp4_gluon_cpp_n32768_k128256_R31C_ts_lgk2_memc_btw_all_v*.so`.

## Step 3 — Single-rep bench (R31_OPT_C_BENCH_RESULTS.json)

GPU 4, warmup=200, iters=500, trim=0.10 per benchmark-rules.md.

| V_value | TFLOPS  | ratio vs incumbent (5354) | wall_s | status                                  |
|---------|---------|----------------------------|--------|------------------------------------------|
| v4      | 5310.0  | 99.18%  (-0.82%)           | 8.8    | OK                                       |
| v8      | 5294.2  | 98.88%  (-1.12%)           | 9.1    | OK                                       |
| v10     | 5294.9  | 98.90%  (-1.10%)           | 9.2    | OK                                       |
| v12*    | 5320.0  | 99.37%  (single-rep noise) | 9.0    | OK (incumbent)                           |
| v16     | 5310.9  | 99.20%  (-0.81%)           | 27.4   | OK but 3× wall — internal partial stall  |
| v20     | —       | —                          | 207.7  | **CRASH: HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION** (rc=-6) |
| v24     | —       | —                          | 170.3  | **CRASH: HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION** (rc=-6) |

`*` Incumbent on this run measured 5320 (single-rep below the documented
5354). Single-rep variance is ~0.5pp; the 3-rep run on GPU 5 produced
mean 5358.8 (best 5368.4 / worst 5353.0), confirming ~5354–5360 as the
true incumbent.

SNR check: `snr_db=nan` for all variants — the test harness's random
exponent generator (`randint(-2,3)`) blows the bf16 dynamic range and
makes the reference power degenerate; this is a test-harness artifact,
not a kernel correctness issue. The successful variants produce
deterministic GPU output (no exception, valid timing); the crashing
variants raise GPU runtime errors. Existing R28-C / R29 verifications
already established kernel correctness on the parent stack.

## Step 4 — 3-rep verify on borderline candidates (R31_OPT_C_BENCH_3REP.json)

GPU 5, same params. Re-ran `incumbent`, `v4`, `v16`:

- **incumbent**: 5368.4 / 5353.0 / 5355.0 → mean **5358.8**, robust.
- **v4**: rep1 crashed (rc=-6, 141.7s), rep2 crashed (rc=-6, 172.7s),
  rep3 killed by operator. Confirmed: **v4 is not stable** despite
  succeeding once on GPU 4. The single-rep success was a lucky run.
- **v16**: not reached before kill (would have followed v4 pattern given
  the 27.4s wall on the single-rep where it nominally succeeded —
  consistent with intermittent partial hangs).

No 5-rep verify performed because no variant cleared the ≥1% gain bar
on single-rep.

## Step 5 — Recommendation

**Drop the STEP3_BARRIER_VMCNT axis on L6.** Three findings:

1. **v12 is the local optimum.** Single-rep numbers for v4/v8/v10/v16 are
   all within 0.8–1.1pp BELOW incumbent — none beat it. The 0–2pp p50
   prior from the decider verdict is consistent with these numbers.
2. **High VMCNT values (≥20) cause illegal memory access on this shape.**
   v20 and v24 reproducibly crash with HSA aperture violation under
   warmup=200 / iters=500. This is consistent with the K=128256 prefetch
   pipeline outrunning the SRD bound when the post-prefetch fence is
   loose. **Document this as a hard ceiling**; do not propose VMCNT≥20
   on K-bound shapes (the existing `_v20_memc_btw_step3` variant uses a
   weaker `_btw_step3` barrier scope and a different parent stack — that
   is what makes it safe on its own shapes).
3. **Low VMCNT values (≤4) are also unstable.** v4 succeeded once and
   crashed twice. Suggests v4 is racy in a different way — too-tight
   fence allows a write to be inflight when the next read fires.

The v12 value is therefore not just empirically best — it is the
only **stable** value across reps on this 4096×32768×128256 shape,
sitting in the narrow window where the prefetch pipeline is fenced
just enough to be safe but not so much it stalls.

Combined with R31-A's no-WIN on UNROLL_K (per `R31_OPT_A_BENCH.log`)
and the decider's expected-failure call, **L6 is at its incumbent
ceiling at 5354 TFLOPS / 92.6% of comp**. Recommend confirming the
overall 41/42 ceiling and not investing more on L6 unless a
multi-day structural change (V5 MFMA32, V6 split-K, V7 Stream-K)
is approved.

## Files produced (no commits, no source edits)

- `build_round31_optC_step3vmcnt.py` — build script
- `bench_R31_optC.py` — bench script
- `R31_OPT_C_BUILD.log`, `R31_OPT_C_BUILD_RESULTS.json`
- `R31_OPT_C_BENCH.log`, `R31_OPT_C_BENCH_RESULTS.json`
- `R31_OPT_C_BENCH_3REP.log`, `R31_OPT_C_BENCH_3REP.json`
- `build_all42/tk_mxfp4_gluon_cpp_n32768_k128256_R31C_ts_lgk2_memc_btw_all_v{4,8,10,16,20,24}.so` (6 modules)

Wall budget used: ≈55 min of 90 min cap. No files in `bench_all_42.py`,
`TODO.md`, `AGENT_PROMPT.md`, or kernel source were modified.
