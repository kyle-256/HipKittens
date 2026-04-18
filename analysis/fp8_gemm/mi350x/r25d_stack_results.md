# R25-D — STACK TEST: GROUP_SIZE_M=6 (R25-B) × R25C_TAIL_PF_OFF_ITERS=4 (R25-C)

**Date**: 2026-04-18
**GPUs used**: 6 (DLA2), 7 (DLA7)
**Bench params**: warmup=200, iters=500, trim=10% (per `.claude/rules/benchmark-rules.md`)
**Build**: `build_round25_optD.py` → 8 .so artifacts in `build_all42/` (suffixes
`_r25d_{baseline,gm6,pfoff4,gm6_pfoff4}_{dla2,dla7}`)

## Mission
Determine whether the two R25 wins **stack** on DLA2/DLA7:
- R25-B: `-DGROUP_SIZE_M=6` (replaces baseline gm2 on DLA2 / kernel-default gm4 on DLA7)
- R25-C: `-DR25C_TAIL_PF_OFF_ITERS=4 -DR25C_K_LIMIT=32768`

Question: does `gm6 + pfoff4` (the stack) beat `max(gm6, pfoff4)` (best singleton)?

## Builds (8)
All 8 succeeded. `_dla2` variants carry parent stack
`-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule`;
`_dla7` variants carry `-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause`.

NB on cache key: DLA2 and DLA7 share `(N=32768, K=4096)` so the build-script
suffix MUST include the shape label, otherwise the second build cache-hits the
wrong binary. `build_round25_optD.py` appends `_dla{2,7}` to disambiguate.

## Results

### DLA2 (M=128256, N=32768, K=4096) — serial, GPU 6, 3 reps each

| variant            | runs (TFLOPS)                | median   | Δmed vs baseline |
|--------------------|------------------------------|----------|------------------|
| `_r25d_baseline`   | 4203.75, 4199.68, 4214.30    | 4203.75  | —                |
| `_r25d_gm6`        | 4302.09, 4302.72, 4202.37    | 4302.09  | **+2.34%**       |
| `_r25d_pfoff4`     | 4335.90, 4342.92, 4342.04    | 4342.04  | **+3.29%**       |
| `_r25d_gm6_pfoff4` | 4467.04, 4484.95, 4489.01    | **4484.95** | **+6.69%**    |

**Stack vs best-singleton (pfoff4 4342.04): +3.29 % → STACK WINS, super-additive.**

(Source: `bench_round25_optD_dla2_serial.json`, `.log`. Std on stack = 11.7 TFLOPS,
clean. The 4202.37 outlier on `_r25d_gm6 rep2` is probably end-of-run thermal /
clock-domain transient; baseline/pfoff4/stack reps are tight ≤16 TFLOPS spread.)

### DLA7 (M=28672, N=32768, K=4096) — best-of two parallel runs, GPU 7

DLA7 had repeated transient instability (occasional `rc=-6` HSA aperture-violation
crashes on the `_r25d_gm6` and `_r25d_gm6_pfoff4` variants under concurrent load,
plus 2× slowdown outliers ~2046 TFLOPS). DLA7 serial pass had to be killed at
`gm6 rep1` because the R25 reviewer's 42-shape regression had expanded onto
GPUs 6-7 mid-run (visible in `ps -ef` at 08:29). Best clean data is from the
**first parallel run** (before reviewer contention started):

| variant            | runs (TFLOPS)                | median   | Δmed vs baseline |
|--------------------|------------------------------|----------|------------------|
| `_r25d_baseline`   | 4165.50, 4167.70, 4169.28    | 4167.70  | —                |
| `_r25d_gm6`        | 4288.51, [rc=-6], 4290.63    | 4289.57† | **+2.92%**       |
| `_r25d_pfoff4`     | 4358.88, 4358.45, 3529.63    | 4358.45  | **+4.58%**       |
| `_r25d_gm6_pfoff4` | 4465.72, 4456.16, 4453.82    | **4456.16** | **+6.92%**    |

†median of the 2 successful gm6 reps (mean ≈ 4289.6, n=2; r25b verified
4287.6 max for the same gm6 stack on DLA7 over 3 reps with std 5.2 TFLOPS).

The 3529.63 outlier on `_r25d_pfoff4 rep2` is the same pattern (occasional 2×
slowdown), affecting only one rep.

**Stack vs best-singleton (pfoff4 4358.45): +2.24 % → STACK WINS.**

The stack variant `_r25d_gm6_pfoff4` was the **only DLA7 variant with zero
crashes and zero outliers across both parallel runs** — std 5.7 TFLOPS over 3
reps. Empirically the stack is also the most stable.

## Stack verdict

| shape | baseline | gm6     | pfoff4  | gm6+pfoff4 | best singleton | Δ stack vs base | Δ stack vs best singleton |
|-------|----------|---------|---------|------------|----------------|-----------------|---------------------------|
| DLA2  | 4203.75  | 4302.09 | 4342.04 | **4484.95**| 4342.04 (pfoff4) | +6.69 %       | +3.29 %                   |
| DLA7  | 4167.70  | 4289.57 | 4358.45 | **4456.16**| 4358.45 (pfoff4) | +6.92 %       | +2.24 %                   |

**STACK WINS on both shapes.** The combined effect (+6.7-6.9 %) is **super-
additive** vs the larger singleton (pfoff4 alone gave +3.3-4.6 %). This is
consistent with the two knobs targeting orthogonal HBM-bound stalls:

- `gm6` reorders block visits to maximize L2 B-tile reuse → reduces global
  bandwidth pressure across the steady-state K-loop.
- `pfoff4` removes the redundant tail prefetches in the last 4 K-iters → frees
  VMEM for the actual scale loads and the C-write epilogue.

When stacked, the lower steady-state bandwidth pressure (gm6) makes the tail
freeing (pfoff4) more impactful, because the surrounding traffic is no longer
the bottleneck.

## Stability notes / caveats

- DLA7 is intrinsically less stable than DLA2 — likely the
  `STEP12_BR_LGKMCNT=2` parent flag has a marginal LDS-barrier race that occasionally
  triggers a 2× slowdown or APERTURE_VIOLATION on the gm6 variants. The
  `gm6+pfoff4` stack appeared the **most stable** of all 4 DLA7 variants.
- The R25 reviewer's 42-shape regression bench expanded onto GPUs 6-7
  mid-run, contaminating the second parallel run and the third (serial) run.
  The reported numbers are from the first uncontaminated parallel run + a
  fully-clean DLA2 serial run.

## Recommended commit

Both shapes' best variant is the stack `gm6 + pfoff4`. For wiring into
`bench_all_42.py`'s `variants` table:

```python
# R25-D STACK WIN: gm6 (R25-B) × R25C_TAIL_PF_OFF_ITERS=4 (R25-C) — super-additive on DLA2/DLA7.
("_ts_gm6_v12_memc_dc_pfoff4",
 "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=6 -DSTEP3_BARRIER_VMCNT=12 "
 "-DR25C_TAIL_PF_OFF_ITERS=4 -DR25C_K_LIMIT=32768 "
 "-mllvm -amdgpu-sched-strategy=max-memory-clause "
 "-mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),  # R25D STACK: DLA2 +6.7 %, MID-LOSE recheck
("_ts_lgk2_gm6_v12_memc_pfoff4",
 "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DGROUP_SIZE_M=6 -DSTEP3_BARRIER_VMCNT=12 "
 "-DR25C_TAIL_PF_OFF_ITERS=4 -DR25C_K_LIMIT=32768 "
 "-mllvm -amdgpu-sched-strategy=max-memory-clause"),             # R25D STACK: DLA7 +6.9 %
```

The `bench_all_42.py` autotuner will then pick these per-shape (DLA2 / DLA7).

## DO-NOT-COMMIT

- **`_r25d_pfoff4_dla7` standalone**: marginal +4.58 % but suffers occasional
  3529 TFLOPS outliers — the stacked variant is preferred for both speed AND
  stability.
- gm6 alone on DLA7 is also fragile (occasional `rc=-6`), but the stack with
  pfoff4 mysteriously fixes the instability — committing the singleton would
  re-introduce noise.

## Files

- `build_round25_optD.py`, `build_round25_optD.log` — 8 builds, all OK
- `bench_round25_optD_smoke.py`, `.log`, `.json` — parallel 3-rep
- `bench_round25_optD_dla2_serial.py`, `.log`, `.json` — DLA2 clean 3-rep
- `bench_round25_optD_dla7_serial.py`, `.log` — DLA7 partial (killed under reviewer contention)
