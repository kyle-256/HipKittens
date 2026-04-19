# R38 Opt B — Tail-iter prefetch hardening (progress log)

Date: 2026-04-19
Branch: `mxfp4`
Goal: structurally fix the 9 R37 CRASH shapes (HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION)
without regressing the 14 R37 WINs.

## Hypothesis tested

R37 fix-B (fused step34 backport into the legacy `R37_FIX_B` branch) interacts badly with
the R25-C tail-pf-off optimization (`R25C_TAIL_PF_OFF_ITERS > 0`). All 9 CRASH variants
have a `_pfoff{4,14,19}` tag, i.e. R25-C is active. The hypothesis was that the
`if (!_r25c_tail_no_pf)` runtime gate around `emit_pf_tail<0>(pf_*_p)` interacts with
the eagerly-constructed `tile_pf_params` structs at the loop top to leave the
compiler-generated `vmcnt` / `s_barrier` count metadata inconsistent across tail iters.

## Repro on a single CRASH shape

Test shape: `m32768_n4096_k2048` with variant `ts_lgk2_gm6_v12_memc_pfoff4` (CRASH on R37).

Confirmed crash on R37 binary: `R38B_finite_check.py R37` produces
```
  rep 0  finite=0.984214
GPU core dump created: gpucore.1401607
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION
```
i.e. R37 was already producing finite=0.984 (below the 0.995 gate) before the GPU
faulted on rep 1. The "CRASH" label in the R37 leaderboard hides a pre-existing
WRONG_OUTPUT condition.

## Control: R25C disabled

Built a control kernel with `R25C_TAIL_PF_OFF_ITERS=0` (always-emit, no R25C). This
ran 800 reps clean — no fault. **Confirmed: R25C tail-pf-off is the trigger.**

## Fix attempts

### Fix B1 (defer make_pf_params construction inside the gate) — DEAD

Moved the four `make_pf_params(...)` calls from the loop top into the
`if (!_r25c_tail_no_pf)` body, hypothesizing the per-iter struct construction
(~128 bytes/iter of voffsets+lds_addrs) was the source of the scratch-frame
mismatch. Built clean. **Smoke test still crashed** at rep 0–100. The struct
construction was not the issue.

### Fix B2 (always emit prefetches, ignore R25C in R37 path) — STRUCTURAL FIX, PERF REGRESSION

Replaced the gated emit with unconditional emit, with `pf_bt` clamped via the
existing `(bt+2 < k_byte_iters) ? bt+2 : k_byte_iters-1` ternary. Rationale: the
control test already showed always-emit doesn't crash. The only correctness risk
on tail iters is overwriting an in-use LDS slot — but `bt+2 ≥ k_byte_iters` means
no future loop iter consumes that slot (loop bound is `bt+1 < k_byte_iters`).

**Smoke test result on m32768_n4096_k2048**: NO CRASH across 10 reps. finite=0.98–0.99
(below 0.995 gate but stable). Same finite range as R37 produced on its single rep
before crashing, so Fix B2 did not introduce the WRONG_OUTPUT — it merely uncovered it.

## Full 42-shape bench (R38B build, all variants)

Methodology: warmup=200, iters=500, trim=10%, finite>=0.995 gate, 8 GPUs.

| Bucket | R37 | R38B |
|---|---|---|
| WIN  | 14 | 5  |
| LOSE | 0  | 14 |
| WRONG_OUTPUT | 19 | 23 |
| CRASH | 9  | **0** |

### CRASH transition (9 shapes — primary deliverable)

| Shape (MxNxK) | R37 | R38B status | R38B finite | R38B tflops |
|---|---|---|---|---|
| 16384x4096x3072  | CRASH | **WIN**          | 0.9979 | 3500 |
| 32768x6144x2048  | CRASH | **WIN**          | 0.9961 | 3275 |
| 128256x32768x4096| CRASH | LOSE             | 0.9969 | 3935 |
| 32768x4096x2048  | CRASH | WRONG_OUTPUT     | 0.9863 | --- |
| 16384x28672x2048 | CRASH | WRONG_OUTPUT     | 0.9831 | --- |
| 32768x28672x2048 | CRASH | WRONG_OUTPUT     | 0.9904 | --- |
| 4096x32768x6144  | CRASH | WRONG_OUTPUT     | 0.9942 | --- |
| 14336x32768x4096 | CRASH | WRONG_OUTPUT     | 0.9921 | --- |
| 28672x32768x4096 | CRASH | WRONG_OUTPUT     | 0.9919 | --- |

**0/9 still crash. 3/9 have correct output. 6/9 are stable WRONG_OUTPUT with finite
just below the 0.995 gate (0.983–0.994).** None faulted under stress (50+ reps).

### WIN regressions (14 R37 WINs after R38B)

| R37→R38B class | Count | Notes |
|---|---|---|
| WIN→WIN    | 2  | 16384x4096x3072 (CRASH→WIN counted separately), 16384x4096x4096 |
| WIN→LOSE   | 9  | extra prefetch overhead on tail iters costs 8–22% perf |
| WIN→WRONG  | 3  | always-emit broke correctness on 3 R37-correct shapes |

The WIN→WRONG demotions are: 4096x14336x16384 (0.9964→0.9565), 28672x4096x8192
(0.9955→0.9939), 28672x4096x16384 (0.9983→0.9851). Two are right at the gate.

### WRONG→better (silver lining)

| Shape | R37 finite | R38B finite | R38B class |
|---|---|---|---|
| 16384x4096x2048   | 0.6372 | **0.9998** | WIN  |
| 4096x32768x128256 | 0.7613 | 0.9987     | LOSE |

Two shapes that R37 corrupted (because R25-C dropped a needed pf in the fused-step34
path) are now correct under always-emit.

### WRONG→catastrophic (cost)

| Shape | R37 finite | R38B finite |
|---|---|---|
| 4096x4096x32768   | 0.7445 | 0.0218 |
| 4096x6144x32768   | 0.6623 | 0.0329 |
| 4096x28672x32768  | 0.6216 | 0.1461 |
| 4096x128256x32768 | 0.6357 | 0.1507 |
| 14336x4096x32768  | 0.6603 | 0.1728 |

K=32768 + small-M shapes regress catastrophically. Always-emit collides with
something specific to deep-K + persistent-XCD layouts.

## Files produced

- `kernel_mxfp4_gluon_cpp.cpp` — Fix B2 implementation guarded by `R38B_TAIL_FIX` (default OFF)
- `build_R38B.py` — builder; sets `-DR38B_TAIL_FIX=1` and strips the bad scheduler flag
- `bench_all_42_R38B.py` — bench harness pointing at `build_R38B/`, gate finite>=0.995
- `bench_all42_results_R38_optB.json` — full 42-shape results
- `R38B_BENCH_RUN.log` — bench run output
- `R38B_BUILD_MANIFEST.json` — build manifest (30 unique builds, 0 failures)
- `R38B_smoke_one.py`, `R38B_finite_check.py` — single-shape repro / diff harnesses
- `build_R38B/` — 30 .so files (one per (N,K,variant))
