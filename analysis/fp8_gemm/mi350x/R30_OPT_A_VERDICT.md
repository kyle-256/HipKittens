# R30 OPT A — Cross-shape transplant scout VERDICT

Date: 2026-04-18  Author: R30-Opt-A (Opus 4.7)
Mission: validate the 5 "free win" variants flagged in R30_DECIDER_VERDICT Q2.

**Verdict: ALL 5 CANDIDATES DEAD — HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION
(GPU fault) at the target shapes after ~50–100 kernel invocations.**

The decider's "missing per_variant entry → free transplant opportunity" framing
was wrong: those entries are missing because the parallel-bench harness silently
drops crashed variants (`bench_all42_parallel_R25_FINAL.py:263-267` returns
`None` on `rc != 0`). All 3 candidate variants use `BARRIER_TO_WAITCNT` macros
which `bench_all_42.py:462-467` flags as **"CORRECTNESS-RISKY — only safe on
shapes where parent SNR is high"**. At these specific (M, N, K) the cross-wave
barriers ARE load-bearing and removing them produces an out-of-bounds VMEM access.

Total wall: ~14 min (4.9s build + ~4 min single-rep crashes + ~2 min reproducer).

---

## 1. Build status (5/5 PASS)

Builds use proper `-DM_DIM`, `-DN_DIM`, `-DK_DIM`, parent variant flags, no
K_EXACT. SO suffix `_r30oa_m{M}` to avoid colliding with the canonical
default-M=8192 SOs in `build_all42/`.

| Variant | Target (M×N×K) | Status | Compile (s) | Occ | VGPR/AGPR/Spills |
|---|---|:---:|---:|---:|---|
| ts_lgk2_memc_btw_all       | 128256×32768×4096  | PASS | 4.8 | 1 | (not parsed; -Rpass off here) |
| ts_lgk2_v12_memc_btw_all   | 128256×32768×4096  | PASS | 4.8 | 1 | — |
| v20_memc_btw_step3         | 128256×32768×4096  | PASS | 4.7 | 1 | — |
| ts_lgk2_memc_btw_all       | 32768×14336×2048   | PASS | 4.6 | 1 | — |
| v20_memc_btw_step3         | 32768×14336×2048   | PASS | 4.6 | 1 | — |

Detail: `R30_OPT_A_BUILD.json`, logs in `build_all42/compile_R30OA_*.log`.

---

## 2. Bench results (single-rep, warmup=200 iters=500 trim=0.10) — ALL FAULT

5/5 errored with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` (rc=-6) before
the timed iters loop completed. Parallel run on GPUs 0–4 (verified idle).

| Label                          | Shape (M×N×K)       | GPU | TFLOPS | Verdict |
|---|---|---:|---:|---|
| DLA2_lgk2_memc_btw_all         | 128256×32768×4096   | 0   |   —   | **APERTURE_FAULT** |
| DLA2_lgk2_v12_memc_btw_all     | 128256×32768×4096   | 1   |   —   | **APERTURE_FAULT** |
| DLA2_v20_memc_btw_step3        | 128256×32768×4096   | 2   |   —   | **APERTURE_FAULT** |
| S5L_lgk2_memc_btw_all          | 32768×14336×2048    | 3   |   —   | **APERTURE_FAULT** |
| S5L_v20_memc_btw_step3         | 32768×14336×2048    | 4   |   —   | **APERTURE_FAULT** |

All variants survive a one-shot call with finite output (finite_frac ≈ 0.50,
matches parent harness behaviour). The fault is iteration-count-dependent: the
S5L `lgk2_memc_btw_all` kernel completes 50 iters cleanly then faults during
the 50→80 iter range; the S5L `v20_memc_btw_step3` faults between 10 and 30
iters. This is the classic late-iteration cross-wave race that the
BARRIER_TO_WAITCNT comments warn about: removing the inner-loop `s_barrier`
allows successive K-iters to overlap incorrectly under pressure, and after
enough iters a wavefront reads/writes outside the SRD aperture.

| Variant on (M=32768, N=14336, K=2048) | OK at | First FAULT |
|---|---|---|
| ts_lgk2_memc_btw_all   | 50 iters | between 50–80 iters |
| v20_memc_btw_step3     | 10 iters | between 10–30 iters |

(DLA2 reproducer not separately bisected — variant fault confirmed end-to-end
in the bench run with full 200-warmup; reproducer would need ~3 min/run.)

Detail: `R30_OPT_A_BENCH.json`, `R30_OPT_A_BENCH.log`.

---

## 3. 5-rep verify

**N/A — no variant produced a TFLOPS measurement, so nothing to verify.**

---

## 4. Recommendation

**DO NOT add any of these 5 variants to bench_all_42.py for these target shapes.**
They will crash the GPU when wired in. The parallel-bench harness silently
swallows the crash (returns `None`), so they would simply re-appear as
"missing" in subsequent bench JSONs — wasting ~2 min/shape × 5 shapes per
run with no gain.

**Underlying lesson for the decider:** "missing per_variant entry" ≠ "free
transplant candidate". Before recommending a transplant, the decider must
either (a) confirm the variant ran on at least one shape with similar
M/N/K geometry, or (b) flag any BARRIER_TO_WAITCNT-bearing variant as
correctness-suspect and require an aperture probe. This R30 OPT A run
disconfirms 3 of the 5 R30 Q2 candidates and provides no path to lift the
DLA2 (109%) or S5L (111.2%) ratios via transplant.

**Path forward for these two shapes:**
- DLA2 (128256×32768×4096) is already at 109% comp — no urgent gap.
- 32768×14336×2048 is at 111.2% comp — no urgent gap.
- The only real residual is L6 (4096×32768×128256) at 92.6%; per the decider,
  Optimizer A's V8 R25E static-peel revival is the right play, not Q2 transplants.

---

## 5. Artifacts

- `R30_OPT_A_BUILD.json` / `R30_OPT_A_BUILD.log` — build status (5/5 PASS)
- `R30_OPT_A_BENCH.json` / `R30_OPT_A_BENCH.log` — single-rep bench (5/5 FAULT)
- `r30_opt_a_build.py` / `r30_opt_a_bench.py` — scripts
- `build_all42/tk_mxfp4_gluon_cpp_n{N}_k{K}_{variant}_r30oa_m{M}.cpython-310-x86_64-linux-gnu.so`
   × 5 — compiled SOs (kept for any follow-up; safe to delete)
- `build_all42/compile_R30OA_*.log` × 5 — per-build hipcc logs
- `build_all42/wrap_n{N}_k{K}_{variant}_r30oa_m{M}.cpp` × 5 — pybind module wrappers
