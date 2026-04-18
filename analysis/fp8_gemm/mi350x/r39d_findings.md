# R39 Dev D — Predicate wire-in audit

## Charter

R38 Reviewer found R37 Dev B's 8B-KV wire-in was DEAD (compiled-in but unreached). R38 wrap commit `66ef02d8` patched the dispatcher predicate. R39 task: audit ALL existing R28-R38 predicate wire-ins for the same class of bug, plus verify the R38 wrap fix actually fires +24-26% on 8B-KV in production.

## TL;DR

- **R38 wrap fix `66ef02d8` VERIFIED** — 8B-KV (M=4096 N=1024 K=4096) PROD .so via dispatcher path measures **+27.15% (GPU5 Welch t=+68.0)** and **+27.71% (GPU0 t=+84.3)**, matching R37 Dev B's kernel-direct +24.96% claim (and the R38 Reviewer PATCHED-wire +26.66% measurement).
- **All 9 predicate wire-ins FIRE correctly** through the production dispatcher path (`gemm_crr_pq_v2` → `dispatch_pq_v2<CRR>`). 0 wire-in bugs found beyond the one R38 already fixed.
- **0 false-positive predicates** — 8192³ default build emits no advisories.
- **NEW methodology bug found in audit infrastructure (NOT a wire-in bug)**: when `r37_paired_bench_2so.py` is invoked with two .so built from the SAME `PY_MODULE_NAME` (e.g. both `tk_mxfp8_layouts`), Python's `importlib.util.spec_from_file_location` returns the SAME in-memory module for the second load → both `MOD_A` and `MOD_B` resolve to the SAME .so → measured Δ% collapses to 0. Fix: build paired .so with distinct `-DPY_MODULE_NAME=...` flags. Documented as R39+ rule below.

## Predicate audit table

R39 Dev D enumerated `static int warned_*` symbols in `kernel_mxfp8_layouts.cpp` and verified each on its triggering shape via the production dispatcher path. **All audits use the R37 Dev C / R38 Dev D paired-build flow**: distinct `-DPY_MODULE_NAME=...` per .so + `r37_paired_bench_2so.py` for HB shrink (kernel-swap) predicates / `advisory_check.py` for advisory-only predicates. Each shape was built per shape (M_DIM/N_DIM/K_DIM are compile-time).

| # | predicate                  | symbol                  | shape (M,N,K)            | layout type    | wire-in  | expected Δ% (kernel-direct)  | observed Δ% (dispatcher path)       | status   |
|---|---|---|---|---|---|---|---|---|
| 1 | V2-RCR 8B Q/O (R36 Dev C)   | `warned_qo_8b_rcr`      | (4096, 4096, 4096)       | advisory       | stderr canary | n/a (advisory, no kernel swap) | advisory FIRES                      | FIRES    |
| 2 | V2-RCR 70B Q/O (R36 Dev C)  | `warned_qo_70b_rcr`     | (4096, 8192, 8192)       | advisory       | stderr canary | n/a                            | advisory FIRES                      | FIRES    |
| 3 | V2-RRR 70B Down (R32 Dev C) | `warned_70b_down`       | (4096, 8192, 28672)      | advisory       | stderr canary | n/a                            | advisory FIRES                      | FIRES    |
| 4 | V2-RRR 70B GU (R33 Dev C)   | `warned_70b_gateup`     | (4096, 28672, 8192)      | advisory       | stderr canary | n/a                            | advisory FIRES                      | FIRES    |
| 5 | V2-RRR 70B KV (R33 Dev C)   | `warned_70b_kv`         | (4096, 1024, 8192)       | advisory       | stderr canary | n/a                            | advisory FIRES                      | FIRES    |
| 6 | V2-RRR 8B KV (R33 Dev C)    | `warned_8b_kv`          | (4096, 1024, 4096)       | advisory       | stderr canary | n/a                            | advisory FIRES                      | FIRES    |
| 7 | V2-RRR 8B GU (R34 Dev B)    | `warned_8b_gateup`      | (4096, 14336, 4096)      | advisory       | stderr canary | n/a                            | advisory FIRES                      | FIRES    |
| 8 | V2-RRR 8B Down (R36 Dev B)  | `warned_8b_down`        | (4096, 4096, 14336)      | advisory       | stderr canary | n/a                            | advisory FIRES                      | FIRES    |
| 9 | HB shrink B1 70B-KV (R37 Dev A) | `warned_n1024_hbshrink` (K=8192 branch) | (4096, 1024, 8192) | kernel-swap | predicate + dispatch | +28.02% (R36) / +28-30% (R37 4-GPU) / +28.07-28.58% (R38 Reviewer) | **+28.59% (GPU5 Welch t=+165.2)** | FIRES    |
| 10| HB shrink B1 8B-KV (R37 Dev B + R38 wrap fix `66ef02d8`) | `warned_n1024_hbshrink` (K=4096 branch) | (4096, 1024, 4096) | kernel-swap | predicate + dispatch | +24.96% (R37 Dev B kernel-direct) / +26.66% (R38 Reviewer PATCHED-wire) | **+27.15% (GPU5 t=+68.0) / +27.71% (GPU0 t=+84.3)** | **FIRES (R38 wrap fix CONFIRMED)** |
| neg | 8192³ default                | n/a                     | (8192, 8192, 8192)       | n/a            | n/a            | 0 advisories (no shape match)  | 0 advisories                        | OK       |

Audit collateral in `/tmp/r39d_audit/`:
- `quick_test.py` — single dispatcher-path call + stderr capture (proves wire-in advisory fires)
- `advisory_check.py` — same as above but with module-name-from-filename so multiple .so can co-load
- `8bkv_R38exact_gpu0.log`, `8bkv_R38exact_gpu5.log` — 8B-KV R38 wrap fix verification
- `70bkv_R38exact_gpu5.log` — 70B-KV R37 Dev A reproduction control
- `build_adv_*.log` — per-shape build logs

## R38 wrap fix `66ef02d8` verification — PASS

Per the critical task: R38 wrap commit `66ef02d8` patched `kernel_mxfp8_layouts.cpp:5749` from `g.k == 8192` to `(g.k == 8192 || g.k == 4096)` so that 8B-KV (K=4096) now goes through the HB shrink B1 dispatch instead of falling through to default V2-CRR.

Result on M=4096 N=1024 K=4096 via `r37_paired_bench_2so.py` (paired BABA, n=10 (GPU5) / n=6 (GPU0), distinct PY_MODULE_NAME built per .so):

| GPU | sclk-pre/post-bench | DEFAULT median TF | HBSHRINK median TF | Δ%      | Welch t   |
|-----|---------------------|-------------------|--------------------|---------|-----------|
| 5   | 2331/2395 MHz       | 693.13            | 881.33             | +27.15% | +68.0     |
| 0   | 2376/2376 MHz       | 681.96            | 870.92             | +27.71% | +84.3     |

Both runs:
- Stderr advisory `[tk_mxfp8_layouts] gemm_crr_pq_v2: HB shrink Stage B1 (BLK_M=128, PIPE=1) ACTIVE for N=1024 tall-thin (M=4096, N=1024, K=4096)` confirmed firing.
- nm-gate `nm tk_mxfp8_r39d_8b_b1*.so | grep -c hbshrink` = 4 (PROD), 0 (DEFAULT) — confirms hbshrink kernel symbol present in PROD only.
- Correctness PASS: SNR 49.61 dB, det 3/3, pass_rate 100% on both .so.
- All 8B-KV PROD samples in [864.80, 892.69] TF — clean cluster, no bimodality.

R38 wrap fix `66ef02d8` is CONFIRMED CRITICAL FIX SUCCESSFUL.

## NEW methodology bug found during audit (R39+ MUST follow)

### Bug

When invoking `r37_paired_bench_2so.py` to compare two .so files, if both .so are built **without** `-DPY_MODULE_NAME` overrides (i.e., both default to the same `tk_mxfp8_layouts` pybind11 module name), Python's `importlib.util.spec_from_file_location("tk_mxfp8_layouts", path)` + `spec.loader.exec_module(mod)` will resolve **the second load to the SAME in-memory module as the first** (CPython caches by module name across `sys.modules`-like state in pybind11's init). The result: `MOD_A` and `MOD_B` both reference the SAME .so (whichever was loaded first via `MOD_A`), and the bench measures noise around 0% Δ instead of the true delta.

### Symptom (caught here)

Initial 8B-KV bench with both .so built using `PY_MODULE_NAME=tk_mxfp8_layouts` (the default) on GPU0:
- DEFAULT median 657.50 TF, HBSHRINK median 658.17 TF → **Δ = +0.10%, Welch t = -0.95** (false-negative)

After rebuilding with distinct `-DPY_MODULE_NAME=tk_mxfp8_r39d_8b_default` and `-DPY_MODULE_NAME=tk_mxfp8_r39d_8b_b1` on the same GPU0:
- DEFAULT median 681.96 TF, HBSHRINK median 870.92 TF → **Δ = +27.71%, Welch t = +84.3** (correct)

### Fix (R39+ MUST follow)

For ALL paired `r37_paired_bench_2so.py` runs, build each .so with a UNIQUE `-DPY_MODULE_NAME=<unique_name>` and pass `MOD_A=<unique_name_A> MOD_B=<unique_name_B>`. The R38 Reviewer's `r38_reviewer_ship_verify.sh` already does this correctly (`MODNAME_A="tk_mxfp8_r38rev_${CELL}_default" MODNAME_B="tk_mxfp8_r38rev_${CELL}_b1"`). Any bench script or one-off command that uses two .so files MUST follow this pattern.

This is **not a wire-in bug** in the production code — it is a methodology bug in any audit harness that loads two .so under the same module name. R39 Dev D recommends amending the R38 NEW methodology rules to mandate distinct PY_MODULE_NAME for paired builds.

## R39+ recommendations

1. **【methodology — R39 NEW】** When using `r37_paired_bench_2so.py` (or any 2-.so paired bench), each .so MUST be built with a unique `-DPY_MODULE_NAME=<x>` and the bench MUST pass matching `MOD_A=<x_A> MOD_B=<x_B>`. Without this, both modules resolve to the SAME .so and Δ% collapses to noise. Recommend adding a defensive check at top of `r37_paired_bench_2so.py`: `assert mod_A is not mod_B, "MOD_A and MOD_B must be distinct module names"`.
2. **【high】** Adopt the R39+ priority list item 2 (`MXFP8_DISPATCH_TRACE=1`) tracepoint mechanism — even with this audit's 100% pass rate, runtime trace would have caught the methodology bug above immediately (would show only one .so being invoked across both A and B branches).
3. **【close】** No new wire-in bugs found in R28-R38 predicate fan-out. R38 Reviewer's catch was the only one.

## Cumulative tally update

R38 wrap commit `66ef02d8` is fully validated by R39 Dev D 2-GPU triangulation. No new closures from this audit (audit-only, no kernel changes).
