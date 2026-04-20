# R58 Worker I-2 (Opt O) Verdict — L8 HK probe DEAD on correctness; ACCEPT_FALLBACK

**Date:** 2026-04-19
**Worker:** I-2 (Opt O — L8 HK kernel probe)
**Cell:** L8 = `(M, N, K) = (4096, 32768, 128256)`
**Current baseline:** R52D2B_AITER 256x256 (R57 reviewer p50 = **97.75%**)
**Verdict:** **ACCEPT_FALLBACK (D-3A-1)**; HK 256x256 axis on L8 K=128256 **CLOSED** by correctness failure.

---

## Summary

Two HK kernel binaries existed for the `n32768/k128256` shape (both 256x256 lgk2 v12). Both were SMOKE-benched on GPU 0 at seed=101 (warmup=200, iters=500, trim=0.10). **Both produced catastrophic-wrong output:**

| Variant | SO path | fin (gate >= 0.97) | wcf (gate < 0.02) | snr_med_db | tflops |
|---|---|---:|---:|---:|---:|
| R40B safe (primary) | `build_R40B/tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_v12_btw_all_R40B_safe.so` | **0.804** | **0.128** | 36.25 | n/a (WRONG_OUTPUT) |
| R37 memc (fallback) | `build_R37/tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_v12_memc_btw_all_R37.so` | **0.775** | **0.066** | 31.79 | n/a (WRONG_OUTPUT) |

Neither candidate reaches the correctness gate (fin >= 0.97 AND wcf < 0.02 AND snr_med_db >= 10.0). The `wcf` of 0.128 / 0.066 indicates **6.6%-12.8% catastrophically wrong cells** — this is the well-known "17% deterministic-wrong cells" failure mode documented in project memory for HK on intermediate/large-K shapes (R35 Opt B / R37 root cause).

Because perf (TFLOPS) is gated behind correctness in the harness (matches the project-wide protocol), **no perf measurement was obtained**, and there was no need to escalate to a 10-run.

## Decision flow

1. **Discovery**: globbed `analysis/fp8_gemm/mi350x/build_*` for `*k128256*.so`. Found 2 dedicated HK 256x256 lgk2 v12 builds (R40B + R37) plus ~80 alternate `n32768_k128256_*` variants in `build_all42/` (older-vintage configurations: gm{1,2,8,16,32,64}, ext_br, no_embed, f34_*, memc, lgk{2,4}, tv{0,16}, etc.). Both targeted candidates are the most recent and most-tuned variants for this exact shape.
2. **Primary SMOKE (R40B)**: `WRONG_OUTPUT` at fin=0.804, wcf=0.128 → STOP_DEAD.
3. **Fallback SMOKE (R37)**: `WRONG_OUTPUT` at fin=0.775, wcf=0.066 → STOP_DEAD.
4. **No 10-run**: per Opt O gate spec §3 (STOP_DEAD).
5. **D-3A-1 protection**: keep R52D2B_AITER 256x256 manifest entry verbatim. AITER share unchanged at 39/42; HK cells unchanged at 3/42.

## Axis closure

**HK 256x256 lgk2 v12 axis on L8 K=128256 = CLOSED by correctness failure.**

- 2 SMOKE attempts (most-tuned R40B safe + R37 memc fallback) both well below correctness gate
- Correctness gap (fin=0.78-0.80 vs gate 0.97; wcf=0.07-0.13 vs gate 0.02) is structural, not noise; the 17% deterministic-wrong-cells cohort applies
- This is **complementary** to the already-CLOSED aiter alt-tile axis (R56 G-4 128x512 -13.31pp, 192x256 -11.84pp; R57 H-3 224x256 -10.69pp)
- **No untried alt-K=128256 HK binaries** exist in the repo with the R39A/R44A correctness fixes ported in (those fixes were applied to lower-K cohorts).

## R59 recommendations

**Preferred (option B): close Opt O axis permanently for L8 K=128256.** Accept R52D2B_AITER 256x256 at 97.75% as the durable L8 baseline. The strict-VC 42/42 leaderboard is already 100%; the LOSE classification on L8 is a 2.25pp perf-only gap at the aiter K=128256 internal ceiling. R59 budget is better spent on:
- Opt N (gate-tightening) adoption decision
- Opt P (128x256 alt-tile completion on 3 HK cells) if not run in R58
- Methodology/closure work rather than further L8 attacks

**Option A (high-risk, deferred):** build a from-scratch HK kernel for `n32768/k128256` with R39A TAIL_SCALE_CLAMP + R44A back-edge drain + R44D FINITE_GATE 0.97 protocol pre-applied. Estimated cost: ~1 build round + 1 verification round + 1 perf round = 3 R-rounds. Probability of WIN: very low (aiter at internal ceiling for K/N=4 ratio; HK has not been competitive on K=128256 since R37). **Not recommended unless R59 explicitly elects to spend budget on a >=2-round build campaign.**

## Files produced

- `bench_R58I2_O1.py` — bench harness (HK gemm_rcr invocation pattern; mirrors `bench_R44D_10run.py`)
- `R58_OPT_O1_SMOKE.{json,log}` — R40B SMOKE results (WRONG_OUTPUT)
- `R58_OPT_O1b_SMOKE.{json,log}` — R37 fallback SMOKE results (WRONG_OUTPUT)
- `R58I2_O1_INTEGRATION_FRAGMENT.json` — manifest fragment with ACCEPT_FALLBACK verdict
- `R58_OPT_I2_VERDICT.md` — this file

No 10-run files (skipped per STOP_DEAD).
No manifest changes proposed; reviewer integration carries R57 manifest forward byte-identical for cell L8.

## Floor target reached

The R58 plan §7 Floor target ("0 PROMOTE + Opt N report delivered" + "SMOKE-DEAD on Opt O acceptable") is **met** by this verdict. L8 axis closure is documented; manifest is preserved; bit-determinism on L8 (AITER 256x256) is preserved.
