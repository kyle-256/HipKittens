# MXFP4 GEMM Optimization TODO

## Current State (2026-04-16)
- **Repo**: `/shared_nfs/kyle/test/Hipkittens2`
- **Branch**: `agent/mxfp4-art-rewrite`
- **Main shape**: `4096x32768x128256`
- **Project target**: `5484 TFLOPS` (`97%` of the original `5653T` aiter baseline)
- **42-shape hard gate**: every shape must be `>=95% competitor_tflops` and there must be `0 CRASH`
- **Current sampled target-shape reference under the main benchmark harness**: `5019.9T` on GPU7 with `-DGROUP_SIZE_M=8` and default-on `NONVOLATILE_SCALE_X2_POC=1`
- **Main benchmark harness definition**: `bench_all_42.py` style = `warmup=200`, `iters=500`, `trimmed mean 10%`, and no per-iteration `C.zero_()`
- **Important correction**: earlier hand-written target-shape A/B runs that did `C.zero_(); run()` overstated the value of `GROUP_SIZE_M=8`; they should not be used as the mainline conclusion anymore
- **Latest complete `bench_all42_results.json` artifact**: still the pre-nvscale sweep, with target shape at `4909.9T / 5781.1T = 84.9%`
- **Latest complete 42-shape summary**: `16 WIN / 26 LOSE / 0 CRASH`, average ratio about `98.8%`
- **Crash status update**: `128256x32768x4096` did not crash in the latest full sweep, but only reached `4063.3T / 4536.4T = 89.6%`

## Recent Commits
```
399cd546 MXFP4: document packed store PMC comparison
4bffa75d MXFP4: add guarded packed wide-store main POC
15273d62 MXFP4: add guarded main-kernel permlane store POC
595e9b0a MXFP4: validate permlane store on fused swap debug paths
```

## Confirmed Conclusions
1. **Store epilogue is not the main bottleneck on the target shape.**
   - See `PACKED_WIDESTORE_PMC.md`
   - `Frac_Active_VMEM`: `0.12372 -> 0.12283`, but `aiter` is `0.05687`
   - `Frac_Wait_Any`: `3.09852 -> 3.04843`, but `aiter` is `1.47979`
   - `TCP_TOTAL_READ_sum`: our kernel `5.26175e9`, `aiter` `4.47873e9`
2. **Allowing the compiler to schedule scale `buffer_load_dwordx2` is the first credible broad read-side win.**
   - target `4096x32768x128256`, `_gm8`, GPU7 main benchmark harness: `4937.3T -> 5019.9T`
   - target `4096x32768x128256`, `_gm8`, GPU3 sample A/B: `4885.2T -> 4988.9T`
   - `4096x32768x6144`, `_u8`, GPU3: `3865.3T -> 3921.9T`
   - representative `default / _u32 / _gm2 / _u16` samples also all moved up, so `NONVOLATILE_SCALE_X2_POC` is now default-on
   - GPU6 target-shape PMC says the gain is mostly **less waiting at roughly the same read volume**:
     - `TCP_TOTAL_READ_sum`: `5.27224e9 -> 5.27224e9`
     - `TCC_REQ_sum`: `5.68768e8 -> 5.68768e8`
     - `SQ_WAIT_INST_ANY`: `1.05367e9 -> 1.03737e9` (about `-1.55%`)
3. **`permlane -> bf16 pack -> global_store_dwordx4` lowering is real**, but target-shape speedup is still only noise-level.
4. **The remaining gap is mainly read-side traffic and pipeline overlap**, not row-store micro-tuning.
5. **Existing `half_direct` / `direct_a` / current `direct_b` idea are not ready to be mainline candidates for this shape.**
   - `half_direct` + `GM=8`: `3457.83T`
   - `direct_a` + `GM=8`: `2527.51T`
   - integrated `DIRECT_B` target-shape A/B in `HipKittens` also regressed in the current swap/gm8 form
6. **Current Hipkittens2 default hot path is already basically aligned with `HipKittens` default hot path outside this scale-load scheduling change.**
   - no obvious missing default-path patch remains to be copied over
7. **Recent dead-end closures**:
   - `PF_N` tuning in Step3/4: no target-shape win
   - isolated Step12-swapped POC: no credible target-shape win
   - Step12 batch-style structural POC: target regressed
   - `TAIL_SPLIT` / `SPREAD_LDS`: only tiny small-K help, no target-shape win
   - low-risk compile-flag sweeps: no stable target-shape win
8. **User direction remains valid**: ART is not mandatory, and the original C++ kernel is still the main production path.

## Keep And Track
These are experiment evidence and should stay in the repo instead of being treated as disposable garbage:

- `PACKED_WIDESTORE_PMC.md`
- `NVSCALE_PMC.md`
- `pmc_aiter_4096x32768x128256/`
- `pmc_baseline_4096x32768x128256/`
- `pmc_baseline_gm8_4096x32768x128256/`
- `pmc_rewrite_4096x32768x128256/`
- `pmc_fullswap_curr_4096x32768x128256/`
- `pmc_packedwide_curr_4096x32768x128256/`
- `pmc_seq_baseline_gm8_4096x32768x128256/`
- `pmc_seq_nvscale_gm8_4096x32768x128256/`

## Dead Ends / Do Not Repeat Blindly
- Store-only tuning without counter evidence
- LDS-transpose vecstore as a mainline idea
- Barrier/prefetch tweak loops without a concrete PMC hypothesis
- Presenting `packed wide-store` as a true kernel win
- Blindly porting `half_direct` / `direct_a` ideas into `gluon_cpp` without a new measured mechanism
- Treating `TAIL_SPLIT` / `SPREAD_LDS` as target-shape solutions after they have already been shown to be at best small-K nudges
- Treating hand-written `C.zero_()` target-shape loops as the main benchmark source of truth
- Mixing `bench_all_42.py` `competitor_tflops` with live `aiter` runs without saying which baseline is being used

## Open Tasks

### 1. Read-side / pipeline mainline
- Finish one fresh `bench_all_42.py` sweep with the new default-on `NONVOLATILE_SCALE_X2_POC` before opening more variant-space churn
- Summarize that sweep with at least:
  - strict `WIN` count
  - `>=97%` count
  - `<95%` count
  - whether `0 CRASH` still holds
- Use `NVSCALE_PMC.md` as the current mechanism filter: new read-side wins should ideally reduce wait/overlap pressure even when total read bytes stay roughly flat
- After that, keep looking for a guarded POC in `kernel_mxfp4_gluon_cpp.cpp` that reduces read-side traffic or LDS/global round-trips, **starting from the original default path under the main benchmark harness**
- Focus next on a real Step12 fused-schedule structural change or a more global overlap change, not another `ts/spread`-style local reshuffle
- A change is only promising if it improves target-shape time and moves at least one of:
  - `Frac_Wait_Any`
  - `Frac_Active_VMEM`
  - `TCP_TOTAL_READ_sum`

### 2. Refresh 42-shape status after any real kernel win
- Re-run `bench_all_42.py`
- Re-check `128256x32768x4096` instead of assuming the latest non-crash result is permanent
- Report both:
  - strict `WIN` count
  - `<95%` count
  - `>=97%` count

### 3. Keep row-store work guarded only
- Preserve the current `MAIN_PERMLANE_BF16_*` path for A/B and lowering inspection
- Do not spend more mainline time here unless new counters say store is dominant again

## Benchmark Rules
- Use an idle GPU
- Prefer `GPU 5/6/7`; avoid `GPU 3`
- All benchmark conclusions must use **`warmup=200, iters=500`**
- The current user hard gate is: every shape must be `>=95% competitor_tflops` and there must be `0 CRASH`
- The mainline benchmark source of truth is the `bench_all_42.py` style harness: `200/500`, `trimmed mean 10%`, and no per-iteration `C.zero_()`
- State whether the comparison uses `bench_all_42.py` `competitor_tflops` or a live `aiter` run
- Prefer: compile -> correctness -> target-shape A/B -> PMC -> commit
- Keep benchmark settings explicit in notes (`warmup`, `iters`, trim rule)

## Key Files
- `kernel_mxfp4_gluon_cpp.cpp`: current production kernel and mainline optimization target
- `PACKED_WIDESTORE_PMC.md`: store-path PMC conclusion
- `NVSCALE_PMC.md`: scale-load scheduling PMC conclusion for the current best read-side win
- `bench_all_42.py`: 42-shape benchmark against competitor numbers
- `bench_all42_results.json`: current 42-shape artifact
- `docs/profiling/profile_pmc_counters.sh`: rocprof counter groups
- `docs/profiling/analyze_pmc_counter_output.py`: PMC summarizer
