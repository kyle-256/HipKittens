# MXFP4 GEMM Optimization TODO

## Current State (2026-04-16)
- **Repo**: `/shared_nfs/kyle/test/Hipkittens2`
- **Branch**: `agent/mxfp4-art-rewrite`
- **Main shape**: `4096x32768x128256`
- **Project target**: `5484 TFLOPS` (`97%` of the original `5653T` aiter baseline)
- **Current reproducible default full-swap**: about `4902T` to `4911T`
- **Latest `bench_all42_results.json` on main shape**: `4974.6T / 5781.1T = 86.0%`
- **42-shape summary**: `18 WIN / 23 LOSE / 1 CRASH`, with `25/42 >= 97% of competitor_tflops`
- **Recorded crash shape in the current sweep artifact**: `128256x32768x4096`

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
2. **`permlane -> bf16 pack -> global_store_dwordx4` lowering is real**, but target-shape speedup is still only noise-level.
3. **The remaining gap is mainly read-side traffic and pipeline overlap**, not row-store micro-tuning.
4. **Small target-shape wins like `GROUP_SIZE_M=8` are too small/noisy** to count as a mainline breakthrough.
5. **User direction remains valid**: ART is not mandatory, and the original C++ kernel is still the main production path.

## Keep And Track
These are experiment evidence and should stay in the repo instead of being treated as disposable garbage:

- `PACKED_WIDESTORE_PMC.md`
- `pmc_aiter_4096x32768x128256/`
- `pmc_baseline_4096x32768x128256/`
- `pmc_baseline_gm8_4096x32768x128256/`
- `pmc_rewrite_4096x32768x128256/`
- `pmc_fullswap_curr_4096x32768x128256/`
- `pmc_packedwide_curr_4096x32768x128256/`

## Dead Ends / Do Not Repeat Blindly
- Store-only tuning without counter evidence
- LDS-transpose vecstore as a mainline idea
- Barrier/prefetch tweak loops without a concrete PMC hypothesis
- Presenting `packed wide-store` as a true kernel win
- Mixing `bench_all_42.py` `competitor_tflops` with live `aiter` runs without saying which baseline is being used

## Open Tasks

### 1. Read-side / pipeline mainline
- Find a guarded POC in `kernel_mxfp4_gluon_cpp.cpp` that reduces read-side traffic or LDS/global round-trips
- Focus first on Step3/4 prefetch interleave, Step12 fused scheduling, or the smallest A0-direct / half-direct style experiment
- A change is only promising if it improves target-shape time and moves at least one of:
  - `Frac_Wait_Any`
  - `Frac_Active_VMEM`
  - `TCP_TOTAL_READ_sum`

### 2. Refresh 42-shape status after any real kernel win
- Re-run `bench_all_42.py`
- Re-check `128256x32768x4096` instead of assuming the old crash is fixed forever
- Report both:
  - strict `WIN` count
  - `>=97%` count

### 3. Keep row-store work guarded only
- Preserve the current `MAIN_PERMLANE_BF16_*` path for A/B and lowering inspection
- Do not spend more mainline time here unless new counters say store is dominant again

## Benchmark Rules
- Use an idle GPU
- State whether the comparison uses `bench_all_42.py` `competitor_tflops` or a live `aiter` run
- Prefer: compile -> correctness -> target-shape A/B -> PMC -> commit
- Keep benchmark settings explicit in notes (`warmup`, `iters`, trim rule)

## Key Files
- `kernel_mxfp4_gluon_cpp.cpp`: current production kernel and mainline optimization target
- `PACKED_WIDESTORE_PMC.md`: store-path PMC conclusion
- `bench_all_42.py`: 42-shape benchmark against competitor numbers
- `bench_all42_results.json`: current 42-shape artifact
- `docs/profiling/profile_pmc_counters.sh`: rocprof counter groups
- `docs/profiling/analyze_pmc_counter_output.py`: PMC summarizer
