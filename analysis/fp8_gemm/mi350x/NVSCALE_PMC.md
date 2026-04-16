# Non-Volatile Scale-Load PMC Summary

Target shape: `4096x32768x128256`

This note compares the target-shape `GM=8` baseline against the same kernel with
`NONVOLATILE_SCALE_X2_POC=1`, i.e. allowing the compiler more freedom around the
scale-path `buffer_load_dwordx2`.

## Setup

- Baseline PMC directory: `pmc_seq_baseline_gm8_4096x32768x128256`
- Nvscale PMC directory: `pmc_seq_nvscale_gm8_4096x32768x128256`
- PMC was collected on `GPU 6` with the four `rocprofv3` counter groups from
  `docs/profiling/profile_pmc_counters.sh`
- Target-shape throughput A/B was measured on `GPU 7` with the main
  `bench_all_42.py` single-shape harness:
  - `warmup=200`
  - `iters=500`
  - trimmed mean `10%`
  - no per-iteration `C.zero_()`
- The only intentional functional difference is:
  - baseline: `-DNONVOLATILE_SCALE_X2_POC=0`
  - nvscale: `-DNONVOLATILE_SCALE_X2_POC=1`

## Key Metrics

| Metric | Baseline | Nvscale |
| --- | ---: | ---: |
| target `avg_ms` | `6.9731` | `6.8583` |
| target TFLOPS | `4937.3` | `5019.9` |
| `SQ_WAIT_INST_ANY` | `1.05367e9` | `1.03737e9` |
| `Frac_Wait_Any` | `2.80306` | `2.75933` |
| `Frac_Active_VMEM` | `0.11402` | `0.11369` |
| `SQ_INSTS` | `9.77371e8` | `9.73267e8` |
| `TCP_TOTAL_READ_sum` | `5.27224e9` | `5.27224e9` |
| `TCC_REQ_sum` | `5.68768e8` | `5.68768e8` |
| `L2_Hit_Rate` | `0.81229` | `0.81230` |
| `Scache_Hit_Rate` | `0.91667` | `0.90998` |

Notes:

- `target TFLOPS` improved by about `+82.6T` (`4937.3 -> 5019.9`)
- `SQ_WAIT_INST_ANY` dropped by about `-1.55%`
- `SQ_INSTS` dropped by about `-0.42%`
- `TCP_TOTAL_READ_sum` and `TCC_REQ_sum` are effectively unchanged
- `Frac_Wait_Any` follows the same fallback normalization already used in this
  repo's PMC analysis flow when `SQ_WAVE_CYCLES` is not available, so the
  absolute value is less important than the relative direction

## Interpretation

These counters do **not** support a "we moved fewer bytes" story:

- `TCP_TOTAL_READ_sum` is unchanged
- `TCC_REQ_sum` is unchanged
- `L2_Hit_Rate` is unchanged to noise

Instead, they support a modest but real **schedule / overlap** story:

- `SQ_WAIT_INST_ANY` drops meaningfully
- `Frac_Active_VMEM` improves slightly
- `SQ_INSTS` also drops a bit

The most plausible interpretation is that removing `volatile` from the scale
`buffer_load_dwordx2` gives the compiler just enough freedom to schedule these
loads more cleanly around the existing compute pipeline, reducing wait pressure
without materially changing read volume.

## Conclusion

`NONVOLATILE_SCALE_X2_POC=1` is a real mainline win and should stay enabled by
default:

- it improves the target-shape throughput under the main benchmark harness
- it reproduces across GPUs
- the PMC result is directionally consistent with the observed speedup

At the same time, this is still only a first read-side gain, not the final
solution:

- the target shape is still well below the long-term goal
- the 42-shape summary still needs a fresh post-nvscale sweep
- future work should prioritize additional overlap / schedule improvements over
  more store-only tuning
