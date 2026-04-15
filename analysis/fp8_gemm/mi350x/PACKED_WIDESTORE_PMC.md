# Packed Wide-Store PMC Summary

Target shape: `4096x32768x128256`

This note compares three kernels:

- current full-swap baseline
- guarded packed wide-store POC (`MAIN_PERMLANE_BF16_STORE_POC=1` and `MAIN_PERMLANE_BF16_DWORDX2_STORE_POC=1`)
- existing `aiter` reference PMC snapshot

## Setup

- Baseline PMC directory: `pmc_fullswap_curr_4096x32768x128256`
- Packed-store PMC directory: `pmc_packedwide_curr_4096x32768x128256`
- Aiter PMC directory: `pmc_aiter_4096x32768x128256`
- Both local kernels were profiled with the same `rocprofv3` counter groups from `docs/profiling/profile_pmc_counters.sh`
- The packed-store path was separately verified in device asm to lower to:
  - `v_permlane16_swap_b32_e32`
  - `v_cvt_pk_bf16_f32`
  - `global_store_dwordx4`

## Key Metrics

| Metric | Baseline | Packed wide-store | Aiter |
| --- | ---: | ---: | ---: |
| `Frac_INSTS_VMEM` | 0.08810 | 0.08743 | 0.08715 |
| `Frac_Active_VMEM` | 0.12372 | 0.12283 | 0.05687 |
| `Frac_Wait_Any` | 3.09852 | 3.04843 | 1.47979 |
| `vL1_Read_Frac` | 0.99366 | 0.99366 | 0.99627 |
| `vL1_Write_Frac` | 0.00634 | 0.00634 | 0.00373 |
| `L2_Hit_Rate` | 0.81333 | 0.81341 | 0.81042 |
| `SQ_Busy_Ratio` | 0.98459 | 0.98673 | 0.96653 |
| `TCP_TOTAL_READ_sum` | 5.26175e9 | 5.26175e9 | 4.47873e9 |
| `TCP_TOTAL_WRITE_sum` | 3.35544e7 | 3.35544e7 | 1.67772e7 |
| `SQ_WAIT_INST_ANY` | 1.28045e9 | 1.25528e9 | 4.36125e8 |
| `SQ_CYCLES` | 4.13216e8 | 4.11732e8 | 2.94736e8 |

## Interpretation

The packed wide-store epilogue is real and slightly better than the scalar permlane store, but the improvement is small:

- `Frac_Active_VMEM` only drops from `0.12372` to `0.12283`
- `Frac_Wait_Any` only drops from `3.09852` to `3.04843`
- `SQ_CYCLES` only drops by about `0.36%`

At the same time, the counters that separate the current kernel from `aiter` are still dominated by the read side and the overall pipeline:

- `Frac_Active_VMEM` is still more than `2x` the `aiter` value
- `Frac_Wait_Any` is still about `2x` the `aiter` value
- `TCP_TOTAL_READ_sum` is still about `17%` higher than `aiter`
- `TCP_TOTAL_WRITE_sum` changes are much smaller in absolute terms than the read-side gap

## Conclusion

The packed wide-store path is worth keeping as a guarded epilogue experiment, but these PMC results do **not** support store epilogue work as the main optimization frontier for the target shape.

The next main line should focus on:

- reducing read-side traffic
- reducing LDS/global round-trips
- improving the inner-loop schedule and overall load/compute overlap

Continuing to iterate only on the row-store epilogue is unlikely to close the remaining gap to `aiter`.
