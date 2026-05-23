# KPI Snapshot R95 (post-R43 chunk_size=32, foundation R52-R58 in place)

## Full 24-shape RCR fwd bench

Run: 2026-05-23, chi2811, HK turbo `8ec865a4` + PT outer `6647ff15` (post-R91)
Script: `Primus-Turbo/scripts/_bench_24_v2.py`
Method: 10 warmup + 50 timed iters, single trial

| Shape | v1 TFLOPS | v2 TFLOPS | v2/v1 |
|---|--:|--:|--:|
| dsv3_up_B4_M2048 (K=7168) | 2055 | 2411 | 1.174 |
| dsv3_up_B4_M4096 | 2455 | 2583 | 1.052 |
| dsv3_up_B16_M2048 | 2401 | 2506 | 1.044 |
| dsv3_up_B16_M4096 | 2515 | 2566 | 1.020 |
| dsv3_down_B4_M2048 (K=2048) | 1526 | 1721 | 1.128 |
| dsv3_down_B4_M4096 | 1710 | 1944 | 1.137 |
| dsv3_down_B16_M2048 | 1719 | 1969 | 1.145 |
| dsv3_down_B16_M4096 | 1768 | 2027 | 1.147 |
| qwen_up_B4_M2048 (K=4096) | 1829 | 1694 | **0.926** |
| qwen_up_B4_M4096 | 1804 | 1865 | 1.034 |
| qwen_up_B16_M2048 | 1997 | 2221 | 1.113 |
| qwen_up_B16_M4096 | 2118 | 2257 | 1.065 |
| qwen_down_B4_M2048 (K=1536) | 1313 | 1528 | 1.164 |
| qwen_down_B4_M4096 | 1410 | 1614 | 1.144 |
| qwen_down_B16_M2048 | 1330 | 1549 | 1.165 |
| qwen_down_B16_M4096 | 1447 | 1644 | 1.136 |
| gpt_oss_up_B4_M2048 (K=2880) | 1356 | 1660 | 1.224 |
| gpt_oss_up_B4_M4096 | 1667 | 1991 | 1.194 |
| gpt_oss_up_B16_M2048 | 1585 | 1764 | 1.113 |
| gpt_oss_up_B16_M4096 | 1656 | 1870 | 1.129 |
| gpt_oss_down_B4_M2048 | 1348 | 1633 | 1.211 |
| gpt_oss_down_B4_M4096 | 1631 | 1969 | 1.208 |
| gpt_oss_down_B16_M2048 | 1582 | 1763 | 1.115 |
| gpt_oss_down_B16_M4096 | 1657 | 1883 | 1.136 |

## Summary

- **v2/v1 geomean: 1.120** (+12%)
- 23/24 shape wins
- 1 regression: qwen_up_B4_M2048 -7.4%
- Median win: ~+13%

## Outliers

- Best win: gpt_oss_up_B4_M2048 +22.4% (chunk_size benefit + gpt_oss K=2880 ki=22 medium)
- Worst regression: qwen_up_B4_M2048 -7.4% (K=4096, M=2048, B=4)
  - **R97 follow-up**: isolated bench shows v2/v1 = 1.030 (v2 faster). R95 result was 24-shape consecutive thermal noise. **No actual regression.**

## Comparison to Earlier 8-shape Bench (R67)

R67 (subset): v2/v1 geomean 1.108
R95 (full 24): v2/v1 geomean 1.120

R95 broader sample = +1.2pp higher geomean, more representative of true production gain. R43 chunk_size=32 win **holds across full shape set**.

## Next-Session Priority (per R91)

1. Investigate qwen_up_B4_M2048 regression — only 1/24 shape loss; may be R43 chunk_size suboptimal for this specific shape
2. P1.2 32x32 rewrite (R63 design) for spill=0 + per-shape uniform gain
3. P1.3a split-K (R66 design) for qwen_down B16 shapes (already +13-17% v2/v1, but vs Triton still gap)
