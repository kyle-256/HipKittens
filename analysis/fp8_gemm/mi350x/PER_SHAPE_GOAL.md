# Per-Shape Goal Worksheet (R91)

## 8 production shape × v2/Triton target gates

| Shape | Current v2/T (R72) | Gate (1.15) | Gap | Achievable single-session? |
|-------|--------------------|-----:|------|----:|
| gpt_oss_up_B4_M2048 (K=2880) | 1.102 | 1.15 | -5pp | Maybe (R63 P1.2) |
| gpt_oss_up_B16_M2048 (K=2880) | 1.008 | 1.15 | -14pp | No (need P1.3a) |
| dsv3_up_B4_M4096 (K=7168) | 1.076 | 1.15 | -7pp | Maybe (R63 P1.2) |
| dsv3_up_B16_M2048 (K=7168) | 1.054 | 1.15 | -10pp | Maybe (P1.2) |
| dsv3_down_B16_M4096 (K=2048) | 1.045 | 1.15 | -11pp | No (P1.3a) |
| qwen_up_B4_M4096 (K=4096) | 1.014 | 1.15 | -14pp | No (P1.3a) |
| qwen_down_B16_M2048 (K=1536) | **0.936** | 1.15 | -21pp | No (P1.3a essential) |
| qwen_down_B16_M4096 (K=1536) | 0.965 | 1.15 | -19pp | No (P1.3a essential) |

## Geomean Path to 1.15×

If P1.2 lifts all shapes by +5pp uniformly:
- new geomean = 1.024 × 1.05 ≈ 1.075. Still -7pp from 1.15.

If P1.3a (split-K) additionally lifts short-K (qwen down) by +15pp:
- qwen_down 0.94 → 1.08, qwen_up 1.01 → 1.16
- new geomean ≈ 1.10-1.12. Still -3pp.

If both P1.2 + P1.3a + algorithmic + per-K codegen → +5-8pp each:
- geomean potentially 1.15-1.20.

**Conclusion**: 1.15× target requires combined P1.2 + P1.3a + likely also CK-style per-K codegen. Single-lever max gain ~10pp; need 3+ levers stacked.

## Per-Shape Worst-Case Drilldown (qwen_down B16 M2048)

- B=16, M_per_g=2048, N=4096, K=1536
- ki=12 (very short K)
- bpc=16, per-group tiles = 8 × 16 = 128
- total_tiles = 2048
- WG count = 256 (1 wave/CU)
- Per WG work = 2048/256 = 8 tiles average
- Per tile: ki=12 × 4 acc = 48 mfma cycles + 12 LDS loads + 12 mma blocks

This shape is **persistent-overhead-dominated** (small per-tile work, large dispatch overhead). Split-K won't help here — tile count already large. The lever is **smaller BLK** (128×128 single-acc reduces per-tile cost) or **dispatch path optimization** (faster init_group_cumsum).

## Action for qwen_down

Try: smaller WG (4-warp) with BLK 128×128 single-acc kernel as alternative dispatch target. New session. Mirror Triton's autotune logic (Triton selects smaller block for K<2048).
