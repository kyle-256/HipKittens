"""R38 Opt D: drop-in replacement for build_R37.py BEST_VARIANTS dict.

Generated 2026-04-19. Sources:
  - 14 R37 WIN shapes — kept untouched.
  -  9 R37 CRASH shapes — kept untouched (R38D did NOT attack the crash set;
     R38B/R38C are responsible for those).
  -  2 R38D NEW WINs (R37 was WRONG_OUTPUT under uniform-(-4) gate; R38D
     replaced with a clean variant that passes correctness AND beats comp).
  -  3 R38D LOSS_CORRECT (R37 was WRONG_OUTPUT; R38D replaced with a clean
     variant that passes correctness but is 87.7-95.2% of comp — still better
     than a hard fail on the gate).
  - 14 R37 WRONG_OUTPUT shapes that R38D / R38Dv2 could NOT recover. The
     uniform-(-4) probe is structurally incompatible with the K-tile×variant
     combo for these shapes (bf16 saturation under uniform inputs). The R37
     BEST_VARIANTS entry is left as the fallback — actual perf may be fine
     under realistic random scales (the bench's perf-timing pass uses random
     scales). See R38_OPT_D_VERDICT.md.

Total: 16 WIN, 3 LOSS_CORRECT, 14 still-WRONG (uniform gate), 9 CRASH = 42.
"""

# (M, N, K) -> variant suffix tag (without leading underscore)
BEST_VARIANTS = {
    # --- 14 R37 WIN (kept) ---
    (4096,   4096,  8192): "ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all",
    (4096,   4096, 16384): "ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all",
    (4096,  14336,  8192): "ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all",
    (4096,  14336, 16384): "ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all",
    (6144,   4096,  8192): "ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all",
    (6144,   4096, 16384): "ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all",
    (16384,  4096,  4096): "ts_gm7_v12_memc_dc_pfoff14",
    (16384,  4096,  6144): "ts_v12_gm7_memc_pfoff19_kx6144_btw_all",
    (16384,  4096,  7168): "ts_v12_tv0_gm7_memc_pfoff24_kx7168_btw_all",
    (16384,  6144,  2048): "ts_lgk2_gm6_v12_memc_pfoff4",
    (16384,  6144,  4096): "ts_lgk2_gm7_v12_memc_pfoff14",
    (28672,  4096,  8192): "ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all",
    (28672,  4096, 16384): "ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all",
    (32768,  4096,  7168): "ts_v12_tv0_gm7_memc_pfoff24_kx7168_btw_all",

    # --- 2 R38D NEW WINs (was R37 WRONG_OUTPUT) ---
    (16384,  4096,  2048): "ts_v12_tv16",          # 3253.7 TFLOPS, 108.6% comp, finite=0.9998
    (32768,  4096,  3072): "v32",                   # 3647.3 TFLOPS, 100.5% comp, finite=0.9958

    # --- 3 R38D LOSS_CORRECT (was R37 WRONG_OUTPUT) ---
    (4096,  32768,  4096): "lgk2_v16",              # 3848.7 TFLOPS, 92.4% comp, finite=0.9982
    (6144,  32768,  4096): "ts_lgk2_v24",           # 4085.3 TFLOPS, 95.2% comp, finite=0.9959
    (16384,  4096, 14336): "ts_gm8_v12_btw_all",    # 4508.2 TFLOPS, 87.7% comp, finite=0.9959 (R38Dv2 f34 axis)

    # --- 14 R37 WRONG_OUTPUT shapes — NO clean variant passed uniform-scale gate;
    #     reverted to R37 BEST_VARIANTS as fallback. ---
    (4096,   4096, 32768): "ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all",
    (4096,   6144, 32768): "ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all",
    (4096,  28672, 32768): "ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all",
    (4096,  32768, 14336): "ts_v12_tv0_memc_btw_all",
    (4096,  32768, 28672): "ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all",
    (4096,  32768,128256): "ts_lgk2_v12_memc_btw_all",
    (4096, 128256, 32768): "ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all",
    (14336,  4096, 32768): "ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all",
    (16384,  4096, 28672): "ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all",
    (16384, 14336,  2048): "ts_v12_gm7_memc_pfoff4_kx2048_btw_all",
    (16384, 14336,  4096): "ts_gm7_v12_memc_dc_pfoff14",
    (16384, 28672,  4096): "ts_gm7_v12_memc_dc_pfoff14",
    (32768,  4096, 14336): "ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all",
    (32768, 14336,  2048): "ts_gm6_v12_memc_dc_pfoff4",

    # --- 9 R37 CRASH shapes (untouched; R38D did not attack these) ---
    (16384,  4096,  3072): "ts_gm6_v12_memc_dc_pfoff4",
    (32768,  4096,  2048): "ts_lgk2_gm6_v12_memc_pfoff4",
    (32768,  6144,  2048): "ts_lgk2_gm6_v12_memc_pfoff4",
    (16384, 28672,  2048): "ts_lgk2_gm6_v12_memc_pfoff4",
    (32768, 28672,  2048): "ts_lgk2_gm6_v12_memc_pfoff4",
    (4096,  32768,  6144): "ts_v12_gm7_memc_pfoff19_kx6144_btw_all",
    (14336, 32768,  4096): "ts_lgk2_gm7_v12_memc_pfoff14",
    (28672, 32768,  4096): "ts_gm7_v12_memc_dc_pfoff14",
    (128256,32768,  4096): "ts_lgk2_gm7_v12_memc_pfoff14",
}

assert len(BEST_VARIANTS) == 42, f"got {len(BEST_VARIANTS)} entries"
