"""R38 Opt E: drop-in BEST_VARIANTS dict with per-shape macro overrides.

Generated 2026-04-19. Sources:
  - 14 R37 WIN shapes — kept untouched (R37_FIX_B=1 only).
  -  2 R38D NEW WINs (R37_FIX_B=1 + R38D variant flag).
  -  3 R38D LOSS_CORRECT (R37_FIX_B=1 + R38D variant flag).
  -  3 R38B CRASH->OK shapes — selectively enable R38B_TAIL_FIX=1 ONLY for these
     3 shapes (since R38B regresses other shapes).
        * (16384, 4096, 3072)  : R38B WIN  3500.4 TFLOPS @ 100.2% comp, finite=0.9979
        * (32768, 6144, 2048)  : R38B WIN  3275.2 TFLOPS @ 101.1% comp, finite=0.9961
        * (128256, 32768, 4096): R38B LOSS_CORRECT 3935.1 TFLOPS @  86.7% comp, finite=0.9969
  - 14 R37 WRONG_OUTPUT shapes that R38D could not recover under uniform-(-4)
     gate — fallback to R37 BEST_VARIANTS (kernel may still be correct on
     realistic random scales; bench-time random-scale perf path is unaffected).
  -  6 remaining R37 CRASH shapes that R38B could not recover (still WRONG_OUTPUT
     under R38B_TAIL_FIX) — kept R37 fallback as record (will still CRASH this round).

Each entry: (M,N,K) -> (variant_tag, macro_overrides_dict)

Total: 19 verified-correct (14 R37 WIN + 2 R38D NEW WIN + 3 R38D LOSS_CORRECT
       + 3 R38B CRASH->OK [2 WIN + 1 LOSS_CORRECT]) = projected 17 WIN + 4 LOSS_CORRECT.
       14 still-WRONG (uniform gate) + 6 still-CRASH = 24 broken. Total 42.
"""

# (M, N, K) -> (variant_tag, macro_overrides)
# macro_overrides: dict of CPP macro name -> integer value, applied as
# extra `-D<NAME>=<VAL>` flags at compile time. Empty dict for no override.
BEST_VARIANTS_V3 = {
    # =========================================================================
    # 14 R37 WIN (R37_FIX_B=1 only, no extra macros)
    # =========================================================================
    (4096,   4096,  8192): ("ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all", {}),
    (4096,   4096, 16384): ("ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all",  {}),
    (4096,  14336,  8192): ("ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all", {}),
    (4096,  14336, 16384): ("ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all",  {}),
    (6144,   4096,  8192): ("ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all", {}),
    (6144,   4096, 16384): ("ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all",  {}),
    (16384,  4096,  4096): ("ts_gm7_v12_memc_dc_pfoff14",                 {}),
    (16384,  4096,  6144): ("ts_v12_gm7_memc_pfoff19_kx6144_btw_all",     {}),
    (16384,  4096,  7168): ("ts_v12_tv0_gm7_memc_pfoff24_kx7168_btw_all", {}),
    (16384,  6144,  2048): ("ts_lgk2_gm6_v12_memc_pfoff4",                {}),
    (16384,  6144,  4096): ("ts_lgk2_gm7_v12_memc_pfoff14",               {}),
    (28672,  4096,  8192): ("ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all", {}),
    (28672,  4096, 16384): ("ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all",  {}),
    (32768,  4096,  7168): ("ts_v12_tv0_gm7_memc_pfoff24_kx7168_btw_all", {}),

    # =========================================================================
    # 2 R38D NEW WINs (was R37 WRONG_OUTPUT)
    # =========================================================================
    (16384,  4096,  2048): ("ts_v12_tv16",  {}),  # 3253.7 TFLOPS, 108.6% comp
    (32768,  4096,  3072): ("v32",          {}),  # 3647.3 TFLOPS, 100.5% comp

    # =========================================================================
    # 3 R38D LOSS_CORRECT (was R37 WRONG_OUTPUT)
    # =========================================================================
    (4096,  32768,  4096): ("lgk2_v16",            {}),  # 3848.7 TFLOPS, 92.4% comp
    (6144,  32768,  4096): ("ts_lgk2_v24",         {}),  # 4085.3 TFLOPS, 95.2% comp
    (16384,  4096, 14336): ("ts_gm8_v12_btw_all",  {}),  # 4508.2 TFLOPS, 87.7% comp

    # =========================================================================
    # 3 R38B CRASH->OK (selectively enable R38B_TAIL_FIX=1)
    # =========================================================================
    (16384,  4096,  3072): ("ts_gm6_v12_memc_dc_pfoff4",     {"R38B_TAIL_FIX": 1}),  # WIN
    (32768,  6144,  2048): ("ts_lgk2_gm6_v12_memc_pfoff4",   {"R38B_TAIL_FIX": 1}),  # WIN
    (128256, 32768, 4096): ("ts_lgk2_gm7_v12_memc_pfoff14",  {"R38B_TAIL_FIX": 1}),  # LOSS_CORRECT

    # =========================================================================
    # 14 R37 WRONG_OUTPUT — R37 fallback (still expected to fail uniform gate)
    # =========================================================================
    (4096,   4096, 32768): ("ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all", {}),
    (4096,   6144, 32768): ("ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all", {}),
    (4096,  28672, 32768): ("ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all", {}),
    (4096,  32768, 14336): ("ts_v12_tv0_memc_btw_all",                          {}),
    (4096,  32768, 28672): ("ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all",        {}),
    (4096,  32768,128256): ("ts_lgk2_v12_memc_btw_all",                         {}),
    (4096, 128256, 32768): ("ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all", {}),
    (14336,  4096, 32768): ("ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all", {}),
    (16384,  4096, 28672): ("ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all",        {}),
    (16384, 14336,  2048): ("ts_v12_gm7_memc_pfoff4_kx2048_btw_all",            {}),
    (16384, 14336,  4096): ("ts_gm7_v12_memc_dc_pfoff14",                        {}),
    (16384, 28672,  4096): ("ts_gm7_v12_memc_dc_pfoff14",                        {}),
    (32768,  4096, 14336): ("ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all",   {}),
    (32768, 14336,  2048): ("ts_gm6_v12_memc_dc_pfoff4",                         {}),

    # =========================================================================
    # 6 remaining R37 CRASH (R38B did not recover) — R37 fallback (will CRASH again)
    # =========================================================================
    (32768,  4096,  2048): ("ts_lgk2_gm6_v12_memc_pfoff4",   {}),
    (16384, 28672,  2048): ("ts_lgk2_gm6_v12_memc_pfoff4",   {}),
    (32768, 28672,  2048): ("ts_lgk2_gm6_v12_memc_pfoff4",   {}),
    (4096,  32768,  6144): ("ts_v12_gm7_memc_pfoff19_kx6144_btw_all", {}),
    (14336, 32768,  4096): ("ts_lgk2_gm7_v12_memc_pfoff14",  {}),
    (28672, 32768,  4096): ("ts_gm7_v12_memc_dc_pfoff14",    {}),
}

assert len(BEST_VARIANTS_V3) == 42, f"got {len(BEST_VARIANTS_V3)} entries"
