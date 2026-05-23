# R115 — store_c_tile path cost

4 × mul(acc, scale) + 4 × store_c_tile_mn_masked_grouped per tile.
mul = 32 FMA per acc lane = 32 cycles × 4 acc = 128 cycles.
store = 8 buffer_store_b128 per acc × 4 acc = 32 store ops + masked branching.
~150-200 cycles total store path per tile.
For short K tile (528 cycle main loop): store = ~30% of total tile time. Significant.
Lever: fuse mul into store via dpp_shuf trick (?) or batched store path.
P1.2 32×32 wrapper: store fewer larger ops (mfma_32 writes 16 floats/call vs 4) → store ops -50%.
Estimate: P1.2 saves ~75-100 cycles store per tile, ~5% perf on short K shapes.
