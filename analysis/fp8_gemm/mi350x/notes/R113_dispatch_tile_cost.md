# R113 — dispatch_tile_in_group cost

6-level binary search per tile (find group_idx in cumsum, then br/bc within group).
Each level: 1 cmp + 1 cmov ≈ 2 cycles.
Total: ~12 cycles per tile.
For 2048 tiles / 256 WGs = 8 tiles/WG × 12 cycles = 96 cycles dispatch per WG.
Negligible vs main loop cycles. Not a lever.
