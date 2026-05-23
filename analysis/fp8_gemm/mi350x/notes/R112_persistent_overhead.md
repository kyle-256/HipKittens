# R112 — Persistent kernel overhead breakdown

Per `[[fp8-rrr-attempt-h11_h12]]` worst-shape qwen_down: ~14% kernel topology overhead + ~11% data volume diff.
Persistent kernel prelude (group_offs scan, dispatch_tile_in_group binary search): ~50-100 cycles per tile.
For short tiles (qwen_down K=1536, ~528 cycle main loop): 50 cycle prelude = ~10% overhead.
For long tiles (dsv3 K=7168 ~2500 cycle): 50 cycle prelude = 2% overhead.
Short K shapes structurally penalized by persistent overhead.
Lever: smaller BLK (128×128) to amortize prelude over more tiles per WG.
