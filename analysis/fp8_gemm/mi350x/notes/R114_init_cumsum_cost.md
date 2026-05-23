# R114 — init_group_cumsum_smem cost

Once per WG: G=16 lookups + 16 atomic_add for cumulative sum + store to SMEM.
~64 cycles cold + 1 s_barrier (~10 cycles).
Per persistent WG: 74 cycles paid once, amortized over ~8 tiles = 9 cycles/tile.
For short tiles: 9 cycles / 528 main = 2% overhead.
Not a lever. Already small.
