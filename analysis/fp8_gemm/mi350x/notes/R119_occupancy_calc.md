# R119 — Occupancy calculation

Current v2 BLK_N=256 launches:
- WG size: 512 threads = 8 waves (wave64)
- LDS: 128 KB / WG → 1 WG fits per CU (160 KB cap)
- 1 wave/SIMD enforced via launch_bounds(_NUM_THREADS, 1)
- Per CU: 8 waves total (1 WG × 8 waves)

If P1.2 reduces LDS to 64 KB (As[2]+Bs[2] single buffer):
- 2 WG/CU possible (128 KB total)
- Per CU: 16 waves
- More tiles concurrently → better hiding of dispatch/store overhead

But V+A budget per wave halves (512 → 256 dwords). Need spill=0 + register reduction first.

P1.2 multi-step:
1. spill=0 with same LDS (verify foundation R52-R58)
2. LDS reduction (single buffer) — depends on step 1 fragment lifetime fitting in fewer prefetch slots
3. occ=2 launch_bounds(_,2) — releases per-wave V+A cap
