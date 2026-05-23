# R116 — Prefetch depth analysis

Main loop prefetches 2 K-iter ahead. With 4 outstanding LDS + 4 outstanding vmem per iter, depth-2 = 8 in flight.
RCR_PREFETCH_LGKM=8 = wait when LDS count > 8 (max 8 outstanding LDS reads).
RCR_INIT0_VMCNT=4 / RCR_INIT1_VMCNT=6 = staged vmem release in init.
Could try depth-3 prefetch (would need third A/B LDS buffer). LDS budget: As[3] + Bs[3] = 6×16KB = 96KB > 128KB ceiling. Out of budget.
Depth-2 is max with current LDS layout. P1.2 reducing LDS per buffer enables depth-3 — secondary win after main spill=0 target.
