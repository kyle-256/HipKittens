# R110 — MFMA latency cycle budget

gfx950 mfma_f32_16x16x128_f8f6f4: 16 cycles latency, 64 ops/cycle peak.
4 acc × 8 mfma per acc = 32 mfma per K-iter × 16 cycles = 512 cycles MFMA pipeline.
LDS load: ds_read_b128 = 4 cycles round-trip per quad.
Per main iter: 4 ds_read + 32 mfma = 16 + 512 = 528 cycles steady-state.
HBM load: vmem cycles dominated by DRAM latency (~600 cycles), hidden by mfma pipeline if prefetch deep enough.
With RCR_PREFETCH_LGKM=8 + RCR_STEADY_VMCNT=8: prefetch window = 8 outstanding LDS + 8 outstanding vmem = 16 ops. At 4 ds_read+vmem per iter, prefetch covers 4 iters. Enough.
Per-iter math: BLK 256×256×128 = 16M ops × 1 wave / 1 cycle = ~16K cycles compute, vs ~528 cycles MFMA pipe → mfma is 3% of total iter time, rest is LDS/vmem latency.
This puts bound: HK can never exceed ~33% of peak FLOPS without LDS/vmem optimization.
Triton hits ~25% peak → there's ~8pp room for LDS optimization (P2.2 B pre-transpose).
