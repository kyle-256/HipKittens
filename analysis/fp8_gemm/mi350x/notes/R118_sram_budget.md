# R118 — SRAM budget per CU

CDNA4 (gfx950) MI355X:
- Per CU LDS: 160 KB
- Per CU SGPR: 800 dwords (shared across waves)
- Per CU VGPR: 16384 dwords (split per SIMD; 4 SIMD/CU = 4096 dwords/SIMD)
- Per CU AGPR: 16384 dwords (same allocation as VGPR)
- Per wave (1 wave/SIMD): V+A ≤ 512 dwords

Current v2 BLK_N=256 production:
- LDS: 128 KB (As+Bs 2×2 buffer)
- V: 256, A: 0-256 depending on variant (24/35 spill)
- Total live: ~512 dwords/lane = AT CAP

Implication: cannot increase ping-pong depth or per-tile state without first reducing per-tile primitives.
P1.2 32×32 wrapper releases A budget (per probe R57 V=104, A=0), creating ~150 dword/lane budget for prefetch depth + persistent state.
