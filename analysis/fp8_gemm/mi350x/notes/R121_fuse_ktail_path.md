# R121 — FUSED_KTAIL path detail

FUSED_KTAIL=true template instance hit when K_rem=64 (K%128=64). gpt_oss K=2880 (2880=22×128+64) → all gpt_oss shapes hit FUSED=true.
Other 6 shapes K%128=0 → FUSED=false.

FUSED=true variant adds:
- 2 lambdas (load_a_kt, load_b_kt) for K_tail load
- 2 SRDs (a_srsrc_kt, b_srsrc_kt) with non-swizzled byte addressing
- 4 raw_buffer_load_b128 calls
- 4 mma calls (cA, cB, cC, cD) on the tail K=64 chunk

Spill 24→35 (+11 dwords) = FUSED block additional state.

Per R69 metadata verify: spill 24 (FUSED=false) / 35 (FUSED=true) unchanged through R104-R107 — confirmed stable.

P1.2 multi-session must port FUSED block to 32×32 mfma (mfma_32x32x64 with K=64 = single call covers tail naturally — no fragment stitching).
Predicted FUSED=true spill with 32×32 = 0 (much simpler tail handling).
