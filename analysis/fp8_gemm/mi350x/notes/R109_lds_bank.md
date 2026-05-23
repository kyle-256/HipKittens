# R109 — LDS bank layout note

gfx950 LDS = 64 banks (vs CDNA3 32). Each bank 4-byte wide. fp8e4m3_4 = 4 bytes = 1 bank cell. ST_v2 swizzle is designed for 32-bank CDNA3 — may have suboptimal bank distribution on gfx950.
Per `[[fp8-rrr-attempt-h14]]` rocprof shows `SQ_LDS_BANK_CONFLICT=0` — current ST_v2 actually OK on gfx950.
LDS bank not a lever for grouped fp8 RCR.
