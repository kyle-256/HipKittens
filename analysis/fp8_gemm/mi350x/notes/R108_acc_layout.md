# R108 — Acc layout investigation note

Per warp 64×32 area = 32 floats/lane = 32 AGPR per acc × 4 acc = 128 AGPR. 
Current 16×16×128 mfma uses 8 mfma calls per acc to cover this area, each writes 4 floats.
Compiler is forced to keep all 8 mfma's source frags alive simultaneously for scheduling.
Per `[[fp8-rrr-attempt-h5-diag]]`: removing patched gl<> views → 0 V/spill delta — confirms 4-acc body register pressure is in MFMA chain, not view storage.
