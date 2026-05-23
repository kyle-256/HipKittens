# R124 — v1 nondeterminism residual

Per `[[bn128-race-rcr-fix]]`: RCR bn128 race fixed via triple-buffer + FUSED_KTAIL gate + vmcnt drain.
But R71/R72 smoke runs still see v1 NaN on some shapes (qwen_down B16, dsv3 B16 random):
- Race fix targeted bn128 path specifically
- Some shapes hit bn0 OR autotune-selected bn256 path
- bn256 has separate (different) race that's much rarer (~0.6-1.2% nondet)

v2 dispatcher always uses BLK_N=256 (no bn128 path); v2 doesn't see these issues.

For final P4 (delete v1) decision: v2 must demonstrate equivalent or better correctness on all production shapes. Current bench shows v2 perf wins; correctness needs systematic SNR scan (not just smoke).
