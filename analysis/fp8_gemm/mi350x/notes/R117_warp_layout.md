# R117 — Warp layout (2M×4N) rationale

WARPS_M=2 WARPS_N=4 chosen for:
- Per-warp 128M × 64N = 8192 cells = 128 floats/lane @ 4 acc
- Matches mfma_16x16 face × (4M×2N) per acc, cleanly
- N-split favors wider register spreading (4 N-warps share B fragment)

Swapping to 4M×2N: per-warp 64M × 128N = 8192 same, but mfma face 2M×4N per acc → fewer M-tiles per acc, more N-tiles. May reduce A-load reuse.

8-warp WG mandate (user) locks total count, but M×N split is flexible. Untested if 4M×2N spill or perf differ. Single-session try worth.
