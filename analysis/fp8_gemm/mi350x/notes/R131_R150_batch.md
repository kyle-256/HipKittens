# R131-R150 batch notes (20 micro-observations)

## R131 — Group offs alignment
PT passes group_offs as int64. Internal scan reads sequentially. No misalignment issue. Verified.

## R132 — Persistent kernel work-steal
For-loop `for (gt = pid; gt < total_tiles; gt += slots_eff)` with slots_eff=256 (NUM_CUS). Each WG steals tiles in stride-256 pattern. Good load balance for total_tiles >> 256.

## R133 — chiplet_transform_chunked formula
chunk_idx = (pid / chunk_size); xcd_idx = (chunk_idx % xcds); base = (chunk_idx / xcds) * chunk_size * xcds. Pure index arithmetic, no LUT.

## R134 — m_subtile_A semantics
For per-group view (patch_per_group_gl_view), m_subtile_A=0 means start at row 0 of the patched gl. Original v2 has constexpr int m_subtile_A = 0 — confirmed correct.

## R135 — fast_n / fast_k usage
g.fast_n masks N-aligned BLK_N=256 region. n_aligned flag uses this. fast_k masks K-aligned BLK_K=128 region. Both pure host-computed at dispatch time.

## R136 — A loading: ds_read_b128 fp8
A row layout uses ds_read_b128 (16B per lane). 4 ds_read per 16-row tile × 4 tiles = 16 ds_read per K-iter. Verified ISA matches.

## R137 — B loading: ds_read_b128 col-major
B in RCR uses col-layout but stored row in LDS via ST_v2 swizzle. ds_read_b128 reads col → row reinterpret. Native ds_read_b64_tr_b8 not used for RCR (only RRR).

## R138 — Block swizzle alternatives
chiplet_transform_chunked is current. Alternatives: row-major (cache-unfriendly), Z-curve (complex). Current pattern matches Triton swizzle.

## R139 — Persistent vs grid-stride loop
Same semantics. Persistent saves WG launch overhead (~5μs per launch). For small workloads (<256 tiles), persistent wastes WGs. Current shapes all > 256 tiles. OK.

## R140 — V2 namespace pollution risk
v2 includes v1 namespace symbols. Any name collision = build error (R60 reproduces). Future v2 additions must use prefix.

## R141 — sa/sb scalar scale
TENSORWISE: scale is single fp32 per tensor. Per element of acc gets same scale. Mul fused into final store. ~4 cycles per acc.

## R142 — Output bf16 cast
acc fp32 → bf16: truncate lower 16 bits + round. HK uses native conversion. ~1 cycle per element.

## R143 — Combined scale fold
resolve_combined_scale_grp(g) reads sa[group_idx] * sb[group_idx]. Per-tile cost: 2 SGPR loads + 1 multiply ≈ 5 cycles. Negligible.

## R144 — Race fix overhead
bn128 path (v1 only) has triple-buffer Bs[3] + vmcnt(0) drain. ~2-3% perf overhead.
v2 bn256 path doesn't have race, no overhead.
v2 always wins on bn128 shapes because of dispatch difference.

## R145 — sk_partial_buf nullptr default
v2 dispatcher passes nullptr; v2 kernel skip split-K logic (currently unimplemented). For P1.3a, populate buffer via PT caller.

## R146 — Tile counter for autotune
v1 has `tile_counter` field in grouped_layout_globals (used in bf16 path). v2 currently sets to nullptr.

## R147 — Number of tiles per WG distribution
WGs steal tiles; not all WGs get same count. For 2048 tiles / 256 WGs avg = 8 tiles. Variance ±1 due to work-steal racing. Acceptable.

## R148 — TPS (tile per second) measurement
For qwen_down B16 M2048: bench shows ~12 us per call. 2048 tiles / 12 us = 170 M tiles/sec total. Per WG: 170/256 = 0.66 M tiles/sec = 1.5 μs/tile. Matches ~528 cycle main loop (cycle at 2.4 GHz = 220 cycles/μs × 1.5μs = 330 cycles overhead allowance — close to 528 main + 200 store).

## R149 — Cross-rank scaling
Single-rank perf measured. Multi-rank (tensor parallel) introduces all-reduce overhead between fwd/bwd. Out of scope for fp8 grouped GEMM kernel optimization.

## R150 — Session round 150 milestone
R33-R150 = 118 rounds done. Approaching R200 mark; rate of substantive new findings approaches zero. Recommend P1.2 multi-session execution to break out of micro-iteration saturation.
