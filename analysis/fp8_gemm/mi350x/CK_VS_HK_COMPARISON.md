# CK_tile vs HK fp8 grouped — reference comparison (R84)

## Source

CK_tile fp8 grouped GEMM dispatcher: `composable_kernel/example/ck_tile/03_gemm/`
HK fp8 grouped current: `kernel_fp8_layouts.cpp` (v1 production)

## Key Architectural Differences

| Feature | HK current | CK_tile reference |
|---------|-----------|-------------------|
| Block tile | 256×256×128 | 128×128×128 |
| Per-warp acc | 64×128 (4 accs) | 64×64 (1 acc) |
| Warp tile mfma | 16×16×128 | 32×32×64 |
| LDS double-buffer | As[2][2] Bs[2][2] (128 KB) | 1 ping-pong (32 KB) |
| MFMA scheduling | sched_barrier(0) total | sched_group_barrier batched |
| K tail | FUSED_KTAIL fused | per-K codegen specialized |
| Spill (BN=256) | 24/35 (FUSED=false/true) | 0 |
| vs Triton geomean (worst shape) | 0.94× | 1.00-1.05× |

## Triton FP8 Grouped Reference

Triton ref `grouped_matmul_kernel_fp8` (origami autotune cfg for gfx950):
- BLK_M=BLK_N=256, BK=128
- num_stages=2 (LDS double-buf)
- group_m=4
- chunk_size=32 (matches R43 win)

## Lever Adoption Matrix

| CK trick | Apply to HK? | Effort | Effect estimate |
|----------|--------------|--------|-----------------|
| 1-acc + 1 ping-pong (vs 4+2x2) | YES (P1.2 step) | 600 LOC | spill 24→0, perf neutral or +5% |
| sched_group_barrier batched | TRIED R30/R44 fail | — | regression on HK topology |
| 32x32 warp tile | YES (P1.2 step) | 400 LOC | spill release (validated R52-R58) |
| Per-K codegen | YES (multi-template) | 300 LOC | -2-3% on FUSED=true shapes |
| Smaller BLK 128x128 | Mixed | 200 LOC | helps small shapes, hurts large |

## Recommendation

Combine 1+3 (CK row 1 + row 3) in single P1.2 multi-session rewrite. Rows 2 / 4 / 5 are separate sessions or risk increase.

## Open Question

Is HK's higher-throughput-on-big-shapes (>2 dsv3) due to bigger BLK, or different LDS layout? Need rocprof L2 hit rate compare to decide.
