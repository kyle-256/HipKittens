# Session R33-R99 Handoff README (R101)

## For Next-Session Picking Up

**Start here**: read `SESSION_R33_R99_SUMMARY.md` for full round log.

Then in order:
1. `PLAN_V2.md` — overall multi-phase roadmap (S52-R69 KPI table at end)
2. `P1_2_INTEGRATION_DESIGN.md` — 32×32 mfma rewrite (RECOMMENDED FIRST)
3. `FOUNDATION_PROBE_PATTERN.md` — proof that P1.2 viable
4. `MULTI_SESSION_EXECUTION_CHECKLIST.md` — per-session entry/exit gates
5. `P1_3A_SPLITK_DESIGN.md` — second priority (only path to 1.15× geomean)
6. `P2_2_RRR_DESIGN.md`, `P3_2_CRR_DESIGN.md` — later phases

## State Snapshot

- HK turbo HEAD: `afe3289b` (R98)
- PT outer HEAD: `81f34501` (R98)
- v2/Triton geomean: 1.024 (post-R43, 5-trial median ±0.003)
- v2/v1 geomean: 1.120 (24-shape, R95)
- Production spill: 24/35 (FUSED=false/true) unchanged from R27 baseline

## What's New in v2 kernel (R52-R58)

Search `kernel_fp8_layouts2.cpp` for:
- `mma_32_int4` (R52 wrapper)
- `__probe_v2_mma_32_isolated` (R53)
- `__probe_v2_mma_32_kchain_22` (R54)
- `__probe_v2_mma_32_2acc_k22` (R55)
- `__probe_v2_mma_32_4acc_k22` (R56)
- `__probe_v2_mma_32_8w_4acc_k22` (R57)
- `__probe_v2_mma_32_8w_4acc_lds_k22` (R58)

These are FOUNDATION probes, not production paths. Production still uses 16×16×128 mfma in `grouped_rcr_kernel_body_pinned`.

## Helper Scripts (gitignored, in scripts/)

- `_smoke_p1_0_rcr_v2.py` — 5 shape × 2 bn correctness gate
- `_bench_p1_rcr_v2.py` — 8 shape perf bench
- `_bench_24_v2.py` — 24 shape full bench (NEW R95)
- `_check_spill.py` — amdhsa.kernels spill check (NEW R94, needs polish)
- `_rocprof_qwen_down.py` — single-call profile target (NEW R77)

## Known Constraints to Carry Forward

- `[[no-cache]]` — NO result caches in PT grouped GEMM path
- `[[no-ck-fallback]]` — NO routing worst-shape to composable_kernel
- `[[no-constant-sweep]]` — NO experimental const sweep, ISA-level justification only
- `[[dual-hk-path]]` — EVERY HK edit must `cp` to PT 3rdparty + sync push both
- `[[gfx950-8wave-va-cap]]` — 8-warp WG (1 wave/SIMD) V+A ≤ 256 dwords with launch_bounds(_,1)
- `[[fp8-rcr-unroll-harmful]]` — unroll != 2 on RCR main loop = active-harmful
- `[[fp8-rcr-partial-barriers]]` — never remove partial-warp barriers (race fix + scheduling boundary)
- `[[bn128-race-root-cause]]` — v1 bn=128 has HW race; v2 doesn't hit (always BN=256)

## Reproduce R95 bench

```
ssh login_node2
ssh chi2811
docker exec mlperf_gptoss bash
cd /workspace/code/Primus-Turbo
/opt/venv/bin/python scripts/_bench_24_v2.py
```

Expected: v2/v1 geomean ≈ 1.12, all 24 shapes v2 ≥ v1 (within ±5%).
