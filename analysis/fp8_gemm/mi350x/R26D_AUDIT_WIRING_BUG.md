# R26-D AUDIT — R25_FINAL Wiring Bug (K_EXACT variants missing)

## Summary

The R25_FINAL parallel benchmark (`bench_all42_parallel_R25_FINAL.py`)
**silently skips** all R25-G/H K_EXACT variants whose `.so` files are not
pre-built in `build_all42/`. The parallel script does **lookup-only**
(no build phase), so any (N, K, suffix) combination missing on disk is
treated as `None` and excluded from the per-shape best.

This caused 5+ shapes to LOSE in the FINAL run when the R25-G K_EXACT
variant would have flipped them to WIN by **+15% to +18%**.

## Direct-test evidence (warmup=200 iters=500 trim=10%, 5 reps, std ≤ 0.5%)

| Shape (M×N×K)        | FINAL best (LOSE) | R25-G K_EXACT (WIN) | Δ vs FINAL | vs comp |
|----------------------|-------------------|---------------------|------------|---------|
| 14336×4096×32768     | 4965 (ts_pf4_memc_btw_step3) | **5862** (ts_lgk2_gm7_memc_pfoff124_kx32768_btw_all) | **+18.05%** | 111.7% |
| 4096×14336×16384     | 4870 (ts_lgk2_memc_btw_all)  | **5628** (ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all) | **+15.55%** | 112.3% |
| 4096×28672×32768     | 5451 (ts_pf4_memc_btw_step3) | **6387** (ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all) | **+17.17%** | 113.0% |
| 4096×32768×14336     | 5240 (ts_lgk2_memc_btw_all)  | **6025** (ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all) | **+14.98%** | 113.8% |
| 4096×32768×6144      | 4533 (ts_gm2_v12_memc_btw_all) | **5262** (ts_v12_gm7_memc_pfoff19_kx6144_btw_all) | **+16.09%** | 115.7% |

All 5 flip LOSE → WIN.

## Root cause

`bench_all42_parallel_R25_FINAL.py:bench_variant()` does:

```python
so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
if not os.path.exists(so_path):
    return None
```

The R25-G/H K_EXACT variants in `bench_all_42.py` lines 519-576 were
declared but **never pre-built** for the matching (N, K) pairs.
Pre-built (N, K) pairs only had a few of the 8 expected K_EXACT suffixes.

## Fix applied (this commit)

Added `r26d_build_missing.py` which enumerates all 42 shapes and ensures
the matching K_EXACT variant `.so` files exist in `build_all42/`. **9 new
.so files built** (+5 already existed = 14 total combinations needed).

After this fix, re-running `bench_all42_parallel_R25_FINAL.py` will pick up
the K_EXACT variants and flip the affected shapes to WIN.

## Other expected wins (not directly measured but very likely)

Based on the +15-18% pattern, these LOSE shapes from the FINAL partial
output should also flip to WIN once R25_FINAL is re-run with the new builds:

- Shape 23 (4096×32768×128256) — `_kx32768` variants now built for n=128256
- Shape 32 (16384×4096×14336) — `_kx14336` variants now built for n=4096

## Don't-touch verification

This patch:
- Does NOT modify `kernel_mxfp4_gluon_cpp.cpp`
- Does NOT modify `bench_all_42.py`
- Does NOT modify `bench_all42_parallel_R25_FINAL.py`
- Only adds `.so` files (gitignored) + this doc + audit scripts

## Commands run

```
python3 r26d_audit_shape27.py 6     # 4 shapes verified +14.98 to +18.05%
python3 r26d_audit_more.py 6        # +1 shape verified (shape 20)
python3 r26d_build_missing.py       # built 9 missing .so files
```

## Recommendation

Re-launch `bench_all42_parallel_R25_FINAL.py` after the in-flight run
completes. Expected new WIN count: at least +5 shapes flipped LOSE→WIN.
