# Round 16 Optimizer B — verdict: DEAD END

**Strategy:** Compound R15B SAFE-DIFF LLVM flags on top of R10/R11 iterilp wins
for the 5 deep-LOSE shapes (S1-S5). 5 shapes × 4 flags = 20 candidate compounds.

**Bench rules:** warmup=200, iters=500, trim=10%, GPUs 2/3/4, MI355X (gfx950).
Author: kyle-256 / Kyle.Zhao@amd.com.

## Pipeline

1. `build_round16_optB_iterilp_compounds.py` — built 5 iterilp baselines
   (S1/S2 _r10_iterilp NEW; S3/S4/S5 cached) + 20 compounds. 22 OK + 3 cached,
   0 FAIL in 14.4s.
2. `asm_diff_probe_r16b.py` — SHA-256 .text vs parent and iterilp baselines.
3. `bench_round16_optB_smoke.py` — single-shot perf gate ≥ +0.5pp vs iterilp
   baseline on 10 DIFF compounds.
4. `bench_round16_optB_verify.py` — 5-run replication on smoke-PASS + 3 borderline
   (+0.24..+0.30pp single-shot) noemxpre candidates. Combined gate: mean ≥
   ilp.max AND Δpp ≥ +0.5pp.

**SNR check abandoned:** iterilp baseline is non-deterministic (SGPR-clobber
LLVM bug → ~35% exact-match between two consecutive runs), making byte-exact
SNR vs iterilp meaningless. Aperture-crash detection via bench (HSA error
would surface as bench failure) was used instead.

## ASM-diff matrix (10 DIFF / 10 NOOP_VS_ILP / 0 NOOP_VS_BASE / 0 MISSING)

| Shape | noemxpre | largeivf2 | nolicm | sinkavoidspill |
|-------|----------|-----------|--------|----------------|
| S1 (_lgk2_dc)        | DIFF | NOOP_ILP | DIFF | NOOP_ILP |
| S2 (_u32)            | DIFF | DIFF     | NOOP_ILP | NOOP_ILP |
| S3 (_v20_memc)       | DIFF | NOOP_ILP | DIFF | NOOP_ILP |
| S4 (_u16)            | DIFF | NOOP_ILP | DIFF | NOOP_ILP |
| S5 (_ts_lgk2_memc)   | DIFF | NOOP_ILP | DIFF | NOOP_ILP |

`sinkavoidspill` is NOOP on ALL 5 shapes when stacked on iterilp.
`largeivf2` is NOOP on 4/5 shapes when stacked on iterilp.

## Smoke (single-shot) — 1/10 PASS

| Lab | Tag             | ilp    | cand   | Δ        | Δpp     | gate |
|-----|-----------------|--------|--------|----------|---------|------|
| S1  | noemxpre        | 4681.16| 4695.12| +13.96   | +0.27pp | fail |
| S2  | noemxpre        | 4956.13| 4969.39| +13.26   | +0.24pp | fail |
| S1  | nolicm          | 4826.70| 4700.06| -126.64  | -2.41pp | fail |
| S2  | largeivf2       | 4939.60| 4967.58| +27.98   | +0.51pp | PASS |
| S3  | noemxpre        | 5278.95| 5295.54| +16.59   | +0.30pp | fail |
| S3  | nolicm          | 5248.53| 5113.46| -135.07  | -2.43pp | fail |
| S4  | noemxpre        | 5293.94| 5280.06| -13.88   | -0.25pp | fail |
| S4  | nolicm          | 5418.99| 5295.46| -123.53  | -2.19pp | fail |
| S5  | noemxpre        | 5054.60| 5036.22| -18.38   | -0.35pp | fail |
| S5  | nolicm          | 5064.41| 5048.51| -15.90   | -0.30pp | fail |

## Verify (5-run, GPU 2) — 0/4 combined-gate PASS

| Lab | Tag       | ilp.mean | ilp.max | cand.mean | cand.max | Δmean   | Δpp     | gate_max | gate_pp | combined |
|-----|-----------|---------:|--------:|----------:|---------:|--------:|--------:|----------|---------|----------|
| S2  | largeivf2 |  4958.47 | (n/a)   |   4975.03 | (n/a)    | +16.56  | +0.30pp | fail     | fail    | **FAIL** |
| S1  | noemxpre  |  4697.29 |         |   4705.03 |          |  +7.74  | +0.15pp | fail     | fail    | **FAIL** |
| S2  | noemxpre  |  4963.11 |         |   4960.00 |          |  -3.11  | -0.06pp | fail     | fail    | **FAIL** |
| S3  | noemxpre  |  5261.37 |         |   5271.50 |          | +10.13  | +0.18pp | PASS     | fail    | **FAIL** |

The lone smoke-PASS (S2/largeivf2 +0.51pp) collapsed to +0.30pp on 5-run mean
— single-shot was upper-tail noise.

## Conclusions

- `nolicm` is harmful on top of iterilp: -2.2 to -2.4pp on S1/S3/S4 (LICM
  re-enables critical hoisting around the iterilp scheduler's reordering).
- `noemxpre` is run-to-run noise on iterilp (±0.35pp band).
- `largeivf2` produced one outlier-positive single-shot but does not
  replicate.
- `sinkavoidspill` is dead on iterilp parents (ASM-NOOP everywhere).

**No commits.** No dispatcher updates. R16B SAFE-DIFF compound space is exhausted.

## Dead-end registry additions (20 tags)

```
R16B_iterilp+noemxpre       on S1,S2,S3,S4,S5  (NOISE / regression)
R16B_iterilp+largeivf2      on S1,S2,S3,S4,S5  (NOOP×4, single-outlier×1)
R16B_iterilp+nolicm         on S1,S2,S3,S4,S5  (REGRESSION -2pp)
R16B_iterilp+sinkavoidspill on S1,S2,S3,S4,S5  (NOOP×5)
```

## Frontier note

R16B confirms that the iterilp+R15B-flag axis is fully exhausted on the deep-LOSE
shapes. Any further breakthrough requires moving off the LLVM-flag plane —
kernel-source rewrite (R14 territory) or new parent macros not yet enumerated.
