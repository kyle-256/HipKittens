# R53 Opt D-3A — Aiter `.co` dlopen for 96×640 tile cohort

## Summary
First attempt at applying R50D `.co` dlopen pattern to a **non-256×256 aiter tile** (96×640).
Result: **2/3 PROMOTE**, **1/3 DEAD**. Net round delta: +2 perf claw-backs, no shim rebuild needed.

## Shim rebuild status
**NO REBUILD NEEDED.** Verified pre-bench:
- `R50D_aiter_dlopen.cpp` accepts `tile_M`, `tile_N`, `co_path`, `kernel_name` as runtime kwargs.
- `bdx=256` hardcoded (line 187) was flagged as risk by decider, but verified safe:
  aiter's launcher (`/shared_nfs/kyle/test/aiter/csrc/py_itfs_cu/asm_gemm_a4w4.cu` line 290)
  uses `bdx=256` for **ALL** tiles ("4 wv64" comment, i.e. 4 wavefronts × 64 lanes).
- `f4gemm_bf16_per1x32Fp4_BpreShuffle_96x640.co` exists and exports symbol
  `_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_96x640E` (note 41-char mangled length
  vs 42 for 256x256). Verified via `/opt/rocm/llvm/bin/llvm-nm`.
- `KernelArgs` 372-byte ABI is shape-agnostic.

Pattern is now proven **tile-generic** within the aiter f4gemm_bf16_per1x32Fp4_BpreShuffle family.

## Per-shape verdict

### Shape 1: (M=4096, N=14336, K=16384) — DEAD
- HK pre-R53 comp: 84.53%
- 96×640 SMOKE TFLOPS: **2629.1** = **52.45% comp** (REGRESSION −32.08pp)
- SMOKE was clean (n_OK=1, fin=1.0, wcf=0, snr_med=55.59 dB) → not a correctness failure.
- Root cause hypothesis: 96×640 tile is **not** the aiter heuristic pick for this shape,
  or this `.co` is genuinely worse than HK's existing kernel for this aspect ratio
  (gdx=23, gdy=43 = 989 WGs vs 304 SMs → 3.25× oversubscription, not the same regime
  as shapes 2/3).
- 10-run not run (no point at -32pp regression in SMOKE).
- Artifacts: `R53_OPT_D3A_1_SMOKE.json`, `R53_OPT_D3A_1_SMOKE.log`, `bench_R53D3A_1.py`.

### Shape 2: (M=6144, N=4096, K=16384) — **PROMOTE**
- HK pre-R53 comp: 83.89%
- 96×640 10-run TFLOPS: **4717.9** = **106.54% comp** (+22.65pp vs HK)
- 10-run gate: **PASS** (n_OK=10/10, fin_min=1.0, wcf_max=0, wcf_std=0,
  snr_med 55.57-55.61 dB, all seeds [101..1010]).
- Grid: gdx=ceil(4096/640)=7, gdy=ceil(6144/96)=64.
- Artifacts: `R53_OPT_D3A_2_SMOKE.{json,log}`, `R53_OPT_D3A_2_10RUN.{json,log}`,
  `R53D3A_2_INTEGRATION_FRAGMENT.json`, `bench_R53D3A_2.py`.

### Shape 3: (M=4096, N=6144, K=32768) — **PROMOTE**
- HK pre-R53 comp: 82.39%
- 96×640 10-run TFLOPS: **4965.3** = **131.21% comp** (+48.82pp vs HK — largest
  R53 single-shape gain).
- 10-run gate: **PASS** (n_OK=10/10, fin_min=1.0, wcf_max=0, wcf_std=0,
  snr_med 55.61-55.65 dB, all seeds [101..1010]).
- Grid: gdx=ceil(6144/640)=10, gdy=ceil(4096/96)=43.
- Artifacts: `R53_OPT_D3A_3_SMOKE.{json,log}`, `R53_OPT_D3A_3_10RUN.{json,log}`,
  `R53D3A_3_INTEGRATION_FRAGMENT.json`, `bench_R53D3A_3.py`.

## Aggregate
- 2/3 PROMOTE (66.7%) — better than R52 D-2 round's 3/3, but smaller cohort and
  one DEAD entry (4096,14336,16384) where aiter's 96×640 .co underperforms HK.
- Combined perf delta on the 2 PROMOTEs: +71.47pp aggregate comp.
- Mechanism reusable: any aiter f4gemm_bf16_per1x32Fp4_BpreShuffle_<tile>.co can now
  be dlopen'd via R50D shim AS-IS (no rebuild) for any future shape.

## Methodology compliance
- warmup=200, iters=500, trim=0.10 (CLAUDE.md mandate).
- INDEPENDENT seeds [101..1010] (10-run @ 80% gate per memory note).
- Each shape on a dedicated idle GPU (4, 5, 6) with `HIP_VISIBLE_DEVICES`.
- ZERO HipKittens kernel modification.
