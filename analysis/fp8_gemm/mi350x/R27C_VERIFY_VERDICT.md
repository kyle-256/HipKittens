# R27-C Verify Verdict — `4096×32768×128256`

**Verdict: ERR — DEAD**

## Setup
- SO under test: `build_all42/tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_gm7_memc_pfoff497_kx128256_btw_all.cpython-310-x86_64-linux-gnu.so`
- Shape: M=4096, N=32768, K=128256
- Bench: WARMUP=200, ITERS=500, TRIM=0.10, REPS=5 (per benchmark-rules.md)
- GPU: 3 (verified idle, 0% util at start)
- Harness: `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/r27c_verify_singleshape.py`
- Run log: `R27C_VERIFY_RUN.log`

## Results

| rep | result | wall (s) |
|-----|--------|----------|
| 0   | ERR `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` (rc=-6) | 186.6 |
| 1   | ERR `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` (rc=-6) | 133.0 |
| 2   | ERR `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` (rc=-6) | 131.4 |
| 3   | ERR `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` (rc=-6) | 119.4 |
| 4   | ERR `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` (rc=-6) | 131.4 |

**reps_ok = 0 / 5.** All five runs aborted during warmup with the same fault:

```
:0:rocdevice.cpp :3580: ... Callback: Queue ... aborting with error :
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION:
The agent attempted to access memory beyond the largest legal address. code: 0x29
```

mean / std: **N/A** (no successful reps)
gain vs v1 best (5362.7): N/A
gain vs competitor (5781.1): N/A
SNR: N/A (kernel never produced output)

## Diagnosis

The kernel issues an out-of-bounds buffer access on K=128256. The K_EXACT=128256
specialization with `pfoff=497` (very large prefetch-tail offset, ≈ K_iters - 4 at
K=128256/256≈501) is almost certainly indexing past the end of the A or B tile
stream — the persistent / btw_all dispatch combined with the pfoff bump for the
K=128256 path appears to compute a load address beyond the tile bounds.

Two likely root causes for the compile team to investigate:
1. `K_EXACT=128256` not actually wired into the loop bound (still using the old
   K_DIM derived loop count, but pfoff references the new bound).
2. `pfoff=497` chosen against the pre-R26 K-iter formula; with `kx128256` and
   `btw_all`, the safe pfoff window is tighter and 497 lands past the last
   legal `buffer_load` SRD.

Either way the compiled artifact is **unsafe**. It cannot be wired into
`bench_all_42.py` for v3 — it would crash the entire 42-shape sweep.

## Recommendation

**DEAD for v3.** Do **not** wire this `kx128256` variant into `bench_all_42.py`.

For the next pass, R27-C-Compile should:
- Re-derive `pfoff` for `K=128256` from the actual loop count, not by analogy
  with `K=32768` (`pfoff=120`) or `K=28672` (`pfoff=104`). Likely safe range
  is much smaller than 497.
- Add a runtime guard or, at minimum, a host-side `cudaDeviceSynchronize` +
  bound check on a 1-iter smoke run before claiming a clean compile.
- Verify `K_EXACT=128256` actually flows through to the K-loop bound.

Shape `4096×32768×128256` remains the worst residual LOSE (v1 best 5362.7,
92.8% of competitor 5781.1). The hypothesis (K_EXACT + large pfoff) is not
disproved — it just needs a sound `pfoff` choice and bound-check.
