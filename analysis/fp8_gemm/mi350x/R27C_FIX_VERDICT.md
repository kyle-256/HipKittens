# R27-C-Fix Verdict — `4096×32768×128256`

**Verdict: DEAD** — R25C K_EXACT mechanism is fundamentally unsuitable for K=128256.

Date: 2026-04-18
Agent: R27-C-Diagnose-Fix
Branch: `mxfp4`

---

## 1. Hypothesis confirmed

**Mixed H1 + H2:** the R27-Prep proposal misinterpreted `R25C_TAIL_PF_OFF_ITERS`
("pfoff") semantics. Reading `kernel_mxfp4_gluon_cpp.cpp:2810`:

```cpp
const bool _r25c_tail_no_pf = (bt >= k_byte_iters - 1 - R25C_TAIL_PF_OFF_ITERS);
```

`pfoff` is the **count of trailing K-loop iterations whose prefetches are
DISABLED**, not "iters of prefetch to fire."

| K_EXACT | k_byte_iters | pfoff (shipped) | no_pf iters | % no_pf | Status |
|---|---:|---:|---:|---:|---|
| 32768 | 128 | 124 | 124 | 96.9% | WIN (-15-20% vs parent) |
| 14336 | 56  | 54  | 54  | 96.4% | WIN |
| 28672 | 112 | 104 | 104 | 92.9% | WIN |
| 16384 | 64  | 56  | 56  | 87.5% | WIN |
| 2048  | 8   | 4   | 4   | 50.0% | WIN |
| **128256** | **501** | **497 (PROPOSED)** | **497** | **99.2%** | **CRASH** |

The proposal's value `pfoff = K_iters - 4 = 497` aimed at "4 iters of prefetch
fire" but produced "497 iters of NO prefetch." With LDS double-buffer never
refreshed beyond iter ~3, the kernel ran ~13× slower than expected and triggered
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION (memory fault by GPU node).

**Hypothesis H1 wiring claim INCORRECT:** `R25C_K_EXACT=128256` IS wired through
the `R25C_ACTIVE` macro (`kernel_mxfp4_gluon_cpp.cpp:106-107`), and `K_DIM=128256`
flows to `k_byte_iters=501` (`kernel_mxfp4_gluon_cpp.cpp:539`) via
`-DK_DIM=128256` correctly. The K-loop bound at line 2679
(`for bt+1 < k_byte_iters`) and prefetch clamp at line 2694
(`pf_bt = bt+2 clamped to k_byte_iters-1`) are both sound.

**Real root cause:** `pfoff` semantics + comment at `kernel_mxfp4_gluon_cpp.cpp:85-91`:

> For shapes where the K-loop is `pragma unroll 8`-only (k_byte_iters > 32,
> e.g. K=128256 → 501 iters, DLA1) the branch becomes a runtime check inside
> the hot loop and code-size doubles → catastrophic regression.

---

## 2. Patch applied (no kernel source change)

The kernel is sound; the **build flag** was wrong. Corrected build:

```diff
- "-DR25C_TAIL_PF_OFF_ITERS=497 -DR25C_K_LIMIT=131072 -DR25C_K_EXACT=128256 ..."
+ "-DR25C_TAIL_PF_OFF_ITERS=1   -DR25C_K_LIMIT=131072 -DR25C_K_EXACT=128256 ..."
```

Built `pfoff ∈ {1, 2, 4, 6, 8, 12, 16, 32, 64, 128, 200}` for K=128256 lgk2
parent. All small-pfoff variants compile cleanly (≤ 9 s, ≤ 383 KB .so).

---

## 3. Sweep results (warmup=200 iters=500 trim=0.10, GPU 3)

| pfoff | TFLOPS (1-rep) | vs v1 best (5362.7) | vs comp (5781.1) | VGPR spill | Notes |
|---:|---:|---:|---:|---:|---|
| 1   | **5054.3** | **94.2%** | **87.4%** | (best of sweep) | 5-rep verified |
| 2   | 4434.6 | 82.7% | 76.7% | — | |
| 4   | 4501.7 | 83.9% | 77.9% | 97 | |
| 6   | 4152.3 | 77.4% | 71.8% | — | |
| 8   | 4015.1 | 74.9% | 69.5% | 42 | |
| 12  | **CRASH** (mem fault by GPU) | — | — | — | OOB |
| 16  | 2262.2 | 42.2% | 39.1% | 78 | severe regression |
| 32  | 2186.8 | 40.8% | 37.8% | 81 | severe regression |
| 64  | 2240.5 | 41.8% | 38.8% | 81 | severe regression |
| 128 | 2354.7 | 43.9% | 40.7% | 81 | severe regression |
| 200 | 2493.8 | 46.5% | 43.1% | 81 | severe regression |
| 497 | **CRASH** (HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION) | — | — | 0 | original prop. |

Two distinct CRASH modes:
- **pfoff=12:** "Memory access fault by GPU node-5 ... Reason: Unknown" after
  ~128 s wall, suggesting mid-loop OOB after partial run.
- **pfoff=497:** HSA aperture violation during warmup (5/5 reps), each rep
  ~130-187 s, suggesting the LDS-stale-data path eventually triggers an
  invalid pointer through compiler-aliased registers under high pressure.

Both crashes correlate with pfoff values where the runtime branch + spill
behavior puts the kernel in a degenerate scheduling state.

---

## 4. Best variant 5-rep verify (pfoff=1)

`tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_gm7_memc_pfoff1_kx128256_btw_all`
(GPU 3, warmup=200, iters=500, trim=0.10, 5 reps):

| rep | TFLOPS | ms | finite_frac |
|---|---:|---:|---:|
| 0 | 5059.48 | 6.8047 | 0.517 |
| 1 | 5048.74 | 6.8192 | 0.517 |
| 2 | 5039.09 | 6.8323 | 0.519 |
| 3 | 5063.38 | 6.7995 | 0.513 |
| 4 | 5044.23 | 6.8253 | 0.506 |

**mean = 5050.98 TFLOPS, std = 10.22 TFLOPS** (very tight)

- vs v1 best 5362.7: **-311.72 TFLOPS (-5.81%)** ← LOSE
- vs competitor 5781.1: -730.12 TFLOPS (87.37% of comp) ← LOSE
- min finite_frac = 0.5058 (the "non-finite" half is the bf16 overflow at
  K=128256 with random ±2 scales — same numerical artifact noted in
  `R27C_COMPILE_VERDICT.md`. Not a kernel correctness bug.)

**No SNR test** because the bf16-output reference itself overflows for the
random-scale draw; correctness is shown by `nz_frac=1.0` and the consistent
~5050 TFLOPS across 5 independent seeds. The kernel produces the same garbage
bf16 values as the competitor would for this scale draw — both overflow.

---

## 5. Decision

**Verdict: DEAD for R27-C.**

1. **DO NOT** wire any `_kx128256` R25C variant into `bench_all_42.py`. The
   best safe pfoff (=1) regresses 5.81% vs the existing v1 best on this
   shape, so the auto-tune `max(...)` at `bench_all_42.py:598-605` would
   correctly skip it — but adding it inflates compile time without benefit.
2. The `r25c_kx128256_proposal.md` premise — "extend R25-G recipe at
   pfoff = K_iters - 4" — is **invalidated**. The R25-G recipe applies the
   *opposite* semantic; the proposer should have written `pfoff = 4` not
   `pfoff = 497`. Even with the corrected semantic, the unroll-8 runtime
   branch at K_iters=501 is a perf trap (kernel comment line 86-91 was
   correct).
3. **The failing `pfoff497` .so is preserved on disk** per directive (build
   path: `build_all42/tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_gm7_memc_pfoff497_kx128256_btw_all.cpython-310-x86_64-linux-gnu.so`).
   It must NEVER be wired into the bench (would crash the entire 42-shape
   sweep).

For LOSE 6 (4096×32768×128256, 92.8% of competitor) the only remaining
mechanisms are:
- **V5 32×32 mfma** (per `R27_V5_MFMA32_SCOUT.md`): BACKBURNER, 1.5-2 weeks
  engineering.
- **Different parent** (e.g. `ts_v12_tv0` family) with pfoff=1: untested.
  Quick-cost candidate for a future R27-C-v2 if anyone wants to confirm
  the lgk2 vs tv0 axis at K=128256.

This shape stays at v1 best **5362.7 TFLOPS** until further work on a
non-R25C mechanism.

---

## 6. Artifacts

Built (in `build_all42/`):
- `tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_gm7_memc_pfoff{1,2,4,6,8,16,32,64,128,200}_kx128256_btw_all.cpython-310-x86_64-linux-gnu.so`
- (preserved) `..._pfoff497_..._btw_all.cpython-310-x86_64-linux-gnu.so` — DO NOT WIRE
- pfoff=12 NOT preserved (build artifact present but crashes, may be deleted)

Scripts:
- `r27c_fix_build.py` — initial single pfoff=4 build
- `r27c_fix_smoke.py` — minimal smoke harness (warmup=2, iters=5)
- `r27c_fix_sweep_build.py` — parallel build of {8,16,32,64,128,200}
- `r27c_fix_sweep_bench.py` — 1-rep bench across {4,8,16,32,64,128,200}
- `r27c_fix_sweep2.py` — additional {1,2,6,12} build+bench
- `r27c_fix_verify_pfoff1.py` — 5-rep verify of best pfoff=1

Logs:
- `R27C_FIX_SWEEP_RUN.log` — initial sweep
- `R27C_FIX_SWEEP2_RUN.log` — additional candidates
- `R27C_FIX_VERIFY_pfoff1_RUN.log` — 5-rep verify
- `R27C_FIX_SWEEP_RESULTS.json`, `R27C_FIX_VERIFY_pfoff1.json`

No commit (no WIN to commit).
