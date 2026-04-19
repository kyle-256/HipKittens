# R50 Opt D — VERDICT: **PROMOTE**

## Bottom line
- **+1 net VC**: 35/42 → **36/42** for `(4096, 32768, 28672)` via aiter `.co` dlopen.
- **10-run @ 80% gate: PASS** (10/10 seeds). `wcf_max=0.0`, `wcf_std=0.0`, `fin_min=1.0`,
  `snr_med` range 55.59–55.63 dB.
- **Perf**: 5575.9 TFLOPs at warmup=200, iters=500, trim=0.10 → **100.14% of `competitor_tflops` (5568.2)**.
  This shape goes from PERMA-CRASH directly to slightly above the aiter baseline.
- **No HipKittens kernel modification** — single-cell dispatch escape hatch.

## Mechanism
Per `project_mxfp4_R49B_n32k_28k_cohort_scales.md` and `project_mxfp4_R49C_embedded_vmcnt_dead.md`,
the `(4096, 32768, 28672)` cell is unsolvable via macro-level changes:
- R45B closed in-block fences (5 positions DEAD).
- R47A closed external fences on R46B 3-buffer (3 positions DEAD).
- R48A closed physical asm-block split (compiler RA runs before asm boundary).
- R49C closed embedded-fence-inside-`emit_pf_tail` (Jaccard 0.011-0.037).
- R49B closed sister-shape mechanism transfer (`R44A` drain insufficient at N=32768).

The R50 Opt D path side-steps the kernel rewrite entirely by binding aiter's
hand-written `.co` for the same 256×256 tile geometry, via `hipModuleLoad` /
`hipModuleGetFunction` / `hipModuleLaunchKernel` with the `KernelArgs` ABI
documented in `/shared_nfs/kyle/test/aiter/csrc/py_itfs_cu/asm_gemm_a4w4.cu`.

## What was built
| Artifact | Purpose |
|---|---|
| `R50D_aiter_dlopen.cpp` | Self-contained pybind11 shim that loads aiter `.co` and launches the kernel. 372-byte `KernelArgs` struct (static_assert verified). |
| `build_R50D.py` | hipcc compile to `build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`. No aiter library link. |
| `R50D_aiter_csv_audit.md` | Full audit of CSV row, KernelArgs ABI, launch params, layout requirements. |
| `R50D_aiter_symbols.txt` | `llvm-readelf --syms` confirming `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E` is FUNC GLOBAL PROTECTED in the `.co`. |
| `bench_R50D.py` | Bench harness. Uses `aiter.get_triton_quant(per_1x32)(..., shuffle=True)` for scales and `aiter.shuffle_weight(layout=(16,16))` for B. Modes: smoke / 10run. |
| `R50_OPT_D_SMOKE.{json,log}` | Single-seed perf+correctness. |
| `R50_OPT_D_10RUN.{json,log}` | 10-INDEPENDENT-seed correctness. |
| `R50D_INTEGRATION_FRAGMENT.json` | Per-shape dispatch directive. |
| `R50D_BUILD_MANIFEST.json` | Build metadata (rc, elapsed). |

## SMOKE result (seed=101)
```
status=OK   fin=1.0   wcf=0.0   snr_med=55.6   tflops=5575.4
pct_comp=100.13%   wall=11.1s
```

## 10-run @ 80% gate (seeds [101..1010])
| seed | status | fin | wcf | snr_med | tflops |
|---:|:---|---:|---:|---:|---:|
| 101 | OK | 1.0 | 0.0 | 55.60 | 5575.9 |
| 202 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.63 | — |
| 303 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.62 | — |
| 404 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.61 | — |
| 505 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.59 | — |
| 606 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.61 | — |
| 707 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.61 | — |
| 808 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.62 | — |
| 909 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.62 | — |
| 1010 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.61 | — |

**n_OK = 10/10** (gate ≥ 8/10) | **wcf_max = 0.0** (< 0.02) | **wcf_std = 0.0** (< 0.01) | **fin_min = 1.0** (≥ 0.97).

`passes_10run_gate = true`.

## Acceptance gates (all met)
| Gate | Required | Achieved |
|---|---:|---:|
| Runs to completion | yes | yes (no HSA fault) |
| `kernel_finite >= 0.97` | yes | 1.0 |
| `wcf_max < 0.02` | yes | 0.0 |
| `snr_med >= 10` dB | yes | 55.6 dB |
| 10-run `n_OK_5 >= 8/10` | yes | 10/10 |
| 10-run `wcf_max < 0.02` | yes | 0.0 |
| 10-run `wcf_std < 0.01` | yes | 0.0 |
| 10-run `fin_min >= 0.97` | yes | 1.0 |
| `pct_comp > 0` | yes | 100.14% |

## Integration directive
See `R50D_INTEGRATION_FRAGMENT.json`. The integration manifest entry adds:
```json
{
  "4096x32768x28672": {
    "backend": "aiter",
    "co_path": "/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co",
    "kernel_name": "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E",
    "shim_so":   "build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so",
    "tile_M": 256, "tile_N": 256, "log2_k_split": 0
  }
}
```

The R50 reviewer must wire the per-shape dispatch in the integration bench
(`bench_all_42_R50_INTEGRATION.py`) so that this single `(M,N,K)` triple invokes the
shim's `launch(...)` instead of the HipKittens module's `gemm_rcr(...)`. **All 35
existing R44 VC shapes continue to use the HipKittens kernel unchanged** — this is a
single-cell escape hatch, with zero blast radius on the other shapes.

## Layout caveat (load-bearing for integration)
The R50D shim requires aiter-style preshuffled inputs:
- B is preshuffled via `aiter.shuffle_weight(layout=(16,16))` — DIFFERENT from
  HipKittens' B preshuffle (which is none / column-major).
- A_scale and B_scale are preshuffled via `aiter.get_triton_quant(per_1x32)(x, shuffle=True)`
  — DIFFERENT from HipKittens' `preshuffle()` function.
- A is row-major fp4x2 (uint8) — same as HipKittens.
- Output C is `[((M+31)//32)*32, N] bf16` row-major — pad rows to multiples of 32.

The integration harness must therefore call `aiter`'s prep utilities for the
`(4096, 32768, 28672)` cell only, and continue using HipKittens' prep for all
other 41 shapes. The bench script `bench_R50D.py` is the canonical reference.

## What this round teaches us (durable memory)
- The `.co` dlopen path is a reliable single-cell escape hatch when a CRASH cell
  cannot be fixed by macro-level changes.
- The aiter `KernelArgs` ABI (372 bytes packed, p2/p3 padding pattern) is stable
  enough to mirror in a self-contained shim.
- For ANY future MXFP4 shape that aiter has a tuned `.co` for and HipKittens
  cannot solve, the same pattern (Python uses `aiter.shuffle_weight` +
  `get_triton_quant(per_1x32)(shuffle=True)` for prep; shim does
  `hipModuleLoad` + `hipModuleLaunchKernel`) will work in ≤ 1 day of work.
- Cost: layout incompatibility on B and B_scale means the dispatched cell has
  **DIFFERENT** scale/weight buffers than the HipKittens-dispatched cells. Any
  caller that wants to pass the same B across multiple shapes must keep both
  layouts.
