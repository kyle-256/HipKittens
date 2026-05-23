# Integration Test Plan (R93)

## Layer 1: Correctness (per-session gate)

- `_smoke_p1_0_rcr_v2.py`: 5 shape × 2 bn = 10 case, SNR ≥ 30 dB gate
- For P1.2: add `_smoke_p1_2_32x32.py` mirroring same shapes with 32x32 path explicit
- For P1.3a: add `_smoke_p1_3a_splitk.py` with sk_split_n=1 (no-op) and sk_split_n=2 paths

## Layer 2: Spill Regression (per-session gate)

```python
from amdhsa_meta import all_v2_kernels
# fetches via roc-obj-extract + llvm-readelf
for k in all_v2_kernels:
    assert k.vgpr_spill_count <= entry_baseline[k.name]
```

Add `scripts/_check_spill.py` to formalize.

## Layer 3: Per-Shape Perf Sanity (24 shape)

bench v2 vs Triton vs hk_dense; per-shape ratio must not regress > 5% vs prior baseline.

`scripts/_bench_p1_rcr_v2.py` exists (8 shape subset). Need full 24-shape variant `_bench_24_v2.py`.

## Layer 4: Geomean Gate

v2/Triton geomean ≥ entry baseline.

## Layer 5: Real-Training Transfer

per Rule 11 ([[no-cache]] + bucket K1-K4): each accepted gain must transfer to actual training step, not just isolated kernel bench. Need PT training-loop integration test.

`pytest tests/pytorch/ops/test_grouped_gemm_fp8.py -v -k "..."` is the binding test. Add a "step" integration test that does forward+backward and checks gradient correctness + step TFLOPS.

## Test Matrix

| Layer | Trigger | Cost (wall time) | Frequency |
|-------|---------|------------------|-----------|
| 1 Correctness | Every commit | ~30 s | per-session entry/exit |
| 2 Spill regression | Every kernel-level commit | ~10 s | per-session entry/exit |
| 3 Per-shape perf | Major change | ~5 min | per phase milestone (P1.2 S4, S5, etc) |
| 4 Geomean gate | Major change | ~2 min (single bench) | per session exit |
| 5 Training transfer | Pre-release | ~30 min | per phase end |

## Failure Modes

- **Layer 1 fails**: instant rollback, no commit
- **Layer 2 fails**: investigate spill source via amdhsa.kernels diff, accept only if scratch < 80 B and perf gain > 3%
- **Layer 3 fails (per-shape)**: per-shape decision; if 1 shape regresses but others gain > 2× more, may accept
- **Layer 4 fails**: rollback entire session
- **Layer 5 fails**: rollback + redesign — production gain not real

## Tooling Gaps

Current session has only Layer 1 (smoke) and Layer 4 (geomean bench). Need to add:
- `scripts/_check_spill.py` for Layer 2 automation
- `scripts/_bench_24_v2.py` for Layer 3 full-shape
- `tests/training_step_integration_test.py` for Layer 5

These tooling additions are session 0 prerequisites for P1.2 multi-session work.
