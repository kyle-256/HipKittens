## MXFP4 Benchmark Rules (MANDATORY)

All benchmark measurements MUST use these parameters:
- **warmup = 200**
- **iters = 500** (profiling iterations)
- **trimmed mean**: 10% trim from each end
- **GPU isolation**: `HIP_VISIBLE_DEVICES=N` on an idle GPU (check `rocm-smi` first)

### Quick single-shape test
```python
WARMUP = 200
ITERS = 500
TRIM_FRAC = 0.10
```

### Machine info
- This machine is **MI355X** (gfx950)
- competitor_tflops in bench_all_42.py are the correct baselines for THIS machine
- ~~ASM inline reference at 8192³: **5084 TFLOPS**~~ — **REVOKED 2026-04-17**:
  the inlined `kernel_mxfp4_asm_inline.{cpp,h}` produces INCORRECT output (SNR -1.31 dB
  vs torch reference). The 5084-5258 TFLOPS reading is meaningless. Use only the
  `competitor_tflops` (aiter ASM via Python dispatcher) as the comparator.

### Optimization targets
1. **ALL 42 shapes must beat both Gluon LLIR and aiter ASM** (`competitor_tflops`)
2. No single shape gap > 5% vs max(Gluon, aiter)

### DO NOT
- Use warmup < 200 or iters < 500 for performance measurement
- Trust numbers from a GPU running other workloads
- Report benchmark results without specifying warmup/iters used
