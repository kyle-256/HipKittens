# Rocprof Workflow for fp8 grouped (R85)

## Purpose

Per `[[fp8-rrr-attempt-h14]]` finding: HK gap to dense is bandwidth-bound at large B. Need PMC counters to:
1. Confirm bandwidth/L2 hypothesis per shape
2. Identify which shapes split-K could help
3. Validate post-rewrite perf delta

## Counter Set

Minimum useful set for fp8 grouped GEMM:
```
pmc: SQ_BUSY_CYCLES SQ_INSTS_VALU SQ_INSTS_MFMA SQ_INSTS_VMEM SQ_INSTS_LDS
pmc: TCC_HIT TCC_MISS TCC_REQ
pmc: SQ_WAIT_INST_ANY SQ_WAIT_INST_LDS SQ_WAIT_INST_VMEM
pmc: SQ_LDS_BANK_CONFLICT
```

## Workflow

### 1. Single-call script
`Primus-Turbo/scripts/_rocprof_<shape>.py` — quantize + 50 warmup + 50 measure iters.

### 2. Profile run
```
cd /workspace/code/Primus-Turbo
rocprofv3 -i pmc.txt -o /tmp/<shape>.csv /opt/venv/bin/python scripts/_rocprof_<shape>.py
```

### 3. Output analysis
```python
import csv
rows = list(csv.DictReader(open("/tmp/<shape>.csv")))
gemm_rows = [r for r in rows if "grouped_gemm_fp8_kernel" in r["KernelName"]]
hits = sum(int(r["TCC_HIT"]) for r in gemm_rows)
miss = sum(int(r["TCC_MISS"]) for r in gemm_rows)
print(f"L2 hit rate: {hits/(hits+miss):.3f}")
```

### 4. Interpret

| L2 hit rate | Verdict |
|-------------|---------|
| > 0.85 | compute-bound, kernel-internal lever applies |
| 0.50-0.85 | mixed, mfma scheduling lever may help |
| < 0.50 | bandwidth-bound, only algorithmic lever (split-K) helps |

`[[fp8-rrr-attempt-h14]]` already measured: worst grouped @ B=16 = bandwidth-bound. Other shapes need fresh measurement.

## ssh Escape Issue (R80 blocker)

Multi-level `ssh login_node → ssh chi2811 → docker exec` chain corrupts shell escapes for `<<EOF` heredocs and complex `$var` expansion. Workarounds:

1. Pre-write PMC config file locally + scp via login_node (sandbox blocked SCP in current session, may need user help)
2. Run PMC config write inside docker exec via simple `printf` to single file
3. Set up dedicated `_pmc_runner.sh` in PT scripts/ that takes shape name arg

Multi-session: integrate pmc runner into scripts/ as committed infra.

## Use With Multi-Session Work

After P1.2 / P1.3a land, re-run rocprof on same shape set:
- Validate split-K reduces TCC_REQ (less B streaming)
- Validate 32×32 doesn't regress SQ_INSTS_VMEM
- Final TFLOPS estimate from SQ_INSTS_MFMA × ops_per_mfma / kernel_time
