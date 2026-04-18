# R35 Reviewer — 5th-cycle baseline (N=5) + Phase 2 SHIP confirms

## Verdict: 2 R34/R35 SHIPs CONFIRMED + new methodology gap surfaced for R36+

## Phase 1 — 4-GPU baseline at R35 base commit 758ad933

### Setup
- **Shape:** 70B-KV V2-CRR fastpath (M=4096, N=1024, K=8192) — same as R31/R32/R33/R34 baseline
- **Bench harness:** `r35_reviewer_bench5x.py` (bit-identical to r34_reviewer_bench5x.py)
- **Orchestrate:** `r35_reviewer_4gpu_orchestrate.sh` — auto-retry up to 3× on `sclk-post-preheat < 2200 MHz` (R34 rule)
- **Build md5:** `d34018cd127258362e821f1625821aa1` — bit-identical across all 5 (re)builds
- **GPUs:** physical 4, 5, 6, 0 (sequential to deconflict with R35 Dev A on GPU0)

### Per-GPU medians (5 iters/GPU, sorted-mean trim 10%)

| GPU | TFLOPS median | TFLOPS stdev | SNR dB | Det | sclk post-preheat | attempts |
|---:|---:|---:|---:|:---:|---:|---:|
| 4 | 768.74 | 2.59 | 49.60 | PASS | 2281 | 1 |
| 5 | 767.40 | 1.80 | 49.60 | PASS | 2311 | 1 |
| 6 | 789.28 | 1.10 | 49.60 | PASS | 2313 | 2 |
| 0 | 764.57 | 4.58 | 49.60 | PASS | 2219 | 3 |

- **median of 4** = 768.07 TF
- **min of 4** (R33 sub-rule) = 764.57 TF
- **max of 4** = 789.28 TF
- **spread** = 24.71 TF (3.23%)

### Cross-cycle table (R31–R35)

| Cycle | GPU0 | GPU4 | GPU5 | GPU6 | median-of-4 | high outlier (other-3 median, excess%) |
|:---:|---:|---:|---:|---:|---:|---|
| R31 | 786.38 | 767.31 | 764.28 | 766.14 | 766.72 | GPU0 (766.14, +2.64%) |
| R32 | 768.43 | 768.37 | 786.59 | 776.60 | 772.51 | GPU5 (768.43, +2.36%) |
| R33 | 766.63 | 767.97 | 764.81 | 765.71 | 766.17 | none (excess +0.30%) |
| R34 | 765.12 | 766.30 | 768.88 | 787.13 | 767.59 | GPU6 (766.30, +2.71%) |
| R35 | 764.57 | 768.74 | 767.40 | 789.28 | 768.07 | GPU6 (767.40, **+2.85%**) |

### Findings (Phase 1)

1. **5-cycle median is steady at 766.7–772.5 TF** with no monotonic drift across the 5 wraps.
2. **GPU6 is now a 2-cycle-repeat high-state outlier** (R34: 787.13, R35: 789.28).
   - Previously the high outlier rotated (GPU0 → GPU5 → none → GPU6 → GPU6).
   - The first repeat. Recommend R36+ check ECC counters / preheat duty on GPU6 to determine if this is an aging/calibration drift vs. R34 bug.
3. **R34 sclk-post-preheat ≥ 2200 MHz auto-retry rule WORKED** for GPU6 attempt 1 (1714 MHz, 89 TF → retried to 2313 MHz, 789 TF).
4. **NEW methodology gap surfaced** on GPU0 attempt 1:
   - sclk-post-preheat = **2277 MHz** (passed gate ≥ 2200)
   - But mid-bench sclk dropped to 2091/2076 MHz (parallel-agent contention)
   - tflops_median = 364.73 with stdev 23.77 (4× higher than clean runs)
   - **The gate did not catch this regression.**
   - **Recommendation R36+:** add a second gate either (a) `sclk-post-bench ≥ 2200 MHz` OR (b) per-run `stdev/mean ≤ 1%` ratio check. Both would have caught attempt 1.

## Phase 2 — SHIP CONFIRMS for R34 Dev A (c4) + R34 Dev B / R35 Dev A (c5)

### c4 — R33 Dev C 70B KV SHIP (predicate at M=4096 N=1024 K=8192)
- **GPU 5:** RRR vs CRR Δ% = **+10.44%**, Welch t = **+41.26**
- Predicate fires (advisory log line emitted)
- Correctness PASS (snr 49.60 dB)
- **CONFIRMED**

### c5 — R34 Dev B 8B Gate SHIP / R35 Dev A 5th predicate (M=4096 N=14336 K=4096)
- **GPU 5:** RRR vs CRR Δ% = **+5.77%**, Welch t = **+3.75**
- **GPU 6:** RRR vs CRR Δ% = **+5.76%**, Welch t = **+9.19**
- min Δ% = +5.76% (above STRICT gate +5.0%)
- min Welch t = +3.75 (below STRICT gate 10.0; **SHIP-LITE only**)
- Predicate fires (advisory log line emitted) on both GPUs
- Correctness PASS both GPUs (snr 49.61)
- **CONFIRMED at SHIP-LITE** (R34 STRICT classification preserved; R35 verify keeps SHIP-LITE label until 4-GPU triangulation)

## Action items for R36+
1. **High**: implement post-bench-sclk gate OR per-run stdev/mean ratio gate (catch the GPU0 attempt-1 regression class)
2. **Medium**: investigate GPU6 2-cycle-repeat high-state outlier (ECC, preheat)
3. **Low**: extend Phase 2 c5 verify to GPUs 4/0 to confirm STRICT vs LITE classification at 4-GPU triangulation
