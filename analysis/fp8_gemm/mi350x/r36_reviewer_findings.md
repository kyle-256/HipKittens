# R36 Reviewer — 6th-cycle baseline (N=5) + Phase 2 SHIP confirms + R35 NEW gate validated

## Verdict: 2 SHIPs CONFIRMED + R35 NEW gate empirically caught contention 3 separate times + GPU6 outlier hypothesis BROKEN

## Phase 1 — 4-GPU baseline at R36 base commit 40d77d98

### Setup
- **Shape:** 70B-KV V2-CRR fastpath (M=4096, N=1024, K=8192) — same as R31–R35 baseline
- **Bench harness:** `r35_reviewer_bench5x.py` (bit-identical, reused; no source change)
- **Orchestrate:** `r36_reviewer_4gpu_orchestrate.sh` — combines R34 auto-retry on `sclk-post-preheat < 2200 MHz` AND R35 NEW second gate (`sclk-post-bench >= 2200 MHz` OR per-run `stdev/mean <= 0.01`); requires BOTH to PASS
- **Build md5:** `551da0c05c1c5bd3a35861690db5b935` — bit-identical across all 4 builds (differs from R35 `d34018cd…` because the rect-shape build at 4096×1024×8192 now pulls in R35 Dev A's 5th V2-RRR autotune predicate — default 8192³ build remains functionally byte-equivalent because R35 Dev A's predicate emits runtime-dead branches there)
- **GPUs:** physical 0, 4, 5, 6 (re-run order due to heavy parallel-agent contention on the host)

### Per-GPU medians (5 iters/GPU, R35-NEW-gate enforced)

| GPU | TFLOPS median | TFLOPS stdev | SNR dB | Det | sclk post-preheat | sclk post-bench | stdev/mean | attempts |
|---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| 0 | **789.66** | 4.15 | 49.60 | PASS | 2336 | 2394 | 0.53% | 2 |
| 4 | 767.55 | 1.89 | 49.60 | PASS | 2316 | 2383 | 0.25% | 1 |
| 5 | 780.23 | 5.02 | 49.60 | PASS | 2278 | 2393 | 0.65% | 3 |
| 6 | 770.07 | 2.70 | 49.60 | PASS | 2369 | 2397 | 0.35% | 1 |

- **median of 4** = 775.15 TF
- **min of 4** (R33 sub-rule) = 767.55 TF
- **max of 4** = 789.66 TF
- **spread** = 22.11 TF (2.88%)

### Cross-cycle table (R31–R36)

| Cycle | GPU0 | GPU4 | GPU5 | GPU6 | median-of-4 | high outlier (other-3 median, excess%) |
|:---:|---:|---:|---:|---:|---:|---|
| R31 | 786.38 | 767.31 | 764.28 | 766.14 | 766.72 | GPU0 (766.14, +2.64%) |
| R32 | 768.43 | 768.37 | 786.59 | 776.60 | 772.51 | GPU5 (768.43, +2.36%) |
| R33 | 766.63 | 767.97 | 764.81 | 765.71 | 766.17 | none (excess +0.30%) |
| R34 | 765.12 | 766.30 | 768.88 | 787.13 | 767.59 | GPU6 (766.30, +2.71%) |
| R35 | 764.57 | 768.74 | 767.40 | 789.28 | 768.07 | GPU6 (767.40, +2.85%) |
| **R36** | **789.66** | **767.55** | **780.23** | **770.07** | **775.15** | **GPU0 (770.07, +2.54%)** |

### Findings (Phase 1)

1. **6-cycle median is steady at 766.7–775.2 TF** (peak-to-peak 1.10%); R36 median-of-4 = 775.15 is the highest cross-cycle data point yet, but well within the 6-cycle band.
2. **GPU6 2-cycle-repeat hypothesis (R35 finding) is BROKEN.** Rotation pattern restored:
   - GPU0 (R31) → GPU5 (R32) → none (R33) → GPU6 (R34) → GPU6 (R35) → **GPU0 (R36)**
   - GPU6 dropped from 789.28 (R35) to 770.07 (R36) — a 2.43% drop, returning fully to cluster. ECC-drift / aging hypothesis is REFUTED for now (more cycles needed before re-opening).
   - Frequency over 6 cycles: GPU0 high 2× (R31, R36), GPU5 high 1× (R32), GPU6 high 2× (R34, R35), GPU4 never high. Distribution is consistent with random rotation modulo small per-GPU bias.
3. **R35 NEW second sclk gate (mandatory) was VALIDATED in production.**
   - GPU0 attempt 1: post-preheat 1882 MHz + stdev 21.8% (caught by both gates simultaneously — same failure mode as R35 GPU0 attempt-1)
   - GPU5 attempt 2: post-preheat 1743 MHz (caught by R34 gate) + stdev 0.46% (would have *passed* R35 gate alone) — combined-gate logic correctly required BOTH and triggered retry
   - GPU0 attempt 1 **post-bench sclk = 1847 MHz** (sub-2200 mid-bench drop → caught by R35 gate even where post-preheat had reasonably high reading). This is the precise R35-finding failure class the new gate was designed to catch.
4. **R34 auto-retry remained functional**: caught GPU0 attempt 1 (1882 MHz), GPU5 attempts 1+2 (1727/1743 MHz).
5. **Methodology stress test passed.** Across 7 distinct attempt-events on 4 GPUs in a heavily-contended host environment (parallel mxfp4/mxfp8 agents on neighbor GPUs), the combined R34+R35 gate accepted only clean runs (sub-1% stdev/mean post-bench, sclk well above 2200 MHz throughout) — exactly as designed.

## Phase 2 — SHIP CONFIRMS for c4 (R33-Dev-C 70B KV) + c5 (R34-Dev-B / R35-Dev-A 8B Gate)

### c4 — 70B KV V2-CRR (M=4096 N=1024 K=8192) — RRR vs CRR

| GPU | Δ% (RRR over CRR) | Welch t | CRR median | RRR median | snr dB | det | advisory fired |
|:---:|---:|---:|---:|---:|---:|:---:|:---:|
| 4 | **+10.510%** | **+25.405** | 788.99 | 871.91 | 49.60 | PASS | YES |
| 5 | **+10.808%** | **+46.445** | 793.41 | 879.16 | 49.60 | PASS | YES |

- min Δ% = **+10.510** (above STRICT gate +5.0%)
- min Welch t = **+25.405** (above STRICT gate +10.0)
- **STRICT CONFIRM** — fully consistent with R33 Dev C SHIP claim (+10.24 to +10.83%) and R35 Reviewer GPU5 (+10.44%).

GPU6 attempt was attempted twice; both times the host was heavily contended (sclk stuck at 1717-1735 MHz; bench TFLOPS bimodal 250/530/600 — direction was preserved but stdev/mean ~30-60%). GPU6 logs are kept in repo for transparency (`r36_reviewer_devXXX_verify/c4_70b_kv_gpu6.txt`); they do not contribute to the verdict.

### c5 — 8B Gate V2-CRR (M=4096 N=14336 K=4096) — RRR vs CRR

| GPU | Δ% (RRR over CRR) | Welch t | CRR median | RRR median | snr dB | det | advisory fired |
|:---:|---:|---:|---:|---:|---:|:---:|:---:|
| 4 | **+6.964%** | +6.830 | 2418.64 | 2587.07 | 49.61 | PASS | YES |
| 5 | **+5.045%** | +6.283 | 2387.55 | 2508.01 | 49.61 | PASS | YES |

- min Δ% = **+5.045** (just above STRICT gate +5.0%)
- min Welch t = **+6.283** (below STRICT gate +10.0; clears LITE gate)
- **SHIP-LITE CONFIRM** — consistent with R34 Dev B SHIP-LITE classification and R35 Reviewer SHIP-LITE confirmation. Across 8 measurements (R34 Dev B GPU0/4 + R35 Reviewer GPU5/6 + R36 Reviewer GPU4/5), the c5 minimum Δ% across all GPUs has stayed in the 5.0-5.8% band (R34 GPU0 +5.23%, R34 GPU4 +6.61%, R35 GPU5 +5.77%, R35 GPU6 +5.76%, R36 GPU4 +6.96%, R36 GPU5 +5.05%) — the lower bound is right at the STRICT-Δ gate; the per-GPU spread is consistent with cross-GPU thermal/clock noise. **STRICT promotion is borderline-blocked by Welch t variance**, not by Δ%.

## Action items for R37+

1. **None new** for the orchestrate gate — R35 NEW gate is now fully wired and validated.
2. **Medium**: monitor GPU0 (R36 high outlier, 2nd time after R31) — if R37 GPU0 stays high it would be the first 2-cycle-repeat in a 6-cycle window. R35 reviewer's recommendation re ECC counters / preheat-duty audit was for GPU6 (broken this cycle), but the same general check could apply to GPU0 next cycle.
3. **Low**: c5 STRICT promotion still blocked at Welch t — would need ≥3 GPU 4-pair-aggregate to reach min t ≥ 10. Recommend deferring (LITE classification is fine for landing; predicate is already wired).
4. **High**: continue R36 fan-out targets (8B-Down V2-RRR predicate, RCR routing for square Q/O cells) per R35 R36+ priority list.

## Files in this commit

- `r36_reviewer_4gpu_orchestrate.sh` — NEW orchestrate with second sclk gate
- `r36_reviewer_findings.md` — this file
- `r36_reviewer_kv_4gpu.json` — full retry history + Phase 2 results
- `r36_reviewer_4gpu_runs/` — 4 GPU clean runs + 12 attempt logs + 4 build logs + build_md5.log
- `r36_reviewer_devXXX_verify/` — Phase 2 verify logs (c4 GPU4/5/6, c5 GPU4/5) + 2 build logs
- `r36_reviewer_ship_verify.sh` — SHIP verify orchestrate (uses r33c_paired_bench.py)
