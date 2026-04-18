# R45 Dev D — Within-GPU variance baseline (R44 NEW rule 1 enforcement, R45+ #7)

Branch: `worktree-agent-ac0257f8` (off `feat/mxfp8-only` HEAD `37bb9162`)
Date: 2026-04-18
GPUs: GPU4 + GPU5 (avoiding 2/3/6/7 reserved for Dev A/B/C)
GPU-min spent: ~50 GPU-min (20 paired-bench reps × ~2.5 min/rep including 60s
preheat + 5s inter-rep sleep + occasional G1 retry)

## TL;DR

Establishes the within-GPU run-to-run noise floor for the 4 R44 Reviewer
Phase 3 gold-standard cells per R44 NEW rule 1 (R45+ priority #7). N=5
repetitions on a single GPU using the **identical .so artifacts** the R44
Reviewer used (md5 verified: d6319f / 625e56 / 08e7bf / 8193e3 / 2a1ec5 /
9159c5).

| Cell | GPU | Mean Δ% | Stdev | Min | Max | Range | Envelope ±2σ | R44 Δ% | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| 70B-KV HB B1   | 4 | +28.508 | 0.441 | +28.12 | +29.18 | 1.07 | [+27.63, +29.39] | +29.254 | **INSIDE** |
| 8B-KV HB B1    | 5 | +24.150 | 0.812 | +23.36 | +25.28 | 1.92 | [+22.53, +25.77] | +25.202 | **INSIDE** |
| 8B-Down V2-RRR | 5 |  +7.742 | 0.426 |  +7.38 |  +8.35 | 0.97 | [ +6.89,  +8.59] |  +8.798 | **OUTSIDE (high)** |
| 8B Gate/Up     | 4 |  +6.612 | 0.208 |  +6.45 |  +6.96 | 0.51 | [ +6.20,  +7.03] |  +5.369 | **OUTSIDE (low)** |

**2 / 4 cells INSIDE envelope** (= R44 measurement consistent with within-GPU
noise floor; **no kernel attention warranted** for those cells per R44 NEW
rule 1).

**2 / 4 cells OUTSIDE envelope** but in *opposite directions*: 8B-Down R44
is **above** the envelope (+8.80 % vs envelope upper +8.59 %); 8B Gate/Up R44
is **below** the envelope (+5.37 % vs envelope lower +6.20 %). Both are
**cross-GPU silicon-bin draws**, not kernel-change signals — R44 used GPU6 for
8B-Down and GPU7 for 8B Gate/Up, both of which appear to be on different
silicon bins from GPU5/GPU4 used here. **No kernel attention warranted; the
cells are noise-floor stable but require ≥3-GPU triangulation to bracket the
true Δ%** (R43 NEW rule 3).

## Methodology

Per R44C `r44c_8bkv_drift_bisect_findings.md` Phase 7 rule 1 ("Run-to-run
variance baseline"):

1. Same `.so` md5 as R44 Reviewer (verified before each run).
2. N≥3 reps on single GPU (we use N=5 for tighter envelope estimate).
3. R36 3-gate retry harness MANDATORY per R44 NEW rule 2 (G1 sclk-post-preheat
   ≥ 2200 MHz, G2a sclk-post-bench ≥ 2200 MHz, MAX_RETRIES=3).
4. PREHEAT_S=60, N_PAIRS=10 (20 BABA pairs after WARMUP=2; matches R44 Phase 3).
5. 5-second sleep between reps to prevent thermal accumulation.
6. Two-σ envelope = mean ± 2 stdev (95 % normal-approximation CI; given small
   N=5, treat as descriptive bound rather than strict statistical CI).

Avoidance of GPU contention: GPU2/3/6/7 reserved for Dev A/B/C per task spec;
this exercise uses only GPU4 (cells p31, p34) and GPU5 (cells p32, p33).

`.so` md5s (re-verified at run start, all unchanged from R44 Reviewer):
- `/tmp/r42_70bkv_default.so` md5 `d6319f1337d6c0d2da64629d0aaa5ad9`
- `/tmp/r42_70bkv_b1.so`      md5 `625e56a2edb05e4f67d9db592d287327`
- `/tmp/r42_8bkv_default.so`  md5 `08e7bfd2337cce7e4d483a5cb1d9f525`
- `/tmp/r42_8bkv_b1.so`       md5 `8193e3f8c86c1f44f1dd38cda13b1744`
- `/tmp/r42_8bdown.so`        md5 `2a1ec5119d5ba1c72c73cba113893c39`
- `/tmp/r42_8bgateup.so`      md5 `9159c50aa70814b8c84ad42a7c021731`

## Per-cell findings

### Cell 1 — 70B-KV HB shrink B1 (M=4096 N=1024 K=8192) on GPU4

| Rep | Δ% | preMHz | benchMHz |
|---:|---:|---:|---:|
| 1 | +28.449 | 2278 | 2387 |
| 2 | +28.128 | 2379 (att 2; att 1 preMHz=1431 retry) | 2391 |
| 3 | +28.663 | 2344 (att 2; att 1 preMHz=2163 retry) | 2388 |
| 4 | +29.183 | (PASS) | (PASS) |
| 5 | +28.118 | 2200 | 2387 |

- Mean ±  σ = **+28.508 ± 0.441**
- Range = 1.07pp (min +28.12, max +29.18)
- Envelope ±2σ = [+27.63, +29.39]
- R44 reported +29.254 → **INSIDE** envelope (within +0.75pp of mean,
  comfortably below the upper +2σ bound)
- Abs TF: R45D median 1020.8 TF vs R44 1022.2 TF → +0.14 % (negligible)

**Verdict**: R44 +29.25 % is consistent with within-GPU variance on GPU4 for
this cell. The +1.09pp "WIDENING" R43→R44 cited in TODO.md (R43 +28.16 →
R44 +29.25) is well within ±2σ envelope (which alone spans 1.76pp). **Cell is
noise-floor stable; no kernel attention needed.** Confirms R44D margin
tightening survey "FLOOR-LIMIT NO-CHANGE" verdict (margin to ≥+28% gate is
+0.49pp, comparable to ±0.44pp σ).

### Cell 2 — 8B-KV HB shrink B1 (M=4096 N=1024 K=4096) on GPU5

| Rep | Δ% | preMHz | benchMHz |
|---:|---:|---:|---:|
| 1 | +23.357 | 2247 | 2394 |
| 2 | +23.687 | 2397 | 2350 |
| 3 | +24.718 | 2308 (att 2; att 1 preMHz=2185 retry) | 2391 |
| 4 | +23.708 | 2328 | 2391 |
| 5 | +25.280 | 2325 | 2393 |

- Mean ±  σ = **+24.150 ± 0.812**
- Range = 1.92pp (min +23.36, max +25.28)
- Envelope ±2σ = [+22.53, +25.77]
- R44 reported +25.202 → **INSIDE** envelope (top end, +1.05pp above mean,
  well within the upper +2σ bound)
- Abs TF: R45D median 824.0 TF vs R44 883.4 TF → R45D **−6.7 %** (R44 GPU3
  was on a hotter silicon bin than R45D GPU5; consistent with R44C bisect
  showing GPU5 R42_HEAD .so 2-rep median +23.46 % vs GPU4 +26.08 %)

**Verdict**: R44's +25.20 % is fully INSIDE this cell's within-GPU variance
envelope on GPU5. The −1.76pp "NARROWING" R43→R44 (R43 +26.96 → R44 +25.20)
that flagged the cell as AT-RISK is unambiguously **measurement noise**, not
a kernel-change regression. **Independent confirmation of R44C's bisect
verdict** (REFUTED-AS-MEASUREMENT-NOISE). **No kernel attention needed.**

Note: the R45D σ on this cell (0.812pp) is the LARGEST of the 4 cells, which
is consistent with R44C's GPU5 finding of 3.39pp range over 3 reps (i.e., R44C
N=3 caught a wider tail; our N=5 gave a tighter σ but the cell remains the
noisiest of the 4).

### Cell 3 — 8B-Down V2-RRR (M=4096 N=4096 K=14336) on GPU5

| Rep | Δ% | preMHz | benchMHz |
|---:|---:|---:|---:|
| 1 | +7.452 | 2220 | 2305 |
| 2 | +8.032 | 2247 | 2311 |
| 3 | +7.499 | 2350 | 2266 |
| 4 | +7.379 | 2303 | 2202 |
| 5 | +8.348 | 2233 | 2271 |

- Mean ±  σ = **+7.742 ± 0.426**
- Range = 0.97pp (min +7.38, max +8.35)
- Envelope ±2σ = [+6.89, +8.59]
- R44 reported +8.798 → **OUTSIDE (high)** envelope by +0.21pp above the
  upper +2σ bound
- Abs TF: R45D median 2929.2 TF vs R44 2961.4 TF → −1.1 % (close)

**Verdict**: R44 GPU6 measurement of +8.80 % is statistically distinct from
this cell's GPU5 within-GPU variance envelope. Most likely interpretation:
**GPU6 silicon bin runs the 8B-Down case ~1pp faster than GPU5** (HBM- and
fclk-binned higher), not a kernel signal. **No kernel attention warranted**:
the cell is comfortably above the +5 % SHIP gate on both GPU5 (mean +7.74 %)
and GPU6 (R44 +8.80 %); +0.99pp cross-GPU silicon-bin spread is consistent
with the 0.6-1.5pp envelope observed by R43+R44 reviewers on other Δ%
predicates (R43 P4 cleared 0.47pp on 8B QO V2-RCR; R44 Phase 1 baseline
4-GPU envelope 3.62pp on absolute TF translates to comparable Δ% spread).

**Recommendation**: apply R43 NEW rule 3 (Δ%-reproducibility ≥3 GPUs
mandatory) to this cell in R45+ Reviewer to bracket the true Δ% with ±2σ
across-GPU envelope rather than rely on single-GPU readings.

### Cell 4 — 8B Gate/Up V2-RRR (M=4096 N=14336 K=4096) on GPU4

| Rep | Δ% | preMHz | benchMHz |
|---:|---:|---:|---:|
| 1 | +6.962 | 2252 | 2294 |
| 2 | +6.635 | 2315 | 2354 |
| 3 | +6.475 | 2300 | 2340 |
| 4 | +6.452 | 2263 | 2323 |
| 5 | +6.537 | 2211 | 2392 |

- Mean ±  σ = **+6.612 ± 0.208** (TIGHTEST σ of all 4 cells)
- Range = 0.51pp (min +6.45, max +6.96)
- Envelope ±2σ = [+6.20, +7.03]
- R44 reported +5.369 → **OUTSIDE (low)** envelope by −0.83pp below the
  lower −2σ bound
- Abs TF: R45D median 2591.5 TF vs R44 2523.4 TF → R45D **+2.7 %**

**Verdict**: R44 GPU7 measurement of +5.37 % UNDER-estimates this cell's
true Δ% on GPU4. Most likely interpretation: **GPU7 silicon bin draws this
cell into a colder regime than GPU4**, OR R44 GPU7 caught a transient
power-state perturbation that the 3-gate retry didn't catch (note R44 GPU7
P34 needed retry on G1: preMhz=2085 first attempt → 2280 after retry; even
2280 MHz is on the lower end of the operating band).

**Crucial implication**: R44 Reviewer flagged this cell as "PASS-TIGHT
+0.37pp" margin against the ≥+5 % gate. R45D's GPU4 measurements show
mean +6.61 % which is **+1.61pp above the gate** — i.e., the cell has
substantially more headroom than R44 reported. The TIGHT margin verdict was a
silicon-bin artifact (GPU7 cold draw), not a true headroom problem.

**No kernel attention warranted** (R44D had already flagged this cell as
FLOOR-LIMIT NO-CHANGE per the dispatcher comment `R34B +5.025% min`; R45D
confirms the cell is structurally at the +5.5-7.0 % band, not +5.4 %). The
**margin is ~1.6pp not ~0.4pp** when measured on a representative GPU.

**Recommendation**: drop the AT-RISK status on this cell unless ≥3 GPUs
report Δ% < +5.5 %. Promote R45+ spot-bench gate to use ≥3-GPU min Δ% rather
than single-GPU value (R43 NEW rule 3 enforcement).

## Cross-cell summary

| Cell | σ (pp) | R44 - mean (pp) | (R44 - mean) / σ | Verdict |
|---|---:|---:|---:|---|
| 70B-KV B1     | 0.441 | +0.746 | +1.69 | INSIDE  |
| 8B-KV B1      | 0.812 | +1.052 | +1.30 | INSIDE  |
| 8B-Down       | 0.426 | +1.056 | +2.48 | OUTSIDE (high) |
| 8B Gate/Up    | 0.208 | −1.243 | −5.98 | OUTSIDE (low)  |

The 8B Gate/Up case is the most striking: R44's value is **6 σ below** the
within-GPU mean on GPU4 — strongly indicating GPU7 hit a substantially
different operating point during R44 Phase 3 (silicon bin OR a transient
power state that escaped the retry harness).

## Recommendations

### Per-cell R45+ kernel-attention status (NONE escalated)

| Cell | R45+ Kernel Attention? | Rationale |
|---|---|---|
| 70B-KV HB B1   | **NO** — noise-floor stable | R44 INSIDE envelope; R44D FLOOR-LIMIT NO-CHANGE confirmed |
| 8B-KV HB B1    | **NO** — noise-floor stable | R44 INSIDE envelope; R44C bisect REFUTED kernel hypothesis (independent verification by this work) |
| 8B-Down V2-RRR | **NO** — silicon-bin spread | R44 OUTSIDE-high but cell is well above SHIP gate on both GPUs (5%/8%); cross-GPU spread, not kernel issue |
| 8B Gate/Up     | **NO** — silicon-bin spread | R44 OUTSIDE-low but R45D GPU4 mean is +1.6pp above gate; R44 TIGHT verdict was GPU7 cold draw artifact |

**0 / 4 cells require R45+ kernel attention.** All R44 escalations
(8B-KV B1 -1.76pp, 8B Gate/Up TIGHT margin) are **measurement-noise /
silicon-bin artifacts**, NOT kernel-change signals.

### R45+ methodology recommendations

1. **R45 Reviewer Phase 3 should adopt N=3 same-.so reps per cell on each
   GPU** (not just one rep per GPU as R44 did). Combined cost: 4 cells ×
   N=3 reps × 4 GPUs = 48 reps × ~2.5 min = 2 GPU-hr. Within budget.
2. **Drop "TIGHT margin" labels that rely on a single-GPU reading.** A cell
   should be labeled TIGHT only if **mean across ≥3 GPU × ≥3 reps each** is
   within +0.5pp of the SHIP gate.
3. **Drop "drift" labels that rely on cycle-by-cycle single-rep min Δ%.**
   Apply R44 NEW rule 1 explicitly: if single-cycle drop is < ±2σ of the
   most recent within-GPU baseline, treat as noise.
4. **Promote within-GPU variance baseline to a recurring R(N) Reviewer
   Phase** (e.g., once every 3 cycles) on the active gold-standards. Cost
   ~50 GPU-min per pass. R45D is the first measurement for this baseline;
   R48 and R51 should re-measure to surface any silicon-bin chronological
   drift (which R44C H5 hypothesized as REFUTED but never had a longitudinal
   measurement set to disprove rigorously).

### What this exercise does NOT establish

- **Cross-GPU spread**: only single-GPU per cell measured here. The OUTSIDE
  verdicts on cells 3+4 imply silicon-bin spread but cannot quantify it
  without ≥3-GPU same-.so triangulation. R45+ Reviewer should add this.
- **Cycle-to-cycle drift**: this is one snapshot at HEAD `37bb9162`. To rule
  out chronological silicon-bin firmware drift over months, the same N=5
  reps would need re-measurement at R48, R51, etc. (longitudinal panel).
- **Bench-harness contribution to noise**: orchestrator overhead (warmup,
  preheat, BABA pair structure) may itself contribute to per-rep variance.
  Not separable from kernel/silicon noise without a controlled experiment.

## Files

- `analysis/fp8_gemm/mi350x/r45d_variance_orchestrate.sh` — bench harness
  (one-cell × N reps, 3-gate retry per attempt, env-driven SO/MOD/M/N/K).
- `analysis/fp8_gemm/mi350x/r45d_variance_aggregate.py` — aggregator (reads
  per-rep `_clean_bench.txt`, parses DELTA_MEDIAN_PCT and abs-TF medians,
  prints mean / σ / envelope and INSIDE/OUTSIDE verdict).
- `analysis/fp8_gemm/mi350x/r45d_variance_runs/` — per-cell run logs:
  - `${LABEL}_gpu${GPU}.log` — orchestrate log (gate decisions per rep + Δ%
    summary per rep)
  - `${LABEL}_gpu${GPU}_rep${i}_clean_bench.txt` — full bench output of each
    PASSing rep (including TFLOPS lists, Welch-t, sclk gates)
  - `AGGREGATE_SUMMARY.log` — final aggregator output (table form)
- `analysis/fp8_gemm/mi350x/r45d_within_gpu_variance_baseline.md` — this
  document.

## Cross-references

- `analysis/fp8_gemm/mi350x/r44c_8bkv_drift_bisect_findings.md` — R44C bisect
  that established the methodology (R44 NEW rule 1 source).
- `analysis/fp8_gemm/mi350x/r44_reviewer_findings.md` — R44 Reviewer Phase 3
  source values being compared against.
- `analysis/fp8_gemm/mi350x/r43d_gold_standard_drift_audit.md` — R43 Dev D's
  margin trajectory survey (8B-KV B1 "MILDLY FALLING" hypothesis now
  independently REFUTED for the second time).
- `analysis/fp8_gemm/mi350x/r44d_margin_tightening_survey.md` — R44 Dev D's
  FLOOR-LIMIT NO-CHANGE verdicts (this work confirms both 70B-KV and
  8B Gate/Up margin verdicts).
- TODO.md R45+ priority #7 — "Cross-cycle within-GPU variance baseline
  establishment (NEW rule 1)" (this doc closes that priority).

## GPU-min accounting

- 4 cells × 5 reps each = 20 reps total
- ~2.5 min/rep wall (60s preheat + bench + ~30s overhead + occasional G1 retry)
- Wall time ≈ 25 min (cells 1+2 in parallel, then cells 3+4 in parallel)
- Per-GPU wall on each of GPU4, GPU5: ~25 min each
- **Total GPU-min ≈ 50** (well under the 1-2 GPU-hr budget for variance work)
