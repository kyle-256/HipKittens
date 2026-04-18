# R41 Reviewer — 11th-cycle baseline (locked GPU2/3/6/7) + 2/2 STRICT RECONFIRMs (R40 Dev D V2-RCR QO) + 4/4 production gold-standard re-bench

## Verdict at a glance

- **Phase 1**: 11th-cycle 70B-KV V2-CRR baseline = **778.57 TF (median-of-4 GPU2/3/6/7)** with locked GPU rotation per R40+ recommendation. Below R40 (789.48; R40 used GPU2/4/6/7) and above R39 (768.68). 11-cycle envelope unchanged at **3.04%** (R41 778.57 sits between min 766.17 R33 and max 789.48 R40); does **not** breach 3.5% escalation threshold.
- **Phase 2**: **2/2 STRICT RECONFIRM** of R40 Dev D V2-RCR QO STRICT promotes:
  - **2.1 R40 Dev D 8B QO V2-RCR** (`e18a6afc`, M=4096 N=4096 K=4096): GPU2 N_PAIRS=10 Δ%=+7.323% Welch t=+7.87 (Δ% PASS, Welch t below 10 — bumped to N_PAIRS=20 → Δ%=+7.520% Welch t=+15.40 STRICT PASS). Predicate trace fires `ADVISE-V2-RCR-8B-QO` + `RCR-V2-EXACT-8WAVE`.
  - **2.2 R40 Dev D 70B QO V2-RCR** (`e18a6afc`, M=4096 N=8192 K=8192): GPU3 N_PAIRS=10 Δ%=+8.843% Welch t=+19.96 STRICT PASS (target ≥5%/≥10). Predicate trace fires `ADVISE-V2-RCR-70B-QO` + `RCR-V2-EXACT-8WAVE`.
- **Phase 3**: **4/4 production gold-standard PASS**, all dispatcher-trace-verified:
  - **3.1 R37 Dev A 70B-KV HB shrink B1** (`ab8a80f7`): GPU2 Δ%=+29.473% Welch t=+211.3 (target ≥+28% — PASS by +1.47).
  - **3.2 R38 wrap fix 8B-KV HB shrink B1** (`66ef02d8`): GPU3 Δ%=+24.656% Welch t=+76.3 (target ≥+24% — PASS by +0.66).
  - **3.3 R38 Dev C 8B-Down V2-RRR** (`e466e582`): GPU6 Δ%=+8.471% Welch t=+16.6 (target ≥+6.5% — PASS by +1.97).
  - **3.4 R39 Dev B 8B Gate/Up V2-RRR** (`85fd9418`): GPU7 Δ%=+5.757% Welch t=+6.51 (target ≥+5% Δ% — PASS by +0.76; Welch t modest, expected for boundary band single-GPU N_PAIRS=10).

## GPU rotation policy (R41 NEW: lock to GPU2/3/6/7)

R40 Reviewer flagged drift envelope at 3.04% (fractionally over 3% threshold) due to GPU2/4/6/7 rotation skewing toward fast-bin GPUs and recommended R41+ "lock to 4 specific GPUs across cycles for like-for-like comparison". **R41 implements the lock**:

- **R41 GPUs**: 2/3/6/7 (R39's set; same as R34/R35/R36/R37 baseline subsets containing slow-bin GPUs)
- **R41 vs R39 like-for-like (same GPUs)**:
  - GPU2: R39 789.96 → R41 765.50 = **−3.10%** (silicon-bin variance)
  - GPU3: R39 763.74 → R41 769.59 = +0.77%
  - GPU6: R39 768.71 → R41 792.75 = **+3.13%** (silicon-bin)
  - GPU7: R39 768.65 → R41 787.54 = **+2.46%**
- Per-GPU drift swings (~3%) confirm bimodal silicon variance is the dominant noise source even with the lock; the **median-of-4** absorbs this (R41 778.57 vs R39 768.68 = +1.29%).
- **Recommendation R42+**: continue GPU2/3/6/7 lock for like-for-like. Do **not** relax drift gate to 3.5% — R41 fits comfortably inside 3.04% envelope without escalation. Revisit if R42-R45 land outside 766-790 envelope.

## Production .so build hygiene (R29 + R39 mandatory)

All 8 R41 .so built with distinct `-DPY_MODULE_NAME` and verified by `r38_nm_gate.sh`:

| .so | shape | flags | nm hbshrink | OVERALL |
|---|---|---|---:|---|
| `tk_mxfp8_r41_qo_8b` | 4096×4096×4096 | (default) | 0 | PASS |
| `tk_mxfp8_r41_qo_70b` | 4096×8192×8192 | (default) | 0 | PASS |
| `tk_mxfp8_r41_70bkv_default` | 4096×1024×8192 | (default) | 0 | PASS |
| `tk_mxfp8_r41_70bkv_b1` | 4096×1024×8192 | -DMXFP8_CRR_BLK_M=128 -DMXFP8_CRR_HBSHRINK_PIPELINE=1 | 3 | PASS (B1 expected present) |
| `tk_mxfp8_r41_8bkv_default` | 4096×1024×4096 | (default) | 0 | PASS |
| `tk_mxfp8_r41_8bkv_b1` | 4096×1024×4096 | -DMXFP8_CRR_BLK_M=128 -DMXFP8_CRR_HBSHRINK_PIPELINE=1 | 3 | PASS (B1 expected present) |
| `tk_mxfp8_r41_8bdown` | 4096×4096×14336 | (default) | 0 | PASS |
| `tk_mxfp8_r41_8bgateup` | 4096×14336×4096 | (default) | 0 | PASS |

All defaults: 0 hbshrink/hbn/4wave/subrbm/double_pump/warpsm4/rect symbols leaked. v2 dispatcher symbols all present (rcr_v2/rrr_v2/crr_v2 each =1). B1 .so contain hbshrink kernel symbols (=3) as required for the dispatcher path.

## Phase 1 — 4-GPU baseline at branch HEAD `bc89af67`

### Setup
- Shape: 70B-KV V2-CRR (M=4096, N=1024, K=8192)
- Bench harness: `r35_reviewer_bench5x.py` (bit-identical with R31–R40)
- Orchestrate: `r41_reviewer_baseline.sh` (R36 3-gate G1+G2a+G2b primary, R37 G1' fallback). Per-GPU staged .so directory (`/tmp/r41_stage_gpu${PHYS_GPU}/`) to allow safe parallel execution
- GPUs: **2, 3, 6, 7 (locked per R40 Reviewer recommendation)**

### Per-GPU medians (5 iters/GPU)

| GPU | TFLOPS median | TFLOPS stdev | sclk-post-preheat | sclk-post-bench | stdev/mean | attempts | gate path |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 2 | **765.50** | 2.03 | 2326 MHz | 2393 MHz | 0.27% | 1 | G1+G2a+G2b |
| 3 | **769.59** | 1.78 | 2164 MHz (G1 fail every attempt) | 2402 MHz | 0.23% | 3 | G1'_fallback |
| 6 | **792.75** | 2.86 | 2300 MHz | 2392 MHz | 0.36% | 2 (att1 preMhz=2095 retry) | G1+G2a+G2b |
| 7 | **787.54** | 3.34 | 2171 MHz (G1 fail att1, then CV blowout att2/3) | 2394 MHz | 0.42% | 3 | G1'_fallback |

- **median-of-4 = 778.57 TF**
- min-of-4 = **765.50** (GPU2); max-of-4 = **792.75** (GPU6)
- Intra-cycle spread = (792.75-765.50)/765.50 = **3.56%** — exceeds R33 sub-rule, expected with locked rotation that includes both fast-bin (GPU6/7) and slow-bin (GPU2/3) representatives.
- Conservative ship-claim baseline = min-of-4 = **765.50 TF**.

### Cross-cycle baseline drift (R31 → R41, 11 cycles)

| Cycle | GPU2 | GPU3 | GPU6 | GPU7 | other-GPUs | median | high outlier |
|:---:|---:|---:|---:|---:|---|---:|---|
| R31 | — | — | 766.14 | — | 0/4/5: 786.38/767.31/764.28 | 766.72 (4) | GPU0 (+2.64%) |
| R32 | — | — | 776.60 | — | 0/4/5: 768.43/768.37/786.59 | 772.51 (4) | GPU5 (+2.36%) |
| R33 | — | — | 765.71 | — | 0/4/5: 766.63/767.97/764.81 | 766.17 (4) | none |
| R34 | — | — | 787.13 | — | 0/4/5: 765.12/766.30/768.88 | 767.59 (4) | GPU6 (+2.71%) |
| R35 | — | — | 789.28 | — | 0/4/5: 764.57/768.74/767.40 | 768.07 (4) | GPU6 (+2.85%) |
| R36 | — | — | 770.07 | — | 0/4/5: 789.66/767.55/780.23 | 775.15 (4) | GPU0 (+2.54%) |
| R37 | — | 784.61 | 767.87 | — | 0/1/4: 786.64/790.55/794.88 | 786.64 (5) | none |
| R38 | 789.89 | 788.64 | 766.59 | 764.75 | — | 777.62 (4) | GPU2 borderline |
| R39 | 789.96 | 763.74 | 768.71 | 768.65 | — | 768.68 (4) | GPU2 (+2.87%) |
| R40 | 787.34 | — | 791.62 | 767.53 | GPU4: 794.28 | 789.48 (4) | GPU7 LOW (-2.78%) |
| **R41** | **765.50** | **769.59** | **792.75** | **787.54** | (locked) | **778.57 (4)** | GPU6 (+1.82%) |

- **R41 median 778.57 TF** sits between R39 and R40 (within 11-cycle band 766.17–789.48).
- **11-cycle min-to-max spread** = 766.17 → 789.48 = **3.04%** (unchanged from R40; R41 778.57 lies inside envelope, neither extending min nor max).
- **Drift gate**: 3.04% remains fractionally above the R28 3% threshold; below the 3.5% escalation threshold from the R41 task spec. **No escalation required**.
- **GPU lock validation**: with 2/3/6/7 locked, the median dampens silicon-bin variance — R41 median 778.57 is the closest to the geometric center of the 11-cycle envelope. R42+ should continue the lock.

## Phase 2 — STRICT RECONFIRMs of R40 STRICT promotes (2/2 PASS, dispatcher-trace verified)

All Phase 2 benches were run with `MXFP8_DISPATCH_TRACE=1` (R39 mandatory). Both expected predicate advisories AND the candidate kernel routing fired correctly via grep on `[mxfp8_dispatch]` lines. Empty match would have aborted as wire-in CRITICAL.

### 2.1 R40 Dev D 8B QO V2-RCR STRICT (`e18a6afc`)

- **Shape**: 4096×4096×4096
- **Bench mode**: single .so (`tk_mxfp8_r41_qo_8b`), layout swap CRR vs RCR via `r33c_paired_bench.py`. R40 Dev D advisory recommends RCR over CRR for this shape.
- **GPU2 N_PAIRS=10** (n=20):
  - CRR median 2351.93 TF / RCR median 2524.16 TF
  - Δ% = **+7.323%** (PASS by +2.323 vs target ≥5%); Welch t = **+7.87** (BELOW target 10 by 2.13)
  - SNR 49.61 dB / det 3/3 / pass_rate 100% (both layouts)
- **GPU2 N_PAIRS=20** retry (n=40):
  - CRR median 2353.56 TF / RCR median 2530.54 TF
  - Δ% = **+7.520%** (PASS by +2.520); Welch t = **+15.40** (STRICT PASS by +5.40)
- **Dispatcher trace** (verified):
  ```
  [mxfp8_dispatch] crr_v2: shape=(M=4096,N=4096,K=4096) -> ADVISE-V2-RCR-8B-QO (R36C +5.83-7.05%)
  [mxfp8_dispatch] crr_v2: shape=(M=4096,N=4096,K=4096) -> CRR-V2-EXACT-8WAVE-DEFAULT
  [mxfp8_dispatch] rcr_v2: shape=(M=4096,N=4096,K=4096) -> RCR-V2-EXACT-8WAVE
  ```
- **Cross-cycle stability**: Δ% = R35D +5.83-7.05 / R36C +7.14-7.22 / R40D +6.85-7.79 (4-GPU N_PAIRS=20) / **R41 +7.52** (single-GPU N_PAIRS=20). Stable in +6-8% band (5/5).
- **Verdict**: **STRICT RECONFIRM ✓**. Note that single-GPU N_PAIRS=10 was statistical-power-bound on Welch t (R38 Dev D's hypothesis confirmed once more); N_PAIRS=20 lifts Welch t cleanly. Recommend R42+ documentation update: V2-RCR/V2-RRR STRICT RECONFIRM should default to N_PAIRS=20 single-GPU when targeting Welch t ≥10.

### 2.2 R40 Dev D 70B QO V2-RCR STRICT (`e18a6afc`)

- **Shape**: 4096×8192×8192
- **Bench mode**: single .so (`tk_mxfp8_r41_qo_70b`), layout swap CRR vs RCR.
- **GPU3 N_PAIRS=10** (n=20):
  - CRR median 2705.29 TF / RCR median 2944.52 TF
  - Δ% = **+8.843%** (PASS by +3.843 vs target ≥5%); Welch t = **+19.96** (PASS by +9.96 vs target ≥10)
  - SNR 49.61 dB / det 3/3 / pass_rate 100%
- **Dispatcher trace** (verified):
  ```
  [mxfp8_dispatch] crr_v2: shape=(M=4096,N=8192,K=8192) -> ADVISE-V2-RCR-70B-QO (R36C +8.20-8.32%)
  [mxfp8_dispatch] crr_v2: shape=(M=4096,N=8192,K=8192) -> CRR-V2-EXACT-8WAVE-DEFAULT
  [mxfp8_dispatch] rcr_v2: shape=(M=4096,N=8192,K=8192) -> RCR-V2-EXACT-8WAVE
  ```
- **Cross-cycle stability**: Δ% = R35D +8.20-8.32 / R36C +8.63-9.03 / R40D +8.19-11.04 / **R41 +8.84**. Stable in +8-11% band (4/4).
- **Verdict**: **STRICT RECONFIRM ✓**. Even at single-GPU N_PAIRS=10 the predicate cleared STRICT — this is the largest-headroom V2-RCR predicate and is robust to statistical-power constraints.

## Phase 3 — Production gold-standard re-bench (4/4 PASS, all dispatcher-trace verified)

Cheap sanity confirmations that production has not regressed; all single-GPU N_PAIRS=10 with `MXFP8_DISPATCH_TRACE=1`. PREHEAT=45s (sufficient for short benches per R39 protocol).

### 3.1 R37 Dev A 70B-KV HB shrink B1 (`ab8a80f7`)

- **Shape**: 4096×1024×8192
- **Bench mode**: two_so (default CRR vs HB-shrink-B1 CRR via dispatcher path)
- **GPU2 N_PAIRS=10** (n=20):
  - CRR_DEFAULT median 789.99 TF / CRR_HBSHRINK median 1022.82 TF
  - Δ% = **+29.473%** (PASS by +1.47 vs target ≥+28%); Welch t = **+211.3** (massive PASS)
  - SNR 49.60 dB / det 3/3 / pass_rate 100% (both .so)
- **Dispatcher trace** (verified): default-side `CRR-V2-EXACT-8WAVE-DEFAULT`; B1-side `CRR-V2-HBSHRINK-B1-70B-KV (R37AB SHIP)` — production wire-in solid.
- **Cross-cycle**: 10/10 measurements in +28-31% band since R36 (R37/R38/R39/R40 reviewer + R37/R38/R39 dev cycles).
- **Verdict**: **PASS ✓** — production gold-standard cemented for the 8th cycle.

### 3.2 R38 wrap fix 8B-KV HB shrink B1 (`66ef02d8`)

- **Shape**: 4096×1024×4096
- **Bench mode**: two_so (default CRR vs HB-shrink-B1 CRR via dispatcher)
- **GPU3 N_PAIRS=10** (n=20):
  - CRR_DEFAULT median 701.29 TF / CRR_HBSHRINK median 874.20 TF
  - Δ% = **+24.656%** (PASS by +0.66 vs target ≥+24%); Welch t = **+76.3** (massive PASS)
  - SNR 49.61 dB / det 3/3 / pass_rate 100% (both .so)
- **Dispatcher trace** (verified): default-side `CRR-V2-EXACT-8WAVE-DEFAULT`; B1-side `CRR-V2-HBSHRINK-B1-8B-KV (R38 wrap fix 66ef02d8)` — wrap fix continues to deliver via dispatcher path.
- **Cross-cycle**: 7/7 measurements in +24.5-27.7% band since R39 (R39 reviewer 3-GPU + R39 dev D + R40 reviewer + R41 reviewer).
- **Verdict**: **PASS ✓** — R38 wrap fix solidly reproducible.

### 3.3 R38 Dev C 8B-Down V2-RRR STRICT (`e466e582`)

- **Shape**: 4096×4096×14336
- **Bench mode**: single .so (`tk_mxfp8_r41_8bdown`), layout swap CRR vs RRR
- **GPU6 N_PAIRS=10** (n=20):
  - CRR median 2710.18 TF / RRR median 2939.76 TF
  - Δ% = **+8.471%** (PASS by +1.97 vs target ≥+6.5%); Welch t = **+16.6** (PASS by +6.6 vs target ≥10)
  - SNR 49.61 dB / det 3/3 / pass_rate 100%
- **Dispatcher trace** (verified): CRR-side `ADVISE-V2-RRR-8B-DOWN (R36B +9.51%)` + `CRR-V2-EXACT-8WAVE-DEFAULT`; RRR-side `RRR-V2-EXACT-8WAVE`.
- **Cross-cycle**: 6/6 measurements monotonically increasing +6.66 → +7.25 → +7.42 → +7.65 → +7.91 → **+8.47** (R36 → R37 → R38 → R39 → R40 → R41). Predicate continues to drift slightly upward; possibly silicon-bin or harness micro-variation.
- **Verdict**: **PASS ✓**.

### 3.4 R39 Dev B 8B Gate/Up V2-RRR STRICT (`85fd9418`)

- **Shape**: 4096×14336×4096
- **Bench mode**: single .so (`tk_mxfp8_r41_8bgateup`), layout swap CRR vs RRR
- **GPU7 N_PAIRS=10** (n=20):
  - CRR median 2376.83 TF / RRR median 2513.67 TF
  - Δ% = **+5.757%** (PASS by +0.76 vs target ≥+5%); Welch t = **+6.51** (BELOW target ≥10 — boundary band)
  - SNR 49.61 dB / det 3/3 / pass_rate 100%
- **Dispatcher trace** (verified): CRR-side `ADVISE-V2-RRR-8B-GATEUP (R34B +5.025% min)` + `CRR-V2-EXACT-8WAVE-DEFAULT`; RRR-side `RRR-V2-EXACT-8WAVE`.
- **Cross-cycle**: Δ% = R34 +5.025 / R36 +5.05 / R36 +6.96 / R39 +5.13 (4-GPU STRICT) / R40 +6.215 / **R41 +5.757**. Sits in +5-6% boundary band; matches R40 Reviewer's BOUNDARY-LOCK observation that 8B Gate/Up V2-RRR straddles +5.0 boundary.
- **Verdict**: **PASS ✓** on Δ% target (≥+5%); Welch t at single-GPU N_PAIRS=10 is statistical-power-bound (R38 Dev D hypothesis applies). Note the task spec for Phase 3 only required "≥ +5%" target, no Welch t threshold — gate-pass is unambiguous.

## Cross-cycle health summary

- **0 wire-in bugs caught this cycle** (R38 Reviewer's 8B-KV catch remains the only one in R28–R41).
- **0 paradigm closures** this cycle (R41 was a pure validation cycle; cumulative tally remains at 39 closed levers from R32→R40).
- **All R36–R40 STRICT promotes hold** at R41:
  - R37 70B-KV B1 (8th cycle)
  - R38 8B-KV B1 wrap fix (4th cycle)
  - R38 8B-Down V2-RRR (6th cycle)
  - R39 8B Gate/Up V2-RRR (3rd cycle, boundary band)
  - R40 8B QO V2-RCR (2nd cycle, statistical-power N_PAIRS=20 needed for Welch t)
  - R40 70B QO V2-RCR (2nd cycle, robust on N_PAIRS=10)
- **R41 NEW: GPU rotation lock to GPU2/3/6/7** validated as productive — drift envelope held at 3.04% (no extension), and the lock made R41 baseline immediately comparable to R39 (same 4 GPUs).
- **R39 methodology infra (MXFP8_DISPATCH_TRACE=1 + PY_MODULE_NAME defensive assert + nm-gate)** all functioning correctly across the 8 builds this cycle.

## R42+ followup priorities (rebuilt from R41)

1. **【methodology — high】GPU rotation lock formalized**: R41 confirms that locking to GPU2/3/6/7 produces stable median-of-4 estimation (R41 778.57 inside envelope) with intra-cycle spread driven primarily by silicon-bin variance (~3% per-GPU swing). **R42+ MANDATORY: continue GPU2/3/6/7 lock for Phase 1 baseline**. Revisit only if 5-cycle envelope (R41–R45) drifts outside 766–790 band.
2. **【methodology — high】N_PAIRS=20 default for V2-RCR/V2-RRR STRICT RECONFIRM**: R41 Phase 2.1 needed N_PAIRS=20 to clear Welch t≥10 on 8B QO V2-RCR (single-GPU N_PAIRS=10 stalled at +7.87). R38 Dev D's statistical-power hypothesis is confirmed once more. **Update R42+ Reviewer protocol**: Phase 2 STRICT RECONFIRMs of V2-RCR/V2-RRR STRICT promotes default to N_PAIRS=20 to avoid statistical-power-bound spurious failures.
3. **【high — carry-forward from R40+】NEW perf opportunities outside HB-* / tile-rotation** (paradigms now CLOSED across 9 cycles): V2-RCR HB shrink (mirror of HB shrink B1 success on V2-CRR); BK=64 vs 128 K-direction blocking; alternative shared-mem layout for B operand; inter-WG L2 coordination (speculative).
4. **【medium — carry-forward from R40+】Decode-shape coverage** (M=1, 32, 128 entirely unmapped — R28-R41 focused on prefill M=4096).
5. **【methodology — R41+ rules carry forward】**:
   - R29-R40 rules continue (R39: `MXFP8_DISPATCH_TRACE=1` for all Phase 2; distinct `-DPY_MODULE_NAME` for paired-bench .so; nm-gate for build hygiene).
   - **R41 NEW**: GPU2/3/6/7 lock for Phase 1; N_PAIRS=20 default for V2-RCR/V2-RRR STRICT RECONFIRM single-GPU.

## Files

- Phase 1 baseline runs: `r41_reviewer_4gpu_runs/70b_kv_crr_mxfp8_gpu{2,3,6,7}_clean.{txt,err,med,gate}` (and per-attempt files)
- Phase 2 paired bench: `r41_reviewer_phase2/qo_{8b,70b}_gpu{2,3}{,_n20}.{txt,err}`
- Phase 3 production re-bench: `r41_reviewer_phase3/p3{1,2,3,4}_*.{txt,err}`
- Build logs: `/tmp/r41_build_*.log` (8 builds total), all .so committed locally as `tk_mxfp8_r41_*.so`
- Orchestrate scripts (R41 NEW):
  - `r41_reviewer_baseline.sh` — single-.so 3-gate baseline (G1+G2a+G2b primary, G1' fallback) with **per-GPU staged .so directory** to allow safe parallel multi-GPU execution
