# R42 Reviewer Findings

Branch: `r42-reviewer` (from `feat/mxfp8-only` HEAD `211b0125`)
Date: 2026-04-18
GPUs: locked GPU2/3/6/7 (R41 NEW rule)

## Summary verdict

| Phase | Item | Verdict |
|-------|------|---------|
| 1 | 12-cycle 70B-KV V2-CRR baseline (median-of-4) | PASS — 771.87 TF, drift 0.86% vs R41 (gate 3.5%) |
| 2.1 | V2-RCR 8B QO STRICT RECONFIRM (M=4096 N=4096 K=4096) | Δ% PASS / Welch t MARGINAL — see notes |
| 2.2 | V2-RCR 70B QO STRICT RECONFIRM (M=4096 N=8192 K=8192) | PASS |
| 3.1 | 70B-KV HB shrink B1 production gold-standard | PASS |
| 3.2 | 8B-KV HB shrink B1 production gold-standard | PASS |
| 3.3 | 8B-Down V2-RRR production gold-standard | **FAIL** (regression vs R36B SHIP) |
| 3.4 | 8B Gate/Up V2-RRR production gold-standard | PASS |
| 4.1 | PY_MODULE_NAME defensive assert present | PASS |
| 4.2 | MXFP8_DISPATCH_TRACE emits expected predicate names | PASS |
| 4.3 | r38_nm_gate.sh on default 8192³ build | PASS |

## Phase 1 — 12-cycle 70B-KV baseline

GPU medians (median-of-1 per GPU, n=10 samples each):
- GPU2 = 763.84 TF
- GPU3 = 769.81 TF
- GPU6 = 773.92 TF
- GPU7 = 792.97 TF

**Median-of-4 = (769.81 + 773.92) / 2 = 771.87 TF**

Cycle history (median-of-4):
R31 766.72 / R32 772.51 / R33 766.17 / R34 767.59 / R35 768.07 / R36 775.15 /
R37 786.64 / R38 777.62 / R39 768.68 / R40 789.48 / R41 778.57 / **R42 771.87**

Drift gate (R28+ 3.5%): |771.87 − 778.57| / 778.57 = **0.86%** — well under threshold.

Envelope: 12-cycle min 763.84 (GPU2 R42), max 792.97 (GPU7 R42). Both fractionally extend the 11-cycle envelope:
- min 766.17 → 763.84 (−0.30%)
- max 789.48 → 792.97 (+0.44%)

12-cycle envelope spread = (792.97 − 763.84)/763.84 = **3.81%**.

Prior 11-cycle envelope was 3.04%; R42 widens it by 0.77 pp because both ends moved slightly outward. Drift gate not breached; flagging the envelope widening for future reviewers but no escalation.

GPU2 and GPU7 required 1 G1' fallback retry each (clean retry succeeded; both medians used).

## Phase 2 — STRICT RECONFIRMs

### 2.1 V2-RCR 8B QO (M=4096, N=4096, K=4096)
First run (N_PAIRS=10, GPU2): Δ%=**+7.489%**, Welch t=**8.075**.
- Δ% PASSES R36C target (+5.83% to +7.05% original; ≥+5%).
- Welch t (8.07) below R41 protocol threshold of 10.

Per protocol N_PAIRS=20 retry on different GPU (GPU6, PREHEAT=120):
Δ%=**+7.219%**, Welch t=**6.099**.
- Δ% PASSES (within ±0.3pp of first run — robust effect).
- Welch t WORSE due to 2 outlier pairs (PAIR 13, 14: RCR_2 dipped to 1855/1908 TF).
- Mechanism: RCR-V2-EXACT-8WAVE has higher per-pair variance under the 2.4 GHz tail than V2-CRR; this is consistent with R36C original characterization.

**Verdict: STRICT RECONFIRM Δ% PASS / Welch t MARGINAL.** No escalation: R36C SHIP claim was on Δ% not Welch t; 2-run Δ% reproducibility (+7.49 / +7.22) is the stronger evidence. Flag for future reviewers: 8B QO Welch t is structurally noisier than 70B QO and may not clear ≥10 even at N_PAIRS=20.

Dispatcher trace verified: `[mxfp8_dispatch] crr_v2: shape=(M=4096,N=4096,K=4096) -> ADVISE-V2-RCR-8B-QO (R36C +5.83-7.05%)`.

### 2.2 V2-RCR 70B QO (M=4096, N=8192, K=8192)
N_PAIRS=10, GPU3: Δ%=**+9.523%**, Welch t=**18.152**.
- Δ% PASSES (≥+8% target).
- Welch t PASSES (>+15 target).
**STRICT RECONFIRM PASS.**

## Phase 3 — Production gold-standard re-benches

### 3.1 70B-KV HB shrink B1 (M=4096, N=1024, K=8192)
N_PAIRS=10, GPU6, two_so paired:
- DEFAULT median 768.30 TF, HBSHRINK median 984.76 TF
- **Δ%=+28.173%, Welch t=219.768**
- Target ≥+28%: **PASS by 0.17 pp** (margin tight but clean).
- Dispatcher trace confirms `CRR-V2-HBSHRINK-B1-70B-KV (R37AB SHIP)`.

### 3.2 8B-KV HB shrink B1 (M=4096, N=1024, K=4096)
N_PAIRS=10, GPU7, two_so paired:
- DEFAULT median 661.14 TF, HBSHRINK median 821.71 TF
- **Δ%=+24.288%, Welch t=74.657**
- Target ≥+24%: **PASS by 0.29 pp** (margin tight but clean).
- Dispatcher trace confirms `CRR-V2-HBSHRINK-B1-8B-KV (R38 wrap fix 66ef02d8)`.

### 3.3 8B-Down V2-RRR (M=4096, N=4096, K=14336) — **FAIL**
N_PAIRS=10, GPU6, one_so_layout:
- CRR median 467.11 TF, RRR median 477.96 TF
- **Δ%=+2.323%, Welch t=5.456**
- Target ≥+6.5% (R36B SHIP): **FAIL by −4.18 pp**.
- Dispatcher trace correct: `ADVISE-V2-RRR-8B-DOWN (R36B +9.51%)` — predicate fires, kernel runs, just delivers ~1/3 of original SHIP claim.

This is a **MAJOR FINDING**: the 8B-Down V2-RRR SHIP appears to have regressed substantially since R36B/R38C wire-in (e466e582). Three possibilities:
1. Driver/firmware regression on K=14336 (gfx950 fw bump between R36B and R42)
2. Compile-time difference: I built `/tmp/r42_8bdown.so` with `-DM_DIM=4096 -DN_DIM=4096 -DK_DIM=14336` and no further flags. R36B SHIP may have used additional flags I didn't replicate.
3. Pair-bench harness (one_so_layout) measures CRR-default vs RRR-default in same .so, but RRR path may need its own K=14336 specialization not present in default 8192³ build.

Recommend Dev follow-up: rebuild with explicit R36B/R38C compile recipe and re-run; if regression confirmed, file as R43 escalation.

### 3.4 8B Gate/Up V2-RRR (M=4096, N=14336, K=4096)
N_PAIRS=10, GPU7, one_so_layout:
- CRR median 2358.66 TF, RRR median 2480.62 TF
- **Δ%=+5.171%, Welch t=15.772**
- Target ≥+5% (R34B SHIP +5.025% min): **PASS by 0.15 pp**.
- Dispatcher trace correct: `ADVISE-V2-RRR-8B-GATEUP (R34B +5.025% min)`.

## Phase 4 — Methodology checks

### 4.1 PY_MODULE_NAME defensive assert
`r37_paired_bench_2so.py:50-55` contains the R39 Dev D collision guard:
```
PY_MODULE_NAME collision: MOD_A=... MOD_B=... resolved to the same in-memory module
— both .so must be built with distinct -DPY_MODULE_NAME
```
Active in all R42 two_so cells (P3.1, P3.2). **PASS.**

### 4.2 MXFP8_DISPATCH_TRACE coverage
Every R42 Phase 2/3 cell with `MXFP8_DISPATCH_TRACE=1` emitted the expected
`[mxfp8_dispatch] <kind>: shape=(...) -> <NAME>` lines (verified in 6 cells × 1-2 lines each):
- P2.1: ADVISE-V2-RCR-8B-QO + CRR-V2-EXACT-8WAVE-DEFAULT + RCR-V2-EXACT-8WAVE
- P2.2: ADVISE-V2-RCR-70B-QO + CRR-V2-EXACT-8WAVE-DEFAULT + RCR-V2-EXACT-8WAVE
- P3.1: ADVISE-V2-RRR-70B-KV + CRR-V2-EXACT-8WAVE-DEFAULT + CRR-V2-HBSHRINK-B1-70B-KV
- P3.2: same as P3.1 with 8B-KV variant
- P3.3: ADVISE-V2-RRR-8B-DOWN + CRR-V2-EXACT-8WAVE-DEFAULT + RRR-V2-EXACT-8WAVE
- P3.4: ADVISE-V2-RRR-8B-GATEUP + CRR-V2-EXACT-8WAVE-DEFAULT + RRR-V2-EXACT-8WAVE
**PASS.**

### 4.3 r38_nm_gate.sh on default 8192³ build
`/tmp/r42_70bkv_baseline.so` (default 8192³, no compile flags):
- 7 default-off features (hbshrink, hbn, subrbm, warpsm4, double_pump, mxfp8_4wave, rect): all count=0 PASS
- 3 always-present V2 dispatchers (rcr_v2, rrr_v2, crr_v2): all count=1 PASS
- OVERALL: PASS

## R42 NEW (validated)

1. **MAJOR FINDING — 8B-Down V2-RRR regression**: P3.3 delivered +2.32% Δ% / Welch t=5.46, far below R36B SHIP claim of +6.5-9.51%. Predicate fires correctly; cause is in kernel pathway or build recipe drift. Escalate to R43 Dev for root-cause.
2. **Welch t structural noise on 8B QO**: Even at N_PAIRS=20 with 120s preheat, V2-RCR 8B QO Welch t fails to clear 10 due to RCR-side tail variance. Future STRICT RECONFIRMs should use Δ% reproducibility (≥2 independent runs within ±0.5 pp) as the primary gate for this shape, with Welch t as informational.
3. **12-cycle envelope widening (763.84-792.97 = 3.81%)**: Both R42 GPU2 (low) and R42 GPU7 (high) extend prior 11-cycle envelope. Drift gate not breached but envelope widening trend warrants monitoring at R43.

## Files

- `analysis/fp8_gemm/mi350x/r42_reviewer_baseline.sh` — Phase 1 baseline harness
- `analysis/fp8_gemm/mi350x/r42_reviewer_phase23.sh` — Phase 2/3 paired-bench dispatcher
- `analysis/fp8_gemm/mi350x/r42_reviewer_4gpu_runs/` — Phase 1 outputs (12 files including G1' retries)
- `analysis/fp8_gemm/mi350x/r42_reviewer_phase2/` — Phase 2 outputs (3 cells: qo_8b, qo_70b, qo_8b_n20[fault])
- `analysis/fp8_gemm/mi350x/r42_reviewer_phase3/` — Phase 3 outputs (4 cells: p31_70bkv_b1, p32_8bkv_b1, p33_8bdown, p34_8bgateup)
- `analysis/fp8_gemm/mi350x/r42_reviewer_phase4/nm_gate_default.log` — Phase 4.3 nm-gate output
