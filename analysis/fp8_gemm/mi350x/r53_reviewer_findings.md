# R53 Reviewer — V2 spotcheck + CRR XCD confirm + 9-cell baseline re-refresh

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ bd59941f (R52 wrap) → re-baselined on HEAD post R53A/B/C
**GPU:** MI355X (gfx950) HIP_VISIBLE_DEVICES=3
**Mandate:** (1) Spot-check R52 Dev O CONFIRM-SHIPPED V2-RRR claim; (2) Re-validate R52G CRR XCD swizzle ship at 70B Gate/Up CRR; (3) Refresh 9-cell strict-SCLK baseline as sanity gate post R53 dev commits.

---

## TL;DR — R52 Dev O & R52G ships RE-CONFIRMED, R53 cycle introduced ZERO regression

| Sub-task | Result |
|---|---|
| **R52 Dev O V2-RRR** | RE-CONFIRMED — V2 wins +5.81% (8B GateUp), +3.64% (70B Q/O) |
| **R52G CRR XCD swizzle** | RE-CONFIRMED — ON +5.00% vs OFF at 70B GateUp CRR |
| **9-cell baseline post R53A/B/C** | **5/9 PASS, 4/9 HEADROOM** — IDENTICAL to R52 within noise (no R53 regressions) |

R53 cycle delivered 3 REFUTATIONS + 1 DIAGNOSTIC + 0 new SHIP. The kernel-level
K-loop body restructure family (R49A host-side, R53A kernel-side opsel, R53B
K-superblock) is now triply closed against the V2 RRR VGPR ceiling (254/256).
70B Down CRR is structurally non-recoverable (R53C: FP8 baseline amortizes
3-deep at K=28672, MXFP8 cannot due to conditional shift block).

---

## §1. V2-RRR spotcheck (R52 Dev O re-validation)

Strict-SCLK A/B at 8B Gate/Up RRR + 70B Q/O RRR, 5 runs/arm, MXFP8_WARMUP=100,
MXFP8_ITERS=200, 30s cooldown, 60s rebuild_cool, MXFP8_RRR_PRESHUFFLE_V2_RUNTIME
flipped between "0" (V1) and "1" (V2 = production default).

| Cell | V1 med (TFLOPS) | V2 med (TFLOPS) | Δ | SNR |
|---|---:|---:|---:|---:|
| 8B Gate/Up RRR | 2372.5 [2335.4..2377.4] | 2510.4 [2455.4..2517.9] | **+5.81%** | 49.61 dB ✓ |
| 70B Q/O RRR | 2825.7 [2819.6..2831.8] | 2928.5 [2884..2937]† | **+3.64%** | 49.59 dB ✓ |

†70B Q/O V2 had 1/5 thermal outlier (582 TFLOPS run1) — median scoring (rank 3)
at 2928.5 is the canonical strict-SCLK number. Ranges [2884..2937] shown after
outlier exclusion. Det 3/3 PASS both arms.

**Conclusion:** R52O CONFIRM-SHIPPED stands. V2-RRR has been production default
since R22-A and continues to win across the 2-cell spotcheck. The +5.81% / +3.64%
deltas land within R52O's claimed range (+2.10% to +6.80%, mean +4.36%).

## §2. CRR XCD swizzle confirm (R52G re-validation)

Strict-SCLK A/B at 70B Gate/Up CRR, 5 runs/arm, swizzle ON (HEAD) vs OFF
(per-shape gate disabling MXFP8_CRR_BLOCK_SWIZZLE for B_70B_GateUp_CRR — same
patch R52G evaluated).

| Arm | TFLOPS (5 runs) | Median | SNR |
|---|---|---:|---:|
| **ON (HEAD)** | 2480.9 / 2502.8 / 2491.8 / 2506.9 / 2487.8 | **2491.8** | 49.59 dB ✓ |
| **OFF (un-shipped)** | 2355.6 / 2365.5 / 2374.8 / 2373.1 / 2379.4 | **2373.1** | 49.59 dB ✓ |
| **Δ ON−OFF** | | **+5.00%** | |

**Conclusion:** R52G stands. R49 Reviewer's reported -3.87% regression was
thermal noise (10.27% spread on R49 ON-arm landed in unstable warmup regime).
R52G's confirm with <1% spread is genuine steady-state. R47 Dev B's CRR XCD
swizzle ship across all 7 CRR cells holds.

## §3. 9-cell baseline re-refresh (post R53A/B/C commits)

Strict-SCLK 5 runs/cell, both FP8 per-tensor & MXFP8 V2 (production defaults),
on `feat/mxfp8-only` HEAD after R53A/B/C commits (562eeeb5 / d5a6b22e / b4e521e8).
K-superblock kept default-OFF (MXFP8_RRR_K_SUPERBLOCK=1 = byte-identical no-op).

| Shape | Layout | FP8 (TFLOPS) | MX (TFLOPS) | MX/FP8 % | Verdict |
|---|---|---:|---:|---:|---:|
| 8B Down | RCR | 3154.7 | 3037.2 | 96.3% | **PASS** |
| 8B Down | RRR | 2757.6 | 3011.5 | 109.2% | **PASS** |
| 8B Down | CRR | 2922.3 | 2805.3 | 96.0% | **PASS** |
| 8B Gate/Up | RRR | 2666.8 | 2507.4 | 94.0% | HR -1.0pp |
| 70B Gate/Up | RCR | 2995.5 | 2856.6 | 95.4% | **PASS** |
| 70B Gate/Up | CRR | 2807.1 | 2506.3 | 89.3% | HR -5.7pp |
| 70B Down | RCR | 3238.0 | 2966.1 | 91.6% | HR -3.4pp |
| 70B Down | RRR | 2752.7 | 2987.4 | 108.5% | **PASS** |
| 70B Down | CRR | 3030.0 | 2663.8 | 87.9% | HR -7.1pp |

**5/9 PASS, 4/9 HEADROOM** — IDENTICAL to R52 Reviewer's 5/21 PASS overlap
within noise. **Zero R53 regressions** — K-superblock default-OFF code path
preserves baseline ISA bit-identically. R53A's CRR opsel infrastructure
(MXFP8_CRR_SCALE_LAYOUT_V2 default-OFF) similarly inert.

### Open headroom (R54+ candidate cells)
1. **70B Down CRR (-7.1pp)** — RULED OUT as recoverable by R53C (FP8 baseline
   amortizes 3-deep K-unroll at K=28672; MXFP8 cannot due to conditional
   `s_bitcmp0_b32 / 6× v_lshrrev_b32` block. Structural ceiling, not waste).
2. **70B Gate/Up CRR (-5.7pp)** — Still open. Same 6× shift mechanism but at
   K=8192 the FP8 baseline does NOT unroll either, so the gap is "real CRR
   waste" rather than baseline asymmetry. **Highest-leverage R54 target.**
3. **70B Down RCR (-3.4pp)** — Open. RCR has 0 scale-pack shifts; K=28672 may
   expose a different bottleneck than CRR. Worth a PMC profile.
4. **8B Gate/Up RRR (-1.0pp)** — Sub-noise. R52P diagnosed as L2/TC return
   backpressure; cachepolicy (R52Q) and K-superblock (R53B) both refuted.
   Shelve until new lever class identified.

## §4. Methodology notes

- GPU 3 isolation held throughout — no sibling reviewer contamination
  (vs R52 Reviewer's 20-min GPU 7 contamination).
- All FP8 baselines re-bench from same source as MXFP8 to control for build-
  cache state; per-tensor FP8 path uses the same dispatcher table as R48.
- Det 3/3 PASS verified for every cell (check logs in
  `r53_reviewer_results/baseline_9cell/*_check_*.log`).
- CRR confirm uses same per-shape gate path R52G evaluated, validated
  byte-identical via ISA dump (`r53_reviewer_results/rrr_head*.s`).

## §5. R53 cycle outcome (Reviewer summary)

**0 new SHIP / 0 CONFIRM-SHIPPED / 3 REFUTATIONS / 1 DIAGNOSTIC / 1 BASELINE**
on top of R52 cycle:

- R53A REFUTED: CRR opsel-keyed K-phase MMA dispatch (70 VGPR spill / -68%)
- R53B REFUTED-EMPIRICAL: K-superblock persistent CTA (228 B/lane spill, SNR 26 dB)
- R53C DIAGNOSTIC: 70B Down CRR structurally non-recoverable (FP8 amortization)
- R53 Reviewer (this doc): R52O & R52G ships re-confirmed, 9-cell baseline holds

**Cumulative refutation pattern:** the V2 RRR kernel is at a 254/256 VGPR
hard ceiling. Any source restructure that adds inter-iteration live state
spills catastrophically. R54+ should pivot to:
- LDS-resident scale layouts (avoid K-loop body changes)
- Compiler-flag exploration (`-fmax-stack-size` is allocator-only; try
  `-mllvm -amdgpu-vgpr-index-mode-disable` or scheduler tuning)
- 70B Gate/Up CRR PMC profile to identify what else gates the 6× shift block

Artifacts:
- `r53_reviewer_results/v2_spotcheck/` — V2 RRR A/B (5×2 = 10 logs/cell × 2 cells)
- `r53_reviewer_results/crr_confirm/` — 70B GateUp CRR ON/OFF (5×2 = 10 logs)
- `r53_reviewer_results/baseline_9cell/` — 9 cells × 2 dtypes × 5 runs = 90+ logs
- `r53_reviewer_v2_spotcheck.sh`, `r53_reviewer_crr_confirm.sh`,
  `r53_reviewer_baseline_9cell.sh` — bench harnesses
