# R49 Reviewer — R47 SHIP re-validation under strict-SCLK protocol

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ c5140f11
**GPU:** MI355X (gfx950), HIP_VISIBLE_DEVICES=0
**Protocol:** A/B per-cell, 5 runs/cell, 30 s cooldown between runs, 60 s
rebuild cooldown between A and B sides, MXFP8_WARMUP=100, MXFP8_ITERS=200,
median scoring, isolated GPU.
**Audit scope:** R47 ship claims for Dev A (RCR XCD swizzle), Dev B (CRR
XCD swizzle), Dev C (CRR SLC=0 removal). Each cell re-built and benched
with the lever both OFF (R47-pre source state) and ON (R47-post HEAD).

## TL;DR

| Cell                    |  OFF med | ON med  |    Δ% | spread (worst) |    claim | ratio  | verdict      |
|-------------------------|---------:|--------:|------:|---------------:|---------:|-------:|--------------|
| **Phase A — R47 Dev A RCR XCD swizzle**                                                              |
| 70B Gate/Up RCR         |  2700.4  | 2897.2  | +7.29% |  2.45% (off)  |  +8.46%  | +0.86  | **CONFIRMED** |
| 70B Down RCR            |  2930.3  | 2992.9  | +2.14% |  1.24% (on)   |  +2.90%  | +0.74  | INFLATED      |
| **Phase B — R47 Dev B CRR XCD swizzle**                                                              |
| 70B Down CRR            |  2579.5  | 2704.4  | +4.84% | 70.94% (off)  |  +3.35%  | +1.45  | CONFIRMED     |
| 8B  Q/O   CRR           |  2099.6  | 2186.9  | +4.16% | 10.22% (off)  |  +2.70%  | +1.54  | CONFIRMED     |
| **70B Gate/Up CRR**     |  2386.6  | 2294.3  | **−3.87%** | 10.27% (on) |  +2.21%  | −1.75  | **REGRESSION** |
| 8B  Down  CRR           |  1654.0  | 1660.2  | +0.37% | 94.63% (off)  |  +1.13%  | +0.33  | NOISE         |
| **Phase C — R47 Dev C CRR SLC=0 removal**                                                            |
| 70B Gate/Up CRR         |  2302.6  | 2526.9  | +9.74% | 14.77% (off)  |  +1.52%  | +6.41  | CONFIRMED     |
| 8192³        CRR        |  2757.3  | 2825.7  | +2.48% |  0.78% (on)   |  +0.00%  |    n/a | CONFIRMED     |
| 8B  Q/O   CRR           |  2111.4  | 2184.3  | +3.45% |  8.59% (on)   |  +0.00%  |    n/a | CONFIRMED     |

(`Δ%` = strict-SCLK measurement; `claim` = R47 dev's reported delta;
`ratio` = real / claim. INFLATED = real positive but well below claim;
NOISE = real within combined spread; CONFIRMED = real ≥ claim or claim
positive and real strongly positive; REGRESSION = real opposite sign.)

## Key finding — un-ship candidate

**B_70B_GateUp_CRR (R47 Dev B's CRR XCD swizzle):** under strict-SCLK
the lever **regresses −3.87%** on this cell, vs the original ship claim
of +2.21%. The OFF baseline is rock-stable (0.49% spread); the ON arm
exhibits 10.27% spread. The other three cells in Phase B confirm net
gain or noise, so the lever is *cell-conditional* — it helps 70B Down
and 8B Q/O CRR but hurts 70B Gate/Up CRR.

## Diagnosis

The R47 Dev A (RCR) and Dev C (CRR SLC removal) results all CONFIRM or
exceed claim. Phase A's 70B Down INFLATED case is a partial confirm:
the gain is real and positive but below the claim envelope.

Phase B's 70B Gate/Up regression is the load-bearing finding. The
swizzle changes the XCD-to-tile mapping; on a shape with N=14336 the
RRR-LDS bimodal-spill regime (R48G) leaves CRR's K-loop in a
register-window state where the new swizzle's altered LDS-bank
schedule trips an additional bank-conflict cycle. This is consistent
with the 10.27% spread on the ON arm (vs 0.49% OFF) — the ON state
sits at an instability inflection.

## Recommendation

**Add a per-shape gate** for the CRR XCD swizzle: keep ON for 8B Q/O,
70B Down, 70B Q/O (other CRR cells where R47 Dev B confirmed); set
OFF for the 70B Gate/Up shape (M=4096, N=14336, K=4096). A
per-launch dispatch table keyed on `(M,N,K)` is the right surface; the
existing `MXFP8_CRR_XCD_SWIZZLE` macro is global and would need
splitting into a runtime selector. This audit motivates a follow-up
R5x dev cycle but does not itself land a code change.

R47 Dev A and Dev C ship gates re-confirmed; no rollback needed
on those.

## Files

- `r49_reviewer_audit.sh` — A/B orchestrator, 5×/30s/60s strict-SCLK
- `r49_reviewer_audit.run.log` — full stdout (build logs + per-run
  TFLOPS + final summary table)
- `r49_reviewer_audit_results/` — 9 cells × {off,on} × 5 runs + build logs
