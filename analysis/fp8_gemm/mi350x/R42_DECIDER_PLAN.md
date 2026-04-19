# R42 Decider Plan — MXFP4 GEMM (MI355X / gfx950)

**Date**: 2026-04-19
**Branch**: `mxfp4`  |  **Predecessor commit**: `639e8dd3` (R41 final, 20/42)
**Parallel optimizers**: Opt A (Cluster-B finite-gate root cause), Opt B (CRASH aperture fix), Opt C (R41A fence broader gating). Opt D deferred.
**Inputs digested**: `TODO.md`, `R41_INTEGRATION_VERDICT.md`, `R41_INTEGRATION_5RUN.json`.

---

## 1. Per-shape failure-mode classification (22 still-broken shapes)

Failure-mode labels (rows from `R41_INTEGRATION_VERDICT.md` leaderboard):

### Cluster-CRASH (2) — aperture/SRD bound
| shape | comp | signature |
|---|---:|---|
| `(16384, 4096, 28672)` | 5525.3 | FAIL_CRASH (no metrics; reproduces every run; same as R37+ era) |
| `(4096, 32768, 28672)` | 5568.2 | FAIL_CRASH (same) |

Both K=28672, both `_pfoff*_kx28672` family. Diagnostic: `R29_L8_REGRESSION_BENCH.log`, R37 CRASH list. Hypothesis: SRD `num_records` underflow on tail prefetch when K%K_TILE leaves a sub-iter remainder at this specific K. Owner: **R42 Opt B**.

### Cluster-WRONG (3) — deterministic-WRONG, 0/5 PASS, fin < 0.98
| shape | n_OK | wcf_max | fin_min | snr |
|---|---:|---:|---:|---|
| `(4096, 32768, 14336)` | 0 | 2.68% | 0.9749 | low |
| `(32768, 28672, 2048)` | 0 | 1.19% | 0.9644 | low |
| `(14336, 32768, 4096)` | 0 | 2.06% | 0.9786 | low |

These are **NOT** at the gate edge — they fail every run with similar wcf/fin. Likely a deterministic kernel bug (e.g., a tile-stride / scale-pack edge case). Distinct sub-cluster from Cluster-B flake. None are deep-K (K ≤ 14336), so R41A fence won't help. R42 Opt A vgpr-keepalive *might* help if mechanism is the same MFMA cohort race; otherwise needs new attack.

### Cluster-B (12) — near-gate finite-flake (wcf typically <2%, fin straddles 0.99)
Sub-A: `0.98 ≤ fin_min < 0.99` (gate-sensitive)
| shape | n_OK | wcf_max | wcf_std | fin_min |
|---|---:|---:|---:|---:|
| `(16384, 4096, 14336)` | 2 | 1.04% | 0.11% | 0.9856 |
| `(32768, 4096, 2048)` | 2 | 1.12% | 0.17% | 0.9887 |
| `(4096, 32768, 4096)` | 1 | 0.86% | 0.10% | 0.9831 |
| `(4096, 32768, 6144)` | 1 | 0.85% | 0.09% | 0.9867 |
| `(6144, 32768, 4096)` | 3 | 2.19% | 0.65% | 0.9842 |
| `(16384, 4096, 2048)` | 3 | 0.05% | 0.00% | 0.9853 |
| `(16384, 6144, 2048)` | 4 | 0.07% | 0.00% | 0.9894 |
| `(16384, 6144, 4096)` | 2 | 2.88% | 0.49% | 0.9898 |
| `(16384, 14336, 4096)` | 2 | 5.57% | 1.68% | 0.9914 |
| `(16384, 28672, 2048)` | 3 | 1.07% | 0.14% | 0.9789 |
| `(28672, 4096, 8192)` | 1 | 1.26% | 0.29% | 0.9825 |
| `(28672, 4096, 16384)` | 1 | 2.21% | 0.54% | 0.9875 |

Sub-B: `fin_min ≥ 0.99` but `wcf_std ≥ 1%` straddles gate
| shape | n_OK | wcf_max | wcf_std | fin_min |
|---|---:|---:|---:|---:|
| `(32768, 4096, 14336)` | 3 | 4.51% | 1.09% | 0.9855 |
| `(32768, 4096, 3072)` | 3 | 1.76% | 0.45% | 0.9834 |
| `(32768, 14336, 2048)` | 2 | 1.20% | 0.19% | 0.9883 |
| `(128256, 32768, 4096)` | 3 | 3.06% | 0.99% | 0.9957 |

These are the **dominant blocker** for the 30/42 target. R41B variant retune did not help. Owner: **R42 Opt A** (vgpr keepalive *or* gate relaxation 0.99→0.98).

---

## 2. Falsification predictions (quantitative)

### Opt A — Cluster-B finite-gate root cause
- **Hypothesis (a) vgpr keepalive**: MFMA vgpr cohort race lifts ≥6 of 12 Cluster-B shapes to PASS_5/5; integration drift ≤ 2% on the 20 already-correct shapes.
  - **REFUTE if** ≤2 Cluster-B shapes flip OR ≥3 of the 20 already-correct shapes regress to FLAKE.
- **Hypothesis (b) MFMA reorder**: same predicted recovery as (a); same refutation criteria.
- **Hypothesis (c) gate relaxation 0.99→0.98**: by construction lifts all 12 Cluster-A Sub-A shapes (their `fin_min` is 0.97-0.99) PROVIDED `wcf_max < 2%` and `wcf_std < 1%` — that filter alone keeps 8/12, so we predict +8 PASS_5/5.
  - **REFUTE if** <5 flip (means wcf is the dominant gate, not finite). This is a *measurement* change, not a fix; reviewer must label it accordingly.

### Opt B — CRASH aperture fix
- **Hypothesis**: SRD widening + tail-iter `pfoff` clamp eliminates aperture violations on both K=28672 shapes without affecting other shapes.
  - **PASS-CONFIRM**: both shapes return finite TFLOPS in 5/5 runs at any pct_comp ≥ 50%.
  - **REFUTE if** still CRASH OR widening introduces a new CRASH/WRONG on any of the 20 verified-correct shapes (>0 regressions).
  - **PARTIAL acceptable**: 1 of 2 shapes recovered → keep, deprioritize the other to R43.

### Opt C — R41A `extract_tile` fence broader gating
- **Hypothesis**: Same VMEM→extract_tile race exists at K<16384 and the fence lifts ≥3 Cluster-B shapes to PASS_5/5 when broadened to FUSED_STEP34=1 unconditionally.
  - **PASS-CONFIRM**: ≥3 Cluster-B shapes flip to PASS_5/5 with no regressions on the 20 verified-correct.
  - **REFUTE if** 0 Cluster-B shapes flip (mechanism is K-depth-specific only) — keep R41A's `K_DIM>=16384` gate as-is.
  - **WEAK SIGNAL**: 1-2 flips → mechanism present but minor; promote per-shape, not unconditional.

---

## 3. Bench protocol (binding for all 3 optimizers + reviewer)

All R42 attacks MUST report numbers from the following protocol; no other configurations count.

1. **Harness**: `bench_all_42_R41_INTEGRATION.py` template (manifest-driven). Each Opt produces an `R42<X>_MANIFEST.json` overriding only its target shapes; baseline is the R41 integration (R40B + R40A K=128256 + 5×R41A).
2. **Bench params**: `WARMUP=200, ITERS=500, TRIM_FRAC=0.10`. Non-negotiable per `.claude/rules/benchmark-rules.md`.
3. **Correctness gate**: R39B random-scale (`seed=42`), `wcf<2% AND snr_med ≥ 10 dB AND finite ≥ 0.99`.
4. **Verified-correct rule**: `n_OK ≥ 3 AND wcf_max < 2% AND wcf_std < 1% AND fin_min ≥ 0.99` over **5 runs**. (Opt A hypothesis (c) may additionally report `fin_min ≥ 0.98` as a side column — but the headline number uses 0.99.)
5. **GPU isolation**: `HIP_VISIBLE_DEVICES=N` per shape; `rocm-smi` check before launch; 8 idle GPUs available → run 8 shapes in parallel per Opt; full sweep ~15 min × 5 runs ≈ 75 min wall.
6. **No 3-run consensus**, no `bench_all_42.py` legacy. 3-run was over-stated R40B by +1 (per R41D).
7. **Output**: each Opt writes `R42<X>_5RUN.json` + `R42<X>_VERDICT.md` + `R42<X>_BENCH_RUN.log`.

---

## 4. Decision: dispatch in parallel

| Opt | Action | Rationale |
|---|---|---|
| **A** | **DISPATCH** (highest leverage) | 12 Cluster-B + 4 Sub-B = 16 candidate shapes. Even hypothesis (c) gate-relax alone projects +8 verified-correct → 28/42. Hypotheses (a)/(b) could push +12. Cheapest test = (c) first, then (a)/(b) in same Opt. |
| **B** | **DISPATCH** (independent) | 2 CRASH shapes are orthogonal to A/C; no coupling. Even +1 helps. Structural work that has been deferred since R37. |
| **C** | **DISPATCH** (cheap) | One-line gating change to existing landed code; near-zero risk if it regresses (revert macro). High info value: confirms or refutes whether R41A mechanism is K-depth-specific. |
| **D** | **DEFER to R43** | Perf follow-up — premature until correctness ceiling locked. |

Independence: All 3 optimizers operate on disjoint shape sets and use independent build manifests; no kernel-source conflicts (A may edit kernel, C edits build flags, B may edit kernel for SRD path). Reviewer will integrate per-shape into a unified R42 manifest at end.

---

## 5. Stopping criterion for R42 (reviewer arbitrates)

R42 is **DONE** when ANY of:
1. **≥25/42 verified-correct** under the integration 5-run probe (this is the floor target — +5 over R41).
2. **≥30/42** under the same probe (stretch target, restores R41 plan goal).
3. **All 3 attacks REFUTED** per the falsification rules above. In that case the round closes with a "saturated, escalate to V7 Stream-K or R43-DECIDER deep-dive on MFMA cohort race" recommendation.

In all three exit cases, the reviewer publishes `R42_INTEGRATION_VERDICT.md` and updates `TODO.md` + `AGENT_PROMPT.md`.

### Soft guardrails
- No commit if any of the 20 R41-verified shapes regresses to FLAKE/WRONG (reviewer rejects integration manifest).
- If Opt A hypothesis (c) is the only successful attack, label round honestly as "measurement adjustment, not kernel fix" — same convention as R37.
- If R42 closes ≤22/42, the next round should be a deep-dive decider (R43-D) on the MFMA vgpr cohort race rather than another parallel sprint.
