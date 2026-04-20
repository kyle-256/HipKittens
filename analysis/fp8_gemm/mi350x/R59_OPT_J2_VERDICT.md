# R59 Cohort J-2 (Opt S) Verdict — L3 HK Survivor Rescue Alt-Tile Probe

**Date:** 2026-04-20
**Worker:** B
**Cell:** L3 `(32768, 14336, 2048)` — current HK R40B baseline 100.49% (VC-flipped at R58 ITERS=500: n_OK=9/10, wcf_max=0.0111, fin_min=0.911)
**GPU(s) used:** GPU 2 only (GPU 1 was 100% busy with 2 unrelated PIDs throughout the task; sequential execution on GPU 2)
**Total wall clock:** ~95 s (44.8 s S-1 + 42.3 s S-2 + 8 s setup/JSON emission)
**Shim status:** `build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so` AS-IS (11th consecutive R50D AS-IS reuse, no rebuild)

---

## 1. Per-candidate SMOKE result

| ID | Tile | aiter_eff | Grid (gdx, gdy) | n_OK | fin_min | wcf_max | snr_med | tflops | pct_comp | delta vs HK |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **J-2/S-1** | 96 x 640 | 83.5 | (342, 23) | 5/5 | 1.0 | 0.0 | 55.54 | 2117.6 | **63.19%** | **-37.30pp** |
| **J-2/S-2** | 64 x 1024 | 60.2 | (512, 14) | 5/5 | 1.0 | 0.0 | 55.54 | 3074.4 | **91.73%** | **-8.76pp** |

Both candidates are **bit-deterministic** (wcf=0, fin=1.0 across all 5 seeds) — correctness is clean, perf is the gating issue.

---

## 2. SMOKE escalation decision (per Decider Plan §5 / §10)

Gate thresholds for L3 (HK baseline 100.49%):
- **ESCALATE → 10-run** iff SMOKE pct_comp > **101.49%** (HK + 1.0pp)
- **ACCEPT_FALLBACK** iff SMOKE pct_comp ∈ [**95.49%**, 101.49%]
- **STOP_DEAD** iff SMOKE pct_comp < **95.49%** (HK - 5.0pp)

| ID | SMOKE pct_comp | Decision |
|---|---:|---|
| **J-2/S-1** | 63.19% | **STOP_DEAD** (catastrophic, < 95.49% floor by -32.30pp) |
| **J-2/S-2** | 91.73% | **STOP_DEAD** (< 95.49% floor by -3.76pp) |

No 10-run escalation triggered for either candidate.

---

## 3. Final verdict per candidate

| ID | Verdict | Manifest action |
|---|---|---|
| **J-2/S-1** | **DEAD** | ACCEPT_FALLBACK — keep R40B HK manifest entry verbatim |
| **J-2/S-2** | **DEAD** | ACCEPT_FALLBACK — keep R40B HK manifest entry verbatim |

**Cohort outcome:** 0 PROMOTE / 2 SMOKE_DEAD. L3 manifest entry **unchanged** at R40B HK 100.49% (VC-flipped). L3 alt-tile axis closed.

---

## 4. Mechanism observations

- The mechanism hypothesis from §3 of the decider plan ("wider-N tiles reduce grid_y, may help XCD load profile") was **falsified** for L3 `(32768,14336,2048)`. While both wider-N tiles do reduce grid_y substantially (HK 56 → 23 for S-1, → 14 for S-2), grid_x increases proportionally (HK 128 → 342 for S-1, → 512 for S-2), and the per-tile compute efficiency drops sharply (eff 100 → 83.5 → 60.2). The grid_x explosion + eff drop dominates any grid_y benefit on this M=32768 / N=14336 / K=2048 shape.
- The **monotone trend** across all four AITER alt-tile attempts on this cell is informative:
  | Tile | eff | pct_comp | source |
  |---|---:|---:|---|
  | 256 x 256 | 100 | sub-HK | R55 D-5B/1 |
  | 192 x 256 | ~92 | -14.38pp | R57 L-3 |
  | 128 x 256 | 85.3 | -25.04pp | R58 P-3 |
  | 96 x 640 | 83.5 | **-37.30pp** | **R59 S-1** |
  | 64 x 1024 | 60.2 | **-8.76pp** | **R59 S-2** |

  S-2 64x1024 is the **strongest** AITER candidate after the original 256x256 — its 4x reduction in grid_y partially compensates for the 4x grid_x explosion and 40% eff drop. But it still cannot match HK 256x256 on this cell.
- HK 256x256 R40B remains the unique perf-leader; the cohort-race tail-draw at fin_min=0.911 is intrinsic to ITERS=500 protocol, not curable by AITER tile-swap.

---

## 5. Axis closure status

**This round closes:**
- 96 x 640 AITER alt-tile on L3 `(32768,14336,2048)` — DEAD at -37.30pp
- 64 x 1024 AITER alt-tile on L3 `(32768,14336,2048)` — DEAD at -8.76pp

**Combined with prior closures on this cell:**
- 256 x 256 AITER (R55 D-5B/1) — DEAD (sub-HK)
- 192 x 256 AITER (R57 L-3) — DEAD at -14.38pp
- 128 x 256 AITER (R58 P-3) — DEAD at -25.04pp
- **96 x 640 AITER (R59 S-1) — DEAD at -37.30pp [NEW]**
- **64 x 1024 AITER (R59 S-2) — DEAD at -8.76pp [NEW]**

**→ AITER alt-tile axis FULLY EXHAUSTED on L3 `(32768,14336,2048)`.**

Per Decider Plan §10 STOP rules: the only remaining structural rescue path for this cell is an HK kernel rebuild with R44D FINITE_GATE relaxed 0.97 → 0.95 (deferred to R60+) or pivot to documentation acceptance (Opt U, also deferred to R60+).

---

## 6. Files emitted

- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/bench_R59J2_S1.py`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/bench_R59J2_S2.py`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R59_OPT_S1_SMOKE.json`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R59_OPT_S1_SMOKE.log`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R59_OPT_S2_SMOKE.json`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R59_OPT_S2_SMOKE.log`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R59J2_S1_INTEGRATION_FRAGMENT.json`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R59J2_S2_INTEGRATION_FRAGMENT.json`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R59_OPT_J2_VERDICT.md` (this file)

**No manifest changes** for L3 — R58 entry preserved verbatim by ACCEPT_FALLBACK.

---

## 7. GPU usage notes (for reviewer / decider)

- GPU 1 reported 100% utilization with 2 unrelated PIDs (3714099 + 3723021) at task start and after the spec-mandated 30s wait/retry; never freed during this cohort. GPU 2 was idle throughout.
- Ran S-1 then S-2 sequentially on GPU 2 (cumulative ~95 s) instead of the planned parallel pair on (GPU 1, GPU 2). Wall time still well within the 5 GPU-min/cell hard cap (44.8 s + 42.3 s).
- GPU 0 and GPU 3 left untouched per Decider Plan §4 ("DO NOT use GPUs 0 or 3").
- No interaction with reviewer GPUs (4, 5, 6, 7) or worker C cohort GPUs.
