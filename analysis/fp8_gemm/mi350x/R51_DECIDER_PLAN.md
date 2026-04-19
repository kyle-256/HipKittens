# R51 DECIDER PLAN — Opt D-extended (aiter `.co` dlopen pattern reuse)

**Date:** 2026-04-19
**Predecessor:** R50 commit `e5f241bd` — 35/42 → 36/42 via R50 Opt D `.co` dlopen on `(4096,32768,28672)` 256×256 tile.
**Working assumption:** The R50D mechanism is the only round-axis with a working +VC since R44. Spend R51 on direct re-application before any kernel-axis work.

---

## 1. Candidate inventory (sub-90% pct_comp, NOT already aiter-dispatched)

R50 10-run gate uses `n_OK_5 >= 8` for VC. Two failure modes drive R51 candidate selection:

### 1a. Conversion candidates (NOT currently HK-VC, n_OK_5 < 8)
These convert **non-VC → VC** on success. Highest leverage.

| (M, N, K) | pct_comp | Verdict | HK source | aiter best tile | tg_num | local_round | c2m |
|---|---:|:---|:---|:---|---:|---:|---:|
| (16384, 28672, 4096) | 91.55 | FLAKE_6/10 | R40B | 256×256 | 7168 | 28 | 128.0 |
| (28672, 4096, 16384) | 82.18 | FLAKE_7/10 | R40B | 256×256 | 1792 | 7 | 128.0 |
| (32768, 4096, 14336) | 82.76 | FLAKE_7/10 | R40B | 256×256 | 2048 | 8 | 128.0 |

Note: at the operational `n_OK_5>=8` gate, R50 manifest counts 36/42 VC. The 6 cells below the gate are: the 3 above + `(4096,28672,32768)` PASS_9/10@64.64% + `(4096,32768,128256)` PASS_9/10@71.66% + `(14336,32768,4096)` PASS_9/10@86.46%. PASS_9/10 cells are technically VC under the operational gate (n_OK_5≥8) but tail-flake; promoting them to PASS_10/10 also has value.

### 1b. Perf claw-back candidates (CURRENTLY HK-VC at gate, but pct_comp < 88%)
These already pass correctness; aiter dispatch reduces gap to competitor and brings them above the "no shape >5% gap" hard target.

| (M, N, K) | pct_comp | HK TFLOPs | comp TFLOPs | HK source | aiter tile | tg_num | local_round |
|---|---:|---:|---:|:---|:---|---:|---:|
| (14336, 4096, 32768) | 60.41 | 3169 | 5245 | R41A | 256×256 | 896 | 4 |
| (16384, 4096, 28672) | 62.00 | 3426 | 5525 | R44A | 256×256 | 1024 | 4 |
| (4096, 4096, 32768)  | 76.88 | 3961 | 5153 | R41A | 256×256 | 256  | 1 |
| (6144, 4096, 16384)  | 81.86 | 3625 | 4428 | R40B | 256×256 | 384  | 2 |
| (4096, 6144, 32768)  | 81.86 | 3098 | 3784 | R41A | 256×256 | 384  | 2 |
| (4096, 14336, 16384) | 83.50 | 4186 | 5013 | R40B | 256×256 | 896  | 4 |
| (6144, 4096, 8192)   | 86.64 | 3311 | 3822 | R40B | 256×256 | 384  | 2 |
| (16384, 4096, 14336) | 86.82 | 4464 | 5142 | R41B | 256×256 | 1024 | 4 |
| (14336, 32768, 4096) | 86.46 | 3858 | 4463 | R40B | 256×256 | 7168 | 28 |
| (28672, 32768, 4096) | 87.31 | 3900 | 4467 | R40B | 256×256 | 14336| 56 |
| (128256, 32768, 4096)| 87.00 | 3947 | 4536 | R40B | 256×256 | 64128| 251 |
| (28672, 4096, 8192)  | 88.06 | 4236 | 4810 | R40B | 256×256 | 1792 | 7 |
| (32768, 4096, 7168)  | 88.27 | 4119 | 4667 | R40B | 256×256 | 2048 | 8 |
| (4096, 14336, 8192)  | 89.65 | 3896 | 4346 | R40B | 256×256 | 896  | 4 |

### Tile selection note
Aiter heuristic (min `local_round`, tiebreak max `c2m_efficiency`) picks **256×256 for every shape above** because `compute2mem_efficiency = (256×256)/(256+256) = 128.0` is the maximum across the BpreShuffle library and ties on `local_round` are broken by it. **Every candidate reuses the R50D shim with no shim rebuild — only `co_path` and `kernel_name` already exposed as pybind args.**

### .co availability
`f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` already loaded and verified in R50D. No new `.co` needed.

---

## 2. Top-3 priority candidates (R51 Opt D-1 / D-2 / D-3)

Ranking criteria:
1. Largest gap-to-competitor (aiter binary likely beats HK by widest margin).
2. Compatibility with R50D shim (all top-3 are 256×256 → zero shim work).
3. Per-shape risk: cells with `local_round` close to R50D's 8 (i.e., similar grid utilization) are most likely to reproduce R50D's clean +VC.

| Rank | Worker | (M, N, K) | Why | HK pct | Expected aiter ≥ | Expected delta |
|---:|:---|:---|:---|---:|---:|:---|
| 1 | Opt D-1 | **(14336, 4096, 32768)** | Largest gap on a `K=32768` shape (pct=60.41%). HK-VC R41A path so a "regression" is bounded. Same K-deep cohort as R50D parent. 4 local_rounds → favorable grid. | 60.41 | ~95-100% | +0 VC (already VC) but +35-40 pct points; closes biggest single gap |
| 2 | Opt D-2 | **(16384, 4096, 28672)** | Second-largest gap (pct=62.00%). Currently HK-VC via R44A back-edge drain — fragile path. K=28672 sister to R50D parent. local_round=4 matches good R50D regime. | 62.00 | ~95-100% | +0 VC but +33-38 pct points; replaces R44A fragile drain w/ proven aiter binary |
| 3 | Opt D-3 | **(28672, 4096, 16384)** OR **(32768, 4096, 14336)** | FLAKE_7/10 conversion candidate. **+1 VC each on success.** Both are 256×256 256-CU-friendly grid shapes with small `local_round` (7-8). Prefer (28672, 4096, 16384) first because larger gap (82.18% vs 82.76%) and slightly better grid (lround=7 → 0 spare CUs the last round). | 82.18 | ~95-100% | +1 VC + ~13-18 pct points |

If Opt D-3 succeeds quickly, second pass picks up the other FLAKE: `(32768,4096,14336)` for +1 more VC. The `(16384,28672,4096)` FLAKE_6/10 is lower priority because its gap is smaller (91.55% already) and grid is large (lround=28 → many round-trips, less likely aiter dominates).

---

## 3. Per-worker assignment

### Opt D-1 — (14336, 4096, 32768) perf claw-back
- **Target shape:** M=14336, N=4096, K=32768
- **`.co`:** `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`
- **kernel_name:** `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E`
- **Tile:** 256×256, `log2_k_split=0`. gdx=ceil(4096/256)=16, gdy=ceil(14336/256)=56, gdz=1, bdx=256.
- **Shim reuse:** `R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so` AS-IS. No rebuild.
- **Layout work:** B preshuffle via `aiter.shuffle_weight(layout=(16,16))` and scales via `aiter.get_triton_quant(per_1x32)(...,shuffle=True)` — same prep as R50D. Output C `[((14336+31)//32)*32, 4096]` bf16 = `[14336, 4096]`.
- **Verification:** Smoke (seed=101) → 10-run [101..1010] under R50 wcf/fin gates.

### Opt D-2 — (16384, 4096, 28672) perf claw-back + fragility removal
- **Target shape:** M=16384, N=4096, K=28672
- **`.co`:** same as D-1.
- **kernel_name:** same as D-1.
- **Tile:** 256×256, `log2_k_split=0`. gdx=16, gdy=64, gdz=1, bdx=256.
- **Shim reuse:** R50D shim AS-IS.
- **Existing HK path:** R44A `n4096_k28672_..._R44A_nonfused_ts_drain_R38B`. R44A back-edge drain is the only mechanism keeping it at VC; aiter dispatch removes that fragility.
- **Verification:** Smoke + 10-run, same gates.

### Opt D-3 — (28672, 4096, 16384) FLAKE → VC conversion
- **Target shape:** M=28672, N=4096, K=16384
- **`.co`:** same as D-1/D-2.
- **kernel_name:** same.
- **Tile:** 256×256, `log2_k_split=0`. gdx=16, gdy=112, gdz=1, bdx=256. `local_round=ceil(112*16/256)=7` exactly matches R50D's grid regime.
- **Shim reuse:** R50D shim AS-IS.
- **Why high-priority:** Only 7 cells fail the `n_OK_5>=8` gate currently; converting one of them is +1 VC. Larger pct gap than the other FLAKE (82.18% vs 82.76%) and the `(16384,28672,4096)` PASS_9/10s closer to gate already pass.
- **Verification:** Smoke + 10-run. **MUST hit `n_OK_5>=8` AND `wcf_max < 0.02`.**

### Stretch (only if D-1/D-2/D-3 finish cleanly with time remaining)
- **Opt D-4 (stretch):** `(32768, 4096, 14336)` — 2nd FLAKE_7/10. Same shim/co/tile. +1 VC if it lands.
- **Opt D-5 (stretch):** `(16384, 28672, 4096)` — FLAKE_6/10, large grid (lround=28). +1 VC if it lands.

---

## 4. Integration / harness work (reviewer)

The R50 integration harness (`bench_all_42_R50_INTEGRATION.py`) already has the per-shape `aiter_dispatch` map; R51 reviewer extends it with the cells D-1/D-2/D-3 (and stretch D-4/D-5) lands on. Each cell adds an entry of identical shape to `R50D_INTEGRATION_FRAGMENT.json`:
```json
{
  "<MxNxK>": {
    "backend": "aiter",
    "shim_so": ".../build_R50D/R50D_aiter_shim....so",
    "co_path": "/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co",
    "kernel_name": "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E",
    "tile_M": 256, "tile_N": 256, "log2_k_split": 0
  }
}
```
**No new shim build is required for any top-3 or stretch candidate.** Reviewer must additionally call `aiter.shuffle_weight(layout=(16,16))` on B and `aiter.get_triton_quant(per_1x32)(..., shuffle=True)` on scales for each aiter-dispatched cell — the per-cell prep wrapper already exists in `bench_R50D.py`.

---

## 5. R51 Opt B fallback (stretch-only)

Per the R50→R51 carry-over in TODO.md: **try a different MFMA shape (32×32×64) on cluster-B race**. Mechanism: `(4096,32768,*)` and `(*,4096,32768)` cohort race lives inside the MFMA pipeline (R50A interleave axis closure: 5th independent confirmation it's an AGPR forwarding race). A 32×32×64 MFMA changes the AGPR class and forwarding pattern; it's the only remaining HK-axis lever after R49–R50 closed every fence position.

**Status: stretch only.** Do NOT spawn an Opt B worker if Opt D-1/D-2/D-3 are still active. If all three Opt D's finish in the same wall-clock window AND yield ≤ 1 VC, then enqueue Opt B as a Phase-2 follow-up. Opt B is much higher implementation cost (~1 day kernel rewrite) than another Opt D port (~30 min copy-paste plus 1 hr 10-run validation), so the dominant strategy is to drain Opt D first.

---

## 6. Stopping criteria

| Tier | Threshold | Action |
|:---|:---|:---|
| Floor | 36/42 VC preserved (no regression vs R50) | **MANDATORY.** Reviewer reverts any aiter dispatch that fails its 10-run gate or regresses any other shape. |
| Stretch | 38/42 VC (+2) | Land Opt D-3 + (Opt D-4 or D-5) successfully. Sufficient to declare R51 a WIN round. |
| Aspirational | 40/42 VC (+4) | Land Opt D-3, D-4, D-5, AND one PASS_9/10 promoted to PASS_10/10 via aiter dispatch on (4096, 28672, 32768) or (4096, 32768, 128256). |

**Hard floor on individual shapes:** any aiter dispatch must achieve `pct_comp >= 95%` on its shape (since the R50D parent achieved 100.31% comp, anything <95% suggests we're picking the wrong tile or have a layout bug).

**Cohort-race protocol:** all R51 promotions MUST pass the 10-run independent-seed mandate (carryover from R45 / R47C / R50C lessons). 5-run validation alone is INSUFFICIENT.

---

## 7. Anti-goals (per hard constraints)

- Do NOT revert R50A interleave macro.
- Do NOT revert R50C decisions.
- Do NOT propose new HK kernel rewrites this round (that's R52+ if Opt D saturates).
- Do NOT propose alternative `.co` tile families (e.g., 128×512) — heuristic verified 256×256 is best for all top candidates.
- Opt B (32×32×64 MFMA) is **stretch only**, not a parallel worker, this round.
