# Agent Team Runbook — MXFP8 Optimization

## 任务目标（两条线并行）

1. **长期**：MXFP8 RCR 追平 FP8 per-tensor（GPU0 今日 FP8 RCR 3253.80；MXFP8 RCR 3015.73；差 238 TFLOPS / 7.32%）
2. **静态 gate（已达标）**：MXFP8 RRR / CRR ≥ 历史 RCR 2926.61 × 95% = **2780.28 TFLOPS**
   - RRR: 2889.36 TFLOPS ✅ (+109.08)
   - CRR: 2830.67 TFLOPS ✅ (+50.39)
3. **动态 gate（参考）**：≥ 今日 RCR × 95% = 3015.73 × 0.95 = **2864.94 TFLOPS**
   - RRR: 2889.36 ✅ (+24.42)
   - CRR: 2830.67 ❌ (−34.27, 93.86% of today's RCR — 小幅 miss，与 R14 同向)

## 总则

- 分支：`feat/mxfp8-only`
- 工作目录：`analysis/fp8_gemm/mi350x`
- GPU：8× MI355X (gfx950)，**formal 验收固定 `HIP_VISIBLE_DEVICES=7`**
- Skill：
  - `.cursor/skills/mxfp8-layout-tuning/SKILL.md`
  - `.cursor/skills/fp8-per-tensor-layout-tuning/SKILL.md`
- 测试协议：`test_mxfp8_python.py` / `test_python.py` 内的 **per-iteration `torch.cuda.synchronize()` + `output.zero_()`**，`warmup=100, iters=200`

## 测试 shape 矩阵（R25 起新增）

**正方形 8192³ 仍为 primary formal**，但 SHIP candidate 还须在 LLaMA 实际 shape 上确认 correctness + 无 regression：

| 名称 | M | N | K | 用途 |
|---|---:|---:|---:|---|
| **8192³** (primary) | 8192 | 8192 | 8192 | 所有 formal A/B + SNR + det gate |
| **LLaMA-8B Gate** | 4096 | 14336 | 4096 | 非正方形 MLP shape |
| **LLaMA-8B Down** | 4096 | 4096 | 14336 | K > N 非正方形 |
| **LLaMA-70B Gate** | 4096 | 28672 | 8192 | 大 N shape |
| **LLaMA-70B Q** | 4096 | 8192 | 8192 | attention shape |
| ~~batch decode~~ | ~~128~~ | ~~8192~~ | ~~8192~~ | ~~小 M decode~~ R45 DEPRIORITIZED — memory-bound |

**编译**：dispatcher gates on compile-time `M_DIM`/`N_DIM`/`K_DIM`（`kernel_mxfp8_layouts.cpp:5-12`，默认 8192）。非正方形需 rebuild：
```bash
make CXXFLAGS_EXTRA="-DM_DIM=4096 -DN_DIM=14336 -DK_DIM=4096"
python3 test_mxfp8_python.py 4096 14336 4096
```
每 shape 需**单独 rebuild .so**。

**SHIP gate 扩展（R25 起）**：8192³ formal PASS **且** ≥2 个 LLaMA shape 全 gate PASS 才算 SHIP-ready：
1. **Correctness**: SNR ≥ 48 dB + det 3/3 PASS
2. **性能**: MXFP8 V2 TFLOPS ≥ 同 shape FP8 per-tensor × 95%（MXFP8 不能比 FP8 慢超 5%）
3. **无回归**: 同 shape MXFP8 V2 优化前后 Δ ≥ 0（不能因优化 8192³ 而在 LLaMA shape 上退化）

## 核心约束

1. 不破坏 FP8 per-tensor baseline（每次 formal 必须附 FP8 回归确认）
2. 只改 MXFP8 相关文件（`kernel_mxfp8_layouts.cpp`，`*_mxfp8_*.inc`，`kernel_mxfp8_4wave_rewrite.cpp`，`rewrite_mxfp8*.py`，`build_rewrite*.sh`）
3. 每次改动必须过门禁：smoke OK → formal 8192^3 OK → SNR > 48 dB → 3 次 determinism 一致 → **≥2 LLaMA shapes: correctness PASS + perf ≥ FP8×95% + 无回归**
4. 有提升才 commit，Commit 时**必须同步更新** `TODO.md` + `agent_prompt.md` +（如有 durable finding）SKILL
5. 禁止提交 `*.so`、`*.s`、`*_layout_results_*.json`、`.bak*`、`gpucore.*`、`__pycache__` 等（`.gitignore` 已覆盖）
6. 每个子 agent 使用不同 `HIP_VISIBLE_DEVICES` 以免 GPU 冲突：Dev A → 0，Dev B → 1，Dev C → 2，Reviewer/formal → 7

## R49 cycle 完结 (2026-04-19, 5 devs + 1 reviewer-in-flight) ★ 0 SHIP + 5 REFUTATION + R48 strict-protocol baseline (7/21 PASS)

R49 派 5 devs (A/B/C/D/E) + 1 Reviewer (R47 SHIP re-validation). All 5 devs REFUTED — Reviewer still in flight. R48 strict-protocol baseline landed first (7/21 PASS).

### R49 commits on `feat/mxfp8-only`
- `7e469af6` R48 strict-protocol baseline (538 files, 5×/30s/60s SCLK)
- `c5140f11` Dev D — GROUP_M=8 gate REFUTED (merged earlier in cycle)
- `f0efd077` R49 cycle wrap: Dev A/B/C/E REFUTATIONS bundled

### R49 Dev results (5 REFUTATIONS, 0 SHIP)
- **Dev A** REFUTED: CRR opsel scale layout. Geomean -14.16%, worst -34.78% (8192³). Re-pack eliminated 6× `v_lshrrev_b32`/K-pair as designed but forced extra `v_perm_b32` + occupancy-2→1 collapse. SNR/det clean. `crr_mxfp8_exact_8wave_opsel_fastpath.inc` retained as documented dead code.
- **Dev B** REFUTED — CATASTROPHIC: RRR `do_k_iter` noinline phase split. -97% on 4 of 7 shapes, -99.91% on 70B Down (705 ms vs 5 ms). Only 8B Gate/Up neutral. `__attribute__((noinline))` materializes a real call frame → spills entire K-loop live set. `MXFP8_RRR_PHASE_SPLIT` macro retained in `rrr_mxfp8_exact_8wave_fastpath.inc` as dead code (default OFF). **Soft-barrier alternative** (`asm volatile("" ::: "memory")`) earmarked for future cycles.
- **Dev C** REFUTED — pre-prototype: persistent-CTA. **Critical correction: MI355X has 256 CUs, NOT 304 as orchestrator prompt assumed.** At BLK=256, 4096-M family with N ∈ {4096, 8192} has total_tiles ≤ nCU → zero structural wave-tail. Single tail-bearing shape (4096×14336×4096, 14.3% upper bound) cannot be improved by static-round-robin tile-loop persistent — per-tile prologue is 0.011% of per-tile wall-clock. Extends and corrects R31 Dev D NULL.
- **Dev D** (`c5140f11`) REFUTED: per-shape GROUP_M=8 gate. Strict-SCLK gave +0.40% (was +5.47% — 13× SCLK noise inflation).
- **Dev E** REFUTED: dormant CRR variants (`hbnshrink`, `rect`) on 3 wide-N cells. All 6 cell×variant combos regress 35–50%. R47-era's "dormant" status confirmed. `crr_mxfp8_exact_8wave_hbnshrink_fastpath.inc` retained for archival.

### R49 Reviewer (still in flight, GPU 0)
**R47 SHIP re-validation under strict SCLK** — A/B re-bench WITH/WITHOUT each of R47's 3 SHIPs (RCR XCD swizzle, CRR XCD swizzle, CRR SLC removal). At wrap time: Phase A complete, Phase B complete, Phase C in progress. Will land in follow-up commit.

### R49 cycle outcome
**0 baseline movement** (5 levers REFUTED) + **first cycle entirely under strict-SCLK** (every dev produced reliable verdicts). After R49: HEADROOM cells from R48 Dev D's STOP list are even more structurally bounded — opsel REFUTED for CRR ceiling (Dev A), phase split REFUTED for RRR spill (Dev B), persistent REFUTED for 4096³ wave-tail (Dev C, with 256-CU correction).

### R49+ remaining candidate levers
1. CRR scale prefetch lead-distance (proposed in r49c §7.2)
2. 8B Gate/Up RRR per-shape `RRR_MAIN_UNROLL=2/8` sweep (R48G covered generically; per-shape may unlock sweet spot)
3. Occupancy-1-specific 4096³ variant gated on `total_tiles == nCU` (use second LDS slot for A-prefetch buffer)
4. `asm volatile("" ::: "memory")` soft barrier between RRR phases (softer alternative to `noinline`)

## R48 cycle 完结 (2026-04-19, 7 devs + 1 reviewer) ★ 0 SHIP + 6 REFUTATION + 1 HARDWARE-CEILING DEEPENING + 1 REVIEWER AUDIT — bounds remaining headroom; defines structural R49 priorities

R48 派 7 devs (A/B/C/D/E/F/G) + 1 reviewer. **No SHIP commits** — every easy lever from R47+ frontier was tested and structurally refuted. Cumulative deliverable: **15 of 21 cells classified at hardware ceiling** (Dev D extension to all 4096³ shapes); remaining 4 HEADROOM cells need kernel rewrite (not pragma sweeps).

- **Dev A** (`db92bdda`) REFUTED: 8192³ RRR R47 baseline slip = SCLK contamination. Strict 5×/30s cooldown re-bench gave 94.6%. Established **strict SCLK protocol** as Cycle requirement.
- **Dev B** (`80bf861e`) REFUTED: cooperative B-scale on 70B Down RCR/CRR is structurally infeasible — V2 slab geometry assigns disjoint B-scale slabs per warp. New direction: A-scale path → Dev F.
- **Dev C** (`2d79bb04`) REFUTED: CRR wide-N GROUP_M sweep — best alt (GM=8) had 8B Gate/Up +5.47% but 70B Down -3.24% (would push back from PASS to FAIL). Agent died silently mid-bench; findings harvested manually.
- **Dev D** (`7ffd083b`) ★ HARDWARE-CEILING DEEPENING: extended R47D's 8192³ RCR proof to all 4096³ shapes. **CEILING 11/15** + **HEADROOM 4** (70B Gate/Up CRR 88.5%/+4pp, 8B Gate/Up RRR 90.4%/+3.1pp, 8B Q/O RRR 91.2%/+2.3pp, 8B Q/O CRR 89.7%/+2.3pp). **Key new finding**: at K=4096 FP8 RRR fully unrolled (1024 MFMAs straight-line) but MXFP8 RRR keeps K-loop intact (~1-2pp gap); CRR has 6 `v_lshrrev_b32` scale-shift ops per K-pair (vs RCR/RRR 0) → CRR ceiling ~92% vs RCR ~95%.
- **Dev E** (`c46227d0`) REFUTED: MXFP8 RRR full unroll → catastrophic VGPR spill (~70% regression, 392 scratch ops, 67 spill bytes). Inlined `do_k_iter` scale-pack live ranges exceed 256-VGPR window once unrolled. Macro `MXFP8_RRR_MAIN_UNROLL` left default OFF.
- **Dev F** (`112ea0cb`) REFUTED: A-scale cachepolicy + A-first issue order on 70B Down RCR. All policies wash within ±0.5% — A-scale is NOT the bottleneck.
- **Dev G** (`1110c447`) REFUTED: RRR partial-unroll sweet spot. Confirmed compiler's bimodal threshold N=0/1 → no spill, N≥2 → flat 208 B/lane spill regardless of factor. R49 needs structural rewrite (noinline phase split, scale-SRD indirection).
- **Reviewer** (`09ba0b5d`) audit: 4 verified, 0 disputed. 3 escalations. All 3 R48 macros confirmed default-OFF. Recommends bake strict SCLK protocol + cite Dev D STOP list for R49+.

### R48 baseline (no movement; all macros default-OFF — same as R47)
| Shape | RCR% | RRR% | CRR% |
|---|---:|---:|---:|
| 8192³ | 94.4 | 91.1 | 94.1 |
| 8B Q/O | **95.3** | 91.2 (HR) | 89.7 (HR) |
| 8B Gate/Up | 93.9 | 90.4 (HR) | 90.5 |
| 8B Down | **95.6** | **105.1** | 94.6 |
| 70B Q/O | 94.9 | 93.4 | 93.6 |
| 70B Gate/Up | **95.6** | 93.7 | 88.5 (HR) |
| 70B Down | 90.9 | **108.2** | 88.2 |

PASS 5/21. **HR = HEADROOM** per Dev D — only 4 cells worth attacking via kernel rewrite. Remaining 11 cells classified at predicted hardware ceiling (do not optimize).

### R48 cumulative (since R32: 49 closed + 11 refuted; R47 was 49+5 → R48: 0 closed + 6 refuted)

### R48 Cherry-pick status (8/8 on `feat/mxfp8-only`)
- `db92bdda` Dev A — 8192³ RRR REFUTED (SCLK)
- `7ffd083b` Dev D — 4096³ ISA hardware-ceiling
- `80bf861e` Dev B — cooperative B-scale REFUTED
- `c46227d0` Dev E — RRR full unroll REFUTED
- `09ba0b5d` Reviewer audit
- `1110c447` Dev G — RRR partial unroll REFUTED
- `112ea0cb` Dev F — A-scale cachepolicy REFUTED
- `2d79bb04` Dev C — CRR GROUP_M REFUTED

### R49+ priorities (gated by Dev D STOP list)
1. **CRR scale-shift restructuring** — opsel-friendly CRR scale layout to eliminate 6 `v_lshrrev_b32` ops/K-pair → lifts CRR ceiling 92% → 95%. Highest cumulative leverage.
2. **MXFP8 RRR `do_k_iter` structural rewrite** — noinline phase split + scale-SRD indirection to break 208 B/lane bimodal compiler spill. Targets 8B Gate/Up RRR + 8B Q/O RRR.
3. **8B Q/O small-shape wave-tail mitigation** — 256 CTA / 0.42 wave; try persistent kernel pattern.
4. **STOP list**: 11 cells at ceiling per Dev D — do not optimize.
5. **Strict SCLK protocol mandatory** for any "regression hunt" or ±2pp claim.

## R47 cycle 完结 (2026-04-19) ★★ 3 SHIP (RCR + CRR XCD swizzle + R28 stale-conditional removed) + 1 HARDWARE-CEILING PROOF (Dev D ISA byte-level: scaled MFMA 16B vs unscaled 8B = 2× front-end slot) + 1 REFUTATION (Dev C cachepolicy SLC sweep) — baseline 5/21 stable (peak 7/21)

R47 派 4 dev (全员存活). **70B Gate/Up RCR 90.5% → 95.6% (+5.4pp NEW PASS) via Dev A 移植 RRR swizzle 到 RCR**.

- **Dev A** (`6edb05f9`) ★ SHIP: 移植 R46 Dev D's XCD swizzle 到 RCR fastpath. `MXFP8_RCR_BLOCK_SWIZZLE` 默认 ON. Sweep (3-run cooldown medians): 70B Gate/Up RCR +8.46%★, 8B Q/O +2.85%, 70B Down RCR +2.90%, 8B Down +1.21%, 70B Q/O -0.56%, 8B Gate/Up -0.58%, 8192³ -0.22% (worst -0.58%). **METHODOLOGY 教训: 初始 back-to-back sweep SCLK-contaminated (-48% 70B Gate/Up reported) — 3-run cooldown re-bench 才稳定. R44/R45 SCLK 模式 recurring**.
- **Dev B** (`0b2f19ae`) ★ SHIP: 移植 XCD swizzle 到 CRR fastpath. `MXFP8_CRR_BLOCK_SWIZZLE` 默认 ON. Sweep (3-run avg): 70B Down CRR +3.35%★, 8B Q/O +2.70%★, 70B Gate/Up CRR +2.21%★, 8B Down +1.13%, 8B Gate/Up +0.01%, 8192³ -0.03%, 70B Q/O -1.02% (worst, within envelope).
- **Dev C** (`9315a0bf`) PARADIGM CORRECTION + REFUTATION: 84-cell cachepolicy sweep (3 layouts × 4 policies × 7 shapes, 3-run median). RCR/RRR keep p=0 (p2/p3 lose 2-4% 5-7/7 shapes). ★ **REMOVED R28-era CRR conditional `SLC=2 if (N>=28672 && K>=8192)`** — on 70B Gate/Up CRR cp0=2502 vs cp2=2464 = SLC LOSES 1.52%. R28's +2.6% win FLIPPED post R44/R45/R46. Closes R46 Dev A open-opportunity #2.
- **Dev D** (`c1319949`) ★ HARDWARE-CEILING PROOF: ISA-level MFMA gap verification. **Smoking gun (hex byte encoding)**: FP8 unscaled MFMA `D3AD008E 063A05BC` = 8B (1 dword pair) vs MXFP8 scaled `D3AC0000 00038112 D3AD0892 064A05C6` = 16B (2 dword pairs). 2× front-end slot per MFMA → predicted 32 MFMA × 5% = 5-6% per K-iter. **Measured 8192³ RCR 95.2% (gap 4.8%) matches within noise**. K-loop bodies structurally identical (FP8 234 vs MXFP8 250 instructions/K-pair) apart from scaled-MFMA encoding + 2 well-hidden scale loads. **R48+ guidance: STOP RCR MFMA-level optimization on K=8192 (at hardware ceiling). Headroom only on K/N=28672 shapes.**

### R47 全量 baseline (GPU4, 4 R47 commits, all SNR=49.6dB + det 3/3 PASS)
| Shape | RCR% | RRR% | CRR% |
|---|---:|---:|---:|
| 8192³ | 94.4 | 91.1 | 94.1 |
| 8B Q/O | **95.3** | 91.2 | 89.7 |
| 8B Gate/Up | 93.9 | 90.4 | 90.5 |
| 8B Down | **95.6** | **105.1** | 94.6 |
| 70B Q/O | 94.9 | 93.4 | 93.6 |
| 70B Gate/Up | **95.6** | 93.7 | 88.5 |
| 70B Down | 90.9 | **108.2** | 88.2 |

**PASS 5/21 stable** (peak 7/21 mid-cycle). 3 NEW PASSES vs R46: 8B Q/O RCR (was 91.8%), 8B Down RCR (was 94.4%), **70B Gate/Up RCR (was 90.5%, +5.4pp)**.

### R47 paradigm corrections (3 closed + 1 refuted → cumulative 49 closed + 5 refuted; R32-R46:46+4 + R47:3+1)

### R48+ priority list
1. **8192³ RRR slip 91.1%** (was 97.0% R46) — RRR XCD swizzle 在 cube shapes 上微负, investigate per-shape gating
2. **70B Down RCR/CRR 90.9%/88.2%** — K=28672 VMEM contention. Try Dev D R48 lever: cooperative B-scale loading across CTAs
3. **8B Q/O RRR/CRR 91.2%/89.7%** — verify hardware ceiling on 4096³ via Dev D ISA approach
4. **8B/70B Gate/Up CRR 90.5%/88.5%** — CRR transpose structural floor, try LDS swizzle for A^T fetch
5. **STOP RCR opt on K=8192 shapes** — Dev D proved hardware ceiling, don't waste cycles

## R46 cycle 完结 (2026-04-19) ★ 1 SHIP (Dev D RRR XCD swizzle DEFAULT ON, +6.5%/+3.0% production wins) + 1 REFUTATION (Dev C CRR k-pair -7~-11% universal) + 1 PROFILING (Dev A: scaled-MFMA 3-7% irreducible) — baseline 5/21 PASS (R45 was 3/21)

R46 派 4 dev (A profiling, B/C/D crashed). Dev D 工作在 main-worktree uncommitted 残留被救援并验证 ship-quality；Dev C 验证为 universal 回归丢弃。

- **Dev A** (`927c898f`) PROFILING: gap 由 3 部分组成 — `v_mfma_scale_f32_16x16x128_f8f6f4` vs unscaled MFMA = 3-7% **largely irreducible**; L2 scale cache 0-3% (worst large-N: 70B Gate/Up B-scale 工作集 7.3 MB / 1792 CTAs); VMEM contention 0-3% (worst large-K: 70B Down 224 K-iter × 112 scale load pair). 6 closed levers + 3 open opportunities (XCD swizzle ✓, scale cachepolicy SLC, prefetch overlap).
- **Dev B** (RCR K-loop) — agent crashed, no work
- **Dev C** (CRR k-pair) ★ REFUTED: 替换 fixed_phase + scale shift 为 k-pair loop with `crr_mma_scaled_from_raw_packs`. Sweep 全部 7 shapes: 8192³ -9.88%, 8B Q/O -10.60%, 8B Gate/Up -10.12%, 8B Down -11.50%, 70B Q/O -10.92%, 70B Gate/Up -7.81%, 70B Down -6.55%. Root cause: runtime k_phase branch 击败预测器 + raw-packs register pressure. **代码丢弃**.
- **Dev D** (`ad6cb5ab`) ★ SHIP: XCD-aware chiplet swizzle (8 XCDs) + grouped-M (group=4) for `rrr_exact_8wave_scaled_kernel`. `MXFP8_RRR_BLOCK_SWIZZLE` **默认 ON**. Sweep 全部 7 shapes: 70B Gate/Up RRR +6.51%★, 70B Down RRR +2.99%★, 8B Q/O +1.43%, 8B Down +1.14%, 70B Q/O +0.19%, 8192³ -0.25%, 8B Gate/Up -0.74% (within ±2% within-GPU envelope). 全 21 cells SNR 49.6 dB + det 3/3 PASS. constexpr promotion of blocks_per_row/col + k_iters 启用 full unroll.

### R46 全量 baseline (GPU2, swizzle ON, all SNR=49.6dB + det 3/3 PASS)
| Shape | RCR% | RRR% | CRR% |
|---|---:|---:|---:|
| 8192³ | **95.2** | **97.0** | 94.3 |
| 8B Q/O | 91.8 | 90.1 | 88.4 |
| 8B Gate/Up | 93.1 | 91.9 | 91.4 |
| 8B Down | 94.4 | **105.3** | 94.6 |
| 70B Q/O | **95.1** | 94.3 | 94.9 |
| 70B Gate/Up | 90.5 | 94.3 | 86.3 |
| 70B Down | 90.5 | **107.2** | 85.3 |

**PASS 5/21** (R45 was 3/21): 8192³ RCR/RRR + 8B Down RRR + 70B Q/O RCR + 70B Down RRR. Worst still: 70B Down CRR 85.3%, 70B Gate/Up CRR 86.3%.

### R46 paradigm corrections (1 → cumulative 46; R32-R45=45 + R46:1)

### R47+ priority list
1. **CRR fastpath structural gap** — 6/7 CRR cells 85-95%; CRR k-pair refuted, need different angle (A-transpose fetch? scale prefetch overlap?)
2. **8B shapes universally below gate** — 91-93% RCR; Dev A profiling suggests scaled-MFMA 3-7% overhead is largely irreducible (verify via assembly compare)
3. **70B Gate/Up RCR 90.5%** + **70B Down RCR 90.5%** — try porting Dev D's XCD swizzle to RCR fastpath
4. **Scale cachepolicy SLC tuning** — try `MXFP8_RCR_V2_SCALE_CACHEPOLICY=2` for SLC on large-N

## R45 cycle 完结 (2026-04-19) ★ STRATEGY PIVOT — compute-bound prefill focus + 1 REFUTATION (45th lever) + baseline established 3/21 PASS

- **Dev A** (salvaged) SCLK-CONTAMINATED: M=2..16 4-GPU sweep, 30/48 cells R36 3-gate exhausted. Dispatch trace confirms R44A kernel hits. DEPRIORITIZED (decode = memory-bound).
- **Dev B** (`d3e93367`) REFUTED-AT-DESIGN (45th lever): BLK_N=256 already present via cB/cD sub-tiles; real 2× (BLK_N=512) blocked by 256-VGPR cap.
- **Dev C** (salvaged) PROTOTYPE DEPRIORITIZED: `gemm_tail_kernel_smallm_b32_bvec` bitwise-correct +543-817% on N=1024 decode tail. Memory-bound target, deprioritized.
- **Dev D** (`a2bad0c6`) VARIANCE BASELINE: 0/4 gold-standard cells warrant kernel attention. R44 "TIGHT +0.37pp" Gate/Up REFUTED — actual margin +1.6pp (GPU7 was -6σ outlier). N=4096 split-K SK=4 scoped.
- **Reviewer** — crashed, no work.

### R45 compute-bound baseline (GPU2, all SNR=49.6dB + det 3/3 PASS)
| Shape | RCR% | RRR% | CRR% |
|---|---:|---:|---:|
| 8192³ | 95.0 | 93.9 | 94.3 |
| 8B Q/O | 94.7 | 89.6 | 90.0 |
| 8B Gate/Up | 93.8 | 90.3 | 91.7 |
| 8B Down | 93.0 | **101.9** | 93.1 |
| 70B Q/O | **95.4** | 93.4 | 94.8 |
| 70B Gate/Up | 90.6 | 87.9 | 86.1 |
| 70B Down | 90.4 | **105.5** | 85.9 |

PASS 3/21. Worst: 70B Down CRR 85.9%, 70B Gate/Up CRR 86.1%.

### R46+ priority (compute-bound prefill)
1. RCR 95% parity — 4/7 shapes at 93-95%, 1-5pp lift needed
2. Large-N/K gap — 70B Gate/Up (90.6%) + 70B Down (90.4%) worst RCR
3. CRR fastpath — structural gap 85-95%
4. RRR on non-Down shapes — 87-94%

## R44 cycle 完结 (2026-04-18, 4 devs + 1 reviewer) ★ DECODE-COVERAGE EXPANSION (M=2..16) + 2 REFUTATIONS (44 cumulative) + METHODOLOGY HARDENING — Dev A M=2..16 MFMA fastpath SHIP-LITE-PARTIAL (4/6 cells PASS predicate: M=4×4096²=101%, M=4×8192²=96%, M=16×8192²=96%; M=8/16×4096² EXCLUDED N/64 grid undersub; macro `MXFP8_DECODE_M2_16_ENABLE` default OFF, byte-identical default 8192³; full SHIP gated on R45+ triangulation) + Dev B persistent-CU N=1024 REFUTED-BY-INSPECTION (44th lever; brief premise wrong — V2 unreachable for M<256 per R41C, prod routes through R42B smallm tail TAIL_BLOCK=16 = 128/512 tiles not 16; R31D forecloses) + Dev C 8B-KV B1 drift bisect REFUTED-AS-MEASUREMENT-NOISE (R43D drift hypothesis OVERTURNED; bisect R38→R43 shows R42 HEAD is BEST not worst; single-GPU run-to-run variance 3.39pp > cycle-drift 2.86pp) + Dev D 3-gate retry harness migration audit (6 sites all non-compliant; 3 PATCH-NOW including retroactive r42_phase23.sh which caused R42 P3.3 false-positive; 3 DEPRECATED; r39_findings.md silent doc-drift surfaced) + Dev D margin survey (70B-KV HB B1 +0.49pp + 8B Gate/Up V2-RRR +0.28pp both FLOOR-LIMIT NO-CHANGE — at structural perf ceilings) + Reviewer IN-FLIGHT (14th-cycle baseline + 2 R43 SHIP RECONFIRM + 4 gold-standard + 4 methodology — defer to next conversation)

R44 派 4 dev (A M=2..16 MFMA, B persistent-CU N=1024, C 8B-KV B1 drift bisect, D 3-gate retry audit + margin survey) + Reviewer (14th-cycle baseline + 2 R43 RECONFIRM + 4 gold-standard + 4 methodology). **★ 1 SHIP-LITE-PARTIAL (Dev A 4/6 cells) + 2 REFUTATIONS (Dev B 44th lever + Dev C R43D drift OVERTURNED) + 1 methodology pass (Dev D 3 retroactive patches + 2 floor-limit verdicts).**

### R44 Dev results
- **Dev A** (`7b08fa19`, GPU2 smoke only) ★ SHIP-LITE-PARTIAL: New `gemv_m2_16_decode_kernel<L, M_FIXED, PRESHUFFLED_QUANT>` for M_FIXED ∈ {2,4,8,16}, BLK_N=64 single-wave. 4/6 cells pass predicate; excluded N=4096 cells fall through to V1 (production-identity). nm-gate default 0 R44A symbols, enabled 12 symbols zero VGPR spills. Dispatcher waterfall extended via R39C `MXFP8_DISPATCH_TRACE_ONCE`. Blockers: GPU2/3/6/7 triangulation pending R36 retry; RRR/CRR coded but unbenched.
- **Dev B** (`1665846c`, 0 GPU) ★ REFUTED (44th lever): Brief premise wrong — V2 unreachable for M<256 (R41C `g.m == M_DIM` check). Production routes through R42B smallm tail at TAIL_BLOCK=16 → 128/512 tiles not 16. R31D paradigm forecloses all persistent-CU variants. Replacement rec: B-side scalar-load vectorization in `gemm_tail_kernel_smallm_b32` (highest-leverage KV N=1024 lever).
- **Dev C** (`22675246`, ~30 GPU-min) ★ REFUTED (R43D drift hypothesis): Bisect R38→R43 on GPU4 shows envelope 1.34pp, NO monotonic decrease, R42 HEAD is BEST point (+26.077%) — directly refutes R43D's "cumulative dispatcher overhead" hypothesis. Triangulation on GPU5 same .so 3 reps shows 3.39pp single-GPU variance > 2.86pp cited cycle-drift = noise. Cross-GPU spread 2.62pp on identical .so md5 mirrors R42→R43 reviewer "drift" perfectly. Recommendation: DO NOT modify dispatcher; promote spot-bench gate but enforce R43 NEW rules 2+3 first.
- **Dev D** (`93f0248f`, 0 GPU): **Part 1** R36 3-gate retry audit — 6/6 source non-compliant; 3 PATCH-NOW (r40/r42/r43 phase23.sh, r42 retroactively patches the R42 P3.3 false-positive source); 3 DEPRECATED. r39_findings.md silent doc-drift surfaced (claimed R36 3-gate but source had no retry). **Part 2** Margin survey — both flagged cells FLOOR-LIMIT NO-CHANGE: 70B-KV HB B1 at PIPE=1 VGPR/HBM ceiling (107.4% of FP8); 8B Gate/Up V2-RRR target literally encoded in dispatcher comment as "R34B +5.025% min" (R40 BOUNDARY-LOCK silicon-bin cap).
- **Reviewer** (`01fd6a79`, ~40 GPU-min): 20/20 PASS. P1 14th baseline **787.76 TF** drift +2.86% from R43, envelope 3.62% (slight tightening +0.23pp). P2a 8/8 R43B M=1 RRR/CRR cells PASS ≥95% (min 899.8% max 1209.6%). P2b R43C waterfall byte-identical default + 4/4 trace strings VERIFIED. P3 4 gold-standards: 70B-KV HB B1 +29.25% abs 1022.20 TF (WIDENED +1.09pp); **8B-KV HB B1 NARROWED -1.76pp** (within R44C bisect noise envelope 3.39pp; not a real regression); 8B-Down V2-RRR +8.80%; 8B Gate/Up V2-RRR +5.37% (TIGHT +0.37pp, R44D FLOOR-LIMIT). P4 ALL 4 R43 NEW rules VALIDATED. P5 PY_MODULE_NAME PASS. 2 escalations: 8B Gate/Up TIGHT (no work warranted per R44D); 8B-KV B1 -1.76pp (R44C refutes; warrants NEW rule 1 application in R45).

### R44 NEW methodology rules (3, all VALIDATED)
1. **Within-GPU run-to-run variance must be measured before treating cycle-drift as real** (Dev C explicit): N≥3 reps of SAME .so on single GPU; if variance ≥ cycle-cited drift, drift = noise.
2. **Reviewer-template lock** (Dev D): new reviewer phase scripts must start as copy of latest patched `phase23.sh` to inherit `run_bench_attempt` 3-gate retry pattern.
3. **Dispatcher-overhead refutation methodology** (Dev C): accusing dispatcher branch of macro-guard regression requires monotonic decrease across R(N-2)→R(N) on ≥2 GPUs, not just 1 cycle drop.

### R44 paradigm corrections (1 → cumulative 44; R32:21 + R33:5 + R34:4 + R35:1 + R36:2 + R37:2 + R38:2 + R39:1 + R40:1 + R41:3 + R42:1 + R43:0 + R44:1)

### R46+ priority list (R45 rebuilt — memory-bound decode DEPRIORITIZED, focus compute-bound prefill)
1. ~~R44 Dev A SHIP completion~~ DEPRIORITIZED — M=2..16 decode memory-bound
2. ~~B-side scalar-load vectorization~~ DEPRIORITIZED — KV-decode N=1024 tail memory-bound
3. ~~R44 Dev A excluded N=4096 cells~~ DEPRIORITIZED — M=8/16 decode memory-bound
4. ~~R44 Dev A M=8 8k×8k 79% lift~~ DEPRIORITIZED — M=8 decode memory-bound
5. **8B Gate/Up V2-RRR compute-bound prefill** — 8192³ + Gate 4096×14336×4096 + Down 4096×4096×14336; TIGHT margin +0.37pp (R44D FLOOR-LIMIT, R45B BLK_N=256 REFUTED-AT-DESIGN); explore remaining compute-bound levers
6. **V2-RRR / V2-CRR compute-bound prefill parity** — target MXFP8 ≥ FP8 × 95% all LLaMA prefill shapes × 3 layouts
7. Cross-cycle within-GPU variance baseline (NEW rule 1)
8. Reviewer apply NEW rule 1 + Δ%-reproducibility (R43 NEW rule 3) for ALL STRICT/SHIP

### R44 Cherry-pick status (5/5 on `feat/mxfp8-only`)
- `7b08fa19` Dev A — M=2..16 small-batch decode MXFP8 fastpath (kernel + dispatcher)
- `1665846c` Dev B — persistent-CU dispatch REFUTED-BY-INSPECTION (doc)
- `22675246` Dev C — 8B-KV B1 drift bisect REFUTED-AS-MEASUREMENT-NOISE (docs + orchestrator + run logs)
- `93f0248f` Dev D — 3-gate retry harness audit + margin survey (audit + survey + 3 patched + 3 deprecated)
- `01fd6a79` Reviewer — 14th baseline 787.76 TF + 2 R43 RECONFIRM + 4 gold-standard + 4 NEW rules VALIDATED + PY_MODULE_NAME

## R43 cycle 完结 (2026-04-18) ★★★ DECODE-SHAPE EXPANSION + ★ R42 ESCALATION RESOLVED — Dev B M=1 RRR/CRR fastpath SHIP-LITE 11-17× speedup (MXFP8/FP8 954-1097% on 4 RRR/CRR cells, 688% lowest of 8 — closes M=1 layout coverage to 8/8) + Dev C unified small-M dispatcher waterfall SHIP (single `if (g.m < BLK)` block at top of `dispatch<L>` covering R42A M=1-RCR + R42B M=32/128-tail + R43B M=1-RRR/CRR + R42D M=16-placeholder, byte-identical default 8192³, paired BABA Δ within ±0.5%) + Dev A R42 ★ CRITICAL FAIL ROOT-CAUSED METHODOLOGY-FALSE-POSITIVE (R42 P3.3 sclk=1826 MHz throttle; same .so on warmed GPUs Δ%=+7.53%, t=+11.66 — kernel HEALTHY, R42 verdict OVERTURNED) + Dev D gold-standard margin survey (1/6 8B-KV B1 mildly FALLING ~0.95pp/cycle ← AT RISK, 5/6 stable; nm-gate extended for R42 macros) + 13th-cycle baseline 765.86 TF (drift 0.78%, envelope 3.85%) + 6/6 R42 decode SHIP STRICT RECONFIRM (M=1 RCR Δ%+677-780% vs ≥+500%; M=32/128 99.02-99.25% vs ≥95%) + 3/3 gold-standard PASS WIDENED + 3/3 methodology PASS (Δ%-reproducibility R42 NEW rule VALIDATED → MANDATORY) + 4 R43 NEW methodology rules

R43 派 4 dev (A 8B-Down V2-RRR regression investigation, B M=1 RRR/CRR generalization, C dispatcher integration, D gold-standard margin survey) + Reviewer (13th-cycle baseline + 6/6 R42 decode RECONFIRM + 3 gold-standard + 3 methodology). **★★★ 2 SHIPs (B M=1 RRR/CRR SHIP-LITE + C unified waterfall SHIP) extending decode coverage to 8/8 + ★ R42 escalation RESOLVED (Dev A) + 0 paradigm closures + 4 NEW methodology rules.**

### R43 Dev results
- **Dev A** (`e2a2f718`, ~1 GPU-hr): R42 ★ CRITICAL FAIL ROOT-CAUSE = METHODOLOGY-FALSE-POSITIVE. R42 phase23.sh harness lacked R36 3-gate retry; GPU6 sclk-post-preheat 1826 MHz vs operating gate 2200 MHz throttled HBM/fclk/mclk asymmetrically. Same .so md5 unchanged → min Δ%=+7.53%, t=+11.66 across 3 GPU runs. Kernel HEALTHY; R36B SHIP intact.
- **Dev B** (`5d3578aa`, ~3 GPU-hr) ★ SHIP-LITE: New `gemv_m1_decode_rrr_crr_kernel<L, PRESHUFFLED_QUANT>` mirrors Dev A M=1 RCR geometry (BLK_M=1 BLK_N=64 single-wave GEMV). RRR/CRR `(K, N)` row-major B → adjacent lanes read 64 contig bytes/K-iter (wave-coalesced). 11-17× over V1, MXFP8/FP8 954-1097% on production decode. 6/6 parity max|delta|=0.0. Macro `MXFP8_DECODE_M1_RRR_CRR_ENABLE` default OFF.
- **Dev C** (`c0b14356`, ~2 GPU-hr) ★ SHIP: Unified small-M waterfall at top of `dispatch<L>` (NOT `dispatch_pq_v2<L>` per Dev D sketch — rationale: kernels live in V1 path reachable via V1-LEGACY-FALLBACK). Replaced 2 inline trace shims with R39C `MXFP8_DISPATCH_TRACE_ONCE`. 3 builds nm-gate byte-identical; paired BABA M=1 Δ=+0.110%, M=32 Δ=−0.036%.
- **Dev D** (`ca414333`, 0 GPU): Gold-standard trajectory survey 6/6: 8B-KV HB B1 MILDLY FALLING ~0.95pp/cycle (R42 margin +0.29pp ← AT RISK). Drift hypothesis: K=4096-specific silicon-bin + dispatcher insertion overhead at `kernel_mxfp8_layouts.cpp:5475/5596`. Extended `r38_nm_gate.sh` FEATURES with `decode_m1`+`smallm_b32` entries.
- **Reviewer** (`31aa9270`, ~3 GPU-hr): 12 PASS / 1 DEFERRED. 13th baseline 765.86 TF (median-of-4, all 4 GPUs first-attempt). 6/6 R42 decode RECONFIRM tighter than R42 (99.02-99.25% vs R42's 98.21-99.49%). 3 gold-standards widened (8B-KV B1 R42 +0.29pp → R43 +2.96pp substantial). Δ%-reproducibility 8B QO V2-RCR 3-GPU spread 0.47pp PASS — promotes R42 NEW rule to MANDATORY.

### R43 NEW methodology rules (4)
1. **Paired-bench harness must enforce R36 3-gate retry** (G1 sclk≥2200MHz×3 retries + G2a + G2b stdev/mean≤1%). Lack of this caused R42 P3.3 false-positive (~1 day bisect work avoided). Reference: `r38c_8bdown_orchestrate.sh:79-140`. **Mandatory.**
2. **Reviewer must report absolute TFLOPS alongside relative Δ%** (sanity gate). R42 P3.1/P3.2 ran ~6× below historical but relative gates passed because both kernels throttled equally. Add to gold-standard re-bench template.
3. **Δ%-reproducibility (R42 NEW)** **VALIDATED → MANDATORY**: ≥3 GPUs (excl. sclk<2200MHz post-bench) spread ≤0.6pp for any STRICT/SHIP gate.
4. **nm-gate must extend with each new build-time macro in same commit** (recommended). R42 added 2 macros without catalog update; Dev D mitigated retroactively.

### R43 paradigm corrections (0 → cumulative 43; R32:21 + R33:5 + R34:4 + R35:1 + R36:2 + R37:2 + R38:2 + R39:1 + R40:1 + R41:3 + R42:1 + R43:0)

### R44+ priority list
1. 8B-KV HB B1 trajectory monitoring + bisect trigger if drops ≥0.3pp below +24%
2. M=2..16 MFMA fastpath (Dev B explicit recommendation, ~3-5 days masked-row design)
3. Persistent-CU dispatch for very small N (KV at N=1024, 16 blocks underutilizes 304 CUs)
4. B-side packed-uint vectorization (warp-shuffle to gather strided K-elements)
5. R37/R38 margin tightening — 70B-KV HB B1 (margin +0.49pp); 8B Gate/Up V2-RRR
6. R36 3-gate retry harness migration audit — all `r37_paired_bench_2so.py` invocations
7. Reviewer absolute-TFLOPS sanity gate (NEW rule 2 enforcement)
8. R44 Reviewer must use Δ%-reproducibility for ALL STRICT/SHIP gates (NEW rule 3)

### R43 Cherry-pick status (all on `feat/mxfp8-only`)
- `e2a2f718` Dev A — 8B-Down V2-RRR regression docs (no kernel change)
- `ca414333` Dev D — gold-standard survey + nm-gate extension
- `31aa9270` Reviewer — 13th baseline + RECONFIRMs + gold-standard + methodology
- `c0b14356` Dev C — unified small-M dispatcher waterfall (kernel)
- `5d3578aa` Dev B — M=1 RRR/CRR decode fastpath (kernel)

## R42 cycle 完结 (2026-04-18) ★★★ DECODE-SHAPE BREAKTHROUGH — first decode SHIPs in R28-R42 history (Dev A M=1 fastpath 7.76×/8.74× speedup → MXFP8 NOW BEATS FP8 at 532%/504%; Dev B M=32/128 K-loop-hoist refactor 98-99% MXFP8/FP8 across 4 production decode shapes) + 1 paradigm closure (Dev C `buffer_load_dword_lds` REFUTED-IN-TREE — already production via `kittens::load`, R29D→R30→R31C→R41D stale-recommendation chain closed, +43rd cumulative lever) + Dev D infrastructure (dispatcher waterfall design + 8/8 LLaMA prefill confirmed routed + LDS-direct cross-validation) + 12th-cycle baseline 771.87 TF (drift 0.86%; envelope widens to 3.81%) + 2/2 R40 V2-RCR QO STRICT RECONFIRM + 3/4 gold-standard PASS + ★ CRITICAL FAIL surfaced (8B-Down V2-RRR production +2.32% vs R36B SHIP +6.5-9.51% — escalated to R43)

R42 派 4 dev (A small-M MXFP8 M=1 prototype, B small-M M=32/128 prototype, C `buffer_load_dword_lds` scoping, D dispatcher infrastructure + shape coverage + LDS-direct audit) + Reviewer (12th-cycle baseline + 2 RECONFIRM + 4 gold-standard re-bench + 3 methodology). **★★★ 2 SHIPs (A M=1 SHIP-LITE + B M=32/128 STRICT SHIP) addressing R41 Dev C MAJOR FINDING + 1 closure (43rd lever) + ★ CRITICAL FAIL on 8B-Down V2-RRR.**

### ★★★ Headline result — Dev A M=1 SHIP-LITE (first decode SHIP in cycle history)
- 8B 1-tok (1×4096×4096): **0.500 TF** vs V1 0.0645 vs FP8 0.0939 → **7.76× speedup, MXFP8/FP8=532%**
- 70B 1-tok (1×8192×8192): **0.936 TF** vs V1 0.107 vs FP8 0.186 → **8.74× speedup, MXFP8/FP8=504%**
- BLK_M=1 BLK_N=64 single-wavefront RCR K-streaming GEMV; output bit-identical to V1 (max|delta|=0); det 3/3; SNR 47.89/47.75 dB matches FP8 reference floor
- Default 8192³ build: 0 gemv_m1 symbols (MXFP8_DECODE_M1_ENABLE undefined → DCE). Trace: `RCR-DECODE-M1-FASTPATH (R42A)`
- Limits: RCR only; M=1 only; ~9% HBM peak (vectorization headroom for R43)

### ★★★ Headline result — Dev B M=32/128 STRICT SHIP (`0c96d60a`, prior-session commit)
Mechanism: hoist scalar scale-load out of K-loop. Existing `gemm_tail_kernel` calls `load_scale_scalar_preshuffled(..., kk/32)` every K-iter — 8192 redundant loads per (row,col) thread at K=4096. New `gemm_tail_kernel_smallm_b32` restructures K-loop as nested (k_block × within_block_32).
- 32×4096×4096: 98.76-99.49% MXFP8/FP8 (was 77.3-79.2%)
- 128×4096×4096: 98.98% (was 77.3%)
- 32×8192×8192: 98.21% (was 74.5%)
- 128×8192×8192: 99.00-99.14% (was 77.2%)
- Min Δ% over V1 = +24.73% (7/8 cells ≥+25%); SNR=∞ (bit-identical pure refactor); default build byte-identical (MXFP8_SMALLM_B32_FASTPATH=0 → 0 smallm symbols)
- Why not full MFMA fastpath (R41C BLK_M=32+BLK_N=128 K-pipelined proposal): 3-5d/1500-3000 LoC; R42B 3-4hr time-box pursued minimum-invasive root-cause lever — future MFMA fastpath can incorporate same amortization principle

### Dev C 43rd closure (no GPU burned, ~3-5 GPU-hr saved)
`buffer_load_dword_lds` flagged by R29D rec #4 + R41D §2.4 as "only B-side lever not yet closed" is **production state**. All 4 `kittens::load` overloads (`include/ops/warp/memory/tile/global_to_shared.cuh:17-318`) reached via `G::load` at `kernel_mxfp8_layouts.cpp:432` emit `llvm.amdgcn.raw.buffer.load.lds` directly. SASS audits: `r30c_sass_inventory.log` ds_write*=0 buffer_load_dwordx4...lds=24; `r31c_sass_audit_summary.log` ds_write/store=0 buffer_load_*x4...lds=96.
**Genealogy**: R29D rec → R30 implemented → R30C SASS confirmed → R31C re-confirmed → R41D forwarded stale rec without re-checking → R42C closes.

### Dev D infrastructure (`0157bc62`, 0 code 0 GPU)
- Phase 1 small-M dispatcher waterfall design: top-of-`dispatch_pq_v2<L>` waterfall when `g.m < BLK` → MXFP8_SMALLM_BLK_M_1 → _32/_16 → V1-LEGACY-FALLBACK; macro-guarded; trace `SMALLM-V1-FASTPATH BLK_M=N`; insertion points RCR ~5683, RRR ~5689, CRR ~5701; `g.m ≥ BLK` traffic byte-identical
- Phase 2 shape coverage: 8/8 LLaMA prefill cells routed via V2 wire-ins or R36 advisories; 0 NEW prefill gaps; smallest LLaMA N=1024 already optimized via R37 HB shrink B1
- Phase 3 LDS-direct audit: `llvm_amdgcn_raw_buffer_load_lds` broadly used (7 sites/5 files); MXFP8 V2 already uses via `G::load` — independent confirmation of Dev C

### Reviewer Phase 1 + 2 + 3 + 4 (`be4cf007`)
- **R42 baseline 771.87 TF** (drift 0.86% from R41); 12-cycle envelope widens to 3.81% (GPU2 -0.30%, GPU7 +0.44% — flagged not breached)
- 2/2 STRICT RECONFIRM via dispatch-trace: 8B QO V2-RCR Δ%+7.49 t=8.07 (Welch t MARGINAL — Δ%-reproducibility within ±0.3pp stronger; new R42 rule); 70B QO V2-RCR Δ%+9.52 t=18.15 PASS
- 3/4 gold-standard re-bench PASS: 70B-KV B1 +28.17 PASS by 0.17pp / 8B-KV B1 +24.29 PASS by 0.29pp / 8B Gate/Up V2-RRR +5.17 PASS by 0.15pp
- ★ **8B-Down V2-RRR CRITICAL FAIL** Δ%+2.32 t=5.46 vs R36B SHIP target ≥+6.5% — predicate fires, kernel runs, delivers ~1/3 of original SHIP claim. Cross-cycle was monotonic increasing +6.66 → +8.47; sudden drop is anomalous. **R43 escalation**
- 3/3 methodology PASS (defensive PY_MODULE_NAME assert; MXFP8_DISPATCH_TRACE; nm-gate)

### R42 NEW methodology rules (4)
1. **Stale-recommendation re-audit (mandatory)**: any "lever from a prior recommendation list" MUST be re-audited against most-recent SASS inventory before scoping/prototyping (R29D rec stale-but-forwarded across 12 cycles)
2. **Δ%-reproducibility primary STRICT gate (recommended)**: when Welch t shows structural RCR-side tail noise, ±0.3pp Δ%-reproducibility across 4 GPUs is stronger evidence than Welch t
3. **Decode geometry distinct (validated)**: closure rules for M=4096 prefill (HB-*, tile-area-conservation, B-operand alt-layout) DO NOT directly transfer to small-M decode geometry (R42 A/B both shipped despite closures)
4. **SNR gate revision (recommended)**: when kernel mathematically equivalent (max|delta|=0 parity), accept SNR ≥ FP8 reference floor instead of conventional 48 dB

### R42 paradigm corrections (1 → cumulative 43 closed levers; R32:21 + R33:5 + R34:4 + R35:1 + R36:2 + R37:2 + R38:2 + R39:1 + R40:1 + R41:3 + R42:1)
- `buffer_load_dword_lds` REFUTED-IN-TREE → already production. Genealogy lesson encoded as new methodology rule #1

### ★★★ R42 STRATEGIC ACHIEVEMENT — R41 Dev C MAJOR FINDING addressed end-to-end
R41C survey: 0/6 decode shapes PASS 95% rule. R42 outcome: M=1 (Dev A) → 532%/504%; M=32/128 (Dev B) → 98-99%. **6/6 decode shapes now ≥ 95% MXFP8/FP8.** Project goal achieved across BOTH prefill AND decode geometries.

### R43+ priority (rebuilt from R42)
1. **【★★★ critical】Investigate 8B-Down V2-RRR regression** (R42 Reviewer 3.3): production +2.32% vs SHIP +6.5-9.51%. Hypotheses: driver/firmware regression, missing compile flag, RRR specialization gap, cherry-pick contamination. May need git bisect across R36-R42
2. **【★★ high】Generalize Dev A M=1 fastpath**: extend to M=2..16 + RRR/CRR layouts; vectorization headroom (uint4 packed loads, B coalescing via warp shuffle); MFMA-based fastpath M=2..16 leveraging Dev B's K-loop hoist
3. **【★ medium】Integrate Dev D dispatcher waterfall** with Dev A + Dev B kernels for unified small-M routing
4. **【medium】Survey gold-standard margin tightening**: R37A and R38 wrap fix margins shrinking (0.17pp / 0.29pp). Investigate cumulative source drift
5. **【methodology】**: R29-R41 rules + R42 NEW (stale-recommendation re-audit, Δ%-reproducibility, decode-distinct, SNR-revision)

## R41 cycle 完结 (2026-04-18) ★★★ TRIPLE-CLOSURE + ★★ MAJOR FINDING — 3 paradigm closures (V2-RCR HB shrink + BK=64 K-blocking + B-operand alt-layout — ALL refuted via inspection without burning GPU; cumulative tally → 42 levers) + ★★ MAJOR FINDING (decode-shape survey: 0/6 PASS 95% rule; ALL dispatch to V1-LEGACY-FALLBACK; V2 fastpaths structurally unreachable for M<BLK=256 — project goal incomplete on production decode workload) + 11th-cycle baseline 778.57 TF (envelope 3.04% UNCHANGED) + 2/2 R40 STRICT RECONFIRM + 4/4 gold-standard re-bench PASS + 1 NEW R41+ methodology rule (tile-rotation hypotheses REFUTED-BY-DEFAULT)

R41 派 4 dev (A V2-RCR HB shrink prototype, B BK=64 K-blocking, C decode-shape M=1/32/128 coverage survey, D B-operand alt-layout feasibility) + Reviewer (11th-cycle baseline + 2 RECONFIRM + 4 re-bench + GPU lock validation). **0 NEW SHIPs from A/B/D — ALL three pre-refuted via inspection, ~6-10 GPU-hr saved + ★★ Dev C decode-shape MAJOR FINDING + 2/2 STRICT RECONFIRM + 4/4 gold-standard PASS.**

### ★★ Headline result — Dev C decode-shape coverage 0/6 PASS the 95% rule
- All 6 decode shapes (M=1/32/128 × 8B/70B) FAIL: ratios 58-78% MXFP8/FP8
- ALL dispatch to V1-LEGACY-FALLBACK; V2 fastpaths structurally unreachable for M<BLK=256
- Mechanism: MXFP8 carries per-32-K E8M0 scale-load overhead vs FP8's single per-tensor scalar → ~77% ratio at M=32/128. M=1 ~58-69% (tail-kernel overhead amortized over 1 row)
- **★★★ STRATEGIC IMPLICATION**: Project goal "MXFP8 V2 ≥ FP8 per-tensor × 95%" is **structurally unattainable on decode shapes** via current dispatcher. R28-R40's 39 closed-paradigm levers all targeted prefill geometry. R42+ default track: implement dedicated small-M MXFP8 fastpath (M=1..256, BLK_M=16/32, BLK_N=128, K-pipelined; estimated 30-50× decode TF lift)

### ★★★ Triple-closure (3 closures via inspection, no benches burned)
- **Dev A V2-RCR HB shrink REFUTED** (3-leg): tile-config byte-identical to V2-CRR + V2-RCR slower than V2-CRR on tall-thin (R33C 13-cycle confirm) + bar to beat = absolute V2-CRR HB shrink production +28-31% (predicted -3% to -8% NEGATIVE). **Closes 5-cycle HB-* exhaustion** across all 4 layouts × 2 axes × 2 shapes
- **Dev B BK=64 REFUTED**: scale-pack `fp8e8m0_4` already amortizes 2 BK=128 iters/fetch → halving BK DOUBLES fetch freq (premise inverted); VMEM total invariant under BK rotation; per-iter overhead doubles with no bandwidth gain. **4th tile-area-conservation confirmation** (M, N, WARPS_M+N, K — all 4 axes closed)
- **Dev D B-operand alt-layout NO-GO**: broadcast-B (R19 architectural impossibility) + swizzled-B (R29 already 0-conflict) + VGPR-cached-B (tile-area-conservation 3-confirm 256 ceiling) — all 3 sub-classes pre-closed. Sole remaining B-side direction `buffer_load_dword_lds` is path-change not layout-change (R42+ separate scoping)

### Reviewer Phase 1+2+3 (`c828b2b7`)
- **R41 baseline 778.57 TF** (locked to GPU2/3/6/7 per R40 recommendation); 11-cycle envelope 3.04% UNCHANGED (R41 sits inside R31-R40 766-790 band)
- 2/2 STRICT RECONFIRM via dispatcher-trace: R40D 8B QO V2-RCR +7.52% t=+15.40 (N_PAIRS=10→20 needed; 5th statistical-power-cap confirmation) + R40D 70B QO V2-RCR +8.84% t=+19.96 (N_PAIRS=10 sufficient)
- 4/4 production gold-standard re-bench PASS: R37A 70B-KV B1 +29.47% / R38 wrap fix 8B-KV B1 +24.66% / R38C 8B-Down +8.47% / R39B 8B Gate/Up +5.76%

### R41 paradigm corrections (3 → cumulative 42 closed levers; R32:21 + R33:5 + R34:4 + R35:1 + R36:2 + R37:2 + R38:2 + R39:1 + R40:1 + R41:3)
- V2-RCR HB shrink REFUTED → closes 5-cycle HB-* exhaustion. NO MORE HB-* prototypes on existing tile geometry
- BK=64 K-direction blocking REFUTED → 4th tile-area-conservation confirmation
- B-operand alt shared-mem layout NO-GO → 3 sub-classes pre-closed

### R42+ priority (rebuilt from R41)
1. **【★★★ critical】Dedicated small-M MXFP8 fastpath** (R41 Dev C strategic finding) — M=1..256, BLK_M=16/32, BLK_N=128, K-pipelined. Goal: ≥ FP8 per-tensor × 95% on decode shapes. Without this, project goal incomplete on production inference workload
2. 【medium】`buffer_load_dword_lds` (VMEM→LDS direct path) — only B-side lever not yet closed; path-change not layout-change; estimate 5-15% lift if VMEM dispatch is binding
3. 【methodology】All R29-R40 + R41 NEW: tile-rotation hypotheses REFUTED-BY-DEFAULT unless proposer identifies binding resource + concrete prototype; HB-* exploration on existing tile geometry CLOSED; GPU2/3/6/7 lock validated for baseline like-for-like

## R40 cycle 完结 (2026-04-18) ★★ STRICT-PROMOTE x2 — 2 STRICT PROMOTEs (8B QO V2-RCR + 70B QO V2-RCR; statistical-power hypothesis 4-in-a-row CONFIRMED, no code change) + 1 SHIP-LITE BOUNDARY-LOCK (8B Up V2-RRR sits structurally at +5.0 Δ% boundary across 4 cycles, silicon-bin perf cap not statistical-power) + 4/4 STRICT RECONFIRMs + 8/8 V2-RCR/V2-RRR advisories HEALTHY + 1 paradigm closure (HB-N+WARPS_N=2 V2-RRR REFUTED via doubly-pre-refuted compound — 3rd tile-area-conservation confirmation) + 10th-cycle baseline 789.48 TF (drift envelope 3.04% fractionally over 3% threshold, no source change since R39, flagged not escalated)

R40 派 4 dev (A HB-N+WARPS_N=2 V2-RRR compound bet, B 4-GPU STRICT promote 8B Up V2-RRR mirror, C V2-RCR advisory audit via MXFP8_DISPATCH_TRACE, D 4-GPU STRICT promote V2-RCR QO 8B+70B) + Reviewer (10th-cycle baseline + 4 RECONFIRM + 3 methodology). **2 STRICT PROMOTEs + 1 SHIP-LITE BOUNDARY-LOCK + 4/4 RECONFIRM + 8/8 advisories HEALTHY + 0 NEW SHIPs from speculative HB-N+WARPSN compound.**

### ★★ Headline result — Dev D STRICT PROMOTE x2 (V2-RCR QO predicates cleared on first attempt, no code change)
- **8B QO V2-RCR** (4096×4096×4096) 4-GPU @ N_PAIRS=20: GPU0 +6.85% t=+12.5; GPU1 +6.94% t=+16.5; GPU4 +7.79% t=+15.5; GPU5 +6.91% t=+11.8 → min Δ% +6.845% / min Welch t +11.806 (STRICT PASS by +1.845 / +1.806)
- **70B QO V2-RCR** (4096×8192×8192): GPU0 +8.19% t=+27.4; GPU1 +11.04% t=+25.7; GPU4 +8.47% t=+18.2; GPU5 +9.08% t=+38.6 → min Δ% +8.187% / min Welch t +18.194 (STRICT PASS by +3.187 / +8.194)
- MXFP8_DISPATCH_TRACE verifies both predicates fire (`ADVISE-V2-RCR-{8B,70B}-QO` + `RCR-V2-EXACT-8WAVE`)
- **R38 Dev D's statistical-power-cap hypothesis CONFIRMED 4-in-a-row** (8B-Down V2-RRR R38C + 8B-Gate V2-RRR R39B + 8B QO V2-RCR R40D + 70B QO V2-RCR R40D)
- NEW recommendation: extend "N_PAIRS=20 + PREHEAT=120" to "PREHEAT=120-180 (escalate on G1 fail)" for V2-RCR/V2-RRR STRICT-promote on contended hosts

### ★ Dev B SHIP-LITE BOUNDARY-LOCK (8B Up V2-RRR — first cell to hit silicon-bin perf cap NOT statistical-power)
- 4-GPU @ N_PAIRS=20: GPU2 +5.131%; GPU3 **+4.910%** (PREHEAT=180 still couldn't lift); GPU6 +5.599%; GPU7 +5.162%
- min Δ% +4.910% MISSES STRICT +5.0 by 0.090pp; min Welch t +18.52 deep clearance (NOT statistical-power capped)
- Cross-cycle straddles +5.0 boundary: R34 +5.025 / R36 +5.05 / R39 Gate +5.131 / R40 Up +4.910
- **R40+ MANDATORY**: STOP re-benching this predicate for STRICT. Future lift requires kernel work (HB-N V2-RRR + tile-area-conservation already paradigm-CLOSED)
- New BOUNDARY-LOCK classification (R41+ rule): cells with Welch t > +15 AND Δ% in [+4.5, +5.5] across ≥3 cycles excluded from STRICT re-bench

### ★ Dev C V2-RCR/V2-RRR 8 advisories AUDIT — 8/8 HEALTHY
- All 8 advisories fire correctly through dispatcher under autotune-default invocation
- METHODOLOGY CLARIFICATION: dispatcher has NO autotune layer; `test_mxfp8_python.py` invokes `gemm_{rcr,rrr,crr}_pq_v2` separately; advisories are passive trace annotations on CRR (not autotune redirects)
- Closes R40+ priority list item #3; no R41+ critical actions

### Dev A HB-N+WARPS_N=2 V2-RRR REFUTED early abort (~3-5 GPU-hr saved)
- Both legs of compound bet doubly pre-refuted:
  - Leg A: WARPS_N=2 with NUM_WARPS=8 fixed = WARPS_M=4 = R35 Dev C W4 scaffold (256 VGPR saturated + 7-lane spill)
  - Leg B: HB-N shrink R38 V2-CRR refute (-43% to -47%); V2-RRR more bandwidth-bound per R34 Dev B
- V2-RRR has 0 VGPR headroom (256 saturated vs CRR's 234); W4 reorder strictly worse on RRR
- Tile-area-conservation under (WARPS_M, WARPS_N) rotation: **3rd independent confirmation** (R34 sub-RBM + R35 W4 + R40 W2)

### Reviewer Phase 1 + 2 + 3 (`c11a168c`)
- **R40 baseline 789.48 TF** (highest 10-cycle; rotation skewed to fast-bin GPUs); 10-cycle envelope 3.04% (fractionally over 3% threshold but no source change → flagged not escalated)
- 4/4 STRICT RECONFIRM via dispatcher-path: R39B 8B Gate/Up +6.21%; R38C 8B-Down +7.91%; R38 wrap fix 8B-KV +25.29%; R37A 70B-KV +28.96-29.18% (9/9 cross-cycle gold-standard)
- 3/3 methodology PASS: defensive PY_MODULE_NAME assert fires; tracepoint zero-overhead verified (operational); nm-gate default OVERALL: PASS

### R40 paradigm corrections (1 → cumulative 39 closed levers; R32:21 + R33:5 + R34:4 + R35:1 + R36:2 + R37:2 + R38:2 + R39:1 + R40:1)
- HB-N+WARPS_N=2 V2-RRR REFUTED via doubly-pre-refuted compound (Leg A + Leg B both empirically closed)
- Last remaining HB-* path on existing tile geometry CLOSED. tile-area-conservation 3rd-confirmed
- 8B Up V2-RRR BOUNDARY-LOCK (new R41+ classification)

### R41+ priority (rebuilt from R40)
1. 【high】Survey NEW perf opportunities outside HB-* / tile-rotation (paradigms now CLOSED across 9 cycles): V2-RCR HB shrink (mirror of HB shrink B1 success on V2-CRR); BK=64 vs 128 K-direction blocking; alternative shared-mem layout for B operand; inter-WG L2 coordination (speculative)
2. 【medium】Decode-shape coverage (M=1, 32, 128 entirely unmapped — R28-R40 focused on prefill M=4096)
3. 【methodology】All R29-R39 + R40 NEW: BOUNDARY-LOCK classification mandatory; formalize GPU rotation OR relax drift gate to 3.5%

## R39 cycle 完结 (2026-04-18) ★★ STRICT-PROMOTE + CRITICAL VALIDATION — 1 STRICT PROMOTE (8B Gate/Up V2-RRR R36→R39 cleared after 3 cycles SHIP-LITE) + ★★ R38 wrap fix `66ef02d8` STRICT-VALIDATED in production (8B-KV +24.57% min Welch t +72.6 through dispatcher path) + MXFP8_DISPATCH_TRACE=1 runtime infra deployed (R39+ mandatory) + ALL 9 production predicates AUDITED (0 new wire-in bugs) + 1 paradigm closure (HB-N shrink REFUTED on V2-RRR via tile-config inspection — early abort, ~1 GPU-hr saved) + 1 NEW methodology gap CLOSED (PY_MODULE_NAME collision in r37_paired_bench_2so.py — defensive assert added)

R39 派 4 dev (A HB-N shrink V2-RRR wide-N, B 4-GPU STRICT promote 8B Gate/Up V2-RRR N_PAIRS=15, C MXFP8_DISPATCH_TRACE=1 runtime tracepoint, D audit ALL 9 production predicates + R38 wrap fix independent revalidate) + Reviewer (9th-cycle baseline + ★★ CRITICAL revalidate R38 wrap fix dispatcher-path + 2 STRICT RECONFIRMs). **1 STRICT PROMOTE + ★★ R38 fix DOUBLE-CONFIRMED + 2 STRICT RECONFIRM + 0 new wire-in bugs + 0 NEW SHIPs (HB-N V2-RRR REFUTED).**

### ★★ Headline result — Reviewer + Dev D R38 wrap fix `66ef02d8` STRICT-VALIDATED in production
- Reviewer dispatcher-path 8B-KV: GPU3 +26.00% t=+76.2; GPU6 +25.46% t=+72.6; GPU7 +24.57% t=+84.3 (min STRICT PASS by +19.57 / +62.6)
- Dispatcher trace fires `[HB shrink Stage B1 ACTIVE for N=1024 tall-thin (M=4096, N=1024, K=4096) — R38 wire-in fix]` (predicate `(g.k == 8192 || g.k == 4096)` routes correctly)
- Dev D independent CONFIRM: GPU5 +27.15% t=+68.0 / GPU0 +27.71% t=+84.3
- R37 Dev B's +24.96% SHIP claim NOW REPRODUCIBLE through production .so

### ★★ Dev B STRICT PROMOTE `85fd9418` (8B Gate/Up V2-RRR cleared after 3 cycles SHIP-LITE)
- 4-GPU @ N_PAIRS=20 + PREHEAT=120: GPU2 +5.17% t=+21.45; GPU3 +5.13% t=+12.07; GPU6 +5.66% t=+17.62; GPU7 +5.46% t=+12.89
- min Δ% +5.131% / min Welch t +12.066 (STRICT PASS by +0.131 / +2.066)
- Cross-cycle Δ% rock-solid: +5.025 (R36) / +6.55 (R37) / +5.05 (R38) / +5.13 (R39)
- R38 Dev D statistical-power hypothesis CONFIRMED on 2nd V2-RRR predicate. Welch t scaled cleanly: R36 t=6.28 (n=10) → R39 t=12.07 (n=40)
- N_PAIRS=15 was insufficient; N_PAIRS=20 + PREHEAT=120 is new R40+ default for boundary-band V2-RRR (+5-6% Δ)

### ★ Dev C `021d5a13` MXFP8_DISPATCH_TRACE=1 runtime infrastructure (R39+ mandatory)
- 17 distinct tracepoints catalogued (8 V2-RCR/V2-RRR advisories + 2 hbshrink B1 routes + hbnshrink + default exact_8wave + rect-fast + V1 fallbacks)
- R38 wrap fix smoke-test PASS at deploy: production .so on 8B-KV emits `CRR-V2-HBSHRINK-B1-8B-KV (R38 wrap fix 66ef02d8)`
- Zero default-build overhead: stderr is 0 bytes when env unset; nm-gate OVERALL: PASS unchanged
- **R40+ MANDATORY**: all Phase 2 verification runs with `MXFP8_DISPATCH_TRACE=1 ... 2> trace.err` and greps for expected predicate; empty match = CRITICAL bug abort

### ★ Dev D `010d64e0` predicate audit (0 new wire-in bugs)
- All 9 production predicates AUDITED: 8 V2-RRR/V2-RCR advisories + 2 HB shrink B1 kernel-swaps. ALL 9 fire correctly through dispatcher
- R38 Reviewer's 8B-KV catch was the only wire-in bug in R28-R38
- NEW methodology bug found (NOT wire-in): `r37_paired_bench_2so.py` with same `PY_MODULE_NAME` for both .so → Python aliases both module objects to SAME .so → measured Δ% collapses to noise (+0.10% t<1)
- **Defensive guard added in R39 wrap**: `assert mod_A is not mod_B` + `assert so_a != so_b` in `r37_paired_bench_2so.py`. Future paired-bench with module-name collision aborts with clear error

### Dev A `7724ab6f` (NO SHIP / NEGATIVE — REFUTED via tile-config inspection, ~1 GPU-hr saved)
- V2-RRR `rrr_mxfp8_exact_8wave_fastpath.inc:18-19`: BLK=256, BK=128, WARPS_M=2, WARPS_N=4 — N-partition geometry byte-identical to V2-CRR
- Per-warp accumulator (RBM=64 × RBN=32) byte-identical to V2-CRR
- Only difference is A-fetch layout (row-shared B + transpose vs col-shared B); N-partition geometry identical → R38 Dev A's V2-CRR refute mechanism (bandwidth-bound, WG-grid-doubling, non-load-bearing accumulator drop) transfers directly
- R34 Dev B's V2-RRR > V2-CRR +5-6% makes RRR *more* bandwidth-bound, not less — even worse outlook
- Per R39 task spec time-box rule: abort early before scaffold/build. R40+ followup: speculative WARPS_N=2 compound bet (only path that could relieve bandwidth pressure)

### Reviewer Phase 1 `fd484ce8` (9th-cycle baseline)
- R39 baseline 768.68 TF (median-of-4 GPU2/3/6/7); 9-cycle drift envelope 766-787 TF, 2.67% spread (under 3% threshold)
- GPU2 high outlier +2.87% above other-3; conservative ship-claim baseline = min-of-4 = 763.74 TF
- 3-gate orchestrate clean PASS on all 4 GPUs (no G2a retries needed)

### Reviewer Phase 3 STRICT RECONFIRMs
- R37 Dev A HB shrink B1 70B-KV: GPU3 +29.74% / GPU6 +28.38% — **6/6 cross-cycle measurements at +28-30%** ★★
- R38 Dev C 8B-Down V2-RRR: GPU0 +7.65% / GPU4 +7.65% — Δ% rock-solid R36 +6.66 / R37 +7.25 / R38 +7.42 / R39 +7.65 (Welch t below 10 is expected 2-GPU N=5 statistical-power limit, not signal weakness)

### R39 paradigm corrections (1 → cumulative 38 closed levers; R32:21 + R33:5 + R34:4 + R35:1 + R36:2 + R37:2 + R38:2 + R39:1)
- HB-N shrink REFUTED on V2-RRR wide-N via tile-config inspection (no benches needed, V2-CRR refute transfers byte-for-byte). Only remaining HB-N path is WARPS_N=2 compound bet (deferred to R40+)

### R40+ priority (rebuilt from R39)
1. 【high】4-GPU STRICT-promote remaining R28-R38 SHIP-LITE cells (8B Up V2-RRR mirror; V2-RCR predicates) using N_PAIRS=20+PREHEAT=120 + R38 orchestrate
2. 【medium】HB-N shrink V2-RRR with WARPS_N=2 compound — only structural lever that could relieve bandwidth pressure for HB-N geometry
3. 【medium】V2-RCR advisory audit through MXFP8_DISPATCH_TRACE — verify each of 8 advisories fires for target shape under autotune-default invocation
4. 【methodology】All R29-R38 + R39 NEW: MXFP8_DISPATCH_TRACE=1 mandatory in Phase 2; paired-bench .so MUST have distinct -DPY_MODULE_NAME (defensive assert enforced); STRICT-promote V2-RRR boundary-band defaults to N_PAIRS=20 + PREHEAT=120

## R38 cycle 完结 (2026-04-18) ★★ STRICT-PROMOTE + CRITICAL BUGFIX — 1 STRICT PROMOTE (8B-Down V2-RRR R36→R38) + 1 CRITICAL FIX (R37 Dev B 8B-KV production wire-in was dead) + 2 paradigm closures (HB-N shrink REFUTED on V2-CRR wide-N) + R38 NEW G1' fallback orchestrate + R38 NEW nm-based dead-code gate + 1 NEW methodology gap (predicate-fanout requires dispatcher-path verification)

R38 派 4 dev (A HB-N shrink prototype 8B GU + 70B GU, B HB-N shrink fan-out 70B GU + 8B-Down, C 4-GPU STRICT promote 8B-Down V2-RRR N_PAIRS=15, D R37 NEW orchestrate G1' + nm-gate scripts) + Reviewer (8th-cycle baseline + Phase 2 RECONFIRM R37 SHIPs). **1 STRICT PROMOTE + 1 STRICT RECONFIRM + 0 NEW SHIPs (HB-N REFUTED) + 1 CRITICAL FAIL caught + FIXED in cycle wrap.**

### ★★ Headline result — Dev C STRICT PROMOTE `e466e582` (8B-Down V2-RRR cleared after 2 cycles SHIP-LITE)
- 4-GPU @ N_PAIRS=15: GPU2/3/6/7 min Δ% +7.42% / min Welch t +18.56 (was +5.87 R36 / +6.36 R37 due to N=5 statistical-power cap)
- Cross-cycle Δ% rock-solid: +6.66% (R36) / +7.25% (R37) / +7.42% (R38)
- Closes "K≥14336 K-noise floor" hypothesis — was statistical-power, not performance

### ★★★ Critical fix — `66ef02d8` R37 Dev B 8B-KV production wire-in was DEAD
- R38 Reviewer caught it: dispatcher hard-coded `g.k == 8192` while Dev B only extended `.inc` allow-list to K=4096
- Production .so unfixed: 8B-KV -0.17% / +0.10% (kernel never reached)
- PATCHED-wire test: 8B-KV +26.66% (matches Dev B's +24.96% claim)
- Fix: predicate now `(g.k == 8192 || g.k == 4096)` per Dev B's allow-list

### Dev A `4335cb57` + Dev B `fd673f15` (NO SHIP — HB-N shrink REFUTED on V2-CRR wide-N, 3-shape independent confirmation)
- Dev A 8B GU -43.18% / 70B GU -44.85% (PIPE=0 bit-exact)
- Dev B 70B GU -26.72% / 8B-Down -47.89% (PIPE=0 bit-exact) / 8B-Down PIPE=1 -30.16% (correctness FAIL)
- VGPR savings confirmed (-82, even bigger than HB-M's -74) but no pipeline to feed
- Mechanism: V2-CRR is bandwidth-saturated on wide-N (WARPS_N=4 + RBN=32). Halving N coverage doubles WG grid + barriers + scale-fetch overhead WITHOUT relieving real pressure
- **CLOSES** "BLK_N reduction is symmetric counterpart to BLK_M reduction" on V2-CRR. Scaffolds kept dead-code-gated for R39+ V2-RRR HB-N exploration
- Conflict resolution: both Devs independently created `crr_mxfp8_exact_8wave_hbnshrink_fastpath.inc`; cherry-pick used `-X ours` to keep Dev A's scaffold

### Dev D `87107f54` (methodology hardening, ships orchestrate v2 + nm-gate as R38+ standard)
- **r38_orchestrate.sh**: G1+G1'+G2a+G2b. G1' fallback (`bench_mhz ≥ 2200 AND median > 500 TF`) accepts samples failing pre-bench but passing post-bench. Validated on idle/synthetic-reject/hostile-contention.
- **r38_nm_gate.sh**: drop-in for md5 hygiene (md5 unreliable in env). Symbol catalog: hbshrink (0 default / 4 PROD), hbn/subrbm/4wave/_8wave_rect (0 default), v2 dispatchers (1+ runtime-gated)
- Verification re-bench R37 Dev A B1 on GPU4: +28.82% Welch t +21.78 → SHIP equivalence CONFIRMED

### Reviewer Phase 1+2 `df649778` (8-cycle baseline + 1/2 STRICT + 1/2 critical bug)
- **R38 baseline 777.62 TF** (between R36 775.15 and R37 786.64)
- 8-cycle drift envelope: 766-787 TF, 2.67% spread
- HB shrink B1 70B-KV STRICT RECONFIRM (5/5 cross-cycle: R36 +28%, R37 +30%, R37 4-GPU +25-31%, R38D +29%, R38R +28%)
- HB shrink B1 8B-KV CRITICAL FAIL via wire-in bug → fixed in 66ef02d8

### R38 paradigm corrections (2 → cumulative 37 closed levers; R32:21 + R33:5 + R34:4 + R35:1 + R36:2 + R37:2 + R38:2)
- HB-N shrink REFUTED on V2-CRR wide-N (3-shape independent confirmation)
- R36 Dev B 6th V2-RRR predicate STRICT-promoted; "K-noise floor" hypothesis closed

### R39+ priority (rebuilt from R38)
1. 【critical】Re-validate R38 wrap fix `66ef02d8` 8B-KV production via 4-GPU triangulation through dispatcher path (R38 Reviewer Phase 2 protocol)
2. 【high】MXFP8_DISPATCH_TRACE=1 runtime tracepoint at every dispatch branch — nm-gate cannot catch "compiled-in but unreached" predicates
3. 【medium】HB-N shrink on V2-RRR wide-N — V2-RRR is autotune-preferred for 8B GU per R34 Dev B, may not be bandwidth-saturated
4. 【medium】4-GPU STRICT-promote remaining LITEs (R34/R35 Dev A 8B Gate via Dev D's r38_orchestrate.sh + N_PAIRS=15)
5. 【methodology】All R29-R37 + R38 NEW: orchestrate v2 mandatory, nm-gate mandatory, predicate-fanout SHIPs MUST exercise dispatcher (not .inc-direct kernel calls)

## R37 cycle 完结 (2026-04-18) ★★★ DOMAIN-LOCK — HB shrink B1 production-wired (70B-KV +30.39%) + 8B-KV STRICT SHIP fan-out (+24.96%) + 3/3 R36 STRICT RECONFIRM + B1 domain rule established (N=1024 tall-thin only) + B3 PIPE=3 hybrid CLOSED structural + first V2-CRR cell to clear FP8 reference (107.4%)

R37 派 4 dev (A wire HB shrink B1 production predicate 70B-KV + 4-GPU triangulation, B HB shrink B1 fan-out to 8B-KV + 8B Gate/Up rect, C HB shrink B1 fan-out to 70B Gate/Up + investigate B3 PIPE=3, D median-of-medians retroactive + 4-GPU triangulation R36 Dev B 8B-Down) + Reviewer (7th-cycle baseline + Phase 2 RECONFIRM HB shrink B1 + 2 V2-RCR predicates). **2 STRICT SHIPs + 1 SHIP-LITE + 2 NO-SHIP** (NO-SHIPs lock the B1 domain rule).

### ★★★ Headline result — Dev A HB shrink B1 PRODUCTION wire-in `ab8a80f7` (+30.39% on 70B-KV, FIRST V2-CRR cell to clear FP8)
- Production median-of-medians **1010.70 TF** vs FP8 per-tensor 941.11 TF = **107.4%** (first V2-CRR cell ever to clear FP8 reference)
- 4-GPU triangulation (GPU0/1/3/6, BABA paired N=6): min Δ% +25.13%, min Welch t +13.05 → STRICT PASS
- Default 8192³ build byte-identical (md5 `33b17d2c…` pre/post-edit)
- R36 Dev A HB shrink B1 PROMOTED SHIP → STRICT under R37 4-GPU triangulation
- **R37 NEW methodology**: R36 G1 sclk-post-preheat over-aggressive under 4-concurrent-agent host (50% sample loss); recovered using `bench_mhz ≥ 2200 AND median > 500 TF` post-hoc filter (Dev A r37a recipe)

### ★★ Dev B HB shrink B1 fan-out STRICT SHIP `46a42d18` (8B-KV +24.96%)
- 8B-KV (4096×1024×4096): STRICT SHIP — Δ% +24.96%, min Welch t +32.1
- 8B Gate/Up (4096×14336×4096): NO SHIP -17.26%, predicate refuses (production .so falls through with +0.04%)
- Production allow-list now `{(4096,1024,8192), (4096,1024,4096)}`
- **Domain rule established**: HB shrink B1 wins on N=1024 tall-thin rect regardless of K; loses on wide-N

### Dev C `2cdb0ae4` (NO SHIP both, 2 paradigm closures)
- 70B Gate/Up B1: -28.30% (BLK_M=128 grid-doubling overhead on wide-N — same failure mode as 8192³ -25%)
- B3v2 PIPE=3 hybrid (`MXFP8_CRR_HBSHRINK_PIPELINE=4`): -17.74% (worse than B3 -6.27%) — **B1 LDS interleave fundamentally incompatible with single-buffered SB pipelining; "B4 SB+interleave hybrid" pursuit CLOSED**
- Methodology note: per-build md5 unreliable in env (consecutive identical hipcc → different md5); recommend `nm -D | grep <feature>` gate for compile-flag dead-code verification

### Dev D `5fd5596d` (methodology + 8B-Down 4-GPU)
- Median-of-medians retroactive R31-R36: 7/24 baselines bimodal; max per-cycle shift 0.23%; **NO PRIOR SHIPS INVALIDATED**
- 8B-Down V2-RRR 4-GPU: SHIP-LITE CONFIRM (min Δ% +7.25% > STRICT 5.0 ✓; min Welch t +6.36 < STRICT 10.0 ✗). Need N_PAIRS=10-15 or quieter host for STRICT promotion.

### Reviewer Phase 1+2 `a5f2f3a4` (7th-cycle baseline + 3/3 STRICT RECONFIRMS)
- **R37 baseline 786.64 TF** (+1.48% vs R36 775.15; first "no high outlier" cycle since R33)
- 3/3 STRICT RECONFIRMS independent on GPU1+GPU7: HB shrink B1 70B-KV +28.79%/+27.79%; V2-RCR 8B Q/O +7.94%/+6.75%; V2-RCR 70B Q/O +9.22%/+8.90%
- R36 NEW 3-gate orchestrate validated; G1 sometimes too aggressive under high contention (GPU5 EXHAUSTED 12 attempts/3 sessions)

### R37 paradigm corrections (2 → cumulative 35 closed levers; R32:21 + R33:5 + R34:4 + R35:1 + R36:2 + R37:2)
- HB shrink B1 domain rule: N=1024 tall-thin rect ONLY (regardless of K). Wide-N and square out-of-domain.
- B1 LDS interleave + single-buffered PIPE=3 fundamentally incompatible. No further "B4-style" pipelining permutations on hbshrink path.

### R38+ priority (rebuilt from R37)
1. 【high】4-GPU STRICT-promote 8B-Down V2-RRR with N_PAIRS=10-15 on quiet host (currently SHIP-LITE 2 cycles)
2. 【medium】Investigate non-rect alternative: BLK_N reduction (BLK_N=64 + HB_N=32) for wide-N shapes (8B Gate/Up, 70B Gate/Up, 8B-Down) — parallel "HB-N shrink" SKU candidate
3. 【medium】R38 NEW orchestrate: relax G1 to `bench_mhz ≥ 2200 AND median > 500 TF` post-hoc filter for high-contention runs
4. 【medium】R38 NEW dead-code gate: `nm -D | grep <feature_symbol> == 0` instead of md5 (md5 unreliable in env)
5. 【methodology】All R29-R36 rules carry forward + R37 NEW two methodology rules above

## R36 cycle 完结 (2026-04-18) ★★ MAJOR — 4 SHIPs CONFIRMED + R32 PIPE=3 -15% structural ceiling SMASHED + first V2-RCR autotune predicates + 7 predicates/11 LLaMA cells + R35 NEW second sclk gate validated in production + GPU6 outlier hypothesis BROKEN

R36 派 4 dev (A GPU3 HB shrink Stage 2 re-pipelining critical, B GPU1/4 6th V2-RRR predicate 8B-Down, C GPU1/2/7 RCR Q/O verify+wire, D GPU6 methodology + GPU6 outlier root-cause) + Reviewer (GPU0/4/5/6 6th-cycle baseline + Phase 2 with R35 NEW gate). **4 SHIPs CONFIRMED**.

### ★★ Headline result — Dev A HB shrink Stage B1 SHIP `30d298e8` (+28.02% on 70B-KV V2-CRR)
- Pattern: cross-buffer double-buffer (PIPE=1) — read TIC, prefetch into TOC. NOT in-place same-tic prefetch (which raced and produced SNR 5.29 dB).
- VGPR 160 (-74 vs default 234), 0 spill, bit-exact correctness (max abs diff = 0.0), Welch t=+96.4 on 70B-KV.
- Per-stage table: PIPE=0 -12.16% / PIPE=1 +28.02% / PIPE=2 +13.27% / PIPE=3 -6.27% / 8192³ -25.03% (out-of-domain).
- Build hygiene PASS: pre/post default builds byte-identical (md5 `3c060f2536a4d71b41b146eb6c1e6114`).
- **R32 PIPE=3 -15% structural ceiling SMASHED.** First major V2-CRR perf win in many cycles.

### ★ Dev C 2 STRICT SHIPs `2e63b801` — first V2-RCR autotune predicates ever
- 8B Q + 8B O (4096×4096×4096): min Δ% +7.14% / min t +3.47 — STRICT
- 70B Q + 70B O (4096×8192×8192): min Δ% +8.63% / min t +4.08 — STRICT
- Coverage: **7 predicates / 11 LLaMA cells** (was 5/7).
- Verification step revealed NO prior V2-RCR predicates existed.

### Dev B SHIP-LITE `08452e02` — 6th V2-RRR predicate (8B-Down 4096×4096×14336)
- min Δ% +6.66% / min t +5.87 across GPU1/4. STRICT promotion needs 4-GPU triangulation.

### Dev D `3f0bd96c` — methodology + GPU6 debunk
- NEW `r36_reviewer_orchestrate.sh` with 3-gate logic: G1 sclk-post-preheat ≥ 2200 + G2a sclk-post-bench ≥ 2200 + G2b per-run CV ≤ 1%. All three must pass; up to 3 retries.
- G2b is the most informative new gate (caught GPU0 attempt 1 mid-bench drop, GPU5 accidental concurrency).
- **GPU6 outlier hypothesis BROKEN**: RAS counters all 0, sustained sclk identical to GPU4/5. Bimodal {763, 783} sampling bias, NOT aging. No RMA needed.
- Methodology fix: per-cycle median-of-medians (N≥3 BABA replicates per GPU) supersedes single 5-iter median.

### R36 Reviewer Phase 1 — 6-cycle 70B-KV V2-CRR cross-cycle baseline
- R31 766.72 / R32 772.51 / R33 766.17 / R34 767.59 / R35 768.07 / **R36 775.15** (new high)
- High outlier rotation restored: GPU6 dropped to 770.07; GPU0 now +2.54%
- R34+R35 second-gate validation in production: 7 retry events accepted only clean runs

### R36 Reviewer Phase 2 — SHIP CONFIRMS
- c4 R33 Dev C 70B KV: GPU4 +10.51% t=+25.41; GPU5 +10.81% t=+46.45 — STRICT CONFIRM
- c5 R34/R35 Dev A 8B Gate: GPU4 +6.96% t=+6.83; GPU5 +5.05% t=+6.28 — SHIP-LITE CONFIRM

### R36 paradigm corrections (2 — extends to 33 cumulative closed levers)
- "PIPE>1 impossible on V2-CRR" hypothesis CLOSED. The R32 ceiling was VGPR-headroom-bound; HB shrink (BLK_M=128) frees enough VGPR for PIPE=1 cross-buffer DB.
- "GPU6 needs RMA" hypothesis CLOSED. Sampling bias on bimodal distribution; methodology fix is N≥3 BABA replicates per GPU.

### R36 cumulative tally → 33 closed levers (R32: 21 + R33: 5 + R34: 4 + R35: 1 + R36: 2)

### R37+ priority list (rebuilt from R36 results)
1. **【critical / 2-3 day】Wire HB shrink Stage B1 production predicate** in `dispatch_pq_v2<CRR>` near line 5526 — predicate `(M==4096 && N==1024 && K==8192)`. Dev A's predicate must be 70B-KV-only (8192³ is -25% out-of-domain). Build with `-DMXFP8_CRR_BLK_M=128 -DMXFP8_CRR_HBSHRINK_PIPELINE=1`.
2. **【high / 1 day】4-GPU triangulation** of (a) Dev A's HB shrink B1 (1-GPU → 4-GPU), (b) Dev B's 8B-Down (2-GPU → 4-GPU). Promote SHIP-LITE → STRICT.
3. **【high / 2-3 day】Extend HB shrink B1 to other rect shapes** — 8B-KV, 70B Gate/Up, 8B Gate/Up. Tall-rect tiles benefit most.
4. **【medium / 1-2 day】Investigate Dev A's B3 regression** (PIPE=3 hybrid SB+interleave -6.27%) — R37 candidate B4 cross-buffer DB + interleave hybrid.
5. **【medium / 2 day】Apply Dev D's median-of-medians rule** to existing R31-R36 baseline numbers — re-derive cross-cycle table.
6. **【close — paradigm】"PIPE>1 impossible on V2-CRR" CLOSED.** Default BLK_M=256 path retains R32 ceiling — do not re-prototype PIPE>1 on default kernel.
7. **【methodology — R37+ rules, MUST follow】**:
   - All R29-R35 rules carry forward.
   - **R36 NEW (mandatory)**: Reviewer orchestrate must use 3-gate logic (G1 sclk-post-preheat + G2a sclk-post-bench + G2b per-run CV ≤ 1%). All three must pass; up to 3 auto-retries.
   - **R36 NEW (recommended)**: per-cycle median-of-medians (N≥3 BABA replicates per GPU).

## R35 cycle 完结 (2026-04-18) ★ 1 SHIP (5th V2-RRR autotune predicate → 5/7 LLaMA cells) + 1 STRUCTURAL EMPIRICAL CONFIRM (HB shrink ↓-66 VGPR) + 1 NEW CLOSURE (WARPS_M=4) + 4 R36+ fan-out targets + 1 NEW methodology gap (sclk-mid-bench)

R35 派 4 dev (A GPU0 5th V2-RRR autotune predicate for 8B Gate/Up shape, B GPU3/4 HB shrink BLK_M=128 stages A1-A5, C GPU2 WARPS_M=4 stages A1-A4 + closure, D GPU6/7 LLaMA full 14-shape matrix re-bench) + Reviewer (GPU0/4/5/6 5th-cycle baseline + Phase 2 SHIP verifications). **1 STRICT SHIP** (Dev A 5th predicate covers c5 8B Gate + c6 8B Up).

**Major structural empirical confirm**: R35 Dev B HB shrink (BLK_M=128, HB_M=64) skeleton DELIVERED with **VGPR 168 (-66 vs default 234), 0 spill, 104 KB LDS (-34 KB), occ=2** and bit-exact numerics PASS (max_abs_diff=0.0). Bare skeleton perf -12.16% (Welch t=-71.3) **but empirically validates R34 Dev D §3.2 Option 1 hypothesis** that M-coverage halving DOES save accumulator VGPR (opposite of REFUTED sub-RBM operand shrink).

**Major closure**: R35 Dev C WARPS_M=4 saturates VGPR with 7-lane spill — both sub-RBM (R34) and WARPS_M=4 (R35) now CLOSED. This **closes the entire parallel-template paradigm** for V2-CRR (any per-warp tile reorder preserving total tile area cannot relieve the PIPE=3 accumulator-VGPR ceiling). **HB shrink is now the only first-order lever** that empirically reduces accumulator VGPR. R36 critical task: HB shrink Stage 2 re-pipelining.

### R35 Reviewer Phase 1 — 5-cycle 70B-KV V2-CRR cross-cycle baseline
- R31 766.72 / R32 772.51 / R33 766.17 / R34 767.59 / **R35 768.07** TF (median-of-4 GPUs).
- High outlier: GPU6 +2.85% (first 2-cycle repeat — was GPU0→GPU5→none→GPU6→GPU6).
- Build md5 `d34018cd127258362e821f1625821aa1` bit-identical across 5 builds.
- R34 sclk auto-retry rule WORKED (GPU6 attempt 1: 1714 MHz, 89 TF → recovered to 2313 MHz, 789 TF on attempt 2).

### R35 Reviewer Phase 2 — SHIP CONFIRMS
- **c4** R33 Dev C 70B KV (M=4096 N=1024 K=8192): GPU5 RRR vs CRR +10.44% Welch t=+41.26 — CONFIRMED
- **c5** R34 Dev B / R35 Dev A 8B Gate (M=4096 N=14336 K=4096): GPU5 +5.77% t=+3.75 / GPU6 +5.76% t=+9.19 — CONFIRMED at SHIP-LITE

### R35 Reviewer NEW methodology gap (→ R36 rule)
GPU0 attempt 1: sclk-post-preheat = 2277 MHz (passed gate ≥ 2200) BUT mid-bench dropped to 2091/2076 MHz, tflops_median = 364.73 stdev 23.77 (4× noisier). The R34 gate did NOT catch this. **R36 mandatory rule**: add SECOND gate either (a) sclk-post-bench ≥ 2200 MHz OR (b) per-run stdev/mean ≤ 1%.

### R35 Dev results
- **Dev A SHIP `e0eb3686` → cherry-picked `3aa489df`**: 5th V2-RRR autotune predicate at `kernel_mxfp8_layouts.cpp:5582-5656` for shape (M=4096, N=14336, K=4096); covers c5 8B Gate + c6 8B Up. Effective predicate count: **5 wired / 7 LLaMA cells covered**.
- **Dev B `8096429e` → cherry-picked `a719cadc`**: HB shrink NEW `crr_mxfp8_exact_8wave_hbshrink_fastpath.inc` + dispatch wire-in. Macro `MXFP8_CRR_BLK_M==128` default-off. NEEDS R36 Stage 2 re-pipelining to convert -66 VGPR headroom into +ve perf.
- **Dev C `71ef3b44` → cherry-picked `c6b86803`**: NEW `crr_mxfp8_exact_8wave_warpsm4_fastpath.inc` (424 LOC) + `MXFP8_CRR_WARPS_M` macro (default 2; =4 selects W4 path). Structural closure.
- **Dev D `b1860c38` → cherry-picked `4f846add`**: full 14-shape LLaMA matrix re-bench. All 7 R34-wired predicates remain valid. **4 R36+ fan-out candidates surfaced**:
  - 8B-Down +9.51% (V2-RRR predicate at 4096×4096×14336)
  - 8B-Q +4.83% (route to RCR; +7.05% better)
  - 8B-O +4.73% (route to RCR; +5.83% better)
  - 70B-Q +7.72% / 70B-O +6.94% (route to RCR; +8.32%/+8.20% better)

### R35 paradigm corrections (1 — extends to 31 cumulative closed levers)
- **WARPS_M=4 structural non-recovery** for V2-CRR PIPE=3: rearranging warp×tile partition while preserving total tile area cannot reduce per-warp VGPR. Combined with R34 sub-RBM REFUTATION → **closes the entire parallel-template paradigm** for V2-CRR.

### R35 cumulative tally → 31 closed levers (R32: 21 + R33: 5 + R34: 4 + R35: 1)

### R36+ priority list (rebuilt from R35 results)
1. **【critical / 5-7 day】R35 Dev B HB shrink Stage 2 re-pipelining**. Bare skeleton -12.16% empirically confirms VGPR -66 headroom. Use the headroom to: (a) PIPE=3 LDS pipeline (target R32 PIPE=3 -15% structural ceiling), (b) restore B-tile cycle-2 prefetch, (c) potentially restore sub-tile interleaving. Convert -66 VGPR into +ve perf vs default (BLK_M=256 baseline).
2. **【high / 1-2 day】R36 Dev R36-A: V2-RRR predicate for 8B-Down** (M=4096, N=4096, K=14336) — largest uncovered RRR-vs-CRR gap (+9.51%). 6th V2-RRR autotune predicate.
3. **【high / 2-3 day】R36 Dev R36-B: RCR predicate verification + wire-in for square Q/O cells**. Verify R28-R32 RCR autotune predicates fire at R35 base for 8B-Q/8B-O/70B-Q/70B-O; if missing, wire them. Largest unexploited layout gap (+5.83% to +8.32%).
4. **【medium / 1-2 day】Methodology hardening**: R35 Reviewer's recommended second gate.
5. **【medium / 1-2 day】GPU6 2-cycle-repeat outlier investigation**: ECC counter check, preheat-duty audit.
6. **【close — paradigm】Parallel-template paradigm fully CLOSED**. Do not re-prototype sub-RBM, WARPS_M, or any per-warp tile reorder for V2-CRR PIPE=3.
7. **【methodology — R36+ rules, MUST follow】**: All R29-R34 rules carry forward. **R35 NEW**: orchestrate must add second gate (sclk-post-bench OR per-run stdev/mean) to catch GPU0-attempt-1 class of mid-bench contention regressions.

## R34 cycle 完结 (2026-04-18) ★ 3 SHIPs CONFIRMED (V2-RRR autotune fan-out 5 cells + 8B Gate STRICT + 8B Up SHIP-LITE) + 4 closures + rect-V2 CRR perf paradigm fully CLOSED + sub-RBM-as-VGPR-savings hypothesis REFUTED

R34 派 4 dev (A GPU0/4 5-cell V2-RRR autotune fan-out, B GPU1/2/5/6 4-GPU triangulation R33 SHIP-LITE 8B Gate/Up, C GPU2/3 rect-V2 CRR Stage A2 Path 2, D GPU3 sub-RBM Stage 2) + Reviewer (GPU0/4/5/6 4th-cycle baseline + Phase 2 verification). **3 SHIPs CONFIRMED** by 4-GPU triangulation. **Two paradigm closures**: rect-V2 CRR perf fully CLOSED (R34 Dev C Path 2 -9.7%, the LAST open variant; combined with R33 RCR closure → rect-V2 perf CLOSED in both layouts). sub-RBM-as-VGPR-savings REFUTED (Dev D Stage 2b spills 476 VGPR — operand shrink trades -16 for +128 net accumulator). Path past V2-CRR PIPE=3 ceiling pivots from sub-RBM to **HB shrink (BLK_M=128) or WARPS_M=4**.

### R34 Reviewer Phase 1 — 4-GPU baseline (4th cross-cycle data point, cherry-picked `69fa78e5`)

GPUs 4/5/6/0; all 12 (re)builds bit-identical md5 = `09cb3e014dad03d190eb7d737a2c1a9f`:
- GPU0: 766.63 → **765.12** (-0.20%)
- GPU4: 767.97 → **766.30** (-0.22%)
- GPU5: 764.81 → **768.88** (+0.53%)
- GPU6: 765.71 → **787.13** (**+2.80%, NEW HIGH OUTLIER**)
- **median-of-4: 766.17 → 767.59** (+0.19%); cross-cycle 4-cycle peak-to-peak 0.83%.

**R33 stochastic per-(GPU × cycle) hypothesis STRENGTHENED by N=4**: 3-of-4 cycles have a high-state outlier in +2.36-2.71% band, rotating GPU0 → GPU5 → none → GPU6. R33 sub-rule (min-of-GPUs when one >+1.5% above others) remains in force.

### R34 Reviewer Phase 2 — Per-SHIP verification (orthogonal-GPU triangulation)

| Dev | Claim | R34rev triangulation | Verdict |
|---|---|---|---|
| Dev A (5-cell V2-RRR autotune fan-out @ `kernel_mxfp8_layouts.cpp:5526-5594`) | 4 NEW predicates (c1/c2 share + c4 + c8); c0 70B Down preserved | Reviewer GPU5/6: c4 70B KV +9.81% min Δ; c1 +6.82%. Advisory once-per-shape; 0 false-positives on c3/c5/c7/default. | **CONFIRM SHIP** |
| Dev B (R33 SHIP-LITE 8B Gate/Up promotion) | c5 STRICT (min Δ=+5.025%, t=10.13); c6 LITE (min Δ=+5.340%, t=6.95) | Reviewer GPU0 +5.23% / GPU4 +6.61%. **5-GPU total** for c5 — clears STRICT. | **CONFIRM SHIP** (c5 STRICT, c6 LITE — recommend land same predicate) |
| Dev C (rect-V2 CRR Stage A2 Path 2) | numerics PASS bit-exact, perf -9.7% | n/a | NO SHIP — paradigm CLOSURE |
| Dev D (sub-RBM Stage 2 scaffolding) | type bridge + skeleton clean; perf forecast -ve (476 VGPR) | n/a | NO SHIP — hypothesis REFUTED |

**R34 net = 2-3 SHIPs CONFIRMED.** Cumulative R28-R34: **11-12 SHIPs**.

### R34 Dev results

- **Dev A ★ SHIP** (cherry-picked `5c33c0c3`): Extended `kernel_mxfp8_layouts.cpp:5526-5594` with 3 NEW shape-conditioned host-side advisories. **4 effective predicates cover 5 LLaMA cells** (c0 70B Down + c1+c2 70B Gate/Up shared + c4 70B KV + c8 8B KV). Per-cell min Δ% across GPU0/GPU4: c0 +12.16%, c1 +6.82%, c2 +7.98%, c4 +10.28%, c8 +7.42%. Advisories fire once-per-shape; 0 false-positives. Welch t mid-cycle thermal throttling artifact (BABA pair-ratios robust). Default md5 differs from R33 (runtime-dead branches) — functional gate verified independently (advisory_count=0 on default 8192³).
- **Dev B ★ SHIP (c5 STRICT) + RECOMMEND LAND (c6 LITE)** (cherry-picked `9e6c453d`): 4-GPU + Reviewer 5-GPU triangulation on R33 SHIP-LITE 8B Gate/Up (4096×14336×4096). c5: min Δ=+5.025% / min t=10.13 → **promotes from R33 LITE → R34 STRICT**. c6 (same shape): min Δ=+5.340% / min t=6.95 → recommend land since single shape predicate covers both. Cross-cell consistency: 0.88 pp diff (within R33 0.79-0.93 pp intra-cell spread). **NEW**: corrected GPU6 attribution — same-node DPM contention from parallel mxfp4 agent, not chassis throttle.
- **Dev C ★ NO SHIP — rect-V2 CRR perf paradigm CLOSED** (cherry-picked `0f3eb64d`): Stage A2 Path 2 (HB_N=64 + K-serialized reads via new `load_col_from_v2_st_half_rect_idx` helper) numerics PASS bit-exact but perf **-9.7% GPU2 / -9.05% GPU3** vs square (Welch t=-43/-36). Path 2 LDS = Path 1 LDS = 104,448 B (occ=2 unchanged) — **LDS NOT the blocker**. VGPR 169 unchanged. Structural ceiling = doubled LDS-read issue rate. After R31 stub, R33 Path 1 (-8.5%), R34 Path 2 (-9.7%) — **all numerically-correct rect-V2 CRR variants lose 8-10%**. Only structural recovery = R32 Path 3 helper-rewrite (high-risk to 4 hot kernels, out-of-scope). **rect-V2 CRR perf fully CLOSED.**
- **Dev D ★ NO SHIP — sub-RBM hypothesis REFUTED** (cherry-picked `d28df43e`): Stage 2a (type bridge) + Stage 2b (kernel body skeleton) DONE — all 8 R33 probe errors cleared. Probe build (`MXFP8_CRR_RBM=32 MXFP8_CRR_SUBRBM_PROBE=1`) now rc=0. 4-stride M-loop per warp (HB/RBM_SUB=4), 16 accumulators. Default build byte-identity PASS. **CRITICAL**: Stage 2b skeleton spills **476 VGPRs** (256 saturated). Operand shrink (-16 VGPR) is more than canceled by doubled accumulator vector count (+128 VGPR). **R32 Dev B/R33 Dev D hypothesis REFUTED.** Type bridge from Stage 2a is reusable for HB shrink or WARPS_M=4 alternatives.

### R34 paradigm corrections (4 — extends to 30 cumulative closed levers)

1. **rect-V2 CRR perf paradigm fully CLOSED** (Dev C): all 3 numerically-correct variants lose 8-10%. Combined with R33 Dev B RCR closure → **rect-V2 perf CLOSED in both layouts**.
2. **sub-RBM-as-VGPR-savings hypothesis REFUTED** (Dev D Stage 2b): operand shrink doesn't reduce VGPR pressure — accumulator vector count doubles. **NEVER prototype "halving RBM frees VGPR for pipelining" again.** Pivot to HB shrink or WARPS_M=4.
3. **Same-node parallel-agent DPM contention** (Reviewer methodology bug, NEW R34): `sclk-post-preheat < 2200 MHz` indicates contention. **R35+ rule**: orchestrate auto-retry up to 3x on this signal.
4. **R33 high-regime stochastic hypothesis STRENGTHENED by N=4** (Reviewer Phase 1): 3-of-4 cycles have one high-state GPU rotating GPU0/5/none/6. R33 sub-rule (min-of-GPUs) remains in force.

### R35+ priority list (rebuilt from R34 results)

1. **【critical / 1 day】Land R34 Dev B 8B Gate/Up autotune predicate** in `dispatch_pq_v2<CRR>` near 5526-5594. Single shape `(M=4096 && N=14336 && K=4096)` covers both c5/c6. Brings total to **5 effective predicates covering 7 LLaMA cells**.
2. **【high / 3-5 day】HB shrink (BLK_M=128) attempt** — pivot from sub-RBM (REFUTED) to per-warp M-coverage shrink. Reuses R34 Dev D Stage 2a type bridge. Goal: cut accumulator vector count 2x → drop accumulator VGPR 128 → 64 → unblock SB pipelining (target R32 PIPE=3 -15% ceiling).
3. **【medium / 3-4 day】WARPS_M=4 attempt** — alternative to HB shrink. Different trade-off (more wave-fill, less per-wave register).
4. **【medium / 1-2 day】LLaMA full matrix re-bench** at R34 head — verify 5 autotune predicates produce expected end-to-end speedups.
5. **【close — paradigm】rect-V2 CRR perf-via-MXFP8_RECT_BLK_N lever** (R34 Dev C + Reviewer concur).
6. **【methodology — R35+ rules, MUST follow】**:
   - All R29-R33 rules carry forward.
   - **R34 NEW**: auto-retry 3x on `sclk-post-preheat < 2200 MHz` (same-node contention).
   - **R34 NEW**: check for parallel-agent same-node activity before attributing throttling to hardware.

### R34 Cherry-pick status

Cherry-picked to feat/mxfp8-only:
- `d28df43e` (Dev D sub-RBM Stage 2 + REFUTATION)
- `0f3eb64d` (Dev C rect-V2 CRR Path 2 NO SHIP)
- `5c33c0c3` (Dev A 5-cell V2-RRR autotune fan-out SHIP)
- `9e6c453d` (Dev B 8B Gate STRICT + Up LITE)
- `69fa78e5` (R34 Reviewer Phase 1+2)

All R34 macros default-off; Dev A's autotune predicates emit runtime-dead branches on default 8192³ (advisory_count=0 verified).

## R33 cycle 完结 (2026-04-18) ★ 6 SHIPs CONFIRMED (V2-RRR autotune wire-in + 4 STRICT V2-RRR pivots + rect-V2 RCR numerics) + 5 closures + rect-V2 paradigm CLOSED (CRR Path 1 -8.5% + RCR -33-36%) + sub-RBM Stage 1 scaffolded

R33 派 4 dev (A GPU0/4 V2-RRR autotune wire-in + rect-V2 CRR Stage A2 Path 1, B GPU1/2 rect-V2 RCR Stage A2 numerics fix, C GPU2/3 per-shape RRR sweep across 8 LLaMA cells, D GPU3 sub-RBM Stage 1 scaffolding) + Reviewer (GPU0/4/5/6 4-GPU baseline + Phase 2 SHIP verification on orthogonal GPUs). **6 SHIPs CONFIRMED** by 4-GPU triangulation. **Major paradigm closure**: rect-V2 fully closed for both CRR (Path 1 numerics PASS but -8.5% perf) and RCR (numerics PASS but -33-36% perf). **V2-RRR layout pivot is THE dominant SHIP direction** with 5 cells covered by 4 effective autotune predicates.

### R33 Reviewer Phase 1 — 4-GPU baseline (3rd cross-cycle data point, cherry-picked `c424131f`)

GPUs 0/4/5/6 with `r33_reviewer_bench5x.py` bit-identical to R32; all 4 builds md5=`b1df8e374fc9b49722b9b85941c6d2e2`:
- GPU0: 768.43 → **766.63** (-0.23%)
- GPU4: 768.37 → **767.97** (-0.05%)
- GPU5: 786.59 → **764.81** (**-2.77%, R32 high-state collapsed**)
- GPU6: 776.60 → **765.71** (-1.40%)
- **median-of-4: 772.51 → 766.17** (-0.82%); **R33 spread 3.16 TF / 0.41% — 6× tighter than R31/R32**.

**KEY FINDING (paradigm refinement)**: R32 hypothesis "high regime rotates one-per-cycle" **FALSIFIED by R33** (N=3 cross-cycle). R33 has zero high-state GPUs. Refined: high regime is **stochastic per-(GPU × cycle) firmware/DPM-residency state**, not deterministic round-robin. **R33 sub-rule**: when one GPU >+1.5% above others, use **min-of-GPUs** (not mean) for SHIP gate.

**Canonical baseline R34+**: 70B KV V2-CRR live = **768 TFLOPS** (mean of R31/R32/R33 medians-of-4 = 768.47). SHIP gate ≥840 TFLOPS for 0.92 ratio unchanged.

### R33 Reviewer Phase 2 — Per-SHIP independent verification (4-GPU triangulation per claim)

| Dev | Claim | R33rev triangulation (orthogonal GPUs) | Verdict |
|---|---|---|---|
| Dev A Task 1 (V2-RRR autotune wire-in @ 70B Down 4096×8192×28672) | host-side advisory at `kernel_mxfp8_layouts.cpp:5522-5532` (NOT transparent reroute) | Dev GPU0/4 +11.54%/+12.20% — R33rev GPU5/6 +12.10%/+12.31% (t=16-76). Combined w/ R32 GPU2/4/6 = 7 distinct GPU runs. | **CONFIRM SHIP** |
| Dev A Task 2 (rect-V2 CRR Stage A2 Path 1 — square LDS HB=128 + halve N) | numerics PASS, perf -8.5% | n/a | NO SHIP — paradigm closure |
| Dev B (rect-V2 RCR Stage A2 numerics @ 4096³) | bit-identical correct output (PC=1→2, voff `<<6→<<7`, soff `<<8→<<9`) | Dev GPU1/2 — R33rev GPU4/5: snr 49.61, det PASS, **C[0,:8] bit-identical on all 4 GPUs**. Perf -33-36% confirmed. | **CONFIRM SHIP** (numerics) + closure (perf) |
| Dev C c1 70B Gate 4096×28672×8192 RRR vs CRR | +7.50%/+7.56% (Dev) | R33rev GPU5/6: +7.99%/+7.18% (t=29/43). Min Δ% = +7.18%. | **CONFIRM SHIP** |
| Dev C c2 70B Up 4096×28672×8192 RRR vs CRR | +7.81%/+7.20% (Dev) | R33rev GPU5/6: +7.37%/+7.55% (t=13/11). Min Δ% = +7.20%. | **CONFIRM SHIP** |
| Dev C c4 70B KV 4096×1024×8192 RRR vs CRR | +10.68%/+10.77% (Dev) | R33rev GPU5/6: +10.24%/+10.83% (t=36/32). Min Δ% = +10.24%. | **CONFIRM SHIP** |
| Dev C c8 8B KV 4096×1024×4096 RRR vs CRR | +8.30%/+8.36% (Dev) | R33rev GPU5/6: +8.13%/+8.63% (t=34/15). Min Δ% = +8.13%. | **CONFIRM SHIP** |
| Dev C SHIP-LITE c5/c6 (8B Gate/Up 4096×14336×4096) | +5-6.5% (t=5-8) Dev only | deferred to R34 4-GPU triangulation | DEFERRED |
| Dev D sub-RBM Stage 1 scaffolding | no SHIP claim | n/a | n/a |

**R33 net = 6 NEW SHIPs CONFIRMED.** Cumulative R28-R33: **9 SHIPs**.

### R33 Dev results

- **Dev A ★ SHIP (Task 1) + paradigm closure (Task 2)** (cherry-picked `e37cc1d8`): Task 1 wired V2-RRR autotune entry at `kernel_mxfp8_layouts.cpp:5522-5532` for shape (M=4096,N=8192,K=28672). One-time stderr advisory; NOT transparent reroute (CRR's A=(K,M) vs RRR's A=(M,K) layout incompatibility). Task 2 attempted rect-V2 CRR Stage A2 Path 1 (R32 RECOMMENDED): numerics PASS bit-exact (snr 49.60, det 3/3 @ 4096³) but perf **-8.5%** vs default. **rect-V2 CRR Path 1 closed.** Only remaining open lever: Path 2 (HB=64 K_HALF=0-only) or helper rewrite.
- **Dev B ★ SHIP (numerics) + paradigm closure (perf)** (cherry-picked `9f26c99e`): Identified Dev D R32 Stage A1 architectural bug — B-side scale slab PC=1 (4 bytes/pack) but Dev D issued b64 reads (8 bytes), so adjacent lanes' reads OVERLAPPED. Fix: PC=1→2, voff `<<6→<<7`, soff `<<8→<<9`, slab_bytes 32→64. **Stage A2c numerics bit-exact at 4096³.** Stage A2d perf NO SHIP: rect 1378-1488 TF vs square 2280-2330 TF → **-33 to -36%**. Root cause: rect lacks PIPELINE_SCALE/KPAIR_LOOP/PHASE_U16_CACHE/REMAP_ONCE. **Paradigm closure for rect-V2 RCR perf-via-MXFP8_RECT_BLK_N=64 lever.**
- **Dev C ★ 4 STRICT SHIPs + 2 SHIP-LITE + 3 closures** (cherry-picked `802ba52e`): Per-shape V2-RRR vs V2-CRR sweep across 8 LLaMA cells with BABA + 30s preheat on GPU2/3. Pattern: **V2-RRR > V2-CRR for all "non-square" cells (N ≠ M)**; V2-RCR > V2-RRR for square Q/O cells (N = M). Pattern holds independent of K (K=4096 c8 + K=8192 c1/c2/c4) — **R32 K-magnitude-specificity hypothesis FALSIFIED**. STRICT SHIPs: c1 70B Gate +7.50/56% (t=30/54), c2 70B Up +7.81/7.20% (t=32/44), c4 70B KV +10.68/10.77% (t=30/24), c8 8B KV +8.30/8.36% (t=21/22).
- **Dev D — sub-RBM Stage 1 scaffolding (no SHIP)** (cherry-picked `4fc887f1`): NEW 315-LOC parallel template `crr_mxfp8_exact_8wave_subrbm_fastpath.inc`. Added `MXFP8_CRR_RBM` macro (default 64); `-DMXFP8_CRR_RBM=32` compiles cleanly with stub. 14 static_asserts audited (4 classes); probe gate `MXFP8_CRR_SUBRBM_PROBE` documents 8-error Stage 2 chain. Stage 2 deferred to R34+ (5-7 days estimated).

### R33 paradigm corrections (5 — extends to 26 cumulative closed levers)

1. **rect-V2 paradigm CLOSED for both CRR and RCR** (Dev A Task 2 + Dev B): Path 1 -8.5%; rect-V2 RCR -33-36%. R28-R32 scaffolding direction **exhausted as a SHIP path**.
2. **R32 K-magnitude-specificity hypothesis FALSIFIED** (Dev C): RRR > CRR is shape-dependent (N ≠ M), not K-dependent.
3. **Square Q/O cells favor V2-RCR** (Dev C c3/c7): RCR > RRR for N = M cells. **NEVER hypothesize "one layout dominates everywhere" again.**
4. **R32 high-regime rotation hypothesis FALSIFIED** (Reviewer Phase 1): R33 has zero high-state GPUs. Refined to stochastic per-(GPU × cycle) firmware state. **R33 sub-rule**: min-of-GPUs not mean when one GPU >+1.5% above others.
5. **Dev D R32 scaffolding bug masked by denormal output** (Dev B finding): "Stage A1 SHIP with denormal output" can mask architectural bugs. **Methodology flag**: future cycles must explicitly call out residual numerics-bug-risk.

### R34+ priority list (rebuilt from R33 results)

1. **【critical / 1-2 day】Land 5-cell V2-RRR autotune fan-out in `dispatch_pq_v2<CRR>`**: R32 Dev C C4 (R33 Dev A wired) + R33 Dev C c1/c2 (same shape ⇒ 1 predicate) + c4 + c8. **4 effective predicates cover 5 cells.** Pattern: stderr advisory only (NOT transparent reroute). Run LLaMA matrix to verify no neighboring-shape regression.
2. **【medium / 1-2 day】4-GPU triangulation on R33 Dev C SHIP-LITE cells** (8B Gate/Up 4096×14336×4096): Δ ≥ +5% on Dev's 2 GPUs but t < 10 — needs ≥4-GPU bench to clear t > 10 STRICT gate.
3. **【medium / 1-2 day】rect-V2 CRR Stage A2 Path 2 attempt** (rect LDS HB=64 with K_HALF=0-only helper): only remaining rect-V2 CRR open lever before structural exhaustion.
4. **【high / 5-7 day】sub-RBM Stage 2** (type bridge + kernel body translation): Dev D Stage 1 scaffolded; breaks 8 static_asserts. Only path past V2-CRR PIPE=3 -15% LDS-pipelining ceiling.
5. **【close — paradigm】rect-V2 RCR perf-via-MXFP8_RECT_BLK_N=64 lever** (R33 Dev B + Reviewer concur): rect lacks PIPELINE_SCALE/KPAIR_LOOP/PHASE_U16_CACHE/REMAP_ONCE.
6. **【methodology — R34+ rules, MUST follow】**:
   - All R29/R31/R32 rules carry forward.
   - **R33 NEW**: when cross-GPU SHIP triangulation has any GPU >+1.5% above others, use `min(measured_TFLOPS_across_GPUs)` for SHIP gate, not mean.
   - **R33 NEW**: "Stage A1 SHIP with denormal output" must include explicit residual-numerics-bug-risk callout.

### R33 Cherry-pick status

Cherry-picked to feat/mxfp8-only:
- `4fc887f1` (Dev D sub-RBM Stage 1 — 10 files)
- `e37cc1d8` (Dev A V2-RRR wire-in + rect-V2 CRR Path 1 — 13 files)
- `802ba52e` (Dev C per-shape RRR sweep — 39 files)
- `9f26c99e` (Dev B rect-V2 RCR Stage A2 numerics — 21 files)
- `c424131f` (R33 Reviewer Phase 1+2 — 36 files)

All R33 macros default-off; default builds byte-identical to head. Side-branch commits preserved on r33-{a,b,c,d,rev} for R34+ continuation.

## R32 cycle 完结 (2026-04-18) ★ 2 SHIPs CONFIRMED (V2-RRR @ 70B Down +12.14% + rect-V2 RCR Stage A1) + 7 closures + R31 GPU0-discount rule DEPRECATED

R32 派 4 dev (A GPU0 rect-V2 CRR Stage A2, B GPU1 V2-CRR LDS SB pipelining, C GPU2 K-large MLP shapes, D GPU3 rect-V2 RCR Stage A1) + Reviewer (GPU0/4/5/6 4-GPU baseline reverify w/ sclk fix + Phase 2 SHIP verification). **2 SHIPs CONFIRMED by 3-GPU triangulation** + 7 structural closures + critical methodology correction (R31 GPU0 high-outlier hypothesis FALSIFIED).

### R32 Reviewer Phase 1 — 4-GPU baseline reverify with sclk-fix (commit `390c7b54` r32-rev → cherry-picked `78400cb6`)

GPUs 4/5/6/0 with `rocm-smi -d $PHYS_GPU` patch (R31 paradigm correction now applied):
- GPU0: 786.38 → **768.43** (-2.28%)
- GPU4: 767.31 → **768.37** (+0.14%)
- GPU5: 764.28 → **786.59** (+2.92%) — new high outlier
- GPU6: 766.14 → **776.60** (+1.37%)
- median-of-4: 766.72 → **772.51** (+0.76%)

**KEY FINDING (paradigm correction)**: R31's "GPU0 high-regime outlier" hypothesis FALSIFIED. **High regime rotates per (GPU × cycle), not per-GPU.** R32+ rule: cross-GPU triangulation on ≥2 GPUs; deprecate the GPU0 2.6% discount rule.

### R32 Reviewer Phase 2 — Per-SHIP independent verification

| Dev | Claim | Reviewer triangulation | Verdict |
|---|---|---|---|
| Dev D (rect-V2 RCR Stage A1) | scaffolded + GPU-fault-clean | Default md5 unchanged. Rect VGPR=137 / 0 spill / occ=2 (matches dev). GPU-fault test PASS on GPU4 + GPU5. | **CONFIRM SHIP** |
| Dev C (V2-RRR @ 70B Down 4096×8192×28672) | +12.14% / Welch t=+49.24 | GPU2 dev: +12.14% / t=+49.24. GPU6: +12.57% / t=+76.22. GPU4: +12.21% / t=+28.16. | **CONFIRM SHIP (3-GPU)** |
| Dev A (rect-V2 CRR Stage A2c) | NO SHIP — numerics blocked | n/a | n/a |
| Dev B (V2-CRR LDS SB pipelining) | NO SHIP — best -15-18% vs DB | n/a | n/a |

### R32 Dev results

- **Dev A (rect-V2 CRR Stage A2 host preshuffle + helper rewrite) NO SHIP** (cherry-picked `935fddcb`): A2a (preshuffle_v2_b_rect host fn) DONE; A2c (correct numerics) FAILED — pass_rate 33.53%, DETERMINISM=False at 70B KV. **Architectural blocker**: `load_col_from_v2_st_half<RT, K_HALF>`'s `k_row` indexes LDS rows = N-direction (not K). With K_HALF=1, `k_row` ∈ [64, 127] → OOB for HB_N=64 rect tile. R31 Stage A1 stub (duplicate K_HALF=0 data) is GPU-fault-safe but numerics-wrong by construction. **Two recovery paths for R33+**: (1) Square LDS HB=128 + halve N-work (drop b1/cB/cD, ~3-4h, RECOMMENDED); (2) Rewrite helper sharding to make K_HALF index K (~6-8h). Kernel files reverted to R31 baseline.
- **Dev B (V2-CRR LDS SB pipelining recovery) NO SHIP — 3 NEW STRUCTURAL CLOSURES** (cherry-picked `6a483549`): `MXFP8_CRR_SB_PIPELINE` macro selector (0/1/2/3) layered on R31's SB. PIPE=1: -56.7% w/ 26-lane spill. PIPE=2: -58.1% w/ 26-lane spill. PIPE=3: -14.9% to -18.5% no spill (best, structural ceiling). **Root cause**: A_col_reg = 32 VGPR/wave; concurrent prefetch triggers 26-lane spill. PIPE=3 avoids spill via single `a` reuse but cannot fully overlap VMEM with MMA chain. **NEVER prototype "concurrent prefetch in V2-CRR SB form"** without first restructuring operand-tile geometry.
- **Dev C (K-large MLP shapes investigation) ★ SHIP + 3 NEW CLOSURES** (cherry-picked `cc274c0d`): C1 V2-CRR cp sweep @ 70B Down NEUTRAL. C2 K=28672 vs K=8192 IDENTICAL kernel resources (234 VGPR / 139264 LDS / 0 spill — K-iter pressure hypothesis falsified at source). C3 V2-RCR cp=2/3 @ 8B Down REGRESSION (-3 to -5%). **C4 V2-RRR vs V2-CRR @ 70B Down 4096×8192×28672 = +12.14% (2511.66 → 2816.55 TFLOPS) Welch t=+49.24** ★. Bonus: V2-RRR vs V2-RCR @ 8B Down NEUTRAL (-0.88%, RRR advantage shape-specific). **SHIP recommendation**: per-shape autotune for V2-RRR @ (4096, 8192, 28672) only. Triangulated 3-GPU (+12.21/+12.57/+12.14%, all t > 28).
- **Dev D (rect-V2 RCR Stage A1) ★ INCREMENTAL SHIP** (cherry-picked `ba255e74`): New 610-line `rcr_mxfp8_exact_8wave_rect_fastpath.inc` mirrors Dev A's R31 CRR pattern. Stage A1a PASS (default md5 unchanged). Stage A1b PASS (rect rc=0; GPU3 4096³ V2-RCR ~4ms `STAGE_A1b_RESULT: NO_FAULT`). Resources: VGPR 137 vs 246 (-44%), LDS 98304 vs 131072 (-25%), occ=2 unchanged. **NOTE**: RCR rect did NOT need K_HALF stub — RCR shared tiles use `st_16x128_s` (not `v2_s`); generic `rcr_exact_load_st_to_rt` handles K=0..127 in one load → **smaller blast radius than CRR for Stage A2**. Stage A1c deferred to R33+.

### R32 paradigm corrections (4 — extends R27/R28/R29/R30/R31 closure list to 25 total)

1. **R31 GPU0 2.6% discount rule DEPRECATED** (Reviewer Phase 1): high regime rotates per-cycle. R32+ rule: cross-GPU triangulation on ≥2 GPUs; do NOT apply per-GPU discount.
2. **V2-CRR LDS SB pipelining structural ceiling at PIPE=3 / -15% gap** (Dev B): A_col_reg=32 VGPR/wave forces choice between spill (PIPE=1/2) or LDS-drain serialization (PIPE=3). Sub-RBM tile rewrite is the only path past this.
3. **K-iter LDS/VGPR pressure hypothesis falsified at source for V2-CRR** (Dev C): K=28672 vs K=8192 → identical kernel resources (K-loop is runtime-dimensional, not template-dimensional). NEVER hypothesize "K-loop length affects kernel resources" again.
4. **rect-V2 CRR Stage A1 stub is numerics-wrong by construction** (Dev A): `k_row` indexes N not K; R31 stub is GPU-fault-safe but reads wrong slab data. NEVER ship Stage A1 stub as production.

### R32 corollaries (consolidated lever-closure tally → 25 total)

R27-R32 cumulative paradigm-correction count: **25 closed levers** (adds: V2-CRR SB PIPE=1, PIPE=2, PIPE=3, V2-CRR cp @ 70B Down, V2-RCR cp @ 8B Down, K-iter resource hypothesis, R31 GPU0 discount rule, rect-V2 CRR Stage A1 stub).

### R33+ priority list (rebuilt from R32 results)

1. **【critical / 1-2 day】Wire V2-RRR autotune dispatch for 70B Down (M=4096, N=8192, K=28672)**: Dev C SHIP triangulated by Reviewer on 3 GPUs at +12% (Welch t > 28). Add per-shape selector. Quick win.
2. **【critical / 3-4 day】rect-V2 CRR Stage A2 Path 1 (square LDS HB=128 + halve N-work)**: Dev A's RECOMMENDED recovery from architectural blocker. Drop b1/cB/cD, reuse existing helper. Targets 70B KV V2-CRR ratio 0.84 → ≥0.92 (need >840 TFLOPS).
3. **【critical / 3-4 day】rect-V2 RCR Stage A2 (correct numerics)**: Dev D Stage A1 SHIPPED. RCR has SMALLER blast radius than CRR (no K_HALF stub needed). Targets 4096³ V2-RCR ratio 0.92 → ≥0.95 via 4× tile count.
4. **【medium / 2-3 day】Sub-RBM operand-tile rewrite for V2-CRR SB**: only path past PIPE=3 -15% ceiling. RBM=64 → RBM=32 halves A_col_reg from 32 → 16 VGPR/wave. Multi-day, breaks 8 static_asserts. Defer.
5. **【medium / 1-2 day】Per-shape RRR exploration**: Dev C confirmed RRR is shape-specific. Sweep RRR vs CRR/RCR on remaining 70B cells (Q/O, Gate, Up) and 8B Gate.
6. **【methodology — R33+ rules, MUST follow】**:
   - All harnesses: `rm -f tk_mxfp8_layouts*.so` + log per-build md5 (R29 Dev C) + `rocm-smi -d $PHYS_GPU` not `-d 0` (R31 Reviewer).
   - All `MXFP8_*_PERSISTENT_GRID` macros: build-time assert grid >= total_tiles (R31 Dev D).
   - All in-process A/B: BABA pattern + 30s preheat (R31 Dev D).
   - **R32 update**: SHIP claim normalization = cross-GPU triangulation on ≥2 GPUs; per-GPU discount rule DEPRECATED.
   - **R32 update**: Absolute md5 are reproducer-environment-dependent (LLVM `-Rpass-analysis` remarks include `__FILE__` paths). Intra-environment pre↔post comparison still valid; cross-environment not.
7. **【closed】**: 25 levers per cumulative tally above. Do not re-prototype any of them.

## R31 cycle 完结 (2026-04-18) ★ 1 INCREMENTAL SHIP (Stage A1 scaffolding) + 4 STRUCTURAL CLOSURES + 4-GPU live baseline rebased

R31 派 4 dev (A GPU0 rect-V2 Stage A1, B GPU1 V2-CRR LDS single-buffer, C GPU2 V2-RCR PIPELINE_SCALE, D GPU3 V2-RCR persistent-CU) + Reviewer (GPU0/4/5/6 4-GPU triangulation for 70B KV V2-CRR). **1 incremental SHIP (Dev A Stage A1a+A1b: rect-V2 CRR fastpath kernel scaffolded, GPU-fault-clean, default build byte-identical)** + 4 structural closures + new live baseline 766.72 TFLOPS for 70B KV V2-CRR (median of 4 GPUs).

### R31 Reviewer 4-GPU baseline triangulation (commit `92e3fbc2` r31-rev → cherry-picked `18c75159`)

GPUs 0/4/5/6 with identical 5x preheat + per-build md5 (all 4 builds bit-identical md5=`095e12e2f7e9300e657b593338749341`):
- GPU0 = 786.38 TFLOPS (high outlier, +2.6%)
- GPU4 = 767.31 / GPU5 = 764.28 / GPU6 = 766.14 (low cluster)
- **median of 4 = 766.72 TFLOPS** (canonical baseline for R32+)

R31 cross-GPU spread = 22.1 TFLOPS = 2.89% (vs within-run stdev 0.27-0.60%). Refined R30 verdict: **persistent per-GPU drift, not a regression**. R27 GPU4=787.96 reclassified as outlier-high.

**SHIP-claim normalization (R32+ rule)**: any single-GPU SHIP claim for 70B KV V2-CRR must (a) be measured on a non-GPU0 box, or (b) discount GPU0 reading by 2.6% before applying +1% improvement gate. To lift this cell to ratio ≥0.92 = need >840 TFLOPS (FP8 baseline ~913 × 0.92).

### R31 Reviewer methodology bug (paradigm correction → R32+)

`rocm-smi -d 0` always reads physical GPU0 regardless of `ROCR_VISIBLE_DEVICES` remapping. **R32+ harnesses must `rocm-smi -d $PHYS_GPU` (kernel-driver-visible index, set outside rocr mapping)**. Does NOT invalidate R29-R31 TFLOPS measurements — only sclk verification.

### R31 Dev results

- **Dev A (rect-V2 BLK_M=256/N=128 fastpath Stage A1) ★ INCREMENTAL SHIP** (cherry-picked `7212e5e6`): New 637-line `crr_mxfp8_exact_8wave_rect_fastpath.inc` with parallel template + rect helpers + host dispatcher. Stage A1a PASS (default md5=`79c2816c54a00cf0680d32a36af25c98` byte-identical to head; rect build `-DMXFP8_RECT_BLK_N=64` compiles rc=0). Stage A1b PASS (rect kernel runs on GPU0 70B KV ~2ms with no fault — `STAGE_A1b_RESULT: NO_FAULT`). Stage A1c (correct numerics) deferred to Stage A2 host preshuffle. **Bonus**: rect kernel = 148 VGPR (-37%) / 104448 LDS B/block (-25%) vs square 234 VGPR / 139264 LDS B; both still occupancy=2 (LDS-bound). 22 files / 2399 insertions consolidated to feat/mxfp8-only (R28D + R29A + R30A + R31A scaffolding chain).
- **Dev B (V2-CRR LDS single-buffer As+Bs[1][2]) NO SHIP — STRUCTURAL CLOSURE** (cherry-picked `0d56fbfc` + `dffba365`): Single-buffered both → exact-prediction LDS reduction 139264→**69632 B/block (-50%)**, correctness PASS. LDS halved fits 2 blocks/CU within 163840 cap — but **catastrophic -30% perf regression** on both 8192³ + 70B Gate due to loss of cross-K-iter prefetch pipelining. Macro default-off in tree; R32+ exploration of pipelining-recovery variants.
- **Dev C (V2-RCR PIPELINE_SCALE second-buffer) NO SHIP — STRUCTURAL CLOSURE** (cherry-picked `10032d0c` + `a44d8e89`): (1) The originally-named lever does not exist — V2-RCR scales follow same VMEM→VGPR→MFMA-direct as V2-CRR (SASS confirms 0 `ds_write` in V2-RCR kernel). (2) VGPR-prefetch pivot catastrophically regresses — 246 VGPR / 0 spill → 256 VGPR / **312 VGPR spill / 596 B scratch/lane**; 4096³ collapses to 187 TFLOPS (-92%); 8192³ collapses to 225 TFLOPS (-93%). V2-RCR K-loop is at structural register-pressure ceiling.
- **Dev D (V2-RCR persistent-CU dispatch Approach 1) NO SHIP — STRUCTURAL CLOSURE + CRITICAL CORRECTNESS TRAP** (cherry-picked `cee9c3ae`): Mathematically demonstrated for 4096³ V2-RCR (`total_tiles=256` < `slots=608`) that no persistent-CU scheme can synthesize work that doesn't exist in the (br, bc) tile grid. **CRITICAL TRAP**: `MXFP8_RCR_V2_PERSISTENT_GRID < total_tiles` silently drops tiles via early-exit prologue → SNR collapses to 1.53 dB while reporting ~3× baseline TFLOPS. **Sclk-drift artifact**: single-process sequential build-then-bench reported +8.6% (Welch t=3.43) was pure sclk-drift; paired in-process A/B with 30s preheat (BABA pattern) eliminates drift.

### R31 paradigm corrections (4 — extends R27/R28/R29/R30 closure list to 18 total)

1. **V2-CRR LDS single-buffer naive form is closed** (Dev B): hits LDS reduction target but loses -30% to pipelining loss. Future SB attempts need explicit pipelining-recovery (split global_load early-issue + late-wait, async-copy).
2. **V2-RCR scale path has zero LDS round-trip — same as V2-CRR** (Dev C): SASS-confirmed 0 `ds_write` in V2-RCR kernel. **NEVER prototype "PIPELINE_SCALE second-buffer in LDS for V2-RCR" again.** VGPR-prefetch pivot also closed.
3. **Persistent-CU dispatch cannot help shapes with `total_tiles ≤ num_CUs`** (Dev D): mathematically proven for 4096³ V2-RCR. Inner-loop persistence with WORK-TILE LARGER THAN BLK is structurally infeasible (4× LDS breaks 163840 B/CU cap).
4. **Rect-V2 CRR fastpath foundation lands without GPU fault** (Dev A): first concrete rect-V2 incremental SHIP after 4 cycles of side-branch scaffolding. Rect kernel still 2 blocks/CU (LDS-bound). R32+ Stage A2: host preshuffle + K_HALF=1 helper rewrite.

### R31 corollaries (consolidated lever-closure tally → 18 total)

R27-R31 cumulative paradigm-correction count: **18 closed levers** (s_setprio CRR, s_setprio RCR, sched_barrier mask RCR, cachepolicy gate broadening, cachepolicy RCR, scale LDS double-buffer, split-K-along-K, V1 vs V2 size-gating, square BLK=128 rewrite, LDS bank conflicts CRR, occupancy=3 via VGPR (LDS-binding), buffer_load_dword_lds for V2 scales, H7 B-tile reorder hoist-above-cB RCR, rect ceiling at 2× grid, V2-CRR LDS single-buffer naive, V2-RCR PIPELINE_SCALE LDS+VGPR, persistent-CU when total_tiles ≤ num_CUs, rect-V2 occupancy ≤ 2 blocks/CU even with -25% LDS).

### R32+ priority list (rebuilt from R31 closures)

1. **【critical / 1-2 day】Rect-V2 CRR Stage A2 (host preshuffle + K_HALF=1 helper rewrite)**: continue Dev A's `crr_mxfp8_exact_8wave_rect_fastpath.inc` to enable correct numerics. Add `preshuffle_v2_b_rect` host fn + rewrite `load_col_from_v2_st_half` to handle K_HALF=1 case. Once correct, perf-bench rect vs square at 70B KV — primary SHIP target.
2. **【high / 2-3 day】LDS single-buffer with pipelining recovery for V2-CRR**: rewrite Dev B's naive SB form with split global_load early-issue + late-wait, or async-copy variants. Currently the only path to occ=3 (LDS reduction proven feasible at -50%; only blocker is pipelining-loss recovery).
3. **【medium / 1-2 day】K-large MLP shapes investigation (8B/70B Down)**: 8B Down at 0.9394 + 70B Down at 0.8459 untouched by R28-R31. Map cp_value vs K-extent.
4. **【medium / 2-3 day】Rect BLK_M=256/BLK_N=128 for V2-RCR (Approach 2)**: requires adapting V2-RCR to rect; multi-day. ONLY structural path for 4096³ V2-RCR gap (BLK=128 path resurrection or split-K with atomic).
5. **【methodology — R32+ rules, MUST follow】**:
   - All harnesses: `rm -f tk_mxfp8_layouts*.so` + log per-build md5 (R29 Dev C) + `rocm-smi -d $PHYS_GPU` not `-d 0` (R31 Reviewer).
   - All `MXFP8_*_PERSISTENT_GRID` style macros: build-time assert `grid >= total_tiles` (R31 Dev D).
   - All in-process A/B benches: BABA pattern + 30s preheat to eliminate sclk-drift artifacts (R31 Dev D).
   - SHIP claim normalization for 70B KV V2-CRR: discount GPU0 by 2.6% or use non-GPU0 box (R31 Reviewer).
6. **【closed】**: 18 levers per cumulative tally above. Do not re-prototype any of them.

## R30 cycle 完结 (2026-04-18) ★ 0 SHIP + 4 STRUCTURAL CLOSURES + R28 SHIP健康 + R29 cells resolved (0/3 source-driven)

R30 派 4 dev (A GPU0 rect-V2 fastpath, B GPU1 VGPR/occ=3, C GPU2 buffer_load_dword_lds audit, D GPU3 H7 B-tile reorder) + Reviewer (GPU4 cross-GPU reverify of R29's 3 negative-t cells).

### R30 Reviewer cross-GPU reverify (commit `0e39e6b1` r30-rev)

GPU5 reverify of R29 GPU4 negative-t cells:

| Cell | R27 GPU4 | R29 GPU4 | R30 GPU5 | Verdict |
|---|---:|---:|---:|---|
| 8b_down_crr | 2779.39 | 2712.16 | **2763.81** | **GPU4-state artifact** (R29 was outlier; t=−0.6 vs R27) |
| 70b_kv_crr | 787.96 | 766.86 | **767.23** | environmental drift, not source-driven (matches R29 within 0.05%) |
| 8b_gate_crr | 2405.79 | 2390.72 | **2382.04** | inconclusive (~1% persistent, not source-driven) |

**0/3 are source-driven regressions**. R28 SHIP healthy. Per-GPU dispersion for 70b_kv_crr (GPU0=791, GPU4-R27=788, GPU5=767, GPU4-R29=767) — R31 should run 4-GPU baseline triangulation to fix the live baseline.

### R30 Dev outcomes

- **Dev A (rect-V2 fastpath) NO SHIP**: re-validated 2.5d Path A estimate (600+ lines hand-tuned, 4 hardcoded static_asserts). Path B already shipped as R29 guard. Refined Path A breakdown with line numbers in `r30a_findings.md` for R31. Structural ceiling: rect at 70B KV upper-bounds at 2× grid = 42% CU occupancy (304 CUs) — 1.5× SHIP target near structural ceiling.
- **Dev B (VGPR reduction → occupancy=3) NO SHIP — STRUCTURAL CLOSURE**: V2-CRR baseline 234 VGPR / 0 spill / occ=2 / LDS=139264 B/block. HW ground truth (hipGetDeviceProperties): LDS per CU=163840 B → two blocks need 278528 B = **1.7× LDS overflow**. **LDS, not VGPR, is the binding occupancy constraint.** Compiler silently ignores `mb=3` hint (kernel binary bit-identical, md5 confirmed). Bench: 8192³ +0.02% (NULL t=+0.06); 70B Gate −0.55% t=−3.26 (SPI launch-allocator overhead).
- **Dev C (buffer_load_dword_lds audit) AUDIT-ONLY**: SASS-level trace of V2-CRR scale path. **Zero LDS round-trip** — scales declared as VGPR arrays, populated by `raw_buffer_load_b128/b64` directly into VGPRs, consumed as VGPR operands of `mfma_scale_f32_16x16x128_f8f6f4`. No `ds_write` to convert. Bonus: TK `G::load` already uses `raw_buffer_load_lds` for tile fills (only LDS-bound traffic in V2-CRR) — lever already maxed. **Critical for Dev B: scale-pack VGPRs total ~6 of 232 (2.6%); even eliminating all scale VGPRs cannot help reach occupancy=3.**
- **Dev D (H7 B-tile reorder for V2-RCR) NO SHIP — STRUCTURAL CLOSURE**: 4 reorder variants. v1/v3/v4 break correctness (SNR 7-24 dB, det FAIL) due to LDS lifetime constraint — `Bs[tic][1]` is read by `rcr_exact_load_st_to_rt(b1,...)` at pre-cB; next-iter VMEM `G::load(Bs[tic][1])` issued ABOVE that point races against still-draining LDS reads. Only v2 (defer-both-to-pre-cD) is correctness-safe and is NULL (Welch t=−0.247). **Baseline ordering already at maximum-latency-hiding allowed by LDS lifetime.**

### R30 paradigm corrections (4)

1. **Rectangular V2-CRR fastpath structural ceiling** (Dev A): rect at 70B KV upper-bounds at 2× grid = 42% CU occupancy. 1.5× SHIP target near ceiling. Closing 70B KV to ≥0.92 likely needs rect + another lever (split-K with atomic, or BLK_N=64 second scaffolding).
2. **V2-CRR is LDS-bound, not VGPR-bound for occupancy** (Dev B): `GEMM_MIN_BLOCKS_PER_CU > 2` is CLOSED. NEVER prototype "occ=3 via VGPR reduction" without first dropping LDS footprint below 81920 B (single-buffer As/Bs[2][2]→[1][2] or smaller BLK).
3. **`buffer_load_dword_lds` is N/A for V2-CRR scales** (Dev C): V2 paradigm is scale-direct-to-VGPR with zero LDS round-trip. NEVER prototype "buffer_load_dword_lds for V2 scales". Re-confirms R27 paradigm correction #1 with empirical SASS evidence.
4. **H7 B-tile reorder structurally exhausted for V2-RCR** (Dev D): LDS lifetime pins B-tile loads to ≥pre-cB / ≥pre-cD. NEVER prototype "B-tile reorder hoist-above-cB for V2-RCR" — guaranteed correctness failure.

### Cumulative lever-closure tally (R27-R30)

**14 closed levers** — search space narrowing meaningfully:
1. s_setprio for V2-CRR (R28)
2. s_setprio for V2-RCR (R29)
3. sched_barrier mask relax for V2-RCR (R29)
4. cachepolicy gate broadening beyond R28 boundary (R29)
5. cachepolicy for V2-RCR (R27 + R29 confirm)
6. Scale LDS double-buffer / SCALE_LDS REPLACE (R23/R27)
7. Split-K-along-K (R27)
8. V1 vs V2 runtime size-gating (R26)
9. Square BLK=128 rewrite (R27 — rect not square)
10. LDS bank conflicts for V2-CRR (R29)
11. Occupancy=3 via VGPR reduction (R30 — LDS-binding)
12. `buffer_load_dword_lds` for V2 scales (R30)
13. H7 B-tile reorder hoist-above-cB for V2-RCR (R30)
14. Rect ceiling at 2× grid for 70B KV (R30 structural)

### R31+ priority list (rebuilt from R30 closures)

1. **【critical / 2-3 day】Rectangular BLK_M=256/BLK_N=128 V2-CRR fastpath kernel**: still #1, still hardest. Adjusted target: rect alone caps at ~1.5× on 70B KV. To hit ≥0.92 ratio, may need rect + another lever.
2. **【high / 2-3 day】LDS footprint reduction for V2-CRR**: ONLY remaining path to occupancy=3. Drop 139264 B/block below 81920 B by single-buffering As/Bs[2][2]→[1][2]. Targets 8192³ V2-CRR -8.9% gap.
3. **【high】4-GPU baseline triangulation for 70B KV V2-CRR** (R30 Reviewer rec): per-GPU dispersion observed; fix the live baseline.
4. **【medium / 1-2 day】Dispatch-geometry change for 4096³ V2-RCR**: H7 closed; GRID under-occupancy ceiling (16% CUs idle: 256 blocks vs 304 CUs) binds. Persistent-CU scheduling or BLK=128 resurrection.
5. **【medium / 1-2 day】PIPELINE_SCALE second-buffer for V2-RCR**: R28 Dev D scaffolding on r28-d. Constrained by Dev B's LDS-binding finding for CRR — RCR may have similar headroom.
6. **【methodology】All R31+ orchestrate scripts must `rm -f tk_mxfp8_layouts*.so` + log per-build md5; cross-GPU reverify on flagged cells**.
7. **【closed】**: 14 levers per cumulative tally. Do not re-prototype.

### R30 Cherry-pick status

Cherry-picked to feat/mxfp8-only:
- `r30_reviewer_crossgpu_reverify.json` + `r30_reviewer_findings.md` + `r30_reviewer_crossgpu_orchestrate.sh` + `r30_reviewer_crossgpu_aggregate.py`
- `r30a_findings.md` + `r30a_orchestrate.sh` (Dev A refined Path A breakdown)
- `r30b_findings.md` (Dev B LDS-binding paradigm correction with HW measurements)
- `r30c_findings.md` + `r30c_sass_inventory.log` (Dev C SASS data-flow trace)
- `r30d_findings.md` + `r30d_bench.py` + `r30d_orchestrate.sh` (Dev D H7 closure)

NOT cherry-picked: r30-a R28D+R29A scaffolding (no rect-V2 kernel), r30-d `MXFP8_RCR_V2_BLOAD_REORDER` macro (default-off, enabled variants corrupt/NULL).

Side-branch commits preserved: `7702f000` (r30-a), `2587a01c` (r30-b), `1849bc4e` (r30-c), `11576192` (r30-d), `0e39e6b1` (r30-rev).

## R29 cycle 完结 (2026-04-18) ★ 0 SHIP + R28 SHIP confirmed (Welch t=+10.20) + 5 paradigm corrections (3 levers permanently closed)

R29 派 4 dev (A GPU0 rect-V2 BLK, B GPU1 cp autotune sweep, C GPU2 V2-RCR sched/setprio audit, D GPU3 LDS bank conflict audit) + Reviewer (GPU4 reverify). **Outcome**: 0 dev SHIP, but R28 cachepolicy auto-select SHIP confirmed in production by Reviewer Welch t=+10.20 (+1.99% absolute MXFP8 perf on 70B Gate, matches R28 Dev A's +2.7% to within bench-to-bench noise).

### R29 Reviewer 10-cell reverify summary (commit `73dd4444` r29-rev → cherry-pick `r29_reviewer_*`)

GPU4 5x preheat. Same 10 cells as R27 baseline matrix.

| Cell | R29 ratio | R27 ratio | MXFP8 Welch t | Verdict |
|---|---:|---:|---:|---|
| 8k_rcr | 0.9095 | 0.9324 | -1.04 | regressed (FP8 drift up; MXFP8 unchanged) |
| 8k_rrr | 0.9086 | 0.9270 | +1.87 | regressed (FP8 drift up; MXFP8 +0.27%) |
| 8k_crr | 0.9017 | 0.9266 | +1.95 | regressed (FP8 drift up; MXFP8 +0.37%) |
| 4k_rcr | 0.9218 | 0.9229 | +0.52 | unchanged |
| 8b_gate_crr | 0.8986 | 0.9216 | -4.11 | **MXFP8 regression** (-0.63%) ⚠ re-verify |
| 8b_down_crr | 0.9059 | 0.9394 | -3.66 | **MXFP8 regression** (-2.42%) ⚠ re-verify |
| 70b_qo_rcr | 0.9106 | 0.9373 | -2.29 | regressed (mostly FP8 drift) |
| 70b_kv_crr | 0.8151 | 0.8646 | -8.51 | **MXFP8 regression** (-2.68%) ⚠ re-verify largest |
| 70b_gate_crr | 0.8351 | 0.8351 | **+10.20** | **R28 SHIP CONFIRMED** ✅ |
| 70b_down_crr | 0.8274 | 0.8459 | +1.88 | regressed (FP8 drift up; MXFP8 +0.55%) |

**0/10 pass perf gate (unchanged from R27)**. FP8 drifted +1-3% universally vs R27 (likely DPM/GPU-state on GPU4). Three cells show real MXFP8 absolute drops without source change since R28 — flagged for R30 cross-GPU reverify.

### R29 Dev results

- **Dev A (rect-V2 completion) NO SHIP**: cherry-picked r28-d scaffolding clean + added host-side dispatcher guard preventing GPU memory fault on `-DMXFP8_RECT_BLK_N=64` builds. V1 fallback validated (SNR 49.59, det 3/3) but only 2.66 TFLOPS (294× slower than V2 781). No rect-V2 fastpath kernel exists yet (~2.5 day item). 8192³ no-regression check passed -0.42%. Guard NOT cherry-picked to main (dead code without scaffolding); docs+harness cherry-picked.
- **Dev B (per-shape cp autotune sweep) NO SHIP**: 5-shape sweep cp=0 vs cp=2 — every shape NEUTRAL or LOSE. Closest gate-boundary shape N=20480 K=8192 loses 2.5% with t=-27.5. **R28 gate `(N≥28672 AND K≥8192)` is exactly tight on both axes**, not just good enough.
- **Dev C (V2-RCR sched/setprio sweep) NO SHIP**: 6 cells (4096³+8192³ × {p2, p3, sched_barrier_mask}) all |t|≤0.61, |Δ%|≤0.35%. Lever exhausted. Bonus: documented Makefile build-cache bug (`make clean` doesn't remove .so).
- **Dev D (LDS bank conflict audit) AUDIT-ONLY**: static analysis proves V2-CRR has zero LDS bank conflicts. Existing `(nc ^ sw_k)` swizzle places lanes on all 32 banks per cycle. Empirically corroborated by R22-B/R26-A profiling: SQ_LDS_BANK_CONFLICT/SQ_INSTS_LDS < 1%.

### R29 paradigm corrections (5 — extends R27/R28 closure list)

1. **cp=2 win region is sharp not gradient** (Dev B): R28 gate `(N≥28672 AND K≥8192)` is exactly tight on both axes. NEVER prototype "broaden cachepolicy=2 gate beyond R28 boundary" — even N=20480 (71% of N threshold) loses 2.5% catastrophically.
2. **`s_setprio` is CLOSED for V2-RCR too** (Dev C, extends R28 V2-CRR closure): same wave-uniformity root cause. NEVER prototype "elevate MFMA wave priority" again, neither V2-CRR nor V2-RCR.
3. **`sched_barrier` mask relax is BENIGN for V2-RCR** (Dev C): even at mask=0xB Welch t<0.6. Surrounding `s_barrier()` already pins schedule. NEVER prototype "relax sched_barrier mask" for V2-RCR.
4. **Cachepolicy bits exhausted for V2-RCR** (Dev C, extends R27): cp=1 NULL, cp=2/3 catastrophic. NEVER prototype "tune cachepolicy for V2-RCR" — CRR-specific lever only.
5. **V2-CRR has ZERO LDS bank conflicts** (Dev D): `(nc^sw_k)` swizzle is conflict-free. Rejected sub-experiments: row padding, sw_k bit re-mask, sched_barrier(0xff). NEVER prototype "LDS bank conflict reduction" for V2-CRR.

### R29 structural finding (open question for R30)

- **4096³ V2-RCR may be GRID under-occupancy bounded** (Dev C): 256 blocks at BLK=256 / 304 CUs = 0.84 wave-fill, 16% CUs always idle. Per-kernel optimization may be structurally bounded; closing the 0.92 gap likely requires dispatch-geometry changes (block tile reshape, streamk).

### R29 measurement methodology fix

- **Build-cache hygiene** (Dev C bonus): `make clean` does NOT remove `tk_mxfp8_layouts*.so`. Under race conditions r27/r28 orchestrate scripts could load stale .so masking rebuilds. **R30+ rule**: all orchestrate scripts must explicitly `rm -f tk_mxfp8_layouts*.so` + log per-build md5. Reference impl: `r29c_orchestrate.sh`.

### R30+ priority list (rebuilt from R29 closures)

1. **【critical / 2-3 day】Rectangular BLK_M=256/BLK_N=128 V2 fastpath kernel**: still #1. R29 Dev A added defensive guard but no fastpath. Path A (true rect-V2 kernel) or Path B (force V1 fallback for rect mode). Targets 70B KV V2-CRR 0.8151 → ~0.92 from 2× CU utilization. r29a_bench.py + r29a_orchestrate.sh ready.
2. **【high / 1-2 day】VGPR reduction for V2-CRR occupancy=3** (R29 Dev D recommendation): identify spillable VGPR bands (especially scale broadcast registers held across MFMA quadrants), refactor to occ=3. Targets 8192³ V2-CRR -8.9% gap.
3. **【high / 2-3 day】`buffer_load_dword_lds` direct VMEM→LDS path** (R29 Dev D recommendation): bypass VGPR staging on B-side scale loads. Most invasive R30 lever.
4. **【medium / 1 day】B-tile load reorder (H7) for V2-RCR** (R29 Dev C recommendation): unexplored. Targets 4096³ V2-RCR if not GRID-bounded.
5. **【medium / 1-2 day】PIPELINE_SCALE second-buffer for V2-RCR**: R28 Dev D scaffolding on r28-d. Requires LDS budget analysis.
6. **【low / re-verify】R29 Reviewer's 3 negative-t cells**: cross-GPU reverify before treating as real regressions.
7. **【methodology】All R30+ orchestrate scripts must `rm -f tk_mxfp8_layouts*.so` + log per-build md5**.
8. **【closed】**: s_setprio (CRR+RCR), sched_barrier mask relax (RCR), cachepolicy gate broadening, LDS bank conflicts (CRR), cachepolicy for RCR. Do not re-prototype.

### R29 Cherry-pick status

Cherry-picked to feat/mxfp8-only:
- `r29_reviewer_baseline_gpu4.json` + `r29_reviewer_findings.md` + `r29_reviewer_bench5x.py` + `r29_reviewer_aggregate.py` + `r29_reviewer_orchestrate.sh` (R30 reverify ready)
- `r29a_findings.md` + `r29a_bench.py` + `r29a_orchestrate.sh` + `r29a_cell{1,2,3}_*.txt` (Dev A docs+harness)
- `r29b_findings.md` (5-shape lookup table for R30)
- `r29c_findings.md` + `r29c_orchestrate.sh` (sweep harness with build-cache fix)
- `r29d_lds_bank_audit.md` (audit record)

NOT cherry-picked (kernel/scaffolding-only on side branches): r29-a dispatcher guard (dead without scaffolding), r29-b cp=2 build configs, r29-c MMA_SETPRIO/SCHED_BARRIER_MASK macros (no-op at defaults).

Side-branch commits preserved: `b2cc032f` (r29-a), `b82eab24` (r29-b), `5b48ff76` (r29-c), `63fc644b` (r29-d), `73dd4444` (r29-rev).

## R28 cycle 完结 (2026-04-18) ★ 1 SHIP + 2 NO SHIP + 1 SCAFFOLDING + 2 paradigm corrections

R28 派 4 dev parallel + R27 Reviewer baseline matrix on GPU4. Production state: feat/mxfp8-only commit `<head>` ahead of R26 by R27 macro infra (`12785d98`) + R28 auto-select gate (`88d5a7d5`).

### R27 Reviewer baseline matrix (cherry-picked artifacts: `r27_reviewer_baseline_gpu4.json`, `r27_reviewer_verdicts.json`, `r27_reviewer_aggregate.py`)

GPU4, 5x preheat, FP8 W/I=300/300, MXFP8 W/I=200/300. **0/10 LLaMA cells pass ≥0.95 perf gate.** Cross-machine drift vs R26 GPU7 = ±3% (acceptable).

LLaMA shape ratios sorted worst→best:
| Shape | Layout | V2 ratio |
|---|---|---|
| 70B Gate 4k×28672×8k | crr | 0.8351 → **0.8579 post-R28 Dev A** |
| 70B Down 4k×28672×8k | crr | 0.8459 |
| 70B KV   4k×1024×8k | crr | 0.8646 |
| 8B Gate  4k×14336×4k | crr | 0.9216 |
| 4096³ | rcr | 0.9229 |
| 8192³ | crr | 0.9266 |
| 8192³ | rrr | 0.9270 |
| 8192³ | rcr | 0.9324 |
| 70B Q/O  4k×8k×8k | rcr | 0.9373 |
| 8B Down  4k×4k×14336 | crr | 0.9394 |

Largest gaps remain in N-uneven shapes (Gate, Down, KV).

### R29+ priority list (rebuilt from R28 root-causes)

1. **【critical】Rectangular BLK_M=256/BLK_N=128**: build on Dev D r28-d scaffolding. Audit-corrected scope smaller than R27 thought (`offset:1024 = 8*BK` not HB-derived). Stage 3 fault root cause: V2 dispatch falls through to V1 layout w/ V2-preshuffled scales. Targets 70B KV V2-CRR 0.8646 → projected ≥0.92.
2. **【medium】Per-shape autotune table**: extend Dev A gate to (M,N,K,layout) lookup. Possibly +0.5-1% on 8B Down (currently cp=0).
3. **【medium】4096³ V2-RCR audit**: 0.9229, untouched by R28. RCR-side levers (cp catastrophic, but other levers?).
4. **【low】8192³ V2-CRR -8.9%**: skip until rectangular BLK lands.
5. **【closed】s_setprio**: fully exploited (R28 paradigm #1). Never re-prototype.

### R28 SHIP #1 cherry-pick: `88d5a7d5` cachepolicy=2 auto-select gate (Dev A)

### R28 SHIP #1 cherry-pick: `88d5a7d5` cachepolicy=2 auto-select gate (Dev A)

Extends R27's `MXFP8_CRR_V2_SCALE_CACHEPOLICY` macro at `analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_fastpath.inc:9-21` so it auto-defaults to 2 (SLC) when compile-time `N_DIM>=28672 && K_DIM>=8192`. Outside region stays 0 = binary identical to R27. **+2.7% on 70B Gate V2-CRR (4096×28672×8192) without manual flag.** No regression on 70B KV / 8B Gate / 8192³.

5x preheat-then-bench (GPU0, sclk-verified): B vs B0 Welch t=+12.83, Δ=+61.49 TFLOPS / +2.61%. SNR≥49.59 dB, det 3/3 PASS on all 6 cells. Cherry-pick excludes 254-line build logs (kept on r28-a side branch).

R28 dev B in flight: s_setprio sweep on V2-CRR cp=2 baseline.

### R28 Dev B = NO SHIP + paradigm correction (closed lever)

`s_setprio` sweep on cp=2 baseline: best (MFMA=2, VMEM=0) only +7.72 TFLOPS / +0.33% / Welch t=+2.30 (gate ≥+30/+1.2%/t>3.0).

**Paradigm correction (DO NOT re-litigate)**: The 8-wave V2-CRR kernel does NOT have a wave-id branch. All 8 waves run identical interleaved VMEM+MFMA code in lockstep. Pre-R28 code ALREADY uses `__builtin_amdgcn_s_setprio(1)` during MFMA segments and `s_setprio(0)` to restore. Removing it = -21 TFLOPS; keeping MFMA elevated without restore = -74 TFLOPS. Pushing MFMA prio above 1 hits a hard ceiling (+0.3%) because 8 waves hit setprio in lockstep — no relative reordering possible. **NEVER prototype "elevate MFMA wave priority" again — the lever is fully exploited.**

### R28 Dev C = NO SHIP (cp=3 vs cp=2 indistinguishable, t=-0.77)

cp=3 (GLC|SLC) vs cp=2 (SLC) on 70B Gate V2-CRR: Δ=-0.088%, Welch t=-0.77 — fails ship gate. R27's apparent cp=3 lead was within sd noise. **cp=2 confirmed as correct gate value.**

Diagnostic confirms gate region is doing real work — forcing cp=3 outside gate regresses 70B KV -7.5% and 8B Gate -5%. No production change.

### R28 Dev D = SCAFFOLDING ONLY (Stage 1+2 PASS, Stage 3 FAIL — known fall-through)

Built compile-ready rectangular BLK_M=256/BLK_N=128 scaffolding on r28-d (`ddfd2f80`): `MXFP8_RECT_BLK_N` macro + `load_col_from_v2_st_half_rect` template helper + rect gate in `crr_mxfp8_exact_8wave_fastpath.inc:3-15`. Default build binary identical. Rect mode currently faults at runtime (V2 layout/V1 fallback mismatch — same root cause R27 Dev D found).

**Paradigm correction #2 (audit fix)**: `offset:1024` in `ds_read_b64_tr_b8` is **BK-derived (=8*BK)**, NOT HB-derived as R27 Dev D's audit said. HB dependency localizes to `k_row = row_off + K_HALF*64` math + subtile count only. Rectangular BLK_M=256/BLK_N=128 needs **less rewrite than originally estimated** — primarily B-side N-stride parametrization + V2 half-N scale preshuffle. NOT cherry-picked (no positive perf value yet).

## R27 cycle 完结 (2026-04-18) ★ 1 partial production ship (cachepolicy macro infra) + 3 paradigm correction

5 agent (Dev A GPU0, Dev B GPU1, Dev C GPU2, Dev D GPU3, Reviewer GPU4) 攻 R26 后剩下的 V2 levers。

### R27 SHIP cherry-pick: `12785d98` `MXFP8_*_V2_SCALE_CACHEPOLICY` macro infrastructure (Dev A)
- 4 处 `+ MXFP8_{CRR,RRR}_V2_SCALE_CACHEPOLICY` macro hook on V2 `buffer_load_b128/b64` scale loads in `analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_fastpath.inc:320,332` (+ rcr/rrr counterparts)
- Default 0 = binary identical to current behavior (zero risk to production baseline)
- Per-shape opt-in build flag: `make CXXFLAGS="-DMXFP8_CRR_V2_SCALE_CACHEPOLICY=2"` for **70B Gate V2-CRR** (M=4096 N=28672 K=8192) gives **+63 TFLOPS / +2.7% Welch t≈+13** (5x preheat, GPU0, SNR 49.60 dB, det 3/3 PASS)
- **Shape-dependent — DO NOT enable globally**: cp=2 catastrophic on 4096³ V2-RCR (-25×) and regression on 70B KV (-6.9%) / 8B Gate (-4.2%) / 8192³ V2-RCR (-2.3% breaks 3180 floor)
- Auto-select gate is R28 candidate, NOT yet shipped

### R27 paradigm corrections (write-down for future cycles)
1. **V2 has NO scale LDS** (Dev C verified): R23 Dev B's "SCALE_LDS REPLACE" kill misled subsequent rounds. V2 actually loads scales VMEM→VGPR direct (`crr_mxfp8_exact_8wave_fastpath.inc:304-333`, SCALE_VERSION==2 branch). LDS budget V2-CRR = **128 KB** (As/Bs double-buffer only), 32 KB headroom. R26 "131-139 KB" figure was a different kernel variant. **NEVER prototype "scale LDS double-buffer"** — it is a re-litigation of an already-killed lever.
2. **Split-K-along-K is universally DEAD-END** (Dev B verified): -12% on 70B KV. Each K-chunk sub-grid still launches the same 64 blocks → no CU exposure gain, but pay 2× launch + 2× epilogue + lose K=8192 single-pass B-tile cache reuse. Reduction overhead fine (7.7%, well under 30% gate). Real fix needs split-along-M/N (= R26 Dev A's static_assert path) or streamk (multi-day).
3. **BLK rewrite path: rectangular BLK_M=256/N=128, NOT square BLK=128** (Dev D audit): 22 BLK=256 dependency sites, complexity 1=8 / 2=4 / 3=4 / 4=5 / 5=1. Square BLK=128 forces RBM/RBN below MFMA sweet spot. **Rectangular BLK_M=256/BLK_N=128 keeps RBM=64 (no A-side helper rewrite)**, only needs new B-side helper variant + B-only preshuffle. Est 2-3 days. Hardest blocker = `load_col_from_v2_st_half` family at `kernel_mxfp8_layouts.cpp:414-491` (`ds_read_b64_tr_b8 offset:1024` hardcodes K-stride for HB=128). Dev D shipped 25-line `MXFP8_BLK128` macro proof-of-concept on r27-d branch (default unchanged, BLK128 V1-fallback PASS at 2.71 TFLOPS) — useful infrastructure for R28 rectangular path.

### R28+ priority list (rebuilt from R27 root-causes)
1. **【high / 1-2 day】Auto-select cachepolicy=2 by N_DIM/K_DIM compile-time gate**: extend Dev A macro infra with `#if N_DIM>=28672 && K_DIM>=8192 && IS_CRR` auto-select. Direct +2.7% on 70B Gate without manual flag. Need to find K-threshold precisely (8B Gate K=4096 lost; 70B Gate K=8192 won).
2. **【critical / 2-3 day】Rectangular BLK_M=256/BLK_N=128 V2 path**: per Dev D audit, smaller scope than original square BLK=128 plan. Add new B-side `load_col_from_v2_st_quarter` helper + B-only preshuffle. Targets 70B KV V2-CRR 0.84 (4 N-tile → 8 N-tile → ~2× CU utilization). Validate with `MXFP8_BLK128` infrastructure already shipped on r27-d branch (`c019bec8`).
3. **【medium】s_setprio on MFMA-side waves stacked on cp=3 for 70B Gate** (Dev A H2 not reached): could push +2.7% to +4-5%.
4. **【medium】Streamk scheduling for small-N**: alternative to BLK rewrite. One kernel, internal K-reduction across CUs.
5. **【low】8192³ V2-CRR -8.9%**: still no production-impacting fix path. Skip until others addressed.

### R27 worktree status
- `r27-a`: cherry-picked SHIP `12785d98` to main feat/mxfp8-only
- `r27-b`/`r27-c`/`r27-d`: side-branch only, dead-end docs (`59d32212` / `9b2e1384` / `c019bec8`); r27-d's `MXFP8_BLK128` macro is reference for R28 rectangular path

## R26 cycle 完结 (2026-04-18) ★ 3 关键 paradigm correction，0 production fix

5 agent (Reviewer GPU1, Dev A GPU3, Dev B GPU5, Dev C GPU6, Dev D GPU7) 攻 R25 LLaMA worst cells. 0 production commit, 2 infra cherry-pick to main:
- `954ba8b4`: preheat wrappers (`preheat_then_bench.py` + `preheat_fp8_bench.py`) — **新规范：multi-shape baseline 必须用 preheat 避免 cold-DPM artifact**
- `ee5f985a`: corrected reviewer JSON `r26_reverify.json`

### R26 paradigm corrections

1. **70B KV V2-CRR 0.69 是 cold-DPM throttle artifact**: Reviewer (10x clean GPU7) + Dev A (per-shape preheat, GPU3 throttle 1700 MHz) **independently** measure ~0.84 (+21pp recovery). R25 sweep 协议 cold-throttle 命中 V2 (heavier scale traffic) 比 FP8 重。**R25 baseline 70B KV CRR cell INVALIDATED**; 其余 4 cell Reviewer 10x reverify 全 CONFIRM 在 ≤0.7% drift. Real corrected ratio 0.84.
2. **Down-RRR 1.05 V2-WIN 是 FP8-RRR spill bug, not V2 mechanism**: FP8-RRR `RRR_MAIN_UNROLL=4` 在 K≥14336 spill **39 VGPR + 160 B/lane scratch** (K=4096 时 0/0)；V2-RRR 同 K spill 仅 1 VGPR. Spill 量 32→112→224 (K=4096/14336/28672) 线性匹配 FP8 RRR/RCR ratio collapse 1.00→0.87→0.84. V2 没有"赢"。Port 到 V2-CRR/RCR (后者已 0 spill) NOT APPLICABLE. Dev C 试 `RRR_UNROLL=2` 修 spill 但 -41% (latency hide 不足)。R27+: FP8-RRR 修复需深度重构。
3. **V1 vs V2 size-gating 永久 KILLED**: Dev D 7-size sweep (1024-8192) + Dev A small-N 独立验证。V1 vs V2 全 ±2.5% 内 (Dev A: V1 791.64 vs V2 796.09, Welch t=-5.87 V2 微胜)。**V2 default everywhere is correct**.

### R26 root-cause (no fix)

- **70B Gate/Up V2-CRR 0.84 (large N=28672)**: Dev B PMC 找到 dominant cap = TA backpressure + per-wave SQ_INST_LEVEL_VMEM 4.895x scaling vs work-ratio 4.0x = +22% per-wave VMEM-stall queueing. 4 knobs (CRR_STEADY_VMCNT 2/6, MID_BARRIER off, PREFETCH_LGKM=2) 全 NULL 或 correctness FAIL. R27+ 候选: L2 cache-tag pinning of scale arrays (8 MB scale fits 32 MB L2), persistent-scale LDS prefill (DEAD-END at occ=2 / 160 KB cap), per-buffer TCC counter split.
- **小 N (KV) 7-19% gap**: Dev A H1 (BLOCK_N=128) blocked by hardcoded `static_assert(BLK==256)` in `crr_mxfp8_exact_8wave_fastpath.inc:37` + `crr_mxfp8_4wave_fastpath.inc:86` + 多个 V2 preshuffle helper assumption. RBM/RBN/scale-pack/ST_v2 全 hardcoded. Multi-day rewrite.
- **4096³ Q/O V2 0.92**: Dev D amortization curve 揭示 gap is **structural to MXFP8 not V2-specific** (FP8 -21%, V2 -24% from 8192→4096). 修复 = macro-tile prologue 缩短 OR 新增 128×128 tile (与 small-N 同样 multi-day rewrite).

### R27+ 优先级 (基于 R26 root-cause)

1. **【critical / multi-day】Tile shape rewrite**: BLK=256 hardcoded 是 small-N + 4096³ 共同根本约束. 需重写 V2 preshuffle helper 支持 BLK=128, RBM/RBN 参数化, ST_v2 重 shape.
2. **【high / 1-2 day】L2 cache-tag pinning of scale arrays for large-N CRR** (Dev B R27 候选 #1): 消除 V2 在 N=28672 上的 TA/VMEM scaling
3. **【high / FP8-side】Fix FP8-RRR spill at K≥12000** (Dev C finding): production benefit 12-15% on Down-RRR but with MXFP8 work 正交
4. **【medium】Per-buffer TCC split for large-N CRR**: 确认 scale vs A vs B miss attribution
5. **【low】8192³ V2-CRR -8.9%**: production impact 最低 (LLaMA 不跑 8192³)

## R25 LLaMA baseline 结果 (2026-04-18, GPU5, sclk 2320 MHz, commit `6be73744`/`05bf4fef`) ★ paradigm 再修正：8192³ near-parity 不可推广 — **R26 修正：70B KV CRR 0.69 实为 0.84 throttle artifact**

R25 LLaMA shape baseline 跑完（8 build shape，10 logical shape，60 measurement run）。**Pass rate 2/24 cells**（gate = V2 ≥ FP8 × 0.95）。

| Shape (M×N×K) | RCR ratio | RRR ratio | CRR ratio |
|---|---:|---:|---:|
| 8B Q/O 4096³ | 0.92 ❌ | 0.94 ❌ | 0.92 ❌ |
| 8B KV 4096×1024×4096 | 0.84 ❌ | 0.81 ❌ | 0.83 ❌ |
| 8B Gate/Up 4096×14336×4096 | 0.94 ❌ | 0.93 ❌ | 0.94 ❌ |
| 8B Down 4096×4096×14336 | 0.93 ❌ | **1.05 ✅** | 0.94 ❌ |
| 70B Q/O 4096×8192×8192 | 0.94 ❌ | 0.93 ❌ | 0.93 ❌ |
| 70B KV 4096×1024×8192 | 0.84 ❌ | 0.82 ❌ | **0.69 ❌** |
| 70B Gate/Up 4096×28672×8192 | 0.90 ❌ | 0.90 ❌ | 0.84 ❌ |
| 70B Down 4096×8192×28672 | 0.87 ❌ | **1.05 ✅** | 0.84 ❌ |

**结论**：
1. 唯一 V2 winning regime 是 Down-RRR (1.05) — large-K + RRR 共同特征
2. 小 N (KV-attn N=1024) 是最差 regime: V2-CRR @ 70B 跌到 0.69
3. 8192³ "V2 ≈ FP8" 在 4096³ Q/O 退化到 0.92 → V2 vs FP8 gap 与问题规模强相关
4. R25 8192³ CRR optimization 不再是 #1 priority — LLaMA gaps 全部更大

**完整 baseline JSON**: `llama_baseline_r25.json` @ feat/mxfp8-only HEAD `6be73744`
**Driver script**: `analysis/fp8_gemm/mi350x/run_llama_baseline.sh`

## R25 mainline (8192³ CRR optim) 完结：跨会话 task lost，0 commit，priorities reshuffled

R25 8192³ CRR optimization 5 个 agent (Reviewer + Dev A/B/C/D) 因 session compaction task lost；5 个 worktree (/tmp/wt-r25-{a,b,c,d,rev}) head 仍在 c285cb70。R25 LLaMA findings 让 8192³ CRR -8.9% 不再是 priority — LLaMA shape gap 远大。R26 priority list 完全 rebuild around production shape:

1. **【critical】KV-attn N=1024**: 70B KV V2-CRR 0.69 (最差). 假说: 256×256 block tile 在 N=1024 仅 4 N-tile/grid，CU 利用率严重不足. 需考虑 dynamic dispatch (N<2048 fallback / 专用 small-N kernel) 或 tile shape 调整
2. **【high】大 N CRR (70B Gate/Up V2-CRR 0.84)**: MLP gate/up. N=14336 (V2 0.94) → N=28672 (V2 0.84) 退化. 与 V2 LDS budget / scale b128 浪费 关联
3. **【medium】Down-RRR 1.05 win 推广**: 唯一 V2 wins. mechanism: large K 让 V2 scale-load 摊销充分？ port 到 Down-RCR/CRR
4. **【medium】4096³ Q/O 0.92**: real "square" production. 与 8192³ 0.99 形成 reference, 找 amortization breakpoint

## Baseline (R24 2026-04-18 GPU0 fresh reverify ★ paradigm correction：V2 实际 ≈ FP8 parity；R23 cycle-level 数字 invalidated as PMC-mode + cold-throttle artifacts) **— 注意：仅适用 8192³，LLaMA shapes 见上**

| 版本 | TFLOPS | SNR | 备注 |
| --- | ---: | --- | --- |
| **FP8 RCR (长期 target)** | **3232** (R24) | 49.61 PASS | R24 5x median GPU0, std 9.48 |
| **★ MXFP8 RCR PRESHUFFLE V2 (default on, R21 SHIP)** | **3214** (R24 fresh) | 49.60 PASS | **gap -17 / -0.5% ★ near-parity** |
| MXFP8 RCR V1 (RUNTIME=0 fallback) | 3022.43 | 49.60 PASS | unchanged |
| **★ MXFP8 RRR PRESHUFFLE V2 (default on, R22-A SHIP, commit dabeffa0)** | **3156** (R24 fresh) | 49.59 PASS | **gap -76 / -2.4%** |
| MXFP8 RRR V1 (RUNTIME=0 fallback) | ~2878 | 49.59 PASS | V1 had 19 spills/80B scratch; V2 collapsed to 0 |
| **★ MXFP8 CRR PRESHUFFLE V2 (default on, R22-B SHIP, commit 9a0d0624)** | **2943** (R24 fresh) | 49.60 PASS | **gap -289 / -8.9% ← ONLY meaningful gap remaining** |
| MXFP8 CRR V1 (RUNTIME=0 fallback) | 2711.78 | 49.60 PASS | V1 already PIPELINE_SCALE single-shot 4×b32; CRR baseline more MFMA-bound |

**R22 综合**：V2 preshuffle paradigm 完整覆盖三 layout（RCR + RRR + CRR）全 SHIPPED default-on。V1→V2 收益排序 RRR (5.54%) > RCR (1.84%) > CRR (0.76%)，与 V1 baseline 的 spills + VMEM-issue rate 排序一致。所有 V1 路径保留为 RUNTIME=0 fallback。

**R24 综合 (paradigm correction)**：R23 cycle-level diagnostic 数字全部 invalidated as PMC-mode + GPU3 cold-throttle artifacts。R24 fresh GPU0 measurement (sclk 2353 MHz verified, warmup=100, per-iter sync) 显示 V2-RCR -0.5% / V2-RRR -2.4% / V2-CRR -8.9%，远好于 R23 报告的 -5.5/-6.7/-13.6%。R24 派 3 dev (SQC dcache cut / RRR TCC re-tile / SPI occupancy) 全部 DEAD-END，0 production commit, 2 side-branch commits (5cf85e58 + 76848ea9, NOT cherry-picked)。**Real status**: V2-RCR essentially at FP8 parity; V2-RRR -2.4% within close range; V2-CRR -8.9% 是唯一 structurally real remaining gap (col-major A-LDS layout, R10 census + R11/R23 SNR wall)。

**R24 INVALIDATED hypotheses** (don't re-invest):
- SQC_DCACHE_BUSY +441% (R23 #1 lever) → R24 fresh +38%, readfirstlane SRD-pin no-op (V2 SRDs already SGPR), Welch t -1.06 null
- TCC_MISS V2-RRR +166% (R23 #3 lever) → R24 fresh +3.1%, sched_barrier TA spread Welch t -0.17 null
- SPI VGPR_SIMD_FULL +18.7% (R23 #5 lever) → V2 actually uses fewer VGPRs than FP8 (V2-RCR 246 vs FP8 254); occ=3 architecturally impossible (V2 LDS 131-139 KB > 160000/3 cap)
- All R23 PMC numbers should be treated as cold-cache + dispatch-aggregation artifacts unless reproduced with warmup≥100 + GPU sclk verified ≥2GHz

**R23 综合 (HISTORICAL, INVALIDATED by R24)**：~~R22 V2 三 layout 经一个 session reverify 全 stable，drift < 0.5%。3 个 dev 全 exhausted（V3 preshuffle REJECTED -2.99%；CRR A-LDS Route X FAIL SNR -2.71 dB 第二次 hit R11 wall；Dev C cycle diagnostic 找到 SQC_DCACHE_BUSY_CYCLES +441% 作为 NEW #1 lever）。~~ R24 reverify 推翻 R23 cycle-level numbers; R23 ranked NEW levers (SQC/TCC/SPI/MFMA-VALU coexec) 全部 invalidated. **关键 paradigm 仍正确**: SQ_INSTS_VMEM 是 transaction count 不是 byte count → V3 preshuffle b64→b128 width promotion 给 0% VMEM cut, R23 Dev A confirmed -2.99% regression。

**R21 VERIFIED PASS metrics (Reviewer GPU0 indep reproduction, RCR)**: V2 mean 3080.18 / median 3078.09 / std 4.63；V1 mean 3021.53 / median 3022.43 / std 3.19；Δ +55.66 TFLOPS / +1.84%；Welch t=23.33 (p<<0.001)；rocprofv3 SQ_INSTS_VMEM byte-exact match V1=6,815,744 → V2=5,767,168 (-15.38%)；correctness 256³/1024³/8192³ 全 SNR ≥49.5 + det 3/3 PASS。

**R22 cherry-picked commits (production)**:
- R21: `4abd4f62` (V2 layout foundation) + `efc389ff` (V2-RCR fastpath wiring) + `f7ae35f6` (R21 docs)
- R22-A: `dabeffa0` (V2-RRR fastpath wiring + `gemm_rrr_pq_v2` pybind + `MXFP8_RRR_PRESHUFFLE_V2_RUNTIME` default 1)
- R22-B: `9a0d0624` (V2-CRR fastpath wiring + `gemm_crr_pq_v2` pybind + `MXFP8_CRR_PRESHUFFLE_V2_RUNTIME` default 1)

**V2 implementation 关键 fact** (供未来调试):
- 新 kernel template parameter `SCALE_VERSION` (1=V1, 2=V2)；hot symbol `*_exact_8wave_scaled_kernel<true, 2>` 资源:
  - RCR: VGPR 246 (V1: 254, **-8 VGPR**), occ 2, LDS 131 KB
  - RRR: VGPR 256 (V1: 256), spills 0 (V1: **19 spills + 80B scratch**), occ 2, LDS 135 KB
  - CRR: VGPR 234 (V1: 232, +2), spills 0, occ 2, LDS 139 KB
- Python `preshuffle_scale_matrix_mfma16_v2_rcr_a/_b` **同一函数复用三 layout**：A/B scale shapes 在 RCR/RRR/CRR 之间相同 (M-major / N-major × k_blocks)
- **dual-patch rule** (R21 silent FAIL learning): 必须同时 patch `load_scale_buffer` (main loop) AND `load_scale_packs_for_pair` (warmup/pre-tail/tail) — RRR/CRR 只有单一 lambda 所以简化为 single patch；RCR 有独立 helper 是关键陷阱
- `--offload-device-only -S` 必须 emit `buffer_load_dwordx4` (A) + `buffer_load_dwordx2` (B), 不是 b32 chains
- **V2 已 hide 所有 vmcnt waits**: R22 Dev C diagnostic 证明 V2-RCR 的 vmcnt-wait 比 FP8 LESS (-8.8%)；sched hints 路径永久关闭

**GPU sclk verify (R22 起强制规范)**: 任何 perf 测量前必须 `rocm-smi` 查 sclk ≥ 2GHz；GPU0 throttled at sclk 260MHz / mclk 2000MHz 是 R22 CRR 第一次 verify FAIL 的根因。Reviewer 必须 dispatch fallback GPU (e.g., GPU1/GPU2)。

**Dispatcher per-size rebuild**: dispatcher gates on compile-time `M_DIM`，每 size 必须 rebuild .so

**R21 VERIFIED PASS metrics (Reviewer GPU0 indep reproduction)**: V2 mean 3080.18 / median 3078.09 / std 4.63；V1 mean 3021.53 / median 3022.43 / std 3.19；Δ +55.66 TFLOPS / +1.84%；Welch t=23.33 (p<<0.001)；rocprofv3 SQ_INSTS_VMEM byte-exact match V1=6,815,744 → V2=5,767,168 (-15.38%)；correctness 256³/1024³/8192³ 全 SNR ≥49.5 + det 3/3 PASS, 67108864/67108864 element pass-rate at 8192³。

**R21 cherry-picked commits (production)**: `4abd4f62` (V2 layout foundation + Python preshuffle + verify drivers) + `efc389ff` (V2 fastpath wiring with `llvm_amdgcn_raw_buffer_load_b128`/`b64` + `dispatch_pq_v2<RCR>` + `gemm_rcr_pq_v2` pybind + `MXFP8_RCR_PRESHUFFLE_V2_RUNTIME` runtime gate).

**V2 implementation 关键 fact** (供未来调试):
- 新 kernel template parameter `SCALE_VERSION` (1=V1, 2=V2)；hot symbol `rcr_exact_8wave_scaled_kernel<true, 2>` VGPR **246** (V1: 254, **-8 VGPR**), occupancy 2 unchanged, LDS 131072 unchanged
- Python `preshuffle_scale_matrix_mfma16_v2_rcr_a/_b` 在 V2 packing 前 reorder source row_groups → wave-tile 顺序匹配 b128 dword 顺序 (option a from R20)
- 必须同时 patch `load_scale_buffer` (main loop) AND `load_scale_packs_for_pair` (warmup/pre-tail/tail)；R21 Dev A 初始 256³ FAIL 真因是后者漏掉
- `--offload-device-only -S` 必须 emit `buffer_load_dwordx4` (A) + `buffer_load_dwordx2` (B), 不是 b32 chains

**R18 真瓶颈定性纠正（推翻 R17 假说）**：rocprofv3 cycle-counter 证明 MXFP8 vmcnt + lgkmcnt 等待 cycles 反而比 FP8 少 -157K + -85K = -242K，但 GRBM_GUI_ACTIVE 多 +845K cyc。差异 100% 由 +30% VMEM-issue 速率（6 extra scale buffer_loads/K-block）造成的 dispatch back-pressure (`SQ_WAIT_ANY +1.14M cyc`) 引起。MFMA cycles 完全相同。SQ_LDS_BANK_CONFLICT = 0。**R17 vmcnt-MFMA critical path 假说 FALSIFIED**——compiler scheduler 已经在最优位。

**历史 baseline (GPU7, 2026-04-17 早期会话)** — 仅作 95% gate 锚点，不再用作回归对照：MXFP8 RCR 2926.61 / RRR 2794.26 / CRR 2740.55

**静态 gate** = 2926.61 × 0.95 = **2780.28 TFLOPS** → R15 RRR / CRR 全 PASS  
**动态 gate** = 3015.73 × 0.95 = **2864.94 TFLOPS** → RRR PASS, CRR 差 34.27 TFLOPS (1.14%)

> **R14 → R15 paradigm shift**：R14 GPU0 测得 RCR=2806 / RRR=2873 / CRR=2841（"RRR > RCR 倒置"），R15 fresh GPU0 测得 RCR=3015 / RRR=2889 / CRR=2830（历史顺序 RCR > RRR > CRR 恢复）。R14 倒置是 cold GPU + missing production flag 双重 artifact

> **VGPR 读数陷阱**：`-Rpass-analysis=kernel-resource-usage` 会为每个符号各报一次；MXFP8 RCR PQ 路径的真实 hot kernel 是 `rcr_exact_8wave_scaled_kernel<Lb1>`（VGPR **254** / LDS **131 KB**）。外壳 `gemm_kernel<Layout0,*>` 只是 dispatcher，显示 VGPR 212 / LDS 139 KB，**不是**可用于 headroom 推断的数字。任何基于「40 VGPR headroom」的优化提案都是错的，请以 scaled kernel 符号的 remark 为准。CRR 同理，看 `crr_exact_8wave_scaled_kernel<Lb1>` 符号的 remark。

## 角色定义

### Decision Maker（主 agent）
- 汇编级差距分析、派活、审议结果、决定 commit
- 汇总 reviewer 结论，记录新的死路到 SKILL
- **每次对话结束前必须**：更新 `TODO.md`（进度 + baseline）和 `agent_prompt.md`（若有规则变化）

### Dev A — 4-Wave 完整升级路径
- **目标**：把 8-wave 的 KPAIR_LOOP + SGPR SRD + scale pipeline 都迁到 4-wave fastpath
- **核心依据**：4-wave 有 14–44 VGPR headroom，唯一有空间做 scale cross-iter pipeline 的路径
- **关键文件**：`rcr_mxfp8_4wave_fastpath.inc`，`kernel_mxfp8_4wave_rewrite.cpp`
- **陷阱**：若用 inline ASM，所有 accumulator AGPRs 必须每个 split block 都 `"+a"`
- **里程碑**：formal ≥ 2897 (超 8-wave) 就 commit

### Dev B — 8-Wave v_lshr 消除
- **目标**：隐藏或消除每 iter 6 个 `v_lshrrev_b32` scale remap
- **核心依据**：这 6 条在 RAW 关键路径；但可以尝试 opsel 字节选择（HW 直接读 32-bit pack 里的第 N 字节，无需 shift）
- ~~**已有开关**~~（**R16 永久关闭**：以下 3 个 flag 与 HOIST_HI 在 `#elif` chain 架构互斥；启用任何一个会自动禁用 HOIST_HI 触发 SNR FAIL 或 spill）：
  - ~~`MXFP8_RCR_EXACT_PQ_PHASE_U16_CACHE_ENABLE=1`~~ (HOIST_HI 关闭后 K_PHASE templated lambda + tail path 不兼容，SNR -1.18 dB)
  - ~~`MXFP8_RCR_EXACT_PQ_REMAP_ONCE_ENABLE=1`~~ (VGPR 254→256 + 1 spill + 8B scratch)
  - ~~`MXFP8_RCR_EXACT_PQ_SCALAR_PHASE_PACKS_ENABLE=1`~~ (同上 256+1 spill+8B)
- **已完成**：HOIST_HI opsel 方案上线（reviewer GPU7 验收 2925.64，A/B +17.80；GPU1 head-to-head +25.67；main-loop v_lshr 0；spills 0；VGPR 256→254）。构建加 `-DMXFP8_RCR_EXACT_PQ_HOIST_HI_ENABLE=1` 即可。
- **经验**：K_PHASE 必须是 **编译期常量**（C++20 templated lambda）；tail 区域走 `rcr_mma_scaled_from_packs_exact`（runtime k_phase）——route tail 到 opsel 会复制 MFMA code path 并制造 31 spills。
- **禁地**：skill 明确 `op_sel_hi` phase1 **inline-ASM** 变体「语义能对，长跑回归」——当前 HOIST_HI 用的是 `__builtin_amdgcn_mfma_scale_*` + `op_sel`/`op_sel_hi`，不是 inline ASM，工作正常。

### Dev C — ASM Rewriter 路径
- **目标**：改写 `rewrite_mxfp8.py`（当前只处理 AGPR，对 8-wave PQ=1 是 no-op）使其对 8-wave 生效
- **思路**：把 6 个 `v_lshr` 前移到**上一 phase** 的 MFMA shadow（64-cycle 延迟里隐藏）
- ~~**stash 已有 Dev C 写到一半的 `rewrite_mxfp8_8wave.py` + `build_rewrite_8wave.sh`**~~ **已关闭**：第二轮实测 +0.18%，HOIST_HI 已从根源消除 `v_lshr`，ASM rewriter 无有效优化空间。该方向已废弃，rewriter 脚本未入库
- **pipeline 陷阱**（来自 MXFP4 经验）：
  - `clang -x assembler` 必须带 `-c`，否则 ld.lld 视为预建 DYN ELF 会吞 kernel
  - cuid 必须从 `grep -oP '__hip_cuid_\K[0-9a-f]+' device.s` 动态获取

### Reviewer / Tester
门禁三步（任何一项失败 → 拒收）：
1. **Smoke**：`HIP_VISIBLE_DEVICES=<dev> MXFP8_PRESHUFFLE_QUANT=1 MXFP8_LAYOUTS=rcr MXFP8_WARMUP=5 MXFP8_ITERS=10 MXFP8_CHECK=1 MXFP8_DETERMINISM_RUNS=3 python3 test_mxfp8_python.py 256 256 256`
2. **Formal**：同上但 `HIP_VISIBLE_DEVICES=7 MXFP8_WARMUP=100 MXFP8_ITERS=200` + `8192 8192 8192`
3. **FP8 回归**：`HIP_VISIBLE_DEVICES=7 FP8_WARMUP=100 FP8_ITERS=200 FP8_CHECK=1 FP8_DETERMINISM_RUNS=3 python3 test_python.py 8192 8192 8192` 必须仍 ≥ 3050

通过 → commit 信息：
```
MXFP8 RCR <change>: <TFLOPS> TFLOPS (<+delta%>)

SNR: XX.XX dB, determinism: PASS (3 runs)
VGPR: X, AGPR: Y, spills: Z, LDS: W KB
```

## 第二轮评审结果 (2026-04-17)

stash@{0} 已 pop 并逐条评审。三条路径：

| 路径 | 结果 | 采纳？ |
|---|---|---|
| Dev B — HOIST_HI opsel 消除 v_lshr | reviewer GPU7 formal 2925.64 (SNR+det PASS)；A/B +17.80；Dev B GPU1 A/B +25.67；VGPR 256→254；spills 0；main-loop `v_lshr` 6→0 | **采纳，已 commit** |
| Dev A — 4-wave KPAIR+SCALE_PIPE | GPU0 同 GPU A/B：4-wave bare 2878 → 4-wave KPAIR+PIPE 2900 (+22)；但仍显著低于同 GPU 上的 8-wave KPAIR+SRD+PIPE (2993)；且 4-wave 走 inline ASM 占 256 AGPR + 256 VGPR → occupancy=1，结构劣势根深 | **拒绝**（不能超越 8-wave） |
| Dev C — ASM rewriter `rewrite_mxfp8_8wave.py` + `build_rewrite_8wave.sh` | GPU2 A/B 仅 +5.94 TFLOPS (+0.18%)；HOIST_HI 已从源头消除 `v_lshr`，rewriter 的重排空间被覆盖 | **拒绝**（边际收益 + 维护成本不划算，脚本未入库） |

任何后续 agent 想重启 Dev C 的 ASM rewriter 方向之前，必须先证明 HOIST_HI 框架下还有可被 rewriter 独占抢到的非 `v_lshr` 结构性收益，否则视为重蹈覆辙。

## 第三轮评审结果 (2026-04-17)

主 agent 基于错误假设（40 VGPR headroom）派了 3 个 dev，结果真实 baseline 是 254 VGPR / ~2 headroom。三条路径全 reject：

| 路径 | flag | 结果 | 采纳？ |
|---|---|---|---|
| Dev A — scale cross-iter prefetch (n+1 ring) | `MXFP8_RCR_EXACT_PQ_SCALE_PREFETCH_N1_ENABLE` | scope=0 full ring 4 spills A/B −2.26%；scope=1 B-only spill-free A/B −1.58%；VGPR 不够 | **拒绝** |
| Dev B — tail compile-time K_PHASE dispatch | `MXFP8_RCR_EXACT_PQ_TAIL_DISPATCH_ENABLE` | 正确+资源 clean（少 12 tail v_lshr）但 reviewer GPU7 A/B 10+20 rounds Δ = −0.04% ~ +0.18%，低于 +0.25% 门槛 | **拒绝**（技术正确但噪声级） |
| Dev C — KPAIR 2× unroll (4× k-pairs / iter) | `MXFP8_RCR_EXACT_PQ_KPAIR_UNROLL2_ENABLE` | body 翻倍→live-range 爆 256 VGPR，51 spills 208 B scratch，A/B −54.87% | **拒绝** |

**本轮无代码 commit，仅文档更正**。SKILL / TODO / agent_prompt 三处都加了「真实 VGPR=254，读 compile remarks 必须取 `rcr_exact_8wave_scaled_kernel` 符号，不要取 outer `gemm_kernel` 包装」的纠偏说明。

下一轮必须先解决 **baseline VGPR 压力** 才能再动 cross-iter pipeline / deeper unroll；否则任何需要额外寄存器的方案都会直接 spill。

## 第四轮评审结果 (2026-04-17)

两条路径，都是**不加 VGPR 不加 LDS** 的尝试，仍全 reject：

| 路径 | flag | 结果 | 采纳？ |
|---|---|---|---|
| Dev D — SCALE_LDS 叠加 PIPELINE_SCALE | `MXFP8_RCR_EXACT_PQ_SCALE_LDS_ENABLE` | VGPR 243 / LDS 132 KB / occ 2 / smoke PASS；但 8192 formal A/B −0.76%（5 runs 全输）**且 determinism FAIL**（max abs 1.43）。这个 flag 以前只是「未验证」，现在有硬数据 | **拒绝（regression + det fail）** |
| Dev E — sched V2 `TK_WAIT_VMCNT(6→8)` | `MXFP8_RCR_EXACT_PQ_SCHED_V2_ENABLE` | 正确+资源不变；GPU1 A/B 10 runs Δ +0.11%，Welch-t 0.58，纯噪声 | **拒绝（marginal）** |

关键新 dead-end 已写入 SKILL：SCALE_LDS 叠加现有 PIPELINE_SCALE 是不可走通的，只能**替换**（且需解决 LDS 同步 determinism），非小改动。

**本轮再次无代码 commit，仅 SKILL / TODO / agent_prompt 更新**。剩余差距仍是 145 TFLOPS / 4.73%。下一轮建议的高收益方向只剩**降低 baseline VGPR 压力**（accumulator reshape 或主动下调到 occupancy=1），因为任何 VGPR-neutral 的调度优化都已被压到噪声之下。

## 第五轮评审结果 (2026-04-17)

三条路径，全 reject：

| 路径 | flag | 结果 | 采纳？ |
|---|---|---|---|
| Dev F — AGPR accumulator 重新绑定（per-MFMA `"+a"` inline asm） | `MXFP8_RCR_EXACT_PQ_AGPR_ACC_ENABLE` | VGPR 254→128, AGPR 0→128；但 Spills 0→13, Scratch 56 B/lane；141+144 V↔A shuffle；GPU3 A/B 10 runs Δ −2.19% | **拒绝（regression）** |
| Dev G — scale L2 cache-policy hint (sc0) | `MXFP8_RCR_EXACT_PQ_SCALE_L2_HINT_ENABLE` | 12 scale buffer_load 用 sc0，正确+资源不变；GPU4 A/B 15/20 runs Δ +0.066% Welch-t 0.20 | **拒绝（marginal）** |
| Dev G2 — scale buffer_load_b32 × 2 → b64 合并 | `MXFP8_RCR_EXACT_PQ_SCALE_LOAD_B64_ENABLE` | **结构不可行**：6 scale dwords 来自 6 独立 SRD，最小间距 8192 B；b64 要 X 与 X+4 同 SRD → 无法合并 | **拒绝（broken）** |

**5 轮 7 个 dev 全 reject**。MXFP8 在当前结构下的微调空间已饱和。剩余 145 TFLOPS 差距只能靠高风险结构重构：
- AGPR accumulator **fused-asm block**（8-32 MFMA 打包一个 asm，重写 `_row`/`_impl`，破坏 HOIST_HI 约定）
- SCALE_LDS **完全替代** PIPELINE_SCALE（不是叠加）+ 解 determinism 问题
- `preshuffle_scale_matrix_mfma16` **layout 重设计**让 scale 可 b64 合并（影响所有 MXFP8 kernel + Python 参考）
- 主动下调 **occupancy=1** 换更深 cross-iter pipeline

下一轮建议**不再并行派 3 dev**，选 1 条结构路径深入做 1-2 周。本轮所有 dead-ends 已写入 SKILL。

## 第六轮评审结果 (2026-04-17)

Dev H 单条路径：**强制 occupancy=1 架构性不可能**：

- **Dev H** — FORCE_OCC1 flag：编译器忽略 request（物理事实：512-thread block 在 4-SIMD CU 上最少 2 waves/SIMD），A/B +0.023% 噪声；叠加 pipeline 扩展 → 63 spills −63.36%。**Occupancy-knob 轴已关闭**。
- **结论更新**：occupancy=1 不是"难"也不是"高风险"，是**算术不可能**。要 occ=1 只能改 block 大小（不同 kernel 结构，基本是整个项目重写）。下一轮可行的结构方向收窄到 3 条：SCALE_LDS 完全替代 / AGPR fused-asm block / preshuffle layout 重设计。

**6 轮 / 8 dev agents 全 reject**。SKILL dead-ends 列表再 +1（FORCE_OCC1 的架构不可行证明）。

## 第七轮起：RRR / CRR 95% gate 任务 (2026-04-17)

RCR 方向已达微调上限，决策者转向 RRR / CRR 对齐目标：**≥ 2780.28 TFLOPS**（RCR × 0.95）。

### GPU7 实测 baseline

| Layout | TFLOPS | vs RCR | 状态 |
|---|---:|---:|---|
| RCR | 2926.61 | 100.0% | 参照 |
| RRR | 2794.26 | 95.48% | ✅ 已达标 |
| CRR | 2737.94 | 93.55% | ❌ 差 42.34 TFLOPS |

### CRR 差距诊断

`crr_mxfp8_exact_8wave_fastpath.inc` 主循环 (L407-461) 在每奇数 k 做：
```cpp
for (int g = 0; g < crr_a_pack_count; g++) {
    a0_scale_packs[g] = std::bit_cast<fp8e8m0_4>(
        std::bit_cast<uint32_t>(a0_scale_packs[g]) >> 16);   // C++ 层 shift
    a1_scale_packs[g] = ...;  // 再一条
}
// 再 4 条 for b0, b1
```
这 6 条 C++ shift 编出 6 × `v_lshrrev_b32` per kpair，**跟 RCR pre-HOIST_HI 完全同构**。

关键资产（已经写好，只是主循环没用）：
- `crr_mma_scaled_base<opsel_a, opsel_b>` 支持 2-bit opsel（bit 1 = byte-select lo/hi）
- `crr_exact_cA_with_b1_interleave_raw_phase<K_PHASE, INSERT_AFTER, ...>` compile-time K_PHASE
- `crr_mma_scaled_phase<K_PHASE>(...)` compile-time
- tail 路径（L463-536）已经用 compile-time `_phase<0>` / `_phase<1>`

### CRR R7-R9 评审结果 (2026-04-17)

总计 9 个 dev attempt，1 个采纳 (Dev D)，8 个 reject。

| 路径 | flag / 做法 | 结果 | 采纳？ |
|---|---|---|---|
| R7 Dev A — HOIST_HI K_PHASE templated lambda v1 | 主循环 K_PHASE 模板化 | VGPR 256 + 53 spills, A/B −54% | **拒绝** |
| R7 Dev B — HOIST_HI v2 (削减 helper inline) | 同上 + try减少模板内 inline | VGPR 256 + 91 spills, A/B −67% | **拒绝** |
| R7 Dev C — HOIST_HI v3 (再削) | 同上 | VGPR 256 + 173 spills, A/B −54% | **拒绝** |
| **R7 Dev D — PIPELINE_SCALE only** | `MXFP8_CRR_EXACT_PQ_PIPELINE_SCALE_ENABLE=1` | ✅ GPU7 reviewer 2740.55 (+0.243%)，VGPR 247→232 (−15)，spills 0，det 3/3 | **采纳，commit `8934e95c`** |
| R7 Dev E — sched_barrier 实验 (no body change) | sched_barrier 加 hint | A/B Δ ≈ 0% (噪声)。**结论：v_lshr 不在 critical path** | **拒绝（marginal）** |
| R7 Dev F — `__noinline__` outlined helper | 把 main loop body 拆出 noinline 函数 | catastrophic：correctness 46% / scratch 800–888 B/lane / A/B −97%。**AMDGPU calling conv 无法跨 noinline 边界保持 4 个 accumulator live** | **拒绝（broken）** |
| R8 Dev G — runtime branch HOIST_HI | runtime if/else 替代 templated lambda（避免 codegen 翻倍） | 3 变体全 FAIL gate；V1 256 VGPR + 13 spills，V2 manual unroll 256 + 347 spills；A/B −94%，correctness FAIL | **拒绝（broken）** |
| R8 Dev H — PIPELINE+HOIST 组合 | 同时叠加 PIPELINE_SCALE 与 HOIST_HI templated lambda | 256 VGPR + 29 spills；PIPELINE 的 SRSRC（24×32-bit）与 HOIST_HI 双 phase packs live 互相挤兑 | **拒绝（spills）** |
| R9 Dev I — HOIST + PIPELINE + 关 INTERLEAVE_B1_LDS | 删 `crr_exact_cA_with_b1_interleave`（重 8-MFMA helper），按 RRR 简单 4-MMA 结构走 + HOIST_HI K_PHASE templated lambda + PIPELINE_SCALE | 256 VGPR + 145 spills + 332 B/lane scratch | **拒绝（spills）** |

### 关键架构发现 (R7-R9)

**HOIST_HI K_PHASE templating 与 CRR 4-accumulator main loop 根本不兼容**：

- CRR baseline 247 VGPR / 254 cap → 仅 7 VGPR headroom
- K_PHASE templated lambda 把 4 MMA × 2 phase = 8 inlined MMA blocks 同时展开
- 任何 templated body doubling 都要 9–25 VGPR
- → 必然爆 256 VGPR + spills，**与是否叠加 PIPELINE_SCALE / 是否关 INTERLEAVE_B1_LDS 无关**
- 在 5 个独立尝试（Dev A/B/C/H/I）观察到同一 256-VGPR ceiling，证伪假设

**RRR 为何成功 HOIST_HI 移植**：RRR 用简单 2-MMA-call body（无 `crr_exact_cA_with_b1_interleave` 那种 8 单 MFMA + LDS interleave），K_PHASE 模板化后 inlined codegen 体积可控，在 254 VGPR cap 下能容纳。CRR 4-accumulator + interleave helper 是结构差异，删 interleave 也救不回（Dev I 已证）。

**v_lshr 不在 critical path** (Dev E)：之前以为 6 × `v_lshrrev_b32` 是 main loop bottleneck，sched_barrier 实验证伪。即使理论上消除这 6 条 shift，对总时间影响也在噪声里。HOIST_HI 在 RCR 上的成功更多来自 codegen 重排（更优的 issue order）而不是单纯减 shift。

### 已关闭的死路

- ~~HOIST_HI 移植 CRR (任何形式：templated lambda / runtime branch / outlined noinline)~~：架构性 VGPR 不足
- ~~CRR HOIST_HI + 任意辅助 flag 组合（PIPELINE_SCALE / 关 INTERLEAVE）~~：仍然 spills
- ~~`__noinline__` outlined helper~~：AMDGPU calling conv 不能跨 noinline 保持 4 accumulator live
- ~~sched_barrier-only 调度优化~~：v_lshr 不在 critical path，调度变化在噪声内

### 未来可探索方向（高风险结构性，本会话不做）

- **CRR 4-accumulator → 2-accumulator 重构**：合并 cA/cB/cC/cD → 减一半 VGPR live，留出 templated lambda 空间
- **KPAIR_LOOP 移植 CRR**：参考 RCR 的 KPAIR loop 改造，可能改变 register lifetime 形状
- **重写 `crr_exact_cA_with_b1_interleave_*`**：拆掉 8 个独立 MFMA + LDS interleave，换更紧凑的实现
- **CRR scale loader 重设计**：bypass 现有 scale pack 路径，类似 RCR 的 SGPR-SRD path

### 后续计划

1. CRR PIPELINE_SCALE 已 commit (`8934e95c`)，达到当前微调上限
2. **CRR 不再值得短期投入小改动**：剩 39.73 TFLOPS 必须靠结构性重构，下一会话若要继续应**单条深入**而非并行 dev fan-out
3. RRR 监控；若后续回退到 95% 以下再补
4. 回到 RCR 的剩余 4.70% 差距（三条高风险结构方向）

## 第十轮评审结果 (2026-04-17)

主 agent 转向**先诊断后开方**：先用 rocprofv3 + ASM census 找真实瓶颈，再派 dev。这一轮的核心收获**不是优化**，而是**用三角验证锁定根因**（非-MFMA issue 争用，源头在 LDS 管道）。

### 诊断阶段（2 个并行 analysis agent）
- **rocprofv3 GPU6 8192³ counters (CRR vs RRR)**：MFMA 数量相同（16.7M），MFMA busy cycles 完全相同；但 SQ_BUSY_CU_CYCLES +3.75% / SQ_INSTS_VALU **+61%** / SQ_INSTS_LDS **+50%** / SQ_WAIT_INST_LDS +20%。Per-MFMA 执行：CRR 2.70 VALU + 1.50 LDS vs RRR 1.68 + 1.00。**MFMA 管道完全饱和**，差距 100% 来自非-MFMA issue 争用
- **ASM census per body**：CRR 144 `ds_read_b64_tr_b8` (8 B/读) vs RRR 64 `ds_read_b128` (16 B/读) + 64 `ds_read_b64_tr_b8`。CRR 多 **80 个** LDS 读指令
- **结构根因**：CRR 的 A 侧用 col-major LDS 布局（`A_col_reg` = `rt_fp8e4m3<BK=128,RBM=64,col_l,rt_128x16_s>` = 128 dwords） vs RRR 的 row-major（`A_row_reg` = 16 dwords，**8× 小**）。Col-major A 必须用窄的转置读，这是 layout 内禀属性
- **重要纠偏**：之前以为 RRR 只有 2 个 accumulator——错的，RRR 也是 4 个 cA/cB/cC/cD（同 CRR）。R7-R9 提出的 "CRR 4-accumulator vs RRR 2-accumulator" 假说**完全错误**。真正区别只在 A_col_reg vs A_row_reg

### 开方阶段（3 个并行 dev agent，全 reject）

| 路径 | flag / 做法 | 结果 | 采纳？ |
|---|---|---|---|
| **R10 Dev J** — 2× kpair unroll **WITHOUT** K_PHASE templating | `MXFP8_CRR_EXACT_PQ_KPAIR_UNROLL2_ENABLE` | FAIL，VGPR 232→256 + 26 spills + 104 B/lane scratch。即使无 templating，body doubling 仍触发 live-range 翻倍（phase-0 的 a/b prefetch 撑到 phase-1）。**与 R7-R9 templated 失败同根** | **拒绝（spills）** |
| **R10 Dev K** — `>>16` shift coalesce + `wn*RBN` precompute | `MXFP8_CRR_EXACT_PQ_VALU_TRIM_ENABLE` (split: `_LSHR_PK_ENABLE` + `_INTERLEAVE_PRECOMPUTE_ENABLE`) | MARGINAL，Δ ≈ 0% (VGPR 232 不变 / 0 spills)。**编译器已自动 hoist `wn*RBN`**——profiler "32 v_add per body" 是 pre-hoist 静态分析，不是最终 ISA。`v_alignbit_b32` 与 `v_lshrrev_b32` 占同一 issue pipe | **拒绝（marginal，编译器已优化）** |
| **R10 Dev L** — load reordering (b1 pre-issue + scale hoist) | `MXFP8_CRR_EXACT_PQ_LOAD_REORDER_ENABLE` (sub-A 'b1 pre-issue', sub-B 'scale reorder') | FAIL，sub-A −0.49% / sub-B −1.78% / combined −3.14% (correctness/det 全 OK)。**`lgkmcnt` 等待同时覆盖 LDS + scalar/VMEM scope**——重排不能让 scale buffer_load 与 A/B LDS 真正并行；反而把 scale dest VGPR live range 撑过 A/B 读寄存器期 (232→254) | **拒绝（regression + 揭示 lgkmcnt 同步语义）** |

### R10 关键产出（不是优化，是死路确认 + 真实瓶颈定性）

**真实瓶颈**：CRR 受限于 LDS 管道争用 + lgkmcnt 同步范围，**不是** MFMA、**不是** v_lshr、**不是** opsel 计算、**不是** cache miss、**不是** VMEM。要破 gate 必须改 A 的 LDS 布局（global→LDS 阶段做 transpose 让 A 侧用 `ds_read_b128` 宽读），这是大型重写不在本 sprint scope。

**已穷尽的微调维度（10 轮 / 12 个 dev attempt 全 reject 或 marginal）**：
- HOIST_HI templating (R7-R9 Dev A/B/C/H/I) — VGPR ceiling
- noinline outline (R7 Dev F) — calling conv 不能保 4 acc live
- runtime branch (R8 Dev G) — 同 templating ceiling
- 2× unroll (R10 Dev J) — 同 live-range 翻倍 ceiling
- VALU coalescing (R10 Dev K) — 编译器已优化
- Load reorder (R10 Dev L) — lgkmcnt 同步范围阻塞
- sched_barrier (R7 Dev E) — v_lshr 不在 critical path
- PIPELINE_SCALE (R7 Dev D) — **唯一 win**，+0.243%, 已 commit

### 后续可探索方向（高风险结构性，本会话不做）

1. **A LDS 布局 transpose（最高潜力）**：在 global→LDS 阶段做 transpose 让 A 侧能用 `ds_read_b128`，把 A 侧 LDS 读指令砍半。代价：global load 阶段的 swizzle 复杂化，可能影响 occupancy 和 bank conflicts。需要重写 `load_col_from_v2_st` 配合
2. **MMA 重排：cA + cC fusion**（同 a_pack 复用）：cA 和 cC 都用 b0，cB 和 cD 都用 b1。当前是 cA→cB→cC→cD（每次切 b），改成 cA→cC→cB→cD 可能减少 b 切换，但需重审 a 的 live range
3. **FORCE 4-wave with fused asm block**：换 4-wave 结构（256-thread block），单 SIMD per CU，更深 ILP——但 4-wave 历史已证劣于 8-wave (Dev A round 1)

下一会话不要再用并行 dev fan-out 微调，**单条深入做 LDS 布局 transpose**。

## 第十二轮评审结果 (2026-04-17) — Diagnostic-S 推翻 R10 LDS-pipe 假设，新瓶颈：SPI launch allocator stall

### Diagnostic-S 用 rocprofv3 测了 33 个 cycle-level counter（5 PMC chunk，GPU6 vs GPU6）

**新发现（与 R10 矛盾的核心）**：
- `SQ_LDS_BANK_CONFLICT`、`SQ_LDS_ADDR_CONFLICT`、`SQ_LDS_UNALIGNED_STALL` 在 CRR 和 RRR **都是 0**。R10 的"LDS pipe contention"假设错了，根本没有 bank 冲突
- `SQ_LDS_IDX_ACTIVE` CRR 与 RRR **完全相同**（5.03e7 cycles）。LDS unit 实际"忙碌"程度一样。CRR 的 +50% 指令数没让 LDS 单元更忙——`ds_read_b64_tr_b8` 在 LDS 内就是更轻的 op
- CRR 的 `TCP_PENDING_STALL_CYCLES` -34%，`TA_ADDR_STALLED_BY_TC` -92%，`TCP_TCP_TA_DATA_STALL` -50% —— 访存 backend 比 RRR 更轻
- CRR 的 `SQ_VALU_MFMA_COEXEC_CYCLES` +61% —— ILP 反而好

**真正瓶颈**：
- **`SPI_RA_LDS_CU_FULL_CSN +388%`** （CRR 9.80e11 vs RRR 2.01e11）—— wave 启动器在 CU 上被 LDS 占用槽位卡住，下一个 workgroup 等 5× 长才能 launch
- **`SPI_RA_RES_STALL_CSN +388%`**（同向）
- **`SQC_DCACHE_BUSY_CYCLES +129%`** —— 标量 cache 偏热
- 估算：`(9.8e11 − 2.0e11) / (224 CU × launch overhead) ≈ 3-5%` 端到端代价 —— 与 1.43% gate gap 同量级

**机理**：CRR 用 136 KB LDS/block，RCR 用 131 KB（5 KB 差）；2 blocks/CU × 136 KB = 272 KB，把 CU 的 LDS 池吃满，下一 block 等。RRR 因为 LDS 用量更小，新 block 上得快

### R12 dev fan-out 全部 timeout

派了 4 个 dev：
- **Dev O**（CRR_ROW_SHARED_TRANSPOSE 深度调试 + LDS dump 验证）：worktree 在 42f5407b base，16:51 后静默 1.5h+，**无 commit**。可能在 Step 1 LDS dump 调试卡住
- **Dev P**（CRR_USE_V3_SWIZZLE）：worktree 在 **stale main base** (b027c06b 无源)，**无 commit**
- **Dev R**（`-mllvm` 编译 flag sweep）：worktree 在 stale base，从 main checkout 拷源 build .so，**无 commit**
- **Dev T**（LDS 分配缩减 136→131 KB，**Diagnostic-S 角度**）：worktree 在 42f5407b base，活跃到 18:26（最后 .so build），**无 commit**

R12 唯一产出 = Diagnostic-S 的瓶颈定性更正。R12 commit 只能是文档。

### R12 关键产出

**正确的优化方向（按 SPI launch 假说排序）**：
1. **缩减 CRR LDS/block**（最直接命中瓶颈）：
   - 单缓冲 A 或 B（如果 PIPELINE_SCALE 允许）
   - 把 A 和 B staging 通过 union/手动 offset 重叠（生命周期不重叠时可行）
   - Scale staging area 复用
   - 目标：从 136 KB 降到 ≤131 KB（RCR 等量），让 SPI 占用槽位降到 RRR 水平
2. **`__launch_bounds__(512, 3)`**：尝试提示 SPI 预留 3 blocks/CU。需先确认 LDS 是否能容（≤ 64 KB hard 还是 160 KB？需查 gfx950 spec 或 occupancy 实测）
3. **减少 SQC_DCACHE 压力**：per-CTA 常量改 `s_load_b256` 单次加载，而不是每 iter `s_load`

**已死的方向（R12 证伪，下轮不要再投资）**：
- LDS bank conflict / address conflict 修复 —— 已证 0 个 conflict
- LDS pipe issue rate 优化 —— LDS unit 利用率 CRR=RRR
- TCP/TA 缓存 prefetch —— CRR 已经更轻
- MFMA-VALU 调度重排 —— CRR 已经 +61% coexec

### 新会话建议

1. **不要重派 R7-R11 类型的微调**（HOIST_HI、KPAIR、scale prefetch 等）—— 都已 saturated
2. **优先派 1 个深 dev 做 LDS pack/shrink** —— Diagnostic-S 的发现需要被实测验证
3. 保留 1 个 reviewer agent 做 A/B Welch-t formal
4. 全部用同一个 base（42f5407b），不要让 worktree 落在 stale main 上

## 第十四轮评审结果 (2026-04-17) — paradigm shift：gate 已在 GPU0 上达标，R7-R14 追的"1.43% gap"是测量幻觉

### R14 行动

派 2 个并行 dev：
- **Dev A** — sweep 8 个未试过的 CRR fastpath knob：CRR_INIT0_VMCNT, CRR_INIT1_VMCNT, CRR_STEADY_VMCNT, CRR_EPILOGUE_VMCNT, CRR_PREFETCH_LGKM, CRR_EXACT_B1_LDS_INSERT_AFTER (0-8), CRR_ENABLE_SCHED_BARRIER, CRR_ENABLE_STEADY_MID_BARRIER。无 LDS 改动，纯调度
- **Dev B** — single-buffer B (`Bs[2][2]→Bs[1][2]`), 在 SPI launch-allocator 假说下应是直接命中（−34 KB / −24% LDS shrink）

### R14 实测结果

**Dev A — DEAD-END**：
- 30+ 单 knob/组合 build + bench
- 最佳：b1_8+i0_3+i1_7 在 GPU7 formal **2745.59** TFLOPS（baseline 2750.92，−0.12%）
- 所有信号在 ±25 TFLOPS quick / ±5 TFLOPS formal 噪声带内
- STEADY_MID_BARRIER=0 correctness FAIL；STEADY_VMCNT≤3 −15% 大退化；其余可正确编译/运行但无 win

**Dev B — DEAD-END but PARADIGM-INVALIDATING**：
- 实现 `MXFP8_CRR_SINGLE_BUF_B` 宏（默认 0），correctness 100% PASS, SNR 49.60
- LDS 139264→104448 byte（**−34816 = −34 KB，验证 4× 于 R13 V3 swap 的 8 KB 缩减**）
- VGPR 232→256 (+24，因为新加 vmcnt 等 force scheduler 保更多 state live)
- GPU7 formal: CRR **2747.59** vs baseline 2750.92 = **−0.12% (NOISE)**
- **关键**：如果 R12 Diagnostic-S 的 SPI 假说是真的 dominant bottleneck，那么 24% LDS relief 本应大幅释放 SPI 槽位；实测仅 −0.12% → **SPI_RA_LDS_CU_FULL_CSN +388% 是 symptom 不是 cause**
- 真实 cancel mechanism: single-buffer B 强制 vmcnt(0) wait 在每个 iter，B prefetch 失去与 compute overlap，正好抵消 SPI 收益
- Diff 已 revert

### R14 决策者重测 baseline（critical paradigm-shift）

Dev B 的 GPU7 baseline 显示 RCR = 2726 TFLOPS，**远低于历史 2925.64**。立即 cross-check：

| GPU | RCR | RRR | CRR | CRR/RCR |
|-----|----:|----:|----:|--------:|
| **GPU0** | **2806.17** | **2873.02** | **2840.98** | **101.24%** ✅ |
| GPU0 (run 2) | 2813.02 | — | 2841.60 | 101.02% ✅ |
| GPU0 (run 3) | 2806.74 | — | 2838.67 | 101.14% ✅ |
| **GPU0 formal (200i × 3 det)** | 2808.44 | — | **2835.16** | 100.95% ✅ |
| GPU6 | 2708.25 | 2780.65 | 2715.83 | 100.28% |
| GPU7 (今日) | 2726.20 | 2775.90 | 2736.45 | 100.37% |

**Gate 2780.28 在 GPU0 上达标 (+54.88 TFLOPS, formal SNR 49.60 PASS, det 3/3 PASS, correctness 100%)**

### R14 综合结论 — 8 轮 sprint paradigm shift

1. **Gate 已达标**：CRR 在 GPU0 上 formal 测量 2835.16 TFLOPS > 2780.28 gate
2. **CRR 在所有 GPU 上其实都 ≥ RCR** (CRR/RCR = 100.28-101.24%)，从未存在真实的"CRR 弱于 RCR"问题
3. **R7-R14 追的"1.43% gap"是测量幻觉**：
   - 历史 baseline RCR = 2925.64 (在某个 GPU/状态下测得)
   - 当时 CRR = 2740.55 (在不同 GPU/状态下测得)
   - 两个数字组合产生"-1.43%"幻觉
   - 任何同会话同 GPU 的 RCR vs CRR 实测都显示 CRR ≥ RCR
4. **GPU7 已退化**：RCR 从 2925→2726 (−7%)，整个板子整体性能下降，不是 CRR-specific 问题
5. **R12 SPI hypothesis 已伪证**：34 KB LDS shrink → −0.12% perf 不可能是 SPI bound
6. **PIPELINE_SCALE only (commit 8934e95c) 是真正的最终 production fix**

### 已死的方向（R14 证伪，下轮不要再投资）

- CRR fastpath VMCNT/LGKMCNT/INSERT_AFTER 单 knob 调优 —— 全噪声带
- CRR LDS 缩减作为 SPI 假说命中路径 —— 34 KB shrink = -0.12%, SPI 不是 dominant
- 跨会话/跨 GPU 比较 RCR 与 CRR baseline —— 必然产生伪 gap
- "继续追 CRR +1.43%" 作为目标 —— gate 已 met

### 新规范（R14 起）

1. **Gate 验收必须同会话同 GPU 同时测 RCR + CRR**（per-iteration sync, warmup=100, iters=200, det 3, correctness ON）
2. **GPU0 是当前唯一持续达到历史性能水平的板子**；GPU7 已退化，GPU6 整体偏慢
3. **派 reviewer 时显式 `HIP_VISIBLE_DEVICES=0`** 做 final gate 验收
4. **所有 baseline 数字必须标注测量 GPU + 日期**，未来 sprint 不要再重蹈"跨会话 phantom gap"

### 新会话建议

如果 user 强制继续 CRR 优化（达标后还要继续）：
1. **先在 GPU0 上重测 RCR baseline** —— 可能 RCR 本身有 untapped 收益（GPU0 CRR > GPU0 RCR 说明 CRR 可能已经更优）
2. **或者重新定义 gate 为"今天 RCR × 0.95"** (dynamic gate) —— CRR 也已在所有 GPU 上达标
3. **不要继续微调 CRR 的 sched/wait knobs** —— R14 Dev A 已穷尽且全 noise
4. **不要继续追 LDS 缩减作为 SPI 命中** —— Dev B 已伪证
5. 如果一定要找新方向，看 RCR / RRR 是否有未优化的 inner-loop hint（GPU0 RCR 2806 vs RRR 2873 说明 RRR > RCR by 67 TFLOPS，RCR 可能有 2-3% headroom）

## 第十三轮评审结果 (2026-04-17) — V3 swizzle swap 缩 LDS 8 KB 但 fastpath 正确性破坏

### R13 行动

按 R12 Diagnostic-S 假说路径 (1)（缩 CRR LDS）选了最直接路径：把 CRR fastpath 的 A/B tile 从 `ST_v2a`/`ST_v2`（含 128 B subtile padding）改成 `ST_v3`（0 padding，预期 8 KB shrink）。代码变更：
- `crr_mxfp8_exact_8wave_fastpath.inc:41` 把 `static_assert(!CRR_USE_V3_SWIZZLE, ...)` wrap 进 `#if !CRR_USE_V3_SWIZZLE`
- 同样 wrap 4 个 `CRR_*_REG_ROW_LOAD_*` asserts
- `crr_exact_cA_with_b1_interleave_fixed_phase` 和 `_raw_phase` 函数 templated on STB（之前 hardcoded `const ST_v2&`），加 lambda：if constexpr `std::is_same_v<STB, ST_v3>` → `load_col_from_v3_st(b1, b1_tile, wn*RBN)`，else → `load_col_from_v2_st(...)`

### R13 实测结果

**构建 (V3=1 + fastpath=1)**：✅ VGPR 232→242 (+10), spills 0, occ 2 不变，**LDS Size 139264→131072 byte 完全匹配预期 8 KB shrink**

**正确性**：fastpath FAIL — 8192³ pass-rate **66966620/67108864 = 99.79%** = **142244 个 NaN 输出**，TFLOPS 表面 2457（NaN 下游传播触发 division-by-NaN slowdown），SNR=NaN

**关键 diagnostic（隔离 bug）**：
1. **fastpath OFF + V3=1**（强制走 generic gemm_kernel<Layout::CRR> 用同样 ST_v3 + load_col_from_v3_st）：✅ **PASS** SNR 49.60 dB，pass-rate 100%（但只有 2.71 TFLOPS，generic kernel 慢 1000×）→ **证明 ST_v3 + V3 col-load helpers 本身正确**
2. **fastpath ON + V3=1 + b1 interleave OFF**（`CRR_EXACT_INTERLEAVE_B1_LDS=0`）：仍 FAIL **完全相同 142244 NaN** → bug **不在 b1 interleave**，在 fastpath 更深层的 LDS write/read pipeline ordering

**结论**：V3 swap **架构上对 CRR fastpath 不兼容**，即使 LDS shrink 完美匹配 SPI 假说预期。non-fastpath barrier 重所以不暴露；fastpath pipelined ds_write 与 ds_read_b64_tr_b8 在 double-buffer 循环里有 ordering 问题（具体是否 prefill_swizzled_offsets 与 v3 swizzle 在 double-buffer Bs[2][2] 上有 stride 假设差异，未深查）

**已 revert** `crr_mxfp8_exact_8wave_fastpath.inc`，工作树恢复干净

### R13 关键产出（dead-end，不是优化）

- **V3 + non-fastpath PASSES** —— 可作为 ground truth 验证 V3 byte 布局是 OK 的
- **V3 + fastpath fails identically with/without b1 interleave** —— bug 在更基础层（global→LDS write 与 col-load read 的 pipeline ordering），不是 R13 加的 lambda
- **LDS 8 KB shrink 是真实可达的**（编译器报告确认），但触发的 fastpath 正确性破坏需要重写 fastpath 同步层才能消除

### 已死的方向（R13 证伪，下轮不要再投资）

- ST_v3 swizzle swap 的任何变体 —— fastpath 不兼容（已实测）
- 加 sched_barrier/extra `s_waitcnt lgkmcnt(0)` 救 V3 fastpath —— 任何加粗 barrier 都会消掉 SPI 收益（瓶颈是 launch allocator，不是 LDS bank）

### 新会话建议（按 SPI 假说剩余路径优先级）

1. **`__launch_bounds__(512, 3)`** —— 纯 compile-time 试验，无 LDS swizzle 改动，只在 `crr_exact_8wave_scaled_kernel` 的 `__launch_bounds__` 加 `, 3`。如果 register usage 不超 cap、不强制 spill，会让 SPI 预留 3 blocks/CU 而不是 2，直接命中 SPI_RA_LDS_CU_FULL 瓶颈。**最低风险，最高收益候选**
2. **单缓冲 B（`Bs[1][2]` 而不是 `Bs[2][2]`）** —— 直接砍 32 KB LDS（136→104 KB），让 SPI 占用槽位降到 RRR 以下。代价：B 的 global→LDS prefetch 与 register load 重叠机会减半，需要重新分析 vmcnt schedule。中等风险，需 ~50-100 LoC
3. **per-CTA 常量改 `s_load_b256` 单次加载** —— 减 `SQC_DCACHE_BUSY_CYCLES +129%`。低风险但收益不确定（可能 < 1%）
4. **静默 dev fan-out 模式不工作** —— R12/R13 都证明了，下轮直接 inline 干活更快

## 第十一轮评审结果 (2026-04-17) — A LDS 布局 transpose 两条路径全 BROKEN，B 也是窄读

主 agent 派 2 个 scout 调研可行性 → 选定两条并行 dev：

### Scout 阶段产出
- **Scout-1（A LDS transpose 可行性）**：`ST_crr_a` 现为 `ST_v2a = st_fp8e4m3<HB,BK,st_16x128_v2a_s>`（kernel_mxfp8_layouts.cpp:399-401, 4227-4234），byte-XOR swizzle `((offset>>7)&7)<<4`。RRR 用 `ST_row = st_16x128_s`。`load_transpose` 已存在（kernel_mxfp8_layouts.cpp:1726-1825），处理 col-global→row-LDS。已存在的 `CRR_A_LDS_REENCODE` macro（line 4235-4276）就是这条 transpose 的 partial impl，但通过额外 LDS write/read 而不是直接写。**Verdict**：ATTEMPTABLE，~80-150 LoC
- **Scout-2（替代 MFMA shape）**：仅有 `mfma_scale_f32_16x16x128_f8f6f4` 和 `mfma_scale_f32_32x32x64_f8f6f4` 两个 scaled FP8 intrinsic（mma.cuh:104,119,137）。RRR 和 CRR **都用 16x16x128**——no signal。32x32x64 的 A 操作数 dword 数量相同（8 dwords），但 D=floatx16 vs floatx4 → 4× 输出/issue → 可减少总 issue 量。但需要新 scaled wrapper + RBM/RBN 半化 + scale-pack 索引重设。Effort 150-300 LoC + 1-2 天

### 开方阶段（2 个并行 dev agent，全 reject）

| 路径 | 做法 | 结果 | 采纳？ |
|---|---|---|---|
| **R11 Dev M** — 直接 ST_row 替换 + reg transpose | `CRR_A_LDS_ROW_MAJOR` flag, `ST_crr_a` 改 `st_16x128_s`，主循环用 `load(A_row_reg, sub) → transpose(A_col_reg, A_row_reg)` | **BROKEN**：编译过 (VGPR 248 / 0 spills / occ 2)，但 SNR=−2.71 dB / 1340 TFLOPS。`load_transpose` 写出的 LDS 布局与通用 `load(A_row_reg, ...)` 期望的 row-major 消费不匹配。`CRR_ROW_SHARED_TRANSPOSE` 参考路径自身被 fastpath `static_assert` 关掉，无 known-good baseline | **拒绝（correctness）** |
| **R11 Dev N** — 启用现成 `CRR_A_LDS_REENCODE=1` (stepping-stone) | 强制走 reencode 路径，`load_col_from_v2a_st → transpose → store(Aenc) → load(b128)` | **BROKEN**：1017 TFLOPS / SNR=1 dB。被迫关 8-wave fastpath（`crr_mxfp8_exact_8wave_fastpath.inc:32` 硬 `#error`）；REENCODE 分支调用未-scaled `mma_AB(...)` 而不是宏 `CRR_DO_MMA(...)`——MXFP8 scale 通路未接入。**ISA 验证 wide read 原理正确**：基线 0× ds_read_b128 + 144× ds_read_b64_tr_b8 → REENCODE 48× ds_read_b128 + 144× ds_read_b64_tr_b8 | **拒绝（correctness + 非 fastpath）** |

### R11 关键新发现：B 操作数也是窄读

ASM census 144 个 ds_read_b64_tr_b8 中 ~96 来自 **B**，仅 ~48 来自 A。CRR 的 col-major B layout 同样阻塞 b128 宽读。即使 A 完美修复，**只解决 ~25% 的 LDS pressure**——A-only 修复**结构上无法**到达 +1.43% gate。

### R11 综合结论

CRR gate (2780.28 TFLOPS) 在当前架构下需要 multi-day 结构重写：
1. `load_transpose` 与 `ST_row` 的 LDS 布局对齐调试（需要 instrumented LDS dump 或先把 `CRR_ROW_SHARED_TRANSPOSE` 在 gemm_kernel 非-fastpath 跑通做对照）
2. 加 `A_row_reg` 重载到 `crr_mma_scaled_from_packs` 把 MXFP8 scale 通路接进新 A 路径（约 50-100 LoC，需 mirror RRR 的 `mma_ABt` pattern）
3. **B 操作数 layout 重设计**（解锁剩余 75% LDS pressure）—— Scout 未调研，未知是否有 partial impl

**承认 gate 当前架构不可达**。已 commit 的 `8934e95c` PIPELINE_SCALE 默认开（+0.243%, 2740.55 TFLOPS）是 R7-R11 共 11 轮唯一 strict win。下一会话若续做 CRR：先开 `CRR_ROW_SHARED_TRANSPOSE` 在非 fastpath 跑通做 known-good baseline，再决定要不要投入 multi-day 重写；或转向 RCR 剩余 4.70% 差距。

## 第二十三轮评审结果 (2026-04-18) — ★ R22 V2 stable reverify；3 dev 全 exhausted (V3 REJECTED / CRR A-LDS R11 wall reproduced / Dev C diagnostic 找 SQC dcache +441% 作为 NEW #1 lever)；0 production commit；6 ranked NEW levers for R24+

### R23 派 1 Reviewer + 3 Dev (A/B/C) 并行（GPU0/1/2/3）

按 R22 R23+ priority list 执行：(A) V2 milestone-3 进一步 VMEM-cut, (B) CRR A-LDS row-major rewrite (long-pending R11 wall), (C) V2 cycle-level diagnostic via rocprofv3 PMC.

- **Reviewer (GPU0)** — Task 1: 5x 全 V2 layout baseline reverify
- **Dev A (GPU1)** — V3 preshuffle prototype: b128+b128 替换 V2 b128+b64 via wider scale packing
- **Dev B (GPU2)** — CRR A-LDS row-major Route X: ST_v2a → ST_row + load_transpose + load(A_row_reg) + memcpy reinterpret
- **Dev C (GPU3)** — V2 cycle-level diagnostic: 32 PMC counters × 6 kernels (V2/FP8 × RCR/RRR/CRR)

### Reviewer Task 1 — 5x V2 baseline reverify: stable

- V2-RCR median **3074** (std 6.20)
- V2-RRR median **3033** (std 4.14)
- V2-CRR median **2811** (std 6.83)
- FP8-RCR median **3252** (std 6.40)
- 全 SNR ≥49.5 + det 3/3 PASS
- 跨 R22→R23 drift < 0.5% (V2-RCR -4 / V2-RRR -5 / V2-CRR +79 cross-session noise); 系统 reproducibly stable
- Gaps (R23 reverify): V2-RCR -178 / -5.5%, V2-RRR -219 / -6.7%, V2-CRR -441 / -13.6%

### Dev A — V3 preshuffle prototype: ★ REJECTED, NO COMMIT ★

**Approach**: 推 V2 b128+b64 (24B/wave) 升到 b128+b128 (32B/wave) 通过 V3 preshuffle 让 B scale pack 也能 b128。

**Side commits** (NOT cherry-picked): `3fbfa416` (V3 prototype) + `aef1d03e` (bench harness) on branch `r23-a-preshuffle-v3`.

**关键 finding (negative paradigm-shift)**: **SQ_INSTS_VMEM 是 transaction count, 不是 byte count**；b64→b128 width promotion 给 0% VMEM-issue reduction (transactions 数量不变，仅 width 增大)。R18+R21 "减 VMEM-issue count" paradigm 在 V2 之后 SATURATED — 减不动了。

**5x A/B**: V3 mean **3018** vs V2 baseline mean **3091**, **Δ -92 TFLOPS / -2.99%, Welch t=-7.46** (统计显著 regression)。

**结论**: 路径完全废弃；R24+ 必须找其他 bottleneck class (SQC dcache / SPI launch / TCC working-set)，不再追 VMEM count 削减。

### Dev B — CRR A-LDS row-major Route X: ★ FAIL CORRECTNESS, NO COMMIT ★ (R11 wall 第二次 reproduced)

**Approach**: ST_v2a (col-major) → ST_row (row-major)；`load_transpose<NT>` global→LDS + `load(A_row_reg, sub)` ds_read_b128 (16B) 替换 `load_col_from_v2a_st` ds_read_b64_tr_b8 (8B)；A_col_reg 经 `__builtin_memcpy` reinterpret 复用而非 explicit `transpose(dst, tmp)` register call。

**Build**: clean compile (VGPR 249, +15 vs baseline; LDS 135168, **-4096B**; 0 spills, occ 2 — 资源 healthy)。

**Worktree**: `/tmp/wt-r23-b` on branch `r23-b-crr-a-lds`，修改 uncommitted。

**Correctness FAIL at 8192³**: SNR **-2.71 dB** (threshold 48), 97.62% partial pass-rate, det 3/3 PASS。

**完全复刻 R11 Dev M 的失败模式** (R11 also hit -2.71 dB SNR wall on same approach)。

**Root cause analysis (R23 Dev B 比 R11 进一步)**:
- `load_transpose` + `prefill_transpose_swizzled_offsets` 写入 LDS 的物理 layout 与 `ds_read_b128` 直接读取期望的 row-MMA layout 在 lane-element correspondence 上不匹配
- 97.62% partial pass 表明 systematic permutation within K-blocks 而非随机 (suggests opsel-aware vs straight-read lane-element mapping divergence)
- 三种 next-attempt: (a) element-dump kernel 验证 actual lane-element mapping, (b) keep explicit `transpose(dst, tmp)` register call (CRR_ROW_SHARED_TRANSPOSE generic path) 而非 memcpy, 或 (c) custom `load_transpose` variant matching `ds_read_b128` semantics

**Verdict**: dead-end-with-caveat；R11 + R23 两次 hit 同一墙；CRR A-LDS 需要 multi-day microbenchmark unblock，且 Dev C diagnostic 显示上限 +50-100 TFLOPS（5-10% only）；不是 R24+ top priority。

### Dev C — V2 cycle-level diagnostic via rocprofv3 PMC: ★ COMPLETE, 6 NEW LEVERS RANKED ★

32 PMC counters × 6 kernels (V2-RCR/RRR/CRR + FP8-RCR/RRR/CRR) × 4 chunks。

**mangled name verify**: `_Z29rcr_exact_8wave_scaled_kernelILb1ELi2EEv` 等等 (V2 = `<true, 2>`)；rocprofv3 substring filter `<layout>_exact_8wave_scaled_kernel` (V2) / `<layout>_exact_8wave_kernel` (FP8) confirmed 32 dispatches/chunk。

**GPU3 sclk caveat**: GPU3 idle low-power state but ramps to 1872-1995 MHz under load；`--setperflevel high` 反向 pin 到 low DPM, `auto` 是正确设置；rocprofv3 PMC mode (serialized) wall-clock degrades 但 **per-dispatch event counters 是 clock-invariant** (analysis robust)。

**6 NEW lever findings (ranked by est. upside)**:

1. **SQC_DCACHE_BUSY_CYCLES (NEW #1, R10/R12/R18 历史从未 instrumented)**:
   - V2 vs FP8 = **+441% RCR** (705k vs 130k cycles) / **+91.6% RRR** (991k vs 517k) / **+254% CRR** (571k vs 161k)
   - 根因: wave-tile preshuffle 让 scale pointer arithmetic 变 per-wave scalar code 击中 constant cache 频率剧增
   - **Lever**: precompute slab base ptrs into LDS-scalar/SGPR before K-loop OR single `s_load_dword_x4` for `(a0,a1,b0,b1)_ptr` 替换 K-loop 内 repeated SMEM loads
   - **Est upside +80-150 TFLOPS aggregate**, easy 1-day audit

2. **CRR A-LDS row-major (Dev B path)**:
   - +50% SQ_INSTS_LDS in CRR vs RRR (25.17M vs 16.78M) = 8.4M extra LDS insts/dispatch structural
   - **Bounded ~+50-100 TFLOPS** (5-10% only)
   - **R11+R23 已两次 hit dead-end SNR -2.71 dB wall** → 必须先做 layout-matching microbenchmark
   - 不是 R24+ top priority

3. **RRR TCC working-set re-tile (NEW)**:
   - V2-RRR shows **TCC_MISS +166%** (19.5M vs 7.34M) / **SQ_VMEM_TA_ADDR_FIFO_FULL +597%** vs FP8-RRR
   - 根因: B-operand fp8 fetch in RRR row-major 击中 TCC poorly
   - **Lever**: B-side block re-tile (e.g. 32×32 tiles 替换 16×128) 适配 TCC working set
   - **Est upside +50-80 TFLOPS RRR-only**

4. **V3 scale-preshuffle (Dev A path) — SATURATED on RCR**:
   - SQ_INST_LEVEL_VMEM V2 已 -41% 比 FP8；TA_ADDR_FIFO_FULL 仅 10% utilization
   - Est +0-30 TFLOPS RCR / +30-80 RRR / 0-20 CRR — Dev A REJECTED 已实测确认
   - 不是 the lever；SQC dcache 才是

5. **SPI launch occupancy fix (NEW)**:
   - 全 V2 layout SPI_RA_LDS_CU_FULL_CSN/RES_STALL_CSN/VGPR_SIMD_FULL_CSN uniformly **+18.7-18.95% over FP8**
   - VGPR pressure signal (V2 kernel 用 more VGPRs to hold wave-tile slab pack)
   - **Lever**: audit `Rpass-analysis=kernel-resource-usage` for V2 vs FP8 VGPR delta；recover 8-16 VGPRs via lifetime mgmt 解锁 occupancy
   - **Est +30-80 TFLOPS aggregate**

6. **MFMA-VALU coexec on V2-CRR (NEW)**:
   - SQ_VALU_MFMA_COEXEC_CYCLES V2-CRR vs FP8-CRR **-28.6%** (56.5M vs 79.1M)
   - 根因: V2 preshuffle scales 挤掉了 MFMA shadow 中的 helper VALU
   - **Lever**: schedule helper math (s_load → v_mov / cvt) into MFMA bubbles
   - **Est +20-40 TFLOPS CRR-only**

**Bottleneck attribution per gap pair**:
- V2-RCR (-178 TFLOPS): ~55% VMEM-issue + ~30% other (SPI/SQC) + ~10% LDS + ~5% MFMA
- V2-RRR (-219 TFLOPS): ~65% VMEM (TCC working-set) + ~15% LDS + ~15% other
- V2-CRR (-441 TFLOPS): ~45% VMEM + ~25% LDS + ~15% MFMA-coexec + ~15% other

### R23 综合产出

1. **R22 V2 三 layout 全 stable reproducible** (drift < 0.5% after 1 session)
2. **V3 preshuffle 路径 REJECTED -2.99%**: 关键 paradigm finding 是 SQ_INSTS_VMEM 是 transaction count，b64→b128 width promotion 给 0% VMEM 减少
3. **CRR A-LDS row-major 第二次 hit R11 SNR -2.71 dB wall** (Dev B confirmed naive memcpy approach 不 work)
4. **SQC_DCACHE_BUSY_CYCLES 历史从未 instrumented**: V2 paradigm 引入 +441% scalar-cache pressure
5. **R23 = 0 production commits + 1 docs commit**

### R23 confirms

- V2 paradigm 三 layout 完全 stable, no regressions across session boundaries
- VMEM-issue count cuts has saturated as a lever (R18→R21→R22 paradigm complete)
- **新瓶颈类别**: SQC dcache + SPI launch + TCC working-set 是 R24+ 焦点
- **R11 SNR wall reproducible**: CRR A-LDS layout transpose naive approach (load_transpose + memcpy) 在 R11 + R23 两次 hit -2.71 dB

### R24+ 路径（按优先级，基于 Dev C diagnostic）

1. **SQC dcache pressure cut** (NEW #1，估 +80-150 TFLOPS aggregate)：precompute slab base pointers per CTA into LDS-scalar/SGPR before K-loop；OR single `s_load_dword_x4` for `(a0,a1,b0,b1)_ptr` 替换 K-loop 内 repeated SMEM loads
2. **RRR TCC working-set re-tile** (估 +50-80 TFLOPS RRR-only)：B-side block re-tile 32×32 替换 16×128 适配 TCC working set
3. **SPI launch occupancy fix** (估 +30-80 TFLOPS)：audit V2 VGPR delta vs FP8，recover 8-16 VGPR 解锁 occupancy
4. **MFMA-VALU coexec on V2-CRR** (估 +20-40 TFLOPS CRR-only)：schedule helper math into MFMA bubbles
5. **CRR A-LDS layout-matching microbenchmark** (R11+R23 wall unblock prerequisite)：element-dump kernel mapping `load_transpose+ds_read_b128` vs `load_col_from_v2a_st` 实际 lane-element layout
6. **不要再** revisit V3 scale-preshuffle / V1 b64 / sched hints / SCALE_LDS / naive CRR A-LDS memcpy approach — 全 dead-end

### 新经验 (R23 起)

- **SQ_INSTS_VMEM 是 transaction count, 不是 byte count**：b32→b64→b128 width promotion 不减 VMEM-issue count，所以 R18+R21 paradigm "减 VMEM" 在 V2 后 saturated；下一步必须找其他 bottleneck (SQC/SPI/TCC)
- **SQC_DCACHE_BUSY_CYCLES 历史 R10/R12/R18 从未 instrumented**：V2 paradigm 引入了大幅 scalar cache pressure 是隐藏多轮的关键 overhead
- **rocprofv3 PMC mode caveat**：serialized dispatch 让 wall-clock degrade，但 per-dispatch event counters 是 clock-invariant 所以分析仍 robust（不要用 PMC mode 的 elapsed-time 做 perf 比较）
- **R11 SNR -2.71 dB wall (CRR A-LDS naive transpose)**: 不要再用 `load_transpose` + memcpy reinterpret approach；要么用 explicit `transpose(dst, tmp)` register call (CRR_ROW_SHARED_TRANSPOSE generic path), 要么写 custom `load_transpose` variant matching `ds_read_b128` semantics

## 第二十二轮评审结果 (2026-04-18) — ★ V2 推广至 RRR + CRR 双 SHIPPED ★ Dev A V2-RRR PASS BIG (+159.61 / +5.54%) cherry-picked dabeffa0; Dev B V2-CRR PASS (+20.55 / +0.76%) cherry-picked 9a0d0624; Dev C sched-hints 永久 KILLED；2 production commit；R21 V2 paradigm 完整覆盖三 layout

### R22 派 1 Reviewer + 3 Dev (A/B/C) 并行（GPU0/1/2/3）

按 R21 R22+ path priority list 第一项执行（V2 推广到 RRR/CRR）。同时 Dev C 重新 evaluate sched hints 在 V2 之上（R18 旧 baseline 已变化）。

- **Reviewer (GPU0)** — Task 1: 5x 全 layout baseline；Task 2: standby for V2 verify on each layout
- **Dev A (GPU1)** — V2-RRR milestone-1+2: mirror R21 RCR wiring，复用 Python `preshuffle_scale_matrix_mfma16_v2_rcr_a/_b`
- **Dev B (GPU2)** — V2-CRR milestone-1+2: mirror R21 RCR wiring 到 CRR fastpath
- **Dev C (GPU3)** — sched-hints diagnostic 在 V2-RCR baseline 重新 measure

### Reviewer Task 1 — baseline stable (cross-session drift < 0.5%)

- MXFP8 RCR V2 median 3071, V1 3022 (R21 reproduced)
- MXFP8 RRR V1 median ~2878
- MXFP8 CRR V1 median 2706
- FP8 RCR median 3242
- 全 SNR + det 3/3 PASS

### Dev A — V2-RRR milestone-1+2: ★ PASS BIG, COMMITTED (dabeffa0) ★

**关键 finding 解释为何 RRR 收益最大**：V1 RRR 此前 256 VGPR / **19 spills / 80B scratch** (compiler 在 V1 SRD chains + opsel templating 边界刚好溢出)；V2 collapse 到 256/0/0/0 — 这是 RRR 比 RCR (1.84%) 收益更大的根因 (5.54%)。

**dual-patch rule 简化**: RRR 单一 `load_scale_packs` lambda（无独立 warmup helper），所有 callers (main loop + pre-tail) 统一消费 V2 packs。比 RCR 简化（RCR 必须双 patch main loop + `load_scale_packs_for_pair`）。

**复用 V2-RCR Python preshuffle**: RRR A/B scale base address formulas (`a_base = br*BLK + half*HB + wm*RBM`, `b_base = bc*BLK + half*HB + wn*RBN`) byte-identical to RCR；只 A/B matrix layouts 不同，scales 完全一样。

**Production wiring**:
- `SCALE_VERSION=2` specialization of `rrr_exact_8wave_scaled_kernel`
- `dispatch_rrr_exact_8wave_scaled_v2` + `dispatch_pq_v2<RRR>` + `gemm_rrr_pq_v2` pybind
- `MXFP8_RRR_PRESHUFFLE_V2_RUNTIME` env var (default ON)

**Resource (V2)**: VGPR 256, **0 spills (V1: 19), 0B scratch (V1: 80B)**, occ 2, LDS 135 KB.

**Correctness gates** (HIP_VISIBLE_DEVICES=1, det 3/3): 256/1024/8192³ 全 SNR ≥49.5 PASS, 100% pass-rate.

**8192³ 5x A/B (Dev A GPU1)**: V1 mean ~2878, V2 mean ~3046, **Δ +168.64 TFLOPS / +5.94%, Welch t=4.76**.

### Reviewer Task 2-RRR — independent verify on GPU0: VERIFIED PASS

**5x A/B (Reviewer GPU0)**: **Δ +159.61 TFLOPS / +5.54%**, byte-exact rocprofv3 counter match.

GPU0 vs GPU1 cross-drift < 9 TFLOPS, Δ same-session 一致.

**Gap closure**: V2-RRR vs FP8 RCR ~-212 / -6.3% (vs R21 RRR estimate -371) → **43% gap close in 1 round**.

### Dev B — V2-CRR milestone-1+2: ★ PASS, COMMITTED (9a0d0624) ★

**复用 V2-RCR Python preshuffle**: scale shapes 在 RCR/CRR 之间相同 (M-major/N-major × k_blocks)。

**dual-patch rule 简化**: CRR 单一 `load_raw_scales` 加载器从 main loop / pre-tail / tail 全调用，dual-patch 简化为 single patch covering all paths.

**Production wiring**:
- `SCALE_VERSION=2` specialization of `crr_exact_8wave_scaled_kernel`
- `dispatch_crr_exact_8wave_scaled_v2` + `dispatch_pq_v2<CRR>` + `gemm_crr_pq_v2` pybind
- `MXFP8_CRR_PRESHUFFLE_V2_RUNTIME` env var (default ON)

**rocprofv3 SQ_INSTS_VMEM**: V1 212,992 → V2 180,224 = **-15.38%** (byte-exact match R21 V2-RCR).

**Resource (V2)**: VGPR 234 (V1: 232, +2), 0 spills, occ 2, LDS 139 KB.

**Correctness gates** (HIP_VISIBLE_DEVICES=2, det 3/3): 256/1024/**8192³ 49.60 dB**, 67108864/67108864 PASS.

**8192³ 5x A/B (Dev B GPU2)**: V1 median 2711.78 (std 10.61), V2 median 2732.33 (std 2.99), **Δ +20.55 / +0.76%, Welch t=4.92**.

**CRR 收益小于 RRR/RCR 的原因**: CRR baseline 已 MFMA-bound (not VMEM-issue-bound) + 已用 PIPELINE_SCALE 单 shot 4×b32；marginal cost 4 b32 vs 1 b128 + 1 b64 在 MFMA 吞吐相对小。

### Reviewer Task 2-CRR — environmental noise (GPU0 throttled)，sanity re-verify on GPU1 PASS

第一次 verify FAIL on GPU0：sclk 卡在 260MHz / mclk 2000MHz (severely throttled)；Reviewer 同时 dispatcher M_DIM rebuild 错误；Reviewer 不可能地 claim R21 RCR V2（早 ship 工作）也 FAIL → 系统性 environmental issue。

**Sanity Reviewer dispatch on GPU1** (sclk 2090MHz healthy)：byte-for-byte match Dev B's numbers，**VERIFIED PASS**.

**新规范 (R22 起)**: 任何 perf 测量前必须 `rocm-smi` 查 sclk ≥ 2GHz；GPU sclk throttle 给 misleading FAIL；dispatcher per-size rebuild required (gates on compile-time `M_DIM`)。

### Dev C — sched-hints diagnostic on V2-RCR: ABORT-DIAGNOSTIC, 路径永久 KILLED

在 V2-RCR baseline 重新 measure VMCNT-wait cycles：**V2 vmcnt wait 已 LOWER 比 FP8 (-8.8%)**.

sched hints (sched_barrier / sched_group_barrier) 没有任何可发挥空间；剩余 gap 100% 来自 structural VMEM-issue rate 而非 idle stall.

**永久关闭 sched-hints / sched_barrier / sched_group_barrier 调优方向**（R18 sched-hint REJECT + R22 V2 baseline reaffirm）.

要继续 close gap 必须做 structural VMEM-issue cuts (preshuffle V3 进一步合并？或 cache-line layout 重设计).

### R22 综合产出

1. **★ V2-RRR SHIPPED ★** +159.61 TFLOPS / +5.54% (Reviewer indep verify), gap close 43% in 1 round, commit `dabeffa0`
2. **★ V2-CRR SHIPPED ★** +20.55 TFLOPS / +0.76% (GPU1 sanity re-verify byte-exact), commit `9a0d0624`
3. **sched-hints 路径永久 KILLED** (V2 已 hide 所有 vmcnt waits, 不可能再 hide more)
4. **三 layout V2 paradigm 完整覆盖**：RCR/RRR/CRR 全 wired through `dispatch_pq_v2<L>` + `gemm_*_pq_v2` pybind + `MXFP8_*_PRESHUFFLE_V2_RUNTIME=1` default

### R22 confirms

- V2 layout paradigm 在 3 个 layout 均成功 (R21 RCR + R22 RRR + R22 CRR)
- V1→V2 收益排序 RRR (5.54%) > RCR (1.84%) > CRR (0.76%)，与 V1 baseline spills + VMEM-issue rate 排序一致
- dual-patch rule (R21 silent FAIL learning): 必须 patch BOTH main loop AND `load_scale_packs_for_pair` helper；R22-A RRR / R22-B CRR 都验证正确
- GPU benchmark sanity: 测试前必须 verify GPU sclk (`rocm-smi`) — GPU0 throttled 是 R22 CRR 第一次 verify FAIL 的根因，per-size rebuild 也是必要

### R23+ 路径（按优先级）

1. **V2 milestone-3 进一步 VMEM-cut**：V2 已 -15.4%；可能还能用 64B-aligned coalesce 或 V3 preshuffle 让 b128+b128 (32B) 替换 b128+b64 (24B)
2. **CRR 长期 gap (-15.9%)** 是新焦点（RCR/RRR 已 < 6.3%）：根因仍是 LDS-pipe / col-major A 布局 (R10 census)；A LDS row-major transpose 仍 unattempted as multi-day rewrite
3. **不要再** revisit sched hints / SCALE_LDS REPLACE / V1 layout 下 b64 — 全 dead-end

### 新经验 (R22 起)

- **复用 R21 V2 paradigm 跨 layout 极快**: RRR/CRR scale shapes 与 RCR 相同（M-major/N-major × k_blocks），所以 Python `preshuffle_scale_matrix_mfma16_v2_rcr_a/b` 直接复用，仅需 layout-specific kernel template specialization
- **GPU sclk verify 必须前置**: rocm-smi 检查 sclk ≥ 2GHz；throttled GPU 给出 misleading FAIL
- **dispatcher per-size rebuild**: dispatcher gates on compile-time `M_DIM`，必须按 size rebuild .so

## 第二十一轮评审结果 (2026-04-17) — ★ V2 SHIPPED ★ Dev A milestone-2 PASS (+55.66 TFLOPS / +1.84% / Welch t=23.33), cherry-picked 到 main (4abd4f62 + efc389ff), default-on; Dev B SCALE_LDS REPLACE 永久 KILLED；首次 21 轮 sub-200 gap 关闭（gap -228→-166）

### R21 派 1 Reviewer + 2 Dev (A/B) 并行（GPU0/1/2）

跳过 Diagnostic（R18+R19+R20 paradigm 已 triple-confirmed）。

- **Reviewer (GPU0)** — Task 1: 5x baseline；Task 2: standby for milestone PASS verify
- **Dev A (GPU1)** — preshuffle V2 milestone-2: fastpath wiring + 8192³ A/B benchmark from `r20-a-preshuffle-v2 @ f54e6dfc`
- **Dev B (GPU2)** — SCALE_LDS REPLACE milestone-1.5: defeat compiler LDS aliasing from `r20-b-scale-lds @ 5ac3229d`

### Reviewer Task 1 — baseline stable

- MXFP8 RCR median **3018.44** (std 3.00, very tight, 5 runs in [3015.46, 3022.61])
- FP8 RCR median **3243.67** (std 6.86, 5 runs in [3235.08, 3250.74])
- Gap **-225.23 / -6.94%**, drift vs R18 < 0.2% — STABLE
- 全 SNR + det 3/3 PASS

### Dev A — preshuffle V2 milestone-2: ★ PASS, COMMITTED ★

Branch `r20-a-preshuffle-v2` @ commit `1a29c562`, cherry-picked to `feat/mxfp8-only` as `efc389ff` (with foundation `4abd4f62` from R20-A milestone-1).

**Wave-tile order fix (option a from R20)**: Python `preshuffle_scale_matrix_mfma16_v2_rcr_a/_b` 在 V2 packing 前 reorder source row_groups → wave-tile gather 顺序变成 `{a0p0, a1p0, a0p1, a1p1}` (A pc=4) / `{b0p0, b1p0}` (B pc=2)，与 b128 dword 顺序匹配。

**Production wiring**:
- 新增 kernel template parameter `SCALE_VERSION` (1=V1, 2=V2)
- 新 V2 path 用 explicit `llvm_amdgcn_raw_buffer_load_b128` (A) + `llvm_amdgcn_raw_buffer_load_b64` (B) 替换 4+2 b32 chains
- 一个 wave-tile slab SRD per A and per B
- 新 dispatch `dispatch_pq_v2<RCR>` + pybind `gemm_rcr_pq_v2`
- Runtime gate `MXFP8_RCR_PRESHUFFLE_V2_RUNTIME` (default 1 = V2 on)

**关键 bug fix (worth remembering for R22+)**: 初始 256³ FAIL (SNR -0.40 dB) 真因是 `load_scale_packs_for_pair` 缺 SCALE_VERSION==2 branch；warmup/pre-tail/tail 用 V2 memory through V1 row-base pointers → garbage。**Mirror V2 b128/b64 logic 进 helper** 修复。任何 V2 推广到 RRR/CRR 都必须双 patch (main loop + helper)。

**b128/b64 emission verified** via `--offload-device-only -S`：
```
buffer_load_dwordx4 v[18:21], v175, s[28:31], s26 offen
buffer_load_dwordx2 v[192:193], v176, s[36:39], s25 offen
```

**Resource usage (V2)**: VGPR **246** (V1: 254, **-8 VGPR**), SGPR 52, 0 spills, occupancy 2 unchanged, LDS 131072 unchanged.

**Correctness gates** (HIP_VISIBLE_DEVICES=1, det 3/3 PASS, 100% pass-rate):
| Size | SNR |
|---|---|
| 256³ | 49.56 dB |
| 1024³ | 49.62 dB |
| 8192³ | **49.60 dB** |

**rocprofv3 SQ_INSTS_VMEM @ 8192³**: V1=6,815,744 → V2=5,767,168 = **-15.38%** (matches R18 model prediction exactly).

**8192³ 5x A/B (Dev A GPU1)**: V1 mean 2986.44 (std 5.77), V2 mean 3046.35 (std 4.08), Δ +59.90 / +2.0%, Welch t=18.95.

### Reviewer Task 2 — independent verify on GPU0: VERIFIED PASS

干净 worktree `/tmp/wt-r21-rev` @ HEAD `1a29c5628cbca6824f96993905c053c4ca129d4c` 重建。

**Correctness gates** (GPU0): 256³/1024³/**8192³ 49.60 dB**, 67108864/67108864 (100.00%) at 8192³, det 3/3 全 PASS。

**rocprofv3 byte-exact 匹配 Dev A**: V1 6,815,744 / V2 5,767,168 / -15.38%。

**8192³ 5x A/B (Reviewer GPU0)**:
| Path | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Median | Std |
|---|---|---|---|---|---|---|---|
| V1 | 3023.62 | 3022.43 | 3016.12 | 3021.44 | 3024.03 | **3022.43** | 3.19 |
| V2 | 3087.46 | 3078.09 | 3081.97 | 3076.06 | 3077.33 | **3078.09** | 4.63 |

**Δ median = +55.66 TFLOPS / +1.84%, Welch t = 23.33** (p<<0.001), df ≈ 7.10。

GPU0 数字略低于 Dev A GPU1 (V1 -36 / V2 -32) 是 cross-GPU drift；session-internal Δ 一致 (Dev A: +59.90 / Reviewer: +55.66)。

**Gap closure**: V2 vs FP8 RCR 3243.67 = **-165.58 / -5.10%**（vs R18 -228.42 / -7.03%；vs Reviewer Task 1 V1 -225.23 / -6.94%）→ **第一次 21 轮把 gap 关到 sub-200**。

Decision rule (Δ ≥ +30 TFLOPS AND Welch t > 3.0 AND correctness PASS): **MET on all three**, recommend MERGE.

### R21 综合产出

1. **★ V2 preshuffle SHIPPED ★** — 真实 production-impacting MXFP8 RCR 优化，+55.66 TFLOPS / +1.84% / Welch t=23.33, gap close to -165.58, 默认开启
2. **VGPR 254→246 (-8)** with same occupancy → V2 留出 future optimization headroom
3. **SCALE_LDS REPLACE 永久 KILLED** (Dev B 见下)

### R21 confirms

- V2 layout 是 21 轮唯一真实 production-impacting MXFP8 优化
- VMEM-issue rate 还是核心 bottleneck（R18 paradigm 第三次 reaffirm），但减少手段必须 native VMEM-cut（如 V2 b128/b64），不能借 LDS 中转
- V2 留 1.84% / +55.66 TFLOPS gap residual：剩余 -165.58 vs FP8 RCR

### R22+ 路径（按优先级）

1. **V2 推广到 RRR/CRR layouts**（R21 仅 wired RCR；RRR/CRR fastpath 同结构应该也能 +1-2%；effort 估 1-2 day each）
2. **V2 milestone-3：进一步压 VMEM**（V2 已 -15.4%；可能还能用 sched hints 让 b128/b64 更早 issue 来 hide 更多 latency）
3. **不要再** revisit SCALE_LDS / reuse hunt / vmcnt 假说 / 现 V1 layout 下的 b64/b128 — 全 dead-end

### R21 关键经验沉淀

- **Side-branch + Reviewer-confirm-then-cherry-pick pattern**: Dev A side branch commit → Reviewer GPU0 独立 reproduce → Decision Maker cherry-pick to main。R21 第一次成功完整跑通 production-grade workflow
- **Counter delta + correctness PASS 不足以判 GO**: SCALE_LDS counter PASS (-15.4%) 仍然 -270 TFLOPS regression。necessary but not sufficient
- **R19 linear counter-to-TFLOPS model 在 LDS path 下偏差 ~3-5x**：减 1 VMEM-issue 不等价加 1 LDS-issue + barrier cycle。仅适用 native VMEM-cut path
- **V2 推广必须 dual-patch**: main loop + warmup/pre-tail/tail helper 同时改，否则小尺寸 silent corruption (Dev A 初始 256³ FAIL 真因)

---

### R21 Dev B SCALE_LDS REPLACE milestone-1.5：**STRUCTURAL NO-GO**（Fix A + Fix B 均无效，且 SCALE_LDS 即使绕过 correctness 也是 -270 TFLOPS regression vs baseline）

### R21 Dev B (GPU2) — SCALE_LDS REPLACE milestone-1.5：**REJECT — STRUCTURAL NO-GO**

Branch `r20-b-scale-lds` @ commit `5ac3229d` 沿用，worktree `/tmp/wt-r21-b` 已 revert（纯探索，0 commit）。

**任务**：defeat R20-B 报告的 "compiler folds 6 LDS slot addresses to 3 VGPRs" miscompile，让 2048³/8192³ correctness PASS。

**Fix A (opaque pointer wrap + memory barrier)** — 试 1：
- `auto opaque_u32 = [](u32 v) { asm("v_mov_b32 %0, %1" : "=v"(out) : "v"(v)); return out; };`
- 在每个 LDS 指针 reinterpret_cast 后过一遍 opaque
- 加 `asm volatile("" ::: "memory")` barrier
- **结果**：8192³ SNR=6.62 dB，DEBUG printf 直接观察到 `cached=(00020000,00020200) direct=(817e7d81,7e7e7e7f)` ⇒ ds_read dst register **持有 LDS 地址值** 而非 loaded data。Fix A 改写 root cause 假设。

**Fix A2 (trailing s_waitcnt lgkmcnt(0) at end of load_scale_packs_from_stage)** — 试 2：
- 强制 sync 在 6 个 ds_read 之后
- **结果**：相同 cached=(00020000,...) pattern。waitcnt 来得太晚（compiler 已经 emit waitcnt 在 mfma 之前，但 dst register 在 unrolled 路径中已被读为 address）。

**Fix B (`=&v` early-clobber + per-read waitcnt)** — 试 3：
- 把 `macros::ds_read_b32` 局部展开为 inline asm，使用 `=&v` early-clobber 强制 dst VGPR ≠ smem_ptr VGPR
- 在每个 ds_read 后立即 emit `s_waitcnt lgkmcnt(0)`
- 8192³ SNR=6.63 dB **仍 FAIL**, det FAIL（max abs 0.94），TFLOPS 2340

**核心实测**（R20-B 假说 verified FALSE）：

读 `/tmp/scale_lds_fixB.s` (Fix B 编译产物) line 22639-22692 的 inner loop：
```
v_mov_b32_e32 v2, v151        ; v151 = 0x20000 + lane (slot 0 a0_pack0) ✓
v_mov_b32_e32 v3, v188        ; v188 = 0x20200 + lane (slot 2 a1_pack0) ✓
ds_read_b32 v159, v2          ; → a0_pack0 ✓
s_waitcnt lgkmcnt(0)          ; ✓
ds_read_b32 v175, v3          ; → a1_pack0 ✓
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v2, v189        ; v189 = 0x20100 + lane (slot 1 a0_pack1) ✓
v_mov_b32_e32 v3, v190        ; v190 = 0x20300 + lane (slot 3 a1_pack1) ✓
ds_read_b32 v178, v2          ; → a0_pack1 ✓
s_waitcnt lgkmcnt(0)
ds_read_b32 v176, v3          ; → a1_pack1 ✓
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v2, v191        ; v191 = 0x20800 (b0p0) ✓
v_mov_b32_e32 v3, v192        ; v192 = 0x20900 (b1p0) ✓
ds_read_b32 v171, v2          ; → b0_pack0 ✓
s_waitcnt lgkmcnt(0)
ds_read_b32 v177, v3          ; → b1_pack0 ✓
s_waitcnt lgkmcnt(0)
```
v151, v188, v189, v190, v191, v192 = **6 distinct address VGPRs** holding `0x20000 + lane`, `0x20200 + lane`, `0x20100 + lane`, `0x20300 + lane`, `0x20800 + lane`, `0x20900 + lane` (verified at lines 22317-22324)。下游 mfma_scale 在 line 22757 用 `v159, v171` (a0_p0, b0_p0)，line 22761 用 `v178, v171` (a0_p1, b0_p0) — **dst register assignments to mfma operands 全部 correct**。

**结论：R20-B 的 "compiler folds to 3 VGPRs" 假说 falsified**。assembly 没有 aliasing 问题。

**Bug source 仍未 isolate**。可能性：
1. Inter-wave LDS write/read 顺序 race（虽然 `s_barrier` 在两边都有，但 CTA-wide barrier 可能与 ds_write 的实际 sequencer commit 不同步）
2. lane permutation 不匹配 — writer wave 的 lane L 写 LDS offset L*4，reader wave 的 lane L 期望 LDS offset L*4，但 preshuffled scale 在 register 中的 lane allocation 可能与 LDS read-back 不一致
3. `volatile uint32_t scale_stage_dwords` 与 `ds_write_b32` macro 的 cache coherence — write 进 LDS bank 但 read pulls stale due to某种 SP/DC interaction

**Performance NO-GO（独立于 correctness）**：
即使绕过 correctness，**SCALE_LDS 8192³ TFLOPS = 2340 vs baseline (PIPELINE_SCALE) = 2610**，**Δ = -270 TFLOPS / -10.3% regression**。R19 Diagnostic 模型预测 +50-150 TFLOPS。模型与实测**矛盾**。可能原因：
- LDS-issue 增量（CTA-wide ds_write 16 + ds_read 6 × 8 waves = 64 ds-issues/kpair）超过 R19 估算的 +0.34M / disp
- s_barrier × 2 per kpair latency 远超 50-100 cyc 假设
- VGPR 256 (was 254) + scratch 80 bytes/lane 触发 occupancy/regalloc 退化

**最终判断**：
1. R20-B + R21-B 的 milestone-1 + milestone-1.5 共投入 ~2 day，没有定位 correctness root cause
2. 即便 fix correctness，performance 实测 **regression**，违反 R19 GO 路径前提
3. **SCALE_LDS REPLACE 路径 KILL**

### R21-B 综合产出 = 0 commit + 1 dead-end 永久封死 + R20-B 假说 falsified

1. **R20-B "compiler folds to 3 VGPRs" 假说 falsified**：实测 assembly 6 distinct VGPRs，addresses correct，dst→mfma operand mapping correct
2. **SCALE_LDS REPLACE 路径 STRUCTURAL NO-GO**：correctness root cause 未 isolate **且** performance regression -270 TFLOPS
3. **新 dead-end**：cooperative LDS scale staging + per-wave 6 ds_read pattern under current geometry (BLK=256, BK=128, RBM=64, RBN=32, WARPS_M=2, WARPS_N=4) 永久关闭

### R22+ 路径

1. **集中精力 Dev A preshuffle V2 milestone-2**（`r20-a-preshuffle-v2` branch）— 唯一剩余结构 GO 路径
2. **不要再尝试 SCALE_LDS REPLACE 任何变体**（cooperative staging + ds_write/ds_read 路径 R20-B + R21-B 已穷举主要 workaround）
3. **R20-B 报告的 LDS-aliasing miscompile 不再视为有效假说**（R21-B 实测 assembly falsified）

### R21-B 关键经验沉淀

- **Assembly 必须实测**：R20-B 凭 DEBUG printf 推论"3 VGPRs folding"是误诊。R21-B 通过 hipcc -S 直接读 device assembly，定量驳斥。下次任何"compiler miscompile"假说都要附 .s 文件 line-level 证据
- **early-clobber `=&v` 不是 silver bullet**：当下游 dst register 必须 alias 上游 src register 时（256 VGPR limit 下 register pressure 极高），early-clobber 只能让 compiler 多 emit 一次 v_mov，根本 bug 在别处
- **Counter improvement ≠ TFLOPS improvement**：R20-B SQ_INSTS_VMEM -15.4% 看上去是 R18 paradigm 实证，但 R21-B 实测 TFLOPS -10.3% regression。R19 linear model（counter 改进 → TFLOPS 改进）需要 LDS-issue cost 修正

## 第二十轮评审结果 (2026-04-17) — R19 双 GO 路径 milestone-1 实测：preshuffle V2 PASS + SCALE_LDS REPLACE counter-PASS 但 correctness FAIL；R18 paradigm 实测 reaffirm（-15.4% VMEM cut）；2 commit on side branches，0 production fastpath touch

### R20 派 2 Dev (A/B) 并行（GPU1/2 隔离）

跳过 Reviewer baseline（沿用 R18 5x median 3021.29 / FP8 RCR 3249.71 / gap -228）。

- **Dev A (GPU1)** — preshuffle V2 milestone-1：Python preshuffle + 最小 kernel reference consumer + 256³ correctness gate（fastpath 不触动）
- **Dev B (GPU2)** — SCALE_LDS REPLACE milestone-1：barrier topology 修复 + 重新启用现有 flag + rocprofv3 SQ_INSTS_VMEM kill switch

### Dev A — preshuffle V2 milestone-1：**PASS**

Branch `r20-a-preshuffle-v2` @ commit `f54e6dfc` (worktree `/tmp/wt-r20-a` 保留供 R21)。

**Files** (+549 LOC across 4 files)：
- `test_mxfp8_python.py` (+64) — `preshuffle_scale_matrix_mfma16_v2(scale_exp, pack_count)` Python 实现
- `kernel_mxfp8_layouts.cpp` (+182 at 1721-1858 + 5000-5040 pybind) — V2 lane-offset 模板 + b128/b64 loaders + verify kernel + pybind 入口，全 gated `MXFP8_RCR_PRESHUFFLE_V2_ENABLE` (default 0)
- `verify_preshuffle_v2.py` (+185 new) — CPU byte-equivalence harness
- `test_preshuffle_v2_consumer.py` (+118 new) — GPU 256³ correctness driver

**Bytewise equivalence (CPU)**：
| Size | pack_count | Match |
|---|---|---|
| 256³ | 4 / 2 | 512/512 ✅ |
| 1024³ | 4 / 2 | 8192/8192 ✅ |
| 8192³ | 4 / 2 | 524288/524288 ✅ |

**Kernel correctness gate (HIP_VISIBLE_DEVICES=1)**：
- 256³ V2 b128 (A pc=4) + b64 (B pc=2) vs V1 ref loader：**0 mismatches / 3072 compares**
- 1024³：0 mismatches / 196608 compares
- Production fastpath 256³ sanity (V2 enabled in build but fastpath untouched)：SNR 49.67 dB，100% pass-rate，证明 V2 代码完全孤立
- Resource counters：production `rcr_exact_8wave_scaled_kernel<true>` VGPR 254 / Spills 0 / LDS 131072 = identical to baseline

**R19 spec 修正**：literal `lane_byte_offset_v2 = lane_kblk*256 + lane_nonk*16` 仅适用 PC=4。PC=2 是 `lane_kblk*128 + lane_nonk*8`，k_pair stride `PC*256` not `PC*128`。已抽象为模板 `preshuffle_v2_lane_byte_offset<PC>` / `preshuffle_v2_kpair_byte_offset<PC>` 解决。

**Milestone-2 open knob (R21 必须解决)**：
- V2 slab packing 当前是 consecutive row_groups
- 但 fastpath wave-tile gather 顺序是 `{a0p0, a0p1, a1p0, a1p1}`（M offsets `{wm*RBM, +32, +HB, +HB+32}`）
- Milestone-2 要么 (a) 在 V2 packing 前 reorder source row_groups 让 wave-tile 顺序变成 `{a0p0, a1p0, a0p1, a1p1}` 匹配 b128 dword 顺序（cleaner，保持 b128 byte-stride 不变），要么 (b) 在 consumer emit dword permutation
- 现 verifier kernel 用 `__uint128_t` 直接 deref（compiler 降为普通 global load）；R21 production wiring 必须用显式 `llvm_amdgcn_raw_buffer_load_b128` intrinsic on buffer SRD（mirror `MXFP8_RCR_EXACT_PQ_PIPELINE_SCALE_ENABLE` SRD pattern at `kernel_mxfp8_layouts.cpp:2159-2186`）
- 1000-1200 LOC budget 估计仍然 realistic

### Dev B — SCALE_LDS REPLACE milestone-1：**PARTIAL（counter PASS, 2048³ correctness FAIL）**

Branch `r20-b-scale-lds` @ commit `5ac3229d` 保留；worktree `/tmp/wt-r20-b` 已删。

**Files** (+18, -2)：`kernel_mxfp8_layouts.cpp` 仅。

**R15 Dev C 真因 1 (lines ~363-373)**：SCALE_LDS 与 PIPELINE_SCALE 同时开启时两条路径都写 `*_scale_packs[]`，造成 double-write + wave-divergent VMEM arrival times at SCALE_LDS CTA-wide barrier → R15 det FAIL 根因。**Fix**：在 SCALE_LDS define block 显式 `#undef MXFP8_RCR_EXACT_PQ_PIPELINE_SCALE_ENABLE`。

**R15 Dev C 真因 2 (lines ~2225-2236)**：`sync_scale_stage_for_pair` 缺少 leading `s_waitcnt lgkmcnt(0)` + `s_barrier` 让前一轮 ds_read 在慢 wave 上排空，否则快 wave 已经覆写了 staging slabs。**Fix**：在 lambda 顶部加 leading wait + barrier。

**Gates**：
| Gate | Result | Evidence |
|---|---|---|
| 256³ MXFP8 RCR correctness | **PASS** | max_abs_err 0.0071 (= flag-OFF baseline 0.0071), SNR 49.56 dB, det PASS |
| Kill-switch SQ_INSTS_VMEM @ 8192³ | **PASS** | **5,767,168 vs baseline 6,820,000 = -15.4%** (target <6.3M) |
| 2048³ MXFP8 RCR correctness | **FAIL** | SNR 6.62 dB, deterministic 但语义错 |
| 8192³ formal benchmark | NOT RUN | per task spec correctness FAIL 跳过 |

**2048³ FAIL 根因（out-of-scope finding）**：`hipcc --offload-device-only -S` 显示 compiler 把 6 个 distinct LDS slot addresses (`block_scale_b_row_base_index(wn,0,0)` vs `(wn,1,0)` etc.) 折叠成只 3 个 address VGPRs (v2/v3/v4)，导致 6 个 ds_read 出 cached 索引值 (`00020000`, `00020100`, `00020800`, `00020900`) 而非 scale data。bug 在 K > BK*2 时浮现因为更早 slot writes 仍 alias 后续 reads。Fix 需 (a) volatile pointers / (b) opaque address casts / (c) 不同 LDS staging layout — 显式属于 milestone-1.5 而非 milestone-1。

**Resource at 8192³ build**：VGPR 256 (was 254), SGPR Spill 20, VGPR Spill 0, LDS 135168 (+4 KB scale_stage_dwords), Occupancy 2 unchanged。

**R18 paradigm 实测 reaffirm**：-15.4% VMEM-issue cut **直接证明 R19 Diagnostic 的定量模型（"减少 scale buffer_loads 数量真的会降 VMEM-issue rate"）是对的**。该路径 still GO 但需 milestone-1.5 解锁 +50-150 TFLOPS upside。

### R20 综合产出 = 2 commit on side branches + 1 production commit (docs only) + 0 fastpath touch + R18 paradigm 实测 reaffirm

1. **Preshuffle V2 layout milestone-1 PASS**：byte-equiv 全 sizes + kernel ref consumer 0 error，foundation for R21 milestone-2 (full fastpath wiring + 8192³ benchmark)
2. **SCALE_LDS REPLACE empirically validates R18 paradigm**：-15.4% VMEM-issue cut 是 R18 paradigm shift 第一次实测验证。该路径 still GO 但需 milestone-1.5 (defeat compiler LDS aliasing) 才能解锁 +50-150 TFLOPS upside
3. **Compiler LDS-aliasing miscompile** 是新发现的 hipcc/LLVM 现象

### R20 confirms

- R18 + R19 paradigm 全部经 R20 实测验证（counter 实测 -15.4%）
- Preshuffle V2 是当前唯一可执行无副作用的高 upside 路径，R21 应集中精力 milestone-2
- SCALE_LDS REPLACE 不是 dead-end，但需要 compiler workaround

### R21+ 路径（按优先级）

1. **Dev A 接力 milestone-2**：preshuffle V2 fastpath wiring + 8192³ A/B benchmark（基于 `r20-a-preshuffle-v2` branch）。必须解决 wave-tile gather 顺序 vs slab packing 顺序的失配（option a 推荐）
2. **Dev B 接力 milestone-1.5**：SCALE_LDS LDS-aliasing fix（基于 `r20-b-scale-lds` branch），优先尝试 volatile/opaque address，若仍 alias 改 LDS staging layout
3. **不要再追** reuse hunt / vmcnt 假说 / 现 V1 layout 下的 b64/b128 — R19 三类 dead-end 仍然成立

### 关键经验沉淀（R20 起）

- **Milestone gating works**：R20 前 16 轮全是 "怀疑 GO" → benchmark → fail；R20 第一次用 "milestone-1 (counter+correctness only) 才 commit production" pattern，产出 0 production breakage 但拿到了硬数据
- **Side-branch policy**：milestone-1 PASS 但未到 production-ready 时，commit 到 side branch 保留；production branch 仅 commit docs。R20 worktree `/tmp/wt-r20-a` (Dev A PASS) 保留，`/tmp/wt-r20-b` (Dev B PARTIAL) 已删但 branch 保留
- **Compiler LDS aliasing 是新已知坑**：未来任何 LDS-staging 重构都需用 volatile/opaque address 防御

## 第十九轮评审结果 (2026-04-17) — R18 paradigm shift 验证 + 两条结构性 GO 路径找到（preshuffle V2 / SCALE_LDS REPLACE）；R5/R15 dead-end 在 R18 模型下重新评估为 GO；0 commit（R20+ 实施）

### R19 派 1 Diagnostic + 2 Dev (A/B) 并行（GPU1/2/3）

跳过 Reviewer baseline（R18 刚做完 5x：MXFP8 RCR median 3021.29 / FP8 RCR median 3249.71 / gap -228 TFLOPS / -7.03%）

- **Diagnostic (GPU1)** — SCALE_LDS REPLACE 在 R18 model 下重新评估
- **Dev A (GPU2)** — preshuffle V2 layout redesign concrete prototype + integration plan
- **Dev B (GPU3)** — scale data reuse hunt (cross-iter / cross-wave / cross-half)

### Diagnostic — SCALE_LDS REPLACE: CONDITIONAL GO（R15 dead-end overturned）

R15 Dev C 的 -0.3% ~ +0.2% 估算基于 R17 vmcnt-MFMA 假说（已 falsified by R18）。R19 用 R18 cycle-counter 重新计算：

| 项目 | 当前 PIPELINE_SCALE | SCALE_LDS REPLACE | Δ |
|---|---:|---:|---|
| Per-wave per-kpair scale loads | 6 buffer_load_b32 | 2 buffer_load_b32 | -4 |
| Per-CTA per-kpair VMEM | 48 | 16 | -32 (cuts 2/3) |
| Per dispatch VMEM-issue | (full count) | (full - ~1.05M) | -1.05M |
| 关闭 +30% issue-rate gap | 100% | 33% | **67%** |
| LDS-issue 增量 | 0 | +0.34M / disp + barrier ~50-100 cyc | (small) |
| **TFLOPS recovery** | 0 | **+50-150 best case** | (vs R15 -0.3%~+0.2%) |

**Det fix path**：R15 Dev C 的"加 1 个 barrier"NOT enough。Correct fix:
- (a) 把 row_bases load 提到 CTA prologue 一次（消除 per-k_pair barrier）
- (b) re-sequence 外层 A/B barriers 让 scale barrier 嵌套同 phase
- 估 2-3 day

**GO with milestone guardrails**:
1. Milestone 1 (1 day)：实现 + rocprofv3 验证 SQ_INSTS_VMEM 6.82M → ~5.7-5.8M。如不验证 → ABORT
2. Milestone 2 (1-2 day)：修 det
3. Milestone 3 (0.5 day)：A/B 5x formal，需 ≥+30 TFLOPS 才 commit

### Dev A — Preshuffle V2 Layout Redesign：GO (highest upside)

**当前 V1 layout 限制**（R5 Dev G2 + R16 Dev C 二次 confirm）：6 scale dwords 来自 6 distinct row_groups，最小 stride 8192 B → b64/b128 不可行

**新 V2 layout**：在 wave-tile slab 内交错 row_groups，dword 级粒度
- byte 顺序：`[pack_count] x [half=2] x [k_phase_lo=2] x [lane_nonk=16] x [lane_kblk=4] x [k_pair=padded_kb/8]`
- A 侧 (RBM=64, pack_count=4)：4 dwords 连续 4-byte stride → **单 buffer_load_b128 (16B)**
- B 侧 (RBN=32, pack_count=2)：2 dwords 连续 → **单 buffer_load_b64 (8B)**
- **6 scale loads → 2，83% drop**

**TFLOPS recovery**：
- R18 +1.57M VMEM-issue / +30% rate ≈ 6 loads × 262K instr/load
- 2 loads → +0.52M / +10% issue rate
- Apply linear model：228 × (10/30) ≈ 76 TFLOPS residual gap → **recovery 150-200 TFLOPS**
- Net MXFP8 RCR 期望 **3070-3120 TFLOPS** = 95-97% of FP8 RCR
- floor 150 含 b128 wider load 2× per-issue cycle 调整，ceiling 200 best case

**Python prototype byte-verified**（worktree 已删，但 design preserved in R19 Dev A report）:
```python
# rewrite_mxfp8_v2.py: preshuffle_scale_matrix_mfma16_v2(scale_exp, pack_count)
# Smoke: rows=128, k_blocks=16, pack_count=4 → 4 packed g-bytes match
# encode_scale_matrix_raw byte-for-byte at expected (row, k_block) coords
```

**Kernel-side consumer sketch**:
```cpp
const uint32_t kpair_off = k_pair*1024 + lane_byte_offset_v2;
fp8e8m0_4_x4 packed = bit_cast<...>(buffer_load_b128(a_wave_srd, kpair_off));
a0_scale_packs[0] = packed.x; a1_scale_packs[0] = packed.y;
a0_scale_packs[1] = packed.z; a1_scale_packs[1] = packed.w;
// B: single buffer_load_b64 yielding {b0_pack[0], b1_pack[0]}.
```
`lane_byte_offset_v2 = lane_kblk*256 + lane_nonk*16` (was `lane_kblk*64 + lane_nonk*4`)

**Effort**：~3 day, 1000-1200 LOC across 5+ files
- `test_mxfp8_python.py` preshuffle (+50 LOC, parameterize pack_count)
- `kernel_mxfp8_layouts.cpp` consumer (+150/-100 LOC)
- 4 fastpath `.inc` (~200 LOC each via `build_rewrite.sh` regen)
- `rewrite_mxfp8.py` (~50 LOC)
- 3 test_mxfp8_python.py callsites

**Risk**：(1) det 低（同 byte set，只改 addr mapping）；(2) 正确性中（pack_count parameterization 紧耦合 scale tensor / kernel template）；(3) wider loads per-issue latency 高 → rocprofv3 验证

**GO recommend R20 多日实施**：sequence Dev A (Python preshuffle + ref consumer) → Dev B (fastpath asm regen + kernel rewire) → Dev C (test/benchmark/det)，~1 day each

### Dev B — Scale Data Reuse Hunt: 3 angles 全 NO

| Angle | Verdict | 原因 |
|---|---|---|
| 1 cross-iter A reuse | NO | byte_offset 推进 256 B = 64 dwords，相邻 k_pair 加载完全 disjoint dwords，0 overlap to hoist。intra-k_pair 的 k_phase=0/1 复用已被 HOIST_HI 通过 op_sel 充分利用 |
| 2 cross-wave broadcast via permlane/DPP | NO | AMD CDNA3/4 没有 inter-wave register-to-register primitive (`ds_bpermute`/`permlane16` 都是 intra-wave)。Inter-wave broadcast 必须走 LDS = SCALE_LDS path |
| 3 B scale share between A halves | NO | half=0/1 by HB=128 = distinct row_bases / distinct VMEM transactions。a0/a1 cover M-rows top/bottom 128，independent slabs not same data viewed differently |

**3 angles 全 NO confirms 唯一未证伪结构方向 = preshuffle layout 重设计**（与 Dev A V2 一致）

### R19 综合产出 = 0 commit + 2 GO 路径 + 1 paradigm shift 验证

1. **SCALE_LDS REPLACE GO** (R15 dead-end overturned under R18 model)：3-5 day, +50-150 TFLOPS, det 修复需 structural barrier re-sequence
2. **Preshuffle V2 layout GO** (R5/R16 b64 broken assumption overturned by redesigning layout)：~3 day, **+150-200 TFLOPS**, Python prototype 已 byte-verified
3. **Reuse hunt confirms** preshuffle 是唯一结构方向

### 已死的方向（R19 进一步证伪）

- 任何 cross-iter / cross-wave / cross-half scale reuse 想节省 buffer_loads（R19 Dev B 三角度 NO）
- 任何"调度 / cache 策略 / inline / prefetch / sched_barrier"角度（R3-R18 saturated；R18 Diagnostic 直接 cycle counter 证伪）
- 任何 b64/b128 coalesce with current V1 layout（R5 Dev G2 + R16 Dev C 字节 math 二次 confirm impossible）

### 新会话规范（R19 起）

1. **R20+ 推荐路径**：preshuffle V2 layout（Dev A 已 byte-verified prototype）
   - 3 day 实施，1000-1200 LOC
   - 上限 +150-200 TFLOPS（87.7% of 228 gap），Net MXFP8 RCR 期望 3070-3120 = 95-97% FP8
   - sequence Dev A (Python preshuffle + ref consumer) → Dev B (fastpath regen + kernel rewire) → Dev C (test/benchmark/det)
2. **R20+ backup 路径**：SCALE_LDS REPLACE
   - 3-5 day, +50-150 TFLOPS
   - milestone-1 (1 day) kill switch via rocprofv3 SQ_INSTS_VMEM verification
3. **不要再做 reuse hunt sprint**——R19 Dev B 已证 cross-iter/cross-wave/cross-half 全 NO
4. **不要再追 vmcnt 假说**——R18 直接 cycle counter 证伪
5. **不要再尝试 b64/b128 with current V1 layout**——byte math 二次 confirm impossible
6. 仍坚持 R15 规范：每会话必须 GPU0 baseline 重测；commit author 用 "MXFP8 Decision Maker"；worktree 必须清理

## 第十八轮评审结果 (2026-04-17) — R18 Diagnostic 推翻 R17 vmcnt-MFMA 假说；真瓶颈是 VMEM-issue 速率；1 sched_hint dead-end + AGPR feasibility downgrade；0 commit

### R18 派 1 Reviewer + 1 Diagnostic + 2 dev (A/B) 并行（GPU0/1/2/3 隔离）

- **Reviewer (GPU0)** — 5x RCR + 5x FP8 RCR baseline 重测
- **Diagnostic (GPU1)** — rocprofv3 deep cycle quantification: vmcnt vs lgkmcnt vs VMEM-issue 三方拆分
- **Dev A (GPU2)** — `MXFP8_RCR_EXACT_PQ_SCALE_SCHED_HINT_ENABLE` (sched_group_barrier 强制 scale 早 issue)
- **Dev B (GPU3)** — AGPR fused-asm Path B feasibility scout (R5 Dev F 失败真因 + 修正 path)

### Reviewer (GPU0) — baseline 重测稳定
- MXFP8 RCR 5x median **3021.29** (std 10.45, min 2998.36, max 3023.81)
- FP8 RCR 5x median **3249.71** (std 51.44, 含 2 cold-start 低尾)
- gap **-228.42 TFLOPS / -7.03%**（vs R17 -219/-6.79%）
- 所有 5+5 = 10 次 SNR 49.60-49.61 + det 3/3 PASS

### Diagnostic (GPU1) — paradigm shift：R17 假说 FALSIFIED

| Counter | MXFP8 RCR | FP8 RCR | Δ (MX-FP) |
|---|---:|---:|---:|
| GRBM_GUI_ACTIVE (cyc) | 5,888,754 | 5,043,685 | **+845K (+16.8%)** |
| SQ_VALU_MFMA_BUSY_CYCLES | 536.87M | 536.87M | 0 (identical) |
| MFMA util | **66.8%** | **78.0%** | **−11.2pp** (匹配 R17) |
| SQ_WAIT_INST_ANY (cyc) | 4.09M | 4.33M | **−241K** |
| SQ_WAIT_INST_LDS (cyc) | 0.51M | 0.59M | **−85K** |
| Derived vmcnt wait | 3.58M | 3.74M | **−157K** |
| **SQ_INSTS_VMEM** | 6.82M | 5.24M | **+1.57M (+30%)** |
| SQ_INST_LEVEL_VMEM (in-flight·time) | 60.86M | 43.77M | **+39%** |
| SQ_LDS_BANK_CONFLICT | 0 | 0 | 0 |

- **MXFP8 vmcnt + lgkmcnt 等待都比 FP8 LESS** —— compiler scheduler 已经隐藏了等待
- 真正 +845K GUI gap 来源：6 extra scale buffer_loads / K-block → +30% VMEM-issue → dispatch back-pressure (`SQ_WAIT_ANY +1.14M`)
- **TFLOPS attribution**：完美隐藏 vmcnt 期望恢复 ≈ **0 TFLOPS**（delta 为负）
- 要破 -10pp gap **必须减少 VMEM-issue 速率本身**（即减少 scale buffer_loads/K-block 的数量）

### Dev A (GPU2) — `MXFP8_RCR_EXACT_PQ_SCALE_SCHED_HINT_ENABLE` REJECT（新 dead-end）

- 在 `do_k_iter_body` 前插 `__builtin_amdgcn_sched_group_barrier(0x20, 6, 0)` + `sched_barrier(0)` 强制 6 个 scale VMEM 在 ds_read 之前 issue
- 资源完美 clean：VGPR 254 / 0 spill / occ 2 / SNR 49.56 PASS
- A/B 15 rounds GPU2：BASE 2894.93 vs EXP 2894.76，**Δ -0.006% / Welch-t -0.20**，纯噪声
- **完美 confirms Diagnostic 结论**：compiler 已经在最优位，sched hint 无可发挥空间
- **永久关闭 sched_barrier hint 方向**

### Dev B (GPU3) — AGPR fused-asm feasibility scout

- MFMA helpers map：`kernel_mxfp8_layouts.cpp:805-850` (raw + opsel_phase wrappers); `1001-1043` (per-row 2-MFMA + per-acc 8-MFMA `_impl`); 4 call sites cA/cB/cC/cD per body × KPAIR_LOOP 2 phase = **64 MFMAs/kpair**
- **R5 Dev F 失败真因找到**：4-wave fastpath (`rcr_mxfp8_4wave_fastpath.inc:213-284`) 已用 per-MFMA `asm volatile` + **`ACC16` 宏在每个 MFMA 都列出全部 16 acc tiles 为 `+a`**——R5 Dev F 只列 d0/d1，所以 compiler 在每个 MFMA 边界都重排 V↔A
- 3 条可行 path：
  - Path A 每 row 2-MFMA fuse：~80 LOC, 0.5-1 day
  - Path B 每 acc 8-MFMA fuse：~250 LOC + ACC8 macro, 2-3 days（推荐）
  - Path C 每 kpair 64-MFMA fuse：**结构不可行**（`_impl` 已被 `s_barrier` + Bs subtile loads 切开 cA/cB/cC/cD）
- HOIST_HI + KPAIR_LOOP + PIPELINE_SCALE 模板兼容性：compatible
- **Realistic upside 估算**：原本 +50-100 TFLOPS (1.5-3%)
- **R18 配合 Diagnostic 后下调至 ≈0**：AGPR fusion 解决 VGPR live-range 不解决 VMEM-issue 速率，而 R18 证明 gap 100% 来自 VMEM-issue 而非寄存器压力

### R18 综合产出 = 0 commit + 1 paradigm shift + 1 sched_hint dead-end + AGPR feasibility downgrade

1. **Paradigm shift**：R17 vmcnt-MFMA on critical path 假说 FALSIFIED。真瓶颈是 +30% VMEM-issue 速率造成的 dispatch back-pressure
2. **新 dead-end**：`MXFP8_RCR_EXACT_PQ_SCALE_SCHED_HINT_ENABLE` 永久关闭
3. **AGPR fused-asm 期望收益从 +50-100 TFLOPS 进一步下调至 ≈0**——因为不动 VMEM-issue 速率
4. **唯一仍未证伪的结构方向**：preshuffle scale layout 重设计 — 让 6 个 K-block scale loads 跨 K iteration 摊销（影响 4 fastpath + reference + 3 test caller，2-4 天工作量，上限估 ~0.5-3% TFLOPS）

### 已死的方向（R18 证伪/穷尽，下轮不要再投资）

- vmcnt-MFMA on critical path 假说（R18 Diagnostic 直接 cycle-counter 证伪：vmcnt 等待 MXFP8 反而 LESS 比 FP8）
- sched_group_barrier / sched_barrier hint（R18 Dev A：Δ -0.006% / t -0.20）
- AGPR fused-asm Path B 期望收益 ≈ 0（R18 Diagnostic：gap 是 VMEM-issue 速率，不是 VGPR live-range）
- 任何"调度 / cache 策略 / inline / prefetch"角度（R3-R18 共 16 轮，0 win）

### 新会话规范（R18 起）

1. **R17 假说 vmcnt-MFMA on critical path 已伪证**——文档已更正，下次不要再追这条
2. 真瓶颈是 VMEM-issue 速率，bottleneck 是 dispatch back-pressure 不是 idle stall
3. 唯一未证伪结构方向：preshuffle scale layout 重设计（多文件影响，上限低于 gap，但是唯一可能动 VMEM-issue 速率的杠杆）
4. **AGPR fused-asm 期望收益 ≈ 0**（除非配合 VMEM-issue 速率减少，但那需要 preshuffle layout 改动）
5. 仍坚持 R15 规范：每会话必须 GPU0 baseline 重测；commit author 用 "MXFP8 Decision Maker"

## 第十七轮评审结果 (2026-04-17) — rocprofv3 FP8-vs-MXFP8 RCR 第一次横向比较找到 vmcnt-MFMA critical-path 信号；2 条 scale-pipeline tweak 全 reject + 1 个 SMEM 神话破解；0 commit

### R17 派 1 Reviewer + 1 Diagnostic + 3 dev (A/B/C) 并行（GPU0/1/2/3 隔离）

- **Reviewer (GPU0)** — CRR 5x re-measurement 反驳 R16 的 2775.96 漂移
- **Diagnostic (GPU1)** — 第一次做 rocprofv3 FP8-RCR vs MXFP8-RCR 对比（之前 R10/R12 只比 CRR/RRR）
- **Dev A (GPU0)** — FP8 vs MXFP8 RCR 内层 ASM diff（与 Diagnostic 信号收敛）
- **Dev B (GPU2)** — KPAIR_INLINE_SCALE + SCALE_PREFETCH_N2 实验
- **Dev C (GPU3)** — "+300% SMEM" 源头 hunt via -save-temps + propose fix

### Reviewer (GPU0) — CRR 5x median 2822.30 / std 16.67
- 5/5 sample 全过 static gate 2780.28（min 2796.5, max 2841.7）
- 所有 5 次 SNR 49.60 + det 3/3 PASS
- **R16 的单次 2775.96 是 2.6σ 低端样本**，不是真实退化
- Static gate ✅ confirmed；dynamic gate ❌ -42.6 TFLOPS（与 R15 同向）

### Diagnostic (GPU1) — rocprofv3 FP8-vs-MXFP8 RCR 对比新信号
- MFMA busy% 78%→68%（**-10pp idle**）
- SQ_INSTS_SMEM 表面 "+300%"（24576 → 98304）— **后来 Dev C 证明是 cosmetic**
- SQC_DCACHE_BUSY +18%
- **关键新信号**：vmcnt(3) / vmcnt(4) waitcnt 在 MXFP8 中**紧贴 MFMA 簇之前**，FP8 中是**之后**——暗示 scale→MFMA 数据依赖在 critical path

### Dev A (GPU0) — ASM diff 收敛
确认 Diagnostic 假说。FP8 内层是 MFMA-pure；MXFP8 在每 K iter 主体之前都有一个 scale `buffer_load + vmcnt + MFMA` 的 dependency triple。**这是 -10pp MFMA util gap 的根因**（不是寄存器，不是 LDS bank conflict，不是 cache miss）

### Dev B (GPU2) — KPAIR_INLINE_SCALE + SCALE_PREFETCH_N2 全 REJECT（2 条新 dead-end）

| 实验 | flag | 结果 | 资源 | 采纳？ |
|---|---|---|---|---|
| EXP1 KPAIR_INLINE_SCALE | `MXFP8_RCR_EXACT_PQ_KPAIR_INLINE_SCALE_ENABLE=1` | formal A/B Welch-t -9.59 / **-1.95% 退化** | VGPR 254→256 + 8 spills + 36B scratch | **拒绝（永久关闭）** |
| EXP2-A SCALE_PREFETCH_N2 (B-only, BEFORE body) | `MXFP8_RCR_EXACT_PQ_SCALE_PREFETCH_N2_ENABLE=1` (variant A) | formal A/B Welch-t -8.51 / **-0.81% 退化** | clean +2 VGPR / 0 spill | **拒绝（永久关闭）** |
| EXP2-B SCALE_PREFETCH_N2 (B-only, AFTER body) | (variant B) | 主动 abort | 174 spills / 588B scratch | **拒绝（永久关闭）** |

- EXP1 根因：PIPELINE_SCALE 已经在 body 之前用 SRD/buffer_load_b32 把 scale 拿到，再 inline 一次纯属重复加载
- EXP2-A 根因：强制 per-iter A 重载（为给 prefetch slot 让位），新增的 vmcnt 又落到 critical path 上

### Dev C (GPU3) — "+300% SMEM" 神话破解（measurement artifact，不是 bottleneck）

- 通过 `-save-temps` + ISA 对比 + waves-per-block 反推
- **+73,728 extra SMEM ops 全部来自 prologue 的 `layout_globals` struct**（kernel_mxfp8_layouts.cpp:1906-1919）比 FP8 的 `rcr_exact_8wave_globals` (lean: 3 ptrs + stream，M/N/K constexpr) 多 9 个 s_load_bxxx 字段
- 8192 waves × 9 extra s_loads = **73,728 exactly**（精确匹配 perf counter）
- **量化估算**：73,728 × ~16 cycle / 1216 SIMDs / 1.7 GHz ≈ 570 ns / 10 ms kernel = **<0.006%**
- 真正的 -10pp MFMA util gap 来自 Dev A 的 per-iter scale dependency，**不是** prologue s_loads
- Refactor `layout_globals` → lean 需要碰所有 dispatch site 与 `gemm_kernel` 模板，回归风险高，benefit 低于噪声 → **不投入**
- **永久关闭 "prologue SMEM 是瓶颈" 调查方向**

### R17 综合产出 = 0 commit + 2 个新 dead-end + 1 个 myth-busting + 1 个有价值诊断

1. **新 dead-end**：`MXFP8_RCR_EXACT_PQ_KPAIR_INLINE_SCALE_ENABLE` 与 PIPELINE_SCALE 叠加 -1.95%
2. **新 dead-end**：`SCALE_PREFETCH_N2` (B-only) 变体 A -0.81% / 变体 B 174 spills
3. **Myth-busting**：FP8-vs-MXFP8 "+300% SMEM" 是 cosmetic struct artifact，runtime 占比 <0.006%
4. **有价值诊断**：vmcnt-MFMA dependency triple 是 -10pp MFMA util gap 的真因，但已知所有 prefetch 深一层尝试都触发 spill 或 vmcnt 重新落到 critical path —— 在当前 PIPELINE_SCALE 框架下 saturated
5. **R17 Reviewer 数据修正 R16**：CRR static gate 5/5 PASS，median 2822.30

### R17 confirms

MXFP8 RCR 在当前结构 + PIPELINE_SCALE pipeline 下，**所有非结构性 scale-pipeline tweak 都已饱和**。R3-R17 共 15 轮短-cycle dev fan-out 累计 0 win（R15 hygiene fix 不算优化是默认值修正）。

### 已死的方向（R17 证伪/穷尽，下轮不要再投资）

- `MXFP8_RCR_EXACT_PQ_KPAIR_INLINE_SCALE_ENABLE` 与现有 PIPELINE_SCALE 叠加（R17 Dev B EXP1）
- `SCALE_PREFETCH_N2` (B-only) 任何变体（R17 Dev B EXP2 A/B）
- 把 prologue SMEM count 当性能瓶颈调（R17 Dev C 量化证伪 <0.006% runtime）
- "再深一层 scale prefetch" / "inline 一份 scale 加载" 模式 —— PIPELINE_SCALE 已经覆盖最优，所有重叠方案都退化

### 新会话规范（R17 起）

1. **不要再做"试新 flag"或"调 prefetch / 缓存策略"sprint** —— R3-R17 反复证明短-cycle fan-out 0 win
2. 不要把 SMEM count 当 perf 信号——可能是 cosmetic struct 差异（量化估算 cycles 验证）
3. rocprofv3 FP8-vs-MXFP8 横向比较是新增的诊断手段，但 vmcnt-MFMA critical path 信号已经被 R17 EXP 证伪有可调空间
4. 如果 user 强制继续：必须**单条深度做 multi-day 结构重写**之一（AGPR fused-asm block 接续 R5 Dev F partial impl，或 preshuffle scale layout 重设计影响面 = 4 fastpath + reference + 3 test caller）
5. 仍坚持 R15 规范：每会话必须 GPU0 baseline 重测；commit author 用 "MXFP8 Decision Maker"

## 第十六轮评审结果 (2026-04-17) — 长期目标 RCR vs FP8 (-7.32%) 三条非破坏性路径全 dead-end，0 commit

### R16 派 1 Reviewer + 3 dev 并行

- **Reviewer (GPU0)** — fresh baseline，pure source defaults
- **Dev A (GPU1)** — 测试 agent_prompt.md 老 "Dev B" 段列出的 3 个未试过 flag (`PHASE_U16_CACHE / REMAP_ONCE / SCALAR_PHASE_PACKS`)
- **Dev B (GPU2)** — `-mllvm` AMDGPU compiler flag sweep（R12 Dev R timeout 路径的清算）
- **Dev C (GPU3)** — scale `buffer_load_b64` coalesce 字节级可行性研究（R5 Dev G2 的二次确认）

### R16 GPU0 baseline (Reviewer)

| Layout | TFLOPS | SNR | 状态 | vs R15 drift |
|---|---:|---|---|---:|
| **MXFP8 RCR** | 3010.57 | 49.60 PASS | ✅ | -0.17% |
| **MXFP8 RRR** | 2862.99 | 49.59 PASS | static ✅ / dynamic ✅ (+2.95) | -0.91% |
| **MXFP8 CRR** | 2775.96 | 49.60 PASS | static ❌ -4.32 / dynamic ❌ -84 | -1.93% (噪声边缘) |
| **FP8 RCR** | 3229.67 | 49.61 PASS | (gap 219 TFLOPS / 6.79%) | -0.74% |

注：CRR 这次落到 static gate 下 4.32 TFLOPS 是跨会话漂移，**非真实退化**（同代码同 commit）。FP8 RCR 第一次冷启动 1984 TFLOPS（DVFS 低功耗），后续恢复 3229。

### Dev A 结果 — 3 个老 flag 全 REJECT (架构互斥)

源码 `kernel_mxfp8_layouts.cpp:2587-2605, 2643-2661, 2698-2716, 2752-2770` 的 dispatch chain：
```
#if SCALAR_PHASE_PACKS
#elif PHASE_U16_CACHE
#elif REMAP_ONCE
#elif HOIST_HI       ← 当前 production
#elif OPSEL_PHASE
#else fallback
```

- `PHASE_U16_CACHE=1` —— 关 HOIST_HI → SNR -1.18 dB FAIL（K_PHASE templated lambda + tail path 不兼容）
- `REMAP_ONCE=1` —— VGPR 254→256 + 1 spill + 8B scratch
- `SCALAR_PHASE_PACKS=1` —— VGPR 254→256 + 1 spill + 8B scratch

**永久关闭这 3 个 flag**。HOIST_HI 是它们的 successor。已从下方 "Dev B 已有开关" 段删除。

### Dev B 结果 — `-mllvm` 30+ flag sweep 全 NO-WIN

测过：`promote-alloca-to-vector-limit`, `loop-prefetch`, `set-wave-priority`, `schedule-relaxed-occupancy`, `schedule-metric-bias={50,100}`, `kernarg-preload-count={2,4,8,16}`, `use-amdgpu-trackers`, `disable-clustered-low-occupancy-reschedule`, `disable-unclustered-high-rp-reschedule`, `enable-vopd`, `reassign-regs`, `misched-{cluster,fusion,cyclicpath}`, `enable-post-misched={true,false}`, `enable-pipeliner`, `sched-strategy={minreg,max-ilp,iterative-ilp,iterative-minreg}`, `enable-merge-m0`, `opt-vgpr-liverange`, `dce-in-ra`, `enable-amdgpu-aa`, `prealloc-sgpr-spill-vgprs`, `membound-threshold`, `disable-loop-alignment`

Top-2 quick-bench candidate formal A/B：
| Flag | Quick Δ% | Formal Welch-t | Verdict |
|---|---:|---:|---|
| `promote-alloca-to-vector-limit=2` | +0.51% | -0.01 | noise |
| `use-amdgpu-trackers` | +0.48% | -0.71 | noise |

资源 byte-identical baseline → 编译器对该 hot kernel 是 no-op。Makefile 已经默认 `-O3 -ffast-math --offload-arch=gfx950 -DKITTENS_CDNA4`。**永久关闭 `-mllvm` 调优方向**。

关键发现：
- `-enable-pipeliner` (LLVM SWP) 在 AMDGPU MFMA 循环上 silently inert
- `-enable-post-misched=false` 退化 25% → 默认开是必要的
- 所有 `sched-strategy` 替代项都退化 → 默认 GCN scheduler 是最优

### Dev C 结果 — scale b64 coalesce 字节级证伪

6 个 scale `buffer_load` 的精确映射（生产 ASM 内部循环 `.LBB16_4`）：

| Dest VGPR | SRD | 角色 |
|---|---|---|
| v183 | s[24:27] | A scale half=0 pack=0 |
| v188 | s[40:43] | A scale half=1 pack=0 |
| v190 | s[48:51] | A scale half=0 pack=1 |
| v187 | s[52:55] | A scale half=1 pack=1 |
| v184 | s[36:39] | B scale half=0 pack=0 |
| v189 | s[44:47] | B scale half=1 pack=0 |

`preshuffle_scale_matrix_mfma16` 输出 `(num_row_groups, padded_k_blocks*32)`，每个 row_group 是 8192-byte 连续 slab。6 个 scale dword 落 6 个不同 row_group，最小 dword 间距 a0p0→a0p1 = **+8192 B**（同 SRD 内）。

`buffer_load_b64` 要求 4-byte 间距 → **没有任何一对 dword 满足**。R5 Dev G2 字节 math 二次 confirm。真 b64 coalesce 需 Python preshuffle 重排（dword 级 interleave row_groups），影响 4 个 fastpath + reference + 3 个 test caller，**估 2-4 天，上限 ~0.5% TFLOPS**。**永久关闭**。

### R16 综合产出 = 4 个永久 dead-end + 0 commit

1. `PHASE_U16_CACHE / REMAP_ONCE / SCALAR_PHASE_PACKS` flag（HOIST_HI 互斥）
2. compiler `-mllvm` flag 调优（30+ flag saturated）
3. scale `buffer_load_b64` 合并（preshuffle layout 不允许，字节 math 证伪）
4. 跨 session GPU0 baseline 1-2% 漂移（R14/R15/R16 一致 confirm）

### R16 confirms

MXFP8 RCR 在当前结构下 **3010-3015 TFLOPS 是硬 ceiling**。FP8 RCR 6.79% 差距只能通过 multi-day 结构重写攻克：

**剩余仅两条结构性方向**：
- AGPR fused-asm block（R5 Dev F 有 partial impl 可接续，破坏 HOIST_HI）
- preshuffle scale layout 重设计（影响 4 fastpath + reference + 3 test caller）

A LDS row-major transpose 在 R10/R11 已半路证伪 fastpath 不兼容（pipeline ordering bug），第三条结构方向可视为 also dead unless 重写整个 LDS 同步层。

### 新会话规范（R16 起）

1. **不要再派"试新 flag"sprint** —— R3-R16 共 14 轮证明了短-cycle dev fan-out 在当前结构下 0 win
2. 如果 user 强制继续：必须**单条深度做 multi-day 结构重写**之一
3. 优先 AGPR fused-asm block（接续 R5 Dev F 的 256V→128V/128A 分配框架，但需要重写 fused asm 让 ALL MFMAs 在 1 个 asm block 内避免 V↔A shuffle 开销）
4. preshuffle scale layout 影响面太大，risk 高，benefit 上限 0.5% — 不优先
5. 仍坚持 R15 规范：每会话必须 GPU0 baseline 重测；commit author 用 "MXFP8 Decision Maker"

## 第十五轮评审结果 (2026-04-17) — defaults hygiene fix：源默认值终于匹配生产 build；R14 倒置反转

### R15 派 4 并行 agent
- **Reviewer**（GPU0 formal baseline RCR/RRR/CRR + FP8 RCR/RRR/CRR）
- **Dev A**（RRR vs RCR ASM diff，调查 R14 "RRR > RCR" 倒置）→ defaults hygiene 重大发现
- **Dev B**（`__launch_bounds__(512, 3)` 提示 SPI 预留更多 slots）
- **Dev C**（SCALE_LDS REPLACE PIPELINE_SCALE feasibility study）

### Reviewer GPU0 baseline 结果
- RCR 3000.19 / RRR 2886.93 / CRR 2838.08 / FP8 RCR 3242.25
- **历史顺序 RCR > RRR > CRR 恢复**——R14 的 "RRR > RCR 倒置" 是 cold GPU 状态 artifact

### Dev A — defaults hygiene 重大发现
- 源文件 `kernel_mxfp8_layouts.cpp` 的 `MXFP8_RCR_EXACT_PQ_KPAIR_LOOP_ENABLE` 与 `MXFP8_RCR_EXACT_PQ_PIPELINE_SCALE_ENABLE` 默认值是 `0`，但 README/SKILL 列为 production "current best" flag
- `Makefile` 与 `build_rewrite.sh` **不传任何 -D flag**，所以 fresh `make` 走 fallback 慢路径（gemm_tail_kernel 0.88 TFLOPS）
- Dev A 5+5 A/B：default-0 RCR=2810 → flip-to-1 RCR=2989 (+178 TFLOPS / +6.34%)
- Commit `98c80c20` cherry-pick 到主分支

### 决策者深度审计：扩大 hygiene fix
发现不只 KPAIR_LOOP/PIPELINE_SCALE，**所有** production flag 默认 0，包括：
- `MXFP8_RCR_EXACT_8WAVE_FAST_ENABLE 0` — 不 enable 这个，整个 RCR 8-wave 内核不会编译进二进制
- `MXFP8_RCR_EXACT_PQ_HOIST_HI_ENABLE 0`
- `MXFP8_RRR_EXACT_8WAVE_FAST_ENABLE 0`
- `MXFP8_CRR_EXACT_8WAVE_FAST_ENABLE 0`

**Commit `a8237d01`**：把 4 个 fastpath gate flag 全翻 0→1。Pure source defaults rebuild → RCR 3015.73 / RRR 2889.36 / CRR 2830.67，formal SNR 49.59-49.60 PASS, det 3/3 PASS, correctness 100%。FP8 unchanged.

### Dev B — DEAD-END（永久关闭）
`__launch_bounds__(512, 3)` 编译器**完全 ignore**——MI355X CU LDS = 160 KB，CRR 用 135-139 KB/block 已经把 occupancy cap 在 1 block/CU。VGPR/LDS/Spill/Occ 全部 byte-identical baseline。要 occ 提升必须先解决 LDS 预算（R14 已证 -34KB → -0.12% 净变化，结构上无路）

### Dev C — DEAD-END（永久关闭）
SCALE_LDS REPLACE PIPELINE_SCALE feasibility study 完成：
- R4 stack 失败的 det bug 根源：`sync_scale_stage_for_pair` 在 `do_k_iter_body` 内的 barrier topology mismatch（与 SGPR-SRD 路径并发）
- 即使完美修复 det，cost-benefit 净 −0.3% ~ +0.2%（节省的 ~6 个 SGPR-SRD scale load 已被 MFMA latency 隐藏；新增的 16 ds_write + 24 ds_read + 1 CTA-wide barrier 反而吃 50-100 cycle）
- 估计 2-4 天工作量，期望收益低于噪声 floor
- R5 三条结构性高风险路径（SCALE_LDS / AGPR fused-asm / preshuffle layout）减为两条

### R15 关键产出
1. **Defaults hygiene commit `a8237d01`** —— 源默认值终于匹配文档化生产 build；fresh `make` 不再产出慢 0.88 TFLOPS tail kernel
2. **R14 paradigm-shift 反转**：RCR > RRR > CRR 顺序恢复（R14 的"倒置"是 cold GPU + missing flag 双重测量 artifact）
3. **gate 双 PASS**：静态 gate 2780.28 RRR/CRR 全 over；动态 gate 2864.94 RRR over，CRR 差 1.14%（小幅 miss）
4. **MXFP8 RCR 真实数字 3015.73**（不是 R14 测的 2806），与 FP8 RCR 3253.80 仍差 238 TFLOPS / 7.32%（长期目标）

### 已死的方向（R15 证伪/穷尽，下轮不要再投资）
- `__launch_bounds__(_, N>1)` 调 occupancy（R15 Dev B）—— LDS 预算硬 cap 在 1 block/CU
- SCALE_LDS REPLACE PIPELINE_SCALE（R15 Dev C）—— 即使完美修复 det，期望净收益 < 噪声
- 重做 RCR vs RRR ASM diff 想找"RRR 比 RCR 快"的根因（R15 Dev A）—— 已证那是 missing flag artifact

### 新会话规范（R15 起）
1. **每次 session 必须重测 GPU0 baseline**——R14 的 GPU0 RCR=2806 vs R15 的 RCR=3000 差 194 TFLOPS，可能源于 GPU 热状态/firmware/clock，不能跨 session 直接对比
2. **不要再做"补 95% gate" sprint**——R14/R15 多次 confirm 多 GPU 多状态都达标
3. 若 user 强制继续：转向 RCR vs FP8 的 7.32% 差距（238 TFLOPS）。剩余结构性方向：AGPR fused-asm block / preshuffle scale layout 重设计 / A LDS row-major transpose（R10/R11 半路尝试但破 fastpath）
4. **commit/cherry-pick 时 author 用 "MXFP8 Decision Maker"**（无全局 git config，用 `-c user.email=... -c user.name=...`）
5. **dev worktree 一律 `/tmp/wt-<round>-<letter>`**，最后做 `git worktree remove --force` + `git branch -D <round>-dev-<letter>` 清理

## 工作流

1. Decision maker 每轮选 1–3 个最有把握的方向（不要 4 个同时开花）
2. 并行派 dev agents（Task 工具，不同 `HIP_VISIBLE_DEVICES`）
3. Dev 完成 → reviewer 门禁 → 胜出者 commit
4. 对话结束前：
   - `git status` 必须干净或只有预期的未跟踪文件
   - `TODO.md` 进度已刷新
   - `agent_prompt.md` baseline 已更新
   - SKILL 若有新 durable finding / dead-end 已补充

## 禁止清单

- Python 侧的正确性绕行（`.t().contiguous()`、host-side padding）
- commit `.so`、生成的 `.s`、`_layout_results_*.json`、`__pycache__`
- 未过 smoke 就跑 formal
- 基于短跑（iters < 50）宣布胜利
- 混用 batch timing 和 per-iter timing 的数字做对比
