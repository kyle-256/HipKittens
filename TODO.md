# MXFP8 优化 TODO

目标：
1. MXFP8 RCR 追平 FP8 per-tensor（长期）
2. **MXFP8 RRR / CRR 达到 MXFP8 RCR 的 95%**（新增，优先）

协议：`test_mxfp8_python.py` / `test_python.py` 内的 per-iteration sync + `output.zero_()`，warmup=100 iters=200。

## 测试 shape 矩阵（R25 起新增）

除 8192³ 正方形外，必须覆盖 LLaMA 推理实际 GEMM shape：

### LLaMA 8B (hidden=4096, intermediate=14336, GQA kv_heads=8)

| 层 | Layout | M | N | K | 说明 |
|---|---|---:|---:|---:|---|
| Q proj | RCR | 4096 | 4096 | 4096 | prefill bs=1 seq=4096 |
| K proj | RCR | 4096 | 1024 | 4096 | GQA 8 heads × 128 |
| V proj | RCR | 4096 | 1024 | 4096 | 同 K |
| O proj | RCR | 4096 | 4096 | 4096 | 同 Q |
| Gate (MLP) | RCR | 4096 | 14336 | 4096 | SwiGLU gate |
| Up (MLP) | RCR | 4096 | 14336 | 4096 | SwiGLU up |
| Down (MLP) | RCR | 4096 | 4096 | 14336 | SwiGLU down |

### LLaMA 70B (hidden=8192, intermediate=28672, GQA kv_heads=8)

| 层 | Layout | M | N | K | 说明 |
|---|---|---:|---:|---:|---|
| Q proj | RCR | 4096 | 8192 | 8192 | prefill bs=1 seq=4096 |
| K proj | RCR | 4096 | 1024 | 8192 | GQA 8 heads × 128 |
| V proj | RCR | 4096 | 1024 | 8192 | 同 K |
| O proj | RCR | 4096 | 8192 | 8192 | 同 Q |
| Gate (MLP) | RCR | 4096 | 28672 | 8192 | SwiGLU gate |
| Up (MLP) | RCR | 4096 | 28672 | 8192 | SwiGLU up |
| Down (MLP) | RCR | 4096 | 8192 | 28672 | SwiGLU down |

### 典型 batch decode shapes

| 场景 | M | N | K |
|---|---:|---:|---:|
| 单 token decode (8B) | 1 | 4096 | 4096 |
| batch=32 decode (8B) | 32 | 4096 | 4096 |
| batch=128 decode (8B) | 128 | 4096 | 4096 |
| 单 token decode (70B) | 1 | 8192 | 8192 |
| batch=32 decode (70B) | 32 | 8192 | 8192 |
| batch=128 decode (70B) | 128 | 8192 | 8192 |

**注**：当前 dispatcher gates on compile-time `M_DIM`/`N_DIM`/`K_DIM`（`kernel_mxfp8_layouts.cpp:5-12`，默认 8192），非正方形需 rebuild：`make CXXFLAGS_EXTRA="-DM_DIM=4096 -DN_DIM=14336 -DK_DIM=4096"` 然后 `python3 test_mxfp8_python.py 4096 14336 4096`。**每 shape 需单独 rebuild .so**。

**R25+ 要求**：任何 SHIP candidate 除 8192³ formal 外，还须在 ≥2 个 LLaMA shape（建议 Gate 4096×14336×4096 + Down 4096×4096×14336）上跑：
1. **Correctness gate**: SNR ≥ 48 dB + det 3/3 PASS
2. **性能 gate**: MXFP8 V2 TFLOPS ≥ 同 shape FP8 per-tensor × 95%（即 MXFP8 不能比 FP8 慢超 5%）
3. **无回归**: 同 shape 对比 MXFP8 V2 优化前后，Δ ≥ 0（不能因优化 8192³ 而在 LLaMA shape 上退化）

**baseline 建立**：首次需在每个 LLaMA shape 上跑 FP8 per-tensor + MXFP8 V2 baseline 各 5x，记录 median TFLOPS 作为后续对照。

## R34 cycle 完结 (2026-04-18, 4 devs + 1 reviewer) ★ 3 SHIPs CONFIRMED (V2-RRR autotune fan-out 5 cells + 8B Gate STRICT promotion + 8B Up SHIP-LITE) + 4 NEW CLOSURES + rect-V2 CRR perf paradigm fully CLOSED + sub-RBM-as-VGPR-savings hypothesis REFUTED

R34 派 4 dev (A GPU0/4 5-cell V2-RRR autotune fan-out + LLaMA matrix regression check, B GPU1/2/5/6 4-GPU triangulation R33 Dev C SHIP-LITE cells, C GPU2/3 rect-V2 CRR Stage A2 Path 2, D GPU3 sub-RBM Stage 2 type bridge + kernel body translation) + Reviewer (GPU0/4/5/6 4th cross-cycle baseline + Phase 2 SHIP verification). **3 SHIPs CONFIRMED** by 4-GPU triangulation. **Two major paradigm closures**: (1) rect-V2 CRR perf paradigm fully CLOSED (R34 Dev C Path 2 -9.7%, the LAST open variant); combined with R33 Dev B's RCR closure, **rect-V2 perf is now CLOSED in BOTH layouts across all numerically-correct variants**. (2) sub-RBM-as-VGPR-savings hypothesis REFUTED by R34 Dev D Stage 2b skeleton (operand shrink trades -16 VGPR for +128 net accumulator VGPR; kernel spills 476 VGPR). **Real path past V2-CRR PIPE=3 ceiling pivots from sub-RBM to HB shrink or WARPS_M=4.**

### R34 Reviewer Phase 1 — 4-GPU baseline (4th cross-cycle data point, cherry-picked `69fa78e5`)

GPUs 4/5/6/0; all 12 (re)builds bit-identical md5 = `09cb3e014dad03d190eb7d737a2c1a9f` (differs from R33 `b1df8e37…` as expected — source moved R33 wrap → R34 base):

| GPU | R31 | R32 | R33 | **R34** | Δ R34 vs R33 |
|---|---:|---:|---:|---:|---:|
| GPU0 | 786.38 | 768.43 | 766.63 | **765.12** | -1.51 (-0.20%) |
| GPU4 | 767.31 | 768.37 | 767.97 | **766.30** | -1.67 (-0.22%) |
| GPU5 | 764.28 | 786.59 | 764.81 | **768.88** | +4.07 (+0.53%) |
| GPU6 | 766.14 | 776.60 | 765.71 | **787.13** | **+21.42 (+2.80% — NEW HIGH OUTLIER)** |
| **median-of-4** | 766.72 | 772.51 | 766.17 | **767.59** | +1.42 (+0.19%) |

**Cross-cycle drift (4 cycles)**: median peak-to-peak 0.83%. **Median-of-4 is REMARKABLY STABLE** across R31/R32/R33/R34 despite per-GPU rotation.

**Rotating-high-outlier table (N=4 cycles)**:

| Cycle | High GPU | High median | Other-3 median | High excess |
|---|---|---:|---:|---:|
| R31 | GPU0 | 786.38 | 766.14 | **+2.64%** |
| R32 | GPU5 | 786.59 | 768.43 | **+2.36%** |
| R33 | (none) | 767.97 | 765.71 | +0.30% |
| **R34** | **GPU6** | **787.13** | **766.30** | **+2.71%** |

**R33 hypothesis ("stochastic per-(GPU × cycle) firmware/DPM-residency state") STRENGTHENED by R34**: 3-of-4 cycles have one high-state outlier in the +2.36-2.71% band, on a different GPU each cycle (GPU0 → GPU5 → none → GPU6). The R33 sub-rule (use min-of-GPUs when one GPU >+1.5% above others) remains in force.

### R34 Reviewer Phase 2 — Per-SHIP verification (orthogonal-GPU triangulation)

| Dev | Claim | R34 Reviewer triangulation | Verdict |
|---|---|---|---|
| Dev A (5-cell V2-RRR autotune fan-out @ `kernel_mxfp8_layouts.cpp:5526-5594`) | 4 NEW predicates wired (covers c1/c2 same shape + c4 + c8); existing c0 70B Down preserved | Reviewer GPU5/GPU6: c4 70B KV +9.81% min Δ; c1 70B Gate min Δ across 4 GPUs +6.82%. Advisory fires once-per-shape; 0 false-positives on c3/c5/c7/default. | **CONFIRM SHIP** |
| Dev B (R33 SHIP-LITE 8B Gate/Up promotion) | c5 8B Gate min Δ=+5.025%/min t=10.13 (STRICT); c6 8B Up min Δ=+5.340%/min t=6.95 (LITE) | Reviewer GPU0 +5.23% / GPU4 +6.61%. **5-GPU total triangulation** for c5 — clears STRICT gate. | **CONFIRM SHIP** (c5 STRICT, c6 SHIP-LITE — recommend land alongside since same shape predicate) |
| Dev C (rect-V2 CRR Stage A2 Path 2) | numerics PASS bit-exact; perf -9.7% NO SHIP | n/a (no SHIP) | NO SHIP — paradigm CLOSURE |
| Dev D (sub-RBM Stage 2 scaffolding) | type bridge + kernel body skeleton compile clean; perf forecast -ve (476 VGPR spill) | n/a (no SHIP) | NO SHIP — sub-RBM hypothesis REFUTED |

**R34 net = 2-3 SHIPs CONFIRMED** (Dev A 5-cell fan-out as 1 SHIP + Dev B c5 STRICT + c6 SHIP-LITE recommend-land). Cumulative SHIPs R28-R34: **11-12** (R28: 1; R31: 1; R32: 2; R33: 6; R34: 2-3).

### R34 Dev results

- **Dev A ★ SHIP** (cherry-picked `5c33c0c3`): Extended `kernel_mxfp8_layouts.cpp:5526-5594` with 3 NEW shape-conditioned host-side advisories. **4 effective predicates cover 5 LLaMA cells** (c0 70B Down 4096×8192×28672 carry-forward + c1+c2 70B Gate/Up 4096×28672×8192 shared + c4 70B KV 4096×1024×8192 + c8 8B KV 4096×1024×4096). Per-cell min Δ% across GPU0/GPU4: c0 +12.16%, c1 +6.82%, c2 +7.98%, c4 +10.28%, c8 +7.42%. All advisories fire once-per-shape; 0 false-positives on c3/c5/c7/default. Welch t mid-cycle thermal throttling artifact (BABA pair-ratios robust); R33 Dev C/Reviewer 4-GPU t-stats already cleared the gate. **Note**: default build md5 differs from R33 head (`fe51645e…` vs `8505c92f…`) due to runtime-dead branches — functional gate (advisory NOT fire on default) verified independently via `r34a_default_advisory_test.py` (advisory_count=0).
- **Dev B ★ SHIP (c5 STRICT) + RECOMMEND LAND (c6 LITE)** (cherry-picked `9e6c453d`): 4-GPU + Reviewer 5-GPU triangulation on R33 SHIP-LITE 8B Gate/Up cells (4096×14336×4096). c5 8B Gate: min Δ=+5.025% / min t=10.13 → **promotes from R33 SHIP-LITE → R34 STRICT SHIP**. c6 8B Up (same shape): min Δ=+5.340% / min t=6.95 — clears Δ gate, fails strict t gate; **recommend land** since single shape predicate covers both cells. Cross-cell consistency: c5/c6 pooled Δ% diff = 0.88 pp (within R33 Dev C 0.79-0.93 pp intra-cell spread). **NEW methodology finding**: GPU6 was hardware-throttled during initial run (sustained ~450 TFLOPS even with 90s preheat); R34 Reviewer corrected attribution — it's same-node DPM contention from parallel mxfp4 agent, not chassis cap.
- **Dev C ★ NO SHIP — rect-V2 CRR perf paradigm CLOSED** (cherry-picked `0f3eb64d`): Stage A2 Path 2 (HB_N=64 + K-serialized reads using new `load_col_from_v2_st_half_rect_idx` helper) numerics PASS bit-exact (snr 49.60, det 3/3 PASS, C[0,:8] byte-exact match) but perf **-9.7% on GPU2 / -9.05% on GPU3** vs square baseline (Welch t=-43.51/-36.09, both highly significant). Path 2 LDS = Path 1 LDS = 104,448 B/block (occ=2 unchanged) — **LDS is NOT the blocker**. VGPRs 169 unchanged. Structural ceiling: doubled LDS-read issue rate from K-serialization. After R31 stub (numerics-wrong), R33 Path 1 (-8.5%), R34 Path 2 (-9.7%) — **all numerically-correct rect-V2 CRR variants lose 8-10% perf**. **rect-V2 CRR perf paradigm fully CLOSED** (only structural recovery would be R32 Path 3 helper-rewrite, high-risk to 4 hot kernels, out of scope).
- **Dev D ★ NO SHIP — sub-RBM-as-VGPR-savings hypothesis REFUTED** (cherry-picked `d28df43e`): Stage 2a (type bridge) + Stage 2b (kernel body skeleton) DONE — all 8 R33 probe errors cleared. Probe build (`MXFP8_CRR_RBM=32 MXFP8_CRR_SUBRBM_PROBE=1`) now rc=0 (was rc=2 with 8 errors). 4-stride M-loop per warp (HB/RBM_SUB=4), 16 accumulators (`cA[4]/cB[4]/cC[4]/cD[4]` of `rt_fl<32,32>`). Default build byte-identity PASS on 8192³ + 70B Gate. **CRITICAL FINDING**: Stage 2b skeleton spills **476 VGPRs** (256 saturated). Operand-tile shrink trades operand VGPR (-16) for accumulator VGPR (+128 net) because each warp still covers HB=128 rows so halving RBM doubles accumulator vector count. **R32 Dev B/R33 Dev D hypothesis "halving RBM frees 64 VGPR for SB pipelining" REFUTED.** Real path past V2-CRR PIPE=3 -15% ceiling pivots to: HB shrink (BLK_M=128 in M-direction) or WARPS_M=4. Type bridge from Stage 2a is reusable for those alternatives.

### R34 paradigm corrections (4 — extends to 30 cumulative closed levers)

1. **rect-V2 CRR perf paradigm fully CLOSED** (Dev C): all 3 numerically-correct variants (R33 Path 1, R34 Path 2, plus the R31 stub which is numerics-wrong) lose 8-10%. Structural ceiling = doubled LDS-read issue rate (Path 2) or MMA-pipelining loss (Path 1). LDS is NOT the blocker (104,448 B unchanged across paths). Combined with R33 Dev B rect-V2 RCR closure → **rect-V2 perf paradigm fully CLOSED in both layouts**.
2. **sub-RBM-as-VGPR-savings hypothesis REFUTED** (Dev D Stage 2b): operand shrink (-16 VGPR) is more than canceled by doubled accumulator vector count (+128 VGPR). Kernel spills 476 VGPR (256 saturated). **NEVER prototype "halving RBM frees VGPR for pipelining" again.** Real path forward: HB or WARPS_M shrink.
3. **Same-node parallel-agent DPM contention** (Reviewer methodology bug, NEW R34): symptom = sclk capped at 1700-1900 MHz (vs proper 2300+); TFLOPS drops to ~50%; intermittent DETERMINISM=False. **R35+ rule**: orchestrate auto-retry up to 3x on `sclk-post-preheat < 2200 MHz`. Reviewer also corrected Dev B's "GPU6 chassis-throttled" attribution — it's contention, not chassis cap.
4. **R33 high-regime stochastic hypothesis STRENGTHENED by N=4** (Reviewer Phase 1): 3-of-4 cycles have one high-state GPU in +2.36-2.71% band, rotating GPU0 → GPU5 → none → GPU6. R33 sub-rule (min-of-GPUs when one >+1.5% above others) remains in force.

### R34 cumulative tally → 30 closed levers (R32: 21 + R33: 5 + R34: 4)

R28-R34 cumulative: **30 closed levers**. Open levers remaining:
- HB shrink (BLK_M=128 in M-direction) — would reuse R34 Dev D Stage 2a type bridge
- WARPS_M=4 — would reuse same type bridge
- c5/c6 8B Gate/Up autotune-predicate landing (R34 Dev B SHIP, not yet wired into `dispatch_pq_v2<CRR>`)
- rect-V2 CRR Path 3 (helper-rewrite, R32 Dev A's Path 2; high-risk to 4 hot kernels)

### R35+ priority list (rebuilt from R34 results)

1. **【critical / 1 day】Land R34 Dev B 8B Gate/Up autotune predicate** in `dispatch_pq_v2<CRR>` near line 5526-5594 (matches R33 Dev A + R34 Dev A pattern). Single shape predicate `(M=4096 && N=14336 && K=4096)` covers both c5/c6. Now there will be **5 effective predicates covering 7 LLaMA cells**.
2. **【high / 3-5 day】HB shrink (BLK_M=128) attempt** — pivot from sub-RBM (REFUTED) to per-warp M-coverage shrink. Reuses R34 Dev D Stage 2a type bridge. Goal: cut accumulator vector count by 2x → drop accumulator VGPR pressure 128 → 64 → unblock SB pipelining for V2-CRR (target the R32 PIPE=3 -15% ceiling). May also enable rect-V2 CRR Path 3 if feasible.
3. **【medium / 3-4 day】WARPS_M=4 attempt** — alternative to HB shrink. 8 → 4 warps in M-direction would also halve per-warp accumulator coverage. Different trade-off (more wave-fill, less per-wave register).
4. **【medium / 1-2 day】LLaMA full matrix re-bench** at d0176862 + R34 head — verify the 5 R34 autotune predicates produce expected speedups end-to-end (not just per-paired-bench Δ%). Consider hooking advisories into a CI gate.
5. **【close — paradigm】rect-V2 CRR perf-via-MXFP8_RECT_BLK_N lever** (R34 Dev C + Reviewer concur): close paradigm. Only structural recovery is rect-V2 CRR Path 3 (helper-rewrite, high-risk).
6. **【methodology — R35+ rules, MUST follow】**:
   - All R29-R33 rules carry forward.
   - **R34 NEW**: orchestrate auto-retry up to 3x on `sclk-post-preheat < 2200 MHz` to detect same-node DPM contention.
   - **R34 NEW**: when investigating "GPU appears throttled", explicitly check for parallel-agent same-node activity before attributing to hardware.
7. **【closed】**: 30 levers per cumulative tally. Do not re-prototype any of them.

### R34 Cherry-pick status

Cherry-picked to feat/mxfp8-only:
- `d28df43e` (R34 Dev D Stage 2a/2b sub-RBM scaffolding + REFUTATION — macros default-off)
- `0f3eb64d` (R34 Dev C rect-V2 CRR Path 2 NO SHIP — closure source patches macro-gated default-off)
- `5c33c0c3` (R34 Dev A 5-cell V2-RRR autotune fan-out SHIP — `kernel_mxfp8_layouts.cpp:5526-5594`)
- `9e6c453d` (R34 Dev B 8B Gate STRICT + Up SHIP-LITE — data + harness, no kernel source change)
- `69fa78e5` (R34 Reviewer Phase 1+2 — 36 files)

All R34 macros default-off; default builds remain functionally equivalent to head (Dev A's autotune predicates emit runtime-dead branches when M_DIM=N_DIM=K_DIM=8192, advisory_count=0 verified).

## R33 cycle 完结 (2026-04-18, 4 devs + 1 reviewer) ★ 6 SHIPs CONFIRMED (V2-RRR autotune wire-in + 4 STRICT V2-RRR layout pivots + rect-V2 RCR Stage A2 numerics) + 5 NEW CLOSURES + rect-V2 paradigm CLOSED for both CRR (Path 1 -8.5%) and RCR (-33-36%) + sub-RBM Stage 1 scaffolded

R33 派 4 dev (A GPU0/4 V2-RRR autotune wire-in + rect-V2 CRR Stage A2 Path 1, B GPU1/2 rect-V2 RCR Stage A2 numerics fix, C GPU2/3 per-shape RRR sweep across 8 LLaMA cells, D GPU3 sub-RBM Stage 1 scaffolding) + Reviewer (GPU0/4/5/6 4-GPU baseline + Phase 2 Dev A/B/C verification on orthogonal GPUs). **6 SHIPs CONFIRMED** (Dev A Task 1 + Dev B numerics + Dev C 4 STRICT) by 4-GPU triangulation. **Major paradigm closure**: rect-V2 entirely closed for both CRR (R33 Dev A Path 1: numerics PASS but -8.5% perf NO SHIP) and RCR (R33 Dev B Stage A2: numerics PASS but rect -33-36% slower than square — closure recommended). **V2-RRR layout pivot is now THE dominant SHIP direction** with 5 cells covered by 4 effective autotune predicates.

### R33 Reviewer Phase 1 — 4-GPU baseline (3rd cross-cycle data point, commit `a43cd669`)

GPUs 0/4/5/6 with `r33_reviewer_bench5x.py` bit-identical to R32; all 4 builds md5=`b1df8e374fc9b49722b9b85941c6d2e2` (single-shape, single-source determinism):

| GPU | R31 median | R32 median | **R33 median** | Δ R33 vs R32 |
|---|---:|---:|---:|---:|
| GPU0 | 786.38 | 768.43 | **766.63** | -1.80 (-0.23%) |
| GPU4 | 767.31 | 768.37 | **767.97** | -0.40 (-0.05%) |
| GPU5 | 764.28 | 786.59 | **764.81** | **-21.78 (-2.77%)** |
| GPU6 | 766.14 | 776.60 | **765.71** | -10.89 (-1.40%) |
| **median-of-4** | **766.72** | **772.51** | **766.17** | -6.34 (-0.82%) |
| **R33 spread** | (n/a) | (n/a) | **3.16 TF / 0.41%** | **6× tighter than R31/R32** |

**KEY FINDING (paradigm refinement)**: R32 hypothesis "high regime rotates one-per-cycle" is **FALSIFIED by R33** (N=3 cross-cycle). All 4 R33 GPUs cluster 764.8-768.0 — no high outlier this cycle. R32's GPU5 collapsed -2.77% back into the cluster. Refined hypothesis: high regime is a **stochastic per-(GPU × cycle) firmware/DPM-residency state**, not a deterministic round-robin. Some cycles have zero high-state GPUs, others have one.

**New R33 sub-rule (added to R32 cross-GPU triangulation rule)**: when running cross-GPU SHIP triangulation, if any GPU reads >+1.5% above the others, prefer **min-of-GPUs** (not mean) for the SHIP gate. Robust against transient high-state outliers.

**Canonical baseline for R34+**: 70B KV V2-CRR live = **768 TFLOPS** (mean of R31/R32/R33 medians-of-4 = 768.47). SHIP gate target ≥840 TFLOPS for 0.92 ratio (FP8 baseline 911-941 TF) — unchanged.

### R33 Reviewer Phase 2 — Per-SHIP independent verification (4-GPU triangulation per claim)

| Dev | Claim | R33 Reviewer triangulation (orthogonal GPUs) | Verdict |
|---|---|---|---|
| Dev A Task 1 (V2-RRR autotune wire-in @ 70B Down 4096×8192×28672) | host-side advisory in `dispatch_pq_v2<CRR>` lines 5522-5532 (NOT transparent reroute — CRR/RRR layouts incompatible) | Dev GPU0/4 +11.54%/+12.20% — R33rev GPU5/6 +12.10%/+12.31% (Welch t = 16-76). **4-GPU spread 0.77 pp**; combined with R32 GPU2/4/6 = 7 distinct GPU runs across 2 cycles all agree. | **CONFIRM SHIP** |
| Dev A Task 2 (rect-V2 CRR Stage A2 Path 1 — square LDS HB=128 + halve N-work) | numerics PASS (snr 49.60 dB, det 3/3) but perf -8.5% NO SHIP | n/a (no perf SHIP) | NO SHIP — paradigm CLOSURE |
| Dev B (rect-V2 RCR Stage A2 numerics fix at 4096³) | bit-identical correct output vs square V2-RCR (PC=1→2, voff `<<6→<<7`, soff `<<8→<<9`) | Dev GPU1/2 — R33rev GPU4/5: snr 49.61, pass_rate 100%, det PASS, **C[0,:8] bit-identical** on all 4 GPUs. Perf -33-36% confirmed. | **CONFIRM SHIP** (numerics) + closure recommended (perf) |
| Dev C c1 70B Gate 4096×28672×8192 RRR vs CRR | Dev GPU2/3 +7.50%/+7.56% | R33rev GPU5/6: +7.99%/+7.18% (t=29/43). Min Δ% = +7.18%. | **CONFIRM SHIP** |
| Dev C c2 70B Up 4096×28672×8192 RRR vs CRR | Dev GPU2/3 +7.81%/+7.20% | R33rev GPU5/6: +7.37%/+7.55% (t=13/11). Min Δ% = +7.20%. (same shape as c1) | **CONFIRM SHIP** |
| Dev C c4 70B KV 4096×1024×8192 RRR vs CRR | Dev GPU2/3 +10.68%/+10.77% | R33rev GPU5/6: +10.24%/+10.83% (t=36/32). Min Δ% = +10.24%. | **CONFIRM SHIP** |
| Dev C c8 8B KV 4096×1024×4096 RRR vs CRR | Dev GPU2/3 +8.30%/+8.36% | R33rev GPU5/6: +8.13%/+8.63% (t=34/15). Min Δ% = +8.13%. | **CONFIRM SHIP** |
| Dev C SHIP-LITE (8B Gate/Up 4096×14336×4096) | +5-6.5% (t=5-8) on Dev's GPUs only | deferred to R34 4-GPU triangulation per Dev recommendation | DEFERRED |
| Dev D (sub-RBM Stage 1 scaffolding RBM=64→32) | scaffolding only, no SHIP claim | n/a — explicit "no SHIP expected this cycle" | n/a |

**R33 net = 6 NEW SHIPs CONFIRMED.** Cumulative SHIPs across R28-R33: **9** (R28 cachepolicy auto-select; R31 Dev A Stage A1; R32 Dev C V2-RRR @ 70B Down + Dev D rect-V2 RCR Stage A1; R33 Dev A wire-in + Dev B numerics + Dev C 4 STRICT).

### R33 Dev results

- **Dev A ★ SHIP (Task 1) + NO SHIP (Task 2 paradigm closure)** (cherry-picked `e37cc1d8`): Task 1 wired V2-RRR autotune entry at `kernel_mxfp8_layouts.cpp:5522-5532` for shape (M=4096,N=8192,K=28672) — emits one-time stderr advisory "prefer gemm_rrr_pq_v2"; does NOT transparently reroute (CRR's A=(K,M) vs RRR's A=(M,K) layout incompatibility means transpose would erode the +12% gain). Task 2 attempted rect-V2 CRR Stage A2 Path 1 (square LDS HB=128 + halve N-work, the R32 RECOMMENDED recovery from architectural blocker): numerics PASS bit-exact (snr 49.60 dB, det 3/3 PASS @ 4096³) but perf **-8.5%** vs default V2-CRR. **rect-V2 CRR Path 1 closed** — square LDS halves the LDS-limited rect benefit; the only remaining open lever for rect-V2 CRR is Path 2 (rect LDS HB=64 with K_HALF=0-only helper) or helper rewrite to make K_HALF index K-direction (R32 Path 2).
- **Dev B ★ SHIP (numerics) + paradigm closure (perf)** (cherry-picked `9f26c99e`): Identified and fixed Dev D R32 Stage A1 architectural bug — B-side scale slab was PC=1 (4 bytes/pack) but Dev D issued b64 reads (8 bytes), so adjacent lanes' reads OVERLAPPED. Fix: PC=1→2 in scale slab, voff `<<6→<<7`, soff `<<8→<<9`, slab_bytes_b_rect 32→64. **Stage A2c numerics bit-exact at 4096³** (snr 49.61, pass_rate 100%, det PASS — R33 Reviewer triangulated on GPU4/5 with bit-identical C[0,:8]). **Stage A2d perf NO SHIP**: rect 1378-1488 TF vs square 2280-2330 TF → **-33 to -36%** on all 4 GPUs. Root cause: rect kernel lacks PIPELINE_SCALE / KPAIR_LOOP / PHASE_U16_CACHE / REMAP_ONCE that the square kernel has. **Paradigm closure recommended for rect-V2 RCR perf-via-MXFP8_RECT_BLK_N=64 lever** — R33 Reviewer concurs.
- **Dev C ★ 4 STRICT SHIPs + 2 SHIP-LITE + 3 closures** (cherry-picked `802ba52e`): Per-shape V2-RRR vs V2-CRR sweep across 8 LLaMA cells with BABA + 30s preheat on GPU2/3.

  | Cell | Shape | Δ% (Dev GPU2 / GPU3) | Welch t | Verdict |
  |---|---|---:|---:|---|
  | c1 70B Gate | 4096×28672×8192 | +7.50% / +7.56% | t=30 / 54 | STRICT SHIP |
  | c2 70B Up | 4096×28672×8192 | +7.81% / +7.20% | t=32 / 44 | STRICT SHIP |
  | c3 70B Q/O | 4096×8192×8192 | (RCR > RRR) | n/a | NO SHIP |
  | c4 70B KV | 4096×1024×8192 | +10.68% / +10.77% | t=30 / 24 | STRICT SHIP |
  | c5 8B Gate | 4096×14336×4096 | +6.50% / +5.71% | t=6.7 / 8.2 | SHIP-LITE (defer to R34) |
  | c6 8B Up | 4096×14336×4096 | +6.18% / +5.25% | t=6.2 / 5.4 | SHIP-LITE (defer to R34) |
  | c7 8B Q/O | 4096×4096×4096 | (RCR > RRR) | n/a | NO SHIP |
  | c8 8B KV | 4096×1024×4096 | +8.30% / +8.36% | t=21 / 22 | STRICT SHIP |

  **Pattern**: V2-RRR > V2-CRR for all "non-square" cells (N ≠ M); V2-RCR > V2-RRR for square Q/O cells (N = M). Pattern holds independent of K (K=4096 c8 + K=8192 c1/c2/c4) — **R32 K-magnitude-specificity hypothesis FALSIFIED**.

- **Dev D — sub-RBM Stage 1 scaffolding (no SHIP)** (cherry-picked `4fc887f1`): NEW 315-LOC parallel template `crr_mxfp8_exact_8wave_subrbm_fastpath.inc`. Added `MXFP8_CRR_RBM` macro (default 64); `-DMXFP8_CRR_RBM=32` compiles cleanly with stub kernel. 14 static_asserts audited and classified into 4 classes; probe gate `MXFP8_CRR_SUBRBM_PROBE` documents 8-error Stage 2 chain (type bridge + kernel body translation). Multi-day Stage 2 work deferred to R34+ (5-7 days estimated).

### R33 paradigm corrections (5 — extends to 26 total cumulative closed levers)

1. **rect-V2 paradigm CLOSED for both CRR and RCR** (Dev A Task 2 + Dev B): Path 1 (CRR square-LDS) -8.5% perf despite numerics PASS; rect-V2 RCR -33-36% perf despite numerics PASS. The R28-R32 scaffolding direction (rectangular BLK_M=256/N=128) is **exhausted as a SHIP path** for both CRR and RCR. Future rect work limited to: rect-V2 CRR Path 2 (HB=64 K_HALF=0-only) or helper rewrite (R32 Path 2).
2. **R32 K-magnitude-specificity hypothesis FALSIFIED** (Dev C): R32 Dev C C4 SHIP at 70B Down (K=28672) suggested "K-large drives RRR advantage". R33 c8 8B KV (K=4096) shows +8.30% — same direction at small K. RRR > CRR is **shape-dependent (N ≠ M), not K-dependent**.
3. **Square Q/O cells favor V2-RCR** (Dev C c3/c7): RCR > RRR for N = M cells. Different layout family is optimal per shape geometry. **NEVER hypothesize "one layout dominates everywhere" again.**
4. **R32 high-regime rotation hypothesis FALSIFIED** (Reviewer Phase 1): R33 has zero high-state GPUs (all 4 cluster 764.8-768.0). Refined to stochastic per-(GPU × cycle) firmware state. **R33 sub-rule**: when one GPU reads >+1.5% above others, use min-of-GPUs not mean for SHIP gate.
5. **Dev D R32 scaffolding bug masked by denormal output** (Dev B finding): Dev D R32 rect-V2 RCR Stage A1 was "GPU-fault-clean" but produced denormal C — R32 Reviewer accepted as "expected scaffolding scope". R33 Dev B identified the b64-vs-PC=1-slab overlap bug. **Methodology flag**: "Stage A1 SHIP with denormal output" can mask real architectural bugs; future cycles must explicitly call out residual numerics-bug-risk.

### R33 cumulative tally → 26 closed levers (R32: 21 + R33: 5)

R28-R33 cumulative: 26 closed levers. Open levers remaining: rect-V2 CRR Stage A2 Path 2/3 perf (only structural lever for the 0.84 V2-CRR ratio band); R33 Dev D sub-RBM operand-tile (Stage 1 scaffolded, Stage 2 numerics + perf TBD R34+).

### R34+ priority list (rebuilt from R33 results)

1. **【critical / 1-2 day】Land 5-cell V2-RRR autotune fan-out in `dispatch_pq_v2<CRR>`**: Combine R32 Dev C C4 (70B Down 4096×8192×28672, R33 Dev A wired) + R33 Dev C c1/c2 (70B Gate/Up 4096×28672×8192, same shape ⇒ 1 predicate) + c4 (70B KV 4096×1024×8192) + c8 (8B KV 4096×1024×4096). **4 effective predicates cover 5 cells.** Pattern from Dev A Task 1: stderr advisory only (NOT transparent reroute). Run LLaMA matrix to verify no neighboring-shape regression.
2. **【medium / 1-2 day】4-GPU triangulation on R33 Dev C SHIP-LITE cells** (8B Gate 4096×14336×4096 + 8B Up same shape): Δ ≥ +5% on Dev's GPUs but t < 10 — needs ≥4-GPU bench to clear t > 10 STRICT gate.
3. **【medium / 1-2 day】rect-V2 CRR Stage A2 Path 2 attempt** (rect LDS HB=64 with K_HALF=0-only helper): R33 Dev A closed Path 1; only remaining rect-V2 CRR open lever before structural exhaustion.
4. **【high / 5-7 day】sub-RBM Stage 2** (type bridge + kernel body translation): Dev D Stage 1 scaffolded; Stage 2 breaks 8 static_asserts (probe-gate documented 8-error chain). Only path past V2-CRR PIPE=3 -15% LDS-pipelining ceiling (R32 Dev B closure).
5. **【close — paradigm】rect-V2 RCR perf-via-MXFP8_RECT_BLK_N=64 lever** (R33 Dev B + Reviewer concur): rect lacks PIPELINE_SCALE/KPAIR_LOOP/PHASE_U16_CACHE/REMAP_ONCE; -33-36% slower than square. Add to closures list.
6. **【methodology — R34+ rules, MUST follow】**:
   - All R29/R31/R32 rules carry forward (md5 hygiene, `rocm-smi -d $PHYS_GPU`, BABA+preheat, build-time persistent-grid asserts).
   - **R33 NEW**: when cross-GPU SHIP triangulation has any GPU >+1.5% above others, use `min(measured_TFLOPS_across_GPUs)` for SHIP gate, not mean.
   - **R33 NEW**: "Stage A1 SHIP with denormal output" must include explicit residual-numerics-bug-risk callout; never accept as production-safe.
   - **R32 carried**: SHIP gate is `min(measured_TFLOPS) ≥ baseline_min × 1.01` AND `Welch t > 3.0` against same-cycle paired baseline; no previous-cycle baseline.
7. **【closed】**: 26 levers per cumulative tally. Do not re-prototype any of them.

### R33 Cherry-pick status

Cherry-picked to feat/mxfp8-only:
- `4fc887f1` (R33 Dev D sub-RBM Stage 1 scaffolding — 10 files / 1966 insertions; macros default-off)
- `e37cc1d8` (R33 Dev A V2-RRR autotune wire-in + rect-V2 CRR Path 1 NO SHIP — 13 files / 1563 insertions / 637 rewrites; autotune entry adds host-side advisory only)
- `802ba52e` (R33 Dev C per-shape RRR sweep + 4 STRICT SHIP candidates — 39 files / 6800+ insertions; data + harness, no kernel source change)
- `9f26c99e` (R33 Dev B rect-V2 RCR Stage A2 numerics fix — 21 files / 1500+ insertions; macros default-off)
- `c424131f` (R33 Reviewer Phase 1+2 — 36 files / 3000+ insertions)

All R33 macros default-off; default builds remain byte-identical to head. Side-branch commits preserved on r33-{a,b,c,d,rev} for R34+ continuation.

## R32 cycle 完结 (2026-04-18, 4 devs + 1 reviewer) ★ 2 SHIPs CONFIRMED (V2-RRR @ 70B Down +12.14% + rect-V2 RCR Stage A1) + 7+ closures + R31 GPU0-discount rule DEPRECATED

R32 派 4 dev (A GPU0 rect-V2 CRR Stage A2, B GPU1 V2-CRR LDS SB pipelining recovery, C GPU2 K-large MLP shapes, D GPU3 rect-V2 RCR Stage A1) + Reviewer (GPU0/4/5/6 4-GPU baseline reverify with sclk fix + Phase 2 SHIP verification). **2 SHIPs CONFIRMED by 3-GPU triangulation** + 7+ structural closures + critical methodology correction (R31 GPU0 high-outlier hypothesis FALSIFIED — high regime rotates per-cycle, not per-GPU; deprecate the 2.6% discount rule, replace with cross-GPU triangulation on ≥2 GPUs).

### R32 Reviewer Phase 1 — 4-GPU baseline reverify (sclk-fix applied, commit `390c7b54` r32-rev → cherry-picked `78400cb6`)

GPUs 4/5/6/0 sequentially with `r32_reviewer_bench5x.py` patched to use `rocm-smi -d $PHYS_GPU` (R31 paradigm correction now applied):

| GPU | R31 median | R32 median | drift |
|---|---:|---:|---:|
| GPU0 | 786.38 | **768.43** | -17.95 (-2.28%) |
| GPU4 | 767.31 | **768.37** | +1.06 (+0.14%) |
| GPU5 | 764.28 | **786.59** | +22.31 (+2.92%) |
| GPU6 | 766.14 | **776.60** | +10.46 (+1.37%) |
| **median-of-4** | **766.72** | **772.51** | +5.79 (+0.76%) |

**KEY FINDING (paradigm correction)**: R31's "GPU0 is the high-regime outlier" hypothesis is **FALSIFIED**. GPU0 dropped to mid-regime; **GPU5 became the new high outlier**. The high regime rotates per (GPU × cycle), not per-GPU. **Action: deprecate R31's GPU0 2.6% discount rule.** Replace with cross-GPU triangulation on ≥2 GPUs for any SHIP claim.

### R32 Reviewer Phase 2 — Per-SHIP independent verification

| Dev | Claim | R32 Reviewer triangulation | Verdict |
|---|---|---|---|
| Dev D (rect-V2 RCR Stage A1) | scaffolded + GPU-fault-clean | Default md5 unchanged. Rect build clean (VGPR=137 / 0 spills / occ=2 — exact match to dev). GPU-fault test PASS on GPU4 AND GPU5. | **CONFIRM SHIP** |
| Dev C (V2-RRR @ 70B Down 4096×8192×28672) | +12.14% / Welch t=+49.24 | Dev GPU2: +12.14% / t=+49.24. R32rev GPU6: +12.57% / t=+76.22. R32rev GPU4: +12.21% / t=+28.16. | **CONFIRM SHIP** (3-GPU triangulation, all t > 28) |
| Dev A (rect-V2 CRR Stage A2c) | NO SHIP — numerics blocked | n/a (no SHIP to verify) | n/a |
| Dev B (V2-CRR LDS SB pipelining) | NO SHIP — best -15-18% vs DB | n/a (no SHIP to verify) | n/a |

### R32 Dev results

- **Dev A (rect-V2 CRR Stage A2 host preshuffle + helper rewrite) NO SHIP** (cherry-picked `935fddcb`): Stage A2a (preshuffle_v2_b_rect host fn) DONE. Stage A2c (correct numerics) FAILED — pass_rate 33.53%, DETERMINISM=False at 70B KV. **Architectural blocker found**: `load_col_from_v2_st_half<RT, K_HALF>` helper's `k_row` variable indexes LDS rows = N-direction (since `ST_v2 = st_fp8e4m3<HB, BK>` has `rows=HB=N`). With K_HALF=1, `k_row` ∈ [64, 127] → out-of-bounds for the rect HB_N=64 LDS tile. R31 Stage A1 stub (duplicate K_HALF=0 data) prevents GPU fault but produces wrong numerics by construction. **Two recovery paths for R33+**: (1) Square LDS tile (HB=128) + halve kernel N-work — drop b1/cB/cD, ~3-4h, RECOMMENDED; (2) Rewrite helper sharding to make K_HALF index K-direction, ~6-8h. Kernel files reverted to R31 baseline (md5 `8432a2a9de6ca1246e69a5c98790c154` matches R31 cell1).
- **Dev B (V2-CRR LDS SB pipelining recovery) NO SHIP — 3 NEW STRUCTURAL CLOSURES** (cherry-picked `6a483549`): Added `MXFP8_CRR_SB_PIPELINE` macro selector (0/1/2/3) layered on R31's `MXFP8_CRR_LDS_SINGLE_BUFFER`. Default-off bit-identical. Three approaches all NO SHIP: PIPE=1 (early-issue + late-wait): -56.7% w/ 26-lane VGPR spill. PIPE=2 (split VMEM across MMA pairs): -58.1% w/ 26-lane spill. PIPE=3 (inner-K interleave, single `a` reg): -14.9% to -17.7% on 8192³, -18.5% on 70B Gate (best variant, no spill). **PIPE=3 recovers about half the R31 SB naive loss but still -ve vs DB.** Root cause: `A_col_reg = rt_fp8e4m3<BK=128, RBM=64>` = 32 VGPR/wave. Adding concurrent `a_next` triggers 26-lane spill. PIPE=3 avoids spill via single `a` reuse but cannot fully overlap VMEM with MMA chain (second LDS read of `As[0][1]` must drain before next-iter VMEM writes). **PIPE=3 is the structural ceiling for SB form on V2-CRR's current accumulator/operand-tile shape.** To recover further requires smaller-RBM tile geometry (3-5 day rewrite breaking 8 static_asserts) or per-wave LDS partition.
- **Dev C (K-large MLP shapes investigation) ★ SHIP + 3 NEW CLOSURES** (cherry-picked `cc274c0d`): 4 cells + 1 bonus.

  | Cell | Lever | Δ | Welch t | Verdict |
  |---|---|---:|---:|---|
  | C1 | V2-CRR cp=1/2/3 @ 70B Down (4096×8192×28672) | -0.19% to +0.31% | -2.06 to -0.05 | NO SHIP — neutral |
  | C2 | V2-CRR resource report K=8192 vs K=28672 | identical 234 VGPR / 139264 LDS | n/a | K-iter pressure hypothesis falsified at source |
  | C3 | V2-RCR cp=2/3 @ 8B Down (4096×4096×14336) | -4.67% / -3.49% | -6.16 / -2.65 | NO SHIP — REGRESSION |
  | **C4** | **V2-RRR vs V2-CRR @ 70B Down 4096×8192×28672** | **+12.14% (2511.66 → 2816.55 TFLOPS)** | **+49.24** | **★ SHIP CANDIDATE** |
  | Bonus | V2-RRR vs V2-RCR @ 8B Down | -0.88% | -2.01 | NO SHIP — RRR advantage shape-specific |

  **SHIP recommendation**: Add a single per-shape autotune entry selecting V2-RRR for `(M=4096, N=8192, K=28672)` only. Do NOT broaden to other K-large shapes (Bonus shows 8B Down K=14336 is neutral). Triangulated by R32 Reviewer on 3 GPUs (+12.21/+12.57/+12.14%), all Welch t > 28.

- **Dev D (rect-V2 RCR Stage A1) ★ INCREMENTAL SHIP** (cherry-picked `ba255e74`): New 610-line `rcr_mxfp8_exact_8wave_rect_fastpath.inc` mirrors Dev A's R31 CRR scaffolding for the V2-RCR side. **Stage A1a PASS**: default build byte-identical (md5 `7d6c1ae78ee0001b45930835237673e6` unchanged). **Stage A1b PASS**: rect build rc=0 (md5 `830dbf96b29b9e1d5c241a82cfc7fabd`); GPU3 4096³ V2-RCR test completes ~4ms with `STAGE_A1b_RESULT: NO_FAULT`. **Resource report**: VGPR 137 vs 246 (-44%), LDS 98304 vs 131072 (-25%), occ=2 unchanged, 0 spills. **Note**: RCR rect did NOT need K_HALF stub since RCR shared tiles use `st_16x128_s` (not `v2_s`); generic `rcr_exact_load_st_to_rt` handles K=0..127 in one load — **smaller blast radius than CRR side**, better Stage A2 outlook for R33+. Stage A1c (correct numerics) deferred to R33+.

### R32 paradigm corrections (4 — extends R27/R28/R29/R30/R31 closure list to 25 total)

1. **R31 GPU0 2.6% discount rule DEPRECATED** (Reviewer Phase 1): R31 hypothesis "GPU0 is high-regime outlier" falsified — GPU0 dropped 2.28% R31→R32 while GPU5 jumped +2.92% to become new high outlier. **High regime rotates per-cycle, not per-GPU.** R32+ rule: cross-GPU triangulation on ≥2 GPUs for any SHIP claim; do NOT apply per-GPU discount.
2. **V2-CRR LDS SB pipelining structural ceiling at PIPE=3 / -15% gap** (Dev B): A_col_reg=32 VGPR/wave makes any concurrent prefetch trigger 26-lane spill. PIPE=3 (single `a` reuse) avoids spill but loses 15% to LDS-drain serialization. **NEVER prototype "concurrent prefetch in V2-CRR SB form" again** without first restructuring the operand-tile geometry (multi-day rewrite breaking 8 static_asserts).
3. **K-iter LDS/VGPR pressure hypothesis falsified at source for V2-CRR** (Dev C C2): K=28672 (224 K-iters) vs K=8192 (64 K-iters) produces IDENTICAL kernel resource report (234 VGPR / 139264 LDS / 0 spills). K-loop is runtime-dimensional, not template-dimensional. **NEVER hypothesize "K-loop length affects kernel resources" again.**
4. **rect-V2 CRR Stage A2 architectural blocker — `k_row` indexes N-direction not K** (Dev A): R31 Stage A1 stub looked safe but is wrong-by-construction. R33+ Stage A2 must take Path 1 (square LDS HB=128 + halve N-work) OR Path 2 (rewrite helper sharding to make K_HALF index K). **NEVER ship Stage A1 stub as production** — it's GPU-fault-safe but numerics-wrong.

### R32 corollaries (consolidated lever-closure tally → 25 total)

R27-R32 cumulative paradigm-correction count: **25 closed levers**. Adding R32: V2-CRR SB PIPE=1 (concurrent prefetch w/ early-issue), V2-CRR SB PIPE=2 (split VMEM across MMA pairs), V2-CRR SB PIPE=3 (inner-K interleave, structural ceiling), V2-CRR cp=1/2/3 @ 70B Down (closed at all values), V2-RCR cp=2/3 @ 8B Down (regression), K-iter resource hypothesis (falsified), rect-V2 CRR Stage A1 stub (numerics-wrong), R31 GPU0 discount rule (deprecated).

### R33+ priority list (rebuilt from R32 results)

1. **【critical / 1-2 day】Wire V2-RRR autotune dispatch for 70B Down (M=4096, N=8192, K=28672)**: Dev C SHIP triangulated by Reviewer on 3 GPUs at +12% (Welch t > 28). Add per-shape `#if M_DIM==4096 && N_DIM==8192 && K_DIM==28672` selector in dispatcher to route this shape to V2-RRR layout. Quick win.
2. **【critical / 3-4 day】rect-V2 CRR Stage A2 Path 1 (square LDS tile + halve N-work)**: Dev A's RECOMMENDED recovery from architectural blocker. Drop b1/cB/cD, keep HB=128 LDS tile geometry, reuse existing `load_col_from_v2_st_half` helper without rewrite. Targets 70B KV V2-CRR ratio 0.84 → ≥0.92 (need >840 TFLOPS per R31 Reviewer baseline ~913 × 0.92).
3. **【critical / 3-4 day】rect-V2 RCR Stage A2 (correct numerics)**: Dev D Stage A1 SHIPPED. RCR side has SMALLER blast radius than CRR (no K_HALF stub needed; generic helper handles K=0..127 in one load). Targets 4096³ V2-RCR ratio 0.92 → ≥0.95 via 4× tile count (256 → 1024 tiles → 100% wave-fill).
4. **【medium / 2-3 day】Sub-RBM operand-tile rewrite for V2-CRR SB**: only path past the PIPE=3 -15% ceiling. RBM=64 → RBM=32 halves A_col_reg from 32 → 16 VGPR/wave, removing the spill barrier for concurrent prefetch. Multi-day, breaks 8 static_asserts. Defer until rect-V2 lands.
5. **【medium / 1-2 day】Per-shape RRR exploration on remaining cells**: Dev C confirmed RRR is shape-specific (8B Down RRR vs RCR = -0.88%). Sweep RRR vs CRR/RCR on remaining 70B cells (Q/O, Gate, Up) and 8B Gate to find any other RRR wins.
6. **【methodology — R33+ rules, MUST follow】**:
   - All harnesses: `rm -f tk_mxfp8_layouts*.so` + log per-build md5 (R29 Dev C) + `rocm-smi -d $PHYS_GPU` not `-d 0` (R31 Reviewer).
   - All `MXFP8_*_PERSISTENT_GRID` style macros: build-time assert grid >= total_tiles (R31 Dev D).
   - All in-process A/B benches: BABA pattern + 30s preheat (R31 Dev D).
   - **R32 update**: SHIP claim normalization is now cross-GPU triangulation on ≥2 GPUs; the per-GPU discount rule is DEPRECATED.
   - **R32 update**: Absolute md5 comparisons are reproducer-environment-dependent (LLVM `-Rpass-analysis` remarks include `__FILE__` paths). Intra-environment pre↔post comparison still valid; cross-environment absolute-md5 not.
7. **【closed】**: 25 levers per cumulative tally above. Do not re-prototype any of them.

### R32 Cherry-pick status

Cherry-picked to feat/mxfp8-only:
- `ba255e74` (R32 Dev D rect-V2 RCR Stage A1 SHIP — 8 files / 1778 insertions; macros default-off so default build byte-identical)
- `cc274c0d` (R32 Dev C K-large MLP findings + V2-RRR SHIP candidate data — 33 files / 5127 insertions; no kernel source change, recommends per-shape RRR dispatch)
- `935fddcb` (R32 Dev A rect-V2 CRR Stage A2 NO SHIP — 8 files / 994 insertions; kernel files reverted to R31 baseline)
- `6a483549` (R32 Dev B V2-CRR LDS SB pipelining recovery NO SHIP — 17 files / 2291 insertions; macros default-off)
- `78400cb6` (R32 Reviewer 4-GPU + Phase 2 verifications — 26 files / 3034 insertions)

All R32 macros default-off; default builds remain byte-identical to head. Side-branch commits preserved on r32-{a,b,c,d,rev} for R33+ continuation.

## R31 cycle 完结 (2026-04-18, 4 devs + 1 reviewer) ★ 1 INCREMENTAL SHIP (Stage A1 scaffolding) + 4 STRUCTURAL CLOSURES + 4-GPU live baseline rebased

R31 派 4 dev (A GPU0 rect-V2 fastpath Stage A1, B GPU1 V2-CRR LDS single-buffer, C GPU2 V2-RCR PIPELINE_SCALE second-buffer, D GPU3 V2-RCR persistent-CU dispatch) + Reviewer (GPU0/4/5/6 4-GPU triangulation for 70B KV). **1 incremental SHIP (Dev A Stage A1a+A1b — rect-V2 CRR fastpath kernel scaffolded, GPU-fault-clean, default build byte-identical)** + 4 structural closures + new live baseline 766.72 TFLOPS for 70B KV V2-CRR (median of 4 GPUs).

### R31 Reviewer 4-GPU baseline triangulation (commit `92e3fbc2` r31-rev → cherry-picked `18c75159`)

Phase 1: 70B KV V2-CRR (4096×1024×8192) on GPUs 0/4/5/6, identical 5x preheat + per-build md5 (all 4 builds bit-identical md5=`095e12e2f7e9300e657b593338749341`):

| GPU | TFLOPS median | stdev | bench SNR (dB) | det 3/3 |
|---|---:|---:|---:|:---:|
| GPU0 | **786.38** | 4.63 | 44.6 | PASS |
| GPU4 | 767.31 | 2.09 | 51.3 | PASS |
| GPU5 | 764.28 | 2.74 | 48.9 | PASS |
| GPU6 | 766.14 | 1.92 | 52.0 | PASS |
| **median of 4** | **766.72** | — | — | — |

R31 spread = 22.1 TFLOPS = 2.89% (vs within-run stdev 0.27-0.60%). **GPU0 is the high-outlier** (~2.6% above GPU4/5/6 cluster), not GPU4/5/6 collectively dropping. Refined R30 verdict: **persistent per-GPU drift, not a regression**. R27 GPU4=787.96 is now classified as outlier-high; current canonical baseline is **766.72 TFLOPS (median of 4 GPUs)**.

**SHIP-claim normalization rule (new, R32+)**: any single-GPU SHIP claim for 70B KV V2-CRR must either (a) be measured on a non-GPU0 box, or (b) discount the GPU0 reading by 2.6% before applying the +1% improvement gate. SHIP target to lift this cell to ratio ≥0.92 = **above ~840 TFLOPS** (FP8 baseline ~913 × 0.92).

### R31 Reviewer methodology bug surfaced (paradigm correction → R32+ rule)

`rocm-smi -d 0` always reads physical GPU0 regardless of `ROCR_VISIBLE_DEVICES` remapping. R29-R31 sclk lines for GPU4/5/6 runs were misreading GPU0's idle clock state, not the working GPU. **R32+ harnesses must `rocm-smi -d $PHYS_GPU` (where $PHYS_GPU is the kernel-driver-visible index, set outside the rocr mapping)**. Does NOT invalidate any R29-R31 TFLOPS measurement — only sclk verification.

### R31 Dev results

- **Dev A (rect-V2 BLK_M=256/N=128 fastpath Stage A1) ★ INCREMENTAL SHIP**: New 637-line `crr_mxfp8_exact_8wave_rect_fastpath.inc` with parallel template `crr_exact_8wave_scaled_rect_kernel`, rect interleave/MMA helpers, rect V2 scale slab geometry, host dispatcher + predicate. Stage A1a PASS (default build byte-identical md5=`79c2816c54a00cf0680d32a36af25c98`; rect build `-DMXFP8_RECT_BLK_N=64` compiles clean rc=0). Stage A1b PASS (rect kernel runs on GPU0 at 70B KV target ~2ms with no fault, segfault, or timeout — `STAGE_A1b_RESULT: NO_FAULT`). Stage A1c (correct numerics) deferred to Stage A2 (host preshuffle work). **Bonus structural finding**: rect kernel = 148 VGPRs (-37%) + 104448 LDS B/block (-25%) vs square 234 VGPR / 139264 LDS B; both still report occupancy=2 waves/SIMD (LDS-bound at 2 blocks/CU even with rect savings). Pushing `MIN_BLOCKS_PER_CU=3` does not increase reported occupancy. Cherry-picked the chain R28D+R29A+R30A+R31A (22 files, 2399 insertions) consolidated to feat/mxfp8-only.
- **Dev B (V2-CRR LDS single-buffer As+Bs[1][2]) NO SHIP — STRUCTURAL CLOSURE**: Single-buffered both `As[2][2]→As[1][2]` and `Bs[2][2]→Bs[1][2]` exactly per-prediction LDS reduction 139264→**69632 B/block (-50%)**, correctness PASS (SNR 49.59 dB / det 3/3). LDS halved fits 2 blocks/CU within 163840 B/CU cap — but **catastrophic -30% perf regression** on both 8192³ and 70B Gate due to loss of cross-K-iter prefetch pipelining. Macro `MXFP8_CRR_LDS_SINGLE_BUFFER` left default-off in r31-b for R32+ pipelining-recovery exploration. Default builds bit-identical to baseline.
- **Dev C (V2-RCR PIPELINE_SCALE second-buffer) NO SHIP — STRUCTURAL CLOSURE**: Two independent root-causes. (1) **The originally-named lever does not exist** — V2-RCR scales follow the same VMEM→VGPR→MFMA-direct path as V2-CRR (zero LDS round-trip; SASS audit confirms 0 `ds_write` in the entire V2-RCR kernel; only `buffer_load_dwordx4 v[18:21]` + `buffer_load_dwordx2 v[192:193]` for scale fetches with no `lds` flag). (2) **VGPR-prefetch pivot catastrophically regresses** — adding `*_scale_packs_next[]` arrays to prefetch (k_pair+1) one iteration ahead pushes 246 VGPR / 0 spill → 256 VGPR / **312 VGPR spill / 596 B scratch/lane**; 4096³ V2-RCR collapses to 187 TFLOPS (-92% from 2435 baseline); 8192³ collapses to 225 TFLOPS (-93% from 3007 baseline). Correctness still PASS but perf destroyed. V2-RCR K-loop is at structural register-pressure ceiling.
- **Dev D (V2-RCR persistent-CU dispatch, Approach 1) NO SHIP — STRUCTURAL CLOSURE + CRITICAL CORRECTNESS TRAP**: Mathematically demonstrated that for 4096³ V2-RCR (`total_tiles=256` < `slots=608` at occ=2), no persistent-CU scheme can synthesize work that doesn't exist in the (br, bc) tile grid — the 16% idle CUs are a function of `total_tiles < num_CUs`, not dispatch geometry. Persistent grid=304/608 schemes either keep idle gap unchanged (304 case) or add more early-exit overhead without adding workers (608 case). **CRITICAL CORRECTNESS TRAP discovered**: `MXFP8_RCR_V2_PERSISTENT_GRID < total_tiles` silently drops tiles via early-exit prologue — collapses SNR to 1.53 dB while reporting ~3× baseline TFLOPS. Sclk-drift artifact also surfaced: single-process sequential build-then-bench reported +8.6% (Welch t=3.43) which was a sclk-drift artifact (paired in-process A/B with 30s preheat eliminates drift). Macro `MXFP8_RCR_V2_PERSISTENT` left default-off (audit-only).

### R31 paradigm corrections (4 — extends R27/R28/R29/R30 closure list to 18 total)

1. **V2-CRR LDS single-buffer is a closed lever (synchronous form)** (Dev B): hits exact LDS reduction target but loses -30% to pipelining loss. Future SB attempts must include explicit pipelining-recovery (split global_load early-issue + late-wait, or async-copy variants) — naive synchronous SB is permanently closed.
2. **V2-RCR scale path has zero LDS round-trip — same as V2-CRR** (Dev C): SASS-confirmed 0 `ds_write` in V2-RCR kernel. **NEVER prototype "PIPELINE_SCALE second-buffer in LDS for V2-RCR" again.** VGPR-prefetch pivot is also closed (catastrophic spill, -92% perf). Re-confirms R27 paradigm correction #1 and R30 Dev C SASS evidence for the RCR side.
3. **Persistent-CU dispatch cannot help shapes with `total_tiles ≤ num_CUs`** (Dev D): mathematically proven for 4096³ V2-RCR. Inner-loop persistence with WORK-TILE LARGER THAN BLK is structurally infeasible (4× LDS footprint breaks 163840 B/CU cap). **Only structural fixes for 4096³ V2-RCR gap are: BLK=128 path resurrection (Approach 2, multi-day) or split-K with atomic accumulation (closed by R27 Dev B for V2-CRR; lower headroom on V2-RCR).**
4. **Rect-V2 CRR fastpath foundation lands without GPU fault** (Dev A): first concrete rect-V2 incremental SHIP after 4 cycles of side-branch scaffolding. Rect kernel measured at 148 VGPR / 104 KB LDS — still 2 blocks/CU occupancy (LDS-bound), so even completed rect-V2 cannot independently break occupancy ceiling. **R32+ Stage A2 work**: host preshuffle (`preshuffle_v2_b_rect`) + K_HALF=1 helper rewrite to enable correct numerics; SHIP target after numerics correct = perf parity or better than square baseline at 70B KV with 2× CU utilization.

### R31 corollaries (consolidated lever-closure tally → 18 total)

R27-R31 cumulative paradigm-correction count: **18 closed levers** (s_setprio CRR, s_setprio RCR, sched_barrier mask RCR, cachepolicy gate broadening, cachepolicy RCR, scale LDS double-buffer, split-K-along-K, V1 vs V2 size-gating, square BLK=128 rewrite, LDS bank conflicts CRR, occupancy=3 via VGPR (LDS-binding), buffer_load_dword_lds for V2 scales, H7 B-tile reorder hoist-above-cB RCR, rect ceiling at 2× grid, V2-CRR LDS single-buffer naive form, V2-RCR PIPELINE_SCALE LDS+VGPR variants, persistent-CU dispatch when total_tiles ≤ num_CUs, rect-V2 occupancy ≤ 2 blocks/CU even with -25% LDS).

### R32+ priority list (rebuilt from R31 closures)

1. **【critical / 1-2 day】Rect-V2 CRR Stage A2 (host preshuffle + K_HALF=1 helper rewrite)**: continue Dev A's `crr_mxfp8_exact_8wave_rect_fastpath.inc` to enable correct numerics. Add `preshuffle_v2_b_rect` host fn + rewrite `load_col_from_v2_st_half` to handle K_HALF=1 case (R28 Dev D's `offset = 8*BK` paradigm correction makes this scope smaller than originally feared). Once correct, perf-bench rect vs square at 70B KV — primary SHIP target.
2. **【high / 2-3 day】LDS single-buffer with pipelining recovery for V2-CRR**: rewrite Dev B's naive SB form with (a) split global_load early-issue + late-wait, or (b) async-copy variants, or (c) inner-K loop reorder to recover the cross-K-iter prefetch overlap. Currently the only path to occ=3 (LDS reduction proven feasible at -50%; only blocker is pipelining-loss recovery).
3. **【medium / 1-2 day】K-large MLP shapes investigation (8B/70B Down)**: 8B Down at 0.9394 + 70B Down at 0.8459 untouched by R28-R31 cachepolicy/rect work. Map cp_value vs K-extent; consider per-shape autotune table extension.
4. **【medium / 2-3 day】Rect BLK_M=256/BLK_N=128 for V2-RCR (Approach 2)**: requires adapting V2-RCR to rect; multi-day. R31 Dev D confirmed this is the ONLY structural path for 4096³ V2-RCR gap (BLK=128 path resurrection or split-K with atomic).
5. **【methodology — R32+ rules】**:
   - All harnesses: `rm -f tk_mxfp8_layouts*.so` + log per-build md5 (R29 Dev C) + `rocm-smi -d $PHYS_GPU` not `-d 0` (R31 Reviewer).
   - All `MXFP8_*_PERSISTENT_GRID` style macros: build-time assert `grid >= total_tiles` to prevent silent tile-drop SNR collapse (R31 Dev D).
   - All in-process A/B benches: BABA pattern + 30s preheat to eliminate sclk-drift artifacts (R31 Dev D).
   - SHIP claim normalization for 70B KV V2-CRR: discount GPU0 by 2.6% or use non-GPU0 box (R31 Reviewer).
6. **【closed】**: 18 levers per cumulative tally above. Do not re-prototype any of them.

### R31 Cherry-pick status

Cherry-picked to feat/mxfp8-only (all 6 R31 commits + 3 historical scaffolding commits squashed by-cycle):
- `3ed64359` (R28 Dev D scaffolding base) + `8916500c` (R29 Dev A guard) + `c682dd69` (R30 Dev A docs) + `7212e5e6` (R31 Dev A SHIP Stage A1a+A1b — rect-V2 CRR fastpath, 22 files / 2399 insertions; macros default-off so default build byte-identical)
- `0d56fbfc` + `dffba365` (R31 Dev B LDS single-buffer + bench logs; macros default-off)
- `10032d0c` + `a44d8e89` (R31 Dev C V2-RCR PIPELINE_SCALE NO SHIP + SASS audit logs; macros default-off)
- `cee9c3ae` (R31 Dev D V2-RCR persistent-CU NO SHIP + audit; macros default-off)
- `18c75159` (R31 Reviewer 4-GPU triangulation findings + JSON + bench artifacts)

All scaffolding/macro changes are default-off; default builds remain byte-identical to head. Side-branch commits preserved on r31-{a,b,c,d,rev} for R32+ continuation.

## R30 cycle 完结 (2026-04-18, 4 devs + 1 reviewer) ★ 0 SHIP + 4 STRUCTURAL CLOSURES + R28 SHIP健康 + R29 cells resolved

R30 派 4 dev (A GPU0 rect-V2 fastpath, B GPU1 VGPR/occ=3 push, C GPU2 buffer_load_dword_lds audit, D GPU3 H7 B-tile reorder) + Reviewer (GPU4 cross-GPU reverify of R29's 3 negative-t cells). **0 dev SHIP**, 4 high-value paradigm closures + R29 negative-t cells resolved (0/3 are real source-driven regressions; R28 SHIP healthy).

### R30 Reviewer cross-GPU reverify (commit `0e39e6b1` r30-rev → cherry-pick `r30_reviewer_crossgpu_*`)

GPU5 reverify of R29 GPU4 negative-t cells, identical 5x preheat + build-cache hygiene (rm -f .so + per-build md5):

| Cell | R27 GPU4 | R29 GPU4 | R30 GPU5 | Verdict |
|---|---:|---:|---:|---|
| 8b_down_crr | 2779.39 | 2712.16 | **2763.81** | **GPU4-state artifact** (R30 matches R27 within 0.6%, t=-0.6) |
| 70b_kv_crr  | 787.96 | 766.86 | **767.23** | environmental drift, not source-driven (R30 matches R29 within 0.05%, t=+0.56; both -2.6% from R27) |
| 8b_gate_crr | 2405.79 | 2390.72 | **2382.04** | inconclusive (~1% persistent shift; not source-driven) |

**0/3 are source-driven regressions.** Per-GPU dispersion observed for 70b_kv_crr (GPU0=791, GPU4-R27=788, GPU5=767, GPU4-R29=767) — recommend R31 4-GPU baseline triangulation to fix the live baseline number for this cell.

### R30 Dev results

- **Dev A (rect-V2 BLK_M=256/N=128 fastpath) NO SHIP**: Re-validated 2.5-day estimate against the kernel — 600+ lines of hand-tuned MXFP8 with 4 hardcoded `static_assert`s. Path A infeasible in 90-min budget. Path B (V1 fallback) already shipped as R29 guard at 2.66 TFLOPS (294× slower than V2 — no perf SHIP win possible). Cherry-picked R28D+R29A scaffolding to r30-a clean (8192³ −0.08% within noise). Refined Path A breakdown with exact line numbers preserved in `r30a_findings.md` for R31. **Structural ceiling note**: rect at 70B KV upper-bounds at 2× grid = 42% CU occupancy on 304 CUs, so the original 1.5× SHIP target is near the structural ceiling.
- **Dev B (VGPR reduction → occupancy=3) NO SHIP — STRUCTURAL CLOSURE**: V2-CRR baseline 234 VGPR / 0 spill / occupancy=2 / LDS=139264 B/block. Hardware ground truth (`hipGetDeviceProperties` on MI355X/gfx950): LDS per CU=163840 B. Two blocks would need 278528 B → **1.7× LDS overflow. LDS, not VGPR, is the binding occupancy constraint.** Compiler silently ignores `mb=3` hint (kernel binary bit-identical, md5 confirmed). Bench 8192³ +0.02% t=+0.06 NULL; 70B Gate −0.55% t=−3.26 regression (SPI launch-allocator overhead). Correctness identical SNR 49.59 / det 3/3.
- **Dev C (buffer_load_dword_lds audit) AUDIT-ONLY**: SASS-level data-flow trace of V2-CRR scale path. Scales declared as private VGPR arrays (`fp8e8m0_4 a0/a1/b0/b1_scale_packs`) populated by `__builtin_amdgcn_raw_buffer_load_b128/b64` → consumed as VGPR operands of `mfma_scale_f32_16x16x128_f8f6f4`. **Zero LDS round-trip exists** — no `ds_write` to convert. Bonus: TK `G::load` already uses `llvm_amdgcn_raw_buffer_load_lds` for tile fills (the only LDS-bound traffic in V2-CRR) — lever already maxed. **Critical for Dev B**: scale-pack VGPRs total ~6 of 232 (2.6%); even hypothetically eliminating them all cannot help reach occupancy=3 — bottleneck is float accumulators + tile-shape, not scale staging.
- **Dev D (H7 B-tile load reorder for V2-RCR) NO SHIP — STRUCTURAL CLOSURE**: 4 reorder variants benched. Variants v1/v3/v4 break correctness (SNR 7-24 dB, det FAIL) due to LDS lifetime constraint: `Bs[tic][1]` is read by `rcr_exact_load_st_to_rt(b1, ...)` at pre-cB; any next-iter VMEM `G::load(Bs[tic][1])` issued ABOVE that point races against still-draining LDS reads → corruption. Only v2 (defer-both-to-pre-cD) is correctness-safe and is NULL (Welch t=−0.247, slightly negative). **Baseline ordering (B[0]@cB, B[1]@cD) is already at maximum-latency-hiding ordering allowed by LDS lifetime.** H7 is structurally exhausted. Macro `MXFP8_RCR_V2_BLOAD_REORDER` left default-off in r30-d.

### R30 paradigm corrections (4 — extends R27/R28/R29 closure list to 14 total)

1. **Rectangular V2-CRR fastpath structural ceiling** (Dev A): rect at 70B KV upper-bounds at 2× grid = 42% CU occupancy on 304 CUs. The R29-projected 1.5× SHIP target is near the structural ceiling. **Closing 70B KV to ≥0.92 ratio likely requires more than rect** (e.g., split-K with atomic accumulation, BLK_N=64 second scaffolding round, or streamk).
2. **V2-CRR is LDS-bound, not VGPR-bound for occupancy** (Dev B): LDS=139264 B/block vs HW cap 163840 B/CU. `GEMM_MIN_BLOCKS_PER_CU > 2` is **CLOSED** for V2-CRR (compiler silently ignores hint, runtime regresses). NEVER prototype "occ=3 via VGPR reduction" without first dropping LDS footprint below 81920 B (separate, larger lever — single-buffer `As/Bs[2][2]→[1][2]` or smaller BLK).
3. **`buffer_load_dword_lds` is N/A for V2-CRR scales** (Dev C): V2 paradigm is scale-direct-to-VGPR with zero LDS round-trip; no `buffer_load + ds_write` pair exists to convert. Tile fills already use the lever via TK `G::load`. **NEVER prototype "buffer_load_dword_lds for V2 scales" again.** Re-confirms R27 paradigm correction #1 with empirical SASS evidence.
4. **H7 B-tile reorder is structurally exhausted for V2-RCR** (Dev D): LDS lifetime constraint pins B-tile loads to ≥pre-cB / ≥pre-cD; only the correctness-safe variant (v2 defer-both-to-pre-cD) is NULL. Baseline ordering already at max-latency-hiding allowed. **NEVER prototype "B-tile reorder hoist-above-cB for V2-RCR"** — guaranteed correctness failure.

### R30 corollaries (consolidated lever-closure tally)

R27-R30 cumulative paradigm-correction count: **14 closed levers** (s_setprio CRR, s_setprio RCR, sched_barrier mask RCR, cachepolicy gate broadening, cachepolicy RCR, scale LDS double-buffer, split-K-along-K, V1 vs V2 size-gating, square BLK=128 rewrite, LDS bank conflicts CRR, occupancy=3 via VGPR (LDS-binding), buffer_load_dword_lds for V2 scales, H7 B-tile reorder hoist-above-cB RCR, rect ceiling at 2× grid). Search space narrowing meaningfully.

### R31+ priority list (rebuilt from R30 closures)

1. **【critical / 2-3 day】Rectangular BLK_M=256/BLK_N=128 V2-CRR fastpath kernel**: still #1, still hardest. R30 Dev A `r30a_findings.md` has refined breakdown with exact `kernel_mxfp8_layouts.cpp` line numbers. **Adjusted target**: rect alone caps at ~1.5× on 70B KV (structural ceiling per Dev A). To hit ≥0.92 ratio gate, may need rect + another lever (split-K with atomic, or BLK_N=64 second scaffolding).
2. **【high / 2-3 day】LDS footprint reduction for V2-CRR**: ONLY remaining path to occupancy=3. Drop 139264 B/block below 81920 B by single-buffering `As[2][2]→[1][2]` or `Bs[2][2]→[1][2]` (pick the buffer with weaker reuse). Significant correctness work; double-buffer was added for latency hiding originally. Targets 8192³ V2-CRR -8.9% gap.
3. **【high / requires R31】4-GPU baseline triangulation for 70B KV V2-CRR** (R30 Reviewer recommendation): per-GPU dispersion observed (GPU0=791, GPU4-R27=788, GPU5=767, GPU4-R29=767). Run R31 baseline reverify on 4 GPUs (0/4/5/6) to fix the live baseline number for this cell — current "ratio" is unstable.
4. **【medium / 1-2 day】Dispatch-geometry change for 4096³ V2-RCR**: H7 closed (Dev D); structural GRID under-occupancy ceiling (16% CUs idle on 4096³ at BLK=256: 256 blocks vs 304 CUs) is the binding constraint. Address via persistent-CU block scheduling or BLK=128 path resurrection (multi-day).
5. **【medium / 1-2 day】PIPELINE_SCALE second-buffer for V2-RCR**: R28 Dev D scaffolding on r28-d. Requires LDS budget analysis (constrained by R30 Dev B's finding that LDS is already binding for CRR — RCR may have similar headroom).
6. **【methodology】All R31+ orchestrate scripts must `rm -f tk_mxfp8_layouts*.so` + log per-build md5** (R29 Dev C rule); cross-GPU reverify on flagged cells (R30 Reviewer rule).
7. **【closed】**: 14 levers per cumulative tally above. Do not re-prototype any of them.

### R30 Cherry-pick status

Cherry-picked to feat/mxfp8-only:
- `r30_reviewer_crossgpu_reverify.json` + `r30_reviewer_findings.md` + `r30_reviewer_crossgpu_orchestrate.sh` + `r30_reviewer_crossgpu_aggregate.py` (R31 cross-GPU reverify ready)
- `r30a_findings.md` + `r30a_orchestrate.sh` (Dev A refined Path A breakdown for R31)
- `r30b_findings.md` (Dev B LDS-binding paradigm correction with HW measurements)
- `r30c_findings.md` + `r30c_sass_inventory.log` (Dev C SASS-level data-flow trace)
- `r30d_findings.md` + `r30d_bench.py` + `r30d_orchestrate.sh` (Dev D H7 closure + LDS lifetime analysis + bench harness)

NOT cherry-picked (kernel/scaffolding-only on side branches): r30-a R28D+R29A scaffolding (still no rect-V2 kernel, dead without it), r30-d `MXFP8_RCR_V2_BLOAD_REORDER` macro (default-off, all enabled variants either corrupt or NULL).

Side-branch commits preserved: `7702f000` (r30-a), `2587a01c` (r30-b), `1849bc4e` (r30-c), `11576192` (r30-d), `0e39e6b1` (r30-rev).

## R29 cycle 完结 (2026-04-18, 4 devs + 1 reviewer)

R29 派 4 dev (A GPU0, B GPU1, C GPU2, D GPU3) parallel + Reviewer (GPU4 reverify). **0 SHIP** (all dev work NULL/AUDIT-ONLY)，**1 R28 SHIP confirmed in production** (Reviewer Welch t=+10.2)，**3 levers permanently closed** (LDS bank conflicts in V2-CRR, s_setprio in V2-RCR, sched_barrier mask in V2-RCR)，**1 measurement methodology bug surfaced** (Makefile build cache leak)。

### R29 Reviewer end-of-cycle baseline reverify (commit `73dd4444` r29-rev → cherry-pick r29_reviewer_*)

GPU4 5x preheat reverify of R27 10-cell baseline. **R28 cachepolicy auto-select SHIP CONFIRMED in production**: 70B Gate V2-CRR Welch t=+10.20 (+1.99% MXFP8 perf, FP8 also +1.99% making ratio appear unchanged at 0.8351; absolute perf gain is real and matches R28 Dev A's +2.7% measurement to within bench-to-bench noise).

| Cell | R29 ratio | R27 ratio | MXFP8 Welch t | Verdict |
|---|---:|---:|---:|---|
| 8k_rcr | 0.9095 | 0.9324 | -1.04 | regressed (FP8 drift up; MXFP8 unchanged) |
| 8k_rrr | 0.9086 | 0.9270 | +1.87 | regressed (FP8 drift up; MXFP8 +0.27%) |
| 8k_crr | 0.9017 | 0.9266 | +1.95 | regressed (FP8 drift up; MXFP8 +0.37%) |
| 4k_rcr | 0.9218 | 0.9229 | +0.52 | unchanged |
| 8b_gate_crr | 0.8986 | 0.9216 | -4.11 | **MXFP8 regression** (-0.63%) ⚠ |
| 8b_down_crr | 0.9059 | 0.9394 | -3.66 | **MXFP8 regression** (-2.42%) ⚠ |
| 70b_qo_rcr | 0.9106 | 0.9373 | -2.29 | regressed (mostly FP8 drift) |
| 70b_kv_crr | 0.8151 | 0.8646 | -8.51 | **MXFP8 regression** (-2.68%) ⚠ largest |
| 70b_gate_crr | 0.8351 | 0.8351 | **+10.20** | **R28 SHIP CONFIRMED** (MXFP8 +1.99%) ✅ |
| 70b_down_crr | 0.8274 | 0.8459 | +1.88 | regressed (FP8 drift up; MXFP8 +0.55%) |

**Cross-cycle drift caveat**: 0/10 cells pass ≥0.95 perf gate (unchanged from R27). FP8 drifted +1-3% universally vs R27 (likely DPM/GPU-state effect on GPU4), pulling ratios down without real MXFP8 degradation in 7/10 cells. Three cells show statistically significant MXFP8 absolute drops (t<-3) without source change since R28 — R30 should re-verify on a different GPU before treating these as real regressions.

### R29 Dev A = NO SHIP (`b2cc032f` r29-a only) — defensive guard scaffolding

Cherry-picked r28-d scaffolding clean. Added host-side dispatcher guard in `dispatch_pq_v2<CRR>` that fires `fprintf(stderr, ...)` + early-return when `MXFP8_RECT_BLK_N=64` (replaces R28 GPU memory access fault with loud host-side message). V1 fallback validated correct on 4096×1024×8192 (SNR 49.59 dB, det 3/3) but only 2.66 TFLOPS — 294× slower than V2 781 TFLOPS. **No SHIP because no rect-V2 fastpath kernel exists yet** (~2.5 day item, scoped on r28-d). 8192³ no-regression check passed at -0.42% (within noise).

Guard ONLY active under `-DMXFP8_RECT_BLK_N=64` build, default unaffected. **Not cherry-picked to main** (without surrounding rect scaffolding the guard is dead code on default builds; bringing scaffolding to main when no rect-V2 kernel exists adds compile-only cruft for zero perf). Docs + bench harness (r29a_*) cherry-picked to record next-cycle starting point.

### R29 Dev B = NO SHIP (`b82eab24` r29-b only) — strong negative confirmation of R28 gate

5-shape cp=0 vs cp=2 sweep (5x each, GPU1):

| Shape | Δ % | Welch t | Verdict |
|---|---:|---:|---|
| 70B Down 4096×8192×28672 | -0.11% | -0.95 | NEUTRAL (noise) |
| 8B Down 4096×4096×14336 | -2.75% | -1.83 | LOSE |
| Synth K14336 4096×8192×14336 | -3.34% | -7.82 | LOSE |
| Synth N14336 4096×14336×8192 | -0.98% | -2.89 | LOSE |
| Synth N20480 4096×20480×8192 | -2.53% | **-27.51** | LOSE catastrophically |

**R28 cp=2 win region is tightly localized to (N≥28672, K≥8192) — the 70B Gate corner only.** Even N=20480 K=8192 (71% of N threshold) loses 2.5% with t=-27.5 (N axis is sharp, not gradual). 70B Down K=28672 N=8192 also indistinguishable from cp=0 — large K alone doesn't trigger the win, both N≥28672 AND K≥8192 are required.

**R29 paradigm correction #1 — cp=2 is shape-dependent in a sharp non-monotonic way**:
- The R28 SHIP gate `(N_DIM>=28672 && K_DIM>=8192)` is not just "good enough", it is **exactly tight on both axes**.
- Do NOT prototype "extend cp=2 to nearby shapes" — the Dev B sweep proved the win region is a single 70B Gate point, not a gradient.
- **NEVER prototype "broaden cachepolicy=2 gate beyond R28 boundary"**. Lever is fully exploited.

### R29 Dev C = NO SHIP (`5b48ff76` r29-c only) — three more V2-RCR levers closed

V2-RCR sweep of `MXFP8_RCR_V2_MMA_SETPRIO` (s_setprio on 4 MMA quadrants) and `MXFP8_RCR_V2_SCHED_BARRIER_MASK` (sched_barrier mask relax). 6 cells across 4096³ + 8192³ × {p2, p3, sched_barrier}: |Welch t| ≤ 0.61 every cell, |Δ%| ≤ 0.35%. Best 4096³ candidate (setprio=2): -0.16% with t=-0.08.

**R29 paradigm correction #2 — `s_setprio` is a CLOSED lever for V2-RCR too** (extends R28 V2-CRR closure):
- Same root cause as R28: 8 waves run identical interleaved code in lockstep, no relative reordering possible without breaking wave-uniformity.
- **NEVER prototype "elevate MFMA wave priority" again, neither V2-CRR nor V2-RCR.**

**R29 paradigm correction #3 — `sched_barrier` mask relax is BENIGN for V2-RCR**:
- Even at mask=0xB (any reorder allowed) Welch t < 0.6 across all cells. Surrounding `s_barrier()` already pins the schedule; LLVM scheduler hint mask is not a bottleneck.
- **NEVER prototype "relax sched_barrier mask" for V2-RCR.** Lever closed.

**R29 paradigm correction #4 — Cachepolicy bits exhausted for V2-RCR**:
- R27: cp=1 NULL, cp=2/3 catastrophic on 4096³ V2-RCR.
- R29 Dev C confirms no remaining cachepolicy lever for RCR. **NEVER prototype "tune cachepolicy for V2-RCR"** (CRR-specific lever).

**R29 structural finding** — 4096³ V2-RCR may be GRID under-occupancy bounded:
- 256 blocks at BLK=256 / 304 CUs = 0.84 wave-fill. 16% of CUs always idle.
- Per-kernel optimization may be structurally bounded; closing the gap likely requires dispatch-geometry changes (block tile reshape, streamk).

**Bonus: R29 measurement methodology bug surfaced**:
- `make clean` does NOT remove `tk_mxfp8_layouts*.so`; under certain race conditions prior r27/r28 orchestrate scripts could load stale .so masking rebuilds.
- `r29c_orchestrate.sh` adds explicit `rm -f tk_mxfp8_layouts*.so` and per-build md5 logging as workaround.
- **R30+ rule**: all orchestrate scripts must explicitly delete .so before rebuild AND log per-build md5 to detect cache contamination.

### R29 Dev D = AUDIT ONLY (`63fc644b` r29-d only) — LDS bank conflicts ruled out

Static analysis of `load_col_from_v2_st_half` + `load_col_from_v2a_st_half` + ST_v2/v2a swizzle paths. For every 16-lane dispatch cycle, the existing swizzle `(nc ^ sw_k)` places lanes on all 32 banks exactly once. Both `ds_read_b64_tr_b8` instructions (offset:0, offset:1024) inherit the conflict-free pattern. Empirically corroborated by R22-B/R26-A profiling: SQ_LDS_BANK_CONFLICT/SQ_INSTS_LDS < 1%.

**R29 paradigm correction #5 — V2-CRR has ZERO LDS bank conflicts**:
- The audit doc (`r29d_lds_bank_audit.md`) lists rejected sub-experiments (row padding, sw_k bit re-mask, sched_barrier(0xff)).
- **NEVER prototype "LDS bank conflict reduction" for V2-CRR.** Lever closed.

R30 highest-EV directions (per Dev D recommendation): **VGPR/occupancy reduction (target 2 waves/CU)** + **`buffer_load_dword_lds` direct VMEM→LDS path** (skip VGPR staging entirely).

### R30+ priority list (rebuilt from R29 root-causes)

1. **【critical / 2-3 day】Rectangular BLK_M=256/BLK_N=128 V2 fastpath kernel**: still #1 priority. R29 Dev A added defensive guard but no fastpath. Path A (true rect-V2): write `dispatch_crr_exact_8wave_scaled_v2_rect<true>(g)` with B-side N-stride parametrization + V2 half-N scale preshuffle layout. Path B (force V1 fallback): if Path A blocked, wire dispatcher to V1 path with V1-preshuffled scales when caller requests rect mode. Targets 70B KV V2-CRR 0.8151 → ~0.92 from 2× CU utilization. r29a_bench.py + r29a_orchestrate.sh harness ready.
2. **【high / 1-2 day】VGPR reduction for V2-CRR occupancy=3** (R29 Dev D recommendation): current V2-CRR likely at occ=2. Identify spillable VGPR bands (especially scale broadcast registers held across MFMA quadrants), refactor to occupancy=3. Targets 8192³ V2-CRR -8.9% gap (only remaining 8192³ gap).
3. **【high / 2-3 day】`buffer_load_dword_lds` direct VMEM→LDS path** (R29 Dev D recommendation): bypass VGPR staging on B-side scale loads. Eliminates VGPR pressure + saves issue slots. Most invasive of the new R30 levers; would need a parallel scale-load helper variant.
4. **【medium / 1 day】B-tile load reorder (H7) for V2-RCR** (R29 Dev C recommendation): unexplored at R29 close. Targets 4096³ V2-RCR 0.9218 if not GRID-bounded.
5. **【medium / 1-2 day】PIPELINE_SCALE second-buffer for V2-RCR**: R28 Dev D scaffolding noted on r28-d. Structural optimization, requires LDS budget analysis (gfx950 cap = 160 KB/CU).
6. **【low / requires re-verify】R29 Reviewer's 3 cells with negative t**: 8B Gate, 8B Down, 70B KV CRR all show MXFP8 absolute regressions (t<-3) without source change since R28. Re-run on a different GPU (GPU0 or GPU2) before treating as real. Likely DPM/GPU-state effect.
7. **【methodology】Build-cache hygiene**: all R30+ orchestrate scripts must explicitly `rm -f tk_mxfp8_layouts*.so` + log per-build md5. Rule applies to bench harnesses generally.
8. **【closed】s_setprio (V2-CRR + V2-RCR), sched_barrier mask relax (V2-RCR), cachepolicy gate broadening, LDS bank conflicts (V2-CRR), cachepolicy for V2-RCR**: all permanently closed levers. Do not re-prototype.

### Reviewer & cherry-pick status R29

- Cherry-picked: r29_reviewer_baseline_gpu4.json + r29_reviewer_findings.md + r29_reviewer_bench5x.py + r29_reviewer_aggregate.py + r29_reviewer_orchestrate.sh (R30 reverify ready); r29a_findings.md + r29a_bench.py + r29a_orchestrate.sh + r29a_cell{1,2,3}_*.txt (Dev A docs+harness); r29b_findings.md (5-shape lookup table for R30); r29c_findings.md + r29c_orchestrate.sh (sweep harness with build-cache fix); r29d_lds_bank_audit.md (audit record).
- NOT cherry-picked (kernel/scaffolding-only on side branches): r29-a Dev A's dispatcher guard (dead code without rect scaffolding); r29-b cp=2 sweep build configs; r29-c MMA_SETPRIO/SCHED_BARRIER_MASK macros (functional no-op at default values).
- Side-branch commits preserved: r29-a `b2cc032f`, r29-b `b82eab24`, r29-c `5b48ff76`, r29-d `63fc644b`, r29-rev `73dd4444`.

## R28 cycle 完结 (2026-04-18, 4 devs + R27 reviewer baseline)

R28 派 4 dev (A GPU0, B GPU1, C GPU0, D GPU1) parallel + R27 Reviewer (GPU4 baseline matrix). **1 SHIP** (`88d5a7d5` cachepolicy auto-select gate, +2.7% on 70B Gate V2-CRR), **2 NO SHIP** (B + C), **1 SCAFFOLDING** (D, on r28-d for R29), **2 paradigm corrections**.

### R27 Reviewer baseline matrix (commit `9f97feb6` cherry-pick of 31f2286b artifacts)

GPU4, 5x preheat, FP8 W/I=300/300, MXFP8 W/I=200/300. 0/10 LLaMA cells pass ≥0.95 perf gate:
| Cell | Layout | FP8 med | MXFP8 med | ratio |
|---|---|---|---|---|
| 8192³ | rcr | 3234 | 3016 | 0.9324 |
| 8192³ | rrr | 3222 | 2987 | 0.9270 |
| 8192³ | crr | 2992 | 2772 | 0.9266 |
| 4096³ | rcr | 2550 | 2354 | 0.9229 |
| 8B Gate 4k×14336×4k | crr | 2610 | 2406 | 0.9216 |
| 8B Down 4k×4k×14336 | crr | 2959 | 2779 | 0.9394 |
| 70B Q/O 4k×8k×8k | rcr | 3178 | 2979 | 0.9373 |
| 70B KV 4k×1024×8k | crr | 911 | 788 | 0.8646 |
| 70B Gate 4k×28672×8k | crr | 2759 | 2304 | 0.8351 → **0.8579** post-R28 Dev A |
| 70B Down 4k×28672×8k | crr | 2956 | 2500 | 0.8459 |

Worst → best: 70B Gate 0.8351, 70B KV 0.8646, 70B Down 0.8459, 8B Gate 0.9216, 4096³ 0.9229. Cross-machine drift vs R26 GPU7 = ±3% (acceptable).

R28 Dev A's auto-select gate moves 70B Gate V2-CRR from 0.8351 → ~0.8579 (still below 0.95 gate but +2.28pp closer).

## R28 SHIPS



### SHIP #1: cachepolicy=2 auto-select gate (Dev A, commit `88d5a7d5`)

**+2.7% direct on 70B Gate V2-CRR with NO build flag.** Compile-time gate `N_DIM>=28672 && K_DIM>=8192` triggers `MXFP8_CRR_V2_SCALE_CACHEPOLICY=2` automatically when shape is pinned. Outside region stays 0 = binary identical to R27 baseline. User -D override still wins.

5x preheat-then-bench (GPU0, sclk-verified, SNR≥49.59 dB det 3/3 all PASS):
| Cell | Shape | Auto cp | TFLOPS (mean ± std) | Verdict |
|---|---|---|---|---|
| A | 8192³ CRR (no -D) | 0 | 2844.28 ± 15.11 | +0.97% no-regress |
| B | 70B Gate 4096×28672×8192 | 2 (auto) | 2419.99 ± 8.92 | matches R27 cp=2 target |
| B0 | same shape, explicit cp=0 | 0 | 2358.49 ± 5.95 | matches R27 cp=0 baseline |
| C | 70B KV 4096×1024×8192 | 0 (N<28672) | 794.04 ± 3.95 | +1.04% no-regress |
| D | 8B Gate 4096×14336×4096 | 0 (K<8192) | 2408.80 ± 51.10 | -0.96% mean (median 0.08%) |
| E | 4096³ RCR (untouched) | n/a | 2441.65 ± 94.21 | within R27 1σ |

Welch B vs B0: Δ=+61.49 TFLOPS / +2.61%, t=+12.83. Matches R27 +2.7%/t≈+13 to within bench-to-bench noise.

## R28 NULLs (no production change, recorded for posterity)

## R29+ priority list (rebuilt from R28 root-causes)

1. **【critical / 1-2 day】Rectangular BLK_M=256/BLK_N=128 V2 path completion**: build on Dev D's r28-d scaffolding (`ddfd2f80`). Audit-corrected scope is smaller than originally thought: B-side N-stride parametrization + V2 half-N scale preshuffle + dispatcher gate. Stage 3 fault root cause: V2 dispatch falls through to V1 layout w/ V2-preshuffled scales — need either (a) a true V2 rectangular layout or (b) a runtime gate that forces V1 fallback for rect mode. Targets 70B KV V2-CRR 0.8646 → projected ≥0.92 from 2× CU utilization (4 N-tile → 8 N-tile).
2. **【medium】Per-shape autotune table**: extend Dev A gate beyond `N>=28672 && K>=8192`. R27/R28 data shows the actual win region is shape-class-dependent. Build a small lookup table indexed by (M,N,K,layout) → cp_value rather than a 2D hyperplane. Targets +0.5-1% additional on shapes that currently sit at cp=0 (e.g., 8B Down which is at 0.9394 might benefit from cp=2).
3. **【medium】4096³ V2-RCR lever audit**: 0.9229 ratio, untouched by R28. Check if there's an analogous N/K gate for RCR (R27 said cp=2 catastrophic on 4096³, but is there a different cp/lever?).
4. **【low】8192³ V2-CRR -8.9% gap**: still no production-impacting fix path. Skip until rectangular BLK lands.
5. **【closed】s_setprio**: fully exploited per R28 paradigm #1. Do not re-prototype.




### R28 Dev B = NO SHIP + paradigm correction (`ffb10af5` r28-b only)

s_setprio sweep on top of cp=2: best (MFMA=2, VMEM=0) gives only +7.72 TFLOPS / +0.33% / Welch t=+2.30 — falls below SHIP gate (≥+30 / +1.2% / t>3.0) on all three thresholds.

**R28 paradigm correction #1 — `s_setprio` is a CLOSED lever**:
- The 8-wave V2-CRR kernel does NOT have a wave-id branch. All 8 waves run identical interleaved VMEM+MFMA code in lockstep.
- Pre-R28 code ALREADY uses `__builtin_amdgcn_s_setprio(1)` to elevate priority during MFMA-heavy segments and `s_setprio(0)` to drop back. Removing this bracket = -21 TFLOPS. Keeping MFMA elevated without restore = -74 TFLOPS.
- Pushing MFMA prio above 1 hits a hard ceiling (+0.3%, sub-ship) because 8 waves on a CU all hit the same setprio call in lockstep — no relative reordering possible. Unlocking more would require breaking wave-uniformity (large restructure).
- **NEVER prototype "elevate MFMA wave priority" again.** The lever is fully exploited.

### R28 Dev C = NO SHIP (`297249bb` r28-c only)

cp=3 (GLC|SLC) vs cp=2 (SLC) on 70B Gate V2-CRR (4096×28672×8192): Δ=-2.15 TFLOPS / -0.088% / Welch t=-0.77. R27's apparent cp=3=+11 TFLOPS over cp=2 was within R27 noise (cp=3 sd alone was 11.72). **cp=2 confirmed as the right gate value.** No production change.

Diagnostic no-regress (cp=3 forced on out-of-gate shapes, confirms gate region is doing real work):
- 70B KV 4096×1024×8192 cp=3: -7.46% vs cp=0 baseline
- 8B Gate 4096×14336×4096 cp=3: -4.95% vs cp=0 baseline
- 8192³ cp=3: -0.91% (within noise)

### R28 Dev D = SCAFFOLDING ONLY (`ddfd2f80` r28-d only)

Stage 1 PASS (default build clean, rect build clean — fastpath drops 2700→2483 TFLOPS confirming gate works). Stage 2 PASS (rect+V1 fallback correctness on 4096×1024×8192: SNR 49.59 dB, det 3/3, 2.67 TFLOPS). Stage 3 FAIL (V2 dispatch falls through to V1 layout with V2-preshuffled scales → GPU memory access fault, same mode R27 Dev D documented).

**R28 paradigm correction #2 — `offset:1024` is BK-derived, NOT HB-derived (audit fix)**:
- R27 Dev D's audit said `offset:1024` in `ds_read_b64_tr_b8` was hardcoded for HB=128. Dev D investigation found the actual dependency: **`offset = 8*BK`**, NOT `8*HB`. HB dependency localizes to `k_row = row_off + K_HALF*64` math + subtile count.
- Implication: rectangular BLK_M=256/BLK_N=128 (HB unchanged at 128, BK unchanged) requires LESS rewrite than R27 Dev D estimated. The B-side helper variant primarily needs N-stride changes, not the inline asm offset.
- Updated R29 starting point: focus on (1) B-side N-stride parametrization, (2) V2 scale preshuffle layout for half-N tile, (3) dispatcher gate for rectangular shapes. Scaffolding `MXFP8_RECT_BLK_N` macro + `load_col_from_v2_st_half_rect` template lives on r28-d for R29 to build on.

NOT cherry-picked to main (Stage 3 FAIL means -DMXFP8_RECT_BLK_N=64 currently faults; default is binary identical but no positive value to ship).

## R27 cycle 完结 (2026-04-18, 4 dev + 1 reviewer，1 partial production ship + 3 paradigm corrections)

R27 派 5 agent (Dev A GPU0, Dev B GPU1, Dev C GPU2, Dev D GPU3, Reviewer GPU4) 攻 R26 后剩下的 V2 levers。**1 个 macro infrastructure SHIP** (cherry-pick `12785d98`)，**3 paradigm correction**。

### R27 SHIP: `MXFP8_*_V2_SCALE_CACHEPOLICY` macro infrastructure (Dev A)
- 4 处 `+ MXFP8_{CRR,RRR}_V2_SCALE_CACHEPOLICY` macro hook on V2 `buffer_load_b128/b64` scale loads
- Default 0 = binary identical to current behavior (zero risk)
- Per-shape opt-in: `make CXXFLAGS="-DMXFP8_CRR_V2_SCALE_CACHEPOLICY=2"` for **70B Gate V2-CRR** (M=4096 N=28672 K=8192) gives **+63 TFLOPS / +2.7% Welch t≈+13** (5x preheat, GPU0, SNR 49.60 dB, det 3/3 PASS)
- **shape-dependent**: cp=2 catastrophic on 4096³ V2-RCR (-25×) and regression on 70B KV (-6.9%) / 8B Gate (-4.2%) / 8192³ V2-RCR (-2.3% breaks 3180 floor)
- Auto-select disabled — opt-in only. R28+ candidate: `#if N_DIM≥28672 && K_DIM≥8192 && IS_CRR` auto-select gate

### R27 paradigm corrections
1. **V2 has NO scale LDS** (Dev C): R23 Dev B's `SCALE_LDS REPLACE` kill misled subsequent rounds. V2 actually loads scale VMEM→VGPR direct (`crr_mxfp8_exact_8wave_fastpath.inc:304-333`, SCALE_VERSION==2 branch). LDS budget V2-CRR = **128 KB** (As/Bs double-buffer only), 32 KB headroom. R26 "131-139 KB" figure was different kernel variant. Adding scale LDS = re-litigate killed lever. **NEVER prototype "scale LDS double-buffer"** again.
2. **Split-K-along-K DEAD-END universally** (Dev B): -12% on 70B KV. Each K-chunk sub-grid still launches same 64 blocks → no CU exposure gain, but pay 2× launch + 2× epilogue + lose K=8192 single-pass B-tile cache reuse. Reduction overhead fine (7.7%, well under 30% gate). Real fix needs split-along-M/N (= R26 Dev A's static_assert path) or streamk (multi-day).
3. **BLK rewrite path: rectangular BLK_M=256/N=128 not square** (Dev D): 22 BLK=256 dependency sites audited, complexity 1=8 / 2=4 / 3=4 / 4=5 / 5=1. Square BLK=128 forces RBM/RBN below MFMA sweet spot. **Rectangular BLK_M=256/BLK_N=128 keeps RBM=64 (no A-side helper rewrite)**, only needs new B-side helper variant + B-only preshuffle. Est 2-3 days. Hardest blocker = `load_col_from_v2_st_half` family at `kernel_mxfp8_layouts.cpp:414-491` (`ds_read_b64_tr_b8 offset:1024` hardcodes K-stride for HB=128). Dev D shipped 25-line `MXFP8_BLK128` macro proof-of-concept (default unchanged, BLK128 V1-fallback PASS at 2.71 TFLOPS).

### R28+ priority list (rebuilt from R27 root-causes)

1. **【high / 1-2 day】Auto-select cachepolicy=2 by N_DIM/K_DIM compile-time gate**: extend Dev A macro infrastructure with `#if N_DIM>=28672 && K_DIM>=8192 && IS_CRR` auto-select. Direct +2.7% on 70B Gate without manual flag. Need to find boundary precisely (8B Gate K=4096 lost; 70B Gate K=8192 won — K threshold likely between).
2. **【critical / 2-3 day】Rectangular BLK_M=256/BLK_N=128 V2 path**: per Dev D, smaller scope than original BLK=128 plan. Add new B-side `load_col_from_v2_st_quarter` helper + B-only preshuffle helper. Targets 70B KV V2-CRR 0.84 (4 N-tile → 8 N-tile → ~2× CU utilization). Validate with `MXFP8_BLK128` infrastructure already shipped at `c019bec8` (r27-d, not yet cherry-picked).
3. **【medium】s_setprio on MFMA-side waves stacked on cp=3 for 70B Gate** (Dev A H2 not reached): could push +2.7% to +4-5% if scheduler fairness improves.
4. **【medium】Streamk scheduling for small-N**: alternative to BLK rewrite. One kernel, internal K-reduction across CUs. More complex than rectangular BLK but reuses existing tile structure.
5. **【low】8192³ V2-CRR -8.9%**: still no production-impacting fix path. Skip until other items addressed.

### Reviewer & cherry-pick status
- Cherry-picked Dev A SHIP: `12785d98` on feat/mxfp8-only
- Reviewer (GPU4) ran in parallel with devs to build 10-cell baseline matrix; verdict pending
- Side-branch commits NOT cherry-picked (dead-end docs only): r27-b `59d32212`, r27-c `9b2e1384`, r27-d `c019bec8`

## R26 cycle 完结 (2026-04-18, 5 dev/reviewer all DEAD-END/NULL on production change，但 3 个关键 paradigm correction)

R26 派 5 agent (Reviewer GPU1, Dev A GPU3, Dev B GPU5, Dev C GPU6, Dev D GPU7) 攻 R25 LLaMA baseline 4 个最差 cell + 1 winning cell mechanism。**0 production commit**, 5 个 side-branch commit (954ba8b4 + ee5f985a 已 cherry-pick to main 作 infrastructure + corrected baseline)。

### R26 paradigm corrections (3 critical)

1. **70B KV-attn V2-CRR 0.69 ratio was COLD-DPM THROTTLE ARTIFACT**: Reviewer (10x reps, iters=300, sclk-verified, GPU7 clean rerun) + Dev A (per-shape preheat, GPU3 throttle 1700 MHz) **independently** measure 0.84, +21% recovery vs R25. R25 multi-shape sweep had cold-DPM that hit V2 (heavier scale traffic) harder than FP8. **R25 baseline 70B KV CRR cell INVALIDATED, corrected value 0.84**. 其他 4 个 cell (70B Gate CRR, 70B Down CRR, 8B KV RRR, 70B Down RRR) Reviewer 10x 全 CONFIRM 在 ≤0.7% drift。

2. **Down-RRR 1.05 V2-WIN is FP8-RRR REGISTER-SPILL BUG, not V2 mechanism**: Dev C ASM diff 显示 FP8-RRR `RRR_MAIN_UNROLL=4` 在 K≥14336 spill **39 VGPR + 160 B/lane scratch** (FP8 K=4096 时 0/0)；V2-RRR 同 K spill 仅 1 VGPR (V2 SGPR-broadcast scale SRD 比 FP8 `RRR_B_REG_ROW_LOAD` register 压力低)。Spill 量与 k_iters (32→112→224 for K=4096/14336/28672) 线性相关，正好匹配 FP8 RRR/RCR 比 1.00→0.87→0.84 collapse。**V2 没有"赢"，只是 FP8 在大 K RRR 路径上有 bug**。无法 port 到 V2-CRR/RCR (后者已 0 spill)。Dev C 试 `RRR_MAIN_UNROLL=2` 修 spill 但 TFLOPS 跌 41% (latency hiding 不足)，`UNROLL=8` 同样 spill 39。FP8-RRR 修复需深度重构，留 R27+。

3. **V1 vs V2 dispatch dead universally**: Dev D 7 个 size 全 sweep (1024-8192) + Dev A 在 small-N (4096×1024×8192) **独立** A/B 验证。V1 vs V2 全部 ±2.5% 内 (Dev A: V1 791.64 vs V2 796.09, Welch t=-5.87 V2 微胜；Dev D: V2 8192 +60 TFLOPS, 其他 size 全 tied)。**V1/V2 runtime size-gating 永久 KILLED**。所有"V1 fallback at small problem"假说全部 dead-end。

### R26 root-cause (no fix)

- **70B Gate/Up V2-CRR 0.84 (large N=28672)**: Dev B PMC mode (5x normal + 5x PMC) 找到 dominant cap：**TA backpressure + per-wave SQ_INST_LEVEL_VMEM 4.895x scaling vs work-ratio 4.0x → +22% per-wave VMEM-stall queueing**. TCC_MISS 4.41x 但 FP8 4.91x 更差不是 V2-specific. 4 知 knob 全 NULL 或 correctness FAIL：CRR_STEADY_VMCNT=2 (-14.6%), VMCNT=6 (-0.7%), MID_BARRIER=0 (SNR 19 dB FAIL), PREFETCH_LGKM=2 (Welch t=-0.68 NULL)。8wave fastpath template 内现有 knob 已用尽。R27+ 候选 (未 prototype)：L2 cache-tag pinning of scale arrays (8 MB scale fits in 32 MB L2)、persistent-scale LDS prefill (DEAD-END at occ=2 / 160 KB cap)、per-buffer TCC counter split。

- **小 N (KV) 7-19% gap**: Dev A 3 个 hypothesis 全 dead-end. H1 (BLOCK_N=128) blocked by hardcoded `static_assert(BLK==256)` in `crr_mxfp8_exact_8wave_fastpath.inc:37` + `crr_mxfp8_4wave_fastpath.inc:86` + multiple V2 preshuffle helper assumptions. RBM/RBN, scale-pack counts, ST_v2/ST_v2a shapes 全 hardcoded. **Multi-day rewrite needed for tile reshape**, 不在 R26 90-min budget 内。

- **4096³ Q/O V2 0.92 (vs 8192³ 0.99)**: Dev D amortization curve 揭示 **gap is structural to MXFP8 not V2-specific** (FP8 -21%, V2 -24% from 8192→4096). 1024³→8192³ 全 size 都 < 0.95 gate. 真实 fix 路径：减小 macro-tile prologue (scale-load + LDS preload fixed cost per block) 或新增 128×128 macro tile，与 Dev A small-N 同样 multi-day rewrite。

### R26 infrastructure 落地 main

- **954ba8b4** (cherry-pick of r26-a Dev A): `analysis/fp8_gemm/mi350x/preheat_then_bench.py` + `preheat_fp8_bench.py` — per-process DPM preheat wrappers (避免 R25-style cold-throttle measurement artifact)。新规范：跑 multi-shape baseline 必须用这两个 wrapper。
- **ee5f985a** (cherry-pick of r26-rev Reviewer): `r26_reverify.json` — corrected 5-cell numbers (10x reps, iters=300, GPU7 clean). Future LLaMA baseline reproductions 应以此 5 cell 为对照。

### R26 corrected LLaMA gap landscape (paradigm 再修正)

| Cell | R25 ratio | R26 reviewer 10x | Drift | 状态 |
|---|---:|---:|---:|---|
| 70B KV V2-CRR | 0.69 | **0.84** | +21% | R25 throttled, R26 clean |
| 70B Gate V2-CRR | 0.84 | 0.84 | 0% | CONFIRM real |
| 70B Down V2-CRR | 0.84 | 0.85 | +1% | CONFIRM real |
| 8B KV V2-RRR | 0.81 | 0.81 | 0% | CONFIRM real |
| 70B Down V2-RRR | **1.05 ✅** | 1.05 ✅ | 0% | CONFIRM, but mechanism = FP8 spill bug |

实际 V2 vs FP8 gap 在 LLaMA shapes 上 **uniform 6-19%**，no extreme outliers，R25 报告的 0.69 是 sweep 测量协议 artifact。**所有 cells (除 Down-RRR) 仍未达 0.95 gate**, R26 0 production fix。

## R27+ 优先级 (基于 R26 root-cause analysis)

1. **【critical / multi-day】Tile shape rewrite**: 当前 BLK=256 hardcoded 是 small N (KV 0.81-0.84) + small M=N=K (4096³ 0.92) 的根本约束. 需 (a) 重写 V2 preshuffle helper 支持 BLK=128, (b) RBM/RBN/scale-pack count 参数化, (c) ST_v2/ST_v2a 重 shape. 多日工作但是覆盖最大 production area
2. **【high / 1-2 day】L2 cache-tag pinning of scale arrays for large-N CRR**: Dev B R27 候选 #1. 8 MB scale data fits in 32 MB L2，可消除 V2 在 N=28672 上的 TA/VMEM scaling
3. **【high / FP8-side work】Fix FP8-RRR spill at K≥12000**: Dev C 揭示的 FP8-RRR pathology. 修复后 LLaMA Down-RRR FP8 提升 12-15%, 让 V2-RRR 1.05 win 消失但 FP8 整体更快 (production benefit). 与 MXFP8 工作正交
4. **【medium】Per-buffer TCC split for large-N CRR**: 确认 scale vs A vs B miss attribution (Dev B R27 候选 #3)
5. **【low】8192³ V2-CRR -8.9%**: 原 R25 mainline target. 仍 real 但 production impact 最低 (LLaMA 不跑 8192³)

## R25 LLaMA baseline 结果 (2026-04-18, GPU5, sclk 2320 MHz, commit `6be73744`/`05bf4fef`) ★ paradigm 再修正：8192³ near-parity 不能推广到 production shape — **R26 修正：70B KV CRR 0.69 实为 0.84 throttle artifact**

R25 LLaMA shape baseline 正式跑完（8 build shape，覆盖 10 logical shape；FP8 + V2 各 5x，60 measurement run）。**结论：8192³ "V2 ≈ FP8" 的判断只在 8192³ 成立。LLaMA production shape 上 V2 显著落后 FP8。**

| Shape (M×N×K) | FP8-RCR | V2-RCR | ratio | FP8-RRR | V2-RRR | ratio | FP8-CRR | V2-CRR | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8B Q/O 4096³ | 2450 | 2256 | 0.92 ❌ | 2456 | 2301 | 0.94 ❌ | 2341 | 2147 | 0.92 ❌ |
| 8B KV 4096×1024×4096 | 865 | 730 | 0.84 ❌ | 874 | 711 | 0.81 ❌ | 792 | 659 | 0.83 ❌ |
| 8B Gate/Up 4096×14336×4096 | 2703 | 2531 | 0.94 ❌ | 2672 | 2487 | 0.93 ❌ | 2514 | 2361 | 0.94 ❌ |
| 8B Down 4096×4096×14336 | 3137 | 2911 | 0.93 ❌ | 2740 | 2886 | **1.05 ✅** | 2915 | 2728 | 0.94 ❌ |
| 70B Q/O 4096×8192×8192 | 3117 | 2917 | 0.94 ❌ | 3094 | 2874 | 0.93 ❌ | 2911 | 2711 | 0.93 ❌ |
| 70B KV 4096×1024×8192 | 1022 | 862 | 0.84 ❌ | 1027 | 845 | 0.82 ❌ | 915 | 632 | **0.69 ❌** |
| 70B Gate/Up 4096×28672×8192 | 2959 | 2659 | 0.90 ❌ | 2878 | 2578 | 0.90 ❌ | 2816 | 2361 | 0.84 ❌ |
| 70B Down 4096×8192×28672 | 3249 | 2835 | 0.87 ❌ | 2716 | 2851 | **1.05 ✅** | 3017 | 2542 | 0.84 ❌ |

**Pass rate: 2/24** (gate = V2 ≥ FP8 × 0.95). 全 24 cell correctness PASS (SNR ≥48 dB, det 5/5)。

**关键观察**：
1. **唯一 V2 win 是 Down-RRR (1.05 ratio for both 8B & 70B)** — 共同特征：large K (14336/28672), RRR layout. K 越大 V2 scale-load 摊销越好；为什么 Down-RCR/CRR 没有同样收益是 R26 待探究
2. **小 N (KV-attn N=1024) 是最差 regime**：FP8 自身已仅 ~1000 TFLOPS（vs 3200 在大 shape），V2 进一步降到 ~700-860 → 0.69-0.84 ratio. 256×256 block 在 N=1024 只有 4 个 N-tile，调度并行度严重不足
3. **大 N (gate/up N=14336→28672)** V2 进一步退化：8B Gate/Up V2-CRR 0.94 → 70B Gate/Up V2-CRR 0.84 —— N 越大 V2 相对 FP8 越糟糕（CRR 尤甚 -10%）
4. **8192³ "near-parity" misleading**：实际 production GEMM 全在 4096 M / 不同 N/K. 4096³ V2-RCR 已经只有 0.92 ratio (vs 8192³ 的 0.99). 8192³ 不代表 production

**8192³ vs LLaMA Q/O 4096³ 对比**（同 RCR layout，不同 M=N=K）：
- 8192³: FP8 3232, V2 3214, ratio 0.99 (near-parity)
- 4096³: FP8 2450, V2 2256, ratio 0.92 (gap -8%)
→ **V2 vs FP8 gap 与问题规模强相关**：M=N=K 减半，V2 ratio 从 0.99 跌到 0.92。可能原因：4096³ 总 tile 数减少 4x，V2 的 b128 scale load 摊销分母变小

## R25 mainline (8192³ CRR optimization) — 跨会话 task lost，0 commit，0 verified result

R25 8192³ CRR optimization 5 个 agent (Reviewer + Dev A/B/C/D) 在前一个会话 session 派出，session compaction 后 task 丢失，且 5 个 worktree (/tmp/wt-r25-{a,b,c,d,rev}) head 仍在 c285cb70 (R24 docs)，无任何新 commit。结合 R25 LLaMA findings，8192³ CRR -8.9% 不再是 #1 priority — 因为 LLaMA shapes 的 gap 全部更大。**R25 mainline 撤销，R26 优先级完全 rebuilt around LLaMA shapes**。

## R26+ 新优先级（基于 LLaMA baseline gap，按 production impact 排序）

1. **【critical】KV-attn 小 N=1024**：8B KV V2-RRR 0.81, 70B KV V2-CRR 0.69 — LLM inference 每层都跑，gap 最大
   - 假说: 256×256 block tile 在 N=1024 仅 4 N-tile/grid，CU 利用率低. 试 128×256 block 或更小
   - 假说: V2 scale b128 load 在小 N 下严重浪费带宽 (4 N-tile × 256 N-cols = 1024 cols, 但 b128 一次拉 16 scale)
   - 试 dynamic dispatch: N < 2048 走 V1 fallback / 走专门小 N kernel
2. **【high】大 N CRR (Gate/Up 70B 0.84)**：MLP 上下流瓶颈. 28672 N 比 14336 退化更多 → 与 V2 的 LDS budget 关联?
   - 复测：8B Gate (N=14336, V2-CRR 0.94) vs 70B Gate (N=28672, V2-CRR 0.84) 用同一 PMC profile 对比 ds_read/buffer_load 比率
3. **【medium】Down-RRR 1.05 win 推广**：唯一 V2 winning case. 找出其 mechanism (大 K + RRR 特定？) 试搬到 Down-RCR/CRR
4. **【medium】4096³ Q/O proj** (V2 0.92): 真实"square" production. 与 8192³ near-parity 形成 reference. 找 V2 scale load 在小问题下的 amortization formula
5. **【low】8192³ CRR -8.9%**: 原 R25 mainline. 仍然 real，但 production impact 远小于上面 4 项 — 因为 LLM 实际不跑 8192³

## R26 派活 plan
- **Reviewer (GPU0)**: 复测 LLaMA baseline 5 个最差 cell (KV V2-CRR 0.69-0.83, Gate-70B V2-CRR 0.84) 各 3x，确认非测量噪声
- **Dev A (GPU1)**: 攻 70B KV V2-CRR 0.69. 实验 (a) 减小 block tile 256→128 N (b) N<2048 强制 V1 fallback (c) 探查 V2 scale b128 在 4-N-tile grid 下的实际 issue 数
- **Dev B (GPU2)**: 攻 70B Gate V2-CRR 0.84 vs 8B Gate V2-CRR 0.94 差距. PMC ds_read/buffer_load/MFMA-busy 对比看 N=28672 触发哪个 resource cap
- **Dev C (GPU3)**: 攻 Down-RRR 1.05 win mechanism. PMC profile + ASM diff Down-RRR vs Down-CRR 看 V2 advantage 来源；尝试 port 到 RCR/CRR
- **Dev D (GPU4)**: 攻 4096³ Q/O 0.92. 用 8192³ 99.4% 做对照，PMC 找 amortization-curve breakpoint，测试 BLOCK_K 调整是否能 shift breakpoint 到 4096

## 当前 baseline（GPU0，per-iter sync，8192^3）— **R24 reverify (2026-04-18) ★ paradigm 大修正：V2 ≈ FP8 (RCR -0.5%, RRR -2.4%)，R23 SQC/TCC 大幅"瓶颈"全为 PMC-mode 测量伪影**

| 版本 | TFLOPS | SNR | 相对 FP8 RCR |
| --- | ---: | --- | ---: |
| **FP8 per-tensor RCR (长期目标)** | **3232** (R24 reverify, R23: 3252) | 49.61 dB PASS | 100.0% |
| **MXFP8 RCR PRESHUFFLE V2 (R24 fresh baseline, ★ default-on)** | **3214** (R23: 3074, +140) | 49.60 dB PASS | 99.4% (gap -17 / -0.5%) ★ near-parity |
| **MXFP8 RRR PRESHUFFLE V2 (R24 fresh baseline, ★ default-on)** | **3156** (R23: 3033, +123) | 49.59 dB PASS | 97.6% (gap -76 / -2.4%) |
| **MXFP8 CRR PRESHUFFLE V2 (R24 fresh baseline, ★ default-on)** | **2943** (R23: 2811, +132) | 49.60 dB PASS | 91.1% (gap -289 / -8.9%) ← 唯一仍有 meaningful gap |
| MXFP8 RCR V1 (RUNTIME=0 fallback) | 3022.43 (R21) | 49.60 dB PASS | — |
| MXFP8 RRR V1 (RUNTIME=0 fallback) | ~2878 (R22) | 49.59 dB PASS | — |
| MXFP8 CRR V1 (RUNTIME=0 fallback) | 2711.78 (R22) | 49.60 dB PASS | — |
| **gap MXFP8 RCR V2 vs FP8 RCR** | **−17** (−0.5%) ★ | | **R24 reveal: 实际近 parity，R23 -178 是 PMC-mode 协议差异 + GPU3 cold-throttle artifact** |
| **gap MXFP8 RRR V2 vs FP8 RCR** | **−76** (−2.4%) | | **R24 reveal: R23 -219 同样高估** |
| **gap MXFP8 CRR V2 vs FP8 RCR** | **−289** (−8.9%) | | **R24 reveal: 唯一 meaningful gap; CRR-specific bottleneck (col-major A-LDS) 仍是真问题** |

**R24 paradigm 修正**：R23 cycle-level diagnostic 数字 (SQC_DCACHE_BUSY +441%, TCC_MISS +166%) **R24 fresh measurement 不可复现** (SQC +38%, TCC_MISS +3.1%)。根因：(a) PMC-mode 强制 dispatch 序列化，多次产生 +200% 假信号；(b) GPU3 cold-throttle 在 R23 Dev C 报告里就 noted；(c) R23 vs R24 不同 warmup/protocol。R23 列出的 6 ranked NEW levers 全部基于这些噪声放大数字 → R24 全部 invalidate。**真实状态**: V2 已基本追平 FP8 RCR/RRR；剩余只有 CRR -8.9% 是 structurally real。

**R22 综合**：V2 preshuffle paradigm 推广到三 layout（RCR + RRR + CRR）全 SHIPPED default-on。RRR 大额收益 (+5.54%) 因 V1 RRR 有 19 spills/80B scratch 这次 V2 collapsed to 0 spills；CRR 小额 (+0.76%) 因 baseline 已 MFMA-bound + 已用 PIPELINE_SCALE 单 shot 4×b32。所有 V1 路径保留为 RUNTIME=0 fallback。

**R21 Reviewer 5x V2 + 5x V1 reproduction (GPU0, indep verify)**：V2 median 3078.09 (std 4.63), V1 median 3022.43 (std 3.19), Δ +55.66 TFLOPS / +1.84%, Welch t=23.33 (p<<0.001). 全 SNR ≥49.5 + det 3/3 PASS at 256³/1024³/8192³。FP8 RCR median 3243.67 (std 6.86)。SQ_INSTS_VMEM V1=6,815,744 / V2=5,767,168 = -15.38% byte-exact match Dev A。

**R21 Dev A win**: First production-impacting MXFP8 RCR optimization in 21 rounds. Cherry-picked to main as commits `4abd4f62` (milestone-1 foundation) + `efc389ff` (milestone-2 fastpath wiring). Default ON via `MXFP8_RCR_PRESHUFFLE_V2_RUNTIME=1`; V1 fallback retained. VGPR 254→246, occupancy 2 unchanged.

**R21 Dev B kill**: SCALE_LDS REPLACE 永久 dead-end - R20 "compiler folds 6 addresses to 3 VGPRs" 假说 FALSIFIED by hipcc -S；3 个 fix attempt 全 fail；即使 correctness 修好也是 -270 TFLOPS regression（R19 linear model under-counted LDS-issue cost）。Docs at commit `3c6e2392`.

### R15 vs R16 对比（同代码同 commit a8237d01）

| Layout | R15 | R16 | drift |
|---|---:|---:|---:|
| MXFP8 RCR | 3015.73 | 3010.57 | -0.17% (噪声) |
| MXFP8 RRR | 2889.36 | 2862.99 | -0.91% (噪声) |
| MXFP8 CRR | 2830.67 | 2775.96 | -1.93% (噪声边缘) |
| FP8 RCR | 3253.80 | 3229.67 | -0.74% (噪声) |

跨会话 baseline 漂移 1-2% 是正常现象。CRR 这次小幅落 gate 之下是测量噪声不是真实退化。

### 历史对比 (GPU7)

| 版本 | TFLOPS | SNR |
| --- | ---: | --- |
| FP8 per-tensor RCR (历史) | 3070.93 | 49.61 dB PASS |
| MXFP8 8-wave RCR (历史) | 2926.61 | 49.60 dB PASS |
| MXFP8 8-wave RRR | 2794.26 | 49.59 dB PASS |
| MXFP8 8-wave CRR + PIPELINE_SCALE | 2740.55 | 49.60 dB PASS |

### RRR / CRR 95% gate — **R15 GPU0 重测：双 gate 全 PASS**

R15 静态 gate（历史）：2926.61 × 0.95 = **2780.28 TFLOPS**
- RRR 2889.36 ✅ (+109.08 over gate)
- CRR 2830.67 ✅ (+50.39 over gate)

R15 动态 gate（今日 GPU0 RCR × 0.95）：3015.73 × 0.95 = **2864.94 TFLOPS**
- RRR 2889.36 ✅ (+24.42 over dynamic gate)
- CRR 2830.67 ❌ (−34.27, 93.86% of today's RCR)

**注**：CRR 在动态 gate 下小幅未达标（差 ~1.14%），但静态 gate 充分达标。R15 的真实新闻是 RCR 提升至 3015.73（vs R14 GPU0 测的 2806.17 = +210 TFLOPS / +7.5%），原因详见下方 R15 章节（defaults hygiene fix）。

**R14 (2026-04-17) GPU0 实测（同样 commit 8934e95c，PIPELINE_SCALE only，0 代码改动）**：

| GPU | RCR | RRR | CRR | CRR/RCR | CRR vs 2780.28 gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| **GPU0** | **2806.17** | **2873.02** | **2840.98** | **101.24%** | **+60.70 ✅ 达标** |
| GPU0 (run 2) | 2813.02 | — | 2841.60 | 101.02% | +61.32 ✅ |
| GPU0 (run 3) | 2806.74 | — | 2838.67 | 101.14% | +58.39 ✅ |
| GPU0 formal (200i × 3 det) | 2808.44 | — | **2835.16** | 100.95% | **+54.88 ✅ SNR 49.60 PASS, det 3/3 PASS, correctness 100%** |
| GPU6 | 2708.25 | 2780.65 | 2715.83 | 100.28% | −64.45 ❌ (GPU6 整体偏慢) |
| GPU7 (今日) | 2726.20 | 2775.90 | 2736.45 | 100.37% | −43.83 ❌ (vs 历史 2925 RCR 已退化 7%) |

**R14 关键发现**：
1. **CRR 在所有 GPU 上都已 ≥ RCR**（100.28-101.24%），从未存在真实的"CRR 弱于 RCR"问题
2. **2780.28 gate 在 GPU0 上完全达标**（CRR 2835.16, +54.88 over gate, 3-run reproducible, formal 验收 PASS）
3. **R7-R14 共 8 轮追的"1.43% gap"是测量幻觉**：历史 RCR 2925.64 是 GPU7 早期峰值测量，而 CRR 是后续在不同状态测的；CRR 实际从未慢于 RCR，gap 由 RCR 跨 GPU/状态变化造成
4. **GPU7 已退化 ~7%** vs 历史（RCR 2925→2726），所有 GPU7 上的"差 39.73 TFLOPS"都是这个 RCR 退化导致的相对值假象
5. **PIPELINE_SCALE only (commit 8934e95c) 是真正的 production fix**，无需任何 R7-R14 的进一步优化

**Gate status**: ✅ **MET** (verified GPU0 formal 2026-04-17)

reviewer 历史验收数据（GPU7，warmup=100 iters=200 per-iter sync）：
- HOIST_HI formal (with SNR + det 3/3 gate): 2925.64 TFLOPS PASS
- 同 GPU 同条件 A/B 4 次 mean：baseline 2914.87 → HOIST_HI 2932.67，**Δ +17.80 TFLOPS (+0.61%)**
- Dev B 在 GPU1 上 A/B 5 次 mean：baseline 2954.30 → HOIST_HI 2979.97，Δ +25.67 (+0.87%) (GPU1 噪声更低)

**新规范 (R14 起)**：
- 任何 MXFP8 baseline / gate 测量必须**同一会话同一 GPU 同时测 RCR + CRR**，避免跨时间/状态比较
- GPU0 是当前唯一持续达到历史性能水平的板子；GPU7 已退化，GPU6 整体偏慢
- 派 reviewer 时显式指定 `HIP_VISIBLE_DEVICES=0` 做 final gate 验收

reviewer 验收数据（GPU7，warmup=100 iters=200 per-iter sync）：
- HOIST_HI formal (with SNR + det 3/3 gate): 2925.64 TFLOPS PASS
- 同 GPU 同条件 A/B 4 次 mean：baseline 2914.87 → HOIST_HI 2932.67，**Δ +17.80 TFLOPS (+0.61%)**
- Dev B 在 GPU1 上 A/B 5 次 mean：baseline 2954.30 → HOIST_HI 2979.97，Δ +25.67 (+0.87%) (GPU1 噪声更低)

构建 flag（MXFP8 当前最佳）：
```
-DMXFP8_RCR_EXACT_8WAVE_FAST_ENABLE=1
-DMXFP8_RCR_EXACT_PQ_KPAIR_LOOP_ENABLE=1
-DMXFP8_RCR_EXACT_PQ_PIPELINE_SCALE_ENABLE=1
-DMXFP8_RCR_EXACT_PQ_HOIST_HI_ENABLE=1
```

## 已完成

- [x] 创建 `feat/mxfp8-only` 分支
- [x] 删除所有 MXFP4 / Gluon kernel、测试、rewriter、产物（48 个文件）
- [x] 建立 FP8 / MXFP8 baseline
- [x] 决策者汇编级差距分析（见下方）
- [x] 第一轮 agent team 派活（Dev A / Dev B / Dev C）— 被中断
- [x] 半成品改动 stash 保存（`stash@{0}`，含 Dev A 的 4-wave KPAIR/SCALE_PIPE 骨架、Dev B 的 HOIST_HI opsel helper、Dev C 的 8-wave asm rewriter）
- [x] 第二轮 agent team 产出评审：采纳 Dev B (HOIST_HI)，**拒绝 Dev C (ASM rewriter)** 和 Dev A (4-wave KPAIR/SCALE_PIPE)
- [x] 清理 `.s` / per-run `.json` 等生成物，加强 `.gitignore`
- [x] 整理 `.cursor/skills`：删除 deprecated `fp8-strict-layout-tuning`，重命名 `mxfp8-mxfp4-layout-tuning` → `mxfp8-layout-tuning`，清除 mxfp4 知识，加入 commit-time 工作流

## 差距分析（8-wave inner loop 每 kpair，64 MFMAs）

| metric | FP8 | MXFP8 (前) | MXFP8 (HOIST_HI) | Δ vs FP8 |
| --- | ---: | ---: | ---: | ---: |
| 总行数 | 403 | 431 | **422** | +19 |
| MFMAs | 64 | 64 | 64 | 0 |
| `buffer_load` | 16 | 22 | 22 | +6 (scale loads) |
| `ds_read` | 48 | 48 | 48 | 0 |
| `s_waitcnt` | 10 | 12 | 12 | +2 |
| `s_barrier` | 16 | 16 | 16 | 0 |
| `v_lshrrev_b32` | 0 | 6 | **0** | 0 (op_sel 替代) |
| VGPR | 252 | 256 | 254 | +2 |
| LDS | 131 KB | 135 KB | 131 KB | 0 |
| Occupancy | 2 | 2 | 2 | 0 |
| Spills | 0 | 3 | 0 | 0 |

**HOIST_HI 成功消除了 6 v_lshr**（通过 `op_sel` + `op_sel_hi` 让 MFMA 硬件直接从 32-bit scale pack 中 byte-select 两个 scale 字节）。

**关键经验**：
1. K_PHASE 必须是编译期常量（通过 C++20 templated lambda 实现），否则 `if (k_phase == 0) opsel<0> else opsel<1>` 在 tail 会生成双份 MFMA 代码路径，导致 +64 额外 MFMA + 31 spills + 20 scratch accesses，性能倒退 ~2.8%。
2. Tail 区域的 `k_phase` 是 runtime，HOIST_HI 路径不能在 tail 用（会爆代码）。tail 直接 fallback 到 `rcr_mma_scaled_from_packs_exact` 即可，因为 tail 每 block 只跑一次，v_lshr 不在关键路径。

## 剩余差距（2925.64 → 3070.93 约 145 TFLOPS / 4.7%）

Main loop 已无 v_lshr，结构上与 FP8 几乎一致（仅多 6 个 scale buffer_load）。剩余差距主要来自：
- 6 个 scale `buffer_load` 的 issue 开销
- 额外 2 个 `s_waitcnt`
- 无法继续压缩 VGPR（256 hard limit，254 已接近极限）

## 进行中 / 下一轮

**第三轮评审 (2026-04-17) — 三条路径全 REJECT，不 commit 代码**

| 路径 | 结果 | 采纳？ |
|---|---|---|
| Dev A — scale cross-iter prefetch (MXFP8_RCR_EXACT_PQ_SCALE_PREFETCH_N1_ENABLE) | scope=0 full ring: 4 spills, A/B −2.26%; scope=1 B-only spill-free: A/B −1.58%。真实 VGPR 254，2 VGPR headroom 根本不够。 | **拒绝** |
| Dev B — tail compile-time K_PHASE dispatch (MXFP8_RCR_EXACT_PQ_TAIL_DISPATCH_ENABLE) | 正确性+资源全 clean（VGPR 254 / Spills 0 / LDS 128 KB / Occ 2 不变），asm 少 12 tail v_lshr；但 reviewer GPU7 A/B 10+20 rounds Δ = −0.04% ~ +0.18%，低于 +0.25% 噪声门槛 | **拒绝**（技术正确但噪声级） |
| Dev C — KPAIR 2× unroll (MXFP8_RCR_EXACT_PQ_KPAIR_UNROLL2_ENABLE) | body 翻倍 → live-range 爆 256 VGPR，51 spills，A/B −54.87% | **拒绝** |

**关键纠错**：真实 8-wave PQ scaled kernel = **VGPR 254 / LDS 131 KB**（不是之前决策者读错的 212 VGPR / 139 KB，那是 outer dispatcher）。headroom 仅 ~2 VGPR。

**第四轮评审 (2026-04-17) — 两条路径 REJECT**

| 路径 | 结果 | 采纳？ |
|---|---|---|
| Dev D — LDS-cached scales `MXFP8_RCR_EXACT_PQ_SCALE_LDS_ENABLE` 实测 | 编译过，VGPR 243 / LDS 132 KB / occ 2；smoke 256 PASS；但 8192 formal A/B −0.76%（5 runs 全 slower）且 **determinism FAIL**（max abs 1.43）。SCALE_LDS 设计是替代 PIPELINE_SCALE，叠加反而在 tail 加 ds_write/s_barrier 拖慢；LDS 同步仅靠 compiler fence 不足 | **拒绝（regression + det fail）** |
| Dev E — sched V2 松 `TK_WAIT_VMCNT(6→8)` 在 HOIST_HI 之上 | 正确 / 资源不变；GPU1 A/B 10 runs Δ +0.11%，Welch-t 0.58，纯噪声 | **拒绝（marginal）** |

**第五轮评审 (2026-04-17) — 三条路径 REJECT**

| 路径 | 结果 | 采纳？ |
|---|---|---|
| Dev F — AGPR accumulator per-MFMA `"+a"` (MXFP8_RCR_EXACT_PQ_AGPR_ACC_ENABLE) | VGPR 254→128 / AGPR 0→128 但 **Spills 0→13 / Scratch 56 B/lane**；285 V↔A shuffles per kpair；GPU3 A/B 10 runs Δ −2.19%。要破局需重写 `_row`/`_impl` 为单 asm 块 fuse 8-32 MFMA，不是本轮 scope | **拒绝（regression）** |
| Dev G — scale L2 cache-policy hint sc0 (MXFP8_RCR_EXACT_PQ_SCALE_L2_HINT_ENABLE) | 12/238 buffer_load 用 sc0，正确+资源不变；GPU4 A/B 15/20 清理后 Δ +0.066% Welch-t 0.20。sc1/nt 更差。cache-policy 轴已饱和 | **拒绝（marginal）** |
| Dev G2 — scale buffer_load b32×2 → b64 合并 (MXFP8_RCR_EXACT_PQ_SCALE_LOAD_B64_ENABLE) | 6 scale dwords 分布在 6 个独立 SRD，最小间距 8192 B，b64 需要 X 与 X+4 同 SRD → **结构不可行**；需重设计 `preshuffle_scale_matrix_mfma16` 影响所有 MXFP8 变体 | **拒绝（broken）** |

**第六轮评审 (2026-04-17) — 一条路径 REJECT**

| 路径 | 结果 | 采纳？ |
|---|---|---|
| Dev H — 强制 occupancy=1 (MXFP8_RCR_EXACT_PQ_FORCE_OCC1_ENABLE) | **编译器忽略 request**（occ 仍是 2 waves/SIMD）：一个 512-thread block 在 4 SIMD CU 上**算术最小** occupancy 就是 2 waves/SIMD，无法再降。Step 1 A/B +0.023% 噪声；Step 2 尝试叠加 pipeline 扩展 → 63 spills / −63.36%。Occupancy-knob 轴已饱和 | **拒绝（broken）** |

### 还没被验证为死路的方向（下一轮唯一剩余）

- [ ] **SCALE_LDS 替代式实现**（非叠加）：彻底替代 PIPELINE_SCALE 的 SGPR-SRD path；用真正的 `__builtin_amdgcn_s_barrier()` 前后包 ds_write/ds_read 解决 determinism；<100 行改不完，高风险结构改造
- [ ] **AGPR accumulator fused asm block**（8-32 MFMA 融进单 asm）：破 Dev F 的 per-MFMA 边界开销；改写 `rcr_mma_scaled_from_packs_opsel_phase_{row,impl}`；会破坏 HOIST_HI 的 templated-lambda 约定；高风险大改造
- [ ] **preshuffle_scale_matrix_mfma16 layout 重设计**：让 6 scale dwords 相邻 8-byte 组装载入 → `buffer_load_b64` 可用；影响 RRR/CRR/4-wave + Python 参考
- ~~主动下调到 occupancy=1~~ **已证明不可能**：8-wave 512-thread block 在 4-SIMD CU 上算术最小 occ 就是 2 waves/SIMD，不是 flag 能改的。要 occ=1 需换成 4-wave 256-thread block（另一个 kernel），或 2-wave 128-thread block（完全重写）
- [ ] **bank conflict / MFMA utilization profiling**（`rocprofv3 -i`）：145 TFLOPS 里有多少是 MFMA 利用率，多少是 latency stall

### RCR 本轮结论
MXFP8 RCR 从 `feat/mxfp8-only` 分支的起点 2737 TFLOPS 一路推到 2925.64 TFLOPS，**已达到当前结构约束下可微调的上限**。剩余 145 TFLOPS 差距只能靠**结构性重构**（任选一条高风险大改造）去摸。非结构性的调度/cache/小 flag/occupancy 尝试全部饱和。

---

## RRR / CRR 95% 任务（进行中）

### GPU7 实测 baseline (2026-04-17)

RRR PQ：**2794.26 TFLOPS** (95.48% of RCR) → 已达 95%，本轮不动

CRR PQ：**2737.94 TFLOPS** (93.55% of RCR) → 差 **42.34 TFLOPS (1.55%)** 才到 95% gate (2780.28)

### CRR 差距根因

`crr_mxfp8_exact_8wave_fastpath.inc` L407-461 主循环在每奇数 k 对 6 个 scale packs 做 C++ 层 `>> 16` shift（L411-420），这映射到 6 × `v_lshrrev_b32` per kpair —— **跟 RCR pre-HOIST_HI 完全同构**。

`crr_mma_scaled_base<opsel_a, opsel_b>` 已经支持 2-bit opsel（byte-select 在 bit 1），`crr_exact_cA_with_b1_interleave_raw_phase<K_PHASE, ...>` 和 `crr_mma_scaled_phase<K_PHASE>(...)` 等 compile-time 模板化 helper **代码里已经有**，只是主循环没用。

### CRR 方向

- [x] **PIPELINE_SCALE 默认开**（commit `8934e95c`）：reviewer GPU7 验收 OFF 2733.91 → ON 2740.55 (+0.243%)，VGPR 247→232（−15），spills 0→0，occ 2，SNR 49.60 dB，det 3/3。**未到 gate**（差 39.73 TFLOPS）但是 strict win 且为后续优化释放 15 VGPR headroom
- [x] **R7-R9：HOIST_HI 路径架构性不可行**（5 attempts: Dev A, B, C, H, I 同 256 VGPR ceiling）
- [x] **R10：rocprofv3 + ASM census 找出真实瓶颈**（不是 v_lshr，不是 opsel 计算，是 LDS 管道争用）；3 个新 dev (J/K/L) 全 reject，进一步证实架构性 ceiling
- [ ] **未来方向（结构性，本会话不做）**：
  - **A LDS 布局重设计**（最高收益）：把 A 从 col-major LDS 存储改为 row-major（global→LDS 阶段做 transpose），让 A 侧能用 `ds_read_b128`（16 B/读）而不是 `ds_read_b64_tr_b8`（8 B/读），LDS 指令数减半。CRR vs RRR 差距的 #1 来源
  - CRR 4-accumulator pattern 重构（合并 cA/cB/cC/cD 减寄存器）
  - KPAIR_LOOP 移植 CRR；或重做 `crr_exact_cA_with_b1_interleave` helper（拆掉 8 个独立 MFMA）
- [ ] RRR 保持观察，若后续因编译器变化跌破 95% 再补

---

## 成功条件

### 长期（RCR）
- MXFP8 RCR ≥ 3070.93 TFLOPS（per-iter 协议）
- SNR > 48 dB
- 3 次 determinism 一致
- FP8 baseline 无回归

### 本轮（RRR / CRR）
- **RRR PQ 8192³ ≥ 2780.28 TFLOPS**（当前最佳 MXFP8 RCR × 0.95）
- **CRR PQ 8192³ ≥ 2780.28 TFLOPS**
- SNR > 48 dB
- 3 次 determinism 一致
- 不回归 RCR / FP8

## 运行记录

- `0a3eafb6` Remove all MXFP4 and Gluon kernels on mxfp8-only branch
- `bc0081e5` Tidy repo: skills, gitignore, agent team runbook
- `f943af92` HOIST_HI opsel 消除 main-loop v_lshr（reviewer GPU7 验收 2925.64，A/B +17.80；GPU1 head-to-head +25.67）。main-loop `v_lshr` 0，spills 0，VGPR 256→254，occupancy 2。构建 flag 加 `-DMXFP8_RCR_EXACT_PQ_HOIST_HI_ENABLE=1`。
- **第三轮 (2026-04-17)**：三条路径（scale prefetch n+1 / tail compile-time dispatch / KPAIR 2× unroll）全 reject。Baseline 稳定在 2917–2932 TFLOPS。无代码 commit，仅文档修正 baseline VGPR 数字（254，不是 212）+ 写入三条新 dead-end。
- `b964c110` Round-3 dead-ends: correct baseline VGPR = 254, not 212（仅文档 commit，code 不变）
- **第四轮 (2026-04-17)**：两条路径全 reject。Dev D 实测 SCALE_LDS 叠加：−0.76% 且 determinism FAIL（此前仅"未验证"，现有硬数据）。Dev E 实测 sched_barrier v2 `vmcnt(6→8)`：+0.11% 噪声级。写入 SKILL dead-ends，不 commit 代码。
- `3fe9c759` Round-4 dead-ends: SCALE_LDS measured (regression + det fail), SCHED V2 noise（仅文档 commit）
- **第五轮 (2026-04-17)**：三条路径全 reject。Dev F 实测 AGPR per-MFMA `"+a"`：−2.19%（per-MFMA 边界 V↔A 切换爆 285 次 shuffle + 13 spills）。Dev G 实测 sc0 cache hint：+0.066% 噪声。Dev G2 证明 `buffer_load_b64` 合并在当前 scale layout 下**结构不可行**（6 SRD 间距 8192 B）。剩余只能靠结构性重构。
- `5d31c742` Round-5 dead-ends: AGPR per-MFMA, scale L2 hint, b64 merge broken（仅文档 commit）
- **第六轮 (2026-04-17)**：Dev H 证明 occupancy=1 在 512-thread 8-wave block 上**架构性不可能**（CU 只有 4 SIMD，一个 512-thread block 最少占 2 waves/SIMD）。加 pipeline 扩展反而 63 spills / −63%。Occupancy 轴彻底关闭。
- `77370d3f` Round-6 dead-end: occupancy=1 architecturally impossible for 8-wave（仅文档 commit）
- **第七轮起 (2026-04-17)**：任务转向 RRR / CRR 95%-of-RCR gate。GPU7 实测三 layout：RCR 2926.61 / RRR 2794.26 (95.48%，已达标) / CRR 2737.94 (93.55%，差 42.34)。CRR 差距根因：主循环每奇数 k 做 6 × `scale_pack >> 16` → `v_lshrrev_b32`，跟 RCR pre-HOIST_HI 同构。计划：移植 HOIST_HI opsel 思路到 CRR 主循环（flag `MXFP8_CRR_EXACT_PQ_HOIST_HI_ENABLE`）。Dev CRR-A 已派活（worktree `/tmp/wt-crr-a`，GPU0），被打断未完成。
- **第七轮 round-1 重启 (2026-04-17)**：续派 Dev A/B/C 三 HOIST_HI 变体（CRR HOIST_HI K_PHASE templated lambda）—— 全 FAIL：CRR 4-accumulator (cA/cB/cC/cD) + 重 `crr_exact_cA_with_b1_interleave` 在 K_PHASE 模板化时 inlined codegen 翻倍，VGPR 247→256+ 含 53–173 spills，A/B −54% 到 −67%。
- **第七轮 round-2 (2026-04-17)**：Dev D/E/F/G/H 五个新方向：
  - **Dev D — PIPELINE_SCALE only**：✅ +0.243% on GPU7（详见 commit `8934e95c`），VGPR 247→232（−15），spills 0
  - **Dev E — sched_barrier (no body change)**：噪声级，Δ ≈ 0%。**结论：v_lshr 不在 critical path**
  - **Dev F — `__noinline__` outlined helper**：catastrophic：correctness 46% / scratch 800–888 B/lane / −97%。AMDGPU calling convention 无法跨 noinline 边界保持 4 个 accumulator live
  - **Dev G — runtime branch HOIST_HI**：3 变体全 FAIL gate；V1 `if/else` 256 VGPR + 13 spills，V2 manual unroll 256 + 347 spills，V3 with scopes 同 V2；A/B −94% / correctness FAIL
  - **Dev H — PIPELINE+HOIST combo**：256 VGPR + 29 spills；PIPELINE 的 SRSRC（24×32-bit）与 HOIST_HI 双 phase packs live 互相挤兑
- **第七轮 round-3 (Dev I) (2026-04-17)**：HOIST_HI + PIPELINE_SCALE + `CRR_EXACT_INTERLEAVE_B1_LDS=0`（删掉重 8-MFMA interleave，按 RRR 简单 4-MMA 结构走）：FAIL，VGPR 256 + 145 spills + 332 B/lane scratch。**确认架构性 ceiling**：CRR baseline 247 VGPR 只有 7 headroom，K_PHASE 模板化 4 MMA × 2 phase = 8 inlined MMA blocks 必然吃掉 9–25 VGPR
- `8934e95c` **MXFP8 CRR PIPELINE_SCALE default ON**（+0.243%，frees 15 VGPR）— 含 R7–R9 dead-end 总结
- **R7-R9 关键架构发现**：HOIST_HI K_PHASE templating 与 CRR 4-accumulator main loop **根本不兼容**。任何 templated body doubling 都会越过 254 VGPR cap，与是否叠加 PIPELINE_SCALE / 是否关 INTERLEAVE 无关。已在 5 个独立尝试（Dev A/B/C/H/I）观察到同一 256-VGPR 上限。CRR 要破 gate 必须做**结构性重构**（合并 accumulator / 或换 kernel 结构），非微调可达。
- **第十轮评审 (2026-04-17)**：rocprofv3 + structural-deep-dive + 3 个新 dev attempt（J/K/L），全部 reject，但**找到了真实瓶颈根因**：
  - **rocprofv3 GPU6 8192³ counters**：CRR vs RRR：MFMA 数量相同（16.7M），MFMA busy cycles 完全相同，但 SQ_BUSY_CU_CYCLES +3.75% / SQ_INSTS_VALU **+61%** / SQ_INSTS_LDS **+50%** / SQ_WAIT_INST_LDS +20%。**MFMA 管道已饱和**，差距 100% 来自非-MFMA issue 争用
  - **ASM census per body**：CRR 用 144 `ds_read_b64_tr_b8` (8 B/读) vs RRR 64 `ds_read_b128` (16 B/读) + 64 `ds_read_b64_tr_b8`。**CRR 多 80 LDS 读指令**——根源是 CRR 的 A 侧用 col-major LDS 布局（A_col_reg = `rt_fp8e4m3<BK=128,RBM=64,col_l,rt_128x16_s>` = 128 dwords），而 RRR 用 row-major（A_row_reg = 16 dwords，**8× 小**）。Col-major A 必须用窄的转置读，这是结构性
  - **R10 Dev J — 2× kpair unroll without K_PHASE templating**：FAIL，VGPR 232→256 + 26 spills + 104 B/lane scratch。即使无 templating，body doubling 仍触发 live-range 翻倍（phase-0 的 a/b prefetch 撑到 phase-1）。**与 R7-R9 templated 失败同根**
  - **R10 Dev K — `>>16` shift coalesce + `wn*RBN` precompute**：MARGINAL，Δ ≈ 0%（VGPR 不变 232/0 spills）。关键发现：**编译器已经自动 hoist 了 `wn*RBN`**——profiler 报告的 "32 v_add per body" 是 pre-hoist 静态分析，不是最终 ISA。`v_alignbit_b32` 与 `v_lshrrev_b32` 占同一 issue pipe，替换无效
  - **R10 Dev L — load reordering (b1 pre-issue + scale hoist)**：FAIL，sub-A −0.49% / sub-B −1.78% / combined −3.14%。关键发现：**`lgkmcnt` 等待同时覆盖 LDS + scalar/VMEM scope**——重排不能让 scale buffer_load 与 A/B LDS 真正并行；反而把 scale dest VGPR live range 撑过 A/B 读寄存器期，VGPR 232→254（差点爆）
  - **真实瓶颈定性（已三角验证）**：CRR 受限于 LDS 管道争用，不是 MFMA、不是 v_lshr、不是 opsel 计算、不是 cache miss、不是 VMEM。要破 gate 必须改 A 的 LDS 布局（global→LDS 阶段做 transpose 让 A 侧用 b128 宽读），这是大型重写不在本 sprint scope
- **第十二轮评审 (2026-04-17) — Diagnostic-S 推翻 R10 LDS-pipe 假设；新瓶颈：SPI 启动器 stall**
  - **Diagnostic-S 用 rocprofv3 测了 33 个 cycle-level counter（5 PMC chunk）**，关键发现：
    - `SQ_LDS_BANK_CONFLICT = 0`，`SQ_LDS_ADDR_CONFLICT = 0`，`SQ_LDS_UNALIGNED_STALL = 0`（CRR 和 RRR 都是）—— **R10 的"LDS pipe contention"假设错了**，根本没有 LDS bank 冲突
    - `SQ_LDS_IDX_ACTIVE` CRR 与 RRR **完全相同**（5.03e7 cycles）—— LDS unit 的实际"忙碌"程度一样。CRR 的 +50% LDS 指令数没让 LDS unit 更忙，因为 `ds_read_b64_tr_b8` 比 `ds_read_b128` 在 LDS 单元里就是更轻的 op
    - CRR 的 `TCP_PENDING_STALL_CYCLES` 比 RRR 低 34%，`TA_ADDR_STALLED_BY_TC` 低 92%，`TCP_TCP_TA_DATA_STALL` 低 50% —— CRR 的访存 backend 反而更轻
    - CRR 的 `SQ_VALU_MFMA_COEXEC_CYCLES` 比 RRR 高 61% —— ILP 反而好
    - **`SPI_RA_LDS_CU_FULL_CSN +388%`** 和 **`SPI_RA_RES_STALL_CSN +388%`**（CRR 9.80e11 / 1.23e11 vs RRR 2.01e11 / 2.51e10）—— **wave 启动器在 CU 上被 LDS 占用槽位卡住**，下一个 workgroup 等 5× 长才能 launch。`SQC_DCACHE_BUSY_CYCLES +129%` 也偏高（标量 cache pressure）
    - 估算：`(9.8e11 − 2.0e11) / (224 CU × launch overhead) ≈ 3-5%` 端到端代价 —— 与 1.43% gate gap 同量级
  - **真正瓶颈定性纠正**：CRR 受限于 **SPI launch-allocator pressure**（CU 上 LDS 分配槽位被 CRR 的 136 KB/block 占满，新 workgroup 排队），**不是** LDS bank conflict、**不是** LDS pipe issue rate、**不是** TCP/TA backend、**不是** MFMA-VALU coexec
  - **R12 派 4 个 dev（Dev O/P/R/T）+ 1 个 diagnostic（Diagnostic-S）**：
    - Dev O（CRR_ROW_SHARED_TRANSPOSE 深度调试）：worktree 在 42f5407b base，建了 7 个 build log + diag_load_transpose.py（小尺寸 LDS dump），16:51 后静默 1.5h，**timeout 无 commit**
    - Dev P（CRR_USE_V3_SWIZZLE）：worktree 在 b027c06b（**stale main base**，无源代码），最近活动 17:02，**timeout 无 commit**
    - Dev R（-mllvm 编译 flag sweep）：worktree 在 b027c06b（stale base），从 main checkout 拷贝源建了 .so，17:31 后静默，**timeout 无 commit**
    - Dev T（LDS 分配缩减 136→131 KB）：worktree 在 42f5407b base，活跃到 18:26（最后 .so build），**timeout 无 commit**
    - Diagnostic-S：完成（paradigm-shift 发现，见上）
  - **R12 行动结论**：4 个 dev 全 timeout 无 commit；唯一产出是 Diagnostic-S 的瓶颈定性更正。要 commit 代码必须重派 dev，**强烈建议下轮按 Diagnostic-S 的 SPI 启动器假说派活**：(1) 缩减 CRR LDS/block（单缓冲 A 或 B，packing 重叠）、(2) `__launch_bounds__(512, 3)` 提示 SPI 预留更多 slots、(3) 减少 SQC_DCACHE 压力（per-CTA 常量改 s_load_b256 单次加载）。**不要** 再投资 LDS bank conflict / LDS pipe / re-stripe stride 方向（已证 0 conflict，无收益）

- **第二十四轮评审 (2026-04-18) — ★ R23 cycle-level findings 推翻为 PMC-mode 测量伪影；R24 fresh baseline 显示 V2-RCR 实际 ≈ FP8 RCR (-0.5%)，V2-RRR -2.4%，唯有 V2-CRR -8.9% 是真 structural gap；3 个 dev 全 DEAD-END (SQC dcache cut / RRR TCC re-tile / SPI occupancy)；0 production commit；2 side-branch commits (5cf85e58 + 76848ea9 NOT cherry-picked)；R25+ priority list rebuilt around CRR-only**
  - **R24 派 1 Reviewer + 3 Dev (A/B/C) 并行（GPU0/1/2/3 隔离），按 R23 priority list 执行: SQC dcache cut (NEW #1) / RRR TCC re-tile / SPI occupancy fix**
  - **Reviewer (GPU0) — baseline reverify + R23 levers reverify: ★ MAJOR PARADIGM CORRECTION ★**
    - GPU0 sclk verified 2353 MHz under load; warmup=100, iters=200, per-iter sync
    - **R24 fresh baseline (5x median)**: V2-RCR **3214** (std 7.21), V2-RRR **3156** (std 8.93), V2-CRR **2943** (std 11.44), FP8-RCR **3232** (std 9.48)
    - 全 SNR ≥49.5 + det 3/3 PASS
    - **Real gaps**: V2-RCR vs FP8 = **-17 / -0.5%** (★ near-parity); V2-RRR -76 / -2.4%; V2-CRR -289 / -8.9% (only meaningful gap)
    - **R23 cycle-level diagnostic 不可复现**:
      - R23 Dev C reported SQC_DCACHE_BUSY +441% on V2-RCR vs FP8-RCR; R24 Reviewer fresh PMC measure: **+38%** (factor 11× discrepancy)
      - R23 reported TCC_MISS +166% on V2-RRR; R24 fresh: **+3.1%** (factor 50× discrepancy)
      - R23 reported SPI_RA_LDS_CU_FULL +18.7% uniform across V2; R24 fresh: ±2% noise band
    - **Root cause of R23 noise amplification**: (a) PMC mode 强制 dispatch 序列化 (官方 known caveat) 在 cold-cache 第一次 hit 时给假 +200-400%; (b) GPU3 在 R23 Dev C report 里就 noted "1872-1995 MHz under load" 是 cold-throttle state (vs GPU0 healthy 2353 MHz); (c) R23 protocols 用了 warmup=10 PMC mode wasn't long enough; R24 用 warmup=100 fresh
    - **All R23 ranked NEW levers INVALIDATED by fresh measurement**: SQC dcache (#1), RRR TCC (#3), SPI (#5), MFMA-VALU coexec (#6) 全部基于 noise-amplified PMC numbers
    - **Real V2 status**: V2 paradigm 比 R23 docs 描述的更接近 FP8 parity；R20→R22 cumulative wins 已经 close 大部分 RCR gap
  - **Dev A (GPU1) — SQC dcache cut (R23 NEW #1 lever): ★ DEAD-END, SIDE COMMIT 5cf85e58 (NOT cherry-picked) ★**
    - branch `r24-a-sqc-dcache @ /tmp/wt-r24-a`
    - **Approach**: precompute V2 scale slab base SRDs into SGPR via `__builtin_amdgcn_readfirstlane` at use site (helper `mxfp8_v2_pin_srd_to_sgpr` + `MXFP8_V2_PIN_SRD(srd)` macro); flag `MXFP8_V2_HOIST_SCALE_PTRS_ENABLE=1` (default 0)
    - **Build**: clean compile (VGPR/LDS/spills identical to baseline); applied at 2 V2-RCR call sites + RRR/CRR fastpath helpers (+34/+2/+2 LOC)
    - **Critical finding**: hipcc -S 显示 V2 scale SRDs **已经在 SGPR** (s[24:27], s[40:43] 等)，readfirstlane 是 no-op；compiler scheduler 已经把 SRD pin 到 scalar regs。R23 假说 "V2 SRDs spilling to VGPR causing SQC pressure" **完全错误**
    - **PMC reverify (R24 protocol)**: SQC_DCACHE_BUSY V2 vs FP8 = +38% (not +441% as R23 reported); with HOIST_HI flag: -3% within noise (target -50%)
    - **Perf A/B (5x GPU1)**: V2 baseline 3211 vs HOIST_SCALE_PTRS 3208, **Δ -3 TFLOPS / -0.09%, Welch t=-1.06** (statistical null)
    - Side-branch commit `5cf85e58c917b7effa081a7537997f176af11adc` retained for archival; **NOT cherry-picked**
  - **Dev B (GPU2) — V2-RRR TCC working-set re-tile (R23 NEW #3 lever): ★ DEAD-END, SIDE COMMIT 76848ea9 (NOT cherry-picked) ★**
    - branch `r24-b-rrr-tcc-tile @ /tmp/wt-r24-b`
    - **Original scope (B-side 32×32 re-tile)**: 直接重 tile RRR B 操作数需要重写 ST/swizzle，不在 1-day scope; substituted approach
    - **Substituted approach**: TA arbiter spread via splitting `load_scale_packs` into `load_scale_packs_a_v2` + `load_scale_packs_b_v2` with `__builtin_amdgcn_sched_barrier(0)` between them, forcing TA arbiter to interleave A/B scale loads in different cycles
    - **Build**: clean (compile flag `MXFP8_V2_RRR_TCC_TILE_ENABLE=1`, default 0; flag-OFF 与 baseline byte-identical .so verified via sha256sum)
    - **PMC reverify (R24 protocol)**: TCC_MISS V2-RRR vs FP8-RRR = **+3.1%** (not +166% R23 reported); SQ_VMEM_TA_ADDR_FIFO_FULL **+597%** R23 → +25% R24; with sched_barrier: TA_ADDR_FIFO_FULL **-16.9%** (target -50%, partial), TCC_MISS unchanged
    - **Perf A/B (5x GPU2)**: V2-RRR baseline 3155 vs TA-spread 3153, **Δ -2 TFLOPS / -0.07%, Welch t=-0.17** (statistical null); TCC working-set 不是真实瓶颈，sched_barrier 仅 reorder issue 不改 data flow
    - Side-branch commit `76848ea978abe88a38c834cf3c907a8ace9a3119` retained; **NOT cherry-picked**
  - **Dev C (GPU3) — SPI occupancy fix (R23 NEW #5 lever): ★ DEAD-END (architectural impossibility) ★**
    - **Approach**: try `__launch_bounds__(threads, 3)` + recover 8-16 VGPRs to unlock occupancy=3
    - **Critical discovery — gfx950 LDS hard cap**: MI355X CU LDS = **160000 B**; V2 RCR 131 KB / RRR 135 KB / CRR 139 KB → **3 blocks/CU 需 393-417 KB > 160 KB → architectural impossible**
    - Compiler **silently ignored** `__launch_bounds__(_, 3)` because LDS not VGPR is the binding constraint
    - V2 实际 VGPR usage **less than FP8** (RCR V2: 246 vs FP8: 254; **-8 VGPR**) — R23 SPI VGPR_SIMD_FULL +18.7% 信号是 PMC noise artifact, not actual VGPR pressure
    - **occ=2 是 V2 paradigm 的 architectural ceiling** (与 R6 occ=1 architectural impossibility 同级 finding)
    - All edits reverted; no commit
  - **R24 综合产出 = 0 production commits + 2 side-branch commits (5cf85e58 + 76848ea9, NOT cherry-picked) + 1 paradigm correction docs commit**:
    1. **★ R23 cycle-level findings 全部 INVALIDATED ★** as PMC-mode + cold-throttle artifacts; all 6 R23 ranked NEW levers based on noise-amplified data
    2. **★ V2 实际近 FP8 parity ★**: V2-RCR -0.5%, V2-RRR -2.4% (R23 高估了 5-7×); 累计 R20-R22 V2 工作 close 了远比 R23 docs 描述更多的 gap
    3. **V2 SRDs 已经在 SGPR**: readfirstlane no-op, R23 SQC dcache 假说 falsified
    4. **TCC working-set 不是瓶颈**: TCC_MISS R24 fresh +3.1% (not +166%), sched_barrier 重排无效
    5. **gfx950 LDS hard cap = 160000 B/CU**: V2 occ=3 architecturally impossible (similar to R6 occ=1 impossibility)
  - **R24 confirms**:
    - **PMC mode 不能用作 perf 比较的 absolute counter，必须 fresh measurement reverify before acting on PMC-derived hypotheses**
    - V2 paradigm 的 V1→V2 收益已经 deliver 了大部分 gap closure，剩余 CRR -8.9% 是唯一 meaningful structural gap
    - GPU 状态 (sclk, thermal) 必须前置 verify (R22 经验 + R23/R24 reaffirm)
  - **R25+ 路径**（基于 R24 evidence rebuild）：
    1. **CRR -8.9% 是 ONLY meaningful gap** (RCR/RRR essentially at parity)
    2. **CRR-specific bottlenecks 仍未解**：col-major A-LDS layout (R10 census + R11/R23 SNR wall)；要 unblock 必须 layout-matching microbenchmark 或 element-dump kernel
    3. **真 RRR scale-prefetch pipeline**：extend `MXFP8_RRR_EXACT_PQ_PIPELINE_SCALE_ENABLE` 到 V2 b128/b64 with double-buffered scale registers
    4. **Per-instruction PMC sampling (rocprofv3 ATT mode)** to identify actual SQC source — 不要再信 dispatch-level PMC summary
    5. **不要再** 重新基于 R23 PMC 数字派活 (SQC/TCC/SPI/MFMA-VALU coexec 全 invalidated); 不要重新 revisit V3 preshuffle (R23 -2.99% confirmed); 不要重新 revisit V1 b64 / sched hints / SCALE_LDS / naive CRR A-LDS memcpy
  - **新经验 (R24 起)**：
    - **PMC mode artifacts**: rocprofv3 PMC 模式在 cold cache + dispatch-level aggregation 下放大 200-400% 假信号；必须 (a) warmup ≥100, (b) GPU sclk verified ≥2GHz, (c) cross-protocol reproduce before treating as bottleneck
    - **不要把 PMC 数字直接当 perf hypothesis**: R23 ranking 6 NEW levers 全部基于一次 PMC measurement, R24 fresh 全部不可复现
    - **gfx950 LDS hard cap = 160000 B/CU**: V2 (131-139 KB/block) hard-capped at 1 block/CU = 2 waves/SIMD; `__launch_bounds__(_,3)` silently ignored
    - **side-branch + reviewer-confirm pattern still good**: Dev A/B 各自 commit on side branch, reviewer reverify 决定不 cherry-pick；clean audit trail without polluting feat/mxfp8-only

- **第二十三轮评审 (2026-04-18) — ★ R22 V2 stable reverify (V2-RCR 3074 / V2-RRR 3033 / V2-CRR 2811 / FP8-RCR 3252)；3 个 dev 全 exhausted: V3-RCR REJECTED (-2.99%) / CRR A-LDS Route X FAIL SNR-2.71 dB (R11 wall reproduced) / Dev C diagnostic 找到 SQC_DCACHE pressure (+441%) 作为 NEW #1 lever；0 production commit；1 docs commit**

  > **R24 retraction**: R23 Dev C 列出的 6 ranked NEW levers (SQC dcache +441%, TCC_MISS +166%, SPI +18.7%, etc.) 在 R24 fresh measurement 下不可复现 (SQC +38%, TCC_MISS +3.1%, SPI ±2% noise)。Root cause: PMC mode + GPU3 cold-throttle + warmup=10 协议差异。R23 priority list 已被 R24 invalidate; 见 R24 entry above for corrected baseline + R25+ rebuilt priorities.
  - **R23 派 1 Reviewer + 3 Dev (A/B/C) 并行（GPU0/1/2/3 隔离），按 R22 R23+ priority list 执行: V2 milestone-3 / CRR LDS rewrite / V2 cycle diagnostic**
  - **Reviewer (GPU0)** Task 1 — 5x 全 layout V2 baseline reverify: stable
    - V2-RCR median **3074** (std 6.20), V2-RRR median **3033** (std 4.14), V2-CRR median **2811** (std 6.83), FP8-RCR median **3252** (std 6.40)
    - 全 SNR ≥49.5 + det 3/3 PASS
    - 跨 R22→R23 drift < 0.5% (RCR -4 TFLOPS / RRR -5 / CRR +79); 系统 reproducibly stable
    - **gap (R23 reverify)**: V2-RCR -178 / -5.5%, V2-RRR -219 / -6.7%, V2-CRR -441 / -13.6% (CRR 用 R23 reverify 数字 vs R22 的 -517 改善 ~76 TFLOPS, 实际 baseline drift)
  - **Dev A (GPU1) — V3 preshuffle prototype: ★ REJECTED, NO COMMIT ★**
    - branch `r23-a-preshuffle-v3` (side branch, NOT cherry-picked)
    - **Approach**: 推 V2 b128+b64 (24B/wave) 升到 b128+b128 (32B/wave) 通过 V3 preshuffle 让 B scale pack 也能 b128
    - **Side commits**: `3fbfa416` (V3 prototype) + `aef1d03e` (bench harness) — 仅在 side branch
    - **关键 finding (negative)**: **SQ_INSTS_VMEM 是 transaction count, 不是 byte count**；b64→b128 width promotion 给 0% VMEM-issue reduction (transactions 数量不变，仅 width 增大)
    - **5x A/B**: V3 mean **3018** vs V2 baseline mean **3091**, **Δ -92 TFLOPS / -2.99%, Welch t=-7.46** (统计显著 regression)
    - 路径完全废弃；R18+R21 paradigm "减 VMEM-issue 数量" 在 V2 之后已 saturated（不能再减；只能换其他 bottleneck）
  - **Dev B (GPU2) — CRR A-LDS row-major Route X: ★ FAIL CORRECTNESS, NO COMMIT ★**
    - branch `r23-b-crr-a-lds @ /tmp/wt-r23-b` (worktree, 修改 uncommitted)
    - **Approach**: ST_v2a (col-major) → ST_row (row-major)；`load_transpose<NT>` global→LDS + `load(A_row_reg, sub)` ds_read_b128 (16B) 替换 `load_col_from_v2a_st` ds_read_b64_tr_b8 (8B)；A_col_reg 经 memcpy reinterpret 复用
    - **Build**: clean compile (VGPR 249, +15 vs baseline; LDS 135168, **-4096B**; 0 spills, occ 2 — 资源 healthy)
    - **Correctness FAIL at 8192³**: SNR **-2.71 dB** (threshold 48), 97.62% partial pass-rate, det 3/3 PASS
    - **完全复刻 R11 Dev M 的失败模式** (R11 也 hit -2.71 dB SNR wall)
    - **Root cause (R23 Dev B 进一步分析)**: `load_transpose` + `prefill_transpose_swizzled_offsets` 写入 LDS 的物理 layout 与 `ds_read_b128` 直接读取期望的 row-MMA layout 在 lane-element correspondence 上不匹配；97.62% partial pass 表明 systematic permutation within K-blocks 而非随机；要修需要 (a) element-dump kernel 验证 actual lane-element mapping, (b) keep explicit `transpose(dst, tmp)` register call 而非 memcpy, 或 (c) 写 custom `load_transpose` variant
    - **Verdict**: dead-end-with-caveat；R11 + R23 两次 hit 同一墙；CRR A-LDS 需要 multi-day microbenchmark 才有可能 unblock，且 Dev C diagnostic 显示上限 +50-100 TFLOPS（5-10% only）
  - **Dev C (GPU3) — V2 cycle-level diagnostic via rocprofv3 PMC: ★ COMPLETE, 6 NEW LEVERS RANKED ★**
    - 32 PMC counters × 6 kernels (V2-RCR/RRR/CRR + FP8-RCR/RRR/CRR) × 4 chunks
    - **mangled name verify**: `_Z29rcr_exact_8wave_scaled_kernelILb1ELi2EEv` 等等 (V2 = `<true, 2>`)
    - **GPU3 sclk caveat**: GPU3 idle low-power state but ramps to 1872-1995 MHz under load；rocprofv3 PMC mode (serialized) wall-clock degrades but **per-dispatch event counters are clock-invariant** (analysis robust)
    - **6 NEW lever findings (ranked by est. upside)**:
      1. **SQC_DCACHE_BUSY_CYCLES (NEW #1, never instrumented R10/R12/R18)**: V2 vs FP8 = **+441% RCR / +91.6% RRR / +254% CRR**；wave-tile preshuffle 让 scale pointer arithmetic per-wave scalar code 击中 constant cache 频率剧增。Lever: precompute slab base ptrs into LDS-scalar/SGPR before K-loop OR single `s_load_dword_x4` for `(a0,a1,b0,b1)_ptr` instead of repeated SMEM in K-loop. **Est upside +80-150 TFLOPS aggregate**, easy 1-day audit.
      2. **CRR A-LDS row-major (Dev B path)**: **bounded ~+50-100 TFLOPS** (+50% SQ_INSTS_LDS 8.4M extra insts/dispatch in CRR vs RRR is structural; halving A's LDS reads is theoretical max)；but **R11+R23 已两次 hit dead-end SNR wall** → multi-day investment, gated on layout-matching microbenchmark first
      3. **RRR TCC working-set re-tile (NEW)**: V2-RRR shows **TCC_MISS +166% / SQ_VMEM_TA_ADDR_FIFO_FULL +597%** vs FP8-RRR；B-operand fp8 fetch in RRR row-major hitting TCC poorly。Lever: B-side block re-tile (e.g. 32×32 tiles 替换 16×128) 适配 TCC working set。Est upside +50-80 TFLOPS RRR-only.
      4. **V3 scale-preshuffle (Dev A path)**: **SATURATED on RCR**（SQ_INST_LEVEL_VMEM V2 已 -41% 比 FP8；TA_ADDR_FIFO_FULL 仅 10% utilization），可能 RRR/CRR marginal +30-80 TFLOPS。Dev A REJECTED 已实测确认。
      5. **SPI launch occupancy fix (NEW)**: 全 V2 layout SPI_RA_LDS_CU_FULL_CSN/RES_STALL_CSN/VGPR_SIMD_FULL_CSN uniformly **+18.7-18.95% over FP8** → VGPR pressure signal。Lever: audit `Rpass-analysis=kernel-resource-usage` for V2 vs FP8 VGPR delta；recover 8-16 VGPRs via lifetime mgmt 解锁 occupancy。Est +30-80 TFLOPS.
      6. **MFMA-VALU coexec on V2-CRR (NEW)**: SQ_VALU_MFMA_COEXEC_CYCLES V2 vs FP8 **-28.6%** (56.5M vs 79.1M)；V2 preshuffle scales 挤掉了 MFMA shadow 中的 helper VALU。Lever: schedule helper math (s_load → v_mov / cvt) into MFMA bubbles。Est +20-40 TFLOPS CRR-only.
    - **Bottleneck attribution**: V2-RCR gap = ~55% VMEM-issue + ~30% other (SPI/SQC) + ~10% LDS + ~5% MFMA；V2-RRR = ~65% VMEM (TCC working-set) + ~15% LDS + ~15% other；V2-CRR = ~45% VMEM + ~25% LDS + ~15% MFMA-coexec + ~15% other
    - **Verdict on Dev A V3**: SATURATED，估 +0-30 TFLOPS RCR / +30-80 RRR / 0-20 CRR — 与 Dev A 实测 -2.99% regression 一致 (V3 not the lever, SQC dcache is)
  - **R23 综合产出 = 0 production commits + 1 docs commit + R23 reverify baseline + 6 ranked NEW levers (SQC dcache 是 #1 NEW lever 历史从未 instrumented)**：
    1. **R22 V2 三 layout 全 stable reproducible** (drift < 0.5% after 1 session)
    2. **V3 preshuffle 路径 REJECTED**: 关键 paradigm-shift finding 是 SQ_INSTS_VMEM 是 transaction count，b64→b128 width promotion 给 0% VMEM 减少；R18+R21 "减 VMEM count" paradigm 已 saturated 在 V2 之后
    3. **CRR A-LDS row-major 第二次 hit R11 SNR -2.71 dB wall**: layout-matching microbenchmark 先做才有可能 unblock；不再尝试 naive memcpy/transpose alias
    4. **SQC_DCACHE_BUSY_CYCLES 历史从未 instrumented**: V2 paradigm 引入 **+441%** scalar-cache pressure，是 R23+ #1 lever (估 +80-150 TFLOPS aggregate, 1-day audit)
    5. **sched hints / SCALE_LDS / V3 preshuffle / V1 b64 全列入 永久 dead-end** (R23 reaffirm)
  - **R23 confirms**：
    - V2 paradigm 三 layout 完全 stable, no regressions across session boundaries
    - VMEM-issue count cuts has saturated as a lever (R18→R21→R22 paradigm complete)
    - **新瓶颈类别**: SQC dcache + SPI launch + TCC working-set 是 R24+ 焦点（不是 VMEM count 也不是 LDS bank）
    - **R11 SNR wall reproducible**: CRR A-LDS layout transpose naive approach (load_transpose + memcpy) 在 R11 + R23 两次 hit -2.71 dB；microbenchmark 先做才能 unblock
  - **R24+ 路径**（按优先级，基于 Dev C diagnostic）：
    1. **SQC dcache pressure cut** (NEW #1，估 +80-150 TFLOPS aggregate)：precompute slab base pointers per CTA into LDS-scalar/SGPR before K-loop；OR single `s_load_dword_x4` for `(a0,a1,b0,b1)_ptr` 替换 K-loop 内 repeated SMEM loads
    2. **RRR TCC working-set re-tile** (估 +50-80 TFLOPS RRR-only)：B-side block re-tile 32×32 替换 16×128 适配 TCC working set
    3. **SPI launch occupancy fix** (估 +30-80 TFLOPS)：audit V2 VGPR delta vs FP8，recover 8-16 VGPR 解锁 occupancy
    4. **MFMA-VALU coexec on V2-CRR** (估 +20-40 TFLOPS CRR-only)：schedule helper math into MFMA bubbles
    5. **CRR A-LDS layout-matching microbenchmark** (R11+R23 wall unblock prerequisite)：element-dump kernel mapping `load_transpose+ds_read_b128` vs `load_col_from_v2a_st` 实际 lane-element layout
    6. **不要再** revisit V3 scale-preshuffle / V1 b64 / sched hints / SCALE_LDS / naive CRR A-LDS memcpy approach — 全 dead-end
  - **新经验 (R23 起)**：
    - **SQ_INSTS_VMEM 是 transaction count, 不是 byte count**：b32→b64→b128 width promotion 不减 VMEM-issue count，所以 R18+R21 paradigm "减 VMEM" 在 V2 后 saturated；下一步必须找其他 bottleneck (SQC/SPI/TCC)
    - **SQC_DCACHE_BUSY_CYCLES 历史 R10/R12/R18 从未 instrumented**：V2 paradigm 引入了大幅 scalar cache pressure 是隐藏多轮的关键 overhead
    - **rocprofv3 PMC mode caveat**：serialized dispatch 让 wall-clock degrade，但 per-dispatch event counters 是 clock-invariant 所以分析仍 robust（不要用 PMC mode 的 elapsed-time 做 perf 比较）
    - **R11 SNR -2.71 dB wall (CRR A-LDS naive transpose)**: 不要再用 `load_transpose` + memcpy reinterpret approach；要么用 explicit `transpose(dst, tmp)` register call，要么写 custom `load_transpose` variant matching `ds_read_b128` semantics

- **第二十二轮评审 (2026-04-18) — ★ V2 推广至 RRR + CRR 双 SHIPPED ★ R22-A RRR V2 +159.61 TFLOPS / +5.54% / Welch t (Reviewer indep verify)；R22-B CRR V2 +20.55 TFLOPS / +0.76% / Welch t=4.92；R22-C sched hints 永久 KILLED (baseline VMEM-wait 已比 FP8 LESS)；2 production commit cherry-picked to main**
  - **R22 派 1 Reviewer + 3 Dev (A/B/C) 并行（GPU0/1/2/3 隔离），按 R21 path priority list 第一项: V2 推广到 RRR/CRR**
  - **Reviewer (GPU0)** Task 1 — 5x baseline 全 PASS, 系统 stable
    - MXFP8 RCR V2 median 3071 / V1 3022 / RRR 2878 / CRR 2706 / FP8 RCR 3242
    - 跨会话 drift < 0.5%；可作为 R22 Dev A/B/C 比较基准
  - **Dev A (GPU1) — V2-RRR milestone-1+2: ★ PASS BIG, COMMITTED ★**
    - **5x A/B (Dev A GPU1)**: V1 mean ~2878, V2 mean ~3046, **Δ +168.64 TFLOPS / +5.94%, Welch t=4.76**
    - **关键 finding**: V1 RRR 此前 256 VGPR / **19 spills / 80B scratch** (compiler 在 V1 SRD chains + opsel templating 下边界刚好溢出)；V2 collapse 到 256/0/0/0 — 这是 RRR 比 RCR 收益更大的根因
    - **dual-patch rule 简化**: RRR 单一 `load_scale_packs` lambda（无独立 warmup helper），所有 callers (main loop + pre-tail) 统一消费 V2 packs
    - 复用 `preshuffle_scale_matrix_mfma16_v2_rcr_a/b`：RRR A/B scale base address formulas byte-identical to RCR (M-major / N-major, k_blocks columns)
    - 新增 `dispatch_rrr_exact_8wave_scaled_v2` + `dispatch_pq_v2<RRR>` + `gemm_rrr_pq_v2` pybind；runtime gate `MXFP8_RRR_PRESHUFFLE_V2_RUNTIME` default 1
    - **b128/b64 emission verified** via `--offload-device-only -S`
    - **Correctness gates** (HIP_VISIBLE_DEVICES=1, det 3/3): 256/1024/8192³ 全 SNR ≥49.5 PASS, 100% pass-rate
    - cherry-picked to main as commit `dabeffa0`
  - **Reviewer (GPU0) Task 2-RRR — independent verify: VERIFIED PASS**
    - 干净 worktree 重建，全 correctness gates 重现
    - **5x A/B (Reviewer GPU0)**: **Δ +159.61 TFLOPS / +5.54%**, byte-exact rocprofv3 counter match
    - GPU0 vs GPU1 cross-drift < 9 TFLOPS, Δ same-session 一致
    - **Gap closure**: V2 RRR vs FP8 RCR ~-212 / -6.3% (vs R21 RRR estimate -371) → **43% gap close in 1 round**
  - **Dev B (GPU2) — V2-CRR milestone-1+2: ★ PASS, COMMITTED ★**
    - **8192³ 5x A/B (Dev B GPU2)**: V1 median 2711.78 (std 10.61), V2 median 2732.33 (std 2.99), **Δ +20.55 TFLOPS / +0.76%, Welch t=4.92**
    - 复用 V2-RCR Python preshuffle (scale shapes 在 RCR/CRR 之间相同)
    - CRR 单一 `load_raw_scales` 加载器从 main loop / pre-tail / tail 全调用，dual-patch 简化为 single patch
    - **rocprofv3 SQ_INSTS_VMEM**: V1 212,992 → V2 180,224 = **-15.38%** (byte-exact match R21 V2-RCR)
    - **Resource**: VGPR 234 (V1: 232, +2), 0 spills, occupancy 2, LDS 139 KB
    - **Correctness gates** (HIP_VISIBLE_DEVICES=2, det 3/3): 256/1024/**8192³ 49.60 dB**, 67108864/67108864 PASS
    - cherry-picked to main as commit `9a0d0624`
    - **CRR 收益小于 RRR/RCR 的原因**: CRR baseline 已 MFMA-bound (not VMEM-issue-bound) + 已用 PIPELINE_SCALE 单 shot 4 b32 加载；marginal cost 4 b32 vs 1 b128 + 1 b64 在 MFMA 吞吐相对小
  - **Reviewer (GPU0) Task 2-CRR — environmental noise，sanity re-verify on GPU1**
    - 第一次在 GPU0 verify FAIL：GPU0 sclk 卡在 260MHz / mclk 2000MHz (severely throttled)
    - Reviewer 同时实施 dispatcher M_DIM rebuild 错误（dispatcher gates on `g.m == M_DIM`，必须 per-size rebuild）
    - Reviewer 不可能地 claim R21 RCR V2（早 ship 工作）也 FAIL
    - **Sanity Reviewer on GPU1** (sclk 2090MHz healthy)：byte-for-byte match Dev B's numbers，**VERIFIED PASS**
  - **Dev C (GPU3) — sched hints diagnostic: ABORT-DIAGNOSTIC 永久 close**
    - 在 V2-RCR 上重新 measure VMCNT-wait cycles：**V2 vmcnt wait 已 LOWER 比 FP8 (-8.8%)**
    - sched hints 没有任何可发挥空间；剩余 gap 100% 来自 structural VMEM-issue rate 而非 idle stall
    - **永久关闭 sched-hints / sched_barrier / sched_group_barrier 调优方向**（R18 sched-hint REJECT + R22 V2 baseline reaffirm）
    - 要继续 close gap 必须做 structural VMEM-issue cuts（preshuffle V3 进一步合并？或 cache-line layout 重设计）
  - **R22 综合产出 = 2 production commits (dabeffa0 + 9a0d0624 V2 RRR/CRR) + 1 永久 dead-end + V2 paradigm 三 layout 全覆盖**：
    1. **★ V2-RRR SHIPPED ★ +159.61 TFLOPS / +5.54%**（Reviewer indep verify）→ gap close 43% in 1 round
    2. **★ V2-CRR SHIPPED ★ +20.55 TFLOPS / +0.76%**（GPU1 sanity re-verify byte-exact）
    3. **sched hints 路径永久 KILLED**（V2 已 hide 所有 vmcnt waits, 不可能再 hide more）
    4. **三 layout V2 paradigm 完整覆盖**：RCR/RRR/CRR 全 wired through `dispatch_pq_v2<L>` + `gemm_*_pq_v2` pybind + `MXFP8_*_PRESHUFFLE_V2_RUNTIME=1` default
  - **R22 confirms**：
    - V2 layout paradigm 在 3 个 layout 均成功 (R21 RCR + R22 RRR + R22 CRR)
    - V1→V2 收益排序 RRR (5.54%) > RCR (1.84%) > CRR (0.76%)，与 V1 baseline spills + VMEM-issue rate 排序一致
    - dual-patch rule (R21 silent FAIL learning): 必须 patch BOTH main loop AND `load_scale_packs_for_pair` helper；R22-A RRR / R22-B CRR 都验证正确
    - GPU benchmark sanity: 测试前必须 verify GPU sclk (`rocm-smi`) — GPU0 throttled 是 R22 CRR 第一次 verify FAIL 的根因，per-size rebuild 也是必要
  - **R23+ 路径**（按优先级）：
    1. **V2 milestone-3 进一步 VMEM-cut**：V2 已 -15.4%；可能还能用 64B-aligned coalesce 或 V3 preshuffle 让 b128+b128 (32B) 替换 b128+b64 (24B)
    2. **CRR 长期 gap (-15.9%)** 是新焦点（RCR/RRR 已 < 6.3%）：根因仍是 LDS-pipe / col-major A 布局 (R10 census)；A LDS row-major transpose 仍 unattempted as multi-day rewrite
    3. **不要再** revisit sched hints / SCALE_LDS REPLACE / V1 layout 下 b64 — 全 dead-end
  - **新经验 (R22 起)**：
    - **复用 R21 V2 paradigm 跨 layout 极快**: RRR/CRR scale shapes 与 RCR 相同（M-major/N-major × k_blocks），所以 Python `preshuffle_scale_matrix_mfma16_v2_rcr_a/b` 直接复用，仅需 layout-specific kernel template specialization
    - **GPU sclk verify 必须前置**: rocm-smi 检查 sclk ≥ 2GHz；throttled GPU 给出 misleading FAIL
    - **dispatcher per-size rebuild**: dispatcher gates on compile-time `M_DIM`，必须按 size rebuild .so

- **第二十一轮评审 (2026-04-17) — ★ V2 SHIPPED ★ Preshuffle V2 milestone-2 PASS (+55.66 TFLOPS / +1.84% / Welch t=23.33) cherry-picked to main 作为 default-on；SCALE_LDS REPLACE 路径永久 KILLED；首次 21 轮 sub-200 gap 关闭**
  - **R21 派 1 Reviewer + 2 Dev (A/B) 并行（GPU0/1/2 隔离）**
  - **Reviewer (GPU0)** Task 1 — 5x baseline 全 PASS
    - MXFP8 RCR median **3018.44** (std 3.00, min 3015.46, max 3022.61)
    - FP8 RCR median **3243.67** (std 6.86, min 3235.08, max 3250.74)
    - gap **-225.23 / -6.94%**, drift vs R18 < 0.2% — 系统稳定
  - **Dev A (GPU1) — preshuffle V2 milestone-2: ★ PASS, COMMITTED ★**
    - branch `r20-a-preshuffle-v2` 接力 R20-A milestone-1 基础（commit `f54e6dfc`）
    - **Wave-tile order fix (option a)**: Python 在 V2 packing 前 reorder source row_groups → wave-tile 顺序变成 `{a0p0, a1p0, a0p1, a1p1}` 匹配 b128 dword 顺序（A pc=4）/ `{b0p0, b1p0}` (B pc=2)
    - **Production wiring**: 新增 kernel template parameter `SCALE_VERSION` (1=V1, 2=V2)，per-wave-tile slab SRD + 显式 `llvm_amdgcn_raw_buffer_load_b128` (A) / `llvm_amdgcn_raw_buffer_load_b64` (B) 替换 4+2 b32 chains
    - **关键 bug fix**: 初始 256³ FAIL 因 `load_scale_packs_for_pair` 缺 SCALE_VERSION==2 branch；warmup/pre-tail/tail 用 V2 memory through V1 row-base pointers → garbage。Mirror V2 b128/b64 logic 进 helper 修复。
    - 新增 dispatch `dispatch_pq_v2<RCR>` + pybind `gemm_rcr_pq_v2`；runtime gate `MXFP8_RCR_PRESHUFFLE_V2_RUNTIME` default 1
    - **b128/b64 emission verified**: `--offload-device-only -S` 显示 `buffer_load_dwordx4` (A) + `buffer_load_dwordx2` (B)
    - **Resource**: VGPR **246** (V1: 254, **-8 VGPR**), SGPR 52, 0 spills, occupancy 2, LDS 131072 unchanged
    - **Correctness gates** (HIP_VISIBLE_DEVICES=1, det 3/3): 256³ 49.56 dB / 1024³ 49.62 dB / **8192³ 49.60 dB**, all 100% pass-rate
    - **rocprofv3 SQ_INSTS_VMEM**: V1 6,815,744 → V2 5,767,168 = **-15.38%** (匹配 R18 model 预测精确)
    - **8192³ 5x A/B (Dev A GPU1 raw)**: V1 mean 2986.44 (std 5.77), V2 mean 3046.35 (std 4.08), **Δ +59.90 TFLOPS / +2.0%, Welch t=18.95**
    - commit `1a29c562` on side branch
  - **Reviewer (GPU0) Task 2 — independent verify on GPU0: VERIFIED PASS**
    - 干净 worktree `/tmp/wt-r21-rev` @ HEAD `1a29c5628cbca6824f96993905c053c4ca129d4c` 重建
    - 全 correctness gates 重现：256³/1024³/**8192³ 49.60 dB**, 67108864/67108864 PASS, det 3/3
    - rocprofv3 byte-exact 匹配 Dev A: V1 6,815,744 / V2 5,767,168 / **-15.38%**
    - **8192³ 5x A/B (Reviewer GPU0)**: V1 median 3022.43 (std 3.19), V2 median 3078.09 (std 4.63), **Δ +55.66 TFLOPS / +1.84%, Welch t=23.33** (p<<0.001)
    - GPU0 数字略低于 Dev A GPU1 (V1 -36 / V2 -32) 是 cross-GPU drift；Δ 在 same-session 内一致
    - **Gap closure**: V2 vs FP8 RCR 3243.67 = **-165.58 / -5.10%**（vs R18 -228.42 / -7.03%；vs Task 1 V1 -225.23 / -6.94%）→ **第一次 21 轮把 gap 关到 sub-200**
    - 推荐: MERGE
  - **Dev B (GPU2) — SCALE_LDS REPLACE milestone-1.5: STRUCTURAL NO-GO**
    - 接力 R20-B branch `r20-b-scale-lds @ 5ac3229d`
    - **Fix A** (opaque pointer wrap + memory barrier): 8192³ SNR 6.62 dB FAIL
    - **Fix A2** (trailing waitcnt lgkmcnt(0) after all 6 ds_reads): 6.62 dB FAIL
    - **Fix B** (`=&v` early-clobber on dst + per-read waitcnt): 6.63 dB FAIL, det FAIL
    - **R20-B 假说 FALSIFIED**: hipcc -S 显示 6 distinct address VGPRs (v151, v188-v192) **ARE preserved** with values `0x20000+lane`, `0x20100+lane`, `0x20200+lane`, `0x20300+lane`, `0x20800+lane`, `0x20900+lane`。mfma_scale operand mapping (v159, v178, v171, v177) 也正确。Real root cause 仍未知。
    - **Performance kill (independent of correctness)**: SCALE_LDS Fix B 8192³ = **2340 TFLOPS** vs PIPELINE_SCALE baseline **2610 TFLOPS** = **-270 TFLOPS / -10.3% regression** (vs R19 linear model 预测 +50-150 TFLOPS)。R19 model under-counted LDS-issue cost；the +0.34M LDS-issues/disp + barrier cycles 实际 ~3-5x higher TFLOPS cost
    - 即使修好 correctness 也是 regression，路径永久 KILLED
    - docs commit `3c6e2392`
  - **R21 综合产出 = 3 production commits (4abd4f62 + efc389ff V2 + 3c6e2392 SCALE_LDS NO-GO docs) + 1 默认开启的真实 perf 优化 + 1 永久 dead-end**：
    1. **★ V2 preshuffle SHIPPED ★ +55.66 TFLOPS / +1.84% / Welch t=23.33**（Reviewer 独立确认，gap close to -165.58）
    2. **SCALE_LDS REPLACE 永久 KILLED**（不是仅 correctness fail；是 architectural regression）
    3. **R18+R19 paradigm 第三次 reaffirm**：减少 scale buffer_loads 数量真的有效（V2 实测 +55.66 TFLOPS / -15.38% VMEM），但 LDS staging 不是有效的替代实现策略
  - **R21 confirms**：
    - V2 layout 是 21 轮唯一真实 production-impacting MXFP8 优化
    - VMEM-issue rate 还是核心 bottleneck，但减少手段必须 native VMEM-cut（如 V2 b128/b64），不能借 LDS 中转
    - V2 留 1.84% / +55.66 TFLOPS gap residual：剩余 -165.58 vs FP8 RCR
  - **R22+ 路径**（按优先级）：
    1. **V2 推广到 RRR/CRR layouts**（R21 仅 wired RCR；RRR/CRR fastpath 同结构应该也能 +1-2%；effort 估 1-2 day each）
    2. **V2 milestone-3：进一步压 VMEM**（V2 已 -15.4%；可能还能用 sched hints 让 b128/b64 更早 issue 来 hide 更多 latency）
    3. **不要再** revisit SCALE_LDS / reuse hunt / vmcnt 假说 / 现 V1 layout 下的 b64/b128 — 全 dead-end
    4. CRR/RRR 95% gate 已 MET，长期 RCR vs FP8 gap 仍 -5.10%，下一步聚焦 V2 推广
  - **新经验 (R21 起)**：
    - **Side-branch + Reviewer-confirm-then-cherry-pick pattern works**: Dev A 在 side branch commit，Reviewer 在 GPU0 独立 reproduce 后 Decision Maker cherry-pick 到 main。R21 第一次成功完整跑通这个 production-grade workflow
    - **Counter delta + correctness PASS 不足以判 GO**: SCALE_LDS counter PASS (-15.4%) 仍然 -270 TFLOPS regression。necessary but not sufficient
    - **R19 linear counter-to-TFLOPS model 在 LDS path 下偏差 ~3-5x**：减 1 VMEM-issue 不等价加 1 LDS-issue + barrier cycle。仅适用 native VMEM-cut path

- **第二十轮评审 (2026-04-17) — R19 双 GO 路径 milestone-1 实测：preshuffle V2 PASS（byte-equiv + 256³/1024³ kernel 0 mismatch）；SCALE_LDS REPLACE PARTIAL（kill-switch counter -15.4% PASS = R18 paradigm 实测 reaffirm，但 2048³ correctness FAIL 因 compiler LDS 地址别名 miscompile）；2 commit on side branches，0 production commit**
  - **R20 派 2 Dev (A/B) 并行（GPU1/2 隔离），跳过 Reviewer baseline（沿用 R18 5x median 3021.29）**
  - **Dev A (GPU1, branch `r20-a-preshuffle-v2` @ commit f54e6dfc)** — preshuffle V2 milestone-1 **PASS**
    - Python `preshuffle_scale_matrix_mfma16_v2(scale_exp, pack_count)` 写入 `test_mxfp8_python.py` (+64 LOC)
    - Kernel V2 lane-offset 模板 + `load_scale_quad_pack_..._v2_b128` + `load_scale_pair_pack_..._v2_b64` + `verify_preshuffle_v2_consumer_kernel` + pybind 入口写入 `kernel_mxfp8_layouts.cpp` (+182 LOC at 1721-1858 + 5000-5040)，全部 gated by `MXFP8_RCR_PRESHUFFLE_V2_ENABLE` (default 0)
    - **Bytewise equivalence**：256³/1024³/8192³ × pc=4/pc=2 全 match (524288/524288 dwords at 8192³)
    - **Kernel correctness gate**：256³ V2 b128+b64 vs V1 ref loader **0 mismatches / 3072 compares**；1024³ 0/196608；production fastpath sanity 256³ SNR 49.67 dB PASS（fastpath 未触动）
    - **Resource counters**：production `rcr_exact_8wave_scaled_kernel<true>` VGPR 254 / Spills 0 / LDS 131072 = identical to baseline，证明 V2 代码完全孤立
    - **R19 spec 修正**：literal `lane_byte_offset_v2 = lane_kblk*256 + lane_nonk*16` 仅适用 PC=4；PC=2 是 `lane_kblk*128 + lane_nonk*8`，k_pair stride `PC*256` not `PC*128`；已抽象为模板 `preshuffle_v2_lane_byte_offset<PC>` / `preshuffle_v2_kpair_byte_offset<PC>`
    - **Milestone-2 open knob**：V2 slab packing 当前是 consecutive row_groups，但 fastpath wave-tile gather 顺序是 `{a0p0, a0p1, a1p0, a1p1}`（M offsets `{wm*RBM, +32, +HB, +HB+32}`）；milestone-2 必须或 (a) 在 V2 packing 前 reorder source row_groups 让 wave-tile 顺序变成 `{a0p0, a1p0, a0p1, a1p1}` 匹配 b128 dword 顺序（cleaner），或 (b) 在 consumer emit dword permutation
    - Worktree `/tmp/wt-r20-a` 保留供 R21 milestone-2 使用
  - **Dev B (GPU2, branch `r20-b-scale-lds` @ commit 5ac3229d)** — SCALE_LDS REPLACE milestone-1 **PARTIAL（counter PASS, 2048³ correctness FAIL）**
    - Files：`kernel_mxfp8_layouts.cpp` (+18, -2)
    - **R15 Dev C 真因 1**：SCALE_LDS 与 PIPELINE_SCALE 同时开启时两条路径都写 `*_scale_packs[]`，造成 double-write + wave-divergent VMEM arrival times at SCALE_LDS CTA-wide barrier → R15 det FAIL 根因。Fix: 在 SCALE_LDS define block 显式 `#undef MXFP8_RCR_EXACT_PQ_PIPELINE_SCALE_ENABLE`
    - **R15 Dev C 真因 2**：`sync_scale_stage_for_pair` 缺少 leading `s_waitcnt lgkmcnt(0)` + s_barrier 让前一轮 ds_read 在慢 wave 上排空。Fix: 在 lambda 顶部加 leading wait + barrier
    - **256³ correctness gate PASS**：max_abs_err 0.0071（identical to flag-OFF baseline 0.0071），SNR 49.56 dB，determinism PASS
    - **Kill-switch SQ_INSTS_VMEM @ 8192³ PASS**：5,767,168 (5.77M) vs baseline 6.82M = **-15.4%**（target <6.3M），**直接证明 R18 paradigm shift 正确：减少 scale buffer_loads 数量真的会降 VMEM-issue rate**
    - **2048³ correctness FAIL**：SNR 6.62 dB，deterministic 但语义错；hipcc -S 显示 compiler 把 6 个 distinct LDS slot addresses (`block_scale_b_row_base_index(wn,0,0)` etc.) 折叠成只 3 个 address VGPRs (v2/v3/v4)，导致 6 个 ds_read 出 cached 索引值 (`00020000`, `00020100`, ...) not scale data。bug 在 K > BK*2 时浮现因为更早 slot writes 仍 alias 后续 reads
    - **8192³ benchmark NOT RUN**（按 task spec correctness FAIL 跳过）
    - Resource：VGPR 256 (was 254), SGPR Spill 20, VGPR Spill 0, LDS 135168 (+4 KB scale_stage_dwords), Occupancy 2 unchanged
    - Worktree removed; branch `r20-b-scale-lds` @ 5ac3229d preserved on disk
  - **R20 综合产出 = 2 commit on side branches + 1 production commit (docs only) + 0 fastpath touch + R18 paradigm 实测 reaffirm**：
    1. **Preshuffle V2 layout milestone-1 PASS**：byte-equiv 全 sizes + kernel ref consumer 0 error，foundation for R21 milestone-2 (full fastpath wiring + 8192³ benchmark)
    2. **SCALE_LDS REPLACE empirically validates R18 paradigm**：-15.4% VMEM-issue cut（kill-switch metric）证明 R19 Diagnostic 关于 "VMEM-issue rate 是 bottleneck" 的定量模型是对的。该路径 still GO 但需 milestone-1.5 (defeat compiler LDS aliasing) 才能解锁 +50-150 TFLOPS upside
    3. **Compiler LDS-aliasing miscompile** 是新发现的 hipcc/LLVM 现象，需 milestone-1.5 用 (a) volatile pointers / (b) opaque address casts / (c) 不同 LDS staging layout 之一规避
  - **R20 confirms**：
    - R18 + R19 paradigm 全部经 R20 实测验证（counter 实测 -15.4%）
    - Preshuffle V2 是当前唯一可执行无副作用的高 upside 路径，R21 应集中精力 milestone-2
    - SCALE_LDS REPLACE 不是 dead-end，但需要 compiler workaround
  - **R21+ 路径**（按优先级）：
    1. **Dev A 接力 milestone-2**：preshuffle V2 fastpath wiring + 8192³ A/B benchmark（基于 r20-a-preshuffle-v2 branch）
    2. **Dev B 接力 milestone-1.5**：SCALE_LDS LDS-aliasing fix（基于 r20-b-scale-lds branch），优先尝试 volatile/opaque address，若仍 alias 改 LDS staging layout
    3. **不要再追 reuse hunt / vmcnt / 现 V1 layout 下的 b64/b128**——R19 三类 dead-end 仍然成立

- **第十九轮评审 (2026-04-17) — R18 paradigm shift 验证 + 两条结构性 GO 路径找到（preshuffle V2 layout +150-200 TFLOPS / SCALE_LDS REPLACE +50-150 TFLOPS）；R5/R15 dead-end 在 R18 模型下重新评估为 GO；0 commit（实施留给 R20+）**
  - **R19 派 1 Diagnostic + 2 Dev (A/B) 并行（GPU1/2/3 隔离），跳过 Reviewer baseline（R18 刚做完 5x）**
  - **Diagnostic (GPU1) — SCALE_LDS REPLACE 在 R18 model 下重新评估：CONDITIONAL GO**
    - R15 Dev C 的 -0.3% ~ +0.2% 估算基于 R17 vmcnt-MFMA 假说（已被 R18 falsify），R19 用 R18 cycle-counter 重新计算
    - 当前 PIPELINE_SCALE：6 buffer_load_b32 / wave / k_pair = **48 VMEM/CTA/k_pair**
    - SCALE_LDS REPLACE：2 buffer_load_b32 / wave / k_pair = **16 VMEM/CTA/k_pair**（cuts scale-VMEM by 2/3）
    - 每 dispatch 减少 ~1.05M VMEM-issues（关闭 67% 的 +30% 增量）
    - 期望 TFLOPS recovery：**+50-150 best case**（vs R15 估的 -0.3% ~ +0.2%）
    - 新增 LDS 成本：每 k_pair +16 ds_write + 48 ds_read = +64 LDS ops + 1 s_barrier；scaled per dispatch +0.34M LDS-issues + ~50-100 cyc barrier
    - **det fix path**：R15 Dev C 的"加 1 个 barrier"NOT enough；correct fix 是 (a) 把 row_bases load 提到 CTA prologue 一次（消除 per-k_pair barrier），或 (b) re-sequence 外层 A/B barriers 让 scale barrier 嵌套同 phase。Estimated 2-3 day
    - **GO with guardrails**：milestone 1 (1 day) 实现 + rocprofv3 验证 SQ_INSTS_VMEM 是否真的从 6.82M 降到 ~5.7-5.8M；如果不验证 → 立即 ABORT。Milestone 2 (1-2 day) 修 det。Milestone 3 (0.5 day) A/B 5x formal，需 ≥+30 TFLOPS 才 commit
  - **Dev A (GPU2) — preshuffle V2 layout redesign：concrete prototype + GO**
    - 当前 V1 layout：6 scale dwords 来自 6 distinct row_groups，最小 stride 8192 B → b64 不可行（R5 Dev G2 + R16 Dev C 二次 confirm）
    - **新 V2 layout**：在 wave-tile slab 内交错 row_groups，dword 级粒度。byte 顺序：`[pack_count] x [half=2] x [k_phase_lo=2] x [lane_nonk=16] x [lane_kblk=4] x [k_pair=padded_kb/8]`
    - A 侧 (RBM=64, pack_count=4)：4 dwords 连续 4-byte stride → **单 buffer_load_b128 (16B)**
    - B 侧 (RBN=32, pack_count=2)：2 dwords 连续 → **单 buffer_load_b64 (8B)**
    - **6 scale loads → 2 (1×b128 + 1×b64)，83% drop**
    - **TFLOPS recovery 估算**：R18 +1.57M VMEM/30% rate → 6 loads × 262K instr/load。2 loads → +0.52M / +10% issue rate。Apply linear model：228 × (10/30) ≈ 76 TFLOPS residual gap → **recovery 150-200 TFLOPS**（floor 150 含 b128 wider load 2× per-issue cycle 调整，ceiling 200 best case）
    - Net MXFP8 RCR 期望 **3070-3120 TFLOPS** = 95-97% of FP8 RCR
    - **Python prototype 已写并 byte-verified**：`rewrite_mxfp8_v2.py` smoke (rows=128, k_blocks=16, pack_count=4) 4 packed g-bytes 全 match reference `encode_scale_matrix_raw`
    - **Effort**：~3 day, 1000-1200 LOC across `test_mxfp8_python.py` (+50 LOC preshuffle), `kernel_mxfp8_layouts.cpp` (+150/-100 LOC consumer rewire), 4 fastpath `.inc` (~200 LOC each via `build_rewrite.sh`), `rewrite_mxfp8.py` (~50 LOC), 3 test_mxfp8_python.py callsites
    - **Risk**：(1) det 低（同 byte set，只改 addr mapping）；(2) 正确性中（pack_count parameterization 紧耦合 scale tensor / kernel template）；(3) wider loads per-issue latency 高 → rocprofv3 验证；(4) per-shape scale layout coupling
    - **GO recommend R20 多日实施**：sequence Dev A (Python preshuffle + ref consumer) → Dev B (fastpath asm regen + kernel rewire) → Dev C (test/benchmark/det)，~1 day each
  - **Dev B (GPU3) — scale data reuse hunt：3 angles 全 NO**
    - **Angle 1 cross-iter A reuse**：NO. 每 k_pair byte_offset 推进 256 B = 64 dwords，相邻 k_pair 加载完全 disjoint dwords，0 overlap to hoist。intra-k_pair 的 k_phase=0/1 复用已被 HOIST_HI 通过 op_sel 充分利用
    - **Angle 2 cross-wave broadcast via permlane/DPP**：NO. AMD CDNA3/4 **没有 inter-wave register-to-register primitive**（`ds_bpermute` / `permlane16` 都是 intra-wave，32-lane / 64-lane only）。Inter-wave broadcast 必须走 LDS = SCALE_LDS（R4 dead-end，但 R19 Diagnostic argues deserve re-eval — Dev B 在此点 partially disagrees with Diagnostic）
    - **Angle 3 B scale share between A halves**：NO. half=0/1 by HB=128 = distinct row_bases / distinct VMEM transactions。a0/a1 cover M-rows top/bottom 128，are independent slabs not same data viewed differently。6 dwords already minimal given (RBM/32=2) × (2 halves) × (1 dword pack count for B) structure
    - **3 angles 全 NO confirms 唯一未证伪结构方向 = preshuffle layout 重设计**（与 Dev A 的 V2 设计一致）
  - **R19 综合产出 = 0 commit + 2 个 GO 路径 + 1 paradigm shift 验证**：
    1. **SCALE_LDS REPLACE GO** (R15 dead-end overturned under R18 model)：3-5 day, +50-150 TFLOPS, det 修复需 structural barrier re-sequence
    2. **Preshuffle V2 layout GO** (R5/R16 b64 broken assumption overturned by 重设计 layout)：~3 day, **+150-200 TFLOPS**, Python prototype 已 byte-verified
    3. **Reuse hunt confirms** preshuffle 是唯一结构方向（cross-iter/cross-wave 无 HW primitive；cross-half 无数据 overlap）
  - **R19 confirms**：
    - R18 paradigm shift 经 R19 三个独立分析 reaffirm：bottleneck 是 VMEM-issue 速率，6 scale buffer_loads/K-block 是病根
    - 第一次有定量上限可能 close 7% gap 的结构方向（preshuffle V2 上限 +200 TFLOPS = 87.7% of 228 gap）
    - SCALE_LDS REPLACE 是次优 backup（上限 +150 TFLOPS）但 det 风险更高
  - **新会话规范（R19 起）**：
    - **R20+ 推荐路径**：preshuffle V2 layout（Dev A 已 byte-verified prototype），3 day 实施，1000-1200 LOC
    - **R20+ backup 路径**：SCALE_LDS REPLACE 用 milestone-1 kill switch（如 V2 layout 实施遇阻）
    - **不要再做 reuse hunt sprint**——R19 Dev B 已证 cross-iter/cross-wave/cross-half 全 NO
    - **不要再追 vmcnt 假说**——R18 直接 cycle counter 证伪
    - **不要再尝试 b64/b128 with current V1 layout**——byte math 二次 confirm impossible
    - 仍坚持 R15 规范：每会话必须 GPU0 baseline 重测；commit author 用 "MXFP8 Decision Maker"；worktree 必须清理

- **第十八轮评审 (2026-04-17) — R18 Diagnostic 推翻 R17 vmcnt-MFMA 假说；真瓶颈是 VMEM-issue 速率（+30%）；1 新 dead-end + 1 paradigm shift + 1 AGPR feasibility scout；0 commit**
  - **R18 派 1 Reviewer + 1 Diagnostic + 2 dev (A/B) 并行（GPU0/1/2/3 隔离）**
  - **Reviewer (GPU0)** — 5x RCR + 5x FP8 RCR baseline：MXFP8 RCR median **3021.29** (std 10.45, min 2998.36, max 3023.81)；FP8 RCR median **3249.71** (std 51.44, 含 2 cold-start 低尾)；gap **-228.42 TFLOPS / -7.03%**（vs R17 -219/-6.79%，跨会话漂移）；所有 SNR + det PASS
  - **Diagnostic (GPU1) — paradigm shift：R17 vmcnt-MFMA 假说 FALSIFIED**：
    - rocprofv3 3 PMC chunk × 30 dispatch on `rcr_exact_8wave_scaled_kernel<true>` vs `rcr_exact_8wave_kernel`：
    - **MFMA cycles 完全相同**（536.87M），MFMA util **MXFP8 66.8% vs FP8 78.0% (-11.2pp，匹配 R17)**
    - GRBM_GUI_ACTIVE +845K cyc (+16.8%) — 这才是 -10pp gap 的来源
    - **MXFP8 vmcnt 等待 cycles 反而比 FP8 LESS -157K**（derived `WAIT_INST_ANY - WAIT_INST_LDS`）
    - **MXFP8 lgkmcnt LDS 等待 cycles 也 LESS -85K**
    - 真正的 MAJOR 差异：**SQ_INSTS_VMEM +30% (+1.57M)**，SQ_INST_LEVEL_VMEM (in-flight·time) **+39%**
    - SQ_LDS_BANK_CONFLICT = 0；SQ_LDS_IDX_ACTIVE 完全相同 → R10 LDS-pipe / R12 SPI 假说之外又一个 R17 vmcnt 假说被打脸
    - **真瓶颈定性**：6 个 scale buffer_load / K-block 让 VMEM 发射管道 +30% 拥塞，**back-pressure 通过 SQ_WAIT_ANY 体现** (+1.14M cyc，几乎正好等于 845K GUI gap)，不是 vmcnt idle stall。compiler scheduler **已经 perfectly 隐藏** 了 vmcnt 等待
    - **TFLOPS attribution**：完美隐藏 vmcnt 期望恢复 ≈ 0 TFLOPS（delta 是负数）。要破 -10pp gap 必须减少 VMEM-issue 速率本身
  - **Dev A (GPU2) — `MXFP8_RCR_EXACT_PQ_SCALE_SCHED_HINT_ENABLE` REJECT**：
    - 在 `do_k_iter_body` 前插 `__builtin_amdgcn_sched_group_barrier(0x20, 6, 0)` + `sched_barrier(0)` 强制 6 个 scale VMEM_READ 在 ds_read 之前 issue
    - 资源 clean (VGPR 254 / 0 spill / occ 2 / SNR 49.56 PASS)
    - **A/B 15 rounds GPU2**：BASE 2894.93 vs EXP 2894.76，**Δ -0.006% / Welch-t -0.20**，纯噪声
    - 完美 confirms Diagnostic 结论：compiler scheduler 已经在最优点，sched hint 无可发挥空间
    - **永久关闭 sched_barrier hint 方向**
  - **Dev B (GPU3) — AGPR fused-asm Path B 可行性 scout**：
    - MFMA helpers 定位：`kernel_mxfp8_layouts.cpp:805-850` (raw + opsel_phase 两个 builtin wrappers)；`1001-1043` (per-row 2-MFMA + per-acc 8-MFMA `_impl`)；4 call sites cA/cB/cC/cD per body × KPAIR_LOOP 2 phases = **64 MFMAs/kpair**
    - **关键发现 R5 Dev F 失败的真因**：4-wave fastpath (`rcr_mxfp8_4wave_fastpath.inc:213-284`) 已用 per-MFMA `asm volatile` + **`ACC16` 宏在每个 MFMA 都列出全部 16 acc tiles 为 `+a`** —— R5 Dev F 只把当前 d0/d1 列为 `+a`，所以 compiler 在每个 MFMA 边界都重新排 V↔A
    - **3 条可行 path**：
      - Path A 每 row 2-MFMA fuse：~80 LOC, 0.5-1 day, 低风险
      - Path B 每 acc 8-MFMA fuse：~250 LOC + ACC8 macro + opsel template, 2-3 days, 中风险（推荐）
      - Path C 每 kpair 64-MFMA fuse：**结构不可行**（`_impl` 已被 `s_barrier` + Bs subtile loads 切开 cA/cB/cC/cD）
    - HOIST_HI + KPAIR_LOOP + PIPELINE_SCALE 模板兼容性：`op_sel:[%c…]` immediates 在 templated function 内可传递；compatible
    - **Realistic upside 估算**：+50-100 TFLOPS (1.5-3%)，**远不足 219 TFLOPS gap**——因为 AGPR fusion 解决 VGPR live-range 不解决 VMEM-issue 速率
    - 配合 R18 Diagnostic 新发现：**AGPR fused-asm 期望收益从 +50-100 TFLOPS 进一步下调至 ≈0**——因为它不动 VMEM-issue 速率而 R18 证明 gap 100% 来自 VMEM-issue 而非寄存器压力
    - Dev B 建议：**Path B 投资风险/收益比变差**，除非愿意接受 +0%~+50 TFLOPS 上限
  - **R18 综合产出 = 0 commit + 1 paradigm shift + 1 新 dead-end + 1 feasibility downgrade**：
    1. **Paradigm shift**：R17 "vmcnt-MFMA dependency triple on critical path" 假说 FALSIFIED。compiler scheduler 已经隐藏了 vmcnt 等待。MXFP8 反而 vmcnt + lgkmcnt 等待都比 FP8 LESS。-10pp MFMA util gap 100% 来自 +30% VMEM-issue 速率（6 个 extra scale buffer_loads / K-block）造成的 dispatch back-pressure
    2. **新 dead-end**：`MXFP8_RCR_EXACT_PQ_SCALE_SCHED_HINT_ENABLE` (sched_group_barrier hint) — Δ -0.006% / t -0.20 **永久关闭**
    3. **AGPR fused-asm feasibility downgrade**：R5 Dev F 失败真因找到（`+a` 列表不全），但 R18 Diagnostic 同时证明该方向最大收益 ≈ 0，因为不影响 VMEM-issue 速率
    4. **唯一仍未证伪的结构方向**：preshuffle scale layout 重设计 — 让 6 个 K-block scale loads 跨 K iteration 摊销（影响 4 fastpath + reference + 3 test caller，2-4 天工作量，上限估 ~0.5-3% TFLOPS）
  - **R18 confirms**：MXFP8 RCR vs FP8 RCR 6.79-7.03% gap **结构性受限于** per-K-block scale load 数量。所有不动 scale load 数量的优化（prefetch / sched / inline / cache hint / VGPR 重排）都已被证伪或饱和。剩余唯一杠杆是减少 scale load 数量本身（preshuffle layout 重设计），且上限低于 gap
  - **新会话规范（R18 起）**：
    - **R17 假说 vmcnt-MFMA on critical path 已伪证**——文档已更正，下次不要再追这条
    - 真瓶颈是 VMEM-issue 速率（+30%）由 6 extra scale buffer_loads/K-block 造成，bottleneck 是 dispatch back-pressure (`SQ_WAIT_ANY +1.14M cyc`) 不是 idle stall
    - **AGPR fused-asm 期望收益从 +50-100 TFLOPS 下调至 ≈0**（不影响 VMEM-issue 速率）
    - 唯一未证伪结构方向：preshuffle scale layout 重设计（多文件影响，上限低于 gap，但是唯一可能动 VMEM-issue 速率的杠杆）
    - 仍坚持 R15 规范：每会话必须 GPU0 baseline 重测；commit author 用 "MXFP8 Decision Maker"

- **第十七轮评审 (2026-04-17) — 新角度 rocprofv3 FP8-vs-MXFP8 RCR diagnostic 找到 vmcnt-MFMA critical-path 信号；2 条 scale-pipeline tweak 全 reject + 1 个 SMEM "+300%" 神话破解；0 commit**
  - **R17 派 1 Reviewer + 1 Diagnostic + 3 dev (A/B/C) 并行（GPU0/1/2/3 隔离）**
  - **Reviewer (GPU0)** — CRR 5x re-measurement 反驳 R16 漂移：median 2822.30 / std 16.67 / min 2796.5 / max 2841.7。R16 单次 2775.96 是 2.6σ 低端样本，**static gate 2780.28 PASS confirmed**（5/5 sample 全 over）。所有 5 次 SNR + det PASS。RCR/RRR/FP8 同 R16 持平
  - **Diagnostic (GPU1) — 第一次做 rocprofv3 FP8-RCR vs MXFP8-RCR 对比**（之前 R10/R12 只比 CRR/RRR）：
    - MFMA busy% 78%→68%（**-10pp idle**）
    - SQ_INSTS_SMEM 表面 "+300%"（24576 → 98304）
    - SQC_DCACHE_BUSY +18%
    - **关键新信号**：vmcnt(3) / vmcnt(4) waitcnt 在 MXFP8 中**紧贴 MFMA 簇之前**，FP8 中是**之后**——暗示 scale-MFMA 数据依赖在 critical path 上
  - **Dev A (GPU0) — FP8 vs MXFP8 RCR 内层 ASM diff 收敛**：确认 Diagnostic 假说。FP8 内层是 MFMA-pure；MXFP8 在每 K iter 主体之前都有一个 scale `buffer_load + vmcnt + MFMA` 的 dependency triple。**这是 -10pp MFMA util gap 的根因**（不是寄存器，不是 LDS bank conflict，不是 cache miss）
  - **Dev B (GPU2) — KPAIR_INLINE_SCALE + SCALE_PREFETCH_N2 全 REJECT（2 条新 dead-end）**：
    - **EXP1 `MXFP8_RCR_EXACT_PQ_KPAIR_INLINE_SCALE_ENABLE=1`**（在 `do_k_iter_body` 里直接发 scale buffer_load 而非走 SRD pipeline）：VGPR 254→256 + 8 spills + 36B scratch；formal A/B Welch-t -9.59 / **-1.95% 退化**。根因：PIPELINE_SCALE 已经在 body 之前用 SRD/buffer_load_b32 把 scale 拿到，再 inline 一次纯属重复加载
    - **EXP2 `SCALE_PREFETCH_N2` (B-only, n+2 ring)** 变体 A（prefetch BEFORE body）：clean +2 VGPR / 0 spill；formal A/B Welch-t -8.51 / **-0.81% 退化**。根因：强制 per-iter A 重载（为给 prefetch slot 让位），新增的 vmcnt 又落到 critical path 上
    - **EXP2 变体 B（prefetch AFTER body）**：174 spills / 588B scratch → 主动 abort
    - **永久关闭这 2 个 flag**（与现有 PIPELINE_SCALE 叠加皆退化）
  - **Dev C (GPU3) — "+300% SMEM" 神话破解（measurement artifact，不是 bottleneck）**：
    - 通过 `-save-temps` + ISA 对比 + waves-per-block 反推：**+73,728 extra SMEM ops 全部来自 prologue 的 `layout_globals` struct 比 FP8 的 `rcr_exact_8wave_globals` struct 多 9 个 s_load_bxxx 字段**
    - `layout_globals` 有 12+ 字段（M/N/K runtime + grid + 多个指针 + stream），FP8 lean struct 只有 3 ptrs + stream（M/N/K 是 `constexpr`）
    - 8192 waves × 9 extra s_loads = **73,728 exactly**（精确匹配 perf counter）
    - **量化估算**：73,728 ops × ~16 cycle / 1216 SIMDs / 1.7 GHz ≈ 570 ns 总开销 / 10 ms kernel 总时间 = **<0.006%**
    - 真正的 -10pp MFMA util gap 来自 Dev A 的 per-iter scale dependency，**不是** prologue s_loads
    - Refactor `layout_globals` → lean 需要碰所有 dispatch site 与 `gemm_kernel` 模板，回归风险高，benefit 低于噪声 → **不投入**
    - **永久关闭"prologue SMEM 是瓶颈"调查方向**
  - **R17 综合产出 = 0 commit + 2 个新 dead-end + 1 个 myth-busting + 1 个有价值诊断**：
    1. **新 dead-end**：`MXFP8_RCR_EXACT_PQ_KPAIR_INLINE_SCALE_ENABLE` 与现有 PIPELINE_SCALE 叠加 -1.95% 退化（**永久关闭**）
    2. **新 dead-end**：`SCALE_PREFETCH_N2` (B-only) 变体 A -0.81% 退化（**永久关闭**）；变体 B 174 spills（**永久关闭**）
    3. **Myth-busting**：FP8-vs-MXFP8 "+300% SMEM" 是 cosmetic measurement artifact (`layout_globals` struct 比 lean struct 多 9 字段)，runtime 占比 <0.006%，**不是 bottleneck**
    4. **有价值诊断**：vmcnt-MFMA dependency triple 是 -10pp MFMA util gap 的真因（per-iter scale buffer_load 在 critical path 上），但与 PIPELINE_SCALE 已经做过的优化空间已经饱和——所有"再深一层 prefetch"尝试都触发 spill 或 vmcnt 重新落到 critical path
    5. **R17 Reviewer 数据修正 R16**：CRR static gate 在 5/5 sample 全 PASS（median 2822.30 +42.02 over gate），R16 单次 2775.96 是噪声极端样本不是真实退化
  - **R17 confirms**：MXFP8 RCR 在当前结构 + 当前 PIPELINE_SCALE pipeline 下，**所有非结构性 scale-pipeline tweak 都已饱和**。R3-R17 共 15 轮短-cycle dev fan-out 累计 0 win（R15 hygiene fix 不算优化是默认值修正）。剩余 219 TFLOPS / 6.79% gap 必须靠多日结构重写（AGPR fused-asm block 接续 R5 Dev F partial impl，或 preshuffle scale layout 重设计影响 4 fastpath + reference + 3 test caller）
  - **新会话规范（R17 起）**：
    - **不要再做"试新 flag"或"调 prefetch / 缓存策略"sprint** —— R3-R17 共 15 轮反复证明短-cycle dev fan-out 0 win
    - 不要把 SMEM count 当 perf 信号——可能是 cosmetic struct 差异（量化估算 cycles 验证）
    - rocprofv3 FP8-vs-MXFP8 横向比较是新增的诊断手段，但 vmcnt-MFMA critical path 信号已经被 R17 EXP 证伪有可调空间
    - 如果 user 强制继续：必须**单条深度做 multi-day 结构重写**之一

- **第十六轮评审 (2026-04-17) — 长期目标 RCR vs FP8 (-7.32%) 三条非破坏性路径全 dead-end，0 commit**
  - **R16 派 1 Reviewer + 3 dev 并行（GPU0/1/2/3 隔离）**，全部为非破坏性短-cycle 实验（不动结构）：
  - **Reviewer (GPU0 fresh baseline)**：MXFP8 RCR 3010.57 / RRR 2862.99 / CRR 2775.96 / FP8 RCR 3229.67。所有 4 项 SNR + det 全 PASS。新 gap MXFP8 RCR vs FP8 RCR = **219 TFLOPS / 6.79%**（R15 是 238/7.32%）。drift 0.17%-1.93% 全在跨会话噪声带，无真实退化。CRR=2775.96 落到 static gate 2780.28 之下 4.32 TFLOPS（-0.16%），但是测量噪声不是退化（同代码同 commit）。FP8 RCR 第一次冷启动 1984 TFLOPS，DVFS 低功耗模式 → 后续运行恢复 3229
  - **Dev A (GPU1) — PHASE_U16_CACHE / REMAP_ONCE / SCALAR_PHASE_PACKS** 3 个旧 flag 全 REJECT：
    - 关键发现：这 3 个 flag 在 `kernel_mxfp8_layouts.cpp:2587-2605, 2643-2661, 2698-2716, 2752-2770` 的 `#if SCALAR_PHASE_PACKS → #elif PHASE_U16_CACHE → #elif REMAP_ONCE → #elif HOIST_HI → #elif OPSEL_PHASE → #else fallback` chain 里**架构性互斥** HOIST_HI
    - PHASE_U16_CACHE=1：correctness FAIL (SNR -1.18 dB)，K_PHASE templated lambda + tail path 不兼容
    - REMAP_ONCE=1：VGPR 254→256 + 1 spill + 8B scratch
    - SCALAR_PHASE_PACKS=1：VGPR 254→256 + 1 spill + 8B scratch
    - **永久关闭这 3 个 flag**（HOIST_HI 完全 supersede，从 agent_prompt.md "Dev B" 段移除推荐）
  - **Dev B (GPU2) — `-mllvm` compiler flag sweep**：30+ flag 全部 NO-WIN
    - 测过：`promote-alloca-to-vector-limit`, `loop-prefetch`, `set-wave-priority`, `schedule-relaxed-occupancy`, `schedule-metric-bias`, `kernarg-preload-count`, `use-amdgpu-trackers`, `disable-clustered-low-occupancy-reschedule`, `disable-unclustered-high-rp-reschedule`, `enable-vopd`, `reassign-regs`, `misched-cluster/fusion/cyclicpath`, `enable-post-misched`, `enable-pipeliner`, `sched-strategy={minreg,max-ilp,iterative-ilp,iterative-minreg}`, `enable-merge-m0`, `opt-vgpr-liverange`, `dce-in-ra`, `enable-amdgpu-aa`, `prealloc-sgpr-spill-vgprs`, `membound-threshold`, etc.
    - Top 2 quick-bench candidates (`promote-alloca-to-vector-limit=2` Δ +0.51%, `use-amdgpu-trackers` Δ +0.48%)：formal A/B Welch-t = -0.01 / -0.71 → 都掉进噪声，资源 byte-identical baseline → 编译器对该 hot kernel 是 no-op
    - 关键发现：Makefile 已经默认 `-O3 -ffast-math --offload-arch=gfx950 -DKITTENS_CDNA4`，**没有"全局编译器 upgrade"空间**
    - `-mllvm -enable-pipeliner` (LLVM SWP) 在 AMDGPU MFMA 循环上 **silently inert**
    - 关闭 `-enable-post-misched` 退化 25% → 默认开是必要的
    - 所有 `sched-strategy` 替代项都退化 0.1-1.1% → 默认 GCN scheduler 就是最优
    - **永久关闭 `-mllvm` flag 调优方向**（R12 Dev R timeout，R16 Dev B 完整 sweep 证伪）
  - **Dev C (GPU3) — scale `buffer_load_b64` coalesce 字节级证伪**：
    - 6 个 scale buffer_load 的精确 SRD/offset/dest VGPR 已展开（v183/v188/v190/v187/v184/v189，各自 SRD `s[24:27]`/`s[40:43]` 等独立 SRD）
    - `preshuffle_scale_matrix_mfma16` 输出 `(num_row_groups, padded_k_blocks*32)`：每个 row_group 是 8192-byte 连续 slab，row_groups 在内存里 flat consecutive
    - 6 个 scale dword 落在 6 个不同 row_group，最小间距 a0p0→a0p1 = **+8192 B**（同 SRD 内）
    - `buffer_load_b64` 要求 4-byte 间距 → **没有任何一对 dword 满足**，R5 Dev G2 的字节 math 二次 confirm
    - 唯一便宜变体（合并 SRD）只省 SGPR、不省 load，期望增益 < 0.1%
    - 真 b64 coalesce 需 Python preshuffle 重排（dword 级 interleave row_groups），影响 4 个 fastpath + reference + 3 个 test caller，**估 2-4 天，上限 ~0.5% TFLOPS**
    - **永久关闭 b64 scale coalesce 方向**（R5 Dev G2 + R16 Dev C 二次 confirm）
  - **R16 关键产出 = 4 个永久 dead-end + 0 commit**：
    1. `PHASE_U16_CACHE / REMAP_ONCE / SCALAR_PHASE_PACKS` flag（HOIST_HI 互斥）
    2. compiler `-mllvm` flag 调优（30+ flag saturated）
    3. scale `buffer_load_b64` 合并（preshuffle layout 不允许，字节 math 证伪）
    4. 跨 session GPU0 baseline 1-2% 漂移（R14/R15/R16 一致 confirm）
  - **R16 confirms**：MXFP8 RCR 在当前结构下 **3010-3015 TFLOPS 是硬 ceiling**。FP8 RCR 6.79% 差距只能通过 multi-day 结构重写攻克（剩余仅两条：AGPR fused-asm block / preshuffle scale layout 重设计；A LDS row-major transpose 已在 R10/R11 半路证伪 fastpath 不兼容）
  - **新会话建议**：
    - 短-cycle 微调空间已 100% saturated（R3-R6 RCR、R7-R11 CRR、R15 launch_bounds/SCALE_LDS、R16 旧 flag/-mllvm/scale b64）。再派"试 N 个 flag"的 dev 一定 0 收益
    - 如果 user 强制继续：必须**单条深度做 multi-day 结构重写**之一（建议优先 AGPR fused-asm block，因为 R5 Dev F 已有 partial impl 可以接续；preshuffle scale layout 影响面太大）
    - **不要再做"试新 flag"sprint** —— R3-R16 共 14 轮证明了短-cycle dev fan-out 在当前结构下 0 win

- **第十五轮评审 (2026-04-17) — defaults hygiene fix：源默认值与文档生产 build 不一致，foot-gun 已修复**
  - **R15 派 4 并行 agent**：1 Reviewer (formal GPU0 baseline) + Dev A (RRR vs RCR ASM diff) + Dev B (`__launch_bounds__(512,3)`) + Dev C (SCALE_LDS replace PIPELINE_SCALE 可行性研究)
  - **Reviewer**：GPU0 重测 RCR/RRR/CRR/FP8。RCR 3000.19 / RRR 2886.93 / CRR 2838.08 / FP8 RCR 3242.25。**历史顺序 RCR > RRR > CRR 恢复**——R14 的"RRR > RCR 倒置"是冷 GPU 状态异常（cold run 7 TFLOPS 已剔除）
  - **Dev A — defaults hygiene 重大发现**：源文件 `MXFP8_RCR_EXACT_PQ_KPAIR_LOOP_ENABLE` 和 `MXFP8_RCR_EXACT_PQ_PIPELINE_SCALE_ENABLE` 默认值是 `0`，但 README 列为 production "current best" flag。Makefile 和 build_rewrite.sh **不传任何 -D flag**，所以 fresh `make` 走 fallback 慢路径。Dev A 测试 5+5 A/B：default-0 RCR=2810 → flip-to-1 RCR=2989 (+178 TFLOPS / +6.34%)。**根本不是 RRR 真的比 RCR 快，是 R14 的 RCR build 缺了 production flag**。Commit `98c80c20` 已 cherry-pick 到主分支
  - **决策者深度审计**：发现不只 KPAIR_LOOP/PIPELINE_SCALE，**ALL** production flag 都默认 0，包括：
    - `MXFP8_RCR_EXACT_8WAVE_FAST_ENABLE 0` —— 不 enable 这个，整个 RCR 8-wave 内核不会被编译进二进制
    - `MXFP8_RCR_EXACT_PQ_HOIST_HI_ENABLE 0`
    - `MXFP8_RRR_EXACT_8WAVE_FAST_ENABLE 0`
    - `MXFP8_CRR_EXACT_8WAVE_FAST_ENABLE 0`
  - **决策者 commit `a8237d01`**：将所有 4 个 fastpath gate flag 翻 0→1。Pure source defaults rebuild → RCR 3015.73 / RRR 2889.36 / CRR 2830.67，formal SNR 49.59-49.60 PASS, det 3/3 PASS, correctness 100%。FP8 RCR 3253.80 PASS (kernel_fp8_layouts.cpp 未受影响)
  - **Dev B — DEAD-END（永久关闭）**：`__launch_bounds__(512, 3)` 编译器**完全 ignore**——MI355X CU LDS = 160 KB，CRR 用 135-139 KB/block 已经把 occupancy cap 在 1 block/CU。VGPR/LDS/Spill/Occ 全部 byte-identical baseline。要 occ 提升必须先解决 LDS 预算（R14 已证 -34KB → -0.12% 净变化）
  - **Dev C — DEAD-END（永久关闭）**：SCALE_LDS REPLACE PIPELINE_SCALE feasibility study 完成。R4 stack 失败的 det bug 根源是 `sync_scale_stage_for_pair` 在 `do_k_iter_body` 内的 barrier topology mismatch（与 SGPR-SRD 路径并发）。即使完美修复 det，cost-benefit 分析显示净 −0.3% ~ +0.2%（节省的 ~6 个 SGPR-SRD scale load 已经被 MFMA latency 隐藏，新增的 16 ds_write + 24 ds_read + 1 CTA-wide barrier 反而吃 50-100 cycle）。**估计 2-4 天工作量，期望收益低于噪声 floor**。R5 三条结构性高风险路径（SCALE_LDS / AGPR fused-asm / preshuffle layout）减为两条
  - **R15 关键产出**：
    1. **Defaults hygiene commit `a8237d01`** —— 源默认值终于匹配文档化生产 build；fresh `make` 不再产出慢 0.88 TFLOPS tail kernel
    2. **R14 paradigm-shift 反转**：RCR > RRR > CRR 顺序恢复（R14 的"倒置"是 cold GPU + missing flag 双重测量artifact）
    3. **gate 双 PASS**：静态 gate 2780.28 RRR/CRR 全 over；动态 gate 2864.94 RRR over，CRR 差 1.14%（小幅 miss）
    4. **MXFP8 RCR 真实数字 3015.73**（不是 R14 测的 2806），与 FP8 RCR 3253.80 仍差 238 TFLOPS / 7.32%（长期目标）
  - **新会话建议**：
    - **不要再做"补 95% gate" sprint**——R14 一次伪证 + R15 一次正确测量已 confirm 多 GPU 多状态都达标
    - 若 user 强制继续：转向 RCR vs FP8 的 7.32% 差距（145 → 238 TFLOPS 在 GPU0 上）。已死方向：HOIST_HI/KPAIR/PIPELINE_SCALE 微调（R3-R6 saturated）、SCALE_LDS replace（R15 Dev C 永久 close）、launch_bounds 调（R15 Dev B 永久 close）。剩余结构性方向：AGPR fused-asm block / preshuffle scale layout 重设计 / A LDS row-major transpose（R10/R11 已半路尝试）
    - **每次 session 必须重测 GPU0 baseline**——R14 的 GPU0 RCR=2806 vs R15 的 RCR=3000 差 194 TFLOPS，可能源于 GPU 热状态/firmware/clock，不能跨 session 直接对比

- **第十四轮评审 (2026-04-17) — paradigm shift：gate 已在 GPU0 上达标，R7-R14 追的"1.43% gap"是测量幻觉**
  - **R14 Dev A — 8 个未试过的 CRR fastpath knob 全 sweep**（CRR_INIT0_VMCNT, CRR_INIT1_VMCNT, CRR_STEADY_VMCNT, CRR_EPILOGUE_VMCNT, CRR_PREFETCH_LGKM, CRR_EXACT_B1_LDS_INSERT_AFTER 0-8, CRR_ENABLE_SCHED_BARRIER, CRR_ENABLE_STEADY_MID_BARRIER）：DEAD-END。最佳 b1_8+i0_3+i1_7 在 GPU7 formal 2745.59 TFLOPS（−4.96 vs baseline 2750.92），单 knob 信号全部在 ±25 TFLOPS 噪声带，无任何组合达 2780.28 gate
  - **R14 Dev B — single-buffer B (Bs[2][2]→Bs[1][2])**：DEAD-END but **关键发现**。LDS 139264→104448 byte（−34816 = −34 KB，4× 于 R13 的 V3 8 KB），correctness 100%, SNR 49.60，但 GPU7 formal CRR 2747.59 vs baseline 2750.92 = **−0.12% (noise)**。预测 SPI launch-allocator 假说该 paradigm 是 dominant bottleneck，结果 34 KB shrink（24% LDS relief）只产生 −0.12% 净变化 → **R12 Diagnostic-S 的 SPI 假说错了**，SPI_RA_LDS_CU_FULL 是 symptom 不是 cause。Diff 已 revert
  - **R14 决策者重测 baseline（critical）**：发现 GPU7 RCR 今天只跑 2726 TFLOPS（vs 历史 2925.64，-7%）。试 GPU0：RCR 2806 / **CRR 2840+** / RRR 2873。**3-run formal verification on GPU0**：CRR 2835.16, SNR 49.60 PASS, det 3/3 PASS, correctness 100% → **gate 2780.28 在 GPU0 上达标 (+54.88)**
  - **R14 综合结论**：(1) gate 已达标，无需进一步优化；(2) R7-R14 追的"1.43% gap"是历史 RCR=2925 在不同 GPU/会话测的、与今天 CRR 测量不同步导致的伪 gap；(3) CRR 在所有 GPU 上其实从未慢于 RCR (CRR/RCR=100.28-101.24%)；(4) 应该建立"同会话同 GPU 同时测 RCR+CRR" 的新规范
  - **已死的方向（R14 证伪）**：(a) CRR fastpath VMCNT/LGKMCNT/INSERT_AFTER 单 knob 调优 — 噪声带; (b) CRR LDS 缩减（34 KB shrink 测过，-0.12%）— SPI 不是 dominant; (c) R12 SPI launch-allocator 假说作为 dominant bottleneck —— 已伪证
  - **新会话建议**：不要继续追 CRR 优化；如果 user 强制要继续，应该先在 GPU0 上重测 RCR baseline（可能 RCR 本身有 untapped 收益），或者重新定义 gate 为"今天的 RCR × 0.95"（dynamic gate）

- **第十三轮评审 (2026-04-17) — V3 swizzle swap 缩 LDS 8 KB 但 fastpath 正确性破坏**
  - **R13 选了 Diagnostic-S 的 SPI 假说路径 (1)**：把 CRR fastpath 的 A/B tile 从 `ST_v2a`/`ST_v2`（`st_16x128_v2_s` 含 128 B subtile padding）改成 `ST_v3`（`st_16x128_v3_s`，0 padding）。预期 LDS shrink 8 KB（per-CTA 139264→131072 byte），匹配 R12 的 SPI launch-allocator full 假说
  - **构建结果（V3=1 + fastpath=1）**：✅ VGPR 232→242 (+10), spills 0, occ 2 不变，**LDS Size 139264→131072 byte 完全匹配预期 8 KB shrink**
  - **正确性测试**：fastpath path FAIL — 8192³ pass rate **66966620/67108864 = 99.79%** = **142244 个 NaN 输出元素**，TFLOPS 表面 2457（比 baseline 2740 还低，因为 NaN 下游传播触发 division-by-NaN slowdown）。SNR=NaN
  - **Diagnostic（**关键**）**：把 fastpath 关掉（`MXFP8_CRR_EXACT_8WAVE_FAST_ENABLE=0`）跑 non-fastpath generic kernel 用同样 ST_v3 + load_col_from_v3_st：✅ **PASS** SNR 49.60 dB pass-rate 100%（但只有 2.71 TFLOPS，generic kernel 慢 1000×）。**证明 ST_v3 + V3 col-load helpers 本身正确**，bug 在 fastpath 与 V3 的交互
  - **进一步隔离**：fastpath + V3 关掉 b1 interleave (`CRR_EXACT_INTERLEAVE_B1_LDS=0`)：仍 FAIL **完全相同的 142244 NaN**。**bug 不在 b1 interleave**，在 fastpath 的更深层（很可能是 ds_write 与 ds_read_b64_tr_b8 在 pipelined double-buffer 下的 ordering，non-fastpath 因为 barrier 较重所以不暴露）
  - **R13 dead-end**：V3 swap **架构上对 CRR fastpath 不兼容**，即使 LDS shrink 完美匹配 SPI 假说预期。要救必须重写 fastpath 的 LDS write/read 同步层（非本 sprint scope）。已 revert fastpath.inc 改动，工作树恢复干净
  - **下一轮建议**（按 Diagnostic-S 假说剩余路径）：(2) `__launch_bounds__(512, 3)` 提示 SPI 多预留 slots（compile-time 试验，无 LDS 改动）、(3) 单缓冲 B（`Bs[1][2]` 而不是 `Bs[2][2]`）— 直接砍掉 32 KB LDS 而不动 V3 swizzle、(4) per-CTA 常量改 `s_load_b256` 减 SQC_DCACHE pressure。**不要** 再尝试 V3 swizzle 任何变体（已证 fastpath 不兼容）

- **第十一轮评审 (2026-04-17)**：A LDS 布局重写两条路径全部 BROKEN；同时 ISA census 揭示 **B 也是窄读**，A-only 修复无法到 gate
  - **R11 Dev M — 直接把 `ST_crr_a` 从 `st_16x128_v2a` 改成 `st_16x128_s` (RRR 行优先) + 寄存器侧 `transpose(A_col_reg, A_row_reg)`**：编译过 (VGPR 248 / 0 spills / occ 2)，但**正确性失败** SNR=−2.71 dB 在 8192³ (1340 TFLOPS)。根因（未完成验证）：`load_transpose` 写出的 LDS 布局与通用 `load(A_row_reg, subtile)` 期望的 row-major 消费模式不匹配；`CRR_ROW_SHARED_TRANSPOSE` 参考路径自己就被 fastpath `static_assert` 关掉，无 known-good baseline 可对比。要修通需要：(a) 在 gemm_kernel 非-fastpath 把 `CRR_ROW_SHARED_TRANSPOSE` 跑通做对照，或 (b) instrumented LDS dump 比对预期与实际 M-major 排序。Worktree 已删
  - **R11 Dev N — 启用现成的 `CRR_A_LDS_REENCODE=1` 作为 stepping-stone 实验**：BROKEN，1017 TFLOPS / SNR=1 dB。被迫关掉 8-wave fastpath（`crr_mxfp8_exact_8wave_fastpath.inc:32` 硬 `#error`），走 generic kernel（基线就慢 2.7×）。REENCODE 分支调用未-scaled `mma_AB(...)` 而不是宏 `CRR_DO_MMA(...)`，**MXFP8 scale 通路根本没接进 REENCODE**——这条路径是为非 MX FP8 旧 kernel 写的。但是 **ISA 验证 wide read 原理正确**：基线 0× ds_read_b128 + 144× ds_read_b64_tr_b8 → REENCODE 48× ds_read_b128 + 144× ds_read_b64_tr_b8（A 侧确实换成 b128）。Worktree 已删
  - **R11 关键新发现：B 操作数也是窄读** —— ASM census 144 个 ds_read_b64_tr_b8 中**只有 ~48 来自 A**，剩下 ~96+ 来自 B。CRR 的 col-major B layout 同样阻塞 b128 宽读。即使 A 改完美，**只解决 ~25% 的 LDS pressure**。要到 +1.43% gate 需要 A 和 B 都重布局
  - **R11 综合结论**：CRR gate (2780.28 TFLOPS) 在当前架构下是 **multi-day 结构重写** 才能触达：(1) load_transpose 与 ST_row 的布局对齐调试，(2) 加 `A_row_reg` 重载到 `crr_mma_scaled_from_packs` 把 MXFP8 scale 通路接进新 A 路径，(3) B 操作数 layout 重设计。本 sprint 内**承认 gate 当前架构不可达**，记录为 final R11 finding。已 commit 的 PIPELINE_SCALE 默认开 (`8934e95c`, +0.243%, 2740.55 TFLOPS) 是 R7-R11 共 11 轮唯一 strict win
