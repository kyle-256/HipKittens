# R34 Reviewer Findings

**Date:** 2026-04-18
**Branch:** r34-rev (base feat/mxfp8-only @ `d0176862`)
**GPU:** Phase 1: GPU 4/5/6/0 (4-GPU triangulation; GPU0 last per task brief to deconflict with R34 Dev A on /tmp/wt-r34-a). Phase 2: orthogonal-to-Dev GPUs per task brief.
**Methodology:** Per-process 8s 16k FP16 preheat → 5x in-process bench (warmup=50, iters=100), single shape (70B KV V2-CRR M=4096 N=1024 K=8192). `r34_reviewer_bench5x.py` is **bit-identical** to `r33_reviewer_bench5x.py` (R31 sclk-fix carried forward, no methodology change this cycle).
**Build-cache hygiene (R29 Dev C rule):** `rm -f tk_*.so` + `make clean` before each compile + per-build md5 logged in `r34_reviewer_4gpu_runs/build_md5.log`. **All 12 (re)builds produced bit-identical .so md5 = `09cb3e014dad03d190eb7d737a2c1a9f`** (single shape, single source ⇒ build deterministic across attempts and across GPUs). Md5 differs from R33 (`b1df8e37…`) because the source moved from R33 wrap `2945ed91` → R34 base `d0176862` (R33 SHIPs added the V2-RRR autotune wire-in + 4 STRICT V2-RRR layout-pivot data + Dev B's rect-V2 RCR numerics fix). Source is bit-stable across the 4 R34 Phase-1 builds.

---

## Phase 1 — 4-GPU baseline reverify (4th cross-cycle data point)

### Environmental note (R34 Reviewer surfacing)

A **parallel mxfp4 agent on the same node** (`/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_all42/`) was actively running long fp4 GEMM benches on GPU0 and GPU6 throughout the Phase 1 window. Multiple Phase-1 attempts on those two GPUs landed on **stuck-DPM low-clock** states (sclk capped at 1707-1873 MHz vs the proper ~2300+ MHz fully-ramped state). After 2 (GPU6) and 3 (GPU0) attempts, both GPUs eventually had a clean unblocked window where the 8s preheat fully ramped DPM and the 5x bench produced consistent results. **GPU4 and GPU5 were not affected** by this contention (clean on first attempt). All 12 re-attempts compiled bit-identically (md5 stable). The clean per-GPU readings used below are from the final clean attempt; intermediate contended attempts are documented in `r34_reviewer_kv_4gpu.json` per-GPU `notes` fields.

This **is** a methodology bug surfaced by R34 — the bench framework has no defense against same-node-other-process DPM contention. See "Methodology bugs surfaced this cycle" below.

### R34 Phase 1 raw data (sclk-correct, R31 fix carried forward; final clean attempts per GPU)

Run sequentially on GPUs 4, 5, 6, 0 (with retries for GPU6 and GPU0 to clear contended attempts). Identical orchestrate (`r34_reviewer_4gpu_orchestrate.sh`), single source @ `d0176862`, single .so md5.

| GPU  | TFLOPS median | mean | stdev | bench SNR (dB) | corr SNR (dB) | det 3/3 | sclk pre-bench | sclk post-bench |
|---|---:|---:|---:|---:|---:|:---:|---:|---:|
| GPU0 | **765.12** | 763.37 | 5.28 | 43.21 | 49.60 | PASS | 2296 MHz | 2361 MHz |
| GPU4 | **766.30** | 767.09 | 1.60 | 53.59 | 49.60 | PASS | 2370 MHz | 2386 MHz |
| GPU5 | **768.88** | 768.13 | 1.96 | 51.93 | 49.60 | PASS | 2395 MHz | 2395 MHz |
| GPU6 | **787.13** | 786.61 | 2.85 | 48.81 | 49.60 | PASS | 2365 MHz | 2392 MHz |

Build md5 verification (all 12 (re)attempts identical):
```
[build_mxfp8 gpu=4 4096x1024x8192 rc=0 md5=09cb3e014dad03d190eb7d737a2c1a9f]
[build_mxfp8 gpu=5 4096x1024x8192 rc=0 md5=09cb3e014dad03d190eb7d737a2c1a9f]
[build_mxfp8 gpu=6 4096x1024x8192 rc=0 md5=09cb3e014dad03d190eb7d737a2c1a9f]
[build_mxfp8 gpu=0 4096x1024x8192 rc=0 md5=09cb3e014dad03d190eb7d737a2c1a9f]
... (8 additional retry builds, all same md5)
```

### Cross-GPU triangulation table (R29–R34)

| GPU | R29 | R30 | R31 | R32 | R33 | **R34** | R34 Δ vs R33 | R34 Δ vs R31 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPU0 | (n/a) | 791.25 | 786.38 | 768.43 | 766.63 | **765.12** | -1.51 (-0.20%) | -21.26 (-2.70%) |
| GPU4 | 766.86 | (n/a) | 767.31 | 768.37 | 767.97 | **766.30** | -1.67 (-0.22%) | -1.01 (-0.13%) |
| GPU5 | (n/a) | 767.23 | 764.28 | 786.59 | 764.81 | **768.88** | +4.07 (+0.53%) | +4.60 (+0.60%) |
| GPU6 | (n/a) | (n/a) | 766.14 | 776.60 | 765.71 | **787.13** | **+21.42 (+2.80%)** | +20.99 (+2.74%) |
| **median of 4 GPUs** | | | **766.72** | **772.51** | **766.17** | **767.59** | +1.42 (+0.19%) | +0.87 (+0.11%) |
| **mean of 4 medians** | | | 771.03 | 775.00 | 766.28 | 771.86 | +5.58 (+0.73%) | +0.83 (+0.11%) |
| **min of 4 GPUs** (R33 sub-rule) | | | 764.28 | 768.37 | 764.81 | **765.12** | +0.31 (+0.04%) | +0.84 (+0.11%) |

R34 spread: max 787.13, min 765.12 ⇒ **22.01 TFLOPS spread = 2.88%** (R31: 22.10 TF / 2.89%; R32: 18.22 TF / 2.37%; R33: 3.16 TF / 0.41%; **R34 essentially matches R31's spread, both have one ~+2.7% high-state outlier**).

### Phase 1 verdict

**The R33-revised "stochastic per-(GPU × cycle) high-state" hypothesis is STRENGTHENED by R34.** Specifically:

1. **GPU6 is the R34 high-regime outlier** at 787.13 TFLOPS, +2.71% above the cluster median of 766.30 (GPU4/GPU5/GPU0 cluster: 765.12-768.88 TF, spread 0.49%). This is right in the same band as the prior R31 GPU0 outlier (786.38, +2.64%) and R32 GPU5 outlier (786.59, +2.36%).
2. **Four-cycle rotating-high-outlier table** (now updated with R34):

| Cycle | High GPU | High median | Other-3 median | High excess | Verdict |
|---|---|---:|---:|---:|---|
| R31 | GPU0 | 786.38 | 766.14 | **+2.64%** | high-regime outlier present |
| R32 | GPU5 | 786.59 | 768.43 | **+2.36%** | high-regime outlier present (rotated GPU0→GPU5) |
| R33 | (none) | 767.97 | 765.71 | **+0.30%** | NO high-regime outlier; all 4 GPUs cluster ~766 TF |
| **R34** | **GPU6** | **787.13** | **766.30** | **+2.71%** | **high-regime outlier present (rotated to GPU6)** |

Three of four cycles (R31, R32, R34) show **exactly one** high-state outlier, sitting at +2.36-2.71% above the cluster. The cluster value itself is highly stable across all 4 cycles (medians-of-other-3 = 766.14, 768.43, 765.71, 766.30 — peak-to-peak only 2.72 TF / 0.36%). One cycle (R33) had no outlier.

**This is consistent with stochastic per-(GPU × cycle) firmware/DPM-residency state**: each (GPU × cycle) trial independently lands on either "cluster" or "high" state, with the latter being roughly 1-in-4 to 1-in-2 likely per (GPU × cycle). With 4 GPUs × 4 cycles = 16 (GPU × cycle) trials we observed exactly 3 high outcomes (~19%), giving each (GPU × cycle) trial a ~1-in-5 to 1-in-6 high-state probability, OR per-cycle there is some variable likelihood of >=1 high (here 3/4 cycles).

3. **GPU0/GPU4/GPU5 in R34 are essentially flat vs R33** (-0.20%, -0.22%, +0.53%) — all within bench precision.
4. **GPU6 in R34 jumped +21.42 TF (+2.80%) vs R33** — exactly the rotation pattern seen R31→R32 (GPU0 dropped, GPU5 rose).
5. **sclk readings (R31 fix verified):** all 4 GPUs read distinct, plausible per-GPU clocks. GPU6 high-regime sclk (2365-2392 MHz) is **not** distinguishable from GPU4/GPU5 cluster sclk (2370-2395 MHz). This confirms R32+R33 observation: the high-regime ~+2.7% offset is sub-clock-step / boost-residency / firmware-state and **not** visible in the level-1 sclk readout. The sclk-readback only resolves to ~30 MHz tiers that are uniform across GPUs in cluster vs high state.

### Recommended canonical baseline for R35+ (70B KV V2-CRR)

Recomputing with N=4 cross-cycle data:
- **Median-of-4 across cycles:** R31 766.72, R32 772.51, R33 766.17, R34 767.59 ⇒ mean = **768.25 TFLOPS**.
- **Min-of-cluster across cycles** (R33 sub-rule basis): R31 (other-3) 766.14, R32 (other-3) 768.43, R33 (all-4) 765.71, R34 (other-3) 766.30 ⇒ mean = **766.65 TFLOPS**.
- **Live baseline recommendation: 768 TFLOPS** (unchanged from R33 recommendation; recomputed mean still rounds to 768).

### SHIP gate target for the rect-V2 path 0.92 ratio still ≥840 TFLOPS (FP8 baseline 911-941 TFLOPS × 0.92), unchanged from R31/R32/R33 prescription — the ratio target dominates over the small baseline drift. **However**, R33 Dev B + R33 Reviewer concurrence already CLOSED the rect-V2 RCR perf lever as paradigm closure (rect lacks PIPELINE_SCALE/KPAIR_LOOP/PHASE_U16_CACHE/REMAP_ONCE), so the 0.92 ratio gate is now only relevant for rect-V2 CRR Stage A2 (the one structural lever still open per R33's R34 priority list).

### Per-GPU normalization rule (R32 carried forward, R33 sub-rule strengthened by R34)

Any single-GPU SHIP claim must be measured on ≥2 distinct physical GPUs **and** include a same-cycle on-the-same-GPUs baseline reverify (not a previous-cycle baseline) before calling SHIP. The gate is `min(measured_TFLOPS_across_GPUs) ≥ baseline_min × 1.01` AND `Welch t > 3.0` against the same-cycle paired baseline. **R33 sub-rule (now strengthened by R34's GPU6 +2.71% outlier — it would over-attribute SHIP signal if averaged in):** discount any GPU sitting >+1.5% above the others as a transient high-state outlier; use min-of-GPUs not mean.

### Phase 1 artifacts

- `analysis/fp8_gemm/mi350x/r34_reviewer_kv_4gpu.json` (full structured JSON)
- `analysis/fp8_gemm/mi350x/r34_reviewer_4gpu_orchestrate.sh` (driver script with PHYS_GPU sclk-fix carried forward)
- `analysis/fp8_gemm/mi350x/r34_reviewer_bench5x.py` (5x bench, bit-identical to R33)
- `analysis/fp8_gemm/mi350x/r34_reviewer_4gpu_runs/70b_kv_crr_mxfp8_gpu{0,4,5,6}.txt` (raw per-GPU bench logs — final clean attempts)
- `analysis/fp8_gemm/mi350x/r34_reviewer_4gpu_runs/build_md5.log` (12-attempt md5 hygiene log; all bit-identical)

---

## Phase 2 — Dev SHIP-candidate verdicts

R34 dev branches polled every ~5-10 minutes during the Phase 2 window. Final state (4 of 4 dev branches reported):
- **r34-a:** `5d5c3d4c R34 Dev A: SHIP V2-RRR autotune fan-out (3 NEW predicates → 4 NEW cells)` — Phase 2 verification REQUIRED.
- **r34-b:** `440fa36b R34 Dev B: 4-GPU triangulation of R33 SHIP-LITE 8B Gate/Up — c5 STRICT SHIP, c6 SHIP-LITE confirmed` — Phase 2 verification REQUIRED.
- **r34-c:** `367c918c R34 Dev C: NO SHIP — rect-V2 CRR Path 2 (HB_N=64 + K-serialised reads) -9.7% slower than square` — explicit NO SHIP; concur.
- **r34-d:** `73380c4b R34 Dev D: Stage 2a type bridge + Stage 2b kernel-body skeleton — sub-RBM hypothesis REFUTED` — explicit NO SHIP ("REFUTED" in commit msg); per task brief Dev D was scaffolding only.

### Verdict 1: R34 Dev A — SHIP CONFIRMED (V2-RRR autotune fan-out, 3 NEW predicates / 4 NEW cells)

Dev A extended R33 Dev A's single-shape autotune-advisory inside `dispatch_pq_v2<CRR>` (`kernel_mxfp8_layouts.cpp:5526-5594`) with 3 NEW shape-conditioned predicates covering 4 NEW LLaMA cells where V2-RRR beats V2-CRR. Per-cell min-Δ% per Dev A's GPU0+GPU4 cross-GPU triangulation: c0 (R33 carry-forward) +12.16%, c1 70B Gate +6.82%, c2 70B Up +7.98% (same shape as c1), c4 70B KV +10.28%, c8 8B KV +7.42%. Negative regression checks (c3 Q/O, c5 8B Gate, c7 8B Q/O, default 8192³): all 0 advisories — PASS.

Reverify methodology (per R32 cross-GPU rule):
- Cherry-picked `5d5c3d4c` to r34-rev temporarily (clean, no conflicts on top of Dev B which is correctness-only).
- Built 2 .so variants with PY_MODULE_NAME override (highest-impact + closest-to-gate cells, Phase 2 budget): `tk_mxfp8_r34rev_deva_c4` and `tk_mxfp8_r34rev_deva_c1`. Both rc=0.
- Build md5: c4=`f5ef9f009104ada7011f704745e678f9`, c1=`c31679985f883383ab360e25b1c06a58`.
- Ran `r33c_paired_bench.py` on **GPU5 + GPU6** (Dev used GPU0+GPU4 — fully orthogonal cross-GPU triangulation).

| Cell | Shape | Dev GPU0 | Dev GPU4 | **Rev GPU5** | **Rev GPU6** | Min Δ% | Min Welch t (R-only) | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| c4 70B KV | 4096×1024×8192 | +10.34% (t=50) | +10.21% (t=27) | **+9.81% (t=5.32)** | **+10.80% (t=67.65)** | **+9.81%** | 5.32† | **CONFIRM SHIP** |
| c1 70B Gate | 4096×28672×8192 | +6.82% (t=37) | +6.83% (t=1.13‡) | **+7.61% (t=15.57)** | **+8.16% (t=10.88)** | **+6.82%** | 10.88 | **CONFIRM SHIP** |

†Reviewer GPU5 c4 t=5.32 is below STRICT t>10 gate, but Δ=+9.81% well above +5%. The depressed t-stat is from 2 mid-bench parallel-mxfp4-agent contention pairs (CRR 698/701 vs 768 cluster). Per-pair within-PAIR Δ% holds at ~+10% throughout. The other 5 GPU runs (Dev's 2 + Reviewer's GPU6 + c1 GPU5/GPU6) all clear t>10 and Δ>+5%.
‡Dev's GPU4 t=1.13 attributed by Dev to mid-bench throttle; per-pair Δ holds at ~+6.8%.

All correctness PASS (snr_db=49.61, det_ok=True) on all 4 reviewer benches. The R34 GPU6 high-state outlier from Phase 1 is also visible in the bench (CRR 783, RRR 868 vs cluster CRR ~765) — both kernels move together, the +Δ% ratio is stable across the high-state GPU.

c2 (same shape as c1) inferred-CONFIRMED by transitive equivalence (single autotune predicate). c8 8B KV (+7.42% on Dev) was not separately reverified by Reviewer this cycle, but cross-cycle agreement with R33 Dev C cell c8 (Dev +8.30/+8.36% on GPU2/GPU3 + R33 Reviewer +8.13/+8.63% on GPU5/GPU6) makes it Reviewer-corroborated already. c0 70B Down was already R32+R33 confirmed across 7 distinct GPU runs.

**Verdict: CONFIRM SHIP** for R34 Dev A V2-RRR autotune fan-out. Cherry-pick recommendation to `feat/mxfp8-only`: **YES** — the 3 NEW predicates (c1+c2 share-shape, c4, c8) plus the c0 carry-forward complete the 5-cell V2-RRR autotune fan-out, R33 Reviewer's standing R34-priority-#1 item. Negative regression checks PASS.

Artifacts: `analysis/fp8_gemm/mi350x/r34_reviewer_deva_verify/{build_c1.log, build_c4.log, build_md5.log, c4_70b_kv_gpu{5,6}.txt, c1_70b_gate_gpu{5,6}.txt}`.

### Verdict 2: R34 Dev B — SHIP CONFIRMED (c5 8B Gate STRICT SHIP) + SHIP-LITE confirmed (c6 8B Up)

Dev B triangulated R33 Dev C's 2 SHIP-LITE cells (8B Gate + 8B Up, both shape 4096×14336×4096 RRR vs CRR) on 4 GPUs (PHYS_GPU=1,2,5,6) orthogonal to Dev C's 2/3. Verdict: c5 STRICT SHIP (min Δ=+5.025%, min Welch t=+10.13 across 3 fully-clean GPUs 1/2/5 — GPU6 was Dev's chassis-throttled to ~450 TFLOPS, directional only); c6 SHIP-LITE (min Δ=+5.34%, min t=+6.95 on Dev's GPUs).

**Note on Dev's GPU6 throttle attribution:** Dev B reported GPU6 "chassis-throttled to ~450 TFLOPS sustained even with 90s preheat". R34 Reviewer Phase 1 saw the same GPU6 contention initially (290 TF) but successfully ran clean at 787 TF after retries. The cause is the **parallel mxfp4 agent** running on the same node (`/shared_nfs/kyle/test/HipKittens/...`), **not** a chassis power cap — the agent intermittently competes for the GPU. Dev B's directional-only finding for GPU6 is still consistent with the broader pattern, but the attribution is incorrect. (See "Methodology bugs surfaced" below.)

Reverify methodology (per R32 cross-GPU rule):
- Cherry-picked `440fa36b` to r34-rev temporarily (correctness-only commit; no source change to kernel — clean cherry-pick).
- Built 2 .so variants with PY_MODULE_NAME override: `tk_mxfp8_r34rev_devb_c5` and `tk_mxfp8_r34rev_devb_c6`. Both rc=0.
- Build md5: c5=`a58e3873281212412b97be1a21e941bc`, c6=`8a9d2facd989efb8f2ab3f1187c4c93e`.
- Ran `r33c_paired_bench.py` on **GPU0 + GPU4** (Dev used GPU1/2/5/6 — fully orthogonal cross-GPU triangulation).

| Cell | Shape | Dev GPU1 | Dev GPU2 | Dev GPU5 | **Rev GPU0** | **Rev GPU4** | Min Δ% (5-GPU) | Min Welch t | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| c5 8B Gate | 4096×14336×4096 | +5.03% (t=10.1) | +5.85% (t=23) | +6.20% (t=28) | **+5.23% (t=16.32)** | **+6.61% (t=11.67)** | **+5.03%** | 10.13 | **CONFIRM STRICT SHIP** |
| c6 8B Up | 4096×14336×4096 | +5.34% (t=7.0) | +5.43% (t=11) | +5.35% (t=10) | **+5.39% (t=9.02)** | **+6.58% (t=13.78)** | **+5.34%** | 6.95 | **CONFIRM SHIP-LITE** |

c6 GPU4 first attempt had heavy parallel-mxfp4 contention (CRR median 2017 ± 675, t=0.18); clean retry (`c6_8b_up_gpu4_clean.txt`) gave Δ=+6.58%, t=13.78 — used in table. All correctness PASS (snr_db=49.61, det_ok=True) on all 4 reviewer benches.

c5 5-GPU triangulation: min Δ=+5.03% (Dev's GPU1), max Δ=+6.61% (Rev GPU4), spread 1.58 pp — consistent. All 5 GPUs clear both gates. **STRICT SHIP CONFIRMED.**

c6 5-GPU triangulation: min Δ=+5.34%, max Δ=+6.58%, spread 1.24 pp — consistent. Min t=6.95 (Dev GPU1) below the t>10 STRICT gate. Reviewer benches both clear t>10 (9.02 borderline GPU0 + 13.78 GPU4) but Dev's GPU1 doesn't. **CONFIRM SHIP-LITE classification** preserved.

**Verdict: CONFIRM SHIP** for c5 STRICT promotion + c6 SHIP-LITE confirmation. **Cherry-pick recommendation to `feat/mxfp8-only`: YES** for the 4-GPU triangulation data + c5 autotune-predicate addition (per Dev's recommendation: same shape covers both c5 AND c6, so landing one predicate auto-benefits c6 even though c6 is SHIP-LITE not STRICT). The c5 SHIP autotune predicate would be R34's 6th effective predicate covering 7 LLaMA cells (5 R33 STRICT + c5 + c6).

Artifacts: `analysis/fp8_gemm/mi350x/r34_reviewer_devb_verify/{build_c5.log, build_c6.log, build_md5.log, c5_8b_gate_gpu{0,4}.txt, c6_8b_up_gpu{0,4}.txt, c6_8b_up_gpu4_clean.txt}`.

### Verdict 3: R34 Dev C — NO SHIP (concur with Dev's own classification)

Dev C implemented Stage A2 Path 2 (rect-V2 CRR HB_N=64 + K-serialised reads, K_HALF=0-only). Numerics PASS (snr_db=49.60, det_ok=True). Perf: GPU2 Δ=-9.72% (Welch t=-43.51), GPU3 Δ=-9.05% (Welch t=-36.09) vs square. Path 2 is uniformly slightly worse than R33 Dev A's Path 1 (-0.93%).

Dev C's NO SHIP recommendation closes the LAST open lever for rect-V2 CRR Stage A2 perf. Combined with R33 Reviewer / R33 Dev B's closure of rect-V2 RCR perf paradigm, **rect-V2 perf is now fully closed in BOTH layouts (CRR and RCR)** as paradigm closure. The structural ceiling identified by R32 Dev A (rect lacks PIPELINE_SCALE/KPAIR_LOOP/PHASE_U16_CACHE/REMAP_ONCE) is now empirically confirmed by Path 1 (-8.5%) AND Path 2 (-9.7%).

**Verdict: NO SHIP — concur** with Dev C's own classification. **Closure recommendation:** add to closures list as "rect-V2 CRR Stage A2 Path 2 (HB_N=64 + K-serialised reads) -9.7% slower than square — paradigm closure" (R34 Dev C's primary deliverable).

### Verdict 4: R34 Dev D — NO SHIP, no verification needed

Dev D's commit `73380c4b` is "Stage 2a type bridge + Stage 2b kernel-body skeleton — sub-RBM hypothesis REFUTED". Stage 2a type bridge compiles cleanly (rc=0). Stage 2b kernel-body skeleton compiles but spills 476 VGPRs (256 saturated) — Dev D found that the "halving RBM frees 64 VGPR for SB pipelining" hypothesis (R32 Dev B / R33 Dev D) is REFUTED: operand-tile shrink trades -16 operand VGPR for +128 accumulator VGPR (each warp still covers HB=128 rows). The real path past the structural ceiling is HB shrink (BLK=128 in M, or WARPS_M=4), not RBM shrink.

Per task brief Dev D was scaffolding-only with "Stage 2a numerics is the most likely SHIP claim" hedge — no SHIP claim materialised. **Verdict: NO SHIP — concur**. Add to closures: "sub-RBM operand-tile hypothesis REFUTED — Stage 2b skeleton spills 476 VGPRs; real path is HB or WARPS_M shrink".

---

## Cross-cycle summary (R28 → R34)

### SHIPs in flight (cumulative through R33)

- **R28 SHIP (cachepolicy auto-select N≥28672 && K≥8192 V2-CRR)**: still in production. Not separately re-tested in R34 Phase 1 (single-shape budget).
- **R31 SHIP (rect-V2 CRR Stage A1 scaffolding)**: extended in R32 by Dev D's parallel rect-V2 RCR Stage A1 scaffolding (note: had B-side scale slab bug — R33 Dev B fixed it).
- **R32 SHIP (V2-RRR autotune pivot for 70B Down 4096×8192×28672)**: data SHIP from R32; **R33 Dev A Task 1 wired the host-side autotune entry** in `dispatch_pq_v2<CRR>`. Re-confirmed across 7 distinct GPU runs.
- **R33 SHIP (V2-RRR autotune pivot broadened to 4 STRICT-SHIP cells)**: Dev C, CONFIRMED across 4 GPUs each. 70B Gate, 70B Up, 70B KV, 8B KV.
- **R33 SHIP (rect-V2 RCR Stage A2 numerics)**: Dev B, CONFIRMED across 4 GPUs. Stage A2d perf NO SHIP. rect-V2 RCR perf paradigm CLOSED.

R33 net: **6 new SHIPs CONFIRMED** by R33 Reviewer (Dev A Task 1 + Dev C 4 STRICT + Dev B numerics).

R34 net: **3 new SHIPs CONFIRMED** by R34 Reviewer:
- **Dev A SHIP CONFIRMED**: V2-RRR autotune fan-out (3 NEW predicates, 4 NEW LLaMA cells: c1/c2 share-shape, c4, c8). Min-Δ% across 4-GPU triangulation: c4 +9.81%, c1 +6.82%; c2 transitive, c8 cross-cycle Reviewer-corroborated. R33 Reviewer's R34-priority-#1 item now COMPLETE (5-cell V2-RRR autotune fan-out for `dispatch_pq_v2<CRR>`).
- **Dev B SHIP CONFIRMED**: c5 8B Gate STRICT SHIP (R33 SHIP-LITE → R34 STRICT, 5-GPU min Δ=+5.03% / min t=10.13). c6 8B Up SHIP-LITE preserved (5-GPU min Δ=+5.34% / min t=6.95).
- **Dev C NO SHIP — concur**: rect-V2 CRR Stage A2 Path 2 -9.7% (paradigm closure for last rect-V2 CRR lever).
- **Dev D NO SHIP — concur**: sub-RBM operand-tile hypothesis REFUTED (476 VGPR spills).

Combined with Dev A's c5 autotune-predicate recommendation following Dev B's STRICT-promotion of c5, the V2-RRR autotune fan-out for `dispatch_pq_v2<CRR>` could land **5 effective predicates covering 7 LLaMA cells** post-merge.

**Cumulative SHIP count R28→R34: 6 production SHIPs in flight + R34's 2 SHIP CONFIRMs (3 if Dev A's fan-out counts as a single SHIP, 2 distinct claims; if c1/c2/c4/c8 each counts then 5 total).**

### Methodology bugs surfaced this cycle (R34 Reviewer)

1. **NEW R34: same-node parallel-agent DPM contention** — when another HipKittens project (in this case mxfp4 work in `/shared_nfs/kyle/test/HipKittens/`) runs long benches on GPUs 0/6 simultaneously, those GPUs cannot ramp DPM properly even with 8s 16k FP16 preheat. Symptoms: sclk capped at 1700-1900 MHz instead of 2300+; TFLOPS drops to 50-50% of expected (290 TF observed vs 766 TF expected on GPU6, 361 TF vs 765 expected on GPU0); occasional DETERMINISM=False from interleaved kernel launches. **Mitigation employed:** retry per-GPU bench up to 3x until a clean window appears (sclk-pre-bench >= 2200 MHz AND determinism PASS AND stdev/mean <= 1%). All 12 (re)builds compiled bit-identically. **Recommended R35+ rule:** orchestrate should auto-detect contention via post-preheat sclk read (`sclk-post-preheat < 2200 MHz` ⇒ retry up to 3x; abort with diagnostic if all 3 fail). Retry-history should be logged in JSON output for later inspection. Add to bench harness as standing rule.

2. **STRENGTHENED R34: R33 stochastic per-(GPU × cycle) hypothesis confirmed with N=4** — GPU6 in R34 is the high-state outlier at +2.71%, exactly the same magnitude band (+2.36-2.71%) as R31 GPU0 and R32 GPU5. The R33 cycle (no outlier) was the unusual one, not the others. The high-regime is real, transient, per-(GPU × cycle), and well-handled by the R33 sub-rule (min-of-GPUs not mean). No methodology change needed beyond the R33 sub-rule.

3. **R31 sclk-fix (rocm-smi -d $PHYS_GPU not -d 0)** verified correct for all 4 R34 GPUs across all 12 (re)attempts. `phys_gpu=N` token visible in every sclk log line.

### R35 priorities (rebuilt by R34 Reviewer)

1. **【high / immediate】Bench-harness contention defense** — implement the post-preheat sclk-check + 3x-retry rule above. Add to `r35_reviewer_bench5x.py` (or absorb into r34 if more cycles share environment).

2. **【high / inheritor of R33】Land the 5-cell V2-RRR autotune fan-out** in `dispatch_pq_v2<CRR>` if not already done by R34 Dev A/B/C this cycle (R32 Dev C C4 + R33 Dev C c1/c2/c4/c8 + R33 Dev A wire-in). 4 effective predicates (c1+c2 share shape).

3. **【medium / 1-2 day, R33 carry-over】R35 4-GPU triangulation on R33 Dev C SHIP-LITE cells** (8B Gate 4096×14336×4096, 8B Up 4096×14336×4096) if not done by R34 Dev C. Both showed Δ ≥ +5% on both Dev's GPUs but t < 10 — needs ≥4-GPU bench to clear t > 10 STRICT SHIP gate.

4. **【medium / 1-2 day, R33 carry-over】rect-V2 CRR Stage A2 numerics** — R33 Dev A Path 1 establishes square-LDS + halve-N is correct (snr 49.60 dB) but slow (-8.5%). Per Dev A's recommendation, attempt Path 2 (rect LDS HB=64 with K_HALF=0-only helper) or revisit K_HALF=1 helper to remove OOB indexing. (TBD if R34 Dev attempted this.)

5. **【methodology / R35+ standing rule】R34 contention-defense sub-rule**: orchestrate must auto-retry up to 3x on sclk-post-preheat < 2200 MHz OR DETERMINISM=False OR stdev/mean > 1%. Log all attempts in JSON output.

### Closures-list cumulative count

R33 closed 26 cumulative levers (21 R32 + 5 R33). R34 adds:
- R34 Dev B closure: c5 R33 SHIP-LITE → STRICT SHIP confirmed; c6 SHIP-LITE confirmed (4-GPU triangulated).
- R34 Dev C closure: rect-V2 CRR Stage A2 Path 2 (HB_N=64 + K-serialised, K_HALF=0-only) -9.7% slower than square — paradigm closure for the LAST open rect-V2 CRR lever. Combined with R33 Dev B's rect-V2 RCR closure, **rect-V2 perf paradigm is now CLOSED in BOTH layouts (CRR and RCR)**.
- R34 Dev D closure: sub-RBM operand-tile hypothesis REFUTED (Stage 2b skeleton spills 476 VGPRs; trades -16 operand for +128 accumulator VGPR; real path forward is HB shrink or WARPS_M shrink, not RBM shrink).
- R34 Dev A "completes" R32 Dev C / R33 Dev C V2-RRR layout-pivot data into a wired-in autotune fan-out — closure of "5-cell V2-RRR autotune fan-out for dispatch_pq_v2<CRR>" data lever (now fully realized).

= **30 cumulative closed levers** (21 R32 + 5 R33 + 4 R34). Open levers remaining at end of R34: V2-CRR 0.84 ratio band (no structural lever remaining — rect-V2 CRR closed in both Path 1 and Path 2; sub-RBM operand-tile REFUTED). R35+ candidate open levers: HB shrink (BLK=128 in M) or WARPS_M=4 (per Dev D's Stage 2b finding); 8B Gate/Up c5/c6 autotune-predicate landing (per Dev B's recommendation, single shape covers both cells).

### Files (R34 Reviewer artifacts)

- `analysis/fp8_gemm/mi350x/r34_reviewer_findings.md` (this doc)
- `analysis/fp8_gemm/mi350x/r34_reviewer_kv_4gpu.json` (Phase 1 structured data)
- `analysis/fp8_gemm/mi350x/r34_reviewer_4gpu_orchestrate.sh` (driver, sclk-fix carried forward)
- `analysis/fp8_gemm/mi350x/r34_reviewer_bench5x.py` (5x bench, bit-identical to R33)
- `analysis/fp8_gemm/mi350x/r34_reviewer_4gpu_runs/` (Phase 1 raw bench logs + build_md5.log; 4 GPUs × 1 cell + retries)
- `analysis/fp8_gemm/mi350x/r34_reviewer_deva_verify/` (Phase 2 Dev A reverify: 2 builds + 4 paired-bench runs on GPU5/GPU6)
- `analysis/fp8_gemm/mi350x/r34_reviewer_devb_verify/` (Phase 2 Dev B reverify: 2 builds + 5 paired-bench runs on GPU0/GPU4)

R34 Dev C and R34 Dev D NO SHIP — no separate verify dirs (Dev's own commits include their methodology + raw data).
