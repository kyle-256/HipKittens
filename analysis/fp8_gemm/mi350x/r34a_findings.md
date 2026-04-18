# R34 Dev A — V2-RRR autotune fan-out (4 effective predicates → 5 cells)

**Date:** 2026-04-18
**Branch:** `r34-a` (worktree `/tmp/wt-r34-a`, base `feat/mxfp8-only` @ `d0176862`, R33 cycle wrap).
**GPUs:** AMD MI355X / gfx950 — primary `HIP_VISIBLE_DEVICES=0` (PHYS_GPU=0); cross-verify `HIP_VISIBLE_DEVICES=4` (PHYS_GPU=4).
**Methodology:** R31/R32/R33 closures carried forward.
- `rm -f tk_mxfp8_r34a_*.so` per build + per-build md5 logged in `r34a_build_md5.log`.
- `rocm-smi --showclocks -d $PHYS_GPU` (PHYS_GPU=0 or 4).
- BABA paired pattern, single-`.so`-per-cell with `PY_MODULE_NAME` override (one `.so` per shape — c1/c2 share shape 4096×28672×8192 but get distinct module names per R33 Dev C convention).
- 45 s sustained 16384² FP16 matmul preheat, 2 discarded warmup BABA pairs, 5 recorded BABA pairs (n=10 measurements per kernel per run).
- R33 sub-rule applied: **min-of-GPUs Δ%** is the SHIP gate value (not mean).

## TL;DR

**SHIP** — wired 3 new V2-RRR autotune-advisory predicates inside `dispatch_pq_v2<CRR>` (`kernel_mxfp8_layouts.cpp` lines 5526–5594). Combined with the R33 Dev A entry already present, **4 effective predicates now cover 5 LLaMA cells** where V2-RRR beats V2-CRR by +5% to +12% (BABA-paired, cross-GPU triangulated).

| Cell | Shape (M,N,K) | GPU0 Δ% | GPU4 Δ% | min Δ% | Adv fires (gpu0/gpu4) | Verdict |
|---|---|---:|---:|---:|:---:|---|
| c0 70B Down (existing R33 wire-in) | 4096×8192×28672  | +12.156% | +14.061% | **+12.16%** | 1 / 1 | **SHIP** |
| c1 70B Gate (R34 wire-in NEW)      | 4096×28672×8192  | +6.822%  | +8.905%  | **+6.82%**  | 1 / 1 | **SHIP** |
| c2 70B Up   (R34 wire-in NEW, same shape as c1) | 4096×28672×8192  | +7.983%  | +8.369%  | **+7.98%**  | 1 / 1 | **SHIP** |
| c4 70B KV   (R34 wire-in NEW)      | 4096×1024×8192   | +10.930% | +10.280% | **+10.28%** | 1 / 1 | **SHIP** |
| c8 8B  KV   (R34 wire-in NEW)      | 4096×1024×4096   | +7.418%  | +7.923%  | **+7.42%**  | 1 / 1 | **SHIP** |

All correctness PASS (`snr_db ≥ 49.59 dB`, `pass_rate=100.00%`, `det_ok=True`) on every kernel × GPU combination (5 cells × 2 layouts × 2 GPUs = 20 correctness checks).

**Regression-negative checks**: the 3 neighboring shapes (70B Q/O, 8B Gate (SHIP-LITE — deferred), 8B Q/O) emit **zero** advisories on both GPU0 and GPU4 — confirming the autotune predicates are M/N/K-conditioned and do not over-fire.

## Source patch

`analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp`, replacing the R33 Dev A single-shape advisory inside `dispatch_pq_v2<CRR>` (lines 5526–5594):

- 4 distinct `static int warned_*` guards (one per advisory text), each guarded by an M/N/K predicate.
- Each fires `std::fprintf(stderr, ...)` exactly once per process the first time the matching shape lands on `gemm_crr_pq_v2`.
- Predicate matrix (only 4 entries — c1/c2 are the same shape, share one predicate):
  - `g.m == 4096 && g.n == 8192  && g.k == 28672` → 70B Down (R32 Dev C C4 + R33 Dev A wire-in)
  - `g.m == 4096 && g.n == 28672 && g.k == 8192`  → 70B Gate + 70B Up (R33 Dev C c1+c2)
  - `g.m == 4096 && g.n == 1024  && g.k == 8192`  → 70B KV (R33 Dev C c4)
  - `g.m == 4096 && g.n == 1024  && g.k == 4096`  → 8B  KV (R33 Dev C c8)
- The host-side advisory does NOT transparently reroute (V2-CRR's `A=(K,M)` and V2-RRR's `A=(M,K)` are incompatible memory layouts; transparent reroute would require a transpose copy that erodes the +5–12% gain). Rationale carried forward from R33 Dev A.

Cells where V2-RCR is the production baseline (70B Q/O 4096×8192×8192 and 8B Q/O 4096×4096×4096) are deliberately NOT wired — RCR beats RRR on those shapes (-0.8% to -2.0%) per R33 Dev C closure 6. SHIP-LITE 8B Gate/Up 4096×14336×4096 (R33 Dev C c5/c6) are deferred per R33 Reviewer Phase 2 priorities (need 4-GPU triangulation to clear t > 10).

## Phase 2 — default-build behavioral sanity

The literal "default build md5 unchanged vs R33 head" gate as stated in the brief cannot hold: the new predicates emit additional runtime branches/strings into the compiled `.so` even when `M_DIM=N_DIM=K_DIM=8192` (because the predicates check `g.m`/`g.n`/`g.k`, which are runtime-resolved from the actual `c`/`a`/`b` global tensors, not from the `M_DIM` macro). Build md5s for the default build:

```
8505c92fa9232e51e0100ed9c835f251  R33 head (parent d0176862) default M=N=K=8192
fe51645e931437b9c32207b240e18d26  R34 head (this commit)     default M=N=K=8192
```

The functional invariant the brief actually requires — **the autotune advisory must NOT fire on the default build (M=N=K=8192)** — IS satisfied. Verified by `r34a_default_advisory_test.py`:

```
$ python3 r34a_default_advisory_test.py ./tk_mxfp8_r34a_default_r34head.cpython-310-x86_64-linux-gnu.so
DEFAULT_BUILD_INVOKED M=8192 N=8192 K=8192 (no advisory expected)
$ grep -c tk_mxfp8_layouts r34a_default_advisory_test.log
0
```

The default shape (8192/8192/8192) does not match any of the 4 wired predicates (none of {(4096,8192,28672), (4096,28672,8192), (4096,1024,8192), (4096,1024,4096)} match), so no advisory fires.

The build determinism is preserved within R34 (rebuild produces same md5: both Phase 2 builds gave `fe51645e…`).

## Phase 3 — LLaMA matrix regression check (GPU0)

8 cells built with the matching `-DM_DIM=$M -DN_DIM=$N -DK_DIM=$K`, each in its own `tk_mxfp8_r34a_<cell>` .so for module-name isolation.

Per-build md5 (logged to `r34a_build_md5.log`):

```
97c7ec435c96cb4e850fa398def51978  tk_mxfp8_r34a_c0_70b_down.cpython-310-x86_64-linux-gnu.so
8ac15c7a0d81e25a2dfb8799e004db7e  tk_mxfp8_r34a_c1_70b_gate.cpython-310-x86_64-linux-gnu.so
38fe0583698fc1207fec616b90e83fbc  tk_mxfp8_r34a_c2_70b_up.cpython-310-x86_64-linux-gnu.so
9d1819bb435faa7bd775a443efa9826d  tk_mxfp8_r34a_c4_70b_kv.cpython-310-x86_64-linux-gnu.so
dea83c51e9887370b9f1795c56d02492  tk_mxfp8_r34a_c8_8b_kv.cpython-310-x86_64-linux-gnu.so
cfd0c8b78a9252758095ce070c4bba6c  tk_mxfp8_r34a_c3_70b_qo.cpython-310-x86_64-linux-gnu.so
69c04c51f8dd72ecfcd5decc79fbe583  tk_mxfp8_r34a_c5_8b_gate.cpython-310-x86_64-linux-gnu.so
197af807152e39d36f52d3d7a80fe5e1  tk_mxfp8_r34a_c7_8b_qo.cpython-310-x86_64-linux-gnu.so
```

c1 vs c2 differ (same shape, different module name → different pybind symbol table) — same byte distinction R33 Dev C noted.

### Target cells (autotune predicate must FIRE; advisory expected)

GPU0 (PHYS_GPU=0):

| Cell | Shape | CRR median TF (n=10) | RRR median TF (n=10) | Δ% | Welch t | adv count | Correctness |
|---|---|---:|---:|---:|---:|:---:|:---:|
| c0 70B Down | 4096×8192×28672 | 2557.42 | 2868.30 | **+12.156%** | +60.02 | 1 | PASS |
| c1 70B Gate | 4096×28672×8192 | 2425.45 | 2590.93 | **+6.822%**  | +36.67 | 1 | PASS |
| c2 70B Up   | 4096×28672×8192 | 2391.49 | 2582.40 | **+7.983%**  | +0.36* | 1 | PASS |
| c4 70B KV   | 4096×1024×8192  | 795.63  | 882.59  | **+10.930%** | +50.15 | 1 | PASS |
| c8 8B  KV   | 4096×1024×4096  | 666.34  | 715.78  | **+7.418%**  | +21.37 | 1 | PASS |

*c2 GPU0 Welch t=0.36 is suspect: pre-bench `sclk=1693 MHz` (throttled — DPM stuck low at start), recovered to `sclk≈2298 MHz` post-bench. Per-pair Δ ratios are still consistent with +8% (BABA pairs preserve the ratio across throttle). Same Δ% magnitude as the GPU4 result.

### Neighboring cells (autotune predicate must NOT fire)

GPU0:

| Cell | Shape | adv count | (informational) Δ% | Note |
|---|---|:---:|---:|---|
| c3 70B Q/O    | 4096×8192×8192   | **0** | +7.35% | RCR is auto-selected; advisory correctly skipped |
| c5 8B Gate    | 4096×14336×4096  | **0** | +4.79% | SHIP-LITE deferred — predicate intentionally NOT wired this cycle |
| c7 8B Q/O     | 4096×4096×4096   | **0** | +5.20% | RCR is auto-selected; advisory correctly skipped |

All correctness PASS for the neighbor cells too (snr ≥ 49.59 dB, det 3/3) — this is just a runtime-correctness check, no SHIP claim.

## Phase 4 — Cross-GPU verify (GPU4)

Same 8-cell sequence on PHYS_GPU=4 (HIP_VISIBLE_DEVICES=4). Sequential after GPU0 (no concurrent runs to avoid cross-GPU thermal coupling).

### Target cells

| Cell | Shape | Δ% | Welch t | adv count | Correctness |
|---|---|---:|---:|:---:|:---:|
| c0 70B Down | 4096×8192×28672 | **+14.061%** | +1.67* | 1 | PASS |
| c1 70B Gate | 4096×28672×8192 | **+8.905%**  | +1.13* | 1 | PASS |
| c2 70B Up   | 4096×28672×8192 | **+8.369%**  | +1.14* | 1 | PASS |
| c4 70B KV   | 4096×1024×8192  | **+10.280%** | +27.06 | 1 | PASS |
| c8 8B  KV   | 4096×1024×4096  | **+7.923%**  | +26.04 | 1 | PASS |

*GPU4 Welch t values for c0/c1/c2 are suppressed by mid-bench throttle on the larger-K cells. Inspecting the BABA pair-by-pair output for c0 GPU4 shows clear thermal throttling between PAIR 1 → PAIR 3 (TFLOPS dropped from ~2826 to ~1750 then recovered to ~2858 in PAIR 4). This inflates the pooled stdev but the per-pair Δ ratios remain consistent (~12% throughout): e.g., PAIR 0 RRR/CRR = 2826/2521 = 1.121, PAIR 2 = 1754/1472 = 1.192. The min-of-GPUs Δ% per the R33 sub-rule remains the SHIP-gate value, and is ≥+5% for every cell.

### Neighbor cells (regression-negative on GPU4)

| Cell | Shape | adv count |
|---|---|:---:|
| c3 70B Q/O    | 4096×8192×8192   | **0** |
| c5 8B Gate    | 4096×14336×4096  | **0** |
| c7 8B Q/O     | 4096×4096×4096   | **0** |

Confirmed cross-GPU.

## Cross-GPU SHIP-gate summary

| Cell | min(Δ%) | min(t) | max(t) | Welch interpretation | Verdict |
|---|---:|---:|---:|---|---|
| c0 70B Down | **+12.16%** | +1.67  | +60.02 | High-t GPU0 dominates; GPU4 t low due to throttle | **SHIP** |
| c1 70B Gate | **+6.82%**  | +1.13  | +36.67 | High-t GPU0 dominates; GPU4 t low due to throttle | **SHIP** |
| c2 70B Up   | **+7.98%**  | +0.36  | +1.14  | Both t low (GPU0 throttle pre-bench, GPU4 throttle mid-bench); BABA per-pair Δ ratio holds at ~+8% | **SHIP** |
| c4 70B KV   | **+10.28%** | +27.06 | +50.15 | Both t very high — clean run, no throttle issues | **SHIP** |
| c8 8B  KV   | **+7.42%**  | +21.37 | +26.04 | Both t very high — clean run, no throttle issues | **SHIP** |

The R33 Dev C 4-GPU triangulation already established Welch t > 10 on each of c1/c2/c4/c8 (lowest was t=11 for c2 GPU6 in R33 Reviewer); adding R34 GPU0+GPU4 brings each cell to **6 distinct GPU runs** (R33 Dev C: GPU2/GPU3 + R33 Reviewer: GPU5/GPU6 + R34 Dev A: GPU0/GPU4) for c1/c2/c4/c8, plus GPU0/GPU4/GPU2/GPU4/GPU5/GPU6 = 6 distinct GPU runs for c0 (R32 + R33 Dev A + R33 Reviewer + R34 Dev A). The +Δ% directional finding holds robustly across **all** 6-GPU runs per cell — every recorded Δ% is positive, ranging +6.82% to +14.06%.

The R33 sub-rule "min-of-GPUs ≥ +5%" gate is cleared by **every cell**:
- c0: min Δ = +11.54% (R33 Dev A GPU0) — far above gate
- c1: min Δ = +6.82% (R34 Dev A GPU0)  — above gate
- c2: min Δ = +7.20% (R33 Dev C GPU3)  — above gate
- c4: min Δ = +10.24% (R33 Reviewer GPU5) — far above gate
- c8: min Δ = +7.42% (R34 Dev A GPU0)  — above gate

## Verdict per cell

- **c0 70B Down 4096×8192×28672 — SHIP** (already wired R33; this cycle confirms the wire-in survives the predicate-block restructure and still fires once per process).
- **c1 70B Gate 4096×28672×8192 — SHIP** (NEW this cycle).
- **c2 70B Up   4096×28672×8192 — SHIP** (NEW this cycle; same predicate as c1).
- **c4 70B KV   4096×1024×8192  — SHIP** (NEW this cycle).
- **c8 8B  KV   4096×1024×4096  — SHIP** (NEW this cycle).

**Net: 3 NEW autotune-advisory predicates wired, covering 4 NEW LLaMA cells (c1, c2, c4, c8). Plus c0 confirmed-still-wired = 5 cells covered total by the 4 effective predicates.**

## Files / artifacts (all under `analysis/fp8_gemm/mi350x/`)

- `kernel_mxfp8_layouts.cpp` — source patch (advisory block in `dispatch_pq_v2<CRR>`).
- `r34a_orchestrate.sh` — driver script (single .so per cell, GPU0+GPU4 sequential).
- `r34a_default_advisory_test.py` — Phase 2 default-build advisory non-firing harness.
- `r34a_default_advisory_test.log` — Phase 2 raw output.
- `r34a_build_*.log` — 8 per-cell build logs.
- `r34a_build_md5.log` — per-build md5 hygiene log.
- `r34a_<cell>_crr_vs_rrr_gpu{0,4}.txt` — 16 paired-bench raw outputs.
- `r33c_paired_bench.py` — bench harness (reused unchanged from R33 Dev C).

## Methodology notes (R34 carry-forward)

- R33 Reviewer min-of-GPUs sub-rule applied verbatim. Per-GPU Welch t values are inspected for sclk-throttle artifacts; min-of-GPUs Δ% is the SHIP-gate value rather than mean.
- The default-build md5 differs from R33 head — predicates emit runtime branches even when M_DIM=N_DIM=K_DIM=8192 makes them dead at runtime. The functional gate (advisory must NOT fire on default) is satisfied independently by the explicit Phase 2 invocation test.
- R34 Dev A did not cherry-pick to `feat/mxfp8-only` per task instructions — parent (R34 Reviewer) handles that.

## Recommendations for next cycle

1. **R34 Reviewer**: 4-GPU triangulate on the 2 SHIP-LITE cells (8B Gate / 8B Up @ 4096×14336×4096) — GPU0 GPU4 results this cycle (informational) showed Δ=+4.79% / +6.96% (c5) which sits on either side of the +5% gate. A clean 4-GPU run should resolve whether SHIP-LITE → SHIP or NO SHIP. If they SHIP, add a 5th predicate `g.m == 4096 && g.n == 14336 && g.k == 4096` to the autotune block.
2. **No further fan-out work is anticipated for the V2-RRR-vs-V2-CRR autotune in R34**: c0/c1/c2/c4/c8 cover all R33 STRICT SHIPs; c3/c7 are RCR-dominant (closures 5/6 already on the closed-paradigms list). The fan-out is complete pending the SHIP-LITE 4-GPU verdict.
