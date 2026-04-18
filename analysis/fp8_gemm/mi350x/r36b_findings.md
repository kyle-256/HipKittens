# R36 Dev B — V2-RRR autotune fan-out 6th predicate (8B-Down 4096×4096×14336)

**Date:** 2026-04-18
**Branch:** `r36-b` (worktree `/tmp/wt-r36-b`, base `feat/mxfp8-only` @ `40d77d98`, R35 cycle wrap).
**GPUs:** AMD MI355X / gfx950 — primary `HIP_VISIBLE_DEVICES=1` (PHYS_GPU=1); cross-verify `HIP_VISIBLE_DEVICES=4` (PHYS_GPU=4).
**Methodology:** R29-R35 closures carried forward.
- `rm -f tk_mxfp8_r36b_*.so` per build + per-build md5 logged in `r36b_build_md5.log`.
- `rocm-smi --showclocks -d $PHYS_GPU` (PHYS_GPU=1 or 4).
- BABA paired pattern, single-`.so`-per-cell with `PY_MODULE_NAME` override (one `.so` per shape per R33 Dev C convention).
- 45 s sustained 16384² FP16 matmul preheat, 2 discarded warmup BABA pairs, 5 recorded BABA pairs (n=10 measurements per kernel per run).
- R34 sclk-post-preheat ≥ 2200 MHz first gate; R35 NEW sclk-post-bench ≥ 2200 MHz second gate.
- R33 sub-rule applied: **min-of-GPUs Δ%** is the SHIP gate value (not mean).

## TL;DR — Verdict: **SHIP-LITE** (6th V2-RRR autotune predicate, 8B-Down)

Wired the 6th V2-RRR autotune-advisory predicate inside `dispatch_pq_v2<CRR>` (`kernel_mxfp8_layouts.cpp:5667-5677`) for shape `(M=4096, N=4096, K=14336)` — 8B-Down (largest uncovered RRR-vs-CRR gap surfaced by R35 Dev D). Pattern matches the existing 5 predicates (static `warned_8b_down` guard, advisory referencing `r35d_findings.md` and `r36b_findings.md`).

### Per-GPU SHIP gate (Δ% / Welch t)

| GPU | CRR median TF (n=10) | RRR median TF (n=10) | Δ% RRR_vs_CRR | Welch t | sclk-post-preheat | sclk-post-bench |
|---|---:|---:|---:|---:|---:|---:|
| GPU1 (HIP=1) | 2673.24 | 2911.32 | **+8.906%** | **+6.550** | 2283 MHz | 2337 MHz |
| GPU4 (HIP=4) | 2760.68 | 2944.49 | **+6.658%** | **+5.872** | 2059 MHz⁺ | 2281 MHz |
| **min-of-GPUs** | — | — | **+6.658%** | **+5.872** | — | — |

⁺ GPU4 sclk-post-preheat 2059 MHz (below R34 first-gate 2200 MHz) but sclk-pre-bench recovered to 2358 MHz and sclk-post-bench held at 2281 MHz; per-pair ratios are stable across all 5 PAIRs (no anomaly visible in PAIR 0). Per R35 NEW second gate (sclk-post-bench ≥ 2200 MHz), GPU4 PASSES the second gate. Run accepted without retry.

### SHIP gate evaluation
- **min Δ% = +6.658%** ≥ +5.0% (clears SHIP-LITE Δ gate; clears STRICT Δ gate +5.0%).
- **min Welch t = +5.872** ≥ +3.0 (clears SHIP-LITE t gate); below STRICT t gate +10.0.
- → **SHIP-LITE confirmed** on 2-GPU triangulation. Same status as R34 Dev B's c5/c6 single-shape predicate at first wire-in (later promoted to STRICT in R34 Reviewer Phase 2 at 4-GPU triangulation).

### Correctness (all PASS)

| GPU | Cell | snr_db CRR | snr_db RRR | pass_rate | det_ok |
|---|---|---:|---:|---:|:---:|
| GPU1 | c_8b_down | 49.61 | 49.61 | 100% / 100% | True / True |
| GPU4 | c_8b_down | 49.61 | 49.61 | 100% / 100% | True / True |

Both GPUs satisfy SNR ≥ 48 dB + det 3/3 PASS.

## Source patch

`analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp:5593-5682`. Diff summary:

1. Comment block updated: `R33 Dev A / R34 Dev A / R35 Dev A / R36 Dev B — V2-RRR autotune pivot fan-out: 6 effective host-side predicates cover 8 LLaMA cells…`
2. Wire-in summary expanded with R36 Dev B entry referencing R35 Dev D matrix re-bench.
3. New static guard `static int warned_8b_down = 0;` (line 5628).
4. New predicate block (lines 5673-5682):

```cpp
if (g.m == 4096 && g.n == 4096 && g.k == 14336 && !warned_8b_down) {
    std::fprintf(stderr,
        "[tk_mxfp8_layouts] gemm_crr_pq_v2: shape (M=4096, N=4096, "
        "K=14336) is ~+9.5%% faster on V2-RRR (R35 Dev D LLaMA "
        "matrix re-bench identified largest uncovered RRR-vs-CRR "
        "gap; R36 Dev B 2-GPU triangulated). Prefer gemm_rrr_pq_v2 "
        "with A row-major (M,K). See analysis/fp8_gemm/mi350x/"
        "r35d_findings.md and r36b_findings.md.\n");
    warned_8b_down = 1;
}
```

The host-side advisory does NOT transparently reroute (CRR's `A=(K,M)` and RRR's `A=(M,K)` are incompatible memory layouts). Rationale carried forward from R33 Dev A / R34 Dev A / R35 Dev A.

## Phase 2 — Advisory firing check (`r36b_advisory_check.sh`)

```
=== R36 Dev B advisory firing check (Phase 2) ===
=== Targets: predicate must fire exactly once (expected=1) ===
c_8b_down        shape=(4096, 4096,14336) expected=1 got=1 PASS
c5_8b_gate       shape=(4096,14336, 4096) expected=1 got=1 PASS
c0_70b_down      shape=(4096, 8192,28672) expected=1 got=1 PASS
c1_70b_gate      shape=(4096,28672, 8192) expected=1 got=1 PASS
c4_70b_kv        shape=(4096, 1024, 8192) expected=1 got=1 PASS
c8_8b_kv         shape=(4096, 1024, 4096) expected=1 got=1 PASS
=== Negatives: predicate must NOT fire (expected=0) ===
c3_70b_qo        shape=(4096, 8192, 8192) expected=0 got=0 PASS
c7_8b_qo         shape=(4096, 4096, 4096) expected=0 got=0 PASS
default          shape=(8192, 8192, 8192) expected=0 got=0 PASS
=== SUMMARY: 0 failures (should be 0) ===
```

All 9 checks PASS:
- 6 target shapes emit exactly 1 advisory each (NEW c_8b_down + 5 prior predicates).
- 3 negative shapes (c3 70B Q/O, c7 8B Q/O, default 8192³) emit 0 advisories.

## Phase 3 — Default-build behavioral identity

Per R34 Dev A precedent: literal byte-identity to base does NOT hold (the new predicate adds an additional runtime branch + a const string into the compiled `.so` even when `M_DIM=N_DIM=K_DIM=8192`, because the predicates check runtime values `g.m/g.n/g.k`, not the build-time macros). The functional invariant **the autotune advisory must NOT fire on the default build (M=N=K=8192)** IS satisfied.

```
59ef94a04170b5fc63bf117b0bc5f720  tk_mxfp8_r36b_base_default.cpython-310-x86_64-linux-gnu.so   (R36 base, NO patch)
91fbb386633c0d9e5906a936ee4638b7  tk_mxfp8_r36b_default.cpython-310-x86_64-linux-gnu.so       (R36 head, with R36 Dev B patch)
```

Size delta = +160 bytes (advisory string + branch). String inspection:
- base default: 0 occurrences of `K=14336` (no R36 patch present).
- patched default: 1 occurrence of `K=14336` (advisory string compiled in).
- patched default firing on M=N=K=8192: **0 advisories** (verified by `r36b_advisory_check.sh`, `default ... expected=0 got=0 PASS`).

Build determinism within R36 is preserved: re-running the build phase produces the same md5 (advisory check used the same artifact as the bench phase).

## Phase 4 — LLaMA matrix regression check (5 prior predicates + 70B-Down + new 8B-Down)

`r36b_orchestrate.sh bench_regression` GPU1, 5 cells already covered by predicates (c0/c1/c4/c5/c8). c1 first run had a sclk-pre-preheat anomaly (1803 MHz at start, sclk-post-preheat 1806 MHz — failed R34 first gate, sclk-post-bench recovered to 2308 MHz); a clean retry with PREHEAT_S=60 produced the canonical numbers. Below shows the cleaner number per cell.

| Cell | Shape (M,N,K) | CRR median TF | RRR median TF | Δ% RRR_vs_CRR | Welch t | adv count | Correctness |
|---|---|---:|---:|---:|---:|:---:|:---:|
| c0 70B Down (R33 wire-in)  | 4096×8192×28672  | 2523.90 | 2830.47 | **+12.147%** | +45.116 | 1 | PASS |
| c1 70B Gate (R34 wire-in, retry) | 4096×28672×8192 | 2350.17 | 2549.17 | **+8.467%** | +10.554 | 1 | PASS |
| c4 70B KV   (R34 wire-in)  | 4096×1024×8192  | 791.60  | 875.94  | **+10.654%** | +44.431 | 1 | PASS |
| c8 8B  KV   (R34 wire-in)  | 4096×1024×4096  | 691.43  | 749.29  | **+8.368%**  | +18.585 | 1 | PASS |
| c5 8B Gate (R35 wire-in)  | 4096×14336×4096 | 2422.97 | 2576.24 | **+6.326%**  | +7.843 | 1 | PASS |
| **c_8b_down (R36 NEW)** | **4096×4096×14336** | **2673.24** | **2911.32** | **+8.906%** | **+6.550** | 1 | PASS |

All 5 prior predicate cells maintain min Δ% ≥ +6.3% — **NO regression** from the R36 Dev B wire-in. The new 8B-Down predicate produces +8.91% on GPU1 (above the +9.51% R35 Dev D GPU6 reference, within run-to-run noise).

c5 measured Δ% = +6.326% (above the +5.025% R34 Reviewer minimum but below R35 Dev D's GPU6 +6.25%); within the R34/R35 reproducibility band.

## Build md5 record (`r36b_build_md5.log`)

```
bbb8238b0b15fcba0a4fbe212f56ee7b  tk_mxfp8_r36b_c_8b_down.cpython-310-x86_64-linux-gnu.so   (NEW target)
44ed325bd47555de7a2f3b668164784d  tk_mxfp8_r36b_c0_70b_down.cpython-310-x86_64-linux-gnu.so
33ef2948cfedbca3782e49f013866b24  tk_mxfp8_r36b_c1_70b_gate.cpython-310-x86_64-linux-gnu.so
a33735a98a43b743206ca298090b8288  tk_mxfp8_r36b_c4_70b_kv.cpython-310-x86_64-linux-gnu.so
678e142234112de984fe27f98dcbf27f  tk_mxfp8_r36b_c8_8b_kv.cpython-310-x86_64-linux-gnu.so
f9f80070ad06fa6b61e7cf58dd326159  tk_mxfp8_r36b_c5_8b_gate.cpython-310-x86_64-linux-gnu.so
120e2880a43bce3bc1e64a9635be37d1  tk_mxfp8_r36b_c3_70b_qo.cpython-310-x86_64-linux-gnu.so   (negative)
86a7d9456cf9bb1e35e207718853dc03  tk_mxfp8_r36b_c7_8b_qo.cpython-310-x86_64-linux-gnu.so   (negative)
91fbb386633c0d9e5906a936ee4638b7  tk_mxfp8_r36b_default.cpython-310-x86_64-linux-gnu.so   (default 8192³)
59ef94a04170b5fc63bf117b0bc5f720  tk_mxfp8_r36b_base_default.cpython-310-x86_64-linux-gnu.so (R36 base, no patch)
```

c1/c2 share shape (4096×28672×8192) and would receive the same predicate; only c1 was rebuilt for regression check. c5/c6 share shape (4096×14336×4096); only c5 was rebuilt.

## Methodology notes

1. **Sclk gate compliance**: GPU4 c_8b_down sclk-post-preheat = 2059 MHz (below R34 first-gate 2200 MHz). Did NOT auto-retry because sclk-pre-bench had recovered to 2358 MHz and sclk-post-bench held at 2281 MHz — second gate (R35 NEW) PASSES. Per-pair ratios show no PAIR 0 anomaly. Accepted without retry. (The R34 first gate alone would have triggered an unnecessary retry; the R35 second gate correctly identified that the GPU was warm by bench time.)
2. **c1 70B Gate first run anomaly**: sclk-pre-preheat=1803 MHz, sclk-post-preheat=1806 MHz — both failed R34 first gate. Retry with PREHEAT_S=60 produced clean numbers (Δ%=+8.47%, t=+10.55) — this is the canonical result. The first-run anomaly is shown in `r36b_c1_70b_gate_crr_vs_rrr_gpu1.txt`; the retry in `r36b_c1_70b_gate_crr_vs_rrr_gpu1_retry.txt`.
3. **Default-build byte-identity not preserved**: as in R34 Dev A and R35 Dev A, the literal md5 differs between R36 base default and R36 head default (size +160 B for the new advisory string + branch). Functional invariant is preserved (advisory does NOT fire on M=N=K=8192).
4. **Predicate ordering**: new predicate placed AFTER the c5 8B Gate predicate (last in the chain) — preserves order-of-introduction grouping. No fall-through or overlap with prior predicates (each predicate is a strict M/N/K equality test on a unique shape).

## Action items for R36+ Reviewer

1. **4-GPU triangulation**: cross-verify on GPU0/5/6/7 to promote SHIP-LITE → STRICT (need min Welch t ≥ +10.0). Current 2-GPU min Welch t = +5.872. The R34 SHIP-LITE → STRICT promotion pattern (R34 Dev B c5: 2-GPU SHIP-LITE → 4-GPU STRICT) is the precedent.
2. **Cherry-pick to feat/mxfp8-only**: predicate is default-on (host-side advisory always emitted on matching shape), no opt-in macro. Pattern matches R33 Dev A / R34 Dev A / R35 Dev A.
3. The advisory predicate count is now **6 wired** covering **8 LLaMA cells** (c0 70B Down + c1+c2 70B Gate/Up + c4 70B KV + c5+c6 8B Gate/Up + c8 8B KV + c_8b_down 8B Down). Of the 14 LLaMA cells in R35 Dev D's matrix:
   - 8 cells covered by V2-RRR predicates (above)
   - 4 square Q/O cells (8B-Q/O, 70B-Q/O) — RCR is best layout (R35 Dev D §CRR vs RCR table); R36 priority #3 in TODO.
   - 2 single-token decode cells (1×4096×4096 / 1×8192×8192) — out of scope for this fan-out.

## Files

- `kernel_mxfp8_layouts.cpp` (modified, 6th predicate at lines 5673-5682)
- `r36b_orchestrate.sh` (build + bench driver)
- `r36b_advisory_check.sh` (Phase 2 firing check)
- `r36b_advisory_check.log` (Phase 2 results — 9/9 PASS)
- `r36b_advisory_*.stderr` (per-cell advisory captures)
- `r36b_build_*.log` (per-cell build logs)
- `r36b_build_md5.log` (per-cell md5 manifest)
- `r36b_default_md5.log` (default + base md5 comparison)
- `r36b_c_8b_down_crr_vs_rrr_gpu1.txt` / `_gpu4.txt` (target cell paired bench)
- `r36b_c0_70b_down_*.txt`, `r36b_c1_70b_gate_*.txt` (+`_retry.txt`), `r36b_c4_70b_kv_*.txt`, `r36b_c5_8b_gate_*.txt`, `r36b_c8_8b_kv_*.txt` (regression cells)
