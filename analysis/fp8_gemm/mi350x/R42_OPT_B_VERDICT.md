# R42 Opt B — Verdict: PARTIAL (CRASH → FLAKY_WRONG_OUTPUT, no clean fix)

**Targets**: 2 K=28672 CRASH shapes (5/5 FAIL_CRASH in R41 integration):
- `(M=4096,  N=32768, K=28672)` — competitor 5568.2 TFLOPS
- `(M=16384, N=4096,  K=28672)` — competitor 5525.3 TFLOPS

**Mechanism**: HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION (rc=-6). Phase 1 + Phase 2 sweeps
localize the CRASH to the **conjunction `FUSED_STEP34=1 + TAIL_SPLIT=1`** at K_DIM=28672
(k_byte_iters=112). Disabling EITHER knob alone eliminates the aperture violation but
exposes the original 17%-bf16-overflow correctness bug.

## Phase 1 — REFUTED (R41A fence + tail-pf-off=0)
Both proposed Phase-1 fixes failed:
- `fence1_po104` (R41A_EXTRACT_TILE_FENCE=1): CRASH 3/3
- `fence0_po0` (R25C_TAIL_PF_OFF_ITERS=0): CRASH 3/3
- `fence1_po0` (combined): CRASH 3/3
- `fence0_po104` (control): CRASH 3/3

The R41A fence guard activates at K>=16384, so it DID fire at K=28672 — but the K=28672
CRASH is NOT the same VMEM-vs-LDS race that the fence fixes for the cluster-C K=32768
shapes. The CRASH mechanism is different.
Artifacts: `R42_OPT_B_PHASE1_FENCE.{json,log}`, `build_R42B/`, `R42B_BUILD_MANIFEST.json`.

## Phase 2A — Knob isolation (7-cell sweep, 1-run smoke)
Strip one knob at a time from the parent variant `ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all`
+ FUSED_STEP34=1:

| cell           | (4096,32768,28672) | (16384,4096,28672) |
|----------------|--------------------|--------------------|
| baseline       | CRASH              | CRASH              |
| no_fused       | WRONG (fin=0.61, wcf=0.27) | WRONG (fin=0.67, wcf=0.18) |
| no_btw         | CRASH              | CRASH              |
| gm1            | CRASH              | CRASH              |
| no_lgk2        | CRASH              | CRASH              |
| no_tailsplit   | WRONG (fin=0.53, wcf=0.09) | WRONG (fin=0.52, wcf=0.06) |
| no_r25c        | CRASH              | CRASH              |

Conclusion: **the CRASH is gone iff `FUSED_STEP34=1` AND `TAIL_SPLIT=1` are NOT both set**.
GROUP_SIZE_M, BARRIER_TO_WAITCNT_ALL, STEP12_BR_LGKMCNT, R25C_TAIL_PF_OFF_ITERS are all
innocent. Artifacts: `R42_OPT_B_PHASE2_SWEEP1.{json,log}`, `build_R42B_phase2/`.

## Phase 2B — Try R37_FIX_B + R38B path (no FUSED_STEP34)
Replace FUSED_STEP34 with R37_FIX_B (default) plus R38B/R38F/R39A correctness rescues.
1-run smoke results for (4096,32768,28672) / (16384,4096,28672):

| cell             | shape A                          | shape B                          |
|------------------|----------------------------------|----------------------------------|
| nf_R38B          | **PASS 4126 TFLOPS** (74.1%)     | WRONG (fin=0.984, wcf=0)         |
| nf_R38F2         | WRONG (snr=-12.3)                | WRONG (snr=-12.1)                |
| nf_R38B_R38F2    | WRONG (wcf=0.042)                | WRONG (wcf=0.014)                |
| nf_R38B_R39A     | WRONG (snr=-4.2)                 | WRONG (snr=-2.2)                 |
| nf_R38B_R38F4    | WRONG (snr=-0.1)                 | WRONG (wcf=0.007 fin=0.988)      |

**`nf_R38B` is the only correct cell in 1-run smoke**, but 3-run consensus shows it FLAKES:
- (4096,32768,28672): PASS 1/3 (4124 TFLOPS), WRONG 2/3 (wcf 0.017–0.027 jitters past gate)
- (16384,4096,28672): PASS 0/3 (fin 0.964–0.987 stays just under 0.99 gate, wcf=0)

**No cell achieves stable PASS** under R39B random-scale gate. CRASH is gone, but
correctness isn't won. Best per-shape so far is ~74% of competitor, vs target 100%+.
Artifacts: `R42_OPT_B_PHASE2B_SWEEP1.{json,log}`, `R42_OPT_B_PHASE2B_NFR38B_3RUN.{json,log}`,
`build_R42B_phase2b/`, `R42B_PHASE2B_BUILD_MANIFEST.json`.

## Verdict & implications
- **CRASH root-caused** to `FUSED_STEP34=1 + TAIL_SPLIT=1` interaction at K=28672. This is a
  SEPARATE mechanism from the R41A K=32768 vmcnt fence (different K and different code
  path). The R41A "broader gating" hypothesis (R42 Opt C) is REFUTED for K=28672 — fence
  alone doesn't help.
- **No clean fix delivered** for these 2 shapes within R42 Opt B's scope. The closest
  candidate is `nf_R38B` with the R40B parent variant minus FUSED_STEP34 — it removes the
  CRASH but FLAKES on the random-scale gate (best 4126 TFLOPS = 74% of comp).
- These 2 shapes remain **FAIL** in the integrated leaderboard.

## .so file paths (for integration manifest, if anyone wants to ship the partial fix)
- `build_R42B_phase2b/tk_mxfp4_gluon_cpp_n32768_k28672_ts_lgk2_gm7_pfoff104_kx28672_btw_all_R42Bp2b_nf_R38B.cpython-310-x86_64-linux-gnu.so`
- `build_R42B_phase2b/tk_mxfp4_gluon_cpp_n4096_k28672_ts_lgk2_gm7_pfoff104_kx28672_btw_all_R42Bp2b_nf_R38B.cpython-310-x86_64-linux-gnu.so`

## Recommended follow-ups (out of R42B scope)
1. Investigate the FUSED_STEP34+TAIL_SPLIT inner loop at line 3215 (kernel_mxfp4_gluon_cpp.cpp);
   the unconditional `emit_pf_tail<0>(pf_a0_p, pf_a1_p)` at line 3222 issues prefetches at
   pf_bt = k_byte_iters - 1 with no R25C gate — investigate whether the per-iter `make_pf_params`
   construction emits an OOB voff at the K_DIM=28672 specifically (k_byte_iters=112 boundary).
2. Try a NEW macro `R42B_FUSED_TAILSPLIT_CRASH_FIX` that selectively adds R38B-style "build
   pf_*_p AFTER step34" to the FUSED_STEP34 path (mirroring the R37 fix-B branch's R38B).
3. Strip `BARRIER_TO_WAITCNT_ALL=1` AND switch to `nf_R38B` together — the BTW rewrite
   may interact with R38B's pf-emit ordering.
