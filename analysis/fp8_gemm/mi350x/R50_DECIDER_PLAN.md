# R50 Decider Plan (2026-04-19)

## Round context
- **Baseline**: R44 35/42 VC @ `305fe79d` on branch `mxfp4`. Unchanged through R45-R49.
- **Last 5 of 6 rounds DEAD** (R45 cohort tail-draw; R46 triple-dead; R47 triple-dead; R48 quadruple-dead; R49 quintuple-dead). Compiler-driven optimization frontier on this kernel is exhausted.
- **R50 mandate**: pursue 3 non-overlapping non-compiler-driven axes. Two are correctness attacks at the ISA layer (Opt A intrusive rewrite, Opt D `.co` dlopen); one is a perf-axis pivot decoupled from correctness (Opt C).

## Hard rules (carried from R45-R49 protocol)
- **10-run @ 80% gate is mandatory** for any correctness PROMOTE: n_OK_5>=8 AND wcf_max<0.02 AND wcf_std<0.01 AND fin_min>=FINITE_GATE.
- **INDEPENDENT seeds**: [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010].
- **FINITE_GATE**: 0.97 (R44D promoted; do NOT relax further).
- **Phase-1 Jaccard probe** recommended as prefilter.
- **GPU isolation**: HIP_VISIBLE_DEVICES per worker; check `rocm-smi` first.
- **No commit on DEAD round**.

## R50 worker assignments (3 parallel + 1 reviewer)

### R50 Opt A — Aiter MFMA↔ds_read 1:3 interleaving port INSIDE `kpair_64mfma_step34`
**GPUs**: 0, 1
**Goal**: Per `project_mxfp4_R49A_aiter_vmcnt_dead.md`, aiter's true cluster-B differentiator is the MFMA→ds_read 1:3 fan-out (1 MFMA → 3 ds_reads) vs HipKittens 4 MFMAs → 1 ds_read batch. R50 Opt A is the FOCUSED port of just this interleaving pattern (no other macros) into `kpair_64mfma_step34` asm volatile. The single substantive ISA difference identified in R49A's disasm.

**Mechanism focus**:
- Read `R49A_aiter_256x256.s` lines around the K-loop body to extract aiter's exact MFMA + ds_read interleaving.
- Read `R49A_KERNEL_ISA.s` to compare HipKittens' current 4:1 batch.
- Rewrite ONLY the MFMA + ds_read sequencing inside `kpair_64mfma_step34` — do NOT touch slot rotation, M0 discipline, or vmcnt levels.
- Compiler RA/scheduling runs before asm boundary insertion (per `project_mxfp4_R48A_split_step34_dead.md`), so the rewrite must replace the WHOLE instruction stream of the asm volatile, not split or rearrange.

**Concrete actions**:
1. Read `R49A_aiter_256x256.s` (3415 lines) — locate the K-loop body (long sequence of `v_mfma_scale_f32_16x16x128_f8f6f4` + `ds_read_b128`).
2. Extract aiter's per-MFMA sub-pattern: count exact instructions between consecutive MFMAs in the steady-state K-loop body.
3. Locate `kpair_64mfma_step34` in `kernel_mxfp4_gluon_cpp.cpp` — copy its current asm volatile body to `R50A_old_step34.s` for diff baseline.
4. Construct a NEW asm volatile body that emits 64 MFMAs interleaved 1:3 with ds_reads (so 64 MFMAs + ~192 ds_reads in the same total length). Mind that aiter's ds_reads are reading the NEXT K-pair's data; HipKittens currently batches them at the end of step3/step4. The 1:3 spread shifts the AGPR clobber window.
5. Add macro `R50A_AITER_INTERLEAVE` (default OFF). When enabled, replace `kpair_64mfma_step34`'s asm volatile body with the 1:3 variant.
6. Build for 6 cluster-B target shapes + 1 R44 VC stretch baseline (`4096x4096x8192`) + 4 random R44 VC shapes (regression check). Use R44 manifest for parent build params.
7. ISA verify: dump `.so` and confirm 1:3 fan-out is emitted.
8. SMOKE on 1 seed first. Phase-1 Jaccard prefilter (5-probe INPUT_REUSE=True). 10-run @ 80% gate.

**Targets** (PROMOTE candidates):
- 6 cluster-B wcf-flake: 16384x14336x4096, 16384x28672x4096, 16384x6144x4096, 28672x4096x16384, 32768x4096x14336, 4096x32768x14336

**Stretch baseline** (must NOT regress): 4096x4096x8192

**Output files** (mandatory):
- `R50_OPT_A_VERDICT.md`
- `R50A_INTEGRATION_FRAGMENT.json` (`{}` if dead)
- `R50A_aiter_kloop_body.s` (extracted aiter K-loop body)
- `R50A_old_step34.s` (HipKittens baseline body for diff)
- `R50A_new_step34.s` (your 1:3 rewrite)
- `R50A_KERNEL_ISA.s` (post-build ISA verification)
- `R50_OPT_A_{SMOKE,JACCARD,10RUN}.{json,log}`
- `R50A_BUILD_MANIFEST.json`, `build_R50A.py`, `bench_R50A.py`
- `build_R50A/*.so`

### R50 Opt C — Perf claw-back on R44 VC shapes <90% comp (decoupled from correctness)
**GPUs**: 2, 3
**Goal**: Skip-gate the unsolvable `(4096,32768,28672)` CRASH and pivot to incremental TFLOPs gains on the 18 R44 VC shapes that sit <90% comp. This is a perf-axis round, decoupled from the correctness work — a different attack surface entirely.

**Mechanism focus**:
- The 18 sub-90% VC shapes have ROOM to grow. The R25-F/G `pfoff` mechanism (`project_mxfp4_R25FG_pfoff_mechanism.md`) found +13-21% per shape on K-bound shapes via tail-prefetch elimination. Some of those gains may not be in the current R44 manifest.
- Variant table is exhausted (R43 Opt C closed) but per-shape `pfoff` and `lgk` tuning sweeps NOT done since R25.
- Try: `pfoff` micro-sweep ±4 around current value, `lgk` ∈ {1,2,3}, `gm` ∈ {6,7,8} ONLY for shapes <90% comp. NO macro changes — pure variant-knob retuning.

**Concrete actions**:
1. Read `R44_INTEGRATION_MANIFEST.json` and identify the 18 VC shapes <90% comp (use bench data from R44/R47 to get pct_comp per shape).
2. For each shape, read its R44 manifest entry: `gm, ts, lgk, v, pfoff`. Build a 27-cell sweep (3 lgk × 3 gm × 3 pfoff offsets {curr-4, curr, curr+4}).
3. Use existing `bench_round23_optB.py` style harness (or adapt R47 reviewer harness) for batch bench.
4. SMOKE all 18 shapes × ~9-27 cells = ~162-486 cells. 200 warmup / 500 iters per cell, trimmed mean.
5. PROMOTE criterion: cell improves pct_comp by >=2.0% AND retains VC under 5-run @ 80% gate (this is a perf round, not correctness — VC retention check needed but 5-run is sufficient for perf retuning since VC was already established at R44).
6. Cross-validate: each PROMOTE cell must NOT regress under 10-run @ 80% gate (no cohort race introduced).

**Targets**: 18 R44 VC shapes <90% comp. Promote any with >=2% pct_comp gain.

**Stretch baseline** (must NOT regress): all 35 R44 VC shapes retain VC at 10-run.

**Output files** (mandatory):
- `R50_OPT_C_VERDICT.md`
- `R50C_INTEGRATION_FRAGMENT.json` (`{}` or per-shape promotes)
- `R50C_baseline_pct_comp.json` (R44 baseline per-shape pct_comp)
- `R50C_SWEEP.{json,log}` (full sweep results)
- `R50C_PROMOTES_10RUN.{json,log}` (10-run cross-val of promote candidates)
- `R50C_BUILD_MANIFEST.json`, `build_R50C.py`, `bench_R50C.py`
- `build_R50C/*.so`

### R50 Opt D — Aiter `.co` dlopen for `(4096,32768,28672)` (last CRASH escape hatch)
**GPUs**: 4, 5
**Goal**: Per `project_mxfp4_R49B_n32k_28k_cohort_scales.md` and `project_mxfp4_R49C_embedded_vmcnt_dead.md`, the last CRASH cell `(4096,32768,28672)` cannot be macro-fixed. R50 Opt D side-steps the kernel rewrite entirely by binding to aiter's hand-written `.co` for that one shape via `hipModuleLoadData`. If successful, gives +1 VC (35→36) without any kernel change.

**Mechanism focus**:
- Aiter `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` is at `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/`. Same MFMA shape and tile geometry as HipKittens 256×256.
- Use `hipModuleLoadData` (or `hipModuleLoad` from the `.co` path) → `hipModuleGetFunction` → `hipModuleLaunchKernel`.
- Need to figure out aiter's kernel signature (function name in the `.co`, argument layout).
- Only used as a fallback for `(4096,32768,28672)` — all other shapes use HipKittens kernel.

**Concrete actions**:
1. Inspect aiter `.co` symbols: `llvm-readelf --syms /shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co | grep FUNC` to find kernel function name.
2. Find aiter's kernel call signature (from aiter source if available, or by inspecting `.csv` config file in the same dir).
3. Write a small C++ shim: `R50D_aiter_dlopen.cpp` that loads the `.co`, gets the kernel function, and provides a `aiter_call(A, B, A_scale, B_scale, C, M, N, K)` callable.
4. Wire the shim into the bench harness as a per-shape fallback: if `(M,N,K) == (4096,32768,28672)`, dispatch via aiter shim; else use HipKittens.
5. SMOKE: verify aiter shim runs to completion and produces non-NaN output.
6. Compare aiter's output vs torch reference: SNR computation, wcf_max check.
7. 10-run @ 80% gate on the shim-dispatched cell.
8. Verify aiter's TFLOPs measurement; record perf baseline.

**Targets**: PROMOTE only `(4096,32768,28672)` if aiter shim produces VC under 10-run @ 80% gate.

**Stretch baseline** (must NOT regress): the existing 35 R44 VC shapes use HipKittens, not aiter (so no impact).

**Output files** (mandatory):
- `R50_OPT_D_VERDICT.md`
- `R50D_INTEGRATION_FRAGMENT.json` (`{}` or `{"4096x32768x28672": {...aiter dispatch...}}`)
- `R50D_aiter_symbols.txt` (`.co` symbol dump)
- `R50D_aiter_dlopen.cpp` (shim implementation)
- `R50D_aiter_csv_audit.md` (analysis of aiter `.csv` for the kernel)
- `R50_OPT_D_SMOKE.{json,log}`, `R50_OPT_D_10RUN.{json,log}`
- `R50D_BUILD_MANIFEST.json`, `build_R50D.py`, `bench_R50D.py`
- `build_R50D/*.so`

### R50 Reviewer — 10-run INDEPENDENT-seed integration
**GPUs**: 6, 7
**Goal**: After all 3 workers complete, integrate non-empty fragments into `R50_INTEGRATION_MANIFEST.json` (R44 baseline + worker overrides). Run 10-run @ 80% INDEPENDENT-seed gate. Decide PROMOTE / NO_PROMOTE per cell. Cross-validate against the 35 R44 VC list.

**Skip-gate trigger**: if all 3 worker fragments empty `{}`, skip integration; write `R50_INTEGRATION_VERDICT.md` documenting skip.

**Output files**:
- `R50_INTEGRATION_VERDICT.md`
- `R50_INTEGRATION_MANIFEST.json`
- `R50_INTEGRATION_10RUN.{json,log}` (if integration ran)
- `bench_all_42_R50_INTEGRATION.py`

## Worker dispatch order
- All 3 workers launch IN PARALLEL (different GPUs, independent build dirs).
- Reviewer launches AFTER all 3 worker verdicts arrive.
- Decider (this conversation) coordinates dispatch + final memory writes + commit decision.

## Commit policy (per autonomous directive)
- Commit if integration shows >0 net VC delta vs R44 35/42 OR >0 net pct_comp gain on the 35 VC shapes.
- Skip commit on DEAD round.
- All R50 macros must default OFF; only manifest controls per-shape activation.
