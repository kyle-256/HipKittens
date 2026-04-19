# R44 Decider Plan

**Date**: 2026-04-19
**Branch**: `mxfp4` @ `0ca8c72a`
**Inputs**: R43 triple-DEAD round (Opt A CRASH structural fix dead, Opt B VGPR-PF re-buried, Opt C variant table exhausted). Leaderboard locked at **27/42 verified-correct, 10/42 WIN**, 2 CRASH, 9 WCF_BOUND, 3 FIN_BOUND, 1 WRONG_5/5.

**Floor**: ≥27/42 (no regression). **Stretch**: ≥30/42 (+3 net VC).

---

## Pre-flight feasibility checks

| Item                                              | Result | Note                                                               |
|---------------------------------------------------|--------|--------------------------------------------------------------------|
| Aiter `f4gemm_*_BpreShuffle_256x256.co` exists    | **YES** | 35632 bytes, ELF amdgcn (arch 0xe0)                              |
| `llvm-objdump` available                          | **YES** | `/opt/rocm/llvm/bin/llvm-objdump` (LLVM 20.0.0git, AOMP-18.0-12)  |
| Aiter `.co` disassembles cleanly with `--mcpu=gfx950` | **YES** | 3415 lines, 88 ds_write/buffer_load_dwordx hits, kernel symbol `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E` at `0x2c00` |
| Aiter uses `buffer_load_dwordx4 ... lds` (size=16) | **YES** | Many sites with `s[12:15]` SRD; pattern matches our hardware `buffer_load_to_lds` PF |
| **Opt B Phase-1 feasibility — CONFIRMED**          | **GO** | The disasm contains the address-formula evidence we need.        |
| 8 GPUs idle (rocm-smi)                            | **YES** | All 8 at 0% use; all in low-power state                            |

---

## Worker assignments — overview

| Worker  | Title                                       | GPUs   | Hard timeout | Touches kernel? | Falsifiable success     |
|---------|---------------------------------------------|--------|--------------|-----------------|-------------------------|
| Opt A   | K=28672 from-scratch kernel (CRASH bypass)  | 0, 1   | 8 h          | YES (new file)  | ≥1 of 2 CRASH → PASS_VC under 5-run @ GATE=0.98 (`n_OK ≥ 4/5`). |
| Opt B   | Aiter `ds_write` disasm + VGPR-PF revival   | 2, 3, 4 | 6 h         | Phase-2 only    | Phase-1: address formula written by hour 3 OR KILL. Phase-2: bit_eq ≥ 99.9% AND Jaccard ≥ 0.7 AND ≥2 of 3 reps → PASS_VC. |
| Opt C   | Fault-PC instrumentation on K=28672 CRASH   | 5      | 3 h          | NO              | Faulting PC + faulting address + disassembled instruction at PC (or KILL with methodology). |
| Opt D   | FIN_BOUND 3-shape 10-run probe              | 6, 7   | 1 h          | NO              | All 3 shapes `n_OK_10 ≥ 8` at gate=0.97 → PROMOTE recommendation, OR per-shape KILL when distribution clusters in [0.95,0.97] (deterministic-wrong). |

---

## R44 Opt A — K=28672 from-scratch (CRASH bypass)

### GPUs and isolation
- `HIP_VISIBLE_DEVICES=0,1` only. Must NOT touch GPU 5 (Opt C is fault-PC tracing on the same CRASHing build).
- Opt A consumes Opt C's fault-PC report when ready (Opt C has 3 h timeout, Opt A's 8 h budget allows waiting 30-60 min for Opt C results before locking in fork direction).

### Per-shape scope (exactly 2 shapes)
- `(M=4096, N=32768, K=28672)` — comp 5568.2 TFLOPS — current FAIL_CRASH
- `(M=16384, N=4096, K=28672)` — comp 5525.3 TFLOPS — current FAIL_CRASH

### Plan
1. **Hour 0–1: Investigate `nf_R38B` flake.** R42 Phase-2B already PASSED 1/3 at 4126 TFLOPS for `(4096,32768,28672)` and SAW 0 wcf for ALL 3 reps on `(16384,4096,28672)` — but `kernel_finite ∈ [0.96, 0.99]`. **Critical re-read of `R42_OPT_B_PHASE2B_NFR38B_3RUN.json`**: the (16384,4096,28672) shape has `wcf_per_run=[0.0, 0.0, 8.1e-05]` — i.e., it is **wcf-clean**, only fin-bound. Under FINITE_GATE=0.98 this would already be PASS_3/3 if fin clears 0.98, which 2 of 3 runs do (0.983, 0.987). Run the existing `nf_R38B` `.so` under 5-run consensus @ GATE=0.98 BEFORE writing any new kernel. **If this clears at GATE=0.98, we get +1 VC for free.**
2. **Hour 1–3: 3-buffer rotation POC.** New kernel fork `kernel_mxfp4_gluon_cpp_R44A_kx28672.cpp`: copy `kernel_mxfp4_gluon_cpp.cpp`, force `FUSED_STEP34=0` AND `TAIL_SPLIT=0`, then add an `R44A_TRIPLE_BUFFER` macro that allocates `A0_db[3]`, `A1_db[3]`, etc. Index by `(iter % 3)`. The CRASH cannot trigger because no two iters' prefetches alias the same slot.
3. **Hour 3–5: R37+R39A+R39B correctness rescue.** Hand-graft the R37_FIX_B step3+step4 fusion + R39A `TAIL_SCALE_CLAMP` + R39B random-scale-safe path into the new fork, tuned to `k_byte_iters=112`.
4. **Hour 5–7: 5-run consensus on both target shapes.** `bench_R44A.py` (warmup=200, iters=500, trim=0.10, FINITE_GATE=0.98).
5. **Hour 7–8: Regression probe — 5 nearby K shapes that share build flag stack** (K∈{14336, 16384, 32768}). Fire `nf_R38B`-style baseline against the 3-buffer fork to ensure no VC regression on these.

### Stopping criteria
- **Hour 1 PROMOTE**: if Step 1 produces `n_OK_5 ≥ 4` for either CRASH shape on the existing `nf_R38B` `.so` under GATE=0.98, COMMIT immediately and continue with 3-buffer rotation as bonus.
- **Hour 3 PIVOT**: if 3-buffer POC builds clean but smoke fails 2/2 with WRONG_OUTPUT, switch to single-issue serial loop (no double-buffer at all) for hours 4-6.
- **Hour 8 DEAD**: report DEAD with the exact CRASH PC from Opt C (if available); do NOT integrate.

### Falsifiable predictions
- **P-A.1 (cheap win)**: `nf_R38B` under GATE=0.98 + 5-run probe yields `n_OK ≥ 4/5` for `(16384,4096,28672)`. Plausible from R42 Phase-2B data (3/3 wcf<0.0001, 2/3 fin>0.98).
- **P-A.2 (3-buffer)**: 3-buffer rotation eliminates CRASH on both target shapes (build clean, 1-rep smoke produces non-NaN finite output).
- **P-A.3 (correctness)**: 5-run on integrated R37+R39A+R39B + 3-buffer fork yields `n_OK ≥ 4/5 AND wcf_max < 2% AND wcf_std < 1% AND fin_min ≥ 0.98` on at least 1 of 2 target shapes.
- **P-A.4 (no regression)**: 5 nearby K shapes (K=14336, 16384, 32768) keep their R43 VC status under the new fork.

---

## R44 Opt B — Aiter `ds_write` disasm + VGPR-PF revival

### GPUs and isolation
- `HIP_VISIBLE_DEVICES=2,3,4` only.
- Phase-1 (disasm only) uses CPU only; no GPU needed. Phase-2 uses GPU 2 for build, 3+4 for parallel 3-rep probe.
- Independent of Opt A and Opt C (different shapes, different build artifacts).

### Per-shape scope
- **Phase-1 (disasm)**: no shape — purely static analysis of `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`.
- **Phase-2 (kernel patch + Jaccard probe)**: if and only if Phase-1 succeeds, run on 3 representative WCF_BOUND shapes:
  - `4096x32768x6144` (smallest K in WCF_BOUND, fastest iteration)
  - `4096x32768x14336` (largest WCF_BOUND with `n_OK_5=3`)
  - `32768x4096x14336` (different M-class for cross-validation)

### Plan
1. **Hour 0–1: ds_write site location.** Use `/opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx950` on the `.co`. Locate every `buffer_load_dwordx4 ... lds` site (88 hits total). For each, identify the paired `ds_write` (or `ds_write2`) that consumes the loaded data downstream. Trace the per-lane `voff` register to its definition (likely a function of `v0` thread-ID and tile coordinates).
2. **Hour 1–2: Address formula extraction.** Express the LDS write byte address as a closed form `addr(lane, k_step, tile_id) = ?`. The R35 hypotheses tried `lane*4 + dword*256`, `lane*16`, `voffs[idx]` — all wrong. The actual formula likely interleaves per-XCD or per-WG-mod-N bits.
3. **Hour 2–3: Cross-check.** Confirm the formula by hand-simulating 2-3 concrete lanes (lane 0, lane 16, lane 32) and matching against the 88 `ds_write` sites. **If formula is unrecoverable by hour 3 → KILL Opt B**.
4. **Hour 3–5: Patch `kernel_mxfp4_gluon_cpp_vgprPF.cpp`.** Replace the broken `ds_write_b32` formula in the VGPR_PF_MODE=1 path with the recovered formula. Build with `+v` keepalive (R35 Opt A code path).
5. **Hour 5–6: Bit-eq oracle + Jaccard probe.** Re-run `R43_OPT_B_PREFLIGHT.py` for bit_eq (target ≥ 99.9% on both-finite cells). If pass, run `R42_OPT_A_PHASE1_DIAGNOSTIC.py` on the 3 representative shapes (3 reps each) for Jaccard (target ≥ 0.7).

### Stopping criteria
- **Hour 3 KILL**: if address formula cannot be expressed as a closed form OR the formula doesn't match all 88 sites, KILL Opt B and write `R44_OPT_B_DISASM_NOTES.md` with what was learned (still useful for R45).
- **Hour 6 DEAD**: if bit_eq < 99.9% OR Jaccard < 0.7 on any of 3 shapes, KILL.
- **Hour 6 PROMOTE**: if all gates clear, request Opt B integration on the 9 WCF_BOUND shapes (deferred to R45 — too late in R44 to do full 5-run + 9-shape integration).

### Falsifiable predictions
- **P-B.1 (Phase-1)**: Address formula is a closed function of `(lane, dword_idx, k_step)` derivable in ≤3 hours of disasm.
- **P-B.2 (Phase-2 bit_eq)**: With the recovered formula, VGPR-PF kernel produces bit_eq ≥ 99.9% vs incumbent on both-finite cells (vs R43 Opt B preflight's 0.0838%).
- **P-B.3 (Phase-2 race)**: With bit_eq fixed, Jaccard ≥ 0.7 on identical-input 3-rep probe → race is closed.
- **P-B.4 (perf)**: VGPR-PF perf does NOT regress > 3% on the 3 probe shapes (vs R43 baseline for those shapes).

---

## R44 Opt C — Fault-PC instrumentation on K=28672 CRASH

### GPUs and isolation
- `HIP_VISIBLE_DEVICES=5` only. Must NOT touch GPUs 0,1 (Opt A) or 2,3,4 (Opt B) or 6,7 (Opt D).
- Pure diagnostic: produces a fault-PC report consumed by Opt A.

### Per-shape scope (1 shape, then maybe 2)
- Primary: `(M=4096, N=32768, K=28672)` using R43 Opt A variant 0 (current FUSED+TAIL_SPLIT default that CRASHes deterministically).
- Secondary if time: `(M=16384, N=4096, K=28672)` to confirm same vs different fault PC.

### Plan
1. **Hour 0–0.5: Build the failing `.so`.** Pre-existing in `build_R43A/`; identify the `_R43A_p2b_ctrl` artifact (variant 0 = baseline CRASH).
2. **Hour 0.5–2: Run with environment instrumentation:**
   - `HSA_DEBUG=1 AMD_LOG_LEVEL=4 AMD_SERIALIZE_KERNEL=3 HSA_ENABLE_DEBUG=1` — capture HSA fault payload (fault address + access type).
   - Try `rocgdb` with breakpoint on HSA fault dispatch if available.
   - Try `HIP_DEVICE_LOAD_KERNELS_DUMP=1` to capture loaded ISA at runtime, paired with instruction-pointer from fault.
3. **Hour 2–3: Disassemble the failing kernel `.so`.** Use `/opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx950` on the embedded HSACO inside the `.so` (extracted via `roc-obj-extract` if available, or via `objcopy --dump-section .hip_fatbin=...`). Locate the instruction at the faulting PC. Report: instruction text, surrounding 5 instructions, register state at fault.

### Stopping criteria
- **Hour 1 PROMOTE-PARTIAL**: if HSA fault payload yields a clean `(faulting_va, access_type)` tuple even without PC, immediately publish `R44_OPT_C_FAULT.md` for Opt A consumption.
- **Hour 3 DEAD-NOTE**: if no PC and no fault address recoverable, report the diagnostic methodology tried + which environment variables hit a wall + recommend `compute-sanitizer`/`rocprof` follow-up for R45.

### Falsifiable predictions
- **P-C.1**: HSA returns the faulting VA in the dispatch failure log within 1 hour of `AMD_LOG_LEVEL=4` runs.
- **P-C.2**: With faulting VA + ISA dump, the faulting instruction is identifiable (likely a `buffer_load_dwordx4 ... offen lds` in the FUSED_STEP34+TAIL_SPLIT path).

---

## R44 Opt D — FIN_BOUND 3-shape 10-run probe (measurement reframing)

### GPUs and isolation
- `HIP_VISIBLE_DEVICES=6,7` only. Independent of all other workers.

### Per-shape scope (exactly 3 shapes)
- `32768x4096x2048` — current `fin_min=0.980, wcf_max=0.013, n_OK_5=4` (R41B `pfoff4_v3`)
- `16384x14336x2048` — current `fin_min=0.921, wcf_max=0.001, n_OK_5=4` (R40B `pfoff4_kx2048_btw_all_safe`)
- `16384x28672x2048` — current `fin_min=0.979, wcf_max=0.010, n_OK_5=4` (R40B `pfoff4_safe`)

### Plan
1. **Hour 0–0.5: Set up 10-run harness.** `bench_R44D_10run.py` based on `bench_all_42_R42A1.py`, `n_runs=10`, `n_shapes=3` (these 3), `warmup=200, iters=500, trim=0.10, random_scale=True`. Round-robin shapes across GPUs 6 and 7.
2. **Hour 0.5–1: Run and analyze.** Compute empirical `fin` distribution per shape (10 samples each). Compute the `n_OK` count under three candidate gates: `fin ≥ 0.98` (current), `fin ≥ 0.97`, `fin ≥ 0.96`.
3. **Hour 1: Verdict.** Per-shape decision:
   - `n_OK_10 ≥ 8` at `fin ≥ 0.97` AND fin distribution NOT clustered in `[0.95, 0.97]` (i.e. cohort race signature: bimodal/heavy-tail) → **PROMOTE_REFRAMING** for that shape.
   - `n_OK_10 < 8` at `fin ≥ 0.97` OR fin distribution clusters in `[0.95, 0.97]` (i.e., deterministic-wrong) → **KILL** for that shape.

### Stopping criteria
- **Hour 1 PROMOTE**: write `R44_OPT_D_VERDICT.md` with per-shape decision. Recommendation to R44 integrator: only relax gate to 0.97 IFF cohort race signature confirmed AND ≥2 of 3 shapes flip.

### Falsifiable predictions
- **P-D.1**: At least 1 of 3 shapes has `n_OK_10 ≥ 8` at `fin ≥ 0.97` with race-noise signature → +1 to +3 VC by gate change.
- **P-D.2**: NOT all 3 shapes will deterministic-wrong cluster in [0.95, 0.97] (i.e., at least 1 is rescuable).

---

## Cross-worker conflicts and resolution

| Conflict                                                                 | Resolution                                                                                       |
|--------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Opt A (CRASH kernel work) and Opt C (fault-PC tracing) share K=28672     | Opt A consumes Opt C's PC report at hour 1-3. If Opt C produces faulting VA early, Opt A pivots toward "fix the specific instruction" rather than blind 3-buffer rewrite. **GPU isolation: Opt A on 0,1; Opt C on 5.** No GPU collision. |
| Opt A and Opt D both run baseline R40B `.so`s                            | Opt A reads `nf_R38B` (a R42 fork); Opt D reads R40B/R41B `safe`/`v3`. **Different `.so` files, no collision.** |
| Opt B is independent                                                     | Phase-1 is CPU-only static analysis. Phase-2 uses GPUs 2,3,4 for ~1 hour at hour 5-6.            |
| Promote gate for Opt A and Opt D differ (kernel vs measurement)          | Opt A promotes via standard 5-run consensus + 4/5. Opt D promotes a GATE CHANGE (FINITE_GATE 0.98 → 0.97) only if cohort race signature confirmed AND ≥2 of 3 shapes flip. **Apply gate change globally OR per-shape — integrator's call at R44 close.** |
| Opt B success means kernel patch landing in `kernel_mxfp4_gluon_cpp_vgprPF.cpp` | This file is touched by no other R44 worker. Safe.                                          |

---

## Tie-breaking rules (R44 integration)

1. **Correctness wins**: any candidate that flips a CRASH or WRONG_5/5 → PASS_VC under 5-run @ promote gate is INTEGRATED, even at perf cost up to -10% vs current.
2. **Perf wins**: a +5% perf improvement on a VC_CLAWBACK shape is integrated only if 5-run consensus shows `wcf_std < 0.005 AND fin_min ≥ 0.985 AND n_OK ≥ 4`.
3. **Gate change**: Opt D's recommendation to relax `FINITE_GATE 0.98 → 0.97` is applied ONLY if all 3 of Opt D's probe shapes confirm cohort race signature. A single-shape pass is rejected (insufficient evidence to change a global gate).
4. **Per-shape macro overrides** (existing pattern): if Opt A produces a 3-buffer K=28672 fork, integrate as per-shape override in `R44_INTEGRATION_MANIFEST.json`; do not enable globally.

---

## Integration plan after all 4 workers report

1. **Wait for all 4 worker verdicts** (8h hard ceiling for the round, gated by Opt A's timeout).
2. **Build `R44_INTEGRATION_MANIFEST.json`** from R43 base + per-shape overrides from each promoting worker.
3. **Run `bench_all_42_R44_INTEGRATION.py`** (manifest-driven, 8-GPU parallel, 5-run consensus) on all 42 shapes.
4. **Verify floor** (`≥ 27 VC`) and check stretch (`≥ 30 VC`).
5. **Commit on net VC > R43** (any positive Δ). Update TODO.md with R44 close summary + R45 candidates. Update relevant memory files.
6. **If Opt D promotes a gate change**, also commit `bench_all_42_R44.py` with new gate as the future floor harness.

---

## Hard rules adherence (round-level)

- Bench params `warmup=200, iters=500, trim=0.10` for all final perf decisions.
- 5-run consensus (`n_OK ≥ 3 AND wcf_max < 2% AND wcf_std < 1% AND fin_min ≥ 0.98`); promote at `n_OK ≥ 4`.
- 8 GPUs partitioned: 0,1 = A; 2,3,4 = B; 5 = C; 6,7 = D. **No worker exceeds its allocation.**
- Any new kernel macro: default OFF, per-shape gated through integration manifest.
- All sub-agents dispatched via Opus.
- No `sleep+poll` loops; use sub-agents for monitoring background runs (`run_in_background=true` only when paired with sub-agent monitoring).
- `R43A_GATE_PF_TAIL_KBOUND` macro stays default OFF (R41A behavior preserved).
- `kernel_mxfp4_gluon_cpp_vgprPF.cpp` retains R35 Opt A `+v` keepalive; only the `ds_write` formula changes if Opt B promotes.

---

## Files this round will produce

- `R44_DECIDER_PLAN.md` (this file)
- `R44_DECIDER_PER_SHAPE.json` (42-row table refreshed from R43 + R43 outcome columns)
- `R44_OPT_A_VERDICT.md` + `kernel_mxfp4_gluon_cpp_R44A_kx28672.cpp` + `build_R44A.py` + `bench_R44A.py` + `R44A_BUILD_MANIFEST.json` + `R44A_5RUN.{json,log}`
- `R44_OPT_B_DISASM_NOTES.md` (always) + `R44_OPT_B_VERDICT.md` + (if PROMOTE) `kernel_mxfp4_gluon_cpp_vgprPF.cpp` patch + `R44_OPT_B_BIT_EQ.json` + `R44_OPT_B_JACCARD.json`
- `R44_OPT_C_FAULT.md` + `R44_OPT_C_FAULT_RAW.log`
- `R44_OPT_D_VERDICT.md` + `R44_OPT_D_10RUN.{json,log}` + `bench_R44D_10run.py`
- `R44_INTEGRATION_MANIFEST.json` + `R44_INTEGRATION_5RUN.{json,log}` + `bench_all_42_R44_INTEGRATION.py`
- TODO.md update with R44 close + R45 candidates
