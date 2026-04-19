# R43 Opt A — Verdict: DEAD (no fix candidate flips either CRASH shape to PASS_VC)

**Date**: 2026-04-19
**Branch**: `mxfp4` @ `835a8f01`
**Targets**: 2 K=28672 CRASH shapes (5/5 FAIL_CRASH in R41 integration):
- `(M=4096,  N=32768, K=28672)` — competitor 5568.2 TFLOPS
- `(M=16384, N=4096,  K=28672)` — competitor 5525.3 TFLOPS

**Outcome**: Both shapes remain FAIL on the integrated leaderboard. Neither
A.fix1 (any of 6 sub-variants) nor A.fix2 (widen SRD num_records) produces a
verified-correct kernel under 1-rep smoke at `bench_R43A.py` (warmup=200,
iters=500, trim=0.10, FINITE_GATE=0.98 — same gate as `bench_all_42_R42A1.py`).

## Mechanism summary (carried forward from R42 Opt B)
The CRASH is `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` (rc=-6). R42 Phase-2A
localized it to the conjunction `FUSED_STEP34=1 AND TAIL_SPLIT=1` at
K_DIM=28672 (k_byte_iters=112). Stripping either knob alone removes the CRASH
but exposes the 17%-bf16-overflow correctness bug (`no_fused`/`no_tailsplit` →
fin~0.5, wcf~0.1).

The decider hypothesized the unconditional `emit_pf_tail<0>(pf_a0_p, pf_a1_p)`
+ `emit_pf_tail<0>(pf_bl_p, pf_br_p)` calls at kernel
`kernel_mxfp4_gluon_cpp.cpp:3222-3223` (post-fused-step34, no R25C gate) issue
prefetches whose voff/lds_addr clamp to `pf_bt = k_byte_iters - 1` on the LAST
steady-state iter (`bt = k_byte_iters - 2`), racing the TAIL_SPLIT tail
handler.

## What I built and tested

Macro: `R43A_GATE_PF_TAIL_KBOUND` (default OFF), value-dispatched 1..6:

| variant | mechanism                                                 | (4096,32768,28672) | (16384,4096,28672) |
|---------|-----------------------------------------------------------|--------------------|--------------------|
| 0 (ctrl)| baseline = R42 CRASH reproduction                          | CRASH              | CRASH              |
| 1 (fix1)| skip BOTH emit_pf_tail when bt==k_byte_iters-2            | CRASH              | WRONG (fin=0.59, wcf=0.10) |
| 2 (fix1b)| skip only B-half on clamped iter                          | CRASH              | WRONG (fin=0.54, wcf=0.13) |
| 3 (fix1c)| skip only A-half on clamped iter                          | WRONG (fin=0.56, wcf=0.13) | WRONG (fin=0.60, wcf=0.09) |
| 4 (fix1d)| REPLACE LDS prefetch with L2-only on clamped iter         | WRONG (fin=0.52, wcf=0.13) | WRONG (fin=0.57, wcf=0.08) |
| 5 (fix1e)| R38B-style late `make_pf_params` after step34             | CRASH              | CRASH              |
| 6 (fix1f)| `s_waitcnt vmcnt(0)` fence BEFORE emit_pf_tail            | CRASH              | CRASH              |

5-run consensus skipped — no variant cleared 1-rep smoke on EITHER shape, so
P-A.1 (`n_OK ≥ 4/5 AND tflops ≥ 3500`) is structurally unreachable.

A.fix2 (widen SRD num_records): **REJECTED at design time**. The B-tile SRD
already uses `num_records = 0xFFFFFFFFu` (full 4 GB range — see kernel line
~942 and `include/ops/warp/memory/util/util.cuh:75`). There is nothing to
widen; the aperture violation is NOT triggered by per-buffer bounds checking
but by a different (likely scheduler-/race-related) mechanism in the LDS
double-buffer state machine.

Phases 3 (regression probe) and 4 (speculative K=14336) were both skipped per
the stopping criteria — neither would surface useful data without a viable
PASS_VC candidate to integrate.

## Files
- Kernel diff (default OFF macro added): `kernel_mxfp4_gluon_cpp.cpp` — added
  `R43A_GATE_PF_TAIL_KBOUND` macro definition (default 0) at line ~87 and
  the 6-way dispatch block in the FUSED_STEP34 branch at line ~3222.
- Build script: `build_R43A.py`. Built `.so`s in `build_R43A/` (14 files; 2
  shapes × 7 cells).
- Bench harness: `bench_R43A.py` (5-run consensus capable, FINITE_GATE=0.98).
- Manifest: `R43A_BUILD_MANIFEST.json`.
- Smoke results: `R43_OPT_A_SMOKE_FIX1.{json,log}`,
  `R43_OPT_A_SMOKE_CTRL.{json,log}`, `R43_OPT_A_SMOKE_VARIANTS.{json,log}`,
  `R43_OPT_A_SMOKE_VARIANTS_EF.{json,log}`.

## Recommendation: DEFER to R44 with a different attack axis

Per stopping criteria ("DEAD: both fixes fail on both shapes after ≤4 hours"),
both R43A fix candidates fail. The kernel macro `R43A_GATE_PF_TAIL_KBOUND`
remains in `kernel_mxfp4_gluon_cpp.cpp` default OFF — it does NOT change R41A
behavior on any of the 40 non-CRASH shapes.

**Do NOT integrate** any R43A variant for these 2 shapes. Leave them in
CRASH state in `R43_INTEGRATION_MANIFEST.json` (no manifest update needed
for these 2 shapes; R41A fallback still applies but produces FAIL_CRASH).

### Suggested R44 directions (out of R43A scope)
1. **Build a 28672-specific kernel variant from scratch** without TAIL_SPLIT
   AND without FUSED_STEP34, but with a hand-written R37+R39A+R39B-style
   correctness rescue chain tuned to k_byte_iters=112. The R42 Phase-2B
   `nf_R38B` fork PASSED 1/3 at 4126 TFLOPS (74% comp) on shape A but flaked;
   investigate WHY it flaked (statistics over 50+ runs, not 3) and whether a
   tighter scale-clamp/finite-gate combination can stabilize it.
2. **Investigate the LDS double-buffer collision directly**: instrument the
   FUSED_STEP34+TAIL_SPLIT main loop with `__builtin_amdgcn_s_sleep(0)` /
   manual barrier between step34 and emit_pf_tail; if CRASH still triggers,
   the race is NOT in the issue ordering but in the in-flight buffer-load
   completion versus the next iter's barrier.
3. **Try a 3-buffer rotation** (A0_db[3], A1_db[3], etc.) so the prefetch
   target on iter N is always different from any in-flight slot. High kernel
   surgery cost but eliminates all double-buffer races by construction.
4. **Disassemble the failing build** (`kernel_mxfp4_gluon_cpp_n4096_k28672_…
   _R43A_p2b_ctrl.cpython-310-x86_64-linux-gnu.so`) and check whether the
   compiler is emitting buffer_load_to_lds instructions with computed voff
   that exceeds the per-tile addressing window. The fault PC + faulting
   address from `HSA_DEBUG=1 AMD_LOG_LEVEL=4` would localize to the exact
   instruction.

## Per-shape final state (for R43 integration manifest fragment)

```json
{
  "round": "R43_OPT_A",
  "verdict": "DEAD",
  "shapes_no_change": [
    {"shape": "4096x32768x28672",  "verdict": "FAIL_CRASH (unchanged)"},
    {"shape": "16384x4096x28672", "verdict": "FAIL_CRASH (unchanged)"}
  ],
  "macro_added": "R43A_GATE_PF_TAIL_KBOUND (default OFF, no per-shape gate enabled)"
}
```

## Hard rules compliance
- bench params: warmup=200, iters=500, trim_frac=0.10 ✓ (smoke runs only —
  no 5-run consensus reached because no smoke passed)
- GATE: FINITE_GATE=0.98 (matches R42 Opt A1 reviewer) ✓
- GPUs: HIP_VISIBLE_DEVICES=0,1 only — never touched 2..7 ✓
- Macro: default OFF, R41A behavior preserved when unset ✓
- No sleep+poll loops; build/bench were synchronous in foreground per phase ✓
