# R40 Dev C — V2-RCR/V2-RRR advisory audit through MXFP8_DISPATCH_TRACE

**Status**: COMPLETE — all 8 advisories HEALTHY (FIRES under autotune-default invocation)
**Branch**: `r40-dev-c` (worktree `/tmp/wt-r40-c`, based on `feat/mxfp8-only` @ 518efb3c)
**Touches**: `analysis/fp8_gemm/mi350x/r40c_findings.md` (this file only — audit, no source modifications)
**TODO item**: R40+ priority list #3 ("V2-RCR advisory audit through MXFP8_DISPATCH_TRACE")

## Background

R39 Dev C (`021d5a13`) deployed `MXFP8_DISPATCH_TRACE=1` runtime tracepoint
infrastructure, including 8 V2-RRR/V2-RCR advisories at the top of the
CRR-V2 dispatcher. Advisories are gated only by `trace_enabled()` and shape
predicates `(g.m, g.n, g.k)` — they fall through (no kernel swap) and exist
to inform the caller "for this shape, V2-RRR/V2-RCR layout has been measured
faster than V2-CRR; consider switching layout".

R39 Dev D (`010d64e0`) audited the 9 hard-routing predicates (8 advisories
+ 2 hbshrink kernel-swaps + 1 hbnshrink) against the source — found 0
wire-in bugs. But Dev D's audit covered ROUTING (does the predicate
evaluate true for the right shape?), not RUNTIME (does the dispatcher
actually reach the predicate when invoked through the production .so by the
default test harness?). R40 Dev C closes this gap with end-to-end runtime
tracing.

## Audit protocol

For each of the 8 LLaMA shapes covered by an advisory:

1. Build production .so with `CPPFLAGS="-DM_DIM=<m> -DN_DIM=<n> -DK_DIM=<k>"`
   (no `MXFP8_CRR_BLK_M=128` and no `MXFP8_CRR_BLK_N=128` — i.e., default
   feature flags, only shape dims overridden so the dispatcher's
   `crr_can_use_exact_8wave_scaled` shape predicate matches)
2. nm-gate verify (`r38_nm_gate.sh`): expect default-build hygiene PASS
   (all default-off features count=0; all runtime-gated features count>=1)
3. Run `MXFP8_DISPATCH_TRACE=1 MXFP8_PRESHUFFLE_QUANT=1 MXFP8_WARMUP=2
   MXFP8_ITERS=2 python3 test_mxfp8_python.py M N K 2> trace.err`
4. Grep `[mxfp8_dispatch] crr_v2:` lines from trace.err for the expected
   advisory predicate substring

Time-box: ~10 min build × 8 shapes (parallelized via single bash script,
~90 min serial), but actual wall time was ~2 min (each build ~12-15 s).

**Constraint**: NO kernel modifications permitted (audit, not wire-in).

## Per-shape results

| # | Tag         | M    | N     | K     | Expected predicate                            | FIRED? | Verdict |
|---|-------------|------|-------|-------|-----------------------------------------------|--------|---------|
| 1 | 8b_qo       | 4096 | 4096  | 4096  | `ADVISE-V2-RCR-8B-QO (R36C +5.83-7.05%)`      | **Y**  | HEALTHY |
| 2 | 70b_qo      | 4096 | 8192  | 8192  | `ADVISE-V2-RCR-70B-QO (R36C +8.20-8.32%)`     | **Y**  | HEALTHY |
| 3 | 70b_down    | 4096 | 8192  | 28672 | `ADVISE-V2-RRR-70B-DOWN (R32C +12.14%)`       | **Y**  | HEALTHY |
| 4 | 70b_gateup  | 4096 | 28672 | 8192  | `ADVISE-V2-RRR-70B-GATEUP (R33C +7.18% min)`  | **Y**  | HEALTHY |
| 5 | 70b_kv      | 4096 | 1024  | 8192  | `ADVISE-V2-RRR-70B-KV (R33C +10.24% min)`     | **Y**  | HEALTHY |
| 6 | 8b_kv       | 4096 | 1024  | 4096  | `ADVISE-V2-RRR-8B-KV (R33C +8.13% min)`       | **Y**  | HEALTHY |
| 7 | 8b_gateup   | 4096 | 14336 | 4096  | `ADVISE-V2-RRR-8B-GATEUP (R34B +5.025% min)`  | **Y**  | HEALTHY |
| 8 | 8b_down     | 4096 | 4096  | 14336 | `ADVISE-V2-RRR-8B-DOWN (R36B +9.51%)`         | **Y**  | HEALTHY |

**Verdict count: 8 HEALTHY / 0 DEAD-CODE / 0 OVERRIDDEN**

Every advisory fires through the dispatcher under the test_mxfp8_python.py
default invocation. nm-gate OVERALL=PASS for all 8 builds (build hygiene
intact; no spurious feature symbols leaked into default builds). SNR for
all 8 shapes ≥ 49.59 dB (above 48 dB correctness gate).

## Full trace excerpts

For each shape: rcr_v2 + rrr_v2 + crr_v2 (advisory + fallthrough to
default) all fire as expected. Files saved under `/tmp/wt-r40-c/r40c_audit/`.

### #1 — 8B Q/O (4096×4096×4096) — `dispatch_8b_qo.txt`
```
[mxfp8_dispatch] rcr_v2: shape=(M=4096,N=4096,K=4096) -> RCR-V2-EXACT-8WAVE
[mxfp8_dispatch] rrr_v2: shape=(M=4096,N=4096,K=4096) -> RRR-V2-EXACT-8WAVE
[mxfp8_dispatch] crr_v2: shape=(M=4096,N=4096,K=4096) -> ADVISE-V2-RCR-8B-QO (R36C +5.83-7.05%)
[mxfp8_dispatch] crr_v2: shape=(M=4096,N=4096,K=4096) -> CRR-V2-EXACT-8WAVE-DEFAULT
```

### #2 — 70B Q/O (4096×8192×8192) — `dispatch_70b_qo.txt`
```
[mxfp8_dispatch] rcr_v2: shape=(M=4096,N=8192,K=8192) -> RCR-V2-EXACT-8WAVE
[mxfp8_dispatch] rrr_v2: shape=(M=4096,N=8192,K=8192) -> RRR-V2-EXACT-8WAVE
[mxfp8_dispatch] crr_v2: shape=(M=4096,N=8192,K=8192) -> ADVISE-V2-RCR-70B-QO (R36C +8.20-8.32%)
[mxfp8_dispatch] crr_v2: shape=(M=4096,N=8192,K=8192) -> CRR-V2-EXACT-8WAVE-DEFAULT
```

### #3 — 70B Down (4096×8192×28672) — `dispatch_70b_down.txt`
```
[mxfp8_dispatch] rcr_v2: shape=(M=4096,N=8192,K=28672) -> RCR-V2-EXACT-8WAVE
[mxfp8_dispatch] rrr_v2: shape=(M=4096,N=8192,K=28672) -> RRR-V2-EXACT-8WAVE
[mxfp8_dispatch] crr_v2: shape=(M=4096,N=8192,K=28672) -> ADVISE-V2-RRR-70B-DOWN (R32C +12.14%)
[mxfp8_dispatch] crr_v2: shape=(M=4096,N=8192,K=28672) -> CRR-V2-EXACT-8WAVE-DEFAULT
```

### #4 — 70B Gate/Up (4096×28672×8192) — `dispatch_70b_gateup.txt`
```
[mxfp8_dispatch] rcr_v2: shape=(M=4096,N=28672,K=8192) -> RCR-V2-EXACT-8WAVE
[mxfp8_dispatch] rrr_v2: shape=(M=4096,N=28672,K=8192) -> RRR-V2-EXACT-8WAVE
[mxfp8_dispatch] crr_v2: shape=(M=4096,N=28672,K=8192) -> ADVISE-V2-RRR-70B-GATEUP (R33C +7.18% min)
[mxfp8_dispatch] crr_v2: shape=(M=4096,N=28672,K=8192) -> CRR-V2-EXACT-8WAVE-DEFAULT
```

### #5 — 70B KV (4096×1024×8192) — `dispatch_70b_kv.txt`
```
[mxfp8_dispatch] rcr_v2: shape=(M=4096,N=1024,K=8192) -> RCR-V2-EXACT-8WAVE
[mxfp8_dispatch] rrr_v2: shape=(M=4096,N=1024,K=8192) -> RRR-V2-EXACT-8WAVE
[mxfp8_dispatch] crr_v2: shape=(M=4096,N=1024,K=8192) -> ADVISE-V2-RRR-70B-KV (R33C +10.24% min)
[mxfp8_dispatch] crr_v2: shape=(M=4096,N=1024,K=8192) -> CRR-V2-EXACT-8WAVE-DEFAULT
```

(Note: This shape ALSO routes through `dispatch_crr_exact_8wave_scaled_v2_hbshrink`
when built with `-DMXFP8_CRR_BLK_M=128`. R39 Reviewer + R39 Dev D already
validated that path through dispatcher trace at `66ef02d8` (8B-KV) and
`ab8a80f7` (70B-KV) — see r39c_findings.md "R38 wrap fix smoke test"
section. R40 Dev C only verifies the default-feature-flag advisory path
here.)

### #6 — 8B KV (4096×1024×4096) — `dispatch_8b_kv.txt`
```
[mxfp8_dispatch] rcr_v2: shape=(M=4096,N=1024,K=4096) -> RCR-V2-EXACT-8WAVE
[mxfp8_dispatch] rrr_v2: shape=(M=4096,N=1024,K=4096) -> RRR-V2-EXACT-8WAVE
[mxfp8_dispatch] crr_v2: shape=(M=4096,N=1024,K=4096) -> ADVISE-V2-RRR-8B-KV (R33C +8.13% min)
[mxfp8_dispatch] crr_v2: shape=(M=4096,N=1024,K=4096) -> CRR-V2-EXACT-8WAVE-DEFAULT
```

(Same HB shrink B1 dual-route note as #5 — the R38 wrap fix `66ef02d8`
case has been validated cross-cycle 6/6 measurements at +24-30%; this
audit covers only the advisory path on default feature flags.)

### #7 — 8B Gate/Up (4096×14336×4096) — `dispatch_8b_gateup.txt`
```
[mxfp8_dispatch] rcr_v2: shape=(M=4096,N=14336,K=4096) -> RCR-V2-EXACT-8WAVE
[mxfp8_dispatch] rrr_v2: shape=(M=4096,N=14336,K=4096) -> RRR-V2-EXACT-8WAVE
[mxfp8_dispatch] crr_v2: shape=(M=4096,N=14336,K=4096) -> ADVISE-V2-RRR-8B-GATEUP (R34B +5.025% min)
[mxfp8_dispatch] crr_v2: shape=(M=4096,N=14336,K=4096) -> CRR-V2-EXACT-8WAVE-DEFAULT
```

### #8 — 8B Down (4096×4096×14336) — `dispatch_8b_down.txt`
```
[mxfp8_dispatch] rcr_v2: shape=(M=4096,N=4096,K=14336) -> RCR-V2-EXACT-8WAVE
[mxfp8_dispatch] rrr_v2: shape=(M=4096,N=4096,K=14336) -> RRR-V2-EXACT-8WAVE
[mxfp8_dispatch] crr_v2: shape=(M=4096,N=4096,K=14336) -> ADVISE-V2-RRR-8B-DOWN (R36B +9.51%)
[mxfp8_dispatch] crr_v2: shape=(M=4096,N=4096,K=14336) -> CRR-V2-EXACT-8WAVE-DEFAULT
```

## Methodology notes

1. **Advisory semantics clarified**: The 8 advisories are NOT autotune-
   selectors; they are pure trace messages emitted from inside the CRR
   dispatcher path. There is NO autotune in the dispatcher — the test
   harness explicitly invokes `gemm_rcr_pq_v2`, `gemm_rrr_pq_v2`, and
   `gemm_crr_pq_v2` separately (`test_mxfp8_python.py:439/467/493`),
   which dispatch into rcr_v2/rrr_v2/crr_v2 respectively. The "advisory"
   simply tells the operator: "you called CRR for this shape; if you can
   choose layout, our cycle-history says V2-RCR or V2-RRR is faster".
   The original prompt's hypothesis ("autotune may pick a different
   layout") does not apply to this dispatcher — there is no autotune
   layer. All advisories are reached for-sure when the test harness
   exercises the CRR layout for the matching shape.

2. **Why the prompt's "OVERRIDDEN-BY-HARD-ROUTE" verdict is N/A here**:
   The 2 hard-routing predicates (HB shrink B1 for 70B-KV and 8B-KV) are
   placed AFTER the advisories in the dispatcher source order
   (`kernel_mxfp8_layouts.cpp:5798-5811`). Even when both `-DMXFP8_CRR_BLK_M=128`
   AND the matching shape are present, the trace shows BOTH the advisory
   AND the kernel-swap fire — because the advisory block emits a one-shot
   trace via `record_and_check` and falls through (no `return`), then the
   hard-route catches it and emits its own trace. Verified in R39 Dev C
   findings (R38 wrap fix smoke test, lines 8B-KV / 70B-KV trace blocks
   each show 2 crr_v2 lines: ADVISE + HBSHRINK-B1).

3. **Default test harness exercises ALL 3 layouts**: With
   `MXFP8_LAYOUTS=rcr,rrr,crr` (default), the trace records 3 separate
   dispatch entries per shape (one per layout). The advisory only fires
   when the CRR layout is dispatched; the rcr_v2 and rrr_v2 dispatches
   fire `RCR-V2-EXACT-8WAVE` / `RRR-V2-EXACT-8WAVE` (no advisory).

4. **Build hygiene**: All 8 builds passed `r38_nm_gate.sh`. The runtime-
   gated kernel-swap features (hbshrink, hbn, etc.) had count=0 in every
   default-feature-flag build, confirming no compile-flag pollution.

5. **PRESHUFFLE_QUANT path coverage**: All 8 traces ran with
   `MXFP8_PRESHUFFLE_QUANT=1` (the production path used by R36+ formal
   benches). The non-PRESHUFFLE path (`MXFP8_PRESHUFFLE_QUANT=0`) takes
   the V1 dispatcher entry point (`gemm_rcr/rrr/crr` without `_pq_v2`)
   and would emit the `V1-PQ-DEFAULT` tracepoint — out of scope for this
   audit (advisories live in V2 dispatcher only).

## R41+ recommendations

**No source changes required.** All 8 advisories fire correctly. No
DEAD-CODE found, no OVERRIDDEN-BY-HARD-ROUTE found.

The original concern ("advisories may be compiled-in but skipped at
autotune-time because the autotune scoring picks a different layout") is
factually unfounded for this dispatcher: there is no autotune layer; the
host caller explicitly selects layout via the gemm_{rcr,rrr,crr}_pq_v2
entry points. The advisories are passive trace-only annotations that fire
deterministically when CRR is invoked for the matching shape.

Possible R41+ followup (LOW priority — methodology):

a) **Add advisory test to CI**: Bake the 8-shape audit script
   (`/tmp/wt-r40-c/r40c_audit/run_audit.sh`) into `r38_nm_gate.sh` or a
   new `r40_advisory_gate.sh` so future cycles auto-detect if a refactor
   silently drops one of the 8 advisory predicates from the dispatcher.
   Cost: ~2 min of build time per audit run. Benefit: regression-proof.

b) **Consider moving advisories to caller-side annotation**: Currently
   the advisories live INSIDE the CRR dispatcher and use
   `MXFP8_DISPATCH_TRACE=1` env-gate. Since they are pure information
   (no kernel swap, no perf benefit), an alternative would be to print
   them from the Python test harness when it sees a matching
   (M, N, K, layout) tuple. This decouples the advisory from kernel
   compilation and makes the catalog easier to maintain (one Python
   dict vs 8 if-blocks across the dispatcher). NEUTRAL — not a bug,
   just a code-organization preference.

c) **Cross-validate against gemm_crr_pq_v2 callers**: This audit only
   exercises the test_mxfp8_python.py harness. If other Python harnesses
   in the repo (e.g., r35_reviewer_bench5x.py, r37_paired_bench_2so.py)
   have a different default for MXFP8_LAYOUTS that excludes "crr", the
   advisories would not fire. R39 Reviewer Phase 2 protocol already
   requires `MXFP8_DISPATCH_TRACE=1 ... | grep <predicate>` — already
   covered. No new action needed.

## Files touched

- `analysis/fp8_gemm/mi350x/r40c_findings.md` (this file)

NOT touched (per audit-only constraint):
- `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp`
- any `*.inc` kernel file

## Build/run commands (reference)

```bash
# Build for shape (M, N, K):
THUNDERKITTENS_ROOT=/tmp/wt-r40-c ROCM_PATH=/opt/rocm \
  CPPFLAGS="-DM_DIM=$M -DN_DIM=$N -DK_DIM=$K" \
  make -B TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp

# nm-gate hygiene:
./r38_nm_gate.sh tk_mxfp8_layouts.cpython-310-x86_64-linux-gnu.so

# Trace test:
HIP_VISIBLE_DEVICES=2 MXFP8_DISPATCH_TRACE=1 MXFP8_PRESHUFFLE_QUANT=1 \
  MXFP8_WARMUP=2 MXFP8_ITERS=2 \
  python3 test_mxfp8_python.py $M $N $K 2> trace.err

# Verify expected predicate fired:
grep '\[mxfp8_dispatch\] crr_v2:' trace.err | grep '<predicate substring>'
```

Run the full 8-shape audit:
```bash
/tmp/wt-r40-c/r40c_audit/run_audit.sh
```

## Artifacts

Saved under `/tmp/wt-r40-c/r40c_audit/` (NOT committed — large per-shape
build/test logs):
- `audit.log` — full audit run output with FIRED/SILENT verdicts
- `dispatch_<tag>.txt` — filtered `[mxfp8_dispatch]` lines per shape
- `trace_<tag>.err` — raw stderr per shape
- `stdout_<tag>.log` — test_mxfp8_python.py stdout per shape (TFLOPS, SNR)
- `nm_<tag>.log` — r38_nm_gate.sh output per shape
- `build_<tag>.log` — hipcc build output per shape
- `run_audit.sh` — orchestrator script
