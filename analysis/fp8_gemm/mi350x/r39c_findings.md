# R39 Dev C — MXFP8_DISPATCH_TRACE=1 runtime tracepoint infrastructure

**Status**: SHIP — INFRASTRUCTURE
**Branch**: `r39-dev-c` (worktree `/tmp/wt-r39-c`)
**Touches**: `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp`
**TODO item**: R39+ priority list #2 ("MXFP8_DISPATCH_TRACE=1 runtime tracepoint")

## Background

R38 Reviewer caught a CRITICAL bug class: `R37 Dev B 8B-KV production wire-in
was DEAD` — the kernel symbol `dispatch_crr_exact_8wave_scaled_v2_hbshrink`
was compiled into the .so (R37 Dev C's nm-based dead-code gate showed it
present), but the dispatcher hard-coded `g.k == 8192` while R37 Dev B's
allow-list extended to K=4096. The dispatcher never reached the kernel at
runtime for 8B-KV. R37 Dev C's nm-gate cannot catch this class of bug —
the symbol IS present, just unreachable from the dispatcher.

R39 Dev C delivers the orthogonal countermeasure: a runtime tracepoint at
every dispatch branch, env-gated by `MXFP8_DISPATCH_TRACE=1`, so Reviewer
can grep stderr for the expected predicate name and confirm the production
.so actually fires the right code path for each benchmarked shape.

## Design

`namespace tk_mxfp8_dispatch_trace` (in `kernel_mxfp8_layouts.cpp`):
- `trace_enabled()` — caches `getenv("MXFP8_DISPATCH_TRACE")` once per process.
  Returns true iff the env var is exactly `"1"`. Default behavior: zero stderr
  output, single getenv() on first dispatch.
- `record_and_check(name, M, N, K)` — fixed-size table (CAP=32) of
  `(predicate_name_literal_ptr, M, N, K)` tuples. Returns true iff the
  tuple is new. One-shot per (predicate, shape).
- `emit(layout, name, M, N, K)` — gated by trace_enabled() AND
  record_and_check(); writes one line `[mxfp8_dispatch] <layout>:
  shape=(M=%d,N=%d,K=%d) -> <name>` to stderr.

Macro `MXFP8_DISPATCH_TRACE_ONCE(LAYOUT, NAME, G)` wraps emit() with the
g.m/n/k extraction at every call site.

## Tracepoint catalog

| layout  | predicate name                                          | dispatch target                                | shape gate                                | compile gate            | SHIP origin       |
|---------|---------------------------------------------------------|------------------------------------------------|-------------------------------------------|-------------------------|-------------------|
| rcr_v2  | `RCR-V2-RECT-FAST`                                      | `dispatch_rcr_exact_8wave_scaled_v2_rect`     | `rcr_can_use_exact_8wave_scaled_rect(g)`  | `MXFP8_RECT_BLK_N==64`  | R32 Dev D Stage A1|
| rcr_v2  | `RCR-V2-EXACT-8WAVE`                                    | `dispatch_rcr_exact_8wave_scaled_v2`          | `rcr_can_use_exact_8wave_scaled(g)`       | always                  | R21 milestone-2   |
| rrr_v2  | `RRR-V2-EXACT-8WAVE`                                    | `dispatch_rrr_exact_8wave_scaled_v2`          | `rrr_can_use_exact_8wave_scaled(g)`       | always                  | R22 milestone-1   |
| crr_v2  | `ADVISE-V2-RCR-8B-QO (R36C +5.83-7.05%)`                | (advisory only — falls through)               | M=4096 N=4096 K=4096                      | always                  | R36 Dev C         |
| crr_v2  | `ADVISE-V2-RCR-70B-QO (R36C +8.20-8.32%)`               | (advisory only — falls through)               | M=4096 N=8192 K=8192                      | always                  | R36 Dev C         |
| crr_v2  | `ADVISE-V2-RRR-70B-DOWN (R32C +12.14%)`                 | (advisory only — falls through)               | M=4096 N=8192 K=28672                     | always                  | R32 Dev C / R33 A |
| crr_v2  | `ADVISE-V2-RRR-70B-GATEUP (R33C +7.18% min)`            | (advisory only — falls through)               | M=4096 N=28672 K=8192                     | always                  | R33 Dev C / R34 A |
| crr_v2  | `ADVISE-V2-RRR-70B-KV (R33C +10.24% min)`               | (advisory only — falls through)               | M=4096 N=1024 K=8192                      | always                  | R33 Dev C / R34 A |
| crr_v2  | `ADVISE-V2-RRR-8B-KV (R33C +8.13% min)`                 | (advisory only — falls through)               | M=4096 N=1024 K=4096                      | always                  | R33 Dev C / R34 A |
| crr_v2  | `ADVISE-V2-RRR-8B-GATEUP (R34B +5.025% min)`            | (advisory only — falls through)               | M=4096 N=14336 K=4096                     | always                  | R34 Dev B / R35 A |
| crr_v2  | `ADVISE-V2-RRR-8B-DOWN (R36B +9.51%)`                   | (advisory only — falls through)               | M=4096 N=4096 K=14336                     | always                  | R35 Dev D / R36 B |
| crr_v2  | `CRR-V2-HBSHRINK-B1-70B-KV (R37AB SHIP)`                | `dispatch_crr_exact_8wave_scaled_v2_hbshrink` | M=4096 N=1024 K=8192                      | `MXFP8_CRR_BLK_M==128`  | R37 Dev A/B       |
| crr_v2  | `CRR-V2-HBSHRINK-B1-8B-KV (R38 wrap fix 66ef02d8)`      | `dispatch_crr_exact_8wave_scaled_v2_hbshrink` | M=4096 N=1024 K=4096                      | `MXFP8_CRR_BLK_M==128`  | R38 wrap fix      |
| crr_v2  | `CRR-V2-HBNSHRINK (R38A NO-SHIP, exploratory)`          | `dispatch_crr_exact_8wave_scaled_v2_hbnshrink`| `crr_can_use_exact_8wave_scaled_hbnshrink`| `MXFP8_CRR_BLK_N==128`  | R38 Dev A NO-SHIP |
| crr_v2  | `CRR-V2-EXACT-8WAVE-DEFAULT`                            | `dispatch_crr_exact_8wave_scaled_v2`          | `crr_can_use_exact_8wave_scaled(g)` (M==M_DIM && N==N_DIM && K==K_DIM) | always | R22-B             |
| crr_v2  | `CRR-V2-RECT-FAST`                                      | `dispatch_crr_exact_8wave_scaled_v2_rect`     | `crr_can_use_exact_8wave_scaled_rect(g)`  | `MXFP8_RECT_BLK_N==64`  | R31 Dev A         |
| rcr_v2/rrr_v2/crr_v2 | `V1-LEGACY-FALLBACK (no V2 predicate matched)` | `dispatch<L,true>(g)`                         | (none — final fallback)                   | always                  | (legacy V1)       |
| rcr_pq/rrr_pq/crr_pq | `V1-PQ-DEFAULT`                            | `dispatch<L,true>(g)`                         | (none — V1 entry point)                   | always                  | (legacy V1)       |

## Verification

### (a) Zero-overhead default behavior

Default build (no compile flags) on 8192³:
```
$ HIP_VISIBLE_DEVICES=0 MXFP8_PRESHUFFLE_QUANT=1 MXFP8_LAYOUTS=crr \
    python3 test_mxfp8_python.py 8192  2> stderr.txt
$ wc -c stderr.txt
0 stderr.txt
```
**Result: PASS** — zero stderr output without env var, byte-identical to
pre-R39 default behavior. The single `getenv()` per process on first
dispatch is the only added runtime cost.

### (b) Trace-enabled default build

Same default 8192³ build, with `MXFP8_DISPATCH_TRACE=1`:
```
[mxfp8_dispatch] crr_v2: shape=(M=8192,N=8192,K=8192) -> CRR-V2-EXACT-8WAVE-DEFAULT
```
Exactly one line — confirming the production CRR path fires (no advisory
since 8192³ does not match any LLaMA cell), no V1 fallback.

### (c) nm-gate verification

Default build (`r38_nm_gate.sh tk_mxfp8_layouts.cpython-310-x86_64-linux-gnu.so`):
- All "default-off" features (hbshrink, hbn, subrbm, warpsm4, double_pump,
  mxfp8_4wave, rect): count=0 — PASS
- All runtime-gated features (rcr_v2, rrr_v2, crr_v2): count>=1 — PASS
- New trace helper symbols present (verified via
  `nm -D -C | grep tk_mxfp8_dispatch_trace`):
  - `tk_mxfp8_dispatch_trace::emit(...)`
  - `tk_mxfp8_dispatch_trace::trace_enabled()::cached`
  - `tk_mxfp8_dispatch_trace::record_and_check(...)::overflow_warned`
  - `tk_mxfp8_dispatch_trace::record_and_check(...)::table`
  - `tk_mxfp8_dispatch_trace::record_and_check(...)::n_entries`
- **OVERALL: PASS** (no regression)

### (d) R38 wrap fix smoke test (`66ef02d8`)

Production HB shrink build for 8B-KV (M=N=K dims=4096/1024/4096,
`-DMXFP8_CRR_BLK_M=128 -DMXFP8_CRR_HBSHRINK_PIPELINE=1`):

nm-gate `--expect-active hbshrink`: count=3 — **PASS** (kernel symbols compiled in).

Runtime `MXFP8_DISPATCH_TRACE=1` test on shape 4096×1024×4096:
```
[mxfp8_dispatch] crr_v2: shape=(M=4096,N=1024,K=4096) -> ADVISE-V2-RRR-8B-KV (R33C +8.13% min)
[mxfp8_dispatch] crr_v2: shape=(M=4096,N=1024,K=4096) -> CRR-V2-HBSHRINK-B1-8B-KV (R38 wrap fix 66ef02d8)
```

**RESULT: PASS — R38 wrap fix `66ef02d8` works in production.**

The trace confirms 8B-KV (K=4096) reaches `dispatch_crr_exact_8wave_scaled_v2_hbshrink`
via the predicate added in the R38 wrap fix. Pre-R38 (when dispatcher hard-coded
`g.k == 8192`), the trace would have shown only the `ADVISE-V2-RRR-8B-KV`
advisory followed by `CRR-V2-EXACT-8WAVE-DEFAULT` (or `V1-LEGACY-FALLBACK`),
NOT `CRR-V2-HBSHRINK-B1-8B-KV`.

Cross-check on 70B-KV (4096×1024×8192):
```
[mxfp8_dispatch] crr_v2: shape=(M=4096,N=1024,K=8192) -> ADVISE-V2-RRR-70B-KV (R33C +10.24% min)
[mxfp8_dispatch] crr_v2: shape=(M=4096,N=1024,K=8192) -> CRR-V2-HBSHRINK-B1-70B-KV (R37AB SHIP)
```
Also PASS — R37 Dev A/B wire-in for 70B-KV continues to fire.

## Recommended Reviewer Phase 2 protocol (R39+)

For every Reviewer Phase 2 SHIP-verify run, set `MXFP8_DISPATCH_TRACE=1`
and grep stderr for the expected predicate name:

```bash
# 8B-KV HB shrink B1 wire-in verification:
MXFP8_DISPATCH_TRACE=1 python3 r35_reviewer_bench5x.py mxfp8 crr 4096 1024 4096 \
    2> trace.err
grep -q 'CRR-V2-HBSHRINK-B1-8B-KV' trace.err \
    && echo "PASS: HB shrink B1 fired for 8B-KV" \
    || echo "CRITICAL: HB shrink B1 did NOT fire — dispatcher bug"

# 70B-KV:
MXFP8_DISPATCH_TRACE=1 python3 r35_reviewer_bench5x.py mxfp8 crr 4096 1024 8192 \
    2>&1 | grep CRR-V2-HBSHRINK-B1-70B-KV
```

For NEW SHIPs introducing a new predicate:
1. Add a new `MXFP8_DISPATCH_TRACE_ONCE` call in the dispatcher with a
   unique symbolic name including the SHIP origin (cycle + dev letter +
   commit ref short-SHA if relevant).
2. Add the expected name to the cycle's Reviewer ship-verify orchestrator
   so Phase 2 grep proves the predicate actually fired.
3. Update this catalog table.

## Files touched

- `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp` (+ ~80 LoC helper,
  ~8 stderr advisory blocks refactored to env-gated, 6 routing tracepoints
  added)
- `analysis/fp8_gemm/mi350x/r39c_findings.md` (this file)

## Build commands (reference)

```bash
# Default build (zero-overhead verification):
THUNDERKITTENS_ROOT=/tmp/wt-r39-c ROCM_PATH=/opt/rocm \
  make -B TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp

# Production HB shrink build (8B-KV):
THUNDERKITTENS_ROOT=/tmp/wt-r39-c ROCM_PATH=/opt/rocm \
  CPPFLAGS="-DM_DIM=4096 -DN_DIM=1024 -DK_DIM=4096 -DMXFP8_CRR_BLK_M=128 -DMXFP8_CRR_HBSHRINK_PIPELINE=1" \
  make -B TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp

# Trace-enabled smoke test:
HIP_VISIBLE_DEVICES=0 MXFP8_DISPATCH_TRACE=1 MXFP8_PRESHUFFLE_QUANT=1 \
  MXFP8_LAYOUTS=crr MXFP8_WARMUP=2 MXFP8_ITERS=2 \
  python3 test_mxfp8_python.py 4096 1024 4096 2>&1 | grep '\[mxfp8_dispatch\]'
```
