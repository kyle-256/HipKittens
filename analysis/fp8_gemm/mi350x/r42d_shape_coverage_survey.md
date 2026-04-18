# R42 Dev D — Phase 2: MXFP8 V2 shape coverage survey

**Branch**: r42-dev-d
**Date**: 2026-04-18
**Scope**: Beyond decode (M<256 — covered by R41 Dev C + R42 Devs A/B), survey for any other underserved shape regimes in production that V2 fastpaths miss. Specifically: N<256 wide-N tail, K<128 K-tail, and any LLaMA shape not covered by an existing V2 fastpath nor an HB-shrink variant.

## Production LLaMA shape table (from `llama_baseline_r25.json`)

| logical | M | N | K | layout-route (R36+ advisories) | wired V2 fastpath |
|--|--:|--:|--:|--|--|
| 8B Q/O attn  | 4096 |  4096 |  4096 | ADVISE-V2-RCR-8B-QO (R36C +5.83-7.05%)  | RCR-V2-EXACT-8WAVE |
| 8B K/V attn  | 4096 |  1024 |  4096 | ADVISE-V2-RRR-8B-KV (R33C +8.13% min)   | CRR-V2-HBSHRINK-B1-8B-KV (R37AB SHIP, +24.66%) |
| 8B Gate/Up   | 4096 | 14336 |  4096 | ADVISE-V2-RRR-8B-GATEUP (R34B +5.025% min) | RRR-V2-EXACT-8WAVE (BOUNDARY-LOCK +4.91%) |
| 8B Down      | 4096 |  4096 | 14336 | ADVISE-V2-RRR-8B-DOWN (R36B +9.51%)     | RRR-V2-EXACT-8WAVE |
| 70B Q/O attn | 4096 |  8192 |  8192 | ADVISE-V2-RCR-70B-QO (R36C +8.20-8.32%) | RCR-V2-EXACT-8WAVE |
| 70B K/V attn | 4096 |  1024 |  8192 | ADVISE-V2-RRR-70B-KV (R33C +10.24% min) | CRR-V2-HBSHRINK-B1-70B-KV (R37A SHIP, +29.47%) |
| 70B Gate/Up  | 4096 | 28672 |  8192 | ADVISE-V2-RRR-70B-GATEUP (R33C +7.18% min) | RRR-V2-EXACT-8WAVE |
| 70B Down     | 4096 |  8192 | 28672 | ADVISE-V2-RRR-70B-DOWN (R32C +12.14%)   | RRR-V2-EXACT-8WAVE |

All 8 prefill cells have **dedicated V2 wire-ins or advisories** (R36-R37 wire-in + R32-R36 advisory triggers). Coverage on prefill is complete.

## Decode shape table (from `r41c_findings.md`)

| logical | M | N | K | route today | R42 fastpath assignment |
|--|--:|--:|--:|--|--|
| 8B 1-tok    | 1   | 4096 | 4096 | tail kernel (V1-LEGACY-FALLBACK) | R42 Dev A SMALLM-V1-FASTPATH BLK_M=1 |
| 8B b=32     | 32  | 4096 | 4096 | tail kernel | R42 Dev B SMALLM-V1-FASTPATH BLK_M=32 |
| 8B b=128    | 128 | 4096 | 4096 | tail kernel | R42 Dev B SMALLM-V1-FASTPATH BLK_M=32 (4 tiles) |
| 70B 1-tok   | 1   | 8192 | 8192 | tail kernel | R42 Dev A SMALLM-V1-FASTPATH BLK_M=1 |
| 70B b=32    | 32  | 8192 | 8192 | tail kernel | R42 Dev B SMALLM-V1-FASTPATH BLK_M=32 |
| 70B b=128   | 128 | 8192 | 8192 | tail kernel | R42 Dev B SMALLM-V1-FASTPATH BLK_M=32 (4 tiles) |

## Gap analysis: N<256 wide-N tail (asymmetric counterpart to M<256)

### LLaMA: NONE in scope

All 8 LLaMA prefill cells have N ∈ {1024, 4096, 8192, 14336, 28672} — every N is ≥ 1024 = **4× BLK_N**. Smallest is 1024 (KV-attn). **No N<256 LLaMA shape exists**.

Note: KV-attn is "tall-thin" relative to BLK_N=256 (only 4 N-tiles per row of M-tiles), and this was the trigger for R37 Dev A's HB shrink Stage B1 production wire-in (+28-31% lift). The HB shrink already optimizes the N=1024 case at the top end of "tall-thin"; **there is no LLaMA shape requiring N<256 tail handling**.

### Decode-time KV-cache attention: out of MXFP8 GEMM scope

In production decoding with KV-cache, the `B = K^T` matrix has `K_cache_rows = past_seq_len` and `N = head_dim_per_head_count`. The attention multiplication is **not** a GEMM with N<256; it is `Q @ K^T` with `M = num_heads * head_dim`, etc. — handled by an attention kernel, not the MXFP8 GEMM dispatcher. **No N<256 production GEMM shape identified**.

### Synthetic / non-LLaMA workloads

Possible: small-MoE expert routing where N = expert_hidden / num_active_experts. Not currently in repo workload mix. **Defer to NEW model arrival.**

**Verdict for R42+ N<256 tail**: NO ACTION NEEDED. Current dispatcher's V1-LEGACY-FALLBACK + tail kernel handles N<256 correctness, but no production workload exercises it. Optimization work would be speculative.

## Gap analysis: K<128 K-tail

### LLaMA: NONE in scope

K ∈ {4096, 8192, 14336, 28672} for all 8 prefill + 6 decode cells. Smallest is K=4096 = **32× BK=128**. **No K<128 production shape**.

### MQA/GQA edge cases

Multi-query attention sometimes folds `head_dim = 128` (e.g., LLaMA3-8B GQA: K-head = 8 × 128 = 1024, V-head similar). Even at the smallest, K = 128 = exactly 1 BK iteration — works in default V2 dispatch (not a tail), though `g.fast_k = (g.k/BK)*BK = 128` is only 1 K-block, possibly under-pipelined.

Tail K=8 / K=16 / K=32 / K=64 (LoRA adapter projections, etc.) — not in current workload. The V2 EXACT predicate requires `g.k == K_DIM`; an adapter with K=64 would fall through to V1, which uses `g.fast_k = 0` (since `g.k/BK = 0`) and routes to tail kernel.

**Verdict for R42+ K<128 tail**: NO ACTION NEEDED for current LLaMA workload. Speculative for adapter-style models — flag for R44+ if adapter inference becomes a target.

## Gap analysis: shapes without V2 fastpath nor HB shrink

Cross-referencing `kernel_mxfp8_layouts.cpp:5634-5876` dispatcher chain against all 8 LLaMA prefill cells:

| shape | wired path | notes |
|--|--|--|
| 8B  Q/O    4096³        | RCR-V2-EXACT-8WAVE | covered |
| 70B Q/O    4096×8192²   | RCR-V2-EXACT-8WAVE | covered |
| 8B  K/V    4096×1024×4096 | CRR-V2-HBSHRINK-B1-8B-KV | covered (R37AB SHIP) |
| 70B K/V    4096×1024×8192 | CRR-V2-HBSHRINK-B1-70B-KV | covered (R37AB SHIP) |
| 8B  Gate/Up 4096×14336×4096 | RRR-V2-EXACT-8WAVE | covered (BOUNDARY-LOCK) |
| 70B Gate/Up 4096×28672×8192 | RRR-V2-EXACT-8WAVE | covered |
| 8B  Down   4096²×14336  | RRR-V2-EXACT-8WAVE | covered |
| 70B Down   4096×8192×28672 | RRR-V2-EXACT-8WAVE | covered |

**8/8 prefill cells are V2-fastpath-routed.** This is the primary outcome of R32-R40's 39 closed levers.

**0 prefill gaps identified.** The only outstanding open lever for prefill is `buffer_load_dword_lds` (path-change rather than layout-change — Phase 3 audit covers prior R30 Dev C closure).

## Gap analysis: V2-RRR boundary-lock cells

Per R40 BOUNDARY-LOCK classification: 8B Up V2-RRR (4096×14336×4096) Δ% sits structurally at the +5.0 boundary across 4 cycles. **Welch t deep clearance (>+18) means re-benching cannot lift it**; only kernel optimization could. R40 Dev A REFUTED HB-N+WARPS_N=2 via tile-area-conservation 3-confirm.

**Open lever for 8B Up boundary-lock**: NONE remaining within HB-* / tile-rotation paradigm (paradigm-CLOSED). Possible wedges:
1. `buffer_load_dword_lds` (Phase 3) — if VMEM dispatch is the binding resource
2. K-axis swizzle / scale-prefetch reorder (untested at the +18 Welch t scale)
3. WMMA scheduling (s_setprio / sched_barrier exploration was already CLOSED in R27-R30)

**Recommendation**: BOUNDARY-LOCK cells are not a "gap" in coverage; they are an **optimization plateau**. Survey here flags them but does not propose new prototypes (per R40+ rule: BOUNDARY-LOCK cells are excluded from STRICT re-bench attempts until kernel optimization).

## Gap analysis: layout coverage

V2 has 3 layouts wired: RCR, RRR, CRR. LLaMA prefill uses all 3 (RCR for attn, RRR for MLP via R32-R36 autotune fan-out advisories, CRR as default). Decode uses RCR for attn proj — Phase 1 design covers RCR-first; RRR/CRR are R43+ extensions.

No additional layouts (e.g., CCR — col-major C with row-major A — not used by LLaMA).

## Gap analysis: rect / non-square BLK shapes

`MXFP8_RECT_BLK_N=64` (BLK_N=128) was a R28-R31 prototype direction. Default build is square (BLK=256). Rect was REFUTED on bandwidth ceiling (R30 Dev D rect ceiling at 2× grid, paradigm-CLOSED). No new rect work.

## Findings summary

1. **Prefill: 0 NEW gaps** — all 8 LLaMA prefill cells are V2-routed. R32-R40 39 closed levers exhaust the search space on existing tile geometry.
2. **Decode: 6 gaps** — already assigned to R42 Devs A (M=1) and B (M=32/128). Phase 1 dispatcher design covers integration.
3. **N<256 wide-N tail: 0 production exposure** — no LLaMA shape, no current synthetic workload. Defer until new model arrives.
4. **K<128 K-tail: 0 production exposure** — same as above. Adapter-style models would re-open this; not in current workload.
5. **Boundary-lock cells (8B Up V2-RRR): paradigm-CLOSED** — no new wedge within HB-* / tile-rotation. Only structural avenue is `buffer_load_dword_lds` (Phase 3) IF VMEM dispatch turns out to be the binding resource at +5.0 boundary.

## Recommendations for R43+

1. **Highest priority**: complete R42 Devs A/B small-M fastpath integration (R41 Dev C ★★ MAJOR FINDING resolution). Phase 1 dispatcher design (this doc's sibling `r42d_smallm_dispatch_design.md`) provides the integration contract.
2. **Medium**: pursue `buffer_load_dword_lds` for B-side VMEM dispatch (R42 Dev C scoping). May unlock 5-15% on bandwidth-bound prefill cells (Down, Gate/Up, possibly the 8B Up boundary-lock).
3. **Low / parked**: N<256 and K<128 tail optimization. Re-open if/when production workload shifts to small-N or LoRA-adapter inference.
4. **Methodology**: add `smallm` to `r38_nm_gate.sh` regex set when R42 Devs A/B integrate. Default build (no `MXFP8_SMALLM_BLK_M_*` macros) must show 0 SMALLM symbols.

## Cross-cycle context

- R41 cumulative tally: 42 closed levers across 10 cycles (R32-R41). Prefill paradigm map is dense.
- R41 Dev C's decode-shape MAJOR FINDING is the first identification of a shape regime (M<256) outside the 42-lever closed set.
- This survey confirms: **outside M<BLK, no other shape regime is a coverage gap on the LLaMA workload**. The 42 closed levers + R37 Dev A HB shrink wire-in + R36 Dev B/C autotune fan-out advisories cover the entire LLaMA prefill matrix.
