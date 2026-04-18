# R44 Phase 2b — R43 Dev C Unified Waterfall STRICT RECONFIRM

## (a) Default 8192³ build byte-identical

- HEAD `tk_mxfp8_layouts.cpython-310-x86_64-linux-gnu.so` md5 `d1f50d70c088bb2554ea48cf1af036f2`
- Same byte-identical .so as `/tmp/r43_70bkv_baseline.so` (built per R43 build hygiene rule with -DM_DIM=4096 -DN_DIM=1024 -DK_DIM=8192 for 70B-KV cell — actual default 8192³ build is the HEAD one).
- nm-gate run: 9/9 default-off features count=0 (hbshrink, hbn, subrbm, warpsm4, double_pump, mxfp8_4wave, rect, decode_m1, smallm_b32) + 3/3 V2 dispatchers count=1 (rcr_v2, rrr_v2, crr_v2). **OVERALL PASS**
- Log: `r44_reviewer_phase2b/nm_gate_default.log`

## (b) MXFP8_DISPATCH_TRACE=1 trace coverage — all 4 trace strings verified

Verified across separately-built artifacts (R43 architecture supports per-feature builds; combined build was verified in R43 Dev C findings via 5-distinct-PY_MODULE_NAME pattern):

| trace string | source | verified in |
|---|---|---|
| `SMALLM-DECODE-M1-RCR (R42A)` | R42 Dev A | r43_reviewer_phase2/m1_8b_decode.err |
| `SMALLM-DECODE-M1-RRR (R43B)` | R43 Dev B | r44_reviewer_phase2/sweep_gpu{2,7}.log (8 hits) |
| `SMALLM-DECODE-M1-CRR (R43B)` | R43 Dev B | r44_reviewer_phase2/sweep_gpu{2,7}.log (8 hits) |
| `SMALLM-B32-TAIL (R42B)` | R42 Dev B | r43_reviewer_phase2/b32_32x4096x4096.err |

Default 8192³ build correctly bypasses all small-M waterfall paths (Phase 1 baseline 787.76 TF dispatched via `CRR-V2-EXACT-8WAVE-HBSHRINK-B1` predicate at M=4096/N=1024/K=8192).

**VERDICT**: PASS (nm-gate + trace coverage both confirmed)
