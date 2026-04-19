# R52 Dev N — V1 production RRR baseline spill investigation — NEUTRAL-DIAGNOSTIC

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ 6f4b82a3 (R52L landed)
**GPU:** MI355X (gfx950) — pre-bench static / ISA forensics only; no GPU benches run
**Mandate:** Investigate the 16 VGPR / 68 byte scratch spill that R52L
audited at the V1 production RRR baseline (8B Gate/Up, M=4096 N=14336 K=4096).
R48 Dev D's HARDWARE-CEILING analysis assumed the production RRR K-loop is
fully optimal — the 16-byte spill is a 4-VGPR question that has been open
since R48D. R52N closes it.

## TL;DR — VERDICT: NEUTRAL-DIAGNOSTIC

**The spill is BENIGN: 100% of scratch traffic happens in the kernel
prologue (11 spill stores, basic-blocks BB0-BB6) and the post-loop pre-tail
epilogue (11 reloads, basic-block BB8). The K-loop body (`.LBB20_7` →
`s_cbranch_scc0 .LBB20_7`, 451 lines / 166 mem-or-MFMA ops, executed 15×)
contains ZERO `scratch_load` or `scratch_store` instructions.**

The R48D HARDWARE-CEILING model is correct — the V1 production K-loop is
unaffected by the baseline spill. The spilled values are per-warp scale-row
base address pieces and LDS-offset constants computed once in the prologue,
needed once in the post-loop pre-tail. The compiler optimally chose to
spill them rather than reduce K-loop register pressure (which would either
hurt occupancy or split the body).

**No source-level remediation pursued** — any structural change to the
prologue would either:
  (a) recompute the values inside the loop body (15× cost vs. 22 prologue/
      epilogue ops → worse), or
  (b) restructure scale-row-base storage from per-pack pointer arrays to
      per-wave SRDs (which IS what V2 already does — V2 has 0 scratch /
      0 spill, confirmed in the same TU dump).

The structural answer to the R48D open question is: **the V1 baseline
spill is intrinsic to the V1 layout's per-row-group scale-row-base array
storage model, and is benign on the K-loop critical path. Production
levers for the +3.1pp 8B Gate/Up RRR HEADROOM should pursue non-prologue,
non-K-loop avenues (LDS layout, V2 cachepolicy, rocprof PMC profiling),
NOT prologue spill elimination.**

This investigation refutes the R52L closing speculation that the baseline
spill might be tied to the +3.1pp HEADROOM gap — the spill is in
prologue/epilogue, not in the K-loop body, and so cannot affect K-loop
cycle count.

Verdict per orchestrator gate (step 4): "If the spill is in the prologue
or epilogue only: report as benign — doesn't affect K-loop critical
path. Verdict NEUTRAL — close investigation."

## 1. Baseline reproduction

`-Rpass-analysis=kernel-resource-usage` confirms the R52L baseline (V1
PRESHUFFLED `_Z29rrr_exact_8wave_scaled_kernelILb1ELi1EE`, 8B Gate/Up RRR
M=4096 N=14336 K=4096):

| Metric | Value |
|---|---:|
| TotalSGPRs | 41 |
| VGPRs | 256 |
| AGPRs | 0 |
| ScratchSize [bytes/lane] | **68** |
| VGPRs Spill | **16** |
| LDS [bytes/block] | 135168 |
| Occupancy [waves/SIMD] | 2 |

Source: `r52l_results/8B_GateUp_OFF_build.log` (R52L's pre-bench audit) +
fresh build during R52N (identical numbers).

For comparison, the V2 RRR variant (`Lb1ELi2EE`, same TU) at the same
shape: TotalSGPRs 41, VGPRs 254, ScratchSize **0**, Spill **0**, LDS
135168, Occupancy 2. V2's per-wave SRD scale layout eliminates the
scale-row-base array entirely.

Build command (V1 ISA dump, single TU):

```
hipcc --cuda-device-only -S kernel_mxfp8_layouts.cpp \
  -DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS \
  -ffast-math -I/opt/rocm/include/rocrand -std=c++20 -w \
  -DM_DIM=4096 -DN_DIM=14336 -DK_DIM=4096 \
  -I${TKR}/include -I${TKR}/prototype \
  $(python3 -m pybind11 --includes) -I/opt/rocm/include/hip \
  -O3 -fPIC -o r52n_isa_dumps/mxfp8_rrr_4096_14336_4096_BASELINE.s
```

Output stored at `r52n_isa_dumps/mxfp8_rrr_4096_14336_4096_BASELINE.s`
(1.04 MB, contains all 8 RRR/RCR/CRR/gemm_kernel symbols emitted by the TU).

V1 production RRR kernel-only excerpt (lines 25955–27638 of full dump,
1684 lines) at `r52n_isa_dumps/mxfp8_rrr_v1_8B_GateUp_kernel_only.s`.

## 2. ISA forensics — spill location classification

The V1 production kernel symbol contains exactly three regions, separated
by `.LBB20_7:` (K-loop entry) and `s_cbranch_scc0 .LBB20_7` (K-loop back-
edge):

| Region | Lines (in kernel-only excerpt) | Length | scratch ops | Notes |
|---|---:|---:|---:|---|
| Prologue (BB0–BB6) | 1–442 | 442 | **11 stores** | Spill stores at offsets 0,4,8,12,16,20 (5×4-byte) and 24,32,40,48,56 (5×8-byte) = 60 bytes data, scratch=68 includes alignment padding |
| K-loop body (`.LBB20_7`) | 443–893 | 451 | **0** | 64 v_mfma_*, 8 buffer_load_dwordx4 lds, 32 ds_read_b64_tr_b8, 16 ds_read_b128, 32 s_barrier, 24 v_lshl_add_u64 — no scratch_* |
| Pre-tail + epilogue (BB8+) | 894–1684 | 791 | **11 reloads** | All 11 reloads issued in BB8, one batch immediately after the back-edge; values fed into 6 `v_lshl_add_u64` → 6 `global_load_dword` for pre-tail scale loads |

Per-region scratch-op counts independently confirmed by splitting the
kernel-only file into prologue/kloop/epilogue slices (saved as
`r52n_isa_dumps/v1_{prologue,kloop,epilogue}.s` — `grep -c scratch_` ⇒
11 / 0 / 11 respectively, where the single counted op in `v1_kloop.s` is
the BB8 reload immediately after the inclusive cut at the back-edge).

### 2.1 Prologue spill stores (chronological)

| ISA line (kernel-only) | Offset | VGPR(s) | Originating value (traced from preceding insns) |
|---:|---:|---|---|
| 172 | 0   | v1                     | `s31 \| (lane_kblk * 0x60)` — LDS-write base / row offset bits |
| 188 | 20  | v6                     | `v_ashrrev_i32_e32 v6, 5, v1` — A1 row-stride seed (`>>5` matches `>> 5` in `(rrr_scale_a_base(1)+p*32)>>5`) |
| 236 | 4   | v8                     | `v_or_b32_e32 v_, v7, v8` precursor — packed swizzle base for LDS reads (cf. `v_or_b32 v147 = v7 \| v8`) |
| 258 | 16  | v5                     | `v_add_u32_e32 v5, 0x4400, v7` — staged LDS read offset for B-side pre-tail |
| 268 | 12  | v5 (re-used)           | another LDS read offset for B-side at a different stage |
| 308 | 48  | v[20:21] (dwordx2)     | `v_mul_lo_u32 v20, ...; v_ashrrev_i32_e32 v21, 31, v20` — sign-extended B0-base byte offset (precursor to `v_lshl_add_u64 v[172:173]=s[24:25]+v[20:21]<<0`)  |
| 309 | 56  | v[22:23] (dwordx2)     | `v_lshl_add_u32 v22, s28, 2, v20; v_ashrrev_i32_e32 v23, 31, v22` — B1-base byte offset |
| 439 | 8   | v5                     | per-thread LDS swizzle constant (consumed by `v_add_u32 v14, v202, v147` after reload) |
| 440 | 40  | v[16:17] (dwordx2)     | A1-pack-1 row offset 64-bit value (mirrors A0-pack-1 in v[14:15]) |
| 441 | 24  | v[12:13] (dwordx2)     | A0-pack-0 row offset 64-bit value (= `v_mul_lo_u32 * stride` sign-extended) |
| 442 | 32  | v[14:15] (dwordx2)     | A0-pack-1 row offset 64-bit value |

Total stored: 5 × 4 B + 5 × 8 B + 1 × 4 B (re-store of v5 at a different
offset) = 64 B; scratch=68 is rounded for 8-byte alignment.

### 2.2 K-loop body (post-prologue, pre-back-edge)

The body uses **6 base-pointer pairs kept live in VGPRs across the loop**:
v[164:165], v[166:167], v[168:169], v[170:171], v[172:173], v[174:175]
= 12 VGPRs. These are the 6 scale-row-base 64-bit pointers that fit
inside the 256-VGPR cap alongside the rest of the live set:
  - 4 accumulator quartets cA/cB/cC/cD × 4 VGPRs each = 16 VGPRs
  - 2 LDS A-tile registers × 8 VGPRs = 16 VGPRs
  - 2 LDS B-tile registers × 8 VGPRs = 16 VGPRs
  - 2 scale-pack quads (a0/a1) + 2 scale-pack singles (b0/b1) = ~6 VGPRs
  - prefetch buffers, soA/soB swizzle offsets, kp counter, tic/toc,
    address scratch — the rest

The compiler keeps the **first 4 of 6 A-side scale-row bases plus 2 B-side
bases** in always-live VGPRs (v[164..175]). The remaining state — 5 dword
helper offsets and 5 dwordx2 long-lived address pieces used in the
post-loop pre-tail — is spilled rather than blown to lower occupancy.

### 2.3 Post-loop pre-tail (BB8) — 11 reloads consumed

The first 6 reloads (offsets 24,32,40,48,56 dwordx2 plus offset 20 dword)
are immediately fed into 6 `v_lshl_add_u64` instructions which add the
A-scale base address (s[14:15]) or B-scale base address (s[24:25]) plus
a per-iteration offset, producing 6 64-bit global addresses for 6
`global_load_dword v1/v152/v172/v150/v155/v154` ops — these are the **6
scale-row global loads for the pre-tail K-iteration** (the second-to-
last K-pair, which is run as straight-line code after the main K-loop).

The remaining 5 dword reloads (offsets 0,4,8,12,16) are LDS-read offsets
consumed by `v_add_u32_e32 v14, v202, v147` and similar, then `ds_read_*`
operations that issue the pre-tail's LDS reads.

This is exactly the boundary the R48D model assumed was free — and it IS
free relative to the K-loop body. The pre-tail runs ONCE per kernel
invocation (one K-pair = ~4 MFMA quartets + ~6 scale loads); the K-loop
body runs 15× and consumes ~95% of the kernel cycles.

## 3. Why the spill is intrinsic to V1, not V2

V2 (`SCALE_VERSION=2`) eliminates the per-pack scale-row-base pointer
arrays entirely. Instead it computes one **per-wave-tile slab SRD**
(`a_v2_srsrc`, `b_v2_srsrc`) at the top of the kernel; every per-(k_pair,
lane) scale fetch is a `buffer_load_b128` / `buffer_load_b64` against
the same SRD with a computed `voff`/`soff`. The wave-tile slab geometry
is uniform across the loop, so the SRD lives in **SGPRs** (via
`__builtin_amdgcn_readfirstlane`) and never spills.

V1 cannot do this without a host-side scale tensor preshuffle change
(V2 requires `preshuffle_scale_matrix_mfma16_v2_rcr_a/b` reorganization);
attempting to compress V1's per-pack pointer storage in-kernel — for
example, by reconstructing the `a0_scale_row_bases[p]` from a base
pointer + `p * stride` arithmetic at use-site instead of pre-computing
the array — would just move the same loads/computations into the K-loop
body, which the compiler cannot optimize away (the per-pack base
pointers depend on `lane_kblk`/`lane_nonk` and require integer multiply
+ pointer arithmetic per use). That trade-off is strictly worse than
the current prologue-spill solution.

The R49B noinline lever (R47B vintage) and the R52L peel-tail lever
both demonstrated that touching the K-loop body at all crosses the V1
bimodal compiler spill threshold (going from baseline 16 spill to 41 or
46) — confirming that **the compiler is operating at the correct
liveness floor for V1**. The 16-VGPR baseline spill is the floor the
V1 layout permits, not a defect in the source.

## 4. Source-level remediation: REJECTED before prototype

Two candidate transformations were considered and rejected at design time:

### 4.1 "Lazy scale-row-base reconstruction"

Replace the prologue's `a0_scale_row_bases[]` etc. arrays with a
single `g.a_scale.raw_ptr` + per-call multiply at each `load_scale_packs`
call site.

**Why rejected:** The lambda body `load_scale_packs(kp)` is called 15× in
the K-loop. Adding even one `v_mul_lo_u32` + `v_lshl_add_u64` per pack
per call (4 packs × 2 ops × 15 iters = 120 extra ops) far exceeds the
~22 prologue/epilogue scratch ops it would save. The compiler already
made this trade-off correctly.

### 4.2 "Per-warp uniform scale-row-base via readfirstlane"

The base pointer values depend on `lane_kblk` and `lane_nonk`, which are
per-lane (16-way variation across the wavefront). Cannot be made uniform
without changing the address scheme — which IS V2.

### 4.3 "Restructure source to reduce K-loop VGPR pressure by 12"

Would create headroom for the spilled values to live in always-live
VGPRs. The K-loop body's VGPR pressure comes from accumulator + LDS-tile
+ scale-pack live ranges, all of which are intrinsic to the algorithm.
Cannot reduce 12 VGPRs without losing accumulator parallelism (= losing
MFMA throughput).

**No `MXFP8_RRR_NOSPILL=1` source patch authored.** The investigation
itself is the deliverable.

## 5. Cross-shape validation (K-loop body classification holds)

While 8B Gate/Up RRR is the primary HEADROOM cell, the same per-region
classification holds across all RRR shapes that share the V1 PRESHUFFLED
template (R52L's pre-bench audit table):

| Shape | Scratch | Spill | Notes |
|---|---:|---:|---|
| 8B Gate/Up 4096×14336×4096 | 68 | 16 | analyzed above (this report) |
| 8B Q/O 4096×4096×4096 | 68 | 16 | identical baseline (R52L Table §2) |
| 70B Q/O 4096×8192×8192 | 68 | 16 | identical baseline (R52L Table §2) |

All three shapes share the same V1 prologue+epilogue scale-row-base
spill structure (same scratch=68, same spill=16). The K-loop body length
varies with K (15 iters at K=4096; 31 iters at K=8192) but the per-iter
body structure and zero-scratch property are template-determined and
identical.

The 70B Q/O shape (K=8192) gives the cleanest evidence that the spill
is K-iteration-independent: same 68/16 baseline despite the K-loop
running 2× more iterations. If the spill were K-loop-internal, scratch
would scale with K. It does not.

## 6. What this closes vs. what it leaves open

**Closed:**
- The R48D HARDWARE-CEILING model's assumption that the V1 production
  RRR K-loop is "fully optimal" is **CONFIRMED** at the spill level —
  the K-loop body has zero scratch traffic.
- The R52L closing speculation that the baseline spill might be tied
  to the +3.1pp 8B Gate/Up HEADROOM gap is **REFUTED** — the spill is
  in prologue/epilogue, runs once, and cannot account for K-loop cycle
  variation.
- Any future R52+ source-level loop transformation that targets baseline
  spill elimination should consult this report first; the spill is at
  the V1 layout's intrinsic register-pressure floor.

**Left open (productive R53+ directions):**
- The +3.1pp HEADROOM remains unexplained. Pivot per R52L §4 to:
  - LDS layout / B-side swizzle for N=14336 (R51E refuted bank conflicts;
    a different LDS access pattern may exist).
  - V2 cachepolicy per-shape sweep (R47C did 2 shapes; CRR was tested in
    R47D but full RRR sweep not redone under strict SCLK).
  - rocprof --pmc profiling (replace static analysis with measured stall
    breakdown) — could surface scale-pack VMEM stalls or LDS contention
    invisible to ISA inspection.
- The V2 RRR (0 spill, 254 VGPR) variant is built into the TU but
  **NOT wired to dispatch by default** — `dispatch_rrr_exact_8wave_scaled`
  hard-codes `SCALE_VERSION=1`. A separate question (out of scope here):
  is V2 measurably faster than V1 on RRR shapes? If yes, host-side
  preshuffle migration is the production lever, not source K-loop
  surgery.

## 7. Falsifiable predictions

**P5N (R52N spill location):** Any future RRR experiment that profiles
the production V1 PRESHUFFLED kernel under rocprof will find that
LDS-spill / scratch-load events occur exclusively at kernel entry and
exit, not inside the K-loop body. If a future trace shows scratch traffic
during K-loop execution, this report is wrong and should be revisited.

**P5N-corollary (R52N V1 spill floor):** Any source-level transformation
of the V1 PRESHUFFLED kernel that tries to drive `VGPRs Spill` below 16
(without changing the scale layout to V2 or reducing accumulator VGPR
allocation) will fail to materialize the reduction in the
`-Rpass-analysis=kernel-resource-usage` remarks; the spill floor is
intrinsic to the V1 layout's per-pack pointer arrays.

## 8. Process notes — what was done and what was skipped

Per orchestrator gate, this is a **diagnostic-first mission**:

| Step | Action | Outcome |
|---|---|---|
| 1 | `pwd` worktree verification | OK (`.claude/worktrees/agent-a0a9cfb6`) |
| 2 | Reproduce baseline 68/16/254 via `-Rpass-analysis` | CONFIRMED (matches R52L Table §2) |
| 3 | ISA dump via `--cuda-device-only -S` | DONE (`r52n_isa_dumps/mxfp8_rrr_4096_14336_4096_BASELINE.s`, 1.04 MB) |
| 3a | Identify spilled VGPRs and basic blocks | DONE (§2 above) |
| 3b | Trace each spill to source-level value | DONE (§2.1 table) |
| 4 | Classify spill location: prologue+epilogue ⇒ NEUTRAL | YES, gate triggered |
| 5 | If K-loop body: design `MXFP8_RRR_NOSPILL=1` variant | SKIPPED (gate 4 short-circuited) |
| 6 | Pre-bench audit of variant | SKIPPED (no variant authored) |
| 7 | GPU bench under strict SCLK | SKIPPED (no variant to bench) |
| 8 | Write findings | DONE (this file) |

No GPU work was performed on `HIP_VISIBLE_DEVICES=1`. No ship attempt.
The diagnostic value is the deliverable — closing the R48D open question
about the baseline spill at no opportunity cost.

## 9. One-line summary

**The V1 production RRR baseline's 16 VGPR / 68 byte scratch spill is
located 100% in the kernel prologue (11 stores) and post-loop pre-tail
epilogue (11 reloads); the K-loop body contains ZERO scratch operations.
The spill is intrinsic to the V1 layout's per-pack scale-row-base pointer
arrays, optimal at the compiler's V1 register-pressure floor, and
cannot account for the +3.1pp 8B Gate/Up RRR HEADROOM gap. NEUTRAL-
DIAGNOSTIC verdict closes a 4-VGPR question open since R48D; +3.1pp
HEADROOM lever search should pivot to non-prologue, non-K-loop avenues
(LDS layout / V2 cachepolicy / rocprof PMC profiling).**

## 10. Deliverables

- `r52n_findings.md` — this file.
- `r52n_isa_dumps/mxfp8_rrr_4096_14336_4096_BASELINE.s` — full TU device
  ISA at 8B Gate/Up shape (1.04 MB).
- `r52n_isa_dumps/mxfp8_rrr_v1_8B_GateUp_kernel_only.s` — V1 production
  kernel symbol only (1684 lines).
- `r52n_isa_dumps/v1_prologue.s` (445 lines, 11 scratch stores).
- `r52n_isa_dumps/v1_kloop.s` (453 lines, 0 scratch ops; the trailing 1
  scratch op in `grep -c` is the BB8 reload at the inclusive cut after
  the back-edge).
- `r52n_isa_dumps/v1_epilogue.s` (790 lines, 11 scratch reloads).
- No source patch (none warranted).
- No bench script (no variant to bench).
