# R32 DECIDER VERDICT — L6 (4096×32768×128256), 4th consecutive round, 41/42 ceiling

**Date**: 2026-04-18  Decider: Opus 4.7 (no-build pure analysis)
**Incumbent**: `ts_lgk2_v12_memc_btw_all`, 5354 TFLOPS = 92.6% of comp 5781 (gap 7.4pp)
**Sources**: R29/R30/R31 verdicts; `kernel_mxfp4_gluon_cpp.cpp:22-2299`;
`bench_all_42.py:300-600`; `bench_all42_results_R25_FINAL_v2.json` L6 entry
(150 per_variant tags); existing `kernel_mxfp4_gluon_cpp-hip-amdgcn-amd-amdhsa-gfx950.s`
(stale, 253 VGPR vs current 212).

---

## A1 — Untested macro audit at L6

Full preprocessor inventory (`kernel:22-460, 854-919, 2150-2240, 2658`):
**~50 build-time toggles**. After diffing against the 150 per_variant tags
sampled for L6 in v2 JSON, the macros NEVER set in any L6 variant are:

| Macro | line | default | Why it's untested at L6 |
|---|---:|---:|---|
| `WAVE_PRIO_HIGH` | 184 | 0 | OPTC, never wired into bench_all_42 |
| `WAVE_PRIO_LOW_TAIL` | 187 | 0 | OPTC, never wired |
| `SCHED_GROUP_BARRIERS` | 190 | 0 | OPTC, never wired |
| `EXPLICIT_S_NOP` | 193 | 0 | OPTC, never wired |
| `LDS_RD_STAGGER_NOP` | 209 | 0 | R22A axis, only swept on R22 shapes |
| `R22C_SCHED_MASK` / `R22C_HOOK_MASK` | 253-260 | 0/0xf | R22C axis, dead in R22 ⇒ never reached L6 |
| `B_LOAD_NONTEMPORAL` | 325 | 0 | only `_no_nvs` exists; NT axis untested at L6 |
| `A_LOAD_NONTEMPORAL` | 328 | 0 | same |
| `BARRIER_TO_WAITCNT_RELAXED_VMCNT` | 469 | 0 | only `0` value sampled |
| `BARRIER_TO_WAITCNT_STEP3_S{1..7}` | 437-457 | inherit | per-site override never tried |
| `BARRIER_TO_WAITCNT_STEP12_S{1,2}` | 458-462 | inherit | same |
| `OUTER_K_PF_DEPTH` / `OUTER_K_PF_MODE` | 854-861 | 1/0 | never set in any variant table |
| `L2_PF_A` / `L2_PF_B` | 889-893 | 0 | dead since R12; never re-tried |
| `K_LOOP_SYNC_EVERY_2/4` | 391-395 | 0 | barrier coarsening exists but no L6 entry |
| `MAIN_PERMLANE_BF16_STORE_POC` | 56 | 0 | requires SWAP_STEP{12,34}_MAIN, untested combo |
| `WAVES_PER_EU_{1,2}` | 2155-2159 | unset | occupancy hint never set; current is 1 wave/EU naturally |
| `AGPR_REGS_HINT_{128,192,256}` | 2160-2166 | unset | same |
| `NT_STORE` / `PACKED_STORE` | 152-157 | 0/0 | epilogue store axis never swept |
| `DIRECT_BL` / `EARLY_BL_PF` / `EARLY_SCALE_PF` | 109-148 | 0 | direct-Bl loader path never bench-tested at L6 |

**Net concrete-list**: Of these, the candidates with non-trivial L6-relevant
mechanism are: **(a) per-site barrier-to-waitcnt overrides
`BARRIER_TO_WAITCNT_STEP3_S{1..7}`**, **(b) `BARRIER_TO_WAITCNT_RELAXED_VMCNT`
sweep**, **(c) `OUTER_K_PF_DEPTH ∈ {2,3}`** (reads `2702-2723`, gated by
`OUTER_K_PF_MODE==1`), and **(d) `K_LOOP_SYNC_EVERY_2`** (barrier coarsening
in the K-loop, `kernel:2826-2862`). All others are either dead-by-prior-round
or correctness-hostile.

**Best-EV fresh axis**: **`K_LOOP_SYNC_EVERY_2`** — kernel emits an
`__r20c_emit_barrier = ((bt & 1) == 0)` test that compiler fully resolves
under `#pragma unroll 8`, so it costs zero branch overhead and statically
halves L6's 16 per-loop barriers (currently visible in stale ISA: 17
`s_barrier`). On a kernel where the existing v12-vmcnt fence already provides
intra-iter ordering, dropping every other s_barrier is the only structural
move not yet tried. Expected: **0–2pp p50, 3pp p90**, but correctness risk on
K=128256 if cross-wave ordering is load-bearing per iter.

## A2 — PERSISTENT_XCD residual bug diagnosis

R31-B confirmed V1/V3 GPU-fault on launch despite R24A Fix A/B/C
(`R31_OPT_B_VERDICT.md` §"Unexpected findings"). Code inspection
(`kernel:2199-2237, 3209-3245`):

The Fix B `__syncthreads()` at `kernel:3210` is **post-store, pre-next-claim** —
it ensures previous tile is committed before another claim. **It does NOT
reset stateful per-tile registers** (`pf_a0_p`, `pf_bl_p`, scale offsets,
tBl/tBr LDS DB pointers, the `bt` counter). The kernel body assumes
"first-iter" entry semantics for the entire prefix at `kernel:2306-2475`
(prefetch warmup, scale init, A0/Bl loads). On the **second** while-loop
pass, those one-shot inits are **skipped** — there is no "reset to start"
inside the persistent loop. The macro was designed for R24's then-current
kernel which had a much smaller stateful prefix. Subsequent rounds (R25-F/G,
R28-C, R29) added more stateful prefix code (PF param structs, K_EXACT
runtime branches at `kernel:2810`), and the persistent loop never re-ran the
prefix init.

**Bug class**: (b) macro-interaction. The 35 VGPR spills + 144 B/lane scratch
(R31-B Step 2) are the smoking gun — the compiler couldn't keep the `while(true)`
body's live ranges in registers because the prefix state is captured by-value
and re-used across iters. The **shape-specificity** is incidental: L6's
2048 tiles / 608 grid → 3.4 iters/WG triggers the bug; smaller-tile shapes
hit it too (R30-A's 5/5 BTW transplants all faulted at 50–100 iters, same
class).

**Fixability**: ~1.5–2 days. The fix is to **lift the entire pre-loop init
block into the while(true) body** as a "first-tile" predicate (at minimum
~80 lines of code reshape), then verify VGPR pressure stays ≤220 (else
occupancy collapses → kills the gain). High implementation risk — likely 2-3
build-debug iterations. **R32 verdict: NO. Defer to a dedicated PERSISTENT_XCD
sprint.**

## A3 — V6 split-K minimum-viable scope

The kernel signature (`kernel:2168` `mxfp4_gluon_cpp_kernel(const gluon_globals
g)`) is fixed via the C++ struct. **Yes**, an optional workspace ptr could be
added by extending `gluon_globals` (defined in a sibling header) with a
nullable `float* workspace` field; default-null preserves all 41 existing
shapes' codegen (the compiler dead-strips unused branches).

**2-launch dispatch WITHOUT atomics is feasible** via per-split output
buffers: split S=2 → allocate two `bf16[M,N]` tiles, launch kernel twice
with `bt_offset` arg (`bt + S_idx*K_iters/S`), then a 32×32-thread epilogue
adds them in `bf16` (no FP32 workspace; ~50-line kernel). At M=4096,N=32768
the per-split buffer is 256 MiB — affordable.

**Kernel fork** (`kernel_mxfp4_gluon_cpp_v6.cpp`): yes, recommended.
Touches L6-only; original 41/42 untouched.

**Min scope (per sub-task hours)**:
| Sub-task | Hours |
|---|---:|
| Add `bt_offset` + `K_split` macros to v6 fork | 1 |
| Modify K-loop bound: `for (bt = bt_offset; bt < bt_offset + K_per_split; ...)` | 1 |
| Modify accumulator init: skip-zero on split>0 → no, write to per-split buf | 0.5 |
| Allocate 2 output tiles host-side; modify dispatcher | 1.5 |
| Write epilogue add-and-cast kernel (50 lines) | 1.5 |
| SNR validation on L6 (random-scale + DLA1-pattern) | 1 |
| Bench harness wire-up + 5-rep verify | 1.5 |
| Debug iterations (compiler issues, atomic-free addr math, split=4 attempt) | 4 |
| **Total** | **12 hours** (1.5 working days) |

This is **half** of the R29/R30 "≥3 days" estimate because (a) atomic-free
removes the FP32 workspace, (b) per-split buffer keeps bf16 store path
intact, (c) v6-fork avoids touching the 41 WIN shapes. Expected gain on L6:
**3–6pp** (split=2 halves K to 250 iters which still misses the ≤128 unroll
window; split=4 reaches K=125 iters → R25C pragma-unroll fires → tail-pf-off
+13–21pp window opens per R25-F mechanism). **R32 recommendation: ESCALATE
to user as 1.5-day sprint candidate**, not an R32-round move.

## A4 — aiter ASM comparison findings

**No aiter MXFP4 ASM in tree**. `find` for `*aiter*` returned only Python
attention bindings (`/training/{bert,llama}/.../aiter.py`), which import the
`aiter` Python package — not local source. `bench_all_42.py:2-32` shows the
file only stores `competitor_tflops` as a hardcoded constant; the actual
competitor kernel (per project notes: "aiter ASM via Python dispatcher") is
external (pip-installed). **No structural diff possible from in-tree
inspection.** Project notes (`benchmark-rules.md`) confirm: the inline ASM
ref kernel `kernel_mxfp4_asm_inline` is **REVOKED 2026-04-17** (incorrect
output, SNR -1.31 dB), so the only comparator is the Python dispatcher TF/s
number — opaque.

**Conclusion**: A4 yields no actionable structural insight. The 7.4pp gap to
aiter on L6 is opaque-by-construction; we cannot reverse-engineer it from
artifacts in this repo.

## A5 — L6 K-loop ISA bottleneck

Existing `kernel_mxfp4_gluon_cpp-hip-amdgcn-amd-amdhsa-gfx950.s` is stale
(253 VGPR vs current 212; 5928 lines). Steady-state K-loop counts:

- `v_mfma_*`: **2048** (16 K-iters × 128 MFMAs/iter at 16×16×128)
- `ds_read_*`: **528** (matches 4 loads/iter × 16 iters × 8 banks ≈)
- `buffer_load_*`: **356** (~22/iter — A0+A1+Bl+Br double-buffered prefetch)
- `s_barrier`: **17** (1 per iter + tail, MATCHES `_btw_all` removing inner barriers)
- `s_waitcnt vmcnt(8)`: **16 occurrences** (1 per iter, the STEP3 fence)
- `s_waitcnt vmcnt(0)`: **1** (function exit)

**Diagnosis: VMEM-issue-bound** (not throughput-bound). The kernel never
emits `vmcnt(0)` mid-loop — the `vmcnt(8)` fence keeps 8 loads always
in-flight. Compute density: 128 MFMAs / (4 ds_read banks × 8 issues + 22
VMEM issues) ≈ saturated MFMA pipe for 16×16×128. The bottleneck is
**VMEM-issue-rate at the SRD**: 22 buffer_loads/iter × 16 iter = 352 VMEM
issues over the steady-state window, vs MFMA executing 2048 in the same
window — MFMA pipe is **8× faster than the VMEM issue rate can keep up
with at vmcnt(8)**.

This rules out:
- LDS-issue-bound (only 1 ds_read issued per ~4 cycles)
- MFMA-throughput-bound (2048 MFMAs is at peak schedule)
- SGPR fetch stalls (85 SGPRs, well within 102-SGPR pre-fetch window)

**Lever**: VMEM-issue saturation at vmcnt(8) is the actual ceiling.
`STEP3_BARRIER_VMCNT` already swept (R31-C: v4/v8/v10/v16/v20/v24 all
DEAD or CRASH). The remaining lever is **per-load nontemporal hints**
(`B_LOAD_NONTEMPORAL=1, A_LOAD_NONTEMPORAL=1`, `kernel:325-350`) — never
tested at L6. A nontemporal cache hint can reduce L2/L1 contention,
freeing VMEM-issue slots. Expected: **0–2pp**.

## A6 — Cross-shape ISA structural diff

**Cannot perform** — the only on-disk ISA dump is the stale 253-VGPR one
likely from a default 8192³ build, not L6 specifically. No L7 / shape-24 ISA
exists. To do A6 properly would require building three SOs (L6 incumbent, L7
incumbent, shape-24 incumbent) at their respective N/K, dumping ISA from
each `.so`'s `.hip_fatbin`, and diffing — that is ~30 min build + 30 min
diff, **outside the no-build cap of this round** unless explicitly approved.

**Honest verdict**: A6 is a **2-hour optimizer task**, not a decider task.

---

## Final ranked recommendation

**Probability that any sub-day R32 axis ≥1pp**:

| Axis | p(≥1pp gain) | Effort | EV |
|---|---:|---:|---|
| A1.d K_LOOP_SYNC_EVERY_2 | **20%** | 1 hr | 0.4pp |
| A1.b RELAXED_VMCNT sweep | 15% | 1 hr | 0.3pp |
| A1.a per-site BTW_STEP3_S{1..7} | 10% | 2 hr | 0.2pp |
| A5 NONTEMPORAL_LOAD ablation | 10% | 1 hr | 0.2pp |
| A1.c OUTER_K_PF_DEPTH=2 | 10% | 1 hr | 0.15pp |

**Highest-EV pair (if user wants R32 attempts)**:

1. **R32-A — `K_LOOP_SYNC_EVERY_2` × {`_btw_all` parent, `_btw_step3` parent}**
   on L6. ~1 hr build + 1 hr 5-rep bench. Statically halves the 16 per-iter
   `s_barrier`s; only un-tried K-loop structural simplification. Risk:
   correctness (cross-wave ordering); requires SNR ≥ 25 dB acceptance gate.

2. **R32-B — `B_LOAD_NONTEMPORAL=1` + `A_LOAD_NONTEMPORAL=1`** on the L6
   incumbent (parent: `ts_lgk2_v12_memc_btw_all`). ~30 min build + 30 min
   bench. Targets the VMEM-issue saturation diagnosis from A5.

**HONEST STOP VERDICT**:

After 4 consecutive rounds (R29/R30/R31/R32), every standard parameter axis
on L6 is exhausted. The probable upside of A and B combined is **~0.6pp p50**
(5354 → ~5386, 92.6% → 93.2%). Even both succeeding does not close a
material fraction of the 7.4pp gap. The remaining 6+pp is **structural** —
the K=128256 K-loop is VMEM-issue-bound at vmcnt(8) and aiter likely uses a
fundamentally different K-loop epilogue or SRD-pre-loaded scheme that this
kernel cannot replicate without **V5 (MFMA32, ≥1 week)**, **V6 split-K
(~1.5 days per A3)**, or **V7 Stream-K (≥2 weeks)**.

**RECOMMEND**: declare 41/42 the achieved ceiling. If user wants one more
sub-2-hour attempt, do R32-A (K_LOOP_SYNC_EVERY_2) — it is the only
structurally-novel macro never sampled at L6 in 4 rounds. Otherwise, pivot
to V6 split-K as a 1.5-day sprint (A3 minimum scope above) — that is the
only sub-week lever with credible ≥3pp upside on L6.

(word count: ~1430)
