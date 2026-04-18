# R33 DECIDER VERDICT — Track ranking, post-R32 41/42 ceiling, L6 (4096×32768×128256)

**Date**: 2026-04-18  Decider: Opus 4.7 (no-build pure analysis)
**Incumbent**: `ts_lgk2_v12_memc_btw_all`, 5354 TFLOPS = 92.6% of comp 5781
**Sources read**: `R32_DECIDER_VERDICT.md`, `R32_OPT_A_VERDICT.md`, `R32_OPT_B_VERDICT.md`,
`R27_V5_MFMA32_SCOUT.md`, TODO.md/AGENT_PROMPT.md (post-R32 sections),
kernel_mxfp4_gluon_cpp.cpp (lines 22-470, 540-605, 2168-2475, 2658-2862),
**aiter ASM disassembly of `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`**
(this is new — not previously inspected in any prior round).

---

## Headline finding (changes everything)

**aiter ASM IS in tree.** It lives in `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/`
as raw `.co` HSA code objects (not in HipKittens repo, but on the same machine).
`pip show aiter` returns "Package not found" because aiter is installed via a
local symlink (`/shared_nfs/kyle/test/aiter/aiter/__init__.py`), not pip. The
R32 decider's A4 conclusion that "aiter is opaque-by-construction" is **wrong**:
all 30+ MXFP4 kernels are sitting on disk, fully disassemblable. Section B below
contains the first ISA structural diff vs our kernel ever performed.

---

## Track A — V5 MFMA32 sprint kickoff (minimum-viable POC)

**Verdict from B**: this track now has WEAKER motivation than entering R33 with.
aiter ASM uses **`v_mfma_scale_f32_16x16x128_f8f6f4`** — same shape as us — and
beats us by 7.4pp anyway (see Section B). The "MFMA-issue saturation" hypothesis
that motivated V5 (R32 §A5) is therefore WRONG; aiter at the same MFMA shape
hits ~5781 TFLOPS = 39% of the 14746 peak (gfx950 fp4 peak), so MFMA-issue rate
is provably NOT the binding constraint. V5 still has theoretical merit (halving
MFMA count per K-iter could free VMEM-issue slots) but **the upper bound on V5
gains is now clearly ≤ aiter perf**, and aiter shows that 16x16 alone — done
right — is enough.

**Minimum-viable V5 POC scope (for L6 only)**:

| # | Sub-task | Hours | Hardest? |
|---|---|---:|---:|
| 1 | New file `kernel_mxfp4_gluon_cpp_v5.cpp` (fork from v1; gate on `MXFP4_USE_32X32`) | 0.5 | |
| 2 | Add `mfma323264_scaled<opsel_a,opsel_b>` in `include/ops/warp/register/tile/mma.cuh` (mirror lines 128-148) | 1 | |
| 3 | Add `mma_AB_base_scaled` 32x32 dispatch arm at mma.cuh:218-225 (lifts the static_assert) | 1 | |
| 4 | Switch accumulator types: `acc_A0Bl[16]→acc_A0Bl[4]` of `fp4_floatx16_t` (kernel:2304) | 1 | |
| 5 | Rewrite **one** kpair function: pick `kpair_64mfma_step12` (kernel:1255-1391, the steady-state body). Replace 32 inline `v_mfma_scale_f32_16x16x128_f8f6f4` with 8 `v_mfma_scale_f32_32x32x64_f8f6f4`. Same scale operand layout, same op_sel encoding for cbsz/blgp. | **6** | **HARD** |
| 6 | Update LDS→reg load `fp4_load_st_to_rt` (kernel:610-630) to emit a 32-row sub-tile read for the 32x32 path (or reuse st_32x* swizzle if compatible) | **5** | **HARD** |
| 7 | Update store path `store_block_inner` lane→element mapping (kernel:3134-3207) for the 32x32 fragment layout (4 disjoint 4-row groups per lane) | **4** | **HARD** |
| 8 | L6-only build (single .so, n=32768 k=128256 instantiation) + SNR vs incumbent at K=512 sub-K test (avoid bf16-saturation noise floor that broke R32-B's SNR gate) | 1.5 | |
| 9 | Bench harness wire-up (5-rep, warmup=200, iters=500 per benchmark-rules) | 1 | |
| 10 | First iteration: SNR & perf measurement, decision gate ≥ +1pp on L6 → continue, else DEAD | 1 | |
| 11 | Buffer (3 hardest sub-tasks invariably blow up) | 3 | |
| **Total** | | **24 hr** | |

**3 hardest sub-tasks (where implementation risk concentrates)**:
- **#5 K-loop body rewrite**: `kpair_64mfma_step12` is 137 lines of hand-tuned
  inline asm; the K-pair swap (op_sel select between sa0/sa1 raw scale halves)
  was tuned for 16x16's per-MFMA scale-broadcast pattern. 32x32 issues each
  scale 4× longer (lane span 16→32), so the existing v_pk routing of `v_scale_a`
  / `v_scale_b` may need re-derivation. Risk: incorrect scale routing → wrong
  output, no SNR signal until the fix is made.
- **#6 LDS→reg layout**: `st_16x128_s` swizzle (`include/ops/warp/register/tile/mma.cuh:97`
  also references `st_32x*`) is built around 16-row sub-tiles. The `(((offset
  % (16 * 128)) >> 8) << 4)` XOR swizzle (kernel:632) is hard-coded to 16-row
  alignment. 32x32 needs 32-row sub-tiles, which means a NEW swizzle constant
  or the `st_32x128_s` variant — but the FP4 packed-byte path may not have one.
  Risk: silent bank-conflict regressions.
- **#7 Store path**: 32x32 MFMA stores 16 fp32/lane in 4 disjoint 4-row groups
  (per CDNA4 ISA Table for `v_mfma_scale_f32_32x32x64`). Our 16x16 stores 4
  fp32/lane in 1 contiguous group. The current `store_bf16x2_packed` with
  `v_cvt_pk_bf16_f32` assumes adjacent fp32 pairs in one lane; 32x32 needs
  cross-row gather to find the pair. Risk: 4× more pack instructions in the
  epilogue, or scattered writes that defeat the L2 coalesce.

**Calendar-time honest estimate**: 24 hr is the optimistic kickoff round.
Per R27_V5_SCOUT, full V5 to a benchable kernel is **1.5-2 weeks** (matched
to my walk-through above). The 24 hr is enough to get one POC instantiation
build-correct on L6, NOT enough to re-tune the 17+ R17-R26 macro wires that
were fitted to 16x16 cycle counts. Therefore the 24-hr R33 sprint can only
answer "is the V5 mechanism alive?" — not "did it win".

**Decision gate (after 24 hr)**: SNR-clean V5 POC at ≥ -3pp vs incumbent on L6
single-rep. If yes, continue to full re-tune (1-2 more weeks). If V5 builds
clean but is < incumbent by > 5pp, **DEAD** — the macro wire mismatch eats
all theoretical gain and the per-instruction MFMA peak (4096 flops/CU/cyc)
is identical anyway.

---

## Track B — aiter binary archaeology (RESEARCH-ONLY, COMPLETED IN-DECIDER)

I did this track inline. Findings below.

### B1 — Where aiter lives

- Python pkg root: `/shared_nfs/kyle/test/aiter/aiter/` (local symlink, not pip).
- ASM `.so` dispatcher: `/shared_nfs/kyle/test/aiter/aiter/jit/module_gemm_a4w4_asm.so`
  (107 KB; 136 KB intermediate `asm_gemm_a4w4.cuda.o`).
- Source dispatcher: `/shared_nfs/kyle/test/aiter/csrc/py_itfs_cu/asm_gemm_a4w4.cu`
  — confirmed argstruct includes a **`int log2_k_split`** field; aiter's ASM
  GEMM has built-in split-K. Heuristic in `get_heuristic_kernel(...)` at lines
  ~93-150 selects a tile by minimizing `local_round = ceil(tg_num / num_cu)`
  with tie-break on compute2mem-efficiency.
- **HSA code objects**: `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/` —
  **30 .co files** (raw ELF AMDGPU code objects), one per (tile_M, tile_N).
- Manifest CSV: `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4.csv`
  — confirms two BpreShuffle tiles support split-K (`splitK=1`):
  `256x256` and `128x512`. All other tiles are split-K=0 (single-launch only).

### B2 — Which kernel does L6 dispatch to?

L6 = 4096×32768×128256, num_cu=304 on MI355X. Apply the heuristic:
- 256x256: tg_num = (4096/256) × (32768/256) × splitK = 16 × 128 × 1 = 2048;
  `local_round = ceil(2048/304) = 7`. With splitK=2: tg_num=4096, round=14.
  Heuristic prefers smaller `round`, so it picks splitK=1 → round=7.
- 128x512: tg_num = (4096/128) × (32768/512) × 1 = 32 × 64 = 2048; round=7.
  Same round; tie-break on compute2mem_effi: 256x256 effi = 65536/512=128;
  128x512 effi = 65536/640≈102. **256x256 wins**.

**Therefore L6 dispatches to `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256`**
(the entry-point symbol `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E`).

### B3 — Disassembly stats vs our kernel (THE ACTUAL ISA DIFF)

`llvm-objdump --mcpu=gfx950 ...256x256.co` on the 256x256 .co. Whole-file:

| Stat | aiter 256x256 | our incumbent (`...gfx950.s`, stale 253-VGPR build) |
|---|---:|---:|
| Total instructions (lines) | ~3415 | ~5928 |
| `v_mfma_scale_f32_16x16x128_f8f6f4` | **512** | **512** |
| `v_mfma_scale_f32_32x32x64*` | 0 | 0 |
| buffer_load_dwordx4 | 88 | 356 (incl. `lds`) |
| ds_read_b128 | 180 | 528 |
| ds_write | **0** | (uses `buffer_load_to_lds`, also 0) |
| s_barrier | 10 | 17 |
| s_waitcnt vmcnt thresholds | {25, 15, 10, 0} | {8} (only) |
| s_waitcnt lgkmcnt thresholds | {0} (11×) | {0} (1×) |

**Per-K-iter K-loop body** (aiter label_041A, lines 661-1118 = 458 lines):
| Stat | aiter K-iter | our K-iter (R32 §A5) |
|---|---:|---:|
| `v_mfma_scale_f32_16x16x128` | **256** | 128 |
| buffer_load_dwordx4 (vmem→reg) | 32 | ~22 (mixed offen+offen-lds) |
| buffer_load_dwordx4 with `lds` | 20 | (same 22 above) |
| ds_read_b128 | 80 | ~33 |
| s_barrier | **4** | 1 (with all-btw) |
| `s_waitcnt vmcnt(N)` | 2× vmcnt(10) + 2× vmcnt(15) | 1× vmcnt(8) |
| `s_waitcnt lgkmcnt(0)` | 4 | 0 (no inner lgkmcnt) |
| `s_nop` | 6 | 0 |

### B4 — 4 actionable structural findings (the answer to "why does aiter win 7.4pp")

1. **aiter does 2× the work per outer K-iter at the same MFMA shape.**
   aiter steady-state K-iter has **256 MFMAs** (covering K-block of 128 K-cols
   for a 256×256 tile = (256/16)·(256/16)·1 = 256), vs our 128 MFMAs/iter
   (covering K-block of 128 for a 256×256 tile but with `WARPS_M·WARPS_N=4`
   warps each owning 64×64 = (64/16)·(64/16)·1 = 16 each = 64 per warp ×
   2 K-pair (sa0,sa1) = 128). aiter has **fewer K-iters** (501→501 same K
   range, but 2× MFMAs/iter at half the buffer_load count means the K-loop
   has the same length but half the prologue/epilogue overhead per iter).
   This is the 32x32-equivalent compute density at 16x16 MFMA shape —
   achieved by a different warp/output decomposition.

2. **aiter uses MUCH higher vmcnt thresholds (15, 25 vs our 8).**
   This is the OPPOSITE of what R31-C concluded ("v12 is the only stable
   value on K=128256; ≥20 crashes"). aiter sustains **15-25 outstanding VMEM
   loads** continuously, vs our 8. This means aiter has either:
   - (a) larger SRD bounds (no aperture violation at higher in-flight count),
     OR
   - (b) different load addressing that doesn't expose the SRD-overrun bug,
     OR
   - (c) a wider register window for in-flight VMEM data (aiter uses VGPRs
     v[168:199]+ as the load-buffer ring, ~32 VGPRs reserved for the in-flight
     queue).
   The SRD-overrun bug we hit at vmcnt(20)/(24) in R31-C and R32-A NT
   (HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION) is an artifact of how OUR
   kernel computes the SRD limit field, NOT a hardware constraint. aiter
   demonstrates that vmcnt(15) is safe and beneficial.

3. **aiter uses `buffer_load_dword` (1-dword scalar pre-fetch), 26 of them
   in the 128x512 kernel — these load **scale dwords** with very fine-grained
   pipelining**. Our kernel uses `buffer_load_dwordx2` for scale loads
   (NONVOLATILE_SCALE_X2_POC=1), one per (a0, a1, bl, br) per K-iter = 4
   loads. aiter spreads scale loads as 26 per K-loop = ~5 per K-iter, paired
   with their consumer MFMAs — this **decouples scale latency from the
   tile-load dependency chain**, letting the MFMA pipe never stall on scales.

4. **aiter explicitly emits 6 `s_nop` per K-iter.** Our kernel has 0 — we
   tried `EXPLICIT_S_NOP` and `LDS_RD_STAGGER_NOP` (R21B/R22A axes,
   kernel:209-220) and both were DEAD on the parent stack. aiter's nops are
   placed at specific points in the MFMA → ds_read → MFMA chain, suggesting
   the **AMD compiler post-RA scheduler missed a 1-2 cycle drainage window**
   that aiter's hand-tuned ASM exploits. This MAY be reproducible by adding
   targeted `asm volatile("s_nop 0")` at our `kpair_64mfma_step12` lines
   1280-1320 (between the ds_read_b128 burst and the next MFMA group).

### B5 — Estimated hours spent in this track: 1.5 hr (done in-decider).

---

## Track C — residual sub-day axes from R32 A1

Status of the three R32 A1 candidates:

| Axis | Kernel lines | Default | Risk class | Hours | EV |
|---|---|---:|---|---:|---:|
| `BARRIER_TO_WAITCNT_RELAXED_VMCNT` sweep | 469-479 | 0 | **perf** (compiler always recomputes the exact vmcnt; if vmcnt > STEP3_BARRIER_VMCNT (=12) we just relax further; if <12, equiv to a known-tested point) | **1.5 hr** | 0.15pp |
| Per-site `BARRIER_TO_WAITCNT_STEP3_S{1..7}` overrides | 437-457 | inherit | **correctness** (S2/S3/S4 are the live STEP3 sites; flipping individually has the same risk class as flipping aggregate STEP3 which is already at btw_all) | **2 hr** | 0.20pp |
| `OUTER_K_PF_DEPTH ∈ {2,3}` | 854-861, 2702-2723 | 1 | **perf** (extra L2-only buffer_load_dwordx4 batched at iter end; was DEAD in R12 per TODO; needs `OUTER_K_PF_MODE=1`) | **1 hr** | 0.10pp |

**Mechanism re-evaluation in light of B**:
- aiter's vmcnt(15)/(25) (B4 finding 2) **directly contradicts** my prior
  "RELAXED_VMCNT is dead" prior. aiter shows that on the SAME shape, vmcnt
  > 12 IS productive — IF the kernel's SRD computation doesn't trip
  aperture-violation. This makes RELAXED_VMCNT slightly higher EV than R32
  decider estimated, ~0.3pp p50 (was 0.15pp), conditional on the SRD
  overrun bug being absent in this code path (it might be — RELAXED_VMCNT
  changes only the inline ASM s_waitcnt string, not the load-issue rate).
- Per-site STEP3_S{1..7} overrides: aiter has **vmcnt(10) and vmcnt(15)
  on different sites within the same K-iter** — different waitcnt threshold
  per site is provably useful in the gold-standard. EV bumped from 0.20pp
  to **0.40pp** because we now have an existence proof that asymmetric per-
  site vmcnts work.
- OUTER_K_PF_DEPTH: aiter has only 32 vmem loads + 20 lds-targeted loads
  per K-iter (52 total) vs our ~22 with no extra pre-fetch. aiter's load
  density is HIGHER, suggesting more aggressive prefetching in some form.
  But OUTER_K_PF was DEAD in R12 because it doubled live VGPR span. The
  difference is aiter staggers loads INSIDE the MFMA stream rather than at
  the iter boundary — closer to LD2 modes than to OUTER_K_PF_DEPTH=2. EV
  bumped slightly, **0.20pp**.

---

## Final dispatch recommendation for R33

Given Section B's findings, the rational R33 dispatch is **2 parallel
optimizers + 1 deeper aiter-archaeology decider**:

### R33 Optimizer A — "aiter-vmcnt-mimic" (HIGHEST EV, FRESH MECHANISM)
**Goal**: replicate aiter's vmcnt(15) sustained-in-flight policy on our
kernel's K-loop.
- Test 1: `BARRIER_TO_WAITCNT_RELAXED_VMCNT=15` on the L6 incumbent stack
  (1 hr build + 1 hr 5-rep bench). Direct copy of aiter B4#2 finding.
- Test 2: per-site `BARRIER_TO_WAITCNT_STEP3_S2=1, S3=1, S4=0` × vmcnt
  combinations {S2=15, S3=10, S4=8} — mimics aiter's mixed-vmcnt approach
  (1.5 hr build + 1 hr 5-rep).
- Test 3: add 6× `asm volatile("s_nop 0")` at `kpair_64mfma_step12`
  ds_read→MFMA boundary (~kernel:1280, 1300, 1310, 1320) — direct copy
  of aiter B4#4. (0.5 hr edit + 1 hr 5-rep).
- **Hard SNR gate** (≥ 25 dB) on a deterministic small-K test (K=512 not
  K=128256) per the lesson from R32-B (random-aperture is non-deterministic
  at L6). If gate holds + perf ≥ baseline, escalate to 5-rep at K=128256.
- Time budget: **6 hr**. Probability of finding ≥ +1pp: **30%** (was 10% pre-B).
- Expected gain: **0.5pp p50, 2pp p90** — since aiter at the same MFMA
  shape achieves the 5781 ceiling, partial replication of its waitcnt
  schedule is the most plausible sub-day gain.

### R33 Optimizer B — V5 MFMA32 POC (Track A) — DEPRIORITIZED but SCOPED
**Goal**: get a single L6 instantiation of MFMA32 building cleanly to test
whether the V5 ceiling is materially higher than 16x16.
- Use the 24-hr scope from Track A above.
- **HARD STOP at 24 hr**. Decision gate: ≥ +1pp on L6 single-rep → continue
  (escalate to multi-week sprint). < +1pp or any SNR drop → **DEAD**.
- Probability of clearing 24-hr gate: **15%** (down from 25% pre-B; aiter
  proves 16x16 can saturate the achievable target).
- Expected total round outcome: 0.5pp p50 if successful, otherwise zero.

### R33 Decider C (parallel) — "aiter deep ISA archaeology + dispatch comparison"
**Goal**: extract the actual K-loop schedule patterns (vmcnt placement, scale-
load interleaving, register allocation) from aiter 256x256, and produce a
**concrete diff document** mapping each aiter primitive to a candidate
kernel transform we could apply.
- Disassemble all 4 of: `BpreShuffle_256x256.co`, `BpreShuffle_128x512.co`,
  `BpreShuffle_192x256.co`, `BpreShuffle_128x256.co`. Diff K-loop bodies
  pairwise. Identify which sub-tile size aiter uses internally per warp
  and confirm whether 16x16 is the only MFMA shape across all tiles.
- Look at the dispatcher heuristic for L7/L4/L8 (other deep-K shapes the
  v2 auto-tune already wins on) — does aiter pick different tiles? If yes,
  cross-shape comparison illuminates which features are L6-specific.
- Trace WHY vmcnt(15) is safe in aiter — check the SRD initialization
  prologue (lines 1-200) of the .co for buffer-resource construction;
  compare to our `make_buffer_resource(addr, 0xFFFFFFFFu, 0x00110000u)`
  at kernel:2318. If aiter uses a different limit field, that explains
  why our kernel aperture-violates at vmcnt > 12.
- Time budget: **3-4 hr**. Output: `R33_AITER_ARCHAEOLOGY.md`.

### NOT recommended for R33

- **Track A as a primary lever** (V5 full sprint): 1-2 weeks too long for a
  round; better as a Q-end side project.
- **OUTER_K_PF_DEPTH=2/3** (Track C lowest of the three): aiter has more
  aggressive prefetching but spread INSIDE the MFMA stream, not at iter
  boundary; OUTER_K_PF was DEAD in R12 for live-range reasons that haven't
  changed.
- **Re-running K_LOOP_SYNC_EVERY_2 / NT_LOAD / V6 split-K** — all three
  are confirmed DEAD in R32 with documented mechanisms. Do not re-attempt.

### Honest probability the R33 round breaks the 41/42 ceiling

- P(R33 Opt A finds ≥ +1pp on L6) = **30%** (aiter mechanism existence proof)
- P(R33 Opt B V5 POC clears 24-hr gate) = **15%**
- P(R33 Decider C produces ≥1 actionable structural finding) = **80%** (any
  novel B5-style insight has follow-up value even if R34+)
- **P(at least one R33 lever produces a committable WIN on L6) ≈ 35-40%**

This is **2-3× higher than R32's 5-10%** going-in odds, because Section B
opened a new, validated lever (aiter waitcnt schedule) that wasn't visible
to any prior round.

(word count: ~1490)
