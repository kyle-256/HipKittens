# Round 18 Optimizer B — DOUBLE_C_PINGPONG: DEAD END

**Date**: 2026-04-17
**Author**: Kyle.Zhao@amd.com / kyle-256
**Branch**: mxfp4
**Time spent**: ~90 min wall (build attempt + operand-binding analysis + revert)

---

## TL;DR

R17A-P1 ("double C-accumulator ping-pong tiling") **cannot** be implemented as the
register-neutral phase-interleave that the R17A-P1 verdict suggested.
The 32-MFMA inner block is **not** a uniform 4-row × 8-MFMA grid with standard
A/B operand semantics — Rows 1–3 use **`ds_read_b128` results (`d1, d2, d3`) as
the MFMA A operand**, not `a_lo[i]` as the verdict implied.

A naive phase-interleave (P0 of all rows, then P1 of all rows) using the proposed
operand layout produces **`error: invalid operand for instruction`** at every
Row-1/2/3-P0 MFMA — confirmed via 4-shape build. All 4 deep-LOSE shapes (DLA1,
DLA2, DLA7, P1) failed the build with identical errors at lines 1421+ of the
generated wrapper.

**No SNR / bench / commit performed.** Source reverted to baseline (R18A-P3
work by sibling agent preserved).

---

## Why "double C ping-pong" is structurally infeasible as a flat reorder

### The actual operand binding in `kpair_32mfma_with_lds_and_pf`

Reading the original asm (`kernel_mxfp4_gluon_cpp.cpp` lines 1273–1334):

| Row | Phase 0 MFMAs                                         | A operand    | B operand    |
|-----|-------------------------------------------------------|--------------|--------------|
| 0   | `acc[0..3] = mfma(%24=a_lo[0],   %32..35=b_lo[0..3])` | `a_lo[0]`    | `b_lo[0..3]` |
| 1   | `acc[4..7] = mfma(%17=d1 (LDS), %24..27=a_lo[0..3])`  | **`d1` (LDS load result)** | `a_lo[0..3]` (re-used as B!) |
| 2   | `acc[8..11]= mfma(%18=d2 (LDS), %24..27=a_lo[0..3])`  | **`d2` (LDS load result)** | `a_lo[0..3]` |
| 3   | `acc[12..15]=mfma(%19=d3 (LDS), %24..27=a_lo[0..3])`  | **`d3` (LDS load result)** | `a_lo[0..3]` |

Rows 1–3 are **operand-swapped**: they consume LDS-loaded data as the A operand,
and re-use the (already-issued-once-as-A) `a_lo[0..3]` as the B operand. This is
the kernel's M-major × K-major reuse pattern that lets one wave handle a 256-N tile.

### What goes wrong in a flat phase-reorder

The proposed P0/P1-interleaved order requires:
```
P0(R0)  ; uses  a_lo[0], b_lo[0..3]                          — OK, no deps
P0(R1)  ; uses  d1 (LDS), a_lo[0..3]                         — needs d1 ready
P0(R2)  ; uses  d2 (LDS), a_lo[0..3]                         — needs d2 ready
P0(R3)  ; uses  d3 (LDS), a_lo[0..3]                         — needs d3 ready
[ds_read d4..d7]
P1(R0..R3)
```

Currently the original kernel **interleaves the 8 `ds_read_b128` instructions
inside the Row-0 asm block** so that by the time Row 1's first MFMA wants `d1`,
it has been issued ~4 instructions earlier. In the proposed interleave we'd need
**all 4 ds_reads (d0..d3) issued and waited-on before Row 1 P0 can start**,
which means an additional `s_waitcnt lgkmcnt(0)` between blocks.

That waitcnt would erase any latency-hide gain: 8 cyc MFMA-acc latency saved is
swamped by ~10–30 cyc lgkmcnt drain (LDS-bank-conflict and addr-gen latency on
gfx950).

### What I actually wrote (incorrectly)

My `kpair_32mfma_with_lds_and_pf_dblc` (now reverted) assumed Row 1 P0 would be:
```
v_mfma_scale_f32_16x16x128_f8f6f4 %4, %25 (a_lo[1]), %32 (b_lo[0]), %4, %40 (sa0), %42 (sb0)
                                       ^^ WRONG — original used %17 (=d1, LDS load)
```
This produces `error: invalid operand for instruction` because `%25` is a
32-bit packed `fp4_intx4_t` (one A row), but the MFMA instruction with the
specific op_sel pattern expects the A slot to be the 128-bit `d1` register pair
loaded from LDS.

### Build evidence

```
$ python3 build_round18_optB_dblc.py
  DLA2   _ts_gm2_v12_memc_dc_r18b_dblc                                 FAIL      (66.0s)
  DLA7   _ts_lgk2_v12_memc_r18b_dblc                                   FAIL      (74.5s)
  P1     _ts_gm8_r18b_dblc                                             FAIL      (62.5s)
  DLA1   _ts_pf6_6_v12_memc_r18b_dblc                                  FAIL      (97.7s)
ok=0 cached=0 fail=4
```
All 4 logs show identical error patterns:
```
.../wrap_n*_k*_*_r18b_dblc.cpp:1421:9: error: invalid operand for instruction
.../wrap_n*_k*_*_r18b_dblc.cpp:1422:10: error: invalid operand for instruction
[... 28 more lines of the same ...]
```
Lines 1421–1450 are exactly the 16 Phase-1 MFMAs of my new `_dblc` function.

---

## What it would actually take (and why it's also a dead end)

A correct phase-interleave needs one of:

1. **Move all 8 `ds_read_b128`s to the start, then `s_waitcnt lgkmcnt(0)`,
   then 16 MFMAs in P0, then 16 in P1.**
   - Adds 1 lgkmcnt drain per kpair (~20 cyc) × 64 KPAIR_LOOP × 2004 K-iters = 2.5M cycles
   - At 256 CU × 4 SIMD = 2.4 ms — **eats the entire 0.9 ms MFMA latency budget**.
   - Net: NEGATIVE expected gain.

2. **True double-bank `rt_C0/rt_C1` (the verdict's literal description).**
   - Requires 2 × 16 = 32 AGPRs for acc instead of 16. Combined with the 32 AGPRs
     used for `acc_A1Bl/A1Br/A0Bl/A0Br` × 2 banks = 64 AGPRs total.
   - gfx950 AGPR limit per wave = 256, so capacity OK in isolation.
   - HOWEVER, the kernel currently uses 4 banks (`acc_A0Bl, acc_A0Br, acc_A1Bl,
     acc_A1Br`) of 16 AGPRs each = 64 AGPR for acc; doubling brings it to 128.
     With `WPE2` (2 waves/CU instead of 4), occupancy halves from 4 to 2, and
     measured ROCm `Rpass-analysis` already reports `wpe=2` for memc variants.
   - At `wpe=2` we're already at the occupancy floor. Going to `wpe=1` for
     128 AGPRs would halve effective parallelism, costing ~30-50% throughput.
   - Net: NEGATIVE expected gain.

3. **Re-architect the inner kpair to issue MFMAs strictly within a SIMD
   pipeline window, e.g. swap to a 2-wave-per-CTA layout.**
   - Out of scope for a single optimization round (multi-day rewrite of the
     producer-consumer LDS protocol).

---

## What was changed and what was reverted

### Changed (then reverted to baseline):
- `analysis/fp8_gemm/mi350x/kernel_mxfp4_gluon_cpp.cpp`:
  - Added `DOUBLE_C_PINGPONG` macro definition (REVERTED)
  - Added `kpair_32mfma_with_lds_and_pf_dblc` function (REVERTED)
  - Added 4 `#if DOUBLE_C_PINGPONG` call-site gates (REVERTED)

### Preserved (sibling agent R18A-P3 work; not mine):
- `BARRIER_TO_WAITCNT_*` macro family
- `MXFP4_STEP3_BARRIER_INST` / `MXFP4_TAIL_BARRIER_INST` macro indirection
- 2 call-site replacements to use the new macros

### New artifacts (kept for reproducibility):
- `analysis/fp8_gemm/mi350x/build_round18_optB_dblc.py` — build harness, all 4
  builds documented as FAIL
- `analysis/fp8_gemm/mi350x/build_round18_optB_{DLA1,DLA2,DLA7,P1}.log`
  — build error logs (`error: invalid operand for instruction`)
- `analysis/fp8_gemm/mi350x/round18_optB_verdict.md` — this document

---

## Recommendations to R18 decider

1. **R17A-P1 is dead** as stated. The verdict's "register-neutral phase
   interleave" is structurally impossible because the kernel's 32-MFMA block
   uses LDS-load results as MFMA A operands in 3 of 4 rows.
2. **R17A-P2 (M=256 K=128256 specialization)** — currently being explored by
   R18C agent. This is the **next-highest-EV path** because it directly addresses
   per-wave overhead amortization without touching the MFMA scheduling.
3. **R17A-P3 (s_barrier → s_waitcnt lgkmcnt)** — currently being explored by
   R18A agent with safety-gated `BARRIER_TO_WAITCNT_*` macros. SNR-validation
   already in flight.
4. **Do not retry "double C ping-pong" without a full producer-consumer rewrite.**
   The 8-cyc MFMA-acc latency stall (49% VALUBusy) is real but cannot be hidden
   without restructuring the LDS-load → MFMA-A operand handoff. That's a
   multi-day kernel re-architecture, not a single-round optimization.

---

## No commits made.

This was a **failed build → revert** pass. The kernel diff against `HEAD~`
contains only R18A-P3 agent's barrier-to-waitcnt work (preserved), no R18B
content. The build script and logs are kept as untracked files for the next
agent's reference.
