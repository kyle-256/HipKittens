# R44 Opt B Phase 1 — Aiter `ds_write` Formula Extraction — KILL

**Date**: 2026-04-19
**Worker**: R44 Opt B
**Phase**: 1 (disasm-only, CPU)
**Wall-clock spent**: ~30 min of the 3 h Phase-1 budget
**Outcome**: **DEAD_DISASM (P-B.1 failed: no formula exists in the binary)**

## Headline

The R43 Opt B verdict and the R44 Opt B mandate both rest on a stated assumption:
"the aiter binary contains the per-lane `ds_write` address formula we need to
make HipKittens' R34 VGPR-PF kernel produce correct LDS layout." **That assumption
is false.** Aiter's `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` contains
**zero `ds_write` instructions** of any kind. The disasm decisively shows aiter
uses the *hardware* `buffer_load_dwordx4 ... lds` (and `buffer_load_dword ... lds`
for scales) path — the same path the HipKittens incumbent already uses. There
is no software ds_write formula to recover, because aiter never executes one
on the prefetch path.

This terminates Phase 1 at the falsifiable kill criterion: **P-B.1 (closed-form
addr formula by hour 3) is unsatisfiable, not just unmet.** Phases 2-4 (kernel
patch, bit_eq oracle, Jaccard oracle, 5-run consensus) are not run.

## Hard evidence

Aiter binary disassembled with:

```
/opt/rocm/llvm/bin/llvm-objdump --mcpu=gfx950 -d \
  /shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co \
  > R44_OPT_B_AITER_DISASM_FULL.txt
```

3415 lines, kernel symbol `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E`
at PC `0x2c00`.

Counts of LDS-touching instructions (full kernel):

| Pattern | Count |
|---|---:|
| `ds_write` (any width)                                    | **0**  |
| `ds_store` (any width)                                    | **0**  |
| `ds_read_*` (any width — for MFMA consumption)            | 180    |
| `buffer_load_dwordx4 ... lds` (B-tile HW load-to-LDS)     | 48     |
| `buffer_load_dword ... lds` (scale HW load-to-LDS)        | 12     |
| Total HW buffer→LDS sites                                 | 60     |

(The "88 buffer_load_dwordx4/ds_write sites" cited in the R44 plan was a count of
all `buffer_load_dword*` and `ds_*` instructions. The plan's restatement that
"`ds_write` sites" were ~88 in the binary was incorrect — those are `ds_read`
sites for MFMA-consumption fan-out.)

## What the disasm actually shows: aiter's prefetch idiom

Excerpt from one full per-iter B-tile + scale prefetch cluster
(see `R44_OPT_B_AITER_DISASM.txt` for the 106-line region; here is the
load-relevant subset around PC `0x30D4` at line 264):

```
; B-tile prefetch — 8 successive 16B-wide HW buffer→LDS loads.
; Each load uses M0 to set the LDS write base; data goes from GMEM directly
; to LDS at addresses M0 + (per-lane offset, applied by the HW).
s_add_u32 m0, 0,       s59
buffer_load_dwordx4 v212, s[12:15], 0 offen lds        ; opcode E05D1000

s_add_u32 m0, 0x1080,  s59
buffer_load_dwordx4 v213, s[12:15], 0 offen lds

s_add_u32 m0, 0x2100,  s59
buffer_load_dwordx4 v214, s[12:15], 0 offen lds

s_add_u32 m0, 0x3180,  s59
buffer_load_dwordx4 v215, s[12:15], 0 offen lds

; ─── interleaved scale prefetch ───
s_add_u32 m0, 0,       s60
buffer_load_dword   v222, s[20:23], 0 offen lds        ; opcode E0511000

s_add_u32 m0, 0x4200,  s59
buffer_load_dwordx4 v216, s[12:15], 0 offen lds

s_add_u32 m0, 0x5280,  s59
buffer_load_dwordx4 v217, s[12:15], 0 offen lds

s_add_u32 m0, 0x6300,  s59
buffer_load_dwordx4 v218, s[12:15], 0 offen lds

s_add_u32 m0, 0x7380,  s59
buffer_load_dwordx4 v219, s[12:15], 0 offen lds

s_add_u32 m0, 0x400,   s60
buffer_load_dword   v223, s[20:23], 0 offen lds
```

**Key observations**:

1. The destination VGPR (`v212`-`v219`, `v222`, `v223`) listed by the assembler is
   the *per-instruction VDATA field*, but with the `lds` modifier set, the HW
   ignores VDATA-as-destination — it consumes the per-lane `voffset` from VDATA
   and deposits the gathered data into LDS at `M0 + voff_lane`. (See the `0x100D`
   bits in the encoding — `1000` = lds-direct.) These VGPRs are not actually
   written by the load.
2. M0 is set to `wave_lds_base + small_const_tile_offset` immediately before
   each load. The constants `0x0000, 0x1080, 0x2100, …, 0x7380` follow a
   deterministic stride of `0x1080 = 4224 B = 256 + 256·16` between the 8
   B-tile prefetches in the cluster (matches the `BN=256, K_pack=64` macro tile).
3. Scale loads use a smaller stride (`0x400 = 1024 B`) and only 2 loads per
   cluster — matches `1 scale per 32-K group, BN×K-tile / 32`.
4. Per-lane addressing is *entirely under HW control*. There is no software
   step that computes a per-lane LDS address. The `voff` register passed to
   the load (in VDATA position) is the same `voff` used to gather GMEM via the
   SRD `s[12:15]`; HW reuses it as the in-tile LDS index.
5. **There is no `ds_write` anywhere in the kernel.** All `ds_*` ops are reads
   that fan the LDS payload into MFMA source VGPRs (180 reads, all `ds_read_b128`
   or `ds_read_b32`).

## Why R34 VGPR-PF cannot be fixed by "copying aiter's ds_write formula"

The R34/R35/R43-B fork's hypothesized fix path was:

```
emit_one_pf_vgpr      ; buffer_load_dwordx4 (NO lds modifier) → scratch float4 VGPR
vgpr_keepalive        ; defeat compiler clobber via "+v" constraint
emit_one_pf_dswrite   ; ds_write_b32 quartet @ uniform_M0 + voff_lane
```

The third step (software `ds_write_b32` quartet) is the broken piece. The
hypothesis was that aiter contained a known-correct formula for the per-lane
LDS address `addr(lane, dword_idx, k_step)` — extractable from disasm and
substitutable into the third step.

The disasm refutes the hypothesis at its premise: aiter never performs that
third step. Aiter's pipeline is simply

```
                              ┌───── HW path (atomic load+deposit) ─────┐
[GMEM via SRD s[12:15]]  →    buffer_load_dwordx4 ... lds  →  [LDS via M0+voff]
                              └────────────────────────────────────────┘
```

— which is *exactly* the pipeline HipKittens' incumbent already uses (via the
`llvm_amdgcn_raw_buffer_load_lds` intrinsic, which lowers to the same
`E05D1000` opcode). The two kernels differ in M0/voff scheduling, prefetch
depth, and barrier discipline — but the LDS deposit *mechanism* is identical
in HW. There is no software-side formula to copy because there is no
software-side step.

## Why the R34 VGPR-staged path is unrescuable from this binary

If the goal is to keep the load and the LDS deposit *separate* (so the load
can complete into a VGPR while other work proceeds, then deposit on demand),
the deposit MUST be a `ds_write`. The per-lane LDS address that `ds_write_b32`
takes is determined by software, not HW. The aiter binary cannot tell us
what the "right" formula is for that hypothetical software step because aiter
has chosen NOT to make that split. The architectural choice is: either

  (a) keep the HW atomic load+deposit (aiter / HipKittens incumbent) — get
      automatic per-lane addressing for free, but pay the M0 commit-pointer
      stall, or
  (b) split into VGPR-load + `ds_write` — pay the cost of computing per-lane
      LDS addresses in software, but gain scheduling freedom for the load.

The question "what address formula does (b) need to produce identical LDS
layout to (a)?" is a *derivation* question, not an *extraction-from-binary*
question. The answer for `buffer_load_dwordx4 ... lds size=16` per AMD ISA
is documented as

  `LDS_addr_lane = M0 + lane_id * 16`        (lane-strided, contiguous payload)

and the previously-tried R35 hypothesis `lane*16` (R35 V0 attempt) is ALREADY
the documented-correct one. R35 V0 produced bit_eq=0.09%, not 100% — meaning
the documented formula is itself wrong, OR the M0 register has not been set
to the value the consumer (subsequent `ds_read_b128`) expects.

**Most likely: the failure is not in the address formula but in the M0
assignment.** Aiter sets M0 *fresh per load* (`s_add_u32 m0, <const>, s59`).
The R34 VGPR-PF deposit path uses a single uniform `lds_addrs[idx]` per slot
that is `readfirstlane`'d from `voff` — losing the per-load M0 stride. Even
with the "right" software per-lane formula, the *base* LDS region we deposit
into is wrong, so the consumer reads stale or misaligned data → bit_eq ≈ 0.

## Why P-B.1 is unsatisfiable in any reasonable continuation

To "extract a formula" from a binary that contains no instances of the formula
is impossible. Three additional 8-hour rounds spent on the same disasm would
not change the count of `ds_write` instructions (zero). The failure is not
"we ran out of time" — it is "the source we were told contains the answer
does not contain the answer."

The plan note in R44_DECIDER_PLAN.md ("Aiter `.co` … 88 ds_write/buffer_load
hits") and the R43 verdict's recommendation #3 ("read out aiter's ds_write
address formula from disasm") both rest on the same misread of the binary's
instruction mix.

## What WOULD answer the question (deferred to R45+)

To make the R34/R35 VGPR-staged path work, the *correct* derivation pipeline is:

1. Write a tiny **LDS self-readback test kernel**: launch with one wave,
   set up a known M0, fire `buffer_load_dwordx4 ... lds` against a known
   (per-lane-deterministic) GMEM payload, then dump LDS contents back to GMEM
   via a separate kernel. The HW write pattern (per-lane → LDS-byte map) is
   then directly observable.
2. From that map, derive `addr(lane, dword_idx) = ?` and check it against the
   per-lane M0 + voff scheme.
3. Only then, patch `kernel_mxfp4_gluon_cpp_vgprPF.cpp` `emit_one_pf_dswrite`
   with the observed formula AND restore aiter's per-load M0 fresh-set discipline
   (currently the kernel uses one uniform `lds_addrs[idx]` per slot — pre-baked
   at function entry, not per-load).

Estimated cost: 2-3 h to write the self-readback test kernel + interpret. This
is a sensible R45 task IF the VGPR-PF axis is judged worth one more round
after R44 closes. Given R34 + R35 + R43-B + R44-B (this report) all dead, the
project_mxfp4_vgprpf_compiler_bug.md memory note's "BURY the axis" recommendation
should stand.

## Falsifiable predictions — outcome

| ID | Prediction | Outcome |
|---|---|---|
| P-B.1 (Phase-1) | Address formula extractable as closed function of `(lane, dword_idx, k_step)` from aiter disasm by hour 3 | **REFUTED** — formula does not exist in binary (zero `ds_write` instructions). Time spent ≈ 0.5 h of 3 h budget. |
| P-B.2 (Phase-2 bit_eq) | With recovered formula, VGPR-PF kernel bit_eq ≥ 99.9% vs incumbent | **NOT_RUN** (Phase-1 KILL) |
| P-B.3 (Phase-2 race) | With bit_eq fixed, Jaccard ≥ 0.7 on identical-input 3-rep probe | **NOT_RUN** (Phase-1 KILL) |
| P-B.4 (leaderboard) | ≥2 of 3 representative shapes flip to PASS_VC | **NOT_RUN** (Phase-1 KILL) |
| P-B.5 (perf regression) | VGPR-PF perf does NOT regress >3% on 13 already-VC shapes | **NOT_RUN** (Phase-1 KILL) |

## Recommendation

1. **Do not re-attempt VGPR-PF axis without the self-readback test kernel** described
   above. Three rounds (R34, R35, R43-B) plus this one have failed; the failure
   mode is now fully understood and durable.
2. **Update R45 candidate list**: replace any "extract aiter ds_write formula"
   item with "write LDS self-readback test kernel for `buffer_load_dwordx4 lds size=16`
   on gfx950" — this is the actual pre-requisite to any future VGPR-PF attempt.
3. **Memory file update**: add this round's evidence to
   `project_mxfp4_vgprpf_compiler_bug.md` and `project_mxfp4_aiter_binary_disasm.md`.
   The aiter-disasm note currently implies the binary is informative for
   ds_write recovery — this is misleading and should be corrected.

GPUs 2, 3, 4 released for other R44 workers' overflow.
