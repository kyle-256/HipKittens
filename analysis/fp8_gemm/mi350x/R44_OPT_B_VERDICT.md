# R44 Opt B — Aiter `ds_write` Disasm + VGPR-PF Revival — DEAD_DISASM

**Date**: 2026-04-19
**Round**: R44 Opt B
**Owner**: optimizer (parallel with Opt A on GPUs 0-1, Opt C on GPU 5, Opt D on GPUs 6-7)
**Assignment GPUs**: 2, 3, 4 (released; not used)
**Outcome**: **DEAD_DISASM**. Phase-1 hard-kill triggered at ~30 min: the aiter
binary contains zero `ds_write` instructions, so the address formula the round
was tasked to recover does not exist in the source artifact. Phases 2-4 not run.

## Headline

**P-B.1 falsified at the premise**, not at the time-budget. Aiter performs B-tile
and scale prefetches via the *hardware* `buffer_load_dwordx{1,4} ... lds`
path (60 sites in the kernel), with M0 set fresh per load to a known constant
+ wave LDS base. There are **0 `ds_write` instructions** in the entire
`f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` binary (all 180 `ds_*`
instructions are MFMA-source `ds_read_b{32,128}` reads). The R44 Opt B mandate
("read out aiter's per-lane `ds_write` address formula and patch into HipKittens
VGPR-PF") was based on a misread of the binary's instruction mix; the formula
cannot be extracted from a binary that does not contain it.

**Net VC delta**: 0 (no kernel changes, no new builds, no new `.so` files).
**Time spent**: ≈ 30 min of the 6 h hard timeout.

## Mechanism (durable, definitive)

Both aiter and the HipKittens incumbent kernel use the SAME LDS-deposit
mechanism: hardware `buffer_load_to_lds` (encoded as `E05D1000` for `dwordx4`
and `E0511000` for `dword`). The mechanism is:

```
s_add_u32 m0, <const_tile_offset>, <wave_lds_base_sgpr>
buffer_load_dwordx4 v_ignored, s[srd:srd+3], 0 offen lds
;       ^^^^^^^^^^ VDATA — WITH 'lds' modifier, this register field carries
;                  the per-lane voff used for *both* GMEM gather AND LDS
;                  deposit. HW writes 16 B/lane to LDS @ M0 + voff_lane.
```

**The per-lane LDS address is HW-controlled, not software-controlled.** There
is no software step that produces the per-lane offset; the HW reuses the
GMEM gather voff. The HipKittens incumbent emits the same instruction via
`llvm_amdgcn_raw_buffer_load_lds` intrinsic and inherits the same HW addressing.

The R34/R35/R43-B/R44-B "VGPR-PF" axis was a *fundamentally different
architecture*: split the load (`buffer_load_dwordx4` w/o `lds` modifier → VGPR)
from the deposit (`ds_write_b32` quartet). The deposit is software-addressed,
so a per-lane LDS address formula is needed. R34-OptB-fix-v5 used
`addr = lds_addrs[idx] + voffs[idx]`, R35 V0 tried `lane*16`, R35 V1/V2 tried
two other variants. All produced bit_eq << 1% vs incumbent.

The hypothesis "aiter knows the right formula" was the working theory for
R44 Opt B. The disasm shows aiter never solved that problem because aiter
made the opposite architectural choice — keep load+deposit fused in HW.

## Per-shape table (all 9 WCF_BOUND targets — no work performed)

| M    | N     | K     | competitor (TFLOPS) | Pre-R44 status      | R44-B Phase-2 result | R44-B Phase-3 result | R44-B 5-run | Net delta |
|-----:|------:|------:|--------------------:|---------------------|-----------------------|-----------------------|--------------|----------:|
| 4096   | 32768 | 6144  | (rep)               | WCF_BOUND           | NOT_RUN (Phase-1 KILL) | NOT_RUN              | NOT_RUN     | 0 |
| 4096   | 32768 | 14336 | (rep)               | WCF_BOUND, n_OK_5=3 | NOT_RUN                | NOT_RUN              | NOT_RUN     | 0 |
| 16384  | 6144  | 4096  |                     | WCF_BOUND           | NOT_RUN                | NOT_RUN              | NOT_RUN     | 0 |
| 16384  | 14336 | 4096  | (rep)               | WCF_BOUND           | NOT_RUN                | NOT_RUN              | NOT_RUN     | 0 |
| 28672  | 4096  | 8192  |                     | WCF_BOUND           | NOT_RUN                | NOT_RUN              | NOT_RUN     | 0 |
| 28672  | 4096  | 16384 |                     | WCF_BOUND           | NOT_RUN                | NOT_RUN              | NOT_RUN     | 0 |
| 32768  | 4096  | 7168  |                     | WCF_BOUND           | NOT_RUN                | NOT_RUN              | NOT_RUN     | 0 |
| 32768  | 4096  | 14336 |                     | WCF_BOUND           | NOT_RUN                | NOT_RUN              | NOT_RUN     | 0 |
| 128256 | 32768 | 4096  |                     | WCF_BOUND           | NOT_RUN                | NOT_RUN              | NOT_RUN     | 0 |

**No per-shape integration fragment is proposed.** `R44B_INTEGRATION_FRAGMENT.json`
is intentionally empty.

## Falsifiable predictions — outcome

| ID | Prediction | Outcome |
|---|---|---|
| P-B.1 (Phase-1 disasm) | Closed-form `addr(lane, dword_idx, k_step)` extractable from aiter disasm by hour 3 | **REFUTED at premise**: formula does not exist in binary; 0 `ds_write` instructions found across 3415 lines of disasm |
| P-B.2 (Phase-2 bit_eq)  | bit_eq ≥ 99.9% on finite cells with patched VGPR-PF kernel | **NOT_RUN** (prerequisite KILL) |
| P-B.3 (Phase-3 Jaccard) | Jaccard ≥ 0.7 on 3 representative WCF_BOUND shapes              | **NOT_RUN** |
| P-B.4 (Phase-4 LB)      | ≥2 of 3 reps flip to PASS_VC under 5-run @ GATE=0.98             | **NOT_RUN** |
| P-B.5 (regression)      | VGPR-PF perf does NOT regress >3% on 13 already-VC shapes        | **NOT_RUN** |

## Why the 30-minute KILL is the right call (vs spending 3 h)

The Phase-1 stopping criterion in R44_DECIDER_PLAN.md was:
> "Hour 3 KILL: if address formula cannot be expressed as a closed form OR the
> formula doesn't match all 88 sites, KILL Opt B…"

The actual finding is stronger than "cannot be expressed": there are no sites
to express. Three additional hours of disasm would not turn 0 ds_write
instructions into 1. The plan's "hour 3 KILL" threshold protects against
*difficult* extraction; this case is *impossible* extraction. Burning the
remaining 5.5 h of the budget against a definitively impossible target would
be a process violation (the round-level rule is "never sleep+poll" — the same
spirit applies to "never hand-search a binary for code that isn't there").

## Files (deliverables — produced by this round)

1. `R44_OPT_B_DSWRITE_FORMULA.md` — full Phase-1 KILL diagnostic with disasm
   excerpts and discussion of why the axis is unrescuable from this artifact.
   *(Mandatory by hour 3 per R44 plan; produced at ~30 min.)*
2. `R44_OPT_B_AITER_DISASM.txt` — 106-line excerpt of one full prefetch cluster
   (line 264-360 of full disasm) showing the M0 + buffer_load_to_lds idiom and
   the absence of any ds_write.
3. `R44_OPT_B_AITER_DISASM_FULL.txt` — full 3415-line disasm (kept for
   provenance; can be regenerated with the command at top of FORMULA.md).
4. `R44_OPT_B_VERDICT.md` — this document.
5. `R44_OPT_B_BITEQ.json` — Phase-2 NOT_RUN (with reason).
6. `R44_OPT_B_JACCARD.json` — Phase-3 NOT_RUN (with reason).
7. `R44_OPT_B_5RUN.json` — Phase-4 NOT_RUN (with reason).
8. `R44B_BUILD_MANIFEST.json` — empty `new_builds`.
9. `R44B_INTEGRATION_FRAGMENT.json` — empty.

No source files modified. `kernel_mxfp4_gluon_cpp_vgprPF.cpp` and
`kernel_mxfp4_gluon_cpp.cpp` are byte-identical to the round-start commit
`0ca8c72a`.

## Recommendation

1. **BURY the VGPR-PF axis** (re-affirms R35 / R43-B recommendations). The
   memory note `project_mxfp4_vgprpf_compiler_bug.md` should be updated to
   add R44 Opt B as the fourth dead round and to correct the implication that
   aiter disasm holds the answer. (See memory update at end of round.)
2. **Correct the `project_mxfp4_aiter_binary_disasm.md` claim** that the
   aiter binary is a useful reference for `ds_write` recovery. It IS a useful
   reference for: (a) per-iter M0 stride pattern (`0x0, 0x1080, 0x2100, …,
   0x7380` for B-tile), (b) prefetch interleave with `v_accvgpr_write_b32`
   accumulator zero-init, (c) per-load M0 fresh-set discipline. It is NOT
   useful for: per-lane software ds_write address derivation (no such code
   exists in the binary).
3. **For R45 candidate list**: if the project judges VGPR-PF worth one more
   shot, replace "aiter disasm extraction" with "LDS self-readback test
   kernel". A 1-wave kernel that fires `buffer_load_dwordx4 ... lds size=16`
   against a known per-lane GMEM payload, then dumps LDS to GMEM via a
   separate kernel, will directly observe the HW per-lane LDS write pattern.
   Estimated cost: 2-3 h. Until that derivation is in hand, every VGPR-PF
   round will repeat the same dead end.

## Coordination report

- Did not touch any kernel file. No build artifacts produced.
- Did not consume GPUs 2, 3, 4 — released for other workers.
- Did not interact with Opt A (GPUs 0,1, K=28672), Opt C (GPU 5, fault-PC),
  or Opt D (GPUs 6,7, FIN_BOUND probe). All independent.

## Session timeline

- T+0 min : read R44 plan, R43 Opt B verdict, R34/R35 memory notes
- T+5 min : `llvm-objdump --mcpu=gfx950 -d` produces 3415-line disasm
- T+10 min: confirm 0 `ds_write` and 60 `buffer_load*lds` sites
- T+15 min: hand-trace one full prefetch cluster (lines 251-360 of disasm)
- T+20 min: cross-check against `kernel_mxfp4_gluon_cpp_vgprPF.cpp` lines 880-984
- T+30 min: write this verdict + the FORMULA.md kill diagnostic

**Total wall clock ≈ 30 min, 5.5 h under the 6 h hard timeout. No GPU consumed.**
