# R54 Dev E findings — CRR `v_pk_lshrrev_b32` packed-shift lever

**Cycle:** R54
**Cell:** 70B Gate/Up CRR (M=4096, N=28672, K=8192)
**Baseline:** 89.3% MX/FP8 (HEADROOM -5.7pp)
**Lever:** Replace pairs of `v_lshrrev_b32` (`>> 16` per K-pair scale shift) with
one `v_pk_lshrrev_b32` (packed 32-bit shift on adjacent VGPR pair).
**Origin:** R54 Dev A pivot recommendation: "P3/P4 pivot: `v_pk_lshrrev_b32`
packed shift to halve the cost without LDS or VGPR pressure".

## Verdict — REFUTED-EMPIRICAL-ISA (Phase 0, no kernel build needed)

**The lever is not implementable on gfx950.** The required ISA opcode
(`v_pk_lshrrev_b32`, or any packed 32-bit shift on adjacent VGPR pair) does
not exist in the gfx950 instruction set. Confirmed by direct `llvm-mc`
assembler probe against the in-tree gfx950 instruction descriptions
(LLVM 20.0.0git, AOMP-18.0-12, /opt/rocm/llvm/bin/llvm-mc).

This is a 5th-axis closure of the CRR `>> 16` scale-shift lever family,
joining the 4 prior closures:
1. R49A — host-side scale repack (REFUTED, v_perm penalty)
2. R50A — lead-distance / inter-iter live state (REFUTED, VGPR spill)
3. R53A — opsel-keyed K-phase MMA dispatch (REFUTED, 70 VGPR spill + body dup)
4. R54A — LDS-resident pre-shifted scale layout (REFUTED-EMPIRICAL, LDS
   overflow 188416 vs 163840)
5. **R54E — packed `v_pk_lshrrev_b32` instruction substitution (REFUTED-
   EMPIRICAL-ISA — opcode does not exist on gfx950)** ← this work

## Phase 0: ISA-level feasibility probe

### Step A — Confirm the shift sites exist as predicted

Built mainline baseline at 70B Gate/Up shape:
`make CXXFLAGS_EXTRA="-DM_DIM=4096 -DN_DIM=28672 -DK_DIM=8192"`
(see `r54e_results/build_70B_GateUp_baseline.log`).

Resource summary (matches r54a baseline within rounding — MXFP8_BUILD_*
defines drive minor variant differences):
- `_Z29crr_exact_8wave_scaled_kernelILb1ELi1EEv` (V1, PRESHUFFLED): 225 VGPR / 0 spill / 139264 LDS / occ 2
- `_Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv` (V1, PRESHUFFLED V2 layout): 225 VGPR / 0 spill / 139264 LDS / occ 2
- `_Z29crr_exact_8wave_scaled_kernelILb0ELi1EEv` (V1, no PQ): 225 VGPR / 0 spill / 139264 LDS / occ 2

Disassembled the gfx950 device code (clang-offload-bundler unbundle →
llvm-objdump). All three CRR kernel variants emit **exactly 6 in-place
`v_lshrrev_b32_e32 vN, 16, vN`** instructions in the K-loop body:

```text
000000027a00 <_Z29crr_exact_8wave_scaled_kernelILb1ELi1EEv...>:
  v_lshrrev_b32_e32 v178, 16, v178   ; 286D0
  v_lshrrev_b32_e32 v179, 16, v179   ; 286D4
  v_lshrrev_b32_e32 v180, 16, v180   ; 286D8
  v_lshrrev_b32_e32 v181, 16, v181   ; 286DC
  v_lshrrev_b32_e32 v182, 16, v182   ; 286E0
  v_lshrrev_b32_e32 v183, 16, v183   ; 286E4
```

Six shifts — one per `a0_scale_packs[0/1]`, `a1_scale_packs[0/1]`, `b0_scale_packs[0]`,
`b1_scale_packs[0]` (with `crr_a_pack_count=2`, `crr_b_pack_count=1` for RBM=64
RBN=32). They are **already emitted as a contiguous batch** by the LLVM
scheduler — no inter-instruction reordering for the lever to exploit. This
matches R48/R53/R54A's structural-floor diagnostic: "6× `v_lshrrev_b32`/K-pair
conditional shift block on alternate K-pairs."

Shift-site evidence: `r54e_results/crr_lb1li1_kernel_isa.txt`,
`crr_lb0li1_kernel_isa.txt`, `crr_lb1li2_kernel_isa.txt`.

### Step B — Probe gfx950 packed-shift opcodes

Direct assembler probe (full transcript:
`r54e_results/gfx950_packed_shift_probe.txt`):

| Opcode | gfx950 status |
|---|---|
| `v_pk_lshrrev_b32 v[0:1], 16, v[2:3]` | **invalid instruction** |
| `v_pk_lshlrev_b32 v[0:1], 16, v[2:3]` | **invalid instruction** |
| `v_pk_ashrrev_b32 v[0:1], 16, v[2:3]` | **invalid instruction** |
| `v_dual_lshrrev_b32 ...`              | **invalid instruction** (RDNA-only) |
| `v_pk_lshrrev_b16 v0, 16, v2`         | valid (packed 16-bit shift) |
| `v_lshrrev_b64 v[0:1], 16, v[2:3]`    | valid (cross-32-bit boundary) |
| `v_alignbit_b32 v0, v1, v2, 16`       | valid (single-output 32-bit) |
| `v_pk_mov_b32 v[0:1], v[2:3], v[4:5]` | valid (no shift) |

**There is no packed-32 shift instruction on gfx950.** CDNA4 SIMD-pair VALU
support is limited to 16-bit lanes (`v_pk_*_b16`/`v_pk_*_f16`) and dual-32-bit
move/FMA (`v_pk_mov_b32`, `v_pk_fma_f32`) — none of which can fuse two 32-bit
shifts into a single instruction.

### Step C — Why no available opcode is a viable substitute

The required transform per pack is `dst[31:0] = (src[31:0] >> 16)`, i.e. zero
high half + move byte 2/3 → byte 0/1. Each candidate gfx950 opcode was
analysed against this requirement:

**`v_pk_lshrrev_b16`**: shifts each 16-bit lane independently within a single
32-bit register. With shift=16 (lane) the lanes are zeroed (useless). With
shift=8: pack `[B3 B2 | B1 B0]` → `[0 B3 | 0 B1]` — wrong (we need
`[0 0 | B3 B2]`). Also still operates on a single VGPR — no fusion gain.

**`v_lshrrev_b64`**: 64-bit shift across both VGPRs of the pair.
`v_lshrrev_b64 [r0:r1], 16, [r0:r1]` produces:
- new r0 = `((r1 & 0xFFFF) << 16) | (r0 >> 16)` ← polluted by r1 low bits
- new r1 = `(r1 >> 16)` ← correct

Repair would need `v_and_b32 r0, 0x0000FFFF, r0` afterwards (2 instr for 2
register shifts — zero net savings, and `v_and` with 32-bit immediate would
need `s_mov_b32` + register, costing 1 SGPR + 1 extra instr).

Shift-by-48 produces `r0 = r1 >> 16, r1 = 0` — destroys r1, useless for
this site since both packs are needed afterwards.

**`v_alignbit_b32`**: single 32-bit output — equivalent to one
`v_lshrrev_b32`, no fusion. `v_alignbit_b32 dst, 0, src, 16` ≡
`v_lshrrev_b32 dst, 16, src` cycle-for-cycle.

**`v_perm_b32`**: single 32-bit output, control byte selectors — can express
the shift but cannot fuse two pack shifts into one instruction.

**`v_pk_mov_b32`**: dual 32-bit move only, no shift — no help for the shift
operation itself.

### Step D — Inline-asm path is also blocked

The mandate suggested "inline asm or `__builtin_amdgcn_*` if the compiler
doesn't auto-generate packed shifts". Inline asm is futile because the gfx950
encoder will reject the opcode (see Step B — same llvm-mc backend used by
hipcc inline asm). `__builtin_amdgcn_*` is a thin wrapper around the same
LLVM intrinsic set; there is no `__builtin_amdgcn_pk_lshrrev_b32` for gfx950
because the underlying instruction does not exist.

## Phase 1 / Phase 2 — not reached

Skipped: a build with a `MXFP8_CRR_PK_LSHRREV` macro-gated path would not
compile (assembler rejection) for the only candidate opcode that would
actually halve the cycle count. No useful empirical signal beyond Step B
above.

## Resource-projection summary

| Metric | Baseline | Lever (projected) | Delta |
|---|---|---|---|
| VGPR (CRR V1) | 225 | n/a | n/a (lever not buildable) |
| VGPR Spill | 0 | n/a | n/a |
| LDS | 139264 B | 139264 B (no LDS change planned) | 0 |
| Per-K-pair shift instrs | 6 × `v_lshrrev_b32` | 3 × `v_pk_lshrrev_b32` (target) | -3 instr/pair = -1.5%-3% projected |
| Reality | — | **unachievable on gfx950** | — |

## Reasoning

The `v_pk_lshrrev_b32` instruction-substitution lever was the last
remaining axis of the CRR scale-shift family that didn't either (a) extend
inter-iteration live state (R50A spilled), (b) restructure the K-loop body
(R53A spilled / body-dup'd), (c) repack scales at the host (R49A v_perm
penalty), or (d) consume LDS bandwidth (R54A LDS-overflowed). All four
prior axes failed for orthogonal reasons; this 5th axis fails because the
target hardware simply lacks the requested opcode.

Per the mandate's own escalation:
> "compiles but ISA shows shifts not actually paired → check if approach is
> feasible at all; may need to force via inline asm"

This case is the strict-stronger condition: not "compiler doesn't pair
them" but "no pairing instruction exists". Forcing via inline asm is not
possible because the gfx950 encoder rejects the opcode.

## Closure of the v_lshrrev_b32 lever family

With R49A / R50A / R53A / R54A / R54E all closed, the CRR ~92% structural
floor's "6× v_lshrrev_b32 per K-pair on odd iters" component is now
**triply-closed across both algorithmic restructure (R49A/R50A/R53A/R54A)
and ISA-level substitution (R54E)** axes.

Subsequent CRR HEADROOM-recovery work should pivot away from the scale-shift
critical path and toward orthogonal axes:
- VMEM/LDS scheduling (R32-class pipelining variants)
- Wave-tile/block swizzle recomposition (R47-class)
- Compute path itself (MMA dispatch order, register allocation hints)
- Or accept the structural ceiling and re-prioritize toward shapes/cells
  with larger achievable headroom.

## Artifacts

- `r54e_results/build_70B_GateUp_baseline.log` — baseline build log (CRR 225 VGPR / 0 spill / 139264 LDS / occ 2)
- `r54e_results/crr_lb1li1_kernel_isa.txt` — V1 CRR kernel ISA dump showing the 6 in-place `v_lshrrev_b32 ..., 16, ...` shifts
- `r54e_results/crr_lb1li2_kernel_isa.txt` — V1 CRR (V2 layout) kernel ISA — 6 shifts (v154-v157, v168-v169 — 6 total)
- `r54e_results/crr_lb0li1_kernel_isa.txt` — non-PQ CRR kernel ISA — 6 shifts (v221-v226)
- `r54e_results/gfx950_packed_shift_probe.txt` — full llvm-mc assembler probe transcript

No source modifications committed. No `MXFP8_CRR_PK_LSHRREV` macro added (no
implementation possible).

## Verdict line for cycle wrap

`R54 Dev E: CRR v_pk_lshrrev_b32 packed-shift instruction substitution — REFUTED-EMPIRICAL-ISA — opcode v_pk_lshrrev_b32 does not exist on gfx950 (llvm-mc reject); no packed-32 shift in CDNA4 SIMD-pair set; 5th-axis closure of v_lshrrev_b32 removal lever family.`
