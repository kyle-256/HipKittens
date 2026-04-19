# R54 Dev H findings — V2 RRR wider memory loads (`buffer_load_dwordx4` coalescing)

**Cycle:** R54
**Kernel:** `_Z29rrr_exact_8wave_scaled_kernelILb1ELi2EE` (V2 RRR fastpath)
**Cells:** 8B Gate/Up RRR (M=4096 N=14336 K=4096), 70B Q/O RRR (M=4096 N=8192 K=8192), 8B Down RRR (M=4096 N=4096 K=14336)
**Lever:** Coalesce A/B operand and scale-tensor global loads from many `global_load_b32` into fewer `global_load_b128` (or `global_load_b96`/`global_load_b64`)
**Origin:** R54 Dev H mandate (V2 RRR -1.0pp HEADROOM on 8B Gate/Up).

## Verdict — REFUTED-EMPIRICAL-ISA (Phase 0; no kernel build modification needed)

**The lever's premise is empirically false.** The V2 RRR fastpath kernel as
shipped already issues **only `buffer_load_dwordx4` (b128)** and
**`buffer_load_dwordx2` (b64)** memory loads in its kernel body. There are
**zero** `global_load_dword`/`global_load_b32`/narrow-byte/narrow-short loads
in the V2 RRR kernel. The "many `global_load_dword` per K-pair" assumption in
the mandate brief does not hold against the actual gfx950 disassembly. There
is nothing left to coalesce — the kernel is already at the maximum load width
the gfx950 buffer-load instruction set supports.

This closes the "wider loads / load coalescing" axis on V2 RRR for the same
reason R52Q's cachepolicy refutation closed an earlier axis: the lever
attacks an attribute that, on inspection, does not exist in the production
ISA.

## Phase 0: ISA-level baseline scan

### Step A — Build all 3 RRR cells and confirm baseline VGPR ceiling

Built the V2 RRR kernel at all three target shapes from
`r54h_workspace/` (a pristine copy of `analysis/fp8_gemm/mi350x/`):

| Cell             | Shape                | VGPR | Spill | Scratch | LDS    | Occ |
|------------------|----------------------|-----:|------:|--------:|-------:|----:|
| 8B Gate/Up RRR   | M=4096 N=14336 K=4096| 254  | 0     | 0       | 135168 | 2   |
| 70B Q/O RRR      | M=4096 N=8192 K=8192 | 254  | 0     | 0       | 135168 | 2   |
| 8B Down RRR      | M=4096 N=4096 K=14336| 254  | 0     | 0       | 135168 | 2   |

V2 RRR (`Lb1ELi2EE`) baseline ceiling: **254/256 VGPR — 0 spill — 0 scratch
— occ 2**. (Same numbers as the v2_rrr_vgpr_ceiling memory note.) Build log:
`r54h_results/build_8B_GateUp_resources.txt`.

### Step B — Disassemble V2 RRR kernel body, count all load opcodes

For each cell, dumped `--offload-device-only -S` device assembly via hipcc,
isolated the `_Z29rrr_exact_8wave_scaled_kernelILb1ELi2EE...` function body
(label → next `.amdhsa_kernel` directive), and inventoried every memory
load opcode:

```
==== 8B Gate/Up V2 RRR — kernel body 1614 lines ====
      2 buffer_load_dwordx2     # b64,  scale loads (B side, 2 packs × 32B)
     34 buffer_load_dwordx4     # b128, A operands DTL + A scale loads + B operands DTL

==== 70B Q/O V2 RRR — kernel body 1612 lines ====
      2 buffer_load_dwordx2
     34 buffer_load_dwordx4

==== 8B Down V2 RRR — kernel body 1612 lines ====
      2 buffer_load_dwordx2
     34 buffer_load_dwordx4
```

**No `global_load_dword`, no `buffer_load_dword`, no narrow `ubyte`/`ushort`
loads exist anywhere in the V2 RRR kernel body** (across all three cells).
Identical opcode breakdowns across all three shapes — the load topology is
shape-invariant (the K-loop body is structurally identical; only the loop
trip count varies).

Evidence files:
- `r54h_results/v2rrr_8B_GateUp_baseline_isa.s`
- `r54h_results/v2rrr_70B_QO_baseline_isa.s`
- `r54h_results/v2rrr_8B_Down_baseline_isa.s`

### Step C — Map opcodes to source sites

The 36 buffer loads in the V2 RRR kernel decompose as:

**A operand DTL (direct-to-LDS, 8 loads/K-iter × 8 visible K-iter copies =
~28 of the 34 b128 loads)**: `kernels/.../global_to_shared.cuh::load`
emits `llvm_amdgcn_raw_buffer_load_lds(..., bytes_per_thread=16, ...)` →
`buffer_load_dwordx4 ... lds`. Operand width is dictated by
`ST::underlying_subtile_bytes_per_thread = 16` for fp8e4m3 shared-tile
shapes, which is the maximum b128. Cannot widen further — gfx950 does
not have b256 buffer loads.

**A scale load (V2 path, line 480-482 of `rrr_mxfp8_exact_8wave_fastpath.inc`)**:
`llvm_amdgcn_raw_buffer_load_b128(a_v2_srsrc, ...)` → `buffer_load_dwordx4`
fetching `{a0p0, a1p0, a0p1, a1p1}` (4 packs × 4 bytes = 16 B/lane = b128).
Already maximum width.

**B scale load (V2 path, line 489-494)**: `llvm_amdgcn_raw_buffer_load_b64`
→ `buffer_load_dwordx2` fetching `{b0p0, b1p0}` (2 packs × 4 bytes = 8
B/lane = b64). To widen to b128 would require fetching 4 packs (b0p0, b1p0,
b0p1, b1p1) — but `rrr_b_pack_count = (RBN+31)/32 = 1` for the V2 RRR
8-wave shape (RBN=32). There is no b0p1/b1p1 to fetch; the data does not
exist. Widening is structurally blocked by `rrr_b_pack_count = 1`.

### Step D — Probe gfx950 for any wider load opcode

Direct `llvm-mc` assembler probe (transcript:
`r54h_results/gfx950_wider_load_probe.txt`):

| Opcode | gfx950 status |
|---|---|
| `global_load_b256 v[0:7], ...`             | **invalid instruction** (does not exist) |
| `global_load_b128 v[0:3], ...` (FLAT form) | **instruction not supported on this GPU** |
| `global_load_b64 v[0:1], ...`  (FLAT form) | **instruction not supported on this GPU** |
| `buffer_load_b256 v[0:7], ...`             | **invalid instruction** |
| `buffer_load_b128 v[0:3], ...`             | **instruction not supported on this GPU** |
| `buffer_load_dwordx4 v[0:3], ...`          | valid (= b128, the V2 RRR uses this) |
| `buffer_load_dwordx3 v[0:2], ...`          | valid (= b96, would shrink coalescing) |
| `buffer_load_dwordx2 v[0:1], ...`          | valid (= b64, the V2 RRR uses this for B) |
| `buffer_load_dword v0, ...`                | valid (= b32, the V2 RRR does NOT use this) |

Two relevant facts emerge:
1. **There is no wider-than-128b memory load on gfx950.** `b256` is invalid;
   the V1/RDNA `global_load_b{64,128}` mnemonics target gfx11+ and are not
   supported on CDNA4 (gfx950). The CDNA4 buffer-load ISA tops out at
   `buffer_load_dwordx4 = b128`.
2. **The kernel already uses the maximum.** Every load in the V2 RRR
   kernel body is either `buffer_load_dwordx4` (b128, the maximum) or
   `buffer_load_dwordx2` (b64, used where the per-lane payload is exactly
   8 B because `rrr_b_pack_count = 1`).

### Step E — Cross-check: where DO `global_load_dword` instructions occur?

For comparison, dumped the V1 RRR kernel (`Lb1ELi1EE`, the SCALE_VERSION=1
template instantiation that is **not** dispatched by the production host
path `dispatch_rrr_exact_8wave_scaled_v2`). The V1 RRR kernel body shows:

```
==== V1 RRR (Lb1ELi1EE) — NOT the production V2 path ====
     32 buffer_load_dwordx4
     12 global_load_dword           # ← narrow loads HERE, in V1 only
```

The 12 `global_load_dword` instructions in V1 RRR map to the V1 scale path
(`load_scale_pair_pack_16x128_preshuffled_from_row_base`, .inc lines 516–540)
which fetches one b32 per (pack, side) — six per K-pair. **These are exactly
the loads the R22 V2-RRR layout migration consolidated into one b128
(A side) + one b64 (B side) per K-pair.** Production already uses V2 (host
gate `MXFP8_RRR_PRESHUFFLE_V2_RUNTIME=1`, dispatcher
`dispatch_rrr_exact_8wave_scaled_v2`, also confirmed by every R38–R53
nm_gate log). The mandate's "many `global_load_dword`" description applies
to the **legacy V1 path** that production has not used since R22.

Evidence: `r54h_results/v1rrr_8B_GateUp_baseline_isa.s` (12 narrow loads
visible at lines 434-441 and 915-922 of that .s).

## Phase 1 / Phase 2 — not reached

Phase 1 (build a `MXFP8_RRR_WIDE_LOADS=1` macro-gated path and re-check
VGPR/spill/scratch) was not reached because Phase 0 demonstrated no
narrowable loads exist in the production V2 RRR kernel. Building such a
macro would either be a no-op (no-narrower-load to widen) or, if forced
to fetch additional adjacent K-pair data ahead of schedule, would extend
A scale-pack live ranges across K-pair boundaries — directly violating the
254/256 VGPR ceiling per the v2_rrr_vgpr_ceiling memory note.

Phase 2 (3-cell bench × 5 runs) was not reached because there is nothing to
benchmark.

## Why the four candidate implementations all collapse on V2 RRR

The mandate listed three candidate approaches:

1. **`__builtin_amdgcn_global_load_dwordx4` (vector load builtin)** — would
   emit `buffer_load_dwordx4` (= b128). The kernel **already** emits
   `buffer_load_dwordx4` on every load site that has 16 B of consecutive
   per-lane data (A operand DTL, A scale b128, every load except B scale
   b64). Identity transformation.

2. **Inline asm `global_load_b128 v[V0:V3], voff, soff`** — `global_load_b128`
   is the gfx11+ FLAT mnemonic and is rejected by the gfx950 assembler
   (`instruction not supported on this GPU` per Step D). The CDNA4-correct
   inline asm would be `buffer_load_dwordx4 v[V0:V3], voffset, srsrc, soffset
   offen` — which the kernel already emits via
   `llvm_amdgcn_raw_buffer_load_b128`.

3. **Pointer-cast trick (`uint32_t*` → `__uint128_t*`)** — already used at
   line 481 of `rrr_mxfp8_exact_8wave_fastpath.inc`:
   ```cpp
   const __uint128_t a_raw =
       llvm_amdgcn_raw_buffer_load_b128(a_v2_srsrc, a_voff, a_soff,
                                        MXFP8_RRR_V2_SCALE_CACHEPOLICY);
   ```
   This is the V2-RRR scale-load b128 implementation — already at maximum
   width.

A hypothetical 4th approach — fetch 2 K-pairs of scales in one b128 load on
the B side (2 packs × 2 K-pairs × 4 B = 16 B) — would extend the B scale
live range across K-pairs and add at minimum 4 extra VGPRs (the not-yet-used
b1p0/b1p0' packs). With the V2 RRR at 254/256, that would land at 258 VGPR
projected, hitting the v2_rrr_vgpr_ceiling closure. That same forward-fetch
pattern was the exact failure mode of R49A / R50A / R53A / R53B (all
spilled or pressure-bound), so the closure carries directly.

## Resource-projection summary

| Metric                       | Baseline (V2 RRR)     | This lever (projected)               |
|------------------------------|-----------------------|--------------------------------------|
| VGPR (V2 RRR `Lb1ELi2EE`)    | 254 / 256             | n/a — no implementable transform     |
| VGPR Spill                   | 0                     | n/a                                  |
| Scratch                      | 0                     | n/a                                  |
| LDS                          | 135168 B              | 135168 B (no LDS change planned)     |
| Buffer loads in K-loop body  | 36 (34× b128, 2× b64) | already at gfx950 maximum widths     |
| Narrow `global_load_dword`   | 0                     | 0 (none to coalesce)                 |
| Reality                      | —                     | **lever target does not exist**      |

## Reasoning

The V2 RRR fastpath was already engineered (R22 milestone-1) to coalesce
the V1 layout's per-row-group b32 scale loads into one wave-tile-slab b128
load per K-pair on the A side and one b64 load per K-pair on the B side.
That migration is the reason the production dispatcher is
`dispatch_rrr_exact_8wave_scaled_v2`. The R54 Dev H lever — "coalesce many
`global_load_dword` into `global_load_b128`" — was already realised at
R22 and is the kernel's current shipped state.

This is structurally identical to R54 Dev E's situation: the lever
attacks a property the production kernel doesn't have. Where R54 Dev E
attacked an opcode (`v_pk_lshrrev_b32`) that doesn't exist on gfx950,
R54 Dev H attacks narrow loads that don't exist in the V2 RRR kernel body.

Per the mandate's escalation criteria:

> "ISA-only sanity scan. Build with gate=1. Verify wider load instruction
> (`global_load_b128` or similar) actually appears in disassembly. If ANY
> resource regression → REFUTED-EMPIRICAL, no bench."

The strict-stronger condition holds: the wider load instruction is **already
present** in the baseline disassembly (34× b128, 2× b64) and there is no
narrow-load left to widen, so building a `MXFP8_RRR_WIDE_LOADS=1` gate
either (a) no-ops (compiler emits identical IR), or (b) extends inter-K-pair
live state and busts the 254/256 VGPR ceiling — which prior R49A/R50A/R53A/
R53B closures already documented as the dominant failure mode.

## Closure of the wider-load lever family on V2 RRR

V2 RRR memory-width axis status post-R54 Dev H:

1. **Per-load width** — V2 RRR is at the gfx950 maximum (b128) on all
   eligible sites; b64 sites are width-pinned by `rrr_b_pack_count = 1`.
   **Closed by Phase 0 ISA scan (this work).**
2. **Cross-K-pair coalescing** — adds inter-K-pair live state, pressures
   VGPR ceiling. **Triply-closed by R49A/R50A/R53A/R53B (memory:
   v2_rrr_vgpr_ceiling).**
3. **Hardware ceiling** — gfx950 has no `b256` buffer-load opcode and the
   FLAT `global_load_b{64,128}` mnemonics are not supported on CDNA4.
   **Closed by llvm-mc probe (this work, Step D).**

Subsequent V2 RRR HEADROOM-recovery work should pivot away from the
load-width / load-count critical path and toward orthogonal axes (per the
R52P diagnosis, the dominant cause is L2/TC return backpressure, not
issue-side load count):

- L2 cache locality / XCD-aware swizzle re-tuning (R46 Dev D class)
- TC-pipe scheduling (R52 Dev L tail-peel class — partially explored)
- Compute-side latency hiding (sched_barrier inter-phase masks, R50 Dev D)
- Or accept the 99% MX/FP8 ceiling on the 8B Gate/Up RRR cell as a
  structural floor and re-prioritise toward shapes with larger achievable
  headroom.

## Verdict line for cycle wrap

`R54 Dev H: V2 RRR wider memory loads (buffer_load_dwordx4 coalescing) —
REFUTED-EMPIRICAL-ISA — V2 RRR kernel body already emits 34× b128
+ 2× b64 buffer loads with zero narrow global_load_dword/b32 to coalesce;
gfx950 has no wider-than-b128 buffer load and the FLAT global_load_b128
mnemonic is not supported on CDNA4; the R22 V2-RRR layout migration
already realised this lever. Closes wider-load axis on V2 RRR.`

## Artifacts

- `r54h_results/build_8B_GateUp_resources.txt` — V2 RRR baseline build resource block (254 VGPR / 0 spill / 0 scratch / 135168 LDS / occ 2 confirmed)
- `r54h_results/v2rrr_8B_GateUp_baseline_isa.s` — V2 RRR `Lb1ELi2EE` kernel body for 8B Gate/Up shape (34 b128 + 2 b64, zero narrow loads)
- `r54h_results/v2rrr_70B_QO_baseline_isa.s` — V2 RRR `Lb1ELi2EE` kernel body for 70B Q/O shape (identical opcode mix)
- `r54h_results/v2rrr_8B_Down_baseline_isa.s` — V2 RRR `Lb1ELi2EE` kernel body for 8B Down shape (identical opcode mix)
- `r54h_results/v1rrr_8B_GateUp_baseline_isa.s` — V1 RRR `Lb1ELi1EE` kernel body for 8B Gate/Up shape, included as cross-check showing where the legacy 12× `global_load_dword` lived before the R22 V2 migration coalesced them
- `r54h_results/gfx950_wider_load_probe.txt` — `llvm-mc` assembler probe transcript (b256 invalid, FLAT b128/b64 unsupported on gfx950, MUBUF dwordx{1,2,3,4} all valid)

No source modifications committed. No `MXFP8_RRR_WIDE_LOADS` macro added (no
implementation possible / would no-op or bust 254 VGPR ceiling).
