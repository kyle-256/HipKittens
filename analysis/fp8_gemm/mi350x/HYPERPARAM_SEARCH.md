# Hyperparameter Search Plan (R87)

## Current Locked Hyperparameters

| Param | Current value | Locked by | Sweep allowed? |
|-------|---------------|-----------|----------------|
| BLOCK_SIZE (BLK_M=BLK_N) | 256 | v2 default | NO (P1.2 may add 128) |
| K_BLOCK (BK) | 128 | mfma_f32_*x*x128 sig | NO |
| WARPS_M × WARPS_N | 2×4 (8-warp) | user mandate | NO |
| MIN_BLOCKS_PER_CU | 2 | `[[gfx950-8wave-va-cap]]` | NO |
| BLOCK_SWIZZLE_NUM_XCDS | 8 | MI355X CCD count | NO |
| RCR_MAIN_UNROLL | 2 | `[[fp8-rcr-unroll-harmful]]` | NO |
| chunk_size (R43 win) | 32 | Triton match | NO (already optimal) |
| group_m | caller-passed (4) | PT smoke | NO (caller-controlled) |
| num_xcds | caller-passed (8) | PT smoke | NO (HW match) |
| num_slots (CU count) | NUM_CUS=256 | MI355X HW | NO (R73 verified) |
| RCR_PREFETCH_LGKM | 8 | `[[no-constant-sweep]]` | only ISA-justified |
| RCR_INIT0_VMCNT | 4 | `[[no-constant-sweep]]` | only ISA-justified |
| RCR_INIT1_VMCNT | 6 | `[[no-constant-sweep]]` | only ISA-justified |
| RCR_STEADY_VMCNT | 8 | `[[no-constant-sweep]]` | only ISA-justified |

## Search Permission Policy

User constraint `[[no-constant-sweep]]`: never sweep magic numbers without ISA-level theory. Constants stay at current value unless:
1. Disassemble + identify exact dependency chain
2. Show theoretical optimum (e.g., mfma latency + lgkm round-trip cycles)
3. Verify via single targeted change (not sweep)

## Validated Single-Change Wins

- R3 (HK turbo 1b7646a6, prior to session): b0+b1 split → +3pp dsv3-up worst
- R4 (HK turbo 9c73e332): epilog b0 prefetch → +0.7pp
- R17 (HK turbo 2de20b6f): remove main loop s_setprio → +1.8pp dsv3-up
- **R43 (HK turbo 56f688fe, this session): chunk_size 64→32 → +1.5pp geomean**

Total non-architectural levers exhausted; rest needs P1.2/P1.3a multi-session.

## Forbidden Sweep Examples (Past Failures)

- RCR_STEADY_VMCNT sweep: tried 2026-05-21, ±1% noise across 4-12 range
- chunk_size > 32: R48 dynamic (32/16) caused -10pp on long K
- launch_bounds(_,2): silently ignored on gfx950 8-wave WG (`[[gfx950-8wave-va-cap]]`)
- amdgpu_num_vgpr attribute: R42 ignored, 0 effect

## Future-Proof Lever Worth Investigating (Multi-Session)

1. K_BLOCK=64 + mfma 32×32×64 (P1.2) — already foundation R52-R58
2. BLOCK_SIZE=128 + single-acc (alternative P1.2 path)
3. sk_split_n>1 + reduce kernel (P1.3a)
4. ST swizzle for B pre-transpose (P2.2 RRR only)
