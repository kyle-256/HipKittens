# R35 Opt A — VGPR-PF compiler-bug fix attempt — DEAD

**Date**: 2026-04-19  
**Goal**: Fix the CDNA4 register-allocator clobber that broke R34 Opt B, then test if vmcnt(15) actually unlocks throughput on L6 (4096×32768×128256).  
**Result**: DEAD. The `+v` keepalive defeats the clobber, but the VGPR→ds_write deposit path fails correctness in 4 of 5 variants and crashes with HSA aperture violations in the working configurations.

## What was tried

1. **Replaced the broken `emit_one_pf_dswrite`** in `kernel_mxfp4_gluon_cpp_vgprPF.cpp` (which previously just re-issued LDS-direct, defeating the test) with a real ds_write_b32 quartet in a SINGLE asm block, using the per-lane address `lds_addrs[idx] + voffs[idx]` (matching the buffer_load_to_lds semantic).
2. **Added `vgpr_keepalive(float4&)`** = `asm volatile("" : "+v"(v.x), "+v"(v.y), "+v"(v.z), "+v"(v.w))` — applied (a) right after every `emit_one_pf_vgpr` inside the kpair function, AND (b) right before every `emit_one_pf_dswrite` inside `emit_b_scratch_dswrites`. This forces the compiler to keep the VGPR live across MFMA asm blocks and the iter-end vmcnt drain.
3. **Built 5 variants × 2 K-shapes (4096, 128256) = 10 binaries**, all PASS compile:
   - V0_legacyfork (VGPR_PF_MODE=0): 212 VGPR / 0 spills
   - V0_vgprpf (PF_N=4, vmcnt(0)): 256 VGPR / 172 bytes scratch (SPILLED)
   - V1_vmcnt15 (PF_N=4, vmcnt(15)): 256 VGPR / 172 bytes scratch
   - V2_vmcnt15_n8 (PF_N=8, vmcnt(15)): 220 VGPR / 0 scratch
   - V3_vmcnt12_n4 (PF_N=4, vmcnt(12)): 256 VGPR / 172 bytes scratch

4. **SNR test at L6 (M=4096 N=32768 K=128256)** with subprocess isolation:
   - V0_legacyfork: 100.00% bit-eq vs incumbent → PASS (proves parent kernel is intact under the new code)
   - V0_vgprpf, V1_vmcnt15, V3_vmcnt12_n4 (all PF_N=4): HSA Memory Access Fault (aperture violation), process terminated
   - V2_vmcnt15_n8 (PF_N=8): runs without crash but bit_eq=0.09% (completely wrong data)

## Why R35 Opt A is DEAD

- The `+v` keepalive DOES defeat the clobber (V2_n8 doesn't crash, runs to completion). But the VGPR→ds_write deposit produces wrong data, even though the per-lane LDS address `lds_addrs[idx] + voffs[idx]` matches the documented buffer_load_to_lds semantic.
- For PF_N=4 (mixed VGPR/LDS-direct slots), the kernel HSA-faults — likely because VGPR-PF slots 0-3's deposit-write addresses interact with LDS-direct slots 4-7's addresses in a way that violates wave LDS bounds.
- The R34 Opt B "discard VGPR + re-issue LDS-direct" workaround silently evaded all this by never writing the VGPR data — but then it also never tested the M0-relief hypothesis, so we still don't know if vmcnt(15) is structurally safe.
- Three address formulas were tried (`lane*4 + dword*256`, `lane*16`, `voffs[idx]`); all either crash or produce wrong data in the full kernel context.

## Root cause hypothesis

The LDS region pointed to by `lds_addrs[idx]` may be sized for the EXPECTED hardware-applied write pattern (one specific lane→address mapping), and a software ds_write that uses a DIFFERENT mapping can write outside the wave's LDS slice (causing HSA fault) or write to the wrong subtile slot (causing data corruption). Empirically determining the correct mapping would require synthesizing a test kernel that reads back its own LDS write pattern — out of R35 budget.

## Files

- `analysis/fp8_gemm/mi350x/kernel_mxfp4_gluon_cpp_vgprPF.cpp` — modified with vgpr_keepalive + proper emit_one_pf_dswrite (still broken correctness)
- `analysis/fp8_gemm/mi350x/build_round35_optA.py` — build harness for 5 variants × 2 K-shapes
- `analysis/fp8_gemm/mi350x/snr_R35_optA.py` / `snr_R35_optA_v2.py` / `snr_R35_optA_smallK.py` / `snr_R35_one.py` — SNR validation scripts (all in subprocess isolation due to HSA faults)
- `analysis/fp8_gemm/mi350x/R35_OPT_A_BUILD.log`, `R35_OPT_A_BUILD_RESULTS.json`
- `analysis/fp8_gemm/mi350x/R35_OPT_A_SNR.log`, `R35_OPT_A_SNR_V2_run4.log`, `R35_OPT_A_SNR_smallK.log`

## Verdict: DEAD. L6 ceiling at 92.6% remains.

The vmcnt(15) hypothesis CANNOT be tested without a correct VGPR→LDS deposit path. The path is non-trivial because the LDS layout for size=16 buffer_load_to_lds is not lane*16 contiguous; it depends on the per-lane voff in a way that a software ds_write cannot trivially mimic. R34 Opt B's discard+re-issue trick passed correctness only because it never used the VGPR data.

**41/42 ceiling RE-CONFIRMED post-R35.** Only L6 remains at 92.6%; R29-R35 all DEAD on L6. Recommend STOP per R34 Decider Section 5 Option 3.
