# Round 17 Optimizer C — Untested __attribute__'s on 4 stuck deep-LOSE shapes — VERDICT

**Theme:** Probe 6 untested AMDGPU `__attribute__`'s as kernel-signature
adornments, isolated as the only delta vs each shape's parent flag set.
Hypothesis: explicit register-cap or workgroup-size hints might unjam the
saturated parents on DLA1 / DLA2 / DLA7 / P1.

**Result: 0 commit-worthy wins. DEAD END.** All 3 smoke-PASS candidates
collapsed under 5-run replication. New BROKEN registry entries: 1
unrecognized attribute, 1 always-aperture attribute, 2 register-cap attributes
that catastrophically degrade or break specific shapes.

---

## Methodology

1. **Compile-time recognition probe** (`attr_compile_test_r17c.log`): tested
   each candidate attribute on a stub HIP kernel. 9/10 attrs RECOGNIZED.
   `amdgpu_no_agpr` was REJECTED as `unknown attribute, ignored`.
2. **Build** (`build_round17_optC_attrs.{py,log}`): patched
   `kernel_mxfp4_gluon_cpp.cpp` in-memory to inject one `__attribute__(())`
   line directly above the existing `__global__ __launch_bounds__(...)` line.
   Built 4 shapes × 9 attrs = 36 variants. **32 OK, 4 timed out** —
   `flat_work_group_size(128,512)` on all 4 shapes hung the LLVM scheduler
   past 600 s (kernel actually launches with 256 threads and `__launch_bounds__(256,1)`
   so a min=128/max=512 annotation is a logical conflict; LLVM appears to
   loop on it).
3. **ASM-diff probe** (`asm_diff_probe_r17c.{py,log,json}`): all 32 candidates
   produced `.text` hash ≠ parent and mutually distinct, confirming each
   attribute changed codegen.
4. **SNR safety probe** (`snr_probe_r17c.{py,log,json}`): tiny M=256 random
   fp4 with all-zero scales. Verdict per (shape, attr):
   - `mnwg8` (`amdgpu_max_num_work_groups(8,1,1)`) → **APERTURE on all 4 shapes**.
     Restricting WG count appears to break the kernel's persistent-grid
     assumptions; classified BROKEN-MAX-WG.
   - `vgpr192` → BROKEN-NAN on DLA1; APERTURE on P1 (compiler can't pack
     into 192 VGPRs without spilling into AGPRs that the kernel doesn't
     manage; produces NaN/aperture).
   - `vgpr224` → APERTURE on P1.
   - All other 25 (shape, attr) pairs: OK.
5. **Smoke** (`bench_round17_optC_smoke.{py,log,json}`): warmup=200 iters=500
   trim=10%, GPUs 5/6/7. Single-shot parent + cand. Gate Δpp ≥ +0.5.
   Additional full-M failures discovered:
   - `vgpr224` on DLA1 → APERTURE at full M (small-M SNR didn't catch).
   - `vgpr224` on DLA2/DLA7 → -59 to -60 pp (catastrophic spill).
   - `vgpr192` on DLA2/DLA7 → -78 pp.
   - `sgpr96` on DLA7 → -49 pp.
   3 candidates passed: DLA1/sgpr96 (+1.90pp), P1/fwgs256_256 (+22pp,
   suspicious — base 3839 vs steady ~5000), P1/sgpr80 (+0.56pp, borderline).
6. **Verify** (`bench_round17_optC_verify.{py,log,json}`): 5 reps on GPU 5
   each. Gate: mean_cand ≥ max_base AND Δpp ≥ +1.0.
   - **DLA1/sgpr96**: APERTURE crashes in 1/5 cand and 1/5 base runs;
     mean Δpp = **-4.06pp** → FAIL. The sgpr cap also makes the kernel
     occasionally crash at full M.
   - **P1/fwgs256_256**: mean Δpp = **+0.009pp** → FAIL. Smoke Δpp +22pp
     was 100% base-side bench noise.
   - **P1/sgpr80**: mean Δpp = **-0.92pp** → FAIL.

---

## Per-shape × per-attribute matrix

`AC` = aperture crash at small or full M
`NAN` = output all-NaN at small M
`KILL` = -50pp+ smoke (catastrophic spill / register-cap unmet)
`COMP/timeout` = compiler hung 600 s (`fwgs128_512` only)
`fail` = compiled, ran, smoke Δpp < +0.5
`smoke-PASS` = smoke Δpp ≥ +0.5 (advanced to verify)
`verify-FAIL` = smoke-PASS but 5-run mean Δpp < +1.0 (or aperture in verify)

| Attribute            | DLA1                | DLA2  | DLA7  | P1                  |
|----------------------|---------------------|-------|-------|---------------------|
| `fwgs64_256`         | fail (-4.1pp)       | fail  | fail  | fail (+0.27pp)      |
| `fwgs256_256`        | fail (+0.20pp)      | fail  | fail  | smoke-PASS / verify-FAIL (Δpp +0.01) |
| `fwgs128_512`        | COMP/timeout        | COMP/timeout | COMP/timeout | COMP/timeout |
| `vgpr256`            | fail (+0.22pp)      | fail  | fail  | fail (-0.24pp)      |
| `vgpr224`            | AC (full-M)         | KILL (-59pp) | KILL (-60pp) | AC (small-M) |
| `vgpr192`            | NAN (small-M)       | KILL (-78pp) | KILL (-78pp) | AC (small-M) |
| `sgpr96`             | smoke-PASS / verify-FAIL (Δpp -4.06, AC) | fail | KILL (-50pp) | fail (-0.18pp) |
| `sgpr80`             | fail (+0.04pp)      | fail (parent AC) | fail (-0.32pp) | smoke-PASS / verify-FAIL (Δpp -0.92) |
| `mnwg8`              | AC (small-M)        | AC    | AC    | AC                  |
| `no_agpr` (rejected) | UNRECOGNIZED        | -     | -     | -                   |

---

## New BROKEN registry entries (R17C)

| Tag | Pattern | Shapes affected | Failure mode |
|-----|---------|-----------------|--------------|
| `R17C-UNKNOWN-no_agpr` | `__attribute__((amdgpu_no_agpr))` | (any) | hipcc ROCm 6.x: `warning: unknown attribute 'amdgpu_no_agpr' ignored`. Don't waste cycles on this. |
| `R17C-COMPHANG-fwgs128_512` | `__attribute__((amdgpu_flat_work_group_size(128,512)))` on a kernel launched with 256 threads + `__launch_bounds__(256,1)` | DLA1 / DLA2 / DLA7 / P1 | hipcc hangs the LLVM scheduler past 600 s. Annotation must be consistent with launch_bounds. Use `flat_work_group_size(min,256)` only. |
| `R17C-AC-mnwg8` | `__attribute__((amdgpu_max_num_work_groups(8,1,1)))` | all 4 | HSA aperture violation at small M. Kernel uses persistent-grid pattern; capping max WG to 8 breaks indexing. |
| `R17C-KILL-vgpr224` | `__attribute__((amdgpu_num_vgpr(224)))` | DLA2 / DLA7 / P1 | -59 to -60pp spill on DLA2/DLA7; aperture on P1. The 4 stuck parents need ≥ 256 VGPRs. |
| `R17C-KILL-vgpr192` | `__attribute__((amdgpu_num_vgpr(192)))` | DLA1 / DLA2 / DLA7 / P1 | NaN on DLA1, -78pp on DLA2/DLA7, aperture on P1. Same root cause as vgpr224, more severe. |
| `R17C-KILL-sgpr96-DLA7` | `__attribute__((amdgpu_num_sgpr(96)))` on DLA7 parent | DLA7 only | -49pp; the deep-LOSE DLA7 path needs > 96 SGPRs. |
| `R17C-AC-sgpr96-DLA1-flaky` | `__attribute__((amdgpu_num_sgpr(96)))` on DLA1 parent | DLA1 | Intermittent aperture: 1/5 verify runs crash. Flaky → don't ship. |

---

## Conclusion

The untested-attribute axis is **NOT a breakthrough lever** for the 4 stuck
deep-LOSE shapes. Every recognized attribute either had a noop perf effect
(< +0.5pp at smoke), failed safety (NaN/aperture), or catastrophically
degraded perf via spill. The kernel is finely tuned for ~256 VGPRs / >96
SGPRs / unrestricted WG count / 4 waves with `launch_bounds(256,1)` —
externally constraining any of these via attributes does not improve and
usually breaks it.

This confirms the **R3 "register pressure ceiling"** + **R12 "saturation"**
findings from a new angle: the 4 stuck shapes are at a register-allocation
fixed point that is robust to attribute-level tweaks.

**No commit. Time-box: ~32 min wall clock.**

Author: kyle-256 / Kyle.Zhao@amd.com
Date: 2026-04-17
