# R52 Dev H — RRR `do_k_iter` soft scheduler barrier (`asm volatile("" ::: "memory")`) — REFUTED

**Date:** 2026-04-19
**Branch:** worktree-agent-a60a5edd off feat/mxfp8-only @ 2d20953b
**GPU:** MI355X (gfx950), HIP_VISIBLE_DEVICES=1
**Hypothesis:** R49 Dev B's `__attribute__((noinline))` phase split was too coarse (catastrophic spills, -97%). Its findings.md suggested replacing it with `asm volatile("" ::: "memory")` — a compiler barrier with no call frame, just an instruction-scheduler boundary. Hope: gate the LLVM scheduler at the cB→cC quartet boundary so scale-pack `v_lshrrev` issue overlaps better with VMEM/LDS latency, closing RRR's K=4096 loop-control gap on 8B Gate/Up (the only RRR HEADROOM cell, +3.1pp ceiling per R48 Dev D).

## TL;DR — VERDICT: REFUTED

The barrier is silently optimized away by the LLVM amdgcn scheduler. ISA codegen across the entire RRR kernel is **bit-identical** between baseline (OFF) and treatment (ON) modulo six `;;#ASMSTART/;;#ASMEND` empty-asm marker pairs and a UID hash difference. No instruction is reordered. Strict-SCLK A/B at 8B Gate/Up RRR confirms: Δ = -0.05% (noise). Cross-shape spot-checks: 70B Q/O RRR Δ = +1.01% (noise), 8B Q/O RRR Δ = -2.88% (noise + concurrent-process contention; tiny shape at 0.056 ms/iter).

This closes the inter-phase scheduler-fence lever permanently. Combined with R49 Dev B (noinline split: REFUTED), R50 Dev D (`__builtin_amdgcn_sched_barrier` mask sweep: REFUTED), and R49 Dev B's earlier R47-era attempts at this same barrier, **all known scheduler-boundary tools — empty-asm, sched_barrier intrinsics, and noinline — fail to alter codegen at the cB→cC boundary in RRR's `do_k_iter`.** RRR's K-loop register window is structurally bounded.

## Strict-SCLK A/B data (HIP_VISIBLE_DEVICES=1, 5 runs, 30s cooldown, MXFP8_WARMUP=100, MXFP8_ITERS=200)

| Cell                | OFF med (TFLOPS) | OFF spread | ON med (TFLOPS) | ON spread | Δ%     | Verdict |
|---------------------|-----------------:|-----------:|----------------:|----------:|-------:|---------|
| 8B Gate/Up RRR      | 2545.2           | 9.03%      | 2543.8          | 4.63%     | -0.05% | NEUTRAL |
| 8B Q/O RRR          | 2468.8           | 0.74%      | 2397.8          | 7.80%     | -2.88% | NOISE (contention; ISA identical) |
| 70B Q/O RRR         | 2920.4           | 2.49%      | 2949.9          | 1.72%     | +1.01% | NEUTRAL |

Per-run TFLOPS:

```
8B_GateUp_RRR_off  : 2572.99 2343.14 2555.81 2543.18 2545.21
8B_GateUp_RRR_on   : 2543.47 2553.36 2435.52 2546.65 2543.84
8B_QO_RRR_off      : 2472.30 2480.69 2468.85 2465.68 2462.37
8B_QO_RRR_on       : 2452.61 2336.40 2276.90 2397.76 2464.03
70B_QO_RRR_off     : 2954.40 2881.58 2920.38 2934.12 2891.71
70B_QO_RRR_on      : 2954.75 2903.96 2949.86 2949.31 2952.85
```

The 8B Q/O ON run-2 (2336.4) and run-3 (2276.9) are obvious outliers vs. the other three ON runs (2452, 2397, 2464) and the OFF baseline (2462-2480). During the bench another concurrent process was observed running `python3 test_mxfp8_python.py 4096 14336 4096` on the same host, causing CPU/PCIe scheduling jitter on the very small 8B Q/O cell (0.056 ms/iter — extremely sensitive to host-side latency). Restricting to the un-perturbed runs (1, 5) gives ON ≈ OFF.

## Correctness

`MXFP8_RRR_SOFT_BARRIER=1` 8B Gate/Up RRR: SNR = **49.61 dB** (≥48 dB), pass-rate **100%**, determinism **3/3 PASS**.

## ISA evidence (the load-bearing finding)

Built both variants with identical flags (`-DM_DIM=4096 -DN_DIM=14336 -DK_DIM=4096`) using `--cuda-device-only -S`, full-pipeline GFX950 assembly. Diff:

```
$ diff r52h_isa_dumps/rrr_off.s r52h_isa_dumps/rrr_on.s | wc -l
36
$ diff r52h_isa_dumps/rrr_off.s r52h_isa_dumps/rrr_on.s
5264a5265,5266
> 	;;#ASMEND
> 	;;#ASMSTART
5479a5482,5483
> 	;;#ASMEND
> 	;;#ASMSTART
... (six pair-insertions identical in shape) ...
29597c29609
< 	.type	__hip_cuid_40fab0a53d610f61,@object
---
> 	.type	__hip_cuid_bec60f8e953aa98b,@object
... (UID hash difference, irrelevant) ...
```

Looking at the codegen around the inserted marker (line 5260+):

```
ON:                                              OFF:
v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], …  v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], …
v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], …  v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], …
s_setprio 0                                      s_setprio 0
s_barrier                                        s_barrier
;;#ASMSTART  <-- EMPTY                           ;;#ASMSTART
;;#ASMEND    <-- EMPTY                           ds_read_b128 v[206:209], v173 offset:0
;;#ASMSTART
ds_read_b128 v[206:209], v173 offset:0
;;#ASMEND
;;#ASMSTART
```

The `asm volatile("" ::: "memory")` lowers to a zero-instruction marker that the post-RA AMDGPU scheduler treats as a clobber barrier for memory dependencies, but since the surrounding instructions are MFMAs and DS reads with explicit hardware ordering (s_barrier, s_setprio, s_waitcnt lgkmcnt(0), explicit phase swap), there is **no scheduling freedom for clang to give up**. The empty asm changes nothing about the instruction stream.

## Diagnosis

R49 Dev B's findings.md proposed three soft alternatives:

1. `asm volatile("" ::: "memory")` (this experiment) — **REFUTED**: zero ISA effect.
2. `__builtin_amdgcn_sched_barrier(mask)` (R50 Dev D) — **REFUTED**: -0.07% to +0.10% across 4 mask values, all noise.
3. Compile-time partial unroll (R47A/R48G) — bimodal-spill ceiling reached.

The structural reason all three fail at this boundary: the existing baseline body around the cB→cC transition already has a hand-written *hardware* barrier sequence:

```
__builtin_amdgcn_s_setprio(0);
__builtin_amdgcn_s_barrier();         // hardware barrier
        <-- noinline / sched_barrier / asm volatile placed here -->
load_a(a, As[tic][1], wm);
G::load(Bs[tic][1], …);
__builtin_amdgcn_s_barrier();         // hardware barrier
MXFP8_RRR_PHASE_SB();                  // optional sched barrier
asm volatile("s_waitcnt lgkmcnt(0)"); // hand-written wait
__builtin_amdgcn_s_setprio(1);
```

The MMA wave already issues `s_setprio(0); s_barrier;` between cB and the cC loads, then `s_barrier; s_waitcnt lgkmcnt(0); s_setprio(1);` before cC's MMA. These are *real* hardware fences with hardware ordering. There is no slack in the LLVM machine scheduler for a soft compile-time barrier to rearrange — every reorder is already pinned by the surrounding `s_*` intrinsics.

R49 Dev B's hypothesis (that scheduler-induced latency is the gap) is likely the wrong mechanism. R48 Dev D's diagnosis (8B Gate/Up RRR is K=4096 loop-control overhead) probably points to: (a) the `MXFP8_RRR_PRAGMA_UNROLL_MAIN` factor itself, or (b) the scale-pack `v_lshrrev` chain, neither of which is touched by an inter-phase barrier. The scale-pack chain runs in `load_scale_packs(kp)` *outside* `do_k_iter`, before the lambda is even called — putting a scheduler fence inside the lambda body cannot accelerate it.

## Outcome

Source patch landed (default OFF) for archival, mirroring R49 Dev B's pattern:

- New macro `MXFP8_RRR_SOFT_BARRIER` (default 0) declared next to `MXFP8_RRR_PHASE_SPLIT`.
- One use site at the natural cB→cC boundary in the baseline `!MXFP8_RRR_PHASE_SPLIT` arm of `do_k_iter`.
- When OFF: `MXFP8_RRR_SOFT_BAR()` expands to `do {} while (0)` — bit-identical baseline codegen.
- When ON: `__asm__ volatile("" ::: "memory")` — silently a no-op per ISA dumps above.

No regression risk in tree. The RRR soft-barrier lever is closed cycle-permanently.

## Files

- `r52h_bench.sh` — strict-SCLK A/B driver (3 cells × 5 runs each side)
- `r52h_bench.run.log` — orchestrator stdout (with summary table)
- `r52h_results/` — per-cell per-run logs (`*_off_run{1..5}.log`, `*_on_run{1..5}.log`, `*_check.log`)
- `r52h_isa_dumps/rrr_off.s`, `r52h_isa_dumps/rrr_on.s` — full GFX950 assembly diff (36 lines, all empty-asm markers + UID hash)
- `rrr_mxfp8_exact_8wave_fastpath.inc` — `MXFP8_RRR_SOFT_BARRIER` arm landed default OFF

## Cycle status

Combined with R49 Dev B (noinline) and R50 Dev D (sched_barrier mask), all three RRR `do_k_iter` inter-phase scheduler-fence variants are now refuted under strict SCLK. The 3.1pp gap on 8B Gate/Up RRR (K=4096 loop-control headroom) is not addressable via inter-phase fencing. Future work should target either (a) the scale-pack `v_lshrrev` issue chain inside `load_scale_packs`, or (b) the K-loop pragma unroll factor itself (R48G already explored, bimodal-spill ceiling).
