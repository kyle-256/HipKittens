# Round 7 — FP8 grouped launch_bounds MIN=4 ignored by compiler + a_kt1 scoping cleanup

## Baseline (round-7 entry)

`_metric_grouped_only.py` (single run): `score=815, geomean=0.9781, n=16`.

Worst 4 FP8 shapes:

| shape                        | ratio |
|------------------------------|------:|
| gpt_oss-Down-B4-M4096        | 0.929 |
| gpt_oss-GateUP-B4-M2048      | 0.930 |
| gpt_oss-GateUP-B4-M4096      | 0.940 |
| gpt_oss-Down-B4-M2048        | 0.972 |
| DSV3-Down-B16-M2048          | 0.973 |

## Probe 1: `__launch_bounds__(_NUM_THREADS, 4)` — compiler ignores the hint

Tightened grouped_rcr_kernel's launch_bounds MIN_BLOCKS_PER_CU from `1
→ 4` (from round-5 which tried `→ 2`; that test found `2` achieved but
no gain). Hypothesis: stronger hint would force register allocator to
compress VGPR usage below the ~48/wave threshold needed to fit 4
blocks/CU (= 8 waves/SIMD = max MI355X occupancy).

Build: clean. Resource remarks:

| kernel variant         | VGPR | spill | occupancy | scratch |
|------------------------|:----:|:-----:|:---------:|:-------:|
| `<0,F,F>` (DSV3 path)  | 256  | 67    | 2         | 272     |
| `<0,T,F>` (N-mask path)| 256  | 76    | 2         | 308     |
| `<0,F,T>` (K-fuse path)| 256  | 45    | 2         | 184     |
| `<0,T,T>` (both)       | 256  | 54    | 2         | 220     |

**Bit-identical to baseline (MIN=1) resources.** Compiler treats MIN>1
as a non-binding hint and falls back to whatever fits — here, the
kernel's 256-VGPR demand already saturates the 2 waves/SIMD budget.

Takeaway: **`__launch_bounds__` MIN parameter cannot force occupancy
above what VGPR pressure permits.** Reverted.

## Probe 2: `a_kt1` register-scoping cleanup — no-op codegen

Observation: `A_row_reg a_kt1` was declared at function scope (round-3
fused-K-tail change), only referenced inside `if constexpr (FUSED_KTAIL)`.
Round-3 comment claimed "compiler renames around it; unused in
FUSED_KTAIL=false variants". Round-5 spill deltas (`<0,T,F>` spill=76
vs `<0,T,T>` spill=54 — a +22 spill for the non-FUSED_KTAIL but
N-masked case) suggested this claim may not fully hold.

Moved `A_row_reg a_kt1` from function-scope (line 1995) into the
`if constexpr (FUSED_KTAIL)` block (line ~2290), where the actual
usage is.

Resource deltas post-move (vs baseline):

| kernel variant         | VGPR (Δ) | spill (Δ) | occupancy | scratch (Δ) |
|------------------------|---------:|----------:|:---------:|------------:|
| `<0,F,F>`              | 256 (0)  | 67 (0)    | 2         | 272 (0)     |
| `<0,T,F>`              | 256 (0)  | 76 (0)    | 2         | 308 (0)     |
| `<0,F,T>`              | 256 (0)  | 45 (0)    | 2         | 184 (0)     |
| `<0,T,T>`              | 256 (0)  | 54 (0)    | 2         | 220 (0)     |

**Bit-identical to baseline.** The compiler already eliminates the
unused `a_kt1` slot via standard DCE across the template
specializations; the function-scope declaration was indeed cost-free.

3-run metric post-change: 813 / 817 / 816 (median 816, baseline median
816 — flat within σ≈2).

## Round-7 outcome

- launch_bounds MIN=4: **falsified** (compiler ignores; occupancy
  ceiling is VGPR-bound, not hint-bound).
- a_kt1 scoping: **bit-identical codegen**; shipped anyway as a
  cleanup (intent now matches code — future refactors less likely to
  disturb the DCE invariant).

## Round-8 plan: begin 4-wave grouped port

Dense RCR FP8 has two variants:

- `grouped_rcr_kernel` (line 1973+): 8-wave (WM=2, WN=4, 512 thr,
  occ 2 waves/SIMD). Used by ALL grouped dispatches currently.
- `rcr_4w::kernel` (line 805-1256): 4-wave (WM=2, WN=2, 256 thr,
  **4 blocks/CU = 4 waves/SIMD**). Dispatched by dense when
  `aligned_grid >= RCR_4WAVE_MIN_GRID=3200 && k <= RCR_4WAVE_MAX_K=8192`.

Metric qualifying shapes for 4-wave (if ported to grouped with same
gate criteria):

| shape                  | aligned_grid | k   | qualifies | current ratio |
|------------------------|-------------:|----:|-----------|---------------|
| DSV3-GateUP-B16-M4096  |         4096 | 7168| YES       | 1.010         |
| DSV3-Down-B16-M2048    |         3584 | 2048| YES       | 0.973         |
| DSV3-Down-B16-M4096    |         7168 | 2048| YES       | 0.973         |
| DSV3-GateUP-B32-M2048  |         4096 | 7168| YES       | 1.002         |
| DSV3-Down-B32-M2048    |         7168 | 2048| YES       | 0.980         |
| DSV3-GateUP-B32-M4096  |         8192 | 7168| YES       | 1.053         |
| DSV3-Down-B32-M4096    |        14336 | 2048| YES       | 0.975         |
| DSV3-GateUP-B16-M2048  |         2048 | 7168| NO (<3200)| 0.989         |
| gpt_oss-*              | ≤ ~1500      | 2880| NO        | 0.929-0.990   |

7 of 16 shapes qualify, all DSV3 (the Down subset ratios 0.97-0.98 +
the GateUP-B16-M4096 ratio 1.01 could lift; biggest opportunity is
Down-* family where we lose 2-3 pp consistently).

Porting constraints (RED LINES from task body):

1. **Persistent single-launch MUST be preserved.** rcr_4w::kernel is
   non-persistent (one tile per block, grid=aligned_grid). For grouped,
   we need a persistent variant: launch `NUM_CUS * 4 = 1024` blocks at
   256 threads each (4 waves/block × 4 blocks/CU = 16 waves/CU max),
   stride `NUM_CUS * 4 = 1024` in the outer tile loop.

2. **Per-tile per-group base-pointer recomputation.** rcr_4w uses
   `a_base + row_tile * HB * a_stride + k_tile * BK`; grouped must use
   `a_base + m_start_g * a_stride + row_tile * HB * a_stride + k_tile * BK`
   and re-derive on every iteration of the persistent loop (since `m_start_g`
   depends on which group the current `gt` maps to).

3. **Happy path only initially.** Gate with `N aligned + K aligned +
   aligned_grid >= 3200 && k <= 8192` — this matches the 7-shape DSV3
   qualifying set above. `grouped_rcr_4w_kernel<0, false, false>` only.

Scope estimate: ~2-3 rounds.

- Round 8: write `grouped_rcr_4w_kernel` skeleton, compile + ensure
  no regression of existing kernels. Wire through dispatcher.
  Correctness FIRST (SNR > 25 dB on all 7 DSV3 shapes that take the
  new path), perf measurement second.

- Round 9: tune schedule (s_waitcnt values, barrier placement); the
  rcr_4w::kernel has sched_barrier(0) stuffing that may behave
  differently under persistent-loop context.

- Round 10+: extend to FUSED_KTAIL + N_MASKED_STORE paths if DSV3 wins
  and gpt_oss benefits from it too (note: gpt_oss B=4 won't qualify
  for 4-wave due to low grid).

## Commits

- HipKittens: `a_kt1` scoping cleanup + this note (single commit).
- Primus-Turbo: mirror note.
