# Round-B — gpt_oss FP8 backward (wgrad/dgrad) wait-counter levers saturated; no kernel change

## Goal

Mirror the round-A protocol on the backward kernels (`grouped_var_k_kernel_fp8`
for wgrad, `grouped_rrr_kernel` for dgrad) to find a wait-counter
adjustment that lifts the worst section (wgrad: 1782 / 2800 = 64% of
target) above the run-to-run noise floor.

## Levers swept

Same metric harness as round-A (`_metric_gpt_oss_fp8_kernel.py`,
8 gpt_oss shapes × 3 sections, MI355X GPU 2). All sweeps ran with
the round-A `RCR_PREFETCH_LGKM=8` already applied, so the baseline
reflects post-round-A state (mean score ≈ 688).

### Lever B1 — `CRR_PREFETCH_LGKM` (wgrad main-loop LGKM)

Initial 7-runs-per-cell sweep (run back-to-back per cell) showed
`CRR_LGKM=4` at score median 691 / wgrad 1792 vs baseline `CRR_LGKM=3`
at 686 / 1781 — apparent +5/+11 win.

Skeptical re-test using **A/B alternating with rebuild between every
iteration** (8 paired trials, A=CRR_LGKM=3, B=CRR_LGKM=4):

```
iter 1 A: 685 / 1780    iter 1 B: 691 / 1789  Δ = +6 / +9
iter 2 A: 690 / 1792    iter 2 B: 686 / 1777  Δ = -4 / -15
iter 3 A: 686 / 1782    iter 3 B: 691 / 1795  Δ = +5 / +13
iter 4 A: 690 / 1792    iter 4 B: 690 / 1785  Δ =  0 / -7
iter 5 A: 691 / 1791    iter 5 B: 691 / 1793  Δ =  0 / +2
iter 6 A: 691 / 1788    iter 6 B: 691 / 1794  Δ =  0 / +6
iter 7 A: 686 / 1782    iter 7 B: 691 / 1791  Δ = +5 / +9
iter 8 A: 690 / 1788    iter 8 B: 690 / 1792  Δ =  0 / +4

aggregate: A wgrad=1786.9, B wgrad=1789.5 → Δ = +2.6 T (+0.15%)
           paired sign test on wgrad: 5/8 positive, p ≈ 0.14 (NOT significant)
```

The +5T per-cell apparent win in the back-to-back sweep was correlated
thermal/state drift between cells, not a causal kernel-level
improvement. **CRR_LGKM=3 (current value) is at the empirical optimum.**

### Lever B2 — `CRR_STEADY_VMCNT` (wgrad mid-iter VMCNT wait)

Interleaved sweep with rebuild between every metric run, 5 reps per
cell across `{2, 3, 4 (baseline), 6, 8, 12}`:

```
vmcnt | score   wgrad
    2 | 655.6   1501.6   ← cliff: -280 T regression (wgrad starves)
    3 | 660.8   1551.2   ← cliff: -230 T regression
    4 | 687.6   1783.8   ← baseline (current value)
    6 | 688.2   1785.0
    8 | 688.8   1786.4   ← marginal best, +2.6 T wgrad (within noise)
   12 | 688.0   1784.2
```

VMCNT < 4 is a hard cliff: the wait fires before HBM loads complete,
the next-iter mma reads stale data and the wgrad output is partially
corrupted (numerically — the SNR gate would catch full corruption but
small-magnitude tail noise still drops effective TFLOPS via the
correctness-gated zero contribution path).

VMCNT ≥ 6 is monotonic noise (±1.5 T wgrad across 6/8/12). The current
value 4 is at the cliff edge and the marginal +2.6 T at VMCNT=8 is
within run-to-run noise (~3 T 1σ on wgrad at 5 reps). **CRR_STEADY_VMCNT=4
(current value) is at the empirical optimum.**

### Lever B3 — `RRR_PREFETCH_LGKM` (dgrad main-loop LGKM)

Mirror of round-A, applied to the dgrad kernel. Interleaved sweep with
rebuild between every metric run, 4 reps per cell across
`{4, 6, 8 (baseline), 10, 12, 16}`:

```
lgkm | score   dgrad
   4 | 686.5   2084.5
   6 | 688.5   2092.2
   8 | 688.0   2086.2   ← baseline (current value)
  10 | 689.2   2092.5   ← marginal best, +6.3 T dgrad (within 1.5σ)
  12 | 686.5   2084.5
  16 | 687.2   2087.2
```

LGKM 6/8/10 are all within ±5 T dgrad of each other (~1σ noise on
dgrad at 4 reps). LGKM=10 has the highest mean by +1.2 score points
but the per-iter Δ-to-baseline is `{+1, +5, +0, +1}` with one
"miss-the-other-direction" cycle — too noisy to commit. **RRR_PREFETCH_LGKM=8
(current value) is at the empirical optimum.**

## Conclusion

All wait-counter tunables for the backward kernels are saturated
within the run-to-run noise floor (~±5 T per section, ±1 score point).
The total observable headroom on numeric tunables alone is ≤ +1 score,
which is below the noise floor of the metric. **No kernel change shipped
in round-B.**

## Why this round-B is structurally bounded

The remaining headroom on the FP8 grouped kernels (gpt_oss family,
K%128==64) is dominated by **physical resource saturation**, not by
schedule tuning:

* **VGPR pressure**: `grouped_rcr_kernel<*,*>` reports
  `VGPRs=256, AGPRs=0, Spill=34-54` (post-round-A). The compiler is
  out of registers; every spill is a scratch RMW = 10-20 cycles, and
  the spill count contributes ~5-8% of per-tile time. This cannot be
  fixed by any wait-counter change — only by moving accumulators to
  AGPR (R8-dm note: `round-8-dm-fp8-rcr4w-port-plan-invalidated.md`).

* **AGPR migration is blocked by missing FP8 art-mode mma**: HK's
  `art` (assembly register tile) layer in
  `include/ops/warp/register/tile/assembly/mma.cuh:307` only
  supports bf16 / half operand types — there is no FP8 art-mode
  mma intrinsic wired up. Adding one is multi-week infrastructure
  work outside the scope of "two manual rounds".

* **4-warp port is a same-scale rewrite**: the R57-59 lever_c2
  compile tests demonstrate that switching to 4 warps yields
  AGPR=256 + Spill=0 (via the LLVM heuristic that picks AGPR when
  per-warp accumulator ≥ 256 fp32/lane). But porting the persistent
  grouped kernel from 8w → 4w changes load layouts, sync patterns,
  and doubles per-warp register pressure on a/b operands too —
  estimated 3-6 rounds of cooperative kernel work.

* **K-tail epilog overhead is structural**: gpt_oss K=2880 = 22*128 + 64,
  so the fused K-tail epilog (path B, line ~2852+ of cpp) adds 4
  extra mfma + 2 buffer_load_b128 per tile after the main loop ends.
  Theoretical lower bound: K=2880 takes 22+1 = 23 K-iterations of
  compute even on a hypothetically perfect kernel = 4.5% overhead
  per tile. Some of this could be overlapped with the last main-loop
  iter, but doing so requires reformulating the dependency chain
  between cD and the K-tail mma — round-3-fp8-ktail design work.

## Score progression after manual rounds

```
baseline (R5 auto-optimize, commit 7637aae)         : 686 (mean 687.4)
round-A (RCR_PREFETCH_LGKM 4 → 8)                   : 688 (mean 690.0)
round-B (negative finding; no kernel change)        : 688 (mean 690.0)
                                                      ─────────────────
                                                      manual gain: +2.6 mean
```

Headroom to the 2800 T per-section target (score = 1000) requires
structural kernel work, not parameter tuning. The current state is
near-saturated on the existing 8-warp / VGPR-acc / single-tile-schedule
architecture.

## Suggested next focus areas (out of round-B scope)

1. **Implement FP8 art-mode mma** in HK headers. Once available, port
   `grouped_rcr_kernel`'s 4 accumulators from VGPR `rt_fl` to AGPR
   `art<float, RBM, RBN, col_l, rt_16x16_s>`. Expected gain: ~5-10%
   from spill elimination + freed VGPR for memory ILP.

2. **Reduce the s_barrier count in the steady-state main loop**.
   Currently 4 `__builtin_amdgcn_s_barrier()` per K-iter (~50-100 cyc
   each). Round-2 falsified removing setprio + sched_barrier together
   (-5.6%); a more targeted experiment isolating the 2nd and 4th
   barriers (which gate the cD→cA tile-flip but might be coverable
   by the existing `s_waitcnt lgkmcnt(0)`) is unattempted.

3. **K-tail overlap with main loop's last iteration**. For
   `FUSED_KTAIL=true` shapes, the cD mma of the last main iter and
   the cA K-tail mma have no data dependency — they could be issued
   back-to-back and the lane-zero K-tail load (`raw_buffer_load_b128`)
   could be hoisted to overlap with the main-loop epilog VMCNT wait.
   Estimated 1.5-3% gain on K%128==64 shapes.
