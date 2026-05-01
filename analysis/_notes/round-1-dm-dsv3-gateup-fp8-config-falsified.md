# Round-1 (death-march, fresh run) — DSV3-GateUP FP8 (gm, xcd) config-rule falsified

**Date**: 2026-05-01
**HK HEAD**: `a0644b80` (round-19 FLAT→BUFFER mirror for grouped K-tail/N-tail scalar)
**Primus-Turbo HEAD**: `ea6994d` (round-27 FP8 K-tail a_kt1 hoist falsified)
**Segment focus**: `grp_fp8` (BF16 `[watch]` only, does not drive score)
**Score**: 813-822 (noise band ±10; baseline 811, target 1000)

## 1. Motivation

`grp_FP8` segment has 4 DSV3-GateUP shapes at ratio 0.993-1.036 vs Triton
(tiles_n=16, tiles_m ∈ {8, 16}, k=7168). Currently uncovered by the per-shape
FP8 rule set in `primus_turbo/pytorch/kernels/hipkitten/config.py` → falls
through to the binding default `(group_m=4, num_xcds=None → BLOCK_SWIZZLE_NUM_XCDS=8)`.
BF16 has a matching-tile-geometry rule (`tiles_m==8 + tiles_n==16 + k<=7168
→ gm=1, xcd=4`, round-10). Question: does the BF16 winner port to FP8?

## 2. Probe — 8-candidate sweep

`/tmp/probe_dsv3_gateup_fp8_round1.py` (archived in the Primus-Turbo round-1
commit message). 8 `(gm, xcd)` candidates × 4 shapes, ITERS=80, REPEATS=5,
p20-min as the TF estimator. Result:

```
shape                          best cfg    Δ vs (4,None) default
DSV3-GateUP-B16-M2048          (4, 4)        +0.43%  (noise)
DSV3-GateUP-B16-M4096          (1, 2)        +0.64%  (noise-ish)
DSV3-GateUP-B32-M2048          (2, 4)        +0.00%  (flat)
DSV3-GateUP-B32-M4096          (4, None=8)   DEFAULT WINS  (−0.04 to −0.75pp)
```

Worst-case per-cfg delta: `−1.44%` (B16-M2048, `(8, 4)`), `−1.27%` (B16-M2048,
`(4, 2)`). Noise band ≈ ±0.3-0.5% per shape at these settings.

**No single rule-worthy transfer.** The per-shape best winners differ in
`(gm, xcd)`, with margins below the 0.5pp bar that would justify adding a rule
(round-23 note on FP8 gpt_oss-GateUP-B4-M2048 set that bar: "the candidate's
min still beats the default's max only marginally — but the median gap is
consistent across all 7 repeats" is the *minimum* acceptance; this sweep
doesn't clear even the median bar consistently).

## 3. Why BF16→FP8 transfer fails for tiles_n=16 k=7168

BF16 `(gm=1, xcd=4)` wins at `tiles_m==8 + tiles_n==16 + k<=7168` because:
* BF16 `K_STEP=64`, register tile `rt_16x32_s` A / `rt_16x32_s` B → MMA rate
  lower than FP8;
* Main-loop time is balanced between MFMA and LDS traffic, so tile-schedule
  order (gm=1 walking full N-row per M) maximises B-pack L2 reuse;
* BF16 RCR `num_xcds=4` splits the 8-XCD persistent grid into 4+4 halves
  that align with the 8192+ m_total sliced-by-4 pattern.

FP8 RCR `rcr_mma` is a `K=128` MFMA (gfx950 `v_mfma_f32_16x16x32_fp8_fp8`
is really K=128-wide per round-12 probe), MMA throughput is **2×** BF16 per
cell, so main-loop is MFMA-bound (rcr_mma ~60% of kernel cycles per round-12
rocprof). Tile-schedule order matters less than MFMA issue rate — the default
8-XCD spread already saturates the MFMA issue queue.

Register pressure also differs: FP8 `grouped_rcr_kernel<0,true,true>` sits at
256 VGPR ceiling with 67 spill dwords (round-27 doc). BF16 grouped has lower
pressure (~2 pad-dwords typical). FP8 has no spare VGPRs for register-tile
rebinding that (gm=1, xcd=4) might prefer.

## 4. Verdict

**Do NOT add a DSV3-GateUP FP8 rule.** The default is at or near a local
optimum across the 4-shape family; per-shape winners diverge below the noise
bar; B32-M4096 prefers the default over every probed alternative (up to −0.75pp
penalty for non-default). This round (and the task body) explicitly warns
against burning rounds on config tuning; this falsification anchors that bar
for the next agent.

Next-round levers stay in kernel-level territory (main-loop pipeline / LDS
bank conflicts / direct HBM→reg — see the Primus-Turbo round-1 note §4 for
the 7-option menu and round-2 suggestion).

## 5. Files touched

* `analysis/_notes/round-1-dm-dsv3-gateup-fp8-config-falsified.md` (this file)

No kernel code change shipped. Production kernel state unchanged from HEAD
`a0644b80`.
