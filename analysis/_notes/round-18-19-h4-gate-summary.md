# Round 18-19 — H4 reroute gating consolidation (BF16 + FP8 bwd wins)

## Summary

Rounds 18 and 19 tightened the H4 reroute (RRR → RCR via b transpose)
in Primus-Turbo to skip the transpose copy when it's net cost. Two
shape-conditional gates added; both verified via clean
`git stash` / `git stash pop` bench runs at the same HK kernel binary
(012c7f0f). Forward metric unaffected (gate is on `not trans_b`,
forward is `trans_b=True`).

## Round 18 — FP8 N-extension (Primus aeb37797)

Original FP8 H4 (round 14, commit 4871e4f) gated on
`a.shape[1] % K_BLOCK != 0` (= K_RRR misaligned). Missed gpt_oss-GateUP
(K_RRR=5760 K_BLOCK-aligned, N_RRR=2880 BLOCK_SIZE-misaligned), which
kept hitting external `grouped_ntail_kernel_lds_rrr<64>` +
`grouped_tail_kernel<RRR>` per dispatch_grouped_rrr (line 4799-4861
of kernel_fp8_layouts.cpp).

Round 18 extended the gate:
```
if not trans_b and ((a.shape[1] % K_BLOCK) != 0
                    or (b.shape[-1] % BLOCK_SIZE) != 0):
```
where `K_BLOCK = 128`, `BLOCK_SIZE = 256` for FP8.

After H4 transpose, the RCR call has K_RCR = K_RRR (aligned), N_RCR =
N_RRR (misaligned). Main RCR kernel uses `bpc = ceil_div(g.n,
BLOCK_SIZE)` (line 4598) + N_MASKED_STORE=true template (line 4641),
single-launch with no external tails.

Bench fp8 (gpt_oss-GateUP 4 cases, clean before/after):

  Case               bwd before  bwd after  Δbwd
  ----------------- ----------- ---------- ------
  GateUP B=4 M=2048    635.05     628.83   -1.0 % (noise)
  GateUP B=4 M=4096    814.30     944.74  +16.0 %
  GateUP B=32 M=2048   605.99     681.41  +12.4 %
  GateUP B=32 M=4096   681.41     995.50  +46.1 %
  ---------------- avg ----------------------------
                       684.19     812.62  +18.8 %

## Round 19 — BF16 K+N gate (Primus c3b70e3)

BF16 H4 (round 9, commit 0cff238) was UNCONDITIONAL
(`if not trans_b:`) — added for gpt_oss-Down correctness because
col_l rt_32x16_s register tile had phantom-read in path-B fuse (round
3-8 attempts). But unconditional H4 paid the transpose cost on every
RRR call including DSV3 (everything aligned, BF16 RRR path runs main
kernel only with `need_tail_run = false`).

Round 19 mirrored round-18 FP8 gate:
```
if not trans_b and ((a.shape[1] % K_BLOCK) != 0
                    or (b.shape[-1] % BLOCK_SIZE) != 0):
```
where `K_BLOCK = 64`, `BLOCK_SIZE = 256` for BF16.

DSV3 (everything aligned) → both conditions false → H4 skipped, native
RRR runs without transpose. gpt_oss-Down (N=2880 misaligned) +
gpt_oss-GateUP (N=2880 misaligned) → still reroute.

Bench bf16 (all 16 cases, clean before/after):

  Group         fwd before    fwd after    bwd before   bwd after   Δbwd
  ------------- ------------- ------------ ----------- ---------- ------
  All 16 cases     1203.83      1207.37      708.43      851.83  +20.2 %

DSV3 8 cases (the movers) saw the transpose elimination directly — for
DSV3-GateUP K=7168 N=4096 with B=16, b shape `[16, 4096, 7168]` =
458 MB; `b.transpose().contiguous()` = 916 MB rd+wr at ~3.4 TB/s ≈
270 µs per dA call. Eliminated cleanly.

## Cumulative state post-rounds 18-19

* Forward K-tail fuse main line: complete (rounds 1-3 RCR fuse path B).
* RRR (dA bwd) fuse: H4 reroute (rounds 14, 9) with shape-conditional
  gates (rounds 18, 19) — single-launch RCR fuse path used for
  K-misaligned shapes; native RRR for fully-aligned shapes.
* External `grouped_ktail_kernel_*` and `grouped_tail_kernel<RRR>`
  launches: NOT triggered on metric+bench paths anymore.
* Score plateau: 833-836 across rounds 13-20. No forward kernel-arch
  wedge has been opened in this round budget (the actual remaining
  wedges are FP8 RRR fuse path B/A hybrid — round 17 docs analysis,
  blocked by rt_128x16_s lane-mapping derivation; or MFMA cell-shape
  rewrite 16x16x128 → 32x32x64 — 1-2 round project, deferred).

## Round 20 status

Round 20 ran metric (835, within plateau band). No kernel/Primus
changes shipped — chat session window approaching limit (81 / 90 min,
~9 min remaining). Round 20 only commits this consolidation note in
HK so future rounds resuming on a fresh chat context can see the H4
gating progression without re-deriving.

## Recommendations for round 21+

1. **Forward kernel rewrite** is the only remaining wedge:
   * MFMA cell-shape change 16x16x128 → 32x32x64 (round 15 docs noted
     this as "fundamental kernel template redesign")
   * FP8 RRR fuse path A hybrid via empirical numerical probe (round
     17 docs) — still untested
2. **DO NOT** retry rule tune / num_xcds / RCR_TWO_TILE_MIN_KI
   (already saturated rounds 1-15 multiple times).
3. **DO NOT** add any new external `grouped_ktail_kernel_*` —
   strictly forbidden by task body. Goal is delete remaining ones
   that are now unreachable on metric+bench paths (cleanup).
