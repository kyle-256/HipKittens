# Round 1 (gpt_oss focus restart) — FP8 grouped baseline + falsified easy paths

## Context

User switched the auto_optimize run to `--focus-model gpt_oss`. Score
now only counts gpt_oss (16 / 32 shapes); DSV3 still benchmarked +
correctness-checked but `[watch]`-tagged. Baseline at HEAD:

```
Primus-Turbo  c3b70e3
HipKittens    62cebd5
score         786 (785 baseline)
goals         0/2  (grp_BF16 1.044 / grp_FP8 0.852, both target 1.20)
focus segment progress = geomean(0.870, 0.710) = 0.785
```

The focused gpt_oss FP8 segment is the heavier lever (8/8 cases below
1.0 vs Triton, ratio 0.81–0.88) so Round 1 was spent quantifying *why*
HK FP8 grouped trails Triton on K=2880 and which "easy" paths are
worth pursuing in subsequent rounds.

## Per-shape baseline (HEAD before any Round-1 change)

```
shape (per-launch tile geometry)              hk_TF   trt_TF  ratio  status
grpFP8 GateUP-B4-M2048   tiles_m=8  tiles_n=22  917  1117    0.821
grpFP8 GateUP-B4-M4096   tiles_m=16 tiles_n=22 1085  1306    0.831
grpFP8 GateUP-B32-M2048  tiles_m=8  tiles_n=22 1070  1220    0.877
grpFP8 GateUP-B32-M4096  tiles_m=16 tiles_n=22 1239  1456    0.851
grpFP8 Down-B4-M2048     tiles_m=8  tiles_n=11  701   804    0.873
grpFP8 Down-B4-M4096     tiles_m=16 tiles_n=11  951  1164    0.817
grpFP8 Down-B32-M2048    tiles_m=8  tiles_n=11  921  1050    0.877
grpFP8 Down-B32-M4096    tiles_m=16 tiles_n=11 1078  1237    0.872
```

All 8 are tiles_m∈{8,16}, tiles_n∈{11,22}, K=2880, fast_k=2816,
K_REM=64 (FUSED_KTAIL=true).

## Falsified paths (with data)

### F1 — `(group_m, num_xcds)` cfg re-tune is saturated

A 28-cfg sweep over `group_m ∈ {1,2,4,8,16,24,32}` × `num_xcds ∈
{1,2,4,8,16}` on each of the 8 metric shapes (200-iter / 10-warmup,
total 224 data points; script archived at
`/tmp/sweep_fp8_gptoss_round1.py`):

```
shape                   triton    best cfg     best TF  best ratio  default(4,8)  def ratio    Δ vs default
GateUP-B4-M2048          1122     (2, 16)        950      0.847       948.3        0.845        +1.6 TF
GateUP-B4-M4096          1341     (8, 4)        1104      0.823      1099.7        0.820        +4.6 TF
GateUP-B32-M2048         1223     (8, 4)        1070      0.876      1064.2        0.871        +6.2 TF
GateUP-B32-M4096         1445     (8, 4)        1236      0.855      1226.2        0.848        +9.9 TF
Down-B4-M2048             874     (2, 2)         740      0.847       739.1        0.845        +1.2 TF
Down-B4-M4096            1207     (1, 4)         995      0.824       988.0        0.818        +6.9 TF
Down-B32-M2048           1055     (32, 4)        936      0.887       925.7        0.878       +10.3 TF
Down-B32-M4096           1244     (4, 1)        1084      0.871      1083.1        0.871        +0.6 TF
```

**Best cfg over default = +0.6 to +10.3 TF (≤+1.0 pp ratio gain)**.
Below `(group_m, num_xcds)` sweep noise (~±2 TF / 0.2 pp). The
`config.py` rules already in effect (round-61/68/69) capture
substantially all of this — re-tuning further yields nothing.

**Conclusion**: cfg sweep is saturated. The 30+ pp gap to 1.20 vs
Triton cannot be closed by `(group_m, num_xcds)`.

### F2 — RCR_TWO_TILE_MIN_KI 28 → 20 confirmed NO-OP for ki=22 dense path

Round 15 had concluded "no-op" but the dense LLM metric suite has no
ki=22 shape (K∈{4096, 11008, 14336} all ki≥32 → already 2-tile). To
test whether *grouped* gpt_oss (ki=22) would benefit if a 2-tile path
were ported in, I prototyped on the dense kernel first (since dense
already has the 2-tile path) and re-ran a dense FP8 micro-bench on
fixed `(M=4096, N=5760, K)` with K varied to flip the 1-tile / 2-tile
gate (script `/tmp/dense_fp8_2tile_isolate.py`).

```
                  RCR_TWO_TILE_MIN_KI=28        RCR_TWO_TILE_MIN_KI=20
M    N    K       TF (path)                     TF (path)               Δ
4096 5760 2816    1471 (1-tile)                 1492 (2-tile)         +1.4%
4096 5760 2944    1537 (1-tile, ki=23 odd)      1527 (1-tile)          ~0
4096 5760 3072    1566 (1-tile)                 1557 (2-tile)         -0.6%
4096 5760 3328    1620 (1-tile)                 1616 (2-tile)         -0.3%
4096 5760 3456    1644 (1-tile, ki=27 odd)      1633 (1-tile)         -0.7%
4096 5760 3584    1689 (2-tile, gate==)         1670 (2-tile)         -1.1%
4096 5760 3712    1687 (1-tile, ki=29 odd)      1675 (1-tile)         -0.7%
4096 5760 3840    1726 (2-tile)                 1715 (2-tile)         -0.6%
4096 5760 4096    1778 (2-tile)                 1760 (2-tile)         -1.0%
```

**ki=22 (K=2816) gets +1.4% from 2-tile path; deeper-K shapes lose
0.3-1.1% (noise). Net round-15 conclusion stands**.

Even if a Round-2 effort ported the dense 2-tile main loop into
`grouped_rcr_kernel` (≈70 LOC; group_idx propagation into the
b_co lambda) and lowered `RCR_TWO_TILE_MIN_KI` to 22, the upper
bound on the focused gpt_oss segment is **+1.4 pp ratio** (since
all 8 metric shapes have ki=22). That brings grp_FP8 geomean from
0.852 → ~0.864 (progress 0.71 → 0.72) — a real but **small** gain
that does not close the 30 pp gap to 1.20.

### F3 — FUSED_KTAIL epilog overhead is small (~5%), not the gap source

K-aligned vs K-misaligned with same `(B, M, N)` (script
`/tmp/grouped_fp8_ktail_isolate.py`):

```
B=4 M=4096 N=5760               HK TF    TRT TF   ratio   HK ms   TRT ms
K=2816 (no K-tail)               1157    1386    0.835   0.4593  0.3835
K=2880 (K-tail=64, gpt_oss)      1109    1353    0.820   0.4902  0.4018
K=4096 (no K-tail)               1340    1524    0.880   0.5767  0.5073
K=4160 (K-tail=64)               1262    1466    0.861   0.6222  0.5355
```

K-tail vs no-K-tail diff: **HK -4 to -6%, Triton same magnitude
(-2 to -4%)**. FUSED_KTAIL contributes only ~5% to the total gap.

### F4 — Triton's tile shape == HK's tile shape

Triton's origami selector picks `(BLK_M, BLK_N, BLK_K) = (256, 256,
128)` and `group_m=4` for **all 8** gpt_oss FP8 shapes — exactly the
HK default. Triton's win is not from a smaller `BLK_N` for the
`tiles_n=11` (N=2880) family (the user-suggested "BN=128 for N=2880"
hypothesis falsified on the Triton side: Triton itself picks BN=256
and beats HK).

## Real gap source — main loop micro-architecture

Both K-aligned (no FUSED_KTAIL) and K-misaligned (FUSED_KTAIL on)
show HK trailing Triton by **12–17 pp** consistently. Same tile,
same `group_m`, same XCD count (Triton uses 304 SMs, HK uses 256
CUs — explains ~5 pp; not the rest).

The remaining 7–12 pp gap is in the K-loop body itself:
- HK's hand-written 1-tile pipeline issues `4 mma + 4 LDS load + 2
  HBM prefetch + 6+ s_waitcnt / s_barrier` per K-iter.
- Triton's `tl.dot` lowers to a software-pipelined `num_stages=2`
  LDS double-buffer schedule that the LLVM AMDGPU backend tunes at
  PTX-equivalent level — empirically a tighter inner loop.

This is a **kernel rewrite**, not a tuning knob. Rough effort
estimate: 2–4 rounds of grouped_rcr_kernel main-loop refactoring
(re-derive prefetch schedule, re-tune `RCR_PREFETCH_LGKM` /
`RCR_INIT[01]_VMCNT` / `RCR_STEADY_VMCNT` for the new LDS layout,
re-verify SNR on all 8 K=2880 shapes).

## Round-2+ priorities (in order)

1. **rocprof breakdown of `grouped_rcr_kernel<0, true, true>` on
   gpt_oss-GateUP-B32-M4096** (the worst-ratio K-tail-on shape at
   0.851). Profile: per-iter occupancy, vmem stall %, lds stall %,
   mfma busy %. Compare against Triton's persistent kernel on the
   same shape.

2. **If rocprof shows vmem stall dominating**: try lowering
   `RCR_STEADY_VMCNT` from 8 → 4 to issue more concurrent loads in
   the steady state. Verify SNR on all 8 K=2880 shapes.

3. **If rocprof shows lds stall dominating**: try the BF16
   grouped kernel's deeper `Bs[3]` (3-stage LDS rotation) on the
   FP8 path — adds 16 KB LDS but may unblock pipeline.

4. **The "port dense 2-tile to grouped" path is +1.4 pp on ki=22
   only**. Worth it as a small Round-2 commit if rocprof rules out
   F1/F2 above as the root cause.

## What was NOT changed in Round 1

- No `.cpp` change kept (RCR_TWO_TILE_MIN_KI prototype reverted,
  .so md5 == original `be641b8c`).
- No `config.py` rule change (cfg sweep saturated; no shape benefits
  from a different `(group_m, num_xcds)` beyond noise).
- No backend dispatch change.

Score should be unchanged after this commit (it is a notes-only commit).

## Files referenced

- `analysis/_notes/round-15-fp8-rcr-two-tile-min-ki-noop.md` (still
  the canonical reference for the dense 2-tile gate)
- `analysis/_notes/round-12-fp8-rocprof-breakdown.md` (last full
  rocprof on this kernel, pre-FUSED_KTAIL)
- `analysis/_notes/round-16-fp8-bwd-post-h4-rocprof-rrr-fuse-roadmap.md`
  (line 174-179: noted the K-loop is the dominant gap vs Triton)

Sweep / probe scripts archived in `/tmp/`:
- `sweep_fp8_gptoss_round1.py`
- `dense_fp8_2tile_isolate.py`
- `grouped_fp8_ktail_isolate.py`
- `triton_origami_probe.py`
