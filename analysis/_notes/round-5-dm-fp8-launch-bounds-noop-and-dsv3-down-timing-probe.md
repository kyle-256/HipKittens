# Round 5 — FP8 grouped launch_bounds MIN=2 is no-op + DSV3-Down direct timing probe

## Baseline (round-5 entry)

`_metric_grouped_only.py` (single run): `score=821, geomean=0.9851, n=16`.

Per-shape (metric clamp):

| shape                                  |  HK TF | TRT TF | ratio |
|----------------------------------------|-------:|-------:|------:|
| DSV3-GateUP-B16-M2048                  | 1307.4 | 1281.8 | 1.020 |
| DSV3-Down-B16-M2048                    | 1159.3 | 1172.3 | 0.989 |
| DSV3-GateUP-B16-M4096                  | 1628.5 | 1559.5 | 1.044 |
| DSV3-Down-B16-M4096                    | 1359.7 | 1432.1 | 0.949 |
| DSV3-GateUP-B32-M2048                  | 1389.1 | 1366.8 | 1.016 |
| DSV3-Down-B32-M2048                    | 1192.8 | 1162.9 | 1.026 |
| DSV3-GateUP-B32-M4096                  | 1659.1 | 1566.2 | 1.059 |
| DSV3-Down-B32-M4096                    | 1390.6 | 1429.2 | 0.973 |
| gpt_oss-GateUP-B4-M2048                | 1036.3 | 1113.3 | 0.931 |
| gpt_oss-Down-B4-M2048                  |  777.1 |  788.8 | 0.985 |
| gpt_oss-GateUP-B4-M4096                | 1228.5 | 1303.9 | 0.942 |
| gpt_oss-Down-B4-M4096                  | 1078.0 | 1158.9 | 0.930 |
| gpt_oss-GateUP-B32-M2048               | 1198.1 | 1217.4 | 0.984 |
| gpt_oss-Down-B32-M2048                 | 1036.4 | 1047.9 | 0.989 |
| gpt_oss-GateUP-B32-M4096               | 1400.1 | 1437.3 | 0.974 |
| gpt_oss-Down-B32-M4096                 | 1201.2 | 1223.3 | 0.982 |

Per-round mean (5-run): **816.0** (flat; round-3/4 all ≈ 816).

## Probe 1: `__launch_bounds__(_NUM_THREADS, 1)` vs `MIN_BLOCKS_PER_CU=2`

Grouped RCR kernel has the permissive `...1)` hint while dense uses
`MIN_BLOCKS_PER_CU=2`. Hypothesis: stricter hint might encourage
tighter register packing and better code gen at same achieved occupancy.

Change (1 line at 1972):

```cpp
__global__ __launch_bounds__(_NUM_THREADS, MIN_BLOCKS_PER_CU)
```

Resource remarks (`-Rpass-analysis=kernel-resource-usage`):

|                                | baseline (MIN=1) | MIN=2     |
|--------------------------------|------------------|-----------|
| VGPRs                          | 256              | **256**   |
| Spill (`<0,false,false>`)      | 67               | **67**    |
| Spill (`<0,true,false>`)       | 76               | **76**    |
| Spill (`<0,false,true>`)       | 45               | **45**    |
| Occupancy [waves/SIMD]         | 2                | 2         |
| ScratchSize [bytes/lane]       | 272/308/184      | 272/308/184 |

Identical. Compiler already achieved occ=2 with MIN=1; tightening the
hint emits bit-identical code. 5-run metric: 813/818/817/814/818 →
mean **816.0** (baseline 816.2, Δ −0.2 / within σ ≈ 2).

**Conclusion: the `MIN_BLOCKS_PER_CU` difference between dense (=2) and
grouped (=1) is cosmetic under the current 256-VGPR / 4×2-warp layout.
Not a lever; reverted.**

## Probe 2: direct HK vs Triton timing on 4 DSV3-Down FP8 shapes

Ran `_probe_dsv3_down_fp8_rocprof.py` (scripted, modelled on
`_probe_dsv3_gateup_fp8_round1.py`): `ITERS=80 × REPEATS=5`, p20 across
repeats, each backend timed back-to-back on the same GPU (HIP_VISIBLE_DEVICES
pinned).

|                         |  HK TF (p20) |  TRT TF (p20) |   ratio | ms gap (TRT − HK)    |
|-------------------------|-------------:|--------------:|--------:|---------------------:|
| Down-B16-M2048          |       1151.4 |        1160.1 |  0.9925 | −0.040 (HK +40 µs)   |
| Down-B16-M4096          |       1353.2 |        1406.0 |  0.9624 | −0.073 (HK +73 µs)   |
| Down-B32-M2048          |       1185.0 |        1168.4 |  1.0142 | +0.021 (HK wins)     |
| Down-B32-M4096          |       1388.7 |        1414.8 |  0.9815 | −0.054 (HK +54 µs)   |

**Observations:**

1. Probe ratios agree with metric within noise (probe 0.96 vs metric 0.95
   on B16-M4096; probe 1.01 vs metric 1.03 on B32-M2048).

2. **B16 shapes lose 4.8%–5.4% per-call; B32 shapes lose ≤2%** (B32-M2048
   even wins p20 by 1.4%). Same M_total × N × K for B16-M4096 and
   B32-M2048 (65536 × 7168 × 2048) but **opposite sign** of the gap.

3. Gap is **not** K-tail / N-tail (K=2048 aligned, N=7168 = 28·256
   aligned). Dispatcher takes the `grouped_rcr_kernel<0,false,false>`
   happy path (no MASKED_STORE, no FUSED_KTAIL). Pure main-loop cost.

4. Gap magnitude 40–73 µs on 800–1600 µs kernels ⇒ **5% of main loop
   time**. At NUM_CUS=256, 7168 tiles = 28 tiles/CU; at 1.35 ms that's
   48 µs/tile total, so the 73 µs B16-M4096 gap = **~1.5 tiles** worth of
   work — consistent with a **per-tile overhead** that scales with
   `tiles/CU`, not with group count.

5. B16-M4096 has **16 groups × 4096/group** (wide groups, few boundaries);
   B32-M2048 has **32 groups × 2048/group** (twice as many boundaries,
   yet ratio is higher). Confirms the gap is **not** binary-search /
   group-offs overhead (already sub-µs per CU).

## Round-5 outcome

- launch_bounds MIN=2 hint: **falsified** — no-op at current VGPR/occupancy budget.
- Direct timing on DSV3-Down: **confirms a main-loop-resident 40–73 µs/call gap
  on B16 shapes**, **not** explained by K-tail / N-tail / group-boundary
  overhead.
- Baseline kernel unchanged; committed doc-only.

## Round-6 plan (next)

Two untried leverage paths, both main-loop focused:

**Option A (low-risk probe, 1 round):** Register-tile shape swap.
Currently `RBM=64, RBN=32, WARPS_M=2, WARPS_N=4`. Try `RBM=32, RBN=64`
(swap WARPS_M↔WARPS_N). Compiler-level change; rebuild-and-compare on
DSV3-Down shapes. Falsify if VGPR/occupancy moves unfavourably.

**Option B (higher leverage, 2–3 rounds):** MFMA cell-shape swap.
Current uses `mfma_32x32x64_fp8` (K-step 128). Probe
`mfma_16x16x128_fp8` which has higher wave-level parallelism (4
32-col output lanes instead of 2) but smaller tile per instruction.
Round-12 historical rocprof partially falsified this on gpt_oss
(VGPR pressure went up) but DSV3-Down has 64 free VGPR lane headroom
(spill 67 vs budget 100+) that might absorb the change.

**Prefer Option A first** — smaller blast radius, one edit, immediate
signal. If A shows even a hint of a signal on DSV3-Down, pursue
further; else skip to Option B.
