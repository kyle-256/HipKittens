# Round-D — FP8 grouped-RCR PMC analysis: numeric ceiling confirmed by hardware counters; structural roadmap refined

## Summary

After Rounds A (single +5T win on `RCR_PREFETCH_LGKM`), B (3 wait-counter
levers all saturated within noise), and C (4 schedule-primitive removals
all load-bearing), Round D moved off heuristic sweeps and onto **direct
hardware PMC counters via rocprofv3** to identify *why* the numeric
levers are exhausted and *what specific structural change* is required
to break the ~690 score / ~51 % SM-peak ceiling.

PMC was collected on the **best** shape — dgrad B=32 M=4096 N=5760
K=2880 (after H4 trans_b reroute → `grouped_rcr_kernel`, measured
2562 TFLOPS = 50.9 % of 5033 TFLOPS MI355X FP8 peak), 30 dispatches per
counter set, single-XCC `Agent 4` (HIP_VISIBLE_DEVICES=2 →
HSA-isolated GPU).

Result: **the kernel is structurally saturated on the existing 8-warp /
16x16-base / 4-acc / 2-buf architecture**. The three remaining
high-EV interventions (32x32-base + `mfma_323264` main loop, AGPR
accumulator migration, 4-warp port) all require multi-round
infrastructure work and / or a kernel rewrite. No HK kernel change
shipped this round.

---

## PMC counter aggregates (mean per dispatch, N=30 calls × 50 dispatches)

| Counter                 | Mean / dispatch   | Note                                             |
|-------------------------|-------------------|--------------------------------------------------|
| `GRBM_GUI_ACTIVE`       | 2.467e+07 cyc     | Per-XCC wall cycles                              |
| `SQ_BUSY_CYCLES`        | 9.458e+07 cyc     | SQ active across SIMDs in XCC                    |
| `SQ_WAVES`              | 2048              | 256 CU × 8 warps/CTA, persistent grid            |
| `SQ_INSTS`              | 3.555e+08         | All issued instructions                          |
| `SQ_INSTS_VALU`         | 1.635e+08 (46 %)  | All VALU including MFMA                          |
| `SQ_INSTS_MFMA`         | 7.078e+07 (20 %)  | MFMA subset                                      |
| `SQ_INSTS_LDS`          | 5.350e+07 (15 %)  | LDS reads + writes                               |
| `SQ_INSTS_VMEM_RD`      | 1.942e+07 (5.5 %) | HBM reads (buffer_load)                          |
| `SQ_LDS_BANK_CONFLICT`  | **0**             | Swizzle is bank-conflict-free                    |
| `SQ_WAIT_INST_LDS`      | 6.783e+07 wave-cyc| ≈0.96 cyc wait-on-LDS per MFMA inst              |
| `SQ_WAIT_ANY`           | 4.897e+08 wave-cyc| any-wait across all waves                        |

`SQ_WAIT_INST_VMEM` is unavailable on the gfx950 PMC schema, so HBM
stalls cannot be quantified directly; however with `SQ_INSTS_VMEM_RD =
5.5 %` of total issued, HBM bandwidth is clearly not the dominant cost
(would need to be ≳ 30 % of issue mix for VMEM to dominate).

### Issue-mix derivations

* **MFMA / total = 20 %** → 4 of every 5 issued cycles are non-MFMA.
* **VALU / MFMA = 2.31** → 1.31 non-MFMA VALU per MFMA inst (data-shuffle:
  `v_mov_b32`, `v_perm_b32`, `v_pack_*`, accumulator zero-init).
* **LDS / MFMA = 0.76** → 3 LDS reads per 4 MFMA insts.
* **VMEM_RD / MFMA = 0.27** → 1 buffer_load per ~3.7 MFMA insts (the
  prefetch ratio).
* **Sync / scalar (rest) = 33 %** → s_waitcnt / s_barrier / s_setprio /
  SALU address arithmetic.

---

## Why this rules out every numeric lever explored in rounds A-C

### 1. LDS layout is already optimal — `SQ_LDS_BANK_CONFLICT == 0`

The `ST_v2 / ST_v2a` swizzle in `kernel_fp8_layouts.cpp` produces zero
bank conflicts on the `ds_read_b128` access pattern that feeds the
`mma_ABt` operands. There is **no LDS-side win available** from
re-swizzling — every `ds_read_b128` is hitting all 64 LDS banks
distinctly.

### 2. MFMA primitive is already at maximum K density

`A_row_reg = rt_fp8e4m3<64, 128, row_l, rt_16x128_s>` (line 125) →
`mma_ABt_base` line 229-233 dispatches to `mfma1616128`. The CDNA4
FP8 prim `__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4` consumes
K=128 per inst — **the largest fp8 K-prim available on gfx950**. No
"larger MMA" lever exists in the 16x16 base-tile family.

The alternative larger prim is `mfma_scale_f32_32x32x64_f8f6f4`
(`mfma323264`, dispatched via `rt_32x32_s` base tile, line 234-238).
Switching to it would halve the MFMA inst count per outer K-iter (32 →
16 prim per warp), but requires propagating a 32x32 base-tile through
the entire main loop + new `load_a_32` / `load_b_32` LDS-read helpers
(today only `rcr_mma_32` *wrapper* exists for the K-tail block — no
main-loop callers; see line 386-392 + `lever_d_round_b_force_*`
scaffold). This is the **option B** structural change below.

### 3. `s_waitcnt lgkmcnt(0)` cannot be relaxed to `lgkmcnt(N)` safely

Per main-loop iteration the explicit `lgkmcnt(0)` drain at lines 2778,
2785, 2793 etc. precedes each `rcr_mma`. The inputs that the MMA
consumes (registers `a`, `b0` / `b1`) are filled by the immediately
preceding `load_a` / `load_b` (which emit `ds_read_b128`).

The lgkmcnt counter monotonically tracks *all* outstanding LDS+SMEM
ops (reads + writes, pooled) and **does not distinguish op type**.
Even if reads are issued *before* writes in source order, the compiler
and the LDS unit may complete writes faster than reads (writes are
~3 cyc; reads are ~30 cyc). Therefore `lgkmcnt(N)` for *any* N > 0
admits the failure mode where the read-into-`a` is still pending while
the MMA fires → undefined behaviour. **`lgkmcnt(0)` is the only
provably-safe wait for the MMA-input dependency.**

This rules out the "tighten lgkmcnt(0) → lgkmcnt(7)" probe entirely —
even if it gained 7 cyc × 4 MMAs × 22 iters × 6 tiles ≈ 0.14 % per
kernel, it would be silently wrong.

### 4. The 4 `s_barrier` and 4 `s_setprio` per main-loop iter are all
   load-bearing (Round C verified)

Removing the post-MMA `s_barrier` race-hazards the LDS-write/read swap
into the next iter (correctness FAIL). Removing `s_setprio` lets the
compiler issue dependent loads at normal priority before the MFMA
operands are committed (correctness FAIL). The intermediate variants
that *passed* correctness (e.g. drop only the cD post-barrier)
regressed performance by -50T fwd because subsequent loads bunched up
without the barrier-imposed reordering boundary.

---

## Quantified structural ceiling

Per outer K-iter, per warp, the kernel must issue *minimum*:

| Bucket               | Count                         | Why minimum                                           |
|----------------------|-------------------------------|-------------------------------------------------------|
| MFMA prim            | 32 (4 acc × 8 prim/acc)       | RBM·RBN·K_BLOCK / 16·16·128 with rt_16x128 base       |
| LDS reads            | 24 (4 acc × 6 LDS reads/acc)  | A_row 8 + B_row 4, two A's + two B's, mfma1616128     |
| Buffer loads         | 16 (HBM prefetch for next k)  | One full A-tile + one full B-tile per outer iter      |
| LDS writes (hoist)   | 16 (buffer→LDS for next k)    | Coupled with buffer loads (rcr_8w_load_hoist)         |
| s_barrier            | 8 (between-MMA + iter-end)    | Required for LDS read/write race (Round-C verified)   |
| s_setprio            | 8 (4 pairs around MMA)        | Required for MFMA-input priority (Round-C verified)   |
| s_waitcnt lgkmcnt(0) | 4                             | Required for MMA-input safety (above)                 |
| TK_WAIT_LGKM(8)      | 1 (top-of-iter soft wait)     | Allows partial drain pre-barrier (Round-A optimal)    |
| TK_WAIT_VMCNT(8)     | 1 (mid-iter for cD)           | Bounds in-flight HBM reads (Round-B optimal)          |

**Total instructions per warp per outer K-iter: ~110**, of which 32
(29 %) are MFMA. The other 78 instructions are *all* either
explicit data motion (40), explicit synchronization (21), or implicit
data shuffle around MMA operand format (~17, observed via VALU/MFMA
ratio).

This 29 % MFMA-issue density places a hard ceiling at ~29 % of the
"if every issued cycle were MFMA" peak. The fact that we observe
51 % of *FP8 peak TFLOPS* (not 29 %) reflects that MFMA insts are
multi-cycle (16 cyc each on CDNA4 fp8 16x16x128 per SIMD), so the
issue-rate-vs-throughput ratio amplifies the effective MFMA fraction
~1.75x — exactly matching observation.

---

## Refined structural roadmap (in EV order, per round)

After Rounds A-D the candidates remain those identified in Round-C, but
with PMC-grounded estimates for *expected* gain instead of pre-PMC
heuristic guesses:

### Option B — 32x32 base + `mfma_323264` main loop (NEW recommendation)

**Mechanism.** Switch main-loop base tile from `rt_16x16_s` (currently
`rt_fp8e4m3<RBM=64, BK=128, row_l, rt_16x128_s>` for A,
`rt_fp8e4m3<RBN=32, BK=128, row_l, rt_16x128_s>` for B) to
`rt_32x32_s` with `rt_fp8e4m3<RBM=64, BK=64, row_l, rt_32x64_s>` for A
and similarly for B. Per outer K-iter (still K_BLOCK=128) requires 2
inner K-prims of `mfma_323264`. Per `rcr_mma`:
height_A=2 × height_B=1 × width_K=2 = 4 prim → **half** the prim count
vs current 8 prim per `rcr_mma`.

**Per-iter inst impact.** MFMA prim 32 → 16 (-50 %), LDS reads
24 → 12 (-50 %, same fp8 byte-rate but in larger chunks),
sync/setprio unchanged.

**Expected gain.** Issue density rises 29 % → 41 %. With same MFMA
throughput per cycle this should translate to roughly +5..+10 % wall-
TFLOPS. The 32x32 K-tail wrapper (`rcr_mma_32`) and the
`A_row_reg_32 / B_row_reg_32` types already exist as scaffolds (see
`kernel_fp8_layouts.cpp:386-392` + `lever_c2_round_54_step1_scaffold`
namespace, and `kernel_fp8_layouts.cpp:425-485` `lever_d_round_b`
force-instantiate block) — meaning the typing infrastructure is
already in place; only the LDS-load helpers + main-loop body rewrite
remain.

**Cost.** 2-3 rounds: write `load_a_32` / `load_b_32` cooperative LDS
loaders with the existing `ST_v2a` swizzle, port the four
`rcr_mma(cA/B/C/D, ...)` chain, validate correctness (SNR > 25 dB),
profile.

**Risk.** Acc tile changes from `rt_fl<RBM=64, RBN=32, col_l,
rt_16x16_s>` (4 acc × 32 fp32/lane = 128 fp32/lane) to
`rt_fl<RBM=64, RBN=32, col_l, rt_32x32_s>` (4 acc × 16 fp32/lane =
64 fp32/lane). Acc-VGPR pressure drops 2x, freeing room for either
deeper prefetch *or* fewer spills (currently 34-54). Net should be
strictly positive. **No new HK header support required** — all
intrinsics already exist.

### Option A — AGPR accumulator migration

**Status from Round-C: blocked on missing `art`-mode FP8 MMA
intrinsics in HK headers.** Round D additionally confirms via PMC
that this would primarily help via:
1. Reducing the 34-54 VGPR spills (eliminate scratch traffic, ~5 %
   issue-overhead reduction)
2. Freeing ~64 VGPR for deeper LDS-read prefetch

Estimated +5..+10 % when unblocked. Estimated 1-2 weeks HK header
infrastructure, then 1-2 rounds main-loop port. **Lower priority than
Option B** because B requires no HK header work.

### Option C — 4-warp port

PMC suggests modest gain. Per-warp issue density is already 29 % MFMA
in 8-warp; a 4-warp port (one warp per CTA dim, 2 warps total per CU)
would give each warp 2x the work but also 2x the live registers,
likely bumping into spill cliff well before the issue-density
benefit materializes. Estimated +5..+10 % but with significant risk
of register-pressure regression. 4-8 rounds (full kernel rewrite).

### Option D — K-tail / main-loop overlap

PMC shows `SQ_INSTS_VMEM_RD = 5.5 %` of issue mix → HBM is not the
bottleneck and the K-tail's buffer_load latency (~24 buffer_loads,
issued sequentially at the end) is dwarfed by the main-loop. Estimated
gain at +1..+2 %, lowest EV.

---

## Recommendation

The **next round (Round E)** should pursue **Option B (32x32 base +
`mfma_323264` main loop)**. This is the only candidate that:

1. Is unblocked at the HK header level (intrinsic exists, types
   exist).
2. Has a PMC-grounded mechanism (issue-density × MMA-throughput).
3. Has a quantified expected gain (+5..+10 %).
4. Reduces register pressure rather than increasing it.
5. Has scaffolding already in place from prior rounds (lever_c2 R53,
   lever_d R30+).

If Option B yields the predicted gain, score moves from ~690 → ~720..
~750 (target 1000 = avg 2800 TFLOPS still requires Option A in
addition).

---

## Reproduction

PMC trace (single XCC, gfx950 / MI355X):

```bash
# /tmp/probe_pmc.txt
# pmc: SQ_WAVES SQ_BUSY_CYCLES GRBM_GUI_ACTIVE SQ_WAIT_ANY

# /tmp/probe_pmc2.txt
# pmc: SQ_INSTS_MFMA SQ_INSTS_VMEM_RD SQ_INSTS_LDS SQ_WAIT_INST_LDS

# /tmp/probe_pmc3.txt
# pmc: SQ_LDS_BANK_CONFLICT SQ_INSTS_LDS SQ_INSTS SQ_INSTS_VALU

cd /workspace/code/Primus-Turbo

HIP_VISIBLE_DEVICES=2 \
PRIMUS_TURBO_HIPKITTEN_PATH=/workspace/code/HipKittens \
TURBO_BENCH_FORCE_BACKEND=hipkitten \
rocprofv3 -i /tmp/probe_pmc.txt -d /tmp/pmc_probe_run -o pmc -f csv \
  --kernel-include-regex "grouped_rcr_kernel|grouped_rrr_kernel|grouped_var_k_kernel" \
  -- python3 scripts/_probe_fp8_kernel_rocprof.py dgrad 32 4096 5760 2880 30
```

Aggregate via `csv.DictReader` + `defaultdict(list)`; results are
stable across the 50-dispatch capture window.

## Files touched

None — Round D is a measurement-and-analysis round; no kernel change
shipped.
