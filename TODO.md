# TODO — FP8 / BF16 GEMM on MI350X

## Ground Rules

- **NO JIT per-shape compilation**. Single `.so` per target (`tk_fp8_layouts.so`,
  `tk_bf16_layouts.so`) built by `make`.
- SNR ≥ 48 dB (FP8) / ≥ 47 dB (BF16 vs torch.mm), bit-exact determinism are hard gates.
- Never commit `*.so`, `.autotune_cache.json` is OK to keep (it's text), logs are not.

## Current Status (2026-04-18, post-P14 partial — Dev F memo landed as research; D/E in flight)

P14 launched 3 Devs (D: FP8 CRR_STEADY 2D sweep on GPU 0; E: BF16 CRR
`__launch_bounds__(_,1)` KI=296 surgical re-attempt on GPU 4; F: FP8 RCR
weak-shape per-shape rocprofv3 research on GPU 6). Session ended (loop
runtime expired) with **only Dev F complete**; D and E are mid-bench
in their worktrees and should be picked up next session.

- **Dev F (FP8 RCR weak-shape rocprofv3, GPU 6, complete)** — definitive
  per-shape PMC memo. All 6 weakest TK_RCR shapes show identical 0.92×
  speedup vs BL with **flat 0.75 LDS/MFMA on TK vs flat 0.50 on BL**
  (TK does 1.47–1.50× more LDS-issued instructions per MFMA than BL on
  every weak shape). hipBLASLt picks the same Tensile family for all 6:
  `Cijk_Alik_Bljk_F8BS_..._MT256x256x128_MI16x16x1_..._DTLA1_DTLB1_PGR2_PLR0_SK3_...`
  → MT256×256×128 + 256-thread WG + StreamK SK3 persistent grid +
  PGR2 (DTL on both operands) + PLR0 (no LDS prefetch). Ranked levers:
  (A) DTL on both operands, (B) StreamK persistent grid, (C) wider LDS
  reads. Dev F flagged inconsistency: 0.75 LDS/MFMA + 0 bank conflicts
  looks like reg-staged stores, not pure DTL. **Decider disassembly
  recheck (this session, on `kernel_fp8_layouts-hip-amdgcn-amd-amdhsa-gfx950.o`):**
  658× `buffer_load_dwordx4`, 1336× `ds_read_b128`, 1360×
  `ds_read_b64_tr_b8`, 2688× `v_mfma_f32_16x16x128_f8f6f4`,
  **0× `ds_write*`** — reconfirms P13 Dev C: TK already uses DTL on
  *both* A and B. The PMC LDS-instruction counter Dev F observed counts
  consumer-side `ds_read` ops; the 1.47× gap is therefore **wider /
  fewer LDS reads + StreamK** (BL likely uses `ds_read_b128_tr_b16` or
  similar to halve ds_read count per MFMA), NOT adding DTL. Ranked
  P15 levers reduce to: **(B) StreamK persistent grid, (C) wider LDS
  reads / `ds_read_b128_tr_b16` if applicable.**
- **Dev D (CRR_STEADY 2D sweep, GPU 0, in flight)** — running 16-config
  grid (S1∈{0,2,4,6}, S2∈{0,2,4,6}) on Dev F P12 worktree
  `agent-a5b15c06`. First 2 configs returned: S1=0,S2=0 → CRR geo-mean
  vs BL 1.955× (TK avg 2627.9); S1=0,S2=2 → 1.962× (TK avg 2628.7).
  Δ = +0.007×, well within DVFS noise. ~5 min/config × 14 remaining
  ≈ 70 min more. Resume next session — read partial JSONs at
  `.claude/worktrees/agent-a5b15c06/analysis/fp8_gemm/mi350x/bench_devD_S1*_S2*.json`.
- **Dev E (BF16 CRR `__launch_bounds__(_,1)` KI=296 only, GPU 4,
  complete)** — NO LAND. Restructured `gemm_kernel` into
  `gemm_kernel_body` + a thin `__global__` wrapper, added explicit
  `gemm_kernel<CRR,296>` specialization with
  `amdgpu_waves_per_eu(1, 2)`. Build-log resource: VGPRs 245→170,
  AGPRs 0→192, occupancy 2→1, **SGPR-Spill stays 26**. Plus
  K=18944 CRR shape now hits `hipErrorLaunchFailure` (RCR/RRR fine).
  Definitive conclusion: the 26-SGPR-spill is **scheduling pressure
  from the unroll-2 main-loop address arithmetic, not a VGPR-budget
  contention** — relaxing occupancy doesn't address the root cause.
  Combined with P9 Dev 1's finding (unroll-1 drives spill to 0 but
  costs +0.88pp wall-clock on GPU 2), the BF16 CRR KI=296 spill is
  now confirmed as **an LLVM-scheduler artifact whose elimination via
  either unroll-1 or launch_bounds costs more than the spill itself**.
  Worktree `agent-a66b0af1`. Build log
  `bench_k18944_devE_lb1.log` / `build_p13_devE_lb1.log`.

## Current Status (2026-04-18, post-P13 — no code change landed; key correction filed)

P11 + P12 + P13 ran 10 Dev agents + 1 Reviewer + 2 research agents across
GPUs 0/3/4/5/6/7. **Nothing has landed since P8.** P13 outcomes:

- **P13 Dev A (FP8 RRR drain restructure, GPU 0)**: shipped clean knob
  infrastructure (`RRR_DRAIN1/2/3/4_LGKM` macros, defaults to identity)
  in worktree `agent-a784d2cb`. Found the 4× `lgkmcnt(0)` barriers are
  **load-bearing for correctness** on at least RRR(4096,2048,4096):
  D1=2 hangs that shape's SNR test. Agent silent before completing the
  4-D sweep. NO LAND.
- **P13 Dev B (BF16 CRR `__launch_bounds__(_,1)`, GPU 4)**: silent
  timeout with zero tracked-file changes. NO LAND.
- **P13 Dev C (FP8 RCR DTL feasibility, research-only)**: **disproved
  P12 Dev G's headline DTL hypothesis** — TK FP8 RCR already uses
  gfx950 wide-DTL (`buffer_load_dwordx4 ... lds`, 16B/lane) for 100% of
  hot-path loads. 658 DTL instructions, 0 non-DTL global loads on the
  GEMM hot path. Both A and B operands are DTL, composed cleanly with
  the XOR-based ST_v2a swizzle. The 3% TK_RCR vs BL_RCR gap therefore
  cannot be closed by adding DTL — the lever doesn't exist. The actual
  remaining gap lives in consumer-side `ds_read` interleave / outer-loop
  pipelining / tile-shape combinations (see updated CEILING ANALYSIS
  section below).

The numbers below remain unchanged from post-P8.

## CEILING ANALYSIS (2026-04-18, P12 Dev G — definitive)

Single-launch rocprofv3 on GPU 6, 8192³ FP8 e4m3 → BF16:

| Kernel  | TFLOPs | Wall (cycles) | LDS/MFMA | VALU/MFMA | WAIT/LDS |
|---------|--------|---------------|----------|-----------|----------|
| TK_RCR  | 3111   | 5,252,876     | 0.75     | 1.50      | 1.36     |
| TK_RRR  | 3002   | 5,653,394     | 1.00     | 2.16      | 1.22     |
| TK_CRR  | 2908   | 5,920,020     | 1.50     | 2.60      | 0.88     |
| BL_RCR  | 3203   | 5,135,587     | 0.50     | 1.26      | 0.73     |  ← Custom_ TN hand-written
| BL_RRR  | 2018   | 9,880,062     | 1.21     | 5.50      | 1.87     |  ← Tensile autogen
| BL_CRR  | 1505   | 12,183,826    | 4.51     | 13.55     | 1.80     |  ← Tensile autogen

All 6 kernels issue **identical** SQ_INSTS_VALU_MFMA_F8 = 16,777,216 and
SQ_VALU_MFMA_BUSY_CYCLES = 536,870,912. Same MFMA flavor
(`v_mfma_f32_16x16x128_f8f6f4`). No kernel uses 32×32×16 or scaled-FP8.

**hipBLASLt's own ratios** (its NN/NT Tensile kernels vs its TN custom):
- BL_RRR / BL_RCR = 0.66
- BL_CRR / BL_RCR = 0.50

We are at 0.95/0.92, i.e. **substantially better than hipBLASLt's own
layout-uniformity**. TK_RRR is 1.49× BL_RRR, TK_CRR is 1.93× BL_CRR.

**The 1.000 RRR/RCR target presupposes** a hidden software lever exists.
The data shows there is none for NN/NT layouts:

1. The only kernel that beats us is BL_RCR's hand-written `Custom_` TN
   kernel (3% gap). It uses Direct-To-LDS (DTLA1+DTLB1) — bypasses the
   VGPR roundtrip. DTL only works for the TN coalesced-load pattern;
   Tensile's NN/NT solutions don't enable DTL because their global-load
   strides don't line up.
2. BL_RCR additionally uses MT256×256×128 + 256-VGPR allocation (1
   wave/SIMD) + a hand-scheduled MFMA pipeline ("CMS") + WGM6 mapping.
   Reproducing this in TK is ≥ 3 P-sessions of work, narrows occupancy
   to 1, and the gain only appears for RCR.
3. **NO preshuffle applies symmetrically to hipBLASLt** — they don't get
   that lever either. The constraint is real and external.

**Recommendation (Dev G):** treat the RRR/RCR=1.000, CRR/RCR=0.950
targets as unachievable under the no-preshuffle hard constraint. Reframe
success as:
- TK_RRR / BL_RRR ≥ 1.40 → **achieved 1.49** ✅
- TK_CRR / BL_CRR ≥ 1.80 → **achieved 1.93** ✅
- TK_RCR / BL_RCR ≥ 0.97 → **achieved 0.97** ✅

**P13 Dev C correction (2026-04-18):** Dev G's claim that "TK lacks DTL
and that explains the 3% TK_RCR vs BL_RCR gap" is empirically wrong.
Disassembly of the prebuilt `.o` shows TK already issues 658
`buffer_load_dwordx4 ... lds` (gfx950 wide-DTL, 16B/lane) instructions
for 100% of hot-path loads, with 0 non-DTL global loads on the GEMM hot
path. Both A and B operands are DTL. ST_v2a XOR swizzle composes
cleanly via the swizzled-global-offset trick (no second LDS pass).

The actual remaining levers for TK_RCR catching the last 3% to BL_RCR
must therefore live in:
1. Consumer-side `ds_read_b128` / `ds_read_b64_tr_b8` interleave and
   issue rate (currently 1336+1360 reads / 2688 MFMAs).
2. Outer-loop pipelining depth + prefetch distance (already
   extensively swept in P7/P8).
3. Per-tile-shape choice (BL_RCR uses MT256×256×128; TK uses smaller).
4. Hand-scheduled `s_setprio` + `sched_barrier(0)` discipline (already
   in extensive use; P9 BF16 attempts to extend further didn't land).
5. NOT bank conflicts (Dev E P12 measured `SQ_LDS_BANK_CONFLICT = 0`).
6. NOT MFMA shape (all kernels use the same 16×16×128 f8f6f4).

If a 1.000 RRR/RCR is required, the only path is one of:
- Relax no-preshuffle for B in RRR (reorder to TN-equivalent stride
  offline) — explicitly forbidden by current rules.
- Build a dedicated CMS+MT256² RRR/CRR variant — multi-session effort
  with downside risk on RCR.
- Accept the structural ceiling (current state).

## NEW PRIMARY TARGETS (2026-04-18, P11+) — STATUS: CEILING REACHED

The user's targets (set 2026-04-18) are listed below for reference; P12
work proves they cannot be met within the no-preshuffle constraint.

| Layout | Current (RCR-relative) | Target | Gap | P12 verdict |
|---|---|---|---|---|
| FP8 RRR | 0.950 × RCR | **1.000 × RCR** | +5.0pp | INFEASIBLE no-preshuffle |
| FP8 CRR | 0.922 × RCR | **0.950 × RCR** | +2.8pp | INFEASIBLE no-preshuffle |

Reference absolute geo-means (GPU 3, `bench_no_jit_final.json`):
RCR 2939 TFLOPs / RRR 2793 / CRR 2711.

**Hard constraint additions (still in force):**
- **NO preshuffle / no offline weight permutation.**
- The vs-hipBLASLt geo-means (RRR 1.53x, CRR 1.97x) are non-regression
  gates. P12 confirms: TK still 1.49×/1.93× vs BL on RRR/CRR at 8192³.

**FP8 absolute throughput** (geo-mean tk_tflops over the 48-shape set,
`bench_no_jit_final.json`): RCR 2939 / RRR 2793 / CRR 2711 →
**RRR/RCR = 0.950, CRR/RCR = 0.922**. RRR is exactly on the 95% line of
RCR (23/48 shapes ≥ 0.95); CRR is ~3pp short of 95%. The vs-hipBLASLt
ratios (0.996 / 1.530 / 1.967) overstate RRR/CRR strength because
hipBLASLt itself is much slower on RRR/CRR than on RCR.



### FP8 — ✅ All targets met

Measured on GPU3, `bench_vs_hipblaslt.py --mode full`:

| Layout | Geo-mean vs hipBLASLt | Wins | Status |
|---|---|---|---|
| RCR | 0.996x | 21/56 | ⚠ within noise of 1.00x |
| RRR | 1.530x | 56/56 | ✅ ≥ 1.40x |
| CRR | 1.967x | 56/56 | ✅ ≥ 1.80x |

Last change: `RCR_TWO_TILE_MID_VMCNT 4 → 6` (P8). Original P7 commit message
claimed this was landed, but the file shipped at MID=4; P8 corrects it.

### BF16 — 🚧 Closer but still below 1.0x

Measured on GPU2 with the new per-shape NUM_XCDS autotune:

| Layout | Geo-mean vs torch.mm | Wins | Δ vs pre-P8 |
|---|---|---|---|
| RCR | 0.984x | 6/48 | +1.0pp |
| RRR | 0.980x | 9/48 | +1.6pp |
| CRR | 0.953x | 3/48 | +1.7pp |

torch.mm = hipBLASLt under the hood. Targets are still ≥ 1.00x for each
layout; no layout regressed.

## Open Items — High Priority

### BF16 CRR (biggest residual gap, -4.7pp)
- [ ] **Reduce SGPR spill on CRR KI=128 (26)**. P9 confirmed KI=296 spill
      reduction (26→0 via `#pragma unroll 1`) does NOT translate to perf
      wins under cross-GPU validation — the K=18944 CRR shape *regressed*
      0.88pp on GPU 2 even though spills genuinely dropped to 0 and Dev 1's
      GPU 4 measurement showed +1.2pp. Conclusion: bare unroll reduction
      trades barrier hiding for spill reduction and the trade is net
      negative on CRR. KI=128 still untried; would need a deeper
      restructure (split-lambda, SRD-offset hoist into LDS, etc.) rather
      than just unroll-1. Cosmetic `readfirstlane` hoist of `row*2/col*2`
      is also explored (P9): RCR/RRR neutral, kept out of tree.
- [ ] **GPU clock pinning is silently broken** on this host —
      `rocm-smi --setperflevel high` returns success but perf level
      stays "auto" both with and without sudo. P8 + P9 confirmed.
      Root cause unknown; might be a kernel module issue. Without it,
      ±2pp of DVFS noise dominates any single-knob effect, so any future
      sweep MUST average across ≥ 5 trials AND validate on a 2nd GPU.
- [ ] Per-shape WAITCNT autotune (RCR/RRR `vmcnt`/`lgkmcnt` profiles via
      `WAITCNT_PROFILE` template arg) was prototyped P9 (worktree
      `team-bf16-rcrrrr-mn`): +0.25pp / +0.17pp consistent across 3 runs
      but inside the calibrated DVFS noise band (per-shape stdev ≈ 0.66pp,
      CRR-pinned-code calibration). Bloats .so by ~7×. Diff preserved in
      worktree, NOT landed.

### BF16 RCR / RRR (-1.6 to -2.0pp)
- [ ] Try M↔N kernel swap for shapes where N > M (explicit grid swap, not
      group-by-N swizzle). Per-shape NUM_XCDS already absorbs most of the
      large-N gain; the residual is on small-K + large-N.
- [ ] Tune `s_waitcnt lgkmcnt(8)` / `vmcnt(6)` positions for small-K
      large-N shapes (these were hand-tuned for 8192³).
- [ ] Consider runtime 4-wave path for large-grid shapes (analogous to FP8).

### FP8 RRR / CRR — closed under no-preshuffle (P12 ceiling proven)

Per Dev G's P12 ceiling analysis (above), the 1.000 and 0.950 targets
are not reachable while no-preshuffle is in force. Items kept here only
for archival; do NOT redispatch unless the constraint changes:

- [P12-CLOSED] LDS bank-conflict profiling (RRR/CRR). Dev E (P12) measured
  `SQ_LDS_BANK_CONFLICT = 0` on both. Not the lever.
- [P12-CLOSED] `ds_read_b64` 2-stage schedule + register-level
  `v_perm_b32`. Dev A (P11) showed RRR's 4× `s_waitcnt lgkmcnt(0)`
  hard-drain barriers (lines 1607-1670) mask all per-instruction
  wait-tuning until the structural barriers are removed. Dev D (P11)
  attempted the restructure and could not converge.
- [P12-CLOSED] CRR LDS A double-buffer. Dev C (P11) confirmed
  `__shared__ ST_crr_a As[2][2]` is already double-buffered.
- [P12-CLOSED] CRR `lgkmcnt(0)→lgkmcnt(K)` per-operand restructure.
  Dev F (P12) shipped the knob infrastructure (`CRR_STEADY1_LGKM`,
  `CRR_STEADY2_LGKM`) but the one override they got to bench
  (S1=2,S2=4) gave +0.27pp RRR / +0.43pp CRR — within DVFS noise band.
  Diff lives in `agent-a5b15c06` worktree, defaults to identity, NOT
  landed.
- [P12-CLOSED] Alternate (M_TILE, N_TILE, K_TILE) for RRR-only.
  hipBLASLt's RRR uses MT256×208×128 (asymmetric) — Dev G concluded the
  large tile only helps because it pairs with DTLA1+DTLB1 (Direct-To-LDS),
  which TK doesn't have an implementation for.

### FP8 RCR — only remaining headroom (~3% to BL_RCR custom kernel; ~8% on weak shapes)
- [P14] **Stream-K persistent grid** (Dev F P14 lever B) — hipBLASLt
      uses `SK3` (StreamK) on all 6 weak FP8 RCR shapes. Persistent-grid
      kernel that re-issues tiles inside one workgroup eliminates tail
      effect on small-K large-N shapes. Multi-session implementation;
      no clean TK precedent.
- [P14] **Wider / fewer LDS reads** (Dev F P14 lever C) — TK does
      1.47-1.50× more `ds_read*` per MFMA than BL on every weak shape.
      Investigate `ds_read_b128_tr_b16` if compatible with the FP8 ST_v2a
      swizzle layout (BL ds_read mix unknown — would need disassembly
      sample of one Cijk_ kernel to compare). Net effect bounded by the
      0.75→0.50 LDS/MFMA gap (one of two factors in the 8% weak-shape
      gap; the other is StreamK).
- [P13-CLOSED] ~~Direct-To-LDS (DTLA1+DTLB1) implementation for RCR (TN).~~
      P13 Dev C disassembly grep proved TK ALREADY uses gfx950 wide-DTL
      for 100% of hot-path loads. P14 Decider re-verified after Dev F
      flagged inconsistency: 0× `ds_write*`, 658× `buffer_load_dwordx4`,
      2696× `ds_read*` / 2688× MFMAs in the prebuilt .o. Both A and B
      operands DTL. Lever does not exist; do not redispatch.
- [ ] Per-tile-shape exploration: BL_RCR uses MT256×256×128 with
      256-VGPR/1-wave. TK uses smaller MT with 2-wave. A dedicated
      large-MT RCR variant could close the LDS/MFMA gap (0.75→0.50)
      but requires occupancy trade and per-shape autotune routing.
      ≥ 2 P-sessions estimated.
- [ ] Consumer-side `ds_read_b128`/`ds_read_b64_tr_b8` interleave +
      issue-rate sweep. Current 1336+1360 reads / 2688 MFMAs. Hand
      schedule with `__builtin_amdgcn_sched_barrier(0)` + `s_setprio()`
      already in extensive use (kernel_fp8_layouts.cpp:14-91 macros);
      possible micro-gains in per-shape SCHED_BARRIER placement.
- [ ] Hand-scheduled MFMA pipeline (`__builtin_amdgcn_sched_barrier` +
      `s_setprio` discipline) for RCR. Reduces SQ_WAIT_INST_LDS from
      1.36×LDS to ~0.73×LDS. Risk is high — BF16 P9 attempts didn't land.

### FP8 RCR (within noise of 1.00x; 12 weak shapes still 0.90-0.93x)
- [x] ~~Per-shape NUM_XCDS for FP8~~ — P9 ran a strict per-shape re-bench
      (warmup=30, iters=100, trials=5) on every weak shape. Result:
      **xcd=8 wins on every one of the 12 weak shapes** by 0.1-2.5%. The
      previous attempt's "wins" for xcd∈{4,16} were per-shape thermal
      noise. Mechanism is sound but offers no headroom — closed.
- [ ] Small K + big N remain weak: (M, 28672, 4096), (M, 37888, 3584).
      hipBLASLt likely uses Split-K. Explore deterministic on-chip
      Split-K (no atomics) — no clean implementation idea yet; the
      grid is already sparse enough that "splitting K" inside one block
      doesn't help. Open problem.
- [ ] Revisit KI template specialization with `unroll 1` instead of
      `unroll 2`. **Caveat from P9**: BF16 KI=296 unroll-1 dropped
      spills 26→0 but cost 0.88pp on the target CRR shape. Suggests
      barrier-hiding from unroll-2 outweighs the spill cost on at
      least some kernels. If revisited for FP8, validate on at least
      2 GPUs before committing.

## Open Items — Correctness

- [ ] **BF16 2048³ CRR non-determinism** — at M=N=K=2048, CRR produces
      non-deterministic output (~15-21 bf16 ULP max diff). Root cause
      unknown. All benchmarked LLM shapes are ≥ 4096³ so the 48-shape
      benchmark is unaffected, but this should be fixed before production.

## Open Items — Medium Priority

- [ ] Make `autotune.py` also autotune the 4-wave vs 8-wave path choice for
      FP8 RCR.
- [ ] Benchmark against TRITON backend in Primus-Turbo, not only hipBLASLt.
- [ ] Add a CI script that runs `test_fp8_snr.py` + `quick_snr.py` + a
      5-shape perf sanity check on both directories.

## Closed / Completed

- 2026-04-18 P14 — Fourth agent-team session (3 Devs, all opus, GPUs 0/4/6).
  Dev D in flight at session end (CRR_STEADY 2D sweep, 2/16 configs both
  within DVFS noise); Dev E NO LAND with definitive close on BF16 CRR
  KI=296 launch_bounds; Dev F shipped per-shape PMC research memo for the
  6 weakest TK_RCR shapes. **Decider disassembly recheck reconfirmed
  P13 Dev C: TK uses DTL on both A and B operands (0× ds_write,
  658× buffer_load_dwordx4 in the prebuilt .o).** Ranked P15 levers
  reduce to: (B) StreamK persistent grid, (C) wider/fewer LDS reads
  (e.g. `ds_read_b128_tr_b16` if compatible with FP8 swizzle).
- 2026-04-18 P13 — Third back-to-back agent-team session (3 Devs, all opus,
  GPUs 0/4/6). **Nothing landed.** Headline result: P13 Dev C disproved
  P12 Dev G's DTL hypothesis — TK already uses gfx950 wide-DTL.
  - **Dev A (FP8 RRR `lgkmcnt(0)` drain restructure, GPU 0)** — added
    knob infrastructure `RRR_DRAIN1/2/3/4_LGKM` macros (defaults to
    identity = current `lgkmcnt(0)` behavior) at lines 41-57 of
    `kernel_fp8_layouts.cpp`, replaced 4 raw `asm volatile("s_waitcnt
    lgkmcnt(0)")` calls (lines 1633/1651/1668/1678) with
    `TK_WAIT_LGKM(RRR_DRAINn_LGKM)`. Smoke test with D1=2 caused HANG
    on RRR(4096,2048,4096) SNR test — confirms barrier is **load-bearing
    for correctness**, not just a scheduling hint, on at least one shape.
    Agent silent before completing 4-D sweep. Diff in worktree
    `agent-a784d2cb`. NO LAND.
  - **Dev B (BF16 CRR `__launch_bounds__(_,1)` for KI=128/296, GPU 4)**
    — silent timeout. Zero tracked-file changes. NO LAND.
  - **Dev C (FP8 RCR DTL feasibility scoping, no kernel edits)** —
    disassembled prebuilt `kernel_fp8_layouts-hip-amdgcn-amd-amdhsa-gfx950.o`
    via `llvm-objdump --mcpu=gfx950`. Counted: 658 `buffer_load_dwordx4
    ... lds` (gfx950 wide-DTL, 16B/lane), 0 non-DTL global loads on the
    GEMM hot path, 1336 `ds_read_b128`, 1360 `ds_read_b64_tr_b8`, 2688
    `v_mfma_f32_16x16x128_f8f6f4`. Verified TK source path:
    `include/ops/warp/memory/tile/global_to_shared.cuh:215-222` calls
    `llvm_amdgcn_raw_buffer_load_lds()` from CK-Tile. Both A and B
    operands DTL. ST_v2a XOR swizzle composes via swizzled global offset
    (`prefill_swizzled_offsets` at `:147-152`). **The DTL "missing
    lever" hypothesis from P12 Dev G is empirically false.** The actual
    3% TK_RCR vs BL_RCR gap lives in consumer-side ds_read interleave
    or tile-shape selection, not the global→LDS path. Memo committed to
    project memory (`project_fp8_ceiling.md` updated).
  - **Lessons additive to P12:**
    - Before scoping a "missing instruction X" lever, **grep the
      compiled disassembly to confirm we don't already use it**. P12
      Dev G missed that the 658 `buffer_load_*x4 ... lds` instructions
      ARE the DTL path. P13 Dev C caught it in a 60-min research budget.
    - **Knob-infrastructure-only diffs (no override values benched) are
      not landable.** P12 Dev F and P13 Dev A both shipped clean macro
      infrastructure but neither found an override value that beat
      baseline within the noise band. The infrastructure has zero
      shipping value if it preserves the current behavior at default.
    - The pattern from P11/P12 holds at P13: agents that get into
      multi-hour benchmarking loops with shared kernel sources tend to
      go silent. Cap concurrent same-source Devs at 1 (not 2 as P12
      lesson suggested) when the work involves hipcc rebuild + bench
      iterations.
- 2026-04-18 P11 + P12 — Two back-to-back agent-team sessions (7 Devs +
  1 Reviewer + 1 ceiling-research agent across GPUs 0/3/4/5/6/7).
  **Nothing landed.** Net outcome: **the new RRR/RCR ≥ 1.000 and
  CRR/RCR ≥ 0.950 targets are architecturally infeasible under the
  no-preshuffle constraint** (Dev G ceiling analysis above).
  - **Dev A (P11, RRR waitcnt sweep)** — identified the structural
    blocker: `kernel_fp8_layouts.cpp:1607-1670` has 4× `s_waitcnt
    lgkmcnt(0)` hard-drain barriers per RRR steady-state iteration.
    These mask every per-instruction wait knob. NO LAND.
  - **Dev B (P11, RRR alt-tile)** — silent for 2h+, produced 21 bench
    JSONs but no kernel diff. NO LAND.
  - **Dev C (P11, CRR LDS double-buffer)** — confirmed `As[2][2]` is
    already double-buffered. The 8% gap is structural column-stride
    cost, not a missing buffer. NO LAND.
  - **Dev D (P12, RRR `lgkmcnt(K)` restructure)** — hung after 1h, no
    kernel diff. NO LAND.
  - **Dev E (P12, CRR LDS layout / bank-conflict)** — direct measurement
    `SQ_LDS_BANK_CONFLICT = 0` on both CRR and RCR. Eliminates bank
    conflicts as a lever. NO LAND.
  - **Dev F (P12, CRR `lgkmcnt` knob infrastructure)** — added
    `CRR_STEADY1_LGKM`/`CRR_STEADY2_LGKM` macros at lines 1968/1983,
    defaults to identity. Single override S1=2,S2=4 benched at
    +0.27pp RRR / +0.43pp CRR — within DVFS noise. Agent died before
    finishing the sweep. Diff in `agent-a5b15c06` worktree, NOT landed.
  - **Dev G (P12, hipBLASLt ceiling research, GPU 6, no kernel edits)** —
    rocprofv3 single-launch profile of all 6 kernels (TK ×3 layouts,
    BL ×3 layouts) at 8192³. Findings:
    - Identical SQ_INSTS_VALU_MFMA_F8 = 16,777,216 across all 6.
    - hipBLASLt's own RRR/RCR = 0.66, CRR/RCR = 0.50 (we are at 0.95/0.92,
      i.e. *better* than hipBLASLt's own layout-uniformity).
    - TK_RRR / BL_RRR = 1.49; TK_CRR / BL_CRR = 1.93 at 8192³.
    - The only kernel that beats us is BL_RCR's hand-written `Custom_`
      TN kernel (3% gap). It uses Direct-To-LDS + MT256² + 256-VGPR +
      hand-scheduled CMS + WGM6 — none of which generalize to NN/NT.
    - Verdict: targets are unachievable; treat current state as ceiling.
  - **Lessons additive to P10:**
    - "Vs our own RCR" is not a meaningful target when RCR has access
      to a hardware-specific code path (Direct-To-LDS for TN-coalesced
      loads) that other layouts intrinsically can't use. Future
      "X-layout / RCR" targets need a feasibility check first.
    - rocprofv3 single-launch comparison (TK vs BL same shape, same GPU)
      is a far stronger ceiling-bounding tool than perf benchmarking
      alone. Use it before launching restructure attempts.
    - Multi-Dev parallelism on the same kernel source eventually
      collides: when 4+ Devs share `kernel_fp8_layouts.cpp` with no
      coordinator, several silently hang or never produce a diff.
      P12 should have used at most 2 concurrent Devs on shared files.
    - Negative results have shipping value: the P12 ceiling memo is
      itself the headline result of the session — it stops downstream
      teams from re-attempting the same dead ends.
- 2026-04-18 P10 — Second agent-team session (3 Devs + 1 Reviewer, all opus,
  GPU 0/4/5/7). **Nothing landed.**
  - **Dev A (BF16 CRR SGPR-spill restructure, GPU 4)** — three strategies:
    KI=128 `#pragma unroll 1` (-0.31pp geo-mean), `__builtin_amdgcn_readfirstlane`
    on `row*2/col*2/etc.` (zero spill effect — compiler already proved
    uniformity), manual 2-iter fusion + `sched_barrier(0)` (spills 26→0
    on every CRR KI but VGPRs 245→252, occupancy floor at 2 became
    fragile, **CRR -0.90pp regression**, e.g. (8192,8192,8192) -3.11pp).
    All reverted.
  - **Dev B (BF16 RCR/RRR M↔N swap + small-K WAITCNT, GPU 5)** — Strategy A
    (host grid swap) ruled out: clean version needs a transposed C store
    that the existing `gl<>` API doesn't accept; Python-level transpose
    copy costs 5–25% on target shapes (regression). Strategy B small-K
    WAITCNT (`lgkmcnt(8)→4`, `vmcnt(6)→2` for `KI<96` constexpr branch):
    in DVFS noise, slight regression on (4096,28672,4096) RRR. All reverted.
  - **Dev C (FP8 RCR weak shapes small-K big-N, GPU 0)** — Strategy C
    per-shape `RCR_TWO_TILE_MIN_KI` runtime knob: swept mk∈{0,32,64,128,∞}
    × gm∈{1,2,4,8,16,32} on 13 weak shapes, **mk=0/32 (current default)
    wins everywhere**, mk≥64 uniformly 1-2% slower. Strategy B KI=28/32
    template + `unroll(1)`: clean build, 0 spills, but A/B test gave
    geo-mean +0.03pp (pure noise). Both reverted; only an 8-line "do not
    repeat" comment recommended.
  - **Lessons additive to P9:**
    - For BF16 CRR, the unroll-2 + 26-spill point is genuinely
      Pareto-optimal for the current `main_loop_iter` shape. Reducing
      spills via either `unroll 1` OR manual fusion costs more than it
      saves. **Future CRR work needs a structural restructure** (e.g.
      maintain running SOFF SGPR per LDS slot; `__launch_bounds__(_,1)`
      only for KI=128/296 to trade occupancy for register pressure),
      not another local tweak.
    - The 12 weak FP8 RCR shapes (small-K big-N) appear to be at a
      structural ceiling for the 8-wave 2-tile schedule. A real fix
      requires either a 4-tile in-block split (Strategy A — 2+ days
      of work, deferred) or relaxing the no-atomics rule for true
      Split-K. No amount of WAITCNT/MIN_KI tuning will close it.
    - Pattern reinforced from P9: any sub-1pp signal is noise on this
      host until clock pinning works.
- 2026-04-17 P9 — Agent team session (3 Devs + 1 Reviewer, all opus).
  Three optimization directions explored, **nothing landed**:
  - BF16 CRR KI=296 `#pragma unroll 1` (Dev 1, GPU 4): SGPR spills
    26→0 confirmed in build log, +1.2pp on (8192,3584,18944) on GPU 4
    → -0.88pp on the same shape on GPU 2 (Reviewer). Net geo-mean
    within ±0.3pp on every layout. **Rejected** — barrier-hiding from
    unroll-2 beats spill reduction here, and DVFS noise (clock pinning
    silently broken) hid the regression on Dev's GPU.
  - BF16 RCR/RRR per-shape WAITCNT autotune (Dev 2, GPU 5): consistent
    +0.25pp/+0.17pp across 3 runs but inside calibrated noise band
    (stdev 0.66pp/shape). **Abandoned** — not worth 7× .so bloat.
  - FP8 RCR per-shape NUM_XCDS retry (Dev 3, GPU 0): with proper
    averaging xcd=8 wins on every weak shape; previous "wins" were
    thermal noise. **Closed** — moved this item to the closed list.
  Lessons:
  - `rocm-smi --setperflevel high` silently no-ops on this host. Any
    sweep that doesn't validate on a 2nd GPU is suspect.
  - SGPR spill count is a means, not an end. Confirm the wall-clock
    drop, not just the spill drop.
- 2026-04-17 P8 — BF16 per-shape NUM_XCDS autotune landed: RCR +1.0pp,
  RRR +1.6pp, CRR +1.7pp. FP8 MID_VMCNT 4→6 corrected (P7 commit message
  claimed this but file shipped at 4).
- 2026-04-17 — BF16 CRR-only knob exploration: CRR_MAIN_VMCNT,
  CRR_MAIN_LGKMCNT, CRR_UNROLL={1,4,8}, CRR_NUM_XCDS={4,16}, CRR_CHUNK
  all within noise; root cause is SGPR spill on KI=128/296.
- [x] Removed all JIT per-shape compilation (`jit_gemm.py`, `bench_jit*.py`,
      `kernel_jit_*.cpp`, `*_exact_*_fastpath.inc`, `.jit_cache/`,
      `.jit_bf16_cache/`).
- [x] Removed dead experimental kernels (`kernel_1024/2048/4096/8192/16384.cpp`,
      `kernel_bf16_128/256x128/4wave.cpp`, `kernel_crr.cpp`,
      `kernel_layouts.cpp`).
- [x] FP8 RCR geo-mean ≥ 1.00x achieved (1.005x → drifted to 0.996 noise band).
- [x] BF16 migrated to single-source `kernel_bf16_dynamic.cpp` with runtime
      KI_HINT template dispatch.
- [x] Both directories use runtime group_m autotune; BF16 also autotunes NUM_XCDS.
- [x] Updated skill docs: `bf16-gemm-optimization`, `fp8-rcr-autotune-optimization`,
      `fp8-strict-layout-tuning`.

## How To Run

```bash
# FP8
cd analysis/fp8_gemm/mi350x
THUNDERKITTENS_ROOT=/shared_nfs/kyle/HipKittens2 ROCM_PATH=/opt/rocm make clean
THUNDERKITTENS_ROOT=/shared_nfs/kyle/HipKittens2 ROCM_PATH=/opt/rocm make -j4
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. python3 bench_vs_hipblaslt.py --mode full

# BF16
cd analysis/bf16_gemm/mi350x
THUNDERKITTENS_ROOT=/shared_nfs/kyle/HipKittens2 ROCM_PATH=/opt/rocm make clean
THUNDERKITTENS_ROOT=/shared_nfs/kyle/HipKittens2 ROCM_PATH=/opt/rocm make -j4
HIP_VISIBLE_DEVICES=1 PYTHONPATH=. python3 bench_bf16_vs_torch.py
```
