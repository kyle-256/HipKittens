# Agent Prompt — FP8 / BF16 GEMM on MI350X

Use this document when spawning subagents to continue the FP8/BF16 work.
Read the related skills under `.cursor/skills/` first:
- `cdna4-gemm-kernel-design/SKILL.md` (architecture fundamentals)
- `fp8-strict-layout-tuning/SKILL.md` (FP8 rules + known findings)
- `fp8-rcr-autotune-optimization/SKILL.md` (FP8 current state + next steps)
- `bf16-gemm-optimization/SKILL.md` (BF16 current state + next steps)

## Agent Team Structure

Use this team structure for non-trivial optimization tasks.

### Roles

1. **Decider (主 agent)** — plans work, assigns subagents, reviews results,
   merges changes, writes commits, updates TODO and skill docs. Does not
   itself do heavy kernel edits.

2. **Dev (2-4 subagents)** — each owns one optimization direction (a
   specific layout × weak-shape family, or a specific idea like M↔N swap).
   Each runs in isolation with its own GPU.

3. **Reviewer / QA (1 subagent)** — after all Dev changes merge, runs the
   full 48-shape benchmark, the SNR/determinism gate, and confirms
   non-regression of the strong layouts.

### GPU Assignment

Hardware has 8 MI350X GPUs (GPU 0-7). Assign one GPU per agent:
- FP8 Dev agents: GPU 0, 2, 3
- BF16 Dev agents: GPU 1, 4, 5
- Reviewer: any idle GPU

Never have two agents share a GPU — hipcc invocations overwrite the shared
`.so` binary and invalidate each other's measurements.

## Prompt Template for Dev Subagents

Every Dev subagent prompt MUST include:

### 1. Hard constraints

- **NO JIT** — single `.so` per target. No per-shape `-DM_DIM` compile flags.
- **NO git commits** — leave changes in working tree; Decider commits.
- **GPU isolation** — use only the assigned `HIP_VISIBLE_DEVICES` value.
- **SNR ≥ 48 dB**, **bit-exact determinism** across 3+ runs.
- **Non-regression gates** — cite the current numbers and the minimum
  acceptable numbers for the layouts not being changed.

### 2. Context

- Repo path, branch name.
- Working directory.
- Current measured baseline (print actual TFLOPS, not just ratios).
- List of the specific weak shapes being targeted.
- Which files are the single source, which files are OK to delete.

### 3. Concrete strategies, ranked

Prefer 3-5 ranked ideas over one open-ended task. Each idea should include:
- Expected outcome
- Files to change
- A smoke test + full test that confirms success

### 4. Deliverables

- Exact output format: benchmark JSON path, final summary format.
- "Do NOT commit" in bold.
- Print summary fields: geo_means, top 5 wins, remaining weak shapes,
  VGPRs/occupancy of any new kernel variants.

## Decider Checklist (before commit)

After subagents return:

1. `git status` — list modified + deleted + untracked.
2. Delete transient files: `bench_*.log`, `build_*.log`, `probe_*.py`,
   any `.so` binary, any `bench_v*_baseline.json` (keep only one
   "final" json per directory).
3. Run the Reviewer agent: re-benchmark the modified directory, confirm
   SNR + determinism + non-regression. **The Reviewer MUST use a
   different `HIP_VISIBLE_DEVICES` than the Dev** who claimed the win,
   to rule out per-GPU DVFS thermal artifacts. P9 demonstrated that
   the same code can show +1.2pp on one GPU and -0.88pp on another
   when clocks are not actually pinned (and `rocm-smi --setperflevel high`
   is silently broken on this host).
4. Ensure the strong layouts didn't regress (FP8 RRR ≥ 1.4x, CRR ≥ 1.8x;
   BF16 keep whatever baseline was).
5. Update `TODO.md` with new state and remaining items.
6. Update relevant skill under `.cursor/skills/` — **no more than what's
   needed to reflect reality**. Delete stale guidance.
7. Stage changes explicitly, use a HEREDOC commit message.
8. Never add `--no-verify` or `-i`.

## Example Agent Task (copy-paste starting point)

```
You are <role> on HipKittens <FP8|BF16> GEMM for AMD MI350X.

Environment:
- Repo: /workspace/code/Hipkittens_per_tensor on branch <branch>
- Working dir: analysis/<fp8_gemm|bf16_gemm>/mi350x
- GPU: use HIP_VISIBLE_DEVICES=<N> only

Baseline (measured <date>):
- RCR geo_mean=X.XXXx ; RRR=X.XXXx ; CRR=X.XXXx

Your target: <specific layout / shape family / number>

STRICT CONSTRAINTS:
- NO JIT per-shape compilation; single .so only.
- NO git commits.
- SNR ≥ 48 dB; bit-exact determinism.
- Don't regress RRR (must stay ≥ X.XXx) or CRR (≥ X.XXx).

Strategies in priority order:
  A. <idea> — <expected +/- %>
  B. <idea>
  C. <idea>

Deliverables:
1. Modified files list
2. Final benchmark JSON at <path>
3. Print FINAL SUMMARY: geo_means, wins, top 5 shape improvements,
   remaining weak shapes, VGPRs.
4. DO NOT commit.

Tips:
- <hint about primus_turbo import>
- <hint about cleaning stale .so>
- <hint about watching VGPR remarks>
```

## Session Logging

Every Decider session should append a one-line dated entry to `TODO.md`
"Closed / Completed" section when a target is hit, and update the
"Current Status" table numbers.

Every skill update should reflect only what was verified in the current
session, not aspirational targets.

## Current Primary Targets (P11+, set 2026-04-18) — CEILING REACHED P12, refined P13

The user's targets remain on record but **P12 Dev G proved them
architecturally infeasible under the no-preshuffle constraint** on gfx950
(P13 Dev C corrected the *mechanism* — DTL is already in use):

- ~~FP8 RRR / RCR ≥ 1.000~~ — at 0.95; hipBLASLt's own RRR/RCR is 0.66
- ~~FP8 CRR / RCR ≥ 0.950~~ — at 0.92; hipBLASLt's own CRR/RCR is 0.50

We are currently *better than* hipBLASLt's own layout-uniformity:
- TK_RRR / BL_RRR = **1.49**
- TK_CRR / BL_CRR = **1.93**
- TK_RCR / BL_RCR = **0.97** (3% gap to BL's hand-written `Custom_` TN
  kernel; only single remaining headroom)

**Hard prohibition: NO PRESHUFFLE / NO offline weight permutation.** Still
in force. hipBLASLt also operates under this constraint, which is why even
their RRR/CRR run at 0.66/0.50 of their RCR.

**Decision rule going forward:** before dispatching any new "close X gap"
session, two checks:
1. Run a single-launch rocprofv3 comparison (TK vs BL same shape, same
   GPU) to confirm a real software lever exists.
2. **`llvm-objdump --mcpu=gfx950` of the prebuilt `.o` and grep for the
   instruction the lever names.** P12 Dev G missed that TK already
   issues 658× `buffer_load_dwordx4 ... lds` (gfx950 wide-DTL); P13 Dev C
   caught it in a single research session. Don't dispatch implementation
   Devs on a "missing instruction" hypothesis without disassembly proof.

The only remaining viable optimization target is **TK_RCR catching the
last 3% to BL_RCR**. Per P13 Dev C, the lever is NOT DTL (already in use)
— it is some combination of:
- per-tile-shape (BL uses MT256×256×128, 1 wave/SIMD; TK smaller, 2-wave)
- consumer-side `ds_read` interleave / issue-rate sweep
- per-shape `sched_barrier(0)` placement
All multi-session efforts; expected gain bounded by the 3% gap.

BF16 work is paused unless a clean structural restructure is on the table
(per P10 lessons, local-tweak space is exhausted for CRR).

## Session Log

### 2026-04-18 — P15 (3 research-only Devs, all opus; GPUs 0/3/5; CLOSE)

**Outcome: All 3 Devs returned definitive findings. Dev A + Dev C
silent-timed-out before final memo (P11/P12/P13/P14 pattern continues)
but produced complete research, recovered from streamed assistant
text. No code changes. P16 dispatch decision: operand-reuse restructure
on TK RCR 8-wave inner loop.**

- **Dev A (BL Cijk_ disassembly, GPU 0, complete via partial)** —
  Per-iter slice of one BL `Custom_..._F8F8_..._gfx950` Cijk_ kernel
  for an MT256x256x128 weak shape:

  | Mnemonic                         | TK | BL | TK/BL |
  |----------------------------------|---:|---:|------:|
  | `v_mfma_f32_16x16x128_f8f6f4`    | 64 | 64 | 1.00  |
  | `ds_read_b128`                   | 48 | 32 | 1.50  |
  | `ds_read_b64_tr_b8`/`b128_tr_b16`|  0 |  0 | n/a   |
  | `buffer_load_dwordx4` (DTL)      | 16 | 16 | 1.00  |
  | `s_waitcnt`                      | 10 |  4 | 2.50  |

  **The 1.50× ds_read gap is a count gap, not a width gap.** Both
  use pure `ds_read_b128`; BL fuses more MFMAs per ds_read pair.
  Disasm at `/tmp/p15_dev_a/{bl_kernel.s,bl_main_loop.s,REPORT.md}`.

- **Dev B (Stream-K survey + impl spec, GPU 3, complete)** — **DEFER**.
  Tail-effect analysis on the 6 weak shapes:
  | shape (M,N,K)        | TK/BL today | tail_pct | StreamK helps? |
  |----------------------|-------------|----------|----------------|
  | 16384,37888,3584     | 0.890       |  2.70%   | NO             |
  | 16384,8192,29568     | 0.932       | 18.75%   | YES            |
  | 16384,28672,4096     | 0.910       |  1.79%   | NO             |
  |  8192,37888,3584     | 0.914       |  2.70%   | NO             |
  | 16384,3584,18944     | 0.946       | 35.71%   | YES            |
  |  8192,28672,4096     | 0.917       |  1.79%   | NO             |
  Best-case ≤ 1.5pp on weak-6 geo-mean for ~2-3 P-sessions and
  ~400-600 LOC. Historical precedent: commit `bc4392bf` "Ref is now
  persistent grid matmul and best is non persist" — TK already tried
  + abandoned BF16 persistent grid. CK-Tile reference at
  `/opt/rocm/include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp:247-267,1083-1167`.
  Memo filed to memory: `project_streamk_persistent_grid.md`.

- **Dev C (TK ds_read source survey + gfx950 ISA inventory + ST_v2a
  compatibility, GPU 5, complete via partial)** — Per-template TK
  kernel breakdown of `rcr_exact_8wave_kernel`:

  | Kernel | b128 | tr_b8 | tr_b16 | b96_tr_b6 | mfma | reads/MFMA |
  |--------|-----:|------:|-------:|----------:|-----:|-----------:|
  | RCR    |  168 |    0  |    0   |    0      |  224 | 0.75       |
  | RRR    |   48 |   48  |    0   |    0      |   96 | 1.00       |
  | CRR    |    0 |  144  |    0   |    0      |   96 | 1.50       |

  (Aggregate 1336 b128 + 1360 tr_b8 in P14 Decider table summed all
  3 templates × multiple KI specializations.) **gfx950 has NO
  `ds_read_b128_tr_b8` or `ds_read_b128_tr_b16` instruction** —
  validated via `llvm-objdump --mcpu=gfx950` dictionary on the
  prebuilt `.o`. Source call sites (RCR critical path):
  `include/ops/warp/memory/tile/shared_to_register.cuh:99-104` and
  `:177-186` (both emit `ds_read_b128` for row_l + fp8 + stride==16).

**Combined Lever C verdict:** GO with revised framing. Not a wider
instruction swap (NO-GO), but operand-reuse restructure of TK's RCR
8-wave inner loop to amortize each `ds_read_b128` across more MFMAs.
Closing 48→32 ds_reads/iter matches BL's 0.50 reads/MFMA exactly.

**P16 dispatch (next session):** 1 Dev, 1 day, prototype operand-reuse
restructure on TK `rcr_exact_8wave_kernel`'s K-loop in
`analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`. Risk gates:
(1) VGPR pressure (256-cap → occupancy 2→1; cf. P14 Dev E BF16
    CRR/296 launch_bounds story);
(2) SGPR spill from address-arith hoisting (cf. CRR/296 26-spill in
    `project_bf16_crr_ki296_spill.md`);
(3) FP8 ST_v2a swizzle preservation (must not require preshuffle).
**Gate on rocprofv3 reads/MFMA → 0.50 BEFORE running 6-shape benchmark.**

**Lessons additive to P14:**
1. **Tail-effect calculator is a 30-second lever-killer** for any
   "persistent grid will help shape X" hypothesis. Compute
   `(num_tiles % grid_size) / grid_size` and
   `1 / (num_tiles / grid_size)` per shape; if both are <5% on the
   target shapes, persistent grid cannot close the gap and the
   investigation ends without a Dev sweep.
2. **Always grep git log for past attempts** before scoping a new
   restructure. P15 Dev B's `git log --all --oneline | grep -i
   'stream\|persistent\|sk[0-9]'` surfaced the abandoned BF16 attempt
   in <1 minute and reframed the entire feasibility memo. Add to
   research-Dev prompts.
3. **Per-template disassembly counts beat aggregate counts** when
   reasoning about a single-template kernel's bottleneck. P14 Decider
   reported 1336 b128 + 1360 tr_b8 across all 3 layouts × multiple KI
   specializations; that aggregate looked like 1.0 reads/MFMA but the
   per-RCR-template count is 168/224 = 0.75. Always slice by template
   when the lever is template-specific.
4. **ISA validation costs 1 minute and saves a session.** Dev C ran
   `llvm-objdump --mcpu=gfx950 -d` and grep on the prebuilt `.o` to
   confirm gfx950 has no `ds_read_b128_tr_*`. Reframing Lever C from
   "swap instruction" (dead) to "restructure schedule" (alive) saved
   a P16 implementation Dev from chasing a non-existent intrinsic.
5. **Silent-timeout pattern persists across P11-P15.** Long-running
   research Devs (Dev A, Dev C this session) silent-time-out at
   ~15-25 min wall-time AFTER producing complete findings via
   streaming text. Recovery via `python -c "import json; ..."` on the
   `.jsonl` transcript still works. For P16+, add a "checkpoint to
   `/tmp/<dev>/REPORT.md` every 5 min" instruction so Decider can
   ingest mid-stream rather than relying on final return.
6. **Sibling-repo `git worktree add` blocks parent-repo `git status`.**
   This session, a 9+ min `git worktree add` from `HipKittens3`
   (PID 599501) caused all `git status`/`git diff`/`git commit` in
   `HipKittens2` to hang on shared NFS index/HEAD reads. Treat
   sibling-repo worktree-create as a global serializer; if blocked,
   defer commits to the next session rather than burning Decider time
   debugging git.

### 2026-04-18 — P14 (3 Devs, all opus; GPUs 0/4/6)

**Outcome: All 3 Devs returned definitive verdicts; nothing landed.
Dev D conclusively closes CRR_STEADY knob space; Dev E conclusively
closes BF16 CRR KI=296 launch_bounds path; Dev F + Decider disassembly
recheck close the FP8 RCR DTL lever.**

- **Dev D (FP8 CRR_STEADY1/2_LGKM 2D sweep, GPU 0, complete)** — full
  16-config grid + 3-run noise validation in worktree `agent-a5b15c06`.
  **Cross-config CRR/BL spread = 0.93pp; per-config 3-run noise = 1.5pp.
  Per-config noise EXCEEDS cross-config sweep spread.** Best candidate
  S1=6/S2=4 is +0.19pp over S1=0/S2=0 baseline. Asm verification
  confirms override reaches codegen (`s_waitcnt lgkmcnt(6)` and
  `lgkmcnt(4)` present in CRR steady-state for S1=6/S2=4 vs all
  `lgkmcnt(0)` for baseline). Either (a) LDS ops drain before MMA
  issue, or (b) wait sites are off-critical-path. Knob space
  conclusively closed. Recommendation: drop the macros entirely.
  Dev F's macro infra was never landed in main, so no revert needed —
  just don't redispatch this lever.
- **Dev E (BF16 CRR `__launch_bounds__(_,1)` KI=296 only, GPU 4,
  complete)** — NO LAND. Restructured `gemm_kernel` into
  `gemm_kernel_body` + `__global__` wrapper, added explicit
  `gemm_kernel<CRR,296>` specialization with
  `amdgpu_waves_per_eu(1, 2)`. Build-log resource: VGPRs 245→170,
  AGPRs 0→192, occupancy 2→1, **SGPR-Spill stays 26 (UNCHANGED)**. The
  K=18944 CRR shape additionally hits `hipErrorLaunchFailure`. Verdict:
  the 26-SGPR-spill is **scheduling pressure from the unroll-2
  main-loop address arithmetic, not a VGPR-budget contention** —
  relaxing occupancy doesn't address root cause. Combined with P9 Dev 1's
  finding that unroll-1 fixes the spill but costs +0.88pp wall-clock on
  GPU 2: **BF16 CRR KI=296 SGPR spill is now confirmed as an
  LLVM-scheduler artifact whose elimination via either unroll-1 or
  launch_bounds costs more than the spill itself**. Worktree
  `agent-a66b0af1`.
- **Dev F (FP8 RCR weak-shape per-shape rocprofv3, GPU 6, complete)** —
  Per-shape PMC memo for the 6 weakest TK_RCR shapes. All show identical
  0.92× speedup vs BL with **flat 0.75 LDS/MFMA on TK vs flat 0.50 on
  BL** — TK does 1.47-1.50× more LDS-issued instructions per MFMA on
  every weak shape. hipBLASLt picks the same Tensile family for all 6:
  `Cijk_Alik_Bljk_F8BS_..._MT256x256x128_MI16x16x1_..._DTLA1_DTLB1_PGR2_PLR0_SK3_...`
  → MT256×256×128 + 256-thread WG + StreamK SK3 persistent grid +
  PGR2 + PLR0. Ranked levers: (A) DTL, (B) StreamK persistent grid,
  (C) wider LDS reads. **Dev F flagged inconsistency** that 0.75
  LDS/MFMA + 0 bank conflicts looks like reg-staged stores, not pure
  DTL. Decider performed disassembly recheck on
  `kernel_fp8_layouts-hip-amdgcn-amd-amdhsa-gfx950.o`: 658×
  `buffer_load_dwordx4`, 1336× `ds_read_b128`, 1360×
  `ds_read_b64_tr_b8`, 2688× `v_mfma_f32_16x16x128_f8f6f4`, **0×
  `ds_write*`** — reconfirms P13 Dev C: TK uses DTL on both A and B.
  The PMC LDS-instruction counter Dev F observed counts the
  consumer-side `ds_read` operations; the 1.47× gap is therefore
  **wider-or-fewer LDS reads + StreamK persistent grid**, NOT adding
  DTL. Ranked P15 levers reduce to: **(B) StreamK persistent grid,
  (C) wider/fewer LDS reads (e.g. `ds_read_b128_tr_b16` if applicable
  to the FP8 swizzle layout).**

**Lessons additive to P13:**
1. **Two-step verification on PMC observations.** Dev F observed
   "TK 0.75 LDS/MFMA, BL 0.50 LDS/MFMA" and ranked DTL as lever (A)
   based on hipBLASLt's `DTLA1_DTLB1` autotune string. Disassembly
   showed TK already has DTL on both operands. **Counter labels
   (`SQ_INSTS_LDS`) include `ds_read*`, not just `ds_write*` — a
   high LDS-instruction count is consistent with all-DTL global loads
   plus heavy consumer ds_reads.** Always disambiguate counter
   semantics before mapping a number to a lever name.
2. **Dev F's "flag the inconsistency" pattern is a model for research
   Devs.** Rather than landing a wrong recommendation, Dev F said
   "this looks inconsistent — recheck disassembly". That single
   sentence saved a P15 implementation Dev. Add to research-Dev
   prompts: "If your data conflicts with prior memory or established
   ISA behavior, raise the conflict explicitly rather than picking
   one side."
3. **Dev E's mechanism-level conclusion has shipping value.** Even
   with NO LAND, the build-log evidence (Spill=26 invariant under
   occupancy halving) closes the door on a category of attempts
   ("relax occupancy to fix CRR spill"). File this alongside P9's
   "unroll-1 fixes spill but regresses wall-clock" — the BF16 CRR
   KI=296 spill is now bracketed on both sides.
4. **Loop runtime expiration is a session-end signal.** When the
   `<<autonomous-loop-dynamic>>` scheduler refuses (loop ended), treat
   it as the user wrapping the conversation: write up partial state
   and commit docs immediately, rather than sleeping further on
   in-flight Devs.
5. **Per-config 3-run noise vs cross-config sweep spread is the right
   discriminator for knob spaces.** Dev D ran 16 configs once, then
   re-ran the baseline + best-candidate 3× each. Cross-config spread
   was 0.93pp (CRR/BL); per-config 3-run noise was 1.5pp. The
   inequality `per-config noise > cross-config spread` is a clean
   "this knob has no signal" verdict that beats arbitrary per-shape
   wins. Add to research-Dev prompts: "After your sweep, characterize
   the per-config noise floor with 3-run reruns of two distinct
   configs, then compare to cross-config spread."
6. **Asm verification is part of a definitive knob close.** Dev D
   demonstrated the CRR_STEADY1/2_LGKM override reaches codegen
   (`s_waitcnt lgkmcnt(6)` / `lgkmcnt(4)` present for S1=6/S2=4 vs
   all-`lgkmcnt(0)` for baseline). This eliminates the "knob is dead
   code" hypothesis and lets the verdict hold: the wait sites are
   either off-critical-path or already drained at issue time.

### 2026-04-18 — P13 (3 Devs, all opus; GPUs 0/4/6)

**Outcome: nothing landed; P12 Dev G's DTL premise refuted.**

- **Dev A (FP8 RRR `lgkmcnt(0)` drain restructure, GPU 0)** — added
  `RRR_DRAIN1/2/3/4_LGKM` macros (defaults to identity = `lgkmcnt(0)`),
  replaced 4 raw `asm volatile("s_waitcnt lgkmcnt(0)")` calls in
  `kernel_fp8_layouts.cpp:1633/1651/1668/1678` with `TK_WAIT_LGKM(...)`.
  Smoke test with D1=2 *hung* SNR test on RRR(4096,2048,4096): the
  barrier is **load-bearing for correctness** on at least one shape,
  not just a scheduling hint. Agent silent before completing the
  4-D sweep across non-hanging values. Diff in worktree
  `agent-a784d2cb`. NO LAND.
- **Dev B (BF16 CRR `__launch_bounds__(_,1)` for KI=128/296, GPU 4)** —
  silent timeout. Zero tracked-file changes. NO LAND.
- **Dev C (FP8 RCR DTL feasibility, no kernel edits)** — disassembled
  the prebuilt FP8 `.o` and counted: 658 `buffer_load_dwordx4 ... lds`
  (gfx950 wide-DTL, 16B/lane), 0 non-DTL global loads on the GEMM hot
  path. Confirmed source path:
  `include/ops/warp/memory/tile/global_to_shared.cuh:215-222` calls
  `llvm_amdgcn_raw_buffer_load_lds()`. Both A and B operands DTL.
  ST_v2a XOR swizzle composes via swizzled-global-offset trick.
  **The DTL hypothesis from P12 Dev G is empirically wrong.** Project
  memory (`project_fp8_ceiling.md`) updated with the correction.

**Lessons additive to P12:**
1. **Disassembly grep before lever scoping.** When proposing "we lack
   instruction X, that's why we're slow", ALWAYS first run
   `llvm-objdump --mcpu=gfx950` and grep for X. P12 Dev G's DTL claim
   would have died in 30 seconds of disassembly inspection.
2. **Knob infrastructure with no override values benched is not
   landable.** P12 Dev F shipped `CRR_STEADY1/2_LGKM` macros, P13 Dev A
   shipped `RRR_DRAIN1-4_LGKM` macros — neither found a benched
   override that beat baseline. Default-preserving infra has zero
   shipping value if no override is shown to win.
3. **Cap concurrent same-source Devs at 1**, not 2. The P11/P12/P13
   pattern: agents that get into multi-hour rebuild+bench loops on
   shared kernel sources (4+ Devs in P12) silently hang at high rate.
   Dev A (P13, alone on FP8) and Dev B (P13, alone on BF16) both
   timed out anyway — likely a model-runtime issue, not just file
   contention. Until root cause is found, prefer 1 Dev per source file
   per session.
4. **Research-only Devs return more reliably than implementation Devs.**
   Dev C, Dev G (P12), Dev E (P12) all completed cleanly with
   definitive memos. Dev A/B/D/F (P11/P12/P13) all hung mid-iteration.
   Use research Devs to bound the search space before launching
   implementation Devs.

### 2026-04-18 — P11 + P12 (7 Devs + 1 Reviewer + 1 ceiling research, all opus)

**Outcome: nothing landed; ceiling proven.** Two back-to-back agent-team
sessions targeting the new "FP8 RRR/RCR ≥ 1.000, CRR/RCR ≥ 0.950"
no-preshuffle bar. Critical result: Dev G's rocprofv3 single-launch
profile shows the targets are **architecturally infeasible** — even
hipBLASLt's own RRR/RCR=0.66 and CRR/RCR=0.50.

- **Dev A (P11, RRR waitcnt sweep, GPU 0)** — identified the structural
  blocker: `kernel_fp8_layouts.cpp:1607-1670` has 4× `s_waitcnt
  lgkmcnt(0)` hard-drain barriers per RRR steady-state iteration. They
  mask every per-instruction wait knob. NO LAND.
- **Dev B (P11, RRR alt-tile, GPU 5)** — silent for 2h+, produced 21
  bench JSONs but no kernel diff. Treated as NO LAND (silent timeout).
- **Dev C (P11, CRR LDS double-buffer, GPU 4)** — confirmed
  `__shared__ ST_crr_a As[2][2]` is already double-buffered. The 8% gap
  is structural column-stride cost, not a missing buffer. NO LAND.
- **Dev D (P12, RRR `lgkmcnt(K)` restructure, GPU 4)** — hung 1h+, no
  kernel diff produced. NO LAND (silent timeout).
- **Dev E (P12, CRR LDS layout / bank-conflict, GPU 5)** — direct
  measurement `SQ_LDS_BANK_CONFLICT = 0` on both CRR and RCR.
  Eliminates bank conflicts as a lever. NO LAND.
- **Dev F (P12, CRR `lgkmcnt` knob, GPU 7)** — added
  `CRR_STEADY1_LGKM`/`CRR_STEADY2_LGKM` macros at lines 1968/1983,
  defaults to identity. Single override S1=2,S2=4 benched at +0.27pp
  RRR / +0.43pp CRR — within DVFS noise. Agent died before finishing
  the sweep. Diff in `agent-a5b15c06` worktree, NOT landed.
- **Dev G (P12, ceiling research, GPU 6, no kernel edits)** — definitive
  rocprofv3 single-launch comparison. All 6 kernels (TK ×3 layouts,
  BL ×3 layouts) issue identical SQ_INSTS_VALU_MFMA_F8 = 16,777,216 and
  same MFMA flavor (`v_mfma_f32_16x16x128_f8f6f4`). hipBLASLt's own
  RRR/RCR = 0.66, CRR/RCR = 0.50. TK is 1.49× and 1.93× *faster than
  hipBLASLt* on RRR and CRR. The only kernel beating us is BL_RCR's
  hand-written `Custom_` TN kernel by 3%, using DTL+MT256²+CMS — none
  of which generalize to NN/NT.

**Lessons additive to P10:**
1. **"Vs our own RCR" can be unreachable**, even when "vs hipBLASLt" is
   crushed. RCR has access to TN-coalesced load patterns that NN/NT
   intrinsically can't use (Direct-To-LDS). Future "X-layout / RCR"
   targets need a feasibility check via cross-layout profiling first.
2. **rocprofv3 single-launch comparison (TK vs BL same shape) is the
   strongest ceiling-bounding tool**. Use it BEFORE launching restructure
   attempts. If hipBLASLt itself is no better at the same layout, the
   gap is structural and not addressable by tuning. P12 spent 5 Dev
   slots before Dev G ran this profile.
3. **Multi-Dev parallelism on the same kernel source eventually
   collides.** P12 had 4 concurrent Devs editing
   `kernel_fp8_layouts.cpp`; Devs B, D, F all silently hung or never
   finished a sweep. Cap shared-file Devs at 2 concurrent.
4. **Negative results have shipping value.** P12 Dev G's ceiling memo
   prevents downstream teams from re-attempting the same dead ends and
   should be treated as the headline result of the session.

### 2026-04-18 — P10 (3 Devs + 1 Reviewer in worktrees, all model=opus)

**Outcome: nothing landed for the second session in a row.** All three Dev
tracks bottomed out at DVFS noise OR produced hard regressions when they
moved out of the noise band.

- **Dev A — BF16 CRR SGPR-spill restructure (worktree `agent-a37bb893`,
  GPU 4).** Three strategies tried:
  - Strategy C (KI=128 `#pragma unroll 1`): spills 26→0, VGPRs 245→216
    confirmed in build log. CRR geo-mean -0.31pp. Highly variable
    per-shape (one +0.91pp, one -2.18pp). Same pattern P9 saw on KI=296.
  - Strategy D (`__builtin_amdgcn_readfirstlane` on `row*2`, `row*2+1`,
    `col*2`, `col*2+1`): zero effect on spill count — compiler IR
    already proved uniformity through the chunked transform.
  - Strategy B (manual fusion of two iters + `sched_barrier(0)` + outer
    `#pragma unroll 1`): all CRR KIs spills 26→0, BUT VGPRs 216-245→252.
    With only 4 vector margin to the 256 cap, scheduling degraded:
    CRR -0.90pp geo-mean, (8192,8192,8192) -3.11pp,
    (16384,8192,106496) -3.01pp. **Hard regression.**
  - Conclusion: the unroll-2 + 26-spill point is Pareto-optimal for the
    current `main_loop_iter` shape. Future CRR work needs a real
    structural restructure (running SOFF SGPR per LDS slot, or
    `__launch_bounds__(_,1)` for KI=128/296 only) — not another local
    tweak.
- **Dev B — BF16 RCR/RRR M↔N swap + small-K WAITCNT (worktree
  `agent-a64f2333`, GPU 5).**
  - Strategy A (host grid swap M↔N for tall-N shapes): **architecturally
    infeasible** without invasive changes. Available `gl<>` API doesn't
    accept a transposed C stride; Python-level transpose-copy costs
    5-25% on the target shapes (net regression). Forbidden by no-JIT/
    no-bloat constraints.
  - Strategy B (small-K `lgkmcnt(8)→4`, `vmcnt(6)→2` constexpr-gated on
    `KI<96`): identical VGPR/spill profile, but in DVFS noise on most
    shapes and -0.5 to -1.0pp on (4096,28672,4096) RRR consistently
    across two runs. Reverted.
- **Dev C — FP8 RCR weak shapes (worktree `agent-a5bf93e2`, GPU 0).**
  - Strategy C (per-shape `RCR_TWO_TILE_MIN_KI` runtime knob): swept
    mk∈{0,32,64,128,999999} × gm∈{1,2,4,8,16,32} on 13 weak shapes.
    **mk=0 (default 28) wins or ties on every weak shape**; mk≥64 is
    uniformly 1-2% slower. The current default is already optimal.
  - Strategy B (KI=28/32 template specialization with `if constexpr
    (KI_HINT>0 && KI_HINT<=64) unroll(1)` on the 2-tile main loop):
    clean build, 246 VGPRs, 0 spills, **but A/B geo-mean +0.03pp**
    (pure noise). Per-shape moves were ±1pp in both directions. Reverted.
  - Strategy A (in-block 4-tile split for K-reduction) deferred — would
    need 2+ days and high regression risk; out of session budget.
  - Conclusion: the 12 weak RCR shapes (small-K big-N) appear to be at
    a structural ceiling for the 8-wave 2-tile schedule. Real fix paths
    are either Strategy A (multi-day) or relaxing the no-atomics rule
    for true Split-K.
- **Reviewer / Decider** — no diff promoted to main; nothing to verify
  or commit beyond docs.

**Lessons (additive to P9 lessons):**

6. **VGPR margin matters.** Dev A's Strategy B was the most aggressive
   spill-reduction attempt to date — and the only one that actually
   delivered 26→0 spills on every CRR KI without sacrificing unroll-2.
   It still cost 0.9pp because pushing VGPRs to 252 left only 4 margin
   to the 256 cap and the scheduler degraded. **Track VGPR count, not
   just spill count.** A win on both is rare in the local-tweak space.
7. **Strategy infeasibility is a valid Dev outcome.** Dev B's Strategy A
   correctly stopped at the design step instead of grinding out a
   broken implementation. The Decider should reward this in future
   prompts: "If a strategy needs >1 day or violates the no-JIT/no-bloat
   constraint, write up the obstacle and move to next strategy."
8. **The `vs hipBLASLt` headline can be misleading.** FP8 looks healthy
   at 0.996/1.530/1.967 — but RRR/RCR=0.950 and CRR/RCR=0.922 in absolute
   throughput. Going forward, the primary metric for FP8 is RRR-vs-RCR
   and CRR-vs-RCR, not vs hipBLASLt. (Set by the user 2026-04-18.)
9. **No preshuffle / no offline B reordering.** Don't prototype any
   strategy that depends on it, even as a measurement.

### 2026-04-17 — P9 (3 Devs + 1 Reviewer in worktrees, all model=opus)

**Outcome: nothing landed. All three optimization directions bottomed out
at DVFS noise after cross-GPU validation.**

- **BF16 CRR Dev (worktree `agent-aa1be9f2`, GPU 4)** — KI=296
  `#pragma unroll 1` specialization. Build log confirms SGPR spills
  26 → 0 at KI=296. On Dev's GPU 4: +1.2pp on (8192,3584,18944).
  Reviewer on GPU 2: SAME shape, SAME code → -0.88pp (regression).
  Geo-mean across the layout was -0.28pp on CRR. The unroll-2 → unroll-1
  trade reduces spills but loses barrier-hiding; net negative for CRR
  on this kernel. Cosmetic `readfirstlane` hoist of `row*2/col*2` had
  zero effect on spill count (compiler already factored it). Diff
  preserved in worktree, NOT landed.
- **BF16 RCR/RRR Dev (worktree `team-bf16-rcrrrr-mn`, GPU 5)** — per-shape
  `vmcnt`/`lgkmcnt` autotune via 4-profile `WAITCNT_PROFILE` template
  arg. +0.25pp / +0.17pp consistent across 3 runs but inside the
  calibrated DVFS noise band (per-shape stdev 0.66pp from a CRR
  identical-code calibration). Bloats the .so by ~7×. Diff preserved,
  not landed.
- **FP8 RCR Dev (worktree `agent-a693b720`, GPU 0)** — re-bench of the
  per-shape NUM_XCDS strategy with `--warmup 30 --iters 100 --trials 5`
  on every weak shape. Result: xcd=8 wins on every weak shape by
  0.1-2.5%; the P8 xcd=16 "wins" were thermal noise. Closed item;
  no diff.
- **Decider** — applied Dev 1's diff to main worktree, ran Reviewer
  agent on GPU 2 (different from Dev 1's GPU 4 to rule out thermals),
  Reviewer rejected, working tree reverted. Updated TODO.md +
  agent_prompt.md to record findings. Committed docs only as P9.

**Lessons (additive to P8 lessons):**

1. **`rocm-smi --setperflevel high` is silently broken on this host.**
   Both with and without sudo it returns success but perf level stays
   "auto". Confirmed across P8 + P9. Without clock pinning, ±2pp DVFS
   noise dominates any single-knob effect.
2. **Always cross-validate on a 2nd GPU before committing**, even if
   the change shows a real ISA-level effect (like SGPR spill count
   dropping). Dev 1's KI=296 unroll-1 change ABSOLUTELY dropped spills
   from 26 to 0 — but the wall-clock effect on the target shape was
   *opposite* on a different GPU. The mechanism was real; the
   interpretation was wrong.
3. **SGPR spill count is a means, not an end.** Reducing spills can
   regress perf if the trade (e.g. losing #pragma unroll for spill
   reduction) costs more in barrier-hiding than it saves in load
   pressure. Always measure the wall clock.
4. **+0.25pp consistent across 3 runs is still noise** when the
   calibrated per-shape stdev is 0.66pp. Always calibrate the noise
   floor with an identical-code A/B run before claiming a win in the
   sub-1pp band.
5. **Reviewer must use a different GPU than the Dev** who claimed the
   win. Codifying this in the Decider Checklist now (see step 3).

### 2026-04-17 — P8 (3 Devs in worktrees, all model=opus)
- **FP8 Dev (a67f50ee, GPU 0)** — implemented runtime `g.num_xcds` end-to-end
  in `kernel_fp8_layouts.cpp` + autotune two-phase + bench passthrough.
  4 RCR shapes prefer xcd=16 with measurable wins, but per-shape noise on
  the other 44 cancels at the geo-mean. Mechanism is sound; **not landed**
  pending more aggressive coverage. Diff preserved in worktree.
- **BF16 CRR Dev (a7cbd1fb, GPU 4)** — exhaustive sweep of CRR-only knobs
  (CRR_MAIN_VMCNT/LGKMCNT/UNROLL/NUM_XCDS/CHUNK). All within ±2pp DVFS
  noise band. Reverted. Recommendation for next iteration: pin GPU clocks
  before sweeping; root cause of CRR gap is SGPR spill on KI=128/296.
- **BF16 RCR/RRR Dev (ac9f516a, GPU 5)** — applied same NUM_XCDS-as-runtime
  strategy that FP8 dev tried; **on BF16 it works**: RCR +1.0pp, RRR +1.6pp,
  CRR +1.7pp at the geo-mean, no regressions, SNR/det pass. Landed in P8.
- **Decider** — verified on GPU2/3 in main worktree, committed P8 with both
  the BF16 NUM_XCDS infra and the FP8 MID=6 correction.

Lessons:
- A strategy that's neutral on one kernel can win on another (NUM_XCDS
  was geo-mean-neutral on FP8 because FP8 RCR autotune already pushes
  the strong shapes hard; BF16 had more slack).
- Always pin GPU clocks before doing CRR-class sweeps (the 2pp band
  swallowed several real attempts).
- Sub-agents must `git fetch && git reset --hard <branch>` first if their
  worktree was created from a different branch — `main` doesn't have the
  bf16/fp8 dirs.
