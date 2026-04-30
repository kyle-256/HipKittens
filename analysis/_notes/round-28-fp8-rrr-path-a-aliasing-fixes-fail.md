# Round 28 — FP8 RRR fuse path A aliasing fixes all FAIL (5 attempts)

**Date**: 2026-04-30
**Compile gate**: `FP8_RRR_FUSE_PROBE` (default 0; production unchanged)
**Companion**: `round-27-fp8-rrr-path-a-aliased-a-register.md`

## TL;DR

Round 27 identified that A register VGPR is aliased to c register pool by
the compiler post-Epilog 2 cooperative ops. Round 28 attempts 5 fixes
to break the alias chain — **all 5 fail with SNR ~15 dB** (worse than
no-K-tail floor 16.56 dB). RRR path A is fundamentally blocked by
compiler register allocation policy across cooperative-op gaps.

**Production state**: PROBE block remains `#if`-gated default 0. No
metric change (832 baseline noise band). No regression.

**Recommendation for next round**: Pivot away from RRR path A. Three
viable next directions, in priority order:

1. **BF16 RRR path A probe**: BF16 main kernel has different SGPR/VGPR
   topology. Worth a 1-round empirical probe before declaring
   K-tail-fuse-into-RRR-main universally infeasible. BF16 RRR fuse would
   indirectly help BF16 dA bwd path (currently using H4 transpose
   reroute with 20.2% bwd uplift; native fuse could push further).
2. **FP8 CRR (var-K dB) fuse main-line**: CRR (`grouped_crr_kernel` /
   `grouped_var_k_kernel_fp8`) still has external K-tail launches and
   has not been path-B-fused. Mirror RCR fuse path B for CRR.
3. **FP8 RCR/CRR rule-tune saturation re-probe**: User has marked rule
   tune as saturated, but new HK micro-changes (e.g., `BLOCK_SWIZZLE_NUM_XCDS`
   tuning per shape, `RCR_TWO_TILE_MIN_KI` fine-tuning) may unlock 1-2
   metric points. Lower priority because user has explicitly told us
   not to chase rule tune.

## All 5 attempts and results

Probe shape: `G=1, M=2048, N=2816, K=2880, K_REM=64`. fp32 reference.

| Attempt | Description | SNR (dB) | Spill (dwords) | Comments |
|---|---|---|---|---|
| Round-27 baseline | Path B for A using `a` register, no fix | 15.05 | 94 | Original failure mode (round-27 doc) |
| 1 | Fresh `A_row_reg a_kt` declaration | 15.05 | 94 | Compiler aliased a_kt to same VGPR pool as `a` |
| 2 | Fresh `a_kt + b0_kt + b1_kt` triplet | 15.04 | 94 | Same; even 3-tile freshness no help |
| 3 | `a` + `+v` asm pin BEFORE cooperative ops | 15.06 | 94 | Pin holds at asm time, released across coop gap |
| 4 | `a` + sandwich `+v` pin (before AND after) | 15.07 | 94 | Same — compiler maps to "free" VGPR which overlaps c |
| 5 | Pin all `cA/cB/cC/cD` dwords before load_a_kt | 15.07 | 100 | +6 dwords spill, no SNR improvement |

Floor (no K-tail accumulate): SNR 16.56 dB (round-27 SKIP_A_LOAD test).
Production (external K-tail launches): SNR 43.99 dB.

All 5 attempts give SNR < floor → K-tail accumulate is **actively
hurting** numerical accuracy by adding `c @ b_K_tail` term (not
`a_K_tail @ b_K_tail`). This is consistent with `a` register VGPR
holding c values when load_a_kt writes via direct store (not through
inline-asm output constraint).

## Why each attempt failed

### Attempt 1: `A_row_reg a_kt` fresh declaration

```cpp
A_row_reg a_kt;
load_a_kt_into(a_kt);
rrr_mma(cA, a_kt, b0);
```

Hypothesis: declaring a new variable forces fresh VGPR allocation.

Reality: After Epilog 2 `rrr_mma(cD, a, b1)`, `a` is dead. Compiler
retires `a`'s 32 VGPR. Cooperative ops (~30 cycles) trigger no `a` use.
When `a_kt` is declared and first written by load_a_kt, compiler picks
the cheapest available VGPRs — which are `a`'s freed ones, **already
reassigned to c register pressure pool** (because c is live and needs
storage). So `a_kt` is bound to the same physical VGPRs that hold c.
load_a_kt writes corrupt c.

### Attempt 2: `a_kt + b0_kt + b1_kt` triplet

Same hypothesis, more aggressive. b0/b1 register tiles were dead
after Epilog 2 too. Adding fresh b0_kt/b1_kt should also force fresh.

Reality: b0/b1 are written via `load_b` which uses inline asm with
`"=&v"` output constraint — compiler MUST allocate fresh VGPR for them
(asm contract). So b0/b1 freshness is automatic. The 3-tile freshness
attempt only adds a_kt; same alias problem persists.

### Attempt 3: `a` + `+v` asm pin BEFORE cooperative ops

```cpp
asm volatile("" : "+v"(a_dword)); // pin a's VGPR
// cooperative ops
load_a_kt(0); // writes a
```

Hypothesis: `+v` constraint forces compiler to materialize `a` into
a VGPR at the asm point. After asm, the VGPR contains `a`'s value.
Compiler should keep `a` in this VGPR.

Reality: The `+v` pin is satisfied AT the asm instant. Immediately after,
compiler sees `a` has no further USE until load_a_kt's write
(load_a_kt writes through `*reinterpret_cast<__uint128_t*>` — that's
a write, not a read). Between asm and load_a_kt write, `a` is dead;
compiler retires it. Same outcome as attempt 1.

### Attempt 4: Sandwich `+v` pin (before AND after cooperative ops)

```cpp
asm volatile("" : "+v"(a_dword)); // pin 1
// cooperative ops
asm volatile("" : "+v"(a_dword)); // pin 2
load_a_kt(0);
```

Hypothesis: pin 2 ensures `a` is materialized fresh just before
load_a_kt write. Forces compiler to NOT release `a` across cooperative.

Reality: Pin 2 forces materialization into SOME VGPR at asm-2 time.
That VGPR is whichever the compiler picks at that moment — and the
compiler is FREE to pick any VGPR including those holding c. The pin
only requires the value be IN A VGPR, not that it not overlap c.
Same outcome.

### Attempt 5: Pin all `cA/cB/cC/cD` dwords before load_a_kt

```cpp
// pin every dword of c
for h, w, d: asm volatile("" : "+v"(cA.tiles[h][w].data[d]));
// ... same for cB/cC/cD
load_a_kt(0);
```

Hypothesis: by pinning all c dwords, compiler is forced to keep them
in fixed VGPR slots. When load_a_kt writes `a`, compiler MUST pick
VGPR slots NOT used by c.

Reality: spill went up +6 dwords (94 → 100), confirming the c-pin DID
do something — maybe forcing some c dwords to spill rather than hold
them in VGPR. But SNR unchanged (15.07). Suggests `a`'s VGPR may have
been moved off c (compiler complied with c-pin), but landed on some
OTHER live register that we're also reading via mma. Or compiler
spilled `a` to scratch and reloaded with corrupted data. Either way,
the inline-asm "+v" pin pattern is not powerful enough to disambiguate
all the live-range competing.

## Why path A is fundamentally blocked

The cooperative ops (pre-zero loop + G::load + `__syncthreads()`)
inherently require:

* No use of `a` register tile during the cooperative section
  (cooperative writes LDS, not registers; uses thread-shared work
  distribution; needs sync barriers that retire prior register state).
* A long enough instruction span (~30+ cycles, dozens of ALU ops) for
  the compiler to be confident it can spill-free reuse `a`'s VGPRs.

Combined with c register pressure (4 fp32 acc tiles = 128 dwords/lane
= ~1/2 of the 256 VGPR budget at occupancy=2), the compiler's register
allocator HAS to reuse dead register slots. It will pick `a`'s slots
because `a` is the largest dead value.

The only way to break this:

1. **Fresh c-shaped tile c_kt + add chain**: introduce a new fp32 acc
   tile c_kt, accumulate K-tail mma into c_kt instead of `a→c`, then
   `add(cA, cA, c_kt); add(cB, cB, c_kt); ...`. cA/cB/cC/cD remain
   live → c_kt forced fresh. But STILL needs `a` register tile for
   mma input — same alias problem on `a`.
2. **Bypass `a` register tile entirely**: write inline asm that takes
   raw_buffer_load_b128 outputs (in scratch VGPR) directly as mfma
   inputs. Skip the kittens `A_row_reg a` storage. This is invasive
   (requires deriving fp8 mfma instruction encoding) but would fully
   sidestep the alias.
3. **Architectural change**: rewrite the kernel to keep `a` live across
   the K-tail boundary by avoiding cooperative ops entirely. This is
   what RCR fuse does — but RCR's B is row_l register, allowing
   per-lane raw_buffer_load_b128 for B (no LDS, no cooperative). RRR's
   B is col_l register — fundamentally requires LDS-staged load via
   ds_read_b64_tr_b8 (transpose-from-LDS) for the col_l layout. No way
   to avoid cooperative LDS load for B, hence no way to keep `a` live.

Path A is bounded by these architectural facts. Path B and C are
documented as harder (path C round-3 docs: "catastrophic regression for
FP8, compiler doesn't vectorize byte loads, 30x slow").

## Round 29 alternatives

### Alternative 1: BF16 RRR path A probe (RECOMMENDED)

BF16 main kernel `grouped_rrr_kernel` (in `kernel_bf16_dynamic.cpp`)
has different register pressure (no FP8 mfma scale step, fp32 acc same
size, bf16 a/b half-size). Compiler register allocation may be
different — worth a 1-round probe before declaring K-tail-fuse-into-RRR
universally infeasible.

Steps (mirror round-26 plan but for BF16 RRR):
1. Add `#define BF16_RRR_FUSE_PROBE 0` gate at the BF16 RRR kernel
2. Insert path A K-tail block after BF16 Epilog 2
3. Build with `-DBF16_RRR_FUSE_PROBE=1`
4. Run probe shape `G=1, M=2048, N=2816, K=2880`
5. Decision tree based on SNR:
   * ≥25 dB: implement production hybrid for BF16 RRR
   * <20 dB: same fundamental block as FP8, defer
   * 20-25 dB: borderline, debug more

### Alternative 2: FP8 CRR (var-K dB) fuse main-line

CRR / `grouped_var_k_kernel_fp8` still uses external K-tail launches
(per task body P2). Could mirror RCR fuse path B for CRR if its B
register tile is row_l (which it likely is — CRR is dB, B is normally
[K, N] with K-major access pattern).

### Alternative 3: BF16/FP8 RCR fuse extension

Currently RCR fuse only handles K_REM ∈ {32, 64, 96} (32-aligned). If
we could extend to all K_REM values (e.g., per-lane partial-b128 lane
mask), we could remove `grouped_tail_kernel` external launches for
mismatched K_REM. But task body says metric only sees K_REM=64 cases
for forward, so this is metric-irrelevant.

## Files touched (this round)

* `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`:
  * Updated PROBE block inline comment with round-28 attempts log
    (lines 2698-2785 region)
  * No production behavior change (PROBE=0 default)
* `analysis/_notes/round-28-fp8-rrr-path-a-aliasing-fixes-fail.md`
  (this file)

## Verification

```bash
# PROBE=0 production rebuild
cd analysis/fp8_gemm/mi350x
make TARGET=tk_fp8_layouts -B
# SNR = 43.99 dB (matches pre-PROBE-block production)

# Metric (Primus-Turbo)
cd /workspace/code/Primus-Turbo
python3 scripts/_metric_grouped_only.py
# 833 (baseline 832-836 noise band, no regression)
```

## Round handoff

Next agent: **DO NOT continue with FP8 RRR path A** — it's fundamentally
blocked by compiler register allocation across cooperative-op gaps, as
demonstrated by 5 distinct fix attempts all hitting SNR ~15 dB (below
floor). PROBE block is documented in inline comments + this file for
permanent institutional knowledge.

**Recommended pivot**: Try BF16 RRR path A probe (alternative 1) — 1
round of work. If BF16 fails the same way, escalate to alternative 2
(FP8 CRR var-K dB fuse) or alternative 3 (RCR K_REM extension).
