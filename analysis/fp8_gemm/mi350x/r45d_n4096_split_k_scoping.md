# R45 Dev D — N=4096 grid-undersubscription scoping (split-K vs wider BLK_N)

Branch: `worktree-agent-ac0257f8` (off `feat/mxfp8-only` HEAD `37bb9162`)
Date: 2026-04-18
GPU-min: 0 (pure offline analysis)

## TL;DR

For the R44A `gemv_m2_16_decode_kernel` cells excluded at N=4096 (M=8 → 68 %
MXFP8/FP8; M=16 → 53 %; both fail 95 % gate per `r44a_decode_m2_16_findings.md`),
the proximate cause is grid undersubscription: `N / BLK_N = 4096 / 64 = 64 WGs ≪
304 CUs`, leaving ≥ 79 % of the device idle.

Of the two design options surveyed below, **split-K is the only viable path**.
Wider BLK_N either regresses grid count further (BLK_N=128 → 32 WGs) or wastes
half a wavefront (BLK_N=32 single-wave → 32 active lanes). **Recommended R46+
implementation: split-K with SK ∈ {2, 4} and a fp32 scratch + reduction
epilogue (NOT bf16 atomics — gfx950 lacks native bf16 atomicAdd).**

Expected lift on the two excluded cells:

| Cell | Current MXFP8/FP8 | After SK=4 split-K (estimate) | Notes |
|------|------------------:|------------------------------:|-------|
| M=8 × N=4096 × K=4096  | 68 %  | ~88-95 % | grid 64→256 WGs covers 84 % CUs |
| M=16 × N=4096 × K=4096 | 53 %  | ~75-85 % | per-wave K-loop length divides by SK |

Recommendation: **R46+** (NOT R45 — design only). Estimated implementation
effort: 2-3 days for SK=4 + reduction kernel + dispatcher gate. NOT to be
implemented in R45 per task spec.

## 1. Background — current geometry (R44A reference)

From `r44a_decode_m2_16_fastpath.inc`:

```
constexpr int DECODE_BLK_N     = 64;     // one wavefront per WG
constexpr int DECODE_THREADS   = 64;     // 1 wave/WG
constexpr int DECODE_K_TILE    = 1024;   // K elements per LDS A-fill chunk
constexpr int MAX_M_FIXED      = 16;     // max compile-time row count
```

Grid: `(N / DECODE_BLK_N, 1, 1)` — single dimension.
Block: 64 threads (1 wavefront), `M_FIXED` accumulators per lane in VGPRs.
LDS: `M_FIXED * K_TILE` bytes for A streaming + `M_FIXED * K_BLOCKS` bytes for
A_scale (held entire kernel lifetime).

Per-WG resources at the largest M_FIXED=16, K=8192:
- VGPRs: 73 (kernel does not spill at `DK_UNROLL=4`)
- SGPRs: 83
- LDS: `16 * 1024 + 16 * 256 ≈ 20.5 KB`
- Threads: 64 (1 wavefront)

Occupancy: with 20.5 KB LDS / 160 KB CU LDS → **7 waves/CU** by LDS, but VGPR
budget of 73 × 64 lanes = 4672 / 65536 VGPRs → **8 waves/CU** by VGPR. Limited
by LDS to 7. With 304 CUs that is 2128 waves of latent capacity; the kernel
launches **64 waves at N=4096**, i.e. **3.0 % occupancy**.

At N=8192 the kernel launches 128 waves = 6.0 % occupancy. The cells where
R44A SHIPped (M=4 × 8192² → 96 %, M=8 × 8192² → 79 %, M=16 × 8192² → 96 %) all
ride the same 6 % occupancy but win because per-wave reuse of B-byte-per-row
(M_FIXED FMAs per kk per B-byte) amortises HBM bandwidth enough to compete.

The N=4096 cells lose because **every wave's K-loop is the same length
(K=4096) but there are half as many waves**, so total HBM traffic per cell is
actually identical to the N=8192 case but spread over half the device — the
GEMV becomes serialized at the per-wave HBM level.

## 2. Option A — split-K

### Design sketch

Add a `SK` (split-K) factor; each (col_block, sk_idx) pair is one WG. Each WG
processes `K / SK` elements of the K dimension, accumulates into a per-lane
`acc[M_FIXED]` array as before. Epilogue writes per-WG partial to a scratch
buffer at `scratch[col_block, sk_idx, m, lane]` (fp32). A second reduction
kernel sums the SK partials and writes the final bf16 to `g.c`.

Grid: `(N / BLK_N, SK, 1)` — adds a Y-dim for sk_idx.

```cpp
template<Layout L, int M_FIXED, int SK, bool PRESHUFFLED_QUANT>
__global__ void gemv_m2_16_decode_splitk_kernel(layout_globals g, float* scratch);
```

- `sk_idx = blockIdx.y` ∈ [0..SK)
- `k_lo = sk_idx * (K / SK)`, `k_hi = (sk_idx + 1) * (K / SK)`
- K-tile loop iterates over `[k_lo, k_hi)` instead of `[0, K)`
- Epilogue stores fp32 partial to `scratch[blockIdx.x * SK + sk_idx][m][lane]`

Reduction kernel (fp32 → bf16, single launch):
```cpp
template<int M_FIXED, int SK>
__global__ void reduce_splitk_kernel(float* scratch, layout_globals g);
```

- Grid: `(N / BLK_N, 1, 1)` — same X-dim as main
- Each lane sums SK partials for its (m, col), applies `g.scale`, writes bf16 c

### Resource budget (SK=4, M_FIXED=16, K=4096)

Main kernel per WG:
- VGPRs: ~73 (unchanged from R44A — same per-lane accumulators, K-loop length
  divided by SK doesn't change register count, just loop trip)
- SGPRs: ~85 (small bump for sk_idx, k_lo, k_hi)
- LDS: same 20.5 KB
- Threads: 64

Reduction kernel per WG:
- VGPRs: ~16 (M_FIXED accumulators in fp32 + a few index regs)
- LDS: 0 (no LDS needed — direct global gather of SK fp32 partials per lane)
- Threads: 64

Scratch buffer footprint:
- `scratch_bytes = (N / BLK_N) * SK * M_FIXED * BLK_N * 4`
- At N=4096, SK=4, M_FIXED=16, BLK_N=64: `64 * 4 * 16 * 64 * 4 = 1 MB`
- At M_FIXED=8: 0.5 MB. At M_FIXED=2: 0.125 MB. Trivial; one-shot allocate
  per-shape from the dispatcher.

### Grid impact

| Cell | Current grid | SK=2 grid | SK=4 grid | SK=8 grid |
|------|-------------:|----------:|----------:|----------:|
| M=8 × N=4096 × K=4096  | 64 WGs (3.0 % CUs) | 128 (6.0 %) | **256 (84 %)** | 512 (>100 %, oversubscribed) |
| M=16 × N=4096 × K=4096 | 64                 | 128         | **256**         | 512                   |
| M=8 × N=8192 × K=8192  | 128 (6.0 %)        | 256 (84 %)  | 512             | 1024                  |
| M=4 × N=4096 × K=4096  | 64                 | 128         | 256             | 512                   |

Sweet spot for the excluded cells: **SK=4 → 256 WGs ≈ 84 % of 304 CUs**. SK=8
oversubscribes; reduction overhead would dominate.

### Reduction overhead estimate

Per (col, m) the reduction kernel does SK fp32 reads + 1 bf16 write. At
N=4096, M_FIXED=16, SK=4: total `4096 * 16 = 65536` outputs × (4 reads + 1
write) = 320 KB of HBM traffic. At ~1.5 TB/s peak HBM, that is ~210 ns. Total
kernel runtime for the M=16 case is ~ms, so reduction overhead is < 1 %.

### Atomic-add alternative (REJECTED)

An obvious alternative is to skip the reduction kernel and have each WG do an
atomicAdd to global C. **Rejected:** gfx950 (MI355X) lacks native bf16
atomicAdd. A CAS loop on packed uint16 would serialise all SK waves writing
the same (col, m), defeating the parallelism. fp32 atomicAdd is native but
would require a separate fp32 output buffer + a single-pass conversion kernel
— at which point the cost equals the explicit scratch + reduction approach
but loses the determinism guarantee (atomicAdd ordering is unspecified, so
SNR repeatability would fail the 48 dB gate).

### Predicate

```cpp
__host__ inline bool can_use_decode_m2_16_splitk(const layout_globals& g) {
  if (!can_use_decode_m2_16(g)) return false;
  // Only enable when grid-undersubscribed: N small enough that SK helps.
  // Threshold: N/BLK_N * 4 ≤ 304 (CUs) → N ≤ 4864. Use ≤ 4096 as round threshold.
  if (g.n > 4096) return false;
  // Only enable for M ≥ 8 — at M ≤ 4 the K-loop is fast enough that
  // reduction overhead exceeds grid-saturation lift (per ballpark estimate).
  if (g.m < 8) return false;
  return true;
}
```

### Risks

- **Reduction-kernel correctness**: fp32 → bf16 conversion + global scale
  application must match V1-LEGACY-FALLBACK exactly to maintain SNR ≥ 48 dB.
  Mitigation: reduction kernel is a 30-line copy of R44A epilogue with the
  partial summation prepended; can unit-test the reduction stand-alone.
- **Scratch allocator**: dispatcher needs a per-stream scratch buffer (1 MB
  at the largest M_FIXED=16 SK=4 case). MI355X has 288 GB HBM so absolute
  size is trivial, but allocation hygiene is a non-trivial dispatcher detail.
  Mitigation: pre-allocate at first call and cache by (M_FIXED, N, SK) tuple.
- **3-gate retry harness compatibility**: scratch buffer adds one more shape
  parameter to the bench-script env; existing harness would need a small
  patch to pass `R44A_SK=4` through to the .so build.

## 3. Option B — wider BLK_N (REJECTED)

### Design sketch (BLK_N = 128, two-wavefront WG)

Double `DECODE_BLK_N` to 128 → grid = `N / 128`. To keep one-output-per-lane
geometry, doubles the threads per WG to 128 (= 2 wavefronts).

Grid: at N=4096 → 32 WGs. **Worse than current 64 WGs.**

### Alternative — BLK_N = 32 single-wavefront

Halve `DECODE_BLK_N` to 32 → grid = `N / 32`. Keeps 64 threads per WG (one
wavefront) but only 32 lanes do useful work per kk; the other 32 lanes mask.

Grid: at N=4096 → 128 WGs (better). But: per-WG efficiency drops to 50 %
(half of lanes idle), so even with double the WGs the throughput is identical
to the current geometry. Verified by inspection: total `lanes × WGs` is
unchanged.

### Why this whole axis fails

The fundamental constraint is: **at single-wave WG and N=4096, the product
`(N / BLK_N) × WG_LANES = N` is fixed at 4096 active lanes, regardless of how
we slice WGs along N**. The only way to use more than N lanes is to add
parallelism along K (split-K) or M (multi-row WGs, which would deviate from
GEMV geometry into MFMA territory — covered by R45+ priority #4 separately).

**Verdict: REJECTED.** Wider BLK_N is a dead end for grid-undersubscription;
only split-K (Option A) addresses the root cause.

## 4. Option C — full MFMA path (out of scope)

This is R45+ priority #4 (R44 Dev A "M=8 8k×8k 79 % lift — full MFMA path
needed"). Re-tooling V2 fastpath for arbitrary small BLK_M is a 3-5 day effort
per Dev A's design notes. Orthogonal to split-K — the two could compose
(MFMA-based small-M kernel with K split as a post-hoc occupancy lift).

Not in scope for this scoping doc; split-K vs wider BLK_N is the brief.

## 5. Recommendation

**Implement Option A (split-K, SK=4) in R46+.**

### Ballpark expected lift (single-GPU, M=8/16 × N=4096 × K=4096)

| Cell | Current MXFP8/FP8 | After SK=4 (estimate) | Confidence |
|------|------------------:|----------------------:|------------|
| M=8 × 4096²  | 68 %  | **88-95 %** | medium-high |
| M=16 × 4096² | 53 %  | **75-85 %** | medium      |

Lift estimates from:
- Current cell is 21 % CU-occupied (64 / 304 WGs). SK=4 → 84 % CU-occupied.
- HBM-bound regime: 4× more active CUs ≈ 4× more HBM bandwidth used (until
  saturating ~1.5 TB/s peak). Current cell at M=16 × 4096² hits ~870 GB/s
  effective (HBM = 0.873 TF × 1.4e-3 s/iter × ~1 KB scale traffic → back-of-
  envelope), well below peak. SK=4 should push toward 1.2-1.4 TB/s.
- Reduction overhead < 1 % per Section 2.
- M=16 lift bounded by per-WG inner-FMA-bandwidth at `DK_UNROLL=4` (per Dev A
  finding: M=16 with full unroll spilled 464 VGPRs; reducing to DK_UNROLL=4
  brought spill to zero but at cost of ~50 % FMA throughput vs full unroll
  reachable at smaller M). SK=4 doesn't change this; M=16 cell hits a soft
  ceiling near ~80-85 %.

### Implementation plan (R46+, NOT for R45)

Phase 1 — kernel and reduction (1 day):
1. Copy `gemv_m2_16_decode_kernel` to new file
   `r4Xa_decode_m2_16_splitk_fastpath.inc` (whatever cycle implements).
2. Add `int SK` template parameter; insert `k_lo`/`k_hi` loop bounds.
3. Replace epilogue store with fp32 scratch write.
4. Add 30-line reduction kernel `reduce_splitk_kernel<M_FIXED, SK>`.
5. Add dispatcher `dispatch_decode_m2_16_splitk<L, PQ>` that launches main +
   reduction kernels back-to-back on `g.stream`.

Phase 2 — scratch buffer hygiene (0.5 day):
1. Add `static thread_local std::unordered_map<key, float*>` scratch cache
   keyed by `(M_FIXED, N, SK)`.
2. `hipMallocAsync` first call, retain across subsequent calls.
3. `hipFreeAsync` at module unload (if practical) — otherwise leak as
   process-lifetime cache.

Phase 3 — gate, bench, ship (1 day):
1. Macro `MXFP8_DECODE_M2_16_SPLITK_ENABLE` default OFF (mirror R44A pattern).
2. nm-gate extension: add `decode_m2_16_splitk` and `reduce_splitk` features
   to `r38_nm_gate.sh` per R43 NEW rule 4.
3. Bench harness: extend `r44a_decode_m2_16_bench.py` with SK env var.
4. Single-GPU smoke on M=8/16 × 4096² (the two excluded cells).
5. SHIP gate: 95 % MXFP8/FP8 on at least 1 of 2 excluded cells, no regression
   on any R44A SHIPped cell when SK gate path runs.
6. R46+ Reviewer phase: 4-GPU triangulation per R36 3-gate retry + R44 NEW
   rule 1 within-GPU variance check.

### Out-of-scope (DO NOT do in R46)

- bf16 atomicAdd reduction (correctness risk + serialisation; covered above)
- SK=8 (oversubscribes 304 CUs; reduction overhead would dominate)
- Composing split-K with MFMA path (Priority #4) — wait until MFMA path ships
  separately, then re-evaluate
- Generalising split-K to all M_FIXED including M=2/4 (predicate above
  excludes M ≤ 4; at small M reduction overhead exceeds grid-saturation lift)
- Generalising to N > 4096 (predicate excludes; at N ≥ 8192 grid is already
  ≥ 128 WGs which combined with B-side reuse gives ≥ 95 % MXFP8/FP8 per
  R44A SHIPped cells — no need for SK)

## 6. Cross-references

- `analysis/fp8_gemm/mi350x/r44a_decode_m2_16_findings.md` — R44A SHIP-LITE-PARTIAL
  findings, includes the coverage matrix and excluded-cell root cause analysis.
- `analysis/fp8_gemm/mi350x/r44a_decode_m2_16_fastpath.inc` — kernel under study.
- `analysis/fp8_gemm/mi350x/r44a_decode_m2_16_bench.py` — bench harness to
  extend with SK in Phase 3.
- TODO.md R45+ priority #3 — "R44 Dev A excluded N=4096 cells: split-K or
  wider BLK_N — separate scoping" (this doc closes that scoping item).
- TODO.md R45+ priority #4 — "R44 Dev A M=8 8k×8k 79% lift — full MFMA path"
  (orthogonal; covered separately).
