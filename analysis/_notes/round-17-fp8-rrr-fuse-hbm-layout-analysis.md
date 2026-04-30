# Round 17 — FP8 RRR fuse HBM layout analysis: pure path B is INCOMPATIBLE

Followup to round-16 roadmap that recommended FP8 RRR fuse path B
variant A (direct register K-tail). Round 17 derived the HBM layout
constraints and discovered the path B "direct b128 K-tail load" works
for RCR but is **structurally incompatible** with RRR's HBM layout.

This note documents the layout analysis, explains why round-16's
variant A recommendation was over-optimistic, and proposes a corrected
hybrid path for round 18.

## HBM layout: RCR vs RRR

The B tensor `g.b` has shape `[1, G, dim_outer, dim_inner]` (4D). Its
physical bytes are laid out with `dim_inner` contiguous.

**RCR (forward):**
* `b_co(s, k) → {0, group_idx, s, k}` — outer = N, inner = K
* Byte address: `group_idx * N * K + n_row * K + k_col`
* **K is contiguous**.

**RRR (dA backward):**
* `b_co(s, k) → {0, group_idx, k, s}` — outer = K, inner = N
* Byte address: `group_idx * K * N + k_row * N + n_col`
* **N is contiguous**, K is N-strided.

## Implication for path B (direct register load)

Path B works for RCR because its lane→cell mapping requires loading
contiguous K-cells per lane:
```
RCR per-lane load:
    row_lane = laneid % 16          ← N row picked by lane
    k_lane_byte = (laneid/16) * 32  ← K offset, varies per lane
    16 contiguous K bytes → b128 #1
    16 contiguous K bytes → b128 #2
```

For RRR with the SAME lane mapping (row=lane%16, k=(lane/16)*32), the
required cells are at byte addresses:
```
RRR per-lane load (theoretical):
    bytes = group_idx*K*N + (k_lane_byte..k_lane_byte+15) * N + n_col
```

These bytes are **N-strided, not contiguous**. A single
`raw_buffer_load_b128` (16 contiguous bytes) covers `16/N` cells —
useless for K-tail accumulation.

**Conclusion: pure path B (direct register K-tail b128 loads) does NOT
work for RRR's K-axis.** This is why H4 reroute (`b.transpose(-2,-1)
.contiguous()`) is required: it physically rearranges B from RRR layout
to RCR layout, restoring K-contiguity for the rerouted RCR fuse path.

## What WAS confirmed from round-16

Still true:
* `rt_128x16` = `rt_shape<128, 16, 16>` → 32 fp8 elems/lane, 2 b128
  loads/lane in size. **The register-side feasibility is fine.**
* The blocker is on the HBM-side: RRR's HBM storage doesn't have
  K-contiguity, so the b128-load pattern that works for RCR can't be
  applied directly to RRR's B.

What's NOT possible (round-16 roadmap was wrong here):
* "Variant A — direct register K-tail via `kittens::load`" — the
  helper expects K-contiguous source data, which RRR doesn't have.

## Path A with per-group SRD (CORRECTED proposal for round 18)

For RRR's B, use path A (G::load to LDS with per-group SRD bound),
then existing `load_col_from_st` to read into B_col_reg.

### B side: G::load with per-group SRD

The SRD bound check in `raw_buffer_load_lds` correctly clamps OOB on
RRR's K-tail iff the SRD bound = `(group_idx+1) * K * N` (current group
end). For voffset `group_idx*K*N + k*N + col`:
* In-bound iff `voffset < (group_idx+1)*K*N`, i.e., `k*N + col < K*N`,
  i.e., `k <= K-1` (since `col < N`). So `k=K_global..K_global+K_REM-1`
  (the K-tail) maps to OOB → no-op write to LDS.

If LDS is **pre-zeroed** before the partial G::load, the OOB cells
stay 0 → mma sees zero contribution from K=[K_global, fast_k+K_BLOCK)
→ correct K-tail accumulation.

**Catch**: G::load uses its internal SRD setup which is the FULL
tensor (not per-group). To get per-group bound, we either:
* (a) Replace G::load with a custom load that uses per-group SRD, OR
* (b) Pre-zero LDS before G::load AND use per-group `make_srsrc` +
  `raw_buffer_load_b128` per lane + `ds_write_b128` mirroring G::load's
  swizzle pattern (BF16 round-7 hybrid; SNR 25.45 dB but allclose FAIL
  on ~25 % cells).

### A side: must still use path B (direct register)

A's HBM layout in RRR (and RCR): byte = `M_row*K_stride + K_col`.
**K is contiguous** for A (same as RCR). So path B's b128 K-tail load
works for A — direct register load with SENTINEL for K-OOB lanes.

The catch: row M's K=[K_global, K_global+K_REM) bytes overlap row
M+1's K=[0, K_REM) (since row stride = K_global). SRD bound check at
`M_total * K_global` doesn't reject these; the row M+1 data is loaded
into row M's slot — wrong.

**Fix**: Per-lane SENTINEL based on `k_lane_byte + 16 <= K_REM` (which
RCR fuse path B uses). For lanes where the b128 load would extend past
K_REM, use SENTINEL voffset → `raw_buffer_load_b128` zero-fills VGPR
on OOB. **No row-M+1 contamination because we never issue the load**
when `k_lane_byte + 16 > K_REM`.

### Final hybrid for FP8 RRR fuse

```
1. Pre-zero As[tic][0/1] AND Bs[tic][0/1] (cooperative, ~32 cycles)
2. A K-tail: direct register load via raw_buffer_load_b128 with
   SENTINEL for OOB lanes (mirror RCR fuse path B load_a_kt, line
   2271-2289 in kernel_fp8_layouts.cpp)
3. B K-tail: G::load with per-group SRD construction (need to plumb
   per-group SRD through G::load OR write a custom cooperative load
   that mirrors G::load's swizzle pattern)
4. Use load_a (via subtile_inplace) to read A K-tail from LDS into
   A_row_reg — wait, but step 2 wrote to A_row_reg directly, not LDS,
   so step 4 would overwrite. Skip step 2 for LDS path; use full path
   A for both A and B.

CORRECTED:

1. Pre-zero As[tic][0/1] AND Bs[tic][0/1] (cooperative)
2. A K-tail: cooperative LDS load via raw_buffer_load_b128 (zero-fills
   VGPR on OOB) + manual ds_write to As[tic][0/1]. This bypasses the
   row-M+1 contamination because per-lane SENTINEL prevents issuing
   the OOB load. Mirrors G::load's swizzle pattern.
3. B K-tail: same approach for Bs[tic][0/1] with per-group SRD bound.
4. Use existing load_a / load_b helpers to read into A_row_reg /
   B_col_reg.
5. Call rrr_mma (= mma_AB) to accumulate K-tail into cA/cB/cC/cD.
```

This is **BF16 round-7 hybrid path A** approach. SNR was 25.45 dB but
allclose FAIL on ~25 % cells (`warp_row=0 wc∈{1,3}`). Round-7 docs
diagnosed the bug as deeper than `subtile_inplace` SGPR aliasing,
suspected in `ST_B[1][n_strip]` post-epilog-2 LDS layout.

## Round 18 recommendation: empirical numerical probe FIRST

Before re-implementing the hybrid kernel-side, **run an empirical probe
that verifies whether FP8's `ST_v2` LDS layout has the same
post-epilog-2 staleness issue as BF16's `ST_B`**.

Probe shape: gpt_oss-Down B=4 M=2048 K=2880 N=2880 RRR.

Probe protocol:
1. Build a minimal kernel that:
   a. Runs main loop over K=[0, fast_k=2816)
   b. After main loop, pre-zeros As[tic] / Bs[tic]
   c. Issues partial G::load on K-tail (with full-tensor SRD —
      accepting cross-group contamination on B for the probe's
      accuracy bound; just measures the LDS staleness mode)
   d. Calls rrr_mma once to accumulate
   e. Stores cA/B/C/D
2. Compare to fp32 reference (full-K reduction).
3. SNR threshold: ≥ 25 dB AND allclose pass.

If probe SNR < 20 dB → ST_v2 has the same staleness issue → path A
hybrid will fail same as BF16 round-7 → defer to MFMA cell-shape
rewrite (32x32x64) project.

If probe SNR ≥ 25 dB AND allclose passes → ST_v2 doesn't have the
issue → implement full hybrid kernel for production.

## Why round 17 ships ANALYSIS instead of implementation

* Round-16's variant A recommendation was based on register-side
  feasibility (correct: rt_128x16 has matching b128 load count) but
  missed the HBM-side incompatibility (K is not contiguous in RRR).
* The corrected hybrid (path A with per-group SRD + ds_write +
  pre-zero) is BF16 round-7's approach which had documented allclose
  failure mode — re-implementing in 1 round would likely repeat the
  failure without proper LDS staleness diagnosis.
* The empirical probe in round 18 is the lowest-risk way to either
  unblock the kernel implementation OR confirm we should move to the
  MFMA cell-shape rewrite track.

## Side observation: GateUP (K_RRR aligned) hits external kernels too

Post-round-14 H4 gates on `K_RRR % 128 != 0`. For gpt_oss-GateUP:
`K_RRR = N_out_fwd = 5760`, `5760 % 128 = 0` — H4 does NOT trigger.

GateUP's dA path then enters `dispatch_grouped_rrr` with:
* `fast_k = 5760` (aligned), `fast_n = 2816` (NOT aligned;
  N_RRR = K_in_fwd = 2880, `2880 % 256 = 64`).
* `need_tail_run = (fast_n != n)` → TRUE → launches
  `grouped_ntail_kernel_lds_rrr<64>` (covers cols [2816, 2880)) +
  `grouped_tail_kernel<RRR>` (scalar fallback).

So GateUP dA still pays the external kernel cost on N-tail. This is a
SECOND wedge for FP8 RRR fuse: extending the fuse to handle N-tail
natively (similar to RCR's `N_MASKED_STORE` template + ceil_div bpc
coverage).

But fixing the K-tail fuse first is more impactful (gpt_oss-Down has
both K-tail fuse + transpose elimination potential, ~25-30 % bwd wall
combined).

## Score plateau status

* Round 16 docs commit: `0efafd81`
* Round 17 docs commit: this file
* Score: 832-835 noise across rounds 13-17 (no kernel changes since
  round 14 H4 — best achievable forward-K-tail-fuse + dA reroute
  combination)
* All forward K-tail fuse work shipped (rounds 1-14)
* Remaining wedges are kernel-template rewrites (RRR fuse hybrid;
  MFMA cell-shape change) — multi-round projects with documented
  failure modes from prior rounds. Round 17 commits the corrected
  layout analysis to enable round-18 to make a more informed
  implementation start.
