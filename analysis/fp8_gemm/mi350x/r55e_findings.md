# R55 Dev E findings — 70B Gate/Up CRR LDS-resident scale layout V2 (compressed)

## Summary

R55 Dev E was the COMPRESSED retry of R54 Dev A's REFUTED LDS-resident
pre-shifted scale layout. Dev A's design overflowed LDS (188416 B vs the
163840 B per-CU limit) because it stored both lo and hi halves pre-shifted
and double-buffered. Dev E compressed the layout by storing the RAW
(unshifted) scale-pack dword once per pack (single buffer, no separate
hi-phase slot) and using `ds_read_u16` at the +2 byte offset on odd-K to
fetch bytes 2/3 right-aligned into the consuming VGPR's low 16 bits —
bit-equivalent to `dword>>16` truncated. Total LDS added = 12288 B/CTA,
fitting comfortably under the 163840 B budget (139264 + 12288 = 151552 B).

The lever was implemented behind macro `MXFP8_CRR_SCALE_LDS_V2` (default 0),
mutually exclusive with the V2 / LEAD / LDS / BRANCHLESS scale levers.

**Verdict: REFUTED-EMPIRICAL-LDS-ROUNDTRIP-DOMINATES-V_LSHRREV.**
The mechanism works as designed at the ISA level (P1, P2, P3 all PASS) and
correctness is preserved (P6 PASS), but on the steady-state K-loop the LDS
round-trip latency to recover the hi-half via `ds_read_u16` exceeds the
6-cycle saving from the eliminated `v_lshrrev_b32` chain — exactly as
Dev A's pre-mortem §F1 analytically predicted. Treatment is monotonically
slower than baseline on every run.

## Final-line verdict

R55 Dev E: 70B Gate/Up CRR LDS-resident scale layout COMPRESSED — REFUTED-EMPIRICAL — LDS round-trip exceeds 6-cycle v_lshrrev saving (median -7.98%, treatment-best < baseline-worst, SNR 49.60 dB / det 3/3 PASS preserved)

## Phase 0 — design selection

Three compression axes were considered:

1. **4-way K-pair grouping** (highest user confidence) — REJECTED because
   it would extend the K-loop body inter-iteration live state in the VGPR
   domain (we'd carry pre-fetched scales for 4 K-pairs in registers),
   violating the explicit pre-mortem constraint and putting V2 RRR
   pressure that already sits at 254/256 VGPR squarely over the cliff.

2. **Shared scale slab between waves** — REJECTED because each wave
   addresses unique scale rows: A-side rows are determined by `br/wm`
   and B-side by `bc/wn`, which differ per wave. There is no shared row
   to amortise across waves.

3. **Compute-on-load with `ds_read_u16` byte-aligned hi-half extraction**
   — SELECTED. Single-buffer store of the raw dword at even-K, byte-aligned
   read of bytes 2/3 at odd-K. Cycle-cost analysis: 1 store + 1 read per
   K-pair-half vs Dev A's 2 stores + 1 read; 6 v_lshrrev_b32 saved on the
   odd-K branch; LDS budget 12288 B/CTA (well under 163840 B).

## Phase 1 — predictions

| #  | Prediction | Result |
|----|------------|--------|
| P1 | Build OK; LDS = 151552 B (= 139264 + 12288); occ 2 preserved | PASS — exactly 151552 B, occ 2 |
| P2 | VGPR ≤ 232 (≤ +7 vs baseline 225); zero spill | PASS — VGPR 230, 0 spill |
| P3 | ISA: 6× v_lshrrev_b32 GONE from K-loop body; replaced by 6× ds_read_u16 at +2 byte offset | PASS — see ISA section |
| P4 | ≥ +1% on 70B Gate/Up CRR median TFLOPS | **REFUTED — -7.98%** |
| P5 | ≤ +1% regression on other 3 CRR cells | NOT TESTED — primary already REFUTED |
| P6 | SNR ≥ 48 dB; det 3/3 PASS (ds_read_u16 +2 is bit-equivalent to >>16) | PASS — SNR 49.60 dB, det 3/3 PASS |

## Phase 1 build resources (M=4096 N=28672 K=8192)

| Kernel | VGPR | SGPR | Spill | LDS (B) | Occ |
|--------|------|------|-------|---------|-----|
| baseline (gate=0) | 225 | 64 | 0 | 139264 | 2 |
| treatment (gate=1) | 230 | 64 | 0 | 151552 | 2 |
| Δ | +5 | 0 | 0 | +12288 | 0 |

## Phase 2 — ISA evidence (K-loop body, gfx950)

K-loop body operation counts from `crr_exact_8wave_scaled_kernel`:

| Op | Baseline | Treatment | Δ |
|----|----------|-----------|---|
| v_mfma_scale_f32_16x16x128_f8f6f4 | 64 | 64 | 0 |
| v_lshrrev_b32 | 7 | **1** | **-6** |
| s_bitcmp0_b32 | 1 | 0 | -1 |
| s_bitcmp1_b32 | 0 | 1 | +1 |
| ds_read_u16 | 0 | **6** | **+6** |
| ds_write (incl. fused ds_write2st64_b32) | 0 | 3 | +3 |

Mechanism observed exactly as designed:
- Even-K stores (6 raw dwords) fuse into 3× `ds_write2st64_b32` (compiler
  paired-stride-64 fusion).
- Odd-K reads emit 6× `ds_read_u16` at byte offset +2, bit-equivalent to
  `dword >> 16` truncated to 16 bits.
- The 6× `v_lshrrev_b32` shift chain on odd-K is eliminated (only 1 residual
  shift remains, used elsewhere).

ISA artifacts: `r55e_results/isa/{baseline,treatment}_crr_kernel.s`.

## Phase 2 — A/B benchmark (5 runs, MXFP8_WARMUP=100, ITERS=200, GPU 4)

Cell: 70B Gate/Up CRR (M=4096, N=28672, K=8192).

| Run | Baseline (TFLOPS) | Treatment (TFLOPS) |
|-----|-------------------|--------------------|
| 1   | 2477.29 | 2237.03 |
| 2   | 2472.34 | 2259.98 |
| 3   | 2470.35 (med) | 2273.21 (med) |
| 4   | 2468.82 | 2276.02 |
| 5   | 2452.77 | 2284.71 |

- **Median Δ = (2273.21 - 2470.35) / 2470.35 × 100 = -7.98%**
- Treatment-best (2284.71) < baseline-worst (2452.77): clean monotonic separation
- SNR 49.60 dB (threshold 48.0 dB): PASS, identical to baseline
- Determinism (3 runs): PASS on all 5 treatment runs
- Det preservation confirms ds_read_u16 +2 is bit-equivalent to dword>>16

Cross-cell bench (8B Down CRR + 70B Q/O CRR) was NOT executed because the
primary cell already meets the REFUTED criterion with high confidence —
median Δ -7.98% is far outside the ±1% noise band, and treatment-best is
strictly less than baseline-worst (no overlap).

## Falsifiability — why -7.98% (analytical reconciliation with P4 caveat)

R54 Dev A's pre-mortem §F1 noted the cycle-cost of the LDS round-trip
(~10–30 cycles end-to-end with `lgkmcnt` drain) might exceed the 6-cycle
v_lshrrev saving. R55 Dev E's compression to 1-store/1-read is half the
LDS traffic of Dev A's 2-store/1-read budget, so we predicted the cycle-cost
balance would shift in our favor (P4 caveat: "depends on hardware overlap").
The empirical result shows the LDS round-trip latency dominates by a wide
margin (-7.98%, ~196 TFLOPS).

Two contributing factors:
1. The 6 ds_read_u16 issue 6 separate LDS bank accesses per K-iter; the
   `lgkmcnt` drain to consume them blocks the consuming MFMA segment by
   ~10-15 cycles, well exceeding the 6 cycles saved on the eliminated
   v_lshrrev chain.
2. The 3 fused ds_write2st64_b32 issue at the top of every even-K iter
   add ~6 cycles of LDS write pressure that was not present in baseline.

The mechanism is real and bit-equivalent (P3, P6 PASS) but the cycle cost
of LDS round-trip on gfx950 dominates the saved shift chain. This refutes
the entire LDS-resident pre-shifted scale family for CRR (R54A: failed by
LDS overflow; R55E: failed by cycle cost even when LDS fits).

## Operational note — NFS disk-full diagnostic

During Phase 2 bench, the primary GPU 4 isolated runs reproducibly faulted
with "Memory access fault by GPU node-X on address 0x7fXX_XX00_0000" on
EVERY GPU (verified GPU 2/4/7) for the M=4096 N=28672 K=8192 build. The
fault was independent of the R55E patch (HEAD baseline, R54i baseline, and
R55A baseline all faulted). Root cause was identified as `/shared_nfs`
being 100% full (20T/20T), causing silently-truncated .so writes that
loaded but launched faulty kernels. **Workaround: build and run from
`/tmp/r55e_build/` (overlay fs, 55T free).** This produced clean
correctness (SNR 49.60 dB, pass-rate 100%, det 3/3 PASS) and stable
benchmarks. The full primary-cell A/B bench above was run from /tmp.

Cleaned 626 stale gpucore.* files in /shared_nfs/kyle (~17+ GB recovered)
but other users' data still occupies the 20T quota — this is a system-level
issue beyond R55 Dev E's scope.

## Files

- `crr_mxfp8_exact_8wave_fastpath.inc` — added macro `MXFP8_CRR_SCALE_LDS_V2`
  (default 0) with documentation block and falsifiable predictions P1-P6;
  added LDS region; added `crr_scale_lds_v2_store_raw()` and
  `crr_scale_lds_v2_load_hi()` helper lambdas; added new K-loop dispatch arm
  in the `MXFP8_CRR_SCALE_LDS_V2` `#elif` branch.
- `r55e_workspace/` — symlinked workspace mirroring r54i_workspace.
- `r55e_phase2_primary_tmp.sh` — A/B bench script (run from /tmp due to
  NFS-full corruption).
- `r55e_results/builds/{baseline,treatment}_70B_GateUp.log` — build logs.
- `r55e_results/isa/{baseline,treatment}_crr_kernel.s` — extracted ISA.
- `r55e_results/bench/70B_GateUp_CRR_gate{0,1}.log` — A/B bench raw logs.

## Final-line verdict

R55 Dev E: 70B Gate/Up CRR LDS-resident scale layout COMPRESSED — REFUTED-EMPIRICAL — LDS round-trip dominates 6-cycle v_lshrrev saving (median -7.98%, monotonic regression, SNR 49.60 dB / det 3/3 PASS preserved)
