# R22A — LDS_RD_STAGGER_NOP — DEAD END

## Hypothesis (from R21-recon)

R21-recon profiled DLA1/DLA2/DLA7 and found:

| shape | TCP_TA_DATA_STALL / GRBM | LDS bank conflict | HBM peak |
|-------|--------------------------|-------------------|----------|
| DLA1 (4096×32768×128256) | 167.8 % | 0.000 % | 7.8 % |
| DLA2 (128256×32768×4096) | 292.6 % | 0.000 % | 19.9 % |
| DLA7 (28672×32768×4096)  | 294.1 % | 0.000 % | 17.2 % |

The high TCP stall with **zero** bank conflict pointed at LDS port-side
sub-arbitration: multiple `ds_read_b128` issued in adjacent cycles exceeding the
LDS port arbiter's per-cycle bandwidth. The hypothesis was that inserting 1-2
`s_nop` between consecutive `ds_read_b128` rows in the hot path would let the
arbiter drain and reduce TCP stall, while NOPs would be hidden by MFMA pipeline.

## What was built

Added a single source-level macro `LDS_RD_STAGGER_NOP` (default 0) that injects
`LDS_NOP_STR` after each `ds_read_b128` line in **4 hot-path KPAIR functions**:

| Function | ds_read_b128 sites |
|----------|-------------------|
| `kpair_32mfma_with_lds_and_pf`            (_S2)  | 8 |
| `kpair_32mfma_with_lds_rowspread_pf`      (_S3)  | 8 |
| `kpair_32mfma_with_lds_and_pf_swapped_sel` (_S4) | 8 |
| `kpair_64mfma_step12_swapped_sel`         (fused)| 16 |
| **total** | **40** |

Variants:
- `LDS_RD_STAGGER_NOP=0` (default) — empty string, bit-identical to parent
- `LDS_RD_STAGGER_NOP=1` — `s_nop 0\n` (1 idle cycle)
- `LDS_RD_STAGGER_NOP=2` — `s_nop 1\n` (2 idle cycles)

Defaults are 0 so the kernel source change is a no-op for any other agent /
existing variant. Future probes can opt-in by passing `-DLDS_RD_STAGGER_NOP=N`.

## Aperture (3 seeds × warmup=20 iters=50 × GPUs 6-7)

| shape | variant | mean TFLOPS | stddev/mean |
|-------|---------|-------------|-------------|
| DLA2 | baseline | 4231.66 | 0.15 % |
| DLA2 | nop1     | 4207.85 | 0.74 % |
| DLA2 | nop2     | 3745.21 | **11.27 %** (FAIL — high variance) |
| DLA7 | baseline | 4290.28 | 0.09 % |
| DLA7 | nop1     | 4216.87 | 0.44 % |
| DLA7 | nop2     | 3685.97 | **25.84 %** (FAIL — high variance) |
| DLA1 | baseline | 5122.36 | 0.22 % |
| DLA1 | nop1     | 5044.30 | 0.21 % |
| DLA1 | nop2     | 5007.06 | 0.69 % |

`nop2` is unstable (some runs at ~50 % of expected throughput, suggesting it
crosses an MFMA pipeline-stall threshold that interacts with thermals or
DRAM banking).

## Smoke (warmup=200 iters=500 trim=10 % × GPUs 6-7)

| shape | baseline TFLOPS | nop1 Δ | nop2 Δ |
|-------|-----------------|--------|--------|
| DLA2 (128256×32768×4096) | 4206.41 | **+0.01 %** (noise) | **−1.89 %** |
| DLA7 (28672×32768×4096)  | 4297.45 | **−0.93 %** | **−1.12 %** |
| DLA1 (4096×32768×128256) | 4948.59 | **−0.01 %** (noise) | **−1.63 %** |

Gate was `Δpp ≥ +1.5pp` to advance to verify. **None passed.** Best
result on any shape was +0.01 % (within noise). All `nop2` variants regressed.

## Mechanistic explanation

The hypothesis was wrong. Adding `s_nop` between `ds_read_b128` rows does
**not** relieve the TCP_TA_DATA_STALL. Three plausible reasons:

1. **TCP_TA_DATA_STALL counts the *L1-cache-miss* TA-side stall, not LDS port
   sub-arbitration.** Looking at the PMC name more carefully:
   `SQ_PERF_SEL_TCP_TA_DATA_STALL` is asserted whenever the TCP (L1 cache) is
   waiting on the TA (texture-address). For an MXFP4 GEMM with no textures,
   this means **`buffer_load_to_lds` is the load with TCP-TA back-pressure**,
   not `ds_read`. The high stall (200-300 % of GRBM) reflects HBM/L2 latency
   on the producer side (buffer_load), not the LDS port arbiter on the
   consumer side. Spreading `ds_read_b128` does nothing for this — the
   producer is stalled regardless.

2. **The MFMA in-flight occupancy already provides 16+ cycles of latency
   between back-to-back `ds_read_b128` rows.** Each `v_mfma_scale_f32_16x16x128`
   has a 32-cycle issue + 16-cycle pipeline; a single `s_nop 0` (1 idle cycle)
   is invisible at this scale. So the experiment couldn't have shown a benefit
   *if* the bottleneck were LDS sub-arbitration — meaning the negative result
   is consistent with both "wrong bottleneck identified" and "right bottleneck,
   nop too small to matter".

3. **`s_nop 1` (nop2 variant) regression** indicates that the NOPs DO leak
   through the MFMA pipeline at high LDS_RD_STAGGER_NOP — i.e. the issue
   stream IS time-pressured. So MFMA throughput is *not* infinite, but the
   stall is upstream of LDS reads.

## Conclusion

`LDS_RD_STAGGER_NOP` is a **DEAD END** for DLA1 / DLA2 / DLA7. The
TCP_TA_DATA_STALL signal was misinterpreted as LDS port contention; it is
much more likely TCP-TA back-pressure from `buffer_load_to_lds` (HBM-side
producer stall). This is consistent with the raw HBM utilization (DLA2 19.9 %,
DLA7 17.2 %) being well below peak — indicating L2-cache-miss latency, not
HBM-bandwidth-limit, gates the producer.

## Recommendations

- **Reject** any future probe that targets only the consumer-side LDS
  (`ds_read_*` scheduling). The bottleneck is producer-side.
- **Pursue** instead:
  - L2 hit-rate improvement via better tile re-use / static XCD remap (the
    `STATIC_XCD_REMAP` macro family already exists; not yet tuned for DLA2/DLA7)
  - `buffer_load_to_lds` SRD/DLC/SLC cache-control bits (untouched)
  - L2 prefetch hints via `s_inst_prefetch` or `__builtin_amdgcn_global_load_lds`
    with explicit cache hints
  - `MAX_KGROUP_PAIRS_PER_LOOP` reduction (pull more outer-loop iterations
    forward to overlap producer L2 latency with consumer MFMA)
- **Keep** the `LDS_RD_STAGGER_NOP` macro framework in place (defaults = no-op,
  zero-overhead). Future agents may find it useful for probing other functions
  if the producer-side fix opens a real LDS-port-arbiter bottleneck.

## Files

- Macro definition: `kernel_mxfp4_gluon_cpp.cpp` lines 161-186
- Inline injection: 40 `LDS_NOP_STR` insertions after `ds_read_b128` in the 4
  hot-path KPAIR functions (lines ~1462, ~1638, ~1862, ~1782)
- Build: `build_round22_optA.py`
- Aperture: `bench_round22_optA_aperture.py`
- Smoke: `bench_round22_optA_smoke.py` / `.json` / `.log`
