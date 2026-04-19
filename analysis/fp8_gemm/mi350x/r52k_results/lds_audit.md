R52K LDS audit — bail clause triggered before any compile or GPU bench
=====================================================================

Authority anchors:
- R50C build-remark capture (r50c_findings.md §2):
    crr_exact_8wave_scaled_kernel<true, 2>:
      TotalSGPRs: 50   VGPRs: 227   Occupancy [waves/SIMD]: 2
      LDS Size [bytes/block]: 139264 = 136 KB   Spill: 0/0
- Hardware: MI355X gfx950, 256 CUs, 160 KB addressable LDS per CU
  (hipDeviceAttributeMaxSharedMemoryPerMultiprocessor = 163840)
- Source declaration: crr_mxfp8_exact_8wave_fastpath.inc:323-324
    __shared__ ST_crr_a As[CRR_LDS_OUTER][2];   // CRR_LDS_OUTER = 2 production
    __shared__ ST_crr_b Bs[CRR_LDS_OUTER][2];

Per-tile arithmetic:
  ST_crr_a = ST_v2a = st_fp8e4m3<HB=128, BK=128, st_16x128_v2a_s>
            = 128 * 128 * 1 byte = 16384 B = 16 KB
  ST_crr_b = ST_v2  = st_fp8e4m3<HB=128, BK=128, st_16x128_v2_s>  = 16 KB

Existing LDS budget (CRR):
  As[2][2]            64 KB
  Bs[2][2]            64 KB
  scale packs etc.   ~ 8 KB
  -----------------------------
  TOTAL              136 KB
  free               24 KB

Proposed extension (MXFP8_CRR_ASIDE_DOUBLEBUF=1, As[3][2]):
  added slots         2 (one per wave-tile id)
  added bytes        32 KB
  new total         168 KB
  HW cap            160 KB
  OVERAGE             8 KB  ==> cannot launch

Smaller variant (As_pf[1], single-wave-tile prefetch):
  added bytes        16 KB
  new total         152 KB  (fits, 8 KB margin)
  ISA delta vs baseline:
    expected: +1 * (buffer_load_dwordx4 + ds_write_b128) per K-iter for
              the prefetched A wave-tile
    actual:   ZERO -- the production K-loop ALREADY issues this via
              global_load_a(As[toc][1], br * 2 + 1, k + 1) on line 204
              into the existing tic/toc slot. No new ds_write would be
              emitted because the LDS landing slot is the same.
  See r52k_findings.md §3 for the four-line K-loop excerpt with the
  existing 3-deep iter-ahead pipeline.

Conclusion: no fitting variant is non-trivial; no non-trivial variant
fits. Bail per prompt step 6 ("REFUTE without GPU bench"). No GPU time
consumed; no source patch landed; no compile attempted.
