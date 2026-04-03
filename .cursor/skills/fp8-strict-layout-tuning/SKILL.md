---
name: fp8-strict-layout-tuning
description: "[DEPRECATED] This skill has been split. Use fp8-per-tensor-layout-tuning for FP8 per-tensor work, or mxfp8-mxfp4-layout-tuning for MXFP8/MXFP4 microscaling work."
---
# FP8 Strict Layout Tuning (Deprecated)

This skill has been split into two focused skills:

- **`fp8-per-tensor-layout-tuning`**: FP8 per-tensor strict-layout GEMM on gfx950/MI350X.
  Use for RCR/RRR/CRR performance, bank conflicts, determinism, CRR/RRR loaders, Primus-Turbo.

- **`mxfp8-mxfp4-layout-tuning`**: MXFP8 and MXFP4 microscaling GEMM on gfx950/MI350X.
  Use for preshuffle-quant, scale-pack scheduling, buffer_load SRD, KPAIR_LOOP, mfma_scale.

Read the appropriate skill above instead of this file.
