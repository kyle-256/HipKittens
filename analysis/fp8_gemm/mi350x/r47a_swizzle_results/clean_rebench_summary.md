# R47A clean re-bench (3-of-3, cooldown between cells, median TFLOPS)

The original `r47a_rcr_swizzle_sweep.sh` long-running back-to-back execution
suffered SCLK contamination on at least three cells (notably 70B Gate/Up
swizzle dropped to 1395 TF in the original sweep — re-bench shows 2912 TF,
+8.46% over baseline). Below are the trustworthy numbers used for the
default-ON decision.

| Shape                                | M    | N     | K     | Baseline TF (median of 3) | Swizzle TF (median of 3) | Δ%      |
|--------------------------------------|------|-------|-------|---------------------------|--------------------------|---------|
| 8192 cube                            | 8192 | 8192  | 8192  | 3056.3                    | 3049.6                   | -0.22%  |
| 8B Q/O           (4096x4096x4096)    | 4096 | 4096  | 4096  | 2322.9                    | 2389.0                   | +2.85%  |
| 8B Gate/Up       (4096x14336x4096)   | 4096 | 14336 | 4096  | 2548.5                    | 2533.7                   | -0.58%  |
| 8B Down          (4096x4096x14336)   | 4096 | 4096  | 14336 | 2946.3                    | 2981.8                   | +1.21%  |
| 70B Q/O          (4096x8192x8192)    | 4096 | 8192  | 8192  | 2920.1                    | 2903.8                   | -0.56%  |
| 70B Gate/Up      (4096x28672x8192)   | 4096 | 28672 | 8192  | 2685.1                    | 2912.4                   | +8.46% ★|
| 70B Down         (4096x8192x28672)   | 4096 | 8192  | 28672 | 2910.8                    | 2995.1                   | +2.90% ★|

Worst regression: -0.58% (within -1% bound). Four shapes >= +1%. Decision rule met.

## Correctness gates

- 8192³ RCR + swizzle: SNR 49.60 dB, det 3/3 PASS.
- 70B Gate/Up RCR + swizzle: SNR 49.59 dB, det 3/3 PASS.
- All other shapes also PASS at SNR ≥ 48 dB (default test threshold).

## Verdict

SHIP — default ON.
