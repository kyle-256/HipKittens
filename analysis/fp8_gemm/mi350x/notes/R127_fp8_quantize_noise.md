# R127 — fp8 quantize noise floor

quantize_fp8 TENSORWISE: scale = max_abs / 448 (e4m3 max), then x→fp8 with rounding.
Inputs scaled ×0.05 in smoke = max_abs ≈ 0.05*sqrt(2π) ≈ 0.12, scale ≈ 2.7e-4.
fp8 e4m3 has ~3 sigbits precision, quantize error ~1/2^4 of scale = 1.7e-5.
Per-element output bf16 (sbits 7) cannot resolve below ~1e-3.
Theoretical SNR floor: log10((max_abs/quant_err)^2)*10 ≈ 70-80 dB if pure quantize noise.
Observed bench SNR 30-55 dB → some compute path adds 20-50 dB noise beyond quantize.
Source likely fp32 accumulator → bf16 cast (rounding).
SNR > 30 dB always acceptable per `[[fp8 spec]]`.
