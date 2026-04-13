# D192/V128 Asymmetric Attention Notes

This directory now contains an experimental forward path for asymmetric
attention with `D_QK=192` and `D_V=128`.

## Main Files

- `kernel_d192v128.cpp`: canonical forward kernel wired into `Makefile`
  when `ATTN_D_QK`/`ATTN_D_V` are provided.
- `test_python_d192v128.py`: forward correctness and performance check.
- `bench_d192v128.py`: multi-`N` benchmark driver that recompiles per shape.
- `profile_d192v128.sh`: `rocprofv3` helper for kernel-level timing.

## Preserved Experimental Variants

These are kept on purpose as checkpoints/reference material and are not
selected by the `Makefile` directly:

- `kernel_d192v128_current.cpp`
- `kernel_d192v128_linter.cpp`
- `kernel_d192v128_v1_conservative.cpp`
- `kernel_d192v128_v2_optimized.cpp`
- `kernel_d192v128_v3_tuned.cpp`

## Related Work

- `../gqa_causal_backwards/`: asymmetric forward/backward-prep path plus a
  work-in-progress main backward kernel for the same dimensions.
- `../../../training/llama/csrc/`: MLA forward/non-causal backward experiments
  for `D_QK=192`, `D_V=128`.

## Typical Commands

```bash
make ATTN_N=4096 ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_D_QK=192 ATTN_D_V=128
python test_python_d192v128.py 16 4096 64 8 1
python bench_d192v128.py 1024 2048 4096 8192
```
