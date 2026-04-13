## Commands

```bash
make ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=4096

python test_python.py 16 4096 64 8 1 
```

## Asymmetric D192/V128

```bash
make asymmetric ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=4096 ATTN_D_QK=192 ATTN_D_V=128

python test_python_d192v128.py 16 4096 64 8 1
```
