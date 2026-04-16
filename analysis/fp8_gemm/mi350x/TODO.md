# MXFP4 GEMM Optimization TODO

## Current State (2026-04-16)
- **Repo**: `/shared_nfs/kyle/test/HipKittens`
- **Branch**: `mxfp4`
- **42-shape result**: **18/42 WIN** (warmup=200, iters=500, GPU0)
- **Cursor (Hipkittens2)**: 17/42 WIN (warmup=200, iters=500, GPU5)
- **我们领先 1 WIN，26/42 shapes 绝对 TFLOPS 更高**
- **28/42 shapes 在 5% gap 以内，14/42 超过 5% gap**
- **Target shape**: 4096×32768×128256 = 4967T (85.9% of comp 5781T)

## 逐行对比 (我们 vs Cursor)
```
Shape                      我们    Cursor   差值    我们%   Cur%   谁赢
16384x4096x2048           3145     3046    +99   105.0%  101.7%  >>> 我
16384x4096x3072           3665     3600    +65   104.9%  103.1%  >>> 我
16384x6144x2048           3262     3219    +43   107.0%  105.6%  >>> 我
32768x4096x2048           3295     3260    +35   105.2%  104.1%  >>> 我
32768x4096x3072           3758     3784    -26   103.5%  104.2%  Cursor
32768x6144x2048           3361     3335    +26   103.7%  102.9%  >>> 我
16384x14336x2048          3356     3378    -22   101.7%  102.3%  Cursor
32768x14336x2048          3352     3397    -45   100.0%  101.4%  Cursor
4096x4096x16384           4899     4801    +98   105.5%  103.4%  >>> 我
6144x4096x16384           4413     4260   +153    99.7%   96.2%  >>> 我
4096x4096x8192            4371     4127   +244   110.4%  104.2%  >>> 我
4096x4096x32768           5188     5228    -41   100.7%  101.5%  Cursor
4096x6144x32768           4656     4549   +107   123.0%  120.2%  >>> 我
4096x14336x8192           4334     4283    +51    99.7%   98.6%  >>> 我
4096x28672x32768          5352     5214   +137    94.7%   92.3%  >>> 我
4096x32768x4096           4064     4029    +34    97.5%   96.7%  >>> 我
4096x32768x6144           4186     4140    +46    92.0%   91.0%  >>> 我
4096x32768x28672          5165     5134    +31    92.8%   92.2%  >>> 我
4096x32768x128256(target) 4967     5089   -122    85.9%   88.0%  Cursor
4096x128256x32768         5192     5140    +53   162.5%  160.9%  >>> 我
6144x4096x8192            3969     3790   +179   103.8%   99.2%  >>> 我
16384x4096x4096           4080     3981    +99   103.3%  100.7%  >>> 我
16384x4096x6144           4458     4367    +90   104.6%  102.5%  >>> 我
16384x4096x7168           4555     4473    +81   102.5%  100.7%  >>> 我
16384x4096x14336          5032     5081    -48    97.9%   98.8%  Cursor
16384x4096x28672          4968     4901    +67    89.9%   88.7%  >>> 我
16384x6144x4096           4168     4123    +46   103.1%  102.0%  >>> 我
16384x14336x4096          4219     4254    -35    99.1%  100.0%  Cursor
32768x4096x7168           4530     4502    +28    97.1%   96.5%  >>> 我
32768x4096x14336          4821     4889    -68    92.3%   93.6%  Cursor
128256x32768x4096         3960     4146   -186    87.3%   91.4%  Cursor
```
**我们更好: 26 shapes | Cursor更好: 10 shapes | 相近: 6 shapes**

## Recent Commits
```
c2873f92 direct-B kernel POC — correct but 28% slower (buffer_load latency)
e6cefd88 direct-B POC + spot-test script + expanded auto-tune
08abebd8 tail-split + rowspread + auto-tune → 18/42 WIN
d197e572 add SWAP operand variants to bench_all_42.py
ea990de7 port MFMA operand swap from HK2
8f10b09e reorder store blocks by M-half (A0Bl,A0Br,A1Bl,A1Br)
```

## 已做的优化
1. **Store block reorder** (A0Bl,A0Br,A1Bl,A1Br) — +0.8% 全局
2. **SWAP operand port** from HK2 — 正确但略慢，作为 auto-tune 选项
3. **TAIL_SPLIT=1** — 小K(≤4096)帮助+1-2%, 大K(≥7168)退化
4. **SPREAD_LDS=1** (rowspread ds_reads) — 退化1-2%，作为选项
5. **20-variant auto-tune** (GM1/2/4/8/16, U8/16/32, SWAP, TS, 组合)

## 已证明的死路 (不要再尝试)
| 方向 | 结果 | 原因 |
|------|------|------|
| BK=256 (双倍K-block) | 不可行 | LDS需256KB, max 160KB |
| Direct-B (不preshuffle) | 正确但慢28% | buffer_load 200cy vs ds_read 20cy |
| Preshuffle-B 2-pass | 正确但慢56% | K-loop跑两遍, A tile重载 |
| Preshuffle-B 1-pass | NaN | 126 VGPR spills (256 AGPR + B tiles > 512) |
| Rowspread ds_reads | 退化1-2% | 破坏编译器跨row调度 |
| Non-volatile scale loads | NaN | 编译器重排scale load到消费之后 |
| sched_group_barrier | 无改善 | 编译器已全局最优 |
| ds_bpermute wide stores | 退化16% | LDS crossbar延迟 |
| Direct-A (half/full) | 慢19-44% | buffer_load延迟, VGPR spills |

## 结构性限制 (不preshuffle B 无法突破)
- **B走LDS**: 比aiter多18% TCP read traffic, 2x Frac_Wait_Any
- **256 AGPR**: 4 acc blocks 占满, B tile 数据必须在 256 VGPR 内
- **LDS swizzle**: 不仅是bank conflict avoidance, 实际做数据重排 (已证明 identity)
- **buffer_load vs ds_read**: 10x延迟差距, LDS prefetch pipeline 完全隐藏

## preshuffle-b 分支 (实验性, 不合并)
- **Branch**: `mxfp4-preshuffle-b`
- CK preshuffle格式: `B[N,K//2].view(N//16,16,K//128,4,16).permute(0,2,3,1,4)`
- 2-pass accumulator: 0 spills, 正确, 但慢56% (K-loop跑两遍)
- 1-pass: 126 VGPR spills → NaN
- **结论**: preshuffle 需要全 ASM kernel 才能发挥性能

## Cursor 在做的 (可以学习)
- **schedulable scale loads**: `asm` 代替 `asm volatile` (在他们代码上工作, 在我们这里NaN)
- **STEP3_BARRIER_VMCNT=12**: large-N shapes 上有收益
- **spread_lds + 各种组合**: 针对 large-N shapes 的 auto-tune

## 优先方向 (如果继续优化)
1. **更多 auto-tune 组合** — VMCNT=12, STEP3_PF_N/STEP4_PF_N 变体
2. **Non-volatile scale loads debug** — 找到为什么在我们的代码上 NaN
3. **per-shape 精准调参** — 针对 10 个 Cursor 赢的 shapes

## Benchmark Rules
- **warmup=200, iters=500**, trimmed mean 10%
- 用空闲GPU (`rocm-smi` 确认0%)
- `HIP_VISIBLE_DEVICES=N`
- MI355X 上 competitor_tflops 是正确 baseline

## 关键文件
| 文件 | 用途 |
|------|------|
| `kernel_mxfp4_gluon_cpp.cpp` | 主生产内核 (18/42 WIN) |
| `kernel_mxfp4_direct_b.cpp` | Direct-B 实验内核 (正确但慢) |
| `bench_all_42.py` | 42-shape benchmark (20 auto-tune variants) |
| `spot_test.py` | 单shape多variant测试 (22 variants) |
| `bench_all42_results.json` | 最新42-shape结果 |

## 环境设置 (换机器必读)
```bash
cd /shared_nfs/kyle/test/HipKittens
git checkout mxfp4
cd analysis/fp8_gemm/mi350x

# 用GPU 1-4跑benchmark
HIP_VISIBLE_DEVICES=1 python3 bench_all_42.py

# 单shape测试
HIP_VISIBLE_DEVICES=3 python3 spot_test.py 4096 32768 128256 5781.1

# Cursor的仓库 (只读参考)
# /shared_nfs/kyle/test/Hipkittens2/analysis/fp8_gemm/mi350x/
```
