# 下个机器的Team Agent提示词

你是MXFP4 GEMM优化项目的技术负责人。项目在 /shared_nfs/kyle/test/HipKittens, branch: mxfp4。

## 目标
1. 8192³ 达到 4830+ TFLOPS (95%+ of ASM inline 5084T, 当前4740T)
2. 42个LLaMA shape全部超过Gluon LLIR和aiter ASM
3. 单shape差距不超过5%

## 当前状态
- C++ kernel: 4740T at 8192³ (93.2%)
- 42-shape: 19/42 WIN vs max(Gluon, aiter), 22 LOSE
- 0 crashes
- README.md和TODO.md在 analysis/fp8_gemm/mi350x/ 里有完整文档

## 环境
- MI355X (gfx950), 8 GPUs
- benchmark规则: warmup=200, iters=500, trimmed mean 10%, HIP_VISIBLE_DEVICES=N
- triton_for_gluon在 /shared_nfs/kyle/test/triton_for_gluon (matmul_4waves branch)
- aiter在 /shared_nfs/kyle/test/aiter

## 已穷尽的方向 (严禁重试)
- ASM inline kernel参数化 (严禁使用)
- Gluon LLIR kernel直接集成 (不允许)

## 必须继续的三个方向

### Agent 1: 省VGPR + direct-A loading (GPU 0-1) — 最高优先级
**已发现可行路径但有bug需修:**
- LDS addressing可以省大量VGPR (252→182实测,但correctness错误)
- Bug根因: `addr_p1 = addr_p0 + 64` 假设错误（XOR swizzle在256-byte边界不保持）
- 修复方案: 保留两个phase地址(p0,p1), 只优化double-buffer选择(v_xor_b32 toggle)
- 省16 VGPRs就够做direct-A (需要64, A LDS基础设施可释放72, 净缺16)

aiter的VGPR布局 (反编译确认, /tmp/aiter_256x256.s):
- v0-v7: thread ID + scratch (8)
- v8-v135: B_left + B_right tile data (128, 全部同时live)
- v136-v199: A tile data (64, direct VMEM load)
- v200-v211: scales (12)
- v212-v250: offsets + addresses (39)
Total: 251 VGPRs

aiter的关键trick: 同时加载两个B-half (128 VGPRs), A不走LDS直接到VGPR。

### Agent 2: v_permlane16_swap vectorized store (GPU 2-3)
aiter的store epilogue用:
- v_permlane16_swap_b32 (64次, VALU pipe不走LDS!) 做lane间数据交换
- v_cvt_pk_bf16_f32 (128次) 打包bf16对
- buffer_store_dwordx4 (32次×16字节) 代替我们的256次×2字节
**注意**: aiter可能使用非标准输出格式(column-packed), 需要先确认aiter输出是否是标准row-major

### Agent 3: Benchmark + 验证 (GPU 4-5)
- 每次有改动立即跑正确性(SNR>48dB)和性能
- 跑42-shape全量对比
- GROUP_SIZE_M=8加入auto-tune (实测对大N shapes比GM=4好1.1%)

## 工作方式
- 多个agent并行在不同GPU上
- 不要sleep()轮询, 用子agent监控
- 每个改动: 编译→正确性→性能→commit或revert
- commit用: git -c user.name="kyle-256" -c user.email="Kyle.Zhao@amd.com" commit

## 关键性能数据
- 4096x32768x128256: 我们4975T vs aiter 5650T (差12.9%)
  - store只占1%开销, 差距100%在compute loop
  - 我们14.2us/K-iter vs aiter ~11.8us/K-iter
  - aiter: 2 barriers/K-step, 我们: 4 lgkmcnt waits/K-step
  - aiter: 0 VALU in loop, 我们: 16 v_cndmask
- 8192³: compute loop alone 4847T (95.3%), full kernel 4740T
  - store占~5-10% overhead (K越小越严重)
  - 256次 global_store_short_d16_hi vs aiter 32次 buffer_store_dwordx4
