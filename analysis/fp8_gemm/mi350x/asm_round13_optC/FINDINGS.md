# Round 13 OptC ASM-diff Findings

Built on 4096x32768x28672 (DLA3 / R11 _v20_memc_r11_iterilp WIN).
Parent flags: `-DSTEP3_BARRIER_VMCNT=20 -mllvm -amdgpu-sched-strategy=max-memory-clause`

## ASM .s sizes

| Variant | Size | Notes |
|---------|------|-------|
| parent_only (memc) | 224583 | baseline, max-memory-clause active |
| a_iterilp_base (parent + iterilp last) | 223905 | **iterilp active, memc OVERRIDDEN** |
| b_iterminreg_alone | 227184 | iterminreg replaces memc |
| c_iterilp_mmc_dup (iterilp first, mmc last) | 224583 | **identical to parent_only** |
| d_iterilp_pad10 | 223905 | byte-identical to a (cuid-only diff) |
| e_iterilp_pad25 | 223905 | byte-identical to a (cuid-only diff) |
| f_iterilp_brlgk4 | 223905 | real diff: lgkmcnt(0)→lgkmcnt(4) |
| g_iterminreg_mmc | 224583 | identical to c (last sched-strategy wins) |
| i_iter_maxocc | 224637 | distinct, real strategy |

## Key takeaways

1. **`-mllvm -amdgpu-sched-strategy=` is LAST-SPEC-WINS**: when both `iterative-ilp` and
   `max-memory-clause` appear on the cmdline, only the LAST one takes effect.
2. **The R10/R11 `_*_iterilp` WIN variants are PURE iterilp**: the parents'
   `max-memory-clause` flag was silently overridden by the appended iterilp flag.
3. **`memc + iterilp` stack is STRUCTURALLY IMPOSSIBLE** via flags — they use the same
   single-valued LLVM option. They cannot be combined.
4. **`amdgpu-mfma-padding-ratio=N` is a NO-OP on top of iterilp** (size and content
   bit-identical modulo `__hip_cuid` random hash). Confirms R9 OptB finding extends
   to iterilp baseline. **Drop pad10/pad25 from R13C probe.**
5. **`iterative-minreg`** and **`iterative-max-occupancy-experimental`** produce DISTINCT
   ASM (227184, 224637 vs 223905) — real alternative schedulers worth probing.
6. **`STEP12_BR_LGKMCNT=N`** (source-level `#define`) IS orthogonal to sched-strategy and
   produces real ASM changes (lgkmcnt(0)→lgkmcnt(4)). Stackable with iterilp.

## R13C probe set (drop pad ratio, drop memc-stack)

For each of the 5 R10/R11 WIN shapes:
- `_r13c_iterminreg`     — replace iterilp with iterminreg
- `_r13c_itermaxocc`     — replace iterilp with iter-max-occupancy-experimental
- `_r13c_iterilp_brlgk4` — iterilp + STEP12_BR_LGKMCNT=4
- `_r13c_iterilp_v24`    — iterilp + STEP3_BARRIER_VMCNT=24 (override parent VMCNT if any)
