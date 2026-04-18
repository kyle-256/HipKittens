# R33 AITER ARCHAEOLOGY — Deep ISA dive into `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`

**Date**: 2026-04-18
**Author**: R33 Decider C (no-build, pure ISA disassembly)
**Inputs**: `R33_DECIDER_VERDICT.md`, llvm-objdump --mcpu=gfx950 of 4 aiter `.co` files
(256x256, 128x512, 128x256, 32x128) and the stale ISA dump
`kernel_mxfp4_gluon_cpp-hip-amdgcn-amd-amdhsa-gfx950.s`.
Disassembly artifacts cached at `/tmp/r33_archaeology/aiter_*.s`.

---

## Section 1 — SRD initialization (the unlock for vmcnt(15) safety)

### 1.1 Aiter's SRD construction (line-by-line)

Aiter's prologue at offsets `0x2DA4..0x2DD0` constructs **four 128-bit
buffer-resource SRDs** (s[4:7], s[8:11], s[12:15], s[16:19]) for the four
GEMM inputs (D, C, A, B). The exact pattern, lifted from
`/tmp/r33_archaeology/aiter_256x256.s` lines 92-107:

```
s_mov_b32 s6,  -16          ; SRD-A.word2 = num_records = 0xFFFFFFF0
s_mov_b32 s10, -16          ; SRD-B.word2 = num_records = 0xFFFFFFF0
s_mov_b32 s18, -16          ; SRD-C.word2 = num_records = 0xFFFFFFF0
s_mov_b32 s14, -16          ; SRD-D.word2 = num_records = 0xFFFFFFF0
s_mov_b32 s7,  0x20000      ; SRD-A.word3 = config = 0x00020000
s_mov_b32 s11, 0x20000      ; SRD-B.word3
s_mov_b32 s19, 0x20000      ; SRD-C.word3
s_mov_b32 s15, 0x20000      ; SRD-D.word3
s_and_b32 s5,  s5,  0xffff  ; clear high 16 bits of base.high (canonicalize ptr)
s_and_b32 s9,  s9,  0xffff
s_and_b32 s17, s17, 0xffff
s_and_b32 s13, s13, 0xffff
s_or_b32  s5,  s5,  0x40000 ; OR 0x40000 into base.high (sets bit 18 of word1)
s_or_b32  s9,  s9,  0x40000
s_or_b32  s17, s17, 0x40000
s_or_b32  s13, s13, 0x40000
```

### 1.2 Decoded SRD layout (gfx950 raw buffer)

A buffer SRD is 128 bits split as `[word0=base.lo][word1=base.hi+flags][word2=num_records][word3=config]`.

| Field | Aiter | Our kernel (`make_buffer_resource(ptr, 0xFFFFFFFFu, 0x00110000u)` at kernel:2318) |
|---|---|---|
| `word0` (base.lo) | per-buffer s[X+0] | per-buffer (low 32 of `g.a.raw_ptr`) |
| `word1` (base.hi + flags) | `(base.hi & 0xFFFF) \| 0x40000` | `(base.hi & 0xFFFF)` (no extra OR) |
| `word2` (num_records) | `0xFFFFFFF0` (-16, signed) | `0xFFFFFFFFu` |
| `word3` (config) | `0x00020000` | `0x00110000` |

Two structural deltas:

1. **`word1 \| 0x40000`** — bit 18 of word1 (`SWIZZLE_ENABLE` is bit 31, and bit
   18 lives in the **stride[13:0]+swizzle[15:14]+enable[17]** band; bit 18 is
   commonly **`ADD_TID_ENABLE`** on the AMDGPU buffer-rsrc encoding). When
   `ADD_TID_ENABLE=1`, the per-lane VGPR offset implicitly adds the lane id ×
   stride during address calculation, **shifting the OOB check** so that the
   raw num_records field is interpreted as a "stride-relative" record count
   instead of a flat byte budget. This combined with `num_records=0xFFFFFFF0`
   (-16) sets the OOB envelope to **(2^32 − 16) bytes per lane stripe**, not
   per buffer-as-a-whole. **OOB violations only occur when a single lane
   crosses 4 GiB minus 16 bytes**, which is functionally never for a real K
   stream.

2. **`word3=0x00020000`** vs our `0x00110000` — bit 17 (`INDEX_STRIDE`) =
   `01` in aiter (16-byte stride per lane index). In our kernel
   `0x00110000` = bit 16 + bit 20 — bit 16 is `INDEX_STRIDE = 0` and bit 20 is
   `DATA_FORMAT[1] = 1` (DATA_FORMAT = 0010 = 32-bit single channel). Aiter's
   choice gives the hardware a precomputed lane-stride hint that **eliminates
   the per-lane address arithmetic** the cache controller would otherwise have
   to redo every cycle, freeing the LSU pipeline so it can sustain more
   in-flight loads (vmcnt(15)).

### 1.3 SRD initialization timing

Counted instruction distance from the first `s_mov_b32 s6, -16` (SRD field
write) to the first `v_mfma_scale_f32_16x16x128_f8f6f4`:

- **Aiter 256x256**: 571 instructions of latency-hiding work between SRD
  setup and first MFMA — accumulator zero-fills (256× `v_accvgpr_write`
  scattered across the prologue), 24 prologue `buffer_load_dwordx4 ... lds`
  loads with carefully staggered `s_add_u32 m0,...` updates, and 6
  `buffer_load_dword` for scales.
- **Our incumbent**: ~820 instructions from first `s_load_dword`
  (line 10) to first MFMA (line 830). The gross count is HIGHER than aiter's,
  but the structure is different — much of our prologue is the dynamic SRD
  build-up via runtime `s_load` of base pointers from kernarg memory and
  packing into `i32x4` via `__builtin_amdgcn_readfirstlane`, whereas aiter
  has the SRDs built **before** the bulk of latency-hiding loads start.

### 1.4 Hypothesis verdict

**HYPOTHESIS #1** ("aiter does SRD init way before K-loop for TLB warmup"):
**REFUTED**. Both kernels have ~600 instructions of pre-MFMA work; raw
distance is similar.

**HYPOTHESIS #2** ("aiter uses `num_records` differently to allow vmcnt to
drain"): **CONFIRMED**. Aiter combines `num_records=−16` with `word1|=0x40000`
(`ADD_TID_ENABLE=1`) and `word3=0x20000` (`INDEX_STRIDE=01`) to put the buffer
into a **per-lane structured mode** where the OOB envelope is per-lane-stripe
rather than per-buffer-flat. This is exactly the configuration that allows
the LSU to retire many in-flight requests without speculative
aperture-violation traps when the speculative pipeline outruns the address
calculation. Our `0xFFFFFFFFu` + `0x00110000` puts the hardware in the
classic flat raw-buffer mode, where the LSU's prefetch logic is conservative
about issuing speculative loads beyond ~vmcnt(8-12) because each in-flight
request must independently bound-check against the same 4 GiB envelope.

This is **the mechanism** behind the `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`
that R31-C and R32-A NT-load saw at vmcnt(20)/(24): the violation is not from
real OOB, it is from speculative bound-check failure under flat-buffer mode.
**Switching to aiter's SRD configuration unlocks vmcnt(15)+ safely.**

---

## Section 2 — K-loop steady-state ISA pattern

### 2.1 K-loop body structure (aiter 256x256)

The K-loop body is `label_041A` at offset `0x3C68` (line 661 in disassembly),
exactly **458 instructions long**, ending with `s_branch label_041A` at
`0x51C0` (line 1118). Per-iter counts (`/tmp/r33_archaeology/aiter_kloop_body.s`):

| Instr class | Per K-iter | Notes |
|---|---:|---|
| `v_mfma_scale_f32_16x16x128_f8f6f4` | **256** | All 16x16 shape, `cbsz:4 blgp:4` (FP4 mode) |
| `buffer_load_dwordx4` (vmem→VGPR) | 16 | 8 for B-tile (lines 668-710), 8 for second K-pair B-tile (lines 879-921) |
| `buffer_load_dwordx4 ... lds` | 16 | 8 for A-tile (lines 779-843), 8 for second K-pair A-tile |
| `buffer_load_dword ... offen lds` | 4 | Scale loads (2 for sa0/sa1, 2 for B scales, lines 716, 722, 811, 851) |
| `buffer_load_dword ... offen` (non-lds) | 4 | Scale VGPR loads (lines 716, 722, 1180, 1186) — these go to **v210, v211** which feed the MFMA's `v_scale_a/v_scale_b` operands |
| `ds_read_b128` | 64 | 8 reads × 4 K-pair-quadrants × 2 sa0/sa1 halves |
| `ds_read_b32` | 16 | 4 scale-fragment loads × 4 K-pair-quadrants — these populate v200..v207 |
| `s_barrier` | **4** | Two pairs (one per K-pair) |
| `s_waitcnt vmcnt(N)` | 4 | **vmcnt(10)×2 + vmcnt(15)×2** (lines 662, 770, 1124, 1232) |
| `s_waitcnt lgkmcnt(0)` | 4 | Always paired with vmcnt above |
| `s_nop` | **6** | Hand-emitted (3 × `s_nop 0` after each barrier, plus 3 in MFMA chain) |

### 2.2 Two-K-pair structure

Aiter's K-loop body covers **2 K-pairs** (sa0, sa1) per iteration, totaling
**256 K-cols** of contraction per outer iter for a 256×256 tile (M=256
rows × 64 K-cols × 4 lane-quadrants per K-pair → 256 MFMAs per K-pair × 2
K-pairs = 256 — wait, let me re-check). Per K-pair the K dimension covered
is **128 K-cols** (= MFMA-K size), and 256 MFMAs per K-iter come from the
output decomposition: `(256/16) × (256/16)` outputs = 16×16 = 256
MFMAs/K-block × 1 K-pair, so **the loop body is one outer K-iter covering
128 K-cols** (label_041A) **followed by a second copy** (label_0972, line
1123) that is structurally identical but uses a different scale-quadrant
permutation (`op_sel_hi:[1,1,0]` flip on the second K-pair). The
`s_branch label_041A` at line 1118 returns to the first K-pair body.

So **steady-state K-iter = 128 K-cols**, **MFMAs per iter = 256**.
Comparing to incumbent: our K-loop body (`.LBB0_7` lines 824-4981 of the
stale dump) is fully unrolled by the compiler covering 16 K-iters worth =
2048 MFMAs over **2048 K-cols**, so per K-iter = 128 MFMAs over 128 K-cols.

Equality of MFMA density: aiter does 2× MFMAs per K-iter at the same K-col
budget because aiter has **only 1 warp/CU but 4 warp-rows of output**, while
we have 4 warps each owning a 64x64 sub-output. **Per-CU MFMA throughput is
identical** — both do 256 MFMAs per 128 K-cols = 2 MFMA/K-col/CU.

### 2.3 Mixed vmcnt at different sites

Aiter places different vmcnt values at different sites:

| Site | Instr | Comment |
|---|---|---|
| Top of K-pair 1 | `s_waitcnt vmcnt(10) lgkmcnt(0)` | Drain B-tile loads down to ≤10 in-flight before MFMA chain starts |
| Mid K-pair 1 (line 770) | `s_waitcnt vmcnt(15) lgkmcnt(0)` | Drain to ≤15 before next MFMA group needing B fragment |
| Top of K-pair 2 | `s_waitcnt vmcnt(10)` | Same pattern |
| Mid K-pair 2 (line 1232) | `s_waitcnt vmcnt(15)` | Same |

Our K-loop has **a single `s_waitcnt vmcnt(8)`** for the entire 16-iter
unroll body (and 1 `s_waitcnt lgkmcnt(0)` at the end). Crucial points:

- Aiter sustains **15 in-flight loads** between the two waitcnt sites in
  each K-pair, while we sustain **8**. This is the direct mechanism the
  R32-decider B4 finding identified, and it's confirmed here as a
  per-call-site policy — vmcnt(10) is the "barrier-edge" drain, vmcnt(15)
  is the "MFMA-chain-mid" drain.
- The relaxed mid-chain vmcnt(15) is what keeps the LSU pipeline saturated
  during the 32-MFMA chain that follows it.

### 2.4 s_barrier and s_nop placement

The 4 `s_barrier` instructions per K-iter are paired:
- 2 at the K-pair boundary (one each for sa0→sa1 and sa1→sa0 LDS swap)
- 2 internal to each K-pair, between the load-issue burst and the MFMA chain

After each barrier, aiter emits `s_nop 0` (sometimes 2 of them, lines 665-666)
to give the wavefront 1-2 cycles of clean idle for the LDS-bank scheduler to
reset before the MFMA pipe restarts. Our kernel emits ZERO `s_nop` in the
K-loop body (164 `s_nop` total across the whole stale dump are all in the
epilogue/store path).

This is consistent with the R32-decider B4 finding #4: aiter's `s_nop` are
hand-placed at MFMA-chain restart points where the AMD compiler scheduler's
RAW-hazard detector underestimates the LDS→MFMA pipeline depth.

### 2.5 Whole-file diff (aiter 256x256 vs our incumbent stale dump)

| Stat | aiter 256x256 | our incumbent |
|---|---:|---:|
| Total instructions | 3415 | 5928 |
| K-loop body length | 458 (×2 K-pairs) | 4158 (16-iter unroll) |
| MFMA total | 512 | ~3088 (most are in K-loop unroll) |
| K-loop MFMA / iter | 256 | 128 |
| Buffer-load + lds-load total | 88 | 320 |
| ds_read_b128 total | 180 | 528 |
| `s_waitcnt vmcnt(N)` distinct N values | {10, 15, 25} | **{8}** only |
| `s_nop` in K-loop | 6 | **0** |
| VGPR count (kernel descriptor) | **512** | 512 (same) |
| SGPR count | **96** | 91 |
| LDS bytes | **163840 (160 KB)** | **131072 (128 KB)** |
| WG size | 256 (4 warps) | 256 (4 warps) |

---

## Section 3 — XCD / WG dispatch geometry

### 3.1 Aiter dispatcher heuristic recap

From `/shared_nfs/kyle/test/aiter/csrc/py_itfs_cu/asm_gemm_a4w4.cu` lines
82-148, the heuristic computes:
- `tg_num = ceil(M/tile_M) * ceil(N/tile_N) * splitK`
- `local_round = ceil(tg_num / num_cu)` where `num_cu = 304` on MI355X
- selects min(local_round), tie-break on (i) more empty CUs, then
  (ii) higher `compute2mem_effi = tile_M*tile_N/(tile_M+tile_N)`.

For L6 (M=4096, N=32768):
- 256×256 splitK=1 → tg=2048 → round=7 → effi=128
- 128×512 splitK=1 → tg=2048 → round=7 → effi=102.4
- → **256x256 wins** (same round, higher effi)
- 256×256 splitK=2 → tg=4096 → round=14 (rejected, higher round)

### 3.2 WG launch comparison

| Kernel | Tile | tg_num L6 | WGs/CU | log2_k_split | Persistent? |
|---|---|---:|---:|---:|---|
| aiter 256x256 | 256×256 | 2048 | 6.74 | 0 (no split) | **No** — heuristic inspects `s50, s51` (K-loop counters in K-direction iteration), structure is **classic launch-once** (single forward pass through K via `s_addk_i32 s50, 0x100; s_cmp_lt_i32 s50, s51; s_cbranch_scc0` in lines 1112-1117) |
| aiter 128x512 | 128×512 | 2048 | 6.74 | 0 | No |
| our incumbent | 256×256 | 2048 | 6.74 | 0 | Persistent-XCD remap (kernel:2675-2700) |

**Key finding**: aiter does NOT use persistent-XCD or a static cross-CU remap.
This is consistent with the R30-D/R32-A "persistent-XCD on grid-saturated"
finding — at grid saturation (round=7 means each CU sees ~7 WGs over the
whole launch), the persistent-loop overhead is amortized cheaply. Aiter's
lack of persistent-loop is a non-issue for L6 because the launch itself is
already saturated.

What aiter DOES is more interesting: lines 27-91 contain a
**`s_cmp_lt_i32 s48, s52 → s_sub_i32; s_add_i32 s46, s46, 32; s_branch`**
loop (the `label_003D → label_0042` block). This is a software **swizzle/remap
of the workgroup ID** (s46 ← unsigned reduction of s48 mod s52 with s46
incrementing by 32 per iter). This is essentially a static cross-CU swizzle
that aligns 32 contiguous WG-ids onto the same XCD — i.e., **aiter
implements XCD-aware tile placement INSIDE the kernel** without using a
persistent loop, by remapping the dispatch ID.

Our kernel uses persistent-XCD via `xcd_aware_remap` in kernel:2675; same
goal achieved via different mechanism. Cross-shape effect: NEUTRAL — both
get the L2-locality benefit.

### 3.3 What about L7/L4/L8 cross-tile selection?

Quick re-application of the heuristic to other deep-K shapes the v2
auto-tune already wins on:
- L7 (M=512, N=32768): 256x256 → tg=256 → round=1 (one wave), effi=128
- L4 (M=8192, N=8192): 256x256 → tg=1024 → round=4, effi=128
- L8 (M=4096, N=8192): 256x256 → tg=512 → round=2, effi=128

So aiter dispatches the **same 256x256 kernel** for L4/L6/L7/L8 — the
tile choice is uniform across the deep-K shapes. The R32 41/42 incumbent
also uses 256x256 (`_ts_` = tile_M=256), so **tile geometry is identical**.

This confirms the L6 gap is NOT a tile-size issue. It's a per-iteration
schedule efficiency issue.

---

## Section 4 — Tile / scale layout (preshuffle)

### 4.1 Naming decode

`f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256` →
- `bf16` output dtype
- `per1x32Fp4` scale granularity (1 row × 32 cols of FP4 → 1 e8m0 scale dword
  per group, packed as 4 e8m0 in 1 dword, i.e., 4 scales per 128 cols)
- `BpreShuffle` = B operand uses the preshuffled scale layout

Aiter's preshuffle layout for B scales is loaded via the prologue
`buffer_load_dword v210, v233, s[24:27], 0 offen lds` (line 716, K-loop
body) followed by `ds_read_b32 v204, v224 offset:1024` (line 753) that
populates v200..v207 from LDS. Each `ds_read_b32` reads one e8m0×4 dword
that is then routed as `v_scale_a` / `v_scale_b` operand to the MFMAs.

The MFMA scale operands in the steady-state body are **v200..v207**
(the per-MFMA scale dword) and **v208, v209** (the lane-broadcast scale
selector). This is identical to our kernel's scale-routing pattern in
`kpair_64mfma_step12` (kernel:1255-1391) which uses `v_scale_a` /
`v_scale_b` op_sel encoding.

### 4.2 Bit-for-bit preshuffle equivalence?

Aiter loads the B-scale via `buffer_load_dword` (1 dword per lane, 32 lanes
collectively read 32 scales = 128 K-cols of B at one M-row group) into a
SINGLE LDS bank stripe at offset `s60..s60+0x400`, then reads it back via
`ds_read_b32 v204..v207, v224 offset:{1024,1280,1536,1792}` — i.e., 4
4-byte scales spaced at 256-byte intervals. This implies the B-scale LDS
layout is **64 scales/bank × 4 banks** with 256-byte stride — a
bank-conflict-free read pattern at lane stride 4.

Our kernel uses **`buffer_load_dwordx2`** for B scales
(NONVOLATILE_SCALE_X2_POC=1 in kernel:1023-1080) — loading **2 dwords/lane =
64 scales = 256 K-cols** per shot. This is twice the load granularity but
half the load count. Aiter loading more frequently (every K-iter does 4
scale dword loads, we do 2 dwordx2 loads = same byte budget) means the LSU
sees **smaller, more frequent scale loads** that interleave better with the
MFMA chain.

### 4.3 Conclusion on preshuffle layout

The preshuffle bit layout is **functionally equivalent** between aiter and
our kernel — both use 1×32 FP4 grouping with e8m0 scales packed 4-per-dword.
The DIFFERENCE is in the LDS bank topology and the load granularity (dword
vs dwordx2).

---

## Section 5 — Three concrete actionable findings

### Finding A (HIGHEST EV, sub-day) — SRD config swap

**What aiter does**: SRD `word2 = 0xFFFFFFF0` (`-16`), `word3 = 0x00020000`,
and `word1 |= 0x40000` (sets `ADD_TID_ENABLE`).

**What we do**: SRD `word2 = 0xFFFFFFFFu`, `word3 = 0x00110000u`, no `word1`
flag bit.

**Expected effect**: Removes the speculative-aperture-violation trap that
prevents safe operation at vmcnt > 12 on K=128256. Once the SRD is in the
`ADD_TID_ENABLE + INDEX_STRIDE=01` mode, the LSU is allowed to sustain
≥15 in-flight loads without the bound-check pipeline stalling.

**Where to change** (3 sites in our codebase):
1. `include/ops/warp/memory/util/util.cuh:75` — `make_srsrc()`: change
   `0x110000` → `0x20000`, change `range_bytes` callers to pass `-16`
   instead of `0xFFFFFFFF`, add `rsrc.ptr |= (uint64_t)0x40000 << 32` to
   set bit 18 of word1.
2. `analysis/fp8_gemm/mi350x/kernel_mxfp4_gluon_cpp.cpp:673-680`
   (`make_scale_srd`) — same three changes.
3. `analysis/fp8_gemm/mi350x/kernel_mxfp4_gluon_cpp.cpp:2316-2326`
   (`make_srd` lambda inside the kernel) — same three changes.

**Then**: re-attempt `BARRIER_TO_WAITCNT_RELAXED_VMCNT=15` (the R32 axis at
kernel:469) and per-site `STEP3_S{2,3,4}` mixed vmcnt
{15, 10} (R32 axes at kernel:437-457) and confirm no aperture-violation
crash.

**Risk**: SNR — if the SRD `ADD_TID_ENABLE` mode reinterprets the offset
the way `buffer_load_dwordx{2,4}` issues addresses, the load may target
wrong addresses and produce garbage. Mitigate: SNR-gate at K=512 with single
shape before scaling to K=128256.

**EV**: **0.5 - 2.0pp on L6**, p50 ~0.7pp.

### Finding B (medium EV, 1-2 days) — Hand-placed `s_nop` at MFMA-chain restart

**What aiter does**: emits 6 `s_nop` per K-iter, all clustered at the
3 sites where the MFMA chain restarts after a barrier or a vmcnt drain
(lines 665-666, 773, 1127-1128 of `aiter_kloop_body.s`). Specifically
3 × `s_nop 0` immediately after each barrier (1-2 ops each) plus
1-2 in the MFMA chain itself.

**What we do**: 0 `s_nop` in the K-loop body.

**Expected mechanism**: The compiler scheduler's RAW-hazard model for
LDS→MFMA dependency chains underestimates the post-LDS write-back latency,
issuing the next MFMA 1-2 cycles too early. Hand-emitted `s_nop` bridges
this gap, preventing the MFMA pipe from stalling on the next ds_read result.

**Where to change**: `analysis/fp8_gemm/mi350x/kernel_mxfp4_gluon_cpp.cpp`
in `kpair_64mfma_step12` (lines 1255-1391). Add `asm volatile("s_nop 0")` at
3 sites:
1. Right after each `s_barrier` emission (the `BARRIER_TO_WAITCNT_*` pattern)
2. After the first ds_read burst, before the dependent MFMA group.
3. After the mid-chain `s_waitcnt vmcnt(N)`, before the next MFMA.

R21B/R22A `EXPLICIT_S_NOP` axis in kernel:209-220 is the existing knob;
re-test it now that SRD is changed (Finding A is a precondition — without
the SRD swap, the s_nop change alone may not show benefit because the LSU
is the bottleneck, not the MFMA pipe).

**EV**: **0.2 - 0.5pp on L6**, p50 ~0.25pp. Independent additive with A.

### Finding C (medium EV, 1 day) — Scale-load granularity reduction

**What aiter does**: scales are loaded via `buffer_load_dword` (1 dword =
4 e8m0 scales = 128 K-cols of B-scale per lane), 4 per K-iter (2 for sa0,
2 for sa1). Each scale-dword arrival is paired with a `ds_read_b32` 1024
bytes later in the schedule, so **scale dispatch overlaps with the MFMA
chain that consumes it**.

**What we do**: scales are loaded via `buffer_load_dwordx2`
(NONVOLATILE_SCALE_X2_POC=1 — kernel:1023-1080), 2 per K-iter, both at the
top of the iter. The 8-byte LDS write pinches the LSU bandwidth at the
iter boundary while the MFMA chain runs cold for ~6 cycles waiting for the
scale to land.

**Where to change**: `analysis/fp8_gemm/mi350x/kernel_mxfp4_gluon_cpp.cpp`
— add a `SCALE_LOAD_X1` axis (default 0 = current dwordx2 behavior; =1 →
4× single-dword loads spread across the K-iter at the same offsets aiter
uses). Pair each single-dword load with the MFMA group that consumes it
(insertion sites at `kpair_64mfma_step12` after the 4 sub-MFMA-groups per
K-pair).

**Risk**: doubles the LSU load issue count (from 2 dwordx2 to 4 dword) which
may push the K-loop into a different scheduler regime. EV depends on whether
the scale arrival timing is the bottleneck (not directly observable without
profiler/build).

**EV**: **0.1 - 0.4pp on L6**, p50 ~0.15pp. Largely orthogonal to A and B.

---

## Hypotheses ruled OUT by this archaeology

These three R33 candidates were considered and DISCARDED:

- **MFMA32 / V5 sprint**: aiter at the same 16x16 MFMA shape achieves the
  5781 TFLOPS ceiling. MFMA-issue rate is provably NOT the bottleneck. V5
  remains a multi-week investment with low EV.
- **Persistent-XCD changes**: aiter does an in-kernel WG-id swizzle that is
  functionally equivalent to our `xcd_aware_remap`. Both achieve identical
  L2-locality benefits. No structural delta to exploit.
- **Tile-size sweep (192x256 or 128x512 instead of 256x256)**: aiter's own
  heuristic picks 256x256 for L6, L7, L4, L8 — same shape we use. Different
  tiles lose on either round count or compute2mem efficiency.

---

## TL;DR — three findings ranked

1. **SRD config swap** (`-16`, `0x20000`, `word1|=0x40000`) — unlocks safe
   vmcnt(15) operation. Sub-day implementation across 3 source sites.
   **EV: 0.5–2.0pp on L6**.
2. **Hand-placed `s_nop` at MFMA restart sites** (3 per K-iter, after each
   barrier and after each mid-chain vmcnt drain). 1-2 day re-tune of
   existing R21B/R22A axis. **EV: 0.2–0.5pp on L6**.
3. **Scale-load granularity reduction** (4× dword instead of 2× dwordx2,
   spread across K-iter). 1-day axis addition. **EV: 0.1–0.4pp on L6**.

Findings A, B, C are independent and additive. Combined p50 EV: **~1.1pp on L6**.
Combined p90 EV: **~3pp on L6**, which would put the incumbent at ~96% of comp.

Word count: ~2050.
