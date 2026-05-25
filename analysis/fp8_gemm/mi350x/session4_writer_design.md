# Session 4 — HBM→LDS B-transpose writer 算法设计

**目的**: 把 HBM `B[K][N]` (row-major, K stride = N_global bytes) 写入 LDS 的 `Bs[N][K]` (N-major, identity swizzle, 128 bytes/row, Session 3 验证过的 `st_128x128_n_major` layout)。一个 BLK_K=128 × BLK_N=128 tile = 16384 bytes。

**用途**: 替换 RRR kernel 主循环中 `G::load(Bs[...], B_gl, coord)` 的 6 个 site (prolog ×2 + main prefetch ×2 + epilog ×2)。G::load 的 prefill_swizzled_offsets 假设 HBM↔LDS 同方向 row-major，无法表达转置写入；因此必须自写 writer。

**两条候选实现路径**:
- **Path L (LDS-staging)**: load HBM b128 → ds_write_b128 到 staging LDS (K-major) → sync → ds_read_b128 (N-major 切线) → ds_write_b128 到最终 Bs。简单但需 2× LDS budget + 2× LDS ops。
- **Path P (cross-lane bpermute)**: load HBM b128 → ds_bpermute_b32 × 4 + v_perm_b32 × 3 byte-shuffle in register → ds_write_b128 到最终 Bs。LDS ops 减半，但算法复杂。

Session 4 (本 session) deliverable = **Path L 走完 + Path P 设计文档落档**；Session 4.1 后续 swap 到 Path P。

---

## 1. Lane / 数据布局

### 1.1 输入: HBM row-major
- `B[K=0..127][N=0..127]`, fp8e4m3, K stride = `N_global` bytes (BLK_K × N_global per tile column)。
- 写 probe 时 N_global = 128 (单 tile)，kernel 集成时 N_global = real N。

### 1.2 输出: LDS N-major (Session 3 `st_128x128_n_major`)
- `Bs[N=0..127][K=0..127]`, `byte_offset(n, k) = n*128 + k` (identity swizzle, sizeof T=1)。
- 共 16384 bytes。

### 1.3 8-warp 协作划分
- WG = 512 threads = 8 warps × 64 lanes。
- 16384 bytes / 512 = 32 bytes/thread → 每 thread 2 × ds_write_b128 (16 bytes each)。
- 划分: **每 warp 处理 16 K-rows × 128 N-cols** (warp w 负责 K=w*16..w*16+15)。
  - 每 warp 64 lanes × 32 bytes = 2048 bytes = 16K × 128N。
  - **load 阶段**: 1 lane 1 b128 (16 contiguous N bytes for 1 K row); 64 lanes × 1 b128 = 16K × 64N。所以每 warp 要 2 轮 load 才能覆盖 16K × 128N。
  - **transpose 阶段**: 在 register 内 (Path P) 或经 LDS staging (Path L)。
  - **store 阶段**: 1 lane 1 b128 (16 contiguous K bytes for 1 N col); 64 lanes × 1 b128 = 64N × 16K。所以每 warp 要 2 轮 store 才能覆盖 128N × 16K。

```
warp w 负责 K-strip [w*16, w*16+16):
  for (iter = 0; iter < 2; ++iter) {
    n_block_base = iter * 64;          // 0 或 64
    // LOAD: 1 lane 1 b128 (16 N bytes for K=k_base+l, N=n_block_base+0..63 swept by 64 lanes)
    // wait, 64 lanes × 16 N bytes/lane = 1024 N bytes = 1024/16 = 64 lanes worth of distinct N positions
    // Need restructure: how do 64 lanes cover 16K × 64N = 1024 bytes / lane=16B → 64 lanes works ✓
    //   Assign lane l to (k_off = l & 15, n_off_chunk = l >> 4) → K=k_base + (l&15), N starts at n_block_base + (l>>4)*16
    ... (see §2)
  }
```

---

## 2. Path L (LDS staging) — 本 session 实装路径

### 2.1 阶段

```
Phase 1 (LOAD HBM → register):
  Each lane l (in warp w) issues 1 b128 from HBM:
    K_row = w*16 + (l & 15)               // K ∈ [w*16, w*16+15]
    N_chunk = (l >> 4) & 3                // 0..3
    N_col_start = n_block_base + N_chunk*16
    addr = HBM_base + K_row*N_stride + N_col_start
    reg128 = buffer_load_b128(addr)
  →每 warp 1 round 覆盖 16K × 64N (4 lanes per K row × 4 K rows per quad lane = 16K × 4*16N).
  →每 warp 2 rounds (iter=0,1) 覆盖 16K × 128N.

Phase 2 (REGISTER → LDS staging, K-major):
  Staging LDS: Bs_stage[w][16K rows × 128N cols] = byte_offset(k, n) = k*128 + n.
  Same row-major layout as HBM — each lane writes its own 16 contiguous N bytes to its K row.
  Each lane l: 1 ds_write_b128 to Bs_stage[w][k_off][n_col_start].
  Cost: 1 ds_write_b128/lane/iter = 2 ds_write_b128/lane total per warp.
  Bank conflict: 16 N bytes (4 banks) × 4 lanes same K row in same bank quad. No conflict
  because addresses differ by N_chunk*16 bytes (different banks within a row).

Phase 3 (LDS staging → LDS final, N-major) — TRANSPOSE happens here:
  Each lane reads from staging in N-major access pattern, writes to final Bs N-major.
  Goal: each lane should write 1 b128 = 16 contiguous K bytes for 1 N col → final Bs[N=n_col][K=k_strip..].

  Each warp covers 16K × 128N tile. After transpose, output is 128N rows × 16K cols stripe.
  Output assignment per warp:
    Lane l (in warp w):
      N_row = (l & 63)                        // covers N = 0..63 in iter=0, 64..127 in iter=1
      K_strip = w*16                          // 16 K bytes starting at K=k_strip
      reads from staging: bytes Bs_stage[w][k_off=0..15][N=n_row_block_base + (l & 63)]
      assembles into 16-byte register (gather byte-by-byte from staging)
      writes to final: Bs[N=n_row_block_base + (l & 63)][K=k_strip..k_strip+15]

  PROBLEM: gathering 16 strided bytes from staging is 16 ds_read_b8 per lane (slow).

  ALTERNATIVE: read 4 ds_read_b32 per lane (4 strided dwords, each 4 K bytes for given N col).
  Layout: Bs_stage[w][k=0..15][n_row] needs k strided by 128 bytes, n_row fixed.
  4 ds_read_b32: each reads 4 contiguous K (rows of staging at k, k+1, k+2, k+3 same n_row...)
  Actually that's wrong direction — same n_row, different k = stride 128 bytes (not contiguous).
  → not feasible as 4 ds_read_b32 of 4 bytes each — actually IS feasible, each read is 4 bytes
    but staging only has 4 bytes contiguous at one (k, n_row). 4 ds_read_b32 with stride=128.

  So 4 ds_read_b32/lane (1 per K-row chunk of 4) → assemble 16-byte register → 1 ds_write_b128.

  Total Phase 3: 4 ds_read_b32 + 1 ds_write_b128 per lane per iter (or per output N-row).
  Each warp 2 iter × 1 b128 store = 2 b128 stores/lane.
  But 4 ds_read_b32 reads with stride=128 bytes = same bank! → 4-way bank conflict.

  Mitigation: rearrange staging layout so the 4 strided reads hit different banks.
  E.g., staging K-row stride = 132 bytes (add 4-byte padding per row) → 4 reads access banks
  spread by stride 33 ≠ 0 mod 64 → conflict-free.
```

### 2.2 LDS budget check (160 KB / CU)
- Final Bs (Session 3 layout): 16384 bytes/tile × double-buffer = 32 KB.
- Staging Bs_stage: 16384 bytes/warp × 8 warps = 128 KB ❌ over budget.

→ **Single-staging-buffer 共享**: 8 warps 不能并发写不同 staging。需 serialize 或 fold staging size。

**Fix**: staging 只在 transpose 期间使用，写 → sync → 读 → sync → 释放。在 prolog 阶段 we don't need scratch for other purpose, so reuse As LDS region? 但 As 同时也在 load。Risky。

**Better**: 减小 K-strip to 8 K rows/warp → staging = 8K × 128N × 8 warps = 8192 bytes/warp × 8 = 64 KB. + final 32 KB = 96 KB → still might conflict with As (64 KB) + scratch.

**Cleanest**: 让 8 warps 共享 1 个 staging slot, 流水化 (warp 0 transpose first, then warp 1, ...) → 完全 serialize, 性能差。

**Final choice for Path L**: 用 **single staging slot of 16K * 128N = 16 KB** + 8-warp 协作往这一个 staging 写入并读回 (整 WG 同步)。Workflow:
```
1. All 8 warps load HBM → register (each warp loads its K-strip): 64 b128/warp × 8 = 512 b128/WG
2. All 8 warps write register → staging LDS (K-major, full 128K × 128N tile, 16KB):
   - Each lane writes 1 b128 to (k_row, n_col_start) per iter
3. __syncthreads()
4. All 8 warps read staging LDS → register (N-major access):
   - Each lane reads 4 strided b32 from staging for 1 (k_strip_within_warp, n_col)
5. Each lane writes 1 b128 to final Bs (N-major)
```

Wait — step 2 writes the **entire tile** to staging (overwriting any warp's region with another warp's data is fine because each warp has unique K rows). And step 4 reads from staging anywhere (all 8 warps can read).

→ Single 16KB staging = 1× tile. + final 16KB tile × 2 buf = 32KB. Total Bs region: 16K (staging, shared) + 32K (final, double-buf) = 48 KB. As 64 KB. Total ≈ 112 KB out of 160 KB → OK.

**This is the Path L scheme.**

---

## 3. Path P (cross-lane bpermute) — Session 4.1 实装

### 3.1 16-lane subgroup byte transpose

Within a 16-lane subgroup (e.g. lanes 0..15 of a warp), perform a 16-byte × 16-lane → 16-byte × 16-lane byte transpose.

**Input** (after HBM load): lane `l` (l ∈ 0..15) holds 16 bytes = `B[K=k_base+l][N=n_base+0..n_base+15]`.
- 4 dwords per lane. Dword d = N[4d..4d+3] for K=k_base+l.

**Output**: lane `l` (l ∈ 0..15) holds 16 bytes = `B[K=k_base+0..k_base+15][N=n_base+l]`.
- 4 dwords per lane. Dword d = K[4d..4d+3] for N=n_base+l.

### 3.2 Algorithm (per output dword d on each lane l)

For each output dword d ∈ {0..3}:
  Output dword has 4 bytes = K[4d+0..4d+3] for N=l.
  Source for byte j ∈ {0..3}: input lane (4d+j), at byte position l of that lane's 16 bytes,
    i.e., dword (l/4), byte (l%4).

  4 ds_bpermute_b32 phases (j=0..3):
    r_j = ds_bpermute_b32(value = my_dword(l/4), src_lane = 4d+j)
    // r_j on lane l holds: input lane (4d+j)'s dword (l/4)
    // We need byte (l%4) of this for the output

  Then assemble output dword:
    sel_lo = ((l%4)) | ((l%4)+4)<<8     // byte (l%4) from r_0, byte (l%4) from r_1
    out_lo = perm(r_0, r_1, sel_lo) & 0x0000FFFF  // keep only bytes 0,1
    sel_hi = 0 | 0<<8 | ((l%4))<<16 | ((l%4)+4)<<24
    out_hi = perm(r_2, r_3, sel_hi) & 0xFFFF0000  // keep only bytes 2,3
    output_dword[d] = out_lo | out_hi
  // 3 v_perm_b32 calls per output dword (2 perms + 1 OR; or rewrite as 2 perms + bit-masks)

**Total per 16-lane subgroup**: 4 output dwords × (4 ds_bpermute + 3 v_perm) = 16 ds_bpermute + 12 v_perm per subgroup.

### 3.3 Scaling to 128×128 tile

Per warp processes 16 K-rows × 128 N-cols = 8 sub-groups (each 16K × 16N).
Each lane in the warp is in exactly 1 sub-group at a time.
Within a warp, the 16-lane sub-groups operate **simultaneously** on different N-blocks
(sub-group g handles N=g*16..g*16+15 of the same K-strip).

But a warp has 64 lanes, only 16 are in one sub-group. So a warp actually has 4
sub-groups operating in parallel, each handling different K-rows of the same N-block? Let me re-check.

Actually: 64 lanes in a warp, divided into 4 sub-groups of 16. Each sub-group handles one
16K × 16N byte block. But we want to cover 16K × 128N per warp. So 4 sub-groups can cover
4 × 16N = 64N. Need 2 iterations per warp (iter=0 covers N=0..63, iter=1 N=64..127).

Per sub-group at any iter: handles 16K (= entire warp's K-strip) × 16N. The 4 sub-groups
differ in their N-block.

Setup per lane (l = 0..63 within warp):
  k_local = l & 15           // 0..15, K offset within K-strip
  n_block_g = l >> 4         // 0..3, which N-block (16-N-wide) within iter
  K_row = w*16 + k_local
  N_col_start = n_block_base + n_block_g*16
  // Phase 1: each lane loads 16 N bytes (its sub-group's N-block) from HBM
  load_b128 from HBM[K_row][N_col_start..N_col_start+15]

After load, organized into 4 sub-groups of 16 lanes; each sub-group has 16-lane × 16-byte
input ready for transpose.

Phase 2 (transpose, per sub-group):
  16 ds_bpermute + 12 v_perm produces N-major 16-byte data per lane.

Phase 3 (store):
  Each lane writes 1 ds_write_b128 to final Bs:
    N_row = N_col_start + k_local       // (l & 15 became the N index after transpose)
    K_pos = w*16                        // first K of warp's strip
    final_addr = Bs[N_row*128 + K_pos]
    ds_write_b128(final_addr, my_16_byte_output)

→ 1 ds_write_b128/lane/iter = 2 ds_write_b128/lane/warp = 16/warp = 128/WG total stores.

**Cost per warp per 128×128 tile**:
- Loads: 4 sub-groups × 1 b128/lane × 2 iter = 2 b128 loads/lane (16 lanes per sub-group, 4 sub-groups, 2 iter)
- Wait: per iter, all 64 lanes do 1 b128 load = 64 b128/warp/iter. × 2 iter = 128 b128/warp.
  (vs Path L: 128 b128 loads + 128 ds_write to staging + 64×4 ds_read from staging + 128 ds_write final.
   Path P: 128 b128 loads + 128 ds_write final + transpose in register.)
- Stores: 64 b128/warp/iter × 2 iter = 128 b128 ds_write/warp.
- Transpose: 4 sub-groups × (16 bpermute + 12 v_perm) × 2 iter = 224 instructions/warp.

vs G::load (row-major, no transpose): 128 b128 loads + 128 b128 stores = 256 LDS ops.

Path P overhead vs G::load: +224 inline-asm instructions per warp per tile. With 8 warps,
this is ~1800 extra instructions per WG per tile. Per MFMA-K-iter we do 1 tile load, so
roughly +200 inst per K-iter — likely fine relative to 32 MFMAs per K-iter.

### 3.4 Bank conflict analysis (Path P, final Bs N-major writes)

Each lane writes 1 ds_write_b128 to Bs[N_row][K_pos], where N_row = N_col_start + (l&15)
and K_pos = w*16 (warp-uniform).

Within 1 sub-group (16 lanes): the 16 N_rows are consecutive (differ by 1), so the 16 b128
writes target N=0..15 (within the sub-group's N-block) at the same K_pos.

LDS layout: byte_offset(N, K) = N*128 + K. Bank (gfx950 64 banks of 4 bytes) = (offset/4) & 63.
For 16 lanes writing b128 (each writes 4 banks): lane l writes banks
((N_row*128 + K_pos)/4 + 0..3) & 63 = (N_row*32 + K_pos/4 + 0..3) & 63.

Since N_rows are 16 consecutive integers: bank addresses differ by 32 mod 64.
Lane l banks: (l*32 + base) & 63 for the 1st of 4 b128 byte-lanes.
- Lane 0: banks 0,1,2,3
- Lane 1: banks 32,33,34,35
- Lane 2: banks 0,1,2,3 (lane 2's first bank = 64 mod 64 = 0)
- Lane 3: banks 32,33,34,35
- ...

→ **2-way bank conflict** (lanes 0/2/4/.../14 share banks; 1/3/.../15 share banks).

Mitigation: change Bs layout to add 4-byte (or other) padding per N-row, or use XOR swizzle on
Bs to break the 32-stride pattern. The Session 3 ST already has identity swizzle; switching to
a swizzled variant (e.g., `((offset>>7) & 7) << 4` XOR mirror of st_16x128_v2) might fix.

**Defer to Session 4.2**: swizzled-layout `st_128x128_n_major_v2` + update Session 3 load
specialization to undo the swizzle on read side.

### 3.5 Path L bank conflict

Path L Phase 4 reads 4 ds_read_b32/lane with K-stride 128 bytes. Address pattern per lane:
(k=0..3) × 128 + n_row. 4 reads from same N row at different K positions. Banks:
((k*128 + n_row)/4) & 63 = ((k*32 + n_row/4)) & 63.

For 4 reads from same lane (k=0..3): banks differ by 32 mod 64 = all same! → **4-way conflict** per lane.
But cross-lane: 16 lanes in sub-group access 16 different N rows simultaneously, so all 64 lanes
hit different N positions. Actual conflict on serialized lane-internal reads.

→ Both paths have bank conflicts. Both need Session 4.2 mitigation.

---

## 4. Session 4 / 4.1 / 4.2 split decision

| Item                                        | Session 4 | Session 4.1 | Session 4.2 |
|---------------------------------------------|-----------|-------------|-------------|
| Design doc (this file)                      | ✅        |             |             |
| 16-lane primitive byte-transpose probe      | ✅ (Path L) |          |             |
| Full 128×128 writer probe (correctness)     | ✅ (Path L) |          |             |
| Round-trip vs Session 3 load (mismatch=0)   | ✅        |             |             |
| Path P implementation (cross-lane bpermute) |           | ✅          |             |
| Bank-conflict-free LDS layout               |           |             | ✅          |
| rocprof bank-conflict counter measurement   |           |             | ✅          |
| Writer microbenchmark vs G::load            |           |             | ✅          |
| Kernel integration (Session 5)              |           |             |             |

**Justification for the split**:
- Path L correctness primitive is the **必要前置** for Session 5 integration: without a working
  writer, integration cannot start. Path L is simple and verifiable end-to-end in 1 session.
- Path P is **perf optimization** of an already-correct writer. It can ride on Path L's probe
  harness (same round-trip verifier) and be measured against Path L directly.
- Bank-conflict-free layout requires Session 3 ST/load update plus rocprof on chi2811 — both
  multi-step. Defer to its own session.

This split respects the "拆细自交付" principle: Session 4 = "writer works correctly", Session
4.1 = "writer fast (Path P)", Session 4.2 = "writer optimal (no bank conflict)".

---

## 5. Path L probe harness (`tests/probes/rrr_b_writer_probe.cu`)

```
constexpr int K_DIM = 128, N_DIM = 128;
constexpr int TILE_BYTES = K_DIM * N_DIM;   // 16384

__global__ void __launch_bounds__(512, 1)
b_writer_probe(uint8_t* hbm_b, uint8_t* dst_lds_dump) {
    __shared__ __align__(16) uint8_t Bs_stage[TILE_BYTES];        // 16 KB staging
    __shared__ __align__(16) uint8_t Bs_final[TILE_BYTES];        // 16 KB final N-major

    int tid = threadIdx.x;
    int warp_id = tid >> 6;          // 0..7
    int lane_id = tid & 63;          // 0..63

    // PHASE 1+2: load HBM b128 → staging LDS K-major
    //   warp w covers K = w*16..w*16+15 (16 K rows).
    //   2 iter × 4 sub-groups × 16 lanes = 128 b128 per warp = 64 per lane / 2 iter.
    //   Per lane per iter: 1 b128 load + 1 b128 store.
    for (int iter = 0; iter < 2; ++iter) {
        int n_block_base = iter * 64;     // 0 or 64
        int k_local = lane_id & 15;       // 0..15
        int n_chunk = (lane_id >> 4) & 3; // 0..3, 4 N-blocks of 16 cols
        int K_row = warp_id * 16 + k_local;
        int N_col_start = n_block_base + n_chunk * 16;
        size_t hbm_off = K_row * N_DIM + N_col_start;
        size_t lds_off = K_row * N_DIM + N_col_start;  // staging K-major same as HBM
        __uint128_t v = *reinterpret_cast<__uint128_t*>(&hbm_b[hbm_off]);
        *reinterpret_cast<__uint128_t*>(&Bs_stage[lds_off]) = v;
    }
    __syncthreads();

    // PHASE 3+4: staging LDS → final LDS N-major via 4 ds_read_b32/lane + 1 ds_write_b128/lane
    //   warp w produces final Bs[N_row][K = w*16..w*16+15].
    //   2 iter × 64 lanes = 128 N-rows per warp (but warp only handles 16 K, so per warp
    //   actually produces 128 N × 16 K = 2048 bytes). Distributed: 1 lane = 1 N-row per iter.
    for (int iter = 0; iter < 2; ++iter) {
        int n_block_base = iter * 64;
        int N_row = n_block_base + lane_id;             // covers N = 0..127 over 2 iter × 64 lanes
        int K_strip = warp_id * 16;
        // 4 ds_read_b32: gather 4 dwords each = 4 bytes from staging at (K=K_strip+4i..K_strip+4i+3, N=N_row)
        // Wait: 4 contiguous K bytes for fixed N — staging is K-major so K stride = N_DIM bytes.
        //       contiguous K bytes would be at different N positions in staging. We want 4 K x 1 N.
        //
        // Layout: staging[k][n] = staging[k*N_DIM + n]. For fixed n=N_row, varying k=0..15:
        //   addresses are N_row, N_row+128, N_row+256, ..., N_row+15*128.
        // These are STRIDED by 128 = bank-collision pattern.
        //
        // 4 ds_read_b32 (each 4 bytes contiguous K): NO, K is strided so can't do b32 of 4 K bytes.
        // 16 ds_read_b8 (each 1 byte at K=k, N=N_row): yes, but slow.
        //
        // Alternative: pack 4 K bytes into 1 dword via per-byte ds_read_b8 + shift+OR.
        // Total: 16 ds_read_b8 + bit-packing → 4 dwords assembled.
        uint8_t out16[16];
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            out16[k] = Bs_stage[(K_strip + k) * N_DIM + N_row];
        }
        // 1 ds_write_b128 to final
        size_t final_off = N_row * K_DIM + K_strip;
        *reinterpret_cast<__uint128_t*>(&Bs_final[final_off]) = *reinterpret_cast<__uint128_t*>(out16);
    }
    __syncthreads();

    // Dump final to HBM for host verification.
    for (int i = tid; i < TILE_BYTES; i += 512) {
        dst_lds_dump[i] = Bs_final[i];
    }
}
```

**Verification**: host produces reference `ref[n*128 + k] = hbm_b[k*128 + n]`, compares to
`dst_lds_dump`. Pass criterion: mismatch == 0 over 16384 bytes.

**Note on Path L scatter**: 16 ds_read_b8/lane + bit-packing is slow (≈64 LDS ops/lane). Path P
will replace this with 16 ds_bpermute_b32/lane sub-group (much better instruction count
per byte moved). Session 4 doesn't optimize this — correctness only.

---

## 6. Open questions / Session 4.1+ TODOs

1. Path P 16-lane sub-group transpose: validate the 16 bpermute + 12 v_perm count empirically
   (compiler may DCE / coalesce).
2. Path L scatter (16 ds_read_b8) is acceptable for probe but unacceptable for kernel; Session 4.1
   must replace with Path P.
3. Bank conflict mitigation: try `st_128x128_n_major_v2` with XOR swizzle `((offset>>7) & 7) << 4`,
   update Session 3 `load` specialization to mirror XOR. Session 4.2.
4. LDS budget audit: As 64 KB + staging 16 KB + final 32 KB = 112 KB. Need to confirm against
   kernel's existing budget (scratch + other tiles).
5. Kernel integration: 6 `G::load(Bs[...])` sites + 6+ `load_b` sites. Gate by macro
   `RRR_B_PRETRANS`. Session 5.

