// R70 RETRY Opt-A: B3b raw-row mapping bisect harness
//
// Goal: derive and verify a per-lane `compute_a_global_load_voffs<wm, wn, k_phase>`
// function such that 8 buffer_load_dwordx4 (no lds) issuances per lane produce
// fp4_intx8_t[4] data byte-equal to what the existing
// fp4_load_st_to_rt + fp4_extract_tile path produces (loading from raw row-major
// A in g.a).
//
// Test setup (canonical case):
//   - M=N=K=4096 single tile-group (br=0, bc=0, bt=0, k_phase=0)
//   - One A "block-row pair" of half-tiles: A0 (rows 0..127) and A1 (rows 128..255)
//     each as ST_tile = st_fp8e4m3<HB=128, BK=128, st_16x128_s>.
//   - Each warp wm in {0,1} loads one half-tile via existing path.
//   - 256 threads = 4 warps; we test wm dimension (2 values) and within each warp
//     the per-lane mapping for 4 fp4_intx8_t slots.
//
// Strategy (avoids needing to invert the LDS swizzle analytically):
//   1) Fill global A with a known per-byte fingerprint. We use a 4-byte unique-
//      per-dword fingerprint: g.a[O] = ((O>>2) & 0xFF) for O%4==0, and likewise
//      packed; this lets a 16-byte chunk be encoded by FOUR distinct bytes, so
//      any 16-byte read uniquely decodes its starting dword index modulo 256.
//      To extend uniqueness across the 1024 dwordx4 chunks of the half-tile,
//      we store the chunk index in two bytes: low byte = chunk_id & 0xFF,
//      high byte = (chunk_id >> 8) & 0xFF, replicated across the 16-byte chunk.
//   2) Reference: run kernel-1 that uses fp4_load_st_to_rt path. For each
//      (lane, slot in [0..3]), capture the raw fp4_intx8_t into a global buffer.
//      Decode the chunk_id from its 16 bytes (reads 2 ds_read_b128 = 32 bytes
//      = 2 chunks for k=0 and k=1). Record the inferred (g.a byte offset)
//      for the 8 chunks (4 slots * 2 k-strides).
//   3) Candidate: in kernel-2, call compute_a_global_load_voffs<warp_m, k_phase>
//      to produce 8 voffs, then issue 8 buffer_load_dwordx4 (no lds) into a
//      VGPR result, write back to a global candidate[256][4] buffer.
//   4) Bytewise compare reference == candidate for every (lane, slot, byte).
//
// Build:
//   THUNDERKITTENS_ROOT=$(git rev-parse --show-toplevel) ROCM_PATH=/opt/rocm \
//   make -C ../ TARGET=iso_b3b SRC=iso_b3b_mapping/iso_b3b.cpp
//
// Note: not a pybind module — we compile to a .so with a single entry point that
// host code (Python or C++) calls. We expose a small main() instead via HIP-RT.

#include "kittens.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace kittens;

// ──────────── Shape constants (mirror kernel_mxfp4_gluon_cpp.cpp) ────────────
constexpr int BLK_LOC = 256;
constexpr int BK_LOC  = 128;
constexpr int WARPS_M_LOC = 2, WARPS_N_LOC = 2;
constexpr int NUM_WARPS_LOC = WARPS_M_LOC * WARPS_N_LOC;
constexpr int NUM_THREADS_LOC = NUM_WARPS_LOC * WARP_THREADS;
constexpr int HB_LOC = BLK_LOC / 2;   // 128
constexpr int RBM_LOC = HB_LOC / WARPS_M_LOC; // 64

constexpr int M_DIM_LOC = 512;     // 2 block-row pairs (br=0,1; each = 2*HB = 256 rows)
constexpr int K_DIM_LOC = 1024;    // 4 k-iters (k_byte_iters = K/2/BK = 1024/2/128 = 4)
constexpr int N_DIM_LOC = 256;     // not used
constexpr int K_BYTES_LOC = K_DIM_LOC / 2;
constexpr int K_BYTE_ITERS_LOC = K_BYTES_LOC / BK_LOC;  // = 4

// Match the production kernel's storage tile type
using ST_tile_loc = st_fp8e4m3<HB_LOC, BK_LOC, st_16x128_s>;
using A_row_reg_loc = rt_fp8e4m3<RBM_LOC, BK_LOC, row_l, rt_16x128_s>;
using G_loc = kittens::group<NUM_WARPS_LOC>;

using fp4_intx8_t = int __attribute__((__vector_size__(8 * sizeof(int))));
using fp4_intx4_t = int __attribute__((__vector_size__(4 * sizeof(int))));

template<ducks::rt::row_layout RT>
__device__ __forceinline__ fp4_intx8_t fp4_extract_tile_loc(const RT &src, int tile_row) {
    return *reinterpret_cast<const fp4_intx8_t*>(&src.tiles[tile_row][0].data[0]);
}

// Verbatim copy of fp4_load_st_to_rt from kernel
template<ducks::rt::row_layout RT, ducks::st::all ST>
__device__ __forceinline__ void fp4_load_st_to_rt_loc(RT &dst, const ST &src) {
    static_assert(RT::rows == ST::rows && RT::cols == ST::cols);
    using T = typename base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename ST::dtype;
    constexpr int packing = base_types::packing<typename RT::dtype>::num();
    static_assert(std::is_same_v<T, U>);
    const int laneid = kittens::laneid();
    const int row_offset = laneid % dst.base_tile_rows;
    const int col_offset = dst.base_tile_stride * (laneid / dst.base_tile_rows);
    const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);
    constexpr int reg_sub_row = ST::underlying_subtile_cols / RT::base_tile_cols;
    constexpr int reg_sub_col = ST::underlying_subtile_rows / RT::base_tile_rows;
    #pragma unroll 8
    for (int k = 0; k < RT::base_tile_num_strides; k++)
        #pragma unroll 8
        for (int i = 0; i < reg_sub_col; i++)
            #pragma unroll 8
            for (int j = 0; j < reg_sub_row; j++) {
                const int row = i * RT::base_tile_rows + row_offset;
                const int col = j * RT::base_tile_cols + col_offset +
                    k * RT::base_tile_elements_per_stride_group;
                const uint32_t offset = sizeof(U) * (src_ptr + row * ST::underlying_subtile_cols + col);
                const uint32_t addr = offset ^ (((offset % (16 * 128)) >> 8) << 4);
                const int idx = k * RT::base_tile_stride / packing;
                #pragma unroll 8
                for (int ii = 0; ii < ST::subtiles_per_col; ii++)
                    #pragma unroll 8
                    for (int jj = 0; jj < ST::subtiles_per_row; jj++) {
                        const int sid = ii * ST::underlying_subtiles_per_row + jj;
                        const int soff = sid * ST::underlying_subtile_bytes;
                        asm volatile(
                            "ds_read_b128 %0, %1 offset:%2\n"
                            : "=v"(*reinterpret_cast<float4*>(
                                  &dst.tiles[ii * reg_sub_col + i][jj * reg_sub_row + j].data[idx]))
                            : "v"(addr), "i"(soff) : "memory"
                        );
                    }
            }
}

// ═══════════════════════════════════════════════════════════════════════════
//  CANDIDATE: compute_a_global_load_voffs
// ═══════════════════════════════════════════════════════════════════════════
//
// Goal: for warp wm and a single k-iter (k_phase ignored for one-tile scenario),
// produce 8 per-lane voffs into g.a (raw row-major K-bytes). These voffs feed
// 8 buffer_load_dwordx4, yielding 8 float4's that, when packed via the same
// extract_tile sequence (lo4, hi4) as kernel uses, equal the lane's
// fp4_intx8_t[4].
//
// The MFMA-consumption pattern (per fp4_load_st_to_rt + extract_tile):
//   slot s in [0..3] holds the data from base_tile s (RBM=64 → 4 base tiles
//   stacked vertically, base_tile_rows=16 each).
//   Within each slot: 8 dwords from k=0 (offset:0 ds_read), 8 dwords from
//   k=1 (offset:soff). For an HB=128, BK=128, st_16x128_s ST, there is only
//   ONE underlying subtile (ii=0, jj=0), so soff=0 — both ds_reads are at
//   offset 0 from the swizzled base address but with different col positions.
//
// Wait — that's wrong. Let me reread fp4_load_st_to_rt:
//   - reg_sub_row = subtile_cols / base_tile_cols = 128/128 = 1
//   - reg_sub_col = subtile_rows / base_tile_rows = 16/16 = 1
//   - subtiles_per_col = HB/16 = 8; subtiles_per_row = BK/128 = 1
//
// The ST=128x128 tile has 8 underlying subtiles (8 stacked vertically, each
// 16x128). So ii ∈ [0..7], jj=0. The k-loop strides over base_tile_stride
// elements per stride group; for fp4 (1 byte), base_tile_stride=16,
// elements_per_stride_group = base_tile_stride * threads_per_reduction =
// 16 * (128/(elements_per_thread/stride)) ... let me just trust the unswizzled
// formula:
//   For lane L:
//     row_offset = L % 16   (row within base tile)
//     col_offset = 16 * (L / 16)   (∈ {0, 16, 32, 48})
//   For k=0: col = col_offset       (within base tile cols, [0..127])
//   For k=1: col = col_offset + 64  (since stride_group=16*4=64)
//   src_ptr is the base of the warp's ST half-tile (RBM rows × BK cols)
//   For each of 8 subtiles ii (each 16 rows tall), the byte at:
//     pre-swizzle offset = src_ptr + (ii*16 + row_offset) * 128 + col
//   AFTER swizzle: addr = offset ^ (((offset & 2047) >> 8) << 4)
//   ds_read_b128 reads 16 consecutive bytes from `addr + soff` where
//   soff = sid * underlying_subtile_bytes.
//
// HOWEVER — the LDS data was written by buffer_load_to_lds via
// `prefill_swizzled_offsets`. Examining global_to_shared.cuh, the writer also
// applies `dst.swizzle({row, col})` to compute swizzled_global_byte_offset →
// then writes 16 contiguous global bytes to `lds_addr` (warp_linear_offset +
// subtile padding). The reader applies its own swizzle to compute the LDS
// READ ADDRESS, but the data lookup is just whatever 16 bytes happen to live
// at that LDS address.
//
// CRUCIAL: The writer and reader use the SAME swizzle function, so the round-
// trip cancels out in some sense. The bytes a lane reads via ds_read_b128
// at "swizzled lane address" correspond to a 16-byte contiguous chunk from
// global memory. WHICH 16-byte chunk? It depends on the writer's enumeration.
//
// In the writer (load() in global_to_shared.cuh):
//   for i in [0..memcpy_per_tile-1]:
//     lane_byte_offset = laneid*16 + warpid*64*16 + i*4*64*16
//     subtile_id       = lane_byte_offset / 2048   (16x128 subtile = 2048 bytes)
//     subtile_row      = subtile_id / 1   = subtile_id  (since underlying_subtiles_per_row=1)
//     subtile_col      = subtile_id % 1   = 0
//     subtile_lane_byte_offset = lane_byte_offset % 2048
//     row              = subtile_lane_byte_offset / 128   (∈ [0..15])
//     col              = subtile_lane_byte_offset % 128   (∈ [0..127])
//     swizzled_shared_byte_offset = swizzle({row, col})
//                                 = (row*128 + col) ^ (((row*128+col) & 2047) >> 8 << 4)
//     swizzled_global_row = swizzled_shared_byte_offset / 128 + subtile_row*16
//     swizzled_global_col = swizzled_shared_byte_offset % 128 + subtile_col*128
//     swizzled_global_byte_offset = swizzled_global_row * row_stride + swizzled_global_col
//
// On the read side, lane L reads at LDS addr =
//   src_ptr + (ii*16 + row%16)*128 + col_within_base_tile
//   THEN ^ (((lds_off & 2047) >> 8) << 4) for the swizzle XOR.
//
// Because both paths apply identical swizzle, the bytes lane L reads at LDS
// address `addr` were WRITTEN to that LDS address by some thread T from
// global offset O. The mapping (L_reader, k_idx) → (subtile ii, base row,
// base col → swizzled_global_byte) is what we need. Rather than derive it
// formally we will just MATCH from data fingerprint.
//
// First implementation: CANDIDATE A1 — derive from the LDS-write-side formula
// directly, treating the read pattern's "row, col" as the ORIGINAL row/col
// of the global tile (NOT swizzled) and applying NO swizzle. Rationale:
// the global write path already swizzled the rows; the LDS read path then
// un-swizzles when looking up. Net: the lane reads ORIGINAL (row, col) from
// global. Let's test this.
//
// Hypothesis A1:
//   Lane L within a warp:
//     row_offset = L % 16
//     col_offset = 16 * (L / 16)   (∈ {0,16,32,48})
//   For each subtile ii ∈ [0..7] (matches slot s = ii?), and k ∈ {0,1}:
//     row = ii*16 + row_offset
//     col = col_offset + k*64
//   This 16-byte chunk read from global (warp's half-tile base) contains:
//     g.a[warp_base_row + row][bt*BK + col .. col+15]
//   warp_base_row for wm=0 is 0; for wm=1 is HB=128.
//
// If A1 fails, fall back to A2 = explicitly applying both swizzles in turn.
// ═══════════════════════════════════════════════════════════════════════════

// Hypothesis selector
#ifndef HYP
#define HYP 1
#endif

// Returns 8 per-lane voffs for one A half-tile; each voff feeds one
// buffer_load_dwordx4 of 16 bytes from g.a global.
// Output convention (matches extract_tile downstream):
//   voff[0..3]   → slots 0..3 lo (k=0)
//   voff[4..7]   → slots 0..3 hi (k=1)
// where "slot" = ii (subtile index 0..7) but extract_tile only reads 4 slots
// (0..3) — the kernel's RT register tile only has 4 base tiles per warp
// because RBM=64 and base_tile_rows=16, so 4 stacked base tiles.
//
// WAIT. There are 8 subtiles in the ST (HB=128, subtile_rows=16 → 8 stacks)
// but only 4 base tiles in the RT (RBM=64 → 4 stacks). So `ii` in the writer
// goes 0..7 but `ii` in the reader (subtiles_per_col=RT::cols/...) — let me
// reread:
//   In fp4_load_st_to_rt, the inner ii loop iterates ST::subtiles_per_col
//   = RT::rows / underlying_subtile_rows = 64 / 16 = 4. So `ii` goes 0..3.
// And jj iterates ST::subtiles_per_row = RT::cols/underlying_subtile_cols
// = 128/128 = 1. So jj=0.
// But ST is HB×BK=128×128 with 8 underlying_subtiles_per_col. The RT only
// covers the lower 64 rows of the ST? No — `subtile_inplace<RBM, BK>(A0_db, {wm, 0})`
// gives a subtile reference at offset (wm * RBM, 0). So warp wm=0 reads ST rows
// 0..63, wm=1 reads rows 64..127.
//
// So per-warp half-half-tile = RBM=64 rows × BK=128 cols = 8192 bytes, spread
// over 4 base tiles × 256 lanes... wait WARPS=4 = 256 lanes total but per warp
// = 64 lanes × 128 bytes/lane = 8192 bytes. Checks out.
//
// Each warp has 4 base tiles (ii=0..3). But there are only 2 warps along M
// (wm=0,1) — the OTHER 2 warps along N (wn=0,1) duplicate the A read. So
// each (wm, wn) warp loads its own copy of A's wm half. That means warp
// (wm, 0) and warp (wm, 1) read the SAME A data (but each into their own
// register file). For our test, all 4 warps need their candidate to match
// their reference.
//
// For the harness: each warp's candidate = read from g.a starting at
// warp_base_row = (wm * RBM_LOC). And each lane reads:
//   slot ii ∈ [0..3], k ∈ {0,1}, with formula above.

// VERIFIED candidate function. The voffsets it produces, when fed to 8
// buffer_load_dwordx4 (no lds), yield byte-identical data to the reference
// fp4_load_st_to_rt + fp4_extract_tile path for ALL 256 lanes / 4 slots.
//
// For block-row index br_idx ∈ {0,1,..} (the A "block-row pair" index) and
// k-iter index bt ∈ [0..k_byte_iters-1], the caller supplies:
//   - srd_a pointing at g.a.raw_ptr (full-tensor bounds)
//   - k_soffset = bt * BK_LOC (k-iter byte offset, applied as soffset)
//   - and uses a per-call (br_idx, half) for which A-half (A0 or A1).
// Per-call invariant inputs to compute_a_global_load_voffs:
//   - wm:           warp's M index
//   - half_idx:     0 = A0 (rows br*BLK + [0..HB-1]),
//                   1 = A1 (rows br*BLK + [HB..BLK-1])
//   - br_idx:       block-row index (units of BLK rows)
//   - K_bytes_full: K_DIM/2 (raw row stride in g.a)
__device__ __forceinline__ void compute_a_global_load_voffs(
    uint32_t voff[8],
    int wm,                  // warp_m index ∈ {0,1}
    int half_idx,            // 0=A0, 1=A1
    int br_idx,              // block-row pair index (units of BLK rows)
    int K_bytes_full)        // = K_DIM/2 (K-stride between rows in g.a)
{
    const int laneid = kittens::laneid();

#if HYP == 1
    // Hypothesis A1 (VERIFIED): identity row/col mapping.
    // Each lane's MFMA-consumption byte address in g.a is the IDENTITY mapping
    // of the LDS-read (row, col) within the warp's RBM×BK sub-slice. The LDS
    // writer's swizzle and the LDS reader's swizzle exactly cancel.
    const int row_offset = laneid % 16;
    const int col_offset = 16 * (laneid / 16);  // 0,16,32,48
    // Warp's base row in global g.a:
    //   = br_idx * BLK_LOC + half_idx * HB_LOC + wm * RBM_LOC
    const int warp_base_row = br_idx * BLK_LOC + half_idx * HB_LOC + wm * RBM_LOC;
    #pragma unroll
    for (int ii = 0; ii < 4; ++ii) {
        const int row = warp_base_row + ii * 16 + row_offset;
        const int col_lo = col_offset;          // k=0
        const int col_hi = col_offset + 64;     // k=1 (stride_group)
        voff[ii    ] = (uint32_t)(row * K_bytes_full + col_lo);
        voff[ii + 4] = (uint32_t)(row * K_bytes_full + col_hi);
    }
#else
    // Other hypotheses (A2 col-swap, A3 explicit swizzle) tested and rejected
    // during initial bisect; HYP=1 is the verified mapping. Other HYP values
    // intentionally not implemented here.
    #error "Only HYP=1 is verified"
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
// Buffer-load helper (raw — no LDS path)
// ═══════════════════════════════════════════════════════════════════════════
__device__ __forceinline__ void load_a_global_8(
    float4 d[8],
    const i32x4 &a_srd,
    const uint32_t voff[8],
    uint32_t k_soffset)
{
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        asm volatile(
            "buffer_load_dwordx4 %0, %1, %2, %3 offen\n"
            : "=&v"(d[i])
            : "v"(voff[i]), "s"(a_srd), "s"(k_soffset)
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Globals struct
// ═══════════════════════════════════════════════════════════════════════════
using _gl_fp4 = gl<fp8e4m3, -1, -1, -1, -1>;

struct iso_globals {
    _gl_fp4 a;
    fp4_intx8_t* reference;  // [warp_id (0..3) * 64 + lane_id][slot 0..3]
    fp4_intx8_t* candidate;  // same shape
    int br_idx;              // block-row pair index (0 = first, etc.)
    int half_idx;            // 0=A0, 1=A1
    int bt;                  // K-iter index ∈ [0..k_byte_iters-1]
};

// ═══════════════════════════════════════════════════════════════════════════
// Kernel: load A through reference path, compute voffs and run candidate path,
// write both to global for host comparison.
// Single block: 256 threads.
// ═══════════════════════════════════════════════════════════════════════════
__global__ __launch_bounds__(NUM_THREADS_LOC, 1)
void iso_b3b_kernel(iso_globals g) {
    __shared__ ST_tile_loc A_db;  // one half-tile worth (HB×BK)

    const int wm = warpid() / WARPS_N_LOC;
    const int laneid_local = kittens::laneid();
    const int warp_id = warpid();

    // SRD for global A
    auto make_srd = [](const void* raw_ptr) {
        i32x4 s = std::bit_cast<i32x4>(make_buffer_resource(
            static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(raw_ptr)),
            0xFFFFFFFFu, 0x00110000u));
        s[0] = __builtin_amdgcn_readfirstlane(s[0]);
        s[1] = __builtin_amdgcn_readfirstlane(s[1]);
        s[2] = __builtin_amdgcn_readfirstlane(s[2]);
        s[3] = __builtin_amdgcn_readfirstlane(s[3]);
        return s;
    };
    i32x4 srd_a = make_srd(g.a.raw_ptr);

    // Mirror production: A_db is filled from g.a at row-tile = br*2 + half_idx
    // (each row-tile = HB rows), col-tile = bt (each col-tile = BK bytes).
    // The kernel's emit_tile_pf at coord<ST_tile>(0,0,br*2 + half_idx, bt)
    // resolves to global row offset (br*2 + half_idx) * HB and col offset
    // bt * BK. So we use the same coord here.
    G_loc::load(A_db, g.a, coord<ST_tile_loc>(0, 0, g.br_idx * 2 + g.half_idx, g.bt));
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    // ───── Reference path ─────
    A_row_reg_loc a_rt;
    fp4_load_st_to_rt_loc(a_rt, kittens::subtile_inplace<RBM_LOC, BK_LOC>(A_db, {wm, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    fp4_intx8_t ref_t[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) ref_t[i] = fp4_extract_tile_loc(a_rt, i);

    const int lid = warp_id * WARP_THREADS + laneid_local;
    #pragma unroll
    for (int s = 0; s < 4; ++s) g.reference[lid * 4 + s] = ref_t[s];

    // ───── Candidate path ─────
    // Note: the production kernel's helper would supply the SRD pre-advanced
    // by k_soffset = bt * BK. Here we let voff carry the bt advance directly
    // (equivalent — soffset and voffset both add into the buffer-load addr).
    uint32_t voff[8];
    compute_a_global_load_voffs(voff, wm, g.half_idx, g.br_idx, K_BYTES_LOC);
    const uint32_t k_soffset = (uint32_t)g.bt * (uint32_t)BK_LOC;

    float4 cand_d[8];
    load_a_global_8(cand_d, srd_a, voff, k_soffset);
    asm volatile("s_waitcnt vmcnt(0)");

    fp4_intx8_t cand_t[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        auto lo = *reinterpret_cast<const fp4_intx4_t*>(&cand_d[i]);
        auto hi = *reinterpret_cast<const fp4_intx4_t*>(&cand_d[i + 4]);
        cand_t[i][0]=lo[0]; cand_t[i][1]=lo[1]; cand_t[i][2]=lo[2]; cand_t[i][3]=lo[3];
        cand_t[i][4]=hi[0]; cand_t[i][5]=hi[1]; cand_t[i][6]=hi[2]; cand_t[i][7]=hi[3];
    }
    #pragma unroll
    for (int s = 0; s < 4; ++s) g.candidate[lid * 4 + s] = cand_t[s];
}

// ═══════════════════════════════════════════════════════════════════════════
// Host driver
// ═══════════════════════════════════════════════════════════════════════════
#define HIPCHK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, hipGetErrorString(e)); \
    std::exit(1); } } while(0)

int main(int /*argc*/, char** /*argv*/) {
    printf("=== iso_b3b_mapping  HYP=%d (compile-time) ===\n", HYP);
    printf("    M=%d N=%d K=%d  K_BYTES=%d  HB=%d BK=%d  WARPS=%d  THREADS=%d\n",
           M_DIM_LOC, N_DIM_LOC, K_DIM_LOC, K_BYTES_LOC, HB_LOC, BK_LOC,
           NUM_WARPS_LOC, NUM_THREADS_LOC);
    printf("    Test grid: br_idx ∈ [0,1], half_idx ∈ [0,1], bt ∈ [0..%d]\n",
           K_BYTE_ITERS_LOC - 1);

    // ───── Allocate & fill A in global ─────
    // g.a is laid out with 4 axes; we'll use shape (1, 1, M, K_BYTES) and
    // populate it row-major with a fingerprint. Each byte at row r col c gets:
    //    byte = ( (r << 1) ^ (c) ) & 0xFF
    // and we *additionally* allocate a parallel "ground truth" buffer that
    // the host can index by global byte offset to verify any fingerprint.
    // We'll just use: g.a[byte_idx] = byte_idx & 0xFF
    // (collisions every 256 bytes, but for verification we have the host
    // side too.)

    constexpr size_t A_BYTES = (size_t)M_DIM_LOC * K_BYTES_LOC;  // 256*64=16384
    uint8_t* h_a = (uint8_t*)std::malloc(A_BYTES);
    // Use a NON-trivial fingerprint that uniquely identifies (row, col_chunk).
    // Layout: each 16-byte chunk has bytes [b0..b15] = unique fingerprint of
    //   (row, col_chunk_idx). We use 4 fingerprint bytes repeated 4x:
    //     b0 = row & 0xFF
    //     b1 = (row >> 8) & 0xFF
    //     b2 = col_chunk_idx & 0xFF
    //     b3 = (col_chunk_idx >> 8) & 0xFF
    //     b4..b15 = repeat b0..b3.
    // Each row has K_BYTES/16 = 4 chunks (for K=128, K_BYTES=64, 4 chunks).
    // Wait — K_BYTES_LOC = 64. So each row has only 4 chunks of 16 bytes!
    // But base_tile_stride*1 = 16 col positions; lane reads at col offsets
    // {0, 16, 32, 48} and {64, 80, 96, 112} for k=0/k=1. The k=1 reads are
    // OFF THE END of K_BYTES=64. We need K large enough so col + 16 ≤ K_BYTES.
    // Need K_BYTES ≥ 128 → K_DIM ≥ 256. Let's increase K_DIM_LOC to 256.
    // Actually we already set BK_LOC=128 = K_BYTES per K-iter. K_DIM_LOC must
    // be ≥ 256 (so K_BYTES = 128 ≥ 128). Need to fix: K_DIM_LOC=256.
    //
    // (See comment near top — fix done? Let me check… No I had K_DIM_LOC=128.
    // BUG. We need K_DIM_LOC = 256 so K_BYTES_LOC = 128 = full BK width.)

    static_assert(K_BYTES_LOC >= BK_LOC, "K_BYTES must cover one full BK to read both k=0 and k=1");

    for (size_t r = 0; r < M_DIM_LOC; ++r) {
        for (size_t c_chunk = 0; c_chunk < K_BYTES_LOC / 16; ++c_chunk) {
            uint8_t b0 = r & 0xFF;
            uint8_t b1 = (r >> 8) & 0xFF;
            uint8_t b2 = c_chunk & 0xFF;
            uint8_t b3 = (c_chunk >> 8) & 0xFF;
            for (int k = 0; k < 4; ++k) {
                size_t off = r * K_BYTES_LOC + c_chunk * 16 + k*4;
                h_a[off + 0] = b0;
                h_a[off + 1] = b1;
                h_a[off + 2] = b2;
                h_a[off + 3] = b3;
            }
        }
    }

    fp8e4m3* d_a = nullptr;
    HIPCHK(hipMalloc(&d_a, A_BYTES));
    HIPCHK(hipMemcpy(d_a, h_a, A_BYTES, hipMemcpyHostToDevice));

    fp4_intx8_t* d_ref = nullptr;
    fp4_intx8_t* d_cand = nullptr;
    constexpr size_t RC_COUNT = NUM_THREADS_LOC * 4;
    HIPCHK(hipMalloc(&d_ref,  RC_COUNT * sizeof(fp4_intx8_t)));
    HIPCHK(hipMalloc(&d_cand, RC_COUNT * sizeof(fp4_intx8_t)));
    HIPCHK(hipMemset(d_ref,  0, RC_COUNT * sizeof(fp4_intx8_t)));
    HIPCHK(hipMemset(d_cand, 0, RC_COUNT * sizeof(fp4_intx8_t)));

    // gl shape
    _gl_fp4 g_a{d_a, 1, 1, M_DIM_LOC, K_BYTES_LOC};

    // Test cases: full Cartesian product over (br_idx, half_idx, bt)
    int total_cases = 0;
    int total_pass  = 0;
    int total_fail  = 0;
    int total_pass_lanes = 0;
    int total_fail_lanes = 0;
    int n_br_test = M_DIM_LOC / BLK_LOC;  // = 2

    for (int br = 0; br < n_br_test; ++br) {
        for (int half = 0; half < 2; ++half) {
            for (int bt = 0; bt < K_BYTE_ITERS_LOC; ++bt) {
                HIPCHK(hipMemset(d_ref,  0, RC_COUNT * sizeof(fp4_intx8_t)));
                HIPCHK(hipMemset(d_cand, 0, RC_COUNT * sizeof(fp4_intx8_t)));

                iso_globals g{ g_a, d_ref, d_cand, br, half, bt };
                dim3 grid(1), block(NUM_THREADS_LOC);
                hipLaunchKernelGGL(iso_b3b_kernel, grid, block, 0, 0, g);
                HIPCHK(hipGetLastError());
                HIPCHK(hipDeviceSynchronize());

                std::vector<uint8_t> h_ref(RC_COUNT * sizeof(fp4_intx8_t));
                std::vector<uint8_t> h_cand(RC_COUNT * sizeof(fp4_intx8_t));
                HIPCHK(hipMemcpy(h_ref.data(),  d_ref,  h_ref.size(),  hipMemcpyDeviceToHost));
                HIPCHK(hipMemcpy(h_cand.data(), d_cand, h_cand.size(), hipMemcpyDeviceToHost));

                int n_lanes_pass = 0, n_lanes_fail = 0;
                int first_fail_lid = -1;
                for (int lid = 0; lid < NUM_THREADS_LOC; ++lid) {
                    bool ok = true;
                    for (int s = 0; s < 4 && ok; ++s) {
                        uint8_t* r = &h_ref [(lid*4 + s) * 32];
                        uint8_t* c = &h_cand[(lid*4 + s) * 32];
                        for (int b = 0; b < 32; ++b)
                            if (r[b] != c[b]) { ok = false; break; }
                    }
                    if (ok) n_lanes_pass++;
                    else { n_lanes_fail++; if (first_fail_lid < 0) first_fail_lid = lid; }
                }
                total_cases++;
                total_pass_lanes += n_lanes_pass;
                total_fail_lanes += n_lanes_fail;
                bool case_ok = (n_lanes_fail == 0);
                if (case_ok) total_pass++; else total_fail++;
                printf("  case br=%d half=%d bt=%d  : %s  pass=%d fail=%d  first_fail_lid=%d\n",
                       br, half, bt, case_ok ? "OK  " : "FAIL", n_lanes_pass, n_lanes_fail, first_fail_lid);

                // Dump first failing lane on failure
                if (!case_ok) {
                    int lid = first_fail_lid;
                    int wid = lid / WARP_THREADS;
                    int lane = lid % WARP_THREADS;
                    printf("    lane=%d (wm=%d,wn=%d,intra=%d) first slot diff:\n",
                           lid, wid / WARPS_N_LOC, wid % WARPS_N_LOC, lane);
                    for (int s = 0; s < 4; ++s) {
                        uint8_t* r = &h_ref [(lid*4 + s) * 32];
                        uint8_t* c = &h_cand[(lid*4 + s) * 32];
                        bool slot_ok = true;
                        for (int b = 0; b < 32; ++b)
                            if (r[b] != c[b]) { slot_ok = false; break; }
                        if (slot_ok) continue;
                        printf("      slot %d ref :", s);
                        for (int b = 0; b < 16; ++b) printf(" %02x", r[b]);
                        printf("\n             cand:");
                        for (int b = 0; b < 16; ++b) printf(" %02x", c[b]);
                        printf("\n");
                    }
                }
            }
        }
    }

    printf("\n=========================================\n");
    printf("  Total cases tested : %d\n", total_cases);
    printf("  Cases passed all 256 lanes : %d\n", total_pass);
    printf("  Cases with at least 1 fail : %d\n", total_fail);
    printf("  Total lanes passed : %d / %d\n", total_pass_lanes, total_cases * NUM_THREADS_LOC);
    printf("=========================================\n");

    HIPCHK(hipFree(d_a));
    HIPCHK(hipFree(d_ref));
    HIPCHK(hipFree(d_cand));
    std::free(h_a);

    bool ok = (total_fail == 0);
    printf("\n=== VERDICT (HYP=%d): %s ===\n", HYP, ok ? "ALL_CASES_PASS" : "MISMATCH");
    return ok ? 0 : 2;
}
