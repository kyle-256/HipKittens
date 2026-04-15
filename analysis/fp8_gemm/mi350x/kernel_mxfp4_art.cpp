// MXFP4 ART (Assembly Register Tile) kernel for MI355X (gfx950)
//
// Based on kernel_mxfp4_gluon_cpp.cpp but with explicit register allocation
// via ART-style inline ASM to eliminate VGPR spills.
//
// Key differences from the original:
//   - __attribute__((amdgpu_num_vgpr(29))) restricts compiler to v[0:28]
//   - All tile data in pinned VGPRs: B in v[30:61], A in v[62:93]
//   - A tiles loaded direct from global via buffer_load_dwordx4 (no A in LDS)
//   - Only B tiles in LDS (Bl_db[2], Br_db[2])
//   - All MFMA/ds_read/buffer_load use explicit register numbers
//   - Accumulators in a[0:255] (256 AGPRs)
//
// Register map (from art_register_map.md):
//   a[0:63]    = Accumulator block 0 (A0*Bl)
//   a[64:127]  = Accumulator block 1 (A0*Br)
//   a[128:191] = Accumulator block 2 (A1*Bl)
//   a[192:255] = Accumulator block 3 (A1*Br)
//   v[0:28]    = Compiler reserved
//   v[29]      = Output scale
//   v[30:61]   = B tile current (from LDS)
//   v[62:93]   = A tile current (direct from global)
//   v[94:125]  = Next B tile (from LDS)
//   v[126:157] = Next A tile (from global)
//   v[158:165] = B scales (4 values * 2)
//   v[166:173] = A scales (4 values * 2)
//   v[174:177] = B LDS base addresses
//   v[178]     = A voffset base
//   v[179:255] = Store temporaries

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <type_traits>
using namespace kittens;

#ifndef M_DIM
#define M_DIM 8192
#endif
#ifndef K_DIM
#define K_DIM 8192
#endif
#ifndef N_DIM
#define N_DIM 8192
#endif

constexpr int BLK = 256;
constexpr int BK  = 128;
constexpr int WARPS_M = 2, WARPS_N = 2;
constexpr int _NUM_WARPS   = WARPS_M * WARPS_N;
constexpr int _NUM_THREADS = _NUM_WARPS * WARP_THREADS;
constexpr int HB = BLK / 2;
constexpr int RBM = HB / WARPS_M;
constexpr int RBN = HB / WARPS_N;

constexpr int K_BYTES = K_DIM / 2;
constexpr int k_byte_iters = K_BYTES / BK;

using ST_tile = st_fp8e4m3<HB, BK, st_16x128_s>;
using A_row_reg = rt_fp8e4m3<RBM, BK, row_l, rt_16x128_s>;
using B_row_reg = rt_fp8e4m3<RBN, BK, row_l, rt_16x128_s>;

using G = kittens::group<_NUM_WARPS>;
using _gl_fp4   = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_scale = gl<fp8e8m0, -1, -1, -1, -1>;
using _gl_bf16  = gl<bf16, -1, -1, -1, -1>;

struct gluon_globals {
    _gl_fp4 a, b;
    _gl_scale a_scale, b_scale;
    _gl_bf16 c;
    float scale = 1.0f;
};

using fp4_intx8_t   = int __attribute__((__vector_size__(8 * sizeof(int))));
using fp4_intx4_t   = int __attribute__((__vector_size__(4 * sizeof(int))));
using fp4_floatx4_t = float __attribute__((__vector_size__(4 * sizeof(float))));

// ── Scale helpers (same as original) ──

__device__ __forceinline__ const uint8_t* preshuffled_scale_row_base_ptr(
    const _gl_scale& src, int row_group) {
    return reinterpret_cast<const uint8_t*>(src.raw_ptr + src.idx(coord<>(row_group, 0)));
}

__device__ __forceinline__ i32x4 make_scale_srd(const uint8_t* ptr) {
    i32x4 srd = std::bit_cast<i32x4>(make_buffer_resource(
        static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(ptr)),
        0xFFFFFFFFu, 0x00110000u));
    srd[0] = __builtin_amdgcn_readfirstlane(srd[0]);
    srd[1] = __builtin_amdgcn_readfirstlane(srd[1]);
    srd[2] = __builtin_amdgcn_readfirstlane(srd[2]);
    srd[3] = __builtin_amdgcn_readfirstlane(srd[3]);
    return srd;
}

__device__ __forceinline__ void load_pq_scale_x2_async(
    i32x4 srsrc, uint32_t voffset, uint32_t soffset,
    fp8e8m0_4 &out_lo, fp8e8m0_4 &out_hi) {
    uint64_t pair;
    asm volatile(
        "buffer_load_dwordx2 %0, %1, %2, %3 offen"
        : "=v"(pair)
        : "v"(voffset), "s"(srsrc), "s"(soffset)
    );
    out_lo = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(pair));
    out_hi = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(pair >> 32));
}

// ── Tile prefetch (B tiles to LDS, same as original) ──

static constexpr int PF_MPT = (HB * BK * sizeof(fp8e4m3)) / (16 * _NUM_THREADS);

__device__ __forceinline__ void emit_tile_pf(
    auto &dst, const auto &src, const auto &idx,
    const uint32_t *so, i32x4 srd, const void *base, uint32_t lb)
{
    using ST = std::remove_reference_t<decltype(dst)>;
    using T = typename ST::dtype;
    coord<> uc = idx.template unit_coord<2, 3>();
    T* gptr = (T*)&src[uc];
    uint32_t soff = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<const char*>(gptr) - reinterpret_cast<const char*>(base)));
    asm volatile("" : "+s"(soff));
    const uint32_t lds_tile_base = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&dst.data[0])));
    const uint32_t warp_off = lb - lds_tile_base;
    constexpr int BPM = 16 * _NUM_THREADS;
    #pragma unroll
    for (int i = 0; i < PF_MPT; ++i) {
        const uint32_t lin = warp_off + i * BPM;
        const uint32_t sid = lin / ST::underlying_subtile_bytes;
        uint32_t lds_b = lds_tile_base + lin + sid * ST::subtile_padding;
        asm volatile("" : "+s"(lds_b));
        llvm_amdgcn_raw_buffer_load_lds(
            std::bit_cast<int32x4_t>(srd),
            (as3_uint32_ptr)(uintptr_t)lds_b,
            16, so[i], soff, 0,
            static_cast<int>(coherency::cache_all));
    }
}

// ── LDS address computation for B tile ds_read ──

template<ducks::rt::row_layout RT, ducks::st::all ST>
__device__ __forceinline__ void compute_lds_base_addrs(
    const ST &src, uint32_t &addr_p0, uint32_t &addr_p1)
{
    const int laneid = kittens::laneid();
    const int row_offset = laneid % RT::base_tile_rows;
    const int col_offset = RT::base_tile_stride * (laneid / RT::base_tile_rows);
    const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);
    using U = typename ST::dtype;
    constexpr int subcols = ST::underlying_subtile_cols;
    const uint32_t off0 = sizeof(U) * (src_ptr + row_offset * subcols + col_offset);
    addr_p0 = off0 ^ (((off0 % (16 * 128)) >> 8) << 4);
    const int col1 = col_offset + RT::base_tile_elements_per_stride_group;
    const uint32_t off1 = sizeof(U) * (src_ptr + row_offset * subcols + col1);
    addr_p1 = off1 ^ (((off1 % (16 * 128)) >> 8) << 4);
}

// ── Tile prefetch params (for individual emit_one_pf calls) ──

struct tile_pf_params {
    int32x4_t srd;
    uint32_t soff;
    uint32_t lds_addrs[PF_MPT];
    uint32_t voffs[PF_MPT];
};

template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ __forceinline__ tile_pf_params make_pf_params(
    ST &dst, const GL &src, const COORD &idx,
    const uint32_t *so, i32x4 srd_in, const void *base_ptr, uint32_t lds_base)
{
    using T = typename ST::dtype;
    constexpr int BPM = 16 * _NUM_THREADS;
    coord<> uc = idx.template unit_coord<2, 3>();
    T* gptr = (T*)&src[uc];
    uint32_t soff = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<const char*>(gptr) - reinterpret_cast<const char*>(base_ptr)));
    const uint32_t lds_tile_base = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&dst.data[0])));
    const uint32_t warp_off = lds_base - lds_tile_base;
    tile_pf_params p;
    p.srd = std::bit_cast<int32x4_t>(srd_in);
    p.soff = soff;
    #pragma unroll
    for (int i = 0; i < PF_MPT; ++i) {
        p.voffs[i] = so[i];
        const uint32_t lin = warp_off + i * BPM;
        const uint32_t sid = lin / ST::underlying_subtile_bytes;
        p.lds_addrs[i] = lds_tile_base + lin + sid * ST::subtile_padding;
    }
    return p;
}

__device__ __forceinline__ void emit_one_pf(const tile_pf_params& p, int idx) {
    llvm_amdgcn_raw_buffer_load_lds(
        std::bit_cast<int32x4_t>(p.srd),
        (as3_uint32_ptr)(uintptr_t)p.lds_addrs[idx],
        16, p.voffs[idx], p.soff, 0,
        static_cast<int>(coherency::cache_all));
}

// ══════════════════════════════════════════════════════════════
// ART-style MFMA macro (copied from HipKittens3 macros.cuh)
// ══════════════════════════════════════════════════════════════

namespace art_macros {

template<int GPR_D, int GPR_A, int GPR_B, int GPR_C,
         int GPR_SA, int GPR_SB,
         int SEL_A, int SEL_B,
         int HI_A, int HI_B>
__device__ __forceinline__ void mfma_scale_f32_16x16x128_fp4() {
    static_assert(GPR_SA < 256 && GPR_SB < 256, "Scale registers must be VGPRs");

    constexpr auto d_lo = GPR_D >= 256 ? GPR_D - 256 : GPR_D;
    constexpr auto d_hi = d_lo + 3;
    constexpr auto a_lo = GPR_A >= 256 ? GPR_A - 256 : GPR_A;
    constexpr auto a_hi = a_lo + 3;
    constexpr auto b_lo = GPR_B >= 256 ? GPR_B - 256 : GPR_B;
    constexpr auto b_hi = b_lo + 3;
    constexpr auto c_lo = GPR_C >= 256 ? GPR_C - 256 : GPR_C;
    constexpr auto c_hi = c_lo + 3;

    constexpr bool dA = GPR_D >= 256, aA = GPR_A >= 256, bA = GPR_B >= 256, cA = GPR_C >= 256;

    #define _FP4_ASM_BODY(D_PFX, A_PFX, B_PFX, C_PFX, OPSEL, OPSELHI) \
        asm volatile("v_mfma_scale_f32_16x16x128_f8f6f4 " \
            D_PFX "[%0:%1], " A_PFX "[%2:%3], " B_PFX "[%4:%5], " \
            C_PFX "[%6:%7], v[%8], v[%9] " OPSEL " " OPSELHI " cbsz:4 blgp:4" \
            : : "n"(d_lo), "n"(d_hi), "n"(a_lo), "n"(a_hi), \
                "n"(b_lo), "n"(b_hi), "n"(c_lo), "n"(c_hi), \
                "n"(GPR_SA), "n"(GPR_SB))

    #define _DO_MFMA(DP, AP, BP, CP) do { \
        if constexpr (SEL_A==0 && SEL_B==0 && HI_A==0 && HI_B==0) { \
            _FP4_ASM_BODY(DP, AP, BP, CP, "", "op_sel_hi:[0,0,0]"); \
        } else if constexpr (SEL_A==0 && SEL_B==1 && HI_A==0 && HI_B==0) { \
            _FP4_ASM_BODY(DP, AP, BP, CP, "op_sel:[0,1,0]", "op_sel_hi:[0,0,0]"); \
        } else if constexpr (SEL_A==1 && SEL_B==0 && HI_A==0 && HI_B==0) { \
            _FP4_ASM_BODY(DP, AP, BP, CP, "op_sel:[1,0,0]", "op_sel_hi:[0,0,0]"); \
        } else if constexpr (SEL_A==1 && SEL_B==1 && HI_A==0 && HI_B==0) { \
            _FP4_ASM_BODY(DP, AP, BP, CP, "op_sel:[1,1,0]", "op_sel_hi:[0,0,0]"); \
        } else if constexpr (SEL_A==0 && SEL_B==0 && HI_A==1 && HI_B==1) { \
            _FP4_ASM_BODY(DP, AP, BP, CP, "", "op_sel_hi:[1,1,0]"); \
        } else if constexpr (SEL_A==0 && SEL_B==1 && HI_A==1 && HI_B==1) { \
            _FP4_ASM_BODY(DP, AP, BP, CP, "op_sel:[0,1,0]", "op_sel_hi:[1,1,0]"); \
        } else if constexpr (SEL_A==1 && SEL_B==0 && HI_A==1 && HI_B==1) { \
            _FP4_ASM_BODY(DP, AP, BP, CP, "op_sel:[1,0,0]", "op_sel_hi:[1,1,0]"); \
        } else if constexpr (SEL_A==1 && SEL_B==1 && HI_A==1 && HI_B==1) { \
            _FP4_ASM_BODY(DP, AP, BP, CP, "op_sel:[1,1,0]", "op_sel_hi:[1,1,0]"); \
        } \
    } while(0)

    // Our case: D=AGPR, A=VGPR, B=VGPR, C=AGPR
    if constexpr (dA && !aA && !bA && cA) { _DO_MFMA("a","v","v","a"); }
    else if constexpr (dA && aA && bA && cA) { _DO_MFMA("a","a","a","a"); }
    else if constexpr (dA && aA && !bA && cA) { _DO_MFMA("a","a","v","a"); }
    else if constexpr (dA && !aA && bA && cA) { _DO_MFMA("a","v","a","a"); }
    else if constexpr (!dA && aA && bA && cA) { _DO_MFMA("v","a","a","a"); }
    else if constexpr (dA && aA && !bA && !cA) { _DO_MFMA("a","a","v","v"); }
    else if constexpr (dA && !aA && bA && !cA) { _DO_MFMA("a","v","a","v"); }
    else if constexpr (!dA && aA && bA && !cA) { _DO_MFMA("v","a","a","v"); }
    else if constexpr (!dA && aA && !bA && cA) { _DO_MFMA("v","a","v","a"); }
    else if constexpr (!dA && !aA && bA && cA) { _DO_MFMA("v","v","a","a"); }
    else if constexpr (dA && !aA && !bA && !cA) { _DO_MFMA("a","v","v","v"); }
    else if constexpr (!dA && aA && !bA && !cA) { _DO_MFMA("v","a","v","v"); }
    else if constexpr (!dA && !aA && bA && !cA) { _DO_MFMA("v","v","a","v"); }
    else if constexpr (!dA && !aA && !bA && cA) { _DO_MFMA("v","v","v","a"); }
    else { _DO_MFMA("v","v","v","v"); }

    #undef _DO_MFMA
    #undef _FP4_ASM_BODY
}

// ART-style ds_read_b128 to explicit VGPRs
template<int GPR_START>
__device__ __forceinline__ void ds_read_b128(uint32_t addr, int offset) {
    constexpr int GPR_END = GPR_START + 3;
    asm volatile("ds_read_b128 v[%0:%1], %2 offset:%3"
        : : "n"(GPR_START), "n"(GPR_END), "v"(addr), "i"(offset) : "memory");
}

// ART-style buffer_load_dwordx4 to explicit VGPRs
template<int GPR_START>
__device__ __forceinline__ void buffer_load_dwordx4(const i32x4 &srd, uint32_t voffset, uint32_t soffset, int ioffset = 0) {
    constexpr int GPR_END = GPR_START + 3;
    asm volatile("buffer_load_dwordx4 v[%0:%1], %2, %3, %4 offen offset:%5"
        : : "n"(GPR_START), "n"(GPR_END), "v"(voffset), "s"(srd), "s"(soffset), "i"(ioffset) : "memory");
}

// ART-style buffer_load_dwordx2 for scale loading to explicit VGPRs
template<int GPR_START>
__device__ __forceinline__ void buffer_load_dwordx2(i32x4 srd, uint32_t voffset, uint32_t soffset) {
    constexpr int GPR_END = GPR_START + 1;
    asm volatile("buffer_load_dwordx2 v[%0:%1], %2, %3, %4 offen"
        : : "n"(GPR_START), "n"(GPR_END), "v"(voffset), "s"(srd), "s"(soffset) : "memory");
}

// v_accvgpr_read to explicit VGPR
template<int VGPR_DST, int AGPR_SRC>
__device__ __forceinline__ void v_accvgpr_read_b32() {
    asm volatile("v_accvgpr_read_b32 v[%0], a[%1]"
        : : "n"(VGPR_DST), "n"(AGPR_SRC) : "memory");
}

// v_mul_f32 on explicit GPR
template<int DST, int SRC0, int SRC1>
__device__ __forceinline__ void v_mul_f32() {
    asm volatile("v_mul_f32 v[%0], v[%1], v[%2]"
        : : "n"(DST), "n"(SRC0), "n"(SRC1));
}

// v_cvt_pk_bf16_f32 on explicit GPRs
template<int DST, int SRC0, int SRC1>
__device__ __forceinline__ void v_cvt_pk_bf16_f32() {
    asm volatile("v_cvt_pk_bf16_f32 v[%0], v[%1], v[%2]"
        : : "n"(DST), "n"(SRC0), "n"(SRC1));
}

} // namespace art_macros

// ══════════════════════════════════════════════════════════════
// Register clobber: tell compiler v[29:255] and a[0:255] are ours
// ══════════════════════════════════════════════════════════════

__device__ __forceinline__ void clobber_art_registers() {
    // Clobber v[29:255]
    asm volatile("" :::
        "v29",
        "v30","v31","v32","v33","v34","v35","v36","v37","v38","v39",
        "v40","v41","v42","v43","v44","v45","v46","v47","v48","v49",
        "v50","v51","v52","v53","v54","v55","v56","v57","v58","v59",
        "v60","v61","v62","v63","v64","v65","v66","v67","v68","v69",
        "v70","v71","v72","v73","v74","v75","v76","v77","v78","v79",
        "v80","v81","v82","v83","v84","v85","v86","v87","v88","v89",
        "v90","v91","v92","v93","v94","v95","v96","v97","v98","v99"
    );
    asm volatile("" :::
        "v100","v101","v102","v103","v104","v105","v106","v107","v108","v109",
        "v110","v111","v112","v113","v114","v115","v116","v117","v118","v119",
        "v120","v121","v122","v123","v124","v125","v126","v127","v128","v129",
        "v130","v131","v132","v133","v134","v135","v136","v137","v138","v139",
        "v140","v141","v142","v143","v144","v145","v146","v147","v148","v149",
        "v150","v151","v152","v153","v154","v155","v156","v157","v158","v159",
        "v160","v161","v162","v163","v164","v165","v166","v167","v168","v169",
        "v170","v171","v172","v173","v174","v175","v176","v177","v178","v179",
        "v180","v181","v182","v183","v184","v185","v186","v187","v188","v189",
        "v190","v191","v192","v193","v194","v195","v196","v197","v198","v199"
    );
    asm volatile("" :::
        "v200","v201","v202","v203","v204","v205","v206","v207","v208","v209",
        "v210","v211","v212","v213","v214","v215","v216","v217","v218","v219",
        "v220","v221","v222","v223","v224","v225","v226","v227","v228","v229",
        "v230","v231","v232","v233","v234","v235","v236","v237","v238","v239",
        "v240","v241","v242","v243","v244","v245","v246","v247","v248","v249",
        "v250","v251","v252","v253","v254","v255"
    );
    // Clobber a[0:255]
    asm volatile("" :::
        "a0","a1","a2","a3","a4","a5","a6","a7","a8","a9",
        "a10","a11","a12","a13","a14","a15","a16","a17","a18","a19",
        "a20","a21","a22","a23","a24","a25","a26","a27","a28","a29",
        "a30","a31","a32","a33","a34","a35","a36","a37","a38","a39",
        "a40","a41","a42","a43","a44","a45","a46","a47","a48","a49",
        "a50","a51","a52","a53","a54","a55","a56","a57","a58","a59",
        "a60","a61","a62","a63","a64","a65","a66","a67","a68","a69",
        "a70","a71","a72","a73","a74","a75","a76","a77","a78","a79",
        "a80","a81","a82","a83","a84","a85","a86","a87","a88","a89",
        "a90","a91","a92","a93","a94","a95","a96","a97","a98","a99"
    );
    asm volatile("" :::
        "a100","a101","a102","a103","a104","a105","a106","a107","a108","a109",
        "a110","a111","a112","a113","a114","a115","a116","a117","a118","a119",
        "a120","a121","a122","a123","a124","a125","a126","a127","a128","a129",
        "a130","a131","a132","a133","a134","a135","a136","a137","a138","a139",
        "a140","a141","a142","a143","a144","a145","a146","a147","a148","a149",
        "a150","a151","a152","a153","a154","a155","a156","a157","a158","a159",
        "a160","a161","a162","a163","a164","a165","a166","a167","a168","a169",
        "a170","a171","a172","a173","a174","a175","a176","a177","a178","a179",
        "a180","a181","a182","a183","a184","a185","a186","a187","a188","a189",
        "a190","a191","a192","a193","a194","a195","a196","a197","a198","a199"
    );
    asm volatile("" :::
        "a200","a201","a202","a203","a204","a205","a206","a207","a208","a209",
        "a210","a211","a212","a213","a214","a215","a216","a217","a218","a219",
        "a220","a221","a222","a223","a224","a225","a226","a227","a228","a229",
        "a230","a231","a232","a233","a234","a235","a236","a237","a238","a239",
        "a240","a241","a242","a243","a244","a245","a246","a247","a248","a249",
        "a250","a251","a252","a253","a254","a255"
    );
}

// ══════════════════════════════════════════════════════════════
// Clobber all AGPRs — tells compiler that accumulator values are unknown
// Must be called after MFMA blocks to prevent compiler from caching stale values
// ══════════════════════════════════════════════════════════════
__device__ __forceinline__ void clobber_all_agprs() {
    asm volatile("" :::
        "a0","a1","a2","a3","a4","a5","a6","a7","a8","a9",
        "a10","a11","a12","a13","a14","a15","a16","a17","a18","a19",
        "a20","a21","a22","a23","a24","a25","a26","a27","a28","a29",
        "a30","a31","a32","a33","a34","a35","a36","a37","a38","a39",
        "a40","a41","a42","a43","a44","a45","a46","a47","a48","a49",
        "a50","a51","a52","a53","a54","a55","a56","a57","a58","a59",
        "a60","a61","a62","a63","a64","a65","a66","a67","a68","a69",
        "a70","a71","a72","a73","a74","a75","a76","a77","a78","a79",
        "a80","a81","a82","a83","a84","a85","a86","a87","a88","a89",
        "a90","a91","a92","a93","a94","a95","a96","a97","a98","a99"
    );
    asm volatile("" :::
        "a100","a101","a102","a103","a104","a105","a106","a107","a108","a109",
        "a110","a111","a112","a113","a114","a115","a116","a117","a118","a119",
        "a120","a121","a122","a123","a124","a125","a126","a127","a128","a129",
        "a130","a131","a132","a133","a134","a135","a136","a137","a138","a139",
        "a140","a141","a142","a143","a144","a145","a146","a147","a148","a149",
        "a150","a151","a152","a153","a154","a155","a156","a157","a158","a159",
        "a160","a161","a162","a163","a164","a165","a166","a167","a168","a169",
        "a170","a171","a172","a173","a174","a175","a176","a177","a178","a179",
        "a180","a181","a182","a183","a184","a185","a186","a187","a188","a189",
        "a190","a191","a192","a193","a194","a195","a196","a197","a198","a199"
    );
    asm volatile("" :::
        "a200","a201","a202","a203","a204","a205","a206","a207","a208","a209",
        "a210","a211","a212","a213","a214","a215","a216","a217","a218","a219",
        "a220","a221","a222","a223","a224","a225","a226","a227","a228","a229",
        "a230","a231","a232","a233","a234","a235","a236","a237","a238","a239",
        "a240","a241","a242","a243","a244","a245","a246","a247","a248","a249",
        "a250","a251","a252","a253","a254","a255"
    );
}

// ══════════════════════════════════════════════════════════════
// Zero accumulators a[0:255]
// ══════════════════════════════════════════════════════════════

__device__ __forceinline__ void zero_accumulators() {
    #pragma unroll
    for (int i = 0; i < 256; i += 4) {
        asm volatile(
            "v_accvgpr_write_b32 a[%0], 0\n"
            "v_accvgpr_write_b32 a[%1], 0\n"
            "v_accvgpr_write_b32 a[%2], 0\n"
            "v_accvgpr_write_b32 a[%3], 0\n"
            : : "n"(i), "n"(i+1), "n"(i+2), "n"(i+3));
    }
}

// ══════════════════════════════════════════════════════════════
// ART MFMA Step: 32 MFMAs for one (A_half x B_half) block
//
// Accumulator layout (per block, 64 AGPRs):
//   Row 0 (A subtile 0): acc[base+0..3], acc[base+4..7], acc[base+8..11], acc[base+12..15]
//   Row 1 (A subtile 1): acc[base+16..19], acc[base+20..23], acc[base+24..27], acc[base+28..31]
//   Row 2 (A subtile 2): acc[base+32..35], acc[base+36..39], acc[base+40..43], acc[base+44..47]
//   Row 3 (A subtile 3): acc[base+48..51], acc[base+52..55], acc[base+56..59], acc[base+60..63]
//
// MFMA operand mapping (SAME as original, NOT swapped):
//   MFMA A operand <- A-matrix data (NOT B!) = v[A_BASE + subtile*8 + phase*4]
//   MFMA B operand <- B-matrix data = v[B_BASE + subtile*8 + phase*4]
//   Scale A <- A scales = v[SA_BASE + subtile_pair]
//   Scale B <- B scales = v[SB_BASE + subtile_pair]
//
// Wait, re-reading the original kernel...
// In the original KPAIR:
//   MFMA A = "a0l" which is the A-matrix lo4
//   MFMA B = "b0l" which is the B-matrix lo4
// The asm: v_mfma ... %0, %24, %32, %0, %40, %42
//   %0  = acc
//   %24 = a0l (A matrix, MFMA A operand)
//   %32 = b0l (B matrix, MFMA B operand)
//   %40 = sa0 (A scale, scale_A)
//   %42 = sb0 (B scale, scale_B)
//
// So: MFMA_A = A_matrix, MFMA_B = B_matrix (no swap)
// scale_A = A_scale, scale_B = B_scale
//
// For ART with explicit regs:
//   A tile data: v[A_BASE..A_BASE+31], subtile i at A_BASE + i*8, lo=+0..3, hi=+4..7
//   B tile data: v[B_BASE..B_BASE+31], subtile j at B_BASE + j*8, lo=+0..3, hi=+4..7
//   A scales: v[SA_BASE], v[SA_BASE+1]  (for subtile pairs 0/1 and 2/3)
//   B scales: v[SB_BASE..SB_BASE+1]
//
// The 32 MFMAs per block follow the pattern:
// Row R (A subtile R), Phase P, Col C (B subtile C):
//   acc_offset = R*16 + C*4
//   MFMA src_A = A_BASE + R*8 + P*4   (P=0: lo, P=1: hi)
//   MFMA src_B = B_BASE + C*8 + P*4
//   scale_A = SA_BASE + (R >= 2 ? 1 : 0)
//   scale_B = SB_BASE + (C >= 2 ? 1 : 0)
//   sel_A = (R & 1)  (odd rows use op_sel A=1)
//   sel_B = (C & 1)  (odd cols use op_sel B=1)
//   hi_A = P, hi_B = P
// ══════════════════════════════════════════════════════════════

// ══════════════════════════════════════════════════════════════
// Interleaved Step 1: 32 MFMAs + 8 ds_read (Br) + 8 buffer_load (A1)
// First 8 MFMAs interleaved 1:1 with ds_reads for Br
// Then emit_one_pf for A1 buffer_loads between row blocks
// Last 24 MFMAs pure
// ══════════════════════════════════════════════════════════════
template<int ACC_BASE, int A_BASE, int B_BASE, int SA_BASE, int SB_BASE,
         int BR_DST, int A1_DST>
__device__ __forceinline__ void art_32mfma_step1_interleaved(
    uint32_t br_p0, uint32_t br_p1,
    i32x4 a_srd, uint32_t a_voff, uint32_t a1_soff)
{
    constexpr uint32_t ss = 16 * (K_DIM / 2);  // subtile stride in bytes (16 rows × K/2 bytes/row)
    // Row 0 Phase 0: 4 MFMAs + 4 ds_read Br (1:1 interleave)
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+0, A_BASE+0, B_BASE+0, ACC_BASE+0, SA_BASE+0, SB_BASE+0, 0,0, 0,0>();
    art_macros::ds_read_b128<BR_DST+0>(br_p0, 0);
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+4, A_BASE+0, B_BASE+8, ACC_BASE+4, SA_BASE+0, SB_BASE+0, 0,1, 0,0>();
    art_macros::ds_read_b128<BR_DST+4>(br_p1, 0);
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+8, A_BASE+0, B_BASE+16, ACC_BASE+8, SA_BASE+0, SB_BASE+1, 0,0, 0,0>();
    art_macros::ds_read_b128<BR_DST+8>(br_p0, 2048);
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+12, A_BASE+0, B_BASE+24, ACC_BASE+12, SA_BASE+0, SB_BASE+1, 0,1, 0,0>();
    art_macros::ds_read_b128<BR_DST+12>(br_p1, 2048);
    // Row 0 Phase 1: 4 MFMAs + 4 ds_read Br
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+0, A_BASE+4, B_BASE+4, ACC_BASE+0, SA_BASE+0, SB_BASE+0, 0,0, 1,1>();
    art_macros::ds_read_b128<BR_DST+16>(br_p0, 4096);
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+4, A_BASE+4, B_BASE+12, ACC_BASE+4, SA_BASE+0, SB_BASE+0, 0,1, 1,1>();
    art_macros::ds_read_b128<BR_DST+20>(br_p1, 4096);
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+8, A_BASE+4, B_BASE+20, ACC_BASE+8, SA_BASE+0, SB_BASE+1, 0,0, 1,1>();
    art_macros::ds_read_b128<BR_DST+24>(br_p0, 6144);
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+12, A_BASE+4, B_BASE+28, ACC_BASE+12, SA_BASE+0, SB_BASE+1, 0,1, 1,1>();
    art_macros::ds_read_b128<BR_DST+28>(br_p1, 6144);
    // 8 buffer_loads for A1 (interleaved between row 1 and row 2)
    art_macros::buffer_load_dwordx4<A1_DST+0>(a_srd, a_voff, a1_soff + 0*ss, 0);
    art_macros::buffer_load_dwordx4<A1_DST+4>(a_srd, a_voff, a1_soff + 0*ss, 64);
    // Row 1 Phase 0+1: 8 MFMAs pure
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+16, A_BASE+8, B_BASE+0, ACC_BASE+16, SA_BASE+0, SB_BASE+0, 1,0, 0,0>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+20, A_BASE+8, B_BASE+8, ACC_BASE+20, SA_BASE+0, SB_BASE+0, 1,1, 0,0>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+24, A_BASE+8, B_BASE+16, ACC_BASE+24, SA_BASE+0, SB_BASE+1, 1,0, 0,0>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+28, A_BASE+8, B_BASE+24, ACC_BASE+28, SA_BASE+0, SB_BASE+1, 1,1, 0,0>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+16, A_BASE+12, B_BASE+4, ACC_BASE+16, SA_BASE+0, SB_BASE+0, 1,0, 1,1>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+20, A_BASE+12, B_BASE+12, ACC_BASE+20, SA_BASE+0, SB_BASE+0, 1,1, 1,1>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+24, A_BASE+12, B_BASE+20, ACC_BASE+24, SA_BASE+0, SB_BASE+1, 1,0, 1,1>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+28, A_BASE+12, B_BASE+28, ACC_BASE+28, SA_BASE+0, SB_BASE+1, 1,1, 1,1>();
    // More A1 buffer_loads between row 2 and row 3
    art_macros::buffer_load_dwordx4<A1_DST+8>(a_srd, a_voff, a1_soff + 1*ss, 0);
    art_macros::buffer_load_dwordx4<A1_DST+12>(a_srd, a_voff, a1_soff + 1*ss, 64);
    // Row 2 Phase 0+1: 8 MFMAs pure
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+32, A_BASE+16, B_BASE+0, ACC_BASE+32, SA_BASE+1, SB_BASE+0, 0,0, 0,0>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+36, A_BASE+16, B_BASE+8, ACC_BASE+36, SA_BASE+1, SB_BASE+0, 0,1, 0,0>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+40, A_BASE+16, B_BASE+16, ACC_BASE+40, SA_BASE+1, SB_BASE+1, 0,0, 0,0>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+44, A_BASE+16, B_BASE+24, ACC_BASE+44, SA_BASE+1, SB_BASE+1, 0,1, 0,0>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+32, A_BASE+20, B_BASE+4, ACC_BASE+32, SA_BASE+1, SB_BASE+0, 0,0, 1,1>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+36, A_BASE+20, B_BASE+12, ACC_BASE+36, SA_BASE+1, SB_BASE+0, 0,1, 1,1>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+40, A_BASE+20, B_BASE+20, ACC_BASE+40, SA_BASE+1, SB_BASE+1, 0,0, 1,1>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+44, A_BASE+20, B_BASE+28, ACC_BASE+44, SA_BASE+1, SB_BASE+1, 0,1, 1,1>();
    art_macros::buffer_load_dwordx4<A1_DST+16>(a_srd, a_voff, a1_soff + 2*ss, 0);
    art_macros::buffer_load_dwordx4<A1_DST+20>(a_srd, a_voff, a1_soff + 2*ss, 64);
    // Row 3 Phase 0+1: 8 MFMAs pure
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+48, A_BASE+24, B_BASE+0, ACC_BASE+48, SA_BASE+1, SB_BASE+0, 1,0, 0,0>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+52, A_BASE+24, B_BASE+8, ACC_BASE+52, SA_BASE+1, SB_BASE+0, 1,1, 0,0>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+56, A_BASE+24, B_BASE+16, ACC_BASE+56, SA_BASE+1, SB_BASE+1, 1,0, 0,0>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+60, A_BASE+24, B_BASE+24, ACC_BASE+60, SA_BASE+1, SB_BASE+1, 1,1, 0,0>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+48, A_BASE+28, B_BASE+4, ACC_BASE+48, SA_BASE+1, SB_BASE+0, 1,0, 1,1>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+52, A_BASE+28, B_BASE+12, ACC_BASE+52, SA_BASE+1, SB_BASE+0, 1,1, 1,1>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+56, A_BASE+28, B_BASE+20, ACC_BASE+56, SA_BASE+1, SB_BASE+1, 1,0, 1,1>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+60, A_BASE+28, B_BASE+28, ACC_BASE+60, SA_BASE+1, SB_BASE+1, 1,1, 1,1>();
    art_macros::buffer_load_dwordx4<A1_DST+24>(a_srd, a_voff, a1_soff + 3*ss, 0);
    art_macros::buffer_load_dwordx4<A1_DST+28>(a_srd, a_voff, a1_soff + 3*ss, 64);
}

// Emit 32 MFMAs for one accumulator block (no interleaved loads — pure compute).
// ACC_BASE: AGPR base (256+offset), A_BASE/B_BASE: VGPR bases, SA_BASE/SB_BASE: scale VGPR bases
template<int ACC_BASE, int A_BASE, int B_BASE, int SA_BASE, int SB_BASE>
__device__ __forceinline__ void art_32mfma() {
    // Row 0, Phase 0 (4 MFMAs)
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+ 0, A_BASE+ 0, B_BASE+ 0, ACC_BASE+ 0, SA_BASE+0, SB_BASE+0, 0,0, 0,0>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+ 4, A_BASE+ 0, B_BASE+ 8, ACC_BASE+ 4, SA_BASE+0, SB_BASE+0, 0,1, 0,0>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+ 8, A_BASE+ 0, B_BASE+16, ACC_BASE+ 8, SA_BASE+0, SB_BASE+1, 0,0, 0,0>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+12, A_BASE+ 0, B_BASE+24, ACC_BASE+12, SA_BASE+0, SB_BASE+1, 0,1, 0,0>();
    // Row 0, Phase 1
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+ 0, A_BASE+ 4, B_BASE+ 4, ACC_BASE+ 0, SA_BASE+0, SB_BASE+0, 0,0, 1,1>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+ 4, A_BASE+ 4, B_BASE+12, ACC_BASE+ 4, SA_BASE+0, SB_BASE+0, 0,1, 1,1>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+ 8, A_BASE+ 4, B_BASE+20, ACC_BASE+ 8, SA_BASE+0, SB_BASE+1, 0,0, 1,1>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+12, A_BASE+ 4, B_BASE+28, ACC_BASE+12, SA_BASE+0, SB_BASE+1, 0,1, 1,1>();
    // Row 1, Phase 0
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+16, A_BASE+ 8, B_BASE+ 0, ACC_BASE+16, SA_BASE+0, SB_BASE+0, 1,0, 0,0>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+20, A_BASE+ 8, B_BASE+ 8, ACC_BASE+20, SA_BASE+0, SB_BASE+0, 1,1, 0,0>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+24, A_BASE+ 8, B_BASE+16, ACC_BASE+24, SA_BASE+0, SB_BASE+1, 1,0, 0,0>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+28, A_BASE+ 8, B_BASE+24, ACC_BASE+28, SA_BASE+0, SB_BASE+1, 1,1, 0,0>();
    // Row 1, Phase 1
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+16, A_BASE+12, B_BASE+ 4, ACC_BASE+16, SA_BASE+0, SB_BASE+0, 1,0, 1,1>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+20, A_BASE+12, B_BASE+12, ACC_BASE+20, SA_BASE+0, SB_BASE+0, 1,1, 1,1>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+24, A_BASE+12, B_BASE+20, ACC_BASE+24, SA_BASE+0, SB_BASE+1, 1,0, 1,1>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+28, A_BASE+12, B_BASE+28, ACC_BASE+28, SA_BASE+0, SB_BASE+1, 1,1, 1,1>();
    // Row 2, Phase 0
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+32, A_BASE+16, B_BASE+ 0, ACC_BASE+32, SA_BASE+1, SB_BASE+0, 0,0, 0,0>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+36, A_BASE+16, B_BASE+ 8, ACC_BASE+36, SA_BASE+1, SB_BASE+0, 0,1, 0,0>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+40, A_BASE+16, B_BASE+16, ACC_BASE+40, SA_BASE+1, SB_BASE+1, 0,0, 0,0>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+44, A_BASE+16, B_BASE+24, ACC_BASE+44, SA_BASE+1, SB_BASE+1, 0,1, 0,0>();
    // Row 2, Phase 1
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+32, A_BASE+20, B_BASE+ 4, ACC_BASE+32, SA_BASE+1, SB_BASE+0, 0,0, 1,1>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+36, A_BASE+20, B_BASE+12, ACC_BASE+36, SA_BASE+1, SB_BASE+0, 0,1, 1,1>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+40, A_BASE+20, B_BASE+20, ACC_BASE+40, SA_BASE+1, SB_BASE+1, 0,0, 1,1>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+44, A_BASE+20, B_BASE+28, ACC_BASE+44, SA_BASE+1, SB_BASE+1, 0,1, 1,1>();
    // Row 3, Phase 0
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+48, A_BASE+24, B_BASE+ 0, ACC_BASE+48, SA_BASE+1, SB_BASE+0, 1,0, 0,0>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+52, A_BASE+24, B_BASE+ 8, ACC_BASE+52, SA_BASE+1, SB_BASE+0, 1,1, 0,0>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+56, A_BASE+24, B_BASE+16, ACC_BASE+56, SA_BASE+1, SB_BASE+1, 1,0, 0,0>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+60, A_BASE+24, B_BASE+24, ACC_BASE+60, SA_BASE+1, SB_BASE+1, 1,1, 0,0>();
    // Row 3, Phase 1
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+48, A_BASE+28, B_BASE+ 4, ACC_BASE+48, SA_BASE+1, SB_BASE+0, 1,0, 1,1>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+52, A_BASE+28, B_BASE+12, ACC_BASE+52, SA_BASE+1, SB_BASE+0, 1,1, 1,1>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+56, A_BASE+28, B_BASE+20, ACC_BASE+56, SA_BASE+1, SB_BASE+1, 1,0, 1,1>();
    art_macros::mfma_scale_f32_16x16x128_fp4<ACC_BASE+60, A_BASE+28, B_BASE+28, ACC_BASE+60, SA_BASE+1, SB_BASE+1, 1,1, 1,1>();
}

// ══════════════════════════════════════════════════════════════
// Load B tile from LDS to explicit VGPRs
// Loads 4 subtiles × 2 phases (lo+hi) = 8 ds_read_b128 = 32 VGPRs
// ══════════════════════════════════════════════════════════════
template<int DST_BASE>
__device__ __forceinline__ void art_load_b_from_lds(uint32_t addr_p0, uint32_t addr_p1) {
    // Subtile 0: p0 offset:0 (lo), p1 offset:0 (hi)
    art_macros::ds_read_b128<DST_BASE+ 0>(addr_p0, 0);     // subtile 0, lo
    art_macros::ds_read_b128<DST_BASE+ 4>(addr_p1, 0);     // subtile 0, hi
    // Subtile 1: p0 offset:2048, p1 offset:2048
    art_macros::ds_read_b128<DST_BASE+ 8>(addr_p0, 2048);  // subtile 1, lo
    art_macros::ds_read_b128<DST_BASE+12>(addr_p1, 2048);  // subtile 1, hi
    // Subtile 2: p0 offset:4096, p1 offset:4096
    art_macros::ds_read_b128<DST_BASE+16>(addr_p0, 4096);  // subtile 2, lo
    art_macros::ds_read_b128<DST_BASE+20>(addr_p1, 4096);  // subtile 2, hi
    // Subtile 3: p0 offset:6144, p1 offset:6144
    art_macros::ds_read_b128<DST_BASE+24>(addr_p0, 6144);  // subtile 3, lo
    art_macros::ds_read_b128<DST_BASE+28>(addr_p1, 6144);  // subtile 3, hi
}

// ══════════════════════════════════════════════════════════════
// Load A tile from global to explicit VGPRs via buffer_load_dwordx4
// A is stored as FP4 packed: each subtile is 8 ints = 32 bytes.
// Per-thread offset into A tile is computed from lane geometry.
// ══════════════════════════════════════════════════════════════
template<int DST_BASE>
__device__ __forceinline__ void art_load_a_from_global(
    i32x4 a_srd, uint32_t a_voffset, uint32_t a_soffset_in)
{
    // Ensure soffset is definitely scalar
    uint32_t a_soffset = __builtin_amdgcn_readfirstlane(a_soffset_in);
    // Each buffer_load_dwordx4 loads 16 bytes (4 dwords).
    // A subtile = 8 dwords = 2 loads of dwordx4.
    // 4 subtiles = 8 loads total.
    // The voffset is per-thread; soffset points to the tile's global base.
    // Subtile stride in bytes: within the 16x128 tile, each subtile occupies
    // different rows. The layout matches the ds_read pattern.
    //
    // For direct-A with buffer_load, we need to match the original's register
    // data layout. The original kernel does fp4_load_st_to_rt which reads from
    // LDS with specific swizzled addresses. We need the same data arrangement.
    //
    // In the original kernel, after fp4_load_st_to_rt + fp4_extract_tile,
    // each subtile A[i] is fp4_intx8_t = 8 ints arranged as:
    //   [lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3]
    // where lo comes from ds_read with addr_p0 and hi from addr_p1.
    //
    // For buffer_load_dwordx4, we load raw data from global memory.
    // The A data SRD points to the raw A matrix. The per-thread voffset
    // determines which elements this thread loads.
    //
    // The key insight: the ds_read_b128 with p0/p1 addresses loads data
    // from the swizzled LDS layout. When loading directly from global,
    // we need to compute the correct byte offset for each subtile.
    //
    // For the rt_16x128_s tile layout:
    //   base_tile_rows = 16, base_tile_stride = 4 (ints packed)
    //   Each subtile in the 64x128 warp tile has dimensions determined
    //   by the subtile layout.
    //
    // Rather than recomputing all the address math here, we use the
    // fact that each thread's A voffset is fixed for the entire tile,
    // and the subtile offsets are encoded as soffset increments.
    //
    // Load pattern: 8 loads, each dwordx4 (16 bytes)
    // subtile i, lo half: offset = i * SUBTILE_STRIDE
    // subtile i, hi half: offset = i * SUBTILE_STRIDE + 16

    // For the 64x128 A tile in fp4 format:
    // Each "row" is 128 fp4 = 64 bytes. 64 rows = 4096 bytes.
    // Subtile rows: 16 rows each, 4 subtiles.
    // Per subtile: 16 rows * 64 bytes/row = 1024 bytes.
    // Per thread within subtile: 1 ds_read_b128 from each of p0, p1
    //   = 2 * 16 bytes = 32 bytes = 8 ints.
    //
    // With buffer_load from global, the byte offset per thread is:
    //   base = warp_m * 64 * (K/2) + lane_row * (K/2) + lane_col * 16
    // But we want to parametrize by (bt, subtile) via soffset.
    //
    // Actually, we need the A voffset to represent the per-thread
    // offset within the tile. Let's compute it similarly to how
    // the original kernel computes LDS addresses.

    // For now: use 8 buffer_load_dwordx4 with ioffset for subtile separation.
    // The a_voffset encodes the per-thread position (row*K_stride + col*16).
    // Subtile stride = 16 rows * (K_BYTES) bytes per row... but this is
    // in the global buffer which has stride K_BYTES per row.
    // Actually, each subtile covers 16 rows of the A matrix.
    // subtile i covers rows [i*16 .. (i+1)*16 - 1] of the warp's 64-row chunk.
    // The lo/hi split within ds_read corresponds to two K-groups.
    //
    // The voffset for a thread in subtile i, phase p:
    //   voff = a_voffset_base + i * 16 * K_BYTES + p * <phase_stride>
    //
    // But this is getting complicated. Let's use the simpler approach:
    // compute all 8 voffsets as soffset increments from the base.

    // Each load: 4 dwords = 16 bytes
    // For the rt_16x128_s layout:
    //   base_tile_rows = 16
    //   base_tile_stride = 4 (packed ints per stride group)
    //   Within a 16x128 base tile, data is organized as:
    //     lane row = laneid % 16
    //     lane_col_group = laneid / 16  (0..3)
    //   Each thread reads 4 consecutive ints from its position.
    //
    // In global memory (row-major FP4 packed):
    //   A[m][k] is stored at byte offset m * K_BYTES + k/2
    //   For a thread at (row_offset, col_group) in a 16x128 subtile:
    //     global_row = tile_m_base + subtile * 16 + row_offset
    //     global_k_byte = tile_k_base + col_group * 16  (each group = 16 bytes = 128 fp4)
    //   Wait, K=128 per tile means 128 fp4 = 64 bytes per row per tile.
    //   A ds_read_b128 loads 16 bytes = 32 fp4.
    //   With 2 reads (p0, p1), each thread loads 64 bytes? No...
    //
    // Actually, each ds_read_b128 loads 4 floats (16 bytes). In fp4 packing,
    // 16 bytes = 128 fp4 values? No: fp4 is 4 bits, so 16 bytes = 32 nibbles?
    // Wait, fp8e4m3 is 1 byte. The tile is st_fp8e4m3 using fp8 dtype,
    // even though actual data is FP4. fp8e4m3 occupies 1 byte.
    // So 16 bytes = 16 fp8e4m3 values per ds_read_b128.
    //
    // A 16x128 subtile in fp8e4m3 = 16*128 = 2048 bytes.
    // With the swizzled layout, each subtile occupies 2048 bytes in LDS
    // plus padding.
    //
    // For direct-A loading from global memory:
    // We need to load the same data that would have been in LDS.
    // The global A tensor layout is row-major with fp8 dtype.
    // A[m][k] at byte offset: m * K_DIM + k (since fp8 = 1 byte, but
    // the gl uses K_DIM/2 for packed format? Let me check.)
    //
    // The gl<fp8e4m3> has raw_ptr pointing to fp8 data.
    // For MXFP4, the actual data is FP4 packed as fp8e4m3 container.
    // Each "element" in the gl is 1 byte.
    // The A matrix dimensions used are: rows=M_DIM, cols=K_DIM/2 (since
    // K_DIM FP4 elements = K_DIM/2 bytes when packed 2 per byte).
    // Wait no, looking at the original code, the tiles are st_fp8e4m3<HB, BK, ...>
    // where BK=128. Since fp8e4m3 is 1 byte, this is 128 bytes per row.
    // But K_DIM FP4 at 4 bits = K_DIM/2 bytes. So BK=128 bytes = 256 FP4 values.
    // This matches K_BYTES = K_DIM/2 and k_byte_iters = K_BYTES/BK.

    // For direct-A buffer_load:
    // We need the per-thread byte offset into the A global tensor.
    // The tile coordinates for A0 at iteration bt:
    //   A0[br*2, bt] in tile coordinates
    //   -> row base = br*2 * HB = br*BLK, col base = bt * BK
    //   Per warp: row base += wm * RBM = wm * 64
    //
    // Within the warp's 64x128 chunk, the ds_read layout gives us:
    //   subtile i (i=0..3): rows [i*16..(i+1)*16-1]
    //   Phase 0 (p0): first 64 bytes (columns 0..63 in fp8)
    //   Phase 1 (p1): next 64 bytes (columns 64..127 in fp8)
    //
    // Per thread (laneid = lid):
    //   row_within_subtile = lid % 16
    //   col_base = (lid / 16) * 16  (stride group, 16 fp8 per group)
    //
    // Global byte offset for subtile i, phase p:
    //   global_row = tile_m_base + wm * 64 + i * 16 + (lid % 16)
    //   global_col_byte = bt * BK + p * 64 + (lid / 16) * 16
    //   offset = global_row * K_BYTES + global_col_byte
    //
    // We can factor this as:
    //   a_voffset_base = (lid % 16) * K_BYTES + (lid / 16) * 16
    //   a_soffset_tile = (br * BLK + wm * 64) * K_BYTES + bt * BK
    //   subtile_stride = 16 * K_BYTES
    //   phase_stride   = 64
    //
    // Then: offset = a_soffset_tile + i * subtile_stride + p * phase_stride + a_voffset_base
    //
    // We'll set:
    //   voffset = a_voffset  (per-thread, stored in v[178])
    //   soffset = a_soffset_tile (uniform per block, changes each bt iteration)
    //   ioffset for subtile/phase = i * subtile_stride + p * phase_stride
    //
    // But ioffset is limited to 12 bits (0..4095). subtile_stride = 16 * K_BYTES.
    // For K_DIM=8192, K_BYTES=4096, subtile_stride=65536 >> 4095.
    // So we can't use ioffset. Instead, fold subtile offset into soffset.
    //
    // Better approach: issue 8 loads with different soffsets:
    // load i (subtile s, phase p):
    //   soffset = a_soffset_tile + s * subtile_stride + p * phase_stride

    constexpr int SUBTILE_STRIDE_BYTES = 16; // will be multiplied by K_BYTES at runtime
    // We need K_BYTES at compile time for the offset calculation... but it's constexpr!

    // subtile_stride = 16 * K_BYTES (in bytes)
    // phase_stride = 64 (bytes)
    // These are constant, so we can compute them as soffset adjustments.

    constexpr int SS = 16 * K_BYTES;  // subtile stride
    constexpr int PS = 64;            // phase stride

    // Load subtile 0, lo (phase 0)
    art_macros::buffer_load_dwordx4<DST_BASE+ 0>(a_srd, a_voffset, __builtin_amdgcn_readfirstlane(a_soffset + 0*SS + 0*PS));
    // Load subtile 0, hi (phase 1)
    art_macros::buffer_load_dwordx4<DST_BASE+ 4>(a_srd, a_voffset, __builtin_amdgcn_readfirstlane(a_soffset + 0*SS + 1*PS));
    // Load subtile 1, lo
    art_macros::buffer_load_dwordx4<DST_BASE+ 8>(a_srd, a_voffset, __builtin_amdgcn_readfirstlane(a_soffset + 1*SS + 0*PS));
    // Load subtile 1, hi
    art_macros::buffer_load_dwordx4<DST_BASE+12>(a_srd, a_voffset, __builtin_amdgcn_readfirstlane(a_soffset + 1*SS + 1*PS));
    // Load subtile 2, lo
    art_macros::buffer_load_dwordx4<DST_BASE+16>(a_srd, a_voffset, __builtin_amdgcn_readfirstlane(a_soffset + 2*SS + 0*PS));
    // Load subtile 2, hi
    art_macros::buffer_load_dwordx4<DST_BASE+20>(a_srd, a_voffset, __builtin_amdgcn_readfirstlane(a_soffset + 2*SS + 1*PS));
    // Load subtile 3, lo
    art_macros::buffer_load_dwordx4<DST_BASE+24>(a_srd, a_voffset, __builtin_amdgcn_readfirstlane(a_soffset + 3*SS + 0*PS));
    // Load subtile 3, hi
    art_macros::buffer_load_dwordx4<DST_BASE+28>(a_srd, a_voffset, __builtin_amdgcn_readfirstlane(a_soffset + 3*SS + 1*PS));
}

// ══════════════════════════════════════════════════════════════
// Load scales to explicit VGPRs
// ══════════════════════════════════════════════════════════════
template<int DST_BASE>
__device__ __forceinline__ void art_load_scale_x2(i32x4 srd, uint32_t voffset, uint32_t soffset) {
    art_macros::buffer_load_dwordx2<DST_BASE>(srd, voffset, soffset);
}

// ══════════════════════════════════════════════════════════════
// Main kernel
// ══════════════════════════════════════════════════════════════

__global__ __launch_bounds__(_NUM_THREADS, 1)
void mxfp4_art_kernel(const gluon_globals g) {
    static_assert(K_BYTES % BK == 0 && N_DIM % BLK == 0 && M_DIM % BLK == 0);

    constexpr int bpc = N_DIM / BLK;

    // B tiles in LDS (no A in LDS for direct-A loading)
    __shared__ ST_tile Bl_db[2], Br_db[2];

    // XCD-aware dispatch + GROUP_SIZE_M swizzle (same as original)
    constexpr int NUM_XCDS = 8;
#ifndef GROUP_SIZE_M
#define GROUP_SIZE_M 4
#endif
    constexpr int GROUP_M = GROUP_SIZE_M;
    const int total_blocks = gridDim.x;
    const int bpr = total_blocks / bpc;

    const int raw_bid = blockIdx.x;
    const int pids_per_xcd = (total_blocks + NUM_XCDS - 1) / NUM_XCDS;
    int tall_xcds = total_blocks % NUM_XCDS;
    if (tall_xcds == 0) tall_xcds = NUM_XCDS;
    const int xcd = raw_bid % NUM_XCDS;
    const int local_pid = raw_bid / NUM_XCDS;
    int bid;
    if (xcd < tall_xcds) {
        bid = xcd * pids_per_xcd + local_pid;
    } else {
        bid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid;
    }
    if (bid >= total_blocks) return;

    const int num_pig = GROUP_M * bpc;
    const int gid = bid / num_pig;
    const int fpm = gid * GROUP_M;
    const int gsm = (bpr - fpm < GROUP_M) ? (bpr - fpm) : GROUP_M;
    const int br = fpm + (bid % gsm);
    const int bc = (bid % num_pig) / gsm;
    const int wm = warpid() / WARPS_N, wn = warpid() % WARPS_N;

    // ── Setup SRDs and offsets (before clobbering registers) ──

    uint32_t so_b[PF_MPT];
    G::prefill_swizzled_offsets(Bl_db[0], g.b, so_b);

    // Scale SRDs
    const uint32_t lane_soff_x2 =
        (static_cast<uint32_t>(kittens::laneid() / 16) << 7) |
        (static_cast<uint32_t>(kittens::laneid() % 16) << 3);

    i32x4 a0_scale_srd = make_scale_srd(preshuffled_scale_row_base_ptr(
        g.a_scale, (br * BLK + wm * RBM) >> 6));
    i32x4 a1_scale_srd = make_scale_srd(preshuffled_scale_row_base_ptr(
        g.a_scale, (br * BLK + HB + wm * RBM) >> 6));
    i32x4 bl_scale_srd = make_scale_srd(preshuffled_scale_row_base_ptr(
        g.b_scale, (bc * BLK + wn * RBN) >> 6));
    i32x4 br_scale_srd = make_scale_srd(preshuffled_scale_row_base_ptr(
        g.b_scale, (bc * BLK + HB + wn * RBN) >> 6));

    // Tile SRDs
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
    i32x4 srd_a = make_srd(g.a.raw_ptr), srd_b = make_srd(g.b.raw_ptr);
    const void *base_b = (const void*)g.b.raw_ptr;

    // B tile LDS prefetch setup
    constexpr int epw = 16 / sizeof(fp8e4m3) * WARP_THREADS;
    const uint32_t wlo = (warpid() % _NUM_WARPS) * epw * sizeof(fp8e4m3);
    auto lb = [&](auto &t) -> uint32_t {
        return __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(&t.data[0]) + wlo));
    };
    uint32_t lb_bl[2], lb_br[2];
    for (int d = 0; d < 2; ++d) {
        lb_bl[d]=lb(Bl_db[d]); lb_br[d]=lb(Br_db[d]);
    }

    auto load_b_tiles = [&](int bt, int db) {
        emit_tile_pf(Bl_db[db], g.b, coord<ST_tile>(0,0,bc*2,    bt), so_b, srd_b, base_b, lb_bl[db]);
        emit_tile_pf(Br_db[db], g.b, coord<ST_tile>(0,0,bc*2+1,  bt), so_b, srd_b, base_b, lb_br[db]);
    };

    // LDS addresses for B tile ds_reads
    uint32_t bl_0_p0, bl_0_p1, bl_1_p0, bl_1_p1;
    uint32_t br_0_p0, br_0_p1, br_1_p0, br_1_p1;
    compute_lds_base_addrs<B_row_reg>(kittens::subtile_inplace<RBN, BK>(Bl_db[0], {wn, 0}), bl_0_p0, bl_0_p1);
    compute_lds_base_addrs<B_row_reg>(kittens::subtile_inplace<RBN, BK>(Bl_db[1], {wn, 0}), bl_1_p0, bl_1_p1);
    compute_lds_base_addrs<B_row_reg>(kittens::subtile_inplace<RBN, BK>(Br_db[0], {wn, 0}), br_0_p0, br_0_p1);
    compute_lds_base_addrs<B_row_reg>(kittens::subtile_inplace<RBN, BK>(Br_db[1], {wn, 0}), br_1_p0, br_1_p1);

    // A direct-load parameters
    // Per-thread voffset for A:
    //   row_within_subtile = laneid % 16
    //   col_group = laneid / 16 (0..3)
    //   voffset = row_within_subtile * K_BYTES + col_group * 16
    const int lid = kittens::laneid();
    const uint32_t a_voffset_base = (lid % 16) * K_BYTES + (lid / 16) * 16;

    // A soffset base for tile A0 at iteration bt:
    //   (br * BLK + wm * RBM) * K_BYTES + bt * BK
    // For A1: + HB * K_BYTES
    const uint32_t a0_soff_base = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>((br * BLK + wm * RBM) * K_BYTES));
    const uint32_t a1_soff_base = a0_soff_base + HB * K_BYTES;

    // Output scale
    const float out_scale = g.scale;

    // ── Clobber all ART registers ──
    clobber_art_registers();

    // ── Zero accumulators ──
    zero_accumulators();
    clobber_all_agprs();

    // Store output scale in v[29]
    asm volatile("v_mov_b32 v[29], %0" : : "v"(out_scale));

    // ═══════════ Prologue: load first B tiles ═══════════
    load_b_tiles(0, 0);
    if (k_byte_iters > 1) load_b_tiles(1, 1);

    // Load first scales (to compiler-managed vars, then move to pinned regs in loop)
    fp8e8m0_4 pf_a0_s[2], pf_a1_s[2], pf_bl_s[2], pf_br_s[2];
    load_pq_scale_x2_async(a0_scale_srd, lane_soff_x2, 0, pf_a0_s[0], pf_a0_s[1]);
    load_pq_scale_x2_async(a1_scale_srd, lane_soff_x2, 0, pf_a1_s[0], pf_a1_s[1]);
    load_pq_scale_x2_async(bl_scale_srd, lane_soff_x2, 0, pf_bl_s[0], pf_bl_s[1]);
    load_pq_scale_x2_async(br_scale_srd, lane_soff_x2, 0, pf_br_s[0], pf_br_s[1]);

    // Wait for B tiles to arrive in LDS
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __builtin_amdgcn_s_barrier();

    // Pre-load first A0 tile from global to v[62:93]
    {
        uint32_t a0_soff = __builtin_amdgcn_readfirstlane(a0_soff_base + 0 * BK);
        art_load_a_from_global<62>(srd_a, a_voffset_base, a0_soff);
    }

    // Pre-load first Bl tile from LDS to v[30:61]
    art_load_b_from_lds<30>(bl_0_p0, bl_0_p1);

    // Wait for A global loads and B LDS reads
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");

    // No debug MFMAs

    // ═══════════ Main loop ═══════════
#ifdef UNROLL_K
  #if UNROLL_K == 0
    #pragma unroll
  #else
    #pragma unroll UNROLL_K
  #endif
#elif (K_DIM / 256) <= 16
    #pragma unroll
#elif (K_DIM / 256) <= 32
    #pragma unroll 16
#else
    #pragma unroll 8
#endif
    for (int bt = 0; bt < k_byte_iters; ++bt) {
        const int cur = bt & 1;
        const int nxt = 1 - cur;

        // Select B LDS addresses for current double-buffer slot
        const uint32_t sel_bl_p0 = cur ? bl_1_p0 : bl_0_p0;
        const uint32_t sel_bl_p1 = cur ? bl_1_p1 : bl_0_p1;
        const uint32_t sel_br_p0 = cur ? br_1_p0 : br_0_p0;
        const uint32_t sel_br_p1 = cur ? br_1_p1 : br_0_p1;
        const uint32_t nxt_bl_p0 = nxt ? bl_1_p0 : bl_0_p0;
        const uint32_t nxt_bl_p1 = nxt ? bl_1_p1 : bl_0_p1;

        // Copy scales from prefetch vars to pinned scale VGPRs
        // A0 scales -> v[166:167], A1 scales -> v[168:169]
        // Bl scales -> v[158:159], Br scales -> v[160:161]
        {
            unsigned sa0_0 = std::bit_cast<unsigned>(pf_a0_s[0]);
            unsigned sa0_1 = std::bit_cast<unsigned>(pf_a0_s[1]);
            unsigned sa1_0 = std::bit_cast<unsigned>(pf_a1_s[0]);
            unsigned sa1_1 = std::bit_cast<unsigned>(pf_a1_s[1]);
            unsigned sbl_0 = std::bit_cast<unsigned>(pf_bl_s[0]);
            unsigned sbl_1 = std::bit_cast<unsigned>(pf_bl_s[1]);
            unsigned sbr_0 = std::bit_cast<unsigned>(pf_br_s[0]);
            unsigned sbr_1 = std::bit_cast<unsigned>(pf_br_s[1]);

            asm volatile(
                "v_mov_b32 v[166], %0\n"
                "v_mov_b32 v[167], %1\n"
                "v_mov_b32 v[168], %2\n"
                "v_mov_b32 v[169], %3\n"
                "v_mov_b32 v[158], %4\n"
                "v_mov_b32 v[159], %5\n"
                "v_mov_b32 v[160], %6\n"
                "v_mov_b32 v[161], %7\n"
                : : "v"(sa0_0), "v"(sa0_1), "v"(sa1_0), "v"(sa1_1),
                    "v"(sbl_0), "v"(sbl_1), "v"(sbr_0), "v"(sbr_1)
            );
        }

        // Prefetch next scales
        {
            const uint32_t nxt_scale = static_cast<uint32_t>(bt + 1 < k_byte_iters ? bt + 1 : bt) << 9;
            load_pq_scale_x2_async(a0_scale_srd, lane_soff_x2, nxt_scale, pf_a0_s[0], pf_a0_s[1]);
            load_pq_scale_x2_async(a1_scale_srd, lane_soff_x2, nxt_scale, pf_a1_s[0], pf_a1_s[1]);
            load_pq_scale_x2_async(bl_scale_srd, lane_soff_x2, nxt_scale, pf_bl_s[0], pf_bl_s[1]);
            load_pq_scale_x2_async(br_scale_srd, lane_soff_x2, nxt_scale, pf_br_s[0], pf_br_s[1]);
        }

        // Prepare B tile prefetch params for next iteration
        const int pf_bt = (bt + 2 < k_byte_iters) ? (bt + 2) : (k_byte_iters - 1);
        tile_pf_params pf_bl_p = make_pf_params(Bl_db[cur], g.b, coord<ST_tile>(0,0,bc*2,   pf_bt), so_b, srd_b, base_b, lb_bl[cur]);
        tile_pf_params pf_br_p = make_pf_params(Br_db[cur], g.b, coord<ST_tile>(0,0,bc*2+1, pf_bt), so_b, srd_b, base_b, lb_br[cur]);

        // ═══ Step 1+2: 64 MFMAs (A0×Bl, A0×Br) + 8 ds_read Br + 8 buffer_load A1 ═══
        // ONE monolithic asm block for maximum MFMA pipeline utilization
        {
            uint32_t a1_soff = __builtin_amdgcn_readfirstlane(a1_soff_base + bt * BK);
            constexpr uint32_t ss = 16 * (K_DIM / 2);
            uint32_t s0 = a1_soff, s1 = a1_soff + 64;
            uint32_t s2 = a1_soff + ss, s3 = a1_soff + ss + 64;
            uint32_t s4 = a1_soff + 2*ss, s5 = a1_soff + 2*ss + 64;
            uint32_t s6 = a1_soff + 3*ss, s7 = a1_soff + 3*ss + 64;
            asm volatile(
                // Step 1 Row 0 Phase 0: 4 MFMAs + 4 ds_read Br
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[30:33], a[0:3], v[166], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "ds_read_b128 v[94:97], %0 offset:0\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[38:41], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "ds_read_b128 v[98:101], %1 offset:0\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[46:49], a[8:11], v[166], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "ds_read_b128 v[102:105], %0 offset:2048\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:65], v[54:57], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "ds_read_b128 v[106:109], %1 offset:2048\n"
                // Step 1 Row 0 Phase 1: 4 MFMAs + 4 ds_read Br
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[34:37], a[0:3], v[166], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "ds_read_b128 v[110:113], %0 offset:4096\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[42:45], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "ds_read_b128 v[114:117], %1 offset:4096\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[50:53], a[8:11], v[166], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "ds_read_b128 v[118:121], %0 offset:6144\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[58:61], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "ds_read_b128 v[122:125], %1 offset:6144\n"
                // Step 1 Rows 1-3: 24 pure MFMAs + 8 buffer_loads for A1
                "buffer_load_dwordx4 v[126:129], %2, %3, %4 offen\n"
                "buffer_load_dwordx4 v[130:133], %2, %3, %5 offen\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:73], v[30:33], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:73], v[38:41], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:73], v[46:49], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[54:57], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[34:37], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:77], v[42:45], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "buffer_load_dwordx4 v[134:137], %2, %3, %6 offen\n"
                "buffer_load_dwordx4 v[138:141], %2, %3, %7 offen\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[50:53], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:77], v[58:61], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[30:33], a[32:35], v[167], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[38:41], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:81], v[46:49], a[40:43], v[167], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[78:81], v[54:57], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "buffer_load_dwordx4 v[142:145], %2, %3, %8 offen\n"
                "buffer_load_dwordx4 v[146:149], %2, %3, %9 offen\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[34:37], a[32:35], v[167], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:85], v[42:45], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[50:53], a[40:43], v[167], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[82:85], v[58:61], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[86:89], v[30:33], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[86:89], v[38:41], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "buffer_load_dwordx4 v[150:153], %2, %3, %10 offen\n"
                "buffer_load_dwordx4 v[154:157], %2, %3, %11 offen\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[86:89], v[46:49], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[86:89], v[54:57], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[90:93], v[34:37], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:93], v[42:45], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[90:93], v[50:53], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[58:61], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                // Wait for Br ds_reads
                "s_waitcnt lgkmcnt(0)\n"
                // Step 2: A0×Br (32 MFMAs) — Br data now in v[94:125]
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[62:65], v[94:97], a[64:67], v[166], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[62:65], v[102:105], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[62:65], v[110:113], a[72:75], v[166], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[62:65], v[118:121], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[98:101], a[64:67], v[166], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[66:69], v[106:109], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[66:69], v[114:117], a[72:75], v[166], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:69], v[122:125], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[70:73], v[94:97], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[70:73], v[102:105], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[70:73], v[110:113], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[70:73], v[118:121], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:77], v[98:101], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[74:77], v[106:109], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[74:77], v[114:117], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[74:77], v[122:125], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[94:97], a[96:99], v[167], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[102:105], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:81], v[110:113], a[104:107], v[167], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[118:121], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[98:101], a[96:99], v[167], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[106:109], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[114:117], a[104:107], v[167], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:85], v[122:125], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[86:89], v[94:97], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[86:89], v[102:105], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[86:89], v[110:113], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[86:89], v[118:121], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[90:93], v[98:101], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[90:93], v[106:109], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[90:93], v[114:117], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:93], v[122:125], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                : : "v"(sel_br_p0), "v"(sel_br_p1),
                    "v"(a_voffset_base), "s"(srd_a),
                    "s"(s0), "s"(s1), "s"(s2), "s"(s3),
                    "s"(s4), "s"(s5), "s"(s6), "s"(s7)
                : "memory"
            );
        }

        // Wait for A1 global loads + scale prefetches from Step 1
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

        // B prefetch to LDS for bt+2 (only emit_one_pf, no duplicate load_b_tiles)
        #pragma unroll
        for (int i = 0; i < PF_MPT; ++i) emit_one_pf(pf_bl_p, i);
        #pragma unroll
        for (int i = 0; i < PF_MPT; ++i) emit_one_pf(pf_br_p, i);

        // ═══ Steps 3+4: monolithic 64 MFMAs + 8 ds_read Bl[nxt] + 8 buffer_load A0[nxt] ═══
        // vmcnt(0) + barrier at asm entry for tighter scheduling
        {
            uint32_t nxt_a0_soff = __builtin_amdgcn_readfirstlane(
                a0_soff_base + (bt + 1 < k_byte_iters ? bt + 1 : bt) * BK);
            constexpr uint32_t ss = 16 * (K_DIM / 2);
            uint32_t s0 = nxt_a0_soff, s1 = nxt_a0_soff + 64;
            uint32_t s2 = nxt_a0_soff + ss, s3 = nxt_a0_soff + ss + 64;
            uint32_t s4 = nxt_a0_soff + 2*ss, s5 = nxt_a0_soff + 2*ss + 64;
            uint32_t s6 = nxt_a0_soff + 3*ss, s7 = nxt_a0_soff + 3*ss + 64;
            asm volatile(
                // Wait for B prefetches + barrier for LDS double-buffer
                "s_waitcnt vmcnt(0)\n"
                "s_barrier\n"
                // ── Step 3: A1×Bl (32 MFMAs) + 8 buffer_load A0[nxt]→v[62:93] ──
                // A1=v[126:157], Bl=v[30:61], acc=a[128:191], scales=v[168:169]/v[158:159]
                // Row 0 Phase 0: 4 MFMAs + 2 buffer_load A0[nxt]
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[126:129], v[30:33], a[128:131], v[168], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[126:129], v[38:41], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "buffer_load_dwordx4 v[62:65], %2, %3, %4 offen\n"
                "buffer_load_dwordx4 v[66:69], %2, %3, %5 offen\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[126:129], v[46:49], a[136:139], v[168], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[126:129], v[54:57], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                // Row 0 Phase 1: 4 MFMAs + 2 buffer_load A0[nxt]
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[130:133], v[34:37], a[128:131], v[168], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[130:133], v[42:45], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "buffer_load_dwordx4 v[70:73], %2, %3, %6 offen\n"
                "buffer_load_dwordx4 v[74:77], %2, %3, %7 offen\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[130:133], v[50:53], a[136:139], v[168], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[130:133], v[58:61], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                // Rows 1-3: 24 pure MFMAs + 4 more buffer_load A0[nxt]
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[134:137], v[30:33], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[134:137], v[38:41], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "buffer_load_dwordx4 v[78:81], %2, %3, %8 offen\n"
                "buffer_load_dwordx4 v[82:85], %2, %3, %9 offen\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[134:137], v[46:49], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[134:137], v[54:57], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[138:141], v[34:37], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[138:141], v[42:45], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "buffer_load_dwordx4 v[86:89], %2, %3, %10 offen\n"
                "buffer_load_dwordx4 v[90:93], %2, %3, %11 offen\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[138:141], v[50:53], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[138:141], v[58:61], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[142:145], v[30:33], a[160:163], v[169], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[142:145], v[38:41], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[142:145], v[46:49], a[168:171], v[169], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[142:145], v[54:57], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[146:149], v[34:37], a[160:163], v[169], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[146:149], v[42:45], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[146:149], v[50:53], a[168:171], v[169], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[146:149], v[58:61], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[150:153], v[30:33], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[150:153], v[38:41], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[150:153], v[46:49], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[150:153], v[54:57], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[154:157], v[34:37], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[154:157], v[42:45], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[154:157], v[50:53], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[154:157], v[58:61], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                // ── Step 4: A1×Br (32 MFMAs) + 8 ds_read Bl[nxt]→v[30:61] ──
                // A1=v[126:157], Br=v[94:125], acc=a[192:255], scales=v[168:169]/v[160:161]
                // Bl[nxt] ds_reads are SAFE here: Step 4 uses Br not Bl
                // Row 0 Phase 0: 4 MFMAs + 4 ds_read Bl[nxt]
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[126:129], v[94:97], a[192:195], v[168], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "ds_read_b128 v[30:33], %0 offset:0\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[126:129], v[102:105], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "ds_read_b128 v[34:37], %1 offset:0\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[126:129], v[110:113], a[200:203], v[168], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "ds_read_b128 v[38:41], %0 offset:2048\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[126:129], v[118:121], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "ds_read_b128 v[42:45], %1 offset:2048\n"
                // Row 0 Phase 1: 4 MFMAs + 4 ds_read Bl[nxt]
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[130:133], v[98:101], a[192:195], v[168], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "ds_read_b128 v[46:49], %0 offset:4096\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[130:133], v[106:109], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "ds_read_b128 v[50:53], %1 offset:4096\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[130:133], v[114:117], a[200:203], v[168], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "ds_read_b128 v[54:57], %0 offset:6144\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[130:133], v[122:125], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "ds_read_b128 v[58:61], %1 offset:6144\n"
                // Rows 1-3: 24 pure MFMAs
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[134:137], v[94:97], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[134:137], v[102:105], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[134:137], v[110:113], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[134:137], v[118:121], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[138:141], v[98:101], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[138:141], v[106:109], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[138:141], v[114:117], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[138:141], v[122:125], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[142:145], v[94:97], a[224:227], v[169], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[142:145], v[102:105], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[142:145], v[110:113], a[232:235], v[169], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[142:145], v[118:121], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[146:149], v[98:101], a[224:227], v[169], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[146:149], v[106:109], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[146:149], v[114:117], a[232:235], v[169], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[146:149], v[122:125], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[150:153], v[94:97], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[150:153], v[102:105], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[150:153], v[110:113], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[150:153], v[118:121], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[154:157], v[98:101], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[154:157], v[106:109], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[154:157], v[114:117], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[154:157], v[122:125], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4\n"
                : : "v"(nxt_bl_p0), "v"(nxt_bl_p1),
                    "v"(a_voffset_base), "s"(srd_a),
                    "s"(s0), "s"(s1), "s"(s2), "s"(s3),
                    "s"(s4), "s"(s5), "s"(s6), "s"(s7)
                : "memory"
            );
        }
        // Wait for Bl[nxt] ds_reads + A0[nxt] buffer_loads
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
    }

    // No debug code

    // ═══════════ Store C — scalar bf16 stores (same as original) ═══════════
    // Read accumulators, scale, convert to bf16, and store.

    auto store_block = [&](int acc_base_agpr, int mh, int nh) {
        const int tile_r = br * WARPS_M * 2 + WARPS_M * mh + wm;
        const int tile_c = bc * WARPS_N * 2 + WARPS_N * nh + wn;
        bf16 *dst_ptr = g.c.raw_ptr + static_cast<size_t>(tile_r * 64) * g.c.cols()
                        + static_cast<size_t>(tile_c * 64);
        const int row_stride = g.c.cols();
        const int row_off = 4 * (lid / 16);
        const int col_off = lid % 16;

        // Read output scale from v[29]
        float scale_val;
        asm volatile("v_mov_b32 %0, v[29]" : "=v"(scale_val));

        // Process 4x4 = 16 MFMA tiles
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                // Read 4 accumulators for this MFMA tile
                float f0, f1, f2, f3;
                const int agpr_idx = acc_base_agpr + i * 16 + j * 4;
                asm volatile(
                    "v_accvgpr_read_b32 %0, a[%4]\n"
                    "v_accvgpr_read_b32 %1, a[%5]\n"
                    "v_accvgpr_read_b32 %2, a[%6]\n"
                    "v_accvgpr_read_b32 %3, a[%7]\n"
                    : "=v"(f0), "=v"(f1), "=v"(f2), "=v"(f3)
                    : "n"(agpr_idx), "n"(agpr_idx+1), "n"(agpr_idx+2), "n"(agpr_idx+3));

                // Apply scale and convert to bf16
                f0 *= scale_val;
                f1 *= scale_val;
                f2 *= scale_val;
                f3 *= scale_val;

                const int row_base = i * 16 + row_off;
                const int col = j * 16 + col_off;
                dst_ptr[(row_base + 0) * row_stride + col] = base_types::convertor<bf16, float>::convert(f0);
                dst_ptr[(row_base + 1) * row_stride + col] = base_types::convertor<bf16, float>::convert(f1);
                dst_ptr[(row_base + 2) * row_stride + col] = base_types::convertor<bf16, float>::convert(f2);
                dst_ptr[(row_base + 3) * row_stride + col] = base_types::convertor<bf16, float>::convert(f3);
            }
        }
    };

    // ═══ Store C: read ALL 64 AGPRs per block in ONE asm, then process ═══
    // Root cause: compiler uses AGPRs as scratch during store, clobbering other blocks.
    // Fix: batch-read 16 AGPRs (one MFMA row) at a time in a single asm, keeping them
    // in compiler-managed VGPRs. Process immediately, then move to next row.
    // This prevents the compiler from touching AGPRs during the store processing.

    auto store_block_safe = [&](int acc_base, int mh, int nh) {
        const int tile_r = br * WARPS_M * 2 + WARPS_M * mh + wm;
        const int tile_c = bc * WARPS_N * 2 + WARPS_N * nh + wn;
        bf16 *dst_ptr = g.c.raw_ptr + static_cast<size_t>(tile_r * 64) * g.c.cols()
                        + static_cast<size_t>(tile_c * 64);
        const int row_stride = g.c.cols();
        const int row_off = 4 * (lid / 16);
        const int col_off = lid % 16;

        // Read ALL 64 AGPRs and multiply by scale in ONE asm block.
        // This prevents compiler AGPR interference AND folds the scale multiply
        // into the asm, eliminating 64 compiler v_mul_f32 instructions.
        // The outputs are pre-scaled floats; the C++ loop only does bf16 convert + store.
        float vals[64];
        const float sc = g.scale;
        asm volatile(
            "v_accvgpr_read_b32 %0, a[%64]\n"   "v_accvgpr_read_b32 %1, a[%65]\n"
            "v_accvgpr_read_b32 %2, a[%66]\n"   "v_accvgpr_read_b32 %3, a[%67]\n"
            "v_accvgpr_read_b32 %4, a[%68]\n"   "v_accvgpr_read_b32 %5, a[%69]\n"
            "v_accvgpr_read_b32 %6, a[%70]\n"   "v_accvgpr_read_b32 %7, a[%71]\n"
            "v_accvgpr_read_b32 %8, a[%72]\n"   "v_accvgpr_read_b32 %9, a[%73]\n"
            "v_accvgpr_read_b32 %10, a[%74]\n"  "v_accvgpr_read_b32 %11, a[%75]\n"
            "v_accvgpr_read_b32 %12, a[%76]\n"  "v_accvgpr_read_b32 %13, a[%77]\n"
            "v_accvgpr_read_b32 %14, a[%78]\n"  "v_accvgpr_read_b32 %15, a[%79]\n"
            "v_accvgpr_read_b32 %16, a[%80]\n"  "v_accvgpr_read_b32 %17, a[%81]\n"
            "v_accvgpr_read_b32 %18, a[%82]\n"  "v_accvgpr_read_b32 %19, a[%83]\n"
            "v_accvgpr_read_b32 %20, a[%84]\n"  "v_accvgpr_read_b32 %21, a[%85]\n"
            "v_accvgpr_read_b32 %22, a[%86]\n"  "v_accvgpr_read_b32 %23, a[%87]\n"
            "v_accvgpr_read_b32 %24, a[%88]\n"  "v_accvgpr_read_b32 %25, a[%89]\n"
            "v_accvgpr_read_b32 %26, a[%90]\n"  "v_accvgpr_read_b32 %27, a[%91]\n"
            "v_accvgpr_read_b32 %28, a[%92]\n"  "v_accvgpr_read_b32 %29, a[%93]\n"
            "v_accvgpr_read_b32 %30, a[%94]\n"  "v_accvgpr_read_b32 %31, a[%95]\n"
            "v_accvgpr_read_b32 %32, a[%96]\n"  "v_accvgpr_read_b32 %33, a[%97]\n"
            "v_accvgpr_read_b32 %34, a[%98]\n"  "v_accvgpr_read_b32 %35, a[%99]\n"
            "v_accvgpr_read_b32 %36, a[%100]\n" "v_accvgpr_read_b32 %37, a[%101]\n"
            "v_accvgpr_read_b32 %38, a[%102]\n" "v_accvgpr_read_b32 %39, a[%103]\n"
            "v_accvgpr_read_b32 %40, a[%104]\n" "v_accvgpr_read_b32 %41, a[%105]\n"
            "v_accvgpr_read_b32 %42, a[%106]\n" "v_accvgpr_read_b32 %43, a[%107]\n"
            "v_accvgpr_read_b32 %44, a[%108]\n" "v_accvgpr_read_b32 %45, a[%109]\n"
            "v_accvgpr_read_b32 %46, a[%110]\n" "v_accvgpr_read_b32 %47, a[%111]\n"
            "v_accvgpr_read_b32 %48, a[%112]\n" "v_accvgpr_read_b32 %49, a[%113]\n"
            "v_accvgpr_read_b32 %50, a[%114]\n" "v_accvgpr_read_b32 %51, a[%115]\n"
            "v_accvgpr_read_b32 %52, a[%116]\n" "v_accvgpr_read_b32 %53, a[%117]\n"
            "v_accvgpr_read_b32 %54, a[%118]\n" "v_accvgpr_read_b32 %55, a[%119]\n"
            "v_accvgpr_read_b32 %56, a[%120]\n" "v_accvgpr_read_b32 %57, a[%121]\n"
            "v_accvgpr_read_b32 %58, a[%122]\n" "v_accvgpr_read_b32 %59, a[%123]\n"
            "v_accvgpr_read_b32 %60, a[%124]\n" "v_accvgpr_read_b32 %61, a[%125]\n"
            "v_accvgpr_read_b32 %62, a[%126]\n" "v_accvgpr_read_b32 %63, a[%127]\n"
            // Scale all 64 values by sc (%128)
            "v_mul_f32 %0, %128, %0\n"   "v_mul_f32 %1, %128, %1\n"
            "v_mul_f32 %2, %128, %2\n"   "v_mul_f32 %3, %128, %3\n"
            "v_mul_f32 %4, %128, %4\n"   "v_mul_f32 %5, %128, %5\n"
            "v_mul_f32 %6, %128, %6\n"   "v_mul_f32 %7, %128, %7\n"
            "v_mul_f32 %8, %128, %8\n"   "v_mul_f32 %9, %128, %9\n"
            "v_mul_f32 %10, %128, %10\n" "v_mul_f32 %11, %128, %11\n"
            "v_mul_f32 %12, %128, %12\n" "v_mul_f32 %13, %128, %13\n"
            "v_mul_f32 %14, %128, %14\n" "v_mul_f32 %15, %128, %15\n"
            "v_mul_f32 %16, %128, %16\n" "v_mul_f32 %17, %128, %17\n"
            "v_mul_f32 %18, %128, %18\n" "v_mul_f32 %19, %128, %19\n"
            "v_mul_f32 %20, %128, %20\n" "v_mul_f32 %21, %128, %21\n"
            "v_mul_f32 %22, %128, %22\n" "v_mul_f32 %23, %128, %23\n"
            "v_mul_f32 %24, %128, %24\n" "v_mul_f32 %25, %128, %25\n"
            "v_mul_f32 %26, %128, %26\n" "v_mul_f32 %27, %128, %27\n"
            "v_mul_f32 %28, %128, %28\n" "v_mul_f32 %29, %128, %29\n"
            "v_mul_f32 %30, %128, %30\n" "v_mul_f32 %31, %128, %31\n"
            "v_mul_f32 %32, %128, %32\n" "v_mul_f32 %33, %128, %33\n"
            "v_mul_f32 %34, %128, %34\n" "v_mul_f32 %35, %128, %35\n"
            "v_mul_f32 %36, %128, %36\n" "v_mul_f32 %37, %128, %37\n"
            "v_mul_f32 %38, %128, %38\n" "v_mul_f32 %39, %128, %39\n"
            "v_mul_f32 %40, %128, %40\n" "v_mul_f32 %41, %128, %41\n"
            "v_mul_f32 %42, %128, %42\n" "v_mul_f32 %43, %128, %43\n"
            "v_mul_f32 %44, %128, %44\n" "v_mul_f32 %45, %128, %45\n"
            "v_mul_f32 %46, %128, %46\n" "v_mul_f32 %47, %128, %47\n"
            "v_mul_f32 %48, %128, %48\n" "v_mul_f32 %49, %128, %49\n"
            "v_mul_f32 %50, %128, %50\n" "v_mul_f32 %51, %128, %51\n"
            "v_mul_f32 %52, %128, %52\n" "v_mul_f32 %53, %128, %53\n"
            "v_mul_f32 %54, %128, %54\n" "v_mul_f32 %55, %128, %55\n"
            "v_mul_f32 %56, %128, %56\n" "v_mul_f32 %57, %128, %57\n"
            "v_mul_f32 %58, %128, %58\n" "v_mul_f32 %59, %128, %59\n"
            "v_mul_f32 %60, %128, %60\n" "v_mul_f32 %61, %128, %61\n"
            "v_mul_f32 %62, %128, %62\n" "v_mul_f32 %63, %128, %63\n"
            : "=&v"(vals[0]),  "=&v"(vals[1]),  "=&v"(vals[2]),  "=&v"(vals[3]),
              "=&v"(vals[4]),  "=&v"(vals[5]),  "=&v"(vals[6]),  "=&v"(vals[7]),
              "=&v"(vals[8]),  "=&v"(vals[9]),  "=&v"(vals[10]), "=&v"(vals[11]),
              "=&v"(vals[12]), "=&v"(vals[13]), "=&v"(vals[14]), "=&v"(vals[15]),
              "=&v"(vals[16]), "=&v"(vals[17]), "=&v"(vals[18]), "=&v"(vals[19]),
              "=&v"(vals[20]), "=&v"(vals[21]), "=&v"(vals[22]), "=&v"(vals[23]),
              "=&v"(vals[24]), "=&v"(vals[25]), "=&v"(vals[26]), "=&v"(vals[27]),
              "=&v"(vals[28]), "=&v"(vals[29]), "=&v"(vals[30]), "=&v"(vals[31]),
              "=&v"(vals[32]), "=&v"(vals[33]), "=&v"(vals[34]), "=&v"(vals[35]),
              "=&v"(vals[36]), "=&v"(vals[37]), "=&v"(vals[38]), "=&v"(vals[39]),
              "=&v"(vals[40]), "=&v"(vals[41]), "=&v"(vals[42]), "=&v"(vals[43]),
              "=&v"(vals[44]), "=&v"(vals[45]), "=&v"(vals[46]), "=&v"(vals[47]),
              "=&v"(vals[48]), "=&v"(vals[49]), "=&v"(vals[50]), "=&v"(vals[51]),
              "=&v"(vals[52]), "=&v"(vals[53]), "=&v"(vals[54]), "=&v"(vals[55]),
              "=&v"(vals[56]), "=&v"(vals[57]), "=&v"(vals[58]), "=&v"(vals[59]),
              "=&v"(vals[60]), "=&v"(vals[61]), "=&v"(vals[62]), "=&v"(vals[63])
            : "n"(acc_base+0),  "n"(acc_base+1),  "n"(acc_base+2),  "n"(acc_base+3),
              "n"(acc_base+4),  "n"(acc_base+5),  "n"(acc_base+6),  "n"(acc_base+7),
              "n"(acc_base+8),  "n"(acc_base+9),  "n"(acc_base+10), "n"(acc_base+11),
              "n"(acc_base+12), "n"(acc_base+13), "n"(acc_base+14), "n"(acc_base+15),
              "n"(acc_base+16), "n"(acc_base+17), "n"(acc_base+18), "n"(acc_base+19),
              "n"(acc_base+20), "n"(acc_base+21), "n"(acc_base+22), "n"(acc_base+23),
              "n"(acc_base+24), "n"(acc_base+25), "n"(acc_base+26), "n"(acc_base+27),
              "n"(acc_base+28), "n"(acc_base+29), "n"(acc_base+30), "n"(acc_base+31),
              "n"(acc_base+32), "n"(acc_base+33), "n"(acc_base+34), "n"(acc_base+35),
              "n"(acc_base+36), "n"(acc_base+37), "n"(acc_base+38), "n"(acc_base+39),
              "n"(acc_base+40), "n"(acc_base+41), "n"(acc_base+42), "n"(acc_base+43),
              "n"(acc_base+44), "n"(acc_base+45), "n"(acc_base+46), "n"(acc_base+47),
              "n"(acc_base+48), "n"(acc_base+49), "n"(acc_base+50), "n"(acc_base+51),
              "n"(acc_base+52), "n"(acc_base+53), "n"(acc_base+54), "n"(acc_base+55),
              "n"(acc_base+56), "n"(acc_base+57), "n"(acc_base+58), "n"(acc_base+59),
              "n"(acc_base+60), "n"(acc_base+61), "n"(acc_base+62), "n"(acc_base+63),
              "v"(sc)
        );

        // vals[] already pre-multiplied by scale — just convert to bf16 and store
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const int idx = i * 16 + j * 4;
                const int row_base = i * 16 + row_off;
                const int col = j * 16 + col_off;
                dst_ptr[(row_base+0)*row_stride+col] = base_types::convertor<bf16, float>::convert(vals[idx]);
                dst_ptr[(row_base+1)*row_stride+col] = base_types::convertor<bf16, float>::convert(vals[idx+1]);
                dst_ptr[(row_base+2)*row_stride+col] = base_types::convertor<bf16, float>::convert(vals[idx+2]);
                dst_ptr[(row_base+3)*row_stride+col] = base_types::convertor<bf16, float>::convert(vals[idx+3]);
            }
        }
    };

    store_block_safe(0,   0, 0);
    store_block_safe(64,  0, 1);
    store_block_safe(128, 1, 0);
    store_block_safe(192, 1, 1);
}

void dispatch_art(gluon_globals g) {
    int m = static_cast<int>(g.c.rows());
    int n = static_cast<int>(g.c.cols());
    const dim3 grid((m / BLK) * (n / BLK));
    mxfp4_art_kernel<<<grid, dim3(_NUM_THREADS), 0>>>(g);
}

PYBIND11_MODULE(tk_mxfp4_art, m) {
    m.doc() = "MXFP4 ART kernel (explicit register allocation)";
    py::bind_function<dispatch_art>(m, "gemm_rcr",
        &gluon_globals::a, &gluon_globals::b,
        &gluon_globals::a_scale, &gluon_globals::b_scale,
        &gluon_globals::c);
}
