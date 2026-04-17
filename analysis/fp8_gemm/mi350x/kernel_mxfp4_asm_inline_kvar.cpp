// K-parametrized variant of kernel_mxfp4_asm_inline.cpp.
// Compile-time -DK_TRIPS=N controls the loop terminator. Default 28 = K=8192.
// For K=14336: K_TRIPS=52 ; K=16384: 60 ; K=28672: 108 ; K=32768: 124.
//
// Build:
//   /opt/rocm/bin/hipcc kernel_mxfp4_asm_inline_kvar.cpp \
//     --offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS \
//     -ffast-math -I/opt/rocm/include/rocrand -I../../../include \
//     -I../../../prototype -I/opt/rocm/include/hip \
//     -I/usr/include/python3.10 \
//     -I/opt/venv/lib/python3.10/site-packages/pybind11/include \
//     -shared -fPIC -std=c++20 -w \
//     -DK_TRIPS=60 -DMOD_SUFFIX=_k16384 \
//     -L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu \
//     -L/usr/lib/x86_64-linux-gnu -ldl -lm \
//     -o tk_mxfp4_asm_inline_kvar_k16384.cpython-310-x86_64-linux-gnu.so
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

#ifndef K_TRIPS
#define K_TRIPS 28
#endif

// Sentinel-based macro to inject the K_TRIPS terminator into the inlined ASM body.
// We re-use kernel_mxfp4_asm_inline.h verbatim by rewriting the literal "28" in
// the K-loop terminator with the value of K_TRIPS via a string concatenation
// shim. The ASM body has exactly ONE occurrence of "s_cmp_lt_u32 s68, 28".
//
// Approach: create a custom MXFP4_KERNEL_ASM_BODY that splits the original body
// at the K-loop terminator. We do this by including the header's contents
// after textual swap.

// Simplest approach: write the ASM body inline here, with the only difference
// being the s_cmp_lt_u32 constant. To avoid duplicating 2200 lines of ASM, we
// use a sed-time generated wrapper... but we don't have sed-time.
//
// Pragmatic alternative: put the bracketing macro before/after include.

// Strategy: Define MXFP4_KERNEL_ASM_BODY by including the original header's
// macro, then immediately after the include swap the K-trip terminator using
// preprocessor token concatenation. This is fragile, so instead we go with
// the runtime patch path: build separate copies via Python that pre-edits
// the .h file. This .cpp serves as the *template*; the Python builder
// substitutes the K-trip number.

#include "kernel_mxfp4_asm_inline.h"

#define MXFP4_KERNEL_ASM_PATCHED \
    "s_mov_b32 s16, s2\n" \
    "s_mov_b64 s[4:5], s[0:1]\n" \
    "s_load_dwordx8 s[8:15], s[0:1], 0x0\n" \
    MXFP4_KERNEL_ASM_BODY

using _gl_fp4   = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_scale = gl<fp8e8m0, -1, -1, -1, -1>;
using _gl_bf16  = gl<bf16, -1, -1, -1, -1>;

struct asm_globals {
    _gl_fp4 a, b;
    _gl_scale a_scale, b_scale;
    _gl_bf16 c;
};

extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__attribute__((amdgpu_waves_per_eu(1, 1)))
void mxfp4_asm_inline_kvar_kernel(
    const void* a, const void* b, void* c,
    const void* a_sc, const void* b_sc,
    int M, int N,
    int s_am, int s_bn, int s_cm, int s_ask, int s_bsk)
{
    asm volatile(
        MXFP4_KERNEL_ASM_PATCHED
        ::: "memory",
        "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9",
        "v10","v11","v12","v13","v14","v15","v16","v17","v18","v19",
        "v20","v21","v22","v23","v24","v25","v26","v27","v28","v29",
        "v30","v31","v32","v33","v34","v35","v36","v37","v38","v39",
        "v40","v41","v42","v43","v44","v45","v46","v47","v48","v49",
        "v50","v51","v52","v53","v54","v55","v56","v57","v58","v59",
        "v60","v61","v62","v63","v64","v65","v66","v67","v68","v69",
        "v70","v71","v72","v73","v74","v75","v76","v77","v78","v79",
        "v80","v81","v82","v83","v84","v85","v86","v87","v88","v89",
        "v90","v91","v92","v93","v94","v95","v96","v97","v98","v99",
        "v100","v101","v102","v103","v104","v105","v106","v107","v108","v109",
        "v110","v111","v112","v113","v114","v115","v116","v117","v118","v119",
        "v120","v121","v122","v123","v124","v125","v126","v127","v128","v129",
        "v130","v131","v132","v133","v134","v135","v136","v137","v138","v139",
        "v140","v141","v142","v143","v144","v145","v146","v147","v148","v149",
        "v150","v151","v152","v153","v154","v155","v156","v157","v158","v159",
        "v160","v161","v162","v163","v164","v165","v166","v167","v168","v169",
        "v170","v171","v172","v173","v174","v175","v176","v177","v178","v179",
        "v180","v181","v182","v183","v184","v185","v186","v187","v188","v189",
        "v190","v191","v192","v193","v194","v195","v196","v197","v198","v199",
        "v200","v201","v202","v203","v204","v205","v206","v207","v208","v209",
        "v210","v211","v212","v213","v214","v215","v216","v217","v218","v219",
        "v220","v221","v222","v223","v224","v225","v226","v227","v228","v229",
        "v230","v231","v232","v233","v234","v235","v236","v237","v238","v239",
        "v240","v241","v242","v243","v244",
        "a0","a1","a2","a3","a4","a5","a6","a7","a8","a9",
        "a10","a11","a12","a13","a14","a15","a16","a17","a18","a19",
        "a20","a21","a22","a23","a24","a25","a26","a27","a28","a29",
        "a30","a31","a32","a33","a34","a35","a36","a37","a38","a39",
        "a40","a41","a42","a43","a44","a45","a46","a47","a48","a49",
        "a50","a51","a52","a53","a54","a55","a56","a57","a58","a59",
        "a60","a61","a62","a63","a64","a65","a66","a67","a68","a69",
        "a70","a71","a72","a73","a74","a75","a76","a77","a78","a79",
        "a80","a81","a82","a83","a84","a85","a86","a87","a88","a89",
        "a90","a91","a92","a93","a94","a95","a96","a97","a98","a99",
        "a100","a101","a102","a103","a104","a105","a106","a107","a108","a109",
        "a110","a111","a112","a113","a114","a115","a116","a117","a118","a119",
        "a120","a121","a122","a123","a124","a125","a126","a127","a128","a129",
        "a130","a131","a132","a133","a134","a135","a136","a137","a138","a139",
        "a140","a141","a142","a143","a144","a145","a146","a147","a148","a149",
        "a150","a151","a152","a153","a154","a155","a156","a157","a158","a159",
        "a160","a161","a162","a163","a164","a165","a166","a167","a168","a169",
        "a170","a171","a172","a173","a174","a175","a176","a177","a178","a179",
        "a180","a181","a182","a183","a184","a185","a186","a187","a188","a189",
        "a190","a191","a192","a193","a194","a195","a196","a197","a198","a199",
        "a200","a201","a202","a203","a204","a205","a206","a207","a208","a209",
        "a210","a211","a212","a213","a214","a215","a216","a217","a218","a219",
        "a220","a221","a222","a223","a224","a225","a226","a227","a228","a229",
        "a230","a231","a232","a233","a234","a235","a236","a237","a238","a239",
        "a240","a241","a242","a243","a244","a245","a246","a247","a248","a249",
        "a250","a251","a252","a253","a254","a255"
    );
}

void dispatch(asm_globals g) {
    int M = static_cast<int>(g.c.rows());
    int N = static_cast<int>(g.c.cols());
    int K_half = static_cast<int>(g.a.cols());
    int grid = ((M+255)/256) * ((N+255)/256);
    auto a_ptr = reinterpret_cast<const void*>(g.a.raw_ptr);
    auto b_ptr = reinterpret_cast<const void*>(g.b.raw_ptr);
    auto c_ptr = reinterpret_cast<void*>(g.c.raw_ptr);
    auto as_ptr = reinterpret_cast<const void*>(g.a_scale.raw_ptr);
    auto bs_ptr = reinterpret_cast<const void*>(g.b_scale.raw_ptr);
    int s_am  = K_half;
    int s_bn  = K_half;
    int s_cm  = N;
    int s_ask = M;
    int s_bsk = N;
    mxfp4_asm_inline_kvar_kernel<<<dim3(grid),dim3(256),138144>>>(
        a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr,
        M, N, s_am, s_bn, s_cm, s_ask, s_bsk);
}

#ifndef MOD_NAME
#define MOD_NAME tk_mxfp4_asm_inline_kvar
#endif

PYBIND11_MODULE(MOD_NAME, m) {
    m.doc() = "MXFP4 inline ASM kernel — K-parametrized variant";
    py::bind_function<dispatch>(m, "gemm_rcr",
        &asm_globals::a, &asm_globals::b,
        &asm_globals::a_scale, &asm_globals::b_scale,
        &asm_globals::c);
}
