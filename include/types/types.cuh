/**
 * @file
 * @brief An aggregate header file for all the register and shared types defined by ThunderKittens.
 */

#pragma once

#include "register/register.cuh"
#include "shared/shared.cuh"
#include "global/global.cuh"

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

namespace kittens {

/**
 * @brief Row vector type alias.
 *
 * This template alias provides a convenient way to refer to the row vector type
 * associated with a given class or type `T`. It assumes that the class `T` has
 * a nested type named `row_vec`.
 *
 * @tparam T The class or type for which the row vector type is defined.
 *
 * Example usage:
 * @code
 * kittens::row_vec<decltype(some_tile)> row_vector;
 * @endcode
 */
template<typename T>
using row_vec = T::row_vec;

/**
 * @brief Column vector type alias.
 *
 * This template alias provides a convenient way to refer to the column vector type
 * associated with a given class or type `T`. It assumes that the class `T` has
 * a nested type named `col_vec`.
 *
 * @tparam T The class or type for which the column vector type is defined.
 *
 * Example usage:
 * @code
 * kittens::col_vec<decltype(some_tile)> col_vector;
 * @endcode
 */
template<typename T>
using col_vec = T::col_vec;

// ^ this code lives here because it applies to both sv and rv types

// register tile layouts
using row_l = ducks::rt_layout::row;
using col_l = ducks::rt_layout::col;

// register vector layouts
using align_l = ducks::rv_layout::align;
using ortho_l = ducks::rv_layout::ortho;
using naive_l = ducks::rv_layout::naive;

// register tile shapes
using rt_16x16_s = ducks::rt_shape::rt_16x16;
using rt_32x32_s = ducks::rt_shape::rt_32x32;
using rt_32x32_8_s = ducks::rt_shape::rt_32x32_8;
using rt_16x32_s = ducks::rt_shape::rt_16x32;
using rt_32x16_s = ducks::rt_shape::rt_32x16;
using rt_32x16_4_s = ducks::rt_shape::rt_32x16_4;
using rt_16x32_4_s = ducks::rt_shape::rt_16x32_4;
using rt_16x128_s = ducks::rt_shape::rt_16x128;
using rt_128x16_s = ducks::rt_shape::rt_128x16;
// Round-26-dm (auto-optimize R29 / Lever D Round-A step 1):
// Public ``_s`` aliases for the FP8 32x32x64 MFMA cell-shape family.
// The underlying ``rt_32x64`` / ``rt_64x32`` rt_shape structs were
// added to ``rt_shape.cuh`` in R14-dm and are already enumerated in
// ``ducks::rt_shape::all``; the ``mma_AB_base`` / ``mma_ABt_base``
// 32x32x64 dispatch branches in ``ops/warp/register/tile/mma.cuh``
// already reference these shapes via the ``ducks::rt_shape::rt_32x32``
// + A_rows == 32 + A_cols == 64 (etc.) compile-time predicates. The
// missing ``_s`` aliases here meant downstream kernels (in
// ``analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp``) could not
// declare ``rt_fp8e4m3<R, C, row_l, rt_32x64_s>`` etc. without
// referencing the inner namespace path explicitly. Defining the
// aliases is purely cosmetic — no codegen change for any existing
// caller (these shapes have no callers yet); subsequent rounds wire
// in the actual K-tail port using these aliases.
using rt_32x64_s = ducks::rt_shape::rt_32x64;
using rt_64x32_s = ducks::rt_shape::rt_64x32;

// shared tile shapes
using st_16x16_s = ducks::st_shape::st_16x16;
using st_16x16_swizzled_s = ducks::st_shape::st_16x16_swizzled;
using st_32x32_s = ducks::st_shape::st_32x32;
using st_16x32_s = ducks::st_shape::st_16x32;
using st_32x16_s = ducks::st_shape::st_32x16;
using st_8x32_s = ducks::st_shape::st_8x32;
using st_16x128_s = ducks::st_shape::st_16x128;
using st_16x128_v2_s = ducks::st_shape::st_16x128_v2;
using st_16x128_v2a_s = ducks::st_shape::st_16x128_v2a;
using st_16x128_v3_s = ducks::st_shape::st_16x128_v3;
using st_64x32_padded_b128_s = ducks::st_shape::st_64x32_padded_b128;
// Lever D Round-B step 1 (auto-optimize R37 / dm-R64):
// Public alias for the 32x64 FP8 shared-memory tile layout used by the
// prospective 32x32x64 MFMA cell-shape migration. The underlying
// ``st_32x64`` struct is defined in ``st_shape.cuh`` with identity
// swizzle for infrastructure validation; R38+ will either refine the
// swizzle in place or spawn peer variants (``st_32x64_v2`` etc.)
// once the bank-conflict-free layout is derived from the mfma_323264
// input lane map. Pairs with ``rt_32x64_s`` / ``rt_64x32_s`` register
// tile aliases above.
using st_32x64_s = ducks::st_shape::st_32x64;
using st_128x16_s = ducks::st_shape::st_128x16;

}
