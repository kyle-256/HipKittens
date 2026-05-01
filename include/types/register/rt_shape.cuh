/**
 * @file
 * @brief Layouts and their manipulations for register tiles.
 */

#pragma once

#include <concepts>

namespace kittens {
namespace ducks {
/**
* @namespace rt_shape
* 
* @brief A namespace for template metaprogramming with register tile layouts.
* Assumption below is that the col is the reduction dimension
*/
namespace rt_shape {
 
template<int _rows, int _cols, int _stride>
struct rt_shape {
    static constexpr int rows = _rows;
    static constexpr int cols = _cols;
    static constexpr int stride = _stride;
    static constexpr int num_elements = rows*cols;
    static constexpr int elements_per_thread = num_elements / kittens::WARP_THREADS;
    static constexpr int num_strides = elements_per_thread / stride;
};

using rt_16x16 = rt_shape<16, 16, 4>;
using rt_32x32 = rt_shape<32, 32, 4>;
using rt_32x32_8 = rt_shape<32, 32, 8>;
using rt_16x32 = rt_shape<16, 32, 8>;
using rt_32x16 = rt_shape<32, 16, 8>;
using rt_32x16_4 = rt_shape<32, 16, 4>;
using rt_16x32_4 = rt_shape<16, 32, 4>;
using rt_16x128 = rt_shape<16, 128, 16>;
using rt_128x16 = rt_shape<128, 16, 16>;
// Round-14-dm: FP8 32x32x64 MFMA cell-shape scaffolding (multi-round
// migration plan — see Primus-Turbo notes/round-14-dm-fp8-mfma-cell-
// shape-scaffold.md). The mfma_AB(t)_base dispatchers in
// include/ops/warp/register/tile/mma.cuh already wire 32x32x64 to
// mfma323264 when A_rows==32 && A_cols==64 (and B_rows/cols match for
// the AB or ABt overload), but rt_32x64 / rt_64x32 were missing from
// concept all → the dispatcher branch was unreachable. Adding the
// shape types here unblocks subsequent rounds (15+) that migrate
// FP8 grouped's A_row_reg / B_row_reg / accumulator from
// rt_16x128_s / rt_128x16_s / rt_16x16_s to rt_32x64_s / rt_64x32_s
// / rt_32x32_s. Stride=16 mirrors the existing FP8 family (rt_16x128
// / rt_128x16) — fp8e4m3_4 packed (num_packed=4) gives
// elements_per_thread=32 (=32x64/64 lanes), stride=16 →
// packed_per_stride=4 and num_strides=2, identical pack ratio to the
// 16x128 path. Verified this round by completing a clean rebuild of
// kernel_fp8_layouts.cpp (no callers yet, no behaviour change for any
// existing kernel; the rt_shape::all concept widening admits the new
// types where the dispatcher's `if constexpr` shape predicate already
// enumerates rt_32x32 D + 32x64 A + 32x64 B for ABt and 32x64 A +
// 64x32 B for AB).
using rt_32x64 = rt_shape<32, 64, 16>;
using rt_64x32 = rt_shape<64, 32, 16>;

template<typename T>
concept all = std::is_same_v<T, rt_16x16> || 
              std::is_same_v<T, rt_32x32> || 
              std::is_same_v<T, rt_32x32_8> || 
              std::is_same_v<T, rt_16x32> || 
              std::is_same_v<T, rt_32x16> || 
              std::is_same_v<T, rt_32x16_4> || 
              std::is_same_v<T, rt_16x32_4> ||
              std::is_same_v<T, rt_16x128> ||
              std::is_same_v<T, rt_128x16> ||
              std::is_same_v<T, rt_32x64> ||
              std::is_same_v<T, rt_64x32>;

/**
 * @brief A struct to generate a transposed layout.
 * Note: on CDNA4, the accumulator layout becomes the col layout when transposed.
 */
 template<all L> struct transpose      { using type = rt_16x16; };
 template<>      struct transpose<rt_32x32> { using type = rt_32x32; };
 template<>      struct transpose<rt_32x32_8> { using type = rt_32x32_8; };
 template<>      struct transpose<rt_16x32> { using type = rt_32x16; };
 template<>      struct transpose<rt_32x16> { using type = rt_16x32; };
 template<>      struct transpose<rt_32x16_4> { using type = rt_16x32_4; };
 template<>      struct transpose<rt_16x32_4> { using type = rt_32x16_4; };
 template<>      struct transpose<rt_16x128> { using type = rt_128x16; };
 template<>      struct transpose<rt_128x16> { using type = rt_16x128; };
 template<>      struct transpose<rt_32x64>  { using type = rt_64x32;  };
 template<>      struct transpose<rt_64x32>  { using type = rt_32x64;  };
} // namespace rt_shape
} // namespace ducks
} // namespace kittens