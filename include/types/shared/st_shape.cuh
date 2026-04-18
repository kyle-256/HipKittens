/**
 * @file
 * @brief Layouts and their manipulations for shared tiles.
 */

#pragma once

#include <concepts>

namespace kittens {
namespace ducks {
/**
* @namespace st_shape
* 
* @brief A namespace for template metaprogramming with shared tile layouts.
*/
namespace st_shape {

struct st_16x16 {
    static constexpr int rows = 16;
    static constexpr int cols = 16;
    static constexpr int subtile_padding = 0;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __forceinline__ static const uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        using T = _T;

        const uint32_t offset = sizeof(T)*(r*cols + c);

        if constexpr (sizeof(T) == 2 || sizeof(T) == 4) {
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
        
    }
};

struct st_16x16_swizzled {
    static constexpr int rows = 16;
    static constexpr int cols = 16;
    static constexpr int subtile_padding = 0;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2) {
            return 4;
        } else if constexpr (sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __forceinline__ static const uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        using T = _T;

        const uint32_t offset = sizeof(T)*(r*cols + c);

        if constexpr (sizeof(T) == 2) {
            const int swizzle = ((offset % 512) >> 7) << 3;
            const int swizzled_offset = offset ^ swizzle;
            return swizzled_offset;
        } else if constexpr (sizeof(T) == 4) {
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
        
    }
};

struct st_32x32 {
    static constexpr int rows = 32;
    static constexpr int cols = 32;
    static constexpr int subtile_padding = 0;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __forceinline__ static const uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        using T = _T;

        const uint32_t offset = sizeof(T)*(r*cols + c);

        if constexpr (sizeof(T) == 2) {
            const int first_swizzle = ((offset % 1024) >> 9) << 5;
            const int second_swizzle = ((offset % 2048) >> 10) << 4;
            const int swizzled_offset = offset ^ first_swizzle ^ second_swizzle;
            return swizzled_offset;
        } else if constexpr (sizeof(T) == 4) {
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

struct st_16x32 {
    static constexpr int rows = 16;
    static constexpr int cols = 32;
    static constexpr int subtile_padding = 0;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __forceinline__ static const uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        using T = _T;

        const uint32_t offset = sizeof(T)*(r*cols + c);

        if constexpr (sizeof(T) == 2) {
            const int swizzle = ((offset % 1024) >> 9) << 5;
            const int swizzled_offset = offset ^ swizzle;
            return swizzled_offset;
        } else if constexpr (sizeof(T) == 4) {
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

struct st_32x16 {
    static constexpr int rows = 32;
    static constexpr int cols = 16;
    static constexpr int subtile_padding = 0;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __forceinline__ static const uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        using T = _T;

        const uint32_t offset = sizeof(T)*(r*cols + c);

        if constexpr (sizeof(T) == 2) {
            const int swizzle = ((offset % 1024) >> 9) << 4;
            const int swizzled_offset = offset ^ swizzle;
            return swizzled_offset;
        } else if constexpr (sizeof(T) == 4) {
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

struct st_8x32 {
    static constexpr int rows = 8;
    static constexpr int cols = 32;
    static constexpr int subtile_padding = 0;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __forceinline__ static const uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        using T = _T;

        const uint32_t offset = sizeof(T)*(r*cols + c);

        if constexpr (sizeof(T) == 2 || sizeof(T) == 4) {
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

struct st_16x128 {
    static constexpr int rows = 16;
    static constexpr int cols = 128;
    static constexpr int subtile_padding = 0;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 1) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __forceinline__ static const uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        using T = _T;

        const uint32_t offset = sizeof(T)*(r*cols + c);

        if constexpr (sizeof(T) == 1) {
            const int swizzle = ((offset % (16*128)) >> 8) << 4;
            const int swizzled_offset = offset ^ swizzle;
            return swizzled_offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

struct st_16x128_v2 {
    static constexpr int rows = 16;
    static constexpr int cols = 128;
    static constexpr int subtile_padding = 128;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 1) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __forceinline__ static const uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        using T = _T;

        const uint32_t offset = sizeof(T)*(r*cols + c);

        if constexpr (sizeof(T) == 1) {
            const int swizzle = ((offset >> 7) & 7) << 4;
            const int swizzled_offset = offset ^ swizzle;
            return swizzled_offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

struct st_16x128_v2a {
    static constexpr int rows = 16;
    static constexpr int cols = 128;
    static constexpr int subtile_padding = 128;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 1) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __forceinline__ static const uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        using T = _T;
        const uint32_t offset = sizeof(T)*(r*cols + c);
        if constexpr (sizeof(T) == 1) {
            const int swizzle = ((offset >> 7) & 7) << 4;
            const int swizzled_offset = offset ^ swizzle;
            return swizzled_offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

struct st_16x128_v3 {
    static constexpr int rows = 16;
    static constexpr int cols = 128;
    static constexpr int subtile_padding = 0;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 1) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __forceinline__ static const uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        using T = _T;

        const uint32_t offset = sizeof(T)*(r*cols + c);

        if constexpr (sizeof(T) == 1) {
            const int swizzle = (r & 15) << 3;
            const int swizzled_offset = offset ^ swizzle;
            return swizzled_offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

// P23 Dev B (BF16 RCR Route 1, padded-b128 design):
//   Identity-swizzle 64x32 BF16 subtile with 32-byte padding tacked onto
//   the END of each 4096-B subtile. The padding (not any within-subtile
//   permutation) is what breaks the 128-B LDS-bank alias for stride-128
//   ds_read_b128 on gfx950's 32-bank x 4-B banks.
//
//   Per-subtile bytes  : 64 * 32 * 2 = 4096
//   Per-subtile stride : 4096 + 32   = 4128
//   bytes_per_thread   : 16          (b128 DTL writes from g2s)
//
//   Used as st_bf<128, 64, st_64x32_padded_b128> for both ST_A and ST_B
//   in BF16 RCR (Route 1). Within-subtile swizzle is IDENTITY so that
//   prefill_swizzled_offsets's HBM voffset reconstruction at lines
//   149-151 of include/ops/warp/memory/tile/global_to_shared.cuh stays
//   bit-correct without a code change. Padding is applied between
//   subtiles via underlying_subtile_stride_bytes (see st.cuh:68,83 +
//   global_to_shared.cuh:60,98,212,236,305).
struct st_64x32_padded_b128 {
    static constexpr int rows = 64;
    static constexpr int cols = 32;
    static constexpr int subtile_padding = 32;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __forceinline__ static const uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        using T = _T;
        const uint32_t offset = sizeof(T)*(r*cols + c);
        if constexpr (sizeof(T) == 2) {
            // Identity: padding lives between subtiles, not inside.
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

struct st_128x16 {
    static constexpr int rows = 128;
    static constexpr int cols = 16;
    static constexpr int subtile_padding = 0;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 1) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __forceinline__ static const uint32_t swizzle (int2 coord) {
        using T = _T;
        const uint32_t offset = sizeof(T)*(coord.x*cols + coord.y);
        if constexpr (sizeof(T) == 1) {
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

template<typename T>
concept all = std::is_same_v<T, st_16x16> || 
              std::is_same_v<T, st_16x16_swizzled> || 
              std::is_same_v<T, st_32x32> || 
              std::is_same_v<T, st_16x32> || 
              std::is_same_v<T, st_32x16> || 
              std::is_same_v<T, st_8x32>  ||
              std::is_same_v<T, st_16x128> ||
              std::is_same_v<T, st_16x128_v2> ||
              std::is_same_v<T, st_16x128_v2a> ||
              std::is_same_v<T, st_16x128_v3> ||
              std::is_same_v<T, st_64x32_padded_b128> ||
              std::is_same_v<T, st_128x16>;


} // namespace st_shape
} // namespace ducks
} // namespace kittens