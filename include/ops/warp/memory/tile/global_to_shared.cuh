/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

template<int axis, bool assume_aligned,
         ducks::st::all ST, ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>,
         int N_THREADS = WARP_THREADS>
__device__ inline void load(ST& dst, const GL& src, const COORD& idx)
{
    using T = typename ST::dtype;

    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    constexpr int memcpy_per_tile = ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);
    static_assert(ST::rows * ST::cols * sizeof(T) >= bytes_per_warp, "shared tile must be at least 1024 bytes");
    
    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int laneid = kittens::laneid();
    const int warpid = kittens::warpid() % num_warps;

    const int row_stride = src.template stride<axis>();

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* global_ptr = (T*)&src[unit_coord];
    i32x4 srsrc = make_srsrc(global_ptr, row_stride * ST::rows * sizeof(T));

    const uintptr_t lds_tile_base = reinterpret_cast<uintptr_t>(&dst.data[0]);

    if constexpr (memcpy_per_tile > 0) {

        #pragma unroll
        for (int i = 0; i < memcpy_per_tile; i++) {

            const int lane_byte_offset = (laneid * bytes_per_thread) + (warpid * bytes_per_warp) + (i * num_warps * bytes_per_warp);
            const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
            const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
            const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
            const int subtile_lane_byte_offset = lane_byte_offset % ST::underlying_subtile_bytes;

            const int row = subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
            const int col = (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);

            const uint32_t swizzled_shared_byte_offset = dst.swizzle({row, col});

            const int swizzled_global_row = (swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes) + subtile_row * ST::underlying_subtile_rows;
            const int swizzled_global_col = (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T) + subtile_col * ST::underlying_subtile_cols;
            const uint32_t swizzled_global_byte_offset = (swizzled_global_row * row_stride + swizzled_global_col) * sizeof(T);

            const int warp_linear_offset = (warpid * bytes_per_warp) + (i * num_warps * bytes_per_warp);
            const int lds_subtile_id = warp_linear_offset / ST::underlying_subtile_bytes;
            uintptr_t lds_addr = lds_tile_base + warp_linear_offset + lds_subtile_id * ST::subtile_padding;
            as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

            llvm_amdgcn_raw_buffer_load_lds(
                srsrc, // buffer resource
                lds_ptr,
                bytes_per_thread,
                swizzled_global_byte_offset,
                0, 
                0, // instruction offset
                static_cast<int>(coherency::cache_all)); // cache coherency
        }
    }
    // there are leftover loads that need to be handled here
    if constexpr (memcpy_per_tile * (bytes_per_thread * N_THREADS) != ST::rows * ST::cols * sizeof(T)) {

        constexpr int leftover_bytes = ST::rows * ST::cols * sizeof(T) - memcpy_per_tile * (bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;
        constexpr int leftover_warps = leftover_threads / kittens::WARP_THREADS;

        if (warpid < leftover_warps) {
            const int lane_byte_offset = (laneid * bytes_per_thread) + (warpid * bytes_per_warp) + (memcpy_per_tile * num_warps * bytes_per_warp);
            const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
            const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
            const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
            const int subtile_lane_byte_offset = lane_byte_offset % ST::underlying_subtile_bytes;

            const int row = subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
            const int col = (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);

            const uint32_t swizzled_shared_byte_offset = dst.swizzle({row, col});

            const int swizzled_global_row = (swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes) + subtile_row * ST::underlying_subtile_rows;
            const int swizzled_global_col = (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T) + subtile_col * ST::underlying_subtile_cols;
            const uint32_t swizzled_global_byte_offset = (swizzled_global_row * row_stride + swizzled_global_col) * sizeof(T);

            const int warp_linear_offset = (warpid * bytes_per_warp) + (memcpy_per_tile * num_warps * bytes_per_warp);
            const int lds_subtile_id = warp_linear_offset / ST::underlying_subtile_bytes;
            uintptr_t lds_addr = lds_tile_base + warp_linear_offset + lds_subtile_id * ST::subtile_padding;
            as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

            llvm_amdgcn_raw_buffer_load_lds(
                srsrc, // buffer resource
                lds_ptr,
                bytes_per_thread,
                swizzled_global_byte_offset,
                0, 
                0, // instruction offset
                static_cast<int>(coherency::cache_all)); // cache coherency
        }
    }
}

template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx) {
    load<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx);
}

template<int axis, bool assume_aligned,
         ducks::st::all ST, ducks::gl::all GL,
         int N_THREADS = WARP_THREADS>
__device__ inline void prefill_swizzled_offsets(
    ST& dst, const GL& src, uint32_t* swizzled_offsets)
{
    using T = typename ST::dtype;
 
    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    constexpr int memcpy_per_tile =  ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);
    static_assert(ST::rows * ST::cols * sizeof(T) >= bytes_per_warp, "shared tile must be at least 1024 bytes");

    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int laneid = kittens::laneid();
    const int warpid = kittens::warpid() % num_warps;

    const int row_stride = src.template stride<axis>();

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {
        const int lane_byte_offset = (laneid * bytes_per_thread) + (warpid * bytes_per_warp) + (i * num_warps * bytes_per_warp);
        const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
        const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
        const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
        const int subtile_lane_byte_offset = lane_byte_offset % ST::underlying_subtile_bytes;

        int row = subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
        int col = (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);
        const uint32_t swizzled_shared_byte_offset = dst.swizzle({row, col});

        const int swizzled_global_row = (swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes) + subtile_row * ST::underlying_subtile_rows;
        const int swizzled_global_col = (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T) + subtile_col * ST::underlying_subtile_cols;
        const uint32_t swizzled_global_byte_offset = (swizzled_global_row * row_stride + swizzled_global_col) * sizeof(T);
        swizzled_offsets[i] = swizzled_global_byte_offset;
    }

    // there are leftover loads that need to be handled here
    if constexpr (memcpy_per_tile * (bytes_per_thread * N_THREADS) != ST::rows * ST::cols * sizeof(T)) {

        constexpr int leftover_bytes = ST::rows * ST::cols * sizeof(T) - memcpy_per_tile * (bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;
        constexpr int leftover_warps = leftover_threads / kittens::WARP_THREADS;

        if (warpid < leftover_warps) {
            const int lane_byte_offset = (laneid * bytes_per_thread) + (warpid * bytes_per_warp) + (memcpy_per_tile * num_warps * bytes_per_warp);
            const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
            const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
            const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
            const int subtile_lane_byte_offset = lane_byte_offset % ST::underlying_subtile_bytes;

            const int row = subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
            const int col = (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);

            const uint32_t swizzled_shared_byte_offset = dst.swizzle({row, col});

            const int swizzled_global_row = (swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes) + subtile_row * ST::underlying_subtile_rows;
            const int swizzled_global_col = (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T) + subtile_col * ST::underlying_subtile_cols;
            const uint32_t swizzled_global_byte_offset = (swizzled_global_row * row_stride + swizzled_global_col) * sizeof(T);

            swizzled_offsets[memcpy_per_tile] = swizzled_global_byte_offset;
        }
    }
}

template<int axis, bool assume_aligned,
         ducks::st::all ST, ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>,
         int N_THREADS = WARP_THREADS>
__device__ inline void load(ST& dst, const GL& src, const COORD& idx, const uint32_t* swizzled_offsets)
{
    using T = typename ST::dtype;

    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    constexpr int elements_per_warp = bytes_per_warp / sizeof(T);
    constexpr int memcpy_per_tile = ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);
    static_assert(ST::rows * ST::cols * sizeof(T) >= bytes_per_warp, "shared tile must be at least 1024 bytes");
    
    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int warpid = kittens::warpid() % num_warps;

    const int row_stride = src.template stride<axis>();
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* global_ptr = (T*)&src[unit_coord];
    i32x4 srsrc = make_srsrc(global_ptr, row_stride * ST::rows * sizeof(T));

    const uintptr_t lds_tile_base2 = reinterpret_cast<uintptr_t>(&dst.data[0]);

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {

        const int warp_linear_offset = (warpid * bytes_per_warp) + (i * num_warps * bytes_per_warp);
        const int lds_subtile_id = warp_linear_offset / ST::underlying_subtile_bytes;
        uintptr_t lds_addr = lds_tile_base2 + warp_linear_offset + lds_subtile_id * ST::subtile_padding;
        as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

        llvm_amdgcn_raw_buffer_load_lds(
            srsrc, // buffer resource
            lds_ptr,
            bytes_per_thread,
            swizzled_offsets[i],
            0, 
            0, // instruction offset
            static_cast<int>(coherency::cache_all)); // cache coherency
    }

    // there are leftover loads that need to be handled here
    if constexpr (memcpy_per_tile * (bytes_per_thread * N_THREADS) != ST::rows * ST::cols * sizeof(T)) {

        constexpr int leftover_bytes = ST::rows * ST::cols * sizeof(T) - memcpy_per_tile * (bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;
        constexpr int leftover_warps = leftover_threads / kittens::WARP_THREADS;

        if (warpid < leftover_warps) {

            const int warp_linear_offset = (warpid * bytes_per_warp) + (memcpy_per_tile * num_warps * bytes_per_warp);
            const int lds_subtile_id = warp_linear_offset / ST::underlying_subtile_bytes;
            uintptr_t lds_addr = lds_tile_base2 + warp_linear_offset + lds_subtile_id * ST::subtile_padding;
            as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

            llvm_amdgcn_raw_buffer_load_lds(
                srsrc, // buffer resource
                lds_ptr,
                bytes_per_thread,
                swizzled_offsets[memcpy_per_tile],
                0, 
                0, // instruction offset
                static_cast<int>(coherency::cache_all)); // cache coherency
        }
    }
}

template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx, const uint32_t* swizzled_offsets) {
    load<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx, swizzled_offsets);
}


using as3_uint32_ptr = __attribute__((address_space(3))) unsigned int*;
inline __device__ __forceinline__ uint32_t to_sgpr_u32(uint32_t x) {
    x = __builtin_amdgcn_readfirstlane(x); // make uniform
    asm volatile("" : "+s"(x));            // keep in SGPR class
    return x;
}

template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<ST>, int N_THREADS = WARP_THREADS>
__attribute__((always_inline)) 
__device__ __forceinline__ void load(ST& dst, const GL& src, const COORD& idx,
                                const uint32_t* __restrict__ swizzled_offsets,
                                i32x4 SRD,
                                const void* base_ptr, const uint32_t lds_base)
{
    using T = typename ST::dtype;
    static_assert(sizeof(T) == 2 || sizeof(T) == 1, "only supporting 16 and 8-bit dtypes");

    constexpr int bytes_per_thread = 16;
    constexpr int bytes_per_memcpy = bytes_per_thread * N_THREADS;
    constexpr int memcpy_per_tile  = (ST::rows * ST::cols * sizeof(T)) / bytes_per_memcpy;
    static_assert(bytes_per_memcpy % 16 == 0, "LDS bump must be 16-aligned");

    constexpr int elem_per_thread = bytes_per_thread / sizeof(T);
    constexpr int elem_per_warp   = elem_per_thread * kittens::WARP_THREADS;

    // ---- compute per-tile base pointer and scalar offset (SOFF) ----
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* __restrict__ gptr = (T*)&src[unit_coord];

    uint32_t SOFF = to_sgpr_u32(static_cast<uint32_t>(
    reinterpret_cast<const char*>(gptr) - reinterpret_cast<const char*>(base_ptr)
    ));

    // // ---- LDS base (byte address) as SGPR (wave-uniform) ----
    // const int num_warps = N_THREADS / kittens::WARP_THREADS;
    // const int wid = warpid() % num_warps;
    // uint32_t lds_base = to_sgpr_u32(static_cast<uint32_t>(
    // reinterpret_cast<uintptr_t>(&dst.data[0]) + wid * elem_per_warp * sizeof(T)
    // ));

    const uint32_t lds_tile_base3 = to_sgpr_u32(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&dst.data[0])));
    const uint32_t warp_offset = lds_base - lds_tile_base3;

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; ++i) {
        const uint32_t linear_offset = warp_offset + i * bytes_per_memcpy;
        const uint32_t subtile_id_lds = linear_offset / ST::underlying_subtile_bytes;
        // Rebuild lds_byte through readfirstlane to coerce SGPR class without
        // the inline-asm "+s" clobber that breaks under clang 22.0.0
        // ("illegal VGPR to SGPR copy" backend error). readfirstlane produces
        // an SGPR result identical to what the asm clobber requested.
        const uint32_t lds_byte = __builtin_amdgcn_readfirstlane(
            lds_tile_base3 + linear_offset + subtile_id_lds * ST::subtile_padding);

        llvm_amdgcn_raw_buffer_load_lds(
            SRD, 
            (as3_uint32_ptr)(uintptr_t)lds_byte, 
            16, 
            swizzled_offsets[i], 
            SOFF, 
            0,
            static_cast<int>(coherency::cache_all)
        );
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx, const uint32_t* __restrict__ swizzled_offsets, i32x4 srd, const void* base_ptr, uint32_t lds_base) {
    load<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx, swizzled_offsets, srd, base_ptr, lds_base);
}

/**
 * @brief Stores data from a shared memory tile into global memory.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory tile.
 * @param row_stride[in] The stride between rows in the destination array.
 */

template<int axis, bool assume_aligned, 
        ducks::st::all ST, ducks::gl::all GL, 
        ducks::coord::tile COORD=coord<ST>, int N_THREADS=WARP_THREADS>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {

    using T = typename ST::dtype;
    using U = typename GL::dtype;

    static_assert(std::is_same_v<T, U>, "T and U must be the same type");
    static_assert(!std::is_same_v<T, fp8e4m3>, "Unsupported type for store");

    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int elems_per_thread = bytes_per_thread / sizeof(T);
    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    constexpr int memcpy_per_tile =  ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);

    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int laneid = kittens::laneid();
    const int warpid = kittens::warpid() % num_warps;

    const int row_stride = dst.template stride<axis>();

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    uintptr_t dst_ptr = reinterpret_cast<uintptr_t>(&dst[unit_coord]);
    uintptr_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);

    if constexpr (memcpy_per_tile > 0) {

        #pragma unroll
        for (int i = 0; i < memcpy_per_tile; i++) {
            const int lane_byte_offset = (laneid * bytes_per_thread) + (warpid * bytes_per_warp) + (i * num_warps * bytes_per_warp);
            const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
            const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
            const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
            const int subtile_lane_byte_offset = lane_byte_offset % ST::underlying_subtile_bytes;

            const int row = subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
            const int col = (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);
            const uint32_t swizzled_shared_byte_offset = src.swizzle({row, col});

            const int swizzled_global_row = (swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes) + subtile_row * ST::underlying_subtile_rows;
            const int swizzled_global_col = (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T) + subtile_col * ST::underlying_subtile_cols;
            const uint32_t swizzled_global_byte_offset = (swizzled_global_row * row_stride + swizzled_global_col) * sizeof(T);

            U* dst_elem_ptr = (U*)(dst_ptr + swizzled_global_byte_offset);
            T* src_elem_ptr = (T*)(src_ptr + lane_byte_offset);

            #pragma unroll
            for (int j = 0; j < elems_per_thread; j++) {
                dst_elem_ptr[j] = kittens::base_types::convertor<U, T>::convert(src_elem_ptr[j]);
            }
        }
    }

    if constexpr (memcpy_per_tile * (bytes_per_thread * N_THREADS) != ST::rows * ST::cols * sizeof(T)) {

        constexpr int leftover_bytes = ST::rows * ST::cols * sizeof(T) - memcpy_per_tile * (bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;
        constexpr int leftover_warps = leftover_threads / kittens::WARP_THREADS;

        if (warpid < leftover_warps) {
            const int lane_byte_offset = (laneid * bytes_per_thread) + (warpid * bytes_per_warp) + (memcpy_per_tile * num_warps * bytes_per_warp);
            const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
            const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
            const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
            const int subtile_lane_byte_offset = lane_byte_offset % ST::underlying_subtile_bytes;

            const int row = subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
            const int col = (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);
            const uint32_t swizzled_shared_byte_offset = src.swizzle({row, col});

            const int swizzled_global_row = (swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes) + subtile_row * ST::underlying_subtile_rows;
            const int swizzled_global_col = (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T) + subtile_col * ST::underlying_subtile_cols;
            const uint32_t swizzled_global_byte_offset = (swizzled_global_row * row_stride + swizzled_global_col) * sizeof(T);

            U* dst_elem_ptr = (U*)(dst_ptr + swizzled_global_byte_offset);
            T* src_elem_ptr = (T*)(src_ptr + lane_byte_offset);

            #pragma unroll
            for (int j = 0; j < elems_per_thread; j++) {
                dst_elem_ptr[j] = kittens::base_types::convertor<U, T>::convert(src_elem_ptr[j]);
            }
        }
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    store<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx);
}

// =============================================================================
// Session 9 (RRR v2 B-pretranspose): Path L HBM→LDS B-transpose writer.
// =============================================================================
//
// Promotes the Session 4 probe writer (tests/probes/rrr_b_writer_probe.cu)
// into a header API so kernel-body code can call it without re-implementing
// the transpose. Hard-wired for a 128 K × 128 N tile (one st_128x128_n_major
// instance), 512-thread WG (8 warps), fp8e4m3 data.
//
// Caller responsibilities:
//   - `dst_n_major`  : an LDS tile of type ST = st_fp8e4m3<128,128,
//                      st_128x128_n_major_s> (16 KB, identity swizzle).
//   - `hbm_b_tile_ptr` : first byte of the desired 128 K × 128 N tile in
//                       HBM. Caller computes (k_start*hbm_k_stride + n_start).
//                       Tile must lie wholly within the K×N HBM buffer
//                       (OOB safety not provided; for K-tail use the legacy
//                        path).
//   - `hbm_k_stride_bytes` : bytes per K-row in HBM (= N_full * sizeof(fp8)).
//   - `stage_lds`    : caller-supplied 16384-byte aligned LDS scratch.
//                     Caller controls aliasing / lifetime across phases.
//   - WG = 512 threads (8 warps × 64 lanes); function uses `threadIdx.x`.
//   - Caller is responsible for any pre-call __syncthreads (if stage_lds is
//     aliased with another consumer). The function itself issues 2 internal
//     barriers (after Phase 1+2, after Phase 3+4) so that on return the
//     LDS tile is consistent across all warps.
//
// Algorithm (Path L = LDS staging — Session 4 design doc §2):
//   Phase 1+2: HBM[K_row][N_col] → stage_lds[K_row*128 + N_col] (K-major)
//              8 warps × 16 K-rows/warp = 128 K covered; 2 iters × 16 N/iter
//              = 128 N covered. 2 b128 loads + 2 b128 writes per lane.
//   Phase 3+4: stage_lds[K_row*128 + N_row] (gather 16 K) → dst_n_major
//              at offset N_row*128 + K_strip. 16 ds_read_b8 + 1 ds_write_b128
//              per lane × 2 iters.
//   (Session 4.1 deferred Path P optimisation — cross-lane bpermute — has
//    NOT landed; this function ships the Path L baseline.)
//
// Performance note (from Session 4 probe ISA):
//   - 16 ds_read_b128 (HBM via b128 reinterpret_cast)
//   - 16 ds_write_b128 (staging + final)
//   - 64 ds_read_u8  per 16x16 byte block × 2 = 128 byte gathers
//   Single-tile transpose roughly 4× cost of a plain G::load row-major; for
//   the RRR body the amortised cost is acceptable so long as it is hidden
//   under mfma issue. Real perf budget is decided in Session 9.3.
template<ducks::st::all ST>
__device__ __forceinline__ void write_b_transpose_n_major_path_L(
    ST&             dst_n_major,
    const fp8e4m3*  hbm_b_tile_ptr,
    uint32_t        hbm_k_stride_bytes,
    uint8_t*        stage_lds)
{
    static_assert(std::is_same_v<typename ST::shape,
                                 ducks::st_shape::st_128x128_n_major>,
                  "write_b_transpose_n_major_path_L requires "
                  "st_128x128_n_major destination");
    static_assert(ST::rows == 128 && ST::cols == 128,
                  "write_b_transpose_n_major_path_L hard-wired for 128x128 tile");

    constexpr int K_DIM = 128;
    constexpr int N_DIM = 128;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 6;     // 0..7
    const int lane_id = tid & 63;     // 0..63

    // -------- Phase 1+2: HBM → staging LDS (K-major) --------
    #pragma unroll
    for (int iter = 0; iter < 2; ++iter) {
        const int n_block_base = iter * 64;
        const int k_local      = lane_id & 15;
        const int n_chunk      = (lane_id >> 4) & 3;
        const int K_row        = warp_id * 16 + k_local;
        const int N_col_start  = n_block_base + n_chunk * 16;

        const fp8e4m3* hbm_addr = reinterpret_cast<const fp8e4m3*>(
            reinterpret_cast<const uint8_t*>(hbm_b_tile_ptr) +
            static_cast<size_t>(K_row) * hbm_k_stride_bytes + N_col_start);
        const size_t lds_off = static_cast<size_t>(K_row) * N_DIM + N_col_start;

        __uint128_t v =
            *reinterpret_cast<const __uint128_t*>(hbm_addr);
        *reinterpret_cast<__uint128_t*>(&stage_lds[lds_off]) = v;
    }
    __syncthreads();

    // -------- Phase 3+4: staging → final Bs (N-major) --------
    uint8_t* dst_raw = reinterpret_cast<uint8_t*>(&dst_n_major.data[0]);
    #pragma unroll
    for (int iter = 0; iter < 2; ++iter) {
        const int n_block_base = iter * 64;
        const int N_row        = n_block_base + lane_id;
        const int K_strip      = warp_id * 16;

        uint8_t out16[16];
        #pragma unroll
        for (int k_in_strip = 0; k_in_strip < 16; ++k_in_strip) {
            const int K_global = K_strip + k_in_strip;
            out16[k_in_strip] = stage_lds[static_cast<size_t>(K_global) * N_DIM + N_row];
        }
        const size_t final_off = static_cast<size_t>(N_row) * K_DIM + K_strip;
        *reinterpret_cast<__uint128_t*>(&dst_raw[final_off]) =
            *reinterpret_cast<const __uint128_t*>(&out16[0]);
    }
    __syncthreads();
}
}
