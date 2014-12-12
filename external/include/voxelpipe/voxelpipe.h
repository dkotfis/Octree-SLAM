/*
 * Copyright (c) 2010-2011, NVIDIA Corporation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of NVIDIA Corporation nor the
 *     names of its contributors may be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*! \file voxelizer.h
 *  \brief high-level API to perform voxelization
 */

#pragma once

#include <nih/basic/types.h>
#include <nih/time/timer.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <voxelpipe/base.h>
#include <voxelpipe/coarse.h>
#include <voxelpipe/fine.h>
#include <voxelpipe/abuffer.h>
#include <voxelpipe/compact_ranges.h>
#include <voxelpipe/b40c/LsbRadixSort/radixsort_multi_cta.cu>
#include <voxelpipe/b40c/LsbRadixSort/radixsort_early_exit.cu>

#include <queue>

namespace voxelpipe {

///
/// High Level API context for performing blending-based voxelization.
///
template <int32 LOG_RESOLUTION, int32 LOG_TILE_SIZE>
struct FRContext
{
    static const int32 log_M = LOG_RESOLUTION - LOG_TILE_SIZE;
    static const int32 M     = 1 << log_M;
    static const int32 N     = 1 << LOG_RESOLUTION;

    typedef typename tile_id_selector<log_M>::type tile_id_type;

public:
    FineRasterStats         fr_stats;           // stats for last call
    nih::Timer              coarse_timer;       // stats for last call
    nih::Timer              fine_timer;         // stats for last call
    nih::Timer              sorting_timer;      // stats for last call
    nih::Timer              splitting_timer;    // stats for last call
    nih::Timer              blend_timer;        // stats for last call
    float                   coarse_time;        // cumulative stats
    float                   fine_time;          // cumulative stats
    float                   sorting_time;       // cumulative stats
    float                   splitting_time;     // cumulative stats
    float                   blend_time;         // cumulative stats

    int32                   n_tri_tile_pairs;   // number of triangles in tiles

    FRContext();

    /// reserve memory for a given amount of input triangles and a given
    /// amount of intermediate triangle-tile pairs (output by the coarse raster stage).
    /// Voxelization will fail if the buffers are not big enough.
    void reserve(const uint32 n_tris, const uint32 n_tri_tiles);

    /// execute the coarse raster stage.
    ///
    /// \param n_triangles      number of input triangles
    /// \param n_vertices       number of input vertices
    /// \param triangles        triangle data / vertex indices
    /// \param vertices         vertex data
    /// \param bbox0            scene bounding box
    /// \param bbox1            scene bounding box
    void coarse_raster(
        const int32   n_triangles,
        const int32   n_vertices,
        const int4*   triangles,
        const float4* vertices,
        const float3  bbox0,
        const float3  bbox1);

    /// execute the fine raster stage.
    ///
    /// \param n_triangles      number of input triangles
    /// \param n_vertices       number of input vertices
    /// \param triangles        triangle data / vertex indices
    /// \param vertices         vertex data
    /// \param bbox0            scene bounding box
    /// \param bbox1            scene bounding box
    /// \param fb               output framebuffer
    /// \param shader           user shader
    template <int32 VoxelType, int32 VoxelFormat, int32 VoxelizationType, int32 BlendingMode, typename ShaderType>
    void fine_raster(
        const int32   n_triangles,
        const int32   n_vertices,
        const int4*   triangles,
        const float4* vertices,
        const float3  bbox0,
        const float3  bbox1,
        void*         fb,
        ShaderType    shader = ShaderType());

private:
    thrust::device_vector<Tri_bbox>         d_tri_bbox;
    thrust::device_vector<int32>            d_tile_count;
    thrust::device_vector<int32>            d_tile_start;
    thrust::device_vector<int32>            d_tile_end;
    thrust::device_vector<int32>            d_compacted_tile_start;
    thrust::device_vector<int32>            d_compacted_tile_end;
    thrust::device_vector<int32>            d_compacted_tile_id;
    thrust::device_vector<tile_id_type>     d_tile_ids;
    thrust::device_vector<int32>            d_tile_tris;
    //thrust::device_vector<uint64>           d_sample_masks;
    //thrust::device_vector<float4>           d_plane_eqs;
    thrust::device_vector<int32>            d_block_offsets;

    thrust::device_vector<uint32>           d_batch_counter;
    thrust::device_vector<uint32>           d_fragment_counter;

    Compact_ranges                          m_compact_ranges;

    thrust::device_vector<uint8>            d_tile_buffer;

    b40c::MultiCtaRadixSortStorage<tile_id_type, int32> m_sorting_storage;
    int32                                               m_reserved_storage;
};

///
/// High Level API context for performing A-buffer based voxelization.
///
template <nih::int32 LOG_RESOLUTION>
struct ABufferContext
{
    static const int32 N = 1 << LOG_RESOLUTION;

    typedef uint32 tile_id_type;

public:
    FineRasterStats         fr_stats;           // stats for last call
    nih::Timer              setup_timer;        // stats for last call
    nih::Timer              emit_timer;         // stats for last call
    nih::Timer              sorting_timer;      // stats for last call
    float                   setup_time;         // cumulative stats
    float                   emit_time;          // cumulative stats
    float                   sorting_time;       // cumulative stats

    ABufferContext();

    /// reserve memory for a given amount of input triangles and a given
    /// amount of output fragments.
    /// Voxelization will fail if the buffers are not big enough.
    void reserve(const uint32 n_tris, const uint32 n_fragments);

    /// execute the voxelization.
    ///
    /// \param n_triangles      number of input triangles
    /// \param n_vertices       number of input vertices
    /// \param triangles        triangle data / vertex indices
    /// \param vertices         vertex data
    /// \param bbox0            scene bounding box
    /// \param bbox1            scene bounding box
    void run(
        const int32    n_triangles,
        const int32    n_vertices,
        const int4*    triangles,
        const float4*  vertices,
        const float3   bbox0,
        const float3   bbox1);

    /// return the vector of triangle ids of the generated fragments
    int32* get_fragment_tris() { return thrust::raw_pointer_cast( &d_fragment_tris.front() ); }

    /// return the vector of voxel ids of the generated fragments
    int32* get_fragment_ids()  { return thrust::raw_pointer_cast( &d_fragment_ids.front() ); }

private:
    thrust::device_vector<Tri_bbox>         d_tri_bbox;
    thrust::device_vector<uint8>            d_triangle_class;
    thrust::device_vector<int32>            d_triangle_idx;
    thrust::device_vector<int32>            d_fragment_tris;
    thrust::device_vector<int32>            d_fragment_ids;
    thrust::device_vector<int32>            h_large_tris;
    thrust::device_vector<int32>            d_large_tris;
    thrust::device_vector<int32>            d_block_offsets;

    thrust::device_vector<uint32>           d_batch_counter;
    thrust::device_vector<uint32>           d_fragment_counter;

    b40c::MultiCtaRadixSortStorage<uint8, int32>        m_sorting_storage;
    int32                                               m_reserved_storage;
    b40c::MultiCtaRadixSortStorage<int32, int32>        m_frag_sorting_storage;
    int32                                               m_frag_reserved_storage;
};

} // namespace voxelpipe

#include "../../external/src/voxelpipe/voxelpipe_inline.h"