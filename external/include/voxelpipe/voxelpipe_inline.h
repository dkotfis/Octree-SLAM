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

/*! \file voxelizer_inline.h
 *  \brief high-level API to perform voxelization, implementations
 */

#include <thrust/gather.h>

namespace voxelpipe {

template <int32 log_N, int32 log_T>
FRContext<log_N,log_T>::FRContext() :
    d_tile_count( 1 ),
    d_batch_counter( 1 ),
    d_fragment_counter( 1 ),
    d_block_offsets( 1024 ),
    d_tile_start( M*M*M+1 ),
    d_tile_end( M*M*M+1 ),
    d_compacted_tile_start( M*M*M+1 ),
    d_compacted_tile_end( M*M*M+1 ),
    d_compacted_tile_id( M*M*M+1 ),
    m_compact_ranges( M*M*M ),
    m_sorting_storage( 0 ),
    m_reserved_storage( 0 ),
    coarse_time( 0 ),
    fine_time( 0 ),
    sorting_time( 0 ),
    splitting_time( 0 ),
    blend_time( 0 )
    {}

template <int32 log_N, int32 log_T>
void FRContext<log_N,log_T>::reserve(const uint32 n_tris, const uint32 n_tri_tiles)
{
    d_tri_bbox.resize( n_tris );
    //d_plane_eqs.resize( n_tri_tiles );
    //d_sample_masks.resize( n_tri_tiles );
    d_tile_ids.resize( n_tri_tiles );
    d_tile_tris.resize( n_tri_tiles );
};

template <int32 log_N>
ABufferContext<log_N>::ABufferContext() :
    d_batch_counter( 1 ),
    d_fragment_counter( 1 ),
    h_large_tris( 4 ),
    d_large_tris( 4 ),
    m_sorting_storage( 0 ),
    m_reserved_storage( 0 ),
    m_frag_sorting_storage( 0 ),
    m_frag_reserved_storage( 0 ),
    setup_time( 0 ),
    emit_time( 0 ),
    sorting_time( 0 )
    {}

template <int32 log_N>
void ABufferContext<log_N>::reserve(const uint32 n_tris, const uint32 n_fragments)
{
    d_tri_bbox.resize( n_tris );
    d_triangle_class.resize( n_tris );
    d_triangle_idx.resize( n_tris );
    d_fragment_tris.resize( n_fragments );
    d_fragment_ids.resize( n_fragments );
};


inline int32 split_tiles(
    const int32 n_tiles,
    const int32 max_tiles,
    thrust::device_vector<int32>& d_compacted_tile_start,
    thrust::device_vector<int32>& d_compacted_tile_end,
    thrust::device_vector<int32>& d_compacted_tile_id,
    thrust::device_vector<int32>& d_tile_range_start,
    thrust::device_vector<int32>& d_tile_range_end);

template <int32 log_N, int32 log_T>
void FRContext<log_N,log_T>::coarse_raster(
    const int32   n_triangles,
    const int32   n_vertices,
    const int4*   triangles,
    const float4* vertices,
    const float3  bbox0,
    const float3  bbox1)
{
    const float3 bbox_delta = make_float3(
        (bbox1.x - bbox0.x) / float(N),
        (bbox1.y - bbox0.y) / float(N),
        (bbox1.z - bbox0.z) / float(N) );

    const float3 inv_bbox_delta = make_float3(
        float(N) / (bbox1.x - bbox0.x),
        float(N) / (bbox1.y - bbox0.y),
        float(N) / (bbox1.z - bbox0.z));

    //b40c::SingleGridRadixSortingEnactor<tile_id_type, int32> sorting_enactor;
    b40c::EarlyExitRadixSortingEnactor<tile_id_type, int32> sorting_enactor;

    coarse_timer.start();

    // reset tile counter
    d_batch_counter[0] = 0;
    d_tile_count[0] = 0;
  #if 1
    // launch coarse counting kernel
    voxelpipe::coarse_raster<log_T,tile_id_type>(
        N,
        log_N,
        n_triangles,
        n_vertices,
        bbox0,
        bbox1,
        inv_bbox_delta,
        vertices,
        triangles,
        thrust::raw_pointer_cast( &d_batch_counter.front() ),
        thrust::raw_pointer_cast( &d_tile_count.front() ),
        thrust::raw_pointer_cast( &d_tile_ids.front() ),
        thrust::raw_pointer_cast( &d_tile_tris.front() ),
        thrust::raw_pointer_cast( &d_tri_bbox.front() ) );

    n_tri_tile_pairs = d_tile_count[0];
  #else
    // launch coarse counting kernel
    voxelpipe::coarse_raster_count<log_T,tile_id_type>(
        N,
        log_N,
        n_triangles,
        n_vertices,
        bbox0,
        bbox1,
        inv_bbox_delta,
        vertices,
        triangles,
        thrust::raw_pointer_cast( &d_tile_count.front() ),
        thrust::raw_pointer_cast( &d_block_offsets.front() ),
        thrust::raw_pointer_cast( &d_tri_bbox.front() ) );

    n_tri_tile_pairs = d_tile_count[0];

    if (d_tile_ids.size() < (size_t) n_tri_tile_pairs)
    {
        d_tile_ids.resize( n_tri_tile_pairs );
        d_tile_tris.resize( n_tri_tile_pairs );
        //d_sample_masks.resize( n_tri_tile_pairs );
        //d_plane_eqs.resize( n_tri_tile_pairs );
    }

    // reset tile counter
    d_tile_count[0] = 0;

    // launch coarse setup kernel
    voxelpipe::coarse_raster_setup<log_T>(
        N,
        log_N,
        n_triangles,
        thrust::raw_pointer_cast( &d_tri_bbox.front() ),
        thrust::raw_pointer_cast( &d_block_offsets.front() ),
        thrust::raw_pointer_cast( &d_tile_ids.front() ),
        thrust::raw_pointer_cast( &d_tile_tris.front() ) );
  #endif

    sorting_timer.start();

    // sort the tile-triangle pairs by tile id
    if (m_reserved_storage < n_tri_tile_pairs)
    {
        m_reserved_storage = n_tri_tile_pairs;

        cudaFree( m_sorting_storage.d_keys[1] );
        cudaFree( m_sorting_storage.d_values[1] );

        m_sorting_storage = b40c::MultiCtaRadixSortStorage<tile_id_type, int32>( n_tri_tile_pairs );
    }
    else
    {
        m_sorting_storage.num_elements = n_tri_tile_pairs;
        m_sorting_storage.selector     = 0;
    }

    m_sorting_storage.d_keys[0]   = thrust::raw_pointer_cast( &d_tile_ids.front() );
    m_sorting_storage.d_values[0] = thrust::raw_pointer_cast( &d_tile_tris.front() );

    sorting_enactor.EnactSort( m_sorting_storage );
    cudaThreadSynchronize();

    sorting_timer.stop();
    sorting_time += sorting_timer.seconds();

    // TODO: this copy could be removed by just swapping pointers
    if (m_sorting_storage.selector)
    {
        cudaMemcpy(
        	m_sorting_storage.d_keys[0], 
        	m_sorting_storage.d_keys[m_sorting_storage.selector], 
        	sizeof(tile_id_type) * n_tri_tile_pairs, 
        	cudaMemcpyDeviceToDevice );

        cudaMemcpy(
        	m_sorting_storage.d_values[0], 
        	m_sorting_storage.d_values[m_sorting_storage.selector], 
        	sizeof(int32) * n_tri_tile_pairs, 
        	cudaMemcpyDeviceToDevice );
    }

    cudaThreadSynchronize();

    coarse_timer.stop();
    coarse_time += coarse_timer.seconds();
}

template <int32 log_N, int32 log_T>
template <int32 VoxelType, int32 VoxelFormat, int32 VoxelizationType, int32 BlendingMode, typename ShaderType>
void FRContext<log_N,log_T>::fine_raster(
    const int32   n_triangles,
    const int32   n_vertices,
    const int4*   triangles,
    const float4* vertices,
    const float3  bbox0,
    const float3  bbox1,
    void*         fb,
    ShaderType    shader)
{
    const float3 bbox_delta = make_float3(
        (bbox1.x - bbox0.x) / float(N),
        (bbox1.y - bbox0.y) / float(N),
        (bbox1.z - bbox0.z) / float(N) );

    const float3 inv_bbox_delta = make_float3(
        float(N) / (bbox1.x - bbox0.x),
        float(N) / (bbox1.y - bbox0.y),
        float(N) / (bbox1.z - bbox0.z));

    // we now have the triangles sorted by tile and can proceed to (a) checking whether
    // they really intersect the tile, (b) rasterizing them within the tile
    fine_timer.start();

    // compute begin and end of each tile
    thrust::fill( d_tile_start.begin(), d_tile_start.begin() + M*M*M, 0 );
    thrust::fill( d_tile_end.begin(),   d_tile_end.begin()   + M*M*M, 0 );
    {
        cudaThreadSynchronize();

        voxelpipe::compute_tile_ranges(
            n_tri_tile_pairs,
            thrust::raw_pointer_cast( &d_tile_ids.front() ),
            thrust::raw_pointer_cast( &d_tile_start.front() ),
            thrust::raw_pointer_cast( &d_tile_end.front() ) );

        const uint32 first_tile_id = (d_tile_ids[0])                  >> TILE_ID_SHIFT;
        const uint32 last_tile_id  = (d_tile_ids[n_tri_tile_pairs-1]) >> TILE_ID_SHIFT;

        d_tile_start[ first_tile_id ] = 0;
        d_tile_end[ last_tile_id ]    = n_tri_tile_pairs;

        cudaThreadSynchronize();
    }

    // compact the list of non-empty tiles
    const int32 n_tiles = m_compact_ranges.run(
        thrust::raw_pointer_cast( &d_compacted_tile_start.front() ),
        thrust::raw_pointer_cast( &d_compacted_tile_end.front() ),
        thrust::raw_pointer_cast( &d_compacted_tile_id.front() ),
        thrust::raw_pointer_cast( &d_tile_start.front() ),
        thrust::raw_pointer_cast( &d_tile_end.front() ),
        M*M*M );

  #if 1
    splitting_timer.start();

    const int32 tile_count = split_tiles(
        n_tiles,
        M*M*M,
        d_compacted_tile_start,
        d_compacted_tile_end,
        d_compacted_tile_id,
        d_tile_start,
        d_tile_end );

    splitting_timer.stop();
    splitting_time += splitting_timer.seconds();
  #else
    const int32 tile_count = n_tiles;

    // TODO: set d_tile_start and d_tile_end
  #endif

    const size_t tile_buffer_size =
        size_t( tile_count ) *
        sizeof( typename FR::TileOp<VoxelType,BlendingMode,log_T>::storage_type ) *
        FR::TileOp<VoxelType,BlendingMode,log_T>::STORAGE_SIZE;

    if (d_tile_buffer.size() < tile_buffer_size)
        d_tile_buffer.resize( tile_buffer_size );

    // reset batch counter
    d_batch_counter[0] = 0;

    // fine raster
    voxelpipe::fine_raster<ShaderType, log_T, VoxelType, VoxelizationType, BlendingMode>(
        N,
        log_N,
        tile_count,
        n_triangles,
        n_vertices,
        triangles,
        vertices,
        thrust::raw_pointer_cast( &d_tri_bbox.front() ),
        bbox0,
        bbox1,
        bbox_delta,
        inv_bbox_delta,
        thrust::raw_pointer_cast( &d_compacted_tile_start.front() ),
        thrust::raw_pointer_cast( &d_compacted_tile_end.front() ),
        thrust::raw_pointer_cast( &d_compacted_tile_id.front() ),
        thrust::raw_pointer_cast( &d_tile_tris.front() ),
        thrust::raw_pointer_cast( &d_tile_buffer.front() ),
        thrust::raw_pointer_cast( &d_batch_counter.front() ),
        //thrust::raw_pointer_cast( &d_sample_masks.front() ),
        //thrust::raw_pointer_cast( &d_plane_eqs.front() ),
        fr_stats,
        shader );

    blend_timer.start();

    // reset batch counter
    d_batch_counter[0] = 0;

    voxelpipe::blend_tiles<log_T,VoxelType,VoxelFormat,BlendingMode>(
        N,
        log_N,
        thrust::raw_pointer_cast( &d_tile_start.front() ),
        thrust::raw_pointer_cast( &d_tile_end.front() ),
        thrust::raw_pointer_cast( &d_tile_buffer.front() ),
        fb,
        thrust::raw_pointer_cast( &d_batch_counter.front() ),
        fr_stats );

    blend_timer.stop();
    blend_time += blend_timer.seconds();

    fine_timer.stop();
    fine_time += fine_timer.seconds();
}

template <int32 log_N>
void ABufferContext<log_N>::run(
    const int32   n_triangles,
    const int32   n_vertices,
    const int4*   triangles,
    const float4* vertices,
    const float3  bbox0,
    const float3  bbox1)
{
    const float3 bbox_delta = make_float3(
        (bbox1.x - bbox0.x) / float(N),
        (bbox1.y - bbox0.y) / float(N),
        (bbox1.z - bbox0.z) / float(N) );

    const float3 inv_bbox_delta = make_float3(
        float(N) / (bbox1.x - bbox0.x),
        float(N) / (bbox1.y - bbox0.y),
        float(N) / (bbox1.z - bbox0.z));

    b40c::EarlyExitRadixSortingEnactor<uint8, int32> sorting_enactor;
    b40c::EarlyExitRadixSortingEnactor<int32, int32> frag_sorting_enactor;

    // start triangle setup
    setup_timer.start();

    // clear counters
    h_large_tris[0] = 0;
    h_large_tris[1] = 0;
    h_large_tris[2] = 0;
    h_large_tris[3] = 0;
    d_large_tris    = h_large_tris;

    if (d_tri_bbox.size() < size_t( n_triangles ))
    {
        d_tri_bbox.resize( n_triangles );
        d_triangle_class.resize( n_triangles );
        d_triangle_idx.resize( n_triangles );
    }

    // launch coarse counting kernel
    voxelpipe::setup<0>(
        N,
        log_N,
        n_triangles,
        n_vertices,
        bbox0,
        bbox1,
        inv_bbox_delta,
        vertices,
        triangles,
        thrust::raw_pointer_cast( &d_triangle_class.front() ),
        thrust::raw_pointer_cast( &d_triangle_idx.front() ),
        thrust::raw_pointer_cast( &d_tri_bbox.front() ),
        thrust::raw_pointer_cast( &d_large_tris.front() ) );

    h_large_tris = d_large_tris;

    // sort the triangles by their class
    if (m_reserved_storage < n_triangles)
    {
        m_reserved_storage = n_triangles;

        if (m_sorting_storage.d_keys[1])   cudaFree( m_sorting_storage.d_keys[1] );
        if (m_sorting_storage.d_values[1]) cudaFree( m_sorting_storage.d_values[1] );

        m_sorting_storage = b40c::MultiCtaRadixSortStorage<uint8, int32>( n_triangles );
    }
    else
    {
        m_sorting_storage.num_elements = n_triangles;
        m_sorting_storage.selector     = 0;
    }

    m_sorting_storage.d_keys[0]   = thrust::raw_pointer_cast( &d_triangle_class.front() );
    m_sorting_storage.d_values[0] = thrust::raw_pointer_cast( &d_triangle_idx.front() );

    sorting_enactor.EnactSort( m_sorting_storage );

    setup_timer.stop();
    setup_time += setup_timer.seconds();

    // start fragment emission
    emit_timer.start();

    d_fragment_counter[0] = 0;

    const int32 large_tris = h_large_tris[0] + h_large_tris[1] + h_large_tris[2] + h_large_tris[3];
    const int32 small_tris = n_triangles - large_tris;

    const int32* sorted_triangle_idx_ptr = m_sorting_storage.d_values[ m_sorting_storage.selector ];

    // process small triangles
    if (small_tris)
    {
        d_batch_counter[0] = 0;

        voxelpipe::A_buffer_unsorted_small<voxelpipe::DefaultShader<voxelpipe::Bit>, voxelpipe::Bit, voxelpipe::THIN_RASTER>(
            N,
            log_N,
            small_tris,
            n_vertices,
            sorted_triangle_idx_ptr,
            triangles,
            vertices,
            thrust::raw_pointer_cast( &d_tri_bbox.front() ),
            bbox0,
            bbox1,
            bbox_delta,
            inv_bbox_delta,
            thrust::raw_pointer_cast( &d_batch_counter.front() ),
            thrust::raw_pointer_cast( &d_fragment_counter.front() ),
            thrust::raw_pointer_cast( &d_fragment_tris.front() ),
            thrust::raw_pointer_cast( &d_fragment_ids.front() ),
            fr_stats );
    }

    // process large triangles
    if (large_tris)
    {
        d_batch_counter[0] = 0;

        voxelpipe::A_buffer_unsorted_large<voxelpipe::DefaultShader<voxelpipe::Bit>, voxelpipe::Bit, voxelpipe::THIN_RASTER>(
            N,
            log_N,
            large_tris,
            n_vertices,
            sorted_triangle_idx_ptr + small_tris,
            triangles,
            vertices,
            thrust::raw_pointer_cast( &d_tri_bbox.front() ),
            bbox0,
            bbox1,
            bbox_delta,
            inv_bbox_delta,
            thrust::raw_pointer_cast( &d_batch_counter.front() ),
            thrust::raw_pointer_cast( &d_fragment_counter.front() ),
            thrust::raw_pointer_cast( &d_fragment_tris.front() ),
            thrust::raw_pointer_cast( &d_fragment_ids.front() ),
            fr_stats );
    }

    const int32 n_fragments = d_fragment_counter[0];

    emit_timer.stop();
    emit_time += emit_timer.seconds();

    // start fragment sorting
    sorting_timer.start();

    // sort the triangles by their class
    if (m_frag_reserved_storage < n_fragments)
    {
        m_frag_reserved_storage = n_fragments;

        if (m_frag_sorting_storage.d_keys[1])   cudaFree( m_frag_sorting_storage.d_keys[1] );
        if (m_frag_sorting_storage.d_values[1]) cudaFree( m_frag_sorting_storage.d_values[1] );

        m_frag_sorting_storage = b40c::MultiCtaRadixSortStorage<int32, int32>( n_fragments );
    }
    else
    {
        m_frag_sorting_storage.num_elements = n_fragments;
        m_frag_sorting_storage.selector     = 0;
    }

    m_frag_sorting_storage.d_keys[0]   = thrust::raw_pointer_cast( &d_fragment_ids.front() );
    m_frag_sorting_storage.d_values[0] = thrust::raw_pointer_cast( &d_fragment_tris.front() );

    frag_sorting_enactor.EnactSort( m_frag_sorting_storage );

    if (m_frag_sorting_storage.selector)
    {
        cudaMemcpy(
        	m_frag_sorting_storage.d_keys[0], 
        	m_frag_sorting_storage.d_keys[m_sorting_storage.selector], 
        	sizeof(tile_id_type) * n_fragments, 
        	cudaMemcpyDeviceToDevice );

        cudaMemcpy(
        	m_frag_sorting_storage.d_values[0], 
        	m_frag_sorting_storage.d_values[m_sorting_storage.selector], 
        	sizeof(int32) * n_fragments, 
        	cudaMemcpyDeviceToDevice );
    }

    sorting_timer.stop();
    sorting_time += sorting_timer.seconds();
}


struct Tile_range
{
    Tile_range() {}
    Tile_range(const int32 begin, const int32 end, const int32 id) :
        m_begin( begin ), m_end( end ), m_id( id ) {}

    // comparison functor
    bool operator() (const Tile_range t1, const Tile_range t2) const
    {
        return t1.m_end - t1.m_begin < t2.m_end - t2.m_begin;
    }

    int32 m_begin;
    int32 m_end;
    int32 m_id;
};

inline int32 split_tiles(
    const int32 n_tiles,
    const int32 max_tiles,
    thrust::device_vector<int32>& d_compacted_tile_start,
    thrust::device_vector<int32>& d_compacted_tile_end,
    thrust::device_vector<int32>& d_compacted_tile_id,
    thrust::device_vector<int32>& d_tile_range_start,
    thrust::device_vector<int32>& d_tile_range_end)
{
    const int32 TRIS_PER_TILE = 8 * 1024;

    thrust::host_vector<int32> h_tile_start( n_tiles );
    thrust::host_vector<int32> h_tile_end( n_tiles );
    thrust::host_vector<int32> h_tile_id( n_tiles );
    thrust::host_vector<int32> h_tile_range_start( max_tiles, 0 );
    thrust::host_vector<int32> h_tile_range_end( max_tiles, 0 );
    thrust::copy( d_compacted_tile_start.begin(), d_compacted_tile_start.begin() + n_tiles, h_tile_start.begin() );
    thrust::copy( d_compacted_tile_end.begin(),   d_compacted_tile_end.begin()   + n_tiles, h_tile_end.begin() );
    thrust::copy( d_compacted_tile_id.begin(),    d_compacted_tile_id.begin()    + n_tiles, h_tile_id.begin() );

    thrust::host_vector<int32> new_tile_id;
    thrust::host_vector<int32> new_tile_start;
    thrust::host_vector<int32> new_tile_end;
/*
    new_tile_id.reserve( 8*n_tiles );
    new_tile_start.reserve( 8*n_tiles );
    new_tile_end.reserve( 8*n_tiles );

    // keep a priority queue
    std::priority_queue<
        Tile_range,
        std::vector<Tile_range>,
        Tile_range> tile_queue;

    // queue tiles to be split, and copy small ones directly to the output
    for (int32 tid = 0; tid < n_tiles; ++tid)
    {
        if (h_tile_end[tid] - h_tile_start[tid] > (TRIS_PER_TILE*3)/2)
            tile_queue.push( Tile_range( h_tile_start[tid], h_tile_end[tid], h_tile_id[tid] ) );
        else
        {
            new_tile_start.push_back( h_tile_start[tid] );
            new_tile_end.push_back( h_tile_end[tid] );
            new_tile_id.push_back( h_tile_id[tid] );
        }
    }

    // keep splitting large tiles until we can
    while (tile_queue.size() && (new_tile_id.size() + tile_queue.size() < h_tile_id.size()*8))
    {
        const Tile_range top = tile_queue.top();
        tile_queue.pop();

        const int32 mid = (top.m_begin + top.m_end) / 2;

        if (top.m_end - top.m_begin > TRIS_PER_TILE*2)
        {
            tile_queue.push( Tile_range( top.m_begin,   mid,        top.m_id ) );
            tile_queue.push( Tile_range( mid,           top.m_end,  top.m_id ) );
        }
        else if (top.m_end - top.m_begin > (TRIS_PER_TILE*3)/2)
        {
            new_tile_start.push_back( top.m_begin );
            new_tile_end.push_back( mid );
            new_tile_id.push_back( top.m_id );

            new_tile_start.push_back( mid );
            new_tile_end.push_back( top.m_end );
            new_tile_id.push_back( top.m_id );
        }
        else
        {
            new_tile_start.push_back( top.m_begin );
            new_tile_end.push_back( top.m_end );
            new_tile_id.push_back( top.m_id );
        }
    }
    // flush the queue
    while (tile_queue.size())
    {
        const Tile_range top = tile_queue.top();
        tile_queue.pop();

        new_tile_start.push_back( top.m_begin );
        new_tile_end.push_back( top.m_end );
        new_tile_id.push_back( top.m_id );
    }
    const int32 tile_count = (int32)new_tile_id.size();

    // sort tiles by tile id
    // TODO
*/
    new_tile_id.reserve( 2*max_tiles );
    new_tile_start.reserve( 2*max_tiles );
    new_tile_end.reserve( 2*max_tiles );

    for (int32 t = 0; t < n_tiles; ++t)
    {
        const int32 begin = h_tile_start[ t ];
        const int32 end   = h_tile_end[ t ];
        const int32 tid   = h_tile_id[ t ];

        if (end - begin == 0)
        {
            h_tile_range_start[ tid ] = 0;
            h_tile_range_end[ tid ] = 0;
            continue;
        }

        const int32 new_tile_set_start = int32( new_tile_start.size() );

        int32 new_end = min( begin + TRIS_PER_TILE, end );

        new_tile_id.push_back( tid );
        new_tile_start.push_back( begin );
        new_tile_end.push_back( new_end );

        while (new_end < end)
        {
            // compute new tile boundaries
            const uint32 new_begin = new_end;
            new_end = min( new_end + TRIS_PER_TILE, end );

            // write the new tile
            new_tile_id.push_back( tid );
            new_tile_start.push_back( new_begin );
            new_tile_end.push_back( new_end );
        }

        // remember the beginning and end of this set of tiles
        h_tile_range_start[ tid ] = new_tile_set_start;
        h_tile_range_end[ tid ]   = int32( new_tile_start.size() );
    }
    const int32 tile_count = (int32)new_tile_id.size();

    // copy new tiles back to the device
    if (d_compacted_tile_start.size() < size_t( tile_count ))
    {
        d_compacted_tile_start.resize( tile_count );
        d_compacted_tile_end.resize( tile_count );
        d_compacted_tile_id.resize( tile_count );
    }
    thrust::copy( new_tile_start.begin(), new_tile_start.begin() + tile_count, d_compacted_tile_start.begin() );
    thrust::copy( new_tile_end.begin(),   new_tile_end.begin()   + tile_count, d_compacted_tile_end.begin() );
    thrust::copy( new_tile_id.begin(),    new_tile_id.begin()    + tile_count, d_compacted_tile_id.begin() );

    thrust::copy( h_tile_range_start.begin(), h_tile_range_start.begin() + max_tiles, d_tile_range_start.begin() );
    thrust::copy( h_tile_range_end.begin(),   h_tile_range_end.begin()   + max_tiles, d_tile_range_end.begin() );

    return (int32)tile_count;
}

} // namespace voxelpipe
