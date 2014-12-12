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

/*! \file coarse.h
 *  \brief Coarse Raster kernels
 */

#pragma once

#include <voxelpipe/common.h>
#include <voxelpipe/utils.h>

#if VOXELPIPE_CR_SCANLINE_SORTING
#define TILE_ID_SHIFT 6
#else
#define TILE_ID_SHIFT 2
#endif

namespace voxelpipe {


///
/// compute a triangle's integer bbounding box and its major axis
///
inline __device__ void setup_triangle(
    const int32   N,
    const float3  bbox0,
    const float3  bbox1,
    const float3  inv_bbox_delta,
    const float4  v0,
    const float4  v1,
    const float4  v2,
    int3&         tri_bbox0i,
    int3&         tri_bbox1i,
    int32&        tri_axis)
{
    float3 tri_bbox0;
    tri_bbox0.x = (v0.x - bbox0.x) * inv_bbox_delta.x;
    tri_bbox0.y = (v0.y - bbox0.y) * inv_bbox_delta.y;
    tri_bbox0.z = (v0.z - bbox0.z) * inv_bbox_delta.z;
    tri_bbox0.x = fminf( (v1.x - bbox0.x) * inv_bbox_delta.x, tri_bbox0.x );
    tri_bbox0.y = fminf( (v1.y - bbox0.y) * inv_bbox_delta.y, tri_bbox0.y );
    tri_bbox0.z = fminf( (v1.z - bbox0.z) * inv_bbox_delta.z, tri_bbox0.z );
    tri_bbox0.x = fminf( (v2.x - bbox0.x) * inv_bbox_delta.x, tri_bbox0.x );
    tri_bbox0.y = fminf( (v2.y - bbox0.y) * inv_bbox_delta.y, tri_bbox0.y );
    tri_bbox0.z = fminf( (v2.z - bbox0.z) * inv_bbox_delta.z, tri_bbox0.z );

    float3 tri_bbox1;
    tri_bbox1.x = (v0.x - bbox0.x) * inv_bbox_delta.x;
    tri_bbox1.y = (v0.y - bbox0.y) * inv_bbox_delta.y;
    tri_bbox1.z = (v0.z - bbox0.z) * inv_bbox_delta.z;
    tri_bbox1.x = fmaxf( (v1.x - bbox0.x) * inv_bbox_delta.x, tri_bbox1.x );
    tri_bbox1.y = fmaxf( (v1.y - bbox0.y) * inv_bbox_delta.y, tri_bbox1.y );
    tri_bbox1.z = fmaxf( (v1.z - bbox0.z) * inv_bbox_delta.z, tri_bbox1.z );
    tri_bbox1.x = fmaxf( (v2.x - bbox0.x) * inv_bbox_delta.x, tri_bbox1.x );
    tri_bbox1.y = fmaxf( (v2.y - bbox0.y) * inv_bbox_delta.y, tri_bbox1.y );
    tri_bbox1.z = fmaxf( (v2.z - bbox0.z) * inv_bbox_delta.z, tri_bbox1.z );

    // compute integer bbox
    tri_bbox0i = make_int3(
        min( max( int( tri_bbox0.x ), 0 ), N-1 ),
        min( max( int( tri_bbox0.y ), 0 ), N-1 ),
        min( max( int( tri_bbox0.z ), 0 ), N-1 ) );

    tri_bbox1i = make_int3(
        min( max( int( ceil( tri_bbox1.x ) ), 0 ), N-1 ),
        min( max( int( ceil( tri_bbox1.y ) ), 0 ), N-1 ),
        min( max( int( ceil( tri_bbox1.z ) ), 0 ), N-1 ) );

    const float3 edge0 = make_float3( v1.x - v0.x, v1.y - v0.y, v1.z - v0.z );
    const float3 edge2 = make_float3( v0.x - v2.x, v0.y - v2.y, v0.z - v2.z );
    const float3 n = anti_cross( edge0, edge2 );

    // compute dominant axis
    const bool byx = fabsf( n.y ) > fabsf( n.x );
    const bool byz = fabsf( n.y ) > fabsf( n.z );
    const bool bzx = fabsf( n.z ) > fabsf( n.x );
    tri_axis = byx ? (byz ? 1 : 2) : (bzx ? 2 : 0);
}

///
/// This kernel counts how many triangle-tile pairs are emitted for each triangle.
/// Additionally, the (integer grid) bounding box is output for each triangle,
/// together with its dominant axis.
///
template <int32 BLOCK_SIZE, int32 log_TILE_SIZE>
__global__ void setup_kernel(
    const int32     grid_size,
    const int32     N_grids,
    const int32     N_blocks,
    const int32     N,
    const int32     log_N,
    const int32     N_tris,
    const float3    bbox0,
    const float3    bbox1,
    const float3    inv_bbox_delta,
    const float4*   vertices,
    const int4*     triangles,
    uint8*          output_offsets,
    int32*          output_tris,
    Tri_bbox*       output_tri_bbox,
    int32*          output_large_tris)
{
    const uint32 WARP_COUNT = BLOCK_SIZE >> 5;
    volatile __shared__ uint32 sm_red[ BLOCK_SIZE * 2 ];
    volatile __shared__ uint32 sm_red_cta[ WARP_COUNT*2*4+1 ];

    int4 large_popc4 = make_int4(0,0,0,0);

    for (int32 grid = 0; grid < N_grids; ++grid)
    {
        const int32 tri_idx = threadIdx.x + blockIdx.x*BLOCK_SIZE + grid * grid_size;

        const bool valid = tri_idx < N_tris;

        int32 large_class = 4;

        if (valid)
        {
            const int4 tri = triangles[ tri_idx ];
            const float4 v0 = tex1Dfetch( tex_vertices, tri.x );
            const float4 v1 = tex1Dfetch( tex_vertices, tri.y );
            const float4 v2 = tex1Dfetch( tex_vertices, tri.z );

            int3  tri_bbox0i;
            int3  tri_bbox1i;
            int32 tri_axis;

            setup_triangle(
                N,
                bbox0,
                bbox1,
                inv_bbox_delta,
                v0, v1, v2,
                tri_bbox0i,
                tri_bbox1i,
                tri_axis );

            // write out triangle bbox
            ((int4*)output_tri_bbox)[ tri_idx*Tri_bbox::N_INT4   ] = make_int4( tri_bbox0i.x, tri_bbox0i.y, tri_bbox0i.z, tri_bbox1i.x );
            ((int4*)output_tri_bbox)[ tri_idx*Tri_bbox::N_INT4+1 ] = make_int4( tri_bbox1i.y, tri_bbox1i.z, tri_axis, 0 );

            const int32 tile_count_x = (tri_bbox1i.x >> log_TILE_SIZE) - (tri_bbox0i.x >> log_TILE_SIZE) + 1;
            const int32 tile_count_y = (tri_bbox1i.y >> log_TILE_SIZE) - (tri_bbox0i.y >> log_TILE_SIZE) + 1;
            const int32 tile_count_z = (tri_bbox1i.z >> log_TILE_SIZE) - (tri_bbox0i.z >> log_TILE_SIZE) + 1;

        #if 0
            const uint32 tri_tile_count =
                (tri_axis == 0) ? tile_count_y * tile_count_z :
                (tri_axis == 1) ? tile_count_x * tile_count_z :
                                  tile_count_x * tile_count_y;

            const uint32 log_size = min( log2( tri_tile_count ), 15u );
            const uint32 triangle_class = uint32( (tri_axis << 4) + log_size );
        #elif 0
            const uint32 tri_tile_count_u = (tri_axis == 0) ? tile_count_y : tile_count_x;
            const uint32 tri_tile_count_v = (tri_axis == 2) ? tile_count_y : tile_count_z;

            const uint32 log_size = min( log2( tri_tile_count_u ), 3u ) * 4 + min( log2( tri_tile_count_v ), 3u );
            const uint32 triangle_class = uint32( (tri_axis << 4) + log_size );
        #elif 1
            const uint32 tri_tile_count_u = (tri_axis == 0) ? tile_count_y : tile_count_x;
            const uint32 tri_tile_count_v = (tri_axis == 2) ? tile_count_y : tile_count_z;

            const uint32 log_size = nih::min( nih::log2( tri_tile_count_u ), 3u ) * 4 + nih::min( nih::log2( tri_tile_count_v ), 3u );

            if (tri_tile_count_u > 16)
                large_class = 1;
            else if (tri_tile_count_v > 16)
                large_class = 3;
            else if (tri_tile_count_u == 16)
                large_class = 0;
            else if (tri_tile_count_v == 16)
                large_class = 2;

            uint32 triangle_class = uint32( (tri_axis << 4) + log_size );
            if (large_class < 4)
                triangle_class = 48 + large_class*3 + tri_axis;
        #else
            const uint32 log_size = min( log2( tile_count_x * tile_count_y * tile_count_z ), 15u );
            const uint32 triangle_class = uint32( (tri_axis << 4) + log_size );
        #endif

            output_offsets[ tri_idx ] = triangle_class;
            output_tris[ tri_idx ] = tri_idx;
        }

        // count how many large triangles we emitted for each type
        pop_count4<BLOCK_SIZE>(
            large_class & 3,
            large_class < 4,
            sm_red,
            sm_red_cta );

        if (threadIdx.x == 0)
        {
            large_popc4.x += pop_count4_total<BLOCK_SIZE>( 0, sm_red_cta );
            large_popc4.y += pop_count4_total<BLOCK_SIZE>( 1, sm_red_cta );
            large_popc4.z += pop_count4_total<BLOCK_SIZE>( 2, sm_red_cta );
            large_popc4.w += pop_count4_total<BLOCK_SIZE>( 3, sm_red_cta );
        }
    }

    // do a single atomic per block to count the large triangles
    if (threadIdx.x == 0)
    {
        atomicAdd( output_large_tris+0, large_popc4.x );
        atomicAdd( output_large_tris+1, large_popc4.y );
        atomicAdd( output_large_tris+2, large_popc4.z );
        atomicAdd( output_large_tris+3, large_popc4.w );
    }
}

///
/// This function performs setup of each triangle´s bbox and computes a class type
///
template <int32 log_TILE_SIZE>
inline void setup(
    const int32     N,
    const int32     log_N,
    const int32     N_tris,
    const int32     N_vertices,
    const float3    bbox0,
    const float3    bbox1,
    const float3    inv_bbox_delta,
    const float4*   vertices,
    const int4*     triangles,
    uint8*          triangle_offsets,
    int32*          output_tris,
    Tri_bbox*       output_tri_bbox,
    int32*          output_large_tris)
{
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();

    tex_vertices.normalized = false;
    tex_vertices.filterMode = cudaFilterModePoint;

    cudaBindTexture( 0, &tex_vertices, vertices, &channel_desc, sizeof(float4)*N_vertices );

    cudaThreadSynchronize();

    const int32 BLOCK_SIZE = 256;
    const int32 block_count = thrust::detail::device::cuda::arch::max_active_blocks(setup_kernel<BLOCK_SIZE, log_TILE_SIZE>, BLOCK_SIZE, 0);

    dim3 dim_block = dim3( BLOCK_SIZE, 1, 1 );
    dim3 dim_grid  = dim3( block_count, 1, 1 );

    const int32 grid_size = BLOCK_SIZE * block_count;
    const int32 N_grids   = (N_tris + grid_size-1) / grid_size;

    setup_kernel<BLOCK_SIZE, log_TILE_SIZE> <<<dim_grid,dim_block>>>(
        grid_size,
        N_grids,
        block_count,
        N,
        log_N,
        N_tris,
        bbox0,
        bbox1,
        inv_bbox_delta,
        vertices,
        triangles,
        triangle_offsets,
        output_tris,
        output_tri_bbox,
        output_large_tris );

    cudaThreadSynchronize();

    cudaUnbindTexture( &tex_vertices );
}

///
/// This kernel counts how many triangle-tile pairs are emitted for each triangle.
/// Additionally, the (integer grid) bounding box is output for each triangle,
/// together with its dominant axis.
///
template <int32 BLOCK_SIZE, int32 log_TILE_SIZE, typename tile_id_type>
__global__ void coarse_raster_kernel(
    const int32     grid_size,
    const int32     N_grids,
    const int32     N_blocks,
    const int32     N,
    const int32     log_N,
    const int32     N_tris,
    const float3    bbox0,
    const float3    bbox1,
    const float3    inv_bbox_delta,
    const float4*   vertices,
    const int4*     triangles,
    uint32*         batch_counter,
    int32*          tile_count,
    tile_id_type*   output_tile_ids,
    int32*          output_tile_tris,
    Tri_bbox*       output_tri_bbox)
{
    const int32 WARP_COUNT = BLOCK_SIZE >> 5;

    volatile __shared__ int32 sm_red[ BLOCK_SIZE * 2 ];
    volatile __shared__ int32 sm_broadcast[ WARP_COUNT ];

    const int32 warp_id  = threadIdx.x >> 5;
    const int32 warp_tid = threadIdx.x & 31;

  #if VOXELPIPE_CR_PERSISTENT_THREADS
    for (;;)
    {
        // fetch a new bacth of work for the warp
        if (warp_tid == 0)
            sm_broadcast[ warp_id ] = atomicAdd( batch_counter, 32 );

        // broadcast batch offset to entire warp
        const int32 batch_offset = sm_broadcast[ warp_id ];

        // retire warp when done processing triangles
        if (batch_offset >= N_tris)
            break;

        const int32 tri_idx = batch_offset + warp_tid;
  #else
    for (int32 grid = 0; grid < N_grids; ++grid)
    {
        const int32 tri_idx = threadIdx.x + blockIdx.x*BLOCK_SIZE + grid * grid_size;
  #endif
        const bool valid = (tri_idx < N_tris);

        int3  tri_bbox0i;
        int3  tri_bbox1i;
        int32 tri_axis = 0;

        if (valid)
        {
            // fetch triangle
            const int4 tri = triangles[ tri_idx ];
            const float4 v0 = tex1Dfetch( tex_vertices, tri.x );
            const float4 v1 = tex1Dfetch( tex_vertices, tri.y );
            const float4 v2 = tex1Dfetch( tex_vertices, tri.z );

            // perform coarse setup
            setup_triangle(
                N,
                bbox0,
                bbox1,
                inv_bbox_delta,
                v0, v1, v2,
                tri_bbox0i,
                tri_bbox1i,
                tri_axis );

            // write out triangle bbox
            ((int4*)output_tri_bbox)[ tri_idx*Tri_bbox::N_INT4   ] = make_int4( tri_bbox0i.x, tri_bbox0i.y, tri_bbox0i.z, tri_bbox1i.x );
            ((int4*)output_tri_bbox)[ tri_idx*Tri_bbox::N_INT4+1 ] = make_int4( tri_bbox1i.y, tri_bbox1i.z, tri_axis, 0 );
        }

        // compute and scan the number of output tiles
        const int32 tile_count_x = (tri_bbox1i.x >> log_TILE_SIZE) - (tri_bbox0i.x >> log_TILE_SIZE) + 1;
        const int32 tile_count_y = (tri_bbox1i.y >> log_TILE_SIZE) - (tri_bbox0i.y >> log_TILE_SIZE) + 1;
        const int32 tile_count_z = (tri_bbox1i.z >> log_TILE_SIZE) - (tri_bbox0i.z >> log_TILE_SIZE) + 1;

        const int tri_tile_count = valid ?
            tile_count_x * tile_count_y * tile_count_z :
            0;

        const int32 fragment_scan = scan_warp( tri_tile_count, warp_tid, sm_red + 64 * warp_id );
        if (warp_tid == 31)
            sm_broadcast[ warp_id ] = atomicAdd( tile_count, fragment_scan );

        const int32 offset = sm_broadcast[ warp_id ] + fragment_scan - tri_tile_count;

        // write out triangle-tile id pairs
        if (valid)
        {
            const int32 N_tiles = (N >> log_TILE_SIZE);
            const int32 tile_offset_x = (tri_bbox0i.x >> log_TILE_SIZE);
            const int32 tile_offset_y = (tri_bbox0i.y >> log_TILE_SIZE);
            const int32 tile_offset_z = (tri_bbox0i.z >> log_TILE_SIZE);

            for (int32 x = 0; x < tile_count_x; ++x)
            {
                for (int32 y = 0; y < tile_count_y; ++y)
                {
                    for (int32 z = 0; z < tile_count_z; ++z)
                    {
                        const int32 idx = x + y * tile_count_x + z * tile_count_x * tile_count_y;
                        const int32 tile_id =
                            (x + tile_offset_x) +
                            (y + tile_offset_y) * N_tiles +
                            (z + tile_offset_z) * N_tiles * N_tiles;

                    #if VOXELPIPE_CR_SCANLINE_SORTING
                        // compute clamped bbox side
                        const int32 T = 1 << log_TILE_SIZE;

                        /*const uint32 scanline_len = tri_axis > 0 ?
                            min( tri_bbox1i.x, (x + tile_offset_x + 1)*T - 1 ) - 
                            max( tri_bbox0i.x, (x + tile_offset_x + 0)*T ) :
                            min( tri_bbox1i.y, (y + tile_offset_y + 1)*T - 1 ) - 
                            max( tri_bbox0i.y, (y + tile_offset_y + 0)*T );*/
                        const uint32 scanline_len = tri_axis != 2 ?
                            min( tri_bbox1i.z, (z + tile_offset_z + 1)*T - 1 ) - 
                            max( tri_bbox0i.z, (z + tile_offset_z + 0)*T ) :
                            min( tri_bbox1i.y, (y + tile_offset_y + 1)*T - 1 ) - 
                            max( tri_bbox0i.y, (y + tile_offset_y + 0)*T );

                        const uint32 tri_class = max( min( scanline_len, 15u ), 0u ) | (uint32(tri_axis) << 4);
                    #else
                        const uint32 tri_class = tri_axis;
                    #endif

                        const uint32 tri_descriptor = (uint32(tile_id) << TILE_ID_SHIFT) | tri_class;

                        output_tile_ids[ offset + idx ]  = tri_descriptor;
                        output_tile_tris[ offset + idx ] = tri_idx;
                    }
                }
            }
        }
    }
}

///
/// This function counts how many triangle-tile pairs are emitted for each triangle
/// by calling the corresponding CUDA kernel.
/// Additionally, the (integer grid) bounding box is output for each triangle,
/// together with its dominant axis.
///
template <int32 log_TILE_SIZE, typename tile_id_type>
void coarse_raster(
    const int32     N,
    const int32     log_N,
    const int32     N_tris,
    const int32     N_vertices,
    const float3    bbox0,
    const float3    bbox1,
    const float3    inv_bbox_delta,
    const float4*   vertices,
    const int4*     triangles,
    uint32*         batch_counter,
    int32*          tile_count,
    tile_id_type*   output_tile_ids,
    int32*          output_tile_tris,
    Tri_bbox*       output_tri_bbox)
{
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();

    tex_vertices.normalized = false;
    tex_vertices.filterMode = cudaFilterModePoint;

    cudaBindTexture( 0, &tex_vertices, vertices, &channel_desc, sizeof(float4)*N_vertices );
    {
        uint32 zero = 0;
        cudaMemcpy(
            batch_counter,
            &zero,
            sizeof(uint32),
            cudaMemcpyHostToDevice );
    }
    {
        int32 zero = 0;
        cudaMemcpy(
            tile_count,
            &zero,
            sizeof(int32),
            cudaMemcpyHostToDevice );
    }

    cudaThreadSynchronize();

    const int32 BLOCK_SIZE = 256;
    const int32 block_count = thrust::detail::device::cuda::arch::max_active_blocks(coarse_raster_kernel<BLOCK_SIZE,log_TILE_SIZE,tile_id_type>, BLOCK_SIZE, 0);

    dim3 dim_block = dim3( BLOCK_SIZE, 1, 1 );
    dim3 dim_grid  = dim3( block_count, 1, 1 );

    const int32 grid_size = BLOCK_SIZE * block_count;
    const int32 N_grids   = (N_tris + grid_size-1) / grid_size;

    coarse_raster_kernel<BLOCK_SIZE,log_TILE_SIZE,tile_id_type> <<<dim_grid,dim_block>>>(
        grid_size,
        N_grids,
        block_count,
        N,
        log_N,
        N_tris,
        bbox0,
        bbox1,
        inv_bbox_delta,
        vertices,
        triangles,
        batch_counter,
        tile_count,
        output_tile_ids,
        output_tile_tris,
        output_tri_bbox );

    cudaThreadSynchronize();

    cudaUnbindTexture( &tex_vertices );
}

///
/// This kernel counts how many triangle-tile pairs are emitted for each triangle.
/// Additionally, the (integer grid) bounding box is output for each triangle,
/// together with its dominant axis.
///
template <int32 BLOCK_SIZE, int32 log_TILE_SIZE>
__global__ void coarse_raster_count_kernel(
    const int32     grid_size,
    const int32     N_grids,
    const int32     N_blocks,
    const int32     N,
    const int32     log_N,
    const int32     N_tris,
    const float3    bbox0,
    const float3    bbox1,
    const float3    inv_bbox_delta,
    const float4*   vertices,
    const int4*     triangles,
    int32*          tile_count,
    int32*          block_offsets,
    Tri_bbox*       output_tri_bbox)
{
    const int32 WARP_COUNT = BLOCK_SIZE >> 5;

    volatile __shared__ int sm_red[ BLOCK_SIZE * 2 ];
    volatile __shared__ int sm_red_cta[ WARP_COUNT ];

    int32 global_scan = 0;

    for (int32 grid = 0; grid < N_grids; ++grid)
    {
        const int32 tri_idx = threadIdx.x + blockIdx.x*BLOCK_SIZE + grid * grid_size;

        const bool valid = (tri_idx < N_tris);

        float4 v0, v1, v2;
        if (valid)
        {
            const int4 tri = triangles[ tri_idx ];
            v0 = tex1Dfetch( tex_vertices, tri.x );
            v1 = tex1Dfetch( tex_vertices, tri.y );
            v2 = tex1Dfetch( tex_vertices, tri.z );
        }

        int3  tri_bbox0i;
        int3  tri_bbox1i;
        int32 tri_axis;

        setup_triangle(
            N,
            bbox0,
            bbox1,
            inv_bbox_delta,
            v0, v1, v2,
            tri_bbox0i,
            tri_bbox1i,
            tri_axis );

        // write out triangle bbox
        if (valid)
        {
            ((int4*)output_tri_bbox)[ tri_idx*Tri_bbox::N_INT4   ] = make_int4( tri_bbox0i.x, tri_bbox0i.y, tri_bbox0i.z, tri_bbox1i.x );
            ((int4*)output_tri_bbox)[ tri_idx*Tri_bbox::N_INT4+1 ] = make_int4( tri_bbox1i.y, tri_bbox1i.z, tri_axis, 0 );
        }

        const int32 tile_count_x = (tri_bbox1i.x >> log_TILE_SIZE) - (tri_bbox0i.x >> log_TILE_SIZE) + 1;
        const int32 tile_count_y = (tri_bbox1i.y >> log_TILE_SIZE) - (tri_bbox0i.y >> log_TILE_SIZE) + 1;
        const int32 tile_count_z = (tri_bbox1i.z >> log_TILE_SIZE) - (tri_bbox0i.z >> log_TILE_SIZE) + 1;

        const int tri_tile_count = valid ?
            tile_count_x * tile_count_y * tile_count_z :
            0;

        global_scan += block_reduce<WARP_COUNT>( tri_tile_count, sm_red, sm_red_cta );
    }

    // output the block offset
    if (threadIdx.x == 0)
    {
      #if VOXELPIPE_CR_DETERMINISTIC_OFFSETS
        block_offsets[ blockIdx.x ] = global_scan;
      #else
        block_offsets[ blockIdx.x ] = atomicAdd( tile_count, global_scan );
      #endif
    }
}

///
/// This kernel performs the coarse rasterization setup, by emitting an unsorted list
/// of triangle-tile pairs.
/// The tile id is OR´ed with the dominant axis in the lowest bits, so as to allow
/// efficient sorting by tile id and dominant axis at the same time.
///
template <int32 BLOCK_SIZE, int32 log_TILE_SIZE, typename tile_id_type>
__global__ void coarse_raster_setup_kernel(
    const int32     grid_size,
    const int32     N_grids,
    const int32     N,
    const int32     log_N,
    const int32     N_tris,
    const Tri_bbox* tri_bbox,
    const int32*    block_offsets,
    tile_id_type*   output_tile_ids,
    int32*          output_tile_tris)
{
    const int32 WARP_COUNT = BLOCK_SIZE >> 5;

    int32 block_offset = block_offsets[ blockIdx.x ];

    volatile __shared__ int sm_red[ BLOCK_SIZE * 2 ];
    volatile __shared__ int sm_red_cta[ WARP_COUNT+1 ];

    for (int32 grid = 0; grid < N_grids; ++grid)
    {
        const int32 tri_idx = threadIdx.x + blockIdx.x*BLOCK_SIZE + grid * grid_size;

        const bool valid = (tri_idx < N_tris);

        int3 tri_bbox0i;
        int3 tri_bbox1i;
        int32 tri_axis;
        fetch_tri_bbox( tri_bbox, tri_idx, tri_bbox0i, tri_bbox1i, tri_axis );

        const int32 tile_count_x = (tri_bbox1i.x >> log_TILE_SIZE) - (tri_bbox0i.x >> log_TILE_SIZE) + 1;
        const int32 tile_count_y = (tri_bbox1i.y >> log_TILE_SIZE) - (tri_bbox0i.y >> log_TILE_SIZE) + 1;
        const int32 tile_count_z = (tri_bbox1i.z >> log_TILE_SIZE) - (tri_bbox0i.z >> log_TILE_SIZE) + 1;

        const int tri_tile_count = valid ?
            tile_count_x * tile_count_y * tile_count_z :
            0;

        const int32 offset = block_scan<WARP_COUNT>( tri_tile_count, sm_red, sm_red_cta ) - tri_tile_count + block_offset;

        block_offset += block_scan_total<WARP_COUNT>( sm_red_cta );

        // write out triangle-tile id pairs
        if (valid)
        {
            const int32 N_tiles = (N >> log_TILE_SIZE);
            const int32 tile_offset_x = (tri_bbox0i.x >> log_TILE_SIZE);
            const int32 tile_offset_y = (tri_bbox0i.y >> log_TILE_SIZE);
            const int32 tile_offset_z = (tri_bbox0i.z >> log_TILE_SIZE);

            for (int32 x = 0; x < tile_count_x; ++x)
                for (int32 y = 0; y < tile_count_y; ++y)
                    for (int32 z = 0; z < tile_count_z; ++z)
                    {
                        const int32 i = x + y * tile_count_x + z * tile_count_x * tile_count_y;
                        const int32 tile_id =
                            (x + tile_offset_x) +
                            (y + tile_offset_y) * N_tiles +
                            (z + tile_offset_z) * N_tiles * N_tiles;

                        //output_tile_ids[ offset + i ]  = (tile_id << TILE_ID_SHIFT) | (tri_axis << 2) | tri_class;
                        output_tile_ids[ offset + i ]  = (uint32(tile_id) << TILE_ID_SHIFT) | uint32(tri_axis);
                        output_tile_tris[ offset + i ] = tri_idx;
                    }
        }
    }
}

///
/// This function counts how many triangle-tile pairs are emitted for each triangle
/// by calling the corresponding CUDA kernel.
/// Additionally, the (integer grid) bounding box is output for each triangle,
/// together with its dominant axis.
///
template <int32 log_TILE_SIZE, typename tile_id_type>
void coarse_raster_count(
    const int32     N,
    const int32     log_N,
    const int32     N_tris,
    const int32     N_vertices,
    const float3    bbox0,
    const float3    bbox1,
    const float3    inv_bbox_delta,
    const float4*   vertices,
    const int4*     triangles,
    int32*          tile_count,
    int32*          block_offsets,
    Tri_bbox*       output_tri_bbox)
{
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();

    tex_vertices.normalized = false;
    tex_vertices.filterMode = cudaFilterModePoint;

    cudaBindTexture( 0, &tex_vertices, vertices, &channel_desc, sizeof(float4)*N_vertices );

    int32 zero = 0;
    cudaMemcpy(
        tile_count,
        &zero,
        sizeof(int32),
        cudaMemcpyHostToDevice );

    cudaThreadSynchronize();

    const int32 BLOCK_SIZE = 256;
    const int32 block_count0 = thrust::detail::device::cuda::arch::max_active_blocks(coarse_raster_count_kernel<BLOCK_SIZE,log_TILE_SIZE>, BLOCK_SIZE, 0);
    const int32 block_count1 = thrust::detail::device::cuda::arch::max_active_blocks(coarse_raster_setup_kernel<BLOCK_SIZE,log_TILE_SIZE,tile_id_type>, BLOCK_SIZE, 0);
    const int32 block_count = min( block_count0, block_count1 );

    dim3 dim_block = dim3( BLOCK_SIZE, 1, 1 );
    dim3 dim_grid  = dim3( block_count, 1, 1 );

    const int32 grid_size = BLOCK_SIZE * block_count;
    const int32 N_grids   = (N_tris + grid_size-1) / grid_size;

    coarse_raster_count_kernel<BLOCK_SIZE,log_TILE_SIZE> <<<dim_grid,dim_block>>>(
        grid_size,
        N_grids,
        block_count,
        N,
        log_N,
        N_tris,
        bbox0,
        bbox1,
        inv_bbox_delta,
        vertices,
        triangles,
        tile_count,
        block_offsets,
        output_tri_bbox );

    cudaThreadSynchronize();

    cudaUnbindTexture( &tex_vertices );

  #if VOXELPIPE_CR_DETERMINISTIC_OFFSETS
    exclusive_scan( block_count, block_offsets, tile_count );
  #endif
}

///
/// This function performs the coarse rasterization setup, by emitting an unsorted list
/// of triangle-tile pairs, by calling the corresponding CUDA kernel.
/// The tile id is OR´ed with the dominant axis in the lowest bits, so as to allow
/// efficient sorting by tile id and dominant axis at the same time.
///
template <int32 log_TILE_SIZE, typename tile_id_type>
void coarse_raster_setup(
    const int32     N,
    const int32     log_N,
    const int32     N_tris,
    const Tri_bbox* tri_bbox,
    const int32*    block_offsets,
    tile_id_type*   output_tile_ids,
    int32*          output_tile_tris)
{
    cudaThreadSynchronize();

    const int32 BLOCK_SIZE = 256;
    const int32 block_count0 = thrust::detail::device::cuda::arch::max_active_blocks(coarse_raster_count_kernel<BLOCK_SIZE,log_TILE_SIZE>, BLOCK_SIZE, 0);
    const int32 block_count1 = thrust::detail::device::cuda::arch::max_active_blocks(coarse_raster_setup_kernel<BLOCK_SIZE,log_TILE_SIZE,tile_id_type>, BLOCK_SIZE, 0);
    const int32 block_count = min( block_count0, block_count1 );

    dim3 dim_block = dim3( BLOCK_SIZE, 1, 1 );
    dim3 dim_grid  = dim3( block_count, 1, 1 );

    const int32 grid_size = BLOCK_SIZE * block_count;
    const int32 N_grids   = (N_tris + grid_size-1) / grid_size;

    coarse_raster_setup_kernel<BLOCK_SIZE,log_TILE_SIZE> <<<dim_grid,dim_block>>>(
        grid_size,
        N_grids,
        N,
        log_N,
        N_tris,
        tri_bbox,
        block_offsets,
        output_tile_ids,
        output_tile_tris );

    cudaThreadSynchronize();
}

///
/// Compute the (begin,end) range of triangles corresponding to each tile.
///
template <int32 BLOCK_SIZE, typename tile_id_type>
__global__ void compute_tile_ranges(
    const int32         grid_size,
    const int32         n_grids,
    const int32         N_tri_tiles,
    const tile_id_type* tile_ids,
    int32*              tile_start,
    int32*              tile_end)
{
    ///
    /// Each thread reads the tile id corresponding to a given triangle-tile pair and
    /// compares it with its predecessor: if it differs, it writes the corresponding
    /// index in the corresponding tile_start and in the predecessor´s tile_end.
    ///
    const int32 thread_id = threadIdx.x + blockIdx.x*BLOCK_SIZE;

    for (int32 grid = 0; grid < n_grids; ++grid)
    {
        const int32 i = thread_id + 1 + grid*grid_size;
        if (i >= N_tri_tiles)
            return;

        const uint32 id      = tile_ids[i]   >> TILE_ID_SHIFT;
        const uint32 id_prev = tile_ids[i-1] >> TILE_ID_SHIFT;

        if (id != id_prev)
        {
            tile_start[id]    = i;
            tile_end[id_prev] = i;
        }
    }
}

///
/// Compute the (begin,end) range of triangles corresponding to each tile.
///
template <typename tile_id_type>
void compute_tile_ranges(
    const int32         N_tri_tiles,
    const tile_id_type* tile_ids,
    int32*              tile_start,
    int32*              tile_end)
{
    const int32 BLOCK_SIZE = 512;
    dim3 dim_block = dim3( BLOCK_SIZE, 1, 1 );
    dim3 dim_grid  = dim3( thrust::detail::device::cuda::arch::max_active_blocks(compute_tile_ranges<BLOCK_SIZE,tile_id_type>, BLOCK_SIZE, 0), 1, 1 );

    const int32 grid_size = dim_grid.x * BLOCK_SIZE;
    const int32 n_grids   = (N_tri_tiles + grid_size-1) / grid_size;

    compute_tile_ranges<BLOCK_SIZE> <<<dim_grid,dim_block>>>(
        grid_size,
        n_grids,
        N_tri_tiles,
        tile_ids,
        tile_start,
        tile_end );

    cudaThreadSynchronize();
}

} // namespace voxelpipe
