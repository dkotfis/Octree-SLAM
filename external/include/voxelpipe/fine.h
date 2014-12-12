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

/*! \file fine.h
 *  \brief Fine Raster kernels
 */

#pragma once

#include <voxelpipe/common.h>
#include <voxelpipe/utils.h>
#include <voxelpipe/tile.h>

namespace voxelpipe {

namespace FR {

inline __device__
int32 find_fragment(const int32 x, volatile int32* frag_scan, const int32 offset, const int32 count)
{
    // find the first element strictly greater than x
    int32 lo = 0;
    int32 hi = count;

    while (hi - lo > 1)
    {
        const int32 mid = (lo + hi) >> 1;
        const float mid_val = frag_scan[ (mid + offset) & 63 ];

        if (x < mid_val)
            hi = mid;
        else
            lo = mid;
    }
    if (lo == count-1)
        return count-1;

    return x >= frag_scan[ (lo + offset) & 63 ] ? lo + 1 : lo;
}

struct FRStats
{
    FRStats() {}

    int32*  samples_sum;
    int32*  samples_sum2;
    int32*  samples_count;
    int32*  utilization_sum[2];
    int32*  utilization_count[2];
    uint32* fragment_count;
};

template <typename shader_type, int32 BLOCK_SIZE, int32 LOG_TILE_SIZE, int32 VoxelType, int32 VoxelizationType, int32 BlendingMode>
struct fine_raster_block {};

template <typename shader_type, int32 BLOCK_SIZE, int32 LOG_TILE_SIZE, int32 VoxelType, int32 BlendingMode>
struct fine_raster_block<shader_type, BLOCK_SIZE,LOG_TILE_SIZE,VoxelType,CONSERVATIVE_RASTER,BlendingMode>
{
    typedef TileOp<VoxelType,BlendingMode,LOG_TILE_SIZE> tile_op_type;
    typedef typename tile_op_type::storage_type          storage_type;
    typedef typename tile_op_type::pixel_type            pixel_type;

    __device__ static void process(
        storage_type*   sm_tile,
        const int32     tile_id,
        const int32     N,
        const int32     log_N,
        const Tri_bbox* tri_bbox,
        const float3    bbox0,
        const float3    bbox1,
        const float3    bbox_delta,
        const float3    inv_bbox_delta,
        const int32     tile_begin,
        const int32     tile_end,
        const int32*    tile_tris,
        storage_type*   tile,
        FRStats         prf_stats,
        shader_type     shader);
};

template <typename shader_type, int32 BLOCK_SIZE, int32 LOG_TILE_SIZE, int32 VoxelType, int32 BlendingMode>
struct fine_raster_block<shader_type, BLOCK_SIZE,LOG_TILE_SIZE,VoxelType,THIN_RASTER,BlendingMode>
{
    typedef TileOp<VoxelType,BlendingMode,LOG_TILE_SIZE> tile_op_type;
    typedef typename tile_op_type::storage_type          storage_type;
    typedef typename tile_op_type::pixel_type            pixel_type;

    __device__ static void process(
        storage_type*   sm_tile,
        const int32     tile_id,
        const int32     N,
        const int32     log_N,
        const Tri_bbox* tri_bbox,
        const float3    bbox0,
        const float3    bbox1,
        const float3    bbox_delta,
        const float3    inv_bbox_delta,
        const int32     tile_begin,
        const int32     tile_end,
        const int32*    tile_tris,
        storage_type*   tile,
        FRStats         prf_stats,
        shader_type     shader);
};

inline __device__ void compute_scanline_bounds(
    const float3 b,
    const float3 n_delta_u,
    const float3 inv_delta_u,
    int32&       min_u,
    int32&       max_u)
{
    // b.x + u * n_delta_u.x >= 0.0f <=> (n_delta_u.x > 0 ? u >= -b.x / n_delta_u.x : u <= -b.x / n_delta_u.x);
    if (n_delta_u.x > 0.0f)        min_u = max( min_u, int32( ceilf( -b.x * inv_delta_u.x ) ) );
    else if (n_delta_u.x < 0.0f)   max_u = min( max_u, int32( -b.x * inv_delta_u.x ) );
    else if (b.x < 0.0f)
        min_u = max_u+1; // invalid range

    if (n_delta_u.y > 0.0f)        min_u = max( min_u, int32( ceilf( -b.y * inv_delta_u.y ) ) );
    else if (n_delta_u.y < 0.0f)   max_u = min( max_u, int32( -b.y * inv_delta_u.y ) );
    else if (b.y < 0.0f)
        min_u = max_u+1; // invalid range

    if (n_delta_u.z > 0.0f)        min_u = max( min_u, int32( ceilf( -b.z * inv_delta_u.z ) ) );
    else if (n_delta_u.z < 0.0f)   max_u = min( max_u, int32( -b.z * inv_delta_u.z ) );
    else if (b.z < 0.0f)
        min_u = max_u+1; // invalid range
}

template <int32 AXIS, int32 LOG_TILE_SIZE, int32 BLOCK_STRIDE>
__device__ __inline__ void generate_mask(
    volatile uint32* sm_mask,
    const int32   tri_id,
    const int3    tile,
    const float3  bbox0,
    const float3  bbox_delta,
    const float3  inv_bbox_delta,
    const int3    tri_bbox0,
    const int3    tri_bbox1,
    const float4  v0,
    const float4  v1,
    const float4  v2,
    const float3  edge0,
    const float3  edge1,
    const float3  edge2,
    const float3  n,
    FRStats       prf_stats)
{
    typedef uvw<AXIS> sel;

    float3 n_delta_u;
    float3 n_delta_v;
    float3 a;
    float4 plane_eq;

    triangle_setup<AXIS>(
        bbox0,
        bbox_delta,
        inv_bbox_delta,
        v0,
        v1,
        v2,
        edge0,
        edge1,
        edge2,
        n,
        n_delta_u,
        n_delta_v,
        a );

    plane_setup<AXIS>(
        bbox0,
        v0,
        n,
        plane_eq );

#if VOXELPIPE_ENABLE_PROFILING
    const int32 global_warp_id = (threadIdx.x + blockIdx.x*blockDim.x) >> 5;
    const int32 warp_id  = (threadIdx.x >> 5);
    const int32 warp_tid = (threadIdx.x & 31);

    // storage for utilization stats
             __shared__ int32 sm_utilization_sum[ 32 ];
    volatile __shared__ int32 sm_utilization_count[ 32 ];

    sm_utilization_sum[ warp_id ]   = prf_stats.utilization_sum[1][ global_warp_id ];
    sm_utilization_count[ warp_id ] = prf_stats.utilization_count[1][ global_warp_id ];

    // keep stats on the number of sample tests
    const int32 v_samples = sel::v( tri_bbox1 ) - sel::v( tri_bbox0 ) + 1;

    atomicAdd( &prf_stats.samples_sum[ global_warp_id ], v_samples );
    atomicAdd( &prf_stats.samples_sum2[ global_warp_id ], v_samples*v_samples );
    if (warp_tid == 0)
        prf_stats.samples_count[ global_warp_id ] += 32;
#endif

    #if VOXELPIPE_FR_SCANLINE_CLIPPING
    const float3 inv_delta_u = make_float3(
        1.0f / n_delta_u.x,
        1.0f / n_delta_u.y,
        1.0f / n_delta_u.z );
    #endif

    for (int32 v = sel::v( tri_bbox0 ); v <= sel::v( tri_bbox1 ); ++v)
    {
        #if VOXELPIPE_ENABLE_PROFILING
        {
            // keep utilization stats
            const int32 active_mask = __ballot( true );
            sm_utilization_sum[ warp_id ]   += __popc( active_mask );
            sm_utilization_count[ warp_id ] += 32;
        }
        #endif

        const float3 b = make_float3(
            a.x + float(v) * n_delta_v.x,
            a.y + float(v) * n_delta_v.y,
            a.z + float(v) * n_delta_v.z );

        int32 min_u = sel::u( tri_bbox0 );
        int32 max_u = sel::u( tri_bbox1 );
        compute_scanline_bounds( b, n_delta_u, inv_delta_u, min_u, max_u );

        // make sure invalid ranges are representable in our packed format
        if (min_u > max_u)
        {
            min_u = sel::u( tile )+1;
            max_u = sel::u( tile );
        }

        const uint32 l_mask = uint32(min_u - sel::u( tile ));
        const uint32 r_mask = uint32(max_u - sel::u( tile )) << LOG_TILE_SIZE;

        const uint32 bit_index   = 2*LOG_TILE_SIZE * (v - sel::v( tri_bbox0 ));
        const uint32 word_index  = bit_index >> 5;
        const uint32 word_offset = bit_index & 31;

        const uint32 mask = (l_mask | r_mask) << word_offset;
        sm_mask[ word_index * BLOCK_STRIDE ] |= mask;
    }

#if VOXELPIPE_ENABLE_PROFILING
    // write utilization stats to gmem
    prf_stats.utilization_sum[1][ global_warp_id ]   = sm_utilization_sum[ warp_id ];
    prf_stats.utilization_count[1][ global_warp_id ] = sm_utilization_count[ warp_id ];
#endif
};

template <int32 AXIS, int32 LOG_TILE_SIZE, int32 BLOCK_STRIDE, typename storage_type, typename tile_op_type, typename shader_type>
__device__ __inline__ void rasterize_scanline(
    volatile uint32* sm_mask,
    storage_type*    sm_tile,
    const int32   tri_id,
    const int32   scanline_base,
    const int32   scanline,
    const int3    tile,
    const float3  bbox0,
    const float3  bbox_delta,
    const float3  inv_bbox_delta,
    const float4  v0,
    const float4  v1,
    const float4  v2,
    const float3  edge0,
    const float3  edge1,
    const float3  edge2,
    const float3  n,
    FRStats       prf_stats,
    shader_type   shader)
{
    typedef uvw<AXIS> sel;
    typedef typename tile_op_type::pixel_type pixel_type;

    const uint32 T = 1 << LOG_TILE_SIZE;

    float4 plane_eq;

    // precompute plane equation
    plane_setup<AXIS>(
        bbox0,
        v0,
        n,
        plane_eq );

    // precompute the barycentrics matrix for the projected triangle
    //
    // We take (v0, edge0, -edge2) as a reference frame.
    // In this frame the barycentrics can be obtained solving for (edge0,-edge2) * (u,v)^t = p - v0.
    // Hence, we precalculate the rows of the matrix inv(edge0,-edge2), called e0 and e1.
    const float  bary_det = -plane_eq.w;
    const float2 e0 = make_float2( -sel::v( edge2 ) * bary_det, sel::u( edge2 ) * bary_det );
    const float2 e1 = make_float2( -sel::v( edge0 ) * bary_det, sel::u( edge0 ) * bary_det );

    const int32 v  = scanline_base + scanline;
    const float vf = (v+0.5f) * sel::v( bbox_delta );

    // fetch scanline
    const uint32 bit_index   = 2*LOG_TILE_SIZE * scanline;
    const uint32 word_index  = bit_index >> 5;
    const uint32 word_offset = bit_index & 31;

    const uint32 mask  = sm_mask[ word_index * BLOCK_STRIDE ] >> word_offset;
    const uint32 min_u = (mask & (T-1))                    + sel::u( tile );
    const uint32 max_u = ((mask >> LOG_TILE_SIZE) & (T-1)) + sel::u( tile );

    for (int32 u = min_u; u <= max_u; ++u)
    {
        // compute W depth
        const float uf = (u+0.5f) * sel::u( bbox_delta );
        const float wf = plane_eq.z - (plane_eq.x*uf + plane_eq.y*vf);
        const int32 w = int32( wf * sel::w( inv_bbox_delta ) );

        if (w >= sel::w( tile ) && w < sel::w( tile ) + T)
        {
            VOXELPIPE_PROFILE( atomicAdd( prf_stats.fragment_count, 1 ) );

            const int32 u_stride = sel::u_stride( T );
            const int32 v_stride = sel::v_stride( T );
            const int32 w_stride = sel::w_stride( T );

            const int32 pixel =
                (u - sel::u( tile ))*u_stride +
                (v - sel::v( tile ))*v_stride +
                (w - sel::w( tile ))*w_stride;

            // compute barycentrics
            const float2 p = make_float2( uf - sel::u( v0 ), vf - sel::v( v0 ) );
            const float bary0 = e0.x * p.x + e0.y * p.y;
            const float bary1 = e1.x * p.x + e1.y * p.y;

            // compute pixel coordinates
            const int3 uvw = make_int3( u, v, w );
            const int3 xyz = make_int3( sel::x( uvw ), sel::y( uvw ), sel::z( uvw ) );

            // shade the fragment
            const pixel_type value = shader.shade( tri_id, v0, v1, v2, n, bary0, bary1, xyz );

            // write to the FB
            tile_op_type::write( sm_tile, pixel, value );
        }
    }
};

template <int32 AXIS, int32 T, typename storage_type, typename tile_op_type, typename shader_type>
__device__ __inline__ void rasterize(
    storage_type* sm_tile,
    const int32   tri_id,
    const int3    tile,
    const float3  bbox0,
    const float3  bbox_delta,
    const float3  inv_bbox_delta,
    const int3    tri_bbox0,
    const int3    tri_bbox1,
    const float4  v0,
    const float4  v1,
    const float4  v2,
    const float3  edge0,
    const float3  edge1,
    const float3  edge2,
    const float3  n,
    FRStats       prf_stats,
    shader_type   shader)
{
    typedef uvw<AXIS> sel;
    typedef typename tile_op_type::pixel_type pixel_type;

    float3 n_delta_u;
    float3 n_delta_v;
    float3 a;
    float4 plane_eq;

    triangle_setup<AXIS>(
        bbox0,
        bbox_delta,
        inv_bbox_delta,
        v0,
        v1,
        v2,
        edge0,
        edge1,
        edge2,
        n,
        n_delta_u,
        n_delta_v,
        a );

    plane_setup<AXIS>(
        bbox0,
        v0,
        n,
        plane_eq );

    // precompute the barycentrics matrix for the projected triangle
    //
    // We take (v0, edge0, -edge2) as a reference frame.
    // In this frame the barycentrics can be obtained solving for (edge0,-edge2) * (u,v)^t = p - v0.
    // Hence, we precalculate the rows of the matrix inv(edge0,-edge2), called e0 and e1.
    const float  bary_det = -plane_eq.w;
    const float2 e0 = make_float2( -sel::v( edge2 ) * bary_det, sel::u( edge2 ) * bary_det );
    const float2 e1 = make_float2( -sel::v( edge0 ) * bary_det, sel::u( edge0 ) * bary_det );

#if VOXELPIPE_ENABLE_PROFILING
    const int32 global_warp_id = (threadIdx.x + blockIdx.x*blockDim.x) >> 5;
    const int32 warp_id  = (threadIdx.x >> 5);
    const int32 warp_tid = (threadIdx.x & 31);

    // storage for utilization stats
             __shared__ int32 sm_utilization_sum[ 32 ];
    volatile __shared__ int32 sm_utilization_count[ 32 ];

    sm_utilization_sum[ warp_id ]   = prf_stats.utilization_sum[1][ global_warp_id ];
    sm_utilization_count[ warp_id ] = prf_stats.utilization_count[1][ global_warp_id ];

    // keep stats on the number of sample tests
    const int32 u_samples = sel::u( tri_bbox1 ) - sel::u( tri_bbox0 ) + 1;
    const int32 v_samples = sel::v( tri_bbox1 ) - sel::v( tri_bbox0 ) + 1;
    const int32 uv_samples = u_samples * v_samples;

    atomicAdd( &prf_stats.samples_sum[ global_warp_id ], uv_samples );
    atomicAdd( &prf_stats.samples_sum2[ global_warp_id ], uv_samples*uv_samples );
    if (warp_tid == 0)
        prf_stats.samples_count[ global_warp_id ] += 32;
#endif

    #if VOXELPIPE_FR_SCANLINE_CLIPPING
    const float3 inv_delta_u = make_float3(
        1.0f / n_delta_u.x,
        1.0f / n_delta_u.y,
        1.0f / n_delta_u.z );
    #endif

    for (int32 v = sel::v( tri_bbox0 ); v <= sel::v( tri_bbox1 ); ++v)
    {
        const float3 b = make_float3(
            a.x + float(v) * n_delta_v.x,
            a.y + float(v) * n_delta_v.y,
            a.z + float(v) * n_delta_v.z );

        int32 min_u = sel::u( tri_bbox0 );
        int32 max_u = sel::u( tri_bbox1 );
        #if VOXELPIPE_FR_SCANLINE_CLIPPING
        compute_scanline_bounds( b, n_delta_u, inv_delta_u, min_u, max_u );
        #endif

        for (int32 u = min_u; u <= max_u; ++u)
        {
            #if VOXELPIPE_ENABLE_PROFILING
            {
                // keep utilization stats
                const int32 active_mask = __ballot( true );
                sm_utilization_sum[ warp_id ]   += __popc( active_mask );
                sm_utilization_count[ warp_id ] += 32;
            }
            #endif

        #if VOXELPIPE_FR_SCANLINE_CLIPPING
            const bool inside = true;
        #else
            const float c0 = b.x + float(u) * n_delta_u.x;
            const float c1 = b.y + float(u) * n_delta_u.y;
            const float c2 = b.z + float(u) * n_delta_u.z;

            const bool inside =
                (c0 >= 0.0f) &&
                (c1 >= 0.0f) &&
                (c2 >= 0.0f) ;
        #endif

            if (inside) // set the bit for this column
            {
                // compute W depth
                const float uf = (u+0.5f) * sel::u( bbox_delta );
                const float vf = (v+0.5f) * sel::v( bbox_delta );
                const float wf = plane_eq.z - (plane_eq.x*uf + plane_eq.y*vf);
                //const float wf = wf_base - n_u*uf;
                const int32 w = int32( wf * sel::w( inv_bbox_delta ) );

                if (w >= sel::w( tile ) && w < sel::w( tile ) + T)
                {
                    VOXELPIPE_PROFILE( atomicAdd( prf_stats.fragment_count, 1 ) );

                    const int32 u_stride = sel::u_stride( T );
                    const int32 v_stride = sel::v_stride( T );
                    const int32 w_stride = sel::w_stride( T );

                    const int32 pixel =
                        (u - sel::u( tile ))*u_stride +
                        (v - sel::v( tile ))*v_stride +
                        (w - sel::w( tile ))*w_stride;

                    // compute barycentrics
                    const float2 p = make_float2( uf - sel::u( v0 ), vf - sel::v( v0 ) );
                    const float bary0 = e0.x * p.x + e0.y * p.y;
                    const float bary1 = e1.x * p.x + e1.y * p.y;

                    // compute pixel coordinates
                    const int3 uvw = make_int3( u, v, w );
                    const int3 xyz = make_int3( sel::x( uvw ), sel::y( uvw ), sel::z( uvw ) );

                    // shade the fragment
                    const pixel_type value = shader.shade( tri_id, v0, v1, v2, n, bary0, bary1, xyz );

                    // write to the FB
                    tile_op_type::write( sm_tile, pixel, value );
                }
            }
        }
    }

#if VOXELPIPE_ENABLE_PROFILING
    // write utilization stats to gmem
    prf_stats.utilization_sum[1][ global_warp_id ]   = sm_utilization_sum[ warp_id ];
    prf_stats.utilization_count[1][ global_warp_id ] = sm_utilization_count[ warp_id ];
#endif
};

template <int32 AXIS, int32 T, typename storage_type, typename tile_op_type, typename shader_type>
__device__ __inline__ void rasterize_26sep(
    storage_type* sm_tile,
    const int32   tri_id,
    const int3    tile,
    const float3  bbox0,
    const float3  bbox_delta,
    const float3  inv_bbox_delta,
    const int3    tri_bbox0,
    const int3    tri_bbox1,
    const float4  v0,
    const float4  v1,
    const float4  v2,
    const float3  edge0,
    const float3  edge1,
    const float3  edge2,
    const float3  n,
    FRStats       prf_stats,
    shader_type   shader)
{
    typedef uvw<AXIS> sel;
    typedef typename tile_op_type::pixel_type pixel_type;

    float3 w_n_delta_u;
    float3 w_n_delta_v;
    float3 w_a;

    float3 v_n_delta_u;
    float3 v_n_delta_v;
    float3 v_a;

    float3 u_n_delta_u;
    float3 u_n_delta_v;
    float3 u_a;

    float4 plane_eq;

    plane_setup<sel::W>(
        bbox0,
        v0,
        n,
        plane_eq );

    triangle_setup<sel::W>(
        bbox0,
        bbox_delta,
        inv_bbox_delta,
        v0,
        v1,
        v2,
        edge0,
        edge1,
        edge2,
        n,
        w_n_delta_u,
        w_n_delta_v,
        w_a );

    triangle_setup<sel::U>(
        bbox0,
        bbox_delta,
        inv_bbox_delta,
        v0,
        v1,
        v2,
        edge0,
        edge1,
        edge2,
        n,
        u_n_delta_u,
        u_n_delta_v,
        u_a );

    triangle_setup<sel::V>(
        bbox0,
        bbox_delta,
        inv_bbox_delta,
        v0,
        v1,
        v2,
        edge0,
        edge1,
        edge2,
        n,
        v_n_delta_u,
        v_n_delta_v,
        v_a );

    // precompute the barycentrics matrix for the projected triangle
    //
    // We take (v0, edge0, -edge2) as a reference frame.
    // In this frame the barycentrics can be obtained solving for (edge0,-edge2) * (u,v)^t = p - v0.
    // Hence, we precalculate the rows of the matrix inv(edge0,-edge2), called e0 and e1.
    const float  bary_det = -plane_eq.w;
    const float2 e0 = make_float2( -sel::v( edge2 ) * bary_det, sel::u( edge2 ) * bary_det );
    const float2 e1 = make_float2( -sel::v( edge0 ) * bary_det, sel::u( edge0 ) * bary_det );

#if VOXELPIPE_ENABLE_PROFILING
    const int32 global_warp_id = (threadIdx.x + blockIdx.x*blockDim.x) >> 5;
    const int32 warp_id  = (threadIdx.x >> 5);
    const int32 warp_tid = (threadIdx.x & 31);

    // storage for utilization stats
             __shared__ int32 sm_utilization_sum[ 32 ];
    volatile __shared__ int32 sm_utilization_count[ 32 ];

    sm_utilization_sum[ warp_id ]   = prf_stats.utilization_sum[1][ global_warp_id ];
    sm_utilization_count[ warp_id ] = prf_stats.utilization_count[1][ global_warp_id ];

    // keep stats on the number of sample tests
    const int32 u_samples = sel::u( tri_bbox1 ) - sel::u( tri_bbox0 ) + 1;
    const int32 v_samples = sel::v( tri_bbox1 ) - sel::v( tri_bbox0 ) + 1;
    const int32 uv_samples = u_samples * v_samples;

    atomicAdd( &prf_stats.samples_sum[ global_warp_id ], uv_samples );
    atomicAdd( &prf_stats.samples_sum2[ global_warp_id ], uv_samples*uv_samples );
    if (warp_tid == 0)
        prf_stats.samples_count[ global_warp_id ] += 32;
#endif

    #if VOXELPIPE_FR_SCANLINE_CLIPPING
    const float3 w_inv_delta_u = make_float3(
        1.0f / w_n_delta_u.x,
        1.0f / w_n_delta_u.y,
        1.0f / w_n_delta_u.z );
    #endif

    for (int32 v = sel::v( tri_bbox0 ); v <= sel::v( tri_bbox1 ); ++v)
    {
        const float3 b = make_float3(
            w_a.x + float(v) * w_n_delta_v.x,
            w_a.y + float(v) * w_n_delta_v.y,
            w_a.z + float(v) * w_n_delta_v.z );

        int32 min_u = sel::u( tri_bbox0 );
        int32 max_u = sel::u( tri_bbox1 );
        #if VOXELPIPE_FR_SCANLINE_CLIPPING
        compute_scanline_bounds( b, w_n_delta_u, w_inv_delta_u, min_u, max_u );
        #endif

        for (int32 u = min_u; u <= max_u; ++u)
        {
            #if VOXELPIPE_ENABLE_PROFILING
            {
                // keep utilization stats
                const int32 active_mask = __ballot( true );
                sm_utilization_sum[ warp_id ]   += __popc( active_mask );
                sm_utilization_count[ warp_id ] += 32;
            }
            #endif

        #if VOXELPIPE_FR_SCANLINE_CLIPPING
            const bool inside = true;
        #else
            const float c0 = b.x + float(u) * w_n_delta_u.x;
            const float c1 = b.y + float(u) * w_n_delta_u.y;
            const float c2 = b.z + float(u) * w_n_delta_u.z;

            const bool inside =
                (c0 >= 0.0f) &&
                (c1 >= 0.0f) &&
                (c2 >= 0.0f) ;
        #endif

            if (inside) // set the bit for this column
            {
                // compute W depth range
                const float min_uf = (u + (plane_eq.x < 0.0f ? 1.0f : 0.0f)) * sel::u( bbox_delta );
                const float min_vf = (v + (plane_eq.y < 0.0f ? 1.0f : 0.0f)) * sel::v( bbox_delta );
                const float min_wf = plane_eq.z - (plane_eq.x*min_uf + plane_eq.y*min_vf);
                      int32 min_w  = int32( min_wf * sel::w( inv_bbox_delta ) );

                const float max_uf = (u + (plane_eq.x > 0.0f ? 1.0f : 0.0f)) * sel::u( bbox_delta );
                const float max_vf = (v + (plane_eq.y > 0.0f ? 1.0f : 0.0f)) * sel::v( bbox_delta );
                const float max_wf = plane_eq.z - (plane_eq.x*max_uf + plane_eq.y*max_vf);
                      int32 max_w  = int32( max_wf * sel::w( inv_bbox_delta ) );

                if (max_w < min_w)
                {
                    int32 tmp = min_w;
                    min_w = max_w;
                    max_w = tmp;
                }

                min_w = max( min_w, sel::w( tile ) );
                max_w = min( max_w, sel::w( tile ) + T - 1 );

                // do the remaining 2 projection tests for each voxel in the w-column
                for (int32 w = min_w; w <= max_w; ++w)
                {
                    // U-axis test
                    {
                        const float u_u = AXIS == 2 ? float(v) : float(w);
                        const float u_v = AXIS == 2 ? float(w) : float(v);

                        // test the U-axis
                        const float3 u_c = make_float3(
                            u_a.x + u_u * u_n_delta_u.x + u_v * u_n_delta_v.x,
                            u_a.y + u_u * u_n_delta_u.y + u_v * u_n_delta_v.y,
                            u_a.z + u_u * u_n_delta_u.z + u_v * u_n_delta_v.z );

                        if ((u_c.x < 0.0f) ||
                            (u_c.y < 0.0f) ||
                            (u_c.z < 0.0f))
                            continue;
                    }

                    // V-axis test
                    {
                        const float v_u = AXIS == 0 ? float(w) : float(u);
                        const float v_v = AXIS == 0 ? float(u) : float(w);

                        // test the V-axis
                        const float3 v_c = make_float3(
                            v_a.x + v_u * v_n_delta_u.x + v_v * v_n_delta_v.x,
                            v_a.y + v_u * v_n_delta_u.y + v_v * v_n_delta_v.y,
                            v_a.z + v_u * v_n_delta_u.z + v_v * v_n_delta_v.z );

                        if ((v_c.x < 0.0f) ||
                            (v_c.y < 0.0f) ||
                            (v_c.z < 0.0f))
                            continue;
                    }

                    VOXELPIPE_PROFILE( atomicAdd( prf_stats.fragment_count, 1 ) );

                    const int32 u_stride = sel::u_stride( T );
                    const int32 v_stride = sel::v_stride( T );
                    const int32 w_stride = sel::w_stride( T );

                    const int32 pixel =
                        (u - sel::u( tile ))*u_stride +
                        (v - sel::v( tile ))*v_stride +
                        (w - sel::w( tile ))*w_stride;

                    // compute barycentrics
                    const float uf = (u+0.5f) * sel::u( bbox_delta );
                    const float vf = (v+0.5f) * sel::v( bbox_delta );

                    const float2 p = make_float2( uf - sel::u( v0 ), vf - sel::v( v0 ) );
                    const float bary0 = e0.x * p.x + e0.y * p.y;
                    const float bary1 = e1.x * p.x + e1.y * p.y;

                    // compute pixel coordinates
                    const int3 uvw = make_int3( u, v, w );
                    const int3 xyz = make_int3( sel::x( uvw ), sel::y( uvw ), sel::z( uvw ) );

                    // shade the fragment
                    const pixel_type value = shader.shade( tri_id, v0, v1, v2, n, bary0, bary1, xyz );

                    // write to the FB
                    tile_op_type::write( sm_tile, pixel, value );
                }
            }
        }
    }

#if VOXELPIPE_ENABLE_PROFILING
    // write utilization stats to gmem
    prf_stats.utilization_sum[1][ global_warp_id ]   = sm_utilization_sum[ warp_id ];
    prf_stats.utilization_count[1][ global_warp_id ] = sm_utilization_count[ warp_id ];
#endif
};

template <typename shader_type, int32 BLOCK_SIZE, int32 LOG_TILE_SIZE, int32 VoxelType, int32 BlendingMode>
__device__ void fine_raster_block<shader_type, BLOCK_SIZE,LOG_TILE_SIZE,VoxelType,THIN_RASTER,BlendingMode>::process(
    storage_type*   sm_tile,
    const int32     tile_id,
    const int32     N,
    const int32     log_N,
    const Tri_bbox* tri_bbox,
    const float3    bbox0,
    const float3    bbox1,
    const float3    bbox_delta,
    const float3    inv_bbox_delta,
    const int32     tile_begin,
    const int32     tile_end,
    const int32*    tile_tris,
    storage_type*   tile,
    FRStats         prf_stats,
    shader_type     shader)
{
    const int32 log_M = log_N - LOG_TILE_SIZE;
    const int32 M = N >> LOG_TILE_SIZE;
    const int32 T = 1 << LOG_TILE_SIZE;

    if (tile_end == tile_begin)
        return;

    // clear the tile's surface
    {
        const int32 PIXELS_PER_THREAD = (tile_op_type::STORAGE_SIZE + BLOCK_SIZE-1) / BLOCK_SIZE;
        for (int32 p = 0; p < PIXELS_PER_THREAD; ++p)
        {
            if (p * BLOCK_SIZE + threadIdx.x < tile_op_type::STORAGE_SIZE)
                sm_tile[ p * BLOCK_SIZE + threadIdx.x ] = tile_op_type::clearword();
        }
    }

    const int32 tile_z = T * (tile_id >> (log_M*2));
    const int32 tile_y = T * ((tile_id >> log_M) & (M-1));
    const int32 tile_x = T * (tile_id & (M-1));

#if VOXELPIPE_ENABLE_PROFILING
    // storage for utilization stats
             __shared__ int32 sm_utilization_sum[ 32 ];
    volatile __shared__ int32 sm_utilization_count[ 32 ];

    const int32 global_warp_id = (threadIdx.x + blockIdx.x*blockDim.x) >> 5;
    {
        const int32 warp_id  = (threadIdx.x >> 5);

        sm_utilization_sum[ warp_id ]   = prf_stats.utilization_sum[0][ global_warp_id ];
        sm_utilization_count[ warp_id ] = prf_stats.utilization_count[0][ global_warp_id ];
    }
#endif
    const int32 warp_id  = threadIdx.x >> 5;
    const int32 warp_tid = threadIdx.x & 31;

    // persistent threads: each warp will loop fetching batches of work
             __shared__ int32 sm_batch_counter[1];
    volatile __shared__ int32 sm_broadcast[32];

    sm_batch_counter[0] = 0;

    // make sure the zeroed batch counter and the cleared tile are visible to everybody
    __syncthreads();

#if VOXELPIPE_FR_FRAGMENT_DISTRIBUTION
    int32 tri_read   = 0;
    int32 tri_write  = 0;
    int32 frag_read  = 0;
    int32 frag_write = 0;

    const int32 MASK_BITS  = (T*LOG_TILE_SIZE*2);
    const int32 MASK_WORDS = (MASK_BITS+31) >> 5;

    volatile __shared__ int32  sm_tri_id[ BLOCK_SIZE*2 ];
    volatile __shared__ uint32 sm_tri_base[ BLOCK_SIZE*2 ];
    volatile __shared__ int32  sm_tri_frag[ BLOCK_SIZE*2 ];
    volatile __shared__ int32  sm_tri_frag_ex[ BLOCK_SIZE*2 ];
    volatile __shared__ uint32 sm_tri_mask[ MASK_WORDS*BLOCK_SIZE*2 ];

    volatile __shared__ int32 sm_temp[ BLOCK_SIZE*2 ];

    volatile int32*  sm_warp_tri_id      = sm_tri_id      + warp_id*64;
    volatile int32*  sm_warp_tri_frag    = sm_tri_frag    + warp_id*64;
    volatile int32*  sm_warp_tri_frag_ex = sm_tri_frag_ex + warp_id*64;
    volatile int32*  sm_warp_temp        = sm_temp        + warp_id*64;
    volatile uint32* sm_warp_tri_base    = sm_tri_base    + warp_id*64;
    volatile uint32* sm_warp_tri_mask    = sm_tri_mask    + warp_id*64;

    // loop fetching more work
    for (;;)
    {
        // need to queue more fragments?
        if (frag_write - frag_read < 32)
        {
            do
            {
                // fetch a new bacth of work for the warp
                if (warp_tid == 0)
                    sm_broadcast[ warp_id ] = atomicAdd( sm_batch_counter, 32 );

                // broadcast batch offset to entire warp
                const int32 batch_offset = sm_broadcast[ warp_id ] + tile_begin;

                // check whether the warp can be terminated
                if (batch_offset >= tile_end)
                    break;

                const int32 tri_idx = batch_offset + warp_tid;

                // check whether the given triangle is in range
                bool valid = (tri_idx < tile_end);

                const int32 tri_id = valid ? tile_tris[ tri_idx ] : uint32(-1);

                float4 v0, v1, v2;
                float3 edge0, edge1, edge2, n;

                // test whether the triangle plane really intersects the tile
                if (valid)
                {
                    // fetch the triangle
                    const int4 tri = tex1Dfetch( tex_triangles, tri_id );

                    v0 = tex1Dfetch( tex_vertices, tri.x );
                    v1 = tex1Dfetch( tex_vertices, tri.y );
                    v2 = tex1Dfetch( tex_vertices, tri.z );

                    // compute triangle큦 edges and normal
                    edge0 = make_float3( v1.x - v0.x, v1.y - v0.y, v1.z - v0.z );
                    edge1 = make_float3( v2.x - v1.x, v2.y - v1.y, v2.z - v1.z );
                    edge2 = make_float3( v0.x - v2.x, v0.y - v2.y, v0.z - v2.z );
                    n = anti_cross( edge0, edge2 );

                    // compute the bbox큦 critical point
                    const float3 c = make_float3(
                        n.x > 0 ? bbox_delta.x * T : 0.0f,
                        n.y > 0 ? bbox_delta.y * T : 0.0f,
                        n.z > 0 ? bbox_delta.z * T : 0.0f );

                    // precompute discriminants for the bbox/triangle plane test
                    const float r1 =
                        n.x * (c.x - v0.x) +
                        n.y * (c.y - v0.y) +
                        n.z * (c.z - v0.z);

                    const float r2 =
                        n.x * (bbox_delta.x * T - c.x - v0.x) +
                        n.y * (bbox_delta.y * T - c.y - v0.y) +
                        n.z * (bbox_delta.z * T - c.z - v0.z);

                    const float np =
                        n.x * (bbox0.x + tile_x * bbox_delta.x) +
                        n.y * (bbox0.y + tile_y * bbox_delta.y) +
                        n.z * (bbox0.z + tile_z * bbox_delta.z);

                    valid = ((np + r1) * (np + r2) <= 0.0f);
                }
                #if VOXELPIPE_ENABLE_PROFILING
                {
                    // keep utilization stats
                    const int32 active_mask = __ballot( valid );
                    sm_utilization_sum[ warp_id ]   += __popc( active_mask );
                    sm_utilization_count[ warp_id ] += 32;
                }
                #endif

                // count how many "fragments" (here: scanlines) we need to process
                int32 n_frag = 0;
                int32 frag_base;
                int3  tri_bbox0;
                int3  tri_bbox1;
                int32 tri_axis;
                if (valid)
                {
                    // fetch triangle bbox
                    fetch_tri_bbox( tri_id, tri_bbox0, tri_bbox1, tri_axis );

                    // clamp triangle bbox to the tile
                    tri_bbox0.x = max( tri_bbox0.x, tile_x );
                    tri_bbox0.y = max( tri_bbox0.y, tile_y );
                    tri_bbox0.z = max( tri_bbox0.z, tile_z );

                    tri_bbox1.x = min( tri_bbox1.x, tile_x + T-1 );
                    tri_bbox1.y = min( tri_bbox1.y, tile_y + T-1 );
                    tri_bbox1.z = min( tri_bbox1.z, tile_z + T-1 );

                    n_frag = (tri_axis == 2) ?
                        (tri_bbox1.y - tri_bbox0.y) + 1 :
                        (tri_bbox1.z - tri_bbox0.z) + 1;

                    frag_base = (tri_axis == 2) ? tri_bbox0.y : tri_bbox0.z;
                }

                // scan the number of fragments
                const int32 frag_scan  = scan_warp( n_frag, warp_tid, sm_warp_temp ) + frag_write;
                const int32 frag_count = scan_warp_total( sm_warp_temp );

                const uint32 active_mask = __ballot( valid );
                if (valid)
                {
                    const int32 idx = (tri_write + __popc(active_mask << (32 - warp_tid))) & 63;
                    sm_warp_tri_id[ idx ]      = tri_id;
                    sm_warp_tri_base[ idx ]    = (frag_base << 2) | tri_axis;
                    sm_warp_tri_frag[ idx ]    = frag_scan;
                    sm_warp_tri_frag_ex[ idx ] = frag_scan - n_frag;

                    // clear mask
                    for (int32 w = 0; w < MASK_WORDS; ++w)
                        sm_warp_tri_mask[ idx + 2*BLOCK_SIZE*w ] = 0;

                    if (tri_axis == 2)
                    {
                        generate_mask< Z, LOG_TILE_SIZE, 2*BLOCK_SIZE >(
                            sm_warp_tri_mask + idx,
                            tri_id,
                            make_int3( tile_x, tile_y, tile_z ),
                            bbox0,
                            bbox_delta,
                            inv_bbox_delta,
                            tri_bbox0,
                            tri_bbox1,
                            v0,
                            v1,
                            v2,
                            edge0,
                            edge1,
                            edge2,
                            n,
                            prf_stats );
                    }
                    else if (tri_axis == 1)
                    {
                        generate_mask< Y, LOG_TILE_SIZE, 2*BLOCK_SIZE >(
                            sm_warp_tri_mask + idx,
                            tri_id,
                            make_int3( tile_x, tile_y, tile_z ),
                            bbox0,
                            bbox_delta,
                            inv_bbox_delta,
                            tri_bbox0,
                            tri_bbox1,
                            v0,
                            v1,
                            v2,
                            edge0,
                            edge1,
                            edge2,
                            n,
                            prf_stats );
                    }
                    else
                    {
                        generate_mask< X, LOG_TILE_SIZE, 2*BLOCK_SIZE >(
                            sm_warp_tri_mask + idx,
                            tri_id,
                            make_int3( tile_x, tile_y, tile_z ),
                            bbox0,
                            bbox_delta,
                            inv_bbox_delta,
                            tri_bbox0,
                            tri_bbox1,
                            v0,
                            v1,
                            v2,
                            edge0,
                            edge1,
                            edge2,
                            n,
                            prf_stats );
                    }
                }

                tri_write  += __popc( active_mask );
                frag_write += frag_count;
            }
            while (frag_write - frag_read < 32);
        } // if (frag_write - frag_read < 32)

        // nothing else to do...
        if (frag_read == frag_write)
            break;

        // tag triangle boundaries
        sm_warp_temp[ warp_tid ] = 0;
        if (tri_read + warp_tid < tri_write)
        {
            const int32 idx = sm_warp_tri_frag[(tri_read + warp_tid) & 63] - frag_read;
            if (idx > 0 && idx <= 32)
                sm_warp_temp[idx - 1] = 1;
        }

        const uint32 boundary_mask = __ballot( sm_warp_temp[ warp_tid ] );

        // distribute fragments
        if (warp_tid < frag_write - frag_read)
        {
            //const int32 buffer_idx = (find_fragment( frag_read + warp_tid, sm_warp_tri_frag, tri_read, tri_write - tri_read ) + tri_read) & 63;
            const int32 buffer_idx = (tri_read + __popc( boundary_mask << (32 - warp_tid) )) & 63;
            const int32 tri_id     = sm_warp_tri_id[ buffer_idx ];
            const uint32 tri_base  = sm_warp_tri_base[ buffer_idx ];
            const int32 frag_idx   = frag_read + warp_tid - sm_warp_tri_frag_ex[ buffer_idx ];

            const int32 frag_base  = tri_base >> 2;
            const int32 tri_axis   = tri_base & 3;

            // fetch the triangle
            const int4 tri = tex1Dfetch( tex_triangles, tri_id );

            const float4 v0 = tex1Dfetch( tex_vertices, tri.x );
            const float4 v1 = tex1Dfetch( tex_vertices, tri.y );
            const float4 v2 = tex1Dfetch( tex_vertices, tri.z );

            // compute triangle큦 edges and normal
            const float3 edge0 = make_float3( v1.x - v0.x, v1.y - v0.y, v1.z - v0.z );
            const float3 edge1 = make_float3( v2.x - v1.x, v2.y - v1.y, v2.z - v1.z );
            const float3 edge2 = make_float3( v0.x - v2.x, v0.y - v2.y, v0.z - v2.z );
            const float3 n = anti_cross( edge0, edge2 );

            if (tri_axis == 2)
            {
                rasterize_scanline< Z, LOG_TILE_SIZE, 2*BLOCK_SIZE, storage_type, tile_op_type, shader_type >(
                    sm_warp_tri_mask + buffer_idx,
                    sm_tile,
                    tri_id,
                    frag_base,
                    frag_idx,
                    make_int3( tile_x, tile_y, tile_z ),
                    bbox0,
                    bbox_delta,
                    inv_bbox_delta,
                    v0,
                    v1,
                    v2,
                    edge0,
                    edge1,
                    edge2,
                    n,
                    prf_stats );
            }
            else if (tri_axis == 1)
            {
                rasterize_scanline< Y, LOG_TILE_SIZE, 2*BLOCK_SIZE, storage_type, tile_op_type, shader_type >(
                    sm_warp_tri_mask + buffer_idx,
                    sm_tile,
                    tri_id,
                    frag_base,
                    frag_idx,
                    make_int3( tile_x, tile_y, tile_z ),
                    bbox0,
                    bbox_delta,
                    inv_bbox_delta,
                    v0,
                    v1,
                    v2,
                    edge0,
                    edge1,
                    edge2,
                    n,
                    prf_stats );
            }
            else
            {
                rasterize_scanline< X, LOG_TILE_SIZE, 2*BLOCK_SIZE, storage_type, tile_op_type, shader_type >(
                    sm_warp_tri_mask + buffer_idx,
                    sm_tile,
                    tri_id,
                    frag_base,
                    frag_idx,
                    make_int3( tile_x, tile_y, tile_z ),
                    bbox0,
                    bbox_delta,
                    inv_bbox_delta,
                    v0,
                    v1,
                    v2,
                    edge0,
                    edge1,
                    edge2,
                    n,
                    prf_stats );
            }
        } // if (warp_tid < frag_write - frag_read)

        frag_read = ::min(frag_read + 32, frag_write);
        tri_read += __popc(boundary_mask);
    }
#else
    for (;;)
    {
        // fetch a new batch of work for the warp
        if (warp_tid == 0)
            sm_broadcast[ warp_id ] = atomicAdd( sm_batch_counter, 32 );

        // broadcast batch offset to entire warp
        const int32 batch_offset = sm_broadcast[ warp_id ] + tile_begin;

        // check whether the warp can be terminated
        if (batch_offset >= tile_end)
            break;

        const int32 tri_idx = batch_offset + warp_tid;

        // check whether the given triangle is in range
        if (tri_idx >= tile_end)
            continue;

        // fetch triangle id
        const int32 tri_id = tile_tris[ tri_idx ];

        // fetch triangle
        const int4 tri = tex1Dfetch( tex_triangles, tri_id );

        const float4 v0 = tex1Dfetch( tex_vertices, tri.x );
        const float4 v1 = tex1Dfetch( tex_vertices, tri.y );
        const float4 v2 = tex1Dfetch( tex_vertices, tri.z );

        // fetch triangle bbox
        int3 tri_bbox0;
        int3 tri_bbox1;
        fetch_tri_bbox( tri_id, tri_bbox0, tri_bbox1 );

        // clamp triangle bbox to the tile
        tri_bbox0.x = nih::max( tri_bbox0.x, tile_x );
        tri_bbox0.y = nih::max( tri_bbox0.y, tile_y );
        tri_bbox0.z = nih::max( tri_bbox0.z, tile_z );

        tri_bbox1.x = nih::min( tri_bbox1.x, tile_x + T-1 );
        tri_bbox1.y = nih::min( tri_bbox1.y, tile_y + T-1 );
        tri_bbox1.z = nih::min( tri_bbox1.z, tile_z + T-1 );

        // compute triangle큦 edges and normal
        const float3 edge0 = make_float3( v1.x - v0.x, v1.y - v0.y, v1.z - v0.z );
        const float3 edge1 = make_float3( v2.x - v1.x, v2.y - v1.y, v2.z - v1.z );
        const float3 edge2 = make_float3( v0.x - v2.x, v0.y - v2.y, v0.z - v2.z );
        const float3 n = anti_cross( edge0, edge2 );

        #if VOXELPIPE_ENABLE_PROFILING
        {
            // keep utilization stats
            sm_utilization_count[ warp_id ] += 32;
        }
        #endif
        // test whether the triangle plane really intersects the tile
        {
            // compute the bbox큦 critical point
            const float3 c = make_float3(
                n.x > 0 ? bbox_delta.x * T : 0.0f,
                n.y > 0 ? bbox_delta.y * T : 0.0f,
                n.z > 0 ? bbox_delta.z * T : 0.0f );

            // precompute discriminants for the bbox/triangle plane test
            const float r1 =
                n.x * (c.x - v0.x) +
                n.y * (c.y - v0.y) +
                n.z * (c.z - v0.z);

            const float r2 =
                n.x * (bbox_delta.x * T - c.x - v0.x) +
                n.y * (bbox_delta.y * T - c.y - v0.y) +
                n.z * (bbox_delta.z * T - c.z - v0.z);

            const float np =
                n.x * (bbox0.x + tile_x * bbox_delta.x) +
                n.y * (bbox0.y + tile_y * bbox_delta.y) +
                n.z * (bbox0.z + tile_z * bbox_delta.z);

            if ((np + r1) * (np + r2) > 0.0f)
                continue;
        }
        #if VOXELPIPE_ENABLE_PROFILING
        {
            // keep utilization stats
            const int32 active_mask = __ballot( true );
            sm_utilization_sum[ warp_id ] += __popc( active_mask );
        }
        #endif

        const bool byx = fabsf( n.y ) > fabsf( n.x );
        const bool byz = fabsf( n.y ) > fabsf( n.z );
        const bool bzx = fabsf( n.z ) > fabsf( n.x );
        const int32 tri_axis = byx ? (byz ? 1 : 2) : (bzx ? 2 : 0);

        // compute Z-axis projection tests (XY plane)
        if (tri_axis == 2)
        {
            rasterize< Z, T, storage_type, tile_op_type, shader_type >(
                sm_tile,
                tri_id,
                make_int3( tile_x, tile_y, tile_z ),
                bbox0,
                bbox_delta,
                inv_bbox_delta,
                tri_bbox0,
                tri_bbox1,
                v0,
                v1,
                v2,
                edge0,
                edge1,
                edge2,
                n,
                prf_stats,
                shader );
        }
        else if (tri_axis == 1) // compute Y-axis projection tests (XZ plane)
        {
            rasterize< Y, T, storage_type, tile_op_type, shader_type >(
                sm_tile,
                tri_id,
                make_int3( tile_x, tile_y, tile_z ),
                bbox0,
                bbox_delta,
                inv_bbox_delta,
                tri_bbox0,
                tri_bbox1,
                v0,
                v1,
                v2,
                edge0,
                edge1,
                edge2,
                n,
                prf_stats,
                shader );
        }
        else // compute X-axis projection tests (YZ plane)
        {
            rasterize< X, T, storage_type, tile_op_type, shader_type >(
                sm_tile,
                tri_id,
                make_int3( tile_x, tile_y, tile_z ),
                bbox0,
                bbox_delta,
                inv_bbox_delta,
                tri_bbox0,
                tri_bbox1,
                v0,
                v1,
                v2,
                edge0,
                edge1,
                edge2,
                n,
                prf_stats,
                shader );
        }
    }
#endif

    __syncthreads();

    // write out tile
    {
        const int32 WORDS_PER_THREAD = (tile_op_type::STORAGE_SIZE + BLOCK_SIZE-1) / BLOCK_SIZE;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < WORDS_PER_THREAD; ++p)
        {
            if (p * BLOCK_SIZE + threadIdx.x < tile_op_type::STORAGE_SIZE)
                tile[ p * BLOCK_SIZE + threadIdx.x ] = sm_tile[ p * BLOCK_SIZE + threadIdx.x ];
        }
    }

#if VOXELPIPE_ENABLE_PROFILING
    {
        const int32 warp_id  = (threadIdx.x >> 5);

        // write utilization stats to gmem
        prf_stats.utilization_sum[0][ global_warp_id ]   = sm_utilization_sum[ warp_id ];
        prf_stats.utilization_count[0][ global_warp_id ] = sm_utilization_count[ warp_id ];
    }
#endif
}

template <typename shader_type, int32 BLOCK_SIZE, int32 LOG_TILE_SIZE, int32 VoxelType, int32 BlendingMode>
__device__ void fine_raster_block<shader_type,BLOCK_SIZE,LOG_TILE_SIZE,VoxelType,CONSERVATIVE_RASTER,BlendingMode>::process(
    storage_type*   sm_tile,
    const int32     tile_id,
    const int32     N,
    const int32     log_N,
    const Tri_bbox* tri_bbox,
    const float3    bbox0,
    const float3    bbox1,
    const float3    bbox_delta,
    const float3    inv_bbox_delta,
    const int32     tile_begin,
    const int32     tile_end,
    const int32*    tile_tris,
    storage_type*   tile,
    FRStats         prf_stats,
    shader_type     shader)
{
    const int32 log_M = log_N - LOG_TILE_SIZE;
    const int32 M = N >> LOG_TILE_SIZE;
    const int32 T = 1 << LOG_TILE_SIZE;

    if (tile_end == tile_begin)
        return;

    // clear the tile's surface
    {
        const int32 PIXELS_PER_THREAD = (tile_op_type::STORAGE_SIZE + BLOCK_SIZE-1) / BLOCK_SIZE;
        for (int32 p = 0; p < PIXELS_PER_THREAD; ++p)
        {
            if (p * BLOCK_SIZE + threadIdx.x < tile_op_type::STORAGE_SIZE)
                sm_tile[ p * BLOCK_SIZE + threadIdx.x ] = tile_op_type::clearword();
        }
    }

    const int32 tile_z = T * (tile_id >> (log_M*2));
    const int32 tile_y = T * ((tile_id >> log_M) & (M-1));
    const int32 tile_x = T * (tile_id & (M-1));

#if VOXELPIPE_ENABLE_PROFILING
    // storage for utilization stats
             __shared__ int32 sm_utilization_sum[ 32 ];
    volatile __shared__ int32 sm_utilization_count[ 32 ];

    const int32 global_warp_id = (threadIdx.x + blockIdx.x*blockDim.x) >> 5;
    {
        const int32 warp_id  = (threadIdx.x >> 5);

        sm_utilization_sum[ warp_id ]   = prf_stats.utilization_sum[0][ global_warp_id ];
        sm_utilization_count[ warp_id ] = prf_stats.utilization_count[0][ global_warp_id ];
    }
#endif
    const int32 warp_id  = threadIdx.x >> 5;
    const int32 warp_tid = threadIdx.x & 31;

    // persistent threads: each warp will loop fetching batches of work
             __shared__ int32 sm_batch_counter[1];
    volatile __shared__ int32 sm_broadcast[32];

    sm_batch_counter[0] = 0;

    // make sure the zeroed batch counter and the cleared tile are visible to everybody
    __syncthreads();

    for (;;)
    {
        // fetch a new batch of work for the warp
        if (warp_tid == 0)
            sm_broadcast[ warp_id ] = atomicAdd( sm_batch_counter, 32 );

        // broadcast batch offset to entire warp
        const int32 batch_offset = sm_broadcast[ warp_id ] + tile_begin;

        // check whether the warp can be terminated
        if (batch_offset >= tile_end)
            break;

        const int32 tri_idx = batch_offset + warp_tid;

        // check whether the given triangle is in range
        if (tri_idx >= tile_end)
            continue;

        // fetch triangle id
        const int32 tri_id = tile_tris[ tri_idx ];

        // fetch triangle
        const int4 tri = tex1Dfetch( tex_triangles, tri_id );

        const float4 v0 = tex1Dfetch( tex_vertices, tri.x );
        const float4 v1 = tex1Dfetch( tex_vertices, tri.y );
        const float4 v2 = tex1Dfetch( tex_vertices, tri.z );

        // fetch triangle bbox
        int3 tri_bbox0;
        int3 tri_bbox1;
        fetch_tri_bbox( tri_id, tri_bbox0, tri_bbox1 );

        // clamp triangle bbox to the tile
        tri_bbox0.x = max( tri_bbox0.x, tile_x );
        tri_bbox0.y = max( tri_bbox0.y, tile_y );
        tri_bbox0.z = max( tri_bbox0.z, tile_z );

        tri_bbox1.x = min( tri_bbox1.x, tile_x + T-1 );
        tri_bbox1.y = min( tri_bbox1.y, tile_y + T-1 );
        tri_bbox1.z = min( tri_bbox1.z, tile_z + T-1 );

        // compute triangle큦 edges and normal
        const float3 edge0 = make_float3( v1.x - v0.x, v1.y - v0.y, v1.z - v0.z );
        const float3 edge1 = make_float3( v2.x - v1.x, v2.y - v1.y, v2.z - v1.z );
        const float3 edge2 = make_float3( v0.x - v2.x, v0.y - v2.y, v0.z - v2.z );
        const float3 n = anti_cross( edge0, edge2 );

        #if VOXELPIPE_ENABLE_PROFILING
        {
            // keep utilization stats
            sm_utilization_count[ warp_id ] += 32;
        }
        #endif

        // test whether the triangle plane really intersects the tile
        {
            // compute the bbox큦 critical point
            const float3 c = make_float3(
                n.x > 0 ? bbox_delta.x * T : 0.0f,
                n.y > 0 ? bbox_delta.y * T : 0.0f,
                n.z > 0 ? bbox_delta.z * T : 0.0f );

            // precompute discriminants for the bbox/triangle plane test
            const float r1 =
                n.x * (c.x - v0.x) +
                n.y * (c.y - v0.y) +
                n.z * (c.z - v0.z);

            const float r2 =
                n.x * (bbox_delta.x * T - c.x - v0.x) +
                n.y * (bbox_delta.y * T - c.y - v0.y) +
                n.z * (bbox_delta.z * T - c.z - v0.z);

            const float np =
                n.x * (bbox0.x + tile_x * bbox_delta.x) +
                n.y * (bbox0.y + tile_y * bbox_delta.y) +
                n.z * (bbox0.z + tile_z * bbox_delta.z);

            if ((np + r1) * (np + r2) > 0.0f)
                continue;
        }
        #if VOXELPIPE_ENABLE_PROFILING
        {
            // keep utilization stats
            const int32 active_mask = __ballot( true );
            sm_utilization_sum[ warp_id ] += __popc( active_mask );
        }
        #endif

        const bool byx = fabsf( n.y ) > fabsf( n.x );
        const bool byz = fabsf( n.y ) > fabsf( n.z );
        const bool bzx = fabsf( n.z ) > fabsf( n.x );
        const int32 tri_axis = byx ? (byz ? 1 : 2) : (bzx ? 2 : 0);

        // compute Z-axis projection tests (XY plane)
        if (tri_axis == 2)
        {
            rasterize_26sep< Z, T, storage_type, tile_op_type, shader_type >(
                sm_tile,
                tri_id,
                make_int3( tile_x, tile_y, tile_z ),
                bbox0,
                bbox_delta,
                inv_bbox_delta,
                tri_bbox0,
                tri_bbox1,
                v0,
                v1,
                v2,
                edge0,
                edge1,
                edge2,
                n,
                prf_stats,
                shader );
        }
        else if (tri_axis == 1) // compute Y-axis projection tests (XZ plane)
        {
            rasterize_26sep< Y, T, storage_type, tile_op_type, shader_type >(
                sm_tile,
                tri_id,
                make_int3( tile_x, tile_y, tile_z ),
                bbox0,
                bbox_delta,
                inv_bbox_delta,
                tri_bbox0,
                tri_bbox1,
                v0,
                v1,
                v2,
                edge0,
                edge1,
                edge2,
                n,
                prf_stats,
                shader );
        }
        else // compute X-axis projection tests (YZ plane)
        {
            rasterize_26sep< X, T, storage_type, tile_op_type, shader_type >(
                sm_tile,
                tri_id,
                make_int3( tile_x, tile_y, tile_z ),
                bbox0,
                bbox_delta,
                inv_bbox_delta,
                tri_bbox0,
                tri_bbox1,
                v0,
                v1,
                v2,
                edge0,
                edge1,
                edge2,
                n,
                prf_stats,
                shader );
        }
    }

    __syncthreads();

    // write out tile
    {
        const int32 WORDS_PER_THREAD = (tile_op_type::STORAGE_SIZE + BLOCK_SIZE-1) / BLOCK_SIZE;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < WORDS_PER_THREAD; ++p)
        {
            if (p * BLOCK_SIZE + threadIdx.x < tile_op_type::STORAGE_SIZE)
                tile[ p * BLOCK_SIZE + threadIdx.x ] = sm_tile[ p * BLOCK_SIZE + threadIdx.x ];
        }
    }

#if VOXELPIPE_ENABLE_PROFILING
    {
        const int32 warp_id  = (threadIdx.x >> 5);

        // write utilization stats to gmem
        prf_stats.utilization_sum[0][ global_warp_id ]   = sm_utilization_sum[ warp_id ];
        prf_stats.utilization_count[0][ global_warp_id ] = sm_utilization_count[ warp_id ];
    }
#endif
}

template <typename shader_type, int32 BLOCK_SIZE, int32 LOG_TILE_SIZE, int32 VoxelType, int32 VoxelizationType, int32 BlendingMode>
__global__ void fine_raster_kernel(
    const int32     grid_size,
    const int32     n_grids,
    const int32     N,
    const int32     N_tiles,
    const int32     log_N,
    const Tri_bbox* tri_bbox,
    const float3    bbox0,
    const float3    bbox1,
    const float3    bbox_delta,
    const float3    inv_bbox_delta,
    const int32*    tile_starts,
    const int32*    tile_ends,
    const int32*    tile_ids,
    const int32*    tile_tris,
    void*           tile_buffer,
    uint32*         batch_counter,
    FRStats         prf_stats,
    shader_type     shader)
{
    typedef TileOp<VoxelType,BlendingMode,LOG_TILE_SIZE> tile_op_type;
    typedef typename tile_op_type::storage_type          storage_type;

    __shared__ storage_type sm_tile[ tile_op_type::STORAGE_SIZE ];

  #if VOXELPIPE_FR_PERSISTENT_THREADS
    volatile __shared__ int32 sm_broadcast[1];

    for (;;)
    {
        __syncthreads(); // block before switching tile

        // fetch anoter tile to work on
        if (threadIdx.x == 0)
            *sm_broadcast = atomicAdd( batch_counter, 1 );

        __syncthreads(); // make sure sm_broadcast is visible to everybody

        // broadcast tile index to the entire block
        const int32 tile_idx = *sm_broadcast;
        if (tile_idx >= N_tiles)
            break;
  #else
    for (int32 grid = 0; grid < n_grids; ++grid)
    {
        const int32 tile_idx = blockIdx.x + grid * grid_size;
  #endif
        if (tile_idx < N_tiles)
        {
            storage_type* tile = (storage_type*)(tile_buffer) + tile_op_type::STORAGE_SIZE * tile_idx;

            fine_raster_block<shader_type, BLOCK_SIZE, LOG_TILE_SIZE, VoxelType, VoxelizationType, BlendingMode>::process(
                sm_tile,
                tile_ids[ tile_idx ],
                N,
                log_N,
                tri_bbox,
                bbox0,
                bbox1,
                bbox_delta,
                inv_bbox_delta,
                tile_starts[ tile_idx ],
                tile_ends[ tile_idx ],
                tile_tris,
                tile,
                prf_stats,
                shader );
        }
        __syncthreads();
    }
}

template <int32 LOG_TILE_SIZE> struct Block_config    { static const int32 BLOCK_SIZE = 256; };
template <>                    struct Block_config<2> { static const int32 BLOCK_SIZE = 64;  };
template <>                    struct Block_config<3> { static const int32 BLOCK_SIZE = 64;  };

} // namespace FR

template <typename shader_type, int32 LOG_TILE_SIZE, int32 VoxelType, int32 VoxelizationType, int32 BlendingMode>
void fine_raster(
    const int32     N,
    const int32     log_N,
    const int32     N_tiles,
    const int32     N_tris,
    const int32     N_vertices,
    const int4*     triangles,
    const float4*   vertices,
    const Tri_bbox* tri_bbox,
    const float3    bbox0,
    const float3    bbox1,
    const float3    bbox_delta,
    const float3    inv_bbox_delta,
    const int32*    tile_starts,
    const int32*    tile_ends,
    const int32*    tile_ids,
    const int32*    tile_tris,
    void*           tile_buffer,
    uint32*         batch_counter,
    FineRasterStats& stats,
    shader_type     shader = shader_type())
{
    cudaChannelFormatDesc channel_desc_tri_bbox = cudaCreateChannelDesc<int4>();
    tex_tri_bbox.normalized = false;
    tex_tri_bbox.filterMode = cudaFilterModePoint;

    cudaChannelFormatDesc channel_desc_tri = cudaCreateChannelDesc<int4>();
    tex_triangles.normalized = false;
    tex_triangles.filterMode = cudaFilterModePoint;

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
    tex_vertices.normalized = false;
    tex_vertices.filterMode = cudaFilterModePoint;

    cudaBindTexture( 0, &tex_tri_bbox,  tri_bbox,  &channel_desc_tri_bbox,  sizeof(int4)*N_tris*Tri_bbox::N_INT4 );
    cudaBindTexture( 0, &tex_triangles, triangles, &channel_desc_tri,       sizeof(int4)*N_tris );
    cudaBindTexture( 0, &tex_vertices,  vertices,  &channel_desc,           sizeof(float4)*N_vertices );

    uint32 zero = 0;

    cudaMemcpy(
        batch_counter,
        &zero,
        sizeof(uint32),
        cudaMemcpyHostToDevice );

    cudaThreadSynchronize();

    const int32 BLOCK_SIZE = FR::Block_config<LOG_TILE_SIZE>::BLOCK_SIZE;

    cudaFuncSetCacheConfig( FR::fine_raster_kernel<shader_type,BLOCK_SIZE,LOG_TILE_SIZE,VoxelType,VoxelizationType,BlendingMode>, cudaFuncCachePreferShared );

    dim3 dim_block = dim3( BLOCK_SIZE, 1, 1 );
    dim3 dim_grid  = dim3( thrust::detail::device::cuda::arch::max_active_blocks(FR::fine_raster_kernel<shader_type,BLOCK_SIZE,LOG_TILE_SIZE,VoxelType,VoxelizationType,BlendingMode>, BLOCK_SIZE, 0), 1, 1 );

    const int32 grid_size = dim_grid.x;
    const int32 n_grids   = (N_tiles + grid_size-1) / grid_size;

    FR::FRStats prf_stats;
#if VOXELPIPE_ENABLE_PROFILING
    const int32 n_warps = dim_grid.x * BLOCK_SIZE / 32;
    thrust::device_vector<int32>  prf_samples_sum( n_warps, 0 );
    thrust::device_vector<int32>  prf_samples_sum2( n_warps, 0 );
    thrust::device_vector<int32>  prf_samples_count( n_warps, 0 );
    thrust::device_vector<int32>  prf_utilization_sum0( n_warps, 0 );
    thrust::device_vector<int32>  prf_utilization_count0( n_warps, 0 );
    thrust::device_vector<int32>  prf_utilization_sum1( n_warps, 0 );
    thrust::device_vector<int32>  prf_utilization_count1( n_warps, 0 );
    thrust::device_vector<uint32> prf_fragment_count( 1 );
    prf_stats.samples_sum          = thrust::raw_pointer_cast( &prf_samples_sum.front() );
    prf_stats.samples_sum2         = thrust::raw_pointer_cast( &prf_samples_sum2.front() );
    prf_stats.samples_count        = thrust::raw_pointer_cast( &prf_samples_count.front() );
    prf_stats.utilization_sum[0]   = thrust::raw_pointer_cast( &prf_utilization_sum0.front() );
    prf_stats.utilization_count[0] = thrust::raw_pointer_cast( &prf_utilization_count0.front() );
    prf_stats.utilization_sum[1]   = thrust::raw_pointer_cast( &prf_utilization_sum1.front() );
    prf_stats.utilization_count[1] = thrust::raw_pointer_cast( &prf_utilization_count1.front() );
    prf_stats.fragment_count       = thrust::raw_pointer_cast( &prf_fragment_count.front() );
#endif

    FR::fine_raster_kernel<shader_type, BLOCK_SIZE, LOG_TILE_SIZE, VoxelType, VoxelizationType, BlendingMode> <<<dim_grid,dim_block>>>(
        grid_size,
        n_grids,
        N,
        N_tiles,
        log_N,
        tri_bbox,
        bbox0,
        bbox1,
        bbox_delta,
        inv_bbox_delta,
        tile_starts,
        tile_ends,
        tile_ids,
        tile_tris,
        tile_buffer,
        batch_counter,
        prf_stats,
        shader );

    cudaThreadSynchronize();

    cudaUnbindTexture( &tex_vertices );
    cudaUnbindTexture( &tex_triangles );
    cudaUnbindTexture( &tex_tri_bbox );

#if VOXELPIPE_ENABLE_PROFILING
    {
        thrust::host_vector<int32> h_prf_sum( prf_samples_sum );
        thrust::host_vector<int32> h_prf_sum2( prf_samples_sum2 );
        thrust::host_vector<int32> h_prf_count( prf_samples_count );
        float avg = 0.0f;
        float var = 0.0f;
        for (int32 warp_id = 0; warp_id < n_warps; ++warp_id)
        {
            const float sum  = float( h_prf_sum[ warp_id ] );
            const float sum2 = float( h_prf_sum2[ warp_id ] );
            const float cnt  = float( h_prf_count[ warp_id ] );
            if (cnt)
            {
                avg += sum / cnt;
                var += (cnt * sum2 - sum*sum) / (cnt * (cnt-1));
            }
        }
        stats.samples_avg = avg / float(n_warps);
        stats.samples_std = sqrtf( var / float(n_warps) );
    }
    {
        thrust::host_vector<int32> h_prf_sum( prf_utilization_sum0 );
        thrust::host_vector<int32> h_prf_count( prf_utilization_count0 );
        float avg = 0.0f;
        int32 n_active_warps = 0;
        for (int32 warp_id = 0; warp_id < n_warps; ++warp_id)
        {
            const float sum  = float( h_prf_sum[ warp_id ] );
            const float cnt  = float( h_prf_count[ warp_id ] );
            if (cnt)
            {
                avg += sum / cnt;
                n_active_warps++;
            }
        }
        stats.utilization[0] = avg / float(n_active_warps);
    }
    {
        thrust::host_vector<int32> h_prf_sum( prf_utilization_sum1 );
        thrust::host_vector<int32> h_prf_count( prf_utilization_count1 );
        float avg = 0.0f;
        int32 n_active_warps = 0;
        for (int32 warp_id = 0; warp_id < n_warps; ++warp_id)
        {
            const float sum  = float( h_prf_sum[ warp_id ] );
            const float cnt  = float( h_prf_count[ warp_id ] );
            if (cnt)
            {
                avg += sum / cnt;
                n_active_warps++;
            }
        }
        stats.utilization[1] = avg / float(n_active_warps);
    }
    {
        thrust::host_vector<uint32> h_fragment_count( prf_fragment_count );
        stats.fragment_count = h_fragment_count[0];
    }
#endif
}

} // namespace voxelpipe
