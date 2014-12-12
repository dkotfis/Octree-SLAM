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

/*! \file abuffer.h
 *  \brief A-buffer kernels
 */

#pragma once

#include <voxelpipe/common.h>
#include <voxelpipe/utils.h>

namespace voxelpipe {

namespace AB {

struct ABStats
{
    ABStats() {}

    int32*  samples_sum;
    int32*  samples_sum2;
    int32*  samples_count;
    int32*  utilization_sum[2];
    int32*  utilization_count[2];
};


template <int32 AXIS>
__device__ __inline__ void emit_fragments(
    volatile int32*      sm_red,
    volatile uint32*     sm_broadcast,
    const bool           valid,
    const int32   N,
    const int32   tri_id,
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
    uint32*       fragment_counter,
    int32*        fragment_tris,
    int32*        fragment_ids,
    ABStats       prf_stats)
{
    typedef uvw<AXIS> sel;

    const int32 warp_id  = threadIdx.x >> 5;
    const int32 warp_tid = threadIdx.x & 31;

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

    const int32 u_samples = sel::u( tri_bbox1 ) - sel::u( tri_bbox0 ) + 1;
    const int32 v_samples = sel::v( tri_bbox1 ) - sel::v( tri_bbox0 ) + 1;
    const int32 uv_samples = valid ? u_samples * v_samples : 0;

#if VOXELPIPE_ENABLE_PROFILING
    // storage for utilization stats
             __shared__ int32 sm_utilization_sum[ 32 ];
    volatile __shared__ int32 sm_utilization_count[ 32 ];

    const int32 global_warp_id = (threadIdx.x + blockIdx.x*blockDim.x) >> 5;

    sm_utilization_sum[ warp_id ]   = prf_stats.utilization_sum[1][ global_warp_id ];
    sm_utilization_count[ warp_id ] = prf_stats.utilization_count[1][ global_warp_id ];

    // keep stats on the number of sample tests
    const int32 n_samples = (sel::u( tri_bbox1 ) - sel::u( tri_bbox0 )) * (sel::v( tri_bbox1 ) - sel::v( tri_bbox0 ));
    atomicAdd( &prf_stats.samples_sum[ global_warp_id ], n_samples );
    atomicAdd( &prf_stats.samples_sum2[ global_warp_id ], n_samples*n_samples );
    if (warp_tid == 0)
        prf_stats.samples_count[ global_warp_id ] += 32;
#endif

    int32 u = sel::u( tri_bbox0 );
    int32 v = sel::v( tri_bbox0 );

    float b0 = a.x + v * n_delta_v.x;
    float b1 = a.y + v * n_delta_v.y;
    float b2 = a.z + v * n_delta_v.z;

    float vf = (v+0.5f) * sel::v( bbox_delta );

    for (int32 s = 0; __any( s < uv_samples ); ++s, ++u)
    {
        #if VOXELPIPE_ENABLE_PROFILING
        {
            // keep utilization stats
            const int32 active_mask = __ballot( s < uv_samples );
            sm_utilization_sum[ warp_id ]   += __popc( active_mask );
            sm_utilization_count[ warp_id ] += 32;
        }
        #endif

        // update to the next scanline
        if (u > sel::u( tri_bbox1 ))
        {
            u = sel::u( tri_bbox0 );
            v++;

            b0 = a.x + v * n_delta_v.x;
            b1 = a.y + v * n_delta_v.y;
            b2 = a.z + v * n_delta_v.z;

            vf = (v+0.5f) * sel::v( bbox_delta );
        }

        const float c0 = b0 + u * n_delta_u.x;
        const float c1 = b1 + u * n_delta_u.y;
        const float c2 = b2 + u * n_delta_u.z;

        // compute W depth
        const float uf = (u+0.5f) * sel::u( bbox_delta );
        const float wf = plane_eq.z - (plane_eq.x*uf + plane_eq.y*vf);
        const int32 w = int32( wf * sel::w( inv_bbox_delta ) );

        const bool inside =
            (s < uv_samples) &
            (c0 >= 0.0f)     &
            (c1 >= 0.0f)     &
            (c2 >= 0.0f)     &
            (w >= 0)         &
            (w <  N);

        if (__any( inside ) == false)
            continue;

        const uint32 inside_mask  = __ballot( inside );
        const uint32 inside_scan  = __popc( inside_mask << (31 - warp_tid) );

        if (warp_tid == 31)
            sm_broadcast[ warp_id ] = atomicAdd( fragment_counter, inside_scan );

        const uint32 output_offset = sm_broadcast[ warp_id ] + inside_scan - 1;

        if (inside)
        {
            const int32 u_stride = sel::u_stride( N );
            const int32 v_stride = sel::v_stride( N );
            const int32 w_stride = sel::w_stride( N );

            const int32 pixel =
                u * u_stride +
                v * v_stride +
                w * w_stride;

            fragment_tris[ output_offset ] = tri_id;
            fragment_ids[ output_offset ]  = pixel;
        }
    }

#if VOXELPIPE_ENABLE_PROFILING
    // write utilization stats to gmem
    prf_stats.utilization_sum[1][ global_warp_id ]   = sm_utilization_sum[ warp_id ];
    prf_stats.utilization_count[1][ global_warp_id ] = sm_utilization_count[ warp_id ];
#endif
}

template <int32 AXIS>
__device__ __inline__ void emit_fragments_u(
    volatile int32*      sm_red,
    volatile uint32*     sm_broadcast,
    const int32   N,
    const int32   tri_id,
    const int32   scanline,
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
    uint32*       fragment_counter,
    int32*        fragment_tris,
    int32*        fragment_ids,
    ABStats       prf_stats)
{
    typedef uvw<AXIS> sel;

    const int32 warp_id  = threadIdx.x >> 5;
    const int32 warp_tid = threadIdx.x & 31;

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

    const int32 u_samples = sel::u( tri_bbox1 ) - sel::u( tri_bbox0 ) + 1;
    const int32 v_samples = sel::v( tri_bbox1 ) - sel::v( tri_bbox0 ) + 1;

    const int32 v = sel::v( tri_bbox0 ) + scanline;

    const float b0 = a.x + v * n_delta_v.x;
    const float b1 = a.y + v * n_delta_v.y;
    const float b2 = a.z + v * n_delta_v.z;

    const float vf = (v+0.5f) * sel::v( bbox_delta );
    const float wf_v = plane_eq.z - plane_eq.y*vf;

    for (int32 u = sel::u( tri_bbox0 ); u <= sel::u( tri_bbox1 ); ++u)
    {
        const float c0 = b0 + u * n_delta_u.x;
        const float c1 = b1 + u * n_delta_u.y;
        const float c2 = b2 + u * n_delta_u.z;

        // compute W depth
        const float uf = (u+0.5f) * sel::u( bbox_delta );
        const float wf = wf_v - plane_eq.x*uf;
        const int32 w = int32( wf * sel::w( inv_bbox_delta ) );

        const bool inside =
            (v  <= sel::v( tri_bbox1 )) &
            (c0 >= 0.0f)     &
            (c1 >= 0.0f)     &
            (c2 >= 0.0f)     &
            (w >= 0)         &
            (w <  N);

        if (__any( inside ) == false)
            continue;

        const uint32 inside_mask  = __ballot( inside );
        const uint32 inside_scan  = __popc( inside_mask << (31 - warp_tid) );

        if (warp_tid == 31)
            sm_broadcast[ warp_id ] = atomicAdd( fragment_counter, inside_scan );

        const uint32 output_offset = sm_broadcast[ warp_id ] + inside_scan - 1;

        if (inside)
        {
            const int32 u_stride = sel::u_stride( N );
            const int32 v_stride = sel::v_stride( N );
            const int32 w_stride = sel::w_stride( N );

            const int32 pixel =
                u * u_stride +
                v * v_stride +
                w * w_stride;

            fragment_tris[ output_offset ] = tri_id;
            fragment_ids[ output_offset ]  = pixel;
        }
    }
}
template <int32 AXIS>
__device__ __inline__ void emit_fragments_v(
    volatile int32*      sm_red,
    volatile uint32*     sm_broadcast,
    const int32   N,
    const int32   tri_id,
    const int32   scanline,
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
    uint32*       fragment_counter,
    int32*        fragment_tris,
    int32*        fragment_ids,
    ABStats       prf_stats)
{
    typedef uvw<AXIS> sel;

    const int32 warp_id  = threadIdx.x >> 5;
    const int32 warp_tid = threadIdx.x & 31;

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

    const int32 u_samples = sel::u( tri_bbox1 ) - sel::u( tri_bbox0 ) + 1;
    const int32 v_samples = sel::v( tri_bbox1 ) - sel::v( tri_bbox0 ) + 1;

    const int32 u = sel::u( tri_bbox0 ) + scanline;

    const float b0 = a.x + u * n_delta_u.x;
    const float b1 = a.y + u * n_delta_u.y;
    const float b2 = a.z + u * n_delta_u.z;

    const float uf = (u+0.5f) * sel::u( bbox_delta );
    const float wf_u = plane_eq.z - plane_eq.x*uf;

    for (int32 v = sel::v( tri_bbox0 ); v <= sel::v( tri_bbox1 ); ++v)
    {
        const float c0 = b0 + v * n_delta_v.x;
        const float c1 = b1 + v * n_delta_v.y;
        const float c2 = b2 + v * n_delta_v.z;

        // compute W depth
        const float vf = (v+0.5f) * sel::v( bbox_delta );
        const float wf = wf_u - plane_eq.y*vf;
        const int32 w = int32( wf * sel::w( inv_bbox_delta ) );

        const bool inside =
            (u  <= sel::u( tri_bbox1 )) &
            (c0 >= 0.0f)     &
            (c1 >= 0.0f)     &
            (c2 >= 0.0f)     &
            (w >= 0)         &
            (w <  N);

        if (__any( inside ) == false)
            continue;

        const uint32 inside_mask  = __ballot( inside );
        const uint32 inside_scan  = __popc( inside_mask << (31 - warp_tid) );

        if (warp_tid == 31)
            sm_broadcast[ warp_id ] = atomicAdd( fragment_counter, inside_scan );

        const uint32 output_offset = sm_broadcast[ warp_id ] + inside_scan - 1;

        if (inside)
        {
            const int32 u_stride = sel::u_stride( N );
            const int32 v_stride = sel::v_stride( N );
            const int32 w_stride = sel::w_stride( N );

            const int32 pixel =
                u * u_stride +
                v * v_stride +
                w * w_stride;

            fragment_tris[ output_offset ] = tri_id;
            fragment_ids[ output_offset ]  = pixel;
        }
    }
}


template <typename shader_type, int32 BLOCK_SIZE, int32 VoxelType, int32 VoxelizationType>
__global__ void A_buffer_unsorted_small_kernel(
    const int32     grid_size,
    const int32     n_grids,
    const int32     N,
    const int32     log_N,
    const int32     N_tris,
    const int32*    tri_index,
    const float3    bbox0,
    const float3    bbox1,
    const float3    bbox_delta,
    const float3    inv_bbox_delta,
    uint32*         batch_counter,
    uint32*         fragment_counter,
    int32*          fragment_tris,
    int32*          fragment_ids,
    ABStats         prf_stats)
{
    const int32 BATCH_SIZE = 32; // must be a multiple of 32

    const int32 warp_id  = threadIdx.x >> 5;
    const int32 warp_tid = threadIdx.x & 31;

    volatile __shared__ int32  sm_red[ BLOCK_SIZE * 2 ];
    volatile __shared__ uint32 sm_broadcast[ 32 ];

    // each warp keeps getting some work
    for (;;)
    {
        // fetch a batch of work
        if (warp_tid == 0)
            sm_broadcast[ warp_id ] = atomicAdd( batch_counter, BATCH_SIZE );

        const int32 batch_begin = sm_broadcast[ warp_id ];

        if (batch_begin >= N_tris)
            break;

        // process the batch
        for (uint32 batch_offset = 0; batch_offset < BATCH_SIZE; batch_offset += 32)
        {
            const int32 tri_idx = batch_begin + batch_offset + warp_tid;
            const bool valid = tri_idx < N_tris;

            int32  tri_id;
            float4 v0, v1, v2;
            float3 edge0, edge1, edge2, n;
            int3   tri_bbox0;
            int3   tri_bbox1;
            int32  tri_axis;

            if (valid)
            {
                tri_id = tri_index[ tri_idx ];

                // process triangle
                const int4 tri = tex1Dfetch( tex_triangles, tri_id );

                v0 = tex1Dfetch( tex_vertices, tri.x );
                v1 = tex1Dfetch( tex_vertices, tri.y );
                v2 = tex1Dfetch( tex_vertices, tri.z );

                // fetch triangle bbox
                fetch_tri_bbox( tri_id, tri_bbox0, tri_bbox1, tri_axis );

                // compute triangle´s edges and normal
                edge0 = make_float3( v1.x - v0.x, v1.y - v0.y, v1.z - v0.z );
                edge1 = make_float3( v2.x - v1.x, v2.y - v1.y, v2.z - v1.z );
                edge2 = make_float3( v0.x - v2.x, v0.y - v2.y, v0.z - v2.z );
                n = anti_cross( edge0, edge2 );
            }
            else
            {
                tri_id    = 0;
                tri_bbox0 = make_int3( 0, 0, 0 );
                tri_bbox1 = make_int3( 0, 0, 0 );
                tri_axis  = 0;
            }

#define VOXELPIPE_FAST_PATH 0
#if VOXELPIPE_FAST_PATH
            const int32 frag_count_x = tri_bbox1.x - tri_bbox0.x + 1;
            const int32 frag_count_y = tri_bbox1.y - tri_bbox0.y + 1;
            const int32 frag_count_z = tri_bbox1.z - tri_bbox0.z + 1;

            const uint32 frag_count_u = (tri_axis == 0) ? frag_count_y : frag_count_x;
            const uint32 frag_count_v = (tri_axis == 2) ? frag_count_y : frag_count_z;

            if (__all( frag_count_u * frag_count_v <= 64 ))
            {
                // fast path
                if (__any( tri_axis == Z ))
                {
                    fast_emit_fragments< Z >(
                        sm_red,
                        sm_broadcast,
                        valid && (tri_axis == Z),
                        N,
                        tri_id,
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
                        fragment_counter,
                        fragment_tris,
                        fragment_ids,
                        prf_stats );
                }
                if (__any( tri_axis == Y ))
                {
                    fast_emit_fragments< Y >(
                        sm_red,
                        sm_broadcast,
                        valid && (tri_axis == Y),
                        N,
                        tri_id,
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
                        fragment_counter,
                        fragment_tris,
                        fragment_ids,
                        prf_stats );
                }
                if (__any( tri_axis == X ))
                {
                    fast_emit_fragments< X >(
                        sm_red,
                        sm_broadcast,
                        valid && (tri_axis == X),
                        N,
                        tri_id,
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
                        fragment_counter,
                        fragment_tris,
                        fragment_ids,
                        prf_stats );
                }
            }
            else
#endif
            {
                // slow path
                if (__any( tri_axis == Z ))
                {
                    emit_fragments< Z >(
                        sm_red,
                        sm_broadcast,
                        valid && (tri_axis == Z),
                        N,
                        tri_id,
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
                        fragment_counter,
                        fragment_tris,
                        fragment_ids,
                        prf_stats );
                }
                if (__any( tri_axis == Y ))
                {
                    emit_fragments< Y >(
                        sm_red,
                        sm_broadcast,
                        valid && (tri_axis == Y),
                        N,
                        tri_id,
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
                        fragment_counter,
                        fragment_tris,
                        fragment_ids,
                        prf_stats );
                }
                if (__any( tri_axis == X ))
                {
                    emit_fragments< X >(
                        sm_red,
                        sm_broadcast,
                        valid && (tri_axis == X),
                        N,
                        tri_id,
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
                        fragment_counter,
                        fragment_tris,
                        fragment_ids,
                        prf_stats );
                }
            }
        }
    }
}

template <typename shader_type, int32 BLOCK_SIZE, int32 VoxelType, int32 VoxelizationType>
__global__ void A_buffer_unsorted_large_kernel(
    const int32     grid_size,
    const int32     n_grids,
    const int32     N,
    const int32     log_N,
    const int32     N_tris,
    const int32*    tri_index,
    const float3    bbox0,
    const float3    bbox1,
    const float3    bbox_delta,
    const float3    inv_bbox_delta,
    uint32*         batch_counter,
    uint32*         fragment_counter,
    int32*          fragment_tris,
    int32*          fragment_ids,
    ABStats         prf_stats)
{
    const int32 BATCH_SIZE = 1; // controls load balancing granularity and per-warp atomics frequency,
                                // i.e. one atomic per warp every BATCH_SIZE triangles

    const int32 warp_id  = threadIdx.x >> 5;
    const int32 warp_tid = threadIdx.x & 31;

    volatile __shared__ int32  sm_red[ BLOCK_SIZE * 2 ];
    volatile __shared__ uint32 sm_broadcast[ 32 ];

    // each warp keeps getting some work
    for (;;)
    {
        // fetch a batch of work for the warp
        if (warp_tid == 0)
            sm_broadcast[ warp_id ] = atomicAdd( batch_counter, BATCH_SIZE );

        const int32 batch_begin = sm_broadcast[ warp_id ];

        if (batch_begin >= N_tris)
            break;

        // process the batch
        for (uint32 batch_offset = 0; batch_offset < BATCH_SIZE; batch_offset++)
        {
            // each thread in the warp processes the same triangle
            const int32 tri_idx = batch_begin + batch_offset;
            if (tri_idx >= N_tris)
                break;

            int32  tri_id;
            float4 v0, v1, v2;
            float3 edge0, edge1, edge2, n;
            int3   tri_bbox0;
            int3   tri_bbox1;
            int32  tri_axis;

            tri_id = tri_index[ tri_idx ];

            // process triangle
            const int4 tri = tex1Dfetch( tex_triangles, tri_id );

            v0 = tex1Dfetch( tex_vertices, tri.x );
            v1 = tex1Dfetch( tex_vertices, tri.y );
            v2 = tex1Dfetch( tex_vertices, tri.z );

            // fetch triangle bbox
            fetch_tri_bbox( tri_id, tri_bbox0, tri_bbox1, tri_axis );

            // compute triangle´s edges and normal
            edge0 = make_float3( v1.x - v0.x, v1.y - v0.y, v1.z - v0.z );
            edge1 = make_float3( v2.x - v1.x, v2.y - v1.y, v2.z - v1.z );
            edge2 = make_float3( v0.x - v2.x, v0.y - v2.y, v0.z - v2.z );
            n = anti_cross( edge0, edge2 );

            // select whether to process horizontal or vertical scanlines
            const uint32 frag_count_u = 1 + ((tri_axis == 0) ? tri_bbox1.y - tri_bbox0.y : tri_bbox1.x - tri_bbox0.x);
            const uint32 frag_count_v = 1 + ((tri_axis == 2) ? tri_bbox1.y - tri_bbox0.y : tri_bbox1.z - tri_bbox0.z);

            if (frag_count_u >= frag_count_v)
            {
                for (int32 v_base = 0; v_base < frag_count_v; v_base += 32)
                {
                    if (tri_axis == Z)
                    {
                        emit_fragments_u< Z >(
                            sm_red,
                            sm_broadcast,
                            N,
                            tri_id,
                            v_base + warp_tid,
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
                            fragment_counter,
                            fragment_tris,
                            fragment_ids,
                            prf_stats );
                    }
                    else if (tri_axis == Y)
                    {
                        emit_fragments_u< Y >(
                            sm_red,
                            sm_broadcast,
                            N,
                            tri_id,
                            v_base + warp_tid,
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
                            fragment_counter,
                            fragment_tris,
                            fragment_ids,
                            prf_stats );
                    }
                    else
                    {
                        emit_fragments_u< X >(
                            sm_red,
                            sm_broadcast,
                            N,
                            tri_id,
                            v_base + warp_tid,
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
                            fragment_counter,
                            fragment_tris,
                            fragment_ids,
                            prf_stats );
                    }
                }
            }
            else // process vertical lines
            {
                for (int32 u_base = 0; u_base < frag_count_u; u_base += 32)
                {
                    if (tri_axis == Z)
                    {
                        emit_fragments_v< Z >(
                            sm_red,
                            sm_broadcast,
                            N,
                            tri_id,
                            u_base + warp_tid,
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
                            fragment_counter,
                            fragment_tris,
                            fragment_ids,
                            prf_stats );
                    }
                    else if (tri_axis == Y)
                    {
                        emit_fragments_v< Y >(
                            sm_red,
                            sm_broadcast,
                            N,
                            tri_id,
                            u_base + warp_tid,
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
                            fragment_counter,
                            fragment_tris,
                            fragment_ids,
                            prf_stats );
                    }
                    else
                    {
                        emit_fragments_v< X >(
                            sm_red,
                            sm_broadcast,
                            N,
                            tri_id,
                            u_base + warp_tid,
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
                            fragment_counter,
                            fragment_tris,
                            fragment_ids,
                            prf_stats );
                    }
                }
            }
        }
    }
}

} // namespace AB


template <typename shader_type, int32 VoxelType, int32 VoxelizationType>
void A_buffer_unsorted_small(
    const int32     N,
    const int32     log_N,
    const int32     N_tris,
    const int32     N_vertices,
    const int32*    tri_index,
    const int4*     triangles,
    const float4*   vertices,
    const Tri_bbox* tri_bbox,
    const float3    bbox0,
    const float3    bbox1,
    const float3    bbox_delta,
    const float3    inv_bbox_delta,
    uint32*         batch_counter,
    uint32*         fragment_count,
    int32*          fragment_tris,
    int32*          fragment_ids,
    FineRasterStats& stats)
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
        sizeof(int32),
        cudaMemcpyHostToDevice );

    cudaThreadSynchronize();

    const int32 BLOCK_SIZE = 512;

    dim3 dim_block = dim3( BLOCK_SIZE, 1, 1 );
    dim3 dim_grid  = dim3( thrust::detail::device::cuda::arch::max_active_blocks(AB::A_buffer_unsorted_small_kernel<shader_type,BLOCK_SIZE,VoxelType,VoxelizationType>, BLOCK_SIZE, 0), 1, 1 );

    const int32 grid_size = dim_grid.x * BLOCK_SIZE;
    const int32 n_grids   = (N_tris + grid_size-1) / grid_size;

    AB::ABStats prf_stats;
#if VOXELPIPE_ENABLE_PROFILING
    const int32 n_warps = dim_grid.x * BLOCK_SIZE / 32;
    thrust::device_vector<int32> prf_samples_sum( n_warps, 0 );
    thrust::device_vector<int32> prf_samples_sum2( n_warps, 0 );
    thrust::device_vector<int32> prf_samples_count( n_warps, 0 );
    thrust::device_vector<int32> prf_utilization_sum0( n_warps, 0 );
    thrust::device_vector<int32> prf_utilization_count0( n_warps, 0 );
    thrust::device_vector<int32> prf_utilization_sum1( n_warps, 0 );
    thrust::device_vector<int32> prf_utilization_count1( n_warps, 0 );
    prf_stats.samples_sum       = thrust::raw_pointer_cast( &prf_samples_sum.front() );
    prf_stats.samples_sum2      = thrust::raw_pointer_cast( &prf_samples_sum2.front() );
    prf_stats.samples_count     = thrust::raw_pointer_cast( &prf_samples_count.front() );
    prf_stats.utilization_sum[0]   = thrust::raw_pointer_cast( &prf_utilization_sum0.front() );
    prf_stats.utilization_count[0] = thrust::raw_pointer_cast( &prf_utilization_count0.front() );
    prf_stats.utilization_sum[1]   = thrust::raw_pointer_cast( &prf_utilization_sum1.front() );
    prf_stats.utilization_count[1] = thrust::raw_pointer_cast( &prf_utilization_count1.front() );
#endif

    AB::A_buffer_unsorted_small_kernel<shader_type, BLOCK_SIZE, VoxelType, VoxelizationType> <<<dim_grid,dim_block>>>(
        grid_size,
        n_grids,
        N,
        log_N,
        N_tris,
        tri_index,
        bbox0,
        bbox1,
        bbox_delta,
        inv_bbox_delta,
        batch_counter,
        fragment_count,
        fragment_tris,
        fragment_ids,
        prf_stats );

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
        for (int32 warp_id = 0; warp_id < n_warps; ++warp_id)
        {
            const float sum  = float( h_prf_sum[ warp_id ] );
            const float cnt  = float( h_prf_count[ warp_id ] );
            if (cnt)
                avg += sum / cnt;
        }
        stats.utilization[0] = avg / float(n_warps);
    }
    {
        thrust::host_vector<int32> h_prf_sum( prf_utilization_sum1 );
        thrust::host_vector<int32> h_prf_count( prf_utilization_count1 );
        float avg = 0.0f;
        for (int32 warp_id = 0; warp_id < n_warps; ++warp_id)
        {
            const float sum  = float( h_prf_sum[ warp_id ] );
            const float cnt  = float( h_prf_count[ warp_id ] );
            if (cnt)
                avg += sum / cnt;
        }
        stats.utilization[1] = avg / float(n_warps);
    }
#endif
}
template <typename shader_type, int32 VoxelType, int32 VoxelizationType>
void A_buffer_unsorted_large(
    const int32     N,
    const int32     log_N,
    const int32     N_tris,
    const int32     N_vertices,
    const int32*    tri_index,
    const int4*     triangles,
    const float4*   vertices,
    const Tri_bbox* tri_bbox,
    const float3    bbox0,
    const float3    bbox1,
    const float3    bbox_delta,
    const float3    inv_bbox_delta,
    uint32*         batch_counter,
    uint32*         fragment_count,
    int32*          fragment_tris,
    int32*          fragment_ids,
    FineRasterStats& stats)
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
        sizeof(int32),
        cudaMemcpyHostToDevice );

    cudaThreadSynchronize();

    const int32 BLOCK_SIZE = 512;
    const int32 WARP_COUNT = BLOCK_SIZE >> 5;

    dim3 dim_block = dim3( BLOCK_SIZE, 1, 1 );
    dim3 dim_grid  = dim3( thrust::detail::device::cuda::arch::max_active_blocks(AB::A_buffer_unsorted_large_kernel<shader_type,BLOCK_SIZE,VoxelType,VoxelizationType>, BLOCK_SIZE, 0), 1, 1 );

    const int32 grid_size = dim_grid.x * WARP_COUNT;
    const int32 n_grids   = (N_tris + grid_size-1) / grid_size;

    AB::ABStats prf_stats;
#if VOXELPIPE_ENABLE_PROFILING
    const int32 n_warps = dim_grid.x * BLOCK_SIZE / 32;
    thrust::device_vector<int32> prf_samples_sum( n_warps, 0 );
    thrust::device_vector<int32> prf_samples_sum2( n_warps, 0 );
    thrust::device_vector<int32> prf_samples_count( n_warps, 0 );
    thrust::device_vector<int32> prf_utilization_sum0( n_warps, 0 );
    thrust::device_vector<int32> prf_utilization_count0( n_warps, 0 );
    thrust::device_vector<int32> prf_utilization_sum1( n_warps, 0 );
    thrust::device_vector<int32> prf_utilization_count1( n_warps, 0 );
    prf_stats.samples_sum       = thrust::raw_pointer_cast( &prf_samples_sum.front() );
    prf_stats.samples_sum2      = thrust::raw_pointer_cast( &prf_samples_sum2.front() );
    prf_stats.samples_count     = thrust::raw_pointer_cast( &prf_samples_count.front() );
    prf_stats.utilization_sum[0]   = thrust::raw_pointer_cast( &prf_utilization_sum0.front() );
    prf_stats.utilization_count[0] = thrust::raw_pointer_cast( &prf_utilization_count0.front() );
    prf_stats.utilization_sum[1]   = thrust::raw_pointer_cast( &prf_utilization_sum1.front() );
    prf_stats.utilization_count[1] = thrust::raw_pointer_cast( &prf_utilization_count1.front() );
#endif

    AB::A_buffer_unsorted_large_kernel<shader_type, BLOCK_SIZE, VoxelType, VoxelizationType> <<<dim_grid,dim_block>>>(
        grid_size,
        n_grids,
        N,
        log_N,
        N_tris,
        tri_index,
        bbox0,
        bbox1,
        bbox_delta,
        inv_bbox_delta,
        batch_counter,
        fragment_count,
        fragment_tris,
        fragment_ids,
        prf_stats );

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
        for (int32 warp_id = 0; warp_id < n_warps; ++warp_id)
        {
            const float sum  = float( h_prf_sum[ warp_id ] );
            const float cnt  = float( h_prf_count[ warp_id ] );
            if (cnt)
                avg += sum / cnt;
        }
        stats.utilization[0] = avg / float(n_warps);
    }
    {
        thrust::host_vector<int32> h_prf_sum( prf_utilization_sum1 );
        thrust::host_vector<int32> h_prf_count( prf_utilization_count1 );
        float avg = 0.0f;
        for (int32 warp_id = 0; warp_id < n_warps; ++warp_id)
        {
            const float sum  = float( h_prf_sum[ warp_id ] );
            const float cnt  = float( h_prf_count[ warp_id ] );
            if (cnt)
                avg += sum / cnt;
        }
        stats.utilization[1] = avg / float(n_warps);
    }
#endif
}

} // namespace voxelpipe
