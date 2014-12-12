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

/*! \file common.h
 *  \brief common definitions & utilities
 */

#pragma once

#include <device_functions.h>
#include <thrust/scan.h>
#include <voxelpipe/base.h>
#include "thrust_arch.h""

#define VOXELPIPE_CR_PERSISTENT_THREADS     1
#define VOXELPIPE_CR_DETERMINISTIC_OFFSETS  1
#define VOXELPIPE_CR_SCANLINE_SORTING       0

#define VOXELPIPE_FR_PERSISTENT_THREADS     1
#define VOXELPIPE_FR_FRAGMENT_DISTRIBUTION  0
#define VOXELPIPE_FR_SCANLINE_CLIPPING      1

#define VOXELPIPE_ENABLE_PROFILING          0


#if VOXELPIPE_ENABLE_PROFILING
  #define VOXELPIPE_PROFILE(x) x
#else
  #define VOXELPIPE_PROFILE(x)
#endif

namespace voxelpipe {

/// pack a normal in a uint32 with the given number of bits per component
template <int32 BITS> __device__ uint32 to_packed_normal(const float3 n);

struct FineRasterStats
{
    float   samples_avg;
    float   samples_std;
    float   utilization[2];
    uint32  fragment_count;
};

enum Voxel_enum
{
    Bit    = 0,
    Float  = 1,
    Float2 = 2,
    Float3 = 3,
    Float4 = 4
};

enum Voxelization_enum
{
    THIN_RASTER           = 0, // 6-separating
    CONSERVATIVE_RASTER   = 1  // 26-separating
};

enum Blending_enum
{
    NO_BLENDING  = 0,   // replace previous value
    ADD_BLENDING = 1,   // add to previous value
    MIN_BLENDING = 2,   // keep min value 
    MAX_BLENDING = 3    // keep max value
};

enum Voxel_format
{
    BIT_FORMAT   = 0,   // single-bit per pixel
    FP32S_FORMAT = 1,   // 32 bit floating point components, strided (RRRRRR,GGGGGG,BBBBBB)
    FP32V_FORMAT = 2,   // 32 bit floating point components, vector  (RGB,RGB,RGB)
    U8S_FORMAT   = 3,   // 8 bit integer components, strided         (RRRRRR,GGGGGG,BBBBBB)
    U8V_FORMAT   = 4,   // 8 bit integer components, vector          (RGB,RGB,RGB)
    U16S_FORMAT  = 5,   // 16 bit integer components, strided        (RRRRRR,GGGGGG,BBBBBB)
    U16V_FORMAT  = 6    // 16 bit integer components, vector         (RGB,RGB,RGB)
};

template <int32 VoxelType>
struct DefaultShader
{
};
template <>
struct DefaultShader<Bit>
{
    __device__ bool shade(
        const int32 tri_id,
        const float4 v0,
        const float4 v1,
        const float4 v2,
        const float3 n,
        const float  bary0,
        const float  bary1,
        const int3   xyz) const
    {
        return true;
    }
};
template <>
struct DefaultShader<Float4>
{
    __device__ float4 shade(
        const int32 tri_id,
        const float4 v0,
        const float4 v1,
        const float4 v2,
        const float3 n,
        const float  bary0,
        const float  bary1,
        const int3   xyz)
    {
        return make_float4( 1.0f, 1.0f, 1.0f, 1.0f );
    }
};
struct CheckerShader
{
    __device__ float shade(
        const int32 tri_id,
        const float4 v0,
        const float4 v1,
        const float4 v2,
        const float3 n,
        const float  bary0,
        const float  bary1,
        const int3   xyz) const
    {
        return (bary0 < 0.5f && bary1 < 0.5f) || (bary0 >= 0.5f && bary1 >= 0.5f) ? 1.0f : 0.0f;
    }
};
struct BaryShader
{
    __device__ float2 shade(
        const int32 tri_id,
        const float4 v0,
        const float4 v1,
        const float4 v2,
        const float3 n,
        const float  bary0,
        const float  bary1,
        const int3   xyz)
    {
        return make_float2( bary0, bary1 );
    }
};
struct GNormalShader
{
    __device__ float4 shade(
        const int32 tri_id,
        const float4 v0,
        const float4 v1,
        const float4 v2,
        const float3 n,
        const float  bary0,
        const float  bary1,
        const int3   xyz) const
    {
        const float inv_len = rsqrtf( n.x*n.x + n.y*n.y + n.z*n.z );
        return make_float4( n.x * inv_len, n.y * inv_len, n.z * inv_len, 1.0f );
    }
};
struct IDShader
{
    __device__ float shade(
        const int32 tri_id,
        const float4 v0,
        const float4 v1,
        const float4 v2,
        const float3 n,
        const float  bary0,
        const float  bary1,
        const int3   xyz) const
    {
        return __int_as_float( tri_id + 1 );
    }
};
struct PackedNormalShader
{
    __device__ float shade(
        const int32 tri_id,
        const float4 v0,
        const float4 v1,
        const float4 v2,
        const float3 n,
        const float  bary0,
        const float  bary1,
        const int3   xyz) const
    {
        // pack the result in a uint32
        union {
            uint32 u;
            float  f;
        } packed_normal;

        packed_normal.u = to_packed_normal<12>( n );

        // and return it as a float
        return packed_normal.f;
    }
};

// pack a normal in a uint32 with the given number of bits per component
template <int32 BITS>
__device__ uint32 to_packed_normal(const float3 n)
{
    const uint32 COMP_WORD = 1u << BITS;

    const bool byx = fabsf( n.y ) > fabsf( n.x );
    const bool byz = fabsf( n.y ) > fabsf( n.z );
    const bool bzx = fabsf( n.z ) > fabsf( n.x );
    const int32 axis = byx ? (byz ? 1 : 2) : (bzx ? 2 : 0);

    // compute the sign of the dominant axis (sgn_w), and normalized coordinates for the others (nu & nv)
    uint32 sgn_w;
    float nu;
    float nv;

    if (axis == 2)
    {
        const float inv_len = __frcp_rn( n.z );
        nu = n.x * inv_len;
        nv = n.y * inv_len;
        sgn_w = n.z >= 0.0f ? 1 : 0;
    }
    else if (axis == 1)
    {
        const float inv_len = __frcp_rn( n.y );
        nu = n.x * inv_len;
        nv = n.z * inv_len;
        sgn_w = n.y >= 0.0f ? 1 : 0;
    }
    else
    {
        const float inv_len = __frcp_rn( n.x );
        nu = n.y * inv_len;
        nv = n.z * inv_len;
        sgn_w = n.x >= 0.0f ? 1 : 0;
    }

    return (axis+1u) | (sgn_w << 2) |
        (nih::max( nih::min( uint32( (nu + 1.0f)*0.5f * COMP_WORD ), COMP_WORD - 1u ), 0u ) << 3) |
        (nih::max( nih::min( uint32( (nv + 1.0f)*0.5f * COMP_WORD ), COMP_WORD - 1u ), 0u ) << (3+BITS));
}

} // namespace voxelpipe
