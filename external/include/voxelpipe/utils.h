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

/*! \file utils.h
 *  \brief common utilities
 */

#pragma once

#include <device_functions.h>
#include <voxelpipe/base.h>

namespace voxelpipe {

template <int32 log_N_TILES> struct tile_id_selector    { typedef uint32 type; };
template <>                  struct tile_id_selector<2> { typedef uint8  type; };
template <>                  struct tile_id_selector<3> { typedef uint16 type; };
#if !VOXELPIPE_CR_SCANLINE_SORTING
template <>                  struct tile_id_selector<4> { typedef uint16 type; };
#endif

enum XYZ
{
    X = 0,
    Y = 1,
    Z = 2
};

struct Tri_bbox
{
    int3    m_bbox0;
    int3    m_bbox1;
    int32   m_pad[2];

    static const uint32 N_INT4 = 2;
};

extern texture<float4> tex_vertices;
extern texture<int4>   tex_triangles;
extern texture<int4>   tex_tri_bbox;

//
// utility functions
//

__forceinline__ __device__ void swap(float4& a, float4& b)
{
    const float4 tmp = a;
    a = b;
    b = tmp;
}
__forceinline__ __device__ float3 min3(const float4 a, const float4 b, const float4 c)
{
    return make_float3(
        fminf( a.x, fminf( b.x, c.x ) ),
        fminf( a.y, fminf( b.y, c.y ) ),
        fminf( a.z, fminf( b.z, c.z ) ) );
}
__forceinline__ __device__ float3 max3(const float4 a, const float4 b, const float4 c)
{
    return make_float3(
        fmax( a.x, fmaxf( b.x, c.x ) ),
        fmax( a.y, fmaxf( b.y, c.y ) ),
        fmax( a.z, fmaxf( b.z, c.z ) ) );
}
__forceinline__ __device__ float3 cross(const float3& op1, const float3& op2)
{
	return make_float3(
		op1.y*op2.z - op1.z*op2.y,
		op1.z*op2.x - op1.x*op2.z,
		op1.x*op2.y - op1.y*op2.x );
}
__forceinline__ __device__ float3 anti_cross(const float3& op1, const float3& op2)
{
	return make_float3(
		op1.z*op2.y - op1.y*op2.z,
		op1.x*op2.z - op1.z*op2.x,
		op1.y*op2.x - op1.x*op2.y );
}

template <typename Vec> struct base_type {};

template <> struct base_type<float2> { typedef float value_type; };
template <> struct base_type<float3> { typedef float value_type; };
template <> struct base_type<float4> { typedef float value_type; };
template <> struct base_type<int2>   { typedef int value_type; };
template <> struct base_type<int3>   { typedef int value_type; };
template <> struct base_type<int4>   { typedef int value_type; };

template <int32 axis>
struct uvw {};

template <>
struct uvw<0>
{
    static const int32 U = 1;
    static const int32 V = 2;
    static const int32 W = 0;

    template <typename Vec> __device__ __forceinline__ static typename base_type<Vec>::value_type u(const Vec a) { return a.y; }
    template <typename Vec> __device__ __forceinline__ static typename base_type<Vec>::value_type v(const Vec a) { return a.z; }
    template <typename Vec> __device__ __forceinline__ static typename base_type<Vec>::value_type w(const Vec a) { return a.x; }
    template <typename Vec> __device__ __forceinline__ static typename base_type<Vec>::value_type x(const Vec a) { return a.z; }
    template <typename Vec> __device__ __forceinline__ static typename base_type<Vec>::value_type y(const Vec a) { return a.x; }
    template <typename Vec> __device__ __forceinline__ static typename base_type<Vec>::value_type z(const Vec a) { return a.y; }

    __device__ __forceinline__ static int32 u_stride(const int32 T) { return T; }
    __device__ __forceinline__ static int32 v_stride(const int32 T) { return T*T; }
    __device__ __forceinline__ static int32 w_stride(const int32 T) { return 1; }

    __device__ __forceinline__ static float ccw(const float3 n) { return n.x > 0.0f ? 1.0f : -1.0f; }
};
template <>
struct uvw<1>
{
    static const int32 U = 0;
    static const int32 V = 2;
    static const int32 W = 1;

    template <typename Vec> __device__ __forceinline__ static typename base_type<Vec>::value_type u(const Vec a) { return a.x; }
    template <typename Vec> __device__ __forceinline__ static typename base_type<Vec>::value_type v(const Vec a) { return a.z; }
    template <typename Vec> __device__ __forceinline__ static typename base_type<Vec>::value_type w(const Vec a) { return a.y; }
    template <typename Vec> __device__ __forceinline__ static typename base_type<Vec>::value_type x(const Vec a) { return a.x; }
    template <typename Vec> __device__ __forceinline__ static typename base_type<Vec>::value_type y(const Vec a) { return a.z; }
    template <typename Vec> __device__ __forceinline__ static typename base_type<Vec>::value_type z(const Vec a) { return a.y; }

    __device__ __forceinline__ static int32 u_stride(const int32 T) { return 1; }
    __device__ __forceinline__ static int32 v_stride(const int32 T) { return T*T; }
    __device__ __forceinline__ static int32 w_stride(const int32 T) { return T; }

    __device__ __forceinline__ static float ccw(const float3 n) { return n.y < 0.0f ? 1.0f : -1.0f; }
};
template <>
struct uvw<2>
{
    static const int32 U = 0;
    static const int32 V = 1;
    static const int32 W = 2;

    template <typename Vec> __device__ __forceinline__ static typename base_type<Vec>::value_type u(const Vec a) { return a.x; }
    template <typename Vec> __device__ __forceinline__ static typename base_type<Vec>::value_type v(const Vec a) { return a.y; }
    template <typename Vec> __device__ __forceinline__ static typename base_type<Vec>::value_type w(const Vec a) { return a.z; }
    template <typename Vec> __device__ __forceinline__ static typename base_type<Vec>::value_type x(const Vec a) { return a.x; }
    template <typename Vec> __device__ __forceinline__ static typename base_type<Vec>::value_type y(const Vec a) { return a.y; }
    template <typename Vec> __device__ __forceinline__ static typename base_type<Vec>::value_type z(const Vec a) { return a.z; }

    __device__ __forceinline__ static int32 u_stride(const int32 T) { return 1; }
    __device__ __forceinline__ static int32 v_stride(const int32 T) { return T; }
    __device__ __forceinline__ static int32 w_stride(const int32 T) { return T*T; }

    __device__ __forceinline__ static float ccw(const float3 n) { return n.z > 0.0f ? 1.0f : -1.0f; }
};

template <int32 AXIS> __device__ __forceinline__ float3 swap_coords(const float3 a) { return make_float3( uvw<AXIS>::u( a ), uvw<AXIS>::v( a ),  uvw<AXIS>::w( a ) ); }
template <int32 AXIS> __device__ __forceinline__ float4 swap_coords(const float4 a) { return make_float4( uvw<AXIS>::u( a ), uvw<AXIS>::v( a ),  uvw<AXIS>::w( a ), a.w ); }
template <int32 AXIS> __device__ __forceinline__ int3   swap_coords(const int3 a)   { return make_int3( uvw<AXIS>::u( a ), uvw<AXIS>::v( a ),  uvw<AXIS>::w( a ) ); }
template <int32 AXIS> __device__ __forceinline__ int4   swap_coords(const int4 a)   { return make_int4( uvw<AXIS>::u( a ), uvw<AXIS>::v( a ),  uvw<AXIS>::w( a ), a.w ); }

/// perform triangle setup
template <int32 AXIS>
__device__ void triangle_setup(
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
    float3&       n_delta_u,
    float3&       n_delta_v,
    float3&       a)
{
    typedef uvw<AXIS> sel;

    const float sgn_w = sel::ccw( n );

    const float2 n0 = make_float2( -sel::v( edge0 ) * sgn_w, sel::u( edge0 ) * sgn_w );
    const float  d0 = -(n0.x * sel::u( v0 ) + n0.y * sel::v( v0 )) +
        fmaxf( 0.0f, sel::u( bbox_delta )*n0.x ) +
        fmaxf( 0.0f, sel::v( bbox_delta )*n0.y );

    const float2 n1 = make_float2( -sel::v( edge1 ) * sgn_w, sel::u( edge1 ) * sgn_w );
    const float  d1 = -(n1.x * sel::u( v1 ) + n1.y * sel::v( v1 )) +
        fmaxf( 0.0f, sel::u( bbox_delta )*n1.x ) +
        fmaxf( 0.0f, sel::v( bbox_delta )*n1.y );

    const float2 n2 = make_float2( -sel::v( edge2 ) * sgn_w, sel::u( edge2 ) * sgn_w );
    const float  d2 = -(n2.x * sel::u( v2 ) + n2.y * sel::v( v2 )) +
        fmaxf( 0.0f, sel::u( bbox_delta )*n2.x ) +
        fmaxf( 0.0f, sel::v( bbox_delta )*n2.y );

    a.x = (n0.x * sel::u( bbox0 ) + n0.y * sel::v( bbox0 )) + d0;
    a.y = (n1.x * sel::u( bbox0 ) + n1.y * sel::v( bbox0 )) + d1;
    a.z = (n2.x * sel::u( bbox0 ) + n2.y * sel::v( bbox0 )) + d2;

    n_delta_u = make_float3(
        n0.x * sel::u( bbox_delta ),
        n1.x * sel::u( bbox_delta ),
        n2.x * sel::u( bbox_delta ) );

    n_delta_v = make_float3(
        n0.y * sel::v( bbox_delta ),
        n1.y * sel::v( bbox_delta ),
        n2.y * sel::v( bbox_delta ) );
}

/// perform plane setup
template <int32 AXIS>
__device__ void plane_setup(
    const float3  bbox0,
    const float4  v0,
    const float3  n,
    float4&       plane_eq)
{
    typedef uvw<AXIS> sel;

    const float inv_n = __frcp_rn( sel::w( n ) );
    plane_eq.x = sel::u( n ) * inv_n;
    plane_eq.y = sel::v( n ) * inv_n;
    plane_eq.z =
           plane_eq.x * sel::u( v0 )
         + plane_eq.y * sel::v( v0 )
         + sel::w( v0 ) - sel::w( bbox0 )
         - plane_eq.x * sel::u( bbox0 )
         - plane_eq.y * sel::v( bbox0 );
    plane_eq.w = inv_n;
}

// intra-warp inclusive scan
template <typename T> __device__ __forceinline__ T scan_warp(T val, const int32 tidx, volatile T *red)
{
    // pad initial segment with zeros
    red[tidx] = 0;
    red += 32;

    // Hillis-Steele scan
    red[tidx] = val;
    val += red[tidx-1];  red[tidx] = val;
    val += red[tidx-2];  red[tidx] = val;
    val += red[tidx-4];  red[tidx] = val;
    val += red[tidx-8];  red[tidx] = val;
    val += red[tidx-16]; red[tidx] = val;
	return val;
}
// return the total from a scan_warp
template <typename T> __device__ __forceinline__ T scan_warp_total(volatile T *red) { return red[63]; }

// generic chunked scan
template <int32 CHUNK_SIZE>
__device__ __forceinline__ uint32 scan(const uint32 val, const int tidx, volatile unsigned int* red) { return 0; }

/// quarter-warp inclusive scan
template <>
__device__ __forceinline__ uint32 scan<8>(uint32 val, const int tidx, volatile unsigned int* red)
{
    red[tidx] = 0;
    red += 8;

    // Hillis-Steele scan
    red[tidx] = val;
    val += red[tidx-1];  red[tidx] = val;
    val += red[tidx-2];  red[tidx] = val;
    val += red[tidx-4];  red[tidx] = val;
	return val;
}
/// half-warp inclusive scan
template <>
__device__ __forceinline__ uint32 scan<16>(uint32 val, const int tidx, volatile unsigned int* red)
{
    red[tidx] = 0;
    red += 16;

    // Hillis-Steele scan
    red[tidx] = val;
    val += red[tidx-1];  red[tidx] = val;
    val += red[tidx-2];  red[tidx] = val;
    val += red[tidx-4];  red[tidx] = val;
    val += red[tidx-8];  red[tidx] = val;
	return val;
}
/// warp inclusive scan
template <>
__device__ __forceinline__ uint32 scan<32>(uint32 val, const int tidx, volatile unsigned int* red)
{
    red[tidx] = 0;
    red += 32;

    // Hillis-Steele scan
    red[tidx] = val;
    val += red[tidx-1];  red[tidx] = val;
    val += red[tidx-2];  red[tidx] = val;
    val += red[tidx-4];  red[tidx] = val;
    val += red[tidx-8];  red[tidx] = val;
    val += red[tidx-16]; red[tidx] = val;
	return val;
}

/// single-SM exclusive scan kernel
template <int32 BLOCK_SIZE>
__global__ void exclusive_scan_kernel(
    const int32 N,
    int32* values,
    int32* sum,
    int32* init)
{
    const int32 WARP_COUNT = BLOCK_SIZE >> 5;

    volatile __shared__ int sm_red[ BLOCK_SIZE * 2 ];
    volatile __shared__ int sm_red_cta[ WARP_COUNT+1 ];

    const int32 warp_id  = threadIdx.x >> 5;
    const int32 warp_tid = threadIdx.x & 31;

    int32 curry = init ? *init : 0;

    for (int32 offset = 0; offset < N; offset += BLOCK_SIZE)
    {
        __syncthreads(); // protect from previous iteration

        volatile int* red = sm_red + warp_id*64;

        const int32 idx = offset + threadIdx.x;

        const int32 value = idx < N ? values[ idx ] : 0;

        const int32 pop_scan = scan_warp( value, warp_tid, red );

        if (warp_tid == 31)
            sm_red_cta[ warp_id+1 ] = pop_scan;

        __syncthreads(); // wait for every warp to write its pop count

        if (threadIdx.x == 0)
        {
            sm_red_cta[0] = curry;
            for (int32 i = 1; i <= WARP_COUNT; ++i)
                sm_red_cta[i] += sm_red_cta[i-1];

            curry = sm_red_cta[ WARP_COUNT ];
        }

        __syncthreads(); // wait for first warp to finish scanning the warp values

        if (idx < N)
            values[ idx ] = sm_red_cta[ warp_id ] + pop_scan - value;
    }

    if (threadIdx.x == 0 && sum)
        *sum = curry;
}

/// single-SM exclusive scan
inline void exclusive_scan(
    const int32         N,
    int32*              values,
    int32*              sum = 0,
    int32*              init = 0)
{
    const int32 BLOCK_SIZE = 512;
    dim3 dim_block = dim3( BLOCK_SIZE, 1, 1 );
    dim3 dim_grid  = dim3( 1, 1, 1 );

    exclusive_scan_kernel<BLOCK_SIZE> <<<dim_grid,dim_block>>>( N, values, sum, init );

    cudaThreadSynchronize();
}

/// do a block-wide reduction
template <int32 WARP_COUNT>
__device__ inline int32 block_reduce(
    const int32 value,
    volatile int* red,
    volatile int* red_cta)
{
    __syncthreads(); // protect from previous iteration

    const int32 warp_id  = threadIdx.x >> 5;
    const int32 warp_tid = threadIdx.x & 31;

    volatile int* warp_red = red + warp_id*64;

    const int32 pop_scan = scan_warp( value, warp_tid, warp_red );

    if (warp_tid == 31)
        red_cta[ warp_id ] = pop_scan;

    __syncthreads(); // wait for every warp to write its pop count

    int32 count = 0;
    if (threadIdx.x == 0)
    {
        for (int32 i = 0; i < WARP_COUNT; ++i)
            count += red_cta[i];
    }
    return count;
}
/// do a block-wide scan
template <int32 WARP_COUNT, typename T>
__device__ inline int32 block_scan(
    const T value,
    volatile T* red,
    volatile T* red_cta)
{
    __syncthreads(); // protect from previous iteration

    const int32 warp_id  = threadIdx.x >> 5;
    const int32 warp_tid = threadIdx.x & 31;

    volatile T* warp_red = red + warp_id*64;

    const T pop_scan = scan_warp( value, warp_tid, warp_red );

    if (warp_tid == 31)
        red_cta[ warp_id+1 ] = pop_scan;

    __syncthreads(); // wait for every warp to write its pop count

    if (threadIdx.x == 0)
    {
        red_cta[0] = 0;
        for (int32 i = 1; i <= WARP_COUNT; ++i)
            red_cta[i] += red_cta[i-1];
    }

    __syncthreads(); // wait for the first thread to finish scanning the per-warp values

    return red_cta[ warp_id ] + pop_scan;
}
/// return the total of a block-wide scan
template <int32 WARP_COUNT, typename T>
__device__ inline int32 block_scan_total(volatile T* red_cta) { return red_cta[ WARP_COUNT ]; }


///
/// compute an inclusive pop count for elements belonging to 4 different classes at once
///
template <int32 BLOCK_SIZE>
__device__ uint32 pop_count4(
    const int32     digit,
    const bool      bit,
    volatile uint32* sm_red,
    volatile uint32* sm_red_cta)
{
    __syncthreads(); // protect from other iterations

    const uint32 WARP_COUNT = BLOCK_SIZE >> 5;

    const int32 warp_id  = threadIdx.x >> 5;
    const int32 warp_tid = threadIdx.x & 31;

    volatile uint32* red = sm_red + warp_id * 64;

    // do a compound per-warp pop-count on 4 counters
    const uint32 warp_popc4 = scan_warp( bit ? 1u << (digit*8) : 0u, warp_tid, red );

    // save the compound pop-count for each warp
    if (warp_tid == 31)
        sm_red_cta[ warp_id ] = warp_popc4;

    __syncthreads(); // wait until all per-warp pop-counts have been written

    // unpack the compound pop-counts and scan them
    if (threadIdx.x < 32)
    {
        const uint32 thread_digit = threadIdx.x / WARP_COUNT;
        const uint32 thread_chunk = threadIdx.x & (WARP_COUNT-1);

        const uint32 chunk_popc  = sm_red_cta[ thread_chunk ];
        const uint32 digit_shift = thread_digit*8;

        const uint32 value = (chunk_popc >> digit_shift) & 255u;

        sm_red_cta[ warp_tid ] = scan<WARP_COUNT>( value, thread_chunk, sm_red + thread_digit*WARP_COUNT*2 );
    }

    __syncthreads(); // wait until all per-warp pop-counts have been written

    const uint32 warp_popc    = (warp_popc4 >> digit*8) & 255u;
    const uint32 warp_offset  = warp_id ? sm_red_cta[ digit*WARP_COUNT + warp_id-1 ] : 0;
    return warp_offset + warp_popc;
}
template <int32 BLOCK_SIZE>
__device__ uint32 pop_count4_total(const int32 digit, volatile uint32* sm_red_cta)
{
    const uint32 WARP_COUNT = BLOCK_SIZE >> 5;
    return sm_red_cta[ digit*WARP_COUNT + WARP_COUNT-1 ];
}

/// fetch a triangle bbox and its dominant axis
inline __device__ void fetch_tri_bbox(
    const Tri_bbox* tri_bbox,
    const int32 tri_id,
    int3&       tri_bbox0,
    int3&       tri_bbox1,
    int32&      tri_axis)
{
    {
        const int4 tmp = ((const int4*)tri_bbox)[ tri_id*Tri_bbox::N_INT4 ];
        tri_bbox0.x = tmp.x;
        tri_bbox0.y = tmp.y;
        tri_bbox0.z = tmp.z;
        tri_bbox1.x = tmp.w;
    }
    {
        const int4 tmp = ((const int4*)tri_bbox)[ tri_id*Tri_bbox::N_INT4 + 1 ];
        tri_bbox1.y = tmp.x;
        tri_bbox1.z = tmp.y;
        tri_axis    = tmp.z;
    }
}

/// fetch a triangle bbox and its dominant axis
inline __device__ void fetch_tri_bbox(
    const int32 tri_id,
    int3&       tri_bbox0,
    int3&       tri_bbox1,
    int32&      tri_axis)
{
    {
        const int4 tmp = tex1Dfetch( tex_tri_bbox, tri_id*Tri_bbox::N_INT4 );
        tri_bbox0.x = tmp.x;
        tri_bbox0.y = tmp.y;
        tri_bbox0.z = tmp.z;
        tri_bbox1.x = tmp.w;
    }
    {
        const int4 tmp = tex1Dfetch( tex_tri_bbox, tri_id*Tri_bbox::N_INT4 + 1 );
        tri_bbox1.y = tmp.x;
        tri_bbox1.z = tmp.y;
        tri_axis    = tmp.z;
    }
}

/// fetch a triangle bbox
inline __device__ void fetch_tri_bbox(
    const int32 tri_id,
    int3&       tri_bbox0,
    int3&       tri_bbox1)
{
    {
        const int4 tmp = tex1Dfetch( tex_tri_bbox, tri_id*Tri_bbox::N_INT4 );
        tri_bbox0.x = tmp.x;
        tri_bbox0.y = tmp.y;
        tri_bbox0.z = tmp.z;
        tri_bbox1.x = tmp.w;
    }
    {
        const int4 tmp = tex1Dfetch( tex_tri_bbox, tri_id*Tri_bbox::N_INT4 + 1 );
        tri_bbox1.y = tmp.x;
        tri_bbox1.z = tmp.y;
    }
}

} // namespace voxelpipe
