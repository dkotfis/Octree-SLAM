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

/*! \file tile.h
 *  \brief utility classes to perform tile I/O in shared and global memory
 */

#pragma once

#include <voxelpipe/common.h>

namespace voxelpipe {

namespace FR {

// template base class implementing blending operations
template <typename T, int32 OP>
struct BlendingOp {};

// specialization of BlendingOp to <float,NO_BLENDING>
template <>
struct BlendingOp<float,NO_BLENDING>
{
    __device__ __forceinline__ static float clearword() { return __int_as_float( 0xFFFFFFFF ); }
    __device__ __forceinline__ static void  atomicOp(float* storage, const float value) { *storage = value; }
    __device__ __forceinline__ static float op(const float value1, const float value2)  { return __float_as_int( value2 ) != __float_as_int( clearword() ) ? value2 : value1; }
};
// specialization of BlendingOp to <float,ADD_BLENDING>
template <>
struct BlendingOp<float,ADD_BLENDING>
{
    __device__ __forceinline__ static float clearword() { return 0.0f; }
    __device__ __forceinline__ static void  atomicOp(float* storage, const float value) { atomicAdd( storage, value ); }
    __device__ __forceinline__ static float op(const float value1, const float value2)  { return value1 + value2; }
};
// specialization of BlendingOp to <float,MIN_BLENDING>
template <>
struct BlendingOp<float,MIN_BLENDING>   // NOTE: works only for positive floats
{
    __device__ __forceinline__ static float clearword() { return  1.0e8f; }
    __device__ __forceinline__ static void  atomicOp(float* storage, const float value) { atomicMin( (int*)storage, __float_as_int(value) ); }
    __device__ __forceinline__ static float op(const float value1, const float value2)  { return value1 < value2 ? value1 : value2; }
};
// specialization of BlendingOp to <float,MAX_BLENDING>
template <>
struct BlendingOp<float,MAX_BLENDING>   // NOTE: works only for positive floats
{
    __device__ __forceinline__ static float clearword() { return 0.0f; }
    __device__ __forceinline__ static void  atomicOp(float* storage, const float value) { atomicMax( (int*)storage, __float_as_int(value) ); }
    __device__ __forceinline__ static float op(const float value1, const float value2)  { return value1 > value2 ? value1 : value2; }
};

// quantize a floating point in [0,1) to 8 bits
__device__ __forceinline__ uint8 quantize8(const float x)
{
    return uint8( fmaxf( fminf( x * 256.0f, 255.0f ), 0.0f ) );
}

// template base class for performing tile operations
template <int32 VoxelType, int32 BlendingMode, int32 LOG_TILE_SIZE>
struct TileOp {};

// template base class for performing tile I/O
template <int32 VoxelType, int32 VoxelFormat, int32 LOG_TILE_SIZE, int32 CTA_SIZE>
struct TileIO {};

// specialization of TileOp to <VoxelType = Bit>
template <int32 BlendingMode,int32 LOG_TILE_SIZE>
struct TileOp<Bit,BlendingMode,LOG_TILE_SIZE>
{
    static const int32 T  = 1 << LOG_TILE_SIZE;
    static const int32 STORAGE_SIZE = (T*T*T) >> 5;

    typedef uint32 storage_type;
    typedef bool   pixel_type;

    // word used to clear the tile
    __device__ static storage_type clearword() { return 0; }

    // write to a pixel
    __device__ static void write(storage_type* storage, const int32 pixel, const bool value)
    {
        if (value)
        {
            const int32 word_index = pixel >> 5;
            const int32 bit_index  = pixel & 31;
            atomicOr( storage + word_index, 1 << bit_index );
        }
    }
    // blend two values
    __device__ static storage_type blend(const storage_type value1, const storage_type value2)
    {
        return value1 | value2;
    }
};
// specialization of TileOp to <VoxelType = Bit,LOG_TILE_SIZE = 2>
// if the tile is small enough, we can represent all the booleans in shared memory as integers
// which don´t need any atomics to be set to 1 (if any thread sets a word to 1, the word will
// be set independently of whether there are conflicts or not)
template <int32 BlendingMode>
struct TileOp<Bit,BlendingMode,2>
{
    static const int32 T  = 1 << 2;
    static const int32 STORAGE_SIZE = T*T*T;

    typedef int32 storage_type;
    typedef bool  pixel_type;

    // word used to clear the tile
    __device__ static storage_type clearword() { return 0; }

    // write to a pixel
    __device__ static void write(storage_type* storage, const int32 pixel, const bool value)
    {
        if (value)
            storage[ pixel ] = 1;
    }
    // blend two values
    __device__ static storage_type blend(const storage_type value1, const storage_type value2)
    {
        return value1 | value2;
    }
};
// specialization of TileOp to <VoxelType = Bit,LOG_TILE_SIZE = 3>
// if the tile is small enough, we can represent all the booleans in shared memory as integers
// which don´t need any atomics to be set to 1 (if any thread sets a word to 1, the word will
// be set independently of whether there are conflicts or not)
template <int32 BlendingMode>
struct TileOp<Bit,BlendingMode,3>
{
    static const int32 T  = 1 << 3;
    static const int32 STORAGE_SIZE = T*T*T;

    typedef int32 storage_type;
    typedef bool  pixel_type;

    // word used to clear the tile
    __device__ static storage_type clearword() { return 0; }

    // write to a pixel
    __device__ static void write(storage_type* storage, const int32 pixel, const bool value)
    {
        if (value)
            storage[ pixel ] = value;
    }
    // blend two values
    __device__ static storage_type blend(const storage_type value1, const storage_type value2)
    {
        return value1 | value2;
    }
};
/*
// specialization of TileOp to <VoxelType = Bit,LOG_TILE_SIZE = 4>
// if the tile is small enough, we can represent all the booleans in shared memory as integers
// which don´t need any atomics to be set to 1 (if any thread sets a word to 1, the word will
// be set independently of whether there are conflicts or not)
template <>
struct TileOp<Bit,4>
{
    static const int32 T  = 1 << 4;
    static const int32 STORAGE_SIZE = T*T*T;

    typedef int32 storage_type;
    typedef bool  pixel_type;

    __device__ static void write(storage_type* storage, const int32 pixel, const bool value)
    {
        if (value)
            storage[ pixel ] = 1;
    }
};
*/

// specialization of TileOp to <VoxelType = Float>
template <int32 BlendingMode, int32 LOG_TILE_SIZE>
struct TileOp<Float,BlendingMode,LOG_TILE_SIZE>
{
    static const int32 T  = 1 << LOG_TILE_SIZE;
    static const int32 STORAGE_SIZE = T*T*T;

    typedef float  storage_type;
    typedef float  pixel_type;
    typedef BlendingOp<storage_type,BlendingMode> blending_op;

    // word used to clear the tile
    __device__ static storage_type clearword() { return blending_op::clearword(); }

    // write to a pixel
    __device__ static void write(storage_type* storage, const int32 pixel, const float value)
    {
        blending_op::atomicOp( storage + pixel,           value );
    }
    // blend two values
    __device__ static storage_type blend(const storage_type value1, const storage_type value2)
    {
        return blending_op::op( value1, value2 );
    }
};
// specialization of TileOp to <VoxelType = Float2>
template <int32 BlendingMode, int32 LOG_TILE_SIZE>
struct TileOp<Float2,BlendingMode,LOG_TILE_SIZE>
{
    static const int32 T  = 1 << LOG_TILE_SIZE;
    static const int32 STORAGE_SIZE = T*T*T*2;

    typedef float  storage_type;
    typedef float2 pixel_type;
    typedef BlendingOp<storage_type,BlendingMode> blending_op;

    // word used to clear the tile
    __device__ static storage_type clearword() { return blending_op::clearword(); }

    // write to a pixel
    __device__ static void write(storage_type* storage, const int32 pixel, const float2 value)
    {
        blending_op::atomicOp( storage + pixel,           value.x );
        blending_op::atomicOp( storage + pixel + T*T*T,   value.y );
    }
    // blend two values
    __device__ static storage_type blend(const storage_type value1, const storage_type value2)
    {
        return blending_op::op( value1, value2 );
    }
};
// specialization of TileOp to <VoxelType = Float3>
template <int32 BlendingMode, int32 LOG_TILE_SIZE>
struct TileOp<Float3,BlendingMode,LOG_TILE_SIZE>
{
    static const int32 T  = 1 << LOG_TILE_SIZE;
    static const int32 STORAGE_SIZE = T*T*T*3;

    typedef float  storage_type;
    typedef float3 pixel_type;
    typedef BlendingOp<storage_type,BlendingMode> blending_op;

    // word used to clear the tile
    __device__ static storage_type clearword() { return blending_op::clearword(); }

    // write to a pixel
    __device__ static void write(storage_type* storage, const int32 pixel, const float3 value)
    {
        blending_op::atomicOp( storage + pixel,           value.x );
        blending_op::atomicOp( storage + pixel + T*T*T,   value.y );
        blending_op::atomicOp( storage + pixel + T*T*T*2, value.z );
    }
    // blend two values
    __device__ static storage_type blend(const storage_type value1, const storage_type value2)
    {
        return blending_op::op( value1, value2 );
    }
};
// specialization of TileOp to <VoxelType = Float>
template <int32 BlendingMode, int32 LOG_TILE_SIZE>
struct TileOp<Float4,BlendingMode,LOG_TILE_SIZE>
{
    static const int32 T  = 1 << LOG_TILE_SIZE;
    static const int32 STORAGE_SIZE = T*T*T*4;

    typedef float  storage_type;
    typedef float4 pixel_type;
    typedef BlendingOp<storage_type,BlendingMode> blending_op;

    // word used to clear the tile
    __device__ static storage_type clearword() { return blending_op::clearword(); }

    // write to a pixel
    __device__ static void write(storage_type* storage, const int32 pixel, const float4 value)
    {
        blending_op::atomicOp( storage + pixel,           value.x );
        blending_op::atomicOp( storage + pixel + T*T*T,   value.y );
        blending_op::atomicOp( storage + pixel + T*T*T*2, value.z );
        blending_op::atomicOp( storage + pixel + T*T*T*3, value.w );
    }
    // blend two values
    __device__ static storage_type blend(const storage_type value1, const storage_type value2)
    {
        return blending_op::op( value1, value2 );
    }
};


// specialization of TileIO to <VoxelType = Bit,VoxelFormat = BIT_FORMAT>
template <int32 LOG_TILE_SIZE, int32 CTA_SIZE>
struct TileIO<Bit,BIT_FORMAT,LOG_TILE_SIZE,CTA_SIZE>
{
    typedef TileOp<Bit,ADD_BLENDING,LOG_TILE_SIZE> tile_op_type;
    typedef typename tile_op_type::storage_type storage_type;
    const static int32 STORAGE_SIZE = tile_op_type::STORAGE_SIZE;

    // load tile from gmem
    __device__ static void load(
        storage_type*       storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 WORDS_PER_THREAD = (STORAGE_SIZE + CTA_SIZE-1) / CTA_SIZE;

        storage_type* tile = (storage_type*)(fb) + tile_id * STORAGE_SIZE;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < WORDS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < STORAGE_SIZE)
                storage[ pixel ] = tile[ pixel ];
        }
    }

    // save tile to gmem
    __device__ static void save(
        const storage_type* storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 WORDS_PER_THREAD = (STORAGE_SIZE + CTA_SIZE-1) / CTA_SIZE;

        storage_type* tile = (storage_type*)(fb) + tile_id * STORAGE_SIZE;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < WORDS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < STORAGE_SIZE)
                tile[ pixel ] = storage[ pixel ];
        }
    }
};

// specialization of TileIO to <VoxelType = Float,VoxelFormat = FP32S_FORMAT>
template <int32 LOG_TILE_SIZE, int32 CTA_SIZE>
struct TileIO<Float,FP32S_FORMAT,LOG_TILE_SIZE,CTA_SIZE>
{
    typedef TileOp<Float,ADD_BLENDING,LOG_TILE_SIZE> tile_op_type;
    typedef typename tile_op_type::storage_type storage_type;
    const static int32 STORAGE_SIZE = tile_op_type::STORAGE_SIZE;

    // load tile from gmem
    __device__ static void load(
        storage_type*       storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 WORDS_PER_THREAD = (STORAGE_SIZE + CTA_SIZE-1) / CTA_SIZE;

        float* tile = (float*)(fb) + tile_id * STORAGE_SIZE;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < WORDS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < STORAGE_SIZE)
                storage[ pixel ] = tile[ pixel ];
        }
    }

    // save tile to gmem
    __device__ static void save(
        const storage_type* storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 WORDS_PER_THREAD = (STORAGE_SIZE + CTA_SIZE-1) / CTA_SIZE;

        float* tile = (float*)(fb) + tile_id * STORAGE_SIZE;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < WORDS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < STORAGE_SIZE)
                tile[ pixel ] = storage[ pixel ];
        }
    }
};
// specialization of TileIO to <VoxelType = Float,VoxelFormat = U8S_FORMAT>
template <int32 LOG_TILE_SIZE, int32 CTA_SIZE>
struct TileIO<Float,U8S_FORMAT,LOG_TILE_SIZE,CTA_SIZE>
{
    typedef TileOp<Float,ADD_BLENDING,LOG_TILE_SIZE> tile_op_type;
    typedef typename tile_op_type::storage_type storage_type;
    const static int32 STORAGE_SIZE = tile_op_type::STORAGE_SIZE;

    // load tile from gmem
    __device__ static void load(
        storage_type*       storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 WORDS_PER_THREAD = (STORAGE_SIZE + CTA_SIZE-1) / CTA_SIZE;

        uint8* tile = (uint8*)(fb) + tile_id * STORAGE_SIZE;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < WORDS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < STORAGE_SIZE)
                storage[ pixel ] = float( tile[ pixel ] ) / 256.0f;
        }
    }

    // save tile to gmem
    __device__ static void save(
        const storage_type* storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 WORDS_PER_THREAD = (STORAGE_SIZE + CTA_SIZE-1) / CTA_SIZE;

        uint8* tile = (uint8*)(fb) + tile_id * STORAGE_SIZE;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < WORDS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < STORAGE_SIZE)
                tile[ pixel ] = quantize8( storage[ pixel ] );
        }
    }
};
// specialization of TileIO to <VoxelType = Float2,VoxelFormat = FP32S_FORMAT>
template <int32 LOG_TILE_SIZE, int32 CTA_SIZE>
struct TileIO<Float2,FP32S_FORMAT,LOG_TILE_SIZE,CTA_SIZE>
{
    typedef TileOp<Float2,ADD_BLENDING,LOG_TILE_SIZE> tile_op_type;
    typedef typename tile_op_type::storage_type storage_type;
    const static int32 STORAGE_SIZE = tile_op_type::STORAGE_SIZE;

    // load tile from gmem
    __device__ static void load(
        storage_type*       storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 WORDS_PER_THREAD = (STORAGE_SIZE + CTA_SIZE-1) / CTA_SIZE;

        float* tile = (float*)(fb) + tile_id * STORAGE_SIZE;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < WORDS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < STORAGE_SIZE)
                storage[ pixel ] = tile[ pixel ];
        }
    }

    // save tile to gmem
    __device__ static void save(
        const storage_type* storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 WORDS_PER_THREAD = (STORAGE_SIZE + CTA_SIZE-1) / CTA_SIZE;

        float* tile = (float*)(fb) + tile_id * STORAGE_SIZE;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < WORDS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < STORAGE_SIZE)
                tile[ pixel ] = storage[ pixel ];
        }
    }
};
// specialization of TileIO to <VoxelType = Float2,VoxelFormat = FP32V_FORMAT>
template <int32 LOG_TILE_SIZE, int32 CTA_SIZE>
struct TileIO<Float2,FP32V_FORMAT,LOG_TILE_SIZE,CTA_SIZE>
{
    typedef TileOp<Float2,ADD_BLENDING,LOG_TILE_SIZE> tile_op_type;
    typedef typename tile_op_type::storage_type storage_type;
    const static int32 STORAGE_SIZE = tile_op_type::STORAGE_SIZE;
    const static int32 T = 1 << LOG_TILE_SIZE;

    // load tile from gmem
    __device__ static void load(
        storage_type*       storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 TILE_PIXELS = T*T*T;
        const int32 PIXELS_PER_THREAD = (TILE_PIXELS + CTA_SIZE-1) / CTA_SIZE;

        float2* tile = (float2*)(fb) + tile_id * TILE_PIXELS;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < PIXELS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < TILE_PIXELS)
            {
                const float2 val = tile[ pixel ];
                storage[ pixel + TILE_PIXELS*0 ] = val.x;
                storage[ pixel + TILE_PIXELS*1 ] = val.y;
            }
        }
    }

    // save tile to gmem
    __device__ static void save(
        const storage_type* storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 TILE_PIXELS = T*T*T;
        const int32 PIXELS_PER_THREAD = (TILE_PIXELS + CTA_SIZE-1) / CTA_SIZE;

        float2* tile = (float2*)(fb) + tile_id * TILE_PIXELS;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < PIXELS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < TILE_PIXELS)
            {
                tile[ pixel ] =
                    make_float2(
                        storage[ pixel + TILE_PIXELS*0 ],
                        storage[ pixel + TILE_PIXELS*1 ] );
            }
        }
    }
};
// specialization of TileIO to <VoxelType = Float2,VoxelFormat = U8S_FORMAT>
template <int32 LOG_TILE_SIZE, int32 CTA_SIZE>
struct TileIO<Float2,U8S_FORMAT,LOG_TILE_SIZE,CTA_SIZE>
{
    typedef TileOp<Float2,ADD_BLENDING,LOG_TILE_SIZE> tile_op_type;
    typedef typename tile_op_type::storage_type storage_type;
    const static int32 STORAGE_SIZE = tile_op_type::STORAGE_SIZE;

    // load tile from gmem
    __device__ static void load(
        storage_type*       storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 WORDS_PER_THREAD = (STORAGE_SIZE + CTA_SIZE-1) / CTA_SIZE;

        uint8* tile = (uint8*)(fb) + tile_id * STORAGE_SIZE;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < WORDS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < STORAGE_SIZE)
                storage[ pixel ] = float( tile[ pixel ] ) / 256.0f;
        }
    }

    // save tile to gmem
    __device__ static void save(
        const storage_type* storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 WORDS_PER_THREAD = (STORAGE_SIZE + CTA_SIZE-1) / CTA_SIZE;

        uint8* tile = (uint8*)(fb) + tile_id * STORAGE_SIZE;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < WORDS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < STORAGE_SIZE)
                tile[ pixel ] = quantize8( storage[ pixel ] );
        }
    }
};
// specialization of TileIO to <VoxelType = Float2,VoxelFormat = U8V_FORMAT>
template <int32 LOG_TILE_SIZE, int32 CTA_SIZE>
struct TileIO<Float2,U8V_FORMAT,LOG_TILE_SIZE,CTA_SIZE>
{
    typedef TileOp<Float2,ADD_BLENDING,LOG_TILE_SIZE> tile_op_type;
    typedef typename tile_op_type::storage_type storage_type;
    const static int32 STORAGE_SIZE = tile_op_type::STORAGE_SIZE;
    const static int32 T = 1 << LOG_TILE_SIZE;

    // load tile from gmem
    __device__ static void load(
        storage_type*       storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 TILE_PIXELS = T*T*T;
        const int32 PIXELS_PER_THREAD = (TILE_PIXELS + CTA_SIZE-1) / CTA_SIZE;

        uint16* tile = (uint16*)(fb) + tile_id * TILE_PIXELS;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < PIXELS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < TILE_PIXELS)
            {
                const uint16 val = tile[ pixel ];
                storage[ pixel + TILE_PIXELS*0 ] = float( (val >>  0) & 0x00000000FF ) / 256.0f;
                storage[ pixel + TILE_PIXELS*1 ] = float( (val >>  8) & 0x00000000FF ) / 256.0f;
            }
        }
    }

    // save tile to gmem
    __device__ static void save(
        const storage_type* storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 TILE_PIXELS = T*T*T;
        const int32 PIXELS_PER_THREAD = (TILE_PIXELS + CTA_SIZE-1) / CTA_SIZE;

        uint16* tile = (uint16*)(fb) + tile_id * TILE_PIXELS;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < PIXELS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < TILE_PIXELS)
            {
                tile[ pixel ] =
                    quantize8(  storage[ pixel + TILE_PIXELS*0 ] )       |
                    quantize8(  storage[ pixel + TILE_PIXELS*1 ] ) << 8;
            }
        }
    }
};
// specialization of TileIO to <VoxelType = Float3,VoxelFormat = FP32S_FORMAT>
template <int32 LOG_TILE_SIZE, int32 CTA_SIZE>
struct TileIO<Float3,FP32S_FORMAT,LOG_TILE_SIZE,CTA_SIZE>
{
    typedef TileOp<Float3,ADD_BLENDING,LOG_TILE_SIZE> tile_op_type;
    typedef typename tile_op_type::storage_type storage_type;
    const static int32 STORAGE_SIZE = tile_op_type::STORAGE_SIZE;

    // load tile from gmem
    __device__ static void load(
        storage_type*       storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 WORDS_PER_THREAD = (STORAGE_SIZE + CTA_SIZE-1) / CTA_SIZE;

        float* tile = (float*)(fb) + tile_id * STORAGE_SIZE;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < WORDS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < STORAGE_SIZE)
                storage[ pixel ] = tile[ pixel ];
        }
    }

    // save tile to gmem
    __device__ static void save(
        const storage_type* storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 WORDS_PER_THREAD = (STORAGE_SIZE + CTA_SIZE-1) / CTA_SIZE;

        float* tile = (float*)(fb) + tile_id * STORAGE_SIZE;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < WORDS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < STORAGE_SIZE)
                tile[ pixel ] = storage[ pixel ];
        }
    }
};
// specialization of TileIO to <VoxelType = Float3,VoxelFormat = FP32V_FORMAT>
template <int32 LOG_TILE_SIZE, int32 CTA_SIZE>
struct TileIO<Float3,FP32V_FORMAT,LOG_TILE_SIZE,CTA_SIZE>
{
    typedef TileOp<Float3,ADD_BLENDING,LOG_TILE_SIZE> tile_op_type;
    typedef typename tile_op_type::storage_type storage_type;
    const static int32 STORAGE_SIZE = tile_op_type::STORAGE_SIZE;
    const static int32 T = 1 << LOG_TILE_SIZE;

    // load tile from gmem
    __device__ static void load(
        storage_type*       storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 TILE_PIXELS = T*T*T;
        const int32 PIXELS_PER_THREAD = (TILE_PIXELS + CTA_SIZE-1) / CTA_SIZE;

        float3* tile = (float3*)(fb) + tile_id * TILE_PIXELS;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < PIXELS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < TILE_PIXELS)
            {
                const float3 val = tile[ pixel ];
                storage[ pixel + TILE_PIXELS*0 ] = val.x;
                storage[ pixel + TILE_PIXELS*1 ] = val.y;
                storage[ pixel + TILE_PIXELS*2 ] = val.z;
            }
        }
    }

    // save tile to gmem
    __device__ static void save(
        const storage_type* storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 TILE_PIXELS = T*T*T;
        const int32 PIXELS_PER_THREAD = (TILE_PIXELS + CTA_SIZE-1) / CTA_SIZE;

        float3* tile = (float3*)(fb) + tile_id * TILE_PIXELS;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < PIXELS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < TILE_PIXELS)
            {
                tile[ pixel ] =
                    make_float3(
                        storage[ pixel + TILE_PIXELS*0 ],
                        storage[ pixel + TILE_PIXELS*1 ],
                        storage[ pixel + TILE_PIXELS*2 ] );
            }
        }
    }
};
// specialization of TileIO to <VoxelType = Float3,VoxelFormat = U8S_FORMAT>
template <int32 LOG_TILE_SIZE, int32 CTA_SIZE>
struct TileIO<Float3,U8S_FORMAT,LOG_TILE_SIZE,CTA_SIZE>
{
    typedef TileOp<Float3,ADD_BLENDING,LOG_TILE_SIZE> tile_op_type;
    typedef typename tile_op_type::storage_type storage_type;
    const static int32 STORAGE_SIZE = tile_op_type::STORAGE_SIZE;

    // load tile from gmem
    __device__ static void load(
        storage_type*       storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 WORDS_PER_THREAD = (STORAGE_SIZE + CTA_SIZE-1) / CTA_SIZE;

        uint8* tile = (uint8*)(fb) + tile_id * STORAGE_SIZE;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < WORDS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < STORAGE_SIZE)
                storage[ pixel ] = float( tile[ pixel ] ) / 256.0f;
        }
    }

    // save tile to gmem
    __device__ static void save(
        const storage_type* storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 WORDS_PER_THREAD = (STORAGE_SIZE + CTA_SIZE-1) / CTA_SIZE;

        uint8* tile = (uint8*)(fb) + tile_id * STORAGE_SIZE;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < WORDS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < STORAGE_SIZE)
                tile[ pixel ] = quantize8( storage[ pixel ] );
        }
    }
};
// specialization of TileIO to <VoxelType = Float3,VoxelFormat = U8V_FORMAT>
template <int32 LOG_TILE_SIZE, int32 CTA_SIZE>
struct TileIO<Float3,U8V_FORMAT,LOG_TILE_SIZE,CTA_SIZE>
{
    typedef TileOp<Float3,ADD_BLENDING,LOG_TILE_SIZE> tile_op_type;
    typedef typename tile_op_type::storage_type storage_type;
    const static int32 STORAGE_SIZE = tile_op_type::STORAGE_SIZE;
    const static int32 T = 1 << LOG_TILE_SIZE;

    // load tile from gmem
    __device__ static void load(
        storage_type*       storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 TILE_PIXELS = T*T*T;
        const int32 PIXELS_PER_THREAD = (TILE_PIXELS + CTA_SIZE-1) / CTA_SIZE;

        uint8* tile = (uint8*)(fb) + tile_id * TILE_PIXELS * 3;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < PIXELS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < TILE_PIXELS)
            {
                const uint8 val_x = tile[ pixel*3+0 ];
                const uint8 val_y = tile[ pixel*3+1 ];
                const uint8 val_z = tile[ pixel*3+2 ];
                storage[ pixel + TILE_PIXELS*0 ] = float( val_x ) / 256.0f;
                storage[ pixel + TILE_PIXELS*1 ] = float( val_y ) / 256.0f;
                storage[ pixel + TILE_PIXELS*2 ] = float( val_z ) / 256.0f;
            }
        }
    }
    // save tile to gmem
    __device__ static void save(
        const storage_type* storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 TILE_PIXELS = T*T*T;
        const int32 PIXELS_PER_THREAD = (TILE_PIXELS + CTA_SIZE-1) / CTA_SIZE;

        uint8* tile = (uint8*)(fb) + tile_id * TILE_PIXELS * 3;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < PIXELS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < TILE_PIXELS)
            {
                tile[ pixel*3+0 ] = quantize8( storage[ pixel + TILE_PIXELS*0 ] );
                tile[ pixel*3+1 ] = quantize8( storage[ pixel + TILE_PIXELS*1 ] );
                tile[ pixel*3+2 ] = quantize8( storage[ pixel + TILE_PIXELS*2 ] );
            }
        }
    }
};
// specialization of TileIO to <VoxelType = Float4,VoxelFormat = FP32S_FORMAT>
template <int32 LOG_TILE_SIZE, int32 CTA_SIZE>
struct TileIO<Float4,FP32S_FORMAT,LOG_TILE_SIZE,CTA_SIZE>
{
    typedef TileOp<Float4,ADD_BLENDING,LOG_TILE_SIZE> tile_op_type;
    typedef typename tile_op_type::storage_type storage_type;
    const static int32 STORAGE_SIZE = tile_op_type::STORAGE_SIZE;

    // load tile to gmem
    __device__ static void load(
        storage_type*       storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 WORDS_PER_THREAD = (STORAGE_SIZE + CTA_SIZE-1) / CTA_SIZE;

        float* tile = (float*)(fb) + tile_id * STORAGE_SIZE;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < WORDS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < STORAGE_SIZE)
                storage[ pixel ] = tile[ pixel ];
        }
    }
    // save tile to gmem
    template <int32 CTA_SZ>
    __device__ static void save(
        const storage_type* storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 WORDS_PER_THREAD = (STORAGE_SIZE + CTA_SZ-1) / CTA_SZ;

        float* tile = (float*)(fb) + tile_id * STORAGE_SIZE;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < WORDS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SZ + threadIdx.x;
            if (pixel < STORAGE_SIZE)
                tile[ pixel ] = storage[ pixel ];
        }
    }
};
// specialization of TileIO to <VoxelType = Float4,VoxelFormat = FP32V_FORMAT>
template <int32 LOG_TILE_SIZE, int32 CTA_SIZE>
struct TileIO<Float4,FP32V_FORMAT,LOG_TILE_SIZE,CTA_SIZE>
{
    typedef TileOp<Float4,ADD_BLENDING,LOG_TILE_SIZE> tile_op_type;
    typedef typename tile_op_type::storage_type storage_type;
    const static int32 STORAGE_SIZE = tile_op_type::STORAGE_SIZE;
    const static int32 T = 1 << LOG_TILE_SIZE;

    // load tile from gmem
    __device__ static void load(
        storage_type*       storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 TILE_PIXELS = T*T*T;
        const int32 PIXELS_PER_THREAD = (TILE_PIXELS + CTA_SIZE-1) / CTA_SIZE;

        float4* tile = (float4*)(fb) + tile_id * TILE_PIXELS;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < PIXELS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < TILE_PIXELS)
            {
                const float4 val = tile[ pixel ];
                storage[ pixel + TILE_PIXELS*0 ] = val.x;
                storage[ pixel + TILE_PIXELS*1 ] = val.y;
                storage[ pixel + TILE_PIXELS*2 ] = val.z;
                storage[ pixel + TILE_PIXELS*3 ] = val.w;
            }
        }
    }

    // save tile to gmem
    __device__ static void save(
        const storage_type* storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 TILE_PIXELS = T*T*T;
        const int32 PIXELS_PER_THREAD = (TILE_PIXELS + CTA_SIZE-1) / CTA_SIZE;

        float4* tile = (float4*)(fb) + tile_id * TILE_PIXELS;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < PIXELS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < TILE_PIXELS)
            {
                tile[ pixel ] =
                    make_float4(
                        storage[ pixel + TILE_PIXELS*0 ],
                        storage[ pixel + TILE_PIXELS*1 ],
                        storage[ pixel + TILE_PIXELS*2 ],
                        storage[ pixel + TILE_PIXELS*3 ] );
            }
        }
    }
};
// specialization of TileIO to <VoxelType = Float4,VoxelFormat = U8S_FORMAT>
template <int32 LOG_TILE_SIZE, int32 CTA_SIZE>
struct TileIO<Float4,U8S_FORMAT,LOG_TILE_SIZE,CTA_SIZE>
{
    typedef TileOp<Float4,ADD_BLENDING,LOG_TILE_SIZE> tile_op_type;
    typedef typename tile_op_type::storage_type storage_type;
    const static int32 STORAGE_SIZE = tile_op_type::STORAGE_SIZE;

    // load tile from gmem
    __device__ static void load(
        storage_type*       storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 WORDS_PER_THREAD = (STORAGE_SIZE + CTA_SIZE-1) / CTA_SIZE;

        uint8* tile = (uint8*)(fb) + tile_id * STORAGE_SIZE;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < WORDS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < STORAGE_SIZE)
                storage[ pixel ] = float( tile[ pixel ] ) / 256.0f;
        }
    }

    // save tile to gmem
    __device__ static void save(
        const storage_type* storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 WORDS_PER_THREAD = (STORAGE_SIZE + CTA_SIZE-1) / CTA_SIZE;

        uint8* tile = (uint8*)(fb) + tile_id * STORAGE_SIZE;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < WORDS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < STORAGE_SIZE)
                tile[ pixel ] = quantize8( storage[ pixel ] );
        }
    }
};
// specialization of TileIO to <VoxelType = Float4,VoxelFormat = U8V_FORMAT>
template <int32 LOG_TILE_SIZE, int32 CTA_SIZE>
struct TileIO<Float4,U8V_FORMAT,LOG_TILE_SIZE,CTA_SIZE>
{
    typedef TileOp<Float4,ADD_BLENDING,LOG_TILE_SIZE> tile_op_type;
    typedef typename tile_op_type::storage_type storage_type;
    const static int32 STORAGE_SIZE = tile_op_type::STORAGE_SIZE;
    const static int32 T = 1 << LOG_TILE_SIZE;

    // load tile from gmem
    __device__ static void load(
        storage_type*       storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 TILE_PIXELS = T*T*T;
        const int32 PIXELS_PER_THREAD = (TILE_PIXELS + CTA_SIZE-1) / CTA_SIZE;

        uint32* tile = (uint32*)(fb) + tile_id * TILE_PIXELS;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < PIXELS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < TILE_PIXELS)
            {
                const uint32 val = tile[ pixel ];
                storage[ pixel + TILE_PIXELS*0 ] = float( (val >>  0) & 0x00000000FF ) / 256.0f;
                storage[ pixel + TILE_PIXELS*1 ] = float( (val >>  8) & 0x00000000FF ) / 256.0f;
                storage[ pixel + TILE_PIXELS*2 ] = float( (val >> 16) & 0x00000000FF ) / 256.0f;
                storage[ pixel + TILE_PIXELS*3 ] = float( (val >> 24) & 0x00000000FF ) / 256.0f;
            }
        }
    }
    // save tile to gmem
    __device__ static void save(
        const storage_type* storage,
        void*               fb,
        int32               tile_id)
    {
        const int32 TILE_PIXELS = T*T*T;
        const int32 PIXELS_PER_THREAD = (TILE_PIXELS + CTA_SIZE-1) / CTA_SIZE;

        uint32* tile = (uint32*)(fb) + tile_id * TILE_PIXELS;

        // loop through the number of words assigned to each thread
        for (int32 p = 0; p < PIXELS_PER_THREAD; ++p)
        {
            const int32 pixel = p * CTA_SIZE + threadIdx.x;
            if (pixel < TILE_PIXELS)
            {
                tile[ pixel ] =
                    quantize8(  storage[ pixel + TILE_PIXELS*0 ] )       |
                    quantize8(  storage[ pixel + TILE_PIXELS*1 ] ) << 8  |
                    quantize8(  storage[ pixel + TILE_PIXELS*2 ] ) << 16 |
                    quantize8(  storage[ pixel + TILE_PIXELS*3 ] ) << 24;
            }
        }
    }
};

///
/// This kernel blends all virtual tiles corresponding to each geometrical tile in the final output grid.
/// Each geometrical tile is assigned to a single CTA, which loads all its virtual tiles one at a time
/// and blends them in.
///
template <int32 CTA_SIZE, int32 log_TILE_SIZE, int32 VoxelType, int32 VoxelFormat, int32 BlendingMode>
__global__ void tile_blender(
    const int32     N,
    const int32     log_N,
    const int32     M,
    const int32     log_M,
    uint32*         batch_counter,
    const int32*    tile_starts,
    const int32*    tile_ends,
    void*           tile_buffer,
    void*           fb)
{
    typedef TileOp<VoxelType,BlendingMode,log_TILE_SIZE>                  tile_op_type;
    typedef TileIO<VoxelType,VoxelFormat,log_TILE_SIZE,CTA_SIZE> tile_io_type;
    typedef typename tile_op_type::storage_type storage_type;

    const int32 WORDS_PER_THREAD = (tile_op_type::STORAGE_SIZE + CTA_SIZE-1) / CTA_SIZE;
    const int32 STORAGE_SIZE     = tile_op_type::STORAGE_SIZE;

    volatile __shared__ uint32 sm_broadcast[1];

    // shared memory tile
    __shared__ storage_type sm_tile[STORAGE_SIZE];

    for (;;)
    {
        __syncthreads(); // block before switching tile

        // fetch anoter tile to work on
        if (threadIdx.x == 0)
            *sm_broadcast = atomicAdd( batch_counter, 1 );

        __syncthreads(); // make sure sm_broadcast is visible to everybody

        // broadcast tile index to the entire block
        const int32 tile_id = *sm_broadcast;
        if (tile_id >= M*M*M)
            break;

        const int32 tile_start = tile_starts[ tile_id ];
        const int32 tile_end   = tile_ends[ tile_id ];

        //const int32 tile_z = TILE_SIZE * (tile_id >> (log_M*2));
        //const int32 tile_y = TILE_SIZE * ((tile_id & (M*M-1)) >> log_M);
        //const int32 tile_x = TILE_SIZE * (tile_id & (M-1));

        if (tile_end == tile_start)
            return;

        // load first tile
        {
            const storage_type* tile_address = (const storage_type*)(tile_buffer) + tile_start * STORAGE_SIZE;

            for (int32 p = 0; p < WORDS_PER_THREAD; ++p)
            {
                const uint32 word_index = p * CTA_SIZE + threadIdx.x;
                if (word_index < STORAGE_SIZE)
                    sm_tile[word_index] = tile_address[word_index];
            }
        }
        __syncthreads();

        // blend all the other tiles in
        for (int32 tile_idx = tile_start + 1; tile_idx < tile_end; ++tile_idx)
        {
            const storage_type* tile_address = (const storage_type*)(tile_buffer) + tile_idx * STORAGE_SIZE;

            for (int32 p = 0; p < WORDS_PER_THREAD; ++p)
            {
                const uint32 word_index = p * CTA_SIZE + threadIdx.x;
                if (word_index < STORAGE_SIZE)
                    sm_tile[word_index] = tile_op_type::blend( sm_tile[word_index], tile_address[word_index] );
            }

            __syncthreads();
        }

        if (BlendingMode == NO_BLENDING)
        {
            // filter away the tile_op_type::clearword bits from unwritten pixels
            for (int32 p = 0; p < WORDS_PER_THREAD; ++p)
            {
                const uint32 word_index = p * CTA_SIZE + threadIdx.x;
                if (word_index < STORAGE_SIZE)
                {
                    if (__float_as_int( sm_tile[word_index] ) == __float_as_int( tile_op_type::clearword() ))
                        sm_tile[word_index] = storage_type(0);
                }
            }
            __syncthreads();
        }

        // perform final output to gmem
        // NOTE: the output FB is organized into tiles. The reason for this is that otherwise re-packing
        // the compressed binary layouts would necessitate atomics near the tile boundaries (at least for
        // tile sizes smaller than 32).
        {
            tile_io_type::save(
                &sm_tile[0],
                fb,
                tile_id );
        }
    }
}

} // namespace FR

template <int32 log_TILE_SIZE, int32 VoxelType, int32 VoxelFormat, int32 BlendingMode>
void blend_tiles(
    const int32     N,
    const int32     log_N,
    const int32*    tile_starts,
    const int32*    tile_ends,
    void*           tile_buffer,
    void*           fb,
    uint32*         batch_counter,
    FineRasterStats& stats)
{
    typedef FR::TileOp<VoxelType,BlendingMode,log_TILE_SIZE> tile_op_type;
    typedef typename tile_op_type::storage_type              storage_type;

    const int32 log_M = log_N - log_TILE_SIZE;
    const int32 M     = 1 << log_M;

    const int32 BLOCK_SIZE = 512;
    dim3 dim_block = dim3( BLOCK_SIZE, 1, 1 );
    dim3 dim_grid  = dim3( M*M*M, 1, 1 );

    //const int32 N_shared = tile_op_type::STORAGE_SIZE * sizeof( storage_type ) + 16;

    uint32 zero = 0;

    cudaMemcpy(
        batch_counter,
        &zero,
        sizeof(uint32),
        cudaMemcpyHostToDevice );

    cudaThreadSynchronize();

    // increase shared memory space
    cudaFuncSetCacheConfig( FR::tile_blender<BLOCK_SIZE,log_TILE_SIZE,VoxelType,VoxelFormat,BlendingMode>, cudaFuncCachePreferShared );

    FR::tile_blender<BLOCK_SIZE,log_TILE_SIZE,VoxelType,VoxelFormat,BlendingMode> <<<dim_grid,dim_block/*, N_shared*/>>>(
        N,
        log_N,
        M,
        log_M,
        batch_counter,
        tile_starts,
        tile_ends,
        tile_buffer,
        fb );

    cudaThreadSynchronize();
}

} // namespace voxelpipe
