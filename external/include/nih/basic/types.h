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

#pragma once

#include <assert.h>

#ifdef __CUDACC__
    #define NIH_HOST_DEVICE __host__ __device__
    //#if (__CUDA_ARCH__ > 0)
    //    #define NIH_HOST_DEVICE __device__
    //#else
    //    #define NIH_HOST_DEVICE __host__
    //#endif
    #define NIH_HOST   __host__
    #define NIH_DEVICE __device__
#else
    #define NIH_HOST_DEVICE 
    #define NIH_HOST
    #define NIH_DEVICE
#endif

#define NIH_API_CS
#define NIH_API_SS

namespace nih {

struct device_space {};
struct host_space   {};

typedef unsigned char		uint8;
typedef char				int8;
typedef unsigned short		uint16;
typedef short				int16;
typedef unsigned int		uint32;
typedef int					int32;
typedef unsigned long long	uint64;
typedef long long			int64;

//#define NIH_FORCE_INLINE __forceinline
#if defined(__GNUC__)
#define FORCE_INLINE __attribute__((always_inline))
#else
#define FORCE_INLINE __forceinline
#endif

template <typename Out, typename In>
union BinaryCast
{
    In  in;
    Out out;
};

template <typename Out, typename In>
FORCE_INLINE NIH_HOST_DEVICE Out binary_cast(const In in)
{
    BinaryCast<Out,In> inout;
    inout.in = in;
    return inout.out;
}

} // namespace nih
