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

#include <cmath>
#include <limits>
#include <nih/basic/types.h>

namespace nih {

#define M_PIf     3.141592653589793238462643383279502884197169399375105820974944592f
#define M_PI_2f   6.283185307179586f
#define M_INV_PIf 0.3183098861837907f

#define M_PI 3.141592653589793238462643383279502884197169399375105820974944592
#define M_PI_2 (2.0 * M_PI)

#if defined(__GNUC__)
#define _finite(n) finite(n)
#define _isnan(n) isnan(n)
#endif

#include <float.h>

inline bool is_finite(const double x) { return _finite(x) != 0; }
inline bool is_nan(const double x) { return _isnan(x) != 0; }
inline bool is_finite(const float x) { return _finite(x) != 0; }
inline bool is_nan(const float x) { return _isnan(x) != 0; }

/// sign function
template <typename T>
inline NIH_HOST_DEVICE T sgn(const T x) { return x > 0 ? T(1) : T(-1); }

/// round a floating point number
inline NIH_HOST_DEVICE float round(const float x)
{
	const int y = x > 0.0f ? int(x) : int(x)-1;
	return (x - float(y) > 0.5f) ? float(y)+1.0f : float(y);
}

/// minimum of two floats
inline NIH_HOST_DEVICE float min(const float a, const float b) { return a < b ? a : b; }

/// maximum of two floats
inline NIH_HOST_DEVICE float max(const float a, const float b) { return a > b ? a : b; }

/// minimum of two int32
inline NIH_HOST_DEVICE int32 min(const int32 a, const int32 b) { return a < b ? a : b; }

/// maximum of two int32
inline NIH_HOST_DEVICE int32 max(const int32 a, const int32 b) { return a > b ? a : b; }

/// minimum of two uint32
inline NIH_HOST_DEVICE uint32 min(const uint32 a, const uint32 b) { return a < b ? a : b; }

/// maximum of two uint32
inline NIH_HOST_DEVICE uint32 max(const uint32 a, const uint32 b) { return a > b ? a : b; }

/// quantize the float x in [0,1] to an integer [0,...,n[
inline NIH_HOST_DEVICE uint32 quantize(const float x, const uint32 n)
{
	return (uint32)max( min( int32( x * float(n) ), int32(n-1) ), int32(0) );
}
/// compute the floating point module of a quantity with sign
inline float NIH_HOST_DEVICE mod(const float x, const float m) { return x > 0.0f ? fmodf( x, m ) : m - fmodf( -x, m ); }

inline NIH_HOST_DEVICE uint32 log2(uint32 n)
{
    uint32 m = 0;
    while (n > 0)
    {
        n >>= 1;
        m++;
    }
    return m-1;
}

#ifdef __CUDA_ARCH__

inline NIH_DEVICE float fast_pow(const float a, const float b)
{
    return __powf(a,b);
}
inline NIH_DEVICE float fast_sin(const float x)
{
    return __sinf(x);
}
inline NIH_DEVICE float fast_cos(const float x)
{
    return __cosf(x);
}
inline NIH_DEVICE float fast_sqrt(const float x)
{
    return __fsqrt_rn(x);
}

#else

inline NIH_HOST_DEVICE float fast_pow(const float a, const float b)
{
    return ::powf(a,b);
}
inline NIH_HOST_DEVICE float fast_sin(const float x)
{
    return sinf(x);
}
inline NIH_HOST_DEVICE float fast_cos(const float x)
{
    return cosf(x);
}
inline NIH_HOST_DEVICE float fast_sqrt(const float x)
{
    return sqrtf(x);
}

#endif

#ifdef __CUDACC__
inline NIH_DEVICE uint16 float_to_half(const float x) { return __float2half_rn(x); }
inline NIH_DEVICE float  half_to_float(const uint32 h) { return __half2float(h); }
#endif

template <typename T>
struct Field_traits
{
#ifdef __CUDACC__
	NIH_HOST_DEVICE static T min() { return T(); }
    NIH_HOST_DEVICE static T max() { return T(); }
#else
	static T min()
    {
        return std::numeric_limits<T>::is_integer ?
             std::numeric_limits<T>::min() :
            -std::numeric_limits<T>::max();
    }
	static T max() { return std::numeric_limits<T>::max(); }
#endif
};

#ifdef __CUDACC__
template <>
struct Field_traits<float>
{
	NIH_HOST_DEVICE static float min() { return -float(1.0e+30f); }
    NIH_HOST_DEVICE static float max() { return  float(1.0e+30f); }
};
template <>
struct Field_traits<double>
{
	NIH_HOST_DEVICE static double min() { return -double(1.0e+30); }
    NIH_HOST_DEVICE static double max() { return  double(1.0e+30); }
};
template <>
struct Field_traits<int32>
{
	NIH_HOST_DEVICE static int32 min() { return -(1 << 30); }
    NIH_HOST_DEVICE static int32 max() { return  (1 << 30); }
};
template <>
struct Field_traits<int64>
{
	NIH_HOST_DEVICE static int64 min() { return -(int64(1) << 62); }
    NIH_HOST_DEVICE static int64 max() { return  (int64(1) << 62); }
};
template <>
struct Field_traits<uint32>
{
	NIH_HOST_DEVICE static uint32 min() { return 0; }
    NIH_HOST_DEVICE static uint32 max() { return (1u << 31u); }
};
template <>
struct Field_traits<uint64>
{
	NIH_HOST_DEVICE static uint64 min() { return 0; }
    NIH_HOST_DEVICE static uint64 max() { return (uint64(1) << 63u); }
};
#endif

} // namespace nih
