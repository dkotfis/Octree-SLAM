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

/*! \file vector.h
 *   \brief Define linear-algebra vector classes
 */

#pragma once

#include <nih/basic/types.h>
#include <nih/basic/numbers.h>
#include <vector>
#include <algorithm>
#include <cmath>

namespace nih {

/*! \addtogroup linalg Linear Algebra
 */

/*! \addtogroup vectors Vectors
 *  \ingroup linalg
 *  \{
 */

///
/// Abstract linear algebra vector class, templated over type and dimension
///
template <typename T, size_t DIM>
struct Vector
{
	typedef T           value_type;
	typedef T           Field_type;
	static const size_t kDimension = DIM;

    /// empty constructor
    ///
	NIH_HOST NIH_DEVICE FORCE_INLINE Vector() {}

    /// copy constructor
    ///
    /// \param v    input vector
    NIH_HOST NIH_DEVICE FORCE_INLINE explicit Vector(const T v)
	{
		for (size_t i = 0; i < kDimension; i++)
			x[i] = v;
	}
    /// copy constructor
    ///
    /// \param v    input array
	NIH_HOST NIH_DEVICE FORCE_INLINE explicit Vector(const T* v)
	{
		for (size_t i = 0; i < kDimension; i++)
			x[i] = v[i];
	}
    /// constructor
    ///
    /// \param v    input DIM-1 vector
    /// \param w    last component
	NIH_HOST NIH_DEVICE FORCE_INLINE Vector(const Vector<T,DIM-1>& v, const T w)
	{
		for (size_t i = 0; i < DIM-1; i++)
			x[i] = v[i];

		x[DIM-1] = w;
	}

    /// assignment operator
    ///
    /// \param v    input vector
	NIH_HOST NIH_DEVICE FORCE_INLINE Vector& operator=(const Vector& v)
	{
		for (size_t i = 0; i < kDimension; i++)
			x[i] = v.x[i];
		return *this;
	}
    /// const indexing operator
    ///
    /// \param i    component index
	NIH_HOST NIH_DEVICE FORCE_INLINE const T& operator[](const size_t i) const	{ return x[i]; }
    /// indexing operator
    ///
    /// \param i    component index
	NIH_HOST NIH_DEVICE FORCE_INLINE T& operator[](const size_t i)		{ return x[i]; }

    /// vector dimension
    ///
	NIH_HOST NIH_DEVICE size_t dimension() const { return kDimension; }

    // compatibility with std::vector and sparse/dynamic vectors
    void resize(const size_t n) {}

	T x[DIM];
};


///
/// Abstract linear algebra vector class, templated over type and specialized to dimension 2
///
template <typename T>
struct Vector<T,2>
{
	typedef T           value_type;
	typedef T           Field_type;
	static const size_t kDimension = 2u;

    /// empty constructor
    ///
	NIH_HOST NIH_DEVICE FORCE_INLINE Vector() {}

    /// copy constructor
    ///
    /// \param v    input vector
    NIH_HOST NIH_DEVICE FORCE_INLINE explicit Vector(const T v)
	{
		for (size_t i = 0; i < kDimension; i++)
			x[i] = v;
	}
    /// copy constructor
    ///
    /// \param v    input array
	NIH_HOST NIH_DEVICE FORCE_INLINE explicit Vector(const T* v)
	{
		for (size_t i = 0; i < kDimension; i++)
			x[i] = v[i];
	}
    /// component-wise constructor
    ///
    /// \param v0   component 0
    /// \param v1   component 1
	NIH_HOST NIH_DEVICE FORCE_INLINE Vector(const T v0, const T v1)
	{
		x[0] = v0;
		x[1] = v1;
	}
    /// constructor
    ///
    /// \param v    input 1d vector
    /// \param v1   second component
	NIH_HOST NIH_DEVICE FORCE_INLINE Vector(const Vector<T,1>& v, const T v1)
	{
		x[0] = v[0];
		x[1] = v1;
	}

    /// assignment operator
    ///
    /// \param v    input vector
    NIH_HOST NIH_DEVICE FORCE_INLINE Vector& operator=(const Vector& v)
	{
		for (size_t i = 0; i < kDimension; i++)
			x[i] = v.x[i];
		return *this;
	}
    /// const indexing operator
    ///
    /// \param i    component index
	NIH_HOST NIH_DEVICE FORCE_INLINE const T& operator[](const size_t i) const	{ return x[i]; }
    /// indexing operator
    ///
    /// \param i    component index
	NIH_HOST NIH_DEVICE FORCE_INLINE T& operator[](const size_t i)		{ return x[i]; }

    /// vector dimension
    ///
	NIH_HOST NIH_DEVICE size_t dimension() const { return kDimension; }

	T x[2];
};
///
/// Abstract linear algebra vector class, templated over type and specialized to dimension 3
///
template <typename T>
struct Vector<T,3>
{
	typedef T           value_type;
	typedef T           Field_type;
	static const size_t kDimension = 3u;

    /// empty constructor
    ///
	NIH_HOST NIH_DEVICE FORCE_INLINE Vector() {}

    /// copy constructor
    ///
    /// \param v    input vector
    NIH_HOST NIH_DEVICE FORCE_INLINE explicit Vector(const T v)
	{
		for (size_t i = 0; i < kDimension; i++)
			x[i] = v;
	}
    /// copy constructor
    ///
    /// \param v    input array
	NIH_HOST NIH_DEVICE FORCE_INLINE explicit Vector(const T* v)
	{
		for (size_t i = 0; i < kDimension; i++)
			x[i] = v[i];
	}
    /// component-wise constructor
    ///
    /// \param v0   component 0
    /// \param v1   component 1
    /// \param v2   component 2
    NIH_HOST NIH_DEVICE FORCE_INLINE Vector(const T v0, const T v1, const T v2)
	{
		x[0] = v0;
		x[1] = v1;
		x[2] = v2;
	}
    /// constructor
    ///
    /// \param v    input 2d vector
    /// \param v3   third component
	NIH_HOST NIH_DEVICE FORCE_INLINE Vector(const Vector<T,2>& v, const T v2)
	{
		x[0] = v[0];
		x[1] = v[1];
		x[2] = v2;
	}

    /// assignment operator
    ///
    /// \param v    input vector
    NIH_HOST NIH_DEVICE FORCE_INLINE Vector& operator=(const Vector& v)
	{
		for (size_t i = 0; i < kDimension; i++)
			x[i] = v.x[i];
		return *this;
	}
    /// const indexing operator
    ///
    /// \param i    component index
	NIH_HOST NIH_DEVICE FORCE_INLINE const T& operator[](const size_t i) const	{ return x[i]; }
    /// indexing operator
    ///
    /// \param i    component index
	NIH_HOST NIH_DEVICE FORCE_INLINE T& operator[](const size_t i)		{ return x[i]; }

    /// vector dimension
    ///
	NIH_HOST NIH_DEVICE size_t dimension() const { return kDimension; }

	T x[3];
};

///
/// Abstract linear algebra vector class, templated over type and specialized to dimension 4
///
template <typename T>
struct Vector<T,4>
{
	typedef T           value_type;
	typedef T           Field_type;
	static const size_t kDimension = 4u;

    /// empty constructor
    ///
	NIH_HOST NIH_DEVICE FORCE_INLINE Vector() {}
    /// copy constructor
    ///
    /// \param v    input vector
	NIH_HOST NIH_DEVICE FORCE_INLINE explicit Vector(const T v)
	{
		for (size_t i = 0; i < kDimension; i++)
			x[i] = v;
	}
    /// copy constructor
    ///
    /// \param v    input array
	NIH_HOST NIH_DEVICE FORCE_INLINE explicit Vector(const T* v)
	{
		for (size_t i = 0; i < kDimension; i++)
			x[i] = v[i];
	}
    /// component-wise constructor
    ///
    /// \param v0   component 0
    /// \param v1   component 1
    /// \param v2   component 2
    /// \param v3   component 3
	NIH_HOST NIH_DEVICE FORCE_INLINE Vector(const T v0, const T v1, const T v2, const T v3)
	{
		x[0] = v0;
		x[1] = v1;
		x[2] = v2;
		x[3] = v3;
	}
    /// constructor
    ///
    /// \param v    input 3d vector
    /// \param v3   fourth component
	NIH_HOST NIH_DEVICE FORCE_INLINE Vector(const Vector<T,3>& v, const T v3)
	{
		x[0] = v[0];
		x[1] = v[1];
		x[2] = v[2];
		x[3] = v3;
	}

    /// assignment operator
    ///
    /// \param v    input vector
	NIH_HOST NIH_DEVICE FORCE_INLINE Vector& operator=(const Vector& v)
	{
		for (size_t i = 0; i < kDimension; i++)
			x[i] = v.x[i];
		return *this;
	}

    /// const indexing operator
    ///
    /// \param i    component index
	NIH_HOST NIH_DEVICE FORCE_INLINE const T& operator[](const size_t i) const	{ return x[i]; }
    /// indexing operator
    ///
    /// \param i    component index
	NIH_HOST NIH_DEVICE FORCE_INLINE T& operator[](const size_t i)		{ return x[i]; }

    /// vector dimension
    ///
	NIH_HOST NIH_DEVICE size_t dimension() const { return kDimension; }

	T x[4];
};

/// 2-dimensional swizzling
///
/// \param op   input vector
template <typename T, size_t DIM>
FORCE_INLINE NIH_HOST_DEVICE Vector<T,2> xy(const Vector<T,DIM>& op);

/// 2-dimensional swizzling
///
/// \param op   input vector
template <typename T, size_t DIM>
FORCE_INLINE NIH_HOST_DEVICE Vector<T,2> yx(const Vector<T,DIM>& op);

/// 3-dimensional swizzling
///
/// \param op   input vector
template <typename T, size_t DIM>
FORCE_INLINE NIH_HOST_DEVICE Vector<T,3> xyz(const Vector<T,DIM>& op);

/// 3-dimensional swizzling
///
/// \param op   input vector
template <typename T, size_t DIM>
FORCE_INLINE NIH_HOST_DEVICE Vector<T,3> xzy(const Vector<T,DIM>& op);

/// 3-dimensional swizzling
///
/// \param op   input vector
template <typename T, size_t DIM>
FORCE_INLINE NIH_HOST_DEVICE Vector<T,3> zyx(const Vector<T,DIM>& op);

/// 3-dimensional swizzling
///
/// \param op   input vector
template <typename T, size_t DIM>
FORCE_INLINE NIH_HOST_DEVICE Vector<T,3> zxy(const Vector<T,DIM>& op);

/// 3-dimensional swizzling
///
/// \param op   input vector
template <typename T, size_t DIM>
FORCE_INLINE NIH_HOST_DEVICE Vector<T,3> yxz(const Vector<T,DIM>& op);

/// 3-dimensional swizzling
///
/// \param op   input vector
template <typename T, size_t DIM>
FORCE_INLINE NIH_HOST_DEVICE Vector<T,3> yzx(const Vector<T,DIM>& op);

/// equality predicate operator
///
/// \param op1  first vector
/// \param op2  second vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE bool operator==(const Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// inequality predicate operator
///
/// \param op1  first vector
/// \param op2  second vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE bool operator!=(const Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// multiplication operator
///
/// \param op1  first vector
/// \param op2  second vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM> operator*(const Vector<T,DIM>& op1, const T op2);

/// division operator
///
/// \param op1  first vector
/// \param op2  second vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM> operator/(const Vector<T,DIM>& op1, const T op2);

/// multiplication operator
///
/// \param op1  first vector
/// \param op2  second vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM> operator*(const T op1, const Vector<T,DIM>& op2);

/// multiplication operator
///
/// \param op1  first vector
/// \param op2  second vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM>& operator*=(Vector<T,DIM>& op1, const T op2);

/// addition operator
///
/// \param op1  first vector
/// \param op2  second vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM> operator+(const Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// subtraction operator
///
/// \param op1  first vector
/// \param op2  second vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM> operator-(const Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// multiplication operator
///
/// \param op1  first vector
/// \param op2  second vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM> operator*(const Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// division operator
///
/// \param op1  first vector
/// \param op2  second vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM> operator/(const Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// addition operator
///
/// \param op1  first vector
/// \param op2  second vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM>& operator+=(Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// subtraction operator
///
/// \param op1  first vector
/// \param op2  second vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM>& operator-=(Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// multiplication operator
///
/// \param op1  first vector
/// \param op2  second vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM>& operator*=(Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// division operator
///
/// \param op1  first vector
/// \param op2  second vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM>& operator/=(Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// division operator
///
/// \param op1  first vector
/// \param op2  second vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM>& operator/=(Vector<T,DIM>& op1, const T op2);

/// negation operator
///
/// \param op1  input vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM> operator-(const Vector<T,DIM>& op1);

/// intensity
///
/// \param v  input vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE T intensity(const Vector<T,DIM>& v);

/// intensity
///
/// \param v  input vector
template <typename T>
NIH_HOST NIH_DEVICE FORCE_INLINE T intensity(const T v);

/// dot product
///
/// \param op1  first vector
/// \param op2  second vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE T dot(const Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// cross product
///
/// \param op1  first vector
/// \param op2  second vector
template <typename T>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,3u> cross(const Vector<T,3u>& op1, const Vector<T,3u>& op2);

/// reflect a vector against a given normal
///
/// \param I    input vector
/// \param N    input normal
template <typename T>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,3> reflect(const Vector<T,3> I, const Vector<T,3> N);

/// return a vector orthogonal to a given one
///
/// \param v    input vector
template <typename T>
NIH_HOST NIH_DEVICE Vector<T,3>	orthogonal(const Vector<T,3> v);

/// Euclidean distance
///
/// \param op1  first point
/// \param op2  second point
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE T euclidean_distance(const Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// square Euclidean distance
///
/// \param op1  first point
/// \param op2  second point
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE T square_euclidean_distance(const Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// Euclidean norm
///
/// \param v    input vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE T norm(const Vector<T,DIM>& op);

/// square Euclidean norm
///
/// \param v    input vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE T sq_norm(const Vector<T,DIM>& op);

/// normalize a vector
///
/// \param v    input vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM> normalize(const Vector<T,DIM> v);

/// compute the minimum component
///
/// \param v    input vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE T min_comp(const Vector<T,DIM>& v);

/// compute the maximum component
///
/// \param v    input vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE T max_comp(const Vector<T,DIM>& v);

/// compute the minimum component
///
/// \param v    input vector
template <typename T>
NIH_HOST NIH_DEVICE FORCE_INLINE T min_comp(const Vector<T,3>& v);

/// compute the maximum component
///
/// \param v    input vector
template <typename T>
NIH_HOST NIH_DEVICE FORCE_INLINE T max_comp(const Vector<T,3>& v);

/// compute the index of the maximum element
///
/// \param v    input vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE uint32 max_element(const Vector<T,DIM>& v);

/// compute the component-wise min between two vectors
///
/// \param v1    first operand
/// \param v2    second operand
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM> min(const Vector<T,DIM>& v1, const Vector<T,DIM>& v2);

/// compute the component-wise max between two vectors
///
/// \param v1    first operand
/// \param v2    second operand
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM> max(const Vector<T,DIM>& v1, const Vector<T,DIM>& v2);

/// compute the largest dimension of a given vector
///
/// \param v    input vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE uint32 largest_dim(const Vector<T,DIM>& v);

/// return a normal facing in the opposite direction wrt the view vector
///
/// \param n        normal vector
/// \param view     view vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM> face_forward(const Vector<T,DIM>& n, const Vector<T,DIM>& view);

/// compute the modulus of a vector
///
/// \param v    input vector
/// \param m    modulus
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM> mod(const Vector<T,DIM>& v, const float m);

typedef Vector<float,2> Vector2f;
typedef Vector<float,3> Vector3f;
typedef Vector<float,4> Vector4f;
typedef Vector<double,2> Vector2d;
typedef Vector<double,3> Vector3d;
typedef Vector<double,4> Vector4d;
typedef Vector<int32,2> Vector2i;
typedef Vector<int32,3> Vector3i;
typedef Vector<int32,4> Vector4i;

template <typename T>
struct Vector_traits
{
	typedef T Field_type;
	typedef T value_type;
};
template <typename T>
struct Vector_traits<T*>
{
	typedef T Field_type;
	typedef T value_type;
};

template <typename T, size_t DIM>
struct Vector_traits< Vector<T,DIM> >
{
	typedef T Field_type;
    typedef T value_type;
};

///
/// Abstract linear algebra vector class, templated over type and with dynamic dimension
///
template <typename T>
struct Dynamic_vector
{
	typedef T           value_type;
	typedef T           Field_type;

    /// empty constructor
    ///
	NIH_HOST FORCE_INLINE Dynamic_vector() {}

    /// constructor
    ///
    /// \param dim  vector dimension
    NIH_HOST explicit Dynamic_vector(const size_t dim) : x( dim ) {}

    /// constructor
    ///
    /// \param dim  vector dimension
    /// \param v    scalar value
    NIH_HOST Dynamic_vector(const size_t dim, const T v) : x( dim, v ) {}

    /// constructor
    ///
    /// \param dim  vector dimension
    /// \param v    input array
    NIH_HOST Dynamic_vector(const size_t dim, const T* v) :
        x( dim )
	{
		for (size_t i = 0; i < dim; i++)
			x[i] = v[i];
	}
    /// copy constructor
    ///
    /// \param v    input vector
    NIH_HOST FORCE_INLINE Dynamic_vector(const Dynamic_vector& v) : x( v.x ) {}

    /// assignment operator
    ///
    /// \param v    input vector
	NIH_HOST FORCE_INLINE Dynamic_vector& operator=(const Dynamic_vector& v)
	{
        x = v.x;
		return *this;
	}

    /// const indexing operator
    ///
    /// \param i    component index
    NIH_HOST FORCE_INLINE const T& operator[](const size_t i) const   { return x[i]; }
    /// indexing operator
    ///
    /// \param i    component index
    NIH_HOST FORCE_INLINE T&       operator[](const size_t i)         { return x[i]; }

    /// vector dimension
    ///
	NIH_HOST FORCE_INLINE size_t dimension() const { return x.size(); }

    /// resize dimension
    ///
    /// \param n   new vector size
    NIH_HOST void resize(const size_t n) { x.resize(n); }

    std::vector<T> x;
};

template <typename T>
struct Vector_traits< Dynamic_vector<T> >
{
	typedef T Field_type;
    typedef T value_type;
};


template <typename T>
NIH_HOST bool operator==(const Dynamic_vector<T>& op1, const Dynamic_vector<T>& op2);

template <typename T, size_t DIM>
NIH_HOST bool operator!=(const Dynamic_vector<T>& op1, const Dynamic_vector<T>& op2);

template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T> operator*(const Dynamic_vector<T>& op1, const T op2);

template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T> operator/(const Dynamic_vector<T>& op1, const T op2);

template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T> operator*(const T op1, const Dynamic_vector<T>& op2);

template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T>& operator*=(Dynamic_vector<T>& op1, const T op2);

template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T> operator+(const Dynamic_vector<T>& op1, const Dynamic_vector<T>& op2);

template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T> operator-(const Dynamic_vector<T>& op1, const Dynamic_vector<T>& op2);
template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T> operator*(const Dynamic_vector<T>& op1, const Dynamic_vector<T>& op2);
template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T> operator/(const Dynamic_vector<T>& op1, const Dynamic_vector<T>& op2);

template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T>& operator+=(Dynamic_vector<T>& op1, const Dynamic_vector<T>& op2);

template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T>& operator-=(Dynamic_vector<T>& op1, const Dynamic_vector<T>& op2);

template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T>& operator*=(Dynamic_vector<T>& op1, const Dynamic_vector<T>& op2);

template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T>& operator/=(Dynamic_vector<T>& op1, const Dynamic_vector<T>& op2);

template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T>& operator/=(Dynamic_vector<T>& op1, const T op2);

template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T> operator-(const Dynamic_vector<T>& op1);

template <typename T>
NIH_HOST FORCE_INLINE T intensity(const Dynamic_vector<T>& v);

template <typename T>
NIH_HOST FORCE_INLINE T dot(const Dynamic_vector<T>& op1, const Dynamic_vector<T>& op2);

template <typename T>
NIH_HOST FORCE_INLINE T norm(const Dynamic_vector<T>& op);

template <typename T>
NIH_HOST FORCE_INLINE T sq_norm(const Dynamic_vector<T>& op);

template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T> normalize(const Dynamic_vector<T> v);

template <typename T>
NIH_HOST FORCE_INLINE T max_comp(const Dynamic_vector<T>& v);

/*! \}
 */

} // namespace nih

#include <nih/linalg/vector_inline.h>